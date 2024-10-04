import Mathlib
import Mathlib.Algebra.EuclideanSpace
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Group.Permutations
import Mathlib.Algebra.Module.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.ParametricIntegral
import Mathlib.Analysis.SpecialFunctions
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Binomial
import Mathlib.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Parity
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Nat.PrimeFactorization
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Time.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Init.Algebra.Order
import Mathlib.LinearAlgebra.Determinant
import Mathlib.Probability.Basic
import Mathlib.Probability.Probability
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Statistics.NormalDistribution
import Mathlib.Tactic

namespace birdhouse_volume_difference_l261_261686

-- Definitions to capture the given conditions
def sara_width_ft : ℝ := 1
def sara_height_ft : ℝ := 2
def sara_depth_ft : ℝ := 2

def jake_width_in : ℝ := 16
def jake_height_in : ℝ := 20
def jake_depth_in : ℝ := 18

-- Convert Sara's dimensions to inches
def ft_to_in (x : ℝ) : ℝ := x * 12
def sara_width_in := ft_to_in sara_width_ft
def sara_height_in := ft_to_in sara_height_ft
def sara_depth_in := ft_to_in sara_depth_ft

-- Volume calculations
def volume (width height depth : ℝ) := width * height * depth
def sara_volume := volume sara_width_in sara_height_in sara_depth_in
def jake_volume := volume jake_width_in jake_height_in jake_depth_in

-- The theorem to prove the difference in volume
theorem birdhouse_volume_difference : sara_volume - jake_volume = 1152 := by
  -- Proof goes here
  sorry

end birdhouse_volume_difference_l261_261686


namespace projection_of_a_in_direction_of_2sqrt3b_l261_261573

open Real

def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (-4, 7)

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1^2 + v.2^2)

noncomputable def scalar_mult (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)

noncomputable def projection (a b : ℝ × ℝ) : ℝ :=
  let scalar_proj := dot_product a b / magnitude b
  in scalar_proj / magnitude b * magnitude a

theorem projection_of_a_in_direction_of_2sqrt3b :
  projection vector_a (scalar_mult (2 * sqrt 3) vector_b) = sqrt 65 / 5 :=
by
  sorry

end projection_of_a_in_direction_of_2sqrt3b_l261_261573


namespace correct_calculation_l261_261768

theorem correct_calculation : (6 + (-13)) = -7 :=
by
  sorry

end correct_calculation_l261_261768


namespace final_number_appended_is_84_l261_261855

noncomputable def arina_sequence := "7172737475767778798081"

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def is_divisible_by_12 (n : ℕ) : Prop := n % 12 = 0

-- Define adding numbers to the sequence
def append_number (seq : String) (n : ℕ) : String := seq ++ n.repr

-- Create the full sequence up to 84 and check if it's divisible by 12
def generate_full_sequence : String :=
  let base_seq := arina_sequence
  let full_seq := append_number (append_number (append_number (append_number (append_number (append_number (append_number (append_number (append_number arina_sequence 82) 83) 84))) 85) 86) 87) 88 
  full_seq

theorem final_number_appended_is_84 : (∃ seq : String, is_divisible_by_12(seq.to_nat) ∧ seq.ends_with "84") := 
by
  sorry

end final_number_appended_is_84_l261_261855


namespace sum_of_powers_zero_mod_sum_of_powers_divisible_l261_261782

theorem sum_of_powers_zero_mod (m k : ℕ) (hm_odd : m % 2 = 1) (hk_odd : k % 2 = 1) : 
  (∑ i in Finset.range m, i^k) % m = 0 := 
sorry

theorem sum_of_powers_divisible (m k : ℕ) (hm_odd : m % 2 = 1) (hk_odd : k % 2 = 1) : 
  m ∣ ∑ i in Finset.range (m - 1) + 1, i^k :=
sorry

end sum_of_powers_zero_mod_sum_of_powers_divisible_l261_261782


namespace rowing_speed_upstream_l261_261809

theorem rowing_speed_upstream (V_s V_downstream : ℝ) (V_s_eq : V_s = 28) (V_downstream_eq : V_downstream = 31) : 
  V_s - (V_downstream - V_s) = 25 := 
by
  sorry

end rowing_speed_upstream_l261_261809


namespace range_of_a_l261_261602

variable {x a : ℝ}

theorem range_of_a (h1 : x < 0) (h2 : 2 ^ x - a = 1 / (x - 1)) : 0 < a ∧ a < 2 :=
sorry

end range_of_a_l261_261602


namespace range_of_a_for_inequality_l261_261599

open Real

theorem range_of_a_for_inequality (a : ℝ) : 
  (∀ x : ℝ, ¬(a*x^2 - |x + 1| + 2*a < 0)) ↔ a ≥ (sqrt 3 + 1) / 4 := 
by
  sorry

end range_of_a_for_inequality_l261_261599


namespace sum_of_possible_b_values_for_rational_roots_l261_261141

theorem sum_of_possible_b_values_for_rational_roots :
  let Δ (b : ℕ) : ℤ := 25 - 8 * b
  in ∑ b in {b | b > 0 ∧ isSquare (Δ b) ∧ Δ b ≥ 0}, b = 5 :=
by
  sorry

end sum_of_possible_b_values_for_rational_roots_l261_261141


namespace decimal_to_binary_51_l261_261891

theorem decimal_to_binary_51 : nat.to_digits 2 51 = [1, 1, 0, 0, 1, 1] :=
sorry

end decimal_to_binary_51_l261_261891


namespace sequence_not_arithmetic_progression_and_fifth_term_l261_261650

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 1 then 5
  else if n = 2 then 3
  else 2 * sequence (n - 2) + sequence (n - 1)

theorem sequence_not_arithmetic_progression_and_fifth_term :
  (∃ n, sequence (n + 2) ≠ 2 * sequence n + sequence (n + 1)) ∧ sequence 5 = 45 :=
by
  -- Proof skipped; marked as sorry
  sorry

end sequence_not_arithmetic_progression_and_fifth_term_l261_261650


namespace speed_difference_is_28_l261_261985

def distance_henry : ℝ := 8
def distance_alice : ℝ := 10

def time_henry_minutes : ℝ := 40
def time_alice_minutes : ℝ := 15

def time_henry_hours : ℝ := time_henry_minutes / 60
def time_alice_hours : ℝ := time_alice_minutes / 60

def speed_henry : ℝ := distance_henry / time_henry_hours
def speed_alice : ℝ := distance_alice / time_alice_hours

def speed_difference : ℝ := speed_alice - speed_henry

theorem speed_difference_is_28 : speed_difference = 28 := by
  sorry

end speed_difference_is_28_l261_261985


namespace probability_not_pair_l261_261514

theorem probability_not_pair : let n := Nat.choose 6 2 in
                              let m := Nat.choose 6 2 - Nat.choose 3 1 in
                              (m : ℚ) / n = 4 / 5 :=
by
  sorry

end probability_not_pair_l261_261514


namespace modulus_of_complex_l261_261529

-- Some necessary imports for complex numbers and proofs in Lean
open Complex

theorem modulus_of_complex (x y : ℝ) (h : (1 + I) * x = 1 + y * I) : abs (x + y * I) = Real.sqrt 2 :=
by
  sorry

end modulus_of_complex_l261_261529


namespace profit_function_l261_261406

def cost_per_unit : ℝ := 8

def daily_sales_quantity (x : ℝ) : ℝ := -x + 30

def profit_per_unit (x : ℝ) : ℝ := x - cost_per_unit

def total_profit (x : ℝ) : ℝ := (profit_per_unit x) * (daily_sales_quantity x)

theorem profit_function (x : ℝ) : total_profit x = -x^2 + 38*x - 240 :=
  sorry

end profit_function_l261_261406


namespace area_triangle_FOH_l261_261008

variables (EF GH h : ℝ) (O : Type) [metric_space O] [has_inner O] [has_norm O] 
  [has_metric O] [linear_ordered_field ℝ] [square_integral_domain ℝ]

-- Assign given conditions
def trapezoid_EFGH := (EF = 24) ∧ (GH = 36) ∧ (EG ⟂ FH) ∧ (area_trapezoid_efgh = 360)

-- Prove the area of triangle FOH
theorem area_triangle_FOH (h : ℝ) (EF GH : ℝ) (area_trapezoid_efgh : ℝ) := 
  (EF = 24) ∧ (GH = 36) ∧ (area_trapezoid_efgh = 360) ∧ (EG ⟂ FH) → 
  area_triangle_FOH = 108 := 
by 
  sorry

end area_triangle_FOH_l261_261008


namespace solve_for_r_l261_261079

theorem solve_for_r (r : ℝ) : 8 = 2^(6 * r - 1) → r = 2 / 3 :=
by
  sorry

end solve_for_r_l261_261079


namespace functional_equation_identity_l261_261914

def f : ℝ → ℝ := sorry

theorem functional_equation_identity (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x * f y + y) = f (f (x * y)) + y) : 
  ∀ y : ℝ, f y = y :=
sorry

end functional_equation_identity_l261_261914


namespace bonnets_per_orphanage_l261_261665

theorem bonnets_per_orphanage :
  let monday_bonnets := 10
  let tuesday_wednesday_bonnets := 2 * monday_bonnets
  let thursday_bonnets := monday_bonnets + 5
  let friday_bonnets := thursday_bonnets - 5
  let total_bonnets := monday_bonnets + tuesday_wednesday_bonnets + thursday_bonnets + friday_bonnets
  let orphanages := 5
  total_bonnets / orphanages = 11 :=
by
  let monday_bonnets := 10
  let tuesday_wednesday_bonnets := 2 * monday_bonnets
  let thursday_bonnets := monday_bonnets + 5
  let friday_bonnets := thursday_bonnets - 5
  let total_bonnets := monday_bonnets + tuesday_wednesday_bonnets + thursday_bonnets + friday_bonnets
  let orphanages := 5
  sorry

end bonnets_per_orphanage_l261_261665


namespace minimum_at_x_eq_2_l261_261210

def q (x : ℝ) : ℝ := (x - 5)^2 + (x + 1)^2 - 6

theorem minimum_at_x_eq_2 : ∀ x : ℝ, q(x) ≥ q(2) := 
by 
  sorry

end minimum_at_x_eq_2_l261_261210


namespace find_const_c_l261_261431

def a (n : ℕ) : ℝ := ∫ x in (0 : ℝ)..(1 : ℝ), x^3 * (1 - x)^n

theorem find_const_c : 
  ∑' n, (n : ℝ) + (5 : ℝ) * (a n - a (n + 1)) = (1 / 3 : ℝ) := 
sorry

end find_const_c_l261_261431


namespace proof_problem_l261_261528

variable (γ θ α : ℝ)
variable (x y : ℝ)

def condition1 := x = γ * Real.sin ((θ - α) / 2)
def condition2 := y = γ * Real.sin ((θ + α) / 2)

theorem proof_problem
  (h1 : condition1 γ θ α x)
  (h2 : condition2 γ θ α y)
  : x^2 - 2*x*y*Real.cos α + y^2 = γ^2 * (Real.sin α)^2 :=
by
  sorry

end proof_problem_l261_261528


namespace find_y_l261_261624

theorem find_y (DEG EFG y : ℝ) 
  (h1 : DEG = 150)
  (h2 : EFG = 40)
  (h3 : DEG = EFG + y) :
  y = 110 :=
by
  sorry

end find_y_l261_261624


namespace circumscribed_circles_equal_l261_261165

theorem circumscribed_circles_equal 
  {A B C D H Z E T : Type*}
  [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited H] 
  [inhabited Z] [inhabited E] [inhabited T]
  (h_triangle_ABC : triangle A B C)
  (h_AB_lt_BC_lt_AC : AB < BC ∧ BC < AC)
  (h_A_on_circle_c : circle (C) (ABC))
  (h_circle_A_AB : circle (A) (AB))
  (h_D_on_BC : intersects (h_circle_A_AB) (BC) D)
  (h_H_on_circle_c : intersects (h_circle_A_AB) (h_A_on_circle_c) H)
  (h_circle_A_AC : circle (A) (AC))
  (h_Z_on_BC : intersects (h_circle_A_AC) (BC) Z)
  (h_E_on_circle_c : intersects (h_circle_A_AC) (h_A_on_circle_c) E)
  (h_T_intersect : ∃ T, lines_intersect T ZH ED) :
  let circle_TDZ := circumscribed_circle T D Z in
  let circle_TEH := circumscribed_circle T E H in
  circumradius circle_TDZ = circumradius circle_TEH :=
sorry

end circumscribed_circles_equal_l261_261165


namespace arithmetic_sequence_sum_l261_261362

def sum_of_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_sum (a₁ d : ℕ)
  (h₁ : a₁ + (a₁ + 6 * d) + (a₁ + 13 * d) + (a₁ + 17 * d) = 120) :
  sum_of_arithmetic_sequence a₁ d 19 = 570 :=
by
  sorry

end arithmetic_sequence_sum_l261_261362


namespace smallest_solution_x4_minus_50x2_plus_625_eq_0_l261_261755

theorem smallest_solution_x4_minus_50x2_plus_625_eq_0 : ∃ x : ℝ, x^4 - 50 * x^2 + 625 = 0 ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y := 
sorry

end smallest_solution_x4_minus_50x2_plus_625_eq_0_l261_261755


namespace technicians_count_l261_261704

noncomputable def total_salary := 8000 * 21
noncomputable def average_salary_all := 8000
noncomputable def average_salary_technicians := 12000
noncomputable def average_salary_rest := 6000
noncomputable def total_workers := 21

theorem technicians_count :
  ∃ (T R : ℕ),
  T + R = total_workers ∧
  average_salary_technicians * T + average_salary_rest * R = total_salary ∧
  T = 7 :=
by
  sorry

end technicians_count_l261_261704


namespace last_appended_number_is_84_l261_261839

theorem last_appended_number_is_84 : 
  ∃ N : ℕ, 
    let s := "7172737475767778798081" ++ (String.intercalate "" (List.map toString [82, 83, 84])) in
    (N = 84) ∧ (s.toNat % 12 = 0) :=
by
  sorry

end last_appended_number_is_84_l261_261839


namespace product_of_real_roots_of_equation_l261_261907

theorem product_of_real_roots_of_equation : 
  ∀ x : ℝ, (x^4 + (x - 4)^4 = 32) → x = 2 :=
sorry

end product_of_real_roots_of_equation_l261_261907


namespace arrange_PERCEPTION_l261_261905

theorem arrange_PERCEPTION :
  ∀ n k1 k2 k3 : ℕ, n = 10 → k1 = 2 → k2 = 2 → k3 = 2 →
  (nat.factorial n) / (nat.factorial k1 * nat.factorial k2 * nat.factorial k3) = 453600 := 
by
  intros n k1 k2 k3 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end arrange_PERCEPTION_l261_261905


namespace factorization_correct_l261_261913

-- Define the input expression
def expr (x y : ℝ) : ℝ := 2 * x^3 - 18 * x * y^2

-- Define the factorized form
def factorized_expr (x y : ℝ) : ℝ := 2 * x * (x + 3*y) * (x - 3*y)

-- Prove that the original expression is equal to the factorized form
theorem factorization_correct (x y : ℝ) : expr x y = factorized_expr x y := 
by sorry

end factorization_correct_l261_261913


namespace math_scores_population_l261_261001

/-- 
   Suppose there are 50,000 students who took the high school entrance exam.
   The education department randomly selected 2,000 students' math scores 
   for statistical analysis. Prove that the math scores of the 50,000 students 
   are the population.
-/
theorem math_scores_population (students : ℕ) (selected : ℕ) 
    (students_eq : students = 50000) (selected_eq : selected = 2000) : 
    true :=
by
  sorry

end math_scores_population_l261_261001


namespace mu_value_l261_261931

open ProbabilityTheory

variables (μ σ : ℝ) (ξ : ℝ → ℝ)

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = f x

noncomputable def P (A : set ℝ) : ℝ := sorry -- Assuming the probability function

axiom normal_dist (ξ : ℝ) : Prop := sorry -- Normal distribution axiom

theorem mu_value 
  (h₁ : normal_dist ξ) 
  (h₂ : ∀ x, is_even_function (λ x, P {y | x ≤ y ∧ y ≤ x+3})) : 
  μ = 3 / 2 := 
sorry

end mu_value_l261_261931


namespace BC_intersections_are_vertices_l261_261457

-- Define the context of the problem

structure Triangle :=
  (A B C H D M E O : Point)
  (BC : Line)
  (AH : Altitude A BC H)
  (AD : AngleBisector A B C D)
  (AM : Median A B C M)
  (BC_perpendicular_M : Perpendicular BC M)
  (AE_perpendicular_bisector : PerpendicularBisector A E O)
  (BC_perpendicular_M_O : Perpendicular BC M O)

-- Constructor property ensuring existence of critical points in triangle construction
noncomputable def Triangle_construction : Triangle :=
  { A := ...,
    B := ...,
    C := ...,
    H := ...,
    D := ...,
    M := ...,
    E := ...,
    O := ...,
    BC := ...,
    AH := sorry,  -- Proof that AH is an altitude
    AD := sorry,  -- Proof that AD is an angle bisector
    AM := sorry,  -- Proof that AM is a median
    BC_perpendicular_M := sorry,  -- Proof that BC is perpendicular at M
    AE_perpendicular_bisector := sorry,  -- Proof that AE is the perpendicular bisector
    BC_perpendicular_M_O := sorry  -- Proof that BC is perpendicular at M passing through O
  }

-- Prove that B and C are the intersections of line BC with circle centered at O with radius OA
theorem BC_intersections_are_vertices :
  let (B_intersection, C_intersection) := find_intersections BC (circle_centered_at O (radius OA))
  in B_intersection = B ∧ C_intersection = C :=
by sorry

end BC_intersections_are_vertices_l261_261457


namespace find_line_eqn_l261_261542

-- Definitions and conditions based on the problem statement
def ellipse (x y : ℝ) : Prop := (x^2 / 36) + (y^2 / 9) = 1
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
def line (x y : ℝ) (a b c : ℝ) : Prop := a * x + b * y + c = 0

-- Proof statement
theorem find_line_eqn (h_midpoint : midpoint (x1, y1) (x2, y2) = (4, 2))
  (h_ellipse1 : ellipse x1 y1) (h_ellipse2 : ellipse x2 y2) :
  ∃ a b c, a = 1 ∧ b = 2 ∧ c = -8 ∧ (∀ x y, line x y a b c ↔ x + 2 * y - 8 = 0) :=
sorry

end find_line_eqn_l261_261542


namespace range_of_a_l261_261558
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -x^3 + 1 + a
noncomputable def g (x : ℝ) : ℝ := 3 * Real.log x

theorem range_of_a (h : ∀ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f x a = -g x) : 
  0 ≤ a ∧ a ≤ Real.exp 3 - 4 := 
sorry

end range_of_a_l261_261558


namespace vector_cross_product_coordinates_l261_261020

variables (a1 a2 a3 b1 b2 b3 : ℝ)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1, a.2.2 * b.1 - a.1 * b.2.2, a.1 * b.2.1 - a.2.1 * b.1)

theorem vector_cross_product_coordinates :
  cross_product (a1, a2, a3) (b1, b2, b3) = 
    (a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1) :=
by
sorry

end vector_cross_product_coordinates_l261_261020


namespace value_of_p_at_1_5_l261_261102

variables (p : ℝ → ℝ)

-- Condition from the problem
axiom point_on_graph : p 1.5 = 4

-- Theorem stating that p(1.5) = 4
theorem value_of_p_at_1_5 : p 1.5 = 4 :=
by
  exact point_on_graph

end value_of_p_at_1_5_l261_261102


namespace ticket_costs_l261_261019

theorem ticket_costs (ticket_price : ℕ) (number_of_tickets : ℕ) : ticket_price = 44 ∧ number_of_tickets = 7 → ticket_price * number_of_tickets = 308 :=
by
  intros h
  cases h
  sorry

end ticket_costs_l261_261019


namespace find_y_l261_261038

theorem find_y (y : ℕ) : (8000 * 6000 = 480 * 10 ^ y) → y = 5 :=
by
  intro h
  sorry

end find_y_l261_261038


namespace normal_symmetric_l261_261942

noncomputable def normal_pdf (μ σ : ℝ) (x : ℝ) : ℝ :=
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-((x - μ) ^ 2) / (2 * σ ^ 2))

def P {μ σ : ℝ} (X : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, X x dx

theorem normal_symmetric (μ σ c : ℝ) (h :  P (λ x => normal_pdf μ σ x) (c + 2) ∞ = 
  P (λ x => normal_pdf μ σ x) (-∞) (c - 2)) (hμ : μ = 5) (hσ : σ = 9) : c = 5 :=
by
  sorry

end normal_symmetric_l261_261942


namespace ada_original_seat_l261_261696

theorem ada_original_seat {positions : Fin 6 → Fin 6} 
  (Bea Ceci Dee Edie Fred Ada: Fin 6)
  (h1: Ada = 0)
  (h2: positions (Bea + 1) = Bea)
  (h3: positions (Ceci - 2) = Ceci)
  (h4: positions Dee = Edie ∧ positions Edie = Dee)
  (h5: positions Fred = Fred) :
  Ada = 1 → Bea = 1 → Ceci = 3 → Dee = 4 → Edie = 5 → Fred = 6 → Ada = 1 :=
by
  intros
  sorry

end ada_original_seat_l261_261696


namespace number_of_solutions_l261_261110

open Real

noncomputable def system_solutions :=
  {p : ℝ × ℝ × ℝ × ℝ | ∃ (x y z w : ℝ),
    p = (x, y, z, w) ∧
    (x = w + z + z * w * x) ∧
    (y = z + x + z * x * y) ∧
    (z = x + y + x * y * z) ∧
    (w = y + z + y * z * w)}

theorem number_of_solutions : Fintype.card system_solutions = 5 :=
  sorry

end number_of_solutions_l261_261110


namespace number_of_true_propositions_l261_261461

variable {R : Type*} [LinearOrderedField R]

def proposition1 : Prop :=
∀ x : R, x^2 - x + (1 / 4) ≥ 0

def proposition2 : Prop :=
∃ x : R, x > 0 ∧ Real.log x + (1 / Real.log x) ≤ 2

def proposition3 (a b c : R) : Prop :=
(¬ ∀ a b c : R, a > b ↔ a * c^2 > b * c^2)

def proposition4 : Prop :=
∀ x : R, (3^x - 3^(-x)) = -(3^(-x) - 3^x)

theorem number_of_true_propositions : 
proposition1 ∧ proposition2 ∧ proposition4 ∧ proposition3 = false ↔ 3 = 3 := 
by
  sorry

end number_of_true_propositions_l261_261461


namespace partition_six_into_three_parts_l261_261578

theorem partition_six_into_three_parts :
  ∃ (P : ℕ → ℕ → Prop), (P 6 3 ∧ ∑ _ in (finset.range 7).pmap (λ n _, {n}.to_finset) _) = 5 :=
by sorry

end partition_six_into_three_parts_l261_261578


namespace volume_of_box_l261_261340

variable (width length height : ℝ)
variable (Volume : ℝ)

-- Given conditions
def w : ℝ := 9
def l : ℝ := 4
def h : ℝ := 7

-- The statement to prove
theorem volume_of_box : Volume = l * w * h := by
  sorry

end volume_of_box_l261_261340


namespace sin_sum_arcsin_arctan_l261_261475

theorem sin_sum_arcsin_arctan :
  let a := Real.arcsin (4 / 5)
  let b := Real.arctan (1 / 2)
  Real.sin (a + b) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end sin_sum_arcsin_arctan_l261_261475


namespace geometric_sequence_sum_condition_l261_261627

theorem geometric_sequence_sum_condition
  (a_1 r : ℝ) 
  (S₄ : ℝ := a_1 * (1 + r + r^2 + r^3)) 
  (S₈ : ℝ := S₄ + a_1 * (r^4 + r^5 + r^6 + r^7)) 
  (h₁ : S₄ = 1) 
  (h₂ : S₈ = 3) :
  a_1 * r^16 * (1 + r + r^2 + r^3) = 8 := 
sorry

end geometric_sequence_sum_condition_l261_261627


namespace correct_population_l261_261003

variable (P : ℕ) (S : ℕ)
variable (math_scores : ℕ → Type)

-- Assume P is the total number of students who took the exam.
-- Let math_scores(P) represent the math scores of P students.

def population_data (P : ℕ) : Prop := 
  P = 50000

def sample_data (S : ℕ) : Prop :=
  S = 2000

theorem correct_population (P : ℕ) (S : ℕ) (math_scores : ℕ → Type)
  (hP : population_data P) (hS : sample_data S) : 
  math_scores P = math_scores 50000 :=
by {
  sorry
}

end correct_population_l261_261003


namespace circle_tangent_intersects_bisects_angle_l261_261044

noncomputable def circle_eq (x y c : ℝ) : Prop :=
  (x - 2)^2 + (y - c)^2 = (3 / 2)^2

theorem circle_tangent_intersects (x y c : ℝ)
  (ht : circle_eq (2 : ℝ) 0 c)
  (ha : circle_eq (0 : ℝ) y c)
  (hab : abs (y - (2 * c - y)) = 3) :
  circle_eq x y c :=
sorry

theorem bisects_angle (B : ℝ × ℝ) (P Q : ℝ × ℝ)
  (hP : (P.1^2 / 8) + (P.2^2 / 4) = 1)
  (hQ : (Q.1^2 / 8) + (Q.2^2 / 4) = 1)
  (hB : B.1 = 0 ∧ B.2 = (2 * 5 / 2 - B.2) - 3 / 2) :
  (∃ A : ℝ × ℝ, (A.1 = 0 ∧ abs (A.2 - B.2) = 3) ∧ ∀ ⦃P Q : ℝ × ℝ⦄,
    angle A P Q = angle A Q P) :=
sorry

end circle_tangent_intersects_bisects_angle_l261_261044


namespace min_AP_BP_l261_261535

noncomputable theory

variable (A B C D P M : ℝ)

def midpoint (x y : ℝ) : ℝ := (x + y) / 2

def is_midpoint (x y m : ℝ) : Prop := m = midpoint x y 

def distance (x y : ℝ) : ℝ := abs (x - y)

def is_regular_tetrahedron (a b c d : ℝ) : Prop :=
  distance a b = 1 ∧
  distance a c = 1 ∧
  distance a d = 1 ∧
  distance b c = 1 ∧
  distance b d = 1 ∧
  distance c d = 1

def on_segment (d m p : ℝ) : Prop := p ≥ m ∧ p ≤ d

theorem min_AP_BP (A B C D M P : ℝ) 
  (hTetra : is_regular_tetrahedron A B C D)
  (hMid : is_midpoint A C M)
  (hOnSeg : on_segment D M P) :
  (distance A P + distance B P) = sqrt (1 + sqrt 6 / 3) :=
sorry

end min_AP_BP_l261_261535


namespace two_f_eq_six_over_two_plus_x_l261_261583

-- Given a function f such that f(x+1) = 3 / (3 + x) for x > 0
def f (x : ℝ) := 3 / (3 + (x - 1))

-- Theorem to prove
theorem two_f_eq_six_over_two_plus_x (x : ℝ) (hx : 0 < x) : 
  2 * f x = 6 / (2 + x) :=
sorry

end two_f_eq_six_over_two_plus_x_l261_261583


namespace prime_between_30_and_40_with_remainder_7_l261_261345

theorem prime_between_30_and_40_with_remainder_7 (n : ℕ) 
  (h1 : Nat.Prime n) 
  (h2 : 30 < n) 
  (h3 : n < 40) 
  (h4 : n % 12 = 7) : 
  n = 31 := 
sorry

end prime_between_30_and_40_with_remainder_7_l261_261345


namespace evaluate_expression_l261_261122

theorem evaluate_expression :
  floor (-5.77) + ceil (-3.26) + floor (15.93) + ceil (32.10) = 39 := by
  sorry

end evaluate_expression_l261_261122


namespace coeff_x2_expansion_l261_261130

theorem coeff_x2_expansion (n : ℕ) : 
  (∑ i in (range n).erase 0, (2^(i-1)) * ∑ j in range i, 2^(j-1)) = 
  (2^n - 1) * (2^n - 2) / 6 := 
sorry

end coeff_x2_expansion_l261_261130


namespace different_sums_l261_261801

def coin := ℕ -- Using ℕ to represent the value of each coin in cents.

def is_coin (v : coin) : Prop :=
  v = 1 ∨ v = 5 ∨ v = 10 ∨ v = 50

def coin_combinations := 
  { (1, 1), (1, 5), (1, 10), (1, 50), 
    (5, 5), (5, 10), (5, 50), (10, 50) }

def coin_sum (c1 c2 : coin) : coin := c1 + c2

theorem different_sums : 
  ∃ S : set coin, S = {coin_sum c1 c2 | (c1, c2) ∈ coin_combinations} ∧ S.card = 8 :=
by
  sorry

end different_sums_l261_261801


namespace correct_expansion_l261_261030

variables {x y : ℝ}

theorem correct_expansion : 
  (-x + y)^2 = x^2 - 2 * x * y + y^2 := sorry

end correct_expansion_l261_261030


namespace right_triangle_area_l261_261228

-- Definitions corresponding to the conditions
def base : ℝ := 30
def height : ℝ := 34

-- Statement asserting the area calculation
theorem right_triangle_area (A : ℝ) : A = (1 / 2) * base * height → A = 510 :=
by
  sorry

end right_triangle_area_l261_261228


namespace cos_eq_neg_cos_36_l261_261485

noncomputable def cos_14pi_over_5_eq_neg_cos_36_deg : Prop :=
  cos (14 * Real.pi / 5) = -cos (36 * Real.pi / 180)

theorem cos_eq_neg_cos_36 {θ : ℝ} (h : θ = 14 * Real.pi / 5) : 
  cos θ = -cos (36 * Real.pi / 180) :=
by
  rw h
  exact cos_14pi_over_5_eq_neg_cos_36_deg
  sorry

end cos_eq_neg_cos_36_l261_261485


namespace constant_term_expansion_l261_261706

theorem constant_term_expansion : 
  let expr := (x^2 - (1 / (sqrt 5 * x^3)))
  (expr ^ 5).coeff 0 = 2 :=
by
  sorry

end constant_term_expansion_l261_261706


namespace circle_center_l261_261799

noncomputable def center_of_circle (a b : ℚ) : Prop :=
  let p1 : ℚ × ℚ := (1, 0)
  let p2 : ℚ × ℚ := (2, 4)
  let tangentToParabola := y_intercept * y_intercept = x_intercept * x_intercept - 4 * x_intercept + 4
  let slopeCondition := (b - 4) = - (1 / 4) * (a - 2)
  let distanceCondition := (a - 2) * (a - 2) + (b - 4) * (b - 4) = (a - 1) * (a - 1) + b * b
  let tangentToXAxis := b * b = (a - 1) * (a - 1)
  a = 178 / 15 ∧ b = 53 / 15

theorem circle_center : center_of_circle (178 / 15) (53 / 15) :=
  sorry

end circle_center_l261_261799


namespace highest_numbered_street_on_gretzky_street_l261_261037

theorem highest_numbered_street_on_gretzky_street 
  (gretzky_length_km : ℝ) 
  (distance_between_streets_m : ℝ) 
  (start_point_streets_numbered : ℕ) 
  (total_streets : ℕ) 
  (numbered_streets : ℕ) :
  gretzky_length_km = 5.6 → 
  distance_between_streets_m = 350 → 
  start_point_streets_numbered = 1 → 
  total_streets = (5600 / 350).to_nat → 
  numbered_streets = total_streets - 1 → 
  numbered_streets = 15 :=
by
  intros hl dsm spn ts ns hl_eq dsm_eq spn_eq ts_calc ns_calc
  rw [hl_eq, dsm_eq] at ts_calc
  simp at ts_calc
  rw [ts_calc] at ns_calc
  exact ns_calc

end highest_numbered_street_on_gretzky_street_l261_261037


namespace find_m_l261_261725

theorem find_m (x₁ x₂ y₁ y₂ : ℝ) (m : ℝ) 
  (h_parabola_A : y₁ = 2 * x₁^2) 
  (h_parabola_B : y₂ = 2 * x₂^2) 
  (h_symmetry : y₂ - y₁ = 2 * (x₂^2 - x₁^2)) 
  (h_product : x₁ * x₂ = -1/2) 
  (h_midpoint : (y₂ + y₁) / 2 = (x₂ + x₁) / 2 + m) :
  m = 3 / 2 :=
by
  sorry

end find_m_l261_261725


namespace period_ends_at_5_pm_l261_261927

open Time

def rain_duration : TimeSpan := Time.of_seconds (2 * 3600) -- 2 hours
def no_rain_duration : TimeSpan := Time.of_seconds (6 * 3600) -- 6 hours
def start_time : Time := Time.mk 9 0 0 -- 9:00 am

def end_time := start_time + rain_duration + no_rain_duration

theorem period_ends_at_5_pm : end_time = Time.mk 17 0 0 := begin
  sorry
end

end period_ends_at_5_pm_l261_261927


namespace f_neg_l261_261154

-- Define f as a function that works differently based on the sign of x.
def f (x : ℝ) : ℝ := if x > 0 then 2 * x * (1 - x) else 2 * x * (1 + x)

-- The hypothesis is that f(x) is an odd function.
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f(x)

-- Given conditions in Lean.
variables {f : ℝ → ℝ}
  (h_odd : is_odd_function f)
  (h_pos : ∀ x : ℝ, x > 0 → f(x) = 2 * x * (1 - x))

-- The theorem statement.
theorem f_neg (x : ℝ) (hx_neg : x < 0) : f(x) = 2 * x * (1 + x) := sorry

end f_neg_l261_261154


namespace positive_integers_with_solution_l261_261113

noncomputable def f (q : ℚ) : ℚ := sorry

theorem positive_integers_with_solution :
  (∀ p : ℚ, p.prime → f p = 1) →
  (∀ a b : ℚ, 0 < a → 0 < b → f (a * b) = a * f b + b * f a) →
  { n : ℕ | ∃ c : ℚ, 0 < c ∧ n * f c = c } = { n | n = 1 ∨ ∃ (primes : finset ℕ), (∀ p ∈ primes, nat.prime p) ∧ primes.prod = n } :=
by
  intros,
  sorry

end positive_integers_with_solution_l261_261113


namespace math_scores_population_l261_261002

/-- 
   Suppose there are 50,000 students who took the high school entrance exam.
   The education department randomly selected 2,000 students' math scores 
   for statistical analysis. Prove that the math scores of the 50,000 students 
   are the population.
-/
theorem math_scores_population (students : ℕ) (selected : ℕ) 
    (students_eq : students = 50000) (selected_eq : selected = 2000) : 
    true :=
by
  sorry

end math_scores_population_l261_261002


namespace probability_heart_first_10_second_double_deck_l261_261011

theorem probability_heart_first_10_second_double_deck :
  (let heart_cards := 24
       ten_cards := 8
       total_cards := 104
       first_heart_not_10 := (heart_cards - 2) / total_cards
       first_10_heart := 2 / total_cards
       second_10_any_suit := ten_cards / (total_cards - 1)
       second_10_not_first_10_heart := (ten_cards - 2) / (total_cards - 1)
       case1 := first_heart_not_10 * second_10_any_suit
       case2 := first_10_heart * second_10_not_first_10_heart
   in case1 + case2 = 47 / 2678) := sorry

end probability_heart_first_10_second_double_deck_l261_261011


namespace rooks_on_checkerboard_l261_261318

theorem rooks_on_checkerboard (n : ℕ) (board_size : ℕ) (even_rooks : ℕ) (odd_rooks : ℕ)
  (checkerboard : matrix (fin board_size) (fin board_size) bool)
  (coloring : (i j : fin board_size) → bool) :
  board_size = 9 → n = 9 → even_rooks = 4 → odd_rooks = 5 →
  coloring = λ i j, (i.val + j.val) % 2 = 0 → 
  (finset.univ.filter (λ x : fin board_size × fin board_size, coloring x.1 x.2)).card = even_rooks^2 + odd_rooks^2 →
  ∑ (perm : equiv.perm (fin even_rooks)), 1 * ∑ (qerm : equiv.perm (fin odd_rooks)), 1 = 2880 :=
by 
  intros _ _ _ _ _ _ _ _ _ _;
  sorry

end rooks_on_checkerboard_l261_261318


namespace determine_functions_l261_261894

theorem determine_functions (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f(x - f(y)) = f(x + y) + f(y)) → (f = λ x, 0) ∨ (f = λ x, -2 * x) :=
begin
  intro h,
  sorry -- This is where the proof would be.
end

end determine_functions_l261_261894


namespace complex_modulus_sqrt5_l261_261936

theorem complex_modulus_sqrt5 (a : ℝ) (h : |complex.mk 1 a| = real.sqrt 5) :
  a = 2 ∨ a = -2 := 
sorry

end complex_modulus_sqrt5_l261_261936


namespace no_nat_triplet_square_l261_261909

theorem no_nat_triplet_square (m n k : ℕ) : ¬ (∃ a b c : ℕ, m^2 + n + k = a^2 ∧ n^2 + k + m = b^2 ∧ k^2 + m + n = c^2) :=
by sorry

end no_nat_triplet_square_l261_261909


namespace convex_quadrilateral_proof_l261_261231

noncomputable def is_perpendicular {P Q R : Type} [euclidean_space P] (a : P) (b : P) : Prop :=
line[P] a ⊥ line[P] b

noncomputable def midpoint (p q : ℝ × ℝ) : ℝ × ℝ :=
((p.1 + q.1) / 2, (p.2 + q.2) / 2)

noncomputable def incenter (a b c : ℝ × ℝ) : ℝ × ℝ :=
-- Implementation for incenter (irrelevant for the statement)
sorry 

theorem convex_quadrilateral_proof (A B C D E I M : ℝ × ℝ)
  (h_convex : convex_quadrilateral A B C D)
  (h_AC_EQ_BD_EQ_AB : A.distance B = A.distance C ∧ A.distance C = B.distance D ∧ A.distance D = A.distance B)
  (h_AC_perpendicular_BD : is_perpendicular A C B D)
  (h_E_foot : foot_perpendicular E A C B D)
  (h_I_incenter: I = incenter A E B)
  (h_M_midpoint: M = midpoint A B) :
  is_perpendicular M I C D ∧ M.distance I = 1 / 2 * C.distance D :=
by
  sorry

end convex_quadrilateral_proof_l261_261231


namespace balanced_2020x2020_grid_l261_261881

def cell := ℤ
def color := cell

def grid (n : ℕ) := array (array color n) n

def num_black {n : ℕ} (g : grid n) (i : ℕ) : ℕ :=
  vector.sum (vector.map (λ x, if x = -1 then 1 else 0) (g.read i).to_list)

def num_white {n : ℕ} (g : grid n) (i : ℕ) : ℕ :=
  vector.sum (vector.map (λ x, if x = 1 then 1 else 0) (g.read i).to_list)

def row_balanced {n : ℕ} (g : grid n) : Prop :=
  ∀ i : ℕ, num_black g i = num_white g i

def col_balanced {n : ℕ} (g : grid n) : Prop :=
  ∀ j : ℕ, num_black g j = num_white g j

def balanced_grid {n : ℕ} (g : grid n) : Prop :=
  row_balanced g ∧ col_balanced g

theorem balanced_2020x2020_grid (g : grid 2020)
  (h : ∀ i j : ℕ, i < 2020 → j < 2020 →
    (g.read i).read j = -1 → num_white g i + num_white g j > num_black g i + num_black g j)
  (h' : ∀ i j : ℕ, i < 2020 → j < 2020 →
     (g.read i).read j = 1 → num_black g i + num_black g j > num_white g i + num_white g j) :
  balanced_grid g :=
begin
  sorry
end

end balanced_2020x2020_grid_l261_261881


namespace factors_of_48_are_multiples_of_6_l261_261986

theorem factors_of_48_are_multiples_of_6:
  let factors := [1, 2, 3, 4, 6, 8, 12, 16, 24, 48] in
  let multiples_of_6 := [6, 12, 24, 48] in
  count (λ x, x ∈ multiples_of_6) factors = 4 :=
by
  -- We first state the factors of 48
  have factors_of_48 : list ℕ := [1, 2, 3, 4, 6, 8, 12, 16, 24, 48],
  -- Let multiples_of_6 be [6, 12, 24, 48]
  have multiples_6 : list ℕ := [6, 12, 24, 48],
  -- Verify the counts and other details later directly
  exact sorry

end factors_of_48_are_multiples_of_6_l261_261986


namespace smallest_number_proof_largest_number_proof_l261_261503

def condition (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ 
  let s := (n / 100) + (n / 10 % 10) + (n % 10) in
  let s₁ := ((n - 75) / 100) + ((n - 75) / 10 % 10) + ((n - 75) % 10) in
  s = 3 * s₁

def smallest_three_digit_number : ℕ := 189
def largest_three_digit_number : ℕ := 675

theorem smallest_number_proof : ∀ n, condition n → n = smallest_three_digit_number ∨ n = largest_three_digit_number :=
by 
  sorry

theorem largest_number_proof : ∀ n, condition n → n = smallest_three_digit_number ∨ n = largest_three_digit_number :=
by 
  sorry

end smallest_number_proof_largest_number_proof_l261_261503


namespace rice_weight_between_9_8_and_10_2_l261_261405

noncomputable def rice_weight_probability : ℝ :=
let μ := 10
let σ := 0.1
let Φ : ℝ → ℝ := normalCDF ⟨μ, σ⟩
Φ 2

theorem rice_weight_between_9_8_and_10_2 :
  rice_weight_probability = 0.9544 :=
sorry

end rice_weight_between_9_8_and_10_2_l261_261405


namespace babysitting_earnings_is_31_l261_261778

-- Define the conditions based on the given problem
variables (net_profit : ℕ) (lemonade_gross_revenue : ℕ) (operating_costs : ℕ)

-- Given conditions
def conditions : Prop := 
  net_profit = 44 ∧
  lemonade_gross_revenue = 47 ∧
  operating_costs = 34

-- Define the total gross revenue from all activities
def total_gross_revenue (net_profit : ℕ) (operating_costs : ℕ) : ℕ :=
  net_profit + operating_costs

-- Define the earnings from babysitting
def earnings_from_babysitting (total_gross_revenue : ℕ) (lemonade_gross_revenue : ℕ) : ℕ :=
  total_gross_revenue - lemonade_gross_revenue

-- Prove the earnings from babysitting is 31
theorem babysitting_earnings_is_31 : 
  conditions net_profit lemonade_gross_revenue operating_costs →
  earnings_from_babysitting (total_gross_revenue net_profit operating_costs) lemonade_gross_revenue = 31 :=
by
  intro h
  cases h with h_net_profit h_rest
  cases h_rest with h_lemonade_gross h_operating_costs
  rw [h_net_profit, h_lemonade_gross, h_operating_costs]
  unfold total_gross_revenue earnings_from_babysitting
  simp
  sorry

end babysitting_earnings_is_31_l261_261778


namespace volume_difference_l261_261691

def sara_dimensions : (ℤ × ℤ × ℤ) := (1 * 12, 2 * 12, 2 * 12) -- dimensions in inches
def jake_dimensions : (ℤ × ℤ × ℤ) := (16, 20, 18) -- dimensions already in inches

def volume (dims : (ℤ × ℤ × ℤ)) : ℤ :=
  dims.1 * dims.2 * dims.3

theorem volume_difference :
  volume sara_dimensions - volume jake_dimensions = 1152 :=
by
  sorry

end volume_difference_l261_261691


namespace part1_part2_l261_261520

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | 1 < x ∧ x < 3}
def N : Set ℝ := {x | x^2 - 6*x + 8 ≤ 0}
def H (a : ℝ) : Set ℝ := {x | abs (x - a) <= 2}

def symdiff (A B : Set ℝ) : Set ℝ := A ∩ (U \ B)

theorem part1 :
  symdiff M N = {x | 1 < x ∧ x < 2} ∧
  symdiff N M = {x | 3 ≤ x ∧ x ≤ 4} :=
by
  sorry

theorem part2 (a : ℝ) :
  symdiff (symdiff N M) (H a) =
    if a ≥ 4 ∨ a ≤ -1 then {x | 1 < x ∧ x < 2}
    else if 3 < a ∧ a < 4 then {x | 1 < x ∧ x < a - 2}
    else if -1 < a ∧ a < 0 then {x | a + 2 < x ∧ x < 2}
    else ∅ :=
by
  sorry

end part1_part2_l261_261520


namespace num_remaining_elements_l261_261732

open Nat

def isMultiple (m n : Nat) : Prop := ∃ k : Nat, n = k * m

def T : Set Nat := { n | 1 ≤ n ∧ n ≤ 100 }

def non_multiple_of_4_or_5 (n : Nat) : Prop := ¬isMultiple 4 n ∧ ¬isMultiple 5 n

theorem num_remaining_elements :
  (T.filter non_multiple_of_4_or_5).card = 60 := by
  sorry

end num_remaining_elements_l261_261732


namespace binary_digits_70_l261_261111

-- Definition of the binary representation of a decimal number
def binary_representation (n : ℕ) : list ℕ :=
  if n = 0 then [0]
  else
    (list.range (nat.find_max (λ k, 2^k ≤ n) + 1)).reverse.map
      (λ k, if n.test_bit k then 1 else 0)

-- The proposition that the binary representation has 7 digits
theorem binary_digits_70 : (binary_representation 70).length = 7 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end binary_digits_70_l261_261111


namespace find_k_l261_261143

theorem find_k : ∃ k : ℝ, (3 * k - 4) / (k + 7) = 2 / 5 ∧ k = 34 / 13 :=
by
  use 34 / 13
  sorry

end find_k_l261_261143


namespace necessary_but_not_sufficient_l261_261570

def line_l1 (a : ℝ) : set (ℝ × ℝ) := { p | p.1 + a * p.2 - 2 = 0 }
def line_l2 (a : ℝ) : set (ℝ × ℝ) := { p | (a + 1) * p.1 - a * p.2 + 1 = 0 }

def is_parallel (l1 l2 : set (ℝ × ℝ)) : Prop :=
  ∀ x1 y1 x2 y2, (l1 (x1, y1) ∧ l2 (x2, y2)) → (x1 * y2 = x2 * y1)

theorem necessary_but_not_sufficient (a : ℝ) :
  (a = -2 → is_parallel (line_l1 a) (line_l2 a)) ∧
  (is_parallel (line_l1 a) (line_l1 a) → (a = -2 ∨ a = 0)) :=
by
  sorry

end necessary_but_not_sufficient_l261_261570


namespace S_12_value_l261_261361

section

-- Define \( a_n \) as given
def a (n : ℕ) : ℚ := (2 * n + 1) / (n * (n + 1) * (n + 2))

-- Define \( S_n \) as the sum of \( a_1 \) to \( a_n \)
def S (n : ℕ) : ℚ := ∑ i in Finset.range n, a (i + 1)

-- State the theorem to prove
theorem S_12_value : S 12 = 201 / 182 := 
sorry

end

end S_12_value_l261_261361


namespace largest_multiple_of_8_less_than_neg_63_l261_261747

theorem largest_multiple_of_8_less_than_neg_63 : 
  ∃ n : ℤ, (n < -63) ∧ (∃ k : ℤ, n = 8 * k) ∧ (∀ m : ℤ, (m < -63) ∧ (∃ l : ℤ, m = 8 * l) → m ≤ n) :=
sorry

end largest_multiple_of_8_less_than_neg_63_l261_261747


namespace smallest_solution_l261_261763

theorem smallest_solution : ∃ x : ℝ, x^4 - 50 * x^2 + 625 = 0 ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y :=
by {
  use -5,
  split,
  sorry, -- here would be the proof that (-5)^4 - 50 * (-5)^2 + 625 = 0
  intros y hy,
  sorry -- here would be the proof that for any y such that y^4 - 50 * y^2 + 625 = 0, -5 ≤ y
}

end smallest_solution_l261_261763


namespace total_turns_two_hours_l261_261827

theorem total_turns_two_hours 
  (turnsA : ℕ) (timeA : ℕ) (turnsB : ℕ) (timeB : ℕ) 
  (two_hours_in_minutes : ℕ) (total_turns : ℕ) :
  turnsA = 6 → timeA = 30 → turnsB = 10 → timeB = 45 → two_hours_in_minutes = 120 → total_turns = (12 * 120) + (1600) :=
begin
  sorry
end

end total_turns_two_hours_l261_261827


namespace clear_chessboard_l261_261297

theorem clear_chessboard (board : ℕ → ℕ → ℕ) (h : ∀ i j, board i j > 0 ∧ 0 ≤ i < 8 ∧ 0 ≤ j < 8) :
  ∃ n : ℕ, ∀ i j, (iterate (λ b, remove_or_double_move b) n board) i j = 0 :=
by sorry

-- Definitions for allowed moves
def remove_row (board : ℕ → ℕ → ℕ) (r : ℕ) : ℕ → ℕ → ℕ :=
λ i j, if i = r then board i j - 1 else board i j

def double_col (board : ℕ → ℕ → ℕ) (c : ℕ) : ℕ → ℕ → ℕ :=
λ i j, if j = c then 2 * board i j else board i j

-- Helper function to apply one of the allowed moves
def remove_or_double_move (board : ℕ → ℕ → ℕ) : ℕ → ℕ → ℕ :=
λ i j, if condition_to_remove_row board i then remove_row board i i j else double_col board j i j

-- Placeholder condition for example purposes
def condition_to_remove_row (board : ℕ → ℕ → ℕ) (r : ℕ) : Prop :=
∀ j, board r j > 0 

end clear_chessboard_l261_261297


namespace hyperbola_eq_standard_l261_261549

noncomputable def hyperbola_standard_equation
  (center : ℝ × ℝ)
  (focus : ℝ × ℝ)
  (asymptote_slope : ℝ)
  (asymptote_slope_neg : ℝ)
  (c_sq : ℝ)
  (asq_plus_bsq : ℝ)
  : Prop :=
  let c := real.sqrt c_sq in
  let a := 6 in  -- these values are derived from conditions
  let b := 8 in  -- above
  center = (0, 0) ∧
  focus = (c, 0) ∧
  asymptote_slope = 4/3 ∧
  asymptote_slope_neg = -4/3 ∧
  c_sq = a^2 + b^2 ∧
  a^2 + b^2 = asq_plus_bsq ∧
  c = 10 ∧ 
  (a : ℝ) = 6 ∧ (b : ℝ) = 8 ∧
  (asq_plus_bsq = 100) →
  ∀ x y : ℝ, (x^2 / 36) - (y^2 / 64) = 1

theorem hyperbola_eq_standard (x y : ℝ) :
  hyperbola_standard_equation (0, 0) (10, 0) (4/3) (-4/3) 100 36 :=
sorry

end hyperbola_eq_standard_l261_261549


namespace cost_of_lima_beans_l261_261199

theorem cost_of_lima_beans (L : ℝ) (x : ℝ) (total_weight : ℝ) (corn_weight : ℝ) (corn_cost_per_pound : ℝ) (mixture_cost_per_pound : ℝ) (mixture_total_weight : ℝ) : 
  (x + corn_weight = total_weight) →
  (total_weight = mixture_total_weight) →
  (corn_cost_per_pound = 0.50) →
  (16 = corn_weight) →
  (mixture_cost_per_pound = 0.65) →
  (mixture_total_weight = 25.6) →
  (x = total_weight - corn_weight) →
  (25.6 * 0.65 = 16.64) →
  (16 * 0.50 = 8.00) →
  (9.6 * L + 8.00 = 16.64) →
  (L = 0.90) :=
by
  intros _ _ _ _ _ _ _ _ _ h
  assumption

#check cost_of_lima_beans -- to ensure the theorem is correct and type-checks

end cost_of_lima_beans_l261_261199


namespace tan_4125_eq_neg_sqrt3_minus2_l261_261883

noncomputable def tan_of_angle (deg : ℝ) : ℝ := Real.tan (deg * Real.pi / 180)

theorem tan_4125_eq_neg_sqrt3_minus2 :
  tan_of_angle 4125 = -(2 - Real.sqrt 3) :=
by
  have h1 : 4125 % 360 = 165 :=
    by norm_num
  have h2 : tan_of_angle 165 = -(2 - Real.sqrt 3) :=
    by sorry -- This is a known identity shown in the problem's solution
  rw [← h1, tan_of_angle] at h2
  exact h2

end tan_4125_eq_neg_sqrt3_minus2_l261_261883


namespace N_exists_l261_261919

noncomputable def matrixMul (A B : Matrix ℝ) : Matrix ℝ :=
  let m := A.rows
  let n := B.cols
  let p := A.cols
  let q := B.rows
  if p = q then
    Matrix.mul A B
  else
    panic! "Matrix dimensions do not match!"

def N := Matrix ℝ 3 3

def A : N := ![
  [-4, 6, 0],
  [6, -9, 0],
  [0, 0, 1]
]

def B : N := ![
  [2, 0, 0],
  [0, 3, 0],
  [0, 0, 2]
]

theorem N_exists : 
  ∃ N : N, matrixMul N A = B := 
    sorry

end N_exists_l261_261919


namespace volume_surface_ratio_l261_261417

-- Define the structure of the shape
structure Shape where
  center_cube : unit
  surrounding_cubes : Fin 6 -> unit
  top_cube : unit

-- Define the properties for the calculation
def volume (s : Shape) : ℕ := 8
def surface_area (s : Shape) : ℕ := 28
def ratio_volume_surface_area (s : Shape) : ℚ := volume s / surface_area s

-- Main theorem statement
theorem volume_surface_ratio (s : Shape) : ratio_volume_surface_area s = 2 / 7 := sorry

end volume_surface_ratio_l261_261417


namespace last_score_is_88_l261_261664

theorem last_score_is_88 
    (scores : List ℕ := [52, 61, 67, 72, 77, 88])
    (total : ℕ := 417)
    (average_is_integer : ∀n, 1 ≤ n ∧ n ≤ 6 → ( ∑ i in scores.take n, i) / n ∈ ℕ):
  ∃ (last_score : ℕ), last_score = 88 := 
begin
    sorry
end

end last_score_is_88_l261_261664


namespace alexander_payment_l261_261391

variable (child_ticket_cost adult_ticket_cost : ℕ)

theorem alexander_payment 
  (h1 : child_ticket_cost = 600)
  (h2 : ∃ B, 2 * child_ticket_cost + 3 * B = 3600 ∧ 3 * child_ticket_cost + 2 * B + 200):
  2 * child_ticket_cost + 3 * adult_ticket_cost = 3600 :=
by
  obtain ⟨B, hB₁, hB₂⟩ := h2
  rw [h1, hB₁]
  sorry

end alexander_payment_l261_261391


namespace parabola_directrix_correct_l261_261500

-- Define the given parabola equation
def parabola_eq (x : ℝ) : ℝ := (x^2 - 4 * x + 4) / 8

-- Define what the correct directrix equation should be
def directrix_eq : ℝ := -1 / 4

-- State the theorem to be proven
theorem parabola_directrix_correct :
  ∀ x : ℝ, ∃ y : ℝ, y = parabola_eq x → y = directrix_eq := sorry

end parabola_directrix_correct_l261_261500


namespace alexander_payment_l261_261392

variable (child_ticket_cost adult_ticket_cost : ℕ)

theorem alexander_payment 
  (h1 : child_ticket_cost = 600)
  (h2 : ∃ B, 2 * child_ticket_cost + 3 * B = 3600 ∧ 3 * child_ticket_cost + 2 * B + 200):
  2 * child_ticket_cost + 3 * adult_ticket_cost = 3600 :=
by
  obtain ⟨B, hB₁, hB₂⟩ := h2
  rw [h1, hB₁]
  sorry

end alexander_payment_l261_261392


namespace polynomial_divisibility_a_l261_261145

theorem polynomial_divisibility_a (n : ℕ) : 
  (n % 3 = 1 ∨ n % 3 = 2) ↔ (x^2 + x + 1 ∣ x^(2*n) + x^n + 1) :=
sorry

end polynomial_divisibility_a_l261_261145


namespace construct_triangle_given_midpoints_and_bisector_line_l261_261887

-- Define the midpoints as N and M and the line l
variables {A B C N M : Type} (l : Type)

-- Assume that N is the midpoint of AC
def is_midpoint_AC (N : Type) (A C : Type) : Prop := 
  ∃ N, N = (A + C) / 2

-- Assume that M is the midpoint of BC
def is_midpoint_BC (M : Type) (B C : Type) : Prop := 
  ∃ M, M = (B + C) / 2

-- Assume the bisector of angle A lies on line l
def bisector_angle_A (A : Type) (l : Type) : Prop := 
  ∃ A l, is_angle_bisector A l

-- Main theorem
theorem construct_triangle_given_midpoints_and_bisector_line 
  (N M A B C : Type) (l : Type) 
  (h1 : is_midpoint_AC N A C)
  (h2 : is_midpoint_BC M B C)
  (h3 : bisector_angle_A A l) : 
  ∃ (triangle : Type), is_triangle ABC := 
sorry

end construct_triangle_given_midpoints_and_bisector_line_l261_261887


namespace distinguishing_feature_of_selection_structure_is_decision_box_l261_261346

-- Definitions based on the conditions provided in part a)
def selection_structure_contains_decision_box : Prop := 
  ∃ cb : Type, (selection_structure cb) ∧ (¬ sequential_structure cb)

-- Theorems that need to be proven based on the equivalent problem in part c)
theorem distinguishing_feature_of_selection_structure_is_decision_box : 
  selection_structure_contains_decision_box := 
sorry -- Proof goes here

end distinguishing_feature_of_selection_structure_is_decision_box_l261_261346


namespace recliner_sales_increase_l261_261413

theorem recliner_sales_increase (P N N' : ℝ) (h₁ : N' = 1.8 * N) : 
  N' = 1.8 * N → (N' - N) / N * 100 = 80 := 
by
  intro h₁
  calc 
    (N' - N) / N * 100 = (1.8 * N - N) / N * 100 : by rw h₁
                    ... = 0.8 * N / N * 100 : by norm_num
                    ... = 0.8 * 100 : by field_simp
                    ... = 80 : by norm_num

end recliner_sales_increase_l261_261413


namespace polynomial_evaluation_l261_261601

theorem polynomial_evaluation (x y : ℝ) (h : 2 * x^2 + 3 * y + 3 = 8) : 6 * x^2 + 9 * y + 8 = 23 :=
sorry

end polynomial_evaluation_l261_261601


namespace union_of_A_and_B_l261_261211

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem union_of_A_and_B : A ∪ B = {x | 2 < x ∧ x < 10} :=
by
  sorry

end union_of_A_and_B_l261_261211


namespace equal_heights_l261_261409

-- Definitions for given conditions
def cone_volume (r h_cone : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h_cone
def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

-- Conditions
def h_cone : ℝ := 9 -- height of the cone
def r : ℝ := 1 -- arbitrary as it cancels out

-- Statement to prove
theorem equal_heights : ∀ h : ℝ, 
  cylinder_volume r h = cone_volume r h_cone → h = (1/3) * h_cone :=
by
  intro h assump
  sorry

#eval equal_heights -- Verify if the compiled statement

end equal_heights_l261_261409


namespace alexander_paid_amount_l261_261393

/-- The amount Alexander paid for the tickets is 3600 rubles. -/
theorem alexander_paid_amount :
  let A := 600 in
  let B := 800 in
  let cost_alexander := 2 * A + 3 * B in
  let cost_anna := 3 * A + 2 * B in
  cost_alexander = cost_anna + 200 → cost_alexander = 3600 :=
by
  intros A B cost_alexander cost_anna h
  unfold A B cost_alexander cost_anna at *
  have h1: 2 * 600 + 3 * 800 = 3600 := by norm_num
  rw h1
  have h2: 3 * 600 + 2 * 800 = 2800 := by norm_num
  have h_final: 3600 = 2800 + 200 := by norm_num
  rw h_final at h
  exact h

end alexander_paid_amount_l261_261393


namespace range_of_a_l261_261595

noncomputable def log_base (b x : ℝ) : ℝ := real.log x / real.log b

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < 0 → log_base (2*a) (x + 1) > 0) → 0 < a ∧ a < 1 / 2 :=
by sorry

end range_of_a_l261_261595


namespace sum_of_segments_l261_261682

-- Define the divisions and their properties
structure Rectangle (α : Type*) :=
  (EF FG EH GH EG : α)
  (EF_length FG_length EH_length GH_length EG_length : ℕ)

-- Define the lengths of the sides and the diagonal
variables {α : Type*} (r : Rectangle α)
variables (E F G H P Q : α)
variables (length_EF length_FG length_EG : ℕ)

structure Point (α : Type*) :=
  (x y : α)

structure Segment (α : Type*) :=
  (P Q : Point α)

open Point
open Segment

-- Given conditions of the problem
axiom EF_length : length_EF = 5
axiom FG_length : length_FG = 2
axiom EG_length : length_EG = nat.sqrt 29

-- Define the segment lengths
noncomputable def segment_length (k : ℕ) : α :=
√(29) * (1 - k/84)

-- Prove the final length of the segments
theorem sum_of_segments : 
  2 * ∑ k in finset.range 84, segment_length α k - √(29) = 83 * √(29) := 
begin
  sorry
end

end sum_of_segments_l261_261682


namespace frustum_volume_correct_l261_261426

noncomputable def base_length := 20 -- cm
noncomputable def base_width := 10 -- cm
noncomputable def original_altitude := 12 -- cm
noncomputable def cut_height := 6 -- cm
noncomputable def base_area := base_length * base_width -- cm^2
noncomputable def original_volume := (1 / 3 : ℚ) * base_area * original_altitude -- cm^3
noncomputable def top_area := base_area / 4 -- cm^2
noncomputable def smaller_pyramid_volume := (1 / 3 : ℚ) * top_area * cut_height -- cm^3
noncomputable def frustum_volume := original_volume - smaller_pyramid_volume -- cm^3

theorem frustum_volume_correct :
  frustum_volume = 700 :=
by
  sorry

end frustum_volume_correct_l261_261426


namespace non_zero_digits_in_decimal_l261_261203

theorem non_zero_digits_in_decimal (a b : ℕ) (h₁ : a = 84) (h₂ : b = 2^5 * 5^9) : 
  num_non_zero_digits (a / b) = 2 :=
begin
  -- Math proof in Lean would go here
  sorry
end

end non_zero_digits_in_decimal_l261_261203


namespace rooks_on_checkerboard_l261_261321

theorem rooks_on_checkerboard (n : ℕ) (board_size : ℕ) (even_rooks : ℕ) (odd_rooks : ℕ)
  (checkerboard : matrix (fin board_size) (fin board_size) bool)
  (coloring : (i j : fin board_size) → bool) :
  board_size = 9 → n = 9 → even_rooks = 4 → odd_rooks = 5 →
  coloring = λ i j, (i.val + j.val) % 2 = 0 → 
  (finset.univ.filter (λ x : fin board_size × fin board_size, coloring x.1 x.2)).card = even_rooks^2 + odd_rooks^2 →
  ∑ (perm : equiv.perm (fin even_rooks)), 1 * ∑ (qerm : equiv.perm (fin odd_rooks)), 1 = 2880 :=
by 
  intros _ _ _ _ _ _ _ _ _ _;
  sorry

end rooks_on_checkerboard_l261_261321


namespace value_of_a_l261_261560

variable (a : ℝ)

/-- The given function -/
def f (x : ℝ) : ℝ := a * Real.log x + (1/2) * a * x^2 - 2 * x

/-- The derivative of the function -/
def f_prime (x : ℝ) : ℝ := a / x + a * x - 2

/-- The function g(x) as defined in the solution -/
def g (x : ℝ) : ℝ := 2 * x / (1 + x^2)

/-- The main problem restated in Lean -/
theorem value_of_a (h : ∀ x ∈ Ioo 1 2, f_prime a x ≤ 0) : a < 1 := sorry

end value_of_a_l261_261560


namespace linear_increase_l261_261224

-- Define a linear relationship condition between x and y
def linear_relation (Δx Δy : ℝ) (m : ℝ) : Prop := Δy = m * Δx

theorem linear_increase (m : ℝ) (h : m = 6 / 4) :
  linear_relation 12 18 m :=
by {
  -- From the condition m = 6 / 4, we have m = 1.5
  have : m = 1.5 := by linarith,
  -- verifying that 18 is the result of 12 times 1.5
  simp [linear_relation, *, mul_comm] 
}

end linear_increase_l261_261224


namespace compare_sqrt_differences_l261_261932

theorem compare_sqrt_differences :
  let a := (Real.sqrt 7) - (Real.sqrt 6)
  let b := (Real.sqrt 3) - (Real.sqrt 2)
  a < b :=
by
  sorry -- Proof goes here

end compare_sqrt_differences_l261_261932


namespace rectangle_area_l261_261341

theorem rectangle_area (length : ℝ) (width_dm : ℝ) (width_m : ℝ) (h1 : length = 8) (h2 : width_dm = 50) (h3 : width_m = width_dm / 10) : 
  (length * width_m = 40) :=
by {
  sorry
}

end rectangle_area_l261_261341


namespace def_product_is_zero_l261_261456

theorem def_product_is_zero
  (d e f : ℝ)
  (h_polynomial : ∀ x, (x^3 + d * x^2 + e * x + f = 0 ↔ x = real.cos (real.pi / 5) ∨ x = real.cos (3 * real.pi / 5) ∨ x = real.cos (4 * real.pi / 5))) :
  d * e * f = 0 :=
by {
  sorry
}

end def_product_is_zero_l261_261456


namespace isosceles_right_triangle_perimeter_l261_261075

theorem isosceles_right_triangle_perimeter :
  let L1 := {p : ℝ × ℝ | p.1 = 1}
  let L2 := {p : ℝ × ℝ | p.2 = -p.1 + 1}
  let L3 := {p : ℝ × ℝ | p.2 = -p.1 ∧ p = (0, 0) ∨ p.2 = -p.1 ∧ p.1 = 1 ∨ p = (0, 0)}
  let A := (1, -1)
  let B := (1, 0)
  let C := (0, 0)
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)
  in d A B + d B C + d A C = 1 + 2 * Real.sqrt 2
:= sorry

end isosceles_right_triangle_perimeter_l261_261075


namespace correct_calculation_l261_261769

theorem correct_calculation : (6 + (-13)) = -7 :=
by
  sorry

end correct_calculation_l261_261769


namespace distance_difference_l261_261315

-- Definitions related to the problem conditions
variables (v D_AB D_BC D_AC : ℝ)

-- Conditions
axiom h1 : D_AB = v * 7
axiom h2 : D_BC = v * 5
axiom h3 : D_AC = 6
axiom h4 : D_AC = D_AB + D_BC

-- Theorem for proof problem
theorem distance_difference : D_AB - D_BC = 1 :=
by sorry

end distance_difference_l261_261315


namespace pi_times_diagonal_irrational_l261_261818

theorem pi_times_diagonal_irrational
  {m n p q : ℤ} (hn : n ≠ 0) (hq : q ≠ 0) :
  let l := (m : ℚ) / n,
      w := (p : ℚ) / q,
      d := Real.sqrt (l^2 + w^2)
  in Irrational (π * d) :=
by
  sorry

end pi_times_diagonal_irrational_l261_261818


namespace hyperbola_center_origin_opens_vertically_l261_261807

noncomputable def t_squared : ℝ :=
  let a_sq := (64 / 5 : ℝ) in
  let y := 2 in
  let x := 2 in
  let frac := (frac := y^2 / 4 - 5 * x^2 / a_sq) in
  (frac + 5 / 16 - 1) in
  frac * 4 / 16

theorem hyperbola_center_origin_opens_vertically
  (a_sq : ℝ := 64 / 5)
  (y : ℝ := 2)
  (x : ℝ := 2) : t_squared = 21 / 4 :=
by
  sorry

end hyperbola_center_origin_opens_vertically_l261_261807


namespace condition_equiv_l261_261705

theorem condition_equiv (p q : Prop) : (¬ (p ∧ q) ∧ (p ∨ q)) ↔ ((p ∨ q) ∧ (¬ p ↔ q)) :=
  sorry

end condition_equiv_l261_261705


namespace polar_graph_is_two_straight_lines_l261_261715

theorem polar_graph_is_two_straight_lines (θ ρ : ℝ) (h₁ : sin (2 * θ) = 0) (h₂ : ρ ≥ 0) :
    (ρ = 0) ∨ (θ = 0 ∨ θ = π ∨ θ = π/2 ∨ θ = 3 * π/2) :=
sorry

end polar_graph_is_two_straight_lines_l261_261715


namespace hyperbola_properties_l261_261959

-- Define hyperbola with given foci and length of real axis
def hyperbola (F1 F2 : ℝ × ℝ) (a : ℝ) :=
  ∃ b: ℝ, ∀ x y, y^2 / a^2 - x^2 / b^2 = 1

-- The given conditions for the foci and real axis length
theorem hyperbola_properties : 
  ∀ F1 F2 : ℝ × ℝ, F1 = (0, 2) → F2 = (0, -2) → (2 : ℝ),
  let a := 1 in 
  let c := 2 in 
  let e := c / a in 
  (e = 2) ∧ 
  let b_squared := c^2 - a^2 in
  ∃ Q : ℝ × ℝ, Q.1 = -sqrt(3) * Q.2 ∧ (F1.1 - Q.1) * (F2.1 - Q.1) + (F1.2 - Q.2) * (F2.2 - Q.2) = 0 → 
  let area := (|F1.1 - F2.1| * (sqrt(3) * |Q.2|)) / 2 in 
  area = 2 * sqrt(3) :=
begin
  intros F1 F2 h_F1 h_F2 _,
  have h_a : F1 = (0, 2), by assumption,
  have h_b : F2 = (0, -2), by assumption,
  let c := 2,
  let a := 1,
  let e := c / a,
  have h_e : e = 2, by simp [e, c, a],
  let b_squared := c^2 - a^2,
  use (0, 1),
  split,
  { 
    simp only [F1, (0, 2), Q = (-sqrt 3 * 1, 1) ],
    sorry -- proof would continue from here verifying that F1Q and F2Q are perpendicular and area calculation
  }
end

end hyperbola_properties_l261_261959


namespace max_non_congruent_non_similar_triangles_l261_261432

/-- Define the properties of a triangle -/
structure Triangle :=
(a b c : ℕ)
(h_a_ge_b : a ≥ b)
(h_b_ge_c : b ≥ c)
(h_sum : b + c > a)
(h_length : a < 7 ∧ b < 7 ∧ c < 7)

/-- Define the set S to contain all triangles satisfying the specified conditions -/
def S : Finset Triangle :=
  { t | t.a < 7 ∧ t.b < 7 ∧ t.c < 7 ∧ t.a ≥ t.b ∧ t.b ≥ t.c ∧ t.b + t.c > t.a}.toFinset

/-- The maximal cardinality of set S is 13, given that no two triangles are congruent or similar -/
theorem max_non_congruent_non_similar_triangles : S.card = 13 :=
  sorry

end max_non_congruent_non_similar_triangles_l261_261432


namespace find_a_and_other_root_l261_261958

theorem find_a_and_other_root (a : ℝ) (h : (2 : ℝ) ^ 2 - 3 * (2 : ℝ) + a = 0) :
  a = 2 ∧ ∃ x : ℝ, x ^ 2 - 3 * x + a = 0 ∧ x ≠ 2 ∧ x = 1 := 
by
  sorry

end find_a_and_other_root_l261_261958


namespace square_diff_correctness_l261_261382

theorem square_diff_correctness (x y : ℝ) :
  let A := (x + y) * (x - 2*y)
  let B := (x + y) * (-x + y)
  let C := (x + y) * (-x - y)
  let D := (-x + y) * (x - y)
  (∃ (a b : ℝ), B = (a + b) * (a - b)) ∧ (∀ (p q : ℝ), A ≠ (p + q) * (p - q)) ∧ (∀ (r s : ℝ), C ≠ (r + s) * (r - s)) ∧ (∀ (t u : ℝ), D ≠ (t + u) * (t - u)) :=
by
  sorry

end square_diff_correctness_l261_261382


namespace number_of_rainy_tuesdays_l261_261298

theorem number_of_rainy_tuesdays (num_mondays : ℕ) (rain_monday : ℝ) (rain_tuesday : ℝ) (extra_rain : ℝ) :
  num_mondays = 7 → rain_monday = 1.5 → rain_tuesday = 2.5 → extra_rain = 12 →
  let total_rain_monday := num_mondays * rain_monday,
      total_rain_tuesday := (total_rain_monday + extra_rain) in
      (total_rain_tuesday / rain_tuesday = 9) :=
by {
  intro h_num_mondays h_rain_monday h_rain_tuesday h_extra_rain,
  dsimp only [total_rain_monday, total_rain_tuesday],
  rw [h_num_mondays, h_rain_monday, h_rain_tuesday, h_extra_rain, ←add_assoc],
  sorry
}

end number_of_rainy_tuesdays_l261_261298


namespace choose_7_from_16_l261_261731

theorem choose_7_from_16 : (Nat.choose 16 7) = 11440 := 
by
  sorry

end choose_7_from_16_l261_261731


namespace complex_number_quadrant_l261_261156

theorem complex_number_quadrant (z : ℂ) (hz : 2 * z + conj z = 3 - I) : z.re > 0 ∧ z.im < 0 :=
sorry

end complex_number_quadrant_l261_261156


namespace problem_solution_l261_261050

theorem problem_solution:
  2019 ^ Real.log (Real.log 2019) - Real.log 2019 ^ Real.log 2019 = 0 :=
by
  sorry

end problem_solution_l261_261050


namespace square_of_third_side_l261_261600

theorem square_of_third_side (a b : ℕ) (h1 : a = 4) (h2 : b = 5) 
    (h_right_triangle : (a^2 + b^2 = c^2) ∨ (b^2 + c^2 = a^2)) : 
    (c = 9) ∨ (c = 41) :=
sorry

end square_of_third_side_l261_261600


namespace problem_statement_l261_261206

variables {a c b d : ℝ} {x y q z : ℕ}

-- Given conditions:
def condition1 (a c : ℝ) (x q : ℕ) : Prop := a^(x + 1) = c^(q + 2)
def condition2 (a c : ℝ) (y z : ℕ) : Prop := c^(y + 3) = a^(z+ 4)

-- Goal statement
theorem problem_statement (a c : ℝ) (x y q z : ℕ) (h1 : condition1 a c x q) (h2 : condition2 a c y z) :
  (q + 2) * (z + 4) = (y + 3) * (x + 1) :=
sorry

end problem_statement_l261_261206


namespace units_digit_sum_l261_261042

-- Definition of conditions
def units_digit_35 := 5
def units_digit_5_power (n : ℕ) : ℕ := 5
def digit_cycle_3 := [3, 9, 7, 1]

-- Function to get the units digit of powers of 3
def units_digit_3_power (n : ℕ) : ℕ :=
  let cycle := digit_cycle_3
  cycle[(n % cycle.length)]

-- Statement to prove
theorem units_digit_sum :
  units_digit_5_power 87 + units_digit_3_power 45 = 8 :=
by
  sorry

end units_digit_sum_l261_261042


namespace sin_2x_solution_l261_261202

theorem sin_2x_solution (x : ℝ) 
  (h : 2 * Real.sin x + Real.cos x + Real.tan x + Real.cot x + Real.sec x + Real.csc x = 9) :
  Real.sin (2 * x) = 38 - 2 * Real.sqrt 361 :=
  sorry

end sin_2x_solution_l261_261202


namespace volume_tetrahedron_375sqrt2_l261_261912

noncomputable def tetrahedronVolume (area_ABC : ℝ) (area_BCD : ℝ) (BC : ℝ) (angle_ABC_BCD : ℝ) : ℝ :=
  let h_BCD := (2 * area_BCD) / BC
  let h_D_ABD := h_BCD * Real.sin angle_ABC_BCD
  (1 / 3) * area_ABC * h_D_ABD

theorem volume_tetrahedron_375sqrt2 :
  tetrahedronVolume 150 90 12 (Real.pi / 4) = 375 * Real.sqrt 2 := by
  sorry

end volume_tetrahedron_375sqrt2_l261_261912


namespace total_cost_table_chairs_sofa_with_discount_and_tax_l261_261067

theorem total_cost_table_chairs_sofa_with_discount_and_tax:
  let cost_of_table : ℝ := 140
  let cost_of_chair : ℝ := (1/7) * cost_of_table
  let cost_of_sofa  : ℝ := 2 * cost_of_table
  let discount      : ℝ := 0.10 * cost_of_table
  let discounted_cost_of_table : ℝ := cost_of_table - discount
  let total_cost_before_tax : ℝ := discounted_cost_of_table + 4 * cost_of_chair + cost_of_sofa
  let sales_tax : ℝ := 0.07 * total_cost_before_tax
  in total_cost_before_tax + sales_tax = 520.02 :=
by
  sorry

end total_cost_table_chairs_sofa_with_discount_and_tax_l261_261067


namespace sin_sum_arcsin_arctan_l261_261479

-- Definitions matching the conditions
def a := Real.arcsin (4 / 5)
def b := Real.arctan (1 / 2)

-- Theorem stating the question and expected answer
theorem sin_sum_arcsin_arctan : 
  Real.sin (a + b) = (11 * Real.sqrt 5) / 25 := 
by 
  sorry

end sin_sum_arcsin_arctan_l261_261479


namespace souvenirs_spending_l261_261447

theorem souvenirs_spending :
  ∃ T : ℝ, T + 146 = 347 ∧ T + 347 = 548 :=
begin
  sorry
end

end souvenirs_spending_l261_261447


namespace count_parallelogram_conditions_is_fifteen_l261_261540

structure Quadrilateral (A B C D : Type) :=
  (AB_parallel_CD : Prop)
  (BC_parallel_AD : Prop)
  (AB_eq_CD : Prop)
  (BC_eq_AD : Prop)
  (angle_A_eq_angle_C : Prop)
  (angle_B_eq_angle_D : Prop)

def parallelogram_conditions_count (q : Quadrilateral ℝ) : ℕ :=
  (if q.AB_parallel_CD then 1 else 0) +
  (if q.BC_parallel_AD then 1 else 0) +
  (if q.AB_eq_CD then 1 else 0) +
  (if q.BC_eq_AD then 1 else 0) +
  (if q.angle_A_eq_angle_C then 1 else 0) +
  (if q.angle_B_eq_angle_D then 1 else 0)

theorem count_parallelogram_conditions_is_fifteen (q : Quadrilateral ℝ) :
  (parallelogram_conditions_count q).choose 2 = 15 :=
sorry

end count_parallelogram_conditions_is_fifteen_l261_261540


namespace MN_distance_l261_261645

-- Definitions for points A, B, C, D and their primes
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (2, 0, 0)
def C : ℝ × ℝ × ℝ := (2, 1, 0)
def D : ℝ × ℝ × ℝ := (0, 1, 0)

def A' : ℝ × ℝ × ℝ := (0, 0, 12)
def B' : ℝ × ℝ × ℝ := (2, 0, 10)
def C' : ℝ × ℝ × ℝ := (2, 1, 16)
def D' : ℝ × ℝ × ℝ := (0, 1, 20)

-- Midpoints M and N
def M : ℝ × ℝ × ℝ := ((A'.1 + C'.1) / 2, (A'.2 + C'.2) / 2, (A'.3 + C'.3) / 2)
def N : ℝ × ℝ × ℝ := ((B'.1 + D'.1) / 2, (B'.2 + D'.2) / 2, (B'.3 + D'.3) / 2)

-- Distance function
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2 + (p1.3 - p2.3) ^ 2)

-- Proof problem statement
theorem MN_distance : distance M N = 1 :=
by
  sorry

end MN_distance_l261_261645


namespace b_more_days_than_a_is_12_l261_261036

-- Definitions and conditions
def b_to_a_extra_days (x : ℕ) : Prop :=
  ∃ (days_b days_a : ℕ),
    (b_independent_days: days_b = 36) ∧ 
    (b_task_completion : 0.6 * days_b / days_b = 0.6) ∧
    (time_a_works : ∃ work_days: ℝ, work_days = 21.6 - 12) ∧
    (a_task_completion : ∃ days_a y: ℝ, 0.4 * y = 9.6 ∧ days_a = y ∧ y = 24) ∧
    (x = days_b - days_a)

theorem b_more_days_than_a_is_12 : b_to_a_extra_days 12 :=
sorry

end b_more_days_than_a_is_12_l261_261036


namespace parabola_directrix_l261_261493

theorem parabola_directrix (x y : ℝ) : 
  (∀ x: ℝ, y = -4 * x ^ 2 + 4) → (y = 65 / 16) := 
by 
  sorry

end parabola_directrix_l261_261493


namespace probability_of_cube_divides_product_l261_261742

open Finset

noncomputable def cube_divides_probability : ℚ :=
let S : Finset ℕ := {2, 3, 4, 6, 8, 9} in
let comb : Finset (Finset ℕ) := S.powerset.filter (λ s, s.card = 3) in
let favorable : Finset (Finset ℕ) := comb.filter (λ s, let l := s.to_list.sorted (· ≤ ·) in (l.nth_le 0 (by simp [list.sorted_nth_le])) ^ 3 ∣ (l.nth_le 1 (by simp [list.sorted_nth_le])) * (l.nth_le 2 (by simp [list.sorted_nth_le]))) in
(favorable.card : ℚ) / (comb.card : ℚ)

theorem probability_of_cube_divides_product (S : Finset ℕ) (hS : S = {2, 3, 4, 6, 8, 9}) :
  cube_divides_probability = 1 / 5 := by
  rw [cube_divides_probability, hS]
  sorry

end probability_of_cube_divides_product_l261_261742


namespace geometric_progression_sum_of_squares_is_zero_l261_261531

noncomputable def geometric_progression_sum_is_zero : Prop :=
  ∃ a r : ℕ, 
    (a > 0) ∧ 
    (r > 0) ∧ 
    (a < 200) ∧ 
    (r < 200) ∧ 
    (a * (1 + r + r^2 + r^3 + r^4) = 341) ∧ 
    (a ≠ 0) ∧ 
    (∀ n ∈ {0, 1, 2, 3, 4}, let term := a * r^n in 
      (term < 200) ∧ 
      (term^⟨2⟩ ≤ 200) → false)) 
  
theorem geometric_progression_sum_of_squares_is_zero : geometric_progression_sum_is_zero → ∑ n in {0, 1, 2, 3, 4}, let term := a * r^n in if is_square term then term else 0 = 0 := sorry

end geometric_progression_sum_of_squares_is_zero_l261_261531


namespace final_number_appended_is_84_l261_261856

noncomputable def arina_sequence := "7172737475767778798081"

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def is_divisible_by_12 (n : ℕ) : Prop := n % 12 = 0

-- Define adding numbers to the sequence
def append_number (seq : String) (n : ℕ) : String := seq ++ n.repr

-- Create the full sequence up to 84 and check if it's divisible by 12
def generate_full_sequence : String :=
  let base_seq := arina_sequence
  let full_seq := append_number (append_number (append_number (append_number (append_number (append_number (append_number (append_number (append_number arina_sequence 82) 83) 84))) 85) 86) 87) 88 
  full_seq

theorem final_number_appended_is_84 : (∃ seq : String, is_divisible_by_12(seq.to_nat) ∧ seq.ends_with "84") := 
by
  sorry

end final_number_appended_is_84_l261_261856


namespace decision_block_has_two_exits_l261_261772

-- Define the conditions based on the problem
def output_block_exits := 1
def processing_block_exits := 1
def start_end_block_exits := 0
def decision_block_exits := 2

-- The proof statement
theorem decision_block_has_two_exits :
  (output_block_exits = 1) ∧
  (processing_block_exits = 1) ∧
  (start_end_block_exits = 0) ∧
  (decision_block_exits = 2) →
  decision_block_exits = 2 :=
by
  sorry

end decision_block_has_two_exits_l261_261772


namespace problem_1_problem_2_l261_261451

-- Problem 1 Lean Statement
theorem problem_1 :
  sqrt (9 : ℝ) + sqrt (5^2 : ℝ) + cbrt (-27 : ℝ) = 5 :=
by
  sorry

-- Problem 2 Lean Statement
theorem problem_2 :
  (-3 : ℝ)^2 - abs (-1/2 : ℝ) - sqrt (9 : ℝ) = 11 / 2 :=
by
  sorry

end problem_1_problem_2_l261_261451


namespace polynomial_product_rule_polynomial_derivative_roots_polynomial_zero_in_disc_l261_261512

-- Part (a)
theorem polynomial_product_rule (p q : polynomial ℂ) :
  (polynomial.derivative (p * q) = polynomial.derivative p * q + p * polynomial.derivative q) :=
sorry

-- Part (b)
theorem polynomial_derivative_roots (p : polynomial ℂ) (roots : list ℂ)
  (h : p = polynomial.C (p.leading_coeff) * ∏ r in roots, (X - C r)) :
  (polynomial.derivative p / p = ∑ r in roots, 1 / (X - C r)) :=
sorry

-- Part (c)
theorem polynomial_zero_in_disc (p : polynomial ℂ) (roots : list ℂ)
  (h_monic : p.leading_coeff = 1)
  (h_degrees : p.nat_degree = roots.length)
  (h_modulus_1 : |roots.head| = 1) 
  (h_modulus_all : ∀ r ∈ roots.tail, |r| ≤ 1) :
  ∃ z ∈ metric.closed_ball roots.head 1, p.derivative.eval z = 0 :=
sorry

end polynomial_product_rule_polynomial_derivative_roots_polynomial_zero_in_disc_l261_261512


namespace min_initial_bags_l261_261196

theorem min_initial_bags :
  ∃ x : ℕ, (∃ y : ℕ, (y + 90 = 2 * (x - 90) ∧ x + (11 * x - 1620) / 7 = 6 * (2 * x - 270 - (11 * x - 1620) / 7))
             ∧ x = 153) :=
by { sorry }

end min_initial_bags_l261_261196


namespace ones_digit_expression_l261_261140

theorem ones_digit_expression :
  ((73 ^ 1253 * 44 ^ 987 + 47 ^ 123 / 39 ^ 654 * 86 ^ 1484 - 32 ^ 1987) % 10) = 2 := by
  sorry

end ones_digit_expression_l261_261140


namespace crayon_boxes_needed_l261_261307

theorem crayon_boxes_needed (total_crayons : ℕ) (crayons_per_box : ℕ) (h1 : total_crayons = 80) (h2 : crayons_per_box = 8) : (total_crayons / crayons_per_box) = 10 :=
by
  sorry

end crayon_boxes_needed_l261_261307


namespace evaluate_expression_l261_261658

noncomputable def a : ℕ := 2
noncomputable def b : ℕ := 1

theorem evaluate_expression : (1 / 2)^(b - a + 1) = 1 :=
by
  sorry

end evaluate_expression_l261_261658


namespace ellipse_equation_constant_product_l261_261179

def ellipse (a b x y: ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def eccentricity (c a: ℝ) : Prop :=
  c / a = Real.sqrt 3 / 2

def triangleArea (A B : ℝ × ℝ) (O : ℝ × ℝ) : ℝ :=
  (A.1 * B.2 - A.2 * B.1) / 2

def point_on_ellipse (x0 y0 a b: ℝ) (h: ellipse a b x0 y0) : Prop := h

theorem ellipse_equation_constant_product 
  (a b c: ℝ)
  (h1 : a > b) (h2 : b > 0)
  (h3 : eccentricity c a)
  (h4 : triangleArea (a, 0) (0, b) (0, 0) = 1): 
  (ellipse a b (2 : ℝ) (1 : ℝ)) ∧ (∀ x0 y0: ℝ, 
    point_on_ellipse x0 y0 a b (by sorry) →
    let A := (a, 0)
    let B := (0, b)
    let P := (x0, y0)
    let N := (sorry : ℝ × ℝ)
    let M := (sorry : ℝ × ℝ) in
    abs (A.1 - N.1) * abs (B.2 - M.2) = 4) :=
sorry

end ellipse_equation_constant_product_l261_261179


namespace mike_falls_short_l261_261288

/--
Proof Statement: Given that the passing score is 30% of the maximum marks, which is 770,
and Mike scored 212 marks, prove that Mike falls short of passing by 19 marks.
-/
theorem mike_falls_short :
  let max_marks := 770
  let passing_percentage := 0.30
  let mike_score := 212
  let passing_marks := passing_percentage * max_marks
  passing_marks - mike_score = 19 :=
by
  -- Declaration of variables
  let max_marks := 770
  let passing_percentage := 0.30
  let mike_score := 212
  let passing_marks := passing_percentage * max_marks

  -- Calculation (will be replaced by actual proof)
  have : passing_marks - mike_score = 19 := sorry
  
  exact this

end mike_falls_short_l261_261288


namespace cos_2alpha_minus_pi_over_6_l261_261659

theorem cos_2alpha_minus_pi_over_6 (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hSin : Real.sin (α + π / 6) = 3 / 5) :
  Real.cos (2 * α - π / 6) = 24 / 25 :=
sorry

end cos_2alpha_minus_pi_over_6_l261_261659


namespace range_of_a_if_p_true_range_of_a_if_p_and_q_true_l261_261524

variable {a : ℝ}
def f (x : ℝ) : ℝ := x^2 + (a - 1) * x

def p : Prop := ∀ x > 1, 2 * x + (a - 1) > 0

def q : Prop := a > 0

-- Problem (1): If p is true, find the range of the real number a.
theorem range_of_a_if_p_true (hp : p) : a > -1 := sorry

-- Problem (2): If "p and q" are both true, find the range of the real number a.
theorem range_of_a_if_p_and_q_true (hp : p) (hq : q) : a > 0 := sorry

end range_of_a_if_p_true_range_of_a_if_p_and_q_true_l261_261524


namespace solve_for_b_l261_261997

theorem solve_for_b (b : ℚ) (h : b + 2 * b / 5 = 22 / 5) : b = 22 / 7 :=
sorry

end solve_for_b_l261_261997


namespace triangle_exists_l261_261458

namespace TriangleProof

-- Definitions for variables used in problem
variables {A B C F S : Type} {varphi : ℝ}

-- Given conditions
axiom eq_side : ∀ {A B C : Type}, B ≠ A → C = B → (∃ (AB AC : ℝ), AB = AC)
axiom median_condition : ∀ {B F : Type}, B ≠ F → (∃ (med : ℝ), med = BF)
axiom angle_condition : ∀ {BF AC : Type}, ∃ (angle : ℝ), angle = varphi

-- Definition stating there a triangle exists that satisfies the conditions
def construct_triangle (A B C F S : Type) (varphi : ℝ) : Prop :=
  eq_side ∧ median_condition ∧ angle_condition

-- Theorem statement for the problem
theorem triangle_exists (A B C F S : Type) (varphi : ℝ) :
  construct_triangle A B C F S varphi :=
sorry

end TriangleProof

end triangle_exists_l261_261458


namespace find_a1_l261_261221

theorem find_a1 (a : ℕ → ℝ) (h₁ : ∀ k, k ∈ {1..2018} → (∃ x y : ℝ, (x = a k) ∧ (y = (1/4) * (a k)^2)))
                (h₂ : ∀ k, k ∈ {1..2017} → ((a k > a (k + 1)) ∧ 
                    ((a k - a (k + 1))^2 + ((1/4) * (a k)^2 - (1/4) * (a (k + 1))^2)^2 
                    = ((1/4) * (a k)^2 + (1/4) * (a (k + 1))^2)^2))
                (h₃ : a 2018 = 1 / 2018) :
                a 1 = 2 / 2019 :=
sorry

end find_a1_l261_261221


namespace rhombus_side_length_l261_261160

theorem rhombus_side_length (d1 d2 : ℝ) (hd1 : d1 = 10) (hd2 : d2 = 24) :
  let side := real.sqrt ((d1 / 2)^2 + (d2 / 2)^2)
  in side = 13 :=
by
  sorry

end rhombus_side_length_l261_261160


namespace coord_on_line_at_distance_l261_261917

def line_coord (t : ℝ) : ℝ × ℝ :=
  (-2 - Real.sqrt 2 * t, 3 + Real.sqrt 2 * t)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem coord_on_line_at_distance :
  ∃ t : ℝ, distance (line_coord t) (-2, 3) = Real.sqrt 2 ∧ (line_coord t = (-3, 4) ∨ line_coord t = (-1, 2)) :=
by
  sorry

end coord_on_line_at_distance_l261_261917


namespace probability_closer_to_6_than_0_is_0_6_l261_261422

noncomputable def probability_closer_to_6_than_0 : ℝ :=
  let total_length := 7
  let segment_length_closer_to_6 := 4
  let probability := (segment_length_closer_to_6 : ℝ) / total_length
  probability

theorem probability_closer_to_6_than_0_is_0_6 :
  probability_closer_to_6_than_0 = 0.6 := by
  sorry

end probability_closer_to_6_than_0_is_0_6_l261_261422


namespace percentage_passed_l261_261673

theorem percentage_passed 
  (total_students : ℕ)
  (not_passed : ℕ)
  (total_given : total_students = 804)
  (not_passed_given : not_passed = 201) :
  (((total_students - not_passed).toRat / total_students.toRat) * 100) = 75 := 
by
  sorry

end percentage_passed_l261_261673


namespace nine_point_centers_coincide_l261_261259

noncomputable def nine_point_center (Δ : Triangle ℝ) : Point ℝ := sorry

variable {A B C I O A' B' C' : Point ℝ}

def triangle_ABC := Triangle.mk A B C
def incenter_ABC := I
def circumcenter_ABC := O

def A'_def := refl I A O
def B'_def := refl I B O
def C'_def := refl I C O

def triangle_A'B'C' := Triangle.mk A' B' C'

theorem nine_point_centers_coincide :
  nine_point_center triangle_ABC = nine_point_center triangle_A'B'C' :=
sorry

end nine_point_centers_coincide_l261_261259


namespace pentagon_perimeter_form_l261_261418

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

def pentagon_perimeter : ℝ :=
  (distance (0,0) (2,1)) + 
  (distance (2,1) (3,3)) + 
  (distance (3,3) (1,4)) + 
  (distance (1,4) (0,3)) + 
  (distance (0,3) (0,0))

theorem pentagon_perimeter_form :
  ∃ (a b c : ℕ), pentagon_perimeter = a + b * real.sqrt 2 + c * real.sqrt 10 ∧ a + b + c = 4 := 
sorry

end pentagon_perimeter_form_l261_261418


namespace smallest_five_digit_number_divisible_by_2_3_8_9_l261_261377

theorem smallest_five_digit_number_divisible_by_2_3_8_9 : 
  ∃ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ (n % 72 = 0) ∧ ∀ m : ℕ, m ≥ 10000 → m < n → m % 72 ≠ 0 :=
begin
  use 10008,
  split,
  { exact dec_trivial }, -- 10008 ≥ 10000
  split,
  { exact dec_trivial }, -- 10008 < 100000
  split,
  { exact dec_trivial }, -- 10008 % 72 = 0
  { intros m hm1 hm2,
    exact dec_trivial } -- ∀ m : ℕ, m ≥ 10000 → m < 10008 → m % 72 ≠ 0
end

end smallest_five_digit_number_divisible_by_2_3_8_9_l261_261377


namespace number_of_defective_pens_l261_261222

-- Define the total number of pens and the probability condition
def totalPens := 10
def probNonDefective := 0.6222222222222222

-- Definition for the predicate that the number of defective pens is 2 given the conditions
theorem number_of_defective_pens (D N : ℕ) (h1 : D + N = totalPens) (h2 : ((N:ℚ) / 10) * ((N - 1) / 9) = probNonDefective) : D = 2 := 
sorry

end number_of_defective_pens_l261_261222


namespace bowl_capacity_percentage_l261_261794

theorem bowl_capacity_percentage
    (initial_half_full : ℕ)
    (added_water : ℕ)
    (total_water : ℕ)
    (full_capacity : ℕ)
    (percentage_filled : ℚ) :
    initial_half_full * 2 = full_capacity →
    initial_half_full + added_water = total_water →
    added_water = 4 →
    total_water = 14 →
    percentage_filled = (total_water * 100) / full_capacity →
    percentage_filled = 70 := 
by
    intros h1 h2 h3 h4 h5
    sorry

end bowl_capacity_percentage_l261_261794


namespace circle_radius_eq_l261_261225

theorem circle_radius_eq (r : ℝ) (AB : ℝ) (BC : ℝ) (hAB : AB = 10) (hBC : BC = 12) : r = 25 / 4 := by
  sorry

end circle_radius_eq_l261_261225


namespace mph_to_fps_l261_261797

theorem mph_to_fps (C G : ℝ) (x : ℝ) (hC : C = 60 * x) (hG : G = 40 * x) (h1 : 7 * C - 7 * G = 210) :
  x = 1.5 :=
by {
  -- Math proof here, but we insert sorry for now
  sorry
}

end mph_to_fps_l261_261797


namespace no_similar_triangle_after_cuts_l261_261813

theorem no_similar_triangle_after_cuts :
  ∀ (T : Triangle), 
    T.angles = (20, 20, 140) →
    (∀ T2, ((T2 ∈ (cut_along_angle_bisector T)) → is_bad_triangle T2)) →
    ∀ (Tn : Triangle), Tn ∈ (cuts_sequence T) → ¬(is_similar T Tn) :=
sorry

end no_similar_triangle_after_cuts_l261_261813


namespace quadratic_real_roots_l261_261926

theorem quadratic_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 + 3 * x - 1 = 0) ↔ k ≥ -9 / 4 ∧ k ≠ 0 :=
by
  sorry

end quadratic_real_roots_l261_261926


namespace sequence_a_10th_term_l261_261162

noncomputable def sequence_a : ℕ → ℚ 
| 1 := 1
| 2 := 2 / 3
| (n + 3) := 
  let p_n2 := sequence_a (n + 1).num 
  let q_n2 := sequence_a (n + 1).den
  let p_n1 := sequence_a n.num 
  let q_n1 := sequence_a n.den
  (p_n1 + p_n2) / (q_n1 + q_n2)

theorem sequence_a_10th_term :
  sequence_a 10 = 4181 / 6765 :=
sorry

end sequence_a_10th_term_l261_261162


namespace solution_set_of_quadratic_inequality_l261_261961

theorem solution_set_of_quadratic_inequality (a b : ℝ)
  (h1 : ∀ x : ℝ, x ∈ Ioo (-1 : ℝ) (1 / 2 : ℝ) → ax^2 + bx + 3 > 0)
  (h2 : a < 0) :
  ∀ x : ℝ, x ∈ Ioo (-1 : ℝ) 2 → 3x^2 + bx + a < 0 :=
sorry

end solution_set_of_quadratic_inequality_l261_261961


namespace value_of_expression_l261_261519

theorem value_of_expression (A B C D : ℝ) (h1 : A - B = 30) (h2 : C + D = 20) :
  (B + C) - (A - D) = -10 :=
by
  sorry

end value_of_expression_l261_261519


namespace max_area_regular_ngon_max_perimeter_regular_ngon_l261_261386

noncomputable def max_area_ngon (n : ℕ) (r : ℝ) : ℝ := sorry

noncomputable def max_perimeter_ngon (n : ℕ) (r : ℝ) : ℝ := sorry

theorem max_area_regular_ngon (n : ℕ) (r : ℝ) :
  ∀ (P : set (fin n → ℝ×ℝ)), (is_inscribed_in_circle P r) → (area P ≤ max_area_ngon n r) :=
by sorry

theorem max_perimeter_regular_ngon (n : ℕ) (r : ℝ) :
  ∀ (P : set (fin n → ℝ×ℝ)), (is_inscribed_in_circle P r) → (perimeter P ≤ max_perimeter_ngon n r) :=
by sorry

end max_area_regular_ngon_max_perimeter_regular_ngon_l261_261386


namespace find_x_l261_261368

def balanced (a b c d : ℝ) : Prop :=
  a + b + c + d = a^2 + b^2 + c^2 + d^2

theorem find_x (x : ℝ) : 
  (∀ a b c d : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ balanced a b c d → (x - a) * (x - b) * (x - c) * (x - d) ≥ 0) ↔ x ≥ 3 / 2 :=
sorry

end find_x_l261_261368


namespace joe_spends_50_per_month_l261_261252

variable (X : ℕ) -- amount Joe spends per month

theorem joe_spends_50_per_month :
  let initial_amount := 240
  let resale_value := 30
  let months := 12
  let final_amount := 0 -- this means he runs out of money
  (initial_amount = months * X - months * resale_value) →
  X = 50 := 
by
  intros
  sorry

end joe_spends_50_per_month_l261_261252


namespace min_bottles_l261_261452

theorem min_bottles (milk_needed_oz : ℝ) (oz_to_L : ℝ) (bottle_volume_ml : ℝ) (ml_to_L : ℝ) :
  milk_needed_oz = 60 →
  oz_to_L = 33.8 →
  bottle_volume_ml = 250 →
  ml_to_L = 1000 →
  (ceil ((milk_needed_oz / oz_to_L) * ml_to_L / bottle_volume_ml) ≥ 8) :=
by
  intros h1 h2 h3 h4
  sorry

end min_bottles_l261_261452


namespace GBA_eq_HDA_l261_261243

variables (A E C D F G H : Point)
variables (O P : Circle)
variables (AF CE : Line)
variables (BG DH : Line)

-- Conditions
axiom E_on_AD : E ∈ AD
axiom F_on_CD : F ∈ CD
axiom AF_inter_CE_at_G : G ∈ (AF ∩ CE)
axiom O_circumcircle_AEG : ∀ p : Point, p ∈ O ↔ p ≠ (triangle_circumcenter A E G)
axiom P_circumcircle_CFG : ∀ p : Point, p ∈ P ↔ p ≠ (triangle_circumcenter C F G)
axiom O_inter_P_at_H : H ∈ (O ∩ P)

-- Question
theorem GBA_eq_HDA : ∠GBA = ∠HDA :=
sorry

end GBA_eq_HDA_l261_261243


namespace pentagon_division_l261_261119

structure Pentagon (V : Type) :=
(A B C D E : V)
(is_regular : ∀ (X Y : V), X ≠ Y → ∃ side_length : ℝ, side_length > 0 ∧ 
  (X, Y) ∈ {(A,B), (B,C), (C,D), (D,E), (E,A)} → dist X Y = side_length)

theorem pentagon_division (V : Type) [metric_space V] [has_zero V] (p : Pentagon V) :
∃ (triangles : set (set V)), 
  (∃ T1 T2 T3 T4 T5 : set V, 
    is_triangle T1 ∧ is_triangle T2 ∧ is_triangle T3 ∧ is_triangle T4 ∧ is_triangle T5 ∧ 
    triangles = {T1, T2, T3, T4, T5} ∧ 
    (∀ T ∈ triangles, ∃ t1 t2 t3 : set V, t1 ≠ t2 ∧ t2 ≠ t3 ∧ t3 ≠ t1 ∧ borders T t1 ∧ borders T t2 ∧ borders T t3 ∧ t1 ∈ triangles ∧ t2 ∈ triangles ∧ t3 ∈ triangles)) :=
sorry

end pentagon_division_l261_261119


namespace parabola_focus_to_equation_l261_261390

-- Define the focus of the parabola
def F : (ℝ × ℝ) := (5, 0)

-- Define the standard equation of the parabola
def parabola_equation (x y : ℝ) : Prop :=
  y^2 = 20 * x

-- State the problem in Lean
theorem parabola_focus_to_equation : 
  (F = (5, 0)) → ∀ x y, parabola_equation x y :=
by
  intro h_focus_eq
  sorry

end parabola_focus_to_equation_l261_261390


namespace transform_equation_to_general_form_l261_261338

theorem transform_equation_to_general_form :
  ∀ (x : ℝ), (2 * x^2 = -3 * x + 1) → (2 * x^2 + 3 * x - 1 = 0) :=
by 
  intros x h,
  sorry

end transform_equation_to_general_form_l261_261338


namespace model_price_and_schemes_l261_261365

theorem model_price_and_schemes :
  ∃ (x y : ℕ), 3 * x = 2 * y ∧ x + 2 * y = 80 ∧ x = 20 ∧ y = 30 ∧ 
  ∃ (count m : ℕ), 468 ≤ m ∧ m ≤ 480 ∧ 
                   (20 * m + 30 * (800 - m) ≤ 19320) ∧ 
                   (800 - m ≥ 2 * m / 3) ∧ 
                   count = 13 ∧ 
                   800 - 480 = 320 :=
sorry

end model_price_and_schemes_l261_261365


namespace probability_convex_sequence_l261_261270

open Real

/-- Define the chosen points and conditions -/
variables (x : ℕ → ℝ)

/-- The main theorem stating the probability computation -/
theorem probability_convex_sequence :
  (∀ i, 1 ≤ i ∧ i ≤ 100 → 2 * x i ≥ x (i-1) + x (i+1))
  ∧ x 0 = 0 ∧ x 101 = 0 
  ∧ (∀ i, 1 ≤ i ∧ i ≤ 100 → x i ∈ set.Icc 0 1)
  → 
  sorry =
  1 / (100 * (fact 100)^2) * choose 200 99 :=
sorry

end probability_convex_sequence_l261_261270


namespace range_of_a_l261_261195

theorem range_of_a (a : ℝ) :
  (¬ ∃ x₀ : ℝ, x₀^2 - a*x₀ + 2 < 0) ↔ (a^2 ≤ 8) :=
by
  sorry

end range_of_a_l261_261195


namespace solutions_equation1_solutions_equation2_l261_261308

-- Definition for the first equation
def equation1 (x : ℝ) : Prop := 4 * x^2 - 9 = 0

-- Definition for the second equation
def equation2 (x : ℝ) : Prop := 2 * x^2 - 3 * x - 5 = 0

theorem solutions_equation1 (x : ℝ) :
  equation1 x ↔ (x = 3 / 2 ∨ x = -3 / 2) := 
  by sorry

theorem solutions_equation2 (x : ℝ) :
  equation2 x ↔ (x = 1 ∨ x = 5 / 2) := 
  by sorry

end solutions_equation1_solutions_equation2_l261_261308


namespace external_angle_bisectors_create_acute_triangle_l261_261712

-- Define the concept of a triangle (you might need a proper definition here, adjust as per your existing library if needed)
structure Triangle :=
  (A B C : Type) -- Use appropriate type if necessary, e.g. points

-- Define the external angle bisectors and the formation of a new triangle
def external_angle_bisectors_form_acute_triangle (T : Triangle) : Prop :=
  ∀ LMN, -- assuming LMN is formed by external angle bisectors
    acute_triangle LMN -- acute_triangle should be defined as a triangle with all angles < 90 degrees

-- We state the main theorem based on the problem
theorem external_angle_bisectors_create_acute_triangle 
  (ABC : Triangle) 
  (LMN_formed_by_external_bisectors : external_angle_bisectors_form_acute_triangle ABC) :
  acute_triangle LMN :=
by {
  sorry -- proof goes here
}

end external_angle_bisectors_create_acute_triangle_l261_261712


namespace range_of_g_is_nonnegative_iff_a_is_1_or_3_f_eq_g_for_any_x1_in_neg1_1_iff_a_in_5_6_to_2_l261_261948

-- Define the functions f and g
def f (x : ℝ) : ℝ := (1 / 2) * x + (5 / 2)
def g (x a : ℝ) : ℝ := x^2 - 2 * a * x + 4 * a - 3

-- Problem 1: Proving the range of g(x) is [0, +∞) if and only if a = 1 or a = 3
theorem range_of_g_is_nonnegative_iff_a_is_1_or_3 (a : ℝ) :
  (∀ (y : ℝ), ∃ (x : ℝ), g x a = y ∧ y ≥ 0) ↔ (a = 1 ∨ a = 3) :=
sorry

-- Problem 2: Proving the conditions on a such that for any x1 ∈ [-1, 1], there exists x2 ∈ [-1, 1] such that f(x1) = g(x2)
theorem f_eq_g_for_any_x1_in_neg1_1_iff_a_in_5_6_to_2 (a : ℝ) :
  ((∀ (x1 : ℝ), x1 ∈ set.Icc (- 1) 1 → ∃ (x2 : ℝ), x2 ∈ set.Icc (- 1) 1 ∧ f x1 = g x2 a) ↔ (5 / 6 ≤ a ∧ a ≤ 2)) :=
sorry

end range_of_g_is_nonnegative_iff_a_is_1_or_3_f_eq_g_for_any_x1_in_neg1_1_iff_a_in_5_6_to_2_l261_261948


namespace salary_increase_percent_l261_261672

variable (E : ℝ) (S : ℝ)

-- Condition 1: The number of employees decreased by 10%, so new number of employees is 0.9E
def new_number_of_employees : ℝ := 0.9 * E

-- Condition 2: Total salary before and after the decrease remains the same (100% of initial total salary)
def total_salary : ℝ := E * S

-- Condition 3: The new average salary after the change
def new_avg_salary : ℝ := total_salary / new_number_of_employees

-- Goal: Prove that the percent increase in the average salary is 11.11%
theorem salary_increase_percent : 
    (new_avg_salary / S - 1) * 100 = 11.11 := by
  sorry

end salary_increase_percent_l261_261672


namespace find_angle_A_min_area_triangle_l261_261247

variables {a b c : ℝ}
variables (A B C : ℝ) (D: ℝ) (AD: ℝ) 
noncomputable def law_of_sines := ∀ {a b c A B C : ℝ}, (a = b * sin A / sin B ) ∧ (b = c * sin B / sin C) ∧ (c = a * sin C / sin A)

-- Given conditions
axiom cond1 : (sin A + sin B) * (a - b) = c * (sin C - sin B)
axiom cond2 : AD = 2
axiom cond3 : A = B + C
axiom cond4 : a = b * (sin A / sin B)

theorem find_angle_A (A B C a b c : ℝ) (AD: ℝ) 
    (h1: (sin A + sin B) * (a - b) = c * (sin C - sin B))
    (h2: AD = 2)
    (h3: A = B + C)
    (h4: a = b * (sin A / sin B))
    : A = π / 3 := 
by 
  sorry

theorem min_area_triangle (A B C a b c : ℝ) (AD: ℝ) 
    (h1: (sin A + sin B) * (a - b) = c * (sin C - sin B))
    (h2: AD = 2)
    (h3: A = B + C)
    (h4: a = b * (sin A / sin B))
    : S_triangle_ABC ≥  4 * sqrt(3) / 3 :=
by 
  sorry

end find_angle_A_min_area_triangle_l261_261247


namespace local_min_value_f_max_min_values_f_on_interval_l261_261973

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 6

theorem local_min_value_f : ∃ x, f x = 2 ∧ (∀ y, f x ≤ f y) :=
sorry

theorem max_min_values_f_on_interval :
  (∀ x ∈ Icc (-1 : ℝ) (1 : ℝ), f x ≤ 6) ∧
  (∀ x ∈ Icc (-1 : ℝ) (1 : ℝ), f x ≥ 2)
:= sorry

end local_min_value_f_max_min_values_f_on_interval_l261_261973


namespace simplify_fraction_l261_261882

theorem simplify_fraction :
  (3 - 6 + 12 - 24 + 48 - 96) / (6 - 12 + 24 - 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end simplify_fraction_l261_261882


namespace tangent_line_eq_l261_261334

noncomputable theory
open_locale classical

variables {α : Type*} [linear_ordered_field α]
variables (x y a : α) (A : α × α)

def circle_eq (x y a : α) : Prop := 
  x^2 + y^2 + a * x + 2 = 0

def line_eq (k b : α) (x y : α) : Prop := 
  y = k * x + b

def is_tangent (x y a : α) (A : α × α) (l : α → α → Prop) : Prop :=
  circle_eq A.1 A.2 a ∧ ∃ k b, l k b A.1 A.2 ∧ (∀ x y, line_eq k b x y → 
  (real.sqrt ((x - 2)^2 + y^2) = real.sqrt 2))

theorem tangent_line_eq (A : ℚ × ℚ) (x y a : ℚ) (l : ℚ → ℚ → ℚ → ℚ → Prop) 
  (hA : A = (3,1)) (h_circle : circle_eq x y a) (h_tangent : is_tangent x y a A l) : 
  l 1 (-4) x y :=
sorry

end tangent_line_eq_l261_261334


namespace appended_number_divisible_by_12_l261_261857

theorem appended_number_divisible_by_12 :
  ∃ N, (N = 88) ∧ (∀ n, n ∈ finset.range N \ 71 → (let large_number := (list.range (N + 1)).filter (λ x, 71 ≤ x ∧ x ≤ N) in
       (list.foldr (λ a b, a * 100 + b) 0 large_number) % 12 = 0)) :=
by
  sorry

end appended_number_divisible_by_12_l261_261857


namespace rooks_on_checkerboard_l261_261323

theorem rooks_on_checkerboard :
  let n := 9
  let board := λ (i j : ℕ), (i % 2 = j % 2) -- checkerboard pattern condition
  let (evenCoords := ((fin n).val.enum.filter (λ (i,j), i%2=0 ∧ j%2=0)).length, -- even r-bord condition
       oddCoords := ((fin n).val.enum.filter (λ (i,j), i%2=1 ∧ j%2=1)).length
       in
     even_coords_board_size == 4 ∧ odd_coords_board_size == 5)
  /- Assert that the number of ways to place non-attacking rooks on the black cells is 4! * 5! -/
  then 
    (∃(φ : (fin n).val.enum.map(board).card = 2880,
  sorry

end rooks_on_checkerboard_l261_261323


namespace arithmetic_and_geometric_sequence_l261_261946

theorem arithmetic_and_geometric_sequence (a : ℕ → ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + 2) 
  (h_geom_seq : (a 2)^2 = a 0 * a 3) : 
  a 1 + a 2 = -10 := 
sorry

end arithmetic_and_geometric_sequence_l261_261946


namespace smallest_solution_x4_minus_50x2_plus_625_eq_0_l261_261754

theorem smallest_solution_x4_minus_50x2_plus_625_eq_0 : ∃ x : ℝ, x^4 - 50 * x^2 + 625 = 0 ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y := 
sorry

end smallest_solution_x4_minus_50x2_plus_625_eq_0_l261_261754


namespace Joey_age_l261_261253

-- Define the basic data
def ages : List ℕ := [4, 6, 8, 10, 12]

-- Define the conditions
def cinema_ages (x y : ℕ) : Prop := x + y = 18
def soccer_ages (x y : ℕ) : Prop := x < 11 ∧ y < 11
def stays_home (x : ℕ) : Prop := x = 6

-- The goal is to prove Joey's age
theorem Joey_age : ∃ j, j ∈ ages ∧ stays_home 6 ∧ (∀ x y, cinema_ages x y → x ≠ j ∧ y ≠ j) ∧ 
(∃ x y, soccer_ages x y ∧ x ≠ 6 ∧ y ≠ 6) ∧ j = 8 := by
  sorry

end Joey_age_l261_261253


namespace darryl_had_8_cantaloupes_left_l261_261459

namespace MelonSales

variable {α : Type*} [LinearOrderedField α]

structure Darryl := 
(CantaloupePrice : α)
(HoneydewPrice : α)
(InitialCantaloupes : α)
(InitialHoneydews : α)
(DroppedCantaloupes : α)
(RottenHoneydews : α)
(FinalHoneydews : α)
(TotalRevenue : α)

def cantaloupes_left_at_end_of_day (d : Darryl) : α :=
  d.InitialCantaloupes - d.DroppedCantaloupes - (d.TotalRevenue - d.HoneydewPrice * (d.InitialHoneydews - d.RottenHoneydews - d.FinalHoneydews)) / d.CantaloupePrice

theorem darryl_had_8_cantaloupes_left (d : Darryl) : 
  (d.CantaloupePrice = 2) → 
  (d.HoneydewPrice = 3) → 
  (d.InitialCantaloupes = 30) →
  (d.InitialHoneydews = 27) →
  (d.DroppedCantaloupes = 2) →
  (d.RottenHoneydews = 3) →
  (d.FinalHoneydews = 9) →
  (d.TotalRevenue = 85) →
  cantaloupes_left_at_end_of_day d = 8 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end MelonSales

end darryl_had_8_cantaloupes_left_l261_261459


namespace smallest_solution_l261_261764

theorem smallest_solution : ∃ x : ℝ, x^4 - 50 * x^2 + 625 = 0 ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y :=
by {
  use -5,
  split,
  sorry, -- here would be the proof that (-5)^4 - 50 * (-5)^2 + 625 = 0
  intros y hy,
  sorry -- here would be the proof that for any y such that y^4 - 50 * y^2 + 625 = 0, -5 ≤ y
}

end smallest_solution_l261_261764


namespace harrys_total_cost_l261_261197

def cost_large_pizza : ℕ := 14
def cost_per_topping : ℕ := 2
def number_of_pizzas : ℕ := 2
def number_of_toppings_per_pizza : ℕ := 3
def tip_percentage : ℚ := 0.25

def total_cost (c_pizza c_topping tip_percent : ℚ) (n_pizza n_topping : ℕ) : ℚ :=
  let inital_cost := (c_pizza + c_topping * n_topping) * n_pizza
  let tip := inital_cost * tip_percent
  inital_cost + tip

theorem harrys_total_cost : total_cost 14 2 0.25 2 3 = 50 := 
  sorry

end harrys_total_cost_l261_261197


namespace line_and_circle_separate_l261_261463

noncomputable def line_equation (x y : ℝ) : ℝ := 3 * x + 4 * y - 14

noncomputable def circle_equation (x y : ℝ) : ℝ := (x - 1)^2 + (y + 1)^2 - 4

noncomputable def distance_point_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  (abs (a * p.1 + b * p.2 + c)) / (sqrt (a^2 + b^2))

theorem line_and_circle_separate :
  distance_point_to_line (1, -1) 3 4 (-14) > 2 :=
by
  sorry

end line_and_circle_separate_l261_261463


namespace tessellation_condition_l261_261513

namespace TessellationByArcs

-- Definitions of conditions
def Euclidean_plane := ℝ × ℝ
def is_arc_of_circle (curve : Euclidean_plane → Euclidean_plane) : Prop := sorry -- placeholder for arc condition

variable (figures : Set (Euclidean_plane → Euclidean_plane))
variable (arcs_bound : ∀ f ∈ figures, ∃ arcs: ℕ, arcs ≥ 3 ∧ ∀ curve, curve ∈ f → is_arc_of_circle curve)
variable (tessellates : Set (Euclidean_plane → Euclidean_plane) → Prop := λ figures, ∀ p, ∃ finset (f ∈ figures), p ∈ ⋃ s ∈ finset, s)

-- The theorem expressing the tessellation condition
theorem tessellation_condition (h1: tessellates figures) : ∃ n, n > 2 ∧ ∀ f ∈ figures, ∃ arcs: ℕ, arcs = 2*n ∧ ∀ curve, curve ∈ f → is_arc_of_circle curve :=
by
  sorry

end TessellationByArcs

end tessellation_condition_l261_261513


namespace f_sin_periodic_f_monotonically_increasing_f_minus_2_not_even_f_symmetric_about_point_l261_261189

noncomputable def f (x : ℝ) : ℝ := (4 * Real.exp x) / (Real.exp x + 1)

theorem f_sin_periodic : ∀ x, f (Real.sin (x + 2 * Real.pi)) = f (Real.sin x) := sorry

theorem f_monotonically_increasing : ∀ x y, x < y → f x < f y := sorry

theorem f_minus_2_not_even : ¬(∀ x, f x - 2 = f (-x) - 2) := sorry

theorem f_symmetric_about_point : ∀ x, f x + f (-x) = 4 := sorry

end f_sin_periodic_f_monotonically_increasing_f_minus_2_not_even_f_symmetric_about_point_l261_261189


namespace range_of_f_a3_solution_set_inequality_l261_261561

noncomputable def f (x a : ℝ) : ℝ := x^2 - (a+1)*x + a

-- Prove the range of f(x) on the interval [-1, 3] when a = 3 is [-1, 8]
theorem range_of_f_a3 : ∃ lo hi : ℝ, 
  (lo = -1 ∧ hi = 8) ∧ ∀ x ∈ set.Icc (-1 : ℝ) 3, f x 3 ∈ set.Icc lo hi := sorry

-- Prove the solution sets for f(x) > 0 based on the value of a
theorem solution_set_inequality (a x : ℝ) :
  if a > 1 then
    (f x a > 0 ↔ x < 1 ∨ x > a)
  else if a < 1 then
    (f x a > 0 ↔ x < a ∨ x > 1)
  else
    (a = 1 → (f x a > 0 ↔ x ≠ 1)) := sorry

end range_of_f_a3_solution_set_inequality_l261_261561


namespace total_time_calculation_l261_261893

-- Conditions
def basketball_school : Nat := 15 * 5
def basketball_weekend : Nat := 30 * 2 
def soccer_specific_days : Nat := 20 * 3
def gymnastics_days : Nat := 30 * 2 
def soccer_saturday : Float := 45 / 2
def swimming_saturday : Float := 60 / 2

-- Total time
def total_time_of_practice : Float := 
  float_of_nat basketball_school + 
  float_of_nat basketball_weekend + 
  float_of_nat soccer_specific_days + 
  float_of_nat gymnastics_days + 
  soccer_saturday + 
  swimming_saturday

-- Theorem statement
theorem total_time_calculation : total_time_of_practice = 307.5 :=
  by sorry

end total_time_calculation_l261_261893


namespace factorial_division_identity_l261_261449

theorem factorial_division_identity : 
  (factorial (factorial 4)) / (factorial 4) = factorial 23 := by
  sorry

end factorial_division_identity_l261_261449


namespace correct_population_l261_261004

variable (P : ℕ) (S : ℕ)
variable (math_scores : ℕ → Type)

-- Assume P is the total number of students who took the exam.
-- Let math_scores(P) represent the math scores of P students.

def population_data (P : ℕ) : Prop := 
  P = 50000

def sample_data (S : ℕ) : Prop :=
  S = 2000

theorem correct_population (P : ℕ) (S : ℕ) (math_scores : ℕ → Type)
  (hP : population_data P) (hS : sample_data S) : 
  math_scores P = math_scores 50000 :=
by {
  sorry
}

end correct_population_l261_261004


namespace last_integer_in_sequence_is_15625_l261_261348

theorem last_integer_in_sequence_is_15625 :
  ∃ n : ℕ, (1,000,000 / 2^n = 15,625 ∧ 1,000,000 / 2^(n+1) ≠ ⌊1,000,000 / 2^(n+1)⌋ ) :=
by
  sorry

end last_integer_in_sequence_is_15625_l261_261348


namespace solve_complex_eq_l261_261212

theorem solve_complex_eq (z : ℂ) (h : (2 - 3 * complex.I) * z = 5 - complex.I) : z = 1 + complex.I := 
  sorry

end solve_complex_eq_l261_261212


namespace parametric_curve_constants_l261_261071

noncomputable def parametric_x (t : ℝ) : ℝ := 3 * Real.sin t + Real.cos t
noncomputable def parametric_y (t : ℝ) : ℝ := 3 * Real.cos t

theorem parametric_curve_constants :
  ∃ (a b c d : ℝ), 
  a = 1/9 ∧ b = -2/27 ∧ c = -1/9 ∧ d = 1 ∧ 
  ∀ t : ℝ, (a * (parametric_x t)^2 + b * (parametric_x t) * (parametric_y t) + c * (parametric_y t)^2 = d) :=
by
  use 1/9, -2/27, -1/9, 1
  split; norm_num
  sorry

end parametric_curve_constants_l261_261071


namespace sum_terms_sequence_l261_261628

noncomputable def geometric_sequence := ℕ → ℝ

variables (a : geometric_sequence)
variables (r : ℝ) (h_pos : ∀ n, a n > 0)

-- Geometric sequence condition
axiom geom_seq (n : ℕ) : a (n + 1) = a n * r

-- Given condition
axiom h_condition : a 3 * a 5 + a 2 * a 10 + 2 * a 4 * a 6 = 100

-- The goal is to prove that a_4 + a_6 = 10
theorem sum_terms_sequence : a 4 + a 6 = 10 :=
by
  sorry

end sum_terms_sequence_l261_261628


namespace time_per_toy_l261_261437

theorem time_per_toy (total_toys : ℕ) (total_hours : ℕ) (h : total_toys = 50 ∧ total_hours = 100) :
  total_hours / total_toys = 2 :=
by
  cases h with h1 h2
  rw [h1, h2]
  norm_num
sorry

end time_per_toy_l261_261437


namespace part1_prove_K_squared_part2_expectation_correct_l261_261007

/-
Part 1: Prove that K^2 < 2.706 with 90% confidence given the survey results
-/

def employee_category : Type :=
| WalkingPioneer : employee_category
| WalkingStar : employee_category

def gender : Type :=
| Male : gender
| Female : gender

structure Employee :=
  (category : employee_category)
  (gender : gender)

noncomputable def chi_squared_statistic (a b c d : ℕ) : ℝ :=
  let n := (a + b + c + d: ℕ)
  (n * ((a * d - b * c : ℕ)^2) : ℝ) / (((a+b)*(c+d)*(a+c)*(b+d) : ℕ) : ℝ)

theorem part1_prove_K_squared (a b c d : ℕ) (K_squared : ℝ) (critical_value : ℝ) :
  (a = 24) → (b = 16) → (c = 16) → (d = 14) →
  K_squared = chi_squared_statistic a b c d →
  critical_value = 2.706 →
  K_squared < critical_value :=
sorry

/-
Part 2: Prove the probability distribution and expectation of X
-/

def distribution_of_X : fin 4 → ℝ
| 0 => 64 / 125
| 1 => 48 / 125
| 2 => 12 / 125
| _ => 1 / 125

noncomputable def expectation_of_X : ℝ :=
(list.range 4).sum (λ i => i * distribution_of_X i)

theorem part2_expectation_correct :
  expectation_of_X = 3 / 5 :=
sorry

end part1_prove_K_squared_part2_expectation_correct_l261_261007


namespace root_equation_alpha_beta_property_l261_261598

theorem root_equation_alpha_beta_property {α β : ℝ} (h1 : α^2 + α - 1 = 0) (h2 : β^2 + β - 1 = 0) :
    α^2 + 2 * β^2 + β = 4 :=
by
  sorry

end root_equation_alpha_beta_property_l261_261598


namespace expected_adjacent_red_pairs_in_circle_l261_261317

-- Definitions and conditions
def deck := fin 104 -- represents the 104 cards
def red_cards : finset deck := finset.range 52 -- out of 104 cards, 52 are red

-- Expected number of adjacent red pairs
noncomputable def expected_adjacent_red_pairs : ℚ :=
  (52:ℚ) * (51:ℚ) / (103:ℚ)

-- Statement of the theorem
theorem expected_adjacent_red_pairs_in_circle :
  expected_adjacent_red_pairs = 2652 / 103 :=
by
  sorry

end expected_adjacent_red_pairs_in_circle_l261_261317


namespace roots_cubic_polynomial_l261_261652

theorem roots_cubic_polynomial (r s t : ℝ)
  (h₁ : 8 * r^3 + 1001 * r + 2008 = 0)
  (h₂ : 8 * s^3 + 1001 * s + 2008 = 0)
  (h₃ : 8 * t^3 + 1001 * t + 2008 = 0)
  (h₄ : r + s + t = 0) :
  (r + s)^3 + (s + t)^3 + (t + r)^3 = 753 := 
sorry

end roots_cubic_polynomial_l261_261652


namespace customers_who_didnt_tip_l261_261090

def initial_customers : ℕ := 39
def added_customers : ℕ := 12
def customers_who_tipped : ℕ := 2

theorem customers_who_didnt_tip : initial_customers + added_customers - customers_who_tipped = 49 := by
  sorry

end customers_who_didnt_tip_l261_261090


namespace cubic_roots_reciprocal_sum_l261_261553

theorem cubic_roots_reciprocal_sum {α β γ : ℝ} 
  (h₁ : α + β + γ = 6)
  (h₂ : α * β + β * γ + γ * α = 11)
  (h₃ : α * β * γ = 6) :
  (1 / α^2) + (1 / β^2) + (1 / γ^2) = 49 / 36 := 
by 
  sorry

end cubic_roots_reciprocal_sum_l261_261553


namespace part_I_part_II_l261_261536

open Nat

-- Define the sequence {a_n} and its first n terms sum S_n
def S_n (a : ℕ → ℕ) (n : ℕ) : ℕ := 2 * a n - 2

-- Define the sequence {S_n} sum
def T_n (S : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in range n, S (i + 1)

-- Part (I): Prove the general formula for the terms of the sequence {a_n}
theorem part_I (a : ℕ → ℕ) (h : ∀ n : ℕ, n > 0 → S_n a n = ∑ i in range n, a (i + 1)) : ∀ n : ℕ, n > 0 → a n = 2 ^ n :=
by sorry

-- Part (II): Prove the sum of the first n terms of the sequence {S_n}
theorem part_II (a : ℕ → ℕ) (h : ∀ n : ℕ, n > 0 → a n = 2 ^ n) : ∀ n : ℕ, T_n (S_n a) n = 2 ^ (n + 2) - 4 - 2 * n :=
by sorry

end part_I_part_II_l261_261536


namespace percentage_of_total_capacity_used_l261_261692

theorem percentage_of_total_capacity_used:
  let initial_documents := 180
  let initial_photos := 380
  let initial_videos := 290
  let deletion_photos := 65.9
  let deletion_videos := 98.1
  let addition_documents := 20.4
  let addition_photos := 37.6
  let total_capacity := 1500

  let new_documents := initial_documents + addition_documents
  let new_photos := initial_photos - deletion_photos + addition_photos
  let new_videos := initial_videos - deletion_videos

  let compressed_documents := new_documents * 0.95
  let compressed_photos := new_photos * 0.88
  let compressed_videos := new_videos * 0.80

  let total_used_space := compressed_documents + compressed_photos + compressed_videos
  let percentage_used := (total_used_space / total_capacity) * 100

  percentage_used ≈ 43.56 :=
by
  sorry

end percentage_of_total_capacity_used_l261_261692


namespace length_of_second_platform_l261_261825

-- Definitions
def length_train : ℝ := 230
def time_first_platform : ℝ := 15
def length_first_platform : ℝ := 130
def total_distance_first_platform : ℝ := length_train + length_first_platform
def time_second_platform : ℝ := 20

-- Statement to prove
theorem length_of_second_platform : 
  ∃ L : ℝ, (total_distance_first_platform / time_first_platform) = ((length_train + L) / time_second_platform) ∧ L = 250 :=
by
  sorry

end length_of_second_platform_l261_261825


namespace rooks_on_checkerboard_l261_261325

theorem rooks_on_checkerboard :
  let n := 9
  let board := λ (i j : ℕ), (i % 2 = j % 2) -- checkerboard pattern condition
  let (evenCoords := ((fin n).val.enum.filter (λ (i,j), i%2=0 ∧ j%2=0)).length, -- even r-bord condition
       oddCoords := ((fin n).val.enum.filter (λ (i,j), i%2=1 ∧ j%2=1)).length
       in
     even_coords_board_size == 4 ∧ odd_coords_board_size == 5)
  /- Assert that the number of ways to place non-attacking rooks on the black cells is 4! * 5! -/
  then 
    (∃(φ : (fin n).val.enum.map(board).card = 2880,
  sorry

end rooks_on_checkerboard_l261_261325


namespace probability_of_x_gt_3y_in_rectangle_l261_261675

noncomputable theory
open_locale classical

-- Define the rectangular region
structure Point :=
  (x : ℝ)
  (y : ℝ)

def in_rectangle (p : Point) : Prop :=
  0 ≤ p.x ∧ p.x ≤ 3012 ∧ 0 ≤ p.y ∧ p.y ≤ 3013

def in_triangle (p : Point) : Prop :=
  0 ≤ p.x ∧ p.x ≤ 3012 ∧ 0 ≤ p.y ∧ p.y ≤ (p.x / 3)

-- Define the probability calculation
def probability_x_gt_3y : ℝ :=
  (1 / 2) * (3012 * 1004) / (3012 * 3013)

-- Statement to show probability calculation is correct
theorem probability_of_x_gt_3y_in_rectangle :
  ∀ (p : Point), in_rectangle p → (x > 3y) (p) →
  probability_x_gt_3y = 1004 / 3013 :=
by
  sorry

end probability_of_x_gt_3y_in_rectangle_l261_261675


namespace hyperbola_constants_l261_261267

-- Definitions 
def F1 : ℝ × ℝ := (-4, 2 - sqrt 8 / 2)
def F2 : ℝ × ℝ := (-4, 2 + sqrt 8 / 2)

-- The result statement we should prove
theorem hyperbola_constants :
  ∃ h k a b : ℝ, 
    (-4, 2) = ((F1.1 + F2.1) / 2, (F1.2 + F2.2) / 2) ∧ 
    2 * a = 2 ∧ 
    sqrt 2 = sqrt (a^2 + b^2) ∧ 
    h + k + a + b = 0 :=
by 
  use -4, 2, 1, 1
  split
  { sorry } -- midpoint calculation
  split 
  { sorry } -- constant difference
  split 
  { sorry } -- distance between foci
  { sorry } -- final sum

end hyperbola_constants_l261_261267


namespace rhombus_unique_property_l261_261726

-- Definitions of properties
def parallel_sides (shape : Type) : Prop := sorry
def diagonals_bisect (shape : Type) : Prop := sorry
def diagonals_perpendicular (shape : Type) : Prop := sorry
def adjacent_angles_supplementary (shape : Type) : Prop := sorry

-- Definitions of shapes
structure Rhombus :=
(parallel_sides : parallel_sides Rhombus)
(diagonals_bisect : diagonals_bisect Rhombus)
(diagonals_perpendicular : diagonals_perpendicular Rhombus)
(adjacent_angles_supplementary : adjacent_angles_supplementary Rhombus)

structure Rectangle :=
(parallel_sides : parallel_sides Rectangle)
(diagonals_bisect : diagonals_bisect Rectangle)
(--note: this property is not included as it does not hold for rectangles)
(adjacent_angles_supplementary : adjacent_angles_supplementary Rectangle)

theorem rhombus_unique_property :
  ∀ (R : Rhombus) (Rec : Rectangle), diagonals_perpendicular Rhombus ∧ ¬ diagonals_perpendicular Rectangle :=
by
  intros R Rec
  split
  sorry
  sorry

end rhombus_unique_property_l261_261726


namespace diana_exceeds_apollo_by_2_probability_l261_261908

def sum_outcomes_dice : finset ℕ := finset.range 13 \ {0, 1}

def favorable_outcome (d a : ℕ) : Prop := d > a + 1

def count_favorable_outcomes : ℕ :=
  sum_outcomes_dice.sum $ λ d,
    sum_outcomes_dice.sum $ λ a,
      if favorable_outcome d a then 1 else 0

def total_possible_outcomes : ℕ := 36 * 36

def probability_event_happening : ℚ :=
  (count_favorable_outcomes : ℚ) / (total_possible_outcomes : ℚ)

theorem diana_exceeds_apollo_by_2_probability :
  probability_event_happening = 47 / 432 :=
by sorry

end diana_exceeds_apollo_by_2_probability_l261_261908


namespace simplify_product_l261_261694

theorem simplify_product : (∏ n in finset.range (500*6+1), (6*n + 3)/(6*n - 3)) = 600.6 :=
by
  sorry

end simplify_product_l261_261694


namespace no_sum_seven_lt_sixteen_exists_sum_five_lt_twelve_l261_261355

variables (a : ℕ → ℝ)

def sum_gt_seven_pt_three {a : ℕ → ℝ} (h : ∀ (i j k : ℕ), i ≠ j ∧ j ≠ k ∧ k ≠ i → a i + a j + a k > 7) : Prop :=
∀ (i j k : ℕ), i ≠ j → j ≠ k → k ≠ i → a i + a j + a k > 7

def sum_lt_sixteen_pt_seven {a : ℕ → ℝ} : Prop :=
∃ S : ℕ → list ℕ, sum (S.map (λ s, a s)) < 16

def sum_lt_twelve_pt_five {a : ℕ → ℝ} : Prop :=
∃ S : ℕ → list ℕ, sum (S.map (λ s, a s)) < 12

theorem no_sum_seven_lt_sixteen (h : ∀ (i j k : ℕ), i ≠ j ∧ j ≠ k ∧ k ≠ i → a i + a j + a k > 7) : ¬ (∃ S : ℕ → list ℕ, sum (S.map (λ s, a s)) < 16) :=
sorry

theorem exists_sum_five_lt_twelve (h : ∀ (i j k : ℕ), i ≠ j ∧ j ≠ k ∧ k ≠ i → a i + a j + a k > 7) : ∃ S : ℕ → list ℕ, sum (S.map (λ s, a s)) < 12 :=
sorry

end no_sum_seven_lt_sixteen_exists_sum_five_lt_twelve_l261_261355


namespace max_parking_spaces_l261_261299

theorem max_parking_spaces (n : ℕ) : 
  (∀ f_bus s_bus, 
    (f_bus = {5, 6, 7}) ∧ 
    (s_bus = {n - 9, n - 8, n - 7}) ∧ 
    ∃ four_more_buses : set (ℕ), 
      (∀ b, b ∈ four_more_buses → b = 3 * k ∧ k ∈ ℕ ∧ 
      ∀ x ∈ four_more_buses, 
        four_more_buses ⊆ ∅ ∨ ⊆ (set.univ \ f_bus) ∨ ⊆ (set.univ \ s_bus)) ∧ 
        (n ≤ 29)) 
  := 
sorry

end max_parking_spaces_l261_261299


namespace mixed_bag_cost_l261_261880

def cost_per_pound_colombian : ℝ := 5.5
def cost_per_pound_peruvian : ℝ := 4.25
def total_weight : ℝ := 40
def weight_colombian : ℝ := 28.8

noncomputable def cost_per_pound_mixed_bag : ℝ :=
  (weight_colombian * cost_per_pound_colombian + (total_weight - weight_colombian) * cost_per_pound_peruvian) / total_weight

theorem mixed_bag_cost :
  cost_per_pound_mixed_bag = 5.15 :=
  sorry

end mixed_bag_cost_l261_261880


namespace inequality_solution_fractional_equation_solution_l261_261399

-- Proof Problem 1
theorem inequality_solution (x : ℝ) : (1 - x) / 3 - x < 3 - (x + 2) / 4 → x > -2 :=
by
  sorry

-- Proof Problem 2
theorem fractional_equation_solution (x : ℝ) : (x - 2) / (2 * x - 1) + 1 = 3 / (2 * (1 - 2 * x)) → false :=
by
  sorry

end inequality_solution_fractional_equation_solution_l261_261399


namespace factor_multiplication_of_q_l261_261999

variable (w v : ℝ) (f : ℝ → ℝ) (z : ℝ)

def q := 5 * w / (4 * v * f (z ^ 2))

theorem factor_multiplication_of_q :
  (q (4 * w) v (λ y => 2 * f y) (3 * z)) = (2 / 9) * (q w v f z) :=
by
sorry

end factor_multiplication_of_q_l261_261999


namespace exists_distinct_natural_numbers_with_perfect_squares_l261_261915

theorem exists_distinct_natural_numbers_with_perfect_squares :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (∃ m1 m2 : ℕ, a^2 + 2 * c * d + b^2 = m1^2 ∧ c^2 + 2 * a * b + d^2 = m2^2) :=
by
  -- We need to find a, b, c, and d such that all conditions are satisfied
  use 1, 6, 2, 3
  repeat {split, norm_num, cc}
  existsi 7
  existsi 5
  norm_num
  split; ring
  split;
  norm_num
  sorry

end exists_distinct_natural_numbers_with_perfect_squares_l261_261915


namespace magnitude_of_two_a_minus_b_l261_261572

namespace VectorMagnitude

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (3, -2)

-- Function to calculate the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Vector operation 2a - b
def two_a_minus_b : ℝ × ℝ := (2 * a.1 - b.1, 2 * a.2 - b.2)

-- The theorem to prove
theorem magnitude_of_two_a_minus_b : magnitude two_a_minus_b = Real.sqrt 17 := by
  sorry

end VectorMagnitude

end magnitude_of_two_a_minus_b_l261_261572


namespace find_sphere_radius_l261_261987

open Real

noncomputable def shots_to_sphere_radius (n: ℕ) (r: ℝ) : ℝ :=
  real.cbrt (n * r^3)

theorem find_sphere_radius :
  shots_to_sphere_radius 216 1 = 6 :=
by
  sorry

end find_sphere_radius_l261_261987


namespace area_of_triangle_RTO_l261_261617

universe u

variables {Point : Type u} [AffineSpace Point ℝ]
variables (E F G H R S U T O : Point)
variables (z : ℝ) [nonneg z]

def is_parallelogram (A B C D : Point) : Prop :=
  ∃ (mid : Point), mid = (A +ₗ B) / 2 ∧ mid = (C +ₗ D) / 2

def trisects (line1 line2 : ℝ → Point) (pt : Point) : Prop :=
  ∃ t1 t2, 0 < t1 ∧ t1 < t2 ∧ t2 < 1 ∧ line1 t1 = pt ∧ line1 t2 = pt

def extended (line : ℝ → Point) (pt : Point) : Prop :=
  ∃ t, t > 1 ∧ line t = pt

def area_of_parallelogram (A B C D : Point) (area : ℝ) : Prop :=
  ∃ (basis : VectorSpace Basis ℝ), basis.volume = area

theorem area_of_triangle_RTO 
  (H_parallelogram : is_parallelogram E F G H)
  (H_trisects_FG : trisects (λ t, E + t • (G - E)) S)
  (H_meets_EH_at_R : extended (λ t, E + t • (H - G)) R)
  (H_trisects_EH : trisects (λ t, G + t • (H - G)) U)
  (H_meets_EF_at_T : extended (λ t, G + t • (F - G)) T)
  (H_intersection : ∃ O, (λ t, E + t • (R - E)) intersects (λ t, G + t • (T - G)) O)
  (H_area_EFGH : area_of_parallelogram E F G H z) :
  area_of_triangle R T O (z / 3) :=
sorry

end area_of_triangle_RTO_l261_261617


namespace final_number_appended_is_84_l261_261853

noncomputable def arina_sequence := "7172737475767778798081"

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def is_divisible_by_12 (n : ℕ) : Prop := n % 12 = 0

-- Define adding numbers to the sequence
def append_number (seq : String) (n : ℕ) : String := seq ++ n.repr

-- Create the full sequence up to 84 and check if it's divisible by 12
def generate_full_sequence : String :=
  let base_seq := arina_sequence
  let full_seq := append_number (append_number (append_number (append_number (append_number (append_number (append_number (append_number (append_number arina_sequence 82) 83) 84))) 85) 86) 87) 88 
  full_seq

theorem final_number_appended_is_84 : (∃ seq : String, is_divisible_by_12(seq.to_nat) ∧ seq.ends_with "84") := 
by
  sorry

end final_number_appended_is_84_l261_261853


namespace find_a_l261_261523

theorem find_a (a : ℝ) (h : (1 : ℂ) + 3 * complex.I * (1 + a * complex.I) ∈ ℝ) : a = -3 :=
sorry

end find_a_l261_261523


namespace distance_from_D_to_plane_alpha_l261_261539

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

noncomputable def distance_to_plane (D C N : Point3D) : ℝ :=
  let DC := Point3D.mk (C.x - D.x) (C.y - D.y) (C.z - D.z)
  let dot_product := DC.x * N.x + DC.y * N.y + DC.z * N.z
  let norm := Math.sqrt (N.x^2 + N.y^2 + N.z^2)
  (abs dot_product) / norm

def D : Point3D := ⟨0, 3, 0⟩
def C : Point3D := ⟨1, 2, 0⟩
def N : Point3D := ⟨1, 0, 1⟩

theorem distance_from_D_to_plane_alpha : distance_to_plane D C N = (Math.sqrt 2) / 2 :=
by
  sorry

end distance_from_D_to_plane_alpha_l261_261539


namespace total_items_left_in_store_l261_261872

noncomputable def items_ordered : ℕ := 4458
noncomputable def items_sold : ℕ := 1561
noncomputable def items_in_storeroom : ℕ := 575

theorem total_items_left_in_store : 
  (items_ordered - items_sold) + items_in_storeroom = 3472 := 
by 
  sorry

end total_items_left_in_store_l261_261872


namespace parallel_line_through_point_l261_261711

-- Problem: Prove the equation of the line that passes through the point (1, 1)
-- and is parallel to the line 2x - y + 1 = 0 is 2x - y - 1 = 0.

theorem parallel_line_through_point (x y : ℝ) (c : ℝ) :
  (2*x - y + 1 = 0) → (x = 1) → (y = 1) → (2*1 - 1 + c = 0) → c = -1 → (2*x - y - 1 = 0) :=
by
  sorry

end parallel_line_through_point_l261_261711


namespace k_divides_ak_plus_bk_l261_261268

theorem k_divides_ak_plus_bk (a b k : ℕ) (l : ℕ) (ha : a > 1) (hb : b > 1) 
  (hodd₁ : a % 2 = 1) (hodd₂ : b % 2 = 1) (hsum : a + b = 2^l) (hk : k ∈ (nat.range (k + 1)).tail) :
  (k^2 ∣ a^k + b^k) ↔ k = 1 :=
by
  sorry

end k_divides_ak_plus_bk_l261_261268


namespace range_F_l261_261375

noncomputable def F (x : ℝ) : ℝ := |2 * x + 2| - |2 * x - 2|

theorem range_F : set.range F = set.Icc (-4 : ℝ) 4 :=
by
  sorry

end range_F_l261_261375


namespace find_t_find_roots_sum_find_m_l261_261974

-- Definitions and conditions
def monotonic_y (a : ℝ) := ∀ x : ℝ, (0 < x ∧ x ≤ real.sqrt a → (x + (a / x)) ≥ y) ∧ (real.sqrt a ≤ x → (x + (a / x)) ≤ y)

def f (t : ℝ) (x : ℝ) := abs (t * (x + 4 / x) - 5)

-- (1)
theorem find_t (t : ℝ) (h : ∀ x : ℝ, (0 < x ∧ x < 2 → f t x < f t x) ∨ (2 < x → f t x > f t x)) : t ≥ 5 / 4 :=
sorry

-- (2)
theorem find_roots_sum (t : ℝ) (k : ℝ) 
(h : t = 1 ∧ ∃ (x1 x2 x3 x4 : ℝ), f 1 x1 - k = 0 ∧ f 1 x2 - k = 0 ∧ f 1 x3 - k = 0 ∧ f 1 x4 - k = 0 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 < x4) : 
  x1 + x2 + x3 + x4 = 10 :=
sorry

-- (3)
theorem find_m (a b : ℝ) (m : ℝ) 
(h : t = 1 ∧ 0 < a ∧ a < b ∧ b ≤ 2 ∧ ∃ M ∈ (0, a], ∃ N ∈ [a, b], f t M = ma ∧ f t N = mb) : 
  1 / 2 ≤ m ∧ m < 9 / 16 :=
sorry

end find_t_find_roots_sum_find_m_l261_261974


namespace chewbacca_gum_l261_261107

variable {y : ℝ}

theorem chewbacca_gum (h1 : 25 - 2 * y ≠ 0) (h2 : 40 + 4 * y ≠ 0) :
    25 - 2 * y/40 = 25/(40 + 4 * y) → y = 2.5 :=
by
  intros h
  sorry

end chewbacca_gum_l261_261107


namespace distance_post_office_l261_261810

theorem distance_post_office 
  (D : ℝ)
  (speed_to_post_office : ℝ := 25)
  (speed_back : ℝ := 4)
  (total_time : ℝ := 5 + (48 / 60)) :
  (D / speed_to_post_office + D / speed_back = total_time) → D = 20 :=
by
  sorry

end distance_post_office_l261_261810


namespace equality_of_segments_l261_261258

-- Define the type for points in a Euclidean plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the type for angles in degrees
def Angle := ℝ

-- Conditions of the problem
variables (A B C D P Q R S T : Point)
variables (m_APB m_TRB m_DQC m_PSR m_PAR m_PBD m_PAD m_PCB : Angle)

-- Isosceles triangle with AB = AC
def is_isosceles_triangle (A B C : Point) : Prop :=
  (dist A B) = (dist A C)

-- D is the foot of the perpendicular from A to BC
def is_foot_of_perpendicular (A D B C : Point) : Prop :=
  ∃ h : Angle, h = 90 ∧
  (B.x - D.x) * (C.x - D.x) + (B.y - D.y) * (C.y - D.y) = 0

-- Interior point P of triangle ADC such that angle APB > 90° and angle conditions
def interior_point_conditions (A P B D C : Point) (m_APB m_PBD m_PAD m_PCB : Angle) : Prop :=
  (m_APB > 90) ∧ ((m_PBD + m_PAD) = m_PCB)

-- Intersection conditions
def intersection_conditions (C P D Q A : Point) (B P D R A : Point) : Prop :=
  lies_on Q (line C P) ∧ lies_on Q (line D A) ∧
  lies_on R (line B P) ∧ lies_on R (line D A)

-- Define properties of points on lines or line segments
def lies_on (P : Point) (l : Line) : Prop :=
  l.contains P

def segment (A B : Point) : Set Point :=
  { P : Point | P = A ∨ P = B }

-- Angle conditions for T and S
def angle_conditions (T R B D Q : Point) (S P R A : Point) (m_TRB m_DQC m_PSR m_PAR : Angle) : Prop :=
  (m_TRB = m_DQC) ∧ (m_PSR = 2 * m_PAR)

-- Definition of distances for length comparison
def dist (P Q : Point) : ℝ :=
  sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

-- The main theorem
theorem equality_of_segments (A B C D P Q R S T : Point)
  (m_APB m_TRB m_DQC m_PSR m_PAR m_PBD m_PAD m_PCB : Angle)
  (h_iso : is_isosceles_triangle A B C)
  (h_perp : is_foot_of_perpendicular A D B C)
  (h_interior : interior_point_conditions A P B D C m_APB m_PBD m_PAD m_PCB)
  (h_intersect : intersection_conditions C P D Q A R B P D R A)
  (h_angle : angle_conditions T R B D Q S P R A m_TRB m_DQC m_PSR m_PAR) :
  dist R S = dist R T := sorry

end equality_of_segments_l261_261258


namespace sum_of_xy_l261_261564

theorem sum_of_xy (x y : ℝ) (h1 : x^3 - 6*x^2 + 12*x = 13) (h2 : y^3 + 3*y - 3*y^2 = -4) : x + y = 3 :=
by sorry

end sum_of_xy_l261_261564


namespace S_n_expression_a_n_expression_l261_261943

/-- Define the sequence S_n and a_n and prove the conditions given. -/
open Nat

def S_n (n : ℕ) : ℚ :=
  match n with
  | 0 => 0
  | n + 1 => (2 * (n + 1) : ℚ) / (n + 2)

theorem S_n_expression (n : ℕ) (h : n > 0) : S_n n = (2 * n : ℚ) / (n + 1) := 
  sorry

theorem a_n_expression (n : ℕ) (h : n > 0) : (S_n n).to_real = (n : ℚ)^2 * (a n) := 
  -- Here we assume the definition of a_n can be derived correctly.
  sorry

end S_n_expression_a_n_expression_l261_261943


namespace product_sequence_l261_261879

theorem product_sequence :
  (∏ n in (finset.range 8).map (λ n, n + 1), (1 + 1 / (n : ℚ))) = 9 :=
by sorry

end product_sequence_l261_261879


namespace sunny_ahead_by_10_meters_l261_261220

-- Define the speeds of Sunny and Windy in the first race.
variables (s w : ℝ)

-- Define the conditions given in the problem.
-- Sunny finishes 20 meters ahead in a 200-meter race.
condition1 : (200 / s) = (180 / w)

-- In the second race:
-- Sunny runs the first 100 meters at a speed reduced by 10% and then at his original speed.
-- Windy's constant speed is equivalent to her average speed in the first race.

theorem sunny_ahead_by_10_meters :
  let t_sunny := (100 / (0.9 * s) + 100 / s) in
  let t_windy := 200 / w in
  t_sunny < t_windy ∧ (w * (t_windy - t_sunny)) = 10 :=
by
  -- Here we skip the proof details and use sorry.
  sorry

end sunny_ahead_by_10_meters_l261_261220


namespace last_appended_number_is_84_l261_261840

theorem last_appended_number_is_84 : 
  ∃ N : ℕ, 
    let s := "7172737475767778798081" ++ (String.intercalate "" (List.map toString [82, 83, 84])) in
    (N = 84) ∧ (s.toNat % 12 = 0) :=
by
  sorry

end last_appended_number_is_84_l261_261840


namespace polygon_area_l261_261702

/-- Given the conditions:
    1. The area of polygon ABCDEF is 78.
    2. AB = 10
    3. BC = 11
    4. FA = 7
    Prove that DE + EF = 12 
-/
theorem polygon_area (h1 : area_of_polygon ABCDEF = 78) (h2 : side_length AB = 10) (h3 : side_length BC = 11) (h4 : side_length FA = 7) :
  let DE := 11 - 7 in 
  let EF := 32 / DE in
  DE + EF = 12 :=
by
  sorry

end polygon_area_l261_261702


namespace smallest_solution_x4_minus_50x2_plus_625_eq_0_l261_261757

theorem smallest_solution_x4_minus_50x2_plus_625_eq_0 :
  ∃ (x : ℝ), (x * x * x * x - 50 * x * x + 625 = 0) ∧ (∀ y, (y * y * y * y - 50 * y * y + 625 = 0) → x ≤ y) :=
sorry

end smallest_solution_x4_minus_50x2_plus_625_eq_0_l261_261757


namespace birdhouse_volume_difference_l261_261683

theorem birdhouse_volume_difference :
  let sara_width := 1
  let sara_height := 2
  let sara_depth := 2
  let jake_width := 16 / 12
  let jake_height := 20 / 12
  let jake_depth := 18 / 12
  let sara_volume := sara_width * sara_height * sara_depth
  let jake_volume := jake_width * jake_height * jake_depth
  let volume_difference := sara_volume - jake_volume
  volume_difference ≈ 0.668 :=
by
  sorry

end birdhouse_volume_difference_l261_261683


namespace reflect_coordinates_l261_261719

variables {m b : ℝ}

theorem reflect_coordinates (m b : ℝ) : 
  let x1 : ℝ := 2
  let y1 : ℝ := 3
  let x2 : ℝ := 10
  let y2 : ℝ := 7
  let midpoint_x : ℝ := (x1 + x2) / 2 
  let midpoint_y : ℝ := (y1 + y2) / 2 
  let midpoint : ℝ × ℝ := (midpoint_x, midpoint_y)
  in (midpoint_y = m * midpoint_x + b) → m + b = 15 :=
by
  sorry

end reflect_coordinates_l261_261719


namespace find_f_expression_l261_261526

theorem find_f_expression (x : ℝ) (h : x ≠ -1) : 
  (∀ t : ℝ, t = (1 - x) / (1 + x) → f t = x) →
  f x = (1 - x) / (1 + x) :=
by
  intro H
  -- Proof omitted
  sorry

end find_f_expression_l261_261526


namespace polynomial_divisibility_l261_261681

theorem polynomial_divisibility
  (a b c : ℝ) (h : a ≠ 0) :
  ∃ P : Polynomial ℝ, (Polynomial.X ^ 2 + 1) ∣ (a * P^2 + b * P + c) :=
sorry

end polynomial_divisibility_l261_261681


namespace solve_inequality_system_l261_261698

theorem solve_inequality_system (x : ℝ) 
  (h1 : 3 * x - 1 > x + 1) 
  (h2 : (4 * x - 5) / 3 ≤ x) 
  : 1 < x ∧ x ≤ 5 :=
by
  sorry

end solve_inequality_system_l261_261698


namespace population_under_50000_l261_261337

def population_percentage (p1 p2 : ℕ) : ℕ := p1 + p2

theorem population_under_50000
  (p1 p2 : ℕ)
  (h1 : p1 = 30)
  (h2 : p2 = 45) :
  population_percentage p1 p2 = 75 :=
by
  simp [population_percentage, h1, h2]
  sorry

end population_under_50000_l261_261337


namespace cone_lateral_area_l261_261998

theorem cone_lateral_area (r l S: ℝ) (h1: r = 1 / 2) (h2: l = 1) (h3: S = π * r * l) : 
  S = π / 2 :=
by
  sorry

end cone_lateral_area_l261_261998


namespace exist_pos_integers_m_n_l261_261260

def d (n : ℕ) : ℕ :=
  -- Number of divisors of n
  sorry 

theorem exist_pos_integers_m_n :
  ∃ (m n : ℕ), (m > 0) ∧ (n > 0) ∧ (m = 24) ∧ 
  ((∃ (triples : Finset (ℕ × ℕ × ℕ)),
    (∀ (a b c : ℕ), (a, b, c) ∈ triples ↔ (0 < a) ∧ (a < b) ∧ (b < c) ∧ (c ≤ m) ∧ (d (n + a) * d (n + b) * d (n + c)) % (a * b * c) = 0) ∧ 
    (triples.card = 2024))) :=
sorry

end exist_pos_integers_m_n_l261_261260


namespace employees_in_organization_l261_261616

-- Definitions based on the given conditions
def num_below_10k : ℕ := 250
def num_between_10k_50k : ℕ := 500
def percent_below_50k : ℝ := 0.75

-- The total number of employees earning less than 50k $
def total_below_50k : ℕ := num_below_10k + num_between_10k_50k

-- Proving the total number of employees in the organization
theorem employees_in_organization (E : ℝ) : E = 1000 :=
by
  have H1 : total_below_50k = 750 := by rfl
  have H2 : percent_below_50k * E = 750 := by rfl
  sorry

end employees_in_organization_l261_261616


namespace smallest_solution_of_quartic_equation_l261_261750

theorem smallest_solution_of_quartic_equation :
  ∃ x : ℝ, (x^4 - 50 * x^2 + 625 = 0) ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y := sorry

end smallest_solution_of_quartic_equation_l261_261750


namespace commute_days_l261_261092

def a : ℕ := 2  -- number of days taking car morning and bicycle evening
def b : ℕ := 12 -- number of days taking bicycle morning and car evening
def c : ℕ := 8  -- number of days taking car in both morning and evening
def y : ℕ := a + b + c

theorem commute_days (h1 : a + c = 10) (h2 : b = 12) (h3 : a + b = 14) : y = 22 :=
by {
  unfold a b c y,
  sorry
}

end commute_days_l261_261092


namespace count_sets_l261_261532

theorem count_sets (S : Set ℝ) : 
  (S ⊆ {1, 2, 3, 4, 5}) ∧ S ≠ ∅ ∧ (∀ a ∈ S, (6 - a) ∈ S) → ∃! n : ℕ, n = 7 := 
begin
  sorry
end

end count_sets_l261_261532


namespace total_puppies_is_12_l261_261800

theorem total_puppies_is_12 (f m : Nat) (rfm : Float) 
  (h_f : f = 2) (h_m : m = 10) (h_rfm : rfm = 0.2) 
  (h_ratio : (f / m : Float) = rfm) : 
  f + m = 12 := by
  sorry

end total_puppies_is_12_l261_261800


namespace count_integers_j_l261_261584

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ d in (Finset.range (n + 1)).filter (n % · = 0), d

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_integers_j (h : Nat) :
  ∃ (count : ℕ), count = 16 ∧ (∀ j, 1 ≤ j ∧ j ≤ 3025 →
  sum_of_divisors j = 1 + Nat.sqrt j + j →
  ∃ p, is_prime p ∧ j = p * p) :=
sorry

end count_integers_j_l261_261584


namespace arash_half_cake_l261_261884

theorem arash_half_cake :
  ∀ (cake : Fin 5 → ℝ) (adjacent : Fin 5 → Fin 5 → Prop),
    (∀ i j, adjacent i j → adjacent j i) →
    (∀ keqadjacent cake adjacent, ∃ (arash_pieces : Finset (Fin 5)),
      arash_pieces.card = 3 ∧ (∑ i in arash_pieces, cake i) ≥ ∑ i in fin 5, cake i / 2) :=
begin
  sorry
end

end arash_half_cake_l261_261884


namespace new_apples_grew_l261_261679

-- The number of apples originally on the tree.
def original_apples : ℕ := 11

-- The number of apples picked by Rachel.
def picked_apples : ℕ := 7

-- The number of apples currently on the tree.
def current_apples : ℕ := 6

-- The number of apples left on the tree after picking.
def remaining_apples : ℕ := original_apples - picked_apples

-- The number of new apples that grew on the tree.
def new_apples : ℕ := current_apples - remaining_apples

-- The theorem we need to prove.
theorem new_apples_grew :
  new_apples = 2 := by
    sorry

end new_apples_grew_l261_261679


namespace sellingPrice_is_459_l261_261663

-- Definitions based on conditions
def costPrice : ℝ := 540
def markupPercentage : ℝ := 0.15
def discountPercentage : ℝ := 0.2608695652173913

-- Calculating the marked price based on the given conditions
def markedPrice (cp : ℝ) (markup : ℝ) : ℝ := cp + (markup * cp)

-- Calculating the discount amount based on the marked price and the discount percentage
def discount (mp : ℝ) (discountPct : ℝ) : ℝ := discountPct * mp

-- Calculating the selling price
def sellingPrice (mp : ℝ) (discountAmt : ℝ) : ℝ := mp - discountAmt

-- Stating the final proof problem
theorem sellingPrice_is_459 :
  sellingPrice (markedPrice costPrice markupPercentage) (discount (markedPrice costPrice markupPercentage) discountPercentage) = 459 :=
by
  sorry

end sellingPrice_is_459_l261_261663


namespace find_principal_l261_261822

-- Definitions based on conditions
def simple_interest (P R T : ℚ) : ℚ := (P * R * T) / 100

-- Given conditions
def SI : ℚ := 6016.75
def R : ℚ := 8
def T : ℚ := 5

-- Stating the proof problem
theorem find_principal : 
  ∃ P : ℚ, simple_interest P R T = SI ∧ P = 15041.875 :=
by {
  sorry
}

end find_principal_l261_261822


namespace initial_bags_of_rice_l261_261823

theorem initial_bags_of_rice (sold restocked final initial : Int) 
  (h1 : sold = 23)
  (h2 : restocked = 132)
  (h3 : final = 164) 
  : ((initial - sold) + restocked = final) ↔ initial = 55 :=
by 
  have eq1 : ((initial - sold) + restocked = final) ↔ initial - 23 + 132 = 164 := by rw [h1, h2, h3]
  simp [eq1]
  sorry

end initial_bags_of_rice_l261_261823


namespace directrix_parabola_l261_261494

-- Given the equation of the parabola and required transformations:
theorem directrix_parabola (d : ℚ) : 
  (∀ x : ℚ, y = -4 * x^2 + 4) → d = 65 / 16 :=
by sorry

end directrix_parabola_l261_261494


namespace minimum_chess_pieces_l261_261739

theorem minimum_chess_pieces (n : ℕ) : 
  (n % 3 = 1) ∧ (n % 5 = 3) ∧ (n % 7 = 5) → 
  n = 103 :=
by 
  sorry

end minimum_chess_pieces_l261_261739


namespace solve_x_l261_261376

theorem solve_x : ∃ x : ℝ, 65 + (5 * x) / (180 / 3) = 66 ∧ x = 12 := by
  sorry

end solve_x_l261_261376


namespace valid_pairs_count_l261_261139

def is_valid (n : ℕ) : Prop :=
  n > 0 ∧ (n / 10 % 10 ≠ 0) ∧ (n / 100 % 10 ≠ 0)

theorem valid_pairs_count :
  {p : ℕ × ℕ | is_valid p.1 ∧ is_valid p.2 ∧ p.1 + p.2 = 1100}.card = 889 :=
by sorry

end valid_pairs_count_l261_261139


namespace exists_gcd_one_l261_261676

theorem exists_gcd_one (p q r : ℤ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h : Int.gcd p (Int.gcd q r) = 1) : ∃ a : ℤ, Int.gcd p (q + a * r) = 1 :=
sorry

end exists_gcd_one_l261_261676


namespace question1_question2_l261_261791

-- Proof problem for Question 1
theorem question1 : 27 ^ (2 / 3 : ℝ) + 16 ^ (-1 / 2 : ℝ) - (1 / 2 : ℝ) ^ (-2 : ℝ) - (8 / 27 : ℝ) ^ (-2 / 3 : ℝ) = 3 :=
by
  sorry

-- Proof problem for Question 2
theorem question2 (a : ℝ) (h : a ≥ 1) : (sqrt (a - 1)) ^ 2 + abs (1 - a) + (1 - a) ^ (1 / 3 : ℝ) = a - 1 :=
by
  sorry

end question1_question2_l261_261791


namespace no_such_prism_exists_l261_261428

-- Define the conditions as given in the problem
variables {A B C A1 B1 C1 D K: Point}
variable (s : Sphere)
variable (prism : RegularTriangularPrism A B C A1 B1 C1)
variable (isInscribed : s.isInscribed prism)

-- Define the additional conditions
variable (D_K_eq : dist D K = 2 * Real.sqrt 6)
variable (D_A_eq : dist D A = 6)

-- Prove that such a prism does not exist
theorem no_such_prism_exists :
  ¬ ∃ (A B C A1 B1 C1 D K: Point) 
      (prism : RegularTriangularPrism A B C A1 B1 C1) 
      (s : Sphere) 
      (isInscribed : s.isInscribed prism) 
      (D_K_eq : dist D K = 2 * Real.sqrt 6) 
      (D_A_eq : dist D A = 6), 
      True :=
by 
  sorry

end no_such_prism_exists_l261_261428


namespace employee_salaries_l261_261146

variables (n m p q : ℝ)

-- Definitions according to the conditions
def salary_m := 1.40 * n
def salary_diff_m_n := salary_m - n
def salary_p := 0.85 * salary_diff_m_n
def salary_q := 1.10 * salary_p
def total_salary := n + salary_m + salary_p + salary_q

theorem employee_salaries :
  total_salary n m p q = 3000 ∧
  salary_m = 1.40 * n ∧
  salary_p = 0.85 * salary_diff_m_n ∧
  salary_q = 1.10 * salary_p ∧
  n ≈ 963.59 ∧
  m ≈ 1349.03 ∧
  p ≈ 327.62 ∧
  q ≈ 360.34 :=
by
  sorry -- Proof to be provided.

end employee_salaries_l261_261146


namespace smallest_solution_l261_261762

theorem smallest_solution : ∃ x : ℝ, x^4 - 50 * x^2 + 625 = 0 ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y :=
by {
  use -5,
  split,
  sorry, -- here would be the proof that (-5)^4 - 50 * (-5)^2 + 625 = 0
  intros y hy,
  sorry -- here would be the proof that for any y such that y^4 - 50 * y^2 + 625 = 0, -5 ≤ y
}

end smallest_solution_l261_261762


namespace correct_option_is_C_l261_261771

-- Our conditions as mathematical expressions
def condition_A : Prop := (+6) + (-13) = +7
def condition_B : Prop := (+6) + (-13) = -19
def condition_C : Prop := (+6) + (-13) = -7
def condition_D : Prop := (-5) + (-3) = 8

-- The proposition we need to prove
theorem correct_option_is_C : condition_C ∧ ¬condition_A ∧ ¬condition_B ∧ ¬condition_D :=
by 
  sorry

end correct_option_is_C_l261_261771


namespace total_logs_in_stack_l261_261433

theorem total_logs_in_stack :
  let a1 := 15 in
  let an := 5 in
  let n := 11 in
  (n / 2 * (a1 + an) = 110) :=
by
  let a1 := 15
  let an := 5
  let n := 11
  have h1 : (n / 2 * (a1 + an) = 110) := sorry
  exact h1

end total_logs_in_stack_l261_261433


namespace tan_alpha_neg_four_over_three_l261_261149

theorem tan_alpha_neg_four_over_three (α : ℝ) (h_cos : Real.cos α = -3/5) (h_alpha_range : α ∈ Set.Ioo (-π) 0) : Real.tan α = -4/3 :=
  sorry

end tan_alpha_neg_four_over_three_l261_261149


namespace geometric_sequence_fourth_term_l261_261885

theorem geometric_sequence_fourth_term :
  let a₁ := 3^(3/4)
  let a₂ := 3^(2/4)
  let a₃ := 3^(1/4)
  ∃ a₄, a₄ = 1 ∧ a₂ = a₁ * (a₃ / a₂) ∧ a₃ = a₂ * (a₄ / a₃) :=
by
  sorry

end geometric_sequence_fourth_term_l261_261885


namespace solve_gcd_triples_eq_l261_261896

def is_odd (n : ℕ) : Prop := n % 2 = 1

def satisfies_conditions (a b c : ℕ) : Prop :=
  Nat.gcd a 20 = b ∧ Nat.gcd b 15 = c ∧ Nat.gcd a c = 5

noncomputable def all_solutions_set : set (ℕ × ℕ × ℕ) :=
  { (20 * k, 20, 5) | k > 0 ∧ is_odd k } ∪ 
  { (10 * t, 10, 5) | t > 0 ∧ is_odd t } ∪ 
  { (5 * m, 5, 5) | m > 0 }

theorem solve_gcd_triples_eq : 
  { (a, b, c) | satisfies_conditions a b c } = all_solutions_set :=
sorry

end solve_gcd_triples_eq_l261_261896


namespace arrange_PERCEPTION_l261_261906

theorem arrange_PERCEPTION :
  ∀ n k1 k2 k3 : ℕ, n = 10 → k1 = 2 → k2 = 2 → k3 = 2 →
  (nat.factorial n) / (nat.factorial k1 * nat.factorial k2 * nat.factorial k3) = 453600 := 
by
  intros n k1 k2 k3 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end arrange_PERCEPTION_l261_261906


namespace chessboard_sum_bound_l261_261057

theorem chessboard_sum_bound 
  (a : Fin 8 → Fin 8 → ℝ) 
  (h1 : ∑ i j, a i j = 1956) 
  (h2 : ∑ i, a i i = 112)
  (h3 : ∑ i, a i (7 - i) = 112)
  (h4 : ∀ i j, a i j = a j i ∨ a i j = a j (7 - i) ∨ a i j = a (7 - j) i ∨ a i j = a (7 - i) (7 - j)) : 
  ∀ i, (∑ j, a i j < 518) ∧ (∑ j, a j i < 518) := 
sorry

end chessboard_sum_bound_l261_261057


namespace probability_closer_to_6_than_0_is_0_6_l261_261423

noncomputable def probability_closer_to_6_than_0 : ℝ :=
  let total_length := 7
  let segment_length_closer_to_6 := 4
  let probability := (segment_length_closer_to_6 : ℝ) / total_length
  probability

theorem probability_closer_to_6_than_0_is_0_6 :
  probability_closer_to_6_than_0 = 0.6 := by
  sorry

end probability_closer_to_6_than_0_is_0_6_l261_261423


namespace largest_pot_cost_l261_261284

noncomputable def cost_of_largest_pot (x : ℝ) : ℝ :=
  x + 5 * 0.15

theorem largest_pot_cost :
  ∃ (x : ℝ), (6 * x + 5 * 0.15 + (0.15 + 2 * 0.15 + 3 * 0.15 + 4 * 0.15 + 5 * 0.15) = 8.85) →
    cost_of_largest_pot x = 1.85 :=
by
  sorry

end largest_pot_cost_l261_261284


namespace proof_problem_l261_261182

-- Define events for tossing a coin twice
inductive CoinToss : Type
| heads : CoinToss
| tails : CoinToss

def eventA_twice := (CoinToss.heads, CoinToss.heads)
def eventB_twice := (CoinToss.tails, CoinToss.tails)
def is_complementary (A B : Prop) := (A ∨ B) ∧ ¬(A ∧ B)

-- Define events for selecting defective products
inductive Product : Type
| defective : Product
| non_defective : Product

def choose3_defective : list Product → nat := sorry -- Assume this function is correctly implemented

def eventA_products := ∃ l : list Product, choose3_defective l ≤ 2
def eventB_products := ∃ l : list Product, choose3_defective l ≥ 2

-- Defining mutual exclusivity
def is_mutually_exclusive (A B : Prop) := ¬(A ∧ B)

theorem proof_problem :
  (¬ is_complementary (eventA_twice = eventB_twice) (eventB_twice = eventB_twice)) ∧
  is_mutually_exclusive (eventA_twice = eventA_twice) (eventB_twice = eventB_twice) ∧ 
  ¬ is_mutually_exclusive eventA_products eventB_products →
  (1 = 1) :=
by
  sorry

end proof_problem_l261_261182


namespace dino_dolls_count_l261_261435

theorem dino_dolls_count (T : ℝ) (H : 0.7 * T = 140) : T = 200 :=
sorry

end dino_dolls_count_l261_261435


namespace hyperbola_eccentricity_l261_261976

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∀ x y : ℝ, (y = x ∨ y = -x) → y = (b / a) * x) :
  (sqrt (1 + (b / a)^2) = sqrt 2) :=
by 
  sorry

end hyperbola_eccentricity_l261_261976


namespace sum_of_possible_ks_l261_261239

theorem sum_of_possible_ks (j k : ℕ) (hj : 0 < j) (hk : 0 < k) (h : (1 : ℚ) / j + (1 : ℚ) / k = 1 / 4) :
    k ∈ {20, 12, 8, 6, 5} ∧ {5, 6, 8, 12, 20}.sum = 51 :=
by sorry

end sum_of_possible_ks_l261_261239


namespace amy_required_hours_per_week_l261_261831

variable (summer_hours_per_week : ℕ) (summer_weeks : ℕ) (summer_pay : ℕ) 
variable (pay_raise_percent : ℕ) (school_year_weeks : ℕ) (required_school_year_pay : ℕ)

def summer_hours_total := summer_hours_per_week * summer_weeks
def summer_hourly_pay := summer_pay / summer_hours_total
def new_hourly_pay := summer_hourly_pay + (summer_hourly_pay / 10)  -- 10% pay raise
def total_needed_hours := required_school_year_pay / new_hourly_pay
def required_hours_per_week := total_needed_hours / school_year_weeks

theorem amy_required_hours_per_week :
  summer_hours_per_week = 40 →
  summer_weeks = 12 →
  summer_pay = 4800 →
  pay_raise_percent = 10 →
  school_year_weeks = 36 →
  required_school_year_pay = 7200 →
  required_hours_per_week = 18 := sorry

end amy_required_hours_per_week_l261_261831


namespace arina_sophia_divisible_l261_261838

theorem arina_sophia_divisible (N: ℕ) (k: ℕ) (large_seq: list ℕ): 
  (k = 81) → 
  (large_seq = (list.range' 71 (k + 1)).append (list.range' 82 (N + 1))) → 
  (nat.digits 10 (list.foldl (λ n d, 10 * n + d) 0 large_seq)).nat_sum % 3 = 0 → 
  (nat.digits 10 (list.foldl (λ n d, 10 * n + d) 0 large_seq)).last_digits 2 % 4 = 0 →
  (list.foldl (λ n d, 10 * n + d) 0 large_seq) % 12 = 0 → 
  N = 84 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end arina_sophia_divisible_l261_261838


namespace clothing_probability_l261_261579

/-- I have a drawer with 6 shirts, 8 pairs of shorts, 7 pairs of socks, and 3 jackets in it.
    If I reach in and randomly remove four articles of clothing, what is the probability that 
    I get one shirt, one pair of shorts, one pair of socks, and one jacket? -/
theorem clothing_probability :
  let total_articles := 6 + 8 + 7 + 3
  let total_combinations := Nat.choose total_articles 4
  let favorable_combinations := 6 * 8 * 7 * 3
  (favorable_combinations : ℚ) / total_combinations = 144 / 1815 :=
by
  let total_articles := 6 + 8 + 7 + 3
  let total_combinations := Nat.choose total_articles 4
  let favorable_combinations := 6 * 8 * 7 * 3
  suffices (favorable_combinations : ℚ) / total_combinations = 144 / 1815
  by
    sorry
  sorry

end clothing_probability_l261_261579


namespace quadratic_ineq_solution_set_l261_261977

theorem quadratic_ineq_solution_set (a b c : ℝ) (h1 : a < 0) (h2 : ∀ x : ℝ, 3 < x → x < 6 → ax^2 + bx + c > 0) :
  ∀ x : ℝ, x < (1 / 6) ∨ x > (1 / 3) → cx^2 + bx + a < 0 := by 
  sorry

end quadratic_ineq_solution_set_l261_261977


namespace min_n_for_circuit_l261_261738

theorem min_n_for_circuit
  (n : ℕ) 
  (p_success_component : ℝ)
  (p_work_circuit : ℝ) 
  (h1 : p_success_component = 0.5)
  (h2 : p_work_circuit = 1 - p_success_component ^ n) 
  (h3 : p_work_circuit ≥ 0.95) :
  n ≥ 5 := 
sorry

end min_n_for_circuit_l261_261738


namespace Tony_paid_1560_83_usd_l261_261640

-- Definitions from the conditions
def cost_lego_block := 250
def cost_toy_sword := 100
def cost_play_dough := 30

def first_day_purchases := (2, 3)  -- (Lego blocks, Toy swords)
def second_day_purchases := (1, 2, 10)  -- (Lego blocks, Toy swords, Play doughs)

def first_day_discount := 0.20
def second_day_discount := 0.10

def first_day_exchange_rates := (0.85, 0.75)  -- (EUR to USD, GBP to USD)
def second_day_exchange_rates := (0.84, 0.74)  -- (EUR to USD, GBP to USD)

def sales_tax := 0.05

-- Total cost calculation
noncomputable def total_cost_usd : ℝ :=
  let lego_blocks_first := 2 * cost_lego_block * (1 - first_day_discount)
  let toy_swords_first := 3 * cost_toy_sword * (1 - first_day_discount) / 0.85
  let subtotal_first := lego_blocks_first + toy_swords_first
  let total_first := subtotal_first * (1 + sales_tax)

  let lego_blocks_second := cost_lego_block * (1 - second_day_discount)
  let toy_swords_second := 2 * cost_toy_sword * (1 - second_day_discount) / 0.84
  let play_doughs_second := 10 * cost_play_dough * (1 - second_day_discount) / 0.74
  let subtotal_second := lego_blocks_second + toy_swords_second + play_doughs_second
  let total_second := subtotal_second * (1 + sales_tax)

  total_first + total_second

theorem Tony_paid_1560_83_usd : total_cost_usd = 1560.83 := by
  -- Proof is omitted
  sorry

end Tony_paid_1560_83_usd_l261_261640


namespace remaining_volume_ball_l261_261063

-- Define the given parameters
def radius_ball : ℝ := 12 -- 24 cm / 2
def volume_sphere (r : ℝ) : ℝ := (4 / 3) * π * r^3
def radius_hole : ℝ := 1 -- 2 cm / 2
def depth_hole : ℝ := 6

-- Total volume of the ball before holes are drilled
def volume_ball_before : ℝ := volume_sphere radius_ball

-- Volume of a single cylindrical hole
def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h
def volume_hole : ℝ := volume_cylinder radius_hole depth_hole

-- Total volume of four cylindrical holes
def total_volume_holes : ℝ := 4 * volume_hole

-- Prove the remaining volume after drilling four holes
theorem remaining_volume_ball : volume_ball_before - total_volume_holes = 2280 * π := by
  -- volume_sphere 12 = (4 / 3) * π * 12^3 = 2304 * π
  -- volume_cylinder 1 6 = π * 1^2 * 6 = 6 * π
  -- total_volume_holes = 4 * 6 * π = 24 * π
  -- remaining volume = 2304 * π - 24 * π = 2280 * π
  sorry

end remaining_volume_ball_l261_261063


namespace arina_sophia_divisible_l261_261833

theorem arina_sophia_divisible (N: ℕ) (k: ℕ) (large_seq: list ℕ): 
  (k = 81) → 
  (large_seq = (list.range' 71 (k + 1)).append (list.range' 82 (N + 1))) → 
  (nat.digits 10 (list.foldl (λ n d, 10 * n + d) 0 large_seq)).nat_sum % 3 = 0 → 
  (nat.digits 10 (list.foldl (λ n d, 10 * n + d) 0 large_seq)).last_digits 2 % 4 = 0 →
  (list.foldl (λ n d, 10 * n + d) 0 large_seq) % 12 = 0 → 
  N = 84 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end arina_sophia_divisible_l261_261833


namespace AO_perpendicular_B_l261_261644

variables {A B C B' C' O : Type} [triangle ABC]

-- Let B' and C' be the feet of the altitudes from B and C respectively
def is_foot_of_altitude (B' : Type) (C' : Type) (B : Type) (C : Type) : Prop := sorry

-- Let O be the circumcenter of triangle ABC
def is_circumcenter (O : Type) (ABC : Type) : Prop := sorry

-- Proof that (AO) is perpendicular to (B'C')
theorem AO_perpendicular_B'C' (h1 : is_foot_of_altitude B' C' B C) (h2 : is_circumcenter O ABC) :
  perpendicular (line_through A O) (line_through B' C') :=
sorry

end AO_perpendicular_B_l261_261644


namespace gauss_floor_of_root_l261_261517

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

theorem gauss_floor_of_root : 
  ∃ x_0 : ℝ, f x_0 = 0 ∧ (2 ≤ x_0 ∧ x_0 < 3) → ∀ x_0, ⌊x_0⌋ = 2 :=
by
  contradiction -- Replace this with the actual proof. For now, it stands in to skip the proof.
  sorry

end gauss_floor_of_root_l261_261517


namespace geometric_sequence_a4_l261_261626

-- Define the terms of the geometric sequence
variable {a : ℕ → ℝ}

-- Define the conditions of the problem
def a2_cond : Prop := a 2 = 2
def a6_cond : Prop := a 6 = 32

-- Define the theorem we want to prove
theorem geometric_sequence_a4 (a2_cond : a 2 = 2) (a6_cond : a 6 = 32) : a 4 = 8 := by
  sorry

end geometric_sequence_a4_l261_261626


namespace digit_counts_l261_261126

theorem digit_counts :
  ∃ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℕ),
  a_0 = 1 ∧
  a_1 = 11 ∧
  a_2 = 2 ∧
  a_3 = 1 ∧
  a_4 = 1 ∧
  a_5 = 1 ∧
  a_6 = 1 ∧
  a_7 = 1 ∧
  a_8 = 1 ∧
  a_9 = 1 := 
by
  use 1, 11, 2, 1, 1, 1, 1, 1, 1, 1
  sorry

end digit_counts_l261_261126


namespace total_distance_traveled_l261_261635

def distance_first_leg : ℝ := 80.0 * 2
def distance_second_leg : ℝ := 65.0 * 4
def distance_third_leg : ℝ := 75.0 * 3
def distance_fourth_leg : ℝ := 70.0 * 5

def total_distance : ℝ := distance_first_leg + distance_second_leg + distance_third_leg + distance_fourth_leg

theorem total_distance_traveled : total_distance = 995.0 :=
by
  have h1 : distance_first_leg = 160.0 := by norm_num
  have h2 : distance_second_leg = 260.0 := by norm_num
  have h3 : distance_third_leg = 225.0 := by norm_num
  have h4 : distance_fourth_leg = 350.0 := by norm_num
  simp [total_distance, h1, h2, h3, h4]
  norm_num
  done

end total_distance_traveled_l261_261635


namespace not_all_pieces_found_l261_261046

theorem not_all_pieces_found (n : ℕ) (m : ℕ) 
  (vova : ℕ → ℕ := λx, 7 * x) (dima : ℕ → ℕ := λx, 4 * x) 
  (initial_pieces : ℕ := 1)
  (janitor_collected_pieces : ℕ := 2019) : 
  (janitor_collected_pieces - initial_pieces) % 3 ≠ 0 :=
by sorry

end not_all_pieces_found_l261_261046


namespace projection_of_a_onto_e_l261_261950

variables (a e : EuclideanSpace ℝ (Fin 3))
variable (θ : ℝ)
variable (ha : ∥a∥ = 4)
variable (he : ∥e∥ = 1)
variable (hθ : θ = (2 / 3) * Real.pi)

noncomputable def projection := ∥a∥ * Real.cos θ

theorem projection_of_a_onto_e :
  projection a θ = -2 :=
by
  rw [he, ha]
  sorry

end projection_of_a_onto_e_l261_261950


namespace time_to_pass_bridge_l261_261086

-- Definitions directly from the conditions
def length_of_train : ℝ := 360
def length_of_bridge : ℝ := 140
def speed_kmph : ℝ := 30

-- Convert speed from km/hour to meters/second
def speed_mps := speed_kmph * 1000 / 3600

-- Total distance to cover
def total_distance := length_of_train + length_of_bridge

-- Prove the time it takes for the train to pass the bridge is approximately 60.02 seconds
theorem time_to_pass_bridge :
  abs ((total_distance / speed_mps) - 60.02) < 0.01 :=
by
  sorry

end time_to_pass_bridge_l261_261086


namespace event_A_consists_of_3_basic_events_l261_261767

-- Define the type for a coin outcome
inductive Coin : Type
| heads : Coin
| tails : Coin

-- Define the possible outcomes of tossing two fair coins
def outcomes : List (Coin × Coin) :=
  [(Coin.heads, Coin.heads), (Coin.heads, Coin.tails), (Coin.tails, Coin.heads), (Coin.tails, Coin.tails)]

-- Define the event "At least one coin lands heads up"
def at_least_one_heads (outcome : Coin × Coin) : Prop :=
  outcome.fst = Coin.heads ∨ outcome.snd = Coin.heads

-- The main statement
theorem event_A_consists_of_3_basic_events :
  (List.filter at_least_one_heads outcomes).length = 3 :=
by
  sorry

end event_A_consists_of_3_basic_events_l261_261767


namespace anna_cannot_afford_tour_l261_261099

noncomputable def future_value (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def future_cost (C0 : ℝ) (i : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  C0 * (1 + i / n) ^ (n * t)

theorem anna_cannot_afford_tour :
  let P := 40000
  let r := 0.05
  let n := 1
  let t := 3
  let C0 := 45000
  let i := 0.05
  future_value P r n t < future_cost C0 i n t :=
  by
    let P := 40000
    let r := 0.05
    let n := 1
    let t := 3
    let C0 := 45000
    let i := 0.05
    have fv := future_value P r n t
    have fc := future_cost C0 i n t
    show fv < fc from sorry

end anna_cannot_afford_tour_l261_261099


namespace proof_problem_l261_261669

-- Define the initial value and operations.

def initial_value : ℝ := 555.55
def division_factor : ℝ := 1 / 3
def subtract_value : ℝ := 333.33
def rounded_value_to_nearest_hundredth (x : ℝ) : ℝ := Real.round (x * 100) / 100

-- Define the main assertion to be proved.

theorem proof_problem : 
  (rounded_value_to_nearest_hundredth (initial_value * division_factor) - subtract_value) = -148.15 :=
by
  sorry

end proof_problem_l261_261669


namespace handshakes_count_example_l261_261100

-- Define a conference group with specific properties
structure Conference :=
  (total_people : ℕ)
  (group_A_size : ℕ)
  (group_B_size : ℕ)
  (group_A_knows_each_other : ℕ)
  (exceptions_in_A : ℕ)
  (exceptions_not_know_count : ℕ)

def conference_example : Conference :=
  { total_people := 40,
    group_A_size := 25,
    group_B_size := 15,
    group_A_knows_each_other := 25,
    exceptions_in_A := 5,
    exceptions_not_know_count := 3
  }

-- Define the math proof statement
def calculate_handshakes (C : Conference) : ℕ :=
  let groupA_B_handshakes := C.group_B_size * C.group_A_size in
  let groupB_handshakes := nat.choose C.group_B_size 2 in
  let groupA_handshakes := C.exceptions_in_A * C.exceptions_not_know_count in
  groupA_B_handshakes + groupB_handshakes + groupA_handshakes

theorem handshakes_count_example : calculate_handshakes conference_example = 495 :=
by
  -- Proof can be filled later
  sorry

end handshakes_count_example_l261_261100


namespace angle_same_terminal_side_l261_261349

theorem angle_same_terminal_side (α : ℝ) : 
  (∃ k : ℤ, α = k * 360 - 100) ↔ (∃ k : ℤ, α = k * 360 + (-100)) :=
sorry

end angle_same_terminal_side_l261_261349


namespace angle4_is_35_l261_261148

theorem angle4_is_35
  (angle1 angle2 angle3 angle4 angle5 angle6 : ℝ)
  (h1 : angle1 + angle2 = 180)
  (h2 : angle3 = angle4)
  (ha : angle1 = 50)
  (h_opposite : angle5 = 60)
  (triangle_sum : angle1 + angle5 + angle6 = 180)
  (supplementary_angle : angle2 + angle6 = 180) :
  angle4 = 35 :=
by
  sorry

end angle4_is_35_l261_261148


namespace cyclic_sum_inequality_l261_261656

theorem cyclic_sum_inequality (a b c d e : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < e) (h6 : a * b * c * d * e = 1) :
    (cyclic_sum (λ x, (a + (a * b * c)) / (1 + (a * b) + (a * b * c * d)))) ≥ 10 / 3 :=
sorry

-- Placeholder definition for cyclic_sum to make the theorem statement valid
def cyclic_sum (f : ℝ → ℝ) : ℝ := sorry

end cyclic_sum_inequality_l261_261656


namespace smallest_solution_x4_minus_50x2_plus_625_eq_0_l261_261758

theorem smallest_solution_x4_minus_50x2_plus_625_eq_0 :
  ∃ (x : ℝ), (x * x * x * x - 50 * x * x + 625 = 0) ∧ (∀ y, (y * y * y * y - 50 * y * y + 625 = 0) → x ≤ y) :=
sorry

end smallest_solution_x4_minus_50x2_plus_625_eq_0_l261_261758


namespace volume_difference_l261_261690

def sara_dimensions : (ℤ × ℤ × ℤ) := (1 * 12, 2 * 12, 2 * 12) -- dimensions in inches
def jake_dimensions : (ℤ × ℤ × ℤ) := (16, 20, 18) -- dimensions already in inches

def volume (dims : (ℤ × ℤ × ℤ)) : ℤ :=
  dims.1 * dims.2 * dims.3

theorem volume_difference :
  volume sara_dimensions - volume jake_dimensions = 1152 :=
by
  sorry

end volume_difference_l261_261690


namespace time_left_after_council_room_is_zero_l261_261670

-- Define the conditions
def totalTimeAllowed : ℕ := 30
def travelToSchoolTime : ℕ := 25
def walkToLibraryTime : ℕ := 3
def returnBooksTime : ℕ := 4
def walkToCouncilRoomTime : ℕ := 5
def submitProjectTime : ℕ := 3

-- Calculate time spent up to the student council room
def timeSpentUpToCouncilRoom : ℕ :=
  travelToSchoolTime + walkToLibraryTime + returnBooksTime + walkToCouncilRoomTime + submitProjectTime

-- Question: How much time is left after leaving the student council room to reach the classroom without being late?
theorem time_left_after_council_room_is_zero (totalTimeAllowed travelToSchoolTime walkToLibraryTime returnBooksTime walkToCouncilRoomTime submitProjectTime : ℕ):
  totalTimeAllowed - timeSpentUpToCouncilRoom = 0 := by
  sorry

end time_left_after_council_room_is_zero_l261_261670


namespace train_speed_l261_261436

theorem train_speed (length_train : ℝ) (length_bridge : ℝ) (time_taken : ℝ)
  (h_train : length_train = 100)
  (h_bridge : length_bridge = 120)
  (h_time : time_taken = 21.998240140788738) :
  let total_distance := length_train + length_bridge in
  let speed_mps := total_distance / time_taken in
  let speed_kmph := speed_mps * 3.6 in
  speed_kmph = 36 :=
by
  rw [h_train, h_bridge, h_time]
  let total_distance := 220
  let speed_mps := total_distance / 21.998240140788738
  let speed_kmph := speed_mps * 3.6
  sorry

end train_speed_l261_261436


namespace factors_of_M_l261_261897

theorem factors_of_M : 
  let M := 2^4 * 3^3 * 5^2 * 7^1 in
  (∃ n : ℕ, n = (5 * 4 * 3 * 2)) := sorry

end factors_of_M_l261_261897


namespace range_of_m_eq_interval_l261_261556

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := sin (2 * x - π / 6) - m

theorem range_of_m_eq_interval : 
  (∃ a b : ℝ, (a < b ∧ 0 ≤ a ∧ b ≤ π / 2 ∧ f a m = 0 ∧ f b m = 0)) ↔ 
    m ∈ set.Ico (1 / 2 : ℝ) 1 := 
by
  sorry

end range_of_m_eq_interval_l261_261556


namespace hyperbola_and_line_l261_261803

-- Define conditions for hyperbola
def isHyperbolaCenteredAtOrigin (C : ℝ → ℝ → Prop) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, C x y ↔ (x^2 / a^2 - y^2 / b^2 = 1)

def hasRightFocus (C : ℝ → ℝ → Prop) (c : ℝ) : Prop :=
  let focus := (2 * real.sqrt 3 / 3, 0) in
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (c = 2 * real.sqrt 3 / 3) ∧
              (c^2 = a^2 + b^2) ∧ 
              C focus.1 focus.2

def hasAsymptotes (C : ℝ → ℝ → Prop) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x : ℝ, (C x (real.sqrt 3 * x) ∧ C x (-real.sqrt 3 * x))

-- Main Problem Statement
theorem hyperbola_and_line (C : ℝ → ℝ → Prop) (k : ℝ) :
  isHyperbolaCenteredAtOrigin C →
  hasRightFocus C (2 * real.sqrt 3 / 3) →
  hasAsymptotes C →
  (∀ x y, C x y ↔ (3*x^2 - y^2 = 1)) ∧ 
  (∀ A B, 
    (∃ x₁ x₂ y₁ y₂, 
    A = (x₁, y₁) ∧ B = (x₂, y₂) ∧ 
    y₁ = k * x₁ + 1 ∧ y₂ = k * x₂ + 1 ∧ 
    (x₁ * x₂ + y₁ * y₂ = 0)) → 
    k = 1 ∨ k = -1) :=
by
  sorry

end hyperbola_and_line_l261_261803


namespace arithmetic_progression_roots_geometric_progression_roots_harmonic_sequence_roots_l261_261697

-- Arithmetic Progression
theorem arithmetic_progression_roots (a b c : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 - x2 = x2 - x3 ∧ x1 + x2 + x3 = -a ∧ x1 * x2 + x2 * x3 + x1 * x3 = b ∧ -x1 * x2 * x3 = c) 
  ↔ (b = (2 * a^3 + 27 * c) / (9 * a)) :=
sorry

-- Geometric Progression
theorem geometric_progression_roots (a b c : ℝ) :
  (∃ x1 x2 x3 : ℝ, x2 / x1 = x3 / x2 ∧ x1 + x2 + x3 = -a ∧ x1 * x2 + x2 * x3 + x1 * x3 = b ∧ -x1 * x2 * x3 = c) 
  ↔ (b = a * c^(1/3)) :=
sorry

-- Harmonic Sequence
theorem harmonic_sequence_roots (a b c : ℝ) :
  (∃ x1 x2 x3 : ℝ, (x1 - x2) / (x2 - x3) = x1 / x3 ∧ x1 + x2 + x3 = -a ∧ x1 * x2 + x2 * x3 + x1 * x3 = b ∧ -x1 * x2 * x3 = c) 
  ↔ (a = (2 * b^3 + 27 * c) / (9 * b^2)) :=
sorry

end arithmetic_progression_roots_geometric_progression_roots_harmonic_sequence_roots_l261_261697


namespace greatest_integer_less_than_target_l261_261949

theorem greatest_integer_less_than_target :
  let S := (1 / (2! * 19!) + 1 / (3! * 18!) + 1 / (4! * 17!) + 1 / (5! * 16!) + 1 / (6! * 15!) +
            1 / (7! * 14!) + 1 / (8! * 13!) + 1 / (9! * 12!) + 1 / (10! * 11!))
  let M := S * (1! * 20!)
  ∃ m : ℕ, m = 499 ∧ m ≤ M / 100 ∧ (M / 100 < m + 1) := 
sorry

end greatest_integer_less_than_target_l261_261949


namespace rectangle_diagonal_l261_261425

theorem rectangle_diagonal (l b : ℝ) (h1 : 2 * (l + b) = 5 * b) (h2 : l * b = 216) :
    real.sqrt (l^2 + b^2) = 6 * real.sqrt 13 :=
by
  sorry

end rectangle_diagonal_l261_261425


namespace question_I_question_II_l261_261574

noncomputable def f (x : ℝ) : ℝ := (sin (2 * x - π / 6) + 2)

theorem question_I (k : ℤ) :
  ∀ x : ℝ, (k * π + π / 3 ≤ x) ∧ (x ≤ k * π + 5 * π / 6) → f x ≤ f (x + ε) := sorry

theorem question_II (A : ℝ) (b S : ℝ) (a c : ℝ := 2 * sqrt 3) (c := 4) :
  ∀ A : ℝ, A = π / 3 ∧ f(A) = 3 → b = 2 ∧ S = 2 * sqrt 3 := sorry

end question_I_question_II_l261_261574


namespace curve_equiv_A_inv_transformation_l261_261660

-- Define matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 2], ![2, 3]]

-- Define the target inverse matrix
def A_inv : Matrix (Fin 2) (Fin 2) ℤ := ![[-3, 2], ![2, -1]]

-- Define the curve transformation matrix
def C_prime_eq (x' y' : ℝ) : Prop := x' ^ 2 - y' ^ 2 = 1

-- Define the original curve equation
def C_eq (x y : ℝ) : Prop := 5 * x ^ 2 - 8 * x * y + 3 * y ^ 2 = 1

-- Main theorem statement
theorem curve_equiv_A_inv_transformation :
  -- First, A_inv should be the inverse of A
  (A.mul A_inv = 1 ∧ A_inv.mul A = 1) ∧
  -- Second, the transformation keeps curve_equivalence
  (∀ (x y : ℝ), C_eq x y ↔
              C_prime_eq ((-3:ℝ) * x + (2:ℝ) * y) ((2:ℝ) * x - y)) :=
by sorry

end curve_equiv_A_inv_transformation_l261_261660


namespace a_n_general_formula_S_n_sum_formula_l261_261279

-- Sequence definition according to given conditions
def a : ℕ → ℕ
| 1 := 3
| (n + 1) := 3 * a n - 4 * n

-- b_n definition as 2^n * a_n
def b (n : ℕ) : ℕ := 2^n * a n

-- Sum of first n terms of the sequence b_n
def S : ℕ → ℕ
| 0 := 0
| (n + 1) := S n + b (n + 1)

theorem a_n_general_formula (n : ℕ) : a n = 2 * n + 1 :=
by sorry

theorem S_n_sum_formula (n : ℕ) : S n = (2 * n - 1) * 2^(n + 1) + 2 :=
by sorry

end a_n_general_formula_S_n_sum_formula_l261_261279


namespace find_tangent_line_to_curves_l261_261710

noncomputable def tangent_line_to_curves_tangent (t : ℝ) : Prop :=
  let f : ℝ → ℝ := λ x, Real.exp x
  let g : ℝ → ℝ := λ x, -x ^ 2 / 4
  let tangent_line (t : ℝ) (x : ℝ) : ℝ := Real.exp t * (x - t) + Real.exp t
  in
  (e^t + t - 1 = 0) ∧
  (∀ x, tangent_line t x = -x ^ 2 / 4 → y = x + 1)

theorem find_tangent_line_to_curves : ∃ t, tangent_line_to_curves_tangent t := sorry

end find_tangent_line_to_curves_l261_261710


namespace min_variance_of_hours_worked_l261_261776

theorem min_variance_of_hours_worked :
  ∀ (x y: ℝ), (5 + x + 8 + 11 + y) / 5 = 8 → x + y = 16 → 
  ∃ v, v = (1/5) * ((5 - 8)^2 + (x - 8)^2 + (8 - 8)^2 + (11 - 8)^2 + (16 - x - 8)^2) ∧ v = 18 / 5 :=
by
  assume x y
  assume avg : (5 + x + 8 + 11 + y) / 5 = 8
  assume xy_eq : x + y = 16
  existsi (1/5) * ((5 - 8)^2 + (x - 8)^2 + (8 - 8)^2 + (11 - 8)^2 + (16 - x - 8)^2)
  split
  case left =>
    exact rfl
  case right =>
    simp
    sorry

end min_variance_of_hours_worked_l261_261776


namespace sum_of_coefficients_l261_261928

theorem sum_of_coefficients (a : ℕ → ℚ) :
  (∀ x : ℚ, (1 - 2 * x) ^ 9 = (∑ i in range 10, a i * x ^ i)) →
  (∑ i in range 10, a i) = -1 :=
by {
  intro h,
  specialize h 1,
  simp at h,
  sorry
}

end sum_of_coefficients_l261_261928


namespace problem_f_neg1_4_sum_l261_261934

def f (x : ℝ) : ℝ :=
  if x < 2 then -x^2 + 3 * x
  else 2 * x - 1

theorem problem_f_neg1_4_sum : f (-1) + f 4 = 3 := by
  sorry

end problem_f_neg1_4_sum_l261_261934


namespace x_intercept_of_line_l261_261371

theorem x_intercept_of_line : ∀ x y : ℝ, 2 * x + 3 * y = 6 → y = 0 → x = 3 :=
by
  intros x y h_line h_y_zero
  sorry

end x_intercept_of_line_l261_261371


namespace a_general_formula_S_n_formula_l261_261281

-- Given conditions
variable (a : ℕ → ℕ) (n : ℕ)

axiom a_1 : a 1 = 3
axiom a_recurrence : ∀ n : ℕ, a (n + 1) = 3 * a n - 4 * n

-- General formula for the sequence
theorem a_general_formula : ∀ n: ℕ, a n = 2 * n + 1 :=
by
  sorry

-- Given sequence b_n in terms of a_n
def b (n : ℕ) := 2^n * a n

-- Sum of the first n terms of b_n
def S (n : ℕ) := ∑ k in finset.range n, b k

-- Sum formula
theorem S_n_formula : ∀ n : ℕ, S n = (2*n - 1) * 2^(n + 1) + 2 :=
by
  sorry

end a_general_formula_S_n_formula_l261_261281


namespace perimeters_difference_l261_261015

def perimeter_first_figure (length : ℕ) (width : ℕ) : ℕ :=
  2 * (length + width)

def perimeter_second_figure (length1 : ℕ) (width1 : ℕ) (height_factor : ℕ) : ℕ :=
  2 * (length1 + 3 * width1)

theorem perimeters_difference :
  let length1 := 6;
  let width1 := 1;
  let length2 := 3;
  let width2 := 1;
  let height_factor := 3 in
  perimeter_first_figure length1 width1 - perimeter_second_figure length2 width2 height_factor = 2 := 
by
  sorry

end perimeters_difference_l261_261015


namespace national_park_sightings_l261_261604

def january_sightings : ℕ := 26

def february_sightings : ℕ := 3 * january_sightings

def march_sightings : ℕ := february_sightings / 2

def total_sightings : ℕ := january_sightings + february_sightings + march_sightings

theorem national_park_sightings : total_sightings = 143 := by
  sorry

end national_park_sightings_l261_261604


namespace paintable_area_correct_l261_261639

-- Defining lengths
def bedroom_length : ℕ := 15
def bedroom_width : ℕ := 11
def bedroom_height : ℕ := 9

-- Defining the number of bedrooms
def num_bedrooms : ℕ := 4

-- Defining the total area not to be painted per bedroom
def area_not_painted_per_bedroom : ℕ := 80

-- The total wall area calculation
def total_wall_area_per_bedroom : ℕ :=
  2 * (bedroom_length * bedroom_height) + 2 * (bedroom_width * bedroom_height)

-- The paintable wall area per bedroom calculation
def paintable_area_per_bedroom : ℕ :=
  total_wall_area_per_bedroom - area_not_painted_per_bedroom

-- The total paintable area across all bedrooms calculation
def total_paintable_area : ℕ :=
  paintable_area_per_bedroom * num_bedrooms

-- The theorem statement
theorem paintable_area_correct : total_paintable_area = 1552 := by
  sorry -- Proof is omitted

end paintable_area_correct_l261_261639


namespace arrange_PERCEPTION_l261_261898

theorem arrange_PERCEPTION : 
  let n := 10 
  let k_E := 2
  let k_P := 2
  let k_I := 2
  nat.factorial n / (nat.factorial k_E * nat.factorial k_P * nat.factorial k_I) = 453600 :=
by
  sorry

end arrange_PERCEPTION_l261_261898


namespace part_a_ctg_half_angle_sum_part_b_tg_half_angle_sum_l261_261783

theorem part_a_ctg_half_angle_sum (α β γ : ℝ) (a b c p r r_a r_b r_c : ℝ) 
  (hα : α + β + γ = π) :
  cot (α / 2) + cot (β / 2) + cot (γ / 2) = p / r :=
by 
sorry

theorem part_b_tg_half_angle_sum (α β γ : ℝ) (a b c p r r_a r_b r_c : ℝ)
  (hα : α + β + γ = π) :
  tan (α / 2) + tan (β / 2) + tan (γ / 2) = (a / r_a + b / r_b + c / r_c) / 2 :=
by
sorry

end part_a_ctg_half_angle_sum_part_b_tg_half_angle_sum_l261_261783


namespace smallest_solution_l261_261761

theorem smallest_solution : ∃ x : ℝ, x^4 - 50 * x^2 + 625 = 0 ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y :=
by {
  use -5,
  split,
  sorry, -- here would be the proof that (-5)^4 - 50 * (-5)^2 + 625 = 0
  intros y hy,
  sorry -- here would be the proof that for any y such that y^4 - 50 * y^2 + 625 = 0, -5 ≤ y
}

end smallest_solution_l261_261761


namespace tetrahedron_midpoint_distances_equal_l261_261620

structure Point :=
(x y z : ℝ)

structure Tetrahedron :=
(A B C D : Point)
(perpendicular : ∀ (P : Point), (P.x - A.x) * (B.x - A.x) + (P.y - A.y) * (B.y - A.y) + (P.z - A.z) * (B.z - A.z) = 0)

def distance (P Q : Point) : ℝ :=
(real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2 + (P.z - Q.z)^2))

def midpoint (P Q : Point) : Point :=
{ x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2, z := (P.z + Q.z) / 2 }

theorem tetrahedron_midpoint_distances_equal (T : Tetrahedron) (P : Point) :
  let M₁ := midpoint T.A T.C
      M₂ := midpoint T.B T.D
      M₃ := midpoint T.A T.D
      M₄ := midpoint T.B T.C in
  distance P M₁^2 + distance P M₂^2 = distance P M₃^2 + distance P M₄^2 :=
by sorry

end tetrahedron_midpoint_distances_equal_l261_261620


namespace eval_poly_at_roots_l261_261271

noncomputable def z := Complex.ofReal (Real.cos (2 * Real.pi / 2011)) + Complex.I * Complex.ofReal (Real.sin (2 * Real.pi / 2011))

noncomputable def P (x : ℂ) :=
  x^2008 + 3 * x^2007 + 6 * x^2006 + ∑ i in List.range (2006), (i + 1) * x^i + 2009 * 2010 / 2

theorem eval_poly_at_roots:
  ∏ i in Finset.range (2010), P (z ^ (i + 1)) = 2011^2009 * (1005^2011 - 1004^2011) :=
by
  sorry

end eval_poly_at_roots_l261_261271


namespace perception_arrangements_l261_261903

theorem perception_arrangements : 
  let n := 10
  let p := 2
  let e := 2
  let i := 2
  let r := 1
  let c := 1
  let t := 1
  let o := 1
  let n' := 1 
  (n.factorial / (p.factorial * e.factorial * i.factorial * r.factorial * c.factorial * t.factorial * o.factorial * n'.factorial)) = 453600 := 
by
  sorry

end perception_arrangements_l261_261903


namespace Abie_total_spent_l261_261093

variable (initial_bags : Nat) (shared_fraction : ℚ) (bought_bags : Nat) 
variable (coupon_bags : Nat) (initial_price_per_bag : ℚ) (coupon_price_fraction : ℚ)

def total_spent (initial_bags: Nat) (shared_fraction: ℚ) (bought_bags: Nat) 
                (coupon_bags: Nat) (initial_price_per_bag: ℚ) (coupon_price_fraction: ℚ) : ℚ :=
  let shared_bags := shared_fraction * initial_bags
  let after_sharing_bags := initial_bags - shared_bags
  let half_price := initial_price_per_bag / 2
  let cost_of_bought_bags := bought_bags * half_price
  let coupon_price := initial_price_per_bag * coupon_price_fraction
  let cost_of_coupon_bags := coupon_bags * coupon_price
  let initial_cost := initial_bags * initial_price_per_bag
  initial_cost + cost_of_bought_bags + cost_of_coupon_bags

theorem Abie_total_spent : 
  total_spent 20 (2/5) 18 4 2 (3/4) = 64 := by
  rfl -- this is just to check the expression is correct and equal to 64
  sorry -- since the proof is not required

end Abie_total_spent_l261_261093


namespace find_a_l261_261562

def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 1 then 2^x + 1 else x^2 + a * x

theorem find_a (a : ℝ) (h : f (f 0 a) a = 3 * a) : a = 4 := by
  sorry

end find_a_l261_261562


namespace triangle_product_equality_l261_261010

variables {A B C A₁ B₁ C₁ : Type}
variables [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] [EuclideanGeometry A₁] [EuclideanGeometry B₁] [EuclideanGeometry C₁]

def angle (a b c : Type) [EuclideanGeometry a] [EuclideanGeometry b] [EuclideanGeometry c] : Real := 
    sorry -- Definition for the angle

def length (x y : Type) [EuclideanGeometry x] [EuclideanGeometry y] : Real :=
    sorry -- Definition for the length

theorem triangle_product_equality 
  (ABC : Triangle A B C)
  (A₁B₁C₁ : Triangle A₁ B₁ C₁)
  (h₁ : angle A B C + angle A₁ B₁ C₁ = 180)
  (h₂ : angle B A C = angle B₁ A₁ C₁) :
  length A B * length A₁ B₁ + length A C * length A₁ C₁ = length B C * length B₁ C₁ := 
sorry

end triangle_product_equality_l261_261010


namespace bd_eq_ac_sin_alpha_l261_261618

-- Define a cyclic quadrilateral
structure CyclicQuadrilateral (A B C D : Type) :=
(angleB : ℝ) (angleD : ℝ) (angleA : ℝ)

-- Assume angles B and D are 90 degrees and angle A is alpha
def quadrilateral_conditions (A B C D : Type) (alpha : ℝ) :=
  CyclicQuadrilateral A B C D (real.pi / 2) (real.pi / 2) alpha

-- Lean theorem statement based on provided problem
theorem bd_eq_ac_sin_alpha (A B C D : Type) (α : ℝ) :
  quadrilateral_conditions A B C D α →
  ∃ (BD AC : ℝ), BD = AC * real.sin α :=
by
  -- Proof content to be filled here
  sorry

end bd_eq_ac_sin_alpha_l261_261618


namespace police_can_catch_gangster_l261_261622

structure Square :=
  (side_length : ℝ)
  (police_speed : ℝ)
  (gangster_speed : ℝ)

def can_catch (sq : Square) : Prop :=
  let u := sq.police_speed
  let v := sq.gangster_speed
  u > (1/3) * v

theorem police_can_catch_gangster (side_length : ℝ) (police_speed : ℝ) 
  (gangster_speed : ℝ) (h : police_speed > 1/3 * gangster_speed): 
  ∃ (police : ℝ × ℝ) (gangster : ℝ × ℝ), police ≠ gangster :=
begin
  let sq := Square.mk side_length police_speed gangster_speed,
  have h1 : can_catch sq, from h,
  sorry
end

end police_can_catch_gangster_l261_261622


namespace log_sqrt_7_of_343sqrt7_l261_261124

noncomputable def log_sqrt_7 (y : ℝ) : ℝ := 
  Real.log y / Real.log (Real.sqrt 7)

theorem log_sqrt_7_of_343sqrt7 : log_sqrt_7 (343 * Real.sqrt 7) = 4 :=
by
  sorry

end log_sqrt_7_of_343sqrt7_l261_261124


namespace large_number_divisible_by_12_l261_261867

theorem large_number_divisible_by_12 : (Nat.digits 10 ((71 * 10^168) + (72 * 10^166) + (73 * 10^164) + (74 * 10^162) + (75 * 10^160) + (76 * 10^158) + (77 * 10^156) + (78 * 10^154) + (79 * 10^152) + (80 * 10^150) + (81 * 10^148) + (82 * 10^146) + (83 * 10^144) + 84)).foldl (λ x y => x * 10 + y) 0 % 12 = 0 := 
sorry

end large_number_divisible_by_12_l261_261867


namespace at_most_100_return_times_l261_261737

theorem at_most_100_return_times 
  (total_soldiers : ℕ := 1000000) 
  (segments : ℕ := 100)
  (perm : equiv.perm (fin segments)) :
  (∀ i : fin segments, ∃ c : ℕ, perm^c i = i) →
  (countable (λ x : fin segments, classical.some (exists_cycle_length perm))) ≤ 100 :=
by
  sorry

end at_most_100_return_times_l261_261737


namespace sequence_properties_l261_261193

theorem sequence_properties (k : ℕ) (h : 0 < k) :
  (∀ n : ℕ, ∃ a : ℕ, a = a_n k n) ∧
  (∀ n : ℕ, 4 * k ∣ (a_n k (4 * n + 1) - 1)) :=
by
  let rec a_n : ℕ → ℕ → ℕ
  | k, 0     := 0
  | k, n + 1 := 2 * k * a_n k n + nat.sqrt ((4 * k^2 - 1) * a_n k n ^ 2 + 1)
  sorry

end sequence_properties_l261_261193


namespace Helly_theorem_l261_261736

theorem Helly_theorem (n : ℕ) (hn : n ≥ 3) 
(M : fin n → set ℝ) 
(hconv : ∀ i, convex (M i)) 
(hcommon : ∀ i j k : fin n, i ≠ j → j ≠ k → i ≠ k → 
  (∃ x, x ∈ M i ∧ x ∈ M j ∧ x ∈ M k)) : 
  ∃ x, ∀ i : fin n, x ∈ M i :=
sorry

end Helly_theorem_l261_261736


namespace nine_rooks_checkerboard_l261_261327

theorem nine_rooks_checkerboard :
  let num_ways_4x4 := 4.factorial
  let num_ways_5x5 := 5.factorial
  num_ways_4x4 * num_ways_5x5 = 2880 :=
by
  sorry

end nine_rooks_checkerboard_l261_261327


namespace number_of_non_hospitalized_smokers_l261_261226

theorem number_of_non_hospitalized_smokers (total_students : ℕ) (smoking_ratio : ℝ) (hospitalized_ratio : ℝ) (h_total : total_students = 300) (h_smoking : smoking_ratio = 0.40) (h_hospitalized : hospitalized_ratio = 0.70) : 
  (total_students * smoking_ratio).to_nat - ((total_students * smoking_ratio) * hospitalized_ratio).to_nat = 36 := 
by
  sorry

end number_of_non_hospitalized_smokers_l261_261226


namespace ratio_a_b_is_34_l261_261591

theorem ratio_a_b_is_34 (a b : ℝ) (h1 : (a + b) / 2 = 3 * real.sqrt (a * b)) (h2 : a > b ∧ b > 0) :
  a / b = 34 :=
sorry

end ratio_a_b_is_34_l261_261591


namespace prove_x_equals_3_l261_261077

-- Assume the following definitions:
variable {x y : ℝ} -- Defining distance variables x and y
variable person_walks_x_east : ℝ -- The distance walked east by the person
variable person_walks_y_new_direction : ℝ -- The distance walked in the new direction
variable distance_from_start : ℝ -- The final distance from the starting point

-- Define the conditions based on the problem
def walks_east (x : ℝ) : ℝ := x
def turns_right_120 (y : ℝ) : ℝ := y
def final_distance (distance : ℝ) : ℝ := distance

-- Statement to be proved
theorem prove_x_equals_3:
  (walks_east x = person_walks_x_east) →
  (turns_right_120 y = person_walks_y_new_direction) →
  (final_distance 3 = distance_from_start) →
  (distance_from_start = 3) →
  person_walks_x_east = 3 :=
by
  intros hx hy hd h3
  sorry -- Proof to be filled in

end prove_x_equals_3_l261_261077


namespace percent_correct_both_l261_261784

-- Definitions based on given conditions in the problem
def P_A : ℝ := 0.63
def P_B : ℝ := 0.50
def P_not_A_and_not_B : ℝ := 0.20

-- Definition of the desired result using the inclusion-exclusion principle based on the given conditions
def P_A_and_B : ℝ := P_A + P_B - (1 - P_not_A_and_not_B)

-- Theorem stating our goal: proving the probability of both answering correctly is 0.33
theorem percent_correct_both : P_A_and_B = 0.33 := by
  sorry

end percent_correct_both_l261_261784


namespace sum_approx_1832_l261_261765

noncomputable def sum_series (n : ℕ) : ℝ :=
  ∑ k in finset.range n, (3 / (↑k * (↑k + 3)))

theorem sum_approx_1832 : abs (sum_series 2023 - 1.832) < 0.001 :=
sorry

end sum_approx_1832_l261_261765


namespace integral1_proof_integral2_proof_l261_261104

noncomputable def integral1 : ℝ :=
  ∫ x in 0 .. 1, ∫ y in x .. 2 * x, (x - y + 1)

theorem integral1_proof : integral1 = 1 / 3 :=
by
  sorry

noncomputable def integral2 : ℝ :=
  ∫ y in 1 .. 4, ∫ x in x .. y, (y^3 / (x^2 + y^2))

theorem integral2_proof : integral2 = 14 * Real.pi / 3 :=
by
  sorry

end integral1_proof_integral2_proof_l261_261104


namespace tangent_line_equation_l261_261554

theorem tangent_line_equation (x y : ℝ) (hx : x = 1) (hy : y = 2) :
  ∃ (m b : ℝ), y = m * x + b ∧ y = 4 * x - 2 :=
by
  sorry

end tangent_line_equation_l261_261554


namespace final_number_appended_is_84_l261_261852

noncomputable def arina_sequence := "7172737475767778798081"

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def is_divisible_by_12 (n : ℕ) : Prop := n % 12 = 0

-- Define adding numbers to the sequence
def append_number (seq : String) (n : ℕ) : String := seq ++ n.repr

-- Create the full sequence up to 84 and check if it's divisible by 12
def generate_full_sequence : String :=
  let base_seq := arina_sequence
  let full_seq := append_number (append_number (append_number (append_number (append_number (append_number (append_number (append_number (append_number arina_sequence 82) 83) 84))) 85) 86) 87) 88 
  full_seq

theorem final_number_appended_is_84 : (∃ seq : String, is_divisible_by_12(seq.to_nat) ∧ seq.ends_with "84") := 
by
  sorry

end final_number_appended_is_84_l261_261852


namespace find_general_formula_and_m_values_l261_261537

variable {a : ℕ → ℝ}
variable (S : ℕ → ℝ)
variable {m : ℝ}
variable {x : ℝ}

/-- Given conditions -/
axiom a1 : a 1 = 2
axiom common_difference_not_zero : ∃ d : ℝ, d ≠ 0 ∧ a 2 = a 1 + d ∧ a 5 = a 1 + 4 * d
axiom bn_def : ∀ n : ℕ, S n = S n-1 + (8 / (a n * a (n + 1)))

noncomputable def satisfies_geometric_sequence := ∀ {a : ℕ → ℝ} (d : ℝ), a 2 = a 1 + d ∧ a 5 = a 1 + 4*d → d = 4

/-- Proof Problem -/
theorem find_general_formula_and_m_values 
  (d : ℝ)
  (common_difference : ∀ n : ℕ, a n = 4 * n - 2)
  (Sn_lt_1 : ∀ n : ℕ, S n < 1)
  (geq_one : ∀ n : ℕ, x ∈ set.Icc 2 4 → x^2 + m*x + m ≥ S n)
  : m ≥ -1 :=
sorry

end find_general_formula_and_m_values_l261_261537


namespace sin_expression_calculation_l261_261366

theorem sin_expression_calculation :
  (Real.sin (40 * Real.pi / 180) * Real.sin (10 * Real.pi / 180) + Real.sin (50 * Real.pi / 180) * Real.sin (100 * Real.pi / 180)) = (Real.sqrt 3 / 2) :=
by
  sorry

end sin_expression_calculation_l261_261366


namespace simplify_expression_is_3_l261_261274

noncomputable def simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 3) : ℝ :=
  1 / (y^2 + z^2 - x^2) + 1 / (x^2 + z^2 - y^2) + 1 / (x^2 + y^2 - z^2)

theorem simplify_expression_is_3 (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 3) :
  simplify_expression x y z hx hy hz h = 3 :=
  sorry

end simplify_expression_is_3_l261_261274


namespace mean_cars_l261_261774

theorem mean_cars (a b c d e : ℝ) (h1 : a = 30) (h2 : b = 14) (h3 : c = 14) (h4 : d = 21) (h5 : e = 25) : 
  (a + b + c + d + e) / 5 = 20.8 :=
by
  -- The proof will be provided here
  sorry

end mean_cars_l261_261774


namespace rhombus_area_l261_261040

-- Define the coordinates of the vertices of the rhombus
def v1 : ℝ × ℝ := (0, 3.5)
def v2 : ℝ × ℝ := (11, 0)
def v3 : ℝ × ℝ := (0, -3.5)
def v4 : ℝ × ℝ := (-11, 0)

-- Define the lengths of the diagonals based on the vertices
def d1 : ℝ := dist v1 v3  -- Length of diagonal along y-axis
def d2 : ℝ := dist v2 v4  -- Length of diagonal along x-axis

-- Prove the area of the rhombus
theorem rhombus_area : (d1 * d2) / 2 = 77 := by
  -- Length of the diagonal along y-axis is 3.5 - (-3.5) = 7
  have d1_eq : d1 = 7 := by sorry  -- Proof that d1 is 7 units
  -- Length of the diagonal along x-axis is 11 - (-11) = 22
  have d2_eq : d2 = 22 := by sorry  -- Proof that d2 is 22 units
  -- Calculate the area
  calc
    (d1 * d2) / 2 = (7 * 22) / 2 : by
      rw [d1_eq, d2_eq]
    ... = 154 / 2 : by 
      norm_num
    ... = 77 : by 
      norm_num

end rhombus_area_l261_261040


namespace inequality_proof_l261_261301

variable {x y z : ℝ}

theorem inequality_proof (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_prod : x * y * z = 1) :
  (x / (y + z) + y / (x + z) + z / (x + y)) ≤ (x * Real.sqrt(x)) / 2 + (y * Real.sqrt(y)) / 2 + (z * Real.sqrt(z)) / 2 :=
by
  sorry

end inequality_proof_l261_261301


namespace complement_of_M_l261_261147

noncomputable def U : Set ℝ := Set.univ
noncomputable def M : Set ℝ := {a | a ^ 2 - 2 * a > 0}
noncomputable def C_U_M : Set ℝ := U \ M

theorem complement_of_M :
  C_U_M = {a | 0 ≤ a ∧ a ≤ 2} :=
by
  sorry

end complement_of_M_l261_261147


namespace sequence_a_n_100_l261_261216

theorem sequence_a_n_100 : ∃ (a : ℕ → ℕ), a 1 = 2 ∧ (∀ n ≥ 1, a (n + 1) = a n + 2 * n) ∧ a 100 = 9902 :=
begin
  sorry
end

end sequence_a_n_100_l261_261216


namespace number_of_ordered_pairs_l261_261115

theorem number_of_ordered_pairs :
  {p : ℕ × ℕ // (p.1 > 0 ∧ p.2 > 0) ∧ (p.1^2 - 4 * p.2 < 0) ∧ (p.2^2 - 4 * p.1 < 0)}.card = 2 :=
sorry

end number_of_ordered_pairs_l261_261115


namespace roots_f_in_interval_l261_261412

open Real

noncomputable def f : ℝ → ℝ := sorry -- assume f meets the given condition in the problem

lemma f_symmetric_1 (x : ℝ) : f(3 + x) = f(3 - x) :=
sorry -- given in the problem

lemma f_symmetric_2 (x : ℝ) : f(9 + x) = f(9 - x) :=
sorry -- given in the problem

axiom f_at_1_zero : f 1 = 0 -- given in the problem

theorem roots_f_in_interval : 
  (finset.card (finset.filter (λ x : ℝ, f x = 0) (finset.Icc (-1000 : ℝ) 1000))) ≥ 334 :=
sorry -- to prove

end roots_f_in_interval_l261_261412


namespace solution_set_of_inequality_l261_261543

variable {a x : ℝ}

theorem solution_set_of_inequality (h : 2 * a + 1 < 0) : 
  {x : ℝ | x^2 - 4 * a * x - 5 * a^2 > 0} = {x | x < 5 * a ∨ x > -a} := by
  sorry

end solution_set_of_inequality_l261_261543


namespace solution_system_of_equations_l261_261353

theorem solution_system_of_equations :
  ∃ x y z : ℝ, 
    (x + y = 5) ∧ 
    (y + z = -1) ∧ 
    (x + z = -2) ∧
    (x = 2) ∧
    (y = 3) ∧
    (z = -4) :=
by
  exists 2
  exists 3
  exists -4
  split; trivial; split; trivial; split; trivial; split; trivial; split; trivial
  sorry -- complete the rest with appropriate steps and justifications if necessary.

end solution_system_of_equations_l261_261353


namespace appended_number_divisible_by_12_l261_261859

theorem appended_number_divisible_by_12 :
  ∃ N, (N = 88) ∧ (∀ n, n ∈ finset.range N \ 71 → (let large_number := (list.range (N + 1)).filter (λ x, 71 ≤ x ∧ x ≤ N) in
       (list.foldr (λ a b, a * 100 + b) 0 large_number) % 12 = 0)) :=
by
  sorry

end appended_number_divisible_by_12_l261_261859


namespace notebook_cost_l261_261219

theorem notebook_cost (s : ℕ) (c : ℕ) (n : ℕ) :
    s = 42 → (∃ s', s' > 21 ∧ s' * c * n = 3213 ∧ n > 1 ∧ c > n) → c = 17 :=
begin
  -- Proof goes here
  sorry
end

end notebook_cost_l261_261219


namespace find_a1_l261_261945

variable (a : ℕ → ℕ)
variable (q : ℕ)
variable (h_q_pos : 0 < q)
variable (h_a2a6 : a 2 * a 6 = 8 * a 4)
variable (h_a2 : a 2 = 2)

theorem find_a1 :
  a 1 = 1 :=
by
  sorry

end find_a1_l261_261945


namespace calculate_G51_l261_261581

noncomputable def G : ℕ → ℕ
| 1 := 3
| (n+1) := (3 * G n + 2) / 2

theorem calculate_G51 : G 51 = 70349216 := 
sorry

end calculate_G51_l261_261581


namespace honor_students_count_l261_261000

noncomputable def G : ℕ := 13
noncomputable def B : ℕ := 11
def E_G : ℕ := 3
def E_B : ℕ := 4

theorem honor_students_count (h1 : G + B < 30) 
    (h2 : (E_G : ℚ) / G = 3 / 13) 
    (h3 : (E_B : ℚ) / B = 4 / 11) :
    E_G + E_B = 7 := 
sorry

end honor_students_count_l261_261000


namespace occurs_1992_once_l261_261144

def largestOddDivisor (n : ℕ) : ℕ :=
  if n % 2 = 1 then n else largestOddDivisor (n / 2)

def f (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 + n / largestOddDivisor n else 2 * ((n + 1) / 2)

def a : ℕ → ℕ
| 1 => 1
| (n + 1) => f (a n)

theorem occurs_1992_once (n : ℕ) : ∃ k : ℕ, a k = 1992 ∧ (∀ m, a m = 1992 → m = k) :=
sorry

end occurs_1992_once_l261_261144


namespace BD_perpendicular_to_EF_l261_261310

variables {α β : Plane}
variables {EF : Line}
variables {A B C D : Point}
variables {AB CD : Line}

-- Given conditions
axiom AB_perpendicular_to_α : AB ⊥ α
axiom CD_perpendicular_to_β : CD ⊥ β
axiom feet_B : foot AB α = B
axiom feet_D : foot CD β = D
axiom AC_perpendicular_to_α : (C - A) ⊥ α
axiom angle_AC_with_α_eq_angle_AC_with_β : angle (C - A) α = angle (C - A) β
axiom projection_AC_on_β_lies_on_BD : proj (C - A) β = BD ∨ proj (C - A) β = -BD
axiom AC_parallel_to_EF : (C - A) ∥ EF

-- Conclusion to be proven
theorem BD_perpendicular_to_EF : (B - D) ⊥ EF :=
sorry

end BD_perpendicular_to_EF_l261_261310


namespace max_value_abs_expression_l261_261157

noncomputable def circle_eq (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 1

theorem max_value_abs_expression (x y : ℝ) (h : circle_eq x y) : 
  ∃ t : ℝ, |3 * x + 4 * y - 3| = t ∧ t ≤ 8 :=
sorry

end max_value_abs_expression_l261_261157


namespace pure_imaginary_solution_l261_261335

theorem pure_imaginary_solution (m : ℝ) (h₁ : m^2 - m - 4 = 0) (h₂ : m^2 - 5 * m - 6 ≠ 0) :
  m = (1 + Real.sqrt 17) / 2 ∨ m = (1 - Real.sqrt 17) / 2 :=
sorry

end pure_imaginary_solution_l261_261335


namespace time_to_send_data_in_minutes_l261_261469

def blocks := 100
def chunks_per_block := 256
def transmission_rate := 100 -- chunks per second
def seconds_per_minute := 60

theorem time_to_send_data_in_minutes :
    (blocks * chunks_per_block) / transmission_rate / seconds_per_minute = 4 := by
  sorry

end time_to_send_data_in_minutes_l261_261469


namespace sequence_sum_divisibility_l261_261402

theorem sequence_sum_divisibility (a : ℕ → ℕ)
  (h1 : ∀ n, a n ∣ a (n+5))
  (h2 : ∀ n, a (n+8) ∣ a n) :
  (∑ i in Finset.range 100, a i) = (∑ i in Finset.range 100, a (2020 - 100 + i)) :=
sorry

end sequence_sum_divisibility_l261_261402


namespace correct_statements_l261_261181

theorem correct_statements (a b : ℝ) (p q : Prop):
  ¬ (∀ a b : ℝ, ∃! x : ℝ, a*x + b = 0) ∧
  ( (p ∧ q → p ∨ q) ∧ 
    ¬ (if (even a ∧ even b) then even (a + b) else true) ∧ 
    (¬ (∃ x : ℝ, sin x ≤ 1) = (∀ x : ℝ, sin x > 1)) ) :=
by
  sorry

end correct_statements_l261_261181


namespace smallest_number_after_operations_l261_261190

theorem smallest_number_after_operations :
  ∀ (initial_number : ℕ), (initial_number = 1836549) →
  (∀ (n : ℕ), swap_adjacent_and_decrement n initial_number → swap_adjacent_and_decrement (n-2) initial_number → swap_adjacent_and_decrement (n-2) 1010101) :=
sorry

end smallest_number_after_operations_l261_261190


namespace total_duration_in_minutes_l261_261101

theorem total_duration_in_minutes 
  (tv_hours : ℕ) (tv_minutes_per_hour : ℕ)
  (vg_hours : ℕ) (vg_additional_minutes : ℕ) (vg_minutes_per_hour : ℕ)
  (walk_hours : ℕ) (walk_additional_minutes : ℕ) (walk_minutes_per_hour : ℕ)
  (tv_duration : ℕ := tv_hours * tv_minutes_per_hour)
  (vg_duration : ℕ := vg_hours * vg_minutes_per_hour + vg_additional_minutes)
  (walk_duration : ℕ := walk_hours * walk_minutes_per_hour + walk_additional_minutes) :
  tv_hours = 4 ∧ tv_minutes_per_hour = 60 ∧
  vg_hours = 2 ∧ vg_additional_minutes = 30 ∧ vg_minutes_per_hour = 60 ∧
  walk_hours = 1 ∧ walk_additional_minutes = 45 ∧ walk_minutes_per_hour= 60 →
  tv_duration + vg_duration + walk_duration = 495 :=
by
  intros h
  cases h
  rfl


end total_duration_in_minutes_l261_261101


namespace line_containing_chord_l261_261592

variable {x y x₁ y₁ x₂ y₂ : ℝ}

def ellipse_eq (x y : ℝ) : Prop := (x^2 / 9 + y^2 / 4 = 1)

def midpoint_condition (x₁ x₂ : ℝ) (y₁ y₂ : ℝ) : Prop := 
  (x₁ + x₂ = 2) ∧ (y₁ + y₂ = 2)

theorem line_containing_chord (h₁ : ellipse_eq x₁ y₁) 
                               (h₂ : ellipse_eq x₂ y₂) 
                               (hmp : midpoint_condition x₁ x₂ y₁ y₂)
    : 4 * 1 + 9 * 1 - 13 = 0 := 
sorry

end line_containing_chord_l261_261592


namespace dataSet_properties_l261_261344

noncomputable def dataSet : List ℝ := [50, 85, 82.5, 30, 70, 200, 75, 95, 55]

def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

def median (l : List ℝ) : ℝ :=
  let sorted := l.qsort (· < ·)
  sorted.get! (sorted.length / 2)

def mode (l : List ℝ) : Option ℝ :=
  l.foldr (λ x acc => 
    let count := l.count x 
    match acc with
    | none => some x
    | some (y, cnt) => 
      if count > cnt then 
        some (x, count)
      else acc
  ) none |>.map Prod.fst

theorem dataSet_properties (x : ℝ) 
  (h1 : mean dataSet = x)
  (h2 : median dataSet = x)
  (h3 : mode dataSet = some x) :
  x = 82.5 := by 
  sorry

end dataSet_properties_l261_261344


namespace intersection_point_l261_261135

def line_parametric (t : ℝ) : ℝ × ℝ × ℝ :=
  (3 + 2 * t, -1 + 3 * t, -3 + 2 * t)

def on_plane (x y z : ℝ) : Prop :=
  3 * x + 4 * y + 7 * z - 16 = 0

theorem intersection_point : ∃ t, line_parametric t = (5, 2, -1) ∧ on_plane 5 2 (-1) :=
by
  use 1
  sorry

end intersection_point_l261_261135


namespace range_of_a_l261_261580

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ¬ (ax^2 - ax + 1 ≤ 0)) ↔ 0 ≤ a ∧ a < 4 :=
by
  sorry

end range_of_a_l261_261580


namespace find_certain_number_l261_261722

theorem find_certain_number (x y : ℝ)
  (h1 : (28 + x + 42 + y + 104) / 5 = 62)
  (h2 : (48 + 62 + 98 + 124 + x) / 5 = 78) :
  y = 78 :=
by
  sorry

end find_certain_number_l261_261722


namespace exists_nat_const_general_term_l261_261789
open Classical

-- First, define the sequence {a_n} with the given recurrence relation
def a_seq (x y : ℝ) : ℕ → ℝ
| 0       := x
| 1       := y
| (n + 2) := (a_seq n * a_seq (n + 1) + 1) / (a_seq n + a_seq (n + 1))

-- Define Fibonacci numbers
def fib : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

-- Question 1: For which real numbers x and y does there always exist a positive integer n_0 such that a_n is constant for n ≥ n_0?
theorem exists_nat_const (x y : ℝ) (h1 : abs y = 1) (h2 : x ≠ -y) : ∃ n0 : ℕ, ∀ n : ℕ, n ≥ n0 → a_seq x y n = a_seq x y n0 := sorry

-- Question 2: General term for a_n 
theorem general_term (x y : ℝ) (n : ℕ) :
  a_seq x y n =
  (x + 1)^(fib (n-2)) * (y + 1)^(fib (n-1)) + (x - 1)^(fib (n-2)) * (y - 1)^(fib (n-1)) /
  (x + 1)^(fib (n-2)) * (y + 1)^(fib n) - (x - 1)^(fib (n-2)) * (y - 1)^(fib n) := sorry

end exists_nat_const_general_term_l261_261789


namespace reflect_coordinates_l261_261718

variables {m b : ℝ}

theorem reflect_coordinates (m b : ℝ) : 
  let x1 : ℝ := 2
  let y1 : ℝ := 3
  let x2 : ℝ := 10
  let y2 : ℝ := 7
  let midpoint_x : ℝ := (x1 + x2) / 2 
  let midpoint_y : ℝ := (y1 + y2) / 2 
  let midpoint : ℝ × ℝ := (midpoint_x, midpoint_y)
  in (midpoint_y = m * midpoint_x + b) → m + b = 15 :=
by
  sorry

end reflect_coordinates_l261_261718


namespace coefficient_of_term_in_expansion_l261_261471

theorem coefficient_of_term_in_expansion :
  let a b c : ℝ in
  (∑' (i j k : ℕ) (h : i + j + k = 6), (if (i = 1 ∧ j = 2 ∧ k = 3) then (6.choose 1) * (5.choose 2) * (3.choose 3) else 0)) = 60 :=
by
  sorry

end coefficient_of_term_in_expansion_l261_261471


namespace range_of_expression_l261_261648

theorem range_of_expression (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 ≤ β ∧ β ≤ π / 2) :
  -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π :=
sorry

end range_of_expression_l261_261648


namespace ratio_of_a_to_b_l261_261785

variable (a b c d : ℝ)

theorem ratio_of_a_to_b (h1 : c = 0.20 * a) (h2 : c = 0.10 * b) : a = (1 / 2) * b :=
by
  sorry

end ratio_of_a_to_b_l261_261785


namespace sum_of_remainders_correct_l261_261424

def sum_of_remainders : ℕ :=
  let remainders := [43210 % 37, 54321 % 37, 65432 % 37, 76543 % 37, 87654 % 37, 98765 % 37]
  remainders.sum

theorem sum_of_remainders_correct : sum_of_remainders = 36 :=
by sorry

end sum_of_remainders_correct_l261_261424


namespace estimate_distance_to_lightning_l261_261294

noncomputable def speed_of_sound : ℝ := 1088 -- feet per second
noncomputable def time_sound_travelled : ℝ := 10 -- seconds
noncomputable def feet_per_mile : ℝ := 5280 -- feet/mile

theorem estimate_distance_to_lightning : 
  (Real.floor(((speed_of_sound * time_sound_travelled) / feet_per_mile + 0.25) * 2) / 2) = 2 := 
by { sorry }

end estimate_distance_to_lightning_l261_261294


namespace geometric_sequence_root_l261_261168

theorem geometric_sequence_root:
  (∃ (a : ℕ → ℝ), (geomseq a) ∧ (quadratic_roots a 3 7 68 256) ∧ (a 4 = 8)) →
  (∃ (a : ℕ → ℝ), a 6 = 32) := by
  sorry

-- Additional definitions that might be required
def geomseq (a : ℕ → ℝ) : Prop := ∃ r, ∀ n, a (n + 1) = r * a n

def quadratic_roots (a : ℕ → ℝ) (m n : ℕ) (b c : ℝ) : Prop :=
  a m = a n ∧ a m^2 - b * a m + c = 0 ∧ a n^2 - b * a n + c = 0

end geometric_sequence_root_l261_261168


namespace small_boxes_count_l261_261808

theorem small_boxes_count (chocolates_per_box : ℕ) (total_chocolates : ℕ) 
    (h_chocolates_per_box : chocolates_per_box = 25)
    (h_total_chocolates : total_chocolates = 475) : 
    total_chocolates / chocolates_per_box = 19 :=
by
  rw [h_chocolates_per_box, h_total_chocolates]
  norm_num
  sorry

end small_boxes_count_l261_261808


namespace second_discount_percentage_is_10_l261_261730

-- Define the given conditions as constants
def original_price : ℝ := 495
def first_discount_percentage : ℝ := 0.15
def sale_price_after_discounts : ℝ := 378.675

-- Define the calculation of the first discount and the price after the first discount
def first_discount_amount : ℝ := original_price * first_discount_percentage
def price_after_first_discount : ℝ := original_price - first_discount_amount

-- Define the theorem stating the second discount percentage is 10%
theorem second_discount_percentage_is_10 : 
  ∃ D : ℝ, 
  sale_price_after_discounts = price_after_first_discount - (price_after_first_discount * D / 100) 
  ∧ D = 10 :=
begin
  -- Calculation and verification go here
  sorry
end

end second_discount_percentage_is_10_l261_261730


namespace vanessa_deleted_files_l261_261367

theorem vanessa_deleted_files (initial_music_files : ℕ) (initial_video_files : ℕ) (files_left : ℕ) (files_deleted : ℕ) :
  initial_music_files = 13 → initial_video_files = 30 → files_left = 33 → 
  files_deleted = (initial_music_files + initial_video_files) - files_left → files_deleted = 10 :=
by
  sorry

end vanessa_deleted_files_l261_261367


namespace swim_back_distance_l261_261419

theorem swim_back_distance 
    (swimming_speed_still : ℝ)
    (water_speed : ℝ)
    (time : ℝ)
    (effective_speed := swimming_speed_still - water_speed)
    (distance := effective_speed * time) :
    swimming_speed_still = 4 → water_speed = 2 → time = 3 → distance = 6 := 
by 
    intros h1 h2 h3; 
    rw [h1, h2, h3]; 
    exact rfl

end swim_back_distance_l261_261419


namespace top_black_second_red_probability_l261_261812

-- Define the problem conditions in Lean
def num_standard_cards : ℕ := 52
def num_jokers : ℕ := 2
def num_total_cards : ℕ := num_standard_cards + num_jokers

def num_black_cards : ℕ := 26
def num_red_cards : ℕ := 26

-- Lean statement
theorem top_black_second_red_probability :
  (num_black_cards / num_total_cards * num_red_cards / (num_total_cards - 1)) = 338 / 1431 := by
  sorry

end top_black_second_red_probability_l261_261812


namespace final_remainder_l261_261509

def F (s : List ℕ) : ℕ :=
  ∑ i in Finset.range (s.length - 1), (-1)^(i + 1) * (s[i] - s[i + 1])^2

def S : List ℕ := (List.range 1000).map (λ i, 2^(i + 1))

def all_subsequences (l : List ℕ) : List (List ℕ) :=
  List.foldr (λ x acc, acc ++ (acc.map (λ xs, x :: xs))) [[]] l

def R : ℕ :=
  ∑ m in all_subsequences S, F m

theorem final_remainder : R % 1000 = 500 := 
  sorry

end final_remainder_l261_261509


namespace expression_simplification_l261_261886

theorem expression_simplification :
  (2 + 3) * (2^3 + 3^3) * (2^9 + 3^9) * (2^27 + 3^27) = 3^41 - 2^41 := 
sorry

end expression_simplification_l261_261886


namespace number_of_lineups_case1_number_of_lineups_case2_l261_261062

-- Assuming the required definitions and conditions
noncomputable def team := {veterans: ℕ, new_players: ℕ}
noncomputable def conditions (team: team) := 
  (team.veterans = 7 ∧ team.new_players = 5) ∧
  true -- specific veteran must play, handled in the question setup,
       -- two specific new players cannot play, handled in the question setup

-- First proof problem statement
theorem number_of_lineups_case1 (player_1_out: Prop) (player_2_out: Prop) : 
  ∃ (lineups: ℕ), lineups = 126 :=
begin
  sorry
end

-- Second proof problem statement
theorem number_of_lineups_case2 (forwards: ℕ) (guards: ℕ) (A_and_B: Prop) :
  ∃ (lineups: ℕ), lineups = 636 :=
begin
  sorry
end

end number_of_lineups_case1_number_of_lineups_case2_l261_261062


namespace arrange_PERCEPTION_l261_261900

theorem arrange_PERCEPTION : 
  let n := 10 
  let k_E := 2
  let k_P := 2
  let k_I := 2
  nat.factorial n / (nat.factorial k_E * nat.factorial k_P * nat.factorial k_I) = 453600 :=
by
  sorry

end arrange_PERCEPTION_l261_261900


namespace find_x_l261_261207

theorem find_x (x : ℕ) : 8000 * 6000 = x * 10^5 → x = 480 := by
  sorry

end find_x_l261_261207


namespace real_solutions_eq_l261_261261

noncomputable def greatest_int_le (a : ℝ) : ℤ :=
  Int.floor a

theorem real_solutions_eq (x : ℝ) : 
  (x ^ (2 * greatest_int_le x) = 2022) ↔ 
  (x = (Real.sqrt 2022 / 2022) ∨ x = Real.root 6 2022) :=
by
  sorry

end real_solutions_eq_l261_261261


namespace last_appended_number_is_84_l261_261842

theorem last_appended_number_is_84 : 
  ∃ N : ℕ, 
    let s := "7172737475767778798081" ++ (String.intercalate "" (List.map toString [82, 83, 84])) in
    (N = 84) ∧ (s.toNat % 12 = 0) :=
by
  sorry

end last_appended_number_is_84_l261_261842


namespace min_value_of_expression_min_value_achieved_l261_261937

theorem min_value_of_expression (x : ℝ) (h : x > 0) : 
  (x + 3 / (x + 1)) ≥ 2 * Real.sqrt 3 - 1 := 
sorry

theorem min_value_achieved (x : ℝ) (h : x = Real.sqrt 3 - 1) : 
  (x + 3 / (x + 1)) = 2 * Real.sqrt 3 - 1 := 
sorry

end min_value_of_expression_min_value_achieved_l261_261937


namespace C1_polar_length_segment_AB_l261_261232

-- Definition of the parametric equations as conditions
def C1_parametric (θ : ℝ) : ℝ × ℝ :=
  (3 + 3 * Real.cos θ, 3 * Real.sin θ)

-- Definition of the polar equations as conditions
def C2_polar (θ : ℝ) : ℝ :=
  Real.sqrt 3 * Real.sin θ + Real.cos θ

def C3_polar (θ : ℝ) : Prop :=
  θ = Real.pi / 3

-- Translate (I) as a theorem to prove the polar equation of C1
theorem C1_polar (θ : ℝ) : ∃ ρ, ρ = 6 * Real.cos θ :=
  sorry -- proof to be added

-- Translate (II) as a theorem to prove the length of segment AB is 1
theorem length_segment_AB : 
  let ρ1 := 6 * Real.cos (Real.pi / 3) in
  let ρ2 := Real.sqrt 3 * Real.sin (Real.pi / 3) + Real.cos (Real.pi / 3) in
  | ρ1 - ρ2 | = 1 :=
  sorry -- proof to be added

end C1_polar_length_segment_AB_l261_261232


namespace sine_eq_neg_half_l261_261988

theorem sine_eq_neg_half (x : ℝ) (cond : 0 ≤ x ∧ x < 360) : 
nat.card {x | ∃ x, (0 ≤ x ∧ x < 360 ∧ Real.sin (x * Real.pi / 180) = -0.5)} = 2 :=
sorry

end sine_eq_neg_half_l261_261988


namespace reflection_add_m_b_l261_261716

-- Definitions for the positions of points A and B
def Point := ℝ × ℝ
def A : Point := (2, 3)
def B : Point := (10, 7)

-- Definition for the reflection line equation parameters
variables (m b : ℝ)

-- Lean theorem statement
theorem reflection_add_m_b : 
  (∃ (m b : ℝ), ∀ (p : Point), 
    let A' := (2 * 6 - p.1, 2 * 5 - p.2) in  -- A' is the reflection of A
    A' = B ∧ m + b = 15) :=
sorry

end reflection_add_m_b_l261_261716


namespace grocery_delivery_amount_l261_261828

theorem grocery_delivery_amount (initial_savings final_price trips : ℕ) 
(fixed_charge : ℝ) (percent_charge : ℝ) (total_saved : ℝ) : 
  initial_savings = 14500 →
  final_price = 14600 →
  trips = 40 →
  fixed_charge = 1.5 →
  percent_charge = 0.05 →
  total_saved = final_price - initial_savings →
  60 + percent_charge * G = total_saved →
  G = 800 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end grocery_delivery_amount_l261_261828


namespace trigonometric_identity_l261_261955

def α_in_second_quadrant (α : ℝ) : Prop := π/2 < α ∧ α < π

def point_on_terminal_side (P : ℝ × ℝ) (x : ℝ) (α : ℝ) : Prop := 
  P = (x, sqrt 5)

def cos_relation (x α : ℝ) : Prop :=
  cos α = (sqrt 2 / 4) * x

theorem trigonometric_identity (α x : ℝ) (P : ℝ × ℝ) 
  (h1 : α_in_second_quadrant α) 
  (h2 : point_on_terminal_side P x α) 
  (h3 : cos_relation x α) :
  4 * cos (α + π / 2) - 3 * tan α = sqrt 15 - sqrt 10 :=
sorry

end trigonometric_identity_l261_261955


namespace equal_numbers_on_cards_for_n_gt_1_l261_261410

theorem equal_numbers_on_cards_for_n_gt_1 (n : ℕ) (h : n > 1) 
    (nums : Fin n → ℕ) 
    (H : ∀ i j, ∃ k k_nums, (nums i + nums j) / 2 = Real.geom_mean k_nums) :
    ∀ i j, nums i = nums j := 
sorry

end equal_numbers_on_cards_for_n_gt_1_l261_261410


namespace total_fish_correct_l261_261876

def Billy_fish : ℕ := 10
def Tony_fish : ℕ := 3 * Billy_fish
def Sarah_fish : ℕ := Tony_fish + 5
def Bobby_fish : ℕ := 2 * Sarah_fish
def Jenny_fish : ℕ := Bobby_fish - 4
def total_fish : ℕ := Billy_fish + Tony_fish + Sarah_fish + Bobby_fish + Jenny_fish

theorem total_fish_correct : total_fish = 211 := by
  sorry

end total_fish_correct_l261_261876


namespace max_f_max_a_sub_b_solutions_ffx_eq_x_l261_261185

-- Define the function f as given in the problem
def f (x : ℝ) : ℝ := 1

-- Proof per Part I: The maximum of the function f
theorem max_f : ∀ x : ℝ, f x = 1 :=
by
  intro x
  unfold f
  rfl

-- Proof per Part II: Show that the maximum value of a - b is e^2
theorem max_a_sub_b (a b : ℝ) (h_tangent : ∀ t : ℝ, y = a * x + b) : ∃ t : ℝ, (a - b) ≤ Real.exp 2 :=
sorry

-- Proof per Part III: Show that solutions to f[f(x)] = x are x = 0 and x = 1
theorem solutions_ffx_eq_x : ∀ x : ℝ, f (f x) = x ↔ x = 0 ∨ x = 1 :=
by
  intro x
  split
  {
    intro h
    unfold f at h
    have : x = 1 := by
      simp at h
      exact h
    right
    exact this
  }
  {
    intro h
    cases h
    {
      unfold f
      simp
    }
    {
      unfold f
      simp
    }
  }

end max_f_max_a_sub_b_solutions_ffx_eq_x_l261_261185


namespace incorrect_value_used_in_initial_calculation_l261_261343

-- Definitions inferred from the conditions
def mean1 : ℝ := 250
def mean2 : ℝ := 251
def correct_value : ℝ := 165

-- The problem statement in Lean
theorem incorrect_value_used_in_initial_calculation :
  ∃ (v_incorrect : ℝ), 
    let total_initial := 30 * mean1 in
    let total_correct := 30 * mean2 in
    let sum_diff := total_correct - total_initial in
    v_incorrect = correct_value + sum_diff :=
by
  -- The proof will be provided here
  sorry

end incorrect_value_used_in_initial_calculation_l261_261343


namespace calculate_percentage_passed_l261_261230

theorem calculate_percentage_passed (F_H F_E F_HE : ℝ) (h1 : F_H = 0.32) (h2 : F_E = 0.56) (h3 : F_HE = 0.12) :
  1 - (F_H + F_E - F_HE) = 0.24 := by
  sorry

end calculate_percentage_passed_l261_261230


namespace distance_point_to_line_l261_261158

-- Definitions based on conditions
def line_parametric (t : ℝ) : ℝ × ℝ := (t + 3, 3 - t)

def circle_parametric (θ : ℝ) (h : 0 ≤ θ ∧ θ < 2 * Real.pi) : ℝ × ℝ := (2 * Real.cos θ, 2 * Real.sin θ + 1)

def line_standard (x y : ℝ) : Prop := x + y - 6 = 0

def point_center_circle : ℝ × ℝ := (0, 1)

-- Theorem
theorem distance_point_to_line : 
  let (x0, y0) := point_center_circle
  in ∀ A B C, 
     A = 1 → B = 1 → C = -6 → 
     (∀ x y, line_standard x y ↔ A * x + B * y + C = 0) →
  Real.abs (A * x0 + B * y0 + C) / Real.sqrt (A ^ 2 + B ^ 2) = 5 * Real.sqrt 2 / 2 :=
by
  intros
  -- Initial code to integrate hypothesis into the environment. The actual proof is omitted.
  sorry

end distance_point_to_line_l261_261158


namespace system_of_equations_solution_l261_261699

theorem system_of_equations_solution (x : ℝ) (x_1 x_2 x_3 x_4 x_5 : ℕ → ℝ)
  (h₁ : x_1 1 * x_1 2 * x_1 3 = x_1 1 + x_1 2 + x_1 3)
  (h₂ : x_1 2 * x_1 3 * x_1 4 = x_1 2 + x_1 3 + x_1 4)
  (h₃ : x_1 3 * x_1 4 * x_1 5 = x_1 3 + x_1 4 + x_1 5)
  (h₄ : ∀ n : ℕ, n > 4 → 
        x_1 n * x_1 (n+1) * x_1 (n+2) = x_1 n + x_1 (n+1) + x_1 (n+2))
  (h₅ : x_1 1985 * x_1 1986 * x_1 1987 = x_1 1985 + x_1 1986 + x_1 1987)
  (h₆ : x_1 1986 * x_1 1987 * x_1 1 = x_1 1986 + x_1 1987 + x_1 1)
  (h₇ : x_1 1987 * x_1 1 * x_1 2 = x_1 1987 + x_1 1 + x_1 2) :
  x = 0 ∨ x = √3 ∨ x = -√3 :=
sorry

end system_of_equations_solution_l261_261699


namespace Freddy_travel_time_l261_261910

-- Define the distances and times
def distance_A_B := 480 -- Distance from city A to city B in km
def time_Eddy := 3 -- Time taken by Eddy to travel from city A to city B in hours
def speed_ratio := 2.1333333333333333 -- Ratio of Eddy's speed to Freddy's speed
def distance_A_C := 300 -- Distance from city A to city C in km

-- Calculate Eddy's speed
def speed_Eddy := distance_A_B / time_Eddy

-- Calculate Freddy's speed based on the ratio
def speed_Freddy := speed_Eddy / speed_ratio

-- Calculate the time for Freddy's journey
def time_Freddy := distance_A_C / speed_Freddy

-- The final statement that needs to be proved
theorem Freddy_travel_time : time_Freddy = 4 := by
  -- The proof is omitted
  sorry

end Freddy_travel_time_l261_261910


namespace sum_first_nine_terms_arithmetic_sequence_l261_261176

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d, ∀ n, a (n + 2) = a (n + 1) + d

theorem sum_first_nine_terms_arithmetic_sequence 
    (h_arith : is_arithmetic_sequence a) 
    (h_a5 : a 5 = 5) :
    (∑ i in Finset.range 9, a i) = 45 := by
  sorry

end sum_first_nine_terms_arithmetic_sequence_l261_261176


namespace polynomial_roots_arithmetic_progression_complex_root_l261_261487

theorem polynomial_roots_arithmetic_progression_complex_root :
  ∃ a : ℝ, (∀ (r d : ℂ), (r - d) + r + (r + d) = 9 → (r - d) * r + (r - d) * (r + d) + r * (r + d) = 30 → d^2 = -3 → 
  (r - d) * r * (r + d) = -a) → a = -12 :=
by sorry

end polynomial_roots_arithmetic_progression_complex_root_l261_261487


namespace correct_point_on_hyperbola_l261_261563

-- Given condition
def hyperbola_condition (x y : ℝ) : Prop := x * y = -4

-- Question (translated to a mathematically equivalent proof)
theorem correct_point_on_hyperbola :
  hyperbola_condition (-2) 2 :=
sorry

end correct_point_on_hyperbola_l261_261563


namespace cracker_calories_l261_261466

theorem cracker_calories (cc : ℕ) (hc1 : ∀ (n : ℕ), n = 50 → cc = 50) (hc2 : ∀ (n : ℕ), n = 7 → 7 * 50 = 350) (hc3 : ∀ (n : ℕ), n = 10 * cc → 10 * cc = 10 * cc) (hc4 : 350 + 10 * cc = 500) : cc = 15 :=
by
  sorry

end cracker_calories_l261_261466


namespace selection_ways_l261_261358

open Finset

theorem selection_ways : 
  let n := 8;
  let english_translation := 5;
  let software_design := 4;
  let both := 1; -- person A
  let english_without_both := english_translation - both; -- 4
  let software_without_both := software_design - both; -- 3
  let total_selection := 5;
  let english_needed := 3;
  let software_needed := 2;
  let choose (n k : ℕ) : ℕ := (Finset.range n).choose k
  in
  choose english_without_both english_needed * choose software_without_both software_needed +
    choose (english_without_both - 1) (english_needed - 1) * choose software_without_both software_needed +
    choose english_without_both english_needed * choose (software_without_both - 1) (software_needed - 1) = 42 :=
by sorry

end selection_ways_l261_261358


namespace trig_expression_value_l261_261940

theorem trig_expression_value (α : ℝ) (x y : ℝ) (h : (x, y) = (1, 3)) :
  (∃ r : ℝ, r = Real.sqrt (x^2 + y^2) ∧ 
   ∀ α, 
   sin (π - α) - sin (π / 2 + α) = 
   Real.sin α - Real.cos α ∧ 
   cos (α - 2 * π) = Real.cos α)
  → (sin (π - α) - sin (π / 2 + α)) / (2 * cos (α - 2 * π)) = 1 :=
by sorry

end trig_expression_value_l261_261940


namespace equation_value_l261_261024

-- Define the expressions
def a := 10 + 3
def b := 7 - 5

-- State the theorem
theorem equation_value : a^2 + b^2 = 173 := by
  sorry

end equation_value_l261_261024


namespace sin_arcsin_plus_arctan_l261_261483

theorem sin_arcsin_plus_arctan (a b : ℝ) (ha : a = Real.arcsin (4/5)) (hb : b = Real.arctan (1/2)) :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2)) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end sin_arcsin_plus_arctan_l261_261483


namespace machine_performance_l261_261282

noncomputable def machine_A_data : List ℕ :=
  [4, 1, 0, 2, 2, 1, 3, 1, 2, 4]

noncomputable def machine_B_data : List ℕ :=
  [2, 3, 1, 1, 3, 2, 2, 1, 2, 3]

noncomputable def mean (data : List ℕ) : ℝ :=
  (data.sum : ℝ) / data.length

noncomputable def variance (data : List ℕ) (mean : ℝ) : ℝ :=
  (data.map (λ x => (x - mean) ^ 2)).sum / data.length

theorem machine_performance :
  let mean_A := mean machine_A_data
  let mean_B := mean machine_B_data
  let variance_A := variance machine_A_data mean_A
  let variance_B := variance machine_B_data mean_B
  mean_A = 2 ∧ mean_B = 2 ∧ variance_A = 1.6 ∧ variance_B = 0.6 ∧ variance_B < variance_A := 
sorry

end machine_performance_l261_261282


namespace select_integers_divisible_l261_261641

theorem select_integers_divisible (k : ℕ) (s : Finset ℤ) (h₁ : s.card = 2 * 2^k - 1) :
  ∃ t : Finset ℤ, t ⊆ s ∧ t.card = 2^k ∧ (t.sum id) % 2^k = 0 :=
sorry

end select_integers_divisible_l261_261641


namespace tangent_line_ln_curve_l261_261960

theorem tangent_line_ln_curve (a : ℝ) : (∃ (f : ℝ → ℝ), f = λ x, Real.log x + a ∧ (∀ (x : ℝ), x > 0 → (f x = x → Real.log x + a = x)) ∧ (∀ (x : ℝ), x > 0 → (iteratedDeriv 1 f x = 1 → Real.log x - x  = a - 1))) → a = 1 :=
by
  sorry

end tangent_line_ln_curve_l261_261960


namespace orthogonal_vectors_z_eq_zero_l261_261507

variable (z : ℝ)

def v : ℝ × ℝ × ℝ × ℝ := (2, -1, 3, 5)
def w : ℝ × ℝ × ℝ × ℝ := (-1, z, 4, -2)

theorem orthogonal_vectors_z_eq_zero
  (h : v.fst * w.fst + v.snd * w.snd + v.snd.snd * w.snd.snd + v.snd.snd.snd * w.snd.snd.snd = 0) :
  z = 0 := sorry

end orthogonal_vectors_z_eq_zero_l261_261507


namespace minimal_N_is_101_l261_261442

-- Definition of the problem conditions and question
def minimal_N_for_trick (N : ℕ) : Prop :=
  ∀ (d : Fin N → ℕ), 
  (∀ (pos : ℕ), pos < N - 1 → 
  ∃! (a b : ℕ × ℕ), 
  d pos = a ∧ 
  d (pos + 1) = b)

-- Theorem stating the minimal N
theorem minimal_N_is_101 : ∃ N, minimal_N_for_trick N ∧ N = 101 :=
by {
  apply Exists.intro 101,
  split,
  {
    -- condition that Arutyun and Amayak can always determine the hidden digits
    intros d pos hpos,
    have hid_pos := sorry, -- This is where the detailed proof would go
    exact hid_pos,
  },
  {
    -- Proof that N = 101 is indeed the minimal N
    refl,
  }
}

end minimal_N_is_101_l261_261442


namespace perpendicular_condition_sufficient_not_necessary_l261_261465

theorem perpendicular_condition_sufficient_not_necessary (m : ℝ) :
  (∀ x y : ℝ, m * x + (2 * m - 1) * y + 1 = 0) →
  (∀ x y : ℝ, 3 * x + m * y + 3 = 0) →
  (∀ a b : ℝ, m = -1 → (∃ c d : ℝ, 3 / a = 1 / b)) →
  (m = -1 → (m = -1 → (3 / (-m / (2 * m - 1)) * m) / 2 - (3 / m) = -1)) :=
by sorry

end perpendicular_condition_sufficient_not_necessary_l261_261465


namespace remainder_mod_7_l261_261916

theorem remainder_mod_7 (x y z : ℤ) (hx : x < 7) (hy : y < 7) (hz : z < 7)
  (h1 : x + 3 * y + 2 * z ≡ 2 [MOD 7])
  (h2 : 3 * x + 2 * y + z ≡ 5 [MOD 7])
  (h3 : 2 * x + y + 3 * z ≡ 3 [MOD 7]) :
  (x * y * z) ≡ 3 [MOD 7] :=
sorry

end remainder_mod_7_l261_261916


namespace solution_of_log_equation_l261_261653

noncomputable def f (x : ℝ) : ℝ := 8 - x - log x

theorem solution_of_log_equation (x k : ℝ) (h₁ : 8 - x = log x) (h₂ : x ∈ (k, k+1)) (h₃ : k ∈ ℤ) :
  k = 7 :=
by
  sorry

end solution_of_log_equation_l261_261653


namespace john_task_completion_l261_261637

theorem john_task_completion (J : ℝ) (h : 5 * (1 / J + 1 / 10) + 5 * (1 / J) = 1) : J = 20 :=
by
  sorry

end john_task_completion_l261_261637


namespace part1_monotone_increasing_part2_odd_function_part3_solution_set_l261_261188

def f (x : ℝ) : ℝ := x / (4 - x^2)

theorem part1_monotone_increasing : ∀ (x1 x2 : ℝ), 0 < x1 ∧ x1 < x2 ∧ x2 < 2 → f(x1) < f(x2) := 
sorry

theorem part2_odd_function : ∀ (x : ℝ), x ∈ set.Ioo (-2 : ℝ) 2 → f(-x) = -f(x) := 
sorry

theorem part3_solution_set : ∀ (t : ℝ), t ∈ set.Ioo (-2 : ℝ) 2 → f(t) + f(1 - 2 * t) > 0 ↔ t ∈ set.Ioo (-1/2 : ℝ) 1 := 
sorry

end part1_monotone_increasing_part2_odd_function_part3_solution_set_l261_261188


namespace division_simplification_l261_261877

theorem division_simplification : 180 / (12 + 13 * 3) = 60 / 17 := by
  sorry

end division_simplification_l261_261877


namespace wheel_center_travel_distance_l261_261091

theorem wheel_center_travel_distance (radius : ℝ) (revolutions : ℝ) (flat_surface : Prop) 
  (h_radius : radius = 2) (h_revolutions : revolutions = 2) : 
  radius * 2 * π * revolutions = 8 * π :=
by
  rw [h_radius, h_revolutions]
  simp [mul_assoc, mul_comm]
  sorry

end wheel_center_travel_distance_l261_261091


namespace minimum_weight_of_each_crate_l261_261824

theorem minimum_weight_of_each_crate :
  (∀ n, n ∈ {3, 4, 5} → ∃ w, w > 0 ∧ n * w ≤ 6250) →
  (∃ w_min, w_min > 0 ∧ w_min = 6250 / 5) :=
by
  sorry

end minimum_weight_of_each_crate_l261_261824


namespace range_of_a_l261_261981

variable (ℝ : Type) [linear_ordered_field ℝ]
variables (A : set ℝ) (B : set ℝ)
variable (a : ℝ)

def U : set ℝ := set.univ
def A : set ℝ := { x | x < 0 }
def B : set ℝ := { -1, -3, a }
def complement_A : set ℝ := { x | x ≥ 0 }

theorem range_of_a (h : (complement_A ∩ B).nonempty) : a ≥ 0 := 
sorry

end range_of_a_l261_261981


namespace max_colors_cube_l261_261534

theorem max_colors_cube (n : ℕ) (h : 2 ≤ n) 
  (colors : Fin (n^3) → Fin (3 * n)) 
  (distinct_prisms : ∀ (i j k : Fin n), 
      ∃ (s1 s2 s3 : Finset (Fin (3 * n))), 
      ((∀ i, colors (Fin.val (Fin.add i n n n)) ∈ s1) ∧
       (∀ j, colors (Fin.val (Fin.add j n n n)) ∈ s2) ∧
       (∀ k, colors (Fin.val (Fin.add k n n n)) ∈ s3) ∧ 
       s1 = s2 ∧ s2 = s3)) :
  ∃ (max_colors : ℕ), max_colors = (3 * n^2) / 2 :=
by
  -- Proof omitted as per instructions
  sorry

end max_colors_cube_l261_261534


namespace problem_solution_l261_261244

def number_of_valid_tuples : ℕ := -- defining the correct answer
  864

open Finset

/-- Given conditions for the problem -/
def valid_last_digit (digit : ℕ) : Prop :=
  digit = 0 ∨ digit = 5

def valid_digit (digit : ℕ) : Prop :=
  digit ∈ {0, 2, 4, 5, 7, 9}

def sum_mod_3 (digits : List ℕ) : Prop :=
  (11 + digits.sum) % 3 = 0

def valid_11_digit_number (digits : List ℕ) : Prop :=
  digits.length = 5 ∧ valid_last_digit (digits.getLast!) ∧ digits.all valid_digit ∧ sum_mod_3 digits

/-- Main theorem statement -/
theorem problem_solution :
  ({ digits : List ℕ // valid_11_digit_number digits }.card = number_of_valid_tuples) :=
by
  sorry

end problem_solution_l261_261244


namespace smallest_solution_x4_minus_50x2_plus_625_eq_0_l261_261756

theorem smallest_solution_x4_minus_50x2_plus_625_eq_0 : ∃ x : ℝ, x^4 - 50 * x^2 + 625 = 0 ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y := 
sorry

end smallest_solution_x4_minus_50x2_plus_625_eq_0_l261_261756


namespace min_xy_l261_261265

-- Definitions of the conditions
variable (x y : ℝ)
variable (hx : 0 < x)
variable (hy : 0 < y)
variable (h : (3 / (2 + x) + 3 / (2 + y)) = 1)

-- The problem statement to prove
theorem min_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (3 / (2 + x) + 3 / (2 + y)) = 1) :
  ∃ z : ℝ, z = 16 ∧ ∀ w : ℝ, (hx : 0 < x) → (hy : 0 < y) → (h : (3 / (2 + x) + 3 / (2 + y)) = 1) → (x * y) = w → w = z :=
begin
  sorry
end

end min_xy_l261_261265


namespace smallest_n_f_gt_15_l261_261651

def f (n : ℕ) : ℕ :=
  Inf {k // k ! % n = 0}

theorem smallest_n_f_gt_15 (n : ℕ) (hn1 : 12 ∣ n) (hn2 : 11 ∣ n) (h : f(n) > 15) : n = 396 :=
  sorry

end smallest_n_f_gt_15_l261_261651


namespace remainder_problem_l261_261381

theorem remainder_problem (n : ℤ) (h : n % 25 = 4) : (n + 15) % 5 = 4 := by
  sorry

end remainder_problem_l261_261381


namespace trig_identity_l261_261545

theorem trig_identity (α : ℝ) (h : Real.cos (75 * Real.pi / 180 + α) = 1 / 3) :
  Real.sin (α - 15 * Real.pi / 180) + Real.cos (105 * Real.pi / 180 - α) = -2 / 3 :=
sorry

end trig_identity_l261_261545


namespace a8_eq_128_l261_261939

-- Definitions of conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

variables {a : ℕ → ℝ} {q : ℝ}

-- Conditions
axiom a2_eq_2 : a 2 = 2
axiom a3_mul_a4_eq_32 : a 3 * a 4 = 32
axiom is_geometric : is_geometric_sequence a q

-- Statement to prove
theorem a8_eq_128 : a 8 = 128 :=
sorry

end a8_eq_128_l261_261939


namespace binomial_sum_eq_l261_261677

theorem binomial_sum_eq {n : ℕ} (h : 0 < n) :
  ∑ i in finset.range (n+1), (nat.choose (2*n+1) (2*i)) * (nat.choose (2*i) i) * 2^(2*n-2*i+1)
   = nat.choose (4*n+2) (2*n+1) := sorry

end binomial_sum_eq_l261_261677


namespace probability_four_streetlights_replacement_expected_number_of_streetlights_replacement_l261_261043

-- Defining the probability calculations
def count_valid_sequences (n k : ℕ) : ℕ :=
  Nat.choose (n - k + 1) k

def total_possible_sequences (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_four_replacements (n k : ℕ) : ℚ :=
  (count_valid_sequences n k) / (total_possible_sequences n k)

-- Defining expected value calculations
def expected_replacements (n : ℕ) : ℚ :=
  let sequences := [1, 1, 0.777, 0.416, 0.119] -- From the actual solution's steps
  (sequences.sum) / sequences.length

-- Theorem to prove part (a)
theorem probability_four_streetlights_replacement :
  probability_four_replacements 9 4 = 5 / 42 :=
by sorry

-- Theorem to prove part (b)
theorem expected_number_of_streetlights_replacement :
  expected_replacements 9 ≈ 3.32 :=
by sorry

end probability_four_streetlights_replacement_expected_number_of_streetlights_replacement_l261_261043


namespace hyperbola_asymptotes_l261_261339

noncomputable def a_h : Real :=
  let x := -4 / 3
  let y := 3 * x + 6
  let h := x
  let k := y
  let (a: Real) := 7 * Real.sqrt 6 / 2
  (a + h) == (21 * Real.sqrt 6 - 8) / 6

theorem hyperbola_asymptotes
  (h k a b : Real)
  (h_eq : h = -4 / 3)
  (k_eq : k = 2)
  (a_eq : a = (7 * Real.sqrt 6) / 2)
  (a_h : b = (7 * Real.sqrt 2) / 6)
  (pt_condition : (-4 / 3, 2, (7 * Real.sqrt 6) / 2, (7 * Real.sqrt 2) / 6) → (1, 9) satisfies (y - k) ^ 2 / a ^ 2 - (x - h) ^ 2 / b ^ 2 = 1) :
    a + h = (21 * Real.sqrt 6 - 8) / 6 :=
by
  sorry

end hyperbola_asymptotes_l261_261339


namespace arrange_PERCEPTION_l261_261904

theorem arrange_PERCEPTION :
  ∀ n k1 k2 k3 : ℕ, n = 10 → k1 = 2 → k2 = 2 → k3 = 2 →
  (nat.factorial n) / (nat.factorial k1 * nat.factorial k2 * nat.factorial k3) = 453600 := 
by
  intros n k1 k2 k3 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end arrange_PERCEPTION_l261_261904


namespace exists_point_lt_2f_l261_261049

noncomputable def non_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem exists_point_lt_2f {f : ℝ → ℝ} (h : ∀ x, 0 < f x) (hf : non_decreasing f) :
  ∃ a : ℝ, f (a + 1 / f a) < 2 * f a :=
begin
  sorry
end

end exists_point_lt_2f_l261_261049


namespace prime_sum_of_primes_l261_261045

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def distinct_primes (p q r s : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime s ∧ p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s

theorem prime_sum_of_primes (p q r s : ℕ) :
  distinct_primes p q r s →
  is_prime (p + q + r + s) →
  is_square (p^2 + q * s) →
  is_square (p^2 + q * r) →
  (p = 2 ∧ q = 7 ∧ r = 11 ∧ s = 3) ∨ (p = 2 ∧ q = 7 ∧ r = 3 ∧ s = 11) :=
by
  sorry

end prime_sum_of_primes_l261_261045


namespace two_pow_a_plus_two_pow_neg_a_l261_261994

theorem two_pow_a_plus_two_pow_neg_a (a : ℝ) (h : a * Real.log 4 / Real.log 3 = 1) :
  2^a + 2^(-a) = (4 * Real.sqrt 3) / 3 :=
by
  sorry

end two_pow_a_plus_two_pow_neg_a_l261_261994


namespace fruits_selection_l261_261369

theorem fruits_selection : ∃ (n k : ℕ), (n = 4) ∧ (k = 2) ∧ nat.choose n k = 6 :=
by
  use [4, 2]
  sorry

end fruits_selection_l261_261369


namespace triangle_angles_l261_261114

theorem triangle_angles (a b c m_a m_b m_c : ℝ) (A B C : Type) 
  (h1 : m_a ≥ a) (h2 : m_b ≥ b) (ha : m_a = b) (hb : m_b = a) 
  (triangle_ABC : triangle A B C a b c m_a m_b m_c) : 
  angle A C B = 90 ∧ angle B A C = 45 ∧ angle A B C = 45 := 
sorry

end triangle_angles_l261_261114


namespace find_c_l261_261714

-- Defining the variables and conditions given in the problem
variables (a b c : ℝ)

-- Conditions
def vertex_condition : Prop := (2, -3) = (a * (-3)^2 + b * (-3) + c, -3)
def point_condition : Prop := (7, -1) = (a * (-1)^2 + b * (-1) + c, -1)

-- Problem Statement
theorem find_c 
  (h_vertex : vertex_condition a b c)
  (h_point : point_condition a b c) :
  c = 53 / 4 :=
sorry

end find_c_l261_261714


namespace problem1_problem2_problem3_l261_261169

noncomputable def f (a : ℝ) (n : ℕ) := a ^ n

theorem problem1 {a : ℝ} {n : ℕ} (ha : 0 < a) (hn : 0 < n) :
  f(a, n) = a ^ n :=
sorry

theorem problem2 {a : ℝ} (ha : a ≥ 3) (hn : ℕ) :
  (a^n - 1) / (a^n + 1) ≥ n / (n + 1) :=
sorry

theorem problem3 {a : ℝ} (ha0 : 0 < a) (ha1 : a < 1) (n : ℕ) :
  (finset.range n).sum (λ k, 1 / (f a k - f a (2 * k))) >
  6 * (f a 1 - f a (n + 1)) / (f a 0 - f a 1) :=
sorry

end problem1_problem2_problem3_l261_261169


namespace matrix_multiple_of_4_l261_261642

theorem matrix_multiple_of_4 
  (n : ℕ) 
  (h₁ : n ≥ 3) 
  (A : matrix (fin n) (fin n) ℤ) 
  (h₂ : ∀ i j : fin n, A i j = 1 ∨ A i j = -1)
  (h₃ : ∀ k : fin n, A k 0 = 1)
  (h₄ : ∀ i j : fin n, i ≠ j → finset.univ.sum (λ k, A k i * A k j) = 0) : 
  n % 4 = 0 := 
sorry

end matrix_multiple_of_4_l261_261642


namespace suitable_k_count_l261_261138

theorem suitable_k_count :
  { k : ℕ | k ≤ 291000 ∧ (k^2 - 1) % 291 = 0 }.card = 4000 :=
by
  sorry

end suitable_k_count_l261_261138


namespace methane_hydrate_scientific_notation_l261_261287

theorem methane_hydrate_scientific_notation :
  (9.2 * 10^(-4)) = 0.00092 :=
by sorry

end methane_hydrate_scientific_notation_l261_261287


namespace no_prime_condition_l261_261701

open Nat

theorem no_prime_condition (m n : ℕ) (hm : 1 < m) (hn : 1 < n) 
  (hdiv : (m + n - 1) ∣ (m^2 + n^2 - 1)) : ¬ Prime (m + n - 1) := 
by
  sorry

end no_prime_condition_l261_261701


namespace calculate_fraction_l261_261585

theorem calculate_fraction (x y : ℚ) (h1 : x = 5 / 6) (h2 : y = 6 / 5) : (1 / 3) * x^8 * y^9 = 2 / 5 := by
  sorry

end calculate_fraction_l261_261585


namespace rooks_on_checkerboard_l261_261332

theorem rooks_on_checkerboard : ∃ n : ℕ, n = 2880 ∧ ∀ (board : fin 9 × fin 9 → Prop), 
    (∀ r1 r2 c1 c2 : fin 9, r1 ≠ r2 → c1 ≠ c2 → board ⟨r1, c1⟩ → board ⟨r2, c2⟩ → false) ↔ 
    (board (0,0) ∨ board (0,1) ∨ board (1,0) ∨ board (1,1) → 
    ∃ f : fin 9 → fin 9, function.injective f ∧ ∀ i, board ⟨i, f i⟩) := 
begin
  use 2880,
  split,
  { refl, },
  sorry
end

end rooks_on_checkerboard_l261_261332


namespace subtraction_identity_l261_261051

theorem subtraction_identity : 4444444444444 - 2222222222222 - 444444444444 = 1777777777778 :=
  by norm_num

end subtraction_identity_l261_261051


namespace centroid_coincides_l261_261397

noncomputable def centroid_of_triangle (A B C : Point) : Point := sorry -- Definition of the centroid of a triangle

theorem centroid_coincides 
  (A B C D : Point)
  (hAB : A ≠ B)
  (hCD_altitude : Altitude C D A B) -- represents that CD is the altitude from C to AB
  (M1 : Point := centroid_of_triangle A C D)
  (M2 : Point := centroid_of_triangle B C D)
  (Z : Point := centroid_of_triangle A B C)
  : Z = centroid_of_triangle A C D ∧ Z = centroid_of_triangle B C D := 
sorry

end centroid_coincides_l261_261397


namespace money_left_after_purchase_l261_261313

-- The costs and amounts for each item
def bread_cost : ℝ := 2.35
def num_bread : ℝ := 4
def peanut_butter_cost : ℝ := 3.10
def num_peanut_butter : ℝ := 2
def honey_cost : ℝ := 4.50
def num_honey : ℝ := 1

-- The coupon discount and budget
def coupon_discount : ℝ := 2
def budget : ℝ := 20

-- Calculate the total cost before applying the coupon
def total_before_coupon : ℝ := num_bread * bread_cost + num_peanut_butter * peanut_butter_cost + num_honey * honey_cost

-- Calculate the total cost after applying the coupon
def total_after_coupon : ℝ := total_before_coupon - coupon_discount

-- Calculate the money left over after the purchase
def money_left_over : ℝ := budget - total_after_coupon

-- The theorem to be proven
theorem money_left_after_purchase : money_left_over = 1.90 :=
by
  -- The proof of this theorem will involve the specific calculations and will be filled in later
  sorry

end money_left_after_purchase_l261_261313


namespace find_t_squared_l261_261804
noncomputable section

-- Definitions of the given conditions
def hyperbola_opens_vertically (x y : ℝ) : Prop :=
  (y^2 / 4 - 5 * x^2 / 16 = 1)

-- Statement of the problem
theorem find_t_squared (t : ℝ) 
  (h1 : hyperbola_opens_vertically 4 (-3))
  (h2 : hyperbola_opens_vertically 0 (-2))
  (h3 : hyperbola_opens_vertically 2 t) : 
  t^2 = 8 := 
sorry -- Proof is omitted, it's just the statement

end find_t_squared_l261_261804


namespace hiking_rate_up_the_hill_l261_261777

theorem hiking_rate_up_the_hill (r_down : ℝ) (t_total : ℝ) (t_up : ℝ) (r_up : ℝ) :
  r_down = 6 ∧ t_total = 3 ∧ t_up = 1.2 → r_up * t_up = 9 * t_up :=
by
  intro h
  let ⟨hrd, htt, htu⟩ := h
  sorry

end hiking_rate_up_the_hill_l261_261777


namespace distance_between_tangent_and_secant_l261_261606

theorem distance_between_tangent_and_secant
  (r : ℝ)
  (h_r_pos : 0 < r)
  (circle : ℝ → ℝ → Prop)
  (center O : ℝ × ℝ)
  (K M : ℝ × ℝ)
  (chord_length : ℝ)
  (tangent : ℝ → ℝ → Prop)
  (secant : ℝ → ℝ → Prop)
  (h1 : chord_length = r / 2)
  (h2 : circle O.1 O.2 → circle center.1 center.2)
  (h3 : center.1 = O.1 ∧ center.2 = O.2)
  (h4 : K ≠ M)
  (h5 : tangent K.1 K.2)
  (h6 : secant M.1 M.2)
  (h7 : ∀ p : ℝ × ℝ, tangent p.1 p.2 ↔ (p.1 = K.1 → p.2 = K.2))
  (h8 : ∀ q : ℝ × ℝ, secant q.1 q.2 ↔ (q.1 = M.1 → q.2 = M.2)) :
  let α := ∠center K M in 
  cos α = 7 / 8 ∧ sorry := sorry
  EK = (r - r * (7 / 8)) in 
  r / 8

end distance_between_tangent_and_secant_l261_261606


namespace range_of_a_for_three_distinct_zeros_l261_261594

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - 3 * x + a

theorem range_of_a_for_three_distinct_zeros : 
  ∀ a : ℝ, (∀ x y : ℝ, x ≠ y → f x a = 0 → f y a = 0 → (f (1:ℝ) a < 0 ∧ f (-1:ℝ) a > 0)) ↔ (-2 < a ∧ a < 2) := 
by
  sorry

end range_of_a_for_three_distinct_zeros_l261_261594


namespace additional_savings_in_cents_l261_261735

/-
The book has a cover price of $30.
There are two discount methods to compare:
1. First $5 off, then 25% off.
2. First 25% off, then $5 off.
Prove that the difference in final costs (in cents) between these two discount methods is 125 cents.
-/
def book_price : ℝ := 30
def discount_cash : ℝ := 5
def discount_percentage : ℝ := 0.25

def final_price_apply_cash_first (price : ℝ) (cash_discount : ℝ) (percentage_discount : ℝ) : ℝ :=
  (price - cash_discount) * (1 - percentage_discount)

def final_price_apply_percentage_first (price : ℝ) (percentage_discount : ℝ) (cash_discount : ℝ) : ℝ :=
  (price * (1 - percentage_discount)) - cash_discount

def savings_comparison (price : ℝ) (cash_discount : ℝ) (percentage_discount : ℝ) : ℝ :=
  (final_price_apply_cash_first price cash_discount percentage_discount) - 
  (final_price_apply_percentage_first price percentage_discount cash_discount)

theorem additional_savings_in_cents : 
  savings_comparison book_price discount_cash discount_percentage * 100 = 125 :=
  by sorry

end additional_savings_in_cents_l261_261735


namespace student_correct_answers_l261_261389

theorem student_correct_answers :
  ∃ c w : ℕ, c + w = 60 ∧ 4 * c - w = 140 ∧ c = 40 :=
by
  use 40, 20
  split
  case left =>
    exact rfl
  case right =>
    split
    case left =>
      exact rfl
    case right =>
      exact rfl


end student_correct_answers_l261_261389


namespace math_scores_exceed_90_l261_261218

open MeasureTheory Probability

noncomputable def math_score_pdf (σ : ℝ) : ℝ → ℝ :=
  λ x, (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(x - 65)^2 / (2 * σ^2))

theorem math_scores_exceed_90 :
  ∃ (σ : ℝ),
    ∀ (f : ℝ → ℝ) (X : MeasureTheory.PMF ℝ),
      f = math_score_pdf σ ∧
      P (40 ≤ X ∧ X ≤ 90) = 0.9 →
      40,000 * (P (X > 90)) = 2000 :=
by sorry

end math_scores_exceed_90_l261_261218


namespace trigonometric_expression_l261_261533

noncomputable def sin_alpha : ℝ := -((2 * sqrt 5) / 5)
noncomputable def cos_alpha : ℝ := (sqrt 5) / 5
noncomputable def tan_alpha : ℝ := -2

theorem trigonometric_expression (α : ℝ) 
  (h₁ : sin α = sin_alpha)
  (h₂ : cos α = cos_alpha) :
  (cos (π / 2 + α) * sin (-π - α)) / (cos (11 * π / 2 - α) * sin (9 * π / 2 + α)) = -2 :=
by
  sorry

end trigonometric_expression_l261_261533


namespace cyclic_quad_area_l261_261538

noncomputable def can_form_cyclic_quad (a b c d : ℕ) (s : ℕ) : Prop :=
  let area := Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d)) in
  area = Real.sqrt 61560

theorem cyclic_quad_area :
  ∃ (a b c d : ℕ), a = 13 ∧ b = 14 ∧ c = 15 ∧ d = 24 ∧
  let s := (a + b + c + d) / 2
  in can_form_cyclic_quad a b c d s :=
by
  use 13, 14, 15, 24
  simp only [-sub_eq_add_neg, add_assoc, add_left_comm, mul_comm (13 : ℕ), add_comm (13 : ℕ), int.eq_nat_of_add_eq_add_left, inv_mul_eq_iff_eq_mul, add_zero, add_sub, add_sub_cancel, nat.cast_eq_zero_iff_le]
  sorry

end cyclic_quad_area_l261_261538


namespace nine_rooks_checkerboard_l261_261326

theorem nine_rooks_checkerboard :
  let num_ways_4x4 := 4.factorial
  let num_ways_5x5 := 5.factorial
  num_ways_4x4 * num_ways_5x5 = 2880 :=
by
  sorry

end nine_rooks_checkerboard_l261_261326


namespace sequence_properties_and_sum_l261_261649

noncomputable def a (n : ℕ) := 2 * n - 1
noncomputable def b (n : ℕ) := 2 ^ n

def S (n : ℕ) := ∑ i in finset.range n, a i / b i

theorem sequence_properties_and_sum (n : ℕ) :
  -- Arithmetic and Geometric Sequences Verification
  a 1 = 1 ∧
  b 1 = 2 ∧
  (a 2 + b 3 = 11) ∧
  (a 3 + b 5 = 37) ∧
  -- Formulas for a_n and b_n Verification
  (∀ n, a n = 2 * n - 1) ∧
  (∀ n, b n = 2 ^ n) ∧
  -- Sum of the sequence S_n Verification
  S n = 3 - (2 * n + 3) / 2 ^ n
:=
begin
  sorry
end

end sequence_properties_and_sum_l261_261649


namespace enlarged_logo_height_l261_261094

-- Definitions of the conditions given in the problem
def original_width : ℝ := 2
def original_height : ℝ := 1.5
def new_width : ℝ := 8

-- Definition of the proportional enlargement factor
def enlargement_factor : ℝ := new_width / original_width

-- Theorem stating the problem
theorem enlarged_logo_height (h : new_width = 8) : original_height * enlargement_factor = 6 := 
by
  -- Enlargement factor calculation
  have factor : enlargement_factor = 4 := by
    rw [enlargement_factor, new_width, original_width]
    norm_num
  
  -- Calculation of new height via proportion
  calc
    original_height * enlargement_factor = 1.5 * 4 := by
      rw [original_height, factor]
    _ = 6 := by norm_num

end enlarged_logo_height_l261_261094


namespace part1_x1_part1_x0_part1_xneg2_general_inequality_l261_261109

-- Prove inequality for specific values of x
theorem part1_x1 : - (1/2 : ℝ) * (1: ℝ)^2 + 2 * (1: ℝ) < -(1: ℝ) + 5 := by
  sorry

theorem part1_x0 : - (1/2 : ℝ) * (0: ℝ)^2 + 2 * (0: ℝ) < -(0: ℝ) + 5 := by
  sorry

theorem part1_xneg2 : - (1/2 : ℝ) * (-2: ℝ)^2 + 2 * (-2: ℝ) < -(-2: ℝ) + 5 := by
  sorry

-- Prove general inequality for all real x
theorem general_inequality (x : ℝ) : - (1/2 : ℝ) * x^2 + 2 * x < -x + 5 := by
  sorry

end part1_x1_part1_x0_part1_xneg2_general_inequality_l261_261109


namespace find_a_range_absolute_difference_inequality_l261_261971

section PartI
variables {a : ℝ} (f : ℝ → ℝ) (h : ∀ x > 0, f x ≤ 0)
def f_definition (x : ℝ) : ℝ := a * log x - x^2 + 1

theorem find_a_range (h₁ : ∀ x > 0, f x ≤ 0) : a = 2 :=
by {
  sorry
}
end PartI

section PartII
variables {a : ℝ} (f : ℝ → ℝ) (x₁ x₂ : ℝ) 
def f_definition (x : ℝ) : ℝ := a * log x - x^2 + 1
def g (x : ℝ) : ℝ := f x + x

theorem absolute_difference_inequality (h₂ : a ≤ -1/8) (h₃ : x₁ > 0) (h₄ : x₂ > 0) : |f x₁ - f x₂| ≥ |x₁ - x₂| :=
by {
  sorry
}
end PartII

end find_a_range_absolute_difference_inequality_l261_261971


namespace bonnets_per_orphanage_l261_261667

/--
Mrs. Young makes bonnets for kids in the orphanage.
On Monday, she made 10 bonnets.
On Tuesday and Wednesday combined she made twice more than on Monday.
On Thursday she made 5 more than on Monday.
On Friday she made 5 less than on Thursday.
She divided up the bonnets evenly and sent them to 5 orphanages.
Prove that the number of bonnets Mrs. Young sent to each orphanage is 11.
-/
theorem bonnets_per_orphanage :
  let monday := 10
  let tuesday_wednesday := 2 * monday
  let thursday := monday + 5
  let friday := thursday - 5
  let total_bonnets := monday + tuesday_wednesday + thursday + friday
  let orphanages := 5
  total_bonnets / orphanages = 11 :=
by
  sorry

end bonnets_per_orphanage_l261_261667


namespace Bernardo_wins_with_sum_of_digits_l261_261448

def Bernardo_op (x : ℤ) : ℤ := 2 * x
def Silvia_op (x : ℤ) : ℤ := x + 70

theorem Bernardo_wins_with_sum_of_digits :
  ∃ M : ℤ, 0 ≤ M ∧ M ≤ 799 ∧ (36.25 ≤ M ∧ M ≤ 47.5) ∧ 
  (M.digits.sum = 10) :=
by
  sorry

end Bernardo_wins_with_sum_of_digits_l261_261448


namespace triangle_inequality_part_a_triangle_inequality_part_b_l261_261781

variable {a b c S : ℝ}

/-- Part (a): Prove that for any triangle ABC, the inequality a^2 + b^2 + c^2 ≥ 4 √3 S holds
    where equality holds if and only if ABC is an equilateral triangle. -/
theorem triangle_inequality_part_a (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (area_S : S > 0) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S :=
sorry

/-- Part (b): Prove that for any triangle ABC,
    the inequality a^2 + b^2 + c^2 - (a - b)^2 - (b - c)^2 - (c - a)^2 ≥ 4 √3 S
    holds where equality also holds if and only if a = b = c. -/
theorem triangle_inequality_part_b (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (area_S : S > 0) :
  a^2 + b^2 + c^2 - (a - b)^2 - (b - c)^2 - (c - a)^2 ≥ 4 * Real.sqrt 3 * S :=
sorry

end triangle_inequality_part_a_triangle_inequality_part_b_l261_261781


namespace class_A_students_l261_261359

variable (A B : ℕ)

theorem class_A_students 
    (h1 : A = (5 * B) / 7)
    (h2 : A + 3 = (4 * (B - 3)) / 5) :
    A = 45 :=
sorry

end class_A_students_l261_261359


namespace problem_l261_261546

variable {a b c : ℝ} -- Introducing variables a, b, c as real numbers

-- Conditions:
-- a, b, c are distinct positive real numbers
def distinct_pos (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a 

theorem problem (h : distinct_pos a b c) : 
  (b + c - a) / a + (a + c - b) / b + (a + b - c) / c > 3 :=
sorry 

end problem_l261_261546


namespace reduced_price_of_oil_l261_261427

/-- 
Given:
1. The original price per kg of oil is P.
2. The reduced price per kg of oil is 0.65P.
3. Rs. 800 can buy 5 kgs more oil at the reduced price than at the original price.
4. The equation 5P - 5 * 0.65P = 800 holds true.

Prove that the reduced price per kg of oil is Rs. 297.14.
-/
theorem reduced_price_of_oil (P : ℝ) (h1 : 5 * P - 5 * 0.65 * P = 800) : 
        0.65 * P = 297.14 := 
    sorry

end reduced_price_of_oil_l261_261427


namespace appended_number_divisible_by_12_l261_261860

theorem appended_number_divisible_by_12 :
  ∃ N, (N = 88) ∧ (∀ n, n ∈ finset.range N \ 71 → (let large_number := (list.range (N + 1)).filter (λ x, 71 ≤ x ∧ x ≤ N) in
       (list.foldr (λ a b, a * 100 + b) 0 large_number) % 12 = 0)) :=
by
  sorry

end appended_number_divisible_by_12_l261_261860


namespace monotonic_f_on_interval_l261_261596

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (x + Real.pi / 10) - 2

theorem monotonic_f_on_interval : 
  ∀ x y : ℝ, 
    x ∈ Set.Icc (Real.pi / 2) (7 * Real.pi / 5) → 
    y ∈ Set.Icc (Real.pi / 2) (7 * Real.pi / 5) → 
    x ≤ y → 
    f x ≤ f y :=
sorry

end monotonic_f_on_interval_l261_261596


namespace parabola_focus_ellipse_l261_261550

theorem parabola_focus_ellipse (a : ℝ) 
(h1 : ∃ (F : ℝ × ℝ), F = (-2, 0) ∧ focus_parabola a F ∧ focus_ellipse F) :
a = -8 := 
sorry

-- Definitions used in the statement.

def focus_parabola (a : ℝ) (F : ℝ × ℝ) : Prop :=
F = (-2, 0)

def focus_ellipse (F : ℝ × ℝ) : Prop :=
F = (-2, 0)

end parabola_focus_ellipse_l261_261550


namespace fraction_of_phone_numbers_begin_with_8_and_end_with_5_l261_261869

theorem fraction_of_phone_numbers_begin_with_8_and_end_with_5 :
  let total_numbers := 7 * 10^7
  let specific_numbers := 10^6
  specific_numbers / total_numbers = 1 / 70 := by
  sorry

end fraction_of_phone_numbers_begin_with_8_and_end_with_5_l261_261869


namespace tangent_circle_l261_261720

theorem tangent_circle (ABC : Triangle) (D E F : Point) (M : Point) (L K : Point) :
  inscribed_tangent ABC D E F → midpoint M E F → 
  circumscribed_intersects_line (triangle DMF) AB L → 
  circumscribed_intersects_line (triangle DME) AC K → 
  tangent (circumscribed_circle (triangle AKL)) BC :=
by
sorry

end tangent_circle_l261_261720


namespace find_t_squared_l261_261805
noncomputable section

-- Definitions of the given conditions
def hyperbola_opens_vertically (x y : ℝ) : Prop :=
  (y^2 / 4 - 5 * x^2 / 16 = 1)

-- Statement of the problem
theorem find_t_squared (t : ℝ) 
  (h1 : hyperbola_opens_vertically 4 (-3))
  (h2 : hyperbola_opens_vertically 0 (-2))
  (h3 : hyperbola_opens_vertically 2 t) : 
  t^2 = 8 := 
sorry -- Proof is omitted, it's just the statement

end find_t_squared_l261_261805


namespace infection_does_not_spread_with_9_cells_minimum_infected_cells_needed_l261_261291

noncomputable def grid_size := 10
noncomputable def initial_infected_count_1 := 9
noncomputable def initial_infected_count_2 := 10

def condition (n : ℕ) : Prop := 
∀ (infected : ℕ) (steps : ℕ), infected = n → 
  infected + steps * (infected / 2) < grid_size * grid_size

def can_infect_entire_grid (n : ℕ) : Prop := 
∀ (infected : ℕ) (steps : ℕ), infected = n ∧ (
  ∃ t : ℕ, infected + t * (infected / 2) = grid_size * grid_size)

theorem infection_does_not_spread_with_9_cells :
  ¬ can_infect_entire_grid initial_infected_count_1 :=
by
  sorry

theorem minimum_infected_cells_needed :
  condition initial_infected_count_2 :=
by
  sorry

end infection_does_not_spread_with_9_cells_minimum_infected_cells_needed_l261_261291


namespace max_quadratic_value_l261_261027

def quadratic_function (x m : ℝ) : ℝ := -(x - m) ^ 2 + m ^ 2 + 1

theorem max_quadratic_value (m : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 → quadratic_function x m ≤ 4) →
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 1 ∧ quadratic_function x m = 4) →
  m = 2 ∨ m = -real.sqrt 3 :=
begin
  sorry
end

end max_quadratic_value_l261_261027


namespace cube_construction_equiv_classes_l261_261059

theorem cube_construction_equiv_classes :
  (number_of_equivalent_classes (construct_cube 6 2) = 4) :=
sorry

end cube_construction_equiv_classes_l261_261059


namespace circle_isosceles_l261_261654

theorem circle_isosceles (A B C D E P Q T : Type) [h1 : dr_A : Circle] [h2 : dr_B : Circle] [h3 : dr_C : Circle] [h4 : dr_D : Circle] [h5 : dr_E : Circle] (hab : AB = BC) (hcd : CD = DE) 
(hP : ∃ P, P ∈ (AD) ∩ (BE)) 
(hQ : ∃ Q, Q ∈ (AC) ∩ (BD)) 
(hT : ∃ T, T ∈ (BD) ∩ (CE)) : 
isosceles_triangle P Q T :=
begin
  sorry
end

end circle_isosceles_l261_261654


namespace major_axis_endpoints_of_ellipse_l261_261131

theorem major_axis_endpoints_of_ellipse :
  ∀ x y, 6 * x^2 + y^2 = 6 ↔ (x = 0 ∧ (y = -Real.sqrt 6 ∨ y = Real.sqrt 6)) :=
by
  -- Proof
  sorry

end major_axis_endpoints_of_ellipse_l261_261131


namespace smallest_solution_x4_minus_50x2_plus_625_eq_0_l261_261760

theorem smallest_solution_x4_minus_50x2_plus_625_eq_0 :
  ∃ (x : ℝ), (x * x * x * x - 50 * x * x + 625 = 0) ∧ (∀ y, (y * y * y * y - 50 * y * y + 625 = 0) → x ≤ y) :=
sorry

end smallest_solution_x4_minus_50x2_plus_625_eq_0_l261_261760


namespace mutually_exclusive_and_complementary_l261_261223

-- Definitions based on the problem statement
def event_A (Ω : Type) (a : Ω → Prop) := ∀ x, a x → ¬(∃ b, b x)
def event_B (Ω : Type) (a b : Ω → Prop) := ∃ x, a x ∨ b x
def event_C (Ω : Type) (a b : Ω → Prop) := ∀ x, (a x ∨ b x) → ¬(a x ∧ b x)
def event_D (Ω : Type) (a : Ω → Prop) := ∀ x, ¬(a x)
def event_E (Ω : Type) (a b : Ω → Prop) := ∀ x, ¬(a x ∨ b x)

-- Main theorem: Show that events B and E are mutually exclusive and complementary
theorem mutually_exclusive_and_complementary
  (Ω : Type) (a b : Ω → Prop) :
  (∀ x, event_B Ω a b x → ¬(event_E Ω a b x))
  ∧ 
  (∀ x, event_B Ω a b x ∨ event_E Ω a b x) :=
by
  sorry

end mutually_exclusive_and_complementary_l261_261223


namespace find_divisible_number_l261_261504

theorem find_divisible_number (n : ℕ) (h : n - 12 = 1008) : 
  LCM 36 48 56 = 1008 ∧ (n - 12) % 1008 = 0 := by
sorry

end find_divisible_number_l261_261504


namespace correct_propositions_l261_261097

variables {α : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]
variables {β : Type*} [LinearOrder β] [TopologicalSpace β] [OrderTopology β]
variables {f g : α → β}
variables {a b c d : α}

def prop_1 := ∃ x0 ∈ set.Icc a b, f x0 > g x0 ∧ max (set.Icc a b) f > min (set.Icc a b) g
def prop_2 := (∀ x ∈ set.Icc a b, f x > g x) ∧ min (set.Icc a b) (λ x, f x - g x) > 0
def prop_3 := (∀ x1 ∈ set.Icc a b, ∀ x2 ∈ set.Icc c d, f x1 > g x2) ∧ min (set.Icc a b) f > max (set.Icc c d) g
def prop_4 := ∃ x1 ∈ set.Icc a b, ∃ x2 ∈ set.Icc c d, f x1 > g x2 ∧ min (set.Icc a b) f > min (set.Icc c d) g

theorem correct_propositions : (prop_2 ∧ prop_3) := by sorry

end correct_propositions_l261_261097


namespace average_value_of_items_in_loot_box_l261_261254

-- Definitions as per the given conditions
def cost_per_loot_box : ℝ := 5
def total_spent : ℝ := 40
def total_loss : ℝ := 12

-- Proving the average value of items inside each loot box
theorem average_value_of_items_in_loot_box :
  (total_spent - total_loss) / (total_spent / cost_per_loot_box) = 3.50 := by
  sorry

end average_value_of_items_in_loot_box_l261_261254


namespace unique_intersection_line_hyperbola_l261_261342

theorem unique_intersection_line_hyperbola (k : ℝ) :
  (∀ x, (y = k * (x - sqrt 2)) → x^2 - y^2 = 1 -> ∃! p : ℝ × ℝ, p ∈ {(x, y) | x^2 - y^2 = 1} ∩ {(x, y) | y = k * (x - sqrt 2)}) →
  (k = 1 ∨ k = -1) :=
sorry

end unique_intersection_line_hyperbola_l261_261342


namespace sqrt_eq_solution_l261_261488

theorem sqrt_eq_solution (z : ℝ) : ∃ z, sqrt (10 + 3 * z) = 14 ↔ z = 62 :=
by sorry

end sqrt_eq_solution_l261_261488


namespace num_integer_pairs_satisfying_m_plus_n_eq_mn_l261_261724

theorem num_integer_pairs_satisfying_m_plus_n_eq_mn : 
  ∃ (m n : ℤ), (m + n = m * n) ∧ ∀ (m n : ℤ), (m + n = m * n) → 
  (m = 0 ∧ n = 0) ∨ (m = 2 ∧ n = 2) :=
by
  sorry

end num_integer_pairs_satisfying_m_plus_n_eq_mn_l261_261724


namespace range_of_function_f_area_of_triangle_l261_261969

noncomputable def function_f (x : ℝ) : ℝ :=
  2 * sqrt 3 * sin x ^ 2 + 2 * sin x * cos x - sqrt 3

theorem range_of_function_f :
  (∀ x : ℝ, (π / 3) ≤ x ∧ x ≤ 11 * π / 24 → sqrt 3 ≤ function_f x ∧ function_f x ≤ 2) := 
sorry

theorem area_of_triangle (a b r : ℝ) (h1 : a = sqrt 3) (h2 : b = 2) (h3 : r = 3 * sqrt 2 / 4) :
  ∃ S : ℝ, S = sqrt 2 :=
sorry

end range_of_function_f_area_of_triangle_l261_261969


namespace isosceles_triangle_perimeter_l261_261614

open Classical

noncomputable def valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem isosceles_triangle_perimeter (a b c : ℕ) (h_iso : a = b ∨ b = c ∨ a = c)
  (h_len : {a, b, c} = {3, 7, 7} ∨ {a, b, c} = {7, 3, 7} ∨ {a, b, c} = {7, 7, 3})
  (h_valid : valid_triangle a b c) :
  a + b + c = 17 :=
by
  sorry

end isosceles_triangle_perimeter_l261_261614


namespace polygon_diagonals_l261_261816

theorem polygon_diagonals (n : ℕ) (k_0 k_1 k_2 : ℕ)
  (h1 : 2 * k_2 + k_1 = n)
  (h2 : k_2 + k_1 + k_0 = n - 2) :
  k_2 ≥ 2 :=
sorry

end polygon_diagonals_l261_261816


namespace ratio_of_cost_to_selling_price_l261_261729

-- Define the given conditions
def cost_price (CP : ℝ) := CP
def selling_price (CP : ℝ) : ℝ := CP + 0.25 * CP

-- Lean statement for the problem
theorem ratio_of_cost_to_selling_price (CP SP : ℝ) (h1 : SP = selling_price CP) : CP / SP = 4 / 5 :=
by
  sorry

end ratio_of_cost_to_selling_price_l261_261729


namespace find_x_l261_261142

noncomputable def x : ℝ :=
  0.49

theorem find_x (h : (Real.sqrt 1.21) / (Real.sqrt 0.81) + (Real.sqrt 0.81) / (Real.sqrt x) = 2.507936507936508) : 
  x = 0.49 :=
sorry

end find_x_l261_261142


namespace derivative_at_1_l261_261183

noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1 * x + (1 / 2) * x^2

theorem derivative_at_1 : deriv f 1 = Real.exp 1 := 
by 
  sorry

end derivative_at_1_l261_261183


namespace large_number_divisible_by_12_l261_261865

theorem large_number_divisible_by_12 : (Nat.digits 10 ((71 * 10^168) + (72 * 10^166) + (73 * 10^164) + (74 * 10^162) + (75 * 10^160) + (76 * 10^158) + (77 * 10^156) + (78 * 10^154) + (79 * 10^152) + (80 * 10^150) + (81 * 10^148) + (82 * 10^146) + (83 * 10^144) + 84)).foldl (λ x y => x * 10 + y) 0 % 12 = 0 := 
sorry

end large_number_divisible_by_12_l261_261865


namespace twenty_four_point_solution_l261_261621

theorem twenty_four_point_solution : (5 - (1 / 5)) * 5 = 24 := 
by 
  sorry

end twenty_four_point_solution_l261_261621


namespace isosceles_right_triangle_area_proof_l261_261080

noncomputable def side_length_square_of_area (A : ℝ) : ℝ :=
  real.sqrt A

def isosceles_right_triangle_area (leg_length : ℝ) : ℝ :=
  (1 / 2) * leg_length * leg_length

theorem isosceles_right_triangle_area_proof :
  side_length_square_of_area 64 = 8 →
  side_length_square_of_area 256 = 16 →
  isosceles_right_triangle_area 8 = 32 :=
by
  intros h1 h2
  rw [side_length_square_of_area, isosceles_right_triangle_area]
  sorry

end isosceles_right_triangle_area_proof_l261_261080


namespace find_minimum_value_of_fx_l261_261214

theorem find_minimum_value_of_fx :
  ∃ (x : ℝ), f x = -9/4 ∧ (∀ y : ℝ, f y ≥ -9/4) :=
begin
  -- Define the function f
  let f : ℝ → ℝ := λ x, (x - 1) * (x + 2) * (x^2 - x - 2),
  
  -- Symmetry condition implies proof based on substituted values
    sorry
end

end find_minimum_value_of_fx_l261_261214


namespace scientific_notation_correct_l261_261354

def original_number : ℕ := 31900

def scientific_notation_option_A : ℝ := 3.19 * 10^2
def scientific_notation_option_B : ℝ := 0.319 * 10^3
def scientific_notation_option_C : ℝ := 3.19 * 10^4
def scientific_notation_option_D : ℝ := 0.319 * 10^5

theorem scientific_notation_correct :
  original_number = 31900 ∧ scientific_notation_option_C = 3.19 * 10^4 ∧ (original_number : ℝ) = scientific_notation_option_C := 
by 
  sorry

end scientific_notation_correct_l261_261354


namespace find_three_digit_number_l261_261489

def is_three_digit_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000

def digits_sum (n : ℕ) : ℕ :=
  let a := n / 100
  let b := (n % 100) / 10
  let c := n % 10
  a + b + c

theorem find_three_digit_number : 
  ∃ n : ℕ, is_three_digit_number n ∧ n^2 = (digits_sum n)^5 ∧ n = 243 :=
sorry

end find_three_digit_number_l261_261489


namespace greatest_k_inequality_l261_261134

theorem greatest_k_inequality (n : ℤ) (h : n ≥ 2) :  
  (⟦n / Real.sqrt 3⟧ + 1) > n^2 / (Real.sqrt (3 * n^2 - 5)) := 
sorry

end greatest_k_inequality_l261_261134


namespace necessary_not_sufficient_l261_261790

variable (a b : ℝ)

theorem necessary_not_sufficient : 
  (a > b) -> ¬ (a > b+1) ∨ (a > b+1 ∧ a > b) :=
by
  intro h
  have h1 : ¬ (a > b+1) := sorry
  have h2 : (a > b+1 -> a > b) := sorry
  exact Or.inl h1

end necessary_not_sufficient_l261_261790


namespace appended_number_divisible_by_12_l261_261858

theorem appended_number_divisible_by_12 :
  ∃ N, (N = 88) ∧ (∀ n, n ∈ finset.range N \ 71 → (let large_number := (list.range (N + 1)).filter (λ x, 71 ≤ x ∧ x ≤ N) in
       (list.foldr (λ a b, a * 100 + b) 0 large_number) % 12 = 0)) :=
by
  sorry

end appended_number_divisible_by_12_l261_261858


namespace distance_AB_l261_261870

noncomputable theory
open_locale classical

variables (A B : Type) (time_to_meet : ℝ) (xiao_cheng_speed xiao_chen_speed additional_speed total_speed : ℝ)

def initial_conditions (start_time: ℝ) : Prop :=
(time_to_meet = 5/3) ∧
(additional_speed = 10) ∧ 
(total_speed = xiao_cheng_speed + xiao_chen_speed) ∧
(time_to_meet - 1/6) * (xiao_cheng_speed + additional_speed) = time_to_meet * total_speed - (time_to_meet - 1/6) * xiao_cheng_speed ∧
(time_to_meet - 1/3) * xiao_chen_speed = (time_to_meet - 1/3) * total_speed - 1/3 * xiao_chen_speed

theorem distance_AB (start_time: ℝ) (distance : ℝ) : 
initial_conditions A B start_time → 
total_speed = 90 → 
distance = (5/3) * total_speed := 
sorry

end distance_AB_l261_261870


namespace smallest_solution_of_quartic_equation_l261_261749

theorem smallest_solution_of_quartic_equation :
  ∃ x : ℝ, (x^4 - 50 * x^2 + 625 = 0) ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y := sorry

end smallest_solution_of_quartic_equation_l261_261749


namespace bonnets_per_orphanage_l261_261668

/--
Mrs. Young makes bonnets for kids in the orphanage.
On Monday, she made 10 bonnets.
On Tuesday and Wednesday combined she made twice more than on Monday.
On Thursday she made 5 more than on Monday.
On Friday she made 5 less than on Thursday.
She divided up the bonnets evenly and sent them to 5 orphanages.
Prove that the number of bonnets Mrs. Young sent to each orphanage is 11.
-/
theorem bonnets_per_orphanage :
  let monday := 10
  let tuesday_wednesday := 2 * monday
  let thursday := monday + 5
  let friday := thursday - 5
  let total_bonnets := monday + tuesday_wednesday + thursday + friday
  let orphanages := 5
  total_bonnets / orphanages = 11 :=
by
  sorry

end bonnets_per_orphanage_l261_261668


namespace soccer_team_wins_l261_261081

theorem soccer_team_wins :
  let total_games := 130
  let win_percentage := 0.60
  total_games * win_percentage = 78 :=
by
  let total_games := 130
  let win_percentage := 0.60
  show total_games * win_percentage = 78
  sorry

end soccer_team_wins_l261_261081


namespace colored_isosceles_triangle_exists_l261_261269

theorem colored_isosceles_triangle_exists (n : ℤ) (n_gt_3 : n > 3)
    (polygon : Fin (4 * ↑n + 1) → Prop) (color_count : ∃ C : Finset (Fin (4 * ↑n + 1)), C.card = (2:ℤ) * (n:ℕ)) :
    ∃ (v1 v2 v3 : Fin (4 * ↑n + 1)), v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧ 
        (v2 - v1) % (4 * n + 1) = (v3 - v1) % (4 * n + 1) :=
sorry

end colored_isosceles_triangle_exists_l261_261269


namespace probability_odd_sum_and_product_gt_100_l261_261312

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def valid_pairs (a b : ℕ) : Prop :=
  a ≠ b ∧ 1 ≤ a ∧ a ≤ 19 ∧ 1 ≤ b ∧ b ≤ 19 ∧ is_odd (a + b) ∧ a * b > 100

theorem probability_odd_sum_and_product_gt_100 : 
  (finset.card (finset.filter valid_pairs (finset.Icc 1 19).product (finset.Icc 1 19))) = 3 ∧
  (finset.card (finset.Icc 1 19).product (finset.Icc 1 19)) = 171 →
  (finset.card (finset.filter valid_pairs (finset.Icc 1 19).product (finset.Icc 1 19)).to_real / 
   (finset.card (finset.Icc 1 19).product (finset.Icc 1 19)).to_real = 1/57) :=
by
  sorry

end probability_odd_sum_and_product_gt_100_l261_261312


namespace sin_sum_arcsin_arctan_l261_261474

theorem sin_sum_arcsin_arctan :
  let a := Real.arcsin (4 / 5)
  let b := Real.arctan (1 / 2)
  Real.sin (a + b) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end sin_sum_arcsin_arctan_l261_261474


namespace sin_sum_arcsin_arctan_l261_261480

-- Definitions matching the conditions
def a := Real.arcsin (4 / 5)
def b := Real.arctan (1 / 2)

-- Theorem stating the question and expected answer
theorem sin_sum_arcsin_arctan : 
  Real.sin (a + b) = (11 * Real.sqrt 5) / 25 := 
by 
  sorry

end sin_sum_arcsin_arctan_l261_261480


namespace bob_age_l261_261443

variable {b j : ℝ}

theorem bob_age (h1 : b = 3 * j - 20) (h2 : b + j = 75) : b = 51 := by
  sorry

end bob_age_l261_261443


namespace monkey_climbs_tree_in_15_hours_l261_261415

theorem monkey_climbs_tree_in_15_hours :
  ∀ (h t u v : ℕ), t = 51 ∧ u = 7 ∧ v = 4 → (u - v) * (div (t - u) (u - v)) + u ≥ t → t = 51 ∧ u = 7 ∧ v = 4 → div (t - u) (u - v) + 1 = 15 :=
by {
  intros h t u v,
  intros htuv_htuv h2 htuv_htuv,
  sorry
}

end monkey_climbs_tree_in_15_hours_l261_261415


namespace solve_for_x_l261_261588

theorem solve_for_x (x : Real) : 
  27^(x + 1) = 80 + 3 * 27^x → x = (Real.log (10 / 3)) / (3 * Real.log 3) :=
sorry

end solve_for_x_l261_261588


namespace number_of_equilateral_triangles_in_dodecagon_is_4_number_of_scalene_triangles_in_dodecagon_is_168_l261_261713

-- Part (a)
theorem number_of_equilateral_triangles_in_dodecagon_is_4 :
  ∀ (dodecagon : Finset (Fin 12)), dodecagon.card = 12 →
  (∃ t : Finset (Fin 12), t.card = 3 ∧
  (∃ (a b c : Fin 12), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ t = {a, b, c} ∧
  ∃ (n : ℕ), n = 4 ∧ ∀ i ∈ t, ∃ j k ∈ t, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧
  ((j = i + Fin.of_nat n) ∨ (k = i + Fin.of_nat n)))) →
   Finset.card {t : Finset (Fin 12) | t.card = 3 ∧
  (∃ (a b c : Fin 12), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  ∃ (n : ℕ), n = 4 ∧ ∀ i ∈ t, ∃ j k ∈ t, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧
  ((j = i + Fin.of_nat n) ∨ (k = i + Fin.of_nat n)))} = 4 := by
  sorry

-- Part (b)
theorem number_of_scalene_triangles_in_dodecagon_is_168 :
  ∀ (dodecagon : Finset (Fin 12)), dodecagon.card = 12 →
  let totalTriangles := dodecagon.powerset.filter (λ s, s.card = 3).card,
      equilateralTriangles := 4,
      isoscelesTriangles :=
        Finset.sum dodecagon (λ v, (dodecagon.filter (λ w, v ≠ w)).powerset.filter
          (λ s, s.card = 2 ∧ ∀ x y ∈ s, v + 4 = x ∨ v + 4 = y ∨ x + 4 = y)).card in
  totalTriangles - isoscelesTriangles - equilateralTriangles = 168 := by
  sorry

end number_of_equilateral_triangles_in_dodecagon_is_4_number_of_scalene_triangles_in_dodecagon_is_168_l261_261713


namespace min_sum_of_consecutive_natural_numbers_l261_261733

theorem min_sum_of_consecutive_natural_numbers (a b c : ℕ) 
  (h1 : a + 1 = b)
  (h2 : a + 2 = c)
  (h3 : a % 9 = 0)
  (h4 : b % 8 = 0)
  (h5 : c % 7 = 0) :
  a + b + c = 1488 :=
sorry

end min_sum_of_consecutive_natural_numbers_l261_261733


namespace log3_increasing_on_gt1_l261_261304

-- Conditions/Definitions
def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

def f (x : ℝ) : ℝ := log3 (x - 1)

-- Theorem statement to prove
theorem log3_increasing_on_gt1 : ∀ x y : ℝ, 1 < x → x < y → f x < f y :=
by sorry

end log3_increasing_on_gt1_l261_261304


namespace find_last_num_divisible_by_12_stopping_at_84_l261_261848

theorem find_last_num_divisible_by_12_stopping_at_84 :
  ∃ N, (N = 84) ∧ (71 ≤ N) ∧ (let concatenated := string.join ((list.range (N - 70)).map (λ i, (string.of_nat (i + 71)))) in 
    (nat.divisible (int.of_nat (string.to_nat concatenated)) 12)) :=
begin
  sorry
end

end find_last_num_divisible_by_12_stopping_at_84_l261_261848


namespace minimum_ratio_proof_l261_261398

def rectangle (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0

def diagonal_length (a b : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2)

def first_swimmer_distance (a b : ℝ) : ℝ :=
  2 * diagonal_length a b

def second_swimmer_ratio (a : ℝ) : ℝ :=
  2018 / (2018 + 2019) * a

def minimum_ratio (a b : ℝ) : ℝ :=
  1

theorem minimum_ratio_proof (a b : ℝ) (h : rectangle a b) :
  minimum_ratio a b = 1 := 
sorry

end minimum_ratio_proof_l261_261398


namespace soldier_initial_consumption_l261_261608

theorem soldier_initial_consumption :
  ∀ (s d1 n : ℕ) (c2 d2 : ℝ), 
    s = 1200 → d1 = 30 → n = 528 → c2 = 2.5 → d2 = 25 → 
    36000 * (x : ℝ) = 108000 → x = 3 := 
by {
  sorry
}

end soldier_initial_consumption_l261_261608


namespace part1_part2_l261_261187

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
    (Real.sin x + 1) / Real.exp (Real.pi) - a / Real.exp x

theorem part1 (x : ℝ) (hx : x ∈ Set.Icc (-Real.pi) 0) : f x (-1) > 1 := by
    sorry

theorem part2 (h : (Set.Icc Real.pi (2 * Real.pi)).countable) : 
    ((fun x => f x 1 = 0) ⁻¹' {true}).count (Set.Icc Real.pi (2 * Real.pi)) = 2 := by
    sorry

end part1_part2_l261_261187


namespace min_quotient_l261_261137

def digits_distinct (a b c : ℕ) : Prop := 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

def quotient (a b c : ℕ) : ℚ := 
  (100 * a + 10 * b + c : ℚ) / (a + b + c : ℚ)

theorem min_quotient (a b c : ℕ) (h1 : b > 3) (h2 : c ≠ b) (h3: digits_distinct a b c) : 
  quotient a b c ≥ 19.62 :=
sorry

end min_quotient_l261_261137


namespace polynomial_roots_square_real_l261_261657

theorem polynomial_roots_square_real (
  (n : ℕ) 
  (a : fin (n + 1) → ℂ) 
  (b : fin (n + 1) → ℂ) 
  (p q : polynomial ℂ)
  (hp : p = polynomial.C a[0] + ∑ i in finset.range n, polynomial.C a[i+1] * polynomial.X ^ (n-i)) 
  (hq : q = polynomial.C b[0] + ∑ i in finset.range n, polynomial.C b[i+1] * polynomial.X ^ (n-i)) 
  (roots_p_q : (∀ (x ∈ finset.univ : finset ℂ), x ∈ p.roots → x^2 ∈ q.roots)) 
  (h_even : ∑ i in finset.filter (λ x, even x) (finset.range (n + 1)), a i ∈ ℝ) 
  (h_odd : ∑ i in finset.filter (λ x, odd x) (finset.range (n + 1)), a i ∈ ℝ)
: (∑ i in finset.range (n + 1), b i) ∈ ℝ := 
sorry

end polynomial_roots_square_real_l261_261657


namespace find_a_range_l261_261184

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (5 * a - 1) * x + 4 * a else Real.log x / Real.log a

def is_decreasing (a : ℝ) : Prop :=
  ∀ (x y : ℝ), x < y → f a x ≥ f a y

theorem find_a_range :
  {a : ℝ | is_decreasing a} = set.Icc (1/9 : ℝ) (1/5 : ℝ) :=
sorry

end find_a_range_l261_261184


namespace OC_expression_l261_261407

theorem OC_expression (O A B C D : Point) (r : ℝ) (s c θ : ℝ) 
  (h1 : dist O A = 2) 
  (h2 : dist O A = r)
  (h3 : tangent_point O A B) 
  (h4 : tangent_point O D B) 
  (h5 : A ≠ D) 
  (h6 : angle A O B = θ)
  (h7 : lies_on_line C (line_through O A)) 
  (h8 : bisects_angle B C (angle A B O)) 
  (h9 : angle A O D = 2 * θ)
  (h10 : s = sin θ)
  (h11 : c = cos θ) :
  dist O C = 2 / (1 + s) :=
sorry

end OC_expression_l261_261407


namespace smallest_solution_x4_minus_50x2_plus_625_eq_0_l261_261753

theorem smallest_solution_x4_minus_50x2_plus_625_eq_0 : ∃ x : ℝ, x^4 - 50 * x^2 + 625 = 0 ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y := 
sorry

end smallest_solution_x4_minus_50x2_plus_625_eq_0_l261_261753


namespace large_number_divisible_by_12_l261_261868

theorem large_number_divisible_by_12 : (Nat.digits 10 ((71 * 10^168) + (72 * 10^166) + (73 * 10^164) + (74 * 10^162) + (75 * 10^160) + (76 * 10^158) + (77 * 10^156) + (78 * 10^154) + (79 * 10^152) + (80 * 10^150) + (81 * 10^148) + (82 * 10^146) + (83 * 10^144) + 84)).foldl (λ x y => x * 10 + y) 0 % 12 = 0 := 
sorry

end large_number_divisible_by_12_l261_261868


namespace inequality_limit_eq_l261_261302

variable (α β : ℝ) (x : ℕ → ℝ)

-- Define S_\alpha(x)
noncomputable def S (a : ℝ) (x : ℕ → ℝ) : ℝ := sorry

-- Define S_0(x) as a special case
noncomputable def S_0 (x : ℕ → ℝ) : ℝ := S 0 x

-- Definition placeholder for inequality statement
theorem inequality (h1 : α < 0) (h2 : 0 < β) : S α x ≤ S 0 x ∧ S 0 x ≤ S β x := 
sorry

-- Definition placeholder for limit statement
theorem limit_eq (h1 : α → 0) (h2 : β → 0) : 
  (lim ((α → -0), S α x) = S 0 x) ∧ (lim ((β → +0), S β x) = S 0 x) :=
  sorry

end inequality_limit_eq_l261_261302


namespace spring_length_ratio_l261_261380

-- Define the given lengths in the problem
def original_length : ℝ := 4.25
def increase_length : ℝ := 29.75
def length_with_weight : ℝ := original_length + increase_length

-- State the final proof goal
theorem spring_length_ratio : (length_with_weight / original_length) ≈ 8 := by
  sorry

end spring_length_ratio_l261_261380


namespace count_books_before_yardsale_l261_261286

variable (magazines books_after_yardsale books_bought books_before_yardsale : ℕ)

-- Given conditions
def Melanie_has_magazines : magazines = 31 := sorry
def Melanie_now_has_books : books_after_yardsale = 87 := sorry
def Melanie_bought_books : books_bought = 46 := sorry

-- Theorem to prove
theorem count_books_before_yardsale 
  (H1 : Melanie_has_magazines magazines)
  (H2 : Melanie_now_has_books books_after_yardsale)
  (H3 : Melanie_bought_books books_bought) :
  books_before_yardsale = 41 :=
begin
  have h : books_before_yardsale = books_after_yardsale - books_bought,
  { sorry },
  rw [H2, H3] at h,
  exact h,
end

end count_books_before_yardsale_l261_261286


namespace rightmost_three_digits_of_7_pow_1987_l261_261116

theorem rightmost_three_digits_of_7_pow_1987 :
  7^1987 % 1000 = 543 :=
by
  sorry

end rightmost_three_digits_of_7_pow_1987_l261_261116


namespace largest_cube_divisor_of_N_largest_square_divisor_of_N_l261_261096

def N : ℕ := List.prod (List.range' 1 17)

theorem largest_cube_divisor_of_N (N : ℕ) (hN : N = List.prod (List.range' 1 17)) : ∃ k : ℕ, k ^ 3 = 1440 ^ 3 :=
by
  use 1440
  sorry

theorem largest_square_divisor_of_N (N : ℕ) (hN : N = List.prod (List.range' 1 17)) : ∃ k : ℕ, k ^ 2 = 120960 ^ 2 :=
by
  use 120960
  sorry

end largest_cube_divisor_of_N_largest_square_divisor_of_N_l261_261096


namespace drawing_consecutive_balls_l261_261401

-- The definition for the balls in the bin and the condition under which we draw them
def consecutive_draws (n m k : ℕ) : Prop :=
  n + k - 1 < m

-- The main theorem stating the number of ways to draw the balls as required by the problem
theorem drawing_consecutive_balls : consecutive_draws 4 20 1 ↔ 17 := 
by sorry

end drawing_consecutive_balls_l261_261401


namespace trains_crossing_time_l261_261017

theorem trains_crossing_time (train_length : ℕ) (speed_km_hr : ℕ) (relative_speed_conversion_factor : ℕ) 
  (total_distance : ℕ) (time : ℕ) :
  train_length = 120 →
  speed_km_hr = 18 →
  relative_speed_conversion_factor = (1000 / 3600) →
  total_distance = train_length * 2 →
  time = total_distance / ((speed_km_hr * relative_speed_conversion_factor) * 2) →
  time = 24 := by
  intros h_train_length h_speed_km_hr h_conversion_factor h_total_distance h_time
  rw [h_train_length, h_speed_km_hr, h_conversion_factor, h_total_distance, h_time]
  sorry

end trains_crossing_time_l261_261017


namespace fuel_tank_ethanol_l261_261035

theorem fuel_tank_ethanol (x : ℝ) (H : 0.12 * x + 0.16 * (208 - x) = 30) : x = 82 := 
by
  sorry

end fuel_tank_ethanol_l261_261035


namespace students_with_failing_grades_cyclists_in_swimming_section_l261_261607

-- Define the total number of students
def total_students : ℕ := 25

-- Define the number of students in specific sports sections
def cyclists : ℕ := 8
def swimmers : ℕ := 13
def skiers : ℕ := 17

-- No student participates in all three sections
def no_all_three_sections : Prop := ∀ (c ∈ cyclists) (s ∈ swimmers) (k ∈ skiers), c ≠ s ∧ c ≠ k ∧ s ≠ k

-- All athletes receive grades of 3 or higher
def athletes_grades := ∀ (a ∈ total_students - 6), a ≥ 3

-- Define the number of students with non-failing grades
def non_failed_students : ℕ := total_students - 6

-- The problem ultimately boils down to proving:
theorem students_with_failing_grades : (total_students - non_failed_students) = 0 :=
by
  -- Failing grade condition is given by definition.
  sorry

theorem cyclists_in_swimming_section : (cyclists ∩ swimmers).card = 2 :=
by
  -- Inclusion-exclusion principle and problem constraints.
  let intersect_sum := 38 - total_students,
  have h := 13, -- Derived from steps
  exact intersect_sum
  sorry

end students_with_failing_grades_cyclists_in_swimming_section_l261_261607


namespace shield_area_l261_261395

theorem shield_area (r : ℝ) (π_approx : ℝ) (h_r : r = 1) (h_π : π_approx = 3.14) :
  (1 / 6) * π_approx * r^2 = 0.52 :=
by
  have h1 : r^2 = 1, by rw [h_r, one_pow]
  have h2 : (1 / 6) * π_approx * 1 = (1 / 6) * 3.14, by rw [h1, h_π]
  have h3 : (1 / 6) * 3.14 = 0.52, by norm_num [div_eq_mul_one_div]
  rw [h2, h3]
  sorry

end shield_area_l261_261395


namespace log_division_simplification_l261_261695

theorem log_division_simplification (log_base_half : ℝ → ℝ) (log_base_half_pow5 :  log_base_half (2 ^ 5) = 5 * log_base_half 2)
  (log_base_half_pow1 : log_base_half (2 ^ 1) = 1 * log_base_half 2) :
  (log_base_half 32) / (log_base_half 2) = 5 :=
sorry

end log_division_simplification_l261_261695


namespace count_odd_expressions_l261_261311

-- Definitions of odd and even
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Given conditions
variable (d e : ℤ)
variable (d_odd : is_odd d)
variable (e_even : is_even e)

-- Proposition to prove
theorem count_odd_expressions :
  let exp1 := d + d,
      exp2 := (e + e) * d,
      exp3 := d * d,
      exp4 := d * (e + d)
  in (if is_odd exp1 then 1 else 0) +
     (if is_odd exp2 then 1 else 0) +
     (if is_odd exp3 then 1 else 0) +
     (if is_odd exp4 then 1 else 0) = 2 := by
  sorry

end count_odd_expressions_l261_261311


namespace sum_first_four_terms_eq_12_l261_261947

noncomputable def a : ℕ → ℤ := sorry -- An arithmetic sequence aₙ

-- Given conditions
axiom h1 : a 2 = 4
axiom h2 : a 1 + a 5 = 4 * a 3 - 4

theorem sum_first_four_terms_eq_12 : (a 1 + a 2 + a 3 + a 4) = 12 := 
by {
  sorry
}

end sum_first_four_terms_eq_12_l261_261947


namespace angles_on_axes_l261_261233

theorem angles_on_axes :
  (set_of (α : ℝ) ∃ k : ℤ, α = k * (π / 2)) =
  (set_of (α : ℝ) ∃ k : ℤ, α = k * π) ∪ (set_of (α : ℝ) ∃ k : ℤ, α = k * π + (π / 2)) :=
sorry

end angles_on_axes_l261_261233


namespace term_in_census_is_population_l261_261708

def term_for_entire_set_of_objects : String :=
  "population"

theorem term_in_census_is_population :
  term_for_entire_set_of_objects = "population" :=
sorry

end term_in_census_is_population_l261_261708


namespace parallel_planes_ACG_BEH_l261_261693

structure Vertex :=
  (x y z : ℝ)

def A : Vertex := ⟨1, 0, 0⟩
def B : Vertex := ⟨1, 1, 0⟩
def C : Vertex := ⟨0, 1, 0⟩
def D : Vertex := ⟨0, 0, 0⟩
def E : Vertex := ⟨1, 1, 1⟩
def F : Vertex := ⟨0, 1, 1⟩
def G : Vertex := ⟨0, 0, 1⟩
def H : Vertex := ⟨1, 0, 1⟩

def plane (p1 p2 p3 : Vertex) : ℝ × ℝ × ℝ × ℝ :=
  let a := (p2.y - p1.y) * (p3.z - p1.z) - (p2.z - p1.z) * (p3.y - p1.y)
  let b := (p2.z - p1.z) * (p3.x - p1.x) - (p2.x - p1.x) * (p3.z - p1.z)
  let c := (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
  let d := -(a * p1.x + b * p1.y + c * p1.z)
  (a, b, c, d)

def distance_between_planes (plane1 plane2 : ℝ × ℝ × ℝ × ℝ) : ℝ :=
  let (a1, b1, c1, d1) := plane1
  let (a2, b2, c2, d2) := plane2
  abs (d1 - d2) / real.sqrt (a1 * a1 + b1 * b1 + c1 * c1)

theorem parallel_planes_ACG_BEH :
  let plane_ACG := plane A C G
  let plane_BEH := plane B E H
  plane_ACG.1 = plane_BEH.1 ∧
  plane_ACG.2 = plane_BEH.2 ∧
  plane_ACG.3 = plane_BEH.3 ∧
  distance_between_planes plane_ACG plane_BEH = 1 / real.sqrt 3 :=
by {
  sorry
}

end parallel_planes_ACG_BEH_l261_261693


namespace cost_price_of_item_l261_261066

theorem cost_price_of_item 
  (retail_price : ℝ) (reduction_percentage : ℝ) 
  (additional_discount : ℝ) (profit_percentage : ℝ) 
  (selling_price : ℝ) 
  (h1 : retail_price = 900)
  (h2 : reduction_percentage = 0.1)
  (h3 : additional_discount = 48)
  (h4 : profit_percentage = 0.2)
  (h5 : selling_price = 762) :
  ∃ x : ℝ, selling_price = 1.2 * x ∧ x = 635 := 
by {
  sorry
}

end cost_price_of_item_l261_261066


namespace sin_sum_arcsin_arctan_l261_261476

theorem sin_sum_arcsin_arctan :
  let a := Real.arcsin (4 / 5)
  let b := Real.arctan (1 / 2)
  Real.sin (a + b) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end sin_sum_arcsin_arctan_l261_261476


namespace simplify_expression_is_3_l261_261275

noncomputable def simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 3) : ℝ :=
  1 / (y^2 + z^2 - x^2) + 1 / (x^2 + z^2 - y^2) + 1 / (x^2 + y^2 - z^2)

theorem simplify_expression_is_3 (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 3) :
  simplify_expression x y z hx hy hz h = 3 :=
  sorry

end simplify_expression_is_3_l261_261275


namespace sum_of_x_values_l261_261505

theorem sum_of_x_values (x : ℂ) (h₁ : x ≠ -3) (h₂ : 3 = (x^3 - 3 * x^2 - 10 * x) / (x + 3)) : x + (5 - x) = 5 :=
sorry

end sum_of_x_values_l261_261505


namespace digit_sum_9_l261_261060

def digits := {n : ℕ // n < 10}

theorem digit_sum_9 (a b : digits) 
  (h1 : (4 * 100) + (a.1 * 10) + 3 + 984 = (1 * 1000) + (3 * 100) + (b.1 * 10) + 7) 
  (h2 : (1 + b.1) - (3 + 7) % 11 = 0) 
: a.1 + b.1 = 9 :=
sorry

end digit_sum_9_l261_261060


namespace circumcircles_common_point_l261_261989

-- Definitions from condition
variables (A B C I D P E Q F R S T U : Type)

-- Conditions from problem in Lean types
axiom h1 : incenter I A B C
axiom h2 : center I (meet BC D P) ∧ D nearer_to B than P
axiom h3 : center I (meet CA E Q) ∧ E nearer_to C than Q
axiom h4 : center I (meet AB F R) ∧ F nearer_to A than R
axiom h5 : meet (line E F) (line Q R) = S
axiom h6 : meet (line F D) (line R P) = T
axiom h7 : meet (line D E) (line P Q) = U

-- Theorem that matches the problem statement
theorem circumcircles_common_point :
  has_intersect (circumcircle D U P) (circumcircle E S Q) (circumcircle F T R) = I :=
by sorry

end circumcircles_common_point_l261_261989


namespace ratio_equation_solution_l261_261117

theorem ratio_equation_solution (x : ℝ) :
  (4 + 2 * x) / (6 + 3 * x) = (2 + x) / (3 + 2 * x) → (x = 0 ∨ x = 4) :=
by
  -- the proof steps would go here
  sorry

end ratio_equation_solution_l261_261117


namespace panels_per_home_panels_needed_per_home_l261_261125

theorem panels_per_home (P : ℕ) (total_homes : ℕ) (shortfall : ℕ) (homes_installed : ℕ) :
  total_homes = 20 →
  shortfall = 50 →
  homes_installed = 15 →
  (P - shortfall) / homes_installed = P / total_homes →
  P = 200 :=
by
  intro h1 h2 h3 h4
  sorry

theorem panels_needed_per_home :
  (200 / 20) = 10 :=
by
  sorry

end panels_per_home_panels_needed_per_home_l261_261125


namespace min_value_of_f_l261_261788

-- Define the problem domain: positive real numbers
variables (a b c x y z : ℝ)
variables (hpos_a : a > 0) (hpos_b : b > 0) (hpos_c : c > 0)
variables (hpos_x : x > 0) (hpos_y : y > 0) (hpos_z : z > 0)

-- Define the given equations
variables (h1 : c * y + b * z = a)
variables (h2 : a * z + c * x = b)
variables (h3 : b * x + a * y = c)

-- Define the function f(x, y, z)
noncomputable def f (x y z : ℝ) : ℝ :=
  x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z)

-- The theorem statement: under the given conditions the minimum value of f(x, y, z) is 1/2
theorem min_value_of_f :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    c * y + b * z = a →
    a * z + c * x = b →
    b * x + a * y = c →
    f x y z = 1 / 2) :=
sorry

end min_value_of_f_l261_261788


namespace cos_theta_max_l261_261964

noncomputable def max_cos_theta (θ : ℝ) (f : ℝ → ℝ) : Prop :=
  f = (λ x, Real.sin x - 2 * Real.cos x) ∧ ∀ x, f x ≤ f θ

theorem cos_theta_max (θ : ℝ) :
  max_cos_theta θ (λ x, Real.sin x - 2 * Real.cos x) →
  Real.cos θ = - (2 * Real.sqrt 5) / 5 := by
  sorry

end cos_theta_max_l261_261964


namespace angle_B_degrees_l261_261053

variables (A B C P Q : Type) [AddGroup A]
variables (triangle_ABC : isosceles A B C (base A C))
variables (P_on_CB : on P C B)
variables (Q_on_AB : on Q A B)
variables (eq1 : distance A C = distance A P)
variables (eq2 : distance A P = distance P Q)
variables (eq3 : distance P Q = distance Q B)

theorem angle_B_degrees 
(h1 : is_isosceles triangle_ABC.base)
(h2 : on P_on_CB)
(h3 : on Q_on_AB)
(h4 : distance A C = distance A P)
(h5 : distance A P = distance P Q)
(h6 : distance P Q = distance Q B) :
∠B = 25 * (5/7) :=
sorry

end angle_B_degrees_l261_261053


namespace smallest_positive_period_monotonically_increasing_interval_range_on_interval_l261_261559

noncomputable def f (x : ℝ) : ℝ :=
  sin x ^ 2 + 2 * sqrt 3 * sin x * cos x + 3 * cos x ^ 2

theorem smallest_positive_period :
  ∃ T > 0, ∀ x ∈ ℝ, f(x + T) = f(x) ∧ T = π :=
sorry

theorem monotonically_increasing_interval :
  ∀ k : ℤ, ∀ x ∈ Icc (k * π - π / 3) (k * π + π / 6),
  monotone f :=
sorry

theorem range_on_interval :
  ∀ x ∈ Icc (-π / 6) (π / 3), 1 ≤ f(x) ∧ f(x) ≤ 4 :=
sorry

end smallest_positive_period_monotonically_increasing_interval_range_on_interval_l261_261559


namespace increasing_sequence_sum_bounds_l261_261980

open Nat

def sequence (a : ℕ → ℝ) : Prop :=
(a 1 = 3) ∧ (∀ n, 2 * a (n + 1) = (a n) ^ 2 - 2 * a n + 4)

theorem increasing_sequence (a : ℕ → ℝ) (h : sequence a) : ∀ n, a (n + 1) > a n :=
sorry

theorem sum_bounds (a : ℕ → ℝ) (h : sequence a) (n : ℕ) (hn : 0 < n) :
    (1 / 3) ≤ (∑ i in range n, 1 / a (i + 1)) ∧ (∑ i in range n, 1 / a (i + 1) ≤ 1 - (2 / 3) ^ n) :=
sorry

end increasing_sequence_sum_bounds_l261_261980


namespace nine_rooks_checkerboard_l261_261329

theorem nine_rooks_checkerboard :
  let num_ways_4x4 := 4.factorial
  let num_ways_5x5 := 5.factorial
  num_ways_4x4 * num_ways_5x5 = 2880 :=
by
  sorry

end nine_rooks_checkerboard_l261_261329


namespace problem_statement_l261_261567

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -6 < x ∧ x < 1}

theorem problem_statement : M ∩ N = N := by
  ext x
  constructor
  · intro h
    exact h.2
  · intro h
    exact ⟨h.2, h⟩

end problem_statement_l261_261567


namespace average_percentage_increase_l261_261638

variables (initial_A new_A : ℕ) (initial_B new_B : ℕ) (initial_C new_C : ℕ)

def percentage_increase (initial new : ℕ) : ℝ :=
  (new - initial : ℚ) / initial * 100

theorem average_percentage_increase :
  initial_A = 60 → new_A = 80 →
  initial_B = 100 → new_B = 120 →
  initial_C = 150 → new_C = 180 →
  (percentage_increase initial_A new_A +
  percentage_increase initial_B new_B +
  percentage_increase initial_C new_C) / 3 = 24.44 :=
by
  intros hA1 hA2 hB1 hB2 hC1 hC2
  sorry

end average_percentage_increase_l261_261638


namespace emma_coupons_no_monday_l261_261121

theorem emma_coupons_no_monday : 
  ∃ first_day, 
  first_day = "Thursday" ∧ 
  ∀ n, 0 ≤ n < 5 → ¬ is_monday ((first_day_index "Thursday" + 12 * n) % 7) :=
begin
  sorry
end

/-- Helper function to convert day name to index --/
def first_day_index (day : String) : Nat :=
  match day with
  | "Sunday"    => 0
  | "Monday"    => 1
  | "Tuesday"   => 2
  | "Wednesday" => 3
  | "Thursday"  => 4
  | "Friday"    => 5
  | "Saturday"  => 6
  | _           => 0 -- default case

/-- Helper function to determine if a given day index is Monday --/
def is_monday (day_index : Nat) : Prop :=
  day_index = 1

end emma_coupons_no_monday_l261_261121


namespace problem_I_problem_II_l261_261929

variable (α : ℝ) (hα1 : 0 < α ∧ α < π/2) (h𝑠𝑖𝑛 : Real.sin α = 4/5)

theorem problem_I : Real.tan α = 4/3 := 
by
  sorry

theorem problem_II : (Real.sin (α + π) - 2 * Real.cos (π / 2 + α)) / (-Real.sin (-α) + Real.cos (π + α)) = 4 :=
by
  sorry

end problem_I_problem_II_l261_261929


namespace cost_of_camel_l261_261054

-- Define the costs of camels, horses, oxen, and elephants
variables (C H O E : ℝ)

-- Each definition directly reflects the conditions from the problem.

def condition1 : Prop := 10 * C = 24 * H
def condition2 : Prop := 16 * H = 4 * O
def condition3 : Prop := 6 * O = 4 * E
def condition4 : Prop := 10 * E = 130000

-- The theorem that needs to be proven.
theorem cost_of_camel (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : 
  C = 5200 :=
begin
  sorry
end

end cost_of_camel_l261_261054


namespace dodecahedron_path_count_l261_261819

/-- A regular dodecahedron with constraints on movement between faces. -/
def num_ways_dodecahedron_move : Nat := 810

/-- Proving the number of different ways to move from the top face to the bottom face of a regular dodecahedron via a series of adjacent faces, such that each face is visited at most once, and movement from the lower ring to the upper ring is not allowed is 810. -/
theorem dodecahedron_path_count :
  num_ways_dodecahedron_move = 810 :=
by
  -- Proof goes here
  sorry

end dodecahedron_path_count_l261_261819


namespace triangle_angle_zero_degrees_l261_261611

theorem triangle_angle_zero_degrees {a b c : ℝ} (h : (a + b + c) * (a + b - c) = 4 * a * b) :
  ∃ (C : ℝ), C = 0 ∧ c = 0 :=
sorry

end triangle_angle_zero_degrees_l261_261611


namespace frog_climbs_well_in_22_minutes_l261_261073

theorem frog_climbs_well_in_22_minutes :
  let well_depth := 12
  let climb_per_cycle := 3
  let slip_per_cycle := 1
  let net_gain_per_cycle := climb_per_cycle - slip_per_cycle
  let climb_time := 497 / 60 -- 8 hours and 17 minutes in hours
  let slip_time := climb_time / 3
  let total_cycles_to_reach_nine_meters := 4
  let last_climb := 3
  let total_time_cycles := total_cycles_to_reach_nine_meters * (climb_time + slip_time)
  let total_time := total_time_cycles + climb_time
  total_time * 60 ≈ 22 * 60 := sorry

end frog_climbs_well_in_22_minutes_l261_261073


namespace algebraic_expr_value_at_neg_one_l261_261316

-- Define the expression "3 times the square of x minus 5"
def algebraic_expr (x : ℝ) : ℝ := 3 * x^2 + 5

-- Theorem to state the value when x = -1 is 8
theorem algebraic_expr_value_at_neg_one : algebraic_expr (-1) = 8 := 
by
  -- The steps to prove are skipped with 'sorry'
  sorry

end algebraic_expr_value_at_neg_one_l261_261316


namespace convex_polygon_enclosed_parallelogram_l261_261033

def convex_polygon (P : Type) : Prop := sorry -- Define convex polygon property

def area (P : Type) : ℝ := sorry -- Function to determine the area of a polygon

theorem convex_polygon_enclosed_parallelogram (P : Type) [convex_polygon P] (h : area P = 1) :
  ∃ Q, parallelogram Q ∧ area Q = 2 :=
sorry

end convex_polygon_enclosed_parallelogram_l261_261033


namespace rooks_on_checkerboard_l261_261322

theorem rooks_on_checkerboard :
  let n := 9
  let board := λ (i j : ℕ), (i % 2 = j % 2) -- checkerboard pattern condition
  let (evenCoords := ((fin n).val.enum.filter (λ (i,j), i%2=0 ∧ j%2=0)).length, -- even r-bord condition
       oddCoords := ((fin n).val.enum.filter (λ (i,j), i%2=1 ∧ j%2=1)).length
       in
     even_coords_board_size == 4 ∧ odd_coords_board_size == 5)
  /- Assert that the number of ways to place non-attacking rooks on the black cells is 4! * 5! -/
  then 
    (∃(φ : (fin n).val.enum.map(board).card = 2880,
  sorry

end rooks_on_checkerboard_l261_261322


namespace number_of_routes_from_A_to_L_is_6_l261_261746

def A_to_B_or_E : Prop := True
def B_to_A_or_C_or_F : Prop := True
def C_to_B_or_D_or_G : Prop := True
def D_to_C_or_H : Prop := True
def E_to_A_or_F_or_I : Prop := True
def F_to_B_or_E_or_G_or_J : Prop := True
def G_to_C_or_F_or_H_or_K : Prop := True
def H_to_D_or_G_or_L : Prop := True
def I_to_E_or_J : Prop := True
def J_to_F_or_I_or_K : Prop := True
def K_to_G_or_J_or_L : Prop := True
def L_from_H_or_K : Prop := True

theorem number_of_routes_from_A_to_L_is_6 
  (h1 : A_to_B_or_E)
  (h2 : B_to_A_or_C_or_F)
  (h3 : C_to_B_or_D_or_G)
  (h4 : D_to_C_or_H)
  (h5 : E_to_A_or_F_or_I)
  (h6 : F_to_B_or_E_or_G_or_J)
  (h7 : G_to_C_or_F_or_H_or_K)
  (h8 : H_to_D_or_G_or_L)
  (h9 : I_to_E_or_J)
  (h10 : J_to_F_or_I_or_K)
  (h11 : K_to_G_or_J_or_L)
  (h12 : L_from_H_or_K) : 
  6 = 6 := 
by 
  sorry

end number_of_routes_from_A_to_L_is_6_l261_261746


namespace emily_irises_after_addition_l261_261728

theorem emily_irises_after_addition
  (initial_roses : ℕ)
  (added_roses : ℕ)
  (ratio_irises_roses : ℕ)
  (ratio_roses_irises : ℕ)
  (h_ratio : ratio_irises_roses = 3 ∧ ratio_roses_irises = 7)
  (h_initial_roses : initial_roses = 35)
  (h_added_roses : added_roses = 30) :
  ∃ irises_after_addition : ℕ, irises_after_addition = 27 :=
  by
    sorry

end emily_irises_after_addition_l261_261728


namespace order_of_a_b_c_l261_261264

noncomputable def a : ℝ := Real.log 2 / Real.log 3 -- a = log_3 2
noncomputable def b : ℝ := Real.log 2 -- b = ln 2
noncomputable def c : ℝ := Real.sqrt 5 -- c = 5^(1/2)

theorem order_of_a_b_c : a < b ∧ b < c := by
  sorry

end order_of_a_b_c_l261_261264


namespace greatest_possible_selling_price_l261_261387

theorem greatest_possible_selling_price (n : ℕ) (products : ℕ) (average_price : ℕ) (min_price : ℕ) 
    (count_less_than_1000 : ℕ) :
    products = 25 →
    average_price = 1200 →
    min_price = 400 →
    count_less_than_1000 = 10 →
    (∃ highest_price : ℕ, 
    highest_price = 12000) :=
by {
    intro h_products h_average_price h_min_price h_count_less_than_1000,
    use 12000,
    sorry
}

end greatest_possible_selling_price_l261_261387


namespace max_height_reached_l261_261061

def height (t : ℝ) : ℝ := -20 * t * t - 40 * t + 50

theorem max_height_reached : ∃ t : ℝ, height t = 70 := 
by
  refine ⟨-1, ?_⟩
  sorry

end max_height_reached_l261_261061


namespace final_position_after_120_moves_l261_261814

noncomputable def particle_position (n : ℕ) : ℂ :=
  let ω : ℂ := Complex.exp (Complex.I * Real.pi / 3) in
  let sum_ω_powers := finset.sum (finset.range n) (λ k, ω ^ k) in
  6 * ω^n + 12 * sum_ω_powers

theorem final_position_after_120_moves : particle_position 120 = 1446 :=
by {
  have ω6_eq_1 : Complex.exp (Complex.I * Real.pi * 2) = 1 := by
    rw [Complex.exp_eq_exp_2π_I, Complex.one_rpow],
  have ω_to_120_eq_1 : (Complex.exp (Complex.I * Real.pi / 3)) ^ 120 = 1 := by
    rw [← Complex.exp_nat_mul, mul_comm, nat_cast_mul, mul_comm, mul_assoc, mul_one, mul_comm],
  rw [particle_position, ω_to_120_eq_1, ω6_eq_1, finset.sum_const, nat.cast_id, finset.card_range, mul_comm, mul_assoc],
  norm_num,
  sorry
}

end final_position_after_120_moves_l261_261814


namespace find_a_range_l261_261970

noncomputable def f (x : ℝ) : ℝ := 2 * |x - 1| + Real.log 3 ((x - 1)^2)

theorem find_a_range (a : ℝ) : (∀ x, 1 < x ∧ x ≤ 2 → f(a * x) ≤ f(x + 3)) ↔ (a ≥ -3 / 2 ∧ a ≤ 5 / 2) :=
by
  sorry

end find_a_range_l261_261970


namespace combined_projection_matrix_correct_l261_261262

open Matrix

theorem combined_projection_matrix_correct (v0 : Vector ℝ 2) :
  let v1 := (1 / 17 : ℝ) • (vec.fromAngular "\begin{pmatrix} 4 \\ 1 \end{pmatrix}")
  let v2 := (1 / 5 : ℝ) • (vec.fromAngular "\begin{pmatrix} 1 \\ 2 \end{pmatrix}")
  ∀ x y, (fromAngular "\begin{pmatrix} 4 \\ 1 \end{pmatrix}") * 
    (fromAngular "\begin{pmatrix} 1 \\ 2 \end{pmatrix}") = 
    (fromAngular "\begin{pmatrix} \frac{24}{85} & \frac{6}{85} \\ \frac{48}{85} & \frac{12}{85} \end{pmatrix}") :=
sorry

-- The proof is omitted and represented by 'sorry'

end combined_projection_matrix_correct_l261_261262


namespace smallest_4_digit_multiple_of_3_l261_261022

/-- The smallest 4-digit number is 1000 (and explicitly state the range of 4-digit numbers). -/
def smallest_4_digit : ℕ := 1000

def is_4_digit (n : ℕ) : Prop := n ≥ 1000 ∧ n ≤ 9999

/-- A function to check if a number is divisible by 3. -/
def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

/-- There exists a smallest 4-digit number that is divisible by 3 and that number is 1002. -/
theorem smallest_4_digit_multiple_of_3 : ∃ n, is_4_digit n ∧ divisible_by_3 n ∧ ∀ m, is_4_digit m → divisible_by_3 m → n ≤ m := 
by
  use 1002
  split
  { -- Prove 1002 is a 4-digit number
    exact (by decide : is_4_digit 1002) }
  split
  { -- Prove 1002 is divisible by 3
    exact (by norm_num : divisible_by_3 1002) }
  { -- Prove that it's the smallest 4-digit number divisible by 3
    intros m Hm Hm_div3
    exact (by linarith : 1002 ≤ m) }

end smallest_4_digit_multiple_of_3_l261_261022


namespace option_d_can_form_triangle_l261_261773

noncomputable def satisfies_triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem option_d_can_form_triangle : satisfies_triangle_inequality 2 3 4 :=
by {
  -- Using the triangle inequality theorem to check
  sorry
}

end option_d_can_form_triangle_l261_261773


namespace expected_remaining_bullets_l261_261811

noncomputable def prob_hit := 0.6
noncomputable def prob_miss := 1 - prob_hit
noncomputable def xi_expected_value : ℝ :=
    0 * (prob_miss ^ 3) +
    1 * (prob_hit * (prob_miss ^ 2)) +
    2 * (prob_hit * prob_miss) +
    3 * prob_hit

theorem expected_remaining_bullets : xi_expected_value = 2.376 :=
by
    unfold xi_expected_value
    sorry

end expected_remaining_bullets_l261_261811


namespace length_of_DB_l261_261242

open Real

theorem length_of_DB
  (A B C D : Point)
  (h1 : angle A B C = π / 2)
  (h2 : angle A D B = π / 2)
  (h3 : dist A C = 25)
  (h4 : dist A D = 7) :
  dist D B = 3 * sqrt 14 :=
sorry

end length_of_DB_l261_261242


namespace items_left_in_store_l261_261874

theorem items_left_in_store: (4458 - 1561) + 575 = 3472 :=
by 
  sorry

end items_left_in_store_l261_261874


namespace find_last_num_divisible_by_12_stopping_at_84_l261_261847

theorem find_last_num_divisible_by_12_stopping_at_84 :
  ∃ N, (N = 84) ∧ (71 ≤ N) ∧ (let concatenated := string.join ((list.range (N - 70)).map (λ i, (string.of_nat (i + 71)))) in 
    (nat.divisible (int.of_nat (string.to_nat concatenated)) 12)) :=
begin
  sorry
end

end find_last_num_divisible_by_12_stopping_at_84_l261_261847


namespace clara_cookies_l261_261454

theorem clara_cookies (x : ℕ) :
  50 * 12 + x * 20 + 70 * 16 = 3320 → x = 80 :=
by
  sorry

end clara_cookies_l261_261454


namespace sum_of_series_l261_261566

theorem sum_of_series : 
  (x : ℝ) (y : ℝ), (M : Set ℝ) (N : Set ℝ), 
  M = {x, x * y, Real.log (x * y)} → 
  N = {0, |x|, y} → 
  M = N →
  (x + (1 / y) + x^2 + (1 / y^2) + x^3 + (1 / y^3) + ... + x^2001 + (1 / y^2001)) = -2 :=
sorry

end sum_of_series_l261_261566


namespace min_value_g_of_f_l261_261972

theorem min_value_g_of_f (a c k : ℝ) (h_a : a = 2) (h_c : c = -3) :
  let f := λ x : ℝ, 2 * x^2 + x - 3
  let g := λ x : ℝ, 2 * x^2 + (4 * k + 4) * x - 3 in
  let h := λ k : ℝ, 
    if k ≤ -7 then 12 * k + 27
    else if -4 < k ∧ k < -2 then -2 * k^2 - 4 * k - 5
    else 4 * k + 3 in
  ∀ x ∈ set.Icc (1 : ℝ) (3 : ℝ), g x = h k :=
sorry

end min_value_g_of_f_l261_261972


namespace quadratic_inequality_solution_l261_261352

theorem quadratic_inequality_solution :
  {x : ℝ | -3 < x ∧ x < 1} = {x : ℝ | x * (x + 2) < 3} :=
by
  sorry

end quadratic_inequality_solution_l261_261352


namespace collinear_midpoints_l261_261014

theorem collinear_midpoints (
  ΔABC : Triangle ℝ,
  H : Point ℝ,
  h_orthocenter : is_orthocenter H ΔABC,
  l₁ l₂ : Line ℝ,
  h_perpendicular : l₁ ⊥ l₂,
  h_through_orthocenter : H ∈ l₁ ∧ H ∈ l₂
) : ∃ (P₁ P₂ P₃ : Point ℝ), 
  (is_midpoint (segment_intersection_with_line ΔABC.side1 l₁) P₁) ∧
  (is_midpoint (segment_intersection_with_line ΔABC.side2 l₁) P₂) ∧
  (is_midpoint (segment_intersection_with_line ΔABC.side3 l₁) P₃) ∧
  collinear {P₁, P₂, P₃} := sorry

end collinear_midpoints_l261_261014


namespace stationary_white_noise_correlation_correct_l261_261132

noncomputable def stationary_white_noise_correlation (τ : ℝ) (s₀ : ℝ) : ℝ :=
  2 * Math.pi * s₀ * Real.delta τ

theorem stationary_white_noise_correlation_correct {s₀ : ℝ} (τ : ℝ) :
  stationary_white_noise_correlation τ s₀ = 2 * Math.pi * s₀ * Real.delta τ :=
by 
  sorry

end stationary_white_noise_correlation_correct_l261_261132


namespace tan_alpha_eq_l261_261522

theorem tan_alpha_eq : ∀ (α : ℝ),
  (Real.tan (α - (5 * Real.pi / 4)) = 1 / 5) →
  Real.tan α = 3 / 2 :=
by
  intro α h
  sorry

end tan_alpha_eq_l261_261522


namespace large_number_divisible_by_12_l261_261864

theorem large_number_divisible_by_12 : (Nat.digits 10 ((71 * 10^168) + (72 * 10^166) + (73 * 10^164) + (74 * 10^162) + (75 * 10^160) + (76 * 10^158) + (77 * 10^156) + (78 * 10^154) + (79 * 10^152) + (80 * 10^150) + (81 * 10^148) + (82 * 10^146) + (83 * 10^144) + 84)).foldl (λ x y => x * 10 + y) 0 % 12 = 0 := 
sorry

end large_number_divisible_by_12_l261_261864


namespace modulus_of_z_l261_261935

-- Define the complex number z
def z : ℂ := (1 - complex.I) * (1 + complex.I)

-- Theorem statement: The modulus of z is equal to 2
theorem modulus_of_z : complex.abs z = 2 := 
by sorry

end modulus_of_z_l261_261935


namespace share_ratio_l261_261052

theorem share_ratio (A B C : ℝ) (x : ℝ) (h1 : A + B + C = 500) (h2 : A = 200) (h3 : A = x * (B + C)) (h4 : B = (6/9) * (A + C)) :
  A / (B + C) = 2 / 3 :=
by
  sorry

end share_ratio_l261_261052


namespace domain_of_f_log2_l261_261957

theorem domain_of_f_log2 (f : ℝ → ℝ) :
  (∀ x, 4 ≤ x ∧ x ≤ 9 → f (x^2 - 1) ∈ set.univ) →
  (∀ x, 2^15 ≤ x ∧ x ≤ 2^80 → f (Real.log x / Real.log 2) ∈ set.univ) :=
begin
  intros h_dom1 x h_bound,
  sorry
end

end domain_of_f_log2_l261_261957


namespace length_BC_inscribed_circle_l261_261727

-- Define the setup
variables {A B C O : Point}
variables {r : ℝ} (h_r : r = 4)
variables (α : ℝ) (h_α : α = 30) -- angle unit in degrees

-- Use the assumption and setup to state the proof problem
theorem length_BC_inscribed_circle :
  ∃ (BC : ℝ), BC = 4 * Real.sqrt 3 :=
begin
  sorry
end

end length_BC_inscribed_circle_l261_261727


namespace add_expression_l261_261400

theorem add_expression {k : ℕ} :
  (2 * k + 2) + (2 * k + 3) = (2 * k + 2) + (2 * k + 3) := sorry

end add_expression_l261_261400


namespace sin_arcsin_plus_arctan_l261_261482

theorem sin_arcsin_plus_arctan (a b : ℝ) (ha : a = Real.arcsin (4/5)) (hb : b = Real.arctan (1/2)) :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2)) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end sin_arcsin_plus_arctan_l261_261482


namespace area_of_T_l261_261263

noncomputable def ω : ℂ := -1/2 + (1/2) * complex.I * real.sqrt 3

def T (a b c : ℝ) : ℂ := 2 * a + b * ω + c * ω^2

theorem area_of_T : (∃ a b c : ℝ, 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2) →
  (complex.abs (T 2 2 2 - T 0 0 0) = 6 * real.sqrt 3) :=
by sorry

end area_of_T_l261_261263


namespace tan_alpha_value_l261_261582

-- Define the given variables and conditions
def alpha_in_third_quadrant (α : ℝ) : Prop :=
  π < α ∧ α < 3 * π / 2

def given_equation (α : ℝ) : Prop :=
  tan (π / 4 - α) = (2 / 3) * tan (α + π)

-- The main theorem statement
theorem tan_alpha_value (α : ℝ) (h1 : alpha_in_third_quadrant α) (h2 : given_equation α) : 
  tan α = 1 / 2 :=
by
  sorry

end tan_alpha_value_l261_261582


namespace min_value_P2_l261_261309

noncomputable def P (x : ℝ) : ℝ := sorry

theorem min_value_P2 :
  (∀ x : ℝ, P x = P (x : ℂ).re) ∧
  P 0 = 2 ∧
  P 1 = 3 ∧
  (∀ z : ℂ, P z = 0 → complex.abs z = 1) →
  P 2 = 54 := sorry

end min_value_P2_l261_261309


namespace tin_silver_ratio_l261_261779

theorem tin_silver_ratio (T S : ℝ) (h1 : T + S = 30)
  (h2 : 0.1375 * T + 0.075 * S = 3) :
  T / S = 2 / 3 := 
begin
  sorry
end

end tin_silver_ratio_l261_261779


namespace no_partition_of_square_into_finite_hexagons_with_inner_angles_lt_180_l261_261678

theorem no_partition_of_square_into_finite_hexagons_with_inner_angles_lt_180 :
  ¬ ∃ (hexagons : Finset (Set Point)) (partition : Set Point → Prop),
    (∀ h ∈ hexagons, is_convex_hexagon_with_all_inner_angles_lt_180 h) ∧
    partitions_square partition hexagons :=
sorry

-- Definitions used in the theorem
def is_convex_hexagon_with_all_inner_angles_lt_180 (hexagon : Set Point) : Prop := 
  hexagon.is_convex ∧ hexagon.is_hexagon ∧ ∀ angle ∈ hexagon.inner_angles, angle < 180

def partitions_square (partition : Set Point → Prop) (hexagons : Finset (Set Point)) : Prop := 
  ∀ p ∈ square, partition p = ∃ h ∈ hexagons, p ∈ h ∧
  disjoint_hexagons hexagons

def disjoint_hexagons (hexagons : Finset (Set Point)) : Prop :=
  ∀ (h1 h2 ∈ hexagons), h1 ≠ h2 → (h1 ∩ h2) = ∅

def square : Set Point := { p : Point | p.is_in_square }

end no_partition_of_square_into_finite_hexagons_with_inner_angles_lt_180_l261_261678


namespace find_four_numbers_proportion_l261_261129

theorem find_four_numbers_proportion :
  ∃ (a b c d : ℝ), 
  a + d = 14 ∧
  b + c = 11 ∧
  a^2 + b^2 + c^2 + d^2 = 221 ∧
  a * d = b * c ∧
  a = 12 ∧
  b = 8 ∧
  c = 3 ∧
  d = 2 :=
by
  sorry

end find_four_numbers_proportion_l261_261129


namespace geometric_sequence_first_term_l261_261460

theorem geometric_sequence_first_term (a r : ℝ) (h1 : a * r^2 = 18) (h2 : a * r^4 = 72) : a = 4.5 := by
  sorry

end geometric_sequence_first_term_l261_261460


namespace lim_to_infinity_of_arith_geo_seq_l261_261982

noncomputable def a: ℝ
noncomputable def c: ℝ

theorem lim_to_infinity_of_arith_geo_seq (h1 : a + c = 2) (h2 : (a^2 * c^2 = 1)) 
(h3 : a ≠ c) :
  (Real.lim (λ n, (↑n : ℕ) → (a + c) / (a^2 + c^2)) = 0) := 
by 
  sorry

end lim_to_infinity_of_arith_geo_seq_l261_261982


namespace collinear_points_b_value_l261_261464

theorem collinear_points_b_value (b : ℝ)
    (h : let slope1 := (4 - (-6)) / ((2 * b + 1) - 4)
         let slope2 := (1 - (-6)) / ((-3 * b + 2) - 4)
         slope1 = slope2) :
    b = -1 / 44 :=
by
  have slope1 := (4 - (-6)) / ((2 * b + 1) - 4)
  have slope2 := (1 - (-6)) / ((-3 * b + 2) - 4)
  have := h
  sorry

end collinear_points_b_value_l261_261464


namespace find_a_if_pure_imaginary_l261_261527

variables (a : ℝ)

def z1 : ℂ := complex.of_real a + complex.I * 3
def z2 : ℂ := complex.of_real 3 + complex.I * (-4)

theorem find_a_if_pure_imaginary (h : (z1 a / z2).re = 0) : a = 4 := by
  sorry

end find_a_if_pure_imaginary_l261_261527


namespace four_numbers_modulo_23_l261_261403

theorem four_numbers_modulo_23 (S : Finset ℕ) (hS : ∀ x ∈ S, x ≤ 501) (h_card : S.card = 250) (t : ℤ) :
  ∃ (a1 a2 a3 a4 : ℕ), a1 ∈ S ∧ a2 ∈ S ∧ a3 ∈ S ∧ a4 ∈ S ∧ (a1 + a2 + a3 + a4 : ℤ) ≡ t [MOD 23] :=
sorry

end four_numbers_modulo_23_l261_261403


namespace sin_arcsin_plus_arctan_l261_261481

theorem sin_arcsin_plus_arctan (a b : ℝ) (ha : a = Real.arcsin (4/5)) (hb : b = Real.arctan (1/2)) :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2)) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end sin_arcsin_plus_arctan_l261_261481


namespace shaded_areas_equality_condition_l261_261623

variable (φ : ℝ) (s : ℝ)
variable (0 < φ) (φ < π / 4)
variable (C : Set Point) -- The center of the circle
variable (BCD ACE AB : LineSegment) -- The line segments
variable (tangent : AB.isTangent (Circle C s) φ)
variable (A B D E : Point) -- Various points on the geometry

-- Condition statement
def necessary_and_sufficient_condition : Prop :=
  tan (2 * φ) = 2 * φ

-- The theorem statement
theorem shaded_areas_equality_condition :
  necessary_and_sufficient_condition φ :=
by
  sorry

end shaded_areas_equality_condition_l261_261623


namespace spinner_final_direction_is_west_l261_261249

-- Define the initial conditions and movements
def initial_direction : String := "north"
def clockwise_revolutions : ℚ := 11 / 2
def counterclockwise_revolutions : ℚ := 11 / 4

-- Define the net movement calculation
noncomputable def net_movement : ℚ := clockwise_revolutions - counterclockwise_revolutions

-- Define the function to determine the final direction based on the net movement
def final_direction (movement : ℚ) : String :=
  let fractional_part := movement - movement.toInt
  if fractional_part = 0 then "north"
  else if fractional_part = 0.25 then "east"
  else if fractional_part = 0.5 then "south"
  else if fractional_part = 0.75 then "west"
  else "undefined"

-- Verify that the final direction after the given moves is "west"
theorem spinner_final_direction_is_west :
  final_direction net_movement = "west" :=
by
  sorry

end spinner_final_direction_is_west_l261_261249


namespace minimum_value_of_f_l261_261920

def f (x : ℝ) : ℝ := 3 * (Real.cbrt x) + 4 / (x ^ 2)

theorem minimum_value_of_f (x : ℝ) (hx : 0 < x) : 
  ∃ y, (∀ ε > 0, y ≤ f x ∧ f x = y + ε) ∧ y = 7 :=
sorry

end minimum_value_of_f_l261_261920


namespace appended_number_divisible_by_12_l261_261862

theorem appended_number_divisible_by_12 :
  ∃ N, (N = 88) ∧ (∀ n, n ∈ finset.range N \ 71 → (let large_number := (list.range (N + 1)).filter (λ x, 71 ≤ x ∧ x ≤ N) in
       (list.foldr (λ a b, a * 100 + b) 0 large_number) % 12 = 0)) :=
by
  sorry

end appended_number_divisible_by_12_l261_261862


namespace slope_angle_of_line_l261_261351

theorem slope_angle_of_line (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) :
    ∃ θ ∈ set.Ico 0 real.pi, θ = real.pi - real.arctan (a / b) :=
by
  sorry

end slope_angle_of_line_l261_261351


namespace airbnb_cost_is_3200_l261_261285

variable (TotalCost SharePerPerson NumberOfPeople CarCost AirbnbCost : ℕ)

-- Given conditions
def SharePerPerson := 500
def NumberOfPeople := 8
def CarCost := 800
def TotalCost := SharePerPerson * NumberOfPeople
def AirbnbCost := TotalCost - CarCost

-- Statement to prove
theorem airbnb_cost_is_3200 : AirbnbCost = 3200 := by
  sorry

end airbnb_cost_is_3200_l261_261285


namespace perfect_square_solution_l261_261918

-- Definitions to set up the problem
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m ∣ p, m = 1 ∨ m = p

def is_perfect_square (k : ℕ) : Prop := ∃ m : ℕ, m * m = k

-- Main theorem statement
theorem perfect_square_solution (n p : ℕ) (hn : n ≥ 1) (hp : is_prime p) (hs : is_perfect_square (n * p + n^2)) : 
∃ k : ℕ, n = (p - 1) ^ 2 / 4 := 
sorry

end perfect_square_solution_l261_261918


namespace avg_cost_equals_0_22_l261_261069

-- Definitions based on conditions
def num_pencils : ℕ := 150
def cost_pencils : ℝ := 24.75
def shipping_cost : ℝ := 8.50

-- Calculating total cost and average cost
noncomputable def total_cost : ℝ := cost_pencils + shipping_cost
noncomputable def avg_cost_per_pencil : ℝ := total_cost / num_pencils

-- Lean theorem statement
theorem avg_cost_equals_0_22 : avg_cost_per_pencil = 0.22 :=
by
  sorry

end avg_cost_equals_0_22_l261_261069


namespace area_of_FDBG_l261_261633

theorem area_of_FDBG {A B C D E F G : Type*}
  [EuclideanGeometry] 
  (hABC : IsTriangle A B C)
  (hAB : dist A B = 40)
  (hAC : dist A C = 20)
  (hAreaABC : area A B C = 200)
  (hMidD : midpoint A B = D)
  (hMidE : midpoint A C = E)
  (hBisector : angle_bisector A B C intersects [D, E] at [F, G]) :
  area F D B G = 120 := 
  sorry

end area_of_FDBG_l261_261633


namespace find_CK_length_find_angle_ACB_l261_261453

-- Definitions for given conditions
variable {ABC : Triangle}
variable {O O₁ O₂ : Point}
variable {K₁ K₂ K : Point}
variable {CK₁ BK₂ BC r : ℝ}

-- Conditions from the problem
axiom O₁_center_C : IsCenter O₁ (InscribedCircle C (Triangle ABC))
axiom O₂_center_B : IsCenter O₂ (InscribedCircle B (Triangle ABC))
axiom O_center_ABC : IsCenter O (Incircle (Triangle ABC))
axiom touch_K₁ : TouchesAt (Circle (O₁) r) (Line BC) K₁
axiom touch_K₂ : TouchesAt (Circle (O₂) r) (Line BC) K₂
axiom touch_K : TouchesAt (Incircle (Triangle ABC)) (Line BC) K
axiom CK₁_eq_3 : CK₁ = 3
axiom BK₂_eq_7 : BK₂ = 7
axiom BC_eq_16 : BC = 16
axiom touch_K₃ : TouchesAt (Circle (O₁) r) (Line AC) K₃
axiom O₁_center_OK₁K₃ : IsCenter O₁ (CircumscribedCircle (Triangle O K₁ K₃))

-- Problem a): Prove the length of segment CK
theorem find_CK_length : CK = 24 / 5 := by
  sorry

-- Problem b): Prove the angle ∠ACB
theorem find_angle_ACB : ∠ACB = 2 * Real.arcsin (3 / 5) := by
  sorry

end find_CK_length_find_angle_ACB_l261_261453


namespace find_a_l261_261993

theorem find_a (a : ℝ) (h : ∃ (b : ℝ), (16 * (x : ℝ) * x) + 40 * x + a = (4 * x + b) ^ 2) : a = 25 := sorry

end find_a_l261_261993


namespace inequality_holds_l261_261396

theorem inequality_holds {n : ℕ} (n_pos : 0 < n) 
  (a : Fin n → ℝ) (h : ∀ i, 0 < a i) : 
  (∑ k in Finset.range n, (k + 1) / (∑ j in Finset.range (k + 1), a ⟨j, nat.lt_succ_of_le (Finset.mem_range_succ.1 (Finset.mem_range.2 (lt_of_le_of_lt (nat.zero_le j) (nat.lt_succ_self n - 1)))))])) < 
  4 * (∑ k in Finset.range n, 1 / a ⟨k, (nat.lt_succ_of_le (Finset.mem_range_succ.1 (Finset.mem_range.2 (lt_of_le_of_lt (nat.zero_le k) (nat.lt_succ_self n - 1)))))⟩) :=
sorry

end inequality_holds_l261_261396


namespace area_of_equilateral_triangle_l261_261634

theorem area_of_equilateral_triangle (D E F Q : EuclideanGeometry.Point ℝ) 
  (h_def_eq : Equilateral {D, E, F}) 
  (h_dq : dist D Q = 7) 
  (h_eq : dist E Q = 5) 
  (h_fq : dist F Q = 9) 
  : area ({D, E, F} : EuclideanGeometry.Triangle ℝ) = 35 :=
sorry

end area_of_equilateral_triangle_l261_261634


namespace rooks_on_checkerboard_l261_261330

theorem rooks_on_checkerboard : ∃ n : ℕ, n = 2880 ∧ ∀ (board : fin 9 × fin 9 → Prop), 
    (∀ r1 r2 c1 c2 : fin 9, r1 ≠ r2 → c1 ≠ c2 → board ⟨r1, c1⟩ → board ⟨r2, c2⟩ → false) ↔ 
    (board (0,0) ∨ board (0,1) ∨ board (1,0) ∨ board (1,1) → 
    ∃ f : fin 9 → fin 9, function.injective f ∧ ∀ i, board ⟨i, f i⟩) := 
begin
  use 2880,
  split,
  { refl, },
  sorry
end

end rooks_on_checkerboard_l261_261330


namespace greatest_length_of_pieces_l261_261095

/-- Alicia has three ropes with lengths of 28 inches, 42 inches, and 70 inches.
She wants to cut these ropes into equal length pieces for her art project, and she doesn't want any leftover pieces.
Prove that the greatest length of each piece she can cut is 7 inches. -/
theorem greatest_length_of_pieces (a b c : ℕ) (h1 : a = 28) (h2 : b = 42) (h3 : c = 70) :
  ∃ (d : ℕ), d > 0 ∧ d ∣ a ∧ d ∣ b ∧ d ∣ c ∧ ∀ e : ℕ, e > 0 ∧ e ∣ a ∧ e ∣ b ∧ e ∣ c → e ≤ d := sorry

end greatest_length_of_pieces_l261_261095


namespace large_number_divisible_by_12_l261_261863

theorem large_number_divisible_by_12 : (Nat.digits 10 ((71 * 10^168) + (72 * 10^166) + (73 * 10^164) + (74 * 10^162) + (75 * 10^160) + (76 * 10^158) + (77 * 10^156) + (78 * 10^154) + (79 * 10^152) + (80 * 10^150) + (81 * 10^148) + (82 * 10^146) + (83 * 10^144) + 84)).foldl (λ x y => x * 10 + y) 0 % 12 = 0 := 
sorry

end large_number_divisible_by_12_l261_261863


namespace real_solutions_to_polynomial_eq_l261_261127

theorem real_solutions_to_polynomial_eq (m : ℝ) : 
  (∀ x : ℝ, x(x + 1)(x + 2)(x + 3) = m - 1 → (m < 0 ∨
  (m = 0 ∧ (x = - ((3 + sqrt(5)) / 2) ∨ x = - ((3 - sqrt(5)) / 2)) ∨
  (0 < m ∧ m < 25 / 16 ∧ (x = -3 / 2 + sqrt(m + 5 / 4) ∨ x = -3 / 2 - sqrt(m + 5 / 4))) ∨
  (m = 25 / 16 ∧ x = -3 / 2) ∨
  (m > 25 / 16 ∨ false))) :=
sorry

end real_solutions_to_polynomial_eq_l261_261127


namespace total_items_left_in_store_l261_261873

noncomputable def items_ordered : ℕ := 4458
noncomputable def items_sold : ℕ := 1561
noncomputable def items_in_storeroom : ℕ := 575

theorem total_items_left_in_store : 
  (items_ordered - items_sold) + items_in_storeroom = 3472 := 
by 
  sorry

end total_items_left_in_store_l261_261873


namespace equal_areas_of_triangles_l261_261741

open Real

theorem equal_areas_of_triangles 
  (O A B C M P D: Point) 
  (radius : ℝ)
  (h1 : is_center O (semicircle A B))
  (h2 : on_circumference C (semicircle A B))
  (h3 : perp_to AB C M)
  (h4 : perp_to (tangent_at B) C P)
  (h5 : tangent_intersect C (line_at_B P) D)
  :
  (area (triangle C P D) = area (triangle C A M)) ↔
  (dist A M / dist O B = 1 / 3) := 
sorry

end equal_areas_of_triangles_l261_261741


namespace distance_between_M1_M2_l261_261571

-- Definitions and conditions
def point (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)

def dist_3d (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2 + (p2.3 - p1.3) ^ 2)

noncomputable def M1 : ℝ × ℝ × ℝ := point (-1) 0 2
noncomputable def M2 : ℝ × ℝ × ℝ := point 0 3 1

-- Proof problem
theorem distance_between_M1_M2 :
  dist_3d M1 M2 = Real.sqrt 11 :=
by
  sorry

end distance_between_M1_M2_l261_261571


namespace alyssa_puppies_l261_261440

theorem alyssa_puppies (initial now given : ℕ) (h1 : initial = 12) (h2 : now = 5) : given = 7 :=
by
  have h3 : given = initial - now := by sorry
  rw [h1, h2] at h3
  exact h3

end alyssa_puppies_l261_261440


namespace triangle_area_OPQ_l261_261178

theorem triangle_area_OPQ
(C: { p: ℝ × ℝ | ∃ x, p.2 ^ 2 = 4 * x } ) 
(F: ℝ × ℝ) 
(l: ℝ × ℝ → Prop)
(P Q: ℝ × ℝ)
(h1: ∀x y, l (x, y) → (x, y) ∈ C)
(h2: ∀x y, l (x, y) → ∃ m, x = m * y + 1)
(h3: l F)
(h4: (P = (x1, y1) ∧ Q = (x2, y2)))
(h5: (x1, y1) ∈ C ∧ (x2, y2) ∈ C)
(h6: (x1 - 1, y1) + 2 * (x2 - 1, y2) = (0, 0)):
abs ((y1 - y2) / 2) * ∥ (F.1, F.2) ∥ = 3 * sqrt 2 / 2 :=
by sorry

end triangle_area_OPQ_l261_261178


namespace sum_of_segments_equal_295_l261_261009

noncomputable def triangle : Type :=
{ AB BC AC : ℝ // ABC's sides satisfy AB = 15 ∧ BC = 20 ∧ AC = 25 }

noncomputable def divide_segment (n : ℕ) (l : ℝ) : list ℝ := 
list.range (n + 1) |>.map (λ k, (k * l) / n.to_real)

def P_k (k : ℕ) : ℝ := (k * 15) / 60
def Q_k (k : ℕ) : ℝ := (k * 20) / 60
def R_k (k : ℕ) : ℝ := sorry  -- Similar definition for line AC

def length_PkQk (k : ℕ) : ℝ := 
let h := (20 * (60 - k)) / 60 in let b := (15 * k) / 60 
in real.sqrt (h^2 + b^2)

theorem sum_of_segments_equal_295 : 
    let P_Q_Sum := ∑ k in finset.range 59, length_PkQk k 
    /\ let Alt_Seg_Sum := ∑ k in finset.range 60, (sorry : length definition for R_k)
    -> (2 * P_Q_Sum + Alt_Seg_Sum) = 295 :=
begin
  sorry
end

end sum_of_segments_equal_295_l261_261009


namespace original_proposition_contrapositive_proposition_converse_of_proposition_number_of_true_propositions_l261_261191

variable (a b : ℝ)

theorem original_proposition
  (h : a + b = 1) : ab ≤ 1 / 4 := sorry

theorem contrapositive_proposition
  (h1 : ab > 1 / 4) : a + b ≠ 1 := sorry

theorem converse_of_proposition : 
  (∃ a b : ℝ, ab ≤ 1 / 4 ∧ a + b ≠ 1) := 
begin
  use [1/3, 1/3],
  split,
  {
    have h : (1/3) * (1/3) = 1/9, by norm_num,
    rw h,
    exact by norm_num,
  },
  {
    norm_num,
  }
end

theorem number_of_true_propositions : 
  (number_true : ℕ) :=
    have original_is_true : ∃ a b : ℝ, a + b = 1 ∧ ab ≤ 1 / 4 :=
      begin
        use [1 / 2, 1 / 2],
        split,
        {
          norm_num
        },
        have : (1 / 2) * (1 / 2) = 1 / 4, by norm_num,
        rw this,
        exact le_refl (1 / 4)
      end,
    have contrapositive_is_true : ∃ a b : ℝ, (ab ≤ 1 / 4 ∧ a + b ≠ 1) := 
      by apply converse_of_proposition,
    have converse_is_false : not (∀ a b : ℝ, ab ≤ 1 / 4 → a + b = 1) :=
      λ h, show false, 
        from h 1/3 1/3 
          (by norm_num) 
          (by norm_num),
    have negation_is_false : not (∀ a b : ℝ, not (a + b = 1) → ab > 1 / 4) :=
      λ h, show false, 
        from h 1/2 1/2 
          (λ hn h1 h2, hn h1),
    show nat := 1

end original_proposition_contrapositive_proposition_converse_of_proposition_number_of_true_propositions_l261_261191


namespace range_of_a_l261_261979

theorem range_of_a 
  (h : ¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + 1/2 ≤ 0) : a ∈ Ioo (-1 : ℝ) (3 : ℝ) :=
sorry

end range_of_a_l261_261979


namespace find_k_l261_261240

-- Conditions
def t : ℕ := 6
def is_nonzero_digit (n : ℕ) : Prop := n > 0 ∧ n < 10

-- Given these conditions, we need to prove that k = 9
theorem find_k (k t : ℕ) (h1 : t = 6) (h2 : is_nonzero_digit k) (h3 : is_nonzero_digit t) :
    (8 * 10^2 + k * 10 + 8) + (k * 10^2 + 8 * 10 + 8) - 16 * t * 10^0 * 6 = (9 * 10 + 8) + (9 * 10^2 + 8 * 10 + 8) - (16 * 6 * 10^1 + 6) → k = 9 := 
sorry

end find_k_l261_261240


namespace directrix_of_parabola_l261_261497

def parabola_eq (x : ℝ) : ℝ := -4 * x^2 + 4

theorem directrix_of_parabola : 
  ∃ y : ℝ, y = 65 / 16 :=
by
  sorry

end directrix_of_parabola_l261_261497


namespace min_students_in_class_l261_261605

noncomputable def min_possible_students (b g : ℕ) : Prop :=
  (3 * b) / 4 = 2 * (2 * g) / 3 ∧ b = (16 * g) / 9

theorem min_students_in_class : ∃ (b g : ℕ), min_possible_students b g ∧ b + g = 25 :=
by
  sorry

end min_students_in_class_l261_261605


namespace rooks_on_checkerboard_l261_261319

theorem rooks_on_checkerboard (n : ℕ) (board_size : ℕ) (even_rooks : ℕ) (odd_rooks : ℕ)
  (checkerboard : matrix (fin board_size) (fin board_size) bool)
  (coloring : (i j : fin board_size) → bool) :
  board_size = 9 → n = 9 → even_rooks = 4 → odd_rooks = 5 →
  coloring = λ i j, (i.val + j.val) % 2 = 0 → 
  (finset.univ.filter (λ x : fin board_size × fin board_size, coloring x.1 x.2)).card = even_rooks^2 + odd_rooks^2 →
  ∑ (perm : equiv.perm (fin even_rooks)), 1 * ∑ (qerm : equiv.perm (fin odd_rooks)), 1 = 2880 :=
by 
  intros _ _ _ _ _ _ _ _ _ _;
  sorry

end rooks_on_checkerboard_l261_261319


namespace apples_given_by_nathan_l261_261832

theorem apples_given_by_nathan (initial_apples : ℕ) (total_apples : ℕ) (given_by_nathan : ℕ) :
  initial_apples = 6 → total_apples = 12 → given_by_nathan = (total_apples - initial_apples) → given_by_nathan = 6 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end apples_given_by_nathan_l261_261832


namespace negation_of_statement_6_l261_261555

variable (Teenager Adult : Type)
variable (CanCookWell : Teenager → Prop)
variable (CanCookWell' : Adult → Prop)

-- Conditions from the problem
def all_teenagers_can_cook_well : Prop :=
  ∀ t : Teenager, CanCookWell t

def some_teenagers_can_cook_well : Prop :=
  ∃ t : Teenager, CanCookWell t

def no_adults_can_cook_well : Prop :=
  ∀ a : Adult, ¬CanCookWell' a

def all_adults_cannot_cook_well : Prop :=
  ∀ a : Adult, ¬CanCookWell' a

def at_least_one_adult_cannot_cook_well : Prop :=
  ∃ a : Adult, ¬CanCookWell' a

def all_adults_can_cook_well : Prop :=
  ∀ a : Adult, CanCookWell' a

-- Theorem to prove
theorem negation_of_statement_6 :
  at_least_one_adult_cannot_cook_well Adult CanCookWell' = ¬ all_adults_can_cook_well Adult CanCookWell' :=
sorry

end negation_of_statement_6_l261_261555


namespace problem_l261_261186

noncomputable def f (x : ℝ) : ℝ := x^(-3) + Real.sin x + 1

theorem problem (a : ℝ) (h : f a = 3) : f (-a) = -1 :=
begin
  sorry
end

end problem_l261_261186


namespace isosceles_triangle_among_six_points_l261_261441

theorem isosceles_triangle_among_six_points
  (A1 A2 A3 A4 A5 O : Point)
  (hpentagon : ∀ (i j : ℕ), i ≠ j → dist (nth_point [A1, A2, A3, A4, A5] i) (nth_point [A1, A2, A3, A4, A5] j) = dist (nth_point [A1, A2, A3, A4, A5] ((i + 1) % 5)) (nth_point [A1, A2, A3, A4, A5] ((j + 1) % 5)))
  (hcenter : ∀ (i : ℕ), dist O (nth_point [A1, A2, A3, A4, A5] i) = dist O (nth_point [A1, A2, A3, A4, A5] ((i + 1) % 5))) :
  ∀ (P Q R : Point), P ≠ Q → Q ≠ R → P ≠ R → P ∈ {A1, A2, A3, A4, A5, O} → Q ∈ {A1, A2, A3, A4, A5, O} → R ∈ {A1, A2, A3, A4, A5, O} → 
  is_isosceles_triangle P Q R :=
by
  sorry

end isosceles_triangle_among_six_points_l261_261441


namespace probability_of_event_l261_261266

open Real

noncomputable def probability_satisfying_conditions 
  (x : ℝ)
  (hx : 100 ≤ x ∧ x < 200) 
  (h1 : floor (sqrt x) = 12) : ℝ :=
  if floor (sqrt (100 * x)) = 120 then
    (146.41 - 144) / (169 - 144)
  else 
    0

theorem probability_of_event
  (prob : ℝ) 
  (hprob : prob = 241 / 2500) :
  ∀ x : ℝ, 
    (100 ≤ x ∧ x < 200) ∧ (floor (sqrt x) = 12) → 
    probability_satisfying_conditions x ‹_› = prob :=
by
  sorry

end probability_of_event_l261_261266


namespace sin_arcsin_plus_arctan_l261_261484

theorem sin_arcsin_plus_arctan (a b : ℝ) (ha : a = Real.arcsin (4/5)) (hb : b = Real.arctan (1/2)) :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2)) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end sin_arcsin_plus_arctan_l261_261484


namespace directrix_parabola_l261_261496

-- Given the equation of the parabola and required transformations:
theorem directrix_parabola (d : ℚ) : 
  (∀ x : ℚ, y = -4 * x^2 + 4) → d = 65 / 16 :=
by sorry

end directrix_parabola_l261_261496


namespace min_value_of_x_squared_plus_8x_l261_261378

theorem min_value_of_x_squared_plus_8x : ∀ x : ℝ, (x^2 + 8 * x) ≥ -16 ∧ ∃ x : ℝ, (x = -4 ∧ x^2 + 8 * x = -16) := 
by
  intro x
  split
  · apply sorry -- show that (x^2 + 8 * x) ≥ -16
  · use -4
    split
    · refl
    · exact sorry -- show that -4^2 + 8 * -4 = -16

end min_value_of_x_squared_plus_8x_l261_261378


namespace soap_per_pound_correct_l261_261289

-- Definition of the amount of soap used and weight of clothes.
def soap_used : ℝ := 18 -- ounces of soap
def weight_of_clothes : ℝ := 9 -- pounds of clothes

-- Definition of the amount of soap per pound of clothes
def soap_per_pound (soap : ℝ) (weight : ℝ) : ℝ := soap / weight

-- Theorem statement: proving that Mrs. Hilt uses 2 ounces of soap to wash a pound of clothes.
theorem soap_per_pound_correct : soap_per_pound soap_used weight_of_clothes = 2 :=
by
  -- Placeholder for the proof
  sorry

end soap_per_pound_correct_l261_261289


namespace probability_p_3_3_mn_value_l261_261416

noncomputable def P : ℕ × ℕ → ℚ
| (0, 0) := 1
| (x, 0) := 0
| (0, y) := 0
| (x, y) := (1/3) * (P (x-1, y) + P (x, y-1) + P (x-1, y-1))

theorem probability_p_3_3 : P (3, 3) = 7 / 81 := sorry

theorem mn_value : (7 : ℚ).numerator + ((7 / 81 : ℚ).denom : ℚ) = 11 := sorry

end probability_p_3_3_mn_value_l261_261416


namespace complex_power_evaluation_l261_261470

theorem complex_power_evaluation (i : ℂ) (h : i^4 = 1) : i^8 + i^20 + i^{-14} = 1 := by
  sorry

end complex_power_evaluation_l261_261470


namespace function_property_sum_l261_261530

noncomputable def f (x : ℕ) : ℝ :=
-- Assuming a function f that satisfies the given conditions. This will be properly defined in the proof.
sorry

theorem function_property_sum :
  (f 1 ^ 2 + f 2) / f 1 + (f 2 ^ 2 + f 4) / f 3 + (f 3 ^ 2 + f 6) / f 5 +
  (f 4 ^ 2 + f 8) / f 7 + (f 5 ^ 2 + f 10) / f 9 = 30 :=
begin
  -- Assuming f satisfies the functional equation
  assume f_property : ∀ p q, f (p + q) = f p * f q,
  -- Assuming the value of f(1)
  assume f_1 : f 1 = 3,
  -- Proof goes here
  sorry
end

end function_property_sum_l261_261530


namespace fewest_keystrokes_to_400_main_theorem_l261_261795

theorem fewest_keystrokes_to_400 : ∀ (start : ℕ), start = 1 → (∃ (k : ℕ), keystrokes_400 k ∧ k = 10) :=
by sorry

-- Definitions for the problem context
def keystrokes_400 : ℕ → Prop
| 0     := false
| (n+1) := (n ≥ 0) ∧ ((200 = 400 / 2) →
                      (100 = 400 / 2 / 2) →
                      (50 = 400 / 2 / 2 / 2) →
                      (25 = 400 / 2 / 2 / 2 - 1) →
                      (24 = 400 / 2 / 2 / 2 - 1 - 1) →
                      (12 = 400 / 2 / 2 / 2 - 1 - 1 / 2) →
                      (6 = 400 / 2 / 2 / 2 - 1 - 1 / 2 / 2) →
                      (3 = 400 / 2 / 2 / 2 - 1 - 1 / 2 / 2 / 2) →
                      (2 = 400 / 2 / 2 / 2 - 1 - 1 / 2 / 2 / 2 1 - 1) →
                      (1 = 400 / 2 / 2 / 2 - 1 - 1 / 2 / 2 / 2 1 - 1 / 2))

noncomputable theory
def reach_400 : ℕ → ℕ → ℕ
| 0     _ := 0
| (n+1) x := if x = 400 then (n+1) else if x % 2 = 0 then reach_400 n (x / 2) else reach_400 n (x - 1)

-- Main theorem derived from the problem statement
theorem main_theorem : reach_400 10 1 = 400 :=
by sorry

end fewest_keystrokes_to_400_main_theorem_l261_261795


namespace polar_to_rectangular_coords_l261_261892

def polar_to_rectangular (r θ : ℝ) : ℝ × ℝ := (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_rectangular_coords :
  polar_to_rectangular 3 (3 * Real.pi / 4) = (-3 * Real.sqrt 2 / 2, 3 * Real.sqrt 2 / 2) :=
by
  sorry

end polar_to_rectangular_coords_l261_261892


namespace smallest_solution_of_quartic_equation_l261_261751

theorem smallest_solution_of_quartic_equation :
  ∃ x : ℝ, (x^4 - 50 * x^2 + 625 = 0) ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y := sorry

end smallest_solution_of_quartic_equation_l261_261751


namespace choose_courses_l261_261820

theorem choose_courses (A B : ℕ) (total_courses chosen_courses : ℕ) :
    A = 3 → B = 4 → total_courses = 3 →
    (∃ k1 k2 : ℕ, k1 + k2 = chosen_courses ∧ 1 ≤ k1 ∧ 1 ≤ k2 ∧ chosen_courses = total_courses) →
    (nat.choose 3 2 * nat.choose 4 1 + nat.choose 3 1 * nat.choose 4 2 = 30) :=
by sorry

end choose_courses_l261_261820


namespace reflection_add_m_b_l261_261717

-- Definitions for the positions of points A and B
def Point := ℝ × ℝ
def A : Point := (2, 3)
def B : Point := (10, 7)

-- Definition for the reflection line equation parameters
variables (m b : ℝ)

-- Lean theorem statement
theorem reflection_add_m_b : 
  (∃ (m b : ℝ), ∀ (p : Point), 
    let A' := (2 * 6 - p.1, 2 * 5 - p.2) in  -- A' is the reflection of A
    A' = B ∧ m + b = 15) :=
sorry

end reflection_add_m_b_l261_261717


namespace min_unit_cubes_l261_261603

theorem min_unit_cubes (l w h : ℕ) (S : ℕ) (hS : S = 52) 
  (hSurface : 2 * (l * w + l * h + w * h) = S) : 
  ∃ l w h, l * w * h = 16 :=
by
  -- start the proof here
  sorry

end min_unit_cubes_l261_261603


namespace solution_set_transformation_l261_261962

noncomputable def solution_set_of_first_inequality (a b : ℝ) : Set ℝ :=
  {x | a * x^2 - 5 * x + b > 0}

noncomputable def solution_set_of_second_inequality (a b : ℝ) : Set ℝ :=
  {x | b * x^2 - 5 * x + a > 0}

theorem solution_set_transformation (a b : ℝ)
  (h : solution_set_of_first_inequality a b = {x | -3 < x ∧ x < 2}) :
  solution_set_of_second_inequality a b = {x | x < -3 ∨ x > 2} :=
by
  sorry

end solution_set_transformation_l261_261962


namespace least_sum_of_exponents_of_distinct_powers_of_two_l261_261208

theorem least_sum_of_exponents_of_distinct_powers_of_two (n : ℕ) (k : ℕ) (S : Finset ℕ) 
  (h_sum : S.sum (λ i, 2^i) = 1983)
  (h_card : S.card ≥ 5) :
  S.sum id = 55 :=
sorry

end least_sum_of_exponents_of_distinct_powers_of_two_l261_261208


namespace ratio_speed_l261_261745

def diameter_cyclist_A : ℝ := 1000 -- meters
def laps_cyclist_A : ℕ := 3
def time_cyclist_A : ℝ := 10 -- minutes

def length_cyclist_B : ℝ := 5000 -- meters
def round_trips_cyclist_B : ℕ := 2
def time_cyclist_B : ℝ := 5 -- minutes

theorem ratio_speed : 
  let distance_A := laps_cyclist_A * π * diameter_cyclist_A,
      speed_A := distance_A / time_cyclist_A,
      distance_B := round_trips_cyclist_B * 2 * length_cyclist_B,
      speed_B := distance_B / time_cyclist_B in
  (speed_A / speed_B) = 3 * π / 40 :=
by
  sorry

end ratio_speed_l261_261745


namespace number_of_odd_card_subsets_l261_261655

def X : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}

theorem number_of_odd_card_subsets (X : Finset ℕ) (h : X = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}) :
  (Finset.powerset X).filter (λ Y, Y.card % 2 = 1).card = 65536 := by
  sorry

end number_of_odd_card_subsets_l261_261655


namespace root_interval_exists_l261_261277

def f (x : ℝ) : ℝ := 2 * x + Real.log x - 6

theorem root_interval_exists :
  ∃ m : ℝ, f m = 0 ∧ 2 < m ∧ m < 3 :=
sorry

end root_interval_exists_l261_261277


namespace arithmetic_sequence_relation_l261_261930

variables {a b : ℕ → ℕ} {S T : ℕ → ℕ}

-- Given conditions
def condition1 (n : ℕ) : Prop := ∀ m : ℕ, S m = (m * (2 * n + 1)a m / 2) ∧ T m = (m * (4 * n - 2)b m / 2)

-- Given relation between S_n and T_n
def condition2 (n : ℕ) : Prop := T n ≠ 0 → S n * (4 * n - 2) = T n * (2 * n + 1)

-- The statement we need to prove
theorem arithmetic_sequence_relation (h₁ : condition1 10) (h₂ : condition2 20) :
  (a 10) / (b 3 + b 18) + (a 11) / (b 6 + b 15) = 41 / 78 := by
  sorry

end arithmetic_sequence_relation_l261_261930


namespace possible_values_for_t_l261_261951

noncomputable def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def b (a1 d : ℝ) (n : ℕ) : ℝ :=
  Real.sin (arithmetic_sequence a1 d n)

theorem possible_values_for_t :
  ∃ t : ℕ, t ≤ 8 ∧ (∀ n : ℕ, b 0 (real.pi/2) n = b 0 (real.pi/2) (n + t)) ∧ (∃ S : set ℝ, S = {b 0 (real.pi/2) n | n : ℕ} ∧ S.card = 4) →
  t = 4 :=
by
  sorry

end possible_values_for_t_l261_261951


namespace requiredSheetsOfPaper_l261_261439

-- Define the conditions
def englishAlphabetLetters : ℕ := 26
def timesWrittenPerLetter : ℕ := 3
def sheetsOfPaperPerLetter (letters : ℕ) (times : ℕ) : ℕ := letters * times

-- State the theorem equivalent to the original math problem
theorem requiredSheetsOfPaper : sheetsOfPaperPerLetter englishAlphabetLetters timesWrittenPerLetter = 78 := by
  sorry

end requiredSheetsOfPaper_l261_261439


namespace number_of_tangent_lines_with_slope_3_l261_261557

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Define the derivative of f
def f_prime (x : ℝ) : ℝ := 3 * x^2

-- Define the condition for the slope of the tangent line being equal to 3
def is_slope_3 (x : ℝ) : Prop := f_prime x = 3

-- State the theorem: the number of solutions to this condition is 2
theorem number_of_tangent_lines_with_slope_3 : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ is_slope_3 x1 ∧ is_slope_3 x2) :=
by
  have h1 : is_slope_3 1 := by
    unfold is_slope_3 f_prime
    norm_num
  have h2 : is_slope_3 (-1) := by
    unfold is_slope_3 f_prime
    norm_num
  use [1, -1]
  split
  · exact ne_of_gt (zero_lt_one)
  split
  · exact h1
  · exact h2

end number_of_tangent_lines_with_slope_3_l261_261557


namespace lcm_revolutions_5040_l261_261589

noncomputable def lcm_tire_revolutions (distance_in_miles : ℝ) (d_f : ℝ) (d_r : ℝ) : ℕ :=
  let D := distance_in_miles * 5280 * 12  -- converting miles to inches
  let C_f := Real.pi * d_f  -- circumference of front tires
  let C_r := Real.pi * d_r  -- circumference of rear tires
  let revolutions_f := D / C_f  -- revolutions of front tires
  let revolutions_r := D / C_r  -- revolutions of rear tires
  Nat.lcm (revolutions_f.to_nat) (revolutions_r.to_nat)

theorem lcm_revolutions_5040 :
  lcm_tire_revolutions (1 / 2) 10 12 = 5040 := by sorry

end lcm_revolutions_5040_l261_261589


namespace final_number_appended_is_84_l261_261854

noncomputable def arina_sequence := "7172737475767778798081"

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def is_divisible_by_12 (n : ℕ) : Prop := n % 12 = 0

-- Define adding numbers to the sequence
def append_number (seq : String) (n : ℕ) : String := seq ++ n.repr

-- Create the full sequence up to 84 and check if it's divisible by 12
def generate_full_sequence : String :=
  let base_seq := arina_sequence
  let full_seq := append_number (append_number (append_number (append_number (append_number (append_number (append_number (append_number (append_number arina_sequence 82) 83) 84))) 85) 86) 87) 88 
  full_seq

theorem final_number_appended_is_84 : (∃ seq : String, is_divisible_by_12(seq.to_nat) ∧ seq.ends_with "84") := 
by
  sorry

end final_number_appended_is_84_l261_261854


namespace sara_red_flowers_l261_261305

theorem sara_red_flowers (yellow_flowers bouquets : ℕ) (h1 : yellow_flowers = 24) (h2 : bouquets = 8) (h3 : yellow_flowers % bouquets = 0) : 
  (yellow_flowers / bouquets) * bouquets = 24 :=
by
  have yellow_per_bouquet := yellow_flowers / bouquets
  have red_per_bouquet := yellow_per_bouquet
  have total_red_flowers := red_per_bouquet * bouquets
  show total_red_flowers = 24, by sorry

end sara_red_flowers_l261_261305


namespace balance_force_l261_261013

structure Vector2D where
  x : ℝ
  y : ℝ

def F1 : Vector2D := ⟨1, 1⟩
def F2 : Vector2D := ⟨2, 3⟩

def vector_add (a b : Vector2D) : Vector2D := ⟨a.x + b.x, a.y + b.y⟩
def vector_neg (a : Vector2D) : Vector2D := ⟨-a.x, -a.y⟩

theorem balance_force : 
  ∃ F3 : Vector2D, vector_add (vector_add F1 F2) F3 = ⟨0, 0⟩ ∧ F3 = ⟨-3, -4⟩ := 
by
  sorry

end balance_force_l261_261013


namespace last_appended_number_is_84_l261_261841

theorem last_appended_number_is_84 : 
  ∃ N : ℕ, 
    let s := "7172737475767778798081" ++ (String.intercalate "" (List.map toString [82, 83, 84])) in
    (N = 84) ∧ (s.toNat % 12 = 0) :=
by
  sorry

end last_appended_number_is_84_l261_261841


namespace sum_of_coordinates_of_X_l261_261647

open EuclideanGeometry

noncomputable def X_coord_sum (X Y Z : Point) := (X.1 + X.2)

theorem sum_of_coordinates_of_X 
  (X Y Z : Point) 
  (h1 : Z = Point (1, -3))
  (h2 : Y = Point (3, 5))
  (h3 : ∃ x1 y1, X = Point (x1, y1) ∧ Z = Point ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2)) : 
  X_coord_sum X Y Z = -12 :=
sorry

end sum_of_coordinates_of_X_l261_261647


namespace set_complement_union_l261_261541

namespace ProblemOne

def A : Set ℝ := {x | x ≤ -3 ∨ x ≥ 2}
def B : Set ℝ := {x | 1 < x ∧ x < 5}

theorem set_complement_union :
  (Aᶜ ∪ B) = {x : ℝ | -3 < x ∧ x < 5} := sorry

end ProblemOne

end set_complement_union_l261_261541


namespace fraction_equiv_reduced_l261_261370

theorem fraction_equiv_reduced :
  let x : ℚ := (integer_repr "0.1353535...") in
  let x1 : ℚ := 1000 * x = 135 + (35 / 1000) in
  let x2 : ℚ := 10 * x = 1 + (35 / 100) in
  let x3 : ℚ := 990 * x = 134 in
  x = 67 / 495 :=
by
  let x := (integer_repr "0.1353535...")
  have x1000 : 1000 * x = 135 + (35 / 1000) := sorry
  have x10 : 10 * x = 1 + (35 / 100) := sorry
  have x990 : 990 * x = 134 := sorry
  have x_frac_eq : x = (134 / 990) := sorry
  have reduced_frac : 134 / 990 = 67 / 495 := sorry
  have eq_frac : x = 67 / 495 := sorry
  assumption

end fraction_equiv_reduced_l261_261370


namespace probability_different_color_and_label_sum_more_than_3_l261_261615

-- Definitions for the conditions:
structure Coin :=
  (color : Bool) -- True for Yellow, False for Green
  (label : Nat)

def coins : List Coin := [
  Coin.mk true 1,
  Coin.mk true 2,
  Coin.mk false 1,
  Coin.mk false 2,
  Coin.mk false 3
]

def outcomes : List (Coin × Coin) :=
  [(coins[0], coins[1]), (coins[0], coins[2]), (coins[0], coins[3]), (coins[0], coins[4]),
   (coins[1], coins[2]), (coins[1], coins[3]), (coins[1], coins[4]),
   (coins[2], coins[3]), (coins[2], coins[4]), (coins[3], coins[4])]

def different_color_and_label_sum_more_than_3 (c1 c2 : Coin) : Bool :=
  c1.color ≠ c2.color ∧ (c1.label + c2.label > 3)

def valid_outcomes : List (Coin × Coin) :=
  outcomes.filter (λ p => different_color_and_label_sum_more_than_3 p.fst p.snd)

-- Proof statement:
theorem probability_different_color_and_label_sum_more_than_3 :
  (valid_outcomes.length : ℚ) / (outcomes.length : ℚ) = 3 / 10 :=
by
  sorry

end probability_different_color_and_label_sum_more_than_3_l261_261615


namespace sequence_general_formula_sequence_sum_b_l261_261163

theorem sequence_general_formula {a : ℕ → ℕ} (h₁ : a 1 = 1)
    (h₂ : ∀ n, a (n + 1) ^ 2 - a n ^ 2 = 8 * n) :
    ∀ n, a n = 2 * n - 1 := by
  sorry

theorem sequence_sum_b {a b : ℕ → ℤ} (h₁ : a 1 = 1)
    (h₂ : ∀ n, a (n + 1) ^ 2 - a n ^ 2 = 8 * n)
    (h₃ : ∀ n, a n = 2 * n - 1)
    (h₄ : ∀ n, b n = a n * Real.sin (a n / 2 * Real.pi)) :
    (∑ k in Finset.range 2023, b (k + 1)) = 2023 := by
  sorry

end sequence_general_formula_sequence_sum_b_l261_261163


namespace radius_of_third_circle_l261_261012

theorem radius_of_third_circle {P Q R : Point} (r : ℝ) (hPQ : dist P Q = 8) (hPR : dist P R = 3 + r) (hQR : dist Q R = 5 - r) : r = 5 :=
  sorry

end radius_of_third_circle_l261_261012


namespace total_symmetric_scanning_codes_l261_261429

-- Definition of a grid and its properties for the proof
structure Grid (n : ℕ) :=
  (colors : Fin n → Fin n → Bool) -- Boolean functions representing color of the grid

def is_symmetric (grid : Grid 7) : Prop :=
  ∀ r c, grid.colors r c = grid.colors (6 - c) r ∧
             grid.colors r c = grid.colors (6 - r) (6 - c) ∧
             grid.colors r c = grid.colors c (6 - r) ∧
             grid.colors r c = grid.colors (6 - r) c ∧
             grid.colors r c = grid.colors (6 - c) r ∧
             grid.colors r c = grid.colors (c) r ∧
             grid.colors r c = grid.colors (6 - c) (6 - r)

def has_both_colors (grid : Grid 7) : Prop :=
  ∃ r1 c1, grid.colors r1 c1 ≠ grid.colors 0 0 ∧
  ∃ r2 c2, grid.colors r2 c2 = grid.colors 0 0

-- Main proof statement
theorem total_symmetric_scanning_codes : ∃ n, n = 1022 ∧
  ∀ grid : Grid 7, is_symmetric grid ∧ has_both_colors grid → grid.colors = n :=
begin
  sorry
end

end total_symmetric_scanning_codes_l261_261429


namespace perception_arrangements_l261_261902

theorem perception_arrangements : 
  let n := 10
  let p := 2
  let e := 2
  let i := 2
  let r := 1
  let c := 1
  let t := 1
  let o := 1
  let n' := 1 
  (n.factorial / (p.factorial * e.factorial * i.factorial * r.factorial * c.factorial * t.factorial * o.factorial * n'.factorial)) = 453600 := 
by
  sorry

end perception_arrangements_l261_261902


namespace remaining_amount_after_shopping_l261_261021

theorem remaining_amount_after_shopping (initial_amount spent_percentage remaining_amount : ℝ)
  (h_initial : initial_amount = 4000)
  (h_spent : spent_percentage = 0.30)
  (h_remaining : remaining_amount = 2800) :
  initial_amount - (spent_percentage * initial_amount) = remaining_amount :=
by
  sorry

end remaining_amount_after_shopping_l261_261021


namespace sixty_percent_of_fifty_minus_thirty_percent_of_thirty_l261_261200

theorem sixty_percent_of_fifty_minus_thirty_percent_of_thirty : 
  (60 / 100 : ℝ) * 50 - (30 / 100 : ℝ) * 30 = 21 :=
by
  sorry

end sixty_percent_of_fifty_minus_thirty_percent_of_thirty_l261_261200


namespace club_profit_is_325_l261_261408

-- Definitions for the conditions provided in the problem

def cost_per_bottle : ℝ := 3 / 6
def total_bottles : ℕ := 1500
def discount_threshold : ℕ := 1200
def discount_rate : ℝ := 0.9
def selling_price_per_bottle : ℝ := 2 / 3

-- Calculate the total cost accounting for the discount
def total_cost : ℝ :=
  if total_bottles > discount_threshold then total_bottles * cost_per_bottle * discount_rate
  else total_bottles * cost_per_bottle

-- Calculate the total revenue from selling the bottles
def total_revenue : ℝ := total_bottles * selling_price_per_bottle

-- Calculate the profit
def profit : ℝ := total_revenue - total_cost

-- Statement to prove
theorem club_profit_is_325 : profit = 325 := by
  sorry

end club_profit_is_325_l261_261408


namespace javier_total_time_l261_261636

-- Define the given constants and conditions
def outlining_time : ℝ := 30
def first_break_time : ℝ := 10
def additional_writing_time : ℝ := 28
def rewriting_time : ℝ := 15
def second_break_time : ℝ := 5
def total_writing_time : ℝ := outlining_time + additional_writing_time
def practicing_time : ℝ := (total_writing_time + rewriting_time) / 2

-- Formalize the time calculations in Lean
def total_time : ℝ := outlining_time + 
                     first_break_time +
                     total_writing_time +
                     rewriting_time +
                     second_break_time +
                     practicing_time

-- Prove that the total time spent is 154.5 minutes
theorem javier_total_time : total_time = 154.5 := by
  sorry

end javier_total_time_l261_261636


namespace line_equation_rect_ellipse_equation_rect_min_distance_l261_261630

-- conditions
def line_polar_equation (ρ θ : ℝ) : Prop := 
  ρ * cos(θ) + 2 * ρ * sin(θ) + 3 * sqrt 2 = 0

def ellipse_polar_equation (ρ θ : ℝ) : Prop := 
  ρ^2 = 4 / (cos(θ)^2 + 4 * sin(θ)^2)

-- proof problem statements
theorem line_equation_rect (ρ θ x y : ℝ) (h : x = ρ * cos(θ))
  (k : y = ρ * sin(θ)) (h_line : line_polar_equation ρ θ) : 
    x + 2 * y + 3 * sqrt 2 = 0 := sorry

theorem ellipse_equation_rect (ρ θ x y : ℝ) (h : x = ρ * cos(θ))
  (k : y = ρ * sin(θ)) (h_ellipse : ellipse_polar_equation ρ θ) : 
    x^2 / 4 + y^2 = 1 := sorry

theorem min_distance (α : ℝ) : 
  ∃ Q : ℝ × ℝ, 
    (Q.1 = 2 * cos(α) ∧ Q.2 = sin(α)) ∧
    ∀ P : ℝ × ℝ, (P.1 + 2 * P.2 + 3 * sqrt 2 = 0 → 
    min_dist_PQ P Q = sqrt 10 / 5) := sorry

end line_equation_rect_ellipse_equation_rect_min_distance_l261_261630


namespace perception_arrangements_l261_261901

theorem perception_arrangements : 
  let n := 10
  let p := 2
  let e := 2
  let i := 2
  let r := 1
  let c := 1
  let t := 1
  let o := 1
  let n' := 1 
  (n.factorial / (p.factorial * e.factorial * i.factorial * r.factorial * c.factorial * t.factorial * o.factorial * n'.factorial)) = 453600 := 
by
  sorry

end perception_arrangements_l261_261901


namespace magnitude_ON_l261_261171

-- Given a point M in 3D space
def point_M : ℝ × ℝ × ℝ := (3, 3, 4)

-- Define the projection of M onto the xz-plane (which sets y-coordinate to 0)
def projection_N (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (p.1, 0, p.2)

-- Calculate the magnitude of the vector from origin to the projection on xz-plane
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

-- Assuming the point N is the projection of M onto xz-plane
def N : ℝ × ℝ × ℝ := projection_N point_M

-- The magnitude of vector ON should be 5
theorem magnitude_ON : magnitude N = 5 := by
  -- Calculate the required steps
  sorry

end magnitude_ON_l261_261171


namespace max_black_cells_1000_by_1000_l261_261295

def maxBlackCells (m n : ℕ) : ℕ :=
  if m = 1 then n else if n = 1 then m else m + n - 2

theorem max_black_cells_1000_by_1000 : maxBlackCells 1000 1000 = 1998 :=
  by sorry

end max_black_cells_1000_by_1000_l261_261295


namespace closest_point_l261_261501

-- Definitions
def line_eqn (p : ℝ × ℝ) : Prop := p.snd = (p.fst - 3) / 3
def target_point : ℝ × ℝ := (0, 2)
def closest_point_on_line : ℝ × ℝ := (9 / 10, -7 / 10)

-- Theorem
theorem closest_point (p : ℝ × ℝ) (h : line_eqn p) : p = closest_point_on_line :=
sorry

end closest_point_l261_261501


namespace a_n_general_formula_S_n_sum_formula_l261_261278

-- Sequence definition according to given conditions
def a : ℕ → ℕ
| 1 := 3
| (n + 1) := 3 * a n - 4 * n

-- b_n definition as 2^n * a_n
def b (n : ℕ) : ℕ := 2^n * a n

-- Sum of first n terms of the sequence b_n
def S : ℕ → ℕ
| 0 := 0
| (n + 1) := S n + b (n + 1)

theorem a_n_general_formula (n : ℕ) : a n = 2 * n + 1 :=
by sorry

theorem S_n_sum_formula (n : ℕ) : S n = (2 * n - 1) * 2^(n + 1) + 2 :=
by sorry

end a_n_general_formula_S_n_sum_formula_l261_261278


namespace cube_sphere_radius_l261_261041

theorem cube_sphere_radius :
  (s : ℝ) = 6.5 →
  let surface_area_cube := 6 * s^2 in
  let surface_area_sphere := 4 * Real.pi * r^2 in
  surface_area_cube = surface_area_sphere →
  Int.round (Real.sqrt (surface_area_cube / (4 * Real.pi))) = 4 :=
by
  intros
  let s := 6.5
  let surface_area_cube = 6 * s^2
  let surface_area_sphere = 4 * Real.pi * r^2
  assume surface_area_cube = surface_area_sphere
  have h : Int.round (Real.sqrt (surface_area_cube / (4 * Real.pi))) = 4
  apply sorry

end cube_sphere_radius_l261_261041


namespace parabola_directrix_l261_261491

theorem parabola_directrix (x y : ℝ) : 
  (∀ x: ℝ, y = -4 * x ^ 2 + 4) → (y = 65 / 16) := 
by 
  sorry

end parabola_directrix_l261_261491


namespace last_appended_number_is_84_l261_261844

theorem last_appended_number_is_84 : 
  ∃ N : ℕ, 
    let s := "7172737475767778798081" ++ (String.intercalate "" (List.map toString [82, 83, 84])) in
    (N = 84) ∧ (s.toNat % 12 = 0) :=
by
  sorry

end last_appended_number_is_84_l261_261844


namespace diane_current_money_l261_261118

-- Problem condition definitions
def cost_of_cookies : ℕ := 65
def amount_needed : ℕ := 38

-- Problem question as a proof goal
theorem diane_current_money : ℕ := cost_of_cookies - amount_needed

-- The expected answer
example : diane_current_money = 27 := by
  unfold diane_current_money
  simp [cost_of_cookies, amount_needed]
  sorry

end diane_current_money_l261_261118


namespace alice_speed_exceeds_l261_261707

theorem alice_speed_exceeds (distance : ℕ) (v_bob : ℕ) (time_diff : ℕ) (v_alice : ℕ)
  (h_distance : distance = 220)
  (h_v_bob : v_bob = 40)
  (h_time_diff : time_diff = 1/2) : 
  v_alice > 44 := 
sorry

end alice_speed_exceeds_l261_261707


namespace sin_sum_arcsin_arctan_l261_261478

-- Definitions matching the conditions
def a := Real.arcsin (4 / 5)
def b := Real.arctan (1 / 2)

-- Theorem stating the question and expected answer
theorem sin_sum_arcsin_arctan : 
  Real.sin (a + b) = (11 * Real.sqrt 5) / 25 := 
by 
  sorry

end sin_sum_arcsin_arctan_l261_261478


namespace sphere_volume_given_surface_area_l261_261734

-- Given the surface area of the sphere is 400π square centimeters
def surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

-- The radius calculated from the surface area
def radius (s : ℝ) : ℝ := Real.sqrt (s / (4 * Real.pi))

-- Volume of the sphere given the radius
def volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem sphere_volume_given_surface_area :
  surface_area (radius 400 * Real.pi) = 400 * Real.pi →
  volume (radius (400 * Real.pi)) = 4000/3 * Real.pi :=
by
  sorry

end sphere_volume_given_surface_area_l261_261734


namespace arithmetic_example_l261_261056

theorem arithmetic_example : 3889 + 12.808 - 47.80600000000004 = 3854.002 := 
by
  sorry

end arithmetic_example_l261_261056


namespace area_of_inscribed_square_l261_261082

theorem area_of_inscribed_square :
  (∃ t : ℝ, (2 * t)^2 * 2/3 = 1 ) → 
  (∃ s : ℝ, s = 2 * sqrt(2/3)*sqrt(8) / 3 → 
  s^2 = 32/3) := sorry

end area_of_inscribed_square_l261_261082


namespace selling_price_is_correct_l261_261680

noncomputable def purchase_price : ℝ := 36400
noncomputable def repair_costs : ℝ := 8000
noncomputable def profit_percent : ℝ := 54.054054054054056

noncomputable def total_cost := purchase_price + repair_costs
noncomputable def selling_price := total_cost * (1 + profit_percent / 100)

theorem selling_price_is_correct :
    selling_price = 68384 := by
  sorry

end selling_price_is_correct_l261_261680


namespace max_min_on_interval_tangent_line_eq1_tangent_line_eq2_l261_261967

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x
def interval : set ℝ := set.Icc (-2 : ℝ) 1
def P : ℝ × ℝ := (2, -6)

theorem max_min_on_interval : (∀ x ∈ interval, -2 ≤ f x ∧ f x ≤ 2) ∧ (∀ y ∈ (f '' interval), y = -2 ∨ y = 2) :=
sorry

theorem tangent_line_eq1 : ∃ t : ℝ, t = 0 ∨ t = 3 ∧ (P.fst, 2) ≠ P.snd → 3*P.fst + P.snd = 0 :=
sorry

theorem tangent_line_eq2 : ∃ t : ℝ, t = 0 ∨ t = 3 ∧ (P.fst, 2) ≠ P.snd → 24*P.fst - P.snd - 54 = 0 :=
sorry

end max_min_on_interval_tangent_line_eq1_tangent_line_eq2_l261_261967


namespace large_number_divisible_by_12_l261_261866

theorem large_number_divisible_by_12 : (Nat.digits 10 ((71 * 10^168) + (72 * 10^166) + (73 * 10^164) + (74 * 10^162) + (75 * 10^160) + (76 * 10^158) + (77 * 10^156) + (78 * 10^154) + (79 * 10^152) + (80 * 10^150) + (81 * 10^148) + (82 * 10^146) + (83 * 10^144) + 84)).foldl (λ x y => x * 10 + y) 0 % 12 = 0 := 
sorry

end large_number_divisible_by_12_l261_261866


namespace line_length_limit_l261_261414

theorem line_length_limit : 
  ∑' n : ℕ, 1 / ((3 : ℝ) ^ n) + (1 / (3 ^ (n + 1))) * (Real.sqrt 3) = (3 + Real.sqrt 3) / 2 :=
sorry

end line_length_limit_l261_261414


namespace median_of_high_jump_results_l261_261445

-- Condition: the results of 7 male high jump athletes
def high_jump_results := [1.52, 1.58, 1.75, 1.58, 1.81, 1.65, 1.72]

-- The problem to prove: the median of these results is 1.65
theorem median_of_high_jump_results : (high_jump_results.sorted_nth (⌊high_jump_results.length / 2⌋) = 1.65) :=
by
  -- Skip the proof with sorry
  sorry

end median_of_high_jump_results_l261_261445


namespace sum_of_integer_solutions_eq_zero_l261_261925

theorem sum_of_integer_solutions_eq_zero :
  (∑ x in {x : ℤ | x^4 - 49 * x^2 + 576 = 0}, x) = 0 := 
by
  sorry

end sum_of_integer_solutions_eq_zero_l261_261925


namespace find_f_of_2_l261_261525

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f_of_2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 := by
  sorry

end find_f_of_2_l261_261525


namespace anna_cannot_afford_tour_l261_261098

noncomputable def future_value (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def future_cost (C0 : ℝ) (i : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  C0 * (1 + i / n) ^ (n * t)

theorem anna_cannot_afford_tour :
  let P := 40000
  let r := 0.05
  let n := 1
  let t := 3
  let C0 := 45000
  let i := 0.05
  future_value P r n t < future_cost C0 i n t :=
  by
    let P := 40000
    let r := 0.05
    let n := 1
    let t := 3
    let C0 := 45000
    let i := 0.05
    have fv := future_value P r n t
    have fc := future_cost C0 i n t
    show fv < fc from sorry

end anna_cannot_afford_tour_l261_261098


namespace find_a_l261_261766

theorem find_a (a : ℝ) (h : (1 / Real.log 3 / Real.log a) + (1 / Real.log 5 / Real.log a) + (1 / Real.log 7 / Real.log a) = 1) : 
  a = 105 := 
sorry

end find_a_l261_261766


namespace relationship_of_x_y_z_l261_261152

theorem relationship_of_x_y_z
  (a b c x y z : ℝ)
  (h1 : log a b = -1)
  (h2 : 2^a > 3)
  (h3 : c > 1)
  (hx : x = - log b (sqrt a))
  (hy : y = log b c)
  (hz : z = (1 / 3) * a) :
  z > x ∧ x > y :=
by
  sorry

end relationship_of_x_y_z_l261_261152


namespace triangle_max_area_l261_261217

theorem triangle_max_area (a b c : ℝ) (h : 2 * a^2 + b^2 + c^2 = 4) :
  let area := 1 / 2 * b * c * Real.sin (Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) in
  ∃ a b c, 2 * a^2 + b^2 + c^2 = 4 ∧ 
  area = (Real.sqrt 5) / 5 := sorry

end triangle_max_area_l261_261217


namespace find_last_num_divisible_by_12_stopping_at_84_l261_261846

theorem find_last_num_divisible_by_12_stopping_at_84 :
  ∃ N, (N = 84) ∧ (71 ≤ N) ∧ (let concatenated := string.join ((list.range (N - 70)).map (λ i, (string.of_nat (i + 71)))) in 
    (nat.divisible (int.of_nat (string.to_nat concatenated)) 12)) :=
begin
  sorry
end

end find_last_num_divisible_by_12_stopping_at_84_l261_261846


namespace g_g_one_third_l261_261956

def g (x : ℝ) : ℝ := x^(-2) + (x^(-2)) / (1 + x^(-2))

theorem g_g_one_third : g (g (1 / 3)) ≈ 0.0203 :=
begin
  sorry -- Proof omitted
end

end g_g_one_third_l261_261956


namespace solution_set_for_inequality_l261_261953

-- Define f as an even function
def is_even (f : ℝ → ℝ) := ∀ x, f (-x) = f x

-- Define the function f according to the given condition for x ≥ 0
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 2*x else (x^2 - 2*x)

-- State the main theorem to be proved
theorem solution_set_for_inequality :
  is_even f → (∀ x ≥ 0, f x = x^2 - 2*x) → 
  { x : ℝ | f (x+1) < 3 } = set.Ioo (-4 : ℝ) (2 : ℝ) :=
by
  sorry

end solution_set_for_inequality_l261_261953


namespace length_of_ae_l261_261034

def consecutive_points_on_line (a b c d e : ℝ) : Prop :=
  ∃ (ab bc cd de : ℝ), 
  ab = 5 ∧ 
  bc = 2 * cd ∧ 
  de = 4 ∧ 
  a + ab = b ∧ 
  b + bc = c ∧ 
  c + cd = d ∧ 
  d + de = e ∧
  a + ab + bc = c -- ensuring ac = 11

theorem length_of_ae (a b c d e : ℝ) 
  (h1 : consecutive_points_on_line a b c d e) 
  (h2 : a + 5 = b)
  (h3 : b + 2 * (c - b) = c)
  (h4 : d - c = 3)
  (h5 : d + 4 = e)
  (h6 : a + 5 + 2 * (c - b) = c) :
  e - a = 18 :=
sorry

end length_of_ae_l261_261034


namespace ellipse_standard_form_l261_261173

theorem ellipse_standard_form
  (a b : ℝ) 
  (h1 : a > b ∧ b > 0) 
  (focus_dist : 2 * Real.sqrt 6 = 2 * Real.sqrt (a^2 - b^2)) 
  (passes_through : (Real.sqrt 3, Real.sqrt 2) ∈ {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}) :
  (\frac {x^{2}}{a^{2}}+ \frac {y^{2}}{b^{2}}=1)
:= . by sorry

end ellipse_standard_form_l261_261173


namespace tan2α_sin_β_l261_261544

open Real

variables {α β : ℝ}

axiom α_acute : 0 < α ∧ α < π / 2
axiom β_acute : 0 < β ∧ β < π / 2
axiom sin_α : sin α = 4 / 5
axiom cos_alpha_beta : cos (α + β) = 5 / 13

theorem tan2α : tan 2 * α = -24 / 7 :=
by sorry

theorem sin_β : sin β = 16 / 65 :=
by sorry

end tan2α_sin_β_l261_261544


namespace part1_l261_261966

theorem part1 (a x0 : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a ^ x0 = 2) : a ^ (3 * x0) = 8 := by
  sorry

end part1_l261_261966


namespace projection_of_a_onto_b_l261_261569

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (3, 4)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def projection (v w : ℝ × ℝ) : ℝ :=
  dot_product v w / magnitude w

theorem projection_of_a_onto_b : projection a b = -1 :=
by
  sorry

end projection_of_a_onto_b_l261_261569


namespace germs_killed_in_common_l261_261068

theorem germs_killed_in_common :
  ∃ x : ℝ, x = 5 ∧
    ∀ A B C : ℝ, A = 50 → 
    B = 25 → 
    C = 30 → 
    x = A + B - (100 - C) := sorry

end germs_killed_in_common_l261_261068


namespace no_solution_for_inequalities_l261_261978

theorem no_solution_for_inequalities (m : ℝ) :
  (∀ x : ℝ, x - m ≤ 2 * m + 3 ∧ (x - 1) / 2 ≥ m → false) ↔ m < -2 :=
by
  sorry

end no_solution_for_inequalities_l261_261978


namespace brownie_pieces_count_l261_261661

def pan_width : ℕ := 24
def pan_height : ℕ := 15
def brownie_width : ℕ := 3
def brownie_height : ℕ := 2

theorem brownie_pieces_count : (pan_width * pan_height) / (brownie_width * brownie_height) = 60 := by
  sorry

end brownie_pieces_count_l261_261661


namespace range_of_f_l261_261153

noncomputable def f (x : ℝ) : ℝ := sin x * tan x + cos x * cot x

theorem range_of_f :
  set.range f = set.Ici (real.sqrt 2) :=
begin
  sorry
end

end range_of_f_l261_261153


namespace alexander_paid_amount_l261_261394

/-- The amount Alexander paid for the tickets is 3600 rubles. -/
theorem alexander_paid_amount :
  let A := 600 in
  let B := 800 in
  let cost_alexander := 2 * A + 3 * B in
  let cost_anna := 3 * A + 2 * B in
  cost_alexander = cost_anna + 200 → cost_alexander = 3600 :=
by
  intros A B cost_alexander cost_anna h
  unfold A B cost_alexander cost_anna at *
  have h1: 2 * 600 + 3 * 800 = 3600 := by norm_num
  rw h1
  have h2: 3 * 600 + 2 * 800 = 2800 := by norm_num
  have h_final: 3600 = 2800 + 200 := by norm_num
  rw h_final at h
  exact h

end alexander_paid_amount_l261_261394


namespace total_ear_muffs_bought_l261_261446

-- Define the number of ear muffs bought before December
def ear_muffs_before_dec : ℕ := 1346

-- Define the number of ear muffs bought during December
def ear_muffs_during_dec : ℕ := 6444

-- The total number of ear muffs bought by customers
theorem total_ear_muffs_bought : ear_muffs_before_dec + ear_muffs_during_dec = 7790 :=
by
  sorry

end total_ear_muffs_bought_l261_261446


namespace find_number_of_triangles_l261_261112

def is_solution (x1 x2 : ℕ) : bool :=
  let y1 := 2017 - 37 * x1
  let y2 := 2017 - 37 * x2
  2017 * (x1 - x2) % 2 = 0

noncomputable def count_solutions : ℕ :=
  let evens := {x : ℕ | x ≤ 54 ∧ x % 2 = 0}.card
  let odds := {x : ℕ | x ≤ 54 ∧ x % 2 = 1}.card
  nat.choose evens 2 + nat.choose odds 2

theorem find_number_of_triangles : count_solutions = 729 :=
by
  sorry

end find_number_of_triangles_l261_261112


namespace favorite_movies_total_hours_l261_261256

theorem favorite_movies_total_hours (michael_hrs joyce_hrs nikki_hrs ryn_hrs sam_hrs alex_hrs : ℕ)
  (H1 : nikki_hrs = 30)
  (H2 : michael_hrs = nikki_hrs / 3)
  (H3 : joyce_hrs = michael_hrs + 2)
  (H4 : ryn_hrs = (4 * nikki_hrs) / 5)
  (H5 : sam_hrs = (3 * joyce_hrs) / 2)
  (H6 : alex_hrs = 2 * michael_hrs) :
  michael_hrs + joyce_hrs + nikki_hrs + ryn_hrs + sam_hrs + alex_hrs = 114 := 
sorry

end favorite_movies_total_hours_l261_261256


namespace players_indoor_and_outdoor_l261_261039

-- Define the conditions
constant total_players : ℕ := 400
constant outdoor_players : ℕ := 350
constant indoor_players : ℕ := 110

-- Define the nature of the problem
theorem players_indoor_and_outdoor (B : ℕ) :
  total_players = (outdoor_players - B) + (indoor_players - B) + B →
  B = 60 :=
by
  intros h
  sorry

end players_indoor_and_outdoor_l261_261039


namespace tangent_line_equation_l261_261954

theorem tangent_line_equation :
  (∀ x, f (-x) = -f x) → (∀ x, x < 0 → f x = log (-x) + 2 * x) →
  (∃ a b c : ℝ, (∀ x y : ℝ, y = f x → a * x + b * y + c = 0) ∧
    (∀ x y : ℕ, (x, y) = (1, f 1) → a * x + b * y + c = 0)) :=
  sorry

end tangent_line_equation_l261_261954


namespace rooks_on_checkerboard_l261_261333

theorem rooks_on_checkerboard : ∃ n : ℕ, n = 2880 ∧ ∀ (board : fin 9 × fin 9 → Prop), 
    (∀ r1 r2 c1 c2 : fin 9, r1 ≠ r2 → c1 ≠ c2 → board ⟨r1, c1⟩ → board ⟨r2, c2⟩ → false) ↔ 
    (board (0,0) ∨ board (0,1) ∨ board (1,0) ∨ board (1,1) → 
    ∃ f : fin 9 → fin 9, function.injective f ∧ ∀ i, board ⟨i, f i⟩) := 
begin
  use 2880,
  split,
  { refl, },
  sorry
end

end rooks_on_checkerboard_l261_261333


namespace probability_closer_to_6_than_0_l261_261421

-- Defining the probability problem
theorem probability_closer_to_6_than_0 :
  let interval := set.Icc 0 7 in
  (∀ point : ℝ, point ∈ interval → point > 3) → 
  (measure_theory.measure_space.measure (set.Icc 3 7) / 
   measure_theory.measure_space.measure interval = 4 / 7) :=
by
  sorry

end probability_closer_to_6_than_0_l261_261421


namespace fraction_increase_by_two_times_l261_261590

theorem fraction_increase_by_two_times (x y : ℝ) : 
  let new_val := ((2 * x) * (2 * y)) / (2 * x + 2 * y)
  let original_val := (x * y) / (x + y)
  new_val = 2 * original_val := 
by
  sorry

end fraction_increase_by_two_times_l261_261590


namespace tan_half_angle_positive_l261_261205

theorem tan_half_angle_positive (k : ℤ) (θ : ℝ) 
  (h : (π / 2 + 2 * k * π) < θ ∧ θ < (π + 2 * k * π)) : 
  0 < tan (θ / 2) := 
sorry

end tan_half_angle_positive_l261_261205


namespace find_last_num_divisible_by_12_stopping_at_84_l261_261850

theorem find_last_num_divisible_by_12_stopping_at_84 :
  ∃ N, (N = 84) ∧ (71 ≤ N) ∧ (let concatenated := string.join ((list.range (N - 70)).map (λ i, (string.of_nat (i + 71)))) in 
    (nat.divisible (int.of_nat (string.to_nat concatenated)) 12)) :=
begin
  sorry
end

end find_last_num_divisible_by_12_stopping_at_84_l261_261850


namespace students_exceed_guinea_pigs_l261_261467

theorem students_exceed_guinea_pigs :
  let classrooms := 5
  let students_per_classroom := 20
  let guinea_pigs_per_classroom := 3
  let total_students := classrooms * students_per_classroom
  let total_guinea_pigs := classrooms * guinea_pigs_per_classroom
  total_students - total_guinea_pigs = 85 :=
by
  -- using the conditions and correct answer identified above
  let classrooms := 5
  let students_per_classroom := 20
  let guinea_pigs_per_classroom := 3
  let total_students := classrooms * students_per_classroom
  let total_guinea_pigs := classrooms * guinea_pigs_per_classroom
  show total_students - total_guinea_pigs = 85
  sorry

end students_exceed_guinea_pigs_l261_261467


namespace external_common_tangents_of_circles_l261_261180

/--
Given two circles with equations (x + 3)^2 + y^2 = 9 and (x - 1)^2 + y^2 = 1,
prove that the equations of the external common tangent lines of these circles are
y = (sqrt 3 / 3) * (x - 3) and y = - (sqrt 3 / 3) * (x - 3).
-/
theorem external_common_tangents_of_circles 
    (A : ℝ → ℝ → Prop := λ x y, (x + 3)^2 + y^2 = 9)
    (B : ℝ → ℝ → Prop := λ x y, (x - 1)^2 + y^2 = 1) :
    ∃ k : ℝ, (A (x:=3) (y:=k * (3 - 3)) → B (x:=k * (3 - 1)) (y:=k * (x - 1))) ∧ 
             (k = sqrt 3 / 3 ∨ k = - sqrt 3 / 3) := sorry

end external_common_tangents_of_circles_l261_261180


namespace arina_sophia_divisible_l261_261836

theorem arina_sophia_divisible (N: ℕ) (k: ℕ) (large_seq: list ℕ): 
  (k = 81) → 
  (large_seq = (list.range' 71 (k + 1)).append (list.range' 82 (N + 1))) → 
  (nat.digits 10 (list.foldl (λ n d, 10 * n + d) 0 large_seq)).nat_sum % 3 = 0 → 
  (nat.digits 10 (list.foldl (λ n d, 10 * n + d) 0 large_seq)).last_digits 2 % 4 = 0 →
  (list.foldl (λ n d, 10 * n + d) 0 large_seq) % 12 = 0 → 
  N = 84 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end arina_sophia_divisible_l261_261836


namespace simplify_expression_l261_261547

theorem simplify_expression {a b c : ℝ} (h₁ : a + b > c) (h₂ : a + c > b) (h₃ : b + c > a) :
  |a + c - b| - |a + b + c| + |2b + c| = c :=
sorry

end simplify_expression_l261_261547


namespace bug_returns_eighth_move_l261_261064

def recurrence_relation (P : ℕ → ℚ) : Prop :=
∀ n, P (n + 1) = (2 / 3 : ℚ) - (1 / 3 : ℚ) * P n

def initial_condition (P : ℕ → ℚ) : Prop :=
P 0 = 1

theorem bug_returns_eighth_move :
  ∃ (P : ℕ → ℚ),
    recurrence_relation P ∧
    initial_condition P ∧
    P 8 = (3248 / 6561 : ℚ) ∧
    nat.coprime 3248 6561 ∧
    3248 + 6561 = 9809 :=
by
  sorry

end bug_returns_eighth_move_l261_261064


namespace solve_cubic_eq_l261_261924

theorem solve_cubic_eq (z : ℂ) : z^3 = 27 ↔ (z = 3 ∨ z = - (3 / 2) + (3 / 2) * Complex.I * Real.sqrt 3 ∨ z = - (3 / 2) - (3 / 2) * Complex.I * Real.sqrt 3) :=
by
  sorry

end solve_cubic_eq_l261_261924


namespace age_of_b_l261_261388

variables {a b : ℕ}

theorem age_of_b (h₁ : a + 10 = 2 * (b - 10)) (h₂ : a = b + 11) : b = 41 :=
sorry

end age_of_b_l261_261388


namespace gcd_solutions_l261_261486

theorem gcd_solutions (x m n p: ℤ) (h_eq: x * (4 * x - 5) = 7) (h_gcd: Int.gcd m (Int.gcd n p) = 1)
  (h_form: ∃ x1 x2: ℤ, x1 = (m + Int.sqrt n) / p ∧ x2 = (m - Int.sqrt n) / p) : m + n + p = 150 :=
by
  have disc_eq : 25 + 112 = 137 :=
    by norm_num
  sorry

end gcd_solutions_l261_261486


namespace train_speed_l261_261085

/-- Define the conditions -/
def train_length : ℝ := 100 -- meters
def time_to_cross : ℝ := 6 -- seconds
def man_speed_kmph : ℝ := 5 -- kmph
def man_speed_mps : ℝ := (man_speed_kmph * 1000) / 3600 -- convert to m/s

/-- Define the hypothesis and the theorem we want to prove -/
theorem train_speed (relative_speed_mps : ℝ) :
  relative_speed_mps = train_length / time_to_cross →
  let train_speed_mps := relative_speed_mps - man_speed_mps in
  let train_speed_kmph := (train_speed_mps * 3600) / 1000 in
  train_speed_kmph = 55 :=
by
  intros h1
  let train_speed_mps := relative_speed_mps - man_speed_mps
  let train_speed_kmph := (train_speed_mps * 3600) / 1000
  have h2 : train_speed_mps = relative_speed_mps - man_speed_mps := rfl
  have h3 : relative_speed_mps = train_length / time_to_cross := h1
  have h4 : train_speed_kmph = (train_speed_mps * 3600) / 1000 := rfl
  sorry

end train_speed_l261_261085


namespace find_f_expression_l261_261151

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_expression (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 1) : 
  f (x) = (1 / (x - 1)) :=
by sorry

example (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 1) (hx: f (1 / x) = x / (1 - x)) :
  f x = 1 / (x - 1) :=
find_f_expression x h₀ h₁

end find_f_expression_l261_261151


namespace geometric_sequence_eighth_term_l261_261372

noncomputable def a_8 : ℕ :=
  let a₁ := 8
  let r := 2
  a₁ * r^(8-1)

theorem geometric_sequence_eighth_term : a_8 = 1024 := by
  sorry

end geometric_sequence_eighth_term_l261_261372


namespace lattice_points_integer_distance_l261_261511

theorem lattice_points_integer_distance {n : ℕ} (hn : 3 ≤ n) :
  ∃ (points : fin n → ℤ × ℤ), 
    (∀ i j, i ≠ j → points i ≠ points j) ∧ 
    (¬ ∃ (a b : ℤ) (c : ℤ), ∀ i, (a * (points i).fst + b * (points i).snd = c)) ∧ 
    (∀ i j, i ≠ j → ∃ k : ℤ, (points i).fst - (points j).fst = k * k ∧ 
                             (points i).snd - (points j).snd = k * k) :=
by 
  sorry

end lattice_points_integer_distance_l261_261511


namespace count_integers_l261_261577

def floor (x : ℚ) : ℤ := if (x % 1 = 0) then x.nat_abs else x.nat_abs - 1

theorem count_integers (n : ℕ) :
  (floor (n / 2) + floor (n / 3) + floor (n / 6) = n) ↔
  (n < 2007 ∧ ∃ (k : ℕ), k < 335 ∧ n = 6 * k) :=
sorry

end count_integers_l261_261577


namespace bisect_iff_bisect_l261_261296

open EuclideanGeometry

variables {A B C D E F L R S T : Point} 

-- Assume the conditions
axiom h1 : Collinear A B C
axiom h2 : D ∈ Segment B C
axiom h3 : E ∈ Segment A C
axiom h4 : F ∈ Segment A B
axiom h5 : Parallel Lines D E A B
axiom h6 : Parallel Lines D F A C
axiom h7 : Ratio (Length B D) (Length D C) = Ratio (Length A B) (Length A C) ^ 2
axiom h8 : ∃ circle1 : Circle, OnCircle A E F R ∧ OnCircle circle1 A
axiom h9 : Tangent (Circumcircle A B C) A S
axiom h10 : LineIntersection E F B C L
axiom h11 : LineIntersection S R T

-- Lean 4 goal statement: Prove SR bisects AB if and only if BS bisects TL
theorem bisect_iff_bisect (hSR_bisects_AB : Bisects S R A B) : Bisects B S T L ↔ Bisects S R A B :=
sorry

end bisect_iff_bisect_l261_261296


namespace calculate_fraction_l261_261586

theorem calculate_fraction (x y : ℚ) (h1 : x = 5 / 6) (h2 : y = 6 / 5) : (1 / 3) * x^8 * y^9 = 2 / 5 := by
  sorry

end calculate_fraction_l261_261586


namespace second_derivative_trig_l261_261921

noncomputable def y (x : ℝ) : ℝ := sorry  -- define y(x) implicitly

theorem second_derivative_trig :
  ∀ x : ℝ, (HasDerivAt (λ y, (Real.arctan (2 * y) + y)) x) →
  ∃ y'' : ℝ, y'' = - (8 * (y x) * (1 + 4 * (y x)^2)) / (3 + 4 * (y x)^2)^3 :=
by
  intro x h
  have hy : ∀ x : ℝ, Real.arctan (2 * y x) + y x - x = 0 → (HasDerivAt (y x)) :=
    sorry  -- Given condition converted to Lean definition
  have hy' : ∀ x : ℝ, HasDerivAt (y x) → y' x = (1 + 4 * (y x)^2) / (3 + 4 * (y x)^2) :=
    sorry  -- Equation derived for first derivative
  have hy'' : ∀ x : ℝ, y'' x = - (8 * (y x) * (1 + 4 * (y x)^2)) / (3 + 4 * (y x)^2)^3 :=
    sorry  -- Final solution for second derivative
  exact ⟨hy'', h⟩

end second_derivative_trig_l261_261921


namespace min_pieces_for_net_l261_261076

theorem min_pieces_for_net (n : ℕ) : ∃ (m : ℕ), m = n * (n + 1) := by
  sorry

end min_pieces_for_net_l261_261076


namespace simplify_expression_l261_261272

variable {R : Type} [LinearOrderedField R]

theorem simplify_expression (x y z : R) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) (h_sum : x + y + z = 3) :
  (1 / (y^2 + z^2 - x^2) + 1 / (x^2 + z^2 - y^2) + 1 / (x^2 + y^2 - z^2)) =
    3 / (-9 + 6 * y + 6 * z - 2 * y * z) :=
  sorry

end simplify_expression_l261_261272


namespace number_in_tens_place_is_7_l261_261029

theorem number_in_tens_place_is_7
  (digits : Finset ℕ)
  (a b c : ℕ)
  (h1 : digits = {7, 5, 2})
  (h2 : 100 * a + 10 * b + c > 530)
  (h3 : 100 * a + 10 * b + c < 710)
  (h4 : a ∈ digits)
  (h5 : b ∈ digits)
  (h6 : c ∈ digits)
  (h7 : ∀ x ∈ digits, x ≠ a → x ≠ b → x ≠ c) :
  b = 7 := sorry

end number_in_tens_place_is_7_l261_261029


namespace sum_possible_values_of_M_l261_261510

/-- For a set of five distinct lines in a plane, the sum of all possible values of M, where M is the number of distinct points that lie on two or more lines, is 55. -/
theorem sum_possible_values_of_M : 
  ∑ m in {m : ℕ | ∃ L : finset (set point), L.card = 5 ∧ m = cardinal.mk { p : point | ∃ l₁ l₂ ∈ L, l₁ ≠ l₂ ∧ p ∈ l₁ ∩ l₂ } }.finset, m = 55 := 
by sorry

end sum_possible_values_of_M_l261_261510


namespace roots_of_polynomial_l261_261128

-- Define the polynomial
def poly := fun (x : ℝ) => x^3 - 7 * x^2 + 14 * x - 8

-- Define the statement
theorem roots_of_polynomial : (poly 1 = 0) ∧ (poly 2 = 0) ∧ (poly 4 = 0) :=
  by
  sorry

end roots_of_polynomial_l261_261128


namespace find_sin_2alpha_l261_261991

theorem find_sin_2alpha (α : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) 
    (h2 : 3 * Real.cos (2 * α) = Real.sqrt 2 * Real.sin (π / 4 - α)) : 
  Real.sin (2 * α) = -8 / 9 := 
sorry

end find_sin_2alpha_l261_261991


namespace rate_percent_l261_261434

variables (P : ℝ) (r : ℝ)
def A (n : ℕ) : ℝ := P * (1 + r / 100) ^ n

theorem rate_percent
  (A2 : ℝ := 2420)
  (A3 : ℝ := 2783)
  (h1 : A 2 = A2)
  (h2 : A 3 = A3) :
  r = 15 :=
by sorry

end rate_percent_l261_261434


namespace apple_is_in_box_B_l261_261740

variable (P : Type) [DecidableEq P]
variable (Box : P -> Prop)
(variable has_apple_in_box : P -> Prop)
(variable note_on_box_GL : P -> Prop)

-- Conditions
def note_on_A := Box A = has_apple_in_box A
def note_on_B := Box B = ¬has_apple_in_box B
def note_on_C := Box C = ¬has_apple_in_box A

-- Only one of these notes tells the truth.
def one_note_truthful :=
  ( (\exists! (A), note_on_box_GL A) \/
    (\exists! (B), note_on_box_GL B) \/
    (\exists! (C), note_on_box_GL C) )

-- Question: Prove that the apple is in box B
theorem apple_is_in_box_B :
  one_note_truthful → has_apple_in_box B := 
sorry

end apple_is_in_box_B_l261_261740


namespace part1_I_part1_II_part2_III_l261_261150

open Real

-- Definitions of the conditions
variables {a x y : ℝ}

-- Lean statements corresponding to the problem
theorem part1_I (ha : a + a⁻¹ = 5 / 2) (ha_gt1 : 1 < a) :
  a^(-1/2) + a^(1/2) = 3*sqrt(2) / 2 :=
sorry

theorem part1_II (ha : a + a⁻¹ = 5 / 2) (ha_gt1 : 1 < a) :
  a^(3/2) + a^(-3/2) = 9*sqrt(2) / 4 :=
sorry

theorem part2_III (hlog_eq : 2 * log (x - 2 * y) = log x + log y) (hx_gt2y : 0 < x - 2 * y) :
  log a (y / x) = -2 :=
sorry

end part1_I_part1_II_part2_III_l261_261150


namespace isosceles_triangle_ADE_l261_261632

variables {A B C D F E : Type}
variables [IsTrapezoid A B C D] [IsParallel AB CD] [AngleBisector AF (∠ D A B)]
variables [Intersection E (Line AF) CD]

theorem isosceles_triangle_ADE : IsIsoscelesTriangle A D E :=
sorry

end isosceles_triangle_ADE_l261_261632


namespace calc_exp_product_l261_261878

theorem calc_exp_product :
  (cbrt 125) * (root 4 256) * (sqrt 16) = 80 := sorry

end calc_exp_product_l261_261878


namespace parabola_directrix_l261_261492

theorem parabola_directrix (x y : ℝ) : 
  (∀ x: ℝ, y = -4 * x ^ 2 + 4) → (y = 65 / 16) := 
by 
  sorry

end parabola_directrix_l261_261492


namespace max_distance_between_points_on_spheres_l261_261374

noncomputable def max_distance_on_spheres : ℝ :=
  let C1 := (3 : ℝ, -4 : ℝ, 7 : ℝ)
  let C2 := (-8 : ℝ, 9 : ℝ, -10 : ℝ)
  let radius1 := 23
  let radius2 := 76
  let distance_centers := Real.sqrt ((3 + 8)^2 + (-4 - 9)^2 + (7 + 10)^2)
  radius1 + distance_centers + radius2

theorem max_distance_between_points_on_spheres :
  max_distance_on_spheres = 99 + Real.sqrt 579 :=
by
  -- Proof (or "sorry" to skip it)
  sorry

end max_distance_between_points_on_spheres_l261_261374


namespace bonnets_per_orphanage_l261_261666

theorem bonnets_per_orphanage :
  let monday_bonnets := 10
  let tuesday_wednesday_bonnets := 2 * monday_bonnets
  let thursday_bonnets := monday_bonnets + 5
  let friday_bonnets := thursday_bonnets - 5
  let total_bonnets := monday_bonnets + tuesday_wednesday_bonnets + thursday_bonnets + friday_bonnets
  let orphanages := 5
  total_bonnets / orphanages = 11 :=
by
  let monday_bonnets := 10
  let tuesday_wednesday_bonnets := 2 * monday_bonnets
  let thursday_bonnets := monday_bonnets + 5
  let friday_bonnets := thursday_bonnets - 5
  let total_bonnets := monday_bonnets + tuesday_wednesday_bonnets + thursday_bonnets + friday_bonnets
  let orphanages := 5
  sorry

end bonnets_per_orphanage_l261_261666


namespace propositions_validity_l261_261829

theorem propositions_validity (a b : ℝ) (r l α : ℝ) (f : ℝ → ℝ) (x : ℝ) :
  (∀ x ∈ ℝ, sin x ≥ -1) ∧ ¬(∃ x ∈ ℝ, sin x < -1) ∧
  (a < b ∧ b < 0 → ¬( - (1 / a) > - (1 / b) )) ∧
  (∀ x, f x = (log a (x - 1)) + 1 → f 2 = 1) ∧ 
  (2 * r + l = 6 → (1 / 2) * l * r = 2 → (0 < α ∧ α < π) → α = 1) :=
by
  intros
  sorry

end propositions_validity_l261_261829


namespace smallest_degree_polynomial_l261_261700

theorem smallest_degree_polynomial : 
  ∃ (p : Polynomial ℚ), 
    (p ≠ 0) ∧ 
    (p.eval (2 - Real.sqrt 5) = 0) ∧ 
    (p.eval (-2 - Real.sqrt 5) = 0) ∧ 
    (p.eval (-1 + Real.sqrt 3) = 0) ∧ 
    (p.eval (-1 - Real.sqrt 3) = 0) ∧ 
    (Degree.of_polynomial p = 6) :=
sorry

end smallest_degree_polynomial_l261_261700


namespace sum_possible_values_of_k_l261_261236

theorem sum_possible_values_of_k (j k : ℕ) (h : j > 0) (h1 : k > 0) (h2 : 1 / (j:ℚ) + 1 / (k:ℚ) = 1 / 4) : 
  sum_possible_values_of_k (h2) = 51 :=
by
  sorry

end sum_possible_values_of_k_l261_261236


namespace sum_possible_values_of_k_l261_261237

theorem sum_possible_values_of_k (j k : ℕ) (h : j > 0) (h1 : k > 0) (h2 : 1 / (j:ℚ) + 1 / (k:ℚ) = 1 / 4) : 
  sum_possible_values_of_k (h2) = 51 :=
by
  sorry

end sum_possible_values_of_k_l261_261237


namespace cannot_form_right_triangle_with_set_C_l261_261383

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem cannot_form_right_triangle_with_set_C :
  ¬ is_right_triangle 4 7 5 := by
  sorry

end cannot_form_right_triangle_with_set_C_l261_261383


namespace sum_even_coefficients_l261_261518

theorem sum_even_coefficients :
  ∀ (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 a_12 : ℤ),
    (∀ x : ℤ, (x - 1)^4 * (x + 2)^8 = a * x^12 + a_1 * x^11 + a_2 * x^10 + a_3 * x^9 +
                                      a_4 * x^8 + a_5 * x^7 + a_6 * x^6 + a_7 * x^5 +
                                      a_8 * x^4 + a_9 * x^3 + a_{10} * x^2 + a_{11} * x + a_{12})
    → a_2 + a_4 + a_6 + a_8 + a_{10} + a_{12} = 7 := by
  intros a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 a_12 h,
  sorry

end sum_even_coefficients_l261_261518


namespace integral_problem_correct_l261_261787

noncomputable def integral_problem : Prop :=
  ∫ x in 0..4, (e ^ sqrt((4 - x) / (4 + x))) * (1 / ((4 + x) * sqrt(16 - x ^ 2))) = (1 / 4) * (Real.exp 1 - 1)

theorem integral_problem_correct : integral_problem :=
by
  sorry

end integral_problem_correct_l261_261787


namespace smallest_solution_x4_minus_50x2_plus_625_eq_0_l261_261759

theorem smallest_solution_x4_minus_50x2_plus_625_eq_0 :
  ∃ (x : ℝ), (x * x * x * x - 50 * x * x + 625 = 0) ∧ (∀ y, (y * y * y * y - 50 * y * y + 625 = 0) → x ≤ y) :=
sorry

end smallest_solution_x4_minus_50x2_plus_625_eq_0_l261_261759


namespace fixed_point_f_l261_261028

variable (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1)

def f (x : ℝ) : ℝ := a^(x - 2) - 3

theorem fixed_point_f : f a 2 = -2 := by
  sorry

end fixed_point_f_l261_261028


namespace directrix_parabola_l261_261495

-- Given the equation of the parabola and required transformations:
theorem directrix_parabola (d : ℚ) : 
  (∀ x : ℚ, y = -4 * x^2 + 4) → d = 65 / 16 :=
by sorry

end directrix_parabola_l261_261495


namespace det_nonzero_if_k_minors_eq_zero_l261_261058

open Matrix

variables {n k : ℕ} (A : Matrix (Fin n) (Fin n) ℂ)

noncomputable def minors_n_minus_1 (A : Matrix (Fin n) (Fin n) ℂ) : ℕ :=
  -- Assume a function that calculates the number of (n-1)-order minors of A equal to 0
  sorry

-- The main statement:
theorem det_nonzero_if_k_minors_eq_zero (hn : 2 ≤ n) (hk : 1 ≤ k) (hk2 : k ≤ n - 1) (hA : minors_n_minus_1 A = k) :
  det A ≠ 0 :=
sorry

end det_nonzero_if_k_minors_eq_zero_l261_261058


namespace possible_values_of_M_l261_261515

def is_two_digit_number (M : ℕ) := 10 ≤ M ∧ M < 100
def digits (M : ℕ) (p q : ℕ) := M = 10 * p + q
def reversed_digits (M : ℕ) (p q : ℕ) := 10 * q + p
def perfect_cube (n : ℕ) := ∃ k : ℕ, k^3 = n
def cube_condition (diff : ℕ) := 8 < diff ∧ diff ≤ 64 ∧ perfect_cube diff
def divisible_by_5 (M : ℕ) := M % 5 = 0

theorem possible_values_of_M (M : ℕ) (p q : ℕ):
  is_two_digit_number M →
  digits M p q →
  let diff := abs ((10 * p + q) - (10 * q + p)) in
  cube_condition diff →
  divisible_by_5 M →
  M ∈ {25, 30, 85} :=
by
  intros
  sorry

end possible_values_of_M_l261_261515


namespace guitar_price_proof_l261_261871

def total_guitar_price (x : ℝ) : Prop :=
  0.20 * x = 240 → x = 1200

theorem guitar_price_proof (x : ℝ) (h : 0.20 * x = 240) : x = 1200 :=
by
  sorry

end guitar_price_proof_l261_261871


namespace arithmetic_sequence_a3_value_l261_261235

theorem arithmetic_sequence_a3_value {a : ℕ → ℕ}
  (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_sum : a 1 + a 2 + a 3 + a 4 + a 5 = 20) :
  a 3 = 4 :=
sorry

end arithmetic_sequence_a3_value_l261_261235


namespace Wilsons_number_l261_261031

theorem Wilsons_number (N : ℝ) (h : N - N / 3 = 16 / 3) : N = 8 := sorry

end Wilsons_number_l261_261031


namespace value_modulo_7_l261_261025

theorem value_modulo_7 : 
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 7 = 5 := 
  by 
  sorry

end value_modulo_7_l261_261025


namespace a_general_formula_S_n_formula_l261_261280

-- Given conditions
variable (a : ℕ → ℕ) (n : ℕ)

axiom a_1 : a 1 = 3
axiom a_recurrence : ∀ n : ℕ, a (n + 1) = 3 * a n - 4 * n

-- General formula for the sequence
theorem a_general_formula : ∀ n: ℕ, a n = 2 * n + 1 :=
by
  sorry

-- Given sequence b_n in terms of a_n
def b (n : ℕ) := 2^n * a n

-- Sum of the first n terms of b_n
def S (n : ℕ) := ∑ k in finset.range n, b k

-- Sum formula
theorem S_n_formula : ∀ n : ℕ, S n = (2*n - 1) * 2^(n + 1) + 2 :=
by
  sorry

end a_general_formula_S_n_formula_l261_261280


namespace domain_f_domain_g_intersection_M_N_l261_261593

namespace MathProof

open Set

def M : Set ℝ := { x | -2 < x ∧ x < 4 }
def N : Set ℝ := { x | x < 1 ∨ x ≥ 3 }

theorem domain_f :
  (M = { x : ℝ | -2 < x ∧ x < 4 }) := by
  sorry

theorem domain_g :
  (N = { x : ℝ | x < 1 ∨ x ≥ 3 }) := by
  sorry

theorem intersection_M_N : 
  (M ∩ N = { x : ℝ | (-2 < x ∧ x < 1) ∨ (3 ≤ x ∧ x < 4) }) := by
  sorry

end MathProof

end domain_f_domain_g_intersection_M_N_l261_261593


namespace find_m_l261_261984

variables {m : ℝ}

def vec_a : ℝ × ℝ := (1, real.sqrt 3)
def vec_b : ℝ × ℝ := (3, m)
def angle : ℝ := real.pi / 6

theorem find_m : 
  let dot_product := (vec_a.1 * vec_b.1) + (vec_a.2 * vec_b.2)
  let norm_a := real.sqrt ((vec_a.1 ^ 2) + (vec_a.2 ^ 2))
  let norm_b := real.sqrt ((vec_b.1 ^ 2) + (vec_b.2 ^ 2))
  let cos_theta := real.cos angle
  dot_product = norm_a * norm_b * cos_theta →
  m = real.sqrt 3 :=
by
  sorry

end find_m_l261_261984


namespace luke_bus_time_l261_261662

theorem luke_bus_time
  (L : ℕ)   -- Luke's bus time to work in minutes
  (P : ℕ)   -- Paula's bus time to work in minutes
  (B : ℕ)   -- Luke's bike time home in minutes
  (h1 : P = 3 * L / 5) -- Paula's bus time is \( \frac{3}{5} \) of Luke's bus time
  (h2 : B = 5 * L)     -- Luke's bike time is 5 times his bus time
  (h3 : L + P + B + P = 504) -- Total travel time is 504 minutes
  : L = 70 := 
sorry

end luke_bus_time_l261_261662


namespace meeting_point_l261_261356

theorem meeting_point (n : ℕ) (petya_start vasya_start petya_end vasya_end meeting_lamp : ℕ) : 
  n = 100 → petya_start = 1 → vasya_start = 100 → petya_end = 22 → vasya_end = 88 → meeting_lamp = 64 :=
by
  intros h_n h_p_start h_v_start h_p_end h_v_end
  sorry

end meeting_point_l261_261356


namespace last_appended_number_is_84_l261_261843

theorem last_appended_number_is_84 : 
  ∃ N : ℕ, 
    let s := "7172737475767778798081" ++ (String.intercalate "" (List.map toString [82, 83, 84])) in
    (N = 84) ∧ (s.toNat % 12 = 0) :=
by
  sorry

end last_appended_number_is_84_l261_261843


namespace probability_A2_l261_261088

/-- Definitions based on the conditions -/
def P_A1 : ℝ := 0.5
def P_B1 : ℝ := 0.5
def P_A2_given_A1 : ℝ := 0.4
def P_A2_given_B1 : ℝ := 0.6

/-- Theorem statement based on the problem -/
theorem probability_A2 : 
  let P_A2 := P_A1 * P_A2_given_A1 + P_B1 * P_A2_given_B1 in
  P_A2 = 0.5 :=
by
  sorry

end probability_A2_l261_261088


namespace log10_sum_diff_l261_261450

noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log10_sum_diff :
  log10 32 + log10 50 - log10 8 = 2.301 :=
by
  sorry

end log10_sum_diff_l261_261450


namespace hyperbola_proof_l261_261975

variable (a b c k : ℝ)
variables (A B P Q M : ℝ × ℝ)
variable (x y : ℝ)

def hyperbola (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def focus (F : ℝ × ℝ) : Prop := F = (2, 0)
def asymptote1 (x : ℝ) : ℝ := sqrt 3 * x
def asymptote2 (x : ℝ) : ℝ := - (sqrt 3 * x)

def slope_P (P : ℝ × ℝ) := - sqrt 3
def slope_Q (Q : ℝ × ℝ) := sqrt 3

def cond1 (M A B : ℝ × ℝ) : Prop := ∃ k, M = (k * (A.1 - B.1), k * (A.2 - B.2)) 
def cond2 (P Q A B : ℝ × ℝ) : Prop := P.2 - Q.2 = (P.1 - Q.1) * (A.2 - B.2) / (A.1 - B.1)
def cond3 (M A B : ℝ × ℝ) : Prop := dist M A = dist M B

theorem hyperbola_proof (h_hyperbola : hyperbola 1 (sqrt 3) x y)
    (h_focus : focus (2,0)) 
    (h_asymptote1 : ∀ x, asymptote1 x = sqrt 3 * x) 
    (h_asymptote2 : ∀ x, asymptote2 x = -sqrt 3 * x)
    (h_PQ_parallel_AB: cond2 P Q A B) 
    (h_M_on_AB: cond1 M A B):
      cond3 M A B := 
sorry

end hyperbola_proof_l261_261975


namespace Jerry_travel_time_l261_261251

theorem Jerry_travel_time
  (speed_j speed_b distance_j distance_b time_j time_b : ℝ)
  (h_speed_j : speed_j = 40)
  (h_speed_b : speed_b = 30)
  (h_distance_b : distance_b = distance_j + 5)
  (h_time_b : time_b = time_j + 1/3)
  (h_distance_j : distance_j = speed_j * time_j)
  (h_distance_b_eq : distance_b = speed_b * time_b) :
  time_j = 1/2 :=
by
  sorry

end Jerry_travel_time_l261_261251


namespace initial_current_correct_steady_state_voltage_correct_heat_dissipated_correct_l261_261065

def capacitor_current (C U0 R : ℝ) : ℝ :=
  U0 / R

def steady_state_voltage (C U0 : ℝ) : ℝ :=
  U0 / 5

def heat_dissipated (C U0 : ℝ) : ℝ :=
  (2 / 5) * C * (U0 ^ 2)

theorem initial_current_correct (C U0 R : ℝ) :
  capacitor_current C U0 R = U0 / R :=
by
  sorry

theorem steady_state_voltage_correct (C U0 : ℝ) :
  steady_state_voltage C U0 = U0 / 5 :=
by
  sorry

theorem heat_dissipated_correct (C U0 : ℝ) :
  heat_dissipated C U0 = (2 / 5) * C * (U0 ^ 2) :=
by
  sorry

end initial_current_correct_steady_state_voltage_correct_heat_dissipated_correct_l261_261065


namespace rooks_on_checkerboard_l261_261320

theorem rooks_on_checkerboard (n : ℕ) (board_size : ℕ) (even_rooks : ℕ) (odd_rooks : ℕ)
  (checkerboard : matrix (fin board_size) (fin board_size) bool)
  (coloring : (i j : fin board_size) → bool) :
  board_size = 9 → n = 9 → even_rooks = 4 → odd_rooks = 5 →
  coloring = λ i j, (i.val + j.val) % 2 = 0 → 
  (finset.univ.filter (λ x : fin board_size × fin board_size, coloring x.1 x.2)).card = even_rooks^2 + odd_rooks^2 →
  ∑ (perm : equiv.perm (fin even_rooks)), 1 * ∑ (qerm : equiv.perm (fin odd_rooks)), 1 = 2880 :=
by 
  intros _ _ _ _ _ _ _ _ _ _;
  sorry

end rooks_on_checkerboard_l261_261320


namespace beads_pulled_out_l261_261106

theorem beads_pulled_out (white_beads black_beads : ℕ) (frac_black frac_white : ℚ) (h_black : black_beads = 90) (h_white : white_beads = 51) (h_frac_black : frac_black = (1/6)) (h_frac_white : frac_white = (1/3)) : 
  white_beads * frac_white + black_beads * frac_black = 32 := 
by
  sorry

end beads_pulled_out_l261_261106


namespace chocolate_bars_left_l261_261798

noncomputable def initial_bars : ℕ := 500

def thomas_and_friends_take (bars : ℕ) : ℕ :=
  (bars / 3).div 7 * 7 + 2

def piper_take (bars : ℕ) : ℕ := bars / 4 - 7

def paul_take (piper_initial_take : ℕ) : ℕ := piper_initial_take + 5

def bars_left (initial : ℕ) (total_taken : ℕ) : ℕ := initial - total_taken

theorem chocolate_bars_left :
  let initial := initial_bars in
  let friends_take := thomas_and_friends_take initial in
  let piper_initial_take := initial / 4 in
  let piper_net_take := piper_take initial in
  let paul_net_take := paul_take piper_initial_take in
  bars_left initial (friends_take + piper_net_take + paul_net_take) = 96 :=
  sorry

end chocolate_bars_left_l261_261798


namespace total_calories_burned_l261_261293

def base_distance : ℝ := 15
def records : List ℝ := [0.1, -0.8, 0.9, 16.5 - base_distance, 2.0, -1.5, 14.1 - base_distance, 1.0, 0.8, -1.1]
def calorie_burn_rate : ℝ := 20

theorem total_calories_burned :
  (base_distance * 10 + (List.sum records)) * calorie_burn_rate = 3040 :=
by
  sorry

end total_calories_burned_l261_261293


namespace smallest_solution_of_quartic_equation_l261_261752

theorem smallest_solution_of_quartic_equation :
  ∃ x : ℝ, (x^4 - 50 * x^2 + 625 = 0) ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y := sorry

end smallest_solution_of_quartic_equation_l261_261752


namespace area_of_polygon_ABCDEF_l261_261941

-- Definitions based on conditions
def AB : ℕ := 8
def BC : ℕ := 10
def DC : ℕ := 5
def FA : ℕ := 7
def GF : ℕ := 3
def ED : ℕ := 7
def height_GF_ED : ℕ := 2

-- Area calculations based on given conditions
def area_ABCG : ℕ := AB * BC
def area_trapezoid_GFED : ℕ := (GF + ED) * height_GF_ED / 2

-- Proof statement
theorem area_of_polygon_ABCDEF :
  area_ABCG - area_trapezoid_GFED = 70 :=
by
  simp [area_ABCG, area_trapezoid_GFED]
  sorry

end area_of_polygon_ABCDEF_l261_261941


namespace correct_average_weight_of_class_l261_261227

theorem correct_average_weight_of_class:
  let n := 40 in
  let avg_weight := 62.3 in
  let misread_weights := (54, 70, 63) in
  let correct_weights := (58, 65, 68) in
  let incorrect_total := avg_weight * n in
  let correction := ((correct_weights.1 - misread_weights.1) +
                     (correct_weights.2 - misread_weights.2) +
                     (correct_weights.3 - misread_weights.3)) in
  let correct_total := incorrect_total + correction in
  let correct_avg_weight := correct_total / n in
  correct_avg_weight = 62.4 :=
by
  sorry

end correct_average_weight_of_class_l261_261227


namespace length_of_box_l261_261384

theorem length_of_box 
  (width height num_cubes length : ℕ)
  (h_width : width = 16)
  (h_height : height = 13)
  (h_cubes : num_cubes = 3120)
  (h_volume : length * width * height = num_cubes) :
  length = 15 :=
by
  sorry

end length_of_box_l261_261384


namespace rhombus_diagonal_length_l261_261172

theorem rhombus_diagonal_length (area d1 d2 : ℝ) (h₁ : area = 24) (h₂ : d1 = 8) (h₃ : area = (d1 * d2) / 2) : d2 = 6 := 
by sorry

end rhombus_diagonal_length_l261_261172


namespace geometric_sequence_ratio_l261_261625

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 2 + a 8 = 15) 
  (h2 : a 3 * a 7 = 36) 
  (h_geom : ∀ n, a (n + 1) = a n * q) : 
  (a 19 / a 13 = 4) ∨ (a 19 / a 13 = 1 / 4) :=
by
  sorry

end geometric_sequence_ratio_l261_261625


namespace bell_rings_count_l261_261283

-- Defining the conditions
def bell_rings_per_class : ℕ := 2
def total_classes_before_music : ℕ := 4
def bell_rings_during_music_start : ℕ := 1

-- The main proof statement
def total_bell_rings : ℕ :=
  total_classes_before_music * bell_rings_per_class + bell_rings_during_music_start

theorem bell_rings_count : total_bell_rings = 9 := by
  sorry

end bell_rings_count_l261_261283


namespace intersection_eq_l261_261194

-- Define the sets M and N
def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := { x | -1 ≤ x ∧ x ≤ 1 }

-- The statement to prove
theorem intersection_eq : M ∩ N = {0, 1} := by
  sorry

end intersection_eq_l261_261194


namespace paint_house_wash_windows_l261_261911

theorem paint_house (people_paint : ℕ) (days_paint : ℕ) (work_days_paint : ℕ) (days_required : ℕ) : 
  (people_paint = 8) → (days_paint = 5) → 
  work_days_paint = people_paint * days_paint → 
  ((work_days_ppaint / days_required).ceil - people_paint = 6) 
  := by
  sorry

theorem wash_windows (people_wash : ℕ) (days_wash : ℕ) (work_days_wash : ℕ) (extra_people : ℕ) (days_needed : ℕ) : 
  (people_wash = 8) → (days_wash = 4) → 
  work_days_wash = people_wash * days_wash → 
  (extra_people = 4) →
  days_needed = (work_days_wash / (people_wash + extra_people)).ceil →
  days_needed = 3 
  := by
  sorry

end paint_house_wash_windows_l261_261911


namespace arina_sophia_divisible_l261_261834

theorem arina_sophia_divisible (N: ℕ) (k: ℕ) (large_seq: list ℕ): 
  (k = 81) → 
  (large_seq = (list.range' 71 (k + 1)).append (list.range' 82 (N + 1))) → 
  (nat.digits 10 (list.foldl (λ n d, 10 * n + d) 0 large_seq)).nat_sum % 3 = 0 → 
  (nat.digits 10 (list.foldl (λ n d, 10 * n + d) 0 large_seq)).last_digits 2 % 4 = 0 →
  (list.foldl (λ n d, 10 * n + d) 0 large_seq) % 12 = 0 → 
  N = 84 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end arina_sophia_divisible_l261_261834


namespace fraction_subtraction_l261_261204

theorem fraction_subtraction (x y : ℝ) (h : x / y = 3 / 2) : (x - y) / y = 1 / 2 := 
by 
  sorry

end fraction_subtraction_l261_261204


namespace ratio_TR_UR_l261_261336

-- Definitions for problem conditions
def Square (PQRS : Type) :=
  ∃ P Q R S : PQRS, ∃ (PQ RS : PQRS → ℝ) (side_length : ℝ),
    PQ P = 0 ∧ PQ R = side_length ∧
    RS P = 0 ∧ RS S = side_length ∧
    RS Q = side_length ∧ RS R = 0 

def QuarterCircle (arc : Type) (Q S : arc) :=
  ∃ (radius : ℝ), radius > 0 ∧
  ∀ (point : arc), distance Q point ≤ radius ∧ distance S point ≤ radius

def Midpoint (midpoint : Type) (Q R : midpoint) (U : midpoint) :=
  distance Q U = distance U R

def LiesOn (point : Type) (S R : point) (T : point) :=
  true -- Simplification for problem context

def Tangent (line : Type) (arc : Type) (T U : line) :=
  true -- Simplification for problem context

-- Variables for proof
variable {PQRS arc midpoint point line : Type}
variables {P Q R S T U : PQRS}
variables {side_length : ℝ}
variables [Square PQRS]
variables [QuarterCircle arc Q S]
variables [Midpoint midpoint Q R U]
variables [LiesOn point S R T]
variables [Tangent line arc T U]

-- Original math problem transformed into a Lean 4 statement.
theorem ratio_TR_UR (TR UR : ℝ) :
  let side_length := 2 in
  let UQ := side_length / 2 in
  let UR := side_length / 2 in
  let x := TR in
  1^2 + x^2 = (3 - x)^2 →
  x = 4/3 →
  TR / UR = 4 / 3 :=
by
  sorry -- Proof is assumed and not required.

end ratio_TR_UR_l261_261336


namespace appended_number_divisible_by_12_l261_261861

theorem appended_number_divisible_by_12 :
  ∃ N, (N = 88) ∧ (∀ n, n ∈ finset.range N \ 71 → (let large_number := (list.range (N + 1)).filter (λ x, 71 ≤ x ∧ x ≤ N) in
       (list.foldr (λ a b, a * 100 + b) 0 large_number) % 12 = 0)) :=
by
  sorry

end appended_number_divisible_by_12_l261_261861


namespace karen_locks_l261_261257

theorem karen_locks : 
  let L1 := 5
  let L2 := 3 * L1 - 3
  let Lboth := 5 * L2
  Lboth = 60 :=
by
  let L1 := 5
  let L2 := 3 * L1 - 3
  let Lboth := 5 * L2
  sorry

end karen_locks_l261_261257


namespace probability_boarding_251_l261_261444

theorem probability_boarding_251 :
  let interval_152 := 5
  let interval_251 := 7
  let total_events := interval_152 * interval_251
  let favorable_events := (interval_152 * interval_152) / 2
  (favorable_events / total_events : ℚ) = 5 / 14 :=
by 
  sorry

end probability_boarding_251_l261_261444


namespace solution_set_of_inequality_l261_261551

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (h : ∀ x, f(x+1) = f(-(x+1)))
axiom strictly_increasing (h : ∀ {x y}, x < y → f(x+1) < f(y+1))

theorem solution_set_of_inequality (x : ℝ) :
  f(-2^x) > f(-8) ↔ x < 3 :=
by
  sorry

end solution_set_of_inequality_l261_261551


namespace bulk_bag_holds_40_oz_l261_261576

def bag_cost_before_coupon := 25 -- dollars
def coupon_value := 5 -- dollars
def cost_per_serving_after_coupon := 0.50 -- dollars per serving
def serving_size := 1 -- oz per serving

theorem bulk_bag_holds_40_oz : 
  let bag_cost_after_coupon := bag_cost_before_coupon - coupon_value in
  let number_of_servings := bag_cost_after_coupon / cost_per_serving_after_coupon in
  let ounces_per_bag := number_of_servings * serving_size in
  ounces_per_bag = 40 :=
by
  sorry

end bulk_bag_holds_40_oz_l261_261576


namespace solve_for_a_l261_261552

theorem solve_for_a (a : ℝ) : (∃ x : ℝ, 2 * x + a - 9 = 0 ∧ x = 2) → a = 5 :=
by
  intro h
  cases h with x hx
  cases hx with eq_x sol_eq
  sorry

end solve_for_a_l261_261552


namespace raffle_tickets_sold_l261_261385

theorem raffle_tickets_sold (total_amount : ℕ) (ticket_cost : ℕ) (tickets_sold : ℕ) 
    (h1 : total_amount = 620) (h2 : ticket_cost = 4) : tickets_sold = 155 :=
by {
  sorry
}

end raffle_tickets_sold_l261_261385


namespace white_pairs_coincide_l261_261120

theorem white_pairs_coincide 
  (red_half : ℕ) (blue_half : ℕ) (white_half : ℕ)
  (red_pairs : ℕ) (blue_pairs : ℕ) (red_white_pairs : ℕ) :
  red_half = 2 → blue_half = 4 → white_half = 6 →
  red_pairs = 1 → blue_pairs = 2 → red_white_pairs = 2 →
  2 * (red_half - red_pairs + blue_half - 2 * blue_pairs + 
       white_half - 2 * red_white_pairs) = 4 :=
by
  intros 
    h_red_half h_blue_half h_white_half 
    h_red_pairs h_blue_pairs h_red_white_pairs
  rw [h_red_half, h_blue_half, h_white_half, 
      h_red_pairs, h_blue_pairs, h_red_white_pairs]
  sorry

end white_pairs_coincide_l261_261120


namespace sqrt_inequality_l261_261462

theorem sqrt_inequality : 2 * Real.sqrt 2 - Real.sqrt 7 < Real.sqrt 6 - Real.sqrt 5 := by sorry

end sqrt_inequality_l261_261462


namespace length_of_BC_in_triangle_l261_261246

theorem length_of_BC_in_triangle :
  ∀ (A B C : Type) [MetricSpace A] [HasDistance A] [AddGroup C] [Module ℝ C] [InnerProductSpace ℝ C],
    ∀ (a b c : C),
      dist a b = sqrt 19 →
      dist a c = 2 →
      ∠ b a c = real.pi * (2 / 3) →
      dist b c = 3 :=
by
  intros A B C _ _ _ _ _ a b c hab hac hbac
  -- sorry to complete the proof
  sorry

end length_of_BC_in_triangle_l261_261246


namespace nina_homework_total_l261_261290

-- Definitions based on conditions
def ruby_math_homework : Nat := 6
def ruby_reading_homework : Nat := 2
def nina_math_homework : Nat := 4 * ruby_math_homework
def nina_reading_homework : Nat := 8 * ruby_reading_homework
def nina_total_homework : Nat := nina_math_homework + nina_reading_homework

-- The theorem to prove
theorem nina_homework_total : nina_total_homework = 40 := by
  sorry

end nina_homework_total_l261_261290


namespace shaded_region_area_l261_261350

def diagonal := 8 -- AB = 8 cm
def congruent_squares (n: ℕ) := 20 -- The shaded region consists of 20 congruent squares

-- Given AB is the diagonal of a square, calculate the area of the shaded region
theorem shaded_region_area :
  let area_square := (diagonal ^ 2) / 2 in
  let small_square_area := area_square / 16 in
  congruent_squares * small_square_area = 40 :=
sorry

end shaded_region_area_l261_261350


namespace min_value_of_expression_l261_261548

theorem min_value_of_expression (x y : ℝ) (h1 : x > -1) (h2 : y > 0) (h3 : x + y = 1) : 
  ∃ min_value, min_value = 9 / 2 ∧ ∀ z, z = (1 / (x + 1) + 4 / y) → z ≥ min_value :=
sorry

end min_value_of_expression_l261_261548


namespace pentagon_area_l261_261245

noncomputable def area_of_pentagon (A B C D E : Point) : Real :=
  if AB = 1 ∧ AE = 1 ∧ CD = 1 ∧ (BC + DE = 1) ∧ (angle ABC = 90) ∧ (angle AED = 90)
  then area_pentagon ABCDE
  else 0

theorem pentagon_area :
  ∀ (A B C D E : Point),
    AB = 1 →
    AE = 1 →
    CD = 1 →
    BC + DE = 1 →
    angle ABC = 90 →
    angle AED = 90 →
    area_of_pentagon A B C D E = 1 := 
by sorry

end pentagon_area_l261_261245


namespace find_last_num_divisible_by_12_stopping_at_84_l261_261845

theorem find_last_num_divisible_by_12_stopping_at_84 :
  ∃ N, (N = 84) ∧ (71 ≤ N) ∧ (let concatenated := string.join ((list.range (N - 70)).map (λ i, (string.of_nat (i + 71)))) in 
    (nat.divisible (int.of_nat (string.to_nat concatenated)) 12)) :=
begin
  sorry
end

end find_last_num_divisible_by_12_stopping_at_84_l261_261845


namespace max_unmarried_women_l261_261292

theorem max_unmarried_women (total_people : ℕ) (fraction_women : ℚ) (fraction_married : ℚ)
  (htotal : total_people = 80)
  (hwomen : fraction_women = 2/5)
  (hmarried : fraction_married = 1/2) :
  let number_of_women := fraction_women * total_people,
      number_of_married := fraction_married * total_people
  in number_of_women = 32 ∧ number_of_married = 40 ∧ number_of_women = 32 :=
by
  sorry

end max_unmarried_women_l261_261292


namespace hyperbola_center_origin_opens_vertically_l261_261806

noncomputable def t_squared : ℝ :=
  let a_sq := (64 / 5 : ℝ) in
  let y := 2 in
  let x := 2 in
  let frac := (frac := y^2 / 4 - 5 * x^2 / a_sq) in
  (frac + 5 / 16 - 1) in
  frac * 4 / 16

theorem hyperbola_center_origin_opens_vertically
  (a_sq : ℝ := 64 / 5)
  (y : ℝ := 2)
  (x : ℝ := 2) : t_squared = 21 / 4 :=
by
  sorry

end hyperbola_center_origin_opens_vertically_l261_261806


namespace profit_difference_correct_l261_261817

noncomputable def profit_difference (P : ℝ) (rX rY rZ : ℚ) : ℝ :=
  let sum_ratios := (rX + rY + rZ).to_real
  let part_value := P / sum_ratios
  let shareX := (rX.to_real) * part_value
  let shareY := (rY.to_real) * part_value
  let shareZ := (rZ.to_real) * part_value
  max shareX (max shareY shareZ) - min shareX (min shareY shareZ)

theorem profit_difference_correct
  (P : ℝ) (rX rY rZ : ℚ)
  (hP : P = 3000) 
  (hX : rX = 2/3) 
  (hY : rY = 1/4) 
  (hZ : rZ = 1/6) :
  profit_difference P rX rY rZ = 1384.61 := 
by
  simp [profit_difference, hP, hX, hY, hZ]
  -- Continue the proof as necessary
  sorry

end profit_difference_correct_l261_261817


namespace distance_midpoint_directrix_l261_261793

noncomputable def parabola : ℝ → ℝ := λ x, (8 * x) ^ (1/2)

def point_A : ℝ × ℝ := (8, 8)
def point_F : ℝ × ℝ := (4, 0)
def directrix : ℝ := -4

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem distance_midpoint_directrix :
  ∀ (B : ℝ × ℝ),
  B.1 = 1/2 ∧ B.2 = -2 →
  ∥ (midpoint point_A B).1 + 4 ∥ = 25 / 4 :=
  by
    intros B hB
    cases hB with hBx hBy
    rw [midpoint]
    rw [fst_add, snd_add]
    rw [fst_div, snd_div]
    rw [hBx, hBy]
    rw [fst_point_A, snd_point_A]
    norm_num
    sorry

end distance_midpoint_directrix_l261_261793


namespace sqrt_inequality_l261_261723

variables (a b c : ℝ)

theorem sqrt_inequality (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  sqrt (b^2 - a * c) < sqrt 3 * a :=
sorry

end sqrt_inequality_l261_261723


namespace find_a10_l261_261192

noncomputable theory
open_locale classical

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → (1 / a (n + 1) = 1 / a n + 1 / 3)

theorem find_a10 {a : ℕ → ℝ} (h : sequence a) : a 10 = 1 / 4 :=
by sorry

end find_a10_l261_261192


namespace birdhouse_volume_difference_l261_261688

-- Definitions to capture the given conditions
def sara_width_ft : ℝ := 1
def sara_height_ft : ℝ := 2
def sara_depth_ft : ℝ := 2

def jake_width_in : ℝ := 16
def jake_height_in : ℝ := 20
def jake_depth_in : ℝ := 18

-- Convert Sara's dimensions to inches
def ft_to_in (x : ℝ) : ℝ := x * 12
def sara_width_in := ft_to_in sara_width_ft
def sara_height_in := ft_to_in sara_height_ft
def sara_depth_in := ft_to_in sara_depth_ft

-- Volume calculations
def volume (width height depth : ℝ) := width * height * depth
def sara_volume := volume sara_width_in sara_height_in sara_depth_in
def jake_volume := volume jake_width_in jake_height_in jake_depth_in

-- The theorem to prove the difference in volume
theorem birdhouse_volume_difference : sara_volume - jake_volume = 1152 := by
  -- Proof goes here
  sorry

end birdhouse_volume_difference_l261_261688


namespace total_selling_price_correct_l261_261074

def original_price : ℝ := 100
def discount_percent : ℝ := 0.30
def tax_percent : ℝ := 0.08

theorem total_selling_price_correct :
  let discount := original_price * discount_percent
  let sale_price := original_price - discount
  let tax := sale_price * tax_percent
  let total_selling_price := sale_price + tax
  total_selling_price = 75.6 := by
sorry

end total_selling_price_correct_l261_261074


namespace total_distance_covered_l261_261072

noncomputable def distance_covered_by_fly (radius : ℝ) (final_segment : ℝ) : ℝ :=
  let diameter := 2 * radius
  let other_leg := real.sqrt (diameter ^ 2 - final_segment ^ 2)
  diameter + final_segment + other_leg

theorem total_distance_covered
  (radius : ℝ)
  (final_segment : ℝ)
  (h_radius : radius = 75)
  (h_final_segment : final_segment = 100) :
  distance_covered_by_fly radius final_segment = 361.8 := by
  sorry

end total_distance_covered_l261_261072


namespace arina_sophia_divisible_l261_261837

theorem arina_sophia_divisible (N: ℕ) (k: ℕ) (large_seq: list ℕ): 
  (k = 81) → 
  (large_seq = (list.range' 71 (k + 1)).append (list.range' 82 (N + 1))) → 
  (nat.digits 10 (list.foldl (λ n d, 10 * n + d) 0 large_seq)).nat_sum % 3 = 0 → 
  (nat.digits 10 (list.foldl (λ n d, 10 * n + d) 0 large_seq)).last_digits 2 % 4 = 0 →
  (list.foldl (λ n d, 10 * n + d) 0 large_seq) % 12 = 0 → 
  N = 84 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end arina_sophia_divisible_l261_261837


namespace final_number_appended_is_84_l261_261851

noncomputable def arina_sequence := "7172737475767778798081"

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def is_divisible_by_12 (n : ℕ) : Prop := n % 12 = 0

-- Define adding numbers to the sequence
def append_number (seq : String) (n : ℕ) : String := seq ++ n.repr

-- Create the full sequence up to 84 and check if it's divisible by 12
def generate_full_sequence : String :=
  let base_seq := arina_sequence
  let full_seq := append_number (append_number (append_number (append_number (append_number (append_number (append_number (append_number (append_number arina_sequence 82) 83) 84))) 85) 86) 87) 88 
  full_seq

theorem final_number_appended_is_84 : (∃ seq : String, is_divisible_by_12(seq.to_nat) ∧ seq.ends_with "84") := 
by
  sorry

end final_number_appended_is_84_l261_261851


namespace none_of_these_l261_261995

-- Problem Statement:
theorem none_of_these (r x y : ℝ) (h1 : r > 0) (h2 : x ≠ 0) (h3 : y ≠ 0) (h4 : x^2 + y^2 > x^2 * y^2) :
  ¬ (-x > -y) ∧ ¬ (-x > y) ∧ ¬ (1 > -y / x) ∧ ¬ (1 < x / y) :=
by
  sorry

end none_of_these_l261_261995


namespace log_sqrt_7_of_343sqrt7_l261_261123

noncomputable def log_sqrt_7 (y : ℝ) : ℝ := 
  Real.log y / Real.log (Real.sqrt 7)

theorem log_sqrt_7_of_343sqrt7 : log_sqrt_7 (343 * Real.sqrt 7) = 4 :=
by
  sorry

end log_sqrt_7_of_343sqrt7_l261_261123


namespace time_for_B_is_24_days_l261_261032

noncomputable def A_work : ℝ := (1 / 2) / (3 / 4)
noncomputable def B_work : ℝ := 1 -- assume B does 1 unit of work in 1 day
noncomputable def total_work : ℝ := (A_work + B_work) * 18

theorem time_for_B_is_24_days : 
  ((A_work + B_work) * 18) / B_work = 24 := by
  sorry

end time_for_B_is_24_days_l261_261032


namespace absolute_value_inequality_l261_261502

theorem absolute_value_inequality (x : ℝ) : 
  (2 ≤ |x - 3| ∧ |x - 3| ≤ 4) ↔ (-1 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 7) := 
by sorry

end absolute_value_inequality_l261_261502


namespace lines_concurrent_l261_261248

-- Declaration of the necessary components for the proof
variables {K I A R E : Type*}
variables [InnerProductSpace ℝ K] [InnerProductSpace ℝ I] [InnerProductSpace ℝ A]
          [InnerProductSpace ℝ R] [InnerProductSpace ℝ E]
variables (points : ℝ → Type*) (lies_on : points R → Prop)
variables (KA KI AE RI : ℝ)
variables (bisector : ℝ)

-- Conditions
axiom KA_lt_KI : KA < KI
axiom perpendicular_R : is_perpendicular (foot R (bisector angle_of K))
axiom perpendicular_E : is_perpendicular (foot E (bisector angle_of K))

-- Proof goal
theorem lines_concurrent :
  ∃ M : points, lies_on M IE ∧ lies_on M RA ∧ lies_on M (⊥ KR) :=
sorry

end lines_concurrent_l261_261248


namespace distance_and_area_of_triangle_l261_261643

noncomputable def vertex_of_parabola (a b c : ℝ) : (ℝ × ℝ) :=
  let x := -b / (2 * a) in
  (x, a * x^2 + b * x + c)

theorem distance_and_area_of_triangle :
  let A := vertex_of_parabola 1 6 5 in
  let B := vertex_of_parabola 1 (-4) 12 in
  let distance := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) in
  let area := (1 / 2) * Real.abs (A.1 * (B.2 - 0) + B.1 * (0 - A.2)) in
  distance = 13 ∧ area = 8 := by
  sorry

end distance_and_area_of_triangle_l261_261643


namespace max_sum_of_entries_l261_261821

def sum_of_numbers : ℕ := 2 + 3 + 4 + 5 + 7 + 11 + 13 + 14

def mul_table (a b c d e f g h : ℕ) : ℕ :=
  (a + b + c + d) * (e + f + g + h)

theorem max_sum_of_entries : ∃ a b c d e f g h : ℕ,
  {a, b, c, d, e, f, g, h} = {2, 3, 4, 5, 7, 11, 13, 14} ∧
  mul_table a b c d e f g h = 868 :=
sorry

end max_sum_of_entries_l261_261821


namespace negation_proposition_1_negation_proposition_2_proof_proposition_1_proof_proposition_2_l261_261775

theorem negation_proposition_1 (x : ℝ) :
  (¬ ∃ x : ℝ, x^2 - x + 1 < 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≥ 0 :=
begin
  sorry, -- Prove that the negation of the existential proposition is equivalent to the universal proposition
end

theorem negation_proposition_2 :
  (¬ ∃ x : ℝ, x^2 - 4 = 0) ↔ ∀ x : ℝ, ¬ (x^2 - 4 = 0) :=
begin
  sorry, -- Prove that the negation of the existential proposition is equivalent to the universal proposition
end

theorem proof_proposition_1 :
  (¬ ∃ x : ℝ, x^2 - x + 1 < 0) :=
begin
  -- The discriminant of x^2 - x + 1 is negative, so this polynomial has no real roots
  sorry,
end

theorem proof_proposition_2 :
  ∃ x : ℝ, x^2 - 4 = 0 :=
begin
  -- Solve x^2 - 4 = 0, which has real roots x = 2 and x = -2
  sorry,
end

end negation_proposition_1_negation_proposition_2_proof_proposition_1_proof_proposition_2_l261_261775


namespace solution_set_of_linear_system_l261_261830

theorem solution_set_of_linear_system :
  {p : ℝ × ℝ | let (x, y) := p in x + y = 3 ∧ x - y = 1} = {(2, 1)} :=
sorry

end solution_set_of_linear_system_l261_261830


namespace percentage_reduction_l261_261802

theorem percentage_reduction :
  let P := 60
  let R := 45
  (900 / R) - (900 / P) = 5 →
  (P - R) / P * 100 = 25 :=
by 
  intros P R h
  have h1 : R = 45 := rfl
  have h2 : P = 60 := sorry
  rw [h1] at h
  rw [h2]
  sorry -- detailed steps to be filled in the proof

end percentage_reduction_l261_261802


namespace seating_arrangements_l261_261360

theorem seating_arrangements :
  ∃ (n : ℕ), 
  let front_seats := 4 in
  let back_seats := 5 in
  let non_adjacent := λ (a b : ℕ), abs (a - b) ≠ 1 in
  n = (3 * 2) + (6 * 2) + (5 * 4 * 2) :=
begin
  existsi 58,
  sorry
end

end seating_arrangements_l261_261360


namespace part_a_part_b_l261_261048

-- Definitions based on first proof problem
def side_length_greater_than_one (a : ℝ) := a > 1
def broken_line_length_one (l : ℝ) := l = 1
def equilateral_triangle  {V : Type*} [metric_space V] (A B C : V) :=
(dist A B = dist B C) ∧ (dist B C = dist C A)

noncomputable def enclose_broken_line_in_triangle (a l : ℝ) : Prop := 
∀ {V : Type*} [metric_space V] (A B C : V), equilateral_triangle A B C → 
side_length_greater_than_one a → broken_line_length_one l →
∃ (f : ℝ → V), (∀ t, dist (f t) (f (0 : ℝ)) ≤ 1) ∧ (dist (f (0 : ℝ)) (f 1) = l)

theorem part_a (a l : ℝ) : 
side_length_greater_than_one a → broken_line_length_one l → enclose_broken_line_in_triangle a l :=
by
  intros
  sorry


-- Definitions based on second proof problem
def side_length_equal_one (a : ℝ) := a = 1
def convex_broken_line_length_one (l : ℝ) := l = 1

noncomputable def enclose_convex_broken_line_in_triangle (a l : ℝ) : Prop := 
∀ {V : Type*} [metric_space V] (A B C : V), equilateral_triangle A B C → 
side_length_equal_one a → convex_broken_line_length_one l →
∃ (f : ℝ → V), (∀ t, dist (f t) (f (0 : ℝ)) ≤ 1) ∧ (dist (f (0 : ℝ)) (f 1) = l)

theorem part_b (a l : ℝ) : 
side_length_equal_one a → convex_broken_line_length_one l → enclose_convex_broken_line_in_triangle a l :=
by
  intros
  sorry

end part_a_part_b_l261_261048


namespace rooks_on_checkerboard_l261_261324

theorem rooks_on_checkerboard :
  let n := 9
  let board := λ (i j : ℕ), (i % 2 = j % 2) -- checkerboard pattern condition
  let (evenCoords := ((fin n).val.enum.filter (λ (i,j), i%2=0 ∧ j%2=0)).length, -- even r-bord condition
       oddCoords := ((fin n).val.enum.filter (λ (i,j), i%2=1 ∧ j%2=1)).length
       in
     even_coords_board_size == 4 ∧ odd_coords_board_size == 5)
  /- Assert that the number of ways to place non-attacking rooks on the black cells is 4! * 5! -/
  then 
    (∃(φ : (fin n).val.enum.map(board).card = 2880,
  sorry

end rooks_on_checkerboard_l261_261324


namespace sin_sum_arcsin_arctan_l261_261473

theorem sin_sum_arcsin_arctan :
  let a := Real.arcsin (4 / 5)
  let b := Real.arctan (1 / 2)
  Real.sin (a + b) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end sin_sum_arcsin_arctan_l261_261473


namespace part_a_part_b_l261_261164

-- Definitions based on conditions
noncomputable def S_oplus {S : Set ℕ} (n : ℕ) : Set ℕ :=
{m | ∃ s ∈ S, m = s + n }

noncomputable def S_seq : ℕ → Set ℕ
| 1 := {1}
| (k + 1) := (S_oplus (S_seq k) (k + 1)) ∪ {2 * (k + 1) - 1}

-- Part (a) statement
theorem part_a : { n : ℕ | ∀ k, n ∉ S_seq k } = {2^m | m ∈ ℕ} :=
sorry

-- Part (b) statement
theorem part_b : ∃ n, 1994 ∈ S_seq n ∧ n = 7 :=
sorry

end part_a_part_b_l261_261164


namespace sum_of_squared_residuals_l261_261565

theorem sum_of_squared_residuals :
  let regression_line (x : ℝ) := 2 * x + 1
  let data_points := [(2, 4.9), (3, 7.1), (4, 9.1)]
  let residuals := data_points.map (λ p, p.2 - regression_line p.1)
  (residuals.map (λ e, e ^ 2)).sum = 0.03 :=
by
  sorry

end sum_of_squared_residuals_l261_261565


namespace find_subtracted_number_l261_261083

theorem find_subtracted_number (x y : ℤ) (h1 : x = 129) (h2 : 2 * x - y = 110) : y = 148 := by
  have hx : 2 * 129 - y = 110 := by
    rw [h1] at h2
    exact h2
  linarith

end find_subtracted_number_l261_261083


namespace terminal_side_in_third_quadrant_l261_261521

theorem terminal_side_in_third_quadrant (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : 
  (∃ k : ℤ, α = k * π + π / 2 + π) := sorry

end terminal_side_in_third_quadrant_l261_261521


namespace compute_f_at_919_l261_261952

-- Given conditions
def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

def periodic_function (f : ℝ → ℝ) : Prop :=
∀ x, f (x + 4) = f (x - 2)

noncomputable def f (x : ℝ) : ℝ :=
if h : x ∈ [-3, 0] then 6^(-x) else sorry

-- Lean statement for the proof problem
theorem compute_f_at_919 (f : ℝ → ℝ)
    (h_even : is_even_function f)
    (h_periodic : periodic_function f)
    (h_defined : ∀ x ∈ [-3, 0], f x = 6^(-x)) :
    f 919 = 6 := sorry

end compute_f_at_919_l261_261952


namespace bus_passengers_cannot_pay_fare_l261_261357

theorem bus_passengers_cannot_pay_fare :
  ∀ (passengers : ℕ) (coins : ℕ) (fare_per_passenger : ℕ),
    (passengers = 40) ∧ 
    (coins = 49) ∧ 
    (fare_per_passenger = 5) ∧ 
    (∀ coins_value ∈ {10, 15, 20}) → 
    ¬ (∃ arrangement : (list ℕ), 
        arrangement.length = 40 ∧ 
        (arrangement.sum * fare_per_passenger = 200) ∧ 
        arrangement.sum ≤ coins)
:= 
by
  sorry

end bus_passengers_cannot_pay_fare_l261_261357


namespace distance_points_3D_l261_261133

open Real

def distance_between_points (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

theorem distance_points_3D {p1 p2 : ℝ × ℝ × ℝ} (h1 : p1 = (3, -2, 5)) (h2 : p2 = (7, 4, 2)) :
  distance_between_points p1 p2 = sqrt 61 :=
by
  rw [h1, h2]
  simp [distance_between_points]
  norm_num
  sorry

end distance_points_3D_l261_261133


namespace cats_visit_cost_l261_261006

variable (T_doctor_visit T_payout_for_doctor_visit Cat_insurance_coverage T_total_payment Cat_total_visit_cost : ℝ)
variable (insurance_coverage_percentage : ℝ)

-- Assume the conditions given in the problem
def problem_conditions :=
  T_doctor_visit = 300 ∧
  insurance_coverage_percentage = 0.75 ∧
  T_payout_for_doctor_visit = 0.25 * T_doctor_visit ∧
  Cat_insurance_coverage = 60 ∧
  T_total_payment = 135

-- We want to prove that the cat's visit cost $195
theorem cats_visit_cost (h : problem_conditions) : Cat_total_visit_cost = 195 :=
sorry -- To be proved

end cats_visit_cost_l261_261006


namespace find_last_num_divisible_by_12_stopping_at_84_l261_261849

theorem find_last_num_divisible_by_12_stopping_at_84 :
  ∃ N, (N = 84) ∧ (71 ≤ N) ∧ (let concatenated := string.join ((list.range (N - 70)).map (λ i, (string.of_nat (i + 71)))) in 
    (nat.divisible (int.of_nat (string.to_nat concatenated)) 12)) :=
begin
  sorry
end

end find_last_num_divisible_by_12_stopping_at_84_l261_261849


namespace find_g_solve_inequality_find_lambda_range_l261_261175

noncomputable theory

def f (x : ℝ) := x^2 + 2 * x
def g (x : ℝ) := -x^2 + 2 * x

theorem find_g :
  g = (λ x, -x^2 + 2 * x) := sorry

theorem solve_inequality :
  ∀ x : ℝ, g x ≥ f x - |x - 1| ↔ -1/2 ≤ x ∧ x < 1 := sorry

theorem find_lambda_range (h : ℝ → ℝ) (λ : ℝ) :
  (h = λ x, g x - λ * f x + 1) → 
  (∀ x, -1 ≤ x ∧ x ≤ 1 → h x ≤ h (x + ε) for all ε > 0) → 
  λ ≤ 0 := sorry

end find_g_solve_inequality_find_lambda_range_l261_261175


namespace sin_phi_eq_1_l261_261646

variable (A B C D P Q : Point)
variable (AB CD AD BC : Line)
variable (ABCD_rect : is_rectangle A B C D)
variable (AB_len AD_len : ℝ)
variable (P_mid Q_mid : Point)

def is_midpoint (P : Point) (A B : Point) : Prop := 
  P = midpoint A B

axiom AB_val : AB_len = 4
axiom AD_val : AD_len = 2
axiom P_mid_AB : is_midpoint P A B
axiom Q_mid_BC : is_midpoint Q B C

noncomputable def sin_phi : ℝ :=
  let phi := find_angle_between_lines AD P in
  real.sin phi

theorem sin_phi_eq_1 : sin_phi A B C D P Q AB CD AD BC ABCD_rect AB_len AD_len P_mid Q_mid = 1 :=
by
  sorry

end sin_phi_eq_1_l261_261646


namespace picked_number_is_45_l261_261201

def chosen_number (n : ℕ) :=
  n ∈ {41, 43, 45, 47} ∧ 43 < n ∧ n < 46

theorem picked_number_is_45 : ∃ n, chosen_number n ∧ n = 45 :=
by
  sorry

end picked_number_is_45_l261_261201


namespace sum_of_possible_ks_l261_261238

theorem sum_of_possible_ks (j k : ℕ) (hj : 0 < j) (hk : 0 < k) (h : (1 : ℚ) / j + (1 : ℚ) / k = 1 / 4) :
    k ∈ {20, 12, 8, 6, 5} ∧ {5, 6, 8, 12, 20}.sum = 51 :=
by sorry

end sum_of_possible_ks_l261_261238


namespace sum_of_perpendiculars_eq_twice_side_l261_261815

noncomputable def square_perpendicular_sum_equal_twice_side (s : ℝ) (P : ℝ × ℝ)
  (AB BC CD DA : ℝ) (d1 d2 d3 d4 : ℝ) : Prop :=
  let ABCD := s*s in
  let area_expression := s * (d1 + d2 + d3 + d4) / 2 in
  ABCD = area_expression

theorem sum_of_perpendiculars_eq_twice_side (s : ℝ) (P : ℝ × ℝ)
  (d1 d2 d3 d4 : ℝ)
  (h1 : d1 = abs(P.1 - 0))
  (h2 : d2 = abs(P.2 - 0))
  (h3 : d3 = abs(P.1 - s))
  (h4 : d4 = abs(P.2 - s)) :
  d1 + d2 + d3 + d4 = 2 * s :=
sorry

end sum_of_perpendiculars_eq_twice_side_l261_261815


namespace AUME_seating_arrangements_l261_261314

open Finset

noncomputable def number_of_valid_arrangements : ℕ :=
  let factorial := nat.fact 4 in
  281 * (factorial ^ 3)

theorem AUME_seating_arrangements :
  ∃ M : ℕ, 
    let factorial := nat.fact 4 in
    factorial * factorial * factorial * M = 281 * (factorial * factorial * factorial * 1) :=
begin
  use 281,
  simp,
end

end AUME_seating_arrangements_l261_261314


namespace sin_sum_arcsin_arctan_l261_261477

-- Definitions matching the conditions
def a := Real.arcsin (4 / 5)
def b := Real.arctan (1 / 2)

-- Theorem stating the question and expected answer
theorem sin_sum_arcsin_arctan : 
  Real.sin (a + b) = (11 * Real.sqrt 5) / 25 := 
by 
  sorry

end sin_sum_arcsin_arctan_l261_261477


namespace largest_expr_is_a_squared_plus_b_squared_l261_261933

noncomputable def largest_expression (a b : ℝ) (h₁ : a > 1) (h₂ : b > 1) (h₃ : a ≠ b) : Prop :=
  (a^2 + b^2 > a - b) ∧ (a^2 + b^2 > a + b) ∧ (a^2 + b^2 > 2 * a * b)

theorem largest_expr_is_a_squared_plus_b_squared (a b : ℝ) (h₁ : a > 1) (h₂ : b > 1) (h₃ : a ≠ b) : 
  largest_expression a b h₁ h₂ h₃ :=
by
  sorry

end largest_expr_is_a_squared_plus_b_squared_l261_261933


namespace min_value_of_expression_l261_261136

theorem min_value_of_expression : ∀ x y : ℝ, (x^2 * y - 1)^2 + (x + y - 1)^2 ≥ 1 :=
by {
  intro x y,
  sorry
}

end min_value_of_expression_l261_261136


namespace expression_value_l261_261568

theorem expression_value (x y z w : ℝ) (h1 : x = -5) (h2 : y = 8) (h3 : z = 3) (h4 : w = 2) :
  Real.sqrt(2 * z * (w - y) ^ 2 - x ^ 3 * y) + Real.sin(Real.pi * z) * x * w ^ 2 - Real.tan(Real.pi * x ^ 2) * z ^ 3 = Real.sqrt 1216 :=
by
  -- This is where the proof would go
  sorry

end expression_value_l261_261568


namespace volume_difference_l261_261689

def sara_dimensions : (ℤ × ℤ × ℤ) := (1 * 12, 2 * 12, 2 * 12) -- dimensions in inches
def jake_dimensions : (ℤ × ℤ × ℤ) := (16, 20, 18) -- dimensions already in inches

def volume (dims : (ℤ × ℤ × ℤ)) : ℤ :=
  dims.1 * dims.2 * dims.3

theorem volume_difference :
  volume sara_dimensions - volume jake_dimensions = 1152 :=
by
  sorry

end volume_difference_l261_261689


namespace collection_bound_l261_261575

variables {S I X : Type} {t n : ℕ}
variable {A : Type}
variable {𝒜 : set (set A)}

def is_subset (𝒜 : set (set A)) (T : set A) : Prop := ∀ a ∈ 𝒜, a ⊆ T

theorem collection_bound (h₁ : is_subset 𝒜 S) (h₂ : ∀ A B ∈ 𝒜, A ∪ B ≠ X) :
  𝒜.to_finset.card ≤ choose (n - 1) (nat.floor (n / 2) - 1) :=
sorry

end collection_bound_l261_261575


namespace construct_triangle_l261_261890

-- Definitions for the given conditions
def midpoint (P Q R : Point) : Prop :=
  (P.x + Q.x = 2 * R.x) ∧ (P.y + Q.y = 2 * R.y)

def lies_on (P : Point) (l : Line) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

def bisector (A : Point) (B C : Point) (l : Line) : Prop :=
  ∃ P, lies_on P l ∧ ∃ S, (2 * S.x = A.x + P.x) ∧ (2 * S.y = A.y + P.y) ∧ midpoint A S B

-- The main theorem for proving the existence of triangle ABC
theorem construct_triangle
  (N M : Point) (l : Line)
  (hN_mid_AC : ∃ A C, midpoint N A C)
  (hM_mid_BC : ∃ B C, midpoint M B C)
  (hl_bisector : bisector A B C l) :
  ∃ A B C, midpoint N A C ∧ midpoint M B C ∧ bisector A B C l :=
sorry

end construct_triangle_l261_261890


namespace ellipse_equations_l261_261166

theorem ellipse_equations (x y : ℝ) (a b : ℝ) (h1 : a = 3 * b) (h2 : (3^2 / a^2) + (2^2 / b^2) = 1 ∨ (2^2 / a^2) + (3^2 / b^2) = 1) :
  ((a^2 = 45 ∧ b^2 = 5) ∨ (a^2 = 85 ∧ b^2 = 85 / 9)) →
  ((∃ h : (x^2 / 45) + (y^2 / 5) = 1) ∨ ∃ h : (y^2 / 85) + (x^2 / (85 / 9)) = 1) :=
by
  sorry

end ellipse_equations_l261_261166


namespace no_negative_exponents_l261_261990

theorem no_negative_exponents (a b c d e f : ℤ) :
  2^a + 2^b + 5^e = 3^c + 3^d + 5^f → 
  (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0) :=
by
  sorry

end no_negative_exponents_l261_261990


namespace gabby_fruit_problem_l261_261516

-- Definitions of variables and constants
def W : ℕ := 1
def x : ℕ := 12
def P : ℕ := W + x
def Pl : ℕ := 3 * P

-- Stating the problem conditions in Lean 4
theorem gabby_fruit_problem : 
  W + P + Pl = 53 → (P = W + 12) :=
by
  -- Proving the condition, inserting proof steps if desired
  intros h,
  have hW : W = 1 := rfl,
  have hP : P = W + x := rfl,
  have hPl : Pl = 3 * P := rfl,
  rw [hW, hP, hPl] at h,
  assumption

end gabby_fruit_problem_l261_261516


namespace first_candidate_valid_vote_percentage_l261_261229

theorem first_candidate_valid_vote_percentage (total_votes invalid_vote_percentage second_candidate_valid_votes : ℕ) 
  (h1 : total_votes = 7000) 
  (h2 : invalid_vote_percentage = 20) 
  (h3 : second_candidate_valid_votes = 2520) : 
  (let valid_votes := (total_votes * (100 - invalid_vote_percentage)) / 100 in 
  let first_candidate_votes := valid_votes - second_candidate_valid_votes in 
  (first_candidate_votes * 100 / valid_votes) = 55) := 
by
  sorry

end first_candidate_valid_vote_percentage_l261_261229


namespace actual_price_of_food_l261_261780

theorem actual_price_of_food (P : ℝ) (h : 1.32 * P = 132) : P = 100 := 
by
  sorry

end actual_price_of_food_l261_261780


namespace towel_area_decrease_l261_261084

theorem towel_area_decrease (L B : ℝ) :
  let A_original := L * B
  let L_new := 0.8 * L
  let B_new := 0.9 * B
  let A_new := L_new * B_new
  let percentage_decrease := ((A_original - A_new) / A_original) * 100
  percentage_decrease = 28 := 
by
  sorry

end towel_area_decrease_l261_261084


namespace new_avg_contribution_l261_261587

theorem new_avg_contribution (A : ℝ) 
  (h1 : 3 * A + 150 = 1.5 * A * 4) : 
  1.5 * A = 75 :=
by
  have hA : A = 50 := by linarith
  rw hA
  linarith

end new_avg_contribution_l261_261587


namespace distance_from_center_to_point_l261_261743

theorem distance_from_center_to_point (C O E1 E2 E3 : ℝ^3)
  (angle_C_E1_E2 : angle C E1 E2 = π / 3)
  (angle_C_E2_E3 : angle C E2 E3 = π / 3)
  (angle_C_E3_E1 : angle C E3 E1 = π / 3)
  (radius_O_E : dist O E1 = 1 ∧ dist O E2 = 1 ∧ dist O E3 = 1) :
  dist C O = √3 := 
sorry

end distance_from_center_to_point_l261_261743


namespace find_x_value_l261_261992

theorem find_x_value (x: Real) (h: sin (2 * x) * sin (4 * x) = cos (2 * x) * cos (4 * x)) : 
  x = (11.25 * Real.pi / 180) :=
sorry

end find_x_value_l261_261992


namespace watermelon_sales_correct_l261_261826

def total_watermelons_sold 
  (customers_one_melon : ℕ) 
  (customers_three_melons : ℕ) 
  (customers_two_melons : ℕ) : ℕ :=
  (customers_one_melon * 1) + (customers_three_melons * 3) + (customers_two_melons * 2)

theorem watermelon_sales_correct :
  total_watermelons_sold 17 3 10 = 46 := by
  sorry

end watermelon_sales_correct_l261_261826


namespace hourMinuteHands90DegreeAngleSecondTime_l261_261198

noncomputable def timeFor90DegreeAngleAfter4OClock : ℕ :=
  let X : ℕ := 38 in
  X

theorem hourMinuteHands90DegreeAngleSecondTime : timeFor90DegreeAngleAfter4OClock = 38 :=
sorry

end hourMinuteHands90DegreeAngleSecondTime_l261_261198


namespace arrange_PERCEPTION_l261_261899

theorem arrange_PERCEPTION : 
  let n := 10 
  let k_E := 2
  let k_P := 2
  let k_I := 2
  nat.factorial n / (nat.factorial k_E * nat.factorial k_P * nat.factorial k_I) = 453600 :=
by
  sorry

end arrange_PERCEPTION_l261_261899


namespace sum_of_first_100_odd_numbers_l261_261105

theorem sum_of_first_100_odd_numbers:
  (∑ i in Finset.range 100, (2 * i + 1)) = 10000 :=
sorry

end sum_of_first_100_odd_numbers_l261_261105


namespace birdhouse_volume_difference_l261_261687

-- Definitions to capture the given conditions
def sara_width_ft : ℝ := 1
def sara_height_ft : ℝ := 2
def sara_depth_ft : ℝ := 2

def jake_width_in : ℝ := 16
def jake_height_in : ℝ := 20
def jake_depth_in : ℝ := 18

-- Convert Sara's dimensions to inches
def ft_to_in (x : ℝ) : ℝ := x * 12
def sara_width_in := ft_to_in sara_width_ft
def sara_height_in := ft_to_in sara_height_ft
def sara_depth_in := ft_to_in sara_depth_ft

-- Volume calculations
def volume (width height depth : ℝ) := width * height * depth
def sara_volume := volume sara_width_in sara_height_in sara_depth_in
def jake_volume := volume jake_width_in jake_height_in jake_depth_in

-- The theorem to prove the difference in volume
theorem birdhouse_volume_difference : sara_volume - jake_volume = 1152 := by
  -- Proof goes here
  sorry

end birdhouse_volume_difference_l261_261687


namespace arithmetic_sequence_common_difference_l261_261963

theorem arithmetic_sequence_common_difference 
    (a_2 : ℕ → ℕ) (S_4 : ℕ) (a_n : ℕ → ℕ → ℕ) (S_n : ℕ → ℕ → ℕ → ℕ)
    (h1 : a_2 2 = 3) (h2 : S_4 = 16) 
    (h3 : ∀ n a_1 d, a_n a_1 n = a_1 + (n-1)*d)
    (h4 : ∀ n a_1 d, S_n n a_1 d = n / 2 * (2*a_1 + (n-1)*d)) : ∃ d, d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l261_261963


namespace inequality_selection_l261_261792

theorem inequality_selection (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) :
  4 * (a^3 + b^3) > (a + b)^3 := 
by sorry

end inequality_selection_l261_261792


namespace john_bought_three_puzzles_l261_261255

-- Define the conditions
def first_puzzle_pieces : ℕ := 1000
def second_and_third_puzzle_pieces : ℕ := first_puzzle_pieces + first_puzzle_pieces / 2
def total_pieces := first_puzzle_pieces + 2 * second_and_third_puzzle_pieces
def given_total_pieces : ℕ := 4000

-- State the theorem
theorem john_bought_three_puzzles :
  total_pieces = given_total_pieces →
  1 + 2 = 3 :=
by
  intro h
  have h1 : first_puzzle_pieces = 1000 := rfl
  have h2 : second_and_third_puzzle_pieces = 1500 := by
    unfold second_and_third_puzzle_pieces
    rw h1
    norm_num
  have h3 : total_pieces = 4000 := by
    unfold total_pieces
    rw [h1, h2]
    norm_num
  exact eq.trans h h3 ▸ rfl

end john_bought_three_puzzles_l261_261255


namespace blue_red_ratio_l261_261108

noncomputable def radius (diameter : ℝ) : ℝ := diameter / 2

noncomputable def area (radius : ℝ) : ℝ := Real.pi * radius^2

theorem blue_red_ratio :
  let red_diam := 2 in
  let blue_diam := 6 in
  let red_radius := radius red_diam in
  let blue_radius := radius blue_diam in
  let red_area := area red_radius in
  let blue_area := area blue_radius - red_area in
  (blue_area / red_area) = 8 := by
  let red_diam := 2
  let blue_diam := 6
  let red_radius := radius red_diam
  let blue_radius := radius blue_diam
  let red_area := area red_radius
  let blue_area := area blue_radius - red_area
  have : (blue_area / red_area) = 8 := by
    sorry
  exact this

end blue_red_ratio_l261_261108


namespace sum_of_squares_l261_261347

variables {A B C D E : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]

-- Define the points, triangle, and segments
def right_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
  (dist A B)^2 + (dist B C)^2 = (dist A C)^2

def divides_into_three_equal_parts (D E B C : Type) [MetricSpace D] [MetricSpace E] [MetricSpace B] [MetricSpace C] : Prop :=
  dist B D = dist D E ∧ dist E C = dist D E

-- Define the statement to be proven
theorem sum_of_squares (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
  (h1 : right_triangle A B C) 
  (h2 : divides_into_three_equal_parts D E B C) :
  (dist A D)^2 + (dist D E)^2 + (dist A E)^2 = (2 / 3) * (dist B C)^2 :=
sorry

end sum_of_squares_l261_261347


namespace length_of_c_area_bounded_by_c_l261_261455

noncomputable def curve_length (a b : ℝ) (x y : ℝ → ℝ) :=
  ∫ t in a..b, sqrt ((deriv x t)^2 + (deriv y t)^2) 

noncomputable def area_region (a b : ℝ) (x y : ℝ → ℝ) :=
  ∫ t in a..b, y t * deriv x t

noncomputable def x (t : ℝ) : ℝ := real.exp (-t) * real.cos t
noncomputable def y (t : ℝ) : ℝ := real.exp (-t) * real.sin t

theorem length_of_c : 
  curve_length 0 (real.pi / 2) x y = sqrt 2 * (1 - real.exp (- real.pi / 2)) := 
  sorry

theorem area_bounded_by_c : 
  area_region 0 (real.pi / 2) x y = (1 - real.exp (-real.pi)) / 4 := 
  sorry

end length_of_c_area_bounded_by_c_l261_261455


namespace smallest_positive_integer_divides_l261_261922

theorem smallest_positive_integer_divides (m : ℕ) : 
  (∀ z : ℂ, z ≠ 0 → (z^11 + z^10 + z^8 + z^7 + z^5 + z^4 + z^2 + 1) ∣ (z^m - 1)) →
  (m = 88) :=
sorry

end smallest_positive_integer_divides_l261_261922


namespace fraction_of_time_riding_at_15mph_l261_261250

variable (t_5 t_15 : ℝ)

-- Conditions
def no_stops : Prop := (t_5 ≠ 0 ∧ t_15 ≠ 0)
def average_speed (t_5 t_15 : ℝ) : Prop := (5 * t_5 + 15 * t_15) / (t_5 + t_15) = 10

-- Question to be proved
theorem fraction_of_time_riding_at_15mph (h1 : no_stops t_5 t_15) (h2 : average_speed t_5 t_15) :
  t_15 / (t_5 + t_15) = 1 / 2 :=
sorry

end fraction_of_time_riding_at_15mph_l261_261250


namespace workers_together_time_l261_261786

theorem workers_together_time (hA : ∀ (job : ℕ), job = 1 → ∃ t : ℕ, t = 7)
                             (hB : ∀ (job : ℕ), job = 1 → ∃ t : ℕ, t = 10) :
  let t := (70 : ℝ) / (17 : ℝ) in
  t = 70 / 17 := 
by
  sorry

end workers_together_time_l261_261786


namespace f_evaluation_l261_261968

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≥ 4 then (1 / 2) ^ x else f (x + 1)

theorem f_evaluation : f (1 + log 2 5) = 1 / 20 := by
  sorry

end f_evaluation_l261_261968


namespace rooks_on_checkerboard_l261_261331

theorem rooks_on_checkerboard : ∃ n : ℕ, n = 2880 ∧ ∀ (board : fin 9 × fin 9 → Prop), 
    (∀ r1 r2 c1 c2 : fin 9, r1 ≠ r2 → c1 ≠ c2 → board ⟨r1, c1⟩ → board ⟨r2, c2⟩ → false) ↔ 
    (board (0,0) ∨ board (0,1) ∨ board (1,0) ∨ board (1,1) → 
    ∃ f : fin 9 → fin 9, function.injective f ∧ ∀ i, board ⟨i, f i⟩) := 
begin
  use 2880,
  split,
  { refl, },
  sorry
end

end rooks_on_checkerboard_l261_261331


namespace simplify_expression_l261_261273

variable {R : Type} [LinearOrderedField R]

theorem simplify_expression (x y z : R) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) (h_sum : x + y + z = 3) :
  (1 / (y^2 + z^2 - x^2) + 1 / (x^2 + z^2 - y^2) + 1 / (x^2 + y^2 - z^2)) =
    3 / (-9 + 6 * y + 6 * z - 2 * y * z) :=
  sorry

end simplify_expression_l261_261273


namespace max_value_k_l261_261631

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ (∀ n : ℕ, (n ≥ 1) → (a (n+1) = if n % 3 = 0 then a (n-1) else a n + 5))

noncomputable def max_k (a : ℕ → ℕ) (N : ℕ) : ℕ :=
  if a N ≤ 2021 then N else max_k a (N-1)

theorem max_value_k (a : ℕ → ℕ) (N : ℕ) (h : sequence a) :
  max_k a 1211 = 1211 := 
sorry

end max_value_k_l261_261631


namespace average_speed_interval_l261_261078

theorem average_speed_interval {s t : ℝ → ℝ} (h_eq : ∀ t, s t = t^2 + 1) : 
  (s 2 - s 1) / (2 - 1) = 3 :=
by
  sorry

end average_speed_interval_l261_261078


namespace proportionate_enlargement_l261_261103

theorem proportionate_enlargement 
  (original_width original_height new_width : ℕ)
  (h_orig_width : original_width = 3)
  (h_orig_height : original_height = 2)
  (h_new_width : new_width = 12) : 
  ∃ (new_height : ℕ), new_height = 8 :=
by
  -- sorry to skip proof
  sorry

end proportionate_enlargement_l261_261103


namespace perpendicular_relationship_l261_261170

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Definitions used in conditions
variables (a b : V)
axiom lengths_equal (h : ∥a∥ = ∥b∥)
axiom non_collinear (h₁ : ¬∃ k : ℝ, a = k • b)

-- Goal: to prove the relationship
theorem perpendicular_relationship (h : ∥a∥ = ∥b∥) (h₁ : ¬∃ k : ℝ, a = k • b) :
  ⟪a + b, a - b⟫ = 0 :=
sorry

end perpendicular_relationship_l261_261170


namespace birdhouse_volume_difference_l261_261684

theorem birdhouse_volume_difference :
  let sara_width := 1
  let sara_height := 2
  let sara_depth := 2
  let jake_width := 16 / 12
  let jake_height := 20 / 12
  let jake_depth := 18 / 12
  let sara_volume := sara_width * sara_height * sara_depth
  let jake_volume := jake_width * jake_height * jake_depth
  let volume_difference := sara_volume - jake_volume
  volume_difference ≈ 0.668 :=
by
  sorry

end birdhouse_volume_difference_l261_261684


namespace volume_of_inscribed_sphere_l261_261070

theorem volume_of_inscribed_sphere (perimeter : ℝ) (h : perimeter = 28) : 
  ∃ (V : ℝ), V ≈ 928.318 ∧ V = (4/3) * Real.pi * (perimeter / 4 * Real.sqrt 3 / 2)^3 := 
by 
  sorry

end volume_of_inscribed_sphere_l261_261070


namespace sum_of_coefficients_binomial_expansion_l261_261379

theorem sum_of_coefficients_binomial_expansion :
  (Finset.range 9).sum (λ k, Nat.choose 8 k) = 256 := 
by
  sorry

end sum_of_coefficients_binomial_expansion_l261_261379


namespace height_of_brick_l261_261796

-- Definitions of wall dimensions
def L_w : ℝ := 700
def W_w : ℝ := 600
def H_w : ℝ := 22.5

-- Number of bricks
def n : ℝ := 5600

-- Definitions of brick dimensions (length and width)
def L_b : ℝ := 25
def W_b : ℝ := 11.25

-- Main theorem: Prove the height of each brick
theorem height_of_brick : ∃ h : ℝ, h = 6 :=
by
  -- Will add the proof steps here eventually
  sorry

end height_of_brick_l261_261796


namespace retailer_profit_percentage_l261_261468

theorem retailer_profit_percentage (items_sold : ℕ) (profit_per_item : ℝ) (discount_rate : ℝ)
  (discounted_items_needed : ℝ) (total_profit : ℝ) (item_cost : ℝ) :
  items_sold = 100 → 
  profit_per_item = 30 →
  discount_rate = 0.05 →
  discounted_items_needed = 156.86274509803923 →
  total_profit = 3000 →
  (discounted_items_needed * ((item_cost + profit_per_item) * (1 - discount_rate) - item_cost) = total_profit) →
  ((profit_per_item / item_cost) * 100 = 16) :=
by {
  sorry 
}

end retailer_profit_percentage_l261_261468


namespace find_constants_l261_261490

theorem find_constants :
  ∃ (A B C : ℝ), (∀ x : ℝ, x ≠ 3 → x ≠ 4 → 
  (6 * x / ((x - 4) * (x - 3) ^ 2)) = (A / (x - 4) + B / (x - 3) + C / (x - 3) ^ 2)) ∧
  A = 24 ∧
  B = - 162 / 7 ∧
  C = - 18 :=
by
  use 24, -162 / 7, -18
  sorry

end find_constants_l261_261490


namespace gasoline_reduction_l261_261215

-- Assuming the price and quantity are real numbers.
variables (P Q : ℝ)

-- Conditions of the problem
def price_increase := 1.25 * P
def new_spend := 1.10 * (P * Q)

-- Calculation based on the conditions
def new_quantity := (new_spend / price_increase)

-- Expected reduction in percentage
def reduction_percentage := 1 - (new_quantity / Q)

-- The theorem we want to prove:
theorem gasoline_reduction : reduction_percentage P Q = 0.12 := 
by
  sorry

end gasoline_reduction_l261_261215


namespace arina_sophia_divisible_l261_261835

theorem arina_sophia_divisible (N: ℕ) (k: ℕ) (large_seq: list ℕ): 
  (k = 81) → 
  (large_seq = (list.range' 71 (k + 1)).append (list.range' 82 (N + 1))) → 
  (nat.digits 10 (list.foldl (λ n d, 10 * n + d) 0 large_seq)).nat_sum % 3 = 0 → 
  (nat.digits 10 (list.foldl (λ n d, 10 * n + d) 0 large_seq)).last_digits 2 % 4 = 0 →
  (list.foldl (λ n d, 10 * n + d) 0 large_seq) % 12 = 0 → 
  N = 84 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end arina_sophia_divisible_l261_261835


namespace birdhouse_volume_difference_l261_261685

theorem birdhouse_volume_difference :
  let sara_width := 1
  let sara_height := 2
  let sara_depth := 2
  let jake_width := 16 / 12
  let jake_height := 20 / 12
  let jake_depth := 18 / 12
  let sara_volume := sara_width * sara_height * sara_depth
  let jake_volume := jake_width * jake_height * jake_depth
  let volume_difference := sara_volume - jake_volume
  volume_difference ≈ 0.668 :=
by
  sorry

end birdhouse_volume_difference_l261_261685


namespace min_sequence_fraction_l261_261944

theorem min_sequence_fraction (a : ℕ → ℕ)
  (h₀ : a 1 = 15)
  (h₁ : ∀ n, (a (n + 1) - a n) / n = 2) :
  ∃ m ∈ (set.range (λ n, n + 15 / n - 1)), m = 27 / 4 :=
sorry

end min_sequence_fraction_l261_261944


namespace units_digit_of_30_factorial_is_0_l261_261023

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_30_factorial_is_0 : units_digit (factorial 30) = 0 := by
  sorry

end units_digit_of_30_factorial_is_0_l261_261023


namespace largest_subset_no_three_divisors_or_multiples_l261_261373

def max_subset_size (n : ℕ) : ℕ :=
  if n % 2 = 0 then n + 1 else n

theorem largest_subset_no_three_divisors_or_multiples 
  (S : Finset ℕ) (h₁ : ∀ x ∈ S, x ∈ (Finset.range 2014)) 
  (h₂ : ∀ a b c ∈ S, a ≠ b → b ≠ c → c ≠ a → ¬(a ∣ (b - c) ∨ (b - c) ∣ a)) :
  S.card ≤ 672 :=
sorry

end largest_subset_no_three_divisors_or_multiples_l261_261373


namespace directrix_of_parabola_l261_261498

def parabola_eq (x : ℝ) : ℝ := -4 * x^2 + 4

theorem directrix_of_parabola : 
  ∃ y : ℝ, y = 65 / 16 :=
by
  sorry

end directrix_of_parabola_l261_261498


namespace find_tangent_line_to_curves_l261_261709

noncomputable def tangent_line_to_curves_tangent (t : ℝ) : Prop :=
  let f : ℝ → ℝ := λ x, Real.exp x
  let g : ℝ → ℝ := λ x, -x ^ 2 / 4
  let tangent_line (t : ℝ) (x : ℝ) : ℝ := Real.exp t * (x - t) + Real.exp t
  in
  (e^t + t - 1 = 0) ∧
  (∀ x, tangent_line t x = -x ^ 2 / 4 → y = x + 1)

theorem find_tangent_line_to_curves : ∃ t, tangent_line_to_curves_tangent t := sorry

end find_tangent_line_to_curves_l261_261709


namespace acid_solution_problem_l261_261744

theorem acid_solution_problem (n : ℕ) (h : n > 30) : 
  let y := 15 * n / (n + 35)
  let initial_acid := n * n / 100
  let added_acid := y / 5
  let total_solution := n + y
  let final_acid := (n - 15) * total_solution / 100
  in initial_acid + added_acid = final_acid
:=
sorry

end acid_solution_problem_l261_261744


namespace probability_closer_to_6_than_0_l261_261420

-- Defining the probability problem
theorem probability_closer_to_6_than_0 :
  let interval := set.Icc 0 7 in
  (∀ point : ℝ, point ∈ interval → point > 3) → 
  (measure_theory.measure_space.measure (set.Icc 3 7) / 
   measure_theory.measure_space.measure interval = 4 / 7) :=
by
  sorry

end probability_closer_to_6_than_0_l261_261420


namespace generalized_triangle_theorem_l261_261610

noncomputable def incircle_relation (a b c : ℝ) (α β γ : ℝ) (rho : ℝ) : Prop :=
  (a + b - c = 2 * rho * Real.cot (γ / 2)) ∧
  (a + c - b = 2 * rho * Real.cot (β / 2)) ∧
  (b + c - a = 2 * rho * Real.cot (α / 2))

theorem generalized_triangle_theorem 
  (a b c : ℝ) (α β γ : ℝ) (rho : ℝ)
  (triangle : 0 < α ∧ α < π)
  (incircle_tangent_points : 0 < beta ∧ β < π)
  (radius_positive : 0 < rho) :
  incircle_relation a b c α β γ rho := by
  sorry

end generalized_triangle_theorem_l261_261610


namespace angle_BCM_eq_angle_ABD_l261_261300

/- Define the objects: points, quadrilateral, angles, and midpoints. -/
variables {A B C D M : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] [AddGroup M]

/- Define midpoint -/
def is_midpoint (M : Type) (A B : Type) := ∀ (P : Type), P = M ↔ P = (A + B) / 2

/- Define angles -/
def angle (A B C : Type) := ∀ (P Q R : Type), ∃ θ : ℝ, θ = ∠BAC

/- The given conditions translated to Lean definitions. -/
variables (AB_midpoint : is_midpoint M A B)
variables (angle_BCA_eq_angle_ACD : angle B C A = angle A C D)
variables (angle_CDA_eq_angle_BAC : angle C D A = angle B A C)

/- Proof statement: Prove that angle BCM equals angle ABD given the conditions. -/
theorem angle_BCM_eq_angle_ABD
    (h1 : is_midpoint M A B)
    (h2 : angle B C A = angle A C D)
    (h3 : angle C D A = angle B A C)
    : angle B C M = angle A B D :=
sorry

end angle_BCM_eq_angle_ABD_l261_261300


namespace div_by_7_or_11_l261_261303

theorem div_by_7_or_11 (z x y : ℕ) (hx : x < 1000) (hz : z = 1000 * y + x) (hdiv7 : (x - y) % 7 = 0 ∨ (x - y) % 11 = 0) :
  z % 7 = 0 ∨ z % 11 = 0 :=
by
  sorry

end div_by_7_or_11_l261_261303


namespace trapezoid_crop_fraction_l261_261087

-- Define the trapezoid with given lengths and angles
variables (AB CD AD BC : ℝ) (angleA angleB : ℝ)
def is_trapezoid : Prop :=
  (AB = 100) ∧ (CD = 100) ∧ (BC = 150) ∧ (AD = 200) ∧ (angleA = 75) ∧ (angleB = 75)

-- Define the theorem to calculate the fraction of crop closer to AB
theorem trapezoid_crop_fraction (h : is_trapezoid AB CD AD BC angleA angleB) : 
  ∃ fraction : ℝ, (0 ≤ fraction) ∧ (fraction ≤ 1) ∧ (calculate_fraction AB CD AD BC angleA angleB = fraction) := by
  sorry

end trapezoid_crop_fraction_l261_261087


namespace nine_rooks_checkerboard_l261_261328

theorem nine_rooks_checkerboard :
  let num_ways_4x4 := 4.factorial
  let num_ways_5x5 := 5.factorial
  num_ways_4x4 * num_ways_5x5 = 2880 :=
by
  sorry

end nine_rooks_checkerboard_l261_261328


namespace division_sqrt_400_eq_10_l261_261026

theorem division_sqrt_400_eq_10 (x : ℝ) : sqrt 400 / x = 10 → x = 2 :=
by
  sorry

end division_sqrt_400_eq_10_l261_261026


namespace construct_triangle_given_midpoints_and_bisector_line_l261_261888

-- Define the midpoints as N and M and the line l
variables {A B C N M : Type} (l : Type)

-- Assume that N is the midpoint of AC
def is_midpoint_AC (N : Type) (A C : Type) : Prop := 
  ∃ N, N = (A + C) / 2

-- Assume that M is the midpoint of BC
def is_midpoint_BC (M : Type) (B C : Type) : Prop := 
  ∃ M, M = (B + C) / 2

-- Assume the bisector of angle A lies on line l
def bisector_angle_A (A : Type) (l : Type) : Prop := 
  ∃ A l, is_angle_bisector A l

-- Main theorem
theorem construct_triangle_given_midpoints_and_bisector_line 
  (N M A B C : Type) (l : Type) 
  (h1 : is_midpoint_AC N A C)
  (h2 : is_midpoint_BC M B C)
  (h3 : bisector_angle_A A l) : 
  ∃ (triangle : Type), is_triangle ABC := 
sorry

end construct_triangle_given_midpoints_and_bisector_line_l261_261888


namespace area_of_smaller_triangle_leq_l261_261508

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_of_smaller_triangle_leq
  (a1 b1 c1 a2 b2 c2 : ℝ)
  (A1 A2 : ℝ)
  (h1 : a1 ≤ a2)
  (h2 : b1 ≤ b2)
  (h3 : c1 ≤ c2)
  (h_trig1 : A1 = triangle_area a1 b1 c1)
  (h_trig2 : A2 = triangle_area a2 b2 c2)
  (h_acute : c2^2 < a2^2 + b2^2) : A1 ≤ A2 :=
  sorry

end area_of_smaller_triangle_leq_l261_261508


namespace animals_complete_task_l261_261438

theorem animals_complete_task (x : ℝ) : 
  let monkeys := 239
      termites := 622
      ants := 345
      termite_speed := 1
      monkey_speed := 1.5 * termite_speed
      ant_speed := 0.75 * termite_speed
      termite_work_per_hour := termite_speed
      monkey_work_per_hour := monkey_speed
      ant_work_per_hour := ant_speed
      termites_total_work := termites * termite_work_per_hour
      monkeys_total_work := monkeys * monkey_work_per_hour
      ants_total_work := ants * ant_work_per_hour
      total_work_per_hour := termites_total_work + monkeys_total_work + ants_total_work in
  total_work_per_hour = 1239.25 * termite_speed ∧
  (Wx / total_work_per_hour) = x / 1239.25 :=
begin
  sorry
end

end animals_complete_task_l261_261438


namespace sin_cos_sum_l261_261177

theorem sin_cos_sum (α : ℝ) (h : ∃ x y : ℝ, (x^2 + y^2 = 169) ∧ (sin α = y / sqrt (x^2 + y^2)) ∧ (cos α = x / sqrt (x^2 + y^2))) 
                     (hx : x = 5) (hy : y = -12) : 
                     (sin α + cos α) = - (7 / 13) := 
by
  sorry

end sin_cos_sum_l261_261177


namespace final_state_probability_l261_261609

-- Define the initial state and conditions of the problem
structure GameState where
  raashan : ℕ
  sylvia : ℕ
  ted : ℕ
  uma : ℕ

-- Conditions: each player starts with $2, and the game evolves over 500 rounds
def initial_state : GameState :=
  { raashan := 2, sylvia := 2, ted := 2, uma := 2 }

def valid_statements (state : GameState) : Prop :=
  state.raashan = 2 ∧ state.sylvia = 2 ∧ state.ted = 2 ∧ state.uma = 2

-- Final theorem statement
theorem final_state_probability :
  let states := 500 -- representing the number of rounds
  -- proof outline implies that after the games have properly transitioned and bank interactions, the probability is calculated
  -- state after the transitions
  ∃ (prob : ℚ), prob = 1/4 ∧ valid_statements initial_state :=
  sorry

end final_state_probability_l261_261609


namespace max_value_of_c_l261_261597

theorem max_value_of_c (a b c: ℝ) (h1: 2 ^ a + 2 ^ b = 2 ^ (a + b)) (h2: 2 ^ a + 2 ^ b + 2 ^ c = 2 ^ (a + b + c)) : 
c ≤ 2 - Real.log2 3 :=
sorry

end max_value_of_c_l261_261597


namespace probability_right_of_y_axis_half_l261_261674

structure Point2D := 
  (x : ℝ)
  (y : ℝ)

def P : Point2D := ⟨-4, 4⟩
def Q : Point2D := ⟨2, -2⟩
def R : Point2D := ⟨4, -2⟩
def S : Point2D := ⟨-2, 4⟩

def isParallelogram (P Q R S : Point2D) : Prop :=
  -- Definition of a parallelogram
  (P.x + R.x = Q.x + S.x) ∧ (P.y + R.y = Q.y + S.y)

def isRightOfYAxis (point : Point2D) : Prop :=
  point.x > 0

noncomputable def areaRightOfYAxis (P Q R S : Point2D) : ℝ :=
  -- Assuming function calculates the area to the right of the y-axis of a parallelogram
  sorry

theorem probability_right_of_y_axis_half : 
  isParallelogram P Q R S → 
  (areaRightOfYAxis P Q R S / areaOfParallelogram P Q R S) = 1/2 :=
sorry

end probability_right_of_y_axis_half_l261_261674


namespace factory_toys_produced_per_week_l261_261411

axiom toys_produced_per_day : ℕ
axiom days_working_per_week : ℕ

def toys_produced_per_week := toys_produced_per_day * days_working_per_week

theorem factory_toys_produced_per_week : 
  toys_produced_per_day = 2000 ∧ days_working_per_week = 4 → toys_produced_per_week = 8000 := 
by
  intro h,
  cases h with h1 h2,
  unfold toys_produced_per_week,
  rw [h1, h2],
  norm_num,
  done

end factory_toys_produced_per_week_l261_261411


namespace items_left_in_store_l261_261875

theorem items_left_in_store: (4458 - 1561) + 575 = 3472 :=
by 
  sorry

end items_left_in_store_l261_261875


namespace find_integer_solution_of_equations_l261_261895

theorem find_integer_solution_of_equations :
  ∃ (s : Finset (ℤ × ℤ × ℤ)),
    s = {⟨3, 8, 5⟩, ⟨8, 3, 5⟩, ⟨3, -5, -8⟩, ⟨-5, 8, -3⟩, ⟨-5, 3, -8⟩, ⟨8, -5, -3⟩} ∧
    ∀ (x y z : ℤ), 
    (⟨x, y, z⟩ ∈ s ↔ x + y - z = 6 ∧ x^3 + y^3 - z^3 = 414) := by
  sorry

end find_integer_solution_of_equations_l261_261895


namespace find_k_max_triangle_area_l261_261965

-- Definitions required for the context
def ellipse_eq (x y : ℝ) := (x^2 / 3) + y^2 = 1
def line_eq (k m x y: ℝ) := y = k * x + m
def dot_product_zero (x1 y1 x2 y2 : ℝ) := x1 * x2 + y1 * y2 = 0
def dist_origin_line (k m : ℝ) := abs m / real.sqrt (1 + k^2) = real.sqrt 3 / 2
def area_triangle (x1 y1 x2 y2 : ℝ) := 1 / 2 * abs (x1 * y2 - y1 * x2)

-- Problem 1: Prove for given m = 1 and dot product of OA and OB is zero, k must be sqrt(3)/3 or -sqrt(3)/3.
theorem find_k (k x1 x2 y1 y2 : ℝ) (h1 : line_eq k 1 x1 y1) (h2 : line_eq k 1 x2 y2) 
  (h3 : ellipse_eq x1 y1) (h4 : ellipse_eq x2 y2) (h5 : dot_product_zero x1 y1 x2 y2) : 
  k = real.sqrt(3) / 3 ∨ k = -real.sqrt(3) / 3 := sorry

-- Problem 2: Prove given the distance from the origin to the line is sqrt(3)/2, the max area of triangle AOB is sqrt(3)/2.
theorem max_triangle_area (k m x1 x2 y1 y2 : ℝ) (h1 : dist_origin_line k m)
  (h2 : line_eq k m x1 y1) (h3 : line_eq k m x2 y2) 
  (h4 : ellipse_eq x1 y1) (h5 : ellipse_eq x2 y2) : 
  area_triangle x1 y1 x2 y2 = sqrt(3) / 2 := sorry

end find_k_max_triangle_area_l261_261965


namespace tenth_term_l261_261613

-- Define the conditions
variables {a d : ℤ}

-- The conditions of the problem
axiom third_term_condition : a + 2 * d = 10
axiom sixth_term_condition : a + 5 * d = 16

-- The goal is to prove the tenth term
theorem tenth_term : a + 9 * d = 24 :=
by
  sorry

end tenth_term_l261_261613


namespace minimum_distance_exists_l261_261748

theorem minimum_distance_exists : ∃ a b : ℤ, (dist (2019, 470) (21 * (a : ℤ) - 19 * (b : ℤ), 19 * (b : ℤ) + 21 * (a : ℤ)) = Real.sqrt 101) := sorry

end minimum_distance_exists_l261_261748


namespace smallest_possible_union_size_l261_261306

theorem smallest_possible_union_size {X Y : Type} (hX : Fintype.card X = 30)
    (hY : Fintype.card Y = 25)
    (hXY_min : 10 ≤ HasInter.inter X Y)
    (hXY_max : HasInter.inter X Y ≤ 20) : 
    ∃ n, n = 35 ∧ Fintype.card (X ∪ Y) = n := 
by 
    sorry

end smallest_possible_union_size_l261_261306


namespace inverse_sine_function_l261_261721

theorem inverse_sine_function :
  ∀ x : ℝ, x ∈ set.Icc (-(Real.pi / 2)) (Real.pi / 2) → ∀ y : ℝ, y = Real.sin x →
  (x ∈ set.Icc (-1) 1 → Real.arcsin y = x) :=
by
  sorry

end inverse_sine_function_l261_261721


namespace find_question_mark_l261_261404

theorem find_question_mark : ∃ X : ℤ, 27474 + X + 1985 - 2047 = 31111 ∧ X = 3699 :=
begin
  sorry
end

end find_question_mark_l261_261404


namespace max_chord_length_line_eq_orthogonal_vectors_line_eq_l261_261155

-- Definitions
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4
def point_P (x y : ℝ) : Prop := x = 2 ∧ y = 1
def line_eq (slope intercept x y : ℝ) : Prop := y = slope * x + intercept

-- Problem 1: Prove the equation of line l that maximizes the length of chord AB
theorem max_chord_length_line_eq : 
  (∀ x y : ℝ, point_P x y → line_eq 1 (-1) x y)
  ∧ (∀ x y : ℝ, circle_eq x y → point_P x y) 
  → (∀ x y : ℝ, line_eq 1 (-1) x y) :=
by sorry

-- Problem 2: Prove the equation of line l given orthogonality condition of vectors
theorem orthogonal_vectors_line_eq : 
  (∀ x y : ℝ, point_P x y → line_eq (-1) 3 x y)
  ∧ (∀ x y : ℝ, circle_eq x y → point_P x y) 
  → (∀ x y : ℝ, line_eq (-1) 3 x y) :=
by sorry

end max_chord_length_line_eq_orthogonal_vectors_line_eq_l261_261155


namespace locus_of_points_M_l261_261159

variables {A B C D M E F : Point}

-- Given conditions
def is_not_parallelogram (A B C D : Point) : Prop := sorry -- A formal definition would be required

def directed_area (X Y Z : Point) : ℝ := sorry -- Directed area of triangle XYZ in Euclidean space

-- Given quadrilateral ABCD which is not a parallelogram
axiom ABCD_not_parallelogram : is_not_parallelogram A B C D

-- Midpoints of diagonals
def midpoint (X Y : Point) : Point := sorry -- Definition for midpoint

axiom E_midpoint : E = midpoint A C
axiom F_midpoint : F = midpoint B D

-- Question to prove
theorem locus_of_points_M :
  (directed_area A M B + directed_area C D M = directed_area B C M + directed_area D A M) →
  lies_on_line M E F :=
sorry

end locus_of_points_M_l261_261159


namespace evaluate_sum_l261_261174

noncomputable def f : ℝ → ℝ := sorry

axiom f_add_mul (a b : ℝ) : f (a + b) = f a * f b
axiom f_one : f 1 = 2

theorem evaluate_sum : ∑ k in Finset.range 1008, f (2 * (k + 1)) / f (2 * (k + 1) - 1) = 2016 := 
by
  sorry

end evaluate_sum_l261_261174


namespace find_f_neg_half_l261_261938

def odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f (x : ℝ) : ℝ :=
  if x > 0 then 4^x else -1 -- place-holder for undefined values

theorem find_f_neg_half : (∀ x, x > 0 → f x = 4^x) → odd f → f (-1/2) = -2 :=
by
  intro pos_def odd_def
  sorry

end find_f_neg_half_l261_261938


namespace find_unknown_number_l261_261703

theorem find_unknown_number (x : ℕ) :
  (x + 30 + 50) / 3 = ((20 + 40 + 6) / 3 + 8) → x = 10 := by
    sorry

end find_unknown_number_l261_261703


namespace part_a_l261_261055

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem part_a :
  ∀ (N : ℕ), (N = (sum_of_digits N) ^ 2) → (N = 1 ∨ N = 81) :=
by
  intros N h
  sorry

end part_a_l261_261055


namespace perfect_rectangle_squares_l261_261209

noncomputable def side_lengths : List ℕ := [2, 5, 7, 25, 28, 33]

theorem perfect_rectangle_squares (C D A B E F : ℕ) 
  (hC : C = 16 - 9)
  (hD : D = 16 + 9)
  (hA : A = 9 - 7)
  (hB : B = 7 - 2)
  (hE : E = 5 + 7 + 16)
  (hF : F = 28 + 5) :
  List ℕ = [2, 5, 7, 25, 28, 33] :=
sorry

end perfect_rectangle_squares_l261_261209


namespace determine_quadrants_l261_261167

variable {α : ℝ}

theorem determine_quadrants (h : sin α + cos α = (2 * real.sqrt 6) / 5) :
  (π / 2 < α ∧ α < π) ∨ (3 * π / 2 < α ∧ α < 2 * π) :=
by
  sorry

end determine_quadrants_l261_261167


namespace divide_squares_into_equal_piles_l261_261047

theorem divide_squares_into_equal_piles :
  ∃ (piles : list (list ℕ)), list.nodup piles ∧
  list.join piles = (list.range 81).map (λ n, (n + 1)^2) ∧
  list.all piles (λ pile, list.sum pile = list.sum ((list.range 81).map (λ n, (n + 1)^2)) / 3) := by
sorry

end divide_squares_into_equal_piles_l261_261047


namespace job_fair_problem_l261_261005

theorem job_fair_problem :
  (A_hired_prob = 4/9) →
  (B_hired_prob = t/3) →
  (C_hired_prob = t/3) →
  (0 < t ∧ t < 3) →
  (independent_events : independent [A_hired_prob, B_hired_prob, C_hired_prob]) →
  (all_hired_prob = 16/81) →
  (t = 2) ∧
  (let prob_A_hired := 4 / 9 in
   let prob_B_hired := 2 / 3 in
   let prob_0_hired := (5 / 9) * (1 / 3) in
   let prob_1_hired := (4 / 9) * (1 / 3) + (5 / 9) * (2 / 3) in
   let prob_2_hired := (4 / 9) * (2 / 3) in
   let E_ξ := 2 * prob_2_hired + 1 * prob_1_hired + 0 * prob_0_hired in
   E_ξ = 10 / 9) →
sorry

end job_fair_problem_l261_261005


namespace vessel_reaches_boat_in_shortest_time_l261_261089

-- Define the given conditions as hypotheses
variable (dist_AC : ℝ) (angle_C : ℝ) (speed_CB : ℝ) (angle_B : ℝ) (speed_A : ℝ)

-- Assign values to variables based on the problem statement
def vessel_distress_boat_condition : Prop :=
  dist_AC = 10 ∧ angle_C = 45 ∧ speed_CB = 9 ∧ angle_B = 105 ∧ speed_A = 21

-- Define the time (in minutes) for the vessel to reach the fishing boat
noncomputable def shortest_time_to_reach_boat : ℝ :=
  25

-- The theorem that we need to prove given the conditions
theorem vessel_reaches_boat_in_shortest_time :
  vessel_distress_boat_condition dist_AC angle_C speed_CB angle_B speed_A → 
  shortest_time_to_reach_boat = 25 := by
    intros
    sorry

end vessel_reaches_boat_in_shortest_time_l261_261089


namespace construct_triangle_l261_261889

-- Definitions for the given conditions
def midpoint (P Q R : Point) : Prop :=
  (P.x + Q.x = 2 * R.x) ∧ (P.y + Q.y = 2 * R.y)

def lies_on (P : Point) (l : Line) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

def bisector (A : Point) (B C : Point) (l : Line) : Prop :=
  ∃ P, lies_on P l ∧ ∃ S, (2 * S.x = A.x + P.x) ∧ (2 * S.y = A.y + P.y) ∧ midpoint A S B

-- The main theorem for proving the existence of triangle ABC
theorem construct_triangle
  (N M : Point) (l : Line)
  (hN_mid_AC : ∃ A C, midpoint N A C)
  (hM_mid_BC : ∃ B C, midpoint M B C)
  (hl_bisector : bisector A B C l) :
  ∃ A B C, midpoint N A C ∧ midpoint M B C ∧ bisector A B C l :=
sorry

end construct_triangle_l261_261889


namespace fifth_and_sixth_equations_sum_first_n_terms_specific_series_sum_l261_261671

theorem fifth_and_sixth_equations :
  (a_5 = (1 : ℝ)/(9 * 11) ∧ a_5 = (1 / 2) * ((1 : ℝ)/9 - (1 : ℝ)/11)) ∧
  (a_6 = (1 : ℝ)/(11 * 13) ∧ a_6 = (1 / 2) * ((1 : ℝ)/11 - (1 : ℝ)/13)) :=
sorry

theorem sum_first_n_terms (n : ℕ) : 
  (\biggΣ i in range n, (\frac{1}{(2 * i + 1) * (2 * i + 3)}) = \frac{n}{2 * n + 1}) :=
sorry

theorem specific_series_sum (n : ℕ) :
  (\biggΣ i in range (n/3), (\frac{1}{(i - 2) * i})) = (\frac{n-1}{3 * n}) :=
sorry

end fifth_and_sixth_equations_sum_first_n_terms_specific_series_sum_l261_261671


namespace sum_of_reciprocal_gp_l261_261276

theorem sum_of_reciprocal_gp (n : ℕ) :
  let a := 2
  let r := -2
  let original_gp_sum := (a * (1 - r^n)) / (1 - r)
  let reciprocal_gp_sum := ((1 : ℚ) / a) * (1 - ((1 : ℚ) / r)^n) / (1 - ((1 : ℚ) / r))
  in reciprocal_gp_sum = (1 - (-1)^n * (1/(2^n))) / 3 := 
by
  sorry

end sum_of_reciprocal_gp_l261_261276


namespace rabbit_position_after_100_jumps_l261_261018

-- Define the conditions and sequences in Lean
def jump_distance : ℕ → ℕ := λ n, 20 * (n + 1)  -- distance for the n-th sequence of jumps
def jump_direction : ℕ → ℤ := λ n, if n % 2 = 0 then 1 else -1 -- +1 for south (even), -1 for north (odd)

-- Summing the position after each jump sequence
def total_displacement : ℕ → ℤ :=
  λ n, ∑ i in range (n + 1), jump_direction i * jump_distance i

-- We need to prove that total_displacement 99 / 100 = -10 meters (total_displacement 99 == -1000)
theorem rabbit_position_after_100_jumps : total_displacement 99 = -1000 :=
by sorry

end rabbit_position_after_100_jumps_l261_261018


namespace no_le_2mo_l261_261363

variables {A B C O M N : Type*} [ordered_field A B C O M N]
variables {distance : A × A → ℝ}

-- Supposing A, B, and C are points forming a triangle,
-- and O is the centroid of this triangle,
-- and M lies on AB and N lies on AC such that a line passes through these points.

def centroid_property : Prop :=
∀ (t : triangle A B C) (o : point O), o = centroid t

theorem no_le_2mo {A B C O M N : Type*} [ordered_field A B C O M N]
  (h1 : centroid_property ABC O)
  (h2 : M ∈ line A B)
  (h3 : N ∈ line A C)
  (h4 : collinear {O, M, N}) :
  distance N O ≤ 2 * distance M O :=
by {
  sorry
}

end no_le_2mo_l261_261363


namespace set_union_example_l261_261983

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem set_union_example (a b : ℝ) (h₁ : {3, log2 a} ∩ {a, b} = {2}) : 
  {3, log2 a} ∪ {a, b} = {2, 3, 4} := by
{
  sorry
}

end set_union_example_l261_261983


namespace total_interval_length_l261_261506

noncomputable def interval_length : ℝ :=
  1 / (1 + 2^Real.pi)

theorem total_interval_length :
  ∀ x : ℝ, x < 1 ∧ Real.tan (Real.log x / Real.log 4) > 0 →
  (∃ y, interval_length = y) :=
by
  sorry

end total_interval_length_l261_261506


namespace tan_B_find_c_l261_261612

/-
In acute triangle ABC, the sides opposite to angles A, B, C are a, b, c, respectively.
Given that sin A = 3/5, tan (A - B) = -1/2, and b = 5. 
(1) Find the value of tan B;
(2) If b = 5, find the value of c.
-/

variable (A B C a b c : ℝ)
variable (h1 : Real.sin A = 3 / 5)
variable (h2 : Real.tan (A - B) = -1 / 2)
variable (hb : b = 5)

theorem tan_B : tan B = 2 :=
sorry

theorem find_c : c = 11 / 2 :=
sorry

end tan_B_find_c_l261_261612


namespace number_of_participants_2004_l261_261241

theorem number_of_participants_2004 :
  let initial := 1000
  let rate := 1.6
  let p2001 := initial * rate
  let p2002 := p2001 * rate
  let p2003 := p2002 * rate
  let p2004 := p2003 * rate
  (p2004.round : ℤ) = 6554 :=
by sorry

end number_of_participants_2004_l261_261241


namespace unique_seating_scheme_l261_261364

theorem unique_seating_scheme :
  ∃! (x y : ℕ), (x + ∑ i in finset.range (y - 1), (x + i + 1) = 2004) ∧ y > 20 :=
by
  sorry

end unique_seating_scheme_l261_261364


namespace smallest_n_condition_l261_261923

theorem smallest_n_condition :
  ∃ n : ℕ, n > 0 ∧ (∀ (s : Finset ℕ), s.card = n →
  (∀ (x : ℕ) (H : x ∈ s), x > 1 ∧ x ≤ 2009 ∧ (∀ y ∈ s, x ≠ y → coprime x y)) →
  (∃ p : ℕ, p ∈ s ∧ Prime p)) ∧ n = 15 :=
sorry

end smallest_n_condition_l261_261923


namespace problem_solution_l261_261996

theorem problem_solution :
  (∑ k in finset.range 10, (k + 1) * 2^(k + 1)) = 18434 := 
sorry

end problem_solution_l261_261996


namespace stratified_sampling_correct_l261_261430

-- Defining the conditions
def first_grade_students : ℕ := 600
def second_grade_students : ℕ := 680
def third_grade_students : ℕ := 720
def total_sample_size : ℕ := 50
def total_students := first_grade_students + second_grade_students + third_grade_students

-- Expected number of students to be sampled from first, second, and third grades
def expected_first_grade_sample := total_sample_size * first_grade_students / total_students
def expected_second_grade_sample := total_sample_size * second_grade_students / total_students
def expected_third_grade_sample := total_sample_size * third_grade_students / total_students

-- Main theorem statement
theorem stratified_sampling_correct :
  expected_first_grade_sample = 15 ∧
  expected_second_grade_sample = 17 ∧
  expected_third_grade_sample = 18 := by
  sorry

end stratified_sampling_correct_l261_261430


namespace factor_expression_l261_261472

-- Define the variables a, b, and c as real numbers
variables (a b c : ℝ)

-- Define x, y, z according to the given conditions
def x : ℝ := a - b
def y : ℝ := b - c
def z : ℝ := c - a

-- State the theorem
theorem factor_expression : ((a - b) ^ 2 + (b - c) ^ 2 + (c - a) ^ 2) / ((a - b) + (b - c) + (c - a)) = 0 :=
by 
  -- Using x + y + z = 0
  have h : x + y + z = 0 := by 
    unfold x y z  -- Simplify the definitions of x, y, z
    ring         -- Use ring to simplify
  -- Simplify the original expression
  rw [add_assoc (a - b) (b - c) (c - a), add_comm (b - c) (c - a), add_assoc (a - b) (c - a) (b - c)] 
  rw [add_assoc]  -- More rewriting to simplify
  rw ←add_assoc
  rw [add_comm (b - c) (c - a)] 
  rw [add_comm (a - b + (c - a)) (b - c)]
  rw [h]       -- Substitute x + y + z = 0
  simp         -- Simplify the expression

  have hnz: (a - b + (b - c) + (c - a)) = 0 := by sorry -- condition holds
 
  exact hnz

end factor_expression_l261_261472


namespace median_length_YN_area_triangle_l261_261619

variable (X Y Z N : Point)
variable (A B : ℝ)
variable (XY YZ XZ YN : ℝ)

-- Conditions
def right_triangle := ang XYZ = 90
def midpoint_XZ := dist X N = dist N Z
def side_XY := XY = 6
def side_YZ := YZ = 8
def hypotenuse_XZ := XZ = Real.sqrt (XY^2 + YZ^2)
def median_YN := YN = XZ / 2

-- Proof that median YN is 5.0 cm
theorem median_length_YN : right_triangle → midpoint_XZ → side_XY → side_YZ → median_YN := by
  intros
  sorry

-- Proof that area of the triangle is 24 cm^2
theorem area_triangle : right_triangle → side_XY → side_YZ → (let Area := 1/2 * XY * YZ => Area = 24) := by
  intros
  sorry

end median_length_YN_area_triangle_l261_261619


namespace correct_option_is_C_l261_261770

-- Our conditions as mathematical expressions
def condition_A : Prop := (+6) + (-13) = +7
def condition_B : Prop := (+6) + (-13) = -19
def condition_C : Prop := (+6) + (-13) = -7
def condition_D : Prop := (-5) + (-3) = 8

-- The proposition we need to prove
theorem correct_option_is_C : condition_C ∧ ¬condition_A ∧ ¬condition_B ∧ ¬condition_D :=
by 
  sorry

end correct_option_is_C_l261_261770


namespace directrix_of_parabola_l261_261499

def parabola_eq (x : ℝ) : ℝ := -4 * x^2 + 4

theorem directrix_of_parabola : 
  ∃ y : ℝ, y = 65 / 16 :=
by
  sorry

end directrix_of_parabola_l261_261499


namespace smartphone_demand_inverse_proportional_l261_261213

theorem smartphone_demand_inverse_proportional (k : ℝ) (d d' p p' : ℝ) 
  (h1 : d = 30)
  (h2 : p = 600)
  (h3 : p' = 900)
  (h4 : d * p = k) :
  d' * p' = k → d' = 20 := 
by 
  sorry

end smartphone_demand_inverse_proportional_l261_261213


namespace distance_from_origin_to_circle_center_l261_261629

theorem distance_from_origin_to_circle_center :
  ∀ (θ : ℝ), ∃ (r : ℝ), 
  (r = 3 * Real.sqrt 2 * Real.cos (θ + Real.pi / 4) + 7 * Real.sin θ) →
  ∀ x y, (x = r * Real.cos θ) ∧ (y = r * Real.sin θ) →
  ∀ (hx hy : ℝ), (hx = x - 3 / 2) ∧ (hy = y - 2) →
  (hx ^ 2 + hy ^ 2 = 25 / 4) →
  Real.sqrt ((3/2)^2 + 2^2) = 5/2 :=
by important_conditions 
   sorry

end distance_from_origin_to_circle_center_l261_261629


namespace second_player_wins_30_digits_l261_261016

theorem second_player_wins_30_digits :
  (∀ (digits : ℕ → ℕ), (∀ i, i < 30 → digits i ∈ {1, 2, 3, 4, 5}) →
    (∀ i, i < 15 → digits (2 * i) + digits (2 * i + 1) = 6) →
    ((∑ i in finRange 30, digits i) % 9 = 0)) :=
  sorry

end second_player_wins_30_digits_l261_261016


namespace lakeside_volleyball_club_players_l261_261234

theorem lakeside_volleyball_club_players (x y total_cost players : ℕ)
    (hx : x = 10)
    (hy : y = x + 15)
    (total_cost = 5600)
    (∀ player, player_cost = 2 * (x + y)) :
  players = total_cost / player_cost :=
by
  sorry

end lakeside_volleyball_club_players_l261_261234


namespace sum_sequence_S_n_l261_261161

variable {S : ℕ+ → ℚ}
noncomputable def S₁ : ℚ := 1 / 2
noncomputable def S₂ : ℚ := 5 / 6
noncomputable def S₃ : ℚ := 49 / 72
noncomputable def S₄ : ℚ := 205 / 288

theorem sum_sequence_S_n (n : ℕ+) :
  (S 1 = S₁) ∧ (S 2 = S₂) ∧ (S 3 = S₃) ∧ (S 4 = S₄) ∧ (∀ n : ℕ+, S n = n / (n + 1)) :=
by
  sorry

end sum_sequence_S_n_l261_261161
