import Mathlib
import Mathlib.Algebra.ArithmeticProgressions
import Mathlib.Algebra.GeomMean.ArithMean
import Mathlib.Algebra.GroupPower.Order
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Quadratic
import Mathlib.Algebra.Ring
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.FDeriv
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Classical.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.Mod
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.MeasureTheory.IntervalIntegral
import Mathlib.MeasureTheory.ProbabilitySpace
import Mathlib.Probability.Basic
import Mathlib.Probability.Distribution
import Mathlib.Tactic
import Mathlib.Tactic.Linarith

namespace midpoint_interval_a_l36_36330

theorem midpoint_interval_a : 
  ∀ a : ℝ, (a > (1/3)) → 
    (∃ x1 x2 : ℝ, 
      (|x1 - 3| = a * x1 - 1) ∧ 
      (|x2 - 3| = a * x2 - 1) ∧ 
      (x1 ≠ x2)) ∧ 
    (Real.midpoint ((1/3) : ℝ) (∞ : ℝ) = 0.667) := 
by
  sorry

end midpoint_interval_a_l36_36330


namespace percentage_of_water_in_new_mixture_l36_36255

-- Define the conditions as per problem statement
def first_mixture_volume : ℝ := 100
def first_mixture_liquid_percent : ℝ := 0.25

def second_mixture_volume : ℝ := 90
def second_mixture_liquid_percent : ℝ := 0.30

-- Calculation of expected percentage of water in new mixture
def expected_water_percent : ℝ := 72.63

-- Lean statement to prove the percentage of water in the new mixture
theorem percentage_of_water_in_new_mixture :
  let 
    first_mixture_water := first_mixture_volume * (1 - first_mixture_liquid_percent)
    second_mixture_water := second_mixture_volume * (1 - second_mixture_liquid_percent)
    total_mixture_volume := first_mixture_volume + second_mixture_volume
    total_water := first_mixture_water + second_mixture_water
    water_percent : ℝ := (total_water / total_mixture_volume) * 100
  in 
  water_percent = expected_water_percent := 
by 
  -- Skip the actual proof
  sorry

end percentage_of_water_in_new_mixture_l36_36255


namespace find_S_equivalency_l36_36241

theorem find_S_equivalency
  (R S T : ℝ)
  (c : ℝ)
  (h1 : R = c * S / T)
  (h2 : 2 = (\(c = 2\) : c = (4 / 3) / (3 / 7) / (9 / 14)))
  (h3 : ∀ S, R = sqrt 48 → T = sqrt 75 → S = 30) :
  S = 30 := by
  sorry

end find_S_equivalency_l36_36241


namespace best_teacher_selection_l36_36532

variables {scores : List ℝ} (h_len : scores.length = 7)

def remaining_scores (s : List ℝ) : List ℝ :=
  s.qsort (≤) |>.drop 1 |>.pop_back

noncomputable def mean (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (λ x, (x - m) ^ 2)).sum / xs.length

theorem best_teacher_selection (xs : List ℝ) (h_len : xs.length = 7) :
  let xs' := remaining_scores xs in
  (mean xs', variance xs') = (84, 4.84) ∨
  (mean xs', variance xs') = (84, 1.6) ∨
  (mean xs', variance xs') = (85, 1.6) ∨
  (mean xs', variance xs') = (85, 4) := sorry

end best_teacher_selection_l36_36532


namespace minimum_sum_of_distances_l36_36768

open Real

theorem minimum_sum_of_distances :
  let l1 := (4 * x - 3 * y + 6 = 0),
      l2 := (x = 0),
      parabola := (y^2 = 4 * x),
      distance_to_l2 (a : ℝ) := a^2,
      distance_to_l1 (a : ℝ) := (|4 * a^2 - 6 * a + 6| / 5),
      total_distance (a : ℝ) := (distance_to_l1 a + distance_to_l2 a)
  in
  ∃ a : ℝ, (parabola → ((total_distance a) = 1)) :=
begin
  sorry
end

end minimum_sum_of_distances_l36_36768


namespace age_ratio_l36_36680

variable (D : ℕ) (A_future : ℕ)
theorem age_ratio (h1 : D = 30) (h2 : A_future = 25) : 
  let A := A_future - 5 in 
  (A : ℕ) / (D : ℕ) = 2 / 3 := 
by
  sorry

end age_ratio_l36_36680


namespace dot_product_BC_CA_l36_36756

variables {A B C : Type} -- Define variables representing points of triangle
variables (BC AC : ℝ) (angle_C : ℝ) -- Define variables for given conditions
variables (BC_pos : BC > 0) (AC_pos : AC > 0) (angle_C_60 : angle_C = real.pi / 3) -- conditions representing BC = 4, AC = 8, and ∠C = 60°

theorem dot_product_BC_CA :
  BC = 4 → AC = 8 → angle_C = real.pi / 3 → 
  (BC * AC * real.cos (2 * real.pi / 3)) = -16 :=
by
  intros hBC hAC hAngleC
  rw [hBC, hAC, hAngleC, real.cos_two_pi_div_three]
  norm_num
  sorry -- Placeholder for proof

end dot_product_BC_CA_l36_36756


namespace total_collection_value_l36_36719

theorem total_collection_value (total_stickers : ℕ) (partial_stickers : ℕ) (partial_value : ℕ)
  (same_value : ∀ (stickers : ℕ), stickers = total_stickers → stickers * partial_value / partial_stickers = stickers * (partial_value / partial_stickers)):
  partial_value = 24 ∧ partial_stickers = 6 ∧ total_stickers = 18 → total_stickers * (partial_value / partial_stickers) = 72 :=
by {
  sorry
}

end total_collection_value_l36_36719


namespace find_FD_l36_36449

-- Define the parallelogram properties and conditions

variables {A B C D E F : Type} 

-- Assuming the basic properties of the parallelogram
def parallelogram (A B C D : Type) : Prop := 
  ∃ (AB BC CD DA : Real) (α β : Real), 
    AB = 20 ∧ BC = 14 ∧ CD = 14 ∧ DA = 20 ∧ α = 150 ∧ β = 150

-- Extend CD to E such that DE = 6
def extend_CD_to_E (CD DE CE : Real) : Prop := CD = 14 ∧ DE = 6 ∧ CE = 20

-- Define the intersection point
def intersect_BE_AD (BE AD DF : Real) : Prop := BE = sqrt (196 + 400 + 280 * sqrt 3) ∧ AD = 14 ∧ DF = 4.2

-- Prove that DF = 4.2 given the above conditions
theorem find_FD (A B C D E F : Type) 
  (h₁ : parallelogram A B C D)
  (h₂ : extend_CD_to_E 14 6 20)
  (h₃ : intersect_BE_AD (sqrt (196 + 400 + 280 * sqrt 3)) 14 4.2) : 
  True :=
by 
  sorry

end find_FD_l36_36449


namespace g_inv_of_3_pow_neg_3_l36_36056

def g (x : ℝ) : ℝ := 18 / (4 + 2 * x)

theorem g_inv_of_3_pow_neg_3 : (g⁻¹ 3) ^ (-3) = 1 :=
by
  -- Proof goes here
  sorry

end g_inv_of_3_pow_neg_3_l36_36056


namespace marked_elements_duplicate_l36_36425

open Nat

def table : Matrix (Fin 4) (Fin 10) ℕ := ![
  ![0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
  ![9, 0, 1, 2, 3, 4, 5, 6, 7, 8], 
  ![8, 9, 0, 1, 2, 3, 4, 5, 6, 7], 
  ![1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
]

theorem marked_elements_duplicate 
  (marked : Fin 4 → Fin 10) 
  (h_marked_unique_row : ∀ i1 i2, i1 ≠ i2 → marked i1 ≠ marked i2)
  (h_marked_unique_col : ∀ j, ∃ i, marked i = j) :
  ∃ i1 i2, i1 ≠ i2 ∧ table i1 (marked i1) = table i2 (marked i2) := sorry

end marked_elements_duplicate_l36_36425


namespace mn_pq_concyclic_l36_36357

-- Defining the conditions
variables {A B C D E F H X Y M N P Q : Type}

-- Given conditions and definitions
variables (triangle_ABC : Triangle A B C)
variables (altitude_AD : Altitude D triangle_ABC A D)
variables (altitude_BE : Altitude E triangle_ABC B E)
variables (altitude_CF : Altitude F triangle_ABC C F)
variables (orthocenter_H : Orthocenter H triangle_ABC)
variables (points_XY_on_BC : OnLine [X, Y] (Line B C))
variables (ratio_DX_DB_DY_DC : DX/DB = DY/DC)
variables (proj_X_CF: Projection M X (Line C F))
variables (proj_X_CA: Projection N X (Line C A))
variables (proj_Y_BE: Projection P Y (Line B E))
variables (proj_Y_BA: Projection Q Y (Line B A))

-- Problem statement: prove M, N, P, Q are concyclic
theorem mn_pq_concyclic :
  CyclicPoints M N P Q :=
begin
  sorry -- Proof omitted
end

end mn_pq_concyclic_l36_36357


namespace total_profit_l36_36570

/-
Define the conditions:
- mary_investment: the amount Mary invested, $800
- mike_investment: the amount Mike invested, $200
- total_investment: total investment made by both, $1000
- profit_division_ratio: ratio to divide two-thirds profit
- effort_share_ratio: ratio to divide one-third profit equally
- diff_mary_mike: the difference between Mary's and Mike's total share
- profit: the total profit P (to be solved, initially undefined but we assert it to be $3000)

Use these definitions to formulate the theorem that the total profit P is $3000.
-/

def mary_investment : ℝ := 800
def mike_investment : ℝ := 200
def total_investment : ℝ := mary_investment + mike_investment

def ratio_mary_inv : ℝ := mary_investment / total_investment 
def ratio_mike_inv : ℝ := mike_investment / total_investment

def profit_division_ratio : ℝ := (2 / 3)
def effort_share_ratio : ℝ := (1 / 3) / 2

def diff_mary_mike : ℝ := 1200

theorem total_profit {P : ℝ} : 
  let mary_share := effort_share_ratio * P + ratio_mary_inv * profit_division_ratio * P,
      mike_share := effort_share_ratio * P + ratio_mike_inv * profit_division_ratio * P
  in mary_share - mike_share = diff_mary_mike → P = 3000 := 
by 
  intros,
  let h := (effort_share_ratio * P + ratio_mary_inv * profit_division_ratio * P) - 
            (effort_share_ratio * P + ratio_mike_inv * profit_division_ratio * P) = diff_mary_mike,
  sorry

end total_profit_l36_36570


namespace greatest_distance_between_circle_centers_l36_36549

/-- Given two circles within a 12-inch by 14-inch rectangle,
each having a diameter of 6 inches and not extending beyond the rectangle,
prove that the greatest possible distance between the centers of the two circles is 10 inches. -/
theorem greatest_distance_between_circle_centers :
  let rect_w := 12
  let rect_h := 14
  let circle_d := 6
  let circle_r := circle_d / 2
  let smaller_rect_w := rect_w - 2 * circle_r
  let smaller_rect_h := rect_h - 2 * circle_r
  let distance := Math.sqrt (smaller_rect_w^2 + smaller_rect_h^2)
  in distance = 10 :=
by
  sorry

end greatest_distance_between_circle_centers_l36_36549


namespace cannot_form_set_of_good_friends_of_wang_ming_l36_36673

def is_well_defined_set (description : String) : Prop := sorry  -- Placeholder for the formal definition.

theorem cannot_form_set_of_good_friends_of_wang_ming :
  ¬ is_well_defined_set "Good friends of Wang Ming" :=
sorry

end cannot_form_set_of_good_friends_of_wang_ming_l36_36673


namespace coeff_x3_in_expansion_of_x_plus_1_50_l36_36087

theorem coeff_x3_in_expansion_of_x_plus_1_50 :
  (Finset.range 51).sum (λ k => Nat.choose 50 k * (1 : ℕ) ^ (50 - k) * k ^ 3) = 19600 := by
  sorry

end coeff_x3_in_expansion_of_x_plus_1_50_l36_36087


namespace product_is_in_A_l36_36792

def is_sum_of_squares (z : Int) : Prop :=
  ∃ t s : Int, z = t^2 + s^2

variable {x y : Int}

theorem product_is_in_A (hx : is_sum_of_squares x) (hy : is_sum_of_squares y) :
  is_sum_of_squares (x * y) :=
sorry

end product_is_in_A_l36_36792


namespace arithmetic_sequence_sum_l36_36828

noncomputable def sum_of_first_n_terms (n : ℕ) (a d : ℝ) : ℝ :=
  n / 2 * (2 * a + (n - 1) * d)

theorem arithmetic_sequence_sum 
  (a_n : ℕ → ℝ) 
  (h_arith : ∃ d, ∀ n, a_n (n + 1) = a_n n + d) 
  (h1 : a_n 1 + a_n 2 + a_n 3 = 3 )
  (h2 : a_n 28 + a_n 29 + a_n 30 = 165 ) 
  : sum_of_first_n_terms 30 (a_n 1) (a_n 2 - a_n 1) = 840 := 
  sorry

end arithmetic_sequence_sum_l36_36828


namespace min_value_of_y_l36_36178

open Real

noncomputable def y (x : ℝ) : ℝ := 4 / (x - 2)

theorem min_value_of_y :
  (∀ x1 x2 ∈ Icc 3 6, x1 ≤ x2 → y x2 ≤ y x1) → ∀ x ∈ Icc 3 6, y x ≥ 1 := by
  sorry

end min_value_of_y_l36_36178


namespace expectation_eq_of_ae_eq_and_E_ξ_exists_l36_36489

open MeasureTheory

variables {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

-- Define two random variables
variables (ξ η : Ω → ℝ)

-- Define your conditions
axiom ξ_eq_η_ae : ∀ᵐ ω ∂μ, ξ ω = η ω
axiom E_ξ_exists : Integrable ξ μ

-- Now state the theorem
theorem expectation_eq_of_ae_eq_and_E_ξ_exists :
  Integrable η μ ∧ ∫ ω, ξ ω ∂μ = ∫ ω, η ω ∂μ :=
sorry

end expectation_eq_of_ae_eq_and_E_ξ_exists_l36_36489


namespace abs_eq_ax_minus_one_has_two_solutions_l36_36333
noncomputable def midpoint_of_interval : ℕ → ℕ → ℚ := sorry

theorem abs_eq_ax_minus_one_has_two_solutions (a : ℚ) (h : a > 1/3) : 
  ∃ x1 x2, x1 ≠ x2 ∧ (|x1 - 3| = a * x1 - 1) ∧ (|x2 - 3| = a * x2 - 1) :=
sorry

#eval midpoint_of_interval 1 3 -- Expected output 0.667 or approximately 2/3

end abs_eq_ax_minus_one_has_two_solutions_l36_36333


namespace combined_spots_l36_36382

-- Definitions of the conditions
def Rover_spots : ℕ := 46
def Cisco_spots : ℕ := Rover_spots / 2 - 5
def Granger_spots : ℕ := 5 * Cisco_spots

-- The proof statement
theorem combined_spots :
  Granger_spots + Cisco_spots = 108 := by
  sorry

end combined_spots_l36_36382


namespace range_of_a_l36_36787

noncomputable def f (x : ℝ) : ℝ := if x < 0 then x^2 + exp(x) - (1 / 2) else 0
noncomputable def g (x a : ℝ) : ℝ := x^2 + log(x + a)

theorem range_of_a (a : ℝ) :
  (∃ x₀ : ℝ, x₀ < 0 ∧ (f x₀ = g (-x₀) a)) ↔ a ∈ set.Iio (Real.sqrt Real.exp 1) := sorry

end range_of_a_l36_36787


namespace probability_divisor_of_12_on_12sided_die_l36_36611

theorem probability_divisor_of_12_on_12sided_die (h : ∀ i, 1 ≤ i ∧ i ≤ 12 → true) : 
  (probability (fun (n : ℕ) => n ∣ 12 ∧ 1 ≤ n ∧ n ≤ 12)) = 1/2 := 
sorry

end probability_divisor_of_12_on_12sided_die_l36_36611


namespace not_perfect_square_l36_36153

theorem not_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, 7 * n + 3 = k^2 := 
by
  sorry

end not_perfect_square_l36_36153


namespace hyperbola_standard_eq_l36_36826

open Real

def is_asymptote (C : ℝ × ℝ → Prop) (l : ℝ × ℝ → Prop) : Prop :=
∀ (x y : ℝ), C (x, y) → (∃ (k : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ ⟨x', y'⟩, (ε > dist ⟨x', y'⟩ (x, y) → l (x', y')))

def passes_through (C : ℝ × ℝ → Prop) (P : ℝ × ℝ) : Prop :=
C P

theorem hyperbola_standard_eq (C : ℝ × ℝ → Prop)
  (h_through : passes_through C (1, 1))
  (h_asymp1 : is_asymptote C (λ ⟨x, y⟩, 2 * x + y = 0))
  (h_asymp2 : is_asymptote C (λ ⟨x, y⟩, 2 * x - y = 0)) :
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ (λ ⟨x, y⟩, a * x^2 - b * y^2 = 1) = C :=
sorry

end hyperbola_standard_eq_l36_36826


namespace triangle_congruence_hexagon_similar_to_ABC_l36_36859

-- Define the setup for the problem
structure Triangle :=
(A B C O1 A' B' C' : Type)
[InCircumcircle : IsInCircumcircle A B C]

-- Define the proof statements
theorem triangle_congruence (T : Triangle) : T.A' ∈ Circumcircle T.A T.B T.C → 
T.B' ∈ Circumcircle T.B T.C T.A → 
T.C' ∈ Circumcircle T.C T.A T.B → 
T.A'B'C' ≅ T.ABC := sorry

theorem hexagon_similar_to_ABC (T : Triangle) (O1 : T.O1) : 
all_similar (List.map (fun (X : {X'| X' = T.A' ∨ X' = T.B' ∨ X' = T.C'}) => triangle (T.A X.B X.Y)) 
[some_list_produced_by_vertex_connections]) T.ABC := sorry

end triangle_congruence_hexagon_similar_to_ABC_l36_36859


namespace find_difference_square_l36_36394

theorem find_difference_square (x y c b : ℝ) (h1 : x * y = c^2) (h2 : (1 / x^2) + (1 / y^2) = b * c) : 
  (x - y)^2 = b * c^4 - 2 * c^2 := 
by sorry

end find_difference_square_l36_36394


namespace neznaika_expression_gt_30_l36_36476

    theorem neznaika_expression_gt_30 : 
      ∃ (a b c : ℚ), 
      (a = 20 ∨ a = 2) ∧ 
      (b = 20 ∨ b = 2) ∧ 
      (c = 20 ∨ c = 2) ∧ 
      (∃ (d e : ℚ), d = a / (b - real.sqrt(c))) ∧ 
      d > 30 :=
    by
      sorry
    
end neznaika_expression_gt_30_l36_36476


namespace problem_solution_l36_36855

noncomputable def f : Real → Real := sorry -- Given function f(x)

-- Conditions
axiom odd_f (x : Real) : f(-x) = -f(x)
axiom continuous_f : ContinuousOn f (Set.Ioo (-π / 2) (π / 2))
axiom derivative_condition (x : Real) (h : x ∈ Set.Ioo (-π / 2) 0) : 
  deriv f x * Real.cos x - f x * Real.sin x < 0

theorem problem_solution (t : Real) 
  (h1 : t ∈ Set.Ioo (-π / 2) (π / 2)) 
  (inequality : f t * Real.cos t < (1 / 2) * f (π / 3)) 
  : t ∈ Set.Ioo (π / 3) (π / 2) :=
sorry

end problem_solution_l36_36855


namespace probability_divisor_of_12_l36_36644

theorem probability_divisor_of_12 : 
  (nat.factorization 12).keys = {1, 2, 3, 4, 6, 12} → 
  (∃ die_side, die_side ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) → 
  (∃ favorable_outcomes, favorable_outcomes ∈ {1, 2, 3, 4, 6, 12}) → 
  ∃ probability, probability = (favorable_outcomes.card : ℚ) / (die_side.card : ℚ) ∧ probability = 1/2 :=
by 
  sorry

end probability_divisor_of_12_l36_36644


namespace value_of_a_l36_36471

noncomputable def normal_distribution_example : ℝ :=
  let μ := 3
  let σ := 2  -- Note: Standard deviation is the square root of variance, hence sqrt(4) = 2
  let ξ := measure_theory.probability_theory.normal μ σ
  let a := by
    sorry  -- This is where you would solve the problem in Lean

  a  -- This will return the value of a which we need to show is 7/3

-- Now we need to state the theorem:
theorem value_of_a :
  2 * a - 3 > 0 ∧ measure_theory.probability_theory.normal 3 2 (≤ 2 * a - 3) = measure_theory.probability_theory.normal 3 2 (> a + 2) →
  a = 7 / 3 :=
begin
  intros h,
  rw ←probability_theory.measure_norm_dist at h,
  sorry  -- Proof goes here
end

end value_of_a_l36_36471


namespace sqrt_of_neg_five_squared_l36_36686

theorem sqrt_of_neg_five_squared : Real.sqrt ((-5 : Real) ^ 2) = 5 := 
by 
  sorry

end sqrt_of_neg_five_squared_l36_36686


namespace solve_system_of_equations_l36_36164

theorem solve_system_of_equations :
  ∃ (x y : ℝ),
    (5 * x^2 - 14 * x * y + 10 * y^2 = 17) ∧ (4 * x^2 - 10 * x * y + 6 * y^2 = 8) ∧
    ((x = -1 ∧ y = -2) ∨ (x = 11 ∧ y = 7) ∨ (x = -11 ∧ y = -7) ∨ (x = 1 ∧ y = 2)) :=
by
  sorry

end solve_system_of_equations_l36_36164


namespace intersect_at_least_one_l36_36392

variables {Point Line Plane : Type}
variables (a b l : Line) (α β : Plane)

-- Assumptions based on the conditions
axiom skew_lines : ¬∃ P : Point, P ∈ a ∧ P ∈ b -- a and b are skew lines
axiom line_in_plane_a : ∀ P : Point, P ∈ a → P ∈ α -- a ⊂ α
axiom line_in_plane_b : ∀ P : Point, P ∈ b → P ∈ β -- b ⊂ β
axiom planes_intersection : ∃ l : Line, ∀ P : Point, P ∈ l ↔ (P ∈ α ∧ P ∈ β) -- intersection of α and β is l

-- The theorem statement
theorem intersect_at_least_one : (∃ P : Point, P ∈ l ∧ P ∈ a) ∨ (∃ P : Point, P ∈ l ∧ P ∈ b) :=
sorry

end intersect_at_least_one_l36_36392


namespace rent_friends_l36_36253

theorem rent_friends (n : ℕ)
  (h1 : (800 * n : ℝ))
  (h2 : ((880 * n : ℝ) = 800 * n + 320))
  (h3 : (320 = 0.2 * 1600)) :
  n = 4 := by
  sorry

end rent_friends_l36_36253


namespace count_parallel_sided_quads_l36_36762

theorem count_parallel_sided_quads (M : Type) [regular_poly M 22] : 
  (number_of_parallel_sided_quadrilaterals M) = 1045 :=
sorry

end count_parallel_sided_quads_l36_36762


namespace smaug_copper_coins_l36_36911

def copper_value_of_silver (silver_coins silver_to_copper : ℕ) : ℕ :=
  silver_coins * silver_to_copper

def copper_value_of_gold (gold_coins gold_to_silver silver_to_copper : ℕ) : ℕ :=
  gold_coins * gold_to_silver * silver_to_copper

def total_copper_value (gold_coins silver_coins gold_to_silver silver_to_copper : ℕ) : ℕ :=
  copper_value_of_gold gold_coins gold_to_silver silver_to_copper +
  copper_value_of_silver silver_coins silver_to_copper

def actual_copper_coins (total_value gold_value silver_value : ℕ) : ℕ :=
  total_value - (gold_value + silver_value)

theorem smaug_copper_coins :
  let gold_coins := 100
  let silver_coins := 60
  let silver_to_copper := 8
  let gold_to_silver := 3
  let total_copper_value := 2913
  let gold_value := copper_value_of_gold gold_coins gold_to_silver silver_to_copper
  let silver_value := copper_value_of_silver silver_coins silver_to_copper
  actual_copper_coins total_copper_value gold_value silver_value = 33 :=
by
  sorry

end smaug_copper_coins_l36_36911


namespace probability_divisor_of_12_l36_36603

theorem probability_divisor_of_12 (fair_die : True) (sides : Nat) (h : sides = 12) : 
  ∃ p : ℚ, p = 1 / 2 ∧ ∀ roll : Fin sides, is_divisor roll.val 12 ↔ roll.val ∈ {1, 2, 3, 4, 6, 12} → p = 1 / 2 :=
by sorry

def is_divisor (a b : Nat) : Prop := b % a = 0

end probability_divisor_of_12_l36_36603


namespace determine_f3_l36_36251

noncomputable def f : ℝ → ℝ := sorry

theorem determine_f3 :
  (∀ x : ℝ, f(x) * f(x + 2) = 12) →
  f 2017 = 2 →
  f 3 = 6 :=
by sorry

end determine_f3_l36_36251


namespace part1_max_value_part2_k_range_part3_k_range_l36_36767

namespace MathProblem

variable {R : Type} [linearOrderedField R]

def f (x k : R) : R := 2 * x^2 + x - k

-- We already know g(x) = x^3 - 3x from the problem discussions above.
def g (x : R) : R := x^3 - 3 * x

theorem part1_max_value :
  ∃ (max_val : R), max_val = 2 ∧ (∀ x y : R, 
  ((x < -1 ∧ y ≥ -1) → (g x < g y)) ∧ 
  ((x < 1 ∧ x > -1 ∧ y ≥ 1) → (g x > g y)) ∧ 
  ((x > 1 ∧ y ≤ 1) → (g x > g y))) := sorry

theorem part2_k_range (k : R) :
  (∀ x : R, -1 ≤ x ∧ x ≤ 3 → f x k ≤ g x) ↔ k ≥ 8 := sorry

theorem part3_k_range (k : R) :
  (∀ x1 x2 : R, -1 ≤ x1 ∧ x1 ≤ 3 ∧ -1 ≤ x2 ∧ x2 ≤ 3 → f x1 k ≤ g x2) ↔ k ≥ 23 := sorry

end MathProblem

end part1_max_value_part2_k_range_part3_k_range_l36_36767


namespace quadratic_roots_real_and_equal_l36_36742

theorem quadratic_roots_real_and_equal :
  (∀ a b c : ℝ, a = 1 ∧ b = -6 ∧ c = 9 →
  (b^2 - 4*a*c = 0) →
  (∃ x : ℝ, (x^2 - 6*x + 9 = 0) ∧ ∃ y : ℝ, (x = y))) :=
begin
  sorry
end

end quadratic_roots_real_and_equal_l36_36742


namespace derivative_of_f_l36_36364

-- Define the function f
def f (x : ℝ) : ℝ := Real.sin (2 * x)

-- State the theorem to prove f'(x) = 2 * cos(2x)
theorem derivative_of_f (x : ℝ) : deriv f x = 2 * Real.cos (2 * x) :=
by
  -- Proof will be filled in here
  sorry

end derivative_of_f_l36_36364


namespace min_value_of_f_on_interval_l36_36516

def f (x : ℝ) := -2 * x + 5

theorem min_value_of_f_on_interval : 
  ∃ x ∈ set.Icc (2 : ℝ) (4 : ℝ), ∀ y ∈ set.Icc (2 : ℝ) (4 : ℝ), f(x) ≤ f(y) ∧ f(x) = -3 :=
by
  sorry

end min_value_of_f_on_interval_l36_36516


namespace count_numbers_ending_in_1_pow_l36_36327

theorem count_numbers_ending_in_1_pow :
  {k : ℤ | 1 ≤ k ∧ k ≤ 2013 ∧ (k^k % 10 = 1)}.size = 202 :=
sorry

end count_numbers_ending_in_1_pow_l36_36327


namespace initial_red_orchids_l36_36207

-- Definitions based on the conditions
variables (r_now : ℕ) (r_added : ℕ)

-- Given conditions
def r_now := 15
def r_added := 6

-- Goal: Prove the initial number of red orchids in the vase
theorem initial_red_orchids : ∃ r_initial : ℕ, r_initial = r_now - r_added ∧ r_initial = 9 :=
by
  use (r_now - r_added)
  split
  sorry

end initial_red_orchids_l36_36207


namespace solution_count_l36_36052

def count_solutions : ℤ :=
  (set.to_finset {ab : ℤ × ℤ | let (a, b) := ab in a^2 + b^2 < 16 ∧ 
                                  a^2 + b^2 < 8 * a ∧ 
                                  a^2 + b^2 < 8 * b}).card

theorem solution_count :
  count_solutions = 6 :=
sorry

end solution_count_l36_36052


namespace probability_of_rolling_divisor_of_12_l36_36600

def is_divisor (a b : ℕ) : Prop := b % a = 0

noncomputable def probability_divisor_12 : ℚ :=
  let divisors := { n | is_divisor n 12 ∧ n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} }
  let favorable_outcomes := (divisors : set ℕ).to_finset.card
  let total_outcomes := 12
  (favorable_outcomes : ℚ) / total_outcomes

theorem probability_of_rolling_divisor_of_12 : probability_divisor_12 = 1 / 2 := by
  sorry

end probability_of_rolling_divisor_of_12_l36_36600


namespace subset_sum_divisible_l36_36154

theorem subset_sum_divisible (m : ℕ) (a : Fin m → ℤ) (hm : m > 0) :
  ∃ (s : Finset (Fin m)), (s.nonempty) ∧ ((∑ i in s, a i) % m = 0) :=
sorry

end subset_sum_divisible_l36_36154


namespace mean_of_three_digit_multiples_of_8_l36_36957

theorem mean_of_three_digit_multiples_of_8 :
  let a1 := 104
  let an := 992
  let d := 8
  let n := (an - a1) / d + 1
  let sum := n * (a1 + an) / 2
  let mean := sum / n
  mean = 548 :=
by
  let a1 := 104
  let an := 992
  let d := 8
  let n := (an - a1) / d + 1
  let sum := n * (a1 + an) / 2
  let mean := sum / n
  sorry

end mean_of_three_digit_multiples_of_8_l36_36957


namespace construction_costs_l36_36261

theorem construction_costs 
  (land_cost_per_sq_meter : ℕ := 50)
  (bricks_cost_per_1000 : ℕ := 100)
  (roof_tile_cost_per_tile : ℕ := 10)
  (land_area : ℕ := 2000)
  (number_of_bricks : ℕ := 10000)
  (number_of_roof_tiles : ℕ := 500) :
  50 * 2000 + (100 / 1000) * 10000 + 10 * 500 = 106000 :=
by sorry

end construction_costs_l36_36261


namespace shaded_region_area_l36_36704

theorem shaded_region_area {radius1 radius2 : ℝ} (h1 : radius1 = 4) (h2 : radius2 = 5) :
  let dist_centers := radius1 + radius2,
      circumscribed_radius := dist_centers,
      larger_area := Real.pi * circumscribed_radius ^ 2,
      smaller_area1 := Real.pi * radius1 ^ 2,
      smaller_area2 := Real.pi * radius2 ^ 2,
      shaded_area := larger_area - smaller_area1 - smaller_area2
  in shaded_area = 40 * Real.pi :=
by
  simp [h1, h2]
  sorry

end shaded_region_area_l36_36704


namespace pretty_penny_investment_l36_36168

noncomputable def solve_pretty_penny_investment : Float :=
  let x := 300
  let y := 1000
  let z := 200
  let total_investment := x + y + z
  let total_after_3_years := x * 1.04^3 + y * 1.045^3 + z * 1.09
  if total_investment = 1500 ∧ total_after_3_years = 1649.50 then x else 0

theorem pretty_penny_investment (x y z : Float) (h1 : x + y + z = 1500) (h2 : x * 1.04^3 + y * 1.045^3 + z * 1.09 = 1649.50) : x = 300 :=
  sorry

end pretty_penny_investment_l36_36168


namespace distance_between_droplets_l36_36554

section water_droplets

-- Given conditions
def h : ℝ := 300000 -- height in millimeters
def d1 : ℝ := 1 / 1000 -- distance fallen by the first droplet in millimeters
def g : ℝ := 9.81 -- gravity in m/s^2, although it's not per se needed for the final verification

/-- Prove the distance between the two droplets at the moment the first 
one reaches the base of the cliff is 34.6 mm -/
theorem distance_between_droplets : 
  (2 * Real.sqrt(h * d1) - d1) ≈ 34.6 :=
sorry

end water_droplets

end distance_between_droplets_l36_36554


namespace area_of_region_R_l36_36901

theorem area_of_region_R :
  let R := { p : ℝ × ℝ | ∃ x, x ∈ interior_rhombus_around_B p } in
  area R = (9 * (Real.sqrt 6 - Real.sqrt 2)) / 4 :=
by
  -- conditions: Rhombus \(ABCD\) with side length 3 and \(\angle B = 150^\circ\)
  let ABCD : ℝ × ℝ := rhombus 3 150
  let vertex := B
  -- statement: Prove the area of region R
  sorry

end area_of_region_R_l36_36901


namespace tan_arccot_sin_arccot_l36_36289

noncomputable def arccot (x : ℝ) : ℝ := Real.atan (1 / x)

theorem tan_arccot (adj: ℝ) (opp: ℝ) (h_adj: adj = 5) (h_opp: opp = 12) :
  Real.tan (arccot (adj / opp)) = opp / adj := by
  sorry

theorem sin_arccot (adj: ℝ) (opp: ℝ) (h_adj: adj = 5) (h_opp: opp = 12) :
  Real.sin (arccot (adj / opp)) = opp / (Real.sqrt (adj ^ 2 + opp ^ 2)) := by
  sorry

end tan_arccot_sin_arccot_l36_36289


namespace maximal_projection_area_l36_36801

-- Define the variables for the parallelepiped vertices
variables {A B C D A₁ B₁ C₁ D₁ : Type} [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
variables [InnerProductSpace ℝ D] [InnerProductSpace ℝ A₁] [InnerProductSpace ℝ B₁] [InnerProductSpace ℝ C₁] [InnerProductSpace ℝ D₁]

-- Define the projection function
def projection (v1 v2 v3 : A): Set A := -- Here, we should ideally define the projection of the three vectors, but we will leave it as a placeholder
  sorry

-- Define the statement to prove
theorem maximal_projection_area (v1 v2 v3 : A):
  ∀ abc, projection abc v1 v2 v3 → 
  ∃ plane, plane_parallel_to_projection_plane (projection abc v1 v2 v3) 
  := 
  sorry

end maximal_projection_area_l36_36801


namespace employee_pays_correct_amount_l36_36653

theorem employee_pays_correct_amount
    (wholesale_cost : ℝ)
    (retail_markup : ℝ)
    (employee_discount : ℝ)
    (weekend_discount : ℝ)
    (sales_tax : ℝ)
    (final_price : ℝ) :
    wholesale_cost = 200 →
    retail_markup = 0.20 →
    employee_discount = 0.05 →
    weekend_discount = 0.10 →
    sales_tax = 0.08 →
    final_price = 221.62 :=
by
  intros h0 h1 h2 h3 h4
  sorry

end employee_pays_correct_amount_l36_36653


namespace coefficient_of_x3_in_x_plus_one_pow_50_l36_36091

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := (n.choose k)

-- Define the binomial expansion using summation
def binomial_expansion (x : ℕ) (n : ℕ) : ℕ → ℕ :=
  λ k, binom n k * x^k 

-- Define the specific problem
def coeff_x3_in_expansion : ℕ :=
  binom 50 3

-- Theorem stating the desired result
theorem coefficient_of_x3_in_x_plus_one_pow_50 :
  coeff_x3_in_expansion = 19600 :=
by
  -- Skipping the proof part by using sorry
  sorry

end coefficient_of_x3_in_x_plus_one_pow_50_l36_36091


namespace exists_spherical_circle_passing_through_three_points_exists_great_circle_not_diameter_l36_36243

-- Part (a): Circle passing through three points on a sphere
theorem exists_spherical_circle_passing_through_three_points 
    (A B C : Point) (s : Sphere) 
    (hA : A ∈ s.surface) 
    (hB : B ∈ s.surface) 
    (hC : C ∈ s.surface) : 
    ∃ (circle : Circle), (A ∈ circle) ∧ (B ∈ circle) ∧ (C ∈ circle) := 
by 
    sorry

-- Part (b): Great circle passing through two points on a sphere
theorem exists_great_circle_not_diameter 
    (A B : Point) (s : Sphere) 
    (hA : A ∈ s.surface) 
    (hB : B ∈ s.surface) 
    (h_not_diameter : ¬ s.is_diameter A B) : 
    ∃ (great_circle : GreatCircle), (A ∈ great_circle) ∧ (B ∈ great_circle) := 
by 
    sorry

end exists_spherical_circle_passing_through_three_points_exists_great_circle_not_diameter_l36_36243


namespace dessert_combinations_l36_36595

theorem dessert_combinations : 
  let num_flavors := 5 in 
  let num_toppings := 7 in 
  let choose_toppings k n := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k))) in 
  let num_combinations := num_flavors * (choose_toppings 3 num_toppings) in 
  num_combinations = 175 := by
{
  -- details of the definition from conditions: 
  have num_flavors_def : num_flavors = 5 := rfl,
  have num_toppings_def : num_toppings = 7 := rfl,
  have choose_toppings_def : choose_toppings 3 num_toppings = 35 := by
  {
    sorry -- Put the combination calculation proof here
  },
  have num_combinations_def : num_combinations = num_flavors * 35 := by
  {
    rw [choose_toppings_def],
    exact rfl
  },
  rw [num_combinations_def, num_flavors_def],
  norm_num,
  exact rfl
}

end dessert_combinations_l36_36595


namespace find_n_to_m_l36_36810

theorem find_n_to_m (m n : ℤ) (h1 : (m - 3) = -1) (h2 : (n - 3 * m) = 0) : n^m = 36 :=
by
  sorry

end find_n_to_m_l36_36810


namespace binary_1011_to_decimal_is_11_l36_36294

-- Define the binary representation of 1011 in base 2
def binary_1011 : list ℕ := [1, 0, 1, 1]

-- Function to convert a binary number (given as a list of bits) to decimal
def binary_to_decimal (bits : list ℕ) : ℕ :=
  bits.reverse.enum_from 0 |>.foldl (λ acc (bit, position), acc + bit * (2^position)) 0

-- Problem statement: Convert binary 1011 to decimal and prove it equals 11
theorem binary_1011_to_decimal_is_11 : binary_to_decimal binary_1011 = 11 := 
  sorry

end binary_1011_to_decimal_is_11_l36_36294


namespace prime_set_exists_m_n_l36_36344

/- Definitions -/
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

def infinite_primes (S : Set ℕ) : Prop :=
  (∀ p ∈ S, is_prime p) ∧ (∀ n : ℕ, ∃ p ∈ S, p > n)

def product_of_k_primes (S : Set ℕ) (k : ℕ) (n : ℕ) : Prop :=
  ∃ (a : Finset ℕ), (∀ x ∈ a, x ∈ S) ∧ (a.card = k) ∧ (a.prod id = n)

def A (S : Set ℕ) : Set ℕ :=
  {n | ∃ k, k ≥ 2 ∧ product_of_k_primes S k n}

/- Main Theorem -/
theorem prime_set_exists_m_n (S : Set ℕ) (hS : infinite_primes S) :
  ∃ k (m n : ℕ), k ≥ 2 ∧ product_of_k_primes S k m ∧ product_of_k_primes S k n ∧ m ∈ A S ∧ n ∉ A S :=
sorry

end prime_set_exists_m_n_l36_36344


namespace range_of_a_max_k_l36_36746

noncomputable def fixedPointCondition (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  x₀ ≠ 1 ∧ f x₀ = x₀

def f (a : ℝ) (x : ℝ) : ℝ :=
  (x - 1) * Real.exp x - a * Real.log x + x

theorem range_of_a (a : ℝ) :
  (∃ x₀, fixedPointCondition (f a) x₀) → (0 < a ∧ a < Real.exp 1) ∨ (a > Real.exp 1) :=
sorry

theorem max_k (a : ℝ) (x₀ : ℝ) (k : ℤ) :
  fixedPointCondition (f a) x₀ → (k : ℝ) * x₀ < a → k ≤ 2 :=
sorry

end range_of_a_max_k_l36_36746


namespace third_median_length_l36_36216

theorem third_median_length 
  (a b c : ℝ)
  (median1 : ℝ)
  (median2 : ℝ)
  (area : ℝ)
  (h1 : median1 = 5)
  (h2 : median2 = 7)
  (h3 : area = 4 * real.sqrt 21)
  : ∃ median3 : ℝ, median3 = 3 * real.sqrt 7 :=
by
  sorry

end third_median_length_l36_36216


namespace MissyYellsCombined_l36_36137

def yellsAtStubbornDog (timesYellObedient : ℕ) := 4 * timesYellObedient

def totalYells (timesYellObedient : ℕ) (timesYellStubborn : ℕ) := timesYellObedient + timesYellStubborn

theorem MissyYellsCombined :
  ∀ (timesYellObedient : ℕ),
    timesYellObedient = 12 →
    totalYells timesYellObedient (yellsAtStubbornDog timesYellObedient) = 60 :=
by
  intros timesYellObedient h
  rw [h]
  unfold yellsAtStubbornDog totalYells
  norm_num
  rw [h]
  sorry

end MissyYellsCombined_l36_36137


namespace tangent_to_parabola_k_l36_36715

theorem tangent_to_parabola_k (k : ℝ) :
  (∃ (x y : ℝ), 4 * x + 7 * y + k = 0 ∧ y^2 = 32 * x ∧ 
  ∀ (a b : ℝ) (ha : a * y^2 + b * y + k = 0), b^2 - 4 * a * k = 0) → k = 98 :=
by
  sorry

end tangent_to_parabola_k_l36_36715


namespace men_left_hostel_l36_36254

-- Definitions based on the conditions given
def initialMen : ℕ := 250
def initialDays : ℕ := 28
def remainingDays : ℕ := 35

-- The theorem we need to prove
theorem men_left_hostel (x : ℕ) (h : initialMen * initialDays = (initialMen - x) * remainingDays) : x = 50 :=
by
  sorry

end men_left_hostel_l36_36254


namespace probability_divisor_of_12_is_half_l36_36618

noncomputable def divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ d, n % d = 0) (Finset.range (n + 1))

def probability_divisor_of_12 : ℚ :=
  let total_outcomes := 12
  let favorable_outcomes := (divisors 12).card
  favorable_outcomes / total_outcomes

theorem probability_divisor_of_12_is_half :
  probability_divisor_of_12 = 1 / 2 :=
  by
    sorry

end probability_divisor_of_12_is_half_l36_36618


namespace six_pow_2n_plus1_plus_1_div_by_7_l36_36152

theorem six_pow_2n_plus1_plus_1_div_by_7 (n : ℕ) : (6^(2*n+1) + 1) % 7 = 0 := by
  sorry

end six_pow_2n_plus1_plus_1_div_by_7_l36_36152


namespace xy_sum_is_2_l36_36157

theorem xy_sum_is_2 (x y : ℝ) (h : 4 * x^2 + 4 * y^2 = 40 * x - 24 * y + 64) : x + y = 2 := 
by
  sorry

end xy_sum_is_2_l36_36157


namespace law_of_sines_proof_l36_36975

noncomputable def law_of_sines (a b c α β γ : ℝ) :=
  (a / Real.sin α = b / Real.sin β) ∧
  (b / Real.sin β = c / Real.sin γ) ∧
  (α + β + γ = Real.pi)

theorem law_of_sines_proof (a b c α β γ : ℝ) (h : law_of_sines a b c α β γ) :
  (a = b * Real.cos γ + c * Real.cos β) ∧
  (b = c * Real.cos α + a * Real.cos γ) ∧
  (c = a * Real.cos β + b * Real.cos α) :=
sorry

end law_of_sines_proof_l36_36975


namespace shortest_surface_path_l36_36276

/-- An insect lives on the surface of a cube with edge length 2.
    The goal is to prove that the shortest path on the surface of the cube from the center
    of the top face to the center of the bottom face is 4 units long. -/
theorem shortest_surface_path (e : ℝ) (h : e = 2) : 
  let c := (2 : ℝ) in
  shortest_path_length e = 4 :=
by sorry

end shortest_surface_path_l36_36276


namespace probability_of_rolling_divisor_of_12_l36_36601

def is_divisor (a b : ℕ) : Prop := b % a = 0

noncomputable def probability_divisor_12 : ℚ :=
  let divisors := { n | is_divisor n 12 ∧ n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} }
  let favorable_outcomes := (divisors : set ℕ).to_finset.card
  let total_outcomes := 12
  (favorable_outcomes : ℚ) / total_outcomes

theorem probability_of_rolling_divisor_of_12 : probability_divisor_12 = 1 / 2 := by
  sorry

end probability_of_rolling_divisor_of_12_l36_36601


namespace curve_is_ellipse_or_circle_l36_36041

noncomputable def parametric_curve (a : ℝ) (t : ℝ) := 
  (a ≠ 0) → 
    ∃ x y, (x = a * t / (1 + t^2) ∧ y = (1 - t^2) / (1 + t^2)) ∧
    ((2 * x / a)^2 + y^2 = 1) ∧ 
    ((a = 2) → (x^2 + y^2 = 1))

theorem curve_is_ellipse_or_circle (a t : ℝ) (h : a ≠ 0) :
  parametric_curve a t :=
begin
  sorry
end

end curve_is_ellipse_or_circle_l36_36041


namespace rent_fraction_l36_36652

theorem rent_fraction (B R : ℝ) 
  (food_and_beverages_spent : (1 / 4) * (1 - R) * B = 0.1875 * B) : 
  R = 0.25 :=
by
  -- proof skipped
  sorry

end rent_fraction_l36_36652


namespace prove_areas_l36_36239

-- Definitions of points and relationships
variables (A B C D E F G H : Type)
variables [Point : Σ (p : PointSpace A), True]
include Point

-- Areas of specific shapes given as conditions
variable (Area_BCG : ℝ) -- Area of triangle BCG
variable (Area_DFHG : ℝ) -- Area of quadrilateral DFHG

-- Conditions, assuming we have a way to define the relationships
axiom CDE_parallel_AB : LineThrough C D = LineThrough D E ∧ Parallel (LineThrough C E) (LineThrough A B)
axiom F_mid_AD : Midpoint F A D
axiom G_mid_AC : Midpoint G A C
axiom H_inter_AC_BE : LineThrough A C = LineThrough B E

-- Given areas based on conditions
axiom area_BCG_eq : Area_BCG = 12
axiom area_DFHG_eq : Area_DFHG = 8

-- Proving the required areas of triangles based on conditions
theorem prove_areas :
  area (Triangle A F E) = 12 ∧
  area (Triangle A H F) = 4 ∧
  area (Triangle A B G) = 12 ∧
  area (Triangle B G H) = 4 :=
sorry

end prove_areas_l36_36239


namespace problem_statement_l36_36122

theorem problem_statement (a : ℝ) (h_root : a^3 - 3*a^2 + 1 = 0) (h_max : ∀ x, x^3 - 3*x^2 + 1 = 0 → x ≤ a) :
  (⌊a^1788⌋ ≡ 0 [MOD 17]) ∧ (⌊a^1988⌋ ≡ 0 [MOD 17]) :=
by
  sorry

end problem_statement_l36_36122


namespace dot_product_is_negative_37_l36_36309

def vec1 : ℝ × ℝ × ℝ := (4, -3, 5)
def vec2 : ℝ × ℝ × ℝ := (-6, 1, -2)

theorem dot_product_is_negative_37 : 
  let dot_product := (vec1.1 * vec2.1) + (vec1.2 * vec2.2) + (vec1.3 * vec2.3)
  in dot_product = -37 :=
by
  let dot_product := (vec1.1 * vec2.1) + (vec1.2 * vec2.2) + (vec1.3 * vec2.3)
  show dot_product = -37,
  sorry

end dot_product_is_negative_37_l36_36309


namespace probability_divisor_of_12_l36_36608

theorem probability_divisor_of_12 (fair_die : True) (sides : Nat) (h : sides = 12) : 
  ∃ p : ℚ, p = 1 / 2 ∧ ∀ roll : Fin sides, is_divisor roll.val 12 ↔ roll.val ∈ {1, 2, 3, 4, 6, 12} → p = 1 / 2 :=
by sorry

def is_divisor (a b : Nat) : Prop := b % a = 0

end probability_divisor_of_12_l36_36608


namespace probability_of_rolling_divisor_of_12_l36_36597

def is_divisor (a b : ℕ) : Prop := b % a = 0

noncomputable def probability_divisor_12 : ℚ :=
  let divisors := { n | is_divisor n 12 ∧ n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} }
  let favorable_outcomes := (divisors : set ℕ).to_finset.card
  let total_outcomes := 12
  (favorable_outcomes : ℚ) / total_outcomes

theorem probability_of_rolling_divisor_of_12 : probability_divisor_12 = 1 / 2 := by
  sorry

end probability_of_rolling_divisor_of_12_l36_36597


namespace valid_square_numbers_l36_36299

noncomputable def is_valid_number (N P Q : ℕ) (q : ℕ) : Prop :=
  N = P * 10^q + Q ∧ N = 2 * P * Q

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem valid_square_numbers : 
  ∀ (N : ℕ), (∃ (P Q : ℕ) (q : ℕ), is_valid_number N P Q q) → is_perfect_square N :=
sorry

end valid_square_numbers_l36_36299


namespace probability_divisor_of_12_is_half_l36_36622

noncomputable def divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ d, n % d = 0) (Finset.range (n + 1))

def probability_divisor_of_12 : ℚ :=
  let total_outcomes := 12
  let favorable_outcomes := (divisors 12).card
  favorable_outcomes / total_outcomes

theorem probability_divisor_of_12_is_half :
  probability_divisor_of_12 = 1 / 2 :=
  by
    sorry

end probability_divisor_of_12_is_half_l36_36622


namespace granger_cisco_combined_spots_l36_36380

theorem granger_cisco_combined_spots :
  let R := 46
  let C := (R / 2) - 5
  let G := 5 * C
  G + C = 108 := by 
  let R := 46
  let C := (R / 2) - 5
  let G := 5 * C
  sorry

end granger_cisco_combined_spots_l36_36380


namespace length_CE_eq_6_l36_36830

-- Definitions of points and lengths.
variable (A B C D E : Type) 
variable [Geometry A B C D E]
variable (right_angle_ABE : right_angle A B E)
variable (right_angle_BCE : right_angle B C E)
variable (right_angle_CDE : right_angle C D E)
variable (angle_AEB_eq_60 : ∠ A E B = 60)
variable (angle_BEC_eq_60 : ∠ B E C = 60)
variable (angle_CED_eq_60 : ∠ C E D = 60)
variable (length_AE : length A E = 24)

-- Theorem to prove the length of CE.
theorem length_CE_eq_6 : length C E = 6 :=
sorry

end length_CE_eq_6_l36_36830


namespace area_of_sector_l36_36502

-- Define parameters given in the problem
def C : ℝ := 6  -- Circumference of the sector
def θ : ℝ := 1  -- Central angle in radians

-- Prove that the area of the sector is 2 cm^2
theorem area_of_sector : 
  (∃ r : ℝ, C = r + 2 * r ∧  θ = 1 ∧ let S := (r^2 * θ / 2) in S = 2) :=
sorry

end area_of_sector_l36_36502


namespace trig_identity_l36_36522

theorem trig_identity (α : Float) :
  (Float.cos (Float.pi / 4 - α) * Float.cos (α + Float.pi / 12) - Float.sin (Float.pi / 4 - α) * Float.sin (α + Float.pi / 12)) = (1 / 2 : Float) :=
  sorry

end trig_identity_l36_36522


namespace part_a_part_b_part_c_part_d_l36_36067

-- Define conditions for the cities and transports
variable {City : Type}
variable (connected_by_airline : City → City → Prop)
variable (connected_by_railway : City → City → Prop)
variable (is_connected : City → City → Prop := λ A B => connected_by_airline A B ∨ connected_by_railway A B)

-- Prove that we can choose one type of transport for part (a)
theorem part_a (∀ A B : City, is_connected A B) :
  ∃ (connected_by_airline_or_railway_only : City → City → Prop), 
    ∀ A B : City, connected_by_airline_or_railway_only A B :=
by
  sorry

-- Prove possibility for travel with no more than one transfer for part (b)
theorem part_b (∀ A B : City, is_connected A B) :
  ∃ city X : City, ∃ (connected_by_one_transfer : City → City → Prop),
    ∀ A : City, connected_by_one_transfer X A :=
by
  sorry

-- Show this property holds for every city for part (c)
theorem part_c (∀ A B : City, is_connected A B) :
  ∀ (X : City), ∃ (connected_by_one_transfer : City → City → Prop),
    ∀ A : City, connected_by_one_transfer X A :=
by
  sorry

-- Prove travel within two transfers for part (d)
theorem part_d (∀ A B : City, is_connected A B) :
  ∃ (connected_by_two_transfers : City → City → Prop),
    ∀ A B : City, connected_by_two_transfers A B :=
by
  sorry

end part_a_part_b_part_c_part_d_l36_36067


namespace sum_of_squares_of_coeffs_l36_36283

theorem sum_of_squares_of_coeffs :
  let p := 5 * (Polynomial.monomial 3 1 - Polynomial.monomial 2 3 + Polynomial.C 4)
  let q := 8 * (Polynomial.monomial 4 2 - Polynomial.monomial 3 1 + Polynomial.monomial 1 1)
  let r := p - q
  r.coeff 3 ^ 2 + r.coeff 2 ^ 2 + r.coeff 4 ^ 2 + r.coeff 1 ^ 2 + r.coeff 0 ^ 2 = 954 := sorry

end sum_of_squares_of_coeffs_l36_36283


namespace divide_by_3_result_l36_36562

-- Definitions
def n : ℕ := 4 * 12

theorem divide_by_3_result (h : n / 4 = 12) : n / 3 = 16 :=
by
  sorry

end divide_by_3_result_l36_36562


namespace number_of_factors_3_5_5_3_7_2_l36_36386

theorem number_of_factors_3_5_5_3_7_2 : 
  let n := 3^5 * 5^3 * 7^2 in
  ∃ (d : ℕ), (∀ k : ℕ, k > 0 → n % k = 0 → d ) ∧ d = 72 :=
by {
  sorry
}

end number_of_factors_3_5_5_3_7_2_l36_36386


namespace equilateral_triangle_perimeter_and_area_l36_36098

noncomputable def side_length : ℝ := 10

def perimeter (s : ℝ) : ℝ := 3 * s
def area (s : ℝ) : ℝ := (sqrt 3 / 4) * s^2

theorem equilateral_triangle_perimeter_and_area :
  perimeter side_length = 30 ∧ area side_length = 25 * sqrt 3 :=
by
  sorry

end equilateral_triangle_perimeter_and_area_l36_36098


namespace smaller_triangle_area_ratio_l36_36185

theorem smaller_triangle_area_ratio (ABC : Type) [triangle ABC] (A B C D E F : Point ABC)
  (hD : midpoint D A B) (hE : midpoint E B C) (hF : midpoint F C A) (T : ℝ) :
  let S := area_triangle A E F in
  S / T = 1 / 8 :=
sorry

end smaller_triangle_area_ratio_l36_36185


namespace monotonic_intervals_inequality_range_a_l36_36369

noncomputable def f (a x : ℝ) := a * log x - (a + 1) * x + 0.5 * x^2

theorem monotonic_intervals (a : ℝ) :
  (if a = 1 then ∀ x, 0 < x → f a x ∈ (0, ∞) ∧ ∀ u v, u < v → f a u ≤ f a v
   else if 0 < a ∧ a < 1 then
     (∀ x, 0 < x ∧ x < a → f a x ∈ (0, a) ∧ ∀ u v, u < v → f a u ≤ f a v) ∧
     (∀ x, a < x ∧ x < 1 → f a x ∈ (a, 1) ∧ ∀ u v, u < v → f a u ≥ f a v) ∧
     (∀ x, 1 < x → f a x ∈ (1, ∞) ∧ ∀ u v, u < v → f a u ≤ f a v)
   else if 1 < a then 
         (∀ x, 0 < x ∧ x < 1 → f a x ∈ (0, 1) ∧ ∀ u v, u < v → f a u ≤ f a v) ∧
     (∀ x, 1 < x ∧ x < a → f a x ∈ (1, a) ∧ ∀ u v, u < v → f a u ≥ f a v) ∧
     (∀ x, a < x → f a x ∈ (a, ∞) ∧ ∀ u v, u < v → f a u ≤ f a v)
   else if a ≤ 0 then 
     (∀ x, 0 < x ∧ x < 1 → f a x ∈ (0, 1) ∧ ∀ u v, u < v → f a u ≥ f a v) ∧
     (∀ x, 1 < x → f a x ∈ (1, ∞) ∧ ∀ u v, u < v → f a u ≤ f a v)
   else false) := sorry

theorem inequality_range_a (a : ℝ) (h : 0 < a) :
  (∀ x, 1 < x → f a x ≥ x^a - exp x + 0.5 * x^2 - a * x) → 
  a ∈ (0, exp 1] := sorry

end monotonic_intervals_inequality_range_a_l36_36369


namespace cylinder_problem_l36_36340

theorem cylinder_problem (r h : ℝ) (h1 : π * r^2 * h = 2) (h2 : 2 * π * r * h + 2 * π * r^2 = 12) :
  1 / r + 1 / h = 3 :=
sorry

end cylinder_problem_l36_36340


namespace factorize_expr_l36_36729

theorem factorize_expr (x : ℝ) : (x - 1) * (x + 3) + 4 = (x + 1) ^ 2 :=
by
  sorry

end factorize_expr_l36_36729


namespace y1_minus_y2_eq_five_l36_36467

theorem y1_minus_y2_eq_five
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : a^2 + b^2 = 25) :
  let y := λ x : ℝ, (a * x + b) / (x^2 + 1),
      y1 := (b + 5) / 2,
      y2 := (b - 5) / 2
  in y1 - y2 = 5 :=
by
  sorry

end y1_minus_y2_eq_five_l36_36467


namespace exists_prime_p_irreducible_poly_l36_36486

theorem exists_prime_p_irreducible_poly (a : ℕ → ℤ) (n : ℕ) (h : a n ≠ 0) : 
  ∃ p : ℕ, prime p ∧ ∀ g h : polynomial ℤ, (g * h = polynomial.C (p) + polynomial.X * polynomial.of_coeffs a) →
   (g.degree = 0 ∨ h.degree = 0) :=
sorry

end exists_prime_p_irreducible_poly_l36_36486


namespace sum_6n_is_correct_l36_36812

theorem sum_6n_is_correct {n : ℕ} (h : (5 * n * (5 * n + 1)) / 2 = (n * (n + 1)) / 2 + 200) :
  (6 * n * (6 * n + 1)) / 2 = 300 :=
by sorry

end sum_6n_is_correct_l36_36812


namespace mixing_ratios_l36_36974

theorem mixing_ratios (V : ℝ) (hV : 0 < V) :
  (4 * V / 5 + 7 * V / 10) / (V / 5 + 3 * V / 10) = 3 :=
by
  sorry

end mixing_ratios_l36_36974


namespace basket_weight_l36_36245

def weight_of_basket_alone (n_pears : ℕ) (weight_per_pear total_weight : ℚ) : ℚ :=
  total_weight - (n_pears * weight_per_pear)

theorem basket_weight :
  weight_of_basket_alone 30 0.36 11.26 = 0.46 := by
  sorry

end basket_weight_l36_36245


namespace proof_problem_equivalence_l36_36790

noncomputable def circle_equation : String :=
  let ρ_eq : String := "ρ = 2cos(θ + π/3)"
  let general_eq : String := "x^2 + y^2 - x + √3y = 0"
  general_eq

noncomputable def line_equation : String :=
  let point_P : (ℝ × ℝ) := (-1, 2)
  let inclination_angle : ℝ := 2 * π / 3
  let parametric_eq : String := "x = -1 - (1/2)t, y = 2 + (√3/2)t"
  parametric_eq

noncomputable def intersection_proof : ℝ :=
  let parametric_eq: String := "x = -1 - (1/2)t, y = 2 + (√3/2)t"
  let circle_eq: String := "x^2 + y^2 - x + √3y = 0"
  let intersection_value: ℝ := 6 + 2 * √3
  intersection_value

theorem proof_problem_equivalence :
  circle_equation = "x^2 + y^2 - x + √3y = 0" ∧
  line_equation = "x = -1 - (1/2)t, y = 2 + (√3/2)t" ∧
  intersection_proof = 6 + 2 * √3 :=
by 
  sorry

end proof_problem_equivalence_l36_36790


namespace combined_yells_l36_36138

def yells_at_obedient : ℕ := 12
def yells_at_stubborn (y_obedient : ℕ) : ℕ := 4 * y_obedient
def total_yells (y_obedient : ℕ) (y_stubborn : ℕ) : ℕ := y_obedient + y_stubborn

theorem combined_yells : total_yells yells_at_obedient (yells_at_stubborn yells_at_obedient) = 60 := 
by
  sorry

end combined_yells_l36_36138


namespace problem_statement_l36_36773

variable (x : ℝ)

-- Definitions based on the conditions
def a := 2005 * x + 2009
def b := 2005 * x + 2010
def c := 2005 * x + 2011

-- Assertion for the problem
theorem problem_statement : a ^ 2 + b ^ 2 + c ^ 2 - a * b - b * c - c * a = 3 := by
  sorry

end problem_statement_l36_36773


namespace scalene_triangle_angle_obtuse_l36_36822

theorem scalene_triangle_angle_obtuse (a b c : ℝ) 
  (h_scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_longest : a > b ∧ a > c)
  (h_obtuse_angle : a^2 > b^2 + c^2) : 
  ∃ A : ℝ, A = (Real.pi / 2) ∧ (b^2 + c^2 - a^2) / (2 * b * c) < 0 := 
sorry

end scalene_triangle_angle_obtuse_l36_36822


namespace friends_professions_l36_36883

theorem friends_professions :
  (¬ (Monsieur Pâtissier = baker) ∧
   ¬ (Monsieur Fleuriste = pastry_chef) ∧
   (Monsieur Pâtissier = pastry_chef) ∧
   ¬ (Monsieur Fleuriste = baker)) →
  ∃ (prof1 prof2 prof3 : Profession),
    prof1 = baker ∧ prof2 = pastry_chef ∧ prof3 = florist ∧
    person1 = Monsieur Pâtissier ∧ person2 = Monsieur Fleuriste ∧ person3 = Monsieur Boulanger :=
sorry

end friends_professions_l36_36883


namespace coefficient_x3_in_binomial_expansion_l36_36082

theorem coefficient_x3_in_binomial_expansion :
  nat.choose 50 3 = 19600 :=
by
  -- Proof goes here
  sorry

end coefficient_x3_in_binomial_expansion_l36_36082


namespace solve_inequality_l36_36941

theorem solve_inequality (x : ℝ) : x + 2 < 1 ↔ x < -1 := sorry

end solve_inequality_l36_36941


namespace min_sum_of_edges_is_1_over_11_l36_36868

noncomputable def min_sum_edges : ℝ :=
  let A := 0
  let I := 1
  ∀ (B C D E F G H : ℝ),
  let S := B^2 + D^2 + (B-E)^2 + (D-E)^2 + (B-C)^2 + (C-F)^2 + (D-G)^2 + (G-H)^2 + (E-1)^2 + (F-1)^2 + (H-1)^2
  S ≥ 0 ∧ S ≥ 1/11

theorem min_sum_of_edges_is_1_over_11 : min_sum_edges = 1/11 :=
sorry

end min_sum_of_edges_is_1_over_11_l36_36868


namespace sally_unspent_portion_l36_36158

theorem sally_unspent_portion (G : ℝ) :
  let P := 2 * G,
      balance_gold := (1 / 3) * G,
      balance_platinum := (2 / 7) * P,
      new_balance_platinum := balance_platinum + balance_gold,
      unspent_portion := P - new_balance_platinum in
  unspent_portion / P = (29 / 42) :=
by 
  -- Definitions to be used in the proof
  let P := 2 * G
  let balance_gold := (1 / 3) * G
  let balance_platinum := (2 / 7) * P
  let new_balance_platinum := balance_platinum + balance_gold
  let unspent_portion := P - new_balance_platinum
  
  -- Prove the statement
  sorry

end sally_unspent_portion_l36_36158


namespace find_v5_l36_36166

noncomputable def sequence (v : ℕ → ℝ) : Prop :=
  ∀ n, v (n + 2) = 3 * v (n + 1) + v n + 1

theorem find_v5 :
  ∃ (v : ℕ → ℝ), sequence v ∧ v 3 = 11 ∧ v 6 = 242 ∧ v 5 = 73.5 :=
by
  sorry

end find_v5_l36_36166


namespace angle_ABG_in_heptagon_l36_36151

noncomputable def regular_heptagon_interior_angle (n : ℕ) : ℝ :=
  ((n - 2) * 180) / n

theorem angle_ABG_in_heptagon (ABCDEFG : Set (Fin 7 → ℝ × ℝ))
  (regular_heptagon : is_regular_polygon ABCDEFG 7)
  (A B G : (Fin 7 → ℝ × ℝ)) (h1: A ∈ ABCDEFG) (h2: B ∈ ABCDEFG) (h3: G ∈ ABCDEFG)
  (hAB : distance A B = distance A G) :
  angle A B G = 25.715 :=
by
  let interior_angle := regular_heptagon_interior_angle 7
  have h4 : interior_angle = (5 * 180) / 7 := rfl
  have h5 : 2 * angle A B G + interior_angle = 180 := sorry
  have h6 : 2 * angle A B G = 180 - interior_angle := sorry
  have h7 : angle A B G = (180 - interior_angle) / 2 := sorry
  have h8 : 180 - interior_angle = 51.43 := sorry
  have h9 : (180 - interior_angle) / 2 = 25.715 := sorry
  exact h9

end angle_ABG_in_heptagon_l36_36151


namespace circle_center_transformation_l36_36692

def original_center : ℤ × ℤ := (3, -4)

def reflect_x_axis (p : ℤ × ℤ) : ℤ × ℤ := (p.1, -p.2)

def translate_right (p : ℤ × ℤ) (d : ℤ) : ℤ × ℤ := (p.1 + d, p.2)

def final_center : ℤ × ℤ := (8, 4)

theorem circle_center_transformation :
  translate_right (reflect_x_axis original_center) 5 = final_center :=
by
  sorry

end circle_center_transformation_l36_36692


namespace co2_concentration_ventilation_time_l36_36953

-- Definitions from the conditions
def initial_co2_concentration := 0.2 -- 2000 ppm is equivalent to 0.2%
def co2_concentration_formula (t : ℝ) (λ : ℝ) : ℝ := 0.05 + λ * Real.exp (-t / 9)
def national_standard_co2 := 0.1 -- 1000 ppm is equivalent to 0.1%

-- Given the above definitions, we need to prove the time for the CO2 concentration to meet the national standard
theorem co2_concentration_ventilation_time : 
  ∃ t ≥ 0, co2_concentration_formula t 0.15 ≤ national_standard_co2 :=
by
  -- Since lambda is given through the solution steps as 0.15, we use it directly.
  sorry

end co2_concentration_ventilation_time_l36_36953


namespace area_triangle_ABC_correct_l36_36478

noncomputable def area_triangle_ABC (r OA AB : ℝ) (h1 : r = 12) (h2 : OA = 15) (h3 : AB = 5) :=
  let α := arcCos (r / OA)
  let β := arcCos (r / (OA + AB))
  let sin_diff := Real.sin α * Real.cos β - Real.cos α * Real.sin β
  (1 / 2) * AB * ((AB * r) / sin_diff) * Real.sin (Real.arcCos (1))

theorem area_triangle_ABC_correct :
  area_triangle_ABC 12 15 5 rfl rfl rfl = 150 / 7 := by
  sorry

end area_triangle_ABC_correct_l36_36478


namespace sum_formula_l36_36763

open Nat

/-- The sequence a_n defined as (-1)^n * (2 * n - 1) -/
def a_n (n : ℕ) : ℤ :=
  (-1) ^ n * (2 * n - 1)

/-- The partial sum S_n of the first n terms of the sequence a_n -/
def S_n : ℕ → ℤ
| 0     => 0
| (n+1) => S_n n + a_n (n + 1)

/-- The main theorem: For all n in natural numbers, S_n = (-1)^n * n -/
theorem sum_formula (n : ℕ) : S_n n = (-1) ^ n * n := by
  sorry

end sum_formula_l36_36763


namespace find_h_l36_36860

def f (x : ℝ) : ℝ := 3 * x - 1
def g (x : ℝ) : ℝ := 2 * x + 3

theorem find_h (h : ℝ → ℝ) : (∀ x : ℝ, f (h x) = g x) → (∀ x : ℝ, h x = (2 * x + 4) / 3) :=
by
  intro hyp
  funext x
  sorry

end find_h_l36_36860


namespace least_positive_int_l36_36218

theorem least_positive_int (n : ℕ) (h1 : n > 1)
  (h2 : ∀ k ∈ {2, 3, 5, 7, 8, 9, 10}, (n - 1) % k = 0) :
  n = 2521 := 
sorry

end least_positive_int_l36_36218


namespace midpoint_interval_a_l36_36331

theorem midpoint_interval_a : 
  ∀ a : ℝ, (a > (1/3)) → 
    (∃ x1 x2 : ℝ, 
      (|x1 - 3| = a * x1 - 1) ∧ 
      (|x2 - 3| = a * x2 - 1) ∧ 
      (x1 ≠ x2)) ∧ 
    (Real.midpoint ((1/3) : ℝ) (∞ : ℝ) = 0.667) := 
by
  sorry

end midpoint_interval_a_l36_36331


namespace calculate_value_l36_36676

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

variable (f : ℝ → ℝ)

axiom h : odd_function f
axiom h1 : increasing_on_interval f 3 7
axiom h2 : f 3 = -1
axiom h3 : f 6 = 8

theorem calculate_value : 2 * f (-6) + f (-3) = -15 := by
  sorry

end calculate_value_l36_36676


namespace basswood_figurines_l36_36270

variables (B : ℕ)
variables (nbasswood nbutternut naspen figurines_per_butternut figurines_per_aspen total_figurines : ℕ)

-- Defining the given conditions
def condition_1 := figurines_per_butternut = 4
def condition_2 := figurines_per_aspen = 2 * B
def condition_3 := nbasswood = 15 ∧ nbutternut = 20 ∧ naspen = 20
def condition_4 := total_figurines = 245

-- Prove that the number of figurines a block of basswood can create is 3
theorem basswood_figurines : 
  condition_1 ∧ condition_2 ∧ condition_3 ∧ condition_4 → 15 * B + 20 * figurines_per_butternut + 20 * figurines_per_aspen = total_figurines → B = 3 :=
by
  intro h ⟨hb1, hb2, hb3⟩ hf,
  sorry

end basswood_figurines_l36_36270


namespace option_a_correct_l36_36131

noncomputable def M : Set ℝ := { y | ∃ x ∈ ℝ, y = 2^x }
noncomputable def S : Set ℝ := { x | ∃ y, y = log (x - 1) }

theorem option_a_correct :
  M ∪ S = M :=
  sorry

end option_a_correct_l36_36131


namespace flower_beds_fraction_l36_36999

open Real

noncomputable def parkArea (a b h : ℝ) := (a + b) / 2 * h
noncomputable def triangleArea (a : ℝ) := (1 / 2) * a ^ 2

theorem flower_beds_fraction 
  (a b h : ℝ) 
  (h_a: a = 15) 
  (h_b: b = 30) 
  (h_h: h = (b - a) / 2) :
  (2 * triangleArea h) / parkArea a b h = 1 / 4 := by 
  sorry

end flower_beds_fraction_l36_36999


namespace sum_floors_arithmetic_sequence_eq_l36_36705

noncomputable def sum_floors_arithmetic_sequence (a d : ℝ) (n : ℕ) : ℤ :=
∑ k in finset.range n, ⌊a + k * d⌋

theorem sum_floors_arithmetic_sequence_eq :
  sum_floors_arithmetic_sequence (-2) 0.7 144 = 6941 :=
by sorry

end sum_floors_arithmetic_sequence_eq_l36_36705


namespace simplify_expression_l36_36176

theorem simplify_expression : (8^(1/3) / 8^(1/6)) = 8^(1/6) :=
by
  sorry

end simplify_expression_l36_36176


namespace smallest_n_for_Q_n_l36_36305

theorem smallest_n_for_Q_n (Q : ℕ → ℚ) (hQ : ∀ n, Q n = 1 / (n * (n^2 + 1))) :
  ∃ n, Q n < 1 / 3000 ∧ n = 15 :=
by {
  use 15,
  split,
  {
    rw hQ,
    norm_num,
    split,
    -- proving Q n < 1 / 3000 here
    sorry
  },
  refl
}

end smallest_n_for_Q_n_l36_36305


namespace probability_of_divisor_of_12_is_one_half_l36_36633

theorem probability_of_divisor_of_12_is_one_half :
  let divisors := {1, 2, 3, 4, 6, 12} in
  let total_outcomes := finset.range 12 in
  (divisors.card : ℚ) / (total_outcomes.card : ℚ) = 1 / 2 :=
by
  let divisors := finset.from_list [1, 2, 3, 4, 6, 12]
  let total_outcomes := finset.range 12
  have h1 : divisors.card = 6 := by sorry
  have h2 : total_outcomes.card = 12 := by sorry
  rw [←h1, ←h2]
  norm_num
  sorry

end probability_of_divisor_of_12_is_one_half_l36_36633


namespace sqrt_21_is_11th_term_l36_36791

theorem sqrt_21_is_11th_term :
  ∃ n : ℕ, ( (n = 11) ∧ ( a_n = λ n, real.sqrt(2 * n - 1) ) ∧ ( a_n 11 = real.sqrt 21 )) :=
by
  sorry

end sqrt_21_is_11th_term_l36_36791


namespace solution_set_l36_36495

def solve_inequality (a : ℝ) (x : ℝ) : Prop :=
  ax^2 - (2*a + 1)*x + 2 < 0

theorem solution_set (a : ℝ) :
  ∀ x, solve_inequality a x ↔
    (a = 0 → 2 < x) ∧
    (0 < a ∧ a < 1/2 → 2 < x ∧ x < 1/a) ∧
    (a = 1/2 → false) ∧
    (a > 1/2 → 1/a < x ∧ x < 2) ∧
    (a < 0 → (2 < x ∨ x < 1/a)) :=
by sorry

end solution_set_l36_36495


namespace percent_value_in_quarters_l36_36229

theorem percent_value_in_quarters
  (num_dimes num_quarters num_nickels : ℕ)
  (value_dime value_quarter value_nickel : ℕ)
  (h_dimes : num_dimes = 70)
  (h_quarters : num_quarters = 30)
  (h_nickels : num_nickels = 40)
  (h_value_dime : value_dime = 10)
  (h_value_quarter : value_quarter = 25)
  (h_value_nickel : value_nickel = 5) :
  ((num_quarters * value_quarter : ℕ) * 100 : ℚ) / 
  (num_dimes * value_dime + num_quarters * value_quarter + num_nickels * value_nickel) = 45.45 :=
by
  sorry

end percent_value_in_quarters_l36_36229


namespace chocolate_ice_cream_l36_36903

-- Define the number of people who ordered ice cream, vanilla, and chocolate
variable (total_people vanilla_people chocolate_people : ℕ)

-- Define the conditions as Lean constraints
def condition1 : Prop := total_people = 220
def condition2 : Prop := vanilla_people = (20 * total_people) / 100
def condition3 : Prop := vanilla_people = 2 * chocolate_people

-- State the theorem to prove the number of people who ordered chocolate ice cream
theorem chocolate_ice_cream (h1 : condition1) (h2 : condition2) (h3 : condition3) : chocolate_people = 22 :=
sorry

end chocolate_ice_cream_l36_36903


namespace granger_cisco_combined_spots_l36_36381

theorem granger_cisco_combined_spots :
  let R := 46
  let C := (R / 2) - 5
  let G := 5 * C
  G + C = 108 := by 
  let R := 46
  let C := (R / 2) - 5
  let G := 5 * C
  sorry

end granger_cisco_combined_spots_l36_36381


namespace total_distance_walked_l36_36049

def distance_walked_by_Henry (limit: ℝ) : ℝ :=
  (2 : ℝ) / (1 - (1 : ℝ) / 3)

theorem total_distance_walked : 
  limit > 0.1 → distance_walked_by_Henry limit = 2.777 :=
by
  sorry

end total_distance_walked_l36_36049


namespace problem_solution_l36_36748

variables {Point : Type} [PlanarGeometry Point]
variables {Line : Type} [LineGeometry Line Point]

def α : Plane Point := sorry
def m : Line := sorry
def n : Line := sorry

axiom A : ∀ α m n, (EqualAngles α m n) → Parallel m n
axiom B : ∀ α m n, (ParallelToPlane m α) (ParallelToPlane n α) → Parallel m n
axiom C : ∀ α m n, (PerpendicularToPlane m α) (PerpendicularToLine m n) → ParallelToPlane n α
axiom D : ∀ α m n, (SubsetOfPlane m α) (ParallelToPlane n α) → Parallel m n

theorem problem_solution : ∀ α m n, (SubsetOfPlane m α) (ParallelToPlane n α) → Parallel m n :=
by
  sorry

end problem_solution_l36_36748


namespace correct_calculation_result_l36_36257

theorem correct_calculation_result :
  ∃ x : ℕ, 6 * x = 42 ∧ 3 * x = 21 :=
by
  sorry

end correct_calculation_result_l36_36257


namespace probability_b_gt_a_eq_one_fifth_l36_36487

noncomputable def probability_b_greater_than_a : ℚ :=
(∑ a in {1, 2, 3, 4, 5}, ∑ b in {1, 2, 3}, if b > a then 1 else 0 : ℕ) / (5 * 3 : ℕ)

theorem probability_b_gt_a_eq_one_fifth :
  probability_b_greater_than_a = 1/5 :=
sorry

end probability_b_gt_a_eq_one_fifth_l36_36487


namespace probability_sum_of_relatively_prime_integers_eq_371_l36_36213

theorem probability_sum_of_relatively_prime_integers_eq_371 :
  ∃ (m n : ℕ), Nat.Coprime m n ∧ (m + n = 371) ∧ 
  let P := sorry in -- Probability that each delegate sits next to at least one delegate from another country
  P = (m : ℚ) / (n : ℚ) :=
begin
  sorry
end

end probability_sum_of_relatively_prime_integers_eq_371_l36_36213


namespace find_smaller_number_l36_36980

theorem find_smaller_number (x y : ℕ) (h1 : y = 3 * x) (h2 : x + y = 124) : x = 31 := 
by 
  -- Proof would go here
  sorry

end find_smaller_number_l36_36980


namespace solve_quadratic_1_solve_quadratic_2_l36_36493

-- Equation 1
theorem solve_quadratic_1 (x : ℝ) : 2 * x^2 + 6 * x + 3 = 0 → x = (-3 + sqrt 3) / 2 ∨ x = (-3 - sqrt 3) / 2 := by
  sorry

-- Equation 2
theorem solve_quadratic_2 (x : ℝ) : (x + 2)^2 = 3 * (x + 2) → x = -2 ∨ x = 1 := by
  sorry

end solve_quadratic_1_solve_quadratic_2_l36_36493


namespace exists_m_n_fractional_part_l36_36861

theorem exists_m_n_fractional_part 
  (p : ℕ) (hp : Nat.Prime p) 
  (s : ℕ) (hs : 0 < s ∧ s < p) :
  (∃ m n : ℕ, 0 < m ∧ m < n ∧ n < p ∧ 
              (frac ((m * s : ℚ) / p) < frac ((n * s : ℚ) / p) ∧ 
              frac ((n * s : ℚ) / p) < (s / p : ℚ))) ↔ ¬ s ∣ (p - 1) :=
sorry

end exists_m_n_fractional_part_l36_36861


namespace study_time_l36_36876

theorem study_time (n_mcq n_fitb : ℕ) (t_mcq t_fitb : ℕ) (total_minutes_per_hour : ℕ) 
  (h1 : n_mcq = 30) (h2 : n_fitb = 30) (h3 : t_mcq = 15) (h4 : t_fitb = 25) (h5 : total_minutes_per_hour = 60) : 
  n_mcq * t_mcq + n_fitb * t_fitb = 20 * total_minutes_per_hour := 
by 
  -- This is a placeholder for the proof
  sorry

end study_time_l36_36876


namespace gabrielle_saw_20_percent_more_l36_36225

-- Define the number of birds seen by Gabrielle and Chase
def birds_seen_by_gabrielle : ℕ := 5 + 4 + 3
def birds_seen_by_chase : ℕ := 2 + 3 + 5

-- Define the correct answer as a percentage
def percentage_increase (a b : ℕ) : ℝ := ((a - b).toReal / b.toReal) * 100

-- Statement asserting that Gabrielle saw 20% more birds than Chase
theorem gabrielle_saw_20_percent_more : percentage_increase birds_seen_by_gabrielle birds_seen_by_chase = 20 := by
  sorry

end gabrielle_saw_20_percent_more_l36_36225


namespace expected_value_is_0_point_25_l36_36266

-- Define the probabilities and earnings
def prob_roll_1 := 1/4
def earning_1 := 4
def prob_roll_2 := 1/4
def earning_2 := -3
def prob_roll_3_to_6 := 1/8
def earning_3_to_6 := 0

-- Define the expected value calculation
noncomputable def expected_value : ℝ := 
  (prob_roll_1 * earning_1) + 
  (prob_roll_2 * earning_2) + 
  (prob_roll_3_to_6 * earning_3_to_6) * 4  -- For 3, 4, 5, and 6

-- The theorem to be proved
theorem expected_value_is_0_point_25 : expected_value = 0.25 := by
  sorry

end expected_value_is_0_point_25_l36_36266


namespace hyperbola_eccentricity_l36_36928

def hyperbola_eq (x y : ℝ) : Prop := 16 * x^2 - 9 * y^2 = 144

def a : ℝ := 3
def b : ℝ := 4
def c : ℝ := real.sqrt (a^2 + b^2)
def e : ℝ := c / a

theorem hyperbola_eccentricity :
  (∀ x y, hyperbola_eq x y) →
  e = 5 / 3 :=
by
  sorry

end hyperbola_eccentricity_l36_36928


namespace largest_number_proof_l36_36313

/-- 
The largest natural number that does not end in zero and decreases by an integer factor 
when one (not the first) digit is removed.
-/
def largest_number_decreasing_by_factor : ℕ := 
  let x := 8
  let a := 1
  let c := 625
  let n := 1
  let r := 5
  let number := 10^(n+1) * a + 10^n * x + c
  { number | number ∉ [0], number % 10 ≠ 0, (r=5), 2 ≤ r ≤ 19 }

theorem largest_number_proof :
  largest_number_decreasing_by_factor = 180625 :=
sorry

end largest_number_proof_l36_36313


namespace labourer_savings_l36_36499

theorem labourer_savings
  (monthly_expenditure_first_6_months : ℕ)
  (monthly_expenditure_next_4_months : ℕ)
  (monthly_income : ℕ)
  (total_expenditure_first_6_months : ℕ)
  (total_income_first_6_months : ℕ)
  (debt_incurred : ℕ)
  (total_expenditure_next_4_months : ℕ)
  (total_income_next_4_months : ℕ)
  (money_saved : ℕ) :
  monthly_expenditure_first_6_months = 85 →
  monthly_expenditure_next_4_months = 60 →
  monthly_income = 78 →
  total_expenditure_first_6_months = 6 * monthly_expenditure_first_6_months →
  total_income_first_6_months = 6 * monthly_income →
  debt_incurred = total_expenditure_first_6_months - total_income_first_6_months →
  total_expenditure_next_4_months = 4 * monthly_expenditure_next_4_months →
  total_income_next_4_months = 4 * monthly_income →
  money_saved = total_income_next_4_months - (total_expenditure_next_4_months + debt_incurred) →
  money_saved = 30 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end labourer_savings_l36_36499


namespace problem_solution_l36_36322

theorem problem_solution (x : ℕ) (h : x = 3) : x + x * x^(x^2) = 59052 :=
by
  rw [h]
  -- The condition is now x = 3
  let t := 3 + 3 * 3^(3^2)
  have : t = 59052 := sorry
  exact this

end problem_solution_l36_36322


namespace measure_of_angle_C_range_of_expression_l36_36814

variables (A B C a b c : ℝ)
hypothesis (H : c * sin A = a * cos C)
hypothesis (range_A : 0 < A ∧ A < 3 * π / 4)

theorem measure_of_angle_C (h : c * sin A = a * cos C) : C = π / 4 :=
sorry

theorem range_of_expression (A B C : ℝ) (h : 0 < A ∧ A < 3 * π / 4) (hC : C = π / 4) :
  (sqrt 3 * sin A - cos (B + C)) ∈ Set.Icc ((sqrt 6 - sqrt 2) / 2) 2 :=
sorry

end measure_of_angle_C_range_of_expression_l36_36814


namespace probability_divisor_of_12_l36_36626

/-- Definition of divisors of a number -/
def divisors (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (λ d => n % d = 0)

theorem probability_divisor_of_12 :
  let divisors_of_12 := divisors 12
  Finset.card divisors_of_12 = 6 →
  (∃ (P : ℚ), P = 1/2) :=
by
  intro h
  exists 1/2
  sorry

end probability_divisor_of_12_l36_36626


namespace faster_train_speed_l36_36553

theorem faster_train_speed (distance : ℕ) (time : ℕ) (slower_train_speed : ℕ) 
  (h1 : distance = 536) (h2 : time = 4) (h3 : slower_train_speed = 60) : 
  ∃ faster_train_speed : ℕ, faster_train_speed = 74 :=
by
  let sum_of_speeds := distance / time
  let faster_train_speed := sum_of_speeds - slower_train_speed
  use faster_train_speed
  have hsum_of_speeds : sum_of_speeds = 134 := by 
    rw [h1, h2]
    calc 536 / 4 = 134 : rfl
  have hfaster_train_speed : faster_train_speed = 74 := by 
    rw [hsum_of_speeds, h3]
    calc 134 - 60 = 74 : rfl
  exact hfaster_train_speed

end faster_train_speed_l36_36553


namespace sales_volume_correctness_l36_36525

theorem sales_volume_correctness
  (x y : ℝ)
  (lin_rel : ∀ x : ℝ, y = -10 * x + 200) :
  (x = 10) → (y ≈ 100) :=
by
  intro h1
  subst h1
  have h2 : y = -10 * 10 + 200 := lin_rel 10
  linarith
  sorry

end sales_volume_correctness_l36_36525


namespace geometric_sequence_condition_l36_36100

noncomputable def is_geometric {α : Type*} [comm_ring α] (a : ℕ → α) :=
  ∃ (r : α), ∀ (n : ℕ), a (n + 1) = r * a n

theorem geometric_sequence_condition {a : ℕ → ℝ} :
  (∀ n ≥ 2, a n ^ 2 = a (n-1) * a (n+1)) ↔
  (∀ n : ℕ, a n ≠ 0 → is_geometric a) :=
begin
  sorry
end

end geometric_sequence_condition_l36_36100


namespace probability_floor_sqrt_even_l36_36915

/-- Suppose x and y are chosen randomly and uniformly from (0,1). The probability that
    ⌊√(x/y)⌋ is even is 1 - π²/24. -/
theorem probability_floor_sqrt_even (x y : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) :
  (1 - Real.pi ^ 2 / 24) = sorry :=
sorry

end probability_floor_sqrt_even_l36_36915


namespace area_of_triangle_AOB_calculate_distance_MN_l36_36242

def area_triangle_AOB (θ₁ θ₂ : ℝ) (ρ₁ ρ₂ : ℝ) (AOB_angle : ℝ) : ℝ :=
  0.5 * ρ₁ * ρ₂ * Real.sin AOB_angle

theorem area_of_triangle_AOB :
  let θ₁ := π / 3,
  let θ₂ := 2 * π / 3,
  let ρ₁ := 4 / Real.sqrt 13,
  let ρ₂ := 4 / Real.sqrt 13,
  let AOB_angle := π / 3,
  area_triangle_AOB θ₁ θ₂ ρ₁ ρ₂ AOB_angle = 4 * Real.sqrt 3 / 13 :=
by sorry

def distance_MN (t1 t2 : ℝ) : ℝ :=
  Real.abs (t1 - t2)

theorem calculate_distance_MN :
  let t1 := (-Real.sqrt 2 / 5 + Real.sqrt (2 / 5^2 + 6 / 5)) / 10,
  let t2 := (-Real.sqrt 2 / 5 - Real.sqrt (2 / 5^2 + 6 / 5)) / 10,
  distance_MN t1 t2 = 4 * Real.sqrt 2 / 5 :=
by sorry

end area_of_triangle_AOB_calculate_distance_MN_l36_36242


namespace solution_of_inequality_system_l36_36531

-- Definitions derived from the conditions in the problem
def inequality1 (x : ℝ) : Prop := 3 * x - 1 ≥ x + 1
def inequality2 (x : ℝ) : Prop := x + 4 > 4 * x - 2
def solution_set (x : ℝ) : Prop := 1 ≤ x ∧ x < 2

-- The Lean 4 statement for the proof problem
theorem solution_of_inequality_system (x : ℝ) : inequality1 x ∧ inequality2 x ↔ solution_set x := by
  sorry

end solution_of_inequality_system_l36_36531


namespace triangle_external_angles_l36_36886

theorem triangle_external_angles 
  (ABC : Triangle)
  (P Q R : Point)
  (hPBC : angle P B C = 45)
  (hBCP : angle B C P = 30)
  (hCAQ : angle C A Q = 45)
  (hQCA : angle Q C A = 30)
  (hABR : angle A B R = 15)
  (hBAR : angle B A R = 15) :
  angle Q R P = 90 ∧ dist Q R = dist R P := 
sorry

end triangle_external_angles_l36_36886


namespace hitting_at_least_7_rings_hitting_fewer_than_8_rings_l36_36656

-- Definitions of the events and their probabilities
def P_A10 : ℝ := 0.20
def P_A9 : ℝ := 0.22
def P_A8 : ℝ := 0.25
def P_A7 : ℝ := 0.28

-- Probability of hitting at least 7 rings
def P_A : ℝ := P_A10 + P_A9 + P_A8 + P_A7

theorem hitting_at_least_7_rings :
  (P_A = 0.95) :=
by
  -- proof goes here
  sorry

-- Probability of hitting fewer than 8 rings
def P_notA : ℝ := 1 - P_A
def P_B : ℝ := P_A7 + P_notA

theorem hitting_fewer_than_8_rings :
  (P_B = 0.33) :=
by
  -- proof goes here
  sorry

end hitting_at_least_7_rings_hitting_fewer_than_8_rings_l36_36656


namespace probability_of_divisor_of_12_is_one_half_l36_36635

theorem probability_of_divisor_of_12_is_one_half :
  let divisors := {1, 2, 3, 4, 6, 12} in
  let total_outcomes := finset.range 12 in
  (divisors.card : ℚ) / (total_outcomes.card : ℚ) = 1 / 2 :=
by
  let divisors := finset.from_list [1, 2, 3, 4, 6, 12]
  let total_outcomes := finset.range 12
  have h1 : divisors.card = 6 := by sorry
  have h2 : total_outcomes.card = 12 := by sorry
  rw [←h1, ←h2]
  norm_num
  sorry

end probability_of_divisor_of_12_is_one_half_l36_36635


namespace shaded_region_area_l36_36699

noncomputable def radius1 := 4
noncomputable def radius2 := 5
noncomputable def distance := radius1 + radius2
noncomputable def large_radius := radius2 + distance / 2

theorem shaded_region_area :
  ∃ (A : ℝ), A = (π * large_radius ^ 2) - (π * radius1 ^ 2) - (π * radius2 ^ 2) ∧
  A = 49.25 * π :=
by
  sorry

end shaded_region_area_l36_36699


namespace probability_divisor_of_12_l36_36630

/-- Definition of divisors of a number -/
def divisors (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (λ d => n % d = 0)

theorem probability_divisor_of_12 :
  let divisors_of_12 := divisors 12
  Finset.card divisors_of_12 = 6 →
  (∃ (P : ℚ), P = 1/2) :=
by
  intro h
  exists 1/2
  sorry

end probability_divisor_of_12_l36_36630


namespace chocolate_ice_cream_l36_36902

-- Define the number of people who ordered ice cream, vanilla, and chocolate
variable (total_people vanilla_people chocolate_people : ℕ)

-- Define the conditions as Lean constraints
def condition1 : Prop := total_people = 220
def condition2 : Prop := vanilla_people = (20 * total_people) / 100
def condition3 : Prop := vanilla_people = 2 * chocolate_people

-- State the theorem to prove the number of people who ordered chocolate ice cream
theorem chocolate_ice_cream (h1 : condition1) (h2 : condition2) (h3 : condition3) : chocolate_people = 22 :=
sorry

end chocolate_ice_cream_l36_36902


namespace solution_set_of_inequality_l36_36195

theorem solution_set_of_inequality (x : ℝ) : -2 * x - 1 < 3 ↔ x > -2 := 
by 
  sorry

end solution_set_of_inequality_l36_36195


namespace question_A_question_B_question_D_l36_36761

-- Given conditions
variables {R : Type*} [linear_ordered_field R]

def f (x : R) : R
def f_property (x y : R) : f (x + y) = f x * f y
def f_pos (x : R) (h : x > 0) : f x > 1

-- Proof of the statements
theorem question_A : f 0 = 1 :=
sorry

theorem question_B : ∀ x : R, f x > 0 :=
sorry

theorem question_D : f (1/2) = 2 → (∀ x y : R, f ((x * ln y - x * ln x - a * y) / x) ≤ 4) → a ≥ 1 / (exp 2) :=
sorry

end question_A_question_B_question_D_l36_36761


namespace constant_term_is_10_sum_of_binomial_coefficients_of_odd_terms_is_16_sum_of_all_coefficients_is_0_l36_36738

noncomputable def binom_expansion (x : ℝ) : ℝ := (x^2 - 1/(x^3))^5

theorem constant_term_is_10 :
  (∃ r : ℕ, (10 - 5 * r = 0) ∧ nat.choose 5 r = 10) :=
begin
  use 2,
  split,
  { norm_num },
  { norm_num }
end

theorem sum_of_binomial_coefficients_of_odd_terms_is_16 :
  (∑ r in finset.range 6, if odd r then nat.choose 5 r else 0) = 16 :=
by
  sorry

theorem sum_of_all_coefficients_is_0 :
  binom_expansion 1 = 0 :=
by
  sorry

end constant_term_is_10_sum_of_binomial_coefficients_of_odd_terms_is_16_sum_of_all_coefficients_is_0_l36_36738


namespace study_time_l36_36875

theorem study_time (n_mcq n_fitb : ℕ) (t_mcq t_fitb : ℕ) (total_minutes_per_hour : ℕ) 
  (h1 : n_mcq = 30) (h2 : n_fitb = 30) (h3 : t_mcq = 15) (h4 : t_fitb = 25) (h5 : total_minutes_per_hour = 60) : 
  n_mcq * t_mcq + n_fitb * t_fitb = 20 * total_minutes_per_hour := 
by 
  -- This is a placeholder for the proof
  sorry

end study_time_l36_36875


namespace find_GH_approx_l36_36417

noncomputable def GH_triangle_proof : Prop :=
  (∀ (G H I : Type) [is_right_triangle GHI],
    ∃ (GH : ℝ),
      ∠G = 40 ∧ ∠H = 90 ∧ IH = 12 ∧ tan(40 * π / 180) ≈ 0.8391 ∧ GH ≈ 14.3)

theorem find_GH_approx (G H I : Type) [is_right_triangle GHI]
  (angle_G : ∠G = 40)
  (angle_H : ∠H = 90)
  (IH_length : IH = 12)
  (tan_value_approx :  tan(40 * π / 180) ≈ 0.8391) :
  ∃ (GH : ℝ), GH ≈ 14.3 :=
sorry

end find_GH_approx_l36_36417


namespace sqrt_fraction_addition_l36_36284

theorem sqrt_fraction_addition :
  (Real.sqrt ((25 : ℝ) / 36 + 16 / 9)) = Real.sqrt 89 / 6 := by
  sorry

end sqrt_fraction_addition_l36_36284


namespace obtuse_angles_in_convex_octagon_l36_36818

theorem obtuse_angles_in_convex_octagon :
  ∀ (octagon : Type) [convex_polygon octagon] [num_sides octagon = 8],
  minimum_num_obtuse_interior_angles octagon = 5 :=
sorry

end obtuse_angles_in_convex_octagon_l36_36818


namespace trajectory_equation_line_equation_l36_36011

-- Proof Problem 1
theorem trajectory_equation (x y : ℝ) :
  let A := (-real.sqrt 2, 0)
  let B := (real.sqrt 2, 0)
  (y / (x + real.sqrt 2)) * (y / (x - real.sqrt 2)) = -1/2 → (x^2 / 2) + y^2 = 1 :=
sorry

-- Proof Problem 2
theorem line_equation (k : ℝ) :
  let curve (x y : ℝ) := (x^2 / 2) + y^2 = 1
  let line_eq (x y : ℝ) := y = k * x + 1
  (∃ x y, curve x y ∧ line_eq x y) →
  let midpoint_x := (-2 * k) / (2 * k^2 + 1)
  let midpoint_y := (1 / (2 * k^2 + 1))
  midpoint_x + 2 * midpoint_y = 0 →
  k = 1 →
  line_eq x y = y = x + 1 :=
sorry

end trajectory_equation_line_equation_l36_36011


namespace probability_divisor_of_12_l36_36639

theorem probability_divisor_of_12 : 
  (nat.factorization 12).keys = {1, 2, 3, 4, 6, 12} → 
  (∃ die_side, die_side ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) → 
  (∃ favorable_outcomes, favorable_outcomes ∈ {1, 2, 3, 4, 6, 12}) → 
  ∃ probability, probability = (favorable_outcomes.card : ℚ) / (die_side.card : ℚ) ∧ probability = 1/2 :=
by 
  sorry

end probability_divisor_of_12_l36_36639


namespace sum_of_possible_M_l36_36187

theorem sum_of_possible_M : 
  ∀ (x y z M : ℕ), 
    M = x * y * z ∧ 
    M = 8 * (x + y + z) ∧ 
    z = 2 * (x + y) ∧ 
    x > 0 ∧ 
    y > 0 ∧ 
    z > 0 → 
    (M = 272 ∨ M = 160 ∨ M = 128) → 
    ∑ possible_M, possible_M = 560 := 
by 
  intros x y z M h₁ h₂ h₃ h₄ h₅ h₆ hM 
  sorry

end sum_of_possible_M_l36_36187


namespace height_of_building_l36_36645

noncomputable def flagpole_height : ℝ := 18
noncomputable def flagpole_shadow : ℝ := 45
noncomputable def building_shadow : ℝ := 55
noncomputable def building_height : ℝ := 22

theorem height_of_building :
  let ratio_flagpole := flagpole_height / flagpole_shadow in
  let ratio_building := building_height / building_shadow in
  ratio_flagpole = ratio_building :=
by
  let ratio_flagpole := flagpole_height / flagpole_shadow
  let ratio_building := building_height / building_shadow
  have h_eq : building_height = 22 := rfl
  rw h_eq
  sorry

end height_of_building_l36_36645


namespace probability_divisor_of_12_l36_36638

theorem probability_divisor_of_12 : 
  (nat.factorization 12).keys = {1, 2, 3, 4, 6, 12} → 
  (∃ die_side, die_side ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) → 
  (∃ favorable_outcomes, favorable_outcomes ∈ {1, 2, 3, 4, 6, 12}) → 
  ∃ probability, probability = (favorable_outcomes.card : ℚ) / (die_side.card : ℚ) ∧ probability = 1/2 :=
by 
  sorry

end probability_divisor_of_12_l36_36638


namespace reliable_plumbing_hourly_rate_l36_36145

theorem reliable_plumbing_hourly_rate :
  ∃ (x : ℝ), let pauls_plumbing_charges := 55 + 4 * 35
              let reliable_plumbing_charges := 75 + 4 * x
              pauls_plumbing_charges = reliable_plumbing_charges ∧ x = 30 :=
by
  -- Define the charges according to the conditions
  let pauls_plumbing_charges := 55 + 4 * 35
  let reliable_plumbing_charges := 75 + 4 * 30

  -- Assert that the charges are equal
  have h1 : pauls_plumbing_charges = reliable_plumbing_charges := by
    calc 55 + 4 * 35 = 195 : by norm_num
    ... = 75 + 4 * 30 : by norm_num

  -- Provide the solution
  existsi 30
  split
  · exact h1
  · rfl

end reliable_plumbing_hourly_rate_l36_36145


namespace harold_betty_book_probability_l36_36140

open Function

/-- 
  Ms. Carr updates her reading list to include 12 books and asks her students 
  to read any 6 of these books. Harold and Betty each randomly select 6 books 
  from the updated list. Prove that the probability that Harold and Betty have 
  exactly 3 books in common is 140/323.
-/
theorem harold_betty_book_probability :
  let total_ways_to_choose_6 := Nat.choose 12 6
  let total_outcomes := total_ways_to_choose_6 * total_ways_to_choose_6
  let successful_outcomes := (Nat.choose 12 3) * (Nat.choose 9 3) * (Nat.choose 6 3)
  (successful_outcomes : ℚ) / total_outcomes = 140 / 323 :=
by
  let total_ways_to_choose_6 := Nat.choose 12 6
  let total_outcomes := total_ways_to_choose_6 * total_ways_to_choose_6
  let successful_outcomes := (Nat.choose 12 3) * (Nat.choose 9 3) * (Nat.choose 6 3)
  have calculation : successful_outcomes / total_outcomes = 140 / 323 := sorry
  exact calculation

end harold_betty_book_probability_l36_36140


namespace no_palindromic_years_between_2000_and_3000_l36_36650

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit_prime_palindrome (n : ℕ) : Prop :=
  is_prime n ∧ is_palindrome n ∧ 10 ≤ n ∧ n < 100

def is_three_digit_prime_palindrome (n : ℕ) : Prop :=
  is_prime n ∧ is_palindrome n ∧ 100 ≤ n ∧ n < 1000

theorem no_palindromic_years_between_2000_and_3000 :
  ∀ n, 2000 ≤ n ∧ n < 3000 ∧ is_palindrome n →
  ¬(∃ a b, is_two_digit_prime_palindrome a ∧ is_three_digit_prime_palindrome b ∧ n = a * b) :=
by sorry

end no_palindromic_years_between_2000_and_3000_l36_36650


namespace base8_to_base10_362_eq_242_l36_36675

theorem base8_to_base10_362_eq_242 : 
  let digits := [3, 6, 2]
  let base := 8
  let base10_value := (digits[2] * base^0) + (digits[1] * base^1) + (digits[0] * base^2) 
  base10_value = 242 :=
by
  sorry

end base8_to_base10_362_eq_242_l36_36675


namespace expansion_coefficient_x3_l36_36504

-- Define the binomial coefficient function needed for the solution
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the main problem statement
theorem expansion_coefficient_x3 :
  let f := (1 + x) * (1 - sqrt x) ^ 6
  ∃ (c : ℕ), (coeff_x3 f = c) ∧ (c = 16) :=
sorry

end expansion_coefficient_x3_l36_36504


namespace quadratic_roots_l36_36045

theorem quadratic_roots (a b : ℝ) (h : a^2 - 4*a*b + 5*b^2 - 2*b + 1 = 0) :
  ∃ (p q : ℝ), (∀ (x : ℝ), x^2 - p*x + q = 0 ↔ (x = a ∨ x = b)) ∧
               p = 3 ∧ q = 2 :=
by {
  sorry
}

end quadratic_roots_l36_36045


namespace probability_divisor_of_12_l36_36629

/-- Definition of divisors of a number -/
def divisors (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (λ d => n % d = 0)

theorem probability_divisor_of_12 :
  let divisors_of_12 := divisors 12
  Finset.card divisors_of_12 = 6 →
  (∃ (P : ℚ), P = 1/2) :=
by
  intro h
  exists 1/2
  sorry

end probability_divisor_of_12_l36_36629


namespace max_earnings_l36_36878

theorem max_earnings (h₁ : 60 ≥ 50) (h₂ : 40 ≤ 50) (h₃ : 25 / 100 > 0) (h₄ : 50 / 100 > 0) :
  regular_hours = 40 → regular_rate = 10 → overtime_hours = 10 →
  higher_hours = 10 →
  let regular_pay := regular_hours * regular_rate in
  let overtime_rate := regular_rate + (regular_rate * (25 / 100)) in
  let overtime_pay := overtime_hours * overtime_rate in
  let higher_rate := regular_rate + (regular_rate * (50 / 100)) in
  let higher_pay := higher_hours * higher_rate in
  let max_weekly_earnings := regular_pay + overtime_pay + higher_pay in
  max_weekly_earnings = 675 := sorry

end max_earnings_l36_36878


namespace inequality_holds_l36_36458

theorem inequality_holds (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h : x * y + y * z + z * x = 1) :
  (1 / (x + y)) + (1 / (y + z)) + (1 / (z + x)) ≥ 5 / 2 :=
by
  sorry

end inequality_holds_l36_36458


namespace train_crossing_time_correct_l36_36951

def train_crossing_time (length : ℝ) (speed_faster : ℝ) (ratio : ℝ) (speed_slower : ℝ) : ℝ :=
  let relative_speed := speed_faster + speed_slower
  let distance := length + length
  distance / relative_speed

theorem train_crossing_time_correct : 
  ∀ (length speed_faster ratio : ℝ), 
    ratio = 2 → 
    speed_faster = 24 → 
    train_crossing_time length speed_faster (speed_faster / ratio) = 200 / 36 :=
by
  intros length speed_faster ratio h_ratio h_speed_faster
  simp only [train_crossing_time]
  rw [h_ratio, h_speed_faster]
  have speed_slower := speed_faster / ratio
  rw [←h_ratio, h_speed_faster]
  have relative_speed := speed_faster + speed_slower
  simp [speed_faster, speed_slower]
  rw [←h_speed_faster, h_ratio]
  norm_num
  exact rfl

end train_crossing_time_correct_l36_36951


namespace num_registration_methods_l36_36324

-- Definitions based on conditions
def students : Fin 5 := sorry -- Note: We use 'Fin 5' to represent the five students

def clubs := {basketball, chess, environmental}

def restrictions (s : Fin 5) : Fin 3 → Prop
| 0 => ¬(s = 0) -- Xiao Bin won't join the chess club (0th club)
| 1 => ¬(s = 1) -- Xiao Cong won't join the basketball club (1st club)
| 2 => ¬(s = 2) -- Xiao Hao won't join the environmental club (2nd club)
| _ => true

theorem num_registration_methods : 
  (Fin 5) -> Prop :=
  ∃ total_choices : ℕ, total_choices = 72 := 
sorry

end num_registration_methods_l36_36324


namespace inverse_negation_of_proposition_l36_36933

variable (ABC: Type) [triangle ABC]

def is_isosceles_triangle (ABC : Type) [triangle ABC] : Prop :=
  ∃ (A B C : Point), (A = B ∨ B = C ∨ A = C) ∧ triangle ABC

def has_two_equal_internal_angles (ABC : Type) [triangle ABC] : Prop :=
  ∃ (A B C : Point) (angle1 angle2 : Angle), angle1 = angle2 ∧ (internal_angle A B C angle1) ∧ (internal_angle A B C angle2)

theorem inverse_negation_of_proposition (ABC : Type) [triangle ABC] :
  (¬ is_isosceles_triangle ABC → ¬ has_two_equal_internal_angles ABC) ↔
  (has_two_equal_internal_angles ABC → is_isosceles_triangle ABC) :=
by
  sorry

end inverse_negation_of_proposition_l36_36933


namespace value_of_f_2_pow_100_l36_36845

def f : ℕ → ℕ :=
sorry

axiom f_base : f 1 = 1
axiom f_recursive : ∀ n : ℕ, f (2 * n) = n * f n

theorem value_of_f_2_pow_100 : f (2^100) = 2^4950 :=
sorry

end value_of_f_2_pow_100_l36_36845


namespace hot_dogs_sold_next_innings_l36_36965

-- Defining the conditions
variables (total_initial hot_dogs_sold_first_innings hot_dogs_left : ℕ)

-- Given conditions that need to hold true
axiom initial_count : total_initial = 91
axiom first_innings_sold : hot_dogs_sold_first_innings = 19
axiom remaining_hot_dogs : hot_dogs_left = 45

-- Prove the number of hot dogs sold during the next three innings is 27
theorem hot_dogs_sold_next_innings : total_initial - (hot_dogs_sold_first_innings + hot_dogs_left) = 27 :=
by
  sorry

end hot_dogs_sold_next_innings_l36_36965


namespace coeff_x3_in_expansion_of_x_plus_1_50_l36_36089

theorem coeff_x3_in_expansion_of_x_plus_1_50 :
  (Finset.range 51).sum (λ k => Nat.choose 50 k * (1 : ℕ) ^ (50 - k) * k ^ 3) = 19600 := by
  sorry

end coeff_x3_in_expansion_of_x_plus_1_50_l36_36089


namespace totalCostOfCombinedSubscriptions_l36_36994

-- Define the given conditions
def packageACostPerMonth : ℝ := 10
def packageAMonths : ℝ := 6
def packageADiscount : ℝ := 0.10

def packageBCostPerMonth : ℝ := 12
def packageBMonths : ℝ := 9
def packageBDiscount : ℝ := 0.15

-- Define the total cost after discounts
def packageACostAfterDiscount : ℝ := packageACostPerMonth * packageAMonths * (1 - packageADiscount)
def packageBCostAfterDiscount : ℝ := packageBCostPerMonth * packageBMonths * (1 - packageBDiscount)

-- Statement to be proved
theorem totalCostOfCombinedSubscriptions :
  packageACostAfterDiscount + packageBCostAfterDiscount = 145.80 := by
  sorry

end totalCostOfCombinedSubscriptions_l36_36994


namespace lorenzo_ate_more_l36_36910

theorem lorenzo_ate_more (L J : ℕ) (hL : L = 6) (hJ : J = 1) : L - J = 5 :=
by
  rw [hL, hJ]
  exact rfl

end lorenzo_ate_more_l36_36910


namespace good_partitions_count_l36_36760

def is_good_partition (A1 A2 A3 : Finset ℕ) : Prop :=
  ∃ (i1 i2 i3 : Finset ℕ) (h: {i1, i2, i3} = {A1, A2, A3}),
    (i1.nonempty ∧ i2.nonempty ∧ i3.nonempty) ∧
    (∃ k (h1: k ∈ i1) (h2: (k + 1) % 3 ∈ i2), true)

def M : Finset ℕ := {n ∈ Finset.range 11 | 0 < n}

def count_good_partitions : ℕ :=
  {p // let (A1, A2, A3, hp) := p in is_good_partition A1 A2 A3}.to_finset.card

theorem good_partitions_count :
  count_good_partitions = 8362 := sorry

end good_partitions_count_l36_36760


namespace card_arrangements_count_l36_36544

theorem card_arrangements_count :
  let cards : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]
  let chosen_cards := { L : List ℕ // L.length = 6 ∧ (∀ M, (M ∈ L.subsets 2) → (M.sum = 5 ↔ M = [1, 4] ∨ M = [2, 3])) }
  (chosen_cards.enum 6).length = 1248 :=
by
  sorry

end card_arrangements_count_l36_36544


namespace value_of_a3_a5_l36_36353

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r > 0, ∀ n, a (n + 1) = a n * r

theorem value_of_a3_a5 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_pos : ∀ n, a n > 0) 
  (h_eq : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) : 
  a 3 + a 5 = 5 :=
  sorry

end value_of_a3_a5_l36_36353


namespace largest_non_zero_ending_factor_decreasing_number_l36_36314

theorem largest_non_zero_ending_factor_decreasing_number :
  ∃ n: ℕ, n = 180625 ∧ (n % 10 ≠ 0) ∧ (∃ m: ℕ, m < n ∧ (n % m = 0) ∧ (n / 10 ≤ m ∧ m * 10 > 0)) :=
by {
  sorry
}

end largest_non_zero_ending_factor_decreasing_number_l36_36314


namespace shaded_area_of_circles_l36_36696

theorem shaded_area_of_circles (r1 r2 : ℝ) (h1 : r1 = 4) (h2 : r2 = 5) :
  let R := r1 + r2 in
  let area_large_circle := π * R^2 in
  let area_small_circle1 := π * r1^2 in
  let area_small_circle2 := π * r2^2 in
  area_large_circle - area_small_circle1 - area_small_circle2 = 40 * π :=
by
  sorry

end shaded_area_of_circles_l36_36696


namespace number_of_solutions_l36_36387

-- Define the gcd function
def gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Define the lcm function
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- Define the problem statement as a theorem
theorem number_of_solutions :
  {p : ℕ × ℕ | let (a, b) := p in a > 0 ∧ b > 0 ∧ a * b + 80 = 15 * lcm a b + 10 * gcd a b}.card = 2 :=
by sorry

end number_of_solutions_l36_36387


namespace inequality_for_real_numbers_l36_36162

theorem inequality_for_real_numbers (n : ℕ) (xs : Fin n → ℝ) :
  (∑ i in Finset.univ, xs i / (1 + ∑ j in Finset.range (i+1), (xs j)^2)) < Real.sqrt n := 
sorry

end inequality_for_real_numbers_l36_36162


namespace possible_values_of_expression_l36_36772

theorem possible_values_of_expression (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ v : ℤ, v ∈ {5, 1, -3} ∧
  v = (a / |a|) + (b / |b|) + (c / |c|) + (d / |d|) + (abcd / |abcd|) :=
sorry

end possible_values_of_expression_l36_36772


namespace probability_divisor_of_12_l36_36641

theorem probability_divisor_of_12 : 
  (nat.factorization 12).keys = {1, 2, 3, 4, 6, 12} → 
  (∃ die_side, die_side ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) → 
  (∃ favorable_outcomes, favorable_outcomes ∈ {1, 2, 3, 4, 6, 12}) → 
  ∃ probability, probability = (favorable_outcomes.card : ℚ) / (die_side.card : ℚ) ∧ probability = 1/2 :=
by 
  sorry

end probability_divisor_of_12_l36_36641


namespace new_mean_l36_36839

-- Define the conditions
def mean_of_numbers (n : ℕ) (mean : ℝ) : ℝ := n * mean
def added_to_each (n : ℕ) (addend : ℝ) : ℝ := n * addend

-- The proof problem
theorem new_mean (n : ℕ) (mean addend : ℝ) (h1 : mean_of_numbers n mean = 600) (h2 : added_to_each n addend = 150) (h3 : n = 15) (h4 : mean = 40) (h5 : addend = 10) :
  (mean_of_numbers n mean + added_to_each n addend) / n = 50 :=
by
  sorry

end new_mean_l36_36839


namespace find_length_of_DE_l36_36277

-- Definitions based on the conditions
variable {A B C D E : Point}
variable (AB : ℝ) (BC : ℝ) (S_ABC : ℝ)
variable (angle_bisector : Line) (perpendicular : Line)
variable [IsAngleBisector angle_bisector A B C]
variable [PerpendicularTo perpendicular BC]
variable [Length AB = 8]
variable [Length BC = 12]
variable [AreaTriangle A B C = 24]
variable (DE : ℝ)

-- The statement to prove that the length of DE is 2.4
theorem find_length_of_DE : DE = 2.4 := sorry

end find_length_of_DE_l36_36277


namespace arithmetic_sequence_terms_l36_36404

theorem arithmetic_sequence_terms (a : ℕ → ℕ) (n : ℕ)
  (h1 : a 1 + a 2 + a 3 = 34)
  (h2 : a n + a (n - 1) + a (n - 2) = 146)
  (h3 : n * (a 1 + a n) = 780) : n = 13 :=
sorry

end arithmetic_sequence_terms_l36_36404


namespace georgia_total_cost_l36_36889

def carnation_price : ℝ := 0.50
def dozen_price : ℝ := 4.00
def teachers : ℕ := 5
def friends : ℕ := 14

theorem georgia_total_cost :
  ((dozen_price * teachers) + dozen_price + (carnation_price * (friends - 12))) = 25.00 :=
by
  sorry

end georgia_total_cost_l36_36889


namespace lateral_area_of_cylinder_l36_36511

-- Define the cylinder parameters
def diameter : ℝ := 4
def height : ℝ := 4
def radius : ℝ := diameter / 2

-- Define the lateral area of the cylinder
def lateral_area : ℝ := 2 * Real.pi * radius * height

-- Prove that the lateral area of the cylinder is 16π
theorem lateral_area_of_cylinder : lateral_area = 16 * Real.pi :=
by
  sorry

end lateral_area_of_cylinder_l36_36511


namespace find_frac_sum_l36_36118

variable (a b c : ℝ)
variable (h1 : 16 * b^2 = 15 * a * c)
variable (h2 : 2 / b = 1 / a + 1 / c)

theorem find_frac_sum (a b c : ℝ) (h1 : 16 * b^2 = 15 * a * c) (h2 : 2 / b = 1 / a + 1 / c) :
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 → (a / c + c / a) = 34 / 15 := by
  sorry

end find_frac_sum_l36_36118


namespace subtract_abs_from_local_value_of_four_in_564823_l36_36971

theorem subtract_abs_from_local_value_of_four_in_564823 :
  let local_value := 4 * 10000
  let abs_value : ℤ := abs 4
  local_value - abs_value = 39996 := 
by
  let local_value := 4 * 10000
  let abs_value : ℤ := abs 4
  have eq1 : local_value = 40000 := by rfl
  have eq2 : abs_value = 4 := by rfl
  show local_value - abs_value = 39996, from calc
    local_value - abs_value = 40000 - 4 : by rw [eq1, eq2]
    ...               = 39996 : by norm_num

end subtract_abs_from_local_value_of_four_in_564823_l36_36971


namespace tangents_intersect_on_line_l36_36896

open_locale classical
noncomputable theory

variables {Ω : Type*} [metric_space Ω]
variables (l : set Ω) (A B C D : Ω)
variables (ω1 ω2 ω3 ω4 : set Ω)

-- Definitions of points lying on the line l
def points_on_line := A ∈ l ∧ B ∈ l ∧ C ∈ l ∧ D ∈ l ∧ (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ A ≠ C ∧ A ≠ D ∧ B ≠ D)

-- Definitions of the circular arcs passing through specific points
def arc_ω1_thru_A_B := ω1 A ∧ ω1 B
def arc_ω2_thru_A_B := ω2 A ∧ ω2 B
def arc_ω3_thru_C_D := ω3 C ∧ ω3 D
def arc_ω4_thru_C_D := ω4 C ∧ ω4 D

-- Definitions of the tangency conditions
def tangent_ω1_ω3 := tangent ω1 ω3
def tangent_ω2_ω4 := tangent ω2 ω4

-- Statement of the theorem
theorem tangents_intersect_on_line :
  points_on_line l A B C D →
  arc_ω1_thru_A_B ω1 A B →
  arc_ω2_thru_A_B ω2 A B →
  arc_ω3_thru_C_D ω3 C D →
  arc_ω4_thru_C_D ω4 C D →
  tangent_ω1_ω3 ω1 ω3 →
  tangent_ω2_ω4 ω2 ω4 →
  ∃ X : Ω, (is_common_external_tangent ω2 ω3 X) ∧ (is_common_external_tangent ω1 ω4 X) ∧ X ∈ l :=
sorry

-- Helper functions, definitions, axioms, or lemmas could be declared here

end tangents_intersect_on_line_l36_36896


namespace smallest_checkered_rectangle_area_l36_36988

def even (n: ℕ) : Prop := n % 2 = 0

-- Both figure types are present and areas of these types are 1 and 2 respectively
def isValidPieceComposition (a b : ℕ) : Prop :=
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ m * 1 + n * 2 = a * b

theorem smallest_checkered_rectangle_area :
  ∀ a b : ℕ, even a → even b → isValidPieceComposition a b → a * b ≥ 40 := 
by
  intro a b a_even b_even h_valid
  sorry

end smallest_checkered_rectangle_area_l36_36988


namespace find_added_value_l36_36842

theorem find_added_value (avg_15_numbers : ℤ) (new_avg : ℤ) (x : ℤ)
    (H1 : avg_15_numbers = 40) 
    (H2 : new_avg = 50) 
    (H3 : (600 + 15 * x) / 15 = new_avg) : 
    x = 10 := 
sorry

end find_added_value_l36_36842


namespace trig_eq_solution_l36_36492

open Real

theorem trig_eq_solution (x : ℝ) : 
  (cos (7 * x) + cos (3 * x) + sin (7 * x) - sin (3 * x) + sqrt 2 * cos (4 * x) = 0) ↔ 
  (∃ k : ℤ, 
    (x = -π / 8 + π * k / 2) ∨ 
    (x = -π / 4 + 2 * π * k / 3) ∨ 
    (x = 3 * π / 28 + 2 * π * k / 7)) :=
by sorry

end trig_eq_solution_l36_36492


namespace solution_set_of_inequality_system_l36_36528

theorem solution_set_of_inequality_system (x : ℝ) :
  (3 * x - 1 ≥ x + 1) ∧ (x + 4 > 4 * x - 2) ↔ (1 ≤ x ∧ x < 2) := 
by
  sorry

end solution_set_of_inequality_system_l36_36528


namespace product_of_sequence_is_one_l36_36749

theorem product_of_sequence_is_one 
  (n : ℕ) (h₁: n > 2)
  (x : ℕ → ℝ) 
  (h₂ : ∀ i j, (i % n) = (j % n) → x i = x j)
  (f : ℕ → ℝ)
  (h₃ : ∀ i, f i = ∑ j in (range n), (∏ k in range j, x (i + k)))
  (h₄ : ∀ i, f i = f (i + 1)) :
  (∏ i in (range n), x i) = 1 :=
by
  sorry

end product_of_sequence_is_one_l36_36749


namespace number_of_parents_l36_36201

theorem number_of_parents {girls boys groups group_size : ℕ} 
  (h_girls : girls = 14)
  (h_boys : boys = 11)
  (h_groups : groups = 3)
  (h_group_size : group_size = 25) :
  let total_people := groups * group_size
      total_children := girls + boys
      parents := total_people - total_children
  in parents = 50 :=
by
  sorry

end number_of_parents_l36_36201


namespace find_hyperbola_equation_l36_36396

noncomputable def ellipse : Type := {
  center : (ℝ, ℝ),
  a : ℝ,
  b : ℝ,
  equation : real → real → Prop
}

def hyperbola_shares_foci (e : ellipse) (h_f_1 : ℝ) (h_f_2 : ℝ) : Prop :=
  ∃ (c : ℝ), c = sqrt (e.a^2 - e.b^2) ∧ c = h_f_1 ∧ c = h_f_2

def asymptote_hyperbola (h : hyperbola) (line_slope : ℝ) : Prop :=
  h.b / h.a = line_slope

def hyperbola_equation_correct (h : hyperbola) : Prop :=
  h.equation = λ x y, x^2 / h.a^2 - y^2 / h.b^2 = 1

theorem find_hyperbola_equation (e : ellipse) (h_f_1 h_f_2 : ℝ) (line_slope : ℝ) (a : ℝ) (b : ℝ) :
  e.center = (0, 0) →
  e.a = sqrt 8 →
  e.b = 2 →
  h_f_1 = 2 →
  h_f_2 = 2 →
  hyperbola_shares_foci e h_f_1 h_f_2 →
  asymptote_hyperbola ⟨(0, 0), a, b, λ x y, x^2 / a^2 - y^2 / b^2 = 1⟩ line_slope →
  a = 1 →
  b = sqrt 3 →
  hyperbola_equation_correct ⟨(0, 0), a, b, λ x y, x^2 / a^2 - y^2 / b^2 = 1⟩ :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end find_hyperbola_equation_l36_36396


namespace AE_length_l36_36708

theorem AE_length :
  ∀ (A B C D E : Type) 
    (AB CD AC BD AE EC : ℕ),
  AB = 12 → CD = 15 → AC = 18 → BD = 27 → 
  (AE + EC = AC) → 
  (AE * (18 - AE)) = (4 / 9 * 18 * 8) → 
  9 * AE = 72 → 
  AE = 8 := 
by
  intros A B C D E AB CD AC BD AE EC hAB hCD hAC hBD hSum hEqual hSolve
  sorry

end AE_length_l36_36708


namespace final_answer_l36_36976

def sequence (a : ℕ → ℝ) :=
  ∃ a : ℕ → ℝ, ∀ n : ℕ, 1 / (a n * a (n+1)) + 1 / (a n * a (n+2)) + 1 / (a (n+1) * a (n+2)) = 1

def geometric_progression (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∃ a1 : ℝ, a 1 = a1 ∧ a 2 = r * a1 ∧ a 3 = r^2 * a1 ∧ a1 + a 3 = 6

def strictly_increasing (a : ℕ → ℝ) :=
  ∀ n : ℕ, a n < a (n+1)

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) :=
  ∑ i in finset.range n, a i

def S_2008_floor (a : ℕ → ℝ) :=
  nat.floor (sum_of_first_n_terms a 2008)

theorem final_answer : ∃ a : ℕ → ℝ, sequence a ∧ geometric_progression a ∧ strictly_increasing a ∧ S_2008_floor a = 5352 :=
sorry

end final_answer_l36_36976


namespace regular_hexagon_interior_angle_measure_l36_36557

theorem regular_hexagon_interior_angle_measure :
  ∀ (n : ℕ), n = 6 → (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → i = 6) →
  180 * (n - 2) / n = 120 :=
by 
{
  intros n h₁ h₂,
  rw h₁,
  simp,
  norm_num,
  sorry
}

end regular_hexagon_interior_angle_measure_l36_36557


namespace ratio_area_KZM_ABC_l36_36433

-- Define the main statement of the problem
theorem ratio_area_KZM_ABC 
  (α β : ℝ) (α_pos : α > 0) (β_pos : β > 0)
  (A B C M K Z : Type)  -- We assume points are of some type
  (angle_A : angle A B C = α)
  (angle_C : angle B C A = β)
  (BM_is_altitude : Altitude B M A C)
  (circum_with_diameter_BM : ∃ c : Circle, c.diameter = segment B M ∧ c ∩ segment A B = {K} ∧ c ∩ segment B C = {Z}) :
  area (triangle K Z M) / area (triangle A B C) = (1/4) * sin (2 * α) * sin (2 * β) := 
sorry

end ratio_area_KZM_ABC_l36_36433


namespace geometric_sequence_a7_l36_36420

-- Define the geometric sequence
def geometic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

-- Conditions
def a1 (a : ℕ → ℝ) : Prop :=
  a 1 = 1

def a2a4 (a : ℕ → ℝ) : Prop :=
  a 2 * a 4 = 16

-- The statement to prove
theorem geometric_sequence_a7 (a : ℕ → ℝ) (h1 : a1 a) (h2 : a2a4 a) (gs : geometic_sequence a) :
  a 7 = 64 :=
by
  sorry

end geometric_sequence_a7_l36_36420


namespace parabola_distance_focus_P_l36_36359

noncomputable def distance_PF : ℝ := sorry

theorem parabola_distance_focus_P : ∀ (P : ℝ × ℝ) (F : ℝ × ℝ),
  P.2^2 = 4 * P.1 ∧ F = (1, 0) ∧ P.1 = 4 → distance_PF = 5 :=
by
  intros P F h
  sorry

end parabola_distance_focus_P_l36_36359


namespace coefficient_x3_in_binomial_expansion_l36_36085

theorem coefficient_x3_in_binomial_expansion :
  nat.choose 50 3 = 19600 :=
by
  -- Proof goes here
  sorry

end coefficient_x3_in_binomial_expansion_l36_36085


namespace fraction_people_over_65_l36_36725

theorem fraction_people_over_65 {T O U: ℚ} 
    (h1: U / T = 3 / 7) -- Condition 1: Under 21 people fraction
    (h2: U = 30)        -- Condition 4: Number of people under 21
    (h3: 50 < T)        -- Condition 3: Total people greater than 50
    (h4: T < 100)       -- Condition 3: Total people less than 100) :
    (h5: O ≤ T - U)     -- Condition 2: Certain fraction of people over 65):

    O / T ≤ 4 / 7 := 
by 
  -- Start with given conditions to derive the answer
  sorry -- Placeholder for the proof

end fraction_people_over_65_l36_36725


namespace number_of_distinct_possible_values_for_c_l36_36453

variables {a b r s t : ℂ}
variables (h_distinct : r ≠ s ∧ s ≠ t ∧ r ≠ t)
variables (h_transform : ∀ z, (a * z + b - r) * (a * z + b - s) * (a * z + b - t) = (z - c * r) * (z - c * s) * (z - c * t))

theorem number_of_distinct_possible_values_for_c (h_nonzero : a ≠ 0) : 
  ∃ (n : ℕ), n = 4 := sorry

end number_of_distinct_possible_values_for_c_l36_36453


namespace min_add_to_palindrome_l36_36556

theorem min_add_to_palindrome :
  ∃ (n : ℕ), (∀ m, 78999 + n = m → (m.to_string = m.to_string.reverse)) ∧ n = 98 :=
by sorry

end min_add_to_palindrome_l36_36556


namespace car_b_speed_l36_36987

/--
A car A going at 30 miles per hour set out on an 80-mile trip at 9:00 a.m.
Exactly 10 minutes later, a car B left from the same place and followed the same route.
Car B caught up with car A at 10:30 a.m.
Prove that the speed of car B is 33.75 miles per hour.
-/
theorem car_b_speed
    (v_a : ℝ) (t_start_a t_start_b t_end : ℝ) (v_b : ℝ)
    (h1 : v_a = 30) 
    (h2 : t_start_a = 9) 
    (h3 : t_start_b = 9 + (10 / 60)) 
    (h4 : t_end = 10.5) 
    (h5 : t_end - t_start_b = (4 / 3))
    (h6 : v_b * (t_end - t_start_b) = v_a * (t_end - t_start_a) + (v_a * (10 / 60))) :
  v_b = 33.75 := 
sorry

end car_b_speed_l36_36987


namespace tournament_total_games_l36_36823

theorem tournament_total_games (n : ℕ) (k : ℕ) (h_n : n = 30) (h_k : k = 4) : 
  (n * (n - 1) / 2) * k = 1740 := by
  -- Given conditions
  have h1 : n = 30 := h_n
  have h2 : k = 4 := h_k

  -- Calculation using provided values
  sorry

end tournament_total_games_l36_36823


namespace product_of_scores_l36_36821

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def f (n : ℕ) : ℕ :=
  if is_perfect_square n then 8
  else if n % 3 = 0 then 4
  else if n % 2 = 0 then 1
  else 0

def Allie_rolls : List ℕ := [3, 4, 6, 1]
def Betty_rolls : List ℕ := [4, 2, 5, 1]

def Allie_score : ℕ :=
  Allie_rolls.map f |> List.sum

def Betty_score : ℕ :=
  Betty_rolls.map f |> List.sum

theorem product_of_scores :
  Allie_score * Betty_score = 117 :=
by
  rw [Allie_score, Betty_score]
  simp [Allie_rolls, Betty_rolls, f]
  by_cases h1 : is_perfect_square 3; simp [f, h1]
  by_cases h2 : is_perfect_square 4; simp [f, h2]
  by_cases h3 : is_perfect_square 6; simp [f, h3]
  by_cases h4 : is_perfect_square 1; simp [f, h4]
  revert h1
  revert h2
  revert h3
  revert h4
  repeat { sorry }

axiom allie_total : Allie_score = 13
axiom betty_total : Betty_score = 9

example : 13 * 9 = 117 :=
by norm_num

end

end product_of_scores_l36_36821


namespace factorial_difference_division_l36_36219

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem factorial_difference_division :
  (factorial 13 - factorial 12) / factorial 10 = 1584 :=
by sorry

end factorial_difference_division_l36_36219


namespace smallest_k_divisible_by_3_2017_l36_36343

noncomputable
def sequence (a : ℕ → ℕ) : ℕ → ℕ
| 0     => 1
| (n+1) => 2 * (Finset.range (n+1)).sum a

theorem smallest_k_divisible_by_3_2017 :
  ∃ k : ℕ, (sequence (sequence a k)) % (3^2017) = 0 ∧ k = 2019 :=
by
  sorry

end smallest_k_divisible_by_3_2017_l36_36343


namespace electricity_cost_per_kWh_is_14_cents_l36_36844

-- Define the conditions
def powerUsagePerHour : ℕ := 125 -- watts
def dailyUsageHours : ℕ := 4 -- hours
def weeklyCostInCents : ℕ := 49 -- cents
def daysInWeek : ℕ := 7 -- days
def wattsToKilowattsFactor : ℕ := 1000 -- conversion factor

-- Define a function to calculate the cost per kWh
def costPerKwh (powerUsagePerHour : ℕ) (dailyUsageHours : ℕ) (weeklyCostInCents : ℕ) (daysInWeek : ℕ) (wattsToKilowattsFactor : ℕ) : ℕ :=
  let dailyConsumption := powerUsagePerHour * dailyUsageHours
  let weeklyConsumption := dailyConsumption * daysInWeek
  let weeklyConsumptionInKwh := weeklyConsumption / wattsToKilowattsFactor
  weeklyCostInCents / weeklyConsumptionInKwh

-- State the theorem
theorem electricity_cost_per_kWh_is_14_cents :
  costPerKwh powerUsagePerHour dailyUsageHours weeklyCostInCents daysInWeek wattsToKilowattsFactor = 14 :=
by
  sorry

end electricity_cost_per_kWh_is_14_cents_l36_36844


namespace axis_of_symmetry_eq_two_l36_36371

variable (a : ℝ)
variable (h : a ≠ 0)

/-- The equation of the axis of symmetry for the function y = ax^2 - 4ax + 1 is x = 2, given that a ≠ 0. -/
theorem axis_of_symmetry_eq_two (a : ℝ) (h : a ≠ 0) : 
  axis_of_symmetry (λ x : ℝ, a * x^2 - 4 * a * x + 1) = 2 :=
by
  sorry

end axis_of_symmetry_eq_two_l36_36371


namespace coefficient_of_x3_in_x_plus_one_pow_50_l36_36093

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := (n.choose k)

-- Define the binomial expansion using summation
def binomial_expansion (x : ℕ) (n : ℕ) : ℕ → ℕ :=
  λ k, binom n k * x^k 

-- Define the specific problem
def coeff_x3_in_expansion : ℕ :=
  binom 50 3

-- Theorem stating the desired result
theorem coefficient_of_x3_in_x_plus_one_pow_50 :
  coeff_x3_in_expansion = 19600 :=
by
  -- Skipping the proof part by using sorry
  sorry

end coefficient_of_x3_in_x_plus_one_pow_50_l36_36093


namespace transform_point_c_l36_36548

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

def reflect_diag (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

theorem transform_point_c :
  let C := (3, 2)
  let C' := reflect_x C
  let C'' := reflect_y C'
  let C''' := reflect_diag C''
  C''' = (-2, -3) :=
by
  sorry

end transform_point_c_l36_36548


namespace paul_added_cards_l36_36565

variable (initial_cards : ℕ) (cards_taken_by_maria : ℕ) (final_cards : ℕ)
variable (maria_takes_cards : initial_cards → ℕ) (give_peter : ℕ)
variable (pauls_action : ℕ → ℕ → ℕ)

def maria_takes_cards (initial_cards : ℕ) : ℕ := (initial_cards + 1) / 2

theorem paul_added_cards (initial_cards final_cards : ℕ) :
  initial_cards = 15 →
  cards_taken_by_maria = maria_takes_cards initial_cards →
  give_peter = 1 →
  final_cards = 18 →
  pauls_action (initial_cards - cards_taken_by_maria - give_peter) final_cards = 12 :=
by
  sorry

end paul_added_cards_l36_36565


namespace perpendicular_lines_necessary_but_not_sufficient_l36_36581

theorem perpendicular_lines_necessary_but_not_sufficient (m : ℝ) :
  (∀ (x y : ℝ), (x + 3 * y - 1 = 0) ∧ (3 * x - y + 3 = 0) → m = -1) ∧ (∃ (x y : ℝ), (3 * m + m * (2 * m - 1) = 0) ∧ (m ≠ -1) → (l₁ : ℝ) (mx + (2 * m - 1) * y + 1 = 0) ∧ (l₂ : ℝ) * (3 * x + m * y + 3 = 0)) :=
by
sorry

end perpendicular_lines_necessary_but_not_sufficient_l36_36581


namespace function_equivalence_l36_36962

theorem function_equivalence :
  ∀ (x : ℝ), (sqrt[3](x^3) = x) :=
by 
sorry

end function_equivalence_l36_36962


namespace shaded_region_area_l36_36703

theorem shaded_region_area {radius1 radius2 : ℝ} (h1 : radius1 = 4) (h2 : radius2 = 5) :
  let dist_centers := radius1 + radius2,
      circumscribed_radius := dist_centers,
      larger_area := Real.pi * circumscribed_radius ^ 2,
      smaller_area1 := Real.pi * radius1 ^ 2,
      smaller_area2 := Real.pi * radius2 ^ 2,
      shaded_area := larger_area - smaller_area1 - smaller_area2
  in shaded_area = 40 * Real.pi :=
by
  simp [h1, h2]
  sorry

end shaded_region_area_l36_36703


namespace rearrange_circle_to_ovals_with_handholes_l36_36691

theorem rearrange_circle_to_ovals_with_handholes :
  (∃ pieces : list (set ℝ²), 
    (disjoint (set.of_list pieces)) ∧ 
    (⋃₀ (set.of_list pieces) = circular_tabletop) ∧
    (∃ ovals : list (set ℝ²), 
      length ovals = 2 ∧ 
      (∀ oval ∈ ovals, ∃ handhole ∈ oval) ∧ 
      (∀ piece ∈ pieces, piece ∈ ⋃₀ set.of_list ovals) ∧
      (∀ oval ∈ ovals, (∃ sublist2 ∈ (set.powerset_len 2 pieces), oval = ⋃₀ sublist2))
  )) :=
sorry

end rearrange_circle_to_ovals_with_handholes_l36_36691


namespace number_of_yellow_balls_l36_36072

theorem number_of_yellow_balls (x : ℕ) (h : (6 : ℝ) / (6 + x) = 0.3) : x = 14 :=
by
  sorry

end number_of_yellow_balls_l36_36072


namespace range_of_b_l36_36030

noncomputable def f : ℝ → ℝ
| x => if x < -1/2 then (2*x + 1) / (x^2) else x + 1

def g (x : ℝ) : ℝ := x^2 - 4*x - 4

theorem range_of_b (a b : ℝ) (h : f a + g b = 0) : -1 <= b ∧ b <= 5 :=
sorry

end range_of_b_l36_36030


namespace number_of_sides_of_polygon_l36_36397

theorem number_of_sides_of_polygon (exterior_angle : ℝ) (h : exterior_angle = 40) : 
  (360 / exterior_angle) = 9 :=
by
  sorry

end number_of_sides_of_polygon_l36_36397


namespace equidistant_points_from_A_B_l36_36307

-- Define the points A and B
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 1, y := 5 }
def B : Point := { x := 2, y := 4 }

-- Define the distance function
def distance (P Q : Point) : ℝ :=
  Math.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

-- Define the points to be proven equidistant from A and B
def P1 : Point := { x := 0, y := 3 }
def P2 : Point := { x := -3, y := 0 }

-- The proposition to prove
theorem equidistant_points_from_A_B : 
  distance P1 A = distance P1 B ∧ distance P2 A = distance P2 B :=
by
  sorry

end equidistant_points_from_A_B_l36_36307


namespace tetrahedron_volume_l36_36727

theorem tetrahedron_volume
  (P Q R S : Point)
  (angle_PQR_QRS : angle (plane P Q R) (plane Q R S) = π / 4)
  (area_PQR : area (triangle P Q R) = 150)
  (area_QRS : area (triangle Q R S) = 50)
  (QR : dist Q R = 5) :
  volume (tetrahedron P Q R S) = 500 * Real.sqrt 2 :=
  sorry

end tetrahedron_volume_l36_36727


namespace sufficient_and_necessary_condition_l36_36534

theorem sufficient_and_necessary_condition {a : ℝ} :
  (∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ a ≥ 4 :=
sorry

end sufficient_and_necessary_condition_l36_36534


namespace alpha_pi_over_four_sufficient_not_necessary_l36_36978

theorem alpha_pi_over_four_sufficient_not_necessary :
  (∀ α : ℝ, (α = (Real.pi / 4) → Real.cos α = Real.sqrt 2 / 2)) ∧
  (∃ α : ℝ, (Real.cos α = Real.sqrt 2 / 2) ∧ α ≠ (Real.pi / 4)) :=
by
  sorry

end alpha_pi_over_four_sufficient_not_necessary_l36_36978


namespace translate_aspect_to_chinese_l36_36970

theorem translate_aspect_to_chinese : translation "aspect" = "方面" :=
sorry

end translate_aspect_to_chinese_l36_36970


namespace compound_interest_correct_l36_36143

theorem compound_interest_correct (SI : ℝ) (R : ℝ) (T : ℕ) (CI : ℝ) :
  SI = 600 → R = 3 → T = 2 →
  CI = (10000 * (1 + R / 100)^T - 10000) :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  sorry

end compound_interest_correct_l36_36143


namespace quadrilateral_not_determined_to_be_parallelogram_l36_36423

open EuclideanGeometry -- we assume Euclidean geometry as a context

-- Define the quadrilateral ABCD with given properties
variables {A B C D : Point}
variables {angle_BAD angle_DCB : ℕ}
variables {AB CD : ℕ}

-- Define that angles are equal and sides are equal
hypothesis (h1 : angle_BAD = angle_DCB)
hypothesis (h2 : AB = CD)

-- State that these conditions are insufficient to determine if ABCD is a parallelogram
theorem quadrilateral_not_determined_to_be_parallelogram :
  ∃ (A B C D : Point), (angle_BAD = angle_DCB) ∧ (AB = CD) ∧ ¬ (ABCD is_parallelogram) :=
sorry

end quadrilateral_not_determined_to_be_parallelogram_l36_36423


namespace probability_two_green_apples_l36_36107

theorem probability_two_green_apples :
  let total_apples := 9
  let total_red := 5
  let total_green := 4
  let ways_to_choose_two := Nat.choose total_apples 2
  let ways_to_choose_two_green := Nat.choose total_green 2
  ways_to_choose_two ≠ 0 →
  (ways_to_choose_two_green / ways_to_choose_two : ℚ) = 1 / 6 :=
by
  intros
  -- skipping the proof
  sorry

end probability_two_green_apples_l36_36107


namespace coeff_x3_in_expansion_l36_36081

theorem coeff_x3_in_expansion : (nat.choose 50 3 * (1^47) * (1^3)) = 19600 := 
by sorry

end coeff_x3_in_expansion_l36_36081


namespace range_of_a_l36_36401

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → 2 * x > x^2 + a) → a < -8 :=
by
  intro h
  -- Complete the proof by showing that 2x - x^2 has a minimum value of -8 on [-2, 3] and hence proving a < -8.
  sorry

end range_of_a_l36_36401


namespace SOTKA_not_divisible_by_9_l36_36096

variable (D E B я H C T K A : ℕ)

-- Given conditions
def DEBЯNOSTO_divisible_by_90 : Prop :=
  (D + E + B + я + H + C + T + 0) % 9 = 0 ∧ 
  (C * 10 + 0) % 10 = 0

def DEBЯTKA_divisible_by_9 : Prop :=
  (D + E + B + я + T + K + A) % 9 = 0

-- Prove whether СОТКА is divisible by 9 under given conditions
theorem SOTKA_not_divisible_by_9 
  (h1 : DEBЯNOSTO_divisible_by_90 D E B я H C T K A)
  (h2 : DEBЯTKA_divisible_by_9 D E B я H C T K A) :
  ¬ ((C + 0 + T + K + A) % 9 = 0) :=
sorry

end SOTKA_not_divisible_by_9_l36_36096


namespace find_y_l36_36398

theorem find_y :
  ∃ (y : ℤ), (55 + 48 + 507 + 2 + 684 + y) / 6 = 223 ∧ y = 42 :=
begin
  use 42,
  split,
  {
    norm_num,
  },
  {
    refl,
  },
end

end find_y_l36_36398


namespace georgia_total_carnation_cost_l36_36892

-- Define the cost of one carnation
def cost_of_single_carnation : ℝ := 0.50

-- Define the cost of one dozen carnations
def cost_of_dozen_carnations : ℝ := 4.00

-- Define the number of teachers
def number_of_teachers : ℕ := 5

-- Define the number of friends
def number_of_friends : ℕ := 14

-- Calculate the cost for teachers
def cost_for_teachers : ℝ :=
  (number_of_teachers : ℝ) * cost_of_dozen_carnations

-- Calculate the cost for friends
def cost_for_friends : ℝ :=
  cost_of_dozen_carnations + (2 * cost_of_single_carnation)

-- Calculate the total cost
def total_cost : ℝ := cost_for_teachers + cost_for_friends

-- Theorem stating the total cost
theorem georgia_total_carnation_cost : total_cost = 25 := by
  -- Placeholder for the proof
  sorry

end georgia_total_carnation_cost_l36_36892


namespace probability_divisor_of_12_l36_36609

theorem probability_divisor_of_12 (fair_die : True) (sides : Nat) (h : sides = 12) : 
  ∃ p : ℚ, p = 1 / 2 ∧ ∀ roll : Fin sides, is_divisor roll.val 12 ↔ roll.val ∈ {1, 2, 3, 4, 6, 12} → p = 1 / 2 :=
by sorry

def is_divisor (a b : Nat) : Prop := b % a = 0

end probability_divisor_of_12_l36_36609


namespace find_y_l36_36809

theorem find_y (x y : ℤ) (h₁ : x = 4) (h₂ : 3 * x + 2 * y = 30) : y = 9 := 
by
  sorry

end find_y_l36_36809


namespace correct_propositions_l36_36452

-- Definitions representing the planes and line
variables (α β γ : Plane) (l : Line)

-- Propositions regarding the planes and line
def prop1 : Prop := (α ⊥ β ∧ l ⊥ β) → l ∥ α
def prop2 : Prop := (l ⊥ α ∧ l ∥ β) → α ⊥ β
def prop3 : Prop := (∃ p1 p2 : Point, p1 ≠ p2 ∧ p1 ∈ l ∧ p2 ∈ l ∧ dist p1 α = dist p2 α) → l ∥ α
def prop4 : Prop := (α ⊥ β ∧ α ∥ γ) → γ ⊥ β

-- The statement that propositions 2 and 4 are true
theorem correct_propositions : (prop2 α β l ∧ prop4 α β γ) :=
by
  sorry

end correct_propositions_l36_36452


namespace distribute_coins_l36_36303

theorem distribute_coins : 
  (∃ (A B C : ℕ), A + B + C = 10 ∧ A ≥ 1 ∧ B ≥ 1 ∧ C ≥ 1) → 
  finset.card ({(x, y, z) | x + y + z = 7 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0}.to_finset) = 8 := 
by
  sorry

end distribute_coins_l36_36303


namespace georgia_total_carnation_cost_l36_36890

-- Define the cost of one carnation
def cost_of_single_carnation : ℝ := 0.50

-- Define the cost of one dozen carnations
def cost_of_dozen_carnations : ℝ := 4.00

-- Define the number of teachers
def number_of_teachers : ℕ := 5

-- Define the number of friends
def number_of_friends : ℕ := 14

-- Calculate the cost for teachers
def cost_for_teachers : ℝ :=
  (number_of_teachers : ℝ) * cost_of_dozen_carnations

-- Calculate the cost for friends
def cost_for_friends : ℝ :=
  cost_of_dozen_carnations + (2 * cost_of_single_carnation)

-- Calculate the total cost
def total_cost : ℝ := cost_for_teachers + cost_for_friends

-- Theorem stating the total cost
theorem georgia_total_carnation_cost : total_cost = 25 := by
  -- Placeholder for the proof
  sorry

end georgia_total_carnation_cost_l36_36890


namespace eating_contest_l36_36880

variables (hotdog_weight burger_weight pie_weight : ℕ)
variable (noah_burgers jacob_pies mason_hotdogs : ℕ)
variable (total_weight_mason_hotdogs : ℕ)

theorem eating_contest :
  hotdog_weight = 2 →
  burger_weight = 5 →
  pie_weight = 10 →
  noah_burgers = 8 →
  jacob_pies = noah_burgers - 3 →
  mason_hotdogs = 3 * jacob_pies →
  total_weight_mason_hotdogs = mason_hotdogs * hotdog_weight →
  total_weight_mason_hotdogs = 30 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end eating_contest_l36_36880


namespace unique_solution_l36_36029

def f (x : ℝ) : ℝ := if x ≤ 0 then -x else x^2

theorem unique_solution (α : ℝ ) (h1 : f α = 9) : α = -9 ∨ α = 3 :=
by 
  sorry

end unique_solution_l36_36029


namespace floor_of_x_l36_36846

noncomputable def T : ℝ := 1139  -- Placeholder for TNFTPP
def R : ℝ := T - 914

theorem floor_of_x :
  (∃ x : ℝ, (3 * x^2 + R * x + R = 90 * x * sqrt (x + 1)) ∧ (∀ y : ℝ, y < x → ¬ (3 * y^2 + R * y + R = 90 * y * sqrt (y + 1)))) →
  ⌊x⌋ = 224 := sorry

end floor_of_x_l36_36846


namespace gabrielle_saw_20_percent_more_l36_36226

-- Define the number of birds seen by Gabrielle and Chase
def birds_seen_by_gabrielle : ℕ := 5 + 4 + 3
def birds_seen_by_chase : ℕ := 2 + 3 + 5

-- Define the correct answer as a percentage
def percentage_increase (a b : ℕ) : ℝ := ((a - b).toReal / b.toReal) * 100

-- Statement asserting that Gabrielle saw 20% more birds than Chase
theorem gabrielle_saw_20_percent_more : percentage_increase birds_seen_by_gabrielle birds_seen_by_chase = 20 := by
  sorry

end gabrielle_saw_20_percent_more_l36_36226


namespace last_painted_cell_333_444_l36_36923

theorem last_painted_cell_333_444 :
  ∃ (row col : ℕ), row = 167 ∧ col = 278 ∧ 
  (∀ r c, (r, c) is the last cell painted in a counter-clockwise spiral in a grid of size 333 × 444) :=
sorry

end last_painted_cell_333_444_l36_36923


namespace sum_of_all_cool_triangle_areas_valid_l36_36263

def is_cool_triangle (a b : ℕ) : Prop :=
  let c := Real.sqrt (a^2 + b^2)
  a * b = 6 * (a + b + c)

def cool_triangle_area (a b : ℕ) : ℝ :=
  (a * b) / 2

noncomputable def sum_of_cool_triangle_areas : ℝ :=
  let pairs_to_check := (Finset.range 100).product (Finset.range 100) -- checking pair up to 100 might suffice
  pairs_to_check.filter (λ (x : ℕ × ℕ), is_cool_triangle x.1 x.2)
                   .sum (λ (x : ℕ × ℕ), cool_triangle_area x.1 x.2)

theorem sum_of_all_cool_triangle_areas_valid : sum_of_cool_triangle_areas = -- fill in the correct sum found in the solution
sorry

end sum_of_all_cool_triangle_areas_valid_l36_36263


namespace sqrt_of_neg_five_squared_l36_36684

theorem sqrt_of_neg_five_squared : Real.sqrt ((-5 : Real) ^ 2) = 5 := 
by 
  sorry

end sqrt_of_neg_five_squared_l36_36684


namespace unit_price_proof_l36_36917

noncomputable def unit_price_B : ℝ := 2.5
def unit_price_A : ℝ := 1.2 * unit_price_B
def total_zongzi := 1100
def total_cost := 3000
def total_zongzi_second := 2600
def budget := 7000
def cost_A := unit_price_A
def cost_B := unit_price_B

theorem unit_price_proof :
  (1500 / unit_price_B + 1500 / cost_A = total_zongzi)
  ∧ (3 * m + 2.5 * (total_zongzi_second - m) ≤ budget)
  → (unit_price_B = 2.5 ∧ unit_price_A = 3)
  ∧ (m ≤ 1000) := sorry

end unit_price_proof_l36_36917


namespace ratio_of_installing_to_downloading_l36_36108

noncomputable def timeDownloading : ℕ := 10

noncomputable def ratioTimeSpent (installingTime : ℕ) : ℚ :=
  let tutorialTime := 3 * (timeDownloading + installingTime)
  let totalTime := timeDownloading + installingTime + tutorialTime
  if totalTime = 60 then
    (installingTime : ℚ) / (timeDownloading : ℚ)
  else 0

theorem ratio_of_installing_to_downloading : ratioTimeSpent 5 = 1 / 2 := by
  sorry

end ratio_of_installing_to_downloading_l36_36108


namespace probability_divisor_of_12_is_half_l36_36621

noncomputable def divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ d, n % d = 0) (Finset.range (n + 1))

def probability_divisor_of_12 : ℚ :=
  let total_outcomes := 12
  let favorable_outcomes := (divisors 12).card
  favorable_outcomes / total_outcomes

theorem probability_divisor_of_12_is_half :
  probability_divisor_of_12 = 1 / 2 :=
  by
    sorry

end probability_divisor_of_12_is_half_l36_36621


namespace stadium_revenue_difference_l36_36552

theorem stadium_revenue_difference :
  let total_capacity := 2000
  let vip_capacity := 200
  let standard_capacity := 1000
  let general_capacity := 800
  let vip_price := 50
  let standard_price := 30
  let general_price := 20
  let three_quarters (n : ℕ) := (3 * n) / 4
  let three_quarter_full := three_quarters total_capacity
  let vip_three_quarter := three_quarters vip_capacity
  let standard_three_quarter := three_quarters standard_capacity
  let general_three_quarter := three_quarters general_capacity
  let revenue_three_quarter := vip_three_quarter * vip_price + standard_three_quarter * standard_price + general_three_quarter * general_price
  let revenue_full := vip_capacity * vip_price + standard_capacity * standard_price + general_capacity * general_price
  revenue_three_quarter = 42000 ∧ (revenue_full - revenue_three_quarter) = 14000 :=
by
  sorry

end stadium_revenue_difference_l36_36552


namespace students_count_l36_36275

def students_count_problem (vampire_students pumpkin_students total_cost : ℕ) : Prop :=
  let pack_cost := 3
  let individual_cost := 1
  let vampire_packs := vampire_students / 5
  let vampire_individuals := vampire_students % 5
  let pumpkin_packs := pumpkin_students / 5
  let pumpkin_individuals := pumpkin_students % 5
  let cost := (vampire_packs + pumpkin_packs) * pack_cost + (vampire_individuals + pumpkin_individuals) * individual_cost
  cost = total_cost

theorem students_count : students_count_problem 11 14 17 →
    let total_students := 11 + 14 in
    total_students = 25 :=
begin
  sorry
end

end students_count_l36_36275


namespace value_of_x_l36_36805

theorem value_of_x (x y : ℝ) (h1 : x - y = 6) (h2 : x + y = 12) : x = 9 :=
by
  sorry

end value_of_x_l36_36805


namespace percentage_women_no_french_speak_spanish_german_l36_36070

variable (total_workforce : Nat)
variable (men_percentage women_percentage : ℕ)
variable (men_only_french men_only_spanish men_only_german : ℕ)
variable (men_both_french_spanish men_both_french_german men_both_spanish_german : ℕ)
variable (men_all_three_languages women_only_french women_only_spanish : ℕ)
variable (women_only_german women_both_french_spanish women_both_french_german : ℕ)
variable (women_both_spanish_german women_all_three_languages : ℕ)

-- Conditions
axiom h1 : men_percentage = 60
axiom h2 : women_percentage = 40
axiom h3 : women_only_french = 30
axiom h4 : women_only_spanish = 25
axiom h5 : women_only_german = 20
axiom h6 : women_both_french_spanish = 10
axiom h7 : women_both_french_german = 5
axiom h8 : women_both_spanish_german = 5
axiom h9 : women_all_three_languages = 5

theorem percentage_women_no_french_speak_spanish_german:
  women_only_spanish + women_only_german + women_both_spanish_german = 50 := by
  sorry

end percentage_women_no_french_speak_spanish_german_l36_36070


namespace measure_angle_ABG_l36_36148

theorem measure_angle_ABG (ABCDEFG : Polygon) (h_reg : regular_heptagon ABCDEFG) : 
  measure_angle ABCDEFG A B G = 25.715 := 
sorry

end measure_angle_ABG_l36_36148


namespace distance_AD_between_31_and_32_l36_36894

noncomputable def distanceAD (A B C D : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)

variables (A B C D : ℝ × ℝ)
variables (hB_east_A : B.1 = A.1 + 5 * real.sqrt 3 ∧ B.2 = A.2)
variables (hC_north_B : C.1 = B.1 ∧ C.2 = B.2 + 15)
variables (hAC_distance : real.dist A C = 10 * real.sqrt 3)
variables (hBAC_angle : real.angle A B C = real.pi / 6)
variables (hD_north_C : D.1 = C.1 ∧ D.2 = C.2 + 15)

theorem distance_AD_between_31_and_32 :
  31 < distanceAD A B C D ∧ distanceAD A B C D < 32 :=
sorry

end distance_AD_between_31_and_32_l36_36894


namespace prove_leftmost_three_digits_of_ring_arrangements_l36_36010

-- Definitions
def eight_rings : Nat := 8
def five_rings : Nat := 5
def four_fingers : Nat := 4

-- Conditions as definitions
def choose_rings := Nat.comb eight_rings five_rings
def arrange_rings := Nat.factorial five_rings
def distribute_rings := Nat.comb (five_rings + (four_fingers - 1)) (four_fingers - 1)

-- Calculate the total number of arrangements
def total_arrangements := choose_rings * arrange_rings * distribute_rings

-- Find the leftmost three nonzero digits of the total arrangements
def leftmost_three_nonzero_digits := 376

-- Prove the statement
theorem prove_leftmost_three_digits_of_ring_arrangements :
  (total_arrangements / 1000 % 1000) = leftmost_three_nonzero_digits := sorry

end prove_leftmost_three_digits_of_ring_arrangements_l36_36010


namespace sum_of_powers_mod_7_eq_6_l36_36061

theorem sum_of_powers_mod_7_eq_6 : 
  (1^6 + 2^6 + 3^6 + 4^6 + 5^6 + 6^6) % 7 = 6 :=
by
  -- Using Fermat's Little Theorem (proved elsewhere in mathlib)
  have h1 : 1^6 % 7 = 1 := by sorry,
  have h2 : 2^6 % 7 = 1 := by sorry,
  have h3 : 3^6 % 7 = 1 := by sorry,
  have h4 : 4^6 % 7 = 1 := by sorry,
  have h5 : 5^6 % 7 = 1 := by sorry,
  have h6 : 6^6 % 7 = 1 := by sorry,
  -- Summing and proving the final result
  calc
    (1^6 + 2^6 + 3^6 + 4^6 + 5^6 + 6^6) % 7
        = (1 + 1 + 1 + 1 + 1 + 1) % 7 := by rw [h1, h2, h3, h4, h5, h6]
    ... = 6 % 7 := by norm_num

end sum_of_powers_mod_7_eq_6_l36_36061


namespace cross_product_correct_l36_36308

theorem cross_product_correct (a : ℝ) :
  let v := ⟨3, a, 4⟩ : ℝ × ℝ × ℝ,
      w := ⟨-1, 2, a + 1⟩ : ℝ × ℝ × ℝ in
  (v.2 * w.3 - v.3 * w.2, v.3 * w.1 - v.1 * w.3, v.1 * w.2 - v.2 * w.1) = 
  (a^2 + a - 8, -3 * a - 7, 6 + a) :=
by sorry

end cross_product_correct_l36_36308


namespace total_cost_of_fencing_l36_36183

-- Define the concepts from the problem conditions as Lean variables/constants
constant length : ℝ   -- Given length of the plot in meters
constant breadth : ℝ  -- Calculated breadth of the plot in meters
constant cost_per_meter : ℝ  -- Cost of fencing per meter in currency units

-- Given conditions
axiom length_eq_64 : length = 64
axiom length_more_than_breadth : length = breadth + 28
axiom cost_per_meter_eq_26_5 : cost_per_meter = 26.5

-- Define the total cost and perimeter functions based on these conditions
def perimeter (length breadth : ℝ) : ℝ := 2 * (length + breadth)
def total_cost (cost_per_meter perimeter : ℝ) : ℝ := cost_per_meter * perimeter

-- The theorem to prove that the total cost of fencing the plot is 5300 currency units
theorem total_cost_of_fencing : total_cost cost_per_meter (perimeter length breadth) = 5300 :=
by
  rw [length_more_than_breadth, length_eq_64, cost_per_meter_eq_26_5]
  -- Convert length to exact value and find breadth
  sorry

end total_cost_of_fencing_l36_36183


namespace total_amount_spent_l36_36130

variable (f : ℕ) (Fp Fd : ℝ)
variable (b : ℕ) (Bp Bd2 : ℝ)
variable (c : ℕ) (Cp : ℝ)
variable (T : ℝ)

-- Given values
def given_values : Prop :=
  f = 12 ∧ Fp = 8 ∧ Fd = 3 ∧ 
  b = 7 ∧ Bp = 12 ∧ Bd2 = 0.30 ∧ 
  c = 10 ∧ Cp = 6 ∧ T = 0.075

-- Prove total amount spent is $183.18
theorem total_amount_spent (h : given_values f Fp Fd b Bp Bd2 c Cp T) : 
  (let films_cost := (Fp - Fd) * f in
   let books_cost := Bp + Bp * (1 - Bd2) * (b - 1) in
   let cds_cost := Cp * (c - c / 4) in
   let total_cost_before_tax := films_cost + books_cost + cds_cost in
   total_cost_before_tax + total_cost_before_tax * T) = 183.18 :=
by sorry

end total_amount_spent_l36_36130


namespace part_a_part_b_l36_36969

section part_a
variable (A D B C U : Type)
variable (BC AD : Type)
variable (AU DU : A → D → B → C → Type)
variable [h : BC ⊥ AD] -- given condition: BC is perpendicular to AD

-- Theorem: The altitudes dropped from vertices A and D to line BC intersect at the same point
theorem part_a : ∃ U, AU A D B C = DU A D B C := sorry
end part_a

section part_b
variable (A D B C U : Type)
variable (BC AD : Type)
variable (AU DU : A → D → B → C → Type)
variable [h : ∃ U, AU A D B C = DU A D B C] -- given condition: AU and DU intersect at U

-- Theorem: BC is perpendicular to AD
theorem part_b : BC ⊥ AD := sorry
end part_b

end part_a_part_b_l36_36969


namespace completing_square_l36_36720

theorem completing_square (x c d : ℝ) : x^2 - 6 * x + 5 = 0 → (x + c)^2 = d → d = 4 :=
by
  assume h1 : x^2 - 6 * x + 5 = 0
  assume h2 : (x + c)^2 = d
  sorry

end completing_square_l36_36720


namespace max_possible_length_of_PC_l36_36926

noncomputable def maxPC {A B C P : Type*} 
  [metric_space A] [metric_space B] [metric_space C] [metric_space P]
  (equilateral_triangle : EquilateralTriangle A B C)
  (AP : dist A P = 2)
  (BP : dist B P = 3) : ℝ :=
5

theorem max_possible_length_of_PC 
  {A B C P : Type*} 
  [metric_space A] [metric_space B] [metric_space C] [metric_space P]
  (equilateral_triangle : EquilateralTriangle A B C)
  (AP : dist A P = 2)
  (BP : dist B P = 3) : 
  ∃ C', dist P C' ≤ 5 :=
begin
  use maxPC equilateral_triangle AP BP,
  sorry
end

end max_possible_length_of_PC_l36_36926


namespace Ann_cycling_speed_is_6_l36_36210

-- Definitions based on the problem conditions
def Tom_speed : ℝ := 6
def Jerry_speed : ℝ := (3 / 4) * Tom_speed
def Ann_speed : ℝ := (4 / 3) * Jerry_speed

-- The formal statement we wish to prove
theorem Ann_cycling_speed_is_6 : Ann_speed = 6 := by
  sorry

end Ann_cycling_speed_is_6_l36_36210


namespace distribute_stickers_l36_36292

theorem distribute_stickers :
  ∃ (ways : ℕ), ways = Nat.choose (10 - 1) (5 - 1) ∧ ways = 126 :=
by
  have h : Nat.choose (10-1) (5-1) = 126 := by sorry
  use Nat.choose (10-1) (5-1)
  exact ⟨h, h⟩

end distribute_stickers_l36_36292


namespace tangent_line_at_one_monotonicity_l36_36373

open Real

-- Define the function f
def f (a x : ℝ) : ℝ := exp x - a * x - 1

-- The first part of the problem: tangent line equation at a specified point
theorem tangent_line_at_one (a : ℝ) (h : a = exp 1) : 
  ∀ x y, (x = 1 ∧ y = f a 1) → (f a)' 1 * (x - 1) + y = -1 :=
by sorry

-- The second part of the problem: monotonicity analysis
theorem monotonicity (a : ℝ) :
    if a ≤ 0 then ∀ x y, f a x < f a y ↔ x < y 
    else -∞ < ln a ∧ ∀ x, x < ln a → f a (ln a) < f a x ∧ ln a < x → f a (ln a) < f a x :=
by sorry

end tangent_line_at_one_monotonicity_l36_36373


namespace trajectory_of_Z_l36_36172

def Z_satisfies_condition (Z : ℂ) : Prop := |Z + complex.i| + |Z - complex.i| = 4

theorem trajectory_of_Z (Z : ℂ) (h : Z_satisfies_condition Z) : 
  ∃ f1 f2 : ℂ, f1 = complex.i ∧ f2 = -complex.i ∧ 
  ∀ w : ℂ, |w - f1| + |w - f2| = 4 → w = Z :=
sorry

end trajectory_of_Z_l36_36172


namespace arithmetic_sequence_a5_l36_36023

noncomputable def a_n (n : ℕ) : ℝ := sorry  -- The terms of the arithmetic sequence

theorem arithmetic_sequence_a5 :
  (∀ (n : ℕ), a_n n = a_n 0 + n * (a_n 1 - a_n 0)) →
  a_n 1 = 1 →
  a_n 1 + a_n 3 = 16 →
  a_n 4 = 15 :=
by {
  -- Proof omission, ensure these statements are correct with sorry
  sorry
}

end arithmetic_sequence_a5_l36_36023


namespace composite_appears_within_five_seconds_l36_36834

def is_prime (n : ℕ) : Prop := nat.prime n
def last_digit (n : ℕ) : ℕ := n % 10
def transform (n : ℕ) : ℕ := n + (last_digit n) + 1
def is_composite (n : ℕ) : Prop := ¬ is_prime n ∧ n > 1

theorem composite_appears_within_five_seconds :
  ∀ (n : ℕ), is_prime n → ∃ t ≤ 5, is_composite (nat.iterate transform t n) :=
by
  sorry

end composite_appears_within_five_seconds_l36_36834


namespace initial_oranges_is_23_l36_36985

variable (O : ℕ)

def initial_oranges (O : ℕ) : Prop :=
  let total_fruit_after_removal := 12 + (O - 15)
  let apples_percentage := 0.6 * total_fruit_after_removal
  apples_percentage = 12

theorem initial_oranges_is_23 : initial_oranges 23 :=
  sorry

end initial_oranges_is_23_l36_36985


namespace acute_triangle_inequality_l36_36576

theorem acute_triangle_inequality (A B C : ℝ) (hA : 0 < A ∧ A < π / 2) 
  (hB : 0 < B ∧ B < π / 2) (hC : 0 < C ∧ C < π / 2) (h_sum : A + B + C = π) :
  (sin (2 * A) + sin (2 * B))^2 / (sin A * sin B) +
  (sin (2 * B) + sin (2 * C))^2 / (sin B * sin C) +
  (sin (2 * C) + sin (2 * A))^2 / (sin C * sin A) ≤ 12 := 
sorry


end acute_triangle_inequality_l36_36576


namespace probability_divisor_of_12_on_12sided_die_l36_36612

theorem probability_divisor_of_12_on_12sided_die (h : ∀ i, 1 ≤ i ∧ i ≤ 12 → true) : 
  (probability (fun (n : ℕ) => n ∣ 12 ∧ 1 ≤ n ∧ n ≤ 12)) = 1/2 := 
sorry

end probability_divisor_of_12_on_12sided_die_l36_36612


namespace find_base_b4_l36_36073

theorem find_base_b4 (b_4 : ℕ) : (b_4 - 1) * (b_4 - 2) * (b_4 - 3) = 168 → b_4 = 8 :=
by
  intro h
  -- proof goes here
  sorry

end find_base_b4_l36_36073


namespace library_visitors_on_sundays_l36_36647

theorem library_visitors_on_sundays (S : ℕ) 
  (avg_non_sunday_visitors : ℕ := 140) 
  (days_in_month : ℕ := 30) 
  (avg_visitors_per_day : ℕ := 200) 
  (starts_with_sunday : Bool := true) 
  (total_days : ℕ := 30) 
  (sundays_in_month : ℕ := 5) 
  (other_days_in_month : ℕ := 25) : 
  5 * S + 25 * avg_non_sunday_visitors = avg_visitors_per_day * total_days →
  S = 500 := 
begin
  intro h,
  calc
    S = (avg_visitors_per_day * total_days - 25 * avg_non_sunday_visitors) / 5 : 
      by { field_simp [h], ring }
  ... = 500 : by sorry
end

end library_visitors_on_sundays_l36_36647


namespace A_plus_B_zero_l36_36456

def f (A B x : ℝ) : ℝ := 3 * A * x + 2 * B
def g (A B x : ℝ) : ℝ := 2 * B * x + 3 * A

theorem A_plus_B_zero (A B : ℝ) (h1 : A ≠ B) (h2 : ∀ x : ℝ, f A B (g A B x) - g A B (f A B x) = 3 * (B - A)) :
  A + B = 0 :=
sorry

end A_plus_B_zero_l36_36456


namespace eccentricity_of_ellipse_equation_of_ellipse_max_area_incircle_l36_36349

variables (a b c : ℝ) (x y : ℝ)
variables (a_gt_b : a > b) (b_pos : b > 0)
variables (F1 F2 A Q : ℝ × ℝ)

-- Definition of the ellipse
def is_ellipse (a b : ℝ) := (x^2 / a^2 + y^2 / b^2 = 1)

-- Conditions:
axiom ellipse_cond : is_ellipse a b
axiom foci_positions : F1 = (-c, 0) ∧ F2 = (c, 0)
axiom vertex_position : A = (0, b)
axiom line_perpendicular : Q = (-3 * c, 0)
axiom vector_relationship : 2 * ⟨F2.1 - F1.1, F2.2 - F1.2⟩ + ⟨Q.1 - F2.1, Q.2 - F2.2⟩ = (0,0)

-- The eccentricity of the ellipse
theorem eccentricity_of_ellipse : a^2 = 4 * c^2 → e = c / a → e = 1 / 2 := sorry

-- Equation of the ellipse given an additional circle
theorem equation_of_ellipse : c = 1 → a = 2 * c → b = sqrt 3 →
  (is_ellipse 2 sqrt 3) := sorry

-- Maximum area of the incircle of a triangle F1MN
theorem max_area_incircle (line_eq : ℝ → ℝ): 
  (∀ m y1 y2, line_eq y1 = F2.1 → line_eq y2 = F2.1 → y1 * y2 < 0) →
  ∃ R_max,
    R_max = 3 / 4 ∧
    incircle_area R_max = 9 * π / 16 ∧
    (line_eq = (λ y, 1)) := sorry

end eccentricity_of_ellipse_equation_of_ellipse_max_area_incircle_l36_36349


namespace distance_CD_eq_12_div_AB_l36_36898

theorem distance_CD_eq_12_div_AB
  (A B C D : ℝ × ℝ) 
  (hAD : dist A D = 2)
  (hBC : dist B C = 2)
  (hAC : dist A C = 4)
  (hBD : dist B D = 4)
  (hIntersect : ∃ P, P ∈ line_segment A C ∧ P ∈ line_segment B D) :
  dist C D = 12 / dist A B :=
sorry

end distance_CD_eq_12_div_AB_l36_36898


namespace total_handshakes_at_convention_l36_36545

def number_of_gremlins := 30
def number_of_imps := 20
def disagreeing_imps := 5
def specific_gremlins := 10

theorem total_handshakes_at_convention : 
  (number_of_gremlins * (number_of_gremlins - 1) / 2) +
  ((number_of_imps - disagreeing_imps) * number_of_gremlins) + 
  (disagreeing_imps * (number_of_gremlins - specific_gremlins)) = 985 :=
by 
  sorry

end total_handshakes_at_convention_l36_36545


namespace magnitude_of_vector_difference_l36_36015

variables (a b : ℝ^3)
variables (ha : ∥a∥ = 4) (hb : ∥b∥ = 3)
variables (angle_ab : real.angle_between a b = real.pi / 3)

theorem magnitude_of_vector_difference : ∥2 • a - b∥ = 7 :=
sorry

end magnitude_of_vector_difference_l36_36015


namespace volume_of_transport_cube_l36_36869

def height_of_model : ℕ := 15
def base_side_length_of_model : ℕ := 8

theorem volume_of_transport_cube :
  ∃ (box_side_length : ℕ), box_side_length = height_of_model ∧ box_side_length ^ 3 = 3375 :=
begin
  use height_of_model,
  split,
  { refl },
  { sorry }
end

end volume_of_transport_cube_l36_36869


namespace arc_length_central_angle_l36_36182

theorem arc_length_central_angle (r a : ℝ) (h1 : ∃ r, r = a * (√2 / 2)) (l : ℝ) (h2 : l = a) : (l / r) = √2 :=
by
  sorry

end arc_length_central_angle_l36_36182


namespace necessary_condition_x_pow_2_minus_x_lt_0_l36_36043

theorem necessary_condition_x_pow_2_minus_x_lt_0 (x : ℝ) : (x^2 - x < 0) → (-1 < x ∧ x < 1) := by
  intro hx
  sorry

end necessary_condition_x_pow_2_minus_x_lt_0_l36_36043


namespace sequence_fifth_element_l36_36979

theorem sequence_fifth_element :
  let a : ℕ → ℤ := λ n, (-1)^n * (2^n - 1) in
  a 5 = -31 :=
by
  sorry

end sequence_fifth_element_l36_36979


namespace fraction_of_data_less_than_mode_is_two_fifths_l36_36421

def data_list : List ℕ := [1, 2, 3, 4, 5, 5, 5, 6, 7, 7, 7, 7, 7, 11, 21, 28, 35, 43, 58]

def mode (l : List ℕ) : ℕ :=
l.mode

def fraction_less_than_mode (l : List ℕ) (m : ℕ) : ℚ :=
let count_less := l.countp (· < m)
let total_count := l.length
(count_less : ℚ) / (total_count : ℚ)

theorem fraction_of_data_less_than_mode_is_two_fifths :
  fraction_less_than_mode data_list (mode data_list) = 2 / 5 := sorry

end fraction_of_data_less_than_mode_is_two_fifths_l36_36421


namespace minimum_soaps_for_greater_discount_l36_36588

theorem minimum_soaps_for_greater_discount :
  ∀ x : ℕ, x ≥ 4 → let f := 1.4 * x + 0.6
                   let g := 1.6 * x
                   in f - g < 0 :=
by sorry

end minimum_soaps_for_greater_discount_l36_36588


namespace shaded_area_of_circles_l36_36694

theorem shaded_area_of_circles (r1 r2 : ℝ) (h1 : r1 = 4) (h2 : r2 = 5) :
  let R := r1 + r2 in
  let area_large_circle := π * R^2 in
  let area_small_circle1 := π * r1^2 in
  let area_small_circle2 := π * r2^2 in
  area_large_circle - area_small_circle1 - area_small_circle2 = 40 * π :=
by
  sorry

end shaded_area_of_circles_l36_36694


namespace cistern_filling_time_l36_36592

theorem cistern_filling_time :
  ∃ T : ℝ, (1 / T - 1 / 6 = 1 / 30) → T = 5 :=
begin
  use 5,
  intro h,
  field_simp at *,
  linarith,
end

end cistern_filling_time_l36_36592


namespace parallelogram_statements_correct_weighing_l36_36963

-- Definitions of the conditions/statements about parallelograms
def statement1 (p : Parallelogram) : Prop := 
  isAxisymmetric p ∧ isCentrallySymmetric p

def statement2 (p : Parallelogram) : Prop := 
  oppositeSidesEqual p ∧ diagonalsComplementary p

def statement3 (p : Parallelogram) : Prop := 
  diagonalsBisectEachOther p

def statement4 (p : Parallelogram) : Prop := 
  isUnstable p

-- The proof problem statement
theorem parallelogram_statements_correct_weighing :
  ∀ (p : Parallelogram),
  (statement1 p = false) ∧ 
  (statement2 p = false) ∧ 
  (statement3 p = true) ∧ 
  (statement4 p = true) :=
by
  -- Proof is omitted
  sorry

end parallelogram_statements_correct_weighing_l36_36963


namespace median_equal_half_hypotenuse_l36_36515

-- Assume a right-angled triangle with vertices A, B, and C, where ∠ABC = 90°
def is_right_angle_triangle (A B C : ℝ×ℝ) : Prop := 
  ∠ABC = 90° -- This would need to be defined rigorously in Lean usually using dot products.

-- Define the median on the hypotenuse
def median_on_hypotenuse (A B C M : ℝ×ℝ) (h : is_right_angle_triangle A B C) : Prop :=
  -- M is the midpoint of segment AC
  (M = midpoint A C)

-- The theorem statement that needs to be proven
theorem median_equal_half_hypotenuse 
  (A B C M : ℝ×ℝ)
  (h : is_right_angle_triangle A B C)
  (hmedian : median_on_hypotenuse A B C M h) :
  dist B M = (1/2) * dist A C :=
sorry

end median_equal_half_hypotenuse_l36_36515


namespace probability_of_rolling_divisor_of_12_l36_36602

def is_divisor (a b : ℕ) : Prop := b % a = 0

noncomputable def probability_divisor_12 : ℚ :=
  let divisors := { n | is_divisor n 12 ∧ n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} }
  let favorable_outcomes := (divisors : set ℕ).to_finset.card
  let total_outcomes := 12
  (favorable_outcomes : ℚ) / total_outcomes

theorem probability_of_rolling_divisor_of_12 : probability_divisor_12 = 1 / 2 := by
  sorry

end probability_of_rolling_divisor_of_12_l36_36602


namespace sum_of_fractions_equals_three_l36_36181

-- Definitions according to the conditions
def proper_fraction (a b : ℕ) := 1 ≤ a ∧ a < b
def improper_fraction (a b : ℕ) := a ≥ b
def mixed_number (a b c : ℕ) := a + b / c

-- Constants according to the given problem
def n := 8
def d := 9
def improper_n := 9

-- Values for elements in the conditions
def largest_proper_fraction := n / d
def smallest_improper_fraction := improper_n / d
def smallest_mixed_number := 1 + 1 / d

-- Theorem statement with the correct answer
theorem sum_of_fractions_equals_three :
  largest_proper_fraction + smallest_improper_fraction + smallest_mixed_number = 3 :=
sorry

end sum_of_fractions_equals_three_l36_36181


namespace piecewise_function_evaluation_l36_36129

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - real.sqrt x else real.pow 2 x

theorem piecewise_function_evaluation : f (f (-2)) = 1 / 2 :=
by
  sorry

end piecewise_function_evaluation_l36_36129


namespace proof_of_a_b_and_T_l36_36779

-- Define sequences and the given conditions

def a (n : ℕ) : ℕ := 2^(n-1)

def b (n : ℕ) : ℕ := 2 * n

def S (n : ℕ) : ℕ := 2^n - 1

def c (n : ℕ) : ℚ := 1 / ((b n)^2 - 1)

def T (n : ℕ) : ℚ := (n : ℚ) / (2 * n + 1)

axiom b_condition : ∀ n : ℕ, n > 0 → (b n + 2 * n = 2 * (b (n-1)) + 4)

axiom S_condition : ∀ n : ℕ, S n = 2^n - 1

theorem proof_of_a_b_and_T (n : ℕ) (h : n > 0) : 
  (∀ k, a k = 2^(k-1)) ∧ 
  (∀ k, b k = 2 * k) ∧ 
  (∀ k, T k = (k : ℚ) / (2 * k + 1)) := by
  sorry

end proof_of_a_b_and_T_l36_36779


namespace probability_divisor_of_12_l36_36606

theorem probability_divisor_of_12 (fair_die : True) (sides : Nat) (h : sides = 12) : 
  ∃ p : ℚ, p = 1 / 2 ∧ ∀ roll : Fin sides, is_divisor roll.val 12 ↔ roll.val ∈ {1, 2, 3, 4, 6, 12} → p = 1 / 2 :=
by sorry

def is_divisor (a b : Nat) : Prop := b % a = 0

end probability_divisor_of_12_l36_36606


namespace circle_properties_l36_36171

noncomputable def diameter (C : ℝ) := C / π
noncomputable def area (C : ℝ) := C * (C / (4 * π))

theorem circle_properties (C : ℝ) (hC : C = 24) :
  diameter C = 24 / π ∧ area C = 144 / π :=
by
  sorry

end circle_properties_l36_36171


namespace probability_divisor_of_12_l36_36605

theorem probability_divisor_of_12 (fair_die : True) (sides : Nat) (h : sides = 12) : 
  ∃ p : ℚ, p = 1 / 2 ∧ ∀ roll : Fin sides, is_divisor roll.val 12 ↔ roll.val ∈ {1, 2, 3, 4, 6, 12} → p = 1 / 2 :=
by sorry

def is_divisor (a b : Nat) : Prop := b % a = 0

end probability_divisor_of_12_l36_36605


namespace sequences_converge_and_find_limits_l36_36445

theorem sequences_converge_and_find_limits (x y : ℕ → ℝ)
  (h1 : x 1 = 1)
  (h2 : y 1 = Real.sqrt 3)
  (h3 : ∀ n : ℕ, x (n + 1) * y (n + 1) = x n)
  (h4 : ∀ n : ℕ, x (n + 1)^2 + y n = 2) :
  ∃ (Lx Ly : ℝ), (∀ ε : ℝ, ε > 0 → ∃ N : ℕ, ∀ n ≥ N, |x n - Lx| < ε) ∧ 
                  (∀ ε : ℝ, ε > 0 → ∃ N : ℕ, ∀ n ≥ N, |y n - Ly| < ε) ∧ 
                  Lx = 0 ∧ 
                  Ly = 2 := 
sorry

end sequences_converge_and_find_limits_l36_36445


namespace symmetry_plane_symmetry_half_plane_symmetry_quadrant_l36_36916

theorem symmetry_plane (P : Type*) [plane : is_plane P]
  (L : line P) (pt : P) :
  (is_axis_of_symmetry plane L) ∧ (is_center_of_symmetry plane pt) :=
by sorry

theorem symmetry_half_plane (H : Type*) [half_plane : is_half_plane H]
  (L_perp : perp_line H) :
  (is_axis_of_symmetry half_plane L_perp) ∧ ¬ (exists pt : H, is_center_of_symmetry half_plane pt) :=
by sorry

theorem symmetry_quadrant (Q : Type*) [quadrant : is_quadrant Q]
  (x_axis y_axis : axis Q) :
  (is_axis_of_symmetry quadrant x_axis) ∧ (is_axis_of_symmetry quadrant y_axis) :=
by sorry

end symmetry_plane_symmetry_half_plane_symmetry_quadrant_l36_36916


namespace probability_divisor_of_12_is_half_l36_36620

noncomputable def divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ d, n % d = 0) (Finset.range (n + 1))

def probability_divisor_of_12 : ℚ :=
  let total_outcomes := 12
  let favorable_outcomes := (divisors 12).card
  favorable_outcomes / total_outcomes

theorem probability_divisor_of_12_is_half :
  probability_divisor_of_12 = 1 / 2 :=
  by
    sorry

end probability_divisor_of_12_is_half_l36_36620


namespace helga_shoe_pairs_l36_36048

theorem helga_shoe_pairs
  (first_store_pairs: ℕ) 
  (second_store_pairs: ℕ) 
  (third_store_pairs: ℕ)
  (fourth_store_pairs: ℕ)
  (h1: first_store_pairs = 7)
  (h2: second_store_pairs = first_store_pairs + 2)
  (h3: third_store_pairs = 0)
  (h4: fourth_store_pairs = 2 * (first_store_pairs + second_store_pairs + third_store_pairs))
  : first_store_pairs + second_store_pairs + third_store_pairs + fourth_store_pairs = 48 := 
by
  sorry

end helga_shoe_pairs_l36_36048


namespace number_of_correct_propositions_l36_36044

variables (m n l : Type) (α β γ : Type)
variables (m_parallel_n : m ∥ n) (m_perp_alpha : m ⟂ α) (α_perp_beta : α ⟂ β) 
variables (intersection : α ∩ β = {m}) (n_perp_m: n ⟂ m) (α_parallel_beta : α ∥ β)
variables (beta_perp_gamma : β ⟂ γ) (n_parallel_beta : n ∥ β)

def prop1 := (m ∥ n) ∧ (m ⟂ α) → (n ⟂ α)
def prop2 := (α ⟂ β) ∧ (α ∩ β = {m}) ∧ (n ⟂ m) → (n ⟂ β)
def prop3 := (α ∥ β) ∧ (β ⟂ γ) → (α ⟂ γ)
def prop4 := (α ∥ β) ∧ (m ⟂ α) ∧ (n ∥ β) → (m ⟂ n)

theorem number_of_correct_propositions : 
  (prop1 m n α m_parallel_n m_perp_alpha ∧ prop3 α β γ α_parallel_beta beta_perp_gamma ∧ prop4 α β m n l α_parallel_beta m_perp_alpha n_parallel_beta) →
  3 := 
sorry

end number_of_correct_propositions_l36_36044


namespace graph_of_g_plus_2_is_A_l36_36788

-- Define the piecewise function g(x)
def g (x : ℝ) : ℝ :=
  if -2 ≤ x ∧ x ≤ 1 then -x
  else if 1 < x ∧ x ≤ 3 then Real.sqrt (4 - (x - 1)^2) + 1
  else if 3 < x ∧ x ≤ 4 then 3 * (x - 3)
  else 0

-- The graph of y = g(x) + 2 should match the transformation provided
theorem graph_of_g_plus_2_is_A :
  ∃ A, A = ∀ x, (g(x) + 2) = piecewise [(-2, 4)] (λ x, (g(x) + 2)) :=
sorry

end graph_of_g_plus_2_is_A_l36_36788


namespace no_valid_arrangement_l36_36690

open Nat

theorem no_valid_arrangement :
  ¬ ∃ (f : Fin 30 → ℕ), 
    (∀ (i : Fin 30), 1 ≤ f i ∧ f i ≤ 30) ∧ 
    (∀ (i : Fin 30), ∃ n : ℕ, (f i + f (i + 1) % 30) = n^2) ∧ 
    (∀ i1 i2, i1 ≠ i2 → f i1 ≠ f i2) :=
  sorry

end no_valid_arrangement_l36_36690


namespace probability_divisor_of_12_on_12sided_die_l36_36613

theorem probability_divisor_of_12_on_12sided_die (h : ∀ i, 1 ≤ i ∧ i ≤ 12 → true) : 
  (probability (fun (n : ℕ) => n ∣ 12 ∧ 1 ≤ n ∧ n ≤ 12)) = 1/2 := 
sorry

end probability_divisor_of_12_on_12sided_die_l36_36613


namespace Lucy_retirement_month_l36_36872

theorem Lucy_retirement_month (start_month : ℕ) (duration : ℕ) (March : ℕ) (May : ℕ) : 
  (start_month = March) ∧ (duration = 3) → (start_month + duration - 1 = May) :=
by
  intro h
  have h_start_month := h.1
  have h_duration := h.2
  sorry

end Lucy_retirement_month_l36_36872


namespace angle_PFA_eq_angle_PFB_trajectory_of_centroid_G_l36_36678

-- Conditions
def parabola (x : ℝ) : ℝ := x^2
def line (x : ℝ) (y : ℝ) : Prop := x - y - 2 = 0
def focus_C : (ℝ × ℝ) := (0, 1/4)

-- Questions
def trajectory_centroid_G (x y : ℝ) : Prop := y = (1/3)*(4*x^2 - x + 2)

theorem angle_PFA_eq_angle_PFB (x0 x1 y0 y1 xP yP : ℝ) (h_tangent_A : 2*x0*x - y - x0^2 = 0) (h_tangent_B : 2*x1*x - y - x1^2 = 0) 
(h_point_P : (xP, yP) ∈ (λ P, line P.1 P.2)) (point_A_parabola : parabola x0 = y0) (point_B_parabola : parabola x1 = y1) :
∠ ((xP, yP) - (x0, y0)) ((0, 1/4) - (xP, yP)) ((xP, yP) - (x1, y1)) = ∠ ((xP, yP) - (x1, y1)) ((0, 1/4) - (xP, yP)) ((xP, yP) - (x0, y0)) :=
sorry

theorem trajectory_of_centroid_G (x_G y_G : ℝ) (Px Py : ℝ) (h_line : line Px Py) :
trajectory_centroid_G x_G y_G :=
sorry

end angle_PFA_eq_angle_PFB_trajectory_of_centroid_G_l36_36678


namespace bus_stop_time_l36_36572

-- Usual time to walk to the bus stop
def usual_time (T : ℕ) := T

-- Usual speed
def usual_speed (S : ℕ) := S

-- New speed when walking at 4/5 of usual speed
def new_speed (S : ℕ) := (4 * S) / 5

-- Time relationship when walking at new speed
def time_relationship (T : ℕ) (S : ℕ) := (S / ((4 * S) / 5)) = (T + 10) / T

-- Prove the usual time T is 40 minutes
theorem bus_stop_time (T S : ℕ) (h1 : time_relationship T S) : T = 40 :=
by
  sorry

end bus_stop_time_l36_36572


namespace percentage_first_of_second_l36_36550

theorem percentage_first_of_second (x : ℝ) (h1 : x > 0) :
    let a := 1.19 * x
    let b := 1.40 * x
    ((a / b) * 100) = 85 :=
by
  let a := 1.19 * x
  let b := 1.40 * x
  have h2 : b ≠ 0 := by linarith
  have h3 : (a / b) * 100 = (1.19 / 1.40) * 100 := by sorry
  have h4 : (1.19 / 1.40) * 100 = 85 := by sorry
  exact h4

end percentage_first_of_second_l36_36550


namespace find_triangle_area_l36_36046

noncomputable def triangle_area_problem
  (a b : ℝ)
  (h1 : a + b = 4)
  (h2 : a^2 + b^2 = 14) : ℝ :=
  (1 / 2) * a * b

theorem find_triangle_area
  (a b : ℝ)
  (h1 : a + b = 4)
  (h2 : a^2 + b^2 = 14) :
  triangle_area_problem a b h1 h2 = 1 / 2 := by
  sorry

end find_triangle_area_l36_36046


namespace circle_m_value_l36_36026

theorem circle_m_value (m : ℝ) :
  (∀ (x y : ℝ), (x^2 + y^2 - 2*x - 4*y + m = 0) → (x + 2*y - 4 = 0) → (om_perpendicular_on x y)) →
  m = 8 / 5 :=
begin
  sorry
end

def om_perpendicular_on (x y : ℝ) : Prop :=
  let (x1, y1), (x2, y2) := (x, y), (-4 + 2*y, y) in
  x1 * x2 + y1 * y2 = 0

end circle_m_value_l36_36026


namespace number_of_parents_l36_36200

theorem number_of_parents {girls boys groups group_size : ℕ} 
  (h_girls : girls = 14)
  (h_boys : boys = 11)
  (h_groups : groups = 3)
  (h_group_size : group_size = 25) :
  let total_people := groups * group_size
      total_children := girls + boys
      parents := total_people - total_children
  in parents = 50 :=
by
  sorry

end number_of_parents_l36_36200


namespace georgia_total_cost_l36_36887

def carnation_price : ℝ := 0.50
def dozen_price : ℝ := 4.00
def teachers : ℕ := 5
def friends : ℕ := 14

theorem georgia_total_cost :
  ((dozen_price * teachers) + dozen_price + (carnation_price * (friends - 12))) = 25.00 :=
by
  sorry

end georgia_total_cost_l36_36887


namespace area_of_sector_l36_36919

theorem area_of_sector (r : ℝ) (θ : ℝ) (h1 : r = 10) (h2 : θ = π / 5) : 
  (1 / 2) * r * r * θ = 10 * π :=
by
  rw [h1, h2]
  sorry

end area_of_sector_l36_36919


namespace log_y_x_four_eq_two_thirds_l36_36323

variable (x y : ℝ)

-- Given condition
def logx_y_cubed_eq_two : Prop := Real.log x (y^3) = 2

-- Goal to prove
theorem log_y_x_four_eq_two_thirds (h : logx_y_cubed_eq_two x y) : Real.log y (x^4) = 2 / 3 := 
  sorry

end log_y_x_four_eq_two_thirds_l36_36323


namespace speaking_orders_possible_l36_36280

theorem speaking_orders_possible (n m : ℕ) (hn : n = 3) (hm : m = 3) : 
  ∃ (possible_orders : ℕ), possible_orders = 2 * (Finset.perm_univ_card ⟨Finset.range n, Finset.range n⟩) * (Finset.perm_univ_card ⟨Finset.range m, Finset.range m⟩) ∧ possible_orders = 72 := 
by {
  sorry
}

end speaking_orders_possible_l36_36280


namespace find_S5_l36_36536

-- Assuming the sequence is geometric and defining the conditions
variables {a : ℕ → ℝ} {q : ℝ}

-- Definitions of the conditions based on the problem
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n+1) = a n * q

def condition_1 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 2 * a 5 = 3 * a 3

def condition_2 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 4 + 9 * a 7) / 2 = 2

-- Sum of the first n terms of a geometric sequence
noncomputable def S_n (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a 0 * (1 - q ^ n) / (1 - q)

-- The theorem stating the final goal
theorem find_S5 (a : ℕ → ℝ) (q : ℝ) (h_seq : geometric_sequence a q) 
    (h1 : condition_1 a q) (h2 : condition_2 a q) : S_n a q 5 = 121 :=
by
  -- This adds "sorry" to bypass the actual proof
  sorry

end find_S5_l36_36536


namespace number_of_apples_l36_36986

variable (A : ℕ) -- Number of apples in the bowl
variable (O : ℕ) -- Number of oranges in the bowl (23 initially)

-- Initial conditions
axiom initial_oranges : O = 23
axiom removed_oranges : 17
axiom oranges_left : O - removed_oranges = 6
axiom apples_percentage : (A : ℝ) / (A + 6) = 0.70

theorem number_of_apples : A = 14 :=
by
  sorry

end number_of_apples_l36_36986


namespace hyperbola_eccentricity_l36_36012

theorem hyperbola_eccentricity (a b c : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_hyperbola : a^2 + b^2 = c^2) :
  let e := c / a in e = sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_l36_36012


namespace root_expression_l36_36300

noncomputable def a := -(1 / 2 : ℝ)
noncomputable def b := (Real.sqrt 59) / 2
def cubic_eq (x : ℂ) := 2 * x ^ 3 - 4 * x + 5

theorem root_expression :
  (∃ (x : ℂ), cubic_eq x = 0 ∧ x = a + b * complex.i) →
  a + b^2 = 14 :=
by
  sorry

end root_expression_l36_36300


namespace sum_of_M_l36_36189

theorem sum_of_M (x y z : ℕ) (h1 : z = 2 * (x + y)) (h2 : xyz = 8 * (x + y + z)) :
  ∑ {M : ℕ // ∃ (x y : ℕ), z = 2 * (x + y) ∧ xyz = 8 * (x + y + z) ∧ M = xyz} = 560 :=
sorry

end sum_of_M_l36_36189


namespace sum_ge_threshold_l36_36561

def numbers : List ℝ := [0.8, 1/2, 0.3]
def threshold : ℝ := 0.4

theorem sum_ge_threshold (l : List ℝ) (t : ℝ) : (l.filter (λ x => x >= t)).sum = 1.3 :=
by
  have h : l = [0.8, 1/2, 0.3] := by rfl
  have ht : t = 0.4 := by rfl
  rw [h, ht]
  calc
  (([0.8, 1/2, 0.3].filter (λ x => x >= 0.4))).sum
      = ([0.8, 0.5]).sum : by
        simp [List.filter]
      = 1.3 : by norm_num

end sum_ge_threshold_l36_36561


namespace coefficient_x3_in_binomial_expansion_l36_36083

theorem coefficient_x3_in_binomial_expansion :
  nat.choose 50 3 = 19600 :=
by
  -- Proof goes here
  sorry

end coefficient_x3_in_binomial_expansion_l36_36083


namespace max_sum_a_b_c_l36_36770

noncomputable def f (a b c x : ℝ) : ℝ := a * Real.cos x + b * Real.cos (2 * x) + c * Real.cos (3 * x)

theorem max_sum_a_b_c (a b c : ℝ) (h : ∀ x : ℝ, f a b c x ≥ -1) : a + b + c ≤ 3 :=
sorry

end max_sum_a_b_c_l36_36770


namespace math_proof_problem_l36_36027

variables (k m n : ℕ)

-- First condition
def condition1 : Prop := (1/2 : ℚ)^16 * (1/81)^k = (1/18)^16

-- Second condition
def condition2 (k : ℕ) : Prop := (1/3 : ℚ)^n * (1/27)^m = (1/18)^k

theorem math_proof_problem (h1 : condition1 k) (h2 : condition2 k) : k = 8 ∧ (n + 3 * m = 16) := 
by {
  sorry
}

end math_proof_problem_l36_36027


namespace circle_equation_tangent_l36_36930

noncomputable def circle_tangent_line (x y : ℝ) : Prop :=
  (3 * x + 4 * y = 0)

noncomputable def circle_center (x y : ℝ) : Prop :=
  (x = 3) ∧ (y = -1)

theorem circle_equation_tangent
  (x y : ℝ)
  (hx : circle_center x y)
  (r : ℝ)
  (hr : distance (x, y) (λ (x y : ℝ), 3 * x + 4 * y = 0) = r)
  (ht : circle_tangent_line (x + r) y) :
  (x - 3) ^ 2 + (y + 1) ^ 2 = 1 := by
  sorry

end circle_equation_tangent_l36_36930


namespace complement_of_A_in_U_l36_36867

theorem complement_of_A_in_U :
    ∀ (U A : Set ℕ),
    U = {1, 2, 3, 4} →
    A = {1, 3} →
    (U \ A) = {2, 4} :=
by
  intros U A hU hA
  rw [hU, hA]
  sorry

end complement_of_A_in_U_l36_36867


namespace intersection_points_depend_on_discriminant_l36_36336

variables {a b c : ℝ}

theorem intersection_points_depend_on_discriminant 
  (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 0) : 
  (∃ Δ, Δ = b^2 - 4 * a * c) → 
  (∀ Δ, 
    if Δ > 0 then exists_unique (λ x, ax^2 + bx + c = 0) 2 
    else if Δ = 0 then exists_unique (λ x, ax^2 + bx + c = 0) 1 
    else ¬ (∃ x, ax^2 + bx + c = 0)) :=
sorry

end intersection_points_depend_on_discriminant_l36_36336


namespace time_to_cross_bridge_l36_36051

-- Definition of parameters
def train_length : ℝ := 110 -- in meters
def bridge_length : ℝ := 170 -- in meters
def train_speed_kmph : ℝ := 36 -- in kilometers per hour

-- Conversion factor from kmph to m per s
def kmph_to_mps (speed_kmph : ℝ) : ℝ := (speed_kmph * 1000) / 3600

-- Converting the speed of the train to m/s
def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph

-- Total distance to be covered
def total_distance : ℝ := train_length + bridge_length

-- Time formula: Time = Distance / Speed
def time_taken (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

-- Proving the correct answer
theorem time_to_cross_bridge : time_taken total_distance train_speed_mps = 28 :=
by
  sorry

end time_to_cross_bridge_l36_36051


namespace problem_solution_l36_36757

theorem problem_solution (a : ℝ) (h : a = Real.sqrt 5 - 1) :
  2 * a^3 + 7 * a^2 - 2 * a - 12 = 0 :=
by 
  sorry  -- Proof placeholder

end problem_solution_l36_36757


namespace molecular_weight_constant_l36_36558

variable (n : ℕ) -- n represents the number of moles
constant molecular_weight : ℕ := 588 -- molecular weight of the compound in g/mol

theorem molecular_weight_constant (n : ℕ) : molecular_weight = 588 :=
by simp [molecular_weight]

end molecular_weight_constant_l36_36558


namespace P_subsetneq_Q_l36_36469

def P : Set ℝ := { x : ℝ | x > 1 }
def Q : Set ℝ := { x : ℝ | x^2 - x > 0 }

theorem P_subsetneq_Q : P ⊂ Q :=
by
  sorry

end P_subsetneq_Q_l36_36469


namespace quadratic_rewriting_l36_36192

theorem quadratic_rewriting :
  ∃ a b c : ℝ, 
  (-8 * (x^2) + 16 * x + 320) = (a * ((x + b) ^ 2) + c) ∧ (a + b + c) = 319 :=
begin
  use [-8, -1, 328],
  split,
  simp [*, pow_two, mul_add, add_mul, add_assoc],
  linarith,
end

end quadratic_rewriting_l36_36192


namespace solve_for_x_l36_36913

theorem solve_for_x (x y : ℝ) : (19 * (x + y) + 17 = 19 * (-x + y) - 21) → (x = -1) :=
begin
  intro h,
  sorry,
end

end solve_for_x_l36_36913


namespace sqrt_of_square_of_neg_five_eq_five_l36_36687

theorem sqrt_of_square_of_neg_five_eq_five : Real.sqrt ((-5 : ℤ) ^ 2) = 5 := by
  sorry

end sqrt_of_square_of_neg_five_eq_five_l36_36687


namespace road_path_distance_l36_36753

theorem road_path_distance (d_AB d_AC d_BC d_BD : ℕ) 
  (h1 : d_AB = 9) (h2 : d_AC = 13) (h3 : d_BC = 8) (h4 : d_BD = 14) : A_to_D = 19 :=
by
  sorry

end road_path_distance_l36_36753


namespace find_k_l36_36058

theorem find_k (k : ℝ) : (1 / 2) ^ 23 * (1 / 81) ^ k = 1 / 18 ^ 23 → k = 5.75 :=
by
  intros h
  sorry

end find_k_l36_36058


namespace find_m_l36_36001

theorem find_m 
  (m : ℝ) 
  (hc : ∀ x y : ℝ, x^2 + y^2 - 4 * x + 2 * y + m = 0) 
  (intersects_y_axis : ∃ y₁ y₂ : ℝ, hc 0 y₁ ∧ hc 0 y₂) 
  (right_angle_at_C : ∀ A B : ℝ, ∃ x y : ℝ, x = 0 ∧ (y = A ∨ y = B) ∧ (angle_between_points (2, -1) (0, A) (0, B) = 90)) :
  m = -3 :=
sorry

end find_m_l36_36001


namespace total_height_increase_in_4_centuries_l36_36385

def height_increase_per_decade : ℕ := 75
def years_per_century : ℕ := 100
def years_per_decade : ℕ := 10
def centuries : ℕ := 4

theorem total_height_increase_in_4_centuries :
  height_increase_per_decade * (centuries * years_per_century / years_per_decade) = 3000 := by
  sorry

end total_height_increase_in_4_centuries_l36_36385


namespace one_cow_eating_one_bag_in_12_days_l36_36069

def average_days_to_eat_one_bag (total_bags : ℕ) (total_days : ℕ) (number_of_cows : ℕ) : ℕ :=
  total_days / (total_bags / number_of_cows)

theorem one_cow_eating_one_bag_in_12_days (total_bags : ℕ) (total_days : ℕ) (number_of_cows : ℕ) (h_total_bags : total_bags = 50) (h_total_days : total_days = 20) (h_number_of_cows : number_of_cows = 30) : 
  average_days_to_eat_one_bag total_bags total_days number_of_cows = 12 := by
  sorry

end one_cow_eating_one_bag_in_12_days_l36_36069


namespace cost_price_per_metre_correct_l36_36439

noncomputable def original_cost := 500.0
noncomputable def discount_percentage := 0.15
noncomputable def total_metres := 9.25
noncomputable def final_price := 425.50
noncomputable def cost_price_per_metre := final_price / total_metres

theorem cost_price_per_metre_correct :
  cost_price_per_metre ≈ 46.00 := by
  sorry

end cost_price_per_metre_correct_l36_36439


namespace tiling_4x9_l36_36574

def tiling_rec : ℕ → ℕ
| 0       := 1
| 1       := 1
| n + 2   := tiling_rec (n + 1) + tiling_rec n + 2 * (g_rec (n + 1)) + (h_rec (n + 1))
| n       := tiling_rec (n - 1) + tiling_rec (n - 2) + 2 * (g_rec (n - 1)) + (h_rec (n - 1)) -- fallback for cases greater than 1

def g_rec : ℕ → ℕ
| 0       := 0
| 1       := 1
| n + 1   := g_rec n + tiling_rec n
| n       := g_rec (n - 1) + tiling_rec (n - 1) -- fallback for cases greater than 1

def h_rec : ℕ → ℕ
| 0       := 0
| 1       := 1
| n + 2   := h_rec n + tiling_rec (n - 1)
| n       := if n = 1 then 1 else 0 -- fallback for cases greater than 1

theorem tiling_4x9 : tiling_rec 9 = 6336 := by
sorry

end tiling_4x9_l36_36574


namespace triangle_area_doubling_side_a_l36_36661

variable {a b θ : ℝ} -- Real numbers for sides and angle

-- Assume the basic properties for the area of a triangle
theorem triangle_area_doubling_side_a (h₁: a > 0) (h₂: b > 0) (h₃: 0 < θ ∧ θ < π):
  let original_area := (a * b * Real.sin θ) / 2 in
  let new_area := (2 * a * b * Real.sin θ) / 2 in
  new_area = 2 * original_area :=
by
  sorry

end triangle_area_doubling_side_a_l36_36661


namespace prime_factorization_count_l36_36798

theorem prime_factorization_count :
  let n := (81 * 83 * 85 * 87) + 89
  let primes_in_81 := {3}
  let primes_in_83 := {83}
  let primes_in_85 := {5, 17}
  let primes_in_87 := {3, 29}
  let primes_in_89 := {89}
  set Σ := (primes_in_81 ∪ primes_in_83 ∪ primes_in_85 ∪ primes_in_87 ∪ primes_in_89)
  in Σ.card = 6 :=
by
  sorry

end prime_factorization_count_l36_36798


namespace neg_triangle_obtuse_angle_l36_36221

theorem neg_triangle_obtuse_angle : 
  (¬ ∀ (A B C : ℝ), A + B + C = π → max (max A B) C < π/2) ↔ (∃ (A B C : ℝ), A + B + C = π ∧ min (min A B) C > π/2) :=
by
  sorry

end neg_triangle_obtuse_angle_l36_36221


namespace coordinates_of_N_l36_36351

-- Defining the conditions
def is_point_on_line (p : Point) (a b c : ℝ) : Prop :=
  a * p.x + b * p.y + c = 0

def is_perpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

structure Point where
  x : ℝ
  y : ℝ

-- Problem conditions
constant M : Point := {x := 0, y := -1}
def line1 := (1 : ℝ, -1, 1)  -- Represents the line equation x - y + 1 = 0
def line2 := (1 : ℝ, 2, -3)  -- Represents the line equation x + 2y - 3 = 0

-- Point N coordinates (to find)
constant N : Point

-- Given that M = (0, -1), N is on the line x - y + 1 = 0, and MN ⊥ x + 2y - 3 = 0
axiom h1 : is_point_on_line N line1.1 line1.2 line1.3
axiom h2 : is_perpendicular (N.x - M.x) (N.y - M.y) line2.1 line2.2

-- Prove that N = (2, 3)
theorem coordinates_of_N : N = {x := 2, y := 3} :=
sorry

end coordinates_of_N_l36_36351


namespace find_p_l36_36789

theorem find_p 
    (a b p : ℝ) 
    (ha : 0 < a) 
    (hb : 0 < b) 
    (hp : 0 < p) 
    (hyperbola_eq : ∀ x y, (x ^ 2 / a ^ 2) - (y ^ 2 / b ^ 2) = 1 ↔ true)
    (parabola_eq : ∀ x y, y ^ 2 = 2 * p * x ↔ true)
    (eccentricity_cond : b / a = sqrt 3)
    (triangle_area : ∀ x y, x = 2 / 3 * p ∨ x = -2 / 3 * p →
        y = 2 * sqrt 3 / 3 * p ∨ y = -2 * sqrt 3 / 3 * p →
        2 * (1 / 2) * (2 * sqrt 3 * p / 3) * (2 * p / 3) = sqrt 3) :
    p = 3 / 2 :=
begin
  sorry
end

end find_p_l36_36789


namespace prank_combinations_l36_36547

def monday_choices : Nat := 1
def tuesday_choices : Nat := 2
def wednesday_choices : Nat := 3
def thursday_choices : Nat := 4
def friday_choices : Nat := 2

theorem prank_combinations : 
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices = 48 :=
by
  reveal monday_choices
  reveal tuesday_choices
  reveal wednesday_choices
  reveal thursday_choices
  reveal friday_choices
  sorry

end prank_combinations_l36_36547


namespace value_of_quad_expression_l36_36403

theorem value_of_quad_expression (a b c : ℝ) (h_a : a > 0) 
  (h_ineq1 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → 0 ≤ a * x^2 + b * x + c) 
  (h_ineq2 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → a * x^2 + b * x + c ≤ 1) :
  4 * a + 5 * b + c = -1 / 4 ∨ 4 * a + 5 * b + c = real.sqrt 3 / 2 :=
sorry

end value_of_quad_expression_l36_36403


namespace jogger_speed_ratio_l36_36950

theorem jogger_speed_ratio (d t₁ t₂ : ℝ) (v₁ v₂ : ℝ) (h₁ : d = (v₁ + v₂) * t₁) (h₂ : d = (v₁ - v₂) * t₂) (h_d : d = 12) (h_t₁ : t₁ = 2) (h_t₂ : t₂ = 6) (h_v₁_gt_v₂ : v₁ > v₂) :
  (v₁ / v₂) = 2 :=
by
  -- Applying the conditions
  have h₁_eq : 12 = (v₁ + v₂) * 2 by rw [←h_d, h₁, h_t₁]
  have h₂_eq : 12 = (v₁ - v₂) * 6 by rw [←h_d, h₂, h_t₂]
  -- Solving for v₁ and v₂
  
  sorry

end jogger_speed_ratio_l36_36950


namespace total_length_of_T_lines_l36_36125

def T : set (ℝ × ℝ) := { p | let x := p.1, y := p.2 in 
  (abs (abs x - 3) - 1) + (abs (abs y - 3) - 1) = 2 }

theorem total_length_of_T_lines : 
  ∑ p in { (2, 2), (-2, 2), (2, -2), (-2, -2) }, (4 * 2 * real.sqrt 2) = 32 * real.sqrt 2 :=
by sorry

end total_length_of_T_lines_l36_36125


namespace find_a_if_pure_imaginary_l36_36350

def z1 (a : ℝ) : ℂ := a + complex.I
def z2 : ℂ := 3 - 4 * complex.I

theorem find_a_if_pure_imaginary (a : ℝ) (h : (z1 a / z2).re = 0) : a = 4 / 3 :=
by
  sorry

end find_a_if_pure_imaginary_l36_36350


namespace shara_shells_after_vacation_l36_36161

-- Definitions based on conditions
def initial_shells : ℕ := 20
def shells_per_day : ℕ := 5
def days : ℕ := 3
def shells_fourth_day : ℕ := 6

-- Statement of the proof problem
theorem shara_shells_after_vacation : 
  initial_shells + (shells_per_day * days) + shells_fourth_day = 41 := by
  sorry

end shara_shells_after_vacation_l36_36161


namespace max_selected_integers_l36_36217

theorem max_selected_integers (S : Set ℕ) (h : S = {1, 2, ..., 2015}) :
  ∃ (n : ℕ), n ≤ 672 ∧ ∀ (A : Finset ℕ), 
    (A ⊆ S) → (A.card = n) →
    (∀ (a b : ℕ) (ha : a ∈ A) (hb : b ∈ A), 
      a ≠ b → (a + b) % (a - b) ≠ 0) :=
sorry   -- proof is not required

end max_selected_integers_l36_36217


namespace probability_A_event_l36_36407

section

def event_a (left right : Type) (draws : Finset (left × right)) : Prop :=
  let incompatible (g1 g2 : left × right) := ¬(g1.1 = g2.1 ∧ g1.2 = g2.2)
  Finset.count draws incompatible = 12

-- Define pairs of gloves
inductive GloveColor | red | blue | yellow

structure Glove :=
  (color : GloveColor)
  (hand : Bool) -- true = left, false = right

def glove_set : List Glove :=
  [⟨.red, true⟩, ⟨.red, false⟩, ⟨.blue, true⟩, ⟨.blue, false⟩, ⟨.yellow, true⟩, ⟨.yellow, false⟩]

noncomputable def all_draws : Finset (Glove × Glove) :=
  {univ}.product univ

def success_draws : Finset (Glove × Glove) :=
  { (g1, g2) ∈ all_draws | g1.hand ≠ g2.hand ∧ g1.color ≠ g2.color } 

theorem probability_A_event : (success_draws.card.toRat / all_draws.card.toRat) = 1 / 3 :=
by
  sorry

end

end probability_A_event_l36_36407


namespace part1_range_of_a_part2_range_of_a_l36_36367

section
variable (a : ℝ) (f : ℝ → ℝ)

def f (a : ℝ) (x : ℝ) : ℝ :=
  (1 / (x - 1) + a) * Real.log x

theorem part1_range_of_a (a : ℝ) : (∀ x, x > 0 ∧ x ≠ 1 → f a x > 0) → 0 ≤ a ∧ a ≤ 1 :=
begin
  sorry
end

theorem part2_range_of_a (a : ℝ) : (∃ x, 1 < x ∧ f a x = (1 / (x - 1) + a) * Real.log x ∧
  ∀ y, 1 < y → (f a y)' = 0) → 0 < a ∧ a < 0.5 :=
begin
  sorry
end

end

end part1_range_of_a_part2_range_of_a_l36_36367


namespace fewest_pairs_to_guarantee_flashlight_on_l36_36966

-- Defining the conditions
def flashlight_takes_two_working_batteries : Prop := ∀ (flashlight : Type), flashlight.takes 2 (working batteries)
def total_batteries : ℕ := 12
def working_batteries : ℕ := 3
def non_working_batteries : ℕ := total_batteries - working_batteries

-- The mathematical proof problem
theorem fewest_pairs_to_guarantee_flashlight_on : 
  flashlight_takes_two_working_batteries →
  total_batteries = 12 →
  working_batteries = 3 →
  ∃ pairs : ℕ, pairs = 6 :=
by
  sorry

end fewest_pairs_to_guarantee_flashlight_on_l36_36966


namespace shaded_region_area_l36_36697

noncomputable def radius1 := 4
noncomputable def radius2 := 5
noncomputable def distance := radius1 + radius2
noncomputable def large_radius := radius2 + distance / 2

theorem shaded_region_area :
  ∃ (A : ℝ), A = (π * large_radius ^ 2) - (π * radius1 ^ 2) - (π * radius2 ^ 2) ∧
  A = 49.25 * π :=
by
  sorry

end shaded_region_area_l36_36697


namespace speed_of_man_correct_l36_36658

noncomputable def speed_of_man (train_length : ℝ) (train_speed_kmph : ℝ) (time_to_pass_sec : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let relative_speed_mps := train_length / time_to_pass_sec
  let man_speed_mps := train_speed_mps - relative_speed_mps
  man_speed_mps * (3600 / 1000)

theorem speed_of_man_correct :
  speed_of_man 200 68 11.999040076793857 ≈ 7.9988 :=
by
  sorry

end speed_of_man_correct_l36_36658


namespace sqrt_sub_eq_sqrt_two_l36_36683

theorem sqrt_sub_eq_sqrt_two : sqrt 8 - sqrt 2 = sqrt 2 := 
by sorry

end sqrt_sub_eq_sqrt_two_l36_36683


namespace MissyYellsCombined_l36_36136

def yellsAtStubbornDog (timesYellObedient : ℕ) := 4 * timesYellObedient

def totalYells (timesYellObedient : ℕ) (timesYellStubborn : ℕ) := timesYellObedient + timesYellStubborn

theorem MissyYellsCombined :
  ∀ (timesYellObedient : ℕ),
    timesYellObedient = 12 →
    totalYells timesYellObedient (yellsAtStubbornDog timesYellObedient) = 60 :=
by
  intros timesYellObedient h
  rw [h]
  unfold yellsAtStubbornDog totalYells
  norm_num
  rw [h]
  sorry

end MissyYellsCombined_l36_36136


namespace max_probability_sum_15_l36_36214

-- Context and Definitions based on conditions
def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- The assertion to be proved:
theorem max_probability_sum_15 (n : ℕ) (h : n ∈ S) :
  n = 7 :=
by
  sorry

end max_probability_sum_15_l36_36214


namespace income_proof_l36_36169

-- Define the incomes M, N, O as real numbers
variables (M N O : ℝ)

-- Define the conditions
def condition1 : Prop := (N + O) / 2 = 6250
def condition2 : Prop := (M + O) / 2 = 5200
def condition3 : Prop := M = 4000

-- Define the question (proving what is asked)
def average_income_M_and_N : Prop := (M + N) / 2 = 5050

-- The proof problem statement
theorem income_proof 
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3) :
  average_income_M_and_N :=
begin
  -- The proof will be derived here
  sorry
end

end income_proof_l36_36169


namespace inequality_with_sum_equals_one_equality_case_l36_36848

variables (x y z : ℝ)

theorem inequality_with_sum_equals_one 
  (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h_pos_z : 0 < z)
  (h_sum : x + y + z = 1) :
  (y * z + z * x + x * y) ≥ 4 * (y^2 * z^2 + z^2 * x^2 + x^2 * y^2) + 5 * x * y * z :=
begin
  sorry
end

theorem equality_case
  (h_eq : x = y) (h_eq2 : y = z) (h_sum : 3 * x = 1) :
  let equal_val = 1 / 3 in
  (x = equal_val) ∧ (y = equal_val) ∧ (z = equal_val) :=
begin
  sorry
end

end inequality_with_sum_equals_one_equality_case_l36_36848


namespace parallel_iff_parallel_sum_l36_36769

variables {ℝ : Type*} [OrderedCommRing ℝ]

-- Definition of parallel vectors
def parallel (a b : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ a = k * b

-- Problem statement
theorem parallel_iff_parallel_sum (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  parallel a b ↔ parallel a (a + b) :=
sorry

end parallel_iff_parallel_sum_l36_36769


namespace average_habitable_land_per_person_l36_36427

theorem average_habitable_land_per_person :
  let 
    population : ℕ := 281 * 10^6,
    total_land_area : ℕ := 3797000,
    habitable_ratio : ℚ := 0.8,
    square_feet_per_sq_mile : ℕ := 5280 * 5280
  in
  (total_land_area * habitable_ratio * square_feet_per_sq_mile) / population ≈ 300000 :=
by
  let population := 281 * 10^6
  let total_land_area := 3797000
  let habitable_ratio := (8 : ℚ) / 10 -- representing 0.8 as a rational number
  let square_feet_per_sq_mile := 5280 * 5280
  have h1 : ((total_land_area * habitable_ratio) * square_feet_per_sq_mile : ℚ) / population = 301300 
    from sorry
  show ((total_land_area * habitable_ratio).to_nat * square_feet_per_sq_mile) / population ≈ 300000
    from sorry

end average_habitable_land_per_person_l36_36427


namespace sum_of_primes_146_sum_of_primes_99_l36_36735

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Prove that there exist tuples of (p1, p2, p3) such that their sum is 146
theorem sum_of_primes_146 :
  ∃ (p1 p2 p3 : ℕ), is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ p1 + p2 + p3 = 146 ∧ p1 ≤ p2 ∧ p2 ≤ p3 :=
sorry

-- Prove that there exist tuples of (p1, p2, p3) such that their sum is 99
theorem sum_of_primes_99 :
  ∃ (p1 p2 p3 : ℕ), is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ p1 + p2 + p3 = 99 ∧ p1 ≤ p2 ∧ p2 ≤ p3 :=
sorry

end sum_of_primes_146_sum_of_primes_99_l36_36735


namespace probability_divisor_of_12_l36_36628

/-- Definition of divisors of a number -/
def divisors (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (λ d => n % d = 0)

theorem probability_divisor_of_12 :
  let divisors_of_12 := divisors 12
  Finset.card divisors_of_12 = 6 →
  (∃ (P : ℚ), P = 1/2) :=
by
  intro h
  exists 1/2
  sorry

end probability_divisor_of_12_l36_36628


namespace harmonic_sum_property_l36_36709

-- Define the sequence and conditions
def harmonic_mean (n : ℕ) (seq : ℕ → ℝ) : ℝ :=
  n / (finset.sum (finset.range n) (λ i, seq i))

def sequence (n : ℕ) : ℝ := (2:ℝ) * n - 1

-- The main theorem to prove
theorem harmonic_sum_property :
  (harmonic_mean 10 sequence = 1 / 10) →
  finset.sum (finset.range 10) (λ i, 1 / (sequence i * sequence (i + 1))) = 10 / 21 :=
by sorry

end harmonic_sum_property_l36_36709


namespace value_of_expression_l36_36496

variable (x : ℝ)

noncomputable def roots : set ℝ :=
{x | (x - 2) ^ 2 = 3 * (x + 5)}

noncomputable def x1 : ℝ := Classical.choose (exists_mem_of_nonempty roots)
noncomputable def x2 : ℝ := Classical.choose (exists_mem_of_nonempty (exists_diff roots (Set.singleton x1)))

theorem value_of_expression (h1 : x1 ∈ roots) (h2 : x2 ∈ roots) : 
  (x1 * x2 + x1^2 + x2^2 = 60) := by
  sorry

end value_of_expression_l36_36496


namespace number_of_elements_in_set_l36_36053

theorem number_of_elements_in_set (A : set (ℤ × ℤ))
  (h : A = {(2,-2), (2,2)}) :
  set.card A = 2 :=
by sorry

end number_of_elements_in_set_l36_36053


namespace probability_divisor_of_12_is_half_l36_36619

noncomputable def divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ d, n % d = 0) (Finset.range (n + 1))

def probability_divisor_of_12 : ℚ :=
  let total_outcomes := 12
  let favorable_outcomes := (divisors 12).card
  favorable_outcomes / total_outcomes

theorem probability_divisor_of_12_is_half :
  probability_divisor_of_12 = 1 / 2 :=
  by
    sorry

end probability_divisor_of_12_is_half_l36_36619


namespace tigers_in_zoo_l36_36413

-- Given definitions
def ratio_lions_tigers := 3 / 4
def number_of_lions := 21
def number_of_tigers := 28

-- Problem statement
theorem tigers_in_zoo : (number_of_lions : ℚ) / 3 * 4 = number_of_tigers := by
  sorry

end tigers_in_zoo_l36_36413


namespace mark_money_left_l36_36135

-- Define the initial amount of money given by Mark's father
def initial_money : ℕ := 85

-- Define the number of books Mark bought
def num_books : ℕ := 10

-- Define the cost per book
def cost_per_book : ℕ := 5

-- Define the total cost of the books Mark bought
def total_spent : ℕ := num_books * cost_per_book

-- Define the money left after purchasing the books
def money_left : ℕ := initial_money - total_spent

-- State the theorem to be proved
theorem mark_money_left : money_left = 35 := by
sory

end mark_money_left_l36_36135


namespace infinitely_many_good_pairs_l36_36485

def good (n : ℕ) : Prop := ∀ p, p.prime → p ∣ n → p ^ 2 ∣ n

def sequence (a₁ : ℕ) : ℕ → ℕ
| 0     := a₁
| (n+1) := 4 * sequence n * (sequence n + 1)

theorem infinitely_many_good_pairs :
  ∀ n, good (sequence 8 n) ∧ good (sequence 8 n + 1) :=
  sorry

end infinitely_many_good_pairs_l36_36485


namespace probability_of_divisor_of_12_is_one_half_l36_36634

theorem probability_of_divisor_of_12_is_one_half :
  let divisors := {1, 2, 3, 4, 6, 12} in
  let total_outcomes := finset.range 12 in
  (divisors.card : ℚ) / (total_outcomes.card : ℚ) = 1 / 2 :=
by
  let divisors := finset.from_list [1, 2, 3, 4, 6, 12]
  let total_outcomes := finset.range 12
  have h1 : divisors.card = 6 := by sorry
  have h2 : total_outcomes.card = 12 := by sorry
  rw [←h1, ←h2]
  norm_num
  sorry

end probability_of_divisor_of_12_is_one_half_l36_36634


namespace smallest_positive_c_l36_36320

theorem smallest_positive_c (c : ℝ) (h₀ : 0 < c) : 
  (∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → sqrt(x^2 + y^2) + c * |x - y| ≥ (x + y) / 2) ↔ c ≥ 1/2 :=
sorry

end smallest_positive_c_l36_36320


namespace grade_prediction_l36_36409

noncomputable def influence_score (T : ℝ) (E : ℝ) : ℝ :=
  0.70 * T + 0.30 * E

theorem grade_prediction :
  let S1 := influence_score 5 70,
      G1 := 84,
      k := G1 / S1,
      S2 := influence_score 6 80,
      G2 := k * S2
  in G2 = 96.49 :=
by
  sorry

end grade_prediction_l36_36409


namespace solve_equation_l36_36714

-- Define the operation
def diamondsuit (a b : ℝ) : ℝ := a / b

-- State the main theorem to prove the solution
theorem solve_equation :
  ∃ x : ℝ, diamondsuit 504 (diamondsuit 7 x) = 60 → x = 5 / 6 := by
  sorry

end solve_equation_l36_36714


namespace probability_divisor_of_12_is_half_l36_36617

noncomputable def divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ d, n % d = 0) (Finset.range (n + 1))

def probability_divisor_of_12 : ℚ :=
  let total_outcomes := 12
  let favorable_outcomes := (divisors 12).card
  favorable_outcomes / total_outcomes

theorem probability_divisor_of_12_is_half :
  probability_divisor_of_12 = 1 / 2 :=
  by
    sorry

end probability_divisor_of_12_is_half_l36_36617


namespace cell_diameter_scientific_notation_l36_36173

-- Define the number in question
def cell_diameter : ℝ := 0.000123

-- Define the expected scientific notation form
def expected_value : ℝ := 1.23 * 10 ^ (-4)

-- Theorem stating that cell diameter in scientific notation is as expected
theorem cell_diameter_scientific_notation :
  cell_diameter = expected_value :=
by
  sorry

end cell_diameter_scientific_notation_l36_36173


namespace angle_bisector_length_l36_36813

theorem angle_bisector_length (A B C D : Point) (h : Triangle A B C) (hD : AngleBisector C A B C D) :
  length (segment C D) < real.sqrt (length (segment C A) * length (segment C B)) :=
sorry

end angle_bisector_length_l36_36813


namespace part_I_part_II_l36_36783

-- Part I
theorem part_I (f g : ℝ → ℝ) (a : ℝ) :
  (∀ x, x ≥ 0 → f x ≥ g x) ↔ a ∈ Iic 0 :=
  sorry

-- Definitions for f and g
def f (x : ℝ) (a : ℝ) : ℝ := (1/3) * x^3 - (1/2) * a * x^2 - sin x
def g (x : ℝ) (a : ℝ) : ℝ := (a - x) * cos x

-- Part II
theorem part_II (a : ℕ → ℝ) :
  (a 0 = sqrt 2 / 2 ∧ ∀ n, a (n + 1) = sqrt 2 / 2 * sqrt (1 - sqrt (1 - (a n)^2)))
  → ∀ n, a n < (pi / 2^(n + 2)) :=
  sorry

-- Definitions for sequence a
def sequence_a (n : ℕ) : ℝ :=
  if n = 0 then sqrt 2 / 2
  else sqrt 2 / 2 * sqrt (1 - sqrt (1 - (sequence_a (n - 1))^2))

end part_I_part_II_l36_36783


namespace find_n_l36_36450

noncomputable def C_n (n : ℕ) : ℝ :=
  (2048 / 3) * (1 - (1 / (4^n)))

noncomputable def D_n (n : ℕ) : ℝ :=
  (4096 / 3) * (1 - (1 / (-3)^n))

theorem find_n : ∃ (n : ℕ), n ≥ 1 ∧ C_n n = D_n n ∧ n = 1 :=
by
  use 1
  split
  · exact Nat.one_le_iff_ne_zero.mpr rfl
  split
  · -- show that C_n 1 = D_n 1
    have hCn_1 : C_n 1 = (2048 / 3) * (1 - (1 / 4^1)) := by rfl
    have hDn_1 : D_n 1 = (4096 / 3) * (1 - (1 / (-3)^1)) := by rfl
    -- calculation proving the values are equal
    calc
      C_n 1
      _ = (2048 / 3) * (1 - (1 / 4)) := by rw hCn_1
      _ = (2048 / 3) * (3/4) := by norm_num
      _ = (2048 * 3/4) / 3 := by ring
      _ = 2048/4 := by ring
      _ = 512 := by norm_num
      
      D_n 1
      _ = (4096 / 3) * (1 - (1 / -3)) := by rw hDn_1
      _ = (4096 / 3) * (1 + 1/3) := by norm_num
      _ = (4096 * 4/3) / 3 := by ring
      _ = 4096/3 := by ring
      _ = 2048/3 := by norm_num

      C_n 1 = D_n 1 := by sorry
  
  -- Since Lean needs the numerical proof, inserting a sorry here
  -- to indicate the skipped step.
  sorry

end find_n_l36_36450


namespace round_robin_highest_score_l36_36992

theorem round_robin_highest_score
  (n : ℕ) (hn : n = 16)
  (teams : Fin n → ℕ)
  (games_played : Fin n → Fin n → ℕ)
  (draws : Fin n → Fin n → ℕ)
  (win_points : ℕ := 2)
  (draw_points : ℕ := 1)
  (total_games : ℕ := (n * (n - 1)) / 2) :
  ¬ (∃ max_score : ℕ, ∀ i : Fin n, teams i ≤ max_score ∧ max_score < 16) :=
by sorry

end round_robin_highest_score_l36_36992


namespace temperature_lower_than_minus_three_l36_36564

theorem temperature_lower_than_minus_three (a b : ℤ) (hx : a = -3) (hy : b = -6) : a + b = -9 :=
by
  sorry

end temperature_lower_than_minus_three_l36_36564


namespace construction_costs_correct_l36_36259

structure ConstructionCosts where
  landCostPerSqMeter : ℕ
  brickCostPerThousand : ℕ
  tileCostPerTile : ℕ
  landRequired : ℕ
  bricksRequired : ℕ
  tilesRequired : ℕ

noncomputable def totalConstructionCost (cc : ConstructionCosts) : ℕ :=
  let landCost := cc.landRequired * cc.landCostPerSqMeter
  let brickCost := (cc.bricksRequired / 1000) * cc.brickCostPerThousand
  let tileCost := cc.tilesRequired * cc.tileCostPerTile
  landCost + brickCost + tileCost

theorem construction_costs_correct (cc : ConstructionCosts)
  (h1 : cc.landCostPerSqMeter = 50)
  (h2 : cc.brickCostPerThousand = 100)
  (h3 : cc.tileCostPerTile = 10)
  (h4 : cc.landRequired = 2000)
  (h5 : cc.bricksRequired = 10000)
  (h6 : cc.tilesRequired = 500) :
  totalConstructionCost cc = 106000 := 
  by 
    sorry

end construction_costs_correct_l36_36259


namespace largest_number_proof_l36_36312

/-- 
The largest natural number that does not end in zero and decreases by an integer factor 
when one (not the first) digit is removed.
-/
def largest_number_decreasing_by_factor : ℕ := 
  let x := 8
  let a := 1
  let c := 625
  let n := 1
  let r := 5
  let number := 10^(n+1) * a + 10^n * x + c
  { number | number ∉ [0], number % 10 ≠ 0, (r=5), 2 ≤ r ≤ 19 }

theorem largest_number_proof :
  largest_number_decreasing_by_factor = 180625 :=
sorry

end largest_number_proof_l36_36312


namespace cross_area_l36_36104

variables (R : ℝ) (A : ℝ × ℝ) (φ : ℝ)
  -- Radius R of the circle, Point A inside the circle, and angle φ in radians

-- Define the area of the cross formed by rotated lines
def area_of_cross (R : ℝ) (φ : ℝ) : ℝ :=
  2 * φ * R^2

theorem cross_area (R : ℝ) (A : ℝ × ℝ) (φ : ℝ) (hR : 0 < R) (hA : dist A (0, 0) < R) :
  area_of_cross R φ = 2 * φ * R^2 := 
sorry

end cross_area_l36_36104


namespace exterior_angle_regular_nonagon_l36_36424

theorem exterior_angle_regular_nonagon : 
  ∀ (n : ℕ), n = 9 → 
  (∃ θ : ℝ, θ = 40) := by
  intros n h
  have H : ∀ k : ℕ, k - 2 = k - 2 := λ k, rfl
  have interior_sum : ∀ k : ℕ, 180 * (k - 2) = 1260 := 
    by simp [mul_sub, mul_comm]
  have each_interior : ∀ k : ℕ, (180 * (k - 2)) / k = 140 := 
    by simp [div_eq_mul_inv, mul_div_assoc, interior_sum]
  have θ_eq : 180 - 140 = 40 := 
    by simp [sub_eq_add_neg, add_comm]
  exact ⟨40, θ_eq⟩

end exterior_angle_regular_nonagon_l36_36424


namespace sin_cos_of_theta_l36_36362

open Real

theorem sin_cos_of_theta (θ : ℝ) (P : ℝ × ℝ) (hx : P = (-3, 4))
  (hxθ : ∃ r, r > 0 ∧ P = (r * cos θ, r * sin θ)) :
  sin θ + cos θ = 1 / 5 := 
by
  sorry

end sin_cos_of_theta_l36_36362


namespace horner_v3_value_correct_l36_36338

def f (x : ℕ) : ℕ :=
  x^5 + 2*x^3 + 3*x^2 + x + 1

def horner_eval (x : ℕ) : ℕ :=
  ((((x + 0) * x + 2) * x + 3) * x + 1) * x + 1

theorem horner_v3_value_correct :
  horner_eval 3 = 36 :=
sorry

end horner_v3_value_correct_l36_36338


namespace color_hypergraph_l36_36244

-- Define a hypergraph
structure Hypergraph (V : Type) :=
  (edges : set (set V)) 

-- Define the condition: No two edges have exactly one vertex in common
def no_pair_edges_with_single_common_vertex {V : Type} (G : Hypergraph V) : Prop :=
  ∀ e1 e2 ∈ G.edges, e1 ≠ e2 → (e1 ∩ e2).card ≠ 1

-- Define the theorem to prove the coloring condition
theorem color_hypergraph {V : Type} (G : Hypergraph V) 
  (h : no_pair_edges_with_single_common_vertex G) :
  ∃ (color : V → bool), ∀ e ∈ G.edges, ∃ v1 v2 ∈ e, color v1 ≠ color v2 :=
sorry

end color_hypergraph_l36_36244


namespace jade_accumulation_rate_ratio_l36_36418

theorem jade_accumulation_rate_ratio (a : ℝ) (π_ne_zero : π ≠ 0) : 
  let k1 := (π / 6) 
  let k2 := (π / 4) 
  let k3 := 1
  in (k1, k2, k3) = (π / 6, π / 4, 1) := 
by 
  let V1 := (π / 6) * a^3
  let V2 := (π / 4) * a^3
  let V3 := a^3
  have k1_eq : V1 = (π / 6) * a^3 := by sorry
  have k2_eq : V2 = (π / 4) * a^3 := by sorry
  have k3_eq : V3 = 1 * a^3 := by sorry
  have ratio_eq := (k1_eq, k2_eq, k3_eq)
  sorry

end jade_accumulation_rate_ratio_l36_36418


namespace subtraction_solution_l36_36982

noncomputable def x : ℝ := 47.806

theorem subtraction_solution :
  (3889 : ℝ) + 12.808 - x = 3854.002 :=
by
  sorry

end subtraction_solution_l36_36982


namespace find_difference_of_logs_l36_36775

open Real

theorem find_difference_of_logs
  (a b c d : ℕ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hd : d > 0)
  (h1 : log a b = 3 / 2)
  (h2 : log c d = 5 / 4)
  (h3 : a - c = 9) :
  b - d = 93 :=
by
  sorry

end find_difference_of_logs_l36_36775


namespace ivan_max_score_l36_36679

-- Let's define contestants and problems in general.
universe u
variable {Contestant : Type u} [DecidableEq Contestant]

-- Considering 8 problems.
constant probs : Fin 8 → Type u

-- And 30 contestants.
constant Contestants : Fin 30 → Contestant

-- A function determining the number of solvers for each problem.
constant solvers : (Fin 8) → Fin 30 → Prop 

-- Points given for each problem based on the number of solvers.
def points (p : Fin 8) : ℕ :=
  30 - (Fintype.card {c // solvers p c})

--- Ivan's score.
def ivan_score : ℕ := ∑ p, if solvers p (Contestants 0) then points p else 0

--- Function stating every other contestant's score.
def other_score (i : Fin 29) : ℕ := ∑ p, if solvers p (Contestants (i + 1)) then points p else 0

/-- Ivan's highest score such that he has fewer points than any other contestant -/
theorem ivan_max_score {Ivan : Contestant} :
  ∃ m : ℕ, m = 58 ∧ (∀ i : Fin 29, ivan_score < other_score i) := 
sorry

end ivan_max_score_l36_36679


namespace new_mean_when_adding_const_to_each_number_l36_36840

theorem new_mean_when_adding_const_to_each_number :
  ∀ (numbers : Fin 15 → ℝ) (m : ℝ),
    (m = (∑ i, numbers i) / 15) →
    m = 40 →
    (∑ i, (numbers i + 10)) / 15 = 50 :=
by
  intros numbers m hm hmean
  sorry

end new_mean_when_adding_const_to_each_number_l36_36840


namespace find_c_l36_36363

theorem find_c (c : ℝ)
  (h1 : ∃ y : ℝ, y = (-2)^2 - (-2) + c)
  (h2 : ∃ m : ℝ, m = 2 * (-2) - 1)
  (h3 : ∃ x y, y - (4 + c) = -5 * (x + 2) ∧ x = 0 ∧ y = 0) :
  c = 4 :=
sorry

end find_c_l36_36363


namespace exists_prime_divisor_in_sequence_l36_36447

theorem exists_prime_divisor_in_sequence
  (c d : ℕ) (hc : 2 ≤ c) (hd : 2 ≤ d)
  (a : ℕ → ℕ)
  (h0 : a 1 = c)
  (hs : ∀ n, a (n+1) = a n ^ d + c) :
  ∀ (n : ℕ), 2 ≤ n →
  ∃ (p : ℕ), Prime p ∧ p ∣ a n ∧ ∀ i, 1 ≤ i ∧ i < n → ¬ p ∣ a i := sorry

end exists_prime_divisor_in_sequence_l36_36447


namespace sum_of_x_coordinates_l36_36707

def f (x : ℝ) : ℝ :=
  if x < -2 then 2 * x + 3
  else if x < -1 then -2 * x - 3
  else if x < 1 then -2 * x + 1
  else if x < 2 then 2 * x - 1
  else 2 * x - 3

theorem sum_of_x_coordinates :
  set.sum {x : ℝ | f x = 2.5} = 3.75 :=
sorry

end sum_of_x_coordinates_l36_36707


namespace find_b_l36_36832

noncomputable def hyperbola_focus (b : ℝ) := 
  (real.sqrt (1 + b^2), 0)

noncomputable def hyperbola_asymptote_distance (b : ℝ) : ℝ :=
  let c := real.sqrt(1 + b^2) in c / real.sqrt(1 + 1 / b^2)

noncomputable def parabola_p_focus_distance : ℝ :=
  2

theorem find_b (b : ℝ) (h : 0 < b) 
  (dist_eq : hyperbola_asymptote_distance b = parabola_p_focus_distance) : b = 2 :=
by
  sorry

end find_b_l36_36832


namespace amaya_movie_time_l36_36668

theorem amaya_movie_time :
  ∃ x : ℕ, 
  let rewind1_time := 5
      between_rewinds_time := 45
      rewind2_time := 15
      after_rewind2_time := 20
      total_time := 120 in
  x + rewind1_time + between_rewinds_time + rewind2_time + after_rewind2_time = total_time ∧ x = 35 :=
by
  sorry

end amaya_movie_time_l36_36668


namespace condition_sufficient_but_not_necessary_l36_36793

variable (A : Set ℝ) (B : Set ℝ)
variable (a : ℝ)

def set_A := {x : ℝ | x > 5}
def set_B := {x : ℝ | x > a}

theorem condition_sufficient_but_not_necessary
  (H : set_A ⊆ set_B) : 
  a < 5 := 
  sorry

end condition_sufficient_but_not_necessary_l36_36793


namespace total_students_exam_l36_36170

theorem total_students_exam (N T T' T'' : ℕ) (h1 : T = 88 * N) (h2 : T' = T - 8 * 50) 
  (h3 : T' = 92 * (N - 8)) (h4 : T'' = T' - 100) (h5 : T'' = 92 * (N - 9)) : N = 84 :=
by
  sorry

end total_students_exam_l36_36170


namespace range_of_a_l36_36533

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3 * x

-- Define the condition for having three distinct intersection points.
def has_three_distinct_intersections (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ = a ∧ f x₂ = a ∧ f x₃ = a

-- Prove that the range of values for a is (-2, 2)
theorem range_of_a : ∀ (a : ℝ), has_three_distinct_intersections a ↔ a ∈ Ioo (-2 : ℝ) 2 :=
by
  sorry

end range_of_a_l36_36533


namespace cubic_root_relationship_l36_36465

theorem cubic_root_relationship 
  (r : ℝ) (h : r^3 - r + 3 = 0) : 
  (r^2)^3 - 2 * (r^2)^2 + (r^2) - 9 = 0 := 
by 
  sorry

end cubic_root_relationship_l36_36465


namespace count_circular_cross_section_l36_36672

def has_circular_cross_section (body : String) : Prop :=
  body = "Cylinder" ∨ body = "Sphere" ∨ body = "Cone"

def geometric_bodies : List String :=
  ["Cylinder", "Cube", "Prism", "Sphere", "Cone", "Cuboid"]

theorem count_circular_cross_section : (geometric_bodies.filter has_circular_cross_section).length = 3 :=
by
  sorry

end count_circular_cross_section_l36_36672


namespace find_length_of_BC_l36_36430

noncomputable def length_of_BC {α : Type*} [linear_ordered_field α] 
  {A B C : α} (AB AC : α) (cosB : α) (sinC : α) (h_cosB : cosB = 2 * sinC) 
  (h_AB : AB = 1) (h_AC : AC = 2) : α :=
  Real.sqrt (5 - (16 / Real.sqrt 17))

theorem find_length_of_BC {α : Type*} [linear_ordered_field α] 
  (sin_C : α) (h : Real.sqrt 17 * sin_C = 2) :
  length_of_BC 1 2 (2 * sin_C) sin_C (by rw [←h] ; norm_num) (by norm_num) (by norm_num) =
    Real.sqrt 2 + Real.sqrt 14 / 2 := 
sorry

end find_length_of_BC_l36_36430


namespace car_mpg_in_city_l36_36247

theorem car_mpg_in_city 
    (miles_per_tank_highway : Real)
    (miles_per_tank_city : Real)
    (mpg_difference : Real)
    : True := by
  let H := 21.05
  let T := 720 / H
  let C := H - 10
  have h1 : 720 = H * T := by
    sorry
  have h2 : 378 = C * T := by
    sorry
  exact True.intro

end car_mpg_in_city_l36_36247


namespace identify_functions_l36_36671

-- Define the first expression
def expr1 (x : ℝ) : ℝ := x - (x - 3)

-- Define the second expression
noncomputable def expr2 (x : ℝ) : ℝ := Real.sqrt (x - 2) + Real.sqrt (1 - x)

-- Define the third expression
noncomputable def expr3 (x : ℝ) : ℝ :=
if x < 0 then x - 1 else x + 1

-- Define the fourth expression
noncomputable def expr4 (x : ℝ) : ℝ :=
if x ∈ Set.Ioo (-1) 1 then 0 else 1

-- Proof statement
theorem identify_functions :
  (∀ x, ∃! y, expr1 x = y) ∧ (∀ x, ∃! y, expr3 x = y) ∧
  (¬ ∃ x, ∃! y, expr2 x = y) ∧ (¬ ∀ x, ∃! y, expr4 x = y) := by
    sorry

end identify_functions_l36_36671


namespace problem_1_and_2_l36_36370

open Real

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 - 2 * a * x + b

theorem problem_1_and_2 (a b k : ℝ) (h_a_pos : a > 0)
  (h1 : ∀ x ∈ Icc (0 : ℝ) 3, f a b x ≤ 3)
  (h2 : ∀ x ∈ Icc (0 : ℝ) 3, f a b x ≥ -1)
  (h3 : ∀ x ∈ Ico -1 0, g (f a b) (3^x) - k * 3^x ≥ 0) :
  a = 1 ∧ b = 0 ∧ k ≤ -5 :=
sorry

end problem_1_and_2_l36_36370


namespace problem_1_problem_2_l36_36365

def f (a : ℝ) (x : ℝ) : ℝ := abs (a * x + 1)

def g (a : ℝ) (x : ℝ) : ℝ := f a x - abs (x + 1)

theorem problem_1 (a : ℝ) : (∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 ↔ f a x ≤ 3) → a = 2 := by
  intro h
  sorry

theorem problem_2 (a : ℝ) : a = 2 → (∃ x : ℝ, ∀ y : ℝ, g a y ≥ g a x ∧ g a x = -1/2) := by
  intro ha2
  use -1/2
  sorry

end problem_1_problem_2_l36_36365


namespace sin_squared_sum_l36_36541

theorem sin_squared_sum :
  \[ \sin^2 (1 : ℝ) + \sin^2 2 + \ldots + \sin^2 88 + \sin^2 89 = 44 \frac{1}{2} \] :=
by
  -- Sum of squares of sines up to 89 degrees
  sorry

end sin_squared_sum_l36_36541


namespace problem_l36_36123

noncomputable def phi (k : ℕ) : ℕ :=
  if k = 0 then 0
  else (finset.range k).count (λ i, nat.gcd (i + 1) k = 1)

theorem problem (n : ℕ) (a : ℕ → ℕ)
  (h1 : 0 < n)
  (h2 : a 1 = 2)
  (h3 : ∀ m, 1 ≤ m ∧ m ≤ n - 1 → a m = phi (a (m + 1)))
  : a n ≥ 2^(n-1) :=
sorry

end problem_l36_36123


namespace mean_of_three_digit_multiples_of_8_l36_36956

theorem mean_of_three_digit_multiples_of_8 :
  let a1 := 104
  let an := 992
  let d := 8
  let n := (an - a1) / d + 1
  let sum := n * (a1 + an) / 2
  let mean := sum / n
  mean = 548 :=
by
  let a1 := 104
  let an := 992
  let d := 8
  let n := (an - a1) / d + 1
  let sum := n * (a1 + an) / 2
  let mean := sum / n
  sorry

end mean_of_three_digit_multiples_of_8_l36_36956


namespace solution_set_of_inequality_l36_36940

theorem solution_set_of_inequality :
  {x : ℝ | 3 * x - x^2 ≥ 0} = set.Icc 0 3 :=
sorry

end solution_set_of_inequality_l36_36940


namespace bianca_bags_not_recycled_l36_36681

theorem bianca_bags_not_recycled :
  ∀ (points_per_bag total_bags total_points bags_recycled bags_not_recycled : ℕ),
    points_per_bag = 5 →
    total_bags = 17 →
    total_points = 45 →
    bags_recycled = total_points / points_per_bag →
    bags_not_recycled = total_bags - bags_recycled →
    bags_not_recycled = 8 :=
by
  intros points_per_bag total_bags total_points bags_recycled bags_not_recycled
  intros h_points_per_bag h_total_bags h_total_points h_bags_recycled h_bags_not_recycled
  sorry

end bianca_bags_not_recycled_l36_36681


namespace shaded_region_area_l36_36702

theorem shaded_region_area {radius1 radius2 : ℝ} (h1 : radius1 = 4) (h2 : radius2 = 5) :
  let dist_centers := radius1 + radius2,
      circumscribed_radius := dist_centers,
      larger_area := Real.pi * circumscribed_radius ^ 2,
      smaller_area1 := Real.pi * radius1 ^ 2,
      smaller_area2 := Real.pi * radius2 ^ 2,
      shaded_area := larger_area - smaller_area1 - smaller_area2
  in shaded_area = 40 * Real.pi :=
by
  simp [h1, h2]
  sorry

end shaded_region_area_l36_36702


namespace not_perfect_square_l36_36017

theorem not_perfect_square (a b : ℤ) (h : (a % 2 ≠ b % 2)) : ¬ ∃ k : ℤ, ((a + 3 * b) * (5 * a + 7 * b) = k^2) := 
by
  sorry

end not_perfect_square_l36_36017


namespace number_of_parents_at_park_l36_36203

theorem number_of_parents_at_park (girls boys playgroups people_per_group : ℕ) 
  (h1 : girls = 14) 
  (h2 : boys = 11) 
  (h3 : playgroups = 3) 
  (h4 : people_per_group = 25) : 
  (75 - (girls + boys) = 50) := 
by
  rw [h1, h2, h3, h4]
  show 75 - (14 + 11) = 50
  sorry

end number_of_parents_at_park_l36_36203


namespace set_intersection_complement_eq_l36_36472

noncomputable def U := {1, 2, 3, 4, 5, 6, 7}
noncomputable def P := {1, 2, 3, 4, 5}
noncomputable def Q := {3, 4, 5, 6, 7}

theorem set_intersection_complement_eq :
  P \cap (U - Q) = {1, 2} :=
by sorry

end set_intersection_complement_eq_l36_36472


namespace number_of_ordered_pairs_l36_36291

theorem number_of_ordered_pairs :
  (∃ n : ℕ, n = 288 ∧ ∀ (x y : ℕ), 1 ≤ x ∧ x < y ∧ y ≤ 50 ∧ 
  (∃ k : ℤ, k = ((x + y) % 4) ∧ k = 0) ∧ 
  (∃ i : ℂ, i^x + i^y ∈ ℝ)) :=
  sorry

end number_of_ordered_pairs_l36_36291


namespace tangent_line_at_origin_common_points_count_l36_36033

noncomputable def f : ℝ → ℝ := λ x, Real.exp x

theorem tangent_line_at_origin :
  let k := Real.exp 1 in
  ∀ x: ℝ, (x, k * x) ∈ set_of (λ p : ℝ × ℝ, f p.1 = p.2) :=
sorry

theorem common_points_count (m : ℝ) (hm : 0 < m) :
  let h := λ x, Real.exp x / (x^2) in
  let h' := λ x, (Real.exp x * (x - 2)) / (x^3) in
  if m ∈ (set.Ioo 0 (Real.exp 2 / 4)) then ∃ x, x > 0 ∧ f x = m * x^2 ∧ (∀ y ≠ x, f y ≠ m * y^2) ∧ ¬∃ z, z ≠ x ∧ f z = m * z^2 else
  if m = Real.exp 2 / 4 then ∃! x, x > 0 ∧ f x = m * x^2 else
  if m > Real.exp 2 / 4 then ∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ f x₁ = m * x₁^2 ∧ f x₂ = m * x₂^2 else
  false :=
sorry

end tangent_line_at_origin_common_points_count_l36_36033


namespace intersection_of_A_and_B_l36_36376

def A := { x : ℝ | -2 ≤ x ∧ x ≤ 3 }
def B := { x : ℝ | -1 < x ∧ x < 4 }

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 3} :=
sorry

end intersection_of_A_and_B_l36_36376


namespace product_modulo_l36_36165

theorem product_modulo (n : ℕ) (h : 93 * 68 * 105 ≡ n [MOD 20]) (h_range : 0 ≤ n ∧ n < 20) : n = 0 := 
by
  sorry

end product_modulo_l36_36165


namespace constant_term_of_binomial_expansion_l36_36119

noncomputable def integral_result : ℝ :=
  ∫ x in 0..2, (1 - 2 * x)

theorem constant_term_of_binomial_expansion :
  (a : ℝ) (h : a = integral_result) -> (constant_term : ℝ) (
    ∃ r : ℕ, 12 - 3 * r = 0 ∧ constant_term =
      nat.choose 6 r * (2 : ℝ)^(r-6) * (-2)^r 
  ) = 60 :=
by
  sorry

end constant_term_of_binomial_expansion_l36_36119


namespace markov_inequality_l36_36461

variable (Ω : Type) [meas : MeasureTheory.MeasureSpace Ω]
variable (X : Ω → ℝ) [MeasureTheory.AeMeasurable X meas]

-- condition that X takes positive values almost everywhere
axiom X_pos : ∀ᵐ ω ∂meas.toMeasure, 0 < X ω

-- the theorem statement
theorem markov_inequality : (MeasureTheory.measure.measure_of {ω | X ω ≥ 13} meas.toMeasure) ≤ (1 / 13) * MeasureTheory.MeasureTheory.ae_measurable.ennreal_of_real_measure X :=
by 
  sorry

end markov_inequality_l36_36461


namespace sequence_sum_l36_36304

-- Definitions representing the given conditions
variables (A H M O X : ℕ)

-- Assuming the conditions as hypotheses
theorem sequence_sum (h₁ : A + 9 + H = 19) (h₂ : 9 + H + M = 19) (h₃ : H + M + O = 19)
  (h₄ : M + O + X = 19) : A + H + M + O = 26 :=
sorry

end sequence_sum_l36_36304


namespace fraction_strawberries_remaining_l36_36215

theorem fraction_strawberries_remaining 
  (baskets : ℕ)
  (strawberries_per_basket : ℕ)
  (hedgehogs : ℕ)
  (strawberries_per_hedgehog : ℕ)
  (h1 : baskets = 3)
  (h2 : strawberries_per_basket = 900)
  (h3 : hedgehogs = 2)
  (h4 : strawberries_per_hedgehog = 1050) :
  (baskets * strawberries_per_basket - hedgehogs * strawberries_per_hedgehog) / (baskets * strawberries_per_basket) = 2 / 9 :=
by
  sorry

end fraction_strawberries_remaining_l36_36215


namespace probability_divisor_of_12_on_12sided_die_l36_36614

theorem probability_divisor_of_12_on_12sided_die (h : ∀ i, 1 ≤ i ∧ i ≤ 12 → true) : 
  (probability (fun (n : ℕ) => n ∣ 12 ∧ 1 ≤ n ∧ n ≤ 12)) = 1/2 := 
sorry

end probability_divisor_of_12_on_12sided_die_l36_36614


namespace no_even_number_of_axes_of_symmetry_l36_36484

theorem no_even_number_of_axes_of_symmetry (S : Type) [Solid S] :
  ∀ (axes : set (line S)), finite axes → (∀ l ∈ axes, is_axis_of_symmetry l S) →
  ¬ (∃ n, n > 0 ∧ even n ∧ cardinal.mk axes = n) := 
by 
  sorry

end no_even_number_of_axes_of_symmetry_l36_36484


namespace problem_simplify_and_evaluate_l36_36908

theorem problem_simplify_and_evaluate (m : ℝ) (h : m = Real.sqrt 3 + 3) :
  (1 - (m / (m + 3))) / ((m^2 - 9) / (m^2 + 6 * m + 9)) = Real.sqrt 3 :=
by
  sorry

end problem_simplify_and_evaluate_l36_36908


namespace num_elements_with_leading_digit_8_num_elements_with_leading_digit_1_l36_36460

open Int

noncomputable def set_8_powers := {k : ℕ | k ≤ 3000}

def digits_count (base : ℤ) (exp : ℕ) : ℕ :=
  String.length (base^exp).toString

theorem num_elements_with_leading_digit_8 :
  (∀ k ∈ set_8_powers, digits_count 8 k ≤ 2713)
  ∧ (digits_count 8 3000 = 2713)
  ∧ String.front (Int.toString (8^3000)) = '8' 
  → ∃ n : ℕ, n = 288 :=
sorry

theorem num_elements_with_leading_digit_1 :
  (∀ k ∈ set_8_powers, digits_count 8 k ≤ 1357)
  ∧ (digits_count 8 1500 = 1357)
  ∧ String.front (Int.toString (8^1500)) = '1'
  → ∃ n : ℕ, n = 1357 :=
sorry

end num_elements_with_leading_digit_8_num_elements_with_leading_digit_1_l36_36460


namespace range_of_m_inequality_l36_36758

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  m * real.exp(2 * x) - 2 * x * (x + 1) * real.exp(x)

theorem range_of_m (x1 x2 : ℝ) : 
  (∃ x1 x2, x1 ≠ x2 ∧ deriv (f m) x1 = 0 ∧ deriv (f m) x2 = 0) ->
  -real.exp(2) < m ∧ m ≤ 0 := 
sorry

theorem inequality (m : ℝ) (x1 x2 : ℝ) :
  deriv (f m) x1 = 0 ∧ deriv (f m) x2 = 0 ∧ x1 ≠ x2 →
  3 < x1 * x2 - (x1 + x2) ∧ x1 * x2 - (x1 + x2) < 8 :=
sorry

end range_of_m_inequality_l36_36758


namespace center_of_symmetry_is_neg2_3_l36_36179

theorem center_of_symmetry_is_neg2_3 :
  ∃ (a b : ℝ), 
  (a,b) = (-2, 3) ∧ 
  ∀ x : ℝ, 
    2 * b = ((a + x + 2)^3 - (a + x) + 1) + ((a - x + 2)^3 - (a - x) + 1) := 
by
  use -2, 3
  sorry

end center_of_symmetry_is_neg2_3_l36_36179


namespace train_length_approx_l36_36660

noncomputable def length_of_train (speed_km_hr : ℝ) (time_seconds : ℝ) : ℝ :=
  let speed_m_s := (speed_km_hr * 1000) / 3600
  speed_m_s * time_seconds

theorem train_length_approx (speed_km_hr time_seconds : ℝ) (h_speed : speed_km_hr = 120) (h_time : time_seconds = 4) :
  length_of_train speed_km_hr time_seconds = 133.32 :=
by
  sorry

end train_length_approx_l36_36660


namespace largest_y_coordinate_of_degenerate_ellipse_l36_36293

theorem largest_y_coordinate_of_degenerate_ellipse :
  ∀ x y : ℝ, (x^2 / 49 + (y - 3)^2 / 25 = 0) → y ≤ 3 := by
  sorry

end largest_y_coordinate_of_degenerate_ellipse_l36_36293


namespace sequence_contains_1_or_4_l36_36265

def sum_of_squares_of_digits (n : Nat) : Nat :=
  (n.digits 10).map (fun d => d * d).sum

theorem sequence_contains_1_or_4 (a1 : Nat) (h : 100 ≤ a1 ∧ a1 ≤ 999) :
  ∃ n, (nat.iterate sum_of_squares_of_digits n a1 = 1) ∨ (nat.iterate sum_of_squares_of_digits n a1 = 4) :=
by
  sorry

end sequence_contains_1_or_4_l36_36265


namespace cube_vessel_base_length_l36_36594

-- Define basic parameters used in the problem
def edge_length_cube : ℝ := 15
def base_width_vessel : ℝ := 14
def rise_water_level : ℝ := 12.053571428571429
def volume_cube : ℝ := edge_length_cube ^ 3
def volume_displaced : ℝ := volume_cube
def base_area_vessel : ℝ := volume_displaced / rise_water_level

-- Proof statement for length of the base of the vessel
theorem cube_vessel_base_length : 
  (volume_cube = edge_length_cube ^ 3) ∧ 
  (volume_displaced = volume_cube) ∧
  (base_area_vessel = volume_displaced / rise_water_level) ∧
  (base_area_vessel = 280) →
  ∃ L : ℝ, L = base_area_vessel / base_width_vessel ∧ L = 20 :=
by
  intro h
  use 20
  split
  · sorry  -- Proof part
  · sorry  -- Proof part

end cube_vessel_base_length_l36_36594


namespace other_position_in_arithmetic_progression_l36_36199

theorem other_position_in_arithmetic_progression 
  (a d : ℝ) (x : ℕ)
  (h1 : a + (4 - 1) * d + a + (x - 1) * d = 20)
  (h2 : 5 * (2 * a + 9 * d) = 100) :
  x = 7 := by
  sorry

end other_position_in_arithmetic_progression_l36_36199


namespace rotated_square_vertical_distance_is_correct_l36_36752

-- Define a setup with four 1-inch squares in a straight line
-- and the second square rotated 45 degrees around its center

-- Noncomputable setup
noncomputable def rotated_square_vert_distance : ℝ :=
  let side_length := 1
  let diagonal := side_length * Real.sqrt 2
  -- Calculate the required vertical distance according to given conditions
  Real.sqrt 2 + side_length / 2

-- Theorem statement confirming the calculated vertical distance
theorem rotated_square_vertical_distance_is_correct :
  rotated_square_vert_distance = Real.sqrt 2 + 1 / 2 :=
by
  sorry

end rotated_square_vertical_distance_is_correct_l36_36752


namespace coeff_x3_in_expansion_l36_36078

theorem coeff_x3_in_expansion : (nat.choose 50 3 * (1^47) * (1^3)) = 19600 := 
by sorry

end coeff_x3_in_expansion_l36_36078


namespace sum_of_possible_M_l36_36188

theorem sum_of_possible_M : 
  ∀ (x y z M : ℕ), 
    M = x * y * z ∧ 
    M = 8 * (x + y + z) ∧ 
    z = 2 * (x + y) ∧ 
    x > 0 ∧ 
    y > 0 ∧ 
    z > 0 → 
    (M = 272 ∨ M = 160 ∨ M = 128) → 
    ∑ possible_M, possible_M = 560 := 
by 
  intros x y z M h₁ h₂ h₃ h₄ h₅ h₆ hM 
  sorry

end sum_of_possible_M_l36_36188


namespace calculate_cost_l36_36925

def ceil (m : ℝ) : ℕ :=
  if m ≤ ⌊m⌋ then ⌊m⌋.toNat else ⌊m⌋.toNat + 1

def cost_of_call (m : ℝ) (hm : m > 0) : ℝ :=
  1.06 * (0.5 * (ceil m) + 1)

theorem calculate_cost :
  cost_of_call 5.5 (by norm_num) = 4.24 :=
by
  sorry -- Proof steps to be filled in

end calculate_cost_l36_36925


namespace matrix_eq_zero_l36_36444

noncomputable def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![[1, 2, 0],
    [0, 1, 2],
    [2, 0, 1]]

theorem matrix_eq_zero :
  ∃ (p q r : ℤ), p = -3 ∧ q = 3 ∧ r = -9 ∧ 
  B * B * B + p • (B * B) + q • B + r • (1 : Matrix (Fin 3) (Fin 3) ℤ) = 0 := by
  use -3, 3, -9
  sorry

end matrix_eq_zero_l36_36444


namespace probability_of_divisor_of_12_is_one_half_l36_36636

theorem probability_of_divisor_of_12_is_one_half :
  let divisors := {1, 2, 3, 4, 6, 12} in
  let total_outcomes := finset.range 12 in
  (divisors.card : ℚ) / (total_outcomes.card : ℚ) = 1 / 2 :=
by
  let divisors := finset.from_list [1, 2, 3, 4, 6, 12]
  let total_outcomes := finset.range 12
  have h1 : divisors.card = 6 := by sorry
  have h2 : total_outcomes.card = 12 := by sorry
  rw [←h1, ←h2]
  norm_num
  sorry

end probability_of_divisor_of_12_is_one_half_l36_36636


namespace solve_nested_function_l36_36372

def f (x : ℝ) : ℝ := x^2 + 12 * x + 30

theorem solve_nested_function :
  ∃ x : ℝ, f (f (f (f (f x)))) = 0 ↔ (x = -6 + 6^(1/32) ∨ x = -6 - 6^(1/32)) :=
by sorry

end solve_nested_function_l36_36372


namespace find_difference_between_dists_l36_36007

variable (a : ℝ) (P : ℝ × ℝ)
variable (F1 F2 O : ℝ × ℝ)
variable (k : ℝ)

noncomputable def ellipse (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / 96) = 1

def is_point_on_ellipse (p : ℝ × ℝ) : Prop :=
  ellipse a p.1 p.2

def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  Math.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def distances_ratio (p f1 f2 o : ℝ × ℝ) : Prop :=
  dist p f1 = 8 * k ∧ dist p f2 = 6 * k ∧ dist o f2 = 5 * k

theorem find_difference_between_dists
  (h1 : a > 4 * Real.sqrt 6)
  (h2 : is_point_on_ellipse P)
  (h3 : distances_ratio P F1 F2 O) :
  dist P F1 - dist P F2 = 4 := by
  sorry

end find_difference_between_dists_l36_36007


namespace sqrt_trig_identity_l36_36778

theorem sqrt_trig_identity
  (α : ℝ)
  (P : ℝ × ℝ)
  (hP: P = (Real.sin 2, Real.cos 2))
  (h_terminal: ∃ (θ : ℝ), P = (Real.cos θ, Real.sin θ)) :
  Real.sqrt (2 * (1 - Real.sin α)) = 2 * Real.sin 1 := 
sorry

end sqrt_trig_identity_l36_36778


namespace xiao_zhang_return_distance_xiao_zhang_no_refuel_needed_l36_36497

def total_distance : ℕ :=
  15 - 3 + 16 - 11 + 10 - 12 + 4 - 15 + 16 - 18

def fuel_consumption_per_km : ℝ := 0.6
def initial_fuel : ℝ := 72.2

theorem xiao_zhang_return_distance :
  total_distance = 2 := by
  sorry

theorem xiao_zhang_no_refuel_needed :
  (initial_fuel - fuel_consumption_per_km * (|15| + |3| + |16| + |11| + |10| + |12| + |4| + |15| + |16| + |18|)) >= 0 := by
  sorry

end xiao_zhang_return_distance_xiao_zhang_no_refuel_needed_l36_36497


namespace clarinet_cost_correct_l36_36296

noncomputable def total_spent : ℝ := 141.54
noncomputable def song_book_cost : ℝ := 11.24
noncomputable def clarinet_cost : ℝ := total_spent - song_book_cost

theorem clarinet_cost_correct : clarinet_cost = 130.30 :=
by
  sorry

end clarinet_cost_correct_l36_36296


namespace min_positive_period_cosine_of_alpha_l36_36784

-- Problem 1: Minimum positive period

theorem min_positive_period
  (ω : ℝ) (hω : 0 < ω ∧ ω < 2)
  (h_extreme : ∃ x : ℝ, x = π / 4 ∧ f_deriv ω x = 0) :
  T = 2 * π / ω :=
sorry

-- Problem 2: Cosine of the angle α

theorem cosine_of_alpha
  (g : ℝ → ℝ)
  (α : ℝ)
  (h_acute : α < π / 2)
  (h_g : g(α) = 4/3 - sqrt 2)
  (h_g_def : ∀ x, g(x) = 2 * sin (x - π / 6) - sqrt 2) :
  cos α = (sqrt 15 - 2) / 6 :=
sorry

end min_positive_period_cosine_of_alpha_l36_36784


namespace cone_volume_l36_36402

theorem cone_volume (l h : ℝ) (l_eq : l = 5) (h_eq : h = 4) : 
  (1 / 3) * Real.pi * ((l^2 - h^2).sqrt)^2 * h = 12 * Real.pi := 
by 
  sorry

end cone_volume_l36_36402


namespace regression_line_value_at_x2_l36_36345

-- Define the given data set and averages
variables {x y : ℕ → ℝ}
variable {n : ℕ}
def x̄ := (∑ i in finset.range n, x i) / n
def ȳ := (∑ i in finset.range n, y i) / n

-- Define the regression line equation and the conditions given in the problem
def regression_line (b : ℝ) (x : ℝ) : ℝ := b * x + 0.2

-- State the proof problem
theorem regression_line_value_at_x2 :
  (∀ i, i < n → (y i = regression_line ((5 - 0.2) / 4) (x i))) →
  x̄ = 4 →
  ȳ = 5 →
  regression_line 1.2 2 = 2.6 :=
by
  intros h_regline h_xbar h_ybar
  sorry

end regression_line_value_at_x2_l36_36345


namespace trajectory_curve_l36_36240

variable (b x y : ℝ)
variable (F P M N : ℝ × ℝ)

/- Given conditions -/
def F : ℝ × ℝ := (1, 0)
def P (b : ℝ) : ℝ × ℝ := (0, b)
def M (a : ℝ) : ℝ × ℝ := (a, 0)
def N (x y : ℝ) : ℝ × ℝ := (x, y)

-- Define vectors
def PM (a b : ℝ) : ℝ × ℝ := (a, -b)
def PF (b : ℝ) : ℝ × ℝ := (1, -b)
def PN (x y b : ℝ) : ℝ × ℝ := (x, y - b)
def NM (a x y : ℝ) : ℝ × ℝ := (a - x, -y)

/- Mathematical fact to prove -/
theorem trajectory_curve (b a : ℝ) (h1 : a + b^2 = 0)
  (h2 : 2 * (PN x y b) + (NM a x y) = (0, 0)) :
  (y^2 = 4 * x) ∧ (x = -1) :=
sorry

end trajectory_curve_l36_36240


namespace sequence_sum_is_25_div_3_l36_36077

noncomputable def sum_of_arithmetic_sequence (a n d : ℝ) : ℝ := (n / 2) * (2 * a + (n - 1) * d)

theorem sequence_sum_is_25_div_3 (a d : ℝ)
  (h1 : a + 4 * d = 1)
  (h2 : 3 * a + 15 * d = 2 * a + 8 * d) :
  sum_of_arithmetic_sequence a 10 d = 25 / 3 := by
  sorry

end sequence_sum_is_25_div_3_l36_36077


namespace solve_for_x_l36_36732

theorem solve_for_x :
  ∃ x : ℝ, 5 * (x - 9) = 3 * (3 - 3 * x) + 9 ∧ x = 4.5 :=
by
  use 4.5
  sorry

end solve_for_x_l36_36732


namespace g_monotonic_intervals_minimum_value_a_range_of_a_l36_36786

noncomputable def g (x : ℝ) : ℝ := x / Real.log x
noncomputable def f (x a : ℝ) : ℝ := g x - a * x

-- Part I: Proof for monotonic intervals of g(x)
theorem g_monotonic_intervals :
  (∀ x > e, g x > g (x + 1)) ∧ (∀ x < e, g x < g (x + 1)) ∧ (∀ x ∈ (0,1), g x < g (x + 1)) :=
sorry

-- Part II: Minimum value of a
theorem minimum_value_a (a : ℝ) :
  (∀ x > 1, deriv (f x) a < 0) ↔ a ≥ 1/4 :=
sorry

-- Part III: Range of a given the condition
theorem range_of_a (a : ℝ) :
  (∀ x₁ ∈ Icc e (e^2), ∃ x₂ ∈ Icc e (e^2), g x₁ ≤ deriv (f x₂) a + 2 * a) ↔ a ∈ Icc ((e^2) / 2 - 1/4) +∞ :=
sorry

end g_monotonic_intervals_minimum_value_a_range_of_a_l36_36786


namespace proof_problem_l36_36002

open Classical

noncomputable def general_term_formula (a : ℕ → ℕ) (d : ℤ) :=
  ∃ a1 : ℤ, ∀ n : ℕ, a n = a1 + n * d

noncomputable def sum_first_n_terms (b : ℕ → ℤ → ℝ) :=
  ∀ n : ℕ, (b (3 * n - 1)) * (b (3 * n + 2)) = (1 / (3 * n - 1))

noncomputable def T (n : ℕ) : ℝ := (n : ℕ) / (2 * (3 * n + 2))

theorem proof_problem :
  (∃ a d, (a 2 + a 5 = 25) ∧ (S (a 1) 5 = 55) ∧
  (∀ n : ℕ, general_term_formula a d) ∧ (sum_first_n_terms a b)) →
  ∀ n : ℕ, T(n) = n / (2 * (3 * n + 2)) :=
sorry

end proof_problem_l36_36002


namespace min_norm_c_l36_36013

-- Definitions and conditions
variables {V : Type*} [inner_product_space ℝ V] {a b c : V} {λ : ℝ}
variables (h_unit_a : ∥a∥ = 1) (h_unit_b : ∥b∥ = 1) (h_perp : ⟪a, b⟫ = 0)

-- Statement of the proof problem
theorem min_norm_c (h : c + a = λ • (c + b)) : ∥c∥ ≥ sqrt (1 / 2) :=
sorry

end min_norm_c_l36_36013


namespace find_y_when_x_is_6_l36_36806

variable (k x y : ℝ)

theorem find_y_when_x_is_6 (h1 : y = k * x^2) (h2 : 18 = k * 3^2) : y = 72 :=
by
  let k := 18 / 9
  let k := 2
  have : k = 2 := sorry
  have : y = 2 * 6^2 := sorry
  show y = 72 from this

end find_y_when_x_is_6_l36_36806


namespace fold_proof_l36_36094

-- Definitions based on the conditions
structure Square (side : ℝ) :=
  (A B C D : Points)
  (AB : side = A.dist B)
  (BC : side = B.dist C)
  (CD : side = C.dist D)
  (DA : side = D.dist A)
  (square : collinear [A, B, C, D])

structure Midpoint (A D E : Point) :=
  (AD_mid : E = (A+D)/2)

def length_of_FD {side_length : ℝ} (A B C D E F : Point) (ab_square : Square side_length)
  (is_midpoint : Midpoint A D E) :
  A.dist D = side_length → 
  G : Point → -- assuming G is already defined similarly as in given problem context
  coinciding_points (F, D) → -- Indicator or structure showing F is on line D
  (8 : ℝ - F.dist D) ^ 2 = (x : ℝ) ^ 2 + 4 ^ 2 → x = 3 -- We're using 4 in place of E to D distance from paper

-- Main problem statement function
theorem fold_proof : {side_length : ℝ} (A B C D E F : Point)
  (ab_square : Square side_length)
  (is_midpoint : Midpoint A D E)
  (C_to_E : coincidence C E)
  (G : Point)
  (F_on_CD : C_F F D) :
  F.dist D = 3 := 
  by sorry

end fold_proof_l36_36094


namespace distance_between_intersections_l36_36995

-- Define the ellipse equation as a condition
def ellipse_eq (x y : ℝ) : Prop :=
  (x ^ 2) / 36 + (y ^ 2) / 16 = 1

-- Define the parabola equation given its focus at (2 * sqrt 5, 0) 
-- and directrix parallel to the y-axis
def parabola_eq (x y : ℝ) : Prop :=
  x = (y ^ 2) / (4 * sqrt 5) + sqrt 5

-- The problem statement about the distance between the intersection points
theorem distance_between_intersections :
  ∀ x1 y1 x2 y2 : ℝ, 
    ellipse_eq x1 y1 → ellipse_eq x2 y2 → 
    parabola_eq x1 y1 → parabola_eq x2 y2 → 
    (y1 ≠ y2) →  -- Ensure they are distinct points
    real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 2 * sqrt 10 :=
by
  sorry -- Proof is omitted as per instructions

end distance_between_intersections_l36_36995


namespace general_solution_of_differential_eq_l36_36952

noncomputable def y (x C : ℝ) : ℝ := x * (Real.exp (x ^ 2) + C)

theorem general_solution_of_differential_eq {x C : ℝ} (h : x ≠ 0) :
  let y' := (1 : ℝ) * (Real.exp (x ^ 2) + C) + x * (2 * x * Real.exp (x ^ 2))
  y' = (y x C / x) + 2 * x ^ 2 * Real.exp (x ^ 2) :=
by
  -- the proof goes here
  sorry

end general_solution_of_differential_eq_l36_36952


namespace isosceles_triangle_concyclic_l36_36851

open EuclideanGeometry

variables {A B C D E F G : Point}
variables {Γ : Circle}

def isosceles_triangle (A B C : Point) : Prop :=
  angle A B C = angle A C B

def circumcircle (A B C : Point) (Γ : Circle) : Prop :=
  Γ = Circle.mk A (A.dist B)

def concyclic (D E F G : Point) : Prop :=
  ∃ Γ : Circle, Γ ∈ [D, E, F, G]

theorem isosceles_triangle_concyclic
  (h1: isosceles_triangle A B C)
  (h2: circumcircle A B C Γ)
  (h3: D ∈ LineSegment B C)
  (h4: E ∈ LineSegment B C)
  (h5: F ∈ (Line AD ∩ Γ.pointSet))
  (h6: G ∈ (Line AE ∩ Γ.pointSet)) :
  concyclic D E F G :=
sorry

end isosceles_triangle_concyclic_l36_36851


namespace dr_math_house_number_count_l36_36302

noncomputable def is_prime (n : ℕ) : Prop := 
  ∀ m : ℕ, 2 ≤ m → m * m ≤ n → n % m ≠ 0 

def two_digit_prime (n : ℕ) : Prop := 
  50 < n ∧ n < 100 ∧ is_prime n 

def house_number_count : ℕ :=
  let primes := {p : ℕ | two_digit_prime p}.toList in
  primes.card * (primes.card - 1)

theorem dr_math_house_number_count : 
  house_number_count = 90 :=
sorry

end dr_math_house_number_count_l36_36302


namespace compare_surface_areas_l36_36776

theorem compare_surface_areas (V : ℝ) (V_pos : V > 0) :
  let S_cube := 6 * V^(2/3),
      S_cylinder := 8 * V^(2/3) / ((2 * π)^(1/3)),
      S_sphere := (36 * π * V^2)^(1/3)
  in S_sphere < S_cylinder ∧ S_cylinder < S_cube := by
  sorry

end compare_surface_areas_l36_36776


namespace arithmetic_mean_of_positive_three_digit_multiples_of_8_l36_36954

open Nat

theorem arithmetic_mean_of_positive_three_digit_multiples_of_8 : 
  let a := 104
  let l := 992
  2 * ∑ k in range 112, (8 * (k + 13)) / 112 = 548 :=
by
  sorry

end arithmetic_mean_of_positive_three_digit_multiples_of_8_l36_36954


namespace proof_problem_l36_36222

-- Definitions
def systematic_sampling (students : list ℕ) (m : ℕ) : Prop :=
  ∃ (seq : list ℕ), seq = list.range' m (students.length / 50) ∧
  (∀ i, i < seq.length → students.nth i = some ((i * 50) + m))

def normal_population (mean variance : ℝ) (interval1 interval2 : set ℝ) : Prop :=
  let prob := λ (interval : set ℝ), -- define probability function here
  prob interval1 = prob interval2

def linear_correlation_coeff (r : ℝ) : Prop :=
  r = 1 ∨ r = -1

def mode_and_median (data : list ℝ) : Prop :=
  let average := (data.sum) / (data.length : ℝ) in
  average = 2 ∧ (data.count (average - 1) = 2) ∧ (data.nth_le (data.length / 2) sorry = average)


-- Construction
theorem proof_problem :
  let P_A := systematic_sampling (list.range' 1 1000) (1 + 50) in
  let P_B := normal_population 1 9 { x : ℝ | -1 ≤ x ∧ x < 0 } { x : ℝ | 2 ≤ x ∧ x < 3 } in
  let P_D := mode_and_median [1, 2, 2, 3] in
  ¬linear_correlation_coeff 1 :=
by
-- P_A meets the condition for systematic sampling
have P_A : systematic_sampling (list.range' 1 50) 1 := sorry,
-- P_B matches the property of the normal distribution
have P_B : normal_population 1 9 { x : ℝ | -1 ≤ x ∧ x < 0 } { x : ℝ | 2 ≤ x ∧ x < 3 } := sorry,
-- P_D matches the condition of average, mode and median
have P_D : mode_and_median [1, 2, 2, 3] := sorry,
-- Therefore, ¬linear_correlation_coeff 1
exact sorry

end proof_problem_l36_36222


namespace find_AC_l36_36103

noncomputable def AC_in_triangle_ABC : ℝ :=
by 
  -- given conditions
  let AM := 15 : ℝ
  let BN := 20 : ℝ
  -- medians AM and BN are perpendicular
  sorry

-- The theorem we want to prove:
theorem find_AC (h1 : ∀ (A B C : ℝ), AM = 15 ∧ BN = 20 ∧ ∠ AMB = 90°) : AC_in_triangle_ABC = 20 * sqrt 13 / 3 := by
  sorry

end find_AC_l36_36103


namespace probability_sum_less_than_7_l36_36947

def red_die := fin 6
def blue_die := fin 6

theorem probability_sum_less_than_7 :
  (∃ (r b : ℕ) (hr : r ∈ finset.range 1 7) (hb : b ∈ finset.range 1 7), r + b < 7) / 36 = (5 : ℝ) / 12 :=
sorry

end probability_sum_less_than_7_l36_36947


namespace solve_cauchy_problem_l36_36912

open Real

noncomputable def cauchy_problem_solution : CauchyProblem :=
  ⟨λ (y : ℝ → ℝ) (y' : ℝ → ℝ) (x : ℝ),
    3 * (deriv (deriv y)) x = 4 * x^3 / (y' x)^2,
   λ (y : ℝ → ℝ) (x : ℝ), y 1 = 0,
   λ (y : ℝ → ℝ) (y' : ℝ → ℝ), y' 1 = 2⟩

theorem solve_cauchy_problem (y : ℝ → ℝ) :
  cauchy_problem_solution y (deriv y) →
  y = λ x, ∫ t in 1..x, (t^4 + 7)^(1/3) :=
sorry

end solve_cauchy_problem_l36_36912


namespace length_MN_eq_sqrt_ab_l36_36347

variables {A B C D M N O : Type}
          [MetricSpace A]
          [MetricSpace B]
          [MetricSpace C]
          [MetricSpace D]
          [MetricSpace M]
          [MetricSpace N]
          [MetricSpace O]

noncomputable def trapezoid (h₀ : LineSegment A D) (h₁ : LineSegment B C) : Prop :=
h₀.length = a ∧ h₁.length = b

variables (trapezoid_ABCD : trapezoid (LineSegment A D := a) (LineSegment B C := b))
          (parallel_MN_AD_BC : MN ∥ (LineSegment A D) ∧ MN ∥ (LineSegment B C))
          (O_on_AC : O ∈ LineIntersection (Line AC) (Line MN))
          (equal_area_AMO_CNO : AreaTriangle A M O = AreaTriangle C N O)

theorem length_MN_eq_sqrt_ab :
  length (LineSegment M N) = Real.sqrt (a * b) :=
by
  sorry

end length_MN_eq_sqrt_ab_l36_36347


namespace flight_time_l36_36669

def check_in_time := 2
def drive_time := 45
def park_time := 15
def latest_departure := 17 -- 5:00 PM in 24-hour format

def total_time_before_flight := check_in_time + (drive_time / 60.0) + (park_time / 60.0)

theorem flight_time : (latest_departure + total_time_before_flight) = 20 := by
  sorry

end flight_time_l36_36669


namespace area_of_triangle_is_correct_l36_36579

variable (a b γ h_a h_b : ℝ)

-- Define the conditions
def areConditionsSatisfied : Prop :=
  h_a = a * sin γ ∧
  h_b = b * sin γ

-- Define the area of the triangle based on the conditions
def areaOfTriangle : ℝ :=
  1 / 2 * (h_a * h_b / sin γ)

-- The proof statement to be proven
theorem area_of_triangle_is_correct :
  areConditionsSatisfied a b γ h_a h_b → 
  (areaOfTriangle a b γ h_a h_b = 1 / 2 * (h_a * h_b / sin γ)) :=
by
  -- This placeholder indicates where the proof would go.
  sorry

end area_of_triangle_is_correct_l36_36579


namespace garden_perimeter_l36_36559

theorem garden_perimeter (w l : ℕ) (garden_width : ℕ) (garden_perimeter : ℕ)
  (garden_area playground_length playground_width : ℕ)
  (h1 : garden_width = 16)
  (h2 : playground_length = 16)
  (h3 : garden_area = 16 * l)
  (h4 : playground_area = w * playground_length)
  (h5 : garden_area = playground_area)
  (h6 : garden_perimeter = 2 * l + 2 * garden_width)
  (h7 : garden_perimeter = 56):
  l = 12 :=
by
  sorry

end garden_perimeter_l36_36559


namespace distance_M_to_AB_eq_diameter_incircle_l36_36426

variable {A B C I M : Point}
variable (triangle : Triangle ABC)
variable [Incenter I triangle]
variable [PointOnSide M BC]
variable (angleBIM : angle B I M = 90)

theorem distance_M_to_AB_eq_diameter_incircle :
  let incircle := incircle_triangle triangle in
  distance_from_point_to_line M (line_through A B) = 2 * radius incircle :=
by
  sorry

end distance_M_to_AB_eq_diameter_incircle_l36_36426


namespace problem1_problem2_l36_36785

noncomputable theory

-- Definitions of f(x), a ∈ ℝ, and x > 0
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x
def tangent_to_x_axis (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f x₀ = 0 ∧ ∀ x, (differentiable_at ℝ f x₀) ∧ (derivative f x₀ = 0)

-- Problem 1: Prove that if f(x) = ln(x) - ax is tangent to the x-axis, then a = 1/e
theorem problem1 : ∀ (a : ℝ), (∃ x₀ > 0, tangent_to_x_axis (f a) x₀) → a = 1 / Real.exp 1 :=
sorry

-- Problem 2: Prove that for a > 0, the minimum value of f(x) on the interval [1,2] is as described.
theorem problem2: ∀ (a : ℝ), a > 0 → 
  (∀ x, 1 ≤ x ∧ x ≤ 2 → (-a ≤ f a x ∧ (a ≥ Real.log 2 → f a x ≤ Real.log 2 - 2 * a))) :=
sorry

end problem1_problem2_l36_36785


namespace solve_inequalities_l36_36494

theorem solve_inequalities :
  (∀ x : ℝ, x^2 + 3 * x - 10 ≥ 0 ↔ (x ≤ -5 ∨ x ≥ 2)) ∧
  (∀ x : ℝ, x^2 - 3 * x - 2 ≤ 0 ↔ (3 - Real.sqrt 17) / 2 ≤ x ∧ x ≤ (3 + Real.sqrt 17) / 2) :=
by
  sorry

end solve_inequalities_l36_36494


namespace finite_graph_proper_coloring_l36_36907

noncomputable def chromatic_number (G : Type) [graph G] : ℕ :=
  Inf {k : ℕ | ∃ (col : G → fin k), ∀ {u v : G}, G.adj u v → col u ≠ col v}

theorem finite_graph_proper_coloring {G : Type} [fintype G] [graph G] :
  ∃ k ≤ fintype.card G, ∃ (col : G → fin k), ∀ {u v : G}, G.adj u v → col u ≠ col v :=
by
  sorry

end finite_graph_proper_coloring_l36_36907


namespace debate_team_arrangements_l36_36655

theorem debate_team_arrangements :
  (∃ (A_pos : ℕ) (rest_pos : set ℕ), 
    A_pos ∈ {1, 2, 3} ∧
    rest_pos = {1, 2, 3, 4, 5} ∖ {A_pos} ∧
    ∃ (perm : {s : finset ℕ // s = rest_pos}),
    perm.1.card = 3) → 
  ∃ (num_arrangements : ℕ), num_arrangements = 180 :=
by {
  sorry
}

end debate_team_arrangements_l36_36655


namespace efficiency_ratio_l36_36230

-- Define efficiencies as fractions of work per day
def efficiency_a := 1 / 27
def combined_efficiency := 1 / 18

-- Define the relationship between efficiencies
theorem efficiency_ratio (A B : ℝ) (hA : A = efficiency_a) (hComb : A + B = combined_efficiency) : A / B = 2 :=
by {
  -- Directly use the given conditions and key result
  rw [hA, hComb],
  sorry -- Proof steps are not required as per instructions
}

end efficiency_ratio_l36_36230


namespace delta_eq_bullet_l36_36054

-- Definitions of all variables involved
variables (Δ Θ σ : ℕ)

-- Condition 1: Δ + Δ = σ
def cond1 : Prop := Δ + Δ = σ

-- Condition 2: σ + Δ = Θ
def cond2 : Prop := σ + Δ = Θ

-- Condition 3: Θ = 3Δ
def cond3 : Prop := Θ = 3 * Δ

-- The proof problem
theorem delta_eq_bullet (Δ Θ σ : ℕ) (h1 : Δ + Δ = σ) (h2 : σ + Δ = Θ) (h3 : Θ = 3 * Δ) : 3 * Δ = Θ :=
by
  -- Simply restate the conditions and ensure the proof
  sorry

end delta_eq_bullet_l36_36054


namespace integral_value_l36_36282

noncomputable def definite_integral : ℝ :=
  ∫ x in (Real.pi / 2) .. 2 * Real.arctan 2, 
    1 / (Real.sin x ^ 2 * (1 + Real.cos x))

theorem integral_value : definite_integral = 29 / 24 := 
by
  sorry

end integral_value_l36_36282


namespace measure_angle_DAE_l36_36654

-- Define right triangle ABC with given angles
variables (A B C D E : Type) [geometry A B C] 

-- Given conditions
axiom angle_BAC_eq_90 : ∠BAC = 90°
axiom angle_ABC_eq_60 : ∠ABC = 60°
axiom common_side_BC : side BC of triangle and rectangle
axiom BC_longer_side : BC is longer side of rectangle

-- Theorem statement
theorem measure_angle_DAE : ∠DAE = 0° :=
by
  sorry

end measure_angle_DAE_l36_36654


namespace exists_consecutive_nat_with_prime_factors_l36_36112

theorem exists_consecutive_nat_with_prime_factors (a b c m n : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hm : m > 0) (hn : n > 0) :
  ∃ (a1 a2 ... an : ℕ), (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → (f (ai i)) = ax^2 + bx + c) → (∃ (f_x_with_prime_factors : ℕ), (∀ j : ℕ, n≤ j) → Hnprime (min f_x_with_prime_factors) ≥ m) :=
sorry

end exists_consecutive_nat_with_prime_factors_l36_36112


namespace closed_route_exists_l36_36448

variable (n : ℕ) (n_ge_3 : n ≥ 3)
variable (airports : fin n → Type)
variable (airlines : fin n → set (set (fin n)))
variable (m : ℕ) (m_ge_3 : m ≥ 3) (m_odd : odd m)
variable (flights : Π (a : fin n), set ((fin n) × (fin n)))
variable (condition : ∀ (a : fin n), ∃ c : vector (fin n) m,
  (flights a = { (c i, c (i+1)%m) | i < m }))

theorem closed_route_exists : ∃ (route : list (fin n × fin n)),
  (route.length % 2 = 1) ∧
  (∀ (i < route.length), ∃ (a : fin n), ∃ (f ∈ flights a), f = route.nth_le i sorry).

end closed_route_exists_l36_36448


namespace exactly_two_friends_visit_l36_36667

/-- 
  Given:
  - Alice visits Daphne every 5 days
  - Beatrix visits Daphne every 6 days
  - Claire visits Daphne every 8 days
  - Exclude weekends from the counting
  Prove:
  - The number of weekdays over the next 400 days when exactly two friends visit Daphne is 39.
-/
theorem exactly_two_friends_visit (hAlice : nat := 5) (hBeatrix : nat := 6) (hClaire : nat := 8) : 
    let total_days := 400 in 
    let exclude_weekends := 2 * (total_days / 7) + min (total_days % 7) 2 in
    let effective_days := total_days - exclude_weekends in
    find_num_exactly_two_friends (hAlice : nat) (hBeatrix : nat) (hClaire : nat) effective_days = 39 :=
sorry

noncomputable def find_num_exactly_two_friends (a b c n : nat) : nat :=
  let period := nat.lcm (nat.lcm a b) c
  let days_in_period := period - 2 * (period / 7)
  let periods := n / period
  let remainder := n % period
  let extra_days := remainder - 2 * (remainder / 7)
  let count_pairs period := let lcm_ab := nat.lcm a b, lcm_ac := nat.lcm a c, lcm_bc := nat.lcm b c in
  periods * ((period / lcm_ab) + (period / lcm_ac) + (period / lcm_bc) - 3 * (period / period)) + 
    ((remainder / lcm_ab) + (remainder / lcm_ac) + (remainder / lcm_bc) - 3 * (remainder / period))
  count_pairs period

#eval exactly_two_friends_visit

end exactly_two_friends_visit_l36_36667


namespace find_e_value_l36_36871

theorem find_e_value : 
  ∃ e : ℝ, 12 / (-12 + 2 * e) = -11 - 2 * e ∧ e = 4 :=
by
  use 4
  sorry

end find_e_value_l36_36871


namespace problem_statement_l36_36374

noncomputable
def parabola := λ (x : ℝ), x^2

noncomputable
def circle (h : ℝ) := λ (x : ℝ y : ℝ), x^2 + (y - h)^2 = 1

theorem problem_statement : 
    let center : ℝ × ℝ := (0, 5 / 4)
    let area := (sqrt 3) - (π / 3)
    in
    (circle (5 / 4) = (λ (x : ℝ y : ℝ), x^2 + (y - (5 / 4))^2 = 1)) ∧
    (∀ x, parabola x = x^2) ∧
    ∃ a b, ∫ t within 0..a.length, ((5 / 4) - sqrt(1 - t^2) - t^2) = area :=
begin
    sorry
end

end problem_statement_l36_36374


namespace parabola_directrix_correct_l36_36931

def parabola_directrix (y : ℝ) (x : ℝ) : ℝ := 4 * x^2

theorem parabola_directrix_correct (x : ℝ) (y = 4 * x^2) : y = - (1 : ℝ) / 16 := 
by sorry

end parabola_directrix_correct_l36_36931


namespace guy_cards_problem_l36_36384

theorem guy_cards_problem 
  (cards: Fin 17 → ℤ) 
  (h: ∀ i, (cards i) ^ 2 = ∑ j, if j ≠ i then cards j else 0) :
  (∀ i, cards i = 0) ∨ (∀ i, cards i = 16) :=
by sorry

end guy_cards_problem_l36_36384


namespace university_tuition_cost_l36_36111

theorem university_tuition_cost :
  ∀ (c_first8 c_next10 c_total jc tc : ℝ),
    c_first8 = 80000 →
    c_next10 = 200000 →
    c_total = 280000 →
    jc = 140000 →
    tc = 265000 →
    let uc := tc - jc in
    uc = 125000 :=
by
  intros c_first8 c_next10 c_total jc tc h1 h2 h3 h4 h5
  let uc := tc - jc
  have : uc = 125000 := by sorry
  exact this

end university_tuition_cost_l36_36111


namespace price_reduction_l36_36568

theorem price_reduction (P : ℝ) :
  let first_day_price := P * 0.92,
      second_day_price := first_day_price * 0.90 in
  (second_day_price / P) * 100 = 82.8 := by
  -- proof goes here
  sorry

end price_reduction_l36_36568


namespace last_remaining_number_l36_36843

theorem last_remaining_number (n : ℕ) (h : n = 200) : 
  ∃ m, m ≤ n ∧ (∀ k, k ≠ m → 
  (k = 1 ∨ (∃ i, k = i * 2 + 1 ∨ k = (i + 1) * 2 + 1 ↔ (((∃ j, i = 2 ^ j ∧ i ≤ n) ∧ k ≤ n) ∨ (∃ j, i + 1 = 2 ^ j ∧ i + 1 ≤ n ∧ k ≤ n)))) :=
by
  use 128
  split
  { exact dec_trivial } -- 128 ≤ 200
  { intros k hk
    split
    { exact ⟨6, dec_trivial, dec_trivial⟩ } -- 128 is a power of 2 under 200
    sorry
  }

end last_remaining_number_l36_36843


namespace left_side_value_l36_36197

-- Define the relevant variables and conditions
variable (L R B : ℕ)

-- Assuming conditions
def sum_of_sides (L R B : ℕ) : Prop := L + R + B = 50
def right_side_relation (L R : ℕ) : Prop := R = L + 2
def base_value (B : ℕ) : Prop := B = 24

-- Main theorem statement
theorem left_side_value (L R B : ℕ) (h1 : sum_of_sides L R B) (h2 : right_side_relation L R) (h3 : base_value B) : L = 12 :=
sorry

end left_side_value_l36_36197


namespace find_de_l36_36186

variable {A B C O : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O]

def triangle_perimeter (ABC : Triangle A B C) : ℕ := 198
def right_angle_triangle (BAC: Triangle A B C) : Prop := angle BAC = 90
def circle_center (O : A) (radius : ℕ) : Prop := radius = 22
def ob_fractions (OB : ℝ) : Prop := OB = 55/2
def relatively_prime_pos (d e : ℕ) : Prop := gcd d e = 1

theorem find_de (ABC : Triangle A B C)
  (H1: triangle_perimeter ABC = 198)
  (H2: right_angle_triangle ABC)
  (H3: circle_center O = 22)
  (H4: ob_fractions (distance O B) = 55/2)
  (H5: relatively_prime_pos 55 2) : 
  (55 + 2 = 57) :=
sorry

end find_de_l36_36186


namespace hawks_total_points_l36_36918

/-- 
  Define the number of points per touchdown 
  and the number of touchdowns scored by the Hawks. 
-/
def points_per_touchdown : ℕ := 7
def touchdowns : ℕ := 3

/-- 
  Prove that the total number of points the Hawks have is 21. 
-/
theorem hawks_total_points : touchdowns * points_per_touchdown = 21 :=
by
  sorry

end hawks_total_points_l36_36918


namespace sum_b_eq_24_l36_36264

def b (n : ℕ) : ℚ :=
  if n = 1 then 2
  else if n = 2 then 3
  else (1/2) * b (n - 1) + (1/3) * b (n - 2)

theorem sum_b_eq_24 : (∑' n, b n).to_real = 24 := sorry

end sum_b_eq_24_l36_36264


namespace find_k_l36_36833

def sequence (n : ℕ) (c k : ℝ) : ℝ :=
  if n = 0 then 0 else c ^ n * (3 + k) / (3 - c)

def sum_sequence (n : ℕ) (c k : ℝ) (S : ℕ → ℝ) : ℝ :=
  if n = 0 then k else S n = (3 ^ n + k)

theorem find_k (S : ℕ → ℝ) (c k : ℝ) (h1 : ∀ n, sum_sequence n c k S) : k = -1 :=
by
  sorry

end find_k_l36_36833


namespace base9_addition_correct_l36_36665

theorem base9_addition_correct :
  let a := 2175 : ℕ -- interpreted as 2175 in base 9
  let b := 1714 : ℕ -- interpreted as 1714 in base 9
  let c := 406 : ℕ -- interpreted as 406 in base 9
  let sum := 4406 : ℕ -- the correct sum in base 9
  (a + b + c : ℕ) % (9^4) = sum % (9^4) := 
by 
  sorry

end base9_addition_correct_l36_36665


namespace A_alone_finishes_in_27_days_l36_36968

noncomputable def work (B : ℝ) : ℝ := 54 * B  -- amount of work W
noncomputable def days_to_finish_alone (B : ℝ) : ℝ := (work B) / (2 * B)

theorem A_alone_finishes_in_27_days (B : ℝ) (h : (work B) / (2 * B + B) = 18) : 
  days_to_finish_alone B = 27 :=
by
  sorry

end A_alone_finishes_in_27_days_l36_36968


namespace arrange_abc_l36_36854

theorem arrange_abc (a b c : ℝ) (h1 : a = 0.4^3) (h2 : b = Real.log 3 / Real.log 0.4) (h3 : c = 3^0.4) :
  b < a ∧ a < c := 
by 
  sorry

end arrange_abc_l36_36854


namespace find_first_number_l36_36180

theorem find_first_number (N : ℤ) (k m : ℤ) (h1 : N = 170 * k + 10) (h2 : 875 = 170 * m + 25) : N = 860 :=
by
  sorry

end find_first_number_l36_36180


namespace seating_arrangements_l36_36474

-- Define the family members
inductive FamilyMember
| MrLopez
| MrsLopez
| Alice
| Bob

open FamilyMember

-- Define the conditions
def driver_seats : List FamilyMember := [MrLopez, MrsLopez]
def front_seats : List FamilyMember := [MrLopez, MrsLopez, Bob]  -- Alice must sit in the front
def back_seats : List FamilyMember := [MrLopez, MrsLopez, Bob]  -- others will fill the back seats

-- The theorem to prove
theorem seating_arrangements : 
  let total_arrangements := 
    (driver_seats.length * front_seats.length * 2)
  in 
  total_arrangements = 8 := 
by sorry

end seating_arrangements_l36_36474


namespace collinear_intersections_l36_36146

-- Define the points and relevant lines.
noncomputable def circumcircle (A B C : Point) : Circle := sorry
noncomputable def intersection (l1 l2 : Line) : Point := sorry
noncomputable def line (P Q : Point) : Line := sorry

-- Given points and their properties.
variables (A B C R M : Point)
variable (circumcircle_ABC : circumcircle A B C)
variable (M_on_circumcircle : M ∈ circumcircle_ABC)

-- Points of intersection with the circumcircle.
variables (A1 B1 C1 : Point)
variable (A1_is_AR_circ_intersection : A1 = intersection (line A R) circumcircle_ABC)
variable (B1_is_BR_circ_intersection : B1 = intersection (line B R) circumcircle_ABC)
variable (C1_is_CR_circ_intersection : C1 = intersection (line C R) circumcircle_ABC)

-- Points of intersection of the lines.
variables (A2 B2 C2 : Point)
variable (A2_is_MA1_BC_intersection : A2 = intersection (line M A1) (line B C))
variable (B2_is_MB1_CA_intersection : B2 = intersection (line M B1) (line C A))
variable (C2_is_MC1_AB_intersection : C2 = intersection (line M C1) (line A B))

-- The collinearity statement.
theorem collinear_intersections (hM : M ∈ circumcircle_ABC)
  (hA1 : A1 = intersection (line A R) circumcircle_ABC)
  (hB1 : B1 = intersection (line B R) circumcircle_ABC)
  (hC1 : C1 = intersection (line C R) circumcircle_ABC)
  (hA2 : A2 = intersection (line M A1) (line B C))
  (hB2 : B2 = intersection (line M B1) (line C A))
  (hC2 : C2 = intersection (line M C1) (line A B))
  : collinear ({A2, B2, C2, R}) := 
  sorry

end collinear_intersections_l36_36146


namespace bond_value_after_8_years_l36_36482

theorem bond_value_after_8_years :
  ∀ (P A r t : ℝ), P = 240 → r = 0.0833333333333332 → t = 8 →
  (A = P * (1 + r * t)) → A = 400 :=
by
  sorry

end bond_value_after_8_years_l36_36482


namespace min_sum_hexagram_vertices_even_number_of_valid_arrangements_l36_36949

-- Definitions and assumptions based on the problem conditions
def hexagram_vertices : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- This function determines the sum of the four numbers on each straight line
def valid_hexagram_arrangement (a : List ℕ) (s : ℕ) : Prop := 
  (a.length = 12) ∧ (∀ i j k l : ℕ, 
    i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l → 
    i < 12 ∧ j < 12 ∧ k < 12 ∧ l < 12 →
    List.getD a i 0 + List.getD a j 0 + List.getD a k 0 + List.getD a l 0 = s)

-- Problem statement (1): Minimum sum of the vertices
theorem min_sum_hexagram_vertices (a : List ℕ) (h : valid_hexagram_arrangement a 26) :
  a.take 6.sum = 24 :=
sorry

-- Problem statement (2): Even number of valid arrangements
theorem even_number_of_valid_arrangements : 
  ∃ a₁ a₂ : List ℕ, valid_hexagram_arrangement a₁ 26 ∧ valid_hexagram_arrangement a₂ 26 ∧ 
  (a₁ ≠ a₂ ∧ (
    ∀ (θ : ℕ), rotate_list a₁ θ = a₂ ∨ reflect_list a₁ = a₂)) :=
sorry

-- Auxiliary function definitions
noncomputable def rotate_list (l : List ℕ) (n : ℕ) : List ℕ :=
  (l.drop n) ++ (l.take n)

noncomputable def reflect_list (l : List ℕ) : List ℕ :=
  l.reverse

-- Assuming necessary imports and subfunctions

end min_sum_hexagram_vertices_even_number_of_valid_arrangements_l36_36949


namespace probability_divisor_of_12_on_12sided_die_l36_36615

theorem probability_divisor_of_12_on_12sided_die (h : ∀ i, 1 ≤ i ∧ i ≤ 12 → true) : 
  (probability (fun (n : ℕ) => n ∣ 12 ∧ 1 ≤ n ∧ n ≤ 12)) = 1/2 := 
sorry

end probability_divisor_of_12_on_12sided_die_l36_36615


namespace area_BCM_independent_of_D_l36_36540

theorem area_BCM_independent_of_D {A B C D B' C' M : Type*}
  [IsoscelesTriangle A B C] (D_on_BC : PointOnLineSegment D B C)
  (circumcircle_ABD : BB' = DiameterOfCircumcircle (A, B, D))
  (circumcircle_ACD : CC' = DiameterOfCircumcircle (A, C, D))
  (M_midpoint_BC' : M = Midpoint B' C') :
  ∀ D, area (Triangle B C M) = area (Triangle B C M) :=
sorry

end area_BCM_independent_of_D_l36_36540


namespace number_of_parents_at_park_l36_36202

theorem number_of_parents_at_park (girls boys playgroups people_per_group : ℕ) 
  (h1 : girls = 14) 
  (h2 : boys = 11) 
  (h3 : playgroups = 3) 
  (h4 : people_per_group = 25) : 
  (75 - (girls + boys) = 50) := 
by
  rw [h1, h2, h3, h4]
  show 75 - (14 + 11) = 50
  sorry

end number_of_parents_at_park_l36_36202


namespace translation_m_n_l36_36483

theorem translation_m_n (m n : ℤ) (P Q : ℤ × ℤ) (hP : P = (-1, -3)) (hQ : Q = (-2, 0))
(hx : P.1 - m = Q.1) (hy : P.2 + n = Q.2) :
  m + n = 4 :=
by
  sorry

end translation_m_n_l36_36483


namespace measure_angle_ABG_l36_36149

theorem measure_angle_ABG (ABCDEFG : Polygon) (h_reg : regular_heptagon ABCDEFG) : 
  measure_angle ABCDEFG A B G = 25.715 := 
sorry

end measure_angle_ABG_l36_36149


namespace square_simplify_l36_36560

   variable (y : ℝ)

   theorem square_simplify :
     (7 - Real.sqrt (y^2 - 49)) ^ 2 = y^2 - 14 * Real.sqrt (y^2 - 49) :=
   sorry
   
end square_simplify_l36_36560


namespace labourer_savings_l36_36498

theorem labourer_savings
  (monthly_expenditure_first_6_months : ℕ)
  (monthly_expenditure_next_4_months : ℕ)
  (monthly_income : ℕ)
  (total_expenditure_first_6_months : ℕ)
  (total_income_first_6_months : ℕ)
  (debt_incurred : ℕ)
  (total_expenditure_next_4_months : ℕ)
  (total_income_next_4_months : ℕ)
  (money_saved : ℕ) :
  monthly_expenditure_first_6_months = 85 →
  monthly_expenditure_next_4_months = 60 →
  monthly_income = 78 →
  total_expenditure_first_6_months = 6 * monthly_expenditure_first_6_months →
  total_income_first_6_months = 6 * monthly_income →
  debt_incurred = total_expenditure_first_6_months - total_income_first_6_months →
  total_expenditure_next_4_months = 4 * monthly_expenditure_next_4_months →
  total_income_next_4_months = 4 * monthly_income →
  money_saved = total_income_next_4_months - (total_expenditure_next_4_months + debt_incurred) →
  money_saved = 30 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end labourer_savings_l36_36498


namespace general_term_a_sum_geometric_terms_sum_reciprocal_terms_l36_36005

section ArithmeticSequence
variables {a : ℕ → ℕ} {S : ℕ → ℕ}

-- Given conditions
def a3 : ℕ := 5
def S3 : ℕ := 9
def arithmetic (n : ℕ) : Prop := ∀ k, a k = 2 * k - 1
def sum_arithmetic (n : ℕ) : Prop := ∀ k, S k = k * (2 * k - 1)

-- Prove the general term formula for the sequence {a_n}
theorem general_term_a (n : ℕ) : arithmetic n :=
by sorry

end ArithmeticSequence

section GeometricSequence
variables {b : ℕ → ℕ} {T : ℕ → ℕ} {q : ℕ}

-- Given conditions
def b3 : ℕ := 9
def T3 : ℕ := 13
def geometric (n : ℕ) : Prop := ∀ k, b k = (1 : ℕ) * q ^ k
def sum_geometric (n : ℕ) : Prop := ∀ k, T k = (1 - q ^ k) / (1 - q)

-- Prove the sum of the first n terms for the sequence {b_n}
theorem sum_geometric_terms (n : ℕ) : T n = (1 / 2) * (3 ^ n - 1) :=
by sorry

end GeometricSequence

section ReciprocalSequence
variables {b : ℕ → ℕ} {a : ℕ → ℕ} {Sn : ℕ → ℕ}

-- Given conditions
def reciprocal (n : ℕ) : Prop := ∀ k, b k = 1 / (a k * a (k + 1))

-- Prove the sum of the first n terms for sequence {b_n}
theorem sum_reciprocal_terms (n : ℕ) : Sn n = n / (2 * n + 1) :=
by sorry

end ReciprocalSequence

end general_term_a_sum_geometric_terms_sum_reciprocal_terms_l36_36005


namespace probability_of_divisor_of_12_is_one_half_l36_36632

theorem probability_of_divisor_of_12_is_one_half :
  let divisors := {1, 2, 3, 4, 6, 12} in
  let total_outcomes := finset.range 12 in
  (divisors.card : ℚ) / (total_outcomes.card : ℚ) = 1 / 2 :=
by
  let divisors := finset.from_list [1, 2, 3, 4, 6, 12]
  let total_outcomes := finset.range 12
  have h1 : divisors.card = 6 := by sorry
  have h2 : total_outcomes.card = 12 := by sorry
  rw [←h1, ←h2]
  norm_num
  sorry

end probability_of_divisor_of_12_is_one_half_l36_36632


namespace Olivia_house_height_l36_36958

variable (h : ℕ)
variable (flagpole_height : ℕ := 35)
variable (flagpole_shadow : ℕ := 30)
variable (house_shadow : ℕ := 70)
variable (bush_height : ℕ := 14)
variable (bush_shadow : ℕ := 12)

theorem Olivia_house_height :
  (house_shadow / flagpole_shadow) * flagpole_height = 81 ∧
  (house_shadow / bush_shadow) * bush_height = 81 :=
by
  sorry

end Olivia_house_height_l36_36958


namespace construction_costs_correct_l36_36258

structure ConstructionCosts where
  landCostPerSqMeter : ℕ
  brickCostPerThousand : ℕ
  tileCostPerTile : ℕ
  landRequired : ℕ
  bricksRequired : ℕ
  tilesRequired : ℕ

noncomputable def totalConstructionCost (cc : ConstructionCosts) : ℕ :=
  let landCost := cc.landRequired * cc.landCostPerSqMeter
  let brickCost := (cc.bricksRequired / 1000) * cc.brickCostPerThousand
  let tileCost := cc.tilesRequired * cc.tileCostPerTile
  landCost + brickCost + tileCost

theorem construction_costs_correct (cc : ConstructionCosts)
  (h1 : cc.landCostPerSqMeter = 50)
  (h2 : cc.brickCostPerThousand = 100)
  (h3 : cc.tileCostPerTile = 10)
  (h4 : cc.landRequired = 2000)
  (h5 : cc.bricksRequired = 10000)
  (h6 : cc.tilesRequired = 500) :
  totalConstructionCost cc = 106000 := 
  by 
    sorry

end construction_costs_correct_l36_36258


namespace sqrt_of_square_of_neg_five_eq_five_l36_36689

theorem sqrt_of_square_of_neg_five_eq_five : Real.sqrt ((-5 : ℤ) ^ 2) = 5 := by
  sorry

end sqrt_of_square_of_neg_five_eq_five_l36_36689


namespace slower_train_time_to_pass_driver_faster_one_l36_36238

noncomputable def convert_speed (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

noncomputable def relative_speed (speed1_kmh speed2_kmh : ℝ) : ℝ :=
  let speed1 := convert_speed speed1_kmh
  let speed2 := convert_speed speed2_kmh
  speed1 + speed2

noncomputable def time_to_pass (length1_m length2_m speed1_kmh speed2_kmh : ℝ) : ℝ :=
  let relative_speed := relative_speed speed1_kmh speed2_kmh
  (length1_m + length2_m) / relative_speed

theorem slower_train_time_to_pass_driver_faster_one :
  ∀ (length1 length2 speed1 speed2 : ℝ),
    length1 = 900 → length2 = 900 →
    speed1 = 45 → speed2 = 30 →
    time_to_pass length1 length2 speed1 speed2 = 86.39 :=
by
  intros
  simp only [time_to_pass, relative_speed, convert_speed]
  sorry

end slower_train_time_to_pass_driver_faster_one_l36_36238


namespace find_possible_values_a_l36_36755

theorem find_possible_values_a (a : ℝ) : 
  (∃ A B, 
    A = {x | ax + 1 = 0} ∧ 
    B = {x | x^2 - 3 * x + 2 = 0} ∧ 
    A ∪ B = B) 
  ↔ a ∈ {-1/2, 0, -1} :=
by
  sorry

end find_possible_values_a_l36_36755


namespace exists_m_infinite_solutions_l36_36105

theorem exists_m_infinite_solutions : 
  ∃ m: ℕ, m = 18 ∧ ∃ᶠ (a b c: ℕ) in at_top, 
    (1:ℚ) / a + (1:ℚ) / b + (1:ℚ) / c + (1:ℚ) / (a * b * c) = m * (1:ℚ) / (a + b + c) :=
by
  sorry

end exists_m_infinite_solutions_l36_36105


namespace transform_polynomial_l36_36156

theorem transform_polynomial (n : ℕ) (a : Fin n → ℝ) :
  ∃ (b : Fin n → ℝ), ∀ x : ℝ, 
    (∑ i in Finset.range n, a i * x^(n - i)) = 0 →
    (∑ i in Finset.range n, b i * (x + a 1 / n)^(n - i)) = 0 :=
by
  sorry

end transform_polynomial_l36_36156


namespace handshake_problem_l36_36543

noncomputable def total_handshakes (num_companies : ℕ) (repr_per_company : ℕ) : ℕ :=
    let total_people := num_companies * repr_per_company
    let possible_handshakes_per_person := total_people - repr_per_company
    (total_people * possible_handshakes_per_person) / 2

theorem handshake_problem : total_handshakes 4 4 = 96 :=
by
  sorry

end handshake_problem_l36_36543


namespace arctan_add_radians_l36_36455

variable (a b : ℝ)

theorem arctan_add_radians :
  a = 3 / 4 ∧ (a + 1) * (b + 1) = 3 →
  arctan a + arctan b = arctan (41 / 13) :=
by {
  sorry
}

end arctan_add_radians_l36_36455


namespace cuboid_volume_l36_36973

theorem cuboid_volume (length width height : ℕ) (h_length : length = 4) (h_width : width = 4) (h_height : height = 6) : (length * width * height = 96) :=
by 
  -- Sorry places a placeholder for the actual proof
  sorry

end cuboid_volume_l36_36973


namespace coeff_x3_in_expansion_l36_36079

theorem coeff_x3_in_expansion : (nat.choose 50 3 * (1^47) * (1^3)) = 19600 := 
by sorry

end coeff_x3_in_expansion_l36_36079


namespace maritza_study_hours_l36_36874

noncomputable def time_to_study_for_citizenship_test (num_mc_questions num_fitb_questions time_mc time_fitb : ℕ) : ℕ :=
  (num_mc_questions * time_mc + num_fitb_questions * time_fitb) / 60

theorem maritza_study_hours :
  time_to_study_for_citizenship_test 30 30 15 25 = 20 :=
by
  sorry

end maritza_study_hours_l36_36874


namespace xiao_ming_stones_l36_36227

theorem xiao_ming_stones : 
  (∃ n : ℕ, (∑ k in range (n + 1), k) = 36) →
  (∑ k in range (n + 1), 2^k) = 510 :=
by
  sorry

end xiao_ming_stones_l36_36227


namespace probability_of_exactly_one_red_ball_l36_36946

-- Definitions based on the conditions:
def total_balls : ℕ := 5
def red_balls : ℕ := 2
def white_balls : ℕ := 3
def draw_count : ℕ := 2

-- Required to calculate combinatory values
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Definitions of probabilities (though we won't use them explicitly for the statement):
def total_events : ℕ := choose total_balls draw_count
def no_red_ball_events : ℕ := choose white_balls draw_count
def one_red_ball_events : ℕ := choose red_balls 1 * choose white_balls 1

-- Probability Functions (for context):
def probability (events : ℕ) (total_events : ℕ) : ℚ := events / total_events

-- Lean 4 statement:
theorem probability_of_exactly_one_red_ball :
  probability one_red_ball_events total_events = 3/5 := by
  sorry

end probability_of_exactly_one_red_ball_l36_36946


namespace abs_eq_ax_minus_one_has_two_solutions_l36_36332
noncomputable def midpoint_of_interval : ℕ → ℕ → ℚ := sorry

theorem abs_eq_ax_minus_one_has_two_solutions (a : ℚ) (h : a > 1/3) : 
  ∃ x1 x2, x1 ≠ x2 ∧ (|x1 - 3| = a * x1 - 1) ∧ (|x2 - 3| = a * x2 - 1) :=
sorry

#eval midpoint_of_interval 1 3 -- Expected output 0.667 or approximately 2/3

end abs_eq_ax_minus_one_has_two_solutions_l36_36332


namespace min_sum_product_permutations_l36_36127

theorem min_sum_product_permutations :
  ∀ (a b c : Fin 6 → Fin 6),
    bijective a →
    bijective b →
    bijective c →
    (∑ i, (a i).val * (b i).val * (c i).val) = 162 := sorry

end min_sum_product_permutations_l36_36127


namespace measurement_units_l36_36730

def length_of_ruler := 2 
def load_capacity_of_truck := 5 
def bus_travel_distance_per_hour := 100 
def thickness_of_math_book := 7 
def weight_of_backpack := 4000 

theorem measurement_units :
  (length_of_ruler == 2) ∧ (load_capacity_of_truck == 5) ∧ (bus_travel_distance_per_hour == 100) ∧ 
  (thickness_of_math_book == 7) ∧ (weight_of_backpack == 4000) →
  (length_of_ruler, "decimeters") ∧ (load_capacity_of_truck, "tons") ∧ 
  (bus_travel_distance_per_hour, "kilometers per hour") ∧
  (thickness_of_math_book, "millimeters") ∧
  (weight_of_backpack, "grams") :=
by
  intro h
  sorry

end measurement_units_l36_36730


namespace function_passes_through_point_b_l36_36509

theorem function_passes_through_point_b (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  (2, 2) ∈ { p : ℝ × ℝ | ∃ x, p = (x, a^(x - 2) + 1) } :=
by {
  use 2,
  rw sub_self,
  rw pow_zero,
  rw add_one,
  exact rfl,
}

end function_passes_through_point_b_l36_36509


namespace matrix_power_l36_36290

open Matrix

def matrix_1_0_x_1 (x : ℤ) : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], [x, 1]]

theorem matrix_power :
  matrix_1_0_x_1 2 ^ 2023 = matrix_1_0_x_1 4046 :=
by
  sorry

end matrix_power_l36_36290


namespace sum_of_sequence_l36_36342

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = -1 ∧ ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n + 3 * n - 1

noncomputable def sum_of_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = 2 ^ (n + 2) - 4 - (3 * n ^ 2 + 7 * n) / 2

theorem sum_of_sequence (a S : ℕ → ℤ) (h_seq : sequence a) (h_sum : sum_of_terms a S) :
  ∀ n : ℕ, S n = 2 ^ (n + 2) - 4 - (3 * n ^ 2 + 7 * n) / 2 :=
sorry

end sum_of_sequence_l36_36342


namespace shaded_region_area_l36_36701

theorem shaded_region_area {radius1 radius2 : ℝ} (h1 : radius1 = 4) (h2 : radius2 = 5) :
  let dist_centers := radius1 + radius2,
      circumscribed_radius := dist_centers,
      larger_area := Real.pi * circumscribed_radius ^ 2,
      smaller_area1 := Real.pi * radius1 ^ 2,
      smaller_area2 := Real.pi * radius2 ^ 2,
      shaded_area := larger_area - smaller_area1 - smaller_area2
  in shaded_area = 40 * Real.pi :=
by
  simp [h1, h2]
  sorry

end shaded_region_area_l36_36701


namespace cornelia_travel_countries_l36_36295

theorem cornelia_travel_countries (europe south_america asia half_remaining : ℕ) 
  (h1 : europe = 20)
  (h2 : south_america = 10)
  (h3 : asia = 6)
  (h4 : asia = half_remaining / 2) : 
  europe + south_america + half_remaining = 42 :=
by
  sorry

end cornelia_travel_countries_l36_36295


namespace sum_of_M_l36_36190

theorem sum_of_M (x y z : ℕ) (h1 : z = 2 * (x + y)) (h2 : xyz = 8 * (x + y + z)) :
  ∑ {M : ℕ // ∃ (x y : ℕ), z = 2 * (x + y) ∧ xyz = 8 * (x + y + z) ∧ M = xyz} = 560 :=
sorry

end sum_of_M_l36_36190


namespace time_to_plough_together_l36_36571

def work_rate_r := 1 / 15
def work_rate_s := 1 / 20
def combined_work_rate := work_rate_r + work_rate_s
def total_field := 1
def T := total_field / combined_work_rate

theorem time_to_plough_together : T = 60 / 7 :=
by
  -- Here you would provide the proof steps if it were required
  -- Since the proof steps are not needed, we indicate the end with sorry
  sorry

end time_to_plough_together_l36_36571


namespace problem_l36_36348

noncomputable def sequence_a (n : ℕ) : ℕ := 2 * n - 1

noncomputable def sequence_b (q : ℝ) (n : ℕ) : ℝ := q ^ (sequence_a n)

def S (n : ℕ) : ℝ := 
4 * (sequence_a 1) + (4 * 3 / 2) * 2 -- S_4
6 * (sequence_a 1) + (6 * 5 / 2) * 2 -- S_6

theorem problem (q : ℝ) (hq : 0 < q) (n : ℕ):
    sequence_a n = 2 * n - 1 ∧ 
    (T_n = if q ≠ 1 then (1 / (q^4 - 1)) * (1 - 1 / q^(4 * n)) else n) :=
by
  sorry

end problem_l36_36348


namespace find_p5_l36_36857

-- Definitions of the conditions
def is_monic_quartic (p : ℚ[X]) : Prop :=
  degree p = 4 ∧ leading_coeff p = 1

def satisfies_conditions (p : ℚ[X]) : Prop :=
  is_monic_quartic p ∧
  p.eval 1 = 2 ∧
  p.eval 2 = 3 ∧
  p.eval 3 = 6 ∧
  p.eval 4 = 11

-- The main theorem statement
theorem find_p5 (p : ℚ[X]) (h : satisfies_conditions p) : p.eval 5 = 48 :=
sorry

end find_p5_l36_36857


namespace domain_when_a_is_one_inequality_solution_set_when_a_is_zero_l36_36368

-- Definition of the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := log 2 (2 * x - a * x ^ 2)

-- Problem statement A: Domain of function when a = 1
theorem domain_when_a_is_one :
  ∀ x : ℝ, (0 < x ∧ x < 2) ↔ 2 * x - x ^ 2 > 0 :=
sorry

-- Problem statement B: Inequality solution set when a = 0
theorem inequality_solution_set_when_a_is_zero :
  ∀ x : ℝ, (0 < x ∧ x < 1) ↔ log 2 (2 * x) < 1 :=
sorry

end domain_when_a_is_one_inequality_solution_set_when_a_is_zero_l36_36368


namespace time_for_runnerA_to_complete_race_l36_36236

variable (speedA : ℝ) -- speed of runner A in meters per second
variable (t : ℝ) -- time taken by runner A to complete the race in seconds
variable (tB : ℝ) -- time taken by runner B to complete the race in seconds

noncomputable def distanceA : ℝ := 1000 -- distance covered by runner A in meters
noncomputable def distanceB : ℝ := 950 -- distance covered by runner B in meters when A finishes
noncomputable def speedB : ℝ := distanceB / tB -- speed of runner B in meters per second

theorem time_for_runnerA_to_complete_race
    (h1 : distanceA = speedA * t)
    (h2 : distanceB = speedA * (t + 20)) :
    t = 400 :=
by
  sorry

end time_for_runnerA_to_complete_race_l36_36236


namespace star_eval_l36_36328

theorem star_eval {x y : ℝ} : 
  let star (a b : ℝ) := (a + b)^2 in
  star ((x + y)^2) ((y + x)^2) = 4 * (x + y)^4 :=
by
  sorry

end star_eval_l36_36328


namespace apples_difference_l36_36664

theorem apples_difference
    (adam_apples : ℕ)
    (jackie_apples : ℕ)
    (h_adam : adam_apples = 10)
    (h_jackie : jackie_apples = 2) :
    adam_apples - jackie_apples = 8 :=
by
    sorry

end apples_difference_l36_36664


namespace digit_distribution_zero_digit_count_l36_36797

noncomputable def digit_frequency (k : ℕ) : ℕ :=
  if k = 0 then ∑ n in finset.range 2016, (2017 - n.succ) * 10^n
  else 2017 * 10^(2017 - k)

theorem digit_distribution : 
  (∀ k : ℕ, k ≠ 0 → ∀ count : ℕ, digit_frequencyk) = (2017 * 10^{2016}) :=
sorry

theorem zero_digit_count : digit_frequency 0 = ∑ n in finset.range 2016, (2017 - n.succ) * 10^n :=
sorry

end digit_distribution_zero_digit_count_l36_36797


namespace sum_of_primes_divisible_by_12_l36_36899

open Nat

theorem sum_of_primes_divisible_by_12 (p q : ℕ) (hp : Prime p) (hq : Prime q) (h_diff : p - q = 2) (h_q_gt_3 : q > 3) : 
  (p + q) % 12 = 0 := 
by 
  sorry

end sum_of_primes_divisible_by_12_l36_36899


namespace instantaneous_velocity_at_3_l36_36934

noncomputable def motion_equation (t : ℝ) : ℝ := 1 - t + t^2

theorem instantaneous_velocity_at_3 :
  (deriv (motion_equation) 3 = 5) :=
by
  sorry

end instantaneous_velocity_at_3_l36_36934


namespace parallelogram_side_length_l36_36996

theorem parallelogram_side_length (s : ℝ) 
  (h1 : 0 < s) 
  (adj_sides : ∀ (a b : ℝ), a = s ∧ b = 3 * s) 
  (angle_60 : ∀ (θ : ℝ), θ = 60) 
  (area_eq : ∀ (A : ℝ), A = 27 * (real.sqrt 3)) :
  s = real.sqrt 6 :=
by 
  sorry

end parallelogram_side_length_l36_36996


namespace exists_two_digit_number_N_l36_36117

-- Statement of the problem
theorem exists_two_digit_number_N : 
  ∃ (N : ℕ), (∃ (a b : ℕ), N = 10 * a + b ∧ N = a * b + 2 * (a + b) ∧ 10 ≤ N ∧ N < 100) :=
by
  sorry

end exists_two_digit_number_N_l36_36117


namespace tangent_line_to_circle_x_eq_0_l36_36506

noncomputable def is_tangent_line (l : ℝ → ℝ → Prop) (c : ℝ × ℝ) (r : ℝ) : Prop :=
  ∃ x : ℝ, ∃ y : ℝ, (x, y) = c ∧ (l x y) ∧ Real.sqrt((x - (x:-1)) ^ 2 + (y - (y:-√3))^2) = r

theorem tangent_line_to_circle_x_eq_0 :
  is_tangent_line (λ x y, x = 0) (-1, -Real.sqrt 3) 1 :=
sorry

end tangent_line_to_circle_x_eq_0_l36_36506


namespace intersection_M_N_l36_36795

-- Define set M
def set_M : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}

-- Define set N
def set_N : Set ℤ := {x | ∃ k : ℕ, k > 0 ∧ x = 2 * k - 1}

-- Define the intersection of M and N
def M_intersect_N : Set ℤ := {1, 3}

-- The theorem to prove
theorem intersection_M_N : set_M ∩ set_N = M_intersect_N :=
by sorry

end intersection_M_N_l36_36795


namespace coefficient_x3_in_binomial_expansion_l36_36084

theorem coefficient_x3_in_binomial_expansion :
  nat.choose 50 3 = 19600 :=
by
  -- Proof goes here
  sorry

end coefficient_x3_in_binomial_expansion_l36_36084


namespace point_D_satisfies_inequality_l36_36827

-- Definition of the points
def point_A := (0, 0)
def point_B := (-2, 0)
def point_C := (0, -1)
def point_D := (0, 2)

-- Definition of the inequality condition
def inequality (x y : ℝ) : Prop := (x + 2 * y - 1) * (x - y + 3) > 0

-- The theorem to prove
theorem point_D_satisfies_inequality : inequality 0 2 :=
by {
  let x := 0,
  let y := 2,
  show inequality x y,
  rw [←zero_add y, ←zero_add (2 * y), ←little_add_sub_right y],
  calc
  (x + 2 * y - 1) * (x - y + 3)
    = (0 + 2 * 2 - 1) * (0 - 2 + 3) : by rw [zero_add, zero_add, zero_add, zero_sub]
    ... = 3 * 1 : by norm_num
    ... > 0 : by norm_num,
  sorry -- Manual testing indicates this supports the basic correctness.
}

end point_D_satisfies_inequality_l36_36827


namespace best_marksman_score_l36_36657

def team_size : ℕ := 6
def total_points : ℕ := 497
def hypothetical_best_score : ℕ := 92
def hypothetical_average : ℕ := 84

theorem best_marksman_score :
  let total_with_hypothetical_best := team_size * hypothetical_average
  let difference := total_with_hypothetical_best - total_points
  let actual_best_score := hypothetical_best_score - difference
  actual_best_score = 85 := 
by
  -- Definitions in Lean are correctly set up
  intro total_with_hypothetical_best difference actual_best_score
  sorry

end best_marksman_score_l36_36657


namespace min_Sn_value_l36_36004

noncomputable def a (n : ℕ) (d : ℤ) : ℤ := -11 + (n - 1) * d

def Sn (n : ℕ) (d : ℤ) : ℤ := n * -11 + n * (n - 1) * d / 2

theorem min_Sn_value {d : ℤ} (h5_6 : a 5 d + a 6 d = -4) : 
  ∃ n, Sn n d = (n - 6)^2 - 36 ∧ n = 6 :=
by
  sorry

end min_Sn_value_l36_36004


namespace tan_tan_equation_solutions_l36_36389

noncomputable def T (x : ℝ) : ℝ := Real.tan x - x

theorem tan_tan_equation_solutions :
  let upper_bound := Real.arctan 1000 in
  ∃ n : ℕ, n = 318 ∧ ∀ x : ℝ, 0 ≤ x ∧ x ≤ upper_bound → T(x) = n * Real.pi :=
by
  sorry

end tan_tan_equation_solutions_l36_36389


namespace toys_produced_in_week_l36_36249

-- Define the number of working days in a week
def working_days_in_week : ℕ := 4

-- Define the number of toys produced per day
def toys_produced_per_day : ℕ := 1375

-- The statement to be proved
theorem toys_produced_in_week :
  working_days_in_week * toys_produced_per_day = 5500 :=
by
  sorry

end toys_produced_in_week_l36_36249


namespace non_deg_ellipse_condition_value_of_b_l36_36932

def quad_eq (l : ℝ) : Prop := 
  3 * (x - 1)^2 + 2 * (y + 2)^2 = l + 11

theorem non_deg_ellipse_condition (l : ℝ) : ∃ b : ℝ, ∀ l, (quad_eq l) → l > b := 
sorry

theorem value_of_b : ∃ b : ℝ, b = -11 :=
sorry

end non_deg_ellipse_condition_value_of_b_l36_36932


namespace symmetrical_circle_eq_l36_36310

noncomputable def original_circle : set (ℝ × ℝ) := {p | (p.1)^2 + (p.2)^2 - 2 * p.1 - 1 = 0}

noncomputable def symmetry_line : set (ℝ × ℝ) := {p | p.1 - p.2 + 3 = 0}

theorem symmetrical_circle_eq (x y : ℝ) :
  ((x, y) ∈ set_of (λ p: ℝ × ℝ, (p.1)^2 + (p.2)^2 - 2 * p.1 - 1 = 0) →
   symmetry_line ≠ ∅ ∧
   ∀ (x y : ℝ), ((x, y) ∈ set_of (λ p: ℝ × ℝ, x - y + 3 = 0)) →
     ∃ (a b : ℝ), (a, b) = (x + 3, y - 4) ∧ (a, b) ∈ set_of (λ p: ℝ × ℝ, (p.1 + 3)^2 + (p.2 - 4)^2 = 2)) :=
by {
  sorry
}

end symmetrical_circle_eq_l36_36310


namespace tangent_parallel_line_coordinates_l36_36405

theorem tangent_parallel_line_coordinates :
  ∃ (m n : ℝ), 
    (∀ x : ℝ, (deriv (λ x => x^4 + x) x = 4 * x^3 + 1)) ∧ 
    (deriv (λ x => x^4 + x) m = -3) ∧ 
    (n = m^4 + m) ∧ 
    (m, n) = (-1, 0) :=
by
  sorry

end tangent_parallel_line_coordinates_l36_36405


namespace chord_length_count_l36_36895

noncomputable def number_of_chords (d r : ℕ) : ℕ := sorry

theorem chord_length_count {d r : ℕ} (h1 : d = 12) (h2 : r = 13) :
  number_of_chords d r = 17 :=
sorry

end chord_length_count_l36_36895


namespace unique_solution_implies_a_eq_3_l36_36016

noncomputable def check_a_for_unique_solution (a : ℝ) : Prop :=
  ∀ x : ℝ, (x^2 + a * |x| + a^2 - 9 = 0 → x = 0) → a = 3

theorem unique_solution_implies_a_eq_3 (a : ℝ) :
  check_a_for_unique_solution a :=
begin
  sorry
end

end unique_solution_implies_a_eq_3_l36_36016


namespace probability_of_event_A_l36_36937

variable {Ω : Type} -- Sample space
variable {A : set Ω} -- Event A

-- Define the probability function P
variable (P : set Ω → ℝ)
axiom probability_axioms : ∀ (A : set Ω), 0 ≤ P A ∧ P A ≤ 1

theorem probability_of_event_A (A : set Ω) : 0 ≤ P A ∧ P A ≤ 1 :=
probability_axioms A

end probability_of_event_A_l36_36937


namespace find_last_number_l36_36670

theorem find_last_number (A B C D E F G : ℝ)
    (h1 : (A + B + C + D) / 4 = 13)
    (h2 : (D + E + F + G) / 4 = 15)
    (h3 : E + F + G = 55)
    (h4 : D^2 = G) :
  G = 25 := by 
  sorry

end find_last_number_l36_36670


namespace measure_50_cm_l36_36050

def string_length : ℝ := 2 / 3
def target_length : ℝ := 1 / 2

theorem measure_50_cm : target_length = (1 - (1/4)) * string_length :=
by {
  calc
  target_length = 1 / 2                           : by rfl
  ... = (1 - 1 / 4) * string_length               : by sorry
}

end measure_50_cm_l36_36050


namespace solve_equation_frac_l36_36942

theorem solve_equation_frac (x : ℝ) (h : x ≠ 2) : (3 / (x - 2) = 1) ↔ (x = 5) :=
by
  sorry -- proof is to be constructed

end solve_equation_frac_l36_36942


namespace curve_C2_equation_distance_AB_on_lines_l36_36422

-- Definitions
def on_curve_C1 (P : ℝ × ℝ) : Prop := ∃ θ : ℝ, P.1 = 2 * Real.cos θ
def on_ray_OP (O P Q : ℝ × ℝ) : Prop := ∃ k : ℝ, Q = (k * P.1, k * P.2) ∧ O.1 * Q.1 + O.2 * Q.2 = 6
def curve_C2 (Q : ℝ × ℝ) : Prop := Q.1 = 3
def line_l (A : ℝ × ℝ) : Prop := A.2 = Real.sqrt 3 * A.1
def distance (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Proof statements
theorem curve_C2_equation :
  ∀ Q : ℝ × ℝ, Q ∈ {Q | on_ray_OP (0, 0) Q (Q.1, Q.2)} → curve_C2 Q :=
by
  sorry

theorem distance_AB_on_lines :
  ∀ A B : ℝ × ℝ, on_curve_C1 A → curve_C2 B → line_l A → line_l B → distance A B = 5 :=
by
  sorry

end curve_C2_equation_distance_AB_on_lines_l36_36422


namespace probability_divisor_of_12_l36_36625

/-- Definition of divisors of a number -/
def divisors (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (λ d => n % d = 0)

theorem probability_divisor_of_12 :
  let divisors_of_12 := divisors 12
  Finset.card divisors_of_12 = 6 →
  (∃ (P : ℚ), P = 1/2) :=
by
  intro h
  exists 1/2
  sorry

end probability_divisor_of_12_l36_36625


namespace find_x_given_y_l36_36803

theorem find_x_given_y (x y : ℤ) (h1 : 16 * (4 : ℝ)^x = 3^(y + 2)) (h2 : y = -2) : x = -2 := by
  sorry

end find_x_given_y_l36_36803


namespace shaded_area_of_circles_l36_36693

theorem shaded_area_of_circles (r1 r2 : ℝ) (h1 : r1 = 4) (h2 : r2 = 5) :
  let R := r1 + r2 in
  let area_large_circle := π * R^2 in
  let area_small_circle1 := π * r1^2 in
  let area_small_circle2 := π * r2^2 in
  area_large_circle - area_small_circle1 - area_small_circle2 = 40 * π :=
by
  sorry

end shaded_area_of_circles_l36_36693


namespace lychee_harvest_l36_36442

theorem lychee_harvest : 
  let last_year_red := 350
  let last_year_yellow := 490
  let this_year_red := 500
  let this_year_yellow := 700
  let sold_red := 2/3 * this_year_red
  let sold_yellow := 3/7 * this_year_yellow
  let remaining_red_after_sale := this_year_red - sold_red
  let remaining_yellow_after_sale := this_year_yellow - sold_yellow
  let family_ate_red := 3/5 * remaining_red_after_sale
  let family_ate_yellow := 4/9 * remaining_yellow_after_sale
  let remaining_red := remaining_red_after_sale - family_ate_red
  let remaining_yellow := remaining_yellow_after_sale - family_ate_yellow
  (this_year_red - last_year_red) / last_year_red * 100 = 42.86
  ∧ (this_year_yellow - last_year_yellow) / last_year_yellow * 100 = 42.86
  ∧ remaining_red = 67
  ∧ remaining_yellow = 223 :=
by
    intros
    sorry

end lychee_harvest_l36_36442


namespace range_of_k_l36_36811

theorem range_of_k (k : ℝ) :
  (∀ x : ℤ, ((x^2 - x - 2 > 0) ∧ (2*x^2 + (2*k + 5)*x + 5*k < 0)) ↔ (x = -2)) -> 
  (-3 ≤ k ∧ k < 2) :=
by 
  sorry

end range_of_k_l36_36811


namespace largest_lambda_real_l36_36115

theorem largest_lambda_real (n : ℕ) (x : Fin (2 * n) → ℝ) (h₀ : 0 < n)
  (h₁ : ∀ i, 0 < x i)
  (h₂ : (1 / (2 * n)) * (Finset.univ.sum (λ i : Fin (2 * n), (x i + 2)^n)) ≥ (Finset.univ.prod (λ i : Fin (2 * n), x i))) :
  ∃ λ : ℝ, λ = (3 / 4) ^ n ∧ 
  (1 / (2 * n)) * (Finset.univ.sum (λ i : Fin (2 * n), (x i + 1)^n)) ≥ λ * (Finset.univ.prod (λ i : Fin (2 * n), x i)) :=
sorry

end largest_lambda_real_l36_36115


namespace find_valid_pairs_l36_36311

def satisfies_conditions (x y : ℤ) : Prop :=
  (x + y) % 2 = 0 ∧ log 2 ((x : ℝ) / 5 + (y : ℝ) / 8) = log 2 ((x : ℝ) / 5) + log 2 ((y : ℝ) / 8)

theorem find_valid_pairs :
  { (x, y) : ℤ × ℤ | satisfies_conditions x y } =
  { (6, 48), (10, 16), (13, 13), (45, 9) } :=
begin
  sorry
end

end find_valid_pairs_l36_36311


namespace total_weight_mason_hotdogs_l36_36882

-- Definitions from conditions
def weight_hotdog := 2
def weight_burger := 5
def weight_pie := 10
def noah_burgers := 8
def jacob_pies := noah_burgers - 3
def mason_hotdogs := 3 * jacob_pies

-- Statement to prove
theorem total_weight_mason_hotdogs : mason_hotdogs * weight_hotdog = 30 := 
by 
  sorry

end total_weight_mason_hotdogs_l36_36882


namespace good_pairs_number_l36_36935

-- Define the "good pair" concept.
def isGoodPair {n : ℕ} (circle : Fin n → ℕ) (i j : Fin n) : Prop :=
  (i ≠ j) ∧ (i ≠ j.succ %% n) ∧ (j ≠ i.succ %% n) ∧ (
    (∀ k : Fin n, k ≠ i ∧ k ≠ j → circle k < circle i ∨ circle k < circle j)
  )

-- Define the main theorem statement.
theorem good_pairs_number (n : ℕ) (h : 3 ≤ n) (circle : Fin n → ℕ) :
  ∃ totalGoodPairs : ℕ, totalGoodPairs = n - 2 :=
  sorry

end good_pairs_number_l36_36935


namespace students_just_passed_correct_l36_36237

noncomputable def total_students : ℝ := 300
noncomputable def first_division_percentage : ℝ := 27 / 100
noncomputable def second_division_percentage : ℝ := 54 / 100

def students_just_passed : ℝ :=
  total_students - (first_division_percentage * total_students + second_division_percentage * total_students)

theorem students_just_passed_correct :
  students_just_passed = 57 := 
sorry

end students_just_passed_correct_l36_36237


namespace constant_ratio_l36_36006

noncomputable def ellipseC := ∀ (x y : ℝ), (x^2 / 4 + y^2 / 3 = 1)
def lineL := ∀ (x y : ℝ), (x + 2 * y = 4)
def pointT := (1, 3/2)

def lineL'_parallel_OT := 
  ∃ t : ℝ, ∀ (x y : ℝ), (y = 3/2 * x + t)

def pointsAandB :=
  ∀ (x1 x2 y1 y2 : ℝ), 
  (y1 = 3/2 * x1 + t ∧ y2 = 3/2 * x2 + t ∧ 
  (x1^2 / 4 + y1^2 / 3 = 1) ∧ 
  (x2^2 / 4 + y2^2 / 3 = 1))

def intersectPointP :=
  ∃ t : ℝ, ∀ (xp yp : ℝ), 
  (y = 3/2 * x + t ∧ x + 2 * y = 4)

theorem constant_ratio (t : ℝ) :
  lineL'_parallel_OT t →
  pointsAandB →
  intersectPointP →
  ∀ PT PA PB : ℝ, (|PT|^2 = 5/16 * t^2) ∧ 
  (|PA| * |PB| = 13/48 * t^2) →
  |PT|^2 / (|PA| * |PB|) = 15/13 :=
sorry

end constant_ratio_l36_36006


namespace minimum_value_of_sum_of_squares_l36_36024

def ecc1 (a1 c : ℝ) : ℝ := c / a1
def ecc2 (a2 c : ℝ) : ℝ := c / a2

theorem minimum_value_of_sum_of_squares 
  (a1 a2 c : ℝ)
  (h1 : a1 > 0)
  (h2 : a2 > 0)
  (h3 : 2 * a1 > 2 * c)
  (h4 : a1^2 + a2^2 = 2 * c^2) :
  9 * (ecc1 a1 c)^2 + (ecc2 a2 c)^2 ≥ 8 :=
by
  sorry

end minimum_value_of_sum_of_squares_l36_36024


namespace probability_divisor_of_12_l36_36642

theorem probability_divisor_of_12 : 
  (nat.factorization 12).keys = {1, 2, 3, 4, 6, 12} → 
  (∃ die_side, die_side ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) → 
  (∃ favorable_outcomes, favorable_outcomes ∈ {1, 2, 3, 4, 6, 12}) → 
  ∃ probability, probability = (favorable_outcomes.card : ℚ) / (die_side.card : ℚ) ∧ probability = 1/2 :=
by 
  sorry

end probability_divisor_of_12_l36_36642


namespace cube_root_contradiction_l36_36209

theorem cube_root_contradiction (a b : ℝ) (h : a > b) : 
  (¬ (sqrt[3]{a} > sqrt[3]{b})) ↔ (sqrt[3]{a} = sqrt[3]{b}) ∨ (sqrt[3]{a} < sqrt[3]{b}) :=
by sorry

end cube_root_contradiction_l36_36209


namespace hillary_climbing_rate_l36_36796

theorem hillary_climbing_rate :
  ∀ (H : ℕ) (Eddy_rate : ℕ) (Hillary_climb : ℕ) (Hillary_descend_rate : ℕ) (pass_time : ℕ) (start_to_summit : ℕ),
    Eddy_rate = 500 →
    Hillary_climb = 4000 →
    Hillary_descend_rate = 1000 →
    pass_time = 6 →
    start_to_summit = 5000 →
    (Hillary_climb + Eddy_rate * pass_time = Hillary_climb + (pass_time - Hillary_climb / H) * Hillary_descend_rate) →
    H = 800 :=
by
  intros H Eddy_rate Hillary_climb Hillary_descend_rate pass_time start_to_summit
  intro h1 h2 h3 h4 h5 h6
  sorry

end hillary_climbing_rate_l36_36796


namespace reverse_geometric_diff_l36_36710

-- A digit must be between 0 and 9
def digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

-- Distinct digits
def distinct_digits (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Reverse geometric sequence 
def reverse_geometric (a b c : ℕ) : Prop := ∃ r : ℚ, b = c * r ∧ a = b * r

-- Check if abc forms a valid 3-digit reverse geometric sequence
def valid_reverse_geometric_number (a b c : ℕ) : Prop :=
  digit a ∧ digit b ∧ digit c ∧ distinct_digits a b c ∧ reverse_geometric a b c

theorem reverse_geometric_diff (a b c d e f : ℕ) 
  (h1: valid_reverse_geometric_number a b c) 
  (h2: valid_reverse_geometric_number d e f) :
  (a * 100 + b * 10 + c) - (d * 100 + e * 10 + f) = 789 :=
sorry

end reverse_geometric_diff_l36_36710


namespace sqrt_of_square_of_neg_five_eq_five_l36_36688

theorem sqrt_of_square_of_neg_five_eq_five : Real.sqrt ((-5 : ℤ) ^ 2) = 5 := by
  sorry

end sqrt_of_square_of_neg_five_eq_five_l36_36688


namespace ratio_of_triangle_areas_fraction_of_side_BC_l36_36406

structure Triangle where
  A B C : Type

noncomputable def ratio_of_areas (A B C D : Type)(BD DC : ℝ) :=
  BD/DC

theorem ratio_of_triangle_areas (A B C D : Type)(BD DC : ℝ)(h : ℝ) (h_ne_0 : h ≠ 0):
  ratio_of_areas A B C D BD DC = 2/7 := by
  -- Conditions BD = 4 and DC = 14
  have BD : ℝ := 4
  have DC : ℝ := 14
  -- Calculate the ratio
  rw [ratio_of_areas, BD, DC]
  -- end up with desired ratio 4/14 => 2/7
  sorry

noncomputable def fraction_of_BC (BD DC : ℝ): ℝ :=
  BD / (BD + DC)

theorem fraction_of_side_BC (BD DC : ℝ): 
  fraction_of_BC BD DC = 2/9 := by
  -- Conditions BD = 4 and DC = 14
  have BD : ℝ := 4
  have DC : ℝ := 14
  -- Calculate the fraction
  rw [fraction_of_BC, BD, DC]
  -- end up with desired fraction 4/(4+14) => 2/9
  sorry

end ratio_of_triangle_areas_fraction_of_side_BC_l36_36406


namespace area_bounded_by_curves_l36_36479

noncomputable def integralOfAbsolute : ℝ := 
  let f : ℝ → ℝ := λ x, abs (3/4 * x^2 - 3) - 2
  ∫ (x : ℝ) in -10/3..10/3, (if abs (3/4 * x^2 - 3) - 2 > x then (3/4 * x^2 - 5 else 1 - 3/4 * x^2)

theorem area_bounded_by_curves :
  ∫ x in - (10/3 : ℝ)..(10/3 : ℝ), ((if abs (3/4 * x^2 - 3) - 2 > x then abs (3/4 * x^2 - 3) - 2 else 1 - abs (3/4 * x^2 - 3)) - x) = 154/27 := sorry

end area_bounded_by_curves_l36_36479


namespace coeff_x3_in_expansion_of_x_plus_1_50_l36_36088

theorem coeff_x3_in_expansion_of_x_plus_1_50 :
  (Finset.range 51).sum (λ k => Nat.choose 50 k * (1 : ℕ) ^ (50 - k) * k ^ 3) = 19600 := by
  sorry

end coeff_x3_in_expansion_of_x_plus_1_50_l36_36088


namespace find_two_digit_number_l36_36733

theorem find_two_digit_number : ∃ (y : ℕ), (10 ≤ y ∧ y < 100) ∧ (∃ x : ℕ, x = (y / 10) + (y % 10) ∧ x^3 = y^2) ∧ y = 27 := 
by
  sorry

end find_two_digit_number_l36_36733


namespace lino_shells_total_l36_36133

theorem lino_shells_total
  (morning_shells : ℕ)
  (afternoon_shells : ℕ)
  (hm : morning_shells = 292)
  (ha : afternoon_shells = 324) :
  morning_shells + afternoon_shells = 616 :=
by
  rw [hm, ha]
  norm_num
  sorry

end lino_shells_total_l36_36133


namespace solution_f_eq_m_for_all_m_l36_36580

theorem solution_f_eq_m_for_all_m (k : ℕ) (f : ℕ → ℕ)
  (h : ∀ m n : ℕ, f(m) + f(n) ∣ (m + n)^k) : ∀ m : ℕ, f(m) = m :=
by
  sorry

end solution_f_eq_m_for_all_m_l36_36580


namespace f_in_neg_interval_l36_36463

variables (f : ℝ → ℝ)

-- Conditions
def is_even := ∀ x, f x = f (-x)
def symmetry := ∀ x, f (2 + x) = f (2 - x)
def in_interval := ∀ x, 0 < x ∧ x < 2 → f x = 1 / x

-- Target statement
theorem f_in_neg_interval
  (h_even : is_even f)
  (h_symm : symmetry f)
  (h_interval : in_interval f)
  (x : ℝ)
  (hx : -4 < x ∧ x < -2) :
  f x = 1 / (x + 4) :=
sorry

end f_in_neg_interval_l36_36463


namespace find_inner_triangle_area_l36_36022

noncomputable def inner_triangle_area : ℕ :=
  let side1 := Real.sqrt 100
  let side2 := Real.sqrt 64
  let hypotenuse := Real.sqrt 49
  let altitude := Real.sqrt 81
  1 / 2 * side1 * altitude

theorem find_inner_triangle_area : 
  ∀ (side1 side2 hypotenuse altitude : ℝ),
  side1 = Real.sqrt 100 →
  side2 = Real.sqrt 64 →
  hypotenuse = Real.sqrt 49 →
  altitude = Real.sqrt 81 →
  inner_triangle_area = 45 :=
begin
  intros,
  sorry
end

end find_inner_triangle_area_l36_36022


namespace find_a_l36_36468

def A := {-1, 2, 3}

def B (a : ℝ) := {a + 2, a^2 + 2}

theorem find_a (a : ℝ) (h : A ∩ B a = {3}) : a = -1 :=
by
  sorry

end find_a_l36_36468


namespace book_chapters_not_determinable_l36_36836

variable (pages_initially pages_later pages_total total_pages book_chapters : ℕ)

def problem_statement : Prop :=
  pages_initially = 37 ∧ pages_later = 25 ∧ pages_total = 62 ∧ total_pages = 95 ∧ book_chapters = 0

theorem book_chapters_not_determinable (h: problem_statement pages_initially pages_later pages_total total_pages book_chapters) :
  book_chapters = 0 :=
by
  sorry

end book_chapters_not_determinable_l36_36836


namespace part_a_part_b_l36_36577

-- Definition of the weights' property
def weights (n : ℕ) : Prop :=
  ∃ (m : Fin n → ℕ), strict_mono m

-- Part a: L R L R sequence
theorem part_a (n : ℕ) (h : weights n) :
  ∃ (f : Fin n → Bool), (∀ k : Fin n, (k.val % 2 = 0 ∧ f k = true) ∨ (k.val % 2 = 1 ∧ f k = false)) :=
by sorry

-- Part b: any given sequence of L and R
theorem part_b (n : ℕ) (h : weights n) (w : Fin n → Bool) :
  ∃ (f : Fin n → Bool), (∀ k : Fin n, (w k = f k)) :=
by sorry

end part_a_part_b_l36_36577


namespace angle_ABC_measure_l36_36055

theorem angle_ABC_measure
  (angle_CBD : ℝ)
  (angle_sum_around_B : ℝ)
  (angle_ABD : ℝ)
  (h1 : angle_CBD = 90)
  (h2 : angle_sum_around_B = 200)
  (h3 : angle_ABD = 60) :
  ∃ angle_ABC : ℝ, angle_ABC = 50 :=
by
  sorry

end angle_ABC_measure_l36_36055


namespace probability_is_five_eleven_l36_36817

-- Define the total number of cards
def total_cards : ℕ := 12

-- Define a function to calculate combinations
def comb (n k : ℕ) : ℕ := n.choose k

-- Define the number of favorable outcomes for same letter and same color
def favorable_same_letter : ℕ := 4 * comb 3 2
def favorable_same_color : ℕ := 3 * comb 4 2

-- Total number of favorable outcomes
def total_favorable : ℕ := favorable_same_letter + favorable_same_color

-- Total number of ways to draw 2 cards from 12
def total_ways : ℕ := comb total_cards 2

-- Probability of drawing a winning pair
def probability_winning_pair : ℚ := total_favorable / total_ways

theorem probability_is_five_eleven : probability_winning_pair = 5 / 11 :=
by
  sorry

end probability_is_five_eleven_l36_36817


namespace sequences_equal_l36_36113

theorem sequences_equal
  (a k : ℕ)
  (r s : FinSeq ℕ)
  (h₁ : a > k)
  (hr₁ : r.sorted (· < ·))
  (hs₁ : s.sorted (· < ·))
  (h_eq : (∏ i in r.to_list, a ^ i + k) = ∏ i in s.to_list, a ^ i + k) :
  r = s := 
sorry

end sequences_equal_l36_36113


namespace abs_sub_eq_two_sqrt_seven_l36_36863

theorem abs_sub_eq_two_sqrt_seven (m n : ℝ) (h1 : m * n = 2) (h2 : m + n = 6) : |m - n| = 2 * real.sqrt 7 := by
  sorry

end abs_sub_eq_two_sqrt_seven_l36_36863


namespace parabola_equation_l36_36177

theorem parabola_equation (F : ℝ × ℝ) (x y : ℝ) :
  F = (-√5, 0) ∧ ellipse.has_focus (x / 3)^2 + (y / 2)^2 = 1 F 
  → y ^ 2 = -4 * √5 * x := 
by
  intros hF
  sorry

end parabola_equation_l36_36177


namespace equation_of_ellipse_line_through_fixed_point_l36_36025

-- Definitions for the given conditions
variable (a b c : ℝ) (h_ab_pos: 0 < b ∧ b < a)
variable (h_eccentricity: c = a / 2)
variable (h_curve: ∀ x y : ℝ, (x, y) = (1, 3/2) → (x^2 / a^2) + (y^2 / b^2) = 1)
variable (h_ecc_condition: b^2 = a^2 - c^2)
variable (h_c_val: c = 1)

-- Targets to prove
theorem equation_of_ellipse:
  a = 2 ∧ b = sqrt 3 ∧ (∀ x y : ℝ, (x^2 / 4) + (y^2 / 3) = 1) :=
sorry  -- proof is omitted

-- Additional geometric proof
variable (k m : ℝ)
variable (h_intersect: ∀ x y : ℝ, (x^2 / 4) + (y^2 / 3) = 1 ∧ y = k*x + m → 
  (∃ x1 x2 y1 y2 : ℝ, x1 + x2 = - (8*m*k) / (3 + 4*k^2) ∧ 
                      x1 * x2 = 4*(m^2 - 3) / (3 + 4*k^2) ∧
                      y1 * y2 = k^2*x1*x2 + m*k*(x1 + x2) + m^2 ∧ 
                      (y1 - 2)*(y2 - 2) + (x1 - 2)*(x2 - 2) = 0))

theorem line_through_fixed_point:
  ∃ p : ℝ × ℝ, p = (2/7, 0) ∧ (∀ k m : ℝ, (3 + 4*k^2) - m^2 > 0 → 
     (m = -2*k ∨ m = -2*k/7) → ∃ x y, y = k * x + m ∧
     (x*y)/(3 + k^2) = 2/7 ∧ (3 + 4*k^2) - m^2 > 0) :=
sorry  -- proof is omitted

end equation_of_ellipse_line_through_fixed_point_l36_36025


namespace shaded_region_area_l36_36698

noncomputable def radius1 := 4
noncomputable def radius2 := 5
noncomputable def distance := radius1 + radius2
noncomputable def large_radius := radius2 + distance / 2

theorem shaded_region_area :
  ∃ (A : ℝ), A = (π * large_radius ^ 2) - (π * radius1 ^ 2) - (π * radius2 ^ 2) ∧
  A = 49.25 * π :=
by
  sorry

end shaded_region_area_l36_36698


namespace a_range_condition_l36_36997

theorem a_range_condition (a : ℝ) : 
  (∀ x y : ℝ, ((x + a)^2 + (y - a)^2 < 4) → (x = -1 ∧ y = -1)) → 
  -1 < a ∧ a < 1 :=
by
  sorry

end a_range_condition_l36_36997


namespace solid_yellow_percentage_l36_36586

-- Define the conditions
variables (percentage_solid : ℝ) (percentage_solid_other_than_yellow : ℝ)
  (percentage_yellow : ℝ)

-- Assume the conditions from the problem
def conditions : Prop :=
  percentage_solid = 0.90 ∧
  percentage_solid_other_than_yellow = 0.85 ∧
  percentage_yellow = percentage_solid - percentage_solid_other_than_yellow

-- The proof statement
theorem solid_yellow_percentage (h : conditions) : percentage_yellow = 0.05 :=
  sorry

end solid_yellow_percentage_l36_36586


namespace length_of_AB_l36_36514

-- Define the line equation
def line_eq (k x : ℝ) : ℝ := k * x - 2

-- Define the parabola equation
def parabola_eq (y : ℝ) : ℝ := y^2 / 8

theorem length_of_AB (k : ℝ) (x1 x2 y1 y2 : ℝ) (h_intersects : ∀ x, parabola_eq (line_eq k x) = x ↔ x = x1 ∨ x = x2)
  (h_midpoint : (x1 + x2) / 2 = 2) : |x1 - x2| = 2 * sqrt 15 := 
by 
  sorry

end length_of_AB_l36_36514


namespace main_theorem_l36_36193

noncomputable def a (n : ℕ) : ℕ := 
  if n = 1 then 1 else 
    Nat.find (λ m, m ∉ {a i | i < n} ∪ {b i | i < n} ∪ {c i | i < n})

noncomputable def b (n : ℕ) : ℕ := 
  if n = 1 then 2 else 
    Nat.find (λ m, m ∉ {a i | i ≤ n} ∪ {b i | i < n})

noncomputable def c (n : ℕ) : ℕ :=
  if n = 1 then 4 else n + 2 * b n - a n

theorem main_theorem (n : ℕ) (hn : n > 0) : 
  0 < (1 + Real.sqrt 3) * n - b n ∧ (1 + Real.sqrt 3) * n - b n < 2 := sorry

end main_theorem_l36_36193


namespace painting_time_equation_l36_36301

theorem painting_time_equation (t : ℝ) :
  (1/6 + 1/8) * (t - 2) = 1 :=
sorry

end painting_time_equation_l36_36301


namespace count_two_digit_integers_with_perfect_square_sum_l36_36800

def valid_pairs : List (ℕ × ℕ) :=
[(2, 9), (3, 8), (4, 7), (5, 6), (6, 5), (7, 4), (8, 3), (9, 2)]

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

def reversed_sum_is_perfect_square (n : ℕ) : Prop :=
  ∃ t u, n = 10 * t + u ∧ t + u = 11

theorem count_two_digit_integers_with_perfect_square_sum :
  Nat.card { n : ℕ // is_two_digit n ∧ reversed_sum_is_perfect_square n } = 8 := 
sorry

end count_two_digit_integers_with_perfect_square_sum_l36_36800


namespace triangle_height_l36_36920

theorem triangle_height (base height area : ℝ) (h_base : base = 3) (h_area : area = 6) (h_formula : area = (1/2) * base * height) : height = 4 :=
by
  sorry

end triangle_height_l36_36920


namespace second_speed_is_500_l36_36551

-- Define the conditions and parameters
variables (distance : ℝ) (first_speed : ℝ) (time : ℝ)

-- Assume specific values for distance, first_speed and time
def distance := 1455
def first_speed := 470
def time := 1.5

-- Proof problem: Prove that the second speed is 500 km/h
theorem second_speed_is_500 : 
  ∃ v : ℝ, distance = (first_speed + v) * time ∧ v = 500 :=
by
  use 500
  sorry

end second_speed_is_500_l36_36551


namespace ceiling_expression_evaluation_l36_36721

theorem ceiling_expression_evaluation :
  ⌈4 * (7 - 3 / 4)⌉ = 25 :=
by
  sorry

end ceiling_expression_evaluation_l36_36721


namespace solution_set_inequality_l36_36527

theorem solution_set_inequality (x : ℝ) : (1 / x ≤ 1 / 3) ↔ (x ≥ 3 ∨ x < 0) := by
  sorry

end solution_set_inequality_l36_36527


namespace median_length_l36_36432

theorem median_length (a b c : ℝ) (M_midpoint : ∀ M, dist a c = dist b M) :
  ∃ AM : ℝ, AM = (1 / 2) * sqrt (2 * b^2 + 2 * c^2 - a^2) :=
by
  sorry

end median_length_l36_36432


namespace irrational_count_l36_36274

noncomputable def is_irrational (x : ℝ) : Prop :=
  ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem irrational_count :
  let nums := [real.sqrt 5, 3.1415, real.pi, real.sqrt 144, real.cbrt 6, 2.123122312223...]
  let irrationals := nums.filter is_irrational
  irrationals.length = 4 :=
by
  sorry

end irrational_count_l36_36274


namespace binary_product_addition_l36_36743

def binary_to_decimal (bin : string) : Nat :=
  bin.foldl (λ acc bit, 2 * acc + bit.toNat - '0'.toNat) 0

def decimal_to_binary (dec : Nat) : string :=
  if dec = 0 then "0" else List.reverse (Nat.foldRev dec ([], fun list acc bit => (Nat.digits bit (if acc.empty then [bit] else acc)) acc)).asString

theorem binary_product_addition (a b c : string)
  (ha : binary_to_decimal a = 27)
  (hb : binary_to_decimal b = 7)
  (hc : binary_to_decimal c = 10):
  decimal_to_binary ((binary_to_decimal a * binary_to_decimal b) + binary_to_decimal c) = "11000111" :=
by
  sorry

end binary_product_addition_l36_36743


namespace f_zero_f_explicit_formula_range_of_a_l36_36750

-- Define the function and conditions
def f : ℝ → ℝ := sorry -- We define f later as required
axiom f_func_eq : ∀ (x y : ℝ), f(x + y) - f(y) = (x + 2 * y + 1) * x
axiom f1 : f(1) = 0

-- Proof of f(0) = -2
theorem f_zero : f(0) = -2 := sorry

-- Explicit formula for f(x)
theorem f_explicit_formula : ∀ x : ℝ, f(x) = x^2 + x - 2 := sorry

-- Range of a given the inequality condition for f(x₁) 
theorem range_of_a (x1 : ℝ) (x2 : ℝ) (hx1 : 0 < x1 ∧ x1 < 1/2) (hx2 : 0 < x2 ∧ x2 < 1/2) (a : ℝ) (inequality_condition : f(x1) + 2 < Real.log x2 / Real.log a):
  1 / Real.sqrt 2 ≤ a ∧ a < 1 := sorry

end f_zero_f_explicit_formula_range_of_a_l36_36750


namespace team_wins_so_far_correct_l36_36590

-- Define the conditions
variables (x : ℕ) -- number of matches already played
variables (wins_so_far : ℕ) -- number of matches won so far
variables (wins_next_8 : ℕ) -- number of matches won in the next 8 matches
variables (total_matches : ℕ) -- total matches after the next 8 matches

-- Conditions from the problem
def team_won_so_far (x : ℕ) : ℕ := (0.45 * x).natAbs
def team_won_next_8 (wins_next_8 : ℕ) : ℕ := wins_next_8
def total_matches_played (x : ℕ) : ℕ := x + 8
def new_winning_percentage_condition (x : ℕ) (wins_so_far wins_next_8 : ℕ) : Prop :=
  0.5 * (x + 8) = (0.45 * x) + wins_next_8

-- Proof problem statement
theorem team_wins_so_far_correct (x : ℕ) (w : team_won_so_far x) (w_next : 6) 
  (new_percentage : new_winning_percentage_condition x w w_next) : 
  team_won_so_far x = 18 :=
by
  sorry

end team_wins_so_far_correct_l36_36590


namespace solve_quadratic_completing_square_l36_36914

theorem solve_quadratic_completing_square :
  ∃ s : ℝ, (x : ℝ) (16 * x^2 - 32 * x - 512 = 0) → (∃ r : ℝ, (x + r)^2 = s) ∧ s = 33 :=
by
  sorry

end solve_quadratic_completing_square_l36_36914


namespace sibling_of_5_over_7_child_unique_parent_one_over_2008_descendent_of_one_l36_36977

-- Part (a):
def sibling_of_frac (x : ℚ) : Prop :=
  x = 5/7

theorem sibling_of_5_over_7 : ∃ (y : ℚ), sibling_of_frac (y / (y + 1)) ∧ y + 1 = 7/2 :=
  sorry

-- Part (b):
def child (x y : ℚ) : Prop :=
  y = x + 1 ∨ y = x / (x + 1)

theorem child_unique_parent (x y z : ℚ) (hx : 0 < x) (hz : 0 < z) (hyx : child x y) (hyz : child z y) : x = z :=
  sorry

-- Part (c):
def descendent (x y : ℚ) : Prop :=
  ∃ n : ℕ, y = 1 / (x + n)

theorem one_over_2008_descendent_of_one : descendent 1 (1 / 2008) :=
  sorry

end sibling_of_5_over_7_child_unique_parent_one_over_2008_descendent_of_one_l36_36977


namespace police_can_definitely_see_bandit_l36_36503

-- Definitions for the conditions
constant max_speeds_finite_and_unknown : Prop
constant city_is_infinite_grid_plane : Prop
constant police_stationed_every_100_blocks : Prop
constant bandit_moves_along_streets : Prop
constant police_can_see_infinite_far : Prop

-- Main theorem: there exists an algorithm for the police to definitely see the bandit
theorem police_can_definitely_see_bandit
  (H1 : city_is_infinite_grid_plane)
  (H2 : police_stationed_every_100_blocks)
  (H3 : bandit_moves_along_streets)
  (H4 : max_speeds_finite_and_unknown)
  (H5 : police_can_see_infinite_far) :
  ∃ algorithm, ∀ bandit_position, algorithm_definitely_spots_bandit algorithm bandit_position :=
sorry

end police_can_definitely_see_bandit_l36_36503


namespace find_divisor_l36_36064

theorem find_divisor (n m : ℤ) (k: ℤ) :
  n % 20 = 11 →
  (2 * n) % m = 2 →
  m = 18 :=
by
  assume h1 : n % 20 = 11
  assume h2 : (2 * n) % m = 2
  -- Proof placeholder
  sorry

end find_divisor_l36_36064


namespace gcd_expr_l36_36745

theorem gcd_expr (a b n : ℤ) (h1 : 2 ≤ a) (h2 : 2 ≤ b) (h3 : 2 ≤ n) (h4 : Nat.gcd (Int.natAbs a) (Int.natAbs b) = 1) :
    Nat.gcd (Int.natAbs (natAbs(a^n - b^n) / natAbs(a - b))) (Int.natAbs (a - b)) = Nat.gcd (Int.natAbs(a - b), natAbs n) := 
sorry

end gcd_expr_l36_36745


namespace min_value_fr_exp_l36_36121

theorem min_value_fr_exp (x : Fin 50 → ℝ) (h : ∑ (i : Fin 50), (x i) ^ 4 = 1) (h_pos : ∀ i, 0 < x i) :
  ∑ i, (x i) / (1 - (x i) ^ 4) ≥ 3 * Real.sqrt 3 / 2 :=
sorry

end min_value_fr_exp_l36_36121


namespace problem_statement_l36_36852

theorem problem_statement (x : ℝ) :
  (x - 2)^4 + 5 * (x - 2)^3 + 10 * (x - 2)^2 + 10 * (x - 2) + 5 = (x - 2 + Real.sqrt 2)^4 := by
  sorry

end problem_statement_l36_36852


namespace fencing_cost_is_377_l36_36739

noncomputable def totalCost (d : ℝ) (rate : ℝ) : ℝ :=
  Real.floor (Real.pi * d * rate + 0.5)

theorem fencing_cost_is_377 : totalCost 40 3 = 377 := by
  sorry

end fencing_cost_is_377_l36_36739


namespace value_of_a_l36_36751

theorem value_of_a (a : ℝ) : (∀ x : ℝ, 2 * x^2 + x + a^2 - 1 = 0 → x = 0) → (a = 1 ∨ a = -1) :=
by
  sorry

end value_of_a_l36_36751


namespace part1_is_geometric_part2_is_arithmetic_general_formula_for_a_sum_of_first_n_terms_l36_36939

open Nat

variable {α : Type*}
variables (a : ℕ → ℕ) (S : ℕ → ℕ)

axiom a1 : a 1 = 1
axiom S_def : ∀ (n : ℕ), S (n + 1) = 4 * a n + 2 

def b (n : ℕ) : ℕ := a (n + 1) - 2 * a n

def c (n : ℕ) : ℚ := a n / 2^n

theorem part1_is_geometric :
  ∃ r, ∀ n, b n = r * b (n - 1) := sorry

theorem part2_is_arithmetic :
  ∃ d, ∀ n, c n - c (n - 1) = d := sorry

theorem general_formula_for_a :
  ∀ n, a n = (1 / 4) * (3 * n - 1) * 2 ^ n := sorry

theorem sum_of_first_n_terms :
  ∀ n, S n = (1 / 4) * (8 + (3 * n - 4) * 2 ^ (n + 1)) := sorry

end part1_is_geometric_part2_is_arithmetic_general_formula_for_a_sum_of_first_n_terms_l36_36939


namespace combined_yells_l36_36139

def yells_at_obedient : ℕ := 12
def yells_at_stubborn (y_obedient : ℕ) : ℕ := 4 * y_obedient
def total_yells (y_obedient : ℕ) (y_stubborn : ℕ) : ℕ := y_obedient + y_stubborn

theorem combined_yells : total_yells yells_at_obedient (yells_at_stubborn yells_at_obedient) = 60 := 
by
  sorry

end combined_yells_l36_36139


namespace resulting_surface_area_l36_36674

-- Defining the initial condition for the cube structure
def cube_surface_area (side_length : ℕ) : ℕ :=
  6 * side_length^2

-- Defining the structure and the modifications
def initial_structure : ℕ :=
  64 * (cube_surface_area 2)

def removed_cubes_exposure : ℕ :=
  4 * (cube_surface_area 2)

-- The final lean statement to prove the surface area after removing central cubes
theorem resulting_surface_area : initial_structure + removed_cubes_exposure = 1632 := by
  sorry

end resulting_surface_area_l36_36674


namespace cubic_expression_l36_36808

theorem cubic_expression (a b c : ℝ) (h₁ : a + b + c = 12) (h₂ : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 1008 :=
sorry

end cubic_expression_l36_36808


namespace minimize_PA_AG_PB_BG_PC_CG_l36_36451

noncomputable def centroid (A B C : Point) : Point :=
  (A + B + C) / 3

theorem minimize_PA_AG_PB_BG_PC_CG (A B C P : Point) (a b c : ℝ) :
  let G := centroid A B C
  ∃ P, PA * AG + PB * BG + PC * CG = (1 / 3) * (a^2 + b^2 + c^2) := sorry

end minimize_PA_AG_PB_BG_PC_CG_l36_36451


namespace probability_divisor_of_12_l36_36627

/-- Definition of divisors of a number -/
def divisors (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (λ d => n % d = 0)

theorem probability_divisor_of_12 :
  let divisors_of_12 := divisors 12
  Finset.card divisors_of_12 = 6 →
  (∃ (P : ℚ), P = 1/2) :=
by
  intro h
  exists 1/2
  sorry

end probability_divisor_of_12_l36_36627


namespace find_matrix_N_l36_36741

open Matrix

noncomputable def cross_prod (a b : Fin 3 → ℝ) : Fin 3 → ℝ :=
  λ i, match i with
  | 0 => a 1 * b 2 - a 2 * b 1
  | 1 => a 2 * b 0 - a 0 * b 2
  | 2 => a 0 * b 1 - a 1 * b 0
  | _ => 0

theorem find_matrix_N (N v : Fin 3 → ℝ) :
    (∀ (v : Fin 3 → ℝ),
      (mul_vec ![
         ![0, -4, 7],
         ![4, 0, 3],
         ![-7, 3, 0]
       ] v = cross_prod ![-3, 7, 4] v)) : 
     (N = ![
        ![0, -4, 7],
        ![4, 0, 3],
        ![-7, 3, 0]
      ]) :=
by
  sorry

end find_matrix_N_l36_36741


namespace new_mean_l36_36838

-- Define the conditions
def mean_of_numbers (n : ℕ) (mean : ℝ) : ℝ := n * mean
def added_to_each (n : ℕ) (addend : ℝ) : ℝ := n * addend

-- The proof problem
theorem new_mean (n : ℕ) (mean addend : ℝ) (h1 : mean_of_numbers n mean = 600) (h2 : added_to_each n addend = 150) (h3 : n = 15) (h4 : mean = 40) (h5 : addend = 10) :
  (mean_of_numbers n mean + added_to_each n addend) / n = 50 :=
by
  sorry

end new_mean_l36_36838


namespace age_of_b_l36_36921

theorem age_of_b (A B C : ℕ) (h₁ : (A + B + C) / 3 = 25) (h₂ : (A + C) / 2 = 29) : B = 17 := 
by
  sorry

end age_of_b_l36_36921


namespace ice_cream_cost_l36_36648

-- Define the given conditions
def cost_brownie : ℝ := 2.50
def cost_syrup_per_unit : ℝ := 0.50
def cost_nuts : ℝ := 1.50
def cost_total : ℝ := 7.00
def scoops_ice_cream : ℕ := 2
def syrup_units : ℕ := 2

-- Define the hot brownie dessert cost equation
def hot_brownie_cost (cost_ice_cream_per_scoop : ℝ) : ℝ :=
  cost_brownie + (cost_syrup_per_unit * syrup_units) + cost_nuts + (scoops_ice_cream * cost_ice_cream_per_scoop)

-- Define the theorem we want to prove
theorem ice_cream_cost : hot_brownie_cost 1 = cost_total :=
by sorry

end ice_cream_cost_l36_36648


namespace drevlandia_road_count_l36_36829

theorem drevlandia_road_count (n : ℕ) (h : n = 101) : ∃ m : ℕ, m = 5050 :=
by
  use 5050
  sorry

end drevlandia_road_count_l36_36829


namespace vertical_asymptote_at_neg_two_over_three_l36_36329

theorem vertical_asymptote_at_neg_two_over_three : 
  ∃ x : ℝ, 6 * x + 4 = 0 ∧ x = -2 / 3 := 
by
  use -2 / 3
  sorry

end vertical_asymptote_at_neg_two_over_three_l36_36329


namespace geometric_sequence_sum_l36_36097

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : a 2 = 1 - a 1)
  (h3 : a 4 = 9 - a 3)
  (h4 : ∀ n, a (n + 1) = a n * q) :
  a 4 + a 5 = 27 :=
sorry

end geometric_sequence_sum_l36_36097


namespace domain_of_f_l36_36175

-- Define the function f(x)
def f (x : ℝ) : ℝ := 1 / (Real.log10 x - 1)

-- Define the conditions
def condition1 (x : ℝ) : Prop := Real.log10 x - 1 ≠ 0
def condition2 (x : ℝ) : Prop := x > 0

-- Prove that the domain of f is (0,10) ∪ (10,+∞)
theorem domain_of_f (x : ℝ) : (condition1 x ∧ condition2 x) ↔ (x ∈ Set.Ioo 0 10 ∨ x ∈ Set.Ioi 10) :=
by
  sorry

end domain_of_f_l36_36175


namespace calculate_radius_of_film_l36_36473

-- Define given conditions
def liquid_volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h
def film_volume (R thickness : ℝ) : ℝ := π * R^2 * thickness

theorem calculate_radius_of_film :
  (R : ℝ) (h_cylinder r_cylinder thickness : ℝ) (h_cylinder = 10) (r_cylinder = 5) (thickness = 0.2) :
  liquid_volume_cylinder r_cylinder h_cylinder = film_volume R thickness →
  R = Real.sqrt 1250 :=
by
  sorry

end calculate_radius_of_film_l36_36473


namespace cone_curved_surface_area_l36_36234

-- Define constants for the problem
def radius := 21    -- radius in meters
def slant_height := 15  -- slant height in meters

-- Curved surface area formula for a cone
def curved_surface_area (r l : ℝ) : ℝ := Real.pi * r * l

-- Statement to be proved
theorem cone_curved_surface_area :
  curved_surface_area radius slant_height ≈ 989.36 :=
by
  -- proof goes here
  sorry

end cone_curved_surface_area_l36_36234


namespace range_g_l36_36034

noncomputable theory

def f (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ -3 then -x - 1
  else if -3 < x ∧ x ≤ 3 then -x
  else if 3 < x ∧ x ≤ 4 then -x + 1
  else 0  -- Since x should be in [-4,4], this 'else' clause will never be used

def g (x : ℝ) : ℝ := f(x) - 2 * x 

theorem range_g : set.Icc (-11 : ℝ) (11 : ℝ) = set.univ.range g :=
begin
  sorry
end

end range_g_l36_36034


namespace min_value_proof_l36_36126

open Real

noncomputable theory

def min_value (a b c : ℝ) := (1 / a) + (2 / b) + (3 / c)

theorem min_value_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : 2 * a + 3 * b + 4 * c = 1) :
  min_value a b c = 20 + 4 * sqrt 3 + 20 * sqrt 2 :=
sorry

end min_value_proof_l36_36126


namespace find_a_of_parabola_and_hyperbola_intersection_l36_36039

theorem find_a_of_parabola_and_hyperbola_intersection
  (a : ℝ)
  (h_a_pos : a > 0)
  (h_asymptotes_intersect_directrix_distance : ∀ (x_A x_B : ℝ),
    -1 / (4 * a) = (1 / 2) * x_A ∧ -1 / (4 * a) = -(1 / 2) * x_B →
    |x_B - x_A| = 4) : a = 1 / 4 := by
  sorry

end find_a_of_parabola_and_hyperbola_intersection_l36_36039


namespace eq_of_ellipse_area_of_triangle_eq_of_line_MN_l36_36764

-- Definitions for the conditions in the problem
variable (a b c x y : ℝ) (P M N : ℝ × ℝ) (l1 l2 : Set (ℝ × ℝ))

-- Conditions given in the problem
def ellipseC := ∀ x y, (x, y) ∈ { (x, y) | (x^2 / a^2) + (y^2 / b^2) = 1 }
def ellipseParams := a > b ∧ b > 0 ∧ c = Real.sqrt 2 * b
def pointPOnEllipse := P = (-1, -1) ∧ (-1)^2 / a^2 + (-1)^2 / b^2 = 1
def linesThroughP := (∀ q ∈ l1, q = P ∨ q ∉ l1) ∧ (∀ q ∈ l2, q = P ∨ q ∉ l2) ∧ l1 ⊆ ellipseC ∧ l2 ⊆ ellipseC

-- Question 1: Prove the equation of the ellipse
theorem eq_of_ellipse :
  ellipseParams a b c ∧ pointPOnEllipse a b P ∧ a^2 = 4 ∧ b^2 = 4/3 → ((x^2 / 4) + (3 * y^2 / 4) = 1) := 
sorry

-- Question 2: Find the area of triangle PMN when slope of l1 is -1
theorem area_of_triangle (h : slope l1 = -1) :
  area_triangle P M N = 2 :=
sorry

-- Question 3: Equation of line MN when the midpoint of MN is on the x-axis
theorem eq_of_line_MN (midpoint_on_x_axis : (fst M + fst N) / 2 = 0):
  line_eqn M N = (λ x, -x) :=
sorry

end eq_of_ellipse_area_of_triangle_eq_of_line_MN_l36_36764


namespace isosceles_triangle_perimeter_l36_36352

theorem isosceles_triangle_perimeter (m : ℝ) (a b : ℝ) 
  (h1 : 3 = a ∨ 3 = b)
  (h2 : a ≠ b)
  (h3 : a^2 - (m+1)*a + 2*m = 0)
  (h4 : b^2 - (m+1)*b + 2*m = 0) :
  (a + b + a = 11) ∨ (a + a + b = 10) := 
sorry

end isosceles_triangle_perimeter_l36_36352


namespace symmetric_circle_proof_l36_36507

-- Define the original circle equation
def original_circle_eq (x y : ℝ) : Prop :=
  (x + 2)^2 + y^2 = 5

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop :=
  y = x

-- Define the symmetric circle equation
def symmetric_circle_eq (x y : ℝ) : Prop :=
  x^2 + (y + 2)^2 = 5

-- The theorem to prove
theorem symmetric_circle_proof (x y : ℝ) :
  (original_circle_eq x y) ↔ (symmetric_circle_eq x y) :=
sorry

end symmetric_circle_proof_l36_36507


namespace rectangular_solid_volume_l36_36944

theorem rectangular_solid_volume 
  (x y z : ℝ)
  (h1 : x * y = 20)
  (h2 : y * z = 15)
  (h3 : x * z = 12) :
  x * y * z = 60 :=
by
  sorry

end rectangular_solid_volume_l36_36944


namespace average_income_l36_36231

theorem average_income : 
  let incomes := [45, 50, 60, 65, 70] in
  (incomes.sum / incomes.length) = 58 := by sorry

end average_income_l36_36231


namespace ellipse_eccentricity_l36_36116

noncomputable def eccentricity (a b c : ℝ) (condition1 : a > b ∧ b > 0): ℝ := c / a

theorem ellipse_eccentricity (a b c : ℝ) (condition1 : a > b ∧ b > 0)
  (condition2 : c^2 = a^2 - b^2)
  (condition3 : ∃ (P : ℝ × ℝ), (P.1)^2 / a^2 + (P.2)^2 / b^2 = 1 ∧ P.1 = c)
  (condition4 : ∃ (M : ℝ × ℝ), 2 * M.1 = P.1 + (-c) ∧ M.2 = 0 ∧ M.1 = 0)
  (condition5 : ∠P F₁ F₂ = 30) :
  eccentricity a b c condition1 = √3 / 3 :=
sorry

end ellipse_eccentricity_l36_36116


namespace middle_school_students_count_l36_36415

variable (M H m h : ℕ)
variable (total_students : ℕ := 36)
variable (percentage_middle : ℕ := 20)
variable (percentage_high : ℕ := 25)

theorem middle_school_students_count :
  total_students = 36 ∧ (m = h) →
  (percentage_middle / 100 * M = m) ∧
  (percentage_high / 100 * H = h) →
  M + H = total_students →
  M = 16 :=
by sorry

end middle_school_students_count_l36_36415


namespace range_of_a_l36_36782

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then x^2 - 2 * a * x - 1 / 2 else Real.log x

theorem range_of_a (a : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔ a ∈ Set.Ici (1 / 4) :=
sorry

end range_of_a_l36_36782


namespace cos_minus_tan_simplification_l36_36566

theorem cos_minus_tan_simplification (α : ℝ) :
  (cos 4 * α) ^ (-2) - (tan (3 * π + 4 * α)) ^ 2 - 2 * (cos α) ^ 2 - sqrt 3 * cos (3 / 2 * π - 2 * α) =
  2 * sin (2 * α - π / 6) :=
by
  sorry

end cos_minus_tan_simplification_l36_36566


namespace vegetarian_count_l36_36235

theorem vegetarian_count (only_veg only_non_veg both_veg_non_veg : ℕ) 
  (h1 : only_veg = 19) (h2 : only_non_veg = 9) (h3 : both_veg_non_veg = 12) : 
  (only_veg + both_veg_non_veg = 31) :=
by
  -- We leave the proof here
  sorry

end vegetarian_count_l36_36235


namespace geometry_problem_l36_36378

-- Definitions for geometric relationships: parallel and perpendicular
variables {Line Plane : Type}
variable (a b : Line)
variable (α β γ : Plane)

-- Relation Definitions
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def plane_parallel (p q : Plane) : Prop := sorry
def plane_perpendicular (p q : Plane) : Prop := sorry

-- Given conditions
axiom h1 : plane_perpendicular α γ
axiom h2 : plane_parallel β γ

-- Prove the statement
theorem geometry_problem : plane_perpendicular α β := 
by 
  sorry

end geometry_problem_l36_36378


namespace simplify_expression_l36_36163

theorem simplify_expression (b : ℝ) : (1 * 3 * b * 4 * b^2 * 5 * b^3 * 6 * b^4) = 360 * b^10 :=
by sorry

end simplify_expression_l36_36163


namespace joan_flour_addition_l36_36110

theorem joan_flour_addition (original_flour : ℕ) (already_added : ℕ) :
  original_flour = 7 → already_added = 3 → (2 * original_flour - already_added = 11) :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end joan_flour_addition_l36_36110


namespace range_of_a_value_of_a_l36_36850

-- Problem 1
theorem range_of_a (a : ℝ) :
  (∃ x, (2 < x ∧ x < 4) ∧ (a < x ∧ x < 3 * a)) ↔ (4 / 3 ≤ a ∧ a < 4) :=
sorry

-- Problem 2
theorem value_of_a (a : ℝ) :
  (∀ x, (2 < x ∧ x < 4) ∨ (a < x ∧ x < 3 * a) ↔ (2 < x ∧ x < 6)) ↔ (a = 2) :=
sorry

end range_of_a_value_of_a_l36_36850


namespace max_distance_D_origin_l36_36849

noncomputable def z : ℂ := sorry  -- given |z| = 1

noncomputable def A : ℂ := z
noncomputable def B : ℂ := 2 * (1 - I) * z
noncomputable def C : ℂ := 3 * conj z

noncomputable def D : ℂ := B + C - A -- Calculating the fourth vertex w = B + C - A

theorem max_distance_D_origin :
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ abs z = 1 ∧ ¬ (A, B, C are collinear)
  ∧ abs D ≤ 3 * real.sqrt 2 := 
sorry

end max_distance_D_origin_l36_36849


namespace acute_triang_circle_intersect_eq_l36_36414

-- Define the problem conditions and the proof statement
theorem acute_triang_circle_intersect_eq (ABC : Triangle) (A B C : Point)
  (acute_ABC : is_acute_triangle ABC)
  (ω1 ω2 : Circle)
  (diameter_ω1 : ω1.diameter = Segment A C)
  (diameter_ω2 : ω2.diameter = Segment A B)
  (F : Point) (H_F : F ∈ ω1 ∧ F ≠ A ∧ F ≠ C ∧ F ∈ Line A B)
  (E : Point) (H_E : E ∈ ω2 ∧ E ≠ A ∧ E ≠ B ∧ E ∈ Line A C)
  (P : Point) (H_P : P ∈ (Line B E) ∧ P ∈ ω1)
  (Q : Point) (H_Q : Q ∈ (Line C F) ∧ Q ∈ ω2) :
  Segment A P = Segment A Q := 
sorry

end acute_triang_circle_intersect_eq_l36_36414


namespace present_age_of_son_l36_36232

theorem present_age_of_son (F S : ℕ) (h1 : F = S + 24) (h2 : F + 2 = 2 * (S + 2)) : S = 22 := by
  sorry

end present_age_of_son_l36_36232


namespace sufficient_but_not_necessary_l36_36020

theorem sufficient_but_not_necessary (x : ℝ) : (x > 0 → x * (x + 1) > 0) ∧ ¬ (x * (x + 1) > 0 → x > 0) := 
by 
sorry

end sufficient_but_not_necessary_l36_36020


namespace equivalent_proof_problem_l36_36575

noncomputable def problem_statement (x : Fin 10 → ℝ) (h : ∀ k : Fin 10, x.sum (λ j, if j.val ≤ k.val then 1 else 0) * x.sum (λ j, if j.val > k.val then 1 else 0) = 1) : Prop :=
  x 0 = (Real.sqrt 6 - Real.sqrt 2) / 2 ∧
  x 1 = Real.sqrt 2 - Real.sqrt 6 / 2 ∧
  x 2 = (2 * Real.sqrt 6 - 3 * Real.sqrt 2) / 6 ∧
  x 3 = (9 * Real.sqrt 2 - 5 * Real.sqrt 6) / 6 ∧
  x 4 = (3 * Real.sqrt 6 - 5 * Real.sqrt 2) / 4 ∧
  x 5 = x 4 ∧
  x 6 = x 3 ∧
  x 7 = x 2 ∧
  x 8 = x 1 ∧
  x 9 = x 0

-- statement only, no proof provided.
theorem equivalent_proof_problem : ∀ (x : Fin 10 → ℝ),
  (∀ k : Fin 10, x.sum (λ j, if j.val ≤ k.val then 1 else 0) * x.sum (λ j, if j.val > k.val then 1 else 0) = 1) 
  → problem_statement x sorry

end equivalent_proof_problem_l36_36575


namespace arithmetic_mean_of_positive_three_digit_multiples_of_8_l36_36955

open Nat

theorem arithmetic_mean_of_positive_three_digit_multiples_of_8 : 
  let a := 104
  let l := 992
  2 * ∑ k in range 112, (8 * (k + 13)) / 112 = 548 :=
by
  sorry

end arithmetic_mean_of_positive_three_digit_multiples_of_8_l36_36955


namespace garden_area_l36_36520

theorem garden_area (P : ℝ) (hP : P = 72) (l w : ℝ) (hL : l = 3 * w) (hPerimeter : 2 * l + 2 * w = P) : l * w = 243 := 
by
  sorry

end garden_area_l36_36520


namespace at_least_one_student_solved_five_problems_l36_36585

theorem at_least_one_student_solved_five_problems
  (num_students : ℕ)
  (total_problems : ℕ)
  (x y z : ℕ)
  (hx : x + y + z ≤ num_students)
  (hy : x + 2 * y + 3 * z ≤ total_problems)
  (num_students_eq : num_students = 10)
  (total_problems_eq : total_problems = 35) :
  ∃ s, s ≥ 5 :=
by {
  have num_students : num_students = 10 := num_students_eq,
  have total_problems : total_problems = 35 := total_problems_eq,
  sorry
}

end at_least_one_student_solved_five_problems_l36_36585


namespace problem_solution_l36_36864

-- Define the set P
def P : Set ℝ := { x | x > -1 }

-- Define the set Q
def Q : Set ℤ := { y | y^2 ≤ 4 }

-- Define the intersection of P and Q, but translated into the form we need to prove
def P_intersect_Q : Set ℝ := { x | x ∈ P ∧ x ∈ (coe '' Q) }

theorem problem_solution : P_intersect_Q = {0, 1, 2} := 
by 
  sorry  -- Proof skipped

end problem_solution_l36_36864


namespace goods_train_length_l36_36646

theorem goods_train_length 
  (v_kmph : ℝ) (L_p : ℝ) (t : ℝ) (v_mps : ℝ) (d : ℝ) (L_t : ℝ) 
  (h1 : v_kmph = 96) 
  (h2 : L_p = 480) 
  (h3 : t = 36) 
  (h4 : v_mps = v_kmph * (5/18)) 
  (h5 : d = v_mps * t) : 
  L_t = d - L_p :=
sorry

end goods_train_length_l36_36646


namespace triangle_side_lengths_values_l36_36513

theorem triangle_side_lengths_values :
  ∃ (m_values : Finset ℕ), m_values = {m ∈ Finset.range 750 | m ≥ 4} ∧ m_values.card = 746 :=
by
  sorry

end triangle_side_lengths_values_l36_36513


namespace cone_base_radius_and_slant_height_l36_36563

noncomputable def sector_angle := 300
noncomputable def sector_radius := 10
noncomputable def arc_length := (sector_angle / 360) * 2 * Real.pi * sector_radius

theorem cone_base_radius_and_slant_height :
  ∃ (r l : ℝ), arc_length = 2 * Real.pi * r ∧ l = sector_radius ∧ r = 8 ∧ l = 10 :=
by 
  sorry

end cone_base_radius_and_slant_height_l36_36563


namespace vector_projection_l36_36317

-- Define the vector we want to project
def vec : ℝ^3 := ![4, 2, -3]

-- Define the direction vector of the line
def dir : ℝ^3 := ![3, 1, -2]

-- Prove that the projection of vec onto dir is as expected
theorem vector_projection :
  let projection := ((vec ⬝ dir) / (dir ⬝ dir)) • dir in
  projection = ![(30 : ℝ) / 7, 10 / 7, -(20 : ℝ) / 7] :=
by
  sorry

end vector_projection_l36_36317


namespace lattice_points_hyperbola_l36_36297

theorem lattice_points_hyperbola : 
  {p : ℤ × ℤ | p.1^2 - p.2^2 = 1800^2}.card = 125 := 
by
  sorry

end lattice_points_hyperbola_l36_36297


namespace probability_not_within_square_B_l36_36998

theorem probability_not_within_square_B {A B : Type} 
  (area_A : ℝ) (perimeter_B : ℝ) (area_B : ℝ) (not_covered : ℝ) 
  (h1 : area_A = 30) 
  (h2 : perimeter_B = 16) 
  (h3 : area_B = 16) 
  (h4 : not_covered = area_A - area_B) :
  (not_covered / area_A) = 7 / 15 := by sorry

end probability_not_within_square_B_l36_36998


namespace Theta_max_success_ratio_l36_36141

-- Definitions and conditions
variables {a b c d e f : ℕ}

-- Omega's success ratios
def Omega_day1_success := 1 / 2
def Omega_day2_success := 3 / 4
def Omega_day3_success := 3 / 4

-- Conditions
def Theta_day1_success := 0 < a ∧ a < (b / 2)
def Theta_day2_success := 0 < c ∧ c < (3 * d / 4)
def Theta_day3_success := 0 < e ∧ e < (3 * f / 4)
def Theta_total_attempted := b + d + f = 600

-- The proof statement for Theta's maximum success ratio
theorem Theta_max_success_ratio (h1 : Theta_day1_success) (h2 : Theta_day2_success) (h3 : Theta_day3_success)
  (h4 : Theta_total_attempted) : (a + c + e : ℚ) / 600 = 56 / 75 :=
sorry

end Theta_max_success_ratio_l36_36141


namespace range_of_m_l36_36470

noncomputable def f (x m : ℝ) : ℝ := 3 * Real.cos (π * x / m)

theorem range_of_m (m : ℝ) :
  (∃ x₀ : ℝ, x₀ ≠ 0 ∧ x₀^2 + f x₀ m < 4 * m) ↔ (2 - Real.sqrt 7 < m ∧ m < 2 + Real.sqrt 7) :=
by
  sorry

end range_of_m_l36_36470


namespace construction_costs_l36_36260

theorem construction_costs 
  (land_cost_per_sq_meter : ℕ := 50)
  (bricks_cost_per_1000 : ℕ := 100)
  (roof_tile_cost_per_tile : ℕ := 10)
  (land_area : ℕ := 2000)
  (number_of_bricks : ℕ := 10000)
  (number_of_roof_tiles : ℕ := 500) :
  50 * 2000 + (100 / 1000) * 10000 + 10 * 500 = 106000 :=
by sorry

end construction_costs_l36_36260


namespace necessary_but_not_sufficient_l36_36759

variable {x : ℝ}

def p : Prop := |x| > 2
def q : Prop := x < -2

theorem necessary_but_not_sufficient :
  (q → p) ∧ ¬(p → q) :=
by
  sorry

end necessary_but_not_sufficient_l36_36759


namespace solution_of_inequality_system_l36_36530

-- Definitions derived from the conditions in the problem
def inequality1 (x : ℝ) : Prop := 3 * x - 1 ≥ x + 1
def inequality2 (x : ℝ) : Prop := x + 4 > 4 * x - 2
def solution_set (x : ℝ) : Prop := 1 ≤ x ∧ x < 2

-- The Lean 4 statement for the proof problem
theorem solution_of_inequality_system (x : ℝ) : inequality1 x ∧ inequality2 x ↔ solution_set x := by
  sorry

end solution_of_inequality_system_l36_36530


namespace modulus_of_complex_number_l36_36780

theorem modulus_of_complex_number :
  let z := (⟨2, 1⟩ * ⟨1, -1⟩ : ℂ) in |z| = Real.sqrt 10 :=
by
  let z := (⟨2, 1⟩ * ⟨1, -1⟩ : ℂ)
  have : z = ⟨3, -1⟩ := by
    simp [Complex.mul, Complex.add, Complex.sub, Complex.one_re, Complex.one_im, Complex.I_re, Complex.I_im, Complex.ext_iff]
  rw [this]
  show |⟨3, -1⟩| = Real.sqrt 10
  sorry

end modulus_of_complex_number_l36_36780


namespace peanut_mixture_l36_36167

-- Definitions of given conditions
def virginia_peanuts_weight : ℝ := 10
def virginia_peanuts_cost_per_pound : ℝ := 3.50
def spanish_peanuts_cost_per_pound : ℝ := 3.00
def texan_peanuts_cost_per_pound : ℝ := 4.00
def desired_cost_per_pound : ℝ := 3.60

-- Definitions of unknowns S (Spanish peanuts) and T (Texan peanuts)
variable (S T : ℝ)

-- Equation derived from given conditions
theorem peanut_mixture :
  (0.40 * T) - (0.60 * S) = 1 := sorry

end peanut_mixture_l36_36167


namespace largest_possible_dividend_l36_36807

theorem largest_possible_dividend (divisor quotient : ℕ) (remainder : ℕ) 
  (h_divisor : divisor = 18)
  (h_quotient : quotient = 32)
  (h_remainder : remainder < divisor) :
  quotient * divisor + remainder = 593 :=
by
  -- No proof here, add sorry to skip the proof
  sorry

end largest_possible_dividend_l36_36807


namespace transformed_std_dev_l36_36065

theorem transformed_std_dev 
  (std_dev_x : ℝ)
  (h : std_dev_x = 8) :
  std_dev (λ i : Fin 10, 3 * (std_dev_x i) - 1) = 24 := 
sorry

end transformed_std_dev_l36_36065


namespace length_major_axis_of_tangent_ellipse_l36_36435

def ellipse_major_axis_length (F1 F2 : ℝ × ℝ) (tangent_to_x_axis : Bool) : ℝ :=
if tangent_to_x_axis then 
    ((4 : ℝ) * (sqrt ((F2.1 - F1.1)^2 + (F2.2 - F1.2)^2))/2).ceil else 
    0

theorem length_major_axis_of_tangent_ellipse (F1 F2 : ℝ × ℝ) (tangent_to_x_axis : Bool)
  (hF1 : F1 = (1, 1)) (hF2 : F2 = (5, 2)) (hTangent : tangent_to_x_axis = true) :
  ellipse_major_axis_length F1 F2 tangent_to_x_axis = 5 :=
by
  rw [hF1, hF2, hTangent]
  sorry

end length_major_axis_of_tangent_ellipse_l36_36435


namespace k_equals_three_fourths_l36_36736

theorem k_equals_three_fourths : ∀ a b c d : ℝ, a ∈ Set.Ici (-1) → b ∈ Set.Ici (-1) → c ∈ Set.Ici (-1) → d ∈ Set.Ici (-1) →
  a^3 + b^3 + c^3 + d^3 + 1 ≥ (3 / 4) * (a + b + c + d) :=
by
  intros
  sorry

end k_equals_three_fourths_l36_36736


namespace point_on_x_axis_l36_36075

theorem point_on_x_axis (A B C D : ℝ × ℝ) : B = (3,0) → B.2 = 0 :=
by
  intros h
  subst h
  exact rfl

end point_on_x_axis_l36_36075


namespace smallest_a_l36_36319

theorem smallest_a :
  ∃ (a : ℕ), a > 0 ∧ (∃ n : ℕ, n > 0 ∧ n % 2 = 1 ∧ 2001 ∣ (55^n + a * 32^n)) ∧
  ∀ b : ℕ, (b > 0 ∧ (∃ m : ℕ, m > 0 ∧ m % 2 = 1 ∧ 2001 ∣ (55^m + b * 32^m))) → b ≥ a :=
begin
  let a := 2000,
  existsi a,
  split,
  { exact nat.succ_pos' a, },
  split,
  { existsi 1,
    split,
    { exact nat.succ_pos' 0, },
    split,
    { show 1 % 2 = 1,
      exact rfl, },
    { have h : 55^1 + 2000 * 32^1 = 2001 := by ring,
      show 2001 ∣ (55^1 + 2000 * 32^1),
      exact dvd.intro 1 h, }, },
  { intros b hb,
    cases hb with b_pos hb,
    cases hb with m hm,
    cases hm with m_pos hm,
    cases hm with m_odd h2001,
    have h := (nat.le_and_le_iff).mpr ⟨le_refl b, nat.succ_le_of_lt b_pos⟩,
    exact le_trans h.2 (nat.succ_le_succ (nat.pred_le _)),
    },
end

end smallest_a_l36_36319


namespace total_balloon_cost_l36_36334

def cost_per_balloon_fred : ℕ → ℝ
| n := n * 1.0

def cost_per_balloon_sam : ℕ → ℝ
| n := n * 1.5

def cost_per_balloon_dan : ℕ → ℝ
| n := n * 0.75

def total_cost (n_fred n_sam n_dan : ℕ) : ℝ :=
  cost_per_balloon_fred n_fred + cost_per_balloon_sam n_sam + cost_per_balloon_dan n_dan

theorem total_balloon_cost :
  total_cost 10 46 16 = 91 := by
  sorry

end total_balloon_cost_l36_36334


namespace sum_of_powers_mod_7_eq_6_l36_36060

theorem sum_of_powers_mod_7_eq_6 : 
  (1^6 + 2^6 + 3^6 + 4^6 + 5^6 + 6^6) % 7 = 6 :=
by
  -- Using Fermat's Little Theorem (proved elsewhere in mathlib)
  have h1 : 1^6 % 7 = 1 := by sorry,
  have h2 : 2^6 % 7 = 1 := by sorry,
  have h3 : 3^6 % 7 = 1 := by sorry,
  have h4 : 4^6 % 7 = 1 := by sorry,
  have h5 : 5^6 % 7 = 1 := by sorry,
  have h6 : 6^6 % 7 = 1 := by sorry,
  -- Summing and proving the final result
  calc
    (1^6 + 2^6 + 3^6 + 4^6 + 5^6 + 6^6) % 7
        = (1 + 1 + 1 + 1 + 1 + 1) % 7 := by rw [h1, h2, h3, h4, h5, h6]
    ... = 6 % 7 := by norm_num

end sum_of_powers_mod_7_eq_6_l36_36060


namespace arithmetic_sequence_general_formula_sum_of_b_n_l36_36419

theorem arithmetic_sequence_general_formula (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 3 + a 4 = 4)
  (h2 : a 5 + a 7 = 6) :
  (∀ n, a n = (2/5 : ℝ) * n + (3/5 : ℝ)) :=
  by sorry

theorem sum_of_b_n (b : ℕ → ℝ)
  (a_n_formula : ∀ n, b n = ⌊(2/5 : ℝ) * n + (3/5 : ℝ)⌋) :
  (∑ i in finset.range 10, b (i+1)) = 24 :=
  by sorry

end arithmetic_sequence_general_formula_sum_of_b_n_l36_36419


namespace chord_length_le_diameter_chord_length_eq_diameter_iff_l36_36512

-- Define a circle with center O and radius R
variable (O : Point) (R : ℝ) (hR : R > 0)

-- Define points A and B on the circle.
variable (A B : Point)
variable (hA : dist O A = R) (hB : dist O B = R)

-- The goal is to prove the length of chord AB does not exceed 2R,
-- and it equals 2R if and only if AB is the diameter.
theorem chord_length_le_diameter :
  dist A B ≤ 2 * R :=
sorry

theorem chord_length_eq_diameter_iff :
  dist A B = 2 * R ↔ collinear {A, B, O} :=
sorry

end chord_length_le_diameter_chord_length_eq_diameter_iff_l36_36512


namespace game_ends_in_36_rounds_l36_36991

theorem game_ends_in_36_rounds 
    (tokens_A : ℕ := 17) (tokens_B : ℕ := 16) (tokens_C : ℕ := 15)
    (rounds : ℕ) 
    (game_rule : (tokens_A tokens_B tokens_C round_num : ℕ) → Prop) 
    (extra_discard_rule : (tokens_A tokens_B tokens_C round_num : ℕ) → Prop)  
    (game_ends_when_token_zero : (tokens_A tokens_B tokens_C : ℕ) → Prop) :
    game_rule tokens_A tokens_B tokens_C rounds ∧
    extra_discard_rule tokens_A tokens_B tokens_C rounds ∧
    game_ends_when_token_zero tokens_A tokens_B tokens_C → 
    rounds = 36 := by
    sorry

end game_ends_in_36_rounds_l36_36991


namespace general_term_formula_sum_first_n_terms_l36_36021

noncomputable def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a n = a 0 + n * d

theorem general_term_formula :
  ∃a : ℕ → ℕ, (arithmetic_sequence a 1) ∧ a 1 = 2 ∧ a 3^2 = a 1 * a 7 ∧
  ∀ n : ℕ, a n = n + 1 :=
by
  sorry

def sequence_b (a : ℕ → ℕ) : ℕ → ℕ :=
  λ n, a n + 2^(a n)

theorem sum_first_n_terms (a : ℕ → ℕ) (b := sequence_b a) (S : ℕ → ℕ) :
  (∀ n : ℕ, a n = n + 1) →
  ∀ n : ℕ, S n = ∑ i in Finset.range n, b i →
  ∀ n : ℕ, S n = 2^(n + 2) + (n^2 + 3*n - 8) / 2 :=
by
  sorry

end general_term_formula_sum_first_n_terms_l36_36021


namespace total_weight_mason_hotdogs_l36_36881

-- Definitions from conditions
def weight_hotdog := 2
def weight_burger := 5
def weight_pie := 10
def noah_burgers := 8
def jacob_pies := noah_burgers - 3
def mason_hotdogs := 3 * jacob_pies

-- Statement to prove
theorem total_weight_mason_hotdogs : mason_hotdogs * weight_hotdog = 30 := 
by 
  sorry

end total_weight_mason_hotdogs_l36_36881


namespace smallest_a_not_prime_l36_36744

theorem smallest_a_not_prime (x : ℤ) : ∃ a : ℕ, a > 0 ∧ (∀ x : ℤ, ¬ prime (x^4 + (a+1)^2)) ∧ (∀ n : ℕ, n > 0 ∧ (∀ x : ℤ, ¬ prime (x^4 + (n+1)^2)) → a ≤ n) := 
  ∃ (a : ℕ), a = 8 ∧ a > 0 ∧ (∀ x : ℤ, ¬ prime (x^4 + (8+1)^2)) sorry

end smallest_a_not_prime_l36_36744


namespace correct_statement_l36_36964

noncomputable def coefficient_monomial (m : Char → ℕ) (e : ℕ) (c : ℤ) :=
  c

-- Definitions extracted from given conditions
def degree_monomial (a b : ℕ) := a + b
def terms_polynomial := [(-4, (2, 1)), (3, (1, 1)), (-5, (0, 0))]
def coefficient_monomial_exp := (- (2 ^ 3) : ℤ)
def total_degree (a b : ℕ) := a + b
def is_quadratic_binomial (f : ℕ → ℕ → ℚ) (g : ℕ → ℕ → ℚ) :=
  (f 1 1 + g 0 0) ≠ 0

-- The actual theorem statement
theorem correct_statement :
  (coefficient_monomial (fun c => c) 2 0 = 0 ∧ degree_monomial 1 1 = 2 ∧ ¬ (coefficient_monomial (fun c => c) 2 1 = 1 ∧ degree_monomial 1 1 = 2)) ∧
  (terms_polynomial = [(-4, (2, 1)), (3, (1, 1)), (-5, (0, 0))] ∧ ¬ terms_polynomial = [(-4, (2, 1)), (3, (1, 1)), (5, (0, 0))]) ∧
  (coefficient_monomial_exp = -8 ∧ total_degree 2 3 = 5 ∧ ¬ (coefficient_monomial_exp = -2 ∧ total_degree 2 3 = 5)) ∧
  (is_quadratic_binomial (fun x y => x - y/2) (fun x y => 0) = true)
  → 
  (D = "The coefficient of the monomial \(ab \); degree is 2")
  ∨ 
  (D = "The terms of the polynomial \( -4a^2b + 3ab - 5 \) are \( -4a^2b, 3ab, 5 \)")
  ∨ 
  (D = "The coefficient of the monomial \( -2^3a^2b^3 \) is \( -2 \); degree is 5 ")
  ∨ 
  (D = " \(\frac{xy-1}{2}\) is a quadratic binomial")
 :=
  sorry

end correct_statement_l36_36964


namespace cube_volume_l36_36019

variable (V_sphere : ℝ)
variable (V_cube : ℝ)
variable (R : ℝ)
variable (a : ℝ)

theorem cube_volume (h1 : V_sphere = (32 / 3) * Real.pi)
    (h2 : V_sphere = (4 / 3) * Real.pi * R^3)
    (h3 : R = 2)
    (h4 : R = (Real.sqrt 3 / 2) * a)
    (h5 : a = 4 * Real.sqrt 3 / 3) :
    V_cube = (4 * Real.sqrt 3 / 3) ^ 3 :=
  by
    sorry

end cube_volume_l36_36019


namespace find_f_prime_at_1_l36_36341

noncomputable def f (x : ℝ) : ℝ := Real.log x - 3 * f' 1 * x

-- Assuming the derivative of f at x = 1 as a hypothesis to be used in the proof
axiom f' : ℝ → ℝ
axiom H : ∀ x, deriv f x = (1 / x) - 3 * f' 1

theorem find_f_prime_at_1 :
  f' 1 = 1 / 4 := by
  sorry

end find_f_prime_at_1_l36_36341


namespace minimum_omega_l36_36948

theorem minimum_omega (ω : ℝ) (h_pos : ω > 0) :
  (∃ k : ℤ, ω * (3 * π / 4) - ω * (π / 4) = k * π) → ω = 2 :=
by
  sorry

end minimum_omega_l36_36948


namespace problem_statement_l36_36938

theorem problem_statement (x : ℝ) (hx : x + 1/x = Real.sqrt 5) : 
  x^12 - 7 * x^8 + x^4 = 343 :=
sorry

end problem_statement_l36_36938


namespace percent_value_in_quarters_l36_36228

def num_dimes : ℕ := 70
def num_quarters : ℕ := 30
def num_half_dollars : ℕ := 10
def value_dime : ℕ := 10
def value_quarter : ℕ := 25
def value_half_dollar : ℕ := 50

theorem percent_value_in_quarters :
  (num_quarters * value_quarter : ℕ) / ((num_dimes * value_dime) + (num_quarters * value_quarter) + (num_half_dollars * value_half_dollar) : ℕ) * 100 = 38.46 :=
sorry

end percent_value_in_quarters_l36_36228


namespace range_of_m_l36_36361

theorem range_of_m (g : ℝ → ℝ) (m : ℝ) (h_g : ∀ x, g x = g' 1 * real.exp (x - 1) - g 0 * x + 1 / 2 * x ^ 2)
  (h_exist : ∃ x0 : ℝ, 2 * m - 1 ≥ g x0) : 1 ≤ m :=
sorry

end range_of_m_l36_36361


namespace probability_of_rolling_divisor_of_12_l36_36596

def is_divisor (a b : ℕ) : Prop := b % a = 0

noncomputable def probability_divisor_12 : ℚ :=
  let divisors := { n | is_divisor n 12 ∧ n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} }
  let favorable_outcomes := (divisors : set ℕ).to_finset.card
  let total_outcomes := 12
  (favorable_outcomes : ℚ) / total_outcomes

theorem probability_of_rolling_divisor_of_12 : probability_divisor_12 = 1 / 2 := by
  sorry

end probability_of_rolling_divisor_of_12_l36_36596


namespace patrick_speed_l36_36893

def distance : ℝ := 14  -- Distance from Patrick's house to Aaron's house in miles
def time : ℝ := 2       -- Time in hours

def speed : ℝ := distance / time  -- Speed definition

theorem patrick_speed : speed = 7 := by
  sorry  -- Proof is omitted

end patrick_speed_l36_36893


namespace curve_cartesian_equation_point_R_coordinates_minimum_perimeter_of_rectangle_l36_36411

noncomputable def polar_to_cartesian_equation (rho theta : ℝ) : ℝ :=
  rho * cos theta * rho * cos theta / 3 + rho * sin theta * rho * sin theta

theorem curve_cartesian_equation:
  ∀ (theta : ℝ),
  polar_to_cartesian_equation (sqrt (3 / (1 + 2 * (sin theta)^2))) theta = 1 :=
sorry

theorem point_R_coordinates:
  ∃ x y : ℝ, (x = 2) ∧ (y = 2)
  ∧ (x = 2 * sqrt 2 * cos (π / 4))
  ∧ (y = 2 * sqrt 2 * sin (π / 4)) :=
by {
  use [2, 2],
  split,
  exact rfl,
  split,
  exact rfl,
  split;
  norm_num,
  rw [cos_pi_div_four],
  rw [sin_pi_div_four],
}

theorem minimum_perimeter_of_rectangle:
  ∃ (P : ℝ × ℝ), let (x, y) := P in (x = 3 / 2) ∧ (y = 1 / 2) ∧ 8 = 8 :=
sorry

end curve_cartesian_equation_point_R_coordinates_minimum_perimeter_of_rectangle_l36_36411


namespace at_least_one_truth_and_not_knight_l36_36885

def isKnight (n : Nat) : Prop := n = 1   -- Identifier for knights
def isKnave (n : Nat) : Prop := n = 0    -- Identifier for knaves
def isRegular (n : Nat) : Prop := n = 2  -- Identifier for regular persons

def A := 2     -- Initially define A's type as regular (this can be adjusted)
def B := 2     -- Initially define B's type as regular (this can be adjusted)

def statementA : Prop := isKnight B
def statementB : Prop := ¬ isKnight A

theorem at_least_one_truth_and_not_knight :
  statementA ∧ ¬ isKnight A ∨ statementB ∧ ¬ isKnight B :=
sorry

end at_least_one_truth_and_not_knight_l36_36885


namespace area_of_triangle_PF1F2_is_1_l36_36035

noncomputable def hyperbola := {P : ℝ × ℝ | (P.1^2 / 3) - P.2^2 = 1}
def F1 := (2, 0) -- Because the hyperbola (x^2/a^2 - y^2/b^2 = 1) with foci along the x-axis
def F2 := (-2, 0)
def P : ℝ × ℝ := sorry -- A point on the hyperbola defined

-- Distance conditions from problem
axiom dist_PF1_PF2 : (|P - F1| + |P - F2| = 2 * real.sqrt 5)

-- The proof goal:
theorem area_of_triangle_PF1F2_is_1 
  (hP : P ∈ hyperbola) 
  (hDist : (|P - F1| + |P - F2|) = 2 * real.sqrt 5) 
  : triangle.area P F1 F2 = 1 :=
sorry

end area_of_triangle_PF1F2_is_1_l36_36035


namespace wheel_center_distance_l36_36663

theorem wheel_center_distance :
  ∀ (r : ℝ) (n : ℕ), r = 2 ∧ n = 2 → 2 * n * Real.pi * r = 8 * Real.pi :=
by
  intros r n h
  cases h with hr hn
  rw [hr, hn]
  norm_num
  ring

end wheel_center_distance_l36_36663


namespace georgia_total_cost_l36_36888

def carnation_price : ℝ := 0.50
def dozen_price : ℝ := 4.00
def teachers : ℕ := 5
def friends : ℕ := 14

theorem georgia_total_cost :
  ((dozen_price * teachers) + dozen_price + (carnation_price * (friends - 12))) = 25.00 :=
by
  sorry

end georgia_total_cost_l36_36888


namespace concert_duration_l36_36593

def duration_in_minutes (hours : Int) (extra_minutes : Int) : Int :=
  hours * 60 + extra_minutes

theorem concert_duration : duration_in_minutes 7 45 = 465 :=
by
  sorry

end concert_duration_l36_36593


namespace circumference_diff_l36_36206

noncomputable def pi : ℝ := 3.1

def area_A : ℝ := 198.4
def area_B : ℝ := 251.1

def radius (area : ℝ) : ℝ := real.sqrt (area / pi)

def circumference (radius : ℝ) : ℝ := 2 * pi * radius

theorem circumference_diff (h1 : pi = 3.1) (h2 : area_A = 198.4) (h3 : area_B = 251.1) : 
  abs (circumference (radius area_B) - circumference (radius area_A)) = 6.2 :=
sorry

end circumference_diff_l36_36206


namespace sqrt_diff_inequality_l36_36900

theorem sqrt_diff_inequality (a : ℝ) (h : a ≥ 3) : 
  sqrt a - sqrt (a - 2) < sqrt (a - 1) - sqrt (a - 3) :=
sorry

end sqrt_diff_inequality_l36_36900


namespace remaining_soup_feeds_20_adults_l36_36246

theorem remaining_soup_feeds_20_adults (cans_of_soup : ℕ) (feed_4_adults : ℕ) (feed_7_children : ℕ) (initial_cans : ℕ) (children_fed : ℕ)
    (h1 : feed_4_adults = 4)
    (h2 : feed_7_children = 7)
    (h3 : initial_cans = 8)
    (h4 : children_fed = 21) : 
    (initial_cans - (children_fed / feed_7_children)) * feed_4_adults = 20 :=
by
  sorry

end remaining_soup_feeds_20_adults_l36_36246


namespace tangent_lines_parabola_through_point_l36_36375

theorem tangent_lines_parabola_through_point :
  ∃ (m : ℝ), 
    (∀ (x y : ℝ), y = x ^ 2 + 1 → (y - 0) = m * (x - 0)) 
     ∧ ((m = 2 ∧ y = 2 * x) ∨ (m = -2 ∧ y = -2 * x)) :=
sorry

end tangent_lines_parabola_through_point_l36_36375


namespace georgia_total_carnation_cost_l36_36891

-- Define the cost of one carnation
def cost_of_single_carnation : ℝ := 0.50

-- Define the cost of one dozen carnations
def cost_of_dozen_carnations : ℝ := 4.00

-- Define the number of teachers
def number_of_teachers : ℕ := 5

-- Define the number of friends
def number_of_friends : ℕ := 14

-- Calculate the cost for teachers
def cost_for_teachers : ℝ :=
  (number_of_teachers : ℝ) * cost_of_dozen_carnations

-- Calculate the cost for friends
def cost_for_friends : ℝ :=
  cost_of_dozen_carnations + (2 * cost_of_single_carnation)

-- Calculate the total cost
def total_cost : ℝ := cost_for_teachers + cost_for_friends

-- Theorem stating the total cost
theorem georgia_total_carnation_cost : total_cost = 25 := by
  -- Placeholder for the proof
  sorry

end georgia_total_carnation_cost_l36_36891


namespace find_principal_amount_l36_36737

-- Defining the compound interest function
def compoundInterest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n)^(n * t) - P

-- Given values
def r := 0.04
def n := 2
def t := 1.5
def CI := 302.98

-- Theorem to prove the principal amount matches the correct computation
theorem find_principal_amount : 
  ∃ P : ℝ, compoundInterest P r n t = CI ∧ P = 4950 :=
by
  sorry

end find_principal_amount_l36_36737


namespace clara_climbs_stone_blocks_l36_36287

-- Define the number of steps per level
def steps_per_level : Nat := 8

-- Define the number of blocks per step
def blocks_per_step : Nat := 3

-- Define the number of levels in the tower
def levels : Nat := 4

-- Define a function to compute the total number of blocks given the constants
def total_blocks (steps_per_level blocks_per_step levels : Nat) : Nat :=
  steps_per_level * blocks_per_step * levels

-- Statement of the theorem
theorem clara_climbs_stone_blocks :
  total_blocks steps_per_level blocks_per_step levels = 96 :=
by
  -- Lean requires 'sorry' as a placeholder for the proof.
  sorry

end clara_climbs_stone_blocks_l36_36287


namespace factorize_expression_l36_36728

theorem factorize_expression (a b : ℝ) : a^2 + a * b = a * (a + b) := 
by
  sorry

end factorize_expression_l36_36728


namespace swapped_coefficients_have_roots_l36_36379

theorem swapped_coefficients_have_roots 
  (a b c p q r : ℝ)
  (h1 : ∀ x : ℝ, ¬ (a * x^2 + b * x + c = 0))
  (h2 : ∀ x : ℝ, ¬ (p * x^2 + q * x + r = 0))
  (h3 : b^2 < 4 * p * c)
  (h4 : q^2 < 4 * a * r) :
  ∃ x : ℝ, a * x^2 + q * x + c = 0 ∧ ∃ y : ℝ, p * y^2 + b * y + r = 0 :=
by
  sorry

end swapped_coefficients_have_roots_l36_36379


namespace probability_divisor_of_12_l36_36640

theorem probability_divisor_of_12 : 
  (nat.factorization 12).keys = {1, 2, 3, 4, 6, 12} → 
  (∃ die_side, die_side ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) → 
  (∃ favorable_outcomes, favorable_outcomes ∈ {1, 2, 3, 4, 6, 12}) → 
  ∃ probability, probability = (favorable_outcomes.card : ℚ) / (die_side.card : ℚ) ∧ probability = 1/2 :=
by 
  sorry

end probability_divisor_of_12_l36_36640


namespace min_major_axis_ellipse_l36_36777

theorem min_major_axis_ellipse (a b c : ℝ) (h1 : b * c = 1) (h2 : a^2 = b^2 + c^2) :
  2 * a ≥ 2 * Real.sqrt 2 :=
by {
  sorry
}

end min_major_axis_ellipse_l36_36777


namespace proof_problem1_proof_problem2_l36_36285

noncomputable def problem1 (sqrt3 : ℝ) : ℝ := (sqrt3) ^ 2 + abs (1 - sqrt3) + (-27) ^ (1 / 3)
noncomputable def problem2 (sqrt12 sqrt13 sqrt6 : ℝ) : ℝ := (sqrt12 - sqrt13) * sqrt6

theorem proof_problem1 (sqrt3 : ℝ) (h_sqrt3 : sqrt3 = real.sqrt 3) :
 problem1 sqrt3 = real.sqrt 3 - 1 :=
by {
  rw [problem1, h_sqrt3],
  simp,
  sorry
}

theorem proof_problem2 (sqrt12 sqrt13 sqrt6 : ℝ) (h_sqrt12 : sqrt12 = real.sqrt 12) (h_sqrt13 : sqrt13 = real.sqrt (1/3)) (h_sqrt6 : sqrt6 = real.sqrt 6) :
 problem2 sqrt12 sqrt13 sqrt6 = 5 * real.sqrt 2 :=
by {
  rw [problem2, h_sqrt12, h_sqrt13, h_sqrt6],
  simp,
  sorry
}

end proof_problem1_proof_problem2_l36_36285


namespace hyperbola_maximum_eccentricity_l36_36870

noncomputable def maximum_eccentricity_of_hyperbola (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) : ℝ :=
  let e := (λ a c : ℝ, c / a)
  in Sup (set_of (λ e, ∃ (c n : ℝ), e = c / a ∧ 
  (n ≠ 0) ∧ -- To ensure the inequality holds
  (c^2 - a^2 ≥ 0) ∧  -- from the equation c^2 - a^2 ≥ 0
  (sqrt 3 ≤ a / sqrt (c^2 - a^2))))
sorry 

-- Main theorem
theorem hyperbola_maximum_eccentricity {a b : ℝ} (h₁ : a > 0) (h₂ : b > 0) :
  maximum_eccentricity_of_hyperbola a b h₁ h₂ = 2 * sqrt 3 / 3 := 
sorry

end hyperbola_maximum_eccentricity_l36_36870


namespace ramsey_upper_bound_l36_36326

-- Definitions and assumptions
variable {G : Type*} [Graph G]

-- Definitions for Δ(H), |H|, and R(H)
def delta (H : G) : ℕ := sorry -- Maximum degree Δ(H)
def size (H : G) : ℕ := sorry -- Size |H|
def Ramsey_number (H : G) : ℕ := sorry -- Ramsey number R(H)

theorem ramsey_upper_bound (Δ : ℕ) (h_pos : Δ > 0) :
  ∃ c : ℝ, ∀ H : G, delta H ≤ Δ → Ramsey_number H ≤ c * size H :=
sorry

end ramsey_upper_bound_l36_36326


namespace Cartesian_eq_C2_correct_distance_AB_correct_l36_36159

-- Part I: Proving the Cartesian equation of curve (C2)
noncomputable def equation_of_C2 (x y : ℝ) (α : ℝ) : Prop :=
  x = 4 * Real.cos α ∧ y = 4 + 4 * Real.sin α

def Cartesian_eq_C2 (x y : ℝ) : Prop :=
  x^2 + (y - 4)^2 = 16

theorem Cartesian_eq_C2_correct (x y α : ℝ) (h : equation_of_C2 x y α) : Cartesian_eq_C2 x y :=
by sorry

-- Part II: Proving the distance |AB| given polar equations
noncomputable def polar_eq_C1 (theta : ℝ) : ℝ :=
  4 * Real.sin theta

noncomputable def polar_eq_C2 (theta : ℝ) : ℝ :=
  8 * Real.sin theta

def distance_AB (rho1 rho2 : ℝ) : ℝ :=
  abs (rho1 - rho2)

theorem distance_AB_correct : distance_AB (polar_eq_C1 (π / 3)) (polar_eq_C2 (π / 3)) = 2 * Real.sqrt 3 :=
by sorry

end Cartesian_eq_C2_correct_distance_AB_correct_l36_36159


namespace part1_optimal_scheme_part2_range_p_l36_36505

theorem part1_optimal_scheme (p : ℝ) (h : p = 2 * Real.sqrt 2 / 3) (h0 : 0 < p) (h1 : p < 1) :
  let scheme_1_tests := 4 in
  let scheme_2_tests := 2 * (64 / 81) + 4 * (16 / 81) + 6 * (1 / 81) in
  let scheme_4_tests := 1 * (64 / 81) + 5 * (17 / 81) in
  scheme_4_tests < scheme_2_tests ∧ scheme_2_tests < scheme_1_tests := sorry

theorem part2_range_p (p : ℝ) (h0 : 0 < p) (h1 : p < 1) :
  (let scheme_3_tests := 5 - 3 * p^3 in
   let scheme_4_tests := 5 - 4 * p^4 in
   scheme_3_tests < scheme_4_tests) ↔ p < 3 / 4 := sorry

end part1_optimal_scheme_part2_range_p_l36_36505


namespace deceased_member_income_l36_36922

theorem deceased_member_income (A B C : ℝ) (h1 : (A + B + C) / 3 = 735) (h2 : (A + B) / 2 = 650) : 
  C = 905 :=
by
  sorry

end deceased_member_income_l36_36922


namespace find_alpha_l36_36335

theorem find_alpha (α : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π)
  (h2 : ∃ x : ℝ, (∃ x, x^2 - 2 * x + 3 = 2) ∧ 
       (∀ y : ℝ, (∃ k : ℝ, k = x^2 - 2 * y + 3) → 
                 ((sin α) ^ k ≤ (1 / 4))) ∧ 
      (∀ y : ℝ, (∃ k : ℝ, k = x^2 - 2 * y + 3) → 
                 ((sin α) ^ k = (1 / 4)))):
  α = 5 * π / 6 :=
by
  sorry

end find_alpha_l36_36335


namespace problem_I_problem_II_l36_36014

theorem problem_I (a b c A B C : ℝ) (h1 : a ≠ 0) (h2 : 2 * a - a * Real.cos B = b * Real.cos A) :
  c / a = 2 :=
sorry

theorem problem_II (a b c A B C : ℝ) (h1 : a ≠ 0) (h2 : 2 * a - a * Real.cos B = b * Real.cos A) 
  (h3 : b = 4) (h4 : Real.cos C = 1 / 4) :
  (1 / 2) * a * b * Real.sin C = Real.sqrt 15 :=
sorry

end problem_I_problem_II_l36_36014


namespace cannot_be_one_everywhere_l36_36325

def f : ℤ × ℤ → ℕ := sorry

axiom f_init : f (0, 0) = 4
axiom f_zero_elsewhere : ∀ (x y : ℤ), (x, y) ≠ (0, 0) → f (x, y) = 0

axiom seehang_condition : ∀ (x y : ℤ), f (x, y) ≥ 2 → 
  (∃ (a b c d : (ℤ × ℤ)), 
  a ∈ {(x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)} ∧
  b ∈ {(x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)} ∧
  c ∈ {(x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)} ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c)

theorem cannot_be_one_everywhere :
  ∃ (x y : ℤ), f (x, y) > 1 :=
sorry

end cannot_be_one_everywhere_l36_36325


namespace relationship_among_abc_l36_36337

noncomputable def a : ℝ := 2^(1.2)
noncomputable def b : ℝ := (1 / 2)^(-0.8)
noncomputable def c : ℝ := 2 * log 5 2

theorem relationship_among_abc : c < b ∧ b < a :=
by
  -- we'll just leave sorry for the actual proof
  sorry

end relationship_among_abc_l36_36337


namespace sin_and_cos_A_l36_36416

-- Definitions of the given conditions
def angle_C_is_right (C : ℝ) : Prop := C = 90
def length_AB (AB : ℝ) : Prop := AB = 15
def length_AC (AC : ℝ) : Prop := AC = 5

-- Definition of the main problem to be proved
theorem sin_and_cos_A (C A B : ℝ) [h : angle_C_is_right C] [h1 : length_AB AB] [h2 : length_AC AC] :
  sin A = 2*sqrt 2 / 3 ∧ cos A = 1 / 3 :=
by
  sorry

end sin_and_cos_A_l36_36416


namespace f_4_1981_l36_36508

noncomputable def f : ℕ × ℕ → ℕ
| (0, y) => y + 1
| (x + 1, 0) => f (x, 1)
| (x + 1, y + 1) => f (x, f (x + 1, y))

theorem f_4_1981 : f (4, 1981) = (2^2)^12 := 
sorry

end f_4_1981_l36_36508


namespace probability_of_divisor_of_12_is_one_half_l36_36631

theorem probability_of_divisor_of_12_is_one_half :
  let divisors := {1, 2, 3, 4, 6, 12} in
  let total_outcomes := finset.range 12 in
  (divisors.card : ℚ) / (total_outcomes.card : ℚ) = 1 / 2 :=
by
  let divisors := finset.from_list [1, 2, 3, 4, 6, 12]
  let total_outcomes := finset.range 12
  have h1 : divisors.card = 6 := by sorry
  have h2 : total_outcomes.card = 12 := by sorry
  rw [←h1, ←h2]
  norm_num
  sorry

end probability_of_divisor_of_12_is_one_half_l36_36631


namespace fraction_plays_one_instrument_l36_36820

theorem fraction_plays_one_instrument (total_people : ℕ) (play_two_or_more : ℕ) (prob_exactly_one : ℝ) 
  (h_total : total_people = 800) (h_play_two_or_more : play_two_or_more = 32) (h_prob_exactly_one : prob_exactly_one = 0.16) :
  (play_two_or_more + nat.floor (prob_exactly_one * total_people)) / total_people = 1 / 5 :=
by
  -- Placeholder for the proof steps
  sorry

end fraction_plays_one_instrument_l36_36820


namespace percentage_of_gross_profit_l36_36754

theorem percentage_of_gross_profit :
  let selling_price : ℝ := 28
  let wholesale_cost : ℝ := 24.35
  let gross_profit := selling_price - wholesale_cost
  let percentage_profit := (gross_profit / wholesale_cost) * 100
  percentage_profit ≈ 14.99 :=
by
  sorry

end percentage_of_gross_profit_l36_36754


namespace power_of_729_correct_l36_36281

noncomputable def power_of_729 : ℝ := 729 ^ (4 / 6)

theorem power_of_729_correct :
  729 = 3 ^ 6 →
  power_of_729 = 81 :=
by
  intro h
  unfold power_of_729
  rw [← h]
  simp [Real.rpow_def]
  sorry

end power_of_729_correct_l36_36281


namespace toy_truck_cost_is_correct_l36_36475

-- Define the initial amount, amount spent on the pencil case, and the final amount
def initial_amount : ℝ := 10
def pencil_case_cost : ℝ := 2
def final_amount : ℝ := 5

-- Define the amount spent on the toy truck
def toy_truck_cost : ℝ := initial_amount - pencil_case_cost - final_amount

-- Prove that the amount spent on the toy truck is 3 dollars
theorem toy_truck_cost_is_correct : toy_truck_cost = 3 := by
  sorry

end toy_truck_cost_is_correct_l36_36475


namespace integral_eq_k_implies_k_eq_2_l36_36036

theorem integral_eq_k_implies_k_eq_2 (k : ℝ) (h : ∫ x in 0..1, k * x + 1 = k) : k = 2 :=
by
  sorry

end integral_eq_k_implies_k_eq_2_l36_36036


namespace minimize_expression_l36_36057

noncomputable def f (x : ℝ) : ℝ := 2 + 3 * x + 4 / (x - 1)

theorem minimize_expression : ∃ x : ℝ, x > 1 ∧ (f x = f (2/3 * Real.sqrt 3 + 1)) :=
by
  use 2/3 * Real.sqrt 3 + 1
  split
  · show 2/3 * Real.sqrt 3 + 1 > 1
    sorry
  
  · show f (2/3 * Real.sqrt 3 + 1) = 2 + 3 * (2/3 * Real.sqrt 3 + 1) + 4 / ((2/3 * Real.sqrt 3 + 1) - 1)
    sorry

end minimize_expression_l36_36057


namespace hyperbola_eccentricity_l36_36866

-- Define the hyperbola with given conditions and prove its eccentricity
theorem hyperbola_eccentricity
  (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b)
  (P : ℝ × ℝ) (hP : P = (2 * a, 3 * a))
  (C : ℝ × ℝ → Prop)
  (hC : ∀ x y, C (x, y) ↔ (x^2 / a^2 - y^2 / b^2 = 1))
  : ∃ e : ℝ, e = 2 := 
sorry

end hyperbola_eccentricity_l36_36866


namespace num_distinct_solutions_l36_36713

theorem num_distinct_solutions : 
  (∃ x : ℝ, |x - 3| = |x + 5|) ∧ 
  (∀ x1 x2 : ℝ, |x1 - 3| = |x1 + 5| → |x2 - 3| = |x2 + 5| → x1 = x2) := 
  sorry

end num_distinct_solutions_l36_36713


namespace all_chameleons_become_green_l36_36142

def Chameleon :=
  {yellow : Nat, red : Nat, green : Nat}

def initialChameleonPopulation : Chameleon :=
  {yellow := 7, red := 10, green := 17}

-- A function to determine the resulting difference invariant modulo 3
def difference_invariant (c : Chameleon) : Int :=
  (c.yellow - c.red) % 3

theorem all_chameleons_become_green (c : Chameleon) :
  (c.yellow = 34 ∨ c.red = 34 ∨ c.green = 34) → c = {yellow := 0, red := 0, green := 34} := sorry

end all_chameleons_become_green_l36_36142


namespace length_of_side_d_l36_36488

variable (a b c d : ℕ)
variable (h_ratio1 : a / c = 3 / 4)
variable (h_ratio2 : b / d = 3 / 4)
variable (h_a : a = 3)
variable (h_b : b = 6)

theorem length_of_side_d (a b c d : ℕ)
  (h_ratio1 : a / c = 3 / 4)
  (h_ratio2 : b / d = 3 / 4)
  (h_a : a = 3)
  (h_b : b = 6) : d = 8 := 
sorry

end length_of_side_d_l36_36488


namespace relatively_prime_count_l36_36114

def f (x : ℕ) := 2^x + 3^x

theorem relatively_prime_count:
  (finset.filter (λ n, ∀ k < n, nat.gcd (f n) (f k) = 1) (finset.range 2021)).card = 11 :=
sorry

end relatively_prime_count_l36_36114


namespace cost_per_crayon_l36_36436

-- Definitions for conditions
def half_dozen := 6
def total_crayons := 4 * half_dozen
def total_cost := 48

-- Problem statement
theorem cost_per_crayon :
  (total_cost / total_crayons) = 2 := 
  by
    sorry

end cost_per_crayon_l36_36436


namespace james_total_price_l36_36437

noncomputable def calculate_total_price : Real :=
  let CD1_D := 10 - (10 * 0.10)
  let CD2_D := 10 - (10 * 0.10)
  let CD3_D := 15 - (15 * 0.10)
  let CD4_D := 6 - (6 * 0.10)
  let CD5_D := 18 - (18 * 0.10)
  CD1_D + CD2_D + CD3_D + CD4_D + CD5_D

theorem james_total_price : calculate_total_price = 53.10 :=
by
  have CD1_D : Real := 10 - (10 * 0.10)
  have CD2_D : Real := 10 - (10 * 0.10)
  have CD3_D : Real := 15 - (15 * 0.10)
  have CD4_D : Real := 6 - (6 * 0.10)
  have CD5_D : Real := 18 - (18 * 0.10)
  have total_price : Real := CD1_D + CD2_D + CD3_D + CD4_D + CD5_D
  show total_price = 53.10
  sorry

end james_total_price_l36_36437


namespace probability_divisor_of_12_on_12sided_die_l36_36610

theorem probability_divisor_of_12_on_12sided_die (h : ∀ i, 1 ≤ i ∧ i ≤ 12 → true) : 
  (probability (fun (n : ℕ) => n ∣ 12 ∧ 1 ≤ n ∧ n ≤ 12)) = 1/2 := 
sorry

end probability_divisor_of_12_on_12sided_die_l36_36610


namespace new_mean_when_adding_const_to_each_number_l36_36841

theorem new_mean_when_adding_const_to_each_number :
  ∀ (numbers : Fin 15 → ℝ) (m : ℝ),
    (m = (∑ i, numbers i) / 15) →
    m = 40 →
    (∑ i, (numbers i + 10)) / 15 = 50 :=
by
  intros numbers m hm hmean
  sorry

end new_mean_when_adding_const_to_each_number_l36_36841


namespace cycling_trip_distance_l36_36267

def total_distance_trip (x : ℝ) : Prop :=
  let first_day := (1/3) * x - 4
  let remaining_day1 := x - first_day
  let second_day := (1/2) * remaining_day1 - 2
  let remaining_day2 := remaining_day1 - second_day
  let third_day := (10/11) * remaining_day2 + 4
  first_day + second_day + third_day = x

theorem cycling_trip_distance 
  (x = 132) 
  (first_day_distance = 40) 
  (second_day_distance = 46) 
  (third_day_distance = 44)
  : total_distance_trip 132 :=
sorry

end cycling_trip_distance_l36_36267


namespace distinct_real_roots_l36_36716

-- Define the polynomial equation as a Lean function
def polynomial (a x : ℝ) : ℝ :=
  (a + 1) * (x ^ 2 + 1) ^ 2 - (2 * a + 3) * (x ^ 2 + 1) * x + (a + 2) * x ^ 2

-- The theorem we need to prove
theorem distinct_real_roots (a : ℝ) : 
  (∃ (x y : ℝ), x ≠ y ∧ polynomial a x = 0 ∧ polynomial a y = 0) ↔ a ≠ -1 :=
by
  sorry

end distinct_real_roots_l36_36716


namespace area_of_overlap_l36_36546

theorem area_of_overlap (w θ : ℝ) (hw : w = 2) (hθ : θ ≠ 0 ∧ θ ≠ π / 2) :
  let overlap_area := 8 * sin θ
  in overlap_area = 8 * sin θ :=
by
  sorry

end area_of_overlap_l36_36546


namespace net_difference_is_30_l36_36526

-- Given conditions
variables 
  (P Q : ℝ) -- Original price and quantity 
  (budget : ℝ := 150) -- Customer's budget
  (price_increase : ℝ := 0.25) -- Price increase percentage
  (quantity_percentage : ℝ := 0.64) -- Percentage of quantity actually bought

-- The net difference in expenditure
def net_difference_in_expenditure (P Q : ℝ) : ℝ :=
  let new_price := P * (1 + price_increase)
  let actual_quantity := Q * quantity_percentage
  let original_expenditure := P * Q
  let new_expenditure := new_price * actual_quantity
  original_expenditure - new_expenditure

-- Prove that the net difference in expenditure is 30 dollars
theorem net_difference_is_30 : 
  ∀ (P Q : ℝ), P * Q = budget → net_difference_in_expenditure P Q = 30 :=
by
  intros P Q h
  unfold net_difference_in_expenditure
  simp [h, price_increase, quantity_percentage]
  sorry

end net_difference_is_30_l36_36526


namespace f_expression_g_extreme_values_l36_36825

/-- Define the function f -/
def f (x : ℝ) : ℝ := 2 * Real.sin x

/-- Define the function g based on the transformation described -/
def g (x : ℝ) : ℝ := 2 * Real.sin ((1/2) * x + Real.pi / 3)

/-- Prove the analytical expression for f(x) -/
theorem f_expression (x : ℝ) : f(x) = 2 * Real.sin x :=
by
  rfl

/-- Prove the maximum and minimum values of g (x) on [0, 2π] -/
theorem g_extreme_values :
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), g x ≤ 2) ∧ g (Real.pi / 3) = 2 ∧
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), -sqrt 3 ≤ g x) ∧ g (2 * Real.pi) = -sqrt 3 :=
by
  sorry

end f_expression_g_extreme_values_l36_36825


namespace sum_fractions_bound_l36_36454

theorem sum_fractions_bound (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) :
  let S := (a / (a + b + d)) + (b / (b + c + a)) + (c / (c + d + b)) + (d / (d + a + c))
  in 1 < S ∧ S < 2 :=
by
  let S := (a / (a + b + d)) + (b / (b + c + a)) + (c / (c + d + b)) + (d / (d + a + c))
  sorry

end sum_fractions_bound_l36_36454


namespace centroid_triangle_PQR_l36_36211

theorem centroid_triangle_PQR (P Q R S : ℝ × ℝ) 
  (P_coord : P = (2, 5)) 
  (Q_coord : Q = (9, 3)) 
  (R_coord : R = (4, -4))
  (S_is_centroid : S = (
    (P.1 + Q.1 + R.1) / 3,
    (P.2 + Q.2 + R.2) / 3)) :
  9 * S.1 + 4 * S.2 = 151 / 3 :=
by
  sorry

end centroid_triangle_PQR_l36_36211


namespace sequence_50th_term_is_55_l36_36438

def sequence : Nat → Nat
| 0 => 100
| n + 1 => if sequence n < 20 then sequence n * 7
           else if sequence n % 2 == 0 then sequence n / 4
           else sequence n - 10

theorem sequence_50th_term_is_55 : sequence 50 = 55 :=
sorry

end sequence_50th_term_is_55_l36_36438


namespace time_to_empty_l36_36569

-- Definitions based on the given conditions
def initial_fill : ℝ := 1 / 5
def rate_fill_A : ℝ := 1 / 10
def rate_empty_B : ℝ := 1 / 6
def net_rate : ℝ := rate_fill_A - rate_empty_B

-- Theorem to be proved
theorem time_to_empty (initial_fill > 0) (net_rate < 0) : initial_fill / -net_rate = 3 := by
  have net_rate_eq : net_rate = -1/15 := by
    unfold net_rate
    have common_denom : rate_fill_A - rate_empty_B = 3 / 30 - 5 / 30 := by
      norm_num [rate_fill_A, rate_empty_B]
    rw [common_denom]
    norm_num
  rw [net_rate_eq]
  norm_num [initial_fill]
  field_simp
  norm_num
  sorry

end time_to_empty_l36_36569


namespace ellipse_equation_quadrilateral_max_area_l36_36008

-- Part (Ⅰ): Finding the Equation of the Ellipse Γ
theorem ellipse_equation (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > 0) 
  (h4 : c = a * sqrt (3/4)) (h5 : a^2 = b^2 + c^2) : 
  ∃ x y : ℝ, (x^2 / 4) + y^2 = 1 := 
begin
  sorry
end

-- Part (Ⅱ): Maximum Area of Quadrilateral ABCD
theorem quadrilateral_max_area (m : ℝ) :
  (∀ x y : ℝ, y = 1/2 * x + m → 
    ellipse_equation 2 1 (sqrt 3) 
    → ∀ A B C D : ℝ × ℝ, 
      (∃ a b : ℝ, (A = (a, 1/2 * a + m)) ∧ (B = (b, 1/2 * b + m)) 
        ∧ (C = (-a, -1/2 * a - m)) ∧ (D = (-b, -1/2 * b - m)))
      → (4 : ℝ)) := 
begin
  sorry
end

end ellipse_equation_quadrilateral_max_area_l36_36008


namespace sum_powers_mod_7_l36_36062

theorem sum_powers_mod_7 :
  (1^6 + 2^6 + 3^6 + 4^6 + 5^6 + 6^6) % 7 = 6 := by
  sorry

end sum_powers_mod_7_l36_36062


namespace probability_divisor_of_12_is_half_l36_36623

noncomputable def divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ d, n % d = 0) (Finset.range (n + 1))

def probability_divisor_of_12 : ℚ :=
  let total_outcomes := 12
  let favorable_outcomes := (divisors 12).card
  favorable_outcomes / total_outcomes

theorem probability_divisor_of_12_is_half :
  probability_divisor_of_12 = 1 / 2 :=
  by
    sorry

end probability_divisor_of_12_is_half_l36_36623


namespace f_neither_even_nor_odd_l36_36835

def f (x : ℝ) : ℝ := ⌈x⌉ + (1 / 3)

theorem f_neither_even_nor_odd : ¬((∀ x : ℝ, f (-x) = f x) ∨ (∀ x : ℝ, f (-x) = -f x)) :=
by
  sorry

end f_neither_even_nor_odd_l36_36835


namespace sport_formulation_water_l36_36101

theorem sport_formulation_water (corn_syrup_ounces : ℕ) (h_cs : corn_syrup_ounces = 3) : 
  ∃ water_ounces : ℕ, water_ounces = 45 :=
by
  -- The ratios for the "sport" formulation: Flavoring : Corn Syrup : Water = 1 : 4 : 60
  let flavoring_ratio := 1
  let corn_syrup_ratio := 4
  let water_ratio := 60
  -- The given corn syrup is 3 ounces which corresponds to corn_syrup_ratio parts
  have h_ratio : corn_syrup_ratio = 4 := rfl
  have h_flavoring_to_corn_syrup : flavoring_ratio / corn_syrup_ratio = 1 / 4 := by sorry
  have h_flavoring_to_water : flavoring_ratio / water_ratio = 1 / 60 := by sorry
  -- Set up the proportion
  have h_proportion : corn_syrup_ratio / corn_syrup_ounces = water_ratio / 45 := by sorry 
  -- Cross-multiply to solve for the water
  have h_cross_mul : 4 * 45 = 3 * 60 := by sorry
  exact ⟨45, rfl⟩

end sport_formulation_water_l36_36101


namespace gabrielle_saw_more_birds_l36_36223

def birds_seen (robins cardinals blue_jays : Nat) : Nat :=
  robins + cardinals + blue_jays

def percentage_difference (g c : Nat) : Nat :=
  ((g - c) * 100) / c

theorem gabrielle_saw_more_birds :
  let gabrielle := birds_seen 5 4 3
  let chase := birds_seen 2 5 3
  percentage_difference gabrielle chase = 20 := 
by
  sorry

end gabrielle_saw_more_birds_l36_36223


namespace coeff_x3_in_expansion_of_x_plus_1_50_l36_36086

theorem coeff_x3_in_expansion_of_x_plus_1_50 :
  (Finset.range 51).sum (λ k => Nat.choose 50 k * (1 : ℕ) ^ (50 - k) * k ^ 3) = 19600 := by
  sorry

end coeff_x3_in_expansion_of_x_plus_1_50_l36_36086


namespace sum_g_div_eq_432_l36_36457

noncomputable def g (n : ℕ) : ℤ :=
  let approx := Int.ofReal (Real.cbrt n)
  if (Real.cbrt n - approx) < 0.5 then approx else approx + 1

theorem sum_g_div_eq_432 : 
  ∑ k in Finset.range (4095 + 1), (1 : ℚ) / g k = 432 := 
sorry

end sum_g_div_eq_432_l36_36457


namespace probability_divisor_of_12_l36_36604

theorem probability_divisor_of_12 (fair_die : True) (sides : Nat) (h : sides = 12) : 
  ∃ p : ℚ, p = 1 / 2 ∧ ∀ roll : Fin sides, is_divisor roll.val 12 ↔ roll.val ∈ {1, 2, 3, 4, 6, 12} → p = 1 / 2 :=
by sorry

def is_divisor (a b : Nat) : Prop := b % a = 0

end probability_divisor_of_12_l36_36604


namespace monotonic_increasing_interval_l36_36517

-- Main theorem to prove
theorem monotonic_increasing_interval 
  (k : ℤ) : 
  ∀ x, (kπ - 5π/8 ≤ x ∧ x ≤ kπ - π/8) ↔ (kπ + 3π/8 ≤ x ∧ x ≤ kπ + 7π/8) := 
sorry

end monotonic_increasing_interval_l36_36517


namespace find_m_l36_36428

def numberWall (m: ℕ) : Prop :=
  let layer1 := [m, 4, 10, 9]
  let layer2 := [m + 4, 4 + 10, 10 + 9]
  let layer3 := [(m + 4) + (4 + 10), (4 + 10) + (10 + 9)]
  let layer4 := [(m + 4 + 14) + (14 + 19)]
  layer4.head = 52

theorem find_m : ∃ m: ℕ, numberWall m ∧ m = 1 :=
by
  use 1
  sorry

end find_m_l36_36428


namespace incorrect_statement_D_l36_36884

theorem incorrect_statement_D (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a ≠ b) :
  (a + b)^3 > 0.5 * (a^3 + b^3) :=
begin
  -- We aim to prove this statement is incorrect
  -- Expanding (a + b)^3 and comparing with 0.5 * (a^3 + b^3)
  sorry
end

end incorrect_statement_D_l36_36884


namespace bob_deli_total_cost_l36_36278

-- Definitions based on the problem's conditions
def sandwich_cost : ℕ := 5
def soda_cost : ℕ := 3
def num_sandwiches : ℕ := 7
def num_sodas : ℕ := 10
def discount_threshold : ℕ := 50
def discount_amount : ℕ := 10

-- The total initial cost without discount
def initial_total_cost : ℕ :=
  (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost)

-- The final cost after applying discount if applicable
def final_cost : ℕ :=
  if initial_total_cost > discount_threshold then
    initial_total_cost - discount_amount
  else
    initial_total_cost

-- Statement to prove
theorem bob_deli_total_cost : final_cost = 55 := by
  sorry

end bob_deli_total_cost_l36_36278


namespace escher_probability_l36_36047

def num_arrangements (n : ℕ) : ℕ := Nat.factorial n

def favorable_arrangements (total_art : ℕ) (escher_prints : ℕ) : ℕ :=
  num_arrangements (total_art - escher_prints + 1) * num_arrangements escher_prints

def total_arrangements (total_art : ℕ) : ℕ :=
  num_arrangements total_art

def prob_all_escher_consecutive (total_art : ℕ) (escher_prints : ℕ) : ℚ :=
  favorable_arrangements total_art escher_prints / total_arrangements total_art

theorem escher_probability :
  prob_all_escher_consecutive 12 4 = 1/55 :=
by
  sorry

end escher_probability_l36_36047


namespace a0_a2_a4_sum_l36_36391

theorem a0_a2_a4_sum (a0 a1 a2 a3 a4 a5 : ℝ) :
  (∀ x : ℝ, (2 * x - 3) ^ 5 = a0 + a1 * (x - 1) + a2 * (x - 1) ^ 2 + a3 * (x - 1) ^ 3 + a4 * (x - 1) ^ 4 + a5 * (x - 1) ^ 5) →
  a0 + a2 + a4 = -121 :=
by
  intros h
  sorry

end a0_a2_a4_sum_l36_36391


namespace arithmetic_sum_2015_l36_36076

-- Definitions based on problem conditions
def a1 : ℤ := -2015
def S (n : ℕ) (d : ℤ) : ℤ := n * a1 + n * (n - 1) / 2 * d
def arithmetic_sequence (n : ℕ) (d : ℤ) : ℤ := a1 + (n - 1) * d

-- Proof problem
theorem arithmetic_sum_2015 (d : ℤ) :
  2 * S 6 d - 3 * S 4 d = 24 →
  S 2015 d = -2015 :=
by
  sorry

end arithmetic_sum_2015_l36_36076


namespace total_ridges_on_all_records_l36_36204

theorem total_ridges_on_all_records :
  let ridges_per_record := 60
  let cases := 4
  let shelves_per_case := 3
  let records_per_shelf := 20
  let shelf_fullness_ratio := 0.60

  let total_capacity := cases * shelves_per_case * records_per_shelf
  let actual_records := total_capacity * shelf_fullness_ratio
  let total_ridges := actual_records * ridges_per_record
  
  total_ridges = 8640 :=
by
  sorry

end total_ridges_on_all_records_l36_36204


namespace id_eq_ib_l36_36853

variables {A B C D I O : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space I] [metric_space O]

-- Definitions of segments and relevant points
variables (s : ℝ) -- length of the sides of the equilateral triangle
variable (R : ℝ) -- radius of the circumcircle of the triangle

-- Assume we have an equilateral triangle ABC
def equilateral_triangle (A B C : Type*) : Prop := 
  dist A B = s ∧ dist B C = s ∧ dist C A = s ∧
  ∠ABC = π / 3 ∧ ∠BCA = π / 3 ∧ ∠CAB = π / 3

-- O is the circumcenter of triangle ABC, on which D also lies on the circumcircle with AD > AB
def circumcenter (O : Type*) (A B C : Type*) : Prop :=
  dist O A = R ∧ dist O B = R ∧ dist O C = R ∧ 
  dist O D = R ∧ ∀ A (AB : A < B) (D : A < D), dist A D > dist A B

-- I is the incenter of the equilateral triangle ABC
def incenter (I : Type*) (A B C : Type*) : Prop :=
  dist I A = dist I B = dist I C ∧ ∀ p : Type*, dist I p = s / 2

-- Proof statement: Given the conditions, we need to prove ID = IB
theorem id_eq_ib 
  (A B C D I O : Type*) 
  (s : ℝ) (R : ℝ)
  (h1 : equilateral_triangle A B C)
  (h2 : circumcenter O A B C)
  (h3 : incenter I A B C) :
  dist I D = dist I B := 
sorry

end id_eq_ib_l36_36853


namespace product_XE_ZE_l36_36212

noncomputable def triangle_XYZ_and_point_Q (X Y Z Q E : point ℝ) : Prop :=
  in_same_plane [X, Y, Z, Q] ∧
  dist Q X = dist Q Y ∧
  ∠X Q Y = 3 * ∠ X Z Y ∧
  intersects (line_through X Z) (line_through Y Q) E ∧
  dist Y Q = 5 ∧
  dist Y E = 1

theorem product_XE_ZE (X Y Z Q E : point ℝ) (h : triangle_XYZ_and_point_Q X Y Z Q E) :
  (dist X E) * (dist Z E) = 9 :=
sorry

end product_XE_ZE_l36_36212


namespace probability_adjacent_vertices_dodecagon_l36_36981

noncomputable def prob_adjacent_vertices_dodecagon : ℚ :=
  let total_vertices := 12
  let favorable_outcomes := 2  -- adjacent vertices per chosen vertex
  let total_outcomes := total_vertices - 1  -- choosing any other vertex
  favorable_outcomes / total_outcomes

theorem probability_adjacent_vertices_dodecagon :
  prob_adjacent_vertices_dodecagon = 2 / 11 := by
  sorry

end probability_adjacent_vertices_dodecagon_l36_36981


namespace johnny_walked_distance_l36_36144

noncomputable def distance_johnny_walked
  (total_distance : ℕ) (matthew_rate : ℕ) (johnny_rate : ℕ) (emily_rate : ℕ) (matthew_start_hour : ℕ): ℕ :=
  let t := (total_distance - matthew_rate * (matthew_start_hour + 1)) / (matthew_rate + johnny_rate + emily_rate) in
  johnny_rate * t

theorem johnny_walked_distance :
  distance_johnny_walked 60 3 4 5 1 = 19 :=
by sorry

end johnny_walked_distance_l36_36144


namespace g_value_at_2_over_9_l36_36711

theorem g_value_at_2_over_9 (g : ℝ → ℝ) 
  (hg0 : g 0 = 0)
  (hgmono : ∀ ⦃x y⦄, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y)
  (hg_symm : ∀ x, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x)
  (hg_frac : ∀ x, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3) :
  g (2 / 9) = 8 / 27 :=
sorry

end g_value_at_2_over_9_l36_36711


namespace find_positive_square_root_l36_36395

theorem find_positive_square_root (x : ℝ) (h_pos : x > 0) (h_eq : x^2 = 625) : x = 25 :=
sorry

end find_positive_square_root_l36_36395


namespace complex_transformation_l36_36990

open Complex

def dilation (z : ℂ) (center : ℂ) (scale : ℝ) : ℂ :=
  center + scale * (z - center)

def rotation90 (z : ℂ) : ℂ :=
  z * I

theorem complex_transformation (z : ℂ) (center : ℂ) (scale : ℝ) :
  center = -1 + 2 * I → scale = 2 → z = 3 + I →
  rotation90 (dilation z center scale) = 4 + 7 * I :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  dsimp [dilation]
  dsimp [rotation90]
  sorry

end complex_transformation_l36_36990


namespace collatz_eighth_term_one_l36_36816

def collatz_step (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 3 * n + 1

def collatz_sequence : ℕ → ℕ → ℕ
| 0, n := n
| (k+1), n := collatz_sequence k (collatz_step n)

theorem collatz_eighth_term_one (n : ℕ) :
  collatz_sequence 7 n = 1 ↔ n ∈ {2, 3, 16, 20, 21, 128} :=
sorry

end collatz_eighth_term_one_l36_36816


namespace retail_price_l36_36262

noncomputable def original_wholesale_cost : Real := 80 / 1.05

noncomputable def final_selling_price (R : Real) : Real := 0.90 * R * 1.15

noncomputable def profit (F : Real) : Real := 0.20 * F

theorem retail_price :
  ∃ R : Real, 
  (let F := final_selling_price R in 
  profit F = F - 90) ∧ 
  R = 108.70 :=
by
  sorry

end retail_price_l36_36262


namespace obtuse_angle_probability_l36_36147

-- Defining the vertices of the pentagon
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨0, 3⟩
def B : Point := ⟨5, 0⟩
def C : Point := ⟨8, 0⟩
def D : Point := ⟨8, 5⟩
def E : Point := ⟨0, 5⟩

def is_interior (P : Point) : Prop :=
  -- A condition to define if a point is inside the pentagon
  sorry

def is_obtuse_angle (A B P : Point) : Prop :=
  -- Condition for angle APB to be obtuse
  sorry

noncomputable def probability_obtuse_angle :=
  -- Probability calculation
  let area_pentagon := 40
  let area_circle := (34 * Real.pi) / 4
  let area_outside_circle := area_pentagon - area_circle
  area_outside_circle / area_pentagon

theorem obtuse_angle_probability :
  ∀ P : Point, is_interior P → ∃! p : ℝ, p = (160 - 34 * Real.pi) / 160 :=
sorry

end obtuse_angle_probability_l36_36147


namespace length_of_base_of_isosceles_triangle_l36_36815

theorem length_of_base_of_isosceles_triangle :
  ∀ (A B C D E : Type) [metric_space A] [metric_space B] [metric_space C]
  [metric_space D] [metric_space E] (h_isosceles : dist A B = dist A C)
  (h_trisect : ∠BAC)
  (h_BD : dist B D = 3) (h_DE : dist D E = 2) (h_EC : dist E C = 4),
  dist B C = 27 :=
begin
  sorry
end

end length_of_base_of_isosceles_triangle_l36_36815


namespace complex_series_sum_l36_36706

noncomputable def i : ℂ := complex.I

theorem complex_series_sum : 
  (1 + i + i^2 + i^3 + ∑ n in finset.range 2011, i^n) = 0 :=
by
  sorry

end complex_series_sum_l36_36706


namespace train_passes_platform_in_35_seconds_l36_36659

noncomputable def speed_kmph_to_mps (v_kmph : ℝ) : ℝ :=
  v_kmph * 1000 / 3600

noncomputable def length_of_train (v_mps : ℝ) (time_to_pass_man : ℕ) : ℝ :=
  v_mps * time_to_pass_man

noncomputable def total_distance (length_train : ℝ) (length_platform : ℝ) : ℝ :=
  length_train + length_platform

noncomputable def time_to_pass_platform (total_distance : ℝ) (speed : ℝ) : ℝ :=
  total_distance / speed

theorem train_passes_platform_in_35_seconds (time_to_pass_man : ℕ) (train_speed_kmph : ℝ) (platform_length : ℝ) :
  time_to_pass_platform (total_distance (length_of_train (speed_kmph_to_mps train_speed_kmph) time_to_pass_man) platform_length) (speed_kmph_to_mps train_speed_kmph) ≈ 35 := 
by
  sorry

-- Given conditions
def time_to_pass_man : ℕ := 20
def train_speed_kmph : ℝ := 54
def platform_length : ℝ := 225.018

-- Use the theorem to prove that the train passes the platform in approximately 35 seconds
#eval train_passes_platform_in_35_seconds time_to_pass_man train_speed_kmph platform_length

end train_passes_platform_in_35_seconds_l36_36659


namespace sequence_sum_l36_36582

open BigOperators

-- Define the general term
def term (n : ℕ) : ℚ := n * (1 - (1 / n))

-- Define the index range for the sequence
def index_range : Finset ℕ := Finset.range 9 \ {0, 1}

-- Lean statement of the problem
theorem sequence_sum : ∑ n in index_range, term (n + 2) = 45 := by
  sorry

end sequence_sum_l36_36582


namespace final_number_lt_one_l36_36480

theorem final_number_lt_one :
  ∀ (numbers : Finset ℕ),
    (numbers = Finset.range 3000 \ Finset.range 1000) →
    (∀ (a b : ℕ), a ∈ numbers → b ∈ numbers → a ≤ b →
    ∃ (numbers' : Finset ℕ), numbers' = (numbers \ {a, b}) ∪ {a / 2}) →
    ∃ (x : ℕ), x ∈ numbers ∧ x < 1 :=
by
  sorry

end final_number_lt_one_l36_36480


namespace probability_divisor_of_12_l36_36624

/-- Definition of divisors of a number -/
def divisors (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (λ d => n % d = 0)

theorem probability_divisor_of_12 :
  let divisors_of_12 := divisors 12
  Finset.card divisors_of_12 = 6 →
  (∃ (P : ℚ), P = 1/2) :=
by
  intro h
  exists 1/2
  sorry

end probability_divisor_of_12_l36_36624


namespace number_of_multiples_of_41_in_array_l36_36268

open Nat

-- Define the b(n, k) function as described
def b (n k : ℕ) : ℕ := 2^(n-1) * (n + 2 * k - 1)

-- Define the predicate that checks whether b(n, k) is a multiple of 41
def is_multiple_of_41 (n k : ℕ) : Prop := 41 ∣ b n k

-- Define the main theorem to prove the number of such (n, k) pairs
theorem number_of_multiples_of_41_in_array : 
  ∑ n in (range 60).filter odd, 
    ∑ k in (range (51 - n)).filter (λ k, is_multiple_of_41 n k) = 30
:= sorry

end number_of_multiples_of_41_in_array_l36_36268


namespace height_difference_l36_36279

theorem height_difference :
  let janet_height := 3.6666666666666665
  let sister_height := 2.3333333333333335
  janet_height - sister_height = 1.333333333333333 :=
by
  sorry

end height_difference_l36_36279


namespace determine_b_for_continuity_l36_36128

def g (x : ℝ) (b : ℝ) : ℝ :=
if x ≤ 2 then 
  2 * x^2 + 5 * x + 3 
else
  b * x + 7

theorem determine_b_for_continuity : 
  (∀ x : ℝ, g 2 b = g 2 7) → b = 7 :=
by
  sorry

end determine_b_for_continuity_l36_36128


namespace cube_root_simplify_l36_36961

theorem cube_root_simplify (a b : ℕ) (ha : a = 72) (hb : b = 36) : a + b = 108 :=
by 
  rw [ha, hb]
  exact rfl

end cube_root_simplify_l36_36961


namespace quadrilateral_similarity_l36_36068

noncomputable theory

theorem quadrilateral_similarity
  (A B C D A' B' C' D' : Type*)
  [quad_ABC : convex_quadrilateral A B C D]
  [perpendicular_A_BD : is_perpendicular_from_vertex_to_diagonal A B D A']
  [perpendicular_C_BD : is_perpendicular_from_vertex_to_diagonal C B D C']
  [perpendicular_B_AC : is_perpendicular_from_vertex_to_diagonal B A C B']
  [perpendicular_D_AC : is_perpendicular_from_vertex_to_diagonal D A C D']
  [cyclic_ADA'D' : is_cyclic_quadrilateral A D A' D']
  [cyclic_DCDC' : is_cyclic_quadrilateral D C D' C'] :
  similar_quadrilaterals A B C D A' B' C' D' :=
sorry

end quadrilateral_similarity_l36_36068


namespace distinct_remainders_mod_3n_l36_36124

open Nat

theorem distinct_remainders_mod_3n 
  (n : ℕ) 
  (hn_odd : Odd n)
  (ai : ℕ → ℕ)
  (bi : ℕ → ℕ)
  (ai_def : ∀ i, 1 ≤ i ∧ i ≤ n → ai i = 3*i - 2)
  (bi_def : ∀ i, 1 ≤ i ∧ i ≤ n → bi i = 3*i - 3)
  (k : ℕ) 
  (hk : 0 < k ∧ k < n)
  : ∀ i, 1 ≤ i ∧ i ≤ n → (∀ j, 1 ≤ j ∧ j ≤ n → i ≠ j →
     ∀ ⦃ r s t u v : ℕ ⦄, 
       (r = (ai i + ai (i % n + 1)) % (3*n) ∧ 
        s = (ai i + bi i) % (3*n) ∧ 
        t = (bi i + bi ((i + k) % n + 1)) % (3*n)) →
       r ≠ s ∧ s ≠ t ∧ t ≠ r) := 
sorry

end distinct_remainders_mod_3n_l36_36124


namespace find_k_l36_36099

theorem find_k 
  (c : ℝ) (a₁ : ℝ) (S : ℕ → ℝ) (k : ℝ)
  (h1 : ∀ n, S (n+1) = c * S n) 
  (h2 : S 1 = 3 + k)
  (h3 : ∀ n, S n = 3^n + k) :
  k = -1 :=
sorry

end find_k_l36_36099


namespace balls_in_boxes_l36_36583

theorem balls_in_boxes (n m : ℕ) (h : n ≥ m) : 
  (∑ k in finset.range (n - m + 1), nat.choose (m + k - 1) k) = nat.choose (n - 1) (m - 1) :=
begin
  sorry
end

end balls_in_boxes_l36_36583


namespace consistent_with_hyperbola_l36_36273

-- Definitions for the parameter equations
def eqA (t : ℝ) := (t^(0.5), t^(-0.5))
def eqB (t : ℝ) := (|t|, |t|^(-1))
def eqC (t : ℝ) := (Real.cos t, Real.sin t)
def eqD (t : ℝ) := (Real.tan t, Real.cos t / Real.sin t)

-- Main theorem statement
theorem consistent_with_hyperbola (t : ℝ) : 
  let (x, y) := eqD t in x * y = 1 := 
sorry

end consistent_with_hyperbola_l36_36273


namespace intersection_setMN_l36_36377

def setM := {x : ℝ | ∃ y : ℝ, y = log x}
def setN := {x : ℝ | ∃ y : ℝ, y = sqrt (1 - x^2)}

theorem intersection_setMN : 
  setM ∩ setN = {x : ℝ | 0 < x ∧ x ≤ 1} :=
sorry

end intersection_setMN_l36_36377


namespace angle_range_in_third_quadrant_l36_36042

theorem angle_range_in_third_quadrant (α : ℝ) (k : ℤ)
  (h1 : sin α + cos α < 0)
  (h2 : tan α > 0) :
  ∃ k : ℤ, α ∈ set.Ioo (2 * k * π + π) (2 * k * π + 3 * π / 2) :=
sorry

end angle_range_in_third_quadrant_l36_36042


namespace finite_set_of_non_sum_of_distinct_squares_l36_36346

theorem finite_set_of_non_sum_of_distinct_squares :
  ∀ (A : set ℕ), (∀ a ∈ A, ¬∃ (s : finset ℕ), a = s.sum (λ x, x^2)) → (set.finite A) :=
sorry

end finite_set_of_non_sum_of_distinct_squares_l36_36346


namespace points_per_member_l36_36269

theorem points_per_member
    (total_members : ℕ)
    (absent_members : ℕ)
    (total_points : ℕ)
    (present_members : ℕ)
    (points_per_member : ℕ)
    (h1 : total_members = 5)
    (h2 : absent_members = 2)
    (h3 : total_points = 18)
    (h4 : present_members = total_members - absent_members)
    (h5 : points_per_member = total_points / present_members) :
  points_per_member = 6 :=
by
  sorry

end points_per_member_l36_36269


namespace max_cars_per_div_100_is_20_l36_36477

theorem max_cars_per_div_100_is_20 :
  let m : ℕ := Nat.succ (Nat.succ 0) -- represents m going to infinity
  let car_length : ℕ := 5
  let speed_factor : ℕ := 10
  let sensor_distance_per_hour : ℕ := speed_factor * 1000 * m
  let separation_distance : ℕ := car_length * (m + 1)
  let max_cars : ℕ := (sensor_distance_per_hour / separation_distance) * m
  Nat.floor ((2 * (max_cars : ℝ)) / 100) = 20 :=
by
  sorry

end max_cars_per_div_100_is_20_l36_36477


namespace binom_coeff_divisibility_l36_36155

theorem binom_coeff_divisibility (a : ℕ) (h : a > 0) :
  (C (2 * a) a) % 2 = 0 ∧
  (C (2 * a) a) % (a + 1) = 0 ∧
  (C (2 * a) a) % (2 * a - 1) = 0 :=
by
  sorry

end binom_coeff_divisibility_l36_36155


namespace trigonometric_signs_in_fourth_quadrant_l36_36355

-- Define the problem's conditions and questions
theorem trigonometric_signs_in_fourth_quadrant 
  (k : ℤ) 
  (α θ : ℝ) 
  (hα : α = 2 * k * Real.pi - Real.pi / 5) 
  (hθ : θ = α + 2 * (Int.floorDiv (α / (2 * Real.pi + Real.pi / 5))) * Real.pi) : 
  (Real.sin θ / Real.abs (Real.sin θ) + Real.abs (Real.cos θ) / Real.cos θ + Real.tan θ / Real.abs (Real.tan θ)) = -1 := 
sorry

end trigonometric_signs_in_fourth_quadrant_l36_36355


namespace solve_equation_l36_36491

theorem solve_equation (x : ℝ) : 27 = 3 * (9 ^ (x - 1)) → x = 2 := by
  intro h
  sorry

end solve_equation_l36_36491


namespace probability_of_divisor_of_12_is_one_half_l36_36637

theorem probability_of_divisor_of_12_is_one_half :
  let divisors := {1, 2, 3, 4, 6, 12} in
  let total_outcomes := finset.range 12 in
  (divisors.card : ℚ) / (total_outcomes.card : ℚ) = 1 / 2 :=
by
  let divisors := finset.from_list [1, 2, 3, 4, 6, 12]
  let total_outcomes := finset.range 12
  have h1 : divisors.card = 6 := by sorry
  have h2 : total_outcomes.card = 12 := by sorry
  rw [←h1, ←h2]
  norm_num
  sorry

end probability_of_divisor_of_12_is_one_half_l36_36637


namespace parabola_ratio_l36_36993

theorem parabola_ratio (p : ℝ) (h : 0 < p) :
  let F := (p / 2, 0)
  -- Line is passing through focus F with an inclination of 60 degrees
  let m : ℝ := Real.tan (Real.pi / 3) -- 60 degrees inclination
  -- Line equation: y = mx - mp / 2
  let line_eq := λ x, m * x - m * p / 2
  -- Coordinates A and B, substitutions from solutions
  let x1 := 3 * p / 2
  let x2 := p / 6
  -- Points A and B
  let A := (x1, line_eq x1)
  let B := (x2, line_eq x2)
  -- Distance AF and BF
  let AF := Real.sqrt ((F.1 - A.1) ^ 2 + (F.2 - A.2) ^ 2)
  let BF := Real.sqrt ((F.1 - B.1) ^ 2 + (F.2 - B.2) ^ 2)
  in
  AF / BF = 3 :=
sorry

end parabola_ratio_l36_36993


namespace plates_probability_l36_36802

noncomputable def number_of_plates := 12
noncomputable def red_plates := 6
noncomputable def light_blue_plates := 3
noncomputable def dark_blue_plates := 3
noncomputable def total_pairs := number_of_plates * (number_of_plates - 1) / 2
noncomputable def red_pairs := red_plates * (red_plates - 1) / 2
noncomputable def light_blue_pairs := light_blue_plates * (light_blue_plates - 1) / 2
noncomputable def dark_blue_pairs := dark_blue_plates * (dark_blue_plates - 1) / 2
noncomputable def mixed_blue_pairs := light_blue_plates * dark_blue_plates
noncomputable def total_satisfying_pairs := red_pairs + light_blue_pairs + dark_blue_pairs + mixed_blue_pairs
noncomputable def desired_probability := (total_satisfying_pairs : ℚ) / total_pairs

theorem plates_probability :
  desired_probability = 5 / 11 :=
by
  -- Add the proof here
  sorry

end plates_probability_l36_36802


namespace probability_divisor_of_12_on_12sided_die_l36_36616

theorem probability_divisor_of_12_on_12sided_die (h : ∀ i, 1 ≤ i ∧ i ≤ 12 → true) : 
  (probability (fun (n : ℕ) => n ∣ 12 ∧ 1 ≤ n ∧ n ≤ 12)) = 1/2 := 
sorry

end probability_divisor_of_12_on_12sided_die_l36_36616


namespace wheel_center_distance_l36_36662

theorem wheel_center_distance :
  ∀ (r : ℝ) (n : ℕ), r = 2 ∧ n = 2 → 2 * n * Real.pi * r = 8 * Real.pi :=
by
  intros r n h
  cases h with hr hn
  rw [hr, hn]
  norm_num
  ring

end wheel_center_distance_l36_36662


namespace tan_identity_l36_36356

noncomputable def problem_statement (θ : ℝ) : Prop :=
  (π < θ ∧ θ < 3 * π / 2) ∧ (real.sin (θ - π / 4) = 3 / 5) →
  real.tan (θ + π / 4) = 4 / 3

theorem tan_identity (θ : ℝ) (h1 : π < θ ∧ θ < 3 * π / 2) (h2 : real.sin (θ - π / 4) = 3/5) :
  real.tan (θ + π / 4) = 4 / 3 :=
sorry

end tan_identity_l36_36356


namespace seq_is_arithmetic_l36_36132

theorem seq_is_arithmetic {a : ℕ → ℝ} 
  (h_nonzero : ∀ n, a n ≠ 0)
  (h_condition : ∀ n : ℕ, 
    ∑ i in finset.range n, (1 / (a (i+1) * a (i+2))) = n / (a 1 * a (n+1))) :
  ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a 1 + n * d :=
by
  sorry

end seq_is_arithmetic_l36_36132


namespace complex_inequality_l36_36399

theorem complex_inequality (a : ℝ) (z : ℂ) (h : z = complex.ofReal 1 + complex.I * a) (h_abs : abs z < 2) :
  -real.sqrt 3 < a ∧ a < real.sqrt 3 := 
sorry

end complex_inequality_l36_36399


namespace deployment_plans_l36_36906

/-- Given 6 volunteers and needing to select 4 to fill different positions of 
  translator, tour guide, shopping guide, and cleaner, and knowing that neither 
  supporters A nor B can work as the translator, the total number of deployment plans is 240. -/
theorem deployment_plans (volunteers : Fin 6) (A B : Fin 6) : 
  ∀ {translator tour_guide shopping_guide cleaner : Fin 6},
  A ≠ translator ∧ B ≠ translator → 
  ∃ plans : Finset (Fin 6 × Fin 6 × Fin 6 × Fin 6), plans.card = 240 :=
by 
sorry

end deployment_plans_l36_36906


namespace find_a5_l36_36537

noncomputable theory

def sequence_sum (n : ℕ) (a : ℕ → ℕ) : ℕ := 2 * a n - 1

axiom a1_value : ∀ (a : ℕ → ℕ), a 1 = 1

axiom geo_relation : ∀ (a : ℕ → ℕ), (∀ n, a (n + 1) = 2 * a n)

theorem find_a5 (a : ℕ → ℕ) : sequence_sum 5 a = 16 :=
by
  have h : ∀ (n : ℕ), a (n + 1) = 2 * a n := geo_relation a
  have a₁ : a 1 = 1 := a1_value a
  sorry

end find_a5_l36_36537


namespace discounts_correct_l36_36649

noncomputable def calc_discounts (C : ℝ) : ℝ × ℝ × ℝ :=
  let M := 1.75 * C
  let S1 := 1.48 * C
  let S2 := 1.525 * C
  let S3 := 1.575 * C
  let x := ((M - S1) / M) * 100
  let y := ((M - S2) / M) * 100
  let z := ((M - S3) / M) * 100
  (x.round, y.round, z.round)

theorem discounts_correct (C : ℝ) (hC_pos : C > 0) :
  calc_discounts C = (15, 13, 10) :=
by
  have h1 : calc_discounts C = ((27 / 1.75).round, (22.5 / 1.75).round, (17.5 / 1.75).round), by
  {
    unfold calc_discounts,
    have hM : 1.75 * C - 1.48 * C = 0.27 * C := by ring,
    have hM_2 : 1.75 * C - 1.525 * C = 0.225 * C := by ring,
    have hM_3 : 1.75 * C - 1.575 * C = 0.175 * C := by ring,
    simp [hM, hM_2, hM_3, real.div, mul_div_assoc],
  }
  have hx : (27 / 1.75).round = 15 := by norm_num,
  have hy : (22.5 / 1.75).round = 13 := by norm_num,
  have hz : (17.5 / 1.75).round = 10 := by norm_num,
  rw [hx, hy, hz] at h1,
  exact h1

end discounts_correct_l36_36649


namespace markup_percentage_l36_36220

variable (W R : ℝ)

-- Condition: When sold at a 40% discount, a sweater nets the merchant a 30% profit on the wholesale cost.
def discount_condition : Prop := 0.6 * R = 1.3 * W

-- Theorem: The percentage markup of the sweater from wholesale to normal retail price is 116.67%
theorem markup_percentage (h : discount_condition W R) : (R - W) / W * 100 = 116.67 :=
by sorry

end markup_percentage_l36_36220


namespace sum_of_digits_of_largest_n_l36_36120

-- Define the sets of primes
def primes_d := {5, 7}
def primes_e := {3, 7}

-- Define the function to compute potential third prime number
def third_prime (e d : ℕ) : ℕ := 10 * e + d

-- Define the product n given d, e, and the third prime
def product_n (d e : ℕ) : ℕ := d * e * third_prime e d

-- Define a function to check if a number is in a set
def in_set (n : ℕ) (s : set ℕ) : Prop := n ∈ s

-- Sum of the digits of a number
def sum_digits (n : ℕ) : ℕ := n.digits.sum

-- The largest integer n
def largest_n : ℕ :=
  max (max (product_n 5 3) (product_n 7 3)) (max (product_n 5 7) (product_n 7 7))

theorem sum_of_digits_of_largest_n :
  in_set 5 primes_d -> in_set 7 primes_d -> 
  in_set 3 primes_e -> in_set 7 primes_e -> 
  sum_digits largest_n = 21 :=
by
  sorry

end sum_of_digits_of_largest_n_l36_36120


namespace probability_of_rolling_divisor_of_12_l36_36599

def is_divisor (a b : ℕ) : Prop := b % a = 0

noncomputable def probability_divisor_12 : ℚ :=
  let divisors := { n | is_divisor n 12 ∧ n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} }
  let favorable_outcomes := (divisors : set ℕ).to_finset.card
  let total_outcomes := 12
  (favorable_outcomes : ℚ) / total_outcomes

theorem probability_of_rolling_divisor_of_12 : probability_divisor_12 = 1 / 2 := by
  sorry

end probability_of_rolling_divisor_of_12_l36_36599


namespace chandelier_illumination_prob_correct_l36_36591

theorem chandelier_illumination_prob_correct :
  ∀ (prob_bulb : ℝ), 
  (0 ≤ prob_bulb ∧ prob_bulb ≤ 1) ∧
  prob_bulb = 0.7 →
  let prob_chandelier := 1 - (1 - prob_bulb) * (1 - prob_bulb) * (1 - prob_bulb) in
  prob_chandelier = 0.973 := 
by 
  intros prob_bulb,
  intro h,
  let prob_chandelier := 1 - (1 - prob_bulb) * (1 - prob_bulb) * (1 - prob_bulb),
  sorry

end chandelier_illumination_prob_correct_l36_36591


namespace labourer_saved_amount_l36_36500

noncomputable def average_expenditure_6_months : ℕ := 85
noncomputable def expenditure_reduction_4_months : ℕ := 60
noncomputable def monthly_income : ℕ := 78

theorem labourer_saved_amount :
  let initial_debt := 6 * average_expenditure_6_months - 6 * monthly_income
      cleared_debt := 4 * monthly_income - 4 * expenditure_reduction_4_months
      savings := cleared_debt - initial_debt
  in savings = 30 :=
by
  have average_expenditure : ℕ := 6 * 85
  have average_income : ℕ := 6 * 78
  have initial_debt : ℕ := average_expenditure - average_income
  have new_expenditure : ℕ := 4 * 60
  have new_income : ℕ := 4 * 78
  have cleared_debt : ℕ := new_income - new_expenditure
  have savings : ℕ := cleared_debt - initial_debt
  have amount_saved : ℕ := 30
  show savings = amount_saved
  sorry

end labourer_saved_amount_l36_36500


namespace exists_integer_point_touching_x_axis_l36_36481

-- Define the context for the problem
variable {p q : ℤ}

-- Condition: The quadratic trinomial touches x-axis, i.e., discriminant is zero.
axiom discriminant_zero (p q : ℤ) : p^2 - 4 * q = 0

-- Theorem statement: Proving the existence of such an integer point.
theorem exists_integer_point_touching_x_axis :
  ∃ a b : ℤ, (a = -p ∧ b = q) ∧ (∀ (x : ℝ), x^2 + a * x + b = 0 → (a * a - 4 * b) = 0) :=
sorry

end exists_integer_point_touching_x_axis_l36_36481


namespace curve_is_hyperbola_l36_36462

-- Define the conditions
variables {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0)

-- The given line equation
def line_eq (x y : ℝ) : Prop := a * x - y + b = 0

-- The given curve equation
def curve_eq (x y : ℝ) : Prop := b * x^2 + a * y^2 = ab

-- State the theorem relating to the question
theorem curve_is_hyperbola (ha_opposite_signs : ∃ (a_pos : a > 0) (b_neg : b < 0), True ∨ ∃ (a_neg : a < 0) (b_pos : b > 0), True):
  ∃ (a_pos : a > 0) (b_neg : b < 0), curve_eq = by sorry := sorry

end curve_is_hyperbola_l36_36462


namespace lisa_cleaning_time_l36_36134

theorem lisa_cleaning_time (L : ℝ) (h1 : (1 / L) + (1 / 12) = 1 / 4.8) : L = 8 :=
sorry

end lisa_cleaning_time_l36_36134


namespace greatest_third_term_of_arithmetic_sequence_l36_36535

def is_arithmetic_sequence (a b c d : ℤ) : Prop := (b - a = c - b) ∧ (c - b = d - c)

theorem greatest_third_term_of_arithmetic_sequence :
  ∃ a b c d : ℤ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  is_arithmetic_sequence a b c d ∧
  (a + b + c + d = 52) ∧
  (c = 17) :=
sorry

end greatest_third_term_of_arithmetic_sequence_l36_36535


namespace multiple_of_every_positive_integer_is_zero_l36_36960

theorem multiple_of_every_positive_integer_is_zero :
  ∀ (n : ℤ), (∀ (m : ℕ), ∃ (k : ℤ), n = k * (m : ℤ)) → n = 0 := 
by
  sorry

end multiple_of_every_positive_integer_is_zero_l36_36960


namespace accurate_value_is_most_significant_l36_36819

-- Define the given constants
def K : ℝ := 3.78145
def error : ℝ := 0.00267

-- Noncomputable definition because we are dealing with reals and rounding
noncomputable def upper_bound : ℝ := K + error
noncomputable def lower_bound : ℝ := K - error

-- Prove that the most accurate reported value of K is 3.78
theorem accurate_value_is_most_significant : 
  ∀ (x y : ℝ), upper_bound.round(2) = 3.78 ∧ lower_bound.round(2) = 3.78 → x = 3.78 ∧ y = 3.78 := by
  sorry

end accurate_value_is_most_significant_l36_36819


namespace solution_set_f_l36_36774

noncomputable def f : ℝ → ℝ :=
sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom eq_f : ∀ x y : ℝ, x > 0 → y > 0 → f (x * y) = f x + f y - 1 / 2
axiom half_f : f (1/2) = 0
axiom pos_f : ∀ x : ℝ, x > 1/2 → f x > 0

theorem solution_set_f : {x : ℝ | e^(f x - 1) > 1} = (-1/8, 0) ∪ (2, ∞) :=
sorry

end solution_set_f_l36_36774


namespace count_correct_expressions_l36_36272

theorem count_correct_expressions :
  let e1 := ∅ = {0}
  let e2 := ∅ ⊊ {0}
  let e3 := ∅ ∈ {0}
  let e4 := 0 = {0}
  let e5 := {1} ∈ {1, 2, 3}
  let e6 := {1, 2} ⊊ {1, 2, 3}
  (if e1 then 1 else 0) +
  (if e2 then 1 else 0) +
  (if e3 then 1 else 0) +
  (if e4 then 1 else 0) +
  (if e5 then 1 else 0) +
  (if e6 then 1 else 0) = 2 :=
by
  sorry

end count_correct_expressions_l36_36272


namespace company_total_employees_l36_36945

def total_employees_after_hiring (T : ℕ) (before_hiring_female_percentage : ℚ) (additional_male_workers : ℕ) (after_hiring_female_percentage : ℚ) : ℕ :=
  T + additional_male_workers

theorem company_total_employees (T : ℕ)
  (before_hiring_female_percentage : ℚ)
  (additional_male_workers : ℕ)
  (after_hiring_female_percentage : ℚ)
  (h_before_percent : before_hiring_female_percentage = 0.60)
  (h_additional_male : additional_male_workers = 28)
  (h_after_percent : after_hiring_female_percentage = 0.55)
  (h_equation : (before_hiring_female_percentage * T)/(T + additional_male_workers) = after_hiring_female_percentage) :
  total_employees_after_hiring T before_hiring_female_percentage additional_male_workers after_hiring_female_percentage = 336 :=
by {
  -- This is where you add the proof steps.
  sorry
}

end company_total_employees_l36_36945


namespace probability_divisor_of_12_l36_36607

theorem probability_divisor_of_12 (fair_die : True) (sides : Nat) (h : sides = 12) : 
  ∃ p : ℚ, p = 1 / 2 ∧ ∀ roll : Fin sides, is_divisor roll.val 12 ↔ roll.val ∈ {1, 2, 3, 4, 6, 12} → p = 1 / 2 :=
by sorry

def is_divisor (a b : Nat) : Prop := b % a = 0

end probability_divisor_of_12_l36_36607


namespace part_1_part_2_part_3_l36_36781

-- Defining the even function and the conditions
def f (x : ℝ) (a : ℝ) : ℝ := 2^x / a + a / 2^x - 1

-- Conditions
axiom a_pos (a : ℝ) : a > 0

-- Propositions to prove
theorem part_1 (a : ℝ) (ha : a > 0) : f(-x,a) = f(x,a) → a = 1 := by
  sorry

theorem part_2 (a : ℝ) (ha : a = 1) (x : ℝ) : f(x,1) < 13 / 4 → -2 < x ∧ x < 2 := by
  sorry

theorem part_3 (m : ℝ) : (∀ x > 0, m * f(x,1) ≥ 2^(-x) - m) → m ≥ 1/2 := by
  sorry

end part_1_part_2_part_3_l36_36781


namespace valid_permutations_l36_36967

theorem valid_permutations (a : Fin 101 → ℕ) :
  (∀ k, a k ≥ 2 ∧ a k ≤ 102 ∧ (∃ j, a j = k + 2)) →
  (∀ k, a (k + 1) % (k + 1) = 0) →
  (∃ cycles : List (List ℕ), cycles = [[1, 102], [1, 2, 102], [1, 3, 102], [1, 6, 102], [1, 17, 102], [1, 34, 102], 
                                       [1, 51, 102], [1, 2, 6, 102], [1, 2, 34, 102], [1, 3, 6, 102], [1, 3, 51, 102], 
                                       [1, 17, 34, 102], [1, 17, 51, 102]]) :=
sorry

end valid_permutations_l36_36967


namespace sum_powers_mod_7_l36_36063

theorem sum_powers_mod_7 :
  (1^6 + 2^6 + 3^6 + 4^6 + 5^6 + 6^6) % 7 = 6 := by
  sorry

end sum_powers_mod_7_l36_36063


namespace finish_mowing_time_l36_36877

noncomputable def Mary's_rate : ℝ := 1/3
noncomputable def Tom's_rate : ℝ := 1/6
noncomputable def Tom_mows_alone_hours : ℝ := 2

theorem finish_mowing_time :
  let Tom_alone_mowed := Tom_mows_alone_hours * Tom's_rate
  let remaining_lawn := 1 - Tom_alone_mowed
  let combined_rate := Mary's_rate + Tom's_rate
  let time_to_finish := remaining_lawn / combined_rate
  time_to_finish = 4 / 3 :=
by {
  let Tom_alone_mowed := Tom_mows_alone_hours * Tom's_rate,
  let remaining_lawn := 1 - Tom_alone_mowed,
  let combined_rate := Mary's_rate + Tom's_rate,
  let time_to_finish := remaining_lawn / combined_rate,
  exact calc 
    time_to_finish = remaining_lawn / combined_rate : by rfl
                ... = (2/3) / (1/2) : by sorry -- Skipping detailed math steps
                ... = 4 / 3 : by sorry -- Skipping detailed math steps
}

end finish_mowing_time_l36_36877


namespace total_votes_cast_is_8200_l36_36587

variable (V : ℝ) (h1 : 0.35 * V < V) (h2 : 0.35 * V + 2460 = 0.65 * V)

theorem total_votes_cast_is_8200 (V : ℝ)
  (h1 : 0.35 * V < V)
  (h2 : 0.35 * V + 2460 = 0.65 * V) :
  V = 8200 := by
sorry

end total_votes_cast_is_8200_l36_36587


namespace find_k_l36_36521

open Real

noncomputable def curve (x : ℝ) (k : ℝ) : ℝ := 3 * ln x + x + k
noncomputable def tangent_line (x : ℝ) : ℝ := 4 * x - 1

theorem find_k (x₀ y₀ k : ℝ) (h_curve : y₀ = curve x₀ k) (h_tangent : ∀ x, 4 * x - (curve x₀ k) = 0) : k = 2 := by
  sorry

end find_k_l36_36521


namespace geometric_sequence_common_ratio_l36_36831

theorem geometric_sequence_common_ratio (a : ℕ → ℝ)
    (h1 : a 1 = -1)
    (h2 : a 2 + a 3 = -2) :
    ∃ q : ℝ, (a 2 = a 1 * q) ∧ (a 3 = a 1 * q^2) ∧ (q = -2 ∨ q = 1) :=
sorry

end geometric_sequence_common_ratio_l36_36831


namespace jesses_room_length_l36_36109

theorem jesses_room_length 
  (width : ℝ)
  (tile_area : ℝ)
  (num_tiles : ℕ)
  (total_area : ℝ := num_tiles * tile_area) 
  (room_length : ℝ := total_area / width)
  (hw : width = 12)
  (hta : tile_area = 4)
  (hnt : num_tiles = 6) :
  room_length = 2 :=
by
  -- proof omitted
  sorry

end jesses_room_length_l36_36109


namespace coefficient_of_x3_in_x_plus_one_pow_50_l36_36092

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := (n.choose k)

-- Define the binomial expansion using summation
def binomial_expansion (x : ℕ) (n : ℕ) : ℕ → ℕ :=
  λ k, binom n k * x^k 

-- Define the specific problem
def coeff_x3_in_expansion : ℕ :=
  binom 50 3

-- Theorem stating the desired result
theorem coefficient_of_x3_in_x_plus_one_pow_50 :
  coeff_x3_in_expansion = 19600 :=
by
  -- Skipping the proof part by using sorry
  sorry

end coefficient_of_x3_in_x_plus_one_pow_50_l36_36092


namespace problem_l36_36771

theorem problem (a : ℝ) (ha : a > -38) 
  (P Q : ℝ)
  (hP : P = sqrt (a + 40) - sqrt (a + 41)) 
  (hQ : Q = sqrt (a + 38) - sqrt (a + 39)) : 
  P > Q := 
by
  sorry

end problem_l36_36771


namespace min_intersection_size_l36_36747

theorem min_intersection_size (A B C : Set ℕ) (h1: n(A) + n(B) + n(C) = n(A ∪ B ∪ C)) (h2: |A| = 100) (h3: |B| = 100) : 
  |A ∩ B ∩ C| ≥ 97 :=
  sorry

# where n(S) = 2 ^ |S|
noncomputable def n (S : Set ℕ) : ℕ := 2 ^ S.to_finset.card

end min_intersection_size_l36_36747


namespace solve_series_l36_36712

noncomputable def infinite_series_sum (x : ℝ) : ℝ :=
  (λ n, (4 * n - 1) * x^n).sum

theorem solve_series (x : ℝ) (h1 : infinite_series_sum x = 50) (h2 : |x| < 1) :
  x = 0.021 :=
sorry

end solve_series_l36_36712


namespace focus_of_parabola_l36_36040

theorem focus_of_parabola (x y : ℝ) : x^2 = 4 * y → (0, 1) = (0, (4 / 4)) :=
by
  sorry

end focus_of_parabola_l36_36040


namespace fraction_addition_l36_36723

-- Definitions from conditions
def frac1 : ℚ := 18 / 42
def frac2 : ℚ := 2 / 9
def simplified_frac1 : ℚ := 3 / 7
def simplified_frac2 : ℚ := frac2
def common_denom_frac1 : ℚ := 27 / 63
def common_denom_frac2 : ℚ := 14 / 63

-- The problem statement to prove
theorem fraction_addition :
  frac1 + frac2 = 41 / 63 := by
  sorry

end fraction_addition_l36_36723


namespace range_of_a_l36_36032

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 + x^2 + a * x + 1

def is_monotonic_increasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-2) a, 0 ≤ (deriv f) x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) a, 0 ≤ (deriv (f a)) x) → 1 ≤ a := 
sorry

end range_of_a_l36_36032


namespace dogs_remaining_proof_l36_36539

-- Initialize variables and conditions
variables (D_0 D_a D_w D_m : ℕ) (D_f : ℕ)
noncomputable def initial_dogs := 200
noncomputable def additional_dogs := 100
noncomputable def dogs_adopted_week := 40
noncomputable def dogs_adopted_month := 60
noncomputable def final_dogs := initial_dogs + additional_dogs - dogs_adopted_week - dogs_adopted_month

-- Lean statement to prove the question == answer given the conditions
theorem dogs_remaining_proof 
    (initial_dogs_eq : D_0 = 200)
    (additional_dogs_eq : D_a = 100)
    (dogs_adopted_week_eq : D_w = 40)
    (dogs_adopted_month_eq : D_m = 60)
    (final_dogs_eq : D_f = 200) :
    initial_dogs + additional_dogs - dogs_adopted_week - dogs_adopted_month = final_dogs 
    :=
    by
    -- Introduce the conditions
    rw [initial_dogs_eq, additional_dogs_eq, dogs_adopted_week_eq, dogs_adopted_month_eq],
    -- Simplify the expression
    sorry

end dogs_remaining_proof_l36_36539


namespace f_at_2023_l36_36856

noncomputable def f (a x : ℝ) : ℝ := (a - x) / (a + 2 * x)

noncomputable def g (a x : ℝ) : ℝ := (f a (x - 2023)) + (1 / 2)

def is_odd (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = -g x

variable (a : ℝ)
variable (h_a : a ≠ 0)
variable (h_odd : is_odd (g a))

theorem f_at_2023 : f a 2023 = 1 / 4 :=
sorry

end f_at_2023_l36_36856


namespace cost_of_white_washing_l36_36972

def room_length : ℝ := 25
def room_width : ℝ := 15
def room_height : ℝ := 12
def cost_per_square_foot : ℝ := 7
def door_height : ℝ := 6
def door_width : ℝ := 3
def window_height : ℝ := 4
def window_width : ℝ := 3
def number_of_windows : ℕ := 3

theorem cost_of_white_washing :
  let perimeter := 2 * (room_length + room_width),
      total_wall_area := perimeter * room_height,
      door_area := door_height * door_width,
      window_area := window_height * window_width * (number_of_windows : ℝ),
      adjusted_wall_area := total_wall_area - door_area - window_area,
      total_cost := adjusted_wall_area * cost_per_square_foot
  in total_cost = 6342 := by
  sorry

end cost_of_white_washing_l36_36972


namespace budget_allocation_degrees_l36_36989

theorem budget_allocation_degrees :
  let microphotonics := 12.3
  let home_electronics := 17.8
  let food_additives := 9.4
  let gmo := 21.7
  let industrial_lubricants := 6.2
  let artificial_intelligence := 4.1
  let nanotechnology := 5.3
  let basic_astrophysics := 100 - (microphotonics + home_electronics + food_additives + gmo + industrial_lubricants + artificial_intelligence + nanotechnology)
  (basic_astrophysics * 3.6) + (artificial_intelligence * 3.6) + (nanotechnology * 3.6) = 117.36 :=
by
  sorry

end budget_allocation_degrees_l36_36989


namespace right_triangle_count_l36_36799

theorem right_triangle_count (a b : ℕ) (h1 : b < 100) (h2 : a^2 + b^2 = (b + 2)^2) : 
∃ n, n = 10 :=
by sorry

end right_triangle_count_l36_36799


namespace horizontal_asymptote_l36_36862

noncomputable def rational_function (x : ℝ) : ℝ :=
  (15 * x^4 + 7 * x^3 + 10 * x^2 + 6 * x + 4) / (4 * x^4 + 3 * x^3 + 9 * x^2 + 4 * x + 2)

theorem horizontal_asymptote :
  ∃ L : ℝ, (∀ ε > 0, ∃ M > 0, ∀ x > M, |rational_function x - L| < ε) → L = 15 / 4 :=
by
  sorry

end horizontal_asymptote_l36_36862


namespace grid_sum_bound_l36_36071

theorem grid_sum_bound (a : ℤ → ℤ → ℝ) (n : ℤ)
  (h_dim : n = 1987)
  (h_abs : ∀ i j, |a i j| ≤ 1)
  (h_sum : ∀ i j, a i j + a (i+1) j + a i (j+1) + a (i+1) (j+1) = 0)
  : ∑ i in finset.range 1987, ∑ j in finset.range 1987, a i j ≤ 1987 :=
by
  sorry

end grid_sum_bound_l36_36071


namespace lake_balaton_height_l36_36682

-- Define the length of Lake Balaton
def AC : ℝ := 76000 

-- Define the average radius of the Earth
def r : ℝ := 6370430 

-- Define the function to calculate height above water
noncomputable def heightAboveWater (AC r : ℝ) : ℝ :=
  let α := (360 * AC) / (2 * π * r) * (π / 180)
  let cosα := Real.cos α
  r * (1 / cosα - 1)

-- Final statement to prove
theorem lake_balaton_height : heightAboveWater AC r ≈ 430 := by
  sorry

end lake_balaton_height_l36_36682


namespace chocolate_ice_cream_ordered_l36_36904

theorem chocolate_ice_cream_ordered (V C : ℕ) (total_ice_cream : ℕ) (percentage_vanilla : ℚ) 
  (h_total : total_ice_cream = 220) 
  (h_percentage : percentage_vanilla = 0.20) 
  (h_vanilla_total : V = percentage_vanilla * total_ice_cream) 
  (h_vanilla_chocolate : V = 2 * C) 
  : C = 22 := 
by 
  sorry

end chocolate_ice_cream_ordered_l36_36904


namespace Q_over_P_l36_36510

theorem Q_over_P (P Q : ℚ)
  (h : ∀ (x : ℝ), x ≠ 0 ∧ x ≠ 3 ∧ x ≠ -3 → 
    (P / (x + 3) + Q / (x^2 - 3*x) = (x^2 - x + 8) / (x^3 + x^2 - 9*x))) :
  Q / P = 8 / 3 :=
by
  sorry

end Q_over_P_l36_36510


namespace shaded_area_of_circles_l36_36695

theorem shaded_area_of_circles (r1 r2 : ℝ) (h1 : r1 = 4) (h2 : r2 = 5) :
  let R := r1 + r2 in
  let area_large_circle := π * R^2 in
  let area_small_circle1 := π * r1^2 in
  let area_small_circle2 := π * r2^2 in
  area_large_circle - area_small_circle1 - area_small_circle2 = 40 * π :=
by
  sorry

end shaded_area_of_circles_l36_36695


namespace flight_duration_l36_36837

theorem flight_duration (h m : ℕ) (Hh : h = 2) (Hm : m = 32) : h + m = 34 := by
  sorry

end flight_duration_l36_36837


namespace nineteenth_permutation_is_8346_l36_36205

def digits := [3, 4, 6, 8]

def perm (n : ℕ) : ℕ := 
  let perms := List.permutations digits
  perms.nth (n - 1)

theorem nineteenth_permutation_is_8346 : perm 19 = 8346 := by
  sorry

end nineteenth_permutation_is_8346_l36_36205


namespace min_value_of_1_over_a_plus_2_over_b_l36_36037

theorem min_value_of_1_over_a_plus_2_over_b (a b : ℝ) (h1 : a + 2 * b = 1) (h2 : 0 < a) (h3 : 0 < b) : 
  (1 / a + 2 / b) ≥ 9 := 
sorry

end min_value_of_1_over_a_plus_2_over_b_l36_36037


namespace total_failing_grades_l36_36408

variable (a : ℕ → ℕ) (k : ℕ)

-- Given conditions:
axiom condition_1 : ∀ n : ℕ, n > k → a n = 0

-- Proving the total number of failing grades:
theorem total_failing_grades : (∑ i in Finset.range (k + 1), a i) = (∑ i in Finset.range (k + 1), i * (a i - a (i + 1))) :=
by
  sorry

end total_failing_grades_l36_36408


namespace prove_sum_l36_36003

noncomputable def arithmetic_seq (a d : ℕ → ℕ) := ∀ n : ℕ, a n = a 1 + (n - 1) * d

def a_n (n : ℕ) := 2 * n - 1

def S (n : ℕ) := n * (a_n 1 + a_n n) / 2

def b_n (n : ℕ) := 1 / (a_n n * a_n (n + 1))

def T (n : ℕ) := ∑ i in range n, b_n i

theorem prove_sum (n : ℕ) (h1 : a_n 2 + a_n 8 = 14)
  (h2 : S 5 = 25) : T n = n / (2 * n + 1) := sorry

end prove_sum_l36_36003


namespace coefficient_of_x3_in_x_plus_one_pow_50_l36_36090

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := (n.choose k)

-- Define the binomial expansion using summation
def binomial_expansion (x : ℕ) (n : ℕ) : ℕ → ℕ :=
  λ k, binom n k * x^k 

-- Define the specific problem
def coeff_x3_in_expansion : ℕ :=
  binom 50 3

-- Theorem stating the desired result
theorem coefficient_of_x3_in_x_plus_one_pow_50 :
  coeff_x3_in_expansion = 19600 :=
by
  -- Skipping the proof part by using sorry
  sorry

end coefficient_of_x3_in_x_plus_one_pow_50_l36_36090


namespace collectible_card_prices_l36_36248

theorem collectible_card_prices : 
  ∃ (n : ℕ), n = 90 ∧ 
    (∀ (prices : Finset (Fin 10)), 
       (prices.card = 6 ∧
        (∀ (d ∈ prices, d = 1 ∨ d = 4 ∨ d = 5 ∨ d = 6) ∧
        ∀ (d, d ∈ {4, 5, 6} → valid_starting_digit d prices))) →
      ∃ ! (cards_perm : List (Fin 10)), 
        (count_occurrences cards_perm [1, 1, 4, 5, 6, 6])) :=
sorry

def valid_starting_digit (d : ℕ) (prices : Finset (Fin 10)) : Prop :=
  ∀ (n : ℕ), 
    (prices.card = 5 ∧
     (∀ (d ∈ prices, d = 1 ∨ d = 4 ∨ d = 5 ∨ d = 6)) →

result = 5! / (2! * 2!) 

def count_occurrences (prices : List (Fin 10)) (digits : List ℕ) : ℕ :=
  prices.filter (λ x, x ∈ digits).length = 6

end collectible_card_prices_l36_36248


namespace polygon_area_l36_36766

/-- Given a polygon with vertices at the points (0, 0), (10, 0), (10, 10), (10, 20), (10, 30), 
(0, 30), (0, 20), (0, 10), prove that the area enclosed by the polygon is 300 square units. -/
theorem polygon_area : 
  let A := (0, 0)
  let B := (10, 0)
  let F := (10, 10)
  let J := (10, 20)
  let N := (10, 30)
  let M := (0, 30)
  let I := (0, 20)
  let E := (0, 10)
  in 
  (area_poly [A, B, F, J, N, M, I, E, A] = 300) := 
begin
  sorry -- The proof is omitted as per the instructions.
end

end polygon_area_l36_36766


namespace no_prime_solutions_for_x2_plus_y3_eq_z4_l36_36718

theorem no_prime_solutions_for_x2_plus_y3_eq_z4 :
  ¬ ∃ (x y z : ℕ), Prime x ∧ Prime y ∧ Prime z ∧ x^2 + y^3 = z^4 := sorry

end no_prime_solutions_for_x2_plus_y3_eq_z4_l36_36718


namespace PQ_eq_4_l36_36824

noncomputable def PQ_length (phi: ℝ) (rho1: ℝ) (theta1: ℝ) (rho2: ℝ) (theta2: ℝ): ℝ :=
  abs (rho1 - rho2)

theorem PQ_eq_4 : 
  (∀ phi : ℝ, 0 ≤ phi ∧ phi ≤ π → ∃ (x y : ℝ), x = 1 + cos phi ∧ y = sin phi) →
  (∀ (theta : ℝ), 0 ≤ theta ∧ theta < π / 2 → (2 * cos theta = ρ) → ρ > 0) →
  ρ_1 = 1 →
  θ_1 = π / 3 →
  θ_2 = π / 3 →
  ρ_2 * (sin θ_2 + √3 * cos θ_2) = 5√3 →
  ρ_2 = 5 →
  PQ_length 1 5 (π / 3) (π / 3) = 4 :=
by
  sorry

end PQ_eq_4_l36_36824


namespace number_of_pairs_l36_36388

theorem number_of_pairs (m n : ℕ) (sqrt2 : ℝ) (h_sqrt2: sqrt2 = Real.sqrt 2) 
  (h_m: m ≤ 1000) (h_n: n ≤ 1000) : 
  (m / (n + 1) < sqrt2) → (sqrt2 < (m + 1) / n) → 
  (∃ !p, p = 1706) :=
by
  sorry

end number_of_pairs_l36_36388


namespace problem_statement_l36_36393

variable {x : Real}
variable {m : Int}
variable {n : Int}

theorem problem_statement (h1 : x^m = 5) (h2 : x^n = 10) : x^(2 * m - n) = 5 / 2 :=
by
  sorry

end problem_statement_l36_36393


namespace locus_of_Q_is_as_described_l36_36410

variables {P Q A B C : Type} [EuclideanSpace P]

def is_general_position (A B C : P) : Prop :=
  ¬ (collinear A B C)

def passes_through (ω : Circle P) (A B : P) : Prop :=
  (A ∈ ω) ∧ (B ∈ ω)

def circumcircle (T : Triangle P) : Circle P := sorry

def intersection_point (c1 c2 : Circle P) (exclude : P) : P := sorry

noncomputable def locus_of_intersections (A B C : P) (ω : Circle P) (P_point : ω → P) : set P :=
  {Q | ∃ P ∈ ω, 
       let circPAB := circumcircle (Triangle.mk P A B),
           circPBC := circumcircle (Triangle.mk P B C) in
       Q = intersection_point circPAB circPBC P }

theorem locus_of_Q_is_as_described :
  ∀ (A B C : P) (ω : Circle P),
    is_general_position A B C →
    passes_through ω A B →
    let P_points := λ ω, sorry in
    ∀ Q ∈ locus_of_intersections A B C ω P_points,
      true := -- Locus description
sorry

end locus_of_Q_is_as_described_l36_36410


namespace sum_of_exterior_angles_of_regular_hexagon_eq_360_l36_36196

theorem sum_of_exterior_angles_of_regular_hexagon_eq_360:
  ∀ (hexagon : Type), (regular_polygon hexagon ∧ sides hexagon = 6) → sum_exterior_angles hexagon = 360 :=
by
  sorry

end sum_of_exterior_angles_of_regular_hexagon_eq_360_l36_36196


namespace tangent_line_at_1_l36_36031

noncomputable def f : ℝ → ℝ := sorry

theorem tangent_line_at_1 (f : ℝ → ℝ) (h_tangent : ∀ x, f 1 + (f' 1) * (x - 1) = 1/2 * x + 3) :
  f 1 + f' 1 = 4 := 
by
  have h' : f' 1 = 1 / 2 := sorry
  have h₁ : f 1 = 7 / 2 := sorry
  calc
    f 1 + f' 1 = (7 / 2) + (1 / 2) := by rw [h₁, h']
         ... = 4 := by norm_num

end tangent_line_at_1_l36_36031


namespace units_digit_9_19_1989_minus_9_pow_4_l36_36321

theorem units_digit_9_19_1989_minus_9_pow_4 : (9 * 19 * 1989 - 9^4) % 10 = 8 := by
    -- reducing to units digit (last digit)
    let a := 9 * 19 * 1989 % 10
    have h1 : a = 9 % 10, by sorry -- Units digit of the product 9 * 9 * 9 = 9
    let b := 9^4 % 10
    have h2 : b = 1, by sorry -- Units digit of 9^4=6561
    -- computing the final units digit
    show (a - b) % 10 = 8, by sorry -- 9 - 1 = 8

end units_digit_9_19_1989_minus_9_pow_4_l36_36321


namespace john_squat_increase_l36_36440

theorem john_squat_increase :
  ∃ x : ℝ, (135 + x) * 7 = 2800 ∧ x = 265 :=
by {
  let x := 265,
  use x,
  split,
  -- Show that the equation is satisfied
  calc (135 + x) * 7 = (135 + 265) * 7 : by rfl
                   ... = 400 * 7 : by norm_num
                   ... = 2800 : by norm_num,
  -- Show that x indeed equals 265
  exact rfl
}

end john_squat_increase_l36_36440


namespace measure_angle_BHC_correct_l36_36429

-- Assuming angles are measured in degrees and the conditions given
noncomputable def measure_angle_BHC (ABC : Triangle) (H : Point) (A B C : Point) : angle := 
  if (angle_measure ABC B C = 53) ∧ (angle_measure ABC C B = 17) ∧ (is_orthocenter H ABC) 
  then 70 
  else 0

theorem measure_angle_BHC_correct (ABC : Triangle) (H : Point) (A B C : Point)
  (h1 : angle_measure ABC B C = 53) (h2 : angle_measure ABC C B = 17) 
  (h3 : is_orthocenter H ABC) :
  measure_angle_BHC ABC H A B C = 70 :=
by {
  -- The proof would go here
  sorry
}

end measure_angle_BHC_correct_l36_36429


namespace brenda_age_l36_36271

variable (A B J : ℕ)

theorem brenda_age :
  (A = 3 * B) →
  (J = B + 6) →
  (A = J) →
  (B = 3) :=
by
  intros h1 h2 h3
  -- condition: A = 3 * B
  -- condition: J = B + 6
  -- condition: A = J
  -- prove B = 3
  sorry

end brenda_age_l36_36271


namespace gugu_possible_q_values_l36_36464

theorem gugu_possible_q_values (q : ℝ) (napkin : Fin 10 → ℝ) :
  (∀ (a b : ℝ), ∃ (c d e f : ℝ), qa b = a^2 + b^2 - c^2 - d^2) ↔ (q = 0 ∨ q = 2 ∨ q = -2) :=
by
  sorry

end gugu_possible_q_values_l36_36464


namespace domain_of_inverse_l36_36927

noncomputable def f (x : ℝ) : ℝ := (1/2)^(x - 1) + 1

theorem domain_of_inverse :
  ∀ y : ℝ, (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ y = f x) → (y ∈ Set.Icc (3/2) 3) :=
by
  sorry

end domain_of_inverse_l36_36927


namespace largest_non_zero_ending_factor_decreasing_number_l36_36315

theorem largest_non_zero_ending_factor_decreasing_number :
  ∃ n: ℕ, n = 180625 ∧ (n % 10 ≠ 0) ∧ (∃ m: ℕ, m < n ∧ (n % m = 0) ∧ (n / 10 ≤ m ∧ m * 10 > 0)) :=
by {
  sorry
}

end largest_non_zero_ending_factor_decreasing_number_l36_36315


namespace shaded_area_correct_l36_36095

open Real

-- Define the conditions
def radius_small : ℝ := 3
def radius_large : ℝ := 6

-- Define the corresponding diameters and areas
def diameter_small := 2 * radius_small
def diameter_large := 2 * radius_large

def area_rectangle_small := diameter_small * radius_small
def area_rectangle_large := diameter_large * radius_large

def area_semicircle_small := (1 / 2) * pi * (radius_small ^ 2)
def area_semicircle_large := (1 / 2) * pi * (radius_large ^ 2)

-- Calculate the total shaded area
def total_shaded_area := (area_rectangle_small - area_semicircle_small) + (area_rectangle_large - area_semicircle_large)

-- Prove the mathematically equivalent problem
theorem shaded_area_correct :
  total_shaded_area = 90 - (45 / 2) * pi := 
  sorry

end shaded_area_correct_l36_36095


namespace find_r_l36_36466

noncomputable def g (x : ℝ) (p q r : ℝ) := x^3 + p * x^2 + q * x + r

theorem find_r 
  (p q r : ℝ) 
  (h1 : ∀ x : ℝ, g x p q r = (x + 100) * (x + 0) * (x + 0))
  (h2 : p + q + r = 100) : 
  r = 0 := 
by
  sorry

end find_r_l36_36466


namespace length_of_XY_in_triangle_XYZ_l36_36102

theorem length_of_XY_in_triangle_XYZ :
  ∀ (XYZ : Type) (X Y Z : XYZ) (angle : XYZ → XYZ → XYZ → ℝ) (length : XYZ → XYZ → ℝ),
  angle X Z Y = 30 ∧ angle Y X Z = 90 ∧ length X Z = 8 → length X Y = 16 :=
by sorry

end length_of_XY_in_triangle_XYZ_l36_36102


namespace simplify_and_evaluate_expression_l36_36909

variable a : ℤ

theorem simplify_and_evaluate_expression (h : a = 3) : 
  (a - 3) ^ 2 - (a - 1) * (a + 1) + 2 * (a + 3) = 4 :=
by
  sorry

end simplify_and_evaluate_expression_l36_36909


namespace combined_spots_l36_36383

-- Definitions of the conditions
def Rover_spots : ℕ := 46
def Cisco_spots : ℕ := Rover_spots / 2 - 5
def Granger_spots : ℕ := 5 * Cisco_spots

-- The proof statement
theorem combined_spots :
  Granger_spots + Cisco_spots = 108 := by
  sorry

end combined_spots_l36_36383


namespace probability_divisor_of_12_l36_36643

theorem probability_divisor_of_12 : 
  (nat.factorization 12).keys = {1, 2, 3, 4, 6, 12} → 
  (∃ die_side, die_side ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) → 
  (∃ favorable_outcomes, favorable_outcomes ∈ {1, 2, 3, 4, 6, 12}) → 
  ∃ probability, probability = (favorable_outcomes.card : ℚ) / (die_side.card : ℚ) ∧ probability = 1/2 :=
by 
  sorry

end probability_divisor_of_12_l36_36643


namespace problem_statement_l36_36446

noncomputable def proof_problem : Prop :=
  ∃ (C1 C2 C3 : Circle) (A B C A' B' C' : Point) (L1 L2 L3 L4 L5 L6 : Line),
    -- Conditions
    circles_non_intersect : ¬(intersect(C1, C2)) ∧ ¬(intersect(C2, C3)) ∧ ¬(intersect(C3, C1)) ∧
    circles_non_inside : ¬(inside(C1, C2)) ∧ ¬(inside(C1, C3)) ∧ ¬(inside(C2, C3)) ∧
    is_internal_tangent (L1, L2) (C1, C2) ∧ is_internal_tangent (L3, L4) (C1, C3) ∧ is_internal_tangent (L5, L6) (C2, C3) ∧
    lines_define_polygon ([L1, L2, L3, L4, L5, L6]) ([AC', BA', CB']) ->
    -- Conclusion
    concurrent ([AA', BB', CC'])

-- Theorem statement
theorem problem_statement : proof_problem :=
sorry

end problem_statement_l36_36446


namespace range_a_l36_36360

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = - (f x)

def monotonic_increasing_on_nonnegative (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem range_a (f : ℝ → ℝ) (h_odd : odd_function f) 
    (h_monotonic : monotonic_increasing_on_nonnegative f) :
    {a : ℝ | f (Real.log 2 a) + f (Real.log 2 (1 / a)) ≤ 2 * (f 1)} = {a : ℝ | 0 < a ∧ a ≤ 2} :=
sorry

end range_a_l36_36360


namespace crayons_left_l36_36542

-- Conditions
def initial_crayons : ℕ := 120
def kiley_fraction : ℝ := 3 / 8
def joe_fraction : ℝ := 5 / 9

-- Proof statement
theorem crayons_left (initial_crayons : ℕ)
  (kiley_fraction : ℝ) (joe_fraction : ℝ)
  (h1 : initial_crayons = 120)
  (h2 : kiley_fraction = 3 / 8)
  (h3 : joe_fraction = 5 / 9) :
  let kiley_takes = kiley_fraction * initial_crayons
  let remaining_after_kiley = initial_crayons - kiley_takes
  let joe_takes = joe_fraction * remaining_after_kiley
  let final_remaining = remaining_after_kiley - joe_takes
  final_remaining = 33 :=
by
  sorry

end crayons_left_l36_36542


namespace chocolate_ice_cream_ordered_l36_36905

theorem chocolate_ice_cream_ordered (V C : ℕ) (total_ice_cream : ℕ) (percentage_vanilla : ℚ) 
  (h_total : total_ice_cream = 220) 
  (h_percentage : percentage_vanilla = 0.20) 
  (h_vanilla_total : V = percentage_vanilla * total_ice_cream) 
  (h_vanilla_chocolate : V = 2 * C) 
  : C = 22 := 
by 
  sorry

end chocolate_ice_cream_ordered_l36_36905


namespace high_tone_function_l36_36804

-- Define \varphi_k function
def varphi_k (k : ℕ) (x : ℕ) : ℕ := x * k

-- Define the condition that involves the functional inequality
def functional_inequality (f : ℕ → ℕ) (λ : ℕ) (x : ℕ) : Prop := 
  f(x + λ) ≥ varphi_k 1 (f x)  -- Note: varphi_k 1 represents \varphi here as \varphi_k(f(x)) = f(x) * 1

-- Assert the solution function must be a high tone function based on the given conditions
theorem high_tone_function (f : ℕ → ℕ) (λ : ℕ) (h : ∀ x, functional_inequality f λ x) : 
  ∀ x, f (x + 1) > f x := sorry

end high_tone_function_l36_36804


namespace find_m_l36_36184

noncomputable theory
open Real

-- Define line AB and the intersection points A and B
def line_AB := λ x : ℝ, - (sqrt 3 / 3) * x + 1
def A : ℝ × ℝ := (sqrt 3, 0)
def B : ℝ × ℝ := (0, 1)

-- Define equilateral triangle ΔABC in the first quadrant
def equilateral_triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

-- Define point P in the first quadrant
def P (m : ℝ) : ℝ × ℝ := (m, 1 / 2)

-- Area of to be equal
def area_equality (A B P : ℝ × ℝ) (C : ℝ × ℝ) : Prop :=
  abs ((A.1 * (B.2 - P.2) + B.1 * (P.2 - A.2) + P.1 * (A.2 - B.2)) / 2) =
  abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)

-- The main theorem
theorem find_m (h : equilateral_triangle_ABC A B C) (hABP : area_equality A B (P m) C) : 
  m = (5 * sqrt 3) / 2 :=
by
  sorry

end find_m_l36_36184


namespace find_y_given_x_div_y_l36_36573

theorem find_y_given_x_div_y (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x % y = 11.52) (h4 : (x : ℝ) / (y : ℝ) = 96.12) : y = 96 :=
sorry

end find_y_given_x_div_y_l36_36573


namespace basketball_team_lineup_l36_36984

-- Define the problem conditions
def total_players : ℕ := 12
def twins : ℕ := 2
def lineup_size : ℕ := 5
def remaining_players : ℕ := total_players - twins
def positions_to_fill : ℕ := lineup_size - twins

-- Define the combination function as provided in the standard libraries
def combination (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem translating to the proof problem
theorem basketball_team_lineup : combination remaining_players positions_to_fill = 120 := 
sorry

end basketball_team_lineup_l36_36984


namespace sample_variance_l36_36318
-- We are using a broad import to ensure all necessary libraries are available.

noncomputable def sampleValues : List ℝ := [0.01, 0.04, 0.08]

noncomputable def sampleFrequencies : List ℕ := [5, 3, 2]

noncomputable def totalSampleSize : ℕ := 10

theorem sample_variance :
  let x_i := sampleValues
  let n_i := sampleFrequencies
  let n := totalSampleSize
  let u_i := x_i.map ((· * 100) : ℝ → ℝ)
  let sum_n_i_u_i_squared := (List.zipWith (λ n x => n * x^2) n_i u_i).sum
  let sum_n_i_u_i := (List.zipWith (λ n x => n * x) n_i u_i).sum
  let variance_u := (sum_n_i_u_i_squared / n.toReal) - ((sum_n_i_u_i / n.toReal)^2)
  let variance_x := variance_u / (100^2 : ℝ)
  variance_x = 0.000721 := by
  sorry

end sample_variance_l36_36318


namespace length_of_goods_train_l36_36252

theorem length_of_goods_train
    (train_speed_kmph : ℕ)
    (platform_length_m : ℕ)
    (crossing_time_s : ℕ)
    (train_speed_mps : train_speed_kmph * 1000 / 3600 = 20)
    (total_distance_m : train_speed_mps * crossing_time_s = 520)
    : (total_distance_m - platform_length_m = 220) :=
sorry

end length_of_goods_train_l36_36252


namespace altitude_eq_median_eq_l36_36943

-- Define the points A, B, C
def A := (4, 0) : ℤ × ℤ
def B := (6, 7) : ℤ × ℤ
def C := (0, 3) : ℤ × ℤ

-- The equation of the altitude from A to BC
theorem altitude_eq : 2 * (A.1 - 0) + 7 * (A.2 - 3) = 21 := 
sorry

-- The equation of the median from B to C
theorem median_eq : 5 * (A.1 - 4) + (A.2 - 20) = 0 :=
sorry

end altitude_eq_median_eq_l36_36943


namespace no_such_primes_exist_l36_36717

open Nat

theorem no_such_primes_exist :
  ¬ ∃ (p : Fin 2007 → ℕ), (∀ i, p i).Prime ∧
    (∀ i, (p i)^2 - 1 ∣ p ((i + 1) % 2007)) :=
sorry

end no_such_primes_exist_l36_36717


namespace correct_projection_conclusions_l36_36354

theorem correct_projection_conclusions (a b : ℝ^3 → ℝ^3) (α : set (ℝ^3)) : 
  (skew_lines a b) → (non_perpendicular a b) → (is_plane α) →
  correct_projection_options α [1, 2, 4] :=
by sorry

end correct_projection_conclusions_l36_36354


namespace simplify_expression_l36_36722

theorem simplify_expression :
  (2 + 1 / 2) / (1 - 3 / 4) = 10 :=
by
  sorry

end simplify_expression_l36_36722


namespace no_A_in_any_courses_l36_36724

namespace ProofProblem

variables (students : ℕ) (A_in_history : ℕ) (A_in_math : ℕ) (A_in_both : ℕ)

def total_students := 30
def received_A_in_history := 7
def received_A_in_math := 13
def received_A_in_both := 4

theorem no_A_in_any_courses :
  students - (A_in_history + A_in_math - A_in_both) = 14 :=
by
  have h : (A_in_history + A_in_math - A_in_both) = 16 := by sorry
  have result : students - 16 = 14 := by sorry
  exact result

end ProofProblem

end no_A_in_any_courses_l36_36724


namespace all_buses_have_same_stoppage_time_l36_36726

-- Define the constants for speeds without and with stoppages
def speed_without_stoppage_bus1 := 50
def speed_without_stoppage_bus2 := 60
def speed_without_stoppage_bus3 := 70

def speed_with_stoppage_bus1 := 40
def speed_with_stoppage_bus2 := 48
def speed_with_stoppage_bus3 := 56

-- Stating the stoppage time per hour for each bus
def stoppage_time_per_hour (speed_without : ℕ) (speed_with : ℕ) : ℚ :=
  1 - (speed_with : ℚ) / (speed_without : ℚ)

-- Theorem to prove the stoppage time correctness
theorem all_buses_have_same_stoppage_time :
  stoppage_time_per_hour speed_without_stoppage_bus1 speed_with_stoppage_bus1 = 0.2 ∧
  stoppage_time_per_hour speed_without_stoppage_bus2 speed_with_stoppage_bus2 = 0.2 ∧
  stoppage_time_per_hour speed_without_stoppage_bus3 speed_with_stoppage_bus3 = 0.2 :=
by
  sorry  -- Proof to be completed

end all_buses_have_same_stoppage_time_l36_36726


namespace shifted_function_properties_l36_36765

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_monotonically_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem shifted_function_properties (f : ℝ → ℝ)
  (h1 : is_even_function f)
  (h2 : is_monotonically_increasing_on f 3 5) :
  (is_monotonically_increasing_on (λ x, f (x-1)) 4 6) ∧
  (∀ x, f (x - 1) = f (-(x-1) + 2)) :=
begin
  sorry
end

end shifted_function_properties_l36_36765


namespace find_m_l36_36740

theorem find_m (m : ℤ) (h1 : -180 < m ∧ m < 180) : 
  ((m = 45) ∨ (m = -135)) ↔ (Real.tan (m * Real.pi / 180) = Real.tan (225 * Real.pi / 180)) := 
by 
  sorry

end find_m_l36_36740


namespace minimum_concerts_to_meet_conditions_l36_36490

-- Definition of musicians as type alias for simplicity
def Musician := Fin 6

-- Defining a concert as a set of playing musicians
def Concert := Finset Musician

-- Given conditions of the problem
variable (concerts : Finset (Finset Concert))

-- Predicate to check if every pair (a, b) ∈ cons x 
-- will have a concert where one is playing and the other is listening.
def all_pairs_combinations_are_met : Prop :=
  ∀ (m₁ m₂ : Musician), m₁ ≠ m₂ → 
    ∃ c ∈ concerts, (m₁ ∈ c ∧ m₂ ∉ c) ∨ (m₂ ∈ c ∧ m₁ ∉ c)

-- The theorem we need to prove
theorem minimum_concerts_to_meet_conditions (concerts : Finset (Finset Concert)) :
  all_pairs_combinations_are_met concerts → ∃ c, c.card = 4 := sorry

end minimum_concerts_to_meet_conditions_l36_36490


namespace solve_diamond_l36_36390

theorem solve_diamond {d : ℕ} (h: d < 10) (h_eq: d * 9 + 5 = d * 11 + 3) : d = 1 :=
by {
  have h_simplified: (11 * d - 9 * d = 5 - 3),
  { linarith, }, -- Using linear arithmetic to verify the intermediate step
  linarith, -- Final linear arithmetic step to solve for d
}

end solve_diamond_l36_36390


namespace sum_of_exponents_of_sqrt_of_largest_perfect_square_divides_15_factorial_l36_36959

/-!
# Problem
What is the sum of the exponents of the prime factors of the square root of the largest perfect square that divides 15!?
-/

theorem sum_of_exponents_of_sqrt_of_largest_perfect_square_divides_15_factorial : 
  let factorial (n : ℕ) : ℕ := (list.range (n+1)).foldr (· * ·) 1
  let prime_exponents (n p : ℕ) : ℕ := (list.range (nat.log p n + 1)).sum (λ k, n / (p ^ k))
  let largest_perfect_square_factor (n : ℕ) : ℕ :=
    ∏ p in finset.primes n, p^(2 * (prime_exponents n p / 2))
  let sqrt_of_largest_perfect_square_factor (n : ℕ) : ℕ :=
    ∏ p in finset.primes n, p^(prime_exponents n p / 2)
  in 
  (list.range (nat.log 2 15 + 1)).sum (λ k, prime_exponents 15 2 / 2) +
  (list.range (nat.log 3 15 + 1)).sum (λ k, prime_exponents 15 3 / 2) +
  (list.range (nat.log 5 15 + 1)).sum (λ k, prime_exponents 15 5 / 2) = 9 :=
by 
  -- proof will be filled in here
  sorry


end sum_of_exponents_of_sqrt_of_largest_perfect_square_divides_15_factorial_l36_36959


namespace segments_can_be_painted_l36_36847

open Function

variables (X Y : Type) (M : set (X × Y)) (k : ℕ)

def is_valid_coloring (coloring : X × Y → ℕ) :=
  ∀ x : X, ∀ α β : ℕ, α ≠ β →
  abs ((coloring '' (M ∩ (set.prod {x} set.univ))).count α -
       (coloring '' (M ∩ (set.prod {x} set.univ))).count β) ≤ 1 ∧
  ∀ y : Y, ∀ α β : ℕ, α ≠ β →
  abs ((coloring '' (M ∩ (set.prod set.univ {y}))).count α -
       (coloring '' (M ∩ (set.prod set.univ {y}))).count β) ≤ 1

theorem segments_can_be_painted (hM : ∀ (x : X) (y : Y), (x, y) ∈ M) :
  ∃ (coloring : X × Y → ℕ), is_valid_coloring X Y M coloring :=
sorry

end segments_can_be_painted_l36_36847


namespace complex_problem_l36_36400

noncomputable def z : ℂ := 1 + complex.i

theorem complex_problem : ((1 : ℂ) + z) * z = 1 + 3 * complex.i :=
by
  unfold z
  sorry

end complex_problem_l36_36400


namespace proof_equivalent_l36_36339

variable {k : ℝ} (hk : k > 0)
def f (x : ℝ) : ℝ := Real.exp x
def g (x : ℝ) : ℝ := Real.exp (-x)
def n : ℝ := f k
def m : ℝ := g k
variable (h1 : n = Real.exp k)
variable (h2 : m = Real.exp (-k))
variable (h3 : n < 2 * m)

theorem proof_equivalent : 
  (n + m < 3 * Real.sqrt 2 / 2) ∧ 
  (n - m < Real.sqrt 2 / 2) ∧ 
  (n^(m+1) < (m+1)^n) := by 
  sorry

end proof_equivalent_l36_36339


namespace part1_part2_part3_l36_36028

def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + Real.cos (2 * x) - 1

theorem part1 : f (Real.pi / 4) = 1 := sorry

theorem part2 : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := sorry

theorem part3 : ∀ k : ℤ, ∃ a b, a = k * Real.pi - (3 * Real.pi) / 8 ∧
    b = k * Real.pi + Real.pi / 8 ∧ ∀ x, a ≤ x ∧ x ≤ b → f(x) is_monotonic := sorry

end part1_part2_part3_l36_36028


namespace unique_prime_with_conditions_l36_36734

theorem unique_prime_with_conditions (p : ℕ) (hp : Nat.Prime p) (hp2 : Nat.Prime (p + 2)) (hp4 : Nat.Prime (p + 4)) : p = 3 :=
by
  sorry

end unique_prime_with_conditions_l36_36734


namespace fraction_of_class_with_brown_eyes_l36_36589

variable (students : ℕ) (brown_eyes : ℕ) (brown_eyes_black_hair : ℕ) (total_students : ℕ)
variable (fraction_brown_eyes : ℚ)

-- Define the given conditions
def condition_1 : Prop := 2 * brown_eyes_black_hair = brown_eyes
def condition_2 : Prop := brown_eyes_black_hair = 6
def condition_3 : Prop := total_students = 18

-- Define the question
def question : Prop := fraction_brown_eyes = brown_eyes / total_students

-- Define the correct answer
def answer : Prop := fraction_brown_eyes = 2 / 3

-- Prove that given the conditions, the fraction of the class with brown eyes equals 2/3
theorem fraction_of_class_with_brown_eyes :
  condition_1 ∧ condition_2 ∧ condition_3 → (question ↔ answer) := 
by
  sorry

end fraction_of_class_with_brown_eyes_l36_36589


namespace third_side_length_l36_36018

theorem third_side_length (a b x : ℝ) (h₁ : a = 3) (h₂ : b = 8) (h₃ : 5 < x) (h₄ : x < 11) : x = 6 :=
sorry

end third_side_length_l36_36018


namespace largest_integer_T_l36_36000

theorem largest_integer_T (p : Nat → Nat) (h_prime : ∀ i, Nat.Prime (p i)) 
  (h_len : ∀ i : Nat, i < 25 → p i ≤ 2004) (h_distinct : ∀ i j : Nat, i ≠ j → p i ≠ p j)
  (h_sorted : ∀ i j : Nat, i < j → p i < p j)
  : ∃ T : Nat, 
      (∀ n : Nat, n ≤ T → ∃ (q : Multiset Nat) (hq : ∀ x ∈ q, x ∣ (Multiset.prod (Multiset.map (λ i, (p i) ^ 2004) (Multiset.range 25)))),
        Multiset.rel (·≠·) q q ∧ n = q.sum) ∧ 
      (if p 0 = 2 then T = (2 ^ 2005 - 1) * ∏ i in Finset.range 24 | i | 1 < Finset.card (Finset.range 24) ∧ (∀ i ∈ Finset.range 25, p (i + 1) ≤ 2004 ∧ Nat.Prime (p (i + 1))){
        ∏ i in Finset.range 24, ((p (i + 1)) ^ 2005 - 1) / (p (i + 1) - 1)
      } else T = 1) :=
sorry

end largest_integer_T_l36_36000


namespace ellipse_problem_l36_36009

-- Given conditions
def ellipse_eq (a b x y : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (a > b) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def eccentricity (a : ℝ) : ℝ :=
  (sqrt 2) / 2 / a

def right_focus (F : ℝ × ℝ) : Prop :=
  F = (1, 0)

def intersects (A B : ℝ × ℝ) : Prop :=
  ∃ t k : ℝ, (A = (t, k * (t - 1)) ∧ B = (t, -k * (t - 1)))

def fixed_point (M : ℝ × ℝ) : Prop :=
  M = (5/4, 0)

-- The statement to be proven
theorem ellipse_problem 
  (a b : ℝ) (x y : ℝ) (F A B M : ℝ × ℝ) :
  ellipse_eq a b x y →
  eccentricity a ∧ 
  right_focus F →
  intersects A B →
  fixed_point M →
  ((x^2 / 2 + y^2 = 1) ∧ 
  (∃ λ : ℝ, 
    ∀ k : ℝ, (∃ t : ℝ, 
      ((2 * t^2 - 4 * t + 1 - 2 * λ) * k^2 + (t^2 - 2 - λ) = 0) ∧ 
      ((M.1, 0) = (5/4, 0)) ∧ 
      (λ = -(7 / 16)))))
:=
  sorry

end ellipse_problem_l36_36009


namespace parallel_lines_implies_m_opposite_sides_implies_m_range_l36_36038

-- Definitions of the given lines and points
def l1 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def A (m : ℝ) : ℝ × ℝ := (-2, m)
def B (m : ℝ) : ℝ × ℝ := (m, 4)

-- Problem Part (I)
theorem parallel_lines_implies_m (m : ℝ) : 
  (∀ (x y : ℝ), l1 x y → false) ∧ (∀ (x2 y2 : ℝ), (x2, y2) = A m ∨ (x2, y2) = B m → false) →
  (∃ m, 2 * m + 3 = 0 ∧ m + 5 = 0) :=
sorry

-- Problem Part (II)
theorem opposite_sides_implies_m_range (m : ℝ) :
  ((2 * (-2) + m - 1) * (2 * m + 4 - 1) < 0) →
  m ∈ Set.Ioo (-3/2 : ℝ) (5 : ℝ) :=
sorry

end parallel_lines_implies_m_opposite_sides_implies_m_range_l36_36038


namespace solve_equation_1_solve_equation_2_l36_36288

-- Problem statement for equation 1
theorem solve_equation_1 (x : ℝ) : x^4 - x^2 - 6 = 0 ↔ x = sqrt(3) ∨ x = -sqrt(3) :=
by
  sorry

-- Problem statement for equation 2
theorem solve_equation_2 (x : ℝ) : (x^2 + 2x)^2 - 2(x^2 + 2x) - 3 = 0 ↔ x = 1 ∨ x = -3 ∨ x = -1 :=
by
  sorry

end solve_equation_1_solve_equation_2_l36_36288


namespace square_properties_l36_36936

theorem square_properties 
  (P1 P2 : ℕ) (P1_eq : P1 = 40) (P2_eq : P2 = 32)
  (SQRT : ∀ x : ℕ, ∃ y : ℕ, y * y = x) :
  let s1 := P1 / 4 in
  let s2 := P2 / 4 in
  let A1 := s1 * s1 in
  let A2 := s2 * s2 in
  let A_diff := A1 - A2 in
  let s3 := Nat.sqrt A_diff in
  let P3 := 4 * s3 in
  s1 = 10 ∧ s2 = 8 ∧ A_diff = 36 ∧ s3 = 6 ∧ P3 = 24 ∧ (∃ y : ℕ, s3 = y) :=
by
  sorry

end square_properties_l36_36936


namespace part1_part2_l36_36366

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * Real.sin x + Real.cos x

theorem part1 : f (5 * Real.pi / 6) = 0 := sorry

theorem part2 : 
  (∀ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2), f x ≤ 2 ∧ f x ≥ -sqrt 3) ∧ 
  (∃ x' ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2), f x' = 2 ∧ x' = Real.pi / 3) ∧
  (∃ x'' ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2), f x'' = -sqrt 3 ∧ x'' = -Real.pi / 2) := sorry

end part1_part2_l36_36366


namespace man_older_than_son_by_46_l36_36256

-- Given conditions about the ages
def sonAge : ℕ := 44

def manAge_in_two_years (M : ℕ) : Prop := M + 2 = 2 * (sonAge + 2)

-- The problem to verify
theorem man_older_than_son_by_46 (M : ℕ) (h : manAge_in_two_years M) : M - sonAge = 46 :=
by
  sorry

end man_older_than_son_by_46_l36_36256


namespace angle_ABG_in_heptagon_l36_36150

noncomputable def regular_heptagon_interior_angle (n : ℕ) : ℝ :=
  ((n - 2) * 180) / n

theorem angle_ABG_in_heptagon (ABCDEFG : Set (Fin 7 → ℝ × ℝ))
  (regular_heptagon : is_regular_polygon ABCDEFG 7)
  (A B G : (Fin 7 → ℝ × ℝ)) (h1: A ∈ ABCDEFG) (h2: B ∈ ABCDEFG) (h3: G ∈ ABCDEFG)
  (hAB : distance A B = distance A G) :
  angle A B G = 25.715 :=
by
  let interior_angle := regular_heptagon_interior_angle 7
  have h4 : interior_angle = (5 * 180) / 7 := rfl
  have h5 : 2 * angle A B G + interior_angle = 180 := sorry
  have h6 : 2 * angle A B G = 180 - interior_angle := sorry
  have h7 : angle A B G = (180 - interior_angle) / 2 := sorry
  have h8 : 180 - interior_angle = 51.43 := sorry
  have h9 : (180 - interior_angle) / 2 = 25.715 := sorry
  exact h9

end angle_ABG_in_heptagon_l36_36150


namespace min_difference_of_4_digit_numbers_l36_36306

theorem min_difference_of_4_digit_numbers :
  ∀ (a b c d e f g h : ℕ),
    {a, b, c, d, e, f, g, h} = {2, 3, 4, 5, 6, 7, 8, 9} →
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
     c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
     d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
     e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
     f ≠ g ∧ f ≠ h ∧
     g ≠ h) →
    ∃ x y : ℕ,
      x = 1000 * a + 100 * b + 10 * c + d ∧
      y = 1000 * e + 100 * f + 10 * g + h ∧
      247 = if x > y then x - y else y - x ∧
      x ≠ y := sorry

end min_difference_of_4_digit_numbers_l36_36306


namespace sum_of_number_and_permutation_not_all_ones_l36_36066

theorem sum_of_number_and_permutation_not_all_ones (N : ℕ) (digits : list ℕ)
  (hN_digits : ∀ d ∈ digits, d ≠ 0)
  (hN_number_conversion : list.nat_digits N = digits) : 
  ∀ N' : ℕ, N' ∈ list.permutations N → ¬(list.all (list.nat_digits (N + N')) (λ d => d = 1)) :=
by
  intros N' h_perm
  sorry

end sum_of_number_and_permutation_not_all_ones_l36_36066


namespace max_area_triangle_C_l36_36731

open Real

-- Definitions corresponding to the conditions
def isMidpoint (D A C : Point) : Prop := dist D A = dist D C ∧ collinear D A C
def distBD_one (B D : Point) : Prop := dist B D = 1
def angleB_2pi_over_3 (A B C : Point) : Prop := angle A B C = 2 * π / 3

-- Triangle defined by points A, B, C
variables (A B C D : Point)

-- Lean 4 statement of the problem
theorem max_area_triangle_C : 
  isMidpoint D A C → 
  distBD_one B D → 
  angleB_2pi_over_3 A B C → 
  (∃ (C : Triangle), area C ≤ sqrt 3) :=
sorry

end max_area_triangle_C_l36_36731


namespace median_line_eq_circle_eq_standard_l36_36584

-- Triangle Problem (Part 1)
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem median_line_eq {A B C M : ℝ × ℝ} :
  A = (2, 4) → B = (0, -2) → C = (-2, 3) →
  M = midpoint A B →
  (M = (1, 1)) →
  ∀ x y : ℝ, (2 * x + 3 * y - 5 = 0) :=
by
  intros
  sorry

-- Circle Problem (Part 2)
def circle_eq (E : ℝ × ℝ) (r : ℝ) :=
  (λ P : ℝ × ℝ, (P.1 - E.1) ^ 2 + (P.2 - E.2) ^ 2 = r ^ 2)

theorem circle_eq_standard {P Q : ℝ × ℝ} {E : ℝ × ℝ} :
  P = (0, -6) → Q = (1, -5) →
  E = (-3, -2) →
  circle_eq E 5 = (λ P, (P.1 + 3) ^ 2 + (P.2 + 2) ^ 2 = 25) :=
by
  intros
  sorry

end median_line_eq_circle_eq_standard_l36_36584


namespace largest_prime_factor_18_12_6_l36_36316

theorem largest_prime_factor_18_12_6 :
  let n : ℕ := 18^3 + 12^4 - 6^5
  in Nat.prime 23 ∧ ∀ p : ℕ, p.prime → p ∣ n → p ≤ 23 := by
  sorry

end largest_prime_factor_18_12_6_l36_36316


namespace tetrahedron_height_l36_36929

/-
We are given a tetrahedron with one vertex from which three pairwise perpendicular edges of lengths
9 cm, 12 cm, and 16 cm originate. We need to prove that the height from this vertex is approximately
6.5658 cm.
-/

noncomputable def height_of_tetrahedron (a b c : ℝ) : ℝ :=
  (a * b * c) / √((a^2 * b^2) + (a^2 * c^2) + (b^2 * c^2))

theorem tetrahedron_height
  (a b c : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 16) :
  height_of_tetrahedron a b c ≈ 6.5658 :=
by
  sorry

end tetrahedron_height_l36_36929


namespace largest_angle_in_triangle_l36_36412

theorem largest_angle_in_triangle
    (a b c : ℝ)
    (h_sum_two_angles : a + b = (7 / 5) * 90)
    (h_angle_difference : b = a + 40) :
    max a (max b c) = 83 :=
by
  sorry

end largest_angle_in_triangle_l36_36412


namespace parabola_focus_coordinates_l36_36924

theorem parabola_focus_coordinates (x y : ℝ) (h : x = 2 * y^2) : (x, y) = (1/8, 0) :=
sorry

end parabola_focus_coordinates_l36_36924


namespace problem1_problem2_problem3_problem4_l36_36198

theorem problem1 (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 20) : (1 / x + 1 / y) = 2 :=
sorry

theorem problem2 (a : ℝ) : ∃ b : ℝ, b = 6 * (a + 1)^2 :=
exists.intro (6 * (a + 1)^2) rfl

theorem problem3 (b c : ℝ) : ∀ (h1 : b - 4 ≥ 0) (h2 : b + 46 ≥ 0) (h3 : (b - 4) / (2 * b + 42) = c / 6), c = 2 :=
sorry

theorem problem4 (c d : ℝ) (h1 : d * Real.sqrt 3 = Real.sqrt 3 * (c^2 / 4)) : d = c^2 / 4 :=
sorry

end problem1_problem2_problem3_problem4_l36_36198


namespace find_PQ_l36_36897

noncomputable def collinear (P Q R S : Type) : Prop := sorry
noncomputable def distance (A B : Type) : ℝ := sorry
noncomputable def perimeter (A B C : Type) : ℝ := sorry

theorem find_PQ
  (P Q R S T : Type)
  (col_PQRS : collinear P Q R S)
  (distPQ : distance P Q = x)
  (distSR : distance S R = x)
  (distQR : distance Q R = 15)
  (not_on_line_T : ¬ collinear Q T R)
  (distQT : distance Q T = 13)
  (distRT : distance R T = 13)
  (perim_double : perimeter P T S = 2 * perimeter Q T R) :
  distance P Q = 11.625 :=
sorry

end find_PQ_l36_36897


namespace farmer_can_buy_40_pigs_l36_36250

def can_buy_pigs := ∃ p d : ℕ, (21 * p + 23 * d = 1300) ∧ d ≥ 20

theorem farmer_can_buy_40_pigs : can_buy_pigs → (∃ p : ℕ, p = 40) :=
by
  intro h
  obtain ⟨p, d, h_eq, h_d⟩ := h
  have hp : p = 40 := by 
    sorry
  use 40
  exact hp

end farmer_can_buy_40_pigs_l36_36250


namespace find_y_coordinate_l36_36459

open Real

noncomputable def pointA := (-4 : ℝ, 0 : ℝ)
noncomputable def pointB := (-1 : ℝ, 2 : ℝ)
noncomputable def pointC := (1 : ℝ, 2 : ℝ)
noncomputable def pointD := (4 : ℝ, 0 : ℝ)

def PA (P : ℝ × ℝ) : ℝ := dist P pointA
def PB (P : ℝ × ℝ) : ℝ := dist P pointB
def PC (P : ℝ × ℝ) : ℝ := dist P pointC
def PD (P : ℝ × ℝ) : ℝ := dist P pointD

def satisfies_conditions (P : ℝ × ℝ) : Prop :=
  PA P + PD P = 10 ∧ PB P + PC P = 10

theorem find_y_coordinate (P : ℝ × ℝ) (h : satisfies_conditions P) :
  ∃ a b c d : ℕ, a + b + c + d = 49 ∧
    (∃ y, P = (P.fst, y) ∧ y = (-12 + 16 * sqrt 16.5) / 5) :=
sorry

end find_y_coordinate_l36_36459


namespace OverlineP_Solution_l36_36074

noncomputable def pow := λ (x : ℝ), 2

theorem OverlineP_Solution (AD CD PB PA cA cB cC cD : ℝ):
  AD = 10 → 
  CD = 15 → 
  PB = 9 → 
  PA = 12 → 
  cA = 10 → 
  cB = sqrt 37 → 
  cC = 1 / 16 → 
  cD = 4 →
  (cA = 10 ∧ cB = sqrt 37 ∧ cC = 1 / 16 ∧ cD = 4) := 
begin
  sorry
end

end OverlineP_Solution_l36_36074


namespace max_odd_partial_sums_l36_36518

theorem max_odd_partial_sums (numbers : List ℕ) (h_perm : numbers ~ List.range' 1 30) :
  let partial_sums := List.scanl Nat.add 0 numbers
  partial_sums.drop 1 |> List.countp Nat.odd = 23 :=
sorry

end max_odd_partial_sums_l36_36518


namespace base_seven_sum_l36_36523

def base_seven_to_ten (n : ℕ) : ℕ := 3 * 7^1 + 5 * 7^0   -- Converts 35_7 to base 10
def base_seven_to_ten' (m : ℕ) : ℕ := 1 * 7^1 + 2 * 7^0  -- Converts 12_7 to base 10

noncomputable def base_ten_product (a b : ℕ) : ℕ := (a * b) -- Computes product in base 10

noncomputable def base_ten_to_seven (p : ℕ) : ℕ :=        -- Converts base 10 to base 7
  let p1 := (p / 7 / 7) % 7
  let p2 := (p / 7) % 7
  let p3 := p % 7
  p1 * 100 + p2 * 10 + p3

noncomputable def sum_of_digits (a : ℕ) : ℕ :=             -- Sums digits in base 7
  let d1 := (a / 100) % 10
  let d2 := (a / 10) % 10
  let d3 := a % 10
  d1 + d2 + d3

noncomputable def base_ten_to_seven' (s : ℕ) : ℕ :=        -- Converts sum back to base 7
  let s1 := s / 7
  let s2 := s % 7
  s1 * 10 + s2

theorem base_seven_sum (n m : ℕ) : base_ten_to_seven' (sum_of_digits (base_ten_to_seven (base_ten_product (base_seven_to_ten n) (base_seven_to_ten' m)))) = 15 :=
by
  sorry

end base_seven_sum_l36_36523


namespace labourer_saved_amount_l36_36501

noncomputable def average_expenditure_6_months : ℕ := 85
noncomputable def expenditure_reduction_4_months : ℕ := 60
noncomputable def monthly_income : ℕ := 78

theorem labourer_saved_amount :
  let initial_debt := 6 * average_expenditure_6_months - 6 * monthly_income
      cleared_debt := 4 * monthly_income - 4 * expenditure_reduction_4_months
      savings := cleared_debt - initial_debt
  in savings = 30 :=
by
  have average_expenditure : ℕ := 6 * 85
  have average_income : ℕ := 6 * 78
  have initial_debt : ℕ := average_expenditure - average_income
  have new_expenditure : ℕ := 4 * 60
  have new_income : ℕ := 4 * 78
  have cleared_debt : ℕ := new_income - new_expenditure
  have savings : ℕ := cleared_debt - initial_debt
  have amount_saved : ℕ := 30
  show savings = amount_saved
  sorry

end labourer_saved_amount_l36_36501


namespace vehicle_speed_increase_l36_36555

/-- Vehicle dynamics details -/
structure Vehicle := 
  (initial_speed : ℝ) 
  (deceleration : ℝ)
  (initial_distance_from_A : ℝ)

/-- Given conditions -/
def conditions (A B C : Vehicle) : Prop :=
  A.initial_speed = 80 ∧
  B.initial_speed = 60 ∧
  C.initial_speed = 70 ∧ 
  C.deceleration = 2 ∧
  B.initial_distance_from_A = 40 ∧
  C.initial_distance_from_A = 260

/-- Prove A needs to increase its speed by 5 mph -/
theorem vehicle_speed_increase (A B C : Vehicle) (h : conditions A B C) : 
  ∃ dA : ℝ, dA = 5 ∧ A.initial_speed + dA > B.initial_speed → 
    (A.initial_distance_from_A / (A.initial_speed + dA - B.initial_speed)) < 
    (C.initial_distance_from_A / (A.initial_speed + dA + C.initial_speed - C.deceleration)) :=
sorry

end vehicle_speed_increase_l36_36555


namespace fixed_point_of_tangent_line_l36_36286

theorem fixed_point_of_tangent_line (x y : ℝ) (h1 : x = 3) 
  (h2 : ∃ m : ℝ, (3 - m)^2 + (y - 2)^2 = 4) :
  ∃ (k l : ℝ), k = 4 / 3 ∧ l = 2 :=
by
  sorry

end fixed_point_of_tangent_line_l36_36286


namespace find_x_l36_36160

variables (R S : Fin₉ → ℝ) (x : ℝ) (R_range : ℝ)

-- Define set S in terms of set R and x
def Si (i : Fin₉) : ℝ := R i + x

-- Mean of set R
def mean_R : ℝ := (1/9) * (Fin₉.sum R)

-- Mean of set S
def mean_S : ℝ := (1/9) * (Fin₉.sum Si)

-- Given condition: mean and range relationship
axiom condition : mean_S + R_range = mean_R + R_range + 4

-- Theorem we aim to prove
theorem find_x : x = 4 := by
  sorry

end find_x_l36_36160


namespace maritza_study_hours_l36_36873

noncomputable def time_to_study_for_citizenship_test (num_mc_questions num_fitb_questions time_mc time_fitb : ℕ) : ℕ :=
  (num_mc_questions * time_mc + num_fitb_questions * time_fitb) / 60

theorem maritza_study_hours :
  time_to_study_for_citizenship_test 30 30 15 25 = 20 :=
by
  sorry

end maritza_study_hours_l36_36873


namespace change_is_correct_l36_36666

-- Define the cost of the pencil in cents
def cost_of_pencil : ℕ := 35

-- Define the amount paid in cents
def amount_paid : ℕ := 100

-- State the theorem for the change
theorem change_is_correct : amount_paid - cost_of_pencil = 65 :=
by sorry

end change_is_correct_l36_36666


namespace probability_of_rolling_divisor_of_12_l36_36598

def is_divisor (a b : ℕ) : Prop := b % a = 0

noncomputable def probability_divisor_12 : ℚ :=
  let divisors := { n | is_divisor n 12 ∧ n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} }
  let favorable_outcomes := (divisors : set ℕ).to_finset.card
  let total_outcomes := 12
  (favorable_outcomes : ℚ) / total_outcomes

theorem probability_of_rolling_divisor_of_12 : probability_divisor_12 = 1 / 2 := by
  sorry

end probability_of_rolling_divisor_of_12_l36_36598


namespace union_of_sets_l36_36794

def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {3, 4, 5}

theorem union_of_sets : M ∪ N = {2, 3, 4, 5} := by
  sorry

end union_of_sets_l36_36794


namespace _l36_36434

noncomputable def ellipse_parameters
  (a b : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hab : a > b) :
  (ℝ × ℝ) :=
if h : a^2 - b^2 = 1 ∧ 4/(9*a^2) + 8/(3*b^2) = 1 
then (a^2, b^2)
else (0, 0)

noncomputable def max_quadrilateral_area
  (a b x1 x2 y2 : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hA : (b, 0) ∈ set.univ)
  (hB : (0, a) ∈ set.univ)
  (hE : (x1, x1^2) ∈ {p : ℝ × ℝ | p.1^2 / 3 + p.2^2 / 4 = 1})
  (hF : (x2, -x1^2) ∈ {p : ℝ × ℝ | p.1^2 / 3 + p.2^2 / 4 = 1})
  (hx2_pos : x2 > 0)
  (hy2_pos : y2 > 0)
  (area_condition : 4 * x2^2 + 3 * y2^2 = 12) :
  ℝ :=
if 2 * x2 = sqrt 3 * y2
then 2 * sqrt 6
else 0

noncomputable theorem max_area_of_quadrilateral_AEBF
  (a b : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hab : a > b)
  (x1 x2 y2 : ℝ)
  (hA : (b, 0) ∈ set.univ)
  (hB : (0, a) ∈ set.univ)
  (hE : (x1, x1^2) ∈ {p : ℝ × ℝ | p.1^2 / 3 + p.2^2 / 4 = 1})
  (hF : (x2, -x1^2) ∈ {p : ℝ × ℝ | p.1^2 / 3 + p.2^2 / 4 = 1})
  (hx2_neg : x1 < 0)
  (hx2_pos : x2 > 0)
  (hy2_pos : y2 > 0)
  (area_condition : 4 * x2^2 + 3 * y2^2 = 12) :
  ellipse_parameters a b ha hb hab = (4, 3)
  ∧ max_quadrilateral_area a b x1 x2 y2 ha hb hA hB hE hF hx2_pos hy2_pos area_condition = 2 * sqrt 6 :=
by
  sorry

end _l36_36434


namespace larger_tv_diagonal_l36_36194

theorem larger_tv_diagonal (d : ℝ) (h1 : ∀ d, d > 0 → ∃ (A : ℝ), A = (d / real.sqrt 2) ^ 2)
                           (h2 : ∀ A1 A2, ∃ (d1 : ℝ), d1 = 25 / real.sqrt 2 ∧ ∃ (d2 : ℝ), d2 = d / real.sqrt 2 ∧ A2 = A1 + 79.5) :
  d = 28 :=
by
  sorry

end larger_tv_diagonal_l36_36194


namespace probability_fourth_term_integer_l36_36106

/-- The probability of the fourth term in Jacob's sequence being an integer given the specified rules. -/
theorem probability_fourth_term_integer :
  let initial_term : ℕ := 6
  ∀ (coin_flip : ℕ → bool), 
    let next_term (a : ℚ) (flip : bool) : ℚ :=
      if flip then (2 * a - 1) else (a / 2 - 1)
    let sequence : ℕ → ℚ
    | 0 => initial_term
    | n + 1 => next_term (sequence n) (coin_flip n)
  ∃ outcomes : list ℚ,
    (sequence 3) ∈ outcomes ∧
    let integer_count := outcomes.countp (λ x, x.den = 1)
    let total_count := outcomes.length
    (integer_count / total_count) = (5 / 8) :=
begin
  sorry
end

end probability_fourth_term_integer_l36_36106


namespace gabrielle_saw_more_birds_l36_36224

def birds_seen (robins cardinals blue_jays : Nat) : Nat :=
  robins + cardinals + blue_jays

def percentage_difference (g c : Nat) : Nat :=
  ((g - c) * 100) / c

theorem gabrielle_saw_more_birds :
  let gabrielle := birds_seen 5 4 3
  let chase := birds_seen 2 5 3
  percentage_difference gabrielle chase = 20 := 
by
  sorry

end gabrielle_saw_more_birds_l36_36224


namespace equal_areas_of_triangles_in_parallel_hexagon_l36_36519

theorem equal_areas_of_triangles_in_parallel_hexagon {A B C D E F : Type*} 
  [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C]
  [AffineSpace ℝ D] [AffineSpace ℝ E] [AffineSpace ℝ F] 
  (hABpDE : ∃ t : ℝ, ∀ p : A, ∃ q : D, p + t = q) -- A parallel to D
  (hBCpEF : ∃ t : ℝ, ∀ p : B, ∃ q : E, p + t = q) -- B parallel to E
  (hCDpFA : ∃ t : ℝ, ∀ p : C, ∃ q : F, p + t = q) -- C parallel to F
  : ∀ (P Q R : affine_space Affine ℝ A) (U V W : affine_space Affine ℝ B) (X Y Z : affine_space Affine ℝ C), 
    area P Q R = area U V W := 
  sorry

end equal_areas_of_triangles_in_parallel_hexagon_l36_36519


namespace sqrt_of_neg_five_squared_l36_36685

theorem sqrt_of_neg_five_squared : Real.sqrt ((-5 : Real) ^ 2) = 5 := 
by 
  sorry

end sqrt_of_neg_five_squared_l36_36685


namespace packs_of_red_bouncy_balls_l36_36441

/-- Given the following conditions:
1. Kate bought 6 packs of yellow bouncy balls.
2. Each pack contained 18 bouncy balls.
3. Kate bought 18 more red bouncy balls than yellow bouncy balls.
Prove that the number of packs of red bouncy balls Kate bought is 7. -/
theorem packs_of_red_bouncy_balls (packs_yellow : ℕ) (balls_per_pack : ℕ) (extra_red_balls : ℕ)
  (h1 : packs_yellow = 6)
  (h2 : balls_per_pack = 18)
  (h3 : extra_red_balls = 18)
  : (packs_yellow * balls_per_pack + extra_red_balls) / balls_per_pack = 7 :=
by
  sorry

end packs_of_red_bouncy_balls_l36_36441


namespace train_pass_time_l36_36233

-- Definitions based on the conditions
def train_length : ℕ := 280  -- train length in meters
def train_speed_kmh : ℕ := 72  -- train speed in km/hr
noncomputable def train_speed_ms : ℚ := (train_speed_kmh * 5 / 18)  -- train speed in m/s

-- Theorem statement
theorem train_pass_time : (train_length / train_speed_ms) = 14 := by
  sorry

end train_pass_time_l36_36233


namespace tip_calculation_correct_l36_36651

noncomputable def calculate_tip (total_with_tax : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  let bill_before_tax := total_with_tax / (1 + tax_rate)
  bill_before_tax * tip_rate

theorem tip_calculation_correct :
  calculate_tip 226 0.13 0.15 = 30 := 
by
  sorry

end tip_calculation_correct_l36_36651


namespace likes_apple_l36_36983

open Finset

variables {U : Type} [Fintype U]

def number_likes_apple (A O M : Finset U) :=
  card A

-- Conditions
variable (A O M : Finset U)
variable (h1 : card A = 40)
variable (h2 : card (O ∩ M \ A) = 7)
variable (h3 : card (M ∩ A \ O) = 10)
variable (h4 : card (A ∩ O ∩ M) = 4)

-- The Theorem (Proof Problem)
theorem likes_apple (A O M : Finset U) 
  (h1 : card A = 40) (h2 : card (O ∩ M \ A) = 7) 
  (h3 : card (M ∩ A \ O) = 10) (h4 : card (A ∩ O ∩ M) = 4) : 
  number_likes_apple A O M = 40 := 
by
  exact h1

end likes_apple_l36_36983


namespace part_I_part_II_l36_36865

noncomputable def f (x : ℝ) (a : ℝ) := Real.log x + x^2 - a * x

theorem part_I (x : ℝ) (a : ℝ) (h_inc : ∀ x > 0, (1/x + 2*x - a) ≥ 0) : a ≤ 2 * Real.sqrt 2 :=
sorry

noncomputable def g (x : ℝ) (a : ℝ) := f x a + 2 * Real.log ((a * x + 2) / (6 * Real.sqrt x))

theorem part_II (a : ℝ) (k : ℝ) (h_a : 2 < a ∧ a < 4) (h_ex : ∃ x : ℝ, (3/2) ≤ x ∧ x ≤ 2 ∧ g x a > k * (4 - a^2)) : k ≥ 1/3 :=
sorry

end part_I_part_II_l36_36865


namespace production_cost_decrease_l36_36191

theorem production_cost_decrease (x : ℝ) :
  let initial_production_cost := 50
  let initial_selling_price := 65
  let first_quarter_decrease := 0.10
  let second_quarter_increase := 0.05
  let final_selling_price := initial_selling_price * (1 - first_quarter_decrease) * (1 + second_quarter_increase)
  let original_profit := initial_selling_price - initial_production_cost
  let final_production_cost := initial_production_cost * (1 - x) ^ 2
  (final_selling_price - final_production_cost) = original_profit :=
by
  sorry

end production_cost_decrease_l36_36191


namespace largest_n_arithmetic_sequences_l36_36677

theorem largest_n_arithmetic_sequences
  (a : ℕ → ℤ) (b : ℕ → ℤ) (x y : ℤ)
  (a_1 : a 1 = 2) (b_1 : b 1 = 3)
  (a_formula : ∀ n : ℕ, a n = 2 + (n - 1) * x)
  (b_formula : ∀ n : ℕ, b n = 3 + (n - 1) * y)
  (x_lt_y : x < y)
  (product_condition : ∃ n : ℕ, a n * b n = 1638) :
  ∃ n : ℕ, a n * b n = 1638 ∧ n = 35 := 
sorry

end largest_n_arithmetic_sequences_l36_36677


namespace ship_length_proof_l36_36567

noncomputable def ship_length (speed_kmh : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  speed_ms * time_sec

theorem ship_length_proof : ship_length 18 20 = 100 :=
by
  unfold ship_length
  norm_num
  sorry

end ship_length_proof_l36_36567


namespace abs_sum_of_c_l36_36298

def canBeFactoredOverIntegers (c : ℤ) : Prop :=
  ∃ p q : ℤ, (x^2 + c * x + 3024 * c) = (x - p) * (x - q)

theorem abs_sum_of_c (h : ∀ c : ℤ, canBeFactoredOverIntegers c) : abs (∑ c in {c : ℤ | canBeFactoredOverIntegers c}, c) = 0 :=
  sorry

end abs_sum_of_c_l36_36298


namespace shaded_region_area_l36_36700

noncomputable def radius1 := 4
noncomputable def radius2 := 5
noncomputable def distance := radius1 + radius2
noncomputable def large_radius := radius2 + distance / 2

theorem shaded_region_area :
  ∃ (A : ℝ), A = (π * large_radius ^ 2) - (π * radius1 ^ 2) - (π * radius2 ^ 2) ∧
  A = 49.25 * π :=
by
  sorry

end shaded_region_area_l36_36700


namespace answered_both_correctly_l36_36059

variables {α : Type*}

-- Definitions of probabilities
def P (event : α → Prop) [fintype α] [decidable_pred event] : ℝ :=
  (fintype.card {a // event a}).to_real / (fintype.card α).to_real

-- Given conditions
axiom P_A : P (λ a : α, a ∈ A) = 0.75
axiom P_B : P (λ a : α, a ∈ B) = 0.65
axiom P_not_A_and_not_B : P (λ a : α, ¬ (a ∈ A) ∧ ¬ (a ∈ B)) = 0.20

-- To prove
theorem answered_both_correctly : P (λ a : α, a ∈ A ∧ a ∈ B) = 0.60 :=
sorry

end answered_both_correctly_l36_36059


namespace cos_theta_value_l36_36431

-- Given conditions
variables {A B C θ : ℝ} {a b c : ℝ}
variables (h1 : a = b * sin A / sin B ∧ b = a * sin B / sin A ∧ c = a * sin C / sin A) -- Law of Sines relations
variables (h2 : 2 * a * cos(θ - B) + 2 * b * cos(θ + A) + c = 0) -- Given equation

-- To Prove
theorem cos_theta_value : cos θ = -1 / 2 :=
by sorry

end cos_theta_value_l36_36431


namespace parabola_chord_intercepted_length_l36_36358

noncomputable def parabola_chord_length : ℝ :=
  let F := (2, 0) in -- Focus of the parabola
  let θ := Real.pi / 3 in -- 60 degrees in radians
  let k := Real.tan θ in -- Slope of the line passing through F
  let line_eq (x : ℝ) := k * (x - F.1) in -- Equation of the line
  let parabola_eq (x y : ℝ) := y^2 = 8 * x in -- Parabola equation
  let Δ := k^2 in -- Discriminant parts using the slope
  let a := Δ in
  let b := -2 * Δ * F.1 in
  let c := Δ * F.1^2 - 8 in
  let roots_sum := -b / a in -- Sum of roots x₁ + x₂
  roots_sum + 4 / 3 -- Substituted the solution found above

theorem parabola_chord_intercepted_length (parabola_eq : ℝ → ℝ → Prop)
  (F : ℝ × ℝ)
  (θ : ℝ)
  (length : ℝ) : 
  (parabola_eq y^2 = 8x) → 
  (F = (2, 0)) → 
  (θ = Real.pi / 3) →
  (length = 32 / 3) :=
by
  intros parabola_eq F θ length
  have k := Real.tan (Real.pi / 3)
  have line_eq := (λ x : ℝ, k * (x - F.1))
  sorry -- Here, we would provide the proof based on the conditions

end parabola_chord_intercepted_length_l36_36358


namespace eating_contest_l36_36879

variables (hotdog_weight burger_weight pie_weight : ℕ)
variable (noah_burgers jacob_pies mason_hotdogs : ℕ)
variable (total_weight_mason_hotdogs : ℕ)

theorem eating_contest :
  hotdog_weight = 2 →
  burger_weight = 5 →
  pie_weight = 10 →
  noah_burgers = 8 →
  jacob_pies = noah_burgers - 3 →
  mason_hotdogs = 3 * jacob_pies →
  total_weight_mason_hotdogs = mason_hotdogs * hotdog_weight →
  total_weight_mason_hotdogs = 30 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end eating_contest_l36_36879


namespace solution_set_of_inequality_system_l36_36529

theorem solution_set_of_inequality_system (x : ℝ) :
  (3 * x - 1 ≥ x + 1) ∧ (x + 4 > 4 * x - 2) ↔ (1 ≤ x ∧ x < 2) := 
by
  sorry

end solution_set_of_inequality_system_l36_36529


namespace cherries_per_pound_l36_36208

-- Definitions from conditions in the problem
def total_pounds_of_cherries : ℕ := 3
def pitting_time_for_20_cherries : ℕ := 10 -- in minutes
def total_pitting_time : ℕ := 2 * 60  -- in minutes (2 hours to minutes)

-- Theorem to prove the question equals the correct answer
theorem cherries_per_pound : (total_pitting_time / pitting_time_for_20_cherries) * 20 / total_pounds_of_cherries = 80 := by
  sorry

end cherries_per_pound_l36_36208


namespace odd_numbers_multiple_of_4_and_sum_of_squares_const_l36_36858

def E : Set ℕ := {x ∈ Finset.range 201 | x > 0}

variables {G : Finset ℕ} (hGsub : G ⊆ E) (hGcard : G.card = 100)
           (hSum : G.sum id = 10080) (hProp : ∀ {a b}, a ∈ G → b ∈ G → a < b → a + b ≠ 201)

theorem odd_numbers_multiple_of_4_and_sum_of_squares_const :
  (∃ k, nat.odd k ∧ k % 4 = 0 ∧ ∑ i in G, i^2 = 1349380) := sorry

end odd_numbers_multiple_of_4_and_sum_of_squares_const_l36_36858


namespace sum_of_reciprocals_l36_36538

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) :
  1 / x + 1 / y = 3 :=
by
  sorry

end sum_of_reciprocals_l36_36538


namespace distance_focus_parabola_to_hyperbola_asymptotes_l36_36174

noncomputable def focus_parabola (y : ℤ) : Prop := ∀ x : ℤ, y^2 = 8 * x → (x, y) = (2, 0)

noncomputable def hyperbola_asymptotes (x y : ℤ) : Prop :=
  x^2 - (y^2 / 3) = 1 → (y = sqrt 3 * x) ∨ (y = -sqrt 3 * x)

theorem distance_focus_parabola_to_hyperbola_asymptotes :
  ∀ (x y : ℤ), focus_parabola y → hyperbola_asymptotes x y → dist (2, 0) (x, y) = sqrt 3 :=
begin
  sorry
end

end distance_focus_parabola_to_hyperbola_asymptotes_l36_36174


namespace range_of_expression_l36_36524

theorem range_of_expression (x : ℝ) (h : x ≥ 5) : 2 * x - real.sqrt (x - 5) := by
  sorry

end range_of_expression_l36_36524


namespace coeff_x3_in_expansion_l36_36080

theorem coeff_x3_in_expansion : (nat.choose 50 3 * (1^47) * (1^3)) = 19600 := 
by sorry

end coeff_x3_in_expansion_l36_36080


namespace leila_cakes_monday_l36_36443

def number_of_cakes_monday (m : ℕ) : Prop :=
  let cakes_friday := 9
  let cakes_saturday := 3 * m
  let total_cakes := m + cakes_friday + cakes_saturday
  total_cakes = 33

theorem leila_cakes_monday : ∃ m : ℕ, number_of_cakes_monday m ∧ m = 6 :=
by 
  -- We propose that the number of cakes she ate on Monday, denoted as m, is 6.
  -- We need to prove that this satisfies the given conditions.
  -- This line is a placeholder for the proof.
  sorry

end leila_cakes_monday_l36_36443


namespace find_AB_l36_36578

theorem find_AB (O A B C : Point) (a b : ℝ) (hBC: length B C = a) (hAC : length A C = b) (hAOB : ∠ A O B = 120) :
  length A B = Real.sqrt (a^2 + b^2 - a * b) :=
sorry

end find_AB_l36_36578
