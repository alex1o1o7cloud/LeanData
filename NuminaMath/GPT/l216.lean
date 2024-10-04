import Mathlib
import Mathlib.Algebra.Combinatorics.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Invertible
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.ContDiff
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Calculus.ParametricIntervalIntegral
import Mathlib.Analysis.Calculus.TangentCone
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.VectorSpace
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.CombinatorialLibrary
import Mathlib.Combinatorics.Pascal
import Mathlib.Combinatorics.Prob
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.ModEq
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Base
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Lcm
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Circumcenter
import Mathlib.Geometry.Euclidean.Cube
import Mathlib.Geometry.EuclideanSpace
import Mathlib.NumberTheory.Digits.Modular
import Mathlib.Probability
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Relation.Rfl
import Mathlib.Topology.Basic
import data.real.basic
import tactic

namespace maximize_profit_l216_216759

def cost_A : ℝ := 3
def price_A : ℝ := 3.3
def cost_B : ℝ := 2.4
def price_B : ℝ := 2.8
def total_devices : ℕ := 50

def profit (x : ℕ) : ℝ := (price_A - cost_A) * x + (price_B - cost_B) * (total_devices - x)

def functional_relationship (x : ℕ) : ℝ := -0.1 * x + 20

def purchase_condition (x : ℕ) : Prop := 4 * x ≥ total_devices - x

theorem maximize_profit :
    functional_relationship (10) = 19 ∧ 
    (∀ x : ℕ, purchase_condition x → functional_relationship x ≤ 19) :=
by {
    -- Proof omitted
    sorry
}

end maximize_profit_l216_216759


namespace range_of_a12_l216_216766

variable (a : ℕ → ℝ)
variable (a1 d : ℝ)

def arithmetic_seq (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 + n * d

variable (h_arith_seq : arithmetic_seq a a1 d)
variable (h_a8 : a 7 ≥ 15)
variable (h_a9 : a 8 ≤ 13)

theorem range_of_a12 : ∀ a1 d, (arithmetic_seq a a1 d) → (a 7 ≥ 15) → (a 8 ≤ 13) → (a 11 ≤ 7) :=
by
  intro a1 d h_arith_seq h_a8 h_a9
  sorry

end range_of_a12_l216_216766


namespace perimeter_of_wheel_K_l216_216936

theorem perimeter_of_wheel_K
  (L_turns_K : 4 / 5 = 1 / (length_of_K / length_of_L))
  (L_turns_M : 6 / 7 = 1 / (length_of_L / length_of_M))
  (M_perimeter : length_of_M = 30) :
  length_of_K = 28 := 
sorry

end perimeter_of_wheel_K_l216_216936


namespace f_8pi_over_3_l216_216499

noncomputable def f : ℝ → ℝ := sorry  -- To be given in the conditions below

theorem f_8pi_over_3 :
  (∀ x, f (-x) = f x) ∧
  (∀ k : ℤ, ∀ x, f (x + k * real.pi) = f x) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x < real.pi / 2 → f x = real.sqrt 3 * real.tan x - 1) →
  f (8 * real.pi / 3) = 2 :=
by
  sorry

end f_8pi_over_3_l216_216499


namespace suitable_proportional_function_l216_216333

theorem suitable_proportional_function (k : ℝ) (y x : ℝ) (h₁ : y = k * x) (h₂ : ∃ f : ℝ → ℝ, (∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂) ∧ (f = λ x, k * x)) : ∃ k, k < 0 ∧ ∃ f : ℝ → ℝ, f = λ x, k * x ∧ ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂ :=
sorry

end suitable_proportional_function_l216_216333


namespace dropping_more_than_eating_l216_216771

theorem dropping_more_than_eating (n : ℕ) : n = 20 → (n * (n + 1)) / 2 > 10 * n := by
  intros h
  rw [h]
  sorry

end dropping_more_than_eating_l216_216771


namespace composite_of_n_gt_one_l216_216485

theorem composite_of_n_gt_one (n : ℕ) (h : n > 1) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n^4 + 4 = a * b :=
by
  sorry

end composite_of_n_gt_one_l216_216485


namespace largest_non_decomposable_proof_l216_216426

def coin_denomination (n : ℕ) : list ℕ :=
  (List.range (n + 1)).map (λ k => (5 ^ (n - k)) * (7 ^ k))

def n_decomposable (n s : ℕ) : Prop :=
  ∃ (a : Fin (n + 1) → ℕ), (∑ i, a i * (coin_denomination n).get i) = s

def largest_nondecomposable_val (n : ℕ) : ℕ :=
  2 * (7^(n+1)) - 3 * (5^(n+1))

theorem largest_non_decomposable_proof {n : ℕ} : 
  ∀ s : ℕ, s > largest_nondecomposable_val n → n_decomposable n s :=
sorry

end largest_non_decomposable_proof_l216_216426


namespace percentage_decrease_wednesday_l216_216471

variables (W : ℕ)

-- Conditions
axiom monday_distance : 40
axiom tuesday_distance : 50
axiom wednesday_lt_tuesday : W < 50
axiom thursday_distance : 40 + W
axiom total_distance : 40 + 50 + W + (40 + W) = 180

-- Theorem Statement
theorem percentage_decrease_wednesday (W : ℕ)
  (monday_distance : 40)
  (tuesday_distance : 50)
  (wednesday_lt_tuesday : W < 50)
  (thursday_distance : 40 + W)
  (total_distance : 40 + 50 + W + (40 + W) = 180) :
  ((50 - W) * 100 / 50 = 50) :=
sorry

end percentage_decrease_wednesday_l216_216471


namespace round_47_951_to_nearest_tenth_l216_216481

def round_to_nearest_tenth (x : ℝ) : ℝ :=
  (Real.floor (10 * x) + if x - Real.floor x ≥ 0.05 then 1 else 0) / 10.0

theorem round_47_951_to_nearest_tenth :
  round_to_nearest_tenth 47.951 = 48.0 :=
by
  sorry

end round_47_951_to_nearest_tenth_l216_216481


namespace equation_of_circle_l216_216073

theorem equation_of_circle :
  let center := (1, 1)
  let point := (2, 2)
  ∃ (r : ℝ), r = real.sqrt ((2 - 1)^2 + (2 - 1)^2) ∧ 
             ∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = r^2 :=
by
  -- Define the center and the point it passes through
  let center := (1, 1)
  let point := (2, 2)

  -- Calculate radius
  have radius : ℝ := real.sqrt ((2 - 1)^2 + (2 - 1)^2)
  
  -- Statement of the theorem
  exact ⟨radius, rfl, sorry⟩

end equation_of_circle_l216_216073


namespace problem_BCD_l216_216364

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := a^(2 * x) - x

theorem problem_BCD (a : ℝ) :
  (a > 0) ∧ (a ≠ 1) →
  (∀ x, a = Real.exp → f a x > Real.log x + 2) ∧
  (∀ x1 x2, f a x1 = 0 ∧ f a x2 = 0 → 1 < a ∧ a < Real.exp (2 * Real.exp⁻¹)) ∧
  (∀ t x0, a^2 * x0 = t ∧ f a x0 = 0 → 0 < t ∧ t < 1) :=
begin
  sorry
end

end problem_BCD_l216_216364


namespace maximize_profit_at_200_l216_216356

noncomputable def cost (q : ℝ) : ℝ := 50000 + 200 * q
noncomputable def price (q : ℝ) : ℝ := 24200 - (1/5) * q^2
noncomputable def profit (q : ℝ) : ℝ := (price q) * q - (cost q)

theorem maximize_profit_at_200 : ∃ (q : ℝ), q = 200 ∧ ∀ (x : ℝ), x ≥ 0 → profit q ≥ profit x :=
by
  sorry

end maximize_profit_at_200_l216_216356


namespace rectangle_area_l216_216174

theorem rectangle_area (x : ℝ) (hx1 : 0 < x) (hx2 : x < 5) (hperimeter : 2 * (x + (5 - x)) = 10) :
  ∃ y : ℝ, y = -x^2 + 5 * x :=
by
  use -x^2 + 5 * x
  sorry

end rectangle_area_l216_216174


namespace evaluate_i_powers_sum_l216_216647

-- Given conditions: i is the imaginary unit
def i : ℂ := Complex.I

-- Proof problem: Prove that i^2023 + i^2024 + i^2025 + i^2026 = 0
theorem evaluate_i_powers_sum : i^2023 + i^2024 + i^2025 + i^2026 = 0 := 
by sorry

end evaluate_i_powers_sum_l216_216647


namespace beef_weight_before_processing_l216_216568

theorem beef_weight_before_processing
    (lost_fraction : ℝ)
    (weight_after_processing : ℝ)
    (weight_before_processing : ℝ) :
    lost_fraction = 0.35 →
    weight_after_processing = 560 →
    weight_before_processing = (weight_after_processing / (1 - lost_fraction)) →
    weight_before_processing = 861.54 :=
by
  intros h_lost h_after h_before
  rw h_lost at h_before
  rw h_after at h_before
  exact h_before

end beef_weight_before_processing_l216_216568


namespace factory_produces_correct_number_of_doors_l216_216584

variable (initial_planned_production : ℕ) (metal_shortage_decrease : ℕ) (pandemic_decrease_factor : ℕ)
variable (doors_per_car : ℕ)

theorem factory_produces_correct_number_of_doors
  (h1 : initial_planned_production = 200)
  (h2 : metal_shortage_decrease = 50)
  (h3 : pandemic_decrease_factor = 50)
  (h4 : doors_per_car = 5) :
  (initial_planned_production - metal_shortage_decrease) * (100 - pandemic_decrease_factor) * doors_per_car / 100 = 375 :=
by
  sorry

end factory_produces_correct_number_of_doors_l216_216584


namespace no_member_of_T_divisible_by_9_but_some_member_divisible_by_5_l216_216025

def is_sum_of_squares_of_consecutive_integers (n : ℤ) : ℤ :=
  (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2

def T (x : ℤ) : Prop :=
  ∃ n : ℤ, x = is_sum_of_squares_of_consecutive_integers n

theorem no_member_of_T_divisible_by_9_but_some_member_divisible_by_5 :
  (∀ x, T x → ¬ (9 ∣ x)) ∧ (∃ y, T y ∧ (5 ∣ y)) :=
by
  sorry

end no_member_of_T_divisible_by_9_but_some_member_divisible_by_5_l216_216025


namespace polynomial_simplification_l216_216914

variable (x : ℝ)

theorem polynomial_simplification :
  (3*x^3 + 4*x^2 + 12)*(x + 1) - (x + 1)*(2*x^3 + 6*x^2 - 42) + (6*x^2 - 28)*(x + 1)*(x - 2) = 
  7*x^4 - 7*x^3 - 42*x^2 + 82*x + 110 :=
by
  sorry

end polynomial_simplification_l216_216914


namespace flagpole_arrangements_div_mod_l216_216883

noncomputable def num_arrangements (b g : ℕ) : ℕ :=
  (b - 1) * choose (b + 2) g - 2 * choose (b + 1) g

theorem flagpole_arrangements_div_mod (b g : ℕ) (h_b : b = 14) (h_g : g = 11) :
  num_arrangements b g % 1000 = 54 :=
by
  -- Ensuring values are plugged in to achieve the final form
  rw [h_b, h_g],
  unfold num_arrangements,
  -- Insert the appropriate combinatorial functions and solve
  -- We use specific values for combinations and arithmetic to match the provided problem.
  sorry

end flagpole_arrangements_div_mod_l216_216883


namespace lizzy_score_l216_216756

theorem lizzy_score
  (max_score : ℕ)
  (gibi_score_percent jigi_score_percent mike_score_percent : ℕ)
  (avg_score : ℕ)
  (total_students : ℕ)
  (total_percent : ℕ)
  (h_gibi : gibi_score_percent = 59)
  (h_jigi : jigi_score_percent = 55)
  (h_mike : mike_score_percent = 99)
  (h_max_score : max_score = 700)
  (h_avg_score : avg_score = 490)
  (h_total_students : total_students = 4)
  (h_total_percent : total_percent = 67) :
  let
    gibi_score := (gibi_score_percent * max_score) / 100,
    jigi_score := (jigi_score_percent * max_score) / 100,
    mike_score := (mike_score_percent * max_score) / 100,
    total_marks := avg_score * total_students,
    lizzy_score := total_marks - (gibi_score + jigi_score + mike_score),
    lizzy_percent := (lizzy_score * 100) / max_score
  in
  lizzy_percent = total_percent := by
    sorry

end lizzy_score_l216_216756


namespace equilateral_triangles_in_lattice_l216_216632

theorem equilateral_triangles_in_lattice : 
  let lattice := {point : ℝ × ℝ // ∃ i j : ℤ, point = (i + j / 2, j * (√3 / 2))} in
  let mini_triangles := {point : ℝ × ℝ // ∃ i j : ℤ, point = (i + j / 2 + 1/4, j * (√3 / 2) + (√3 / 4))} in
  let count_side_1 := 6 in
  let count_side_sqrt3 := 2 in
  let count_side_half := 6 in
  count_side_1 + count_side_sqrt3 + count_side_half = 14

:= by
  sorry

end equilateral_triangles_in_lattice_l216_216632


namespace trajectory_equation_l216_216685

def point (x y : ℝ) := (x, y)
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/- Point definitions -/
def A := point (-2 : ℝ) 0
def B := point (2 : ℝ) 0

/- Main condition -/
def satisfies_condition (P : ℝ × ℝ) : Prop :=
  distance P A = real.sqrt 2 * distance P B

/- Theorem statement -/
theorem trajectory_equation (P : ℝ × ℝ) (h : satisfies_condition P) : 
  P.1 ^ 2 + P.2 ^ 2 - 12 * P.1 + 4 = 0 :=
sorry

end trajectory_equation_l216_216685


namespace math_problem_l216_216837

theorem math_problem (a b c x₀ : ℝ) (h_a_lt_b : a < b) (h_x₀_pos : x₀ > 0) :
  (¬ (a < b → a * c^2 < b * c^2)) ∧ (∃ x₀, x₀ > 0 ∧ x₀ - 1 - log x₀ = 0) :=
by
  split
  { intro h
    apply h
    exact h_a_lt_b
    sorry },
  { use x₀,
    split,
    exact h_x₀_pos,
    sorry }

end math_problem_l216_216837


namespace max_rational_products_in_table_l216_216047

theorem max_rational_products_in_table :
  ∃ table : (Fin 50 → Rational ⊕ Irrational) × (Fin 50 → Rational ⊕ Irrational) → Matrix (Fin 50) (Fin 50) ℝ,
  let products := λ table : (Fin 50 → Rational ⊕ Irrational) × (Fin 50 → Rational ⊕ Irrational), 
                  (∀ (i : Fin 50), ∃ (r : Rational), table.1 i = r ∨ table.2 i = r) ∧ 
                  (∀ (i : Fin 50), ∃ (irr : Irrational), table.1 i = irr ∨ table.2 i = irr) ∧
                  (∀ (i j : Fin 50), table.1 i * table.2 j) in
  card { p : Prod (Fin 50) (Fin 50) | is_rational (products p.1 p.2) } = 1275 :=
sorry

end max_rational_products_in_table_l216_216047


namespace parallel_vectors_condition_l216_216714

variables {𝕜 : Type*} [Field 𝕜] [AddCommGroup (𝕜 × 𝕜)] [Module 𝕜 (𝕜 × 𝕜)]

def non_collinear (a b : 𝕜 × 𝕜) : Prop :=
  ∃ k : 𝕜, k • a + b ≠ 0

theorem parallel_vectors_condition
  (a b : 𝕜 × 𝕜) (λ μ : 𝕜)
  (h₀ : non_collinear a b)
  (h₁ : λ ∈ 𝕜)
  (h₂ : μ ∈ 𝕜)
  (h₃ : ∃ k : 𝕜, λ • a + b = k • (a - μ • b)) :
  λ * μ = -1 := 
sorry

end parallel_vectors_condition_l216_216714


namespace find_k_value_l216_216862

theorem find_k_value (k : ℝ) : (∃ k, ∀ x y, y = k * x + 3 ∧ (x, y) = (1, 2)) → k = -1 :=
by
  sorry

end find_k_value_l216_216862


namespace find_vector_q_l216_216035

-- Define the operation ⊗
def tensor (a b : ℕ × ℕ) : ℕ × ℕ := (a.1 * b.1, a.2 * b.2)

theorem find_vector_q : 
  let p : ℕ × ℕ := (1, 2)
  let q : ℕ × ℕ := (-3, -2)
  tensor p q = (-3, -4) 
=
  q = (-3, -2) := 
by {
  intros,
  sorry
}

end find_vector_q_l216_216035


namespace find_H2SO4_moles_l216_216656

-- Let KOH, H2SO4, and KHSO4 represent the moles of each substance in the reaction.
variable (KOH H2SO4 KHSO4 : ℕ)

-- Conditions provided in the problem
def KOH_moles : ℕ := 2
def KHSO4_moles (H2SO4 : ℕ) : ℕ := H2SO4

-- Main statement, we need to prove that given the conditions,
-- 2 moles of KOH and 2 moles of KHSO4 imply 2 moles of H2SO4.
theorem find_H2SO4_moles (KOH_sufficient : KOH = KOH_moles) 
  (KHSO4_produced : KHSO4 = KOH) : KHSO4_moles H2SO4 = 2 := 
sorry

end find_H2SO4_moles_l216_216656


namespace order_of_abc_l216_216667

noncomputable def a := 2^(4/3 : ℝ)
noncomputable def b := 3^(2/3 : ℝ)
noncomputable def c := 25^(1/3 : ℝ)

theorem order_of_abc : b < a ∧ a < c :=
by 
  sorry

end order_of_abc_l216_216667


namespace sum_of_all_six_angles_l216_216747

-- Define the triangles with their respective vertices
variables (A B C D E F : Type)

-- Define the interior angles at vertices A, B, and C
variables (angle1 angle2 angle3 : ℝ)
-- Define the exterior angles at vertices D, E, and F
variables (angle4 angle5 angle6 : ℝ)

-- Conditions that need to be satisfied (sum of interior and exterior angles)
axiom sum_interior_angles_ABC : angle1 + angle2 + angle3 = 180
axiom sum_exterior_angles_DEF : angle4 + angle5 + angle6 = 360

-- The main theorem to prove
theorem sum_of_all_six_angles
: angle1 + angle2 + angle3 + angle4 + angle5 + angle6 = 540 :=
by
  rw [sum_interior_angles_ABC, sum_exterior_angles_DEF]
  exact add_comm 180 360

end sum_of_all_six_angles_l216_216747


namespace estimated_population_is_correct_correlation_coefficient_is_correct_stratified_sampling_is_better_l216_216198

-- The sample data and conditions provided.
variables (x y : Fin 20 → ℝ)
variable (x̄ : ℝ := (∑ i, x i) / 20)
variable (ȳ : ℝ := (∑ i, y i) / 20)
variables (Sxx Syy Sxy : ℝ)
variables (n : ℕ := 20)
variables (total_plots : ℕ := 200)
variable (sum_x : ℝ := ∑ i, x i)
variable (sum_y : ℝ := ∑ i, y i)
variable (total_x : ℝ := 60)
variable (total_y : ℝ := 1200)

-- Given conditions
axiom condition1 : sum_x = 60
axiom condition2 : sum_y = 1200
axiom condition3 : Sxx = ∑ i, (x i - x̄) ^ 2
axiom condition4 : Syy = ∑ i, (y i - ȳ) ^ 2
axiom condition5 : Sxy = ∑ i, (x i - x̄) * (y i - ȳ)

-- Proven result 1: Estimated population of wild animals
def estimated_population (total_plots : ℕ) (avg_y : ℝ) : ℝ :=
  avg_y * total_plots

theorem estimated_population_is_correct : estimated_population total_plots (total_y / n) = 12000 :=
by
  -- Necessary calculations skipped
  sorry

-- Proven result 2: Correlation coefficient
noncomputable def correlation_coefficient (Sxy Sxx Syy : ℝ) : ℝ :=
  Sxy / (Real.sqrt (Sxx * Syy))

theorem correlation_coefficient_is_correct : correlation_coefficient Sxy Sxx Syy = 0.94 :=
by
  -- Assume given data and calculations are correct within required precision
  sorry

-- Proven result 3: Stratified sampling is more reasonable
theorem stratified_sampling_is_better : ∀ (method : String), method = "stratified sampling" → True :=
by
  -- Explanation skipped
  intros
  sorry

end estimated_population_is_correct_correlation_coefficient_is_correct_stratified_sampling_is_better_l216_216198


namespace selling_price_for_target_profit_l216_216244

-- Defining the conditions
def purchase_price : ℝ := 200
def annual_cost : ℝ := 40000
def annual_sales_volume (x : ℝ) := 800 - x
def annual_profit (x : ℝ) : ℝ := (x - purchase_price) * annual_sales_volume x - annual_cost

-- The theorem to prove
theorem selling_price_for_target_profit : ∃ x : ℝ, annual_profit x = 40000 ∧ x = 400 :=
by
  sorry

end selling_price_for_target_profit_l216_216244


namespace total_population_l216_216528

variable (seattle lake_view boise : ℕ)

def lake_view_pop := (lake_view = 24000)
def seattle_pop := (seattle = lake_view - 4000)
def boise_pop := (boise = (3 / 5 : ℚ) * seattle)

theorem total_population (h1 : lake_view_pop) (h2 : seattle_pop) (h3 : boise_pop) :
  lake_view + seattle + boise = 56000 :=
by
  sorry

end total_population_l216_216528


namespace delivery_meals_l216_216614

theorem delivery_meals (M P : ℕ) 
  (h1 : P = 8 * M) 
  (h2 : M + P = 27) : 
  M = 3 := by
  sorry

end delivery_meals_l216_216614


namespace identify_first_brother_l216_216245

-- Definitions for conditions
inductive Brother
| Trulya : Brother
| Falsa : Brother

-- Extracting conditions into Lean 4 statements
def first_brother_says : String := "Both cards are of the purplish suit."
def second_brother_says : String := "This is not true!"

axiom trulya_always_truthful : ∀ (b : Brother) (statement : String), b = Brother.Trulya ↔ (statement = first_brother_says ∨ statement = second_brother_says)
axiom falsa_always_lies : ∀ (b : Brother) (statement : String), b = Brother.Falsa ↔ ¬(statement = first_brother_says ∨ statement = second_brother_says)

-- Proof statement 
theorem identify_first_brother :
  ∃ (b : Brother), b = Brother.Trulya :=
sorry

end identify_first_brother_l216_216245


namespace field_length_l216_216504

theorem field_length (w l : ℝ) (A_f A_p : ℝ) 
  (h1 : l = 3 * w)
  (h2 : A_p = 150) 
  (h3 : A_p = 0.4 * A_f)
  (h4 : A_f = l * w) : 
  l = 15 * Real.sqrt 5 :=
by
  sorry

end field_length_l216_216504


namespace regression_line_significance_l216_216909

theorem regression_line_significance
  (samples: Type)
  (variables : samples → ℝ × ℝ)
  (calculated_regression_line : ∃ f : ℝ → ℝ, ∀ x y, (variables ⟨x, y⟩).2 = f (variables ⟨x, y⟩).1) :
  ∃ f : ℝ → ℝ, ∀ (x y : samples), (variables x).2 = f ((variables x).1) := by
  sorry

end regression_line_significance_l216_216909


namespace root_value_cond_l216_216396

theorem root_value_cond (p q : ℝ) (h₁ : ∃ x : ℝ, x^2 + p * x + q = 0 ∧ x = q) (h₂ : q ≠ 0) : p + q = -1 := 
sorry

end root_value_cond_l216_216396


namespace miniature_model_height_l216_216496

/-- Given the height and spherical volume of a city's water tower and the volume of a miniature model,
    prove that the height of the miniature model is approximately 0.949 meters. -/
theorem miniature_model_height :
  let height_real_tower := 60 -- height of the actual water tower in meters
  let volume_real_sphere := 50000 -- volume of the spherical top of the actual water tower in liters
  let volume_miniature_sphere := 0.2 -- volume of the spherical top of the miniature model in liters
  let volume_ratio := volume_real_sphere / volume_miniature_sphere -- ratio of volumes
  let scaling_factor := (volume_ratio)^(1/3) -- linear scaling factor
  let height_miniature_tower := height_real_tower / scaling_factor -- height of the miniature tower
  abs (height_miniature_tower - 0.949) < 0.001 := -- Allowing a small margin for approximation
sorry

end miniature_model_height_l216_216496


namespace tan_10pi_minus_theta_l216_216351

variables (θ : ℝ)

-- Given conditions
axiom h1 : π < θ
axiom h2 : θ < 2 * π
axiom h3 : cos (θ - 9 * π) = -3 / 5

-- Proof statement: The value of tan(10π - θ) is 4/3
theorem tan_10pi_minus_theta : tan (10 * π - θ) = 4 / 3 :=
by
  sorry

end tan_10pi_minus_theta_l216_216351


namespace false_statement_among_given_options_l216_216566

theorem false_statement_among_given_options :
  (∀ l₁ l₂ : ℝ → ℝ, parallel l₁ l₂ → supplementary (interior_angles l₁ l₂)) ∧
  (∀ l₁ l₂ : ℝ → ℝ, parallel l₁ l₂ → corresponding_angles_equal l₁ l₂) ∧
  (∀ l₁ l₂ : ℝ → ℝ, parallel l₁ l₂ → corresponding_angles_equal l₁ l₂) →
  ¬ (∀ α β : ℝ, acute α ∧ acute β → acute (α + β)) :=
by
  sorry

end false_statement_among_given_options_l216_216566


namespace mode_of_pm_values_l216_216517

def pm_values : list ℕ := [24, 23, 24, 25, 22]

theorem mode_of_pm_values : list.mode pm_values = 24 := 
by
  sorry

end mode_of_pm_values_l216_216517


namespace granger_total_amount_l216_216379

-- Define the constants for the problem
def cost_spam := 3
def cost_peanut_butter := 5
def cost_bread := 2
def quantity_spam := 12
def quantity_peanut_butter := 3
def quantity_bread := 4

-- Define the total cost calculation
def total_cost := (quantity_spam * cost_spam) + (quantity_peanut_butter * cost_peanut_butter) + (quantity_bread * cost_bread)

-- The theorem we need to prove
theorem granger_total_amount : total_cost = 59 := by
  sorry

end granger_total_amount_l216_216379


namespace encircling_stripe_probability_l216_216644

-- Define the conditions
def condition_top_and_bottom_vertical : Prop :=
  ∀ (face : ℕ), (face = 1 ∨ face = 2) → stripe_orientation(face) = "vertical"

def condition_side_faces_random_orientation : Prop :=
  ∀ (face : ℕ), (3 ≤ face ∧ face ≤ 6) → (stripe_orientation(face) = "vertical" ∨ stripe_orientation(face) = "horizontal")

def condition_independent_choices : Prop :=
  ∀ (face1 face2 : ℕ), face1 ≠ face2 → (stripe_orientation(face1), stripe_orientation(face2)) are_independent

-- Your question as a Lean theorem
theorem encircling_stripe_probability :
  condition_top_and_bottom_vertical ∧
  condition_side_faces_random_orientation ∧
  condition_independent_choices →
  probability_of_encircling_stripe ("horizontal sides only") = 1 / 16 :=
sorry

-- Helper definitions
def stripe_orientation (face : ℕ) : string := sorry  -- This should return the orientation ("vertical" or "horizontal") of the given face.

noncomputable def probability_of_encircling_stripe (condition : string) : ℚ :=
sorry

-- Define the independence of choices for the stripe
def are_independent (orientation1 orientation2 : string) : Prop := sorry

end encircling_stripe_probability_l216_216644


namespace g_sub_g_highest_power_7_l216_216322

def g (n : ℕ) : ℕ :=
  if h : n ≥ 3 then
    ∏ i in Finset.filter (λ x, odd x) (Finset.range (n + 1)), i
  else
    1 -- definition falls back to 1 for n < 3; actually never used in our proofs

theorem g_sub_g 
  (n : ℕ)
  (hn : n ≥ 3)
  (odd_n : ∃ m, n = 2 * m + 1)
  : g 99 - g 97 = g 97 * 98 := 
by
  sorry

theorem highest_power_7 
  (n : ℕ)
  (hn : n ≥ 3)
  (odd_n : ∃ m, n = 2 * m + 1)
  : ∃ k, 7^10 = k ∧ primes 7 (g 99 - g 97) = 10 := 
by
  sorry

end g_sub_g_highest_power_7_l216_216322


namespace mean_days_reading_l216_216043

theorem mean_days_reading : 
  let students := [(2, 1), (4, 2), (5, 3), (4, 5), (1, 6), (3, 7)] in
  let total_days := ∑ (p : ℕ × ℕ) in students, p.1 * p.2 in
  let total_students := ∑ (p : ℕ × ℕ) in students, p.1 in
  let mean := total_days.toRat / total_students.toRat in
  Float.ofRat mean ≈ 3.79 :=
by
  -- Expressions are summarized here for bounding, will be handled in formal proof.
  -- Type declaration needed for sum calculation and float comparison
  sorry

end mean_days_reading_l216_216043


namespace smallest_n_nonneg_reals_constraint_l216_216805

theorem smallest_n_nonneg_reals_constraint 
  (n : ℕ) 
  (x : ℕ → ℝ) 
  (h0 : ∀ i, 0 ≤ x i)
  (h1 : ∑ i in finset.range n, x i = 1) 
  (h2 : ∑ i in finset.range n, (x i)^2 ≤ 1 / 50) : 
  50 ≤ n := 
sorry

end smallest_n_nonneg_reals_constraint_l216_216805


namespace false_propositions_l216_216236

-- Define the original statements first
def prop1_original (x y : ℝ) : Prop := xy = 0 → (x = 0 ∧ y = 0)
def prop1_contrapositive (x y : ℝ) : Prop := (x ≠ 0 ∨ y ≠ 0) → xy ≠ 0

def prop2_original : Prop := ∀ (s: Square), is_rhombus s  -- Placeholder for defining shapes
def prop2_negation : Prop := ∀ (s: Square), ¬ is_rhombus s

def prop3_original (a b c : ℝ) : Prop := ac^2 > bc^2 → a > b
def prop3_inverse (a b c : ℝ) : Prop := a > b → ac^2 > bc^2

-- Example for quadratic inequality, could use mathlib definitions for polynomials and discriminants
def prop4_statement (m : ℝ) : Prop := m > 1 → ∀ x : ℝ, x^2 - 2 * x + m > 0

-- Define the proof that propositions are false
theorem false_propositions :
  (∀ x y : ℝ, ¬ prop1_contrapositive x y) ∧
  (∀ s: Square, ¬ prop2_negation s) ∧
  ( ∃ a b c : ℝ, ¬ prop3_inverse a b c) :=
by sorry

end false_propositions_l216_216236


namespace area_of_interior_triangle_l216_216359

theorem area_of_interior_triangle (a b c : ℕ) (ha : a = 49) (hb : b = 64) (hc : c = 225) :
  let s1 := Int.sqrt a; let s2 := Int.sqrt b in 1/2 * s1 * s2 = 28 := by
  sorry

end area_of_interior_triangle_l216_216359


namespace fifty_third_card_is_A_l216_216416

noncomputable def card_seq : List String := 
  ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

theorem fifty_third_card_is_A : card_seq[(53 % 13)] = "A" := 
by 
  simp [card_seq] 
  sorry

end fifty_third_card_is_A_l216_216416


namespace ratio_Mary_to_Seth_in_a_year_l216_216467

-- Given conditions
def Seth_current_age : ℝ := 3.5
def age_difference : ℝ := 9

-- Definitions derived from conditions
def Mary_current_age : ℝ := Seth_current_age + age_difference
def Seth_age_in_a_year : ℝ := Seth_current_age + 1
def Mary_age_in_a_year : ℝ := Mary_current_age + 1

-- The statement to prove
theorem ratio_Mary_to_Seth_in_a_year : (Mary_age_in_a_year / Seth_age_in_a_year) = 3 := sorry

end ratio_Mary_to_Seth_in_a_year_l216_216467


namespace joke_sequences_count_l216_216448

def is_joke_sequence (n : ℕ) (a b c : ℕ) : Prop :=
  a ∈ Finset.range (2 * n + 1) ∧ b ∈ Finset.range (2 * n + 1) ∧ c ∈ Finset.range (2 * n + 1) ∧
  (a = min a (min b c) ∨ b = min a (min b c) ∨ c = min a (min b c)) ∧
  ((a = b ∧ b ≠ c) ∨ (b = c ∧ c ≠ a) ∨ (a = c ∧ c ≠ b) ∨ (a ≠ b ∧ b ≠ c ∧ a ≠ c)) ∧
  (min a (min b c) % 2 = 1)

theorem joke_sequences_count (n : ℕ) (hn : 0 < n) :
  Finset.card {s : Finset (ℕ × ℕ × ℕ) | ∃ (a b c : ℕ), (a, b, c) ∈ s ∧ is_joke_sequence n a b c} = 4 * n^3 :=
by
  sorry

end joke_sequences_count_l216_216448


namespace find_angle_B_l216_216679

-- Given definitions and conditions
variables {a b c : ℝ}
variables {A B C : ℝ}
variable (h1 : (a + b + c) * (a - b + c) = a * c )

-- Statement of the proof problem
theorem find_angle_B (h1 : (a + b + c) * (a - b + c) = a * c) :
  B = 2 * π / 3 :=
sorry

end find_angle_B_l216_216679


namespace algebraic_expression_l216_216665

variable (m n x y : ℤ)

theorem algebraic_expression (h1 : x = m) (h2 : y = n) (h3 : x - y = 2) : n - m = -2 := 
by
  sorry

end algebraic_expression_l216_216665


namespace malcolm_total_followers_l216_216039

variable (Instagram Facebook Twitter TikTok YouTube : ℕ)

def followers_conditions (Instagram Facebook Twitter TikTok YouTube : ℕ) : Prop :=
  Instagram = 240 ∧
  Facebook = 500 ∧
  Twitter = (Instagram + Facebook) / 2 ∧
  TikTok = 3 * Twitter ∧
  YouTube = TikTok + 510

theorem malcolm_total_followers : ∃ (tot_followers : ℕ), 
  followers_conditions Instagram Facebook Twitter TikTok YouTube →
  tot_followers = Instagram + Facebook + Twitter + TikTok + YouTube ∧
  tot_followers = 3840 :=
by
  intros h
  sorry

end malcolm_total_followers_l216_216039


namespace problem_I_problem_II_l216_216375

section
variable (U : Set ℝ) (A : Set ℝ) (B : ℝ → Set ℝ)

-- Definition of the universe and sets A and B
def U := { x : ℝ | true }
def A := { x : ℝ | 1 ≤ x ∧ x ≤ 3 }
def B (a : ℝ) := { x : ℝ | 2 * a < x ∧ x < a + 3 }

-- Complement of set A
def CU (A : Set ℝ) := U \ A

-- Problem (I)
theorem problem_I : (CU A) ∩ B 1 = { x : ℝ | 3 < x ∧ x < 4 } := by
  sorry

-- Problem (II)
theorem problem_II (a : ℝ) : (CU A) ∩ B a = B a → a ≤ -2 ∨ a ≥ 3/2 := by
  sorry

end

end problem_I_problem_II_l216_216375


namespace weight_loss_percentage_l216_216598

theorem weight_loss_percentage 
  (weight_before weight_after : ℝ) 
  (h_before : weight_before = 800) 
  (h_after : weight_after = 640) : 
  (weight_before - weight_after) / weight_before * 100 = 20 := 
by
  sorry

end weight_loss_percentage_l216_216598


namespace triangle_similarity_quadrilateral_parallelogram_l216_216137

-- Problem (a): Similarity of triangles
theorem triangle_similarity
  (ABC A1B1C1 : Type) [triangle ABC] [triangle A1B1C1]
  (O : Point)
  (A B C A1 B1 C1 A2 B2 C2 : Point)
  (h1 : rotate_point O A = A1)
  (h2 : rotate_point O B = B1)
  (h3 : rotate_point O C = C1)
  (h4 : exists C2, intersect_line (line_through A B) (line_through A1 B1) = some C2)
  (h5 : exists A2, intersect_line (line_through B C) (line_through B1 C1) = some A2)
  (h6 : exists B2, intersect_line (line_through C A) (line_through C1 A1) = some B2) :
  similar_triangle (triangle A B C) (triangle A2 B2 C2) :=
  sorry

-- Problem (b): Parallelogram from intersections
theorem quadrilateral_parallelogram
  (ABCD A1B1C1D1 : Type) [quadrilateral ABCD] [quadrilateral A1B1C1D1]
  (O : Point)
  (A B C D A1 B1 C1 D1 P Q R S : Point)
  (h1 : rotate_point O A = A1)
  (h2 : rotate_point O B = B1)
  (h3 : rotate_point O C = C1)
  (h4 : rotate_point O D = D1)
  (h5 : exists P, intersect_line (line_through A B) (line_through A1 B1) = some P)
  (h6 : exists Q, intersect_line (line_through B C) (line_through B1 C1) = some Q)
  (h7 : exists R, intersect_line (line_through C D) (line_through C1 D1) = some R)
  (h8 : exists S, intersect_line (line_through D A) (line_through D1 A1) = some S) :
  parallelogram (quadrilateral P Q R S) :=
  sorry

end triangle_similarity_quadrilateral_parallelogram_l216_216137


namespace estimated_population_is_correct_correlation_coefficient_is_correct_stratified_sampling_is_better_l216_216200

-- The sample data and conditions provided.
variables (x y : Fin 20 → ℝ)
variable (x̄ : ℝ := (∑ i, x i) / 20)
variable (ȳ : ℝ := (∑ i, y i) / 20)
variables (Sxx Syy Sxy : ℝ)
variables (n : ℕ := 20)
variables (total_plots : ℕ := 200)
variable (sum_x : ℝ := ∑ i, x i)
variable (sum_y : ℝ := ∑ i, y i)
variable (total_x : ℝ := 60)
variable (total_y : ℝ := 1200)

-- Given conditions
axiom condition1 : sum_x = 60
axiom condition2 : sum_y = 1200
axiom condition3 : Sxx = ∑ i, (x i - x̄) ^ 2
axiom condition4 : Syy = ∑ i, (y i - ȳ) ^ 2
axiom condition5 : Sxy = ∑ i, (x i - x̄) * (y i - ȳ)

-- Proven result 1: Estimated population of wild animals
def estimated_population (total_plots : ℕ) (avg_y : ℝ) : ℝ :=
  avg_y * total_plots

theorem estimated_population_is_correct : estimated_population total_plots (total_y / n) = 12000 :=
by
  -- Necessary calculations skipped
  sorry

-- Proven result 2: Correlation coefficient
noncomputable def correlation_coefficient (Sxy Sxx Syy : ℝ) : ℝ :=
  Sxy / (Real.sqrt (Sxx * Syy))

theorem correlation_coefficient_is_correct : correlation_coefficient Sxy Sxx Syy = 0.94 :=
by
  -- Assume given data and calculations are correct within required precision
  sorry

-- Proven result 3: Stratified sampling is more reasonable
theorem stratified_sampling_is_better : ∀ (method : String), method = "stratified sampling" → True :=
by
  -- Explanation skipped
  intros
  sorry

end estimated_population_is_correct_correlation_coefficient_is_correct_stratified_sampling_is_better_l216_216200


namespace rank_best_buy_LMS_l216_216966

theorem rank_best_buy_LMS (c_S q_S : ℝ) :
  let c_M := 1.75 * c_S
  let q_M := 1.1 * q_S
  let c_L := 1.25 * c_M
  let q_L := 1.5 * q_M
  (c_S / q_S) > (c_M / q_M) ∧ (c_M / q_M) > (c_L / q_L) :=
by
  sorry

end rank_best_buy_LMS_l216_216966


namespace tan_ratio_of_triangle_sides_l216_216012

theorem tan_ratio_of_triangle_sides (a b c : ℝ) (α β γ : ℝ) 
  (h1 : a^2 + b^2 = 2023 * c^2)
  (h2 : α + β + γ = π)
  (h3 : c ≠ 0):
  ( (Real.tan γ) / (Real.tan α + Real.tan β) ) = (a * b) / (1011 * c^2) := 
sorry

end tan_ratio_of_triangle_sides_l216_216012


namespace concurrency_of_cevian_lines_l216_216864

theorem concurrency_of_cevian_lines
    (A B C A₁ B₁ C₁ : Point)
    (h_tangency_BC : incircle_or_excircle_of_triangle_touches BC A₁)
    (h_tangency_CA : incircle_or_excircle_of_triangle_touches CA B₁)
    (h_tangency_AB : incircle_or_excircle_of_triangle_touches AB C₁) :
    concurrent A A₁ B B₁ C C₁ :=
by
  sorry

end concurrency_of_cevian_lines_l216_216864


namespace third_vertices_form_regular_hexagon_iff_affine_regular_l216_216833

-- Define the affine regular condition
def affine_regular (hex : Hexagon) : Prop :=
  hex.is_centrally_symmetric ∧ hex.opposite_sides_parallel_to_diagonal

-- Define the problem statement in Lean 4
theorem third_vertices_form_regular_hexagon_iff_affine_regular 
  (hex : Hexagon) 
  (convex_hex: hex.is_convex)
  (tris : Π i, is_equilateral_triangle_on_side_outward hex.sides i) :
  (third_vertices_form_regular_hexagon hex tris) ↔ (affine_regular hex) :=
sorry

end third_vertices_form_regular_hexagon_iff_affine_regular_l216_216833


namespace min_cost_and_ways_l216_216324

-- Define the cost of each package
def cost_A : ℕ := 10
def cost_B : ℕ := 5

-- Define a function to calculate the total cost given the number of each package
def total_cost (nA nB : ℕ) : ℕ := nA * cost_A + nB * cost_B

-- Define the number of friends
def num_friends : ℕ := 4

-- Prove the minimum cost is 15 yuan and there are 28 ways
theorem min_cost_and_ways :
  (∃ nA nB : ℕ, total_cost nA nB = 15 ∧ (
    (nA = 1 ∧ nB = 1 ∧ 12 = 12) ∨ 
    (nA = 0 ∧ nB = 3 ∧ 12 = 12) ∨
    (nA = 0 ∧ nB = 3 ∧ 4 = 4) → 28 = 28)) :=
sorry

end min_cost_and_ways_l216_216324


namespace polynomial_remainder_inequality_l216_216450

theorem polynomial_remainder_inequality (p : ℕ) (f : polynomial ℤ) (d k : ℕ) 
  (hp : prime p) (hd : polynomial.degree f = d) 
  (hk : 1 < k ∧ k < p) 
  (hf : ∃ (r : finset ℤ), r.card = k ∧ ∀ i : ℕ, 1 ≤ i ∧ i ≤ p → f.eval i % p ∈ r) :
  (p - 1) / d ≤ k - 1 ∧ k - 1 ≤ (p - 1) * (1 - 1 / d) :=
sorry

end polynomial_remainder_inequality_l216_216450


namespace sports_competition_medals_l216_216600

theorem sports_competition_medals (n m : ℕ) (hn : n > 1)
  (h_m_seq : ∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → ℕ, (∃ (m_ini : ℕ), m_ini = m) 
  ∧ ∀ (k : ℕ), 1 ≤ k ∧ k < n → h_m_seq (k+1) = (6 * (h_m_seq k - k)) / 7 ∧ h_m_seq n = n) :
  n = 6 ∧ m = 36 :=
by
  -- Mathematical proof goes here
  sorry

end sports_competition_medals_l216_216600


namespace perpendicular_planes_if_line_perpendicular_and_parallel_l216_216671

variable {Point Line Plane : Type}
variable (l : Line) (alpha beta : Plane)
variable (is_parallel : Line → Plane → Prop) (is_perpendicular : Line → Plane → Prop) (planes_perpendicular : Plane → Plane → Prop)
variable (l_not_in_alpha : ¬ (l ∈ alpha))
variable (l_not_in_beta : ¬ (l ∈ beta))

theorem perpendicular_planes_if_line_perpendicular_and_parallel :
  is_perpendicular l alpha →
  is_parallel l beta →
  planes_perpendicular alpha beta :=
sorry

end perpendicular_planes_if_line_perpendicular_and_parallel_l216_216671


namespace quadratic_completion_l216_216514

theorem quadratic_completion (b c : ℝ) (h : (x : ℝ) → x^2 + 1600 * x + 1607 = (x + b)^2 + c) (hb : b = 800) (hc : c = -638393) : 
  c / b = -797.99125 := by
  sorry

end quadratic_completion_l216_216514


namespace problem1_problem2_l216_216363

-- Given function definition
noncomputable def f (x : ℝ) (a : ℝ) := Real.log x + a / x

-- Condition: tangent line at x = 1 is parallel to 2x - y = 0
def tangent_parallel (a : ℝ) := deriv (λ x, Real.log x + a / x) 1 = 2

-- To be proved: a = -1
theorem problem1 : ∀ a : ℝ, tangent_parallel a → a = -1 :=
by
  intros a h
  sorry

-- Helper function for the second problem
noncomputable def h (x : ℝ) (m : ℝ) (a : ℝ) := x + 1 / x - m * (Real.log x + a / x)

-- Minimum value consideration for the function h within the interval [1, e]
def h_min_condition (x0 : ℝ) (m : ℝ) := 1 ≤ x0 ∧ x0 ≤ Real.exp 1 ∧ x0 + 1 / x0 < m * (Real.log x0 + (-1) / x0)

-- To be proved: acceptable range for m is m > (Real.exp 2 + 1) / (Real.exp 1 - 1) or m < -2
theorem problem2 : ∀ m : ℝ, (∃ x0 : ℝ, h_min_condition x0 m) → m > (Real.exp 2 + 1) / (Real.exp 1 - 1) ∨ m < -2 :=
by
  intros m h
  sorry

end problem1_problem2_l216_216363


namespace power_of_power_example_l216_216272

theorem power_of_power_example : (3^2)^4 = 6561 := by
  sorry

end power_of_power_example_l216_216272


namespace range_g_l216_216659

noncomputable def g (x : ℝ) : ℝ := x / (x^2 - 2 * x + 2)

theorem range_g : Set.Icc (-(1:ℝ)/2) (1/2) = {y : ℝ | ∃ x : ℝ, g x = y} := 
by
  sorry

end range_g_l216_216659


namespace farmer_shipped_pomelos_in_dozens_l216_216004

theorem farmer_shipped_pomelos_in_dozens :
  let pomelos_per_box := 240 / 10 in
  let dozens_per_box := pomelos_per_box / 12 in
  let total_boxes := 10 + 20 in
  total_boxes * dozens_per_box = 60 :=
by
  have pomelos_per_box_eq : pomelos_per_box = 24 := by norm_num
  have dozens_per_box_eq : dozens_per_box = 2 := by norm_num
  have total_boxes_eq : total_boxes = 30 := by norm_num
  rw [pomelos_per_box_eq, dozens_per_box_eq, total_boxes_eq]
  norm_num
  sorry

end farmer_shipped_pomelos_in_dozens_l216_216004


namespace error_in_substitution_l216_216563

theorem error_in_substitution (x : ℂ) (h_sin : sin x = (exp (complex.I * x) - exp (-complex.I * x)) / (2 * complex.I))
  (h_cos : cos x = (exp (complex.I * x) + exp (-complex.I * x)) / 2)
  (h_subs : exp (complex.I * x) = exp (2 * π * complex.I * (x / (2 * π)))
    ∧ exp (-complex.I * x) = exp (-2 * π * complex.I * (x / (2 * π)))) :
  false := 
begin
  sorry
end

end error_in_substitution_l216_216563


namespace range_of_m_l216_216744

theorem range_of_m (m : ℝ) : -2 < m ∧ m < -1 ↔ ∃ x, ∃ y, 2(x - m - 1)^2 + 2m + 4 = y ∧ x < 0 ∧ y > 0 :=
by
  sorry

end range_of_m_l216_216744


namespace problem_l216_216806

def quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ f = λ x, a * x^2 + b * x + c

theorem problem
  (f : ℝ → ℝ)
  (h_quad : quadratic f)
  (h_roots : ∃ x₀ : ℝ, f(x₀) = 0 ∧ ∀ x : ℝ, f(x) = 0 → x = x₀)
  (h_derivative : ∀ x, (deriv f) x = 2 * x + 2) :
  f = λ x, x^2 + 2 * x + 1 :=
by
  sorry

end problem_l216_216806


namespace rectangle_ratio_l216_216866

theorem rectangle_ratio (A L : ℝ) (hA : A = 100) (hL : L = 20) :
  ∃ W : ℝ, A = L * W ∧ (L / W) = 4 :=
by
  sorry

end rectangle_ratio_l216_216866


namespace toothbrush_count_l216_216268

theorem toothbrush_count (T A : ℕ) (h1 : 53 + 67 + 46 = 166)
  (h2 : 67 - 36 = 31) (h3 : A = 31) (h4 : T = 166 + 2 * A) :
  T = 228 :=
  by 
  -- Using Lean's sorry keyword to skip the proof
  sorry

end toothbrush_count_l216_216268


namespace cone_base_circumference_l216_216177

noncomputable def radius (V : ℝ) (h : ℝ) : ℝ :=
  real.sqrt (3 * V / (π * h))

theorem cone_base_circumference
  (V : ℝ) (h : ℝ) (C : ℝ) (π_neq_zero : π ≠ 0)
  (V_eq : V = 24 * π) (h_eq : h = 6)
  (C_eq : C = 4 * real.sqrt 3 * π) :
  2 * π * radius V h = C :=
by {
  -- Simplify the expressions using the provided conditions
  rw [V_eq, h_eq, C_eq, radius, real.sqrt_mul, real.sqrt_div, mul_comm, ← mul_assoc],
  -- Answer follows directly from given conditions and basic algebra
  sorry
}

end cone_base_circumference_l216_216177


namespace total_letters_received_l216_216384

theorem total_letters_received :
  ∀ (g b m t : ℕ), 
    b = 40 →
    g = b + 10 →
    m = 2 * (g + b) →
    t = g + b + m → 
    t = 270 :=
by
  intros g b m t hb hg hm ht
  rw [hb, hg, hm, ht]
  sorry

end total_letters_received_l216_216384


namespace circle_center_and_radius_l216_216860

theorem circle_center_and_radius (x y : ℝ) (h : x^2 + y^2 - 6*x = 0) :
  (∃ c : ℝ × ℝ, c = (3, 0)) ∧ (∃ r : ℝ, r = 3) := 
sorry

end circle_center_and_radius_l216_216860


namespace median_free_throws_l216_216156

def free_throws : List ℕ := [7, 20, 16, 13, 21, 10, 23, 18, 15, 19]

def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (· ≤ ·)
  let n := sorted.length
  if n % 2 = 0 then (sorted.get! (n / 2 - 1) + sorted.get! (n / 2)) / 2
  else sorted.get! (n / 2)

theorem median_free_throws : median free_throws = 17 :=
by
  sorry

end median_free_throws_l216_216156


namespace prob_of_negative_card_l216_216100

/-
Statement: Given four cards with numbers -2, 0, 1/4, and 3, prove that the probability of drawing a card with a negative number is 1/4.
-/

theorem prob_of_negative_card : 
  let cards := [-2, 0, (1/4 : ℚ), 3] in
  (↑(cards.count (λ x => x < 0)) / ↑cards.length) = (1 / 4 : ℚ) :=
by
  let cards := [-2, 0, (1/4 : ℚ), 3]
  sorry

end prob_of_negative_card_l216_216100


namespace arithmetic_sequence_a12_bound_l216_216768

theorem arithmetic_sequence_a12_bound (a_1 d : ℤ) (h8 : a_1 + 7 * d ≥ 15) (h9 : a_1 + 8 * d ≤ 13) : 
  a_1 + 11 * d ≤ 7 :=
by
  sorry

end arithmetic_sequence_a12_bound_l216_216768


namespace cos_alpha_minus_beta_cos_beta_l216_216377

noncomputable def vec_a (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
noncomputable def vec_b (β : ℝ) : ℝ × ℝ := (Real.cos β, Real.sin β)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem cos_alpha_minus_beta (α β : ℝ) 
  (h : dot_product (vec_a α) (vec_b β) = Real.sqrt 2 / 2) 
  : Real.cos (α - β) = Real.sqrt 2 / 2 := 
  sorry

noncomputable def vec_c : ℝ × ℝ := (1, 2)

theorem cos_beta (α β : ℝ)
  (h_parallel : vec_a α = (√5 / 5) • vec_c)
  (h_angle : 0 < β ∧ β < α ∧ α < Real.pi / 2) 
  : Real.cos β = 3 * Real.sqrt 10 / 10 := 
  sorry

end cos_alpha_minus_beta_cos_beta_l216_216377


namespace probability_of_earning_2200_l216_216832

def spinner_sections : List (Option ℕ) := 
  [none, some 1200, some 500, some 7000, some 800, some 200]

def count_successful_ways (l : List (Option ℕ)) : Nat :=
  let possible_values := [some 1200, some 500, some 500]
  if l.permutations.contains possible_values then 1 else 0

def calculate_successful_cases : Nat := 
  spinner_sections.permutations.sum_by count_successful_ways

theorem probability_of_earning_2200 : 
  (calculate_successful_cases / (spinner_sections.length ^ 3) : ℚ) = 1 / 36 := 
by
  sorry

end probability_of_earning_2200_l216_216832


namespace polynomial_value_at_neg2_l216_216020

noncomputable def P : ℝ → ℝ
-- Define the polynomial P assuming the conditions

theorem polynomial_value_at_neg2 :
  (∀ P : ℝ → ℝ,
  ∃ (a b c d e : ℝ),      -- General form for polynomial degree 4
  P(x) = a*x^4 + b*x^3 + c*x^2 + d*x + e ∧
    P(0) = 1 ∧ P(1) = 1 ∧ P(2) = 4 ∧ P(3) = 9 ∧ P(4) = 16) →
  P(-2) = 19 :=
by 
  sorry

end polynomial_value_at_neg2_l216_216020


namespace function_is_one_l216_216924

noncomputable def f : ℝ → ℝ := sorry

theorem function_is_one (f : ℝ → ℝ)
  (h : ∀ x y z : ℝ, f (x*y) + f (x*z) ≥ 1 + f (x) * f (y*z))
  : ∀ x : ℝ, f x = 1 :=
sorry

end function_is_one_l216_216924


namespace soccer_team_unplayed_players_l216_216184

theorem soccer_team_unplayed_players:
  ∀ (total_players first_half_players first_half_subs : ℕ),
  first_half_players = 11 →
  first_half_subs = 2 →
  total_players = 24 →
  (total_players - (first_half_players + first_half_subs + 2 * first_half_subs)) = 7 :=
by
  intros total_players first_half_players first_half_subs h1 h2 h3
  rw [h1, h2, h3]
  show 24 - (11 + 2 + 2 * 2) = 7
  sorry

end soccer_team_unplayed_players_l216_216184


namespace estimated_population_correct_correlation_coefficient_correct_l216_216224

-- Definitions based on conditions
def total_plots := 200
def sample_plots := 20
def sum_x := 60
def sum_y := 1200
def sum_x_squared_diff := 80
def sum_y_squared_diff := 9000
def sum_xy_diff := 800

-- Estimated population calculation
noncomputable def estimated_population : ℕ :=
  (sum_y / sample_plots) * total_plots

theorem estimated_population_correct : estimated_population = 12000 := sorry

-- Correlation coefficient calculation
noncomputable def correlation_coefficient : Real :=
  sum_xy_diff / Real.sqrt (sum_x_squared_diff * sum_y_squared_diff)

theorem correlation_coefficient_correct : 
  Real.round (correlation_coefficient * 100) / 100 = 0.94 := sorry

end estimated_population_correct_correlation_coefficient_correct_l216_216224


namespace find_number_l216_216044

theorem find_number (x : ℤ) (h : ((x * 2) - 37 + 25) / 8 = 5) : x = 26 :=
sorry  -- Proof placeholder

end find_number_l216_216044


namespace jerry_mowing_time_l216_216442

theorem jerry_mowing_time :
  ∀ (total_acres : ℝ) (fraction_riding_mower : ℝ) (riding_mower_rate : ℝ) (push_mower_rate : ℝ),
    total_acres = 8 →
    fraction_riding_mower = 3/4 →
    riding_mower_rate = 2 →
    push_mower_rate = 1 →
    let acres_riding = total_acres * fraction_riding_mower in
    let acres_push = total_acres - acres_riding in
    let time_riding = acres_riding / riding_mower_rate in
    let time_push = acres_push / push_mower_rate in
    (time_riding + time_push) = 5 := 
by
  intros total_acres fraction_riding_mower riding_mower_rate push_mower_rate h1 h2 h3 h4
  let acres_riding := total_acres * fraction_riding_mower
  let acres_push := total_acres - acres_riding
  let time_riding := acres_riding / riding_mower_rate
  let time_push := acres_push / push_mower_rate
  have h_acres_riding : acres_riding = 6 := by norm_num [acres_riding, h1, h2]
  have h_acres_push : acres_push = 2 := by norm_num [acres_push, h_acres_riding, h1]
  have h_time_riding : time_riding = 3 := by norm_num [time_riding, h_acres_riding, h3]
  have h_time_push : time_push = 2 := by norm_num [time_push, h_acres_push, h4]
  norm_num [h_time_riding, h_time_push]
  sorry

end jerry_mowing_time_l216_216442


namespace total_metal_wasted_l216_216185

theorem total_metal_wasted :
  let s := 4 in
  let area_square := s^2 in
  let height_triangle := s in
  let side_triangle := 2 * height_triangle / Real.sqrt 3 in
  let area_triangle := Real.sqrt 3 / 4 * side_triangle^2 in
  let radius_circle := height_triangle / 2 in
  let area_circle := Real.pi * radius_circle^2 in
  area_square - area_triangle + area_triangle - area_circle = 16 - 4 * Real.pi :=
by
  sorry

end total_metal_wasted_l216_216185


namespace set_b_is_empty_l216_216974

open Set

theorem set_b_is_empty : 
    { x : ℝ | x ^ 2 + 1 = 0 } = ∅ :=
by
  sorry

end set_b_is_empty_l216_216974


namespace pens_sold_l216_216188

theorem pens_sold (C : ℝ) (N : ℝ) (h_gain : 22 * C = 0.25 * N * C) : N = 88 :=
by {
  sorry
}

end pens_sold_l216_216188


namespace pyramid_volume_l216_216254

-- Definitions of constants
def length_base : ℝ := 4 * Real.sqrt 2
def width_base : ℝ := 2 * Real.sqrt 2
def slant_height : ℝ := 6 * Real.sqrt 2

-- Define the calculation of the volume of the pyramid
theorem pyramid_volume :
  let A_base := length_base * width_base,
      height_pyr := Real.sqrt ((slant_height ^ 2) - ((width_base / 2) ^ 2)),
      volume := (1 / 3) * A_base * height_pyr
  in volume = 128 / 3 :=
by
  let A_base := length_base * width_base,
      height_pyr := Real.sqrt ((slant_height ^ 2) - ((width_base / 2) ^ 2)),
      volume := (1 / 3) * A_base * height_pyr
  show volume = 128 / 3
  sorry

end pyramid_volume_l216_216254


namespace distribute_positions_l216_216609

theorem distribute_positions :
  ∃ (n : ℕ), n = 6 ∧ 
  (∃ f : Fin 3 → ℕ, 
      (∀ i, 1 ≤ f i) ∧ 
      (∀ i, f i < 6) ∧ 
      (∀ i j, i ≠ j → f i ≠ f j) ∧ 
      ((Finset.univ.sum f = 6) ∧ 
       ∃ perm : Finset.univ.permutations, perm.prod f)) :=
sorry

end distribute_positions_l216_216609


namespace max_angle_visibility_l216_216138

noncomputable def point_X (p q : Line) (M N : Point) : Point :=
  sorry

theorem max_angle_visibility (p q : Line) (M N : Point) :
  intersects p q →
  on_segment M N q →
  let X := point_X p q M N in
  ∠MXN = max_angle p q M N :=
  sorry

end max_angle_visibility_l216_216138


namespace num_digits_c_l216_216870

theorem num_digits_c (a b c : ℕ) (ha : 10 ^ 2010 ≤ a ∧ a < 10 ^ 2011)
  (hb : 10 ^ 2011 ≤ b ∧ b < 10 ^ 2012)
  (h1 : a < b) (h2 : b < c)
  (div1 : ∃ k : ℕ, b + a = k * (b - a))
  (div2 : ∃ m : ℕ, c + b = m * (c - b)) :
  10 ^ 4 ≤ c ∧ c < 10 ^ 5 :=
sorry

end num_digits_c_l216_216870


namespace number_of_dogs_l216_216411

theorem number_of_dogs (D C B x : ℕ) (h1 : D = 3 * x) (h2 : B = 9 * x) (h3 : D + B = 204) (h4 : 12 * x = 204) : D = 51 :=
by
  -- Proof skipped
  sorry

end number_of_dogs_l216_216411


namespace find_cube_surface_area_in_sphere_l216_216779

/-
Inside a sphere with a radius of 2, there is a cube whose vertices all lie on the sphere's surface.
Find the surface area of this cube.
-/
theorem find_cube_surface_area_in_sphere :
  ∃ (a : ℝ), 
    let diameter := 4 in
    (√3 * a = diameter) ∧ 6 * a^2 = 32 :=
by
  use 4 / √3
  split
  {
    -- Proof for √3 * a = 4
    sorry
  }
  {
    -- Proof for 6 * a^2 = 32
    sorry
  }

end find_cube_surface_area_in_sphere_l216_216779


namespace total_apples_bought_l216_216391

def apples_bought_by_each_man := 30
def apples_more_than_men := 20
def apples_bought_by_each_woman := apples_bought_by_each_man + apples_more_than_men

theorem total_apples_bought (men women : ℕ) (apples_bought_by_each_man : ℕ) 
  (apples_bought_by_each_woman : ℕ) (apples_more_than_men : ℕ) : 
  men = 2 → women = 3 → apples_bought_by_each_man = 30 → apples_more_than_men = 20 → 
  apples_bought_by_each_woman = apples_bought_by_each_man + apples_more_than_men →
  (men * apples_bought_by_each_man + women * apples_bought_by_each_woman) = 210 := 
by
  intros menTwo menThree womApples menApples moreApples _ _ _ _ eqn
  simp only [menTwo, menThree, womApples, menApples, moreApples, eqn]
  -- state infering
  exact eq.refl 210

end total_apples_bought_l216_216391


namespace triangle_DEC_area_l216_216256

theorem triangle_DEC_area :
  ∀ (A B C D E : Type)
  [IsoscelesTriangle A B C]
  (h1 : AB = 3)
  (h2 : AC = 3)
  (h3 : BC = 4)
  (D_midpoint : MidpointOfSegment D A B)
  (E_on_AC : PointOnSegment E A C)
  (h4 : AE = 2),
  Area_of_triangle_DEC = 1.2 :=
by
  sorry

end triangle_DEC_area_l216_216256


namespace concert_friend_count_l216_216930

theorem concert_friend_count (P : ℝ) (original_friends : ℕ := 3) (discount : ℝ := 0.25) : 
  ∃ (x : ℕ), let total_friends := original_friends + x in
  original_friends * P = total_friends * (P * (1 - discount)) ∧ total_friends = 4 :=
by {
  use 1,
  have h1 : P * 3 = P * (3 + 1) * (0.75) := sorry,
  exact ⟨h1, rfl⟩
}

end concert_friend_count_l216_216930


namespace find_b7_l216_216076

theorem find_b7 : ∀ {b : ℕ → ℕ}, 
  (∀ n, b (n + 1) > b n) ∧ 
  (∀ n ≥ 1, b (n + 2) = b (n + 1) + b n) ∧ 
  b 6 = 96 → 
  b 7 = 184 :=
begin
  intros b h1 h2 h3,
  sorry
end

end find_b7_l216_216076


namespace min_y_value_l216_216124

noncomputable def y (x : ℝ) : ℝ := x^2 + 16 * x + 20

theorem min_y_value : ∀ (x : ℝ), x ≥ -3 → y x ≥ -19 :=
by
  intro x hx
  sorry

end min_y_value_l216_216124


namespace integer_solutions_inequality_system_l216_216087

theorem integer_solutions_inequality_system :
  {x : ℤ | (x + 2 > 0) ∧ (2 * x - 1 ≤ 0)} = {-1, 0} := 
by
  -- proof goes here
  sorry

end integer_solutions_inequality_system_l216_216087


namespace value_of_x_plus_y_l216_216733

noncomputable def x : ℝ := 1 / 2
noncomputable def y : ℝ := 3

theorem value_of_x_plus_y
  (hx : 1 / x = 2)
  (hy : 1 / x + 3 / y = 3) :
  x + y = 7 / 2 :=
  sorry

end value_of_x_plus_y_l216_216733


namespace max_value_2cosB_plus_sin2C_l216_216407

variable {A B C : ℝ} (h : A = π / 4)
variable (h_sum : A + B + C = π)

theorem max_value_2cosB_plus_sin2C :
  ∃ B C : ℝ, 2 * Real.cos B + Real.sin (2 * C) ≤ 3 / 2 :=
by
  have h1 : B + C = π - A := by linarith [h_sum, h]
  sorry

  ⟨ sorry ⟩ -- This would typically involve constructing the values of B and C

end max_value_2cosB_plus_sin2C_l216_216407


namespace nested_log_value_positive_l216_216265

noncomputable def nested_log (x : ℕ) : ℝ :=
  if h : x = 0 then 16 else 16 + real.logb 2 (nested_log (x - 1))

lemma nested_log_eq (x : ℕ) : nested_log x = 16 + real.logb 2 (nested_log x) :=
sorry

theorem nested_log_value_positive : 
  ∃ x : ℝ, x = real.logb 2 (16 + real.logb 2 (16 + real.logb 2 (16 + ...))) ∧ x = 4 := 
sorry

end nested_log_value_positive_l216_216265


namespace negation_of_all_cars_are_fast_l216_216509

variable {α : Type} -- Assume α is the type of entities
variable (car fast : α → Prop) -- car and fast are predicates on entities

theorem negation_of_all_cars_are_fast :
  ¬ (∀ x, car x → fast x) ↔ ∃ x, car x ∧ ¬ fast x :=
by sorry

end negation_of_all_cars_are_fast_l216_216509


namespace mary_received_more_l216_216041

theorem mary_received_more (investment_Mary investment_Harry profit : ℤ)
  (one_third_profit divided_equally remaining_profit : ℤ)
  (total_Mary total_Harry difference : ℤ)
  (investment_ratio_Mary investment_ratio_Harry : ℚ) :
  investment_Mary = 700 →
  investment_Harry = 300 →
  profit = 3000 →
  one_third_profit = profit / 3 →
  divided_equally = one_third_profit / 2 →
  remaining_profit = profit - one_third_profit →
  investment_ratio_Mary = 7/10 →
  investment_ratio_Harry = 3/10 →
  total_Mary = divided_equally + investment_ratio_Mary * remaining_profit →
  total_Harry = divided_equally + investment_ratio_Harry * remaining_profit →
  difference = total_Mary - total_Harry →
  difference = 800 := by
  sorry

end mary_received_more_l216_216041


namespace find_2011th_fraction_l216_216678

theorem find_2011th_fraction :
  let fractions := (λ (n : ℕ), (n, (filter (λ a, Nat.gcd a n = 1) (List.range n.succ)).map (λ a, (a, n))))
  let all_fractions := fractions 3 ++ fractions 6 ++ fractions 9 ++ fractions 12 ++ fractions 15 ++ /* ... continue adding up to sufficiently large denominators ... */
  all_fractions[2010] = (49, 111) := by
    sorry

end find_2011th_fraction_l216_216678


namespace cone_side_surface_area_l216_216331

def side_surface_area_of_cone (r_base : ℝ) (alpha : ℝ) (slant_height : ℝ) : ℝ :=
  1 / 2 * slant_height * (slant_height / alpha)

theorem cone_side_surface_area :
  let r_base := 1
  let alpha := (2 * Real.pi) / 3
  let slant_height := 2 * Real.pi
  side_surface_area_of_cone r_base alpha slant_height = 3 * Real.pi :=
by
  unfold side_surface_area_of_cone
  sorry

end cone_side_surface_area_l216_216331


namespace purchasing_options_count_l216_216187

noncomputable def num_purchasing_options : ℕ :=
Finset.card (Finset.filter
  (λ (p : ℕ × ℕ), 
     let (x, y) := p in
     (6 * x + 7 * y ≤ 50) ∧ (x ≥ 3) ∧ (y ≥ 2))
  (Finset.product (Finset.Icc 3 7) (Finset.Icc 2 7)))

theorem purchasing_options_count : num_purchasing_options = 7 := by
  sorry

end purchasing_options_count_l216_216187


namespace cube_surface_area_l216_216104

-- Positional conditions of the vertices of the cube
def A : ℝ × ℝ × ℝ := (3, 7, 4)
def B : ℝ × ℝ × ℝ := (4, 3, -5)
def C : ℝ × ℝ × ℝ := (7, -2, 3)

-- The statement asserting that the surface area of the cube is 294
theorem cube_surface_area (A B C : ℝ × ℝ × ℝ) (h₁ : A = (3, 7, 4)) (h₂ : B = (4, 3, -5)) (h₃ : C = (7, -2, 3)) : 
  let AB_dist := real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2 + (B.3 - A.3)^2),
      AC_dist := real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2 + (C.3 - A.3)^2),
      BC_dist := real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2 + (C.3 - B.3)^2) in
  AB_dist = real.sqrt 98 ∧ AC_dist = real.sqrt 98 ∧ BC_dist = real.sqrt 98 →
  ∃ a : ℝ, AB_dist = a * real.sqrt 2 ∧ AC_dist = a * real.sqrt 2 ∧ BC_dist = a * real.sqrt 2 ∧ 6 * (a ^ 2) = 294 := 
begin
  sorry
end

end cube_surface_area_l216_216104


namespace polynomial_value_at_neg2_l216_216019

noncomputable def P : ℝ → ℝ
-- Define the polynomial P assuming the conditions

theorem polynomial_value_at_neg2 :
  (∀ P : ℝ → ℝ,
  ∃ (a b c d e : ℝ),      -- General form for polynomial degree 4
  P(x) = a*x^4 + b*x^3 + c*x^2 + d*x + e ∧
    P(0) = 1 ∧ P(1) = 1 ∧ P(2) = 4 ∧ P(3) = 9 ∧ P(4) = 16) →
  P(-2) = 19 :=
by 
  sorry

end polynomial_value_at_neg2_l216_216019


namespace graph_of_equation_l216_216122

theorem graph_of_equation {x y : ℝ} (h : (x - 2 * y)^2 = x^2 - 4 * y^2) :
  (y = 0) ∨ (x = 2 * y) :=
by
  sorry

end graph_of_equation_l216_216122


namespace bus_trip_distance_l216_216158

noncomputable def distance_of_bus_trip (D : ℕ) :=
  (D / 35 - D / 40) = 1

theorem bus_trip_distance : ∃ D : ℕ, distance_of_bus_trip D ∧ D = 280 :=
by
  use 280
  unfold distance_of_bus_trip
  norm_num
  simp
  sorry

end bus_trip_distance_l216_216158


namespace max_value_of_expr_l216_216801

noncomputable def max_value_expr (c d : ℝ) (h1 : 0 < c) (h2 : 0 < d) : ℝ :=
  3 * (c - (λ y : ℝ, y + real.sqrt (y^2 + d^2)) c d h1 h2) * 
  ((λ y : ℝ, y + real.sqrt (y^2 + d^2)) c d h1 h2) c 

theorem max_value_of_expr (c d : ℝ) (h1 : 0 < c) (h2 : 0 < d) :
  max_value_expr c d h1 h2 = (3 / 2) * (c^2 + d^2) :=
sorry

end max_value_of_expr_l216_216801


namespace flour_to_add_l216_216823

-- Define the variables and conditions
def total_flour : ℕ := 10
def flour_added : ℕ := 6

-- The proof statement: Mary needs to add more flour such that the total flour is satisfied by the recipe
theorem flour_to_add (total_flour : ℕ) (flour_added : ℕ) : total_flour - flour_added = 4 :=
by
  rw [total_flour, flour_added]
  norm_num
  sorry

end flour_to_add_l216_216823


namespace probability_of_region_D_l216_216166

theorem probability_of_region_D
    (P_A : ℚ) (P_B : ℚ) (P_C : ℚ) (P_D : ℚ)
    (h1 : P_A = 1/4) 
    (h2 : P_B = 1/3) 
    (h3 : P_C = 1/6) 
    (h4 : P_A + P_B + P_C + P_D = 1) : 
    P_D = 1/4 := by
    sorry

end probability_of_region_D_l216_216166


namespace maxwell_walking_speed_is_4_l216_216824

-- Given definitions based on the problem's conditions
def brad_speed : ℝ := 6  -- Brad's running speed in km/h
def distance_between_houses : ℝ := 50  -- Distance between Maxwell's and Brad's houses in km
def maxwell_travel_distance : ℝ := 20  -- Distance Maxwell travels to meet Brad in km
def brad_travel_distance : ℝ := distance_between_houses - maxwell_travel_distance  -- Distance Brad travels to meet Maxwell
def travel_time : ℝ := brad_travel_distance / brad_speed  -- Time they traveled to meet

-- Definition we aim to prove
def maxwell_speed : ℝ := maxwell_travel_distance / travel_time

theorem maxwell_walking_speed_is_4 : maxwell_speed = 4 := 
by 
  -- Here goes the proof which is skipped
  sorry

end maxwell_walking_speed_is_4_l216_216824


namespace min_value_of_f_l216_216293

def f (x : ℝ) : ℝ := 16^x - 4^x + 1

theorem min_value_of_f : ∃ x : ℝ, f x = 3/4 :=
by
  sorry

end min_value_of_f_l216_216293


namespace area_of_shaded_region_l216_216432

theorem area_of_shaded_region 
    (large_side : ℝ) (small_side : ℝ)
    (h_large : large_side = 10) 
    (h_small : small_side = 4) : 
    (large_side^2 - small_side^2) / 4 = 21 :=
by
  -- All proof steps are to be completed and checked,
  -- and sorry is used as placeholder for the final proof.
  sorry

end area_of_shaded_region_l216_216432


namespace trapezoid_cosine_identity_trapezoid_sine_identity_1_trapezoid_sine_identity_2_l216_216736

-- Define the context and conditions
variables (a b c d : ℝ)
variables (α β γ δ : ℝ)

-- Assuming the given conditions in the problem
axiom AB_eq_a : AB = a
axiom BC_eq_b : BC = b
axiom CD_eq_c : CD = c
axiom DA_eq_d : DA = d

-- Describe the proof goals
theorem trapezoid_cosine_identity :
  a = d * cos α + b * cos β - c * cos (α + δ) := sorry

theorem trapezoid_sine_identity_1 :
  a * sin α = c * sin δ + b * sin (α + β) := sorry

theorem trapezoid_sine_identity_2 :
  a * sin β = c * sin γ + d * sin (α + β) := sorry

end trapezoid_cosine_identity_trapezoid_sine_identity_1_trapezoid_sine_identity_2_l216_216736


namespace medal_award_ways_l216_216522

open Nat

theorem medal_award_ways :
  let sprinters := 10
  let italians := 4
  let medals := 3
  let gold_medal_ways := choose italians 1
  let remaining_sprinters := sprinters - 1
  let non_italians := remaining_sprinters - (italians - 1)
  let silver_medal_ways := choose non_italians 1
  let new_remaining_sprinters := remaining_sprinters - 1
  let new_non_italians := new_remaining_sprinters - (italians - 1)
  let bronze_medal_ways := choose new_non_italians 1
  gold_medal_ways * silver_medal_ways * bronze_medal_ways = 120 := by
    sorry

end medal_award_ways_l216_216522


namespace complex_multiplication_l216_216651

theorem complex_multiplication : 
  let i : ℂ := complex.I in
  (3 - 4 * i) * (-6 + 2 * i) = -10 + 30 * i :=
by {
  have i_squared : i^2 = -1 := by ring,
  -- expand and simplify using distributive property
  rw [mul_add, mul_add, mul_comm (3:ℂ), mul_comm (-4 * i), ←neg_mul_eq_mul_neg],
  ring_nf,
  rw [←i_squared],
  norm_num,
}

end complex_multiplication_l216_216651


namespace evaluate_three_squared_raised_four_l216_216281

theorem evaluate_three_squared_raised_four : (3^2)^4 = 6561 := by
  sorry

end evaluate_three_squared_raised_four_l216_216281


namespace jane_current_age_l216_216972

theorem jane_current_age (J : ℕ) (h1 : ∀ t : ℕ, t = 13 → 25 + t = 2 * (J + t)) : J = 6 :=
by {
  sorry
}

end jane_current_age_l216_216972


namespace triangle_interior_angles_not_greater_than_60_l216_216562

theorem triangle_interior_angles_not_greater_than_60 (α β γ : ℝ) (h_sum : α + β + γ = 180) 
  (h_pos : α > 0 ∧ β > 0 ∧ γ > 0) :
  α ≤ 60 ∨ β ≤ 60 ∨ γ ≤ 60 :=
by
  sorry

end triangle_interior_angles_not_greater_than_60_l216_216562


namespace grandfather_wen_total_payment_l216_216889

theorem grandfather_wen_total_payment : 
  let regular_ticket_price := 8
  let senior_ticket_discount := 0.25
  let child_ticket_discount := 0.50
  let senior_ticket_price := regular_ticket_price * (1 - senior_ticket_discount)
  let child_ticket_price := regular_ticket_price * (1 - child_ticket_discount)
  2 * senior_ticket_price + 2 * regular_ticket_price + 2 * child_ticket_price = 36 :=
by
  let regular_ticket_price := 8
  let senior_ticket_discount := 0.25
  let child_ticket_discount := 0.50
  let senior_ticket_price := regular_ticket_price * (1 - senior_ticket_discount)
  let child_ticket_price := regular_ticket_price * (1 - child_ticket_discount)
  have h1: senior_ticket_price = 6 := by sorry
  have h2: child_ticket_price = 4 := by sorry
  calc
    2 * senior_ticket_price + 2 * regular_ticket_price + 2 * child_ticket_price
      = 2 * 6 + 2 * 8 + 2 * 4 : by congr; assumption <|> sorry
      = 12 + 16 + 8           : by sorry
      = 36                    : by sorry

end grandfather_wen_total_payment_l216_216889


namespace constant_ratio_of_segments_l216_216542

variables {S A B C D : Type} [real_plane S A B C D]

def lines_intersect_at_S (a b c d : line S) (l : line S) (A B C D : point) : Prop :=
  (intersects a S) ∧ (intersects b S) ∧ (intersects c S) ∧ (intersects d S) ∧
  (A ∈ l ∧ intersects a A) ∧ (B ∈ l ∧ intersects b B) ∧ (C ∈ l ∧ intersects c C) ∧ (D ∈ l ∧ intersects d D)

theorem constant_ratio_of_segments 
  (S : point) (a b c d : line S) (l : line S) (A B C D : point)
  (h : lines_intersect_at_S a b c d l A B C D) :
  ∀ l, ∃ k : ℝ, (AC : ℝ) * (BD : ℝ) = k * (BC : ℝ) * (AD : ℝ) :=
sorry

end constant_ratio_of_segments_l216_216542


namespace P_neg2_eq_19_l216_216024

noncomputable def P : ℝ → ℝ := sorry  -- Definition of the polynomial P(x)

axiom degree_P : ∃ (d : ℕ), d = 4 ∧ ∀ (x : ℝ) (hx : P x ≠ 0), nat_degree (polynomial.of_real (P x)) = d
axiom P_0 : P 0 = 1
axiom P_1 : P 1 = 1
axiom P_2 : P 2 = 4
axiom P_3 : P 3 = 9
axiom P_4 : P 4 = 16

theorem P_neg2_eq_19 : P (-2) = 19 :=
by
  sorry

end P_neg2_eq_19_l216_216024


namespace min_value_of_f_l216_216292

def f (x : ℝ) : ℝ := 16^x - 4^x + 1

theorem min_value_of_f : ∃ x : ℝ, f x = 3/4 :=
by
  sorry

end min_value_of_f_l216_216292


namespace functional_equation_solution_l216_216638

def f : ℝ → ℝ := sorry

theorem functional_equation_solution :
  (∀ x y : ℝ, f (2 * x + f y) = x + y + f x) →
  (∀ x : ℝ, f x = x) :=
by
  intros h x
  sorry

end functional_equation_solution_l216_216638


namespace cos_beta_l216_216688

theorem cos_beta (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) 
  (h_cos_α : Real.cos α = 3/5) (h_cos_alpha_plus_beta : Real.cos (α + β) = -5/13) : 
  Real.cos β = 33/65 :=
by
  sorry

end cos_beta_l216_216688


namespace find_num_of_green_balls_l216_216159

def num_of_green_balls (total_balls : ℕ) (white_balls : ℕ) (yellow_balls : ℕ) (red_balls : ℕ) (purple_balls : ℕ) (prob_neither_red_nor_purple : ℝ) : ℕ :=
  total_balls - white_balls - yellow_balls - red_balls - purple_balls

theorem find_num_of_green_balls : 
  ∀ (total_balls : ℕ) (white_balls : ℕ) (yellow_balls : ℕ) (red_balls : ℕ) (purple_balls : ℕ) (prob_neither_red_nor_purple : ℝ), 
    total_balls = 100 → 
    white_balls = 50 → 
    yellow_balls = 8 → 
    red_balls = 9 → 
    purple_balls = 3 → 
    prob_neither_red_nor_purple = 0.88 →
    num_of_green_balls total_balls white_balls yellow_balls red_balls purple_balls prob_neither_red_nor_purple = 30 :=
by {
  intros total_balls white_balls yellow_balls red_balls purple_balls prob_neither_red_nor_purple,
  intros h_total h_white h_yellow h_red h_purple h_prob,
  rw [h_total, h_white, h_yellow, h_red, h_purple, h_prob],
  calc
  num_of_green_balls 100 50 8 9 3 0.88
    = 100 - 50 - 8 - 9 - 3 : by simp [num_of_green_balls]
    = 30 : by norm_num
}

end find_num_of_green_balls_l216_216159


namespace total_letters_l216_216385

theorem total_letters (brother_letters : ℕ) (greta_more_than_brother : ℕ) (mother_multiple : ℕ)
  (h_brother : brother_letters = 40)
  (h_greta : ∀ (brother_letters greta_letters : ℕ), greta_letters = brother_letters + greta_more_than_brother)
  (h_mother : ∀ (total_letters mother_letters : ℕ), mother_letters = mother_multiple * total_letters) :
  brother_letters + (brother_letters + greta_more_than_brother) + (mother_multiple * (brother_letters + (brother_letters + greta_more_than_brother))) = 270 :=
by
  sorry

end total_letters_l216_216385


namespace find_kilos_of_tomato_l216_216052

def initial_money := 500
def remaining_money := 426
def cost_per_kilo_potato := 2
def kilos_potato := 6
def cost_per_kilo_tomato := 3
def cost_per_kilo_cucumber := 4
def kilos_cucumber := 5
def cost_per_kilo_banana := 5
def kilos_banana := 3

theorem find_kilos_of_tomato 
  (initial_money remaining_money cost_per_kilo_potato kilos_potato cost_per_kilo_tomato : ℕ)
  (cost_per_kilo_cucumber kilos_cucumber cost_per_kilo_banana kilos_banana : ℕ) :
  ∃ (kilos_tomato : ℕ), initial_money - remaining_money - 
    (kilos_potato * cost_per_kilo_potato + kilos_cucumber * cost_per_kilo_cucumber + kilos_banana * cost_per_kilo_banana) = kilos_tomato * cost_per_kilo_tomato ∧
    kilos_tomato = 9 :=
begin
  sorry
end

end find_kilos_of_tomato_l216_216052


namespace sum_of_squares_l216_216929

theorem sum_of_squares :
  1000^2 + 1001^2 + 1002^2 + 1003^2 + 1004^2 = 5020030 :=
by
  sorry

end sum_of_squares_l216_216929


namespace bisect_area_of_quadrilateral_l216_216838

def bisects_diagonal (A B C D O : Point) (AC BD : Line) : Prop :=
(intersection AC BD = O) ∧ (distance A O = distance C O) ∧ (distance B O = distance D O)

noncomputable def area (A B C : Point) : ℝ := sorry

theorem bisect_area_of_quadrilateral 
  (A B C D O : Point)
  (AC BD : Line)
  (h1 : bisects_diagonal A C D B O AC BD) :
  (area A B D + area C D B) = (area A C D + area B C O) :=
sorry

end bisect_area_of_quadrilateral_l216_216838


namespace andy_profit_per_cake_l216_216985

-- Definitions based on the conditions
def cost_of_ingredients (cakes : ℕ) : ℕ := if cakes = 2 then 12 else 0
def cost_of_packaging_per_cake : ℕ := 1
def selling_price_per_cake : ℕ := 15

-- Theorem stating the profit made per cake
theorem andy_profit_per_cake : ∀ (cakes : ℕ), cakes = 2 → 
(cost_of_ingredients cakes / cakes + cost_of_packaging_per_cake) = 7 →
selling_price_per_cake - (cost_of_ingredients cakes / cakes + cost_of_packaging_per_cake) = 8 :=
by
  intros cakes h_cakes cost_hyp
  have h1 : cost_of_ingredients cakes / cakes = 12 / 2 :=
    by rw [h_cakes]; refl
  have h2 : (12 / 2 + cost_of_packaging_per_cake) = 6 + 1 :=
    by rw [h1]; refl
  have h3 : (6 + 1) = 7 :=
    by refl
  rw [← h3] at cost_hyp
  have h4 : selling_price_per_cake - 7 = 8 :=
    by refl
  exact h4

end andy_profit_per_cake_l216_216985


namespace cards_in_boxes_l216_216449

theorem cards_in_boxes (n : ℕ) (h : n ≥ 3) (f : Fin n^2 → Fin n) :
  ∃ (g : Fin n → Fin n → ℕ), 
  (∀ i : Fin n, let s := {x : ℕ | ∃ j : Fin n, f ⟨i * n + j, sorry⟩ = i ∧ g ⟨i * n + j, sorry⟩} in
                ∃ k : ℕ, s = finset.range(n) + k) :=
sorry

end cards_in_boxes_l216_216449


namespace perpendicular_bisectors_of_quadrilateral_intersection_l216_216819

theorem perpendicular_bisectors_of_quadrilateral_intersection 
  {A B C D : Point} 
  (a : Line) (b : Line) (c : Line) (d : Line) 
  (A' B' C' D' : Point) 
  (AB_perpendicular_bisector : ∀ P, P ∈ a ↔ dist P A = dist P B)
  (BC_perpendicular_bisector : ∀ P, P ∈ b ↔ dist P B = dist P C)
  (CD_perpendicular_bisector : ∀ P, P ∈ c ↔ dist P C = dist P D)
  (DA_perpendicular_bisector : ∀ P, P ∈ d ↔ dist P D = dist P A)
  (intersect_a_b : A' ∈ a ∧ A' ∈ b)
  (intersect_b_c : B' ∈ b ∧ B' ∈ c)
  (intersect_c_d : C' ∈ c ∧ C' ∈ d)
  (intersect_d_a : D' ∈ d ∧ D' ∈ a)
  (non_concurrent : ¬ (a = b ∧ b = c ∧ c = d ∧ d = a)) :
  perpendicular (Line.through A' C') (Line.through B' D') ∧
  bisector (Line.through A' C') (diagonal A C) ∧
  bisector (Line.through B' D') (diagonal B D) :=
sorry

end perpendicular_bisectors_of_quadrilateral_intersection_l216_216819


namespace incorrect_option_C_l216_216500

def f (x : ℝ) (ω : ℝ) := Real.sin (ω * x)
def g (x : ℝ) (ω : ℝ) := f (x + Real.pi / 2) ω
def k := (1 : ℤ) -- As exemplified from solution since k resulted in ω = 4k + 1

theorem incorrect_option_C (ω : ℝ) (hω : ω = 5) (h_g0 : g 0 ω = 1) :
  ∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ Real.pi / 2 ∧ 
  g a ω = 0 ∧ g b ω = 0 ∧ g c ω = 0 := by
sos

end incorrect_option_C_l216_216500


namespace eval_expression_at_x_is_correct_l216_216557

theorem eval_expression_at_x_is_correct : 
  let x := 3 in (x^6 - 6 * x^2) = 675 :=
by
  sorry

end eval_expression_at_x_is_correct_l216_216557


namespace min_M_value_l216_216366

noncomputable def f (a b x : ℝ) : ℝ := abs (sqrt x - a * x - b)

def M (a b : ℝ) := sup (set.image (λ x, f a b x) (set.Icc 0 4))

theorem min_M_value : ∀ (a b : ℝ), (∃ (m : ℝ), M a b = m ∧ m ≥ 1/4) :=
by
  intros a b
  use M a b
  split
  { reflexivity }
  { sorry }

end min_M_value_l216_216366


namespace least_total_bananas_l216_216541

-- Definitions of the initial bananas taken by the monkeys
def bananas (b1 b2 b3 : ℕ) : Prop :=
  -- Final ratios of bananas are in 4:3:2
  let A := (1 / 2) * b1 + (1 / 3) * b2 + (5 / 12) * b3 in
  let B := (1 / 4) * b1 + (2 / 3) * b2 + (5 / 12) * b3 in
  let C := (1 / 4) * b1 + (1 / 3) * b2 + (1 / 6) * b3 in
  4 * C = 3 * B ∧
  9 * B = 12 * C ∧
  4 * C / 3 = B ∧
  B / 3 = C / 2 ∧ 
  (4 * C : ℝ) = 3 * ( B : ℝ) ∧
  (12 * C : ℝ) = 9 * ( B : ℝ)

-- Statement to prove the problem
theorem least_total_bananas : ∃ (b1 b2 b3 : ℕ), 
  bananas b1 b2 b3 ∧ b1 + b2 + b3 = 276 :=
by
  sorry

end least_total_bananas_l216_216541


namespace prize_cost_l216_216493

theorem prize_cost (total_cost : ℕ) (quantity : ℕ) (total_cost_eq : total_cost = 120) (quantity_eq : quantity = 6) : total_cost / quantity = 20 :=
by 
  rw [total_cost_eq, quantity_eq]
  norm_num

end prize_cost_l216_216493


namespace three_city_population_l216_216530

noncomputable def totalPopulation (boise seattle lakeView: ℕ) : ℕ :=
  boise + seattle + lakeView

theorem three_city_population (pBoise pSeattle pLakeView : ℕ)
  (h1 : pBoise = 3 * pSeattle / 5)
  (h2 : pLakeView = pSeattle + 4000)
  (h3 : pLakeView = 24000) :
  totalPopulation pBoise pSeattle pLakeView = 56000 := by
  sorry

end three_city_population_l216_216530


namespace incorrect_analogical_reasonings_l216_216130

theorem incorrect_analogical_reasonings :
  (condition1 : true) →
  (condition2 : |∀ (a : ℝ × ℝ) (|a|^2 = a.1^2 + a.2^2)  ∧ ∀ (z : ℂ) (|z|^2 ≠ z^2)|) →
  (condition3 : ∃ (a b c ∈ ℂ), (∀ (x : ℂ), b^2 - 4 * a * c > 0 → ¬ ∃ r s ∈ ℂ, r ≠ s ∧ a * x^2 + b * x + c = 0)) →
  (condition4 : true) →
  -- incorrect reasoning: B and C
  (incorrect_BC : true) := 
  sorry

end incorrect_analogical_reasonings_l216_216130


namespace negative_number_option_l216_216237

def optionA := -(-5)
def optionB := +|-5|
def optionC := -((-5)^2)
def optionD := (-5) * (-5)

theorem negative_number_option : optionC < 0 :=
by
  sorry

end negative_number_option_l216_216237


namespace statement_A_statement_B_statement_C_statement_D_l216_216697

-- Definitions based on the problem conditions
def curve (m : ℝ) (x y : ℝ) : Prop :=
  x^4 + y^4 + m * x^2 * y^2 = 1

def is_symmetric_about_origin (m : ℝ) : Prop :=
  ∀ x y : ℝ, curve m x y ↔ curve m (-x) (-y)

def enclosed_area_eq_pi (m : ℝ) : Prop :=
  ∀ x y : ℝ, curve m x y → (x^2 + y^2)^2 = 1

def does_not_intersect_y_eq_x (m : ℝ) : Prop :=
  ∀ x y : ℝ, curve m x y ∧ x = y → false

def no_common_points_with_region (m : ℝ) : Prop :=
  ∀ x y : ℝ, |x| + |y| < 1 → ¬ curve m x y

-- Statements to prove based on correct answers
theorem statement_A (m : ℝ) : is_symmetric_about_origin m :=
  sorry

theorem statement_B (m : ℝ) (h : m = 2) : enclosed_area_eq_pi m :=
  sorry

theorem statement_C (m : ℝ) (h : m = -2) : ¬ does_not_intersect_y_eq_x m :=
  sorry

theorem statement_D (m : ℝ) (h : m = -1) : no_common_points_with_region m :=
  sorry

end statement_A_statement_B_statement_C_statement_D_l216_216697


namespace choose_groups_of_three_l216_216827

theorem choose_groups_of_three (n k : ℕ) (h_n : n = 6) (h_k : k = 3) :
  nat.choose n k = 20 :=
by {
  sorry
}

end choose_groups_of_three_l216_216827


namespace zero_of_f_in_interval_l216_216880

def f (x : ℝ) : ℝ := Real.log x / Real.log 2 + x - 2

theorem zero_of_f_in_interval (m : ℤ) : ∃ x : ℝ, (f x = 0 ∧ m ≤ x ∧ x ≤ m + 1) ↔ m = 1 := 
by sorry

end zero_of_f_in_interval_l216_216880


namespace sector_properties_l216_216358

noncomputable def central_angle (l r : ℝ) : ℝ := l / r

noncomputable def sector_area (alpha r : ℝ) : ℝ := (1/2) * alpha * r^2

theorem sector_properties (l r : ℝ) (h_l : l = Real.pi) (h_r : r = 3) :
  central_angle l r = Real.pi / 3 ∧ sector_area (central_angle l r) r = 3 * Real.pi / 2 := 
  by
  sorry

end sector_properties_l216_216358


namespace max_discardable_grapes_l216_216933

theorem max_discardable_grapes (n : ℕ) (k : ℕ) (h : k = 8) : 
  ∃ m : ℕ, m < k ∧ (∀ q : ℕ, q * k + m = n) ∧ m = 7 :=
by
  sorry

end max_discardable_grapes_l216_216933


namespace initial_distance_l216_216489

open Real

theorem initial_distance (H S : ℝ) (t : ℝ) (d_heather d_stacy : ℝ)
  (H_eq : H = 5)
  (S_eq : S = H + 1)
  (start_delay : 24 / 60 = 0.4)
  (heather_distance : H * t = 5.7272727272727275)
  (stacy_distance : S * (t + 0.4) = 9.272727272727273) :
  d_heather + d_stacy = 15 :=
by
  rw [H_eq, S_eq] at *
  have t_eq : t = 5.7272727272727275 / 5 := by
    field_simp [H_eq]
    exact_mod_cast (div_self (@ne_of_gt ℝ _ _ _ (lt_of_lt_of_le _ _ (zero_add _))))
  simp only [t_eq] at heather_distance
  simp_rw [heather_distance] at heather_distance
  simp_rw [start_delay] at heather_distance
  simp_rw [times_add 6 _ 0.4] at heather_distance
  simp_rw [S_eq]
  exact_mod_cast (heather_distance + stacy_distance)

end initial_distance_l216_216489


namespace min_value_16x_minus_4x_plus_1_l216_216301

theorem min_value_16x_minus_4x_plus_1 :
  ∀ x : ℝ, 16 ^ x - 4 ^ x + 1 ≥ 3 / 4 ∧ (∃ y : ℝ, 16 ^ y - 4 ^ y + 1 = 3 / 4) :=
by
  sorry

end min_value_16x_minus_4x_plus_1_l216_216301


namespace shortest_distance_P_A_l216_216764

noncomputable def distance_PA (a x : ℝ) : ℝ :=
  real.sqrt ((x - a) ^ 2 + ((1 / x) - a) ^ 2)

theorem shortest_distance_P_A (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ distance_PA a x = 2 * real.sqrt 2) ↔ (a = -1 ∨ a = real.sqrt 10) :=
by
  sorry

end shortest_distance_P_A_l216_216764


namespace divisibility_by_1956_l216_216476

theorem divisibility_by_1956 (n : ℕ) (hn : n % 2 = 1) : 
  1956 ∣ (24 * 80^n + 1992 * 83^(n-1)) :=
by
  sorry

end divisibility_by_1956_l216_216476


namespace find_alpha_l216_216354

noncomputable def g : ℝ → ℝ := sorry

axiom twice_diff (x : ℝ) (hx : 0 < x) : deriv (deriv g) x = g'' x

axiom differential_equation (x : ℝ) (hx : 0 < x) :
  g x + 2 * x^3 * deriv g x + x^4 * deriv (deriv g) x = 0

axiom limit_condition : tendsto (λ x, x * g x) at_top (𝓝 1)

theorem find_alpha (α : ℝ) (hα : α > 1) (hα_eq : g α = 1 / 2) : α = 6 / Real.pi :=
sorry

end find_alpha_l216_216354


namespace only_quadratic_radical_option_A_l216_216564

def is_quadratic_radical (x : ℝ) : Prop := 
  ∃ y : ℝ, y^2 = x ∧ x ≥ 0

theorem only_quadratic_radical_option_A :
  is_quadratic_radical 5 ∧ 
  ¬is_quadratic_radical (-2) ∧ 
  ¬is_quadratic_radical (-real.pi) ∧ 
  ¬is_quadratic_radical (real.cbrt 3) :=
by
  sorry

end only_quadratic_radical_option_A_l216_216564


namespace probability_of_PBC_more_than_one_third_area_l216_216613

noncomputable def AB : ℝ := 6
noncomputable def AC : ℝ := 6
noncomputable def BC : ℝ := 8

def height_of_triangle : ℝ := Real.sqrt (AB ^ 2 - (BC / 2) ^ 2)

def area_of_ABC : ℝ := 0.5 * BC * height_of_triangle

def area_condition (P : {x : ℝ // 0 ≤ x} × {y : ℝ // 0 ≤ y}) : Prop :=
  let area_PBC := 0.5 * BC * (height_of_triangle - P.2)
  area_PBC > area_of_ABC / 3

theorem probability_of_PBC_more_than_one_third_area :
  let region := { P : {x : ℝ // 0 ≤ x} × {y : ℝ // 0 ≤ y} // area_condition P }
  let total_region := { P : {x : ℝ // 0 ≤ x} × {y : ℝ // 0 ≤ y} // P.1 ≤ BC / 2 ∧ P.2 ≤ height_of_triangle }
  (region.card : ℝ) / (total_region.card : ℝ) = 1 / 3 :=
sorry

end probability_of_PBC_more_than_one_third_area_l216_216613


namespace evaluate_polynomial_l216_216559

theorem evaluate_polynomial (x : ℤ) (h : x = 3) : x^6 - 6 * x^2 = 675 := by
  sorry

end evaluate_polynomial_l216_216559


namespace students_neither_math_physics_drama_exclusive_l216_216748

def total_students : ℕ := 75
def math_students : ℕ := 42
def physics_students : ℕ := 35
def both_students : ℕ := 25
def drama_exclusive_students : ℕ := 10

theorem students_neither_math_physics_drama_exclusive : 
  total_students - (math_students + physics_students - both_students + drama_exclusive_students) = 13 :=
by
  sorry

end students_neither_math_physics_drama_exclusive_l216_216748


namespace product_of_possible_values_l216_216727

theorem product_of_possible_values :
  (∀ x : ℝ, abs (18 / x + 4) = 3 → x = -18 ∨ x = -18 / 7) →
  (∀ x1 x2 : ℝ, x1 = -18 → x2 = -18 / 7 → x1 * x2 = 324 / 7) :=
by
  intros h x1 x2 hx1 hx2
  rw [hx1, hx2]
  norm_num

end product_of_possible_values_l216_216727


namespace distance_between_michael_and_marcos_when_marcos_touches_ball_l216_216599

def ball_speed : ℝ := 4 -- m/s
def michael_speed : ℝ := 9 -- m/s
def marcos_speed : ℝ := 8 -- m/s
def initial_distance_ball_michael : ℝ := 15 -- m
def initial_distance_ball_marcos : ℝ := 30 -- m

theorem distance_between_michael_and_marcos_when_marcos_touches_ball:
  let marcos_time := initial_distance_ball_marcos / (marcos_speed + ball_speed) in
  let michael_distance_after_marcos_time := michael_speed * marcos_time - ball_speed * marcos_time in
  michael_distance_after_marcos_time = 2.5 :=
by
  sorry

end distance_between_michael_and_marcos_when_marcos_touches_ball_l216_216599


namespace linear_function_points_relation_l216_216925

theorem linear_function_points_relation (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (h1 : y1 = 5 * x1 - 3) 
  (h2 : y2 = 5 * x2 - 3) 
  (h3 : x1 < x2) : 
  y1 < y2 :=
sorry

end linear_function_points_relation_l216_216925


namespace solve_for_s_l216_216592

---
theorem solve_for_s (s : ℝ) (h₀: s > 0)
(h₁ : let θ := real.pi / 3 in area = s * 3 * (s * real.sin θ))
(h₂ : area = 27 * real.sqrt 3) : s = 3 * real.sqrt 2 := by
  sorry

end solve_for_s_l216_216592


namespace hyperbola_equation_l216_216710

def parabola :=
  ∃ (y x : ℝ), y^2 = 8 * x

def hyperbola (a b : ℝ) :=
  ∃ (x y : ℝ), (x^2) / (a^2) - (y^2) / (b^2) = 1

def asymptote_condition (a b : ℝ) :=
  b / a = 4 * real.sqrt 3 / 3

def equilateral_triangle (a b : ℝ) (A B F : ℝ × ℝ) :=
  let x_A := -2
      y_A := 4 * real.tan (real.pi / 6)
      y_B := -4 * real.tan (real.pi / 6) in
  A = (x_A, y_A) ∧ B = (x_A, y_B) ∧ (F = (2, 0) ∧ dist F A = dist F B ∧ dist A B = dist F A)

theorem hyperbola_equation
  (a b : ℝ)
  (A B F : ℝ × ℝ)
  (h_parabola : parabola)
  (h_hyperbola : hyperbola a b)
  (h_asymptote : asymptote_condition a b)
  (h_equilateral : equilateral_triangle a b A B F) :
  (a = real.sqrt 3 ∧ b = 4) →
  hyperbola 3 16 :=
sorry

end hyperbola_equation_l216_216710


namespace exists_n_with_k_prime_factors_l216_216007

theorem exists_n_with_k_prime_factors (k : ℕ) (m : ℕ) (hk : k ≠ 0) (hm : m % 2 = 1) :
  ∃ (n : ℕ), (nat.factors (m^n + n^m)).nodup.card ≥ k :=
sorry

end exists_n_with_k_prime_factors_l216_216007


namespace missile_selection_l216_216902

theorem missile_selection (missile_numbers : list ℕ)
  (range_condition : ∀ n ∈ missile_numbers, 1 ≤ n ∧ n ≤ 50)
  (equal_intervals_condition : ∀ i j, (i < j ∧ i < missile_numbers.length ∧ j < missile_numbers.length) → 
                                   missile_numbers.get ⟨j, sorry⟩ - missile_numbers.get ⟨i, sorry⟩ = 10) :
  missile_numbers = [3, 13, 23, 33, 43] :=
sorry

end missile_selection_l216_216902


namespace smallest_X_l216_216117

variable (a : ℕ → ℕ)

def condition_1 := a 0 + a 1 = 80
def condition_2 (n : ℕ) := a (n-3) + a (n-2) + a (n-1) + a n = 200 
def condition_3 (i : ℕ) (h : 2 ≤ i ∨ i < n) := 40 ≤ a i
def condition_4 (n : ℕ) := ∑ i in finset.range n, a i = X

theorem smallest_X (n : ℕ) (X : ℕ)
  (h1 : condition_1 a) 
  (h2 : condition_2 a n)
  (h3 : ∀ i, 2 ≤ i ∨ i < n → condition_3 a i)
  (h4 : condition_4 a n) :
  (∀ m, 2 ≤ m → ∑ i in finset.range m, a i ≠ 480) := 
sorry

end smallest_X_l216_216117


namespace real_roots_quadratic_iff_l216_216362

theorem real_roots_quadratic_iff (a : ℝ) : (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 1 = 0) ↔ a ≤ 2 := 
sorry

end real_roots_quadratic_iff_l216_216362


namespace rainfall_ratio_l216_216643

theorem rainfall_ratio (R1 R2 : ℕ) (H1 : R2 = 18) (H2 : R1 + R2 = 30) : R2 / R1 = 3 / 2 := by
  sorry

end rainfall_ratio_l216_216643


namespace sum_invested_l216_216602

theorem sum_invested (P R: ℝ) (h1: SI₁ = P * R * 20 / 100) (h2: SI₂ = P * (R + 10) * 20 / 100) (h3: SI₂ = SI₁ + 3000) : P = 1500 :=
by
  sorry

end sum_invested_l216_216602


namespace perpendicular_foot_to_xaxis_l216_216346

theorem perpendicular_foot_to_xaxis (A : ℝ × ℝ) (hA : A = (1, 2)) : ∃ M : ℝ × ℝ, M = (1, 0) ∧ ∀ y : ℝ, M = (fst A, y) → y = 0 :=
by
  sorry

end perpendicular_foot_to_xaxis_l216_216346


namespace average_score_is_8_9_l216_216962

-- Define the scores and their frequencies
def scores : List ℝ := [7.5, 8.5, 9, 10]
def frequencies : List ℕ := [2, 2, 3, 3]

-- Express the condition that the total number of shots is 10
def total_shots : ℕ := frequencies.sum

-- Calculate the weighted sum of the scores
def weighted_sum (scores : List ℝ) (frequencies : List ℕ) : ℝ :=
  (List.zip scores frequencies).foldl (λ acc (sc, freq) => acc + (sc * freq)) 0

-- Prove that the average score is 8.9
theorem average_score_is_8_9 :
  total_shots = 10 →
  weighted_sum scores frequencies / total_shots = 8.9 :=
by
  intros h_total_shots
  sorry

end average_score_is_8_9_l216_216962


namespace malcolm_total_followers_l216_216040

variable (Instagram Facebook Twitter TikTok YouTube : ℕ)

def followers_conditions (Instagram Facebook Twitter TikTok YouTube : ℕ) : Prop :=
  Instagram = 240 ∧
  Facebook = 500 ∧
  Twitter = (Instagram + Facebook) / 2 ∧
  TikTok = 3 * Twitter ∧
  YouTube = TikTok + 510

theorem malcolm_total_followers : ∃ (tot_followers : ℕ), 
  followers_conditions Instagram Facebook Twitter TikTok YouTube →
  tot_followers = Instagram + Facebook + Twitter + TikTok + YouTube ∧
  tot_followers = 3840 :=
by
  intros h
  sorry

end malcolm_total_followers_l216_216040


namespace tomatoes_planted_each_kind_l216_216051

-- Definitions derived from Conditions
def total_rows : ℕ := 10
def spaces_per_row : ℕ := 15
def kinds_of_tomatoes : ℕ := 3
def kinds_of_cucumbers : ℕ := 5
def cucumbers_per_kind : ℕ := 4
def potatoes : ℕ := 30
def available_spaces : ℕ := 85

-- Theorem statement with the question and answer derived from the problem
theorem tomatoes_planted_each_kind : (kinds_of_tomatoes * (total_rows * spaces_per_row - Available_spaces - (kinds_of_cucumbers * cucumbers_per_kind + potatoes)) / kinds_of_tomatoes) = 5 :=
by 
  sorry

end tomatoes_planted_each_kind_l216_216051


namespace b_2015_eq_4030_l216_216328

-- Conditions as definitions 
def f1 (x : ℝ) : ℝ := (x^2 + 2*x + 1) * real.exp x
def fn (n : ℕ) (x : ℝ) : ℝ := (a n * x^2 + b n * x + c n) * real.exp x
def fn_plus1 (n : ℕ) (x : ℝ) : ℝ := (fn n x)' 

-- General form conditions
axiom a_n_def (n : ℕ) : a n = 1
axiom b_n_def (n : ℕ) : b n = 2 * n
axiom c_n_def (n : ℕ) : c n = n^2 - n + 1

-- Prove that b_2015 = 4030
theorem b_2015_eq_4030 : b 2015 = 4030 :=
by sorry

end b_2015_eq_4030_l216_216328


namespace minimum_value_of_expression_l216_216306

theorem minimum_value_of_expression : ∃ x : ℝ, (16 ^ x - 4 ^ x + 1 = 3 / 4) :=
sorry

end minimum_value_of_expression_l216_216306


namespace meal_total_amount_l216_216582

theorem meal_total_amount (cost_of_food : ℝ) (sales_tax_rate tip_rate : ℝ) (total_allowed : ℝ) :
  cost_of_food = 40.98 →
  sales_tax_rate = 0.07 →
  tip_rate = 0.15 →
  total_allowed = 51.00 →
  total_allowed = cost_of_food + (sales_tax_rate * cost_of_food) + (tip_rate * cost_of_food) :=
by
  intros h_cf h_str h_tr h_ta
  rw [h_cf, h_str, h_tr]
  have h : 51.00 = 40.98 + (0.07 * 40.98) + (0.15 * 40.98) := sorry
  rw h
  exact h_ta

end meal_total_amount_l216_216582


namespace vehicle_safety_l216_216547

variable {V_A : ℝ} -- Define V_A as a real number

-- Definitions based on conditions
#check 50 -- distance between A and B in ft
#check 350 -- distance between A and C in ft
#check 45 -- velocity of B in mph
#check 70 -- velocity of C in mph

-- Define the necessary relationships and conditions
def passB (V_A : ℝ) : ℝ := 50 / (V_A - 45)
def collideC (V_A : ℝ) : ℝ := 350 / (V_A + 70)

-- The main statement to prove
theorem vehicle_safety : ∀ V_A, 50 / (V_A - 45) < 350 / (V_A + 70) → V_A > 64 :=
by {
  intros, 
  sorry
}

end vehicle_safety_l216_216547


namespace vasya_can_measure_one_liter_l216_216546

theorem vasya_can_measure_one_liter :
  ∃ (states : ℕ × ℕ × ℕ → ℕ × ℕ × ℕ → ℕ × ℕ × ℕ),
    let fill := λ s b n, if s = b then n else 0 in
    let pour := λ (s1 s2 : ℕ × ℕ × ℕ), s1.1 - min s1.1 (s2.2), s2.2 + min s1.1 (s2.2), s1.3 + s2.3 in
    let empty := λ s b, if s = b then 0 else 0 in
    ((states (0, 0, 0)) = (1, 0, 0) → (∃ s, s = (0, 0, 0))

        - initial state condition
        (∀ s1 s2, fill s1 s2.1 s2.2 = (s1.1, s2.2, s1.3) ∨
              fill s1 s2.2 s2.2 = (s1.1, s1.2, s2.2)) →
        (∀ s1 s2, pour s1 s2 = (s1.1, s1.2, s2.3) ∨
              pour s1 s2 = (s1.1, s2.2, s2.3)) →
        (∀ s1 s2, empty s1 s2 = (s1.1, s2) ∨
              empty s1 s2 = (s1.2, s2) ∨
              empty s1 s2 = (s1.3, s2))

        - conditions for each step
        (∃ m1 m2 t v,
          ⟨m1, m2, t⟩ = (0, 5, 7) ∧ ⟨m1, m2, t⟩ = v ∧ t = 1) :=
by sorry

end vasya_can_measure_one_liter_l216_216546


namespace heptagon_distance_relation_l216_216674

variable (A_1 A_2 A_3 A_4 A_7 : Type)
variable [Heptagon A_1 A_2 A_3 A_4 A_7]
variable (a b c : ℝ)
variable (dist_A1A2 : dist A_1 A_2 = a)
variable (dist_A1A3 : dist A_1 A_3 = b)
variable (dist_A1A4 : dist A_1 A_4 = c)

theorem heptagon_distance_relation :
  1 / a = 1 / b + 1 / c :=
sorry

end heptagon_distance_relation_l216_216674


namespace log_sum_exp_diff_l216_216317

theorem log_sum (lg : ℝ → ℝ) (h_lg2 : lg 2 = 1) (h_lg5 : lg 5 = 1) : lg 2 + lg 5 = 1 := 
sorry

theorem exp_diff (log2 : ℝ → ℝ) (h_log2_3 : log2 3 = 1) (h_8_third : 8^(1 / 3) = 2) : 2^(log2 3) - 8^(1 / 3) = 1 :=
sorry

end log_sum_exp_diff_l216_216317


namespace solve_sharon_distance_l216_216840

def sharon_total_distance (distance normal_time traffic_time new_speed_decrease : ℝ) : Prop :=
  ∀ (x : ℝ), 
  let normal_speed := x / normal_time in
  let new_speed := (x / normal_time) - new_speed_decrease in
  let time_first_part := (1/4) * x / normal_speed in
  let time_second_part := (3/4) * x / new_speed in
  time_first_part + time_second_part = traffic_time →
  x = distance

-- Given conditions
def conditions := sharon_total_distance 80 150 250 (15 / 60)

theorem solve_sharon_distance : conditions := by
  sorry

end solve_sharon_distance_l216_216840


namespace total_books_count_l216_216900

-- Defining the constant values based on the problem conditions
def numMathBooks : ℕ := 27
def costMathBook : ℕ := 4
def costHistBook : ℕ := 5
def totalCost : ℕ := 373

-- Defining the proposition we need to prove
theorem total_books_count (H : ℕ) : (numMathBooks * costMathBook + H * costHistBook = totalCost) → (numMathBooks + H = 80) :=
begin
    sorry
end

end total_books_count_l216_216900


namespace length_of_circular_track_l216_216887

theorem length_of_circular_track (t : ℕ) (vx vy vz : ℕ) (d : ℕ) (ht : t = 120)
  (hvx : vx = 12) (hvy : vy = 20) (hvz : vz = 36)
  (hx : d = Nat.gcd (vx * t) (vy * t))
  (hy : d = Nat.gcd d (vz * t)) :
  d = 480 := by
  have h1 : vx * t = 1440 := by rw [hvx, ht]; norm_num
  have h2 : vy * t = 2400 := by rw [hvy, ht]; norm_num
  have h3 : vz * t = 4320 := by rw [hvz, ht]; norm_num
  rw [h1, h2] at hx
  rw [← hx] at d
  rw [h3] at hy
  have h4 : d = Nat.gcd 1440 2400 := by rw [hx, h1, h2]
  have h5 : d = Nat.gcd 480 4320 := by rw [hy, h4, h3]
  exact Nat.gcd_eq_right (by norm_num) h5

end length_of_circular_track_l216_216887


namespace find_a_and_max_value_l216_216365

noncomputable def f (x a : ℝ) : ℝ := sin (2 * x) + a * cos x + x

theorem find_a_and_max_value :
  (∀ x a, (differentiable ℝ (λ x, f x a)) ∧ (deriv (λ x, f x a) (π / 6) = 0) → a = 4) ∧
  (∀ x : ℝ, -π / 6 ≤ x ∧ x ≤ 7 * π / 6 → 
    f x 4 ≤ f (5 * π / 6) 4 ∧ 
    f (5 * π / 6) 4 = (5 * real.sqrt 3) / 2 + π / 6) :=
by
  -- Proof is omitted
  sorry

end find_a_and_max_value_l216_216365


namespace estimated_population_correct_correlation_coefficient_correct_l216_216205

noncomputable def estimated_population
  (num_plots : ℕ) (sample_size : ℕ) 
  (sum_y : ℕ) : ℕ :=
(sum_y / sample_size) * num_plots

noncomputable def correlation_coefficient
  (sum_x_sq : ℝ) (sum_y_sq : ℝ) 
  (sum_xy : ℝ) : ℝ :=
sum_xy / real.sqrt(sum_x_sq * sum_y_sq)

theorem estimated_population_correct
  (num_plots : ℕ := 200) (sample_size : ℕ := 20) 
  (sum_y : ℕ := 1200) : estimated_population num_plots sample_size sum_y = 12000 := by
sorry

theorem correlation_coefficient_correct
  (sum_x_sq : ℝ := 80) (sum_y_sq : ℝ := 9000) 
  (sum_xy : ℝ := 800) : real.abs (correlation_coefficient sum_x_sq sum_y_sq sum_xy - 0.94) < 0.01 := by
sorry

end estimated_population_correct_correlation_coefficient_correct_l216_216205


namespace s_is_arithmetic_progression_l216_216063

variables (s : ℕ → ℕ) (ds1 ds2 : ℕ)

-- Conditions
axiom strictly_increasing : ∀ n, s n < s (n + 1)
axiom s_is_positive : ∀ n, 0 < s n
axiom s_s_is_arithmetic : ∃ d1, ∀ k, s (s k) = s (s 0) + k * d1
axiom s_s_plus1_is_arithmetic : ∃ d2, ∀ k, s (s k + 1) = s (s 0 + 1) + k * d2

-- Statement to prove
theorem s_is_arithmetic_progression : ∃ d, ∀ k, s (k + 1) = s 0 + k * d :=
sorry

end s_is_arithmetic_progression_l216_216063


namespace no_unused_signs_l216_216518

theorem no_unused_signs (total_signs used_signs signs_needed : ℕ) :
  used_signs = 222 ∧ total_signs = 224 ∧ signs_needed = 888 → (total_signs - used_signs = 0) :=
by
  intros h
  have h1 : total_signs - used_signs = 224 - 222 := by sorry
  have h2 : 224 - 222 = 2 := by sorry
  have h3 : 2 = 0 := by sorry -- Placeholder for steps proving 2 = 0 which is actually incorrect premise
  exact h3

end no_unused_signs_l216_216518


namespace simplify_expression_l216_216129

theorem simplify_expression : (- (1 / 216 : ℝ))^(-2 / 3) = 36 := 
by
  sorry

end simplify_expression_l216_216129


namespace harmonic_mean_closest_l216_216247

theorem harmonic_mean_closest :
  let a := 1
  let b := 100
  let harmonic_mean := (2 * a * b) / (a + b)
  abs (harmonic_mean - 2) < abs (harmonic_mean - 1) ∧
  abs (harmonic_mean - 2) < abs (harmonic_mean - 98) ∧
  abs (harmonic_mean - 2) < abs (harmonic_mean - 100) ∧
  abs (harmonic_mean - 2) < abs (harmonic_mean - 199) :=
by 
  let a := 1
  let b := 100
  let harmonic_mean := (2 * a * b) / (a + b)
  sorry

end harmonic_mean_closest_l216_216247


namespace sum_a_2013_l216_216336

-- Define the sequence
def a : ℕ → ℕ
| 0       => 1
| 1       => 2
| 2       => 3
| n + 3   => 24 / (a n * a (n + 1) * a (n + 2))

-- Define the condition of the sequence
axiom a_sequence_condition (n : ℕ) : a n * a (n + 1) * a (n + 2) * a (n + 3) = 24

-- Define the sum we need to calculate
noncomputable def sum_a (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), a i

-- State the theorem
theorem sum_a_2013 : sum_a 2013 = 5031 :=
by
  sorry

end sum_a_2013_l216_216336


namespace fifth_term_of_sequence_is_expected_l216_216753

noncomputable def sequence : ℕ → ℚ
| 0       := 2
| 3       := 50
| (n + 1) := 1/4 * (sequence n + sequence (n + 2))

theorem fifth_term_of_sequence_is_expected :
    sequence 4 = 2798 / 15 := 
by
    sorry

end fifth_term_of_sequence_is_expected_l216_216753


namespace cyclic_fibonacci_divisor_of_p_minus_1_l216_216399

-- Define the condition that square root of 5 exists in p-arithmetic.
def sqroot_of_5_exists (p : Nat) : Prop :=
  ∃ x : ℤ, x^2 ≡ 5 [ZMOD p]

-- Define the cyclic Fibonacci sequence.
def cyclic_fibonacci_sequence (p : Nat) : Prop :=
  ∀ (v : ℕ → ℤ) (n : ℕ), (∀ k, v (k + 2) ≡ v k + v (k + 1) [ZMOD p]) →
  v n ≡ v (n % ((p - 1) : ℕ)) [ZMOD p]

-- The theorem statement
theorem cyclic_fibonacci_divisor_of_p_minus_1 (p : ℕ) (hp : sqroot_of_5_exists p) :
  ∀ (v : ℕ → ℤ), (∀ k, v (k + 2) ≡ v k + v (k + 1) [ZMOD p]) →
  (∃ n, cyclic_fibonacci_sequence p) → n ∣ (p - 1) :=
sorry

end cyclic_fibonacci_divisor_of_p_minus_1_l216_216399


namespace remaining_tomatoes_to_cucumbers_ratio_l216_216945

theorem remaining_tomatoes_to_cucumbers_ratio 
  (initial_tomatoes initial_cucumbers : ℕ)
  (picked_tomatoes_yesterday picked_tomatoes_today picked_cucumbers : ℕ) :
  initial_tomatoes = 171 →
  initial_cucumbers = 225 →
  picked_tomatoes_yesterday = 134 →
  picked_tomatoes_today = 30 →
  picked_cucumbers = 157 →
  (initial_tomatoes - (picked_tomatoes_yesterday + picked_tomatoes_today)) = 7 →
  (initial_cucumbers - picked_cucumbers) = 68 →
  rat.mk 7 68 = (7 / 68 : ℚ) :=
by
  sorry

end remaining_tomatoes_to_cucumbers_ratio_l216_216945


namespace four_digit_even_numbers_count_l216_216717

/-- Number of four-digit even numbers that can be formed using the digits 0, 1, 2 is 17 -/
theorem four_digit_even_numbers_count : 
  {n : ℕ // 1000 ≤ n ∧ n < 10000 ∧ even n ∧ ∀ d ∈ n.digits 10, d ∈ {0,1,2} } → ∃ l : list ℕ, l = [n] ∧ l.length = 17 :=
sorry

end four_digit_even_numbers_count_l216_216717


namespace profit_per_cake_l216_216979

theorem profit_per_cake (ingredient_cost : ℝ) (packaging_cost : ℝ) (selling_price : ℝ) (cake_count : ℝ)
    (h1 : ingredient_cost = 12) (h2 : packaging_cost = 1) (h3 : selling_price = 15) (h4 : cake_count = 2) :
    selling_price - (ingredient_cost / cake_count + packaging_cost) = 8 := by
  sorry

end profit_per_cake_l216_216979


namespace units_digit_of_n_l216_216400

theorem units_digit_of_n (n : ℕ) (h : n = 56^78 + 87^65) : (n % 10) = 3 :=
by
  sorry

end units_digit_of_n_l216_216400


namespace incorrect_pair_not_integer_l216_216633

theorem incorrect_pair_not_integer (a b c m : ℝ) (d : ℤ) :
  let f := λ x : ℝ, a * x^3 + b * x - c / x + (d : ℝ) in
  ∀ pairs : (ℝ × ℝ),
    (pairs = (3, 7) ∨ pairs = (2, 6) ∨ pairs = (5, 11) ∨ pairs = (-1, 4)) →
    (f(m) = pairs.1 ∧ f(-m) = pairs.2) →
    (d = (pairs.1 + pairs.2) / 2) →
    pairs ≠ (-1, 4) :=
begin
  intros f pairs hpairs hf fsum d_ne_pairs,
  sorry
end

end incorrect_pair_not_integer_l216_216633


namespace find_larger_number_l216_216113

variable (x y : ℝ)
axiom h1 : x + y = 27
axiom h2 : x - y = 5

theorem find_larger_number : x = 16 :=
by
  sorry

end find_larger_number_l216_216113


namespace parallelogram_diagonal_square_l216_216794

theorem parallelogram_diagonal_square (A B C D P Q R S : Type)
    (area_ABCD : ℝ) (proj_A_P_BD proj_C_Q_BD proj_B_R_AC proj_D_S_AC : Prop)
    (PQ RS : ℝ) (d_squared : ℝ) 
    (h_area : area_ABCD = 24)
    (h_proj_A_P : proj_A_P_BD) (h_proj_C_Q : proj_C_Q_BD)
    (h_proj_B_R : proj_B_R_AC) (h_proj_D_S : proj_D_S_AC)
    (h_PQ_length : PQ = 8) (h_RS_length : RS = 10)
    : d_squared = 62 + 20*Real.sqrt 61 := sorry

end parallelogram_diagonal_square_l216_216794


namespace pyramid_surface_area_l216_216957

theorem pyramid_surface_area
  (base_side_length : ℝ)
  (peak_height : ℝ)
  (base_area : ℝ)
  (slant_height : ℝ)
  (triangular_face_area : ℝ)
  (total_surface_area : ℝ)
  (h1 : base_side_length = 10)
  (h2 : peak_height = 12)
  (h3 : base_area = base_side_length ^ 2)
  (h4 : slant_height = Real.sqrt (peak_height ^ 2 + (base_side_length / 2) ^ 2))
  (h5 : triangular_face_area = 0.5 * base_side_length * slant_height)
  (h6 : total_surface_area = base_area + 4 * triangular_face_area)
  : total_surface_area = 360 := 
sorry

end pyramid_surface_area_l216_216957


namespace total_coffee_consumed_l216_216782

def Ivory_hourly_coffee := 2
def Kimberly_hourly_coffee := Ivory_hourly_coffee
def Brayan_hourly_coffee := 4
def Raul_hourly_coffee := Brayan_hourly_coffee / 2
def duration_hours := 10

theorem total_coffee_consumed :
  (Brayan_hourly_coffee * duration_hours) + 
  (Ivory_hourly_coffee * duration_hours) + 
  (Kimberly_hourly_coffee * duration_hours) + 
  (Raul_hourly_coffee * duration_hours) = 100 :=
by sorry

end total_coffee_consumed_l216_216782


namespace minimum_guests_l216_216501

theorem minimum_guests (total_food : ℤ) (max_food_per_guest : ℤ) (food_bound : total_food = 325) (guest_bound : max_food_per_guest = 2) : (⌈total_food / max_food_per_guest⌉ : ℤ) = 163 :=
by {
  sorry 
}

end minimum_guests_l216_216501


namespace power_function_through_point_l216_216357

theorem power_function_through_point (k α : ℝ)
  (h : (λ x : ℝ, k * x ^ α) (1/2) = (√2)/2) :
  k - α = 1/2 :=
sorry

end power_function_through_point_l216_216357


namespace john_finishes_third_task_at_535_pm_l216_216445

-- Define the problem conditions and expected answer
theorem john_finishes_third_task_at_535_pm :
  let start_first_task := 135 -- Start time in minutes after noon (2:15 PM -> 2 * 60 + 15)
  let end_second_task := 265 -- End time in minutes after noon (4:25 PM -> 4 * 60 + 25)
  let combined_time_first_two_tasks := 140 -- Combined time of first two tasks in minutes
  let first_task_time := combined_time_first_two_tasks / 2 -- Time of first and third task in minutes
  let start_third_task := end_second_task -- Third task starts immediately after the second
  let end_third_task := start_third_task + first_task_time -- End time of third task
  in 
  end_third_task = 335 := -- End time in minutes after noon (5:35 PM -> 5 * 60 + 35)
sorry

end john_finishes_third_task_at_535_pm_l216_216445


namespace complex_coordinate_proof_l216_216424

noncomputable def i : ℂ := Complex.i

theorem complex_coordinate_proof : 
  ∀ (z : ℂ), z = i * (2 - i) → z = 1 + 2 * i := 
by
  intro z
  intro hz
  sorry

end complex_coordinate_proof_l216_216424


namespace xiaohong_mother_age_l216_216128

variable (x y : ℕ)

/-- A condition stating that when Xiaohong was 3 years old, 
    her mother's age was the same as Xiaohong's age this year. -/
def condition1 (x : ℕ) : Prop := x + 3 = y

/-- A condition stating that when Xiaohong's mother is 78 years old, 
    Xiaohong's age is the same as her mother’s age this year. -/
def condition2 (x : ℕ) : Prop := 78 - x = y

/-- The goal of our problem is to prove that the current age of Xiaohong's mother is 53,
    given the conditions. -/
theorem xiaohong_mother_age (x y : ℕ) 
    (h1 : condition1 x)
    (h2 : condition2 x) : y = 53 :=
by
  sorry

end xiaohong_mother_age_l216_216128


namespace ball_problem_solution_l216_216532

namespace BallProblem

-- Definitions based on the conditions
def num_colors := 4
def balls_per_color := 4
def total_balls := num_colors * balls_per_color

-- Set conditions for drawing two balls of different colors
def different_colors (c1 c2 : ℕ) (h1 : c1 ≠ c2) : Prop := true

-- Number of ways to draw two balls of different colors
def calculate_ways : ℕ :=
  (Nat.choose num_colors 2) * balls_per_color * balls_per_color

-- Probability calculation for the maximum difference in their numbers
def max_diff_prob (num_diff_ways : ℕ) : ℚ :=
  num_diff_ways / calculate_ways

theorem ball_problem_solution :
  (calculate_ways = 96) ∧ (max_diff_prob 12 = (1 : ℚ) / 8) :=
by
  sorry

end BallProblem

end ball_problem_solution_l216_216532


namespace estimated_population_correct_correlation_coefficient_correct_l216_216219

-- Definitions based on conditions
def total_plots := 200
def sample_plots := 20
def sum_x := 60
def sum_y := 1200
def sum_x_squared_diff := 80
def sum_y_squared_diff := 9000
def sum_xy_diff := 800

-- Estimated population calculation
noncomputable def estimated_population : ℕ :=
  (sum_y / sample_plots) * total_plots

theorem estimated_population_correct : estimated_population = 12000 := sorry

-- Correlation coefficient calculation
noncomputable def correlation_coefficient : Real :=
  sum_xy_diff / Real.sqrt (sum_x_squared_diff * sum_y_squared_diff)

theorem correlation_coefficient_correct : 
  Real.round (correlation_coefficient * 100) / 100 = 0.94 := sorry

end estimated_population_correct_correlation_coefficient_correct_l216_216219


namespace calculation_equation_l216_216999

theorem calculation_equation :
  (1 / 2)⁻¹ + (Real.sqrt 2) ^ 2 - 4 * Real.abs (-1 / 2) = 2 := by
  sorry

end calculation_equation_l216_216999


namespace general_term_formula_sum_first_n_terms_l216_216681

def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ (d : ℕ), ∀ n : ℕ, a (n + 1) = a n + d

variables (a : ℕ → ℕ)
variables (a1 : a 1 = 2) (a3a5 : a 3 + a 5 = 10)

theorem general_term_formula :
  arithmetic_seq a ∧ a 1 = 2 ∧ a 3 + a 5 = 10 →
  ∀ n, a n = n + 1 :=
sorry

def b (a : ℕ → ℕ) (n : ℕ) : ℕ := a n * 2 * n

def sum_b_reciprocal (b : ℕ → ℕ) (n : ℕ) : ℕ → ℚ
| 0       := 0
| (n + 1) := sum_b_reciprocal n + (1 / b (n + 1))

theorem sum_first_n_terms :
  (∀ n, a n = n + 1) →
  ∀ n, sum_b_reciprocal (b a) n = (n : ℚ) / (2 * (n + 1)) :=
sorry

end general_term_formula_sum_first_n_terms_l216_216681


namespace excircle_segment_equality_l216_216497

variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables (P Q L M L₁ L₂ : A)
variables (excircle_touches_P : P ∈ ExcircleTouchPts(A, B) := sorry)
variables (excircle_touches_Q : Q ∈ ExcircleTouchPts(A, C) := sorry)
variables (L_midpoint_PQ : L = midpoint P Q)
variables (M_midpoint_BC : M = midpoint B C)
variables (L₁_symmetric : L₁ = symmetric_point L (midpoint B M))
variables (L₂_symmetric : L₂ = symmetric_point L (midpoint C M))

theorem excircle_segment_equality :
  dist L₁ P = dist L₂ Q :=
sorry

end excircle_segment_equality_l216_216497


namespace evaluate_ceiling_expression_l216_216271

theorem evaluate_ceiling_expression : 
  Real.ceil (2 * (10 - (3 / 4))) = 19 := by
  sorry

end evaluate_ceiling_expression_l216_216271


namespace m_plus_n_eq_1243_l216_216943

-- Define the deck and the conditions of removing the pair and an additional card
def deck : Type := Fin 52
def removed_pair (n : ℕ) (h : 1 ≤ n ∧ n ≤ 13) : bool := h.1 && h.2  -- n represents the card number

def remaining_deck (cards_removed : Finset deck) : Finset deck :=
  finset.univ.filter (λ card, ¬ cards_removed.contains card)

lemma exist_matching_pair_and_single_card (cards_removed : Finset deck) :
  ∃ pair n : ℕ, n ≤ 13 ∧ ∃ other : deck, other ≠ pair ∧ cards_removed = {pair, other} := 
  sorry

-- Define the counting and probability calculations
def total_ways_to_select_two (cards : Finset deck) : ℕ := (cards.card * (cards.card - 1)) / 2
def ways_to_form_a_pair (cards : Finset deck) : ℕ :=
  let remaining := remaining_deck cards in
  let pairs_from_4 = 11 * (nat.choose 4 2) in
  let pairs_from_2 = nat.choose 2 2 in 
  pairs_from_4 + pairs_from_2

def probability_forming_pair : ℚ :=
  let total_ways = total_ways_to_select_two (remaining_deck (some cards_removed)) in
  let pair_ways = ways_to_form_a_pair (remaining_deck (some cards_removed)) in
  pair_ways / total_ways

-- Main theorem combining pieces together to establish m + n = 1243
theorem m_plus_n_eq_1243 (cards_removed : Finset deck) :
  probability_forming_pair * 1176 = 67 ∧ 67.gcd 1176 = 1 → 67 + 1176 = 1243 := 
  by {
    intro h,
    exact sorry }

end m_plus_n_eq_1243_l216_216943


namespace maximize_profit_l216_216760

def cost_A : ℝ := 3
def price_A : ℝ := 3.3
def cost_B : ℝ := 2.4
def price_B : ℝ := 2.8
def total_devices : ℕ := 50

def profit (x : ℕ) : ℝ := (price_A - cost_A) * x + (price_B - cost_B) * (total_devices - x)

def functional_relationship (x : ℕ) : ℝ := -0.1 * x + 20

def purchase_condition (x : ℕ) : Prop := 4 * x ≥ total_devices - x

theorem maximize_profit :
    functional_relationship (10) = 19 ∧ 
    (∀ x : ℕ, purchase_condition x → functional_relationship x ≤ 19) :=
by {
    -- Proof omitted
    sorry
}

end maximize_profit_l216_216760


namespace top_cube_white_faces_l216_216262

-- Definitions of the conditions
def painted_faces (gray_faces white_faces : Nat) : Prop := gray_faces + white_faces = 6

axiom denis_condition : ∃ cubes : Fin 10 → (Nat × Nat), ∀ i j, i ≠ j → cubes i ≠ cubes j

-- Given condition for the top cube
def top_cube : (Nat × Nat) := (2, 4) -- 2 gray faces, 4 white faces

theorem top_cube_white_faces :
  let (gray_faces, white_faces) := top_cube in
  painted_faces gray_faces white_faces →
  white_faces = 4 :=
by
  intros (gray_faces, white_faces) h
  simp [painted_faces] at h
  sorry

end top_cube_white_faces_l216_216262


namespace stratified_sampling_example_l216_216939

noncomputable def sample_proportion := 70 / 3500
noncomputable def total_students := 3500 + 1500
noncomputable def sample_size := total_students * sample_proportion

theorem stratified_sampling_example 
  (high_school_students : ℕ := 3500)
  (junior_high_students : ℕ := 1500)
  (sampled_high_school_students : ℕ := 70)
  (proportion_of_sampling : ℝ := sampled_high_school_students / high_school_students)
  (total_number_of_students : ℕ := high_school_students + junior_high_students)
  (calculated_sample_size : ℝ := total_number_of_students * proportion_of_sampling) :
  calculated_sample_size = 100 :=
by
  sorry

end stratified_sampling_example_l216_216939


namespace triangle_similarity_l216_216630

-- Define the conditions in the problem
variables {Point : Type} [Inhabited Point] [AddGroup Point] [HasDist Point] [HasAngle Point]
variables (E F B C M U G : Point)
variables (circle : Set Point)

-- Given conditions
axiom eq_dist_EF : dist E M = dist M F
axiom eq_dist_BC : dist B M = dist M C
axiom diameter : dist E F = 2 * dist E M
axiom M_on_EF : M ∈ line E F
axiom M_on_BC : M ∈ line B C
axiom U_between_BM : Between B U M
axiom G_on_circle : G ∈ circle
axiom EU_extends_to_G : ExtendedLine E U ∩ circle = {U, G}

-- Problem statement
theorem triangle_similarity : Similarity (triangle E U M) (triangle E F G) :=
  sorry

end triangle_similarity_l216_216630


namespace fraction_of_tadpoles_surviving_l216_216946

-- Definitions for the conditions
def frogs := 5
def tadpoles := 3 * frogs
def sustainable_frogs := 8

-- Main goal to prove
theorem fraction_of_tadpoles_surviving : (sustainable_frogs - frogs) / tadpoles = 1 / 5 := 
by
  sorry

end fraction_of_tadpoles_surviving_l216_216946


namespace deductive_reasoning_is_D_l216_216235

def optionA : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, 7 ^ n % 100 = m

def optionB : Prop :=
  ∀ f : ℝ → ℝ, (∀ x : ℝ, deriv (λ x, x^2) = 2 * x) ∧ (∀ x : ℝ, deriv (λ x, x^4) = 4 * x^3) ∧
  ((∀ x : ℝ, deriv cos x = -sin x) →
   (∀ x : ℝ, (∀ f : ℝ → ℝ, even_function f → deriv f = odd_function (λ y, sin y))))

def optionC : Prop :=
  ∀ (a b : ℝ), (equilateral_triangle a b → euclidean_plane_eq_ratios_area a b) ∧
  ((in_space a b) → tetrahedra_volume_ratios a b)

def optionD : Prop :=
  ∀ (M : Type) [alkali_metal M], (∀ m : M, reacts_with_water m)

theorem deductive_reasoning_is_D : optionD :=
  sorry

end deductive_reasoning_is_D_l216_216235


namespace probability_picasso_consecutive_l216_216470

theorem probability_picasso_consecutive :
  let totalArtPieces := 12
  let picassoPrints := 4
  let totalArrangements := 12.factorial
  let blockArrangements := (12 - 4 + 1).factorial * 4.factorial
  (blockArrangements / totalArrangements : ℚ) = 1 / 55 :=
by 
  -- Definitions from Conditions
  let totalArtPieces := 12
  let picassoPrints := 4
  let totalArrangements := 12.factorial
  let blockArrangements := (12 - 4 + 1).factorial * 4.factorial

  -- Calculation based on the provided solution
  have h1 : (12 - 4 + 1).factorial = 9.fact := by rw Nat.factorial
  have h2 : blockArrangements = 9.fact * 4.fact := by rw [h1, Nat.factorial]
  have h3 : 12.factorial = 479001600 := rfl -- 12! = 479001600
  
  -- Obtained Simplified Answer
  have h4 : 9.fact * 4.fact = 8709120 := by norm_num [Nat.factorial, h2]
  have h5 : (9.fact * 4.fact) / 479001600 = 1 / 55 := by norm_num [h3, ←h4]
  
  exact h5

end probability_picasso_consecutive_l216_216470


namespace height_relationship_height_at_90_l216_216857

noncomputable def f (x : ℝ) : ℝ := (1/2) * x

theorem height_relationship :
  (∀ x : ℝ, (x = 10 -> f x = 5) ∧ (x = 30 -> f x = 15) ∧ (x = 50 -> f x = 25) ∧ (x = 70 -> f x = 35)) → (∀ x : ℝ, f x = (1/2) * x) :=
by
  sorry

theorem height_at_90 :
  f 90 = 45 :=
by
  sorry

end height_relationship_height_at_90_l216_216857


namespace tan_10_70_div_eq_sqrt3_div3_cos_expr_eq_2_l216_216997
noncomputable theory

open Real

theorem tan_10_70_div_eq_sqrt3_div3 :
  (tan (10 * π / 180) * tan (70 * π / 180)) / (tan (70 * π / 180) - tan (10 * π / 180) + tan (120 * π / 180)) = sqrt 3 / 3 :=
sorry

theorem cos_expr_eq_2 :
  (2 * cos (40 * π / 180) + cos (10 * π / 180) * (1 + sqrt 3 * tan (10 * π / 180))) / sqrt (1 + cos (10 * π / 180)) = 2 :=
sorry

end tan_10_70_div_eq_sqrt3_div3_cos_expr_eq_2_l216_216997


namespace min_value_of_16x_minus_4x_plus_1_l216_216295

noncomputable def find_min_value : ℝ :=
  let f (x : ℝ) := 16^x - 4^x + 1
  in Inf (set.range f)

theorem min_value_of_16x_minus_4x_plus_1 : find_min_value = 3/4 :=
  sorry

end min_value_of_16x_minus_4x_plus_1_l216_216295


namespace profit_ratio_l216_216884

-- Definitions based on conditions
-- Let A_orig and B_orig represent the original profits of stores A and B
-- after increase and decrease respectively, they become equal

variable (A_orig B_orig : ℝ)
variable (h1 : (1.2 * A_orig) = (0.9 * B_orig))

-- Prove that the original profit of store A was 75% of the profit of store B
theorem profit_ratio (h1 : 1.2 * A_orig = 0.9 * B_orig) : A_orig = 0.75 * B_orig :=
by
  -- Insert proof here
  sorry

end profit_ratio_l216_216884


namespace num_solutions_l216_216658

-- Define the problem and the condition
def matrix_eq (x : ℝ) : Prop :=
  3 * x^2 - 4 * x = 7

-- Define the main theorem to prove the number of solutions
theorem num_solutions : ∃! x : ℝ, matrix_eq x :=
sorry

end num_solutions_l216_216658


namespace tammy_speed_second_day_l216_216143

theorem tammy_speed_second_day :
  ∀ (v1 t1 v2 t2 : ℝ), 
    t1 + t2 = 14 →
    t2 = t1 - 2 →
    v2 = v1 + 0.5 →
    v1 * t1 + v2 * t2 = 52 →
    v2 = 4 :=
by
  intros v1 t1 v2 t2 h1 h2 h3 h4
  sorry

end tammy_speed_second_day_l216_216143


namespace dog_roaming_area_difference_l216_216839

noncomputable def area_half_circle (r : ℝ) : ℝ := (1 / 2) * π * r^2

noncomputable def area_three_quarter_circle (r : ℝ) : ℝ := (3 / 4) * π * r^2

theorem dog_roaming_area_difference :
  let r := 10 in
  (area_three_quarter_circle r - area_half_circle r) = 25 * π :=
by
  sorry

end dog_roaming_area_difference_l216_216839


namespace total_profit_is_28000_l216_216155

noncomputable def investment_A (investment_B : ℝ) : ℝ := 3 * investment_B
noncomputable def period_A (period_B : ℝ) : ℝ := 2 * period_B
noncomputable def profit_B : ℝ := 4000
noncomputable def total_profit (investment_B period_B : ℝ) : ℝ :=
  let x := investment_B * period_B
  let a_share := 6 * x
  profit_B + a_share

theorem total_profit_is_28000 (investment_B period_B : ℝ) : 
  total_profit investment_B period_B = 28000 :=
by
  have h1 : profit_B = 4000 := rfl
  have h2 : investment_A investment_B = 3 * investment_B := rfl
  have h3 : period_A period_B = 2 * period_B := rfl
  simp [total_profit, h1, h2, h3]
  have x_def : investment_B * period_B = 4000 := by sorry
  simp [x_def]
  sorry

end total_profit_is_28000_l216_216155


namespace terminal_side_in_third_or_fourth_quadrant_l216_216730

theorem terminal_side_in_third_or_fourth_quadrant (θ : ℝ) (h : Real.sin (Real.pi + θ) = 4 / 5) : 
    ∃ q, (q = 3 ∨ q = 4) ∧ term_side_in_quadrant θ q := 
begin
  sorry
end

end terminal_side_in_third_or_fourth_quadrant_l216_216730


namespace estimated_population_correct_correlation_coefficient_correct_l216_216204

noncomputable def estimated_population
  (num_plots : ℕ) (sample_size : ℕ) 
  (sum_y : ℕ) : ℕ :=
(sum_y / sample_size) * num_plots

noncomputable def correlation_coefficient
  (sum_x_sq : ℝ) (sum_y_sq : ℝ) 
  (sum_xy : ℝ) : ℝ :=
sum_xy / real.sqrt(sum_x_sq * sum_y_sq)

theorem estimated_population_correct
  (num_plots : ℕ := 200) (sample_size : ℕ := 20) 
  (sum_y : ℕ := 1200) : estimated_population num_plots sample_size sum_y = 12000 := by
sorry

theorem correlation_coefficient_correct
  (sum_x_sq : ℝ := 80) (sum_y_sq : ℝ := 9000) 
  (sum_xy : ℝ := 800) : real.abs (correlation_coefficient sum_x_sq sum_y_sq sum_xy - 0.94) < 0.01 := by
sorry

end estimated_population_correct_correlation_coefficient_correct_l216_216204


namespace combination_property_problem_solution_l216_216149

open Nat

def combination (n k : ℕ) : ℕ :=
  if h : k ≤ n then (factorial n) / (factorial k * factorial (n - k)) else 0

theorem combination_property (n k : ℕ) (h₀ : 1 ≤ k) (h₁ : k ≤ n) :
  combination n k + combination n (k - 1) = combination (n + 1) k := sorry

theorem problem_solution :
  (combination 3 2 + combination 4 2 + combination 5 2 + combination 6 2 + combination 7 2 + 
   combination 8 2 + combination 9 2 + combination 10 2 + combination 11 2 + combination 12 2 + 
   combination 13 2 + combination 14 2 + combination 15 2 + combination 16 2 + combination 17 2 + 
   combination 18 2 + combination 19 2) = 1139 := sorry

end combination_property_problem_solution_l216_216149


namespace number_verification_l216_216611

def is_product_of_four_consecutive (n : ℕ) : Prop :=
  ∃ a : ℕ, n = a * (a + 1) * (a + 2) * (a + 3)

theorem number_verification (h1 : 1680 % 3 = 0) (h2 : ∃ a : ℕ, 1680 = a * (a + 1) * (a + 2) * (a + 3)) : 
  is_product_of_four_consecutive 1680 :=
by
  sorry

end number_verification_l216_216611


namespace jacket_price_restoration_l216_216872

/-- The conditions to be used are:
1. A price reduction by 25% followed by
2. A further price reduction by 15%.

We need to prove that to restore the jacket's price to the original, the percentage increase required is approximately 56.86%.
-/
theorem jacket_price_restoration (original_price reduced_price1 reduced_price2 : ℝ) (h1 : reduced_price1 = original_price * 0.75) (h2 : reduced_price2 = reduced_price1 * 0.85) : 
  ((original_price - reduced_price2) / reduced_price2) * 100 ≈ 56.86 :=
sorry

end jacket_price_restoration_l216_216872


namespace trigonometric_inequality_l216_216813

theorem trigonometric_inequality 
  (α β γ : ℝ)
  (h₁ : 0 < α)
  (h₂ : α < β)
  (h₃ : β < γ)
  (h₄ : γ ≤ π / 2) :
  (cos β - cos γ) / (cos α - cos β) > (8 / π^2) * (γ - β) / (β - α) :=
sorry

end trigonometric_inequality_l216_216813


namespace three_digit_count_eq_22_l216_216394

def is_three_digit_number (n : ℕ) : Prop := n >= 100 ∧ n <= 999

def sum_of_digits (n : ℕ) : ℕ := n / 100 + (n % 100) / 10 + n % 10

def is_perfect_square (s : ℕ) : Prop := ∃ (k : ℕ), k * k = s

def sum_is_perfect_square (n : ℕ) : Prop := is_perfect_square (sum_of_digits n)

def sum_in_range (n : ℕ) : Prop := sum_of_digits n >= 10 ∧ sum_of_digits n <= 27

theorem three_digit_count_eq_22 :
  {n : ℕ | is_three_digit_number n ∧ sum_is_perfect_square n ∧ sum_in_range n}.to_finset.card = 22 :=
by
  sorry

end three_digit_count_eq_22_l216_216394


namespace b_3_pow_100_l216_216453

noncomputable def b : ℕ → ℕ
| 0       := 2
| (n + 1) := if h : 3^(n+1) = n + 1 then b n
             else let m := (n + 1) / 3 in m^2 * b m

theorem b_3_pow_100 : b (3^100) = 2 * 3^9900 := 
by
  sorry

end b_3_pow_100_l216_216453


namespace granger_total_amount_l216_216378

-- Define the constants for the problem
def cost_spam := 3
def cost_peanut_butter := 5
def cost_bread := 2
def quantity_spam := 12
def quantity_peanut_butter := 3
def quantity_bread := 4

-- Define the total cost calculation
def total_cost := (quantity_spam * cost_spam) + (quantity_peanut_butter * cost_peanut_butter) + (quantity_bread * cost_bread)

-- The theorem we need to prove
theorem granger_total_amount : total_cost = 59 := by
  sorry

end granger_total_amount_l216_216378


namespace total_books_on_shelves_l216_216102

-- Definitions based on conditions
def num_shelves : Nat := 150
def books_per_shelf : Nat := 15

-- The statement to be proved
theorem total_books_on_shelves : num_shelves * books_per_shelf = 2250 := by
  sorry

end total_books_on_shelves_l216_216102


namespace part_I_part_II_l216_216452

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - 2 * x + 2 * a

theorem part_I (a : ℝ) :
  let x := Real.log 2
  ∃ I₁ I₂ : Set ℝ,
    (∀ x ∈ I₁, f a x > f a (Real.log 2)) ∧
    (∀ x ∈ I₂, f a x < f a (Real.log 2)) ∧
    I₁ = Set.Iio (Real.log 2) ∧
    I₂ = Set.Ioi (Real.log 2) ∧
    f a (Real.log 2) = 2 * (1 - Real.log 2 + a) :=
by sorry

theorem part_II (a : ℝ) (h : a > Real.log 2 - 1) (x : ℝ) (hx : 0 < x) :
  Real.exp x > x^2 - 2 * a * x + 1 :=
by sorry

end part_I_part_II_l216_216452


namespace exist_disjoint_subsets_with_equal_union_l216_216028

theorem exist_disjoint_subsets_with_equal_union (n : ℕ) (hpos : n > 0)
  (A : Fin (n+2) → Set (Fin n)) (hA : ∀ i, (1 ≤ i : ℕ) ∧ (A i).Nonempty) :
  ∃ (I J : Fin (n+2) → Prop), (∃ i, I i) ∧ (∃ j, J j) ∧ (∀ i, I i → ∀ j, J j → i ≠ j) ∧
  (⋃ (i : Fin (n+2)) (hi : I i), A i) = (⋃ (j : Fin (n+2)) (hj : J j), A j) :=
by
  sorry

end exist_disjoint_subsets_with_equal_union_l216_216028


namespace smallest_positive_y_achieving_max_value_l216_216252

def g (y : ℝ) : ℝ := (Real.cos (y / 4)) + (Real.cos (y / 13))

theorem smallest_positive_y_achieving_max_value :
  ∃ y : ℝ, (0 < y) ∧ (g y = 2) ∧ (∀ y' : ℝ, y' > 0 → g y' = 2 → y' ≥ y) :=
sorry

end smallest_positive_y_achieving_max_value_l216_216252


namespace solve_for_s_l216_216590

-- Definition of the given problem conditions
def parallelogram_sides_60_angle_sqrt_area (s : ℝ) :=
  ∃ (area : ℝ), (area = 27 * Real.sqrt 3) ∧
  (3 * s * s * Real.sqrt 3 = area)

-- Proof statement to demonstrate the equivalence of the theoretical and computed value of s
theorem solve_for_s (s : ℝ) : parallelogram_sides_60_angle_sqrt_area s → s = 3 :=
by
  intro h
  sorry

end solve_for_s_l216_216590


namespace max_value_x_minus_2y_exists_max_value_x_minus_2y_l216_216353

theorem max_value_x_minus_2y 
  (x y : ℝ) 
  (h : x^2 - 4 * x + y^2 = 0) : 
  x - 2 * y ≤ 2 + 2 * Real.sqrt 5 :=
sorry

theorem exists_max_value_x_minus_2y 
  (x y : ℝ) 
  (h : x^2 - 4 * x + y^2 = 0) : 
  ∃ (x y : ℝ), x - 2 * y = 2 + 2 * Real.sqrt 5 :=
sorry

end max_value_x_minus_2y_exists_max_value_x_minus_2y_l216_216353


namespace arithmetic_sequence_property_l216_216010

variable (a : ℕ → ℝ)
variable (m n : ℕ)

-- Definition of sum of the first n terms of an arithmetic sequence
def S (k : ℕ) : ℝ := ∑ i in range k, a (i + 1)

-- Proof statement
theorem arithmetic_sequence_property
  (h_pos : ∀ i, a i > 0)
  (h_a1 : a 1 ≥ 1) :
  S a (2 * m) * S a (2 * n) ≤ (S a (m + n))^2 ∧
  (Real.log (S a (2 * m)) * Real.log (S a (2 * n)) ≤ (Real.log (S a (m + n)))^2) :=
by
  sorry

end arithmetic_sequence_property_l216_216010


namespace value_of_y_l216_216912

theorem value_of_y (y : ℝ) (h : |y| = |y - 3|) : y = 3 / 2 :=
sorry

end value_of_y_l216_216912


namespace original_salary_l216_216922

-- Given conditions as definitions
def salaryAfterRaise (x : ℝ) : ℝ := 1.10 * x
def salaryAfterReduction (x : ℝ) : ℝ := salaryAfterRaise x * 0.95
def finalSalary : ℝ := 1045

-- Statement to prove
theorem original_salary (x : ℝ) (h : salaryAfterReduction x = finalSalary) : x = 1000 :=
by
  sorry

end original_salary_l216_216922


namespace find_larger_number_l216_216110

theorem find_larger_number (a b : ℤ) (h1 : a + b = 27) (h2 : a - b = 5) : a = 16 := by
  sorry

end find_larger_number_l216_216110


namespace min_moves_correct_l216_216135

def min_moves (n : ℕ) (k : ℕ) (c : Fin n → ℕ) : ℕ :=
  let d (i : Fin n) := c i - k
  (Finset.range n).sum (λ i, |(Finset.range (i + 1)).sum (λ j, d j)|)

theorem min_moves_correct (n k : ℕ) (c : Fin n → ℕ) (h_sum_c : (Finset.range n).sum c = n * k) :
  ∃ m, min_moves n k c = m ∧ (∀ i, c i = k + d i) :=
by
  sorry

end min_moves_correct_l216_216135


namespace compute_quotient_of_q_and_r_l216_216540

theorem compute_quotient_of_q_and_r (p q r s t : ℤ) (h_eq_4 : 256 * p + 64 * q + 16 * r + 4 * s + t = 0)
                                     (h_eq_neg3 : -27 * p + 9 * q - 3 * r + s + t = 0)
                                     (h_eq_0 : t = 0)
                                     (h_p_nonzero : p ≠ 0) :
                                     (q + r) / p = -13 :=
by
  have eq1 := h_eq_4
  have eq2 := h_eq_neg3
  rw [h_eq_0] at eq1 eq2
  sorry

end compute_quotient_of_q_and_r_l216_216540


namespace probability_area_l216_216951

noncomputable def probability_x_y_le_five (x y : ℝ) : ℚ :=
  if 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 8 ∧ x + y ≤ 5 then 1 else 0

theorem probability_area {P : ℚ} :
  (∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 8 → P = probability_x_y_le_five x y / (4 * 8)) →
  P = 5 / 16 :=
by
  sorry

end probability_area_l216_216951


namespace three_city_population_l216_216529

noncomputable def totalPopulation (boise seattle lakeView: ℕ) : ℕ :=
  boise + seattle + lakeView

theorem three_city_population (pBoise pSeattle pLakeView : ℕ)
  (h1 : pBoise = 3 * pSeattle / 5)
  (h2 : pLakeView = pSeattle + 4000)
  (h3 : pLakeView = 24000) :
  totalPopulation pBoise pSeattle pLakeView = 56000 := by
  sorry

end three_city_population_l216_216529


namespace probability_prime_multiple_13_l216_216834

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_multiple_of_13 (n : ℕ) : Prop :=
  n % 13 = 0

def probability_card (n : ℕ) : ℚ :=
  if h : 1 ≤ n ∧ n ≤ 100 then 1 / 100 else 0

theorem probability_prime_multiple_13 :
  probability_card (classical.some (exists_prime_multiple_13)) = 1 / 100 :=
sorry

def exists_prime_multiple_13 : { x : ℕ // 1 ≤ x ∧ x ≤ 100 ∧ is_prime x ∧ is_multiple_of_13 x } :=
begin
  existsi 13,
  split, exact nat.le_refl _,
  split, exact nat.le_succ _,
  split,
  { split, exact dec_trivial,
    intro m, intro h,
    cases nat.dvd_prime.mp h (dec_trivial : nat.prime 13) with h1 h2,
    { left, exact h1, },
    { right, exact h2, }
  },
  apply nat.dvd_refl,
end

end probability_prime_multiple_13_l216_216834


namespace evaluate_exponent_l216_216276

theorem evaluate_exponent : (3^2)^4 = 6561 := sorry

end evaluate_exponent_l216_216276


namespace num_valid_triples_l216_216735

theorem num_valid_triples : 
  (∃ (a1 a2 a3: ℕ),
    1 ≤ a1 ∧ a1 < a2 ∧ a2 < a3 ∧ a3 ≤ 14 ∧ 
    a2 - a1 ≥ 3 ∧ a3 - a2 ≥ 3) → 
  (fintype.card { t : ℕ × ℕ × ℕ // 
    let (a1, a2, a3) := t in 
    1 ≤ a1 ∧ a1 < a2 ∧ a2 < a3 ∧ a3 ≤ 14 ∧ 
    a2 - a1 ≥ 3 ∧ a3 - a2 ≥ 3 } = 120) := 
by 
  sorry

end num_valid_triples_l216_216735


namespace negation_of_existential_prop_l216_216370

open Real

theorem negation_of_existential_prop :
  ¬ (∃ x, x ≥ π / 2 ∧ sin x > 1) ↔ ∀ x, x < π / 2 → sin x ≤ 1 :=
by
  sorry

end negation_of_existential_prop_l216_216370


namespace perimeter_of_triangle_l216_216077

-- Definitions for the given conditions
variables {X Y Z P Q R : Type} [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
variables (r : Real) (XP PY : Real)
variables (XY XZ YZ : X → Y → Z → Real)

-- Given conditions
def is_tangent (X Y Z : Type) (r : Real) (P : Type) (XP PY : Real) : Prop :=
  r = 10 ∧
  XP = 15 ∧
  PY = 25

-- The proof problem
theorem perimeter_of_triangle {X Y Z : Type} [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
  (P Q R : X → Y → Z → Prop) (r : Real) (XP PY : Real) (XY XZ YZ : X → Y → Z → Real)
  (tangent : is_tangent X Y Z r P XP PY) :
  2 * (40 + 160 / 11) = 1200 / 11 :=
by
  sorry

end perimeter_of_triangle_l216_216077


namespace smallest_n_l216_216553

theorem smallest_n (n : ℕ) (h : 0 < n) (h1 : 813 * n % 30 = 1224 * n % 30) : n = 10 := 
sorry

end smallest_n_l216_216553


namespace power_of_power_example_l216_216274

theorem power_of_power_example : (3^2)^4 = 6561 := by
  sorry

end power_of_power_example_l216_216274


namespace wendy_furniture_time_l216_216548

variable (chairs tables pieces minutes total_time : ℕ)

theorem wendy_furniture_time (h1 : chairs = 4) (h2 : tables = 4) (h3 : pieces = chairs + tables) (h4 : minutes = 6) (h5 : total_time = pieces * minutes) : total_time = 48 :=
by
  sorry

end wendy_furniture_time_l216_216548


namespace largest_non_decomposable_proof_l216_216427

def coin_denomination (n : ℕ) : list ℕ :=
  (List.range (n + 1)).map (λ k => (5 ^ (n - k)) * (7 ^ k))

def n_decomposable (n s : ℕ) : Prop :=
  ∃ (a : Fin (n + 1) → ℕ), (∑ i, a i * (coin_denomination n).get i) = s

def largest_nondecomposable_val (n : ℕ) : ℕ :=
  2 * (7^(n+1)) - 3 * (5^(n+1))

theorem largest_non_decomposable_proof {n : ℕ} : 
  ∀ s : ℕ, s > largest_nondecomposable_val n → n_decomposable n s :=
sorry

end largest_non_decomposable_proof_l216_216427


namespace f_bound_l216_216818

theorem f_bound (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f (x + 1) - f x = 2 * x + 1) 
  (h2 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |f x| ≤ 1) : ∀ x : ℝ, |f x| ≤ 2 + x^2 :=
by
  sorry

end f_bound_l216_216818


namespace minimum_cos_diff_l216_216573

open Real

theorem minimum_cos_diff (x y z : ℝ) 
  (h1 : sqrt 3 * sin x = tan y)
  (h2 : 2 * sin y = cot z)
  (h3 : sin z = 2 * tan x) : 
  ∃ w, (cos x - cos z = w) ∧ (w = -7 * sqrt 2 / 6) := 
sorry

end minimum_cos_diff_l216_216573


namespace percent_part_of_whole_l216_216153

theorem percent_part_of_whole (Part : ℝ) (Whole : ℝ) : 
  Part = 120 → Whole = 40 → (Part / Whole) * 100 = 300 :=
by
  assume h1 : Part = 120,
  assume h2 : Whole = 40,
  sorry

end percent_part_of_whole_l216_216153


namespace solve_for_x_l216_216732

noncomputable def condition (x : ℝ) : Prop :=
  (35 * x)^2 / 100 * x = (23 / 18 * 9500 / 100) - 175

theorem solve_for_x : ∃ x : ℝ, condition x ∧ x ≈ -0.62857 :=
sorry

end solve_for_x_l216_216732


namespace product_sequence_l216_216998

theorem product_sequence : 
  (∏ n in Finset.range 667, (3 * (n + 1) : ℚ) / (3 * (n + 1) + 3)) = 1 / 668 :=
by
  sorry

end product_sequence_l216_216998


namespace tom_read_pages_l216_216892

theorem tom_read_pages (chapters pages_per_chapter : ℕ) (h1 : chapters = 3) (h2 : pages_per_chapter = 8) :
  chapters * pages_per_chapter = 24 :=
by
  rw [h1, h2]
  norm_num

end tom_read_pages_l216_216892


namespace arithmetic_sequence_seventh_term_l216_216903

variable (a1 a15 : ℚ)
variable (n : ℕ) (a7 : ℚ)

-- Given conditions
def first_term (a1 : ℚ) : Prop := a1 = 3
def last_term (a15 : ℚ) : Prop := a15 = 72
def total_terms (n : ℕ) : Prop := n = 15

-- Arithmetic sequence formula
def common_difference (d : ℚ) : Prop := d = (72 - 3) / (15 - 1)
def nth_term (a_n : ℚ) (a1 : ℚ) (n : ℕ) (d : ℚ) : Prop := a_n = a1 + (n - 1) * d

-- Prove that the 7th term is approximately 33
theorem arithmetic_sequence_seventh_term :
  ∀ (a1 a15 : ℚ) (n : ℕ), first_term a1 → last_term a15 → total_terms n → ∃ a7 : ℚ, 
  nth_term a7 a1 7 ((a15 - a1) / (n - 1)) ∧ (33 - 0.5) < a7 ∧ a7 < (33 + 0.5) :=
by {
  sorry
}

end arithmetic_sequence_seventh_term_l216_216903


namespace distance_between_given_lines_l216_216072

-- Definitions of the lines
def line1 (x y : ℝ) : Prop := x + 2 * y + 4 = 0
def line2 (x y : ℝ) : Prop := 2 * x + 4 * y + 7 = 0

-- Definition of the distance formula between two lines in the form Ax + By + C = 0
def distance_between_parallel_lines (A B C1 C2 : ℝ) : ℝ := 
  (abs (C2 - C1)) / sqrt (A^2 + B^2)

-- The theorem to be proven
theorem distance_between_given_lines : 
  distance_between_parallel_lines 2 4 8 7 = sqrt 5 / 10 :=
  sorry

end distance_between_given_lines_l216_216072


namespace quadrilateral_areas_equal_l216_216147

variables {A B C D P Q O X Y Z T : Type}
variables [inhabited A] [inhabited B] [inhabited C] [inhabited D]
variables (AB CD AC BD AP QB PX QY XZ ZT XT YB OZ YC OT : ℝ)

noncomputable def mid_point (a b : ℝ) : ℝ := (a + b) / 2

-- Conditions
def quadrilateral (A B C D : Type) : Prop := inhabited A ∧ inhabited B ∧ inhabited C ∧ inhabited D
def midpoint_diagonals (BD AC : ℝ) := P = mid_point 0 BD ∧ Q = mid_point 0 AC
def intersection_parallel (P Q : ℝ) (O : Type) : Prop :=
  ∃ x y : ℝ, AP = x * BD ∧ QB = y * AC ∧ O = classical.some (classical.some_spec (classical.some_spec (classical.some ineq rel)))

def midpoints_sides (AB BC CD DA : ℝ) :=
  X = mid_point 0 AB ∧ Y = mid_point 0 BC ∧ Z = mid_point 0 CD ∧ T = mid_point 0 DA

/-- Prove that the areas of the quadrilaterals OXBY, OYCZ, OZDT, and OTAX are equal. -/
theorem quadrilateral_areas_equal 
  (h_quad: quadrilateral A B C D)
  (h_mid_diag: midpoint_diagonals BD AC)
  (h_intersect: intersection_parallel P Q O)
  (h_mid_sides: midpoints_sides AB BC CD DA)
  : [OXBY] = [OYCZ] ∧ [OYCZ] = [OZDT] ∧ [OZDT] = [OTAX] :=
sorry

end quadrilateral_areas_equal_l216_216147


namespace total_distance_proof_l216_216915

-- Conditions for the problem
variables (d : ℝ)
variables (t_total : ℝ)

axiom speed_condition1 : d / 2 + d / 6 + d / 6 = t_total
axiom total_time : t_total = 11 / 60

-- The statement to be proved
theorem total_distance_proof (h1 : speed_condition1 d t_total) (h2 : total_time t_total) : 3 * d = 33 / 50 :=
by sorry

end total_distance_proof_l216_216915


namespace avg_price_two_returned_theorem_l216_216920

-- Defining the initial conditions given in the problem
def avg_price_of_five (price: ℕ) (packets: ℕ) : Prop :=
  packets = 5 ∧ price = 20

def avg_price_of_three_remaining (price: ℕ) (packets: ℕ) : Prop :=
  packets = 3 ∧ price = 12
  
def cost_of_packets (price packets: ℕ) := price * packets

noncomputable def avg_price_two_returned (total_initial_cost total_remaining_cost: ℕ) :=
  (total_initial_cost - total_remaining_cost) / 2

-- The Lean 4 proof statement
theorem avg_price_two_returned_theorem (p1 p2 p3 p4: ℕ):
  avg_price_of_five p1 5 →
  avg_price_of_three_remaining p2 3 →
  cost_of_packets p1 5 = 100 →
  cost_of_packets p2 3 = 36 →
  avg_price_two_returned 100 36 = 32 :=
by
  intros h1 h2 h3 h4
  sorry

end avg_price_two_returned_theorem_l216_216920


namespace trigonometric_identity_l216_216877

theorem trigonometric_identity :
  sin 80 * sin 40 - cos 80 * cos 40 = 1 / 2 :=
by
  -- Sorry statement is used to skip the proof.
  sorry

end trigonometric_identity_l216_216877


namespace angle_CBD_30_l216_216570

theorem angle_CBD_30 (A B C D E : Type) [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] [AddGroup E] 
  (AD DC : A = C) (AE BD : B = D) (angle_AEC : AE = 90) : 
  angle_CBD = 30 := by 
  sorry

end angle_CBD_30_l216_216570


namespace water_hydrogen_oxygen_ratio_l216_216120

/-- In a mixture of water with a total mass of 171 grams, 
    where 19 grams are hydrogen, the ratio of hydrogen to oxygen by mass is 1:8. -/
theorem water_hydrogen_oxygen_ratio 
  (h_total_mass : ℝ) 
  (h_mass : ℝ) 
  (o_mass : ℝ) 
  (h_condition : h_total_mass = 171) 
  (h_hydrogen_mass : h_mass = 19) 
  (h_oxygen_mass : o_mass = h_total_mass - h_mass) :
  h_mass / o_mass = 1 / 8 := 
by
  sorry

end water_hydrogen_oxygen_ratio_l216_216120


namespace power_of_power_example_l216_216275

theorem power_of_power_example : (3^2)^4 = 6561 := by
  sorry

end power_of_power_example_l216_216275


namespace limit_of_sum_of_areas_as_n_approaches_infinity_l216_216161

noncomputable def diagonal (m : ℝ) : ℝ := Real.sqrt (5 * m^2)

noncomputable def first_circle_radius (m : ℝ) : ℝ := diagonal m / 2

noncomputable def first_circle_area (m : ℝ) : ℝ := Real.pi * (first_circle_radius m)^2

noncomputable def scaling_factor : ℝ := (Real.sqrt (5) / 2)^2

noncomputable def sum_of_areas (m : ℝ) (n : ℕ) : ℝ := 
  (first_circle_area m) * ((1 - scaling_factor^(n+1)) / (1 - scaling_factor))

theorem limit_of_sum_of_areas_as_n_approaches_infinity (m : ℝ) :
  ∃ (S : ℝ), Tendsto (λ n, sum_of_areas m n) atTop (𝓝 S) ∧ S = 5 * Real.pi * m^2 :=
by
  sorry

end limit_of_sum_of_areas_as_n_approaches_infinity_l216_216161


namespace total_food_per_day_l216_216534

theorem total_food_per_day :
  let num_puppies := 4
  let num_dogs := 3
  let dog_meal_weight := 4
  let dog_meals_per_day := 3
  let dog_food_per_day := dog_meal_weight * dog_meals_per_day
  let total_dog_food_per_day := dog_food_per_day * num_dogs
  let puppy_meal_weight := dog_meal_weight / 2
  let puppy_meals_per_day := dog_meals_per_day * 3
  let puppy_food_per_day := puppy_meal_weight * puppy_meals_per_day
  let total_puppy_food_per_day := puppy_food_per_day * num_puppies
  total_dog_food_per_day + total_puppy_food_per_day = 108 :=
by
  sorry

end total_food_per_day_l216_216534


namespace sin_cos_identity_l216_216150

theorem sin_cos_identity : 
  sin (real.pi * 18 / 180) * cos (real.pi * 12 / 180) + cos (real.pi * 18 / 180) * sin (real.pi * 12 / 180) = 1 / 2 :=
by
  sorry

end sin_cos_identity_l216_216150


namespace solve_tank_capacity_l216_216097

noncomputable def tank_capacity : ℝ :=
  let rate_A := 5 * 60 in -- Pipe A rate in litres/hour
  let rate_B := 7 * 60 in -- Pipe B rate in litres/hour
  let rate_C := 9 * 60 in -- Pipe C rate in litres/hour
  let total_inlet_rate := rate_A + rate_B + rate_C in
  let leak_rate_A := C / 10 in -- Leak rate with only Pipe A
  let leak_rate_AB := leak_rate_A * 1.15 in
  let leak_rate_ABC := leak_rate_AB * 1.15 in -- Leak rate with all pipes
  let net_rate := total_inlet_rate - leak_rate_ABC in
  let time_empty := 15 in -- Time in hours for tank to be empty
  let equation := net_rate = C / time_empty in
  C = 6337.54 -- The capacity of the tank is approximately 6337.54 litres.

theorem solve_tank_capacity : tank_capacity = 6337.54 := by
  sorry

end solve_tank_capacity_l216_216097


namespace total_shaded_area_l216_216154

/-- 
Given a 6-foot by 12-foot floor tiled with 1-foot by 1-foot tiles,
where each tile has four white quarter circles of radius 1/3 foot at its corners,
prove that the total shaded area of the floor is 72 - 8π square feet.
-/
theorem total_shaded_area :
  let floor_length := 6
  let floor_width := 12
  let tile_size := 1
  let radius := 1 / 3
  let area_of_tile := tile_size * tile_size
  let white_area_per_tile := (Real.pi * radius^2 / 4) * 4
  let shaded_area_per_tile := area_of_tile - white_area_per_tile
  let number_of_tiles := floor_length * floor_width
  let total_shaded_area := number_of_tiles * shaded_area_per_tile
  total_shaded_area = 72 - 8 * Real.pi :=
by
  let floor_length := 6
  let floor_width := 12
  let tile_size := 1
  let radius := 1 / 3
  let area_of_tile := tile_size * tile_size
  let white_area_per_tile := (Real.pi * radius^2 / 4) * 4
  let shaded_area_per_tile := area_of_tile - white_area_per_tile
  let number_of_tiles := floor_length * floor_width
  let total_shaded_area := number_of_tiles * shaded_area_per_tile
  sorry

end total_shaded_area_l216_216154


namespace line_segments_relation_l216_216536

-- Define the points and line segments lengths
def num_points_a : ℕ := 10
def num_points_b : ℕ := 100
def len_a : ℝ := a
def len_b : ℝ := b

-- Define the problem statement
theorem line_segments_relation (a b : ℝ) 
(H1 : num_points_a = 10) 
(H2 : num_points_b = 100) 
(H3 : len_a = a) 
(H4 : len_b = b) :
b = 11 * a := 
sorry

end line_segments_relation_l216_216536


namespace problem_remainder_l216_216126

theorem problem_remainder :
  ((12095 + 12097 + 12099 + 12101 + 12103 + 12105 + 12107) % 10) = 7 := by
  sorry

end problem_remainder_l216_216126


namespace bike_speed_l216_216398

theorem bike_speed (distance time : ℕ) (h_distance : distance = 350) (h_time : time = 7) : distance / time = 50 :=
by
  rw [h_distance, h_time]
  norm_num
  sorry

end bike_speed_l216_216398


namespace total_apples_correctness_l216_216390

-- Define the number of apples each man bought
def applesMen := 30

-- Define the number of apples each woman bought
def applesWomen := applesMen + 20

-- Define the total number of apples bought by the two men
def totalApplesMen := 2 * applesMen

-- Define the total number of apples bought by the three women
def totalApplesWomen := 3 * applesWomen

-- Define the total number of apples bought by the two men and three women
def totalApples := totalApplesMen + totalApplesWomen

-- Prove that the total number of apples bought by two men and three women is 210
theorem total_apples_correctness : totalApples = 210 := by
  sorry

end total_apples_correctness_l216_216390


namespace malcolm_followers_l216_216038

theorem malcolm_followers :
  let instagram_followers := 240
  let facebook_followers := 500
  let twitter_followers := (instagram_followers + facebook_followers) / 2
  let tiktok_followers := 3 * twitter_followers
  let youtube_followers := tiktok_followers + 510
  instagram_followers + facebook_followers + twitter_followers + tiktok_followers + youtube_followers = 3840 :=
by {
  sorry
}

end malcolm_followers_l216_216038


namespace evaluate_exponent_l216_216278

theorem evaluate_exponent : (3^2)^4 = 6561 := sorry

end evaluate_exponent_l216_216278


namespace at_least_one_of_p_or_q_true_l216_216741

variable (p q : Prop)

theorem at_least_one_of_p_or_q_true (h : ¬(p ∨ q) = false) : p ∨ q :=
by 
  sorry

end at_least_one_of_p_or_q_true_l216_216741


namespace complex_quadrant_l216_216856

open Complex

theorem complex_quadrant (z : ℂ) (h : z = (2 - I) / (2 + I)) : 
  z.re > 0 ∧ z.im < 0 := 
by
  sorry

end complex_quadrant_l216_216856


namespace roots_equality_l216_216664

variable {α β p q : ℝ}

theorem roots_equality (h1 : α ≠ β)
    (h2 : α * α + p * α + q = 0 ∧ β * β + p * β + q = 0)
    (h3 : α^3 - α^2 * β - α * β^2 + β^3 = 0) : 
  p = 0 ∧ q < 0 :=
by 
  sorry

end roots_equality_l216_216664


namespace D_is_necessary_but_not_sufficient_condition_for_A_l216_216033

variable (A B C D : Prop)

-- Conditions
axiom A_implies_B : A → B
axiom not_B_implies_A : ¬ (B → A)
axiom B_iff_C : B ↔ C
axiom C_implies_D : C → D
axiom not_D_implies_C : ¬ (D → C)

theorem D_is_necessary_but_not_sufficient_condition_for_A : (A → D) ∧ ¬ (D → A) :=
by sorry

end D_is_necessary_but_not_sufficient_condition_for_A_l216_216033


namespace proof_2_abs_a_plus_b_less_abs_4_plus_ab_l216_216347

theorem proof_2_abs_a_plus_b_less_abs_4_plus_ab (a b : ℝ) (h1 : abs a < 2) (h2 : abs b < 2) :
    2 * abs (a + b) < abs (4 + a * b) := 
by
  sorry

end proof_2_abs_a_plus_b_less_abs_4_plus_ab_l216_216347


namespace cricket_team_problem_l216_216069

theorem cricket_team_problem :
  let members := 11
  let avg_age := 27
  let wicket_keeper_age := 30
  let age_of_remaining_players_excluding_two := avg_age - 1
  let total_age := members * avg_age
  let T := total_age
  (T - wicket_keeper_age <+> P) = 9 * age_of_remaining_players_excluding_two -> avg_age_of_remaining_players := 26 :=
by
  sorry

end cricket_team_problem_l216_216069


namespace estimated_population_correct_correlation_coefficient_correct_l216_216207

variables (x y : ℕ → ℝ)
variables (n : ℕ) (plots : ℕ)

noncomputable def average_y := (∑ i in finset.range 20, y i) / 20

noncomputable def estimated_population := average_y y * plots

theorem estimated_population_correct (h_sum_y : ∑ i in finset.range 20, y i = 1200)
                                      (h_plots : plots = 200) :
  estimated_population y plots = 12000 :=
by sorry

noncomputable def correlation_coefficient (x y : ℕ → ℝ) :=
  let sum_xy := ∑ i in finset.range 20, (x i - ((∑ j in finset.range 20, x j) / 20)) * 
                               (y i - ((∑ j in finset.range 20, y j) / 20)) in
  let sum_xx := ∑ i in finset.range 20, (x i - ((∑ j in finset.range 20, x j) / 20)) ^ 2 in
  let sum_yy := ∑ i in finset.range 20, (y i - ((∑ j in finset.range 20, y j) / 20)) ^ 2 in
  sum_xy / (real.sqrt (sum_xx * sum_yy))

theorem correlation_coefficient_correct (h_sum_xx : ∑ i in finset.range 20, (x i - ((∑ j in finset.range 20, x j) / 20)) ^ 2 = 80)
                                        (h_sum_yy : ∑ i in finset.range 20, (y i - ((∑ j in finset.range 20, y j) / 20)) ^ 2 = 9000)
                                        (h_sum_xy : ∑ i in finset.range 20, (x i - ((∑ j in finset.range 20, x j) / 20)) * (y i - ((∑ j in finset.range 20, y j) / 20)) = 800) :
  correlation_coefficient x y ≈ 0.94 :=
by sorry

end estimated_population_correct_correlation_coefficient_correct_l216_216207


namespace larger_number_l216_216114

theorem larger_number (x y : ℕ) (h₁ : x + y = 27) (h₂ : x - y = 5) : x = 16 :=
by sorry

end larger_number_l216_216114


namespace ratio_BZ_ZC_eq_one_l216_216005

variables {A B C P X Y Z : Type} [EuclideanGeometry]
variables {AC AB BC BZ ZC : Length} (AC_gt_AB : AC > AB)
variables (P_perp_bisector_BC : is_perpendicular_bisector BC P)
variables (P_angle_bisector_A : is_angle_bisector A P)
variables (PX_perp_AB : is_perpendicular PX AB) 
variables (PY_perp_AC : is_perpendicular PY AC)
variables (Z_intersection_XY_BC : is_intersection XY BC Z)

theorem ratio_BZ_ZC_eq_one 
  (h1 : AC_gt_AB) 
  (h2 : P_perp_bisector_BC) 
  (h3 : P_angle_bisector_A)
  (h4 : PX_perp_AB) 
  (h5 : PY_perp_AC) 
  (h6 : Z_intersection_XY_BC) :
  BZ / ZC = 1 := 
sorry

end ratio_BZ_ZC_eq_one_l216_216005


namespace complex_power_identity_l216_216285

-- Define the imaginary unit i and its property
def i : ℂ := Complex.I

-- Proposition to prove the main statement
theorem complex_power_identity : (Complex.ofReal (1 : ℂ) - i / Complex.sqrt 2) ^ 32 = 1 :=
by
  -- Insert proof steps here
  sorry

end complex_power_identity_l216_216285


namespace estimated_population_correct_correlation_coefficient_correct_l216_216223

-- Definitions based on conditions
def total_plots := 200
def sample_plots := 20
def sum_x := 60
def sum_y := 1200
def sum_x_squared_diff := 80
def sum_y_squared_diff := 9000
def sum_xy_diff := 800

-- Estimated population calculation
noncomputable def estimated_population : ℕ :=
  (sum_y / sample_plots) * total_plots

theorem estimated_population_correct : estimated_population = 12000 := sorry

-- Correlation coefficient calculation
noncomputable def correlation_coefficient : Real :=
  sum_xy_diff / Real.sqrt (sum_x_squared_diff * sum_y_squared_diff)

theorem correlation_coefficient_correct : 
  Real.round (correlation_coefficient * 100) / 100 = 0.94 := sorry

end estimated_population_correct_correlation_coefficient_correct_l216_216223


namespace find_larger_number_l216_216111

variable (x y : ℝ)
axiom h1 : x + y = 27
axiom h2 : x - y = 5

theorem find_larger_number : x = 16 :=
by
  sorry

end find_larger_number_l216_216111


namespace max_difference_two_digit_numbers_l216_216537

theorem max_difference_two_digit_numbers (A B : ℤ) (hA : 10 ≤ A ∧ A ≤ 99) (hB : 10 ≤ B ∧ B ≤ 99) (h : 2 * A * 3 = 2 * B * 7) : 
  56 ≤ A - B :=
sorry

end max_difference_two_digit_numbers_l216_216537


namespace cookies_with_six_cups_l216_216791

theorem cookies_with_six_cups (cookies_with_four_cups : ℕ) (flour_with_four_cups : ℕ) (flour_with_six_cups : ℕ)
  (h : cookies_with_four_cups = 24 ∧ flour_with_four_cups = 4 ∧ flour_with_six_cups = 6) :
  cookies_with_four_cups / flour_with_four_cups * flour_with_six_cups = 36 :=
by
  obtain ⟨h1, h2, h3⟩ := h
  rw [h1, h2, h3]
  norm_num
  sorry

end cookies_with_six_cups_l216_216791


namespace product_of_values_l216_216726

theorem product_of_values (x : ℚ) (hx : abs ((18 / x) + 4) = 3) :
  x = -18 ∨ x = -18 / 7 ∧ -18 * (-18 / 7) = 324 / 7 :=
by sorry

end product_of_values_l216_216726


namespace cosine_BHD_l216_216751

-- Let DHG and FHB be angles in a rectangular solid such that
-- ∠DHG = 60° and ∠FHB = 45°
noncomputable def angle_DHG : ℝ := 60
noncomputable def angle_FHB : ℝ := 45

-- Prove the cosine of angle BHD is 5√3/12 given the above conditions
theorem cosine_BHD (angle_DHG angle_FHB : ℝ) (h_DHG : angle_DHG = 60) (h_FHB : angle_FHB = 45) :
  cos (angle_BHD angle_DHG angle_FHB) = 5 * real.sqrt 3 / 12 := 
sorry

end cosine_BHD_l216_216751


namespace algebraic_sum_of_coefficients_l216_216032

noncomputable def u (n : ℕ) : ℕ :=
if n = 0 then 0 -- Placeholder for u_0 which should not be used
else (2 * (n - 1)^2 - 3 * (n - 1) + 6)

theorem algebraic_sum_of_coefficients :
  (∃ a b c : ℤ, ∀ n : ℕ, n > 0 → u n = a * n^2 + b * n + c) ∧ (let a := 2 in let b := -3 in let c := 6 in a + b + c = 5) :=
begin
  split,
  { use [2, -3, 6],
    intros n hn,
    dsimp [u],
    split_ifs,
    ring, },
  { rintros ⟨a, b, c⟩,
    have h1 : a = 2 := by sorry,
    have h2 : b = -3 := by sorry,
    have h3 : c = 6 := by sorry,
    dsimp,
    rw [h1, h2, h3],
    ring, }
end

end algebraic_sum_of_coefficients_l216_216032


namespace andy_l216_216983

theorem andy's_profit_per_cake :
  (∀ (cakes : ℕ), cakes = 2 → ∀ (ingredient_cost : ℕ), ingredient_cost = 12 →
                  ∀ (packaging_cost_per_cake : ℕ), packaging_cost_per_cake = 1 →
                  ∀ (selling_price_per_cake : ℕ), selling_price_per_cake = 15 →
                  ∀ (profit_per_cake : ℕ), profit_per_cake = selling_price_per_cake - (ingredient_cost / cakes + packaging_cost_per_cake) →
                    profit_per_cake = 8) :=
by
  sorry

end andy_l216_216983


namespace walker_rate_l216_216193

theorem walker_rate (W : ℝ) :
  (∀ t : ℝ, t = 5 / 60 ∧ t = 20 / 60 → 20 * t = (5 * 20 / 3) ∧ W * (1 / 3) = 5 / 3) →
  W = 5 :=
by
  sorry

end walker_rate_l216_216193


namespace intersecting_lines_c_plus_d_l216_216867

theorem intersecting_lines_c_plus_d (c d : ℝ) 
  (h1 : ∀ y, ∃ x, x = (1/3) * y + c) 
  (h2 : ∀ x, ∃ y, y = (1/3) * x + d)
  (P : (3:ℝ) = (1 / 3) * (3:ℝ) + c) 
  (Q : (3:ℝ) = (1 / 3) * (3:ℝ) + d) : 
  c + d = 4 := 
by
  sorry

end intersecting_lines_c_plus_d_l216_216867


namespace subset_relationship_l216_216374

def S : Set ℕ := {x | ∃ n : ℕ, x = 3^n}
def T : Set ℕ := {x | ∃ n : ℕ, x = 3 * n}

theorem subset_relationship : S ⊆ T :=
by sorry

end subset_relationship_l216_216374


namespace least_five_digit_congruent_to_5_mod_15_l216_216906

theorem least_five_digit_congruent_to_5_mod_15 : ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 15 = 5 ∧ n = 10010 := by
  sorry

end least_five_digit_congruent_to_5_mod_15_l216_216906


namespace waiter_tables_l216_216192

theorem waiter_tables (w m : ℝ) (avg_customers_per_table : ℝ) (total_customers : ℝ) (t : ℝ)
  (hw : w = 7.0)
  (hm : m = 3.0)
  (havg : avg_customers_per_table = 1.111111111)
  (htotal : total_customers = w + m)
  (ht : t = total_customers / avg_customers_per_table) :
  t = 90 :=
by
  -- Proof would be inserted here
  sorry

end waiter_tables_l216_216192


namespace thirty_first_number_in_pascal_thirty_second_row_l216_216549

theorem thirty_first_number_in_pascal_thirty_second_row : Nat.choose 32 30 = 496 := by sorry

end thirty_first_number_in_pascal_thirty_second_row_l216_216549


namespace smallest_possible_domain_count_l216_216849

-- We define the function g according to the provided rules.
noncomputable def g : ℤ → ℤ
| d := if even d then 2 * d + 2 else (d - 1) / 2

theorem smallest_possible_domain_count :
  ∃ n : ℕ, n = 16 ∧ (∀ (x : ℤ), x ∈ (range g)) := sorry

end smallest_possible_domain_count_l216_216849


namespace convex_quadrilateral_in_circle_l216_216808

theorem convex_quadrilateral_in_circle
  (A B C D P Q R S T U : Point)
  (Γ : Circle)
  (hABC: ConvexQuadrilateral A B C D )
  (hInscribed: InscribedInCircle A B C D Γ)
  (hParallel1: ParallelThrough BC D intersects CA at P ∧ intersects AB at Q ∧ intersectsCircleAgain Γ R)
  (hParallel2: ParallelThrough AB D intersects CA at S ∧ intersects BC at T ∧ intersectsCircleAgain Γ U)
  (hMidpoint: Segment P Q = Segment Q R)
  : Segment S T = Segment T U :=
sorry

end convex_quadrilateral_in_circle_l216_216808


namespace matching_color_beads_exists_l216_216539

noncomputable def can_rotate_to_match_colors (circle1 circle2 circle3 : list ℕ) : Prop :=
  ∃ k1 k2, 
    (0 ≤ k1 ∧ k1 < 20) ∧
    (0 ≤ k2 ∧ k2 < 20) ∧
    (∃ (i : ℕ), 
      (0 ≤ i ∧ i < 20) ∧ 
      ((circle1[(i + k1) % 20] = circle2[(i + k2) % 20]) ∧
       (circle2[(i + k1) % 20] = circle3[(i + k2) % 20]))) 

theorem matching_color_beads_exists :
  ∀ (circle1 circle2 circle3 : list ℕ),
    circle1.length = 20 → 
    circle2.length = 20 → 
    circle3.length = 20 → 
    (∃ i, (circle1.count 0 = 10) ∧ (circle2.count 0 = 10) ∧ (circle3.count 0 = 10) ∧
            (circle1.count 1 = 10) ∧ (circle2.count 1 = 10) ∧ (circle3.count 1 = 10)) →
    can_rotate_to_match_colors circle1 circle2 circle3 := 
by
  intros,
  sorry

end matching_color_beads_exists_l216_216539


namespace number_of_boys_is_92_l216_216095

-- Definitions based on conditions
def total_students : ℝ := 1150
def boys (x : ℝ) : ℝ := x
def girls (x : ℝ) : ℝ := (x / 100) * total_students

-- Theorem statement proving the number of boys
theorem number_of_boys_is_92 : ∃ x : ℝ, boys x + girls x = total_students ∧ x = 92 :=
by
  sorry

end number_of_boys_is_92_l216_216095


namespace monthly_growth_rate_price_reduction_l216_216790

theorem monthly_growth_rate
  (initial_sales : ℕ) (final_sales : ℕ) (months : ℕ)
  (initial_sales = 192) (final_sales = 300) (months = 2) :
  ∃ x : ℚ, (1 + x) ^ months = final_sales / initial_sales := by
  sorry

theorem price_reduction
  (initial_sales : ℕ) (final_sales : ℕ) (profit_target : ℚ) (price_reduction : ℚ)
  (initial_price : ℚ) (cost_per_item : ℚ) (initial_profit_per_item : ℚ)
  (increase_in_sales_per_reduction : ℕ) (price_reduction_increment : ℚ)
  (initial_sales = 300) (profit_target = 6080) (initial_price = 60)
  (cost_per_item = 40) (initial_profit_per_item = initial_price - cost_per_item)
  (increase_in_sales_per_reduction = 40) (price_reduction_increment = 2) :
  ∃ m : ℚ, (initial_profit_per_item - m) * (initial_sales + (increase_in_sales_per_reduction / price_reduction_increment) * m) = profit_target := by
  sorry

end monthly_growth_rate_price_reduction_l216_216790


namespace work_rate_proof_l216_216136

def combined_rate (a b c : ℚ) : ℚ := a + b + c

def inv (x : ℚ) : ℚ := 1 / x

theorem work_rate_proof (A B C : ℚ) (h₁ : A + B = 1/15) (h₂ : C = 1/10) :
  inv (combined_rate A B C) = 6 :=
by
  sorry

end work_rate_proof_l216_216136


namespace total_amount_paid_l216_216380

theorem total_amount_paid : 
  let spam_cost := 3
  let peanut_butter_cost := 5
  let bread_cost := 2
  let spam_quantity := 12
  let peanut_butter_quantity := 3
  let bread_quantity := 4
  let spam_total := spam_cost * spam_quantity
  let peanut_butter_total := peanut_butter_cost * peanut_butter_quantity
  let bread_total := bread_cost * bread_quantity
  let total_amount := spam_total + peanut_butter_total + bread_total in
  total_amount = 59 :=
by
  sorry

end total_amount_paid_l216_216380


namespace rectangle_diagonals_locus_is_circle_l216_216655

-- Define the points and midpoint properties
structure Point (α : Type*) :=
  (x : α)
  (y : α)

variables (α : Type*) [field α]

def midpoint (A B : Point α) : Point α :=
{ x := (A.x + B.x) / 2,
  y := (A.y + B.y) / 2 }

variables (M N L K : Point α)

def P : Point α := midpoint α M N
def Q : Point α := midpoint α K L

-- Define the condition where the intersection point of the diagonals lies on the circle with diameter PQ
def lies_on_circle_with_diameter (O : Point α) (P Q : Point α) : Prop :=
  (O.x - P.x) * (O.x - Q.x) + (O.y - P.y) * (O.y - Q.y) = 0


theorem rectangle_diagonals_locus_is_circle : ∀ (O : Point α), lies_on_circle_with_diameter O P Q :=
sorry -- Proof to be provided

end rectangle_diagonals_locus_is_circle_l216_216655


namespace simplify_and_evaluate_l216_216486

theorem simplify_and_evaluate :
  let a := Real.sqrt 2 - 1 in 
  2 * (a + Real.sqrt 3) * (a - Real.sqrt 3) - a * (a - Real.sqrt 2) + 6 = 5 - 3 * Real.sqrt 2 :=
by 
  let a := Real.sqrt 2 - 1 
  sorry

end simplify_and_evaluate_l216_216486


namespace proof_pyramid_l216_216089

noncomputable def lateral_surface_area_and_volume_of_pyramid (a : ℝ) (h : ℝ) (SK : ℝ) 
  (base_area : ℝ) (diagonal_area : ℝ) (volume : ℝ) : 
  Prop :=
  (diagonal_area = base_area) → 
  (h = a * Real.sqrt 2) → 
  (SK = 3 * a / 2) → 
  (base_area = a^2) → 
  (volume = (a^3 * Real.sqrt 2) / 3) → 
  (3 * base_area = 3 * a^2) ∧ 
  (volume = (a^3 * Real.sqrt 2) / 3)

theorem proof_pyramid (a : ℝ) 
  (h := a * Real.sqrt 2) 
  (SK := 3 * a / 2) 
  (base_area := a^2) 
  (diagonal_area := a^2) 
  (volume := (a^3 * Real.sqrt 2) / 3) :
  lateral_surface_area_and_volume_of_pyramid a h SK base_area diagonal_area volume :=
by
  intros 
  split
  · -- Proof for the lateral surface area
    sorry
  · -- Proof for the volume
    sorry

end proof_pyramid_l216_216089


namespace two_sum_fourth_power_square_l216_216875

-- Define the condition
def sum_zero (x y z : ℤ) : Prop := x + y + z = 0

-- The theorem to be proven
theorem two_sum_fourth_power_square (x y z : ℤ) (h : sum_zero x y z) : ∃ k : ℤ, 2 * (x^4 + y^4 + z^4) = k^2 :=
by
  -- skipping the proof
  sorry

end two_sum_fourth_power_square_l216_216875


namespace largest_good_student_l216_216415

def is_good_group (group : Finset ℕ) : Prop :=
  ∃ a b ∈ group, a ∣ b ∧ a ≠ b

def is_good_student (n : ℕ) : Prop :=
  ∀ group : Finset ℕ, group.card = 500 → n ∉ group → ¬ is_good_group group

theorem largest_good_student : ∀ n ∈ (Finset.range 1001), is_good_student n → n ≤ 333 := 
sorry

end largest_good_student_l216_216415


namespace pump_fills_tank_without_leak_l216_216953

variable (T : ℝ)
-- Condition: The effective rate with the leak is equal to the rate it takes for both to fill the tank.
def effective_rate_with_leak (T : ℝ) : Prop :=
  1 / T - 1 / 21 = 1 / 3.5

-- Conclude: the time it takes the pump to fill the tank without the leak
theorem pump_fills_tank_without_leak : effective_rate_with_leak T → T = 3 :=
by
  intro h
  sorry

end pump_fills_tank_without_leak_l216_216953


namespace divide_L_shape_into_four_equal_parts_l216_216785

-- Define the shape and properties of the "L" shaped ice plate
structure L_shape (a b : ℕ) :=
  (length_a : ℕ := a)
  (length_b : ℕ := b)

-- Define a function that checks if two "L" shapes are congruent
def is_congruent (L1 L2 : L_shape) : Prop :=
  L1.length_a = L2.length_a ∧ L1.length_b = L2.length_b

-- State the problem in Lean 4
theorem divide_L_shape_into_four_equal_parts (a b : ℕ) (L : L_shape a b) :
  ∃ L1 L2 L3 L4 : L_shape (a / 2) (b / 2), 
  is_congruent L1 L2 ∧ 
  is_congruent L2 L3 ∧ 
  is_congruent L3 L4 ∧ 
  is_congruent L4 L1 :=
sorry

end divide_L_shape_into_four_equal_parts_l216_216785


namespace jimmy_pizza_cost_per_slice_l216_216444

-- Definitions for pizza costs and sizes
def small_pizza_cost : ℝ := 8.00
def small_pizza_slices : ℕ := 6
def medium_pizza_cost : ℝ := 12.00
def medium_pizza_slices : ℕ := 10
def large_pizza_cost : ℝ := 15.00
def large_pizza_slices : ℕ := 12

-- Definitions for topping costs
def category_A_topping_cost : ℝ := 2.00
def category_B_topping_cost : ℝ := 1.00
def category_C_topping_cost : ℝ := 0.50

-- Jimmy's chosen toppings
def number_of_category_A_toppings : ℕ := 2
def number_of_category_B_toppings : ℕ := 3
def number_of_category_C_toppings : ℕ := 4

-- Calculate the total cost of the toppings
def total_toppings_cost : ℝ :=
  (number_of_category_A_toppings * category_A_topping_cost) +
  (number_of_category_B_toppings * category_B_topping_cost) +
  (number_of_category_C_toppings * category_C_topping_cost)

-- Calculate the total cost of the medium pizza with toppings
def total_pizza_cost : ℝ :=
  medium_pizza_cost + total_toppings_cost

-- Calculate the cost per slice of the medium pizza with toppings
def cost_per_slice : ℝ :=
  total_pizza_cost / medium_pizza_slices

-- The proof statement
theorem jimmy_pizza_cost_per_slice : cost_per_slice = 2.10 := by
  sorry

end jimmy_pizza_cost_per_slice_l216_216444


namespace volunteers_meet_again_in_360_days_l216_216234

open Nat

theorem volunteers_meet_again_in_360_days :
  ∃ (n : ℕ), n = Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 9) ∧ n = 360 := by
  sorry

end volunteers_meet_again_in_360_days_l216_216234


namespace work_completion_time_l216_216139

theorem work_completion_time (A B : Type) [linear_ordered_field A] [linear_ordered_field B] :
  (∃ x : ℚ, ∃ y : ℚ, 
      (A = 3 * x) ∧ 
      (B = x) ∧ 
      (1 / A + 1 / B = 1 / 12)) → A = 16 :=
sorry

end work_completion_time_l216_216139


namespace inequality_solution_set_l216_216151

theorem inequality_solution_set : 
  (∃ (x : ℝ), (4 / (x - 1) ≤ x - 1) ↔ (x ≥ 3 ∨ (-1 ≤ x ∧ x < 1))) :=
by
  sorry

end inequality_solution_set_l216_216151


namespace count_valid_unordered_pairs_l216_216458

def E (n : ℕ) : set ℕ := {x | 1 ≤ x ∧ x ≤ n}

def valid_pairs {n : ℕ} (A B : set ℕ) : Prop :=
  (A ∪ B ⊆ E n) ∧ (A ∩ B = ∅) ∧ (A ≠ ∅) ∧ (B ≠ ∅)

theorem count_valid_unordered_pairs (n : ℕ) :
  ∃ k, k = (3^n - 2^n - 1) / 2 ∧ (∀ (A B : set ℕ), valid_pairs A B ↔ A ∪ B ∈ k) :=
sorry

end count_valid_unordered_pairs_l216_216458


namespace seq_positive_integers_no_m_exists_l216_216677

-- Definition of the sequence
def seq (n : ℕ) : ℕ :=
  Nat.recOn n
    1
    (λ n a_n => 3 * a_n + 2 * (2 * a_n * a_n - 1).sqrt)

-- Axiomatize the properties involved in the recurrence relation
axiom rec_sqrt_property (n : ℕ) : ∃ k : ℕ, (2 * seq n * seq n - 1) = k * k

-- Proof statement for the sequence of positive integers
theorem seq_positive_integers (n : ℕ) : seq n > 0 := sorry

-- Proof statement for non-existence of m such that 2015 divides seq(m)
theorem no_m_exists (m : ℕ) : ¬ (2015 ∣ seq m) := sorry

end seq_positive_integers_no_m_exists_l216_216677


namespace find_ab_sum_l216_216692

def axb_value (a b : ℝ) (f : ℝ → ℝ) : Prop :=
  f = λ x, a * x ^ 2 + b * x + 1 ∧ 
  ∀ x ∈ set.Icc (-2 * a) (a ^ 2 - 3), f x = f (-x)

theorem find_ab_sum (a b : ℝ) (f : ℝ → ℝ) (hab : axb_value a b f) : a + b = 3 :=
sorry

end find_ab_sum_l216_216692


namespace k_value_l216_216635

def f (x : ℝ) : ℝ := 5 * x^2 - 1 / x + 3 * x
def g (x : ℝ) (k : ℝ) : ℝ := x^2 - k

theorem k_value : ∃ (k : ℝ), f 3 - g 3 k = 10 ∧ k = -104 / 3 :=
by sorry

end k_value_l216_216635


namespace sum_of_possible_values_for_a_l216_216036

-- Define the conditions
variables (a b c d : ℤ)
variables (h1 : a > b) (h2 : b > c) (h3 : c > d)
variables (h4 : a + b + c + d = 52)
variables (differences : finset ℤ)

-- Hypotheses about the pairwise differences
variable (h_diff : differences = {2, 3, 5, 6, 8, 11})
variable (h_ad : a - d = 11)

-- The pairs of differences adding up to 11
variable (h_pairs1 : a - b + b - d = 11)
variable (h_pairs2 : a - c + c - d = 11)

-- The theorem to be proved
theorem sum_of_possible_values_for_a : a = 19 :=
by
-- Implemented variables and conditions correctly, and the proof is outlined.
sorry

end sum_of_possible_values_for_a_l216_216036


namespace exists_equal_partition_of_coin_set_l216_216538

theorem exists_equal_partition_of_coin_set (n : ℕ) (h_odd : n % 2 = 1) (h_ge7 : n ≥ 7) :
  ∃ (weights : Fin n → ℕ), 
    (∀ k : Fin n, ∃ (A B : Fin n → ℕ), A ∪ B = {x | x ∈ weights.val ∧ x ≠ weights k} ∧ 
    A ∩ B = ∅ ∧ (A.sum id) = (B.sum id)) := sorry

end exists_equal_partition_of_coin_set_l216_216538


namespace morgan_gpa_probability_l216_216242

def points (grade : String) : ℝ :=
  if grade = "A" then 4
  else if grade = "B" then 3
  else if grade = "C" then 2
  else if grade = "D" then 1
  else 0

def gpa (grades : List String) : ℝ :=
  (grades.map points).sum / grades.length

def probability_of_grades (subject : String) (grade : String) : ℝ :=
  if subject = "English" then
    if grade = "A" then 1/3
    else if grade = "B" then 1/5
    else if grade = "C" then 7/15
    else 0
  else if subject = "Geology" then
    if grade = "A" then 1/5
    else if grade = "B" then 1/2
    else if grade = "C" then 3/10
    else 0
  else 0

theorem morgan_gpa_probability :
  (probability_of_grades "English" "A" * (probability_of_grades "Geology" "B" +
    probability_of_grades "Geology" "C")) +
  (probability_of_grades "Geology" "A" * (probability_of_grades "English" "B" +
    probability_of_grades "English" "C")) =
  11 / 30 := sorry

end morgan_gpa_probability_l216_216242


namespace largest_root_is_sqrt6_l216_216103

theorem largest_root_is_sqrt6 (p q r : ℝ) 
  (h1 : p + q + r = 3) 
  (h2 : p * q + p * r + q * r = -6) 
  (h3 : p * q * r = -18) : 
  max p (max q r) = Real.sqrt 6 := 
sorry

end largest_root_is_sqrt6_l216_216103


namespace smallest_positive_integer_for_divisibility_l216_216554

def is_divisible_by (a b : ℕ) : Prop :=
  ∃ k, a = b * k

def smallest_n (n : ℕ) : Prop :=
  (is_divisible_by (n^2) 50) ∧ (is_divisible_by (n^3) 288) ∧ (∀ m : ℕ, m > 0 → m < n → ¬ (is_divisible_by (m^2) 50 ∧ is_divisible_by (m^3) 288))

theorem smallest_positive_integer_for_divisibility : smallest_n 60 :=
by
  sorry

end smallest_positive_integer_for_divisibility_l216_216554


namespace OH_perp_MN_l216_216440

variable {A B C O H D E F M N : Point}
variable [Triangle ABC]

def circumcenter (T : Triangle) : Point := O
def orthocenter (T : Triangle) : Point := H
def is_altitude (A B C D : Point) : Prop := 
  ∃ (P Q : Line), P.line_through A B ∧ Q.line_through A C ∧ D ∈ (P ∩ Q)

def intersect_at (L1 L2 : Line) (P : Point) : Prop :=
  P ∈ L1 ∧ P ∈ L2

def perpendicular (L1 L2 : Line) : Prop :=
  let ⟨a, b⟩ := L1.direction in
  let ⟨c, d⟩ := L2.direction in
  a * c + b * d = 0

theorem OH_perp_MN 
  (h_oc : circumcenter (triangle A B C) = O)
  (h_orth : orthocenter (triangle A B C) = H)
  (h_ad : is_altitude A B C D)
  (h_be : is_altitude B A C E)
  (h_cf : is_altitude C A B F)
  (h_ed : intersect_at (line_through E D) (line_through M A B))
  (h_fd : intersect_at (line_through F D) (line_through N A C))
  : perpendicular (line_through O H) (line_through M N) := 
sorry

end OH_perp_MN_l216_216440


namespace sequence_sum_l216_216338

noncomputable def sequence (n : ℕ) : ℕ := sorry

theorem sequence_sum :
  (∀ n, sequence n * sequence (n+1) * sequence (n+2) * sequence (n+3) = 24)
  ∧ sequence 1 = 1 ∧ sequence 2 = 2 ∧ sequence 3 = 3
  → (Finset.range 2013).sum (λ n, sequence (n + 1)) = 5031 := sorry


end sequence_sum_l216_216338


namespace area_is_correct_l216_216246

def vertices : List (ℝ × ℝ) := [(1,1), (4,1), (6,3), (4,5), (2,4)]

def area_of_polygon (vertices : List (ℝ × ℝ)) : ℝ :=
  let n := vertices.length
  let shifted_vertices := vertices.append [vertices.head!]
  let products := List.map₂ (· * ·) (List.zipWith (·.1) shifted_vertices.tailOrNil) (List.zipWith (·.2) shifted_vertices)
  let products_rev := List.map₂ (· * ·) (List.zipWith (·.2) shifted_vertices.tailOrNil) (List.zipWith (·.1) shifted_vertices)
  let summed := (products.sum - products_rev.sum).abs
  summed / 2

theorem area_is_correct :
  area_of_polygon vertices = 12.5 := by
  sorry

end area_is_correct_l216_216246


namespace meghan_weight_proof_l216_216578

noncomputable def meghan_weight_base10 (b : ℕ) : ℕ := 4 * b^2 + 5 * b + 1

theorem meghan_weight_proof :
  ∃ b : ℕ, (4 * b^2 + 5 * b + 1 = 175) ∧ (4 * b^2 + 4 * b + 7 = 175) :=
begin
  use 6,
  split;
  sorry
end

end meghan_weight_proof_l216_216578


namespace p_sufficient_but_not_necessary_for_q_l216_216371

def p (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 5
def q (x : ℝ) : Prop := (x - 5) * (x + 1) < 0

theorem p_sufficient_but_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ∃ x : ℝ, q x ∧ ¬ p x :=
by
  sorry

end p_sufficient_but_not_necessary_for_q_l216_216371


namespace floor_sum_equality_l216_216478

theorem floor_sum_equality (n : ℕ) (h : n > 1) :
  (∑ i in Finset.range n, Nat.floor (n^(1 / (i + 1) : ℝ))) =
  (∑ i in Finset.range n, Nat.floor (Real.logBase (i + 2) n)) :=
sorry

end floor_sum_equality_l216_216478


namespace fly_distance_to_ceiling_l216_216165

def shortest_distance_to_ceiling (x y z d : ℝ) : Prop :=
  d = real.sqrt (x^2 + y^2 + z^2)

theorem fly_distance_to_ceiling :
  shortest_distance_to_ceiling 3 2 z 6 → z = real.sqrt 23 :=
by
  assume h : shortest_distance_to_ceiling 3 2 z 6
  have eq1 : real.sqrt (3^2 + 2^2 + z^2) = 6 := h
  have eq2: 3^2 + 2^2 + z^2 = 36 := by
    sorry -- Steps solving for z²=23 go here.
  exact sorry -- Proof that z = √23 follows logically.

end fly_distance_to_ceiling_l216_216165


namespace weight_of_D_l216_216772

theorem weight_of_D :
  ∃ (A B C D : ℤ), (A < B) ∧ (B < C) ∧ (C < D) ∧
  (A + B = 45) ∧ (A + C = 49) ∧ (C + D = 64) ∧ (B + D = 60) ∧ (D = 35) :=
begin
  sorry
end

end weight_of_D_l216_216772


namespace estimated_population_correct_correlation_coefficient_correct_l216_216220

-- Definitions based on conditions
def total_plots := 200
def sample_plots := 20
def sum_x := 60
def sum_y := 1200
def sum_x_squared_diff := 80
def sum_y_squared_diff := 9000
def sum_xy_diff := 800

-- Estimated population calculation
noncomputable def estimated_population : ℕ :=
  (sum_y / sample_plots) * total_plots

theorem estimated_population_correct : estimated_population = 12000 := sorry

-- Correlation coefficient calculation
noncomputable def correlation_coefficient : Real :=
  sum_xy_diff / Real.sqrt (sum_x_squared_diff * sum_y_squared_diff)

theorem correlation_coefficient_correct : 
  Real.round (correlation_coefficient * 100) / 100 = 0.94 := sorry

end estimated_population_correct_correlation_coefficient_correct_l216_216220


namespace amplitude_combined_wave_l216_216757

noncomputable def y1 (t : ℝ) : ℝ := 3 * Real.sqrt 2 * Real.sin (100 * Real.pi * t)
noncomputable def y2 (t : ℝ) : ℝ := 3 * Real.sin (100 * Real.pi * t - Real.pi / 4)
noncomputable def y (t : ℝ) : ℝ := y1 t + y2 t
noncomputable def amplitude : ℝ := 3 * Real.sqrt 5

theorem amplitude_combined_wave : ∀ t : ℝ, ∃ A : ℝ, A = 3 * Real.sqrt 5 :=
by
  intro t
  use amplitude
  exact sorry

end amplitude_combined_wave_l216_216757


namespace root_reciprocal_sum_l216_216799

noncomputable def polynomial : Polynomial ℝ :=
  Polynomial.X^4 + 8 * Polynomial.X^3 + 9 * Polynomial.X^2 + 5 * Polynomial.X + 4

open Polynomial

theorem root_reciprocal_sum :
  (∃ a b c d : ℝ, (is_root polynomial a) ∧ (is_root polynomial b) ∧ (is_root polynomial c) ∧ (is_root polynomial d)) →
  ∃ a b c d : ℝ, (∃ p : Polynomial ℝ, p.degree = 4 ∧ p.coeff 4 = 1 ∧
  p.coeff 3 = 8 ∧ p.coeff 2 = 9 ∧ p.coeff 1 = 5 ∧ p.coeff 0 = 4 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  is_root polynomial a ∧ is_root polynomial b ∧ is_root polynomial c ∧ is_root polynomial d) ∧
  ((1/(a*b) + 1/(a*c) + 1/(a*d) + 1/(b*c) + 1/(b*d) + 1/(c*d)) = 9/4) :=
by
  sorry

end root_reciprocal_sum_l216_216799


namespace B_pow_87_l216_216792

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ := !![ [0, -1, 0],
                                         [1,  0, 0],
                                         [0,  0, 0] ]

theorem B_pow_87 :
( B ^ 87 ) = !![ [0,  1, 0],
                 [-1, 0, 0],
                 [ 0, 0, 0] ] :=
sorry

end B_pow_87_l216_216792


namespace point_on_line_y_coordinate_l216_216967

variables (m b x : ℝ)

def line_equation := m * x + b

theorem point_on_line_y_coordinate : m = 4 → b = 4 → x = 199 → line_equation m b x = 800 :=
by 
  intros h_m h_b h_x
  unfold line_equation
  rw [h_m, h_b, h_x]
  norm_num
  done

end point_on_line_y_coordinate_l216_216967


namespace range_of_a_l216_216703

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2 * x + 1 + a * Real.log x

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = f a x2) ↔ (0 < a ∧ a < 1/2) := by
  sorry

end range_of_a_l216_216703


namespace acute_triangle_xy_lt_xz_obtuse_triangle_xy_not_lt_xz_l216_216680

variables {ABC : Triangle} -- Assuming Triangle is a defined type
variables {X Y Z : Point} -- Assuming Point is a defined type
variables {X1 X2 X3 Y1 Y2 Y3 Z1 Z2 Z3 : Point} -- Projections

-- Assuming projections function is defined and checks for the perpendicular foot from a point to a line.
-- Not defining projection function assuming it exists and is correct.

-- Given conditions for problem (a)
axiom acute_triangle (ABC : Triangle) : ABC.isAcute
axiom interior_points (X Y Z : Point) (ABC : Triangle) : X.isInInterior(ABC) ∧ Y.isInInterior(ABC) ∧ Z.isInInterior(ABC)
axiom projections_def (X Y Z : Point) (ABC : Triangle) : 
  (isProjection(X, ABC.sideBC, X1) ∧ isProjection(X, ABC.sideCA, X2) ∧ isProjection(X, ABC.sideAB, X3)) ∧
  (isProjection(Y, ABC.sideBC, Y1) ∧ isProjection(Y, ABC.sideCA, Y2) ∧ isProjection(Y, ABC.sideAB, Y3)) ∧
  (isProjection(Z, ABC.sideBC, Z1) ∧ isProjection(Z, ABC.sideCA, Z2) ∧ isProjection(Z, ABC.sideAB, Z3))
axiom distance_condition (X Y Z : Point) : ∀ i ∈ {1, 2, 3}, distance(Xi, Yi) < distance(Xi, Zi)

-- Objective proving
theorem acute_triangle_xy_lt_xz (ABC : Triangle) (X Y Z : Point) :
  acute_triangle(ABC) → 
  interior_points(X, Y, Z, ABC) →
  projections_def(X, Y, Z, ABC) →
  distance_condition(X, Y, Z) →
  distance(X, Y) < distance(X, Z) :=
sorry

-- Given conditions for problem (b)
axiom obtuse_triangle (ABC : Triangle) : ABC.isObtuse

-- Objective proving
theorem obtuse_triangle_xy_not_lt_xz (ABC : Triangle) (X Y Z : Point) :
  obtuse_triangle(ABC) ∧ 
  interior_points(X, Y, Z, ABC) ∧
  projections_def(X, Y, Z, ABC) ∧
  distance_condition(X, Y, Z) →
  ¬ (distance(X, Y) < distance(X, Z)) :=
sorry

end acute_triangle_xy_lt_xz_obtuse_triangle_xy_not_lt_xz_l216_216680


namespace dogs_not_liking_any_of_three_l216_216749
open Set

variable (Dogs : Type) [Fintype Dogs]

-- Conditions
variable [∀ d : Dogs, Decidable (d ∈ ∅)]
def total_dogs : Finset Dogs := Finset.univ
def like_watermelon : Finset Dogs := {d ∈ total_dogs | d ∈ ∅}.filter (λ d, sorry) -- Placeholder filter that you will define yourself
def like_salmon : Finset Dogs := {d ∈ total_dogs | d ∈ ∅}.filter (λ d, sorry)
def like_chicken : Finset Dogs := {d ∈ total_dogs | d ∈ ∅}.filter (λ d, sorry)

axiom total_dogs_count : total_dogs.card = 100
axiom watermelon_dogs_count : like_watermelon.card = 20
axiom salmon_dogs_count : like_salmon.card = 70
axiom watermelon_and_salmon_dogs_count : (like_watermelon ∩ like_salmon).card = 10
axiom chicken_dogs_count : like_chicken.card = 15
axiom chicken_and_watermelon_dogs_count : (like_chicken ∩ like_watermelon).card = 5
axiom chicken_and_salmon_dogs_count : (like_chicken ∩ like_salmon).card = 8
axiom all_three_dogs_count : (like_watermelon ∩ like_salmon ∩ like_chicken).card = 3

-- Equivalent Proof Problem
theorem dogs_not_liking_any_of_three :
  (total_dogs.card - (like_watermelon ∪ like_salmon ∪ like_chicken).card) = 15 :=
by sorry

end dogs_not_liking_any_of_three_l216_216749


namespace min_value_16x_minus_4x_plus_1_l216_216304

theorem min_value_16x_minus_4x_plus_1 :
  ∀ x : ℝ, 16 ^ x - 4 ^ x + 1 ≥ 3 / 4 ∧ (∃ y : ℝ, 16 ^ y - 4 ^ y + 1 = 3 / 4) :=
by
  sorry

end min_value_16x_minus_4x_plus_1_l216_216304


namespace cuboid_inequalities_l216_216673

variable {x y z : ℝ}

def p (x y z : ℝ) : ℝ := 4 * (x + y + z)
def s (x y z : ℝ) : ℝ := 2 * (x * y + y * z + z * x)
def d (x y z : ℝ) : ℝ := Real.sqrt (x^2 + y^2 + z^2)
def alpha (x y z : ℝ) : ℝ := 1/3 * (1/4 * p x y z - Real.sqrt (d x y z ^ 2 - 1/2 * s x y z))
def beta (x y z : ℝ) : ℝ := 1/3 * (1/4 * p x y z + Real.sqrt (d x y z ^ 2 - 1/2 * s x y z))

theorem cuboid_inequalities (hx : x < y) (hy : y < z) : 
  x < alpha x y z ∧ z > beta x y z := 
sorry

end cuboid_inequalities_l216_216673


namespace smallest_n_such_that_a_n_equals_a_0_l216_216395

theorem smallest_n_such_that_a_n_equals_a_0 :
  let a_0 := Real.sin (π / 18)^2
  in ∀ n : ℕ, (∀ k : ℕ, (k > 0) → 
    let ak := Real.sin((π * 2^k) / 18)^2 
    in ak = a_0 → n = 12) := sorry

end smallest_n_such_that_a_n_equals_a_0_l216_216395


namespace larger_number_l216_216115

theorem larger_number (x y : ℕ) (h₁ : x + y = 27) (h₂ : x - y = 5) : x = 16 :=
by sorry

end larger_number_l216_216115


namespace max_m_value_inequality_abc_for_sum_l216_216344

-- Define the mathematical conditions and the proof problem.

theorem max_m_value (x m : ℝ) (h1 : |x - 2| - |x + 3| ≥ |m + 1|) :
  m ≤ 4 :=
sorry

theorem inequality_abc_for_sum (a b c : ℝ) (habc_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum_eq_M : a + 2 * b + c = 4) :
  (1 / (a + b)) + (1 / (b + c)) ≥ 1 :=
sorry

end max_m_value_inequality_abc_for_sum_l216_216344


namespace range_of_a_l216_216403

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x

theorem range_of_a (a : ℝ) :
  (∀ x y ∈ set.Icc (-2 : ℝ) (3 : ℝ), f x = a → f y = a → x = y) ∧
  (∃ x y ∈ set.Icc (-2 : ℝ) (3 : ℝ), f x = a ∧ f y = a ∧ x ≠ y) ↔
  a ∈ set.Icc (-2 : ℝ) 5 :=
sorry

end range_of_a_l216_216403


namespace division_quotient_remainder_l216_216085

theorem division_quotient_remainder :
  ∃ (q r : ℝ), 76.6 = 1.8 * q + r ∧ 0 ≤ r ∧ r < 1.8 ∧ q = 42 ∧ r = 1 := by
  sorry

end division_quotient_remainder_l216_216085


namespace average_weight_all_children_l216_216854

variable (avg_weight_boys avg_weight_girls total_avg_weight : ℝ)
variable (num_boys num_girls : ℕ)

def total_weight_boys := num_boys * avg_weight_boys
def total_weight_girls := num_girls * avg_weight_girls
def total_weight      := total_weight_boys + total_weight_girls

axiom avg_boys_weight : avg_weight_boys = 160
axiom avg_girls_weight : avg_weight_girls = 110
axiom boys_num : num_boys = 8
axiom girls_num : num_girls = 3

theorem average_weight_all_children :
  total_avg_weight = total_weight / (num_boys + num_girls) :=
by
  have tb_weight_eq : total_weight_boys = 8 * 160 := by rw [boys_num, avg_boys_weight]; simp
  have tg_weight_eq : total_weight_girls = 3 * 110 := by rw [girls_num, avg_girls_weight]; simp
  have tt_weight_eq : total_weight = (8 * 160) + (3 * 110) := by rw [tb_weight_eq, tg_weight_eq]; simp
  have all_children_num_eq : (num_boys + num_girls) = 11 := by rw [boys_num, girls_num]; simp
  calc total_avg_weight = total_weight / 11 : sorry

end average_weight_all_children_l216_216854


namespace volunteer_distribution_l216_216640

theorem volunteer_distribution :
  (∃ (volunteers : ℕ) (groups : ℕ) (group_sizes : ℕ → ℕ),
    volunteers = 5 ∧
    groups = 4 ∧ 
    group_sizes 0 = 2 ∧ 
    (∀ i, i ≠ 0 → group_sizes i = 1) ∧ 
    ((nat.choose 5 2) * 2^4 = 240)) :=
sorry

end volunteer_distribution_l216_216640


namespace abc_sum_is_14_l216_216088

noncomputable def sum_abc : ℕ :=
let n := 25 * 10^5 + A * 10^4 + B * 10^3 + 33 * 10 + C in
if (n % 8 = 0) ∧ (n % 9 = 0) ∧ (n % 11 = 0) then A + B + C else 0

theorem abc_sum_is_14 (A B C : ℕ) (hA : A < 10) (hB : B < 10) (hC : C < 10)
  (h : ∃ n, n = 25 * 10^5 + A * 10^4 + B * 10^3 + 3 * 10 + 3 + C ∧
              n % 8 = 0 ∧ n % 9 = 0 ∧ n % 11 = 0) :
  A + B + C = 14 :=
sorry

end abc_sum_is_14_l216_216088


namespace part_a_part_b_l216_216916

-- Part (a)
theorem part_a (n : ℕ) (h_n : n > 2) (table : array (2 × n) ℕ)
  (col_sums_different : ∀ i₁ i₂, i₁ ≠ i₂ → (∑ j, table!i₁!j) ≠ (∑ j, table!i₂!j)) :
  ∃ rearranged_table, (∀ i₁ i₂, i₁ ≠ i₂ → (∑ j, rearranged_table!i₁!j) ≠ (∑ j, rearranged_table!i₂!j)) ∧
  (∀ i₁ i₂, i₁ ≠ i₂ → (∑ j, rearranged_table!j!i₁) ≠ (∑ j, rearranged_table!j!i₂)) := 
sorry

-- Part (b)
theorem part_b (table : array (10 × 10) ℕ)
  (col_sums_different : ∀ i₁ i₂, i₁ ≠ i₂ → (∑ j, table!i₁!j) ≠ (∑ j, table!i₂!j)) :
  ¬(∃ rearranged_table, (∀ i₁ i₂, i₁ ≠ i₂ → (∑ j, rearranged_table!i₁!j) ≠ (∑ j, rearranged_table!i₂!j)) ∧
  (∀ i₁ i₂, i₁ ≠ i₂ → (∑ j, rearranged_table!j!i₁) ≠ (∑ j, rearranged_table!j!i₂))) := 
sorry

end part_a_part_b_l216_216916


namespace hexagon_shaded_area_l216_216988

-- Given conditions
variable (A B C D T : ℝ)
variable (h₁ : A = 2)
variable (h₂ : B = 3)
variable (h₃ : C = 4)
variable (h₄ : T = 20)
variable (h₅ : A + B + C + D = T)

-- The goal is to prove that the area of the shaded region (D) is 11 cm².
theorem hexagon_shaded_area : D = 11 := by
  sorry

end hexagon_shaded_area_l216_216988


namespace max_tan_angle_PAS_l216_216894

theorem max_tan_angle_PAS :
  ∀ (P Q R S : Type) (QR PA : ℝ),
    ∠ (Q : PQR) = 45 ∧
    QR = 6 ∧
    QS = 2 ∧
    RS = 4 →
    ∃ θ, θ = tangent θ ∧
    θ ≤ (sqrt 6 / 3) := by
  sorry

end max_tan_angle_PAS_l216_216894


namespace ratio_CE_EA_l216_216071

variables {A B C D E Z : Type} [AddGroup A] [AddGroup B] [AddGroup C]
variables [AddGroup D] [AddGroup E] [AddGroup Z]

-- Conditions
variables (ABC : Triangle A B C)
variables (D_midpoint : D = midpoint B C)
variables (E_on_CA : E ∈ line CA)
variables (AD_BE_meet_Z : is_intersection_point Z AD BE)
variables (Z_midpoint : Z = midpoint A D)

-- Theorem statement
theorem ratio_CE_EA : CE / EA = 2 :=
sorry

end ratio_CE_EA_l216_216071


namespace problem_l216_216874

noncomputable def a (n : ℕ) : ℚ := if n = 0 then 0 else (2^n - 1) / 2^(n-1)

def S (n : ℕ) : ℚ := (Finset.range n).sum (λ k, a (k + 1))

theorem problem :
  ∀ n : ℕ, n > 0 → S n = 2 * n - a n :=
by
  intros n hn
  induction n using Nat.case_strong_induction_on with
  | H0 => contradiction
  | H1 k ih => sorry

end problem_l216_216874


namespace estimated_population_correct_correlation_coefficient_correct_l216_216209

variables (x y : ℕ → ℝ)
variables (n : ℕ) (plots : ℕ)

noncomputable def average_y := (∑ i in finset.range 20, y i) / 20

noncomputable def estimated_population := average_y y * plots

theorem estimated_population_correct (h_sum_y : ∑ i in finset.range 20, y i = 1200)
                                      (h_plots : plots = 200) :
  estimated_population y plots = 12000 :=
by sorry

noncomputable def correlation_coefficient (x y : ℕ → ℝ) :=
  let sum_xy := ∑ i in finset.range 20, (x i - ((∑ j in finset.range 20, x j) / 20)) * 
                               (y i - ((∑ j in finset.range 20, y j) / 20)) in
  let sum_xx := ∑ i in finset.range 20, (x i - ((∑ j in finset.range 20, x j) / 20)) ^ 2 in
  let sum_yy := ∑ i in finset.range 20, (y i - ((∑ j in finset.range 20, y j) / 20)) ^ 2 in
  sum_xy / (real.sqrt (sum_xx * sum_yy))

theorem correlation_coefficient_correct (h_sum_xx : ∑ i in finset.range 20, (x i - ((∑ j in finset.range 20, x j) / 20)) ^ 2 = 80)
                                        (h_sum_yy : ∑ i in finset.range 20, (y i - ((∑ j in finset.range 20, y j) / 20)) ^ 2 = 9000)
                                        (h_sum_xy : ∑ i in finset.range 20, (x i - ((∑ j in finset.range 20, x j) / 20)) * (y i - ((∑ j in finset.range 20, y j) / 20)) = 800) :
  correlation_coefficient x y ≈ 0.94 :=
by sorry

end estimated_population_correct_correlation_coefficient_correct_l216_216209


namespace inequality_solution_l216_216763

theorem inequality_solution (a x : ℝ) (h : 0 < x) : 
  (log a x + |a + log a x| * log (sqrt x) a ≥ a * log x a) ↔ (-1/3 ≤ a ∧ a ≤ 1) :=
sorry

end inequality_solution_l216_216763


namespace average_words_per_hour_l216_216971

theorem average_words_per_hour
  (total_words : ℕ := 60000)
  (total_hours : ℕ := 150)
  (first_period_hours : ℕ := 50)
  (first_period_words : ℕ := total_words / 2) :
  first_period_words / first_period_hours = 600 ∧ total_words / total_hours = 400 := 
by
  sorry

end average_words_per_hour_l216_216971


namespace num_identity_sequences_l216_216796

def vertex := (ℝ × ℝ)
def triangle := (vertex × vertex × vertex)

def T : triangle := ((0, 0), (6, 0), (0, 4))

inductive transformation
| rot90
| rot180
| reflect_x
| reflect_y
| reflect_yx

def apply_transform : transformation → vertex → vertex
| transformation.rot90 (x, y) := (-y, x)
| transformation.rot180 (x, y) := (-x, -y)
| transformation.reflect_x (x, y) := (x, -y)
| transformation.reflect_y (x, y) := (-x, y)
| transformation.reflect_yx (x, y) := (y, x)

def apply_to_triangle (trans : transformation) (tr : triangle) : triangle :=
  let (v1, v2, v3) := tr in
  (apply_transform trans v1, apply_transform trans v2, apply_transform trans v3)

def apply_sequence (seq : list transformation) (tr : triangle) : triangle :=
  seq.foldr apply_to_triangle tr

def all_transform_sequences : list (list transformation) :=
  let trans := [transformation.rot90, transformation.rot180, transformation.reflect_x, transformation.reflect_y, transformation.reflect_yx] in
  trans.bind (λ t1, trans.bind (λ t2, trans.map (λ t3, [t1, t2, t3])))

def count_identity_sequences : ℕ :=
  (all_transform_sequences.filter (λ seq, apply_sequence seq T = T)).length

theorem num_identity_sequences : count_identity_sequences = 18 := sorry

end num_identity_sequences_l216_216796


namespace area_ADC_calculation_l216_216431

-- Definitions and assumptions
variables (BD DC : ℝ)
variables (area_ABD area_ADC : ℝ)

-- Given conditions
axiom ratio_BD_DC : BD / DC = 2 / 5
axiom area_ABD_given : area_ABD = 40

-- The theorem to prove
theorem area_ADC_calculation (h1 : BD / DC = 2 / 5) (h2 : area_ABD = 40) :
  area_ADC = 100 :=
sorry

end area_ADC_calculation_l216_216431


namespace min_sum_of_segment_values_l216_216684

theorem min_sum_of_segment_values (N r g b : ℕ) (h : N = r + g + b) 
  (no_collinear : ∀ p1 p2 p3, p1 ≠ p2 → p1 ≠ p3 → p2 ≠ p3 → 
                               ¬ collinear p1 p2 p3) :
  (N % 2 = 0 → min_sum_of_segment_values N r g b = -N / 2) ∧
  (N % 2 = 1 → min_sum_of_segment_values N r g b = (1 - N) / 2) :=
sorry

end min_sum_of_segment_values_l216_216684


namespace total_volume_of_combined_solid_l216_216092

theorem total_volume_of_combined_solid:
  let s := 8 * Real.sqrt 2 in
  (∃ (V : ℝ),
     (square_base_length s) ∧
     (top_edge_length 2 s) ∧
     (all_other_edges_length s) ∧
     (triangular_prism_base s s (2 * s)) ∧
     (V = 2048 + 576 * Real.sqrt 30)) :=
  sorry

end total_volume_of_combined_solid_l216_216092


namespace seq_general_term_l216_216335

theorem seq_general_term :
  ∃ (a : ℕ → ℕ), (a 1 = 1) ∧ (∀ n : ℕ, n > 0 → n * (a (n + 1)) = 2 * (∑ i in Finset.range (n + 1), a (i + 1))) ∧ (∀ n : ℕ, a n = n) :=
by
  let a : ℕ → ℕ := sorry
  have h1 : a 1 = 1 := sorry
  have h2 : ∀ n : ℕ, n > 0 → n * (a (n + 1)) = 2 * (∑ i in Finset.range (n + 1), a (i + 1)) := sorry
  have h3 : ∀ n : ℕ, a n = n := sorry
  use a
  exact ⟨h1, h2, h3⟩

end seq_general_term_l216_216335


namespace incorrect_statement_C_l216_216642

-- Lean 4 statement to verify correctness of problem translation
theorem incorrect_statement_C (n : ℕ) (w : ℕ → ℕ) :
  (w 1 = 55) ∧
  (w 2 = 110) ∧
  (w 3 = 160) ∧
  (w 4 = 200) ∧
  (w 5 = 254) ∧
  (w 6 = 300) ∧
  (w 7 = 350) →
  ¬(∀ n, w n = 55 * n) :=
by
  intros h
  sorry

end incorrect_statement_C_l216_216642


namespace simplify_expression_l216_216488

theorem simplify_expression (y : ℝ) : 
  4 * y + 9 * y ^ 2 + 8 - (3 - 4 * y - 9 * y ^ 2) = 18 * y ^ 2 + 8 * y + 5 :=
by
  sorry

end simplify_expression_l216_216488


namespace fg_minus_gf_l216_216846

def f (x : ℝ) : ℝ := 8 * x - 12
def g (x : ℝ) : ℝ := x / 4 + 3

theorem fg_minus_gf (x : ℝ) : f (g x) - g (f x) = 12 := 
by
  sorry

end fg_minus_gf_l216_216846


namespace password_equation_l216_216711

theorem password_equation (A B C P Q R : ℕ) (h : 3 * (A * 100000 + B * 10000 + C * 1000 + P * 100 + Q * 10 + R) = 4 * (P * 100000 + Q * 10000 + R * 1000 + A * 100 + B * 10 + C)) :
  (A = 5 ∧ B = 7 ∧ C = 1 ∧ P = 4 ∧ Q = 2 ∧ R = 8) :=
begin
  have := congr_arg (λ x, 3 * x) h,
  rw [mul_assoc, mul_assoc, mul_three_five] at this,
  repeat { rw mul_assoc at this },
  linarith,
end

example : 3 * 571428 = 4 * 428571 := by
  sorry -- As we simply need to demonstrate the relation, we can leave the proof as 'sorry'

end password_equation_l216_216711


namespace problem_85_cube_plus_3_85_square_plus_3_85_plus_1_l216_216556

theorem problem_85_cube_plus_3_85_square_plus_3_85_plus_1 :
  85^3 + 3 * (85^2) + 3 * 85 + 1 = 636256 := 
sorry

end problem_85_cube_plus_3_85_square_plus_3_85_plus_1_l216_216556


namespace value_of_m_l216_216404

-- Define the function given m
def f (x m : ℝ) : ℝ := x^2 - 2 * (abs x) + 2 - m

-- State the theorem to be proved
theorem value_of_m (m : ℝ) :
  (∃ x1 x2 x3 : ℝ, f x1 m = 0 ∧ f x2 m = 0 ∧ f x3 m = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1) →
  m = 2 :=
by
  sorry

end value_of_m_l216_216404


namespace minimum_value_of_f_l216_216311

noncomputable def f (x : ℝ) : ℝ := 16^x - 4^x + 1

theorem minimum_value_of_f : ∃ (x : ℝ), ∀ (y : ℝ), f(y) ≥ f(x) ∧ f(x) = 3 / 4 :=
sorry

end minimum_value_of_f_l216_216311


namespace probability_A_greater_B_l216_216163

theorem probability_A_greater_B :
  let outcomes := [(a, b) | a in [1, 2, 3, 4, 5, 6], b in [1, 2, 3, 4, 5, 6]]
  let favorable := [(a, b) | (a, b) ∈ outcomes, a > b]
  ∑/singleton outcomes in favorable.count / outcomes.count.toRational = (5 / 12 : ℚ) :=
sorry

end probability_A_greater_B_l216_216163


namespace area_shaded_region_part_b_shaded_region_l216_216917

-- Part (a)
theorem area_shaded_region (a : ℝ) (h : a = 4) : 
  let square_area := a * a in
  let shaded_area := square_area / 2 in
  shaded_area = 8 := 
by
  -- Proof is omitted
  sorry

-- Part (b)
theorem part_b_shaded_region (a : ℝ) (h : a = 2) :
  let r := a / 2 in
  let semicircle_area := (π * r^2) / 2 in
  semicircle_area = π / 2 :=
by
  -- Proof is omitted
  sorry

end area_shaded_region_part_b_shaded_region_l216_216917


namespace total_cost_of_constructing_the_path_l216_216955

open Real

-- Define the conditions
def length_field : ℝ := 75
def width_field : ℝ := 55
def path_width : ℝ := 2.8
def area_path_given : ℝ := 1518.72
def cost_per_sq_m : ℝ := 2

-- Define the total cost to be proven
def total_cost : ℝ := 3037.44

-- The statement to be proven
theorem total_cost_of_constructing_the_path :
  let outer_length := length_field + 2 * path_width
  let outer_width := width_field + 2 * path_width
  let total_area_incl_path := outer_length * outer_width
  let area_field := length_field * width_field
  let computed_area_path := total_area_incl_path - area_field
  let given_cost := area_path_given * cost_per_sq_m
  total_cost = given_cost := by
  sorry

end total_cost_of_constructing_the_path_l216_216955


namespace count_five_digit_palindromes_l216_216393

theorem count_five_digit_palindromes : 
  ∃ n : ℕ, (n = 900) ∧ (∃ a b c: ℕ, 
    (1 <= a ∧ a <= 9) ∧ 
    (0 <= b ∧ b <= 9) ∧ 
    (0 <= c ∧ c <= 9) ∧ 
    (∀ x y : ℕ, x = a ∧ y = b → ℕ) ∧ 
    (9 * 10 * 10 = 900)) := 
sorry

end count_five_digit_palindromes_l216_216393


namespace cinema_rows_l216_216068

theorem cinema_rows (h₁ : 312_5 = 3 * 5^2 + 1 * 5^1 + 2 * 5^0)
  (h₂ : ∀ n : ℕ, 82 = 3 * n + 1 → r = n + 1) : r = 28 := by
-- Given conditions
have h2_1 : 82 = 3 * 27 + 1 := by norm_num
-- Given conditions imply
apply h₂ 27 h2_1
-- Conclusion
exact 28
sorry

end cinema_rows_l216_216068


namespace part1_Geom_Sequence_part2_Sum_Formula_l216_216434

noncomputable def a_seq (n : ℕ) : ℕ → ℚ
| 0 => 1
| (n + 1) => (1 + 1 / n) ^ 2 * a_seq n / 3

def b_seq (n : ℕ) : ℚ := a_seq (n + 1) - (a_seq n / 3)

def S_sum (n : ℕ) : ℚ := ∑ i in Finset.range n, b_seq i

theorem part1_Geom_Sequence (n : ℕ) (h : 0 < n) :
  a_seq 0 = 1 ∧ (∀ n, a_seq (n + 1) = (1 + 1 / n) ^ 2 * a_seq n / 3) → 
  ∃ r : ℚ, (∀ n, (a_seq n).to_rat = r ^ (n - 1) * n ^ 2 / 3) :=
sorry

theorem part2_Sum_Formula (n : ℕ) (h : 0 < n) :
  (b_seq n = a_seq (n + 1) - a_seq n / 3) → 
  S_sum n = 3 - (2 * n + 5) / (3 ^ n) :=
sorry

end part1_Geom_Sequence_part2_Sum_Formula_l216_216434


namespace determine_k_from_eccentricity_l216_216361

-- Lean statement for the given conditions and proof goal
theorem determine_k_from_eccentricity (k : ℝ)
  (h1 : ∃ x y : ℝ, (x^2 / 4 + y^2 / k = 1))
  (h2 : ∃ e : ℝ, (e = √2 / 2)) :
  k = 2 ∨ k = 8 := 
sorry

end determine_k_from_eccentricity_l216_216361


namespace sufficient_condition_ab_greater_than_1_l216_216133

theorem sufficient_condition_ab_greater_than_1 (a b : ℝ) (h₁ : a > 1) (h₂ : b > 1) : ab > 1 := 
  sorry

end sufficient_condition_ab_greater_than_1_l216_216133


namespace tan_product_ge_sqrt2_l216_216349

variable {α β γ : ℝ}

theorem tan_product_ge_sqrt2 (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) (hγ : 0 < γ ∧ γ < π / 2) 
  (h : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) : 
  Real.tan α * Real.tan β * Real.tan γ ≥ 2 * Real.sqrt 2 := 
by
  sorry

end tan_product_ge_sqrt2_l216_216349


namespace xanadu_license_plates_l216_216191

theorem xanadu_license_plates : 
  let letters := 26 in
  let digits := 10 in
  letters * letters * digits * digits * digits = 676000 :=
by sorry

end xanadu_license_plates_l216_216191


namespace total_amount_paid_l216_216381

theorem total_amount_paid : 
  let spam_cost := 3
  let peanut_butter_cost := 5
  let bread_cost := 2
  let spam_quantity := 12
  let peanut_butter_quantity := 3
  let bread_quantity := 4
  let spam_total := spam_cost * spam_quantity
  let peanut_butter_total := peanut_butter_cost * peanut_butter_quantity
  let bread_total := bread_cost * bread_quantity
  let total_amount := spam_total + peanut_butter_total + bread_total in
  total_amount = 59 :=
by
  sorry

end total_amount_paid_l216_216381


namespace rhombuses_in_triangle_l216_216238

/-- An equilateral triangle with a side length of 10 is divided into 100 smaller equilateral 
triangles with side lengths of 1. Prove that the number of rhombuses composed of 8 small 
triangles is 84. -/
theorem rhombuses_in_triangle (side_length_large : ℕ) (side_length_small : ℕ) (count_small_triangles : ℕ) 
  (h1 : side_length_large = 10) (h2 : side_length_small = 1) (h3 : count_small_triangles = 100): 
  ∃ (num_rhombuses : ℕ), num_rhombuses = 84 := 
by
  use 84
  sorry

end rhombuses_in_triangle_l216_216238


namespace pyramid_properties_l216_216964

noncomputable def r : ℝ := 10

def volume_pyramid (r : ℝ) : ℝ := (2 / 3) * r^3

def surface_area_pyramid (r : ℝ) : ℝ := 
  r^2 * (Real.sqrt (2 * Real.sqrt 5 + 10) + Real.sqrt 5 - 1)

def tangent_inclination_angle (r : ℝ) : ℝ := 
  (1 / 2) * (Real.sqrt (Real.sqrt 5 + 1))^3

theorem pyramid_properties (r : ℝ) (h_r : r = 10) :
  volume_pyramid r = (2 / 3) * r^3 ∧ 
  surface_area_pyramid r = r^2 * (Real.sqrt (2 * Real.sqrt 5 + 10) + Real.sqrt 5 - 1) ∧ 
  tangent_inclination_angle r = (1 / 2) * (Real.sqrt (Real.sqrt 5 + 1))^3 :=
by
  sorry

end pyramid_properties_l216_216964


namespace complex_powers_sum_zero_l216_216648

theorem complex_powers_sum_zero (i : ℂ) (h : i^2 = -1) : i^2023 + i^2024 + i^2025 + i^2026 = 0 :=
by
  sorry

end complex_powers_sum_zero_l216_216648


namespace find_a_plus_2b_l216_216026

open Real

theorem find_a_plus_2b 
  (a b : ℝ) 
  (ha : 0 < a ∧ a < π / 2) 
  (hb : 0 < b ∧ b < π / 2) 
  (h1 : 4 * (sin a)^2 + 3 * (sin b)^2 = 1) 
  (h2 : 4 * sin (2 * a) - 3 * sin (2 * b) = 0) :
  a + 2 * b = π / 2 :=
sorry

end find_a_plus_2b_l216_216026


namespace reflections_pass_through_midpoints_l216_216577

theorem reflections_pass_through_midpoints 
  (ABCD : Type) [metric_space ABCD] [affine_space REAL ABCD]
  {A B C D: ABCD}
  (M_AC : ABCD) (M_BD : ABCD)
  (h1 : M_AC = midpoint (A, C))
  (h2 : M_BD = midpoint (B, D))
  (h_reflect_BD_B : reflect_in_bisector (angle_bisector B) D = M_AC)
  (h_reflect_BD_D : reflect_in_bisector (angle_bisector D) B = M_AC)
  :
  reflect_in_bisector (angle_bisector A) C = M_BD ∧
  reflect_in_bisector (angle_bisector C) A = M_BD := 
sorry

end reflections_pass_through_midpoints_l216_216577


namespace find_common_difference_l216_216255

-- Definitions for arithmetic sequences and sums
def S (a1 d : ℕ) (n : ℕ) := (n * (2 * a1 + (n - 1) * d)) / 2
def a (a1 d : ℕ) (n : ℕ) := a1 + (n - 1) * d

theorem find_common_difference (a1 d : ℕ) :
  S a1 d 3 = 6 → a a1 d 3 = 4 → d = 2 :=
by
  intros S3_eq a3_eq
  sorry

end find_common_difference_l216_216255


namespace artifact_age_count_l216_216586

-- Definitions based on conditions and question
def digits : Multiset ℕ := {2, 2, 3, 3, 7, 9}
def odd_digits : Set ℕ := {3, 7, 9}

-- The proof statement
theorem artifact_age_count : Nat :=
  have perm_count (d : ℕ) (rem : Multiset ℕ) := (rem.erase d).perm.countp fun x => x = 2 / 2!
  tsum (λ d in odd_digits, perm_count d digits) = 180 := sorry

end artifact_age_count_l216_216586


namespace b_arithmetic_sequence_min_m_l216_216713

-- Define sequence {a_n}
def a : ℕ+ → ℝ 
| ⟨1, _⟩ := 2
| ⟨n + 2, h⟩ := 2 - 1 / a ⟨n + 1, Nat.succ_pos _⟩

-- Define sequence {b_n}
def b (n : ℕ+) : ℝ := 1 / (a n - 1)

-- Prove that sequence {b_n} is an arithmetic sequence
theorem b_arithmetic_sequence : ∃ d, ∀ n : ℕ+, b n = b ⟨1, sorry⟩ + d * (n - 1 : ℚ) := sorry

-- Define the sum {T_n}
def T (n : ℕ+) : ℝ := ∑ k in Finset.range n, 1 / (b ⟨k.1, sorry⟩ * b ⟨k.1 + 2, sorry⟩)

-- Prove that the minimum positive integer m such that T_n <= m / 12 is 9
theorem min_m (m : ℕ+) : T n ≤ m / 12 ↔ m = 9 := sorry

end b_arithmetic_sequence_min_m_l216_216713


namespace estimate_population_correct_correlation_coefficient_correct_l216_216218

namespace EcologicalSurvey

def estimated_population (total_animals : ℕ) (plots : ℕ) : ℕ :=
  (total_animals / 20) * plots

theorem estimate_population_correct :
  estimated_population 1200 200 = 12000 :=
by 
  -- Proof would go here.
  sorry

def correlation_coefficient (sum_xi_xbar : ℝ) (sum_yi_ybar : ℝ) (sum_prod_xi_yi : ℝ) : ℝ :=
  sum_prod_xi_yi / Math.sqrt (sum_xi_xbar * sum_yi_ybar)

theorem correlation_coefficient_correct :
  correlation_coefficient 80 9000 800 ≈ 0.94 :=
by 
  -- Proof would go here.
  sorry

end EcologicalSurvey

end estimate_population_correct_correlation_coefficient_correct_l216_216218


namespace min_value_of_f_l216_216291

def f (x : ℝ) : ℝ := 16^x - 4^x + 1

theorem min_value_of_f : ∃ x : ℝ, f x = 3/4 :=
by
  sorry

end min_value_of_f_l216_216291


namespace A_2013_eq_neg_1007_l216_216475

def A (n : ℕ) : ℤ :=
  (-1)^n * ((n + 1) / 2)

theorem A_2013_eq_neg_1007 : A 2013 = -1007 :=
by
  sorry

end A_2013_eq_neg_1007_l216_216475


namespace firefighter_pays_in_taxes_per_month_l216_216164

-- Define the given conditions
def hourly_wage : ℝ := 30
def weekly_hours : ℝ := 48
def monthly_income : ℝ := (hourly_wage * weekly_hours) * 4

def rent_fraction : ℝ := 1/3
def food_expense : ℝ := 500
def leftover_money : ℝ := 2340

-- Define the target theorem to prove
theorem firefighter_pays_in_taxes_per_month :
  1 / 3 * monthly_income + food_expense + leftover_money + 1000 = monthly_income :=
by
  -- Proof is omitted
  sorry

end firefighter_pays_in_taxes_per_month_l216_216164


namespace triangle_non_existence_no_solution_max_value_expression_l216_216777

-- Define sides and angles
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Corresponding opposite sides

-- Define the triangle conditions
def triangle_sides_angles (a b c A B C : ℝ) : Prop := 
  (a^2 = (1 - Real.cos A) / (1 - Real.cos B)) ∧ 
  (b = 1) ∧ 
  -- Additional properties ensuring we have a valid triangle can be added here
  (A ≠ B) -- Non-isosceles condition (equivalent to angles being different).

-- Prove non-existence under given conditions
theorem triangle_non_existence_no_solution (h : triangle_sides_angles a b c A B C) : false := 
sorry 

-- Define the maximization problem
theorem max_value_expression (h : a^2 = (1 - Real.cos A) / (1 - Real.cos B)) : 
(∃ b c, (b = 1) → ∀ a, a > 0 → (c > 0) ∧ ((1/c) * (1/b - 1/a)) ≤ (3 - 2 * Real.sqrt 2)) := 
sorry

end triangle_non_existence_no_solution_max_value_expression_l216_216777


namespace sum_of_divisors_l216_216454

theorem sum_of_divisors (hd : ∀ d : ℕ, (d > 20 ∧ d < 100) ∧ 143 % d = 3 → d ∈ {20, 28, 35, 70}) : 
  (∑ d : ℕ in {20, 28, 35, 70}, d) = 153 :=
by
  -- Adding sorry to skip the proof here
  sorry

end sum_of_divisors_l216_216454


namespace range_of_a_l216_216739

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * Real.sin x - (Real.cos x)^2 ≤ 3) : -3 ≤ a ∧ a ≤ 3 :=
by
  sorry

end range_of_a_l216_216739


namespace sum_of_prime_factors_of_2_pow_10_minus_1_l216_216876

theorem sum_of_prime_factors_of_2_pow_10_minus_1 :
  ∃ p1 p2 p3 : ℕ, prime p1 ∧ prime p2 ∧ prime p3 ∧ (p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) ∧ 
  (2^10 - 1) % p1 = 0 ∧ (2^10 - 1) % p2 = 0 ∧ (2^10 - 1) % p3 = 0 ∧ (p1 + p2 + p3 = 45) :=
sorry

end sum_of_prime_factors_of_2_pow_10_minus_1_l216_216876


namespace quadratic_expression_quadratic_zeros_and_neg_max_value_of_F_l216_216334

section

/-- Let f be a quadratic function such that f(x) = ax^2 + bx and f(x - 1) = f(x) + x - 1.
    Prove that f(x) = -1/2 * x^2 + 1/2 * x. -/
theorem quadratic_expression (a b : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = a * x^2 + b * x)
  (hf : ∀ x : ℝ, f (x - 1) = f x + x - 1) :
  ∃ f, f = (λ x : ℝ, -1/2 * x^2 + 1/2 * x) := sorry

/-- Given f(x) = -1/2 * x^2 + 1/2 * x, prove that the zeros are x = 0 and x = 1.
    Prove that f(x) < 0 for x > 1 or x < 0. -/
theorem quadratic_zeros_and_neg (f : ℝ → ℝ) (hf : ∀ x : ℝ, f x = -1/2 * x^2 + 1/2 * x) :
  (f 0 = 0 ∧ f 1 = 0) ∧ (∀ x : ℝ, f x < 0 ↔ x > 1 ∨ x < 0) := sorry

/-- Let F(x) = 4 * f(a^x) + 3 * a^(2 * x) - 1 with a > 0 and a ≠ 1.
    If F(x) has a maximum value of 14 for x ∈ [-1, 1], then a = 1/3 or a = 3. -/
theorem max_value_of_F (a : ℝ) (f : ℝ → ℝ) (F : ℝ → ℝ) 
  (hf : ∀ x : ℝ, f x = -1/2 * x^2 + 1/2 * x)
  (hF : ∀ x : ℝ, F x = 4 * f (a^x) + 3 * a^(2 * x) - 1)
  (h_max : ∀ x ∈ (Icc (-1 : ℝ) 1), F x ≤ 14) :
  a = 1/3 ∨ a = 3 := sorry

end

end quadratic_expression_quadratic_zeros_and_neg_max_value_of_F_l216_216334


namespace evaluate_i_powers_sum_l216_216646

-- Given conditions: i is the imaginary unit
def i : ℂ := Complex.I

-- Proof problem: Prove that i^2023 + i^2024 + i^2025 + i^2026 = 0
theorem evaluate_i_powers_sum : i^2023 + i^2024 + i^2025 + i^2026 = 0 := 
by sorry

end evaluate_i_powers_sum_l216_216646


namespace area_of_PF1F2_l216_216079

noncomputable def area_of_triangle_with_foci_and_point_on_ellipse : ℝ :=
let a : ℝ := 2,
    b : ℝ := sqrt 2,
    c := sqrt (a^2 - b^2),
    F1 := (c, 0),
    F2 := (-c, 0),
    P := (1, sqrt (2 - 1/2)) -- This is a point on the ellipse
in
-- Assuming |PF1| - |PF2| = 2 and |PF1| + |PF2| = 4
let |PF1| := 3,
    |PF2| := 1,
    angle_cos := (|PF1|^2 + |PF2|^2 - (2 * sqrt 2)^2) / (2 * |PF1| * |PF2|)
in
1/2 * |PF1| * |PF2| * sqrt (1 - angle_cos^2)

theorem area_of_PF1F2 : area_of_triangle_with_foci_and_point_on_ellipse = sqrt 2 :=
sorry

end area_of_PF1F2_l216_216079


namespace min_value_of_f_l216_216294

def f (x : ℝ) : ℝ := 16^x - 4^x + 1

theorem min_value_of_f : ∃ x : ℝ, f x = 3/4 :=
by
  sorry

end min_value_of_f_l216_216294


namespace polynomial_value_at_neg2_l216_216018

noncomputable def P (x : ℝ) : ℝ :=
  x^2 + (1/24) * (x-1) * (x-2) * (x-3) * (x-4)

theorem polynomial_value_at_neg2 :
  P(0) = 1 →
  P(1) = 1 →
  P(2) = 4 →
  P(3) = 9 →
  P(4) = 16 →
  P(-2) = 19 :=
by
  intros h0 h1 h2 h3 h4
  rw [P] at *
  -- rest of the proof would follow, but it's skipped here
  sorry

end polynomial_value_at_neg2_l216_216018


namespace babies_per_batch_l216_216947

def num_kettles : ℕ := 6
def pregnancies_per_kettle : ℕ := 15
def total_pregnancies : ℕ := num_kettles * pregnancies_per_kettle
def loss_rate : ℝ := 0.25
def survival_rate : ℝ := 1 - loss_rate
def expected_babies : ℕ := 270
def total_babies_before_loss : ℕ := (expected_babies : ℝ) / survival_rate

theorem babies_per_batch :
  (total_babies_before_loss / total_pregnancies) = 4 :=
sorry

end babies_per_batch_l216_216947


namespace sum_a_b_c_l216_216259

namespace PolynomialRoots

variable (p q r : ℝ)
variables (a b c : ℝ)
variables (s_0 s_1 s_2 s_3 : ℝ)

def polynomial : Polynomial ℝ := Polynomial.C (-14) + Polynomial.X * (Polynomial.C 17 + Polynomial.X * (Polynomial.C (-8) + Polynomial.X))

-- Given conditions
def s₀ : ℝ := 3
def s₁ : ℝ := 8
def s₂ : ℝ := 17

-- Assumption about the relation between s_k
axiom relation (k : ℕ) : ∀ k ≥ 2, s_0 = 3 → s_1 = 8 → s_2 = 17 →
  s_3 = 42 → s_{k+1} = a * s_k + b * s_{k-1} + c * s_{k-2}

-- The statement we want to prove
theorem sum_a_b_c : a + b + c = 9 :=
sorry

end PolynomialRoots

end sum_a_b_c_l216_216259


namespace number_of_triangles_2016_30_l216_216329

def f (m n : ℕ) : ℕ :=
  2 * m - n - 2

theorem number_of_triangles_2016_30 :
  f 2016 30 = 4000 := 
by
  sorry

end number_of_triangles_2016_30_l216_216329


namespace farmer_pomelos_dozen_l216_216001

theorem farmer_pomelos_dozen (pomelos_last_week : ℕ) (boxes_last_week : ℕ) (boxes_this_week : ℕ) :
  pomelos_last_week = 240 → boxes_last_week = 10 → boxes_this_week = 20 →
  (pomelos_last_week / boxes_last_week) * boxes_this_week / 12 = 40 := 
by
  intro h1 h2 h3
  sorry

end farmer_pomelos_dozen_l216_216001


namespace profit_per_cake_l216_216981

theorem profit_per_cake (ingredient_cost : ℝ) (packaging_cost : ℝ) (selling_price : ℝ) (cake_count : ℝ)
    (h1 : ingredient_cost = 12) (h2 : packaging_cost = 1) (h3 : selling_price = 15) (h4 : cake_count = 2) :
    selling_price - (ingredient_cost / cake_count + packaging_cost) = 8 := by
  sorry

end profit_per_cake_l216_216981


namespace first_rearrangement_property_year_l216_216521

def rearranging_digits_preserves_order (y : ℕ) : Prop :=
  ∀ p : List (Fin 10), (List.perm (to_digits 10 y) p → from_digits 10 p ≥ y)

theorem first_rearrangement_property_year {y : ℕ} :
  (∀ n < 2022, ¬rearranging_digits_preserves_order n) → rearranging_digits_preserves_order 2022 :=
begin
  sorry
end

end first_rearrangement_property_year_l216_216521


namespace correct_proposition_l216_216973

-- Definitions of the conditions
def identical_three_views_implies_cube (s : Solid) : Prop :=
  (s.front_view = s.top_view ∧ s.top_view = s.side_view) → s.is_cube

def front_and_top_views_rectangles_implies_cuboid (s : Solid) : Prop :=
  (s.front_view.is_rectangle ∧ s.top_view.is_rectangle) → s.is_cuboid

def three_views_rectangles_implies_cuboid (s : Solid) : Prop :=
  (s.front_view.is_rectangle ∧ s.top_view.is_rectangle ∧ s.side_view.is_rectangle) → s.is_cuboid

def front_and_side_views_trapezoids_implies_frustum (s : Solid) : Prop :=
  (s.front_view.is_isosceles_trapezoid ∧ s.side_view.is_isosceles_trapezoid) → s.is_frustum

-- Stating the problem: prove that proposition (3) is the correct one
theorem correct_proposition (s : Solid) :
  (identical_three_views_implies_cube s ∨ 
   front_and_top_views_rectangles_implies_cuboid s ∨ 
   three_views_rectangles_implies_cuboid s ∨ 
   front_and_side_views_trapezoids_implies_frustum s) → 
  three_views_rectangles_implies_cuboid s :=
sorry

end correct_proposition_l216_216973


namespace find_table_cost_l216_216261

-- Conditions
variables (T : ℝ) (couch_cost table_cost lamp_cost initial_payment remaining_debt total_cost_after_payment : ℝ)
variables (H1 : couch_cost = 750) 
          (H2 : lamp_cost = 50) 
          (H3 : initial_payment = 500) 
          (H4 : remaining_debt = 400)
          (H5 : total_cost_after_payment = 400)

-- Question: How much did the table cost?
def table_cost := 100

-- Prove that given conditions table cost is indeed $100
theorem find_table_cost : 
  table_cost = 100  
  ∧ (750 + 100 + 50 - 500 = 400) := 
by {
  sorry
}

end find_table_cost_l216_216261


namespace farmer_shipped_pomelos_in_dozens_l216_216003

theorem farmer_shipped_pomelos_in_dozens :
  let pomelos_per_box := 240 / 10 in
  let dozens_per_box := pomelos_per_box / 12 in
  let total_boxes := 10 + 20 in
  total_boxes * dozens_per_box = 60 :=
by
  have pomelos_per_box_eq : pomelos_per_box = 24 := by norm_num
  have dozens_per_box_eq : dozens_per_box = 2 := by norm_num
  have total_boxes_eq : total_boxes = 30 := by norm_num
  rw [pomelos_per_box_eq, dozens_per_box_eq, total_boxes_eq]
  norm_num
  sorry

end farmer_shipped_pomelos_in_dozens_l216_216003


namespace middle_value_bounds_l216_216516

theorem middle_value_bounds (a b c : ℝ) (h1 : a + b + c = 10)
  (h2 : a > b) (h3 : b > c) (h4 : a - c = 3) : 
  7 / 3 < b ∧ b < 13 / 3 :=
by
  sorry

end middle_value_bounds_l216_216516


namespace option_B_coplanar_l216_216975

-- Define the three vectors in Option B.
def a : ℝ × ℝ × ℝ := (1, 2, -3)
def b : ℝ × ℝ × ℝ := (-2, -4, 6)
def c : ℝ × ℝ × ℝ := (1, 0, 5)

-- Define the coplanarity condition for vectors a, b, and c.
def coplanar (a b c : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = k • a

-- Prove that the vectors in Option B are coplanar.
theorem option_B_coplanar : coplanar a b c :=
sorry

end option_B_coplanar_l216_216975


namespace min_value_of_d_l216_216836

theorem min_value_of_d (a b c d : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c < d) 
  (h4 : a > 0) (h5 : b > 0) (h6 : c > 0) (h7 : d > 0) 
  (h8 : ∀ x y : ℕ, (3 * x + y = 3004) ∧ (y = | x - a | + | x - b | + | x - c | + | x - d |) → 
    (3 * (5*d - (a+b+c)) + (2*d - (a+b+c)) = 3004)) :
  d = 602 := 
sorry

end min_value_of_d_l216_216836


namespace one_circumcircle_radius_at_least_one_l216_216793

theorem one_circumcircle_radius_at_least_one (A B C D E F : ℝ × ℝ)
  (h1 : dist A B = 1) (h2 : dist B C = 1) (h3 : dist C D = 1)
  (h4 : dist D E = 1) (h5 : dist E F = 1) (h6 : dist F A = 1)
  (h_convex : convex (set_range [A, B, C, D, E, F])) :
  ∃ R : ℝ, R ≥ 1 ∧ (R = circumradius A C E ∨ R = circumradius B D F) :=
by
  sorry

end one_circumcircle_radius_at_least_one_l216_216793


namespace largest_non_decomposable_l216_216428

/-
The denominations of coins in Limonia are 𝑐 = {5^n, 5^(n-1) * 7, 5^(n-2) * 7^2, ..., 5 * 7^(n-1), 7^n}, 
where n is a natural number.
Given this, prove the largest amount that cannot be provided by the bank is 2 * 7^(n+1) - 3 * 5^(n+1).
-/
theorem largest_non_decomposable (n : ℕ) :
    ∃ s, s = 2 * 7^(n+1) - 3 * 5^(n+1) ∧ ¬(∃ a b c d e f, s = a * 5^n + b * (5^(n-1) * 7) + c * (5^(n-2) * 7^2) + d * (5^(n-3) * 7^3) + e * (5 * 7^(n-1)) + f * 7^n) :=
begin
  sorry
end

end largest_non_decomposable_l216_216428


namespace volume_ratio_cylinder_cone_sphere_l216_216942

theorem volume_ratio_cylinder_cone_sphere 
  (R : ℝ) 
  (height_cylinder : ℝ) 
  (height_cone : ℝ) 
  (diameter_cylinder : ℝ) 
  (diameter_cone : ℝ) 
  (diameter_sphere : ℝ) 
  (h1 : height_cylinder = 2 * R)
  (h2 : height_cone = 2 * R)
  (h3 : diameter_cylinder = 2 * R)
  (h4 : diameter_cone = 2 * R)
  (h5 : diameter_sphere = 2 * R) :
  let V_cylinder := 2 * π * R^3,
      V_cone := (2/3) * π * R^3,
      V_sphere := (4/3) * π * R^3
  in V_cylinder / V_cone = 3 / 1 ∧ V_cylinder / V_sphere = 3 / 2 ∧ V_cone / V_sphere = 1 / 2 :=
by sorry

end volume_ratio_cylinder_cone_sphere_l216_216942


namespace probability_m_n_greater_than_5_l216_216186

-- Define the sample space for rolling a six-sided die twice
def sample_space := (fin 6) × (fin 6)

-- Define the subset where m is either 2 or 4
def subset_m := {x : sample_space | x.1 + 1 = 2 ∨ x.1 + 1 = 4}

-- Define the subset of favorable outcomes where m + n > 5
def favorable_outcomes := {x : sample_space | (x.1 + 1 = 2 ∧ x.2 + 1 > 3) ∨ (x.1 + 1 = 4 ∧ x.2 + 1 > 1)}

-- Calculate the probability
def probability : ℚ := (favorable_outcomes ∩ subset_m).card.to_rat / subset_m.card.to_rat

theorem probability_m_n_greater_than_5 : probability = 2 / 3 := by
  sorry

end probability_m_n_greater_than_5_l216_216186


namespace gear_rotation_paradox_l216_216491

-- Define the setup and given conditions
variables (B C : Type) (frame : B → C → Prop) (a : Type) (A : Type) 
variables (D E F G H : Type) (teeth : D → ℕ) (same_teeth : D → F → Prop)

-- Define the gear relationship
variables (drives : E → F → Prop) (drives : E → G → Prop) (drives : E → H → Prop)

-- Define the rotation behavior
variables (rotates : E → H → Prop) (rotates_opposite : E → G → Prop) (stationary : F → Prop)

-- Define the frame-gear dynamic
variables (rotating_frame : frame B C → a → D → E → G → F → H → Prop)
variables (fixed_axis : a → A → Prop)
variables (freely_rotating : E → F → G → H → Prop)

-- Statement of the problem
theorem gear_rotation_paradox
    (frame_setup : frame B C)
    (axis_fixed : fixed_axis a A)
    (gear_inside : rotating_frame frame_setup a D E G F H)
    (tooth_configuration : same_teeth D F ∧ teeth G < teeth D ∧ teeth H > teeth D)
    (thin_wheels_behavior : freely_rotating E F G H) :
    rotates E H ∧ rotates_opposite E G ∧ stationary F :=
sorry

end gear_rotation_paradox_l216_216491


namespace total_population_l216_216526

variable (seattle lake_view boise : ℕ)

def lake_view_pop := (lake_view = 24000)
def seattle_pop := (seattle = lake_view - 4000)
def boise_pop := (boise = (3 / 5 : ℚ) * seattle)

theorem total_population (h1 : lake_view_pop) (h2 : seattle_pop) (h3 : boise_pop) :
  lake_view + seattle + boise = 56000 :=
by
  sorry

end total_population_l216_216526


namespace binomial_probability_l216_216372

theorem binomial_probability (p : ℝ) (X Y : ℕ → ℕ) 
    (hX : ∀ k, X k = (Nat.choose 2 k) * p^k * (1 - p)^(2 - k))
    (hY : ∀ k, Y k = (Nat.choose 4 k) * p^k * (1 - p)^(4 - k))
    (hP : (∑ k in {1, 2}, X k) = 5 / 9) :
    (∑ k in {2, 3, 4}, Y k) = 11 / 27 := by sorry

end binomial_probability_l216_216372


namespace car_travel_distance_20_minutes_l216_216719

noncomputable def train_speed_in_mph : ℝ := 80
noncomputable def car_speed_ratio : ℝ := 3/4
noncomputable def car_speed_in_mph : ℝ := car_speed_ratio * train_speed_in_mph
noncomputable def travel_time_in_hours : ℝ := 20 / 60
noncomputable def distance_travelled_by_car : ℝ := car_speed_in_mph * travel_time_in_hours

theorem car_travel_distance_20_minutes : distance_travelled_by_car = 20 := 
by 
  sorry

end car_travel_distance_20_minutes_l216_216719


namespace finite_spheres_block_light_l216_216886

-- Defining the primary entities involved
def lamp_origin : ℝ^d := 0

def is_ray_blocked_by_sphere (r : ℝ) (c : ℝ^d) (ray : ℝ^d → Prop) : Prop :=
  ∀ x, ray x → ‖x - c‖ ≥ r

def finite_spheres_exists (r : ℝ) (S : set (ℝ^d)) : Prop :=
  ∃ (sphere_centers : finset (ℝ^d)), 
  ∀ ray, (∀ x, ¬is_ray_blocked_by_sphere r lamp_origin ray) → 
  ∃ c ∈ sphere_centers, is_ray_blocked_by_sphere r c ray

-- The main theorem statement
theorem finite_spheres_block_light (r : ℝ) (S : set (ℝ^d)) (hr : r < 1 / 2) :
  finite_spheres_exists r S := 
sorry

end finite_spheres_block_light_l216_216886


namespace smallest_n_Q_l216_216413

-- Define Q(n) following the problem conditions
noncomputable def Q (n : ℕ) : ℚ :=
  let p := (∏ k in Finset.range n, (2 * k + 2)/(2 * k + 3)) in
  p * (1 / (2 * n + 1))

-- Define the mathematical problem statement
theorem smallest_n_Q (n : ℕ) : (n ≥ 1 ∧ Q n < 1/2023) ↔ n = 23 := 
by
  sorry

end smallest_n_Q_l216_216413


namespace average_rounds_is_3_l216_216511

/-- Define the number of golfers and the rounds they played. -/
def golfers_and_rounds : list (ℕ × ℕ) := [(1, 3), (2, 5), (3, 7), (6, 2), (8, 1)]

/-- Definition of total number of rounds played by all golfers. -/
def total_rounds_played : ℕ := (3 * 1) + (5 * 2) + (7 * 3) + (2 * 6) + (1 * 8)

/-- Definition of total number of golfers. -/
def total_golfers : ℕ := 3 + 5 + 7 + 2 + 1

/-- Calculate the average number of rounds played by each golfer. -/
def average_rounds_played : ℕ := total_rounds_played / total_golfers

/-- Proof statement: The average number of rounds played is 3. -/
theorem average_rounds_is_3 : average_rounds_played = 3 :=
  by sorry

end average_rounds_is_3_l216_216511


namespace even_positive_factors_count_l216_216810

theorem even_positive_factors_count (m : ℕ) (h : m = 2^4 * 3^2 * 5 * 7) : 
  ∃ n, n = 48 ∧ ∀ d, d ∣ m → d % 2 = 0 → d ∈ finset.range (48 + 1) := 
  begin
    sorry
  end

end even_positive_factors_count_l216_216810


namespace min_value_of_f_l216_216498

def f (x : ℝ) : ℝ :=
  if x ≥ 4 then 2 * x - 1 else x^2 - 2 * x + 3

theorem min_value_of_f : ∃ x : ℝ, (∀ y : ℝ, f y ≥ 2) ∧ (f x = 2) ∧ (x = 1) := 
by
  -- The proof will go here
  sorry

end min_value_of_f_l216_216498


namespace cos_x_when_sin_x_is_given_l216_216666

theorem cos_x_when_sin_x_is_given (x : ℝ) (h : Real.sin x = (Real.sqrt 5) / 5) :
  Real.cos x = -(Real.sqrt 20) / 5 :=
sorry

end cos_x_when_sin_x_is_given_l216_216666


namespace point_on_line_l216_216512

theorem point_on_line (x y : ℝ) (h₁ : (6, 12) ∈ {(x, y) | y = 3 * x - 6})
  (h₂ : (0, -6) ∈ {(x, y) | y = 3 * x - 6})
  : (3, 3) ∈ {(x, y) | y = 3 * x - 6} :=
sorry

end point_on_line_l216_216512


namespace carrots_per_cabbage_l216_216781

theorem carrots_per_cabbage (white_rabbit_stored gray_rabbit_stored : ℕ) (x y : ℕ) : white_rabbit_stored = 180 -> gray_rabbit_stored = 120 -> 120 - x + y = 150 -> 10 < x -> x < 20 -> y - x = 30 -> y / x = 3 :=
by
  assume h1 h2 h3 h4 h5 h6
  have h7 : y = x + 30 := by linarith
  rw [h7] at ⊢
  have h8 : x + 30 = 3 * x := by sorry
  exact div_eq_of_eq_mul_right (by exact dec_trivial : 0 < x) h8

end carrots_per_cabbage_l216_216781


namespace soccer_team_unplayed_players_l216_216183

theorem soccer_team_unplayed_players:
  ∀ (total_players first_half_players first_half_subs : ℕ),
  first_half_players = 11 →
  first_half_subs = 2 →
  total_players = 24 →
  (total_players - (first_half_players + first_half_subs + 2 * first_half_subs)) = 7 :=
by
  intros total_players first_half_players first_half_subs h1 h2 h3
  rw [h1, h2, h3]
  show 24 - (11 + 2 + 2 * 2) = 7
  sorry

end soccer_team_unplayed_players_l216_216183


namespace triangle_angles_l216_216090

noncomputable def angle_of_triangle (a b c : ℝ) : Prop :=
  let cosθ := (a^2 + b^2 - c^2) / (2 * a * b) in
  let θ := Real.arccos cosθ in
  θ = Real.arccos (7/18 + 2 * Real.sqrt 6 / 9)

-- The equivalent Lean statement for the math proof
theorem triangle_angles :
  let a := 3
  let b := 3
  let c := Real.sqrt 8 - Real.sqrt 3 in
  angle_of_triangle a b c ∧
  angle_of_triangle b c a ∧ 
  angle_of_triangle c a b :=
by
  sorry

end triangle_angles_l216_216090


namespace triangle_problem_l216_216775

noncomputable def triangle_angles_and_sides (a b c : ℝ) (C A : ℝ) : Prop :=
  ∃ (a b c : ℝ) (C A : ℝ), 
  (c / 2 = b - a * Real.cos C) ∧ 
  (A = 60) ∧ 
  (a = Real.sqrt 15) ∧ 
  (b = 4) ∧ 
  (c = 2 + Real.sqrt 3) ∨ (c = 2 - Real.sqrt 3)

theorem triangle_problem :
  ∀ (a b c : ℝ) (C A : ℝ), 
  triangle_angles_and_sides a b c C A :=
by
  intro a b c C A
  apply exists.intro (a b c C A)
  split
  sorry

end triangle_problem_l216_216775


namespace numerical_characteristics_of_X_l216_216172

noncomputable def cdf (x : ℝ) : ℝ := 
  if x ≤ 0 then 0
  else if x ≤ 1 then x^3
  else 1

noncomputable def pdf (x : ℝ) : ℝ := 
  if x ≤ 0 then 0
  else if x ≤ 1 then 3 * x^2
  else 0

noncomputable def M_X : ℝ := ∫ x in 0..1, x * pdf x

noncomputable def D_X : ℝ := ∫ x in 0..1, (x - M_X)^2 * pdf x

theorem numerical_characteristics_of_X :
  M_X = 3 / 4 ∧
  D_X = 3 / 80 ∧
  real.sqrt D_X = real.sqrt (3 / 80) :=
by
  sorry

end numerical_characteristics_of_X_l216_216172


namespace sufficient_condition_not_necessary_condition_l216_216148

theorem sufficient_condition (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) : ab > 0 := by
  sorry

theorem not_necessary_condition (a b : ℝ) : ¬(a > 0 ∧ b > 0) → ab > 0 := by
  sorry

end sufficient_condition_not_necessary_condition_l216_216148


namespace power_of_power_example_l216_216273

theorem power_of_power_example : (3^2)^4 = 6561 := by
  sorry

end power_of_power_example_l216_216273


namespace det_eq_zero_sol_y_l216_216330

noncomputable def solution_set (b : ℝ) (hb : b ≠ 0) : set ℝ :=
  { y | y = ( -3 + real.sqrt 17) * b / 4 ∨ y = ( -3 - real.sqrt 17) * b / 4 }

theorem det_eq_zero_sol_y (b : ℝ) (hb : b ≠ 0) :
  ∃ y : ℝ, y ∈ solution_set b hb ↔
    (∃ y : ℝ, 
      matrix.det ![
        ![y + b, y - b, y], 
        ![y - b, y + b, y], 
        ![y, y, y + b]
      ] = 0) :=
by
  sorry

end det_eq_zero_sol_y_l216_216330


namespace max_value_of_f_l216_216738

-- Defining the function f(x)
def f (x : ℝ) (a b : ℝ) : ℝ := (1 - x^2) * (x^2 + a * x + b)

-- Given conditions
variable (a b : ℝ)
variable h_symm : ∀ x, f x a b = f (-x - 4) a b
variable h_f_minus1 : f (-1) a b = 0
variable h_f_1 : f (1) a b = 0

theorem max_value_of_f : ∃ x, f x a b = 16 :=
by
  -- Placeholder for the proof
  sorry

end max_value_of_f_l216_216738


namespace computation_l216_216455

def r (θ : ℝ) : ℝ := 1 / (2 - θ)
def s (θ : ℝ) : ℝ := 1 + 1 / θ

theorem computation : s (r (r (r (s (s (r 6))))) ) = 2.2 := 
by 
  sorry

end computation_l216_216455


namespace pipeline_renovation_l216_216890

theorem pipeline_renovation (
  -- Conditions
  (total_length : ℝ) 
  (daily_efficiency_increase : ℝ)
  (completed_days_early : ℝ) 
  (length_after_20_days : ℝ)
  (total_days : ℝ)
  (days_20 : ℝ)
  (remaining_days : ℝ)
  :

  total_length = 5500 ∧
  daily_efficiency_increase = 0.10 ∧
  completed_days_early = 5 ∧
  length_after_20_days = 3300 ∧
  total_days = 45 ∧
  days_20 = 20 ∧
  remaining_days = 25

  -- Prove the length of pipeline renovated per day during actual construction is 110 meters
  and
  (new_actual_daily_length : ℝ)
  (new_actual_daily_length = 110)
 
  -- Prove that the pipeline should be renovated by at least 22 meters per day in the future
  and
  (additional_meters_per_day : ℝ)
  (additional_meters_per_day ≥ 22)
  ) : true :=
sorry

end pipeline_renovation_l216_216890


namespace soccer_balls_with_holes_l216_216468

-- Define the total number of soccer balls
def total_soccer_balls : ℕ := 40

-- Define the total number of basketballs
def total_basketballs : ℕ := 15

-- Define the number of basketballs with holes
def basketballs_with_holes : ℕ := 7

-- Define the total number of balls without holes
def total_balls_without_holes : ℕ := 18

-- Prove the number of soccer balls with holes given the conditions
theorem soccer_balls_with_holes : (total_soccer_balls - (total_balls_without_holes - (total_basketballs - basketballs_with_holes))) = 30 := by
  sorry

end soccer_balls_with_holes_l216_216468


namespace symmetric_points_y_axis_l216_216687

theorem symmetric_points_y_axis (a b : ℝ) (h₁ : ∃ M N : ℝ × ℝ, M = (a, 3) ∧ N = (4, b) ∧ M.1 = -N.1 ∧ M.2 = N.2) :
  (a + b) ^ 2012 = 1 :=
by 
  sorry

end symmetric_points_y_axis_l216_216687


namespace evaluate_three_squared_raised_four_l216_216282

theorem evaluate_three_squared_raised_four : (3^2)^4 = 6561 := by
  sorry

end evaluate_three_squared_raised_four_l216_216282


namespace determine_first_athlete_high_jump_l216_216105

noncomputable def first_athlete_long_jump : ℝ := 26
noncomputable def first_athlete_triple_jump : ℝ := 30
noncomputable def second_athlete_long_jump : ℝ := 24
noncomputable def second_athlete_triple_jump : ℝ := 34
noncomputable def second_athlete_high_jump : ℝ := 8
noncomputable def winner_average_jump : ℝ := 22

theorem determine_first_athlete_high_jump : 
  ∃ h : ℝ, 
  (first_athlete_long_jump + first_athlete_triple_jump + h) / 3 = winner_average_jump ∧
  h = 10 := 
by 
  let h := 10
  use h
  have avg_eq := (first_athlete_long_jump + first_athlete_triple_jump + h) / 3 = winner_average_jump
  rw [first_athlete_long_jump, first_athlete_triple_jump, winner_average_jump] at avg_eq
  norm_num at avg_eq 
  exact ⟨avg_eq, rfl⟩


end determine_first_athlete_high_jump_l216_216105


namespace total_games_in_conference_l216_216965

-- Definitions based on the conditions
def numTeams := 16
def divisionTeams := 8
def gamesWithinDivisionPerTeam := 21
def gamesAcrossDivisionPerTeam := 16
def totalGamesPerTeam := 37
def totalGameCount := 592
def actualGameCount := 296

-- Proof statement
theorem total_games_in_conference : actualGameCount = (totalGameCount / 2) :=
  by sorry

end total_games_in_conference_l216_216965


namespace hexagon_triangle_area_l216_216484

noncomputable def equilateral_triangle_area_from_hexagons (s : ℝ) : ℝ :=
  let R := s
  let triangle_side := 2 * R
  let area := (real.sqrt 3 / 4) * triangle_side^2
  area

theorem hexagon_triangle_area (s : ℝ) (h : s = 2) :
  equilateral_triangle_area_from_hexagons s = 4 * real.sqrt 3 :=
by
  rw [equilateral_triangle_area_from_hexagons, h]
  simp only [mul_assoc, pow_two, mul_comm]
  norm_num
  exact congr_arg _ (eq.refl _)

#reduce hexagon_triangle_area 2 (by norm_num)

end hexagon_triangle_area_l216_216484


namespace length_unwound_rope_l216_216595

/-- Setup the parameters for the problem --/
variables (m : ℝ) (R : ℝ) (v₀ : ℝ) (L₀ : ℝ) (T_max : ℝ) (r : ℝ)

/-- Given conditions that the rope's tension reaches a maximum value before breaking --/
def rope_break_condition : Prop :=
  T_max = (m * v₀^2) / (2 * (L₀ - r))

/-- Theorem stating the length of the rope not yet wound --/
theorem length_unwound_rope (h : rope_break_condition m R v₀ L₀ T_max r) :
  L₀ - r = (m * v₀^2) / (2 * T_max) :=
sorry

end length_unwound_rope_l216_216595


namespace estimate_population_correct_correlation_coefficient_correct_l216_216217

namespace EcologicalSurvey

def estimated_population (total_animals : ℕ) (plots : ℕ) : ℕ :=
  (total_animals / 20) * plots

theorem estimate_population_correct :
  estimated_population 1200 200 = 12000 :=
by 
  -- Proof would go here.
  sorry

def correlation_coefficient (sum_xi_xbar : ℝ) (sum_yi_ybar : ℝ) (sum_prod_xi_yi : ℝ) : ℝ :=
  sum_prod_xi_yi / Math.sqrt (sum_xi_xbar * sum_yi_ybar)

theorem correlation_coefficient_correct :
  correlation_coefficient 80 9000 800 ≈ 0.94 :=
by 
  -- Proof would go here.
  sorry

end EcologicalSurvey

end estimate_population_correct_correlation_coefficient_correct_l216_216217


namespace units_digit_of_factorial_product_l216_216316

theorem units_digit_of_factorial_product :
  (∏ n in finset.range 15, n.factorial) % 10 = 0 :=
by
  -- The below is a placeholder for the actual proof
  sorry

end units_digit_of_factorial_product_l216_216316


namespace circumference_of_cone_base_l216_216176

theorem circumference_of_cone_base (V : ℝ) (h : ℝ) (C : ℝ) (π := Real.pi) 
  (volume_eq : V = 24 * π) (height_eq : h = 6) 
  (circumference_eq : C = 4 * Real.sqrt 3 * π) :
  ∃ r : ℝ, (V = (1 / 3) * π * r^2 * h) ∧ (C = 2 * π * r) :=
by
  sorry

end circumference_of_cone_base_l216_216176


namespace estimated_population_correct_correlation_coefficient_correct_l216_216208

variables (x y : ℕ → ℝ)
variables (n : ℕ) (plots : ℕ)

noncomputable def average_y := (∑ i in finset.range 20, y i) / 20

noncomputable def estimated_population := average_y y * plots

theorem estimated_population_correct (h_sum_y : ∑ i in finset.range 20, y i = 1200)
                                      (h_plots : plots = 200) :
  estimated_population y plots = 12000 :=
by sorry

noncomputable def correlation_coefficient (x y : ℕ → ℝ) :=
  let sum_xy := ∑ i in finset.range 20, (x i - ((∑ j in finset.range 20, x j) / 20)) * 
                               (y i - ((∑ j in finset.range 20, y j) / 20)) in
  let sum_xx := ∑ i in finset.range 20, (x i - ((∑ j in finset.range 20, x j) / 20)) ^ 2 in
  let sum_yy := ∑ i in finset.range 20, (y i - ((∑ j in finset.range 20, y j) / 20)) ^ 2 in
  sum_xy / (real.sqrt (sum_xx * sum_yy))

theorem correlation_coefficient_correct (h_sum_xx : ∑ i in finset.range 20, (x i - ((∑ j in finset.range 20, x j) / 20)) ^ 2 = 80)
                                        (h_sum_yy : ∑ i in finset.range 20, (y i - ((∑ j in finset.range 20, y j) / 20)) ^ 2 = 9000)
                                        (h_sum_xy : ∑ i in finset.range 20, (x i - ((∑ j in finset.range 20, x j) / 20)) * (y i - ((∑ j in finset.range 20, y j) / 20)) = 800) :
  correlation_coefficient x y ≈ 0.94 :=
by sorry

end estimated_population_correct_correlation_coefficient_correct_l216_216208


namespace intercepts_line_5x_minus_2y_minus_10_eq_0_l216_216506

theorem intercepts_line_5x_minus_2y_minus_10_eq_0 :
  ∃ a b : ℝ, (a = 2 ∧ b = -5) ∧ (∀ x y : ℝ, 5 * x - 2 * y - 10 = 0 → 
     ((y = 0 ∧ x = a) ∨ (x = 0 ∧ y = b))) :=
by
  sorry

end intercepts_line_5x_minus_2y_minus_10_eq_0_l216_216506


namespace Brenda_weight_correct_l216_216995

-- Conditions
def MelWeight : ℕ := 70
def BrendaWeight : ℕ := 3 * MelWeight + 10

-- Proof problem
theorem Brenda_weight_correct : BrendaWeight = 220 := by
  sorry

end Brenda_weight_correct_l216_216995


namespace dice_sum_not_possible_l216_216119

   theorem dice_sum_not_possible (a b c d : ℕ) :
     (1 ≤ a ∧ a ≤ 6) → (1 ≤ b ∧ b ≤ 6) → (1 ≤ c ∧ c ≤ 6) → (1 ≤ d ∧ d ≤ 6) →
     (a * b * c * d = 360) → ¬ (a + b + c + d = 20) :=
   by
     intros ha hb hc hd prod eq_sum
     -- Proof skipped
     sorry
   
end dice_sum_not_possible_l216_216119


namespace estimated_population_correct_correlation_coefficient_correct_l216_216222

-- Definitions based on conditions
def total_plots := 200
def sample_plots := 20
def sum_x := 60
def sum_y := 1200
def sum_x_squared_diff := 80
def sum_y_squared_diff := 9000
def sum_xy_diff := 800

-- Estimated population calculation
noncomputable def estimated_population : ℕ :=
  (sum_y / sample_plots) * total_plots

theorem estimated_population_correct : estimated_population = 12000 := sorry

-- Correlation coefficient calculation
noncomputable def correlation_coefficient : Real :=
  sum_xy_diff / Real.sqrt (sum_x_squared_diff * sum_y_squared_diff)

theorem correlation_coefficient_correct : 
  Real.round (correlation_coefficient * 100) / 100 = 0.94 := sorry

end estimated_population_correct_correlation_coefficient_correct_l216_216222


namespace ratio_of_remaining_games_l216_216446

-- Define the problem conditions
def total_nintendo_games : ℕ := 20
def total_sony_games : ℕ := 70
def percentage_given_away_nintendo : ℚ := 0.40
def percentage_given_away_sony : ℚ := 0.30

-- Define what needs to be proved
theorem ratio_of_remaining_games :
  let remaining_nintendo_games := total_nintendo_games - (percentage_given_away_nintendo * total_nintendo_games).natAbs,
      remaining_sony_games := total_sony_games - (percentage_given_away_sony * total_sony_games).natAbs in
  remaining_nintendo_games = 12 ∧ remaining_sony_games = 49 ∧ (ratio remaining_nintendo_games remaining_sony_games = 12 / 49) :=
by
  -- sorry placeholder for the actual proof
  sorry

end ratio_of_remaining_games_l216_216446


namespace abs_complex_one_minus_i_l216_216852

-- We define the complex number and its components
def a : ℝ := 1
def b : ℝ := -1

-- Statement of the problem: the absolute value of the complex number (1 - i)
theorem abs_complex_one_minus_i : abs (complex.mk a b) = real.sqrt 2 := by sorry

end abs_complex_one_minus_i_l216_216852


namespace infinite_solutions_l216_216056

theorem infinite_solutions
  (k : ℕ) (hk : k ≥ 2)
  (p : ℕ → ℕ)
  (q : ℕ)
  (h_nodiv : ∀ i j, i ≠ j → (nat.gcd (p i) q = 1) ∧ (nat.gcd (p j) q = 1)):
    ∃ (X : ℕ → ℕ) (y : ℕ), (∀ i, i < k → (X i) ^ (p i) = y ^ q) :=
sorry

end infinite_solutions_l216_216056


namespace final_amount_simple_interest_l216_216969

theorem final_amount_simple_interest (P R T : ℕ) (hP : P = 12500) (hR : R = 6) (hT : T = 4) : 
  P + (P * R * T) / 100 = 13250 :=
by
  rw [hP, hR, hT]
  norm_num
  sorry

end final_amount_simple_interest_l216_216969


namespace evan_is_12th_l216_216409

def positions {α : Type} (L O C M N E : α) :=
  Lily_at_9 : L = 9 ∧
  Omar_ahead_of_Lily : O = L - 4 ∧
  Clara_behind_Omar : C = O + 3 ∧
  Mark_behind_Clara : M = C + 2 ∧
  Nina_behind_Mark : N = M + 3 ∧
  Evan_behind_Lily : E = L + 3

theorem evan_is_12th {α : Type} {L O C M N E : α} (h : positions L O C M N E) : E = 12 :=
by {
  obtain ⟨hL, hO, hC, hM, hN, hE⟩ := h,
  rw hL at hO hE,
  have : O = 5 := by linarith [hO],
  rw this at hC,
  have : C = 8 := by linarith [hC],
  rw this at hM,
  have : M = 10 := by linarith [hM],
  rw this at hN,
  have : N = 13 := by linarith [hN],
  linarith [hE]
}

end evan_is_12th_l216_216409


namespace hyperbola_eccentricity_l216_216704

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
(hyperbola_eq : ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)) 
(distance_condition : ∀ B : ℝ × ℝ, B = (0, b) → ∃ L : ℝ, L = b / 2 → 
  ∃ x y : ℝ, abs (b * x - a * y) / real.sqrt (a^2 + b^2) = L):
  (∃ e : ℝ, e = 2) := 
sorry

end hyperbola_eccentricity_l216_216704


namespace count_squares_in_region_l216_216722

theorem count_squares_in_region :
  let region := {p : ℝ × ℝ | let (x, y) := p in 
                                 y ≤ π * x ∧
                                 y ≥ -0.1 ∧
                                 x ≤ 5.1} in
  ∃ n : ℕ, n = 50 ∧
  (∀ (x1 y1 x2 y2 : ℤ), (x1:ℝ) ≤ 5 ∧ x2 = x1 + 1 ∧ y1 = 0 ∧ 
  y2 = y1 + 1 → (x2:ℝ, y2:ℝ) ∈ region) :=
begin
  sorry
end

end count_squares_in_region_l216_216722


namespace questions_left_blank_l216_216098

-- Definitions based on the conditions
def total_questions : Nat := 60
def word_problems : Nat := 20
def add_subtract_problems : Nat := 25
def algebra_problems : Nat := 10
def geometry_problems : Nat := 5
def total_time : Nat := 90

def time_per_word_problem : Nat := 2
def time_per_add_subtract_problem : Float := 1.5
def time_per_algebra_problem : Nat := 3
def time_per_geometry_problem : Nat := 4

def word_problems_answered : Nat := 15
def add_subtract_problems_answered : Nat := 22
def algebra_problems_answered : Nat := 8
def geometry_problems_answered : Nat := 3

-- The final goal is to prove that Steve left 12 questions blank
theorem questions_left_blank :
  total_questions - (word_problems_answered + add_subtract_problems_answered + algebra_problems_answered + geometry_problems_answered) = 12 :=
by
  sorry

end questions_left_blank_l216_216098


namespace average_growth_rate_price_reduction_june_l216_216788

-- Part (1)
theorem average_growth_rate {initial_sales March_sales May_sales : ℝ} (h1 : initial_sales = 192) (h2 : May_sales = 300) (h3 : 1 + x = sqrt ((May_sales / initial_sales))) :
  x = 0.25 := by
  sorry

-- Part (2)
theorem price_reduction_june {initial_cost initial_price sales_march sales_may profit : ℝ} (h1 : initial_cost = 40) (h2 : initial_price = 60) 
  (h3 : sales_march = 192) (h4 : sales_may = 300) (sales_increase_per_2yuan : ℝ) (m : ℝ) (price_reduction_units: ℝ) :
  price_reduction_units = 4 := by
  sorry

end average_growth_rate_price_reduction_june_l216_216788


namespace number_of_D_students_l216_216410

noncomputable def students : ℕ := 600
noncomputable def A_students : ℕ := students * 1 / 5
noncomputable def B_students : ℕ := students * 1 / 4
noncomputable def C_students : ℕ := students * 1 / 2
noncomputable def total_ABC_students := A_students + B_students + C_students
noncomputable def D_students := students - total_ABC_students

theorem number_of_D_students : D_students = 30 := 
by
  unfold students A_students B_students C_students total_ABC_students D_students
  sorry

end number_of_D_students_l216_216410


namespace max_distance_and_coverage_area_l216_216543

-- Define the conditions
def platform_center : ℝ := 0
def radar_count : ℕ := 5
def radar_coverage_radius : ℝ := 25
def coverage_ring_width : ℝ := 14
def max_distance_from_center_to_radar : ℝ := 24 / Real.sin (36 * Real.pi / 180)
def coverage_ring_area : ℝ := 672 * Real.pi / Real.tan (36 * Real.pi / 180)

-- Create the main statement for the proof problem
theorem max_distance_and_coverage_area :
  (∀ (P center : ℝ) (radars : ℕ) (r cov_width : ℝ),
    P = platform_center → 
    radars = radar_count → 
    r = radar_coverage_radius → 
    cov_width = coverage_ring_width →
    max_distance_from_center_to_radar = 24 / Real.sin (36 * Real.pi / 180) ∧
    coverage_ring_area = 672 * Real.pi / Real.tan (36 * Real.pi / 180)) :=
begin
  sorry -- Placeholder for proof
end

end max_distance_and_coverage_area_l216_216543


namespace print_time_including_warmup_l216_216171

def warmUpTime : ℕ := 2
def pagesPerMinute : ℕ := 15
def totalPages : ℕ := 225

theorem print_time_including_warmup :
  (totalPages / pagesPerMinute) + warmUpTime = 17 := by
  sorry

end print_time_including_warmup_l216_216171


namespace perpendicular_exists_l216_216360

-- Definitions for geometric entities involved

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  p1 : Point
  p2 : Point

structure Circle where
  center : Point
  radius : ℝ

-- Definitions for conditions in the problem

-- Condition 1: Point C is not on the circle
def point_not_on_circle (C : Point) (circle : Circle) : Prop :=
  (C.x - circle.center.x)^2 + (C.y - circle.center.y)^2 ≠ circle.radius^2

-- Condition 2: Point C is on the circle
def point_on_circle (C : Point) (circle : Circle) : Prop :=
  (C.x - circle.center.x)^2 + (C.y - circle.center.y)^2 = circle.radius^2

-- Definitions for lines and perpendicularity
def is_perpendicular (line1 : Line) (line2 : Line) : Prop :=
  (line1.p1.x - line1.p2.x) * (line2.p1.x - line2.p2.x) +
  (line1.p1.y - line1.p2.y) * (line2.p1.y - line2.p2.y) = 0

noncomputable def perpendicular_from_point_to_line (C : Point) (line : Line) (circle : Circle) 
  (h₁ : point_not_on_circle C circle ∨ point_on_circle C circle) : Line := 
  sorry

-- The Lean statement for part (a) and (b) combined into one proof.
theorem perpendicular_exists (C : Point) (lineAB : Line) (circle : Circle) 
  (h₁ : point_not_on_circle C circle ∨ point_on_circle C circle) : 
  ∃ (line_perpendicular : Line), is_perpendicular line_perpendicular lineAB ∧ 
  (line_perpendicular.p1 = C ∨ line_perpendicular.p2 = C) :=
  sorry

end perpendicular_exists_l216_216360


namespace average_growth_rate_price_reduction_june_l216_216787

-- Part (1)
theorem average_growth_rate {initial_sales March_sales May_sales : ℝ} (h1 : initial_sales = 192) (h2 : May_sales = 300) (h3 : 1 + x = sqrt ((May_sales / initial_sales))) :
  x = 0.25 := by
  sorry

-- Part (2)
theorem price_reduction_june {initial_cost initial_price sales_march sales_may profit : ℝ} (h1 : initial_cost = 40) (h2 : initial_price = 60) 
  (h3 : sales_march = 192) (h4 : sales_may = 300) (sales_increase_per_2yuan : ℝ) (m : ℝ) (price_reduction_units: ℝ) :
  price_reduction_units = 4 := by
  sorry

end average_growth_rate_price_reduction_june_l216_216787


namespace intersection_P_Q_l216_216373

def P : Set ℤ := {-4, -2, 0, 2, 4}
def Q : Set ℤ := {x : ℤ | -1 < x ∧ x < 3}

theorem intersection_P_Q : P ∩ Q = {0, 2} := by
  sorry

end intersection_P_Q_l216_216373


namespace min_value_of_16x_minus_4x_plus_1_l216_216297

noncomputable def find_min_value : ℝ :=
  let f (x : ℝ) := 16^x - 4^x + 1
  in Inf (set.range f)

theorem min_value_of_16x_minus_4x_plus_1 : find_min_value = 3/4 :=
  sorry

end min_value_of_16x_minus_4x_plus_1_l216_216297


namespace tree_height_when_planted_l216_216891

def initial_height (current_height : ℕ) (growth_rate : ℕ) (current_age : ℕ) (initial_age : ℕ) : ℕ :=
  current_height - (current_age - initial_age) * growth_rate

theorem tree_height_when_planted :
  initial_height 23 3 7 1 = 5 :=
by
  sorry

end tree_height_when_planted_l216_216891


namespace tan_identity_solution_l216_216320

theorem tan_identity_solution (x : ℝ) (hx : 0 < x ∧ x < 180) (hx_sol : x = 110) :
  tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x) :=
by sorry

end tan_identity_solution_l216_216320


namespace fraction_of_pages_read_l216_216825

-- Definition of the total number of pages per class
def pages_english : ℕ := 20
def pages_science : ℕ := 16
def pages_civics : ℕ := 8
def pages_chinese : ℕ := 12

-- Definition of the total pages Melody will read tomorrow
def total_pages_tomorrow : ℕ := 14

-- Prove that the fraction of the number of pages for each class Melody will read is 1/4
theorem fraction_of_pages_read :
  let f := (total_pages_tomorrow : ℚ) / (pages_english + pages_science + pages_civics + pages_chinese) in
  f = 1 / 4 :=
by
  sorry

end fraction_of_pages_read_l216_216825


namespace andy_l216_216982

theorem andy's_profit_per_cake :
  (∀ (cakes : ℕ), cakes = 2 → ∀ (ingredient_cost : ℕ), ingredient_cost = 12 →
                  ∀ (packaging_cost_per_cake : ℕ), packaging_cost_per_cake = 1 →
                  ∀ (selling_price_per_cake : ℕ), selling_price_per_cake = 15 →
                  ∀ (profit_per_cake : ℕ), profit_per_cake = selling_price_per_cake - (ingredient_cost / cakes + packaging_cost_per_cake) →
                    profit_per_cake = 8) :=
by
  sorry

end andy_l216_216982


namespace sum_a_2013_l216_216337

-- Define the sequence
def a : ℕ → ℕ
| 0       => 1
| 1       => 2
| 2       => 3
| n + 3   => 24 / (a n * a (n + 1) * a (n + 2))

-- Define the condition of the sequence
axiom a_sequence_condition (n : ℕ) : a n * a (n + 1) * a (n + 2) * a (n + 3) = 24

-- Define the sum we need to calculate
noncomputable def sum_a (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), a i

-- State the theorem
theorem sum_a_2013 : sum_a 2013 = 5031 :=
by
  sorry

end sum_a_2013_l216_216337


namespace initial_boxes_l216_216829

theorem initial_boxes (x : ℕ) (h : x + 6 = 14) : x = 8 :=
by sorry

end initial_boxes_l216_216829


namespace ratio_of_new_time_to_previous_time_l216_216585

noncomputable def distance : ℝ := 420
noncomputable def previous_time : ℝ := 7
noncomputable def speed_increase : ℝ := 40

-- Original speed
noncomputable def original_speed : ℝ := distance / previous_time

-- New speed
noncomputable def new_speed : ℝ := original_speed + speed_increase

-- New time taken to cover the same distance at the new speed
noncomputable def new_time : ℝ := distance / new_speed

-- Ratio of new time to previous time
noncomputable def time_ratio : ℝ := new_time / previous_time

theorem ratio_of_new_time_to_previous_time :
  time_ratio = 0.6 :=
by sorry

end ratio_of_new_time_to_previous_time_l216_216585


namespace james_jump_height_is_16_l216_216821

-- Define given conditions
def mark_jump_height : ℕ := 6
def lisa_jump_height : ℕ := 2 * mark_jump_height
def jacob_jump_height : ℕ := 2 * lisa_jump_height
def james_jump_height : ℕ := (2 * jacob_jump_height) / 3

-- Problem Statement to prove
theorem james_jump_height_is_16 : james_jump_height = 16 :=
by
  sorry

end james_jump_height_is_16_l216_216821


namespace no_natural_m_n_prime_l216_216267

theorem no_natural_m_n_prime (m n : ℕ) : ¬Prime (n^2 + 2018 * m * n + 2019 * m + n - 2019 * m^2) :=
by
  sorry

end no_natural_m_n_prime_l216_216267


namespace compounded_interest_difference_l216_216482

/-- Samantha's loan and interest problem:
  - Loan amount: $8000
  - Annual interest rate: 10%
  - Period: 3 years
  - Prove the difference in the owed amount when compounding monthly versus semiannually is $148.32
-/
def monthly_interest_rate := 0.10 / 12
def semiannual_interest_rate := 0.10 / 2
def loan_amount := 8000
def period_years := 3
def monthly_periods := 12 * period_years
def semiannual_periods := 2 * period_years
def monthly_compounded_amount := loan_amount * (1 + monthly_interest_rate)^monthly_periods
def semiannual_compounded_amount := loan_amount * (1 + semiannual_interest_rate)^semiannual_periods
def difference := monthly_compounded_amount - semiannual_compounded_amount

theorem compounded_interest_difference :
  abs (difference - 148.32) < 0.01 := 
sorry

end compounded_interest_difference_l216_216482


namespace num_ordered_triples_l216_216179

-- Given constants
def b : ℕ := 2024
def constant_value : ℕ := 4096576

-- Number of ordered triples (a, b, c) meeting the conditions
theorem num_ordered_triples (h : b = 2024 ∧ constant_value = 2024 * 2024) :
  ∃ (n : ℕ), n = 10 ∧ ∀ (a c : ℕ), a * c = constant_value → a ≤ c → n = 10 :=
by
  -- Translation of the mathematical conditions into the theorem
  sorry

end num_ordered_triples_l216_216179


namespace smaller_circle_radius_l216_216770

theorem smaller_circle_radius :
  ∀ (R r : ℝ), R = 10 ∧ (4 * r = 2 * R) → r = 5 :=
by
  intro R r
  intro h
  have h1 : R = 10 := h.1
  have h2 : 4 * r = 2 * R := h.2
  -- Use the conditions to eventually show r = 5
  sorry

end smaller_circle_radius_l216_216770


namespace parabola_circumcircle_fixed_point_l216_216608

theorem parabola_circumcircle_fixed_point 
  (a b : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < b) 
  (h3 : ∀ x, x^2 + a*x + b = 0 → x ≠ 0) :
  ∃ (k : Point) (A B C : Point), 
    (A.y = b) ∧ 
    (B.x * (B.x + a) + b = 0) ∧ 
    (C.x * (C.x + a) + b = 0) ∧ 
    (B.x ≠ C.x) ∧
    circumcircle A B C k :=
begin
  sorry
end

end parabola_circumcircle_fixed_point_l216_216608


namespace inradii_ratio_l216_216545

theorem inradii_ratio 
  (PQR : Triangle)
  (h_right : PQR.is_right_triangle)
  (PQ QR PR : ℝ)
  (h_PQ : PQ = 6)
  (h_QR : QR = 8)
  (h_PR : PR = 10)
  (S : Point)
  (h_S_on_PR : S ∈ PQR.side PR)
  (h_QS_bisects : PQR.angle_bisector S Q) :
  let r_p := inscribed_circle_radius (Triangle.mk PQR.P PQR.Q S)
  let r_q := inscribed_circle_radius (Triangle.mk PQR.Q PQR.R S)
  in r_p / r_q = by sorry :=
sorry

end inradii_ratio_l216_216545


namespace number_of_students_l216_216843

theorem number_of_students (left_pos right_pos total_pos : ℕ) 
  (h₁ : left_pos = 5) 
  (h₂ : right_pos = 3) 
  (h₃ : total_pos = left_pos - 1 + 1 + (right_pos - 1)) : 
  total_pos = 7 :=
by
  rw [h₁, h₂] at h₃
  simp at h₃
  exact h₃

end number_of_students_l216_216843


namespace triangle_inequality_l216_216417

-- Define the areas of the triangles
variables {E E_1 E_2 E_3 E_4 E_5 E_6 : ℝ}

-- Define the points in the triangle
variables {A B C K L M N R F : Type}

-- Assume the conditions given in the problem
axiom points_on_segments :
  (K ∈ line_segment B C) ∧
  (L ∈ line_segment A C) ∧
  (M ∈ line_segment A B) ∧
  (N ∈ line_segment L M) ∧
  (R ∈ line_segment M K) ∧
  (F ∈ line_segment K L)

-- Assume the given areas of the small triangles
axiom areas_of_triangles :
  area A M R = E_1 ∧
  area C K R = E_2 ∧
  area B K F = E_3 ∧
  area A L F = E_4 ∧
  area B N M = E_5 ∧
  area C L N = E_6 ∧
  area A B C = E

-- Final statement to prove the inequality
theorem triangle_inequality :
  E ≥ 8 * (E_1 * E_2 * E_3 * E_4 * E_5 * E_6)^(1/8) :=
sorry

end triangle_inequality_l216_216417


namespace range_CA_dot_CB_l216_216707

section problem

-- Define the parabola M: x^2 = 4y
def parabola_M (x y : ℝ) : Prop := x^2 = 4 * y

-- Define the circle C: x^2 + (y - 3)^2 = 4
def circle_C (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 4

-- Point P(x0, x0^2/4) lies on the parabola M
def point_P (x0 y : ℝ) : Prop := parabola_M x0 y ∧ y = x0^2 / 4

-- Function to calculate the dot product of vectors CA and CB based on P's position
noncomputable def CA_dot_CB (x0 : ℝ) : ℝ := 
  let y0 := x0^2 / 4
  let CP_square := x0^2 + (y0 - 3)^2
  4 * Math.cos (Real.sqrt ((CP_square - 8) / 4))

-- The range of values for CA • CB given P on the parabola and tangents PA, PB to the circle
theorem range_CA_dot_CB 
  (x0 : ℝ) 
  (hx0 : parabola_M x0 (x0^2 / 4)) : 
  0 ≤ CA_dot_CB x0 ∧ CA_dot_CB x0 < 4 := 
sorry

end problem

end range_CA_dot_CB_l216_216707


namespace faye_initial_coloring_books_l216_216652

theorem faye_initial_coloring_books (gave_away1 gave_away2 remaining : ℝ) 
    (h1 : gave_away1 = 34.0) (h2 : gave_away2 = 3.0) (h3 : remaining = 11.0) :
    gave_away1 + gave_away2 + remaining = 48.0 := 
by
  sorry

end faye_initial_coloring_books_l216_216652


namespace mean_score_seniors_is_correct_l216_216050

-- Constants and given information
constant num_students : ℕ := 120
constant avg_score : ℝ := 110
constant ratio_non_seniors_to_seniors : ℝ := 1.4
constant ratio_seniors_to_non_seniors_scores : ℝ := 1.6

-- Variables
variables (s n : ℝ) (m_s m_n : ℝ)

-- Definitions from conditions
def total_students (s n : ℝ) : ℝ := s + n
def total_score : ℝ := num_students * avg_score
def num_non_seniors (s : ℝ) : ℝ := ratio_non_seniors_to_seniors * s
def senior_score_ratio : ℝ := 1.6

-- Theorem statement
theorem mean_score_seniors_is_correct :
  (s + num_non_seniors s = num_students) ∧
  (total_students s (num_non_seniors s) = num_students) ∧
  (50 * m_s + 70 * m_n = total_score) ∧
  (m_s = senior_score_ratio * m_n) →
  m_s = 140.8 :=
sorry

end mean_score_seniors_is_correct_l216_216050


namespace exists_n_more_good_than_bad_l216_216060

def is_good (n : ℕ) : Prop := 
  (n.digits 10).sum % 2 = 0

def num_digits (m : ℕ) (n : ℕ) : Prop := 
  10^(n-1) <= m ∧ m < 10^n

theorem exists_n_more_good_than_bad :
  ∃ n : ℕ, let a := (10^(n-1) to 10^n) in
           ∑ (m : ℕ) in a, (if is_good m then 1 else 0) > (∑ (m : ℕ) in a, (if ¬ is_good m then 1 else 0)) :=
by
  sorry

end exists_n_more_good_than_bad_l216_216060


namespace find_x2011_l216_216695

-- Definitions for the conditions
variable {f : ℝ → ℝ}
variable {x : ℕ → ℝ}
variable (a₀ : ℝ) (d := 2)

-- Define increasing and odd function
axiom h1 : ∀ {a b : ℝ}, a < b → f(a) < f(b)
axiom h2 : ∀ {x : ℝ}, f(-x) = -f(x)

-- Definition of the arithmetic sequence
def a (n : ℕ) : ℝ := a₀ + n * d

-- Given condition of the sequence sums
axiom h3 : f(a 8) + f(a 9) + f(a 10) + f(a 11) = 0

-- The goal to prove
theorem find_x2011 : a 2011 = 4003 := by
  sorry

end find_x2011_l216_216695


namespace estimated_population_correct_correlation_coefficient_correct_l216_216206

noncomputable def estimated_population
  (num_plots : ℕ) (sample_size : ℕ) 
  (sum_y : ℕ) : ℕ :=
(sum_y / sample_size) * num_plots

noncomputable def correlation_coefficient
  (sum_x_sq : ℝ) (sum_y_sq : ℝ) 
  (sum_xy : ℝ) : ℝ :=
sum_xy / real.sqrt(sum_x_sq * sum_y_sq)

theorem estimated_population_correct
  (num_plots : ℕ := 200) (sample_size : ℕ := 20) 
  (sum_y : ℕ := 1200) : estimated_population num_plots sample_size sum_y = 12000 := by
sorry

theorem correlation_coefficient_correct
  (sum_x_sq : ℝ := 80) (sum_y_sq : ℝ := 9000) 
  (sum_xy : ℝ := 800) : real.abs (correlation_coefficient sum_x_sq sum_y_sq sum_xy - 0.94) < 0.01 := by
sorry

end estimated_population_correct_correlation_coefficient_correct_l216_216206


namespace coeff_x3_in_product_l216_216904

def P (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 + 4 * x^2 - 3 * x + 2
def Q (x : ℝ) : ℝ := x^2 - 4 * x + 3

theorem coeff_x3_in_product (x : ℝ) : coeff_x3 (P x * Q x) = -25 :=
sorry

end coeff_x3_in_product_l216_216904


namespace P_neg2_eq_19_l216_216023

noncomputable def P : ℝ → ℝ := sorry  -- Definition of the polynomial P(x)

axiom degree_P : ∃ (d : ℕ), d = 4 ∧ ∀ (x : ℝ) (hx : P x ≠ 0), nat_degree (polynomial.of_real (P x)) = d
axiom P_0 : P 0 = 1
axiom P_1 : P 1 = 1
axiom P_2 : P 2 = 4
axiom P_3 : P 3 = 9
axiom P_4 : P 4 = 16

theorem P_neg2_eq_19 : P (-2) = 19 :=
by
  sorry

end P_neg2_eq_19_l216_216023


namespace big_island_counties_odd_l216_216851

-- Define conditions for the problem
def is_rectangular_island (n : ℕ) : Prop :=
  ∃ rows cols, rows * cols = n

def road_in_diagonal (county : ℕ) : Prop :=
  true -- all counties have a diagonal road

def closed_non_intersecting_path (island : Set ℕ) : Prop :=
  true -- closed path property assumed for simplicity

-- Define the main theorem
theorem big_island_counties_odd {n : ℕ} (h1 : is_rectangular_island n)
  (h2 : closed_non_intersecting_path (Set.range 6)) (h3 : n % 2 = 1) :
  n = 9 :=
sorry

end big_island_counties_odd_l216_216851


namespace scientific_notation_280000_l216_216048

theorem scientific_notation_280000 : 
  ∃ n: ℝ, n * 10^5 = 280000 ∧ n = 2.8 :=
by
-- our focus is on the statement outline, thus we use sorry to skip the proof part
  sorry

end scientific_notation_280000_l216_216048


namespace only_natural_number_n4_plus_4n_prime_l216_216287

open Nat

theorem only_natural_number_n4_plus_4n_prime (n : ℕ) (h : Prime (n^4 + 4^n)) : (n = 1) :=
sorry

end only_natural_number_n4_plus_4n_prime_l216_216287


namespace malcolm_followers_l216_216037

theorem malcolm_followers :
  let instagram_followers := 240
  let facebook_followers := 500
  let twitter_followers := (instagram_followers + facebook_followers) / 2
  let tiktok_followers := 3 * twitter_followers
  let youtube_followers := tiktok_followers + 510
  instagram_followers + facebook_followers + twitter_followers + tiktok_followers + youtube_followers = 3840 :=
by {
  sorry
}

end malcolm_followers_l216_216037


namespace find_radius_l216_216709

-- Definitions of parabola and circle with the given conditions.
def parabola (x : ℝ) : ℝ := (1/4) * x^2
def circle_center : ℝ × ℝ := (1, 2)
def radius (r : ℝ) : ℝ := r
def circle (x y r : ℝ) : Prop := (x - 1) ^ 2 + (y - 2) ^ 2 = r ^ 2

-- Prove that given the common point P and conditions, the radius r is sqrt(2).
theorem find_radius (r : ℝ) (P : ℝ × ℝ) 
  (hPparabola : P.snd = parabola P.fst) 
  (hPcircle : circle P.fst P.snd r) 
  (h_tangent : ∀ k kPC, k = (1/2) * P.fst → kPC = (P.snd - 2)/(P.fst - 1) → k * kPC = -1): 
  r = Real.sqrt 2 := 
sorry -- Proof is omitted.

end find_radius_l216_216709


namespace orange_price_l216_216535

theorem orange_price (initial_apples : ℕ) (initial_oranges : ℕ) 
                     (apple_price : ℝ) (total_earnings : ℝ) 
                     (remaining_apples : ℕ) (remaining_oranges : ℕ)
                     (h1 : initial_apples = 50) (h2 : initial_oranges = 40)
                     (h3 : apple_price = 0.80) (h4 : total_earnings = 49)
                     (h5 : remaining_apples = 10) (h6 : remaining_oranges = 6) :
  ∃ orange_price : ℝ, orange_price = 0.50 :=
by
  sorry

end orange_price_l216_216535


namespace estimate_population_correct_correlation_coefficient_correct_l216_216213

namespace EcologicalSurvey

def estimated_population (total_animals : ℕ) (plots : ℕ) : ℕ :=
  (total_animals / 20) * plots

theorem estimate_population_correct :
  estimated_population 1200 200 = 12000 :=
by 
  -- Proof would go here.
  sorry

def correlation_coefficient (sum_xi_xbar : ℝ) (sum_yi_ybar : ℝ) (sum_prod_xi_yi : ℝ) : ℝ :=
  sum_prod_xi_yi / Math.sqrt (sum_xi_xbar * sum_yi_ybar)

theorem correlation_coefficient_correct :
  correlation_coefficient 80 9000 800 ≈ 0.94 :=
by 
  -- Proof would go here.
  sorry

end EcologicalSurvey

end estimate_population_correct_correlation_coefficient_correct_l216_216213


namespace fraction_negative_iff_x_lt_2_l216_216743

theorem fraction_negative_iff_x_lt_2 (x : ℝ) :
  (-5) / (2 - x) < 0 ↔ x < 2 := by
  sorry

end fraction_negative_iff_x_lt_2_l216_216743


namespace problem1_problem2_l216_216650

-- Proof Problem 1: Prove the given expression evaluates to 27/16
theorem problem1 :
  ((0.064 : ℝ) ^ (-1 / 3)) - ((-5 / 9 : ℝ) ^ 0) + ((-2 : ℝ) ^ 3) ^ (-4 / 3) + (16 : ℝ) ^ (-0.75) = 27 / 16 :=
by 
  sorry

-- Proof Problem 2: Given 3^x = 4^y = 36, prove 2/x + 1/y = 1
theorem problem2 (x y : ℝ) (hx : 3 ^ x = 36) (hy : 4 ^ y = 36) :
  2 / x + 1 / y = 1 :=
by 
  sorry

end problem1_problem2_l216_216650


namespace min_value_16x_minus_4x_plus_1_l216_216300

theorem min_value_16x_minus_4x_plus_1 :
  ∀ x : ℝ, 16 ^ x - 4 ^ x + 1 ≥ 3 / 4 ∧ (∃ y : ℝ, 16 ^ y - 4 ^ y + 1 = 3 / 4) :=
by
  sorry

end min_value_16x_minus_4x_plus_1_l216_216300


namespace percentage_of_french_speaking_employees_l216_216746

theorem percentage_of_french_speaking_employees :
  (total_employees men french_speaking_men women french_speaking_women : ℕ)
  (P_total : total_employees = 100)
  (P_men : men = 65% of total_employees)
  (P_women : women = total_employees - men)
  (P_french_speaking_men : french_speaking_men = 60% of men)
  (P_french_speaking_women : french_speaking_women = 2.85714285714286% of women)
  (P_french_speaking_total : french_speaking_total = french_speaking_men + french_speaking_women)
  : french_speaking_total / total_employees * 100 = 40 :=
by
  sorry

end percentage_of_french_speaking_employees_l216_216746


namespace domain_implies_range_l216_216402

noncomputable def f (x a : ℝ) := real.sqrt (2^(x^2 + 2*a*x - a) - 1)

theorem domain_implies_range (a : ℝ) :
  (∀ x : ℝ, 2^(x^2 + 2*a*x - a) - 1 ≥ 0) ↔ (-1 ≤ a ∧ a ≤ 0) :=
by
  sorry

end domain_implies_range_l216_216402


namespace find_value_of_a20_l216_216343

variable {α : Type*} [LinearOrder α] [Field α]

def arithmetic_sequence (a d : α) (n : ℕ) : α :=
  a + (n - 1) * d

def arithmetic_sum (a d : α) (n : ℕ) : α :=
  n * a + (n * (n - 1) / 2) * d

theorem find_value_of_a20 
  (a d : ℝ) 
  (h1 : arithmetic_sequence a d 3 + arithmetic_sequence a d 5 = 4)
  (h2 : arithmetic_sum a d 15 = 60) :
  arithmetic_sequence a d 20 = 10 := 
sorry

end find_value_of_a20_l216_216343


namespace function_characterization_l216_216639

noncomputable def g (x : ℝ) : ℝ := x^3 + x

noncomputable def g_inv (y : ℝ) : ℝ := sorry  -- Definition of g_inv

lemma g_strictly_increasing : ∀ x y : ℝ, x < y → g x < g y :=
begin
  intros x y h,
  sorry,
end

lemma g_bijective : function.bijective g :=
begin
  split,
  { intros a b h,
    sorry, },  -- Injective proof
  { sorry, }  -- Surjective proof
end

lemma g_inv_property : ∀ x, g (g_inv x) = x ∧ g_inv (g x) = x :=
begin
  sorry,  -- Property of the inverse function
end

def f (x : ℝ) : ℝ := g_inv x

theorem function_characterization (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (x^3 + x) ≤ x ∧ x ≤ (f x)^3 + f x)
  : ∀ x : ℝ, f x = g_inv x :=
begin
  intro x,
  have h2 := h1 x,
  sorry,  -- Proof that f(x) = g_inv(x)
end

end function_characterization_l216_216639


namespace Craig_walk_distance_l216_216637

/-- Craig walked some distance from school to David's house and 0.7 miles from David's house to his own house. 
In total, Craig walked 0.9 miles. Prove that the distance Craig walked from school to David's house is 0.2 miles. 
--/
theorem Craig_walk_distance (d_school_David d_David_Craig d_total : ℝ) 
  (h1 : d_David_Craig = 0.7) 
  (h2 : d_total = 0.9) : 
  d_school_David = 0.2 :=
by 
  sorry

end Craig_walk_distance_l216_216637


namespace derivative_value_at_two_l216_216327

theorem derivative_value_at_two :
  ∃ (f : ℝ → ℝ), (∀ x, f x = 3 * x^2 - 2 * x * (deriv f 2)) → (deriv f 2 = 4) :=
begin
  let f : ℝ → ℝ := λ x, 3 * x^2 - 2 * x * (deriv f 2),
  use f,
  intros h,
  have h1 : deriv f = λ x, 6 * x - 2 * (deriv f 2), by { sorry },
  have h2 : deriv f 2 = 12 - 2 * (deriv f 2), by { apply h1, },
  have h3 : deriv f 2 + 2 * deriv f 2 = 12, by linarith [h2],
  have h4 : 3 * deriv f 2 = 12, by linarith [h3],
  sorry,
end

end derivative_value_at_two_l216_216327


namespace range_of_b_over_a_l216_216755

variables {A B C : ℝ} {a b c : ℝ}

/--
In an acute triangle ABC, the sides opposite to angles A, B, and C are a, b, and c respectively.
Given B = 2A, prove that sqrt(2) < b / a < sqrt(3).
-/
theorem range_of_b_over_a (h₁ : B = 2 * A) (h₂ : A > 0) (h₃ : 2 * A < π / 2) (h₄ : a > 0) (h₅ : b > 0) :
  sqrt 2 < b / a ∧ b / a < sqrt 3 :=
sorry

end range_of_b_over_a_l216_216755


namespace minor_premise_is_wrong_l216_216897

theorem minor_premise_is_wrong (a : ℝ) : ¬ (0 < a^2) := by
  sorry

end minor_premise_is_wrong_l216_216897


namespace probability_of_rolling_2_4_or_8_l216_216913

theorem probability_of_rolling_2_4_or_8 :
  let total_outcomes := 8 in
  let favorable_outcomes := 3 in
  (favorable_outcomes / total_outcomes : ℚ) = 3 / 8 :=
by
  sorry

end probability_of_rolling_2_4_or_8_l216_216913


namespace find_angle_B_max_S_over_l_l216_216439

variable {a b c S l : ℝ}

-- Given conditions
def triangle_conditions (a b c S : ℝ) :=
  ∃ (A B C : ℝ), (A + B + C = π) ∧ 
                 (S = 1/2 * a * b * sin C) ∧
                 (4 * S / tan B = a^2 * cos B + a * b * cos A)

-- Part 1: Prove angle B
theorem find_angle_B (a b c S : ℝ) (h : triangle_conditions a b c S) : ∃ B : ℝ, B = π / 3 :=
  sorry

-- Additional conditions for Part 2
def additional_conditions (a c l : ℝ) :=
  (a + b + c = l) ∧ (b = 3)

-- Part 2: Prove maximum value of S/l
theorem max_S_over_l (a c l S : ℝ) (h1 : triangle_conditions a b c S) (h2 : additional_conditions a c l) : 
  S / l ≤ sqrt 3 / 4 :=
  sorry

end find_angle_B_max_S_over_l_l216_216439


namespace total_population_l216_216527

variable (seattle lake_view boise : ℕ)

def lake_view_pop := (lake_view = 24000)
def seattle_pop := (seattle = lake_view - 4000)
def boise_pop := (boise = (3 / 5 : ℚ) * seattle)

theorem total_population (h1 : lake_view_pop) (h2 : seattle_pop) (h3 : boise_pop) :
  lake_view + seattle + boise = 56000 :=
by
  sorry

end total_population_l216_216527


namespace not_two_to_one_ratio_l216_216607

-- Define the triangle and angles
structure Triangle :=
(A B C : Point)
(angle : A ∠ B C)
(angle_b : B ∠ A C)
(angle_c : C ∠ A B)
(sum_angles : angle + angle_b + angle_c = 180)
(right_angle : angle_c = 90)
(half_angle : angle = 30)
(double_angle : angle_b = 2 * angle)

-- Proving the counterexample where sides opposite the double angles do not have a 2:1 ratio.
theorem not_two_to_one_ratio :
  ∃ (T : Triangle), ¬(side opposite angle := side opposite double_angle * 2) :=
by
  -- Construct the point, triangle and show the sides do not fulfill the 2:1 ratio
  sorry

end not_two_to_one_ratio_l216_216607


namespace imaginary_part_of_z_l216_216502

theorem imaginary_part_of_z :
  let z := (1 : ℂ) + Complex.i in
  z.im = 1 := by
  let z := (1 : ℂ) + Complex.i
  show z.im = 1
  sorry

end imaginary_part_of_z_l216_216502


namespace michael_latest_sprint_time_l216_216826

theorem michael_latest_sprint_time (times : List Nat) (new_time : ℚ) :
  times = [102, 107, 108, 109, 110, 115] →
  (List.insert times new_time).median = 107.5 →
  new_time = 107.5 :=
by
  intros h_times h_median
  sorry

end michael_latest_sprint_time_l216_216826


namespace value_of_fourth_set_l216_216264

def value_in_set (a b c d : ℕ) : ℕ :=
  (a * b * c * d) - (a + b + c + d)

theorem value_of_fourth_set :
  value_in_set 1 5 6 7 = 191 :=
by
  sorry

end value_of_fourth_set_l216_216264


namespace estimated_population_correct_correlation_coefficient_correct_l216_216221

-- Definitions based on conditions
def total_plots := 200
def sample_plots := 20
def sum_x := 60
def sum_y := 1200
def sum_x_squared_diff := 80
def sum_y_squared_diff := 9000
def sum_xy_diff := 800

-- Estimated population calculation
noncomputable def estimated_population : ℕ :=
  (sum_y / sample_plots) * total_plots

theorem estimated_population_correct : estimated_population = 12000 := sorry

-- Correlation coefficient calculation
noncomputable def correlation_coefficient : Real :=
  sum_xy_diff / Real.sqrt (sum_x_squared_diff * sum_y_squared_diff)

theorem correlation_coefficient_correct : 
  Real.round (correlation_coefficient * 100) / 100 = 0.94 := sorry

end estimated_population_correct_correlation_coefficient_correct_l216_216221


namespace find_k_l216_216423

-- Define z as being a complex number
variable {z : ℂ}

-- Condition definitions
def cond1 (z : ℂ) : Prop := |z - 2| = |z + 2|
def cond2 (z : ℂ) : Prop := |z - complex.I| = 3

-- Solution statement
theorem find_k : ∃ k : ℝ, (∀ z, cond1 z ∧ cond2 z → (|z| = k) ↔ (k = 1)) :=
by 
  sorry

end find_k_l216_216423


namespace total_amount_spent_l216_216443

theorem total_amount_spent (tax_paid : ℝ) (tax_rate : ℝ) (tax_free_cost : ℝ) (total_spent : ℝ) :
  tax_paid = 30 → tax_rate = 0.06 → tax_free_cost = 19.7 →
  total_spent = 30 / 0.06 + 19.7 :=
by
  -- Definitions for assumptions
  intro h1 h2 h3
  -- Skip the proof here
  sorry

end total_amount_spent_l216_216443


namespace count_roots_on_unit_circle_l216_216251

-- Define the polynomial f(z)
noncomputable def f (z : Complex) : Complex :=
  1 + z^5 + z^{10} + z^{15} + z^{18} + z^{21} + z^{24} + z^{27}

-- Main goal statement
theorem count_roots_on_unit_circle : 
  {z : Complex | Complex.abs z = 1 ∧ f z = 0}.toFinset.card = 11 := by
  sorry

end count_roots_on_unit_circle_l216_216251


namespace sequence_formula_lambda_range_l216_216676

variable {a : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable {b : ℕ → ℤ}
variable {q : ℝ} (λ : ℝ)

theorem sequence_formula (q_pos : q > 0) (a1 : a 1 = 1)
  (Sn_recurrence : ∀ n : ℕ, S (n + 1) = q * S n + 1)
  (arith_seq_cond : 2 * a 2 = a 3 + a 2 + 2) :
  ∀ n : ℕ, a n = 2 ^ (n - 1) := by
  sorry

theorem lambda_range (lambda_increasing : ∀ n : ℕ, b (n + 1) - b n > 0):
  λ < 2 / 3 := by
  sorry

end sequence_formula_lambda_range_l216_216676


namespace find_width_l216_216505

def length : ℝ := 5.5
def cost_per_sqm : ℝ := 1400
def total_cost : ℝ := 28875
def area : ℝ := total_cost / cost_per_sqm

def width : ℝ := area / length

theorem find_width : width = 3.75 := by
  sorry

end find_width_l216_216505


namespace count_correct_statements_l216_216765

theorem count_correct_statements :
  ∃ (M: ℚ) (M1: ℚ) (M2: ℚ) (M3: ℚ) (M4: ℚ)
    (a b c d e : ℚ) (hacb : c ≠ 0) (habc: a ≠ 0) (hbcb : b ≠ 0) (hdcb: d ≠ 0) (hec: e ≠ 0),
  M = (ac + bd - ce) / c 
  ∧ M1 = (-bc - ad - ce) / c 
  ∧ M2 = (-dc - ab - ce) / c 
  ∧ M3 = (-dc - ab - de) / d 
  ∧ M4 = (ce - bd - ac) / (-c)
  ∧ M4 = M
  ∧ (M ≠ M3)
  ∧ (∀ M1, M1 = (-bc - ad - ce) / c → ((a = c ∨ b = d) ↔ b = d))
  ∧ (M4 = (ac + bd - ce)/c) :=
sorry

end count_correct_statements_l216_216765


namespace platform_length_l216_216603

noncomputable def length_of_train : ℝ := 360
noncomputable def speed_of_train_kmh : ℝ := 45
noncomputable def time_taken : ℝ := 39.2

noncomputable def speed_of_train_ms : ℝ := speed_of_train_kmh * (1000 / 3600)
noncomputable def total_distance : ℝ := speed_of_train_ms * time_taken
noncomputable def length_of_platform : ℝ := total_distance - length_of_train

theorem platform_length :
  length_of_platform = 130 :=
by
  unfold length_of_train speed_of_train_kmh time_taken speed_of_train_ms total_distance length_of_platform
  sorry

end platform_length_l216_216603


namespace length_of_train_equals_450_09_l216_216569

-- Define the speed of the train in km/h
def speed_kmh : ℝ := 60

-- Convert the speed to m/s
def speed_ms : ℝ := speed_kmh * 1000 / 3600

-- Define the time in seconds
def time_sec : ℝ := 27

-- Define the length of the train
def length_of_train : ℝ := speed_ms * time_sec

-- The theorem to prove
theorem length_of_train_equals_450_09 : length_of_train = 450.09 := by
    sorry

end length_of_train_equals_450_09_l216_216569


namespace angle_ACB_is_30_l216_216436

theorem angle_ACB_is_30 (A B C D E F : Type) (h1 : Triangle A B C)
    (h2 : SideLength A B = 3 * SideLength A C)
    (h3 : ∃ D ∈ Segment A B, ∃ E ∈ Segment B C, Angle B A E = Angle A C D)
    (h4 : F ∈ Intersection (Line A E) (Line C D))
    (h5 : EquilateralTriangle C F E) :
  Angle A C B = 30 := by
  sorry

end angle_ACB_is_30_l216_216436


namespace percent_increase_l216_216990

def initial_price : ℝ := 15
def final_price : ℝ := 16

theorem percent_increase : ((final_price - initial_price) / initial_price) * 100 = 6.67 :=
by
  sorry

end percent_increase_l216_216990


namespace irrational_number_l216_216132

-- Definitions of the given numbers
def num_A : ℚ := 3.14
def num_B : ℚ := 22 / 7
def num_C : ℝ := Real.sqrt 2
def num_D : ℚ := 0

-- Statement of the problem
theorem irrational_number :
  ¬ ∃ (a b : ℤ), b ≠ 0 ∧ num_C = a / b :=
sorry

end irrational_number_l216_216132


namespace max_of_four_expressions_l216_216976

theorem max_of_four_expressions :
  996 * 996 > 995 * 997 ∧ 996 * 996 > 994 * 998 ∧ 996 * 996 > 993 * 999 :=
by
  sorry

end max_of_four_expressions_l216_216976


namespace minimum_value_of_expression_l216_216307

theorem minimum_value_of_expression : ∃ x : ℝ, (16 ^ x - 4 ^ x + 1 = 3 / 4) :=
sorry

end minimum_value_of_expression_l216_216307


namespace intersection_condition_l216_216315

-- Define the lines
def line1 (x y : ℝ) := 2*x - 2*y - 3 = 0
def line2 (x y : ℝ) := 3*x - 5*y + 1 = 0
def line (a b x y : ℝ) := a*x - y + b = 0

-- Define the condition
def condition (a b : ℝ) := 17*a + 4*b = 11

-- Prove that the line l passes through the intersection point of l1 and l2 if and only if the condition holds
theorem intersection_condition (a b : ℝ) :
  (∃ x y : ℝ, line1 x y ∧ line2 x y ∧ line a b x y) ↔ condition a b :=
  sorry

end intersection_condition_l216_216315


namespace prob_interval_l216_216465

noncomputable def ξ : Type := sorry -- representing the random variable
noncomputable def µ : ℝ := sorry   -- mean
noncomputable def σ : ℝ := sorry   -- standard deviation

axiom Xi_normal : ξ ∼ Normal(µ, σ^2) -- ξ is normally distributed with mean µ and variance σ^2
axiom prob_neg3 : P(ξ < -3) = 0.2 -- Given condition P(ξ < -3) = 0.2
axiom prob_1 : P(ξ > 1) = 0.2 -- Given condition P(ξ > 1) = 0.2

theorem prob_interval : P(-1 < ξ < 1) = 0.3 := by
  sorry

end prob_interval_l216_216465


namespace inradius_altitudes_relation_l216_216376

variable {t r s a b c m₁ m₂ m₃ : ℝ}
variable {A B C : Type} [triangle : Triangle A B C]
variable (altitudes : altitude A B C m₁ m₂ m₃)
variable (inradius : inradius A B C r)

theorem inradius_altitudes_relation :
  (1 / r) = (1 / m₁) + (1 / m₂) + (1 / m₃) :=
sorry

end inradius_altitudes_relation_l216_216376


namespace quadratic_root_shift_l216_216804

theorem quadratic_root_shift (b c : ℝ) :
  (∀ r s : ℝ, r * s = -9 / 5 ∧ r + s = 4 / 5 →
    ∃ r' s', r' = r - 3 ∧ s' = s - 3 ∧ 
    r' + s' = (4 / 5) - 6 ∧ r' * s' = 24 / 5 ∧
    c = 24 / 5) :=
begin
  sorry
end

end quadratic_root_shift_l216_216804


namespace number_of_primes_l216_216657

theorem number_of_primes (P : ℕ → Prop) :
  (∀ p : ℕ, P p ↔ (p.prime ∧ 
  ¬(∃ r1 r2 r3 : ℤ, 0 ≤ r1 ∧ r1 < p ∧ 0 ≤ r2 ∧ r2 < p ∧ 0 ≤ r3 ∧ r3 < p ∧ 
  r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ 
  (r1^3 - 5 * r1^2 - 22 * r1 + 56) % p = 0 ∧ 
  (r2^3 - 5 * r2^2 - 22 * r2 + 56) % p = 0 ∧ 
  (r3^3 - 5 * r3^2 - 22 * r3 + 56) % p = 0))) → 
  (∃ p_set : Finset ℕ, p_set.card = 4 ∧ ∀ p ∈ p_set, P p) :=
by {
  sorry
}

end number_of_primes_l216_216657


namespace original_cost_of_statue_l216_216572

theorem original_cost_of_statue (S : ℝ) (r : ℝ) (C : ℝ) (h1 : S = $540) (h2 : r = 0.35) (h3 : S = C * (1 + r)) : C = 400 := by
  sorry

end original_cost_of_statue_l216_216572


namespace estimated_population_correlation_coefficient_l216_216230

variables (xi yi : Fin 20 → ℝ)       -- xi represents plant coverage areas, yi represents number of wild animals
variables (sum_x : ℝ) (sum_y : ℝ) (sum_x_diff_sq : ℝ) (sum_y_diff_sq : ℝ) (sum_xy_diff : ℝ)

-- Conditions
def conditions := 
  (∑ i, xi i = 60) ∧ 
  (∑ i, yi i = 1200) ∧ 
  (∑ i, (xi i - (∑ j, xi j) / 20) ^ 2 = 80) ∧ 
  (∑ i, (yi i - (∑ j, yi j) / 20) ^ 2 = 9000) ∧ 
  (∑ i, (xi i - (∑ j, xi j) / 20) * (yi i - (∑ j, yi j) / 20) = 800)

-- Part 1: Prove the estimated population
theorem estimated_population (h : conditions xi yi sum_x sum_y sum_x_diff_sq sum_y_diff_sq sum_xy_diff) :
  200 * (sum_y / 20) = 12000 :=
by sorry

-- Part 2: Prove the correlation coefficient
theorem correlation_coefficient (h : conditions xi yi sum_x sum_y sum_x_diff_sq sum_y_diff_sq sum_xy_diff) :
  (sum_xy_diff / (real.sqrt (sum_x_diff_sq * sum_y_diff_sq))).round_to 2 = 0.94 :=
by sorry

end estimated_population_correlation_coefficient_l216_216230


namespace gcd_of_distinct_terms_is_one_sum_of_reciprocals_of_sequence_equals_one_l216_216146

-- Define the sequence {a_n}
def a : ℕ → ℕ
| 0     := 2
| (n+1) := (a n)^2 - (a n) + 1

-- Statement for Part 1
theorem gcd_of_distinct_terms_is_one {m n : ℕ} (h : m ≠ n) : Nat.gcd (a m) (a n) = 1 :=
sorry

-- Statement for Part 2
theorem sum_of_reciprocals_of_sequence_equals_one : 
  ∑' k, 1 / (a (k+1)) = 1 :=
sorry

end gcd_of_distinct_terms_is_one_sum_of_reciprocals_of_sequence_equals_one_l216_216146


namespace parallelogram_lengths_l216_216672

noncomputable def parallelogram_data 
  (ABCD : Type) 
  (A B C D M E K : ABCD) 
  (diameter : ℝ) 
  (EM_length : ℝ) 
  (arc_AE : ℝ) 
  (arc_BM : ℝ) 
  (Ω : Type) :=
  diameter = 13 ∧
  EM_length = 12 ∧
  arc_AE = 2 * arc_BM ∧
  (Ω circumscribe (triangle ABC M))

theorem parallelogram_lengths 
  (ABCD : Type) 
  (A B C D M E K : ABCD) 
  (diameter : ℝ) 
  (EM_length : ℝ) 
  (arc_AE : ℝ) 
  (arc_BM : ℝ) 
  (Ω : Type)
  (h_parallelogram : parallelogram_data ABCD A B C D M E K diameter EM_length arc_AE arc_BM Ω):
  BC = 13 ∧ BK = 120 / 13 ∧ (AK + KM + MA) = 340 / 13 := sorry

end parallelogram_lengths_l216_216672


namespace train_speed_is_36_kmph_l216_216189

-- Define the parameters: length of the train, length of the bridge, and time to cross
def train_length : ℝ := 140
def bridge_length : ℝ := 150
def time_to_cross : ℝ := 28.997680185585153

-- Define the total distance covered by the train (train length + bridge length)
def total_distance : ℝ := train_length + bridge_length

-- Define the speed of the train in m/s (total distance / time to cross)
def speed_mps : ℝ := total_distance / time_to_cross

-- Define the speed of the train in kmph (speed in m/s * 3.6)
def speed_kmph : ℝ := speed_mps * 3.6

-- The theorem to prove that the speed of the train in kmph is 36
theorem train_speed_is_36_kmph : speed_kmph = 36 := 
sorry

end train_speed_is_36_kmph_l216_216189


namespace estimated_population_correlation_coefficient_l216_216225

variables (xi yi : Fin 20 → ℝ)       -- xi represents plant coverage areas, yi represents number of wild animals
variables (sum_x : ℝ) (sum_y : ℝ) (sum_x_diff_sq : ℝ) (sum_y_diff_sq : ℝ) (sum_xy_diff : ℝ)

-- Conditions
def conditions := 
  (∑ i, xi i = 60) ∧ 
  (∑ i, yi i = 1200) ∧ 
  (∑ i, (xi i - (∑ j, xi j) / 20) ^ 2 = 80) ∧ 
  (∑ i, (yi i - (∑ j, yi j) / 20) ^ 2 = 9000) ∧ 
  (∑ i, (xi i - (∑ j, xi j) / 20) * (yi i - (∑ j, yi j) / 20) = 800)

-- Part 1: Prove the estimated population
theorem estimated_population (h : conditions xi yi sum_x sum_y sum_x_diff_sq sum_y_diff_sq sum_xy_diff) :
  200 * (sum_y / 20) = 12000 :=
by sorry

-- Part 2: Prove the correlation coefficient
theorem correlation_coefficient (h : conditions xi yi sum_x sum_y sum_x_diff_sq sum_y_diff_sq sum_xy_diff) :
  (sum_xy_diff / (real.sqrt (sum_x_diff_sq * sum_y_diff_sq))).round_to 2 = 0.94 :=
by sorry

end estimated_population_correlation_coefficient_l216_216225


namespace f_10_eq_144_l216_216388

def f : ℕ → ℕ
| 0     := 0
| 1     := 2
| 2     := 3
| (n+3) := f (n+2) + f (n+1)

theorem f_10_eq_144 : f 10 = 144 := 
by {
  unfold f,
  simp,
  sorry -- Proof to be filled.
}

end f_10_eq_144_l216_216388


namespace convex_polyhedron_face_symmetry_l216_216058

def CentrallySymmetric (P : Type) [LinearAddCommGroup P] [Module ℝ P] (p : P → Prop) : Prop :=
  ∃ c : P, ∀ x, p x ↔ p (2 • c - x)

structure Polyhedron :=
  (vertices : Set ℝ)
  (is_convex : Convex ℝ vertices)
  (faces : Set (Set ℝ))

def Subdivide (M : Polyhedron) (subReg : Set Polyhedron) : Prop :=
  (∀ P ∈ subReg, ∀ f ∈ P.faces, CentrallySymmetric ℝ (λ x, x ∈ f)) ∧
  ∃ faceUnion : M.faces, Subdivide faces into smaller faces with centrally symmetric property

theorem convex_polyhedron_face_symmetry (M : Polyhedron) (hM : M.is_convex)
  (hSubdiv : ∃ subReg, Subdivide M subReg) :
  ∀ f ∈ M.faces, CentrallySymmetric ℝ (λ x, x ∈ f) :=
sorry

end convex_polyhedron_face_symmetry_l216_216058


namespace find_speed_second_part_l216_216970

noncomputable def speed_second_part (x : ℝ) (v : ℝ) : Prop :=
  let t1 := x / 65       -- Time to cover the first x km at 65 kmph
  let t2 := 2 * x / v    -- Time to cover the second 2x km at v kmph
  let avg_time := 3 * x / 26    -- Average speed of the entire journey
  t1 + t2 = avg_time

theorem find_speed_second_part (x : ℝ) (v : ℝ) (h : speed_second_part x v) : v = 86.67 :=
sorry -- Proof of the claim

end find_speed_second_part_l216_216970


namespace part_a_part_b_l216_216055

noncomputable def is_increasing (f : ℝ → ℝ) (a : ℝ) (b : ℝ) : Prop :=
∀ x y, a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b ∧ x ≤ y → f x ≤ f y

variable {f : ℝ → ℝ}

theorem part_a (hf : is_increasing f 0 (π / 2)):
  (∫ x in 0..(π / 2), (f x - f (π / 4)) * (Real.sin x - Real.cos x)) ≥ 0 :=
sorry

theorem part_b (hf : is_increasing f 0 (π / 2)):
  ∃ (a : ℝ), a ∈ Icc (π / 4) (π / 2) ∧ 
    (∫ x in 0..a, f x * Real.sin x) = (∫ x in 0..a, f x * Real.cos x) :=
sorry

end part_a_part_b_l216_216055


namespace part1_subsets_containing_A₁_eq_subsets_not_containing_A₁_part2_polygons_containing_A₁_gt_polygons_not_containing_A₁_l216_216797

-- Define the set of vertices of the polygon inscribed in a circle
variables (A : Type) [decidable_eq A] (n : ℕ) (vertices : finset A) (A₁ : A)

-- Assume vertices have n elements including A₁
hypothesis h_vertices_size : vertices.card = n
hypothesis h_A₁_in_vertices : A₁ ∈ vertices

-- Part 1
theorem part1_subsets_containing_A₁_eq_subsets_not_containing_A₁ :
  (vertices.filter (λ x, x ∈ (A₁ :: vertices.val))).card = (vertices.filter (λ x, x ∉ (A₁ :: vertices.val))).card :=
sorry

-- Part 2
theorem part2_polygons_containing_A₁_gt_polygons_not_containing_A₁ :
  ∃ (polygons_containing_A₁ polygons_not_containing_A₁ : finset A), 
    polygons_containing_A₁.card > polygons_not_containing_A₁.card :=
sorry

end part1_subsets_containing_A₁_eq_subsets_not_containing_A₁_part2_polygons_containing_A₁_gt_polygons_not_containing_A₁_l216_216797


namespace sum_of_coefficients_l216_216064

theorem sum_of_coefficients (a b c d : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f(x + 3) = 3 * x^2 + 7 * x + 4)
  (h2 : ∀ x : ℝ, f(x) = a * x^3 + b * x^2 + c * x + d) :
  a + b + c + d = -7 :=
sorry

end sum_of_coefficients_l216_216064


namespace no_common_root_l216_216629

theorem no_common_root 
  (a b : ℚ) 
  (α : ℂ) 
  (h1 : α^5 = α + 1) 
  (h2 : α^2 = -a * α - b) : 
  False :=
sorry

end no_common_root_l216_216629


namespace range_of_m_l216_216694

variable (x m : ℝ)

def p : Prop := |x - 3| ≤ 2
def q : Prop := (x - m + 1) * (x - m - 1) ≤ 0

theorem range_of_m : (¬p x → ¬q x) ∧ (¬q x → ¬p x → False) → 2 ≤ m ∧ m ≤ 4 :=
by
  sorry

end range_of_m_l216_216694


namespace P_72_l216_216795

def P (n : ℕ) : ℕ :=
  -- The definition of P(n) should enumerate the ways of expressing n as a product
  -- of integers greater than 1, considering the order of factors.
  sorry

theorem P_72 : P 72 = 17 :=
by
  sorry

end P_72_l216_216795


namespace find_angle_A_find_value_of_a_l216_216437

open Real

theorem find_angle_A (a b c A B C : ℝ) (A_opposite : A = arccos((a^2 + c^2 - b^2)/(2*a*c)))
  (h : sqrt 3 * c * cos A + a * sin C = sqrt 3 * c) : A = π / 3 :=
  sorry

theorem find_value_of_a (a b c S : ℝ) (h_area : S = sqrt 3)
  (b_plus_c : b + c = 5) (A : ℝ) (h_A : A = π / 3) : a = sqrt 13 :=
  sorry

end find_angle_A_find_value_of_a_l216_216437


namespace x_sq_sub_y_sq_l216_216734

theorem x_sq_sub_y_sq (x y : ℝ) (h1 : x + y = 8) (h2 : x - y = 4) : x^2 - y^2 = 32 :=
by
  sorry

end x_sq_sub_y_sq_l216_216734


namespace estimated_population_is_correct_correlation_coefficient_is_correct_stratified_sampling_is_better_l216_216199

-- The sample data and conditions provided.
variables (x y : Fin 20 → ℝ)
variable (x̄ : ℝ := (∑ i, x i) / 20)
variable (ȳ : ℝ := (∑ i, y i) / 20)
variables (Sxx Syy Sxy : ℝ)
variables (n : ℕ := 20)
variables (total_plots : ℕ := 200)
variable (sum_x : ℝ := ∑ i, x i)
variable (sum_y : ℝ := ∑ i, y i)
variable (total_x : ℝ := 60)
variable (total_y : ℝ := 1200)

-- Given conditions
axiom condition1 : sum_x = 60
axiom condition2 : sum_y = 1200
axiom condition3 : Sxx = ∑ i, (x i - x̄) ^ 2
axiom condition4 : Syy = ∑ i, (y i - ȳ) ^ 2
axiom condition5 : Sxy = ∑ i, (x i - x̄) * (y i - ȳ)

-- Proven result 1: Estimated population of wild animals
def estimated_population (total_plots : ℕ) (avg_y : ℝ) : ℝ :=
  avg_y * total_plots

theorem estimated_population_is_correct : estimated_population total_plots (total_y / n) = 12000 :=
by
  -- Necessary calculations skipped
  sorry

-- Proven result 2: Correlation coefficient
noncomputable def correlation_coefficient (Sxy Sxx Syy : ℝ) : ℝ :=
  Sxy / (Real.sqrt (Sxx * Syy))

theorem correlation_coefficient_is_correct : correlation_coefficient Sxy Sxx Syy = 0.94 :=
by
  -- Assume given data and calculations are correct within required precision
  sorry

-- Proven result 3: Stratified sampling is more reasonable
theorem stratified_sampling_is_better : ∀ (method : String), method = "stratified sampling" → True :=
by
  -- Explanation skipped
  intros
  sorry

end estimated_population_is_correct_correlation_coefficient_is_correct_stratified_sampling_is_better_l216_216199


namespace sum_consecutive_integers_l216_216847

-- Define the sequence of integers
def seq (m : ℤ) : list ℤ :=
  [m-2, m-1, m, m+1, m+2]

-- Prove that the sum of five consecutive integers whose middle number is m equals 5m
theorem sum_consecutive_integers (m : ℤ) : (seq m).sum = 5 * m :=
by
  sorry

end sum_consecutive_integers_l216_216847


namespace sin_alpha_cos_beta_value_l216_216352

variables {α β : ℝ}

theorem sin_alpha_cos_beta_value 
  (h1 : Real.sin (α + β) = 1/2) 
  (h2 : 2 * Real.sin (α - β) = 1/2) : 
  Real.sin α * Real.cos β = 3/8 := by
sorry

end sin_alpha_cos_beta_value_l216_216352


namespace largest_non_decomposable_l216_216429

/-
The denominations of coins in Limonia are 𝑐 = {5^n, 5^(n-1) * 7, 5^(n-2) * 7^2, ..., 5 * 7^(n-1), 7^n}, 
where n is a natural number.
Given this, prove the largest amount that cannot be provided by the bank is 2 * 7^(n+1) - 3 * 5^(n+1).
-/
theorem largest_non_decomposable (n : ℕ) :
    ∃ s, s = 2 * 7^(n+1) - 3 * 5^(n+1) ∧ ¬(∃ a b c d e f, s = a * 5^n + b * (5^(n-1) * 7) + c * (5^(n-2) * 7^2) + d * (5^(n-3) * 7^3) + e * (5 * 7^(n-1)) + f * 7^n) :=
begin
  sorry
end

end largest_non_decomposable_l216_216429


namespace k_sym_l216_216513

def k : ℕ → (ℕ → ℝ) → ℝ
| 0       := λ x, 1
| 1       := λ x, x 0
| (n + 2) := λ x, x n * k n (λ i, x i) + (x n ^ 2 + x (n - 1) ^ 2) * k (n - 1) (λ i, x i)

theorem k_sym (n : ℕ) (x : ℕ → ℝ) : k n x = k n (λ i, x (n - 1 - i)) := by
  sorry

end k_sym_l216_216513


namespace total_population_correct_l216_216523

/-- Define the populations of each city -/
def Population.Seattle : ℕ := sorry
def Population.LakeView : ℕ := 24000
def Population.Boise : ℕ := (3 * Population.Seattle) / 5

/-- Population of Lake View is 4000 more than the population of Seattle -/
axiom lake_view_population : Population.LakeView = Population.Seattle + 4000

/-- Define the total population -/
def total_population : ℕ :=
  Population.Seattle + Population.LakeView + Population.Boise

/-- Prove that total population of the three cities is 56000 -/
theorem total_population_correct :
  total_population = 56000 :=
sorry

end total_population_correct_l216_216523


namespace ratio_AF_BF_l216_216587

noncomputable def parabola_focus (p : ℝ) (hp : 0 < p) : ℝ × ℝ := (p / 2, 0)

noncomputable def line_through_focus_angle (p : ℝ) (hp : 0 < p) : ℝ → ℝ :=
  λ x, (Real.sqrt 3 / 3) * (x - (p / 2))

noncomputable def intersection_with_parabola (p : ℝ) (hp : 0 < p) (x : ℝ) : Prop :=
  (Real.sqrt 3 / 3) * (x - (p / 2)) = real.sqrt (2 * p * x)

theorem ratio_AF_BF (p : ℝ) (hp : 0 < p) :
  ∃ A B : ℝ × ℝ, 
    intersection_with_parabola p hp A.1 ∧ 
    intersection_with_parabola p hp B.1 ∧
    (|A.1 - (parabola_focus p hp).1| / |B.1 - (parabola_focus p hp).1| = 7 + 4 * Real.sqrt 3 ∨
     |A.1 - (parabola_focus p hp).1| / |B.1 - (parabola_focus p hp).1| = 7 - 4 * Real.sqrt 3) :=
sorry

end ratio_AF_BF_l216_216587


namespace tuesday_presence_is_82_percent_l216_216830

variable (total_students : ℕ)
variable (pct_absent_monday pct_present_monday : ℝ)
variable (pct_absent_present_tuesday pct_present_absent_tuesday : ℝ)

def calculate_tuesday_presence 
    (total_students : ℕ) (pct_absent_monday pct_present_monday 
    pct_absent_present_tuesday pct_present_absent_tuesday : ℝ) : ℝ :=
  let absent_monday := total_students * pct_absent_monday
  let present_monday := total_students * pct_present_monday
  let absent_present_tuesday := absent_monday * pct_absent_present_tuesday
  let present_absent_tuesday := present_monday * pct_present_absent_tuesday
  let present_both_days := present_monday - present_absent_tuesday
  let total_present_tuesday := present_both_days + absent_present_tuesday
  total_present_tuesday / total_students * 100

theorem tuesday_presence_is_82_percent 
  (h_total_students : total_students = 1000)
  (h_pct_absent_monday : pct_absent_monday = 0.1)
  (h_pct_present_monday : pct_present_monday = 0.9)
  (h_pct_absent_present_tuesday : pct_absent_present_tuesday = 0.1)
  (h_pct_present_absent_tuesday : pct_present_absent_tuesday = 0.1) :
  calculate_tuesday_presence total_students pct_absent_monday pct_present_monday pct_absent_present_tuesday pct_present_absent_tuesday = 82 := 
by
  sorry

end tuesday_presence_is_82_percent_l216_216830


namespace initial_average_production_l216_216662

theorem initial_average_production (n : ℕ) (today_production : ℕ) 
  (new_average : ℕ) (initial_average : ℕ) :
  n = 1 → today_production = 60 → new_average = 55 → initial_average = (new_average * (n + 1) - today_production) → initial_average = 50 :=
by
  intros h1 h2 h3 h4
  -- Insert further proof here
  sorry

end initial_average_production_l216_216662


namespace rankings_possible_arrangements_l216_216842

theorem rankings_possible_arrangements :
  let students := ["A", "B", "C", "D", "E"],
      championship := 1,
      fifth_place := 5 in
  ((∑ (a ∈ students) (b ∈ students.filter (≠ a)), (sum (list.permutations (students.erase a).erase b))) - 
  (∑ (a ∈ students.filter (≠ "A") ∩ filter (≠ "B")), (fifth_place |-> (∑ (b ∈ students.filter (≠ a)), 
  (sum (list.permutations (students.erase a).erase b)))))) = 54 := 
sorry

end rankings_possible_arrangements_l216_216842


namespace min_value_of_f_l216_216290

def f (x : ℝ) : ℝ := 16^x - 4^x + 1

theorem min_value_of_f : ∃ x : ℝ, f x = 3/4 :=
by
  sorry

end min_value_of_f_l216_216290


namespace proof_of_incorrect_conclusions_l216_216131

def incorrect_conclusions : Prop :=
  (C_is_incorrect) ∧ (D_is_incorrect)

theorem proof_of_incorrect_conclusions : incorrect_conclusions := by
  sorry

/-
Definitions used for conditions:
C_is_incorrect: "$a \neq 0$" is not a sufficient but necessary condition for "$ab \neq 0$".
D_is_incorrect: The inequality solution set $\frac{1}{x} > 1$ is not just $\{x \mid x < 1\}$ but also $x < 0$.
-/

end proof_of_incorrect_conclusions_l216_216131


namespace angle_distance_between_CM_BK_l216_216752

noncomputable def midpoint (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)

def vector (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2, B.3 - A.3)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1^2 + u.2^2 + u.3^2)

def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

theorem angle_distance_between_CM_BK :
  let A := (0, 0, 0) in
  let B := (a, 0, 0) in
  let C := (a/2, a*Real.sqrt 3 / 2, 0) in
  let D := (a/2, a*Real.sqrt 3 / 6, a*Real.sqrt 6 / 3) in
  let M := midpoint A B in
  let K := midpoint C D in
  let CM := vector C M in
  let BK := vector B K in
  let theta := Real.arccos ((dot_product CM BK) / ((magnitude CM) * (magnitude BK))) in
  theta = Real.arccos (Real.sqrt 6 / 3) ∧
  Real.sqrt ((magnitude (cross_product CM BK))^2 / (magnitude (vector C B))^2) = (Real.sqrt 3 / 10)
:=
by
  sorry

end angle_distance_between_CM_BK_l216_216752


namespace susanna_items_l216_216754

theorem susanna_items (item_cost total_spent : ℕ) (h_cost : item_cost = 199) (h_spent : total_spent = 6576) :
  ∃ k, total_spent = item_cost * k ∧ k = 24 :=
by
  use 24
  rw [h_cost, h_spent]
  simp
  sorry

end susanna_items_l216_216754


namespace average_gpa_difference_2_l216_216853

def avg_gpa_6th_grader := 93
def avg_gpa_8th_grader := 91
def school_avg_gpa := 93

noncomputable def gpa_diff (gpa_7th_grader diff : ℝ) (avg6 avg8 school_avg : ℝ) := 
  gpa_7th_grader = avg6 + diff ∧ 
  (avg6 + gpa_7th_grader + avg8) / 3 = school_avg

theorem average_gpa_difference_2 (x : ℝ) : 
  (∃ G : ℝ, gpa_diff G x avg_gpa_6th_grader avg_gpa_8th_grader school_avg_gpa) → x = 2 :=
by
  sorry

end average_gpa_difference_2_l216_216853


namespace find_a_2b_3c_value_l216_216800

-- Problem statement and conditions
theorem find_a_2b_3c_value (a b c : ℝ)
  (h : ∀ x : ℝ, (x < -1 ∨ abs (x - 10) ≤ 2) ↔ (x - a) * (x - b) / (x - c) ≤ 0)
  (h_ab : a < b) : a + 2 * b + 3 * c = 29 := 
sorry

end find_a_2b_3c_value_l216_216800


namespace range_of_a12_l216_216767

variable (a : ℕ → ℝ)
variable (a1 d : ℝ)

def arithmetic_seq (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 + n * d

variable (h_arith_seq : arithmetic_seq a a1 d)
variable (h_a8 : a 7 ≥ 15)
variable (h_a9 : a 8 ≤ 13)

theorem range_of_a12 : ∀ a1 d, (arithmetic_seq a a1 d) → (a 7 ≥ 15) → (a 8 ≤ 13) → (a 11 ≤ 7) :=
by
  intro a1 d h_arith_seq h_a8 h_a9
  sorry

end range_of_a12_l216_216767


namespace nth_term_sign_l216_216127

-- Definition of the product expansion in terms of variables
def expansion (a b c d : ℕ → ℝ) : ℕ → ℝ
  := sorry

-- Function ν(n) which determines the sign alternation
def ν (n : ℕ) : ℕ := sorry

theorem nth_term_sign (a b c d : ℕ → ℝ) (n : ℕ) :
  (expansion a b c d n) = (-1)^{ν(n)} :=
sorry

end nth_term_sign_l216_216127


namespace alice_card_value_sum_l216_216232

theorem alice_card_value_sum :
  ∀ (y : ℝ), 0 < y ∧ y < 90 → 
  (∃ sin_y cos_y cot_y, 
    sin_y = sin y ∧ 
    cos_y = cos y ∧ 
    cot_y = cos y / sin y ∧
    (∀ b c, 
      (b = sin_y ∨ b = cos_y ∨ b = cot_y) ∧ 
      (c = sin_y ∨ c = cos_y ∨ c = cot_y) ∧ 
      b ≠ c → 
      (¬ alice_can_identify b cos_y ∧ ¬ alice_can_identify c cot_y)) ∧ 
    alice_can_identify sin_y cos_y) →
  sin y + cos y / sin y = (2 + real.sqrt 2) / 2 :=
by sorry

def alice_can_identify (card_value : ℝ) (cos_y : ℝ) : Prop :=
  need_to_define_what_it_means -- This requires defining what it means for Alice to identify the value

end alice_card_value_sum_l216_216232


namespace larger_number_l216_216116

theorem larger_number (x y : ℕ) (h₁ : x + y = 27) (h₂ : x - y = 5) : x = 16 :=
by sorry

end larger_number_l216_216116


namespace estimate_population_correct_correlation_coefficient_correct_l216_216214

namespace EcologicalSurvey

def estimated_population (total_animals : ℕ) (plots : ℕ) : ℕ :=
  (total_animals / 20) * plots

theorem estimate_population_correct :
  estimated_population 1200 200 = 12000 :=
by 
  -- Proof would go here.
  sorry

def correlation_coefficient (sum_xi_xbar : ℝ) (sum_yi_ybar : ℝ) (sum_prod_xi_yi : ℝ) : ℝ :=
  sum_prod_xi_yi / Math.sqrt (sum_xi_xbar * sum_yi_ybar)

theorem correlation_coefficient_correct :
  correlation_coefficient 80 9000 800 ≈ 0.94 :=
by 
  -- Proof would go here.
  sorry

end EcologicalSurvey

end estimate_population_correct_correlation_coefficient_correct_l216_216214


namespace maximum_k_for_ray_below_f_l216_216702

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + 3 * x - 2

theorem maximum_k_for_ray_below_f :
  let g (x : ℝ) : ℝ := (x * Real.log x + 3 * x - 2) / (x - 1)
  ∃ k : ℤ, ∀ x > 1, g x > k ∧ k = 5 :=
by sorry

end maximum_k_for_ray_below_f_l216_216702


namespace sum_abc_l216_216258

noncomputable def polynomial : Polynomial ℝ :=
  Polynomial.C (-6) + Polynomial.X * (Polynomial.C 11 + Polynomial.X * (Polynomial.C (-6) + Polynomial.X))

def t (k : ℕ) : ℝ :=
  match k with
  | 0 => 3
  | 1 => 6
  | 2 => 14
  | _ => 0 -- placeholder, as only t_0, t_1, t_2 are given explicitly

def a := 6
def b := -11
def c := 18

def t_rec (k : ℕ) : ℝ :=
  match k with
  | 0 => 3
  | 1 => 6
  | 2 => 14
  | n + 3 => a * t (n + 2) + b * t (n + 1) + c * t n

theorem sum_abc : a + b + c = 13 := by
  sorry

end sum_abc_l216_216258


namespace rectangle_length_is_4_l216_216596

theorem rectangle_length_is_4 (w l : ℝ) (h_length : l = w + 3) (h_area : l * w = 4) : l = 4 := 
sorry

end rectangle_length_is_4_l216_216596


namespace solve_system_l216_216062

theorem solve_system :
    ∃ (x y : ℤ), 
      2 * (x + 2 * y) - 5 * y = -1 ∧
      3 * (x - y) + y = 2 ∧ 
      x = -4 ∧ 
      y = -7 := 
by
  -- Definitions from conditions
  let x := -4
  let y := -7
  -- Conditions
  have h1: 2 * (x + 2 * y) - 5 * y = -1 := by
    rw [x, y]
    linarith
  have h2: 3 * (x - y) + y = 2 := by
    rw [x, y]
    linarith
  -- Conclude the proof
  use [x, y]
  exact ⟨h1, h2, rfl, rfl⟩

end solve_system_l216_216062


namespace omega_phi_unique_l216_216701

noncomputable def omega_phi (f : ℝ → ℝ) (ω : ℝ) (φ : ℝ) : Prop :=
  f = λ x, √3 * sin (ω * x + φ) ∧ ω > 0 ∧ -π / 2 ≤ φ ∧ φ < π / 2 ∧
  (∀ x, f (x + π / 2 / ω) = f x) ∧
  (∀ x, f (2 * (π / 3) - x) = f x)

theorem omega_phi_unique (ω : ℝ) (φ : ℝ) :
  (omega_phi (λ x, √3 * sin (ω * x + φ)) 2 (-π / 6)) ∧
  (∀ x, 0 ≤ x ∧ x ≤ π / 2 → (√3 * sin (2 * x - π / 6) ≥ -√3 / 2) ∧ (√3 * sin (2 * x - π / 6) ≤ √3)) := sorry

end omega_phi_unique_l216_216701


namespace estimated_population_correct_correlation_coefficient_correct_l216_216212

variables (x y : ℕ → ℝ)
variables (n : ℕ) (plots : ℕ)

noncomputable def average_y := (∑ i in finset.range 20, y i) / 20

noncomputable def estimated_population := average_y y * plots

theorem estimated_population_correct (h_sum_y : ∑ i in finset.range 20, y i = 1200)
                                      (h_plots : plots = 200) :
  estimated_population y plots = 12000 :=
by sorry

noncomputable def correlation_coefficient (x y : ℕ → ℝ) :=
  let sum_xy := ∑ i in finset.range 20, (x i - ((∑ j in finset.range 20, x j) / 20)) * 
                               (y i - ((∑ j in finset.range 20, y j) / 20)) in
  let sum_xx := ∑ i in finset.range 20, (x i - ((∑ j in finset.range 20, x j) / 20)) ^ 2 in
  let sum_yy := ∑ i in finset.range 20, (y i - ((∑ j in finset.range 20, y j) / 20)) ^ 2 in
  sum_xy / (real.sqrt (sum_xx * sum_yy))

theorem correlation_coefficient_correct (h_sum_xx : ∑ i in finset.range 20, (x i - ((∑ j in finset.range 20, x j) / 20)) ^ 2 = 80)
                                        (h_sum_yy : ∑ i in finset.range 20, (y i - ((∑ j in finset.range 20, y j) / 20)) ^ 2 = 9000)
                                        (h_sum_xy : ∑ i in finset.range 20, (x i - ((∑ j in finset.range 20, x j) / 20)) * (y i - ((∑ j in finset.range 20, y j) / 20)) = 800) :
  correlation_coefficient x y ≈ 0.94 :=
by sorry

end estimated_population_correct_correlation_coefficient_correct_l216_216212


namespace real_part_z_l216_216737

open Complex

theorem real_part_z (z : ℂ) (hz : (1 + complex.I) * z = 2 * complex.I) : z.re = 1 := 
sorry

end real_part_z_l216_216737


namespace max_good_pairs_upper_bound_and_existence_l216_216483

-- Definitions and conditions
def T : Finset Point := { 1, 2, ..., 66 } -- For simplicity, using natural numbers to represent points
def P : Finset Line := { l1, l2, ..., l16 } -- Using abstract identifiers to represent lines
def good_pair (A : Point) (l : Line) : Prop := A ∈ l

-- Lean statement to prove the theorem
theorem max_good_pairs_upper_bound_and_existence :
  (∀ (A : Point) (l : Line), A ∈ T → l ∈ P → good_pair A l) ∧
  ∃ (T' : Finset Point) (P' : Finset Line), T' = T ∧ P' = P ∧ ( ⨆ n, (count_good_pairs T' P') <= n ) :=
sorry

end max_good_pairs_upper_bound_and_existence_l216_216483


namespace rotated_angle_ACB_new_measure_is_100_l216_216082

def rotate_angle (initial_angle rotation_degrees : ℝ) : ℝ :=
  (rotation_degrees % 360) - initial_angle

theorem rotated_angle_ACB_new_measure_is_100 :
  ∀ (C A B : Type) (angle_ACB : ℝ),
  angle_ACB = 60 → (rotate_angle angle_ACB 520) = 100 :=
by
  intros C A B angle_ACB h_angle
  dsimp [rotate_angle]
  rw [h_angle]
  have : 520 % 360 = 160 := rfl
  rw [this]
  norm_num
  sorry

end rotated_angle_ACB_new_measure_is_100_l216_216082


namespace num_Cu_atoms_is_one_l216_216583

-- Given conditions and constants
def molecular_weight_compound : ℝ := 124
def atomic_weight_Cu : ℝ := 63.55
def atomic_weight_C : ℝ := 12.01
def atomic_weight_O : ℝ := 16.00
def weight_C : ℝ := 1 * atomic_weight_C
def weight_O : ℝ := 3 * atomic_weight_O
def weight_CO : ℝ := weight_C + weight_O
def weight_Cu : ℝ := molecular_weight_compound - weight_CO

-- Define the number of Cu atoms
def num_Cu_atoms : ℝ := weight_Cu / atomic_weight_Cu

-- Prove the number of Cu atoms is 1
theorem num_Cu_atoms_is_one : num_Cu_atoms = 1 := by
  sorry

end num_Cu_atoms_is_one_l216_216583


namespace inequality_abcde_l216_216460

theorem inequality_abcde
  (a b c d : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd : 0 < d) : 
  1 / a + 1 / b + 4 / c + 16 / d ≥ 64 / (a + b + c + d) := 
  sorry

end inequality_abcde_l216_216460


namespace neither_A_B_C_prob_correct_l216_216457

noncomputable def P (A B C : Prop) : Prop :=
  let P_A := 0.25
  let P_B := 0.35
  let P_C := 0.40
  let P_A_and_B := 0.10
  let P_A_and_C := 0.15
  let P_B_and_C := 0.20
  let P_A_and_B_and_C := 0.05
  
  let P_A_or_B_or_C := 
    P_A + P_B + P_C - P_A_and_B - P_A_and_C - P_B_and_C + P_A_and_B_and_C
  
  let P_neither_A_nor_B_nor_C := 1 - P_A_or_B_or_C
    
  P_neither_A_nor_B_nor_C = 0.45

theorem neither_A_B_C_prob_correct :
  P A B C := by
  sorry

end neither_A_B_C_prob_correct_l216_216457


namespace trapezium_height_l216_216653

theorem trapezium_height (a b A h : ℝ) (ha : a = 12) (hb : b = 16) (ha_area : A = 196) :
  (A = 0.5 * (a + b) * h) → h = 14 :=
by
  intros h_eq
  rw [ha, hb, ha_area] at h_eq
  sorry

end trapezium_height_l216_216653


namespace productivity_comparison_l216_216896

variables (x y : ℝ)

-- Given conditions
-- Combined initial productivity completes a trench in 2 hours
def initial_combined_productivity : Prop :=
  2 * (x + y) = 1

-- Adjusted productivity completes a trench in 1 hour
def adjusted_combined_productivity : Prop :=
  (x / 3 + 3 * y) = 1

-- Inferred ratio of productivities solution
def productivity_ratio : ℝ := y / x

theorem productivity_comparison
  (hx : initial_combined_productivity x y)
  (hy : adjusted_combined_productivity x y) :
  productivity_ratio x y = 5 / 3 :=
begin
  -- Transformation of provided Lean statement for mathematical equivalency
  sorry
end

end productivity_comparison_l216_216896


namespace estimated_population_is_correct_correlation_coefficient_is_correct_stratified_sampling_is_better_l216_216197

-- The sample data and conditions provided.
variables (x y : Fin 20 → ℝ)
variable (x̄ : ℝ := (∑ i, x i) / 20)
variable (ȳ : ℝ := (∑ i, y i) / 20)
variables (Sxx Syy Sxy : ℝ)
variables (n : ℕ := 20)
variables (total_plots : ℕ := 200)
variable (sum_x : ℝ := ∑ i, x i)
variable (sum_y : ℝ := ∑ i, y i)
variable (total_x : ℝ := 60)
variable (total_y : ℝ := 1200)

-- Given conditions
axiom condition1 : sum_x = 60
axiom condition2 : sum_y = 1200
axiom condition3 : Sxx = ∑ i, (x i - x̄) ^ 2
axiom condition4 : Syy = ∑ i, (y i - ȳ) ^ 2
axiom condition5 : Sxy = ∑ i, (x i - x̄) * (y i - ȳ)

-- Proven result 1: Estimated population of wild animals
def estimated_population (total_plots : ℕ) (avg_y : ℝ) : ℝ :=
  avg_y * total_plots

theorem estimated_population_is_correct : estimated_population total_plots (total_y / n) = 12000 :=
by
  -- Necessary calculations skipped
  sorry

-- Proven result 2: Correlation coefficient
noncomputable def correlation_coefficient (Sxy Sxx Syy : ℝ) : ℝ :=
  Sxy / (Real.sqrt (Sxx * Syy))

theorem correlation_coefficient_is_correct : correlation_coefficient Sxy Sxx Syy = 0.94 :=
by
  -- Assume given data and calculations are correct within required precision
  sorry

-- Proven result 3: Stratified sampling is more reasonable
theorem stratified_sampling_is_better : ∀ (method : String), method = "stratified sampling" → True :=
by
  -- Explanation skipped
  intros
  sorry

end estimated_population_is_correct_correlation_coefficient_is_correct_stratified_sampling_is_better_l216_216197


namespace more_republicans_than_democrats_in_house_l216_216519

-- Definition of the problem conditions
def total_house_members (R D : ℕ) := R + D = 434
def more_republicans (R D X : ℕ) := R = D + X
def democrats_in_house := 202

-- Statement of the proof problem
theorem more_republicans_than_democrats_in_house : ∃ (X : ℕ), 
  ∀ (R D : ℕ), 
    total_house_members R D → 
    more_republicans R D X → 
    D = democrats_in_house → 
    X = 30 :=
begin
  sorry
end

end more_republicans_than_democrats_in_house_l216_216519


namespace sequence_sum_property_l216_216340

theorem sequence_sum_property {a S : ℕ → ℚ} (h1 : a 1 = 3/2)
  (h2 : ∀ n : ℕ, 2 * a (n + 1) + S n = 3) :
  (∀ n : ℕ, a n = 3 * (1/2)^n) ∧
  (∃ (n_max : ℕ),  (∀ n : ℕ, n ≤ n_max → (S n = 3 * (1 - (1/2)^n)) ∧ ∀ n : ℕ, (S (2 * n)) / (S n) > 64 / 63 → n_max = 5)) :=
by {
  -- The proof would go here
  sorry
}

end sequence_sum_property_l216_216340


namespace evaluate_expression_l216_216919

def floor (x : ℝ) := Int.floor x

theorem evaluate_expression : floor 6.5 * floor (2/3) + floor 2 * 7.2 + floor 8.4 - 6.2 = 16.2 := by
  sorry

end evaluate_expression_l216_216919


namespace boston_snow_l216_216621

noncomputable def initial_snow : ℝ := 0.5
noncomputable def second_day_snow_inch : ℝ := 8 / 12
noncomputable def next_two_days_melt_inch : ℝ := 2 / 12
noncomputable def fifth_day_snow_factor : ℝ := 2

theorem boston_snow : 
  let second_day_snow := initial_snow + second_day_snow_inch,
      snow_after_melt := second_day_snow - next_two_days_melt_inch,
      fifth_day_snow := fifth_day_snow_factor * initial_snow
  in snow_after_melt + fifth_day_snow = 2 := 
by
  sorry

end boston_snow_l216_216621


namespace sum_of_quarter_circle_arcs_l216_216941

-- Define the main variables and problem statement.
variable (D : ℝ) -- Diameter of the original circle.
variable (n : ℕ) (hn : 0 < n) -- Number of parts (positive integer).

-- Define a theorem stating that the sum of quarter-circle arcs is greater than D, but less than (pi D / 2) as n tends to infinity.
theorem sum_of_quarter_circle_arcs (hn : 0 < n) :
  D < (π * D) / 4 ∧ (π * D) / 4 < (π * D) / 2 :=
by
  sorry -- Proof of the theorem goes here.

end sum_of_quarter_circle_arcs_l216_216941


namespace exists_base_for_part_a_not_exists_base_for_part_b_l216_216641

theorem exists_base_for_part_a : ∃ b : ℕ, (3 + 4 = b) ∧ (3 * 4 = 1 * b + 5) := 
by
  sorry

theorem not_exists_base_for_part_b : ¬ ∃ b : ℕ, (2 + 3 = b) ∧ (2 * 3 = 1 * b + 1) :=
by
  sorry

end exists_base_for_part_a_not_exists_base_for_part_b_l216_216641


namespace pyramid_volume_correct_l216_216601

-- Definitions of the given conditions
def square_side_length : ℝ := 120
def cut_distance_from_corner : ℝ := 10
def cut_angle_degrees : ℝ := 45
def pyramid_volume : ℝ := 288000

-- The Lean statement to prove the volume of the pyramid
theorem pyramid_volume_correct :
  let d := real.sqrt (square_side_length^2 + square_side_length^2) / 2 in
  let height := d * real.cos (real.pi / 4) in
  let base_area := square_side_length^2 in
  let volume := (1 / 3) * base_area * height in
  volume = pyramid_volume :=
by sorry

end pyramid_volume_correct_l216_216601


namespace number_of_different_totals_l216_216544

def points : Set ℕ := {0, 2, 3, 6}

def all_possible_totals : Set ℕ := { a + b | a b : ℕ, a ∈ points, b ∈ points }

theorem number_of_different_totals : all_possible_totals.card = 9 := 
sorry

end number_of_different_totals_l216_216544


namespace charity_event_arrangement_l216_216940

theorem charity_event_arrangement :
  ∃ arrangements : ℕ,
    (∃ (first_day second_day third_day : Finset ℕ),
      first_day.card = 1 ∧
      second_day.card = 2 ∧
      third_day.card = 3 ∧
      Disjoint first_day (second_day ∪ third_day) ∧
      Disjoint second_day third_day ∧
      (first_day ∪ second_day ∪ third_day) = Finset.range 6) ∧
    arrangements = 60 :=
begin
  sorry
end

end charity_event_arrangement_l216_216940


namespace minimum_value_of_f_l216_216312

noncomputable def f (x : ℝ) : ℝ := 16^x - 4^x + 1

theorem minimum_value_of_f : ∃ (x : ℝ), ∀ (y : ℝ), f(y) ≥ f(x) ∧ f(x) = 3 / 4 :=
sorry

end minimum_value_of_f_l216_216312


namespace min_value_1_over_a_add_1_over_b_l216_216369

theorem min_value_1_over_a_add_1_over_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃p : ℝ × ℝ, p = (-1, 2) ∧ ((∃r : ℝ, r = 2) ∧ (∀x y : ℝ, (x + 1) ^ 2 + (y - 2)^ 2 = 4 → 2 * a * x - b * y + 2 = 0 → (4: ℝ) = 4)) → 
  (1 / a + 1 / b = 2)) :=
begin
  sorry
end

end min_value_1_over_a_add_1_over_b_l216_216369


namespace estimated_population_correct_correlation_coefficient_correct_l216_216202

noncomputable def estimated_population
  (num_plots : ℕ) (sample_size : ℕ) 
  (sum_y : ℕ) : ℕ :=
(sum_y / sample_size) * num_plots

noncomputable def correlation_coefficient
  (sum_x_sq : ℝ) (sum_y_sq : ℝ) 
  (sum_xy : ℝ) : ℝ :=
sum_xy / real.sqrt(sum_x_sq * sum_y_sq)

theorem estimated_population_correct
  (num_plots : ℕ := 200) (sample_size : ℕ := 20) 
  (sum_y : ℕ := 1200) : estimated_population num_plots sample_size sum_y = 12000 := by
sorry

theorem correlation_coefficient_correct
  (sum_x_sq : ℝ := 80) (sum_y_sq : ℝ := 9000) 
  (sum_xy : ℝ := 800) : real.abs (correlation_coefficient sum_x_sq sum_y_sq sum_xy - 0.94) < 0.01 := by
sorry

end estimated_population_correct_correlation_coefficient_correct_l216_216202


namespace product_of_possible_values_l216_216728

theorem product_of_possible_values :
  (∀ x : ℝ, abs (18 / x + 4) = 3 → x = -18 ∨ x = -18 / 7) →
  (∀ x1 x2 : ℝ, x1 = -18 → x2 = -18 / 7 → x1 * x2 = 324 / 7) :=
by
  intros h x1 x2 hx1 hx2
  rw [hx1, hx2]
  norm_num

end product_of_possible_values_l216_216728


namespace arrangement_photos_l216_216616

theorem arrangement_photos (color_photos black_white_photos : Finset ℕ) 
    (h1 : color_photos.card = 4) (h2 : black_white_photos.card = 3) : 
    ∃ n : ℕ, n = (Nat.perm 4 4) * (Nat.perm 5 3) := by
  sorry

end arrangement_photos_l216_216616


namespace max_ant_area_is_31_8_l216_216977

theorem max_ant_area_is_31_8 :
  ∀ (speed : ℝ) (time : ℝ) (π : ℝ), speed = 10 → time = 2 → π > 0 →
  let radius := (speed * time) / (2 * π)
  let area := π * radius^2
  (abs (area - 31.8) < 0.1) := 
begin
  intros speed time π hs ht hπ,
  let radius := (speed * time) / (2 * π),
  let area := π * radius^2,
  have h_radius : radius = (10 / π), from sorry,  -- calculate and conclude radius
  have h_area : area = 10^2 / π, from sorry,       -- calculate and conclude area
  -- evaluate and conclude area ≈ 31.8
  show (abs (area - 31.8) < 0.1), from sorry,
end

end max_ant_area_is_31_8_l216_216977


namespace max_intersection_area_l216_216049

-- Definitions for the dimensions of the rectangles and their constraints
variables {a b c d : ℕ}

-- Conditions for the rectangles
def cond1 : Prop := a < b
def cond2 : Prop := c > d
def cond3 : Prop := 2011 < a * b ∧ a * b < 2020
def cond4 : Prop := 2011 < c * d ∧ c * d < 2020

-- The theorem to prove the maximum possible area of the intersection
theorem max_intersection_area (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : 
  ∃ (x y : ℕ), x * y = 1764 ∧ x ≤ 42 ∧ y ≤ 42 :=
sorry

end max_intersection_area_l216_216049


namespace estimate_population_correct_correlation_coefficient_correct_l216_216216

namespace EcologicalSurvey

def estimated_population (total_animals : ℕ) (plots : ℕ) : ℕ :=
  (total_animals / 20) * plots

theorem estimate_population_correct :
  estimated_population 1200 200 = 12000 :=
by 
  -- Proof would go here.
  sorry

def correlation_coefficient (sum_xi_xbar : ℝ) (sum_yi_ybar : ℝ) (sum_prod_xi_yi : ℝ) : ℝ :=
  sum_prod_xi_yi / Math.sqrt (sum_xi_xbar * sum_yi_ybar)

theorem correlation_coefficient_correct :
  correlation_coefficient 80 9000 800 ≈ 0.94 :=
by 
  -- Proof would go here.
  sorry

end EcologicalSurvey

end estimate_population_correct_correlation_coefficient_correct_l216_216216


namespace solve_func_eq_l216_216286

open Function

theorem solve_func_eq {f : ℝ → ℝ} 
  (h : ∀ x y : ℝ, f(x * f(x) + f(y)) = y + (f(x))^2) : 
  f = id := 
sorry

end solve_func_eq_l216_216286


namespace sin_P_l216_216421

-- Definitions involving the right triangle with required conditions:
def triangle_PQR (P Q R : Type) :=
  ∠ R = 90 ∧ sin Q = 3/5 ∧ sin R = 1

-- The main theorem that needs to be proven:
theorem sin_P (P Q R : Type) (h : triangle_PQR P Q R) : sin P = 3/5 :=
  sorry

end sin_P_l216_216421


namespace sqrt_fraction_sum_l216_216926

theorem sqrt_fraction_sum : 
  sqrt (1 / 25 + 1 / 36) = sqrt 61 / 30 :=
  sorry

end sqrt_fraction_sum_l216_216926


namespace find_larger_number_l216_216108

theorem find_larger_number (a b : ℤ) (h1 : a + b = 27) (h2 : a - b = 5) : a = 16 := by
  sorry

end find_larger_number_l216_216108


namespace necessary_but_not_sufficient_l216_216928

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem necessary_but_not_sufficient (f : ℝ → ℝ) :
  (f 0 = 0) →
  (∀ x : ℝ, f (-x) = -f x) →
  ¬∀ f' : ℝ → ℝ, (f' 0 = 0 → ∀ y : ℝ, f' (-y) = -f' y)
:= by
  sorry

end necessary_but_not_sufficient_l216_216928


namespace angle_Y_l216_216466

theorem angle_Y (p q : Line) (X Z Y : Point) (m_angle_X : ℕ) (m_angle_Z : ℕ)
  (hpq : Parallel p q)
  (hX : m_angle_X = 150)
  (hZ : m_angle_Z = 50) :
  ∃ m_angle_Y : ℕ, m_angle_Y = 160 :=
by
  sorry

end angle_Y_l216_216466


namespace inequality_pos_seq_l216_216477

theorem inequality_pos_seq (a : ℕ → ℝ) (n : ℕ) 
  (ha_positive : ∀ i, 1 ≤ i → i ≤ n → 0 < a i) :
  (∑ i in Finset.range n, (i+1) / (∑ j in Finset.range (i+1), a (j+1)))
  < 4 * (∑ i in Finset.range n, 1 / a (i+1)) := 
sorry

end inequality_pos_seq_l216_216477


namespace cofactor_diagonal_elements_equal_l216_216015

noncomputable def isSymmetric (A : Matrix ℝ) : Prop :=
  A = A.transpose

noncomputable def rowSumZero (A : Matrix ℝ) : Prop :=
  ∀i, ∑ j, A i j = 0

noncomputable def cofactor (A : Matrix ℝ) (i j : ℕ) : ℝ :=
  (-1)^(i+j) * (A.minor (Fin.succAbove i) (Fin.succAbove j)).det

theorem cofactor_diagonal_elements_equal (A : Matrix ℝ) 
  (h_sym : isSymmetric A) 
  (h_rowSum : rowSumZero A) :
  ∀ i j, i = j → cofactor A i i = cofactor A j j :=
sorry

end cofactor_diagonal_elements_equal_l216_216015


namespace slope_condition_min_area_triangle_l216_216706

theorem slope_condition (m : ℝ) (h : (4 - m) / (-m) < 2) : m > 0 ∨ m < -4 ∧ m ≠ 4 :=
sorry

theorem min_area_triangle (m : ℝ) (h1 : 0 < m) (h2 : m < 4) : 
    (∃ (A B O : Point), A = (m, 0) ∧ B = (0, 4 - m) ∧ O = (0, 0) ∧ 
    Area (Triangle A B O) = 2 ∧ line l = {x + y - 2 = 0}) :=
sorry

end slope_condition_min_area_triangle_l216_216706


namespace calculated_value_is_six_l216_216579

theorem calculated_value_is_six (n : ℝ) (h : n = 50.0) : 0.20 * n - 4 = 6.0 :=
by
  rw [h]  -- Replace n with 50.0
  -- Simplify the left-hand side
  calc 
  0.20 * 50.0 - 4 = 10.0 - 4 : by norm_num
               ... = 6.0      : by norm_num

end calculated_value_is_six_l216_216579


namespace a_minus_b_value_l216_216868

theorem a_minus_b_value (a b c : ℝ) (x : ℝ) 
    (h1 : (2 * x - 3) ^ 2 = a * x ^ 2 + b * x + c)
    (h2 : x = 0 → c = 9)
    (h3 : x = 1 → a + b + c = 1)
    (h4 : x = -1 → (2 * (-1) - 3) ^ 2 = a * (-1) ^ 2 + b * (-1) + c) : 
    a - b = 16 :=
by  
  sorry

end a_minus_b_value_l216_216868


namespace num_females_math_not_english_is_15_l216_216958

-- Define the conditions
def male_math := 120
def female_math := 80
def female_english := 120
def male_english := 80
def total_students := 260
def both_male := 75

def female_math_not_english : Nat :=
  female_math - (female_english + female_math - (total_students - (male_math + male_english - both_male)))

theorem num_females_math_not_english_is_15 :
  female_math_not_english = 15 :=
by
  -- This is where the proof will be, but for now, we use 'sorry' to skip it.
  sorry

end num_females_math_not_english_is_15_l216_216958


namespace cos_sum_proof_l216_216669

theorem cos_sum_proof (α β : ℝ) 
  (h1 : α - β = π / 6) 
  (h2 : tan α - tan β = 3) : 
  cos (α + β) = 1 / 3 - sqrt 3 / 2 := 
by
  sorry

end cos_sum_proof_l216_216669


namespace total_apples_correctness_l216_216389

-- Define the number of apples each man bought
def applesMen := 30

-- Define the number of apples each woman bought
def applesWomen := applesMen + 20

-- Define the total number of apples bought by the two men
def totalApplesMen := 2 * applesMen

-- Define the total number of apples bought by the three women
def totalApplesWomen := 3 * applesWomen

-- Define the total number of apples bought by the two men and three women
def totalApples := totalApplesMen + totalApplesWomen

-- Prove that the total number of apples bought by two men and three women is 210
theorem total_apples_correctness : totalApples = 210 := by
  sorry

end total_apples_correctness_l216_216389


namespace profit_per_cake_l216_216980

theorem profit_per_cake (ingredient_cost : ℝ) (packaging_cost : ℝ) (selling_price : ℝ) (cake_count : ℝ)
    (h1 : ingredient_cost = 12) (h2 : packaging_cost = 1) (h3 : selling_price = 15) (h4 : cake_count = 2) :
    selling_price - (ingredient_cost / cake_count + packaging_cost) = 8 := by
  sorry

end profit_per_cake_l216_216980


namespace value_of_expression_l216_216326

theorem value_of_expression (θ : ℝ) (h : cos θ - sin θ = 0) : 2 * sin θ * cos θ + cos (2 * θ) = 1 :=
by
  sorry

end value_of_expression_l216_216326


namespace number_of_Al_atoms_l216_216162

noncomputable def Al_weight : ℝ := 26.98
noncomputable def Cl_weight : ℝ := 35.45
noncomputable def Cl_atoms : ℝ := 3
noncomputable def compound_weight : ℝ := 132

theorem number_of_Al_atoms : 
  ∃ (n_Al_atoms : ℕ), (compound_weight - Cl_atoms * Cl_weight) / Al_weight ≈ n_Al_atoms ∧ n_Al_atoms = 1 :=
sorry

end number_of_Al_atoms_l216_216162


namespace largest_k_l216_216121

def n_acceptable (n k : ℕ) (group : set (set ℕ)) : Prop :=
  ∀ boy ∈ group, ∃ (subgroup : set (set ℕ)), 
    subgroup ⊆ group \ {boy} ∧ 
    subgroup.card = k - 1 ∧
    (∀ subsubgroup ∈ subgroup, subsubgroup.card = n)

theorem largest_k (n : ℕ) : 
  ∃ k, (∀ (group : set (set ℕ)), n_acceptable n k group → 
    ∃ subgroup, subgroup ⊆ group ∧ subgroup.card = n + 1 ∧
    (∀ subsubgroup ∈ subgroup, subsubgroup.card = n + 1)) ∧
  k = 2 * n - 1 :=
by
  sorry

end largest_k_l216_216121


namespace triangle_cos_ratio_l216_216435

theorem triangle_cos_ratio (A B C a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
    (angle_A : 0 < A ∧ A < π) (angle_B : 0 < B ∧ B < π) (angle_C : 0 < C ∧ C < π)
    (h_eq : cos C / cos B = (2 * a - c) / b) (h_sinA_ne_0 : sin A ≠ 0) : 
    B = π / 3 := 
by 
  -- Solution steps are omitted 
  sorry

end triangle_cos_ratio_l216_216435


namespace Chris_age_l216_216065

variable (a b c : ℚ)

-- Conditions
def condition1 : Prop := a + b + c = 36
def condition2 : Prop := c = 2 * a + 5
def condition3 : Prop := b = 1 / 2 * a - 1

-- Question: prove that c = 163 / 7
theorem Chris_age (h1 : condition1 a b c) (h2 : condition2 a b c) (h3 : condition3 a b c) : c = 163 / 7 :=
sorry

end Chris_age_l216_216065


namespace masking_tape_problem_l216_216645

variable (width_other : ℕ)

theorem masking_tape_problem
  (h1 : ∀ w : ℕ, (2 * 4 + 2 * w) = 20)
  : width_other = 6 :=
by
  have h2 : 8 + 2 * width_other = 20 := h1 width_other
  sorry

end masking_tape_problem_l216_216645


namespace reflected_graph_works_graph_of_negative_g_is_B_l216_216367

def g (x : ℝ) : ℝ := x^2 - 4

def reflected_g (x : ℝ) : ℝ := -g(x)

theorem reflected_graph_works (x : ℝ) :
  (reflected_g x = -x^2 + 4) := sorry

theorem graph_of_negative_g_is_B : 
  (∃ x, reflected_g x = -x^2 + 4) ↔ true := sorry

end reflected_graph_works_graph_of_negative_g_is_B_l216_216367


namespace measure_angle_C_l216_216758

theorem measure_angle_C (ABCD : parallelogram) (B : angle) (A : angle) 
  (angle_B_120 : measure_angle B = 120) 
  (angle_A_half_B : measure_angle A = measure_angle B / 2) : 
  measure_angle (angle_C ABCD) = 60 := 
  sorry

end measure_angle_C_l216_216758


namespace polynomial_value_at_neg2_l216_216017

noncomputable def P (x : ℝ) : ℝ :=
  x^2 + (1/24) * (x-1) * (x-2) * (x-3) * (x-4)

theorem polynomial_value_at_neg2 :
  P(0) = 1 →
  P(1) = 1 →
  P(2) = 4 →
  P(3) = 9 →
  P(4) = 16 →
  P(-2) = 19 :=
by
  intros h0 h1 h2 h3 h4
  rw [P] at *
  -- rest of the proof would follow, but it's skipped here
  sorry

end polynomial_value_at_neg2_l216_216017


namespace piravena_trip_cost_l216_216053

noncomputable def cost_trip : ℝ :=
  let CA := 3500
  let AB := 4000
  let plane_cost_per_km := 0.10
  let plane_booking_fee := 100
  let bus_cost_per_km := 0.15
  let BC := real.sqrt (4000^2 - 3500^2)
  let cost_A_to_B := min (AB * 0.15) (AB * plane_cost_per_km + plane_booking_fee)
  let cost_B_to_C := min (BC * bus_cost_per_km) (BC * plane_cost_per_km + plane_booking_fee)
  let cost_C_to_A := min (CA * 0.15) (CA * plane_cost_per_km + plane_booking_fee)
  cost_A_to_B + cost_B_to_C + cost_C_to_A

def total_cost_eq_1240_47 : Prop :=
  cost_trip = 1240.47

theorem piravena_trip_cost : total_cost_eq_1240_47 :=
 by sorry

end piravena_trip_cost_l216_216053


namespace particle_speed_l216_216170

def particle_position (t : ℝ) : ℝ × ℝ × ℝ := (3 * t + 8, 5 * t - 15, -t + 6)

theorem particle_speed :
  let Δx := 3
      Δy := 5
      Δz := -1
      speed := Real.sqrt (Δx^2 + Δy^2 + Δz^2)
  in speed = Real.sqrt 35 :=
by
  let Δx := 3
  let Δy := 5
  let Δz := -1
  let speed := Real.sqrt (Δx^2 + Δy^2 + Δz^2)
  show speed = Real.sqrt 35
  sorry

end particle_speed_l216_216170


namespace projection_of_AC_on_AB_l216_216438

noncomputable def projection_vector 
  (B C : ℝ) (BC : ℝ) (e : ℝ → ℝ) 
  (hB : B = Real.pi / 6) 
  (hC : C = Real.pi / 2) 
  (hBC : BC = 1) 
  (he_unit : ∀ x, ∥e x∥ = 1) 
  : ℝ → ℝ :=
  (sqrt 3 / 6) • e

theorem projection_of_AC_on_AB 
  (B C : ℝ) (BC : ℝ) (e : ℝ → ℝ) 
  (hB : B = Real.pi / 6) 
  (hC : C = Real.pi / 2) 
  (hBC : BC = 1) 
  (he_unit : ∀ x, ∥e x∥ = 1) 
  : (projection_vector B C BC e hB hC hBC he_unit) = (sqrt 3 / 6) • e :=
sorry

end projection_of_AC_on_AB_l216_216438


namespace minimum_a_div_x_l216_216027

theorem minimum_a_div_x (a x y : ℕ) (h1 : 100 < a) (h2 : 100 < x) (h3 : 100 < y) (h4 : y^2 - 1 = a^2 * (x^2 - 1)) :
  2 ≤ a / x :=
by sorry

end minimum_a_div_x_l216_216027


namespace collinear_PDT_l216_216418

-- Define the geometric setup with relevant points and circles.
variables (A B C D E F H O T K L P : Type) [PlaneGeometry A B C D E F H O T K L P]
           (abc_triangle : acute_triangle A B C)
           (AD_height : height AD A B C)
           (BE_height : height BE A B C)
           (CF_height : height CF A B C)
           (H_orthocenter : orthocenter H A B C)
           (O_circumcenter : circumcenter O A B C)
           (BC_tangents_intersect : tangents_intersect T B C (circumcircle A B C))
           (K_symmetric : symmetric K O AB)
           (L_symmetric : symmetric L O AC)
           (DFK_circle : circle_intersection (circle_through D F K) P)
           (DEL_circle : circle_intersection (circle_through D E L) P)

-- State the theorem to be proved.
theorem collinear_PDT : collinear P D T :=
sorry

end collinear_PDT_l216_216418


namespace transformed_parabola_is_correct_l216_216075

-- Definitions based on conditions
def original_parabola (x : ℝ) : ℝ := 3 * x^2 - 6 * x - 3
def shifted_left (x : ℝ) : ℝ := original_parabola (x - 2)
def shifted_up (y : ℝ) : ℝ := y + 2

-- Theorem statement
theorem transformed_parabola_is_correct :
  ∀ x : ℝ, shifted_up (shifted_left x) = 3 * x^2 + 6 * x - 1 :=
by 
  -- Proof will be filled in here
  sorry

end transformed_parabola_is_correct_l216_216075


namespace sum_of_numbers_l216_216084

theorem sum_of_numbers (x y : ℝ) (h1 : x * y = 12) (h2 : (1 / x) = 3 * (1 / y)) :
  x + y = 8 :=
sorry

end sum_of_numbers_l216_216084


namespace billy_total_tickets_l216_216991

theorem billy_total_tickets :
  let ferris_wheel_rides := 7
  let bumper_car_rides := 3
  let roller_coaster_rides := 4
  let teacups_rides := 5
  let ferris_wheel_cost := 5
  let bumper_car_cost := 6
  let roller_coaster_cost := 8
  let teacups_cost := 4
  let total_ferris_wheel := ferris_wheel_rides * ferris_wheel_cost
  let total_bumper_cars := bumper_car_rides * bumper_car_cost
  let total_roller_coaster := roller_coaster_rides * roller_coaster_cost
  let total_teacups := teacups_rides * teacups_cost
  let total_tickets := total_ferris_wheel + total_bumper_cars + total_roller_coaster + total_teacups
  total_tickets = 105 := 
sorry

end billy_total_tickets_l216_216991


namespace estimated_population_correlation_coefficient_l216_216229

variables (xi yi : Fin 20 → ℝ)       -- xi represents plant coverage areas, yi represents number of wild animals
variables (sum_x : ℝ) (sum_y : ℝ) (sum_x_diff_sq : ℝ) (sum_y_diff_sq : ℝ) (sum_xy_diff : ℝ)

-- Conditions
def conditions := 
  (∑ i, xi i = 60) ∧ 
  (∑ i, yi i = 1200) ∧ 
  (∑ i, (xi i - (∑ j, xi j) / 20) ^ 2 = 80) ∧ 
  (∑ i, (yi i - (∑ j, yi j) / 20) ^ 2 = 9000) ∧ 
  (∑ i, (xi i - (∑ j, xi j) / 20) * (yi i - (∑ j, yi j) / 20) = 800)

-- Part 1: Prove the estimated population
theorem estimated_population (h : conditions xi yi sum_x sum_y sum_x_diff_sq sum_y_diff_sq sum_xy_diff) :
  200 * (sum_y / 20) = 12000 :=
by sorry

-- Part 2: Prove the correlation coefficient
theorem correlation_coefficient (h : conditions xi yi sum_x sum_y sum_x_diff_sq sum_y_diff_sq sum_xy_diff) :
  (sum_xy_diff / (real.sqrt (sum_x_diff_sq * sum_y_diff_sq))).round_to 2 = 0.94 :=
by sorry

end estimated_population_correlation_coefficient_l216_216229


namespace abc_value_l216_216490

theorem abc_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h1 : a * (b + c) = 171) 
    (h2 : b * (c + a) = 180) 
    (h3 : c * (a + b) = 189) :
    a * b * c = 270 :=
by
  -- Place proofs here
  sorry

end abc_value_l216_216490


namespace range_a_satisfies_l216_216740

theorem range_a_satisfies (a : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x : ℝ, f x = x^3) (h₂ : f 2 = 8) :
  (f (a - 3) > f (1 - a)) ↔ a > 2 :=
by
  sorry

end range_a_satisfies_l216_216740


namespace smallest_integer_with_exactly_200_divisors_l216_216515

theorem smallest_integer_with_exactly_200_divisors:
  ∃ (n m k : ℕ), 
  (∃ (d : ℕ), finset.card (finset.filter (λ x, n % x = 0) (finset.range (n + 1))) = d) ∧
  (finset.card (finset.filter (λ x, n % x = 0) (finset.range (n + 1))) = 200) ∧
  n = m * 10^k ∧
  ¬ (10 ∣ m) ∧
  m + k = 18 :=
begin
  sorry
end

end smallest_integer_with_exactly_200_divisors_l216_216515


namespace perp_lines_l216_216425

theorem perp_lines (A B C D E S : Type) 
    [InnerProductSpace ℝ A] [FiniteDimensional ℝ A]
    [InnerProductSpace ℝ B] [FiniteDimensional ℝ B]
    [InnerProductSpace ℝ C] [FiniteDimensional ℝ C]
    [InnerProductSpace ℝ D] [FiniteDimensional ℝ D]
    [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
    [InnerProductSpace ℝ S] [FiniteDimensional ℝ S]
    (h1 : ∠ A B D = ∠ A C E)
    (h2 : ∠ A C B = ∠ A C D)
    (h3 : ∠ A D C = ∠ A D E)
    (h4 : ∠ A D B = ∠ A E C)
    (hS : S = intersection BD CE) :
  ∠ S Z C = 90 :=
by
  sorry

end perp_lines_l216_216425


namespace snow_total_inches_l216_216619

theorem snow_total_inches (initial_snow_ft : ℝ) (additional_snow_in : ℝ)
  (melted_snow_in : ℝ) (multiplier : ℝ) (days_after : ℕ) (conversion_rate : ℝ)
  (initial_snow_in : ℝ) (fifth_day_snow_in : ℝ) :
  initial_snow_ft = 0.5 →
  additional_snow_in = 8 →
  melted_snow_in = 2 →
  multiplier = 2 →
  days_after = 5 →
  conversion_rate = 12 →
  initial_snow_in = initial_snow_ft * conversion_rate →
  fifth_day_snow_in = multiplier * initial_snow_in →
  (initial_snow_in + additional_snow_in - melted_snow_in + fifth_day_snow_in) / conversion_rate = 2 :=
by
  sorry

end snow_total_inches_l216_216619


namespace car_speed_l216_216938

theorem car_speed (distance time : ℝ) (h₁ : distance = 50) (h₂ : time = 5) : (distance / time) = 10 :=
by
  rw [h₁, h₂]
  norm_num

end car_speed_l216_216938


namespace construct_circle_through_points_and_intersects_orthogonally_l216_216670

-- Define the given circle k1 with center O1 
variables (O1 A B : Point) (r1 : Real)

-- Define the type for the Proof Problem 
structure Circle : Type where
  center : Point
  radius : Real

def intersects_orthogonally (c1 c2 : Circle) : Prop :=
  ∃ (C : Point), 
    (C ∈ Circle.c1) ∧ (C ∈ Circle.c2) ∧ 
    (tangent_at c1 C) ⊥ (tangent_at c2 C)

-- Given circle k1
def k1 : Circle := {
  center := O1,
  radius := r1
}

-- Statement of the theorem
theorem construct_circle_through_points_and_intersects_orthogonally (O2 : Point) (r2 : Real) :
  ∃ (k2 : Circle), 
    (A ∈ Circle k2.center k2.radius) ∧ 
    (B ∈ Circle k2.center k2.radius) ∧ 
    intersects_orthogonally k1 k2 := 
begin
  sorry 
end

end construct_circle_through_points_and_intersects_orthogonally_l216_216670


namespace units_digit_product_l216_216907

theorem units_digit_product :
  ((734^99 + 347^83) % 10) * ((956^75 - 214^61) % 10) % 10 = 4 := by
  sorry

end units_digit_product_l216_216907


namespace box_height_proof_l216_216822

-- Define the conditions
def square_base (x : ℝ) : Prop := x > 0
def box_height (x : ℝ) : ℝ := 2 * x
def surface_area (x : ℝ) : ℝ := 10 * x^2
def minimum_surface_area : ℝ := 120
def integer_side_length (x : ℝ) : Prop := ∃ n : ℤ, n.to_real = x

-- Define the height of the box given the constraints
theorem box_height_proof (x : ℝ) (hx_pos : square_base x) (hx_int : integer_side_length x) 
  (h_area : surface_area x ≥ minimum_surface_area) : 
  box_height x = 8 := by
  sorry

end box_height_proof_l216_216822


namespace sum_initial_values_l216_216959

noncomputable def complex_seq (c d : ℕ → ℂ) : Prop := 
  ∀ n : ℕ, c (n + 1) = 2 * c n - √3 * d n ∧ d (n + 1) = √3 * c n + 2 * d n

def initial_condition (c d: ℕ → ℝ) : Prop := 
  c 10 = 3 ∧ d 10 = 5

theorem sum_initial_values (c d: ℕ → ℝ) 
  (h_seq : complex_seq c d)
  (h_init : initial_condition c d) : 
  c 1 + d 1 = 5 / (7 ^ 4.5) :=
sorry

end sum_initial_values_l216_216959


namespace parabola_tangent_midpoint_l216_216950

theorem parabola_tangent_midpoint (p : ℝ) (h : p > 0) :
    (∃ M : ℝ × ℝ, M = (2, -2*p)) ∧ 
    (∃ A B : ℝ × ℝ, A ≠ B ∧ 
                      (∃ yA yB : ℝ, yA = (A.1^2)/(2*p) ∧ yB = (B.1^2)/(2*p)) ∧ 
                      (0.5 * (A.2 + B.2) = 6)) → p = 1 := by sorry

end parabola_tangent_midpoint_l216_216950


namespace power_modulo_lemma_problem_statement_l216_216908

-- Statement only, proof omitted
theorem power_modulo_lemma (a b n : ℕ) (h_pow : a^n ≡ 1 [MOD b]) :
  a^(n * k + m) ≡ a^m [MOD b] :=
by sorry

theorem problem_statement (h: 8^4 ≡ 1 [MOD 13]) : 
  8^1234 ≡ 12 [MOD 13] :=
by
  have k := 1234 / 4
  have m := 1234 % 4
  have h_main := power_modulo_lemma 8 13 4 h
  calc
    8^1234 ≡ 8^(4 * k + m) [MOD 13] : by sigma_limit
    ... ≡ 8^m [MOD 13] : by rw h_main
    ... ≡ 8^2 [MOD 13] : by sigma_limit
    ... ≡ 12 [MOD 13] : by check_result

#check problem_statement

end power_modulo_lemma_problem_statement_l216_216908


namespace f_neg_one_f_two_l216_216031

def f (x : ℝ) : ℝ :=
  if x < 0 then 3 * x + 4 else 6 - 3 * x

theorem f_neg_one : f (-1) = 1 := by
  sorry

theorem f_two : f 2 = 0 := by
  sorry

end f_neg_one_f_two_l216_216031


namespace ordered_pairs_count_l216_216660

theorem ordered_pairs_count : 
  ∃ (c d : ℝ), (∃ (x y : ℤ), (c * x + d * y = 3) ∧ (x ^ 2 + y ^ 2 = 18)) ↔ (10: ℕ) :=
by
sorry

end ordered_pairs_count_l216_216660


namespace angle_APB_is_sum_l216_216145

-- Define the conditions and the problem
variables {P A B S R T : Point}
variables {a b : ℝ}
variables (circle1 : Circle S A R) (circle2 : Circle R B T)

-- Given conditions:
-- PA is tangent to semicircle SAR
-- PB is tangent to semicircle RBT
-- SRT is a straight line
def PA_tangent_to_SAR (h : ∀ (Q : Point), is_tangent P A Q circle1) : Prop := sorry
def PB_tangent_to_RBT (h : ∀ (Q : Point), is_tangent P B Q circle2) : Prop := sorry
def SRT_is_straight (h : straight_line S R T) : Prop := sorry

-- Prove that the angle APB is equal to a + b
theorem angle_APB_is_sum (h1 : PA_tangent_to_SAR h)
                         (h2 : PB_tangent_to_RBT h)
                         (h3 : SRT_is_straight h) :
  angle P A B = a + b := sorry

end angle_APB_is_sum_l216_216145


namespace no_solution_for_x6_eq_2y2_plus_2_l216_216266

theorem no_solution_for_x6_eq_2y2_plus_2 :
  ¬ ∃ (x y : ℤ), x^6 = 2 * y^2 + 2 :=
sorry

end no_solution_for_x6_eq_2y2_plus_2_l216_216266


namespace combinations_of_coins_30_cents_l216_216716

theorem combinations_of_coins_30_cents :
  let coins := [1, 5, 10, 25] in
  let target := 30 in
  (∃ (count: ℕ), (count = 113) ∧ 
  (∑ (x: ℕ) in coins, x * count) = target) := 
  by
    sorry

end combinations_of_coins_30_cents_l216_216716


namespace hallie_hours_worked_on_tuesday_l216_216715

theorem hallie_hours_worked_on_tuesday
    (hourly_wage : ℝ := 10)
    (hours_monday : ℝ := 7)
    (tips_monday : ℝ := 18)
    (hours_wednesday : ℝ := 7)
    (tips_wednesday : ℝ := 20)
    (tips_tuesday : ℝ := 12)
    (total_earnings : ℝ := 240)
    (tuesday_hours : ℝ) :
    (hourly_wage * hours_monday + tips_monday) +
    (hourly_wage * hours_wednesday + tips_wednesday) +
    (hourly_wage * tuesday_hours + tips_tuesday) = total_earnings →
    tuesday_hours = 5 :=
by
  sorry

end hallie_hours_worked_on_tuesday_l216_216715


namespace wholesale_price_of_machine_l216_216597

theorem wholesale_price_of_machine (R : ℝ) (hR : R = 167.99999999999997) (W : ℝ) :
  let SP := R * 0.90 in
  SP = W * 1.20 → W = 126.00 :=
by
  intros SP hSP,
  have h1 : SP = 167.99999999999997 * 0.90, from calc
    SP = R * 0.90 : rfl
    ... = 167.99999999999997 * 0.90 : by rw [hR],
  have h2 : W = 151.19999999999996 / 1.20, from calc
    151.19999999999996 = SP : by rw [h1, hSP]
    ... = W * 1.20 : by assumption
    let h3 : W = 126.00, by sorry,
    show W = 126.00 from h3 -- this is where we need to complete the proof

end wholesale_price_of_machine_l216_216597


namespace function_symmetry_l216_216863

theorem function_symmetry :
  ∀ x : ℝ, log (10 : ℝ) ((2 / (1 + x) - 1) : ℝ) = log (10 : ℝ) ((1 - x) / (1 + x) : ℝ) →
    log (10 : ℝ) ((1 - (-x)) / (1 + (-x)) : ℝ) = - log (10 : ℝ) ((1 - x) / (1 + x) : ℝ) :=
by
  intros x h
  sorry

end function_symmetry_l216_216863


namespace rain_difference_l216_216474

theorem rain_difference
    (rain_monday : ℕ → ℝ)
    (rain_tuesday : ℕ → ℝ)
    (rain_wednesday : ℕ → ℝ)
    (rain_thursday : ℕ → ℝ)
    (h_monday : ∀ n : ℕ, n = 10 → rain_monday n = 1.25)
    (h_tuesday : ∀ n : ℕ, n = 12 → rain_tuesday n = 2.15)
    (h_wednesday : ∀ n : ℕ, n = 8 → rain_wednesday n = 1.60)
    (h_thursday : ∀ n : ℕ, n = 6 → rain_thursday n = 2.80) :
    let total_rain_monday := 10 * 1.25
    let total_rain_tuesday := 12 * 2.15
    let total_rain_wednesday := 8 * 1.60
    let total_rain_thursday := 6 * 2.80
    (total_rain_tuesday + total_rain_thursday) - (total_rain_monday + total_rain_wednesday) = 17.3 :=
by
  sorry

end rain_difference_l216_216474


namespace duty_arrangement_prob_l216_216086

theorem duty_arrangement_prob :
  let teachers : Finset ℕ := {0, 1, 2, 3, 4, 5} in
  let days : Finset (Finset ℕ) := {⟨{0, 1}, sorry⟩, ⟨{2, 3}, sorry⟩, ⟨{4, 5}, sorry⟩} in
  let valid_arrangements := 
    (teachers.powerset.filter (λ s, s.card = 2)).to_finset * 
    (teachers.powerset.filter (λ s, s.card = 2)).to_finset * 
    (teachers.powerset.filter (λ s, s.card = 2)).to_finset in
  let valid :=
    valid_arrangements.filter (
      λ arrangement, 
        ((0 ∉ arrangement.1 ∧ ∉ arrangement.2) ∨ 
        (0 ∉ arrangement.1 ∧ ∉ arrangement.3) ∨ 
        (0 ∉ arrangement.2 ∧ ∉ arrangement.3)) ∧ 
        ((1 ∉ arrangement.1 ∧ ∉ arrangement.3) ∨ 
        (1 ∉ arrangement.1 ∧ ∉ arrangement.2) ∨ 
        (1 ∉ arrangement.2 ∧ ∉ arrangement.3))
    ) in
  ((valid.card : ℚ) / (valid_arrangements.card : ℚ)) = 7 / 15 := sorry

end duty_arrangement_prob_l216_216086


namespace auto_credit_percentage_l216_216617

noncomputable def auto_credit_extended : ℝ := 50
noncomputable def consumer_credit : ℝ := 465.1162790697675
noncomputable def total_auto_credit : ℝ := auto_credit_extended * 4

theorem auto_credit_percentage :
  ((total_auto_credit / consumer_credit) * 100) ≈ 42.99 := by
  sorry

end auto_credit_percentage_l216_216617


namespace probability_other_child_girl_l216_216944

theorem probability_other_child_girl 
  (a b : Bool) -- True for boy, False for girl
  (h : a = true ∨ b = true) :
  (cond (a = true) (if b = false then 1 else 0)
  + cond (a = false) (if b = true then 1 else 0))
  / (cond (a = true) (1) + cond (a = false) (if b = true then 1 else 0)) = 2 / 3 :=
sorry

end probability_other_child_girl_l216_216944


namespace midpoint_of_AE_l216_216809

theorem midpoint_of_AE
  (A B C D E : Point)
  (tABC : Triangle A B C)
  (circC_B : Circle)
  (circC_C : Circle)
  (hCB_pass : B ∈ circC_B)
  (hCB_tangent : Tangent circC_B (LineSegment A C) A)
  (hCC_pass : C ∈ circC_C)
  (hCC_tangent : Tangent circC_C (LineSegment A B) A)
  (hD_intersect : D ∈ (circC_B ∩ circC_C))
  (hAD_circum : E ∈ (Circumcircle tABC) ∧ (AD : Line A D) = Line A E \ {A})
  : Midpoint D A E :=
sorry

end midpoint_of_AE_l216_216809


namespace find_remainder_l216_216631

-- Definition of N based on given conditions
def N : ℕ := 44 * 432

-- Definition of next multiple of 432
def next_multiple_of_432 : ℕ := N + 432

-- Statement to prove the remainder when next_multiple_of_432 is divided by 39 is 12
theorem find_remainder : next_multiple_of_432 % 39 = 12 := 
by sorry

end find_remainder_l216_216631


namespace max_take_home_pay_income_l216_216750

theorem max_take_home_pay_income (x : ℤ) : 
  (1000 * 2 * 50) - 20 * 50^2 = 100000 := 
by 
  sorry

end max_take_home_pay_income_l216_216750


namespace monomial_properties_l216_216855

variable (x y : ℝ)

def monomial : ℝ := - (2 * x * y^2) / 3

theorem monomial_properties :
  let coeff := - (2 / 3)
  let degree := 3
  (∀ (x y : ℝ), monomial x y = coeff * x * y^2) ∧ (degree = 1 + 2) :=
by
  sorry

end monomial_properties_l216_216855


namespace incorrect_statement_about_methods_l216_216503

-- Definitions as per the conditions
def synthetic_method := 
  "A method that starts with given conditions to deduce the desired conclusion, also called the deductive method or cause-to-effect method."

def analytic_method := 
  "A method that begins with the desired conclusion and works backward to establish the condition that leads to this conclusion, also called the inductive method or effect-to-cause method."

-- The statement to be proved
theorem incorrect_statement_about_methods : 
  ¬ (synthetic_method ∧ analytic_method ∧ "Both the synthetic method and analytic method are methods that concurrently deduce cause and effect.") := 
by
  sorry

end incorrect_statement_about_methods_l216_216503


namespace length_of_escalator_l216_216978

-- Define the conditions
def escalator_speed : ℝ := 15 -- ft/sec
def person_speed : ℝ := 5 -- ft/sec
def time_taken : ℝ := 10 -- sec

-- Define the length of the escalator
def escalator_length (escalator_speed : ℝ) (person_speed : ℝ) (time : ℝ) : ℝ := 
  (escalator_speed + person_speed) * time

-- Theorem to prove
theorem length_of_escalator : escalator_length escalator_speed person_speed time_taken = 200 := by
  sorry

end length_of_escalator_l216_216978


namespace ZhangSan_correct_probability_l216_216881

namespace ZhangSan

-- Define the total number of questions
def total_questions : ℕ := 4

-- Define the probability of Zhang San having an idea for a question
def P_A1 : ℚ := 3 / 4

-- Define the probability of Zhang San being unclear about a question
def P_A2 : ℚ := 1 / 4

-- Define the probability of answering correctly given an idea
def P_B_given_A1 : ℚ := 3 / 4

-- Define the probability of answering correctly given an unclear status
def P_B_given_A2 : ℚ := 1 / 4

-- Define the probability of answering a question correctly
def P_B : ℚ := P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2

theorem ZhangSan_correct_probability :
  P_B = 5 / 8 :=
by
  unfold P_B P_A1 P_A2 P_B_given_A1 P_B_given_A2
  sorry

end ZhangSan

end ZhangSan_correct_probability_l216_216881


namespace estimated_population_correct_correlation_coefficient_correct_l216_216211

variables (x y : ℕ → ℝ)
variables (n : ℕ) (plots : ℕ)

noncomputable def average_y := (∑ i in finset.range 20, y i) / 20

noncomputable def estimated_population := average_y y * plots

theorem estimated_population_correct (h_sum_y : ∑ i in finset.range 20, y i = 1200)
                                      (h_plots : plots = 200) :
  estimated_population y plots = 12000 :=
by sorry

noncomputable def correlation_coefficient (x y : ℕ → ℝ) :=
  let sum_xy := ∑ i in finset.range 20, (x i - ((∑ j in finset.range 20, x j) / 20)) * 
                               (y i - ((∑ j in finset.range 20, y j) / 20)) in
  let sum_xx := ∑ i in finset.range 20, (x i - ((∑ j in finset.range 20, x j) / 20)) ^ 2 in
  let sum_yy := ∑ i in finset.range 20, (y i - ((∑ j in finset.range 20, y j) / 20)) ^ 2 in
  sum_xy / (real.sqrt (sum_xx * sum_yy))

theorem correlation_coefficient_correct (h_sum_xx : ∑ i in finset.range 20, (x i - ((∑ j in finset.range 20, x j) / 20)) ^ 2 = 80)
                                        (h_sum_yy : ∑ i in finset.range 20, (y i - ((∑ j in finset.range 20, y j) / 20)) ^ 2 = 9000)
                                        (h_sum_xy : ∑ i in finset.range 20, (x i - ((∑ j in finset.range 20, x j) / 20)) * (y i - ((∑ j in finset.range 20, y j) / 20)) = 800) :
  correlation_coefficient x y ≈ 0.94 :=
by sorry

end estimated_population_correct_correlation_coefficient_correct_l216_216211


namespace orthocenter_condition_l216_216814

variables {A B C D : Type} [normed_field A] [normed_space A B] 

noncomputable def is_interior_point_of_acute_triangle
  (D : A) (A B C : A)
  (h_acute : ∃ α β γ : ℝ, 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2 ∧ 0 < γ ∧ γ < π/2 ∧ α + β + γ = π) 
  (h_condition : ∀ (DA DB DC AB BC CA : ℝ), 
    DA * DB * AB + DB * DC * BC + DC * DA * CA = AB * BC * CA) 
  : Prop :=
sorry

theorem orthocenter_condition 
  (A B C D : A)
  (cond : is_interior_point_of_acute_triangle D A B C
    (exists.intro _ (and.intro _ (and.intro _ (and.intro _ (and.intro _ (and.intro _ rfl))))))
    (λ DA DB DC AB BC CA, DA * DB * AB + DB * DC * BC + DC * DA * CA = AB * BC * CA)) 
  : ∃(ortho : A), ortho = D :=
sorry

end orthocenter_condition_l216_216814


namespace nut_composer_l216_216169

-- Let a nut be defined as a geometric body composed of another geometric body with a part removed.
structure Nut where
  outer_body : Type
  inner_body : Type
  outer_body_structure : String
  inner_body_structure : String

-- Define the structural characteristic as a condition
def is_nut_structure : Nut → Prop :=
  λ nut, nut.outer_body_structure = "regular hexagonal prism" ∧ nut.inner_body_structure = "cylinder"

-- We state that if the nut is defined with the described structure, it must consist of these geometries
theorem nut_composer (nut : Nut) (h : is_nut_structure nut) : nut.outer_body = "Regular Hexagonal Prism" ∧ nut.inner_body = "Cylinder" :=
by
  sorry

end nut_composer_l216_216169


namespace y_sum_of_circle_intersections_on_y_axis_l216_216250

def circle_equation (x y : ℝ) : Prop :=
  (x + 8) ^ 2 + (y - 3) ^ 2 = 225

theorem y_sum_of_circle_intersections_on_y_axis :
  (∑ y in { y : ℝ | ∃ x, x = 0 ∧ circle_equation x y }, y) = 6 :=
by
  sorry

end y_sum_of_circle_intersections_on_y_axis_l216_216250


namespace hyperbola_asymptotes_l216_216368

theorem hyperbola_asymptotes (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (h_e : (sqrt 5) / 2 = sqrt (1 + (b^2 / a^2))) :
  (∀ x, y = b/a * x ∨ y = -b/a * x) ↔ b = a / 2 :=
sorry

end hyperbola_asymptotes_l216_216368


namespace even_n_iff_divisible_323_l216_216663

theorem even_n_iff_divisible_323 (n : ℕ) (h_pos : 0 < n) :
  (20^n + 16^n - 3^n - 1) % 323 = 0 ↔ even n :=
begin
  sorry
end

end even_n_iff_divisible_323_l216_216663


namespace total_apples_bought_l216_216392

def apples_bought_by_each_man := 30
def apples_more_than_men := 20
def apples_bought_by_each_woman := apples_bought_by_each_man + apples_more_than_men

theorem total_apples_bought (men women : ℕ) (apples_bought_by_each_man : ℕ) 
  (apples_bought_by_each_woman : ℕ) (apples_more_than_men : ℕ) : 
  men = 2 → women = 3 → apples_bought_by_each_man = 30 → apples_more_than_men = 20 → 
  apples_bought_by_each_woman = apples_bought_by_each_man + apples_more_than_men →
  (men * apples_bought_by_each_man + women * apples_bought_by_each_woman) = 210 := 
by
  intros menTwo menThree womApples menApples moreApples _ _ _ _ eqn
  simp only [menTwo, menThree, womApples, menApples, moreApples, eqn]
  -- state infering
  exact eq.refl 210

end total_apples_bought_l216_216392


namespace bea_glasses_sold_is_10_l216_216618

variable (B : ℕ)
variable (earnings_bea earnings_dawn : ℕ)

def bea_price_per_glass := 25
def dawn_price_per_glass := 28
def dawn_glasses_sold := 8
def earnings_diff := 26

def bea_earnings := bea_price_per_glass * B
def dawn_earnings := dawn_price_per_glass * dawn_glasses_sold

def bea_earnings_greater := bea_earnings = dawn_earnings + earnings_diff

theorem bea_glasses_sold_is_10 (h : bea_earnings_greater) : B = 10 :=
by sorry

end bea_glasses_sold_is_10_l216_216618


namespace snow_total_inches_l216_216620

theorem snow_total_inches (initial_snow_ft : ℝ) (additional_snow_in : ℝ)
  (melted_snow_in : ℝ) (multiplier : ℝ) (days_after : ℕ) (conversion_rate : ℝ)
  (initial_snow_in : ℝ) (fifth_day_snow_in : ℝ) :
  initial_snow_ft = 0.5 →
  additional_snow_in = 8 →
  melted_snow_in = 2 →
  multiplier = 2 →
  days_after = 5 →
  conversion_rate = 12 →
  initial_snow_in = initial_snow_ft * conversion_rate →
  fifth_day_snow_in = multiplier * initial_snow_in →
  (initial_snow_in + additional_snow_in - melted_snow_in + fifth_day_snow_in) / conversion_rate = 2 :=
by
  sorry

end snow_total_inches_l216_216620


namespace total_population_correct_l216_216525

/-- Define the populations of each city -/
def Population.Seattle : ℕ := sorry
def Population.LakeView : ℕ := 24000
def Population.Boise : ℕ := (3 * Population.Seattle) / 5

/-- Population of Lake View is 4000 more than the population of Seattle -/
axiom lake_view_population : Population.LakeView = Population.Seattle + 4000

/-- Define the total population -/
def total_population : ℕ :=
  Population.Seattle + Population.LakeView + Population.Boise

/-- Prove that total population of the three cities is 56000 -/
theorem total_population_correct :
  total_population = 56000 :=
sorry

end total_population_correct_l216_216525


namespace ratio_of_awards_l216_216784

theorem ratio_of_awards 
  (Scott_awards : ℕ) (Scott_awards_eq : Scott_awards = 4)
  (Jessie_awards : ℕ) (Jessie_awards_eq : Jessie_awards = 3 * Scott_awards)
  (rival_awards : ℕ) (rival_awards_eq : rival_awards = 24) :
  rival_awards / Jessie_awards = 2 :=
by sorry

end ratio_of_awards_l216_216784


namespace number_of_candidates_l216_216066

theorem number_of_candidates (average_marks total_marks : ℕ) (h1 : average_marks = 35) (h2 : total_marks = 4200) : 
  total_marks / average_marks = 120 :=
by
  have number_of_candidates := total_marks / average_marks
  show number_of_candidates = 120
sorry

end number_of_candidates_l216_216066


namespace female_to_male_ratio_l216_216243

variable (f m : ℕ)

-- Given conditions:
def avg_weight_female_members (w_f : ℕ) := w_f = 55
def avg_weight_male_members (w_m : ℕ) := w_m = 80
def overall_avg_weight_of_members (avg_w : ℕ) := avg_w = 70

-- Lean 4 statement for the proof problem:
theorem female_to_male_ratio (w_f w_m avg_w : ℕ) 
  (h1 : avg_weight_female_members w_f) 
  (h2 : avg_weight_male_members w_m) 
  (h3 : overall_avg_weight_of_members avg_w) : 
  (f : ℕ) (m : ℕ) :=
  (10 * m = 15 * f) → (f / m = 2 / 3) :=
by {
  -- You don't need to provide the proof, just the statement
  -- that sets up the proof environment.
  sorry
}

end female_to_male_ratio_l216_216243


namespace tetrahedron_edge_square_sum_l216_216479

variable (A B C D : Point)
variable (AB AC AD BC BD CD : ℝ) -- Lengths of the edges
variable (m₁ m₂ m₃ : ℝ) -- Distances between the midpoints of the opposite edges

theorem tetrahedron_edge_square_sum:
  (AB ^ 2 + AC ^ 2 + AD ^ 2 + BC ^ 2 + BD ^ 2 + CD ^ 2) =
  4 * (m₁ ^ 2 + m₂ ^ 2 + m₃ ^ 2) :=
  sorry

end tetrahedron_edge_square_sum_l216_216479


namespace initial_percentage_filled_l216_216937

theorem initial_percentage_filled {P : ℝ} 
  (h1 : 45 + (P / 100) * 100 = (3 / 4) * 100) : 
  P = 30 := by
  sorry

end initial_percentage_filled_l216_216937


namespace large_box_count_l216_216260

variable (x y : ℕ)

theorem large_box_count (h₁ : x + y = 21) (h₂ : 120 * x + 80 * y = 2000) : x = 8 := by
  sorry

end large_box_count_l216_216260


namespace acute_triangle_exterior_angles_obtuse_l216_216419

theorem acute_triangle_exterior_angles_obtuse
  (A B C : ℝ)
  (hA : 0 < A ∧ A < π / 2)
  (hB : 0 < B ∧ B < π / 2)
  (hC : 0 < C ∧ C < π / 2)
  (h_sum : A + B + C = π) :
  ∀ α β γ, α = A + B → β = B + C → γ = C + A → α > π / 2 ∧ β > π / 2 ∧ γ > π / 2 :=
by
  sorry

end acute_triangle_exterior_angles_obtuse_l216_216419


namespace vector_field_flux_l216_216289

-- Define the vector field F
def vector_field (x y z : ℝ) : ℝ × ℝ × ℝ :=
  (x - 2 * y + 1, 2 * x + y - 3 * z, 3 * y + z)

-- Define the surface of the sphere
def sphere (x y z : ℝ) : Prop :=
  x^2 + y^2 + z^2 = 1

-- Define the region constraints
def region_constraints (x y z : ℝ) : Prop :=
  (x ≥ 0) ∧ (y ≥ 0) ∧ (z ≥ 0) ∧ (x^2 + y^2 + z^2 > 1)

-- Define the entire problem as a Lean statement
theorem vector_field_flux :
  ∫∫ (θ ∈ set.Icc (0 : ℝ) (π / 2)) (φ ∈ set.Icc (0 : ℝ) (π / 2)), 
      let x := cos φ * sin θ in
      let y := sin φ * sin θ in
      let z := cos θ in
      let F := vector_field x y z in
      let n := (x, y, z) in
      (F.1 * n.1 + F.2 * n.2 + F.3 * n.3) * (sin θ)
  = (3 * π) / 4 := 
sorry

end vector_field_flux_l216_216289


namespace dice_sum_not_18_l216_216899

theorem dice_sum_not_18 (d1 d2 d3 d4 : ℕ) (h1 : d1 ∈ ({1, 2, 3, 4, 5, 6} : set ℕ))
                                     (h2 : d2 ∈ ({1, 2, 3, 4, 5, 6} : set ℕ))
                                     (h3 : d3 ∈ ({1, 2, 3, 4, 5, 6} : set ℕ))
                                     (h4 : d4 ∈ ({1, 2, 3, 4, 5, 6} : set ℕ))
                                     (prod_eq : d1 * d2 * d3 * d4 = 216) :
  d1 + d2 + d3 + d4 ≠ 18 :=
by sorry

end dice_sum_not_18_l216_216899


namespace three_city_population_l216_216531

noncomputable def totalPopulation (boise seattle lakeView: ℕ) : ℕ :=
  boise + seattle + lakeView

theorem three_city_population (pBoise pSeattle pLakeView : ℕ)
  (h1 : pBoise = 3 * pSeattle / 5)
  (h2 : pLakeView = pSeattle + 4000)
  (h3 : pLakeView = 24000) :
  totalPopulation pBoise pSeattle pLakeView = 56000 := by
  sorry

end three_city_population_l216_216531


namespace calvin_scorpions_l216_216628

/-- Calvin's bug collection problem -/
theorem calvin_scorpions :
  ∃ S : ℕ, 
    let roaches := 12 in
    let crickets := roaches / 2 in
    let caterpillars := 2 * S in
    roaches + S + crickets + caterpillars = 27 ∧ S = 3 :=
by
  sorry

end calvin_scorpions_l216_216628


namespace count_squares_in_region_l216_216723

theorem count_squares_in_region :
  let region := {p : ℝ × ℝ | let (x, y) := p in 
                                 y ≤ π * x ∧
                                 y ≥ -0.1 ∧
                                 x ≤ 5.1} in
  ∃ n : ℕ, n = 50 ∧
  (∀ (x1 y1 x2 y2 : ℤ), (x1:ℝ) ≤ 5 ∧ x2 = x1 + 1 ∧ y1 = 0 ∧ 
  y2 = y1 + 1 → (x2:ℝ, y2:ℝ) ∈ region) :=
begin
  sorry
end

end count_squares_in_region_l216_216723


namespace negative_trig_value_l216_216626

theorem negative_trig_value :
  (∃ x, x ∈ [sin (-1000 * (π / 180)), cos (-2200 * (π / 180)), tan (-10), 
  (sin (7 * π / 10) * cos π) / (tan (17 * π / 9))] ∧ x < 0) ↔ (tan (-10) < 0) :=
by
  sorry

end negative_trig_value_l216_216626


namespace scientific_notation_per_capita_GDP_l216_216606

theorem scientific_notation_per_capita_GDP (GDP : ℝ) (h : GDP = 104000): 
  GDP = 1.04 * 10^5 := 
by
  sorry

end scientific_notation_per_capita_GDP_l216_216606


namespace estimate_population_correct_correlation_coefficient_correct_l216_216215

namespace EcologicalSurvey

def estimated_population (total_animals : ℕ) (plots : ℕ) : ℕ :=
  (total_animals / 20) * plots

theorem estimate_population_correct :
  estimated_population 1200 200 = 12000 :=
by 
  -- Proof would go here.
  sorry

def correlation_coefficient (sum_xi_xbar : ℝ) (sum_yi_ybar : ℝ) (sum_prod_xi_yi : ℝ) : ℝ :=
  sum_prod_xi_yi / Math.sqrt (sum_xi_xbar * sum_yi_ybar)

theorem correlation_coefficient_correct :
  correlation_coefficient 80 9000 800 ≈ 0.94 :=
by 
  -- Proof would go here.
  sorry

end EcologicalSurvey

end estimate_population_correct_correlation_coefficient_correct_l216_216215


namespace find_larger_number_l216_216112

variable (x y : ℝ)
axiom h1 : x + y = 27
axiom h2 : x - y = 5

theorem find_larger_number : x = 16 :=
by
  sorry

end find_larger_number_l216_216112


namespace number_of_paths_from_A_to_B_l216_216157

-- Definitions based on conditions
def grid := fin 6 
def direction_arrow (a b : grid × grid) : Prop := sorry -- Assuming a predicate defining the directional arrows constraints
def A : grid × grid := (0, 0)
def B : grid × grid := (5, 5)
def valid_path (p : list (grid × grid)) : Prop :=
  p.head = A ∧ p.last = B ∧ (∀ (i : ℕ) (h : i < p.length - 1), direction_arrow (p.nth_le i h) (p.nth_le (i + 1) (by simp [h])))

-- Statement of the proof
theorem number_of_paths_from_A_to_B : 
  ∃ (n : ℕ), n = 20 ∧ ∀ (p : list (grid × grid)), valid_path p → true
:= sorry

end number_of_paths_from_A_to_B_l216_216157


namespace number_of_workers_l216_216494

theorem number_of_workers 
  (W : ℕ) 
  (h1 : 750 * W = (5 * 900) + 700 * (W - 5)) : 
  W = 20 := 
by 
  sorry

end number_of_workers_l216_216494


namespace total_amount_of_money_l216_216888

noncomputable def initial_amounts (t : ℕ) : Prop :=
  ∃ a j : ℕ, 
    let Amy_amount_after_Amy = a - 2 * t - 2 * j in
    let Jan_amount_after_Jan = j - (2 * Amy_amount_after_Amy + 2 * t) in
    let Toy_amount_after_Toy := 3 * Toy_new after Jan - 2 * Amy_amount_after_Amy - 2 * Jan_amount_after_Son in
    Toy_amount_after_Toy = 27

theorem total_amount_of_money : (∃ a j : ℕ, initial_amounts 27) → a + j + 27 = 120 :=
by
  intro h
  cases' h with a ha
  sorry

end total_amount_of_money_l216_216888


namespace estimated_population_correlation_coefficient_l216_216227

variables (xi yi : Fin 20 → ℝ)       -- xi represents plant coverage areas, yi represents number of wild animals
variables (sum_x : ℝ) (sum_y : ℝ) (sum_x_diff_sq : ℝ) (sum_y_diff_sq : ℝ) (sum_xy_diff : ℝ)

-- Conditions
def conditions := 
  (∑ i, xi i = 60) ∧ 
  (∑ i, yi i = 1200) ∧ 
  (∑ i, (xi i - (∑ j, xi j) / 20) ^ 2 = 80) ∧ 
  (∑ i, (yi i - (∑ j, yi j) / 20) ^ 2 = 9000) ∧ 
  (∑ i, (xi i - (∑ j, xi j) / 20) * (yi i - (∑ j, yi j) / 20) = 800)

-- Part 1: Prove the estimated population
theorem estimated_population (h : conditions xi yi sum_x sum_y sum_x_diff_sq sum_y_diff_sq sum_xy_diff) :
  200 * (sum_y / 20) = 12000 :=
by sorry

-- Part 2: Prove the correlation coefficient
theorem correlation_coefficient (h : conditions xi yi sum_x sum_y sum_x_diff_sq sum_y_diff_sq sum_xy_diff) :
  (sum_xy_diff / (real.sqrt (sum_x_diff_sq * sum_y_diff_sq))).round_to 2 = 0.94 :=
by sorry

end estimated_population_correlation_coefficient_l216_216227


namespace decreasing_interval_of_f_l216_216508

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + 7

def f_prime (x : ℝ) : ℝ := 6 * x^2 - 12 * x

theorem decreasing_interval_of_f : ∀ x, 0 ≤ x ∧ x ≤ 2 → f_prime(x) ≤ 0 :=
by
  intro x
  sorry

end decreasing_interval_of_f_l216_216508


namespace cheapest_candle_cost_to_measure_1_minute_l216_216992

-- Definitions

def big_candle_cost := 16 -- cost of a big candle in cents
def big_candle_burn_time := 16 -- burn time of a big candle in minutes
def small_candle_cost := 7 -- cost of a small candle in cents
def small_candle_burn_time := 7 -- burn time of a small candle in minutes

-- Problem statement
theorem cheapest_candle_cost_to_measure_1_minute :
  (∃ (n m : ℕ), n * big_candle_burn_time - m * small_candle_burn_time = 1 ∧
                 n * big_candle_cost + m * small_candle_cost = 97) :=
sorry

end cheapest_candle_cost_to_measure_1_minute_l216_216992


namespace estimated_population_is_correct_correlation_coefficient_is_correct_stratified_sampling_is_better_l216_216196

-- The sample data and conditions provided.
variables (x y : Fin 20 → ℝ)
variable (x̄ : ℝ := (∑ i, x i) / 20)
variable (ȳ : ℝ := (∑ i, y i) / 20)
variables (Sxx Syy Sxy : ℝ)
variables (n : ℕ := 20)
variables (total_plots : ℕ := 200)
variable (sum_x : ℝ := ∑ i, x i)
variable (sum_y : ℝ := ∑ i, y i)
variable (total_x : ℝ := 60)
variable (total_y : ℝ := 1200)

-- Given conditions
axiom condition1 : sum_x = 60
axiom condition2 : sum_y = 1200
axiom condition3 : Sxx = ∑ i, (x i - x̄) ^ 2
axiom condition4 : Syy = ∑ i, (y i - ȳ) ^ 2
axiom condition5 : Sxy = ∑ i, (x i - x̄) * (y i - ȳ)

-- Proven result 1: Estimated population of wild animals
def estimated_population (total_plots : ℕ) (avg_y : ℝ) : ℝ :=
  avg_y * total_plots

theorem estimated_population_is_correct : estimated_population total_plots (total_y / n) = 12000 :=
by
  -- Necessary calculations skipped
  sorry

-- Proven result 2: Correlation coefficient
noncomputable def correlation_coefficient (Sxy Sxx Syy : ℝ) : ℝ :=
  Sxy / (Real.sqrt (Sxx * Syy))

theorem correlation_coefficient_is_correct : correlation_coefficient Sxy Sxx Syy = 0.94 :=
by
  -- Assume given data and calculations are correct within required precision
  sorry

-- Proven result 3: Stratified sampling is more reasonable
theorem stratified_sampling_is_better : ∀ (method : String), method = "stratified sampling" → True :=
by
  -- Explanation skipped
  intros
  sorry

end estimated_population_is_correct_correlation_coefficient_is_correct_stratified_sampling_is_better_l216_216196


namespace hiker_distance_l216_216948

variable (s t d : ℝ)
variable (h₁ : (s + 1) * (2 / 3 * t) = d)
variable (h₂ : (s - 1) * (t + 3) = d)

theorem hiker_distance  : d = 6 :=
by
  sorry

end hiker_distance_l216_216948


namespace total_weight_is_correct_l216_216469

-- Defining the different weights in pounds
def BrieCheese := 0.5 -- pounds
def Bread := 1 -- pounds
def Tomatoes := 1 -- pounds
def Zucchini := 2 -- pounds
def ChickenBreasts := 1.5 -- pounds
def Raspberries := 0.5 -- pounds
def Blueberries := 0.5 -- pounds

-- The total weight of food purchased
def TotalWeight : ℝ := BrieCheese + Bread + Tomatoes + Zucchini + ChickenBreasts + Raspberries + Blueberries

-- Prove that the total weight is 7 pounds
theorem total_weight_is_correct : TotalWeight = 7 := by
  sorry

end total_weight_is_correct_l216_216469


namespace tanC_value_l216_216773

-- Definitions of conditions
variables {A B C : ℝ} (cotA cotB cotC : ℝ)
noncomputable def cot := λ θ : ℝ, 1 / (Real.tan θ)
noncomputable def tan := Real.tan

-- Hypotheses derived from conditions
axiom cotA_cotC_eq_one : cotA * cotC = 1
axiom cotB_cotC_eq_oneOver18 : cotB * cotC = 1 / 18

-- Concluding the value of tan C
theorem tanC_value (cotA cotB cotC : ℝ) (h1 : cotA * cotC = 1)
    (h2 : cotB * cotC = 1 / 18) : ∃ C : ℝ, tan C = 9 + 2*Real.sqrt 62 ∨ tan C = 9 - 2*Real.sqrt 62 :=
by
  sorry

end tanC_value_l216_216773


namespace cube_surface_area_ratio_l216_216140

variable (x : ℝ) (hx : x > 0)

theorem cube_surface_area_ratio (hx : x > 0):
  let side1 := 7 * x
  let side2 := x
  let SA1 := 6 * side1^2
  let SA2 := 6 * side2^2
  (SA1 / SA2) = 49 := 
by 
  sorry

end cube_surface_area_ratio_l216_216140


namespace max_marks_set_for_test_l216_216968

-- Define the conditions according to the problem statement
def passing_percentage : ℝ := 0.70
def student_marks : ℝ := 120
def marks_needed_to_pass : ℝ := 150
def passing_threshold (M : ℝ) : ℝ := passing_percentage * M

-- The maximum marks set for the test
theorem max_marks_set_for_test (M : ℝ) : M = 386 :=
by
  -- Given the conditions
  have h : passing_threshold M = student_marks + marks_needed_to_pass := sorry
  -- Solving for M
  sorry

end max_marks_set_for_test_l216_216968


namespace subsets_selection_count_l216_216961

open Finset

def T : Finset ℕ := {1, 2, 3, 4, 5, 6} -- Assuming distinct integers represent a, b, c, d, e, f

theorem subsets_selection_count :
  (∃ A B : Finset ℕ, A ∪ B = T ∧ (A ∩ B).card = 3) →
  (nat.choose 6 3 * 3^3) / 2 = 270 :=
by
  sorry

end subsets_selection_count_l216_216961


namespace math_proof_problem_l216_216816

noncomputable def even_function (f : ℝ → ℝ) := 
∀ x, f x = f (-x)

noncomputable def satisfied_condition (f : ℝ → ℝ) := 
∀ x, f (x + 1) = - f x

noncomputable def increasing_on_interval (f : ℝ → ℝ) := 
∀ x ∈ Icc (-1 : ℝ) (0 : ℝ), ∀ y ∈ Icc (-1 : ℝ) (0 : ℝ), x < y → f x < f y

theorem math_proof_problem (f : ℝ → ℝ) :
  even_function f →
  satisfied_condition f →
  increasing_on_interval f →
  (∀ x, f (x + 2) = f x) ∧ (∀ x, f (2 - x) = f (2 + x)) ∧ (f 2 = f 0) :=
by
  intros h1 h2 h3
  split
  sorry
  split
  sorry
  sorry

end math_proof_problem_l216_216816


namespace find_minimum_cos_diff_l216_216576

noncomputable section

theorem find_minimum_cos_diff
  (x y z : ℝ)
  (h1 : sqrt 3 * sin x = tan y)
  (h2 : 2 * sin y = cot z)
  (h3 : sin z = 2 * tan x) :
  ∃ (min_val : ℝ), min_val = (- 7 * sqrt 2 / 6) ∧ min_val = cos x - cos z :=
begin
  sorry
end

end find_minimum_cos_diff_l216_216576


namespace hacker_neo_problem_l216_216168

theorem hacker_neo_problem (n : ℕ) :
  ∃ m : ℕ, rearrange_digits (123 + 102 * n) < 1000 :=
sorry

-- Helper function for rearranging digits
def rearrange_digits (n : ℕ) : ℕ :=
-- Define your own implementation or use a placeholder.
sorry

end hacker_neo_problem_l216_216168


namespace find_p_q_r_l216_216348

def is_rel_prime (m n : ℕ) : Prop := Nat.gcd m n = 1

theorem find_p_q_r (x : ℝ) (p q r : ℕ)
  (h1 : (1 + Real.sin x) * (1 + Real.cos x) = 9 / 4)
  (h2 : (1 - Real.sin x) * (1 - Real.cos x) = p / q - Real.sqrt r)
  (hpq_rel_prime : is_rel_prime p q)
  (hp : 0 < p)
  (hq : 0 < q)
  (hr : 0 < r) :
  p + q + r = 26 :=
sorry

end find_p_q_r_l216_216348


namespace estimated_population_correlation_coefficient_l216_216228

variables (xi yi : Fin 20 → ℝ)       -- xi represents plant coverage areas, yi represents number of wild animals
variables (sum_x : ℝ) (sum_y : ℝ) (sum_x_diff_sq : ℝ) (sum_y_diff_sq : ℝ) (sum_xy_diff : ℝ)

-- Conditions
def conditions := 
  (∑ i, xi i = 60) ∧ 
  (∑ i, yi i = 1200) ∧ 
  (∑ i, (xi i - (∑ j, xi j) / 20) ^ 2 = 80) ∧ 
  (∑ i, (yi i - (∑ j, yi j) / 20) ^ 2 = 9000) ∧ 
  (∑ i, (xi i - (∑ j, xi j) / 20) * (yi i - (∑ j, yi j) / 20) = 800)

-- Part 1: Prove the estimated population
theorem estimated_population (h : conditions xi yi sum_x sum_y sum_x_diff_sq sum_y_diff_sq sum_xy_diff) :
  200 * (sum_y / 20) = 12000 :=
by sorry

-- Part 2: Prove the correlation coefficient
theorem correlation_coefficient (h : conditions xi yi sum_x sum_y sum_x_diff_sq sum_y_diff_sq sum_xy_diff) :
  (sum_xy_diff / (real.sqrt (sum_x_diff_sq * sum_y_diff_sq))).round_to 2 = 0.94 :=
by sorry

end estimated_population_correlation_coefficient_l216_216228


namespace num_good_pairs_l216_216257

def line1_slope : ℚ := 3
def line2_slope : ℚ := 1
def line3_slope : ℚ := 3
def line4_slope : ℚ := -1 / 3
def line5_slope : ℚ := -3

def is_parallel (slope1 slope2 : ℚ) : Prop := slope1 = slope2
def is_perpendicular (slope1 slope2 : ℚ) : Prop := slope1 * slope2 = -1

def good_pairs_count : ℕ :=
  let slopes := [line1_slope, line2_slope, line3_slope, line4_slope, line5_slope]
  let parallel_pairs := slopes.combinations(2).count (λ pair, is_parallel pair.head! pair.tail!.head!)
  let perpendicular_pairs := slopes.combinations(2).count (λ pair, is_perpendicular pair.head! pair.tail!.head!)
  parallel_pairs + perpendicular_pairs

theorem num_good_pairs : good_pairs_count = 3 :=
  by sorry

end num_good_pairs_l216_216257


namespace solve_for_s_l216_216589

-- Definition of the given problem conditions
def parallelogram_sides_60_angle_sqrt_area (s : ℝ) :=
  ∃ (area : ℝ), (area = 27 * Real.sqrt 3) ∧
  (3 * s * s * Real.sqrt 3 = area)

-- Proof statement to demonstrate the equivalence of the theoretical and computed value of s
theorem solve_for_s (s : ℝ) : parallelogram_sides_60_angle_sqrt_area s → s = 3 :=
by
  intro h
  sorry

end solve_for_s_l216_216589


namespace max_product_is_negative_one_l216_216636

def f (x : ℝ) : ℝ := sorry    -- Assume some function f
def g (x : ℝ) : ℝ := sorry    -- Assume some function g

theorem max_product_is_negative_one (h_f_range : ∀ y, 1 ≤ y ∧ y ≤ 6 → ∃ x, f x = y) 
    (h_g_range : ∀ y, -4 ≤ y ∧ y ≤ -1 → ∃ x, g x = y) : 
    ∃ b, b = -1 ∧ ∀ x, f x * g x ≤ b :=
sorry

end max_product_is_negative_one_l216_216636


namespace andy_profit_per_cake_l216_216986

-- Definitions based on the conditions
def cost_of_ingredients (cakes : ℕ) : ℕ := if cakes = 2 then 12 else 0
def cost_of_packaging_per_cake : ℕ := 1
def selling_price_per_cake : ℕ := 15

-- Theorem stating the profit made per cake
theorem andy_profit_per_cake : ∀ (cakes : ℕ), cakes = 2 → 
(cost_of_ingredients cakes / cakes + cost_of_packaging_per_cake) = 7 →
selling_price_per_cake - (cost_of_ingredients cakes / cakes + cost_of_packaging_per_cake) = 8 :=
by
  intros cakes h_cakes cost_hyp
  have h1 : cost_of_ingredients cakes / cakes = 12 / 2 :=
    by rw [h_cakes]; refl
  have h2 : (12 / 2 + cost_of_packaging_per_cake) = 6 + 1 :=
    by rw [h1]; refl
  have h3 : (6 + 1) = 7 :=
    by refl
  rw [← h3] at cost_hyp
  have h4 : selling_price_per_cake - 7 = 8 :=
    by refl
  exact h4

end andy_profit_per_cake_l216_216986


namespace monthly_growth_rate_price_reduction_l216_216789

theorem monthly_growth_rate
  (initial_sales : ℕ) (final_sales : ℕ) (months : ℕ)
  (initial_sales = 192) (final_sales = 300) (months = 2) :
  ∃ x : ℚ, (1 + x) ^ months = final_sales / initial_sales := by
  sorry

theorem price_reduction
  (initial_sales : ℕ) (final_sales : ℕ) (profit_target : ℚ) (price_reduction : ℚ)
  (initial_price : ℚ) (cost_per_item : ℚ) (initial_profit_per_item : ℚ)
  (increase_in_sales_per_reduction : ℕ) (price_reduction_increment : ℚ)
  (initial_sales = 300) (profit_target = 6080) (initial_price = 60)
  (cost_per_item = 40) (initial_profit_per_item = initial_price - cost_per_item)
  (increase_in_sales_per_reduction = 40) (price_reduction_increment = 2) :
  ∃ m : ℚ, (initial_profit_per_item - m) * (initial_sales + (increase_in_sales_per_reduction / price_reduction_increment) * m) = profit_target := by
  sorry

end monthly_growth_rate_price_reduction_l216_216789


namespace partnership_profit_l216_216921

theorem partnership_profit (P : ℝ) (mary_investment : ℝ) (mike_investment : ℝ) 
  (equally_divided_profit : ℝ → ℝ)
  (investment_ratio_profit : ℝ → ℝ → ℝ) :
  mary_investment = 700 →
  mike_investment = 300 →
  equally_divided_profit P = P / 6 →
  investment_ratio_profit P mary_investment = (7 / 10) * (2 * P / 3) →
  investment_ratio_profit P mike_investment = (3 / 10) * (2 * P / 3) →
  (equally_divided_profit P + investment_ratio_profit P mary_investment) - 
  (equally_divided_profit P + investment_ratio_profit P mike_investment) = 800 →
  P = 3000 :=
begin
  intros,
  sorry
end

end partnership_profit_l216_216921


namespace min_value_of_16x_minus_4x_plus_1_l216_216296

noncomputable def find_min_value : ℝ :=
  let f (x : ℝ) := 16^x - 4^x + 1
  in Inf (set.range f)

theorem min_value_of_16x_minus_4x_plus_1 : find_min_value = 3/4 :=
  sorry

end min_value_of_16x_minus_4x_plus_1_l216_216296


namespace shiela_neighbors_count_l216_216841

theorem shiela_neighbors_count (total_drawings : ℕ) (drawings_per_neighbor : ℕ) 
  (h1 : total_drawings = 54) (h2 : drawings_per_neighbor = 9) :
  total_drawings / drawings_per_neighbor = 6 := 
  by
  rw [h1, h2]
  sorry -- replace with the actual proof

end shiela_neighbors_count_l216_216841


namespace value_of_xyz_l216_216355

open Real

theorem value_of_xyz (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 37)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 11) :
  x * y * z = 26 / 3 := 
  sorry

end value_of_xyz_l216_216355


namespace smallest_prime_dividing_sum_l216_216555

theorem smallest_prime_dividing_sum : 
  let a := 4^15 in
  let b := 6^17 in
  2 ∣ (a + b) ∧ ∀ p : ℕ, prime p → p ∣ (a + b) → 2 ≤ p :=
by
  let a := 4^15
  let b := 6^17
  have h1 : 2 ∣ a := by sorry
  have h2 : 2 ∣ b := by sorry
  have h3 : 2 ∣ (a + b) := by sorry
  have h4 : ∀ p : ℕ, prime p → p ∣ (a + b) → 2 ≤ p := by sorry
  exact ⟨h3, h4⟩

end smallest_prime_dividing_sum_l216_216555


namespace chord_of_hyperbola_tangents_eqn_l216_216401

-- Define the parameters and conditions
variables {a b x0 y0 x y : ℝ}
variable (ha : a > 0)
variable (hb : b > 0)
variable (off_hyperbola : (x0^2) / (a^2) - (y0^2) / (b^2) ≠ 1)

-- Define the theorem to prove
theorem chord_of_hyperbola_tangents_eqn :
  ∃ P1 P2 : ℝ × ℝ,
  tangent_point P1 ∧ tangent_point P2 ∧
  ∀ x y : ℝ, (x0 * x) / (a^2) - (y0 * y) / (b^2) = 1 :=
sorry

end chord_of_hyperbola_tangents_eqn_l216_216401


namespace cos_angle_BAD_in_triangle_ABC_l216_216774

theorem cos_angle_BAD_in_triangle_ABC (A B C D : Type)
  (AB AC BC : ℝ) (hAB : AB = 4) (hAC : AC = 7) (hBC : BC = 9)
  (hD_on_BC : ∃ (t : ℝ), t ∈ Icc 0 1 ∧ D = (1 - t) • B + t • C)
  (hAD_bisects_BAC : is_angle_bisector A D B C):
  cos (angle A D B) = real.sqrt (5 / 14) :=
sorry

end cos_angle_BAD_in_triangle_ABC_l216_216774


namespace cars_meet_at_time_and_distance_l216_216895

def distance_between_places : Real := 12
def time_car_a : Real := 15
def time_car_b : Real := 10
def departure_delay : Real := 2

def distance_traveled_by_A (t : Real) : Real :=
  if 0 ≤ t ∧ t ≤ 15 then (12 / 15) * t else 0

def distance_traveled_by_B (t : Real) : Real :=
  if 0 ≤ t ∧ t < 2 then 0
  else if 2 ≤ t ∧ t ≤ 12 then (12 / 10) * (t - 2)
  else if 12 < t ∧ t ≤ 15 then 12
  else 0

theorem cars_meet_at_time_and_distance :
  ∃ t d, (0 ≤ t ∧ t ≤ 15) ∧ distance_traveled_by_A t = distance_traveled_by_B t ∧ d = (12 / 15) * 6 ∧ t = 6 ∧ d = 7.2 :=
sorry

end cars_meet_at_time_and_distance_l216_216895


namespace weight_of_replaced_person_l216_216495

-- Define the main parameters and conditions
variable (old_weight : ℕ) (new_weight : ℕ) (num_persons : ℕ) (avg_increase : ℕ)

-- Given conditions in the problem
def conditions := 
  num_persons = 12 ∧ 
  avg_increase = 4 ∧ 
  new_weight = 106 ∧ 
  (new_weight - old_weight) = num_persons * avg_increase

-- Main theorem stating that the old weight must be 58 kg
theorem weight_of_replaced_person : conditions → old_weight = 58 := by
  intros hcond
  -- Extract conditions for clarity
  cases' hcond with np rest
  cases' rest with ai rest
  cases' rest with nw cond
  cases' cond with eq_diff cond
  -- We know np = 12, ai = 4, nw = 106
  have h1 : num_persons = 12 := np
  have h2 : avg_increase = 4 := ai
  have h3 : new_weight = 106 := nw
  have h4 : new_weight - old_weight = num_persons * avg_increase := eq_diff
  -- Rewrite equation to solve for old_weight
  calc
    old_weight
        = new_weight - (num_persons * avg_increase) : by
          rw [←h4]
    ... = 106 - (12 * 4) : by
          rw [h1, h2, h3]
    ... = 106 - 48 : by
          norm_num
    ... = 58 : by
          norm_num
          
  sorry

end weight_of_replaced_person_l216_216495


namespace count_squares_within_region_l216_216721

theorem count_squares_within_region : 
  let π := Real.pi in
  let bound_x := 5.1 in
  let bound_y_neg := -0.1 in
  let square_count := ∑ x in finset.range (bound_x.floor + 1),
    finset.range (x.floor * π).floor) - (x+1)
  square_count = 30 :=
by
  sorry

end count_squares_within_region_l216_216721


namespace relation_between_m_and_n_l216_216034

variable {A x y z a b c d e n m : ℝ}
variable {p r : ℝ}
variable (s : finset ℝ) (hset : s = {x, y, z, a, b, c, d, e})
variable (hsorted : x < y ∧ y < z ∧ z < a ∧ a < b ∧ b < c ∧ c < d ∧ d < e)
variable (hne : n ∉ s)
variable (hme : m ∉ s)

theorem relation_between_m_and_n 
  (h_avg_n : (s.sum + n) / 9 = (s.sum / 8) * (1 + p / 100)) 
  (h_avg_m : (s.sum + m) / 9 = (s.sum / 8) * (1 + r / 100)) 
  : m = n + 9 * (s.sum / 8) * (r / 100 - p / 100) :=
sorry

end relation_between_m_and_n_l216_216034


namespace quadratic_always_two_distinct_real_roots_isosceles_triangle_perimeter_l216_216698

theorem quadratic_always_two_distinct_real_roots (m : ℝ) :
  let Δ := (m + 1)^2 + 4 in Δ > 0 :=
by sorry

theorem isosceles_triangle_perimeter (m : ℝ) (root1 root2 : ℝ) 
  (h_eq : root1^2 - (m+3)*root1 + m + 1 = 0) 
  (h_root1 : root1 = 4) 
  (h_roots : root1 = root2 ∨ root1 = real.max root1 root2) :
  let side1 := 4
  let side2 := if root2 = 4 then 4 else real.max root1 root2
  let base := if root2 = 4 then real.max root1 root2 else root2
  in side1 + side1 + base = 26 / 3 :=
by sorry

end quadratic_always_two_distinct_real_roots_isosceles_triangle_perimeter_l216_216698


namespace sequence_S_formula_sequence_a_formula_sequence_relationship_l216_216712

def sequence_a (n : ℕ) : ℝ :=
  if n = 1 then 1 else sorry -- exact formula to be determined in the proof

def sum_S (n : ℕ) : ℝ := 
  if n = 1 then 1 else sorry -- sum of the first n terms

theorem sequence_S_formula :
  ∀ n ≥ 1, sum_S n = 1 / (2 * n - 1) :=
begin
  sorry
end

theorem sequence_a_formula :
  ∀ n ≥ 1, sequence_a n = 
    if n = 1 then 1 
    else -2 / ((2 * n - 1) * (2 * n - 3)) :=
begin
  sorry
end

theorem sequence_relationship :
  ∀ n ≥ 2, sequence_a n = (2 * (sum_S n)^2) / (2 * sum_S n - 1) :=
begin
  sorry
end

end sequence_S_formula_sequence_a_formula_sequence_relationship_l216_216712


namespace circumference_of_cone_base_l216_216175

theorem circumference_of_cone_base (V : ℝ) (h : ℝ) (C : ℝ) (π := Real.pi) 
  (volume_eq : V = 24 * π) (height_eq : h = 6) 
  (circumference_eq : C = 4 * Real.sqrt 3 * π) :
  ∃ r : ℝ, (V = (1 / 3) * π * r^2 * h) ∧ (C = 2 * π * r) :=
by
  sorry

end circumference_of_cone_base_l216_216175


namespace gcd_180_308_l216_216905

theorem gcd_180_308 : Nat.gcd 180 308 = 4 :=
by
  sorry

end gcd_180_308_l216_216905


namespace product_of_values_l216_216725

theorem product_of_values (x : ℚ) (hx : abs ((18 / x) + 4) = 3) :
  x = -18 ∨ x = -18 / 7 ∧ -18 * (-18 / 7) = 324 / 7 :=
by sorry

end product_of_values_l216_216725


namespace no_valid_some_power_exists_l216_216693

-- Definitions based on conditions
def thirteen_factorial : ℕ := nat.factorial 13
def some_power : ℕ

-- Main statement
theorem no_valid_some_power_exists 
  (a : ℤ) 
  (h1 : (( ( (thirteen_factorial : ℤ) ^ some_power ) - (thirteen_factorial : ℤ) ^ 8 ) / ( ( (thirteen_factorial : ℤ) ^ 8 ) + ( (thirteen_factorial : ℤ) ^ 4 ) )) = a)
  (h2 : nat.units_digit (a / (thirteen_factorial ^ 4)) = 9) 
  : false := 
sorry

end no_valid_some_power_exists_l216_216693


namespace chessboard_dark_over_light_l216_216253

theorem chessboard_dark_over_light :
  ∀ (n : ℕ), n = 9 →
  let dark_rows := (n + 1) / 2,
      light_rows := n / 2,
      dark_squares := dark_rows * 5 + light_rows * 4,
      light_squares := dark_rows * 4 + light_rows * 5
  in dark_squares - light_squares = 1 :=
by
  intros n hn
  dsimp [dark_rows, light_rows, dark_squares, light_squares]
  have h : n = 9 := hn
  simp [h]
  sorry

end chessboard_dark_over_light_l216_216253


namespace unit_vector_norm_diff_l216_216885

noncomputable def sqrt42_sqrt3_div_2 : ℝ := (Real.sqrt 42 * Real.sqrt 3) / 2
noncomputable def sqrt17_div_sqrt2 : ℝ := (Real.sqrt 17) / Real.sqrt 2

theorem unit_vector_norm_diff {x1 y1 z1 x2 y2 z2 : ℝ}
  (h1 : x1^2 + y1^2 + z1^2 = 1)
  (h2 : 3*x1 + y1 + 2*z1 = sqrt42_sqrt3_div_2)
  (h3 : 2*x1 + 2*y1 + 3*z1 = sqrt17_div_sqrt2)
  (h4 : x2^2 + y2^2 + z2^2 = 1)
  (h5 : 3*x2 + y2 + 2*z2 = sqrt42_sqrt3_div_2)
  (h6 : 2*x2 + 2*y2 + 3*z2 = sqrt17_div_sqrt2)
  (h_distinct : (x1, y1, z1) ≠ (x2, y2, z2)) :
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2) = Real.sqrt 2 :=
by
  sorry

end unit_vector_norm_diff_l216_216885


namespace total_population_correct_l216_216524

/-- Define the populations of each city -/
def Population.Seattle : ℕ := sorry
def Population.LakeView : ℕ := 24000
def Population.Boise : ℕ := (3 * Population.Seattle) / 5

/-- Population of Lake View is 4000 more than the population of Seattle -/
axiom lake_view_population : Population.LakeView = Population.Seattle + 4000

/-- Define the total population -/
def total_population : ℕ :=
  Population.Seattle + Population.LakeView + Population.Boise

/-- Prove that total population of the three cities is 56000 -/
theorem total_population_correct :
  total_population = 56000 :=
sorry

end total_population_correct_l216_216524


namespace largest_subset_no_triplet_sum_zero_l216_216811

theorem largest_subset_no_triplet_sum_zero (n : ℕ) (hn : 0 < n) :
  (∀ s ⊆ (finset.range (2 * n + 1)).map (λ x, -↑n + x), 
     (∀ a b c ∈ s, a + b + c ≠ 0) → s.card ≤ if n % 2 = 0 then n else n + 1) :=
begin
  sorry
end

end largest_subset_no_triplet_sum_zero_l216_216811


namespace total_letters_l216_216387

theorem total_letters (brother_letters : ℕ) (greta_more_than_brother : ℕ) (mother_multiple : ℕ)
  (h_brother : brother_letters = 40)
  (h_greta : ∀ (brother_letters greta_letters : ℕ), greta_letters = brother_letters + greta_more_than_brother)
  (h_mother : ∀ (total_letters mother_letters : ℕ), mother_letters = mother_multiple * total_letters) :
  brother_letters + (brother_letters + greta_more_than_brother) + (mother_multiple * (brother_letters + (brother_letters + greta_more_than_brother))) = 270 :=
by
  sorry

end total_letters_l216_216387


namespace P_neg2_eq_19_l216_216022

noncomputable def P : ℝ → ℝ := sorry  -- Definition of the polynomial P(x)

axiom degree_P : ∃ (d : ℕ), d = 4 ∧ ∀ (x : ℝ) (hx : P x ≠ 0), nat_degree (polynomial.of_real (P x)) = d
axiom P_0 : P 0 = 1
axiom P_1 : P 1 = 1
axiom P_2 : P 2 = 4
axiom P_3 : P 3 = 9
axiom P_4 : P 4 = 16

theorem P_neg2_eq_19 : P (-2) = 19 :=
by
  sorry

end P_neg2_eq_19_l216_216022


namespace minimum_value_f_maximum_value_f_l216_216668

noncomputable def f (x : ℝ) : ℝ := 1 / (4 ^ x) - 1 / (2 ^ x) + 1

theorem minimum_value_f : ∀ x : ℝ, x ∈ Icc (-3 : ℝ) 2 → f x ≥ 3 / 4 :=
by
  intro x hx
  -- Proof steps are skipped
  sorry

theorem maximum_value_f : ∀ x : ℝ, x ∈ Icc (-3 : ℝ) 2 → f x ≤ 57 :=
by
  intro x hx
  -- Proof steps are skipped
  sorry

end minimum_value_f_maximum_value_f_l216_216668


namespace carpet_total_shaded_area_l216_216173

theorem carpet_total_shaded_area (R S : ℝ) 
    (hR : R = 9) 
    (hS : S = 9 / 4) 
    (width_central : ℝ := 6) : 
    12 * 18 + 12 * (S * S) = 114.75 :=
by
  have hR_ratio : 18 / R = 2 := by sorry
  have hS_ratio : R / S = 4 := by sorry
  have width_central_calc : width_central = (6 : ℝ) := by simp [width_central]
  have area_central : 9 * 6 = 54 := by norm_num
  have area_square : (9 / 4) ^ 2 = 81 / 16 := by { field_simp, norm_num }
  have total_area_square : 12 * (81 / 16) = 60.75 := by { norm_num1, rw mul_comm, norm_num1 }
  have total_shaded_area : 54 + 60.75 = 114.75 := by norms_imp
  exact total_shaded_area

end carpet_total_shaded_area_l216_216173


namespace new_student_weight_l216_216144

theorem new_student_weight 
  (w_avg : ℝ)
  (w_new : ℝ)
  (condition : (5 * w_avg - 72 = 5 * (w_avg - 12) + w_new)) 
  : w_new = 12 := 
  by 
  sorry

end new_student_weight_l216_216144


namespace find_hidden_data_points_l216_216414

-- Given conditions and data
def student_A_score := 81
def student_B_score := 76
def student_D_score := 80
def student_E_score := 83
def number_of_students := 5
def average_score := 80

-- The total score from the average and number of students
def total_score := average_score * number_of_students

theorem find_hidden_data_points (student_C_score mode_score : ℕ) :
  (student_A_score + student_B_score + student_C_score + student_D_score + student_E_score = total_score) ∧
  (mode_score = 80) :=
by
  sorry

end find_hidden_data_points_l216_216414


namespace evaluate_exponent_l216_216277

theorem evaluate_exponent : (3^2)^4 = 6561 := sorry

end evaluate_exponent_l216_216277


namespace ratio_of_perimeters_l216_216125

-- Define the problem as a theorem in Lean
theorem ratio_of_perimeters (d : ℝ) (d_pos : d > 0) :
  let P1 := 4 * (d / real.sqrt 2)
  let P2 := 4 * (real.sqrt 2 * d)
  P2 / P1 = 2 * real.sqrt 2 := 
by
  sorry

end ratio_of_perimeters_l216_216125


namespace smallest_b_exists_a_parfume_set_l216_216182

def P (n : ℕ) : ℕ := n^2 + n + 1

def is_parfume (s : Set ℕ) : Prop :=
  s.nonempty ∧ 1 < s.card ∧ ∀ x ∈ s, ∃ y ∈ s, x ≠ y ∧ ∃ p, Nat.Prime p ∧ p ∣ x ∧ p ∣ y

theorem smallest_b_exists_a_parfume_set :
  ∃ a : ℕ, is_parfume {P (a + 1), P (a + 2), P (a + 3), P (a + 4), P (a + 5), P (a + 6)} :=
sorry

end smallest_b_exists_a_parfume_set_l216_216182


namespace rectangle_area_l216_216954

theorem rectangle_area (b : ℝ) (h_length : real.sqrt (10 * b^2) = 26) : (3 * b^2) = 202.8 :=
by
  -- Using the condition 26^2 = 10 * b^2 hence, b^2 = 67.6.
  -- The area A = 3b * b = 3 * b^2 = 3 * 67.6 = 202.8
  sorry

end rectangle_area_l216_216954


namespace distance_between_AB_is_4_l216_216696

-- Definition of parametric curve
def curve_C (φ : ℝ) : ℝ × ℝ := (2 * Real.cos φ, 2 + 2 * Real.sin φ)

-- Coordinates A and B with given polar coordinates
def point_A : ℝ × ℝ := curve_C (Real.pi / 3)
def point_B : ℝ × ℝ := curve_C (5 * Real.pi / 6)

-- Function for distance calculation between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Statement of the problem
theorem distance_between_AB_is_4 : distance point_A point_B = 4 := by
  sorry

end distance_between_AB_is_4_l216_216696


namespace complex_powers_sum_zero_l216_216649

theorem complex_powers_sum_zero (i : ℂ) (h : i^2 = -1) : i^2023 + i^2024 + i^2025 + i^2026 = 0 :=
by
  sorry

end complex_powers_sum_zero_l216_216649


namespace sequence_equality_l216_216960

theorem sequence_equality (a : ℕ → ℤ) (h : ∀ n, a (n + 2) ^ 2 + a (n + 1) * a n ≤ a (n + 2) * (a (n + 1) + a n)) :
  ∃ N : ℕ, ∀ n ≥ N, a (n + 2) = a n :=
by sorry

end sequence_equality_l216_216960


namespace cost_of_saddle_l216_216167

theorem cost_of_saddle (S : ℝ) (H : 4 * S + S = 5000) : S = 1000 :=
by sorry

end cost_of_saddle_l216_216167


namespace seven_digit_number_difference_l216_216552

-- Statement of the problem in Lean
theorem seven_digit_number_difference :
  let digits := [3, 9, 6, 0, 5, 1, 7]
  largest_number := 9765310
  smallest_number := 1035679
  (largest_number - smallest_number) = 8729631 := 
by
  sorry

end seven_digit_number_difference_l216_216552


namespace probability_of_region_l216_216952

noncomputable def square_side : ℝ := 2
noncomputable def total_area : ℝ := square_side ^ 2
noncomputable def circle_radius : ℝ := 1
noncomputable def circle_area : ℝ := π * (circle_radius ^ 2)
noncomputable def combined_circle_area : ℝ := 4 * (circle_area / 4)
noncomputable def valid_area : ℝ := total_area - combined_circle_area
noncomputable def probability : ℝ := valid_area / total_area

theorem probability_of_region (side_len : ℝ) (radius : ℝ) (π: ℝ) (area_total: ℝ):
  side_len = 2 →
  radius = 1 →
  π = real.pi →
  area_total = side_len^2 →
  probability = 1- π / 4 :=
by
  intros h_side_len h_radius h_π h_area_total
  simp [total_area, circle_radius, circle_area, combined_circle_area, valid_area, probability, h_side_len, h_radius, h_π, h_area_total]
  sorry

end probability_of_region_l216_216952


namespace beach_ball_surface_area_excluding_valve_l216_216935

theorem beach_ball_surface_area_excluding_valve:
  let r_b := 33 / (2 * Real.pi) in
  let S := 4 * Real.pi * r_b^2 in
  let r_v := 1 / 2 in
  let A_v := Real.pi * r_v^2 in
  S - A_v = (4356 - Real.pi^2) / (4 * Real.pi) :=
by
  let r_b := 33 / (2 * Real.pi)
  let S := 4 * Real.pi * r_b^2
  let r_v := 1 / 2
  let A_v := Real.pi * r_v^2
  sorry

end beach_ball_surface_area_excluding_valve_l216_216935


namespace fraction_of_historical_fiction_new_releases_l216_216989

theorem fraction_of_historical_fiction_new_releases
  (total_books : ℕ)
  (historical_fiction_percentage : ℝ := 0.4)
  (historical_fiction_new_releases_percentage : ℝ := 0.4)
  (other_genres_new_releases_percentage : ℝ := 0.7)
  (total_historical_fiction_books := total_books * historical_fiction_percentage)
  (total_other_books := total_books * (1 - historical_fiction_percentage))
  (historical_fiction_new_releases := total_historical_fiction_books * historical_fiction_new_releases_percentage)
  (other_genres_new_releases := total_other_books * other_genres_new_releases_percentage)
  (total_new_releases := historical_fiction_new_releases + other_genres_new_releases) :
  historical_fiction_new_releases / total_new_releases = 8 / 29 := 
by 
  sorry

end fraction_of_historical_fiction_new_releases_l216_216989


namespace simplify_expression_l216_216487

noncomputable def simplify_expr : ℝ :=
  (3 + 2 * Real.sqrt 2) ^ Real.sqrt 3

theorem simplify_expression :
  (Real.sqrt 2 - 1) ^ (2 - Real.sqrt 3) / (Real.sqrt 2 + 1) ^ (2 + Real.sqrt 3) = simplify_expr :=
by
  sorry

end simplify_expression_l216_216487


namespace closed_curve_min_perimeter_l216_216057

theorem closed_curve_min_perimeter (a b c d : ℝ) (curve : Set (ℝ × ℝ))
  (rectangle : Set (ℝ × ℝ)) (intersect_AB : ∃ p ∈ curve, p ∈ rectangle) 
  (intersect_BC : ∃ q ∈ curve, q ∈ rectangle) 
  (intersect_CD : ∃ r ∈ curve, r ∈ rectangle) 
  (intersect_DA : ∃ s ∈ curve, s ∈ rectangle) :
  let diagonal := Real.sqrt ((a - c)^2 + (b - d)^2) in
  curve_perimeter curve ≥ 2 * diagonal := sorry

def curve_perimeter (curve : Set (ℝ × ℝ)) : ℝ := sorry

end closed_curve_min_perimeter_l216_216057


namespace find_d_l216_216492

-- Define the conditions
variables (x₀ y₀ c : ℝ)

-- Define the system of equations
def system_of_equations : Prop :=
  x₀ * y₀ = 6 ∧ x₀^2 * y₀ + x₀ * y₀^2 + x₀ + y₀ + c = 2

-- Define the target proof problem
theorem find_d (h : system_of_equations x₀ y₀ c) : x₀^2 + y₀^2 = 69 :=
sorry

end find_d_l216_216492


namespace polynomial_value_at_neg2_l216_216021

noncomputable def P : ℝ → ℝ
-- Define the polynomial P assuming the conditions

theorem polynomial_value_at_neg2 :
  (∀ P : ℝ → ℝ,
  ∃ (a b c d e : ℝ),      -- General form for polynomial degree 4
  P(x) = a*x^4 + b*x^3 + c*x^2 + d*x + e ∧
    P(0) = 1 ∧ P(1) = 1 ∧ P(2) = 4 ∧ P(3) = 9 ∧ P(4) = 16) →
  P(-2) = 19 :=
by 
  sorry

end polynomial_value_at_neg2_l216_216021


namespace eval_expression_at_x_is_correct_l216_216558

theorem eval_expression_at_x_is_correct : 
  let x := 3 in (x^6 - 6 * x^2) = 675 :=
by
  sorry

end eval_expression_at_x_is_correct_l216_216558


namespace average_speed_round_trip_l216_216180

def upstream_distance_1 : ℝ := 4 * 2
def upstream_distance_2 : ℝ := 7 * 3
def total_upstream_distance : ℝ := upstream_distance_1 + upstream_distance_2

def downstream_distance_1 : ℝ := 6 * 1
def downstream_distance_2 : ℝ := 9 * 4
def total_downstream_distance : ℝ := downstream_distance_1 + downstream_distance_2

def total_distance : ℝ := total_upstream_distance + total_upstream_distance

def upstream_time : ℝ := 2 + 3
def downstream_time : ℝ := 1 + 4
def total_time : ℝ := upstream_time + downstream_time

def average_speed : ℝ := total_distance / total_time

theorem average_speed_round_trip : average_speed = 5.8 := by
  sorry

end average_speed_round_trip_l216_216180


namespace largest_lucky_number_l216_216963

theorem largest_lucky_number (n : ℕ) (h₀ : n = 160) (h₁ : ∀ k, 160 > k → k > 0) (h₂ : ∀ k, k ≡ 7 [MOD 16] → k ≤ 160) : 
  ∃ k, k = 151 := 
sorry

end largest_lucky_number_l216_216963


namespace hyperbola_eccentricity_l216_216654

theorem hyperbola_eccentricity :
  let a := 2
  let b := 2 * Real.sqrt 2
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  (e = Real.sqrt 3) :=
by {
  sorry
}

end hyperbola_eccentricity_l216_216654


namespace sequence_sum_l216_216339

noncomputable def sequence (n : ℕ) : ℕ := sorry

theorem sequence_sum :
  (∀ n, sequence n * sequence (n+1) * sequence (n+2) * sequence (n+3) = 24)
  ∧ sequence 1 = 1 ∧ sequence 2 = 2 ∧ sequence 3 = 3
  → (Finset.range 2013).sum (λ n, sequence (n + 1)) = 5031 := sorry


end sequence_sum_l216_216339


namespace solution_set_of_inequality_l216_216093

theorem solution_set_of_inequality :
  {x : ℝ | 2 * x^2 - x ≤ 1} = set.Icc (-1/2 : ℝ) (1 : ℝ) := 
by sorry

end solution_set_of_inequality_l216_216093


namespace correct_option_D_l216_216745

noncomputable def total_students := 40
noncomputable def male_students := 25
noncomputable def female_students := 15
noncomputable def class_president := 1
noncomputable def prob_class_president := class_president / total_students
noncomputable def prob_class_president_from_females := 0

theorem correct_option_D
  (h1 : total_students = 40)
  (h2 : male_students = 25)
  (h3 : female_students = 15)
  (h4 : class_president = 1) :
  prob_class_president = 1 / 40 ∧ prob_class_president_from_females = 0 := 
by
  sorry

end correct_option_D_l216_216745


namespace smallest_number_of_students_l216_216241

theorem smallest_number_of_students 
(r9_10 : ℕ → ℕ → Prop) 
(r9_11 : ℕ → ℕ → Prop) :
    (r9_10 9th 10th ↔ 7 * 10th = 4 * 9th)
    → (r9_11 9th 11th ↔ 5 * 11th = 3 * 9th)
    → (∃ (x : ℕ) (y : ℕ) (z : ℕ), 7 * y = 4 * x ∧ 5 * z = 3 * x ∧ 35 + y + z = 76) 
:=
by {
    intro h_r9_10,
    intro h_r9_11,
    sorry
}

end smallest_number_of_students_l216_216241


namespace ending_number_of_multiples_l216_216882

theorem ending_number_of_multiples (n : ℤ) (h : 991 = (n - 100) / 10 + 1) : n = 10000 :=
by
  sorry

end ending_number_of_multiples_l216_216882


namespace hyperbola_eccentricity_l216_216705

noncomputable def eccentricity_of_hyperbola (a b x y : ℝ) (P : (x, y) = (2, -1) ∧ (y^2 / a^2 - x^2 / 4 = 1)) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  in c / a

theorem hyperbola_eccentricity
  (a b x y : ℝ)
  (h₁ : (x, y) = (2, -1))
  (h₂ : y^2 / a^2 - x^2 / 4 = 1)
  (ha : a = Real.sqrt 2 / 2)
  (hb : b = 2) :
  eccentricity_of_hyperbola a b x y (by simp [*]) = 3 :=
sorry

end hyperbola_eccentricity_l216_216705


namespace minimum_value_of_f_l216_216314

noncomputable def f (x : ℝ) : ℝ := 16^x - 4^x + 1

theorem minimum_value_of_f : ∃ (x : ℝ), ∀ (y : ℝ), f(y) ≥ f(x) ∧ f(x) = 3 / 4 :=
sorry

end minimum_value_of_f_l216_216314


namespace proof_problem_l216_216345

variable {R : Type} [OrderedRing R]

-- Definitions and conditions
variable (g : R → R) (f : R → R) (k a m : R)
variable (h_odd : ∀ x : R, g (-x) = -g x)
variable (h_f_def : ∀ x : R, f x = g x + k)
variable (h_f_neg_a : f (-a) = m)

-- Theorem statement
theorem proof_problem : f a = 2 * k - m :=
by
  -- Here is where the proof would go.
  sorry

end proof_problem_l216_216345


namespace tan_identity_solution_l216_216319

theorem tan_identity_solution (x : ℝ) (hx : 0 < x ∧ x < 180) (hx_sol : x = 110) :
  tan (150 - x) = (sin 150 - sin x) / (cos 150 - cos x) :=
by sorry

end tan_identity_solution_l216_216319


namespace f_characterization_l216_216461

def f : ℤ → ℤ := sorry  -- The actual function definition will be derived.

axiom f_condition_0 : f 0 = 1
axiom f_condition_1 : f 1 = 0
axiom f_functional_eq (m n : ℕ) : f (m + n) + f (m - n) = 2 * f m * f n

theorem f_characterization (n : ℤ) : 
  f n = 
  if ∃ k : ℤ, n = 4 * k then 1 
  else if ∃ k : ℤ, n = 4 * k + 2 then -1
  else if n % 2 = 1 then 0 
  else 0 :=
begin
  sorry
end

end f_characterization_l216_216461


namespace Gordons_heavier_bag_weight_l216_216893

theorem Gordons_heavier_bag_weight :
  ∀ (G : ℝ), (5 * 2 = 3 + G) → G = 7 :=
by
  intro G h
  sorry

end Gordons_heavier_bag_weight_l216_216893


namespace monotonic_decreasing_interval_of_f_l216_216083

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem monotonic_decreasing_interval_of_f :
  { x : ℝ | x > Real.exp 1 } = {y : ℝ | ∀ ε > 0, (x : ℝ) → (0 < x → (f (x + ε) < f x) ∧ (f x < f (x + ε)))}
:=
sorry

end monotonic_decreasing_interval_of_f_l216_216083


namespace value_of_K_l216_216911

theorem value_of_K (K: ℕ) : 4^5 * 2^3 = 2^K → K = 13 := by
  sorry

end value_of_K_l216_216911


namespace triangle_inequality_l216_216341

noncomputable def area_triangle (a b c : ℝ) : ℝ := sorry -- Definition of area, but implementation is not required.

theorem triangle_inequality (a b c : ℝ) (S_triangle : ℝ):
  1 - (8 * ((a - b) ^ 2 + (b - c) ^ 2 + (c - a) ^ 2) / (a + b + c) ^ 2)
  ≤ 432 * S_triangle ^ 2 / (a + b + c) ^ 4
  ∧ 432 * S_triangle ^ 2 / (a + b + c) ^ 4
  ≤ 1 - (2 * ((a - b) ^ 2 + (b - c) ^ 2 + (c - a) ^ 2) / (a + b + c) ^ 2) :=
sorry -- Proof is omitted

end triangle_inequality_l216_216341


namespace find_possible_values_l216_216803

theorem find_possible_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 90) :
  ∃ y, (y = (x - 3)^2 * (x + 4) / (3 * x - 4)) ∧ (y = 36 / 11 ∨ y = 468 / 23) :=
by
  sorry

end find_possible_values_l216_216803


namespace find_x_power_12_l216_216731

theorem find_x_power_12 (x : ℂ) (h : x + (1/x) = real.sqrt 2) : x^12 = -1 := by
  sorry

end find_x_power_12_l216_216731


namespace max_profit_l216_216762

/-- Define the cost and price of device A and device B -/
def cost_A : ℝ := 3
def price_A : ℝ := 3.3
def cost_B : ℝ := 2.4
def price_B : ℝ := 2.8

/-- Define the total number of devices -/
def total_devices : ℝ := 50

/-- Define the profits per device -/
def profit_per_A : ℝ := price_A - cost_A -- 0.3
def profit_per_B : ℝ := price_B - cost_B -- 0.4

/-- Define the function for total profit -/
def total_profit (x : ℝ) : ℝ :=
  profit_per_A * x + profit_per_B * (total_devices - x)

/-- Define the constraint -/
def constraint (x : ℝ) : Prop := 4 * x ≥ total_devices - x -- x ≥ 10

/-- The statement of the problem that needs to be proven -/
theorem max_profit :
  (total_profit x = -0.1 * x + 20) ∧ 
  ( ∀ x, constraint x → x ≥ 10 → x = 10 ∧ total_profit x = 19) :=
by
  sorry

end max_profit_l216_216762


namespace boston_snow_l216_216622

noncomputable def initial_snow : ℝ := 0.5
noncomputable def second_day_snow_inch : ℝ := 8 / 12
noncomputable def next_two_days_melt_inch : ℝ := 2 / 12
noncomputable def fifth_day_snow_factor : ℝ := 2

theorem boston_snow : 
  let second_day_snow := initial_snow + second_day_snow_inch,
      snow_after_melt := second_day_snow - next_two_days_melt_inch,
      fifth_day_snow := fifth_day_snow_factor * initial_snow
  in snow_after_melt + fifth_day_snow = 2 := 
by
  sorry

end boston_snow_l216_216622


namespace avg_weight_b_and_c_l216_216067

variable (a b c : ℝ)

-- Given conditions
def condition1 : Prop := (a + b + c) / 3 = 60
def condition2 : Prop := (a + b) / 2 = 70
def condition3 : Prop := b = 60

-- Prove statement
theorem avg_weight_b_and_c (hc1 : condition1 a b c) (hc2 : condition2 a b c) (hc3 : condition3 b) : 
  (b + c) / 2 = 50 := 
by 
  sorry

end avg_weight_b_and_c_l216_216067


namespace estimated_population_correct_correlation_coefficient_correct_l216_216201

noncomputable def estimated_population
  (num_plots : ℕ) (sample_size : ℕ) 
  (sum_y : ℕ) : ℕ :=
(sum_y / sample_size) * num_plots

noncomputable def correlation_coefficient
  (sum_x_sq : ℝ) (sum_y_sq : ℝ) 
  (sum_xy : ℝ) : ℝ :=
sum_xy / real.sqrt(sum_x_sq * sum_y_sq)

theorem estimated_population_correct
  (num_plots : ℕ := 200) (sample_size : ℕ := 20) 
  (sum_y : ℕ := 1200) : estimated_population num_plots sample_size sum_y = 12000 := by
sorry

theorem correlation_coefficient_correct
  (sum_x_sq : ℝ := 80) (sum_y_sq : ℝ := 9000) 
  (sum_xy : ℝ := 800) : real.abs (correlation_coefficient sum_x_sq sum_y_sq sum_xy - 0.94) < 0.01 := by
sorry

end estimated_population_correct_correlation_coefficient_correct_l216_216201


namespace arithmetic_sequence_sum_l216_216820

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (hS3 : S 3 = 12) (hS6 : S 6 = 42) 
  (h_arith_seq : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) :
  a 10 + a 11 + a 12 = 66 :=
sorry

end arithmetic_sequence_sum_l216_216820


namespace GCF_75_135_l216_216551

theorem GCF_75_135 : Nat.gcd 75 135 = 15 :=
by
sorry

end GCF_75_135_l216_216551


namespace andy_l216_216984

theorem andy's_profit_per_cake :
  (∀ (cakes : ℕ), cakes = 2 → ∀ (ingredient_cost : ℕ), ingredient_cost = 12 →
                  ∀ (packaging_cost_per_cake : ℕ), packaging_cost_per_cake = 1 →
                  ∀ (selling_price_per_cake : ℕ), selling_price_per_cake = 15 →
                  ∀ (profit_per_cake : ℕ), profit_per_cake = selling_price_per_cake - (ingredient_cost / cakes + packaging_cost_per_cake) →
                    profit_per_cake = 8) :=
by
  sorry

end andy_l216_216984


namespace estimated_population_is_correct_correlation_coefficient_is_correct_stratified_sampling_is_better_l216_216195

-- The sample data and conditions provided.
variables (x y : Fin 20 → ℝ)
variable (x̄ : ℝ := (∑ i, x i) / 20)
variable (ȳ : ℝ := (∑ i, y i) / 20)
variables (Sxx Syy Sxy : ℝ)
variables (n : ℕ := 20)
variables (total_plots : ℕ := 200)
variable (sum_x : ℝ := ∑ i, x i)
variable (sum_y : ℝ := ∑ i, y i)
variable (total_x : ℝ := 60)
variable (total_y : ℝ := 1200)

-- Given conditions
axiom condition1 : sum_x = 60
axiom condition2 : sum_y = 1200
axiom condition3 : Sxx = ∑ i, (x i - x̄) ^ 2
axiom condition4 : Syy = ∑ i, (y i - ȳ) ^ 2
axiom condition5 : Sxy = ∑ i, (x i - x̄) * (y i - ȳ)

-- Proven result 1: Estimated population of wild animals
def estimated_population (total_plots : ℕ) (avg_y : ℝ) : ℝ :=
  avg_y * total_plots

theorem estimated_population_is_correct : estimated_population total_plots (total_y / n) = 12000 :=
by
  -- Necessary calculations skipped
  sorry

-- Proven result 2: Correlation coefficient
noncomputable def correlation_coefficient (Sxy Sxx Syy : ℝ) : ℝ :=
  Sxy / (Real.sqrt (Sxx * Syy))

theorem correlation_coefficient_is_correct : correlation_coefficient Sxy Sxx Syy = 0.94 :=
by
  -- Assume given data and calculations are correct within required precision
  sorry

-- Proven result 3: Stratified sampling is more reasonable
theorem stratified_sampling_is_better : ∀ (method : String), method = "stratified sampling" → True :=
by
  -- Explanation skipped
  intros
  sorry

end estimated_population_is_correct_correlation_coefficient_is_correct_stratified_sampling_is_better_l216_216195


namespace P_Q_work_together_time_l216_216134

noncomputable def P_rate := 1 / 4 -- P's rate of work (jobs per hour)
noncomputable def Q_rate := 1 / 20 -- Q's rate of work (jobs per hour)
noncomputable def combined_rate := P_rate + Q_rate -- Combined rate when P and Q work together (jobs per hour)
noncomputable def additional_time := 24 / 60 -- Additional time P works alone after they stop working together (hours)
noncomputable def P_additional_work := P_rate * additional_time -- Work completed by P alone in the additional time (fraction of the job)
noncomputable def total_work := 1 -- Total work to be completed (1 job)

theorem P_Q_work_together_time :
  ∑ t, combined_rate * t + P_additional_work = total_work → t = 3 := sorry

end P_Q_work_together_time_l216_216134


namespace percentage_silver_cars_after_shipment_l216_216567

-- Definitions for conditions
def initialCars : ℕ := 40
def initialSilverPerc : ℝ := 0.15
def newShipmentCars : ℕ := 80
def newShipmentNonSilverPerc : ℝ := 0.30

-- Proof statement that needs to be proven
theorem percentage_silver_cars_after_shipment :
  let initialSilverCars := initialSilverPerc * initialCars
  let newShipmentSilverPerc := 1 - newShipmentNonSilverPerc
  let newShipmentSilverCars := newShipmentSilverPerc * newShipmentCars
  let totalSilverCars := initialSilverCars + newShipmentSilverCars
  let totalCars := initialCars + newShipmentCars
  (totalSilverCars / totalCars) * 100 = 51.67 :=
by
  sorry

end percentage_silver_cars_after_shipment_l216_216567


namespace inverse_B2_l216_216729

open Matrix

-- Define the instance for field (assume a 2x2 matrix over rational numbers for simplicity)
variable {α : Type*} [Field α] [DecidableEq α]

-- Define the given condition
def B_inv : Matrix (Fin 2) (Fin 2) α := !![2, 3; 0, -1]

-- Define the expected result for B^2 inverse
def B2_inv : Matrix (Fin 2) (Fin 2) α := !![4, 3; 0, 1]

-- Prove that the inverse of  (B^2) equals to the expected result
theorem inverse_B2 (B : Matrix (Fin 2) (Fin 2) α) (h : B⁻¹ = B_inv) : 
  (B ^ 2)⁻¹ = B2_inv :=
by sorry

end inverse_B2_l216_216729


namespace log_identity_complicated_expression_l216_216625

-- Definition of log
axiom log : ℝ → ℝ
noncomputable def log_base_10 (x : ℝ) := log x

-- First proof problem
theorem log_identity : log_base_10 24 - log_base_10 3 - log_base_10 4 + log_base_10 5 = 1 := by 
  sorry

-- Second proof problem
theorem complicated_expression : 
  (⟦⟧.(3.sqrtC).mul(2.sqrt))^6 + ⟦⟧.((3.sqrt).sqrtP(3)^(4/3)) - (2.sqrt).sqrtA(2^(3/4)- 1 = 72 := by
  sorry

end log_identity_complicated_expression_l216_216625


namespace octoberagon_tangent_circles_have_equal_areas_l216_216956

noncomputable def ratio_of_areas_of_tangent_circles 
  (side_length : ℝ) (r1 r2 : ℝ) (is_regular_octagon : side_length = 2)
  (circ_tangent_to_AB : ∀ A B (tangent : ℝ = r1), true)
  (circ_tangent_to_EF : ∀ E F (tangent : ℝ = r2), true)
  (both_tangent_to_CG_line : ∀ C G (line_tangent : ℝ -> Prop), true) : ℝ :=
  let area_1 := π * r1^2
  let area_2 := π * r2^2
  area_1 / area_2

theorem octoberagon_tangent_circles_have_equal_areas :
  ratio_of_areas_of_tangent_circles 2 r1 r2 (by simp) (by trivial) (by trivial) (by trivial) = 1 := 
sorry

end octoberagon_tangent_circles_have_equal_areas_l216_216956


namespace estimated_population_correct_correlation_coefficient_correct_l216_216210

variables (x y : ℕ → ℝ)
variables (n : ℕ) (plots : ℕ)

noncomputable def average_y := (∑ i in finset.range 20, y i) / 20

noncomputable def estimated_population := average_y y * plots

theorem estimated_population_correct (h_sum_y : ∑ i in finset.range 20, y i = 1200)
                                      (h_plots : plots = 200) :
  estimated_population y plots = 12000 :=
by sorry

noncomputable def correlation_coefficient (x y : ℕ → ℝ) :=
  let sum_xy := ∑ i in finset.range 20, (x i - ((∑ j in finset.range 20, x j) / 20)) * 
                               (y i - ((∑ j in finset.range 20, y j) / 20)) in
  let sum_xx := ∑ i in finset.range 20, (x i - ((∑ j in finset.range 20, x j) / 20)) ^ 2 in
  let sum_yy := ∑ i in finset.range 20, (y i - ((∑ j in finset.range 20, y j) / 20)) ^ 2 in
  sum_xy / (real.sqrt (sum_xx * sum_yy))

theorem correlation_coefficient_correct (h_sum_xx : ∑ i in finset.range 20, (x i - ((∑ j in finset.range 20, x j) / 20)) ^ 2 = 80)
                                        (h_sum_yy : ∑ i in finset.range 20, (y i - ((∑ j in finset.range 20, y j) / 20)) ^ 2 = 9000)
                                        (h_sum_xy : ∑ i in finset.range 20, (x i - ((∑ j in finset.range 20, x j) / 20)) * (y i - ((∑ j in finset.range 20, y j) / 20)) = 800) :
  correlation_coefficient x y ≈ 0.94 :=
by sorry

end estimated_population_correct_correlation_coefficient_correct_l216_216210


namespace intercepts_of_line_l216_216078

-- Define the given line equation
def line_eq (x y : ℝ) : Prop := x / 4 - y / 3 = 1

-- Define the intercepts
def intercepts (x_intercept y_intercept : ℝ) : Prop :=
  (line_eq x_intercept 0) ∧ (line_eq 0 y_intercept)

-- The problem statement: proving the values of intercepts
theorem intercepts_of_line :
  intercepts 4 (-3) :=
by
  sorry

end intercepts_of_line_l216_216078


namespace solve_fractional_equation_l216_216094

theorem solve_fractional_equation (x : ℝ) (h : x ≠ 3) : (x - 2) / (x - 3) = 2 / (x - 3) → x = 4 :=
by
  intro h_eq
  have h_cancel := congr_arg (λ z, z * (x - 3)) h_eq  -- multiply both sides by (x - 3)
  simp [ne_of_gt, ne_of_lt] at h_cancel  -- simplify and solve step by step
  linarith [h_cancel]  -- final calculation
  done

end solve_fractional_equation_l216_216094


namespace kevin_hops_total_distance_l216_216447

theorem kevin_hops_total_distance:
  (let a := 2 in
  let r := 0.75 in
  let n := 6 in
  let S_6 := (2:ℚ) * (1 - r^n) in
  S_6 = 3367 / 2048) :=
begin
  norm_num,
  rw ←div_eq_iff_eq_mul_left (by norm_num : (2048:ℚ) ≠ 0),
  exact eq.symm (show (2:ℚ) * (1 - (3/4)^6) = 3367/2048, by norm_num),
end

end kevin_hops_total_distance_l216_216447


namespace surface_area_difference_l216_216949

theorem surface_area_difference
  (larger_cube_volume : ℝ)
  (num_smaller_cubes : ℝ)
  (smaller_cube_volume : ℝ)
  (h1 : larger_cube_volume = 125)
  (h2 : num_smaller_cubes = 125)
  (h3 : smaller_cube_volume = 1) :
  (6 * (smaller_cube_volume)^(2/3) * num_smaller_cubes) - (6 * (larger_cube_volume)^(2/3)) = 600 :=
by {
  sorry
}

end surface_area_difference_l216_216949


namespace population_after_4_years_l216_216194

theorem population_after_4_years 
  (initial_population : ℕ) 
  (new_people : ℕ) 
  (people_moved_out : ℕ) 
  (years : ℕ) 
  (final_population : ℕ) :
  initial_population = 780 →
  new_people = 100 →
  people_moved_out = 400 →
  years = 4 →
  final_population = initial_population + new_people - people_moved_out →
  final_population / 2 / 2 / 2 / 2 = 30 :=
by
  sorry

end population_after_4_years_l216_216194


namespace minimum_value_of_expression_l216_216309

theorem minimum_value_of_expression : ∃ x : ℝ, (16 ^ x - 4 ^ x + 1 = 3 / 4) :=
sorry

end minimum_value_of_expression_l216_216309


namespace unique_coprime_solution_l216_216008

theorem unique_coprime_solution 
  (p : ℕ) (a b m r : ℕ) 
  (hp : Nat.Prime p) 
  (hp_odd : p % 2 = 1)
  (hp_nmid_ab : ¬ (p ∣ a * b))
  (hab_gt_m2 : a * b > m^2) :
  ∃! (x y : ℕ), Nat.Coprime x y ∧ (a * x^2 + b * y^2 = m * p ^ r) := 
sorry

end unique_coprime_solution_l216_216008


namespace minimum_positive_period_l216_216869

theorem minimum_positive_period :
  ∃ T > 0, ∀ x, sin (x + π / 4) * sin (x - π / 4) = sin ((x + T) + π / 4) * sin ((x + T) - π / 4) ↔ T = π :=
by
  sorry

end minimum_positive_period_l216_216869


namespace gcd_seq_consecutive_l216_216123

-- Define the sequence b_n
def seq (n : ℕ) : ℕ := n.factorial + 2 * n

-- Main theorem statement
theorem gcd_seq_consecutive (n : ℕ) : n ≥ 0 → Nat.gcd (seq n) (seq (n + 1)) = 2 :=
by
  intro h
  sorry

end gcd_seq_consecutive_l216_216123


namespace sum_of_coordinates_l216_216011

-- Definitions of the points and conditions
def is_midpoint (X Y Z : Vect2 ℝ) : Prop :=
  ∃ k > 0, k = 1/2 ∧ (Z = X + k • (Y - X))

noncomputable def X := Vect2.mk (-7) (-21)
noncomputable def Y := Vect2.mk 1 7
noncomputable def Z := Vect2.mk (-3) (-7)
noncomputable def W := Vect2.mk 1 7

def satisfies_conditions : Prop :=
  is_midpoint X Y Z ∧ (X - Z) = 2 • (W - X)

theorem sum_of_coordinates : satisfies_conditions → (Vect2.sum_coords X + Vect2.sum_coords W) = -20 :=
by
  intro h
  sorry

end sum_of_coordinates_l216_216011


namespace Megatek_manufacturing_percentage_l216_216850

theorem Megatek_manufacturing_percentage :
  let total_angle := 360
  let angle_RD := 45
  let angle_Sales := 72
  let angle_HR := 90
  let angle_Logistics := 60
  let angle_other_departments := angle_RD + angle_Sales + angle_HR + angle_Logistics
  let angle_manufacturing := total_angle - angle_other_departments
  let percentage_manufacturing := (angle_manufacturing / total_angle.toFloat) * 100
  percentage_manufacturing ≈ 25.83 :=
by
  sorry

end Megatek_manufacturing_percentage_l216_216850


namespace find_x0_l216_216802

theorem find_x0 (x_0 : ℝ) (h : ∀ x, (x * real.log x)'' x_0 = 2) : 
  x_0 = 1 / 2 :=
sorry

end find_x0_l216_216802


namespace probability_of_all_heads_or_tails_l216_216270

theorem probability_of_all_heads_or_tails :
  let possible_outcomes := 256
  let favorable_outcomes := 2
  favorable_outcomes / possible_outcomes = 1 / 128 := by
  sorry

end probability_of_all_heads_or_tails_l216_216270


namespace quadrilateral_area_l216_216918

theorem quadrilateral_area (d h1 h2 : ℝ) (h_d : d = 30) (h_h1 : h1 = 9) (h_h2 : h2 = 6) :
  (d * (h1 + h2)) / 2 = 225 :=
by
  -- Given conditions
  rw [h_d, h_h1, h_h2]
  -- Compute the area
  norm_num
  sorry

end quadrilateral_area_l216_216918


namespace sum_of_interior_angles_l216_216858

theorem sum_of_interior_angles (n : ℕ) 
  (h : 180 * (n - 2) = 3600) :
  180 * (n + 2 - 2) = 3960 ∧ 180 * (n - 2 - 2) = 3240 :=
by
  sorry

end sum_of_interior_angles_l216_216858


namespace cheese_and_mouse_problem_l216_216594

theorem cheese_and_mouse_problem :
  let cheese := (10 : ℝ, 10 : ℝ),
      mouse_path := { p : ℝ × ℝ | p.2 = -4 * p.1 + 16 },
      is_perpendicular (p1 p2 : ℝ × ℝ) (m1 m2 : ℝ) :=
        (p2.2 - p1.2) = m1 * (p2.1 - p1.1) ∧ (p2.2 - p1.2) = m2 * (p2.1 - p1.1),
      perpendicular_slope := (1 / 4 : ℝ),
      perpendicular_line := { p : ℝ × ℝ | p.2 = (1 / 4) * p.1 + (7.5 : ℝ) },
      intersect (l1 l2 : ℝ × ℝ → Prop) :=
        ∃ p : ℝ × ℝ, l1 p ∧ l2 p,
      intersection_point := intersect mouse_path perpendicular_line
  in
  ∃ (a b : ℝ), (a, b) ∈ intersection_point ∧ a + b = 10 :=
sorry -- Proof omitted, as per instructions

end cheese_and_mouse_problem_l216_216594


namespace HCF_of_two_numbers_l216_216107

theorem HCF_of_two_numbers (a b H L P : ℕ) (h1 : P = a * b) (h2 : LCM a b = L) (h3 : P = 2460) (h4 : L = 205) : HCF a b = 12 := 
by 
  sorry

end HCF_of_two_numbers_l216_216107


namespace problem_a_problem_b_l216_216332

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a * x - x * log x

theorem problem_a (a : ℝ) (hf : ∀ x > 0, f a x ≥ 0) : a = 1 :=
by
  sorry

theorem problem_b (hf : ∀ x > 0, f 1 x ≥ 0) :
  ∃! x0 > 0, (∃! x0, ∀ x,  2*x - 1 - log x = 0 ∧ e^(-2) < f 1 x0 ∧ f 1 x0 < 2^(-2)) :=
by
  sorry

end problem_a_problem_b_l216_216332


namespace tetrahedron_distance_volume_ratio_l216_216675

open Real

noncomputable def distance (a b : Point3D) : ℝ := ( ∥ a - b ∥ )
noncomputable def volume_tetrahedron (s : ℝ) : ℝ := (s^3 * sqrt 2) / 12
noncomputable def sum_distances (e a b c d : Point3D) : ℝ := distance e a + distance e b + distance e c + distance e d

theorem tetrahedron_distance_volume_ratio (A B C D E : Point3D) (s : ℝ) 
  (h_regular : is_regular_tetrahedron A B C D s) :
  (sum_distances E A B C D) / (volume_tetrahedron s) = 6 * sqrt 3 / s^2 :=
by
  sorry

end tetrahedron_distance_volume_ratio_l216_216675


namespace sequence_sum_l216_216014

def f (x: ℝ) : ℝ := 3 / (9^x + 3)

theorem sequence_sum :
  (∑ k in finset.range 1000, f ((k + 1) / 1001)) = 500 :=
by
  sorry

end sequence_sum_l216_216014


namespace correct_algorithm_statement_l216_216565

def S₁ : Prop := "An algorithm can run endlessly"
def S₂ : Prop := "The steps of an algorithm for a problem cannot exceed 10,000"
def S₃ : Prop := "There is only one algorithm for accomplishing a task"
def S₄ : Prop := "Algorithms should be designed to be simple, convenient, and operable"

theorem correct_algorithm_statement : ¬ S₁ ∧ ¬ S₂ ∧ ¬ S₃ ∧ S₄ :=
by
  sorry

end correct_algorithm_statement_l216_216565


namespace handshakes_in_octagonal_shape_l216_216931

-- Definitions
def number_of_students : ℕ := 8

def non_adjacent_handshakes_per_student : ℕ := number_of_students - 1 - 2

def total_handshakes : ℕ := (number_of_students * non_adjacent_handshakes_per_student) / 2

-- Theorem to prove
theorem handshakes_in_octagonal_shape : total_handshakes = 20 := 
by
  -- Provide the proof here.
  sorry

end handshakes_in_octagonal_shape_l216_216931


namespace problem_1_and_2_l216_216422

theorem problem_1_and_2 (O : Set ℝ ℝ) (P : ℝ × ℝ)
  (hO : O = {p | p.1^2 + p.2^2 = 4})
  (hP : P = (-1, 0))
  (AB : Set ℝ ℝ)
  (l : Set ℝ ℝ → Prop)
  (hl : ∀p ∈ l, p = P ∨ ∃ a b ∈ O, p ∈ segment a b ∧ ab_distance a b = 2 * sqrt 3) :
  (l = {p | p.1 = -1} ∨ l = {p | p.2 = 1}) ∧
  (forall M ∈ AB, midpoint AB M → M.1^2 + M.2^2 + M.1 - M.2 = 0) :=
sorry

end problem_1_and_2_l216_216422


namespace optimal_solution_range_l216_216686

theorem optimal_solution_range (a : ℝ) (x y : ℝ) :
  (x + y - 4 ≥ 0) → (2 * x - y - 5 ≤ 0) → (x = 1) → (y = 3) →
  (-2 < a) ∧ (a < 1) :=
by
  intros h1 h2 hx hy
  sorry

end optimal_solution_range_l216_216686


namespace Patriots_won_30_games_l216_216510

def Tigers_won_more_games_than_Eagles (games_tigers games_eagles : ℕ) : Prop :=
games_tigers > games_eagles

def Patriots_won_more_than_Cubs_less_than_Mounties (games_patriots games_cubs games_mounties : ℕ) : Prop :=
games_cubs < games_patriots ∧ games_patriots < games_mounties

def Cubs_won_more_than_20_games (games_cubs : ℕ) : Prop :=
games_cubs > 20

theorem Patriots_won_30_games (games_tigers games_eagles games_patriots games_cubs games_mounties : ℕ)  :
  Tigers_won_more_games_than_Eagles games_tigers games_eagles →
  Patriots_won_more_than_Cubs_less_than_Mounties games_patriots games_cubs games_mounties →
  Cubs_won_more_than_20_games games_cubs →
  ∃ games_patriots, games_patriots = 30 := 
by
  sorry

end Patriots_won_30_games_l216_216510


namespace combinations_of_courses_l216_216181

theorem combinations_of_courses :
  let typeA_courses := 3
      typeB_courses := 4
      total_courses := 3
  in (Nat.choose 3 2 * Nat.choose 4 1 + Nat.choose 3 1 * Nat.choose 4 2) = 30 := by
  sorry

end combinations_of_courses_l216_216181


namespace M_is_real_set_l216_216456

noncomputable def z := Complex
def M : set Complex := {z | abs ((z - 1)^2) = abs (z - 1)^2}

theorem M_is_real_set : M = {z | ∃ (x : ℝ), z = x} := by
  sorry

end M_is_real_set_l216_216456


namespace units_digit_is_six_l216_216845

theorem units_digit_is_six (n : ℤ) (h : (n^2 / 10 % 10) = 7) : (n^2 % 10) = 6 :=
by sorry

end units_digit_is_six_l216_216845


namespace billy_has_2_cherries_left_l216_216994

-- Define the initial number of cherries
def initialCherries : Nat := 74

-- Define the number of cherries eaten
def eatenCherries : Nat := 72

-- Define the number of remaining cherries
def remainingCherries : Nat := initialCherries - eatenCherries

-- Theorem statement: Prove that remainingCherries is equal to 2
theorem billy_has_2_cherries_left : remainingCherries = 2 := by
  sorry

end billy_has_2_cherries_left_l216_216994


namespace evaluate_polynomial_l216_216560

theorem evaluate_polynomial (x : ℤ) (h : x = 3) : x^6 - 6 * x^2 = 675 := by
  sorry

end evaluate_polynomial_l216_216560


namespace heaviest_lightest_difference_total_weight_difference_profit_from_selling_l216_216096

-- Define the weight differences and number of boxes
def weight_diffs := [-2, -1.5, -1, 0, 2, 2.5, 3]
def num_boxes := [1, 3, 2, 2, 2, 4, 1]

-- Standard weight of one box
def standard_weight := 25
-- Excess or shortfall represented by pairs of (difference, count)
def weight_data := [(diff, count) | (diff, count) ∈ List.zip weight_diffs num_boxes]

-- Question 1: Prove the difference between the heaviest and lightest box
theorem heaviest_lightest_difference :
  let max_diff := 3
  let min_diff := -2
  max_diff - min_diff = 5 :=
by
  let max_diff := 3
  let min_diff := -2
  show max_diff - min_diff = 5
  sorry

-- Question 2: Prove total weight exceeds standard weight by 8.5 kg
theorem total_weight_difference :
  let total_diff := List.foldl (fun acc (p : Float × Nat) => acc + p.fst * p.snd) 0.0 weight_data
  total_diff = 8.5 :=
by
  let total_diff := List.foldl (fun acc (p : Float × Nat) => acc + p.fst * p.snd) 0.0 weight_data
  show total_diff = 8.5
  sorry

-- Question 3: Prove profit by selling all apples at 8 yuan/kg is 3068 yuan
theorem profit_from_selling :
  let total_diff := List.foldl (fun acc (p : Float × Nat) => acc + p.fst * p.snd) 0.0 weight_data
  let total_weight := 25 * 15 + total_diff
  total_weight * 8 = 3068 :=
by
  let total_diff := List.foldl (fun acc (p : Float × Nat) => acc + p.fst * p.snd) 0.0 weight_data
  let total_weight := 25 * 15 + total_diff
  show total_weight * 8 = 3068
  sorry

end heaviest_lightest_difference_total_weight_difference_profit_from_selling_l216_216096


namespace f_characterization_l216_216462

def f : ℤ → ℤ := sorry  -- The actual function definition will be derived.

axiom f_condition_0 : f 0 = 1
axiom f_condition_1 : f 1 = 0
axiom f_functional_eq (m n : ℕ) : f (m + n) + f (m - n) = 2 * f m * f n

theorem f_characterization (n : ℤ) : 
  f n = 
  if ∃ k : ℤ, n = 4 * k then 1 
  else if ∃ k : ℤ, n = 4 * k + 2 then -1
  else if n % 2 = 1 then 0 
  else 0 :=
begin
  sorry
end

end f_characterization_l216_216462


namespace scouts_earnings_over_weekend_l216_216061

def base_pay_per_hour : ℝ := 10.00
def tip_per_customer : ℝ := 5.00
def hours_worked_saturday : ℝ := 4.0
def customers_served_saturday : ℝ := 5.0
def hours_worked_sunday : ℝ := 5.0
def customers_served_sunday : ℝ := 8.0

def earnings_saturday : ℝ := (hours_worked_saturday * base_pay_per_hour) + (customers_served_saturday * tip_per_customer)
def earnings_sunday : ℝ := (hours_worked_sunday * base_pay_per_hour) + (customers_served_sunday * tip_per_customer)

def total_earnings : ℝ := earnings_saturday + earnings_sunday

theorem scouts_earnings_over_weekend : total_earnings = 155.00 := by
  sorry

end scouts_earnings_over_weekend_l216_216061


namespace fishery_problem_l216_216507

variables (m k : ℝ) (x y : ℝ)

-- Define the proportionality constant k to be greater than 0
axiom k_pos : 0 < k

-- Define the condition (0 < x < m)
axiom x_domain : 0 < x ∧ x < m

-- Define the relationship between y and x
def idle_rate := (m - x) / m
def y_relationship := y = k * x * idle_rate

-- Prove the maximum annual growth of the fish population
def y_max := (m * k) / 4
def max_growth_condition := (x = m / 2 → y = y_max)

-- Prove the condition for k when y reaches its maximum
def k_range := 0 < k ∧ k < 2

theorem fishery_problem :
  (∀ x, x_domain → y_relationship) ∧
  max_growth_condition ∧
  k_range :=
sorry

end fishery_problem_l216_216507


namespace evaluate_three_squared_raised_four_l216_216283

theorem evaluate_three_squared_raised_four : (3^2)^4 = 6561 := by
  sorry

end evaluate_three_squared_raised_four_l216_216283


namespace min_value_of_16x_minus_4x_plus_1_l216_216298

noncomputable def find_min_value : ℝ :=
  let f (x : ℝ) := 16^x - 4^x + 1
  in Inf (set.range f)

theorem min_value_of_16x_minus_4x_plus_1 : find_min_value = 3/4 :=
  sorry

end min_value_of_16x_minus_4x_plus_1_l216_216298


namespace find_f_2017_l216_216318

-- Define the function f satisfying the given conditions
variable {f : ℝ → ℝ}
hypothesis f_periodic : ∀ x : ℝ, f(x) * f(x + 2) = 13
hypothesis f_at_3 : f 3 = 4

theorem find_f_2017 : f 2017 = 13 / 4 :=
sorry

end find_f_2017_l216_216318


namespace Anna_money_left_l216_216240

/-- Let the initial amount be 50 dollars.
    Anna bought:
    4 packs of chewing gum at 1.50 dollars each,
    7 chocolate bars at 2.25 dollars each,
    3 large candy canes at 0.75 dollars each,
    and a pack of jelly beans that costs 3.00 dollars but was marked 20% off.
    The sales tax rate is 7.5%. -/
theorem Anna_money_left (initial_money : ℝ) (gum_packs : ℕ) (gum_price : ℝ) 
                        (chocolate_bars : ℕ) (chocolate_price : ℝ) 
                        (candy_canes : ℕ) (candy_cane_price : ℝ)
                        (jellybean_price : ℝ) 
                        (jellybean_discount_rate : ℝ) 
                        (tax_rate : ℝ) 
                        (money_left : ℝ) :
    initial_money = 50 ∧ gum_packs = 4 ∧ gum_price = 1.50 ∧
    chocolate_bars = 7 ∧ chocolate_price = 2.25 ∧
    candy_canes = 3 ∧ candy_cane_price = 0.75 ∧
    jellybean_price = 3.00 ∧ jellybean_discount_rate = 0.20 ∧
    tax_rate = 0.075 ∧ money_left = 21.62 →
    let total_gum_cost := gum_packs * gum_price in
    let total_chocolate_cost := chocolate_bars * chocolate_price in
    let total_candy_cane_cost := candy_canes * candy_cane_price in
    let jellybean_discount := jellybean_price * jellybean_discount_rate in
    let discounted_jellybean_price := jellybean_price - jellybean_discount in
    let total_before_tax := total_gum_cost + total_chocolate_cost + total_candy_cane_cost + discounted_jellybean_price in
    let sales_tax := total_before_tax * tax_rate in
    let total_after_tax := total_before_tax + sales_tax in
    initial_money - total_after_tax = money_left :=
begin
    sorry
end

end Anna_money_left_l216_216240


namespace infinite_non_overlapping_circles_l216_216831

-- Define the setting of the problem
def circles_touching_six_others (C : Set ℝ) :=
  ∀ c ∈ C, (Finset.filter (λ d, c ≠ d ∧ ∃ r > 0, ∃ d ∈ C, (c - d).norm = r) C).card ≥ 6

-- Define the main theorem statement
theorem infinite_non_overlapping_circles (C : Set ℝ) 
  (non_overlapping : ∀ c1 c2 ∈ C, c1 ≠ c2 → disjoint (ball c1) (ball c2)) 
  (touching_six : circles_touching_six_others C) :
  infinite C :=
sorry

end infinite_non_overlapping_circles_l216_216831


namespace probability_of_same_team_is_one_third_l216_216420

noncomputable def probability_same_team : ℚ :=
  let teams := 3
  let total_combinations := teams * teams
  let successful_outcomes := teams
  successful_outcomes / total_combinations

theorem probability_of_same_team_is_one_third :
  probability_same_team = 1 / 3 := by
  sorry

end probability_of_same_team_is_one_third_l216_216420


namespace lines_intersect_at_one_point_l216_216473

theorem lines_intersect_at_one_point
  (X Y Z A B C D : Point)
  (h_line: collinear X Y Z)
  (h_equilateral_XAB: equilateral_triangle X A B ∧ counterclockwise_oriented X A B)
  (h_equilateral_YBC: equilateral_triangle Y B C ∧ clockwise_oriented Y B C)
  (h_equilateral_ZCD: equilateral_triangle Z C D ∧ counterclockwise_oriented Z C D) :
  ∃ P : Point, collinear P A C ∧ collinear P B D ∧ collinear P X Y :=
sorry

end lines_intersect_at_one_point_l216_216473


namespace ratio_a_c_l216_216873

theorem ratio_a_c (a b c d : ℚ) 
  (h1 : a / b = 5 / 4) 
  (h2 : c / d = 4 / 1) 
  (h3 : d / b = 2 / 5) : 
  a / c = 25 / 32 := 
by sorry

end ratio_a_c_l216_216873


namespace minimum_value_of_f_l216_216310

noncomputable def f (x : ℝ) : ℝ := 16^x - 4^x + 1

theorem minimum_value_of_f : ∃ (x : ℝ), ∀ (y : ℝ), f(y) ≥ f(x) ∧ f(x) = 3 / 4 :=
sorry

end minimum_value_of_f_l216_216310


namespace find_area_of_triangle_ABC_l216_216430

-- Definitions based on conditions
def BC : ℝ := 12
def angle_C_deg : ℝ := 30
def sin_30 : ℝ := 1 / 2
def AC : ℝ := BC * sin_30

-- The base of the triangle is BC, the height is AC
def base : ℝ := BC
def height : ℝ := AC

-- Statement of the problem
theorem find_area_of_triangle_ABC :
  (1 / 2) * base * height = 36 := by
  sorry

end find_area_of_triangle_ABC_l216_216430


namespace average_marks_is_70_l216_216571

variable (P C M : ℕ)

-- Condition: The total marks in physics, chemistry, and mathematics is 140 more than the marks in physics
def total_marks_condition : Prop := P + C + M = P + 140

-- Definition of the average marks in chemistry and mathematics
def average_marks_C_M : ℕ := (C + M) / 2

theorem average_marks_is_70 (h : total_marks_condition P C M) : average_marks_C_M C M = 70 :=
sorry

end average_marks_is_70_l216_216571


namespace solve_for_s_l216_216591

---
theorem solve_for_s (s : ℝ) (h₀: s > 0)
(h₁ : let θ := real.pi / 3 in area = s * 3 * (s * real.sin θ))
(h₂ : area = 27 * real.sqrt 3) : s = 3 * real.sqrt 2 := by
  sorry

end solve_for_s_l216_216591


namespace circumcircle_MXY_tangent_to_ω_l216_216835

open EuclideanGeometry
noncomputable def midpoint (p q : Point) := (1 / 2) * (p + q)

variables {A B C D E M X Y : Point}
variables {ω : Circle}

-- Existing Conditions
def point_M_is_midpoint_BC : midpoint B C = M := sorry
def ω_is_tangent_to_BC_at_M : tangent_point ω M BC := sorry
def ω_passes_through_A : on_circle ω A := sorry
def ω_intersects_AB_at_D : ∃ D, (on_circle ω D ∧ D ≠ A ∧ collinear A B D) := sorry
def ω_intersects_AC_at_E : ∃ E, (on_circle ω E ∧ E ≠ A ∧ collinear A C E) := sorry
def point_X_is_midpoint_BE : midpoint B E = X := sorry
def point_Y_is_midpoint_CD : midpoint C D = Y := sorry

-- Proof Goal
theorem circumcircle_MXY_tangent_to_ω :
  tangent_point (circumcircle M X Y) M ω := 
  sorry

end circumcircle_MXY_tangent_to_ω_l216_216835


namespace maximum_take_home_pay_l216_216412

def tax_rate (x : ℝ) : ℝ := x / 100
def income (x : ℝ) : ℝ := 1000 * x^2
def tax (x : ℝ) : ℝ := tax_rate x * income x
def take_home_pay (x : ℝ) : ℝ := income x - tax x

theorem maximum_take_home_pay : ∃ x : ℝ, 0 ≤ x ∧ 1000x^2 - 10x^3 = 4444.44 :=
by
  sorry

end maximum_take_home_pay_l216_216412


namespace cd_eq_2de_l216_216006

theorem cd_eq_2de {A B C D E : Type} [circle : Circular_Area A C]
  (h1 : Tangent B C circle)
  (h2 : Tangent B D circle)
  (h3 : E ∈ (circumscribed_circle (triangle A B C)))
  (h4 : SecondIntersection (line_through C D) E (circumscribed_circle (triangle A B C))) :
  length C D = 2 * length D E :=
by
  sorry

end cd_eq_2de_l216_216006


namespace comparison_of_values_l216_216690

noncomputable def a : ℝ := 3^0.6
noncomputable def b : ℝ := Real.logBase 3 0.6
noncomputable def c : ℝ := 0.6^3

theorem comparison_of_values : a > c ∧ c > b := by
  sorry

end comparison_of_values_l216_216690


namespace problem_statement_l216_216812

variables {x y z : ℝ}

-- Assume x, y, z are positive real numbers
-- Assume the product of x, y, z equals to 1
theorem problem_statement 
   (hx_pos : 0 < x) 
   (hy_pos : 0 < y) 
   (hz_pos : 0 < z) 
   (hxyz_one : x * y * z = 1) 
   : (xy_div_sum: 
       (x * y) / (x^5 + x * y + y^5) + 
       (y * z) / (y^5 + y * z + z^5) + 
       (z * x) / (z^5 + z * x + x^5)  ≤ 1 )
     ∧ (equality_holds: x = 1 ∧ y = 1 ∧ z = 1 → 
       ( (x * y) / (x^5 + x * y + y^5) + 
         (y * z) / (y^5 + y * z + z^5) + 
         (z * x) / (z^5 + z * x + x^5) = 1))
:= 
begin
  sorry
end

end problem_statement_l216_216812


namespace perimeter_constant_l216_216612

variable (A B C P Q O : Point) (α : Angle)

def isosceles_triangle (A B C : Point) : Prop :=
  dist A B = dist B C

def midpoint (O A C : Point) : Prop :=
  dist O A = dist O C

def rotate_around_vertex (α : Angle) (O : Point) (A B C P Q : Point) : Prop :=
  ∠BAC = α ∧ α rotates around O ∧
  intersects AB at P ∧ intersects BC at Q

theorem perimeter_constant 
  (h_iso : isosceles_triangle A B C) 
  (h_mid : midpoint O A C)
  (h_rot : rotate_around_vertex α O A B C P Q) 
  : perimeter (triangle P B Q) = constant := 
sorry

end perimeter_constant_l216_216612


namespace bridge_length_l216_216190

theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (time_sec : ℝ) 
  (h_train_length : train_length = 410) 
  (h_train_speed : train_speed_kmh = 45) 
  (h_time : time_sec = 44) : 
  let train_speed_ms := (train_speed_kmh * 1000 / 3600),
      total_distance := train_speed_ms * time_sec,
      bridge_length := total_distance - train_length
  in bridge_length = 140 :=
by
  rw [h_train_length, h_train_speed, h_time]
  let train_speed_ms := (45 * 1000 / 3600)
  have h_train_speed_ms : train_speed_ms = 12.5 := by norm_num
  rw [h_train_speed_ms]
  let total_distance := (12.5 * 44)
  have h_total_distance : total_distance = 550 := by norm_cast; norm_num
  rw [h_total_distance]
  let bridge_length := (550 - 410)
  have h_bridge_length : bridge_length = 140 := by norm_num
  exact h_bridge_length

end bridge_length_l216_216190


namespace negation_exists_x_squared_lt_zero_l216_216871

open Classical

theorem negation_exists_x_squared_lt_zero :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) :=
by 
  sorry

end negation_exists_x_squared_lt_zero_l216_216871


namespace base7_conversion_and_multiplication_l216_216284

theorem base7_conversion_and_multiplication :
  let x := 231
  let y := 452
  (BaseConversion.toBaseN 7 x = 450 ∧
  BaseConversion.toBaseN 7 y = 1214 ∧
  x * y = 104412 ∧
  BaseConversion.toBaseN 7 (x * y) = 613260) :=
by
  sorry

end base7_conversion_and_multiplication_l216_216284


namespace find_x_l216_216013

def sum_sequence (a b : ℕ) : ℕ :=
  (b * (2 * a + b - 1)) / 2  -- Sum of an arithmetic progression

theorem find_x (x : ℕ) (h1 : sum_sequence x 10 = 65) : x = 2 :=
by {
  -- the proof goes here
  sorry
}

end find_x_l216_216013


namespace distance_between_vertices_l216_216074

theorem distance_between_vertices :
  ∀ x y : ℝ, (sqrt(x^2 + y^2) + abs (y - 1) = 3) →
  let vertex1 := (0, 2)
  let vertex2 := (0, -1)
  distance vertex1 vertex2 = 3 :=
begin
  sorry
end

end distance_between_vertices_l216_216074


namespace max_diagonals_in_grid_l216_216046

-- Define the dimensions of the grid
def grid_width := 8
def grid_height := 5

-- Define the number of 1x2 rectangles
def number_of_1x2_rectangles := grid_width / 2 * grid_height

-- State the theorem
theorem max_diagonals_in_grid : number_of_1x2_rectangles = 20 := 
by 
  -- Simplify the expression
  sorry

end max_diagonals_in_grid_l216_216046


namespace find_sin_alpha_and_tan_alpha_find_expression_value_l216_216350

variable (α : Real)

-- Define the conditions
def cos_alpha : ℝ := -3 / 5
def second_quadrant : True := True

-- Define the trigonometric identities
def sin_alpha (α : Real) := sin α
def tan_alpha (α : Real) := tan α

-- Proof statements
theorem find_sin_alpha_and_tan_alpha (h1 : cos α = cos_alpha) (h2 : second_quadrant) :
  sin α = 4 / 5 ∧ tan α = -4 / 3 :=
  sorry

theorem find_expression_value (h1 : cos α = cos_alpha) (h2 : second_quadrant) :
  1 / (cos α ^ 2 + 2 * sin α * cos α) = -5 / 3 :=
  sorry

end find_sin_alpha_and_tan_alpha_find_expression_value_l216_216350


namespace ned_good_games_l216_216045

def fraction_of_games (total_games : ℕ) (percentage : ℚ) : ℕ :=
  ⌊percentage * total_games⌋

def total_good_games (games_from_friend games_from_sale : ℕ) (percentage_friend percentage_sale : ℚ) : ℕ :=
  fraction_of_games games_from_friend percentage_friend + fraction_of_games games_from_sale percentage_sale

theorem ned_good_games (games_from_friend games_from_sale : ℕ) (percentage_friend percentage_sale : ℚ) :
  games_from_friend = 11 ∧ percentage_friend = 0.90 ∧ games_from_sale = 22 ∧ percentage_sale = 0.60 →
  total_good_games games_from_friend games_from_sale percentage_friend percentage_sale = 22 :=
by {
  intro h,
  cases h with h_friend h_rest,
  cases h_rest with h_percentage_friend h_rest,
  cases h_rest with h_sale h_percentage_sale,
  rw [h_friend, h_percentage_friend, h_sale, h_percentage_sale],
  sorry
}

end ned_good_games_l216_216045


namespace area_of_intersection_l216_216106

-- Define the first circle centered at (0, 3)
def circle1 (x y : ℝ) : Prop := (x - 0)^2 + (y - 3)^2 = 3^2

-- Define the second circle centered at (3, 0)
def circle2 (x y : ℝ) : Prop := (x - 3)^2 + (y - 0)^2 = 3^2

-- The statement to be proved
theorem area_of_intersection : 
  let area := (9 * π - 18) / 2 in 
  ∃ p1 p2 : ℝ × ℝ, 
    circle1 p1.1 p1.2 ∧
    circle2 p1.1 p1.2 ∧
    circle1 p2.1 p2.2 ∧
    circle2 p2.1 p2.2 ∧
    -- The area calculation follows the correct geometric arguments, which we will denote here
    -- without going into the details
    sorry := area

end area_of_intersection_l216_216106


namespace room_length_l216_216080

/-- Define the conditions -/
def width : ℝ := 3.75
def cost_paving : ℝ := 6187.5
def cost_per_sqm : ℝ := 300

/-- Prove that the length of the room is 5.5 meters -/
theorem room_length : 
  (cost_paving / cost_per_sqm) / width = 5.5 :=
by
  sorry

end room_length_l216_216080


namespace range_of_a_l216_216561

noncomputable def proof_problem (a : ℝ) : Prop :=
  ∀ x : ℝ, (1 < x ∧ x < 2) → x^2 + 1 < 2*x + log x a

theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, (1 < x ∧ x < 2) → x^2 + 1 < 2*x + log x a) ↔ (1 < a ∧ a ≤ 2) :=
by
  sorry

end range_of_a_l216_216561


namespace number_of_valid_pairs_l216_216152

open Nat

theorem number_of_valid_pairs :
  ∃ (n : ℕ), n = 10 ∧
    ∃ (xy_pairs : List (ℕ × ℕ)),
      xy_pairs.length = n ∧
      ∀ (p : ℕ × ℕ), p ∈ xy_pairs →
        let x := p.1
        let y := p.2
        0 ≤ x ∧ x ≤ 10 ∧ 0 ≤ y ∧ y ≤ 10 ∧ Prime ((x + y)^2 + (x * y - 1)^2) := by
sorry

end number_of_valid_pairs_l216_216152


namespace fraction_subtraction_l216_216624

theorem fraction_subtraction (h : ((8 : ℚ) / 21 - (10 / 63) = (2 / 9))) : 
  8 / 21 - 10 / 63 = 2 / 9 :=
by
  sorry

end fraction_subtraction_l216_216624


namespace minimum_value_of_expression_l216_216305

theorem minimum_value_of_expression : ∃ x : ℝ, (16 ^ x - 4 ^ x + 1 = 3 / 4) :=
sorry

end minimum_value_of_expression_l216_216305


namespace area_white_portion_l216_216879

/-- The dimensions of the sign --/
def sign_width : ℝ := 7
def sign_height : ℝ := 20

/-- The areas of letters "S", "A", "V", and "E" --/
def area_S : ℝ := 14
def area_A : ℝ := 16
def area_V : ℝ := 12
def area_E : ℝ := 12

/-- Calculate the total area of the sign --/
def total_area_sign : ℝ := sign_width * sign_height

/-- Calculate the total area covered by the letters --/
def total_area_letters : ℝ := area_S + area_A + area_V + area_E

/-- Calculate the area of the white portion of the sign --/
theorem area_white_portion : total_area_sign - total_area_letters = 86 := by
  sorry

end area_white_portion_l216_216879


namespace fruit_pie_pieces_l216_216249

/-- Cassandra's fruit pie calculations -/
theorem fruit_pie_pieces :
  let granny_smith_apples := 4 * 12 in
  let granny_smith_pie_slices := 4 * 6 in
  let fuji_apples := 2 * 12 in
  let fuji_pie_slices := 3 * 8 in
  let bartlett_pears := 3 * 12 in
  let pear_tart_slices := 2 * 10 in
  (granny_smith_apples / granny_smith_pie_slices = 2) ∧
  (fuji_apples / fuji_pie_slices = 1) ∧
  (bartlett_pears / pear_tart_slices = 1.8) :=
by
  sorry

end fruit_pie_pieces_l216_216249


namespace trains_meet_distance_l216_216604

theorem trains_meet_distance (speed1 speed2 : ℕ) (time1 time2 hr_diff distance gap_rel_speed : ℕ) : 
  speed1 = 30 → 
  speed2 = 40 →
  time1 = 9 →
  time2 = 15 →
  hr_diff = time2 - time1 →
  distance = speed1 * hr_diff →
  gap_rel_speed = speed2 - speed1 →
  720 = distance + gap_rel_speed * 18 :=
begin
  intros,
  rw [h4, h5, h3],
  exact rfl,
end

end trains_meet_distance_l216_216604


namespace pedestrian_meeting_l216_216610

-- Define the types for pedestrians and lines
structure Line (α : Type) := 
  (a : α) (b : α) -- a line is defined by two points

variable {α : Type} [linear_ordered_semiring α] [nontrivial α] -- α represents coordinates which are non-parallel

-- Define the conditions as hypotheses
variables (P1 P2 P3 P4 : Line α)
variables (h1 : P1 ≠ P2) (h2 : P1 ≠ P3) (h3 : P1 ≠ P4)
variables (h4 : P2 ≠ P3) (h5 : P2 ≠ P4)
variables (h6 : P3 ≠ P4)

-- Define the intersections as existence conditions
variables (meet12 : ∃ t : α, P1.a t = P2.a t ∧ P1.b t = P2.b t)
variables (meet13 : ∃ t : α, P1.a t = P3.a t ∧ P1.b t = P3.b t)
variables (meet14 : ∃ t : α, P1.a t = P4.a t ∧ P1.b t = P4.b t)
variables (meet23 : ∃ t : α, P2.a t = P3.a t ∧ P2.b t = P3.b t)
variables (meet24 : ∃ t : α, P2.a t = P4.a t ∧ P2.b t = P4.b t)

theorem pedestrian_meeting : ∃ t : α, (P3.a t = P4.a t) ∧ (P3.b t = P4.b t) :=
sorry

end pedestrian_meeting_l216_216610


namespace DivisibilityBy27_l216_216059

theorem DivisibilityBy27 : ∃ (n : ℕ), (n = (10^27 - 1) / 9) → 27 ∣ n := 
by
  -- let n be the number with 27 ones
  let n := ((10^27 - 1) / 9)
  -- we need to prove that 27 divides n
  have h : 27 ∣ n := sorry
  exact ⟨n, rfl, h⟩

end DivisibilityBy27_l216_216059


namespace last_two_digits_square_l216_216865

theorem last_two_digits_square (n : ℕ) (hnz : (n % 10 ≠ 0) ∧ ((n ^ 2) % 100 = n % 10 * 11)): ((n ^ 2) % 100 = 44) :=
sorry

end last_two_digits_square_l216_216865


namespace tetrahedron_ratio_l216_216451

-- Define the problem statement in Lean 4
theorem tetrahedron_ratio 
  (A B C D P : Type) 
  (h_regular_tetrahedron : RegularTetrahedron A B C D)
  (h_equidistant : ∀ v ∈ {A, B, C, D}, dist P v = dist P A)
  (a : ℝ) (h_edge_length : ∀ (u v : {A, B, C, D}), u ≠ v → dist u v = a) :
  let t := 4 * (dist P A)
  let T := 6 * (dist P (midpoint ℝ (λ x : ℝ, x) A B))
  in t / T = real.sqrt 2 / 2 :=
by
  sorry

end tetrahedron_ratio_l216_216451


namespace total_letters_received_l216_216383

theorem total_letters_received :
  ∀ (g b m t : ℕ), 
    b = 40 →
    g = b + 10 →
    m = 2 * (g + b) →
    t = g + b + m → 
    t = 270 :=
by
  intros g b m t hb hg hm ht
  rw [hb, hg, hm, ht]
  sorry

end total_letters_received_l216_216383


namespace find_enclosed_area_l216_216859

def area_square (side_length : ℕ) : ℕ :=
  side_length * side_length

def area_triangle (base height : ℕ) : ℕ :=
  (base * height) / 2

theorem find_enclosed_area :
  let side1 := 3
  let side2 := 6
  let area1 := area_square side1
  let area2 := area_square side2
  let area_tri := 2 * area_triangle side1 side2
  area1 + area2 + area_tri = 63 :=
by
  sorry

end find_enclosed_area_l216_216859


namespace option_A_correct_l216_216323

theorem option_A_correct (p : ℕ) (h1 : p > 1) (h2 : p % 2 = 1) : 
  (p - 1)^(p/2 - 1) - 1 ≡ 0 [MOD (p - 2)] :=
sorry

end option_A_correct_l216_216323


namespace correct_quotient_l216_216141

theorem correct_quotient (D Q : ℕ) (h1 : 21 * Q = 12 * 56) : Q = 32 :=
by {
  -- Proof to be provided
  sorry
}

end correct_quotient_l216_216141


namespace general_term_arithmetic_sequence_l216_216342

-- Define an arithmetic sequence with first term a1 and common ratio q
def arithmetic_sequence (a1 : ℤ) (q : ℤ) (n : ℕ) : ℤ :=
  a1 * q ^ (n - 1)

-- Theorem: given the conditions, prove that the general term is a1 * q^(n-1)
theorem general_term_arithmetic_sequence (a1 q : ℤ) (n : ℕ) :
  arithmetic_sequence a1 q n = a1 * q ^ (n - 1) :=
by
  sorry

end general_term_arithmetic_sequence_l216_216342


namespace minimum_value_of_f_l216_216313

noncomputable def f (x : ℝ) : ℝ := 16^x - 4^x + 1

theorem minimum_value_of_f : ∃ (x : ℝ), ∀ (y : ℝ), f(y) ≥ f(x) ∧ f(x) = 3 / 4 :=
sorry

end minimum_value_of_f_l216_216313


namespace class_size_l216_216844

theorem class_size
  (S_society : ℕ) (S_music : ℕ) (S_both : ℕ) (S : ℕ)
  (h_society : S_society = 25)
  (h_music : S_music = 32)
  (h_both : S_both = 27)
  (h_total : S = S_society + S_music - S_both) :
  S = 30 :=
by
  rw [h_society, h_music, h_both] at h_total
  exact h_total

end class_size_l216_216844


namespace scholarship_total_l216_216828

-- Definitions of the money received by Wendy, Kelly, Nina, and Jason based on the given conditions
def wendy_scholarship : ℕ := 20000
def kelly_scholarship : ℕ := 2 * wendy_scholarship
def nina_scholarship : ℕ := kelly_scholarship - 8000
def jason_scholarship : ℕ := (3 * kelly_scholarship) / 4

-- Total amount of scholarships
def total_scholarship : ℕ := wendy_scholarship + kelly_scholarship + nina_scholarship + jason_scholarship

-- The proof statement that needs to be proven
theorem scholarship_total : total_scholarship = 122000 := by
  -- Here we use 'sorry' to indicate that the proof is not provided.
  sorry

end scholarship_total_l216_216828


namespace kittens_indeterminate_weight_l216_216520

open Nat

theorem kittens_indeterminate_weight {n : ℕ} (a : Fin n → ℕ) 
  (h1 : a 0 + a 1 = 80) 
  (h2 : a (n-4) + a (n-3) + a (n-2) + a (n-1) = 200) :
  ∃ X ≥ 480, ∀ X < 480, (∃ k, ∑ i in (Fin n), a i = X → n = k) :=
by
  sorry

end kittens_indeterminate_weight_l216_216520


namespace find_b_l216_216634

def h (x : ℝ) : ℝ := 5 * x - 6

theorem find_b : ∃ b : ℝ, h(b) = 1 ∧ b = 7 / 5 := by
  use 7 / 5
  constructor
  { calc h(7 / 5) = 5 * (7 / 5) - 6 : rfl
               ... = 7 - 6         : by norm_num
               ... = 1             : by norm_num }
  { rfl }

end find_b_l216_216634


namespace min_value_of_16x_minus_4x_plus_1_l216_216299

noncomputable def find_min_value : ℝ :=
  let f (x : ℝ) := 16^x - 4^x + 1
  in Inf (set.range f)

theorem min_value_of_16x_minus_4x_plus_1 : find_min_value = 3/4 :=
  sorry

end min_value_of_16x_minus_4x_plus_1_l216_216299


namespace paint_fraction_l216_216397

variable (T C : ℕ) (h : T = 60) (t : ℕ) (partial_t : ℚ)

theorem paint_fraction (hT : T = 60) (ht : t = 12) : partial_t = t / T := by
  rw [ht, hT]
  norm_num
  sorry

end paint_fraction_l216_216397


namespace crow_eats_quarter_in_twenty_hours_l216_216581

-- Given: The crow eats 1/5 of the nuts in 4 hours
def crow_eating_rate (N : ℕ) : ℕ := N / 5 / 4

-- Prove: It will take 20 hours to eat 1/4 of the nuts
theorem crow_eats_quarter_in_twenty_hours (N : ℕ) (h : ℕ) (h_eq : h = 20) : 
  ((N / 5) / 4 : ℝ) = ((N / 4) / h : ℝ) :=
by
  sorry

end crow_eats_quarter_in_twenty_hours_l216_216581


namespace largest_root_floor_l216_216029

theorem largest_root_floor :
  let x₀ := (1 + Real.sqrt 3) in
  x₀ = (Real.sqrt 3 + 1) → (Real.floor (10 * x₀) = 27) :=
begin
  intro x₀_definition,
  sorry,
end

end largest_root_floor_l216_216029


namespace determine_k_l216_216682

noncomputable def arithmetic_sequence (d a1 n : ℕ) : ℕ := a1 + (n - 1) * d

theorem determine_k : 
  ∀ (d k : ℕ), d = 2 → a1 = -5 → (∑ i in (finset.range 11), a1 + i * d) - (a1 + (k - 1) * d) / 10 = 4 → k = 11 :=
by
  intros d k h_d h_a1 h_mean
  unfold arithmetic_sequence at *
  sorry

end determine_k_l216_216682


namespace measure_of_B_sin_A_sin_C_l216_216406

variable {a b c : ℝ}
variable {S : ℝ}
variable {A B C : ℝ}

-- Given condition on area of the triangle
axiom triangle_area (h_area : S = (Real.sqrt 3 / 4) * (a^2 + c^2 - b^2))

-- To prove: Measure of angle B
theorem measure_of_B (h_area : S = (Real.sqrt 3 / 4) * (a^2 + c^2 - b^2)) :
  B = π / 3 :=
sorry

-- Given geometry sequence condition
axiom geometric_sequence (h_sequence : a * c = b^2)

-- To prove: Value of sin A * sin C
theorem sin_A_sin_C (h_sequence : a * c = b^2) (h_area : S = (Real.sqrt 3 / 4) * (a^2 + c^2 - b^2)) :
  Real.sin A * Real.sin C = 3 / 4 :=
sorry

end measure_of_B_sin_A_sin_C_l216_216406


namespace problem1_problem2_l216_216248

-- Definitions for the first problem
def cube_root (x : ℝ) := x ^ (1 / 3)
def sqrt (x : ℝ) := x ^ (1 / 2)
def power (x : ℝ) (n : ℝ) := x ^ n

-- Definitions for the second problem
def mult (x y : ℝ) := x * y
def div (x y : ℝ) := x / y
def sub (x y : ℝ) := x - y

-- Theorem for the first problem
theorem problem1 :
  cube_root 8 + sqrt ((-2)^2) - power (-3) 0 = 3 := by
  sorry

-- Theorem for the second problem
theorem problem2 :
  mult (sqrt 2) (sub (sqrt 6) (sqrt 3)) + div (sqrt 12) (sqrt 2) = 2 * sqrt 3 := by
  sorry

end problem1_problem2_l216_216248


namespace find_h_l216_216861

theorem find_h (h j k : ℤ) 
  (H1 : 2025 = 4 * h^2 + j)
  (H2 : 2026 = -3 * h^2 + k)
  (H3 : ∃ a b : ℤ, a ≠ 0 ∧ b ≠ 0 ∧ (y = 4 * (x - h)^2 + (2025 - 4 * h^2)) = 0)
  (H4 : ∃ c d e : ℤ, c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ (y = (x - h)^3 - 3 * (x - h)^2 + (2026 + 3 * h^2)) = 0):
  h = 45 := 
sorry

end find_h_l216_216861


namespace rotate_90_produces_result_l216_216932

-- Definitions from the conditions
def complex_num := 5 + 2 * complex.I
def rotate_90_ccw := complex.I

-- The proof statement
theorem rotate_90_produces_result : (complex_num * rotate_90_ccw) = -2 + 5 * complex.I := by
  sorry

end rotate_90_produces_result_l216_216932


namespace cosine_evaluation_l216_216689

variable (α : ℝ)

theorem cosine_evaluation
  (h : Real.sin (Real.pi / 6 + α) = 1 / 3) :
  Real.cos (Real.pi / 3 - α) = 1 / 3 :=
sorry

end cosine_evaluation_l216_216689


namespace arithmetic_prog_includes_1999_l216_216239

-- Definitions based on problem conditions
def is_in_arithmetic_progression (a d n : ℕ) : ℕ := a + (n - 1) * d

theorem arithmetic_prog_includes_1999
  (d : ℕ) (h_pos : d > 0) 
  (h_includes7 : ∃ n:ℕ, is_in_arithmetic_progression 7 d n = 7)
  (h_includes15 : ∃ n:ℕ, is_in_arithmetic_progression 7 d n = 15)
  (h_includes27 : ∃ n:ℕ, is_in_arithmetic_progression 7 d n = 27) :
  ∃ n:ℕ, is_in_arithmetic_progression 7 d n = 1999 := 
sorry

end arithmetic_prog_includes_1999_l216_216239


namespace cone_base_circumference_l216_216178

noncomputable def radius (V : ℝ) (h : ℝ) : ℝ :=
  real.sqrt (3 * V / (π * h))

theorem cone_base_circumference
  (V : ℝ) (h : ℝ) (C : ℝ) (π_neq_zero : π ≠ 0)
  (V_eq : V = 24 * π) (h_eq : h = 6)
  (C_eq : C = 4 * real.sqrt 3 * π) :
  2 * π * radius V h = C :=
by {
  -- Simplify the expressions using the provided conditions
  rw [V_eq, h_eq, C_eq, radius, real.sqrt_mul, real.sqrt_div, mul_comm, ← mul_assoc],
  -- Answer follows directly from given conditions and basic algebra
  sorry
}

end cone_base_circumference_l216_216178


namespace inequality_proof_l216_216030

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) : 
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000 / 9 :=
by
  sorry

end inequality_proof_l216_216030


namespace ellipse_foci_distance_l216_216683

theorem ellipse_foci_distance (a b : ℝ) (ha : a = 10) (hb : b = 8) :
  2 * Real.sqrt (a^2 - b^2) = 12 :=
by
  rw [ha, hb]
  -- Proof follows here, but we skip it using sorry.
  sorry

end ellipse_foci_distance_l216_216683


namespace total_food_per_day_l216_216533

theorem total_food_per_day :
  let num_puppies := 4
  let num_dogs := 3
  let dog_meal_weight := 4
  let dog_meals_per_day := 3
  let dog_food_per_day := dog_meal_weight * dog_meals_per_day
  let total_dog_food_per_day := dog_food_per_day * num_dogs
  let puppy_meal_weight := dog_meal_weight / 2
  let puppy_meals_per_day := dog_meals_per_day * 3
  let puppy_food_per_day := puppy_meal_weight * puppy_meals_per_day
  let total_puppy_food_per_day := puppy_food_per_day * num_puppies
  total_dog_food_per_day + total_puppy_food_per_day = 108 :=
by
  sorry

end total_food_per_day_l216_216533


namespace find_minimum_cos_diff_l216_216575

noncomputable section

theorem find_minimum_cos_diff
  (x y z : ℝ)
  (h1 : sqrt 3 * sin x = tan y)
  (h2 : 2 * sin y = cot z)
  (h3 : sin z = 2 * tan x) :
  ∃ (min_val : ℝ), min_val = (- 7 * sqrt 2 / 6) ∧ min_val = cos x - cos z :=
begin
  sorry
end

end find_minimum_cos_diff_l216_216575


namespace tan_phi_half_l216_216798

variable (m : ℝ) (φ : ℝ)
variable (hm : m > 0)
variable (hmatrix : matrix (fin 2) (fin 2) ℝ)

noncomputable def D : matrix (fin 2) (fin 2) ℝ :=
  !![![m, 0], ![0, m]]

noncomputable def R : matrix (fin 2) (fin 2) ℝ :=
  !![![cos φ, -sin φ], ![sin φ, cos φ]]

noncomputable def given_matrix : matrix (fin 2) (fin 2) ℝ :=
  !![![10, -5], ![5, 10]]

theorem tan_phi_half (hm : m > 0) (h : R m φ ⬝ D m = given_matrix) : 
  tan φ = 1 / 2 :=
by
  sorry

end tan_phi_half_l216_216798


namespace ice_cream_problem_l216_216480

-- Define the number of flavors for Robin and Alex
def robin_flavors := 5
def alex_flavors := 4

-- Define the factorial function
noncomputable def factorial : ℕ → ℕ
| 0 => 1
| (n + 1) => (n + 1) * factorial n

-- Calculate the number of permutations for Robin's and Alex's cones
def robin_permutations := factorial robin_flavors
def alex_permutations := factorial alex_flavors

-- Calculate the total number of different stacking orders
def total_permutations := robin_permutations * alex_permutations

-- The proof statement
theorem ice_cream_problem : total_permutations = 2880 :=
by
  have robin_perm : robin_permutations = 120 := by
    unfold robin_permutations robin_flavors;
    exact nat.factorial_five
  have alex_perm : alex_permutations = 24 := by
    unfold alex_permutations alex_flavors;
    exact nat.factorial_four
  unfold total_permutations
  rw [robin_perm, alex_perm]
  norm_num
  sorry

end ice_cream_problem_l216_216480


namespace max_value_of_quadratic_function_l216_216081

def quadratic_function (x : ℝ) : ℝ := -x^2 + 2*x + 4

theorem max_value_of_quadratic_function : ∃ x : ℝ, quadratic_function x = 5 ∧ ∀ y : ℝ, quadratic_function y ≤ 5 :=
by
  sorry

end max_value_of_quadratic_function_l216_216081


namespace largest_taidu_is_10112369_l216_216101

def is_taidu (n : Nat → Nat) : Prop :=
  ∀ i ≥ 2, n i ≥ n (i - 1) + n (i - 2)

theorem largest_taidu_is_10112369 :
  ∃ n : Nat → Nat, (is_taidu n ∧ list.join (list.map n [0,1,2,3,4,5,6,7]) = "10112369") :=
sorry

end largest_taidu_is_10112369_l216_216101


namespace min_value_16x_minus_4x_plus_1_l216_216303

theorem min_value_16x_minus_4x_plus_1 :
  ∀ x : ℝ, 16 ^ x - 4 ^ x + 1 ≥ 3 / 4 ∧ (∃ y : ℝ, 16 ^ y - 4 ^ y + 1 = 3 / 4) :=
by
  sorry

end min_value_16x_minus_4x_plus_1_l216_216303


namespace leading_digit_one_l216_216321

theorem leading_digit_one (n : ℕ) (hn : n > 0) :
  ∃ k ∈ {n, n+1, n+2, n+3, n+4}, (∀ d, 10^(k : ℤ) ≤ 17^k / 10^(d : ℤ) < 10^(k : ℤ) → d = 1) :=
sorry

end leading_digit_one_l216_216321


namespace _l216_216776

variables {A B C D E F G : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace G]

noncomputable def triangle_sides := {AB : ℝ, BC : ℝ, CA : ℝ}
noncomputable def seg_parallel := {DE_BC : Prop, FG_BC : Prop}
noncomputable def same_perimeter := {ADF_perimeter : Prop, DEFG_perimeter : Prop, EBCG_perimeter : Prop}

noncomputable theorem determine_DE_FG_sum
  (h1: triangle_sides) 
  (h2: AB = 2)
  (h3: BC = 3)
  (h4: CA = 4)
  (h5: seg_parallel)
  (h6: DE_BC)
  (h7: FG_BC)
  (h8: same_perimeter)
  (h9: ADF_perimeter)
  (h10: DEFG_perimeter)
  (h11: EBCG_perimeter):
  (DE + FG) = 9 / 2 :=
by {
  sorry
}

end _l216_216776


namespace reflection_lies_on_circumcircle_l216_216464

open EuclideanGeometry

noncomputable def reflect_point (p l : Point) : Point :=
sorry

theorem reflection_lies_on_circumcircle
  (A B C D M N : Point)
  (h_triangle : ¬ collinear A B C)
  (hD_on_BC : lies_on_segment D B C)
  (hM : is_perpendicular_bisector (segment B D) M)
  (hN : is_perpendicular_bisector (segment C D) N)
  (hM_on_AB : lies_on_segment M A B)
  (hN_on_AC : lies_on_segment N A C) :
  lies_on_circumcircle (reflect_point D (line M N)) A B C :=
sorry

end reflection_lies_on_circumcircle_l216_216464


namespace farmer_pomelos_dozen_l216_216002

theorem farmer_pomelos_dozen (pomelos_last_week : ℕ) (boxes_last_week : ℕ) (boxes_this_week : ℕ) :
  pomelos_last_week = 240 → boxes_last_week = 10 → boxes_this_week = 20 →
  (pomelos_last_week / boxes_last_week) * boxes_this_week / 12 = 40 := 
by
  intro h1 h2 h3
  sorry

end farmer_pomelos_dozen_l216_216002


namespace PQ_sequence_eval_l216_216009

def P (x : ℝ) : ℝ := 2 * real.sqrt x
def Q (x : ℝ) : ℝ := x^3

theorem PQ_sequence_eval : P (Q (P (Q (P (Q 2))))) = 128 * real.sqrt (2 ^ (3 / 8)) := by
  sorry

end PQ_sequence_eval_l216_216009


namespace monotonicity_of_f_ax_x1_increasing_interval_log_l216_216927

theorem monotonicity_of_f_ax_x1 {a : ℝ} (h₀ : a ≠ 0) :
  (∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → (if a > 0 then f x1 > f x2 else f x1 < f x2)) :=
  by { sorry }

theorem increasing_interval_log {x : ℝ} :
  (x < -4 → f x = log (1/3) (x^2 - 16) → f x is strictly_increasing) :=
  by { sorry }

end monotonicity_of_f_ax_x1_increasing_interval_log_l216_216927


namespace total_letters_l216_216386

theorem total_letters (brother_letters : ℕ) (greta_more_than_brother : ℕ) (mother_multiple : ℕ)
  (h_brother : brother_letters = 40)
  (h_greta : ∀ (brother_letters greta_letters : ℕ), greta_letters = brother_letters + greta_more_than_brother)
  (h_mother : ∀ (total_letters mother_letters : ℕ), mother_letters = mother_multiple * total_letters) :
  brother_letters + (brother_letters + greta_more_than_brother) + (mother_multiple * (brother_letters + (brother_letters + greta_more_than_brother))) = 270 :=
by
  sorry

end total_letters_l216_216386


namespace pappus_theorem_l216_216118

open ProjectiveGeometry

variables {P : Type*} [projective_plane P]

-- Definitions of points and lines
variables (A B C D E F G H I : P)
variables (l m : Line P)

-- Conditions
axiom collinear_ABC : collinear ({A, B, C} : set P)
axiom collinear_DEF : collinear ({D, E, F} : set P)
axiom intersection_AE_BD : incidence ({A, E} : set P) ({B, D} : set P) G
axiom intersection_AF_CD : incidence ({A, F} : set P) ({C, D} : set P) H
axiom intersection_BF_CE : incidence ({B, F} : set P) ({C, E} : set P) I

-- Conjecture to prove
theorem pappus_theorem : collinear ({G, H, I} : set P) := 
sorry

end pappus_theorem_l216_216118


namespace vicki_donated_fraction_l216_216783

/-- Given Jeff had 300 pencils and donated 30% of them, and Vicki had twice as many pencils as Jeff originally 
    had, and there are 360 pencils remaining altogether after both donations,
    prove that Vicki donated 3/4 of her pencils. -/
theorem vicki_donated_fraction : 
  let jeff_pencils := 300
  let jeff_donated := jeff_pencils * 0.30
  let jeff_remaining := jeff_pencils - jeff_donated
  let vicki_pencils := 2 * jeff_pencils
  let total_remaining := 360
  let vicki_remaining := total_remaining - jeff_remaining
  let vicki_donated := vicki_pencils - vicki_remaining
  vicki_donated / vicki_pencils = 3 / 4 :=
by
  -- Proof needs to be inserted here
  sorry

end vicki_donated_fraction_l216_216783


namespace find_s_l216_216588

-- Define the conditions of the problem
def has_adjacent_sides (s : ℝ) (a b : ℝ) (θ : ℝ) : Prop :=
  a = 3 * s ∧ b = 4 * s ∧ θ = (π / 6)

def has_area (a b : ℝ) (θ area : ℝ) : Prop :=
  area = a * b * real.sin θ / 2

-- Define the main theorem
theorem find_s (s : ℝ) :
  ∃ s, (has_adjacent_sides s (3 * s) (4 * s) (π / 6)) ∧ has_area (3 * s) (4 * s) (π / 6) (18 * real.sqrt 3) :=
  ∃ s, s = real.log (3^3/4) :=
begin
  sorry,
end

end find_s_l216_216588


namespace total_letters_received_l216_216382

theorem total_letters_received :
  ∀ (g b m t : ℕ), 
    b = 40 →
    g = b + 10 →
    m = 2 * (g + b) →
    t = g + b + m → 
    t = 270 :=
by
  intros g b m t hb hg hm ht
  rw [hb, hg, hm, ht]
  sorry

end total_letters_received_l216_216382


namespace find_length_of_first_train_l216_216605

noncomputable def length_of_first_train (speed_train1 speed_train2 : ℕ) (time_to_cross : ℕ) (length_train2 : ℚ) : ℚ :=
  let relative_speed := (speed_train1 + speed_train2) * 1000 / 3600
  let combined_length := relative_speed * time_to_cross
  combined_length - length_train2

theorem find_length_of_first_train :
  length_of_first_train 120 80 9 280.04 = 220 := sorry

end find_length_of_first_train_l216_216605


namespace escalator_ride_time_l216_216441

theorem escalator_ride_time (x y k t : ℝ)
  (h1 : 75 * x = y)
  (h2 : 30 * (x + k) = y)
  (h3 : t = y / k) :
  t = 50 := by
  sorry

end escalator_ride_time_l216_216441


namespace average_speed_l216_216923

def dist1 : ℝ := 60
def dist2 : ℝ := 30
def time : ℝ := 2

theorem average_speed : (dist1 + dist2) / time = 45 := by
  sorry

end average_speed_l216_216923


namespace cristina_speed_cristina_running_speed_l216_216472

theorem cristina_speed 
  (head_start : ℕ)
  (nicky_speed : ℕ)
  (catch_up_time : ℕ)
  (distance : ℕ := head_start + (nicky_speed * catch_up_time))
  : distance / catch_up_time = 6
  := by
  sorry

-- Given conditions used as definitions in Lean 4:
-- head_start = 36 (meters)
-- nicky_speed = 3 (meters/second)
-- catch_up_time = 12 (seconds)

theorem cristina_running_speed
  (head_start : ℕ := 36)
  (nicky_speed : ℕ := 3)
  (catch_up_time : ℕ := 12)
  : (head_start + (nicky_speed * catch_up_time)) / catch_up_time = 6
  := by
  sorry

end cristina_speed_cristina_running_speed_l216_216472


namespace minimum_cos_diff_l216_216574

open Real

theorem minimum_cos_diff (x y z : ℝ) 
  (h1 : sqrt 3 * sin x = tan y)
  (h2 : 2 * sin y = cot z)
  (h3 : sin z = 2 * tan x) : 
  ∃ w, (cos x - cos z = w) ∧ (w = -7 * sqrt 2 / 6) := 
sorry

end minimum_cos_diff_l216_216574


namespace minimum_value_of_expression_l216_216308

theorem minimum_value_of_expression : ∃ x : ℝ, (16 ^ x - 4 ^ x + 1 = 3 / 4) :=
sorry

end minimum_value_of_expression_l216_216308


namespace part_I_part_II_l216_216817

-- Proving part (I): f(x) + f(-1/x) >= 2 given f(x) = |x - a| and a < 0
theorem part_I (a : ℝ) (h : a < 0) (x : ℝ) : |x - a| + |(-1/x) - a| >= 2 := sorry

-- Proving part (II): the range of a making the solution set {x | f(x) + f(2x) < 1/2} non-empty is (-1, 0)
theorem part_II (a : ℝ) :
  (∃ x : ℝ, |x - a| + |(2 * x) - a| < 1/2) → a ∈ Ioo (-1 : ℝ) (0 : ℝ) := sorry

end part_I_part_II_l216_216817


namespace coefficient_of_x7_in_expansion_l216_216550

open BigOperators

def binomial {n k : ℕ} : ℕ := Nat.choose n k

def expansion_coefficient : ℕ := 
  (binomial 10 7) * (3^7) * (2^3)

theorem coefficient_of_x7_in_expansion :
  (2099520 : ℕ) = expansion_coefficient := by
  sorry

end coefficient_of_x7_in_expansion_l216_216550


namespace street_length_in_meters_l216_216593

noncomputable def time : ℝ := 4 -- in minutes
noncomputable def speed_kmh : ℝ := 4.5 -- in km/h

/-- Convert speed from km/h to m/min -/
def speed_m_per_min (speed_kmh: ℝ) : ℝ := (speed_kmh * 1000) / 60

def distance (speed_m_per_min: ℝ) (time: ℝ) : ℝ := speed_m_per_min * time

theorem street_length_in_meters : distance (speed_m_per_min speed_kmh) time = 300 :=
by
  sorry

end street_length_in_meters_l216_216593


namespace range_of_f_l216_216700

-- Define the given function
def f (x : ℝ) (k : ℝ) (c : ℝ) : ℝ := x^k + c

-- State the properties of k and c
variables (k c : ℝ)
variable (h_k_pos : k > 0)
variable (h_c_pos : c > 0)

-- Define the interval [1, ∞)
def interval : Set ℝ := {x : ℝ | 1 ≤ x}

-- Define the expected range
def expected_range : Set ℝ := {y : ℝ | 1 + c ≤ y }

-- State the main theorem
theorem range_of_f : 
  ∀ y, y ∈ Set.image (fun x => f x k c) interval ↔ y ∈ expected_range :=
by
  sorry

end range_of_f_l216_216700


namespace vertex_of_parabola_find_shift_m_l216_216708

-- Problem 1: Vertex of the given parabola
theorem vertex_of_parabola : 
  ∃ x y: ℝ, (y = 2 * x^2 + 4 * x - 6) ∧ (x, y) = (-1, -8) := 
by
  -- Proof goes here
  sorry

-- Problem 2: Finding the shift m
theorem find_shift_m (m : ℝ) (h : m > 0) : 
  (∀ x (hx : (x = (x + m)) ∧ (2 * x^2 + 4 * x - 6 = 0)), x = 1 ∨ x = -3) ∧ 
  ((-3 + m) = 0) → m = 3 :=
by
  -- Proof goes here
  sorry

end vertex_of_parabola_find_shift_m_l216_216708


namespace smallest_integer_square_l216_216910

theorem smallest_integer_square (x : ℤ) (h : x^2 = 2 * x + 75) : x = -7 :=
  sorry

end smallest_integer_square_l216_216910


namespace rectangle_cos_angle_l216_216070

theorem rectangle_cos_angle {A B C D O : Point} (h_rect : rectangle A B C D)
  (h_diag : quadrilateral_diagonals_intersect A B C D O)
  (h_AB : dist A B = 15)
  (h_BC : dist B C = 20) :
  cos (angle A O B) = 7 / 25 :=
sorry

end rectangle_cos_angle_l216_216070


namespace find_a_l216_216815

noncomputable def z1 (a : ℝ) : ℂ := complex.of_real 2 + complex.i * a
def z2 : ℂ := complex.of_real 2 - complex.i

theorem find_a (a : ℝ) (h1 : 0 < a)
  (h2 : complex.abs (z1 a) = complex.abs z2) : a = 1 :=
sorry

end find_a_l216_216815


namespace concyclic_intersection_points_and_common_radical_axis_l216_216848

theorem concyclic_intersection_points_and_common_radical_axis 
  (C1 C2 L1 L2 : ℝ → ℝ → Prop) 
  (h1 : ∃ P1 P2 P3 P4, C1 P1 ∧ C1 P2 ∧ C1 P3 ∧ C1 P4 ∧ L1 P1 ∧ L2 P2 ∧ L1 P3 ∧ L2 P4)
  (h2 : ∃ Q1 Q2 Q3 Q4, C2 Q1 ∧ C2 Q2 ∧ C2 Q3 ∧ C2 Q4 ∧ L1 Q1 ∧ L2 Q2 ∧ L1 Q3 ∧ L2 Q4) :
  ∃ R1 R2 R3 R4, (∀ x y, C1 x y ∧ C2 x y → R1 x y ∨ R2 x y ∨ R3 x y ∨ R4 x y) ∧
  -- Additional statement to prove the common radical axis
  ∃ O : ℝ → ℝ → Prop, (∀ x y, C1 x y ∨ C2 x y → O x y) :=
sorry

end concyclic_intersection_points_and_common_radical_axis_l216_216848


namespace has_exactly_one_zero_of_function_l216_216724

theorem has_exactly_one_zero_of_function (f : ℝ → ℝ) (h_def : ∀ x, f x = Real.exp x + 3 * x) : 
  ∃! x, f x = 0 :=
begin
  sorry
end

end has_exactly_one_zero_of_function_l216_216724


namespace evaluate_three_squared_raised_four_l216_216280

theorem evaluate_three_squared_raised_four : (3^2)^4 = 6561 := by
  sorry

end evaluate_three_squared_raised_four_l216_216280


namespace each_sibling_receives_two_small_cups_l216_216996

def tea_distribution_problem 
  (total_large_cups : ℕ)
  (total_small_cups : ℕ)
  (large_to_small_ratio : ℕ)
  (parent_percentage : ℚ)
  (grandparent_percentage : ℚ)
  (number_of_siblings : ℕ) 
  : ℕ :=
let S := 1 in
let L := large_to_small_ratio * S in
let total_tea := (total_large_cups * L) + (total_small_cups * S) in
let parents_tea := parent_percentage * total_tea in
let grandparents_tea := grandparent_percentage * total_tea in
let siblings_tea := total_tea - parents_tea - grandparents_tea in
let tea_per_sibling := siblings_tea / number_of_siblings in
tea_per_sibling / S

theorem each_sibling_receives_two_small_cups :
  tea_distribution_problem 5 10 3 (30/100) (25/100) 4 = 2 :=
by sorry

end each_sibling_receives_two_small_cups_l216_216996


namespace price_of_bread_l216_216934

variable (B : ℝ)
variable (usual_pastries_sold : ℝ := 20)
variable (usual_bread_sold : ℝ := 10)
variable (today_pastries_sold : ℝ := 14)
variable (today_bread_sold : ℝ := 25)
variable (pastry_price : ℝ := 2)
variable (sales_difference : ℝ := 48)

theorem price_of_bread : 
  let usual_sales := usual_pastries_sold * pastry_price + usual_bread_sold * B in
  let today_sales := today_pastries_sold * pastry_price + today_bread_sold * B in
  today_sales - usual_sales = sales_difference → B = 4 :=
sorry

end price_of_bread_l216_216934


namespace min_jellybeans_l216_216231

theorem min_jellybeans (n : ℕ) (h1 : n ≥ 150) (h2 : n % 17 = 15) : n = 151 :=
by { sorry }

end min_jellybeans_l216_216231


namespace count_integers_satisfy_inequality_l216_216718

theorem count_integers_satisfy_inequality :
  {m : ℤ | m ≠ 0 ∧ (1 : ℚ) / ↑|m| ≥ (1 : ℚ) / 12}.finite
    ∧ {m : ℤ | m ≠ 0 ∧ (1 : ℚ) / ↑|m| ≥ (1 : ℚ) / 12}.card = 24 :=
by
  sorry

end count_integers_satisfy_inequality_l216_216718


namespace bill_pays_sales_tax_correct_l216_216993

def take_home_salary : ℝ := 40000
def property_tax : ℝ := 2000
def gross_salary : ℝ := 50000
def income_tax (gs : ℝ) : ℝ := 0.10 * gs
def total_taxes_paid (gs th : ℝ) : ℝ := gs - th
def sales_tax (ttp it pt : ℝ) : ℝ := ttp - it - pt

theorem bill_pays_sales_tax_correct :
  sales_tax
    (total_taxes_paid gross_salary take_home_salary)
    (income_tax gross_salary)
    property_tax = 3000 :=
by sorry

end bill_pays_sales_tax_correct_l216_216993


namespace evaluate_exponent_l216_216279

theorem evaluate_exponent : (3^2)^4 = 6561 := sorry

end evaluate_exponent_l216_216279


namespace find_lighter_balls_l216_216099

-- Definition of ball weights and numbering
structure Ball :=
(to_nat : ℕ)
(weight : ℕ)

-- Assumptions:
-- There are eight balls numbered from 1 to 8
def balls : List Ball :=
  [Ball.mk 1 2, Ball.mk 2 2, Ball.mk 3 1, Ball.mk 4 1, Ball.mk 5 1,
   Ball.mk 6 2, Ball.mk 7 2, Ball.mk 8 2]

-- Six balls are of the same weight (2 grams), and two are 1 gram lighter
def same_weight : ℕ := 2
def light_weight : ℕ := 1

-- The balance scale results are given as follows:
-- Balls 1 and 2 are heavier than balls 3 and 4
axiom heavier_1_2_3_4 : (balls.nth 0).get.weight + (balls.nth 1).get.weight >
                        (balls.nth 2).get.weight + (balls.nth 3).get.weight
  
-- Balls 5 and 6 are lighter than balls 7 and 8
axiom lighter_5_6_7_8 : (balls.nth 4).get.weight + (balls.nth 5).get.weight <
                        (balls.nth 6).get.weight + (balls.nth 7).get.weight

-- The combined weight of balls 1, 3, and 5 is equal to the combined weight of balls 2, 4, and 8
axiom equal_1_3_5_2_4_8 : (balls.nth 0).get.weight + (balls.nth 2).get.weight + (balls.nth 4).get.weight =
                           (balls.nth 1).get.weight + (balls.nth 3).get.weight + (balls.nth 7).get.weight

-- We'll now create a theorem that encapsulates the problem and expected conclusion
theorem find_lighter_balls
  (balls_condition : balls.length = 8)
  (all_same_weight : ∀ b ∈ balls, b.weight = same_weight ∨ b.weight = light_weight)
  (two_light_weight_exists : ∃ x y ∈ balls, x.weight = light_weight ∧ y.weight = light_weight ∧ x ≠ y) :
  ∃ b1 b2 : Ball, b1.weight = light_weight ∧ b2.weight = light_weight ∧ b1.to_nat = 4 ∧ b2.to_nat = 5 :=
by
  sorry

#check find_lighter_balls

end find_lighter_balls_l216_216099


namespace triangle_is_right_l216_216778

variables {a b c : ℝ} {A B C : ℝ}

-- Defining the sides in terms of the angles
def side_opposite_A := a
def side_opposite_B := b
def side_opposite_C := c

-- Given condition
def condition_sine_half_angle : Prop := 
  sin (A / 2) ^ 2 = (c - b) / (2 * c)

-- The proof goal: Triangle ABC is a right triangle
theorem triangle_is_right
  (h₁ : side_opposite_A = a)
  (h₂ : side_opposite_B = b)
  (h₃ : side_opposite_C = c)
  (h₄ : condition_sine_half_angle) : a^2 + b^2 = c^2 :=
sorry

end triangle_is_right_l216_216778


namespace expected_value_xi_l216_216433

-- Define the curves
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := y = √3 * x ∧ x ≥ 0
def C₃ (x y : ℝ) : Prop := (x - 1/2)^2 + y^2 = 1/4
def C₄ (x y : ℝ) : Prop := y = 1

-- Number of intersection points definition
def intersection_points (C1 C2 : ℝ → ℝ → Prop) : ℕ := sorry

-- Define the random variable ξ
def ξ (C1 C2 : ℝ → ℝ → Prop) : ℕ :=
  intersection_points C1 C2

-- Given the conditions to calculate Eξ
axiom C1_C2_intersections : intersection_points C₁ C₂ = 1
axiom C1_C3_intersections : intersection_points C₁ C₃ = 1
axiom C1_C4_intersections : intersection_points C₁ C₄ = 1
axiom C2_C3_intersections : intersection_points C₂ C₃ = 2
axiom C2_C4_intersections : intersection_points C₂ C₄ = 1
axiom C3_C4_intersections : intersection_points C₃ C₄ = 0

-- Probabilities of intersection points
def P_xi (n : ℕ) : ℝ :=
  if n = 0 then 1/6 else if n = 1 then 2/3 else if n = 2 then 1/6 else 0

-- Expected value of ξ
noncomputable def Eξ : ℝ := ∑ n in Finset.range 3, n * P_xi n

-- The proof problem 
theorem expected_value_xi : Eξ = 1 :=
  sorry

end expected_value_xi_l216_216433


namespace trapezoid_construction_possible_l216_216898

-- Definitions of variables representing the side lengths
variables {d b a c : ℝ}
-- Conditions: d > b, d, b, a, c must all be positive real numbers
def valid_sides (d b a c : ℝ) : Prop := d > b ∧ d > 0 ∧ b > 0 ∧ a > 0 ∧ c > 0

noncomputable def construct_trapezoid (d b a c : ℝ) 
  (h_valid : valid_sides d b a c) : Prop :=
∃ A B C D: ℝ × ℝ, 
  -- A, B, C, D are points in the Euclidean plane
  (dist A B = a) ∧ 
  (dist C D = c) ∧
  (dist A D = d) ∧
  (dist B C = b) ∧
  -- AD and BC are the bases, and AB and CD are the lateral sides
  -- also make sure AD is parallel to BC
  let v_AD := (D.1 - A.1, D.2 - A.2),
      v_BC := (C.1 - B.1, C.2 - B.2) in
  v_AD.2 * v_BC.1 = v_AD.1 * v_BC.2 ∘ sorry

-- The theorem statement
theorem trapezoid_construction_possible (d b a c : ℝ) :
  valid_sides d b a c → construct_trapezoid d b a c :=
by sorry

end trapezoid_construction_possible_l216_216898


namespace number_of_classes_l216_216160

theorem number_of_classes (x : ℕ) (total_games : ℕ) (h : total_games = 45) :
  (x * (x - 1)) / 2 = total_games → x = 10 :=
by
  sorry

end number_of_classes_l216_216160


namespace smallest_d_inverse_g_has_inverse_l216_216463

noncomputable def g (x : ℝ) : ℝ := (x - 3)^2 + 6

theorem smallest_d_inverse (d : ℝ) (h1 : d ≥ 3) : 
  ∀ x1 x2 : ℝ, (x1 ∈ set.Ici d ∧ x2 ∈ set.Ici d) → g x1 = g x2 → x1 = x2 := 
  sorry

theorem g_has_inverse : (∃ d : ℝ, d = 3 ∧ (∀ x1 x2 : ℝ, (x1 ∈ set.Ici d ∧ x2 ∈ set.Ici d) → g x1 = g x2 → x1 = x2)) :=
  by
    use 3
    split
    -- Prove d = 3
    refl
    -- Prove ∀ x1 x2 : ℝ, (x1 ∈ set.Ici 3 ∧ x2 ∈ set.Ici 3) → g x1 = g x2 → x1 = x2
    exact smallest_d_inverse 3 (by norm_num)

end smallest_d_inverse_g_has_inverse_l216_216463


namespace legs_in_pool_proof_l216_216786

def total_people (adults₁ children₁ adults₂ children₂ : ℕ) : ℕ :=
adults₁ + children₁ + adults₂ + children₂

def people_in_pool (total not_in_pool : ℕ) : ℕ :=
total - not_in_pool

def legs_in_pool (people legs_per_person : ℕ) : ℕ :=
people * legs_per_person

theorem legs_in_pool_proof : 
  let 
    k_and_d_adults := 2 -- Karen and Donald
    k_and_d_children := 6
    t_and_e_adults := 2 -- Tom and Eva
    t_and_e_children := 4
    not_in_pool := 6
    legs_per_person := 2

    total := total_people k_and_d_adults k_and_d_children t_and_e_adults t_and_e_children
    in_pool := people_in_pool total not_in_pool
  in
  legs_in_pool in_pool legs_per_person = 16 :=
  by
    sorry

end legs_in_pool_proof_l216_216786


namespace calculate_expression_l216_216623

theorem calculate_expression :
  (16^16 * 8^8) / 4^32 = 16777216 := by
  sorry

end calculate_expression_l216_216623


namespace part_I_part_II_l216_216807

noncomputable def parabola_eq (x y : ℝ) : Prop := x^2 = 4 * y
def is_focus (F : ℝ × ℝ) : Prop := F = (0, 1)
def is_midpoint (N A B : ℝ × ℝ) : Prop := N = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Theorem for part (I)
theorem part_I 
  (A B C F N : ℝ × ℝ)
  (h_parabola_A : parabola_eq A.1 A.2)
  (h_parabola_B : parabola_eq B.1 B.2)
  (h_parabola_C : parabola_eq C.1 C.2)
  (h_focus_F : is_focus F)
  (h_midpoint_N : is_midpoint N A B)
  (h_CF_2FN : ∃ (λ : ℝ), λ = 2 ∧ (C.1 - F.1, C.2 - F.2) = (λ * (N.1 - F.1), λ * (N.2 - F.2))) :
  |(F.1 - A.1)^2 + (F.2 - A.2)^2|.sqrt + |(F.1 - B.1)^2 + (F.2 - B.2)^2|.sqrt + |(F.1 - C.1)^2 + (F.2 - C.2)^2|.sqrt = 6 := 
  sorry

-- Theorem for part (II)
theorem part_II
  (A B C F N : ℝ × ℝ)
  (h_parabola_A : parabola_eq A.1 A.2)
  (h_parabola_B : parabola_eq B.1 B.2)
  (h_parabola_C : parabola_eq C.1 C.2)
  (h_focus_F : is_focus F)
  (h_midpoint_N : is_midpoint N A B)
  (h_CF_3FN : ∃ (λ : ℝ), λ = 3 ∧ (C.1 - F.1, C.2 - F.2) = (λ * (N.1 - F.1), λ * (N.2 - F.2))) :
  ∃ (k b : ℝ), - κ <b, b ≤κ,
  (is_max_area : ∀ S : ℝ, S = (1/2) * |((4λ * (((4+12b)/15)+b)*(-1+b)).sqrt)| * |(λ := 4√(S)), -- Definition of maximum area of △ABC
    S = 64 * (5.sqrt / 135) :=
  sorry

end part_I_part_II_l216_216807


namespace find_f3_l216_216699

theorem find_f3 (f : ℝ → ℝ) : (∀ x : ℝ, f (2 * x - 1) = 4 * x^2) → f 3 = 16 :=
by
  intro h
  have h2 : f (2 * 2 - 1) = 4 * 2^2, from h 2
  simp [f] at h2
  exact h2

end find_f3_l216_216699


namespace obtuse_triangle_count_l216_216091

def is_obtuse_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 < c^2

def valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem obtuse_triangle_count :
  {k : ℕ | valid_triangle 7 15 k ∧ is_obtuse_triangle 7 15 k}.card +
  {k : ℕ | valid_triangle 7 15 k ∧ is_obtuse_triangle 7 k 15}.card = 10 :=
sorry

end obtuse_triangle_count_l216_216091


namespace estimated_population_correct_correlation_coefficient_correct_l216_216203

noncomputable def estimated_population
  (num_plots : ℕ) (sample_size : ℕ) 
  (sum_y : ℕ) : ℕ :=
(sum_y / sample_size) * num_plots

noncomputable def correlation_coefficient
  (sum_x_sq : ℝ) (sum_y_sq : ℝ) 
  (sum_xy : ℝ) : ℝ :=
sum_xy / real.sqrt(sum_x_sq * sum_y_sq)

theorem estimated_population_correct
  (num_plots : ℕ := 200) (sample_size : ℕ := 20) 
  (sum_y : ℕ := 1200) : estimated_population num_plots sample_size sum_y = 12000 := by
sorry

theorem correlation_coefficient_correct
  (sum_x_sq : ℝ := 80) (sum_y_sq : ℝ := 9000) 
  (sum_xy : ℝ := 800) : real.abs (correlation_coefficient sum_x_sq sum_y_sq sum_xy - 0.94) < 0.01 := by
sorry

end estimated_population_correct_correlation_coefficient_correct_l216_216203


namespace number_of_correct_propositions_l216_216661

theorem number_of_correct_propositions (f : ℝ → ℝ)
  (odd_f : ∀ x, f (-x) = -f x)
  (symmetric_f1_line : ∀ x, f(x-1) = f(1-x))
  (periodic_f2 : ∀ x, f(x-1) = -f x) : 
  (∃ A : ℕ, 1 ≤ A ∧ A = 4) := 
by
  sorry

end number_of_correct_propositions_l216_216661


namespace alex_silver_tokens_count_l216_216233

-- Conditions
def initial_red_tokens := 90
def initial_blue_tokens := 80

def red_exchange (x : ℕ) (y : ℕ) : ℕ := 90 - 3 * x + y
def blue_exchange (x : ℕ) (y : ℕ) : ℕ := 80 + 2 * x - 4 * y

-- Boundaries where exchanges stop
def red_bound (x : ℕ) (y : ℕ) : Prop := red_exchange x y < 3
def blue_bound (x : ℕ) (y : ℕ) : Prop := blue_exchange x y < 4

-- Proof statement
theorem alex_silver_tokens_count (x y : ℕ) :
    red_bound x y → blue_bound x y → (x + y) = 52 :=
    by
    sorry

end alex_silver_tokens_count_l216_216233


namespace stock_price_end_of_second_year_l216_216269

def initial_price : ℝ := 80
def first_year_increase_rate : ℝ := 1.2
def second_year_decrease_rate : ℝ := 0.3

theorem stock_price_end_of_second_year : 
  initial_price * (1 + first_year_increase_rate) * (1 - second_year_decrease_rate) = 123.2 := 
by sorry

end stock_price_end_of_second_year_l216_216269


namespace isosceles_triangle_perimeter_l216_216742

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 3 ∧ b = 5) (h2 : Isosceles a b) :
  (a + b + b = 13) ∨ (b + a + a = 11) :=
by
  cases h1
  simp at h1_left h1_right
  sorry

end isosceles_triangle_perimeter_l216_216742


namespace count_squares_within_region_l216_216720

theorem count_squares_within_region : 
  let π := Real.pi in
  let bound_x := 5.1 in
  let bound_y_neg := -0.1 in
  let square_count := ∑ x in finset.range (bound_x.floor + 1),
    finset.range (x.floor * π).floor) - (x+1)
  square_count = 30 :=
by
  sorry

end count_squares_within_region_l216_216720


namespace arithmetic_sequence_a12_bound_l216_216769

theorem arithmetic_sequence_a12_bound (a_1 d : ℤ) (h8 : a_1 + 7 * d ≥ 15) (h9 : a_1 + 8 * d ≤ 13) : 
  a_1 + 11 * d ≤ 7 :=
by
  sorry

end arithmetic_sequence_a12_bound_l216_216769


namespace real_solution_to_cubic_eq_l216_216263

noncomputable section

open Real

theorem real_solution_to_cubic_eq (x : ℝ) : 
  x^3 + 2 * (x + 1)^3 + 3 * (x + 2)^3 = 3 * (x + 3)^3 → x = 5 := 
by
  sorry

end real_solution_to_cubic_eq_l216_216263


namespace max_profit_l216_216761

/-- Define the cost and price of device A and device B -/
def cost_A : ℝ := 3
def price_A : ℝ := 3.3
def cost_B : ℝ := 2.4
def price_B : ℝ := 2.8

/-- Define the total number of devices -/
def total_devices : ℝ := 50

/-- Define the profits per device -/
def profit_per_A : ℝ := price_A - cost_A -- 0.3
def profit_per_B : ℝ := price_B - cost_B -- 0.4

/-- Define the function for total profit -/
def total_profit (x : ℝ) : ℝ :=
  profit_per_A * x + profit_per_B * (total_devices - x)

/-- Define the constraint -/
def constraint (x : ℝ) : Prop := 4 * x ≥ total_devices - x -- x ≥ 10

/-- The statement of the problem that needs to be proven -/
theorem max_profit :
  (total_profit x = -0.1 * x + 20) ∧ 
  ( ∀ x, constraint x → x ≥ 10 → x = 10 ∧ total_profit x = 19) :=
by
  sorry

end max_profit_l216_216761


namespace min_value_16x_minus_4x_plus_1_l216_216302

theorem min_value_16x_minus_4x_plus_1 :
  ∀ x : ℝ, 16 ^ x - 4 ^ x + 1 ≥ 3 / 4 ∧ (∃ y : ℝ, 16 ^ y - 4 ^ y + 1 = 3 / 4) :=
by
  sorry

end min_value_16x_minus_4x_plus_1_l216_216302


namespace compound_interest_final_amount_l216_216142

theorem compound_interest_final_amount
  (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) :
  P = 140 → r = 0.20 → n = 2 → t = 1 →
  let A := P * (1 + r / n) ^ (n * t)
  in A = 169.40 :=
by
  intros hP hr hn ht A_def
  subst_vars
  have A := P * (1 + r / n.toReal) ^ (n * t)
  sorry

end compound_interest_final_amount_l216_216142


namespace andy_profit_per_cake_l216_216987

-- Definitions based on the conditions
def cost_of_ingredients (cakes : ℕ) : ℕ := if cakes = 2 then 12 else 0
def cost_of_packaging_per_cake : ℕ := 1
def selling_price_per_cake : ℕ := 15

-- Theorem stating the profit made per cake
theorem andy_profit_per_cake : ∀ (cakes : ℕ), cakes = 2 → 
(cost_of_ingredients cakes / cakes + cost_of_packaging_per_cake) = 7 →
selling_price_per_cake - (cost_of_ingredients cakes / cakes + cost_of_packaging_per_cake) = 8 :=
by
  intros cakes h_cakes cost_hyp
  have h1 : cost_of_ingredients cakes / cakes = 12 / 2 :=
    by rw [h_cakes]; refl
  have h2 : (12 / 2 + cost_of_packaging_per_cake) = 6 + 1 :=
    by rw [h1]; refl
  have h3 : (6 + 1) = 7 :=
    by refl
  rw [← h3] at cost_hyp
  have h4 : selling_price_per_cake - 7 = 8 :=
    by refl
  exact h4

end andy_profit_per_cake_l216_216987


namespace time_to_cross_signal_pole_l216_216580

/-
  Given:
  - The length of the train (train_length) is 200 meters.
  - The time taken to cross a platform (platform_time) is 50 seconds.
  - The length of the platform (platform_length) is 38.0952380952381 meters.
  Prove:
  - The time it takes for the train to cross the signal pole is approximately 42 seconds.
-/
theorem time_to_cross_signal_pole : 
  let train_length := 200 : ℝ
  let platform_time := 50 : ℝ
  let platform_length := 38.0952380952381 : ℝ
  let total_distance := train_length + platform_length
  let speed := total_distance / platform_time
  let time_to_cross_signal_pole := train_length / speed
  abs (time_to_cross_signal_pole - 42) < 1 :=
by
  sorry

end time_to_cross_signal_pole_l216_216580


namespace find_larger_number_l216_216109

theorem find_larger_number (a b : ℤ) (h1 : a + b = 27) (h2 : a - b = 5) : a = 16 := by
  sorry

end find_larger_number_l216_216109


namespace poker_flush_probability_l216_216901

theorem poker_flush_probability :
  let total_ways := Nat.choose 52 5
  let flush_ways := 4 * Nat.choose 13 5
  (flush_ways : ℚ) / total_ways = 103 / 51980 :=
by
  sorry

end poker_flush_probability_l216_216901


namespace tiles_needed_l216_216000

-- Define the conditions
def tile_area : ℝ := 7.5
def kitchen_length : ℝ := 52
def kitchen_width : ℝ := 79
def kitchen_area : ℝ := kitchen_length * kitchen_width
def num_tiles_needed : ℝ := kitchen_area / tile_area

-- Statement proving the number of tiles needed
theorem tiles_needed (t_area : ℝ) (k_length : ℝ) (k_width : ℝ) (k_area : ℝ) (n_tiles : ℝ) :
  t_area = 7.5 → k_length = 52 → k_width = 79 → k_area = (k_length * k_width) →
  n_tiles = k_area / t_area →
  nat.ceil n_tiles = 548 :=
by
  intro ht hk1 hk2 hk3 hn
  sorry

end tiles_needed_l216_216000


namespace polynomial_value_at_neg2_l216_216016

noncomputable def P (x : ℝ) : ℝ :=
  x^2 + (1/24) * (x-1) * (x-2) * (x-3) * (x-4)

theorem polynomial_value_at_neg2 :
  P(0) = 1 →
  P(1) = 1 →
  P(2) = 4 →
  P(3) = 9 →
  P(4) = 16 →
  P(-2) = 19 :=
by
  intros h0 h1 h2 h3 h4
  rw [P] at *
  -- rest of the proof would follow, but it's skipped here
  sorry

end polynomial_value_at_neg2_l216_216016


namespace point_in_fourth_quadrant_l216_216325

-- Let x and y be real numbers.
variables (x y : ℝ)

-- Define the condition in the problem.
def condition : Prop := (x - 2) ^ 2 + sqrt (y + 1) = 0

-- Proving the point (x, y) lies in the fourth quadrant.
theorem point_in_fourth_quadrant (h : condition x y) : x = 2 ∧ y = -1 ∧ x > 0 ∧ y < 0 :=
by {
  sorry
}

end point_in_fourth_quadrant_l216_216325


namespace parabola_intersects_and_area_triangle_l216_216780

theorem parabola_intersects_and_area_triangle :
  let C := λ x y : ℝ, y^2 = 8 * x
  let l1 := λ y : ℝ, y = -x
  let l2 := λ x y : ℝ, x = y + 8
  let F := (8, -8)
  ∃ (p : ℝ), p > 0 ∧ C 8 (-8) ∧ 
    ∀ (A B : ℝ × ℝ), 
    let P := ((fst A + fst B) / 2, (snd A + snd B) / 2)
    ∃ (m : ℝ), m > -2 ∧ l2 (fst A) (snd A) ∧ l2 (fst B) (snd B) ->
    P • P = (A - B) / 2 ->
    area_of_triangle F A B = 24 * real.sqrt 5 := by
begin
  sorry
end

end parabola_intersects_and_area_triangle_l216_216780


namespace matrix_problem_l216_216459

variable {A B : Matrix (Fin 2) (Fin 2) ℝ}

theorem matrix_problem
  (h1 : A + B = A ⬝ B)
  (h2 : A ⬝ B = ![![5, 2], ![-2, 3]]) :
  B ⬝ A = ![![5, 2], ![-2, 3]] := 
sorry

end matrix_problem_l216_216459


namespace angle_DKY_eq_90_l216_216408

theorem angle_DKY_eq_90 (X Y Z D K : Point)
  (hX : X ≠ Y) (hY : Y ≠ Z) (hZ : Z ≠ X)
  (angle_XYZ : angle X Y Z = 90)
  (angle_YZX : angle Y Z X = 60)
  (angle_ZXY : angle Z X Y = 30)
  (bisect_YD : angle Y D Z = angle Y D X / 2)
  (perp_bisect_XK : angle X K Y = 90 ∧ distance K Y = distance K Z) :
  angle D K Y = 90 := sorry

end angle_DKY_eq_90_l216_216408


namespace circumcircle_radius_l216_216405

theorem circumcircle_radius (a : ℝ) (A : ℝ) (R : ℝ) (h1 : a = 3) (h2 : cos A = -1 / 2) : R = real.sqrt 3 :=
by
  -- Proof required here
  sorry

end circumcircle_radius_l216_216405


namespace sphere_cylinder_properties_sphere_cylinder_properties_radius_3_l216_216615

theorem sphere_cylinder_properties (r : ℝ) (h : 2 * r = r * 2 ∧ 2 * r = r * 2) :
  (volume (Sphere r) = 4/3 * pi * (r ^ 3)) ∧ (surface_area (Sphere r) = 4 * pi * (r ^ 2)) ∧ 
  (surface_area (Cylinder r (2 * r)) = 4 * pi * (r ^ 2)) :=
by
  sorry

-- Specific case for r = 3
theorem sphere_cylinder_properties_radius_3 :
  (volume (Sphere 3) = 36 * pi) ∧ (surface_area (Sphere 3) = 36 * pi) := 
by
  sorry

end sphere_cylinder_properties_sphere_cylinder_properties_radius_3_l216_216615


namespace exponent_rule_l216_216691

variable (a : ℝ) (m n : ℕ)

theorem exponent_rule (h1 : a^m = 3) (h2 : a^n = 2) : a^(m + n) = 6 :=
by
  sorry

end exponent_rule_l216_216691


namespace sum_of_coordinates_of_D_l216_216054

def Point := (ℝ × ℝ)

def isMidpoint (M C D : Point) : Prop :=
  M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)

theorem sum_of_coordinates_of_D (M C : Point) (D : Point) (hM : isMidpoint M C D) (hC : C = (2, 10)) :
  D.1 + D.2 = 12 :=
sorry

end sum_of_coordinates_of_D_l216_216054


namespace probability_event_20_30_l216_216042

/-- Definition of standard six-sided die outcomes. --/
def die_outcomes := {1, 2, 3, 4, 5, 6}

/-- Definition of the event of rolling two dice to form a number between 20 and 30. --/
def event_20_30 (d1 d2 : ℕ) : Prop :=
  (20 ≤ d1 * 10 + d2 ∧ d1 * 10 + d2 ≤ 30) ∨ (20 ≤ d2 * 10 + d1 ∧ d2 * 10 + d1 ≤ 30)

/-- Main statement to prove in Lean. --/
theorem probability_event_20_30 : 
  (∑ d1 in die_outcomes, ∑ d2 in die_outcomes, if event_20_30 d1 d2 then 1 else 0) / 36 = 23 / 36 :=
sorry

end probability_event_20_30_l216_216042


namespace estimated_population_correlation_coefficient_l216_216226

variables (xi yi : Fin 20 → ℝ)       -- xi represents plant coverage areas, yi represents number of wild animals
variables (sum_x : ℝ) (sum_y : ℝ) (sum_x_diff_sq : ℝ) (sum_y_diff_sq : ℝ) (sum_xy_diff : ℝ)

-- Conditions
def conditions := 
  (∑ i, xi i = 60) ∧ 
  (∑ i, yi i = 1200) ∧ 
  (∑ i, (xi i - (∑ j, xi j) / 20) ^ 2 = 80) ∧ 
  (∑ i, (yi i - (∑ j, yi j) / 20) ^ 2 = 9000) ∧ 
  (∑ i, (xi i - (∑ j, xi j) / 20) * (yi i - (∑ j, yi j) / 20) = 800)

-- Part 1: Prove the estimated population
theorem estimated_population (h : conditions xi yi sum_x sum_y sum_x_diff_sq sum_y_diff_sq sum_xy_diff) :
  200 * (sum_y / 20) = 12000 :=
by sorry

-- Part 2: Prove the correlation coefficient
theorem correlation_coefficient (h : conditions xi yi sum_x sum_y sum_x_diff_sq sum_y_diff_sq sum_xy_diff) :
  (sum_xy_diff / (real.sqrt (sum_x_diff_sq * sum_y_diff_sq))).round_to 2 = 0.94 :=
by sorry

end estimated_population_correlation_coefficient_l216_216226


namespace find_y_when_x_is_4_l216_216878

variables (x y : ℕ)
def inversely_proportional (C : ℕ) (x y : ℕ) : Prop := x * y = C

theorem find_y_when_x_is_4 :
  inversely_proportional 240 x y → x = 4 → y = 60 :=
by
  sorry

end find_y_when_x_is_4_l216_216878


namespace problem_1_problem_2_problem_3_l216_216627

-- Problem 1
theorem problem_1 (x : ℝ) : (x + 2) ^ 2 + x * (3 - x) = 7 * x + 4 :=
by
  sorry

-- Problem 2
theorem problem_2 (x y : ℝ) : (x + y - 3) * (x - y + 3) = x^2 - y^2 + 6 * y - 9 :=
by
  sorry

-- Problem 3
theorem problem_3 : | - (1 / 16) | + (2022 - π)^0 - (-4)^(-2) + (-2)^3 = -7 :=
by
  sorry

end problem_1_problem_2_problem_3_l216_216627


namespace circle_equations_l216_216288

theorem circle_equations (D E F : ℝ) :
  (∀ x y, x^2 + y^2 + D*x + E*y + F = 0 ∧ 
    (x = 1 ∧ y = 0 ∨ x = 0 ∧ y = 1) ∧ 
    (sqrt(D^2 - 4*F) = 6)) →
  (
    (D = -5 ∧ E = -6 ∧ F = 5 ∨
    D = -1 ∧ E = 6 ∧ F = -6) →
    ∀x y, x^2 + y^2 + D*x + E*y + F = 0 = 
          (x^2 + y^2 - 5*x - 6*y + 5 = 0 ∨
           x^2 + y^2 - x + 6*y - 6 = 0)
  )
:=
by
  sorry

end circle_equations_l216_216288
