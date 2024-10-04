import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.GCDMonoid
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Group.Powers
import Mathlib.Algebra.Group.Prod
import Mathlib.Algebra.LinearEquations
import Mathlib.Algebra.Order
import Mathlib.Analysis.Calculus
import Mathlib.Analysis.Circle
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.Trigonometry
import Mathlib.Combinatorics.SimpleGraph.Connectivity
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Digits
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Angle
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Time.Basic
import Mathlib.Geometry
import Mathlib.Geometry.Euclidean.Circles
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Geometry.Euclidean.Triangles
import Mathlib.Logic.Function
import Mathlib.MeasureTheory.ProbabilityTheory
import Mathlib.Meta
import Mathlib.Probability
import Mathlib.Tactic
import Mathlib.Tactic.SolveByElim
import algebra.lcm
import data.nat.basic
import data.nat.prime

namespace number_of_trees_l818_818245

theorem number_of_trees {N : ℕ} 
  (Anya : N = 15) 
  (Borya : N % 11 = 0) 
  (Vera : N < 25) 
  (Gena : N % 22 = 0) 
  (truth_conditions : (Anya ∨ Borya ∨ Vera ∨ Gena) ∧ ∃! p, p) : 
  N = 11 :=
sorry

end number_of_trees_l818_818245


namespace product_of_digits_of_non_divisible_by_5_l818_818456

theorem product_of_digits_of_non_divisible_by_5 :
  ∃ n, n ∈ [3525, 3540, 3565, 3580, 3592] ∧ n % 5 ≠ 0 ∧
  let tens := (n % 100) / 10 in
  let units := n % 10 in
  tens * units = 18 :=
by
  sorry

end product_of_digits_of_non_divisible_by_5_l818_818456


namespace japan_income_doubling_plan_indicates_intervention_l818_818380

theorem japan_income_doubling_plan_indicates_intervention (
  (divided_economy : Prop) 
  (expanded_infrastructure : Prop) 
  (enriched_social_security : Prop)) :
  divided_economy ∧ expanded_infrastructure ∧ enriched_social_security → 
  Strengthened_intervention_in_national_economy 
:= 
by 
  sorry

-- Defining the hypotheses as parameters
def divided_economy : Prop := sorry -- The Japanese government divided the economy into private and public sectors
def expanded_infrastructure : Prop := sorry -- The government expanded roads, harbors, urban planning, sewers, national housing, and other social capital.
def enriched_social_security : Prop := sorry -- The government aimed to enrich social security and welfare.

-- Defining the conclusion as a parameter
def Strengthened_intervention_in_national_economy : Prop := sorry

end japan_income_doubling_plan_indicates_intervention_l818_818380


namespace cos_product_identity_l818_818024

theorem cos_product_identity : cos (2 * Real.pi / 5) * cos (6 * Real.pi / 5) = -1 / 4 := by
  sorry

end cos_product_identity_l818_818024


namespace third_bowler_score_l818_818180

noncomputable def third_bowler (x : ℕ) : Prop :=
  let first_bowler := x -- as derived from the solution, first_bowler has the same score as third_bowler
  let second_bowler := 3 * x
  let total_points := 3 * x + x + x
  total_points = 810

theorem third_bowler_score : ∃ x : ℕ, third_bowler x ∧ x = 162 :=
by
  use 162
  dsimp [third_bowler]
  norm_num
  sorry

end third_bowler_score_l818_818180


namespace find_a_solutions_l818_818670

theorem find_a_solutions :
  ∃ a : ℝ, 
  (a ∈ set.Ioo 0 (real.cbrt (1 / 4)) ∪ set.Ioo 4 32) ∧
  (∃ (x y : ℝ), 
    (a * y - a * x + 2) * (4 * y - 3 * abs (x - a) - x + 5 * a) = 0 ∧
    real.sqrt (x^2 * y^2) = 4 * a ∧
    (∃ (x' y' : ℝ), x' ≠ x ∧ y' ≠ y 
      ∧ (a * y' - a * x' + 2) * (4 * y' - 3 * abs (x' - a) - x' + 5 * a) = 0 
      ∧ real.sqrt (x'^2 * y'^2) = 4 * a
      ∧ (∃ (x'' y'' : ℝ), x'' ≠ x' ∧ x'' ≠ x ∧ y'' ≠ y' ∧ y'' ≠ y 
          ∧ (a * y'' - a * x'' + 2) * (4 * y'' - 3 * abs (x'' - a) - x'' + 5 * a) = 0 
          ∧ real.sqrt (x''^2 * y''^2) = 4 * a
          ∧ (∃ (x''' y''' : ℝ), x''' ≠ x'' ∧ x''' ≠ x' ∧ x''' ≠ x ∧ y''' ≠ y'' ∧ y''' ≠ y' ∧ y''' ≠ y 
              ∧ (a * y''' - a * x''' + 2) * (4 * y''' - 3 * abs (x''' - a) - x''' + 5 * a) = 0 
              ∧ real.sqrt (x'''^2 * y'''^2) = 4 * a
              ∧ (∃ (x'''' y'''' : ℝ), x'''' ≠ x''' ∧ x'''' ≠ x'' ∧ x'''' ≠ x' ∧ x'''' ≠ x ∧ y'''' ≠ y''' ∧ y'''' ≠ y'' ∧ y'''' ≠ y' ∧ y'''' ≠ y 
                  ∧ (a * y'''' - a * x'''' + 2) * (4 * y'''' - 3 * abs (x'''' - a) - x'''' + 5 * a) = 0 
                  ∧ real.sqrt (x''''^2 * y''''^2) = 4 * a
                  ∧ (∃ (x''''' y''''' : ℝ), x''''' ≠ x'''' ∧ x''''' ≠ x''' ∧ x''''' ≠ x'' ∧ x''''' ≠ x' ∧ x''''' ≠ x 
                      ∧ y''''' ≠ y'''' ∧ y''''' ≠ y''' ∧ y''''' ≠ y'' ∧ y''''' ≠ y' ∧ y''''' ≠ y 
                      ∧ (a * y''''' - a * x''''' + 2) * (4 * y''''' - 3 * abs (x''''' - a) - x''''' + 5 * a) = 0 
                      ∧ real.sqrt (x'''''^2 * y'''''^2) = 4 * a 
                    )
                )
            )
        )
    )
  )
:=
sorry

end find_a_solutions_l818_818670


namespace no_rational_satisfies_l818_818463

theorem no_rational_satisfies (a b c d : ℚ) : ¬ ((a + b * Real.sqrt 3)^4 + (c + d * Real.sqrt 3)^4 = 1 + Real.sqrt 3) :=
sorry

end no_rational_satisfies_l818_818463


namespace max_sum_of_arithmetic_sequence_l818_818045

theorem max_sum_of_arithmetic_sequence 
  (d : ℤ) (a₁ a₃ a₅ a₁₅ : ℤ) (S : ℕ → ℤ)
  (h₁ : d ≠ 0)
  (h₂ : a₃ = a₁ + 2 * d)
  (h₃ : a₅ = a₃ + 2 * d)
  (h₄ : a₁₅ = a₅ + 10 * d)
  (h_geom : a₃ * a₃ = a₅ * a₁₅)
  (h_a₁ : a₁ = 3)
  (h_S : ∀ n, S n = n * a₁ + (n * (n - 1) / 2) * d) :
  ∃ n, S n = 4 :=
by
  sorry

end max_sum_of_arithmetic_sequence_l818_818045


namespace find_weight_of_b_l818_818044

variable (a b c d : ℝ)

def average_weight_of_four : Prop := (a + b + c + d) / 4 = 45

def average_weight_of_a_and_b : Prop := (a + b) / 2 = 42

def average_weight_of_b_and_c : Prop := (b + c) / 2 = 43

def ratio_of_d_to_a : Prop := d / a = 3 / 4

theorem find_weight_of_b (h1 : average_weight_of_four a b c d)
                        (h2 : average_weight_of_a_and_b a b)
                        (h3 : average_weight_of_b_and_c b c)
                        (h4 : ratio_of_d_to_a a d) :
    b = 29.43 :=
  by sorry

end find_weight_of_b_l818_818044


namespace min_quotient_value_l818_818235

-- Define the four-digit number as a sum of its digits and a term including a zero digit.
def four_digit_with_zero := {n : ℕ // n < 10000 ∧ ∃ a b c d : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧ {a, b, c, d} \subseteq {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ (1000*a + 100*b + 10*c + d) = n ∧ (a = 0 ∨ b = 0 ∨ c = 0 ∨ d = 0) ∧ (a + b + c + d) ≠ 0}

-- Define quotient function.
def quotient (n : four_digit_with_zero) : ℚ :=
  let ⟨n, hn⟩ := n in
  let ⟨_, h⟩ := hn in
  match h with 
  | ⟨a, ⟨b, ⟨c, ⟨d, ha⟩⟩⟩⟩ :=
    (n : ℚ) / (a + b + c + d)

-- State the minimum quotient value
theorem min_quotient_value : ∃ (n : four_digit_with_zero), quotient n = 105 := 
sorry

end min_quotient_value_l818_818235


namespace least_pos_int_with_six_factors_l818_818100

theorem least_pos_int_with_six_factors :
  ∃ n : ℕ, (∀ m : ℕ, (number_of_factors m = 6 → m ≥ n)) ∧ n = 12 := 
sorry

end least_pos_int_with_six_factors_l818_818100


namespace subtraction_correct_l818_818614

theorem subtraction_correct : 1_000_000_000_000 - 888_777_888_777 = 111_222_111_223 :=
by
  sorry

end subtraction_correct_l818_818614


namespace exp4_is_odd_l818_818546

-- Define the domain for n to be integers and the expressions used in the conditions
variable (n : ℤ)

-- Define the expressions
def exp1 := (n + 1) ^ 2
def exp2 := (n + 1) ^ 2 - (n - 1)
def exp3 := (n + 1) ^ 3
def exp4 := (n + 1) ^ 3 - n ^ 3

-- Prove that exp4 is always odd
theorem exp4_is_odd : ∀ n : ℤ, exp4 n % 2 = 1 := by {
  -- Lean code does not require a proof here, we'll put sorry to skip the proof
  sorry
}

end exp4_is_odd_l818_818546


namespace marnie_initial_chips_l818_818443

theorem marnie_initial_chips (x : ℕ) :
  (x > 0) →
  let total_chips := 100 in
  let first_day_chips := 2 * x in
  let remaining_chips := 90 in
  let total_days := 10 in
  (first_day_chips + remaining_chips = total_chips) →
  x = 5 := 
by {
  intro h_positive,
  intros,
  have h_eq : 2 * x + 90 = 100 := by assumption,
  simp at h_eq,
  -- Conclude that x must be 5
  suffices : x = 5, by assumption,
  linarith,
}

end marnie_initial_chips_l818_818443


namespace number_of_fir_trees_is_11_l818_818275

theorem number_of_fir_trees_is_11 
  (N : ℕ)
  (Anya : N = 15)
  (Borya : N % 11 = 0)
  (Vera : N < 25)
  (Gena : N % 22 = 0)
  (OneBoyOneGirlTrue : (Anya ∨ Borya) ∧ (Vera ∨ Gena) ∧ (¬Anya ∨ ¬Borya) ∧ (¬Vera ∨ ¬Gena)) :
  N = 11 := 
sorry

end number_of_fir_trees_is_11_l818_818275


namespace correct_expression_for_representatives_l818_818595

/-- Definition for the number of representatives y given the class size x
    and the conditions that follow. -/
def elect_representatives (x : ℕ) : ℕ :=
  if 6 < x % 10 then (x + 3) / 10 else x / 10

theorem correct_expression_for_representatives (x : ℕ) :
  elect_representatives x = (x + 3) / 10 :=
by
  sorry

end correct_expression_for_representatives_l818_818595


namespace distance_to_hospital_l818_818089

theorem distance_to_hospital {total_paid base_price price_per_mile : ℝ} (h1 : total_paid = 23) (h2 : base_price = 3) (h3 : price_per_mile = 4) : (total_paid - base_price) / price_per_mile = 5 :=
by
  sorry

end distance_to_hospital_l818_818089


namespace solve_for_z_l818_818875

theorem solve_for_z : ∀ z i : ℂ, i^2 = -1 ∧ (1 - i * z = -2 - 3 * i * z) → z = (-3 * i) / 2 :=
by
  intros z i hi heq
  have hz : 1 + 2 = (3 * i * z - i * z) := by
    rw [heq]
    ring
  have hz' : 3 = 2 * i * z := by
    linarith [hz]
  have z_eq : z = 3 / (2 * i) := by
    exact (mul_left_inj' (by norm_num : (2 * i : ℂ) ≠ 0)).mp (eq_comm.mp hz')
  rw [← mul_div_assoc, hi] at z_eq
  norm_num at z_eq
  assumption

end solve_for_z_l818_818875


namespace max_profit_at_boundary_l818_818973

noncomputable def profit (x : ℝ) : ℝ :=
  -50 * (x - 55) ^ 2 + 11250

def within_bounds (x : ℝ) : Prop :=
  40 ≤ x ∧ x ≤ 52

theorem max_profit_at_boundary :
  within_bounds 52 ∧ 
  (∀ x : ℝ, within_bounds x → profit x ≤ profit 52) :=
by
  sorry

end max_profit_at_boundary_l818_818973


namespace triangle_area_proof_l818_818033

noncomputable def area_of_triangle_ABC : ℝ :=
  8 * Real.sqrt 2

theorem triangle_area_proof (A B C C1 A1 : ℝ) 
  (h1 : is_height_of_triangle A B C C1)
  (h2 : right_angle_at A C1 C)
  (h3 : is_median A A1)
  (h4: isosceles_triangle A B C)
  (h5 : right_triangle B C C1)
  (h6 : median_equals A1 C1 2)
  (h7 : length BC = 4)
  (h8 : length AB = 6)
  (h9 : length AC = 6)
  (h10 : semi_perimeter = (length AB + length BC + length AC) / 2)
  (h: Herons_formula semi_perimeter (length AB) (length BC) (length AC) = 8 * Real.sqrt 2):
  area_of_triangle_ABC = Herons_formula semi_perimeter (length AB) (length BC) (length AC) := 
  by sorry


end triangle_area_proof_l818_818033


namespace E_bisects_angle_BED_l818_818176

variables {A B C D E : Type} [add_comm_group A] [module ℝ A]

def is_parallel (x y : A) : Prop := ∃ (k : ℝ), k ≠ 0 ∧ y = k • x
def is_perpendicular (x y : A) : Prop := ∀ (k : ℝ), y ≠ k • x


-- Conditions defined as Lean hypotheses
variables (AB CD BC AD AC DE CE : A)
variable (E : A)
variable (convex : Prop)

hypothesis h1 : is_parallel AB CD
hypothesis h2 : is_parallel BC AD
hypothesis h3 : is_parallel AC DE
hypothesis h4 : is_perpendicular CE BC

-- Theorem to be proved
theorem E_bisects_angle_BED (ABCDE_convex : convex) : 
  ∃ (x : ℝ), x ≠ 0 ∧ ((x • (B - E)) + (x • (D - E)) = (B - D)) :=
sorry

end E_bisects_angle_BED_l818_818176


namespace cos_neg_30_eq_sqrt_3_div_2_l818_818621

theorem cos_neg_30_eq_sqrt_3_div_2 :
  real.cos (-real.pi / 6) = real.sqrt 3 / 2 :=
sorry

end cos_neg_30_eq_sqrt_3_div_2_l818_818621


namespace f_zero_value_l818_818830

noncomputable def f : ℝ → ℝ := sorry

theorem f_zero_value :
  (∀ x : ℝ, f(x + 2) = f(x + 1) - f(x)) →
  f(1) = real.log10 (3/2) →
  f(2) = real.log10 15 →
  f(0) = -1 :=
by
  intros h1 h2 h3
  sorry

end f_zero_value_l818_818830


namespace emails_per_day_l818_818816

noncomputable def emails_per_day_before_subscription (x : ℕ) : Prop :=
  let emails_first_half := 15 * x in
  let emails_second_half := 15 * (x + 5) in
  let total_emails := emails_first_half + emails_second_half in
  total_emails = 675

theorem emails_per_day (x : ℕ) (h : emails_per_day_before_subscription x) : x = 20 :=
by {
  sorry
}

end emails_per_day_l818_818816


namespace fir_trees_count_l818_818298

theorem fir_trees_count (N : ℕ) :
  (N = 15 ∨ (N < 25 ∧ 11 ∣ N) ∨ 22 ∣ N) ∧ 
  (1 ≤ (if N = 15 then 1 else 0) + (if 11 ∣ N then 1 else 0) + (if N < 25 then 1 else 0) + (if 22 ∣ N then 1 else 0) ∧
   2 ≠ (if N = 15 then 1 else 0) + (if 11 ∣ N then 1 else 0) + (if N < 25 then 1 else 0) + (if 22 ∣ N then 1 else 0)) → N = 11 :=
begin
  sorry
end

end fir_trees_count_l818_818298


namespace simplify_f_value_of_f_l818_818002

variable (α : ℝ)

-- Define the function f(α)
def f (α : ℝ) : ℝ :=
  (sin (π - α) * cos (2 * π - α) * cos (-α + (3 * π / 2))) /
  (cos (π / 2 - α) * sin (-π - α))

-- State the theorem to simplify f(α) for α in the third quadrant
theorem simplify_f (hα : α ∈ Ioc π (3 * π / 2)) : f(α) = -cos(α) := sorry

-- State the theorem for the value of f(α) given sin(α) = -1/5.
theorem value_of_f (hα : α ∈ Ioc π (3 * π / 2)) (h_sin : sin(α) = -1/5) :
  f(α) = 2 * real.sqrt 6 / 5 :=
  sorry

end simplify_f_value_of_f_l818_818002


namespace sum_sequence_l818_818514

noncomputable def a : ℕ → ℚ
| 0 := 1/2
| (n+1) := a n / (3 * a n + 1)

theorem sum_sequence (n : ℕ) : 
    (∑ i in finset.range n, a i * a (i + 1)) = n / (6 * n + 4) := sorry

end sum_sequence_l818_818514


namespace inequality_solution_inequality_solution_minimum_value_l818_818741

variable (a x : ℝ)
variable (f : ℝ → ℝ) (h : ∀ x, x ≠ a → f x = (x^2 + 3) / (x - a))
variable (ha : a ≠ 0)

-- Statement for part (1)
theorem inequality_solution (h1 : a > 0) : 
  (-3/a < x ∧ x < a) ↔ f(x) < x :=
sorry

theorem inequality_solution' (h1 : a < 0) : 
  (x < a ∨ x > -3/a) ↔ f(x) < x :=
sorry

-- Statement for part (2)
theorem minimum_value (hx : x > a) (hmin : ∀ x, x > a → f x ≥ 6) : a = 1 :=
sorry

end inequality_solution_inequality_solution_minimum_value_l818_818741


namespace min_period_and_monotonic_interval_range_of_m_l818_818343

-- Function definition for f(x)
def f (x : ℝ) : ℝ := 2 * sin^2(π / 4 + x) - sqrt 3 * cos(2 * x)

-- Problem 1: Minimum positive period and interval of monotonic increase
theorem min_period_and_monotonic_interval :
  (∀ x : ℝ, f(x) = 2 * sin(2 * x - π / 3) + 1) ∧
  (∀ k : ℤ, 
    (f(x + k * π) = f(x)) ∧ 
    (∀ x : ℝ, f(x) is_increasing_on (Icc (k * π - π / 12) (k * π + 5 * π / 12)))) :=
sorry

-- Problem 2: Range of m
theorem range_of_m (m : ℝ) :
  (∃ x ∈ Icc(π / 4, π / 2), f(x) - m = 2) ↔ m ∈ Icc(0, 1) :=
sorry

end min_period_and_monotonic_interval_range_of_m_l818_818343


namespace vector_sum_magnitude_l818_818718

noncomputable def f (x : ℝ) : ℝ := Real.cos x
noncomputable def g (x : ℝ) : ℝ := Real.tan x

theorem vector_sum_magnitude (x₁ x₂ : ℝ) (hx₁ : 0 ≤ x₁ ∧ x₁ ≤ 2 * Real.pi)
  (hx₂ : 0 ≤ x₂ ∧ x₂ ≤ 2 * Real.pi) (hf : f x₁ = g x₁) (hg : f x₂ = g x₂) :
  ∥(x₁, f x₁) + (x₂, f x₂)∥ = Real.pi :=
by
  sorry

end vector_sum_magnitude_l818_818718


namespace total_games_required_l818_818189

def total_teams : ℕ := 24
def num_groups : ℕ := 4
def teams_per_group : ℕ := 6
def elimination_games_per_group : ℕ := teams_per_group - 1
def first_stage_games : ℕ := num_groups * elimination_games_per_group
def second_stage_teams : ℕ := num_groups
def second_stage_games : ℕ := second_stage_teams - 1
def total_games : ℕ := first_stage_games + second_stage_games

theorem total_games_required : total_games = 23 := by
  unfold total_teams num_groups teams_per_group elimination_games_per_group first_stage_games second_stage_teams second_stage_games total_games
  sorry

end total_games_required_l818_818189


namespace find_x_value_l818_818373

theorem find_x_value (x : ℝ) (h : 0.75 / x = 5 / 6) : x = 0.9 := 
by 
  sorry

end find_x_value_l818_818373


namespace find_abc_l818_818437

noncomputable def polynomial : Polynomial ℝ := X^3 + a * X^2 + b * X + c

def cos_2π_7 := Real.cos(2 * Real.pi / 7)
def cos_4π_7 := Real.cos(4 * Real.pi / 7)
def cos_6π_7 := Real.cos(6 * Real.pi / 7)

theorem find_abc (a b c : ℝ) (h1 : Polynomial.has_root polynomial cos_2π_7)
  (h2 : Polynomial.has_root polynomial cos_4π_7)
  (h3 : Polynomial.has_root polynomial cos_6π_7) :
  a * b * c = 1 / 32 :=
sorry

end find_abc_l818_818437


namespace volume_Q3_l818_818318

noncomputable def sequence_of_polyhedra (n : ℕ) : ℚ :=
match n with
| 0     => 1
| 1     => 3 / 2
| 2     => 45 / 32
| 3     => 585 / 128
| _     => 0 -- for n > 3 not defined

theorem volume_Q3 : sequence_of_polyhedra 3 = 585 / 128 :=
by
  -- Placeholder for the theorem proof
  sorry

end volume_Q3_l818_818318


namespace sum_proper_divisors_512_l818_818961

theorem sum_proper_divisors_512 : ∑ i in Finset.range 9, 2^i = 511 := by
  -- Proof would be provided here
  sorry

end sum_proper_divisors_512_l818_818961


namespace fir_trees_count_l818_818296

theorem fir_trees_count (N : ℕ) :
  (N = 15 ∨ (N < 25 ∧ 11 ∣ N) ∨ 22 ∣ N) ∧ 
  (1 ≤ (if N = 15 then 1 else 0) + (if 11 ∣ N then 1 else 0) + (if N < 25 then 1 else 0) + (if 22 ∣ N then 1 else 0) ∧
   2 ≠ (if N = 15 then 1 else 0) + (if 11 ∣ N then 1 else 0) + (if N < 25 then 1 else 0) + (if 22 ∣ N then 1 else 0)) → N = 11 :=
begin
  sorry
end

end fir_trees_count_l818_818296


namespace range_of_m_l818_818339

noncomputable theory
open Classical

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x ∈ set.Icc (-2 : ℝ) (1 : ℝ) ∧ (2 * m * x + 4 = 0)) ↔ (m ≤ -2 ∨ m ≥ 1) :=
by
  sorry

end range_of_m_l818_818339


namespace characterization_of_f_l818_818229

-- Define the function f from ℝ to ℝ
def f (x : ℝ) : ℝ := sorry

-- The functional equation given in the problem
def functional_eq (x y : ℝ) (h : x ≠ y) : Prop := 
  f ((x + y) / (x - y)) = (f x + f y) / (f x - f y)

-- The main theorem to prove: the only solution is f(x) = x
theorem characterization_of_f :
  (∀ (x y : ℝ), x ≠ y → functional_eq x y (by assumption)) → (∀ x : ℝ, f x = x) :=
begin
  intro H,
  sorry
end

end characterization_of_f_l818_818229


namespace all_lines_through_2_0_not_perpendicular_l818_818492

theorem all_lines_through_2_0_not_perpendicular (k : ℝ) :
  ∀ x y : ℝ, y = k * (x - 2) ↔ (x = 2 ∧ y = 0) → 
  (∃ k : ℝ, y = k * (x - 2) ∧ x ≠ 2) :=
by 
  assumption
sorry

end all_lines_through_2_0_not_perpendicular_l818_818492


namespace part1_part2_l818_818745

-- Definition of the function f
noncomputable def f (x : ℝ) := sqrt 2 * Real.cos (x - π / 12)

-- Given the angle and its properties
variables (θ : ℝ) (h_theta : θ ∈ Set.Ioo (3 * π / 2) (2 * π))
variables (h_cos_theta : Real.cos θ = 3 / 5)

-- Proof statements
theorem part1 : f (π / 3) = 1 := by
  sorry

theorem part2 : f (θ - π / 6) = -1 / 5 := by
  sorry

end part1_part2_l818_818745


namespace polar_coordinates_of_point_l818_818507

theorem polar_coordinates_of_point :
  ∀ (x y : ℝ), x = 1/2 → y = - (Real.sqrt 3) / 2 → 
  ∃ (ρ θ : ℝ), ρ = 1 → θ = 5 * Real.pi / 3 ∧ (ρ * Real.cos θ = x ∧ ρ * Real.sin θ = y) :=
by
  intros x y hx hy
  use 1
  use 5 * Real.pi / 3
  split
  · exact rfl
  split
  · rw [hx, hy]
    norm_num
  sorry

end polar_coordinates_of_point_l818_818507


namespace non_negative_solution_l818_818239

variables {n k : ℕ}
variables {x : Fin n → ℝ}

theorem non_negative_solution (h1 : ∑ i, (x i) ^ k = 1)
    (h2 : ∏ i, (1 + x i) = 2) : 
    ∃ i : Fin n, x i = 1 ∧ ∀ j : Fin n, j ≠ i → x j = 0 := 
by
  sorry

end non_negative_solution_l818_818239


namespace problem1_problem2_problem3_l818_818989

-- Problem 1
theorem problem1 (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : x1 < x2) : 
  (x1 + 1) / (x2 + 1) > x1 / x2 := sorry

-- Problem 2
noncomputable def f (x : ℝ) : ℝ := log (x + 1) - (1 / 2) * log 3 x

theorem problem2 (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : x1 < x2) : 
  f x1 < f x2 := sorry

-- Problem 3
def M := { n : ℤ | 0 < n^2 - 214 * n - 1998 ∧ n^2 - 214 * n - 1998 ≤ 9 }

theorem problem3 : 
  ∃ S : Finset (ℤ), S.card = 4 ∧ ∀ x ∈ S, x ∈ M := sorry

end problem1_problem2_problem3_l818_818989


namespace smallest_integer_with_six_distinct_factors_l818_818128

noncomputable def least_pos_integer_with_six_factors : ℕ :=
  12

theorem smallest_integer_with_six_distinct_factors 
  (n : ℕ)
  (p q : ℕ)
  (a b : ℕ)
  (hp : prime p)
  (hq : prime q)
  (h_diff : p ≠ q)
  (h_n : n = p ^ a * q ^ b)
  (h_factors : (a + 1) * (b + 1) = 6) :
  n = least_pos_integer_with_six_factors :=
by
  sorry

end smallest_integer_with_six_distinct_factors_l818_818128


namespace analytical_expressions_and_range_n_l818_818327

-- Define the functions
def f (x : ℝ) : ℝ := 2^x + 2^(-x)
def g (x : ℝ) : ℝ := 2^(-x) - 2^x

-- Hypotheses stating f is even and g is odd
lemma f_even (x : ℝ) : f (-x) = f x := by
  simp [f, pow_neg]
  ring

lemma g_odd (x : ℝ) : g (-x) = -g x := by
  simp [g, pow_neg]
  ring

-- The relationship between f(x) and g(x)
lemma fg_relation (x : ℝ) : f x + g x = 2^(1 - x) :=
  by simp [f, g, pow_add, pow_neg]
  ring

-- Proof for the range of n, given the inequality
lemma range_n (x y n : ℝ) (h_even : f (-x) = f x) (h_odd : g (-x) = -g x)
  (h_relation : f x + g x = 2^(1 - x)) : 2^(-y^2 - 2*y + n) * f x ≥ 1 -> n ≥ -2 :=
  by intro h
     have h1 : 2^(-y^2 - 2*y + n) ≥ 1 / (2^x + 2^(-x)) := by
       simp [h_even, h_odd, h_relation, f, g, pow_neg, pow_add]
       ring at h
       exact h
     have h2 : (2^x + 2^(-x)) ≥ 2 := by
       linarith [pow_le_of_le sqrt_pos']
     suffices h3 : 2^(-y^2 - 2*y + n) ≥ 1/2
       by linarith
     apply pow_le_iff_le
     apply le_of_eq_or_lt
     left
     linarith
     right
     linarith
     done

-- Final theorem consolidating above lemmas into a single statement
theorem analytical_expressions_and_range_n (x y n : ℝ) :
  2^(-y^2 - 2*y + n) * (2^x + 2^(-x)) ≥ 1 -> n ≥ -2 :=
  by
    intro h
    apply range_n
    exact f_even
    exact g_odd
    exact fg_relation
    exact h
    done

end analytical_expressions_and_range_n_l818_818327


namespace problem1_l818_818567

def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log (1/2) else Real.cos x

theorem problem1 : f (f (-Real.pi / 3)) = 1 :=
by
  -- sorry statement to skip proof
  sorry

end problem1_l818_818567


namespace postage_cost_formula_l818_818779

def postage_cost (W : ℝ) : ℝ :=
  7 * ⌈W⌉ + 3

theorem postage_cost_formula (W : ℝ) : postage_cost W = 7 * ⌈W⌉ + 3 := by
  sorry

end postage_cost_formula_l818_818779


namespace least_distinct_values_l818_818182

/-- Given a list of 2023 positive integers with a unique mode occurring exactly 15 times,
    prove that the least number of distinct values that can occur in the list is 145.
-/
theorem least_distinct_values (n m : ℕ) (h1 : n = 2023) (h2 : m = 15) :
  ∃ k : ℕ, k = 145 ∧ (∀ l : list ℕ, l.length = n → (∃ x : ℕ, l.count x = m → k ≤ l.nodup.count)) :=
sorry

end least_distinct_values_l818_818182


namespace profit_percent_is_25_l818_818558

-- Define the cost price (CP) and selling price (SP) based on the given ratio.
def CP (x : ℝ) := 4 * x
def SP (x : ℝ) := 5 * x

-- Calculate the profit percent based on the given conditions.
noncomputable def profitPercent (x : ℝ) := ((SP x - CP x) / CP x) * 100

-- Prove that the profit percent is 25% given the ratio of CP to SP is 4:5.
theorem profit_percent_is_25 (x : ℝ) : profitPercent x = 25 := by
  sorry

end profit_percent_is_25_l818_818558


namespace probability_distribution_X_l818_818579

noncomputable def hypergeom_pmf (N k l r : ℕ) : ℚ :=
  (Nat.choose k r * Nat.choose (N - k) (l - r)) / (Nat.choose N l)

theorem probability_distribution_X :
  let N := 15
  let k := 5
  let l := 3 in
  (hypergeom_pmf N k l 0 = 24 / 91) ∧
  (hypergeom_pmf N k l 1 = 45 / 91) ∧
  (hypergeom_pmf N k l 2 = 20 / 91) ∧
  (hypergeom_pmf N k l 3 = 2 / 91) :=
by { sorry }

end probability_distribution_X_l818_818579


namespace symmetric_point_proof_l818_818396

noncomputable def point_symmetric_to_x_axis (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1, -A.2)

theorem symmetric_point_proof :
  point_symmetric_to_x_axis (-2, 3) = (-2, -3) :=
by
  sorry

end symmetric_point_proof_l818_818396


namespace total_value_of_item_l818_818544

theorem total_value_of_item (V : ℝ) 
  (h1 : 0.07 * (V - 1000) = 109.20) : 
  V = 2560 :=
sorry

end total_value_of_item_l818_818544


namespace sum_of_pairwise_products_does_not_end_in_2019_l818_818203

theorem sum_of_pairwise_products_does_not_end_in_2019 (n : ℤ) : ¬ (∃ (k : ℤ), 10000 ∣ (3 * n ^ 2 - 2020 + k * 10000)) := by
  sorry

end sum_of_pairwise_products_does_not_end_in_2019_l818_818203


namespace book_division_ways_l818_818571

/-- The number of ways to divide 6 different books into 3 groups, with one group containing
4 books and the other two groups containing 1 book each, is 15. -/
theorem book_division_ways : 
  ∃ (ways : ℕ), ways = 15 ∧ ways = (Nat.choose 6 4) * (Nat.choose 2 1) * (Nat.choose 1 1) / 2 :=
by
  use 15
  simp [Nat.choose]
  -- Proof omitted
  sorry

end book_division_ways_l818_818571


namespace arithmetic_seq_sum_equidistant_l818_818428

theorem arithmetic_seq_sum_equidistant :
  ∃ (a : ℕ → ℕ) (d : ℕ), 
    (∀ n, a n = a 0 + n * d) ∧ 
    (log 2 (a 6) = 3 ⊸ a 6 + a 8 = 16) :=
by
  exists sorry
  exists sorry
  intro n
  split
  case log_2_a7 =>
    sorry
  case a6_a8 =>
    sorry

end arithmetic_seq_sum_equidistant_l818_818428


namespace nylon_cord_length_l818_818552

noncomputable def cord_length (pi_approx : ℝ) : ℝ :=
  30 / pi_approx

theorem nylon_cord_length (pi_approx : ℝ) (hpi : pi_approx ≈ 3.14159) :
  cord_length pi_approx ≈ 9.55 := by
  sorry

end nylon_cord_length_l818_818552


namespace remainder_when_divided_by_6_l818_818590

theorem remainder_when_divided_by_6 :
  ∃ (n : ℕ), (∃ k : ℕ, n = 3 * k + 2 ∧ ∃ m : ℕ, k = 4 * m + 3) → n % 6 = 5 :=
by
  sorry

end remainder_when_divided_by_6_l818_818590


namespace diagonal_sum_le_M_l818_818712

-- Define the conditions
variable {M : ℝ} (hM : 0 < M) {n : ℕ} (A : Fin n → Fin n → ℝ)
variable (h : ∀ x : Fin n → ℝ, (∀ i, x i = 1 ∨ x i = -1) → ∑ k : Fin n, abs (∑ i : Fin n, A k i * x i) ≤ M)

-- Theorem statement
theorem diagonal_sum_le_M : ∑ k : Fin n, abs (A k k) ≤ M :=
sorry

end diagonal_sum_le_M_l818_818712


namespace sin_of_7pi_over_6_l818_818659

theorem sin_of_7pi_over_6 : Real.sin (7 * Real.pi / 6) = -1 / 2 :=
by
  -- Conditions from the statement in a)
  -- Given conditions: \(\sin (180^\circ + \theta) = -\sin \theta\)
  -- \(\sin 30^\circ = \frac{1}{2}\)
  sorry

end sin_of_7pi_over_6_l818_818659


namespace integral_inner_function_evaluation_l818_818303

noncomputable def f (x : ℝ) : ℝ :=
  ∫ t in 0..x, Real.sin t

theorem integral_inner_function_evaluation :
  f (f (Real.pi / 2)) = 1 - Real.cos 1 :=
  sorry

end integral_inner_function_evaluation_l818_818303


namespace increasing_intervals_l818_818900

-- Define the function f(x)
def f (x : ℝ) : ℝ := x - Real.log (x^2)

-- Define the first derivative of f(x)
def f_prime (x : ℝ) : ℝ := (x - 2) / x

-- Prove that the monotonically increasing intervals are (-∞, 0) ∪ (2, +∞)
theorem increasing_intervals : 
  let domain := {x : ℝ | x ≠ 0}
  ∀ x : ℝ, (x ∈ domain) → (f_prime(x) > 0 ↔ (x < 0 ∨ x > 2)) :=
by
  sorry

end increasing_intervals_l818_818900


namespace part_a_proof_part_b_proof_l818_818994

-- Definitions and Conditions
variables {A B C K D E : Type} [Triangle ABC]
variables {rho p r r1 a s : ℝ}
variables {circle_tangent_to_AB_AC_K_center_distance_p : circle.radius = rho ∧ K ∈ center ∧ distance K BC = p}
variables {circle_intersects_BC_at_D_and_E : K.radius ∈ R ∧ intersects BC D ∧ intersects BC E} 

-- Definitions for part (a)
def inradius_of_triangle := r
def semiperimeter_of_triangle := s

-- Proof of part (a)
theorem part_a_proof : a * (p - rho) = 2 * s * (r - rho) :=
sorry

-- Definitions for part (b)
def exradius_corresponding_to_vertex : real := r1

-- Proof of part (b)
theorem part_b_proof : DE = (4 * sqrt (r * r1 * (rho - r) * (r1 - rho))) / (r1 - r) :=
sorry

end part_a_proof_part_b_proof_l818_818994


namespace cos_value_of_W_l818_818794

variables (W X Y Z : Type) [AddGroup W] [AddGroup X]
variables (angleW angleY : ℝ) (WZ YX : ℝ) (WX ZY : ℝ)
variables (perimeter : ℝ)

def cos_of_angleW (aW aY : ℝ) (perim : ℝ) (wz yx : ℝ) (wx zy : ℝ) (hne : wx ≠ zy) : ℝ :=
  let beta := aW in
  let s := wx in
  let t := zy in
  let cosβ := (s^2 - t^2) / (2 * wx * t - wz^2 + yx^2 - 2 * yx * t * cos β) / 300 * (s - t) in
  cos β

theorem cos_value_of_W 
  (hW_eq_Y : angleW = angleY) 
  (hWZ_yX : WZ = 150 ∧ YX = 150) 
  (hne : WX ≠ ZY) 
  (hperimeter : WX + ZY + 2 * 150 = 520) 
  : cos_of_angleW angleW angleY perimeter 150 150 WX ZY hne = 11 / 15 :=
by 
  sorry

end cos_value_of_W_l818_818794


namespace four_distinct_numbers_l818_818974

theorem four_distinct_numbers :
  ∃ (A : Set ℚ), A = {1/5, 6/5, 11/5, 16/5} ∧
  (∀ x ∈ A, (x - 1) ∈ A ∨ (6 * x - 1) ∈ A) :=
by
  let A := {a | a = 1/5 ∨ a = 6/5 ∨ a = 11/5 ∨ a = 16/5}
  have distinct: Set.card A = 4 := sorry
  have condition: ∀ x ∈ A, (x - 1) ∈ A ∨ (6 * x - 1) ∈ A := sorry
  exact ⟨A, by finish, condition⟩

end four_distinct_numbers_l818_818974


namespace number_of_valid_schedules_l818_818599

open Finset

-- Define the conditions
def courses : Finset String := {"Algebra", "Geometry", "Number Theory", "Calculus"}
def periods : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Definition for valid schedule (for the sake of this problem, represent them formally)
def valid_schedule (schedule : ℕ → String) : Prop :=
  ∀ i j, i ≠ j → abs (i - j) ≠ 1 → schedule i ∈ courses → schedule j ∈ courses 

-- The theorem statement
theorem number_of_valid_schedules : ∃ (count : ℕ), count = 504 ∧ ∀ schedule, valid_schedule schedule → true := 
  by
  sorry

end number_of_valid_schedules_l818_818599


namespace find_A_and_evaluate_A_minus_B_l818_818148

-- Given definitions
def B (x y : ℝ) : ℝ := 4 * x ^ 2 - 3 * y - 1
def result (x y : ℝ) : ℝ := 6 * x ^ 2 - y

-- Defining the polynomial A based on the first condition
def A (x y : ℝ) : ℝ := 2 * x ^ 2 + 2 * y + 1

-- The main theorem to be proven
theorem find_A_and_evaluate_A_minus_B :
  (∀ x y : ℝ, B x y + A x y = result x y) →
  (∀ x y : ℝ, |x - 1| * (y + 1) ^ 2 = 0 → A x y - B x y = -5) :=
by
  intro h1 h2
  sorry

end find_A_and_evaluate_A_minus_B_l818_818148


namespace asymptote_hole_sum_l818_818805

noncomputable def number_of_holes (f : ℝ → ℝ) : ℕ := -- Assume a definition to count holes
sorry

noncomputable def number_of_vertical_asymptotes (f : ℝ → ℝ) : ℕ := -- Assume a definition to count vertical asymptotes
sorry

noncomputable def number_of_horizontal_asymptotes (f : ℝ → ℝ) : ℕ := -- Assume a definition to count horizontal asymptotes
sorry

noncomputable def number_of_oblique_asymptotes (f : ℝ → ℝ) : ℕ := -- Assume a definition to count oblique asymptotes
sorry

theorem asymptote_hole_sum :
  let f := λ x => (x^2 + 4*x + 3) / (x^3 - 2*x^2 - x + 2)
  let a := number_of_holes f
  let b := number_of_vertical_asymptotes f
  let c := number_of_horizontal_asymptotes f
  let d := number_of_oblique_asymptotes f
  a + 2 * b + 3 * c + 4 * d = 8 :=
by
  sorry

end asymptote_hole_sum_l818_818805


namespace solve_system_eq_l818_818466

theorem solve_system_eq (x y z t : ℕ) : 
  ((x^2 + t^2) * (z^2 + y^2) = 50) ↔
    (x = 1 ∧ y = 1 ∧ z = 2 ∧ t = 3) ∨
    (x = 3 ∧ y = 2 ∧ z = 1 ∧ t = 1) ∨
    (x = 4 ∧ y = 1 ∧ z = 3 ∧ t = 1) ∨
    (x = 1 ∧ y = 3 ∧ z = 4 ∧ t = 1) :=
by 
  sorry

end solve_system_eq_l818_818466


namespace find_point_B_and_a_l818_818312

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1)/2, (p1.2 + p2.2)/2)

theorem find_point_B_and_a (x y a : ℝ) (B E F : ℝ × ℝ) 
  (h1 : 0 < y)
  (h2 : 0 < x)
  (h3 : B = (x, y))
  (h4 : (x - 1)^2 + (y + 2)^2 = 18)
  (h5 : y = x - 3)
  (h6 : E ≠ F)
  (h7 : ∀ E F, E = (x1, y1) → F = (x2, y2) → (midpoint E F = (4, 1)) → x1 + x2 = 6*(a^2)/(a^2 - 1)) :
  B = (4, 1) ∧ a = 2 := by
  sorry

end find_point_B_and_a_l818_818312


namespace sum_proper_divisors_of_512_l818_818954

theorem sum_proper_divisors_of_512 : ∑ i in finset.range 9, 2^i = 511 :=
by
  -- We are stating that the sum of 2^i for i ranging from 0 to 8 equals 511.
  sorry

end sum_proper_divisors_of_512_l818_818954


namespace tangent_line_l818_818811

namespace Geometry

structure IsoscelesTrapezoid (A B C D : Type) :=
(base1 : B ≠ C)
(base2 : A ≠ D)
(congruent_sides : AB = CD)
(axis_symmetry : ∃ I, ∃ r, ∀ P, P ∈ Ω → dist P I = r)

variable {A B C D I E : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace I] [MetricSpace E]

def circle (center : Type) (radius : ℝ) :=
{ P : Type // dist P center = radius }

noncomputable def tangent (ℓ : Type) (ω : circle I r) :=
∀ P ∈ ℓ, dist P I = r

theorem tangent_line (trapezoid : IsoscelesTrapezoid A B C D) (circumcircle : ∀ P ∈ Ω, dist P I = r) (tangent_point : ∃ E, ∀ E ∈ AB, CE ∩ circumcircle = E) :
  tangent CE (circle I r) :=
sorry

end Geometry

end tangent_line_l818_818811


namespace evaluate_series_l818_818357

-- Definitions
def u : ℝ × ℝ × ℝ := (1/3, 1/3, 1/3)
def v0 : ℝ × ℝ × ℝ := (1, 2, 3)

noncomputable def v : ℕ → ℝ × ℝ × ℝ
| 0     := v0
| (n+1) := let x := u.1 * v n.2 - u.2 * v n.1 + u.3 * v n.1,
               y := - (u.1 * v n.0 - u.0 * v n.1) + u.3 * v n.2,
               z := u.1 * v n.2 - u.2 * v n.1 in
             (x, y, z)

theorem evaluate_series : ∑' n, (3, 2, 1).fst * (v (2 * n)).fst + (3, 2, 1).snd * (v (2 * n)).snd + (3, 2, 1).thd * (v (2 * n)).thd = 1 :=
by
  sorry

end evaluate_series_l818_818357


namespace number_of_trees_is_11_l818_818267

variable {N : ℕ}

-- Conditions stated by each child
def anya_statement : Prop := N = 15
def borya_statement : Prop := N % 11 = 0
def vera_statement : Prop := N < 25
def gena_statement : Prop := N % 22 = 0

-- One boy and one girl told the truth, while the other two lied
def truth_condition : Prop :=
  (borya_statement ∧ vera_statement ∧ ¬anya_statement ∧ ¬gena_statement) ∨
  (borya_statement ∧ ¬vera_statement ∧ ¬anya_statement ∧ gena_statement) ∨
  (¬borya_statement ∧ vera_statement ∧ ¬anya_statement ∧ gena_statement) ∨
  (¬borya_statement ∧ vera_statement ∧ anya_statement ∧ ¬gena_statement)

-- Proving that the number of fir trees is 11
theorem number_of_trees_is_11 (h : truth_condition) : N = 11 := 
by
  sorry

end number_of_trees_is_11_l818_818267


namespace Farrah_total_match_sticks_l818_818225

def boxes := 4
def matchboxes_per_box := 20
def sticks_per_matchbox := 300

def total_matchboxes : Nat :=
  boxes * matchboxes_per_box

def total_match_sticks : Nat :=
  total_matchboxes * sticks_per_matchbox

theorem Farrah_total_match_sticks : total_match_sticks = 24000 := sorry

end Farrah_total_match_sticks_l818_818225


namespace lcm_36_105_l818_818677

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end lcm_36_105_l818_818677


namespace scarves_per_yarn_correct_l818_818847

def scarves_per_yarn (total_yarns total_scarves : ℕ) : ℕ :=
  total_scarves / total_yarns

theorem scarves_per_yarn_correct :
  scarves_per_yarn (2 + 6 + 4) 36 = 3 :=
by
  sorry

end scarves_per_yarn_correct_l818_818847


namespace function_expression_triangle_side_length_a_l818_818762

-- Define vectors and f(x)
def vector_a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
def vector_b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.cos x)
def f (x : ℝ) : ℝ := (vector_a x).fst * (vector_b x).fst + (vector_a x).snd * (vector_b x).snd - 1 / 2

-- First proof problem: expression of f(x)
theorem function_expression (x : ℝ) : f(x) = Real.sin(2 * x + π / 6) :=
sorry

-- Area and side length calculation
variables (A b S : ℝ)
def fA : ℝ := f(A / 2)

-- Triangle given conditions
def triangle_conditions : Prop := (fA = 1) ∧ (b = 1) ∧ (S = Real.sqrt 3)

-- Second proof problem: value of a
theorem triangle_side_length_a (a c : ℝ) (h₁ : triangle_conditions A b S) (h₂ : c = 4) : a = Real.sqrt 13 :=
sorry

end function_expression_triangle_side_length_a_l818_818762


namespace find_value_of_tangent_line_l818_818751

noncomputable def is_tangent (L : ℝ → ℝ → ℝ) (C : ℝ → ℝ → ℝ → ℝ) (m : ℝ) : Prop :=
  ∃ x y, L x y = 0 ∧ C x y m = 0

theorem find_value_of_tangent_line :
  ∀ (m : ℝ),
  (∀ x y : ℝ, 3 * x - 4 * y - 6 ≠ 0) → 
  is_tangent (λ x y, 3 * x - 4 * y - 6) (λ x y m, x^2 + y^2 - 2 * y + m) m → 
  m = -3 := 
  by sorry

end find_value_of_tangent_line_l818_818751


namespace evaluate_expression_l818_818222

noncomputable def expression (a b : ℕ) := (a + b)^2 - (a - b)^2

theorem evaluate_expression:
  expression (5^500) (6^501) = 24 * 30^500 := by
sorry

end evaluate_expression_l818_818222


namespace find_line_equation_l818_818494

def line_equation (p : ℝ × ℝ) (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y, y - p.2 = m * (x - p.1)

theorem find_line_equation : line_equation (-2, 1) (Real.tan (135 * Real.pi / 180)) x y ↔ x + y + 1 = 0 :=
by
  sorry

end find_line_equation_l818_818494


namespace sum_proper_divisors_512_l818_818960

theorem sum_proper_divisors_512 : ∑ i in Finset.range 9, 2^i = 511 := by
  -- Proof would be provided here
  sorry

end sum_proper_divisors_512_l818_818960


namespace regular_polygon_sides_l818_818382

theorem regular_polygon_sides (n : ℕ) (h : 108 = 180 * (n - 2) / n) : n = 5 := 
sorry

end regular_polygon_sides_l818_818382


namespace real_number_a_l818_818736

theorem real_number_a (a : ℝ) (ha : ∃ b : ℝ, z = 0 + bi) : a = 1 :=
sorry

end real_number_a_l818_818736


namespace least_positive_integer_with_six_factors_l818_818109

theorem least_positive_integer_with_six_factors :
  ∃ n : ℕ, (∀ m : ℕ, m > 0 → m < n → (count_factors m ≠ 6)) ∧ count_factors n = 6 ∧ n = 18 :=
sorry

noncomputable def count_factors (n : ℕ) : ℕ :=
sorry

end least_positive_integer_with_six_factors_l818_818109


namespace graph_does_not_pass_first_quadrant_l818_818055

noncomputable def f (x : ℝ) : ℝ := (1/2)^x - 2

theorem graph_does_not_pass_first_quadrant :
  ¬ ∃ x > 0, f x > 0 := by
sorry

end graph_does_not_pass_first_quadrant_l818_818055


namespace randy_trip_length_l818_818870

theorem randy_trip_length (x : ℝ) (h : x / 2 + 30 + x / 4 = x) : x = 120 :=
by
  sorry

end randy_trip_length_l818_818870


namespace train_crosses_platform_in_given_time_l818_818999

-- Conditions
def train_speed_kmph : ℝ := 72
def train_length_m : ℝ := 250.0416
def platform_length_m : ℝ := 270

-- Conversion factor from kmph to m/s
def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * 1000 / 3600

-- Speed of the train in m/s
def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph

-- Total distance covered by the train while crossing the platform
def total_distance_m : ℝ := train_length_m + platform_length_m

-- Time taken (in seconds) = Distance / Speed
def time_to_cross_platform (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

theorem train_crosses_platform_in_given_time :
  time_to_cross_platform total_distance_m train_speed_mps = 26.00208 :=
by
  sorry

end train_crosses_platform_in_given_time_l818_818999


namespace find_p_value_l818_818350

open Real

noncomputable def parabola_proof_problem (p : ℝ) (A B : ℝ × ℝ) (m : ℝ) : Prop :=
  0 < p ∧
  (A.2^2 = 2 * p * A.1) ∧ (B.2^2 = 2 * p * B.1) ∧
  (A.2 + B.2 = -4) ∧
  (m = (A.2 + B.2) / 2) ∧
  (m = -2)

theorem find_p_value : ∃ p A B m, parabola_proof_problem p A B m ∧ p = 2 := 
by {
  sorry,
}

end find_p_value_l818_818350


namespace sin_cos_identity_solution_l818_818986

theorem sin_cos_identity_solution (x : ℝ) :
  (sin (2025 * x))^4 + (cos (2016 * x))^2019 * (cos (2025 * x))^2018 = 1 →
  (∃ (n : ℤ), x = (π / 4050) + (n * π / 2025)) ∨ (∃ (k : ℤ), x = (k * π / 9)) :=
by
  intros h,
  sorry

end sin_cos_identity_solution_l818_818986


namespace race_time_l818_818980

variable (A B : Type)
variables (Va Vb : ℕ) (distance time : ℕ)
variable (finishes_in : A → B → ℕ → ℕ → Prop)
variable (partial_distance : B → ℕ → ℕ → Prop)

theorem race_time (h1 : distance = 200)
                 (h2 : time = 7)
                 (h3 : finishes_in A B distance time) 
                 (h4 : partial_distance B (distance - 35) time):
  distance / time = 200 / 7 :=
by
  sorry

end race_time_l818_818980


namespace isosceles_triangle_l818_818316

theorem isosceles_triangle 
  (ABC : Type)
  [triangle ABC]
  (A B C D E : ABC)
  (l_A l_B : line ABC)
  (h1 : bisects_angle A B C l_A)
  (h2 : bisects_angle B A C l_B)
  (h3 : parallel L1 l_A)
  (h4 : parallel L2 l_B)
  (h5 : intersects L1 l_B D)
  (h6 : intersects L2 l_A E)
  (h7 : parallel (line_through D E) (line_through A B)) :
  is_isosceles ABC :=
sorry

end isosceles_triangle_l818_818316


namespace faruk_ranjith_difference_l818_818196

theorem faruk_ranjith_difference (h_ratio : 35/10 = 350 / 100 ∧ 52.5 / 10 = 525 / 100 ∧ 117.5 / 10 = 1175 / 100 ∧ 75 / 10 = 750 / 100)
  (parts_sum : 350 + 525 + 1175 + 750 = 2800)
  (vasim_share : 2250 / 525 = 4.285714286)
  (faruk_share : 350 * 4.285714286 ≈ 1500)
  (ranjith_share : 1175 * 4.285714286 ≈ 5035.714286) : 
  (ranjith_share - faruk_share) ≈ 3536 :=
begin
  -- Proof goes here
  sorry
end

end faruk_ranjith_difference_l818_818196


namespace circumcenter_fixed_circle_l818_818427

variables {C1 C2 : Circle}  -- Defining the two circles
variables {P Q : Point}  -- Intersection points of the circles
variables {A1 B1 : Point}  -- Variable points on C1
variables {A2 B2 : Point}  -- Points on C2 intersected by lines A1P and B1P respectively
variables {C : Point}  -- Intersection of lines A1B1 and A2B2
variables {O : Point}  -- Center of circumcircle of triangle A1A2C
variables {O1 O2 : Point}  -- Centers of the circles C1 and C2 respectively

axiom intersecting_circles (C1 C2 : Circle) : ∃ P Q, P ≠ Q ∧ on_circle P C1 ∧ on_circle Q C1 ∧ on_circle P C2 ∧ on_circle Q C2
axiom selected_points (A1 B1 : Point) (C1 : Circle) : on_circle A1 C1 ∧ on_circle B1 C1
axiom intersection_point (A1 P A2 B1 B2 : Point) (C2 : Circle) : on_circle A2 C2 ∧ on_circle B2 C2 ∧ 
  line A1 P A2 ∧ line B1 P B2
axiom intersection_lines (A1 B1 A2 B2 : Point) : ∃ C, intersection (line A1 B1) (line A2 B2) = C
axiom cyclic_quadrilateral (A1 A2 C Q : Point) : cyclic_quad A1 A2 C Q

theorem circumcenter_fixed_circle :
  ∀ O O1 O2, circumcenter (triangle A1 A2 C) = O →
  circumcenter (triangle Q O1 O2) = O →
  on_circle O (circumcircle Q O1 O2) :=
by { sorry }

end circumcenter_fixed_circle_l818_818427


namespace shaded_area_calculation_l818_818050

-- Define the dimensions of the grid and the size of each square
def gridWidth : ℕ := 9
def gridHeight : ℕ := 7
def squareSize : ℕ := 2

-- Define the number of 2x2 squares horizontally and vertically
def numSquaresHorizontally : ℕ := gridWidth / squareSize
def numSquaresVertically : ℕ := gridHeight / squareSize

-- Define the area of one 2x2 square and one shaded triangle within it
def squareArea : ℕ := squareSize * squareSize
def shadedTriangleArea : ℕ := squareArea / 2

-- Define the total number of 2x2 squares
def totalNumSquares : ℕ := numSquaresHorizontally * numSquaresVertically

-- Define the total area of shaded regions
def totalShadedArea : ℕ := totalNumSquares * shadedTriangleArea

-- The theorem to be proved
theorem shaded_area_calculation : totalShadedArea = 24 := by
  sorry    -- Placeholder for the proof

end shaded_area_calculation_l818_818050


namespace probability_of_choosing_A_l818_818170

def P (n : ℕ) : ℝ :=
  if n = 0 then 1 else 0.5 + 0.5 * (-0.2)^(n-1)

theorem probability_of_choosing_A (n : ℕ) :
  P n = if n = 0 then 1 else 0.5 + 0.5 * (-0.2)^(n-1) := 
by {
  sorry
}

end probability_of_choosing_A_l818_818170


namespace thirtieth_term_of_sequence_is_424_l818_818519

def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0
def contains_digit_2 (n : ℕ) : Prop := ∃ k, n.digits 10 k = 2

def sequence_term (k : ℕ) : ℕ :=
  let seq := { n : ℕ | is_multiple_of_4 n ∧ contains_digit_2 n}
  (seq.to_list k).get! k

theorem thirtieth_term_of_sequence_is_424 : sequence_term 29 = 424 := 
by sorry

end thirtieth_term_of_sequence_is_424_l818_818519


namespace lcm_of_36_and_105_l818_818673

theorem lcm_of_36_and_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end lcm_of_36_and_105_l818_818673


namespace proposition_truth_l818_818322

-- Define propositions
def p : Prop := ∀ x : ℝ, x > 0 → log x ≥ 0
def q : Prop := ∃ x0 : ℝ, sin x0 = cos x0

-- The proof statement
theorem proposition_truth : ¬ p ∧ q :=
by
  sorry

end proposition_truth_l818_818322


namespace semicircle_radius_is_approx_7_l818_818510

noncomputable def semicircle_radius (perimeter : ℝ) : ℝ := 
  perimeter / (Real.pi + 2)

theorem semicircle_radius_is_approx_7 :
  semicircle_radius 35.99114857512855 ≈ 7 := 
by
  sorry

end semicircle_radius_is_approx_7_l818_818510


namespace lcm_36_105_l818_818682

noncomputable def factorize_36 : List (ℕ × ℕ) := [(2, 2), (3, 2)]
noncomputable def factorize_105 : List (ℕ × ℕ) := [(3, 1), (5, 1), (7, 1)]

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 :=
by
  have h_36 : 36 = 2^2 * 3^2 := by norm_num
  have h_105 : 105 = 3^1 * 5^1 * 7^1 := by norm_num
  sorry

end lcm_36_105_l818_818682


namespace point_Y_on_circumcircle_l818_818562

noncomputable def isosceles_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] :=
  ∃ (AC : Line) (is_isosceles : AC = base_of (triangle A B C))

variables {A B C P Q X Y : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P] 
  [MetricSpace Q] [MetricSpace X] [MetricSpace Y]

def point_on_line_segment (X : Type) (AC : Line) :=
  ∃ (param : ℝ) (h_param : 0 ≤ param ∧ param ≤ 1), X = param * (end_point of Line AC)

def parallelogram (X P B Q : Type) :=
  ∃ (XP : Line) (BQ : Line), X P B Q form_parallelogram using XP and BQ

def symmetric_point (X PQ : Type) :=
  ∃ (Y : Type) (is_symmetric : Y = reflection_of X along line PQ)

def circumcircle_property (A B C Y : Type) :=
  Y lies_on_circumcircle_of (triangle A B C)

theorem point_Y_on_circumcircle (h1 : isosceles_triangle A B C)
  (h2 : point_on_line_segment X (base_of (triangle A B C)))
  (h3 : parallelogram X P B Q)
  (h4 : symmetric_point X (line_through P Q)):
  circumcircle_property A B C Y :=
sorry

end point_Y_on_circumcircle_l818_818562


namespace S_is_not_finitely_union_of_arithmetic_progressions_l818_818476

def is_reciprocal_sum (n : ℕ) : Prop :=
  ∃ (p q : ℕ), n ≠ 0 ∧ 3 * p * q = n * (p + q)

def S : set ℕ := { n | ¬ is_reciprocal_sum n }

theorem S_is_not_finitely_union_of_arithmetic_progressions :
  ¬ ( ∃ (A : list (ℕ × ℕ)), ∀ n ∈ S, ∃ a b ∈ A, ∃ k : ℕ, n = a + b * k ) := 
sorry

end S_is_not_finitely_union_of_arithmetic_progressions_l818_818476


namespace lower_right_square_is_one_l818_818918

open Matrix

def grid_initial : Matrix (Fin 5) (Fin 5) (Option ℕ) :=
  ![ ![ some 1, none, none, none, some 2 ],
     ![ none, some 3, none, none, none ],
     ![ some 5, none, some 4, none, none ],
     ![ none, none, some 1, some 3, none ],
     ![ none, none, none, none, none ] ]

def is_valid_grid (grid : Matrix (Fin 5) (Fin 5) (Option ℕ)) : Prop :=
  (∀ i, Finset.univ.map ⟨fun j => grid i j, sorry⟩ = {1, 2, 3, 4, 5}) ∧
  (∀ j, Finset.univ.map ⟨fun i => grid i j, sorry⟩ = {1, 2, 3, 4, 5})

theorem lower_right_square_is_one :
  ∃ grid : Matrix (Fin 5) (Fin 5) (Option ℕ),
  grid_initial ⊆ grid ∧
  is_valid_grid grid ∧
  grid ⟨4, sorry⟩ ⟨4, sorry⟩ = some 1 :=
sorry

end lower_right_square_is_one_l818_818918


namespace cement_percentage_first_concrete_correct_l818_818175

open Real

noncomputable def cement_percentage_of_first_concrete := 
  let total_weight := 4500 
  let cement_percentage := 10.8 / 100
  let weight_each_type := 1125
  let total_cement_weight := cement_percentage * total_weight
  let x := 2.0 / 100
  let y := 21.6 / 100 - x
  (weight_each_type * x + weight_each_type * y = total_cement_weight) →
  (x = 2.0 / 100)

theorem cement_percentage_first_concrete_correct :
  cement_percentage_of_first_concrete := sorry

end cement_percentage_first_concrete_correct_l818_818175


namespace geometric_series_sum_l818_818538

theorem geometric_series_sum :
  let b1 := (3 : ℚ) / 4 in
  let r := (3 : ℚ) / 4 in
  let n := 15 in
  let result := (∑ i in finset.range n, b1 * r^i) in
  result = 3177878751 / 1073741824 :=
by
  let b1 := (3 : ℚ) / 4
  let r := (3 : ℚ) / 4
  let n := 15
  let result := (∑ i in finset.range n, b1 * r^i)
  exact (∑ i in finset.range 15, (3 : ℚ) / 4 * ((3 : ℚ) / 4)^i) = 3177878751 / 1073741824
  sorry

end geometric_series_sum_l818_818538


namespace total_men_employed_l818_818192

/--
A work which could be finished in 11 days was finished 3 days earlier 
after 10 more men joined. Prove that the total number of men employed 
to finish the work earlier is 37.
-/
theorem total_men_employed (x : ℕ) (h1 : 11 * x = 8 * (x + 10)) : x = 27 ∧ 27 + 10 = 37 := by
  sorry

end total_men_employed_l818_818192


namespace false_statement_c_l818_818971

theorem false_statement_c (a b c : Line) (α : Plane)
  (hA : ∀ (P : Point), ∃! (d : Line), P ∉ a ∧ d ∥ a)
  (hB : ∀ (P Q : Line), (P ⊥ α → Q ⊥ α → P ∥ Q))
  (hD : ∀ (l : Line), (l ⊥ (Line1 : Line) ∧ l ⊥ (Line2 : Line) ↔ Line1 ∩ Line2 = Point → l ⊥ α)) :
  ¬ (∀ (a b c : Line), (a ⊥ c → b ⊥ c → a ∥ b)) :=
begin
  -- Proof goes here
  sorry
end

end false_statement_c_l818_818971


namespace nth_sum_eq_square_l818_818324

theorem nth_sum_eq_square (n : ℕ) : 
  (∑ k in finset.range (3 * n - n + 1), k + n) = (2 * n - 1) ^ 2 :=
by
  sorry

end nth_sum_eq_square_l818_818324


namespace contrapositive_x_squared_eq_one_l818_818891

theorem contrapositive_x_squared_eq_one (x : ℝ) : 
  (x^2 = 1 → x = 1 ∨ x = -1) ↔ (x ≠ 1 ∧ x ≠ -1 → x^2 ≠ 1) := by
  sorry

end contrapositive_x_squared_eq_one_l818_818891


namespace factorial_units_digit_l818_818142

theorem factorial_units_digit (n : ℕ) (h : n ≥ 79) : (nat.factorial n) % 10 = 0 :=
by
  sorry

end factorial_units_digit_l818_818142


namespace part_a_part_b_l818_818825

variable (a b c : ℤ)
variable (h : a + b + c = 0)

theorem part_a : (a^4 + b^4 + c^4) % (a^2 + b^2 + c^2) = 0 :=
by
  -- proof goes here
  sorry

theorem part_b : (a^100 + b^100 + c^100) % (a^2 + b^2 + c^2) = 0 :=
by
  -- proof goes here
  sorry

end part_a_part_b_l818_818825


namespace necessary_but_not_sufficient_condition_l818_818204

theorem necessary_but_not_sufficient_condition :
  (∀ m, -1 < m ∧ m < 5 → ∀ x, 
    x^2 - 2 * m * x + m^2 - 1 = 0 → -2 < x ∧ x < 4) ∧ 
  ¬ (∀ m, -1 < m ∧ m < 5 → ∀ x, 
    x^2 - 2 * m * x + m^2 - 1 = 0 → -2 < x ∧ x < 4) :=
sorry

end necessary_but_not_sufficient_condition_l818_818204


namespace unstable_shape_is_rectangle_l818_818970

-- Definitions based on the problem conditions
def acute_triangle := ∀ (a b c : ℝ), (a + b + c = 180) ∧ (a < 90) ∧ (b < 90) ∧ (c < 90)

def rectangle := ∀ (a b : ℝ), (a > 0) ∧ (b > 0) ∧ (a = b → False)

def right_triangle := ∀ (a b : ℝ), (a >= 0) ∧ (b >= 0) ∧ (a + b = 90)

def isosceles_triangle := ∀ (a b c : ℝ), (a + b + c = 180) ∧ ((a = b) ∨ (b = c) ∨ (a = c))

-- The mathematical statement we aim to prove
theorem unstable_shape_is_rectangle (A B C D : Type)
  (acute_triangle : A)
  (rectangle : B)
  (right_triangle : C)
  (isosceles_triangle : D)
  : B = "unstable_shape" := 
sorry

end unstable_shape_is_rectangle_l818_818970


namespace min_value_sum_distances_l818_818863

open Real

-- Define the parabola's equation
def parabola (x y : ℝ) : Prop := x^2 = -4 * y

-- Define point A
def A : ℝ × ℝ := (1, 0)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := √((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the directrix of the parabola x^2 = -4y, which is y = 1
def directrix (y : ℝ) : Prop := y = 1

-- Define the focus of the parabola, F(0, -1)
def focus : ℝ × ℝ := (0, -1)

-- Lemma to prove the minimum value of the sum of the distances
theorem min_value_sum_distances (P : ℝ × ℝ) (hP : parabola P.1 P.2) :
  distance P A + abs (P.2 - 1) = √2 :=
sorry

end min_value_sum_distances_l818_818863


namespace probability_diagonals_intersect_inside_decagon_l818_818931

/-- Two diagonals of a regular decagon are chosen. 
  What is the probability that their intersection lies inside the decagon?
-/
theorem probability_diagonals_intersect_inside_decagon : 
  let num_diagonals := 35
  let num_pairs := num_diagonals * (num_diagonals - 1) / 2
  let num_intersecting_pairs := 210
  let probability := num_intersecting_pairs / num_pairs
  probability = 42 / 119 :=
by
  -- Definitions based on the conditions
  let num_diagonals := (10 * (10 - 3)) / 2
  let num_pairs := num_diagonals * (num_diagonals - 1) / 2
  let num_intersecting_pairs := 210

  -- Simplified probability
  let probability := num_intersecting_pairs / num_pairs

  -- Sorry used to skip the proof
  sorry

end probability_diagonals_intersect_inside_decagon_l818_818931


namespace mass_of_man_l818_818551

def density_of_water : ℝ := 1000  -- kg/m³
def boat_length : ℝ := 4  -- meters
def boat_breadth : ℝ := 2  -- meters
def sinking_depth : ℝ := 0.01  -- meters (1 cm)

theorem mass_of_man
  (V : ℝ := boat_length * boat_breadth * sinking_depth)
  (m : ℝ := V * density_of_water) :
  m = 80 :=
by
  sorry

end mass_of_man_l818_818551


namespace min_value_of_z_l818_818948

theorem min_value_of_z :
  ∃ (x y : ℝ), let z := x^2 + 3*y^2 + 8*x - 6*y + x*y + 22 in
  z = 3 :=
by
  sorry

end min_value_of_z_l818_818948


namespace probability_intersecting_diagonals_l818_818925

def number_of_vertices := 10

def number_of_diagonals : ℕ := Nat.choose number_of_vertices 2 - number_of_vertices

def number_of_ways_choose_two_diagonals := Nat.choose number_of_diagonals 2

def number_of_sets_of_intersecting_diagonals : ℕ := Nat.choose number_of_vertices 4

def intersection_probability : ℚ :=
  (number_of_sets_of_intersecting_diagonals : ℚ) / (number_of_ways_choose_two_diagonals : ℚ)

theorem probability_intersecting_diagonals :
  intersection_probability = 42 / 119 :=
by
  sorry

end probability_intersecting_diagonals_l818_818925


namespace geometry_progressions_not_exhaust_nat_l818_818074

theorem geometry_progressions_not_exhaust_nat :
  ∃ (g : Fin 1975 → ℕ → ℕ), 
  (∀ i : Fin 1975, ∃ (a r : ℤ), ∀ n : ℕ, g i n = (a * r^n)) ∧
  (∃ m : ℕ, ∀ i : Fin 1975, ∀ n : ℕ, m ≠ g i n) :=
sorry

end geometry_progressions_not_exhaust_nat_l818_818074


namespace part_I_part_II_l818_818740

noncomputable def f : ℝ → ℝ := 
  λ x, if x < 1 then 2^(-x) - 1/4 
       else 1/4 + Real.log x / Real.log 4

theorem part_I (x : ℝ) : f x ≥ 1/4 := 
by 
  sorry

theorem part_II (x₀ : ℝ) : f x₀ = 3/4 ↔ x₀ = 0 ∨ x₀ = 2 := 
by 
  sorry

end part_I_part_II_l818_818740


namespace find_value_of_m_l818_818777

theorem find_value_of_m (m : ℤ) (x : ℤ) (h : (x - 3 ≠ 0) ∧ (x = 3)) : 
  ((x - 1) / (x - 3) = m / (x - 3)) → m = 2 :=
by
  sorry

end find_value_of_m_l818_818777


namespace find_math_marks_l818_818629

theorem find_math_marks
  (e p c b : ℕ)
  (n : ℕ)
  (a : ℚ)
  (M : ℕ) :
  e = 96 →
  p = 82 →
  c = 87 →
  b = 92 →
  n = 5 →
  a = 90.4 →
  (a * n = (e + p + c + b + M)) →
  M = 95 :=
by intros
   sorry

end find_math_marks_l818_818629


namespace value_of_g_l818_818372

-- Defining the function g and its property
def g (x : ℝ) : ℝ := 5

-- Theorem to prove g(x - 3) = 5 for any real number x
theorem value_of_g (x : ℝ) : g (x - 3) = 5 := by
  sorry

end value_of_g_l818_818372


namespace P_inter_Q_l818_818838

open Set

def P : Set ℝ := {x | x^2 - 9 < 0}
def Q : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3 }

theorem P_inter_Q :
  {x : ℤ | (x : ℝ)^2 - 9 < 0} ∩ Q = {-1, 0, 1, 2} :=
by
  sorry

end P_inter_Q_l818_818838


namespace unique_polynomial_solution_l818_818641

theorem unique_polynomial_solution (P : ℝ[X]) :
  (P.eval 2017 = 2016) ∧ (∀ x : ℝ, (P.eval x + 1)^2 = P.eval (x^2 + 1)) → P = X - 1 :=
by
  intro h
  sorry

end unique_polynomial_solution_l818_818641


namespace part_a_answer_part_b_answer_l818_818569

noncomputable def part_a_problem : Prop :=
  ∃! (x k : ℕ), x > 0 ∧ k > 0 ∧ 3^k - 1 = x^3

noncomputable def part_b_problem (n : ℕ) : Prop :=
  n > 1 ∧ n ≠ 3 → ∀ (x k : ℕ), ¬ (x > 0 ∧ k > 0 ∧ 3^k - 1 = x^n)

theorem part_a_answer : part_a_problem :=
  sorry

theorem part_b_answer (n : ℕ) : part_b_problem n :=
  sorry

end part_a_answer_part_b_answer_l818_818569


namespace number_of_fir_trees_is_11_l818_818272

theorem number_of_fir_trees_is_11 
  (N : ℕ)
  (Anya : N = 15)
  (Borya : N % 11 = 0)
  (Vera : N < 25)
  (Gena : N % 22 = 0)
  (OneBoyOneGirlTrue : (Anya ∨ Borya) ∧ (Vera ∨ Gena) ∧ (¬Anya ∨ ¬Borya) ∧ (¬Vera ∨ ¬Gena)) :
  N = 11 := 
sorry

end number_of_fir_trees_is_11_l818_818272


namespace least_positive_integer_with_six_factors_is_18_l818_818137

-- Define the least positive integer with exactly six distinct positive factors.
def least_positive_with_six_factors (n : ℕ) : Prop :=
  (∀ d : ℕ, d ∣ n → d > 0) ∧ (finset.card (finset.filter (λ d, d ∣ n) (finset.range (n + 1)))) = 6

-- Prove that the least positive integer with exactly six distinct positive factors is 18.
theorem least_positive_integer_with_six_factors_is_18 : (∃ n : ℕ, least_positive_with_six_factors n ∧ n = 18) :=
sorry


end least_positive_integer_with_six_factors_is_18_l818_818137


namespace find_distance_l818_818086

-- Conditions: total cost, base price, cost per mile
variables (total_cost base_price cost_per_mile : ℕ)

-- Definition of the distance as per the problem
def distance_from_home_to_hospital (total_cost base_price cost_per_mile : ℕ) : ℕ :=
  (total_cost - base_price) / cost_per_mile

-- Given values:
def total_cost_value : ℕ := 23
def base_price_value : ℕ := 3
def cost_per_mile_value : ℕ := 4

-- The theorem that encapsulates the problem statement
theorem find_distance :
  distance_from_home_to_hospital total_cost_value base_price_value cost_per_mile_value = 5 :=
by
  -- Placeholder for the proof
  sorry

end find_distance_l818_818086


namespace fg_of_neg3_eq_3_l818_818003

def f (x : ℝ) : ℝ := 4 * x - 1
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem fg_of_neg3_eq_3 : f (g (-3)) = 3 :=
by
  sorry

end fg_of_neg3_eq_3_l818_818003


namespace sin_of_7pi_over_6_l818_818663

theorem sin_of_7pi_over_6 : Real.sin (7 * Real.pi / 6) = -1 / 2 :=
by
  sorry

end sin_of_7pi_over_6_l818_818663


namespace probability_spring_mid_autumn_selected_l818_818920

theorem probability_spring_mid_autumn_selected :
  (∃ (S : Finset ℕ), S.card = 3 ∧ {0, 4} ⊆ S ∧ Set.toFinset {0, 1, 2, 3, 4} = {0, 1, 2, 3, 4}) →
  (∃ (n m : ℕ), n = Nat.choose 5 3 ∧ m = Nat.choose 2 2 * Nat.choose 3 1 ∧ 
  (m : ℚ) / (n : ℚ) = 3 / 10) :=
by
  intro H
  use 10 -- n
  use 3 -- m
  split
  · exact Nat.choose 5 3
  split
  · exact Nat.choose 2 2 * Nat.choose 3 1
  · norm_num -- 3 / 10
  sorry

end probability_spring_mid_autumn_selected_l818_818920


namespace transformed_independence_l818_818837

variables {E : Type*} {G : Type*} {n : ℕ}
variables (X : Fin n → E) (g : Fin n → E → G)
variables (measurable_spaces_E : Fin n → MeasurableSpace E)
variables (measurable_spaces_G : Fin n → MeasurableSpace G)
variables (independent_X : independent (λ i, σ X i))
variables (measurability : ∀ i, Measurable (g i))

theorem transformed_independence :
  independent (λ i, σ (g i (X i))) :=
sorry

end transformed_independence_l818_818837


namespace number_of_trees_l818_818247

theorem number_of_trees {N : ℕ} 
  (Anya : N = 15) 
  (Borya : N % 11 = 0) 
  (Vera : N < 25) 
  (Gena : N % 22 = 0) 
  (truth_conditions : (Anya ∨ Borya ∨ Vera ∨ Gena) ∧ ∃! p, p) : 
  N = 11 :=
sorry

end number_of_trees_l818_818247


namespace polynomial_inequality_l818_818460

variables {R : Type*} [linear_ordered_field R]
variables (F G : R → R)

theorem polynomial_inequality (h : ∀ x : R, F (F x) > G (F x) ∧ G (F x) > G (G x)) :
  ∀ x : R, F x > G x :=
begin
  sorry
end

end polynomial_inequality_l818_818460


namespace evaluate_expression_l818_818963

theorem evaluate_expression : 4 * (9 - 3)^2 - 8 = 136 := by
  sorry

end evaluate_expression_l818_818963


namespace correct_solutions_l818_818469

theorem correct_solutions (x y z t : ℕ) : 
  (x^2 + t^2) * (z^2 + y^2) = 50 → 
  (x = 1 ∧ y = 1 ∧ z = 2 ∧ t = 3) ∨ 
  (x = 3 ∧ y = 2 ∧ z = 1 ∧ t = 1) ∨ 
  (x = 4 ∧ y = 1 ∧ z = 3 ∧ t = 1) ∨ 
  (x = 1 ∧ y = 3 ∧ z = 4 ∧ t = 1) :=
sorry

end correct_solutions_l818_818469


namespace exists_perpendicular_line_l818_818547

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

structure DirectionVector :=
  (dx : ℝ)
  (dy : ℝ)
  (dz : ℝ)

noncomputable def parametric_line_through_point 
  (P : Point3D) 
  (d : DirectionVector) : Prop :=
  ∀ t : ℝ, ∃ x y z : ℝ, 
  x = P.x + d.dx * t ∧
  y = P.y + d.dy * t ∧
  z = P.z + d.dz * t

theorem exists_perpendicular_line : 
  ∃ d : DirectionVector, 
    (d.dx * 2 + d.dy * 3 - d.dz = 0) ∧ 
    (d.dx * 4 - d.dy * -1 + d.dz * 3 = 0) ∧ 
    parametric_line_through_point 
      ⟨3, -2, 1⟩ d :=
  sorry

end exists_perpendicular_line_l818_818547


namespace P_Q_diff_is_285_l818_818699

def is_P_type (a b c d : ℕ) : Prop :=
  a > b ∧ b < c ∧ c > d

def is_Q_type (a b c d : ℕ) : Prop :=
  a < b ∧ b > c ∧ c < d

def count_P_Q_diff := 
  (filter (λ x : ℕ × ℕ × ℕ × ℕ, is_P_type x.1 x.2.fst x.2.snd.fst x.2.snd.snd) (list.fin_range 10000)).length -
  (filter (λ x : ℕ × ℕ × ℕ × ℕ, is_Q_type x.1 x.2.fst x.2.snd.fst x.2.snd.snd) (list.fin_range 10000)).length
  
theorem P_Q_diff_is_285 : count_P_Q_diff = 285 := by sorry

end P_Q_diff_is_285_l818_818699


namespace problem_statement_l818_818228

open Real

def my_floor (x : ℝ) : ℤ := int.floor x

def abs_val (x : ℝ) : ℝ := |x|

def expression : ℤ :=
  my_floor (abs_val (-7.6)) + abs (my_floor (-7.6))

theorem problem_statement : expression = 15 :=
by
  sorry

end problem_statement_l818_818228


namespace least_positive_integer_with_six_factors_l818_818117

-- Define what it means for a number to have exactly six distinct positive factors
def hasExactlySixFactors (n : ℕ) : Prop :=
  (n.factorization.support.card = 2 ∧ (n.factorization.values' = [2, 1])) ∨
  (n.factorization.support.card = 1 ∧ (n.factorization.values' = [5]))

-- The main theorem statement
theorem least_positive_integer_with_six_factors : ∃ n : ℕ, hasExactlySixFactors n ∧ ∀ m : ℕ, (hasExactlySixFactors m → n ≤ m) :=
  exists.intro 12 (and.intro
    (show hasExactlySixFactors 12, by sorry)
    (show ∀ m : ℕ, hasExactlySixFactors m → 12 ≤ m, by sorry))

end least_positive_integer_with_six_factors_l818_818117


namespace quadratic_equation_transformation_l818_818871

theorem quadratic_equation_transformation (x : ℝ) :
  (-5 * x ^ 2 = 2 * x + 10) →
  (x ^ 2 + (2 / 5) * x + 2 = 0) :=
by
  intro h
  sorry

end quadratic_equation_transformation_l818_818871


namespace calculate_z_l818_818379

-- Given conditions
def equally_spaced : Prop := true -- assume equally spaced markings do exist
def total_distance : ℕ := 35
def number_of_steps : ℕ := 7
def step_length : ℕ := total_distance / number_of_steps
def starting_point : ℕ := 10
def steps_forward : ℕ := 4

-- Theorem to prove
theorem calculate_z (h1 : equally_spaced)
(h2 : step_length = 5)
: starting_point + (steps_forward * step_length) = 30 :=
by sorry

end calculate_z_l818_818379


namespace prove_x_minus_y_squared_l818_818739

noncomputable section

variables {x y a b : ℝ}

theorem prove_x_minus_y_squared (h1 : x * y = b) (h2 : x / y + y / x = a) : (x - y) ^ 2 = a * b - 2 * b := 
  sorry

end prove_x_minus_y_squared_l818_818739


namespace Marie_finish_time_l818_818011

def Time := Nat × Nat -- Represents time as (hours, minutes)

def start_time : Time := (9, 0)
def finish_two_tasks_time : Time := (11, 20)
def total_tasks : Nat := 4

def minutes_since_start (t : Time) : Nat :=
  let (h, m) := t
  (h - 9) * 60 + m

def calculate_finish_time (start: Time) (two_tasks_finish: Time) (total_tasks: Nat) : Time :=
  let duration_two_tasks := minutes_since_start two_tasks_finish
  let duration_each_task := duration_two_tasks / 2
  let total_time := duration_each_task * total_tasks
  let total_minutes_after_start := total_time + minutes_since_start start
  let finish_hour := 9 + total_minutes_after_start / 60
  let finish_minute := total_minutes_after_start % 60
  (finish_hour, finish_minute)

theorem Marie_finish_time :
  calculate_finish_time start_time finish_two_tasks_time total_tasks = (13, 40) :=
by
  sorry

end Marie_finish_time_l818_818011


namespace number_of_zeros_in_decimal_expansion_of_square_l818_818206

theorem number_of_zeros_in_decimal_expansion_of_square : 
  ∀ N, (N + 1 = 10^8) → (N + 1)^2 = 10^16 → number_of_zeros_in_decimal_expansion ((N + 1)^2) = 16 :=
by
  intros N h₁ h₂
  sorry

end number_of_zeros_in_decimal_expansion_of_square_l818_818206


namespace nat_number_of_the_form_l818_818867

theorem nat_number_of_the_form (a b : ℕ) (h : ∃ (a b : ℕ), a * a * 3 + b * b * 32 = n) :
  ∃ (a' b' : ℕ), a' * a' * 3 + b' * b' * 32 = 97 * n  :=
  sorry

end nat_number_of_the_form_l818_818867


namespace sum_of_digits_of_N_l818_818601

theorem sum_of_digits_of_N (N : ℕ) (h : N * (N + 1) = 10100) : 
  N = 100 → (100.digitSum = 1) :=
by
  sorry

end sum_of_digits_of_N_l818_818601


namespace proof_sufficient_not_necessary_l818_818397

noncomputable def is_geometric_seq (a : ℕ → ℝ) (n1 n2 n3 : ℕ) : Prop :=
  a n2 ^ 2 = a n1 * a n3

noncomputable def arithmetic_seq (a1 : ℝ) (d : ℝ) : ℕ → ℝ
  | 0     => a1
  | (n+1) => a1 + d * n

theorem proof_sufficient_not_necessary:
  ∀ (d : ℝ),
  let a := arithmetic_seq 2 d in
  (is_geometric_seq a 1 2 5 → d = 4) ∧ (is_geometric_seq a 1 2 5 → ∃ d', d' ≠ 4 ∧ is_geometric_seq (arithmetic_seq 2 d') 1 2 5) :=
by
  sorry

end proof_sufficient_not_necessary_l818_818397


namespace complex_quadrant_l818_818488

theorem complex_quadrant (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 1) : 
  let z := complex.mk (real.arccos x - π) (2 - 2^x) in 
  z.im ≥ 0 ∧ z.re ≤ 0 :=
by
  let z := complex.mk (real.arccos x - π) (2 - 2^x)
  sorry

end complex_quadrant_l818_818488


namespace circumcenter_on_line_l818_818809

noncomputable def triangle_circumcenter_on_line
  (A B C A1 B1 C1 K L: Type)
  [triangle A B C]
  [altitude AA1 A B C]
  [altitude BB1 B1 B C]
  [altitude CC1 C1 B C]
  [parallel KL CC1]
  [on_line K BC]
  [on_line L B1C1] :
  Prop :=
  circumcenter A1 K L ∈ AC

theorem circumcenter_on_line
  {A B C A1 B1 C1 K L : Type}
  [triangle A B C]
  [altitude AA1 A B C]
  [altitude BB1 B1 B C]
  [altitude CC1 C1 B C]
  [parallel KL CC1]
  [on_line K BC]
  [on_line L B1C1]
  :
  triangle_circumcenter_on_line A B C A1 B1 C1 K L :=
by
  sorry

end circumcenter_on_line_l818_818809


namespace expression_evaluation_l818_818141

theorem expression_evaluation : 
  3 / 5 * ((2 / 3 + 3 / 8) / 2) - 1 / 16 = 1 / 4 := 
by
  sorry

end expression_evaluation_l818_818141


namespace circle_condition_l818_818378

theorem circle_condition (k : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 4 * x + 2 * y + 5 * k = 0) ↔ k < 1 := 
sorry

end circle_condition_l818_818378


namespace number_of_trees_is_eleven_l818_818259

variables (N : ℕ)

-- Conditions
def Anya (N : ℕ) := N = 15
def Borya (N : ℕ) := 11 ∣ N
def Vera (N : ℕ) := N < 25
def Gena (N : ℕ) := 22 ∣ N

axiom OneBoyOneGirlTruth :
  (∃ (b : Prop) (g : Prop),
    (b ∨ ¬ b) ∧ (g ∨ ¬ g) ∧
    ((b = (Borya N ∨ Gena N)) ∧ (g = (Anya N ∨ Vera N)) ∧
     (b ↔ ¬g) ∧
     ((Anya N ∨ ¬Vera N) ∨ (¬Anya N ∨ Vera N)) ∧
     (Anya N = (N = 15)) ∧
     (Borya N = (11 ∣ N)) ∧
     (Vera N = (N < 25)) ∧
     (Gena N = (22 ∣ N))))

theorem number_of_trees_is_eleven: N = 11 :=
sorry

end number_of_trees_is_eleven_l818_818259


namespace cost_per_minute_l818_818241

theorem cost_per_minute (monthly_fee total_bill billed_minutes : ℝ) (h_monthly_fee : monthly_fee = 2) (h_total_bill : total_bill = 23.36) (h_billed_minutes : billed_minutes = 178) : 
  (total_bill - monthly_fee) / billed_minutes = 0.12 :=
by
  rw [h_monthly_fee, h_total_bill, h_billed_minutes]
  norm_num
  sorry

end cost_per_minute_l818_818241


namespace unique_number_outside_range_f_l818_818720

noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_number_outside_range_f (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
  (h5 : f a b c d 19 = 19) (h6 : f a b c d 97 = 97)
  (h7 : ∀ x, x ≠ -d / c → f a b c d (f a b c d x) = x) : 
  ∀ y : ℝ, y ≠ 58 → ∃ x : ℝ, f a b c d x ≠ y :=
sorry

end unique_number_outside_range_f_l818_818720


namespace optionA_is_01_intersection_function_optionB_is_01_intersection_function_l818_818778

namespace IntersectionFunction

def isIntersectionFunction (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  let domain := {x : ℝ | ∃ y, f x = y}
  let range := {y : ℝ | ∃ x, f x = y}
  ∀ x, (x ∈ domain ∩ range ↔ a ≤ x ∧ x ≤ b)

theorem optionA_is_01_intersection_function :
  isIntersectionFunction (λ x, Real.sqrt (1 - x)) 0 1 :=
by
  -- Proof to be provided
  sorry

theorem optionB_is_01_intersection_function :
  isIntersectionFunction (λ x, 2 * Real.sqrt x - x) 0 1 :=
by
  -- Proof to be provided
  sorry

end IntersectionFunction

end optionA_is_01_intersection_function_optionB_is_01_intersection_function_l818_818778


namespace probability_of_exactly_two_dice_showing_3_l818_818696

-- Definition of the problem conditions
def n_dice : ℕ := 5
def sides : ℕ := 5
def prob_showing_3 : ℚ := 1/5
def prob_not_showing_3 : ℚ := 4/5
def way_to_choose_2_of_5 : ℕ := Nat.choose 5 2

-- Lean proof problem statement
theorem probability_of_exactly_two_dice_showing_3 : 
  (10 : ℚ) * (prob_showing_3 ^ 2) * (prob_not_showing_3 ^ 3) = 640 / 3125 := 
by sorry

end probability_of_exactly_two_dice_showing_3_l818_818696


namespace johns_total_spending_l818_818604

theorem johns_total_spending
    (online_phone_price : ℝ := 2000)
    (phone_price_increase : ℝ := 0.02)
    (phone_case_price : ℝ := 35)
    (screen_protector_price : ℝ := 15)
    (accessories_discount : ℝ := 0.05)
    (sales_tax : ℝ := 0.06) :
    let store_phone_price := online_phone_price * (1 + phone_price_increase)
    let regular_accessories_price := phone_case_price + screen_protector_price
    let discounted_accessories_price := regular_accessories_price * (1 - accessories_discount)
    let pre_tax_total := store_phone_price + discounted_accessories_price
    let total_spending := pre_tax_total * (1 + sales_tax)
    total_spending = 2212.75 :=
by
    sorry

end johns_total_spending_l818_818604


namespace repeating_decimal_to_fraction_l818_818543

theorem repeating_decimal_to_fraction (x : ℚ) (h : x = 0.528528528528...) :
  let fraction := 528 / 999 in
  (fraction.num / fraction.denom).num + (fraction.num / fraction.denom).denom = 509 :=
by
  sorry -- proof skipped

end repeating_decimal_to_fraction_l818_818543


namespace ratio_Jake_sister_l818_818375

theorem ratio_Jake_sister (Jake_weight : ℕ) (total_weight : ℕ) (weight_loss : ℕ) (expected_ratio : ℕ) :
  Jake_weight = 113 →
  total_weight = 153 →
  weight_loss = 33 →
  expected_ratio = 2 →
  (Jake_weight - weight_loss) / (total_weight - Jake_weight) = expected_ratio :=
by
  intros hJake hTotal hLoss hRatio
  sorry

end ratio_Jake_sister_l818_818375


namespace imaginary_unit_property_l818_818915

theorem imaginary_unit_property (i : ℂ) (h : i^2 = -1) : (i - 1)^2 = -2i :=
by
  sorry

end imaginary_unit_property_l818_818915


namespace min_shift_sin_cos_l818_818872

noncomputable def min_shift (ϕ : ℝ) : ℝ :=
  if ϕ > 0 then ϕ else 0

theorem min_shift_sin_cos : ∀ ϕ : ℝ, ϕ > 0 →
  ∃ k : ℤ, ϕ = k * Real.pi + Real.pi / 12 ∧ k * Real.pi + Real.pi / 12 > 0 :=
begin
  intro ϕ,
  intro hϕ,
  use 0,
  split,
  { rw zero_mul,
    simp, },
  { exact Real.pi_div_pos (zero_lt_two : 0 < 2), },
end

end min_shift_sin_cos_l818_818872


namespace seven_digit_permutations_count_l818_818763

theorem seven_digit_permutations_count : 
  let digits := {1, 2, 3, 4, 5, 6, 7}
  let is_odd (n: ℕ) : Prop := n % 2 = 1
  let odd_digits := {d ∈ digits | is_odd d}
  let total_permutations := Nat.factorial 7
  let permutations_with_all_four_odd_adjacent := Nat.factorial 4 * Nat.factorial 3
  let permutations_with_three_odd_adjacent := Nat.factorial 3 * 4 * 12 * Nat.factorial 3,
  ∑ (perms in permutations_with_all_four_odd_adjacent ∪ permutations_with_three_odd_adjacent) = 2736 :=
by {
  let total_permutations := Nat.factorial 7,
  let unwanted_permutations := (Nat.factorial 4 * Nat.factorial 3) + (Nat.factorial 3 * 4 * 12 * Nat.factorial 3),
  exact total_permutations - unwanted_permutations = 2736,
  sorry
}

end seven_digit_permutations_count_l818_818763


namespace chocolate_bars_in_large_box_l818_818181

theorem chocolate_bars_in_large_box (n_small_boxes : ℕ) (n_chocolate_per_box : ℕ) (h1 : n_small_boxes = 15) (h2 : n_chocolate_per_box = 25) :
  n_small_boxes * n_chocolate_per_box = 375 :=
by
  rw [h1, h2]
  norm_num
  sorry

end chocolate_bars_in_large_box_l818_818181


namespace probability_diagonals_intersect_inside_decagon_l818_818932

/-- Two diagonals of a regular decagon are chosen. 
  What is the probability that their intersection lies inside the decagon?
-/
theorem probability_diagonals_intersect_inside_decagon : 
  let num_diagonals := 35
  let num_pairs := num_diagonals * (num_diagonals - 1) / 2
  let num_intersecting_pairs := 210
  let probability := num_intersecting_pairs / num_pairs
  probability = 42 / 119 :=
by
  -- Definitions based on the conditions
  let num_diagonals := (10 * (10 - 3)) / 2
  let num_pairs := num_diagonals * (num_diagonals - 1) / 2
  let num_intersecting_pairs := 210

  -- Simplified probability
  let probability := num_intersecting_pairs / num_pairs

  -- Sorry used to skip the proof
  sorry

end probability_diagonals_intersect_inside_decagon_l818_818932


namespace find_x_l818_818705

def vector := (ℝ × ℝ)

def collinear (u v : vector) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

def a : vector := (1, 2)
def b (x : ℝ) : vector := (x, 1)
def a_minus_b (x : ℝ) : vector := ((1 - x), 1)

theorem find_x (x : ℝ) (h : collinear a (a_minus_b x)) : x = 1/2 :=
by
  sorry

end find_x_l818_818705


namespace largest_stores_visited_l818_818560

theorem largest_stores_visited 
  (stores : ℕ) (total_visits : ℕ) (shoppers : ℕ) 
  (two_store_visitors : ℕ) (min_visits_per_person : ℕ)
  (h1 : stores = 8)
  (h2 : total_visits = 22)
  (h3 : shoppers = 12)
  (h4 : two_store_visitors = 8)
  (h5 : min_visits_per_person = 1)
  : ∃ (max_stores : ℕ), max_stores = 3 := 
by 
  -- Define the exact details given in the conditions
  have h_total_two_store_visits : two_store_visitors * 2 = 16 := by sorry
  have h_remaining_visits : total_visits - 16 = 6 := by sorry
  have h_remaining_shoppers : shoppers - two_store_visitors = 4 := by sorry
  have h_each_remaining_one_visit : 4 * 1 = 4 := by sorry
  -- Prove the largest number of stores visited by any one person is 3
  have h_max_stores : 1 + 2 = 3 := by sorry
  exact ⟨3, h_max_stores⟩

end largest_stores_visited_l818_818560


namespace negation_of_universal_prop_l818_818057

theorem negation_of_universal_prop : 
  (¬ (∀ (x : ℝ), x ^ 2 ≥ 0)) ↔ (∃ (x : ℝ), x ^ 2 < 0) :=
by sorry

end negation_of_universal_prop_l818_818057


namespace bill_amount_each_person_shared_l818_818068

noncomputable def total_bill : ℝ := 139.00
noncomputable def tip_percentage : ℝ := 0.10
noncomputable def num_people : ℝ := 7.00

noncomputable def tip : ℝ := tip_percentage * total_bill
noncomputable def total_bill_with_tip : ℝ := total_bill + tip
noncomputable def amount_each_person_pays : ℝ := total_bill_with_tip / num_people

theorem bill_amount_each_person_shared :
  amount_each_person_pays = 21.84 := by
  -- proof goes here
  sorry

end bill_amount_each_person_shared_l818_818068


namespace geom_seq_increase_neither_suff_nor_nec_l818_818311

-- Define the sequence and its properties
variable {α : Type*} [OrderedField α]

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → α) (q : α) : Prop :=
  ∀ n, a (n+1) = q * a n

-- Define an increasing sequence
def is_increasing_sequence (a : ℕ → α) : Prop :=
  ∀ n, a n < a (n+1)

-- The theorem to prove
theorem geom_seq_increase_neither_suff_nor_nec (a : ℕ → α) (q : α) :
  is_geometric_sequence a q →
  (¬ (q > 1 → is_increasing_sequence a)) ∧
  (¬ (is_increasing_sequence a → q > 1)) :=
by
  sorry

end geom_seq_increase_neither_suff_nor_nec_l818_818311


namespace largest_of_a_b_c_d_e_l818_818738

theorem largest_of_a_b_c_d_e (a b c d e : ℝ)
  (h1 : a - 2 = b + 3)
  (h2 : a - 2 = c - 4)
  (h3 : a - 2 = d + 5)
  (h4 : a - 2 = e - 6) :
  e > a ∧ e > b ∧ e > c ∧ e > d :=
by
  sorry

end largest_of_a_b_c_d_e_l818_818738


namespace find_x_l818_818965

variables (a b c d x : ℕ)

theorem find_x (h1 : (a + x) / (b + x) = 4 * a / 3 * b)
               (h2 : a ≠ b)
               (h3 : b ≠ 0)
               (h4 : c = 4 * a)
               (h5 : d = 3 * b) :
x = (a * b) / (3 * b - 4 * a) :=
sorry

end find_x_l818_818965


namespace probability_diagonals_intersect_inside_decagon_l818_818933

/-- Two diagonals of a regular decagon are chosen. 
  What is the probability that their intersection lies inside the decagon?
-/
theorem probability_diagonals_intersect_inside_decagon : 
  let num_diagonals := 35
  let num_pairs := num_diagonals * (num_diagonals - 1) / 2
  let num_intersecting_pairs := 210
  let probability := num_intersecting_pairs / num_pairs
  probability = 42 / 119 :=
by
  -- Definitions based on the conditions
  let num_diagonals := (10 * (10 - 3)) / 2
  let num_pairs := num_diagonals * (num_diagonals - 1) / 2
  let num_intersecting_pairs := 210

  -- Simplified probability
  let probability := num_intersecting_pairs / num_pairs

  -- Sorry used to skip the proof
  sorry

end probability_diagonals_intersect_inside_decagon_l818_818933


namespace inverse_function_value_l818_818728

open Real

noncomputable def f (a : ℝ) : ℝ → ℝ := λ x => a ^ x
noncomputable def g (a : ℝ) : ℝ → ℝ := λ x => log x / log a

theorem inverse_function_value (a : ℝ) (h_exp : ∀ x, f a x = a ^ x) 
    (h : f a (1 + sqrt 3) * f a (1 - sqrt 3) = 9) : 
  g a (sqrt 10 + 1) + g a (sqrt 10 - 1) = 2 := by
  let h_inv := (h_exp (1 + sqrt 3)) * (h_exp (1 - sqrt 3))
  -- Given f is an exponential function
  have hf := h_inv
  -- Given f(1 + sqrt 3) * f(1 - sqrt 3) = 9
  have h1 : f a (1 + sqrt 3) * f a (1 - sqrt 3) = 9 := by
    exact h
  -- Calculate the logarithm sum in terms of the inverse function
  have h2 : g a (sqrt 10 + 1) + g a (sqrt 10 - 1) = log (sqrt 10 + 1) / log a + log (sqrt 10 - 1) / log a := by
    rw [g]
  -- Compute the specific value
  have h3 : log (sqrt 10 + 1) + log (sqrt 10 - 1) = 2 * log 3 := sorry

  sorry

end inverse_function_value_l818_818728


namespace decorative_integer_sum_l818_818704

-- Define the conditions
def georgina_paints (g p : ℕ) : Prop := p % g = 1 % g
def hugo_paints (h p : ℕ) : Prop := p % h = 3 % h
def isla_paints (i p : ℕ) : Prop := p % i = 2 % i

-- Define the main theorem with the conditions
theorem decorative_integer_sum (g h i : ℕ) (h_g : g > 0) (h_h : h > 0) (h_i : i > 0)
  (georgina_rule : ∀ p, p ∈ finset.range 60 → georgina_paints g p)
  (hugo_rule : ∀ p, p ∈ finset.range 60 → hugo_paints h p)
  (isla_rule : ∀ p, p ∈ finset.range 60 → isla_paints i p)
  (unique_paint : ∀ p, p ∈ finset.range 60 → (georgina_paints g p ∧ ¬hugo_paints h p ∧ ¬isla_paints i p) ∨
                                           (¬georgina_paints g p ∧ hugo_paints h p ∨ ¬isla_paints i p) ∨
                                           (¬georgina_paints g p ∧ ¬hugo_paints h p ∧ isla_paints i p))
  : 100 * g + 10 * h + i = 264 := 
sorry

end decorative_integer_sum_l818_818704


namespace find_side_c_l818_818405

noncomputable def triangle_side_c (A b S : ℝ) (c : ℝ) : Prop :=
  S = 0.5 * b * c * Real.sin A

theorem find_side_c :
  ∀ (c : ℝ), triangle_side_c (Real.pi / 3) 16 (64 * Real.sqrt 3) c → c = 16 :=
by
  sorry

end find_side_c_l818_818405


namespace smallest_integer_with_six_distinct_factors_l818_818133

noncomputable def least_pos_integer_with_six_factors : ℕ :=
  12

theorem smallest_integer_with_six_distinct_factors 
  (n : ℕ)
  (p q : ℕ)
  (a b : ℕ)
  (hp : prime p)
  (hq : prime q)
  (h_diff : p ≠ q)
  (h_n : n = p ^ a * q ^ b)
  (h_factors : (a + 1) * (b + 1) = 6) :
  n = least_pos_integer_with_six_factors :=
by
  sorry

end smallest_integer_with_six_distinct_factors_l818_818133


namespace alex_candles_left_l818_818605

theorem alex_candles_left (candles_start used_candles : ℕ) (h1 : candles_start = 44) (h2 : used_candles = 32) :
  candles_start - used_candles = 12 :=
by
  sorry

end alex_candles_left_l818_818605


namespace coefficient_of_x3_in_binomial_expansion_l818_818487

theorem coefficient_of_x3_in_binomial_expansion :
  (∃ n : ℕ, n = 80 ∧ (2 * (x : α) + 1) ^ 5 = ∑ i in finset.range 6, (nat.choose 5 i) * (2^i) * (x^i) * (1^(5-i)) ∧ i = 3) := sorry

end coefficient_of_x3_in_binomial_expansion_l818_818487


namespace quadratic_eq_standard_form_coefficients_l818_818207

-- Define initial quadratic equation
def initial_eq (x : ℝ) : Prop := (x + 5) * (x + 3) = 2 * x^2

-- Define the quadratic equation in standard form
def standard_form (x : ℝ) : Prop := x^2 - 8 * x - 15 = 0

-- Prove that given the initial equation, it can be converted to its standard form
theorem quadratic_eq_standard_form (x : ℝ) :
  initial_eq x → standard_form x := 
sorry

-- Verify the coefficients of the quadratic term, linear term, and constant term
theorem coefficients (x : ℝ) :
  initial_eq x → 
  (∀ a b c : ℝ, (a = 1) ∧ (b = -8) ∧ (c = -15) → standard_form x) :=
sorry

end quadratic_eq_standard_form_coefficients_l818_818207


namespace distinct_permutations_count_l818_818364

theorem distinct_permutations_count : 
  (finset.univ : finset (fin 5)).card = 5 →
  list_to_multiset [3, 3, 3, 7, 9].card = 5 →
  multiset.count 3 (list_to_multiset [3, 3, 3, 7, 9]) = 3 →
  multiset.count 7 (list_to_multiset [3, 3, 3, 7, 9]) = 1 →
  multiset.count 9 (list_to_multiset [3, 3, 3, 7, 9]) = 1 →
  (finset.univ : finset (fin 5)).card.factorial /
    ((multiset.count 3 (list_to_multiset [3, 3, 3, 7, 9])).factorial *
    (multiset.count 7 (list_to_multiset [3, 3, 3, 7, 9])).factorial *
    (multiset.count 9 (list_to_multiset [3, 3, 3, 7, 9])).factorial) = 20 :=
by
  sorry

end distinct_permutations_count_l818_818364


namespace largest_distance_l818_818947

noncomputable def max_distance_between_spheres 
  (c1 : ℝ × ℝ × ℝ) (r1 : ℝ) 
  (c2 : ℝ × ℝ × ℝ) (r2 : ℝ) : ℝ :=
dist c1 c2 + r1 + r2

theorem largest_distance 
  (c1 : ℝ × ℝ × ℝ) (r1 : ℝ) 
  (c2 : ℝ × ℝ × ℝ) (r2 : ℝ) 
  (h₁ : c1 = (-3, -15, 10))
  (h₂ : r1 = 24)
  (h₃ : c2 = (20, 18, -30))
  (h₄ : r2 = 95) : 
  max_distance_between_spheres c1 r1 c2 r2 = Real.sqrt 3218 + 119 := 
by
  sorry

end largest_distance_l818_818947


namespace problem1_problem2_problem2_gt_problem2_lt_problem3_l818_818353

noncomputable def a_n : ℕ → ℤ := λ n, n^2 - n - 30

/-- Problem 1 Statement -/
theorem problem1 (h : a_n 10 = 60) : a_n 10 = 60 := sorry

/-- Problem 2 Statement -/
theorem problem2 (n : ℕ) (hn : n ≠ 0) : a_n n = 0 ↔ (0 < n ∧ n < 6) := sorry

theorem problem2_gt (n : ℕ) : a_n n > 0 ↔ n ≥ 6 := sorry

theorem problem2_lt (n : ℕ) : a_n n < 0 ↔ (0 < n ∧ n < 6) := sorry

/-- Problem 3 Statement -/
theorem problem3 (S_n : ℕ → ℤ) : ¬(∃ n, S_n n = max (S_n 1) (S_n n)) ∧ ¬(∃ n, S_n n = min (S_n 1) (S_n n)) := sorry

end problem1_problem2_problem2_gt_problem2_lt_problem3_l818_818353


namespace estimate_proportion_l818_818078

theorem estimate_proportion (h1 : ∑ i in [3, 8, 9, 11, 10, 6], i = 47) (h2 : 3 ∈ [3, 8, 9, 11, 10, 6, 3]) (total : nat := 50) : 
  (47 / total.toFloat) * 100 = 94 := 
  by 
  sorry

end estimate_proportion_l818_818078


namespace same_side_interior_not_complementary_l818_818472

-- Defining the concept of same-side interior angles and complementary angles
def same_side_interior (α β : ℝ) : Prop := 
  α + β = 180 

def complementary (α β : ℝ) : Prop :=
  α + β = 90

-- To state the proposition that should be proven false
theorem same_side_interior_not_complementary (α β : ℝ) (h : same_side_interior α β) : ¬ complementary α β :=
by
  -- We state the observable contradiction here, and since the proof is not required we use sorry
  sorry

end same_side_interior_not_complementary_l818_818472


namespace sin_translation_l818_818919

theorem sin_translation :
  ∀ x : ℝ, y = sin (3x - π / 4) ↔ y = sin 3 (x - π / 12) :=
by
  sorry

end sin_translation_l818_818919


namespace angles_equal_l818_818422

open EuclideanGeometry

variables {A B C D P : Point}

theorem angles_equal (hABC: Parallelogram A B C D) (hAPDplusCPB: ∠ A P D + ∠ C P B = 180°) : 
  ∠ P B A = ∠ P D A :=
sorry

end angles_equal_l818_818422


namespace mean_and_median_change_l818_818511

theorem mean_and_median_change :
  let initial_data := [25, 18, 10, 27, 18]
  let corrected_data := [25, 18, 10, 15, 18]
  let original_total := initial_data.sum
  let original_mean := original_total / initial_data.length
  let original_median := (initial_data.sorted.get! (initial_data.length / 2))
  let new_total := corrected_data.sum
  let new_mean := new_total / corrected_data.length
  let new_median := (corrected_data.sorted.get! (corrected_data.length / 2))
  (new_mean = original_mean - 2.4) ∧ (new_median = original_median) :=
by
  sorry

end mean_and_median_change_l818_818511


namespace johns_shirt_percentage_increase_l818_818818

variable (P S : ℕ)

theorem johns_shirt_percentage_increase :
  P = 50 →
  S + P = 130 →
  ((S - P) * 100 / P) = 60 := by
  sorry

end johns_shirt_percentage_increase_l818_818818


namespace function_symmetry_l818_818346

noncomputable def function_min_value (a b : ℝ) (h : a ≠ 0) : ℝ → ℝ :=
  λ x : ℝ, a * Real.sin x - b * Real.cos x

theorem function_symmetry (a b : ℝ) (h₁ : a ≠ 0) (h₂ : function_min_value a b h₁ (3 * Real.pi / 4) = -Real.sqrt (a^2 + b^2)) :
  let f := function_min_value a b h₁ in
  f (π / 4 - x) = -f x ∧ ∀ x : ℝ, (π - x, 0) ∈ function_fix f (π, 0) :=
begin
  sorry
end

end function_symmetry_l818_818346


namespace paint_weight_correct_l818_818058

def weight_of_paint (total_weight : ℕ) (half_paint_weight : ℕ) : ℕ :=
  2 * (total_weight - half_paint_weight)

theorem paint_weight_correct :
  weight_of_paint 24 14 = 20 := by 
  sorry

end paint_weight_correct_l818_818058


namespace flea_survives_indefinitely_l818_818399

noncomputable def jump_and_poison (n : ℕ) : Prop :=
  ∀ (flea_path : ℕ → (ℕ × ℕ)),
    flea_path 0 = (0, 0) → 
    ∀ (poisoned_points : fin n.succ → (ℕ × ℕ)),
    (∀ i, (poisoned_points i ≠ flea_path i)) → 
    (∃ k < n.succ, flea_path k = poisoned_points k)

theorem flea_survives_indefinitely :
  ¬ ∃ n : ℕ, jump_and_poison n := sorry

end flea_survives_indefinitely_l818_818399


namespace sum_of_inscribed_angles_l818_818574

theorem sum_of_inscribed_angles (ABCDE: Type)
  (circum_circle : ∃ (c : Mathlib.Circle), circumscribes c ABCDE)
  (angle_AOE : ∠AOE = 50)
  (angle_COE : ∠COE = 70) :
  ∠CAE + ∠ECD = 60 :=
by sorry

end sum_of_inscribed_angles_l818_818574


namespace nancy_coffee_spending_l818_818852

theorem nancy_coffee_spending :
  let daily_cost := 3.00 + 2.50
  let total_days := 20
  let total_cost := total_days * daily_cost
  total_cost = 110.00 := by
    let daily_cost := 3.00 + 2.50
    let total_days := 20
    let total_cost := total_days * daily_cost
    have h1 : daily_cost = 5.50 := by norm_num
    have h2 : total_cost = 20 * 5.50 := by rw [total_cost, total_days, h1]
    have h3 : total_cost = 110.00 := by norm_num
    exact h3

end nancy_coffee_spending_l818_818852


namespace probability_mod_6_is_1_l818_818197

noncomputable def probability_condition (N : ℕ) : ℕ :=
  if N % 6 = 1 ∨ N % 6 = 3 ∨ N % 6 = 5 then 1 else 0

theorem probability_mod_6_is_1 :
  let count_favorable := Finset.card (Finset.filter (λ N, probability_condition N = 1) (Finset.range 2000))
  let total := 2000
  (count_favorable : ℚ) / total = 1 / 2 :=
by
  sorry

end probability_mod_6_is_1_l818_818197


namespace length_of_BC_l818_818352

theorem length_of_BC (AD DB AC : ℝ) 
  (hAD : AD = 20) (hDB : DB = 9) (hAC : AC = 15)
  (hABD : ∠A D B = 90) (hABC : ∠A C B = 90) (hDAC_ext : D ∈ line.extend AC) :
  (BC : ℝ) = Real.sqrt 706 :=
by
  -- Define points A, B, C, and D
  let A : ℝ × ℝ := (0, 0)
  let D : ℝ × ℝ := (AD, 0)
  let B : ℝ × ℝ := (AD + DB * cos(π / 2), DB * sin(π / 2))
  let C : ℝ × ℝ := (0, AC)

  -- Given lengths
  have hAD_length : dist A D = AD := by
    sorry -- provided as hAD
  have hDB_length : dist D B = DB := by
    sorry -- provided as hDB
  have hAC_length : dist A C = AC := by
    sorry -- provided as hAC

  -- Right angles imply Pythagorean theorem applies
  have ABD_right : angle A D B = π / 2 := by
    sorry -- provided as hABD
  have ABC_right : angle A C B = π / 2 := by
    sorry -- provided as hABC

  -- need to show BC is of the form sqrt(706)
  sorry

end length_of_BC_l818_818352


namespace range_of_fx1_l818_818750

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*x + 1 + a * Real.log x

theorem range_of_fx1 (a x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) (h3 : f x1 a = 0) (h4 : f x2 a = 0) :
    f x1 a > (1 - 2 * Real.log 2) / 4 :=
sorry

end range_of_fx1_l818_818750


namespace eliza_iron_total_l818_818649

-- Definition of the problem conditions in Lean
def blouse_time := 15 -- time to iron a blouse in minutes
def dress_time := 20 -- time to iron a dress in minutes
def blouse_hours := 2 -- hours spent ironing blouses
def dress_hours := 3 -- hours spent ironing dresses

-- Definition to convert hours to minutes
def hours_to_minutes (hours: Int) : Int :=
  hours * 60

-- Definition of the total number of pieces of clothes ironed by Eliza
def total_pieces_iron (blouse_time dress_time blouse_hours dress_hours: Int) : Int :=
  let blouses := hours_to_minutes(blouse_hours) / blouse_time
  let dresses := hours_to_minutes(dress_hours) / dress_time
  blouses + dresses

-- The proof statement
theorem eliza_iron_total : total_pieces_iron blouse_time dress_time blouse_hours dress_hours = 17 :=
by 
  -- To be filled in with the actual proof
  sorry

end eliza_iron_total_l818_818649


namespace sin_seven_pi_div_six_l818_818665

theorem sin_seven_pi_div_six : Real.sin (7 * Real.pi / 6) = -1 / 2 := 
  sorry

end sin_seven_pi_div_six_l818_818665


namespace trajectory_condition_line_passes_through_fixed_point_l818_818759

namespace TrajectoryFixedPoint

def point (α : Type) := prod α α
def A : point ℝ := (-real.sqrt 2, 0)
def B : point ℝ := (real.sqrt 2, 0)

def satisfies_condition (P Q : point ℝ) : Prop :=
  let PA := (fst A - fst P, snd A - snd P)
  let PB := (fst B - fst P, snd B - snd P)
  let PQ := (fst Q - fst P, snd Q - snd P)
  2 * (PA.1 * PB.1 + PA.2 * PB.2) = PQ.1^2 + PQ.2^2

def trajectory_eq (P : point ℝ) : Prop :=
  (fst P)^2 / 4 + (snd P)^2 / 2 = 1

theorem trajectory_condition (P Q : point ℝ) (h : satisfies_condition P Q) :
  trajectory_eq P :=
sorry

def intersects_trajectory (k : ℝ) : point ℝ :=
let x := (2 * k^2) / (2 * k^2 + 1)
let y := -k / (2 * k^2 + 1)
(x, y)

def E1 (k : ℝ) : point ℝ :=
let (x1, y1) := intersects_trajectory k
(x1 / 2, y1 / 2)

def E2 (k : ℝ) : point ℝ :=
let (x2, y2) := intersects_trajectory (1 / k)
(x2 / 2, y2 / 2)

def fixed_point : point ℝ := (2 / 3, 0)

theorem line_passes_through_fixed_point (k : ℝ) :
  (fst (E1 k) - fst (E2 k)) / (snd (E1 k) - snd (E2 k)) =
  (2 / 3, 0) :=
sorry

end TrajectoryFixedPoint

end trajectory_condition_line_passes_through_fixed_point_l818_818759


namespace sum_proper_divisors_512_l818_818957

theorem sum_proper_divisors_512 : ∑ i in {1, 2, 4, 8, 16, 32, 64, 128, 256}, i = 511 :=
by
  have h : {1, 2, 4, 8, 16, 32, 64, 128, 256} = finset.range 9.image (λ n, 2^n) := sorry
  rw [h]
  sorry

end sum_proper_divisors_512_l818_818957


namespace scalene_triangle_not_unique_by_two_non_opposite_angles_l818_818972

theorem scalene_triangle_not_unique_by_two_non_opposite_angles
  (α β : ℝ) (h1 : α > 0) (h2 : β > 0) (h3 : α + β < π) :
  ∃ (γ δ : ℝ), γ ≠ δ ∧ γ + α + β = δ + α + β :=
sorry

end scalene_triangle_not_unique_by_two_non_opposite_angles_l818_818972


namespace probability_shared_course_l818_818530

-- Definition of conditions: four courses and the way students select courses
inductive Course
| a | b | c | d

def courses : Finset Course := {Course.a, Course.b, Course.c, Course.d}

-- Utility to count combinations of selecting 2 courses out of 4
def combinations_count : ℕ := Nat.choose 4 2

-- Total number of ways both students choose their courses
def total_combinations : ℕ := combinations_count * combinations_count

-- Number of favorable cases where students share exactly one course in common
def favorable_cases : ℕ := 4 * 3 * 2 -- C(4, 1) * C(3, 1) * C(2, 1)

-- Calculate the probability
def probability : ℚ := favorable_cases / total_combinations

-- Define the statement to prove
theorem probability_shared_course :
  probability = 2 / 3 :=
by
  -- Proof goes here (for this exercise, we use sorry)
  sorry

end probability_shared_course_l818_818530


namespace average_of_all_11_numbers_l818_818485

-- Define the 11 numbers as a list
variables {numbers : List ℝ}
-- Define the conditions
def average_first_six (numbers : List ℝ) : Prop :=
  numbers.take 6 = [n1, n2, n3, n4, n5, n6] ∧
  (n1 + n2 + n3 + n4 + n5 + n6) / 6 = 88

def average_last_six (numbers : List ℝ) : Prop :=
  numbers.drop 5 = [n6, n7, n8, n9, n10, n11] ∧
  (n6 + n7 + n8 + n9 + n10 + n11) / 6 = 65

def sixth_number (numbers : List ℝ) : Prop :=
  numbers.nth 5 = some 258

-- The main theorem to prove
theorem average_of_all_11_numbers
  (h1 : ∀ numbers, average_first_six numbers)
  (h2 : ∀ numbers, average_last_six numbers)
  (h3 : ∀ numbers, sixth_number numbers) :
  (numbers.sum / 11 = 60) :=
begin
  sorry
end

end average_of_all_11_numbers_l818_818485


namespace area_ratio_S_T_l818_818828

open Set

def T : Set (ℝ × ℝ × ℝ) := {p | let (x, y, z) := p; x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 1}

def supports (p q : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := p
  let (a, b, c) := q
  (x ≥ a ∧ y ≥ b) ∨ (x ≥ a ∧ z ≥ c) ∨ (y ≥ b ∧ z ≥ c)

def S : Set (ℝ × ℝ × ℝ) := {p ∈ T | supports p (1/4, 1/4, 1/2)}

theorem area_ratio_S_T : ∃ k : ℝ, k = 3 / 4 ∧
  ∃ (area_T area_S : ℝ), area_T ≠ 0 ∧ (area_S / area_T = k) := sorry

end area_ratio_S_T_l818_818828


namespace scoring_situations_count_l818_818786

-- Define types for representing the conditions: Problem A and Problem B results
inductive Problem
| A_correct 
| A_incorrect 
| B_correct 
| B_incorrect

-- Define a function to calculate the score of a problem outcome
def problem_score : Problem → ℤ
| Problem.A_correct    := 30
| Problem.A_incorrect  := -30
| Problem.B_correct    := 10
| Problem.B_incorrect  := -10

-- Define a function to calculate the total score of a list of problems
def total_score (problems: List Problem) : ℤ :=
  problems.foldr (λ p acc => problem_score p + acc) 0

-- Prove that the number of different scoring situations that result in a total score of zero is 44
theorem scoring_situations_count : 
  (List.filter (λ ps => total_score ps = 0) (List.replicateM 4 [Problem.A_correct, Problem.A_incorrect, Problem.B_correct, Problem.B_incorrect])).length = 44 :=
  sorry

end scoring_situations_count_l818_818786


namespace number_of_fir_trees_l818_818288

def anya_statement (N : ℕ) : Prop := N = 15
def borya_statement (N : ℕ) : Prop := 11 ∣ N
def vera_statement (N : ℕ) : Prop := N < 25
def gena_statement (N : ℕ) : Prop := 22 ∣ N

def one_boy_one_girl_truth (A B G V : Prop) : Prop :=
  (A ∨ V) ∧ ¬(A ∧ V) ∧ (B ∨ G) ∧ ¬(B ∧ G)

theorem number_of_fir_trees (N : ℕ) :
  anya_statement N ∨ borya_statement N ∨ vera_statement N ∨ gena_statement N ∧
  one_boy_one_girl_truth (anya_statement N) (borya_statement N) (gena_statement N) (vera_statement N) :=
  N = 11 :=
sorry

end number_of_fir_trees_l818_818288


namespace num_correct_statements_l818_818077

theorem num_correct_statements : 
  (∃ (x : ℝ), x^2 - x > 0) ↔ (¬∀ (x : ℝ), x^2 - x ≤ 0) ∧ 
  (∀ (p q : Prop), (p ∨ q) → (¬¬p) → true) ∧ 
  (real.uniform_real [0, π]).prob ((λ x, sin x ≥ 1 / 2) event) = 2 / 3 :=
  sorry

end num_correct_statements_l818_818077


namespace sufficient_but_not_necessary_condition_l818_818706

theorem sufficient_but_not_necessary_condition 
(a b : ℝ) : (b ≥ 0) → ((a + 1)^2 + b ≥ 0) ∧ (¬ (∀ a b, ((a + 1)^2 + b ≥ 0) → b ≥ 0)) :=
by sorry

end sufficient_but_not_necessary_condition_l818_818706


namespace trigonometric_inequality_l818_818022

theorem trigonometric_inequality (x : ℝ) (n m : ℕ) 
  (hx : 0 < x ∧ x < (Real.pi / 2))
  (hnm : n > m) : 
  2 * |Real.sin x ^ n - Real.cos x ^ n| ≤
  3 * |Real.sin x ^ m - Real.cos x ^ m| := 
by 
  sorry

end trigonometric_inequality_l818_818022


namespace number_of_fir_trees_is_11_l818_818274

theorem number_of_fir_trees_is_11 
  (N : ℕ)
  (Anya : N = 15)
  (Borya : N % 11 = 0)
  (Vera : N < 25)
  (Gena : N % 22 = 0)
  (OneBoyOneGirlTrue : (Anya ∨ Borya) ∧ (Vera ∨ Gena) ∧ (¬Anya ∨ ¬Borya) ∧ (¬Vera ∨ ¬Gena)) :
  N = 11 := 
sorry

end number_of_fir_trees_is_11_l818_818274


namespace lcm_of_36_and_105_l818_818676

theorem lcm_of_36_and_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end lcm_of_36_and_105_l818_818676


namespace matthews_annual_income_l818_818381

noncomputable def annual_income (q : ℝ) (I : ℝ) (T : ℝ) : Prop :=
  T = 0.01 * q * 50000 + 0.01 * (q + 3) * (I - 50000) ∧
  T = 0.01 * (q + 0.5) * I → I = 60000

-- Statement of the math proof
theorem matthews_annual_income (q : ℝ) (T : ℝ) :
  ∃ I : ℝ, I = 60000 ∧ annual_income q I T :=
sorry

end matthews_annual_income_l818_818381


namespace total_area_covered_is_34_l818_818702

theorem total_area_covered_is_34
  (side_lengths : List ℕ)
  (areas_decreased : List ℕ)
  (h1 : side_lengths = [2, 3, 4, 5])
  (h2 : areas_decreased = [2, 3, 4, 5]) :
  let areas := side_lengths.map (λ s, s * s),
      total_area_without_overlap := areas.sum,
      total_overlap := 20
  in total_area_without_overlap - total_overlap = 34 := by
  sorry

end total_area_covered_is_34_l818_818702


namespace square_implies_increasing_l818_818515

def seq (a : ℕ → ℤ) :=
  a 1 = 1 ∧ ∀ n > 1, 
    ((a n - 2 > 0 ∧ ¬(∃ m < n, a m = a n - 2)) → a (n + 1) = a n - 2) ∧
    ((a n - 2 ≤ 0 ∨ ∃ m < n, a m = a n - 2) → a (n + 1) = a n + 3)

theorem square_implies_increasing (a : ℕ → ℤ) (n : ℕ) (h_seq : seq a) 
  (h_square : ∃ k, a n = k^2) (h_n_pos : n > 1) : 
  a n > a (n - 1) :=
sorry

end square_implies_increasing_l818_818515


namespace smallest_six_factors_l818_818122

theorem smallest_six_factors (n : ℕ) (h : (n = 2 * 3^2)) : n = 18 :=
by {
    sorry -- proof goes here
}

end smallest_six_factors_l818_818122


namespace light_path_length_l818_818421

-- Definition for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Given definitions as conditions
variable (A B C D E F G H P : Point3D)
variable (AB_CD EF_GH : List Point3D)
hypothesis cube : AB_CD = [A, B, C, D] ∧ EF_GH = [E, F, G, H]
hypothesis edge_length : (distance A B) = 10
hypothesis P_conditions : distance P (Point3D.mk 0 0 10) = 3 ∧ distance P (Point3D.mk 10 0 10) = 4

-- Define the distance function for verifications
noncomputable def distance (a b : Point3D) : ℝ :=
  real.sqrt ((a.x - b.x) ^ 2 + (a.y - b.y) ^ 2 + (a.z - b.z) ^ 2)

-- Proof goal stating the length of the light path
theorem light_path_length :
  let light_path_length := distance A P + distance P (Point3D.mk 10 0 0)
  light_path_length = 50 * real.sqrt 5 := sorry

end light_path_length_l818_818421


namespace least_positive_integer_with_six_factors_l818_818111

theorem least_positive_integer_with_six_factors :
  ∃ n : ℕ, (∀ m : ℕ, m > 0 → m < n → (count_factors m ≠ 6)) ∧ count_factors n = 6 ∧ n = 18 :=
sorry

noncomputable def count_factors (n : ℕ) : ℕ :=
sorry

end least_positive_integer_with_six_factors_l818_818111


namespace rem_frac_l818_818652

def rem (x y : ℚ) : ℚ := x - y * (⌊x / y⌋ : ℤ)

theorem rem_frac : rem (7 / 12) (-3 / 4) = -1 / 6 :=
by
  sorry

end rem_frac_l818_818652


namespace fir_trees_count_l818_818294

theorem fir_trees_count (N : ℕ) :
  (N = 15 ∨ (N < 25 ∧ 11 ∣ N) ∨ 22 ∣ N) ∧ 
  (1 ≤ (if N = 15 then 1 else 0) + (if 11 ∣ N then 1 else 0) + (if N < 25 then 1 else 0) + (if 22 ∣ N then 1 else 0) ∧
   2 ≠ (if N = 15 then 1 else 0) + (if 11 ∣ N then 1 else 0) + (if N < 25 then 1 else 0) + (if 22 ∣ N then 1 else 0)) → N = 11 :=
begin
  sorry
end

end fir_trees_count_l818_818294


namespace lucas_fraction_to_emma_l818_818009

variable (n : ℕ)

-- Define initial stickers
def noah_stickers := n
def emma_stickers := 3 * n
def lucas_stickers := 12 * n

-- Define the final state where each has the same number of stickers
def final_stickers_per_person := (16 * n) / 3

-- Lucas gives some stickers to Emma. Calculate the fraction of Lucas's stickers given to Emma
theorem lucas_fraction_to_emma :
  (7 * n / 3) / (12 * n) = 7 / 36 := by
  sorry

end lucas_fraction_to_emma_l818_818009


namespace sum_proper_divisors_512_l818_818959

theorem sum_proper_divisors_512 : ∑ i in Finset.range 9, 2^i = 511 := by
  -- Proof would be provided here
  sorry

end sum_proper_divisors_512_l818_818959


namespace tan_half_angle_l818_818329

theorem tan_half_angle (α : ℝ) (h1 : π / 2 < α ∧ α < π) (h2 : sin (α + π / 4) = sqrt 2 / 10) :
  tan (α / 2) = 2 := 
by 
  sorry

end tan_half_angle_l818_818329


namespace lesson_arrangements_l818_818575

theorem lesson_arrangements : 
  ∃ (arrangements : ℕ), arrangements = 12 ∧
    ∀ (lessons : list nat),
      length lessons = 5 →
      (∀ l ∈ lessons, l = 1 ∨ l = 2) →
      (lessons.count 2 = 2) →
      (lessons.count 1 = 3) →
      (¬ adjacent lessons 3) →
      (adjacent lessons 2) →
        arrangements = 12 :=
by
  sorry

def adjacent (lst : list nat) (x : nat) : Prop :=
  ∃ i, i + 1 < lst.length ∧ list.nth lst i = some x ∧ list.nth lst (i + 1) = some x

end lesson_arrangements_l818_818575


namespace smallest_range_mean_2017_l818_818701

theorem smallest_range_mean_2017 :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (a + b + c + d) / 4 = 2017 ∧ (max (max a b) (max c d) - min (min a b) (min c d)) = 4 := 
sorry

end smallest_range_mean_2017_l818_818701


namespace average_remaining_two_numbers_l818_818555

theorem average_remaining_two_numbers 
  (a1 a2 a3 a4 a5 a6 : ℝ)
  (h_avg_6 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 2.80)
  (h_avg_2_1 : (a1 + a2) / 2 = 2.4)
  (h_avg_2_2 : (a3 + a4) / 2 = 2.3) :
  (a5 + a6) / 2 = 3.7 :=
by
  sorry

end average_remaining_two_numbers_l818_818555


namespace conjugate_root_of_quadratic_l818_818868

theorem conjugate_root_of_quadratic (a b c: ℝ) (hr: ∃ x: ℂ, x = a + b * complex.I ∧ (a:ℂ) * x^2 + (b:ℂ) * x + (c:ℂ) = 0) : 
  ∃ y: ℂ, y = a - b * complex.I ∧ (a:ℂ) * y^2 + (b:ℂ) * y + (c:ℂ) = 0 :=
sorry

end conjugate_root_of_quadratic_l818_818868


namespace distance_between_4th_and_19th_red_l818_818034

/-- Define the pattern and position of lights on the string. -/
def lights (n : ℕ) : String :=
  if n % 7 < 3 then "red" else "green"

def position_of_red n : ℕ :=
  if n < 3 then n + 1 else 7 * (n / 3) + (1 + n % 3)

def distance_in_feet : ℕ → ℕ → ℕ
| pos1, pos2 => ((pos2 - pos1) * 8) / 12

theorem distance_between_4th_and_19th_red :
  distance_in_feet (position_of_red 3) (position_of_red 18) = 22.67 :=
sorry

end distance_between_4th_and_19th_red_l818_818034


namespace directrix_of_parabola_l818_818493

-- Define the given condition
def parabola_eq (x y : ℝ) : Prop := y = -4 * x^2

-- The problem we need to prove
theorem directrix_of_parabola :
  ∃ y : ℝ, (∀ x : ℝ, parabola_eq x y) ↔ y = 1 / 16 :=
by
  sorry

end directrix_of_parabola_l818_818493


namespace one_eighth_of_N_l818_818166

theorem one_eighth_of_N
  (N : ℝ)
  (h : (6 / 11) * N = 48) : (1 / 8) * N = 11 :=
sorry

end one_eighth_of_N_l818_818166


namespace find_original_number_l818_818819

def sequence_operations (n : ℤ) : ℤ :=
  (2 * (n + 3) - 2) / 2

theorem find_original_number (n : ℤ) (h : sequence_operations n = 9) : n = 7 := by
  unfold sequence_operations at h
  linarith

end find_original_number_l818_818819


namespace dot_product_eq_one_l818_818302

open Real

noncomputable def a : ℝ × ℝ := (2 * sin (35 * (π / 180)), 2 * cos (35 * (π / 180)))
noncomputable def b : ℝ × ℝ := (cos (5 * (π / 180)), -sin (5 * (π / 180)))

theorem dot_product_eq_one : (a.1 * b.1 + a.2 * b.2) = 1 := by
  let θ := 35 * (π / 180)
  let φ := 5 * (π / 180)
  have H : a = (2 * sin θ, 2 * cos θ) :=
    rfl
  have H' : b = (cos φ, -sin φ) :=
    rfl
  sorry

end dot_product_eq_one_l818_818302


namespace grouping_tourists_l818_818526

-- Define the number of tourists and guides
def num_tourists := 8
def num_guides := 3

-- Define the restricted group sizes for the third guide
def valid_group_sizes_for_third_guide := {1, 3, 4}

-- Define the formula to count the total arrangements without restrictions
def total_arrangements : ℕ := (num_guides : ℕ) ^ num_tourists

-- Define the excluded group sizes and compute the number of arrangements for each case
def excluded_cases : ℕ := (binom 8 0) * 2^8 + 
                          (binom 8 2) * 2^6 + 
                          (binom 8 5) * 2^3 + 
                          (binom 8 6) * 2^2 + 
                          (binom 8 7) * 2^1 + 
                          (binom 8 8) * 2^0

-- Calculate the number of valid arrangements
def valid_arrangements : ℕ := total_arrangements - excluded_cases

-- The statement of the proof problem in Lean
theorem grouping_tourists : valid_arrangements = 4608 := 
by 
  -- Proof omitted here
  sorry

end grouping_tourists_l818_818526


namespace least_pos_int_with_six_factors_l818_818101

theorem least_pos_int_with_six_factors :
  ∃ n : ℕ, (∀ m : ℕ, (number_of_factors m = 6 → m ≥ n)) ∧ n = 12 := 
sorry

end least_pos_int_with_six_factors_l818_818101


namespace length_of_EF_l818_818797

theorem length_of_EF (
  AB BC : ℝ 
  (hAB : AB = 9)
  (hBC : BC = 12)
  {D E F : Type} [metric_space D] [metrics.is_metric_space E] [metric_space F]
  (hDE_DF : dist DE DF = dist DE DF) -- DE = DF
  (area_triangle_DEF : ∀ D E F : Type, metrics.dist_full DEF = (1 / 3) * (AB * BC))  
):
  ∃ EF : ℝ, EF = 12 :=
by
  sorry

end length_of_EF_l818_818797


namespace part_i_part_ii_l818_818742

-- Part I: Prove that the solution of the inequality f(x - 1) + f(x + 3) ≥ 6 is x ∈ (-∞, -3] ∪ [3, ∞)
def f (x : ℝ) : ℝ := abs (x - 1)

theorem part_i (x : ℝ) :
  f(x - 1) + f(x + 3) ≥ 6 → x ∈ {x : ℝ | x ≤ -3} ∪ {x : ℝ | x ≥ 3} :=
by
  sorry

-- Part II: Prove that if |a| < 1, |b| < 1, and a ≠ 0, then f(ab) > |a|f(b/a)
theorem part_ii (a b : ℝ) (h1 : abs a < 1) (h2 : abs b < 1) (h3 : a ≠ 0) :
  f (a * b) > abs a * f (b / a) :=
by
  sorry

end part_i_part_ii_l818_818742


namespace trapezoid_max_area_l818_818823

open EuclideanGeometry Set Real

-- Geometry problem statement

theorem trapezoid_max_area
  (A B C D P Q X : Point)
  (h_trapezoid : is_isosceles_trapezoid A B C D)
  (h_parallel : parallel AD BC)
  (h_P_on_CD : lies_on P CD)
  (h_Q_on_DA : lies_on Q DA)
  (h_AP_perpendicular_CD : perpendicular (line_through A P) CD)
  (h_BQ_perpendicular_DA : perpendicular (line_through B Q) DA)
  (h_intersection_X : intersection (line_through A P) (line_through B Q) = X)
  (h_BX_eq_3 : dist B X = 3)
  (h_XQ_eq_1 : dist X Q = 1) :
  area_trapezoid A B C D = 32 := sorry

end trapezoid_max_area_l818_818823


namespace longest_line_segment_l818_818205

-- The problem's definition
def tromino (UnitSquare : Type) (LShape : Type) : Prop :=
  ∃ (u1 u2 u3 : UnitSquare), -- 3 unit squares
    ... -- conditions defining the shape and positioning to form an L-shaped figure

-- The theorem to prove
theorem longest_line_segment (UnitSquare : Type) (LShape : Type) (t : tromino UnitSquare LShape) :
  ∃ (A B : Point), is_perimeter_point t A ∧ is_perimeter_point t B ∧
    ∀ l, (line_segment_halves_area t l) → (line_segment_length A B l ≤ 2.5) :=
sorry

end longest_line_segment_l818_818205


namespace fir_trees_count_l818_818300

theorem fir_trees_count (N : ℕ) :
  (N = 15 ∨ (N < 25 ∧ 11 ∣ N) ∨ 22 ∣ N) ∧ 
  (1 ≤ (if N = 15 then 1 else 0) + (if 11 ∣ N then 1 else 0) + (if N < 25 then 1 else 0) + (if 22 ∣ N then 1 else 0) ∧
   2 ≠ (if N = 15 then 1 else 0) + (if 11 ∣ N then 1 else 0) + (if N < 25 then 1 else 0) + (if 22 ∣ N then 1 else 0)) → N = 11 :=
begin
  sorry
end

end fir_trees_count_l818_818300


namespace find_number_of_elements_l818_818043

theorem find_number_of_elements (n S : ℕ) (h1 : S + 26 = 19 * n) (h2 : S + 76 = 24 * n) : n = 10 := 
sorry

end find_number_of_elements_l818_818043


namespace sum_sequence_l818_818354

noncomputable def a : ℕ+ → ℝ
| 1 => 2
| (nat.succ n) => (nat.succ n) * a n / (n + 2 * a n)

theorem sum_sequence (n : ℕ+) :
  ∑ k in finset.range n, (k + 1) / (a (k + 1) : ℝ) = n^2 - (1/2 : ℝ) * n :=
sorry

end sum_sequence_l818_818354


namespace shark_sightings_l818_818630

theorem shark_sightings (x : ℕ) 
  (h1 : 26 = 5 + 3 * x) : x = 7 :=
by
  sorry

end shark_sightings_l818_818630


namespace cartesian_circle_eq_of_polar_range_sqrt3x_plus_y_l818_818219

section Elective_4_4

variables {t : ℝ}

-- Parametric form of the line l.
def line_param (t : ℝ) : ℝ × ℝ :=
  (-1 - (sqrt 3) / 2 * t, sqrt 3 + 1/2 * t)

-- Polar equation of the circle C.
def polar_eq (θ : ℝ) : ℝ :=
  4 * sin (θ - real.pi / 6)

-- Cartesian coordinate equation of the circle C.
def cartesian_eq (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*(sqrt 3)*y = 0

-- Define the Cartesian points as the solutions for given t.
def point_on_line (t : ℝ) : ℝ × ℝ :=
  let (x, y) := line_param t in (x, y)

-- Prove Cartesian equation of the circle given its polar equation.
theorem cartesian_circle_eq_of_polar :
  ∀ x y : ℝ, (∃ θ : ℝ, polar_eq θ = real.sqrt (x^2 + y^2) ∧ y = real.sqrt (x^2 + y^2) * sin θ ∧ x = real.sqrt (x^2 + y^2) * cos θ) →
    cartesian_eq x y :=
by
  sorry

-- Prove the range of (sqrt 3 * x + y) for point P(x, y) on given line and inside the circle.
theorem range_sqrt3x_plus_y :
  ∀ t : ℝ, -2 ≤ t ∧ t ≤ 2 → 
    let (x, y) := point_on_line t in
    -2 ≤ sqrt 3 * x + y ∧ sqrt 3 * x + y ≤ 2 :=
by
  sorry

end Elective_4_4

end cartesian_circle_eq_of_polar_range_sqrt3x_plus_y_l818_818219


namespace sum_of_squares_lt_l818_818065

theorem sum_of_squares_lt (n : ℕ) (x : fin n → ℝ) (h_sum : (∑ i, x i) = 2) : (∑ i, (x i)^2) < 0.1 :=
by {
  existsi λ i, 0.02,
  have h₁ : (∑ i, (0.02 : ℝ)) = 2, from sorry,
  have h₂ : (∑ i, (0.02 : ℝ)^2) < 0.1, from sorry,
  exact ⟨h₁, h₂⟩
}

end sum_of_squares_lt_l818_818065


namespace total_earnings_l818_818513

theorem total_earnings (daily_earnings : ℕ) (days : ℕ) (total : ℕ) :
  daily_earnings = 33 → days = 5 → total = 165 → daily_earnings * days = total :=
by
  intros h1 h2 h3
  rw [h1, h2]
  exact h3

end total_earnings_l818_818513


namespace base_subtraction_l818_818654

def base8_to_base10 (n : Nat) : Nat :=
  -- base 8 number 54321 (in decimal representation)
  5 * 4096 + 4 * 512 + 3 * 64 + 2 * 8 + 1

def base5_to_base10 (n : Nat) : Nat :=
  -- base 5 number 4321 (in decimal representation)
  4 * 125 + 3 * 25 + 2 * 5 + 1

theorem base_subtraction :
  base8_to_base10 54321 - base5_to_base10 4321 = 22151 := by
  sorry

end base_subtraction_l818_818654


namespace probability_both_red_is_5_over_26_l818_818550

-- Define the number of red, blue, and green balls
def red_balls := 6
def blue_balls := 5
def green_balls := 2

-- Calculate the total number of balls
def total_balls := red_balls + blue_balls + green_balls

-- Number of ways to choose 2 balls from total balls
def total_ways := Nat.choose total_balls 2

-- Number of ways to choose 2 red balls from red balls
def red_ways := Nat.choose red_balls 2

-- The probability that both balls picked are red
def probability_both_red : ℚ := red_ways / total_ways

theorem probability_both_red_is_5_over_26 :
  probability_both_red = 5 / 26 := by
  sorry

end probability_both_red_is_5_over_26_l818_818550


namespace side_length_of_square_l818_818907

theorem side_length_of_square (P : ℝ) (hP : P = 34.8) : 
  let S := P / 4 in S = 8.7 :=
by
  have S : ℝ := P / 4
  show S = 8.7
  sorry

end side_length_of_square_l818_818907


namespace determine_constants_l818_818642

theorem determine_constants :
  ∃ P Q R : ℚ, (∀ x : ℚ, x ≠ 1 → x ≠ 4 → x ≠ 6 → (x^2 - 4 * x + 8) / ((x - 1) * (x - 4) * (x - 6)) = P / (x - 1) + Q / (x - 4) + R / (x - 6)) ∧ 
  P = 1 / 3 ∧ Q = - 4 / 3 ∧ R = 2 :=
by
  -- Proof is left as a placeholder
  sorry

end determine_constants_l818_818642


namespace hyperbola_condition_l818_818698

-- Definitions and hypotheses
def is_hyperbola (m n : ℝ) (x y : ℝ) : Prop := m * x^2 - n * y^2 = 1

-- Statement of the problem
theorem hyperbola_condition (m n : ℝ) : (∃ x y : ℝ, is_hyperbola m n x y) ↔ m * n > 0 :=
by sorry

end hyperbola_condition_l818_818698


namespace min_xy_geometric_sequence_l818_818707

theorem min_xy_geometric_sequence :
  ∀ x y : ℝ, x > 1 → y > 1 → (∃ a, a = 1/2 ∧ ln x * ln y = a^2) → x * y ≥ Real.exp 1 :=
by {
  intros x y h_x_gt_1 h_y_gt_1 h_geom_seq,
  sorry
}

end min_xy_geometric_sequence_l818_818707


namespace smallest_base_conversion_l818_818568

theorem smallest_base_conversion :
  let n1 := 8 * 9 + 5 -- 85 in base 9
  let n2 := 2 * 6^2 + 1 * 6 -- 210 in base 6
  let n3 := 1 * 4^3 -- 1000 in base 4
  let n4 := 1 * 2^7 - 1 -- 1111111 in base 2
  n3 < n1 ∧ n3 < n2 ∧ n3 < n4 :=
by
  let n1 := 8 * 9 + 5
  let n2 := 2 * 6^2 + 1 * 6
  let n3 := 1 * 4^3
  let n4 := 1 * 2^7 - 1
  sorry

end smallest_base_conversion_l818_818568


namespace max_three_digit_numbers_divisible_by_4_in_sequence_l818_818581

theorem max_three_digit_numbers_divisible_by_4_in_sequence (n : ℕ) (a : ℕ → ℕ)
  (h_n : n ≥ 3)
  (h_increasing : ∀ i j, i < j → a i < a j)
  (h_recurrence : ∀ k, k ≤ n - 2 → a (k + 2) = 3 * a (k + 1) - 2 * a k - 2)
  (h_contains_2022 : ∃ k, a k = 2022) :
  ∀ k, a k = 2 * k → 
  (λ count_4 : ℕ, 
    (∀ m, 25 ≤ m ∧ m ≤ 249 → a (2 * m) = 4 * m) → 
    count_4 = 225) :=
begin
  sorry
end

end max_three_digit_numbers_divisible_by_4_in_sequence_l818_818581


namespace projection_F_T_on_OG_l818_818564

noncomputable def Circle := Type u

variables (O : Point) (circle : Circle)
variables {A B C F G T : Point}
variables (diameter : Line)
variables (AB AC BC : Line)

open EuclideanGeometry

-- Assumptions:
axiom chords_ABC : IsChord circle AB ∧ IsChord circle AC
axiom diam_perp_BC : Perpendicular diameter BC
axiom intersection_f : Intersects diameter AB F
axiom intersection_g : Intersects diameter AC G
axiom F_inside_circle : InsideCircle circle F
axiom tangent_GT : TangentToCircle circle G T

-- The theorem to prove
theorem projection_F_T_on_OG :
  ProjectionOnLine F OG T :=
sorry

end projection_F_T_on_OG_l818_818564


namespace composite_infinitely_many_l818_818025

theorem composite_infinitely_many (t : ℕ) (ht : t ≥ 2) :
  ∃ n : ℕ, n = 3 ^ (2 ^ t) - 2 ^ (2 ^ t) ∧ (3 ^ (n - 1) - 2 ^ (n - 1)) % n = 0 :=
by
  use 3 ^ (2 ^ t) - 2 ^ (2 ^ t)
  sorry 

end composite_infinitely_many_l818_818025


namespace smallest_positive_n_common_factor_l818_818535

open Int

theorem smallest_positive_n_common_factor :
  ∃ n : ℕ, n > 0 ∧ gcd (11 * n - 3) (8 * n + 4) > 1 ∧ n = 5 :=
by
  sorry

end smallest_positive_n_common_factor_l818_818535


namespace part1_part2_l818_818052

noncomputable def f : ℝ → ℝ :=
λ x, if (0 ≤ x ∧ x ≤ 1) then 3^x - 1 else
     if (x < 0 ∧ -x ≤ 1) then (1/3)^x - 1 else
     0 -- placeholder for f outside the interval as per problem

variables (x : ℝ)

theorem part1 : x ∈ Icc (-1:ℝ) 0 → f x = (1/3)^x - 1 :=
begin
  intros h,
  -- proof sketch needed
  sorry
end

theorem part2 : f (log (6) / log (1 / 3)) = 1 / 2 :=
begin
  -- proof sketch needed
  sorry
end

end part1_part2_l818_818052


namespace qiqi_average_annual_growth_rate_l818_818464

theorem qiqi_average_annual_growth_rate 
  (initial_weight final_weight : ℤ) (years : ℤ) 
  (h_initial : initial_weight = 40)
  (h_final : final_weight = 48.4) 
  (h_years : years = 2) : 
  (1 + 0.1) ^ years = final_weight / initial_weight :=
by
  rw [h_initial, h_final, h_years]
  norm_num
  sorry

end qiqi_average_annual_growth_rate_l818_818464


namespace harriet_current_age_l818_818389

-- Definitions from the conditions in a)
def mother_age : ℕ := 60
def peter_current_age : ℕ := mother_age / 2
def peter_age_in_four_years : ℕ := peter_current_age + 4
def harriet_age_in_four_years : ℕ := peter_age_in_four_years / 2

-- Proof statement
theorem harriet_current_age : harriet_age_in_four_years - 4 = 13 :=
by
  -- from the given conditions and the solution steps
  let h_current_age := harriet_age_in_four_years - 4
  have : h_current_age = (peter_age_in_four_years / 2) - 4 := by sorry
  have : peter_age_in_four_years = 34 := by sorry
  have : harriet_age_in_four_years = 17 := by sorry
  show 17 - 4 = 13 from sorry

end harriet_current_age_l818_818389


namespace sin_of_7pi_over_6_l818_818662

theorem sin_of_7pi_over_6 : Real.sin (7 * Real.pi / 6) = -1 / 2 :=
by
  sorry

end sin_of_7pi_over_6_l818_818662


namespace ratio_of_areas_eq_five_minus_three_sqrt_three_l818_818459

theorem ratio_of_areas_eq_five_minus_three_sqrt_three (ABC : Triangle) (A B C E : Point)
  (h1 : is_equilateral ABC)
  (h2 : E ∈ (segment A C))
  (h3 : ∠ EBC = 30) :
  ∃ (area_AEB : ℝ) (area_CEB : ℝ), 
  area_AEB / area_CEB = (5 - 3 * Real.sqrt 3) := 
sorry

end ratio_of_areas_eq_five_minus_three_sqrt_three_l818_818459


namespace filled_tank_by_11pm_l818_818408

/-- 
Given:
- The rain starts at 1 pm.
- The fish tank is 18 inches tall.
- The rainfall and evaporation rates are as follows:
  - 1st hour: 2 inches per hour rain, 0.2 inches per hour evaporation.
  - 2nd hour: 1 inch per hour rain, 0.2 inches per hour evaporation.
  - 3rd hour: 0.5 inches per hour rain, 0.2 inches per hour evaporation.
  - 4th hour: 1.5 inches per hour rain, 0.2 inches per hour evaporation.
  - 5th hour: 0.8 inches per hour rain, 0.2 inches per hour evaporation.
  - Afterward: 3 inches per hour rain, 0.1 inches per hour evaporation.

Prove that the tank will be filled with rainwater by 11 pm.
-/
theorem filled_tank_by_11pm :
  ∀ (start_time tall first_hour_rate second_hour_rate third_hour_rate fourth_hour_rate fifth_hour_rate subsequent_rate first_hour_evap second_hour_evap) 
  (hours : ℕ)
  (net_water : ℕ),
  start_time = 1 ∧
  tall = 18 ∧
  first_hour_rate = 2 ∧
  second_hour_rate = 1 ∧
  third_hour_rate = 0.5 ∧
  fourth_hour_rate = 1.5 ∧
  fifth_hour_rate = 0.8 ∧
  subsequent_rate = 3 ∧
  first_hour_evap = 0.2 ∧
  subsequent_hour_evap = 0.1 ∧
  net_water = first_hour_rate - first_hour_evap + second_hour_rate - first_hour_evap + third_hour_rate - first_hour_evap + fourth_hour_rate - first_hour_evap + fifth_hour_rate - first_hour_evap + 5 * (subsequent_rate - subsequent_hour_evap) 
  → net_water = tall 
  → tall / subsequent_rate = hours 
  → start_time + hours = 23
sorry

end filled_tank_by_11pm_l818_818408


namespace largest_apartment_number_l818_818012

theorem largest_apartment_number :
  ∃ apartment_number : ℕ, 
    digits_sum_equal (apartment_number = 9875) ∧ 
    digit_sum 8653421 = 29 ∧ 
    all_digits_distinct apartment_number ∧ 
    digit_sum apartment_number = 29 :=
sorry

def digits_sum_equal (a b : ℕ) : Prop := digit_sum a = digit_sum b

def digit_sum (n : ℕ) : ℕ :=
  (List.sum (int.digits 10 n).to_finset)

def all_digits_distinct (n : ℕ) : Prop :=
  (List.nodup (int.digits 10 n).to_finset.to_list).to_finset

end largest_apartment_number_l818_818012


namespace lcm_36_105_l818_818688

theorem lcm_36_105 : Int.lcm 36 105 = 1260 := by
  sorry

end lcm_36_105_l818_818688


namespace smallest_integer_with_six_distinct_factors_l818_818130

noncomputable def least_pos_integer_with_six_factors : ℕ :=
  12

theorem smallest_integer_with_six_distinct_factors 
  (n : ℕ)
  (p q : ℕ)
  (a b : ℕ)
  (hp : prime p)
  (hq : prime q)
  (h_diff : p ≠ q)
  (h_n : n = p ^ a * q ^ b)
  (h_factors : (a + 1) * (b + 1) = 6) :
  n = least_pos_integer_with_six_factors :=
by
  sorry

end smallest_integer_with_six_distinct_factors_l818_818130


namespace sin_sum_diff_l818_818731

variable (θ : ℝ)
axiom sin_theta : Real.sin θ = 1 / 3
axiom acute_θ : 0 < θ ∧ θ < π / 2

theorem sin_sum_diff (h₀ : Real.sin θ = 1 / 3) (h₁ : 0 < θ ∧ θ < π / 2) : 
    Real.sin (θ + π / 4) - Real.sin (θ - π / 4) = 4 / 3 := 
sorry

end sin_sum_diff_l818_818731


namespace average_of_arithmetic_sequence_l818_818533

theorem average_of_arithmetic_sequence :
  let seq : List ℤ := List.range' (-180) 6 21 in
  (seq.headD + seq.reverse.headD) / 2 = 0 :=
by
  sorry

end average_of_arithmetic_sequence_l818_818533


namespace least_positive_integer_with_six_factors_is_18_l818_818138

-- Define the least positive integer with exactly six distinct positive factors.
def least_positive_with_six_factors (n : ℕ) : Prop :=
  (∀ d : ℕ, d ∣ n → d > 0) ∧ (finset.card (finset.filter (λ d, d ∣ n) (finset.range (n + 1)))) = 6

-- Prove that the least positive integer with exactly six distinct positive factors is 18.
theorem least_positive_integer_with_six_factors_is_18 : (∃ n : ℕ, least_positive_with_six_factors n ∧ n = 18) :=
sorry


end least_positive_integer_with_six_factors_is_18_l818_818138


namespace vectors_coplanar_l818_818610

open Matrix

def a : Fin 3 → ℝ := ![7, 4, 6]
def b : Fin 3 → ℝ := ![2, 1, 1]
def c : Fin 3 → ℝ := ![19, 11, 17]

def mat : Matrix (Fin 3) (Fin 3) ℝ := λ i j, ![a, b, c] i j

theorem vectors_coplanar : det mat = 0 :=
by
  sorry

end vectors_coplanar_l818_818610


namespace total_recruits_211_l818_818512

theorem total_recruits_211 (P N D : ℕ) (total : ℕ) 
  (h1 : P = 50) 
  (h2 : N = 100) 
  (h3 : D = 170) 
  (h4 : ∃ (x y : ℕ), (x = 4 * y ∨ y = 4 * x) ∧ 
                      ((x, P) = (y, N) ∨ (x, N) = (y, D) ∨ (x, P) = (y, D))) :
  total = 211 :=
by
  sorry

end total_recruits_211_l818_818512


namespace lcm_36_105_l818_818681

noncomputable def factorize_36 : List (ℕ × ℕ) := [(2, 2), (3, 2)]
noncomputable def factorize_105 : List (ℕ × ℕ) := [(3, 1), (5, 1), (7, 1)]

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 :=
by
  have h_36 : 36 = 2^2 * 3^2 := by norm_num
  have h_105 : 105 = 3^1 * 5^1 * 7^1 := by norm_num
  sorry

end lcm_36_105_l818_818681


namespace sequence_properties_l818_818715

noncomputable def a_seq (n : ℕ) : ℕ := 2 * n - 1
noncomputable def S (n : ℕ) : ℕ := (n * (2 * n - 1)) / 2

theorem sequence_properties :
  (a_seq 1 = 1) ∧ (a_seq 2 = 3) ∧ (∀ n : ℕ, a_seq n ^ 2 = 4 * S n - 2 * a_seq n - 1) :=
by
  split
  { -- Prove a_seq 1 = 1
    sorry },
  { split
  { -- Prove a_seq 2 = 3
    sorry },
  { -- Prove the general term formula for all n
    intro n,
    sorry } }

end sequence_properties_l818_818715


namespace range_m_l818_818160

def set_A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def set_B (m : ℝ) : Set ℝ := {x | (m - 1) ≤ x ∧ x ≤ (m + 1)}

theorem range_m (m : ℝ) : (∀ x, x ∈ set_B m → x ∈ set_A) ↔ (-1 ≤ m ∧ m ≤ 4) :=
by
  sorry

end range_m_l818_818160


namespace average_weight_section_B_l818_818523

/-- Given the following conditions:
1. There are 60 students in section A, each with an average weight of 60 kg.
2. There are 70 students in section B.
3. The overall average weight of all students in both sections is 70.77 kg.
Prove that the average weight of section B is approximately 79.99 kg.
-/
theorem average_weight_section_B 
  (students_A : ℕ := 60) 
  (avg_weight_A : ℝ := 60) 
  (students_B : ℕ := 70) 
  (overall_avg_weight : ℝ := 70.77) :
  let total_weight_A := students_A * avg_weight_A,
      total_students := students_A + students_B,
      total_weight := total_students * overall_avg_weight,
      total_weight_B := total_weight - total_weight_A,
      avg_weight_B := total_weight_B / students_B 
  in 
  avg_weight_B ≈ 79.99 := 
by
  sorry

end average_weight_section_B_l818_818523


namespace min_OP_PF_squared_l818_818774

def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in (x^2 / 2 + y^2 = 1)

def OP_squared (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P in x^2 + y^2

def PF_squared (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P in (x + 1)^2 + y^2

theorem min_OP_PF_squared :
  ∃ P : ℝ × ℝ, is_on_ellipse P ∧ (∀ Q : ℝ × ℝ, is_on_ellipse Q → OP_squared Q + PF_squared Q ≥ 2) :=
by
  sorry

end min_OP_PF_squared_l818_818774


namespace inscribed_circle_center_orthocenter_angle_l818_818835

open EuclideanGeometry

theorem inscribed_circle_center_orthocenter_angle
  (A B C O H : Point)
  (h_incircle : is_incenter O A B C)
  (h_orthocenter : is_orthocenter H A B C) :
  angle O A B = angle H A C :=
by
  sorry

end inscribed_circle_center_orthocenter_angle_l818_818835


namespace smallest_six_factors_l818_818123

theorem smallest_six_factors (n : ℕ) (h : (n = 2 * 3^2)) : n = 18 :=
by {
    sorry -- proof goes here
}

end smallest_six_factors_l818_818123


namespace complex_z_in_first_quadrant_l818_818489

noncomputable def complex_z : ℂ := (i / (1 + i))

theorem complex_z_in_first_quadrant : 
  (Re complex_z > 0) ∧ (Im complex_z > 0) :=
by
  sorry

end complex_z_in_first_quadrant_l818_818489


namespace delaney_missed_bus_time_l818_818638

def busDepartureTime : Nat := 480 -- 8:00 a.m. = 8 * 60 minutes
def travelTime : Nat := 30 -- 30 minutes
def departureFromHomeTime : Nat := 470 -- 7:50 a.m. = 7 * 60 + 50 minutes

theorem delaney_missed_bus_time :
  (departureFromHomeTime + travelTime - busDepartureTime) = 20 :=
by
  -- proof would go here
  sorry

end delaney_missed_bus_time_l818_818638


namespace lcm_36_105_l818_818684

noncomputable def factorize_36 : List (ℕ × ℕ) := [(2, 2), (3, 2)]
noncomputable def factorize_105 : List (ℕ × ℕ) := [(3, 1), (5, 1), (7, 1)]

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 :=
by
  have h_36 : 36 = 2^2 * 3^2 := by norm_num
  have h_105 : 105 = 3^1 * 5^1 * 7^1 := by norm_num
  sorry

end lcm_36_105_l818_818684


namespace find_A_and_B_l818_818892

theorem find_A_and_B :
  ∃ A B : ℕ, 
  (A ≠ 0 ∧ B ≠ 0) ∧ (A ∈ {6, 7, 8, 9}) ∧ (B ∈ {7, 9}) ∧
  (A ≠ B) ∧ (A ≠ 1) ∧ (A ≠ 2) ∧ (A ≠ 3) ∧ (A ≠ 4) ∧ (A ≠ 5) ∧
  (A * 100000 + 12345) % B = 0 ∧ (123450 + A) % B = 0 ∧ (A, B) = (9, 7) :=
by { sorry }

end find_A_and_B_l818_818892


namespace lamp_probability_l818_818084

theorem lamp_probability (rope_length : ℝ) (pole_distance : ℝ) (h_pole_distance : pole_distance = 8) :
  let lamp_range := 2
  let favorable_segment_length := 4
  let total_rope_length := rope_length
  let probability := (favorable_segment_length / total_rope_length)
  rope_length = 8 → probability = 1 / 2 :=
by
  intros
  sorry

end lamp_probability_l818_818084


namespace find_base_a_l818_818502

theorem find_base_a (a : ℝ) (h1 : 2 ≤ x ∧ x ≤ 3) (h2 : log a 3 = 2 * log a 2) :
  a = 3/2 ∨ a = 2/3 :=
sorry

end find_base_a_l818_818502


namespace simplify_complex_fraction_l818_818030

theorem simplify_complex_fraction : 
  (7 + 18 * complex.I) / (3 - 4 * complex.I) = (-51 / 25 : ℂ) + (82 / 25) * complex.I :=
by {
  sorry
}

end simplify_complex_fraction_l818_818030


namespace least_positive_integer_with_six_factors_is_18_l818_818134

-- Define the least positive integer with exactly six distinct positive factors.
def least_positive_with_six_factors (n : ℕ) : Prop :=
  (∀ d : ℕ, d ∣ n → d > 0) ∧ (finset.card (finset.filter (λ d, d ∣ n) (finset.range (n + 1)))) = 6

-- Prove that the least positive integer with exactly six distinct positive factors is 18.
theorem least_positive_integer_with_six_factors_is_18 : (∃ n : ℕ, least_positive_with_six_factors n ∧ n = 18) :=
sorry


end least_positive_integer_with_six_factors_is_18_l818_818134


namespace sin_of_7pi_over_6_l818_818658

theorem sin_of_7pi_over_6 : Real.sin (7 * Real.pi / 6) = -1 / 2 :=
by
  -- Conditions from the statement in a)
  -- Given conditions: \(\sin (180^\circ + \theta) = -\sin \theta\)
  -- \(\sin 30^\circ = \frac{1}{2}\)
  sorry

end sin_of_7pi_over_6_l818_818658


namespace sqrt_a_is_arithmetic_sum_c_n_l818_818340

section
variables {ℕ : Type} [Preorder ℕ] [HasZero ℕ] [Add ℕ] [Mul ℕ] [OfNat ℕ 2]

-- Define the function f and its inverse
def f (x : ℝ) : ℝ := x - 4 * Real.sqrt x + 4
def f_inv (x : ℝ) : ℝ := (Real.sqrt x + 2) ^ 2

-- Define the sequence {a_n}
def a : ℕ → ℝ
| 1 => 1
| (n + 1) => f_inv (a n)

-- Define sequence {b_n} and the differences being a geometric series
def b : ℕ → ℝ
| 1 => 1
| n => (3 / 2) * (1 - 1 / (3 ^ n))

-- Prove that the sequence {sqrt a_n} is arithmetic
theorem sqrt_a_is_arithmetic {n : ℕ} (hn : 0 < n) : ∃ d, ∀ m, sqrt (a m.succ) - sqrt (a m) = d :=
sorry

-- Define the sequence {c_n}
def c : ℕ → ℝ := λ n, Real.sqrt (a n) * b n

-- Prove the sum S_n of the first n terms of sequence {c_n} is as stated
theorem sum_c_n {n : ℕ} (hn : 0 < n) : ∑ i in Finset.range n, c i.succ = (3 / 2) * (n ^ 2 - 1 + (n + 1) / (3 ^ n)) :=
sorry

end

end sqrt_a_is_arithmetic_sum_c_n_l818_818340


namespace more_eggs_than_marbles_l818_818471

theorem more_eggs_than_marbles (eggs marbles : ℕ) (h1 : eggs = 20) (h2 : marbles = 6) : eggs - marbles = 14 :=
by
  -- Given conditions
  have h_eggs : eggs = 20 := h1
  have h_marbles : marbles = 6 := h2
  -- Difference calculation
  rw [h_eggs, h_marbles]
  show 20 - 6 = 14
  sorry

end more_eggs_than_marbles_l818_818471


namespace number_of_trees_is_eleven_l818_818260

variables (N : ℕ)

-- Conditions
def Anya (N : ℕ) := N = 15
def Borya (N : ℕ) := 11 ∣ N
def Vera (N : ℕ) := N < 25
def Gena (N : ℕ) := 22 ∣ N

axiom OneBoyOneGirlTruth :
  (∃ (b : Prop) (g : Prop),
    (b ∨ ¬ b) ∧ (g ∨ ¬ g) ∧
    ((b = (Borya N ∨ Gena N)) ∧ (g = (Anya N ∨ Vera N)) ∧
     (b ↔ ¬g) ∧
     ((Anya N ∨ ¬Vera N) ∨ (¬Anya N ∨ Vera N)) ∧
     (Anya N = (N = 15)) ∧
     (Borya N = (11 ∣ N)) ∧
     (Vera N = (N < 25)) ∧
     (Gena N = (22 ∣ N))))

theorem number_of_trees_is_eleven: N = 11 :=
sorry

end number_of_trees_is_eleven_l818_818260


namespace cousin_cards_probability_l818_818813

variable {Isabella_cards : ℕ}
variable {Evan_cards : ℕ}
variable {total_cards : ℕ}

theorem cousin_cards_probability 
  (h1 : Isabella_cards = 8)
  (h2 : Evan_cards = 2)
  (h3 : total_cards = 10) :
  (8 / 10 * 2 / 9) + (2 / 10 * 8 / 9) = 16 / 45 :=
by
  sorry

end cousin_cards_probability_l818_818813


namespace gcd_888_1147_l818_818496

theorem gcd_888_1147 : Nat.gcd 888 1147 = 37 := by
  sorry

end gcd_888_1147_l818_818496


namespace part1_part2_part3_l818_818347

noncomputable def f (x : ℝ) : ℝ := 3 * x - Real.exp x + 1

theorem part1 :
  ∃ x0 > 0, f x0 = 0 :=
sorry

theorem part2 (x0 : ℝ) (h1 : f x0 = 0) :
  ∀ x, f x ≤ (3 - Real.exp x0) * (x - x0) :=
sorry

theorem part3 (m x1 x2 : ℝ) (h1 : m > 0) (h2 : x1 < x2) (h3 : f x1 = m) (h4 : f x2 = m):
  x2 - x1 < 2 - 3 * m / 4 :=
sorry

end part1_part2_part3_l818_818347


namespace average_distance_run_l818_818447

theorem average_distance_run :
  let mickey_lap := 250
  let johnny_lap := 300
  let alex_lap := 275
  let lea_lap := 280
  let johnny_times := 8
  let lea_times := 5
  let mickey_times := johnny_times / 2
  let alex_times := mickey_times + 1 + 2 * lea_times
  let total_distance := johnny_times * johnny_lap + mickey_times * mickey_lap + lea_times * lea_lap + alex_times * alex_lap
  let number_of_participants := 4
  let avg_distance := total_distance / number_of_participants
  avg_distance = 2231.25 := by
  sorry

end average_distance_run_l818_818447


namespace max_three_digit_divisible_by_4_sequence_l818_818583

theorem max_three_digit_divisible_by_4_sequence (a : ℕ → ℕ) (n : ℕ) (h1 : ∀ k ≤ n - 2, a (k + 2) = 3 * a (k + 1) - 2 * a k - 2)
(h2 : ∀ k1 k2, k1 < k2 → a k1 < a k2) (ha2022 : ∃ k, a k = 2022) (hn : n ≥ 3) :
  ∃ m : ℕ, ∀ k, 100 ≤ a k ∧ a k ≤ 999 → a k % 4 = 0 → m ≤ 225 := by
  sorry

end max_three_digit_divisible_by_4_sequence_l818_818583


namespace value_of_a_plus_c_l818_818187

-- Definitions for cone dimensions and sphere radius in Lean 4
def cone_base_radius : ℝ := 16
def cone_height : ℝ := 32
def sphere_radius (a c : ℝ) : ℝ := a * (Real.sqrt c - 1)

-- Primary goal: Prove the value of a + c == 13
theorem value_of_a_plus_c (a c : ℝ) (h1 : sphere_radius a c = 8 * (Real.sqrt 5 - 1)) : a + c = 13 :=
by
  -- Add necessary proof here
  sorry

end value_of_a_plus_c_l818_818187


namespace fir_trees_alley_l818_818283

-- Define the statements made by each child
def statementAnya (N : ℕ) : Prop := N = 15
def statementBorya (N : ℕ) : Prop := N % 11 = 0
def statementVera (N : ℕ) : Prop := N < 25
def statementGena (N : ℕ) : Prop := N % 22 = 0

-- Define the condition about the truth and lies
def oneBoyOneGirlTruth (anya_vera_truth: Prop) (borya_gena_truth: Prop) : Prop :=
  anya_vera_truth ∧ borya_gena_truth ∧
  ((statementAnya N ∧ statementVera N) ∨ (statementVera N ∧ statementBorya N)) ∧
  ¬( (statementAnya N ∧ statementGena N) ∨ (statementVera N ∧ statementGena N) ∨
     (statementAnya N ∧ statementBorya N) ∨ (statementBorya N ∧ statementGena N) )

-- Prove that the number of fir trees is 11
theorem fir_trees_alley: ∃ (N : ℕ), statementBorya N ∧ statementVera N ∧ ¬ statementAnya N ∧ ¬ statementGena N ∧ oneBoyOneGirlTruth (¬ statementAnya N ∧ statementVera N) (statementBorya N ∧ ¬ statementGena N) ∧ N = 11 :=
by
  sorry

end fir_trees_alley_l818_818283


namespace least_pos_int_with_six_factors_l818_818105

theorem least_pos_int_with_six_factors :
  ∃ n : ℕ, (∀ m : ℕ, (number_of_factors m = 6 → m ≥ n)) ∧ n = 12 := 
sorry

end least_pos_int_with_six_factors_l818_818105


namespace sin_seven_pi_div_six_l818_818664

theorem sin_seven_pi_div_six : Real.sin (7 * Real.pi / 6) = -1 / 2 := 
  sorry

end sin_seven_pi_div_six_l818_818664


namespace trapezoid_ratio_of_bases_l818_818455

theorem trapezoid_ratio_of_bases
  (ABCD : Type)
  [Trapezoid ABCD]
  (one_angle_is_60 : ∃ A B C D, angle A B D = 60)
  (is_circumscribable : circumscribable ABCD)
  (is_inscribable : inscribable ABCD) :
  ratio (ABCD.base1) (ABCD.base2) = 1 / 3 := 
sorry

end trapezoid_ratio_of_bases_l818_818455


namespace problem1_problem2_l818_818985

-- Problem (1)
theorem problem1 (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a^3 + b^3 >= a*b^2 + a^2*b := 
sorry

-- Problem (2)
theorem problem2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) : 
  (1 + y) / x < 2 ∨ (1 + x) / y < 2 := 
sorry

end problem1_problem2_l818_818985


namespace complex_number_identification_l818_818997

noncomputable def z : ℂ := 1 + complex.i

theorem complex_number_identification : z = 1 + complex.i :=
by
  sorry

end complex_number_identification_l818_818997


namespace min_participants_l818_818992

/-- 
  In a single round-robin Chinese chess competition where:
  - Each participant plays against every other participant exactly once
  - The winner receives 2 points, the loser gets 0 points, and in case of a draw, both players receive 1 point
  - The champion has more points than any other participant
  - The champion has won fewer matches than any other participant
  Prove that the minimum number of participants must be 6.
-/
theorem min_participants (n : ℕ) 
  (round_robin : ∀ (i j : ℕ), i ≠ j → ∃ match : {win : bool // win = tt ∨ win = ff},
  (if match.val = tt then 2 else (if match.val = ff then 0 else 1)) = if i > j then 2 else (if i < j then 0 else 1))
  (points_distributed : ∀ (i j : ℕ), (if i ≠ j then (if i > j then 2 else (if i < j then 0 else 1)) else 0))
  (champion_more_points : ∀ (champ : ℕ), (points_distributed champ ≤ ∀ (x : ℕ), points_distributed x))
  (champ_fewer_wins : ∀ (champ : ℕ), wins champ < ∀ (x : ℕ), wins x)
  : n = 6 := 
sorry

end min_participants_l818_818992


namespace no_nat_nums_satisfying_l818_818462

theorem no_nat_nums_satisfying (x y z k : ℕ) (hx : x < k) (hy : y < k) : x^k + y^k ≠ z^k :=
by
  sorry

end no_nat_nums_satisfying_l818_818462


namespace no_zeros_f_monotonic_intervals_f_minimum_y_l818_818747

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := Real.log x - m * x

theorem no_zeros_f (m : ℝ) :
  (∀ x > 0, f x m ≠ 0) ↔ m > (1 : ℝ) / Real.exp 1 := sorry

theorem monotonic_intervals_f (m : ℝ) :
  (∀ x > 0, x < 1 / m → f' x m > 0) ∧ (∀ x > 1 / m, f' x m < 0) ↔ m > 0 := sorry

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := 2 * f x m + x^2
noncomputable def h (x : ℝ) (c : ℝ) (b : ℝ) : ℝ := Real.log x - c * x^2 - b * x

theorem minimum_y (x1 x2 : ℝ) (h : ℝ → ℝ) (c : ℝ) (b : ℝ) :
  x1 < x2 →
  (b = (Real.log (x1 / x2) / (x1 - x2) - c * (x1 + x2))) →
  (x1 + x2)^2 = m^2 →
  m ≥ (3 * Real.sqrt 2) / 2 →
  ∃ t ∈ Ioo 0 (1 / 2), y = (x1 - x2) * h ((x1 + x2) / 2) ∧ y ≥ -2/3 + Real.log 2 := sorry

end no_zeros_f_monotonic_intervals_f_minimum_y_l818_818747


namespace perimeter_Polygon_ABCDE_l818_818804

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2).sqrt

/-- Prove that the perimeter of polygon ABCDE is 25 + sqrt 41 --/
theorem perimeter_Polygon_ABCDE : 
  let A := (0, 8) in 
  let B := (4, 8) in 
  let C := (4, 4) in
  let D := (0, 0) in 
  let E := (9, 0) in 
  distance A B + distance B C + distance C E + distance E D + distance D A = 25 + Real.sqrt 41 :=
by
  let A := (0, 8)
  let B := (4, 8)
  let C := (4, 4)
  let D := (0, 0)
  let E := (9, 0)

  have hAB : distance A B = 4 := by sorry
  have hBC : distance B C = 4 := by sorry
  have hCE : distance C E = Real.sqrt 41 := by sorry
  have hED : distance E D = 9 := by sorry
  have hDA : distance D A = 8 := by sorry

  calc
    distance A B + distance B C + distance C E + distance E D + distance D A
      = 4 + 4 + Real.sqrt 41 + 9 + 8 : by rw [hAB, hBC, hCE, hED, hDA]
  ... = 25 + Real.sqrt 41 : by ring

end perimeter_Polygon_ABCDE_l818_818804


namespace number_of_trees_is_11_l818_818263

variable {N : ℕ}

-- Conditions stated by each child
def anya_statement : Prop := N = 15
def borya_statement : Prop := N % 11 = 0
def vera_statement : Prop := N < 25
def gena_statement : Prop := N % 22 = 0

-- One boy and one girl told the truth, while the other two lied
def truth_condition : Prop :=
  (borya_statement ∧ vera_statement ∧ ¬anya_statement ∧ ¬gena_statement) ∨
  (borya_statement ∧ ¬vera_statement ∧ ¬anya_statement ∧ gena_statement) ∨
  (¬borya_statement ∧ vera_statement ∧ ¬anya_statement ∧ gena_statement) ∨
  (¬borya_statement ∧ vera_statement ∧ anya_statement ∧ ¬gena_statement)

-- Proving that the number of fir trees is 11
theorem number_of_trees_is_11 (h : truth_condition) : N = 11 := 
by
  sorry

end number_of_trees_is_11_l818_818263


namespace sequence_bound_l818_818910

/-- Definition of the sequence a_n --/
def a : ℕ → ℝ
| 0       := 1
| 1       := 1
| (n + 1) := a n + 1 / a (n - 1)

/-- Theorem stating a_n >= sqrt(n) for all n ≥ 0 --/
theorem sequence_bound (n : ℕ) : a n ≥ real.sqrt n :=
sorry

end sequence_bound_l818_818910


namespace sin_seven_pi_div_six_l818_818666

theorem sin_seven_pi_div_six : Real.sin (7 * Real.pi / 6) = -1 / 2 := 
  sorry

end sin_seven_pi_div_six_l818_818666


namespace system_solution_l818_818880

noncomputable def solve_system (C1 C2 : ℝ) : ℝ → ℝ × ℝ := 
  λ t, 
  let x := C1 * Real.exp t * Real.cos (3 * t) + C2 * Real.exp t * Real.sin (3 * t) 
  in
  let y := C1 * Real.exp t * (Real.cos (3 * t) - 3 * Real.sin (3 * t)) + C2 * Real.exp t * (Real.sin (3 * t) + 3 * Real.cos (3 * t))
  in
  (x, y)

theorem system_solution (C1 C2 : ℝ) (f : ℝ → ℝ × ℝ) :
  (∀ t, (f t).1 = C1 * Real.exp t * Real.cos (3 * t) + C2 * Real.exp t * Real.sin (3 * t)) ∧
  (∀ t, (f t).2 = C1 * Real.exp t * (Real.cos (3 * t) - 3 * Real.sin (3 * t)) + C2 * Real.exp t * (Real.sin (3 * t) + 3 * Real.cos (3 * t))) →
  (∀ t, (f t) = solve_system C1 C2 t)
:= by
  sorry

end system_solution_l818_818880


namespace angle_a_b_is_150_degrees_l818_818729

variables (a b : ℝ ^ 2)

def magnitude (v : ℝ ^ 2) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)
def dot_product (v w : ℝ ^ 2) : ℝ := v.1 * w.1 + v.2 * w.2
def angle (v w : ℝ ^ 2) : real.angle :=
  real.acos (dot_product v w / (magnitude v * magnitude w))

theorem angle_a_b_is_150_degrees
  (h1 : magnitude a = 6 * real.sqrt 3)
  (h2 : magnitude b = 1)
  (h3 : dot_product a b = -9) :
  angle a b = real.angle.of_deg 150 :=
sorry

end angle_a_b_is_150_degrees_l818_818729


namespace inscribe_circle_tangent_l818_818085

variables {α : Type*} [plane_geometry α]
variables (A B C O : α) (R : ℝ) (hR : R > 0)

theorem inscribe_circle_tangent
  (h_angle : is_angle BAC) 
  (h_given_circle : is_circle O R)
  : ∃ O1 r, is_circle O1 r ∧ is_tangent O1 r O R ∧ is_inscribed O1 r BAC :=
begin
  sorry
end

end inscribe_circle_tangent_l818_818085


namespace coeff_of_x3_in_expansion_l818_818889

open Polynomial

noncomputable def poly1 : ℤ[X] := \(2 * X - 1) ^ 6
noncomputable def poly2 : ℤ[X] := X - X⁻¹

theorem coeff_of_x3_in_expansion :
  (poly1 * poly2).coeff 3 = -180 := 
sorry

end coeff_of_x3_in_expansion_l818_818889


namespace sequence_7th_term_l818_818453

/-- Define the sequence based on the given pattern -/
def sequence (n : ℕ) : ℕ := n^2 - 1

/-- The 7th term in the sequence should be 48 -/
theorem sequence_7th_term : sequence 7 = 48 :=
  by sorry

end sequence_7th_term_l818_818453


namespace lcm_of_36_and_105_l818_818675

theorem lcm_of_36_and_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end lcm_of_36_and_105_l818_818675


namespace area_of_triangle_PQR_l818_818404

noncomputable def area_triangle_PQR : ℝ :=
  let PQ := 5
  let PR := 18
  let PM := 9
  let s := (PR + PR + PQ) / 2
  in real.sqrt (s * (s - PR) * (s - PR) * (s - PQ))

theorem area_of_triangle_PQR : area_triangle_PQR = 50.134 :=
by
  unfold area_triangle_PQR
  sorry

end area_of_triangle_PQR_l818_818404


namespace sequence_count_l818_818365

-- We define the properties of the sequence
def valid_sequence (seq : Fin 8 → ℕ) : Prop :=
  (seq 0 % 2 = 1) ∧
  (∀ i : Fin 7, seq i % 2 ≠ seq (i + 1) % 2)

-- We state the theorem with the given conditions and the correct answer
theorem sequence_count : 
  {s : (Fin 8 → ℕ) // valid_sequence s}.card = 390625 := 
  by sorry

end sequence_count_l818_818365


namespace Luca_weight_loss_per_year_l818_818612

def Barbi_weight_loss_per_month : Real := 1.5
def months_in_a_year : Nat := 12
def Luca_years : Nat := 11
def extra_weight_Luca_lost : Real := 81

theorem Luca_weight_loss_per_year :
  (Barbi_weight_loss_per_month * months_in_a_year + extra_weight_Luca_lost) / Luca_years = 9 := by
  sorry

end Luca_weight_loss_per_year_l818_818612


namespace tom_needs_noodle_packages_l818_818528

def beef_pounds : ℕ := 10
def noodle_multiplier : ℕ := 2
def initial_noodles : ℕ := 4
def package_weight : ℕ := 2

theorem tom_needs_noodle_packages :
  (noodle_multiplier * beef_pounds - initial_noodles) / package_weight = 8 := 
by 
  -- Faithfully skipping the solution steps
  sorry

end tom_needs_noodle_packages_l818_818528


namespace longest_chord_of_circle_l818_818501

theorem longest_chord_of_circle (r : ℕ) (h : r = 7) : 2 * r = 14 :=
by
  rw [h]
  exact dec_trivial

end longest_chord_of_circle_l818_818501


namespace nancy_coffee_spending_l818_818851

theorem nancy_coffee_spending :
  let daily_cost := 3.00 + 2.50
  let total_days := 20
  let total_cost := total_days * daily_cost
  total_cost = 110.00 := by
    let daily_cost := 3.00 + 2.50
    let total_days := 20
    let total_cost := total_days * daily_cost
    have h1 : daily_cost = 5.50 := by norm_num
    have h2 : total_cost = 20 * 5.50 := by rw [total_cost, total_days, h1]
    have h3 : total_cost = 110.00 := by norm_num
    exact h3

end nancy_coffee_spending_l818_818851


namespace angle_YDA_eq_2_angle_YCA_l818_818417

theorem angle_YDA_eq_2_angle_YCA
  (A B C D P X Y : Point)
  (h_trapezoid : IsoscelesTrapezoid A B C D)
  (h_inter_AC_BD : P = line_intersection (line A C) (line B D))
  (h_circumcircle_intersect_BC : X ∈ circumcircle (triangle A B P) ∧ X ≠ B)
  (h_Y_on_AX : Y ∈ line A X)
  (h_parallel : parallel (line D Y) (line B C)) :
  angle Y D A = 2 * angle Y C A := 
sorry

end angle_YDA_eq_2_angle_YCA_l818_818417


namespace initial_flour_quantity_l818_818413

theorem initial_flour_quantity (hours: ℕ) (time_per_pizza: ℕ) (flour_per_pizza: ℝ) (leftover_pizzas: ℕ) (total_flour: ℝ) 
  (h1: hours = 7) 
  (h2: time_per_pizza = 10) 
  (h3: flour_per_pizza = 0.5) 
  (h4: leftover_pizzas = 2) 
  (h5: total_flour = 22.0) :
  let pizzas_per_hour := 60 / time_per_pizza in
  let total_pizzas := pizzas_per_hour * hours + leftover_pizzas in
  total_pizzas * flour_per_pizza = total_flour := 
by
  sorry

end initial_flour_quantity_l818_818413


namespace lines_skew_l818_818934

open_locale classical

variables {α β : Type*} [plane α] [plane β]

-- Two lines \( l \) and \( m \) in space (abstracted as types)
variables (l m : Type*) [line l] [line m]

-- Projections of lines \( l \) and \( m \) on plane \( \alpha \)
variables (a1 b1 : α) [projection a1] [projection b1]

-- Projections of lines \( l \) and \( m \) on plane \( \beta \)
variables (a2 b2 : β) [projection a2] [projection b2]

-- Conditions
variable (h1 : a1 ∥ b1)   -- parallel projections on plane \( \alpha \)
variable (h2 : ∃ P : β, a2 = P ∧ b2 = P)   -- intersecting projections on plane \( \beta \)

-- Mathematical problem statement
theorem lines_skew 
  (l m : Type*) [line l] [line m]
  (a1 b1 : α) [projection a1] [projection b1]
  (a2 b2 : β) [projection a2] [projection b2]
  (h1 : a1 ∥ b1) (h2 : ∃ P : β, a2 = P ∧ b2 = P) : 
  skew l m := sorry

end lines_skew_l818_818934


namespace least_positive_integer_with_six_distinct_factors_l818_818094

theorem least_positive_integer_with_six_distinct_factors : ∃ n : ℕ, (∀ k : ℕ, (number_of_factors k = 6) → (n ≤ k)) ∧ (number_of_factors n = 6) ∧ (n = 12) :=
by
  sorry

end least_positive_integer_with_six_distinct_factors_l818_818094


namespace shy_society_even_and_at_least_4_l818_818090

-- Definitions from conditions
def is_shy (group : Type) (acquaintances : group → set group) (p : group) : Prop :=
  (acquaintances p).to_finset.card ≤ 3

def has_at_least_3_shy_acquaintances (group : Type) (acquaintances : group → set group) : Prop :=
  ∀ p, 3 ≤ (acquaintances p).to_finset.filter (is_shy group acquaintances).card

-- Theorem statement
theorem shy_society_even_and_at_least_4 (group : Type) [fintype group] (acquaintances : group → set group)
  (h1 : ∀ p, (acquaintances p).to_finset.card ≤ 3)
  (h2 : has_at_least_3_shy_acquaintances group acquaintances) :
  ∃ (n : ℕ) (H : 4 ≤ n), fintype.card group = n ∧ (even n) := 
begin
  sorry
end

end shy_society_even_and_at_least_4_l818_818090


namespace proof_problem_l818_818434

variables (A B : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)

def condition1 := ∀ x : ℝ, x ≠ 3 → f x = -4 * x^2 + 11 * x + 35
def condition2 := ∀ x : ℝ, x ≠ 3 → f x / (x - 3) = A / (x - 3) + B * (x + 2)
def question := A + B = 31

theorem proof_problem : (condition1 f) → (condition2 f) → question :=
by intros h1 h2; sorry

end proof_problem_l818_818434


namespace distance_to_hospital_l818_818088

theorem distance_to_hospital {total_paid base_price price_per_mile : ℝ} (h1 : total_paid = 23) (h2 : base_price = 3) (h3 : price_per_mile = 4) : (total_paid - base_price) / price_per_mile = 5 :=
by
  sorry

end distance_to_hospital_l818_818088


namespace probability_cosine_between_0_and_1_2_l818_818470

noncomputable def geometric_probability_cosine : ℝ :=
  let interval_x := set.Icc (-real.pi / 2) (real.pi / 2)
  let event_cos := {x | 0 < real.cos x ∧ real.cos x < 1/2}
  (real.volume (interval_x ∩ event_cos)) / real.volume interval_x

theorem probability_cosine_between_0_and_1_2 :
  geometric_probability_cosine = 1/3 := 
sorry

end probability_cosine_between_0_and_1_2_l818_818470


namespace justin_tim_games_l818_818628

/-- 
Crestwood Elementary School has twelve players, Justin and Tim among them. 
Each day at recess, players split into two games of six players each. 
Each combination of six players occurs exactly once throughout the semester.
Prove that the number of times Justin and Tim play in the same game is 210.
-/
theorem justin_tim_games : 
  (∃ players : Finset ℕ, players.card = 12 ∧ 
   ((Justin Tim : ℕ) ∈ players) ∧ 
   ∀ game : Finset (Finset ℕ), (game.card = 6) ∧ 
   (∀ x ∈ game, x ⊆ players) ∧ 
    (∀ unique_game_set : Finset (Finset (Finset ℕ)), 
     unique_game_set.card = Finset.card (Finset.comb players 6)) ∧ 
    (Finset.card {G | Justin ∈ G ∧ Tim ∈ G ∧ G ∈ (Finset.comb players 6)} = 210)) :=
sorry

end justin_tim_games_l818_818628


namespace prod_of_k_l818_818209

noncomputable def g (x k : ℕ) : ℝ := k / (3 * x - 4)

theorem prod_of_k :
  ∀ k : ℝ, (g 3 k = g⁻¹ (k + 2)) → k * k = -10 / 3 :=
by
  sorry

end prod_of_k_l818_818209


namespace find_x_l818_818215

theorem find_x (x y : ℝ) (h : y ≠ -5 * x) : (x - 5) / (5 * x + y) = 0 → x = 5 := by
  sorry

end find_x_l818_818215


namespace smallest_integer_with_six_distinct_factors_l818_818132

noncomputable def least_pos_integer_with_six_factors : ℕ :=
  12

theorem smallest_integer_with_six_distinct_factors 
  (n : ℕ)
  (p q : ℕ)
  (a b : ℕ)
  (hp : prime p)
  (hq : prime q)
  (h_diff : p ≠ q)
  (h_n : n = p ^ a * q ^ b)
  (h_factors : (a + 1) * (b + 1) = 6) :
  n = least_pos_integer_with_six_factors :=
by
  sorry

end smallest_integer_with_six_distinct_factors_l818_818132


namespace area_trapezoid_and_triangle_l818_818157

theorem area_trapezoid_and_triangle
  (a : ℝ) (n : ℕ) (tg_phi : ℝ) :
  (∑ i in Finset.range n, (i + 1) * a) = a * (n * (n + 1)) / 2 ∧
  (1 / 2) * (a^2 * n^3) * tg_phi = (1 / 2) * a^2 * n^3 * tg_phi ∧
  (1 / 8) * (a^2 * n^2 * (n + 1)^2) * tg_phi = (1 / 8) * a^2 * n^2 * (n + 1)^2 * tg_phi :=
by
  sorry

end area_trapezoid_and_triangle_l818_818157


namespace range_of_a_l818_818338

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≥ f a y) ↔ (0 < a ∧ a ≤ 1) :=
by {
  let f : ℝ → ℝ := λ x, if x ≤ 1 then (a - 2) * x + 3 else 2 * a / x,
  sorry
}

end range_of_a_l818_818338


namespace geom_series_sum_l818_818540

def geom_sum (b1 : ℚ) (r : ℚ) (n : ℕ) : ℚ := 
  b1 * (1 - r^n) / (1 - r)

def b1 : ℚ := 3 / 4
def r : ℚ := 3 / 4
def n : ℕ := 15

theorem geom_series_sum :
  geom_sum b1 r n = 3177884751 / 1073741824 :=
by sorry

end geom_series_sum_l818_818540


namespace f_of_sqrt3_l818_818726

-- Definitions based on the given conditions
def f (x : ℝ) : ℝ := 
  let y := -(Real.sqrt 3) in
  if x = (Real.tan y) then (1 / (Real.cos y)^2) else sorry

-- Assertion to be proved
theorem f_of_sqrt3 : f (-Real.sqrt 3) = 4 := 
sorry

end f_of_sqrt3_l818_818726


namespace log_sqrt_sum_l818_818522

theorem log_sqrt_sum : log 10 (sqrt 5) + log 10 (sqrt 20) = 1 := 
sorry

end log_sqrt_sum_l818_818522


namespace correct_statements_l818_818896

/-- The line (3+m)x+4y-3+3m=0 (m ∈ ℝ) always passes through the fixed point (-3, 3) -/
def statement1 (m : ℝ) : Prop :=
  ∀ x y : ℝ, (3 + m) * x + 4 * y - 3 + 3 * m = 0 → (x = -3 ∧ y = 3)

/-- For segment AB with endpoint B at (3,4) and A moving on the circle x²+y²=4,
    the trajectory equation of the midpoint M of segment AB is (x - 3/2)²+(y - 2)²=1 -/
def statement2 : Prop :=
  ∀ x y x1 y1 : ℝ, ((x1, y1) : ℝ × ℝ) ∈ {p | p.1^2 + p.2^2 = 4} → x = (x1 + 3) / 2 → y = (y1 + 4) / 2 → 
    (x - 3 / 2)^2 + (y - 2)^2 = 1

/-- Given M = {(x, y) | y = √(1 - x²)} and N = {(x, y) | y = x + b},
    if M ∩ N ≠ ∅, then b ∈ [-√2, √2] -/
def statement3 (b : ℝ) : Prop :=
  ∃ x y : ℝ, y = Real.sqrt (1 - x^2) ∧ y = x + b → b ∈ [-Real.sqrt 2, Real.sqrt 2]

/-- Given the circle C: (x - b)² + (y - c)² = a² (a > 0, b > 0, c > 0) intersects the x-axis and is
    separate from the y-axis, then the intersection point of the line ax + by + c = 0 and the line
    x + y + 1 = 0 is in the second quadrant -/
def statement4 (a b c : ℝ) : Prop :=
  a > 0 → b > 0 → c > 0 → b > a → a > c →
  ∃ x y : ℝ, (a * x + b * y + c = 0 ∧ x + y + 1 = 0) ∧ x < 0 ∧ y > 0

/-- Among the statements, the correct ones are 1, 2, and 4 -/
theorem correct_statements : 
  (∀ m : ℝ, statement1 m) ∧ statement2 ∧ (∀ b : ℝ, ¬ statement3 b) ∧ 
  (∀ a b c : ℝ, statement4 a b c) :=
by sorry

end correct_statements_l818_818896


namespace smallest_possible_N_l818_818995

theorem smallest_possible_N (N : ℕ) (h : ∀ m : ℕ, m ≤ 60 → m % 3 = 0 → ∃ i : ℕ, i < 20 ∧ m = 3 * i + 1 ∧ N = 20) :
    N = 20 :=
by 
  sorry

end smallest_possible_N_l818_818995


namespace total_number_of_pupils_l818_818789

theorem total_number_of_pupils (initial_girls : ℕ) (initial_boys : ℕ) (new_girls : ℕ) :
  initial_girls = 706 →
  initial_boys = 222 →
  new_girls = 418 →
  initial_girls + new_girls + initial_boys = 1346 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_number_of_pupils_l818_818789


namespace candle_lighting_time_l818_818923

theorem candle_lighting_time :
  ∀ (ell : ℝ), 
  (∀ t₁ <= 300, t₁ = 300 → (ell / 300) * (300 - t₁) = 0) →
  (∀ t₂ <= 180, t₂ = 180 → (ell / 180) * (180 - t₂) = 0) →
  (∃ t, t = 150 → (ℝ.abs ((ell / 300) * (300 - t) - 3 * (ell / 180) * (180 - t)) = 0)) →
  let time := (5 : ℝ) - (t / 60) in
  time = (2 : ℝ) + (30 / 60) :=
sorry

end candle_lighting_time_l818_818923


namespace find_distance_l818_818087

-- Conditions: total cost, base price, cost per mile
variables (total_cost base_price cost_per_mile : ℕ)

-- Definition of the distance as per the problem
def distance_from_home_to_hospital (total_cost base_price cost_per_mile : ℕ) : ℕ :=
  (total_cost - base_price) / cost_per_mile

-- Given values:
def total_cost_value : ℕ := 23
def base_price_value : ℕ := 3
def cost_per_mile_value : ℕ := 4

-- The theorem that encapsulates the problem statement
theorem find_distance :
  distance_from_home_to_hospital total_cost_value base_price_value cost_per_mile_value = 5 :=
by
  -- Placeholder for the proof
  sorry

end find_distance_l818_818087


namespace least_positive_integer_with_six_distinct_factors_l818_818098

theorem least_positive_integer_with_six_distinct_factors : ∃ n : ℕ, (∀ k : ℕ, (number_of_factors k = 6) → (n ≤ k)) ∧ (number_of_factors n = 6) ∧ (n = 12) :=
by
  sorry

end least_positive_integer_with_six_distinct_factors_l818_818098


namespace largest_constant_c_chessboard_red_limit_l818_818697

-- Part 1: Proving the largest constant c
theorem largest_constant_c (n : ℕ) (h : 2 ≤ n) (a : Fin n → ℝ) (nonneg_conds : ∀ i, 0 ≤ a i) (sum_cond : (Finset.univ.sum a) = n) :
  ∃ c : ℝ, (∀ (a : Fin n → ℝ) (nonneg_conds : ∀ i, 0 ≤ a i) (sum_cond : (Finset.univ.sum a) = n),
  (Finset.univ.sum (λ i, 1 / (n + c * (a i) ^ 2))) ≤ (n / (n + c))) ↔ c = (n^2 - n) / (n^2 - n + 1) := 
  sorry

-- Part 2: Proving the limit of l(n) / n^2
theorem chessboard_red_limit (l : ℕ → ℕ) (exists_l : ∀ (n : ℕ), ∃ (r : Fin (n+1) → Fin (n+1) → Prop), 
  (∀ (rhs : ℕ) (h₁ : 1 ≤ rhs) (h₂ : rhs ≤ n), ∃ (i j : ℕ), (r i j) ∧ (1 ≤ i ∧ i ≤ rhs ∧ 1 ≤ j ∧ j ≤ rhs)) ∧ 
  (Finset.univ.card (λ i j, r i j) = l n)) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |l n / (n^2) - (2/7 : ℝ)| < ε :=
  sorry

end largest_constant_c_chessboard_red_limit_l818_818697


namespace solve_election_problem_l818_818791

noncomputable def number_of_votes {A B C : ℕ} (total_votes : ℕ) (invalid_percent : ℝ) (A_percent : ℝ) (B_percent : ℝ) (C_percent : ℝ) : Prop :=
  total_votes = 560000 ∧
  invalid_percent = 0.15 ∧
  A_percent = 0.75 ∧
  B_percent + C_percent = 0.25 ∧
  B_percent ≠ C_percent ∧
  C_percent ≥ 0.10 ∧
  let valid_votes := (1 - invalid_percent) * total_votes in
  A = A_percent * valid_votes ∧
  B = B_percent * valid_votes ∧
  C = C_percent * valid_votes ∧
  A = 357000 ∧
  B = 71400 ∧
  C = 47600

theorem solve_election_problem : number_of_votes 560000 0.15 0.75 0.15 0.10 :=
by
  -- Placeholder for the proof
  sorry

end solve_election_problem_l818_818791


namespace jims_sum_divided_by_anas_sum_l818_818412

noncomputable def sum_of_squares_of_odds (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n + 1), if k % 2 = 1 then (k:ℝ)^2 else 0

noncomputable def sum_of_first_n_integers (n : ℕ) : ℝ :=
  n * (n + 1) / 2

theorem jims_sum_divided_by_anas_sum : 
  (sum_of_squares_of_odds 249) / (sum_of_first_n_integers 249) = 1001 / 6 := 
by
  sorry

end jims_sum_divided_by_anas_sum_l818_818412


namespace participants_adjacent_existence_l818_818178

theorem participants_adjacent_existence:
  (circle : List (String × Bool)) -- Assume circle is a list where each element is a tuple containing a participant and their winner/prize status
  (count_prize_winner : ∃ p, (p ∈ circle) ∧ p.2 = true ∧ p.2 = "prize_winner" ∧ (List.length (filter (λ x, x.2 = "prize_winner") circle) = 20))
  (count_winner : ∃ w, (w ∈ circle) ∧ w.2 = true ∧ w.2 = "winner" ∧ (List.length (filter (λ x, x.2 = "winner") circle) = 25))
  (neighbor_rule : ∀ p, (p.2 = true) → ∃ n, (n ≠ p) ∧ (n.2 = false) ∧ ((n = circle.head ∨ n = circle.last) ∨ (n = circle.head.tail) ∨ (n = circle.last.init))):
  ∃ p1 p2, (p1 ∈ circle) ∧ (p2 ∈ circle) ∧ (p1.2 = false) ∧ (p2.2 = false) ∧ (p1 = circle.head ∨ p1 = circle.last ∨ p1 = circle.tail.head ∨ p1 = circle.init.last) := 
sorry

end participants_adjacent_existence_l818_818178


namespace smallest_six_factors_l818_818120

theorem smallest_six_factors (n : ℕ) (h : (n = 2 * 3^2)) : n = 18 :=
by {
    sorry -- proof goes here
}

end smallest_six_factors_l818_818120


namespace geometric_sequence_a_formula_l818_818714

noncomputable def a (n : ℕ) : ℤ :=
  if n = 1 then 1
  else if n = 2 then 2
  else n - 2

noncomputable def b (n : ℕ) : ℤ :=
  a (n + 1) - a n

theorem geometric_sequence (n : ℕ) (h : n ≥ 2) : 
  b n = (-1) * b (n - 1) := 
  sorry

theorem a_formula (n : ℕ) : 
  a n = (-1) ^ (n - 1) := 
  sorry

end geometric_sequence_a_formula_l818_818714


namespace nancy_coffee_expense_l818_818853

-- Definitions corresponding to the conditions
def cost_double_espresso : ℝ := 3.00
def cost_iced_coffee : ℝ := 2.50
def days : ℕ := 20

-- The statement of the problem
theorem nancy_coffee_expense :
  (days * (cost_double_espresso + cost_iced_coffee)) = 110.00 := by
  sorry

end nancy_coffee_expense_l818_818853


namespace part1_part2_l818_818827

open Nat

variable {a : ℕ → ℝ} -- Defining the arithmetic sequence
variable {S : ℕ → ℝ} -- Defining the sum of the first n terms
variable {m n p q : ℕ} -- Defining the positive integers m, n, p, q
variable {d : ℝ} -- The common difference

-- Conditions
axiom arithmetic_sequence_pos_terms : (∀ k, a k = a 1 + (k - 1) * d) ∧ ∀ k, a k > 0
axiom sum_of_first_n_terms : ∀ n, S n = (n * (2 * a 1 + (n - 1) * d)) / 2
axiom positive_common_difference : d > 0
axiom constraints_on_mnpq : n < p ∧ q < m ∧ m + n = p + q

-- Parts to prove
theorem part1 : a m * a n < a p * a q :=
by sorry

theorem part2 : S m + S n > S p + S q :=
by sorry

end part1_part2_l818_818827


namespace matchsticks_100th_stage_l818_818895

theorem matchsticks_100th_stage (a₁ : ℕ) (d : ℕ) :
  a₁ = 4 → d = 4 → ∃ n : ℕ, n = 100 ∧ a₁ + (n - 1) * d = 400 :=
begin
  intros h1 h2,
  use 100,
  split,
  { refl },
  { simp [h1, h2] },
end

end matchsticks_100th_stage_l818_818895


namespace verka_digit_sets_l818_818532

-- Define the main conditions as:
def is_three_digit_number (a b c : ℕ) : Prop :=
  let num1 := 100 * a + 10 * b + c
  let num2 := 100 * a + 10 * c + b
  let num3 := 100 * b + 10 * a + c
  let num4 := 100 * b + 10 * c + a
  let num5 := 100 * c + 10 * a + b
  let num6 := 100 * c + 10 * b + a
  num1 + num2 + num3 + num4 + num5 + num6 = 1221

-- Prove the main theorem
theorem verka_digit_sets :
  ∃ (a b c : ℕ), is_three_digit_number a a c ∧
                 ((a, c) = (1, 9) ∨ (a, c) = (2, 7) ∨ (a, c) = (3, 5) ∨ (a, c) = (4, 3) ∨ (a, c) = (5, 1)) :=
by sorry

end verka_digit_sets_l818_818532


namespace cost_per_minute_l818_818242

theorem cost_per_minute (monthly_fee cost total_bill : ℝ) (minutes : ℕ) :
  monthly_fee = 2 ∧ total_bill = 23.36 ∧ minutes = 178 → 
  cost = (total_bill - monthly_fee) / minutes → 
  cost = 0.12 :=
by
  intros h1 h2
  sorry

end cost_per_minute_l818_818242


namespace algebraic_expression_value_l818_818760

theorem algebraic_expression_value 
  (θ : ℝ)
  (a := (Real.cos θ, Real.sin θ))
  (b := (1, -2))
  (parallel : ∃ k : ℝ, a = (k * 1, k * -2)) :
  (2 * Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 5 := 
by 
  -- proof goes here 
  sorry

end algebraic_expression_value_l818_818760


namespace ball_distribution_possible_l818_818916

theorem ball_distribution_possible :
  ∃ distribution : List (List ℕ),
    (∀ box ∈ distribution, 10 ≤ box.length) ∧
    (∑ box in distribution, box.length = 800) ∧
    (∀ student : Fin 20, ∃ student_boxes : List (List ℕ), student_boxes.length = 40 ∧ student_boxes.all (λ b, b ∈ distribution)) :=
sorry

end ball_distribution_possible_l818_818916


namespace initial_red_balls_l818_818410

-- Define all the conditions as given in part (a)
variables (R : ℕ)  -- Initial number of red balls
variables (B : ℕ)  -- Number of blue balls
variables (Y : ℕ)  -- Number of yellow balls

-- The conditions
def conditions (R B Y total : ℕ) : Prop :=
  B = 2 * R ∧
  Y = 32 ∧
  total = (R - 6) + B + Y

-- The target statement proving R = 16 given the conditions
theorem initial_red_balls (R: ℕ) (B: ℕ) (Y: ℕ) (total: ℕ) 
  (h : conditions R B Y total): 
  total = 74 → R = 16 :=
by 
  sorry

end initial_red_balls_l818_818410


namespace solve_for_x_l818_818803

noncomputable def square_side_length (perimeter : ℕ) : ℕ :=
  perimeter / 4

noncomputable def square_area (side_length : ℕ) : ℕ :=
  side_length ^ 2

noncomputable def triangle_area (base height : ℕ) : ℕ :=
  (base * height) / 2

theorem solve_for_x (p_square triangle_height : ℕ) (h1 : p_square = 48) (h2 : triangle_height = 48) 
  (h3 : square_area (square_side_length p_square) = triangle_area x triangle_height) : 
  x = 6 :=
begin
  sorry
end

end solve_for_x_l818_818803


namespace smallest_integer_with_six_distinct_factors_l818_818127

noncomputable def least_pos_integer_with_six_factors : ℕ :=
  12

theorem smallest_integer_with_six_distinct_factors 
  (n : ℕ)
  (p q : ℕ)
  (a b : ℕ)
  (hp : prime p)
  (hq : prime q)
  (h_diff : p ≠ q)
  (h_n : n = p ^ a * q ^ b)
  (h_factors : (a + 1) * (b + 1) = 6) :
  n = least_pos_integer_with_six_factors :=
by
  sorry

end smallest_integer_with_six_distinct_factors_l818_818127


namespace principal_sum_l818_818978

theorem principal_sum 
  (si : ℕ)    -- Simple Interest
  (r : ℕ)     -- Rate in Percentage
  (t : ℕ)     -- Time Period in Years
  (p : ℕ)     -- Principal
  (h1 : si = 100)
  (h2 : r = 5)
  (h3 : t = 4) :
  p = 500 :=
by
  -- Use the simple interest formula: SI = (P * R * T) / 100
  have : si = (p * r * t) / 100,
  {
    sorry
  },
  calc
    p = 500        : sorry

end principal_sum_l818_818978


namespace integral_solution_l818_818672

noncomputable def indefinite_integral :=
  ∫ (f : ℝ) in (λ x : ℝ, (1 + x^(2/3))^(4/5) / x^(11/5)), id

theorem integral_solution :
  indefinite_integral = - (5/6) * (λ x : ℝ, (1 / x^(2/3) + 1)^(9 / 5)) + C :=
sorry

end integral_solution_l818_818672


namespace sequence_formula_l818_818911

theorem sequence_formula (x : ℕ → ℤ) :
  x 1 = 1 →
  x 2 = -1 →
  (∀ n, n ≥ 2 → x (n-1) + x (n+1) = 2 * x n) →
  ∀ n, x n = -2 * n + 3 :=
by
  sorry

end sequence_formula_l818_818911


namespace harriet_current_age_l818_818392

theorem harriet_current_age (peter_age harriet_age : ℕ) (mother_age : ℕ := 60) (h₁ : peter_age = mother_age / 2) 
  (h₂ : peter_age + 4 = 2 * (harriet_age + 4)) : harriet_age = 13 :=
by
  sorry

end harriet_current_age_l818_818392


namespace probability_single_white_ball_l818_818167

open Finset

-- Definitions and conditions
def total_outcomes : ℕ := (finset.card (finset.powerset_len 2 (finset.range 5))) / 2

def favorable_outcomes : ℕ := (finset.card (finset.powerset_len 1 (finset.range 2))) * (finset.card (finset.powerset_len 1 (finset.range (2, 3))))

def probability (total favorable : ℕ) : ℝ := (favorable : ℝ) / (total : ℝ)

def P_xi_1 : ℝ := 0.6

-- Theorem statement
theorem probability_single_white_ball :
  probability total_outcomes favorable_outcomes = P_xi_1 := sorry

end probability_single_white_ball_l818_818167


namespace fir_trees_alley_l818_818279

-- Define the statements made by each child
def statementAnya (N : ℕ) : Prop := N = 15
def statementBorya (N : ℕ) : Prop := N % 11 = 0
def statementVera (N : ℕ) : Prop := N < 25
def statementGena (N : ℕ) : Prop := N % 22 = 0

-- Define the condition about the truth and lies
def oneBoyOneGirlTruth (anya_vera_truth: Prop) (borya_gena_truth: Prop) : Prop :=
  anya_vera_truth ∧ borya_gena_truth ∧
  ((statementAnya N ∧ statementVera N) ∨ (statementVera N ∧ statementBorya N)) ∧
  ¬( (statementAnya N ∧ statementGena N) ∨ (statementVera N ∧ statementGena N) ∨
     (statementAnya N ∧ statementBorya N) ∨ (statementBorya N ∧ statementGena N) )

-- Prove that the number of fir trees is 11
theorem fir_trees_alley: ∃ (N : ℕ), statementBorya N ∧ statementVera N ∧ ¬ statementAnya N ∧ ¬ statementGena N ∧ oneBoyOneGirlTruth (¬ statementAnya N ∧ statementVera N) (statementBorya N ∧ ¬ statementGena N) ∧ N = 11 :=
by
  sorry

end fir_trees_alley_l818_818279


namespace smallest_six_factors_l818_818126

theorem smallest_six_factors (n : ℕ) (h : (n = 2 * 3^2)) : n = 18 :=
by {
    sorry -- proof goes here
}

end smallest_six_factors_l818_818126


namespace sub_eq_add_neg_l818_818060

theorem sub_eq_add_neg (a b : ℚ) : a - b = a + -b :=
by
  sorry

example : (-2 : ℚ) - 3 = (-2 : ℚ) + -3 :=
by
  exact sub_eq_add_neg (-2) 3

end sub_eq_add_neg_l818_818060


namespace infinitely_many_composite_an_l818_818061

def seq_a (n : ℕ) : ℕ :=
  (Finset.range n.succ).sum (λ k => (k + 1)^(k + 1))

theorem infinitely_many_composite_an : ∃ᶠ n in at_top, ¬nat.prime (seq_a n) := 
sorry

end infinitely_many_composite_an_l818_818061


namespace black_circle_percentage_l818_818442

theorem black_circle_percentage (r1 r2 r3 r4 r5 : ℝ) (A1 A2 A3 A4 A5 : ℝ) :
  r1 = 3 → r2 = r1 + 3 → r3 = r2 + 3 → r4 = r3 + 3 → r5 = r4 + 3 →
  A1 = π * r1^2 → A2 = π * r2^2 → A3 = π * r3^2 → A4 = π * r4^2 → A5 = π * r5^2 →
  let black_area := A1 + (A3 - A2) in
  (black_area / A5) * 100 = 24 :=
by intros; sorry

end black_circle_percentage_l818_818442


namespace problem_statement_l818_818757

theorem problem_statement :
  ∀ (x : ℝ), 
  let a : ℝ × ℝ := (√3, cos (2 * x)),
      b : ℝ × ℝ := (sin (2 * x), 1),
      f : ℝ → ℝ := λ x, (a.1 * b.1 + a.2 * b.2) in
  (f (π / 12) = √3 ∧ f (2 * π / 3) = -2) →
  ∀ k : ℤ, -π / 2 + k * π ≤ x → x ≤ k * π →
  let g : ℝ → ℝ := λ x, 2 * cos (2 * x + π / 2) in
  is_monotone_increasing (g) (interval (-π / 2 + k * π) (k * π)) :=
begin
  sorry
end

end problem_statement_l818_818757


namespace M_eq_ℙ_l818_818824

open Set

-- Let ℙ be the set of all primes
def ℙ := {p : ℕ | Nat.Prime p}

-- Let M be a subset of ℙ with at least three elements
variable (M : Set ℕ) (hM : M ⊆ ℙ) (hM_size : 3 ≤ M.size)

-- The main condition: for all k ≥ 1 and for all subsets A of M, A ≠ M, all prime factors of (∏ i in A, i - 1) are in M
axiom main_condition :
  ∀ k (hk : 1 ≤ k) (A : Set ℕ) (hA : A ⊆ M) (hA_ne : A ≠ M),
  ∀ p (hp : p.Prime),
  p ∈ (∏ i in A, i) - 1 → p ∈ M

-- The theorem to prove that M = ℙ
theorem M_eq_ℙ : M = ℙ :=
sorry

end M_eq_ℙ_l818_818824


namespace find_c_find_cos_2A_find_area_l818_818403

variables (a b c : ℝ) (A B C : ℝ)

-- Condition definitions
def a_value := a = sqrt 5
def b_value := b = 3
def sin_C_value := ∃ (sinA : ℝ), sin C = 2 * sin A

-- Theorem 1: Prove that c = 2 √5
theorem find_c (h1 : a_value) (h2 : b_value) (h3 : sin_C_value) : c = 2 * sqrt 5 :=
by
  sorry

-- Theorem 2: Prove that cos 2A = 3/5
theorem find_cos_2A (h1 : a_value) (h2 : b_value) (h3 : sin_C_value) (h4 : c = 2 * sqrt 5) : cos (2 * A) = 3 / 5 :=
by
  sorry

-- Theorem 3: Prove that the area of triangle ABC is 3
theorem find_area (h1 : a_value) (h2 : b_value) (h3 : sin_C_value) (h4 : c = 2 * sqrt 5) (h5 : ∃ (sinA : ℝ), sin C = 2 * sin A) : real.sqrt 3 * sin A * sin C = 3 :=
by
  sorry

end find_c_find_cos_2A_find_area_l818_818403


namespace problem_solution_l818_818767

theorem problem_solution : 
  (∃ (N : ℕ), (1 + 2 + 3) / 6 = (1988 + 1989 + 1990) / N) → ∃ (N : ℕ), N = 5967 :=
by
  intro h
  sorry

end problem_solution_l818_818767


namespace smallest_odd_factors_gt_50_l818_818694

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def has_odd_number_of_factors (n : ℕ) : Prop :=
  (Finset.filter (λ d, n % d = 0) (Finset.range (n + 1))).card % 2 = 1

theorem smallest_odd_factors_gt_50 :
  ∃ n : ℕ, n > 50 ∧ has_odd_number_of_factors n ∧ (∀ m : ℕ, m > 50 → has_odd_number_of_factors m → n ≤ m) :=
sorry

end smallest_odd_factors_gt_50_l818_818694


namespace trig_identity_example_l818_818725

theorem trig_identity_example (α : Real) (h : Real.cos α = 3 / 5) : Real.cos (2 * α) + Real.sin α ^ 2 = 9 / 25 := by
  sorry

end trig_identity_example_l818_818725


namespace johns_total_animals_l818_818415

variable (Snakes Monkeys Lions Pandas Dogs : ℕ)

theorem johns_total_animals :
  Snakes = 15 →
  Monkeys = 2 * Snakes →
  Lions = Monkeys - 5 →
  Pandas = Lions + 8 →
  Dogs = Pandas / 3 →
  Snakes + Monkeys + Lions + Pandas + Dogs = 114 :=
by
  intros hSnakes hMonkeys hLions hPandas hDogs
  rw [hSnakes] at hMonkeys
  rw [hMonkeys] at hLions
  rw [hLions] at hPandas
  rw [hPandas] at hDogs
  sorry

end johns_total_animals_l818_818415


namespace solve_trig_eq_l818_818879

theorem solve_trig_eq :
  ∀ x k : ℤ, 
    (x = 2 * π / 3 + 2 * k * π ∨
     x = 7 * π / 6 + 2 * k * π ∨
     x = -π / 6 + 2 * k * π)
    → (|Real.cos x| + Real.cos (3 * x)) / (Real.sin x * Real.cos (2 * x)) = -2 * Real.sqrt 3 := 
by
  intros x k h
  sorry

end solve_trig_eq_l818_818879


namespace a_n_eq_fraction_l818_818752

def seq_a (n : ℕ) : ℤ := 
  if n = 0 then 4
  else if n = 1 then 22
  else seq_a n - 6 * seq_a (n - 1) + seq_a (n - 2)

def seq_y (n : ℕ) : ℤ := 
  if n = 0 then 2 -- Initial value derived from conditions
  else (seq_a n - seq_a (n - 1)) / 2

def seq_x (n : ℕ) : ℤ := 
  if n = 0 then 6 -- Initial value derived from conditions
  else (seq_a n + seq_a (n - 1)) / 2

theorem a_n_eq_fraction (n : ℕ) : 
  ∀ (n : ℕ), n ≤ 1 → (seq_a n = if n = 0 then 4 else 22) ∧ 
              (n ≥ 2 → seq_a n = 6 * seq_a (n - 1) - seq_a (n - 2)) → 
  seq_a n = (seq_y n)^2 + 7 / (seq_x n - seq_y n) :=
by
  sorry

end a_n_eq_fraction_l818_818752


namespace number_of_fir_trees_is_11_l818_818269

theorem number_of_fir_trees_is_11 
  (N : ℕ)
  (Anya : N = 15)
  (Borya : N % 11 = 0)
  (Vera : N < 25)
  (Gena : N % 22 = 0)
  (OneBoyOneGirlTrue : (Anya ∨ Borya) ∧ (Vera ∨ Gena) ∧ (¬Anya ∨ ¬Borya) ∧ (¬Vera ∨ ¬Gena)) :
  N = 11 := 
sorry

end number_of_fir_trees_is_11_l818_818269


namespace calculation_correct_l818_818615

theorem calculation_correct :
  (8 * (9 + 2/5 : ℚ) - 3 = 72 + 1/5 : ℚ) := 
by 
  calc
    8 * (9 + 2/5 : ℚ) - 3 = 8 * 47/5 - 3                    : by rw [←add_div, (show 9 = 45/5, by norm_num)]
                       ... = (8 * 47) / 5 - 3               : by rw [mul_div_assoc, mul_comm]
                       ... = 376 / 5 - 3                    : by norm_num
                       ... = 376 / 5 - 15 / 5               : by norm_cast
                       ... = 361 / 5                        : by rw [sub_div]
                       ... = 72 + 1 / 5                     : by norm_cast

end calculation_correct_l818_818615


namespace fifth_house_number_is_13_l818_818066

theorem fifth_house_number_is_13 (n : ℕ) (a₁ : ℕ) (h₀ : n ≥ 5) (h₁ : (a₁ + n - 1) * n = 117) (h₂ : ∀ i, 1 ≤ i ∧ i ≤ n -> (a₁ + 2 * (i - 1)) = 2*(i-1) + a₁) : 
  (a₁ + 2 * (5 - 1)) = 13 :=
by
  sorry

end fifth_house_number_is_13_l818_818066


namespace not_perpendicular_DE_DE_l818_818191
-- Import necessary libraries

-- Definitions based on the conditions of the problem
variables {D E F D' E' F' : ℝ × ℝ}

-- Conditions as hypotheses
def is_second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

def reflect_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

-- Hypotheses based on conditions
hypothesis hD_second_quad : is_second_quadrant D
hypothesis hE_second_quad : is_second_quadrant E
hypothesis hF_second_quad : is_second_quadrant F

hypothesis hD_reflect : D' = reflect_y_eq_neg_x D
hypothesis hE_reflect : E' = reflect_y_eq_neg_x E
hypothesis hF_reflect : F' = reflect_y_eq_neg_x F

-- The theorem to be proved
theorem not_perpendicular_DE_DE' :
  ¬ (∀ D E, is_second_quadrant D → is_second_quadrant E → 
           reflect_y_eq_neg_x D = D' → reflect_y_eq_neg_x E = E' →
           slope D E * slope D' E' = -1) :=
begin
  sorry
end

end not_perpendicular_DE_DE_l818_818191


namespace seq_inequality_l818_818344

-- Define f(x) and its properties
def f (x : ℝ) : ℝ := Real.log (-x) + a * x - 1 / x

-- Define that f takes an extreme value at x = -1
def takes_extreme_value_at (f : ℝ → ℝ) (x : ℝ) : Prop := 
  (x = -1) → 
  ((∀ y:ℝ, y ≠ -1 → f y ≥ f (-1)) ∨ (∀ y:ℝ, y ≠ -1 → f y ≤ f (-1)))

noncomputable def a : ℝ := 
  if takes_extreme_value_at f (-1) then 0 else sorry

-- Define g(x) and find its minimum value
def g (x : ℝ) : ℝ := Real.log x + 2 * x + 1 / x

def g_min_value : ℝ := 
  if takes_extreme_value_at f (-1) then 3 - Real.log 2 else sorry

-- Define an and Sn sequences
def an_seq : ℕ → ℝ 
| 0 := 1 / 2
| (n + 1) := an_seq n / (an_seq n + 1)

def Sn (n : ℕ) : ℝ := (finset.range n).sum (λ i, an_seq i)

-- Prove the inequality
theorem seq_inequality (n : ℕ) (hn : 0 < n) : 
  2^n * an_seq n ≥ Real.exp (Sn n + an_seq n - 1) := sorry

end seq_inequality_l818_818344


namespace problem_1_problem_2_l818_818436

namespace MathProblem

def domain_A := { x : ℝ | -x^2 - 2x + 8 > 0 }
def range_B := { y : ℝ | ∃ (x : ℝ), y = x + 1 / (x + 1) }
def complement_R_A := { x : ℝ | x ∉ domain_A }
def set_C (a : ℝ) := { x : ℝ | (a * x - 1 / a) * (x + 4) ≤ 0 }

theorem problem_1 : domain_A ∩ range_B = { x : ℝ | (-4 < x ∧ x ≤ -3) ∨ (1 ≤ x ∧ x < 2) } :=
sorry

theorem problem_2 (a : ℝ) (h : set_C a ⊆ complement_R_A) : -real.sqrt 2 / 2 ≤ a ∧ a < 0 :=
sorry

end MathProblem

end problem_1_problem_2_l818_818436


namespace total_english_physical_format_novels_is_135_l818_818367

def total_books : ℕ := 2000
def english_books := total_books / 2
def percentage_english_novels := 0.45
def digital_percentage_novels := 0.70

def english_novels : ℕ := (percentage_english_novels * english_books).to_nat
def physical_percentage_novels := 1 - digital_percentage_novels

def english_physical_novels : ℕ := (physical_percentage_novels * english_novels).to_nat

theorem total_english_physical_format_novels_is_135 :
  english_physical_novels = 135 :=
by
  sorry

end total_english_physical_format_novels_is_135_l818_818367


namespace second_train_track_length_l818_818939

theorem second_train_track_length 
  (time_avg : ℕ) (speed_1 : ℕ) (distance_1 : ℕ) (speed_2 : ℕ) 
  (time_1 : ℕ) (time_2 : ℕ)
  (h1 : time_avg = 4)
  (h2 : speed_1 = 50)
  (h3 : distance_1 = 200)
  (h4 : speed_2 = 80)
  (h5 : time_1 = distance_1 / speed_1)
  (h6 : time_2 = (time_avg * 2 - time_1)) :
  (distance_2 : ℕ) := 
  distance_2 = speed_2 * time_2 := by sorry

end second_train_track_length_l818_818939


namespace cos_double_angle_l818_818565

theorem cos_double_angle (θ : ℝ) :
  (sin (Real.pi / 4 - θ) + cos (Real.pi / 4 - θ) = 1 / 5) → (cos (2 * θ) = -24 / 25) :=
by
  sorry

end cos_double_angle_l818_818565


namespace min_value_of_a_l818_818332

theorem min_value_of_a
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_mono : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y)
  (a : ℝ)
  (h_cond : f (Real.logb 2 a) + f (Real.logb (1/2) a) ≤ 2 * f 1) :
  a = 1/2 := sorry

end min_value_of_a_l818_818332


namespace diametrically_opposite_uncovered_points_l818_818857

theorem diametrically_opposite_uncovered_points 
    (spots: set (set (ℝ × ℝ × ℝ)))
    (finite_spots: finite spots)
    (closed_spots: ∀ s ∈ spots, is_closed s)
    (less_than_half_surface: ∀ s ∈ spots, measure (s: set (ℝ × ℝ × ℝ)) < 2 * π)
    (non_intersecting: ∀ s1 s2 ∈ spots, s1 ≠ s2 → (s1 ∩ s2) = ∅) :
  ∃ p q : ℝ × ℝ × ℝ, p ≠ q ∧ dist p q = π ∧ (∀ s ∈ spots, p ∉ s) ∧ (∀ s ∈ spots, q ∉ s) :=
sorry

end diametrically_opposite_uncovered_points_l818_818857


namespace collinear_PQR_l818_818416

open EuclideanGeometry

variables {A B C D P Q R : Point}

-- Conditions of the problem
axiom AB_eq_2CD : ∀ {AB CD : ℝ}, AB = 2 * CD
axiom A_perp_BC : ∀ {BC : Line}, Perpendicular (LineThrough A BC) BC
axiom B_perp_AD : ∀ {AD : Line}, Perpendicular (LineThrough B AD) AD
axiom C_perp_BC : ∀ {BC : Line}, Perpendicular (LineThrough C BC) BC
axiom D_perp_AD : ∀ {AD : Line}, Perpendicular (LineThrough D AD) AD
axiom P_intersection_rt : IsIntersection P (PerpendicularFrom A BC) (PerpendicularFrom B AD)
axiom Q_intersection_su : IsIntersection Q (PerpendicularFrom C BC) (PerpendicularFrom D AD)
axiom R_intersection_diagonals : IsIntersection R (LineThrough A C) (LineThrough B D)

-- The theorem to be proved
theorem collinear_PQR (A B C D P Q R : Point) :
  ∀ AB CD AD BC : Line,
  AB_eq_2CD ∧ A_perp_BC ∧ B_perp_AD ∧ C_perp_BC ∧ D_perp_AD ∧
  P_intersection_rt ∧ Q_intersection_su ∧ R_intersection_diagonals →
  Collinear P Q R := 
sorry

end collinear_PQR_l818_818416


namespace least_pos_int_with_six_factors_l818_818104

theorem least_pos_int_with_six_factors :
  ∃ n : ℕ, (∀ m : ℕ, (number_of_factors m = 6 → m ≥ n)) ∧ n = 12 := 
sorry

end least_pos_int_with_six_factors_l818_818104


namespace remainder_7_mul_12_pow_24_add_2_pow_24_mod_13_l818_818693

theorem remainder_7_mul_12_pow_24_add_2_pow_24_mod_13 :
  (7 * 12^24 + 2^24) % 13 = 8 := by
  sorry

end remainder_7_mul_12_pow_24_add_2_pow_24_mod_13_l818_818693


namespace B_score_1_probability_correct_A_wins_without_tiebreaker_probability_correct_l818_818527

def prob_A_solve : ℝ := 0.8
def prob_B_solve : ℝ := 0.75

-- Definitions for A and B scoring in rounds
def prob_B_score_1_point : ℝ := 
  prob_B_solve * (1 - prob_B_solve) + (1 - prob_B_solve) * prob_B_solve

-- Definitions for A winning without a tiebreaker
def prob_A_score_1_point : ℝ :=
  prob_A_solve * (1 - prob_A_solve) + (1 - prob_A_solve) * prob_A_solve

def prob_A_score_2_points : ℝ :=
  prob_A_solve * prob_A_solve

def prob_B_score_0_points : ℝ :=
  (1 - prob_B_solve) * (1 - prob_B_solve)

def prob_B_score_total : ℝ :=
  prob_B_score_1_point

def prob_A_wins_without_tiebreaker : ℝ :=
  prob_A_score_2_points * prob_B_score_1_point +
  prob_A_score_2_points * prob_B_score_0_points +
  prob_A_score_1_point * prob_B_score_0_points

theorem B_score_1_probability_correct :
  prob_B_score_1_point = 3 / 8 := 
by
  sorry

theorem A_wins_without_tiebreaker_probability_correct :
  prob_A_wins_without_tiebreaker = 3 / 10 := 
by 
  sorry

end B_score_1_probability_correct_A_wins_without_tiebreaker_probability_correct_l818_818527


namespace sum_of_isosceles_angles_l818_818952

-- Define the vertex positions
def A : ℝ × ℝ := (Real.cos (45 * Real.pi / 180), Real.sin (45 * Real.pi / 180))
def B : ℝ × ℝ := (Real.cos (90 * Real.pi / 180), Real.sin (90 * Real.pi / 180))
def C (θ : ℝ) : ℝ × ℝ := (Real.cos (θ * Real.pi / 180), Real.sin (θ * Real.pi / 180))

-- Define a function to check if the triangle is isosceles
def is_isosceles (θ : ℝ) : Prop :=
  let AB := (A.1 - B.1)^2 + (A.2 - B.2)^2
  let AC := (A.1 - C θ.1)^2 + (A.2 - C θ.2)^2
  let BC := (B.1 - C θ.1)^2 + (B.2 - C θ.2)^2
  (AB = AC) ∨ (AB = BC) ∨ (AC = BC)

-- Define the main theorem
theorem sum_of_isosceles_angles :
  (∑ θ in {θ | is_isosceles θ ∧ 0 ≤ θ ∧ θ ≤ 360}.to_finset, θ) = 675 := 
begin
  sorry
end

end sum_of_isosceles_angles_l818_818952


namespace least_positive_integer_with_six_factors_l818_818115

-- Define what it means for a number to have exactly six distinct positive factors
def hasExactlySixFactors (n : ℕ) : Prop :=
  (n.factorization.support.card = 2 ∧ (n.factorization.values' = [2, 1])) ∨
  (n.factorization.support.card = 1 ∧ (n.factorization.values' = [5]))

-- The main theorem statement
theorem least_positive_integer_with_six_factors : ∃ n : ℕ, hasExactlySixFactors n ∧ ∀ m : ℕ, (hasExactlySixFactors m → n ≤ m) :=
  exists.intro 12 (and.intro
    (show hasExactlySixFactors 12, by sorry)
    (show ∀ m : ℕ, hasExactlySixFactors m → 12 ≤ m, by sorry))

end least_positive_integer_with_six_factors_l818_818115


namespace probability_of_intersection_in_decagon_is_fraction_l818_818928

open_locale big_operators

noncomputable def probability_intersecting_diagonals_in_decagon : ℚ :=
let num_points := 10 in
let diagonals := (num_points.choose 2) - num_points in
let total_diagonal_pairs := (diagonals.choose 2) in
let valid_intersections := (num_points.choose 4) in
valid_intersections / total_diagonal_pairs

theorem probability_of_intersection_in_decagon_is_fraction :
  probability_intersecting_diagonals_in_decagon = 42 / 119 :=
by {
  unfold probability_intersecting_diagonals_in_decagon,
  sorry
}

end probability_of_intersection_in_decagon_is_fraction_l818_818928


namespace fir_trees_alley_l818_818280

-- Define the statements made by each child
def statementAnya (N : ℕ) : Prop := N = 15
def statementBorya (N : ℕ) : Prop := N % 11 = 0
def statementVera (N : ℕ) : Prop := N < 25
def statementGena (N : ℕ) : Prop := N % 22 = 0

-- Define the condition about the truth and lies
def oneBoyOneGirlTruth (anya_vera_truth: Prop) (borya_gena_truth: Prop) : Prop :=
  anya_vera_truth ∧ borya_gena_truth ∧
  ((statementAnya N ∧ statementVera N) ∨ (statementVera N ∧ statementBorya N)) ∧
  ¬( (statementAnya N ∧ statementGena N) ∨ (statementVera N ∧ statementGena N) ∨
     (statementAnya N ∧ statementBorya N) ∨ (statementBorya N ∧ statementGena N) )

-- Prove that the number of fir trees is 11
theorem fir_trees_alley: ∃ (N : ℕ), statementBorya N ∧ statementVera N ∧ ¬ statementAnya N ∧ ¬ statementGena N ∧ oneBoyOneGirlTruth (¬ statementAnya N ∧ statementVera N) (statementBorya N ∧ ¬ statementGena N) ∧ N = 11 :=
by
  sorry

end fir_trees_alley_l818_818280


namespace cost_per_minute_l818_818243

theorem cost_per_minute (monthly_fee cost total_bill : ℝ) (minutes : ℕ) :
  monthly_fee = 2 ∧ total_bill = 23.36 ∧ minutes = 178 → 
  cost = (total_bill - monthly_fee) / minutes → 
  cost = 0.12 :=
by
  intros h1 h2
  sorry

end cost_per_minute_l818_818243


namespace find_b_l818_818772

theorem find_b 
  (a b c : ℝ)
  (h1 : a * b * c = (sqrt ((a + 2) * (b + 3))) / (c + 1))
  (h2 : 6 * b * 11 = 1)
  (h3 : a = 6)
  (h4 : c = 11) : 
  b = 15 := 
sorry

end find_b_l818_818772


namespace total_blue_balloons_l818_818414

def joan_blue_balloons : ℕ := 60
def melanie_blue_balloons : ℕ := 85
def alex_blue_balloons : ℕ := 37
def gary_blue_balloons : ℕ := 48

theorem total_blue_balloons :
  joan_blue_balloons + melanie_blue_balloons + alex_blue_balloons + gary_blue_balloons = 230 :=
by simp [joan_blue_balloons, melanie_blue_balloons, alex_blue_balloons, gary_blue_balloons]

end total_blue_balloons_l818_818414


namespace selling_price_conditions_met_l818_818921

-- Definitions based on the problem conditions
def initial_selling_price : ℝ := 50
def purchase_price : ℝ := 40
def initial_volume : ℝ := 500
def decrease_rate : ℝ := 10
def desired_profit : ℝ := 8000
def max_total_cost : ℝ := 10000

-- Definition for the selling price
def selling_price : ℝ := 80

-- Condition: Cost is below $10000 for the valid selling price
def valid_item_count (x : ℝ) : ℝ := initial_volume - decrease_rate * (x - initial_selling_price)

-- Cost calculation function
def total_cost (x : ℝ) : ℝ := purchase_price * valid_item_count x

-- Profit calculation function 
def profit (x : ℝ) : ℝ := (x - purchase_price) * valid_item_count x

-- Main theorem statement
theorem selling_price_conditions_met : 
  profit selling_price = desired_profit ∧ total_cost selling_price < max_total_cost :=
by
  sorry

end selling_price_conditions_met_l818_818921


namespace multiplied_number_is_6_l818_818981

-- Definitions to set up the conditions of the problem
def avg_original (numbers : List ℝ) : Prop :=
  numbers.length = 5 ∧ (numbers.sum / 5 = 6.8)

def avg_new (numbers : List ℝ) (x : ℝ) : Prop :=
  ∃ (index : Fin 5), (numbers.set index (3 * (numbers.get index))).sum / 5 = 9.2

-- The theorem statement
theorem multiplied_number_is_6 (numbers : List ℝ) :
  avg_original numbers →
  (∃ x, avg_new numbers x ∧ x = 6) :=
sorry

end multiplied_number_is_6_l818_818981


namespace number_of_trees_l818_818246

theorem number_of_trees {N : ℕ} 
  (Anya : N = 15) 
  (Borya : N % 11 = 0) 
  (Vera : N < 25) 
  (Gena : N % 22 = 0) 
  (truth_conditions : (Anya ∨ Borya ∨ Vera ∨ Gena) ∧ ∃! p, p) : 
  N = 11 :=
sorry

end number_of_trees_l818_818246


namespace initial_women_count_l818_818882

-- Definitions of women and children work rates
variables (W C : ℝ)

-- Conditions given
variables
  (days_women : ℝ := 7)
  (days_children : ℝ := 14)
  (days_together : ℝ := 4)
  (num_children : ℕ := 10)

-- Hypothesis
hypothesis
  (H1 : ∃ x : ℝ, x * W * days_women = num_children * C * days_children)
  (H2 : ∃ W C : ℝ, (5 * W + num_children * C) * days_together = x * W * days_women)

-- The theorem to prove, stating that there were initially 4 women working
theorem initial_women_count : ∃ x : ℝ, x = 4 := 
sorry

end initial_women_count_l818_818882


namespace correct_answers_l818_818855

def sound_pressure_level (p p0 : ℝ) : ℝ := 20 * Real.log10 (p / p0)

variables (p1 p2 p3 p0 : ℝ)

-- Conditions based on given problem
lemma gasoline_car_sound_pressure_level :
  60 ≤ sound_pressure_level p1 p0 ∧ sound_pressure_level p1 p0 ≤ 90 :=
begin
  sorry
end

lemma hybrid_car_sound_pressure_level :
  50 ≤ sound_pressure_level p2 p0 ∧ sound_pressure_level p2 p0 ≤ 60 :=
begin
  sorry
end

lemma electric_car_sound_pressure_level :
  sound_pressure_level p3 p0 = 40 :=
begin
  sorry
end

-- Correct answers to be proved
theorem correct_answers :
  (p1 ≥ p2) ∧ ¬ (p2 > 10 * p3) ∧ (p3 = 100 * p0) ∧ (p1 ≤ 100 * p2) :=
begin
  sorry
end

end correct_answers_l818_818855


namespace probability_at_least_one_girl_l818_818173

theorem probability_at_least_one_girl 
  (boys girls : ℕ) 
  (total : boys + girls = 7) 
  (combinations_total : ℕ := Nat.choose 7 2) 
  (combinations_boys : ℕ := Nat.choose 4 2) 
  (prob_no_girls : ℚ := combinations_boys / combinations_total) 
  (prob_at_least_one_girl : ℚ := 1 - prob_no_girls) :
  boys = 4 ∧ girls = 3 → prob_at_least_one_girl = 5 / 7 := 
by
  intro h
  cases h
  sorry

end probability_at_least_one_girl_l818_818173


namespace oscar_classes_l818_818864

theorem oscar_classes (questions_per_student : ℕ) (students_per_class : ℕ) (total_questions : ℕ) :
  questions_per_student = 10 → students_per_class = 35 → total_questions = 1750 → 
  total_questions / (students_per_class * questions_per_student) = 5 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- Skipping proof steps
  sorry

end oscar_classes_l818_818864


namespace missed_bus_time_by_l818_818633

def bus_departure_time : Time := Time.mk 8 0 0
def travel_time_minutes : Int := 30
def departure_time_home : Time := Time.mk 7 50 0
def arrival_time_pickup_point : Time := 
  departure_time_home.addMinutes travel_time_minutes

theorem missed_bus_time_by :
  arrival_time_pickup_point.diff bus_departure_time = 20 * 60 :=
by
  sorry

end missed_bus_time_by_l818_818633


namespace unique_number_between_5_and_9_greater_than_7_l818_818690

theorem unique_number_between_5_and_9_greater_than_7 : ∃! (x : ℕ), 5 < x ∧ x < 9 ∧ 7 < x :=
by
  existsi 8
  split
  { split
    { linarith }
    { linarith } }
  { intros y H
    cases H with Hx1 Hx2
    cases Hx2 with Hx2 Hx3
    omega }
  sorry

end unique_number_between_5_and_9_greater_than_7_l818_818690


namespace projection_of_b_onto_a_l818_818358

variables (a b : ℝ^3)
variables (h_a : ∥a∥ = 3) (h_b : ∥b∥ = 2 * Real.sqrt 3) (h_perpendicular : a ⬝ (a + b) = 0)

theorem projection_of_b_onto_a : (a ⊗ b) / ∥a∥^2 * a = - a :=
by sorry

end projection_of_b_onto_a_l818_818358


namespace least_positive_integer_with_six_factors_is_18_l818_818140

-- Define the least positive integer with exactly six distinct positive factors.
def least_positive_with_six_factors (n : ℕ) : Prop :=
  (∀ d : ℕ, d ∣ n → d > 0) ∧ (finset.card (finset.filter (λ d, d ∣ n) (finset.range (n + 1)))) = 6

-- Prove that the least positive integer with exactly six distinct positive factors is 18.
theorem least_positive_integer_with_six_factors_is_18 : (∃ n : ℕ, least_positive_with_six_factors n ∧ n = 18) :=
sorry


end least_positive_integer_with_six_factors_is_18_l818_818140


namespace allie_distance_to_meet_billie_l818_818938

-- Definitions for the conditions
def distance (A B : ℝ) := 200
def allie_speed := 10
def billie_speed := 7
def angle_with_AB := 45

-- Problem in Lean 4
theorem allie_distance_to_meet_billie
  (A B C : ℝ)
  (h1 : distance A B = 200)
  (h2 : allie_speed = 10)
  (h3 : billie_speed = 7)
  (h4 : angle_with_AB = 45) :
  let t := 20
  in allie_speed * t = 200 :=
by
  sorry

end allie_distance_to_meet_billie_l818_818938


namespace age_ratio_4_years_hence_4_years_ago_l818_818508

-- Definitions based on the conditions
def current_age_ratio (A B : ℕ) := 5 * B = 3 * A
def age_ratio_4_years_ago_4_years_hence (A B : ℕ) := A - 4 = B + 4

-- The main theorem to prove
theorem age_ratio_4_years_hence_4_years_ago (A B : ℕ) 
  (h1 : current_age_ratio A B) 
  (h2 : age_ratio_4_years_ago_4_years_hence A B) : 
  A + 4 = 3 * (B - 4) := 
sorry

end age_ratio_4_years_hence_4_years_ago_l818_818508


namespace sum_of_a_l818_818348

def f (x : ℝ) : ℝ :=
if x < 0 then 3^(x - 4) else log 2 x

theorem sum_of_a (a : ℝ) (h : ∀ x, f x > a ↔ x ∈ set.Ioo (a^2) ⊤) :
  a = 2 ∨ a = 4 → 6 :=
sorry

end sum_of_a_l818_818348


namespace scientific_notation_of_0_000000007_l818_818572

theorem scientific_notation_of_0_000000007 :
  (0.000000007 : ℝ) = 7 * 10^(-9) :=
sorry

end scientific_notation_of_0_000000007_l818_818572


namespace surface_area_one_is_three_pi_l818_818067

-- Conditions: definition of a hemisphere with radius 1
def surface_area_hemisphere (R : ℝ) := 2 * Real.pi * R^2 + Real.pi * R^2

-- Proof problem: prove that given R = 1, the surface area is 3 * π
theorem surface_area_one_is_three_pi : surface_area_hemisphere 1 = 3 * Real.pi :=
by
  sorry

end surface_area_one_is_three_pi_l818_818067


namespace number_of_fir_trees_l818_818290

def anya_statement (N : ℕ) : Prop := N = 15
def borya_statement (N : ℕ) : Prop := 11 ∣ N
def vera_statement (N : ℕ) : Prop := N < 25
def gena_statement (N : ℕ) : Prop := 22 ∣ N

def one_boy_one_girl_truth (A B G V : Prop) : Prop :=
  (A ∨ V) ∧ ¬(A ∧ V) ∧ (B ∨ G) ∧ ¬(B ∧ G)

theorem number_of_fir_trees (N : ℕ) :
  anya_statement N ∨ borya_statement N ∨ vera_statement N ∨ gena_statement N ∧
  one_boy_one_girl_truth (anya_statement N) (borya_statement N) (gena_statement N) (vera_statement N) :=
  N = 11 :=
sorry

end number_of_fir_trees_l818_818290


namespace min_distance_line_curve_l818_818006

/-- 
  Given line l with parametric equations:
    x = 1 + t * cos α,
    y = t * sin α,
  and curve C with the polar equation:
    ρ * sin^2 θ = 4 * cos θ,
  prove:
    1. The Cartesian coordinate equation of C is y^2 = 4x.
    2. The minimum value of the distance |AB|, where line l intersects curve C, is 4.
-/
theorem min_distance_line_curve {t α θ ρ x y : ℝ} 
  (h_line_x: x = 1 + t * Real.cos α)
  (h_line_y: y = t * Real.sin α)
  (h_curve_polar: ρ * (Real.sin θ)^2 = 4 * Real.cos θ)
  (h_alpha_range: 0 < α ∧ α < Real.pi) : 
  (∀ {x y}, y^2 = 4 * x) ∧ (min_value_of_AB = 4) :=
sorry

end min_distance_line_curve_l818_818006


namespace total_hours_l818_818457

-- Definitions of the conditions
variable {K : ℕ} -- Kate's hours
variable {P : ℕ} -- Pat's hours
variable {M : ℕ} -- Mark's hours

-- Conditions
def condition1 : Prop := P = 2 * K
def condition2 : Prop := P = 1 / 3 * M
def condition3 : Prop := M = K + 65

-- Theorem statement
theorem total_hours : condition1 ∧ condition2 ∧ condition3 → (P + K + M = 117) :=
by 
  sorry

end total_hours_l818_818457


namespace angle_opposite_c_is_90_degrees_l818_818384

theorem angle_opposite_c_is_90_degrees (a b c : ℝ) (h : (a + b + c) * (a + b - c) = 2 * a * b) : 
  ∠A = 90 := by
  sorry

end angle_opposite_c_is_90_degrees_l818_818384


namespace train_pass_man_in_16_seconds_l818_818600

noncomputable def speed_km_per_hr := 54
noncomputable def speed_m_per_s := (speed_km_per_hr * 1000) / 3600
noncomputable def time_to_pass_platform := 16
noncomputable def length_platform := 90.0072
noncomputable def length_train := speed_m_per_s * time_to_pass_platform
noncomputable def time_to_pass_man := length_train / speed_m_per_s

theorem train_pass_man_in_16_seconds :
  time_to_pass_man = 16 :=
by sorry

end train_pass_man_in_16_seconds_l818_818600


namespace tan_sum_of_roots_l818_818325

theorem tan_sum_of_roots (α β : ℝ) (hαβ : ∀ x, (2 * x ^ 2 + 3 * x - 7 = 0) ↔ (x = tan α ∨ x = tan β)) :
  tan (α + β) = -1 / 3 :=
by
  sorry

end tan_sum_of_roots_l818_818325


namespace probability_of_intersection_in_decagon_is_fraction_l818_818929

open_locale big_operators

noncomputable def probability_intersecting_diagonals_in_decagon : ℚ :=
let num_points := 10 in
let diagonals := (num_points.choose 2) - num_points in
let total_diagonal_pairs := (diagonals.choose 2) in
let valid_intersections := (num_points.choose 4) in
valid_intersections / total_diagonal_pairs

theorem probability_of_intersection_in_decagon_is_fraction :
  probability_intersecting_diagonals_in_decagon = 42 / 119 :=
by {
  unfold probability_intersecting_diagonals_in_decagon,
  sorry
}

end probability_of_intersection_in_decagon_is_fraction_l818_818929


namespace number_of_true_propositions_is_two_l818_818424

def planes_and_lines (α β γ : Type) [plane α] [plane β] [plane γ]
  (l m n : Type) [line l] [line m] [line n] :=
  (α ≠ β) → (β ≠ γ) → (α ≠ γ) → (l ≠ m) → (m ≠ n) → (n ≠ l) → 
  (α ⊥ γ → β ⊥ γ → α ∥ β) ↔ 
  (m ⊆ α → n ⊆ α → m ∥ β → n ∥ β → α ∥ β) ↔ 
  (α ∥ β → l ⊆ α → l ∥ β) ↔ 
  (α ∩ β = l → β ∩ γ = m → γ ∩ α = n → l ∥ γ → m ∥ n) →
  true.

-- Proving the number of true propositions is exactly 2
theorem number_of_true_propositions_is_two
  (α β γ : Type) [plane α] [plane β] [plane γ]
  (l m n : Type) [line l] [line m] [line n] :
  planes_and_lines α β γ l m n →
  number_of_true_propositions α β γ l m n = 2 :=
sorry

end number_of_true_propositions_is_two_l818_818424


namespace number_of_concave_functions_l818_818608

theorem number_of_concave_functions :
  let f1 (x : ℝ) := 2^x
  let f2 (x : ℝ) := log x / log 2
  let f3 (x : ℝ) := x^2
  let f4 (x : ℝ) := cos (2*x)
  (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < 1 →
    f2 ((x1 + x2) / 2) > (f2 x1 + f2 x2) / 2)
  ∧ (¬ (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < 1 →
    f1 ((x1 + x2) / 2) > (f1 x1 + f1 x2) / 2))
  ∧ (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < 1 →
    f3 ((x1 + x2) / 2) > (f3 x1 + f3 x2) / 2 = false)
  ∧ (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < 1 →
    f4 ((x1 + x2) / 2) > (f4 x1 + f4 x2) / 2 = false) :=
by
  let f1 : ℝ → ℝ := λ x, 2^x
  let f2 : ℝ → ℝ := λ x, log x / log 2
  let f3 : ℝ → ℝ := λ x, x^2
  let f4 : ℝ → ℝ := λ x, cos (2*x)
  sorry

end number_of_concave_functions_l818_818608


namespace total_students_l818_818529

-- Define the initial number of students per school
def init_students_per_school (n : ℕ) : Prop :=
  ∃ S, S = n

-- Define the car capacities
def car_capacity_Xiaoxin (c₁ : ℕ) : Prop :=
  c₁ = 15

def car_capacity_Xiaoxiao (c₂ : ℕ) : Prop :=
  c₂ = 13

-- Define the initial car conditions
def car_condition_initial (x₁ x₂ : ℕ) : Prop :=
  x₂ = x₁ + 1

-- Define the condition after adding one participant to each school
def car_condition_one_more (x₁ x₂ : ℕ) : Prop :=
  x₁ = x₂

-- Define the condition after adding another participant to each school
def car_condition_final (x₁ x₂ : ℕ) : Prop :=
  x₂ = x₁ + 1

-- Final statement: prove the total number of students in the end is 184
theorem total_students (n : ℕ) (S₁ S₂ : ℕ) (c₁ c₂ x₁ x₂ : ℕ) :
  init_students_per_school (n) →
  car_capacity_Xiaoxin (c₁) →
  car_capacity_Xiaoxiao (c₂) →
  car_condition_initial (x₁ x₂) →
  car_condition_one_more (x₁ + 1 x₂ + 1) →
  car_condition_final (x₁ + 2 x₂ + 2) →
  T = S₁ + S₂ →
  T = 184 :=
sorry

end total_students_l818_818529


namespace polynomial_root_problem_l818_818643

noncomputable def alpha : ℝ := sorry

theorem polynomial_root_problem (α : ℝ) (h : α^2 = 2*α + 2) :
  α^5 - 44*α - 32 = 0 :=
by
  have h3 : α^3 = 6*α + 4, by sorry
  have h4 : α^4 = 16*α + 12, by sorry
  have h5 : α^5 = 44*α + 32, by sorry
  sorry

end polynomial_root_problem_l818_818643


namespace EF_perp_AI_l818_818782

-- Define the basic elements of the problem configuration
variables (A B C I D N M E F : Type)
noncomputable def is_incenter (I : Type) : Prop := sorry -- definition of incenter
noncomputable def is_perp (l1 l2 : Type) : Prop := sorry -- definition of perpendicular lines
noncomputable def is_midpoint (M A B : Type) : Prop := sorry -- definition of midpoint

-- Conditions as per the problem statement
axiom incenter_I : is_incenter I
axiom perp_lB_CI : is_perp B CI
axiom perp_lC_BI : is_perp C BI
axiom intersect_D : D = Intersection lB lC
axiom intersect_N : N = Intersection lB AC
axiom intersect_M : M = Intersection lC AB
axiom midpoint_E : is_midpoint E B N
axiom midpoint_F : is_midpoint F C M

-- Lean 4 statement of the proof problem
theorem EF_perp_AI : EF ⟂ AI :=
sorry

end EF_perp_AI_l818_818782


namespace angle_opposite_c_is_90_degrees_l818_818383

theorem angle_opposite_c_is_90_degrees (a b c : ℝ) (h : (a + b + c) * (a + b - c) = 2 * a * b) : 
  ∠A = 90 := by
  sorry

end angle_opposite_c_is_90_degrees_l818_818383


namespace calculate_expression_l818_818617

theorem calculate_expression :
  (-1 : ℤ) ^ 2023 - abs (1 - real.sqrt 3) + real.sqrt 6 * real.sqrt (1/2) = 0 :=
by
  sorry

end calculate_expression_l818_818617


namespace find_k_and_other_root_l818_818017

def quadratic_eq (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

theorem find_k_and_other_root (k β : ℝ) (h1 : quadratic_eq 4 k 2 (-0.5)) (h2 : 4 * (-0.5) ^ 2 + k * (-0.5) + 2 = 0) : 
  k = 6 ∧ β = -1 ∧ quadratic_eq 4 k 2 β := 
by 
  sorry

end find_k_and_other_root_l818_818017


namespace arithmetic_sequence_problem_l818_818802

theorem arithmetic_sequence_problem
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : a 1 + 3 * a 8 + a 15 = 120)
  : 2 * a 9 - a 10 = 24 := by
  sorry

end arithmetic_sequence_problem_l818_818802


namespace parabola_equation_l818_818905

open Classical
noncomputable theory

variables {p x y : ℝ}

def parabola (p : ℝ) := ∃ x y : ℝ, y^2 = 2 * p * x

def focus_dist (p : ℝ) := ∃ M : ℝ × ℝ, (M.1 = 2 * p ∧ |2 * p| = 4 * |p / 2|)

def point_on_parabola (p : ℝ) := ∃ x y : ℝ, x = 3 * p / 2 ∧ y = ±sqrt (3 * p)

def triangle_area (p : ℝ) := ∃ x y : ℝ, (1 / 2 * (p / 2) * (sqrt (3 * p))) = 4 * sqrt 3

theorem parabola_equation (h : 0 < p) 
    (h1 : parabola p)
    (h2 : focus_dist p)
    (h3 : point_on_parabola p)
    (h4 : triangle_area p) : y^2 = 8 * x :=
sorry

end parabola_equation_l818_818905


namespace salary_restoration_l818_818154

-- Define the original salary S
variable (S : ℝ)

-- Define the reduced salary after a 20% reduction
def reduced_salary : ℝ := 0.80 * S

-- The increment needed to bring the reduced salary back to the original salary
def increment := S - reduced_salary S

-- The percentage increase needed
def percentage_increase := (increment S / reduced_salary S) * 100

-- Statement to be proved
theorem salary_restoration : percentage_increase S = 25 := by
  sorry

end salary_restoration_l818_818154


namespace prove_impossible_equal_sugar_l818_818976

-- Define the initial setup and conditions
structure JarsSetup where
  jar2_volume : ℕ -- volume in ml
  jar2_sugar   : ℕ -- sugar in grams
  jar3_volume : ℕ -- volume in ml
  jar3_sugar   : ℕ -- sugar in grams
  measure_cup : ℕ -- volume in ml
  h1 : jar2_volume = 700
  h2 : jar2_sugar = 50
  h3 : jar3_volume = 800
  h4 : jar3_sugar = 60
  h5 : measure_cup = 100

-- Define the type of the impossibility proof
def impossible_equal_sugar (setup : JarsSetup) : Prop :=
  ∀ jar2_sugar_final jar3_sugar_final, 
    (jar2_sugar_final = jar3_sugar_final) → False

-- The main statement
theorem prove_impossible_equal_sugar (setup : JarsSetup) : impossible_equal_sugar setup := 
  by
    intro jar2_sugar_final jar3_sugar_final
    intro h_eq
    sorry

end prove_impossible_equal_sugar_l818_818976


namespace february_25_is_wednesday_l818_818374

theorem february_25_is_wednesday
  (feb_13_friday : ℕ → weekday) -- February 13 is a Friday in a non-leap year
  (H : feb_13_friday 13 = weekday.friday) : 
  feb_13_friday 25 = weekday.wednesday :=
sorry

end february_25_is_wednesday_l818_818374


namespace maximize_profit_at_optimal_quantity_l818_818996

/-- Fixed cost of production in ten thousand yuan -/
def fixed_cost : ℝ := 0.5

/-- Variable cost per 100 units in ten thousand yuan -/
def variable_cost_per_100_units : ℝ := 0.25

/-- Revenue function in ten thousand yuan, where x is in hundreds of units -/
def revenue (x : ℝ) : ℝ := 5 * x - (1/2) * x^2

/-- Profit function as a function of production quantity in hundreds of units -/
def profit (x : ℝ) : ℝ := revenue x - (fixed_cost + variable_cost_per_100_units * x)

/-- The production quantity that maximizes profit -/
def optimal_production_quantity : ℝ := 4.75

theorem maximize_profit_at_optimal_quantity : 
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → profit x ≤ profit optimal_production_quantity :=
by
  sorry

end maximize_profit_at_optimal_quantity_l818_818996


namespace jerry_age_is_13_l818_818446

variable (M J : ℕ)

theorem jerry_age_is_13 (h1 : M = 2 * J - 6) (h2 : M = 20) : J = 13 := by
  sorry

end jerry_age_is_13_l818_818446


namespace triangle_BPQ_area_l818_818713

theorem triangle_BPQ_area
  (ABC : Triangle)
  (right_angle_C : ABC.angle_at C = 90)
  (BC_length : ABC.side_length B C = 26)
  (circle_on_BC : Circle { diameter := 26, center := midpoint B C })
  (AP_tangent : Tangent { point := A, circle := circle_on_BC, except_side := AC })
  (PH_perpendicular : Perpendicular { from := P, to_segment := BC, intersects := AB Q })
  (BH_CH_ratio : Ratio BH CH = 4 / 9) :
  Area (Triangle B P Q) = 24 := 
sorry

end triangle_BPQ_area_l818_818713


namespace no_negative_roots_l818_818646

theorem no_negative_roots (x : ℝ) :
  x^4 - 4 * x^3 - 6 * x^2 - 3 * x + 9 = 0 → 0 ≤ x :=
by
  sorry

end no_negative_roots_l818_818646


namespace price_of_each_shirt_l818_818822

-- Defining the conditions
def total_pants_cost (pants_price : ℕ) (num_pants : ℕ) := num_pants * pants_price
def total_amount_spent (amount_given : ℕ) (change_received : ℕ) := amount_given - change_received
def total_shirts_cost (amount_spent : ℕ) (pants_cost : ℕ) := amount_spent - pants_cost
def price_per_shirt (shirts_total_cost : ℕ) (num_shirts : ℕ) := shirts_total_cost / num_shirts

-- The main statement
theorem price_of_each_shirt (pants_price num_pants amount_given change_received num_shirts : ℕ) :
  num_pants = 2 →
  pants_price = 54 →
  amount_given = 250 →
  change_received = 10 →
  num_shirts = 4 →
  price_per_shirt (total_shirts_cost (total_amount_spent amount_given change_received) 
                   (total_pants_cost pants_price num_pants)) num_shirts = 33
:= by
  sorry

end price_of_each_shirt_l818_818822


namespace least_positive_integer_with_six_factors_l818_818110

theorem least_positive_integer_with_six_factors :
  ∃ n : ℕ, (∀ m : ℕ, m > 0 → m < n → (count_factors m ≠ 6)) ∧ count_factors n = 6 ∧ n = 18 :=
sorry

noncomputable def count_factors (n : ℕ) : ℕ :=
sorry

end least_positive_integer_with_six_factors_l818_818110


namespace angles_with_same_terminal_side_as_60deg_l818_818912

def angles_same_terminal_side_set (α : ℝ) : Prop :=
  ∃ k : ℤ, α = 2 * k * Real.pi + Real.pi / 3

theorem angles_with_same_terminal_side_as_60deg :
  angles_same_terminal_side_set :=
sorry

end angles_with_same_terminal_side_as_60deg_l818_818912


namespace algebraic_expression_value_l818_818770

theorem algebraic_expression_value (x y : ℝ) (h1 : x + 2 * y = 4) (h2 : x - 2 * y = -1) : 
  x^2 - 4 * y^2 = -4 :=
by
  sorry

end algebraic_expression_value_l818_818770


namespace sum_common_arithmetic_seq_l818_818525

theorem sum_common_arithmetic_seq :
  let seq1 := {n | ∃ k, n = 2 + 4 * k ∧ n ≤ 190}
  let seq2 := {n | ∃ k, n = 2 + 6 * k ∧ n ≤ 200}
  let common_seq := seq1 ∩ seq2
  let terms := (∃! n, common_seq n) -- uniqueness conditions for constructing the sequence from the set
  in (∑ n in common_seq, n) = 1472 :=
by
  sorry

end sum_common_arithmetic_seq_l818_818525


namespace solution_interval_exists_l818_818056

noncomputable def f (x : ℝ) : ℝ := (2 : ℝ)^x + x - 7

theorem solution_interval_exists : ∃ x ∈ Ioo (2 : ℝ) 3, f x = 0 := sorry

end solution_interval_exists_l818_818056


namespace simplify_expression_l818_818032

open BigOperators

def a : ℕ := 5488000000
def b : ℕ := 64

theorem simplify_expression : (∛(a) * ∛(b) : ℝ) = 5600 * ∛(2) :=
by
  sorry

end simplify_expression_l818_818032


namespace transform_square_a_transform_square_b_transform_square_c_transform_square_d_l818_818983

noncomputable def transform_a (z: ℂ) : ℂ := complex.I * z
noncomputable def transform_b (z: ℂ) : ℂ := 2 * complex.I * z - 1
noncomputable def transform_c (z: ℂ) : ℂ := z ^ 2
noncomputable def transform_d (z: ℂ) : ℂ := z⁻¹

theorem transform_square_a :
  let A := (0 : ℂ)
  let B := (2 * complex.I)
  let C := (2 + 2 * complex.I)
  let D := (2 : ℂ)
  ∀ (z : ℂ), z ∈ {A, B, C, D} →
    transform_a z ∈ {(0 : ℂ), (-2 : ℂ), (-2 + 2 * complex.I), (2 * complex.I)} := by
  sorry

theorem transform_square_b :
  let A := (0 : ℂ)
  let B := (2 * complex.I)
  let C := (2 + 2 * complex.I)
  let D := (2 : ℂ)
  ∀ (z : ℂ), z ∈ {A, B, C, D} →
    transform_b z ∈ {(-1 : ℂ), (-5 : ℂ), (-4 + 4 * complex.I), (1- 4*complex.I)} := by
  sorry

theorem transform_square_c :
  let A := (0 : ℂ)
  let B := (2 * complex.I)
  let C := (2 + 2 * complex.I)
  let D := (2 : ℂ)
  ∀ (z : ℂ), z ∈ {A, B, C, D} →
    transform_c z ∈ {(0 : ℂ), (-4 : ℂ), (-8, -15*complex.I), (4: ℂ)} := by
  sorry

theorem transform_square_d :
  let A := (0 : ℂ)
  let B := (2 * complex.I)
  let C := (2 + 2 * complex.I)
  let D := (2 : ℂ)
  ∀ (z : ℂ), z ∈ {A, B, C, D} → 
  transform_d z ∈ {z⁻¹ | z≠0} :=
by sorry

end transform_square_a_transform_square_b_transform_square_c_transform_square_d_l818_818983


namespace max_red_squares_no_trimino_min_red_squares_every_trimino_l818_818943

section Chessboard

-- Define an 8x8 chessboard
def Chessboard := Fin 8 × Fin 8

-- Define a trimino as a set of three adjacent squares
def is_adjacent (a b : Chessboard) : Prop :=
  (a.1, a.2 + 1) = b ∨ (a.1, a.2 - 1) = b ∨ (a.1 + 1, a.2) = b ∨ (a.1 - 1, a.2) = b

def is_trimino (squares : Finset Chessboard) : Prop :=
  ∃ a b c : Chessboard, 
    a ∈ squares ∧ b ∈ squares ∧ c ∈ squares ∧ 
    is_adjacent a b ∧ is_adjacent b c ∧ 
    ¬ is_adjacent a c

-- Define the no red trimino condition and maximum red squares
theorem max_red_squares_no_trimino : ∀ colored_squares : Finset Chessboard,
  (∀ trimino : Finset Chessboard, is_trimino trimino → ¬ trimino ⊆ colored_squares) →
  colored_squares.card ≤ 32 :=
sorry

-- Define the at least one red square per trimino condition
theorem min_red_squares_every_trimino : ∀ colored_squares : Finset Chessboard,
  (∀ trimino : Finset Chessboard, is_trimino trimino → (trimino ∩ colored_squares).nonempty) →
  32 ≤ colored_squares.card :=
sorry

end Chessboard

end max_red_squares_no_trimino_min_red_squares_every_trimino_l818_818943


namespace cloth_sold_worth_l818_818151

-- Define the commission rate and commission received
def commission_rate := 0.05
def commission_received := 12.50

-- State the theorem to be proved
theorem cloth_sold_worth : commission_received / commission_rate = 250 :=
by
  sorry

end cloth_sold_worth_l818_818151


namespace missed_bus_time_by_l818_818632

def bus_departure_time : Time := Time.mk 8 0 0
def travel_time_minutes : Int := 30
def departure_time_home : Time := Time.mk 7 50 0
def arrival_time_pickup_point : Time := 
  departure_time_home.addMinutes travel_time_minutes

theorem missed_bus_time_by :
  arrival_time_pickup_point.diff bus_departure_time = 20 * 60 :=
by
  sorry

end missed_bus_time_by_l818_818632


namespace sum_bijections_l818_818418

theorem sum_bijections (f g : ℕ → ℕ) (h₁ : ∀ x, 1 ≤ x ∧ x ≤ 2016 → 1 ≤ f(x) ∧ f(x) ≤ 2016) (h₂ : ∀ y, 1 ≤ y ∧ y ≤ 2016 → 1 ≤ g(y) ∧ g(y) ≤ 2016) (h₃ : function.bijective f) (h₄ : function.bijective g) :
  ∑ i in finset.range 2016, ∑ j in finset.range 2016, (f (i+1) - g (j+1)) ^ 2559 = 0 :=
by 
  sorry

end sum_bijections_l818_818418


namespace poisson_sum_conditional_binomial_poisson_diff_l818_818836
noncomputable theory

-- Define the parameters and distributions
variables (λ μ : ℝ) (hλ : λ > 0) (hμ : μ > 0)

-- Part (a): X + Y ~ Poisson(λ + μ)
theorem poisson_sum (X Y : ℕ → ℕ) 
  (hX : ∀ k, P (X = k) = (λ^k * exp(-λ)) / (k.factorial)) 
  (hY : ∀ k, P (Y = k) = (μ^k * exp(-μ)) / (k.factorial))
  (hXY_ind : ∀ k m, P (X = k ∧ Y = m) = P (X = k) * P (Y = m)):
  ∀ n, P (X + Y = n) = ((λ + μ)^n * exp(-(λ + μ))) / n.factorial :=
sorry

-- Part (b): P(X = k | X + Y = n) is binomial with parameters n and λ / (λ + μ)
theorem conditional_binomial (X Y : ℕ → ℕ) 
  (hX : ∀ k, P (X = k) = (λ^k * exp(-λ)) / (k.factorial)) 
  (hY : ∀ k, P (Y = k) = (μ^k * exp(-μ)) / (k.factorial))
  (hXY_ind : ∀ k m, P (X = k ∧ Y = m) = P (X = k) * P (Y = m))
  (n k : ℕ) (hnk : 0 ≤ k ∧ k ≤ n):
  P (X = k ∣ X + Y = n) = (nat.choose n k) * (λ / (λ + μ))^k * (μ / (λ + μ))^(n - k) :=
sorry

-- Part (c): Distribution of X - Y involving the modified Bessel function
theorem poisson_diff (X Y : ℕ → ℕ) 
  (hX : ∀ k, P (X = k) = (λ^k * exp(-λ)) / (k.factorial)) 
  (hY : ∀ k, P (Y = k) = (μ^k * exp(-μ)) / (k.factorial)) 
  (hXY_ind : ∀ k m, P (X = k ∧ Y = m) = P (X = k) * P (Y = m)) 
  (k : ℤ):
  P (X - Y = k) = exp(-(λ + μ)) * ((λ / μ)^(k / 2)) * modified_bessel_first_kind k (2 * real.sqrt (λ * μ)) :=
sorry

end poisson_sum_conditional_binomial_poisson_diff_l818_818836


namespace sum_fractions_in_simplest_form_l818_818951

def gcd (a b : ℕ) : ℕ := if b = 0 then a else gcd b (a % b)

def is_reduced (k : ℕ) : Prop := gcd k 2014 = 1

def sum_of_fractions : ℕ :=
  let count_reduced := (Finset.range 2014).filter is_reduced
  ∑ k in count_reduced, k / 2014

theorem sum_fractions_in_simplest_form :
  sum_of_fractions = 468 :=
sorry

end sum_fractions_in_simplest_form_l818_818951


namespace graph_translation_sin_l818_818082

theorem graph_translation_sin (
  x : ℝ
) :
  let f1 := λ x : ℝ, sin (2 * x)
  let f2 := λ x : ℝ, sin (2 * x - π / 3)
  let translation := π / 6
  f2 = λ x, f1 (x - translation)
:= sorry

end graph_translation_sin_l818_818082


namespace max_value_function_l818_818156

theorem max_value_function (x y z : ℝ) (hx : 0 < x ∧ x < sqrt 5) (hy : 0 < y ∧ y < sqrt 5) (hz : 0 < z ∧ z < sqrt 5)
  (hxyz : x^4 + y^4 + z^4 ≥ 27) : 
  (1 / (x^2 - 5)) + (1 / (y^2 - 5)) + (1 / (z^2 - 5)) ≤ -3 * sqrt 3 / 2 :=
sorry

end max_value_function_l818_818156


namespace riverFlowRate_l818_818593

-- Define the conditions
def riverDepth : ℝ := 4 -- meters
def riverWidth : ℝ := 65 -- meters
def volumePerMinute : ℝ := 26000 -- cubic meters per minute

-- Define the cross-sectional area of the river
def crossSectionalArea : ℝ := riverDepth * riverWidth

-- Define the flow rate in meters per minute
def flowRateMPerMin : ℝ := volumePerMinute / crossSectionalArea

-- Define the flow rate in km per hour
def flowRateKmph : ℝ := flowRateMPerMin * (1 / 1000) * 60

-- The theorem we need to prove
theorem riverFlowRate : flowRateKmph = 6 := by
  sorry

end riverFlowRate_l818_818593


namespace trig_identity_problem_l818_818484

theorem trig_identity_problem 
  (x : ℝ)
  (h1 : Real.sec x + Real.tan x = 19 / 6) 
  (h2 : ∃ (p q : ℤ), p.gcd q = 1 ∧ Real.csc x + Real.cot x = p / q) :
  ∃ (p q : ℤ), p.gcd q = 1 ∧ p + q = 47 :=
by
  sorry

end trig_identity_problem_l818_818484


namespace problem_statement_l818_818420

theorem problem_statement (n : ℕ) (x : Fin n → ℝ)
  (h1 : ∀ i, 0 < x i ∧ x i < 1) :
  let P := ∏ i, x i,
      S := ∑ i, x i,
      T := ∑ i, (1 / x i)
  in (T - S) / (1 - P) > 2 :=
by
  sorry

end problem_statement_l818_818420


namespace angle_B_is_pi_over_3_l818_818406

-- Definitions
variables {A B C : ℝ} (hA : A < B) (hB : B < C)
          (hTrig : (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) = Real.sqrt 3)

-- Statement
theorem angle_B_is_pi_over_3 : B = Real.pi / 3 :=
begin
  -- Proof to be provided
  sorry
end

end angle_B_is_pi_over_3_l818_818406


namespace Glorys_favorite_number_l818_818449

variable (M G : ℝ)

theorem Glorys_favorite_number :
  (M = G / 3) →
  (M + G = 600) →
  (G = 450) :=
by
sorry

end Glorys_favorite_number_l818_818449


namespace abs_eq_one_fifth_l818_818506

def abs (x : ℝ) : ℝ := if x >= 0 then x else -x

theorem abs_eq_one_fifth (x : ℝ) : abs x = 1 / 5 ↔ x = 1 / 5 ∨ x = -1 / 5 := by
  sorry

end abs_eq_one_fifth_l818_818506


namespace geometric_sequence_condition_l818_818890

-- Define the condition ac = b^2
def condition (a b c : ℝ) : Prop := a * c = b ^ 2

-- Define what it means for a, b, c to form a geometric sequence
def geometric_sequence (a b c : ℝ) : Prop := 
  (b ≠ 0 → a / b = b / c) ∧ (a = 0 → b = 0 ∧ c = 0)

-- The goal is to prove the necessary but not sufficient condition
theorem geometric_sequence_condition (a b c : ℝ) :
  condition a b c ↔ (geometric_sequence a b c → condition a b c) ∧ (¬ (geometric_sequence a b c) → condition a b c ∧ ¬ (geometric_sequence (2 : ℝ) (0 : ℝ) (0 : ℝ))) :=
by
  sorry

end geometric_sequence_condition_l818_818890


namespace find_f_1998_l818_818054

noncomputable def f : ℝ → ℝ := sorry -- Define f as a noncomputable function

theorem find_f_1998 (x : ℝ) (h1 : ∀ x, f (x +1) = f x - 1) (h2 : f 1 = 3997) : f 1998 = 2000 :=
  sorry

end find_f_1998_l818_818054


namespace max_three_digit_divisible_by_4_sequence_l818_818584

theorem max_three_digit_divisible_by_4_sequence (a : ℕ → ℕ) (n : ℕ) (h1 : ∀ k ≤ n - 2, a (k + 2) = 3 * a (k + 1) - 2 * a k - 2)
(h2 : ∀ k1 k2, k1 < k2 → a k1 < a k2) (ha2022 : ∃ k, a k = 2022) (hn : n ≥ 3) :
  ∃ m : ℕ, ∀ k, 100 ≤ a k ∧ a k ≤ 999 → a k % 4 = 0 → m ≤ 225 := by
  sorry

end max_three_digit_divisible_by_4_sequence_l818_818584


namespace average_coins_collected_l818_818815

theorem average_coins_collected (a d n : ℕ) (h_a : a = 5) (h_d : d = 5) (h_n : n = 7) : 
  let l := a + (n - 1) * d in
  let S_n := n * (a + l) / 2 in
  S_n / n = 20 := by
  sorry

end average_coins_collected_l818_818815


namespace at_least_one_nonnegative_l818_818360

theorem at_least_one_nonnegative (x : ℝ) (a b : ℝ) (h1 : a = x^2 - 1) (h2 : b = 4 * x + 5) : ¬ (a < 0 ∧ b < 0) :=
by
  sorry

end at_least_one_nonnegative_l818_818360


namespace a_plus_c_value_l818_818429

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 + a * x + 3

noncomputable def g (x : ℝ) (c : ℝ) : ℝ :=
  x^2 + c * x + 6

theorem a_plus_c_value (a c : ℝ) :
  (∃ x_f x_g, f (-a / 2) c = 0 ∧ g (-c / 2) a = 0 ∧ f 50 a = -50 ∧ g 50 c = -50 ∧ ∀ x, x^2 + a * x + 3 ≥ 3 ∧ x^2 + c * x + 6 ≥ 6 )
  → a + c = -102.18 :=
begin
  sorry
end

end a_plus_c_value_l818_818429


namespace smallest_possible_value_of_n_l818_818545

theorem smallest_possible_value_of_n (n : ℕ) (h : lcm 60 n / gcd 60 n = 45) : n = 1080 :=
by
  sorry

end smallest_possible_value_of_n_l818_818545


namespace symmetric_difference_AB_l818_818208

open Set

def A : Set (ℝ × ℝ) := {p | ∃ x y : ℝ, p = (x, y) ∧ y = x + 1}
def B : Set (ℝ × ℝ) := {p | p = (4, 5)}
def SetDifference (S T : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := {p | p ∈ S ∧ p ∉ T}
def SymmetricDifference (S T : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := (SetDifference S T) ∪ (SetDifference T S)

theorem symmetric_difference_AB :
  SymmetricDifference A B = {p | ∃ x y : ℝ, p = (x, y) ∧ y = x + 1 ∧ x ≠ 4} :=
begin
  sorry
end

end symmetric_difference_AB_l818_818208


namespace general_equation_line_curve_intersections_l818_818806

noncomputable def parametric_line_equations (t : ℝ) : ℝ × ℝ :=
  (1 - (real.sqrt 2) / 2 * t, 2 + (real.sqrt 2) / 2 * t)

def polar_coordinate_curve (θ : ℝ) : ℝ := 4 * real.sin θ

def point_M := (1 : ℝ, 2 : ℝ)

theorem general_equation_line_curve_intersections :
  (∀ t : ℝ, ∃ (x y : ℝ), parametric_line_equations t = (x, y) ∧ x + y = 3) ∧
  (∀ ρ θ : ℝ, polar_coordinate_curve θ = ρ → ρ^2 = (4 * ρ * real.sin θ) ∧ x^2 + y^2 - 4 * y = 0) ∧
  (∃ A B : ℝ × ℝ, (A.1 + A.2 = 3 ∧ B.1 + B.2 = 3) →
  ((M.1 - A.1)^2 + (M.2 - A.2)^2) * ((M.1 - B.1)^2 + (M.2 - B.2)^2) = 3^2)
:= by
  sorry

end general_equation_line_curve_intersections_l818_818806


namespace triangle_base_l818_818886

theorem triangle_base (A h b : ℝ) (hA : A = 15) (hh : h = 6) (hbase : A = 0.5 * b * h) : b = 5 := by
  sorry

end triangle_base_l818_818886


namespace birds_on_fence_l818_818990

theorem birds_on_fence (initial_birds : Nat) (additional_birds : Nat) (storks : Nat) :
  (initial_birds = 6) → (additional_birds = 4) → (storks = 8) → 
  initial_birds + additional_birds + storks = 18 :=
by
  intros h_init h_add h_storks
  rw [h_init, h_add, h_storks]
  norm_num

end birds_on_fence_l818_818990


namespace third_and_fourth_quadrant_points_count_l818_818755

-- Define the sets M and N
def M : Set Int := {1, -2, 3}
def N : Set Int := {-4, 5, 6, -7}

-- Define the predicate for points in the third and fourth quadrants
def inThirdOrFourthQuadrant (x y : Int) : Prop :=
  y < 0

-- Prove the number of such points is 10
theorem third_and_fourth_quadrant_points_count :
  (Finset.card (Finset.filter (λ p : Int × Int, inThirdOrFourthQuadrant p.fst p.snd)
    (Finset.product (M.toFinset) (N.toFinset)))) +
  (Finset.card (Finset.filter (λ p : Int × Int, inThirdOrFourthQuadrant p.snd p.fst)
    (Finset.product (N.toFinset) (M.toFinset)))) = 10 := 
sorry

end third_and_fourth_quadrant_points_count_l818_818755


namespace find_x_coordinate_l818_818591

noncomputable def point_on_plane (x y : ℝ) :=
  (|x + y - 1| / Real.sqrt 2 = |x| ∧
   |x| = |y - 3 * x| / Real.sqrt 10)

theorem find_x_coordinate (x y : ℝ) (h : point_on_plane x y) : 
  x = 1 / (4 + Real.sqrt 10 - Real.sqrt 2) :=
sorry

end find_x_coordinate_l818_818591


namespace find_inverse_value_l818_818722

noncomputable def f (x : ℝ) : ℝ := sorry -- f(x) function definition goes here

theorem find_inverse_value :
  (∀ x : ℝ, f (x - 1) = f (x + 3)) →
  (∀ x : ℝ, f (-x) = f x) →
  (∀ x : ℝ, 4 ≤ x ∧ x ≤ 6 → f x = 2^x + 1) →
  f⁻¹ 19 = 3 - 2 * (Real.log 3 / Real.log 2) :=
by
  intros h1 h2 h3
  -- Proof goes here
  sorry

end find_inverse_value_l818_818722


namespace smallest_integer_with_six_distinct_factors_l818_818129

noncomputable def least_pos_integer_with_six_factors : ℕ :=
  12

theorem smallest_integer_with_six_distinct_factors 
  (n : ℕ)
  (p q : ℕ)
  (a b : ℕ)
  (hp : prime p)
  (hq : prime q)
  (h_diff : p ≠ q)
  (h_n : n = p ^ a * q ^ b)
  (h_factors : (a + 1) * (b + 1) = 6) :
  n = least_pos_integer_with_six_factors :=
by
  sorry

end smallest_integer_with_six_distinct_factors_l818_818129


namespace avg_growth_rate_l818_818158

theorem avg_growth_rate (p1 p2 p3 : ℝ) (h_sum : p1 + p2 + p3 = 1) :
  ∀ p, (p = 2/7 ∨ p = 2/5 ∨ p = 1/3 ∨ p = 1/2 ∨ p = 2/3) → 
        ((1 + p)^3 ≤ (1 + p1) * (1 + p2) * (1 + p3)))
        → p = 1/3 :=
by {
  intros p h_choices h_ineq,
  sorry
}

end avg_growth_rate_l818_818158


namespace division_of_company_l818_818841

variables {G : Type} [graph G]

-- Define the concept of k-unbreakability
def k_unbreakable (k : ℕ) (G : graph G) : Prop :=
  ∀ (partition : fin k → set G), ∃ i, ∃ u v ∈ partition i, u -- v

-- Define the concept of no K_4 subgraph
def no_K_4 (G : graph G) : Prop :=
  ∀ (A : set G), (∀ (u v ∈ A), u -- v) → A.card ≤ 3

-- Assume our specific conditions: G is 3-unbreakable and contains no K_4 subgraph
variables (h1 : k_unbreakable 3 G) (h2 : no_K_4 G)

-- The required theorem statement
theorem division_of_company (G : graph G) (h1 : k_unbreakable 3 G) (h2 : no_K_4 G) :
  ∃ (C D : set G), k_unbreakable 2 (subgraph G C) ∧ k_unbreakable 1 (subgraph G D) :=
sorry

end division_of_company_l818_818841


namespace thirtieth_valid_sequence_term_is_292_l818_818516

def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

def contains_digit_2 (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∣ n ∧ d = 2

def valid_sequence_term (n : ℕ) : Prop :=
  is_multiple_of_4 n ∧ contains_digit_2 n

theorem thirtieth_valid_sequence_term_is_292 :
  (list.filter valid_sequence_term (list.range 10000)).get? 29 = some 292 :=
by
  sorry

end thirtieth_valid_sequence_term_is_292_l818_818516


namespace ratio_of_segments_l818_818724

open Real

variables {A B C M O : Point}

-- Definitions for circumcenter, triangle, and intersection
def is_circumcenter (O : Point) (A B C : Point) : Prop :=
  dist O A = dist O B ∧ dist O B = dist O C

def is_intersection (M O A B C : Point) : Prop :=
  ∃ t : ℝ, O = A + t • (M - A)

-- Angles at points in the triangle
variables {angleA angleB angleC : Real}

-- Main theorem statement
theorem ratio_of_segments 
  (hO : is_circumcenter O A B C)
  (hM : is_intersection M O A B C)
  (angleBOC : ∠ B O C = 2 * angleA)
  (angleAOB : ∠ A O B = 2 * angleC)
  (angleAOC : ∠ A O C = 2 * angleB)
  : BM / MC = sin (2 * angleC) / sin (2 * angleB) :=
sorry

end ratio_of_segments_l818_818724


namespace least_positive_integer_with_six_factors_l818_818118

-- Define what it means for a number to have exactly six distinct positive factors
def hasExactlySixFactors (n : ℕ) : Prop :=
  (n.factorization.support.card = 2 ∧ (n.factorization.values' = [2, 1])) ∨
  (n.factorization.support.card = 1 ∧ (n.factorization.values' = [5]))

-- The main theorem statement
theorem least_positive_integer_with_six_factors : ∃ n : ℕ, hasExactlySixFactors n ∧ ∀ m : ℕ, (hasExactlySixFactors m → n ≤ m) :=
  exists.intro 12 (and.intro
    (show hasExactlySixFactors 12, by sorry)
    (show ∀ m : ℕ, hasExactlySixFactors m → 12 ≤ m, by sorry))

end least_positive_integer_with_six_factors_l818_818118


namespace wings_count_total_l818_818862

def number_of_wings (num_planes : Nat) (wings_per_plane : Nat) : Nat :=
  num_planes * wings_per_plane

theorem wings_count_total :
  number_of_wings 45 2 = 90 :=
  by
    sorry

end wings_count_total_l818_818862


namespace delaney_missed_bus_time_l818_818639

def busDepartureTime : Nat := 480 -- 8:00 a.m. = 8 * 60 minutes
def travelTime : Nat := 30 -- 30 minutes
def departureFromHomeTime : Nat := 470 -- 7:50 a.m. = 7 * 60 + 50 minutes

theorem delaney_missed_bus_time :
  (departureFromHomeTime + travelTime - busDepartureTime) = 20 :=
by
  -- proof would go here
  sorry

end delaney_missed_bus_time_l818_818639


namespace number_of_trees_l818_818249

theorem number_of_trees {N : ℕ} 
  (Anya : N = 15) 
  (Borya : N % 11 = 0) 
  (Vera : N < 25) 
  (Gena : N % 22 = 0) 
  (truth_conditions : (Anya ∨ Borya ∨ Vera ∨ Gena) ∧ ∃! p, p) : 
  N = 11 :=
sorry

end number_of_trees_l818_818249


namespace problem_statement_l818_818883

theorem problem_statement :
  let q := (63 / 10928 : ℚ) in
  let reduced_form_q := (q.num.gcd q.denom = 1 ∧ q = q.num / q.denom) in
  (q.num + q.denom = 10991) :=
by
  let q := 63 / 10928 : ℚ
  let reduced_form_q := q.num.gcd q.denom = 1 ∧ q = q.num / q.denom
  have : q.num + q.denom = 10991 := sorry
  exact this

end problem_statement_l818_818883


namespace number_of_trees_is_11_l818_818261

variable {N : ℕ}

-- Conditions stated by each child
def anya_statement : Prop := N = 15
def borya_statement : Prop := N % 11 = 0
def vera_statement : Prop := N < 25
def gena_statement : Prop := N % 22 = 0

-- One boy and one girl told the truth, while the other two lied
def truth_condition : Prop :=
  (borya_statement ∧ vera_statement ∧ ¬anya_statement ∧ ¬gena_statement) ∨
  (borya_statement ∧ ¬vera_statement ∧ ¬anya_statement ∧ gena_statement) ∨
  (¬borya_statement ∧ vera_statement ∧ ¬anya_statement ∧ gena_statement) ∨
  (¬borya_statement ∧ vera_statement ∧ anya_statement ∧ ¬gena_statement)

-- Proving that the number of fir trees is 11
theorem number_of_trees_is_11 (h : truth_condition) : N = 11 := 
by
  sorry

end number_of_trees_is_11_l818_818261


namespace number_of_trees_is_eleven_l818_818253

variables (N : ℕ)

-- Conditions
def Anya (N : ℕ) := N = 15
def Borya (N : ℕ) := 11 ∣ N
def Vera (N : ℕ) := N < 25
def Gena (N : ℕ) := 22 ∣ N

axiom OneBoyOneGirlTruth :
  (∃ (b : Prop) (g : Prop),
    (b ∨ ¬ b) ∧ (g ∨ ¬ g) ∧
    ((b = (Borya N ∨ Gena N)) ∧ (g = (Anya N ∨ Vera N)) ∧
     (b ↔ ¬g) ∧
     ((Anya N ∨ ¬Vera N) ∨ (¬Anya N ∨ Vera N)) ∧
     (Anya N = (N = 15)) ∧
     (Borya N = (11 ∣ N)) ∧
     (Vera N = (N < 25)) ∧
     (Gena N = (22 ∣ N))))

theorem number_of_trees_is_eleven: N = 11 :=
sorry

end number_of_trees_is_eleven_l818_818253


namespace fx_lt_one_fx_ge_gx_l818_818748

variable {R : Type*} [LinearOrder R] [OrderedCommSemiring R]

noncomputable def f (a x : R) := a^(3*x + 1)
noncomputable def g (a x : R) := (1/a)^(5*x - 2)

theorem fx_lt_one (a x : R) (ha : 0 < a ∧ a < 1) : f a x < 1 ↔ x > -1/3 := 
by sorry

theorem fx_ge_gx (a x : R) (ha : 0 < a ∧ a ≠ 1) : 
(f a x ≥ g a x) ↔ ((0 < a ∧ a < 1 ∧ x ≤ 1/8) ∨ (a > 1 ∧ x ≥ 1/8)) :=
by sorry

end fx_lt_one_fx_ge_gx_l818_818748


namespace no_integers_make_complex_real_l818_818244

open Complex

theorem no_integers_make_complex_real :
  ∃ s : Set ℤ, s = {n : ℤ | (n + 2 * Complex.i)^5 ∈ ℝ} ∧ s = ∅ :=
by
  sorry

end no_integers_make_complex_real_l818_818244


namespace suresh_wife_meeting_time_l818_818556

noncomputable def suresh_speed_kmh : ℝ := 4.5
noncomputable def wife_speed_kmh : ℝ := 3.75
noncomputable def circumference : ℝ := 726
noncomputable def suresh_speed_mpm : ℝ := suresh_speed_kmh * 1000 / 60
noncomputable def wife_speed_mpm : ℝ := wife_speed_kmh * 1000 / 60
noncomputable def relative_speed_mpm : ℝ := suresh_speed_mpm + wife_speed_mpm

theorem suresh_wife_meeting_time : circumference / relative_speed_mpm ≈ 5.28 := 
by
  sorry

end suresh_wife_meeting_time_l818_818556


namespace acute_triangle_angle_C_l818_818790

-- Definitions of geometric properties
variables (A B C O : Type)
variables [AcuteAngleTriangle A B C]
variables (D E F : Type)

-- Definitions that represent geometric properties
def altitude (A B C D : Type) : Prop := sorry
def median (B A C E : Type) : Prop := sorry
def angle_bisector (C A B F : Type) : Prop := sorry
def intersection_at_O (D E F O : Type) : Prop := sorry

-- Given conditions
variable (h_altitude : altitude A B C D)
variable (h_median : median B A C E)
variable (h_angle_bisector : angle_bisector C A B F)
variable (h_intersection : intersection_at_O D E F O)
variable (h_OE_2_OC : OE = 2 * OC)

-- The theorem statement
theorem acute_triangle_angle_C (A B C D E F O : Type) [acute : AcuteAngleTriangle A B C] :
  altitude A B C D → median B A C E → angle_bisector C A B F → intersection_at_O D E F O → OE = 2 * OC → ∠ C = arccos (1 / 7) :=
by
  intros
  -- Skip the proof for now
  sorry

end acute_triangle_angle_C_l818_818790


namespace trapezoid_area_increase_l818_818497

-- Define the conditions
def height : ℕ := 6
def increase_in_base : ℕ := 4

-- Define the problem to prove
theorem trapezoid_area_increase :
  let area_increase := increase_in_base * height
  in area_increase = 24 :=
by
  sorry

end trapezoid_area_increase_l818_818497


namespace number_of_fir_trees_l818_818287

def anya_statement (N : ℕ) : Prop := N = 15
def borya_statement (N : ℕ) : Prop := 11 ∣ N
def vera_statement (N : ℕ) : Prop := N < 25
def gena_statement (N : ℕ) : Prop := 22 ∣ N

def one_boy_one_girl_truth (A B G V : Prop) : Prop :=
  (A ∨ V) ∧ ¬(A ∧ V) ∧ (B ∨ G) ∧ ¬(B ∧ G)

theorem number_of_fir_trees (N : ℕ) :
  anya_statement N ∨ borya_statement N ∨ vera_statement N ∨ gena_statement N ∧
  one_boy_one_girl_truth (anya_statement N) (borya_statement N) (gena_statement N) (vera_statement N) :=
  N = 11 :=
sorry

end number_of_fir_trees_l818_818287


namespace xiao_ming_corrected_calculation_l818_818548

variables (A ab ac bc : ℝ)

theorem xiao_ming_corrected_calculation (h1 : A - 2 * (ab + 2 * bc - 4 * ac) = 3 * ab - 2 * ac + 5 * bc):
  A = - ab + 14 * ac - 3 * bc :=
by
  have h := calc
    A - 2 * (ab + 2 * bc - 4 * ac) + 4 * (ab + 2 * bc - 4 * ac)
    _ = (3 * ab - 2 * ac + 5 * bc) : by rw[h1]
    _ = - ab + 14 * ac - 3 * bc : by ring
  exact h

end xiao_ming_corrected_calculation_l818_818548


namespace sum_of_tangents_l818_818624

noncomputable def g (x : ℝ) : ℝ :=
  max (max (-7 * x - 25) (2 * x + 5)) (5 * x - 7)

theorem sum_of_tangents (a b c : ℝ) (q : ℝ → ℝ) (hq₁ : ∀ x, q x = k * (x - a) ^ 2 + (-7 * x - 25))
  (hq₂ : ∀ x, q x = k * (x - b) ^ 2 + (2 * x + 5))
  (hq₃ : ∀ x, q x = k * (x - c) ^ 2 + (5 * x - 7)) :
  a + b + c = -34 / 3 := 
sorry

end sum_of_tangents_l818_818624


namespace faye_earned_total_money_l818_818657

def bead_necklaces : ℕ := 3
def gem_necklaces : ℕ := 7
def price_per_necklace : ℕ := 7

theorem faye_earned_total_money :
  (bead_necklaces + gem_necklaces) * price_per_necklace = 70 :=
by
  sorry

end faye_earned_total_money_l818_818657


namespace find_integer_k_l818_818561

noncomputable def P : ℤ → ℤ := sorry

theorem find_integer_k :
  P 1 = 2019 ∧ P 2019 = 1 ∧ ∃ k : ℤ, P k = k ∧ k = 1010 :=
by
  sorry

end find_integer_k_l818_818561


namespace number_of_trees_l818_818252

theorem number_of_trees {N : ℕ} 
  (Anya : N = 15) 
  (Borya : N % 11 = 0) 
  (Vera : N < 25) 
  (Gena : N % 22 = 0) 
  (truth_conditions : (Anya ∨ Borya ∨ Vera ∨ Gena) ∧ ∃! p, p) : 
  N = 11 :=
sorry

end number_of_trees_l818_818252


namespace find_x_intervals_l818_818231

theorem find_x_intervals (x : ℝ) :
  (x ∈ set.Iio (-(1 + real.sqrt 13) / 6) ∨
   x ∈ set.Ioo (-(1 - real.sqrt 13) / 6) 0 ∨
   x ∈ set.Ioo 0 ((1 - real.sqrt 13) / 6) ∨
   x ∈ set.Ioc ((1 + real.sqrt 13) / 6) 2) →
  ¬(x * (1 + x - 3 * x^2) = 0) →
  (x^2 + x^3 - 3 * x^4) / (x + x^2 - 3 * x^3) ≤ 2 :=
by
  intro h1 h2
  sorry

end find_x_intervals_l818_818231


namespace triangle_area_comparison_l818_818859

variable {A B C A1 B1 C1 : Type*}
variable [inhabited A1]
variable [inhabited B1]
variable [inhabited C1]
variable (BC CA AB : Set Type*)
variable (τABC : Triangle BC CA AB) 
variable (a1 : Point τABC BC) 
variable (b1 : Point τABC CA) 
variable (c1 : Point τABC AB)

theorem triangle_area_comparison (τ : Triangle τABC a1 b1 c1) :
  area τABC ≤ max (area τABC) (max (area τABC) (area τABC)) := sorry

end triangle_area_comparison_l818_818859


namespace number_of_fir_trees_l818_818285

def anya_statement (N : ℕ) : Prop := N = 15
def borya_statement (N : ℕ) : Prop := 11 ∣ N
def vera_statement (N : ℕ) : Prop := N < 25
def gena_statement (N : ℕ) : Prop := 22 ∣ N

def one_boy_one_girl_truth (A B G V : Prop) : Prop :=
  (A ∨ V) ∧ ¬(A ∧ V) ∧ (B ∨ G) ∧ ¬(B ∧ G)

theorem number_of_fir_trees (N : ℕ) :
  anya_statement N ∨ borya_statement N ∨ vera_statement N ∨ gena_statement N ∧
  one_boy_one_girl_truth (anya_statement N) (borya_statement N) (gena_statement N) (vera_statement N) :=
  N = 11 :=
sorry

end number_of_fir_trees_l818_818285


namespace triangle_with_angle_ratios_l818_818780

theorem triangle_with_angle_ratios {α β γ : ℝ} (h : α + β + γ = 180 ∧ (α / 2 = β / 3) ∧ (α / 2 = γ / 5)) : (α = 90 ∨ β = 90 ∨ γ = 90) :=
by
  sorry

end triangle_with_angle_ratios_l818_818780


namespace expected_value_of_X_l818_818984

noncomputable def A_n_distribution (n : ℕ) : ℕ → ℚ :=
  λ k, if 0 ≤ k ∧ k ≤ n then 1 / (n + 1) else 0

noncomputable def X_distribution : ℕ → ℚ :=
  λ n, ∑ k in Finset.range (n + 1), k * A_n_distribution n k

noncomputable def X : ℚ :=
  ∑' n, X_distribution n / (n + 1)!

theorem expected_value_of_X : E[X] = 1 / 2 :=
sorry

end expected_value_of_X_l818_818984


namespace travel_by_sea_l818_818603

-- Define the variables
variables (total_distance distance_by_land distance_by_sea : ℕ)

-- Provide the given conditions
axiom h1 : total_distance = 601
axiom h2 : distance_by_land = 451

-- Define the statement to prove
theorem travel_by_sea : distance_by_sea = 150 :=
by
  -- Use the conditions provided to prove the statement
  have h : distance_by_sea = total_distance - distance_by_land, from sorry,
  rw [h1, h2] at h,
  exact h

end travel_by_sea_l818_818603


namespace find_t1_t2_l818_818761

-- Define the vectors a and b
def a (t : ℝ) : ℝ × ℝ := (2, t)
def b : ℝ × ℝ := (1, 2)

-- Define the conditions for t1 and t2
def t1_condition (t1 : ℝ) : Prop := (2 / 1) = (t1 / 2)
def t2_condition (t2 : ℝ) : Prop := (2 * 1 + t2 * 2 = 0)

-- The statement to prove
theorem find_t1_t2 (t1 t2 : ℝ) (h1 : t1_condition t1) (h2 : t2_condition t2) : (t1 = 4) ∧ (t2 = -1) :=
by
  sorry

end find_t1_t2_l818_818761


namespace standard_deviation_transformed_data_l818_818735
-- Import necessary libraries

-- Define the variance of the original data
def original_var (x : ℕ → ℝ) (n : ℕ) : ℝ := 16

-- Given that the variance of x_i is 16, prove that the standard deviation of 2x_i + 1 is 8
theorem standard_deviation_transformed_data (x : ℕ → ℝ) (h : original_var x 8 = 16) :
  (∑ i in range 8, ((2 * x i + 1) - (2 * (∑ i in range 8, x i) / 8 + 1))^2 / 8).sqrt = 8 :=
by
  sorry

end standard_deviation_transformed_data_l818_818735


namespace most_significant_action_for_sustainable_utilization_l818_818801

def investigate_population_dynamics (most_significant_action : String) :=
  most_significant_action = "Investigate the population dynamics of the fish species"

theorem most_significant_action_for_sustainable_utilization :
  investigate_population_dynamics ("Investigate the population dynamics of the fish species") :=
by
  -- We'll be skipping the proof using sorry, as instructed
  sorry

end most_significant_action_for_sustainable_utilization_l818_818801


namespace factor_polynomial_l818_818656

theorem factor_polynomial (x : ℝ) : 
  54 * x ^ 5 - 135 * x ^ 9 = 27 * x ^ 5 * (2 - 5 * x ^ 4) :=
by 
  sorry

end factor_polynomial_l818_818656


namespace seven_pointed_star_solution_l818_818475

def valid_arrangement (arrangement : list ℕ) : Prop :=
  (arrangement.length = 14) ∧
  (list.nodup arrangement) ∧
  (∀ i, 1 ≤ i → i ≤ 14 → i ∈ arrangement)

def sums_to_30 (a b c d : ℕ) : Prop :=
  a + b + c + d = 30

def star_sums (arrangement : list ℕ) : Prop :=
  match arrangement with
  | [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14] =>
    sums_to_30 a1 a2 a3 a4 ∧
    sums_to_30 a2 a3 a4 a5 ∧
    sums_to_30 a3 a4 a5 a6 ∧
    sums_to_30 a4 a5 a6 a7 ∧
    sums_to_30 a5 a6 a7 a8 ∧
    sums_to_30 a6 a7 a8 a9 ∧
    sums_to_30 a7 a8 a9 a10 ∧
    sums_to_30 a8 a9 a10 a11 ∧
    sums_to_30 a9 a10 a11 a12 ∧
    sums_to_30 a10 a11 a12 a13 ∧
    sums_to_30 a11 a12 a13 a14 ∧
    sums_to_30 a12 a13 a14 a1 ∧
    sums_to_30 a13 a14 a1 a2 ∧
    sums_to_30 a14 a1 a2 a3
  | _ => false

theorem seven_pointed_star_solution : 
  ∃ arrangement : list ℕ, valid_arrangement arrangement ∧ star_sums arrangement :=
by
  exists [5, 7, 11, 9, 3, 10, 4, 6, 12, 13, 2, 14, 1]
  apply and.intro
  · simp [valid_arrangement]
    split
    · rfl
    · simp
    · intro i h1 h2
      interval_cases i
      repeat {simp}
  · 
sorry

end seven_pointed_star_solution_l818_818475


namespace sum_proper_divisors_of_512_l818_818953

theorem sum_proper_divisors_of_512 : ∑ i in finset.range 9, 2^i = 511 :=
by
  -- We are stating that the sum of 2^i for i ranging from 0 to 8 equals 511.
  sorry

end sum_proper_divisors_of_512_l818_818953


namespace math_problem_ceiling_operations_l818_818221

noncomputable def sqrt_term := Real.ceil (Real.sqrt (25 / 9))
noncomputable def pow3_term := Real.ceil ((25 / 9) ^ 3)
noncomputable def cbrt_term := Real.ceil (Real.cbrt (25 / 9))

theorem math_problem_ceiling_operations :
  sqrt_term + pow3_term + cbrt_term = 26 := sorry

end math_problem_ceiling_operations_l818_818221


namespace harriet_current_age_l818_818390

-- Definitions from the conditions in a)
def mother_age : ℕ := 60
def peter_current_age : ℕ := mother_age / 2
def peter_age_in_four_years : ℕ := peter_current_age + 4
def harriet_age_in_four_years : ℕ := peter_age_in_four_years / 2

-- Proof statement
theorem harriet_current_age : harriet_age_in_four_years - 4 = 13 :=
by
  -- from the given conditions and the solution steps
  let h_current_age := harriet_age_in_four_years - 4
  have : h_current_age = (peter_age_in_four_years / 2) - 4 := by sorry
  have : peter_age_in_four_years = 34 := by sorry
  have : harriet_age_in_four_years = 17 := by sorry
  show 17 - 4 = 13 from sorry

end harriet_current_age_l818_818390


namespace mary_total_zoom_time_l818_818845

noncomputable def timeSpentDownloadingMac : ℝ := 10
noncomputable def timeSpentDownloadingWindows : ℝ := 3 * timeSpentDownloadingMac
noncomputable def audioGlitchesCount : ℝ := 2
noncomputable def audioGlitchDuration : ℝ := 4
noncomputable def totalAudioGlitchTime : ℝ := audioGlitchesCount * audioGlitchDuration
noncomputable def videoGlitchDuration : ℝ := 6
noncomputable def totalGlitchTime : ℝ := totalAudioGlitchTime + videoGlitchDuration
noncomputable def glitchFreeTalkingTime : ℝ := 2 * totalGlitchTime

theorem mary_total_zoom_time : 
  timeSpentDownloadingMac + timeSpentDownloadingWindows + totalGlitchTime + glitchFreeTalkingTime = 82 :=
by sorry

end mary_total_zoom_time_l818_818845


namespace trajectory_of_P_eqn_l818_818355

theorem trajectory_of_P_eqn :
  ∀ {x y : ℝ}, -- For all real numbers x and y
  (-(x + 2)^2 + (x - 1)^2 + y^2 = 3*((x - 1)^2 + y^2)) → -- Condition |PA| = 2|PB|
  (x^2 + y^2 - 4*x = 0) := -- Prove the trajectory equation
by
  intros x y h
  sorry -- Proof to be completed

end trajectory_of_P_eqn_l818_818355


namespace time_to_knit_scarf_l818_818444

theorem time_to_knit_scarf (
    (hats_time : ℕ) := 2,
    (mittens_time : ℕ) := 1,
    (socks_time : ℚ) := 1.5,
    (sweater_time : ℕ) := 6,
    (total_time : ℕ) := 48
  ) : ∃ (S : ℕ), 3 * (hats_time + S + mittens_time + mittens_time + socks_time + socks_time + sweater_time) = total_time :=
begin
  use 3,
  calc 
    3 * (2 + 3 + 1 + 1 + 1.5 + 1.5 + 6) 
        = 3 * 15
    ... = 45
    ... = 48 - 3
    ... = 48 - 39
    ... = 3 * 3,
  ring, 
end.

end time_to_knit_scarf_l818_818444


namespace find_number_exceeds_sixteen_percent_l818_818557

theorem find_number_exceeds_sixteen_percent (x : ℝ) (h : x - 0.16 * x = 63) : x = 75 :=
sorry

end find_number_exceeds_sixteen_percent_l818_818557


namespace math_proof_problem_l818_818800

noncomputable def parametric_line_equations (t : ℝ) : ℝ × ℝ := (1 + (real.sqrt 3 / 2) * t, (1 / 2) * t)

def polar_curve_equation (rho theta : ℝ) : Prop := rho = 4 * real.cos theta

def rectangular_line_equation (x y : ℝ) : Prop := x - real.sqrt 3 * y - 1 = 0

def rectangular_curve_equation (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0

-- Proof statement for the given problems
theorem math_proof_problem :
  (∀ (x y t : ℝ), 
    parametric_line_equations t = (x, y) → rectangular_line_equation x y) ∧ 
  (∀ (rho theta : ℝ), 
    polar_curve_equation rho theta → rectangular_curve_equation rho theta) 
  ∧ (∀ (A B : ℝ × ℝ),
    let P : ℝ × ℝ := (1, 0),
        PA : ℝ := real.sqrt ((A.fst - P.fst)^2 + (A.snd - P.snd)^2),
        PB : ℝ := real.sqrt ((B.fst - P.fst)^2 + (B.snd - P.snd)^2) in
    @classical.some _ (by exact classical.some_spec $
      by sorry) = (real.sqrt 15 / 3) := 
  sorry

end math_proof_problem_l818_818800


namespace max_value_f_compare_magnitude_l818_818743

open Real

def f (x : ℝ) : ℝ := abs (x - 1) - 2 * abs (x + 1)

-- 1. Prove that the maximum value of f(x) is 2.
theorem max_value_f : ∃ x : ℝ, f x = 2 :=
sorry

-- 2. Given the condition, prove 2m + n > 2.
theorem compare_magnitude (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : (1 / m) + (1 / (2 * n)) = 2) : 
  2 * m + n > 2 :=
sorry

end max_value_f_compare_magnitude_l818_818743


namespace arithmetic_sequence_proof_l818_818840

variable (n : ℕ)
variable (a_n S_n : ℕ → ℤ)

noncomputable def a : ℕ → ℤ := 48 - 8 * n
noncomputable def S : ℕ → ℤ := -4 * (n ^ 2) + 44 * n

axiom a_3 : a 3 = 24
axiom S_11 : S 11 = 0

theorem arithmetic_sequence_proof :
  a n = 48 - 8 * n ∧
  S n = -4 * n ^ 2 + 44 * n ∧
  ∃ n, S n = 120 ∧ (n = 5 ∨ n = 6) :=
by
  unfold a S
  sorry

end arithmetic_sequence_proof_l818_818840


namespace delaney_missed_bus_time_l818_818637

def busDepartureTime : Nat := 480 -- 8:00 a.m. = 8 * 60 minutes
def travelTime : Nat := 30 -- 30 minutes
def departureFromHomeTime : Nat := 470 -- 7:50 a.m. = 7 * 60 + 50 minutes

theorem delaney_missed_bus_time :
  (departureFromHomeTime + travelTime - busDepartureTime) = 20 :=
by
  -- proof would go here
  sorry

end delaney_missed_bus_time_l818_818637


namespace three_digit_cubic_units_digit_eq_l818_818188

theorem three_digit_cubic_units_digit_eq :
  ∀ (x y z : ℕ),
  (x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ (y ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ (z ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) →
  (100 * x + 10 * y + z = z^3) →
  (100 * x + 10 * y + z = 125 ∨ 100 * x + 10 * y + z = 216 ∨ 100 * x + 10 * y + z = 729) := by
  sorry

end three_digit_cubic_units_digit_eq_l818_818188


namespace calculate_QR_length_l818_818042

theorem calculate_QR_length (A PQ RS h : ℝ) (hA : A = 200) (hPQ : PQ = 15) (hRS : RS = 20) 
(hh : h = 10) : 
  let QR := 20 - 2.5 * Real.sqrt 5 - 5 * Real.sqrt 3 in
  QR = 20 - 2.5 * Real.sqrt 5 - 5 * Real.sqrt 3 :=
by 
  let QR := 20 - 2.5 * Real.sqrt 5 - 5 * Real.sqrt 3
  sorry

end calculate_QR_length_l818_818042


namespace G_even_l818_818376

variable (a : ℝ) (F : ℝ → ℝ)
variable (h0 : a > 0) (h1 : a ≠ 1) (hF_odd : ∀ x, F (-x) = - F x)

def G (x : ℝ) : ℝ := F x * ((1 / (a ^ x - 1)) + (1 / 2))

theorem G_even : ∀ x : ℝ, G a F x = G a F (-x) :=
by
  sorry

end G_even_l818_818376


namespace max_min_distance_l818_818019

open Real

noncomputable def point_on_ellipse (theta : ℝ) : ℝ × ℝ :=
  (4 * cos theta, 3 * sin theta)

noncomputable def distance_to_line (P : ℝ × ℝ) : ℝ :=
  abs (12 * P.1 - 12 * P.2 - 24) / 5

theorem max_min_distance :
  (∀ (theta : ℝ), 
    let P := point_on_ellipse theta in
    distance_to_line P ≤ (12 / 5 * (2 + sqrt 2))) ∧
  (∀ (theta : ℝ),
    let P := point_on_ellipse theta in
    distance_to_line P ≥ (12 / 5 * (2 - sqrt 2))) :=
by 
exact sorry

end max_min_distance_l818_818019


namespace problem1_problem2_l818_818988

-- Problem (Ⅰ)
theorem problem1 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  (1 + 1 / a) * (1 + 1 / b) ≥ 9 :=
sorry

-- Problem (Ⅱ)
theorem problem2 (a : ℝ) (h1 : ∀ (x : ℝ), x ≥ 1 ↔ |x + 3| - |x - a| ≥ 2) :
  a = 2 :=
sorry

end problem1_problem2_l818_818988


namespace at_least_one_angle_geq_60_l818_818968

theorem at_least_one_angle_geq_60 (α β γ : ℝ) (h_triangle: α + β + γ = 180) 
  (h_α_nonneg: α ≥ 0) (h_β_nonneg: β ≥ 0) (h_γ_nonneg: γ ≥ 0) : 
  (α ≥ 60) ∨ (β ≥ 60) ∨ (γ ≥ 60) :=
by
  -- Assuming the negation of the conclusion: no angle is greater than or equal to 60°
  assume h1 : α < 60
  assume h2 : β < 60
  assume h3 : γ < 60
  -- A contradiction should be derived here
  sorry

end at_least_one_angle_geq_60_l818_818968


namespace geoseq_a2_l818_818710

theorem geoseq_a2 (a : ℕ → ℝ) (q : ℝ) : 
  a 1 = 1 / 4 ∧ (a 3) * (a 5) = 4 * (a 4 - 1) →
  (a 2 = 1 / 2) :=
by
  intro h,
  let a_1 := (1 : ℝ) / 4,
  let a_3 := a_1 * q ^ 2,
  let a_4 := a_1 * q ^ 3,
  let a_5 := a_1 * q ^ 4,
  
  have h1 : a 1 = a_1, from h.left,
  have h2 : a 3 * a 5 = 4 * (a 4 - 1), from h.right,

  -- Express conditions in terms of q and a_1
  rw [h1, h2],
  
  -- Need to prove the main statement
  sorry

end geoseq_a2_l818_818710


namespace part1_part2_part3_l818_818749

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x + a / x + Real.log x

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ :=
  1 - a / x^2 + 1 / x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ :=
  f' x a - x

theorem part1 (a : ℝ) (h : f' 1 a = 0) : a = 2 :=
  sorry

theorem part2 {a : ℝ} (h : ∀ x, 1 < x → x < 2 → f' x a ≥ 0) : a ≤ 2 :=
  sorry

theorem part3 (a : ℝ) :
  ((a > 1 → ∀ x, g x a ≠ 0) ∧ 
  (a = 1 ∨ a ≤ 0 → ∃ x, g x a = 0 ∧ ∀ y, g y a = 0 → y = x) ∧ 
  (0 < a ∧ a < 1 → ∃ x y, x ≠ y ∧ g x a = 0 ∧ g y a = 0)) :=
  sorry

end part1_part2_part3_l818_818749


namespace number_of_correct_conclusions_l818_818212

def converse_proposition (P Q : Prop) : Prop := (Q → P)

def state1 : Prop := 
  let P := (λ x : ℝ, x^2 - 3 * x + 2 = 0) 
  let Q := (λ x : ℝ, x = 1)
  converse_proposition (∀ x, P x → Q x) (∀ x, Q x → ¬ P x)

def state2 : Prop := 
  ∀ a : ℝ, (a ≠ 0 → a^2 + a ≠ 0) ∧ ¬(a^2 + a ≠ 0 → a ≠ 0)

def state3 : Prop := 
  ∀ (p q : Prop), ¬(p ∧ q) → (¬p ∧ ¬q)

def state4 : Prop := 
  (∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0)

theorem number_of_correct_conclusions : 
  (state1 ∧ state2 ∧ ¬state3 ∧ state4) = 3 :=
sorry

end number_of_correct_conclusions_l818_818212


namespace max_f_of_sin_bounded_l818_818503

theorem max_f_of_sin_bounded (x : ℝ) : (∀ y, -1 ≤ Real.sin y ∧ Real.sin y ≤ 1) → ∃ m, (∀ z, (1 + 2 * Real.sin z) ≤ m) ∧ (∀ n, (∀ z, (1 + 2 * Real.sin z) ≤ n) → m ≤ n) :=
by
  sorry

end max_f_of_sin_bounded_l818_818503


namespace injective_func_identity_l818_818230

open Function

theorem injective_func_identity :
  (∀ f : ℕ → ℕ,
    injective f ∧ (∀ n : ℕ, f(f(n)) ≤ (n + f(n)) / 2) →
    (∀ n, f(n) = n)) :=
by
  intro f
  intro h
  cases h with hf hf'
  sorry

end injective_func_identity_l818_818230


namespace lcm_36_105_l818_818687

theorem lcm_36_105 : Int.lcm 36 105 = 1260 := by
  sorry

end lcm_36_105_l818_818687


namespace remaining_dogs_eq_200_l818_818071

def initial_dogs : ℕ := 200
def additional_dogs : ℕ := 100
def first_adoption : ℕ := 40
def second_adoption : ℕ := 60

def total_dogs_after_adoption : ℕ :=
  initial_dogs + additional_dogs - first_adoption - second_adoption

theorem remaining_dogs_eq_200 : total_dogs_after_adoption = 200 :=
by
  -- Omitted the proof as requested
  sorry

end remaining_dogs_eq_200_l818_818071


namespace simplify_expression_l818_818874

theorem simplify_expression :
  (√(1 - 2 * Real.sin (Real.pi / 9) * Real.cos (Real.pi / 9)) /
     (Real.cos (Real.pi / 9) - √(1 - Real.cos (160 * Real.pi / 180) ^ 2))) = 1 :=
by
  sorry

end simplify_expression_l818_818874


namespace train_cross_time_correct_l818_818810

-- Defining the initial conditions
def train_length : ℝ := 120  -- length in meters
def train_speed : ℝ := 72    -- speed in km/hr

-- Conversion factor from km/hr to m/s
def kmhr_to_mps (v : ℝ) : ℝ := v * (1000 / 3600)

-- Calculating the speed in m/s
def speed_mps : ℝ := kmhr_to_mps train_speed

-- Required to prove: time to cross the pole in seconds
def time_to_cross (d : ℝ) (v : ℝ) : ℝ := d / v

-- Main statement
theorem train_cross_time_correct :
  time_to_cross train_length speed_mps = 6 :=
by sorry

end train_cross_time_correct_l818_818810


namespace tangent_parallel_BC_l818_818047

variables (A B C D D1 P A1 M N : Type) [hA : is_cyclic_quadrilateral A B C D] [hAC : is_diagonal_intersection A C B D P]
[hTangent : is_tangent_at_circumcircle P D1 A1 M N]

theorem tangent_parallel_BC :
  is_parallel M N B C :=
begin
  -- Proof steps would go here (omitted)
  sorry
end

end tangent_parallel_BC_l818_818047


namespace fraction_nonneg_if_x_ge_m8_l818_818035

noncomputable def denominator (x : ℝ) : ℝ := x^2 + 4*x + 13
noncomputable def numerator (x : ℝ) : ℝ := x + 8

theorem fraction_nonneg_if_x_ge_m8 (x : ℝ) (hx : x ≥ -8) : numerator x / denominator x ≥ 0 :=
by sorry

end fraction_nonneg_if_x_ge_m8_l818_818035


namespace total_matchsticks_l818_818226

theorem total_matchsticks (boxes : ℕ) (matchboxes_per_box : ℕ) (sticks_per_matchbox : ℕ) 
  (h1 : boxes = 4) (h2 : matchboxes_per_box = 20) (h3 : sticks_per_matchbox = 300) :
  boxes * matchboxes_per_box * sticks_per_matchbox = 24000 :=
by 
  rw [h1, h2, h3];
  norm_num

end total_matchsticks_l818_818226


namespace fir_trees_count_l818_818295

theorem fir_trees_count (N : ℕ) :
  (N = 15 ∨ (N < 25 ∧ 11 ∣ N) ∨ 22 ∣ N) ∧ 
  (1 ≤ (if N = 15 then 1 else 0) + (if 11 ∣ N then 1 else 0) + (if N < 25 then 1 else 0) + (if 22 ∣ N then 1 else 0) ∧
   2 ≠ (if N = 15 then 1 else 0) + (if 11 ∣ N then 1 else 0) + (if N < 25 then 1 else 0) + (if 22 ∣ N then 1 else 0)) → N = 11 :=
begin
  sorry
end

end fir_trees_count_l818_818295


namespace excircle_radius_eq_semiperimeter_implies_right_l818_818461

variables {A B C: Point}
variables {r p : ℝ}

-- Assume definitions and contexts
def is_right_triangle (α β γ : ℝ) : Prop := 
  α^2 + β^2 = γ^2 ∨ β^2 + γ^2 = α^2 ∨ γ^2 + α^2 = β^2

-- Excircle radius equals semiperimeter implies the triangle is right-angled
theorem excircle_radius_eq_semiperimeter_implies_right (h1 : r = p) 
(h2 : r = (A.distance B + B.distance C + C.distance A) / 2) :
  is_right_triangle (A.distance B) (B.distance C) (C.distance A) :=
begin
  sorry
end

end excircle_radius_eq_semiperimeter_implies_right_l818_818461


namespace boy_completes_work_in_nine_days_l818_818553

theorem boy_completes_work_in_nine_days :
  let M := (1 : ℝ) / 6
  let W := (1 : ℝ) / 18
  let B := (1 / 3 : ℝ) - M - W
  B = (1 : ℝ) / 9 := by
    sorry

end boy_completes_work_in_nine_days_l818_818553


namespace number_of_trees_is_11_l818_818266

variable {N : ℕ}

-- Conditions stated by each child
def anya_statement : Prop := N = 15
def borya_statement : Prop := N % 11 = 0
def vera_statement : Prop := N < 25
def gena_statement : Prop := N % 22 = 0

-- One boy and one girl told the truth, while the other two lied
def truth_condition : Prop :=
  (borya_statement ∧ vera_statement ∧ ¬anya_statement ∧ ¬gena_statement) ∨
  (borya_statement ∧ ¬vera_statement ∧ ¬anya_statement ∧ gena_statement) ∨
  (¬borya_statement ∧ vera_statement ∧ ¬anya_statement ∧ gena_statement) ∨
  (¬borya_statement ∧ vera_statement ∧ anya_statement ∧ ¬gena_statement)

-- Proving that the number of fir trees is 11
theorem number_of_trees_is_11 (h : truth_condition) : N = 11 := 
by
  sorry

end number_of_trees_is_11_l818_818266


namespace number_of_trees_is_eleven_l818_818258

variables (N : ℕ)

-- Conditions
def Anya (N : ℕ) := N = 15
def Borya (N : ℕ) := 11 ∣ N
def Vera (N : ℕ) := N < 25
def Gena (N : ℕ) := 22 ∣ N

axiom OneBoyOneGirlTruth :
  (∃ (b : Prop) (g : Prop),
    (b ∨ ¬ b) ∧ (g ∨ ¬ g) ∧
    ((b = (Borya N ∨ Gena N)) ∧ (g = (Anya N ∨ Vera N)) ∧
     (b ↔ ¬g) ∧
     ((Anya N ∨ ¬Vera N) ∨ (¬Anya N ∨ Vera N)) ∧
     (Anya N = (N = 15)) ∧
     (Borya N = (11 ∣ N)) ∧
     (Vera N = (N < 25)) ∧
     (Gena N = (22 ∣ N))))

theorem number_of_trees_is_eleven: N = 11 :=
sorry

end number_of_trees_is_eleven_l818_818258


namespace increase_factor_l818_818942

noncomputable def old_plates : ℕ := 26 * 10^3
noncomputable def new_plates : ℕ := 26^4 * 10^4
theorem increase_factor : (new_plates / old_plates) = 175760 := by
  sorry

end increase_factor_l818_818942


namespace available_codes_count_l818_818452

-- Definitions based on the given conditions
def original_code : ℕ := 146

def is_valid_code (code : ℕ) : Prop :=
  code ∈ {1..9} × {1..9} × {1..9} ∧
  ¬code = 641 ∧
  ¬code = 146 ∧
  (∀ place1 place2, ¬(code[place1] = original_code[place1] ∧ code[place2] = original_code[place2]))

def available_codes : ℕ :=
  card {code | is_valid_code code}

-- The theorem we need to prove
theorem available_codes_count : available_codes = 535 :=
by sorry

end available_codes_count_l818_818452


namespace raft_downstream_time_l818_818491

variables {s v_s v_c : ℝ}

-- Distance covered by the motor ship downstream in 5 hours
def downstream_time (s : ℝ) (v_s v_c : ℝ) : Prop := s / (v_s + v_c) = 5

-- Distance covered by the motor ship upstream in 6 hours
def upstream_time (s : ℝ) (v_s v_c : ℝ) : Prop := s / (v_s - v_c) = 6

-- Time it takes for a raft to float downstream over this distance
theorem raft_downstream_time : 
  ∀ (s v_s v_c : ℝ), 
  downstream_time s v_s v_c ∧ upstream_time s v_s v_c → s / v_c = 60 :=
by
  sorry

end raft_downstream_time_l818_818491


namespace circle_center_radius_1_circle_center_coordinates_radius_1_l818_818163

theorem circle_center_radius_1 (x y : ℝ) : 
  x^2 + y^2 + 2*x - 4*y - 3 = 0 ↔ (x + 1)^2 + (y - 2)^2 = 8 :=
sorry

theorem circle_center_coordinates_radius_1 : 
  (∃ x y : ℝ, x^2 + y^2 + 2*x - 4*y - 3 = 0 ∧ (x, y) = (-1, 2)) ∧ 
  (∃ r : ℝ, r = 2*Real.sqrt 2) :=
sorry

end circle_center_radius_1_circle_center_coordinates_radius_1_l818_818163


namespace lcm_36_105_l818_818686

theorem lcm_36_105 : Int.lcm 36 105 = 1260 := by
  sorry

end lcm_36_105_l818_818686


namespace proof_problem_l818_818201

noncomputable def a : ℚ := 2 / 3
noncomputable def b : ℚ := - 3 / 2
noncomputable def n : ℕ := 2023

theorem proof_problem :
  (a ^ n) * (b ^ n) = -1 :=
by
  sorry

end proof_problem_l818_818201


namespace remaining_dogs_eq_200_l818_818072

def initial_dogs : ℕ := 200
def additional_dogs : ℕ := 100
def first_adoption : ℕ := 40
def second_adoption : ℕ := 60

def total_dogs_after_adoption : ℕ :=
  initial_dogs + additional_dogs - first_adoption - second_adoption

theorem remaining_dogs_eq_200 : total_dogs_after_adoption = 200 :=
by
  -- Omitted the proof as requested
  sorry

end remaining_dogs_eq_200_l818_818072


namespace slices_per_pizza_l818_818091

theorem slices_per_pizza (total_slices number_of_pizzas slices_per_pizza : ℕ) 
  (h_total_slices : total_slices = 168) 
  (h_number_of_pizzas : number_of_pizzas = 21) 
  (h_division : total_slices / number_of_pizzas = slices_per_pizza) : 
  slices_per_pizza = 8 :=
sorry

end slices_per_pizza_l818_818091


namespace number_of_fir_trees_l818_818292

def anya_statement (N : ℕ) : Prop := N = 15
def borya_statement (N : ℕ) : Prop := 11 ∣ N
def vera_statement (N : ℕ) : Prop := N < 25
def gena_statement (N : ℕ) : Prop := 22 ∣ N

def one_boy_one_girl_truth (A B G V : Prop) : Prop :=
  (A ∨ V) ∧ ¬(A ∧ V) ∧ (B ∨ G) ∧ ¬(B ∧ G)

theorem number_of_fir_trees (N : ℕ) :
  anya_statement N ∨ borya_statement N ∨ vera_statement N ∨ gena_statement N ∧
  one_boy_one_girl_truth (anya_statement N) (borya_statement N) (gena_statement N) (vera_statement N) :=
  N = 11 :=
sorry

end number_of_fir_trees_l818_818292


namespace smallest_number_divisible_by_6_l818_818010

noncomputable def sum_digits : Nat := 1 + 2 + 3 + 6 + 9

def is_divisible_by_6 (n : Nat) : Prop :=
  n % 6 = 0

def permutations_of_digits (digits : List Nat) : List (List Nat) :=
  digits.permutations

def to_number (digits : List Nat) : Nat :=
  digits.foldl (λ acc d, acc * 10 + d) 0

def find_smallest_divisible_by_6 : Nat :=
  (permutations_of_digits [1, 2, 3, 6, 9]).map to_number
  |>.filter is_divisible_by_6
  |>.minimum?.getOrElse 0

theorem smallest_number_divisible_by_6 : find_smallest_divisible_by_6 = 12369 := sorry

end smallest_number_divisible_by_6_l818_818010


namespace veronica_cans_of_food_is_multiple_of_4_l818_818941

-- Definitions of the given conditions
def number_of_water_bottles : ℕ := 20
def number_of_kits : ℕ := 4

-- Proof statement
theorem veronica_cans_of_food_is_multiple_of_4 (F : ℕ) :
  F % number_of_kits = 0 :=
sorry

end veronica_cans_of_food_is_multiple_of_4_l818_818941


namespace harkamal_payment_l818_818979

variable (grapes_qty : ℕ) (grapes_rate : ℕ)
variable (mangoes_qty : ℕ) (mangoes_rate : ℕ)

theorem harkamal_payment :
  grapes_qty = 8 →
  grapes_rate = 70 →
  mangoes_qty = 9 →
  mangoes_rate = 65 →
  (grapes_qty * grapes_rate + mangoes_qty * mangoes_rate = 1145) :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end harkamal_payment_l818_818979


namespace apollonian_circle_locus_l818_818758

variables {α : Type*} [metric_space α] [normed_group α] {A B M : α} {k : ℝ}

-- Conditions: A and B are given points, k is a given ratio that is not 1
axiom A : α
axiom B : α
axiom k : ℝ
axiom k_ne_one : k ≠ 1

-- Definition for the distance between two points
def dist (x y : α) : ℝ := sorry

-- Proof problem: given A, B, and k, prove that the set of points M such that
-- the distance |AM| / |MB| = k forms an Apollonian circle
theorem apollonian_circle_locus :
  (set_of (λ M : α, dist M A / dist M B = k)) = sorry :=
sorry

end apollonian_circle_locus_l818_818758


namespace line_equation_l818_818333

noncomputable def line_l (θ : ℝ) : (ℝ → ℝ → Prop) :=
  if θ = π / 2 then 
    λ (x : ℝ) (y : ℝ), x = 0
  else 
    λ (x : ℝ) (y : ℝ), y = (Real.sqrt 3 / 3) * x 

theorem line_equation (θ : ℝ) (h : θ = π / 6 ∨ θ = π / 2) : (line_l θ) = (λ x y, x = 0) ∨ (line_l θ) = (λ x y, y = (Real.sqrt 3 / 3) * x) :=
by
  sorry

end line_equation_l818_818333


namespace not_divides_96_l818_818833

theorem not_divides_96 (m : ℕ) (h1 : m > 0) 
  (h2 : (1 / 3 + 1 / 4 + 1 / 8 + 1 / m : ℚ).denom = 1) : ¬ (m > 96) :=
sorry

end not_divides_96_l818_818833


namespace teachers_not_adjacent_arrangements_l818_818785

theorem teachers_not_adjacent_arrangements :
  let teachers := 2
  let students := 3
  let total_slots := students + 1
  ∀ (Perm_students : Fin (students + 1) ≃ Fin (students + 1)) (Perm_slots : Fin teachers ≃ Fin total_slots), 
    Perm_students.toFun * Perm_slots.toFun = 72 :=
by
  let teachers := 2
  let students := 3
  let total_slots := 4 -- students + 1
  sorry

end teachers_not_adjacent_arrangements_l818_818785


namespace total_money_l818_818844

-- Conditions
def mark_amount : ℚ := 5 / 6
def carolyn_amount : ℚ := 2 / 5

-- Combine both amounts and state the theorem to be proved
theorem total_money : mark_amount + carolyn_amount = 1.233 := by
  -- placeholder for the actual proof
  sorry

end total_money_l818_818844


namespace negation_of_universal_prop_l818_818901

open Classical

variable (ℝ : Type) [Nonempty ℝ] [OrderedRing ℝ]

theorem negation_of_universal_prop :
  (¬ (∀ x : ℝ, x^2 ≠ x)) ↔ (∃ x₀ : ℝ, x₀^2 = x₀) := by
  sorry

end negation_of_universal_prop_l818_818901


namespace dentist_age_l818_818861

theorem dentist_age (x : ℝ) (h : (x - 8) / 6 = (x + 8) / 10) : x = 32 :=
  by
  sorry

end dentist_age_l818_818861


namespace magnitude_proj_v_on_w_l818_818426

variables {V : Type*} [inner_product_space ℝ V] (v w : V)
hypothesis (dot_product : ⟪v, w⟫ = -3)
hypothesis (norm_w : ∥w∥ = 5)

theorem magnitude_proj_v_on_w : ∥(orthogonal_projection (ℝ ∙ w) v)∥ = 3 / 5 := by 
  sorry

end magnitude_proj_v_on_w_l818_818426


namespace min_value_geometric_sequence_l818_818433

theorem min_value_geometric_sequence (s : ℝ) : ∃ min_value : ℝ, min_value = -18 / 7 ∧ ∀ x : ℝ, 9 * x + 21 * x^2 ≥ min_value :=
by
  -- The proof steps are omitted, but here's the statement.
  have min_value := -18 / 7
  use min_value
  split
  case h1 { sorry }
  case h2 { sorry }

end min_value_geometric_sequence_l818_818433


namespace workers_not_worked_days_l818_818179

theorem workers_not_worked_days (W N : ℤ) (h1 : W + N = 30) (h2 : 100 * W - 25 * N = 0) : N = 24 := 
by
  sorry

end workers_not_worked_days_l818_818179


namespace cube_surface_area_proof_l818_818577

noncomputable def cube_surface_area {r : ℝ} (s : ℝ) (h₁ : s = 4 * π) : ℝ :=
  let a := (sqrt (2 * r^2 / 3)) in 6 * a^2

theorem cube_surface_area_proof (h₁ : 4 * π = 4 * π) : cube_surface_area 1 4 * π = 8 :=
sorry

end cube_surface_area_proof_l818_818577


namespace least_positive_integer_with_six_distinct_factors_l818_818095

theorem least_positive_integer_with_six_distinct_factors : ∃ n : ℕ, (∀ k : ℕ, (number_of_factors k = 6) → (n ≤ k)) ∧ (number_of_factors n = 6) ∧ (n = 12) :=
by
  sorry

end least_positive_integer_with_six_distinct_factors_l818_818095


namespace sulfur_production_l818_818400

variable (n_electrons : ℝ)
variable (reaction : ℝ → ℝ → (ℝ × ℝ × ℝ))

-- Condition: The chemical reaction of the process.
def chemical_reaction : ∀ (a : ℝ) (b : ℝ), (ℝ × ℝ × ℝ) :=
  λ a b, (2 * a, 0, 3 * b)

-- Given: 4 * 6.02 * 10^23 electrons are transferred
def electrons_transferred : ℝ := 4 * 6.02 * 10^23

-- We are asked to prove the amount of sulfur produced
theorem sulfur_production :
  chemical_reaction 1 1 = (2, 0, 3) →
  4 * electrons_transferred = 3 → True :=
by
  intros h1 h2
  sorry

end sulfur_production_l818_818400


namespace lcm_of_36_and_105_l818_818674

theorem lcm_of_36_and_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end lcm_of_36_and_105_l818_818674


namespace maxwell_walking_speed_l818_818445

theorem maxwell_walking_speed :
  ∃ v : ℝ, (8 * v + 6 * 7 = 74) ∧ v = 4 :=
by
  exists 4
  constructor
  { norm_num }
  rfl

end maxwell_walking_speed_l818_818445


namespace speed_difference_l818_818578

theorem speed_difference (h_cyclist : 88 / 8 = 11) (h_car : 48 / 8 = 6) :
  (11 - 6 = 5) :=
by
  sorry

end speed_difference_l818_818578


namespace area_of_inscribed_circle_triangle_l818_818171

noncomputable def area_of_triangle (a b c : ℝ) (r : ℝ) : ℝ :=
  1 / 2 * (a + b + c) * r

theorem area_of_inscribed_circle_triangle {A B C X Y Z : Type*} 
  (r : ℝ) 
  (triangle : Triangle A B C) 
  (omega : Circle r) 
  (tangent_to_AB : omega.tangent_to AB X) 
  (diametrically_opposite : omega.diametrically_opposite X Y) 
  (CY_intersects_AB_at_Z : Line C Y × Intersect AB Z) 
  (CA_plus_AZ_eq_one : length (segment C A) + length (segment A Z) = 1) :
  area_of_triangle (length (segment B C)) (length (segment C A)) (length (segment A B)) r = r :=
by
  sorry

end area_of_inscribed_circle_triangle_l818_818171


namespace g_600_l818_818831

def g : ℕ → ℕ := sorry

axiom g_mul (x y : ℕ) (hx : x > 0) (hy : y > 0) : g (x * y) = g x + g y
axiom g_12 : g 12 = 18
axiom g_48 : g 48 = 26

theorem g_600 : g 600 = 36 :=
by 
  sorry

end g_600_l818_818831


namespace number_of_fir_trees_is_11_l818_818271

theorem number_of_fir_trees_is_11 
  (N : ℕ)
  (Anya : N = 15)
  (Borya : N % 11 = 0)
  (Vera : N < 25)
  (Gena : N % 22 = 0)
  (OneBoyOneGirlTrue : (Anya ∨ Borya) ∧ (Vera ∨ Gena) ∧ (¬Anya ∨ ¬Borya) ∧ (¬Vera ∨ ¬Gena)) :
  N = 11 := 
sorry

end number_of_fir_trees_is_11_l818_818271


namespace larger_number_is_588_l818_818897

theorem larger_number_is_588
  (A B hcf : ℕ)
  (lcm_factors : ℕ × ℕ)
  (hcf_condition : hcf = 42)
  (lcm_factors_condition : lcm_factors = (12, 14))
  (hcf_prop : Nat.gcd A B = hcf)
  (lcm_prop : Nat.lcm A B = hcf * lcm_factors.1 * lcm_factors.2) :
  max (A) (B) = 588 :=
by
  sorry

end larger_number_is_588_l818_818897


namespace l_squared_l818_818884

def f (x : ℝ) : ℝ := -x + 3
def g (x : ℝ) : ℝ := 0.5 * x - 1
def h (x : ℝ) : ℝ := 2

def k (x : ℝ) : ℝ := 
  if x ≤ 1 then f x
  else if x < (8 / 3) then h x
  else g x

noncomputable def segment1_length : ℝ := Real.sqrt ((-3 - 1)^2 + (f (-3) - f 1)^2)
noncomputable def segment2_length : ℝ := (8/3) - 1
noncomputable def segment3_length : ℝ := Real.sqrt ((8/3 - 4)^2 + (g (8/3) - g 4)^2)

noncomputable def l : ℝ := segment1_length + segment2_length + segment3_length

theorem l_squared (l : ℝ) : l^2 = (Real.sqrt 52 + 5 / 3 + Real.sqrt (25 / 12))^2 :=
by 
  sorry -- proof is omitted

end l_squared_l818_818884


namespace music_library_disk_space_per_hour_l818_818998

theorem music_library_disk_space_per_hour :
  (15 * 24) * ((24000 / (15 * 24)).round) = 24000 :=
by
  -- 15 days of music
  have days := 15
  -- 24 hours in each day
  have hours_per_day := 24
  -- Total hours of music
  have total_hours := 15 * 24
  -- Total disk space of music
  have total_disk_space := 24000
  -- Disk space per hour
  have megabytes_per_hour := total_disk_space / total_hours
  -- Disk space per hour rounded to nearest whole number
  let rounded_megabytes_per_hour := Real.round megabytes_per_hour
  exact Eq.trans (Eq.symm (mul_div_round_of_not_divisible total_disk_space total_hours)) rfl

end music_library_disk_space_per_hour_l818_818998


namespace fir_trees_alley_l818_818277

-- Define the statements made by each child
def statementAnya (N : ℕ) : Prop := N = 15
def statementBorya (N : ℕ) : Prop := N % 11 = 0
def statementVera (N : ℕ) : Prop := N < 25
def statementGena (N : ℕ) : Prop := N % 22 = 0

-- Define the condition about the truth and lies
def oneBoyOneGirlTruth (anya_vera_truth: Prop) (borya_gena_truth: Prop) : Prop :=
  anya_vera_truth ∧ borya_gena_truth ∧
  ((statementAnya N ∧ statementVera N) ∨ (statementVera N ∧ statementBorya N)) ∧
  ¬( (statementAnya N ∧ statementGena N) ∨ (statementVera N ∧ statementGena N) ∨
     (statementAnya N ∧ statementBorya N) ∨ (statementBorya N ∧ statementGena N) )

-- Prove that the number of fir trees is 11
theorem fir_trees_alley: ∃ (N : ℕ), statementBorya N ∧ statementVera N ∧ ¬ statementAnya N ∧ ¬ statementGena N ∧ oneBoyOneGirlTruth (¬ statementAnya N ∧ statementVera N) (statementBorya N ∧ ¬ statementGena N) ∧ N = 11 :=
by
  sorry

end fir_trees_alley_l818_818277


namespace inequality_proof_l818_818195

variables (A B C : Type)
variables (α β γ : ℝ)
variables (r R : ℝ)
variables (M X : A → B → C)

noncomputable def is_acute_triangle (α β γ : ℝ) : Prop :=
  α < π / 2 ∧ β < π / 2 ∧ γ < π / 2

noncomputable def circumradius_trio_select (A B C : Type) : ℝ := sorry -- placeholder for circumradius definition
noncomputable def inradius_trio_select (A B C : Type) : ℝ := sorry -- placeholder for inradius definition
noncomputable def midpoint_M (B: Type) (C: Type) : A := sorry -- placeholder for midpoint definition
noncomputable def intersection_tangents_X (B: Type) (C: Type) : A := sorry -- placeholder for intersection definition

axiom ABC_is_acute : is_acute_triangle α β γ
axiom A_is_largest_angle : α > β ∧ α > γ
axiom circumradius_R : R = circumradius_trio_select A B C
axiom inradius_r : r = inradius_trio_select A B C
axiom midpoint_M_ : M B C = midpoint_M B C
axiom intersection_X_ : X B C = intersection_tangents_X B C

theorem inequality_proof : 
  ∀ (A B C : Type) (α β γ R r : ℝ) (M X : A → B → C),
    is_acute_triangle α β γ →
    α > β ∧ α > γ →
    R = circumradius_trio_select A B C →
    r = inradius_trio_select A B C →
    M B C = midpoint_M B C →
    X B C = intersection_tangents_X B C →
    (r / R) ≥ (dist (M B C) (A)) / (dist (X B C) (A)) :=
by {
  intros,
  sorry, -- Place the proof here
}

end inequality_proof_l818_818195


namespace table_occupancy_l818_818174

def num_invited : ℕ := 18
def num_no_show : ℕ := 12
def num_tables : ℕ := 2

theorem table_occupancy :
  let num_attended := num_invited - num_no_show in
  let num_each_table := num_attended / num_tables in
  num_each_table = 3 := by
sorry

end table_occupancy_l818_818174


namespace math_problem_l818_818335

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x * Real.exp x else -x * Real.exp (-x)

theorem math_problem (x1 x2 : ℝ) :
  (∀ x, f(-x) = -f(x)) ∧
  (∀ x < 0, f(x) = x * Real.exp x) ∧
  ¬ (∀ x > 0, f(x) = -x * Real.exp (-x)) ∧
  (∀ x, (x ∈ Iio (-1) ∨ x ∈ Ioi 1) → (f' x < 0)) ∧
  (∀ x1 x2, abs (f x1 - f x2) < 2 / Real.exp 1) :=
begin
  sorry -- Proof not required.
end

end math_problem_l818_818335


namespace probability_problem_l818_818609

def ang_blocks : List String := ["red", "blue", "yellow", "white", "green", "orange"]
def ben_blocks : List String := ["red", "blue", "yellow", "white", "green", "orange"]
def jasmin_blocks : List String := ["red", "blue", "yellow", "white", "green", "orange"]

def boxes : Fin 6 := sorry  -- represents 6 empty boxes
def white_restriction (box : Fin 6) : Prop := box ≠ 0  -- white block can't be in the first box

def probability_at_least_one_box_three_same_color : ℚ := 1 / 72  -- The given probability

theorem probability_problem (p q : ℕ) 
  (hpq_coprime : Nat.gcd p q = 1) 
  (hprob_eq : probability_at_least_one_box_three_same_color = p / q) :
  p + q = 73 :=
sorry

end probability_problem_l818_818609


namespace fir_trees_count_l818_818299

theorem fir_trees_count (N : ℕ) :
  (N = 15 ∨ (N < 25 ∧ 11 ∣ N) ∨ 22 ∣ N) ∧ 
  (1 ≤ (if N = 15 then 1 else 0) + (if 11 ∣ N then 1 else 0) + (if N < 25 then 1 else 0) + (if 22 ∣ N then 1 else 0) ∧
   2 ≠ (if N = 15 then 1 else 0) + (if 11 ∣ N then 1 else 0) + (if N < 25 then 1 else 0) + (if 22 ∣ N then 1 else 0)) → N = 11 :=
begin
  sorry
end

end fir_trees_count_l818_818299


namespace periodic_f_2_odd_function_symmetry_problem_statement_l818_818834

noncomputable def f : ℝ → ℝ := 
  λ x, if h1 : 0 < x ∧ x ≤ 1/2 then 2 * x^2 else
  if h2 : 1/2 < x ∧ x < 1 then f (1 - x) else
  if h3 : 1 ≤ x then f (x-2) else
  -f (-x)

theorem periodic_f_2 (x : ℝ) : f (x + 2) = f x := sorry

@[simp] 
theorem odd_function (x : ℝ) : f (-x) = - f x := sorry

@[simp]
theorem symmetry (x : ℝ) : f (x) = f (1 - x) := sorry

theorem problem_statement : f 3 + f (-5/2) = -0.5 := sorry

end periodic_f_2_odd_function_symmetry_problem_statement_l818_818834


namespace extend_finitely_additive_l818_818940

variables {Ω : Type*} 
variables {𝒜 : 𝒫 (Ω)} -- algebra of subsets of Ω
variables {𝒫𝒜 : 𝒫 (Ω) → ℝ} -- finitely-additive measure on 𝒜

-- Definition of a finitely-additive probability measure
def is_finitely_additive (P : 𝒫(Ω) → ℝ) := 
  ∀ (A B : 𝒫 (Ω)), disjoint A B → P (A ∪ B) = P A + P B

-- Definition of a finitely-additive probability measure on an algebra
def is_finitely_additive_on_algebra (P : 𝒫(Ω) → ℝ) (𝒜 : 𝒫 (Ω)) := 
  ∀ (A B ∈ 𝒜), disjoint A B → P (A ∪ B) = P A + P B

-- Proof statement
theorem extend_finitely_additive (P : 𝒫(Ω) → ℝ) (𝒜 : 𝒫(Ω)) 
  (hP : is_finitely_additive_on_algebra P 𝒜) : 
  ∃ (P_max : 𝒫(Ω) → ℝ), 
    is_finitely_additive P_max ∧ 
    ∀ A ∈ 𝒜, P_max A = P A := 
begin
  -- Proof would go here
  sorry
end

end extend_finitely_additive_l818_818940


namespace coeff_x2_term_sum_bins_binom_coeff_seq_l818_818162

-- Problem (1)
theorem coeff_x2_term_sum_bins (n : ℕ) (h1 : 2^n = 64) :
  ∑ k in range (n), if k = 2 then (n choose k) else 0 = 35 :=
by sorry

-- Problem (2)
theorem binom_coeff_seq (n : ℕ) (h2 : 2 * (n choose 5) = (n choose 4) + (n choose 6)) :
  if n = 14 then (2 ^ (2 * 7 - 14) * (14 choose 7) = 3432) 
  else if n = 7 then (2 ^ (2 * 3 - 7) * (7 choose 3) = 35 / 2 ∧ 2 ^ (2 * 4 - 7) * (7 choose 4) = 70)
  else False :=
by sorry

end coeff_x2_term_sum_bins_binom_coeff_seq_l818_818162


namespace incorrect_major_premise_l818_818401

-- Define conditions
def isRhombus (r : Type) [metric_space r] : Prop := sorry
def diagonalsEqual (r : Type) [metric_space r] : Prop := sorry
def isSquare (s : Type) [metric_space s] : Prop := sorry

-- Define major, minor premise and conclusion
def major_premise (r : Type) [metric_space r] : Prop := diagonalsEqual r
def minor_premise (s : Type) [metric_space s] : Prop := isRhombus s
def conclusion (s : Type) [metric_space s] : Prop := diagonalsEqual s

-- The proof problem
theorem incorrect_major_premise (r : Type) [metric_space r] [isRhombus r] (h : major_premise r) : false := sorry

end incorrect_major_premise_l818_818401


namespace sum_possible_values_l818_818509

theorem sum_possible_values (N : ℤ) (h : N * (N - 8) = -7) : 
  ∀ (N1 N2 : ℤ), (N1 * (N1 - 8) = -7) ∧ (N2 * (N2 - 8) = -7) → (N1 + N2 = 8) :=
by
  sorry

end sum_possible_values_l818_818509


namespace find_integer_triplets_l818_818667

theorem find_integer_triplets (x y z : ℤ) : 
  x^3 + y^3 + z^3 - 3 * x * y * z = 2003 ↔ (x = 668 ∧ y = 668 ∧ z = 667) ∨ (x = 668 ∧ y = 667 ∧ z = 668) ∨ (x = 667 ∧ y = 668 ∧ z = 668) :=
by
  sorry

end find_integer_triplets_l818_818667


namespace unique_sum_of_cubes_lt_1000_l818_818765

theorem unique_sum_of_cubes_lt_1000 : 
  let max_cube := 11 
  let max_val := 1000 
  ∃ n : ℕ, n = 35 ∧ ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ max_cube → 1 ≤ b ∧ b ≤ max_cube → a^3 + b^3 < max_val :=
sorry

end unique_sum_of_cubes_lt_1000_l818_818765


namespace last_integer_in_sequence_is_21853_l818_818063

def is_divisible_by (n m : ℕ) : Prop := 
  ∃ k : ℕ, n = m * k

-- Conditions
def starts_with : ℕ := 590049
def divides_previous (a b : ℕ) : Prop := b = a / 3

-- The target hypothesis to prove
theorem last_integer_in_sequence_is_21853 :
  ∀ (a b c d : ℕ),
    a = starts_with →
    divides_previous a b →
    divides_previous b c →
    divides_previous c d →
    ¬ is_divisible_by d 3 →
    d = 21853 :=
by
  intros a b c d ha hb hc hd hnd
  sorry

end last_integer_in_sequence_is_21853_l818_818063


namespace probability_intersecting_diagonals_l818_818927

def number_of_vertices := 10

def number_of_diagonals : ℕ := Nat.choose number_of_vertices 2 - number_of_vertices

def number_of_ways_choose_two_diagonals := Nat.choose number_of_diagonals 2

def number_of_sets_of_intersecting_diagonals : ℕ := Nat.choose number_of_vertices 4

def intersection_probability : ℚ :=
  (number_of_sets_of_intersecting_diagonals : ℚ) / (number_of_ways_choose_two_diagonals : ℚ)

theorem probability_intersecting_diagonals :
  intersection_probability = 42 / 119 :=
by
  sorry

end probability_intersecting_diagonals_l818_818927


namespace exists_infinite_M_l818_818309

open Nat

noncomputable def f : ℕ → ℕ := sorry

axiom f_condition (x : ℕ) : f x + f (x + 2) ≤ 2 * f (x + 1)

theorem exists_infinite_M : ∃ (M : Set ℕ), (M.Infinite ∧ ∀ i j k ∈ M, (i - j) * f k + (j - k) * f i + (k - i) * f j = 0) :=
sorry

end exists_infinite_M_l818_818309


namespace sum_coeffs_no_y_eq_64_l818_818520

theorem sum_coeffs_no_y_eq_64 :
  let expr := (x + y + 3)^3 in
  ∑ (term) in terms_without_y expr, coeff term = 64 :=
by
  sorry

end sum_coeffs_no_y_eq_64_l818_818520


namespace function_example_l818_818627

noncomputable def f : ℝ → ℝ := λ x, x^2

theorem function_example :
  (∀ x y : ℝ, x < y → y < -1 → f x > f y) ∧
  ((∀ x : ℝ, f (-x) = f x) ∨ (∀ x : ℝ, f (-x) = -f x)) ∧
  (∃ m : ℝ, ∀ x : ℝ, f x ≥ m) :=
by
  split
  case a =>
    intros x y hxy hy_neg1
    sorry
  case b =>
    left
    intros x
    sorry
  case c =>
    use 0
    intro x
    sorry

end function_example_l818_818627


namespace equilateral_triangle_congruence_l818_818015

structure EquilateralTriangle (P Q R : Type) :=
(eq_sides : dist P Q = dist Q R ∧ dist Q R = dist R P)

variables {P Q R A B C D E : Type} [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]

theorem equilateral_triangle_congruence (hABC : EquilateralTriangle A B C) 
  (hBDE : EquilateralTriangle B D E) (hD_on_BC : D ∈ line_segment B C) : dist C E = dist A D :=
  sorry

end equilateral_triangle_congruence_l818_818015


namespace route1_cost_le_route2_cost_l818_818076

-- Given a set of cities and the travel costs between them
def travel_cost (city_a city_b : ℕ) : ℕ := sorry -- Assume a function that gives the travel cost between cities

-- Define the set of cities
def cities : Set ℕ := sorry -- Assume a set of cities

-- Define the route cost function which calculates the total cost of a given route
def route_cost (route : List ℕ) : ℕ :=
  route.pairwise (λ a b, travel_cost a b).sum

-- Define a predicate for a valid route (each city is visited exactly once)
def is_valid_route (route : List ℕ) : Prop :=
  route.nodup ∧ route.to_set = cities

-- Route 1: The city with the lowest travel cost is chosen at each step
def route1 : List ℕ := sorry -- Assume we have the route following the lowest cost rule

-- Route 2: The city with the highest travel cost is chosen at each step
def route2 : List ℕ := sorry -- Assume we have the route following the highest cost rule

-- Statement to prove
theorem route1_cost_le_route2_cost
  (h1 : is_valid_route route1)
  (h2 : is_valid_route route2) :
  route_cost route1 ≤ route_cost route2 :=
sorry

end route1_cost_le_route2_cost_l818_818076


namespace power_function_log_sum_l818_818732

noncomputable def f (x : ℝ) : ℝ := x ^ (1 / 2)

theorem power_function_log_sum (h : f (1 / 2) = (Real.sqrt 2) / 2):
  Real.log10 (f 2) + Real.log10 (f 5) = 1 / 2 := by
  sorry

end power_function_log_sum_l818_818732


namespace simplest_quadratic_radical_l818_818194

-- Define the given options as lean definitions
def optionA := Real.sqrt 32
def optionB := Real.sqrt 40
def optionC := Real.sqrt (4 / 3)
def optionD := Real.sqrt 5

-- Define the statement that needs to be proved.
theorem simplest_quadratic_radical :
  (optionD = Real.sqrt 5) ∧ 
  (¬ (optionA = Real.sqrt 5)) ∧ 
  (¬ (optionB = Real.sqrt 5)) ∧ 
  (¬ (optionC = Real.sqrt 5)) :=
sorry

end simplest_quadratic_radical_l818_818194


namespace smallest_n_for_common_factor_l818_818536

theorem smallest_n_for_common_factor : ∃ n : ℕ, n > 0 ∧ (Nat.gcd (11 * n - 3) (8 * n + 4) > 1) ∧ n = 42 := 
by
  sorry

end smallest_n_for_common_factor_l818_818536


namespace lemonade_cost_l818_818238

theorem lemonade_cost 
    (muffin_cost : ℝ)
    (coffee_cost : ℝ)
    (soup_cost : ℝ)
    (salad_cost : ℝ)
    (lunch_is_3_more : ∀ breakfast_cost lunch_cost : ℝ, lunch_cost = breakfast_cost + 3) 
    (breakfast_cost : ℝ := muffin_cost + coffee_cost)
    (lunch_cost : ℝ := breakfast_cost + 3)
    (total_soup_salad_cost : ℝ := soup_cost + salad_cost) :
    muffin_cost = 2 →
    coffee_cost = 4 →
    soup_cost = 3 →
    salad_cost = 5.25 →
    lemonade_cost = lunch_cost - total_soup_salad_cost → lemonade_cost = 0.75 := by
  intros h1 h2 h3 h4 h5
  sorry

end lemonade_cost_l818_818238


namespace number_of_fir_trees_is_11_l818_818276

theorem number_of_fir_trees_is_11 
  (N : ℕ)
  (Anya : N = 15)
  (Borya : N % 11 = 0)
  (Vera : N < 25)
  (Gena : N % 22 = 0)
  (OneBoyOneGirlTrue : (Anya ∨ Borya) ∧ (Vera ∨ Gena) ∧ (¬Anya ∨ ¬Borya) ∧ (¬Vera ∨ ¬Gena)) :
  N = 11 := 
sorry

end number_of_fir_trees_is_11_l818_818276


namespace geometric_series_sum_l818_818539

theorem geometric_series_sum :
  let b1 := (3 : ℚ) / 4 in
  let r := (3 : ℚ) / 4 in
  let n := 15 in
  let result := (∑ i in finset.range n, b1 * r^i) in
  result = 3177878751 / 1073741824 :=
by
  let b1 := (3 : ℚ) / 4
  let r := (3 : ℚ) / 4
  let n := 15
  let result := (∑ i in finset.range n, b1 * r^i)
  exact (∑ i in finset.range 15, (3 : ℚ) / 4 * ((3 : ℚ) / 4)^i) = 3177878751 / 1073741824
  sorry

end geometric_series_sum_l818_818539


namespace range_of_a_l818_818341

def f (x : ℝ) : ℝ := x^3 - 2*x + Real.exp x - (1 / Real.exp x)

theorem range_of_a (a : ℝ) : f (a - 1) + f (2 * a^2) ≤ 0 → -1 ≤ a ∧ a ≤ 1 / 2 := 
by
  sorry

end range_of_a_l818_818341


namespace least_pos_int_with_six_factors_l818_818099

theorem least_pos_int_with_six_factors :
  ∃ n : ℕ, (∀ m : ℕ, (number_of_factors m = 6 → m ≥ n)) ∧ n = 12 := 
sorry

end least_pos_int_with_six_factors_l818_818099


namespace boxes_with_neither_l818_818531

theorem boxes_with_neither (T M C B : ℕ) (h_T : T = 15) (h_M : M = 8) (h_C : C = 5) (h_B : B = 3) : 
  T - (M + C - B) = 5 := 
by {
  rw [h_T, h_M, h_C, h_B], -- Simplify using the given conditions
  norm_num -- Normalize numerical expressions to compute the result
}

end boxes_with_neither_l818_818531


namespace glued_cuboid_dimensions_l818_818602

theorem glued_cuboid_dimensions (a b a' b' c c' : ℕ) 
  (h₁ : a * b * c = 12) 
  (h₂ : a' * b' * c' = 30)
  (h₃ : (a' = a ∧ b' = b) ∨ (a' = a ∧ c' = c) ∨ (b' = b ∧ c' = c)) :
  (∃ x y z, 
    x * y * z = 42 ∧ 
    ((x = 1 ∧ y = 2 ∧ z = 21) ∨ 
    (x = 1 ∧ y = 3 ∧ z = 14) ∨
    (x = 1 ∧ y = 6 ∧ z = 7))) :=
begin
  sorry
end

end glued_cuboid_dimensions_l818_818602


namespace volume_region_inequality_l818_818237

theorem volume_region_inequality : 
  ∃ (V : ℝ), V = (20 / 3) ∧ 
    ∀ (x y z : ℝ), |x + y + z| + |x + y - z| + |x - y + z| + |-x + y + z| ≤ 4 
    → x^2 + y^2 + z^2 ≤ V :=
sorry

end volume_region_inequality_l818_818237


namespace number_of_subsets_of_M_is_4_l818_818505

def M : Set ℕ := {x | x * (x - 3) < 0 ∧ x > 0}

theorem number_of_subsets_of_M_is_4 : Set.toFinset M.card = 4 := by
  sorry

end number_of_subsets_of_M_is_4_l818_818505


namespace museums_visit_order_count_l818_818849

theorem museums_visit_order_count:
  let museums := 6 in
  ∏ i in finset.range(museums), i.succ = 720 := sorry

end museums_visit_order_count_l818_818849


namespace common_ratio_l818_818398

variable (a : ℕ → ℝ) (r : ℝ)
variable (h_geom : ∀ n, a (n+1) = r * a n)
variable (h1 : a 5 * a 11 = 3)
variable (h2 : a 3 + a 13 = 4)

theorem common_ratio (h_geom : ∀ n, a (n+1) = r * a n) (h1 : a 5 * a 11 = 3) (h2 : a 3 + a 13 = 4) :
  (r = 3 ∨ r = -3) :=
by
  sorry

end common_ratio_l818_818398


namespace circumscribed_quadrilateral_l818_818588

theorem circumscribed_quadrilateral 
  (l : Line)
  (circle1 circle2 : Circle)
  (O1 O2 : Point)
  (R1 R2 : ℝ)
  (A1 B1 A2 B2 : Point)
  (α1 α2 : ℝ) 
  (h1 : intersects l circle1 = [A1, B1])
  (h2 : intersects l circle2 = [A2, B2])
  (h3 : center circle1 = O1)
  (h4 : center circle2 = O2)
  (h5 : radius circle1 = R1)
  (h6 : radius circle2 = R2)
  (h7 : central_angle A1 B1 = 2 * α1)
  (h8 : central_angle A2 B2 = 2 * α2) :
  ∃ (circumcenter : Point), 
    is_cyclic_quadrilateral [A1, A2, B1, B2] ∧ 
    on_line circumcenter (line_through O1 O2) :=
sorry

end circumscribed_quadrilateral_l818_818588


namespace number_of_trees_is_11_l818_818262

variable {N : ℕ}

-- Conditions stated by each child
def anya_statement : Prop := N = 15
def borya_statement : Prop := N % 11 = 0
def vera_statement : Prop := N < 25
def gena_statement : Prop := N % 22 = 0

-- One boy and one girl told the truth, while the other two lied
def truth_condition : Prop :=
  (borya_statement ∧ vera_statement ∧ ¬anya_statement ∧ ¬gena_statement) ∨
  (borya_statement ∧ ¬vera_statement ∧ ¬anya_statement ∧ gena_statement) ∨
  (¬borya_statement ∧ vera_statement ∧ ¬anya_statement ∧ gena_statement) ∨
  (¬borya_statement ∧ vera_statement ∧ anya_statement ∧ ¬gena_statement)

-- Proving that the number of fir trees is 11
theorem number_of_trees_is_11 (h : truth_condition) : N = 11 := 
by
  sorry

end number_of_trees_is_11_l818_818262


namespace simplify_complex_fraction_l818_818029

theorem simplify_complex_fraction : 
  (7 + 18 * complex.I) / (3 - 4 * complex.I) = (-51 / 25 : ℂ) + (82 / 25) * complex.I :=
by {
  sorry
}

end simplify_complex_fraction_l818_818029


namespace chess_tournament_l818_818559

-- Definition for the number of players
def numPlayers : ℕ := 5

-- Calculate combination of n taken k at a time
def combination (n k : ℕ) : ℕ :=
  Nat.fact n / (Nat.fact k * Nat.fact (n - k))

-- Define the total number of games played in the tournament
def totalGames : ℕ :=
  combination numPlayers 2

theorem chess_tournament :
  totalGames = 10 :=
by
  -- Proof omitted; this is just the statement definition with placeholder.
  sorry

end chess_tournament_l818_818559


namespace total_matchsticks_l818_818227

theorem total_matchsticks (boxes : ℕ) (matchboxes_per_box : ℕ) (sticks_per_matchbox : ℕ) 
  (h1 : boxes = 4) (h2 : matchboxes_per_box = 20) (h3 : sticks_per_matchbox = 300) :
  boxes * matchboxes_per_box * sticks_per_matchbox = 24000 :=
by 
  rw [h1, h2, h3];
  norm_num

end total_matchsticks_l818_818227


namespace carters_class_A_students_l818_818450

noncomputable def students_receiving_A (total_students : ℕ) (ratio : ℚ) : ℕ :=
  (total_students * ratio).toNat

theorem carters_class_A_students :
  let total_students := 30
  let ratio := (2 : ℚ) / 3
  let count_A := 20
  let percentage := (count_A.toRat / total_students.toRat) * 100 
  students_receiving_A total_students ratio = count_A ∧ percentage = 66.67 := by
  sorry

end carters_class_A_students_l818_818450


namespace min_OP_PF_squared_l818_818773

def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in (x^2 / 2 + y^2 = 1)

def OP_squared (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P in x^2 + y^2

def PF_squared (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P in (x + 1)^2 + y^2

theorem min_OP_PF_squared :
  ∃ P : ℝ × ℝ, is_on_ellipse P ∧ (∀ Q : ℝ × ℝ, is_on_ellipse Q → OP_squared Q + PF_squared Q ≥ 2) :=
by
  sorry

end min_OP_PF_squared_l818_818773


namespace Delaney_missed_bus_by_l818_818634

def time_in_minutes (hours : ℕ) (minutes : ℕ) : ℕ := hours * 60 + minutes

def Delaney_start_time : ℕ := time_in_minutes 7 50
def bus_departure_time : ℕ := time_in_minutes 8 0
def travel_duration : ℕ := 30

theorem Delaney_missed_bus_by :
  Delaney_start_time + travel_duration - bus_departure_time = 20 :=
by
  sorry

end Delaney_missed_bus_by_l818_818634


namespace average_weight_of_all_girls_l818_818486

theorem average_weight_of_all_girls (avg1 : ℝ) (n1 : ℕ) (avg2 : ℝ) (n2 : ℕ) :
  avg1 = 50.25 → n1 = 16 → avg2 = 45.15 → n2 = 8 → 
  ((n1 * avg1 + n2 * avg2) / (n1 + n2)) = 48.55 := 
by
  intros h1 h2 h3 h4
  sorry

end average_weight_of_all_girls_l818_818486


namespace geom_series_sum_l818_818541

def geom_sum (b1 : ℚ) (r : ℚ) (n : ℕ) : ℚ := 
  b1 * (1 - r^n) / (1 - r)

def b1 : ℚ := 3 / 4
def r : ℚ := 3 / 4
def n : ℕ := 15

theorem geom_series_sum :
  geom_sum b1 r n = 3177884751 / 1073741824 :=
by sorry

end geom_series_sum_l818_818541


namespace smallest_positive_n_common_factor_l818_818534

open Int

theorem smallest_positive_n_common_factor :
  ∃ n : ℕ, n > 0 ∧ gcd (11 * n - 3) (8 * n + 4) > 1 ∧ n = 5 :=
by
  sorry

end smallest_positive_n_common_factor_l818_818534


namespace fill_time_after_turn_off_l818_818936

theorem fill_time_after_turn_off (p q : ℚ) (h1 : p = 1/12) (h2 : q = 1/15) :
  let combined_rate := p + q in
  let filled_6_min := combined_rate * 6 in
  let remaining := 1 - filled_6_min in
  let time_for_remaining := remaining / q in
  time_for_remaining = 1.5 :=
by
  sorry

end fill_time_after_turn_off_l818_818936


namespace count_five_digit_numbers_with_digit_8_l818_818691

theorem count_five_digit_numbers_with_digit_8 : 
    let total_numbers := 99999 - 10000 + 1
    let without_8 := 8 * (9 ^ 4)
    90000 - without_8 = 37512 := by
    let total_numbers := 99999 - 10000 + 1 -- Total number of five-digit numbers
    let without_8 := 8 * (9 ^ 4) -- Number of five-digit numbers without any '8'
    show total_numbers - without_8 = 37512
    sorry

end count_five_digit_numbers_with_digit_8_l818_818691


namespace contrapositive_false_1_negation_false_1_l818_818147

theorem contrapositive_false_1 (m : ℝ) : ¬ (¬ (∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) :=
sorry

theorem negation_false_1 (m : ℝ) : ¬ ((m > 0) → ¬ (∃ x : ℝ, x^2 + x - m = 0)) :=
sorry

end contrapositive_false_1_negation_false_1_l818_818147


namespace problem1_problem2_l818_818326

-- Problem 1: Prove the solution set for the inequality f(x) < 8
theorem problem1 : ∀ (x : ℝ), |x - 2| + |x + 2| + 2 < 8 ↔ -3 < x ∧ x < 3 := by
  sorry

-- Problem 2: Prove that a^2 + b^2 + c^2 ≥ 1/3 given the conditions
theorem problem2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h : ∀ x : ℝ, f x = |a - x| + |x + b| + c) 
    (hmin : ∃ x : ℝ, f x = 1) : a^2 + b^2 + c^2 ≥ 1 / 3 := by
  sorry

end problem1_problem2_l818_818326


namespace parabola_sum_l818_818906

theorem parabola_sum (a b : ℝ) 
  (h1 : ∀ x : ℝ, y = ax^2 + 3 →  x ∈ {-sqrt(-3/a), sqrt(-3/a)})
  (h2 : ∀ x : ℝ, y = 7 - bx^2 →  x ∈ {-sqrt(7/b), sqrt(7/b)})
  (h3 : -(sqrt(-3/a)) = -sqrt(7/b))
  (h4 : (2 * sqrt(-3/a)) * (4) = 36) : a + b = 4/27 := by
  sorry

end parabola_sum_l818_818906


namespace least_positive_integer_with_six_distinct_factors_l818_818092

theorem least_positive_integer_with_six_distinct_factors : ∃ n : ℕ, (∀ k : ℕ, (number_of_factors k = 6) → (n ≤ k)) ∧ (number_of_factors n = 6) ∧ (n = 12) :=
by
  sorry

end least_positive_integer_with_six_distinct_factors_l818_818092


namespace least_positive_integer_with_six_factors_l818_818113

-- Define what it means for a number to have exactly six distinct positive factors
def hasExactlySixFactors (n : ℕ) : Prop :=
  (n.factorization.support.card = 2 ∧ (n.factorization.values' = [2, 1])) ∨
  (n.factorization.support.card = 1 ∧ (n.factorization.values' = [5]))

-- The main theorem statement
theorem least_positive_integer_with_six_factors : ∃ n : ℕ, hasExactlySixFactors n ∧ ∀ m : ℕ, (hasExactlySixFactors m → n ≤ m) :=
  exists.intro 12 (and.intro
    (show hasExactlySixFactors 12, by sorry)
    (show ∀ m : ℕ, hasExactlySixFactors m → 12 ≤ m, by sorry))

end least_positive_integer_with_six_factors_l818_818113


namespace solve_eq1_solve_eq2_l818_818480

theorem solve_eq1 (x : ℝ) : 4 * x^2 = 12 * x ↔ x = 0 ∨ x = 3 :=
by
  sorry

theorem solve_eq2 (x : ℝ) : x^2 + 4 * x + 3 = 0 ↔ x = -3 ∨ x = -1 :=
by
  sorry

end solve_eq1_solve_eq2_l818_818480


namespace linear_function_value_change_l818_818832

theorem linear_function_value_change (g : ℝ → ℝ) (h1 : ∀ x y, g(x) - g(y) = 2 * (x - y)) :
  g(10) - g(0) = 20 :=
by {
  sorry
}

end linear_function_value_change_l818_818832


namespace cubs_win_probability_is_57_percent_l818_818885

noncomputable def factorial (n : Nat) : Nat :=
by
  sorry

noncomputable def binomial_coefficient (n k : Nat) : Nat :=
by
  sorry

noncomputable def cubs_win_world_series_probability : ℚ :=
let p_cubs_win := 4 / 7
let p_red_sox_win := 3 / 7
let term (k : Nat) : ℚ :=
  binomial_coefficient (4 + k) k * p_cubs_win^5 * p_red_sox_win^k
(∑ k in Finset.range 5, term k)

theorem cubs_win_probability_is_57_percent :
  cubs_win_world_series_probability = 0.57 :=
by
  sorry

end cubs_win_probability_is_57_percent_l818_818885


namespace multiple_of_fair_tickets_l818_818899

theorem multiple_of_fair_tickets (fair_tickets_sold : ℕ) (game_tickets_sold : ℕ) (h : fair_tickets_sold = game_tickets_sold * x + 6) :
  25 = 56 * x + 6 → x = 19 / 56 := by
  sorry

end multiple_of_fair_tickets_l818_818899


namespace probability_of_intersection_in_decagon_is_fraction_l818_818930

open_locale big_operators

noncomputable def probability_intersecting_diagonals_in_decagon : ℚ :=
let num_points := 10 in
let diagonals := (num_points.choose 2) - num_points in
let total_diagonal_pairs := (diagonals.choose 2) in
let valid_intersections := (num_points.choose 4) in
valid_intersections / total_diagonal_pairs

theorem probability_of_intersection_in_decagon_is_fraction :
  probability_intersecting_diagonals_in_decagon = 42 / 119 :=
by {
  unfold probability_intersecting_diagonals_in_decagon,
  sorry
}

end probability_of_intersection_in_decagon_is_fraction_l818_818930


namespace question1_a_minus_1_question2_A_intersection_B_empty_l818_818754

-- Definitions of sets A and B
def setA (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x < a + 3}
def setB : Set ℝ := {x | x < -1 ∨ x > 5}

-- Question 1: If a=-1, find A ∪ B and (∁_R A) ∩ B
theorem question1_a_minus_1 :
  ∀ a : ℝ, a = -1 → (setA a ∪ setB) = {x | x < 2 ∨ x > 5} ∧ 
            (setA aᶜ ∩ setB) = {x | x < -2 ∨ x > 5} :=
by
  sorry

-- Question 2: If A ∩ B = ∅, find range of values for a
theorem question2_A_intersection_B_empty :
  ∀ a : ℝ, (setA a ∩ setB) = ∅ → (a ≥ 3 ∨ -0.5 ≤ a ∧ a ≤ 2) :=
by 
  sorry

end question1_a_minus_1_question2_A_intersection_B_empty_l818_818754


namespace smallest_slope_tangent_line_l818_818746

/-- A function defined as f(x) = 2x^3 - 3x -/
def f (x : ℝ) : ℝ := 2 * x ^ 3 - 3 * x

/-- The derivative of f defined as f'(x) = 6x^2 - 3 -/
def f' (x : ℝ) : ℝ := 6 * x ^ 2 - 3

theorem smallest_slope_tangent_line
  (tangent_eq : ∀ (x : ℝ), (y : ℝ), y = f' x * (x - 0) + f 0 → y = -3 * x) :
  tangent_eq 0 0 :=
by
  sorry

end smallest_slope_tangent_line_l818_818746


namespace concyclic_A_N_F_P_l818_818164

open EuclideanGeometry

-- Definitions for the terms used in the conditions
variables (A B C M N P D E F : Point)
variables (triangle_ABC : Triangle A B C)
variables (M_mid : Midpoint M B C)
variables (N_mid : Midpoint N C A)
variables (P_mid : Midpoint P A B)
variables (D_on_perp_bisector_AB : OnPerpBisector D A B)
variables (E_on_perp_bisector_AC : OnPerpBisector E A C)
variables (D_on_ray_AM : OnRay A M D)
variables (E_on_ray_AM : OnRay A M E)
variables (BD : LineThrough B D)
variables (CE : LineThrough C E)
variables (F_inside_triangle : InsideTriangle F A B C)

-- The theorem to be proved
theorem concyclic_A_N_F_P 
  (h_triangle : IsAcuteScaleneTriangle triangle_ABC) 
  (h1 : Midpoints M N P A B C) 
  (h2 : OnPerpBisector D A B)
  (h3 : OnPerpBisector E A C)
  (h4 : LineIntersection BD CE F)
  : Concyclic A N F P :=
by
sry -- the proof is not required

end concyclic_A_N_F_P_l818_818164


namespace inequality1_inequality2_l818_818481

noncomputable def sqrt_5 : ℝ := real.sqrt 5

theorem inequality1 (x : ℝ) : 
  x^2 - 5*x + 5 > 0 ↔ x > (5 + sqrt_5) / 2 ∨ x < (5 - sqrt_5) / 2 := 
by sorry

theorem inequality2 (x : ℝ) :
  -2*x^2 + x - 3 < 0 ↔ true :=
by sorry

end inequality1_inequality2_l818_818481


namespace lcm_36_105_l818_818680

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end lcm_36_105_l818_818680


namespace find_x_l818_818305

theorem find_x (x : ℝ) (h : x ^ 2 ∈ ({1, 0, x} : Set ℝ)) : x = -1 := 
sorry

end find_x_l818_818305


namespace problem_solution_l818_818917

noncomputable def a : ℕ → ℚ
| 1 := -0.5
| (n + 1) := 1 / (1 - a n)

def periodic_3 (a : ℕ → ℚ) := ∀ n, a (n + 3) = a n

theorem problem_solution : 
  a 2 = 2/3 ∧ a 3 = 3 ∧ a 4 = -1/2 ∧ 
  periodic_3 a ∧ 
  a 1998 = 3 ∧ a 2000 = 2/3 := by 
{
  sorry
}

end problem_solution_l818_818917


namespace number_of_fir_trees_is_11_l818_818273

theorem number_of_fir_trees_is_11 
  (N : ℕ)
  (Anya : N = 15)
  (Borya : N % 11 = 0)
  (Vera : N < 25)
  (Gena : N % 22 = 0)
  (OneBoyOneGirlTrue : (Anya ∨ Borya) ∧ (Vera ∨ Gena) ∧ (¬Anya ∨ ¬Borya) ∧ (¬Vera ∨ ¬Gena)) :
  N = 11 := 
sorry

end number_of_fir_trees_is_11_l818_818273


namespace possible_values_of_r_l818_818903

noncomputable def r : ℝ := sorry

def is_four_place_decimal (x : ℝ) : Prop := 
  ∃ (a b c d : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ x = a / 10 + b / 100 + c / 1000 + d / 10000

def is_closest_fraction (x : ℝ) : Prop := 
  abs (x - 3 / 11) < abs (x - 3 / 10) ∧ abs (x - 3 / 11) < abs (x - 1 / 4)

theorem possible_values_of_r :
  (0.2614 <= r ∧ r <= 0.2864) ∧ is_four_place_decimal r ∧ is_closest_fraction r →
  ∃ n : ℕ, n = 251 := 
sorry

end possible_values_of_r_l818_818903


namespace max_area_equilateral_triangle_in_rectangle_l818_818185

theorem max_area_equilateral_triangle_in_rectangle :
  let E := (0, 0)
  let F := (12, 0)
  let G := (12, 15)
  let H := (0, 15)
  let rectangle := [E, F, G, H]
  has_vertice_in_rectangle : (p q r : ℝ × ℝ) → p ∈ rectangle ∧ q ∈ rectangle ∧ r ∈ rectangle
  → (√3 / 4) * (side_length ^ 2) := 
  (√3 / 4) * (12 ^ 2 + (12 * √3 - 10) ^ 2) = 156 * √3 - 310 :=
sorry

end max_area_equilateral_triangle_in_rectangle_l818_818185


namespace remaining_dogs_after_adoptions_l818_818069

theorem remaining_dogs_after_adoptions 
  (initial_dogs : ℕ)
  (additional_dogs : ℕ)
  (adopted_week1 : ℕ)
  (adopted_week4 : ℕ) :
  initial_dogs = 200 →
  additional_dogs = 100 →
  adopted_week1 = 40 →
  adopted_week4 = 60 →
  initial_dogs + additional_dogs - adopted_week1 - adopted_week4 = 200 :=
by
  intros h_init h_add h_adopt1 h_adopt2
  rw [h_init, h_add, h_adopt1, h_adopt2]
  rfl

end remaining_dogs_after_adoptions_l818_818069


namespace dice_probability_abs_diff_2_l818_818967

theorem dice_probability_abs_diff_2 :
  let total_outcomes := 36
  let favorable_outcomes := 8
  let probability := favorable_outcomes / total_outcomes
  probability = 2 / 9 :=
by
  sorry

end dice_probability_abs_diff_2_l818_818967


namespace game_win_probability_l818_818183

theorem game_win_probability (p : ℝ) (h1 : irrational p) (h2 : 0 < p ∧ p < 1) :
  ∃ rule : (ℕ → bool) → Prop,
    (∀ (s : ℕ → bool), (rule s → probability_win s = p)) ∧ 
    (game_ends_finite (rule)) :=
by
  -- proof goes here
  sorry

end game_win_probability_l818_818183


namespace triangle_ratios_l818_818977

theorem triangle_ratios (AC BC AD: ℝ) (C D : ℝ × ℝ)
  (h1 : (C.1 - D.1) ≠ 0) (h2 : AC = 4) (h3 : BC = 5) (h4 : AD = 15) 
  (h5 : C.y ≠ D.y):
    ∃ m n : ℕ, (m.gcd n = 1) ∧ (m + n = 91 ∧ ((DE / DB : ℚ) = 4 / 87)) :=
by
  sorry

end triangle_ratios_l818_818977


namespace first_term_exceeds_10000_l818_818894

def sequence (n : ℕ) : ℕ :=
  if n = 1 then 2 else 2 ^ (n - 1)

theorem first_term_exceeds_10000 : ∃ n, sequence n > 10000 ∧ sequence n = 16384 :=
by 
  sorry

end first_term_exceeds_10000_l818_818894


namespace arrangements_ABC_3rooms_l818_818587

open Finset

def people : Finset (Fin 5) := Finset.range 5
def rooms : Finset (Fin 3) := Finset.range 3

theorem arrangements_ABC_3rooms : 
  ∃! (f : Fin 5 → Fin 3), (∀ r ∈ rooms, ∃ a ∈ people, f a = r) ∧ ((¬ ∃ r ∈ rooms, f 0 = r ∧ f 1 = r)) → 114 :=
sorry

end arrangements_ABC_3rooms_l818_818587


namespace simon_students_l818_818478

theorem simon_students (S L : ℕ) (h1 : S = 4 * L) (h2 : S + L = 2500) : S = 2000 :=
by {
  sorry
}

end simon_students_l818_818478


namespace integer_values_of_f_l818_818625

def f (x : ℝ) : ℝ := (1 + x)^(1/3) + (3 - x)^(1/3)

theorem integer_values_of_f :
  ∀ x : ℝ, (∃ n : ℤ, f(x) = n) ↔ x ∈ {1 - sqrt 5, 1 + sqrt 5, 1 - (10/9) * sqrt 3, 1 + (10/9) * sqrt 3} := sorry

end integer_values_of_f_l818_818625


namespace right_triangle_median_area_l818_818798

theorem right_triangle_median_area 
  (A B C D E G : Type) 
  [AddCommGroup A] [Module ℝ A] 
  [AddCommGroup B] [Module ℝ B] 
  [AddCommGroup C] [Module ℝ C] 
  [AddCommGroup D] [Module ℝ D] 
  [AddCommGroup E] [Module ℝ E] 
  [AddCommGroup G] [Module ℝ G]
  (hABC : right_triangle ABC (angle BCA = pi / 2))
  (hAD : median AD (∥AD∥ = 18))
  (hBE : median BE (∥BE∥ = 24))
  (perpendicular : ∥AD∥ * ∥BE∥ = 1):
  area ABC = 288 := 
sorry

end right_triangle_median_area_l818_818798


namespace least_positive_integer_with_six_factors_l818_818107

theorem least_positive_integer_with_six_factors :
  ∃ n : ℕ, (∀ m : ℕ, m > 0 → m < n → (count_factors m ≠ 6)) ∧ count_factors n = 6 ∧ n = 18 :=
sorry

noncomputable def count_factors (n : ℕ) : ℕ :=
sorry

end least_positive_integer_with_six_factors_l818_818107


namespace lcm_36_105_l818_818685

theorem lcm_36_105 : Int.lcm 36 105 = 1260 := by
  sorry

end lcm_36_105_l818_818685


namespace spot_reachable_area_l818_818482

-- Define the hexagon side length and the rope length
def side_length : ℝ := 2
def rope_length : ℝ := 4

-- Define the areas of the sectors
def large_sector_area : ℝ := π * rope_length^2 * (240/360)
def small_sector_area : ℝ := π * side_length^2 * (60/360)

-- The total accessible area outside the hexagon
theorem spot_reachable_area :
  let total_accessible_area := large_sector_area + (2 * small_sector_area)
  total_accessible_area = 12 * π :=
by
  -- Skip the proof
  sorry

end spot_reachable_area_l818_818482


namespace sin_of_7pi_over_6_l818_818661

theorem sin_of_7pi_over_6 : Real.sin (7 * Real.pi / 6) = -1 / 2 :=
by
  sorry

end sin_of_7pi_over_6_l818_818661


namespace fir_trees_alley_l818_818278

-- Define the statements made by each child
def statementAnya (N : ℕ) : Prop := N = 15
def statementBorya (N : ℕ) : Prop := N % 11 = 0
def statementVera (N : ℕ) : Prop := N < 25
def statementGena (N : ℕ) : Prop := N % 22 = 0

-- Define the condition about the truth and lies
def oneBoyOneGirlTruth (anya_vera_truth: Prop) (borya_gena_truth: Prop) : Prop :=
  anya_vera_truth ∧ borya_gena_truth ∧
  ((statementAnya N ∧ statementVera N) ∨ (statementVera N ∧ statementBorya N)) ∧
  ¬( (statementAnya N ∧ statementGena N) ∨ (statementVera N ∧ statementGena N) ∨
     (statementAnya N ∧ statementBorya N) ∨ (statementBorya N ∧ statementGena N) )

-- Prove that the number of fir trees is 11
theorem fir_trees_alley: ∃ (N : ℕ), statementBorya N ∧ statementVera N ∧ ¬ statementAnya N ∧ ¬ statementGena N ∧ oneBoyOneGirlTruth (¬ statementAnya N ∧ statementVera N) (statementBorya N ∧ ¬ statementGena N) ∧ N = 11 :=
by
  sorry

end fir_trees_alley_l818_818278


namespace triangle_perimeter_l818_818387

-- Definitions and given conditions
def side_length_a (a : ℝ) : Prop := a = 6
def inradius (r : ℝ) : Prop := r = 2
def circumradius (R : ℝ) : Prop := R = 5

-- The final proof statement to be proven
theorem triangle_perimeter (a r R : ℝ) (b c P : ℝ) 
  (h1 : side_length_a a)
  (h2 : inradius r)
  (h3 : circumradius R)
  (h4 : P = 2 * ((a + b + c) / 2)) :
  P = 24 :=
sorry

end triangle_perimeter_l818_818387


namespace daily_food_cost_l818_818198

/-
Andrea spends $15890 on her pony in a year, and she pays $500/month to rent a pasture and $60/lesson for two lessons a week.
How much does she pay per day for food?
-/

theorem daily_food_cost 
  (total_annual_expenses : ℕ) 
  (monthly_pasture_rent : ℕ) 
  (lesson_cost : ℕ) 
  (lessons_per_week : ℕ) 
  (weeks_per_year : ℕ)
  (days_per_year : ℕ) :
  total_annual_expenses = 15890 →
  monthly_pasture_rent = 500 →
  lesson_cost = 60 →
  lessons_per_week = 2 →
  weeks_per_year = 52 →
  days_per_year = 365 →
  let annual_pasture_rent : ℕ := 12 * monthly_pasture_rent 
  let total_lessons : ℕ := weeks_per_year * lessons_per_week 
  let annual_lesson_cost : ℕ := total_lessons * lesson_cost 
  let total_other_expenses : ℕ := annual_pasture_rent + annual_lesson_cost 
  let annual_food_cost : ℕ := total_annual_expenses - total_other_expenses 
  let daily_food_cost : ℕ := annual_food_cost / days_per_year 
  daily_food_cost = 10 :=
by
  intros
  rw [h, h_1, h_2, h_3, h_4, h_5]
  unfold annual_pasture_rent total_lessons annual_lesson_cost total_other_expenses annual_food_cost daily_food_cost
  sorry

end daily_food_cost_l818_818198


namespace least_positive_integer_with_six_distinct_factors_l818_818093

theorem least_positive_integer_with_six_distinct_factors : ∃ n : ℕ, (∀ k : ℕ, (number_of_factors k = 6) → (n ≤ k)) ∧ (number_of_factors n = 6) ∧ (n = 12) :=
by
  sorry

end least_positive_integer_with_six_distinct_factors_l818_818093


namespace solve_trig_eq_l818_818878

theorem solve_trig_eq :
  ∀ x k : ℤ, 
    (x = 2 * π / 3 + 2 * k * π ∨
     x = 7 * π / 6 + 2 * k * π ∨
     x = -π / 6 + 2 * k * π)
    → (|Real.cos x| + Real.cos (3 * x)) / (Real.sin x * Real.cos (2 * x)) = -2 * Real.sqrt 3 := 
by
  intros x k h
  sorry

end solve_trig_eq_l818_818878


namespace range_of_a_l818_818304

variable {x a : ℝ}

def p : Prop := -4 < x - a ∧ x - a < 4

def q : Prop := (x - 2) * (3 - x) > 0

theorem range_of_a (h : ¬p → ¬q) : -1 ≤ a ∧ a ≤ 6 :=
sorry

end range_of_a_l818_818304


namespace least_positive_integer_with_six_factors_l818_818112

theorem least_positive_integer_with_six_factors :
  ∃ n : ℕ, (∀ m : ℕ, m > 0 → m < n → (count_factors m ≠ 6)) ∧ count_factors n = 6 ∧ n = 18 :=
sorry

noncomputable def count_factors (n : ℕ) : ℕ :=
sorry

end least_positive_integer_with_six_factors_l818_818112


namespace number_of_trees_is_eleven_l818_818257

variables (N : ℕ)

-- Conditions
def Anya (N : ℕ) := N = 15
def Borya (N : ℕ) := 11 ∣ N
def Vera (N : ℕ) := N < 25
def Gena (N : ℕ) := 22 ∣ N

axiom OneBoyOneGirlTruth :
  (∃ (b : Prop) (g : Prop),
    (b ∨ ¬ b) ∧ (g ∨ ¬ g) ∧
    ((b = (Borya N ∨ Gena N)) ∧ (g = (Anya N ∨ Vera N)) ∧
     (b ↔ ¬g) ∧
     ((Anya N ∨ ¬Vera N) ∨ (¬Anya N ∨ Vera N)) ∧
     (Anya N = (N = 15)) ∧
     (Borya N = (11 ∣ N)) ∧
     (Vera N = (N < 25)) ∧
     (Gena N = (22 ∣ N))))

theorem number_of_trees_is_eleven: N = 11 :=
sorry

end number_of_trees_is_eleven_l818_818257


namespace non_neg_integers_with_abs_lt_2_5_l818_818902

noncomputable def non_negative_integers : Set ℕ := { n : ℕ | |(n : ℤ)| < 2.5 }

theorem non_neg_integers_with_abs_lt_2_5 : non_negative_integers = {0, 1, 2} :=
by
  sorry

end non_neg_integers_with_abs_lt_2_5_l818_818902


namespace least_positive_integer_with_six_factors_is_18_l818_818136

-- Define the least positive integer with exactly six distinct positive factors.
def least_positive_with_six_factors (n : ℕ) : Prop :=
  (∀ d : ℕ, d ∣ n → d > 0) ∧ (finset.card (finset.filter (λ d, d ∣ n) (finset.range (n + 1)))) = 6

-- Prove that the least positive integer with exactly six distinct positive factors is 18.
theorem least_positive_integer_with_six_factors_is_18 : (∃ n : ℕ, least_positive_with_six_factors n ∧ n = 18) :=
sorry


end least_positive_integer_with_six_factors_is_18_l818_818136


namespace set_intersection_l818_818826

theorem set_intersection (A B : Set ℝ) 
  (hA : A = { x : ℝ | 0 < x ∧ x < 5 }) 
  (hB : B = { x : ℝ | -1 ≤ x ∧ x < 4 }) : 
  (A ∩ B) = { x : ℝ | 0 < x ∧ x < 4 } :=
by
  sorry

end set_intersection_l818_818826


namespace circumscribed_quadrilateral_l818_818589

theorem circumscribed_quadrilateral 
  (l : Line)
  (circle1 circle2 : Circle)
  (O1 O2 : Point)
  (R1 R2 : ℝ)
  (A1 B1 A2 B2 : Point)
  (α1 α2 : ℝ) 
  (h1 : intersects l circle1 = [A1, B1])
  (h2 : intersects l circle2 = [A2, B2])
  (h3 : center circle1 = O1)
  (h4 : center circle2 = O2)
  (h5 : radius circle1 = R1)
  (h6 : radius circle2 = R2)
  (h7 : central_angle A1 B1 = 2 * α1)
  (h8 : central_angle A2 B2 = 2 * α2) :
  ∃ (circumcenter : Point), 
    is_cyclic_quadrilateral [A1, A2, B1, B2] ∧ 
    on_line circumcenter (line_through O1 O2) :=
sorry

end circumscribed_quadrilateral_l818_818589


namespace smallest_integer_with_six_distinct_factors_l818_818131

noncomputable def least_pos_integer_with_six_factors : ℕ :=
  12

theorem smallest_integer_with_six_distinct_factors 
  (n : ℕ)
  (p q : ℕ)
  (a b : ℕ)
  (hp : prime p)
  (hq : prime q)
  (h_diff : p ≠ q)
  (h_n : n = p ^ a * q ^ b)
  (h_factors : (a + 1) * (b + 1) = 6) :
  n = least_pos_integer_with_six_factors :=
by
  sorry

end smallest_integer_with_six_distinct_factors_l818_818131


namespace linearity_and_solutions_l818_818873

-- Constants and functions
variable {x : ℝ}
variable {a b k : ℝ → ℝ}

def linear_equation (y : ℝ → ℝ) (n : ℕ) : Prop :=
  let terms := λ (f : ℝ → ℝ) (i : ℕ), deriv^[i] f x
  a x * terms y n +
  b x * terms y (n - 1) +
  k x * y x = 0

-- Theorem Statement
theorem linearity_and_solutions (n : ℕ) :
  ∀ (y1 y2 : ℝ → ℝ)
    (h1 : linear_equation y1 n)
    (h2 : linear_equation y2 n)
    (c1 c2 : ℝ),
    linear_equation (λ x, c1 * y1 x + c2 * y2 x) n ∧
    linear_equation (λ x, (λ _, c1) x) n ∧
    linear_equation (λ x, (λ x, c2 * x + c1) x) n :=
by
  intro y1 y2 h1 h2 c1 c2
  split
  { sorry }
  split
  { sorry }
  { sorry }

end linearity_and_solutions_l818_818873


namespace timeAfter2687Minutes_l818_818046

-- We define a structure for representing time in hours and minutes.
structure Time :=
  (hour : Nat)
  (minute : Nat)

-- Define the current time
def currentTime : Time := {hour := 7, minute := 0}

-- Define a function that computes the time after adding a given number of minutes to a given time
noncomputable def addMinutes (t : Time) (minutesToAdd : Nat) : Time :=
  let totalMinutes := t.minute + minutesToAdd
  let extraHours := totalMinutes / 60
  let remainingMinutes := totalMinutes % 60
  let totalHours := t.hour + extraHours
  let effectiveHours := totalHours % 24
  {hour := effectiveHours, minute := remainingMinutes}

-- The theorem to state that 2687 minutes after 7:00 a.m. is 3:47 a.m.
theorem timeAfter2687Minutes : addMinutes currentTime 2687 = { hour := 3, minute := 47 } :=
  sorry

end timeAfter2687Minutes_l818_818046


namespace lcm_36_105_l818_818683

noncomputable def factorize_36 : List (ℕ × ℕ) := [(2, 2), (3, 2)]
noncomputable def factorize_105 : List (ℕ × ℕ) := [(3, 1), (5, 1), (7, 1)]

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 :=
by
  have h_36 : 36 = 2^2 * 3^2 := by norm_num
  have h_105 : 105 = 3^1 * 5^1 * 7^1 := by norm_num
  sorry

end lcm_36_105_l818_818683


namespace solution_set_l818_818064

noncomputable def perm_A (n : ℕ) (k : ℕ) : ℕ :=
  n.factorial / (n - k).factorial

noncomputable def comb_C (n : ℕ) (k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

theorem solution_set (n : ℕ) : perm_A n 4 ≥ 24 * comb_C n 6 ↔ n ∈ {6, 7, 8, 9} :=
by
  sorry

end solution_set_l818_818064


namespace number_of_trees_is_eleven_l818_818254

variables (N : ℕ)

-- Conditions
def Anya (N : ℕ) := N = 15
def Borya (N : ℕ) := 11 ∣ N
def Vera (N : ℕ) := N < 25
def Gena (N : ℕ) := 22 ∣ N

axiom OneBoyOneGirlTruth :
  (∃ (b : Prop) (g : Prop),
    (b ∨ ¬ b) ∧ (g ∨ ¬ g) ∧
    ((b = (Borya N ∨ Gena N)) ∧ (g = (Anya N ∨ Vera N)) ∧
     (b ↔ ¬g) ∧
     ((Anya N ∨ ¬Vera N) ∨ (¬Anya N ∨ Vera N)) ∧
     (Anya N = (N = 15)) ∧
     (Borya N = (11 ∣ N)) ∧
     (Vera N = (N < 25)) ∧
     (Gena N = (22 ∣ N))))

theorem number_of_trees_is_eleven: N = 11 :=
sorry

end number_of_trees_is_eleven_l818_818254


namespace smallest_a_l818_818733

theorem smallest_a (x a : ℝ) (hx : x > 0) (ha : a > 0) (hineq : x + a / x ≥ 4) : a ≥ 4 :=
sorry

end smallest_a_l818_818733


namespace kira_likes_number_last_digits_l818_818451

theorem kira_likes_number_last_digits :
  ∃ (S : Finset ℕ), S = {0, 2, 4, 6, 8} ∧ ∀ (n : ℕ), (n % 12 = 0) → n % 10 ∈ S := 
begin
  have S_def : Finset ℕ := {0, 2, 4, 6, 8},
  use S_def,
  split,
  { refl },
  {
    intro n,
    intro h_div,
    have h_mod10 : n % 10 = 0 ∨ n % 10 = 2 ∨ n % 10 = 4 ∨ n % 10 = 6 ∨ n % 10 = 8,
      sorry,
    exact h_mod10,
  }
end

end kira_likes_number_last_digits_l818_818451


namespace ellipse_standard_eq_find_lambda_range_area_circle_l818_818320

theorem ellipse_standard_eq (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a > b) :
  ∀ x y : ℝ, 
    (x, y) = (sqrt 2, 1) → 
    (x ^ 2) / a ^ 2 + (y ^ 2) / b ^ 2 = 1 → 
    a = sqrt 2 ∧ b = 1 → 
    x ^ 2 / 2 + y ^ 2 = 1 := 
sorry

theorem find_lambda (A B F1 F2 : ℝ × ℝ) 
  (h1 : F1 = (-1, 0)) 
  (h2 : x^2 = 2 * y) 
  (exists_lambda : ∃ λ : ℝ, 
    λ = -2 * sqrt 2 ∧ 
    |(F2 - A) - (F2 - B)| = λ * ((F1 + A) • (F2 + B))) : 
  λ = -2 * sqrt 2 :=
sorry

theorem range_area_circle (A B F1 F2 : ℝ × ℝ) 
  (exists_lambda : ∃ λ : ℝ, 
    λ = -2 * sqrt 2 ∧ 
    |(F2 - A) - (F2 - B)| = λ * ((F1 + A) • (F2 + B))) 
  (S_circle : ℝ) : 
  0 < S_circle ∧ S_circle ≤ (π / 4) :=
sorry

end ellipse_standard_eq_find_lambda_range_area_circle_l818_818320


namespace find_S20_l818_818734

variable {α : Type*} [AddCommGroup α] [Module ℝ α]
variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
axiom sum_first_n_terms (n : ℕ) : S n = n * (a 1 + a n) / 2
axiom points_collinear (A B C O : α) : Collinear ℝ ({A, B, C} : Set α) ∧ O = 0
axiom vector_relationship (A B C O : α) : O = 0 → C = (a 12) • A + (a 9) • B
axiom line_not_through_origin (A B O : α) : ¬Collinear ℝ ({O, A, B} : Set α)

-- Question: To find S 20
theorem find_S20 (A B C O : α) (h_collinear : Collinear ℝ ({A, B, C} : Set α)) 
  (h_vector : O = 0 → C = (a 12) • A + (a 9) • B) 
  (h_origin : O = 0)
  (h_not_through_origin : ¬Collinear ℝ ({O, A, B} : Set α)) : 
  S 20 = 10 := by
  sorry

end find_S20_l818_818734


namespace product_of_positive_integer_values_of_d_l818_818692

noncomputable def positive_integer_values_of_d {α : Type*} [linear_ordered_field α] (h : α) : list α :=
(list.range 11).filter (λ d, 1 ≤ d ∧ d.val.to_nat ≤ 11)

noncomputable def product_of_list {α : Type*} [comm_ring α] (l : list α) : α :=
l.foldr (λ x y => x * y) 1

theorem product_of_positive_integer_values_of_d :
  product_of_list (positive_integer_values_of_d 11) = 39916800 :=
by sorry

end product_of_positive_integer_values_of_d_l818_818692


namespace solve_problem_l818_818962

def problem_statement : ℝ :=
  (3^1 - 2 + 6^2 - 1)⁻¹ * 4 - 3 = -26 / 9

theorem solve_problem : problem_statement := by
  sorry

end solve_problem_l818_818962


namespace find_first_prime_l818_818909

theorem find_first_prime (p1 p2 z : ℕ) 
  (prime_p1 : Nat.Prime p1)
  (prime_p2 : Nat.Prime p2)
  (z_eq : z = p1 * p2)
  (z_val : z = 33)
  (p2_range : 8 < p2 ∧ p2 < 24)
  : p1 = 3 := 
sorry

end find_first_prime_l818_818909


namespace areas_of_triangles_l818_818808

noncomputable def triangle_OA2C_area : ℝ :=
  let AB := 4
  let AC := 6
  let angle_BAC := 60
  let BC := real.sqrt (AB^2 + AC^2 - 2 * AB * AC * real.cos (angle_BAC * real.pi / 180))
  let R := BC / (2 * real.sin (angle_BAC * real.pi / 180))
  let area := (real.sqrt 3 / 4) * (R^2)
  area

noncomputable def triangle_A1A2C_area : ℝ :=
  let AB := 4
  let AC := 6
  let angle_BAC := 60
  let BC := real.sqrt (AB^2 + AC^2 - 2 * AB * AC * real.cos (angle_BAC * real.pi / 180))
  let A1C := (3 / 5) * BC
  let R := BC / (2 * real.sin (angle_BAC * real.pi / 180))
  let area := (1 / 2) * A1C * R * (1 / 2)
  area

theorem areas_of_triangles :
  triangle_OA2C_area = 7 / real.sqrt 3 ∧ triangle_A1A2C_area = (7 * real.sqrt 3) / 5 :=
by
  sorry

end areas_of_triangles_l818_818808


namespace log_monotonic_increasing_l818_818839

noncomputable def f (a x : ℝ) := Real.log x / Real.log a

theorem log_monotonic_increasing (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : 1 < a) :
  f a (a + 1) > f a 2 := 
by
  -- Here the actual proof will be added.
  sorry

end log_monotonic_increasing_l818_818839


namespace find_prime_p_l818_818330

def is_prime (p: ℕ) : Prop := Nat.Prime p

def is_product_of_three_distinct_primes (n: ℕ) : Prop :=
  ∃ (p1 p2 p3: ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ n = p1 * p2 * p3

theorem find_prime_p (p: ℕ) (hp: is_prime p) :
  (∃ x y z: ℕ, x^p + y^p + z^p - x - y - z = 30) ↔ (p = 2 ∨ p = 3 ∨ p = 5) := 
sorry

end find_prime_p_l818_818330


namespace paint_used_after_four_weeks_l818_818817

def initial_paint : ℝ := 520
def first_week_paint_used : ℝ := initial_paint * 1/4
def remaining_paint_after_first_week : ℝ := initial_paint - first_week_paint_used

def second_week_paint_used : ℝ := remaining_paint_after_first_week * 1/3
def remaining_paint_after_second_week : ℝ := remaining_paint_after_first_week - second_week_paint_used

def third_week_paint_used : ℝ := remaining_paint_after_second_week * 3/8
def remaining_paint_after_third_week : ℝ := remaining_paint_after_second_week - third_week_paint_used

def fourth_week_paint_used : ℝ := remaining_paint_after_third_week * 1/5
def remaining_paint_after_fourth_week : ℝ := remaining_paint_after_third_week - fourth_week_paint_used

def total_paint_used : ℝ := 
  first_week_paint_used + 
  second_week_paint_used + 
  third_week_paint_used + 
  fourth_week_paint_used

theorem paint_used_after_four_weeks : total_paint_used = 390 := by
  sorry

end paint_used_after_four_weeks_l818_818817


namespace integral_exp_plus_2x_l818_818651

theorem integral_exp_plus_2x :
  ∫ x in 0..1, (Real.exp x + 2 * x) = Real.exp 1 := by
sorry

end integral_exp_plus_2x_l818_818651


namespace distance_difference_l818_818898

noncomputable def point := ℝ × ℝ

def Q : point := (2, 0)

def line_eq (x y : ℝ) : Prop := y - 2 * x + 4 = 0
def parabola_eq (x y : ℝ) : Prop := y^2 = 3 * x + 4

def C : point := 
  let x := 4
  let y := 2 * x - 4
  (x, y)

def D : point := 
  let x := 3 / 4
  let y := 2 * x - 4
  (x, y)

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def CQ : ℝ := distance Q C
def DQ : ℝ := distance Q D

theorem distance_difference :
  |CQ - DQ| = |2 * real.sqrt 5 - real.sqrt 8.90625| :=
sorry

end distance_difference_l818_818898


namespace three_digit_number_to_four_digit_l818_818371

theorem three_digit_number_to_four_digit (a : ℕ) (h : 100 ≤ a ∧ a ≤ 999) : (10 * a + 1) = ... := sorry

end three_digit_number_to_four_digit_l818_818371


namespace base_of_frustum_parallelogram_l818_818144

theorem base_of_frustum_parallelogram (ABCD EFGH : Quadrilateral)
  (parallel_ABCD_EFGH : parallel ABCD EFGH)
  (intersecting_diagonals : ∀ AG BH CE DF, intersects AG BH ∧ intersects BH CE ∧ intersects CE DF ∧ intersects DF AG) :
  parallelogram ABCD :=
sorry

end base_of_frustum_parallelogram_l818_818144


namespace pencil_black_part_length_l818_818771

theorem pencil_black_part_length :
  ∀ (total_length purple_length blue_length : ℝ),
  total_length = 4 ∧ purple_length = 1.5 ∧ blue_length = 2 →
  (total_length - (purple_length + blue_length)) = 0.5 :=
by
  intros total_length purple_length blue_length
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  simp [h1, h3, h4]
  sorry

end pencil_black_part_length_l818_818771


namespace lines_intersect_iff_k_eq_1_l818_818935

theorem lines_intersect_iff_k_eq_1 (s t k : ℝ) :
  (∃ (s t : ℝ), ⟨1 + 2 * s, 2 - s, 3 + s * k⟩ = ⟨3 - t * k, 5 + 3 * t, 7 + 2 * t⟩) ↔ k = 1 :=
by 
  sorry

end lines_intersect_iff_k_eq_1_l818_818935


namespace right_triangle_sides_l818_818040

variable (a : ℝ) (h : a ≠ 1)

def S := (a + a⁻¹) / 2
def D := (a - a⁻¹) / 2
def P := 1

theorem right_triangle_sides :
  S a h ^ 2 = D a h ^ 2 + P ^ 2 :=
by
  sorry

end right_triangle_sides_l818_818040


namespace vertices_distance_sum_find_x_plus_y_l818_818645

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

noncomputable def sum_of_distances (P A B C : ℝ × ℝ) : ℝ :=
  distance P A + distance P B + distance P C

theorem vertices_distance_sum :
  let P := (5, 1)
  let A := (0, 0)
  let B := (12, 0)
  let C := (4, 6)
  (sum_of_distances P A B C) = 2 * real.sqrt 26 + 5 * real.sqrt 2 :=
  sorry

theorem find_x_plus_y :
  let x := 2
  let y := 5
  x + y = 7 :=
  by
  rfl

end vertices_distance_sum_find_x_plus_y_l818_818645


namespace maximum_determinant_is_15_l818_818425

open Matrix

noncomputable def largest_determinant (u v w : Vector 3 ℝ) : ℝ :=
  let cross_prod := ![
    v[1] * w[2] - v[2] * w[1],
    v[2] * w[0] - v[0] * w[2],
    v[0] * w[1] - v[1] * w[0]
  ]
  let dot_product := u[0] * cross_prod[0] + u[1] * cross_prod[1] + u[2] * cross_prod[2]
  dot_product

theorem maximum_determinant_is_15 :
  let u := ![
    1 / ( √(3:ℝ)^2 + (-2:ℝ)^2 + (2:ℝ)^2) * 3,
    1 / ( √(3:ℝ)^2 + (-2:ℝ)^2 + (2:ℝ)^2) * -2,
    1 / ( √(3:ℝ)^2 + (-2:ℝ)^2 + (2:ℝ)^2) * 2
  ]
  let v := ![3, -2, 2]
  let w := ![-1, 4, 1]
  u.norm = 1 →
  largest_determinant u v w = 15
:= by
  intros u v w hu
  rw [u, largest_determinant]
  sorry

end maximum_determinant_is_15_l818_818425


namespace part1_f_0_eq_0_part2_f_periodic_T_4_part3_f_analytic_x_in_neg1_to_1_l818_818328

open Function

-- Defining that f is an odd function
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

-- Defining the symmetry about the line x = 1
def symmetry_about_x_eq_1 (f : ℝ → ℝ) := ∀ x : ℝ, f (2 - x) = f (x)

-- Given conditions
variable {f : ℝ → ℝ}
variable (h_odd : is_odd f)
variable (h_sym : symmetry_about_x_eq_1 f)
variable (h_defined_range : ∀ x ∈ set.Ioc 0 1, f x = x)

-- Proof of the first part
theorem part1_f_0_eq_0 : f 0 = 0 :=
sorry

-- Proof of the second part
theorem part2_f_periodic_T_4 : ∃ T > 0, ∀ x, f (x + T) = f x :=
exists.intro 4 $ and.intro (by norm_num) $ λ x, sorry

-- Proof of the third part
theorem part3_f_analytic_x_in_neg1_to_1 : ∀ x ∈ set.Icc (-1 : ℝ) 1, f x = x :=
sorry

end part1_f_0_eq_0_part2_f_periodic_T_4_part3_f_analytic_x_in_neg1_to_1_l818_818328


namespace number_of_trees_l818_818250

theorem number_of_trees {N : ℕ} 
  (Anya : N = 15) 
  (Borya : N % 11 = 0) 
  (Vera : N < 25) 
  (Gena : N % 22 = 0) 
  (truth_conditions : (Anya ∨ Borya ∨ Vera ∨ Gena) ∧ ∃! p, p) : 
  N = 11 :=
sorry

end number_of_trees_l818_818250


namespace area_of_quadrilateral_EFGH_l818_818394

def quadrilateral_area (EF FG GH HE : ℝ) (angle_FGH : ℝ) : ℝ :=
if angle_FGH = 90 then 
  let FH := Real.sqrt (FG^2 + GH^2) in
  let area_FGH := 0.5 * FG * GH in
  let s := (EF + FH + HE) / 2 in
  let area_EFH := Real.sqrt (s * (s - EF) * (s - FH) * (s - HE)) in
  area_FGH + area_EFH
else 0

theorem area_of_quadrilateral_EFGH :
  quadrilateral_area 5 6 8 10 90 = 39 :=
by sorry

end area_of_quadrilateral_EFGH_l818_818394


namespace sin_of_7pi_over_6_l818_818660

theorem sin_of_7pi_over_6 : Real.sin (7 * Real.pi / 6) = -1 / 2 :=
by
  -- Conditions from the statement in a)
  -- Given conditions: \(\sin (180^\circ + \theta) = -\sin \theta\)
  -- \(\sin 30^\circ = \frac{1}{2}\)
  sorry

end sin_of_7pi_over_6_l818_818660


namespace emma_calculator_expected_value_100m_plus_n_l818_818220

-- Define the expected value function
noncomputable def E : ℕ → ℚ
| 0       := 0
| (n+1) := 9 * (E n) + 4.5

-- Define the main theorem to prove.
theorem emma_calculator_expected_value_100m_plus_n :
  (100 * 332145 + 10) = 33214510 :=
by {
  -- Using previously defined expected value function when n = 5
  have h1 : E 5 = 33214.5,
  { -- Calculations for n = 5 to show that E 5 = 33214.5
    sorry }, -- Placeholder for the proof steps
  -- Converting the expected value into the form m / n
  let m := 332145,
  let n := 10,
  -- Calculate the expected value
  have h2 : m / n = 33214.5,
  { sorry }, -- Placeholder for the verification of division
  -- Final calculation for 100m + n
  calc (100 * m + n) = 33214500 + 10 : by rw [mul_add, add_comm, mul_left_comm]
                     ... = 33214510 : by norm_num
}

end emma_calculator_expected_value_100m_plus_n_l818_818220


namespace minimum_tan_theta_is_sqrt7_l818_818626

noncomputable def min_tan_theta (z : ℂ) : ℝ := (Complex.abs (Complex.im z) / Complex.abs (Complex.re z))

theorem minimum_tan_theta_is_sqrt7 {z : ℂ} 
  (hz_real : 0 ≤ Complex.re z)
  (hz_imag : 0 ≤ Complex.im z)
  (hz_condition : Complex.abs (z^2 + 2) ≤ Complex.abs z) :
  min_tan_theta z = Real.sqrt 7 := sorry

end minimum_tan_theta_is_sqrt7_l818_818626


namespace main_proof_l818_818430

variable {m n : Type} -- These types are placeholders for lines
variable {α β : Type} -- These types are placeholders for planes

-- Propositions as hypotheses
def proposition1 (m n : Type) (α β : Type) [perpendicular m α] [parallel n β] [parallel α β] : Prop :=
perpendicular m n

def proposition2 (m n : Type) (α β : Type) [perpendicular m n] [parallel α β] [perpendicular m α] : Prop :=
parallel n β

def proposition3 (m n : Type) (α β : Type) [perpendicular m n] [parallel α β] [parallel m α] : Prop :=
perpendicular n β

def proposition4 (m n : Type) (α β : Type) [perpendicular m α] [not (parallel m n)] [parallel α β] : Prop :=
perpendicular n β

-- The main statement we need to prove
theorem main_proof
  (h1 : proposition1 m n α β)
  (h2 : ¬ (proposition2 m n α β))
  (h3 : ¬ (proposition3 m n α β))
  (h4 : proposition4 m n α β) : true :=
by trivial -- This will be replaced by the actual proof when one is added

#check main_proof -- This is to check if the Lean code builds successfully with the stated problem and definitions

end main_proof_l818_818430


namespace thirtieth_term_of_sequence_is_424_l818_818518

def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0
def contains_digit_2 (n : ℕ) : Prop := ∃ k, n.digits 10 k = 2

def sequence_term (k : ℕ) : ℕ :=
  let seq := { n : ℕ | is_multiple_of_4 n ∧ contains_digit_2 n}
  (seq.to_list k).get! k

theorem thirtieth_term_of_sequence_is_424 : sequence_term 29 = 424 := 
by sorry

end thirtieth_term_of_sequence_is_424_l818_818518


namespace petya_vasya_equal_number_l818_818018

theorem petya_vasya_equal_number :
  ∃ (a b : ℕ), (∃ (M N : ℕ), 
    (∃ (seqP : Fin 20 → ℕ), (∀ i, seqP i = a + i) ∧ M = digit_concat (seqP ∘ Fin.val)) ∧
    (∃ (seqV : Fin 21 → ℕ), (∀ i, seqV i = b + i) ∧ N = digit_concat (seqV ∘ Fin.val)) ∧
    M = N) :=
sorry

end petya_vasya_equal_number_l818_818018


namespace find_retail_price_before_discounts_l818_818766

noncomputable def wholesale_price : ℝ := 108
noncomputable def profit_rate : ℝ := 0.20
noncomputable def tax_rate : ℝ := 0.15
noncomputable def first_discount : ℝ := 0.10
noncomputable def second_discount : ℝ := 0.05
noncomputable def final_selling_price : ℝ := 132.84

theorem find_retail_price_before_discounts :
  let profit := wholesale_price * profit_rate,
      pre_tax_selling_price := wholesale_price + profit,
      tax := profit * tax_rate,
      price_after_tax := pre_tax_selling_price + tax,
      factor := (1 - first_discount) * (1 - second_discount) in
  price_after_tax / factor = 155.44 :=
by
  sorry

end find_retail_price_before_discounts_l818_818766


namespace binom_sum_l818_818653

theorem binom_sum :
  (∑ k in Finset.range 51, (-1:ℤ)^k * (Nat.choose 101 (2 * k))) = 2 ^ 50 := 
by
  sorry

end binom_sum_l818_818653


namespace correct_statement_l818_818146

theorem correct_statement :
  (Real.sqrt (9 / 16) = 3 / 4) :=
by
  sorry

end correct_statement_l818_818146


namespace least_positive_integer_with_six_factors_l818_818114

-- Define what it means for a number to have exactly six distinct positive factors
def hasExactlySixFactors (n : ℕ) : Prop :=
  (n.factorization.support.card = 2 ∧ (n.factorization.values' = [2, 1])) ∨
  (n.factorization.support.card = 1 ∧ (n.factorization.values' = [5]))

-- The main theorem statement
theorem least_positive_integer_with_six_factors : ∃ n : ℕ, hasExactlySixFactors n ∧ ∀ m : ℕ, (hasExactlySixFactors m → n ≤ m) :=
  exists.intro 12 (and.intro
    (show hasExactlySixFactors 12, by sorry)
    (show ∀ m : ℕ, hasExactlySixFactors m → 12 ≤ m, by sorry))

end least_positive_integer_with_six_factors_l818_818114


namespace lily_lemonade_calories_l818_818842

def total_weight (lemonade_lime_juice lemonade_honey lemonade_water : ℕ) : ℕ :=
  lemonade_lime_juice + lemonade_honey + lemonade_water

def total_calories (weight_lime_juice weight_honey : ℕ) : ℚ :=
  (30 * weight_lime_juice / 100) + (305 * weight_honey / 100)

def calories_in_portion (total_weight total_calories portion_weight : ℚ) : ℚ :=
  (total_calories * portion_weight) / total_weight

theorem lily_lemonade_calories :
  let lemonade_lime_juice := 150
  let lemonade_honey := 150
  let lemonade_water := 450
  let portion_weight := 300
  let total_weight := total_weight lemonade_lime_juice lemonade_honey lemonade_water
  let total_calories := total_calories lemonade_lime_juice lemonade_honey
  calories_in_portion total_weight total_calories portion_weight = 201 := 
by
  sorry

end lily_lemonade_calories_l818_818842


namespace number_of_trees_l818_818251

theorem number_of_trees {N : ℕ} 
  (Anya : N = 15) 
  (Borya : N % 11 = 0) 
  (Vera : N < 25) 
  (Gena : N % 22 = 0) 
  (truth_conditions : (Anya ∨ Borya ∨ Vera ∨ Gena) ∧ ∃! p, p) : 
  N = 11 :=
sorry

end number_of_trees_l818_818251


namespace marcella_pairs_left_l818_818153

theorem marcella_pairs_left (initial_pairs : ℕ) (lost_shoes : ℕ) :
  initial_pairs = 24 → lost_shoes = 9 → ∃ max_pairs_left, max_pairs_left = 20 :=
by
  assume h1 : initial_pairs = 24
  assume h2 : lost_shoes = 9
  have h3 : initial_pairs * 2 - lost_shoes = 48 - 9 := sorry
  have h4 : 48 - 9 = 39 := sorry
  have h5 : 20 ≤ 24 := sorry
  have max_pairs_left := 20
  existsi max_pairs_left
  trivial

end marcella_pairs_left_l818_818153


namespace three_7_faced_dice_sum_18_prob_l818_818781

theorem three_7_faced_dice_sum_18_prob :
  let probability := 1 / 7 ^ 3
  in probability * 4 = 4 / 343 :=
by
  sorry

end three_7_faced_dice_sum_18_prob_l818_818781


namespace budget_spent_on_research_and_development_l818_818993

theorem budget_spent_on_research_and_development:
  (∀ budget_total : ℝ, budget_total > 0) →
  (∀ transportation : ℝ, transportation = 15) →
  (∃ research_and_development : ℝ, research_and_development ≥ 0) →
  (∀ utilities : ℝ, utilities = 5) →
  (∀ equipment : ℝ, equipment = 4) →
  (∀ supplies : ℝ, supplies = 2) →
  (∀ salaries_degrees : ℝ, salaries_degrees = 234) →
  (∀ total_degrees : ℝ, total_degrees = 360) →
  (∀ percentage_salaries : ℝ, percentage_salaries = (salaries_degrees / total_degrees) * 100) →
  (∀ known_percentages : ℝ, known_percentages = transportation + utilities + equipment + supplies + percentage_salaries) →
  (∀ rnd_percent : ℝ, rnd_percent = 100 - known_percentages) →
  (rnd_percent = 9) :=
  sorry

end budget_spent_on_research_and_development_l818_818993


namespace number_of_trees_is_eleven_l818_818256

variables (N : ℕ)

-- Conditions
def Anya (N : ℕ) := N = 15
def Borya (N : ℕ) := 11 ∣ N
def Vera (N : ℕ) := N < 25
def Gena (N : ℕ) := 22 ∣ N

axiom OneBoyOneGirlTruth :
  (∃ (b : Prop) (g : Prop),
    (b ∨ ¬ b) ∧ (g ∨ ¬ g) ∧
    ((b = (Borya N ∨ Gena N)) ∧ (g = (Anya N ∨ Vera N)) ∧
     (b ↔ ¬g) ∧
     ((Anya N ∨ ¬Vera N) ∨ (¬Anya N ∨ Vera N)) ∧
     (Anya N = (N = 15)) ∧
     (Borya N = (11 ∣ N)) ∧
     (Vera N = (N < 25)) ∧
     (Gena N = (22 ∣ N))))

theorem number_of_trees_is_eleven: N = 11 :=
sorry

end number_of_trees_is_eleven_l818_818256


namespace remaining_dogs_after_adoptions_l818_818070

theorem remaining_dogs_after_adoptions 
  (initial_dogs : ℕ)
  (additional_dogs : ℕ)
  (adopted_week1 : ℕ)
  (adopted_week4 : ℕ) :
  initial_dogs = 200 →
  additional_dogs = 100 →
  adopted_week1 = 40 →
  adopted_week4 = 60 →
  initial_dogs + additional_dogs - adopted_week1 - adopted_week4 = 200 :=
by
  intros h_init h_add h_adopt1 h_adopt2
  rw [h_init, h_add, h_adopt1, h_adopt2]
  rfl

end remaining_dogs_after_adoptions_l818_818070


namespace locus_of_points_l818_818319

-- Definitions of the concepts used in problem
variables {A B C P : Type}
variables [acute_triangle : ∀ {A B C : Type}, Prop] -- representing acute-angled triangle
variables [circumradius_ABP : ∀ {A B P : Type}, ℝ] -- radius of circumcircle of triangle ABP
variables [circumradius_BCP : ∀ {B C P : Type}, ℝ] -- radius of circumcircle of triangle BCP
variables [circumradius_CAP : ∀ {C A P : Type}, ℝ] -- radius of circumcircle of triangle CAP
variables [circumcircle_ABC : ∀ {A B C : Type}, Set P] -- points on circumcircle of triangle ABC
variables [orthocenter_ABC : ∀ {A B C : Type}, P] -- orthocenter of triangle ABC

-- Statement of the proof problem in Lean 4
theorem locus_of_points (ABC_acute : acute_triangle A B C)
  (equal_circumradii : circumradius_ABP A B P = circumradius_BCP B C P ∧ circumradius_BCP B C P = circumradius_CAP C A P) :
  (P ∈ circumcircle_ABC A B C ∧ P ≠ A ∧ P ≠ B ∧ P ≠ C) ∨ P = orthocenter_ABC A B C :=
sorry

end locus_of_points_l818_818319


namespace ms_rapid_correct_speed_l818_818850

theorem ms_rapid_correct_speed :
  ∃ (r : ℝ), 
    (∀ (d t : ℝ), 
      (d = 30 * (t + 1/6)) ∧ (d = 50 * (t - 1/12)) → 
      r = d / t) ∧ 
    r = 41 :=
begin
  sorry
end

end ms_rapid_correct_speed_l818_818850


namespace least_pos_int_with_six_factors_l818_818103

theorem least_pos_int_with_six_factors :
  ∃ n : ℕ, (∀ m : ℕ, (number_of_factors m = 6 → m ≥ n)) ∧ n = 12 := 
sorry

end least_pos_int_with_six_factors_l818_818103


namespace approximate_fish_count_l818_818152

theorem approximate_fish_count (total_caught1 : ℕ) (tagged1 : ℕ) (total_caught2 : ℕ) (tagged2 : ℕ) 
  (h1 : total_caught1 = 50) (h2 : tagged1 = 50) (h3 : total_caught2 = 50) (h4 : tagged2 = 8) :
  let N := tagged1 * total_caught2 / tagged2 in N ≈ 313 :=
by {
  simp only [h1, h2, h3, h4],
  let N := 50 * 50 / 8,
  have : N = 312.5, sorry,
  norm_num this,
  linarith,
}

end approximate_fish_count_l818_818152


namespace valid_two_digit_numbers_l818_818214

def is_valid_two_digit_number_pair (a b : ℕ) : Prop :=
  0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ a > b ∧ (Nat.gcd (10 * a + b) (10 * b + a) = a^2 - b^2)

theorem valid_two_digit_numbers :
  (is_valid_two_digit_number_pair 2 1 ∨ is_valid_two_digit_number_pair 5 4) ∧
  ∀ a b, is_valid_two_digit_number_pair a b → (a = 2 ∧ b = 1 ∨ a = 5 ∧ b = 4) :=
by
  sorry

end valid_two_digit_numbers_l818_818214


namespace largest_prime_divisor_in_range_l818_818922

theorem largest_prime_divisor_in_range (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1100) :
  ∃ p, nat.prime p ∧ p ≤ nat.floor (real.sqrt 1100) ∧ (∀ q, q ≤ nat.floor (real.sqrt 1100) → q.prime → q ≤ p) :=
begin
  -- Here would go the proof.
  -- We know that the floor of sqrt(1100) is 33,
  -- and we need to show that 31 is the largest prime less than or equal to 33.
  -- The proof would identify 31 as the largest prime below or equal to 33, but this logic 
  -- is not necessary to detail here.
  sorry
end

end largest_prime_divisor_in_range_l818_818922


namespace sum_proper_divisors_512_l818_818956

theorem sum_proper_divisors_512 : ∑ i in {1, 2, 4, 8, 16, 32, 64, 128, 256}, i = 511 :=
by
  have h : {1, 2, 4, 8, 16, 32, 64, 128, 256} = finset.range 9.image (λ n, 2^n) := sorry
  rw [h]
  sorry

end sum_proper_divisors_512_l818_818956


namespace monotonic_increasing_interval_l818_818504

def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem monotonic_increasing_interval : ∀ x : ℝ, x > 2 → ∃ ε > 0, ∀ y : ℝ, y ∈ (x - ε, x + ε) ∧ y > 2 → f y > f x :=
by
  sorry

end monotonic_increasing_interval_l818_818504


namespace sequence_non_periodic_l818_818596

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def sequence (k : ℕ) : ℕ :=
  if sum_of_digits k % 2 = 0 then 0 else 1

theorem sequence_non_periodic : ¬ ∃ (d : ℕ), d > 0 ∧ ∀ (k : ℕ), sequence (k) = sequence (k + d) :=
sorry

end sequence_non_periodic_l818_818596


namespace number_of_fir_trees_l818_818289

def anya_statement (N : ℕ) : Prop := N = 15
def borya_statement (N : ℕ) : Prop := 11 ∣ N
def vera_statement (N : ℕ) : Prop := N < 25
def gena_statement (N : ℕ) : Prop := 22 ∣ N

def one_boy_one_girl_truth (A B G V : Prop) : Prop :=
  (A ∨ V) ∧ ¬(A ∧ V) ∧ (B ∨ G) ∧ ¬(B ∧ G)

theorem number_of_fir_trees (N : ℕ) :
  anya_statement N ∨ borya_statement N ∨ vera_statement N ∨ gena_statement N ∧
  one_boy_one_girl_truth (anya_statement N) (borya_statement N) (gena_statement N) (vera_statement N) :=
  N = 11 :=
sorry

end number_of_fir_trees_l818_818289


namespace circumcircle_PQR_contains_fixed_point_l818_818709

open EuclideanGeometry

variable (A B C D E F P Q R O : Point)
variables [IsConvexQuadrilateral A B C D]
variables [LineIntersects (LineThrough A C) (LineThrough B D) P]
variables [LineIntersects (LineThrough B D) (LineThrough E F) Q]
variables [LineIntersects (LineThrough E F) (LineThrough A C) R]
variables (hBC_AD : Distance B C = Distance A D)
variables (hNotParallel_BC_AD : ¬ Parallel (LineThrough B C) (LineThrough A D))
variables (hBE_DF : Distance B E = Distance D F)

theorem circumcircle_PQR_contains_fixed_point :
  ∃ O : Point, ∀ E F : Point, LineIntersects (LineThrough B C) E ∧ LineIntersects (LineThrough A D) F ∧
  (Distance B E = Distance D F) → O ≠ P ∧
  OnCircumcircle O P Q R := by
sorry

end circumcircle_PQR_contains_fixed_point_l818_818709


namespace number_of_trees_l818_818248

theorem number_of_trees {N : ℕ} 
  (Anya : N = 15) 
  (Borya : N % 11 = 0) 
  (Vera : N < 25) 
  (Gena : N % 22 = 0) 
  (truth_conditions : (Anya ∨ Borya ∨ Vera ∨ Gena) ∧ ∃! p, p) : 
  N = 11 :=
sorry

end number_of_trees_l818_818248


namespace min_value_OP_squared_plus_PF_squared_l818_818776

theorem min_value_OP_squared_plus_PF_squared :
  let O := (0, 0)
  let F := (-1, 0)
  ∃ P : ℝ × ℝ, (P.1^2 / 2 + P.2^2 = 1) ∧ (|O - P|^2 + |P - F|^2 = 2) :=
by
  let P := (x, y)
  sorry

end min_value_OP_squared_plus_PF_squared_l818_818776


namespace exists_point_X_on_circle_l818_818719

variables {α : Type*} [circle α]

-- Define the points on the circle and the chords
variables (A B C D J : α)
variable (X : α)

-- The given conditions
variable (non_intersecting_chords : ¬is_intersecting A B C D)
variable (J_on_CD : is_point_on_chord J C D)
variable (is_point_on_circle : is_point_on_circle X)

-- The property to prove
theorem exists_point_X_on_circle 
  (h1 : non_intersecting_chords) 
  (h2 : J_on_CD) 
  (hx : is_point_on_circle X) : 
  exists (X : α), is_point_on_circle X ∧ 
    ∃ E F, is_intersection_of_chords E F AX BX CD ∧ midpoint J E F :=
sorry

end exists_point_X_on_circle_l818_818719


namespace Delaney_missed_bus_by_l818_818636

def time_in_minutes (hours : ℕ) (minutes : ℕ) : ℕ := hours * 60 + minutes

def Delaney_start_time : ℕ := time_in_minutes 7 50
def bus_departure_time : ℕ := time_in_minutes 8 0
def travel_duration : ℕ := 30

theorem Delaney_missed_bus_by :
  Delaney_start_time + travel_duration - bus_departure_time = 20 :=
by
  sorry

end Delaney_missed_bus_by_l818_818636


namespace range_of_a_l818_818345

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin (π / 6 * x) - 2 * a + 3

noncomputable def g (x : ℝ) : ℝ := (2 * x) / (x^2 + x + 2)

theorem range_of_a (a : ℝ) (x1 x2 : ℝ) (hx1 : 0 ≤ x1 ∧ x1 ≤ 1) (hx2 : 0 ≤ x2 ∧ x2 ≤ 1)
  (h : f a x1 = g x2) : a ∈ Set.Icc (5 / 4 : ℝ) 2 :=
sorry

end range_of_a_l818_818345


namespace total_travel_time_is_7_hours_l818_818411

-- Definitions for the conditions
def distance_to_New_York : ℝ := 300 -- in km
def speed : ℝ := 50 -- in km/h
def rest_stop_duration : ℝ := 0.5 -- in hours
def rest_interval : ℝ := 2 -- in hours

-- Theorem to prove the total travel time
theorem total_travel_time_is_7_hours :
  let driving_time := distance_to_New_York / speed,
      number_of_rest_stops := (driving_time / rest_interval) - 1,
      total_rest_time := number_of_rest_stops * rest_stop_duration,
      total_travel_time := driving_time + total_rest_time
  in total_travel_time = 7 := by
  sorry

end total_travel_time_is_7_hours_l818_818411


namespace lcm_36_105_l818_818679

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end lcm_36_105_l818_818679


namespace smallest_n_for_common_factor_l818_818537

theorem smallest_n_for_common_factor : ∃ n : ℕ, n > 0 ∧ (Nat.gcd (11 * n - 3) (8 * n + 4) > 1) ∧ n = 42 := 
by
  sorry

end smallest_n_for_common_factor_l818_818537


namespace probability_crossing_river_l818_818149

noncomputable def probability_of_crossing (jump_distance river_width : ℝ) : ℝ :=
  let successful_region := (4 - 2) in
  let total_region := river_width in
  successful_region / total_region

theorem probability_crossing_river :
  probability_of_crossing 4 6 = 1 / 3 :=
by
  unfold probability_of_crossing
  sorry

end probability_crossing_river_l818_818149


namespace people_after_five_years_l818_818169

-- Initial number of people
def a0 : ℕ := 15

-- Recurrence relation for the total number of people each year
-- a_{k+1} = 3a_k - 10

noncomputable def a : ℕ → ℕ
| 0     := a0
| (n+1) := 3 * a n - 10

-- Function to calculate the total number of people after 5 years
def people_in_five_years : ℕ := a 5

-- Theorem to state that the total number of people after 5 years is 2435
theorem people_after_five_years : people_in_five_years = 2435 := by
  sorry

end people_after_five_years_l818_818169


namespace find_k_l818_818964

theorem find_k {k : ℝ} (h : (∃ α β : ℝ, α ≠ 0 ∧ β ≠ 0 ∧ α / β = 3 / 1 ∧ α + β = -10 ∧ α * β = k)) : k = 18.75 :=
sorry

end find_k_l818_818964


namespace set_Y_definition_l818_818474

-- Define what it means to be a two-digit prime number
def is_two_digit_prime (n : ℕ) : Prop :=
  nat.prime n ∧ 10 < n ∧ n < 100

-- Define Set X as the set of all two-digit prime numbers
def set_X : set ℕ := {n | is_two_digit_prime n}

-- Define the range of a set
def range (S : set ℕ) : ℕ :=
  let min := Inf S in
  let max := Sup S in
  max - min

-- Definition of set Y
def set_Y (Y : set ℕ) : Prop :=
  (∀ y ∈ Y, y > 97) ∧ range (set_X ∪ Y) = 90

-- The theorem that needs to be proven
theorem set_Y_definition (Y : set ℕ) (hY : set_Y Y) : Inf Y = 101 :=
by sorry

end set_Y_definition_l818_818474


namespace problem_statement_l818_818431

noncomputable def orthocenter (A B C : Point) : Point := sorry
noncomputable def incenter (A B C : Point) : Point := sorry
noncomputable def circumcircle (A B C : Point) : Circle := sorry
noncomputable def intersection (c : Circle) (A B : Point) (P : Point) : Point := sorry
noncomputable def projection (H : Point) (AI : Line) : Point := sorry
noncomputable def reflection (P K : Point) : Point := sorry
def collinear (B H Q : Point) : Prop := sorry

theorem problem_statement (A B C H I P K Q : Point) (circBCI : Circle) :
  H = orthocenter A B C →
  I = incenter A B C →
  circBCI = circumcircle B C I →
  P = intersection circBCI A B B → -- P is the intersection different from B
  K = projection H (line_through A I) →
  Q = reflection P K →
  collinear B H Q :=
by
  sorry

end problem_statement_l818_818431


namespace solve_trig_eq_l818_818877

open Real

theorem solve_trig_eq (k : ℤ) : 
  (∃ x : ℝ, 
    (|cos x| + cos (3 * x)) / (sin x * cos (2 * x)) = -2 * sqrt 3 
    ∧ (x = -π/6 + 2 * k * π ∨ x = 2 * π/3 + 2 * k * π ∨ x = 7 * π/6 + 2 * k * π)) :=
sorry

end solve_trig_eq_l818_818877


namespace max_odd_sums_of_three_consecutive_l818_818611

theorem max_odd_sums_of_three_consecutive (nums : List ℕ) (h₁ : ∀ n ∈ nums, 1000 ≤ n ∧ n ≤ 1997) (h₂ : nums.length = 998):
  (∃ arrangement : List ℕ, ∀ sums : List ℕ, (∀ i ∈ (List.range (nums.length - 2)), sums[i] = arrangement[i] + arrangement[i+1] + arrangement[i+2]) → 
    (sums.filter (λ x, x % 2 = 1)).length ≤ 499) :=
sorry

end max_odd_sums_of_three_consecutive_l818_818611


namespace complement_is_correct_l818_818756

def U : Set ℝ := {x | x^2 ≤ 4}
def A : Set ℝ := {x | abs (x + 1) ≤ 1}
def complement_U_A : Set ℝ := U \ A

theorem complement_is_correct :
  complement_U_A = {x | 0 < x ∧ x ≤ 2} := by
  sorry

end complement_is_correct_l818_818756


namespace correct_option_C_l818_818711

variables {Point : Type*} [EuclideanGeometry Point]
open EuclideanGeometry

-- Definitions to avoid using solutions directly
variable (m : Line Point)
variable (α β : Plane Point)
variable (h_noncoincident : ¬Parallel α β)

-- To ensure lean can process the condition that m is a line and α, β are planes
noncomputable theory

-- Stating the given conditions and required to prove α ⊥ β under said conditions.
theorem correct_option_C (m_perp_α : PerpendicularLinePlane m α) (m_parallel_β : ParallelLinePlane m β) : PerpendicularPlanePlane α β :=
by
  sorry

end correct_option_C_l818_818711


namespace danil_wins_l818_818858

-- Define the board size
def board_size : ℕ := 2017

-- Define the initial position of the bishop
def initial_position : ℕ × ℕ := (1, 1)

-- Define a move as a pair of positions (from, to)
structure Move :=
(from : ℕ × ℕ)
(to : ℕ × ℕ)

-- Check if a move is diagonal and within board boundaries
def valid_move (move : Move) : Prop :=
  (move.to.1 <= board_size ∧ move.to.2 <= board_size) ∧
  (move.to.1 > 0 ∧ move.to.2 > 0) ∧
  abs(move.to.1 - move.from.1) = abs(move.to.2 - move.from.2)

-- Check if a position is on the main diagonal
def on_main_diagonal (pos : ℕ × ℕ) : Prop :=
  pos.1 = pos.2

-- Define that Danil can guarantee a win from the initial position
theorem danil_wins :
  ∃ strategy : list Move, ∀ move : Move, valid_move move ∧
  (move.from = initial_position ∨ move.to = initial_position) →
  ¬ on_main_diagonal move.to ∨
  (∃ new_move : Move, valid_move new_move ∧
   (new_move.from = move.to ∧ on_main_diagonal new_move.to)) :=
sorry

end danil_wins_l818_818858


namespace factor_polynomial_l818_818655

theorem factor_polynomial (x : ℝ) : 
  54 * x ^ 5 - 135 * x ^ 9 = 27 * x ^ 5 * (2 - 5 * x ^ 4) :=
by 
  sorry

end factor_polynomial_l818_818655


namespace least_positive_integer_with_six_factors_l818_818119

-- Define what it means for a number to have exactly six distinct positive factors
def hasExactlySixFactors (n : ℕ) : Prop :=
  (n.factorization.support.card = 2 ∧ (n.factorization.values' = [2, 1])) ∨
  (n.factorization.support.card = 1 ∧ (n.factorization.values' = [5]))

-- The main theorem statement
theorem least_positive_integer_with_six_factors : ∃ n : ℕ, hasExactlySixFactors n ∧ ∀ m : ℕ, (hasExactlySixFactors m → n ≤ m) :=
  exists.intro 12 (and.intro
    (show hasExactlySixFactors 12, by sorry)
    (show ∀ m : ℕ, hasExactlySixFactors m → 12 ≤ m, by sorry))

end least_positive_integer_with_six_factors_l818_818119


namespace intersection_on_circumcircle_l818_818594

-- Define the setting of our problem
constant Point : Type
constant Line : Type
constant Circle : Type

-- Definitions related to the rotation, points, and lines
constant O : Point
constant A₁ A₂ P : Point
constant l₁ l₂ : Line
constant rotation_center_O : Line → Line → Point → Point → Prop

-- The intersection point of lines l₁ and l₂
constant intersection : Line → Line → Point
axiom intersection_l₁_l₂ : intersection l₁ l₂ = P

-- Assume rotation center O maps:
axiom rotation_maps_lines : rotation_center_O l₁ l₂ A₁ A₂

-- Definition of the circumcircle of a triangle
constant circumcircle (A B C : Point) : Circle
constant lies_on (P : Point) (C : Circle) : Prop

-- The theorem to be proved
theorem intersection_on_circumcircle
  (O A₁ A₂ P : Point) (l₁ l₂ : Line)
  (rotation_center_O : Line → Line → Point → Point → Prop)
  (intersection : Line → Line → Point)
  (inter_l₁_l₂ : intersection l₁ l₂ = P)
  (rot_maps : rotation_center_O l₁ l₂ A₁ A₂) :
  lies_on P (circumcircle O A₁ A₂) :=
sorry

end intersection_on_circumcircle_l818_818594


namespace value_of_x_y_l818_818721

theorem value_of_x_y (x y : ℝ) (h : x + 1 + y * complex.I = -complex.I + 2 * x) : x^y = 1 :=
by
  sorry

end value_of_x_y_l818_818721


namespace max_sum_of_abc_l818_818807

theorem max_sum_of_abc (A B C : ℕ) (h₁ : A ≠ B) (h₂ : B ≠ C) (h₃ : A ≠ C) (h₄ : A * B * C = 2310) : 
  A + B + C ≤ 52 :=
sorry

end max_sum_of_abc_l818_818807


namespace g_zero_g_one_l818_818177

variable (g : ℤ → ℤ)

axiom condition1 (x : ℤ) : g (x + 5) - g x = 10 * x + 30
axiom condition2 (x : ℤ) : g (x^2 - 2) = (g x - x)^2 + x^2 - 4

theorem g_zero_g_one : (g 0, g 1) = (-4, 1) := 
by 
  sorry

end g_zero_g_one_l818_818177


namespace least_positive_integer_with_six_distinct_factors_l818_818097

theorem least_positive_integer_with_six_distinct_factors : ∃ n : ℕ, (∀ k : ℕ, (number_of_factors k = 6) → (n ≤ k)) ∧ (number_of_factors n = 6) ∧ (n = 12) :=
by
  sorry

end least_positive_integer_with_six_distinct_factors_l818_818097


namespace stone_shadow_problem_l818_818856

theorem stone_shadow_problem :
  (∀ (height_stick shadow_stick height_stone shadow_stone : ℝ), 
    height_stick = 1 ∧ shadow_stick = 4 ∧ shadow_stone = 20 ∧ shadow_stone ≠ 0 →
    ∃ (r : ℝ) (θ : ℝ), r = 5 ∧ θ = arctan (height_stick / shadow_stick) ∧ θ = arctan (1 / 4)) :=
begin
  sorry,
end

end stone_shadow_problem_l818_818856


namespace sum_of_integer_solutions_l818_818616

theorem sum_of_integer_solutions :
  (∑ x in Finset.filter (λ x : ℤ, 4 < (x - 3)^2 ∧ (x - 3)^2 < 36) (Finset.Icc (-100) 100), x) = 18 :=
sorry

end sum_of_integer_solutions_l818_818616


namespace area_of_shaded_region_l818_818622

noncomputable def area_shaded (side : ℝ) : ℝ :=
  let area_square := side * side
  let radius := side / 2
  let area_circle := Real.pi * radius * radius
  area_square - area_circle

theorem area_of_shaded_region :
  let perimeter := 28
  let side := perimeter / 4
  area_shaded side = 49 - π * 12.25 :=
by
  sorry

end area_of_shaded_region_l818_818622


namespace small_hotdogs_count_l818_818580

-- Definitions from the conditions
def num_large_hotdogs : ℕ := 21
def total_hotdogs : ℕ := 79

-- Formulate the problem as a Lean statement
theorem small_hotdogs_count : 
  let num_small_hotdogs := total_hotdogs - num_large_hotdogs in
  num_small_hotdogs = 58 :=
by
  let num_small_hotdogs := total_hotdogs - num_large_hotdogs
  show num_small_hotdogs = 58
  sorry

end small_hotdogs_count_l818_818580


namespace intersection_volume_l818_818143

open Set

-- Define the region constraints
def region1 (x y z : ℝ) : Prop := |x| + |y| + z ≤ 2
def region2 (x y z : ℝ) : Prop := |x| + |y| + |z - 2| ≤ 2

-- Define the region of interest as the intersection of region1 and region2
def region := {p : ℝ × ℝ × ℝ | region1 p.1 p.2 p.3 ∧ region2 p.1 p.2 p.3}

-- Prove the volume of the intersection region is 1/6
theorem intersection_volume : measure_theory.measure.volume (region) = 1 / 6 := 
by {
  sorry
}

end intersection_volume_l818_818143


namespace total_boys_candies_invariant_l818_818075

-- Define initial count of candies
def initial_candies : ℕ := 2021

-- Define the division with rounding behavior
def candies_taken_by_child (n : ℕ) (k : ℕ) (is_boy : Bool) : ℕ :=
  if is_boy then (n + (k - 1)) / k else n / k

-- Define the function to compute total candies taken by boys
def total_candies_taken_by_boys (sequence : List Bool) (n : ℕ) : ℕ :=
  let rec calc (seq : List Bool) (remaining : ℕ) (boys_taken : ℕ) : ℕ :=
    match seq with
    | [] => boys_taken
    | h :: t =>
      let k := seq.length
      let taken := candies_taken_by_child remaining k h
      calc t (remaining - taken) (if h then boys_taken + taken else boys_taken)
  calc sequence n 0

theorem total_boys_candies_invariant (children : List Bool) :
  ∀ perm : List Bool, perm.perm children → total_candies_taken_by_boys children initial_candies = total_candies_taken_by_boys perm initial_candies :=
  sorry

end total_boys_candies_invariant_l818_818075


namespace modulo_power_l818_818829

theorem modulo_power (a n : ℕ) (p : ℕ) (hn_pos : 0 < n) (hp_odd : p % 2 = 1)
  (hp_prime : Nat.Prime p) (h : a^p ≡ 1 [MOD p^n]) : a ≡ 1 [MOD p^(n-1)] :=
by
  sorry

end modulo_power_l818_818829


namespace function_has_two_zeros_l818_818744

def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 6^x - m else x^2 - 3 * m * x + 2 * m^2

theorem function_has_two_zeros (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f m x1 = 0 ∧ f m x2 = 0) ↔ m ∈ set.Ico (1/2 : ℝ) 1 ∪ set.Ici 6 :=
sorry

end function_has_two_zeros_l818_818744


namespace correct_solutions_l818_818468

theorem correct_solutions (x y z t : ℕ) : 
  (x^2 + t^2) * (z^2 + y^2) = 50 → 
  (x = 1 ∧ y = 1 ∧ z = 2 ∧ t = 3) ∨ 
  (x = 3 ∧ y = 2 ∧ z = 1 ∧ t = 1) ∨ 
  (x = 4 ∧ y = 1 ∧ z = 3 ∧ t = 1) ∨ 
  (x = 1 ∧ y = 3 ∧ z = 4 ∧ t = 1) :=
sorry

end correct_solutions_l818_818468


namespace tetrahedral_probability_l818_818454

open Classical

variable (Ω : Type) (P : set Ω → ℝ)

def regular_tetrahedral_outcome_space : set (ℕ × ℕ) :=
  { (1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (2,4),
    (3,1), (3,2), (3,3), (3,4), (4,1), (4,2), (4,3), (4,4) }

def event_A : set (ℕ × ℕ) :=
  { (1,1), (1,3), (2,2), (2,4), (3,1), (3,3), (4,2), (4,4) }

def event_B : set (ℕ × ℕ) :=
  { (2,1), (2,2), (2,3), (2,4), (4,1), (4,2), (4,3), (4,4) }

def event_C : set (ℕ × ℕ) :=
  { (1,2), (1,4), (2,2), (2,4), (3,2), (3,4), (4,2), (4,4) }

theorem tetrahedral_probability :
  P (event_A) = 1/2 ∧ P (event_B) = 1/2 ∧ P (event_C) = 1/2 ∧
  P (event_A ∩ event_B) = 1/4 ∧ P (event_B ∩ event_C) = 1/4 ∧ P (event_A ∩ event_C) = 1/4 ∧ 
  P (event_A ∩ event_B ∩ event_C) = 1/4 :=
sorry

end tetrahedral_probability_l818_818454


namespace power_of_11_l818_818695

theorem power_of_11 (x : ℕ) :
  let total_prime_factors := 22 + 7 + x in
  total_prime_factors = 31 → x = 2 :=
by
  intro h
  sorry

end power_of_11_l818_818695


namespace slope_of_perpendicular_line_l818_818213

-- Define the given condition, which is the line equation
def line_eq (x y : ℝ) : Prop := 5 * x - 2 * y = 10

-- Define the slope of a line perpendicular to the line_eq
def perp_slope (m : ℝ) : ℝ := -1 / m

-- The slope of the given line
noncomputable def slope_of_line : ℝ := 5 / 2

-- The theorem we want to prove
theorem slope_of_perpendicular_line : perp_slope slope_of_line = -2 / 5 :=
by
  sorry

end slope_of_perpendicular_line_l818_818213


namespace miles_left_to_drive_l818_818159

theorem miles_left_to_drive 
  (total_distance : ℕ) 
  (distance_covered : ℕ) 
  (remaining_distance : ℕ) 
  (h1 : total_distance = 78) 
  (h2 : distance_covered = 32) 
  : remaining_distance = total_distance - distance_covered -> remaining_distance = 46 :=
by
  sorry

end miles_left_to_drive_l818_818159


namespace ivan_years_l818_818041

theorem ivan_years (years months weeks days hours : ℕ) (h1 : years = 48) (h2 : months = 48)
    (h3 : weeks = 48) (h4 : days = 48) (h5 : hours = 48) :
    (53 : ℕ) = (years + (months / 12) + ((weeks * 7 + days) / 365) + ((hours / 24) / 365)) := by
  sorry

end ivan_years_l818_818041


namespace solve_for_number_l818_818079

def thirty_percent_less_than_ninety : ℝ := 0.7 * 90
def one_fourth_more_than (n : ℝ) : ℝ := (5 / 4) * n

theorem solve_for_number :
  ∃ n : ℝ, one_fourth_more_than n = thirty_percent_less_than_ninety ∧ n = 50 :=
by
  sorry

end solve_for_number_l818_818079


namespace smallest_six_factors_l818_818121

theorem smallest_six_factors (n : ℕ) (h : (n = 2 * 3^2)) : n = 18 :=
by {
    sorry -- proof goes here
}

end smallest_six_factors_l818_818121


namespace find_A_find_tan_C_l818_818723

-- Part (I)
theorem find_A (A : ℝ) (h1 : √3 * Real.sin A - Real.cos A = 1) (h2 : 0 < A) (h3 : A < Real.pi) :
  A = Real.pi / 3 := 
sorry

-- Part (II)
theorem find_tan_C (B : ℝ) (C : ℝ)
  (h1 : (1 + Real.sin (2 * B)) / (Real.cos (B) ^ 2 - Real.sin (B) ^ 2) = -3)
  (h2 : B ≠ 0)
  (h3 : A = Real.pi / 3) :
  Real.tan C = (8 + 5 * √3) / 11 := 
sorry

end find_A_find_tan_C_l818_818723


namespace even_func_shift_left_symm_even_func_shift_right_symm_l818_818727

-- Definition of an even function 
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Problem 1: If f is an even function, prove the graph of f(x+2) is symmetric about x = -2
theorem even_func_shift_left_symm {f : ℝ → ℝ} (h : is_even_function f) : 
  ∀ x : ℝ, f (x + 2) = f (-(x + 2) - 4) :=
by
  sorry 

-- Problem 2: If f(x+2) is an even function, prove the graph of f is symmetric about x = 2
theorem even_func_shift_right_symm {f : ℝ → ℝ} (h : is_even_function (λ x, f (x + 2))) : 
  ∀ x : ℝ, f x = f (-(x - 2) + 4) :=
by
  sorry

end even_func_shift_left_symm_even_func_shift_right_symm_l818_818727


namespace angle_D_measure_l818_818310

theorem angle_D_measure (A B C D : ℝ) (hA : A = 50) (hB : B = 35) (hC : C = 35) :
  D = 120 :=
  sorry

end angle_D_measure_l818_818310


namespace Farrah_total_match_sticks_l818_818224

def boxes := 4
def matchboxes_per_box := 20
def sticks_per_matchbox := 300

def total_matchboxes : Nat :=
  boxes * matchboxes_per_box

def total_match_sticks : Nat :=
  total_matchboxes * sticks_per_matchbox

theorem Farrah_total_match_sticks : total_match_sticks = 24000 := sorry

end Farrah_total_match_sticks_l818_818224


namespace angle_Z_is_120_l818_818438

-- Define angles and lines
variables {p q : Prop} {X Y Z : ℝ}
variables (h_parallel : p ∧ q)
variables (hX : X = 100)
variables (hY : Y = 140)

-- Proof statement: Given the angles X and Y, we prove that angle Z is 120 degrees.
theorem angle_Z_is_120 (h_parallel : p ∧ q) (hX : X = 100) (hY : Y = 140) : Z = 120 := by 
  -- Here we would add the proof steps
  sorry

end angle_Z_is_120_l818_818438


namespace exists_path_with_m_plus_1_vertices_l818_818361

-- Define a graph structure
structure Graph :=
  (V : Type) -- Vertex type
  (E : V → V → Prop) -- Edge relation

-- Define the conditions
variables {G : Graph} {n m : ℕ}

-- Condition: The graph G has n vertices
def vertices_count (G : Graph) : ℕ := sorry

-- Condition: The graph G has mn edges
def edges_count (G : Graph) : ℕ := sorry

-- Condition: n > 2m
def n_gt_2m (n m : ℕ) : Prop := n > 2 * m

-- Define what it means to have a path with k vertices
def path_with_k_vertices (G : Graph) (k : ℕ) : Prop := sorry

theorem exists_path_with_m_plus_1_vertices 
  (hV : vertices_count G = n) 
  (hE : edges_count G = m * n) 
  (hnm : n_gt_2m n m) :
  ∃ (p : ℕ), path_with_k_vertices G (m + 1) :=
begin
  sorry
end

end exists_path_with_m_plus_1_vertices_l818_818361


namespace find_solutions_l818_818232

noncomputable def is_solution (x : ℝ) : Prop :=
  ((x - 1) * (x - 3) * (x - 5) * (x - 7) * (x - 5) * (x - 3) * (x - 1)) / 
  ((x - 3) * (x - 5) * (x - 3)) = 1

theorem find_solutions : 
  ∀ x : ℝ, is_solution x ↔ (x ≠ 3) ∧ (x ≠ 5) ∧ (x = 4 ∨ x = 4 + 2 * real.sqrt 10 ∨ x = 4 - 2 * real.sqrt 10) :=
by
  sorry

end find_solutions_l818_818232


namespace minimum_cumulative_score_of_new_champion_l818_818563

theorem minimum_cumulative_score_of_new_champion :
  ∃ (scores : Fin 10 → ℕ), 
  (⟨scores 0, scores 1, scores 2, scores 3, scores 4, scores 5, scores 6, scores 7, scores 8, scores 9⟩ = ⟨9, 8, 7, 6, 5, 4, 3, 2, 1, 0⟩) ∧
  (∀ i j : Fin 10, i ≠ j → 
    ∃ (points : Fin 10 → ℕ), 
    (points i + points j = 1 ∨ points i + points j = 2) ∧
    (∃ (cumulative_scores : Fin 10 → ℕ),
    (cumulative_scores = λ k, scores k + points k) ∧
    (∃ max_score, 
    max_score = List.maximum (List.map (λ k, cumulative_scores k) (List.finRange 10)) ∧
    max_score = 12))) :=
sorry

end minimum_cumulative_score_of_new_champion_l818_818563


namespace parallelogram_base_length_l818_818233

theorem parallelogram_base_length :
  ∀ (A H : ℝ), (A = 480) → (H = 15) → (A = Base * H) → (Base = 32) := 
by 
  intros A H hA hH hArea 
  sorry

end parallelogram_base_length_l818_818233


namespace student_A_more_consistent_l818_818186

def average (scores : List ℝ) : ℝ :=
  scores.sum / scores.length

def max_deviation (scores : List ℝ) (mean : ℝ) : ℝ :=
  (scores.map (fun x => |x - mean|)).maximum.getOrElse 0

def scores_A := [7, 6, 7, 8, 6]
def scores_B := [9, 5, 7, 9, 4]

def mean_A := average scores_A
def mean_B := average scores_B

def max_dev_A := max_deviation scores_A mean_A
def max_dev_B := max_deviation scores_B mean_B

theorem student_A_more_consistent : max_dev_A < max_dev_B :=
  by
    sorry

end student_A_more_consistent_l818_818186


namespace solve_eq_sqrt_exp_l818_818669

theorem solve_eq_sqrt_exp :
  (∀ x : ℝ, (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) → (x = 2 ∨ x = -1)) :=
by
  -- Prove that the solutions are x = 2 and x = -1
  sorry

end solve_eq_sqrt_exp_l818_818669


namespace minimum_PA_PF_l818_818020

open Real

noncomputable def hyperbola (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

def right_focus := (4 : ℝ, 0 : ℝ)
def point_A := (6 : ℝ, 0 : ℝ)
def left_focus := (-4 : ℝ, 0 : ℝ)

def PM (P M : ℝ × ℝ) : ℝ := (dist P M).toReal
def PF (P F : ℝ × ℝ) : ℝ := (dist P F).toReal

def point_on_hyperbola (P : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2

theorem minimum_PA_PF (P : ℝ × ℝ) (hP : point_on_hyperbola P) 
    (hFocusDist : PM P left_focus - PF P right_focus = 8) :
  (|PA| + |PF|) ≥ 6.7082 :=
  sorry

end minimum_PA_PF_l818_818020


namespace yogurt_combinations_l818_818193

-- Definitions: Given conditions from the problem
def num_flavors : ℕ := 5
def num_toppings : ℕ := 7

-- Function to calculate binomial coefficient
def nCr (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement: The problem translated into Lean
theorem yogurt_combinations : 
  (num_flavors * nCr num_toppings 2) = 105 := by
  sorry

end yogurt_combinations_l818_818193


namespace Sn_eq_sum_b_lt_five_elevens_l818_818716

variable (a : ℕ → ℚ) (S : ℕ → ℚ) (b : ℕ → ℚ)

-- Conditions
axiom a1_eq_half : a 1 = 1 / 2
axiom Sn_def : ∀ n : ℕ, n > 0 → S n = n^2 * a n - n * (n - 1)

-- Proof that S_n = n^2 / (n + 1)
theorem Sn_eq : ∀ n : ℕ, n > 0 → S n = n^2 / (n + 1) :=
sorry

-- Define b_n
def b (n : ℕ) : ℚ := S n / (n^3 + 3 * n^2)

-- The main theorem
theorem sum_b_lt_five_elevens : ∀ n : ℕ, n > 0 → (∑ i in Finset.range n, b i.succ) < 5 / 11 :=
sorry

end Sn_eq_sum_b_lt_five_elevens_l818_818716


namespace fir_trees_count_l818_818293

theorem fir_trees_count (N : ℕ) :
  (N = 15 ∨ (N < 25 ∧ 11 ∣ N) ∨ 22 ∣ N) ∧ 
  (1 ≤ (if N = 15 then 1 else 0) + (if 11 ∣ N then 1 else 0) + (if N < 25 then 1 else 0) + (if 22 ∣ N then 1 else 0) ∧
   2 ≠ (if N = 15 then 1 else 0) + (if 11 ∣ N then 1 else 0) + (if N < 25 then 1 else 0) + (if 22 ∣ N then 1 else 0)) → N = 11 :=
begin
  sorry
end

end fir_trees_count_l818_818293


namespace min_value_of_expr_l818_818991

variables (a b c : ℝ)
hypothesis h1 : 0 < a ∧ a < 1
hypothesis h2 : 0 < b ∧ b < 1
hypothesis h3 : 0 < c ∧ c < 1
hypothesis h4 : 3 * a + 2 * b = 2

theorem min_value_of_expr : ∃(x : ℝ), x = (2 / a + 1 / (3 * b)) ∧ x = 16 / 3 :=
by {
  sorry
}

end min_value_of_expr_l818_818991


namespace probability_james_david_l818_818409

theorem probability_james_david (total_workers : ℕ) (chosen_workers : ℕ) (James_David : ℕ) :
  total_workers = 14 →
  chosen_workers = 2 →
  James_David = 1 →
  ((James_David:ℚ) / ((total_workers.choose chosen_workers):ℚ) = (1/91:ℚ)) :=
begin
  intros h_total h_chosen h_jd,
  rw [h_total, h_chosen, h_jd],
  norm_num,
end

end probability_james_david_l818_818409


namespace triangle_area_is_25_l818_818946

-- Definitions directly from conditions
def base : ℝ := 10
def height : ℝ := 5

-- Area calculation using the given formula
def area (base height : ℝ) : ℝ := (base * height) / 2

-- Theorem statement
theorem triangle_area_is_25 : area base height = 25 := 
by
  sorry -- Proof to be filled in

end triangle_area_is_25_l818_818946


namespace sequence_type_l818_818314

theorem sequence_type (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (h₀ : ∀ n, S n = a_n n ^ 2 - 1) (h₁ : ∀ n, S (n + 1) = S n + a_n (n + 1)) :
  (∃ d : ℝ, ∀ n, a_n (n + 1) - a_n n = d) ∨ (∃ r : ℝ, ∀ n, a_n (n + 1) / a_n n = r) :=
begin
  sorry
end

end sequence_type_l818_818314


namespace number_of_trees_is_11_l818_818265

variable {N : ℕ}

-- Conditions stated by each child
def anya_statement : Prop := N = 15
def borya_statement : Prop := N % 11 = 0
def vera_statement : Prop := N < 25
def gena_statement : Prop := N % 22 = 0

-- One boy and one girl told the truth, while the other two lied
def truth_condition : Prop :=
  (borya_statement ∧ vera_statement ∧ ¬anya_statement ∧ ¬gena_statement) ∨
  (borya_statement ∧ ¬vera_statement ∧ ¬anya_statement ∧ gena_statement) ∨
  (¬borya_statement ∧ vera_statement ∧ ¬anya_statement ∧ gena_statement) ∨
  (¬borya_statement ∧ vera_statement ∧ anya_statement ∧ ¬gena_statement)

-- Proving that the number of fir trees is 11
theorem number_of_trees_is_11 (h : truth_condition) : N = 11 := 
by
  sorry

end number_of_trees_is_11_l818_818265


namespace isosceles_right_triangle_hypotenuse_l818_818059

noncomputable def hypotenuse (r p : ℝ) (h1 : r > 0) (h2 : p > r) : ℝ :=
p - r

theorem isosceles_right_triangle_hypotenuse (r p : ℝ) (h1 : r > 0) (h2 : p > r) :
  ∃ c : ℝ, c = hypotenuse r p h1 h2 :=
begin
  use p - r,
  exact rfl,
end

end isosceles_right_triangle_hypotenuse_l818_818059


namespace probability_charlie_daisy_meet_l818_818618

noncomputable def probability_overlap := 
  let P := measure_theory.probability_measure (set.univ : set (ℝ × ℝ)) in
  P ({p : ℝ × ℝ | (|p.1 - p.2| < 1 / 3)})

theorem probability_charlie_daisy_meet :
  probability_overlap = 4 / 9 :=
sorry

end probability_charlie_daisy_meet_l818_818618


namespace inequality_solution_l818_818881

theorem inequality_solution (x : ℝ) : 
  (7 - 2 * (x + 1) ≥ 1 - 6 * x) ∧ ((1 + 2 * x) / 3 > x - 1) ↔ (-1 ≤ x ∧ x < 4) := 
by
  sorry

end inequality_solution_l818_818881


namespace sides_of_triangle_l818_818172

-- Definitions from conditions
variables (a b c : ℕ) (r bk kc : ℕ)
def is_tangent_split : Prop := bk = 8 ∧ kc = 6
def inradius : Prop := r = 4

-- Main theorem statement
theorem sides_of_triangle (h1 : is_tangent_split bk kc) (h2 : inradius r) : a + 6 = 13 ∧ a + 8 = 15 ∧ b = 14 := by
  sorry

end sides_of_triangle_l818_818172


namespace inequality_holds_for_least_M_l818_818644

noncomputable def least_real_M : ℝ :=
  9 * real.sqrt 2 / 32

theorem inequality_holds_for_least_M :
  ∀ (a b c : ℝ), abs (a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2))
  ≤ least_real_M * (a^2 + b^2 + c^2)^2 :=
  sorry

end inequality_holds_for_least_M_l818_818644


namespace cubic_roots_inequality_l818_818737

theorem cubic_roots_inequality
  (a b c p q r : ℝ)
  (h_root: ∀ x : ℝ, x^3 + a*x^2 + b*x + c = 0 → x = p ∨ x = q ∨ x = r)
  (h_le: p ≤ q ∧ q ≤ r) :
  a^2 - 3*b ≥ 0 ∧ real.sqrt (a^2 - 3 * b) ≤ r - p :=
by sorry

end cubic_roots_inequality_l818_818737


namespace math_club_and_basketball_team_l818_818788

open Set

variables (U : Type) (students : Finset U) (S M F B : Finset U)
variables (h_total : students.card = 60) (h_S_union_M : S ∪ M = students)
variables (h_S_inter_M : S ∩ M = ∅) (h_all_clubs : S ∪ M = students)
variables (h_all_sports : F ∪ B = students) (h_SF : (S ∩ F).card = 20)
variables (h_M_card : M.card = 36) (h_B_card : B.card = 22)

theorem math_club_and_basketball_team:
  (M ∩ B).card = 18 :=
  sorry

end math_club_and_basketball_team_l818_818788


namespace angle_opposite_c_is_120_degrees_l818_818386

-- Define lengths of triangle sides a, b, and c.
variables (a b c : ℝ)

-- Define the condition given in the problem.
def condition : Prop :=
  (a + b + c) * (a + b - c) = 2 * a * b

-- State the theorem about the angle opposite the side of length c.
theorem angle_opposite_c_is_120_degrees (h : condition a b c) : 
  ∃ C : ℝ, C = 120 ∧ 
  cos C = -1 / 2 := 
sorry

end angle_opposite_c_is_120_degrees_l818_818386


namespace number_of_integer_solutions_l818_818764

def satisfies_inequality (x : Int) : Prop :=
  abs (7 * x + 5) ≤ 9

theorem number_of_integer_solutions :
  {x : Int | satisfies_inequality x}.finite.to_finset.card = 3 :=
by
  simp [satisfies_inequality]
  sorry

end number_of_integer_solutions_l818_818764


namespace volume_of_solid_l818_818598

noncomputable def volume_of_tetrahedron (S : ℝ) : ℝ :=
  (Math.sqrt 2 * S^3) / 12

theorem volume_of_solid (a b c : ℝ) (h1: a = 6 * Real.sqrt 2) (h2: b = 6 * Real.sqrt 2) (h3: c = 6 * Real.sqrt 2) :
  volume_of_tetrahedron (2 * a) / 2 = 144 * Real.sqrt 2 :=
  by
    sorry

end volume_of_solid_l818_818598


namespace robin_photo_count_l818_818026

theorem robin_photo_count (photos_per_page : ℕ) (full_pages : ℕ) 
  (h1 : photos_per_page = 6) (h2 : full_pages = 122) :
  photos_per_page * full_pages = 732 :=
by
  sorry

end robin_photo_count_l818_818026


namespace number_in_marked_square_is_10_l818_818888

theorem number_in_marked_square_is_10 : 
  ∃ f : ℕ × ℕ → ℕ, 
    (f (0,0) = 5 ∧ f (0,1) = 6 ∧ f (0,2) = 7) ∧ 
    (∀ r c, r > 0 → 
      f (r,c) = f (r-1,c) + f (r-1,c+1)) 
    ∧ f (1, 1) = 13 
    ∧ f (2, 1) = 10 :=
    sorry

end number_in_marked_square_is_10_l818_818888


namespace number_of_pumps_l818_818037

theorem number_of_pumps (P : ℕ) : 
  (P * 8 * 2 = 8 * 6) → P = 3 :=
by
  intro h
  sorry

end number_of_pumps_l818_818037


namespace contrapositive_statement_l818_818490

theorem contrapositive_statement {a b : ℤ} :
  (∀ a b : ℤ, (a % 2 = 1 ∧ b % 2 = 1) → (a + b) % 2 = 0) →
  (∀ a b : ℤ, ¬((a + b) % 2 = 0) → ¬(a % 2 = 1 ∧ b % 2 = 1)) :=
by 
  intros h a b
  sorry

end contrapositive_statement_l818_818490


namespace find_focus_of_parabola_l818_818671

noncomputable def focus_of_parabola (a b c : ℝ) : ℝ × ℝ :=
let h := -b / (2 * a),
    k := c - (b^2 / (4 * a))
in (h, k + 1 / (4 * a))

theorem find_focus_of_parabola : focus_of_parabola (-2) 0 5 = (0, 39 / 8) :=
by sorry

end find_focus_of_parabola_l818_818671


namespace sum_proper_divisors_512_l818_818958

theorem sum_proper_divisors_512 : ∑ i in {1, 2, 4, 8, 16, 32, 64, 128, 256}, i = 511 :=
by
  have h : {1, 2, 4, 8, 16, 32, 64, 128, 256} = finset.range 9.image (λ n, 2^n) := sorry
  rw [h]
  sorry

end sum_proper_divisors_512_l818_818958


namespace min_value_OP_squared_plus_PF_squared_l818_818775

theorem min_value_OP_squared_plus_PF_squared :
  let O := (0, 0)
  let F := (-1, 0)
  ∃ P : ℝ × ℝ, (P.1^2 / 2 + P.2^2 = 1) ∧ (|O - P|^2 + |P - F|^2 = 2) :=
by
  let P := (x, y)
  sorry

end min_value_OP_squared_plus_PF_squared_l818_818775


namespace least_positive_integer_with_six_factors_is_18_l818_818139

-- Define the least positive integer with exactly six distinct positive factors.
def least_positive_with_six_factors (n : ℕ) : Prop :=
  (∀ d : ℕ, d ∣ n → d > 0) ∧ (finset.card (finset.filter (λ d, d ∣ n) (finset.range (n + 1)))) = 6

-- Prove that the least positive integer with exactly six distinct positive factors is 18.
theorem least_positive_integer_with_six_factors_is_18 : (∃ n : ℕ, least_positive_with_six_factors n ∧ n = 18) :=
sorry


end least_positive_integer_with_six_factors_is_18_l818_818139


namespace sufficient_but_not_necessary_condition_l818_818323

theorem sufficient_but_not_necessary_condition (m : ℝ) (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = cos x + m - 1) → (∀ x, 0 ≤ m ∧ m ≤ 2 → ∃ x : ℝ, f x = 0) ∧ ¬(∀ x, 0 ≤ m ∧ m ≤ 1 ↔ ∃ x : ℝ, f x = 0) :=
by 
  sorry

end sufficient_but_not_necessary_condition_l818_818323


namespace difference_of_numbers_is_21938_l818_818521

theorem difference_of_numbers_is_21938 
  (x y : ℕ) 
  (h1 : x + y = 26832) 
  (h2 : x % 10 = 0) 
  (h3 : y = x / 10 + 4) 
  : x - y = 21938 :=
sorry

end difference_of_numbers_is_21938_l818_818521


namespace smallest_b_factors_l818_818236

theorem smallest_b_factors (b : ℕ) (m n : ℤ) (h : m * n = 2023 ∧ m + n = b) : b = 136 :=
sorry

end smallest_b_factors_l818_818236


namespace sum_proper_divisors_of_512_l818_818955

theorem sum_proper_divisors_of_512 : ∑ i in finset.range 9, 2^i = 511 :=
by
  -- We are stating that the sum of 2^i for i ranging from 0 to 8 equals 511.
  sorry

end sum_proper_divisors_of_512_l818_818955


namespace kim_dropped_classes_l818_818820

/-- 
Kim initially takes 4 classes, each lasting 2 hours. After dropping some classes, she now has 
6 hours of classes per day. We need to prove that she dropped exactly 1 class. 
-/
theorem kim_dropped_classes :
  (initial_classes : ℕ) (class_length : ℕ) (initial_total_hours : ℕ) (new_total_hours : ℕ)
  (dropped_classes : ℕ) :
  initial_classes = 4 → 
  class_length = 2 → 
  initial_total_hours = initial_classes * class_length →
  new_total_hours = 6 →
  dropped_classes = (initial_total_hours - new_total_hours) / class_length →
  dropped_classes = 1 :=
by
  intros
  sorry

end kim_dropped_classes_l818_818820


namespace investment_years_l818_818234

noncomputable def principal : ℝ := 500
noncomputable def annual_rate : ℝ := 0.05
noncomputable def compound_interest : ℝ := 138.14
noncomputable def compounding_periods_per_year : ℕ := 1

theorem investment_years : 
  ∃ (t : ℕ), (principal * (1 + annual_rate / compounding_periods_per_year)^(compounding_periods_per_year * t) = principal + compound_interest) := 
begin
  use 5,   -- We use 5 years as found in the solution
  sorry
end

end investment_years_l818_818234


namespace Desiree_age_l818_818210

-- Definitions of the variables involved
variables {D C : ℕ}

-- Conditions given in the problem
def condition1 : Prop := D = 2 * C
def condition2 : Prop := D + 30 = (2 * (C + 30)) / 3 + 14

-- The theorem to be proved
theorem Desiree_age (h1 : condition1) (h2 : condition2) : D = 6 :=
sorry

end Desiree_age_l818_818210


namespace exists_lines_l818_818708

-- Define the circle equation
def circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * x + 2 * y - 4 = 0

-- Define the line equation in terms of y-intercept b
def line (b x y : ℝ) : Prop :=
  y = x + b

-- Define the range of y-intercept b for intersection at two distinct points
def valid_range_b (b : ℝ) : Prop :=
  -3 - 3 * real.sqrt 2 < b ∧ b < -3 + 3 * real.sqrt 2

-- Hypothesize the existence of line l such that the circle with diameter AB passes through the origin
def circle_diameter_origin (b x1 y1 x2 y2 : ℝ) : Prop :=
  line b x1 y1 ∧ line b x2 y2 ∧ circle x1 y1 ∧ circle x2 y2 ∧ 
  ((x1 + x2 = -(b + 1)) ∧ (x1 * x2 = (b^2 + 4*b - 4) / 2) ∧ (x1 * x2 + (x1 + b) * (x2 + b) = 0))

-- Prove the existence of the specific lines within the valid range
theorem exists_lines (b : ℝ) : valid_range_b b →
  ∃ x1 y1 x2 y2, circle_diameter_origin b x1 y1 x2 y2 ↔ (b = -1 + 4 ∨ b = -1 - 4) :=
sorry

end exists_lines_l818_818708


namespace length_OP_is_2_div_3_times_sqrt_73_l818_818799

noncomputable def length_OP (BC AC : ℝ) [nontrivial ℝ] : ℝ :=
  let AB := Real.sqrt (BC^2 + AC^2) in
  let AP := Real.sqrt (AC^2 + (BC / 2)^2) in
  2 / 3 * AP

theorem length_OP_is_2_div_3_times_sqrt_73 :
  length_OP 6 8 = 2 / 3 * Real.sqrt 73 :=
by
  -- Proof is omitted
  sorry

end length_OP_is_2_div_3_times_sqrt_73_l818_818799


namespace degenerate_ellipse_b_value_l818_818623

theorem degenerate_ellipse_b_value :
  ∃ b : ℝ, (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 6 * y + b = 0 → x = -1 ∧ y = 3) ↔ b = 12 :=
by
  sorry

end degenerate_ellipse_b_value_l818_818623


namespace count_ways_choose_boxes_l818_818619

theorem count_ways_choose_boxes : 
  let TotalWays := Nat.choose 6 3
  let WaysWithoutAorB := Nat.choose 4 3
  let WaysWithAtLeastAorB := TotalWays - WaysWithoutAorB
  in WaysWithAtLeastAorB = 16 :=
by
  sorry

end count_ways_choose_boxes_l818_818619


namespace max_three_digit_numbers_divisible_by_4_in_sequence_l818_818582

theorem max_three_digit_numbers_divisible_by_4_in_sequence (n : ℕ) (a : ℕ → ℕ)
  (h_n : n ≥ 3)
  (h_increasing : ∀ i j, i < j → a i < a j)
  (h_recurrence : ∀ k, k ≤ n - 2 → a (k + 2) = 3 * a (k + 1) - 2 * a k - 2)
  (h_contains_2022 : ∃ k, a k = 2022) :
  ∀ k, a k = 2 * k → 
  (λ count_4 : ℕ, 
    (∀ m, 25 ≤ m ∧ m ≤ 249 → a (2 * m) = 4 * m) → 
    count_4 = 225) :=
begin
  sorry
end

end max_three_digit_numbers_divisible_by_4_in_sequence_l818_818582


namespace find_d_minus_r_l818_818769

theorem find_d_minus_r :
  ∃ (d r : ℕ), d > 1 ∧ 1083 % d = r ∧ 1455 % d = r ∧ 2345 % d = r ∧ d - r = 1 := by
  sorry

end find_d_minus_r_l818_818769


namespace question_II_question_III_l818_818351

-- Definitions from the conditions in the problem
def f (x : ℝ) : ℝ := (1 / 3) * x ^ 2 + (2 / 3) * x

def a_n (n : ℕ) : ℝ := if n > 0 then (2 * n + 1) / 3 else 0

def S_n (n : ℕ) : ℝ := (1 / 3) * n ^ 2 + (2 / 3) * n

def b_n (n : ℕ) : ℝ := a_n n * a_n (n + 1) * real.cos ((n + 1) * real.pi)

def T_n (n : ℕ) : ℝ := ∑ i in finset.range n, b_n i

-- Specify that \(T_n \geqslant t n^2\) holds for \(t \leq -\frac{5}{9}\)
theorem question_II (t : ℝ) (h : ∀ n : ℕ, T_n n ≥ t * n ^ 2) : t ≤ -5 / 9 :=
sorry

-- Prove the geometric sequence with common ratio 3 exists and n_k = (3^k-1)/2
theorem question_III (q : ℝ) (h_q : 0 < q ∧ q < 5) :
  (∃ (n_k : ℕ → ℕ) (a₁ : ℝ), (∀ k > 0, a_n (n_k k) = a₁ * q ^ k) ∧ q = 3) :=
sorry

end question_II_question_III_l818_818351


namespace lcm_36_105_l818_818678

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end lcm_36_105_l818_818678


namespace december_fraction_of_yearly_sales_l818_818554

theorem december_fraction_of_yearly_sales (A : ℝ) (h_sales : ∀ (x : ℝ), x = 6 * A) :
    let yearly_sales := 11 * A + 6 * A
    let december_sales := 6 * A
    december_sales / yearly_sales = 6 / 17 := by
  sorry

end december_fraction_of_yearly_sales_l818_818554


namespace Delaney_missed_bus_by_l818_818635

def time_in_minutes (hours : ℕ) (minutes : ℕ) : ℕ := hours * 60 + minutes

def Delaney_start_time : ℕ := time_in_minutes 7 50
def bus_departure_time : ℕ := time_in_minutes 8 0
def travel_duration : ℕ := 30

theorem Delaney_missed_bus_by :
  Delaney_start_time + travel_duration - bus_departure_time = 20 :=
by
  sorry

end Delaney_missed_bus_by_l818_818635


namespace distance_from_O_to_AD_eq_half_BC_l818_818465

-- Define the mathematical conditions
variables {A B C D O : Type} [Circle ABCD O] -- Quadrilateral ABCD inscribed in a circle with center O

-- Define the perpendicularity condition
variables (AC BD : Line) (h_perpendicular : Perpendicular AC BD)

-- The theorem to prove the distance relationship
theorem distance_from_O_to_AD_eq_half_BC (O A D : Point) (BC : Length) :
  let d := distance_from O to AD in d = BC / 2 :=
sorry

end distance_from_O_to_AD_eq_half_BC_l818_818465


namespace find_triangle_sides_l818_818477

noncomputable def triangle_sides (x : ℝ) : Prop :=
  let a := x - 2
  let b := x
  let c := x + 2
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  a + 2 = b ∧ b + 2 = c ∧ area = 6 ∧
  a = 2 * Real.sqrt 6 - 2 ∧
  b = 2 * Real.sqrt 6 ∧
  c = 2 * Real.sqrt 6 + 2

theorem find_triangle_sides :
  ∃ x : ℝ, triangle_sides x := by
  sorry

end find_triangle_sides_l818_818477


namespace distinct_pos_ints_reciprocal_sum_one_l818_818028

theorem distinct_pos_ints_reciprocal_sum_one (n : ℕ) (h : n > 2) : 
  ∃ (S : Finset ℕ), S.card = n ∧ (∑ x in S, 1 / (x : ℚ)) = 1 := 
sorry

end distinct_pos_ints_reciprocal_sum_one_l818_818028


namespace clock_confusion_times_l818_818366

-- Conditions translated into Lean definitions
def h_move : ℝ := 0.5  -- hour hand moves at 0.5 degrees per minute
def m_move : ℝ := 6.0  -- minute hand moves at 6 degrees per minute

-- Overlap condition formulated
def overlap_condition (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ 10 ∧ 11 * (n : ℝ) = k * 360

-- The final theorem statement in Lean 4
theorem clock_confusion_times : 
  ∃ (count : ℕ), count = 132 ∧ 
    (∀ n < 144, (overlap_condition n → false)) :=
by
  -- Proof to be inserted here
  sorry

end clock_confusion_times_l818_818366


namespace sum_of_squares_l818_818202

theorem sum_of_squares (a b : ℕ) (h₁ : a = 300000) (h₂ : b = 20000) : a^2 + b^2 = 9004000000 :=
by
  rw [h₁, h₂]
  sorry

end sum_of_squares_l818_818202


namespace projection_magnitude_AC_onto_AB_l818_818321

noncomputable def point := (ℝ × ℝ × ℝ)

def A : point := (2, 1, 0)
def B : point := (0, 3, 1)
def C : point := (2, 2, 3)

def vector_sub (v w : point) : point := (v.1 - w.1, v.2 - w.2, v.3 - w.3)
def dot_product (v w : point) : ℝ := v.1 * w.1 + v.2 * w.2 + v.3 * w.3
def magnitude (v : point) : ℝ := Real.sqrt (v.1^2 + v.2^2 + v.3^2)
def projection_magnitude (v w : point) : ℝ := (dot_product v w) / (magnitude w)

def AB : point := vector_sub B A
def AC : point := vector_sub C A

theorem projection_magnitude_AC_onto_AB :
  projection_magnitude AC AB = 5 / 3 :=
sorry

end projection_magnitude_AC_onto_AB_l818_818321


namespace number_of_trees_is_11_l818_818264

variable {N : ℕ}

-- Conditions stated by each child
def anya_statement : Prop := N = 15
def borya_statement : Prop := N % 11 = 0
def vera_statement : Prop := N < 25
def gena_statement : Prop := N % 22 = 0

-- One boy and one girl told the truth, while the other two lied
def truth_condition : Prop :=
  (borya_statement ∧ vera_statement ∧ ¬anya_statement ∧ ¬gena_statement) ∨
  (borya_statement ∧ ¬vera_statement ∧ ¬anya_statement ∧ gena_statement) ∨
  (¬borya_statement ∧ vera_statement ∧ ¬anya_statement ∧ gena_statement) ∨
  (¬borya_statement ∧ vera_statement ∧ anya_statement ∧ ¬gena_statement)

-- Proving that the number of fir trees is 11
theorem number_of_trees_is_11 (h : truth_condition) : N = 11 := 
by
  sorry

end number_of_trees_is_11_l818_818264


namespace solve_system_eq_l818_818467

theorem solve_system_eq (x y z t : ℕ) : 
  ((x^2 + t^2) * (z^2 + y^2) = 50) ↔
    (x = 1 ∧ y = 1 ∧ z = 2 ∧ t = 3) ∨
    (x = 3 ∧ y = 2 ∧ z = 1 ∧ t = 1) ∨
    (x = 4 ∧ y = 1 ∧ z = 3 ∧ t = 1) ∨
    (x = 1 ∧ y = 3 ∧ z = 4 ∧ t = 1) :=
by 
  sorry

end solve_system_eq_l818_818467


namespace range_of_a_l818_818369

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * |x - 1| + |x - a| ≥ 2) ↔ (a ≤ -1 ∨ a ≥ 3) :=
sorry

end range_of_a_l818_818369


namespace fabric_per_dress_is_correct_l818_818016

-- Definitions and conditions
def feet_to_yards (feet : ℝ) : ℝ := feet / 3

def total_fabric_in_yards (fabric_has_ft : ℝ) (fabric_needs_ft : ℝ) : ℝ :=
  feet_to_yards fabric_has_ft + feet_to_yards fabric_needs_ft

def fabric_per_dress (total_fabric_yd : ℝ) (num_dresses : ℕ) : ℝ :=
  total_fabric_yd / num_dresses

-- Theorem statement
theorem fabric_per_dress_is_correct :
  fabric_per_dress (total_fabric_in_yards 7 59) 4 = 5.5 :=
by
  sorry

end fabric_per_dress_is_correct_l818_818016


namespace number_of_trees_is_11_l818_818268

variable {N : ℕ}

-- Conditions stated by each child
def anya_statement : Prop := N = 15
def borya_statement : Prop := N % 11 = 0
def vera_statement : Prop := N < 25
def gena_statement : Prop := N % 22 = 0

-- One boy and one girl told the truth, while the other two lied
def truth_condition : Prop :=
  (borya_statement ∧ vera_statement ∧ ¬anya_statement ∧ ¬gena_statement) ∨
  (borya_statement ∧ ¬vera_statement ∧ ¬anya_statement ∧ gena_statement) ∨
  (¬borya_statement ∧ vera_statement ∧ ¬anya_statement ∧ gena_statement) ∨
  (¬borya_statement ∧ vera_statement ∧ anya_statement ∧ ¬gena_statement)

-- Proving that the number of fir trees is 11
theorem number_of_trees_is_11 (h : truth_condition) : N = 11 := 
by
  sorry

end number_of_trees_is_11_l818_818268


namespace correct_operator_is_subtraction_l818_818211

theorem correct_operator_is_subtraction :
  (8 - 2) + 5 * (3 - 2) = 11 :=
by
  sorry

end correct_operator_is_subtraction_l818_818211


namespace angle_opposite_c_is_120_degrees_l818_818385

-- Define lengths of triangle sides a, b, and c.
variables (a b c : ℝ)

-- Define the condition given in the problem.
def condition : Prop :=
  (a + b + c) * (a + b - c) = 2 * a * b

-- State the theorem about the angle opposite the side of length c.
theorem angle_opposite_c_is_120_degrees (h : condition a b c) : 
  ∃ C : ℝ, C = 120 ∧ 
  cos C = -1 / 2 := 
sorry

end angle_opposite_c_is_120_degrees_l818_818385


namespace saved_money_before_mowing_l818_818218

theorem saved_money_before_mowing (earned_per_lawn : ℕ) (number_of_lawns : ℕ) (total_money : ℕ) :
  earned_per_lawn = 8 → number_of_lawns = 5 → total_money = 47 → 
  let saved_money := total_money - earned_per_lawn * number_of_lawns
  in saved_money = 7 :=
begin
  intros h1 h2 h3,
  simp [h1, h2, h3],
  sorry
end

end saved_money_before_mowing_l818_818218


namespace sum_of_xs_eq_seven_l818_818005

-- Definitions based on conditions
def is_solution (x y z : ℂ) : Prop := 
  x + y*z = 8 ∧ y + x*z = 12 ∧ z + x*y = 12

-- Problem statement: Prove the sum of all x_i for solutions is 7
theorem sum_of_xs_eq_seven (S : finset (ℂ × ℂ × ℂ)) 
  (hS : ∀ s ∈ S, ∃ x y z : ℂ, s = (x, y, z) ∧ is_solution x y z) :
  ∑ s in S, (s.1).1 = 7 := 
sorry

end sum_of_xs_eq_seven_l818_818005


namespace n_digit_numbers_modulo_3_l818_818306

def a (i : ℕ) (n : ℕ) : ℕ :=
  if n = 0 then if i = 0 then 1 else 0 else 2 * a i (n - 1) + a ((i + 1) % 3) (n - 1) + a ((i + 2) % 3) (n - 1)

theorem n_digit_numbers_modulo_3 (n : ℕ) (h : 0 < n) : 
  (a 0 n) = (4^n + 2) / 3 :=
sorry

end n_digit_numbers_modulo_3_l818_818306


namespace least_positive_integer_with_six_factors_l818_818108

theorem least_positive_integer_with_six_factors :
  ∃ n : ℕ, (∀ m : ℕ, m > 0 → m < n → (count_factors m ≠ 6)) ∧ count_factors n = 6 ∧ n = 18 :=
sorry

noncomputable def count_factors (n : ℕ) : ℕ :=
sorry

end least_positive_integer_with_six_factors_l818_818108


namespace feathers_per_pound_l818_818013

theorem feathers_per_pound (pounds_per_pillow : ℕ) (pillows : ℕ) (total_feathers : ℕ) (total_pounds : ℕ) :
  pounds_per_pillow = 2 →
  pillows = 6 →
  total_feathers = 3600 →
  total_pounds = pillows * pounds_per_pillow →
  total_feathers / total_pounds = 300 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2] at h4
  have : total_pounds = 12 := h4
  rw [this, h3]
  exact 3600 / 12

end feathers_per_pound_l818_818013


namespace distance_from_yz_plane_l818_818048

theorem distance_from_yz_plane (x z : ℝ) : 
  (abs (-6) = (abs x) / 2) → abs x = 12 :=
by
  sorry

end distance_from_yz_plane_l818_818048


namespace f_neg_two_l818_818483

noncomputable def f : ℝ → ℝ := 
  λ x, if x > 0 then x^2 + 1 else if x < 0 then -(x^2 + 1) else 0

theorem f_neg_two : f (-2) = -5 := 
by 
  unfold f
  split_ifs with h₁ h₂
  -- cases should simplify to exact -5 based on definitions.
  sorry

end f_neg_two_l818_818483


namespace fir_trees_count_l818_818297

theorem fir_trees_count (N : ℕ) :
  (N = 15 ∨ (N < 25 ∧ 11 ∣ N) ∨ 22 ∣ N) ∧ 
  (1 ≤ (if N = 15 then 1 else 0) + (if 11 ∣ N then 1 else 0) + (if N < 25 then 1 else 0) + (if 22 ∣ N then 1 else 0) ∧
   2 ≠ (if N = 15 then 1 else 0) + (if 11 ∣ N then 1 else 0) + (if N < 25 then 1 else 0) + (if 22 ∣ N then 1 else 0)) → N = 11 :=
begin
  sorry
end

end fir_trees_count_l818_818297


namespace solve_for_unknown_l818_818165

theorem solve_for_unknown :
  ∃ (x : ℤ), 300 * 2 + (12 + 4) * x / 8 = 602 := 
begin
  use 1,
  sorry
end

end solve_for_unknown_l818_818165


namespace fish_total_after_transfer_l818_818008

-- Definitions of the initial conditions
def lilly_initial : ℕ := 10
def rosy_initial : ℕ := 9
def jack_initial : ℕ := 15
def fish_transferred : ℕ := 2

-- Total fish after Lilly transfers 2 fish to Jack
theorem fish_total_after_transfer : (lilly_initial - fish_transferred) + rosy_initial + (jack_initial + fish_transferred) = 34 := by
  sorry

end fish_total_after_transfer_l818_818008


namespace players_count_l818_818606

theorem players_count (total_socks : ℕ) (socks_per_player : ℕ) (h1 : total_socks = 16) (h2 : socks_per_player = 2) :
  total_socks / socks_per_player = 8 :=
by
  sorry

end players_count_l818_818606


namespace sum_of_digits_of_15_fac_l818_818887

theorem sum_of_digits_of_15_fac (A B : ℕ) 
  (h_base_rep: nat.digits 10 (15.factorial) = [0, 0, B, 8, A, 3, 4, 7, 6, 7, 0, 3, 0, 1]) 
  (h_fact_ending: B = 0) 
  (h_div_3: (1 + 3 + 0 + 7 + 6 + 7 + 4 + 3 + 8 + B) % 3 = 0) : 
  A + B = 3 := 
sorry

end sum_of_digits_of_15_fac_l818_818887


namespace solve_for_r_l818_818821

theorem solve_for_r : ∃ r : ℚ, 8 = 2^(5 * r + 1) → r = 2 / 5 :=
by
  sorry

end solve_for_r_l818_818821


namespace lines_perpendicular_l818_818913

theorem lines_perpendicular (k1 k2 : ℝ) (h : (Polynomial.X ^ 2 - 3 * Polynomial.X - 1).is_root k1 ∧ 
    (Polynomial.X ^ 2 - 3 * Polynomial.X - 1).is_root k2) : k1 * k2 = -1 := 
by
  sorry

end lines_perpendicular_l818_818913


namespace find_a_l818_818613

theorem find_a (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_max : ∀ x, a * Real.cos(b * x) ≤ 3)
  (h_min : ∀ x, a * Real.cos(b * x) ≥ -3) : 
  a = 3 := 
sorry

end find_a_l818_818613


namespace sequence_general_term_and_sum_l818_818753

theorem sequence_general_term_and_sum (a : ℕ → ℤ) (b : ℕ → ℤ) :
  (a 1 = 1) →
  (∀ n, a (n + 1) = -2 * a n) →
  (∀ n, b 1 = a 4 ∧ b 2 = a 2 - a 3) →
  (∀ n, a n = (-2)^(n-1)) ∧ (∀ n, ∑ i in finset.range n, a (i + 1) = (1 - (-2)^n) / 3) ∧ (b 37 = 64 ∧ ∃ n, a n = 64 ∧ n = 7) :=
by {
  assume h1 h2 h3,
  sorry
}

end sequence_general_term_and_sum_l818_818753


namespace multiple_of_n_equals_60_l818_818377

theorem multiple_of_n_equals_60
  (n : ℕ)
  (h1 : n < 200)
  (h2 : ∃ k, k * n % 60 = 0)
  (h3 : ∃ p1 p2 p3 : ℕ, p1.prime ∧ p2.prime ∧ p3.prime ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ n = p1 * p2 * p3) :
  ∃ m, m = 60 :=
by {
  sorry
}

end multiple_of_n_equals_60_l818_818377


namespace combined_collectors_edition_dolls_l818_818217

-- Definitions based on given conditions
def dina_dolls : ℕ := 60
def ivy_dolls : ℕ := dina_dolls / 2
def luna_dolls : ℕ := ivy_dolls - 10

-- Additional constraints based on the problem statement
def total_dolls : ℕ := dina_dolls + ivy_dolls + luna_dolls
def ivy_collectors_edition_dolls : ℕ := 2/3 * ivy_dolls
def luna_collectors_edition_dolls : ℕ := 1/2 * luna_dolls

-- Proof statement
theorem combined_collectors_edition_dolls :
  ivy_collectors_edition_dolls + luna_collectors_edition_dolls = 30 :=
sorry

end combined_collectors_edition_dolls_l818_818217


namespace mary_total_zoom_time_l818_818846

noncomputable def timeSpentDownloadingMac : ℝ := 10
noncomputable def timeSpentDownloadingWindows : ℝ := 3 * timeSpentDownloadingMac
noncomputable def audioGlitchesCount : ℝ := 2
noncomputable def audioGlitchDuration : ℝ := 4
noncomputable def totalAudioGlitchTime : ℝ := audioGlitchesCount * audioGlitchDuration
noncomputable def videoGlitchDuration : ℝ := 6
noncomputable def totalGlitchTime : ℝ := totalAudioGlitchTime + videoGlitchDuration
noncomputable def glitchFreeTalkingTime : ℝ := 2 * totalGlitchTime

theorem mary_total_zoom_time : 
  timeSpentDownloadingMac + timeSpentDownloadingWindows + totalGlitchTime + glitchFreeTalkingTime = 82 :=
by sorry

end mary_total_zoom_time_l818_818846


namespace least_positive_integer_with_six_factors_l818_818116

-- Define what it means for a number to have exactly six distinct positive factors
def hasExactlySixFactors (n : ℕ) : Prop :=
  (n.factorization.support.card = 2 ∧ (n.factorization.values' = [2, 1])) ∨
  (n.factorization.support.card = 1 ∧ (n.factorization.values' = [5]))

-- The main theorem statement
theorem least_positive_integer_with_six_factors : ∃ n : ℕ, hasExactlySixFactors n ∧ ∀ m : ℕ, (hasExactlySixFactors m → n ≤ m) :=
  exists.intro 12 (and.intro
    (show hasExactlySixFactors 12, by sorry)
    (show ∀ m : ℕ, hasExactlySixFactors m → 12 ≤ m, by sorry))

end least_positive_integer_with_six_factors_l818_818116


namespace thirtieth_valid_sequence_term_is_292_l818_818517

def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

def contains_digit_2 (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∣ n ∧ d = 2

def valid_sequence_term (n : ℕ) : Prop :=
  is_multiple_of_4 n ∧ contains_digit_2 n

theorem thirtieth_valid_sequence_term_is_292 :
  (list.filter valid_sequence_term (list.range 10000)).get? 29 = some 292 :=
by
  sorry

end thirtieth_valid_sequence_term_is_292_l818_818517


namespace max_tetrahedron_distance_l818_818049

noncomputable def max_distance_pq (A B C D : ℝ) : ℝ :=
  let P := A -- Point P on edge AB, specifically at A
  let Q := D -- Point Q on edge CD, specifically at D
  real.sqrt ((A - B)^2 + (C - D)^2 + (P - Q)^2) -- This represents the distance calculation

theorem max_tetrahedron_distance (A B C D P Q : ℝ) (h : true) :
  max_distance_pq 0 0 0 (1 : ℝ) = (2 * real.sqrt 6 / 3) :=
sorry

end max_tetrahedron_distance_l818_818049


namespace probability_intersecting_diagonals_l818_818926

def number_of_vertices := 10

def number_of_diagonals : ℕ := Nat.choose number_of_vertices 2 - number_of_vertices

def number_of_ways_choose_two_diagonals := Nat.choose number_of_diagonals 2

def number_of_sets_of_intersecting_diagonals : ℕ := Nat.choose number_of_vertices 4

def intersection_probability : ℚ :=
  (number_of_sets_of_intersecting_diagonals : ℚ) / (number_of_ways_choose_two_diagonals : ℚ)

theorem probability_intersecting_diagonals :
  intersection_probability = 42 / 119 :=
by
  sorry

end probability_intersecting_diagonals_l818_818926


namespace exist_n_tuple_satisfying_conditions_l818_818419

theorem exist_n_tuple_satisfying_conditions (p q : ℝ) (y : Fin 2017 → ℝ)
  (hp : 0 < p) (hq : 0 < q) (hpq : p + q = 1) :
  ∃ x : Fin 2018 → ℝ,
    ∀ i : Fin 2017, p * max (x i) (x ⟨i.1 + 1, by linarith⟩) + q * min (x i) (x ⟨i.1 + 1, by linarith⟩) = y i ∧ (x ⟨2017, by linarith⟩ = x ⟨0, by linarith⟩) :=
  sorry

end exist_n_tuple_satisfying_conditions_l818_818419


namespace transform_odd_function_l818_818342

def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x) * Real.cos x

theorem transform_odd_function :
  ∀ x : ℝ, f (x - π/8) - 1/2 = -(f (-(x - π/8)) - 1/2) := by
  sorry

end transform_odd_function_l818_818342


namespace king_placements_l818_818393

def king_attack_positions (pos : ℕ × ℕ) : set (ℕ × ℕ) :=
{(pos.1 - 1, pos.2 - 1), (pos.1 - 1, pos.2), (pos.1 - 1, pos.2 + 1),
 (pos.1, pos.2 - 1),                 (pos.1, pos.2 + 1),
 (pos.1 + 1, pos.2 - 1), (pos.1 + 1, pos.2), (pos.1 + 1, pos.2 + 1) }

def non_attacking_king_positions (board_size : ℕ) : ℕ :=
let all_positions := (finset.range board_size).product (finset.range board_size) in
all_positions.card * (all_positions.filter (λ pos2,
  ∀ pos1 ∈ all_positions, king_attack_positions pos1 ⊆ all_positions → ¬ (pos2 ∈ king_attack_positions pos1))).card / 2

theorem king_placements : non_attacking_king_positions 8 = 1806 :=
sorry

end king_placements_l818_818393


namespace sin_angle_DAE_l818_818792

/--
In an equilateral triangle ABC, D is the midpoint of BC and E is a point that trisects AC.
Prove that sin(∠DAE) = 1 / sqrt(5).
-/
theorem sin_angle_DAE (A B C D E : EuclideanGeometry.Point 2) :
  let side := λ A B C : EuclideanGeometry.Point 2, 
               EuclideanGeometry.is_equilateral_triangle A B C →
               EuclideanGeometry.dist A B = 8 ∧ EuclideanGeometry.dist B C = 8 ∧ EuclideanGeometry.dist C A = 8 in
  let D_midpoint := λ B C : EuclideanGeometry.Point 2, D.midpoint (B, C) in
  let E_trisects := λ A C : EuclideanGeometry.Point 2, E.trisects (A, C) in
  side A B C ∧ D_midpoint B C ∧ E_trisects A C →
  Real.sin (angle A D E) = 1 / Real.sqrt 5 := sorry

end sin_angle_DAE_l818_818792


namespace probability_P_closer_to_origin_l818_818184

noncomputable def probability_closer_to_origin (rect : set (ℝ × ℝ)) (origin P : ℝ × ℝ) (target : ℝ × ℝ) : ℝ :=
  let area_rectangle := 6
  let area_triangle := 1
  area_triangle / area_rectangle

theorem probability_P_closer_to_origin (P : ℝ × ℝ) (H : P ∈ set.univ)
  (rect : set (ℝ × ℝ) := (λ p, 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2))
  (origin : ℝ × ℝ := (0, 0))
  (target : ℝ × ℝ := (4, 2)) :
  probability_closer_to_origin rect origin P target = 1 / 6 :=
by
  sorry

end probability_P_closer_to_origin_l818_818184


namespace pinedale_mall_distance_l818_818982

theorem pinedale_mall_distance 
  (speed : ℝ) (time_between_stops : ℝ) (num_stops : ℕ) (distance : ℝ) 
  (h_speed : speed = 60) 
  (h_time_between_stops : time_between_stops = 5 / 60) 
  (h_num_stops : ↑num_stops = 5) :
  distance = 25 :=
by
  sorry

end pinedale_mall_distance_l818_818982


namespace fraction_multiplication_l818_818945

theorem fraction_multiplication :
  (1 / 3 : ℚ) * (1 / 2) * (3 / 4) * (5 / 6) = 5 / 48 := by
  sorry

end fraction_multiplication_l818_818945


namespace students_tried_out_l818_818524

theorem students_tried_out (not_picked : ℕ) (groups : ℕ) (students_per_group : ℕ)
  (h1 : not_picked = 36) (h2 : groups = 4) (h3 : students_per_group = 7) :
  not_picked + groups * students_per_group = 64 :=
by
  sorry

end students_tried_out_l818_818524


namespace simplify_tan20_cot10_l818_818479
open Real

theorem simplify_tan20_cot10 :
  tan 20 + cot 10 = csc 20 :=
by
  sorry

end simplify_tan20_cot10_l818_818479


namespace percentage_failed_in_Hindi_l818_818793

-- Define the percentage of students failed in English
def percentage_failed_in_English : ℝ := 56

-- Define the percentage of students failed in both Hindi and English
def percentage_failed_in_both : ℝ := 12

-- Define the percentage of students passed in both subjects
def percentage_passed_in_both : ℝ := 24

-- Define the total percentage of students
def percentage_total : ℝ := 100

-- Define what we need to prove
theorem percentage_failed_in_Hindi:
  ∃ (H : ℝ), H + percentage_failed_in_English - percentage_failed_in_both + percentage_passed_in_both = percentage_total ∧ H = 32 :=
  by 
    sorry

end percentage_failed_in_Hindi_l818_818793


namespace smallest_prime_factor_sum_four_consecutive_integers_l818_818950

theorem smallest_prime_factor_sum_four_consecutive_integers :
  ∀ (n : ℤ), ∃ p : ℕ, Prime p ∧ p = 2 ∧ p ∣ ((n - 1) + n + (n + 1) + (n + 2)) :=
by
  intro n
  use 2
  sorry

end smallest_prime_factor_sum_four_consecutive_integers_l818_818950


namespace rex_cards_remaining_l818_818014

theorem rex_cards_remaining
  (nicole_cards : ℕ)
  (cindy_cards : ℕ)
  (rex_cards : ℕ)
  (cards_per_person : ℕ)
  (h1 : nicole_cards = 400)
  (h2 : cindy_cards = 2 * nicole_cards)
  (h3 : rex_cards = (nicole_cards + cindy_cards) / 2)
  (h4 : cards_per_person = rex_cards / 4) :
  cards_per_person = 150 :=
by
  sorry

end rex_cards_remaining_l818_818014


namespace circles_intersection_theorem_l818_818924

section circles_intersection

variables {A B : ℝ × ℝ} (m c : ℝ)

-- Two circles intersect at points A(1, 3) and B(m, -1)
def intersect_points (A B : ℝ × ℝ) : Prop :=
  A = (1, 3) ∧ B = (m, -1)

-- Centers of the circles lie on the line x - y + c = 0
def centers_on_line (x y c : ℝ) : Prop :=
  x - y + c = 0

-- Prove that given the above conditions, m + c = 3
theorem circles_intersection_theorem
  (h_intersect : intersect_points (1, 3) (m, -1))
  (h_centers : ∀ (x y : ℝ), centers_on_line x y c)
  : m + c = 3 :=
sorry

end circles_intersection

end circles_intersection_theorem_l818_818924


namespace right_triangle_relation_l818_818787

theorem right_triangle_relation (a b c x : ℝ)
  (h : c^2 = a^2 + b^2)
  (altitude : a * b = c * x) :
  (1 / x^2) = (1 / a^2) + (1 / b^2) :=
sorry

end right_triangle_relation_l818_818787


namespace comparison_of_negatives_l818_818566

theorem comparison_of_negatives : -2 < - (3 / 2) :=
by
  sorry

end comparison_of_negatives_l818_818566


namespace birds_joined_l818_818570

theorem birds_joined (B : ℕ) : 
    (∃ B, let initial_birds := 3 in 
          let initial_storks := 4 in 
          let final_birds := initial_birds + B in 
          let final_storks := initial_storks in
          final_birds = final_storks + 1) → 
    B = 2 :=
by intros h; rcases h with ⟨B, hb⟩; sorry

end birds_joined_l818_818570


namespace each_car_has_4_wheels_l818_818647
-- Import necessary libraries

-- Define the conditions
def number_of_guests := 40
def number_of_parent_cars := 2
def wheels_per_parent_car := 4
def number_of_guest_cars := 10
def total_wheels := 48
def parent_car_wheels := number_of_parent_cars * wheels_per_parent_car
def guest_car_wheels := total_wheels - parent_car_wheels

-- Define the proposition to prove
theorem each_car_has_4_wheels : (guest_car_wheels / number_of_guest_cars) = 4 :=
by
  sorry

end each_car_has_4_wheels_l818_818647


namespace number_of_fir_trees_l818_818286

def anya_statement (N : ℕ) : Prop := N = 15
def borya_statement (N : ℕ) : Prop := 11 ∣ N
def vera_statement (N : ℕ) : Prop := N < 25
def gena_statement (N : ℕ) : Prop := 22 ∣ N

def one_boy_one_girl_truth (A B G V : Prop) : Prop :=
  (A ∨ V) ∧ ¬(A ∧ V) ∧ (B ∨ G) ∧ ¬(B ∧ G)

theorem number_of_fir_trees (N : ℕ) :
  anya_statement N ∨ borya_statement N ∨ vera_statement N ∨ gena_statement N ∧
  one_boy_one_girl_truth (anya_statement N) (borya_statement N) (gena_statement N) (vera_statement N) :=
  N = 11 :=
sorry

end number_of_fir_trees_l818_818286


namespace rational_root_divides_polynomial_evaluation_l818_818866

theorem rational_root_divides_polynomial_evaluation
  {a_0 a_1 a_n : ℤ} {n : ℕ} (f : ℤ[X]) (p q : ℤ) (k : ℤ) 
  (hpq_coprime : p.gcd q = 1) (h_irreducible_root: f.eval ↑p / ↑q = 0)
  (h_polynomial: f = ∑ i in finset.range (n + 1), (a_n i) * X^i) : 
  (p - k * q) ∣ (f.eval k) := 
by 
  sorry

end rational_root_divides_polynomial_evaluation_l818_818866


namespace inequality_holds_l818_818308

-- Define the function f
variable (f : ℝ → ℝ)

-- Given conditions
axiom symmetric_property : ∀ x : ℝ, f (1 - x) = f (1 + x)
axiom increasing_property : ∀ x y : ℝ, (1 ≤ x) → (x ≤ y) → f x ≤ f y

-- The statement of the theorem
theorem inequality_holds (m : ℝ) (h : m < 1 / 2) : f (1 - m) < f m :=
by sorry

end inequality_holds_l818_818308


namespace token_problem_l818_818007

theorem token_problem
  (x y : ℕ) 
  (h1 : x = Nat.floor (0.75 * 60))
  (h2 : y = 60 + (x^2 - 10)) :
  x = 45 ∧ y = 2075 := by
  have h_e : 60 = 60 := by rfl
  have h_x : x = 45 := by
    sorry
  have h_y : y = 2075 := by
    subst h_x
    sorry
  exact ⟨h_x, h_y⟩

end token_problem_l818_818007


namespace unique_nat_number_sum_preceding_eq_self_l818_818689

theorem unique_nat_number_sum_preceding_eq_self :
  ∃! (n : ℕ), (n * (n - 1)) / 2 = n :=
sorry

end unique_nat_number_sum_preceding_eq_self_l818_818689


namespace subtracted_complex_eq_l818_818542

open Complex

theorem subtracted_complex_eq {z : ℂ} (h : Complex.i ^ 2 = -1) :
  (5 + 3 * Complex.i - z = -1 + 4 * Complex.i) → z = 6 - Complex.i :=
by
  intro h₁
  sorry

end subtracted_complex_eq_l818_818542


namespace vegetable_options_l818_818640

open Nat

theorem vegetable_options (V : ℕ) : 
  3 * V + 6 = 57 → V = 5 :=
by
  intro h
  sorry

end vegetable_options_l818_818640


namespace watermelon_weight_l818_818966

theorem watermelon_weight (B W : ℝ) (n : ℝ) 
  (h1 : B + n * W = 63) 
  (h2 : B + (n / 2) * W = 34) : 
  n * W = 58 :=
sorry

end watermelon_weight_l818_818966


namespace magic_box_problem_l818_818441

theorem magic_box_problem (m : ℝ) :
  (m^2 - 2*m - 1 = 2) → (m = 3 ∨ m = -1) :=
by
  intro h
  sorry

end magic_box_problem_l818_818441


namespace root_of_quadratic_eqn_l818_818700

theorem root_of_quadratic_eqn (v : ℝ) (root : ℝ) (hroot : root = (-25 + real.sqrt 361) / 12)
    (h : 6 * root^2 + 25 * root + v = 0) : v = 11 := sorry

end root_of_quadratic_eqn_l818_818700


namespace angle_A_range_BD_CD_l818_818388

variable (A B C : ℝ) (a b c BD CD : ℝ)
variable [h_acute : A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2]
variable [h_opposite : A = Real.arcsin(a / (a / (sin A))) ∧ B = Real.arcsin(b / (b / (sin B))) ∧ C = Real.arcsin(c / (c / (sin C)))]
variable [h_identity : sin(C)^2 + sin(B)^2 - sin(A)^2 = sin(B) * sin(C)]
variable [h_bisector : BD / CD = (a * sin B) / (c * sin C)]

theorem angle_A : A = Real.pi / 3 :=
by
  sorry

theorem range_BD_CD : 1/2 < BD / CD ∧ BD / CD < 2 :=
by
  sorry

end angle_A_range_BD_CD_l818_818388


namespace grasshopper_jump_unoccupied_cells_l818_818795

theorem grasshopper_jump_unoccupied_cells :
  ∀ (n : ℕ), n = 10 →
  ∃ (empty_cells : ℕ), empty_cells ≥ 20 ∧ 
  (∀ (board : fin n → fin n → bool), 
    (∀ (i j : fin n), board i j = tt) →
    (∀ (initial_grasshoppers : fin n → fin n → bool),
      (∀ (i j : fin n), initial_grasshoppers i j = tt) →
      let final_grasshoppers := λ i j, if even (i + j) then initial_grasshoppers (i + 2) (j + 2) else initial_grasshoppers (i - 2) (j - 2)
      in ∀ i j, board i j = ff ↔ initial_grasshoppers i j = tt → ¬ final_grasshoppers i j = tt
    )
  )
  := sorry

end grasshopper_jump_unoccupied_cells_l818_818795


namespace number_of_fir_trees_is_11_l818_818270

theorem number_of_fir_trees_is_11 
  (N : ℕ)
  (Anya : N = 15)
  (Borya : N % 11 = 0)
  (Vera : N < 25)
  (Gena : N % 22 = 0)
  (OneBoyOneGirlTrue : (Anya ∨ Borya) ∧ (Vera ∨ Gena) ∧ (¬Anya ∨ ¬Borya) ∧ (¬Vera ∨ ¬Gena)) :
  N = 11 := 
sorry

end number_of_fir_trees_is_11_l818_818270


namespace f_range_x_range_l818_818717

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 2 - 3) * (1 + Real.log x / Real.log 2)
noncomputable def g (x : ℝ) : ℝ := 4^x - 2^(x + 1) - 3

theorem f_range : ∀ y ∈ Set.Icc (-4:ℝ) (Real.infinity), ∃ x : ℝ, f x = y := 
sorry

theorem x_range (x : ℝ) : 
  (∀ a ∈ Set.Icc (1 / 2) 2, f x - g a ≤ 0) ↔ x ∈ Set.Icc (2^(2 - Real.sqrt 2)) (2^(Real.sqrt 2)) :=
sorry

end f_range_x_range_l818_818717


namespace average_weight_of_class_l818_818155

theorem average_weight_of_class (students_a students_b : ℕ) (avg_weight_a avg_weight_b : ℝ)
  (h_students_a : students_a = 24)
  (h_students_b : students_b = 16)
  (h_avg_weight_a : avg_weight_a = 40)
  (h_avg_weight_b : avg_weight_b = 35) :
  ((students_a * avg_weight_a + students_b * avg_weight_b) / (students_a + students_b)) = 38 := 
by
  sorry

end average_weight_of_class_l818_818155


namespace find_a_and_b_l818_818349

-- Define the variables and conditions
variable {a b : ℝ}

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℝ := ![![1, a], ![-1, b]]

-- Define the eigenvalue and eigenvector
def λ : ℝ := 2
def α : Fin 2 → ℝ := ![2, 1]

-- The theorem to prove
theorem find_a_and_b (h : A.mulVec α = λ • α) : a = 2 ∧ b = 4 :=
by
  matrix.realize_matrix
  simp at h
  exact ⟨sorry, sorry⟩

end find_a_and_b_l818_818349


namespace fir_trees_alley_l818_818284

-- Define the statements made by each child
def statementAnya (N : ℕ) : Prop := N = 15
def statementBorya (N : ℕ) : Prop := N % 11 = 0
def statementVera (N : ℕ) : Prop := N < 25
def statementGena (N : ℕ) : Prop := N % 22 = 0

-- Define the condition about the truth and lies
def oneBoyOneGirlTruth (anya_vera_truth: Prop) (borya_gena_truth: Prop) : Prop :=
  anya_vera_truth ∧ borya_gena_truth ∧
  ((statementAnya N ∧ statementVera N) ∨ (statementVera N ∧ statementBorya N)) ∧
  ¬( (statementAnya N ∧ statementGena N) ∨ (statementVera N ∧ statementGena N) ∨
     (statementAnya N ∧ statementBorya N) ∨ (statementBorya N ∧ statementGena N) )

-- Prove that the number of fir trees is 11
theorem fir_trees_alley: ∃ (N : ℕ), statementBorya N ∧ statementVera N ∧ ¬ statementAnya N ∧ ¬ statementGena N ∧ oneBoyOneGirlTruth (¬ statementAnya N ∧ statementVera N) (statementBorya N ∧ ¬ statementGena N) ∧ N = 11 :=
by
  sorry

end fir_trees_alley_l818_818284


namespace sufficient_but_not_necessary_α__l818_818549

variables {α β : Set Point}
variables {a b m : Set Point}
variables (a_in_α : a ⊆ α) (b_in_β : b ⊆ β) (m_in_α_β : m ⊆ α ∩ β) (b_⟂_m : ∀ p ∈ b, ∀ q ∈ m, ⟂ (Line.mk p q))

theorem sufficient_but_not_necessary_α_⟂_β_⟹_a_⟂_b :
  (∀ p ∈ a, ∀ q ∈ b, ⟂ (Line.mk p q)) → (∀ p ∈ α, ∀ q ∈ β, ⟂ (Plane.mk p q)) :=
sorry

end sufficient_but_not_necessary_α__l818_818549


namespace sqrt_x_plus_inv_sqrt_x_eq_sqrt_152_l818_818004

-- Conditions
variable (x : ℝ) (h₀ : 0 < x) (h₁ : x + 1 / x = 150)

-- Statement to prove
theorem sqrt_x_plus_inv_sqrt_x_eq_sqrt_152 : (Real.sqrt x + Real.sqrt (1 / x) = Real.sqrt 152) := 
sorry -- Proof not needed, skip with sorry

end sqrt_x_plus_inv_sqrt_x_eq_sqrt_152_l818_818004


namespace geom_seq_sum_trig_exp_value_l818_818161

-- Problem 1: Sum of the first n terms of the geometric sequence
theorem geom_seq_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) 
  (h₀ : a 1 = 2) (h₁ : a 1 + a 2 = 6) : S_n = 2^(n+1) - 2 := 
sorry

-- Problem 2: Value of the given trigonometric expression
theorem trig_exp_value (θ : ℝ) 
  (h₀ : tan θ = 3) : 
  (2 * cos^2 (θ / 2) + sin θ - 1) / (sin θ - cos θ) = 2 := 
sorry

end geom_seq_sum_trig_exp_value_l818_818161


namespace smallest_n_l818_818001

theorem smallest_n
  (n : ℕ)
  (y : Fin n → ℝ)
  (h1 : ∀ i, |y i| < 1)
  (h2 : (∑ i, |y i|) = 15 + |∑ i, y i|) :
  n = 16 := sorry

end smallest_n_l818_818001


namespace example_problem_l818_818145

def is_term (t : String) : Prop := t = "a" ∨ t = "-a" ∨ t = "3ab/5" ∨ t = "-2"

def is_polynomial (expr : String) : Prop :=
  ∀ t : String, t ∈ expr.split_on ' ' → is_term t

def not_polynomial_example : Prop :=
  "x + a/x + 1" = "x + a/x + 1" → ¬ is_polynomial "x + a/x + 1"

theorem example_problem : not_polynomial_example := sorry

end example_problem_l818_818145


namespace speed_first_32_miles_l818_818223

theorem speed_first_32_miles (x : ℝ) (y : ℝ) : 
  (100 / x + 0.52 * 100 / x = 32 / y + 68 / (x / 2)) → 
  y = 2 * x :=
by
  sorry

end speed_first_32_miles_l818_818223


namespace solve_trig_eq_l818_818876

open Real

theorem solve_trig_eq (k : ℤ) : 
  (∃ x : ℝ, 
    (|cos x| + cos (3 * x)) / (sin x * cos (2 * x)) = -2 * sqrt 3 
    ∧ (x = -π/6 + 2 * k * π ∨ x = 2 * π/3 + 2 * k * π ∨ x = 7 * π/6 + 2 * k * π)) :=
sorry

end solve_trig_eq_l818_818876


namespace sequence_periodicity_l818_818356

variable {a b : ℕ → ℤ}

theorem sequence_periodicity (h : ∀ n ≥ 3, 
    (a n - a (n - 1)) * (a n - a (n - 2)) + 
    (b n - b (n - 1)) * (b n - b (n - 2)) = 0) : 
    ∃ k > 0, a k + b k = a (k + 2018) + b (k + 2018) := 
    by
    sorry

end sequence_periodicity_l818_818356


namespace ratio_AC_AD_of_parallelogram_inscribed_quadrilaterals_l818_818908

theorem ratio_AC_AD_of_parallelogram_inscribed_quadrilaterals
  (A B C D K L M : Point)
  (h_parallelogram : parallelogram A B C D)
  (h_midpoints : midpoint K A B ∧ midpoint L B C ∧ midpoint M C D)
  (h_inscribed_KBLM : inscribed_quad K B L M)
  (h_inscribed_BCDK : inscribed_quad B C D K) :
  ratio_of_distances (distance A C) (distance A D) = 2 :=
by
  sorry

end ratio_AC_AD_of_parallelogram_inscribed_quadrilaterals_l818_818908


namespace range_of_a_l818_818053

def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def increasing_on_negative (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → y ≤ 0 → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (ha : even_function f) (hb : increasing_on_negative f) 
  (hc : ∀ a : ℝ, f a ≤ f (2 - a)) : ∀ a : ℝ, a < 1 → false :=
by
  sorry

end range_of_a_l818_818053


namespace common_sum_zero_l818_818499

-- Definitions for the conditions
def set_of_integers : Set ℤ := {n | -12 ≤ n ∧ n ≤ 12}
def is_5x5_rectangle (arr : List (List ℤ)) : Prop :=
  List.length arr = 5 ∧ ∀ row ∈ arr, List.length row = 5 ∧ (∀ n ∈ row, n ∈ set_of_integers)

-- Statement of the problem in Lean 4
theorem common_sum_zero (arr : List (List ℤ)) (h : is_5x5_rectangle arr) :
  (∀ r ∈ arr, List.sum r = 0) ∧ (∀ j < 5, List.sum (arr.map (List.nth' · j)) = 0) :=
sorry

end common_sum_zero_l818_818499


namespace mod_6_computation_l818_818620

theorem mod_6_computation (a b n : ℕ) (h₁ : a ≡ 35 [MOD 6]) (h₂ : b ≡ 16 [MOD 6]) (h₃ : n = 1723) :
  (a ^ n - b ^ n) % 6 = 1 :=
by 
  -- proofs go here
  sorry

end mod_6_computation_l818_818620


namespace least_pos_int_with_six_factors_l818_818102

theorem least_pos_int_with_six_factors :
  ∃ n : ℕ, (∀ m : ℕ, (number_of_factors m = 6 → m ≥ n)) ∧ n = 12 := 
sorry

end least_pos_int_with_six_factors_l818_818102


namespace train_crossing_time_correct_l818_818190

noncomputable def train_crossing_time : ℝ :=
  let speed1_kmh : ℝ := 132
  let length1_m : ℝ := 110
  let speed2_kmh : ℝ := 96
  let length2_m : ℝ := 165
  -- Calculate relative speed in meters per second
  let relative_speed_mps : ℝ := (speed1_kmh + speed2_kmh) * 1000 / 3600
  -- Calculate total distance to be covered in meters
  let total_distance_m : ℝ := length1_m + length2_m
  -- Calculate time in seconds
  total_distance_m / relative_speed_mps

theorem train_crossing_time_correct : train_crossing_time ≈ 4.34 :=
by
  sorry

end train_crossing_time_correct_l818_818190


namespace nancy_coffee_expense_l818_818854

-- Definitions corresponding to the conditions
def cost_double_espresso : ℝ := 3.00
def cost_iced_coffee : ℝ := 2.50
def days : ℕ := 20

-- The statement of the problem
theorem nancy_coffee_expense :
  (days * (cost_double_espresso + cost_iced_coffee)) = 110.00 := by
  sorry

end nancy_coffee_expense_l818_818854


namespace lucas_fib_relation_l818_818944

noncomputable def α := (1 + Real.sqrt 5) / 2
noncomputable def β := (1 - Real.sqrt 5) / 2
def Fib : ℕ → ℝ
| 0       => 0
| 1       => 1
| (n + 2) => Fib n + Fib (n + 1)

def Lucas : ℕ → ℝ
| 0       => 2
| 1       => 1
| (n + 2) => Lucas n + Lucas (n + 1)

theorem lucas_fib_relation (n : ℕ) (hn : 1 ≤ n) :
  Lucas (2 * n + 1) + (-1)^(n+1) = Fib (2 * n) * Fib (2 * n + 1) := sorry

end lucas_fib_relation_l818_818944


namespace harriet_current_age_l818_818391

theorem harriet_current_age (peter_age harriet_age : ℕ) (mother_age : ℕ := 60) (h₁ : peter_age = mother_age / 2) 
  (h₂ : peter_age + 4 = 2 * (harriet_age + 4)) : harriet_age = 13 :=
by
  sorry

end harriet_current_age_l818_818391


namespace power_series_expansion_l818_818023

theorem power_series_expansion (x : ℝ) (n : ℕ) (h : |x| < 1) :
  (1 / (1 - x))^n = 1 + ∑ k in (Finset.range n) , (binomial (n + k - 1) (n - 1)) * x^k :=
sorry

end power_series_expansion_l818_818023


namespace circles_tangent_l818_818650

theorem circles_tangent
  (rA rB rC rD rF : ℝ) (rE : ℚ) (m n : ℕ)
  (m_n_rel_prime : Int.gcd m n = 1)
  (rA_pos : 0 < rA) (rB_pos : 0 < rB)
  (rC_pos : 0 < rC) (rD_pos : 0 < rD)
  (rF_pos : 0 < rF)
  (inscribed_triangle_in_A : True)  -- Triangle T is inscribed in circle A
  (B_tangent_A : True)  -- Circle B is internally tangent to circle A
  (C_tangent_A : True)  -- Circle C is internally tangent to circle A
  (D_tangent_A : True)  -- Circle D is internally tangent to circle A
  (B_externally_tangent_E : True)  -- Circle B is externally tangent to circle E
  (C_externally_tangent_E : True)  -- Circle C is externally tangent to circle E
  (D_externally_tangent_E : True)  -- Circle D is externally tangent to circle E
  (F_tangent_A : True)  -- Circle F is internally tangent to circle A at midpoint of side opposite to B's tangency
  (F_externally_tangent_E : True)  -- Circle F is externally tangent to circle E
  (rA_eq : rA = 12) (rB_eq : rB = 5)
  (rC_eq : rC = 3) (rD_eq : rD = 2)
  (rF_eq : rF = 1)
  (rE_eq : rE = m / n)
  : m + n = 23 :=
by
  sorry

end circles_tangent_l818_818650


namespace smallest_six_factors_l818_818124

theorem smallest_six_factors (n : ℕ) (h : (n = 2 * 3^2)) : n = 18 :=
by {
    sorry -- proof goes here
}

end smallest_six_factors_l818_818124


namespace max_min_product_l818_818000

theorem max_min_product (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h1 : x + y + z = 15) (h2 : x * y + y * z + z * x = 45) :
    ∃ m : ℝ, m = min (x * y) (min (y * z) (z * x)) ∧ m ≤ 17.5 :=
by
  sorry

end max_min_product_l818_818000


namespace inscribed_circle_radius_in_quarter_circle_l818_818473

theorem inscribed_circle_radius_in_quarter_circle (R r : ℝ) (hR : R = 4) :
  (r + r * Real.sqrt 2 = R) ↔ r = 4 * Real.sqrt 2 - 4 := by
  sorry

end inscribed_circle_radius_in_quarter_circle_l818_818473


namespace equation_solution_l818_818370

theorem equation_solution (x : ℝ) (h : 8^(Real.log 5 / Real.log 8) = 10 * x + 3) : x = 1 / 5 :=
sorry

end equation_solution_l818_818370


namespace evaluate_expression_l818_818031

-- Definition of the given expression
def expression (x y : ℝ) : ℝ :=
  ((x - 2 * y) ^ 2 - (2 * x + y) * (x - 4 * y) - (-x + 3 * y) * (x + 3 * y)) / -y

-- Conditions
def x_val : ℝ := -1/3
def y_val : ℝ := -1

-- Statement to prove
theorem evaluate_expression : expression x_val y_val = 0 :=
by
  sorry

end evaluate_expression_l818_818031


namespace possible_values_of_a_plus_b_l818_818730

theorem possible_values_of_a_plus_b (a b : ℤ)
  (h1 : ∃ α : ℝ, 0 ≤ α ∧ α < 2 * Real.pi ∧ (∃ (sinα cosα : ℝ), sinα = Real.sin α ∧ cosα = Real.cos α ∧ (sinα + cosα = -a) ∧ (sinα * cosα = 2 * b^2))) :
  a + b = 1 ∨ a + b = -1 := 
sorry

end possible_values_of_a_plus_b_l818_818730


namespace remainder_of_product_l818_818080

theorem remainder_of_product (a b c : ℕ) (h₁ : a % 7 = 3) (h₂ : b % 7 = 4) (h₃ : c % 7 = 5) :
  (a * b * c) % 7 = 4 :=
by
  sorry

end remainder_of_product_l818_818080


namespace midpoint_harry_sandy_l818_818362

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem midpoint_harry_sandy :
  midpoint (15, -8) (-5, 12) = (5, 2) := by
  sorry

end midpoint_harry_sandy_l818_818362


namespace ellipse_eccentricity_l818_818334

open Real

def ellipse (x y k : ℝ) : Prop :=
  x^2 + k * y^2 = 2 * k ∧ k > 0

def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

theorem ellipse_eccentricity {x y k : ℝ}
  (h_ellipse : ellipse x y k)
  (h_focus: (1, 0) = (x, y)) :
  let a := sqrt 3 in
  let c := 1 in
  let e := c / a in
  e = sqrt 3 / 3 :=
by
  sorry

end ellipse_eccentricity_l818_818334


namespace tire_miles_used_l818_818073

theorem tire_miles_used (total_miles : ℕ) (number_of_tires : ℕ) (tires_in_use : ℕ)
  (h_total_miles : total_miles = 40000) (h_number_of_tires : number_of_tires = 6)
  (h_tires_in_use : tires_in_use = 4) : 
  (total_miles * tires_in_use) / number_of_tires = 26667 := 
by 
  sorry

end tire_miles_used_l818_818073


namespace average_age_across_rooms_l818_818784

theorem average_age_across_rooms :
  let room_a_people := 8
  let room_a_average_age := 35
  let room_b_people := 5
  let room_b_average_age := 30
  let room_c_people := 7
  let room_c_average_age := 25
  let total_people := room_a_people + room_b_people + room_c_people
  let total_age := (room_a_people * room_a_average_age) + (room_b_people * room_b_average_age) + (room_c_people * room_c_average_age)
  let average_age := total_age / total_people
  average_age = 30.25 := by
{
  sorry
}

end average_age_across_rooms_l818_818784


namespace five_digit_unique_count_l818_818703

theorem five_digit_unique_count :
  let count1 := Nat.choose 5 3 in
  let count2 := Nat.choose 4 2 in
  let arr := 5.factorial in
  count1 * count2 * arr = 7200 := 
by 
  sorry

end five_digit_unique_count_l818_818703


namespace angle_bisectors_A_eq_60_l818_818783

theorem angle_bisectors_A_eq_60
  (A B C D E O : Type)
  [triangle : is_triangle A B C]
  (bisector_B_C : bisects_angle B C A D E O)
  (OD_EQ_OE : OD = OE)
  (AD_NEQ_AE : AD ≠ AE) :
  angle A = 60 :=
sorry

end angle_bisectors_A_eq_60_l818_818783


namespace max_value_output_l818_818337

theorem max_value_output (a b c : ℝ) (h_a : a = 3) (h_b : b = 7) (h_c : c = 2) : max (max a b) c = 7 := 
by
  sorry

end max_value_output_l818_818337


namespace remainder_s100_mod5_l818_818498

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 1 then 7 else 7 ^ sequence (n - 1)

theorem remainder_s100_mod5 : (sequence 100) % 5 = 3 := by
  sorry

end remainder_s100_mod5_l818_818498


namespace simplify_sqrt_expression_l818_818768

theorem simplify_sqrt_expression (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 4) :
  sqrt (1 - 2 * sin (π + θ) * sin (3 * π / 2 - θ)) = cos θ - sin θ := 
sorry

end simplify_sqrt_expression_l818_818768


namespace ratio_of_shaded_to_white_area_l818_818949

theorem ratio_of_shaded_to_white_area :
  let vertices_at_midpoints := ∀ (n : ℕ), (vertex_positions n ⊆ midpoints (sides n))
  let ratio_of_areas := fraction_of_areas (shaded_area total_area) (white_area total_area)
  in vertices_at_midpoints n → ratio_of_areas = 5 / 3 :=
begin
  sorry
end

end ratio_of_shaded_to_white_area_l818_818949


namespace least_positive_integer_with_six_distinct_factors_l818_818096

theorem least_positive_integer_with_six_distinct_factors : ∃ n : ℕ, (∀ k : ℕ, (number_of_factors k = 6) → (n ≤ k)) ∧ (number_of_factors n = 6) ∧ (n = 12) :=
by
  sorry

end least_positive_integer_with_six_distinct_factors_l818_818096


namespace max_value_of_a_plus_b_l818_818975

theorem max_value_of_a_plus_b (a b : ℕ) 
  (h : 5 * a + 19 * b = 213) : a + b ≤ 37 :=
  sorry

end max_value_of_a_plus_b_l818_818975


namespace fir_trees_alley_l818_818282

-- Define the statements made by each child
def statementAnya (N : ℕ) : Prop := N = 15
def statementBorya (N : ℕ) : Prop := N % 11 = 0
def statementVera (N : ℕ) : Prop := N < 25
def statementGena (N : ℕ) : Prop := N % 22 = 0

-- Define the condition about the truth and lies
def oneBoyOneGirlTruth (anya_vera_truth: Prop) (borya_gena_truth: Prop) : Prop :=
  anya_vera_truth ∧ borya_gena_truth ∧
  ((statementAnya N ∧ statementVera N) ∨ (statementVera N ∧ statementBorya N)) ∧
  ¬( (statementAnya N ∧ statementGena N) ∨ (statementVera N ∧ statementGena N) ∨
     (statementAnya N ∧ statementBorya N) ∨ (statementBorya N ∧ statementGena N) )

-- Prove that the number of fir trees is 11
theorem fir_trees_alley: ∃ (N : ℕ), statementBorya N ∧ statementVera N ∧ ¬ statementAnya N ∧ ¬ statementGena N ∧ oneBoyOneGirlTruth (¬ statementAnya N ∧ statementVera N) (statementBorya N ∧ ¬ statementGena N) ∧ N = 11 :=
by
  sorry

end fir_trees_alley_l818_818282


namespace _l818_818668

noncomputable theorem permutation_identity {a : Fin 2021 → ℕ} (h_perm : ∀ i, a i ∈ Finset.range 2021)
  (h_ineq : ∀ m n : ℕ, |m - n| > 20^21 → (Finset.sum (Finset.range 2021) (λ i, Nat.gcd (m + i) (n + a i)) < 2 * |m - n|)) :
  ∀ i, a i = i :=
by
  sorry

end _l818_818668


namespace cost_per_minute_l818_818240

theorem cost_per_minute (monthly_fee total_bill billed_minutes : ℝ) (h_monthly_fee : monthly_fee = 2) (h_total_bill : total_bill = 23.36) (h_billed_minutes : billed_minutes = 178) : 
  (total_bill - monthly_fee) / billed_minutes = 0.12 :=
by
  rw [h_monthly_fee, h_total_bill, h_billed_minutes]
  norm_num
  sorry

end cost_per_minute_l818_818240


namespace petya_passwords_l818_818458

theorem petya_passwords : 
  let digits := {0, 1, 2, 3, 4, 5, 6, 8, 9} in
  let is_password (password : Fin 4 → ℕ) : Prop := ∀ i, password i ∈ digits in
  let has_at_least_two_identical (password : Fin 4 → ℕ) : Prop := ∃ i j, i ≠ j ∧ password i = password j in
  (Fin 4 → ℕ) → 
  (∀ password, is_password password → has_at_least_two_identical password) = 3537 :=
by sorry

end petya_passwords_l818_818458


namespace projection_is_sqrt_13_l818_818301

-- Definitions of vectors a and b
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-4, 7)

-- Function to calculate the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Function to calculate the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Calculate the projection of b on a
def projection_of_b_on_a : ℝ :=
  (magnitude b) * (dot_product a b) / (magnitude a * magnitude b)

-- Theorem that states the projection of b on a is √13
theorem projection_is_sqrt_13 : projection_of_b_on_a = real.sqrt 13 := by 
  sorry

end projection_is_sqrt_13_l818_818301


namespace least_positive_integer_with_six_factors_is_18_l818_818135

-- Define the least positive integer with exactly six distinct positive factors.
def least_positive_with_six_factors (n : ℕ) : Prop :=
  (∀ d : ℕ, d ∣ n → d > 0) ∧ (finset.card (finset.filter (λ d, d ∣ n) (finset.range (n + 1)))) = 6

-- Prove that the least positive integer with exactly six distinct positive factors is 18.
theorem least_positive_integer_with_six_factors_is_18 : (∃ n : ℕ, least_positive_with_six_factors n ∧ n = 18) :=
sorry


end least_positive_integer_with_six_factors_is_18_l818_818135


namespace eliza_iron_total_l818_818648

-- Definition of the problem conditions in Lean
def blouse_time := 15 -- time to iron a blouse in minutes
def dress_time := 20 -- time to iron a dress in minutes
def blouse_hours := 2 -- hours spent ironing blouses
def dress_hours := 3 -- hours spent ironing dresses

-- Definition to convert hours to minutes
def hours_to_minutes (hours: Int) : Int :=
  hours * 60

-- Definition of the total number of pieces of clothes ironed by Eliza
def total_pieces_iron (blouse_time dress_time blouse_hours dress_hours: Int) : Int :=
  let blouses := hours_to_minutes(blouse_hours) / blouse_time
  let dresses := hours_to_minutes(dress_hours) / dress_time
  blouses + dresses

-- The proof statement
theorem eliza_iron_total : total_pieces_iron blouse_time dress_time blouse_hours dress_hours = 17 :=
by 
  -- To be filled in with the actual proof
  sorry

end eliza_iron_total_l818_818648


namespace incorrect_statement_C_l818_818860

noncomputable def weekly_hours : List ℝ := [6.5, 6.3, 7.8, 9.2, 5.7, 7.9, 8.1, 7.2, 5.8, 8.3]

def average (l : List ℝ) : ℝ :=
  l.sum / l.length

def median (l : List ℝ) : ℝ :=
  let sorted := l.qsort (· ≤ ·)
  if sorted.length % 2 = 0 then
    (sorted.get (sorted.length / 2 - 1) + sorted.get (sorted.length / 2)) / 2
  else
    sorted.get (sorted.length / 2)

def exceeds_8_hours_probability (l : List ℝ) : ℝ :=
  let count := l.countp (λ x => x > 8)
  count / l.length

def range (l : List ℝ) : ℝ :=
  let sorted := l.qsort (· ≤ ·)
  (sorted.get (sorted.length - 1) - sorted.get 0)

-- Problem Statement
theorem incorrect_statement_C :
  let avg_weekly := average weekly_hours
  let avg_daily := avg_weekly / 5
  let med := median weekly_hours
  let prob := exceeds_8_hours_probability weekly_hours
  let orig_range := range weekly_hours
  let new_data := weekly_hours.map (· + 0.5)
  let new_range := range new_data
  med ≠ 6.8 ∧ avg_daily ≥ 1 ∧ prob = 0.3 ∧ new_range = orig_range :=
by
  sorry

end incorrect_statement_C_l818_818860


namespace sum_of_digits_of_second_smallest_multiple_l818_818423

def isDivisibleByAll (n : ℕ) (range : List ℕ) : Prop :=
  ∀ d ∈ range, d > 0 → n % d = 0

def secondSmallestMultiple (n : ℕ) (l : List ℕ) : ℕ :=
  let lcm := l.foldl Nat.lcm 1
  2 * lcm

def sumOfDigits (n : ℕ) : ℕ :=
  n.to_digits.sum

theorem sum_of_digits_of_second_smallest_multiple : 
  let M := secondSmallestMultiple 8 [1, 2, 3, 4, 5, 6, 7] in
  M = 840 → sumOfDigits M = 12 :=
by
  sorry

end sum_of_digits_of_second_smallest_multiple_l818_818423


namespace triangle_rotation_sum_eq_120_l818_818083

theorem triangle_rotation_sum_eq_120 :
  ∃ (n p q : ℝ), 
    0 < n ∧ n < 180 ∧
    (rotate_point (0, 0) n (p, q) = (40, 30)) ∧
    (rotate_point (0, 20) n (p, q) = (52, 30)) ∧
    (rotate_point (30, 0) n (p, q) = (40, 10)) ∧
    n + p + q = 120 :=
by
  sorry

end triangle_rotation_sum_eq_120_l818_818083


namespace centroid_formula_l818_818865

-- Definitions and setup
variable {O A B C Q : ℝ^3}

-- Assume Q is the centroid of triangle ABC.
axiom centroid_definition (h1 : Q = 1/3 • (A + B + C)) : True

-- The theorem we want to prove
theorem centroid_formula (h1 : Q = 1/3 • (A + B + C)) : Q = 1/3 • (A + B + C) :=
by {
  -- proof goes here
  sorry
}

end centroid_formula_l818_818865


namespace least_positive_integer_with_six_factors_l818_818106

theorem least_positive_integer_with_six_factors :
  ∃ n : ℕ, (∀ m : ℕ, m > 0 → m < n → (count_factors m ≠ 6)) ∧ count_factors n = 6 ∧ n = 18 :=
sorry

noncomputable def count_factors (n : ℕ) : ℕ :=
sorry

end least_positive_integer_with_six_factors_l818_818106


namespace geometric_sequence_common_ratio_l818_818585

noncomputable def common_ratio (a1 a2 : ℤ) : ℤ := a2 / a1

theorem geometric_sequence_common_ratio :
  let a1 := 10
  let a2 := -20
  let a3 := 40
  let a4 := -80
  r = -2 := 
  (common_ratio a1 a2 = -2) ∧ 
  (common_ratio a2 a3 = -2) ∧ 
  (common_ratio a3 a4 = -2) := 
by
  sorry

end geometric_sequence_common_ratio_l818_818585


namespace trajectory_eq_ellipse_l818_818313

theorem trajectory_eq_ellipse :
  (∀ M : ℝ × ℝ, (∀ r : ℝ, (M.1 - 4)^2 + M.2^2 = r^2 ∧ (M.1 + 4)^2 + M.2^2 = (10 - r)^2) → false) →
  ∀ x y : ℝ, (x^2 / 25 + y^2 / 9 = 1) :=
by
  sorry

end trajectory_eq_ellipse_l818_818313


namespace shortest_routes_A_to_B_l818_818051

-- We don't need to define the specific map or regions C and D since they don't impact
-- the path counting directly. The main idea here is the grid structure and the path counting.

-- Assume the grid dimensions if necessary, but since it wasn't provided explicitly,
-- we will focus on the path counting.

theorem shortest_routes_A_to_B : 
  let A : ℕ := 0
  let B : ℕ := grid_width * grid_height
  let grid_width : ℕ := 10 -- Assume a 10km x 8km grid based on intermediate descriptions.
  let grid_height : ℕ := 8 
  (number_of_shortest_paths A B grid_width grid_height = 22023) :=
begin
  sorry
end

end shortest_routes_A_to_B_l818_818051


namespace difference_of_numbers_l818_818914

theorem difference_of_numbers (x y : ℝ) (h₁ : x + y = 25) (h₂ : x * y = 144) : |x - y| = 7 :=
sorry

end difference_of_numbers_l818_818914


namespace fixed_point_A_l818_818307

-- Definitions extracted from conditions
variable (O A : Point) (circle_O : Circle)
variable (d : Line) (A' : Point)

-- Given Conditions
axiom A_on_circle_O : A ∈ circle_O
axiom d_passes_through_O : O ∈ d
axiom A'_is_reflection : refl_point A d A'

-- Additional variables for rotating secant
variable (B D : Point)

-- Given that D is intersection of secant through A and line d
axiom D_on_line_d : D ∈ d

-- Given that B is intersection of secant through A and circle_O
axiom B_on_circle_O : B ∈ circle_O

-- Problem Statement: Prove that the circle through O, B, D passes through A'
theorem fixed_point_A' (O A A' B D : Point) 
(circle_O : Circle) (d : Line) 
(A_on_circle_O : A ∈ circle_O)
(d_passes_through_O : O ∈ d)
(A'_is_reflection : refl_point A d A')
(D_on_line_d : D ∈ d)
(B_on_circle_O : B ∈ circle_O) :
  touches (circumscribed O B D) A' := 
sorry

end fixed_point_A_l818_818307


namespace non_intersecting_lines_are_parallel_or_skew_l818_818987

-- Define the concept of 'lines in space' as variables
variable (L1 L2 : ℝ → ℝ × ℝ × ℝ)

-- Define a condition expressing that two lines do not intersect.
def lines_do_not_intersect (L1 L2 : ℝ → ℝ × ℝ × ℝ) : Prop :=
  ∀ t1 t2 : ℝ, L1 t1 ≠ L2 t2

-- State the main theorem:
theorem non_intersecting_lines_are_parallel_or_skew
  (L1 L2 : ℝ → ℝ × ℝ × ℝ)
  (h : lines_do_not_intersect L1 L2) :
  (∃ v1 v2 : ℝ × ℝ × ℝ, ((∀ t : ℝ, L1 t = (0, 0, 0) + t * v1) ∧ (∀ t : ℝ, L2 t = (0, 0, 0) + t * v2) ∧ (∃ t1 t2 : ℝ, L1 t1 ≠ L2 t2))) ∨ 
  (¬(∃ t : ℝ, ∃ a b : ℝ, L1 t = L2 a + (b * (L2 (a + 1) - L2 a)))) := 
sorry

end non_intersecting_lines_are_parallel_or_skew_l818_818987


namespace number_of_trees_is_eleven_l818_818255

variables (N : ℕ)

-- Conditions
def Anya (N : ℕ) := N = 15
def Borya (N : ℕ) := 11 ∣ N
def Vera (N : ℕ) := N < 25
def Gena (N : ℕ) := 22 ∣ N

axiom OneBoyOneGirlTruth :
  (∃ (b : Prop) (g : Prop),
    (b ∨ ¬ b) ∧ (g ∨ ¬ g) ∧
    ((b = (Borya N ∨ Gena N)) ∧ (g = (Anya N ∨ Vera N)) ∧
     (b ↔ ¬g) ∧
     ((Anya N ∨ ¬Vera N) ∨ (¬Anya N ∨ Vera N)) ∧
     (Anya N = (N = 15)) ∧
     (Borya N = (11 ∣ N)) ∧
     (Vera N = (N < 25)) ∧
     (Gena N = (22 ∣ N))))

theorem number_of_trees_is_eleven: N = 11 :=
sorry

end number_of_trees_is_eleven_l818_818255


namespace number_of_chemistry_students_l818_818199

def volleyball_team := Set Int
def physics_students := {x ∈ volleyball_team | true} -- Assume 15 players
def chemistry_students := {x ∈ volleyball_team | true}
def both_subjects_students := {x ∈ volleyball_team | true} -- Assume 10 players

axiom team_size : ∀ team : volleyball_team, Set.card team = 30
axiom physics_size : ∀ physics : physics_students, Set.card physics = 15
axiom both_subjects_size : ∀ both : both_subjects_students, Set.card both = 10
axiom at_least_one_subject : ∀ team : volleyball_team, physics_students ∪ chemistry_students = team

theorem number_of_chemistry_students : 
  ∃ (chemistry : Set Int), Set.card chemistry = 25 :=
by 
  sorry

end number_of_chemistry_students_l818_818199


namespace area_triangle_BDE_l818_818021

noncomputable def points_3D := ℝ × ℝ × ℝ

def dist (p q : points_3D) : ℝ :=
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 + (p.3 - q.3) ^ 2)

def right_angle_triangle (p q r : points_3D) : Prop :=
  dist p q ^ 2 + dist q r ^ 2 = dist p r ^ 2

def plane_parallel (l : points_3D × points_3D) (plane : points_3D × points_3D × points_3D) : Prop :=
  let d1 := l.2.3 - l.1.3
  let normal_vector := (plane.2.2 - plane.1.2, plane.2.3 - plane.1.3, 0)
  normal_vector.2 * d1 = 0 ∧ normal_vector.3 * d1 = 0 -- simplified condition for parallelism with Z-axis

theorem area_triangle_BDE :
  ∃ (A B C D E : points_3D), 
    dist A B = 3 ∧ dist B C = 4 ∧ 
    dist C D = 3 ∧ dist D E = 3 ∧ dist E A = 3 ∧ 
    right_angle_triangle A B C ∧ right_angle_triangle C D E ∧ right_angle_triangle D E A ∧ 
    plane_parallel (D, E) (A, B, C) ∧
    let BD := dist B D 
    let BE := dist B E
    (0.5 * BD * BE = 4.5) :=
by
  sorry

end area_triangle_BDE_l818_818021


namespace circle_equation_l818_818331

-- Definitions for the given conditions
def line1 (x y : ℝ) : Prop := x + y + 2 = 0
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def line2 (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def is_solution (x y : ℝ) : Prop := x^2 + y^2 - 6 * x - 6 * y - 16 = 0

-- Problem statement in Lean
theorem circle_equation : ∃ x y : ℝ, 
  (line1 x y ∧ circle1 x y ∧ line2 (x / 2) (x / 2)) → is_solution x y :=
sorry

end circle_equation_l818_818331


namespace overall_loss_percentage_l818_818597

theorem overall_loss_percentage :
  let CP_radio := 1500
  let SP_radio := 1335
  let CP_tv := 5500
  let SP_tv := 5050
  let CP_fridge := 12000
  let SP_fridge := 11400
  let TCP := CP_radio + CP_tv + CP_fridge
  let TSP := SP_radio + SP_tv + SP_fridge
  let total_loss := TCP - TSP
  let loss_percentage := (total_loss / TCP.toFloat) * 100
  abs (loss_percentage - 6.39) < 0.01 := 
by
  sorry

end overall_loss_percentage_l818_818597


namespace machine_part_masses_l818_818200

theorem machine_part_masses :
  ∃ (x y : ℝ), (y - 2 * x = 100) ∧ (875 / x - 900 / y = 3) ∧ (x = 175) ∧ (y = 450) :=
by {
  sorry
}

end machine_part_masses_l818_818200


namespace basic_properties_of_f_l818_818039

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b / x^2

theorem basic_properties_of_f : 
  ∀ (a b : ℝ), a > 0 → b > 0 →
  (∀ x, x ≠ 0 → f a b x = a * x^2 + b / x^2) ∧
  (f a b = λ x => a * x^2 + b / x^2) ∧
  (∀ x, x ≠ 0 → f a b x ∈ Icc (2 * sqrt (a * b)) ⊤) ∧
  (∀ x, f a b (-x) = f a b x) ∧
  (∀ x y, 0 < x → x < sqrt (b / a) → 0 < y → y < sqrt (b / a) → f a b x > f a b y → x < y) ∧
  (∀ x y, sqrt (b / a) < x → 0 < y → x < y → f a b x < f a b y ) :=
begin
  sorry
end

end basic_properties_of_f_l818_818039


namespace angle_reflection_l818_818402

open EuclideanGeometry

variables {A B C D M E : Point}

def on_trap (A B C D : Point) : Prop := (is_trapezoid A B C D)
def sum_base_eq_diagonal (A B C D : Point) : Prop := (dist A B + dist C D = dist B D)
def midpoint (M B C : Point) : Prop := M = midpoint_line B C
def reflection_point (E C D M : Point) : Prop := reflect C D M = E
def angle_eq (A E B A C D : Point) : Prop  := (angle A E B = angle A C D)

theorem angle_reflection :
  on_trap A B C D →
  sum_base_eq_diagonal A B C D →
  midpoint M B C →
  reflection_point E C D M →
  angle_eq A E B A C D :=
by
  sorry

end angle_reflection_l818_818402


namespace graph_shift_right_l818_818495

theorem graph_shift_right (x : ℝ) :
  (∀ ω > 0, 
    (∃ a : ℝ, ∀ n : ℤ, a + n * (π / 2) = (π / 6 - π / 2 * 2 + π / 2 * 2) / ω ∧ 
    f x + n * (π / 2) = g (x + π / 12))) →
  (∀ x, f x = g (x - π / 12)) :=
by
  sorry

def f (x : ℝ) : ℝ := sin (2 * x + π / 6)
def g (x : ℝ) : ℝ := sin (2 * x)

end graph_shift_right_l818_818495


namespace correct_statements_count_l818_818904

-- Define the four statements
def statement_1 : Prop := ¬(rectangular_prism_is_prism ∧ cube_is_prism)
def statement_2 : Prop := pentagonal_prism_all_edges_equal
def statement_3 : Prop := triangular_prism_all_base_edges_equal
def statement_4 : Prop := geometric_solid_is_polyhedron

-- Define the evaluation of each statement based on domain knowledge
def evaluate_statement_1 : Prop := ¬statement_1 -- A rectangular prism and a cube are prisms
def evaluate_statement_2 : Prop := statement_2  -- All five side edges of a pentagonal prism are equal
def evaluate_statement_3 : Prop := ¬statement_3 -- Not all three edges of the base of a triangular prism are equal
def evaluate_statement_4 : Prop := statement_4  -- A geometric solid formed by several planar polygons is called a polyhedron

-- Count the number of correct statements
def number_of_correct_statements : ℕ :=
  [evaluate_statement_1, evaluate_statement_2, evaluate_statement_3, evaluate_statement_4].count true

-- Proof problem: Prove that the number of correct statements is 2
theorem correct_statements_count : number_of_correct_statements = 2 := by
  sorry

end correct_statements_count_l818_818904


namespace cos2_a_plus_sin2_b_eq_one_l818_818435

variable {a b c : ℝ}

theorem cos2_a_plus_sin2_b_eq_one
  (h1 : Real.sin a = Real.cos b)
  (h2 : Real.sin b = Real.cos c)
  (h3 : Real.sin c = Real.cos a) :
  Real.cos a ^ 2 + Real.sin b ^ 2 = 1 := 
  sorry

end cos2_a_plus_sin2_b_eq_one_l818_818435


namespace smallest_six_factors_l818_818125

theorem smallest_six_factors (n : ℕ) (h : (n = 2 * 3^2)) : n = 18 :=
by {
    sorry -- proof goes here
}

end smallest_six_factors_l818_818125


namespace tram_speed_l818_818448

theorem tram_speed
  (L v : ℝ)
  (h1 : L = 2 * v)
  (h2 : 96 + L = 10 * v) :
  v = 12 := 
by sorry

end tram_speed_l818_818448


namespace puppies_adopted_l818_818814

theorem puppies_adopted (p : ℕ) : (2 * 50 + 3 * 100 + p * 150 = 700) → p = 2 :=
by
  intro h
  have h1 : 100 + 300 + 150 * p = 700 := by rw [mul_comm, mul_comm, ← add_assoc] at h
  sorry

end puppies_adopted_l818_818814


namespace afternoon_shells_eq_l818_818439

def morning_shells : ℕ := 292
def total_shells : ℕ := 616

theorem afternoon_shells_eq :
  total_shells - morning_shells = 324 := by
  sorry

end afternoon_shells_eq_l818_818439


namespace positive_integer_solutions_l818_818363

theorem positive_integer_solutions (n : ℕ) : 
  {p : ℕ × ℕ // p.1 * p.2 = 10^n}.card = (n + 1)^2 :=
by
  sorry

end positive_integer_solutions_l818_818363


namespace min_kings_8x8_min_kings_nxn_l818_818150

-- Defining the minimum number of kings required for an 8x8 chessboard
theorem min_kings_8x8 : ∃ k : ℕ, k = 9 ∧ (∀ (P : ℕ → ℕ → Prop), (∀ i j, P i j) → (Σ k, ∀ i j, i ≤ 8 ∧ j ≤ 8 → P i j)) :=
sorry

-- Defining the minimum number of kings required for an n x n chessboard
theorem min_kings_nxn (n : ℕ) : ∃ k : ℕ, k = (⌈(n + 2) / 3⌉ : ℕ) ^ 2 ∧ (∀ (P : ℕ → ℕ → Prop), (∀ i j, P i j) → (Σ k, ∀ i j, i ≤ n ∧ j ≤ n → P i j)) :=
sorry

end min_kings_8x8_min_kings_nxn_l818_818150


namespace distance_to_midpoint_l818_818796

-- Define the isosceles triangle with given side lengths
structure IsoscelesTriangle :=
  (D E F : Point)
  (DE DF : ℝ)
  (EF : ℝ)
  (hDE : DE = DF)
  (hDEF : EF = 10)
  (hDEval : DE = 13)

open IsoscelesTriangle

-- Midpoint of the segment EF
def midpoint (p1 p2 : Point) : Point :=
  ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩

-- Prove the distance from point D to the midpoint of segment EF
theorem distance_to_midpoint (Δ : IsoscelesTriangle) :
  let M := midpoint Δ.E Δ.F
  sqrt ((Δ.D.x - M.x)^2 + (Δ.D.y - M.y)^2) = 12 :=
by
  sorry

end distance_to_midpoint_l818_818796


namespace minimum_sum_of_square_roots_l818_818315

variable {α : Type*} [LinearOrder α] [LinearOrderedField α]
variable (A : Finset α) -- The set of positive integers
variable (a_i : α) -- A typical element in set A
variable {n : ℕ} -- The number of elements in set A

theorem minimum_sum_of_square_roots :
  (∀ s1 s2 : Finset α, s1 ≠ s2 → (∑ x in s1, x) ≠ (∑ x in s2, x)) →
  ∑ x in A, x = n →
  ∑ x in A, Real.sqrt x = (Real.sqrt 2 + 1) * (Real.sqrt (2^n) - 1) :=
sorry

end minimum_sum_of_square_roots_l818_818315


namespace odd_decreasing_function_l818_818969

theorem odd_decreasing_function :
  (∃ (f : ℝ → ℝ), (f = λ x, sin (-x)) ∧ 
  (∀ x, f (-x) = -f x) ∧
  (∀ x ∈ set.Ioo (0 : ℝ) 1, deriv f x < 0)) ∧ 
  ¬ (∃ (f : ℝ → ℝ), f = sin ∧ ∀ x ∈ set.Ioo (0 : ℝ) 1, deriv f x < 0) ∧
  ¬ (∃ (f : ℝ → ℝ), f = tan ∧ ∀ x ∈ set.Ioo (0 : ℝ) 1, deriv f x < 0) ∧
  ¬ (∃ (f : ℝ → ℝ), f = cos ∧ ∀ x ∈ set.Ioo (0 : ℝ) 1, deriv f x < 0) := by
  sorry

end odd_decreasing_function_l818_818969


namespace average_sale_l818_818586

theorem average_sale (s1 s2 s3 s4 s5 s6 : ℝ) (h1 : s1 = 7435) (h2 : s2 = 7920) (h3 : s3 = 7855) (h4 : s4 = 8230) (h5 : s5 = 7560) (h6 : s6 = 6000) :
  (s1 + s2 + s3 + s4 + s5 + s6) / 6 = 7500 :=
by
  have H : s1 + s2 + s3 + s4 + s5 + s6 = 45000, {
    calc s1 + s2 + s3 + s4 + s5 + s6
    = 7435 + 7920 + 7855 + 8230 + 7560 + 6000 : by rw [h1, h2, h3, h4, h5, h6]
    ... = 45000 : by norm_num,
  },
  rw H,
  norm_num,

end average_sale_l818_818586


namespace number_of_fir_trees_l818_818291

def anya_statement (N : ℕ) : Prop := N = 15
def borya_statement (N : ℕ) : Prop := 11 ∣ N
def vera_statement (N : ℕ) : Prop := N < 25
def gena_statement (N : ℕ) : Prop := 22 ∣ N

def one_boy_one_girl_truth (A B G V : Prop) : Prop :=
  (A ∨ V) ∧ ¬(A ∧ V) ∧ (B ∨ G) ∧ ¬(B ∧ G)

theorem number_of_fir_trees (N : ℕ) :
  anya_statement N ∨ borya_statement N ∨ vera_statement N ∨ gena_statement N ∧
  one_boy_one_girl_truth (anya_statement N) (borya_statement N) (gena_statement N) (vera_statement N) :=
  N = 11 :=
sorry

end number_of_fir_trees_l818_818291


namespace plane_AB_midpoint_CD_perpendicular_CD_l818_818395

-- Define points in 3D space
variables {A B C D : EuclideanSpace ℝ (Fin 3)}

-- Given conditions
def AB_perpendicular_CD (A B C D : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ∃ (u : EuclideanSpace ℝ (Fin 3)), u = B - A ∧ (CD_vector C D · u = 0)

def angle_ACB_eq_angle_ADB (A B C D : EuclideanSpace ℝ (Fin 3)) : Prop :=
  let u := (C - A) - ((C - A) · (B - A) / ∥B - A∥^2 • (B - A)) in
  let v := (D - A) - ((D - A) · (B - A) / ∥B - A∥^2 • (B - A)) in
  ∃ θ : ℝ, θ ≠ π ∧ θ ≠ 0 ∧ θ = real.angle_between u v

-- Prove the statement
theorem plane_AB_midpoint_CD_perpendicular_CD 
  (h1 : AB_perpendicular_CD A B C D)
  (h2 : angle_ACB_eq_angle_ADB A B C D) :
  let midpoint_CD := 1/2 • (C + D) in
  let plane_AB_midpoint_CD := span ℝ {B - A, midpoint_CD - A} in
  ∃ v : EuclideanSpace ℝ (Fin 3), v = D - C ∧ is_orthogonal plane_AB_midpoint_CD v :=
sorry

end plane_AB_midpoint_CD_perpendicular_CD_l818_818395


namespace missed_bus_time_by_l818_818631

def bus_departure_time : Time := Time.mk 8 0 0
def travel_time_minutes : Int := 30
def departure_time_home : Time := Time.mk 7 50 0
def arrival_time_pickup_point : Time := 
  departure_time_home.addMinutes travel_time_minutes

theorem missed_bus_time_by :
  arrival_time_pickup_point.diff bus_departure_time = 20 * 60 :=
by
  sorry

end missed_bus_time_by_l818_818631


namespace optimal_ticket_price_l818_818168

noncomputable def revenue (x : ℕ) : ℤ :=
  if x < 6 then -5750
  else if x ≤ 10 then 1000 * (x : ℤ) - 5750
  else if x ≤ 38 then -30 * (x : ℤ)^2 + 1300 * (x : ℤ) - 5750
  else -5750

theorem optimal_ticket_price :
  revenue 22 = 8330 :=
by
  sorry

end optimal_ticket_price_l818_818168


namespace inscribe_square_in_acute_triangle_l818_818407

-- Define the triangle and the vertices of the square inscribed
variables {A B C K L M N : Point}

-- Assume the given conditions
axiom acute_angled_triangle (ABC : Triangle) : acute ABC
axiom K_on_AB (K : Point) : K ∈ segment A B
axiom N_on_AC (N : Point) : N ∈ segment A C
axiom L_on_BC (L : Point) : L ∈ segment B C
axiom M_on_BC (M : Point) : M ∈ segment B C

-- Definition of an inscribed square KLMN in the triangle ABC
def inscribed_square_in_triangle (ABC : Triangle) (K L M N : Point) : Prop :=
  K ∈ segment A B ∧ N ∈ segment A C ∧ L ∈ segment B C ∧ M ∈ segment B C ∧
  square K L M N

-- The final theorem to be proved
theorem inscribe_square_in_acute_triangle :
  ∃ (K L M N : Point), inscribed_square_in_triangle ABC K L M N :=
sorry

end inscribe_square_in_acute_triangle_l818_818407


namespace ball_hits_ground_at_t_l818_818893

theorem ball_hits_ground_at_t (t : ℝ) : 
  (∃ t, -8 * t^2 - 12 * t + 64 = 0 ∧ 0 ≤ t) → t = 2 :=
by
  sorry

end ball_hits_ground_at_t_l818_818893


namespace solution_set_of_inequality_l818_818368

theorem solution_set_of_inequality (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  {x : ℝ | (a - x) * (x - 1 / a) > 0} = {x : ℝ | a < x ∧ x < 1 / a} :=
sorry

end solution_set_of_inequality_l818_818368


namespace triangle_angle_division_l818_818317

theorem triangle_angle_division (A B C M H D : Type)
  (median_AM : A -> M -> C)
  (altitude_AH : A -> H -> C)
  (angle_bisector_AD : A -> D -> C)
  (angle_A_divided : ∀ α : ℝ, (∠ BAC = 4 * α))
  : α = 22.5 :=
begin
  sorry
end

end triangle_angle_division_l818_818317


namespace sequence_linear_increment_l818_818036

theorem sequence_linear_increment
  (a b c d e f : ℕ)
  (h1 : b - a = 67)
  (h2 : c - b = 67)
  (h3 : d - c = 67)
  (h4 : e - d = 67)
  (h5 : f - e = 67) :
  a = 5 ∧ b = 72 ∧ c = 139 ∧ d = 206 ∧ e = 273 ∧ f = 340 :=
begin
  sorry
end

end sequence_linear_increment_l818_818036


namespace sequence_a2_value_l818_818062

theorem sequence_a2_value (a : ℕ → ℤ) (h1 : a 1 = 27) (h2 : a 9 = 135)
  (hmean : ∀ n, n ≥ 3 → a n = (∑ i in finRange (n - 1), a (i + 1)) / (n - 1)) :
  a 2 = 243 := by
  sorry

end sequence_a2_value_l818_818062


namespace sophie_additional_clothing_purchase_l818_818038

theorem sophie_additional_clothing_purchase
  (initial_budget : ℝ)
  (shirt_cost : ℝ)
  (num_shirts : ℕ)
  (trousers_cost : ℝ)
  (additional_item_cost : ℝ) :
  initial_budget = 260 →
  shirt_cost = 18.5 →
  num_shirts = 2 →
  trousers_cost = 63 →
  additional_item_cost = 40 →
  (initial_budget - (num_shirts * shirt_cost + trousers_cost)) / additional_item_cost = 4 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  have h_shirts_cost : num_shirts * shirt_cost = 37 :=
    calc num_shirts * shirt_cost = 2 * 18.5 : by rw [h₃, h₂]
                           ... = 37 : by norm_num
  have h_total_spent : num_shirts * shirt_cost + trousers_cost = 100 :=
    calc num_shirts * shirt_cost + trousers_cost = 37 + 63 : by rw [h_shirts_cost, h₄]
                                          ... = 100 : by norm_num
  have h_remaining_budget : initial_budget - (num_shirts * shirt_cost + trousers_cost) = 160 :=
    calc initial_budget - (num_shirts * shirt_cost + trousers_cost) = 260 - 100 : by rw [h₁, h_total_spent]
                                               ... = 160 : by norm_num
  have h_num_articles : 160 / additional_item_cost = 4 :=
    calc 160 / additional_item_cost = 160 / 40 : by rw [h₅]
                           ... = 4 : by norm_num
  rw [h_remaining_budget]
  exact h_num_articles
  sorry

end sophie_additional_clothing_purchase_l818_818038


namespace math_problem_l818_818359

open Real

noncomputable def vector_m (x : ℝ) : ℝ × ℝ := (sin (x / 3), -1)
noncomputable def vector_n (A x : ℝ) : ℝ × ℝ := (sqrt 3 / 2 * A, 1 / 2 * A * cos (x / 3))
noncomputable def f (A x : ℝ) : ℝ := let ⟨m1, m2⟩ := vector_m x; let ⟨n1, n2⟩ := vector_n A x in n1 * m1 + n2 * m2

theorem math_problem 
  (A : ℝ) (hA : A > 0) (hx_max : ∀ x, f A x ≤ 2) :
  (∀ x, f A x = 2 * sin (x / 3 - π / 6)) ∧ 
  (∃ T : ℝ, T = 6 * π) ∧ 
  (∀ (α β : ℝ), 0 ≤ α ∧ α ≤ π / 2 → 0 ≤ β ∧ β ≤ π / 2 → 
    f A (3 * α + π / 2) = 10 / 13 → f A (3 * β + 2 * π) = 6 / 5 →
    sin (α - β) = -33 / 65) :=
begin
  sorry
end

end math_problem_l818_818359


namespace incorrect_parallel_planes_statement_l818_818607

theorem incorrect_parallel_planes_statement (A B C D : Prop)
  (hA : ∀ (L : Type) (P1 P2 : L), (P1 ∥ P2) → (L ∩ P1 ≠ ∅) → (L ∩ P2 ≠ ∅))
  (hB : ∀ (P Q R : Type), (Q ∥ R) → (P ∩ Q = l1) → (P ∩ R = l2) → (l1 ∥ l2))
  (hC : ∀ (P Q R : Type), (P ∥ R) → (Q ∥ R) → (P ∥ Q))
  (hD : ∀ (P Q L : Type), (P ∥ L) → (Q ∥ L) → (P ∥ Q) ∨ ¬(P ∥ Q ∧ P ∩ Q = ∅)) :
  ¬D :=
by
  sorry

end incorrect_parallel_planes_statement_l818_818607


namespace div_by_3_pow_101_l818_818027

theorem div_by_3_pow_101 : ∀ (n : ℕ), (∀ k : ℕ, (3^(k+1)) ∣ (2^(3^k) + 1)) → 3^101 ∣ 2^(3^100) + 1 :=
by
  sorry

end div_by_3_pow_101_l818_818027


namespace fir_trees_alley_l818_818281

-- Define the statements made by each child
def statementAnya (N : ℕ) : Prop := N = 15
def statementBorya (N : ℕ) : Prop := N % 11 = 0
def statementVera (N : ℕ) : Prop := N < 25
def statementGena (N : ℕ) : Prop := N % 22 = 0

-- Define the condition about the truth and lies
def oneBoyOneGirlTruth (anya_vera_truth: Prop) (borya_gena_truth: Prop) : Prop :=
  anya_vera_truth ∧ borya_gena_truth ∧
  ((statementAnya N ∧ statementVera N) ∨ (statementVera N ∧ statementBorya N)) ∧
  ¬( (statementAnya N ∧ statementGena N) ∨ (statementVera N ∧ statementGena N) ∨
     (statementAnya N ∧ statementBorya N) ∨ (statementBorya N ∧ statementGena N) )

-- Prove that the number of fir trees is 11
theorem fir_trees_alley: ∃ (N : ℕ), statementBorya N ∧ statementVera N ∧ ¬ statementAnya N ∧ ¬ statementGena N ∧ oneBoyOneGirlTruth (¬ statementAnya N ∧ statementVera N) (statementBorya N ∧ ¬ statementGena N) ∧ N = 11 :=
by
  sorry

end fir_trees_alley_l818_818281


namespace batch_weights_proof_l818_818573

def differences := [-5, -2, 0, 1, 3, 6]
def bag_counts := [1, 4, 3, 4, 5, 3]

def standard_weight : ℕ := 450
def n_bags : ℕ := 20
def acceptable_range := (standard_weight - 5, standard_weight + 5)

def total_difference (diffs : List ℤ) (counts : List ℕ) : ℤ :=
  List.sum (List.map (λ (pair : ℤ × ℕ), pair.fst * pair.snd) (List.zip diffs counts))

def total_weight (std_weight : ℕ) (total_diff : ℤ) (num_bags : ℕ) : ℤ :=
  std_weight * num_bags + total_diff

def average_weight (total_wt : ℤ) (num_bags : ℕ) : ℤ :=
  total_wt / num_bags

def acceptance_rate (diffs : List ℤ) (counts : List ℕ) (range : ℕ × ℕ) : ℕ :=
  let acceptable_counts : ℕ :=
    List.sum (List.map (λ (pair : ℤ × ℕ), if pair.fst.abs ≤ range.snd then pair.snd else 0) (List.zip diffs counts))
  (acceptable_counts * 100) / n_bags

theorem batch_weights_proof : 
  let total_d := total_difference differences bag_counts in
  let total_wt := total_weight standard_weight total_d n_bags in
  let avg_wt := average_weight total_wt n_bags in
  total_wt = 9024 ∧ avg_wt = 451.2 ∧ (avg_wt = standard_weight + 1.2 : bool) ∧ acceptance_rate differences bag_counts acceptable_range = 85 :=
by
  sorry

end batch_weights_proof_l818_818573


namespace runners_meet_at_same_time_after_6000_seconds_l818_818081

noncomputable def runners_meet_time (s1 s2 s3 track_length : ℝ) : ℝ :=
  let t₁ := track_length / 0.5
  let t₂ := track_length / 0.2
  Real.lcm t₁ t₂

theorem runners_meet_at_same_time_after_6000_seconds :
  ∀ (t : ℝ), 
  runners_meet_time 4.4 4.9 5.1 600 = 6000 :=
by
  intro t
  sorry

end runners_meet_at_same_time_after_6000_seconds_l818_818081


namespace tangle_words_with_A_l818_818500

theorem tangle_words_with_A :
  let alphabet := 25
  let words_with_A (n : ℕ) : ℕ :=
    ∑ i in finset.range (n + 1), alphabet^i - 24^i
  words_with_A 5 = 1678698 :=
by
  sorry

end tangle_words_with_A_l818_818500


namespace fraction_equality_l818_818216

theorem fraction_equality : 
  (3 ^ 8 + 3 ^ 6) / (3 ^ 8 - 3 ^ 6) = 5 / 4 :=
by
  -- Expression rewrite and manipulation inside parenthesis can be ommited
  sorry

end fraction_equality_l818_818216


namespace bead_necklaces_count_l818_818848

-- Define the conditions
def cost_per_necklace : ℕ := 9
def gemstone_necklaces_sold : ℕ := 3
def total_earnings : ℕ := 90

-- Define the total earnings from gemstone necklaces
def earnings_from_gemstone_necklaces : ℕ := gemstone_necklaces_sold * cost_per_necklace

-- Define the total earnings from bead necklaces
def earnings_from_bead_necklaces : ℕ := total_earnings - earnings_from_gemstone_necklaces

-- Define the number of bead necklaces sold
def bead_necklaces_sold : ℕ := earnings_from_bead_necklaces / cost_per_necklace

-- The statement to be proved
theorem bead_necklaces_count : bead_necklaces_sold = 7 := by
  sorry

end bead_necklaces_count_l818_818848


namespace angle_bisectors_intersect_l818_818869

theorem angle_bisectors_intersect 
  (A B C D : Type) [has_measure A] [has_measure B] [has_measure C] [has_measure D]
  (AB CD AD BC : ℝ) (h : AB + CD = AD + BC) : 
  ∃ P, is_point_of_intersection_of_angle_bisectors A B C D P :=
sorry

end angle_bisectors_intersect_l818_818869


namespace fencing_required_l818_818592

theorem fencing_required (L W : ℝ) (h1 : L = 40) (h2 : L * W = 680) : 2 * W + L = 74 :=
by
  sorry

end fencing_required_l818_818592


namespace T_shape_perimeter_l818_818937

theorem T_shape_perimeter :
  ∀ (h1 : ℝ) (w1 : ℝ) (h2 : ℝ) (w2 : ℝ),
  h1 = 3 ∧ w1 = 5 ∧ h2 = 2 ∧ w2 = 6 →
  let vertical_edges := 2 * (h1 - 1)
  let horizontal_edges := 2 * w1
  vertical_edges + horizontal_edges = 20 :=
by {
  intros h1 w1 h2 w2 h_cond,
  let vertical_edges := 2 * (h1 - 1),
  let horizontal_edges := 2 * w1,
  sorry
}

end T_shape_perimeter_l818_818937


namespace angle_APB_is_135_l818_818812

theorem angle_APB_is_135 (A B C D P : ℝ) (hA : A ≠ B) (hB : B ≠ C) (hC : C ≠ D) (hD : D ≠ A)
  (h_square : A ≠ C ∧ B ≠ D) -- conditions for square
  (h_ratio : PA / PB = 1 / 2 ∧ PA / PC = 1 / 3) : 
  ∠APB = 135 :=
sorry

end angle_APB_is_135_l818_818812


namespace value_of_T_l818_818432

theorem value_of_T (x : ℝ) : 
  let T := (x-2) ^ 4 + 8 * (x-2) ^ 3 + 24 * (x-2) ^ 2 + 32 * (x-2) + 16 in 
  T = x ^ 4 :=
by 
  let T := (x-2) ^ 4 + 8 * (x-2) ^ 3 + 24 * (x-2) ^ 2 + 32 * (x-2) + 16
  have h₁ : T = (x-2)^4 + 8*(x-2)^3 + 24*(x-2)^2 + 32*(x-2) + 16 := rfl
  sorry

end value_of_T_l818_818432


namespace time_with_cat_total_l818_818843

def time_spent_with_cat (petting combing brushing playing feeding cleaning : ℕ) : ℕ :=
  petting + combing + brushing + playing + feeding + cleaning

theorem time_with_cat_total :
  let petting := 12
  let combing := 1/3 * petting
  let brushing := 1/4 * combing
  let playing := 1/2 * petting
  let feeding := 5
  let cleaning := 2/5 * feeding
  time_spent_with_cat petting combing brushing playing feeding cleaning = 30 := by
  sorry

end time_with_cat_total_l818_818843


namespace total_spending_l818_818440

-- Define the constants
def TL : ℝ := 40
def JL : ℝ := 1 / 2 * TL
def CL : ℝ := 2 * TL
def S : ℝ := 3 * JL
def TC : ℝ := 1 / 4 * TL
def JC : ℝ := 3 * JL
def CC : ℝ := 1 / 2 * CL
def DC : ℝ := 2 * S

-- Define the total spending for Lisa and Carly
def total_lisa : ℝ := TL + JL + CL + S
def total_carly : ℝ := TC + JC + CC + S + DC

-- The proof we need to show
theorem total_spending : total_lisa + total_carly = 490 := by
  sorry

end total_spending_l818_818440


namespace incorrect_statement_C_l818_818336

theorem incorrect_statement_C 
  (x y : ℝ)
  (n : ℕ)
  (data : Fin n → (ℝ × ℝ))
  (h : ∀ (i : Fin n), (x, y) = data i)
  (reg_eq : ∀ (x : ℝ), 0.85 * x - 85.71 = y) :
  ¬ (forall (x : ℝ), x = 160 → ∀ (y : ℝ), y = 50.29) := 
sorry

end incorrect_statement_C_l818_818336


namespace gcd_paving_courtyard_l818_818576

theorem gcd_paving_courtyard :
  Nat.gcd 378 595 = 7 :=
by
  sorry

end gcd_paving_courtyard_l818_818576
