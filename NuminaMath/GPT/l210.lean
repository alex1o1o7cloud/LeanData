import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Combinatorics.Combination
import Mathlib.Algebra.Equation.Basic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Definitions
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Matrix
import Mathlib.Algebra.Order.Group
import Mathlib.Analysis.Calculus.Floor
import Mathlib.Analysis.Geometry.Trigonometry
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Fractions
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Char.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Perm
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Probability.ConditionalProbability
import Mathlib.Data.ProbabilityBasic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Angle.Cotangent
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Data.Time.Basic
import Mathlib.Geometry.Centroid
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Triangle
import Mathlib.LinearAlgebra.Median
import Mathlib.Matrix
import Mathlib.Tactic
import Mathlib.Tactic.Basic

namespace find_marks_in_english_l210_210736

def david_marks_in_english (E : ℝ) : Prop :=
  let M := 95
  let P := 82
  let C := 87
  let B := 92
  let avg := 90.4
  let num_subjects := 5
  avg = (E + M + P + C + B) / num_subjects

theorem find_marks_in_english : david_marks_in_english 96 :=
by
  let E := 96
  let M := 95
  let P := 82
  let C := 87
  let B := 92
  let avg := 90.4
  let num_subjects := 5
  have h : avg = (E + M + P + C + B) / num_subjects
  sorry

end find_marks_in_english_l210_210736


namespace campers_rowing_morning_l210_210169

-- Define the number of campers who went hiking and rowing
variables (total_rowers afternoon_rowers : ℕ)

-- Assume the conditions given in the problem
axiom cond1 : total_rowers = 34
axiom cond2 : afternoon_rowers = 21

-- Define the number of rowers in the morning
def morning_rowers : ℕ := total_rowers - afternoon_rowers

-- State the main theorem we need to prove
theorem campers_rowing_morning :
  morning_rowers total_rowers afternoon_rowers = 13 :=
by
  rw [morning_rowers, cond1, cond2]
  exact rfl

end campers_rowing_morning_l210_210169


namespace mike_spent_on_tires_l210_210550

variables (total_spent_on_car_parts amount_spent_on_speakers amount_spent_on_tires : ℝ)

def mike_scenario (total_spent_on_car_parts = 224.87) (amount_spent_on_speakers = 118.54) : Prop :=
  amount_spent_on_tires = total_spent_on_car_parts - amount_spent_on_speakers

theorem mike_spent_on_tires :
  mike_scenario 224.87 118.54 :=
by
  simp [mike_scenario]
  sorry

end mike_spent_on_tires_l210_210550


namespace number_of_rows_l210_210679

theorem number_of_rows (n : ℕ) :
  (3 * n * (n + 1)) / 2 = 225 → n = 11 :=
by
  -- Given the equation to solve (condition of the problem)
  have h : (3 * n * (n + 1)) / 2 = 225
  sorry -- Proof of the theorem

end number_of_rows_l210_210679


namespace hyperbola_eq_l210_210451

theorem hyperbola_eq :
  ∀ (a b p : ℝ), 
  (a > 0 ∧ b > 0 ∧ p > 0) ∧
  (dist (-a, 0) (p / 2, 0) = 4) ∧
  (∀ x y : ℝ, (x = -1 ∧ y = -1) → y = (b / a) * x) ∧ 
  (-p / 2 = -1) →
  (a = 3 ∧ b = 3) →
  ∀ x y : ℝ, (x^2 / 9 - y^2 / 9 = 1) :=
by
  intros a b p hab hdist hasym hdir hab_eq
  -- Detailed proof would go here
  exact sorry

end hyperbola_eq_l210_210451


namespace sum_of_intersection_coordinates_l210_210097

noncomputable def sum_of_x_coordinates : ℝ → ℝ → Prop :=
λ x1 x2, 2^x1 = -x1 + 6 ∧ log x2 / log 2 = -x2 + 6 ∧ x2 = 2^x1 ∧ x1 + x2 = 6

theorem sum_of_intersection_coordinates :
  ∃ x1 x2 : ℝ, sum_of_x_coordinates x1 x2 :=
begin
  sorry
end

end sum_of_intersection_coordinates_l210_210097


namespace john_mobile_purchase_price_l210_210122

-- Define the conditions
def price_grinder := 15000
def loss_grinder := 0.04 * price_grinder
def selling_price_grinder := price_grinder - loss_grinder

def profit_ratio_mobile : ℝ := 0.10
def selling_price_mobile (M : ℝ) := M * (1 + profit_ratio_mobile)

def overall_profit := 400

-- Define the mathematical property to prove
theorem john_mobile_purchase_price (M : ℝ) 
  (H1 : selling_price_grinder = price_grinder - loss_grinder)
  (H2 : selling_price_mobile M - M = profit_ratio_mobile * M)
  (H3 : (selling_price_mobile M - M) - (price_grinder - selling_price_grinder) = overall_profit) :
  M = 10000 :=
  sorry  -- proof to be completed

end john_mobile_purchase_price_l210_210122


namespace money_left_after_deductions_l210_210366

-- Define the weekly income
def weekly_income : ℕ := 500

-- Define the tax deduction as 10% of the weekly income
def tax : ℕ := (10 * weekly_income) / 100

-- Define the weekly water bill
def water_bill : ℕ := 55

-- Define the tithe as 10% of the weekly income
def tithe : ℕ := (10 * weekly_income) / 100

-- Define the total deductions
def total_deductions : ℕ := tax + water_bill + tithe

-- Define the money left
def money_left : ℕ := weekly_income - total_deductions

-- The statement to prove
theorem money_left_after_deductions : money_left = 345 := by
  sorry

end money_left_after_deductions_l210_210366


namespace find_tangent_line_to_circle_l210_210590

noncomputable def circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x = 0

noncomputable def point_on_circle (P : ℝ × ℝ) : Prop :=
  circle P.1 P.2

noncomputable def tangent_line (x y k : ℝ) : Prop :=
  k*x - y - k + real.sqrt 3 = 0

noncomputable def distance (x₀ y₀ k : ℝ) : ℝ :=
  abs (2*k - k + real.sqrt 3) / (real.sqrt (k^2 + 1))

theorem find_tangent_line_to_circle (P : ℝ × ℝ) (hP : P = (1, real.sqrt 3)) :
  point_on_circle P → tangent_line 1 (real.sqrt 3) (real.sqrt 3 / 3):=
begin
  -- Define the center of the circle (2, 0)
  let C := (2 : ℝ, 0 : ℝ),
  -- Prove that the point \(P = (1, √3)\) is on the circle.
  intro h,
  -- Set slope \(k = √3 / 3\)
  let k := real.sqrt 3 / 3,
  -- The equation of the tangent line through point \(P\)
  have hTangent : tangent_line 1 (real.sqrt 3) k := rfl,
  -- Prove that the distance from the center of the circle to the tangent line is 2.
  have hDist := distance 2 0 k = 2,
  -- Conclude the proof.
  exact hTangent,
end

end find_tangent_line_to_circle_l210_210590


namespace math_problem_l210_210476

noncomputable def prop_1 (l : Line) (β α : Plane) : Prop :=
  l ⊆ β ∧ α ⊥ β → l ⊥ α

noncomputable def prop_2 (l : Line) (β α : Plane) : Prop :=
  l ⊥ β ∧ α ∥ β → l ⊥ α

noncomputable def prop_3 (l : Line) (β α : Plane) : Prop :=
  l ⊥ β ∧ α ⊥ β → l ∥ α

noncomputable def prop_4 (l m : Line) (β α : Plane) : Prop :=
  α ∩ β = m ∧ l ∥ m → l ∥ α

noncomputable def correct_propositions : Prop :=
  prop_2

theorem math_problem (l m : Line) (α β : Plane) :
  prop_1 l β α = false ∧
  prop_2 l β α = true ∧
  prop_3 l β α = false ∧
  prop_4 l m β α = false :=
  by
    sorry

end math_problem_l210_210476


namespace sufficient_but_not_necessary_l210_210866

variable (x : ℝ)

def condition_p := -1 ≤ x ∧ x ≤ 1
def condition_q := x ≥ -2

theorem sufficient_but_not_necessary :
  (condition_p x → condition_q x) ∧ ¬(condition_q x → condition_p x) :=
by 
  sorry

end sufficient_but_not_necessary_l210_210866


namespace arcsin_one_eq_pi_div_two_l210_210725

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = Real.pi / 2 := 
by
  sorry

end arcsin_one_eq_pi_div_two_l210_210725


namespace find_speed_l210_210922

noncomputable def circumference := 15 / 5280 -- miles
noncomputable def increased_speed (r : ℝ) := r + 5 -- miles per hour
noncomputable def reduced_time (t : ℝ) := t - 1 / 10800 -- hours
noncomputable def original_distance (r t : ℝ) := r * t
noncomputable def new_distance (r t : ℝ) := increased_speed r * reduced_time t

theorem find_speed (r t : ℝ) (h1 : original_distance r t = circumference) 
(h2 : new_distance r t = circumference) : r = 13.5 := by
  sorry

end find_speed_l210_210922


namespace bus_speed_kmph_l210_210664

/-- Defining the conditions -/
def bus_length : ℕ := 100
def bridge_length : ℕ := 150
def crossing_time : ℕ := 18

/-- Defining the theorem to prove the speed of the bus in kmph -/
theorem bus_speed_kmph : (bus_length + bridge_length) / crossing_time * 3.6 = 50 := sorry

end bus_speed_kmph_l210_210664


namespace correct_option_l210_210277

-- Define the conditions for the problem
def cond_A : Prop := "Approximate number 4.1 ten thousand accurate to the tenths place" = False
def cond_B : Prop := "Approximate number 0.520 accurate to the hundredths place" = False
def cond_C : Prop := "Approximate number 3.72 accurate to the tenths place" = True
def cond_D : Prop := "Approximate number 5000 accurate to the thousands place" = False

-- The goal is to prove that option C is correct
theorem correct_option : cond_C :=
begin
  sorry
end

end correct_option_l210_210277


namespace interval_of_increase_l210_210457

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Math.sin (2 * x + φ)

theorem interval_of_increase {φ : ℝ} (h1 : |φ| < Real.pi) 
                             (h2 : ∀ x : ℝ, f x φ ≤ |f (Real.pi / 6) φ|)
                             (h3 : f (Real.pi / 2) φ < f (Real.pi / 3) φ)
                             (k : ℤ) : 
  ∃ (a b : ℝ), (a = k * Real.pi - Real.pi / 3 ∧ b = k * Real.pi + Real.pi / 6) :=
begin
  sorry
end

end interval_of_increase_l210_210457


namespace problem_equivalence_l210_210081

-- Define the problem conditions
def non_zero_single_digit (n : Nat) : Prop := n > 0 ∧ n < 10

-- Assume 学, 科, 能, and 力 are non-zero single-digit numbers
variables (xue ke neng li : Nat) 
variables (h1 : non_zero_single_digit xue)
variables (h2 : non_zero_single_digit ke)
variables (h3 : non_zero_single_digit neng)
variables (h4 : non_zero_single_digit li)

-- Define the "△" operation with the given properties
axiom delta1 : (xue * 1000 + ke * 100 + neng * 10 + li) △ 1 = (ke * 1000 + xue * 100 + neng * 10 + li)
axiom delta2 : (xue * 1000 + ke * 100 + neng * 10 + li) △ 2 = (neng * 1000 + li * 100 + ke * 10 + xue)

-- Define the final theorem to be proved
theorem problem_equivalence : (1 * 1000 + 2 * 100 + 3 * 10 + 4) △ 1 △ 2 = (3 * 1000 + 4 * 100 + 1 * 10 + 2) :=
by
  sorry

end problem_equivalence_l210_210081


namespace avg_velocity_2_to_2_1_l210_210603

def motion_eq (t : ℝ) : ℝ := 3 + t^2

theorem avg_velocity_2_to_2_1 : 
  (motion_eq 2.1 - motion_eq 2) / (2.1 - 2) = 4.1 :=
by
  sorry

end avg_velocity_2_to_2_1_l210_210603


namespace num_solutions_eq_six_l210_210851

theorem num_solutions_eq_six : 
  {x : ℤ // x % 25 = 22} ∪ 
  {x : ℤ // x % 25 = 7} ∪ 
  {x : ℤ // x % 25 = 18} ∪ 
  {x : ℤ // x % 25 = 2} ∪ 
  {x : ℤ // x % 25 = 12} ∪ 
  {x : ℤ // x % 25 = 17} = 
  {x : ℤ // x^3 + 3 * x^2 + x + 3 ≡ 0 [MOD 25]} :=
sorry

end num_solutions_eq_six_l210_210851


namespace Alice_spent_19_percent_l210_210714

variable (A B A': ℝ)
def Bob_less_money_than_Alice (A B : ℝ) : Prop :=
  B = 0.9 * A

def Alice_less_money_than_Bob (B A' : ℝ) : Prop :=
  A' = 0.9 * B

theorem Alice_spent_19_percent (A B A' : ℝ) 
  (h1 : Bob_less_money_than_Alice A B)
  (h2 : Alice_less_money_than_Bob B A') :
  ((A - A') / A) * 100 = 19 :=
by
  sorry

end Alice_spent_19_percent_l210_210714


namespace problem_expression_l210_210431

theorem problem_expression (x y : ℝ) (h1 : x - y = 5) (h2 : x * y = 4) : x^2 + y^2 = 33 :=
by sorry

end problem_expression_l210_210431


namespace sum_of_three_distinct_members_card_l210_210482

-- Define the set
def mySet : Set ℕ := {2, 5, 8, 11, 14, 17, 20}

-- Prove that the number of different integers that can be expressed as the sum of three distinct members of the set is 13.
theorem sum_of_three_distinct_members_card : 
  (∃ s : Set ℕ, s = {n | ∃ (a b c ∈ mySet), a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a + b + c = n} ∧ s.card = 13) :=
sorry

end sum_of_three_distinct_members_card_l210_210482


namespace complex_magnitude_l210_210304

variables (i : ℂ)
axiom imaginary_unit : i^2 = -1

theorem complex_magnitude : |2 + i^2 + 2 * i^3| = Real.sqrt 5 :=
by
  have h1 : i^3 = i^2 * i := by sorry
  have h2 : i^2 = -1 := imaginary_unit
  have h3 : i^3 = -i := by sorry
  calc 
    |2 + i^2 + 2 * i^3| = |2 + (-1) + 2 * (-i)| : by sorry
    ... = |1 - 2 * i| : by sorry
    ... = Real.sqrt (1^2 + (-2)^2) : by sorry
    ... = Real.sqrt 5 : by sorry

end complex_magnitude_l210_210304


namespace circles_separated_l210_210742

noncomputable def circle1 (x y : ℝ) : Prop := (x + 2)^2 + (y + 1)^2 = 4
noncomputable def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

theorem circles_separated : 
  ∀ x₁ y₁ x₂ y₂, 
  circle1 x₁ y₁ → circle2 x₂ y₂ → 
  dist (x₁, y₁) (x₂, y₂) > 4 :=
begin
  sorry
end

end circles_separated_l210_210742


namespace arcsin_one_eq_pi_div_two_l210_210718

noncomputable def arcsin (x : ℝ) : ℝ :=
classical.some (exists_inverse_sin x)

theorem arcsin_one_eq_pi_div_two : arcsin 1 = π / 2 :=
sorry

end arcsin_one_eq_pi_div_two_l210_210718


namespace complex_magnitude_l210_210326

open Complex

theorem complex_magnitude :
  ∀ (i : ℂ), i^2 = -1 → i^3 = -i → |2 + i^2 + 2 * i^3| = Real.sqrt 5 :=
by
  intros i h1 h2
  -- skipping proof with sorry
  sorry

end complex_magnitude_l210_210326


namespace problem_solution_l210_210834

theorem problem_solution :
  ∀ (x y z : ℤ),
  4 * x + y + z = 80 →
  3 * x + y - z = 20 →
  x = 20 →
  2 * x - y - z = 40 :=
by
  intros x y z h1 h2 hx
  rw [hx] at h1 h2
  -- Here you could continue solving but we'll use sorry to indicate the end as no proof is requested.
  sorry

end problem_solution_l210_210834


namespace tan_of_angle_in_fourth_quadrant_l210_210863

theorem tan_of_angle_in_fourth_quadrant (a : ℝ) (h1 : sin a = - (5 / 13)) (h2 : 2 * π > a ∧ a > 3 * π / 2) : tan a = - (5 / 12) :=
sorry

end tan_of_angle_in_fourth_quadrant_l210_210863


namespace question_l210_210031

variable (U : Set ℕ) (M : Set ℕ)

theorem question :
  U = {1, 2, 3, 4, 5} →
  (U \ M = {1, 3}) →
  2 ∈ M :=
by
  intros
  sorry

end question_l210_210031


namespace g_neither_even_nor_odd_l210_210113

def g (x : ℝ) : ℝ := ⌊2 * x⌋ + (1 / 3 : ℝ)

theorem g_neither_even_nor_odd :
  ¬ (∀ x : ℝ, g x = g (-x)) ∧ ¬ (∀ x : ℝ, g x = - g (-x)) :=
by
  sorry

end g_neither_even_nor_odd_l210_210113


namespace opposite_of_5_is_neg5_l210_210221

def opposite (n x : ℤ) := n + x = 0

theorem opposite_of_5_is_neg5 : opposite 5 (-5) :=
by
  sorry

end opposite_of_5_is_neg5_l210_210221


namespace g_neither_even_nor_odd_l210_210115

def g (x : ℝ) : ℝ := ⌊2 * x⌋ + (1 / 3 : ℝ)

theorem g_neither_even_nor_odd :
  ¬ (∀ x : ℝ, g x = g (-x)) ∧ ¬ (∀ x : ℝ, g x = - g (-x)) :=
by
  sorry

end g_neither_even_nor_odd_l210_210115


namespace gcd_sequence_l210_210136

theorem gcd_sequence (n : ℕ) : gcd ((7^n - 1)/6) ((7^(n+1) - 1)/6) = 1 := by
  sorry

end gcd_sequence_l210_210136


namespace AX_eq_AY_l210_210653

variable (A B C D E P X Y : Point)
variable (C1 C2 omega : Circle)

-- Conditions
axiom is_regular_pentagon : regular_pentagon A B C D E
axiom C1_through_B_center_A : C1.center = A ∧ B ∈ C1
axiom C2_through_B_center_C : C2.center = C ∧ B ∈ C2
axiom P_is_other_inter_C1_C2 : P ∈ C1 ∧ P ∈ C2 ∧ P ≠ B
axiom omega_center_P_through_E_D : omega.center = P ∧ E ∈ omega ∧ D ∈ omega
axiom omega_intersects_C2_at_X : X ∈ C2 ∧ X ∈ omega ∧ X ≠ D ∧ X ≠ E
axiom omega_intersects_AE_at_Y : Y ∈ omega ∧ Y ∈ line(A, E) ∧ Y ≠ A ∧ Y ≠ E

-- Question
theorem AX_eq_AY : dist A X = dist A Y :=
by
  sorry

end AX_eq_AY_l210_210653


namespace complex_magnitude_l210_210329

open Complex

theorem complex_magnitude :
  ∀ (i : ℂ), i^2 = -1 → i^3 = -i → |2 + i^2 + 2 * i^3| = Real.sqrt 5 :=
by
  intros i h1 h2
  -- skipping proof with sorry
  sorry

end complex_magnitude_l210_210329


namespace problem_solution_l210_210245

def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def sum_of_digits (n : ℕ) : ℕ := n.digits.sum

theorem problem_solution (n : ℕ) (h1 : factorial (n + 1) + factorial (n + 3) = factorial n * 850) : sum_of_digits n = 8 := 
sorry

end problem_solution_l210_210245


namespace number_of_ordered_pairs_l210_210779

theorem number_of_ordered_pairs (T : ℕ) :
    let F : ℕ → ℕ → Prop := 
        λ a b, ∃ r s : ℤ, r + s = a ∧ r * s = b
    in T = (∑ a in finset.Icc 2 200, ∑ b in finset.Icc 0 1000, if F a b then 1 else 0) :=
sorry

end number_of_ordered_pairs_l210_210779


namespace no_fixed_points_range_l210_210781

-- Define the function f and the condition of having no fixed points
def f (x a : ℝ) : ℝ := x^2 + a * x + 1

-- Statement: If the function f(x, a) has no fixed points, then -1 < a < 3
theorem no_fixed_points_range (a : ℝ) :
  ¬ ∃ x : ℝ, f x a = x → -1 < a ∧ a < 3 :=
begin
  sorry
end

end no_fixed_points_range_l210_210781


namespace slope_of_line_polar_circle_equation_l210_210887

def cartesian_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 9

def polar_equation_of_circle (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * ρ * real.cos θ - 5 = 0

def parametric_line (t α x y : ℝ) : Prop :=
  x = t * real.cos α ∧ y = t * real.sin α

def distance_from_center_to_line (k : ℝ) : ℝ :=
  2 * |k| / real.sqrt (1 + k^2)

def line_intersects_circle_at_ab (d : ℝ) : Prop :=
  d = real.sqrt (9 - 7)

theorem slope_of_line
  (t α x y : ℝ) (k : ℝ) :
  parametric_line t α x y →
  let d := distance_from_center_to_line k in
  line_intersects_circle_at_ab d →
  k = 1 ∨ k = -1 :=
by
  -- Definitions and conditions
  intros h1 h2,
  sorry

theorem polar_circle_equation
  (x y ρ θ : ℝ) :
  cartesian_circle x y →
  x = ρ * real.cos θ →
  y = ρ * real.sin θ →
  polar_equation_of_circle ρ θ :=
by
  -- Definitions and conditions
  intros h1 h2 h3,
  sorry

end slope_of_line_polar_circle_equation_l210_210887


namespace circle_points_collinear_l210_210043

theorem circle_points_collinear
    (O1 O2 O3 : Type) (r1 r2 r3 : ℝ) 
    (A1 A2 A3 B1 B2 B3 : Type)
    (h1 : Disjoint O1 O2)
    (h2 : Disjoint O2 O3)
    (h3 : Disjoint O1 O3)
    (hA1 : InternalTangentIntersection A1 O2 O3)
    (hA2 : InternalTangentIntersection A2 O1 O3)
    (hA3 : InternalTangentIntersection A3 O1 O2)
    (hB1 : ExternalTangentIntersection B1 O2 O3)
    (hB2 : ExternalTangentIntersection B2 O1 O3)
    (hB3 : ExternalTangentIntersection B3 O1 O2) 
    : Collinear {A1, A2, B3} ∧ Collinear {A1, B2, A3} ∧ Collinear {B1, A2, A3} ∧ Collinear {B1, B2, B3} :=
sorry

end circle_points_collinear_l210_210043


namespace range_of_merged_set_l210_210952

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_primes := {x : ℕ | 10 ≤ x ∧ x < 100 ∧ is_prime x}

def positive_multiples_of_4_less_than_100 := {y : ℕ | y > 0 ∧ y < 100 ∧ y % 4 = 0}

def merged_set := two_digit_primes ∪ positive_multiples_of_4_less_than_100

def range_of_set (s : set ℕ) := (set.max' s sorry) - (set.min' s sorry)

theorem range_of_merged_set : range_of_set merged_set = 93 :=
by 
  -- proof goes here
  sorry

end range_of_merged_set_l210_210952


namespace find_angleE_l210_210521

-- Definitions based on conditions
variables (EF GH : ℝ) -- lengths are not needed in statement, but symbolic for trapezoid context
variables (angleE angleF angleG angleH : ℝ)
variables (trapezoid_EFGH : EFGH_relation EF GH angleE angleF angleG angleH)

-- Given conditions
def condition1 := EFGH_parallel EF GH angleE angleH
def condition2 := angleE = 3 * angleH
def condition3 := angleG = 2 * angleF

-- Statement to prove
theorem find_angleE (h1: condition1) (h2: condition2) (h3: condition3) : angleE = 135 := by {
  sorry
}

end find_angleE_l210_210521


namespace cost_of_fencing_l210_210646

theorem cost_of_fencing 
  (d : ℝ) (rate : ℝ) (pi_approx : ℝ)
  (d_val : d = 34)
  (rate_val : rate = 2)
  (pi_approx_val : pi_approx = 3.14159) :
  (π * d * rate).ceil = 214 := by
  sorry

end cost_of_fencing_l210_210646


namespace find_f_of_given_g_and_odd_l210_210817

theorem find_f_of_given_g_and_odd (f g : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x)
  (h_g_def : ∀ x, g x = f x + 9) (h_g_val : g (-2) = 3) :
  f 2 = 6 :=
by
  sorry

end find_f_of_given_g_and_odd_l210_210817


namespace dennis_took_away_l210_210904

-- Define the initial and remaining number of cards
def initial_cards : ℕ := 67
def remaining_cards : ℕ := 58

-- Define the number of cards taken away
def cards_taken_away (n m : ℕ) : ℕ := n - m

-- Prove that the number of cards taken away is 9
theorem dennis_took_away :
  cards_taken_away initial_cards remaining_cards = 9 :=
by
  -- Placeholder proof
  sorry

end dennis_took_away_l210_210904


namespace g_neither_even_nor_odd_l210_210102

def g (x : ℝ) : ℝ := ⌊2 * x⌋ + 1/3

theorem g_neither_even_nor_odd : (∀ x, g x ≠ g (-x)) ∧ (∀ x, g x ≠ -g (-x)) :=
by
  sorry

end g_neither_even_nor_odd_l210_210102


namespace tangent_secant_relationship_l210_210786

theorem tangent_secant_relationship
  (P A B C E F O : Point) (circle : Circle O)
  (tangent_PE : tangent circle P E) (tangent_PF : tangent circle P F)
  (secant_thru_P : secant P circle A B) (C_on_EF : on_line C E F) :
  1 / dist P C = (1 / 2) * (1 / dist P A + 1 / dist P B) := 
sorry

end tangent_secant_relationship_l210_210786


namespace axis_of_symmetry_condition_l210_210202

theorem axis_of_symmetry_condition 
  (p q r s : ℝ) (h : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0)
  (symmetry : ∀ a : ℝ, (a, -a) ∈ (λ x, (x, (px + q) / (rx + s))) ↔ (-a, a) ∈ (λ x, (x, (px + q) / (rx + s)))) :
  p = s :=
sorry

end axis_of_symmetry_condition_l210_210202


namespace loss_percentage_is_11_percent_l210_210650

-- Definitions based on conditions
def costPrice : ℝ := 1500
def sellingPrice : ℝ := 1335

-- The statement to prove
theorem loss_percentage_is_11_percent :
  ((costPrice - sellingPrice) / costPrice) * 100 = 11 := by
  sorry

end loss_percentage_is_11_percent_l210_210650


namespace weight_of_brand_b_l210_210238

theorem weight_of_brand_b (w_a w_b : ℕ) (vol_a vol_b : ℕ) (total_volume total_weight : ℕ) 
  (h1 : w_a = 950) 
  (h2 : vol_a = 3) 
  (h3 : vol_b = 2) 
  (h4 : total_volume = 4) 
  (h5 : total_weight = 3640) 
  (h6 : vol_a + vol_b = total_volume) 
  (h7 : vol_a * w_a + vol_b * w_b = total_weight) : 
  w_b = 395 := 
by {
  sorry
}

end weight_of_brand_b_l210_210238


namespace find_sum_S100_l210_210606

noncomputable def a : ℕ → ℤ
| 0     => 1
| 1     => 3
| (n+2) => abs (a (n+1)) - a n

def S (n : ℕ) : ℤ := (Finset.range n).sum a

theorem find_sum_S100 : S 100 = 89 := by
  sorry

end find_sum_S100_l210_210606


namespace sum_of_solutions_l210_210421

theorem sum_of_solutions (x : ℝ) :
  (∀ x, x^2 - 17 * x + 54 = 0) → 
  (∃ r s : ℝ, r ≠ s ∧ r + s = 17) :=
by
  sorry

end sum_of_solutions_l210_210421


namespace triangle_side_length_l210_210894

theorem triangle_side_length (AB AC : ℝ) (angle_BAC : ℝ) (h1 : AB = 4) (h2 : AC = 1) (h3 : angle_BAC = real.pi / 3) :
  ∃ BC : ℝ, BC = real.sqrt 13 :=
by {
  sorry
}

end triangle_side_length_l210_210894


namespace units_digit_8421_1287_l210_210743

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_8421_1287 :
  units_digit (8421 ^ 1287) = 1 := 
by
  sorry

end units_digit_8421_1287_l210_210743


namespace arcsin_one_eq_pi_div_two_l210_210722

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = Real.pi / 2 :=
by
  -- proof steps here
  sorry

end arcsin_one_eq_pi_div_two_l210_210722


namespace fuel_A_volume_l210_210368

-- Declare variables representing the volumes of fuel A and B
variables (V_A V_B : ℝ)

-- Given conditions as Lean definitions
def condition1 : Prop := V_A + V_B = 218
def condition2 : Prop := 0.12 * V_A + 0.16 * V_B = 30

-- The main statement that verifies the required volume of fuel A
theorem fuel_A_volume : condition1 → condition2 → V_A = 122 := by
  intro h1 h2
  -- We'll use sorry to skip the proof
  sorry

end fuel_A_volume_l210_210368


namespace sum_of_roots_l210_210167

theorem sum_of_roots (sum : ℝ) :
  (∀ x : ℝ, ∃ k : ℤ, 
      3 * Real.cos (4 * Real.pi * x / 5) 
      + Real.cos (12 * Real.pi * x / 5) 
      = 2 * Real.cos (4 * Real.pi * x / 5)
      * (3 + Real.tan (Real.pi * x / 5)^2 - 2 * Real.tan (Real.pi * x / 5))
      ↔ (x = (5 / 8) + (5 * k / 4))) → 
  (sum = List.sum (List.filter (λ x, (x > -11 ∧ x < 19)) 
    (List.map (λ k, (5 / 8) + (5 * k / 4)) (List.range' (-9) 24)))) := 
by 
  sorry

end sum_of_roots_l210_210167


namespace weight_of_five_single_beds_l210_210611

-- Define the problem conditions and the goal
theorem weight_of_five_single_beds :
  ∃ S D : ℝ, (2 * S + 4 * D = 100) ∧ (D = S + 10) → (5 * S = 50) :=
by
  sorry

end weight_of_five_single_beds_l210_210611


namespace num_irreducible_polys_even_l210_210126

theorem num_irreducible_polys_even :
  (∃ N : ℕ, (∀ (d : Fin 10 → ℕ), 
    (∀ i, 0 ≤ d i ∧ d i ≤ 9) ∧ 
    d 9 > 0 ∧ 
    irreducible (∑ i in Finset.range 10, d i * x ^ i) → 
    (N = 2 * k))) :=
sorry

end num_irreducible_polys_even_l210_210126


namespace arithmetic_sequence_value_l210_210801

theorem arithmetic_sequence_value :
  ∀ (a : ℕ → ℝ) (d : ℝ),
    (a 1 + a 3 + a 5 + a 7 + a 9 = 10) →
    ((a 7)^2 - (a 1)^2 = 36) →
    ∀ n, (a 11 = 11) :=
by
  sorry

end arithmetic_sequence_value_l210_210801


namespace angle_BAC_eq_70_l210_210671

variables (O A B C : Type)
variables (angle : O → O → ℝ)
variables (O A B C : Prop)

-- Conditions
variables (hao : angle O A B = 120)
variables (hbo : angle O B C = 140)
variables (eq_oa_ob : OA = OB) (eq_ob_oc : OB = OC) (eq_oc_oa : OC = OA)

-- Goal
theorem angle_BAC_eq_70 : angle B A C = 70 :=
by sorry

end angle_BAC_eq_70_l210_210671


namespace find_c_l210_210591

theorem find_c (x : ℝ) (c : ℝ) (h1 : 3 * x + 5 = 4) (h2 : c * x + 6 = 3) : c = 9 :=
by
  sorry

end find_c_l210_210591


namespace paint_and_mow_time_l210_210422

-- Conditions
variable (n1 t1 n2 : ℕ)
variable (t_paint t_mow T : ℝ)

-- Given conditions:
-- 1. Five people can paint a house in 10 hours.
-- 2. Mowing the lawn takes an additional 3 hours per person.
-- 3. Each person works at the same rate for painting.
-- 4. Each person works at a different rate for mowing.

def k1 : ℝ := n1 * t1 -- Total work done in person-hours for painting.

-- Theorem stating the total time for four people to paint the house and mow the lawn.
theorem paint_and_mow_time (h_n1 : n1 = 5) (h_t1 : t1 = 10)
  (h_paint : 4 * t_paint = k1) (h_mow : t_mow = 3) :
  T = t_paint + t_mow := by 
  sorry

-- Use the variables to define the expected outcome 
#check @paint_and_mow_time -- checking if the statement is correct.

end paint_and_mow_time_l210_210422


namespace middle_card_not_determined_l210_210619

-- Define the properties of the cards
def card_values := {a b c : ℕ // a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = 15 ∧ a < b ∧ b < c}

-- Define the statements provided by Alan, Carlos, and Brenda
-- Alan, Carlos, and Brenda's statements imply that from their respective positions they cannot determine the remaining two values
def alan_statement (a : ℕ) : Prop := 
  ∀ {b c : ℕ}, card_values ⟨a, b, c⟩ → ¬( ∀ y z, card_values ⟨y, z, a⟩ → y = b ∧ z = c)

def carlos_statement (c : ℕ) : Prop := 
  ∀ {a b : ℕ}, card_values ⟨a, b, c⟩ → ¬( ∀ x y, card_values ⟨x, y, c⟩ → x = a ∧ y = b)

def brenda_statement (b : ℕ) : Prop := 
  ∀ {a c : ℕ}, card_values ⟨a, b, c⟩ → ¬( ∀ x z, card_values ⟨x, z, b⟩ → x = a ∧ z = c)

-- Define the condition that given all the statements, it still is impossible to uniquely determine the middle card's number
theorem middle_card_not_determined : 
  ∀ b : ℕ, (∃ a c, card_values ⟨a, b, c⟩ ∧ alan_statement a ∧ carlos_statement c ∧ brenda_statement b) 
  → (card_values ⟨_, b, _⟩ ∧ alan_statement _ ∧ carlos_statement _ ∧ brenda_statement _)
:= by sorry

end middle_card_not_determined_l210_210619


namespace g_neither_even_nor_odd_l210_210110

def g (x : ℝ) : ℝ := floor (2 * x) + (1 / 3)

theorem g_neither_even_nor_odd :
  ¬(∀ x : ℝ, g (-x) = g x) ∧ ¬(∀ x : ℝ, g (-x) = -g x) := by
  sorry

end g_neither_even_nor_odd_l210_210110


namespace initial_men_count_l210_210586

theorem initial_men_count (M : ℕ) (A : ℕ) (H1 : 58 - (20 + 22) = 2 * M) : M = 8 :=
by
  sorry

end initial_men_count_l210_210586


namespace log_geometric_sum_l210_210536

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop := ∃ q, q < 1 ∧ ∀ n, a (n+1) = a n * q

theorem log_geometric_sum :
  ∀ (a : ℕ → ℝ), geometric_sequence a → a 1 + a 2 = 11 → a 1 * a 2 = 10 →
  ∑ i in Finset.range 10, Real.log (a (i + 1)) = -35 :=
by
  intros a ha hsum hprod
  sorry

end log_geometric_sum_l210_210536


namespace probability_of_mixed_selection_distribution_of_X_expected_value_of_X_l210_210746

namespace ZongziProblem

open ProbabilityTheory

def total_zongzi : ℕ := 10
def red_bean_zongzi : ℕ := 2
def plain_zongzi : ℕ := 8
def selected_zongzi : ℕ := 3

-- Question 1
theorem probability_of_mixed_selection : 
  let C (n k : ℕ) := nat.choose n k in
  proof_problem := (C(2, 1) * C(8, 2) + C(2, 2) * C(8, 1)) / C(10, 3) = 8 / 15 :=
sorry

-- Question 2
def X : Type := {x : ℕ // x ≤ 2}  -- Representing the number of red bean zongzi selected

def P (x : ℕ) : Rational :=
  if x = 0 then 7 / 15 else
  if x = 1 then 7 / 15 else
  if x = 2 then 1 / 15 else
  0

theorem distribution_of_X : 
  ( ∑ x ∈ {0, 1, 2}, P(x) = 1 ) ∧
  ( P(0) = 7 / 15 ) ∧
  ( P(1) = 7 / 15 ) ∧
  ( P(2) = 1 / 15 ) :=
sorry

theorem expected_value_of_X :
  ∑ x ∈ {0, 1, 2}, x * P(x) = 3 / 5 :=
sorry

end ZongziProblem

end probability_of_mixed_selection_distribution_of_X_expected_value_of_X_l210_210746


namespace scooter_gain_percent_l210_210649

def purchase_price : ℝ := 900
def repair_costs : ℝ := 300
def selling_price : ℝ := 1500

def total_cost := purchase_price + repair_costs
def gain := selling_price - total_cost
def gain_percent := (gain / total_cost) * 100

theorem scooter_gain_percent : gain_percent = 25 := by
  unfold gain_percent total_cost gain
  simp [purchase_price, repair_costs, selling_price]
  norm_num
  sorry

end scooter_gain_percent_l210_210649


namespace angle_quadrant_l210_210429

theorem angle_quadrant (θ : ℝ) (h : sin θ * cos θ < 0) : 
  (π/2 < θ ∧ θ < π) ∨ (3*π/2 < θ ∧ θ < 2*π) :=
sorry

end angle_quadrant_l210_210429


namespace lucas_seq_50_mod_5_l210_210981

def lucas_seq : ℕ → ℕ
| 0 := 2
| 1 := 5
| (n + 2) := lucas_seq (n + 1) + lucas_seq n

theorem lucas_seq_50_mod_5 : lucas_seq 50 % 5 = 0 := by
  sorry

end lucas_seq_50_mod_5_l210_210981


namespace sum_S21_l210_210607

-- Sequence aₙ satisfies the condition aₙ + aₙ₊₁ = n for n ∈ ℕ*
-- And also a₁ = 1
def sequence (a : ℕ+ → ℤ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ+, a n + a (n + 1) = n.val

-- Define the function Sₙ as the sum of the first n terms of the sequence
def S (a : ℕ+ → ℤ) (n : ℕ+) : ℤ :=
  (Finset.range n.val).sum (λ k, a ⟨k + 1, Nat.succ_pos k⟩)

-- Prove that S₂₁ = 100 given the conditions
theorem sum_S21 (a : ℕ+ → ℤ) (h : sequence a) : S a 21 = 100 :=
sorry

end sum_S21_l210_210607


namespace option_A_option_B_option_D_l210_210268

-- Option A
theorem option_A (a b : ℤ × ℤ) (h₁ : a = (1, 2)) (h₂ : b = (3, 1)) : ¬ collinear a b :=
by sorry

-- Option B
theorem option_B (A B C D : ℤ × ℤ) 
  (h₁ : A = (5, -1)) (h₂ : B = (-1, 7)) (h₃ : C = (1, 2)) : 
  parallelogram A B C D → D = (7, -6) :=
by sorry

-- Option D
theorem option_D (a b : ℝ × ℝ) (h₁ : a = (1, 1)) 
  (h₂ : |b| = 4) (θ : ℝ) (h₃ : θ = π / 4) : 
  projection a b = (2, 2) :=
by sorry

end option_A_option_B_option_D_l210_210268


namespace smallest_k_mul_integral_l210_210737

noncomputable def a : ℕ → ℝ
| 0       := 1
| 1       := real.root 17 3
| (n + 2) := a (n + 1) * (a n) ^ 2

theorem smallest_k_mul_integral :
  ∃ (k : ℕ), (∀ n, 1 ≤ n → n ≤ k → (a n ∈ {
    finite ℝ → ℤ}) ∧ k = 14) :=
begin
  sorry
end

end smallest_k_mul_integral_l210_210737


namespace solve_for_x_and_sum_mnp_l210_210161

theorem solve_for_x_and_sum_mnp :
  (∃ x : ℝ, x * (5 * x - 11) = 2) → 
  let m := 11 
  let n := 161 
  let p := 10 
  m + n + p = 182 :=
begin
  assume h,
  let m := 11,
  let n := 161,
  let p := 10,
  show m + n + p = 182,
  sorry
end

end solve_for_x_and_sum_mnp_l210_210161


namespace sandy_total_earnings_l210_210573

-- Define the conditions
def hourly_wage : ℕ := 15
def hours_friday : ℕ := 10
def hours_saturday : ℕ := 6
def hours_sunday : ℕ := 14

-- Define the total hours worked and total earnings
def total_hours := hours_friday + hours_saturday + hours_sunday
def total_earnings := total_hours * hourly_wage

-- State the theorem
theorem sandy_total_earnings : total_earnings = 450 := by
  sorry

end sandy_total_earnings_l210_210573


namespace isosceles_triangle_points_l210_210295

-- Declare the main theorem we want to prove
theorem isosceles_triangle_points (n k : ℕ) (h₁ : n ≥ 3) (vertices : Fin n → ℝ × ℝ) (points : Fin k → ℝ × ℝ)
  (h₂ : ConvexPolygon vertices) (h₃ : InsidePointsPolygon points vertices)
  (h₄ : ∀ (p1 p2 p3 : ℝ × ℝ), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p1 → isosceles p1 p2 p3 (vertices ++ points)) :
  k = 0 ∨ k = 1 := 
sorry

end isosceles_triangle_points_l210_210295


namespace negate_p_l210_210472

theorem negate_p (p : Prop) :
  (∃ x : ℝ, 0 < x ∧ 3^x < x^3) ↔ (¬ (∀ x : ℝ, 0 < x → 3^x ≥ x^3)) :=
by sorry

end negate_p_l210_210472


namespace sum_first_ten_terms_arithmetic_sequence_l210_210439

theorem sum_first_ten_terms_arithmetic_sequence (a d : ℝ) (S10 : ℝ) 
  (h1 : 0 < d) 
  (h2 : (a - d) + a + (a + d) = -6) 
  (h3 : (a - d) * a * (a + d) = 10) 
  (h4 : S10 = 5 * (2 * (a - d) + 9 * d)) :
  S10 = -20 + 35 * Real.sqrt 6.5 :=
by sorry

end sum_first_ten_terms_arithmetic_sequence_l210_210439


namespace relationship_between_heights_is_correlated_l210_210605

theorem relationship_between_heights_is_correlated :
  (∃ r : ℕ, (r = 1 ∨ r = 2 ∨ r = 3 ∨ r = 4) ∧ r = 2) := by
  sorry

end relationship_between_heights_is_correlated_l210_210605


namespace no_real_solution_l210_210411

theorem no_real_solution (x : ℝ) :
  ¬ ∃ x : ℝ, (Complex.mk x 1) * (Complex.mk (x + 1) 1) * (Complex.mk (x + 2) 1) * (Complex.mk (x + 3) 1) \in {z : ℂ | z.re = 0} :=
by sorry

end no_real_solution_l210_210411


namespace problem_sol_l210_210377

-- Assume g is an invertible function
variable (g : ℝ → ℝ) (g_inv : ℝ → ℝ)
variable (h_invertible : ∀ y, g (g_inv y) = y ∧ g_inv (g y) = y)

-- Define p and q such that g(p) = 3 and g(q) = 5
variable (p q : ℝ)
variable (h1 : g p = 3) (h2 : g q = 5)

-- Goal to prove that p - q = 2
theorem problem_sol : p - q = 2 :=
by
  sorry

end problem_sol_l210_210377


namespace cone_base_diameter_l210_210676

theorem cone_base_diameter {r l : ℝ} 
  (h₁ : π * r * l + π * r^2 = 3 * π) 
  (h₂ : 2 * π * r = π * l) : 
  2 * r = 2 :=
by
  sorry

end cone_base_diameter_l210_210676


namespace Ann_keeps_total_cookies_l210_210708

theorem Ann_keeps_total_cookies 
  (baked_or_cookies : ℕ = 40) 
  (baked_sugar_cookies : ℕ = 28) 
  (baked_cc_cookies : ℕ = 55)
  (gave_away_or_cookies : ℕ = 26) 
  (gave_away_sugar_cookies : ℕ = 17) 
  (gave_away_cc_cookies : ℕ = 34) : 
  (baked_or_cookies - gave_away_or_cookies) +
  (baked_sugar_cookies - gave_away_sugar_cookies) +
  (baked_cc_cookies - gave_away_cc_cookies) = 46 :=
by
  -- Definitions according to the conditions
  let cookies_kept_or := baked_or_cookies - gave_away_or_cookies
  let cookies_kept_sugar := baked_sugar_cookies - gave_away_sugar_cookies
  let cookies_kept_cc := baked_cc_cookies - gave_away_cc_cookies
  -- Summing up the cookies kept
  calc
    cookies_kept_or + cookies_kept_sugar + cookies_kept_cc
    = 14 + 11 + 21 : by sorry
    = 46 : by sorry

end Ann_keeps_total_cookies_l210_210708


namespace sugar_total_more_than_two_l210_210371

noncomputable def x (p q : ℝ) : ℝ :=
p / q

noncomputable def y (p q : ℝ) : ℝ :=
q / p

theorem sugar_total_more_than_two (p q : ℝ) (hpq : p ≠ q) :
  x p q + y p q > 2 :=
by sorry

end sugar_total_more_than_two_l210_210371


namespace solvable_implies_mod_six_one_infinitude_of_primes_six_k_plus_one_l210_210542

-- Define the conditions and problem
def prime_gt_three (p : ℕ) : Prop :=
  Nat.Prime p ∧ p > 3

def congruence_solvable (p : ℕ) : Prop :=
  ∃ x : ℕ, (x^2 + x + 1) % p = 0

-- Statement for part a
theorem solvable_implies_mod_six_one (p : ℕ) (h_prime : prime_gt_three p) (h_congruence: congruence_solvable p) : 
  p % 6 = 1 := sorry

-- Statement for part b
theorem infinitude_of_primes_six_k_plus_one :
  ∃ᶠ p in Nat.Prime, p % 6 = 1 := sorry

end solvable_implies_mod_six_one_infinitude_of_primes_six_k_plus_one_l210_210542


namespace triangle_area_is_correct_l210_210882

structure Point where
  x : ℝ
  y : ℝ

noncomputable def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * (abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)))

def A : Point := ⟨0, 0⟩
def B : Point := ⟨2, 2⟩
def C : Point := ⟨2, 0⟩

theorem triangle_area_is_correct : area_of_triangle A B C = 2 := by
  sorry

end triangle_area_is_correct_l210_210882


namespace mary_score_is_95_l210_210121

theorem mary_score_is_95
  (s c w : ℕ)
  (h1 : s > 90)
  (h2 : s = 35 + 5 * c - w)
  (h3 : c + w = 30)
  (h4 : ∀ c' w', s = 35 + 5 * c' - w' → c + w = c' + w' → (c', w') = (c, w)) :
  s = 95 :=
by
  sorry

end mary_score_is_95_l210_210121


namespace solve_cubic_inequality_l210_210957

def polynomial_inequality (x : ℝ) : Prop :=
  -2 * x^3 + 5 * x^2 + 7 * x - 10 < 0

def interval1 (x : ℝ) : Prop :=
  x ∈ set.Ioo (-∞ : ℝ) (-1.35)

def interval2 (x : ℝ) : Prop :=
  x ∈ set.Ioo (1.85) (2 : ℝ)

theorem solve_cubic_inequality (x : ℝ) :
  polynomial_inequality x ↔ interval1 x ∨ interval2 x :=
sorry

end solve_cubic_inequality_l210_210957


namespace g_neither_even_nor_odd_l210_210112

def g (x : ℝ) : ℝ := floor (2 * x) + (1 / 3)

theorem g_neither_even_nor_odd :
  ¬(∀ x : ℝ, g (-x) = g x) ∧ ¬(∀ x : ℝ, g (-x) = -g x) := by
  sorry

end g_neither_even_nor_odd_l210_210112


namespace max_daily_sales_revenue_l210_210227

noncomputable def P (t : ℕ) : ℕ :=
  if 1 ≤ t ∧ t ≤ 24 then t + 2 else if 25 ≤ t ∧ t ≤ 30 then 100 - t else 0

noncomputable def Q (t : ℕ) : ℕ :=
  if 1 ≤ t ∧ t ≤ 30 then 40 - t else 0

noncomputable def y (t : ℕ) : ℕ :=
  P t * Q t

theorem max_daily_sales_revenue :
  ∃ t : ℕ, 1 ≤ t ∧ t ≤ 30 ∧ y t = 115 :=
sorry

end max_daily_sales_revenue_l210_210227


namespace range_of_x_l210_210433

noncomputable def f : ℝ → ℝ := sorry
def D : Set ℝ := {x | x < 0 ∨ x > 0}

axiom f_property : ∀ x y ∈ D, f (x * y) = f x + f y

-- I. Proving the values of f(1) and f(-1)
axiom f_1 : f 1 = 0
axiom f_neg_1 : f (-1) = 0

-- II. Proving the parity of f (odd function)
axiom f_odd : ∀ x ∈ D, f (-x) = -f x

-- III. Proving the range of x given the conditions
axiom f_4 : f 4 = 1
axiom f_inequality : ∀ x ∈ (0, ∞), f (3 * x + 1) + f (2 * x - 6) ≤ 3
axiom f_increasing : ∀ x y ∈ (0, ∞), x < y → f x < f y

theorem range_of_x : {x : ℝ | 3 < x ∧ x ≤ 5} := sorry

end range_of_x_l210_210433


namespace sin_angle_ACB_l210_210513

-- Definitions of the conditions
variable (α β x y AD BD : ℝ)
variable (h1 : x = Real.cos α)
variable (h2 : y = Real.cos β)

-- The problem statement
theorem sin_angle_ACB {α β : ℝ} {x y AD BD : ℝ} :
  x = Real.cos α →
  y = Real.cos β →
  sin (Real.arccos ((AD^2 * (1 / (Real.cos α)^2 - 1) + BD^2 * (1 / (Real.cos β)^2 - 1)) / (2 * AD * BD / (Real.cos α * Real.cos β)))) = sqrt(1 - ( (tan α + tan β) / (2 * x * y) )^2) :=
by
  intros h1 h2
  sorry

end sin_angle_ACB_l210_210513


namespace concurrency_of_lines_l210_210564

open EuclideanGeometry

variables {P B C D E F D1 E1 F1 : Point}
variables (ABC : Triangle)
variables (D_on_BC : OnSegment D B C)
variables (E_on_CA : OnSegment E C A)
variables (F_on_AB : OnSegment F A B)
variables (EF_parallel_BC : Parallel EF BC)
variables (D1_on_BC : OnSegment D1 B C)
variables (D1E1_parallel_DE : Parallel (LineThrough D1 E1) DE)
variables (D1F1_parallel_DF : Parallel (LineThrough D1 F1) DF)
variables (triangle_sim_PBC_DEF : Similar (Triangle P B C) (Triangle D E F))

theorem concurrency_of_lines :
  Concurrent EF E1F1 (LineThrough P D1) :=
sorry

end concurrency_of_lines_l210_210564


namespace question_l210_210027

variable (U : Set ℕ) (M : Set ℕ)

theorem question :
  U = {1, 2, 3, 4, 5} →
  (U \ M = {1, 3}) →
  2 ∈ M :=
by
  intros
  sorry

end question_l210_210027


namespace proof_2_in_M_l210_210036

def U : Set ℕ := {1, 2, 3, 4, 5}

def M : Set ℕ := { x | x ∈ U ∧ x ≠ 1 ∧ x ≠ 3 }

theorem proof_2_in_M : 2 ∈ M :=
by
  sorry

end proof_2_in_M_l210_210036


namespace speed_first_half_journey_l210_210697

theorem speed_first_half_journey : 
  ∀ (v : ℝ),
    (v > 0) ∧
    (let total_time := 25 in
     let total_distance := 560 in
     let half_distance := total_distance / 2 in
     let second_half_speed := 24 in
     let second_half_time := half_distance / second_half_speed in
     let first_half_time := total_time - second_half_time in
     let calculated_speed := half_distance / first_half_time in
     calculated_speed = v) → v = 21 :=
by 
  intros v h,
  sorry

end speed_first_half_journey_l210_210697


namespace celia_savings_l210_210382

def total_spending (food_per_week rent streaming cell_phone : ℕ) : ℕ :=
  let food_per_month := food_per_week * 4
  food_per_month + rent + streaming + cell_phone

def savings_amount (total_spent: ℕ) : ℕ :=
  (total_spent : ℚ) * 0.10

theorem celia_savings
  (food_per_week : ℕ)
  (rent : ℕ)
  (streaming : ℕ)
  (cell_phone : ℕ)
  (h_food_per_week : food_per_week = 100)
  (h_rent : rent = 1500)
  (h_streaming : streaming = 30)
  (h_cell_phone : cell_phone = 50) :
  savings_amount (total_spending food_per_week rent streaming cell_phone) = 198 :=
by
  unfold total_spending savings_amount
  rw [h_food_per_week, h_rent, h_streaming, h_cell_phone]
  norm_num
  -- complete the proof here
  sorry

end celia_savings_l210_210382


namespace tenth_term_arithmetic_sequence_l210_210399

def arithmetic_sequence (a1 d : ℚ) (n : ℕ) : ℚ :=
  a1 + (n - 1) * d

theorem tenth_term_arithmetic_sequence :
  arithmetic_sequence (1 / 2) (1 / 2) 10 = 5 :=
by
  sorry

end tenth_term_arithmetic_sequence_l210_210399


namespace ratio_of_mystery_books_l210_210552

def total_books_on_cart : Nat := 46
def history_books : Nat := 12
def romance_books : Nat := 8
def poetry_books : Nat := 4
def western_novels : Nat := 5
def biographies : Nat := 6

theorem ratio_of_mystery_books :
  let top_section_books := history_books + romance_books + poetry_books,
      bottom_section_books := total_books_on_cart - top_section_books,
      other_bottom_books := western_novels + biographies,
      mystery_books := bottom_section_books - other_bottom_books,
      ratio := (mystery_books, bottom_section_books)
  in ratio = (1, 2) := by
  sorry

end ratio_of_mystery_books_l210_210552


namespace total_recommendation_methods_l210_210888

theorem total_recommendation_methods
  (C1: ∃ (n : ℕ), n = 5)
  (C2: ∀ (univ : String), (univ = "Tsinghua" ∨ univ = "Peking") → (univ = "Tsinghua" → recommends_male n) ∧ (univ = "Peking" → recommends_male n))
  (C3 : ∃ (m f : ℕ), m = 3 ∧ f = 2):
  ∃ (total_methods : ℕ), total_methods = 24 :=
by
  sorry

end total_recommendation_methods_l210_210888


namespace math_proof_problem_l210_210889

/-- Define the conditions for the problem -/
def C1_rect_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x = 0

def C2_polar_eq (rho theta : ℝ) : Prop := rho^2 = 3 / (1 + 2 * (Real.sin theta)^2)

def theta_ray (theta : ℝ) : Prop := theta = π / 3

/-- The parametric equation of C1 -/
def C1_param_eq (α : ℝ) : Prop := 
  ∃ (x y : ℝ), x = 1 + Real.cos α ∧ y =  Real.sin α

/-- The rectangular equation of C2 -/
def C2_rect_eq (x y : ℝ) : Prop := (x^2 / 3) + y^2 = 1

/-- The distance between the intersection points of the ray with C1 and C2 -/
def distance_AB : ℝ := |(sqrt 30 / 5) - 1|

theorem math_proof_problem : 
  (∀ (x y : ℝ), C1_rect_eq x y → 
    ∃ (α : ℝ), C1_param_eq α) ∧

  (∀ (rho θ : ℝ), C2_polar_eq rho θ → 
    ∃ (x y : ℝ), C2_rect_eq x y) ∧

  (theta_ray (π / 3) → 
    let A_ρ : ℝ := 2 * Real.cos (π / 3) in
    let B_ρ : ℝ := sqrt 30 / 5 in
    ∃ (A B : ℝ), |A_ρ - B_ρ| = distance_AB) :=
by 
  sorry

end math_proof_problem_l210_210889


namespace find_m_l210_210473

noncomputable def union_sets (A B : Set ℝ) : Set ℝ :=
  {x | x ∈ A ∨ x ∈ B}

theorem find_m :
  ∀ (m : ℝ),
    (A = {1, 2 ^ m}) →
    (B = {0, 2}) →
    (union_sets A B = {0, 1, 2, 8}) →
    m = 3 :=
by
  intros m hA hB hUnion
  sorry

end find_m_l210_210473


namespace complex_magnitude_l210_210328

open Complex

theorem complex_magnitude :
  ∀ (i : ℂ), i^2 = -1 → i^3 = -i → |2 + i^2 + 2 * i^3| = Real.sqrt 5 :=
by
  intros i h1 h2
  -- skipping proof with sorry
  sorry

end complex_magnitude_l210_210328


namespace correct_mark_l210_210684

theorem correct_mark (x : ℝ) (n : ℝ) (avg_increase : ℝ) :
  n = 40 → avg_increase = 1 / 2 → (83 - x) / n = avg_increase → x = 63 :=
by
  intros h1 h2 h3
  sorry

end correct_mark_l210_210684


namespace second_machine_time_l210_210348

theorem second_machine_time
  (machine1_rate : ℕ)
  (machine2_rate : ℕ)
  (combined_rate12 : ℕ)
  (combined_rate123 : ℕ)
  (rate3 : ℕ)
  (time3 : ℚ) :
  machine1_rate = 60 →
  machine2_rate = 120 →
  combined_rate12 = 200 →
  combined_rate123 = 600 →
  rate3 = 420 →
  time3 = 10 / 7 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end second_machine_time_l210_210348


namespace g_neither_even_nor_odd_l210_210101

def g (x : ℝ) : ℝ := ⌊2 * x⌋ + 1/3

theorem g_neither_even_nor_odd : (∀ x, g x ≠ g (-x)) ∧ (∀ x, g x ≠ -g (-x)) :=
by
  sorry

end g_neither_even_nor_odd_l210_210101


namespace proof_MN_equal_12_l210_210516

theorem proof_MN_equal_12
  (P : Point)
  (L : Point)
  (K : Point)
  (A B Q M N : Point)
  (semicircle : Circle)
  (diameter_AB : Line)
  (h1 : P ∈ semicircle)
  (h2 : L ⊥ diameter_AB ∧ L ∈ diameter_AB ∧ P ≠ L)
  (h3 : K = midpoint P B)
  (h4 : tangent_at semicircle A Q ∧ tangent_at semicircle P Q)
  (h5 : PL ∩ QB = M)
  (h6 : KL ∩ QB = N)
  (h7 : AQ / AB = 5 / 12)
  (h8 : distance Q M = 25) :
  distance M N = 12 :=
by
  sorry

end proof_MN_equal_12_l210_210516


namespace harmonic_mean_CE_l210_210100

theorem harmonic_mean_CE (a b : ℝ) (ABC : Triangle) (D E : Point)
  (h1 : D ∈ interiorAngleBisector ABC ACB)
  (h2 : perpendicularFrom (line D (point C)) (line D (lineIntersect AB)))
  (h3 : E = lineIntersection (perpendicularLineFrom D (point C)) (line CA)) :
  dist (point C) E = (2 * a * b) / (a + b) :=
by
  sorry

end harmonic_mean_CE_l210_210100


namespace circle_area_RS_l210_210264

-- Define the points R and S
def R : ℝ × ℝ := (1, 2)
def S : ℝ × ℝ := (-7, 6)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Define the area of a circle function
def circle_area (r : ℝ) : ℝ := π * r^2

-- Prove that the area of the circle with center at R and passing through S is 80π
theorem circle_area_RS : circle_area (distance R S) = 80 * π :=
  sorry

end circle_area_RS_l210_210264


namespace range_of_a_l210_210459

noncomputable def f (a x : ℝ) : ℝ := exp x - a * exp x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
by
  sorry

end range_of_a_l210_210459


namespace B_transpose_inverse_l210_210535

open Matrix

variables {R : Type*} [Field R]
variables {x y z p q r s t u : R}

def B : Matrix (Fin 3) (Fin 3) R :=
  ![![x, y, z], ![p, q, r], ![s, t, u]]

theorem B_transpose_inverse :
  (B.transpose = B⁻¹) →
  (x^2 + y^2 + z^2 + p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 3) :=
by
  sorry

end B_transpose_inverse_l210_210535


namespace shrimp_appetizer_cost_l210_210629

-- Define the conditions
def shrimp_per_guest : ℕ := 5
def number_of_guests : ℕ := 40
def cost_per_pound : ℕ := 17
def shrimp_per_pound : ℕ := 20

-- Define the proof statement
theorem shrimp_appetizer_cost : 
  (shrimp_per_guest * number_of_guests / shrimp_per_pound) * cost_per_pound = 170 := 
by
  sorry

end shrimp_appetizer_cost_l210_210629


namespace complex_abs_sqrt_five_l210_210321

open Complex

theorem complex_abs_sqrt_five : abs (2 + (-1 : ℂ) + 2 * (-I : ℂ)) = Real.sqrt 5 := 
by
  sorry

end complex_abs_sqrt_five_l210_210321


namespace intersect_sets_l210_210530

open Set

theorem intersect_sets (A B : Set ℝ) (hA : A = {x | abs x < 3}) (hB : B = {x | 2^x > 1}) :
  A ∩ B = {x | 0 < x ∧ x < 3} := 
by
  sorry

end intersect_sets_l210_210530


namespace trig_inequality_l210_210508

-- Define the problem parameters
variables {A B C : ℝ}
variables (tan_A tan_B tan_C : ℝ)

-- Assume the triangle is acute and tan of angles satisfy the given condition
axiom triangle_acute : ∀ {A B C : ℝ}, 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2

axiom tan_identity : tan_A * tan_B * tan_C = tan_A + tan_B + tan_C

-- State the theorem to be proved
theorem trig_inequality 
    (hA: 0 < A ∧ A < π/2)
    (hB: 0 < B ∧ B < π/2)
    (hC: 0 < C ∧ C < π/2)
    (tan_identity : tan_A * tan_B * tan_C = tan_A + tan_B + tan_C) :
    (1/3 * ((tan_A ^ 2) / (tan_B * tan_C) + (tan_B ^ 2) / (tan_C * tan_A) + (tan_C ^ 2) / (tan_A * tan_B)) 
    + 3 * (1 / (tan_A + tan_B + tan_C)) ^ (2/3) ≥ 2) :=
by
  sorry

end trig_inequality_l210_210508


namespace ratio_of_first_part_l210_210570

noncomputable def partB : ℝ := 164.6315789473684
noncomputable def sum_parts : ℝ := 782
noncomputable def ratioB : ℝ := 1 / 3
noncomputable def ratioC : ℝ := 3 / 4

theorem ratio_of_first_part :
  let A := sum_parts - partB - (partB * (ratioC / ratioB)) in
  (A / sum_parts) ≈ 0.6125 :=
by 
  sorry

end ratio_of_first_part_l210_210570


namespace correct_statement_l210_210014

universe u
variable (α : Type u)

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5}

noncomputable def complement_U_M : Set ℕ := {1, 3}

noncomputable def M : Set ℕ := U α \ complement_U_M

theorem correct_statement : 2 ∈ M :=
by {
  sorry
}

end correct_statement_l210_210014


namespace ilya_defeats_dragon_l210_210874

/-- Ilya Muromets will eventually defeat the dragon. -/
theorem ilya_defeats_dragon :
  ∀ (initial_heads : ℕ), 
    (prob_two_head_growth: ℝ) 
      (prob_one_head_growth: ℝ) 
      (prob_no_head_growth: ℝ), 
  prob_two_head_growth = 1/4 → 
  prob_one_head_growth = 1/3 → 
  prob_no_head_growth = 5/12 →
  (∀ (n : ℕ), (initial_heads = 0 → ilya_defeats_dragon 0)) →
  initial_heads > 0 →
  ∃ (p : ℝ), p = 1 :=
begin
  sorry
end

end ilya_defeats_dragon_l210_210874


namespace season_duration_l210_210249

-- Define the given conditions.
def games_per_month : ℕ := 7
def games_per_season : ℕ := 14

-- Define the property we want to prove.
theorem season_duration : games_per_season / games_per_month = 2 :=
by
  sorry

end season_duration_l210_210249


namespace inequality_proof_l210_210620

theorem inequality_proof 
  {a b c : ℝ}
  (ha : 0 ≤ a)
  (hb : 0 ≤ b)
  (hc : 0 ≤ c)
  (h1 : a^2 ≤ b^2 + c^2)
  (h2 : b^2 ≤ c^2 + a^2)
  (h3 : c^2 ≤ a^2 + b^2) :
  (a + b + c) * (a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) ≥ 4 * (a^6 + b^6 + c^6) :=
sorry

end inequality_proof_l210_210620


namespace part1_part2_l210_210434

variable {R : Type*} [LinearOrderedField R]

noncomputable def f (x : R) : R := sorry 

-- Condition 1: f(x) + f(y) = f(xy) for all x, y in R^+
axiom cond1 (x y : R) (hx : 0 < x) (hy : 0 < y) : f(x) + f(y) = f(x * y)

-- Condition 2: f(x) > 0 for x > 1
axiom cond2 (x : R) (hx : 1 < x) : 0 < f(x)

-- Given: f(3) = 1
axiom given1 : f(3) = 1

-- Prove: f(1 / x) = -f(x)
theorem part1 (x : R) (hx : 0 < x) : f(1 / x) = -f(x) := sorry

-- Prove: f(x) is increasing on R^+
theorem part2 (x1 x2 : R) (hx1 : 0 < x1) (hx2 : 0 < x2) (h : x1 < x2) : f(x1) < f(x2) := sorry

-- Solve: f(x) - f(1 / (2 * a - x)) ≥ 2 for x
def solve_part3 (a : R) (ha : 0 < a) : Subtype (λ x : R, f(x) - f(1 / (2 * a - x)) ≥ 2) := 
{val := (a - Real.sqrt (a^2 - 9), a + Real.sqrt (a^2 - 9)),
 property := sorry}

end part1_part2_l210_210434


namespace skateboard_distance_proof_l210_210691

theorem skateboard_distance_proof : 
  let a1 := 8
  let d := 9
  let n := 20 
  let a_n := a1 + (n - 1) * d
  let sum_n := (n * (a1 + a_n)) / 2
  in sum_n = 1870 := 
by
  let a1 := 8
  let d := 9
  let n := 20 
  let a_n := a1 + (n - 1) * d
  let sum_n := (n * (a1 + a_n)) / 2
  sorry

end skateboard_distance_proof_l210_210691


namespace raft_drift_time_l210_210234

-- Define the distance between the villages
def distance : ℝ := 1

-- Define the speed of the steamboat in still water (in units/hour)
def v_s : ℝ := sorry

-- Define the time taken by the steamboat to travel the distance (in hours)
def t_steamboat : ℝ := 1

-- Define the speed of the motorboat in still water (in units/hour)
def v_m : ℝ := 2 * v_s

-- Define the time taken by the motorboat to travel the distance (in hours)
def t_motorboat : ℝ := 45 / 60

-- Define the speed of the river's current (in units/hour)
def v_f : ℝ := sorry

-- Equations for steamboat and motorboat effective speeds
def steamboat_equation := v_s + v_f = distance / t_steamboat
def motorboat_equation := v_m + v_f = distance / t_motorboat

-- Solve for the speeds
def v_s_solution : ℝ := 1 - v_f
def v_f_solution : ℝ := (4 / 3) - 2 * v_s_solution

-- Define the time for the raft to drift from Verkhnie Vasyuki to Nizhnie Vasyuki (in hours)
def raft_time_hours : ℝ := distance / v_f_solution

-- Convert the time to minutes
def raft_time_minutes : ℝ := raft_time_hours * 60

-- Theorem statement that proves the raft drifts in 90 minutes
theorem raft_drift_time : raft_time_minutes = 90 :=
by
  unfold distance v_s t_steamboat v_m t_motorboat v_f steamboat_equation motorboat_equation v_s_solution v_f_solution raft_time_hours raft_time_minutes
  rw [←solve_v_s, ←solve_v_f]
  sorry

end raft_drift_time_l210_210234


namespace mark_sprinted_distance_l210_210933

def speed := 6 -- miles per hour
def time := 4 -- hours

/-- Mark sprinted exactly 24 miles. -/
theorem mark_sprinted_distance : speed * time = 24 := by
  sorry

end mark_sprinted_distance_l210_210933


namespace dragon_resilience_l210_210765

noncomputable def probability_function (x : ℝ) : ℝ :=
  let p0 := 1 / (1 + x + x^2)
  let p1 := x / (1 + x + x^2)
  let p2 := x^2 / (1 + x + x^2)
  p1 * p2^2 * p1 * p0 * p2 * p1 * p0 * p1 * p2

theorem dragon_resilience (x : ℝ) (hx : x > 0) : 
  has_max (λ x, probability_function x) ∧ probability_function (sqrt 97 + 1) / 8 = max :=
sorry

end dragon_resilience_l210_210765


namespace coefficient_of_x3_in_expansion_l210_210183

theorem coefficient_of_x3_in_expansion :
  let binom : ℕ → ℕ → ℕ := nat.choose
  let expansion := (x - 2) * ((x + 1) ^ 5)
  -- Extracting the coefficient of x^3
  (expansion.coeff 3) = -10 :=
by
  sorry

end coefficient_of_x3_in_expansion_l210_210183


namespace x_value_l210_210501

def x_is_75_percent_greater (x : ℝ) (y : ℝ) : Prop := x = y + 0.75 * y

theorem x_value (x : ℝ) : x_is_75_percent_greater x 150 → x = 262.5 :=
by
  intro h
  rw [x_is_75_percent_greater] at h
  sorry

end x_value_l210_210501


namespace complex_expression_magnitude_l210_210301

def i := Complex.I

theorem complex_expression_magnitude :
  |2 + i^2 + 2 * i^3| = Real.sqrt 5 :=
by
  sorry

end complex_expression_magnitude_l210_210301


namespace polynomial_identity_l210_210351

variable (m : ℝ)

theorem polynomial_identity : ∀ P : ℝ → ℝ, (∀ m, P(m) - 3 * m = 5 * m^2 - 3 * m - 5) → (∀ m, P(m) = 5 * m^2 - 5) :=
by
  intros P h
  sorry

end polynomial_identity_l210_210351


namespace length_MD_is_sqrt_8_75_l210_210875

noncomputable theory
open Real

-- Define the points and sides' lengths
def A := (0, 0)
def B := (17, 0)
def C := (21, Real.sqrt 8.75)
def AB := (17:ℝ)
def BC := (21 + Real.sqrt 8.75)
def CA := (Real.sqrt (20^2 + 21^2))

-- Midpoint of AB
def M := ((0 + 17)/2, 0)

-- Coordinates of D on BC
def D := (BC - (BC - 20)/2, 0)

-- Segment MD length calculation
def length_MD : ℝ := Real.sqrt ((M.1 - D.1)^2 + (M.2 - D.2)^2)

-- Statement of the problem
theorem length_MD_is_sqrt_8_75 : length_MD = Real.sqrt (8.75) :=
sorry

end length_MD_is_sqrt_8_75_l210_210875


namespace complex_magnitude_problem_l210_210315

-- Define the imaginary unit with the property i^2 = -1
def i : ℂ := complex.I

-- Prove that the magnitude of the complex number 2 + i² + 2i³ is √5
theorem complex_magnitude_problem : 
  complex.abs (2 + i^2 + 2 * i^3) = real.sqrt 5 := 
by
  -- Use the provided condition i² = -1
  have h : i^2 = -1 := by sorry,
  sorry

end complex_magnitude_problem_l210_210315


namespace exist_int_set_l210_210810

theorem exist_int_set 
  (a1 b1 c1 a2 b2 c2 : ℝ)
  (h : ∀ x y : ℤ, (∃ k : ℤ, a1 * x + b1 * y + c1 = 2 * k) ∨ (∃ k : ℤ, a2 * x + b2 * y + c2 = 2 * k)) : 
  (∀ z ∈ [a1, b1, c1], z ∈ ℤ) ∨ (∀ z ∈ [a2, b2, c2], z ∈ ℤ) := 
sorry

end exist_int_set_l210_210810


namespace triangle_centroid_l210_210413

theorem triangle_centroid :
  let (x1, y1) := (2, 6)
  let (x2, y2) := (6, 2)
  let (x3, y3) := (4, 8)
  let centroid_x := (x1 + x2 + x3) / 3
  let centroid_y := (y1 + y2 + y3) / 3
  (centroid_x, centroid_y) = (4, 16 / 3) :=
by
  let x1 := 2
  let y1 := 6
  let x2 := 6
  let y2 := 2
  let x3 := 4
  let y3 := 8
  let centroid_x := (x1 + x2 + x3) / 3
  let centroid_y := (y1 + y2 + y3) / 3
  show (centroid_x, centroid_y) = (4, 16 / 3)
  sorry

end triangle_centroid_l210_210413


namespace constant_term_in_expansion_l210_210789

theorem constant_term_in_expansion :
  ∃ n : ℕ, ( (1 + x) + (1 + x)^2 + (1 + x)^3 + ... + (1 + x)^n = a_0 + a_1 * x + a_2 * x^2 + ... + a_n * x^n ) ∧
           ( ∑ i in finset.range (n + 1), a_i = 126 ) ∧
           (- nat.choose 6 (6 / 2) = -20 ) :=
sorry

end constant_term_in_expansion_l210_210789


namespace closest_whole_number_to_shaded_area_l210_210340

noncomputable def radius_of_circle (d : ℝ) : ℝ := d / 2

noncomputable def area_of_circle (r : ℝ) : ℝ := π * r^2

noncomputable def area_of_rectangle (length width : ℝ) : ℝ := length * width

noncomputable def area_of_shaded_region (length width d : ℝ) : ℝ :=
  area_of_rectangle length width - area_of_circle (radius_of_circle d)

theorem closest_whole_number_to_shaded_area
  (length width d : ℝ)
  (h_length : length = 4)
  (h_width : width = 5)
  (h_diameter : d = 2) :
  |(17 : ℝ) - (area_of_shaded_region length width d)| ≤ 0.5 :=
by
  sorry

end closest_whole_number_to_shaded_area_l210_210340


namespace numberOfShapesSymmetricAboutAxisButNotCenter_l210_210367

inductive Shape
| EquilateralTriangle
| Parallelogram
| RegularPentagon
| Circle

def isSymmetricAboutAxis : Shape → Prop
| Shape.EquilateralTriangle := True
| Shape.Parallelogram := False
| Shape.RegularPentagon := True
| Shape.Circle := True

def isSymmetricAboutCenter : Shape → Prop
| Shape.EquilateralTriangle := False
| Shape.Parallelogram := True
| Shape.RegularPentagon := False
| Shape.Circle := True

def isSymmetricAboutAxisButNotCenter : Shape → Prop :=
λ s, isSymmetricAboutAxis s ∧ ¬ isSymmetricAboutCenter s

noncomputable def countShapes : Nat :=
[Shape.EquilateralTriangle,
 Shape.Parallelogram,
 Shape.RegularPentagon,
 Shape.Circle].countP isSymmetricAboutAxisButNotCenter

theorem numberOfShapesSymmetricAboutAxisButNotCenter :
  countShapes = 2 :=
by sorry

end numberOfShapesSymmetricAboutAxisButNotCenter_l210_210367


namespace circle_through_intersections_l210_210042

def l1 (x y : ℝ) : Prop := x - 2 * y = 0
def l2 (x y : ℝ) : Prop := y + 1 = 0
def l3 (x y : ℝ) : Prop := 2 * x + y - 1 = 0

def is_intersection_point (x y : ℝ) (l₁ l₂ : ℝ → ℝ → Prop) : Prop :=
  l₁ x y ∧ l₂ x y

def circle (x y : ℝ) : Prop := x^2 + y^2 + x + 2 * y - 1 = 0

theorem circle_through_intersections : 
  (∃ x y, is_intersection_point x y l1 l2 ∧ circle x y) ∧
  (∃ x y, is_intersection_point x y l1 l3 ∧ circle x y) ∧
  (∃ x y, is_intersection_point x y l2 l3 ∧ circle x y) :=
by sorry

end circle_through_intersections_l210_210042


namespace p_sufficient_but_not_necessary_for_q_l210_210441

-- Definitions of propositions p and q
def p (x : ℝ) := x = 1
def q (x : ℝ) := x - 1 = real.sqrt (x - 1)

-- The main theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_but_not_necessary_for_q (x : ℝ) : p x → q x :=
by
  intro hpx
  rw [p, q] at *
  rw hpx
  norm_num

end p_sufficient_but_not_necessary_for_q_l210_210441


namespace length_of_n_l210_210795

noncomputable def f (n : ℕ) : ℕ :=
  Nat.find (λ k, ¬ k | n ∧ k > 0)

def length (n : ℕ) : ℕ :=
  if n % 2 = 1 then 1
  else
    let α := Nat.find (λ k, (n / 2^k) % 2 = 1 ∧ (2^k ∣ n)) - 1;
    let m := n / (2^(α + 1));
    let odd_divisors := List.filter Nat.odd (List.range (2^(α + 1) - 1));
    if odd_divisors.all (λ t, t > 1 → t ∣ m) then 3 else 2

theorem length_of_n (n : ℕ) (h : n ≥ 3) : 
  length n = 
    if n % 2 = 1 then 1
    else 
      let α := Nat.find (λ k, (n / 2^k) % 2 = 1 ∧ (2^k ∣ n)) - 1;
      let m := n / (2^(α + 1));
      let odd_divisors := List.filter Nat.odd (List.range (2^(α + 1) - 1));
      if odd_divisors.all (λ t, t > 1 → t ∣ m) then 3 else 2 :=
sorry

end length_of_n_l210_210795


namespace number_of_valid_n_l210_210783

-- Define the conditions for n
def is_valid_n (n : ℕ) : Prop := (1 ≤ n ∧ n ≤ 150) ∧ (Nat.gcd 18 n = 2)

-- Define the function to count the number of integers satisfying the conditions
def count_valid_n : ℕ :=
  (Finset.filter is_valid_n (Finset.range 151)).card

-- The theorem stating the number of integers n between 1 and 150 such that gcd(18, n) = 2 is 67
theorem number_of_valid_n : count_valid_n = 67 := 
sorry

end number_of_valid_n_l210_210783


namespace perimeter_of_larger_similar_triangle_l210_210707

def isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ a = c)

theorem perimeter_of_larger_similar_triangle :
  ∃ (perimeter : ℝ), 
  let a := 12, b := 12, c := 15,
      larger_a := 45 in
  isosceles_triangle a b c ∧
  (larger_a / c = 3) ∧
  (perimeter = 3 * (a + b + c)) ∧ 
  (perimeter = 117) := sorry

end perimeter_of_larger_similar_triangle_l210_210707


namespace total_students_in_Lansing_l210_210903

def n_schools : Nat := 25
def students_per_school : Nat := 247
def total_students : Nat := n_schools * students_per_school

theorem total_students_in_Lansing :
  total_students = 6175 :=
  by
    -- we can either compute manually or just put sorry for automated assistance
    sorry

end total_students_in_Lansing_l210_210903


namespace complex_magnitude_problem_l210_210311

-- Define the imaginary unit with the property i^2 = -1
def i : ℂ := complex.I

-- Prove that the magnitude of the complex number 2 + i² + 2i³ is √5
theorem complex_magnitude_problem : 
  complex.abs (2 + i^2 + 2 * i^3) = real.sqrt 5 := 
by
  -- Use the provided condition i² = -1
  have h : i^2 = -1 := by sorry,
  sorry

end complex_magnitude_problem_l210_210311


namespace max_volume_cuboid_l210_210342

theorem max_volume_cuboid (x y z : ℕ) (h : 2 * (x * y + x * z + y * z) = 150) : x * y * z ≤ 125 :=
sorry

end max_volume_cuboid_l210_210342


namespace complex_find_value_l210_210134

theorem complex_find_value (z : ℂ) (h : 20 * (abs z)^2 = 3 * (abs (z + 3))^2 + (abs (z^2 + 2))^2 + 37) : 
  z + 9/z = -3 :=
sorry

end complex_find_value_l210_210134


namespace age_difference_10_l210_210226

theorem age_difference_10 (x : ℕ) :
  let Halima := 4 * x,
      Beckham := 3 * x,
      Michelle := 7 * x,
      Jasmine := 5 * x in
  (4 * x + 3 * x + 7 * x + 5 * x = 190) → (Halima - Beckham = 10) :=
by
  intro h
  sorry

end age_difference_10_l210_210226


namespace question_l210_210029

variable (U : Set ℕ) (M : Set ℕ)

theorem question :
  U = {1, 2, 3, 4, 5} →
  (U \ M = {1, 3}) →
  2 ∈ M :=
by
  intros
  sorry

end question_l210_210029


namespace transformed_sum_eq_one_l210_210534

-- Given condition: α, β, γ are the roots of x^3 - x - 1 = 0.
def is_root (f : Real → Real) (x : Real) : Prop := f x = 0

def poly (x : Real) := x^3 - x - 1

-- Define α, β, γ to be the roots of the polynomial
def α : Real := sorry
def β : Real := sorry
def γ : Real := sorry

axiom α_roots : is_root poly α
axiom β_roots : is_root poly β
axiom γ_roots : is_root poly γ

-- Define the function to be proved
def transformed_sum (α β γ : Real) : Real :=
  (1 - α) / (1 + α) + (1 - β) / (1 + β) + (1 - γ) / (1 + γ)

-- The goal is to prove that the transformed sum equals 1
theorem transformed_sum_eq_one : transformed_sum α β γ = 1 := by
  sorry

end transformed_sum_eq_one_l210_210534


namespace unit_vector_perpendicular_projection_of_b_minus_2a_l210_210475

variables (a : ℝ × ℝ) (b : ℝ × ℝ)

-- Define the conditions provided in the problem
def vector_a := (1 : ℝ, 2 : ℝ)
def magnitude_b := (|b| = 1)
def angle_a_b_60 := (∠ (1, 2) b = π / 3)

-- Define the possible unit vectors perpendicular to vector a
def unit_vector_perp_a_1 := (-2 * real.sqrt 5 / 5, real.sqrt 5 / 5)
def unit_vector_perp_a_2 := (2 * real.sqrt 5 / 5, -real.sqrt 5 / 5)

-- Define the projection of the vector (b - 2a) onto vector a
def projection := 1 / 2 - 2 * real.sqrt 5

-- Theorem statements 
theorem unit_vector_perpendicular : 
  ((x y : ℝ), x^2 + y^2 = 1 ∧ x + 2 * y = 0 → (x, y) = unit_vector_perp_a_1 ∨ (x, y) = unit_vector_perp_a_2) :=
sorry

theorem projection_of_b_minus_2a : 
  (b - 2 * (1, 2)).project_on (1, 2) = projection :=
sorry

end unit_vector_perpendicular_projection_of_b_minus_2a_l210_210475


namespace barrels_are_1360_l210_210370

-- Defining the top layer dimensions and properties
def a : ℕ := 2
def b : ℕ := 1
def n : ℕ := 15

-- Defining the dimensions of the bottom layer based on given properties
def c : ℕ := a + n
def d : ℕ := b + n

-- Formula for the total number of barrels
def total_barrels : ℕ := n * ((2 * a + c) * b + (2 * c + a) * d + (d - b)) / 6

-- Theorem to prove
theorem barrels_are_1360 : total_barrels = 1360 :=
by
  sorry

end barrels_are_1360_l210_210370


namespace solution_problem_l210_210156

theorem solution_problem (n : ℕ) (h_pos : n > 0) : ∃ a b : ℤ, n ∣ (4 * a^2 + 9 * b^2 - 1) :=
by
  sorry

end solution_problem_l210_210156


namespace population_increase_duration_l210_210087

noncomputable def birth_rate := 6 / 2 -- people every 2 seconds = 3 people per second
noncomputable def death_rate := 2 / 2 -- people every 2 seconds = 1 person per second
noncomputable def net_increase_per_second := (birth_rate - death_rate) -- net increase per second

def total_net_increase := 172800

theorem population_increase_duration :
  (total_net_increase / net_increase_per_second) / 3600 = 24 :=
by
  sorry

end population_increase_duration_l210_210087


namespace permutation_last_letter_91st_of_BENCH_l210_210966

theorem permutation_last_letter_91st_of_BENCH : 
  let words := ["B", "E", "N", "C", "H"]
  (letters_perms : List (List String)) := List.permutations words
  (sorted_perms : List (List String)) := List.sort (fun a b => List.lex (Ord.compare .compare a b) == Ordering.lt) letters_perms
  List.get sorted_perms 90) == "C" :=
sorry

end permutation_last_letter_91st_of_BENCH_l210_210966


namespace cone_height_l210_210979

theorem cone_height (r_sector : ℝ) (θ_sector : ℝ) :
  r_sector = 3 → θ_sector = (2 * Real.pi / 3) → 
  ∃ (h : ℝ), h = 2 * Real.sqrt 2 := 
by 
  intros r_sector_eq θ_sector_eq
  sorry

end cone_height_l210_210979


namespace number_of_valid_pairings_l210_210239

def knows (n : ℕ) (a b : ℕ) : Prop :=
  b = ((a + 1 - 1) % n) + 1 ∨
  b = ((a - 1 - 1) % n) + 1 ∨
  b = ((a + n / 2 - 1) % n) + 1 ∨
  b = ((a + 2 - 1) % n) + 1

theorem number_of_valid_pairings :
  (Σ P : {P : finset (ℕ × ℕ) | ∀ {x : ℕ}, x ∈ finset.range 12 → ∃ y, (x, y) ∈ P ∧ knows 12 x y}, P = 6).card = 2 :=
sorry

end number_of_valid_pairings_l210_210239


namespace line_A1DC_perpendicular_plane_ABC_line_BC1_parallel_plane_ADC_l210_210947

variables {Point : Type}

structure Line := (p1 p2 : Point)
structure Plane := (p1 p2 p3 : Point)

-- Definitions of points
variables {A A1 B C C1 D E : Point}

-- Midpoint Definitions
def is_midpoint (M A B : Point) : Prop := true -- Placeholder for actual midpoint definition

-- Line and plane definitions
def is_perpendicular (l : Line) (P : Plane) : Prop := true -- Placeholder for actual perpendicular definition
def is_parallel (l : Line) (P : Plane) : Prop := true -- Placeholder for actual parallel definition

variables (A1DC : Line)
variables (ABC ADC : Plane)

-- Conditions
axiom D_midpoint_AB : is_midpoint D A B
axiom ADC_perpendicular_AC : ∀ {A B C D : Point}, is_midpoint D A B → true -- Placeholder for perpendicular condition
axiom ACD_is_isosceles : ∀ {A C D : Point}, true  -- Placeholder
axiom DE_in_ADCC1_not_A1DC : true -- Simplified placeholder
axiom AA11_parallelogram : true -- Simplified placeholder
axiom E_midpoint_AC : is_midpoint E A C
axiom DE_parallel_BC1 : true -- Simplified placeholder

-- Proof Statements
theorem line_A1DC_perpendicular_plane_ABC : is_perpendicular A1DC ABC :=
by sorry

theorem line_BC1_parallel_plane_ADC : is_parallel (Line.mk B C1) ADC :=
by sorry

end line_A1DC_perpendicular_plane_ABC_line_BC1_parallel_plane_ADC_l210_210947


namespace max_positive_numbers_l210_210182

noncomputable def numbers : ℕ := 20

theorem max_positive_numbers 
  (avg_zero : (∑ i in finset.range numbers, (λ x : ℤ, x / numbers)) = 0)
  (n : ℕ) 
  (hpos : n ≤ numbers) : 
  n ≤ 19 :=
sorry

end max_positive_numbers_l210_210182


namespace complex_magnitude_problem_l210_210313

-- Define the imaginary unit with the property i^2 = -1
def i : ℂ := complex.I

-- Prove that the magnitude of the complex number 2 + i² + 2i³ is √5
theorem complex_magnitude_problem : 
  complex.abs (2 + i^2 + 2 * i^3) = real.sqrt 5 := 
by
  -- Use the provided condition i² = -1
  have h : i^2 = -1 := by sorry,
  sorry

end complex_magnitude_problem_l210_210313


namespace roof_difference_l210_210651

noncomputable def roof_diff (W : ℝ) : ℝ :=
  let L := 5 * W
  L - W

theorem roof_difference (W : ℝ) (h1 : 5 * W * W = 784) : roof_diff W ≈ 50.12 :=
by
  let L := 5 * W
  have h2 : L - W ≈ 50.12
  -- Proof to be completed
  sorry

end roof_difference_l210_210651


namespace opposite_of_five_is_neg_five_l210_210214

theorem opposite_of_five_is_neg_five :
  ∃ (x : ℤ), (5 + x = 0) ∧ x = -5 :=
by
  use -5
  split
  · simp
  · rfl

end opposite_of_five_is_neg_five_l210_210214


namespace intersection_A_B_l210_210794

noncomputable def f : ℤ → ℤ := λ x, x ^ 2

def B : Set ℤ := {1, 4}

-- Define a condition that specifies the possible elements of set A
def possible_elements_of_A (x : ℤ) : Prop := f x ∈ B

-- Define the main statement that needs to be proved
theorem intersection_A_B (A : Set ℤ) (hA : ∀ x ∈ A, possible_elements_of_A x) :
  A ∩ B = ∅ ∨ A ∩ B = {1} :=
sorry

end intersection_A_B_l210_210794


namespace bubble_sort_probability_l210_210962

/-- Given n = 50 and the terms r₁, r₂, ..., r₅₀ are distinct from one another and are in random order,
let p/q, in lowest terms, be the probability that the number that begins as r₅ completₑs to thₑ 35th
 place after one bubble pass. Prove that if p/q = 1/1190,
 then p + q = 1191. --/

theorem bubble_sort_probability (n : ℕ) (r : fin n → ℕ) (h1 : n = 50) 
    (h2 : function.injective r) (h3 : ∀ a, a < 50 → a < fin.last n) :
    ∃ (p q : ℕ), (nat.coprime p q ∧ p / q = 1 / 1190) → (p + q = 1191) := sorry

end bubble_sort_probability_l210_210962


namespace winter_sales_l210_210663

theorem winter_sales (spring_sales summer_sales fall_sales : ℕ) (fall_sales_pct : ℝ) (total_sales winter_sales : ℕ) :
  spring_sales = 6 →
  summer_sales = 7 →
  fall_sales = 5 →
  fall_sales_pct = 0.20 →
  fall_sales = ⌊fall_sales_pct * total_sales⌋ →
  total_sales = spring_sales + summer_sales + fall_sales + winter_sales →
  winter_sales = 7 :=
by
  sorry

end winter_sales_l210_210663


namespace find_f_prime_one_l210_210458

noncomputable def f (f'_1 : ℝ) (x : ℝ) := f'_1 * x^3 - 2 * x^2 + 3

theorem find_f_prime_one (f'_1 : ℝ) 
  (h_derivative : ∀ x : ℝ, deriv (f f'_1) x = 3 * f'_1 * x^2 - 4 * x)
  (h_value_at_1 : deriv (f f'_1) 1 = f'_1) :
  f'_1 = 2 :=
by 
  sorry

end find_f_prime_one_l210_210458


namespace like_terms_exponents_l210_210825

theorem like_terms_exponents (m n : ℤ) 
  (h1 : m - 1 = 1) 
  (h2 : m + n = 3) : 
  m = 2 ∧ n = 1 :=
by 
  sorry

end like_terms_exponents_l210_210825


namespace range_of_m_l210_210050

theorem range_of_m (x y m : ℝ) (h1 : x - y = 2 * m + 7) (h2 : x + y = 4 * m - 3) 
  (h3 : x < 0) (h4 : y < 0) : m < -2 / 3 := 
by 
  sorry

end range_of_m_l210_210050


namespace problem_solution_l210_210002

universe u

variable (U : Set Nat) (M : Set Nat)
variable (complement_U_M : Set Nat)

axiom U_def : U = {1, 2, 3, 4, 5}
axiom complement_U_M_def : complement_U_M = {1, 3}
axiom M_def : M = U \ complement_U_M

theorem problem_solution : 2 ∈ M := by
  sorry

end problem_solution_l210_210002


namespace g_value_l210_210492

variable {ℝ : Type*} [Preorder ℝ] [Group ℝ] {f g : ℝ → ℝ}

theorem g_value (h1 : ∀ x y : ℝ, f x y = f x * g y - g x * f y)
  (h2 : f (-2) = f 1) (h3 : f 1 ≠ 0) : g 1 + g (-1) = -1 := by
  sorry

end g_value_l210_210492


namespace maximize_dragon_resilience_l210_210752

noncomputable def p_s (x : ℝ) (s : ℕ) : ℝ :=
  x^s / (1 + x + x^2)

def K : List ℕ :=
  [1, 2, 2, 1, 0, 2, 1, 0, 1, 2]

def P_K (x : ℝ) : ℝ :=
  List.foldr (λ s acc => acc * p_s x s) 1 K

theorem maximize_dragon_resilience :
  let x_opt := (Real.sqrt 97 + 1) / 8 in
  ∀ x > 0, P_K x ≤ P_K x_opt :=
sorry

end maximize_dragon_resilience_l210_210752


namespace triangle_midpoint_equality_l210_210893

-- Define the problem
theorem triangle_midpoint_equality
  (A B C M N P: Type*)
  [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup M] [AddGroup N] [AddGroup P]
  [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ M] [Module ℝ N] [Module ℝ P]
  [AffineSpace A M] [AffineSpace B N] [AffineSpace C P]
  (hM_mid: midpoint A C M)
  (hN_mid: midpoint A B N)
  (hP_median: P ∈ lineSegment B M)
  (hP_not_CN: P ∉ lineSegment C N)
  (hPC_2PN : dist P C = 2 * dist P N) :
  dist A P = dist B C :=
sorry

end triangle_midpoint_equality_l210_210893


namespace complex_magnitude_l210_210327

open Complex

theorem complex_magnitude :
  ∀ (i : ℂ), i^2 = -1 → i^3 = -i → |2 + i^2 + 2 * i^3| = Real.sqrt 5 :=
by
  intros i h1 h2
  -- skipping proof with sorry
  sorry

end complex_magnitude_l210_210327


namespace sin_cos_sixth_power_eq_one_l210_210862

theorem sin_cos_sixth_power_eq_one (α : ℝ) (h : sin α + cos α = 1) : 
  sin α ^ 6 + cos α ^ 6 = 1 := 
sorry

end sin_cos_sixth_power_eq_one_l210_210862


namespace fraction_result_l210_210172

theorem fraction_result (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (2 * x + 3 * y) / (x - 2 * y) = 3) : 
  (x + 2 * y) / (2 * x - y) = 11 / 17 :=
sorry

end fraction_result_l210_210172


namespace first_year_with_sum_of_digits_15_after_2010_l210_210633

theorem first_year_with_sum_of_digits_15_after_2010 : 
  ∃ year : ℕ, year > 2010 ∧ (∑ d in (year.digits 10), d) = 15 ∧ 
  ∀ y : ℕ, y > 2010 → (∑ d in (y.digits 10), d) = 15 → year ≤ y :=
begin
  use 2039,
  split,
  { sorry }, -- proof that 2039 > 2010
  split,
  { sorry }, -- proof that sum of digits of 2039 is 15
  { sorry } -- proof that there's no earlier year with sum of digits 15
end

end first_year_with_sum_of_digits_15_after_2010_l210_210633


namespace opposite_of_five_l210_210218

theorem opposite_of_five : -5 = -5 :=
by
sorry

end opposite_of_five_l210_210218


namespace triangle_proof_l210_210523

theorem triangle_proof (a b c : ℝ) (b_eq_sqrt3 : b = Real.sqrt 3)
          (h : b^2 = a^2 + c^2 + a * c) :
  (∠ B = 2 * π / 3) ∧ 
  (∀ S, (S = (1/2) * a * c * Real.sin (2 * π / 3)) → 
        (S + Real.sqrt 3 * (Real.cos A * Real.cos C) = Real.sqrt 3) →
        (A = π / 6)) :=
by
  sorry

end triangle_proof_l210_210523


namespace sum_is_prime_or_square_prob_l210_210750

-- Define the first ten prime numbers
def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a function to sum a given number with 2 and check if it's either prime or a perfect square
def is_valid_sum (n : ℕ) : Bool :=
  let sum := 2 + n
  sum.isPrime ∨ (let sqrt := Nat.sqrt sum in sqrt * sqrt = sum)

-- Define the number of valid pairs where one number is 2 and the sum is either prime or a perfect square
def num_valid_pairs : ℕ :=
  first_ten_primes.filter is_valid_sum |>.length

-- Define the total ways to draw 2 primes out of 10 without replacement
def total_pairs : ℕ := (first_ten_primes.length * (first_ten_primes.length - 1)) / 2

-- Define the probability
def probability : ℚ := (num_valid_pairs : ℚ) / (total_pairs : ℚ)

-- The theorem to be proven
theorem sum_is_prime_or_square_prob :
  probability = 7 / 45 := by
  sorry

end sum_is_prime_or_square_prob_l210_210750


namespace regions_divided_by_lines_l210_210733

theorem regions_divided_by_lines : 
  let lines := {l | l = (λ x, 2 * x) ∨ l = (λ x, x / 2) ∨ l = (λ x, -x)} 
  (∃ x y, 
  set_of (λ p : ℝ × ℝ, ∃ l ∈ lines, p.2 = l p.1) = set.univ) →
  (set_of (λ p, ∃ m₁ m₂ ∈ lines, m₁ ≠ m₂ ∧ p.2 = m₁ p.1 ∧ p.2 = m₂ p.1) = {(0, 0)}) →
  ∃ regions : ℕ, regions = 6 :=
begin
  sorry
end

end regions_divided_by_lines_l210_210733


namespace correctStatement_l210_210022

variable (U : Set ℕ) (M : Set ℕ)

namespace Proof

-- Given conditions
def universalSet := {1, 2, 3, 4, 5}
def complementM := {1, 3}
def isComplement (M : Set ℕ) : Prop := U \ M = complementM

-- Target statement to be proved
theorem correctStatement (h1 : U = universalSet) (h2 : isComplement M) : 2 ∈ M := by
  sorry

end Proof

end correctStatement_l210_210022


namespace raghu_investment_approx_l210_210262

-- Define the investments
def investments (R : ℝ) : Prop :=
  let Trishul := 0.9 * R
  let Vishal := 0.99 * R
  let Deepak := 1.188 * R
  R + Trishul + Vishal + Deepak = 8578

-- State the theorem to prove that Raghu invested approximately Rs. 2103.96
theorem raghu_investment_approx : 
  ∃ R : ℝ, investments R ∧ abs (R - 2103.96) < 1 :=
by
  sorry

end raghu_investment_approx_l210_210262


namespace sum_of_numbers_on_each_great_circle_is_equal_l210_210196

theorem sum_of_numbers_on_each_great_circle_is_equal 
  (n : ℕ) 
  (h: n ≥ 2) -- Assuming n is at least 2 to have meaningful intersecting great circles.
  (points : Finset ℕ) -- The set of intersection points
  (assign : ℕ → ℕ) -- The function assigning numbers to points
  (h_assign : ∀ x ∈ points, assign x ∈ (Finset.range (n*(n-1)) + 1)) -- All assigned numbers are within correct range.
  (h_complement : ∀ x ∈ points, assign (n*(n-1) + 1 - x) = n*(n-1) + 1 - assign x) -- Complementary assignment condition.
  (h_num_points : points.card = n*(n-1)) -- Total number of points is n*(n-1).
  (circles : Finset (Finset (Finset ℕ))) -- The set of great circles, each as a set of intersection points.
  (h_circles : ∀ c ∈ circles, c.card = n - 1) -- Each great circle intersects in n - 1 points.
  : ∀ c ∈ circles, (∑ p in c, assign p) = (n*(n-1) + 1) * (n - 1) :=
by
  sorry

end sum_of_numbers_on_each_great_circle_is_equal_l210_210196


namespace range_of_a_l210_210420

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^3 - a * x^2 - 4 * a * x + 4 * a^2 - 1 = 0 ∧ ∀ y : ℝ, 
  (y ≠ x → y^3 - a * y^2 - 4 * a * y + 4 * a^2 - 1 ≠ 0)) ↔ a < 3 / 4 := 
sorry

end range_of_a_l210_210420


namespace minimum_height_l210_210526

theorem minimum_height (y : ℝ) (h : ℝ) (S : ℝ) (hS : S = 10 * y^2) (hS_min : S ≥ 150) (h_height : h = 2 * y) : h = 2 * Real.sqrt 15 :=
  sorry

end minimum_height_l210_210526


namespace sum_of_every_third_odd_between_100_300_l210_210639

theorem sum_of_every_third_odd_between_100_300 : 
  let seq := List.filter (λ x, x % 2 = 1) (List.range' 101 199) in -- odd numbers between 101 and 299
  let every_third := List.filteri (λ i, (i % 3 = 0)) seq in           -- every third number in the sequence
  List.sum every_third = 6800 :=
by
  let seq := List.filter (λ x, x % 2 = 1) (List.range' 101 199)
  let every_third := List.filteri (λ i, (i % 3 = 0)) seq
  have h : List.sum every_third = 6800 := sorry
  exact h

end sum_of_every_third_odd_between_100_300_l210_210639


namespace area_of_F1_M_F2_l210_210658

noncomputable def hyperbola (x y : ℝ) : Prop :=
  (x^2 / 9 - y^2 / 4 = 1)

structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Foci (F1 F2 : Point) (hyper : ∀ p : Point, hyperbola p.x p.y → Prop) : Prop :=
  (on_hyperbola : ∀ p : Point, hyperbola p.x p.y → True)
  (angle_60_deg : ∀ M : Point, hyperbola M.x M.y → 
    let d_F1_M := Real.sqrt ((M.x - F1.x)^2 + (M.y - F1.y)^2) in
    let d_F2_M := Real.sqrt ((M.x - F2.x)^2 + (M.y - F2.y)^2) in
    (d_F1_M^2 + d_F2_M^2 - d_F1_M * d_F2_M = 52) → 
    Real.cos (Real.pi / 3) = 1/2)

noncomputable def Area_of_Triangle (F1 F2 M : Point) : ℝ :=
  let d_F1_M := Real.sqrt ((M.x - F1.x)^2 + (M.y - F1.y)^2) in
  let d_F2_M := Real.sqrt ((M.x - F2.x)^2 + (M.y - F2.y)^2) in
  let d_F1_F2 := Real.sqrt ((F2.x - F1.x)^2 + (F2.y - F1.y)^2) in
  1/2 * d_F1_M * d_F2_M * Real.sin (Real.pi / 3)

theorem area_of_F1_M_F2 {F1 F2 M : Point} (hFoci : Foci F1 F2 hyperbola)
  (hM_on_hyper : hyperbola M.x M.y) (h60 : ∀ M : Point,
    hyperbola M.x M.y → Real.cos (Real.pi / 3) = 1/2) :
  Area_of_Triangle F1 F2 M = 4 * Real.sqrt 3 :=
  sorry

end area_of_F1_M_F2_l210_210658


namespace total_scissors_l210_210243

def initial_scissors : ℕ := 54
def added_scissors : ℕ := 22

theorem total_scissors : initial_scissors + added_scissors = 76 :=
by
  sorry

end total_scissors_l210_210243


namespace mango_rate_is_50_l210_210480

theorem mango_rate_is_50 (quantity_grapes kg_grapes_perkg quantity_mangoes total_paid cost_grapes cost_mangoes rate_mangoes : ℕ) 
  (h1 : quantity_grapes = 8) 
  (h2 : kg_grapes_perkg = 70) 
  (h3 : quantity_mangoes = 9) 
  (h4 : total_paid = 1010)
  (h5 : cost_grapes = quantity_grapes * kg_grapes_perkg)
  (h6 : cost_mangoes = total_paid - cost_grapes)
  (h7 : rate_mangoes = cost_mangoes / quantity_mangoes) : 
  rate_mangoes = 50 :=
by sorry

end mango_rate_is_50_l210_210480


namespace sequence_an_not_divisible_by_5_l210_210155

open Nat

def sequence_an (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), 2^(3 * k) * Nat.choose (2 * n + 1) (2 * k + 1)

theorem sequence_an_not_divisible_by_5 (n : ℕ) : ¬ (5 ∣ sequence_an n) :=
  sorry

end sequence_an_not_divisible_by_5_l210_210155


namespace remainder_130_div_k_l210_210785

theorem remainder_130_div_k (k : ℕ) (h_positive : k > 0)
  (h_remainder : 84 % (k*k) = 20) : 
  130 % k = 2 := 
by sorry

end remainder_130_div_k_l210_210785


namespace regular_polygon_sides_l210_210687

theorem regular_polygon_sides (θ : ℝ) (h_angle : θ = 30) (h_sum : ∑ i in {1} : ℝ, i = 360) : 
  ∃ n : ℕ, n = 12 :=
by
  sorry

end regular_polygon_sides_l210_210687


namespace area_diminished_by_64_percent_l210_210645

/-- Given a rectangular field where both the length and width are diminished by 40%, 
    prove that the area is diminished by 64%. -/
theorem area_diminished_by_64_percent (L W : ℝ) :
  let L' := 0.6 * L
  let W' := 0.6 * W
  let A := L * W
  let A' := L' * W'
  (A - A') / A * 100 = 64 :=
by
  sorry

end area_diminished_by_64_percent_l210_210645


namespace sum_of_reciprocals_of_squares_l210_210984

open Nat

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h_prod : a * b = 5) : 
  (1 : ℚ) / (a * a) + (1 : ℚ) / (b * b) = 26 / 25 :=
by
  -- proof steps skipping with sorry
  sorry

end sum_of_reciprocals_of_squares_l210_210984


namespace ways_to_select_at_least_one_defective_l210_210701

open Finset

-- Define basic combinatorial selection functions
def combination (n k : ℕ) := Nat.choose n k

-- Given conditions
def total_products : ℕ := 100
def defective_products : ℕ := 6
def selected_products : ℕ := 3
def non_defective_products : ℕ := total_products - defective_products

-- The question to prove: the number of ways to select at least one defective product
theorem ways_to_select_at_least_one_defective :
  (combination total_products selected_products) - (combination non_defective_products selected_products) =
  (combination 100 3) - (combination 94 3) := by
  sorry

end ways_to_select_at_least_one_defective_l210_210701


namespace expand_expression_l210_210407

theorem expand_expression (x : ℝ) : (x - 3) * (x + 6) = x^2 + 3 * x - 18 :=
by
  sorry

end expand_expression_l210_210407


namespace function_zeros_count_l210_210917

theorem function_zeros_count :
  ∀ (f : ℝ → ℝ) (ω : ℝ),
    (∀ x, f(x) = Real.sin(ω * x - Real.pi / 4)) → 
    ω > 0 →
    (∀ x1 x2, |f x1 - f x2| = 2 → |x1 - x2| = Real.pi / 3) →
    (∃ n, set.finite.to_finset ({x : ℝ | f x = 0} ∩ set.Icc (-Real.pi) Real.pi) = finset.range n.succ ∧
      finset.card (set.finite.to_finset ({x : ℝ | f x = 0} ∩ set.Icc (-Real.pi) Real.pi)) = 6) :=
begin
  intros f ω hf hω hcond,
  sorry
end

end function_zeros_count_l210_210917


namespace all_lines_intersect_at_single_point_l210_210085

theorem all_lines_intersect_at_single_point
    (lines : FinSet (FinSet Point))  -- The set of red and blue lines
    (finite_lines : Finite lines)  -- Given that the number of lines is finite
    (no_parallel_lines : ∀ l1 l2 ∈ lines, l1 ≠ l2 → ¬ Parallel l1 l2)  -- No two lines are parallel
    (cross_intersection_condition : ∀ p ∈ lines, ∃ q ∈ lines, ∀ r ∈ Intersect p q, r ∉ p) : -- Through each intersection point of lines of the same color passes a line of the other color
    ∃ P : Point, ∀ l ∈ lines, P ∈ l :=  -- Prove that all lines pass through a single point
sorry

end all_lines_intersect_at_single_point_l210_210085


namespace intercepted_segment_length_l210_210841

def parametric_eq_line (t : ℝ) : ℝ × ℝ :=
  ( - (Real.sqrt 2) / 2 * t, 1 + (Real.sqrt 2) / 2 * t)

def curve_eq (x y : ℝ) : Prop :=
  y ^ 2 = 4 * x

def passes_through (P : ℝ × ℝ) : Prop :=
  P = (0, 1)

theorem intercepted_segment_length :
∀ t : ℝ, passes_through (0, 1) →
  let (x, y) := parametric_eq_line t in
  curve_eq x y →
  (abs (Real.sqrt ((- (3:ℝ) * Real.sqrt 2) ^ 2 - 2 * 4))) = 8 :=
by
  intros t ht heq
  sorry

end intercepted_segment_length_l210_210841


namespace prob_exactly_one_canteen_needs_rectification_is_0_25_prob_at_least_one_canteen_closed_is_0_34_l210_210194

noncomputable def prob_first_inspection_pass : ℝ := 0.5
noncomputable def prob_second_inspection_pass : ℝ := 0.8

noncomputable def prob_canteen_needs_rectification (n m : ℕ) : ℝ :=
  (nat.choose n m) * (prob_first_inspection_pass ^ (n - m)) * ((1 - prob_first_inspection_pass) ^ m)

noncomputable def prob_exactly_one_needs_rectification : ℝ :=
  prob_canteen_needs_rectification 4 1

theorem prob_exactly_one_canteen_needs_rectification_is_0_25 :
  prob_exactly_one_needs_rectification = 0.25 := sorry

noncomputable def prob_canteen_closed : ℝ :=
  (1 - prob_first_inspection_pass) * (1 - prob_second_inspection_pass)

noncomputable def prob_at_least_one_closed : ℝ :=
  1 - (1 - prob_canteen_closed) ^ 4

theorem prob_at_least_one_canteen_closed_is_0_34 :
  prob_at_least_one_closed = 0.34 := sorry

end prob_exactly_one_canteen_needs_rectification_is_0_25_prob_at_least_one_canteen_closed_is_0_34_l210_210194


namespace smallest_integer_m_l210_210142

noncomputable def alpha : ℝ := real.pi / 9
noncomputable def beta : ℝ := real.pi / 12
noncomputable def theta : ℝ := real.atan (1 / 4)

noncomputable def R (theta : ℝ) : ℝ :=
2 * beta - (2 * alpha - theta)

noncomputable def Rn (theta : ℝ) (n : ℕ) : ℝ :=
theta + n * (- real.pi / 36)

theorem smallest_integer_m : ∃ (m : ℕ), m > 0 ∧ Rn theta m = theta ∧ m = 72 :=
sorry

end smallest_integer_m_l210_210142


namespace angle_sum_equality_l210_210372

noncomputable def triangle_circumscribed (A B C O : Type*) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited O] : Prop :=
  ∃ (AB AC : ℝ) (M K D E P Q : Type*),
    AB > AC ∧
    is_midpoint_minor_arc O M B C ∧
    is_antipodal O A K ∧
    line_through_parallel O AM intersects AB at D ∧
    line_through_parallel O AM intersects_extension CA at E ∧
    line_intersect BM CK P ∧
    line_intersect CM BK Q ∧
    acute_triangle A B C ∧
    inscribed_circle_triangle O A B C 

theorem angle_sum_equality :
  ∀ (A B C O : Type*) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited O]
  (AB AC : ℝ) (M K D E P Q : Type*),
  triangle_circumscribed A B C O →
  ∠OPB + ∠OEB = ∠OQC + ∠ODC :=
by
  intros
  sorry

end angle_sum_equality_l210_210372


namespace line_through_point_area_T_l210_210344

variable (a T : ℝ)

def triangle_line_equation (a T : ℝ) : Prop :=
  ∃ y x : ℝ, (a^2 * y + 2 * T * x - 2 * a * T = 0) ∧ (y = -((2 * T)/a^2) * x + (2 * T) / a) ∧ (x ≥ 0) ∧ (y ≥ 0)

theorem line_through_point_area_T (a T : ℝ) (h₁ : a > 0) (h₂ : T > 0) :
  triangle_line_equation a T :=
sorry

end line_through_point_area_T_l210_210344


namespace stocking_stuffers_total_l210_210056

theorem stocking_stuffers_total 
  (candy_canes_per_child beanie_babies_per_child books_per_child : ℕ)
  (num_children : ℕ)
  (h1 : candy_canes_per_child = 4)
  (h2 : beanie_babies_per_child = 2)
  (h3 : books_per_child = 1)
  (h4 : num_children = 3) :
  candy_canes_per_child + beanie_babies_per_child + books_per_child * num_children = 21 :=
by
  sorry

end stocking_stuffers_total_l210_210056


namespace season_duration_l210_210248

-- Define the given conditions.
def games_per_month : ℕ := 7
def games_per_season : ℕ := 14

-- Define the property we want to prove.
theorem season_duration : games_per_season / games_per_month = 2 :=
by
  sorry

end season_duration_l210_210248


namespace find_a_l210_210836

def f (a : ℝ) (x : ℝ) := a * x^2 + 3 * x - 2

theorem find_a (a : ℝ) (h : deriv (f a) 2 = 7) : a = 1 :=
by {
  sorry
}

end find_a_l210_210836


namespace unique_handshakes_count_l210_210881

-- Definitions from the conditions
def teams : Nat := 4
def players_per_team : Nat := 2
def total_players : Nat := teams * players_per_team

def handshakes_per_player : Nat := total_players - players_per_team

-- The Lean statement to prove the total number of unique handshakes
theorem unique_handshakes_count : (total_players * handshakes_per_player) / 2 = 24 := 
by
  -- Proof steps would go here
  sorry

end unique_handshakes_count_l210_210881


namespace monotonically_increasing_has_monotonically_increasing_interval_l210_210462

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - Real.log x

theorem monotonically_increasing (a : ℝ) : 
  (∀ x ∈ set.Icc 1 2, 2 * a * x - (1 / x) ≥ 0) ↔ a ≥ 1 / 2 :=
sorry

theorem has_monotonically_increasing_interval (a : ℝ) : 
  (∃ x ∈ set.Icc 1 2, 2 * a * x - (1 / x) > 0) ↔ a > 1 / 8 :=
sorry

end monotonically_increasing_has_monotonically_increasing_interval_l210_210462


namespace sequence_formula_l210_210098

def sequence (a : ℕ → ℝ) : Prop :=
  (a 1 = 2) ∧ (∀ n, a (n + 1) = 3 * a n + 5)

theorem sequence_formula (a : ℕ → ℝ) (n : ℕ)
  (h1 : a 1 = 2)
  (h2 : ∀ n, a (n + 1) = 3 * a n + 5) :
  a n = (1 / 2) * 3^(n + 1) - (5 / 2) :=
by
  sorry

end sequence_formula_l210_210098


namespace sum_f_mod_1000_l210_210859

def divisor_count_cond (n : ℕ) (d : ℕ) : Prop :=
  d < n ∨ (n.gcd d > 1)

def f (n : ℕ) : ℕ :=
  (Finset.filter (divisor_count_cond n) (Finset.range (2024 ^ 2024 + 1))).card

theorem sum_f_mod_1000 :
  (∑ n in Finset.range (2024 ^ 2024 + 1), f n) % 1000 = 224 := 
sorry

end sum_f_mod_1000_l210_210859


namespace grapes_purchased_l210_210849

variable (G : ℕ)
variable (rate_grapes : ℕ) (qty_mangoes : ℕ) (rate_mangoes : ℕ) (total_paid : ℕ)

theorem grapes_purchased (h1 : rate_grapes = 70)
                        (h2 : qty_mangoes = 9)
                        (h3 : rate_mangoes = 55)
                        (h4 : total_paid = 1055) :
                        70 * G + 9 * 55 = 1055 → G = 8 :=
by
  sorry

end grapes_purchased_l210_210849


namespace solution_set_of_inequality_l210_210580

-- Definitions for the function properties
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def monotonically_decreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f y ≤ f x

-- The main theorem statement
theorem solution_set_of_inequality 
  (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_monotone_dec : monotonically_decreasing (λ x, f(-x))) :
    {x : ℝ | f(-1) < f(log 10 x)} = {x : ℝ | (0 < x ∧ x < 1/10) ∨ (10 < x)} :=
by
  sorry

end solution_set_of_inequality_l210_210580


namespace F_recurrence_relation_l210_210391

def f (n : ℕ) : ℕ := floor ((3 - real.sqrt 5) / 2 * n)

def F : ℕ → ℕ
| 0 => sorry
| 1 => sorry
| k + 2 => (F (k + 1)) * 3 - F k

theorem F_recurrence_relation (k : ℕ) : F (k + 2) = 3 * F (k + 1) - F k := 
sorry

end F_recurrence_relation_l210_210391


namespace sufficient_but_not_necessary_l210_210655

theorem sufficient_but_not_necessary {a b : ℝ} (h1 : a > 2) (h2 : b > 2) : 
  a + b > 4 ∧ a * b > 4 := 
by
  sorry

end sufficient_but_not_necessary_l210_210655


namespace ferns_fronds_l210_210125

theorem ferns_fronds (
  num_fers : ℕ
  (leaves_per_frond : ℕ) 
  (total_leaves : ℕ) :
  num_ferns = 6 →
  leaves_per_frond = 30 →
  total_leaves = 1260 →
  ∃ fronds_per_fern : ℕ, fronds_per_fern = 7) :=
begin
  assume h1 h2 h3,
  have h_fronds : 1260 / 30 = 42, by sorry,
  have h_each_fern : 42 / 6 = 7, by sorry,
  use 7,
  exact h_each_fern,
end

end ferns_fronds_l210_210125


namespace books_per_shelf_l210_210478

theorem books_per_shelf (mystery_shelves picture_shelves total_books : ℕ)
    (h₁ : mystery_shelves = 5)
    (h₂ : picture_shelves = 3)
    (h₃ : total_books = 32) :
    (total_books / (mystery_shelves + picture_shelves) = 4) :=
by
    sorry

end books_per_shelf_l210_210478


namespace registration_methods_count_l210_210823

theorem registration_methods_count (students : Fin 4) (groups : Fin 3) : (3 : ℕ)^4 = 81 :=
by
  sorry

end registration_methods_count_l210_210823


namespace domain_function_l210_210740

noncomputable def function (x : ℝ) : ℝ := (x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - 1) / (x^2 - 9)

theorem domain_function : ∀ x : ℝ, x ≠ 3 ∧ x ≠ -3 ↔ x ∈ (-∞:ℝ, -3) ∪ (-3, 3) ∪ (3, ∞) :=
by
  intro x
  have h₁ : x^2 - 9 ≠ 0 ↔ x ≠ 3 ∧ x ≠ -3 := by
    split
    · intro h
      constructor
      · intro hx
        rw hx at h
        simp at h
      · intro hx
        rw hx at h
        simp at h
    · intro h
      cases h with h₁ h₂
      intro c
      cases eq_or_eq_neg_of_sq_eq_sq _ _ (eq.symm c)
      · exact h₁ h
      · apply h₂
        rw neg_eq_iff_neg_eq at h
        assumption
    
  show (x ≠ 3 ∧ x ≠ -3) ↔ (x ∈ (-∞:ℝ, -3) ∪ (-3, 3) ∪ (3, ∞)) from
    h₁.trans Iff.rfl 

end domain_function_l210_210740


namespace false_statement_l210_210806

variables (α β : Plane) (m n : Line)

axiom perp_to_plane (l : Line) (p : Plane) : Prop
axiom parallel_to_plane (p₁ p₂ : Plane) : Prop
axiom parallel_lines (l₁ l₂ : Line) : Prop
axiom line_in_plane (l : Line) (p : Plane) : Prop

theorem false_statement :
  (∀ m α β, perp_to_plane m α ∧ perp_to_plane m β → parallel_to_plane α β) ∧
  (∀ m n α, parallel_lines m n ∧ perp_to_plane m α → perp_to_plane n α) ∧
  (∀ m α β, perp_to_plane m α ∧ line_in_plane m β → parallel_to_plane α β) ∧
  ¬ (∀ m α β n, ¬ (parallel_lines m α) ∧ parallel_to_plane α β ∧ line_in_plane n β → parallel_lines m n) := 
sorry

end false_statement_l210_210806


namespace snack_cost_inequality_l210_210347

variables (S : ℝ)

def cost_water : ℝ := 0.50
def cost_fruit : ℝ := 0.25
def bundle_price : ℝ := 4.60
def special_price : ℝ := 2.00

theorem snack_cost_inequality (h : bundle_price = 4.60 ∧ special_price = 2.00 ∧
  cost_water = 0.50 ∧ cost_fruit = 0.25) : S < 15.40 / 16 := sorry

end snack_cost_inequality_l210_210347


namespace train_departure_at_10am_l210_210285

noncomputable def train_departure_time (distance travel_rate : ℕ) (arrival_time_chicago : ℕ) (time_difference : ℤ) : ℕ :=
  let travel_time := distance / travel_rate
  let arrival_time_ny := arrival_time_chicago + 1
  arrival_time_ny - travel_time

theorem train_departure_at_10am :
  train_departure_time 480 60 17 1 = 10 :=
by
  -- implementation of the proof will go here
  -- but we skip the proof as per the instructions
  sorry

end train_departure_at_10am_l210_210285


namespace question_l210_210030

variable (U : Set ℕ) (M : Set ℕ)

theorem question :
  U = {1, 2, 3, 4, 5} →
  (U \ M = {1, 3}) →
  2 ∈ M :=
by
  intros
  sorry

end question_l210_210030


namespace find_y_l210_210409

theorem find_y (y : ℝ) (h : 2 * arctan (1/5) + arctan (1/25) + arctan (1/y) = π / 4) : y = 1210 :=
sorry

end find_y_l210_210409


namespace problem_solution_l210_210006

universe u

variable (U : Set Nat) (M : Set Nat)
variable (complement_U_M : Set Nat)

axiom U_def : U = {1, 2, 3, 4, 5}
axiom complement_U_M_def : complement_U_M = {1, 3}
axiom M_def : M = U \ complement_U_M

theorem problem_solution : 2 ∈ M := by
  sorry

end problem_solution_l210_210006


namespace solve_for_x_l210_210076

variables {x y : ℝ}

theorem solve_for_x (h : x / (x - 3) = (y^2 + 3 * y + 1) / (y^2 + 3 * y - 4)) : 
  x = (3 * y^2 + 9 * y + 3) / 5 :=
sorry

end solve_for_x_l210_210076


namespace num_factors_of_M_l210_210855

def M : ℕ := 2^4 * 3^3 * 7^2

theorem num_factors_of_M : ∃ n, n = 60 ∧ (∀ d e f : ℕ, 0 ≤ d ∧ d ≤ 4 ∧ 0 ≤ e ∧ e ≤ 3 ∧ 0 ≤ f ∧ f ≤ 2 → (2^d * 3^e * 7^f ∣ M) ∧ ∃ k, k = 5 * 4 * 3 ∧ k = n) :=
by
  sorry

end num_factors_of_M_l210_210855


namespace find_c_and_d_l210_210128

noncomputable def T : set (ℝ × ℝ × ℝ) :=
  { p | ∃ x y z, p = (x, y, z) ∧ log 10 (x + y) = z + 1 ∧ log 10 (x ^ 2 + y ^ 2) = z + 2 }

theorem find_c_and_d :
  ∃ c d : ℝ, (∀ x y z : ℝ, (x, y, z) ∈ T → x^3 + y^3 = c * 10^(4*z) + d * 10^(3*z)) ∧ c + d = 29 / 2 :=
sorry

end find_c_and_d_l210_210128


namespace committee_with_one_boy_one_girl_prob_l210_210223

def total_members := 30
def boys := 12
def girls := 18
def committee_size := 6

theorem committee_with_one_boy_one_girl_prob :
  let total_ways := Nat.choose total_members committee_size
  let all_boys_ways := Nat.choose boys committee_size
  let all_girls_ways := Nat.choose girls committee_size
  let prob_all_boys_or_all_girls := (all_boys_ways + all_girls_ways) / total_ways
  let desired_prob := 1 - prob_all_boys_or_all_girls
  desired_prob = 19145 / 19793 :=
by
  sorry

end committee_with_one_boy_one_girl_prob_l210_210223


namespace like_terms_expressions_l210_210826

theorem like_terms_expressions (m n : ℤ) :
  (∀ x y : ℝ, -3 * x ^ (m - 1) * y ^ 3 = 4 * x * y ^ (m + n)) → (m = 2 ∧ n = 1) :=
by
  intro h
  have h_mx_pow : m - 1 = 1 := sorry
  have h_my_pow : 3 = m + n := sorry
  finish

end like_terms_expressions_l210_210826


namespace remainder_of_7_pow_12_mod_100_l210_210638

theorem remainder_of_7_pow_12_mod_100 : (7 ^ 12) % 100 = 1 := 
by sorry

end remainder_of_7_pow_12_mod_100_l210_210638


namespace projection_inequality_l210_210432

-- Define the problem with given Cartesian coordinate system, finite set of points in space, and their orthogonal projections
variable (O_xyz : Type) -- Cartesian coordinate system
variable (S : Finset O_xyz) -- finite set of points in space
variable (S_x S_y S_z : Finset O_xyz) -- sets of orthogonal projections onto the planes

-- Define the orthogonal projections (left as a comment here since detailed implementation is not specified)
-- (In Lean, actual definitions of orthogonal projections would follow mathematical and geometric definitions)

-- State the theorem to be proved
theorem projection_inequality :
  (Finset.card S) ^ 2 ≤ (Finset.card S_x) * (Finset.card S_y) * (Finset.card S_z) := 
sorry

end projection_inequality_l210_210432


namespace exists_100_similar_non_identical_rectangles_l210_210402

theorem exists_100_similar_non_identical_rectangles :
  ∃ (R : Type) [rectangular R], (∃ (S : fin 100 → rectangular R), (∀ i j, i ≠ j → S i ≠ S j) ∧ (∀ i j, i ≠ j → S i.similar_to R)) :=
sorry

end exists_100_similar_non_identical_rectangles_l210_210402


namespace change_in_function_l210_210140

variable {α β : Type*} [Add α] [HasEquiv α] (f : α → β) (x₀ Δx : α)

theorem change_in_function :
  (f (x₀ + Δx) - f x₀) = Δy :=
sorry

end change_in_function_l210_210140


namespace revenue_difference_l210_210584

theorem revenue_difference (price_jersey price_tshirt : ℕ) (num_jerseys num_tshirts : ℕ) 
  (price_jersey_value : price_jersey = 210) (price_tshirt_value : price_tshirt = 240)
  (num_jerseys_value : num_jerseys = 23) (num_tshirts_value : num_tshirts = 177) :
  num_tshirts * price_tshirt - num_jerseys * price_jersey = 37650 :=
by
  rw [price_jersey_value, price_tshirt_value, num_jerseys_value, num_tshirts_value]
  calc
    177 * 240 - 23 * 210 = 42480 - 4830 := by norm_num
    ... = 37650 := by norm_num

end revenue_difference_l210_210584


namespace trailing_zeros_in_sequence_l210_210356

theorem trailing_zeros_in_sequence :
  (∃ (f : ℕ → ℕ) (ftotal : ℕ),
     (∀ n, f n = 3 * n - 2)
     ∧ f 234 = 700
     ∧ (∏ k in finset.range (234), f (k + 1)).toDigits 10 |>.count 0 = 60) := 
by
  have h₁ : ∀ n, f n = 3 * n - 2 := sorry
  have h₂ : f 234 = 700 := sorry
  have prod := (∏ k in finset.range (234), f (k + 1))
  have h₃ : prod.toDigits 10 |>.count 0 = 60 := sorry
  exact ⟨f, 234, ⟨h₁, h₂, h₃⟩⟩

end trailing_zeros_in_sequence_l210_210356


namespace f_4_equals_2_l210_210493

variable (f : ℝ → ℝ)

def periodic_2 := ∀ x, f(x + 2) = f(x)
def f_at_2 := f(2) = 2

theorem f_4_equals_2 (h1 : periodic_2 f) (h2 : f_at_2 f) : f(4) = 2 := by
  sorry

end f_4_equals_2_l210_210493


namespace sandy_earnings_correct_l210_210571

def hourly_rate : ℕ := 15
def hours_worked_friday : ℕ := 10
def hours_worked_saturday : ℕ := 6
def hours_worked_sunday : ℕ := 14

def earnings_friday : ℕ := hours_worked_friday * hourly_rate
def earnings_saturday : ℕ := hours_worked_saturday * hourly_rate
def earnings_sunday : ℕ := hours_worked_sunday * hourly_rate

def total_earnings : ℕ := earnings_friday + earnings_saturday + earnings_sunday

theorem sandy_earnings_correct : total_earnings = 450 := by
  sorry

end sandy_earnings_correct_l210_210571


namespace count_permutations_of_multiples_of_13_l210_210060

/-- The set of multiples of 13 between 100 and 999 -/
def multiples_of_13 : Finset ℕ :=
  Finset.filter (λ n, n % 13 = 0) (Finset.Icc 100 999)

/-- The set of permissible permutations for a natural number -/
def permissible_permutations (n : ℕ) : Finset ℕ :=
  Finset.fromList $ n.digits.permutations.filter (λ l, let m := Nat.ofDigits l in 100 ≤ m ∧ m ≤ 999)

/-- Prove the total count of integers between 100 and 999 whose digits' permutations 
  include multiples of 13 within the same range is 195 -/
theorem count_permutations_of_multiples_of_13 :
  ∑ m in multiples_of_13, (permissible_permutations m).card = 195 :=
by sorry

end count_permutations_of_multiples_of_13_l210_210060


namespace log_difference_is_six_l210_210975

-- Define the property of f(x) = log_a x with a > 0 and a ≠ 1
noncomputable def f (x : ℝ) (a : ℝ) [log_base : log a] : ℝ := log_base x

-- Define the main theorem statement to be proven
theorem log_difference_is_six 
  (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) 
  (x₁ x₂ : ℝ) 
  (h₂ : f x₁ a - f x₂ a = 2) : 
  f (x₁^3) a - f (x₂^3) a = 6 := 
by 
  sorry -- Proof omitted

end log_difference_is_six_l210_210975


namespace stratified_sampling_group_B_l210_210688

theorem stratified_sampling_group_B
  (total_cities : ℕ)
  (group_A_cities : ℕ)
  (group_B_cities : ℕ)
  (group_C_cities : ℕ)
  (total_sampled : ℕ)
  (h_total : total_cities = 24)
  (h_A : group_A_cities = 4)
  (h_B : group_B_cities = 12)
  (h_C : group_C_cities = 8)
  (h_sampled : total_sampled = 6) :
  group_B_cities * total_sampled / total_cities = 3 := 
by
  rw [h_total, h_A, h_B, h_C, h_sampled] 
  -- Provide a simpler proof if necessary, or use algebraic manipulations
  -- onioning the correctness of the statement
  norm_num
  sorry

end stratified_sampling_group_B_l210_210688


namespace count_n_between_401_and_1599_l210_210857

theorem count_n_between_401_and_1599:
  ∃ n, 400 < n.succ ^ 2 ∧ n.succ ^ 2 < 1600 ∧ ( ∀ n, 400 < n.succ ^ 2 ∧ n.succ ^ 2 < 1600 → n = 18) :=
sorry

end count_n_between_401_and_1599_l210_210857


namespace opposite_of_five_l210_210207

theorem opposite_of_five : ∃ y : ℤ, 5 + y = 0 ∧ y = -5 := by
  use -5
  constructor
  . exact rfl
  . sorry

end opposite_of_five_l210_210207


namespace shaded_area_proof_l210_210362

def area_square (side_length : ℝ) : ℝ := side_length^2
def area_circle (radius : ℝ) : ℝ := Real.pi * radius^2
def total_area_circles (num_circles : ℕ) (radius : ℝ) : ℝ := num_circles * area_circle(radius)
def shaded_area (side_length : ℝ) (num_circles : ℕ) (diameter : ℝ) : ℝ := 
  area_square(side_length) - total_area_circles(num_circles, diameter / 2)

theorem shaded_area_proof :
  shaded_area 24 6 8 = 576 - 96 * Real.pi :=
by sorry

end shaded_area_proof_l210_210362


namespace complex_magnitude_l210_210308

variables (i : ℂ)
axiom imaginary_unit : i^2 = -1

theorem complex_magnitude : |2 + i^2 + 2 * i^3| = Real.sqrt 5 :=
by
  have h1 : i^3 = i^2 * i := by sorry
  have h2 : i^2 = -1 := imaginary_unit
  have h3 : i^3 = -i := by sorry
  calc 
    |2 + i^2 + 2 * i^3| = |2 + (-1) + 2 * (-i)| : by sorry
    ... = |1 - 2 * i| : by sorry
    ... = Real.sqrt (1^2 + (-2)^2) : by sorry
    ... = Real.sqrt 5 : by sorry

end complex_magnitude_l210_210308


namespace red_bus_length_l210_210598

variable (C Y R : ℝ)

def condition1 := R = 4 * C
def condition2 := 7 * C = 2 * Y
def condition3 := R = Y + 6

theorem red_bus_length 
  (h1 : condition1 C Y R)
  (h2 : condition2 C Y R)
  (h3 : condition3 C Y R) : 
  R = 48 := sorry

end red_bus_length_l210_210598


namespace range_of_k_l210_210465

noncomputable def is_monotonically_decreasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y ∈ I, x < y → f x ≥ f y

theorem range_of_k (k : ℝ) :
  is_monotonically_decreasing (λ x, k * Real.cos (k * x)) (set.Ioo (Real.pi / 4) (Real.pi / 3)) →
  k ∈ set.Icc (-6) (-4) ∪ set.Ioc 0 3 ∪ set.Icc 8 9 ∪ {-12} :=
sorry

end range_of_k_l210_210465


namespace question_l210_210028

variable (U : Set ℕ) (M : Set ℕ)

theorem question :
  U = {1, 2, 3, 4, 5} →
  (U \ M = {1, 3}) →
  2 ∈ M :=
by
  intros
  sorry

end question_l210_210028


namespace sum_of_roots_in_interval_l210_210165

variable (x : ℝ) (k : ℤ)

noncomputable def trigonometric_equation : Prop :=
  3 * cos ((4 * real.pi * x) / 5) + cos ((12 * real.pi * x) / 5) =
  2 * cos ((4 * real.pi * x) / 5) * (3 + (tan (real.pi * x / 5))^2 - 2 * tan (real.pi * x / 5))

theorem sum_of_roots_in_interval :
  (∃ k : ℤ → (x : ℝ → (-11 ≤ x ∧ x ≤ 19))) → 
  (trigonometric_equation x k) → 
  (∑ k in finset.range (14 - (-9) + 1), 5/8 + 5*k/4) = 112.5 :=
sorry

end sum_of_roots_in_interval_l210_210165


namespace regression_R2_SSR_l210_210092

-- Definitions based on regression analysis context
variable (R2 : ℝ) (SSR : ℝ) -- Coefficient of determination and sum of squares of residuals

-- Assume that as R2 decreases, SSR increases
def regression_property : Prop :=
  (R2 < x → SSR > y) -- for any given x and y, if R2 is smaller, SSR is larger

-- Proof statement (to be proven)
theorem regression_R2_SSR (R2 : ℝ) (SSR : ℝ) : R2 < x → SSR > y := 
by { sorry }

end regression_R2_SSR_l210_210092


namespace number_of_functions_satisfying_conditions_l210_210923

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def f_conditions (f : ℕ → ℕ) : Prop :=
  (∀ s ∈ S, f (f (f s)) = s) ∧ (∀ s ∈ S, (f s - s) % 3 ≠ 0)

theorem number_of_functions_satisfying_conditions :
  (∃ (f : ℕ → ℕ), f_conditions f) ∧ (∃! (n : ℕ), n = 288) :=
by
  sorry

end number_of_functions_satisfying_conditions_l210_210923


namespace S_100_has_100_prime_factors_l210_210895

noncomputable def S : ℕ → ℕ
| 0       := sorry -- initial sum of squares
| (n + 1) := S n * (S n + 1)

theorem S_100_has_100_prime_factors :
  nat.num_distinct_prime_divisors (S 100) ≥ 100 :=
sorry

end S_100_has_100_prime_factors_l210_210895


namespace angles_of_triangles_l210_210908

variables (A B C H H_A H_B H_C : Type) [inhabited A] [inhabited B] [inhabited C] [inhabited H] [inhabited H_A] [inhabited H_B] [inhabited H_C]
variables (angleA angleB angleC : ℝ)

-- Assumption: H is the orthocenter of triangle ABC
-- Assumption: H_A, H_B, H_C are the feet of the altitudes from A, B, and C respectively
-- Assumption: H lies inside triangle ABC, meaning all angles in the triangle are acute

theorem angles_of_triangles (h_orthocenter : is_orthocenter H A B C)
  (h_feet_of_altitudes : feet_of_altitudes H_A H_B H_C A B C H)
  (h_triangle_acute : acute_triangle A B C) :
  
  -- Prove the angles of the triangles in terms of angles of triangle ABC
  (angles_in_triangle A H_B H_C = (angleC, angleB, 180 - (angleC + angleB))) ∧
  (angles_in_triangle H_A B H_C = (angleC, angleA, 180 - (angleC + angleA))) ∧
  (angles_in_triangle H_A H_B C = (angleB, angleA, 180 - (angleB + angleA))) ∧
  (angles_in_triangle H_A H_B H_C = (180 - 2 * angleB, 180 - 2 * angleC, 180 - 2 * angleA)) := sorry

end angles_of_triangles_l210_210908


namespace largest_n_satisfies_l210_210778

noncomputable def sin_plus_cos_bound (n : ℕ) (x : ℝ) : Prop :=
  (Real.sin x)^n + (Real.cos x)^n ≥ 1 / (2 * Real.sqrt n)

theorem largest_n_satisfies :
  ∃ (n : ℕ), (∀ x : ℝ, sin_plus_cos_bound n x) ∧
  ∀ m : ℕ, (∀ x : ℝ, sin_plus_cos_bound m x) → m ≤ 2 := 
sorry

end largest_n_satisfies_l210_210778


namespace period_cosine_function_l210_210637

def function := λ x : ℝ, Real.cos (3 * x + Real.pi / 3)

theorem period_cosine_function : ∃ T : ℝ, (T = 2 * Real.pi / 3) ∧ (∀ x : ℝ, function (x + T) = function x) :=
  sorry

end period_cosine_function_l210_210637


namespace maximize_probability_resilience_l210_210760

theorem maximize_probability_resilience (x : ℝ) (h : x > 0)
  (p : ℕ → ℝ) (K : vector ℕ 10)
  (p_def : ∀ s, p s = x^s / (1 + x + x^2))
  (K_def : K = ⟨[1, 2, 2, 1, 0, 2, 1, 0, 1, 2], by simp [vector.length]⟩) :
  x = (Real.sqrt 97 + 1) / 8 := 
sorry

end maximize_probability_resilience_l210_210760


namespace maximize_probability_resilience_l210_210759

theorem maximize_probability_resilience (x : ℝ) (h : x > 0)
  (p : ℕ → ℝ) (K : vector ℕ 10)
  (p_def : ∀ s, p s = x^s / (1 + x + x^2))
  (K_def : K = ⟨[1, 2, 2, 1, 0, 2, 1, 0, 1, 2], by simp [vector.length]⟩) :
  x = (Real.sqrt 97 + 1) / 8 := 
sorry

end maximize_probability_resilience_l210_210759


namespace find_points_per_enemy_l210_210643

def points_per_enemy (x : ℕ) : Prop :=
  let points_from_enemies := 6 * x
  let additional_points := 8
  let total_points := points_from_enemies + additional_points
  total_points = 62

theorem find_points_per_enemy (x : ℕ) (h : points_per_enemy x) : x = 9 :=
  by sorry

end find_points_per_enemy_l210_210643


namespace complex_magnitude_problem_l210_210312

-- Define the imaginary unit with the property i^2 = -1
def i : ℂ := complex.I

-- Prove that the magnitude of the complex number 2 + i² + 2i³ is √5
theorem complex_magnitude_problem : 
  complex.abs (2 + i^2 + 2 * i^3) = real.sqrt 5 := 
by
  -- Use the provided condition i² = -1
  have h : i^2 = -1 := by sorry,
  sorry

end complex_magnitude_problem_l210_210312


namespace probability_of_two_each_of_hearts_diamonds_clubs_l210_210496

theorem probability_of_two_each_of_hearts_diamonds_clubs :
  let p : ℝ := (nat.factorial 6 / (nat.factorial 2 * nat.factorial 2 * nat.factorial 2))
               * ((1 / 4) ^ 2) * ((1 / 4) ^ 2) * ((1 / 4) ^ 2) * ((1 / 4) ^ 0)
  in p = 90 / 4096 :=
by
  sorry

end probability_of_two_each_of_hearts_diamonds_clubs_l210_210496


namespace not_prime_41_squared_plus_41_plus_41_l210_210117

def is_prime (n : ℕ) : Prop := ∀ m k : ℕ, m * k = n → m = 1 ∨ k = 1

theorem not_prime_41_squared_plus_41_plus_41 :
  ¬ is_prime (41^2 + 41 + 41) :=
by {
  sorry
}

end not_prime_41_squared_plus_41_plus_41_l210_210117


namespace average_cost_is_70_l210_210680

noncomputable def C_before_gratuity (total_bill : ℝ) (gratuity_rate : ℝ) : ℝ :=
  total_bill / (1 + gratuity_rate)

noncomputable def average_cost_per_individual (C : ℝ) (total_people : ℝ) : ℝ :=
  C / total_people

theorem average_cost_is_70 :
  let total_bill := 756
  let gratuity_rate := 0.20
  let total_people := 9
  average_cost_per_individual (C_before_gratuity total_bill gratuity_rate) total_people = 70 :=
by
  sorry

end average_cost_is_70_l210_210680


namespace min_value_l210_210819

variable (x y : ℝ) 
variable (h_condition : x - 2 * y - 4 = 0)

theorem min_value : 2^x + (1 / (4^y)) = 8 := 
sorry

end min_value_l210_210819


namespace no_solution_k_l210_210772

theorem no_solution_k (k : ℝ) : 
  (∀ t s : ℝ, 
    ∃ (a : ℝ × ℝ) (b : ℝ × ℝ) (c : ℝ × ℝ) (d : ℝ × ℝ), 
      (a = (2, 7)) ∧ 
      (b = (5, -9)) ∧ 
      (c = (4, -3)) ∧ 
      (d = (-2, k)) ∧ 
      (a + t • b ≠ c + s • d)) ↔ k = 18 / 5 := 
by
  sorry

end no_solution_k_l210_210772


namespace max_total_length_N3_max_total_length_N4_l210_210149

-- Definitions for the conditions
def equator_length := 1
def ring_road_length := (N : ℕ) → 1
def train_speed := (N : ℕ) → positive_constant := sorry
def no_collision := true

-- Theorem statements for the maximum possible total length of trains for N = 3 and N = 4
theorem max_total_length_N3 : 
  (∀ (a₁ a₂ a₃ : ℝ), a₁ + a₂ ≤ 1 ∧ a₁ + a₃ ≤ 1 ∧ a₂ + a₃ ≤ 1 → a₁ + a₂ + a₃ ≤ 1.5) := sorry

theorem max_total_length_N4 :
  (∀ (a₁ a₂ a₃ a₄ : ℝ), a₁ + a₂ ≤ 1 ∧ a₁ + a₃ ≤ 1 ∧ a₁ + a₄ ≤ 1 ∧ a₂ + a₃ ≤ 1 ∧ a₂ + a₄ ≤ 1 ∧ a₃ + a₄ ≤ 1 → a₁ + a₂ + a₃ + a₄ ≤ 2) := sorry

end max_total_length_N3_max_total_length_N4_l210_210149


namespace complex_expression_magnitude_l210_210297

def i := Complex.I

theorem complex_expression_magnitude :
  |2 + i^2 + 2 * i^3| = Real.sqrt 5 :=
by
  sorry

end complex_expression_magnitude_l210_210297


namespace midpoint_trajectory_l210_210812

theorem midpoint_trajectory (x y p q : ℝ) (h_parabola : p^2 = 4 * q)
  (h_focus : ∀ (p q : ℝ), p^2 = 4 * q → q = (p/2)^2) 
  (h_midpoint_x : x = (p + 1) / 2)
  (h_midpoint_y : y = q / 2):
  y^2 = 2 * x - 1 :=
by
  sorry

end midpoint_trajectory_l210_210812


namespace area_of_quadrilateral_l210_210512

variable {A B C D : Type}
variables [inner_product_space ℝ A]

variables (AB AC AD BC BD DC : A)

-- Conditions
def BD_length : norm BD = 2 := sorry
def AC_perpendicular_BD : inner_product AC BD = 0 := sorry
def condition : inner_product (AB + DC) (BC + AD) = 5 := sorry

theorem area_of_quadrilateral :
  let AC : ℝ := 3 in
  let BD : ℝ := 2 in
  let area : ℝ := 1/2 * AC * BD in
  area = 3 :=
sorry

end area_of_quadrilateral_l210_210512


namespace triangles_not_congruent_if_angles_and_two_sides_equal_l210_210577

theorem triangles_not_congruent_if_angles_and_two_sides_equal (A B C D E F : Type)
  [Triangle A B C] [Triangle D E F]
  (hangle1 : ∠ A ≅ ∠ D) (hangle2 : ∠ B ≅ ∠ E) (hangle3 : ∠ C ≅ ∠ F)
  (hside1 : AB = DE) (hside2 : AC = DF) (hsidenot : BC ≠ EF) :
  ¬(A = D ∧ B = E ∧ C = F) :=
  sorry

end triangles_not_congruent_if_angles_and_two_sides_equal_l210_210577


namespace log_pascals_triangle_l210_210927

-- Define the product of elements in the nth row of Pascal's triangle
def pascal_row_product (n : ℕ) : ℕ := ∏ k in finset.range (n + 1), nat.choose n k

-- Define g(n) as given in the problem
noncomputable def g (n : ℕ) : ℝ := real.log (2^(n*(n + 1)/2) + n)

-- The main theorem statement
theorem log_pascals_triangle (n : ℕ) : (g n) / real.log 2 = (n * (n + 1) / 2) := by
  -- Skipping proof; added to ensure the Lean statement is complete
  sorry

end log_pascals_triangle_l210_210927


namespace time_to_fill_pool_l210_210141

def LindasPoolCapacity : ℕ := 30000
def CurrentVolume : ℕ := 6000
def NumberOfHoses : ℕ := 6
def RatePerHosePerMinute : ℕ := 3
def GallonsNeeded : ℕ := LindasPoolCapacity - CurrentVolume
def RatePerHosePerHour : ℕ := RatePerHosePerMinute * 60
def TotalHourlyRate : ℕ := NumberOfHoses * RatePerHosePerHour

theorem time_to_fill_pool : (GallonsNeeded / TotalHourlyRate) = 22 :=
by
  sorry

end time_to_fill_pool_l210_210141


namespace convex_derivative_inequality_l210_210926

open Set Real

noncomputable def convex_function (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y ∈ Ioo a b, ∀ t ∈ Ioo 0 1, f (t * x + (1 - t) * y) ≤ t * f x + (1 - t) * f y

noncomputable def one_sided_derivative (f : ℝ → ℝ) (x : ℝ) :=
  (differentiable_on ℝ f (Icc x x) ∧ ∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, 0 < h ∧ h < δ → abs ((f (x + h) - f x) / h - one_sided_deriv_within f (Ici x)))
 
theorem convex_derivative_inequality 
  (f : ℝ → ℝ) (a b x1 x2 x3 x4 : ℝ) 
  (h1 : a < b)
  (h_convex : convex_function f a b)
  (h_in_a_b : a < x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧ x4 < b)
  (h_deriv_1 : one_sided_derivative f x2)
  (h_deriv_2 : one_sided_derivative f x3)
  : 
  (f x2 - f x1) / (x2 - x1) ≤ one_sided_deriv_l f x2 ∧ 
  one_sided_deriv_l f x2 ≤ one_sided_deriv_r f x2 ∧ 
  one_sided_deriv_r f x2 ≤ one_sided_deriv_l f x3 ∧ 
  one_sided_deriv_l f x3 ≤ one_sided_deriv_r f x3 ∧ 
  one_sided_deriv_r f x3 ≤ (f x4 - f x3) / (x4 - x3) :=
sorry

end convex_derivative_inequality_l210_210926


namespace correct_statement_l210_210010

universe u
variable (α : Type u)

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5}

noncomputable def complement_U_M : Set ℕ := {1, 3}

noncomputable def M : Set ℕ := U α \ complement_U_M

theorem correct_statement : 2 ∈ M :=
by {
  sorry
}

end correct_statement_l210_210010


namespace max_area_rectangle_shorter_side_l210_210280

theorem max_area_rectangle_shorter_side (side_length : ℕ) (n : ℕ)
  (hsq : side_length = 40) (hn : n = 5) :
  ∃ (shorter_side : ℕ), shorter_side = 8 := by
  sorry

end max_area_rectangle_shorter_side_l210_210280


namespace complex_magnitude_l210_210303

variables (i : ℂ)
axiom imaginary_unit : i^2 = -1

theorem complex_magnitude : |2 + i^2 + 2 * i^3| = Real.sqrt 5 :=
by
  have h1 : i^3 = i^2 * i := by sorry
  have h2 : i^2 = -1 := imaginary_unit
  have h3 : i^3 = -i := by sorry
  calc 
    |2 + i^2 + 2 * i^3| = |2 + (-1) + 2 * (-i)| : by sorry
    ... = |1 - 2 * i| : by sorry
    ... = Real.sqrt (1^2 + (-2)^2) : by sorry
    ... = Real.sqrt 5 : by sorry

end complex_magnitude_l210_210303


namespace part1_part2_l210_210460

def f (x : ℝ) : ℝ := |2 * x + 2| - 5

def g (f : ℝ → ℝ) (x m : ℝ) : ℝ := f(x) + |x - m|

theorem part1 (x : ℝ) : 
  f(x) - |x - 1| ≥ 0 ↔ x ≤ -8 ∨ x ≥ 2 :=
by
  sorry

theorem part2 (m : ℝ) (h : 0 < m) :
  (∀ x : ℝ, (g f x m < 0 ↔ x ≤ -1) ∧
           (g f x m = 0 ↔ x = m) ∧
           (g f x m > 0 ↔ x > m)) ↔
  3 / 2 ≤ m ∧ m < 4 :=
by
  sorry

end part1_part2_l210_210460


namespace count_integers_satisfying_equation_l210_210784

theorem count_integers_satisfying_equation :
  (Set.count ((fun n => n ≤ 1000 ∧ ∃ t : ℝ, (Complex.sin t + Complex.cos t * Complex.i) ^ n = Complex.sin (n * t) + Complex.cos (n * t) * Complex.i) Set.univ)) = 250 :=
sorry

end count_integers_satisfying_equation_l210_210784


namespace problem_solution_l210_210009

universe u

variable (U : Set Nat) (M : Set Nat)
variable (complement_U_M : Set Nat)

axiom U_def : U = {1, 2, 3, 4, 5}
axiom complement_U_M_def : complement_U_M = {1, 3}
axiom M_def : M = U \ complement_U_M

theorem problem_solution : 2 ∈ M := by
  sorry

end problem_solution_l210_210009


namespace sum_min_values_eq_zero_l210_210712

-- Definitions of the polynomials
def P (x : ℝ) (a b : ℝ) : ℝ := x^2 + a*x + b
def Q (x : ℝ) (c d : ℝ) : ℝ := x^2 + c*x + d

-- Main theorem statement
theorem sum_min_values_eq_zero (b d : ℝ) :
  let a := -16
  let c := -8
  (-64 + b = 0) ∧ (-16 + d = 0) → (-64 + b + (-16 + d) = 0) :=
by
  intros
  rw [add_assoc]
  sorry

end sum_min_values_eq_zero_l210_210712


namespace opposite_of_five_l210_210210

theorem opposite_of_five : ∃ y : ℤ, 5 + y = 0 ∧ y = -5 := by
  use -5
  constructor
  . exact rfl
  . sorry

end opposite_of_five_l210_210210


namespace coeff_x3_in_expansion_of_1_plus_2x_pow_n_l210_210455

theorem coeff_x3_in_expansion_of_1_plus_2x_pow_n
  (n : ℕ)
  (h : (∑ i in finset.range (n + 1), (nat.choose n i) * (2:ℤ) ^ i) = 81) :
  (nat.choose n 3) * 2^3 = 32 := 
sorry

end coeff_x3_in_expansion_of_1_plus_2x_pow_n_l210_210455


namespace rational_root_of_polynomial_l210_210729

theorem rational_root_of_polynomial :
  ∀ (b c : ℚ),
  (∀ x : ℂ, x^3 - 4 * x^2 + b * x + c = 0 → x = (4 - Real.sqrt 11) ∨ x = (4 + Real.sqrt 11) ∨ x = -4) →
  (∃ γ : ℚ, x^3 - 4 * x^2 + b * x + c = 0 ∧ γ = -4) :=
by
  intros b c h
  use -4
  split
  { exact sorry }
  { reflexivity }
  sorry

end rational_root_of_polynomial_l210_210729


namespace proof_2_in_M_l210_210035

def U : Set ℕ := {1, 2, 3, 4, 5}

def M : Set ℕ := { x | x ∈ U ∧ x ≠ 1 ∧ x ≠ 3 }

theorem proof_2_in_M : 2 ∈ M :=
by
  sorry

end proof_2_in_M_l210_210035


namespace carboxylic_acid_formula_l210_210690

theorem carboxylic_acid_formula (n : ℕ) (mass_fraction_oxygen : ℝ) :
  mass_fraction_oxygen = 0.4325 →
  (n = 3) →
  let formula := "C" ++ toString n ++ "H" ++ toString (2 * n + 2) ++ "O2" in
  formula = "C3H6O2" :=
by
  intros hmf hn
  sorry

end carboxylic_acid_formula_l210_210690


namespace part1_tangent_line_part2_range_m_l210_210886

-- Definition of the parametric equations of curve C
noncomputable def curve_C_x (θ : ℝ) : ℝ := 1 + sqrt 3 * cos θ
noncomputable def curve_C_y (θ : ℝ) : ℝ := sqrt 3 * sin θ

-- Definition of the Cartesian form of line l for a given parameter m
noncomputable def line_l_cartesian (m : ℝ) (x y : ℝ) : Prop := 
  y + sqrt 3 * x - sqrt 3 * m = 0

-- Part (1): Positional relationship between line l and curve C when m = 3
theorem part1_tangent_line (θ : ℝ) :
  let x := curve_C_x θ
  let y := curve_C_y θ
  line_l_cartesian 3 x y ↔ (x - 1)^2 + y^2 = 3 :=
sorry

-- Part (2): Range of the real number m for which there exists a point on curve C with distance sqrt(3)/2 to line l
theorem part2_range_m (θ : ℝ) (m : ℝ) :
  let x := curve_C_x θ
  let y := curve_C_y θ
  let distance := abs (sqrt 3 - sqrt 3 * m) / 2
  distance = sqrt 3 / 2 →
  -2 ≤ m ∧ m ≤ 4 :=
sorry

end part1_tangent_line_part2_range_m_l210_210886


namespace smallest_positive_period_l210_210980

open Real

-- Define conditions
def max_value_condition (b a : ℝ) : Prop := b + a = -1
def min_value_condition (b a : ℝ) : Prop := b - a = -5

-- Define the period of the function
def period (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

-- Main theorem
theorem smallest_positive_period (a b : ℝ) (h1 : a < 0) 
  (h2 : max_value_condition b a) 
  (h3 : min_value_condition b a) : 
  period (fun x => tan ((3 * a + b) * x)) (π / 9) :=
by
  sorry

end smallest_positive_period_l210_210980


namespace find_f2_l210_210816

-- Define f as an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define g based on f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  f x + 9

-- Given conditions
variables (f : ℝ → ℝ) (h_odd : odd_function f)
variable (h_g_neg2 : g f (-2) = 3)

-- Theorem statement
theorem find_f2 : f 2 = 6 :=
by
  sorry

end find_f2_l210_210816


namespace sector_central_angle_l210_210798

-- The conditions
def r : ℝ := 2
def S : ℝ := 4

-- The question
theorem sector_central_angle : ∃ α : ℝ, |α| = 2 ∧ S = 0.5 * α * r * r :=
by
  sorry

end sector_central_angle_l210_210798


namespace all_options_valid_l210_210201

-- Definition of the line equation
def line_eq (x y : ℝ) : Prop := y = 2 * x - 4

-- Definitions of parameterizations for each option
def option_A (t : ℝ) : ℝ × ℝ := ⟨2 + (-1) * t, 0 + (-2) * t⟩
def option_B (t : ℝ) : ℝ × ℝ := ⟨6 + 4 * t, 8 + 8 * t⟩
def option_C (t : ℝ) : ℝ × ℝ := ⟨1 + 1 * t, -2 + 2 * t⟩
def option_D (t : ℝ) : ℝ × ℝ := ⟨0 + 0.5 * t, -4 + 1 * t⟩
def option_E (t : ℝ) : ℝ × ℝ := ⟨-2 + (-2) * t, -8 + (-4) * t⟩

-- The main statement to prove
theorem all_options_valid :
  (∀ t, line_eq (option_A t).1 (option_A t).2) ∧
  (∀ t, line_eq (option_B t).1 (option_B t).2) ∧
  (∀ t, line_eq (option_C t).1 (option_C t).2) ∧
  (∀ t, line_eq (option_D t).1 (option_D t).2) ∧
  (∀ t, line_eq (option_E t).1 (option_E t).2) :=
by sorry -- proof omitted

end all_options_valid_l210_210201


namespace perpendicular_ba_ac_l210_210710

-- Definitions of points and segments
variables {A B C I D : Type}

-- Definitions of perpendicularity and segments
variable [geometry : Geometry A B C I D]

-- Given conditions:
-- 1. \(I\) is the incenter of \(\triangle ABC\)
-- 2. \(ID \perp BC\) at \(D\)
-- 3. \(AB \cdot AC = 2BD \cdot DC\)

theorem perpendicular_ba_ac
  (h_incenter : I = incenter A B C)
  (h_id_perp_bc : Perpendicular ID BC D)
  (h_ab_ac : AB * AC = 2 * BD * DC) :
  Perpendicular BA AC :=
sorry

end perpendicular_ba_ac_l210_210710


namespace unique_poly_exists_l210_210154

theorem unique_poly_exists (n : ℤ) :
  ∃! (Q : ℤ[X]), (∀ k, (coeff Q k ∈ finset.range 10)) ∧ Q.eval (-2) = n ∧ Q.eval (-5) = n :=
sorry

end unique_poly_exists_l210_210154


namespace three_planes_max_division_l210_210622

-- Define the condition: three planes
variable (P1 P2 P3 : Plane)

-- Define the proof statement: three planes can divide the space into at most 8 parts
theorem three_planes_max_division : divides_space_at_most P1 P2 P3 8 :=
  sorry

end three_planes_max_division_l210_210622


namespace area_of_equilateral_triangle_with_substructures_abc_l210_210884

-- Definition for an equilateral triangle
structure EquilateralTriangle (α : Type) [LinearOrderedField α] :=
(a b c : α)
(is_equilateral : a = b ∧ b = c)

-- Definition for points D, E, F on sides of the triangle, distances known
structure PointsOnSides (α : Type) [LinearOrderedField α] :=
(a b c : α)
(d e f : α)
(on_sides : 0 ≤ d ∧ d ≤ a ∧ 0 ≤ e ∧ e ≤ b ∧ 0 ≤ f ∧ f ≤ c)
(equal_distances : d + e = e + f ∧ e + f = f + d ∧ f + d = 10 * 2)

-- Main theorem
theorem area_of_equilateral_triangle_with_substructures_abc (α : Type) [LinearOrderedField α] 
  (a b c x : α) (h_triang_ABC : EquilateralTriangle α) 
  (h_points_abc : PointsOnSides α) :
  (a = b ∧ b = c) → 
  (10 = 10) →
  ∃ (area : α), area = 400 * (Real.sqrt 3) / 9 :=
begin
  sorry,
end

end area_of_equilateral_triangle_with_substructures_abc_l210_210884


namespace probability_of_picking_two_red_balls_l210_210644

open Nat

theorem probability_of_picking_two_red_balls :
  let total_balls := 4 + 3 + 2
  let total_ways := Combination 9 2
  let favorable_ways := Combination 4 2
  let probability := favorable_ways / total_ways
  probability = 1 / 6 := by
  let total_balls := 9
  let total_ways := Combination total_balls 2
  let favorable_ways := Combination 4 2
  have total_ways_is_36 : total_ways = 36 := by
    calculate the value via the combination formula
    sorry
  have favorable_ways_is_6 : favorable_ways = 6 := by
    calculate the value via the combination formula
    sorry
  let probability := favorable_ways / total_ways
  have probability_is_1_over_6 : probability = (1 / 6) := by
    calculate the final probability value
    sorry
  exact probability_is_1_over_6

end probability_of_picking_two_red_balls_l210_210644


namespace range_of_a_l210_210471

theorem range_of_a
  (a : ℝ)
  (h1 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 ≥ a)
  (h2 : ∃ x0 : ℝ, x0^2 + 2*a*x0 + 2 - a = 0) :
  a ≤ -2 ∨ a = 1 :=
sorry

end range_of_a_l210_210471


namespace S_diff_multiple_of_2011_l210_210229

/-- Define the sequence S_n as per given conditions -/
def S : ℕ → ℕ
| n => if n ≤ 2011 then 1 else S (n - 2012) + S (n - 2011)

/-- The main theorem to prove -/
theorem S_diff_multiple_of_2011 (a : ℕ) : (S (2011 * a) - S a) % 2011 = 0 :=
sorry

end S_diff_multiple_of_2011_l210_210229


namespace value_of_M_l210_210487

theorem value_of_M (M : ℝ) (h : 0.25 * M = 0.35 * 1200) : M = 1680 := 
sorry

end value_of_M_l210_210487


namespace abs_val_equality_l210_210864

theorem abs_val_equality (m : ℝ) (h : |m| = |(-3 : ℝ)|) : m = 3 ∨ m = -3 :=
sorry

end abs_val_equality_l210_210864


namespace dragon_resilience_maximized_l210_210755

noncomputable def probability (x : ℝ) (s : ℕ) : ℝ :=
  x^s / (1 + x + x^2)

noncomputable def prob_vec (x : ℝ) : ℝ :=
  let K := [1, 2, 2, 1, 0, 2, 1, 0, 1, 2]
  K.foldr (λ s acc, acc * probability x s) 1

theorem dragon_resilience_maximized (x : ℝ) : 
  (x = (Real.sqrt 97 + 1) / 8) → 
  (0 < x) →
  ∀ K, K = [1, 2, 2, 1, 0, 2, 1, 0, 1, 2] →
  prob_vec x = (x^12 / (1 + x + x^2)^10) :=
begin
  sorry
end

end dragon_resilience_maximized_l210_210755


namespace cost_of_expensive_feed_l210_210252

open Lean Real

theorem cost_of_expensive_feed (total_feed : Real)
                              (total_cost_per_pound : Real) 
                              (cheap_feed_weight : Real)
                              (cheap_cost_per_pound : Real)
                              (expensive_feed_weight : Real)
                              (expensive_cost_per_pound : Real):
  total_feed = 35 ∧ 
  total_cost_per_pound = 0.36 ∧ 
  cheap_feed_weight = 17 ∧ 
  cheap_cost_per_pound = 0.18 ∧ 
  expensive_feed_weight = total_feed - cheap_feed_weight →
  total_feed * total_cost_per_pound - cheap_feed_weight * cheap_cost_per_pound = expensive_feed_weight * expensive_cost_per_pound →
  expensive_cost_per_pound = 0.53 :=
by {
  sorry
}

end cost_of_expensive_feed_l210_210252


namespace locus_of_M_exists_fixed_point_C_l210_210839

-- Definitions to match with the conditions
def isOnHyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 = 2

def foci1 : ℝ × ℝ := (-2, 0)
def foci2 : ℝ × ℝ := (2, 0)
def origin : ℝ × ℝ := (0, 0)

-- Define the vector conditions
def vectorF1M (M : ℝ × ℝ) : ℝ × ℝ := (M.1 + 2, M.2)
def vectorF1A (A : ℝ × ℝ) : ℝ × ℝ := (A.1 + 2, A.2)
def vectorF1B (B : ℝ × ℝ) : ℝ × ℝ := (B.1 + 2, B.2)
def vectorF1O : ℝ × ℝ := (2, 0)

-- Define the condition for M
def M_condition (M A B : ℝ × ℝ) : Prop :=
  vectorF1M M = vectorF1A A + vectorF1B B + vectorF1O

-- The proof goals
theorem locus_of_M (M A B : ℝ × ℝ) :
  (∃ (M : ℝ × ℝ), (∃ (A B : ℝ × ℝ), isOnHyperbola A.1 A.2 ∧ isOnHyperbola B.1 B.2 ∧ M_condition M A B)) →
  (∃ (x y : ℝ), M = (x, y) ∧ (x - 6)^2 - y^2 = 4) :=
sorry

theorem exists_fixed_point_C (A B : ℝ × ℝ) :
  (∃ (C : ℝ × ℝ), C.2 = 0 ∧ ∀ (A B : ℝ × ℝ), (isOnHyperbola A.1 A.2 ∧ isOnHyperbola B.1 B.2) → ((C.1 - A.1) * (C.1 - B.1) + C.2 * A.2 * C.2 * B.2) = -1) :=
∃ C, C = (1, 0) :=
sorry

end locus_of_M_exists_fixed_point_C_l210_210839


namespace arcsin_one_eq_pi_div_two_l210_210719

noncomputable def arcsin (x : ℝ) : ℝ :=
classical.some (exists_inverse_sin x)

theorem arcsin_one_eq_pi_div_two : arcsin 1 = π / 2 :=
sorry

end arcsin_one_eq_pi_div_two_l210_210719


namespace inequality_holds_iff_l210_210641

theorem inequality_holds_iff (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → x^2 + (a - 4) * x + 4 > 0) ↔ a > 0 :=
by
  sorry

end inequality_holds_iff_l210_210641


namespace probability_even_two_digit_number_l210_210241

theorem probability_even_two_digit_number :
  (probability (draw_two_cards_even 5 {1, 2, 3, 4, 5}) = 2 / 5) :=
sorry

def draw_two_cards_even (n : ℕ) (s : finset ℕ) : set (ℕ × ℕ) :=
  {pair | pair.1 ∈ s ∧ pair.2 ∈ s ∧ (10 * pair.1 + pair.2) % 2 = 0}

def probability (event : set (ℕ × ℕ)) : ℚ :=
  (event.card : ℚ) / ((finset.univ : finset (ℕ × ℕ)).card : ℚ)

end probability_even_two_digit_number_l210_210241


namespace problem_solution_l210_210005

universe u

variable (U : Set Nat) (M : Set Nat)
variable (complement_U_M : Set Nat)

axiom U_def : U = {1, 2, 3, 4, 5}
axiom complement_U_M_def : complement_U_M = {1, 3}
axiom M_def : M = U \ complement_U_M

theorem problem_solution : 2 ∈ M := by
  sorry

end problem_solution_l210_210005


namespace min_moves_move_stack_from_A_to_F_l210_210938

theorem min_moves_move_stack_from_A_to_F : 
  ∀ (squares : Fin 6) (stack : Fin 15), 
  (∃ moves : Nat, 
    (moves >= 0) ∧ 
    (moves == 49) ∧
    ∀ (a b : Fin 6), 
        ∃ (piece_from : Fin 15) (piece_to : Fin 15), 
        ((piece_from > piece_to) → (a ≠ b)) ∧
        (a == 0) ∧ 
        (b == 5)) :=
sorry

end min_moves_move_stack_from_A_to_F_l210_210938


namespace nell_has_252_cards_left_l210_210554

theorem nell_has_252_cards_left (n : ℕ) (j : ℕ) (k : ℕ) :
  n = 528 → j = 11 → k = 287 → (n - (k - j) = 252) :=
by
  intros h_n h_j h_k
  rw [h_n, h_j, h_k]
  norm_num


end nell_has_252_cards_left_l210_210554


namespace triangle_area_at_1_second_l210_210374

noncomputable def side_length : ℝ := 0.4 -- side length of the square in meters
noncomputable def M : ℝ × ℝ := (0, 0.2) -- midpoint of AD
noncomputable def N : ℝ × ℝ := (0.4, 0.2) -- midpoint of BC

def P (t : ℝ) : ℝ × ℝ :=
  if t ≤ 0.4 then (0, 0.2 + t)
  else if t ≤ 0.8 then (t - 0.4, 0.6)
  else if t ≤ 1.2 then (0.4, 0.8 - (t - 0.8))
  else (0.4 - (t - 1.2), 0)

def Q (t : ℝ) : ℝ × ℝ :=
  if t ≤ 0.2 then (0, 0.2 - t)
  else if t ≤ 0.6 then (0.2 + (t - 0.2), 0)
  else if t ≤ 1 then (0.6, t - 0.6)
  else (1 - (t - 1), 0.4)

def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * |(A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))|

theorem triangle_area_at_1_second :
  area_of_triangle N (P 1) (Q 1) = 6 :=
sorry

end triangle_area_at_1_second_l210_210374


namespace sequence_general_term_l210_210454

theorem sequence_general_term 
  (f : ℕ → ℝ) 
  (a : ℝ) 
  (a_pos : a > 0) 
  (a_ne_one : a ≠ 1) 
  (point_on_graph : f 1 = 2) 
  (Sn : ℕ → ℝ) 
  (Sn_def : ∀ n, Sn n = f n - 1) :
  ∀ n, f = (λ n, a^n) → a = 2 → Sn = (λ n, 2^n - 1) → (∀ n, {a_n : ℕ} = 2^(n-1)) :=
sorry

end sequence_general_term_l210_210454


namespace count_ordered_pairs_l210_210062

theorem count_ordered_pairs :
  {p : Int × Int // p.1^2 + p.2^2 < 25 ∧ p.1^2 + p.2^2 < 9 * p.1 ∧ p.1^2 + p.2^2 < 9 * p.2}.toFinset.card = 4 :=
sorry

end count_ordered_pairs_l210_210062


namespace binom_div_undefined_l210_210424

noncomputable def binom (a : ℝ) (k : ℕ) : ℝ :=
  if k = 0 then 1 else (List.prod (List.map (λ i, a - i) (List.range k)) / (Nat.factorial k))

theorem binom_div_undefined : 
  (binom 1 10) / (binom (-1) 10) = 0 → False :=
by
  sorry

end binom_div_undefined_l210_210424


namespace complex_magnitude_l210_210306

variables (i : ℂ)
axiom imaginary_unit : i^2 = -1

theorem complex_magnitude : |2 + i^2 + 2 * i^3| = Real.sqrt 5 :=
by
  have h1 : i^3 = i^2 * i := by sorry
  have h2 : i^2 = -1 := imaginary_unit
  have h3 : i^3 = -i := by sorry
  calc 
    |2 + i^2 + 2 * i^3| = |2 + (-1) + 2 * (-i)| : by sorry
    ... = |1 - 2 * i| : by sorry
    ... = Real.sqrt (1^2 + (-2)^2) : by sorry
    ... = Real.sqrt 5 : by sorry

end complex_magnitude_l210_210306


namespace exists_nat_with_equal_digits_and_divisibility_l210_210896

-- Define a function that checks if a natural number contains equal amounts of digit 7 and digit 5
def equal_sevens_and_fives (n : ℕ) : Prop :=
  let n_str := nat_to_string n
  let count7 := n_str.count('7')
  let count5 := n_str.count('5')
  count7 = count5

-- The main theorem stating the existence of such a natural number
theorem exists_nat_with_equal_digits_and_divisibility : 
  ∃ n : ℕ, (∀ d, d ∈ n.digits 10 → d = 7 ∨ d = 5) ∧ 
           equal_sevens_and_fives n ∧
           7 ∣ n ∧
           5 ∣ n :=
by 
  -- Example from solution based on manual calculations
  let n := 5775 
  have h1 : ∀ d, d ∈ n.digits 10 → d = 7 ∨ d = 5 := sorry
  have h2 : equal_sevens_and_fives n := sorry
  have h3 : 7 ∣ n := sorry
  have h4 : 5 ∣ n := sorry
  exact ⟨n, h1, h2, h3, h4⟩

end exists_nat_with_equal_digits_and_divisibility_l210_210896


namespace complement_union_l210_210474

-- Definitions of sets A and B based on the conditions
def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x ≤ 0}

def B : Set ℝ := {x | x ≥ 1}

-- Theorem to prove the complement of the union of sets A and B within U
theorem complement_union (x : ℝ) : x ∉ (A ∪ B) ↔ (0 < x ∧ x < 1) := by
  sorry

end complement_union_l210_210474


namespace sum_exterior_angles_of_regular_dodecagon_l210_210990

theorem sum_exterior_angles_of_regular_dodecagon : 
  ∀ (P : Type) [polyhedron P] (f : P ≃ (ℕ × ℕ → ℝ)), sum (exterior_angles P f) = 360 :=
sorry

end sum_exterior_angles_of_regular_dodecagon_l210_210990


namespace sabina_college_cost_l210_210160

noncomputable def total_cost_of_college (savings grant_coverage loan: ℝ) : ℝ :=
  let remaining_cost := loan + (1 - grant_coverage) * (total_cost - savings)
  savings + remaining_cost / (1 - grant_coverage)

theorem sabina_college_cost
  (savings : ℝ := 10000)
  (grant_coverage : ℝ := 0.40)
  (loan : ℝ := 12000)
  (total_cost : ℝ := 30000) :
  total_cost_of_college savings grant_coverage loan = total_cost :=
by
  sorry

end sabina_college_cost_l210_210160


namespace selectFourPeopleProbProof_l210_210495

noncomputable def selectFourPeopleProbability
  (totalCouples : ℕ)
  (selectFemales : ℕ)
  (selectMales : ℕ)
  (M : ℕ)
  (binom : ℕ) : ℕ :=
M / binom

theorem selectFourPeopleProbProof
  (totalCouples : ℕ)
  (selectMales : ℕ)
  (selectFemales : ℕ)
  (M : ℕ) :
  totalCouples = 9 →
  selectMales = 2 →
  selectFemales = 2 →
  (∀ (M : ℕ), 0 ≤ M) →
  selectFourPeopleProbability totalCouples selectFemales selectMales M (Nat.choose 9 2) = M / (Nat.choose 9 2) :=
by
  intros h_total h_males h_females h_m
  rw [selectFourPeopleProbability]
  exact h_m M

end selectFourPeopleProbProof_l210_210495


namespace susans_average_speed_l210_210581

theorem susans_average_speed :
  ∀ (total_distance first_leg_distance second_leg_distance : ℕ)
    (first_leg_speed second_leg_speed : ℕ)
    (total_time : ℚ),
    first_leg_distance = 40 →
    second_leg_distance = 20 →
    first_leg_speed = 15 →
    second_leg_speed = 60 →
    total_distance = first_leg_distance + second_leg_distance →
    total_time = (first_leg_distance / first_leg_speed : ℚ) + (second_leg_distance / second_leg_speed : ℚ) →
    total_distance / total_time = 20 :=
by
  sorry

end susans_average_speed_l210_210581


namespace first_year_sum_of_digits_15_l210_210636

def sumOfDigits (year : Nat) : Nat :=
  (year / 1000) + (year % 1000 / 100) + (year % 100 / 10) + (year % 10)

theorem first_year_sum_of_digits_15 : ∃ (y : Nat), y > 2010 ∧ sumOfDigits y = 15 ∧ ∀ z, z > 2010 → sumOfDigits z = 15 → y ≤ z :=
by {
  use 2049,
  split,
  { exact Nat.lt_succ_of_lt Nat.zero_lt_one },
  split,
  { norm_num },
  { sorry }
}

end first_year_sum_of_digits_15_l210_210636


namespace dragon_resilience_maximized_l210_210757

noncomputable def probability (x : ℝ) (s : ℕ) : ℝ :=
  x^s / (1 + x + x^2)

noncomputable def prob_vec (x : ℝ) : ℝ :=
  let K := [1, 2, 2, 1, 0, 2, 1, 0, 1, 2]
  K.foldr (λ s acc, acc * probability x s) 1

theorem dragon_resilience_maximized (x : ℝ) : 
  (x = (Real.sqrt 97 + 1) / 8) → 
  (0 < x) →
  ∀ K, K = [1, 2, 2, 1, 0, 2, 1, 0, 1, 2] →
  prob_vec x = (x^12 / (1 + x + x^2)^10) :=
begin
  sorry
end

end dragon_resilience_maximized_l210_210757


namespace tetrahedron_sum_equal_opposite_edges_l210_210131

variable {a a1 b b1 c c1 : ℝ}
variable {α β γ : ℝ}

-- Conditions
axiom tetrahedron_opposite_edges : a * a1 * cos α = b * b1 * cos β + c * c1 * cos γ ∨ 
                                  b * b1 * cos β = a * a1 * cos α + c * c1 * cos γ ∨ 
                                  c * c1 * cos γ = a * a1 * cos α + b * b1 * cos β

-- Theorem 
theorem tetrahedron_sum_equal_opposite_edges 
    (ha_angle : 0 ≤ α ∧ α ≤ (π / 2))
    (hb_angle : 0 ≤ β ∧ β ≤ (π / 2))
    (hc_angle : 0 ≤ γ ∧ γ ≤ (π / 2)):
    a * a1 * cos α = b * b1 * cos β + c * c1 * cos γ ∨ 
    b * b1 * cos β = a * a1 * cos α + c * c1 * cos γ ∨ 
    c * c1 * cos γ = a * a1 * cos α + b * b1 * cos β :=
begin
  exact tetrahedron_opposite_edges,
end

end tetrahedron_sum_equal_opposite_edges_l210_210131


namespace sin_alpha_given_cos_alpha_plus_pi_over_3_l210_210073

theorem sin_alpha_given_cos_alpha_plus_pi_over_3 
  (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos (α + π / 3) = 1 / 5) : 
  Real.sin α = (2 * Real.sqrt 6 - Real.sqrt 3) / 10 := 
by 
  sorry

end sin_alpha_given_cos_alpha_plus_pi_over_3_l210_210073


namespace bubble_pass_1191_l210_210960

/-- 
  Given a sequence of 50 distinct elements, the probability that the element initially 
  at the 25th position ends up at the 35th position after one bubble pass is \( \frac{1}{1190} \). 
  Therefore, \( p + q = 1191 \) if \( \frac{p}{q} \) is in the lowest terms.
-/
theorem bubble_pass_1191 (n : ℕ) (r : Fin n → ℕ) (hn : n = 50) (h_distinct : Function.Injective r)
  (h_random_order : ∀ (i j : Fin n), i ≤ j -> (r i ≤ r j) → r i = r j) : 
  let p := 1 in let q := 1190 in p + q = 1191 :=
by
  sorry

end bubble_pass_1191_l210_210960


namespace sacksPerSectionDaily_l210_210593

variable (totalSacks : ℕ) (sections : ℕ) (sacksPerSection : ℕ)

-- Conditions from the problem
variables (h1 : totalSacks = 360) (h2 : sections = 8)

-- The theorem statement
theorem sacksPerSectionDaily : sacksPerSection = 45 :=
by
  have h3 : totalSacks / sections = 45 := by sorry
  have h4 : sacksPerSection = totalSacks / sections := by sorry
  exact Eq.trans h4 h3

end sacksPerSectionDaily_l210_210593


namespace mia_mom_total_time_l210_210936

-- Define the conditions as constants
const mom_toys_per_cycle : Nat := 4
const mia_toys_removed_per_cycle : Nat := 3
const total_toys : Nat := 45
const initial_minute_cycles : Nat := 2
const toy_goal_excluding_initial_minute : Nat := total_toys - (mom_toys_per_cycle * initial_minute_cycles)
const net_toys_per_cycle := mom_toys_per_cycle - mia_toys_removed_per_cycle
const cycle_time_seconds : Nat := 30

-- The statement of the final theorem to prove
theorem mia_mom_total_time : 
  cycle_time_seconds * initial_minute_cycles + 
  cycle_time_seconds * (toy_goal_excluding_initial_minute / net_toys_per_cycle) / 60 = 22 := 
by
  sorry

end mia_mom_total_time_l210_210936


namespace hyperbola_condition_l210_210828

theorem hyperbola_condition (k : ℝ) (x y : ℝ) :
  (k ≠ 0 ∧ k ≠ 3 ∧ (x^2 / k + y^2 / (k - 3) = 1)) → 0 < k ∧ k < 3 :=
by
  sorry

end hyperbola_condition_l210_210828


namespace correctStatement_l210_210019

variable (U : Set ℕ) (M : Set ℕ)

namespace Proof

-- Given conditions
def universalSet := {1, 2, 3, 4, 5}
def complementM := {1, 3}
def isComplement (M : Set ℕ) : Prop := U \ M = complementM

-- Target statement to be proved
theorem correctStatement (h1 : U = universalSet) (h2 : isComplement M) : 2 ∈ M := by
  sorry

end Proof

end correctStatement_l210_210019


namespace carB_speed_is_correct_l210_210381

def vA : ℕ := 58
def t : ℝ := 4.75
def d_initial_gap : ℕ := 30
def d_ahead : ℕ := 8

-- Calculate distances
def d_A := vA * t   -- Distance Car A travels in 4.75 hours
def d_gain := d_initial_gap + d_ahead -- Distance Car A needs to gain

-- Define the speed of Car B
def vB : ℝ := (d_A - d_gain) / t

theorem carB_speed_is_correct : vB = 50 := 
by
  -- Skip the proof, only the statement is required
  sorry

end carB_speed_is_correct_l210_210381


namespace collinear_points_l210_210948

-- Define the points with their coordinates
def pointA := (-1, -2)
def pointB := (2, -1)
def pointC := (8, 1)

-- The theorem to prove collinearity
theorem collinear_points (A B C : ℝ × ℝ) : (A = pointA) ∧ (B = pointB) ∧ (C = pointC) → 
  ∃ k : ℝ, B.1 = A.1 + k * (C.1 - A.1) ∧ B.2 = A.2 + k * (C.2 - A.2) :=
by
  sorry

end collinear_points_l210_210948


namespace dragon_resilience_l210_210764

noncomputable def probability_function (x : ℝ) : ℝ :=
  let p0 := 1 / (1 + x + x^2)
  let p1 := x / (1 + x + x^2)
  let p2 := x^2 / (1 + x + x^2)
  p1 * p2^2 * p1 * p0 * p2 * p1 * p0 * p1 * p2

theorem dragon_resilience (x : ℝ) (hx : x > 0) : 
  has_max (λ x, probability_function x) ∧ probability_function (sqrt 97 + 1) / 8 = max :=
sorry

end dragon_resilience_l210_210764


namespace circle_symmetric_about_line_l210_210832

theorem circle_symmetric_about_line :
  ∃ b : ℝ, (∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + 4 = 0 → y = 2*x + b) → b = 4 :=
by
  sorry

end circle_symmetric_about_line_l210_210832


namespace ordinary_equation_of_curve_l210_210520

def parametric_eq_x (t : ℝ) : ℝ := 2 + (real.sqrt 2 / 2) * t
def parametric_eq_y (t : ℝ) : ℝ := 1 + (real.sqrt 2 / 2) * t

theorem ordinary_equation_of_curve (t : ℝ) :
  parametric_eq_x t - parametric_eq_y t = 1 := 
sorry

end ordinary_equation_of_curve_l210_210520


namespace average_weight_of_girls_l210_210971

theorem average_weight_of_girls (avg_weight_boys : ℕ) (num_boys : ℕ) (avg_weight_class : ℕ) (num_students : ℕ) :
  num_boys = 15 →
  avg_weight_boys = 48 →
  num_students = 25 →
  avg_weight_class = 45 →
  ( (avg_weight_class * num_students - avg_weight_boys * num_boys) / (num_students - num_boys) ) = 27 :=
by
  intros h_num_boys h_avg_weight_boys h_num_students h_avg_weight_class
  sorry

end average_weight_of_girls_l210_210971


namespace stocking_stuffers_total_l210_210054

theorem stocking_stuffers_total 
  (candy_canes_per_child beanie_babies_per_child books_per_child : ℕ)
  (num_children : ℕ)
  (h1 : candy_canes_per_child = 4)
  (h2 : beanie_babies_per_child = 2)
  (h3 : books_per_child = 1)
  (h4 : num_children = 3) :
  candy_canes_per_child + beanie_babies_per_child + books_per_child * num_children = 21 :=
by
  sorry

end stocking_stuffers_total_l210_210054


namespace quadratic_transform_l210_210985

theorem quadratic_transform (x : ℝ) : x^2 - 6 * x - 5 = 0 → (x - 3)^2 = 14 :=
by
  intro h
  sorry

end quadratic_transform_l210_210985


namespace angle_B_in_triangle_l210_210876

theorem angle_B_in_triangle
  (a b c : ℝ) 
  (condition : a^2 + c^2 - b^2 = sqrt 3 * a * c) :
  ∠B = π / 6 :=
by
  sorry

end angle_B_in_triangle_l210_210876


namespace problem_A_problem_B_problem_D_l210_210272

-- Option A: Prove that vectors (1, 2) and (3, 1) can be a basis.
theorem problem_A (a b : ℝ × ℝ) (h_a : a = (1, 2)) (h_b : b = (3, 1)) :
  ∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ a ≠ b ∧ a ≠ y•b ∧ b ≠ x•a := sorry

-- Option B: Prove the coordinates of vertex D in a parallelogram ABCD.
theorem problem_B (A B C D : ℝ × ℝ) 
  (h_A : A = (5, -1)) (h_B : B = (-1, 7)) (h_C : C = (1, 2)) (h_D : D = (7, -6)) :
  D = (7, -6) := sorry

-- Option D: Prove the projection of (1, 1) onto (1, 1) is (2, 2).
theorem problem_D (a b : ℝ × ℝ) (h_a : a = (1, 1)) (h_b_norm : ∥b∥ = 4) 
  (h_angle : some_angle = π/4):
  ∃ proj_b : ℝ × ℝ, proj_b = (2, 2) := sorry

end problem_A_problem_B_problem_D_l210_210272


namespace correct_statement_l210_210013

universe u
variable (α : Type u)

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5}

noncomputable def complement_U_M : Set ℕ := {1, 3}

noncomputable def M : Set ℕ := U α \ complement_U_M

theorem correct_statement : 2 ∈ M :=
by {
  sorry
}

end correct_statement_l210_210013


namespace sum_first_100_sum_51_to_100_l210_210715

noncomputable def sum_natural_numbers (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem sum_first_100 : sum_natural_numbers 100 = 5050 :=
  sorry

theorem sum_51_to_100 : sum_natural_numbers 100 - sum_natural_numbers 50 = 3775 :=
  sorry

end sum_first_100_sum_51_to_100_l210_210715


namespace problem_A_problem_B_problem_D_l210_210273

-- Option A: Prove that vectors (1, 2) and (3, 1) can be a basis.
theorem problem_A (a b : ℝ × ℝ) (h_a : a = (1, 2)) (h_b : b = (3, 1)) :
  ∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ a ≠ b ∧ a ≠ y•b ∧ b ≠ x•a := sorry

-- Option B: Prove the coordinates of vertex D in a parallelogram ABCD.
theorem problem_B (A B C D : ℝ × ℝ) 
  (h_A : A = (5, -1)) (h_B : B = (-1, 7)) (h_C : C = (1, 2)) (h_D : D = (7, -6)) :
  D = (7, -6) := sorry

-- Option D: Prove the projection of (1, 1) onto (1, 1) is (2, 2).
theorem problem_D (a b : ℝ × ℝ) (h_a : a = (1, 1)) (h_b_norm : ∥b∥ = 4) 
  (h_angle : some_angle = π/4):
  ∃ proj_b : ℝ × ℝ, proj_b = (2, 2) := sorry

end problem_A_problem_B_problem_D_l210_210273


namespace sum_of_roots_in_interval_l210_210164

variable (x : ℝ) (k : ℤ)

noncomputable def trigonometric_equation : Prop :=
  3 * cos ((4 * real.pi * x) / 5) + cos ((12 * real.pi * x) / 5) =
  2 * cos ((4 * real.pi * x) / 5) * (3 + (tan (real.pi * x / 5))^2 - 2 * tan (real.pi * x / 5))

theorem sum_of_roots_in_interval :
  (∃ k : ℤ → (x : ℝ → (-11 ≤ x ∧ x ≤ 19))) → 
  (trigonometric_equation x k) → 
  (∑ k in finset.range (14 - (-9) + 1), 5/8 + 5*k/4) = 112.5 :=
sorry

end sum_of_roots_in_interval_l210_210164


namespace henry_final_money_l210_210481

def initial_money : ℝ := 11.75
def received_from_relatives : ℝ := 18.50
def found_in_card : ℝ := 5.25
def spent_on_game : ℝ := 10.60
def donated_to_charity : ℝ := 3.15

theorem henry_final_money :
  initial_money + received_from_relatives + found_in_card - spent_on_game - donated_to_charity = 21.75 :=
by
  -- proof goes here
  sorry

end henry_final_money_l210_210481


namespace totalNumberOfCrayons_l210_210993

def numOrangeCrayons (numBoxes : ℕ) (crayonsPerBox : ℕ) : ℕ :=
  numBoxes * crayonsPerBox

def numBlueCrayons (numBoxes : ℕ) (crayonsPerBox : ℕ) : ℕ :=
  numBoxes * crayonsPerBox

def numRedCrayons (numBoxes : ℕ) (crayonsPerBox : ℕ) : ℕ :=
  numBoxes * crayonsPerBox

theorem totalNumberOfCrayons :
  numOrangeCrayons 6 8 + numBlueCrayons 7 5 + numRedCrayons 1 11 = 94 :=
by
  sorry

end totalNumberOfCrayons_l210_210993


namespace value_of_a_parity_of_f_monotonicity_and_inequality_l210_210538

noncomputable def f (x : ℝ) (a : ℝ) := (1 - a * x^2) / (1 + x^2)

theorem value_of_a (a : ℝ) : (∀ x : ℝ, x ≠ 0 → f x a + f (1/x) a = -1) → a = 2 :=
sorry

theorem parity_of_f (x : ℝ) : f x 2 = f (-x) 2 :=
sorry

theorem monotonicity_and_inequality (x : ℝ) :
  (∀ x : ℝ, (x ≥ 0 → f (x) 2  > f(x + 1) 2) ∧ (x < 0 → f (x) 2 < f(x + 1) 2)) ∧ 
  (∀ x : ℝ, x ≠ 1/2 → f(x) 2 + f(1 / (2*x-1)) 2 + 1 < 0 → x ∈ (Set.Ioo (1/3) (1/2) ∪ Set.Ioo (1/2) 1)) :=
sorry

end value_of_a_parity_of_f_monotonicity_and_inequality_l210_210538


namespace zeros_in_interval_l210_210919

noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x - Real.pi / 4)

theorem zeros_in_interval : 
  ∃ (s : Set ℝ) (n : ℕ), 
    s = {x | f x = 0} ∧ 
    s ∩ Icc (-Real.pi) (Real.pi) = {x | x = Real.pi / 12 + k * Real.pi / 3 ∧ k ∈ { -3, -2, -1, 0, 1, 2}}.to_finset ∧ 
    n = 6 := 
sorry

end zeros_in_interval_l210_210919


namespace number_opposite_to_1_is_6_l210_210700

def is_adjacent (i j : ℕ) : Prop :=
  (i ≠ j) ∧ (abs (i // 3 - j // 3) + abs (i % 3 - j % 3) = 1)

def correct_grid (grid : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 9 → ∃ i : ℕ, i < 9 ∧ grid i = n) ∧
  (grid 0 = 3 ∧ grid 2 = 5 ∧ grid 6 = 7 ∧ grid 8 = 9 ∧ -- corners adding up to 24
   ∀ i : ℕ, i < 8 → is_adjacent i (i + 1)) ∧
  grid 1 = 4 ∧ grid 3 = 1 ∧ grid 4 = 2 ∧ grid 5 = 6 ∧ grid 7 = 8

theorem number_opposite_to_1_is_6 (grid : ℕ → ℕ) :
  correct_grid grid → grid 5 = 6 :=
begin
  sorry
end

end number_opposite_to_1_is_6_l210_210700


namespace question_l210_210032

variable (U : Set ℕ) (M : Set ℕ)

theorem question :
  U = {1, 2, 3, 4, 5} →
  (U \ M = {1, 3}) →
  2 ∈ M :=
by
  intros
  sorry

end question_l210_210032


namespace problem_solution_l210_210008

universe u

variable (U : Set Nat) (M : Set Nat)
variable (complement_U_M : Set Nat)

axiom U_def : U = {1, 2, 3, 4, 5}
axiom complement_U_M_def : complement_U_M = {1, 3}
axiom M_def : M = U \ complement_U_M

theorem problem_solution : 2 ∈ M := by
  sorry

end problem_solution_l210_210008


namespace powers_of_two_l210_210410

theorem powers_of_two (n : ℕ) (h : ∀ n, ∃ m, (2^n - 1) ∣ (m^2 + 9)) : ∃ s, n = 2^s :=
sorry

end powers_of_two_l210_210410


namespace min_max_abs_poly_eq_zero_l210_210771

theorem min_max_abs_poly_eq_zero :
  ∃ y : ℝ, (∀ x : ℝ, 0 ≤ x → x ≤ 1 → |x^2 - x^3 * y| ≤ 0) :=
sorry

end min_max_abs_poly_eq_zero_l210_210771


namespace boxes_given_to_mom_l210_210959

theorem boxes_given_to_mom 
  (sophie_boxes : ℕ) 
  (donuts_per_box : ℕ) 
  (donuts_to_sister : ℕ) 
  (donuts_left_for_her : ℕ) 
  (H1 : sophie_boxes = 4) 
  (H2 : donuts_per_box = 12) 
  (H3 : donuts_to_sister = 6) 
  (H4 : donuts_left_for_her = 30)
  : sophie_boxes * donuts_per_box - donuts_to_sister - donuts_left_for_her = donuts_per_box := 
by
  sorry

end boxes_given_to_mom_l210_210959


namespace sum_quotient_reciprocal_eq_one_point_thirty_five_l210_210609

theorem sum_quotient_reciprocal_eq_one_point_thirty_five (x y : ℝ)
  (h1 : x + y = 45) (h2 : x * y = 500) : x / y + 1 / x + 1 / y = 1.35 := by
  -- Proof details would go here
  sorry

end sum_quotient_reciprocal_eq_one_point_thirty_five_l210_210609


namespace roses_problem_l210_210525

variable (R B C : ℕ)

theorem roses_problem
    (h1 : R = B + 10)
    (h2 : C = 10)
    (h3 : 16 - 6 = C)
    (h4 : B = R - C):
  R = B + 10 ∧ R - C = B := 
by 
  have hC: C = 10 := by linarith
  have hR: R = B + 10 := by linarith
  have hRC: R - C = B := by linarith
  exact ⟨hR, hRC⟩

end roses_problem_l210_210525


namespace smallest_possible_list_length_l210_210345

def is_median (l: List ℕ) (m: ℕ) : Prop :=
  let sorted_l := l.sort
  if sorted_l.length % 2 = 1
  then sorted_l[sorted_l.length / 2] = m
  else (sorted_l[(sorted_l.length / 2) - 1] + sorted_l[sorted_l.length / 2]) / 2 = m

def is_mode (l: List ℕ) (mode: ℕ) : Prop :=
  l.count mode = l.map (λ x, l.count x).maximum.getD 0

def is_mean (l: List ℕ) (mean: ℕ) : Prop :=
  l.sum / l.length = mean

theorem smallest_possible_list_length (l: List ℕ) :
  is_median l 8 ∧ is_mode l 9 ∧ is_mean l 10 → l.length = 6 :=
by
  sorry

end smallest_possible_list_length_l210_210345


namespace complex_abs_sqrt_five_l210_210320

open Complex

theorem complex_abs_sqrt_five : abs (2 + (-1 : ℂ) + 2 * (-I : ℂ)) = Real.sqrt 5 := 
by
  sorry

end complex_abs_sqrt_five_l210_210320


namespace correct_statement_l210_210000

open Set

variable (U : Set ℕ) (M : Set ℕ)
variables (hU : U = {1, 2, 3, 4, 5}) (hM : U \ M = {1, 3})

theorem correct_statement : 2 ∈ M :=
by
  sorry

end correct_statement_l210_210000


namespace alex_money_left_l210_210364

noncomputable def weekly_income := 500
def income_tax_rate := 0.10
def water_bill := 55
def tithe_rate := 0.10

theorem alex_money_left : (weekly_income - ((weekly_income * income_tax_rate) + (weekly_income * tithe_rate) + water_bill)) = 345 := 
by
  sorry

end alex_money_left_l210_210364


namespace parabola_hyperbola_focus_condition_l210_210796

theorem parabola_hyperbola_focus_condition (a : ℝ) :
  (y = a * x^2 ∧ y = 2) → (focus of parabola coincides with focus of hyperbola) → (a = -1/8 ∨ a = 1/8) :=
by
  sorry

end parabola_hyperbola_focus_condition_l210_210796


namespace part_a_part_b_l210_210350

-- Definition of a good pair (a, p)
def is_good_pair (a p : ℕ) : Prop :=
  (a^3 + p^3) % (a^2 - p^2) = 0 ∧ a > p

-- Part (a): Prove there exists a value of a such that (a, 19) is a good pair
theorem part_a : ∃ (a : ℕ), is_good_pair a 19 := sorry

-- Part (b): Prove the number of good pairs (a, p) where p is a prime less than 24 is 27
theorem part_b : (Finset.card (Finset.filter (λ p, Prime p ∧ p < 24) (Finset.range 24)) = 9) →
  (Finset.sum
    (Finset.filter (λ p, Prime p ∧ p < 24) (Finset.range 24))
    (λ p, Finset.card (Finset.filter (λ a, (is_good_pair a p)) (Finset.range (p + 1 + p^2)))) = 27) := sorry

end part_a_part_b_l210_210350


namespace second_hand_degrees_per_minute_l210_210551

theorem second_hand_degrees_per_minute (clock_gains_5_minutes_per_hour : true) :
  (360 / 60 = 6) := 
by
  sorry

end second_hand_degrees_per_minute_l210_210551


namespace max_parts_divided_by_three_planes_l210_210624

theorem max_parts_divided_by_three_planes (parts_0_plane parts_1_plane parts_2_planes parts_3_planes: ℕ)
  (h0 : parts_0_plane = 1)
  (h1 : parts_1_plane = 2)
  (h2 : parts_2_planes = 4)
  (h3 : parts_3_planes = 8) :
  parts_3_planes = 8 :=
by
  sorry

end max_parts_divided_by_three_planes_l210_210624


namespace domain_f_l210_210393

noncomputable def f (x : ℝ) : ℝ :=
  (1 / real.sqrt (real.log (5 - 2 * x))) + real.sqrt (real.exp x - 1)

theorem domain_f : set.Ico (0 : ℝ) 2 = 
  {x | (5 - 2 * x > 1) ∧ (real.exp x - 1 ≥ 0)} :=
by
  sorry

end domain_f_l210_210393


namespace option_A_option_B_option_D_l210_210270

-- Option A
theorem option_A (a b : ℤ × ℤ) (h₁ : a = (1, 2)) (h₂ : b = (3, 1)) : ¬ collinear a b :=
by sorry

-- Option B
theorem option_B (A B C D : ℤ × ℤ) 
  (h₁ : A = (5, -1)) (h₂ : B = (-1, 7)) (h₃ : C = (1, 2)) : 
  parallelogram A B C D → D = (7, -6) :=
by sorry

-- Option D
theorem option_D (a b : ℝ × ℝ) (h₁ : a = (1, 1)) 
  (h₂ : |b| = 4) (θ : ℝ) (h₃ : θ = π / 4) : 
  projection a b = (2, 2) :=
by sorry

end option_A_option_B_option_D_l210_210270


namespace deduce_pi_l210_210093

-- Definitions representing the conditions
def circumference := 38
def height := 11
def volume := 2112

-- Using the given volume formula
def volume_formula (C : ℝ) (h : ℝ) : ℝ := (1 / 12) * (C ^ 2) * h

-- Stating the problem to deduce the value of π
theorem deduce_pi (C : ℝ) (h : ℝ) (V : ℝ) (π : ℝ) (radius : ℝ) 
  (hC : C = 2 * π * radius) (hh : h = height) (hV : V = volume) : π = 3 :=
by
  sorry

end deduce_pi_l210_210093


namespace find_value_l210_210494

theorem find_value (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : (a - 2 * b)^2 = 25 :=
by
  sorry

end find_value_l210_210494


namespace sum_of_coefficients_l210_210991

theorem sum_of_coefficients (x y : ℝ) :
  (x = 1) → (y = 1) →
  let expr := (x^2 - 3 * x * y + 2 * y^2)^8 in
  sum_of_coefficients (expr) = 1 :=
by
  intros
  sorry

end sum_of_coefficients_l210_210991


namespace expression_value_l210_210442

theorem expression_value (a b c d m : ℚ) (h1 : a + b = 0) (h2 : a ≠ 0) (h3 : c * d = 1) (h4 : m = -5 ∨ m = 1) :
  |m| - (a / b) + ((a + b) / 2020) - (c * d) = 1 ∨ |m| - (a / b) + ((a + b) / 2020) - (c * d) = 5 :=
by sorry

end expression_value_l210_210442


namespace proof_2_in_M_l210_210034

def U : Set ℕ := {1, 2, 3, 4, 5}

def M : Set ℕ := { x | x ∈ U ∧ x ≠ 1 ∧ x ≠ 3 }

theorem proof_2_in_M : 2 ∈ M :=
by
  sorry

end proof_2_in_M_l210_210034


namespace subset_of_sqrt_11_in_sqrt_13_l210_210544

variable (x : ℝ)
def A : set ℝ := {x | x ≤ Real.sqrt 13}
def a : ℝ := Real.sqrt 11

theorem subset_of_sqrt_11_in_sqrt_13  : {a} ⊆ A :=
by
  sorry

end subset_of_sqrt_11_in_sqrt_13_l210_210544


namespace cos_beta_minus_alpha_l210_210428

-- define f(x)
def f (x : Real) : Real := 5 * sin (x - π / 6)

-- given conditions
variables (α β : Real)
variables (h0 : 0 < α)
variables (h1 : α < β)
variables (h2 : β < 2 * π)
variables (h3 : f α = 1)
variables (h4 : f β = 1)

-- the statement we want to prove
theorem cos_beta_minus_alpha : cos (β - α) = -23 / 25 :=
  by sorry

end cos_beta_minus_alpha_l210_210428


namespace graphs_symmetric_line_l210_210977

theorem graphs_symmetric_line (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) : 
  (symmetric_with_respect_to_line (fun x => a^(x + 1)) (fun x => log a (x + 1)) (fun x => x + 1)) :=
sorry

end graphs_symmetric_line_l210_210977


namespace average_speed_for_trip_l210_210559

-- Definitions of the conditions
def total_distance : ℝ := 250
def distance_first_segment : ℝ := 100
def speed_first_segment : ℝ := 20
def distance_second_segment : ℝ := 150
def speed_second_segment : ℝ := 15

-- Definition of the average speed calculation
def average_speed (total_distance : ℝ) (time_first_segment : ℝ) (time_second_segment : ℝ) : ℝ :=
  total_distance / (time_first_segment + time_second_segment)

-- Calculate the times for each segment
def time_first_segment := distance_first_segment / speed_first_segment
def time_second_segment := distance_second_segment / speed_second_segment

-- Statement of the problem
theorem average_speed_for_trip : average_speed total_distance time_first_segment time_second_segment = 16.67 := by
  sorry

end average_speed_for_trip_l210_210559


namespace find_n_l210_210138

-- Define the function to sum the digits of a natural number n
def digit_sum (n : ℕ) : ℕ := 
  -- This is a dummy implementation for now
  -- Normally, we would implement the sum of the digits of n
  sorry 

-- The main theorem that we want to prove
theorem find_n : ∃ (n : ℕ), digit_sum n + n = 2011 ∧ n = 1991 :=
by
  -- Proof steps would go here, but we're skipping those with sorry.
  sorry

end find_n_l210_210138


namespace max_value_ineq_l210_210912

variable (a b c : ℝ)

theorem max_value_ineq (h₁ : a + b + c = 3) (h₂ : a = 1) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  (∃ (m : ℝ), m = (ab / (a + b) + ac / (a + c) + bc / (b + c)) ∧ isMax m) :=
by
  have h :  a + b + c = 3 := h₁
  have ha' : a = 1 := h₂
  have h₃ : m = (ab / (a + b) + ac / (a + c) + bc / (b + c))
  have isMax : m = 3/2 := 
    -- This is where the detailed proof steps would go, but they are not included in the problem statement.
    sorry
  
  exact ⟨3/2, h₃, isMax⟩

end max_value_ineq_l210_210912


namespace num_even_three_digit_numbers_with_sum_of_tens_and_units_10_l210_210483

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def sum_of_tens_and_units_is_ten (n : ℕ) : Prop :=
  (n / 10 % 10) + (n % 10) = 10

theorem num_even_three_digit_numbers_with_sum_of_tens_and_units_10 : 
  ∃! (N : ℕ), (N = 36) ∧ 
               (∀ n : ℕ, is_three_digit n → is_even n → sum_of_tens_and_units_is_ten n →
                         n = 36) := 
sorry

end num_even_three_digit_numbers_with_sum_of_tens_and_units_10_l210_210483


namespace Jose_Raju_Work_Together_l210_210289

-- Definitions for the conditions
def JoseWorkRate : ℚ := 1 / 10
def RajuWorkRate : ℚ := 1 / 40
def CombinedWorkRate : ℚ := JoseWorkRate + RajuWorkRate

-- Theorem statement
theorem Jose_Raju_Work_Together :
  1 / CombinedWorkRate = 8 := by
    sorry

end Jose_Raju_Work_Together_l210_210289


namespace opposite_of_five_is_neg_five_l210_210212

theorem opposite_of_five_is_neg_five :
  ∃ (x : ℤ), (5 + x = 0) ∧ x = -5 :=
by
  use -5
  split
  · simp
  · rfl

end opposite_of_five_is_neg_five_l210_210212


namespace first_year_with_sum_of_digits_15_after_2010_l210_210634

theorem first_year_with_sum_of_digits_15_after_2010 : 
  ∃ year : ℕ, year > 2010 ∧ (∑ d in (year.digits 10), d) = 15 ∧ 
  ∀ y : ℕ, y > 2010 → (∑ d in (y.digits 10), d) = 15 → year ≤ y :=
begin
  use 2039,
  split,
  { sorry }, -- proof that 2039 > 2010
  split,
  { sorry }, -- proof that sum of digits of 2039 is 15
  { sorry } -- proof that there's no earlier year with sum of digits 15
end

end first_year_with_sum_of_digits_15_after_2010_l210_210634


namespace triangle_angles_l210_210150

theorem triangle_angles (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = 45) : B = 90 ∧ C = 45 :=
sorry

end triangle_angles_l210_210150


namespace negative_solutions_iff_l210_210047

theorem negative_solutions_iff (m x y : ℝ) (h1 : x - y = 2 * m + 7) (h2 : x + y = 4 * m - 3) :
  (x < 0 ∧ y < 0) ↔ m < -2 / 3 :=
by
  sorry

end negative_solutions_iff_l210_210047


namespace rin_craters_difference_l210_210735

theorem rin_craters_difference (d da r : ℕ) (h1 : d = 35) (h2 : da = d - 10) (h3 : r = 75) :
  r - (d + da) = 15 :=
by
  sorry

end rin_craters_difference_l210_210735


namespace sophia_investment_amount_correct_l210_210171

-- Define the known conditions
def principal : ℝ := 2500
def rate : ℝ := 0.06
def time : ℕ := 21

-- Define the formula for compounded yearly interest
def compounded_amount (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r)^t

-- State the theorem we need to prove
theorem sophia_investment_amount_correct :
  compounded_amount principal rate time ≈ 8281.03 :=
by
  -- Proof would go here, but we substitute it with sorry to focus on the statement
  sorry

end sophia_investment_amount_correct_l210_210171


namespace repeating_decimal_fraction_sum_l210_210206

/-- The repeating decimal 3.171717... can be written as a fraction. When reduced to lowest
terms, the sum of the numerator and denominator of this fraction is 413. -/
theorem repeating_decimal_fraction_sum :
  let y := 3.17171717 -- The repeating decimal
  let frac_num := 314
  let frac_den := 99
  let sum := frac_num + frac_den
  y = frac_num / frac_den ∧ sum = 413 := by
  sorry

end repeating_decimal_fraction_sum_l210_210206


namespace shaded_area_correct_l210_210353

-- Define the side length of the hexagon
def side_length : ℝ := 4

-- Define the radius of each quarter-circle
def radius : ℝ := 2

-- Define the area of the hexagon (A_hex)
def hexagon_area : ℝ := (3 * real.sqrt 3 / 2) * side_length^2

-- Define the area of one quarter-circle (A_qcircle)
def quarter_circle_area : ℝ := (1 / 4) * real.pi * radius^2

-- Define the total area of the 12 quarter-circles (A_qcircles)
def total_quarter_circles_area : ℝ := 12 * quarter_circle_area

-- Define the area of the region inside the hexagon but outside the quarter-circles (A_shaded)
def shaded_area : ℝ := hexagon_area - total_quarter_circles_area

-- Prove that the shaded area is equal to 48√3 - 12π
theorem shaded_area_correct : shaded_area = 48 * real.sqrt 3 - 12 * real.pi := by
  sorry

end shaded_area_correct_l210_210353


namespace fifty_eq_x_plus_y_pairs_l210_210833

theorem fifty_eq_x_plus_y_pairs : {p : ℕ × ℕ // p.1 + p.2 = 50 ∧ p.1 > 0 ∧ p.2 > 0}.card = 49 := 
sorry

end fifty_eq_x_plus_y_pairs_l210_210833


namespace mr_william_tax_l210_210770

def total_tax_collected : ℝ := 3840
def williams_percentage : ℝ := 0.1388888888888889
def expected_tax_paid : ℝ := 533.33

theorem mr_william_tax :
  williams_percentage * total_tax_collected ≈ expected_tax_paid :=
by
  sorry

end mr_william_tax_l210_210770


namespace frosting_needed_l210_210563

-- Definitions directly from the problem conditions
def cans_frosting_per_layer_cake := 1
def cans_frosting_per_single_cake := 0.5
def cans_frosting_per_dozen_cupcakes := 0.5
def cans_frosting_per_pan_brownies := 0.5

-- Quantities needed
def layer_cakes_needed := 3
def dozen_cupcakes_needed := 6
def single_cakes_needed := 12
def pans_brownies_needed := 18

-- Proposition to prove
theorem frosting_needed : 
  (cans_frosting_per_layer_cake * layer_cakes_needed) + 
  (cans_frosting_per_dozen_cupcakes * dozen_cupcakes_needed) + 
  (cans_frosting_per_single_cake * single_cakes_needed) + 
  (cans_frosting_per_pan_brownies * pans_brownies_needed) = 21 := 
by 
  sorry

end frosting_needed_l210_210563


namespace parallelepiped_volume_l210_210599

-- Conditions
variables (a b c : ℝ)
variables (hab_perp : is_perpendicular (edge_length a) (edge_length b))
variables (hac_angle : angle (edge_length c) (edge_length a) = real.pi / 3)
variables (hbc_angle : angle (edge_length c) (edge_length b) = real.pi / 3)

-- Theorem stating the volume of the parallelepiped
theorem parallelepiped_volume : volume (parallelepiped a b c hab_perp hac_angle hbc_angle) = (a * b * c * real.sqrt 2) / 2 :=
sorry

end parallelepiped_volume_l210_210599


namespace question_l210_210033

variable (U : Set ℕ) (M : Set ℕ)

theorem question :
  U = {1, 2, 3, 4, 5} →
  (U \ M = {1, 3}) →
  2 ∈ M :=
by
  intros
  sorry

end question_l210_210033


namespace function_zeros_count_l210_210916

theorem function_zeros_count :
  ∀ (f : ℝ → ℝ) (ω : ℝ),
    (∀ x, f(x) = Real.sin(ω * x - Real.pi / 4)) → 
    ω > 0 →
    (∀ x1 x2, |f x1 - f x2| = 2 → |x1 - x2| = Real.pi / 3) →
    (∃ n, set.finite.to_finset ({x : ℝ | f x = 0} ∩ set.Icc (-Real.pi) Real.pi) = finset.range n.succ ∧
      finset.card (set.finite.to_finset ({x : ℝ | f x = 0} ∩ set.Icc (-Real.pi) Real.pi)) = 6) :=
begin
  intros f ω hf hω hcond,
  sorry
end

end function_zeros_count_l210_210916


namespace g_neither_even_nor_odd_l210_210103

def g (x : ℝ) : ℝ := ⌊2 * x⌋ + 1/3

theorem g_neither_even_nor_odd : (∀ x, g x ≠ g (-x)) ∧ (∀ x, g x ≠ -g (-x)) :=
by
  sorry

end g_neither_even_nor_odd_l210_210103


namespace distance_from_origin_l210_210447

open Complex

-- Given conditions
def c : ℂ := 3 / (2 - I) ^ 2

-- The real part of the complex number
def c_re : ℝ := re c

-- The imaginary part of the complex number
def c_im : ℝ := im c

-- Lean 4 statement to prove the distance from the origin to the point (c_re, c_im) is 3 / 5
theorem distance_from_origin : real.sqrt (c_re ^ 2 + c_im ^ 2) = 3 / 5 :=
by
  sorry

end distance_from_origin_l210_210447


namespace scientific_notation_to_decimal_l210_210588

theorem scientific_notation_to_decimal :
  5.2 * 10^(-5) = 0.000052 :=
sorry

end scientific_notation_to_decimal_l210_210588


namespace charitable_woman_l210_210339

theorem charitable_woman (initial_pennies : ℕ) 
  (farmer_share : ℕ) (beggar_share : ℕ) (boy_share : ℕ) (left_pennies : ℕ) 
  (h1 : initial_pennies = 42)
  (h2 : farmer_share = (initial_pennies / 2 + 1))
  (h3 : beggar_share = ((initial_pennies - farmer_share) / 2 + 2))
  (h4 : boy_share = ((initial_pennies - farmer_share - beggar_share) / 2 + 3))
  (h5 : left_pennies = initial_pennies - farmer_share - beggar_share - boy_share) : 
  left_pennies = 1 :=
by
  sorry

end charitable_woman_l210_210339


namespace opposite_of_five_l210_210209

theorem opposite_of_five : ∃ y : ℤ, 5 + y = 0 ∧ y = -5 := by
  use -5
  constructor
  . exact rfl
  . sorry

end opposite_of_five_l210_210209


namespace probability_top_face_odd_dots_l210_210941

noncomputable def prob_odd_top_face_on_d12_die : ℚ :=
  let faces := 12
  let total_dots := (faces * (faces + 1)) / 2
  let total_ways_two_dots := (total_dots * (total_dots - 1)) / 2
  let even_faces := [2, 4, 6, 8, 10, 12]
  let prob_one_dot_removed t := (t : ℚ) * 2 / total_ways_two_dots
  let total_prob_odd := even_faces.sum (prob_one_dot_removed)
  total_prob_odd / faces

theorem probability_top_face_odd_dots : 
  prob_odd_top_face_on_d12_die = 7 / 3003 :=
by
  sorry

end probability_top_face_odd_dots_l210_210941


namespace mark_remaining_money_l210_210549

theorem mark_remaining_money 
  (initial_money : ℕ) (num_books : ℕ) (cost_per_book : ℕ) (total_cost : ℕ) (remaining_money : ℕ) 
  (H1 : initial_money = 85)
  (H2 : num_books = 10)
  (H3 : cost_per_book = 5)
  (H4 : total_cost = num_books * cost_per_book)
  (H5 : remaining_money = initial_money - total_cost) : 
  remaining_money = 35 := 
by
  sorry

end mark_remaining_money_l210_210549


namespace evaluate_expression_l210_210769

theorem evaluate_expression : (3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3)) = 3 := by
  sorry

end evaluate_expression_l210_210769


namespace scientific_notation_of_12_6_l210_210967

noncomputable def scientific_notation (x : ℝ) : ℝ × ℤ :=
let a := x / (10 ^ (Int.ofNat (Nat.log10 (Nat.abs (Int.natAbs (Int.ofNat (Nat.introDigits (Nat.digits 10 (Nat.abs (Int.natAbs (Int.ofNat (Nat.abs x))))))))))))
let n := Int.ofNat (Nat.log10 (Nat.abs (Int.natAbs (Int.ofNat (Nat.abs (x)))))
in (a, n)

theorem scientific_notation_of_12_6 : scientific_notation 12.6 = (1.26, 5) := 
by 
  -- This is where the proof would go
  sorry

end scientific_notation_of_12_6_l210_210967


namespace sum_geometric_series_l210_210192

def infinite_geometric_series_sum (a r : ℝ) (h : |r| < 1) : ℝ := a / (1 - r)
def odd_powers_sum (a r : ℝ) (h : |r| < 1) : ℝ := (a * r) / (1 - r^2)

theorem sum_geometric_series (a r : ℝ) (h : |r| < 1)
  (h1 : infinite_geometric_series_sum a r h = 7)
  (h2 : odd_powers_sum a r h = 3) :
  a + r = 5 / 2 :=
sorry

end sum_geometric_series_l210_210192


namespace paths_count_l210_210332

/-- 
  Statement of the problem: 
  Prove that the number of 16-step paths from (-4,-4) to (4,4) that stay outside 
  or on the boundary of the square -2 ≤ x ≤ 2, -2 ≤ y ≤ 2 at each step, while 
  adhering to the movement conditions, is 1698. 
--/
theorem paths_count :
  let point := ℕ × ℕ
  let boundary_condition (p : (ℤ × ℤ)) : Prop := 
    -2 ≤ p.1 ∧ p.1 ≤ 2 ∧ -2 ≤ p.2 ∧ p.2 ≤ 2
  let move_up (p : (ℤ × ℤ)) : (ℤ × ℤ) := (p.1, p.2 + 1)
  let move_right (p : (ℤ × ℤ)) : (ℤ × ℤ) := (p.1 + 1, p.2)
  let initial := (-4, -4)
  let final := (4, 4)
  let steps := 16

  ∃ (paths : list (list (ℤ × ℤ))), 
    (∀ path ∈ paths, 
      (path.length = steps + 1 ∧ 
      (path.head = initial) ∧ 
      (path.last = final) ∧ 
      (∀ p ∈ path, ¬ boundary_condition p) ∧ 
      (∀ i < steps,
        (path[i + 1] = move_up (path[i]) ∨ path[i + 1] = move_right (path[i]))
      )
    )) ∧ 
    (paths.length = 1698)
    :=
sorry

end paths_count_l210_210332


namespace total_stocking_stuffers_total_stocking_stuffers_hannah_buys_l210_210051

def candy_canes_per_kid : ℕ := 4
def beanie_babies_per_kid : ℕ := 2
def books_per_kid : ℕ := 1
def kids : ℕ := 3

theorem total_stocking_stuffers : candy_canes_per_kid + beanie_babies_per_kid + books_per_kid = 7 :=
by { 
  -- by trusted computation
  sorry
}

theorem total_stocking_stuffers_hannah_buys : 3 * (candy_canes_per_kid + beanie_babies_per_kid + books_per_kid) = 21 :=
by {
  have h : candy_canes_per_kid + beanie_babies_per_kid + books_per_kid = 7 := total_stocking_stuffers,
  rw h,
  norm_num,
}

end total_stocking_stuffers_total_stocking_stuffers_hannah_buys_l210_210051


namespace M_on_AD_iff_BFEC_concyclic_l210_210576

variables (Point : Type) [IncidenceGeometry Point]

-- Definitions of the relevant points
variables (A D M B F E C : Point)

-- Definitions of conditions
variables (circle1 : Circle Point) (circle2 : Circle Point)

-- Condition: Points D, M, F, B are concyclic with circle1.
axiom D_M_FB_concyclic : Circle.circumcircle D M F B = circle1

-- Condition: Points M, A, E, F are concyclic with circle2.
axiom M_A_EF_concyclic : Circle.circumcircle M A E F = circle2

-- The main theorem statement
theorem M_on_AD_iff_BFEC_concyclic :
  (Line (A D)).contains M ↔ Circle.circumcircle B F E C ≠ ∅ :=
sorry

end M_on_AD_iff_BFEC_concyclic_l210_210576


namespace tan_period_intersection_distance_l210_210199

theorem tan_period_intersection_distance (ω : ℝ) (hω_pos : ω > 0) 
  (h_intersect_dist : ∀ n : ℤ, ∃ x : ℝ, y = 3 ∧ f(x) = 3 ∧ (xₙ₊₁ - xₙ) = (π / 4)) :
  f(π / 12) = √3 := by
  sorry

end tan_period_intersection_distance_l210_210199


namespace find_c_l210_210600

variable {a b c : ℝ} 
variable (h_perpendicular : (a / 3) * (-3 / b) = -1)
variable (h_intersect1 : 2 * a + 9 = c)
variable (h_intersect2 : 6 - 3 * b = -c)
variable (h_ab_equal : a = b)

theorem find_c : c = 39 := 
by
  sorry

end find_c_l210_210600


namespace correct_statement_l210_210017

universe u
variable (α : Type u)

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5}

noncomputable def complement_U_M : Set ℕ := {1, 3}

noncomputable def M : Set ℕ := U α \ complement_U_M

theorem correct_statement : 2 ∈ M :=
by {
  sorry
}

end correct_statement_l210_210017


namespace opposite_of_5_is_neg5_l210_210220

def opposite (n x : ℤ) := n + x = 0

theorem opposite_of_5_is_neg5 : opposite 5 (-5) :=
by
  sorry

end opposite_of_5_is_neg5_l210_210220


namespace solve_equation_l210_210744

theorem solve_equation (x : ℝ) :
  3 * x + 6 = abs (-20 + x^2) →
  x = (3 + Real.sqrt 113) / 2 ∨ x = (3 - Real.sqrt 113) / 2 :=
by
  sorry

end solve_equation_l210_210744


namespace cube_paint_problem_l210_210677

-- Definitions and conditions from the problem
def larger_cube_side_length : ℕ := 4
def smaller_cube_side_length : ℕ := 1
def total_smaller_cubes : ℕ := larger_cube_side_length ^ 3
def painted_faces_count : ℕ := 3
def corner_cubes_with_painted_faces : ℕ := 8

-- The theorem to prove the number of 1-inch cubes with blue paint on at least three faces.
theorem cube_paint_problem : 
  let one_inch_cubes_with_three_painted_faces := corner_cubes_with_painted_faces in
  one_inch_cubes_with_three_painted_faces = 8 :=
sorry

end cube_paint_problem_l210_210677


namespace max_parts_divided_by_three_planes_l210_210623

theorem max_parts_divided_by_three_planes (parts_0_plane parts_1_plane parts_2_planes parts_3_planes: ℕ)
  (h0 : parts_0_plane = 1)
  (h1 : parts_1_plane = 2)
  (h2 : parts_2_planes = 4)
  (h3 : parts_3_planes = 8) :
  parts_3_planes = 8 :=
by
  sorry

end max_parts_divided_by_three_planes_l210_210623


namespace shaded_area_correct_l210_210341

noncomputable def shaded_area_outside_smaller_circle_and_inside_larger_circles
    (r_small r_large : ℝ) (cos_inv : ℝ) (sqrt_21 : ℝ) (pi : ℝ)
    (h1 : r_small = 2) (h2 : r_large = 5)
    (h3 : cos_inv = Real.arccos (2 / 5))
    (h4 : sqrt_21 = Real.sqrt 21)
    (h5 : pi = Real.pi) :
    ℝ :=
100 * cos_inv - (12 * sqrt_21) / 5 - 4 * pi

theorem shaded_area_correct : 
    shaded_area_outside_smaller_circle_and_inside_larger_circles 2 5 (Real.arccos (2 / 5)) (Real.sqrt 21) Real.pi = 
    100 * Real.arccos (2 / 5) - (12 * Real.sqrt 21) / 5 - 4 * Real.pi :=
by 
  -- Proof goes here 
  sorry

end shaded_area_correct_l210_210341


namespace maximum_a_pos_integer_greatest_possible_value_of_a_l210_210589

theorem maximum_a_pos_integer (a : ℕ) (h : ∃ x : ℤ, x^2 + (a * x : ℤ) = -20) : a ≤ 21 :=
by
  sorry

theorem greatest_possible_value_of_a : ∃ (a : ℕ), (∀ b : ℕ, (∃ x : ℤ, x^2 + (b * x : ℤ) = -20) → b ≤ 21) ∧ 21 = a :=
by
  sorry

end maximum_a_pos_integer_greatest_possible_value_of_a_l210_210589


namespace base_number_theorem_l210_210867

def base_number_factorial : ℕ :=
  sorry

theorem base_number_theorem : (∃ b : ℕ, (∀ n : ℕ, b^n ∣ fact 16 → n ≤ 6)) → (base_number_factorial = 2) :=
by
  intros h
  sorry

end base_number_theorem_l210_210867


namespace tangent_line_range_m_l210_210678

noncomputable def f (x : ℝ) : ℝ := x^2 + x
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := (1/3) * x^3 - 2 * x + m

theorem tangent_line (x y : ℝ) : x = 1 → y = 2 → (3*x - y - 1 = 0) :=
by sorry

theorem range_m (f g : ℝ → ℝ) (m : ℝ) :
  (∀ x ∈ Icc (-4 : ℝ) 4, f x ≥ g x) →
  (m ≤ -5/3) :=
by sorry

end tangent_line_range_m_l210_210678


namespace permutation_sum_bound_l210_210527

theorem permutation_sum_bound
  (n : ℕ) (x : Fin n → ℝ) (vec_alpha : Fin n → ℝ × ℝ)
  (A : ℝ) (B : ℝ)
  (h1 : n > 1)
  (h2 : A = |∑ i, x i|)
  (h3 : A ≠ 0)
  (h4 : B = Finset.sup (Finset.filter (λ ⟨i, j⟩, i < j)
    (Finset.product Finset.univ Finset.univ)) (λ ⟨i, j⟩, |x j - x i|))
  (h5 : B ≠ 0) :
  ∃ (k : Fin n → Fin n), |∑ i, (x (k i) • vec_alpha i)| ≥ (A * B) / (2 * A + B) * (Finset.sup (Finset.univ) (λ i, |vec_alpha i|)) := 
sorry

end permutation_sum_bound_l210_210527


namespace probability_of_mixed_selection_distribution_of_X_expected_value_of_X_l210_210747

namespace ZongziProblem

open ProbabilityTheory

def total_zongzi : ℕ := 10
def red_bean_zongzi : ℕ := 2
def plain_zongzi : ℕ := 8
def selected_zongzi : ℕ := 3

-- Question 1
theorem probability_of_mixed_selection : 
  let C (n k : ℕ) := nat.choose n k in
  proof_problem := (C(2, 1) * C(8, 2) + C(2, 2) * C(8, 1)) / C(10, 3) = 8 / 15 :=
sorry

-- Question 2
def X : Type := {x : ℕ // x ≤ 2}  -- Representing the number of red bean zongzi selected

def P (x : ℕ) : Rational :=
  if x = 0 then 7 / 15 else
  if x = 1 then 7 / 15 else
  if x = 2 then 1 / 15 else
  0

theorem distribution_of_X : 
  ( ∑ x ∈ {0, 1, 2}, P(x) = 1 ) ∧
  ( P(0) = 7 / 15 ) ∧
  ( P(1) = 7 / 15 ) ∧
  ( P(2) = 1 / 15 ) :=
sorry

theorem expected_value_of_X :
  ∑ x ∈ {0, 1, 2}, x * P(x) = 3 / 5 :=
sorry

end ZongziProblem

end probability_of_mixed_selection_distribution_of_X_expected_value_of_X_l210_210747


namespace log_sum_eq_five_implies_y_l210_210163

theorem log_sum_eq_five_implies_y (y : ℝ) (h : log 3 y + log 9 y = 5) : y = 3^(10/3) :=
sorry

end log_sum_eq_five_implies_y_l210_210163


namespace opposite_of_five_l210_210215

theorem opposite_of_five : -5 = -5 :=
by
sorry

end opposite_of_five_l210_210215


namespace intersection_of_A_and_B_l210_210845

noncomputable def A : Set ℝ := {-2, -1, 0, 1}
noncomputable def B : Set ℝ := {x | x^2 - 1 ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := 
by
  sorry

end intersection_of_A_and_B_l210_210845


namespace triplet_divisibility_cond_l210_210738

theorem triplet_divisibility_cond (a b c : ℤ) (hac : a ≥ 2) (hbc : b ≥ 2) (hcc : c ≥ 2) :
  (a - 1) * (b - 1) * (c - 1) ∣ (a * b * c - 1) ↔ 
  (a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 3 ∧ b = 15 ∧ c = 5) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 2 ∧ b = 8 ∧ c = 4) ∨ 
  (a = 2 ∧ b = 2 ∧ c = 4) ∨ (a = 2 ∧ b = 4 ∧ c = 2) ∨ 
  (a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

end triplet_divisibility_cond_l210_210738


namespace determinant_of_A_is_one_l210_210768

-- Define the matrix
def A (α β γ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![Real.cos α * Real.cos β, Real.cos α * Real.sin β * Real.cos γ, -Real.sin α],
    ![-Real.sin β * Real.cos γ, Real.cos β, Real.sin γ],
    ![Real.sin α * Real.cos β, Real.sin α * Real.sin β * Real.cos γ, Real.cos α]
  ]

-- Define the theorem to prove that the determinant is 1
theorem determinant_of_A_is_one (α β γ : ℝ) : Matrix.det (A α β γ) = 1 := 
by 
  sorry

end determinant_of_A_is_one_l210_210768


namespace midpoint_equality_l210_210515

-- Defining points and their properties
variables {A B C M : Point} -- Points in the coordinate plane
variables [midpoint_condition : Man midpoint_properties : B = C = (2 * M)]

-- Given point A outside the line segment BC
axiom A_outside_BC : ¬ collinear A B C

-- The theorem to be proved
theorem midpoint_equality
  (hm : is_midpoint M B C) :
  (dist A B) ^ 2 + (dist A C) ^ 2 = 
  2 * ((dist A M) ^ 2 + (dist B M) ^ 2) :=
sorry

end midpoint_equality_l210_210515


namespace vehicle_drive_analysis_l210_210696

theorem vehicle_drive_analysis (
  miles_mon : ℕ := 14,
  mpg_mon : ℕ := 24,
  price_mon : ℝ := 2.50,
  miles_tue : ℕ := 17,
  mpg_tue : ℕ := 21,
  price_tue : ℝ := 2.75,
  miles_wed : ℕ := 19,
  mpg_wed : ℕ := 19,
  weather_increase_wed : ℝ := 0.10,
  price_wed : ℝ := 2.65,
  miles_thu : ℕ := 10,
  mpg_thu : ℕ := 26,
  traffic_increase_thu : ℝ := 0.15,
  price_thu : ℝ := 2.80,
  miles_fri : ℕ := 22,
  mpg_fri : ℕ := 18,
  price_fri : ℝ := 2.60) :
  let average_miles : ℝ := (miles_mon + miles_tue + miles_wed + miles_thu + miles_fri) / 5,
      total_fuel : ℝ := (miles_mon / mpg_mon.toReal) 
                    + (miles_tue / mpg_tue.toReal) 
                    + (miles_wed / mpg_wed.toReal * (1 + weather_increase_wed))
                    + (miles_thu / mpg_thu.toReal * (1 + traffic_increase_thu))
                    + (miles_fri / mpg_fri.toReal),
      total_cost : ℝ := (miles_mon / mpg_mon.toReal * price_mon) 
                    + (miles_tue / mpg_tue.toReal * price_tue) 
                    + (miles_wed / mpg_wed.toReal * (1 + weather_increase_wed) * price_wed)
                    + (miles_thu / mpg_thu.toReal * (1 + traffic_increase_thu) * price_thu)
                    + (miles_fri / mpg_fri.toReal * price_fri) in
  average_miles = 16.4 ∧
  total_fuel ≈ 4.1573 ∧
  total_cost ≈ 11.0155 :=
by
  sorry

end vehicle_drive_analysis_l210_210696


namespace sum_of_digits_card_T_set_l210_210911

def sum_of_digits (n : ℕ) : ℕ := (nat.digits 10 n).sum

def T_set : finset ℕ := 
  finset.filter (λ n, sum_of_digits n = 10) 
  (finset.range 1000000).filter (λ n, 100000 ≤ n)
  
theorem sum_of_digits_card_T_set : sum_of_digits (T_set.card) = 9 :=
sorry

end sum_of_digits_card_T_set_l210_210911


namespace smallest_five_digit_perfect_square_and_cube_l210_210266

theorem smallest_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 := 
by
  sorry

end smallest_five_digit_perfect_square_and_cube_l210_210266


namespace find_f2_l210_210815

-- Define f as an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define g based on f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  f x + 9

-- Given conditions
variables (f : ℝ → ℝ) (h_odd : odd_function f)
variable (h_g_neg2 : g f (-2) = 3)

-- Theorem statement
theorem find_f2 : f 2 = 6 :=
by
  sorry

end find_f2_l210_210815


namespace average_price_of_pen_l210_210331

theorem average_price_of_pen :
  ∀ (P Q : ℝ), 
  50 * P + 120 * Q = 1200 ∧ Q = 1.75 → P = 19.8 :=
by
  intros P Q
  assume h : (50 * P + 120 * Q = 1200 ∧ Q = 1.75)
  sorry

end average_price_of_pen_l210_210331


namespace arcsin_one_eq_pi_div_two_l210_210724

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = Real.pi / 2 := 
by
  sorry

end arcsin_one_eq_pi_div_two_l210_210724


namespace correct_statement_l210_210016

universe u
variable (α : Type u)

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5}

noncomputable def complement_U_M : Set ℕ := {1, 3}

noncomputable def M : Set ℕ := U α \ complement_U_M

theorem correct_statement : 2 ∈ M :=
by {
  sorry
}

end correct_statement_l210_210016


namespace prime_remainder_30_l210_210157

theorem prime_remainder_30 (p : ℕ) (hp : Nat.Prime p) (hgt : p > 30) (hmod2 : p % 2 ≠ 0) 
(hmod3 : p % 3 ≠ 0) (hmod5 : p % 5 ≠ 0) : 
  ∃ (r : ℕ), r < 30 ∧ (p % 30 = r) ∧ (r = 1 ∨ Nat.Prime r) := 
by
  sorry

end prime_remainder_30_l210_210157


namespace complete_the_square_l210_210384

theorem complete_the_square (x : ℝ) :
  (∃ c d : ℝ, (x - c)^2 = d) ∧
  x^2 - 6 * x - 16 = 0 →
  ∃ d : ℝ, d = 25 :=
by
  intro h
  obtain ⟨c, d, h1⟩ := h.1
  have eq1 : x^2 - 6 * x - 16 = (x - 3)^2 - 25
  { calc
      x^2 - 6 * x - 16 = x^2 - 6 * x + 9 - 9 - 16 : by rw [(sub_add_eq_sub_sub _ _).symm, add_neg_eq_sub]
                  ... = (x - 3)^2 - 25     : by norm_num }
  rw ← eq1 at h
  exact exists.intro 25 by sorry

end complete_the_square_l210_210384


namespace g_neither_even_nor_odd_l210_210107

def g (x : ℝ) : ℝ := ⌊2 * x⌋ + 1 / 3

theorem g_neither_even_nor_odd : ¬ (∀ x, g (-x) = g x) ∧ ¬ (∀ x, g (-x) = -g x) :=
by
  sorry

end g_neither_even_nor_odd_l210_210107


namespace unique_solution_of_functional_eq_l210_210773

theorem unique_solution_of_functional_eq (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(x + f(y)) + f(xy) = y * f(x) + f(y) + f(f(x))) →
  (∀ x : ℝ, f(x) = x) :=
by
  sorry

end unique_solution_of_functional_eq_l210_210773


namespace greatest_number_of_dimes_l210_210951

noncomputable def maxDimes (total_value : ℝ) : ℕ :=
let d := total_value / 0.11 in
Int.floor d

theorem greatest_number_of_dimes (total_value : ℝ) (h_total_value : total_value = 3.50) :
  maxDimes total_value = 31 := by
  unfold maxDimes
  rw [h_total_value]
  norm_num
  sorry

end greatest_number_of_dimes_l210_210951


namespace sum_of_real_roots_l210_210727

noncomputable def P (x : ℝ) : ℝ := 2 * x^6 - 3 * x^5 + 3 * x^4 + x^3 - 3 * x^2 + 3 * x - 1

theorem sum_of_real_roots :
  let roots := {x : ℝ | P x = 0} in
  (∑ x in roots, x) = -1 / 2 :=
by
  sorry

end sum_of_real_roots_l210_210727


namespace volume_of_triangle_revolution_l210_210610

theorem volume_of_triangle_revolution
  (a b : ℝ) (α : ℝ) :
  volume_of_solid_of_revolution a b α = (π / 3) * a * b * (a + b) * sin α * cos (α / 2) :=
sorry

end volume_of_triangle_revolution_l210_210610


namespace rhombus_perimeter_l210_210973

theorem rhombus_perimeter (d1 d2 : ℝ) (h_d1 : d1 = 18) (h_d2 : d2 = 12) : 
  4 * real.sqrt (d1 * d1 / 4 + d2 * d2 / 4) = 12 * real.sqrt 13 :=
by
  sorry

end rhombus_perimeter_l210_210973


namespace trapezoid_area_correct_l210_210803

noncomputable def calculate_trapezoid_area : ℕ :=
  let parallel_side_1 := 6
  let parallel_side_2 := 12
  let leg := 5
  let radius := 5
  let height := radius
  let area := (1 / 2) * (parallel_side_1 + parallel_side_2) * height
  area

theorem trapezoid_area_correct :
  calculate_trapezoid_area = 45 :=
by {
  sorry
}

end trapezoid_area_correct_l210_210803


namespace population_growth_time_l210_210225

theorem population_growth_time :
  (∀ persons minutes : ℕ, persons = 90 → minutes = 30 → (1 : ℕ) = 20 : ℕ) :=
by
  assume persons minutes,
  assume h1 : persons = 90,
  assume h2 : minutes = 30,
  sorry

end population_growth_time_l210_210225


namespace number_of_valid_schedules_l210_210667

-- Define the subjects involved
inductive Subject
| Chinese 
| Mathematics 
| English 
| Physics 
| Chemistry 
| Biology

open Subject

-- Define the scheduling conditions
def valid_schedule (s : list Subject) : Prop :=
  (s.head = some Mathematics ∨ s.last = some Mathematics) ∧
  (∃ i, s.nth i = some Physics ∧ s.nth (i+1) = some Chemistry ∨ s.nth i = some Chemistry ∧ s.nth (i+1) = some Physics)

-- The main theorem statement
theorem number_of_valid_schedules : 
  {s : list Subject // valid_schedule s} → (∃ (n : ℕ), n = 96) :=
sorry

end number_of_valid_schedules_l210_210667


namespace find_initial_candies_l210_210626

-- Definitions for the conditions
def initial_candies (x : ℕ) : Prop :=
  (3 * x) % 4 = 0 ∧
  (x % 2) = 0 ∧
  ∃ (k : ℕ), 2 ≤ k ∧ k ≤ 6 ∧ (1 * x) / 2 - 20 - k = 4

-- Theorems we need to prove
theorem find_initial_candies (x : ℕ) (h : initial_candies x) : x = 52 ∨ x = 56 ∨ x = 60 :=
sorry

end find_initial_candies_l210_210626


namespace ball_bounce_height_l210_210335

theorem ball_bounce_height :
  ∃ (k : ℕ), 10 * (1 / 2) ^ k < 1 ∧ (∀ m < k, 10 * (1 / 2) ^ m ≥ 1) :=
sorry

end ball_bounce_height_l210_210335


namespace pats_bicycling_speed_l210_210557

theorem pats_bicycling_speed:
  ∀ (total_distance distance_ran speed_ran time_total time_bicycled : ℝ),
    total_distance = 20 →
    distance_ran = 14 →
    speed_ran = 8 →
    time_total = 1.95 →
    time_bicycled = 0.2 →
    (total_distance - distance_ran) / time_bicycled = 30 :=
by
  intros total_distance distance_ran speed_ran time_total time_bicycled
  intros h_total_distance h_distance_ran h_speed_ran h_time_total h_time_bicycled
  rw [h_total_distance, h_distance_ran, h_speed_ran, h_time_total, h_time_bicycled]
  sorry

end pats_bicycling_speed_l210_210557


namespace triangle_equilateral_l210_210205

theorem triangle_equilateral
  (A B C D E F O : Type)
  [is_triangle A B C]
  [is_median A D]
  [is_altitude B E]
  [is_bisector C F]
  [intersect_at A D B E C F O]
  (h : BO = CO) :
  is_equilateral A B C :=
sorry

end triangle_equilateral_l210_210205


namespace final_count_of_strawberry_plants_l210_210547

def initial_plants : ℕ := 3
def months : ℕ := 3
def factor : ℕ := 2
def giveaway_plants : ℕ := 4

theorem final_count_of_strawberry_plants : ℕ :=
  initial_plants * (factor ^ months) - giveaway_plants = 20 := sorry

end final_count_of_strawberry_plants_l210_210547


namespace president_vice_president_selection_l210_210244

theorem president_vice_president_selection (n : ℕ) (h : n = 4) : 
  (Finset.card (Finset.perm_2 (Finset.range n))) = 12 :=
by
  -- proof steps will go here
  sorry

end president_vice_president_selection_l210_210244


namespace isosceles_triangle_OAE_isosceles_at_A_l210_210539

open EuclideanGeometry

-- Definitions of the points A, B and circle Gamma
variables {A B O D E : Point}

-- Conditions
axiom circle_Γ (Γ : Circle) : A ∈ Γ ∧ B ∈ Γ ∧ Γ.center = O
axiom D_bisector (h : Bisection (angle ∠ O A B) D) : D ∈ Γ
axiom E_circumcircle (Ω : Circle) (h : A, B collinear ∧ Circumscribed ∆ O B D Ω) : ∃ E, O B D ∈ Ω ∧ A B ∩ Ω = E

-- Theorem
theorem isosceles_triangle_OAE_isosceles_at_A : IsoscelesTriangle ∆ O A E A :=
by
  -- Proof starts here
  sorry

end isosceles_triangle_OAE_isosceles_at_A_l210_210539


namespace find_a3_a6_a9_l210_210514

variable {α : Type} [AddGroup α] [HasSmul ℤ α]

structure ArithmeticSequence (α : Type) :=
(a : ℕ → α)
(d : α)

def sum_three (a : ℕ → α) (d : α) (i : ℕ) : α := a i + a (i + 3) + a (i + 6)

variable {seq : ArithmeticSequence α}
variable (h1 : sum_three seq.a seq.d 1 = 45)
variable (h2 : sum_three seq.a seq.d 2 = 29)

theorem find_a3_a6_a9 : sum_three seq.a seq.d 3 = 13 :=
by
  sorry

end find_a3_a6_a9_l210_210514


namespace sequence_contains_3044_l210_210188

/-- Infinite sequence S of decimal digits with given conditions -/
def S : ℕ → ℕ
| 0     := 1
| 1     := 9
| 2     := 8
| 3     := 2
| n + 4 := (S n + S (n + 1) + S (n + 2) + S (n + 3)) % 10

/-- Prove the sequence S contains the substring 3, 0, 4, 4 -/
theorem sequence_contains_3044 : ∃ n : ℕ, S n = 3 ∧ S (n + 1) = 0 ∧ S (n + 2) = 4 ∧ S (n + 3) = 4 :=
sorry

end sequence_contains_3044_l210_210188


namespace problem_A_problem_B_problem_D_l210_210271

-- Option A: Prove that vectors (1, 2) and (3, 1) can be a basis.
theorem problem_A (a b : ℝ × ℝ) (h_a : a = (1, 2)) (h_b : b = (3, 1)) :
  ∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ a ≠ b ∧ a ≠ y•b ∧ b ≠ x•a := sorry

-- Option B: Prove the coordinates of vertex D in a parallelogram ABCD.
theorem problem_B (A B C D : ℝ × ℝ) 
  (h_A : A = (5, -1)) (h_B : B = (-1, 7)) (h_C : C = (1, 2)) (h_D : D = (7, -6)) :
  D = (7, -6) := sorry

-- Option D: Prove the projection of (1, 1) onto (1, 1) is (2, 2).
theorem problem_D (a b : ℝ × ℝ) (h_a : a = (1, 1)) (h_b_norm : ∥b∥ = 4) 
  (h_angle : some_angle = π/4):
  ∃ proj_b : ℝ × ℝ, proj_b = (2, 2) := sorry

end problem_A_problem_B_problem_D_l210_210271


namespace point_in_fourth_quadrant_l210_210868

theorem point_in_fourth_quadrant (m : ℝ) (h : m < 0) : (-m + 1 > 0 ∧ -1 < 0) :=
by
  sorry

end point_in_fourth_quadrant_l210_210868


namespace g_neither_even_nor_odd_l210_210104

def g (x : ℝ) : ℝ := ⌊2 * x⌋ + 1/3

theorem g_neither_even_nor_odd : (∀ x, g x ≠ g (-x)) ∧ (∀ x, g x ≠ -g (-x)) :=
by
  sorry

end g_neither_even_nor_odd_l210_210104


namespace negative_triangle_angles_l210_210522

theorem negative_triangle_angles (A B C : Type) [triangle A B C] :
  ¬(C = 90°) → ¬(acute_angle A ∧ acute_angle B) :=
by
  sorry

end negative_triangle_angles_l210_210522


namespace triangle_ACD_perimeter_sum_is_zero_l210_210152

noncomputable def point (α : Type*) : Type* := {p : α × α // p ≠ (0,0)}
noncomputable def distance (α : Type*) [preorder α] (p1 p2 : point α) : α := sorry

theorem triangle_ACD_perimeter_sum_is_zero :
  ∀ (α : Type*) [linear_ordered_comm_ring α],
  ∃ (A C B D : point α),
    distance α A B = 12 ∧ distance α B C = 24 ∧
    distance α B D = distance α B C ∧
    distance α A D = distance α C D ∧
    distance α A D ∈ set_of (λ x, ∃ n : ℕ, x = n) ∧ -- A D is an integer
    distance α B D ∈ set_of (λ x, ∃ n : ℕ, x = n) ∧ -- B D is an integer
    let perimeters := set_of (λ perimeter, 
      ∃ D, distance α A D + distance α C D + distance α A C = perimeter) in
    perimeters = {0} :=
begin
  sorry
end

end triangle_ACD_perimeter_sum_is_zero_l210_210152


namespace raft_drift_time_l210_210236

-- Define the conditions from the problem
variable (distance : ℝ := 1)
variable (steamboat_time : ℝ := 1) -- in hours
variable (motorboat_time : ℝ := 3 / 4) -- 45 minutes in hours
variable (motorboat_speed_ratio : ℝ := 2)

-- Variables for speeds
variable (vs vf : ℝ)

-- Conditions: the speeds and conditions of traveling from one village to another
variable (steamboat_eqn : vs + vf = distance / steamboat_time := by sorry)
variable (motorboat_eqn : (2 * vs) + vf = distance / motorboat_time := by sorry)

-- Time for the raft to travel the distance
theorem raft_drift_time : 90 = (distance / vf) * 60 := by
  -- Proof comes here
  sorry

end raft_drift_time_l210_210236


namespace find_t_l210_210173

variables (a b c : ℝ)
variables (r s t : ℝ)

-- Conditions provided by the problem
axiom h1 : (a + b + c = 3)
axiom h2 : (ab + bc + ca = 5)
axiom h3 : (abc = 8)

-- The goal statement to prove
theorem find_t (h1 : a + b + c = 3) (h2 : a * b + b * c + c * a = 5) (h3 : a * b * c = 8) :
  t = 243 :=
by
  sorry

end find_t_l210_210173


namespace MagicKing_total_episodes_l210_210987

theorem MagicKing_total_episodes :
  let episodes_s1to3 := 3 * 20,
      episodes_s4to8 := 5 * 25,
      episodes_s9to11 := 3 * 30,
      episodes_s12to14 := 3 * 15,
      holiday_specials := 5,
      cancelled_episodes := 3,
      unaired_released_episodes := 2,
      additional_episodes := 4 in
  episodes_s1to3 + episodes_s4to8 + episodes_s9to11 + episodes_s12to14 + 
  holiday_specials - cancelled_episodes + unaired_released_episodes + additional_episodes = 328 :=
by
  sorry

end MagicKing_total_episodes_l210_210987


namespace cotangent_product_identity_l210_210486

theorem cotangent_product_identity :
  (∏ i in Finset.range 45, (1 + Real.cot (i + 1 : ℝ) * Real.pi / 180)) = 2^23 :=
by
  sorry

end cotangent_product_identity_l210_210486


namespace unique_9_tuple_satisfying_condition_l210_210395

theorem unique_9_tuple_satisfying_condition :
  ∃! (a : Fin 9 → ℕ), 
    (∀ i j k : Fin 9, i < j ∧ j < k →
      ∃ l : Fin 9, l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ a i + a j + a k + a l = 100) :=
sorry

end unique_9_tuple_satisfying_condition_l210_210395


namespace constant_term_l210_210430

theorem constant_term (a : ℝ) (h : a = ∫ x in 0..π, sin x) : 
  let poly := (λ x : ℝ, (x - a / x)^8) in
  (polynomial.coeff (polynomial.C x * polynomial.C (a/x) ^ 8) 0) = 1120 :=
by {
  rw h,
  have : a = 2 := by norm_num,
  sorry
}

end constant_term_l210_210430


namespace max_value_pa_pb_pm_l210_210094

def curve_C1_parametric (t : ℝ) : ℝ × ℝ :=
  (-Real.sqrt 3 * t, t)

def curve_C2_parametric (θ : ℝ) : ℝ × ℝ :=
  (4 * Real.cos θ, 4 * Real.sin θ)

def curve_C1_polar (ρ θ : ℝ) : Prop :=
  θ = 5 * Real.pi / 6 ∧ ρ ∈ Set.univ

def curve_C2_polar (ρ θ : ℝ) : Prop :=
  ρ = 4

def point_P : ℝ × ℝ := (1, 0)

def line_l (n α : ℝ) : ℝ × ℝ :=
  (1 + n * Real.cos α, n * Real.sin α)

def intersection_points (n α : ℝ) : (ℝ × ℝ) :=
  let l := line_l n α
  if (l.fst^2 + l.snd^2 = 16) then l else (0, 0)

def pa_pb_pm (P A B M: ℝ × ℝ) : ℝ :=
  if M ≠ (0,0) then (Real.sqrt ((P.fst - A.fst)^2 + (P.snd - A.snd)^2)) *
                  (Real.sqrt ((P.fst - B.fst)^2 + (P.snd - B.snd)^2)) /
                  (Real.sqrt ((P.fst - M.fst)^2 + (P.snd - M.snd)^2))
   else 0

def max_pa_pb_pm (P : ℝ × ℝ) : ℝ :=
  let A := intersection_points 1 (Real.pi / 3)
  let B := intersection_points (-1) (Real.pi / 3)
  let M := point_P   
  pa_pb_pm P A B M

theorem max_value_pa_pb_pm :
  max_pa_pb_pm point_P = 30 :=
by {
  sorry
}

end max_value_pa_pb_pm_l210_210094


namespace perfect_cubes_and_fourths_below_1000_l210_210065

def is_perfect_cube (n: ℕ) : Prop :=
  ∃ k: ℕ, k ^ 3 = n

def is_perfect_fourth (n: ℕ) : Prop :=
  ∃ k: ℕ, k ^ 4 = n

def is_perfect_twelfth (n: ℕ) : Prop :=
  ∃ k: ℕ, k ^ 12 = n

def count_perfect_cubes_below (limit: ℕ) : ℕ :=
  Finset.card (Finset.filter (λ n, is_perfect_cube n) (Finset.range limit))

def count_perfect_fourths_below (limit: ℕ) : ℕ :=
  Finset.card (Finset.filter (λ n, is_perfect_fourth n) (Finset.range limit))

def count_perfect_twelfths_below (limit: ℕ) : ℕ :=
  Finset.card (Finset.filter (λ n, is_perfect_twelfth n) (Finset.range limit))

theorem perfect_cubes_and_fourths_below_1000 : 
  let total := count_perfect_cubes_below 1000 + count_perfect_fourths_below 1000 - count_perfect_twelfths_below 1000 in
  total = 14 :=
by sorry

end perfect_cubes_and_fourths_below_1000_l210_210065


namespace age_difference_l210_210291

variable (a b c : ℕ)

theorem age_difference (h : a + b = b + c + 13) : a - c = 13 :=
by
  sorry

end age_difference_l210_210291


namespace susan_min_packages_l210_210175

theorem susan_min_packages (n : ℕ) (cost_per_package : ℕ := 5) (earnings_per_package : ℕ := 15) (initial_cost : ℕ := 1200) :
  15 * n - 5 * n ≥ 1200 → n ≥ 120 :=
by {
  sorry -- Proof goes here
}

end susan_min_packages_l210_210175


namespace quadrilateral_is_rectangle_l210_210436

noncomputable def radius_of_inscribed_circle_in_triangle (A B C : Point) : ℝ := sorry
def is_rectangle (A B C D : Point) : Prop := sorry

variables {A B C D : Point}

theorem quadrilateral_is_rectangle
  (convex_ABCD : convex_quad A B C D)
  (rABC rBCD rCDA rDAB : ℝ)
  (h1 : radius_of_inscribed_circle_in_triangle A B C = rABC)
  (h2 : radius_of_inscribed_circle_in_triangle B C D = rBCD)
  (h3 : radius_of_inscribed_circle_in_triangle C D A = rCDA)
  (h4 : radius_of_inscribed_circle_in_triangle D A B = rDAB)
  (radius_eq : rABC = rBCD ∧ rBCD = rCDA ∧ rCDA = rDAB) :
  is_rectangle A B C D := sorry

end quadrilateral_is_rectangle_l210_210436


namespace num_factors_of_M_l210_210856

def M : ℕ := 2^4 * 3^3 * 7^2

theorem num_factors_of_M : ∃ n, n = 60 ∧ (∀ d e f : ℕ, 0 ≤ d ∧ d ≤ 4 ∧ 0 ≤ e ∧ e ≤ 3 ∧ 0 ≤ f ∧ f ≤ 2 → (2^d * 3^e * 7^f ∣ M) ∧ ∃ k, k = 5 * 4 * 3 ∧ k = n) :=
by
  sorry

end num_factors_of_M_l210_210856


namespace card_four_digit_numbers_with_condition_l210_210665

def is_valid_two_digit_number (d1 d2 : ℕ) : Prop :=
  d1 ≠ 0 ∧ (d2 = 0 ∨ d2 = 5)

def satisfies_condition (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  let x := n / 1000 in
  let y := (n / 100) % 10 in
  let z := (n / 10) % 10 in
  let w := n % 10 in
  is_valid_two_digit_number x y ∧
  is_valid_two_digit_number x z ∧
  is_valid_two_digit_number x w ∧
  is_valid_two_digit_number y z ∧
  is_valid_two_digit_number y w ∧
  is_valid_two_digit_number z w

theorem card_four_digit_numbers_with_condition : 
  ∃ (count : ℕ), count = 18 ∧
  (∀ n : ℕ, satisfies_condition n → 
    (n ∈ finset.range 10000 ∩ finset.range 1000).card = count) :=
by
  sorry

end card_four_digit_numbers_with_condition_l210_210665


namespace total_stocking_stuffers_total_stocking_stuffers_hannah_buys_l210_210053

def candy_canes_per_kid : ℕ := 4
def beanie_babies_per_kid : ℕ := 2
def books_per_kid : ℕ := 1
def kids : ℕ := 3

theorem total_stocking_stuffers : candy_canes_per_kid + beanie_babies_per_kid + books_per_kid = 7 :=
by { 
  -- by trusted computation
  sorry
}

theorem total_stocking_stuffers_hannah_buys : 3 * (candy_canes_per_kid + beanie_babies_per_kid + books_per_kid) = 21 :=
by {
  have h : candy_canes_per_kid + beanie_babies_per_kid + books_per_kid = 7 := total_stocking_stuffers,
  rw h,
  norm_num,
}

end total_stocking_stuffers_total_stocking_stuffers_hannah_buys_l210_210053


namespace probability_sum_divisible_by_6_l210_210500

theorem probability_sum_divisible_by_6 :
  let nums := {1, 2, 3, 4},
      combinations := ({a, b, c} | a b c : ℕ, a ∈ nums ∧ b ∈ nums ∧ c ∈ nums ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c),
      valid_combinations := {combo | combo.sum % 6 = 0}
  in combination.finite.to_list.count (combo ∈ valid_combinations) / combination.finite.to_list.length = 1 / 4 := sorry

end probability_sum_divisible_by_6_l210_210500


namespace pentagon_diagl_sum_pentagon_diagonal_391_l210_210906

noncomputable def diagonal_sum (AB CD BC DE AE : ℕ) 
  (AC : ℚ) (BD : ℚ) (CE : ℚ) (AD : ℚ) (BE : ℚ) : ℚ :=
  3 * AC + AD + BE

theorem pentagon_diagl_sum (AB CD BC DE AE : ℕ)
  (hAB : AB = 3) (hCD : CD = 3) 
  (hBC : BC = 10) (hDE : DE = 10) 
  (hAE : AE = 14)
  (AC BD CE AD BE : ℚ)
  (hACBC : AC = 12) 
  (hADBC: AD = 13.5)
  (hCEBE: BE = 44 / 3) :
  diagonal_sum AB CD BC DE AE AC BD CE AD BE = 385 / 6 := sorry

theorem pentagon_diagonal_391 (AB CD BC DE AE : ℕ)
  (hAB : AB = 3) (hCD : CD = 3) 
  (hBC : BC = 10) (hDE : DE = 10) 
  (hAE : AE = 14)
  (AC BD CE AD BE : ℚ)
  (hACBC : AC = 12) 
  (hADBC: AD = 13.5)
  (hCEBE: BE = 44 / 3) :
  ∃ m n : ℕ, 
    m.gcd n = 1 ∧
    m / n = 385 / 6 ∧
    m + n = 391 := sorry

end pentagon_diagl_sum_pentagon_diagonal_391_l210_210906


namespace g_neither_even_nor_odd_l210_210109

def g (x : ℝ) : ℝ := floor (2 * x) + (1 / 3)

theorem g_neither_even_nor_odd :
  ¬(∀ x : ℝ, g (-x) = g x) ∧ ¬(∀ x : ℝ, g (-x) = -g x) := by
  sorry

end g_neither_even_nor_odd_l210_210109


namespace alex_money_left_l210_210363

noncomputable def weekly_income := 500
def income_tax_rate := 0.10
def water_bill := 55
def tithe_rate := 0.10

theorem alex_money_left : (weekly_income - ((weekly_income * income_tax_rate) + (weekly_income * tithe_rate) + water_bill)) = 345 := 
by
  sorry

end alex_money_left_l210_210363


namespace sqrt_square_eq_self_l210_210074

theorem sqrt_square_eq_self (a : ℝ) (h : a ≥ 1/2) :
  Real.sqrt ((2 * a - 1) ^ 2) = 2 * a - 1 :=
by
  sorry

end sqrt_square_eq_self_l210_210074


namespace train_passes_jogger_in_37_seconds_l210_210283

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def train_speed_kmph : ℝ := 45
noncomputable def jogger_lead_m : ℝ := 250
noncomputable def train_length_m : ℝ := 120

noncomputable def jogger_speed_mps : ℝ := jogger_speed_kmph * 1000 / 3600
noncomputable def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600
noncomputable def relative_speed_mps : ℝ := train_speed_mps - jogger_speed_mps
noncomputable def total_distance_m : ℝ := jogger_lead_m + train_length_m

theorem train_passes_jogger_in_37_seconds :
  total_distance_m / relative_speed_mps = 37 := by
  sorry

end train_passes_jogger_in_37_seconds_l210_210283


namespace complex_expression_magnitude_l210_210302

def i := Complex.I

theorem complex_expression_magnitude :
  |2 + i^2 + 2 * i^3| = Real.sqrt 5 :=
by
  sorry

end complex_expression_magnitude_l210_210302


namespace tan_period_intersection_distance_l210_210200

theorem tan_period_intersection_distance (ω : ℝ) (hω_pos : ω > 0) 
  (h_intersect_dist : ∀ n : ℤ, ∃ x : ℝ, y = 3 ∧ f(x) = 3 ∧ (xₙ₊₁ - xₙ) = (π / 4)) :
  f(π / 12) = √3 := by
  sorry

end tan_period_intersection_distance_l210_210200


namespace geometric_series_value_of_m_l210_210369

theorem geometric_series_value_of_m :
  ∃ m : ℤ,
  let a₁ := 15 in
  let r₁ := 5 / 15 in
  let r₂ := (5 + m) / 15 in
  let S₁ := a₁ / (1 - r₁) in
  let S₂ := a₁ / (1 - r₂) in
  S₂ = 3 * S₁ ∧ m = 7 :=
begin
  -- Placeholder proof
  sorry,
end

end geometric_series_value_of_m_l210_210369


namespace existence_of_sequence_l210_210745

theorem existence_of_sequence :
  ∃ a : ℕ → ℕ, (∀ n : ℕ, ∃! k : ℕ, a k = n) ∧ (∀ k : ℕ, k > 0 → (∑ i in Finset.range k, a (i + 1)) % k = 0) :=
sorry

end existence_of_sequence_l210_210745


namespace grandson_age_in_months_l210_210939

theorem grandson_age_in_months 
  (Y_y : ℕ) (G_d S_w : ℕ)
  (h0 : Y_y = 84)
  (h1 : G_d = S_w * 7)
  (h2 : ∀ (G_m : ℕ), G_m = Y_y * 12)
  (h3 : G_d / 365 + S_w * 7 / 365 + Y_y = 140) :
  84 * 12 ≈ 1008 :=
by
  sorry

end grandson_age_in_months_l210_210939


namespace hannah_stocking_stuffers_l210_210058

theorem hannah_stocking_stuffers (candy_caness : ℕ) (beanie_babies : ℕ) (books : ℕ) (kids : ℕ) : 
  candy_caness = 4 → 
  beanie_babies = 2 → 
  books = 1 → 
  kids = 3 → 
  candy_caness + beanie_babies + books = 7 → 
  7 * kids = 21 := 
by sorry

end hannah_stocking_stuffers_l210_210058


namespace area_of_one_trapezoid_l210_210518

theorem area_of_one_trapezoid (hexagon_area : ℝ) (triangle_area : ℝ) (num_trapezoids : ℝ) 
  (h_hexagon : hexagon_area = 24) (h_triangle : triangle_area = 4) (h_trapezoids : num_trapezoids = 6) :
  (hexagon_area - triangle_area) / num_trapezoids = 10 / 3 :=
by
  rw [h_hexagon, h_triangle, h_trapezoids]
  norm_num
  sorry

end area_of_one_trapezoid_l210_210518


namespace sum_abs_a_equals_137_l210_210930

open Nat

def a (n : ℕ) : ℤ := 2 * n - 9

theorem sum_abs_a_equals_137 : (∑ k in Finset.range 15, |a (k + 1)|) = 137 := by
  sorry

end sum_abs_a_equals_137_l210_210930


namespace midpoint_D_EF_l210_210435

-- Definitions and conditions
variables {Point : Type*} [MetricSpace Point]
variables (O P A B C E F L D : Point)
variables (line : Point → Point → set Point)

-- Tangent lines PA and PB
def is_tangent (P A : Point) : Prop := sorry

-- Hypotheses based on the conditions
axiom h1 : ¬ (∃ x, (x ∈ (line P A)) ∧ (x ∈ (line P B)))  -- Point P is outside the circle
axiom h2 : is_tangent P A  -- PA is tangent to the circle at A
axiom h3 : is_tangent P B  -- PB is tangent to the circle at B
axiom h4 : C ∈ (set.Points_of_circle O)  -- C is on the circle
axiom h5 : E ∈ (line P A)  -- Tangent at C intersects PA at E
axiom h6 : F ∈ (line P B)  -- Tangent at C intersects PB at F
axiom h7 : L ∈ (line O C)  -- Line OC intersects AB at L
axiom h8 : D ∈ (line L P)  -- Line LP intersects EF at D

-- Theorem statement to prove D is the midpoint of EF
theorem midpoint_D_EF : midpoint E F D := sorry

end midpoint_D_EF_l210_210435


namespace average_speed_l210_210255

theorem average_speed (d d1 d2 s1 s2 : ℝ)
    (h1 : d = 100)
    (h2 : d1 = 50)
    (h3 : d2 = 50)
    (h4 : s1 = 20)
    (h5 : s2 = 50) :
    d / ((d1 / s1) + (d2 / s2)) = 28.57 :=
by
  sorry

end average_speed_l210_210255


namespace A_completion_time_l210_210337

theorem A_completion_time :
  ∃ A : ℝ, (A > 0) ∧ (
    (2 * (1 / A + 1 / 10) + 3.0000000000000004 * (1 / 10) = 1) ↔ A = 4
  ) :=
by
  have B_workday := 10
  sorry -- proof would go here

end A_completion_time_l210_210337


namespace g_neither_even_nor_odd_l210_210114

def g (x : ℝ) : ℝ := ⌊2 * x⌋ + (1 / 3 : ℝ)

theorem g_neither_even_nor_odd :
  ¬ (∀ x : ℝ, g x = g (-x)) ∧ ¬ (∀ x : ℝ, g x = - g (-x)) :=
by
  sorry

end g_neither_even_nor_odd_l210_210114


namespace problem_statement_l210_210802

-- Definitions based on conditions
def ellipse (a b : ℝ) : ℝ × ℝ → Prop := λ p, let ⟨x, y⟩ := p in x^2 / a^2 + y^2 / b^2 = 1
def point_on_ellipse_C (x y : ℝ) := ellipse 2 (sqrt 3) (x, y)
def line_through_F (k : ℝ) (x : ℝ) : ℝ := k * (x - 1)
def slope_of_line (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- Main theorem statement
theorem problem_statement (k : ℝ) (hk : k ≠ 0) :
  let D := (x1, line_through_F k x1)
  let E := (x2, line_through_F k x2)
  let k1 := slope_of_line (0, 0) (x1, line_through_F k x1)
  let k2 := slope_of_line (0, 0) (x2, line_through_F k x2)
  ∀ (x1 x2 : ℝ),
  x1^2 / 4 + (line_through_F k x1)^2 / 3 = 1 →
  x2^2 / 4 + (line_through_F k x2)^2 / 3 = 1 →
  k * k1 + k * k2 = -1 := by
    sorry

end problem_statement_l210_210802


namespace raft_drift_time_l210_210233

-- Define the distance between the villages
def distance : ℝ := 1

-- Define the speed of the steamboat in still water (in units/hour)
def v_s : ℝ := sorry

-- Define the time taken by the steamboat to travel the distance (in hours)
def t_steamboat : ℝ := 1

-- Define the speed of the motorboat in still water (in units/hour)
def v_m : ℝ := 2 * v_s

-- Define the time taken by the motorboat to travel the distance (in hours)
def t_motorboat : ℝ := 45 / 60

-- Define the speed of the river's current (in units/hour)
def v_f : ℝ := sorry

-- Equations for steamboat and motorboat effective speeds
def steamboat_equation := v_s + v_f = distance / t_steamboat
def motorboat_equation := v_m + v_f = distance / t_motorboat

-- Solve for the speeds
def v_s_solution : ℝ := 1 - v_f
def v_f_solution : ℝ := (4 / 3) - 2 * v_s_solution

-- Define the time for the raft to drift from Verkhnie Vasyuki to Nizhnie Vasyuki (in hours)
def raft_time_hours : ℝ := distance / v_f_solution

-- Convert the time to minutes
def raft_time_minutes : ℝ := raft_time_hours * 60

-- Theorem statement that proves the raft drifts in 90 minutes
theorem raft_drift_time : raft_time_minutes = 90 :=
by
  unfold distance v_s t_steamboat v_m t_motorboat v_f steamboat_equation motorboat_equation v_s_solution v_f_solution raft_time_hours raft_time_minutes
  rw [←solve_v_s, ←solve_v_f]
  sorry

end raft_drift_time_l210_210233


namespace population_in_1998_l210_210503

theorem population_in_1998 (population_millions : ℝ) (h1 : population_millions = 30.3) :
  ∃ (n : ℕ), n = 30300000 ∧ n = (population_millions * 1000000).to_nat :=
by
  sorry

end population_in_1998_l210_210503


namespace correctStatement_l210_210024

variable (U : Set ℕ) (M : Set ℕ)

namespace Proof

-- Given conditions
def universalSet := {1, 2, 3, 4, 5}
def complementM := {1, 3}
def isComplement (M : Set ℕ) : Prop := U \ M = complementM

-- Target statement to be proved
theorem correctStatement (h1 : U = universalSet) (h2 : isComplement M) : 2 ∈ M := by
  sorry

end Proof

end correctStatement_l210_210024


namespace moment_generating_function_inequality_l210_210540

theorem moment_generating_function_inequality (X : ℝ → ℝ) 
  (hx_nonneg : ∀ x, 0 ≤ X x) 
  (M : ℝ → ℝ) 
  (hM : ∀ s, M s = ∫ x, (Real.exp (s * X x))) 
  (p s : ℝ) 
  (hp_pos : 0 < p)
  (hs_pos : 0 < s) :
  ∫ x, ((X x) ^ p) ≤ ((p / (Real.exp(1) * s)) ^ p) * M s :=
by
  sorry

end moment_generating_function_inequality_l210_210540


namespace other_leg_length_of_second_triangle_l210_210689

theorem other_leg_length_of_second_triangle
  (x : ℕ) (hypotenuse_first_triangle : ℕ) (leg_first_triangle : ℕ) (scale_factor: ℕ) : 
  x = 7 → hypotenuse_first_triangle = 25 → leg_first_triangle = 24 → scale_factor = 4 →
  (4 * x) = 28 :=
by
  intros
  -- Below might not be necessary but it would be used to 
  -- indicate that further proof could be adapted from the hints in
  -- the existing solution
  unfold hypotenuse_first_triangle leg_first_triangle scale_factor
  sorry

end other_leg_length_of_second_triangle_l210_210689


namespace equation_of_line_M_l210_210840

def line (m b : ℝ) := λ x : ℝ, m * x + b

def given_line := line (-3/4) 5

def slope_of_line_M := (-3/4) / 3
def intercept_of_line_M := 3 * 5

theorem equation_of_line_M :
  ∀ x, line slope_of_line_M intercept_of_line_M x = -1/4 * x + 15 :=
by
  intro x
  simp [slope_of_line_M, intercept_of_line_M, line]
  sorry

end equation_of_line_M_l210_210840


namespace batsman_average_after_12th_innings_l210_210336

theorem batsman_average_after_12th_innings (A : ℕ) (total_runs_11 : ℕ) (total_runs_12 : ℕ ) : 
  total_runs_11 = 11 * A → 
  total_runs_12 = total_runs_11 + 55 → 
  (total_runs_12 / 12 = A + 1) → 
  (A + 1) = 44 := 
by
  intros h1 h2 h3
  sorry

end batsman_average_after_12th_innings_l210_210336


namespace probability_selecting_both_types_X_distribution_correct_E_X_correct_l210_210748

section DragonBoatFestival

/-- The total number of zongzi on the plate -/
def total_zongzi : ℕ := 10

/-- The total number of red bean zongzi -/
def red_bean_zongzi : ℕ := 2

/-- The total number of plain zongzi -/
def plain_zongzi : ℕ := 8

/-- The number of zongzi to select -/
def zongzi_to_select : ℕ := 3

/-- Probability of selecting at least one red bean zongzi and at least one plain zongzi -/
def probability_selecting_both : ℚ := 8 / 15

/-- Distribution of the number of red bean zongzi selected (X) -/
def X_distribution : ℕ → ℚ
| 0 => 7 / 15
| 1 => 7 / 15
| 2 => 1 / 15
| _ => 0

/-- Mathematical expectation of the number of red bean zongzi selected (E(X)) -/
def E_X : ℚ := 3 / 5

/-- Theorem stating the probability of selecting both types of zongzi -/
theorem probability_selecting_both_types :
  let p := probability_selecting_both
  p = 8 / 15 :=
by
  let p := probability_selecting_both
  sorry

/-- Theorem stating the probability distribution of the number of red bean zongzi selected -/
theorem X_distribution_correct :
  (X_distribution 0 = 7 / 15) ∧
  (X_distribution 1 = 7 / 15) ∧
  (X_distribution 2 = 1 / 15) :=
by
  sorry

/-- Theorem stating the mathematical expectation of the number of red bean zongzi selected -/
theorem E_X_correct :
  let E := E_X
  E = 3 / 5 :=
by
  let E := E_X
  sorry

end DragonBoatFestival

end probability_selecting_both_types_X_distribution_correct_E_X_correct_l210_210748


namespace parallel_resistance_example_l210_210674

theorem parallel_resistance_example :
  ∀ (R1 R2 : ℕ), R1 = 3 → R2 = 6 → 1 / (R : ℚ) = 1 / (R1 : ℚ) + 1 / (R2 : ℚ) → R = 2 := by
  intros R1 R2 hR1 hR2 h_formula
  -- Formulation of the resistance equations and assumptions
  sorry

end parallel_resistance_example_l210_210674


namespace task1_task2_l210_210878

-- Define the conditions and the probabilities to be proven

def total_pens := 6
def first_class_pens := 3
def second_class_pens := 2
def third_class_pens := 1

def total_combinations := Nat.choose total_pens 2

def combinations_with_exactly_one_first_class : Nat :=
  (first_class_pens * (total_pens - first_class_pens))

def probability_one_first_class_pen : ℚ :=
  combinations_with_exactly_one_first_class / total_combinations

def combinations_without_any_third_class : Nat :=
  Nat.choose (first_class_pens + second_class_pens) 2

def probability_no_third_class_pen : ℚ :=
  combinations_without_any_third_class / total_combinations

theorem task1 : probability_one_first_class_pen = 3 / 5 := 
  sorry

theorem task2 : probability_no_third_class_pen = 2 / 3 := 
  sorry

end task1_task2_l210_210878


namespace haley_total_trees_l210_210479

-- Define the number of dead trees and remaining trees
def dead_trees : ℕ := 5
def remaining_trees : ℕ := 12

-- Prove the total number of trees Haley originally grew
theorem haley_total_trees :
  (dead_trees + remaining_trees) = 17 :=
by
  -- Providing the proof using sorry as placeholder
  sorry

end haley_total_trees_l210_210479


namespace jim_final_distance_l210_210118

theorem jim_final_distance :
  let south := 60
  let west := 40
  let north := 20
  let east := 10
  let net_south_north := south - north
  let net_west_east := west - east
  let distance := Math.sqrt ((net_south_north ^ 2) + (net_west_east ^ 2))
in distance = 50 := by
  sorry

end jim_final_distance_l210_210118


namespace complex_expression_magnitude_l210_210298

def i := Complex.I

theorem complex_expression_magnitude :
  |2 + i^2 + 2 * i^3| = Real.sqrt 5 :=
by
  sorry

end complex_expression_magnitude_l210_210298


namespace solution_correct_l210_210657

noncomputable def A : ℕ := 2
noncomputable def B : ℕ := 1
noncomputable def C : ℕ := 4
noncomputable def D : ℕ := 9

def is_digit (n : ℕ) : Prop := n >= 0 ∧ n <= 9
def are_distinct (a b c d : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
def eq_ABA (a b : ℕ) : ℕ := a * 100 + b * 10 + a
def eq_CCDC (c d : ℕ) : ℕ := c * 1000 + c * 100 + d * 10 + c

theorem solution_correct :
  is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧
  are_distinct A B C D ∧
  (eq_ABA A B)^2 = eq_CCDC C D ∧
  eq_ABA A B < 316 ∧
  (eq_ABA A B)^2 < 100000 :=
by
  split; repeat { sorry }

end solution_correct_l210_210657


namespace inequality_holds_l210_210567

theorem inequality_holds (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) * (4 * x + y + 2 * z) * (2 * x + y + 8 * z) ≥ (375 / 2) * x * y * z :=
by
  sorry

end inequality_holds_l210_210567


namespace pyramid_segment_SD_l210_210506

noncomputable def segment_SD (r h a slal : ℝ) : ℝ :=
  (78 : ℝ) / 13

theorem pyramid_segment_SD {r h a : ℝ} (r_pos : r > 0) (h_pos : h > 0) (a_pos : a > 0)
  (r_eq : r = 2) (h_eq : h = 6) (a_eq : a = 12) (mn_eq : ∀ M N : ℝ, abs (M - N) = 7) :
  let SD := segment_SD r h a 1 in
  SD = 78 / 13 :=
by
  sorry

end pyramid_segment_SD_l210_210506


namespace count_perfect_cubes_or_fourth_powers_l210_210066

theorem count_perfect_cubes_or_fourth_powers :
  ∃ n : ℕ, n = 14 ∧ ∀ x : ℕ,
  0 < x ∧ x < 1000 → (∃ k : ℕ, x = k^3) ∨ (∃ j : ℕ, x = j^4) ↔ n = 14 :=
begin
  sorry,
end

end count_perfect_cubes_or_fourth_powers_l210_210066


namespace g_neither_even_nor_odd_l210_210108

def g (x : ℝ) : ℝ := ⌊2 * x⌋ + 1 / 3

theorem g_neither_even_nor_odd : ¬ (∀ x, g (-x) = g x) ∧ ¬ (∀ x, g (-x) = -g x) :=
by
  sorry

end g_neither_even_nor_odd_l210_210108


namespace count_of_valid_four_digit_numbers_l210_210852

def is_four_digit_number (a b c d : ℕ) : Prop :=
  a ≠ 0 ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9

def digits_sum_to_twelve (a b c d : ℕ) : Prop :=
  a + b + c + d = 12

def divisible_by_eleven (a b c d : ℕ) : Prop :=
  (a + c - (b + d)) % 11 = 0

theorem count_of_valid_four_digit_numbers : ∃ n : ℕ, n = 20 ∧
  (∀ a b c d : ℕ, is_four_digit_number a b c d →
  digits_sum_to_twelve a b c d →
  divisible_by_eleven a b c d →
  true) :=
sorry

end count_of_valid_four_digit_numbers_l210_210852


namespace arcsin_one_eq_pi_div_two_l210_210720

noncomputable def arcsin (x : ℝ) : ℝ :=
classical.some (exists_inverse_sin x)

theorem arcsin_one_eq_pi_div_two : arcsin 1 = π / 2 :=
sorry

end arcsin_one_eq_pi_div_two_l210_210720


namespace childrens_events_l210_210996

theorem childrens_events (total_cupcakes cupcakes_per_event : ℕ) 
  (h1 : total_cupcakes = 768)
  (h2 : cupcakes_per_event = 96) :
  total_cupcakes / cupcakes_per_event = 8 :=
by
  rw [h1, h2]
  norm_num
  sorry -- Replace this with the actual proof steps if needed

end childrens_events_l210_210996


namespace inequality_holds_l210_210566

theorem inequality_holds (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  x^4 + y^4 + 2 / (x^2 * y^2) ≥ 4 := 
by
  sorry

end inequality_holds_l210_210566


namespace solve_system_l210_210579

variable {R : Type*} [CommRing R]

-- Given conditions
variables (a b c x y z : R)

-- Assuming the given system of equations
axiom eq1 : x + a*y + a^2*z + a^3 = 0
axiom eq2 : x + b*y + b^2*z + b^3 = 0
axiom eq3 : x + c*y + c^2*z + c^3 = 0

-- The goal is to prove the mathematical equivalence
theorem solve_system : x = -a*b*c ∧ y = a*b + b*c + c*a ∧ z = -(a + b + c) :=
by
  sorry

end solve_system_l210_210579


namespace negation_of_quadratic_statement_l210_210292

variable {x a b : ℝ}

theorem negation_of_quadratic_statement (h : x = a ∨ x = b) : x^2 - (a + b) * x + ab = 0 := sorry

end negation_of_quadratic_statement_l210_210292


namespace net_configuration_existence_l210_210151

noncomputable def exists_valid_net_configuration (length width height : ℕ) : Prop :=
  ∃ net: list (list ℕ), 
    (length = 2 ∧ width = 1 ∧ height = 1) → 
    (∑ (square ∈ net), 1 = 9) →

theorem net_configuration_existence :
  exists_valid_net_configuration 2 1 1 :=
sorry

end net_configuration_existence_l210_210151


namespace range_of_k_l210_210463

theorem range_of_k (k : ℝ) (h : ∀ x y : ℝ, x ∈ Ioc (π / 4) (π / 3) → y ∈ Ioc (π / 4) (π / 3) → x < y → k * cos (k * x) > k * cos (k * y)) :
  k ∈ Icc (-6) (-4) ∨ k ∈ Ioc 0 3 ∨ k ∈ Icc 8 9 ∨ k = -12 :=
  sorry

end range_of_k_l210_210463


namespace fraction_value_l210_210075

-- Given conditions:
variables {w x y : ℝ}

-- Definitions directly from conditions
def condition1 : Prop := w * x = y
def condition2 : Prop := (w + x) / 2 = 0.5

-- Problem statement in Lean 4
theorem fraction_value (h1 : condition1) (h2 : condition2) : 5 / w + 5 / x = 20 :=
sorry

end fraction_value_l210_210075


namespace garage_sale_items_count_l210_210287

theorem garage_sale_items_count (h_diff_prices : ∀ x y, x ≠ y → price x ≠ price y)
  (h_14th_highest : ∃ radio, is_14th_highest radio)
  (h_21st_lowest : ∃ radio, is_21st_lowest radio) :
  total_items_sold = 34 := 
sorry

end garage_sale_items_count_l210_210287


namespace sum_of_absolute_values_l210_210800

-- Define the arithmetic sequence
def a (n : ℕ) : ℤ := 2 * n - 7

-- Conditions
def a_sequence (n : ℕ) : Prop := a (n + 1) - a (n) = 2 ∧ a (1) = -5

-- Proof problem
theorem sum_of_absolute_values : (∑ n in Finset.range 6, |a (n + 1)|) = 18 :=
  sorry

end sum_of_absolute_values_l210_210800


namespace discounted_total_wholesale_cost_l210_210359

theorem discounted_total_wholesale_cost 
  (pants_retail : ℝ) (pants_markup : ℝ)
  (shirt_retail : ℝ) (shirt_markup : ℝ)
  (jacket_retail : ℝ) (jacket_markup : ℝ)
  (skirt_retail : ℝ) (skirt_markup : ℝ)
  (dress_retail : ℝ) (dress_markup : ℝ)
  (discount : ℝ) :
  pants_retail = 36 → pants_markup = 0.80 →
  shirt_retail = 45 → shirt_markup = 0.60 →
  jacket_retail = 120 → jacket_markup = 0.50 →
  skirt_retail = 80 → skirt_markup = 0.75 →
  dress_retail = 150 → dress_markup = 0.40 →
  discount = 0.10 →
  ((pants_retail / (1 + pants_markup)) +
   (shirt_retail / (1 + shirt_markup)) +
   (jacket_retail / (1 + jacket_markup)) +
   (skirt_retail / (1 + skirt_markup)) +
   (dress_retail / (1 + dress_markup))) * (1 - discount) = 252.88 :=
begin
  sorry
end

end discounted_total_wholesale_cost_l210_210359


namespace remainder_equality_l210_210531

theorem remainder_equality
  (P P' K D R R' r r' : ℕ)
  (h1 : P > P')
  (h2 : P % K = 0)
  (h3 : P' % K = 0)
  (h4 : P % D = R)
  (h5 : P' % D = R')
  (h6 : (P * K - P') % D = r)
  (h7 : (R * K - R') % D = r') :
  r = r' :=
sorry

end remainder_equality_l210_210531


namespace rectangle_length_width_difference_l210_210437

theorem rectangle_length_width_difference
  (x y : ℝ)
  (h1 : y = 1 / 3 * x)
  (h2 : 2 * x + 2 * y = 32)
  (h3 : Real.sqrt (x^2 + y^2) = 17) :
  abs (x - y) = 8 :=
sorry

end rectangle_length_width_difference_l210_210437


namespace maximize_dragon_resilience_l210_210751

noncomputable def p_s (x : ℝ) (s : ℕ) : ℝ :=
  x^s / (1 + x + x^2)

def K : List ℕ :=
  [1, 2, 2, 1, 0, 2, 1, 0, 1, 2]

def P_K (x : ℝ) : ℝ :=
  List.foldr (λ s acc => acc * p_s x s) 1 K

theorem maximize_dragon_resilience :
  let x_opt := (Real.sqrt 97 + 1) / 8 in
  ∀ x > 0, P_K x ≤ P_K x_opt :=
sorry

end maximize_dragon_resilience_l210_210751


namespace find_a_l210_210133

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - real.sqrt 2

theorem find_a (a : ℝ) (h1 : 0 < a) (h2 : f a (f a (real.sqrt 2)) = -real.sqrt 2) : a = real.sqrt 2 / 2 :=
sorry

end find_a_l210_210133


namespace hyperbola_eccentricity_range_l210_210838

theorem hyperbola_eccentricity_range (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b)
  (h_hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (F : ℝ) (A B : ℝ × ℝ) (h_AF : A.1 = -F ∧ B.1 = -F ∧ A.2 = (b^2 / a) ∧ B.2 = - (b^2 / a))
  (M : ℝ × ℝ) (h_M : M = (a, 0))
  (outside_circle : ∀d : ℝ, d > 0 → ||M.1 - F|| > d)
  : 1 < real.sqrt (1 + (b^2 / a^2)) < 2 :=
by
    sorry

end hyperbola_eccentricity_range_l210_210838


namespace vanessa_total_earnings_l210_210261

theorem vanessa_total_earnings :
  let num_dresses := 7
  let num_shirts := 4
  let price_per_dress := 7
  let price_per_shirt := 5
  (num_dresses * price_per_dress + num_shirts * price_per_shirt) = 69 :=
by
  sorry

end vanessa_total_earnings_l210_210261


namespace prod_fraction_result_l210_210385

theorem prod_fraction_result :
  (∏ n in Finset.range 15, (n + 5) / (n + 1)) = 3876 :=
sorry

end prod_fraction_result_l210_210385


namespace range_of_a_l210_210469

theorem range_of_a (a : ℝ) :
  (∀ x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}, ax + 2004 ≥ 0) →
  -222.66666666666666 ≤ a ∧ a < -200.4 :=
by sorry

end range_of_a_l210_210469


namespace jori_water_left_l210_210900

theorem jori_water_left (initial_gallons used_gallons : ℚ) (h1 : initial_gallons = 3) (h2 : used_gallons = 11 / 4) :
  initial_gallons - used_gallons = 1 / 4 :=
by
  sorry

end jori_water_left_l210_210900


namespace choose_marbles_with_at_least_one_red_l210_210899

-- Given conditions
def marbles : ℕ := 10
def choose_marbles : ℕ := 4
def red_marble_present : Bool := true

-- The final theorem we want to prove
theorem choose_marbles_with_at_least_one_red :
  (nat.choose marbles choose_marbles) - (nat.choose (marbles - 1) choose_marbles) = 84 := by
  -- Conditions directly applied in the theorem, hence no additional step required
  sorry

end choose_marbles_with_at_least_one_red_l210_210899


namespace find_omega_l210_210869

noncomputable def omega := sorry

theorem find_omega (omega : ℝ) : 
  (0 < omega) → 
  (∀ x1 x2, 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ (π / 3) → sin (ω * x1) ≤ sin (ω * x2)) → 
  (∀ x1 x2, (π / 3) ≤ x1 ∧ x1 < x2 ∧ x2 ≤ (π / 2) → sin (ω * x1) ≥ sin (ω * x2)) → 
  ω = (3 / 2) :=
sorry

end find_omega_l210_210869


namespace b_profit_l210_210333

noncomputable def profit_share (x t : ℝ) : ℝ :=
  let total_profit := 31500
  let a_investment := 3 * x
  let a_period := 2 * t
  let b_investment := x
  let b_period := t
  let profit_ratio_a := a_investment * a_period
  let profit_ratio_b := b_investment * b_period
  let total_ratio := profit_ratio_a + profit_ratio_b
  let b_share := profit_ratio_b / total_ratio
  b_share * total_profit

theorem b_profit (x t : ℝ) : profit_share x t = 4500 :=
by
  sorry

end b_profit_l210_210333


namespace correctStatement_l210_210020

variable (U : Set ℕ) (M : Set ℕ)

namespace Proof

-- Given conditions
def universalSet := {1, 2, 3, 4, 5}
def complementM := {1, 3}
def isComplement (M : Set ℕ) : Prop := U \ M = complementM

-- Target statement to be proved
theorem correctStatement (h1 : U = universalSet) (h2 : isComplement M) : 2 ∈ M := by
  sorry

end Proof

end correctStatement_l210_210020


namespace teddy_bears_count_l210_210546

theorem teddy_bears_count (toys_count : ℕ) (toy_cost : ℕ) (total_money : ℕ) (teddy_bear_cost : ℕ) : 
  toys_count = 28 → 
  toy_cost = 10 → 
  total_money = 580 → 
  teddy_bear_cost = 15 →
  ((total_money - toys_count * toy_cost) / teddy_bear_cost) = 20 :=
by
  intros h1 h2 h3 h4
  sorry

end teddy_bears_count_l210_210546


namespace reflection_correct_l210_210780

def vec2 : Type := ℝ × ℝ

def reflection (a b : vec2) : vec2 :=
  let p_denom := (b.1 * b.1 + b.2 * b.2)
  let proj := ((a.1 * b.1 + a.2 * b.2) / p_denom) * b
  (2 * proj.1 - a.1, 2 * proj.2 - a.2)

def a : vec2 := (2, 6)
def b : vec2 := (2, 1)
def r : vec2 := (6, -2)

theorem reflection_correct : reflection a b = r := by
  sorry

end reflection_correct_l210_210780


namespace quadrilateral_area_lt_one_l210_210497

theorem quadrilateral_area_lt_one 
  (a b c d : ℝ) 
  (h_a : a < 1) 
  (h_b : b < 1) 
  (h_c : c < 1) 
  (h_d : d < 1) 
  (h_pos_a : 0 ≤ a)
  (h_pos_b : 0 ≤ b)
  (h_pos_c : 0 ≤ c)
  (h_pos_d : 0 ≤ d) :
  ∃ (area : ℝ), area < 1 :=
by
  sorry

end quadrilateral_area_lt_one_l210_210497


namespace counts_of_perfect_cubes_and_fourth_powers_l210_210069

theorem counts_of_perfect_cubes_and_fourth_powers : 
  finset.card ({n | (∃ k, n = k^3) ∨ (∃ k, n = k^4) ∧ 0 < n ∧ n < 1000}) = 14 :=
begin
  sorry
end

end counts_of_perfect_cubes_and_fourth_powers_l210_210069


namespace dragon_resilience_l210_210766

noncomputable def probability_function (x : ℝ) : ℝ :=
  let p0 := 1 / (1 + x + x^2)
  let p1 := x / (1 + x + x^2)
  let p2 := x^2 / (1 + x + x^2)
  p1 * p2^2 * p1 * p0 * p2 * p1 * p0 * p1 * p2

theorem dragon_resilience (x : ℝ) (hx : x > 0) : 
  has_max (λ x, probability_function x) ∧ probability_function (sqrt 97 + 1) / 8 = max :=
sorry

end dragon_resilience_l210_210766


namespace white_ball_is_random_event_l210_210877

variable {Ω : Type} -- Sample space
variable (E : Event Ω) -- Event of drawing a white ball
variables (a b : Ω) -- The two balls

-- Conditions
axiom condition_1 : a ≠ b -- The balls are distinct
axiom condition_2 : P(E | a) = 1/2 -- Probability of drawing each ball is 1/2
axiom condition_3 : E = Event.occurs a -- Event of drawing the white ball is getting 'a'

-- Proof statement
theorem white_ball_is_random_event : random_event E :=
sorry

end white_ball_is_random_event_l210_210877


namespace reggie_loses_by_2_points_l210_210158

theorem reggie_loses_by_2_points :
  let reggie_layups := 3,
      reggie_free_throws := 2,
      reggie_long_shots := 1,
      brother_long_shots := 4,
      points_per_layup := 1,
      points_per_free_throw := 2,
      points_per_long_shot := 3,
      reggie_points := reggie_layups * points_per_layup + reggie_free_throws * points_per_free_throw + reggie_long_shots * points_per_long_shot,
      brother_points := brother_long_shots * points_per_long_shot
  in brother_points - reggie_points = 2 := 
by
  sorry

end reggie_loses_by_2_points_l210_210158


namespace log_sum_eq_five_implies_y_l210_210162

theorem log_sum_eq_five_implies_y (y : ℝ) (h : log 3 y + log 9 y = 5) : y = 3^(10/3) :=
sorry

end log_sum_eq_five_implies_y_l210_210162


namespace complex_abs_sqrt_five_l210_210318

open Complex

theorem complex_abs_sqrt_five : abs (2 + (-1 : ℂ) + 2 * (-I : ℂ)) = Real.sqrt 5 := 
by
  sorry

end complex_abs_sqrt_five_l210_210318


namespace number_of_three_digit_integers_to_satisfy_congruence_l210_210739

theorem number_of_three_digit_integers_to_satisfy_congruence : 
  (finset.card {y : ℕ | 100 ≤ y ∧ y ≤ 999 ∧ (4325 * y + 692) % 17 = 1403 % 17}) = 53 :=
by {
  sorry
}

end number_of_three_digit_integers_to_satisfy_congruence_l210_210739


namespace part_a_part_b_part_c_l210_210782

-- Define S(n)
def S (n : ℕ) : ℕ := sorry -- S(n) defined as the greatest integer such that for every k ≤ S(n), n^2 can be the sum of squares of k positive integers.

-- Problem Part (a)
theorem part_a (n : ℕ) (h : n ≥ 4) : S(n) ≤ n^2 - 14 := sorry

-- Problem Part (b)
theorem part_b : ∃ n : ℕ, n > 0 ∧ S(n) = n^2 - 14 := 
-- Let's take n = 13 based on the earlier found solution:
⟨13, nat.zero_lt_succ _, sorry⟩

-- Problem Part (c)
theorem part_c : ∃ infinite (n : ℕ), S(n) = n^2 - 14 :=
begin
  use {n : ℕ | ∃ k, n = 13 * 2^k},
  sorry
end

end part_a_part_b_part_c_l210_210782


namespace complex_magnitude_l210_210309

variables (i : ℂ)
axiom imaginary_unit : i^2 = -1

theorem complex_magnitude : |2 + i^2 + 2 * i^3| = Real.sqrt 5 :=
by
  have h1 : i^3 = i^2 * i := by sorry
  have h2 : i^2 = -1 := imaginary_unit
  have h3 : i^3 = -i := by sorry
  calc 
    |2 + i^2 + 2 * i^3| = |2 + (-1) + 2 * (-i)| : by sorry
    ... = |1 - 2 * i| : by sorry
    ... = Real.sqrt (1^2 + (-2)^2) : by sorry
    ... = Real.sqrt 5 : by sorry

end complex_magnitude_l210_210309


namespace num_roots_2013_iter_f_eq_half_l210_210467

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) - 2

-- Predicate that defines the repeated application of f
def iter (n : ℕ) (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  nat.iterate n f x

-- The main theorem
theorem num_roots_2013_iter_f_eq_half : 
  ∃ n : ℕ, n = 2013 ∧ (finset.card { x : ℝ | iter n f x = 1 / 2 }.to_finset) = 4030 :=
sorry

end num_roots_2013_iter_f_eq_half_l210_210467


namespace total_amount_is_218_l210_210256

structure Ticket :=
  (cost : ℕ)
  (winning_numbers : ℕ)

def worth_of_winning_number (i : ℕ) : ℕ :=
  if i = 1 ∨ i = 2 then 15 else 20

def net_amount (ticket : Ticket) : ℕ :=
  let total_payout := (List.range ticket.winning_numbers).sum (λ i, worth_of_winning_number (i+1))
  total_payout - ticket.cost

def Ticket1 := ⟨5, 3⟩
def Ticket2 := ⟨7, 5⟩
def Ticket3 := ⟨4, 2⟩
def Ticket4 := ⟨6, 4⟩

def total_net_amount : ℕ :=
  net_amount Ticket1 + net_amount Ticket2 + net_amount Ticket3 + net_amount Ticket4

theorem total_amount_is_218 :
  total_net_amount = 218 :=
by
  -- proof goes here
  sorry

end total_amount_is_218_l210_210256


namespace discriminant_of_quadratic_l210_210414

theorem discriminant_of_quadratic :
  let a := (5 : ℚ)
  let b := (5 + 1/5 : ℚ)
  let c := (1/5 : ℚ)
  let Δ := b^2 - 4 * a * c
  Δ = 576 / 25 :=
by
  sorry

end discriminant_of_quadratic_l210_210414


namespace no_equilateral_triangle_with_integer_coordinates_ratio_distance_is_rational_l210_210804

-- Part (a) - No equilateral triangle with integer coordinates
theorem no_equilateral_triangle_with_integer_coordinates:
  ∀ (a b c : ℤ × ℤ), (a ≠ b ∧ b ≠ c ∧ c ≠ a) → 
  (dist a b = dist b c ∧ dist b c = dist c a ∧ dist c a = dist a b) → 
  false := 
sorry

-- Part (b) - Rationality of the ratio d(A, BC) / BC
theorem ratio_distance_is_rational
  (A B C : ℤ × ℤ) 
  (hAB_AC: dist A B = dist A C) : 
  ∃ q : ℚ, q = (euclideanDistance A (lineThrough B C)) / (euclideanDistance B C) := 
sorry

end no_equilateral_triangle_with_integer_coordinates_ratio_distance_is_rational_l210_210804


namespace num_people_in_park_at_11am_l210_210253

noncomputable def num_people_enter (n : ℕ) : ℕ :=
  if n = 1 then 2 else 2^(n - 1) + 1 - 1

noncomputable def calc_people (m : ℕ) : ℕ :=
  let seq_sum : ℕ → ℕ
  | 0     => 0
  | k + 1 => seq_sum k + num_people_enter (k + 1)
  seq_sum m

theorem num_people_in_park_at_11am : calc_people 25 = 2^25 + 24 :=
by
  sorry

end num_people_in_park_at_11am_l210_210253


namespace simba_pie_share_l210_210376

theorem simba_pie_share :
  let pie_left := (8 : ℚ) / 9
  let num_people := 4
  let simba_share := pie_left / num_people
  simba_share = (2 : ℚ) / 9 := 
by 
  let pie_left := (8 : ℚ) / 9
  let num_people := 4
  let simba_share := pie_left / num_people
  have h1 : simba_share = (8 / 9) / 4 := rfl
  have h2 : (8 / 9) / 4 = (8 / 9) * (1 / 4) := by rw div_eq_mul_inv
  have h3 : (8 / 9) * (1 / 4) = (8 * 1) / (9 * 4) := by rw mul_div_assoc
  have h4 : (8 * 1) / (9 * 4) = 8 / 36 := by norm_num
  have h5 : 8 / 36 = 2 / 9 := by norm_num
  show simba_share = 2 / 9 from h1.trans (h2.trans (h3.trans (h4.trans h5)))

end simba_pie_share_l210_210376


namespace johns_score_is_82_l210_210880

theorem johns_score_is_82 :
  ∃ (x : ℕ), 2 ∣ x ∧ x % 5 = 0 ∧ x % 3 = 2 ∧ x = 82 :=
by
  use 82
  split
  -- x is even
  exact by norm_num
  split
  -- divisible by 5
  exact by norm_num
  split
  -- mod 3 is 2
  exact by norm_num
  sorry

end johns_score_is_82_l210_210880


namespace find_fx_value_l210_210198

theorem find_fx_value 
  (ω : ℝ) (hω_pos : ω > 0)
  (h_dist : (∀ n : ℤ, (tan (ω * (n + 1) * ω⁻¹)) = 3 → (tan (ω * n * ω⁻¹)) = 3 → (ω⁻¹ = π / 4))) :
   tan (4 * (π / 12)) = sqrt 3 :=
by sorry

end find_fx_value_l210_210198


namespace arithmetic_sum_sequences_l210_210822

theorem arithmetic_sum_sequences (a b : ℕ → ℕ) (h1 : ∀ n, a n = a 0 + n * (a 1 - a 0)) (h2 : ∀ n, b n = b 0 + n * (b 1 - b 0)) (h3 : a 2 + b 2 = 3) (h4 : a 4 + b 4 = 5): a 7 + b 7 = 8 := by
  sorry

end arithmetic_sum_sequences_l210_210822


namespace maximize_dragon_resilience_l210_210753

noncomputable def p_s (x : ℝ) (s : ℕ) : ℝ :=
  x^s / (1 + x + x^2)

def K : List ℕ :=
  [1, 2, 2, 1, 0, 2, 1, 0, 1, 2]

def P_K (x : ℝ) : ℝ :=
  List.foldr (λ s acc => acc * p_s x s) 1 K

theorem maximize_dragon_resilience :
  let x_opt := (Real.sqrt 97 + 1) / 8 in
  ∀ x > 0, P_K x ≤ P_K x_opt :=
sorry

end maximize_dragon_resilience_l210_210753


namespace smallest_y_l210_210267

theorem smallest_y (y : ℕ) :
  (y > 0 ∧ 800 ∣ (540 * y)) ↔ (y = 40) :=
by
  sorry

end smallest_y_l210_210267


namespace common_tangent_length_l210_210257

noncomputable def length_of_common_tangent {r m r1 r2 : ℝ} : ℝ := 
  m / r * sqrt ((r + r1) * (r + r2))

theorem common_tangent_length
  {r m r1 r2 : ℝ}
  (hr : 0 < r)
  (hm : 0 < m)
  (hr1 : 0 < r1)
  (hr2 : 0 < r2)
  (ext_tangency : (m = (r + r1) ∨ m = (r - r1)) ∧ (m = (r + r2) ∨ m = (r - r2))):
  length_of_common_tangent r m r1 r2 = (m / r) * sqrt ((r + r1) * (r + r2)) :=
sorry

end common_tangent_length_l210_210257


namespace maximum_a_inequality_l210_210870

theorem maximum_a_inequality :
  ∀ (n : ℕ), 0 < n →
  (∑ i in finset.range(n+1), (1 / (n + 1 + i : ℝ))) > 25 / 24 :=
sorry

end maximum_a_inequality_l210_210870


namespace greatest_num_fruit_in_each_basket_l210_210144

theorem greatest_num_fruit_in_each_basket : 
  let oranges := 15
  let peaches := 9
  let pears := 18
  let gcd := Nat.gcd (Nat.gcd oranges peaches) pears
  gcd = 3 :=
by
  sorry

end greatest_num_fruit_in_each_basket_l210_210144


namespace complex_abs_sqrt_five_l210_210317

open Complex

theorem complex_abs_sqrt_five : abs (2 + (-1 : ℂ) + 2 * (-I : ℂ)) = Real.sqrt 5 := 
by
  sorry

end complex_abs_sqrt_five_l210_210317


namespace total_games_in_tournament_l210_210669

theorem total_games_in_tournament 
  (classes_grade_one : ℕ)
  (classes_grade_two : ℕ)
  (classes_grade_three : ℕ)
  (formula_for_games : ∀ n, nat.choose n 2)
  (games_grade_one : formula_for_games 5 = 10)
  (games_grade_two : formula_for_games 8 = 28)
  (games_grade_three : formula_for_games 3 = 3) : 
  classes_grade_one = 5 → 
  classes_grade_two = 8 → 
  classes_grade_three = 3 → 
  (formula_for_games classes_grade_one + formula_for_games classes_grade_two + formula_for_games classes_grade_three) = 41 := 
by 
  intros h1 h2 h3 
  sorry

end total_games_in_tournament_l210_210669


namespace perfect_cubes_and_fourths_below_1000_l210_210063

def is_perfect_cube (n: ℕ) : Prop :=
  ∃ k: ℕ, k ^ 3 = n

def is_perfect_fourth (n: ℕ) : Prop :=
  ∃ k: ℕ, k ^ 4 = n

def is_perfect_twelfth (n: ℕ) : Prop :=
  ∃ k: ℕ, k ^ 12 = n

def count_perfect_cubes_below (limit: ℕ) : ℕ :=
  Finset.card (Finset.filter (λ n, is_perfect_cube n) (Finset.range limit))

def count_perfect_fourths_below (limit: ℕ) : ℕ :=
  Finset.card (Finset.filter (λ n, is_perfect_fourth n) (Finset.range limit))

def count_perfect_twelfths_below (limit: ℕ) : ℕ :=
  Finset.card (Finset.filter (λ n, is_perfect_twelfth n) (Finset.range limit))

theorem perfect_cubes_and_fourths_below_1000 : 
  let total := count_perfect_cubes_below 1000 + count_perfect_fourths_below 1000 - count_perfect_twelfths_below 1000 in
  total = 14 :=
by sorry

end perfect_cubes_and_fourths_below_1000_l210_210063


namespace opposite_of_five_is_neg_five_l210_210213

theorem opposite_of_five_is_neg_five :
  ∃ (x : ℤ), (5 + x = 0) ∧ x = -5 :=
by
  use -5
  split
  · simp
  · rfl

end opposite_of_five_is_neg_five_l210_210213


namespace arcsin_one_eq_pi_div_two_l210_210721

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = Real.pi / 2 :=
by
  -- proof steps here
  sorry

end arcsin_one_eq_pi_div_two_l210_210721


namespace find_f_neg_a_l210_210661

noncomputable def f (x : ℝ) : ℝ := x^3 * Real.cos x + 1

variable (a : ℝ)

-- Given condition
axiom h_fa : f a = 11

-- Statement to prove
theorem find_f_neg_a : f (-a) = -9 :=
by
  sorry

end find_f_neg_a_l210_210661


namespace opposite_of_five_l210_210217

theorem opposite_of_five : -5 = -5 :=
by
sorry

end opposite_of_five_l210_210217


namespace mart_income_percentage_l210_210648

theorem mart_income_percentage 
  (J T M : ℝ)
  (h1 : M = 1.60 * T)
  (h2 : T = 0.60 * J) :
  M = 0.96 * J :=
sorry

end mart_income_percentage_l210_210648


namespace johns_donation_l210_210647

theorem johns_donation (A J : ℝ) 
  (h1 : (75 / 1.5) = A) 
  (h2 : A * 2 = 100)
  (h3 : (100 + J) / 3 = 75) : 
  J = 125 :=
by 
  sorry

end johns_donation_l210_210647


namespace min_value_expression_l210_210537

-- Variables
variables {a b : ℝ}

-- Conditions
def positive_real (x : ℝ) : Prop := x > 0

def condition1 := positive_real a
def condition2 := positive_real b
def condition3 := 2 * a + 3 * b = 1

-- Theorem stating the minimum value
theorem min_value_expression :
  condition1 → condition2 → condition3 → 
  ∃ x : ℝ, x = 26 ∧ (∀ p, p = (2 / a + 3 / b) → p ≥ x) :=
by
  sorry

end min_value_expression_l210_210537


namespace exists_point_X_on_circle_bisects_EF_l210_210805

theorem exists_point_X_on_circle_bisects_EF
  (circle : Type)
  (is_circle : is_circle circle)
  (A B C D J : circle)
  (hAB : chord is_circle A B)
  (hCD : chord is_circle C D)
  (hJ : point_on_chord J is_circle C D)
  :
  ∃ X : circle, ∃ E F : circle, 
  chord_intersection AX CD E ∧ 
  chord_intersection BX CD F ∧ 
  midpoint J E F :=
sorry

end exists_point_X_on_circle_bisects_EF_l210_210805


namespace parallel_a_b_projection_a_onto_b_l210_210847

noncomputable section

open Real

def a : ℝ × ℝ := (sqrt 3, 1)
def b (θ : ℝ) : ℝ × ℝ := (cos θ, sin θ)

theorem parallel_a_b (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π) (h_parallel : (a.1 / a.2) = (b θ).1 / (b θ).2) : θ = π / 6 := sorry

theorem projection_a_onto_b (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π) (h_proj : (sqrt 3 * cos θ + sin θ) = -sqrt 3) : b θ = (-1, 0) := sorry

end parallel_a_b_projection_a_onto_b_l210_210847


namespace range_of_k_l210_210466

noncomputable def is_monotonically_decreasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ x y ∈ I, x < y → f x ≥ f y

theorem range_of_k (k : ℝ) :
  is_monotonically_decreasing (λ x, k * Real.cos (k * x)) (set.Ioo (Real.pi / 4) (Real.pi / 3)) →
  k ∈ set.Icc (-6) (-4) ∪ set.Ioc 0 3 ∪ set.Icc 8 9 ∪ {-12} :=
sorry

end range_of_k_l210_210466


namespace like_terms_expressions_l210_210827

theorem like_terms_expressions (m n : ℤ) :
  (∀ x y : ℝ, -3 * x ^ (m - 1) * y ^ 3 = 4 * x * y ^ (m + n)) → (m = 2 ∧ n = 1) :=
by
  intro h
  have h_mx_pow : m - 1 = 1 := sorry
  have h_my_pow : 3 = m + n := sorry
  finish

end like_terms_expressions_l210_210827


namespace volume_KLMN_of_SABC_centers_l210_210982

def volume_tetrahedron (a b c : ℝ) : ℝ :=
  let x := sqrt ((a^2 + c^2 - b^2) / 2)
  let y := sqrt ((b^2 + c^2 - a^2) / 2)
  let z := sqrt ((a^2 + b^2 - c^2) / 2)
  (sqrt 2 / 12) * sqrt ((a^2 + c^2 - b^2) * (b^2 + c^2 - a^2) * (a^2 + b^2 - c^2))

def volume_KLMN (a b c : ℝ) (volume_SABC : ℝ) : ℝ :=
  let numerator := (b + c - a) * (c + a - b) * (a + b - c)
  let denominator := (a + b + c)^3
  numerator / denominator * volume_SABC

theorem volume_KLMN_of_SABC_centers
  (a b c : ℝ)
  (h_a : a = 8)
  (h_b : b = 7)
  (h_c : c = 5) :
  abs (volume_KLMN a b c (volume_tetrahedron a b c) - 0.66) < 0.01 :=
by
  -- Substitute the given edge values
  have hab := volume_tetrahedron 8 7 5
  -- compute volume_KLMN
  have hklmn := volume_KLMN 8 7 5 hab
  -- Now assert that they are approximately equal to 0.66
  calc 
    abs (hklmn - 0.66) < 0.01 := sorry

end volume_KLMN_of_SABC_centers_l210_210982


namespace hannah_stocking_stuffers_l210_210059

theorem hannah_stocking_stuffers (candy_caness : ℕ) (beanie_babies : ℕ) (books : ℕ) (kids : ℕ) : 
  candy_caness = 4 → 
  beanie_babies = 2 → 
  books = 1 → 
  kids = 3 → 
  candy_caness + beanie_babies + books = 7 → 
  7 * kids = 21 := 
by sorry

end hannah_stocking_stuffers_l210_210059


namespace least_distinct_values_l210_210681

theorem least_distinct_values
  (n total : ℕ) (numbers : list ℕ)
  (mode_count : ℕ) (unique_mode : ∀ x ∈ numbers, x ≠ mode_count → count numbers x < mode_count) :
  total = 3000 → mode_count = 12 → count numbers mode_count = 12 → 
  nodup numbers → multi_set.card numbers = total →
  n = 273 := sorry

end least_distinct_values_l210_210681


namespace caesar_cipher_WIN_shift_4_l210_210582

def alphabet := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def caesar_cipher (shift : ℕ) (msg : String) : String :=
  String.mk (msg.data.map (λ c => alphabet.get? ((alphabet.indexOf c + shift) % alphabet.length)).iget)

theorem caesar_cipher_WIN_shift_4 :
  caesar_cipher 4 "WIN" = "AMR" :=
by { unfold caesar_cipher, norm_num, sorry }

end caesar_cipher_WIN_shift_4_l210_210582


namespace integral_x_minus_sin_x_l210_210405

theorem integral_x_minus_sin_x :
  ∫ x in 0..π, (x - sin x) = (π^2 / 2) - 2 :=
by
  sorry

end integral_x_minus_sin_x_l210_210405


namespace pyramid_cross_section_area_is_correct_l210_210972

noncomputable def area_of_cross_section (A B : ℝ) (V_half : ℝ) : ℝ :=
  (1 / 2) * (A + B) - ((1 / 2) * (B * (B / A)^(1 / 3)))

theorem pyramid_cross_section_area_is_correct :
  ∀ (A B : ℝ), A = 8 → B = 1 →
  (area_of_cross_section A B (1/2 * (A + B))) = (16 * real.sqrt 2 + 1 / 2)^(2 / 3) := 
by
  intros,
  sorry

end pyramid_cross_section_area_is_correct_l210_210972


namespace derivative_of_function_y_l210_210774

noncomputable def function_y (x : ℝ) : ℝ := (x^2) / (x + 3)

theorem derivative_of_function_y (x : ℝ) :
  deriv function_y x = (x^2 + 6 * x) / ((x + 3)^2) :=
by 
  -- sorry since the proof is not required
  sorry

end derivative_of_function_y_l210_210774


namespace maximal_pawns_placement_l210_210944

theorem maximal_pawns_placement :
  ∃ (p : ℕ), p = 1009^2 ∧
  ∀ (chessboard : ℕ × ℕ), chessboard = (2019, 2019) →
  ∀ (empty_between : ℕ → ℕ → Prop), 
  (∀ r1 r2 : ℕ, r1 ≠ r2 → empty_between r1 r2) →
  (∀ (rooks pawns : ℕ), rooks = p + 2019 → pawns = p →
  (∃ placement : (ℕ × ℕ) → option (bool), 
    (∀ i j, i ≠ j → (placement (i, j) ≠ some true ∨ empty_between i j)) ∧
    (∑ ij, match placement ij with | some true => 1 | _ => 0 end) = rooks ∧
    (∑ ij, match placement ij with | some false => 1 | _ => 0 end) = pawns)) :=
sorry

end maximal_pawns_placement_l210_210944


namespace problem1_problem2_l210_210380

-- Problem 1: Proving the given equation under specified conditions
theorem problem1 (x y : ℝ) (h : x + y ≠ 0) : ((2 * x + 3 * y) / (x + y)) - ((x + 2 * y) / (x + y)) = 1 :=
sorry

-- Problem 2: Proving the given equation under specified conditions
theorem problem2 (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ 1) : ((a^2 - 1) / (a^2 - 4 * a + 4)) / ((a - 1) / (a - 2)) = (a + 1) / (a - 2) :=
sorry

end problem1_problem2_l210_210380


namespace determine_c_l210_210400

theorem determine_c (c : ℝ) :
  let vertex_x := -(-10 / (2 * 1))
  let vertex_y := c - ((-10)^2 / (4 * 1))
  ((5 - 0)^2 + (vertex_y - 0)^2 = 10^2)
  → (c = 25 + 5 * Real.sqrt 3 ∨ c = 25 - 5 * Real.sqrt 3) :=
by
  sorry

end determine_c_l210_210400


namespace textbook_order_total_cost_l210_210699

theorem textbook_order_total_cost :
  let english_quantity := 35
  let geography_quantity := 35
  let mathematics_quantity := 20
  let science_quantity := 30
  let english_price := 7.50
  let geography_price := 10.50
  let mathematics_price := 12.00
  let science_price := 9.50
  (english_quantity * english_price + geography_quantity * geography_price + mathematics_quantity * mathematics_price + science_quantity * science_price = 1155.00) :=
by sorry

end textbook_order_total_cost_l210_210699


namespace correct_statement_l210_210011

universe u
variable (α : Type u)

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5}

noncomputable def complement_U_M : Set ℕ := {1, 3}

noncomputable def M : Set ℕ := U α \ complement_U_M

theorem correct_statement : 2 ∈ M :=
by {
  sorry
}

end correct_statement_l210_210011


namespace food_last_after_join_l210_210247

-- Define the conditions
def initial_men := 760
def additional_men := 2280
def initial_days := 22
def days_before_join := 2
def initial_food := initial_men * initial_days
def remaining_food := initial_food - (initial_men * days_before_join)
def total_men := initial_men + additional_men

-- Define the goal to prove
theorem food_last_after_join :
  (remaining_food / total_men) = 5 :=
by
  sorry

end food_last_after_join_l210_210247


namespace odd_function_f_f_negative_l210_210450

noncomputable def f (x : ℝ) : ℝ :=
  if hx : 0 < x then x + Real.log x else
  if hx : x < 0 then x - Real.log (-x) else 0

theorem odd_function_f (x : ℝ) :
  f(-x) = -f(x) :=
by
  sorry

theorem f_negative (x : ℝ) (h : x < 0) :
  f(x) = x - Real.log (-x) :=
by
  sorry

end odd_function_f_f_negative_l210_210450


namespace Q1_Q2_l210_210388

variables {x : ℕ → ℝ} 
(hx : ∀ n, 0 ≤ x n ∧ x n < 1)

theorem Q1 : (∃ N : ℕ, ∀ m ≥ N,  x m ∈ Ico 0 (1/2)) ∨ 
             (∃ N : ℕ, ∀ m ≥ N, x m ∈ Ico (1/2) 1) :=
begin
  sorry
end

theorem Q2 (n : ℕ) (hn : n ≥ 1) : ∃ k ∈ fin (2^n), ∃ N : ℕ, ∀ m ≥ N, x m ∈ Ico (k / 2^n : ℝ) ((k + 1) / 2^n) :=
begin
  sorry
end

end Q1_Q2_l210_210388


namespace sequence_properties_l210_210608

noncomputable def S (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in List.range n, a i

theorem sequence_properties (a : ℕ → ℕ) (S_a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a (n + 1) = 2 * S_a n) :
  (∀ n, S_a n = S a n) →
  (∀ n, a n = (if n = 1 then 1 else 2 * 3 ^ (n - 2))) :=
by
  -- placeholder for the proof
  sorry

end sequence_properties_l210_210608


namespace coin_value_is_630_l210_210879

theorem coin_value_is_630 :
  (∃ x : ℤ, x > 0 ∧ 406 * x = 63000) :=
by {
  sorry
}

end coin_value_is_630_l210_210879


namespace min_n_for_log_sum_gt_5_l210_210799

open Real

-- Given sequence
def a (n : ℕ) : ℕ
| 0       := 1
| (n + 1) := (n + 1)

noncomputable def S (n : ℕ) : ℕ := ∑ i in range (n + 1), a i

theorem min_n_for_log_sum_gt_5 :
  ∃ n : ℕ, (∑ k in range (n + 1), log (1 + 1 / (a k) : ℝ) / log 2 > 5) ∧ n = 32 :=
sorry

end min_n_for_log_sum_gt_5_l210_210799


namespace total_crayons_l210_210994

theorem total_crayons (orange_boxes : ℕ) (orange_per_box : ℕ) (blue_boxes : ℕ) (blue_per_box : ℕ) (red_boxes : ℕ) (red_per_box : ℕ) : 
  orange_boxes = 6 → orange_per_box = 8 → 
  blue_boxes = 7 → blue_per_box = 5 →
  red_boxes = 1 → red_per_box = 11 → 
  orange_boxes * orange_per_box + blue_boxes * blue_per_box + red_boxes * red_per_box = 94 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  exact sorry

end total_crayons_l210_210994


namespace complex_magnitude_l210_210307

variables (i : ℂ)
axiom imaginary_unit : i^2 = -1

theorem complex_magnitude : |2 + i^2 + 2 * i^3| = Real.sqrt 5 :=
by
  have h1 : i^3 = i^2 * i := by sorry
  have h2 : i^2 = -1 := imaginary_unit
  have h3 : i^3 = -i := by sorry
  calc 
    |2 + i^2 + 2 * i^3| = |2 + (-1) + 2 * (-i)| : by sorry
    ... = |1 - 2 * i| : by sorry
    ... = Real.sqrt (1^2 + (-2)^2) : by sorry
    ... = Real.sqrt 5 : by sorry

end complex_magnitude_l210_210307


namespace true_propositions_count_l210_210406

-- Conditions
variables (a b c d : ℝ)
variable h1 : a = b
variable h2 : c = d

-- Proposition definitions
def original_proposition := a = b ∧ c = d → a + c = b + d
def contrapositive_proposition := a + c ≠ b + d → a ≠ b ∨ c ≠ d
def converse_proposition := a + c = b + d → a = b ∧ c = d
def inverse_proposition := a ≠ b ∨ c ≠ d → a + c ≠ b + d

-- Mathematical proof problem statement: Prove that there are exactly 2 true propositions
theorem true_propositions_count : 
  (original_proposition (a b c d) h1 h2 → true) ∧ 
  (contrapositive_proposition (a b c d) → true) ∧ 
  (converse_proposition (a b c d) → false) ∧ 
  (inverse_proposition (a b c d) → false) → 
  (2 = 2) :=
by sorry

end true_propositions_count_l210_210406


namespace stocking_stuffers_total_l210_210055

theorem stocking_stuffers_total 
  (candy_canes_per_child beanie_babies_per_child books_per_child : ℕ)
  (num_children : ℕ)
  (h1 : candy_canes_per_child = 4)
  (h2 : beanie_babies_per_child = 2)
  (h3 : books_per_child = 1)
  (h4 : num_children = 3) :
  candy_canes_per_child + beanie_babies_per_child + books_per_child * num_children = 21 :=
by
  sorry

end stocking_stuffers_total_l210_210055


namespace midpoint_of_segment_AE_l210_210910

noncomputable def circle : Type := sorry
noncomputable def line : Type := sorry
noncomputable def tangent_point : circle → line → Type := sorry
noncomputable def intersects : circle → line → set point := sorry
noncomputable def tangent_line : circle → point → line := sorry
noncomputable def segment_midpoint : point → point → point → Prop := sorry

theorem midpoint_of_segment_AE
    (l1 l2 : line)
    (l1_parallel_l2 : l1 ∥ l2)
    (ω : circle)
    (A : tangent_point ω l1)
    (BC : intersects ω l2)
    (B C : point)
    (B_in_BC : B ∈ BC)
    (C_in_BC : C ∈ BC)
    (tangent_C : tangent_line ω C = l1)
    (D : intersects tangent_line ω C l1)
    (BD : line)
    (E : intersects_line_circle BD ω)
    (F : intersects_line_line (ce_line C E) l1)
  : segment_midpoint F A E := 
sorry

end midpoint_of_segment_AE_l210_210910


namespace sequence_general_term_l210_210842

theorem sequence_general_term (a : ℕ → ℤ) (h1 : a 1 = 1) (h_rec : ∀ n, a (n + 1) = 2 * a n + 1) :
  ∀ n, a n = (2 ^ n) - 1 := 
sorry

end sequence_general_term_l210_210842


namespace centroid_of_quadrilateral_l210_210767

variables {A B C D : ℝˣ}

/--
Given a quadrilateral with vertices A, B, C, D,
and equal masses placed at each vertex, 
the centroid of these masses is given by the average of the position vectors of A, B, C, and D.
-/
theorem centroid_of_quadrilateral (A B C D : ℝˣ) :
  (A + B + C + D) / 4 = ∑ x in [A, B, C, D], x / 4 :=
by 
  sorry

end centroid_of_quadrilateral_l210_210767


namespace domain_of_function_y_eq_sqrt_2x_3_div_x_2_l210_210741

def domain (x : ℝ) : Prop :=
  (2 * x - 3 ≥ 0) ∧ (x ≠ 2)

theorem domain_of_function_y_eq_sqrt_2x_3_div_x_2 :
  ∀ x : ℝ, domain x ↔ ((x ≥ 3 / 2) ∧ (x ≠ 2)) :=
by
  sorry

end domain_of_function_y_eq_sqrt_2x_3_div_x_2_l210_210741


namespace radium_decay_heat_coal_equivalence_l210_210932

-- Definitions for given conditions
def radium_heat_equivalence : ℕ := 375000 -- Heat equivalence of 1 kg radium in kg of coal
def earth_radium_mass : ℕ := 10^10 -- Total mass of radium in Earth's crust

-- Proof problem statement
theorem radium_decay_heat_coal_equivalence :
  (earth_radium_mass * radium_heat_equivalence) = 3.75 * 10^15 := 
by
  sorry

end radium_decay_heat_coal_equivalence_l210_210932


namespace equilateral_triangle_l210_210524

theorem equilateral_triangle (A B C E D : Type) [Inhabited A] [Inhabited B] [Inhabited C]
  (AB AC : A → B) (AD CE : A → C) (ECA BAD : C → A)
  (hAD_CE : AD = CE) (hBAD_ECA : BAD = ECA) (hADC_BEC : ∀ c : C, ∃ b : B, true) :
  ∃ (α : ℝ), α = 60 ∧ (∀ (A B C : Type), A = B ∧ B = C ∧ C = 60) :=
by sorry

end equilateral_triangle_l210_210524


namespace problem_equivalence_l210_210921

noncomputable def p (x : ℝ) : ℝ :=
if floor x = 2 ∨ floor x = 3 ∨ floor x = 5 ∨ floor x = 7 ∨ floor x = 11 ∨ floor x = 13 then
  x + 2
else if floor x = 4 ∨ floor x = 9 ∨ floor x = 16 then
  x + 5
else
  let y := if floor x = 6 then 3 else if floor x = 8 then 7 else if floor x = 10 ∨ floor x = 12 then 3 else if floor x = 14 then 7 else if floor x = 15 then 5 else 2 in
  p y + (x + 2 - floor x)

def range_p := [4, 10) ∪ [13, 16] ∪ {21}

theorem problem_equivalence : 
  ∀ x : ℝ, 2 ≤ x ∧ x ≤ 15 → p x ∈ range_p :=
sorry

end problem_equivalence_l210_210921


namespace ratio_of_boys_to_girls_l210_210942

theorem ratio_of_boys_to_girls (B G M : ℤ) 
    (hB_avg : ∀ b, b / B = 90) 
    (hG_avg : ∀ g, g / G = 96) 
    (hM_score : M = 3) 
    (hM_avg : ∀ m, m / M = 92) 
    (hOverall_avg : (90 * B + 96 * G + 92 * M) / (B + G + M) = 94) :
  B.to_rat / G.to_rat = 1 / 5 := 
by
  sorry

end ratio_of_boys_to_girls_l210_210942


namespace find_f_2_l210_210453

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x) = -f (-x)

def given_function (x : ℝ) : ℝ :=
  if x < 0 then 2 ^ (-x) + x ^ 2 else 0 

theorem find_f_2 :
  is_odd_function given_function →
  f(2) = -8 :=
  by
  sorry

end find_f_2_l210_210453


namespace negative_is_always_less_than_positive_l210_210592

theorem negative_is_always_less_than_positive
  (h : ∀ (a b : ℝ), a < 0 → b > 0 → a < b) :
  (∀ (a b : ℝ), a < 0 → b > 0 → "certain") :=
by
  intros a b ha hb
  exact "certain"

end negative_is_always_less_than_positive_l210_210592


namespace largest_angle_measure_l210_210978

def hexagon_internal_angles_sum : ℝ := 720

def ratio_angles (x : ℝ) : list ℝ := [2*x, 3*x, 3*x, 4*x, 5*x, 6*x]

theorem largest_angle_measure :
  ∃ x : ℝ, sum (ratio_angles x) = hexagon_internal_angles_sum ∧ 
          6 * (hexagon_internal_angles_sum / 23) = 4320 / 23 :=
sorry

end largest_angle_measure_l210_210978


namespace simplify_and_evaluate_expression_l210_210955

noncomputable def a : ℝ := real.sqrt 2 - 1

theorem simplify_and_evaluate_expression :
  (1 + 1 / (a - 1)) / (a / (a^2 - 1)) = real.sqrt 2 := by
  sorry

end simplify_and_evaluate_expression_l210_210955


namespace complex_magnitude_l210_210305

variables (i : ℂ)
axiom imaginary_unit : i^2 = -1

theorem complex_magnitude : |2 + i^2 + 2 * i^3| = Real.sqrt 5 :=
by
  have h1 : i^3 = i^2 * i := by sorry
  have h2 : i^2 = -1 := imaginary_unit
  have h3 : i^3 = -i := by sorry
  calc 
    |2 + i^2 + 2 * i^3| = |2 + (-1) + 2 * (-i)| : by sorry
    ... = |1 - 2 * i| : by sorry
    ... = Real.sqrt (1^2 + (-2)^2) : by sorry
    ... = Real.sqrt 5 : by sorry

end complex_magnitude_l210_210305


namespace same_terminal_side_angle_l210_210519

theorem same_terminal_side_angle (θ : ℤ) : θ = -390 → ∃ k : ℤ, 0 ≤ θ + k * 360 ∧ θ + k * 360 < 360 ∧ θ + k * 360 = 330 :=
  by
    sorry

end same_terminal_side_angle_l210_210519


namespace opposite_of_5_is_neg5_l210_210219

def opposite (n x : ℤ) := n + x = 0

theorem opposite_of_5_is_neg5 : opposite 5 (-5) :=
by
  sorry

end opposite_of_5_is_neg5_l210_210219


namespace chess_tournament_time_spent_l210_210945

theorem chess_tournament_time_spent (games : ℕ) (moves_per_game : ℕ)
  (opening_moves : ℕ) (middle_moves : ℕ) (endgame_moves : ℕ)
  (polly_opening_time : ℝ) (peter_opening_time : ℝ)
  (polly_middle_time : ℝ) (peter_middle_time : ℝ)
  (polly_endgame_time : ℝ) (peter_endgame_time : ℝ)
  (total_time_hours : ℝ) :
  games = 4 →
  moves_per_game = 38 →
  opening_moves = 12 →
  middle_moves = 18 →
  endgame_moves = 8 →
  polly_opening_time = 35 →
  peter_opening_time = 45 →
  polly_middle_time = 30 →
  peter_middle_time = 45 →
  polly_endgame_time = 40 →
  peter_endgame_time = 60 →
  total_time_hours = (4 * ((12 * 35 + 18 * 30 + 8 * 40) + (12 * 45 + 18 * 45 + 8 * 60))) / 3600 :=
sorry

end chess_tournament_time_spent_l210_210945


namespace walnut_trees_initial_count_l210_210998

theorem walnut_trees_initial_count (x : ℕ) (h : x + 6 = 10) : x = 4 := 
by
  sorry

end walnut_trees_initial_count_l210_210998


namespace no_intersection_with_x_axis_l210_210499

open Real

theorem no_intersection_with_x_axis (m : ℝ) :
  (∀ x : ℝ, 3 ^ (-(|x - 1|)) + m ≠ 0) ↔ (m ≥ 0 ∨ m < -1) :=
by
  sorry

end no_intersection_with_x_axis_l210_210499


namespace opposite_of_5_is_neg5_l210_210222

def opposite (n x : ℤ) := n + x = 0

theorem opposite_of_5_is_neg5 : opposite 5 (-5) :=
by
  sorry

end opposite_of_5_is_neg5_l210_210222


namespace population_after_4_years_l210_210346

-- Conditions
def initial_population : ℕ := 20
def leaders_each_year : ℕ := 4
def recruits_per_member : ℕ := 3

-- Recursive relation
def next_population (b_k : ℕ) : ℕ := 4 * (b_k - leaders_each_year) + leaders_each_year

-- Main statement
theorem population_after_4_years : ∃ b_4 : ℕ, b_4 = 4100 :=
by
  let b_0 := initial_population
  let b_1 := next_population b_0
  let b_2 := next_population b_1
  let b_3 := next_population b_2
  let b_4 := next_population b_3
  existsi b_4
  rw [next_population, next_population, next_population, next_population]
  -- To satisfy the Lean engine, we use sorry placeholder for steps
  sorry

end population_after_4_years_l210_210346


namespace percentage_reduction_l210_210983

theorem percentage_reduction (P S : ℝ) (h_sales_increase : 1.80 * S) (h_net_effect : 1.53 * P * S)
  (h_eq : P * (1 - x / 100) * 1.80 * S = 1.53 * P * S) : x = 15 :=
sorry

end percentage_reduction_l210_210983


namespace sum_of_coefficients_l210_210596

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem sum_of_coefficients (a b c : ℝ) 
  (h1 : quadratic a b c 3 = 0) 
  (h2 : quadratic a b c 7 = 0)
  (h3 : ∃ x0, (∀ x, quadratic a b c x ≥ quadratic a b c x0) ∧ quadratic a b c x0 = 20) :
  a + b + c = -105 :=
by 
  sorry

end sum_of_coefficients_l210_210596


namespace vasya_wins_l210_210628

-- Definition of the game and players
inductive Player
| Vasya : Player
| Petya : Player

-- Define the problem conditions
structure Game where
  initial_piles : ℕ := 1      -- Initially, there is one pile
  players_take_turns : Bool := true
  take_or_divide : Bool := true
  remove_last_wins : Bool := true
  vasya_first_but_cannot_take_initially : Bool := true

-- Define the function to determine the winner
def winner_of_game (g : Game) : Player :=
  if g.initial_piles = 1 ∧ g.vasya_first_but_cannot_take_initially then Player.Vasya else Player.Petya

-- Define the theorem stating Vasya will win given the game conditions
theorem vasya_wins : ∀ (g : Game), g = {
    initial_piles := 1,
    players_take_turns := true,
    take_or_divide := true,
    remove_last_wins := true,
    vasya_first_but_cannot_take_initially := true
} → winner_of_game g = Player.Vasya := by
  -- Insert proof here
  sorry

end vasya_wins_l210_210628


namespace first_year_sum_of_digits_15_l210_210635

def sumOfDigits (year : Nat) : Nat :=
  (year / 1000) + (year % 1000 / 100) + (year % 100 / 10) + (year % 10)

theorem first_year_sum_of_digits_15 : ∃ (y : Nat), y > 2010 ∧ sumOfDigits y = 15 ∧ ∀ z, z > 2010 → sumOfDigits z = 15 → y ≤ z :=
by {
  use 2049,
  split,
  { exact Nat.lt_succ_of_lt Nat.zero_lt_one },
  split,
  { norm_num },
  { sorry }
}

end first_year_sum_of_digits_15_l210_210635


namespace unique_zero_function_l210_210191

theorem unique_zero_function
    (f : ℝ → ℝ)
    (H : ∀ x y : ℝ, x + y ≠ 0 → f (x * y) = (f x + f y) / (x + y)) :
    ∀ x : ℝ, f x = 0 := 
by 
     sorry

end unique_zero_function_l210_210191


namespace correct_statements_l210_210278

theorem correct_statements :
  (∀ x : ℝ, x > 0 → 2 * x^2 + x + 1 > 2 * x^2 + x + 1 / (x)) ∧
  (∀ x : ℝ, x > 0 → 2 - 3 * x - 4 / x ≤ 2 - 4 * Real.sqrt 3) ∧
  (∀ x : ℝ, x < -1 ∨ x > -1 → (y = 2 / (x + 1)) → DerivFin (f := λ x : ℝ, 2 / (x + 1)) x < 0) ∧
  (∀ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 →
    (x + y) * (y + z) * (z + x) ≥ 8 * x * y * z) := by
  sorry

end correct_statements_l210_210278


namespace proof_2_in_M_l210_210041

def U : Set ℕ := {1, 2, 3, 4, 5}

def M : Set ℕ := { x | x ∈ U ∧ x ≠ 1 ∧ x ≠ 3 }

theorem proof_2_in_M : 2 ∈ M :=
by
  sorry

end proof_2_in_M_l210_210041


namespace evaluate_expression_l210_210861

theorem evaluate_expression (x : ℝ) (h : 3^(2 * x) = 10) : 9^(x + 1) = 90 :=
by sorry

end evaluate_expression_l210_210861


namespace residue_neg_437_mod_13_l210_210397

theorem residue_neg_437_mod_13 : (-437) % 13 = 5 :=
by
  sorry

end residue_neg_437_mod_13_l210_210397


namespace g_neither_even_nor_odd_l210_210111

def g (x : ℝ) : ℝ := floor (2 * x) + (1 / 3)

theorem g_neither_even_nor_odd :
  ¬(∀ x : ℝ, g (-x) = g x) ∧ ¬(∀ x : ℝ, g (-x) = -g x) := by
  sorry

end g_neither_even_nor_odd_l210_210111


namespace zeros_of_composed_function_l210_210456

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else log x / log 2

theorem zeros_of_composed_function :
  {x : ℝ | f(f(x)) + 1 = 0} = {-3, -1/2, 1/4, Real.sqrt 2} :=
by
  sorry

end zeros_of_composed_function_l210_210456


namespace no_real_solutions_l210_210958

theorem no_real_solutions :
  ∀ x y z : ℝ, ¬ (x + y + 2 + 4*x*y = 0 ∧ y + z + 2 + 4*y*z = 0 ∧ z + x + 2 + 4*z*x = 0) :=
by
  sorry

end no_real_solutions_l210_210958


namespace solution_set_of_quadratic_inequality_l210_210988

variable {a x : ℝ} (h_neg : a < 0)

theorem solution_set_of_quadratic_inequality :
  (a * x^2 - (a + 2) * x + 2) ≥ 0 ↔ (x ∈ Set.Icc (2 / a) 1) :=
by
  sorry

end solution_set_of_quadratic_inequality_l210_210988


namespace find_a_l210_210187

noncomputable def quadratic_has_two_distinct_roots (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + a * x + 3 = 0

theorem find_a (a : ℝ) (x1 x2 : ℝ): 
  (∀ x : ℝ, quadratic_has_two_distinct_roots a → x ∈ {x1, x2}) →
  x1^3 - (99 / (2 * x2^2)) = x2^3 - (99 / (2 * x1^2)) →
  a = -6 :=
  sorry

end find_a_l210_210187


namespace dealer_sold_BMWs_l210_210343

theorem dealer_sold_BMWs (total_cars : ℕ) (ford_pct toyota_pct nissan_pct bmw_pct : ℝ)
  (h_total_cars : total_cars = 300)
  (h_ford_pct : ford_pct = 0.1)
  (h_toyota_pct : toyota_pct = 0.2)
  (h_nissan_pct : nissan_pct = 0.3)
  (h_bmw_pct : bmw_pct = 1 - (ford_pct + toyota_pct + nissan_pct)) :
  total_cars * bmw_pct = 120 := by
  sorry

end dealer_sold_BMWs_l210_210343


namespace arcsin_one_eq_pi_div_two_l210_210726

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = Real.pi / 2 := 
by
  sorry

end arcsin_one_eq_pi_div_two_l210_210726


namespace dragon_resilience_l210_210763

noncomputable def probability_function (x : ℝ) : ℝ :=
  let p0 := 1 / (1 + x + x^2)
  let p1 := x / (1 + x + x^2)
  let p2 := x^2 / (1 + x + x^2)
  p1 * p2^2 * p1 * p0 * p2 * p1 * p0 * p1 * p2

theorem dragon_resilience (x : ℝ) (hx : x > 0) : 
  has_max (λ x, probability_function x) ∧ probability_function (sqrt 97 + 1) / 8 = max :=
sorry

end dragon_resilience_l210_210763


namespace tan_double_angle_l210_210813

variable (α : Real) (h1 : α ∈ Ioc (π/2) π) (h2 : cos (3 * π / 2 - α) = -sqrt 3 / 3)

theorem tan_double_angle : tan (2 * α) = - 2 * sqrt 2 := by
  sorry

end tan_double_angle_l210_210813


namespace midpoint_coincides_with_incenter_l210_210672

theorem midpoint_coincides_with_incenter
  (ABC : Triangle)
  (M N : Point)
  (circumcircle_tangent : TangentCircle'
    (TriangleCircumcircle ABC)
    (InCircleTangentToSides ABC AC BC M N))
  (incircle_center : Point := Incenter ABC) : 
  Midpoint M N = incircle_center :=
sorry

end midpoint_coincides_with_incenter_l210_210672


namespace number_of_integers_having_squares_less_than_10_million_l210_210485

theorem number_of_integers_having_squares_less_than_10_million : 
  ∃ n : ℕ, (n = 3162) ∧ (∀ k : ℕ, k ≤ 3162 → (k^2 < 10^7)) :=
by 
  sorry

end number_of_integers_having_squares_less_than_10_million_l210_210485


namespace three_planes_max_division_l210_210621

-- Define the condition: three planes
variable (P1 P2 P3 : Plane)

-- Define the proof statement: three planes can divide the space into at most 8 parts
theorem three_planes_max_division : divides_space_at_most P1 P2 P3 8 :=
  sorry

end three_planes_max_division_l210_210621


namespace determine_angle_GHI_l210_210891

-- Definitions of points and geometric properties
variables (A B C G H I : Type) [Point A B C G H I]

-- Conditions (isosceles triangle ABC)
variable (h1 : AB = AC)
variable (h2 : angle A = 100)

-- Points G, H, I lying on sides BC, CA, AB respectively
variable (G_on_BC : Line BC G) 
variable (H_on_CA : Line CA H)
variable (I_on_AB : Line AB I)

-- CG = CH
variable (h3 : CG = CH)

-- BI = BG
variable (h4 : BI = BG)

-- Goal: Prove that angle GHI = 40 degrees
theorem determine_angle_GHI :
  angle GHI = 40 := excuseNarrator for sor^angleQed

end determine_angle_GHI_l210_210891


namespace min_triangles_with_area_leq_quarter_l210_210505

noncomputable def minimum_good_triangles (A B C D : ℝ × ℝ) (points : set (ℝ × ℝ)) : ℕ :=
  if ∃ (triangle_list : list (ℝ × ℝ × ℝ)), 
    ∀ (T : ℝ × ℝ × ℝ), T ∈ triangle_list → area T ≤ 1 / 4 
  then 2 else 0

theorem min_triangles_with_area_leq_quarter 
    (A B C D : ℝ × ℝ) 
    (h_area_rect : area (A, B, C, D) = 1)
    (points: set (ℝ × ℝ)) 
    (h_points_card : points.card = 5) 
    (h_not_collinear : ∀ (p q r : ℝ × ℝ), p ∈ points → q ∈ points → r ∈ points → 
        ¬(collinear ({p, q, r} : set (ℝ × ℝ)))) 
    : minimum_good_triangles A B C D points = 2 := 
sorry

end min_triangles_with_area_leq_quarter_l210_210505


namespace f_neg_a_l210_210660

-- Definition of the function f
def f (x : ℝ) : ℝ := x^3 * Real.cos x + 1

-- Given condition
variable (a : ℝ)
axiom f_a : f a = 11

-- The goal is to prove f(-a) = -9
theorem f_neg_a : f (-a) = -9 := 
sorry

end f_neg_a_l210_210660


namespace tan_alpha_and_cos_beta_minus_alpha_l210_210831

noncomputable def terminalSideAlphaPassesThrough : Prop :=
  ∃ α : ℝ, ∃ P : ℝ × ℝ, P = (1, 2) ∧ (cos α, sin α) = (P.1 / sqrt (P.1 ^ 2 + P.2 ^ 2), P.2 / sqrt (P.1 ^ 2 + P.2 ^ 2))

noncomputable def symmetricWithRespectToYAxis : Prop :=
  ∃ β α : ℝ, ∀ P : ℝ × ℝ, P = (1, 2) → β = π - α ∧ (cos β, sin β) = (-P.1 / sqrt (P.1 ^ 2 + P.2 ^ 2), P.2 / sqrt (P.1 ^ 2 + P.2 ^ 2))

theorem tan_alpha_and_cos_beta_minus_alpha:
  terminalSideAlphaPassesThrough ∧ symmetricWithRespectToYAxis →
  (∃ α : ℝ, tan α = 2) ∧ (∃ α β : ℝ, β = π - α → cos (β - α) = 3 / 5) :=
sorry

end tan_alpha_and_cos_beta_minus_alpha_l210_210831


namespace compute_factorial_fraction_l210_210386

theorem compute_factorial_fraction :
  (10! / (9! * 11) : ℚ) = 10 / 11 := 
by
  sorry

end compute_factorial_fraction_l210_210386


namespace find_radius_l210_210968

theorem find_radius (r : ℝ) :
  (135 * r * Real.pi) / 180 = 3 * Real.pi → r = 4 :=
by
  sorry

end find_radius_l210_210968


namespace convert_base_8_to_decimal_l210_210811

theorem convert_base_8_to_decimal :
  ∀ r : ℕ, (1 * r^2 + 7 * r + 5 = 125) → (7 * 8 + 6 = 62) :=
by
  intro r h
  have h_r : r = 8,
  { sorry },
  rw h_r
  exact rfl

end convert_base_8_to_decimal_l210_210811


namespace find_fx_value_l210_210197

theorem find_fx_value 
  (ω : ℝ) (hω_pos : ω > 0)
  (h_dist : (∀ n : ℤ, (tan (ω * (n + 1) * ω⁻¹)) = 3 → (tan (ω * n * ω⁻¹)) = 3 → (ω⁻¹ = π / 4))) :
   tan (4 * (π / 12)) = sqrt 3 :=
by sorry

end find_fx_value_l210_210197


namespace sandy_total_earnings_l210_210574

-- Define the conditions
def hourly_wage : ℕ := 15
def hours_friday : ℕ := 10
def hours_saturday : ℕ := 6
def hours_sunday : ℕ := 14

-- Define the total hours worked and total earnings
def total_hours := hours_friday + hours_saturday + hours_sunday
def total_earnings := total_hours * hourly_wage

-- State the theorem
theorem sandy_total_earnings : total_earnings = 450 := by
  sorry

end sandy_total_earnings_l210_210574


namespace positional_relationship_l210_210082

def Point : Type := sorry -- Define the type for points
def Line : Type := sorry  -- Define the type for lines
def Plane : Type := sorry  -- Define the type for planes

def is_on_line (p : Point) (l : Line) : Prop := sorry  -- Definition for a point being on a line
def is_on_plane (p : Point) (α : Plane) : Prop := sorry -- Definition for a point being on a plane

def equal_distance_from_plane (p1 p2 : Point) (α : Plane) : Prop := sorry -- Definition for two points being at equal distances from a plane 

def midpoint (p1 p2 : Point) : Point := sorry -- Definition for the midpoint of two points

-- Given conditions
variables (l : Line) (α : Plane)
variables (A B : Point)
hypotheses (h1 : is_on_line A l) (h2 : is_on_line B l)
            (h3 : equal_distance_from_plane A B α)

-- Desired proof statement
theorem positional_relationship (h4 : (is_on_plane (midpoint A B) α ∨ (¬ is_on_plane (midpoint A B) α))): 
  let relationship := 
  if (¬ is_on_plane (midpoint A B) α) 
  then -- The line is parallel if midpoint is not online
       ∀ (p : Point), is_on_line p l → (¬ is_on_plane p α) 
  else -- The line is intersecting if midpoint is online
       ∃ (p : Point), is_on_line p l ∧ is_on_plane p α
  in true :=
sorry

end positional_relationship_l210_210082


namespace inclination_angle_of_line_l210_210195

theorem inclination_angle_of_line (a : ℝ) (h : a < 0) : 
  let α := (Real.pi + Real.arctan (1 / a)) in
  (∃ (y : ℝ → ℝ), ∀ x : ℝ, y x = (1 / a) * x + (2 / a) ∧
    (0 ≤ α ∧ α < Real.pi) ∧ Real.tan α = 1 / a) := 
sorry

end inclination_angle_of_line_l210_210195


namespace count_perfect_cubes_or_fourth_powers_l210_210067

theorem count_perfect_cubes_or_fourth_powers :
  ∃ n : ℕ, n = 14 ∧ ∀ x : ℕ,
  0 < x ∧ x < 1000 → (∃ k : ℕ, x = k^3) ∨ (∃ j : ℕ, x = j^4) ↔ n = 14 :=
begin
  sorry,
end

end count_perfect_cubes_or_fourth_powers_l210_210067


namespace num_factors_of_M_l210_210854

-- Define the integer M
def M : ℕ := 2^4 * 3^3 * 7^2

-- Prove that M has 60 natural-number factors
theorem num_factors_of_M : (∀ d : ℕ, d ∣ M → d ≠ 0) → (∑ d in (range (M + 1)), if d ∣ M then 1 else 0) = 60 :=
by
  -- Proof omitted
  sorry

end num_factors_of_M_l210_210854


namespace mary_baseball_cards_l210_210935

theorem mary_baseball_cards :
    (let initial_cards := 18 in
    let promised_fred := 26 in
    let promised_jane := 15 in
    let promised_tom := 36 in
    let bought_cards := 40 in
    let steves_cards := 25 in
    let total_initial_and_added := initial_cards + bought_cards + steves_cards in
    let total_promised := promised_fred + promised_jane + promised_tom in
    total_initial_and_added - total_promised = 6) :=
by
    sorry

end mary_baseball_cards_l210_210935


namespace problem1_f_problem2_l210_210814

-- Define the arithmetic sequence {a_n}
def a (n : ℕ) := 25 - 3 * (n - 1)

-- Problem (1): Prove that a_n < 0 for n ≥ 10
theorem problem1_f (n : ℕ) : n ≥ 10 → a n < 0 := by
  intro hn
  unfold a
  show 25 - 3 * (n - 1) < 0
  sorry

-- Problem (2): Sum of the first 10 terms of the subsequence a_1, a_3, ..., a_19
theorem problem2 : (finset.sum (finset.range 10) (λ k, a (1 + 2 * k))) = -20 := by
  sorry

end problem1_f_problem2_l210_210814


namespace range_of_a_l210_210470

-- Define the negation of the original proposition as a function
def negated_prop (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + 4 * x + a > 0

-- State the theorem to be proven
theorem range_of_a (a : ℝ) (h : ¬∃ x : ℝ, a * x^2 + 4 * x + a ≤ 0) : a > 2 :=
  by
  -- Using the assumption to conclude the negated proposition holds
  let h_neg : negated_prop a := sorry
  
  -- Prove the range of a based on h_neg
  sorry

end range_of_a_l210_210470


namespace incorrect_option_D_l210_210569

variable (AB BC BO DO AO CO : ℝ)
variable (DAB : ℝ)
variable (ABCD_is_rectangle ABCD_is_rhombus ABCD_is_square: Prop)

def conditions_statement :=
  AB = BC ∧
  DAB = 90 ∧
  BO = DO ∧
  AO = CO ∧
  (ABCD_is_rectangle ↔ (AB = BC ∧ AB ≠ BC)) ∧
  (ABCD_is_rhombus ↔ AB = BC ∧ AB ≠ BC) ∧
  (ABCD_is_square ↔ ABCD_is_rectangle ∧ ABCD_is_rhombus)

theorem incorrect_option_D
  (h1: BO = DO)
  (h2: AO = CO)
  (h3: ABCD_is_rectangle)
  (h4: conditions_statement AB BC BO DO AO CO DAB ABCD_is_rectangle ABCD_is_rhombus ABCD_is_square):
  ¬ ABCD_is_square :=
by
  sorry
  -- Proof omitted

end incorrect_option_D_l210_210569


namespace symmetric_matrix_diagonal_odd_symmetric_matrix_diagonal_even_l210_210693

theorem symmetric_matrix_diagonal_odd (n : ℕ) (hn : n % 2 = 1) (M : Matrix (Fin n) (Fin n) ℕ) 
  (hM1 : ∀ i j : Fin n, M i j ∈ Fin.succ n) 
  (hM2 : ∀ i j : Fin n, (M i j = M j i))
  (h_rows : ∀ i, (Finset.univ.image (M i)).card = n)
  (h_cols : ∀ j, (Finset.univ.image (fun i => M i j)).card = n) :
  ∀ k : ℕ, k ∈ Fin.succ n → ∃ i : Fin n, M i i = k :=
by sorry

theorem symmetric_matrix_diagonal_even (n : ℕ) (hn : n % 2 = 0) (M : Matrix (Fin n) (Fin n) ℕ) 
  (hM1 : ∀ i j : Fin n, M i j ∈ Fin.succ n) 
  (hM2 : ∀ i j : Fin n, (M i j = M j i))
  (h_rows : ∀ i, (Finset.univ.image (M i)).card = n)
  (h_cols : ∀ j, (Finset.univ.image (fun i => M i j)).card = n) :
  ¬ ∀ k : ℕ, k ∈ Fin.succ n → ∃ i : Fin n, M i i = k :=
by sorry

end symmetric_matrix_diagonal_odd_symmetric_matrix_diagonal_even_l210_210693


namespace frosting_needed_l210_210562

-- Definitions directly from the problem conditions
def cans_frosting_per_layer_cake := 1
def cans_frosting_per_single_cake := 0.5
def cans_frosting_per_dozen_cupcakes := 0.5
def cans_frosting_per_pan_brownies := 0.5

-- Quantities needed
def layer_cakes_needed := 3
def dozen_cupcakes_needed := 6
def single_cakes_needed := 12
def pans_brownies_needed := 18

-- Proposition to prove
theorem frosting_needed : 
  (cans_frosting_per_layer_cake * layer_cakes_needed) + 
  (cans_frosting_per_dozen_cupcakes * dozen_cupcakes_needed) + 
  (cans_frosting_per_single_cake * single_cakes_needed) + 
  (cans_frosting_per_pan_brownies * pans_brownies_needed) = 21 := 
by 
  sorry

end frosting_needed_l210_210562


namespace eccentricity_of_ellipse_area_of_triangle_PBQ_value_of_k_midpoint_PB_l210_210808

-- Given conditions
def B : ℝ × ℝ := (0, -2)
def ellipseM (x y : ℝ) : Prop := (x^2)/4 + (y^2)/2 = 1
def line_l (k x y : ℝ) : Prop := y = k * x + 1

-- (I) Find the eccentricity of the ellipse M
theorem eccentricity_of_ellipse : ∀ (a b c : ℝ), a^2 = 4 → b^2 = 2 → c = real.sqrt(a^2 - b^2) → (real.sqrt(a^2 - b^2) / a) = real.sqrt(2) / 2 :=
by sorry

-- (II) If k = 1/2, find the area of △PBQ
theorem area_of_triangle_PBQ : ∀ (k : ℝ) (x1 x2 : ℝ), k = 1/2 → 
  ellipseM x1 (1 / 2 * x1 + 1) → ellipseM x2 (1 / 2 * x2 + 1) → 
  (1 / 2 * 3 * (|x1| + |x2|) = 4) :=
by sorry

-- (III) When C is the midpoint of PB, find the value of k
theorem value_of_k_midpoint_PB : ∀ (x1 : ℝ), 
  ellipseM x1 (-1 / 2) → (1 / 2 * x1)^2 / 4 + ( (-2 + (-1 / 2))/2)^2 / 2 = 1 → 
  (k = 3 * real.sqrt(14) / 14 ∨ k = -3 * real.sqrt(14) / 14) :=
by sorry

end eccentricity_of_ellipse_area_of_triangle_PBQ_value_of_k_midpoint_PB_l210_210808


namespace basis_plane_vectors_parallelogram_vertex_D_equilateral_triangle_dot_product_projection_coordinates_l210_210275

-- Problem 1: Basis for Plane Vectors
theorem basis_plane_vectors (a b : ℝ × ℝ) (ha : a = (1, 2)) (hb : b = (3, 1)) :
  ∃ (u v : ℝ) (hu : u ≠ 0) (hv : v ≠ 0), u * a + v * b = (1, 0) ∧ u * a + v * b = (0, 1) := 
sorry

-- Problem 2: Coordinates of Vertex D
theorem parallelogram_vertex_D (A B C D : ℝ × ℝ) (hA : A = (5, -1)) (hB : B = (-1, 7)) (hC : C = (1, 2)) :
  (D = (7, -6)) :=
sorry

-- Problem 3: Equilateral Triangle Dot Product
theorem equilateral_triangle_dot_product (A B C : ℝ × ℝ) (hABC : (equilateral_triangle A B C )) :
  〈A - B, B - C〉 ≠ (π / 3) :=
sorry

-- Problem 4: Projection Coordinates
theorem projection_coordinates (a b : ℝ × ℝ) (ha : a = (1, 1)) (hb : abs b = 4) (hd : ∠(a, b) = π / 4) :
   ∃ (p : ℝ × ℝ), projection p a b = (2, 2) :=
sorry

end basis_plane_vectors_parallelogram_vertex_D_equilateral_triangle_dot_product_projection_coordinates_l210_210275


namespace min_value_l210_210078

-- Define the necessary input parameters and conditions as Lean types
variables (a b : ℝ) (C l : ℝ)

-- Assume the constraints given in the problem
variables (h1 : 2 * a + b = 2) (ha : a > 0) (hb : b > 0)

-- Define the target statement to prove
theorem min_value (h1 : 2 * a + b = 2) (ha : a > 0) (hb : b > 0) : 
  ∃ a b, 2 * a + b = 2 ∧ a > 0 ∧ b > 0 ∧ (1 / a + 2 / b = 4) :=
begin
  sorry
end

end min_value_l210_210078


namespace radius_of_tangent_circle_l210_210846

theorem radius_of_tangent_circle (rA : ℝ) (d : ℝ) (rB : ℝ) :
  (d = 5) → (rA = 2) → (rB = d + rA ∨ rB = d - rA) :=
by
  intro hd
  intro hrA
  rw [hd, hrA]
  exact Or.inl (5 + 2)
  exact Or.inr (5 - 2)

end radius_of_tangent_circle_l210_210846


namespace sum_of_roots_l210_210166

theorem sum_of_roots (sum : ℝ) :
  (∀ x : ℝ, ∃ k : ℤ, 
      3 * Real.cos (4 * Real.pi * x / 5) 
      + Real.cos (12 * Real.pi * x / 5) 
      = 2 * Real.cos (4 * Real.pi * x / 5)
      * (3 + Real.tan (Real.pi * x / 5)^2 - 2 * Real.tan (Real.pi * x / 5))
      ↔ (x = (5 / 8) + (5 * k / 4))) → 
  (sum = List.sum (List.filter (λ x, (x > -11 ∧ x < 19)) 
    (List.map (λ k, (5 / 8) + (5 * k / 4)) (List.range' (-9) 24)))) := 
by 
  sorry

end sum_of_roots_l210_210166


namespace entertainment_spending_l210_210159

-- Define the salary and conditions
def salary : ℕ := 7500
def savings : ℕ := 1500
def food_percent : ℝ := 0.4
def rent_percent : ℝ := 0.2
def conveyance_percent : ℝ := 0.1

-- Define the entertainment percentage to be proven
def entertainment_percent : ℝ := 0.1

-- Define the total percentage spent
def total_percentage_spent : ℝ := food_percent + rent_percent + conveyance_percent + (savings.toReal / salary.toReal)

theorem entertainment_spending :
  total_percentage_spent + entertainment_percent = 1 :=
by
  sorry

end entertainment_spending_l210_210159


namespace find_f_of_given_g_and_odd_l210_210818

theorem find_f_of_given_g_and_odd (f g : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x)
  (h_g_def : ∀ x, g x = f x + 9) (h_g_val : g (-2) = 3) :
  f 2 = 6 :=
by
  sorry

end find_f_of_given_g_and_odd_l210_210818


namespace range_of_m_l210_210049

theorem range_of_m (x y m : ℝ) (h1 : x - y = 2 * m + 7) (h2 : x + y = 4 * m - 3) 
  (h3 : x < 0) (h4 : y < 0) : m < -2 / 3 := 
by 
  sorry

end range_of_m_l210_210049


namespace tim_coins_value_l210_210625

variable (d q : ℕ)

-- Given Conditions
def total_coins (d q : ℕ) : Prop := d + q = 18
def quarter_to_dime_relation (d q : ℕ) : Prop := q = d + 2

-- Prove the value of the coins
theorem tim_coins_value (d q : ℕ) (h1 : total_coins d q) (h2 : quarter_to_dime_relation d q) : 10 * d + 25 * q = 330 := by
  sorry

end tim_coins_value_l210_210625


namespace angle_AOD_solution_l210_210517

theorem angle_AOD_solution (x : ℝ) (AOD_eq : ∠AOD = 2.5 * ∠BOC)
  (OA_perp_OC : is_perpendicular OA OC) 
  (OB_perp_OD : is_perpendicular OB OD) :
  ∠AOD = 128.57 :=
by
  -- Assuming the measure of angle BOC in degrees
  let BOC_deg : ℝ := 51.43
  -- By perpendicularity conditions
  let BOD := 90
  let COA := 90
  let COD := BOD - BOC_deg
  let BOA := COA - BOC_deg
  -- Calculation of the angle AOD
  have AOD_calc : ∠AOD = 180 - BOC_deg := sorry
  -- Given angle condition
  have angle_cond : 2.5 * BOC_deg = 180 - BOC_deg := by linarith
  -- Verification of angle AOD
  show ∠AOD = 128.57°, by sorry

end angle_AOD_solution_l210_210517


namespace marble_weight_l210_210555

theorem marble_weight (m d : ℝ) : (9 * m = 4 * d) → (3 * d = 36) → (m = 16 / 3) :=
by
  intro h1 h2
  sorry

end marble_weight_l210_210555


namespace find_a_l210_210498

theorem find_a (a : ℝ) (x : ℝ) : (a - 1) * x^|a| + 4 = 0 → |a| = 1 → a ≠ 1 → a = -1 :=
by
  intros
  sorry

end find_a_l210_210498


namespace valid_paths_in_grid_with_forbidden_segments_l210_210682

theorem valid_paths_in_grid_with_forbidden_segments :
  let totalPaths := Nat.choose 14 4
  let forbiddenPaths1 := Nat.choose 4 1 * Nat.choose 9 3
  let forbiddenPaths2 := Nat.choose 8 2 * Nat.choose 5 2
  let forbiddenPaths := forbiddenPaths1 + forbiddenPaths2
  let validPaths := totalPaths - forbiddenPaths
  validPaths = 385 :=
by
  sorry

end valid_paths_in_grid_with_forbidden_segments_l210_210682


namespace charcoal_drawings_count_l210_210617

/-- Thomas' drawings problem
  Thomas has 25 drawings in total.
  14 drawings with colored pencils.
  7 drawings with blending markers.
  The rest drawings are made with charcoal.
  We assert that the number of charcoal drawings is 4.
-/
theorem charcoal_drawings_count 
  (total_drawings : ℕ) 
  (colored_pencil_drawings : ℕ) 
  (marker_drawings : ℕ) :
  total_drawings = 25 →
  colored_pencil_drawings = 14 →
  marker_drawings = 7 →
  total_drawings - (colored_pencil_drawings + marker_drawings) = 4 := 
  by
    sorry

end charcoal_drawings_count_l210_210617


namespace floor_log3_probability_l210_210949

/-- Define the interval from which x and y are chosen uniformly. -/
def interval : set ℝ := set.Ioo 0 1

/-- Define the floor of log base 3 function. -/
def floor_log3 (x : ℝ) : ℤ := int.floor (real.log x / real.log 3)

/-- Statement of the theorem: the probability that the floor of log base 3 of x equals 
    the floor of log base 3 of y when x, y are chosen uniformly from the interval (0,1) is 1/2. -/
theorem floor_log3_probability :
  ∀ x y ∈ interval, 
  (floor_log3 x = floor_log3 y) → (probability (floor_log3 x = floor_log3 y) interval interval = 1 / 2) := sorry

end floor_log3_probability_l210_210949


namespace trapezoid_area_inequality_l210_210974

theorem trapezoid_area_inequality
  {A B C D P : Type}
  [trapezoid ABCD] (h_AB_base : is_base AB) (h_CD_base : is_base CD) 
  (h_intersect : intersection ABCD = P) (S : Type → ℝ) :
  (S P A B) + (S P C D) > (S P B C) + (S P D A) :=
sorry

end trapezoid_area_inequality_l210_210974


namespace find_lambda_plus_mu_l210_210807

variables (A P : Point) (λ μ : ℝ)
variables (EA EF EP AP : Vector)

def A := (3,0)
def P := (2,0)
def EA := (2,1)
def EF := (1,2)

-- Definition and conditions translated from the given problem:
def AP := (2,0) - (3,0) = (-1,0)
def EP := (2,1) + (-1,0) = (1,1)

-- Lean statement (theorem) for the given proof problem:
theorem find_lambda_plus_mu (λ μ : ℝ) : 
  (1,1) = λ * (2,1) + μ * (1,2) → λ + μ = 2/3 :=
by
  assume h : (1,1) = λ * (2,1) + μ * (1,2)
  sorry

end find_lambda_plus_mu_l210_210807


namespace geometric_sequence_general_term_l210_210416

theorem geometric_sequence_general_term :
  ∀ (n : ℕ), (n > 0) →
  (∃ (a : ℕ → ℕ), a 1 = 1 ∧ (∀ (k : ℕ), k > 0 → a (k+1) = 2 * a k) ∧ a n = 2^(n-1)) :=
by
  sorry

end geometric_sequence_general_term_l210_210416


namespace cone_properties_l210_210631

noncomputable def cone_geom (m a : ℝ) : Prop :=
  ∃ (r h l : ℝ),
  l = Real.sqrt (h^2 + r^2) ∧
  π * r * (r + l) = π * m^2 ∧
  (π / 3) * r^2 * h = (π / 3) * a^3 ∧
  r^2 + Real.sqrt (a^6 + r^6) = m^2 ∧
  r = (1 / 2) * Real.sqrt ((m^3 + Real.sqrt (m^6 - 8 * a^6)) / m) ∧
  h = (m * (m^3 - Real.sqrt (m^6 - 8 * a^6))) / (2 * a^3) ∧
  m = a * Real.sqrt 2 → r = a ∧ h = a ∧
  a = m / Real.sqrt 2 → (π / 3) * r^2 * h = (π / 3) * (m^3 / (2 * Real.sqrt 2)) ∧
  ∃ α : ℝ,
  α = Real.atan (Real.sqrt 2 / 4) ∧
  α = 19 * (π / 180) + 28 * (π / 10800) + 15 * (π / 648000)

theorem cone_properties : ∀ (m a : ℝ), cone_geom m a :=
begin
  intros,
  sorry
end

end cone_properties_l210_210631


namespace evaluate_expression_l210_210860

theorem evaluate_expression (x : ℝ) (h : 3^(2 * x) = 10) : 9^(x + 1) = 90 :=
by sorry

end evaluate_expression_l210_210860


namespace number_of_intersections_l210_210541

theorem number_of_intersections (n : ℕ) : 
  (∑ k in finset.range (n+1), nat.choose (2*k) k * nat.choose (2*(n-k)) (n-k)) = 4^n := 
by
  sorry

end number_of_intersections_l210_210541


namespace parallel_lines_m_value_l210_210829

theorem parallel_lines_m_value :
  (3 * x + 4 * y - 3 = 0) ∧ (6 * x + m * y + 14 = 0) → (m = 8) :=
begin
  sorry
end

end parallel_lines_m_value_l210_210829


namespace complex_expression_magnitude_l210_210299

def i := Complex.I

theorem complex_expression_magnitude :
  |2 + i^2 + 2 * i^3| = Real.sqrt 5 :=
by
  sorry

end complex_expression_magnitude_l210_210299


namespace problem_solution_l210_210004

universe u

variable (U : Set Nat) (M : Set Nat)
variable (complement_U_M : Set Nat)

axiom U_def : U = {1, 2, 3, 4, 5}
axiom complement_U_M_def : complement_U_M = {1, 3}
axiom M_def : M = U \ complement_U_M

theorem problem_solution : 2 ∈ M := by
  sorry

end problem_solution_l210_210004


namespace correct_statement_l210_210012

universe u
variable (α : Type u)

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5}

noncomputable def complement_U_M : Set ℕ := {1, 3}

noncomputable def M : Set ℕ := U α \ complement_U_M

theorem correct_statement : 2 ∈ M :=
by {
  sorry
}

end correct_statement_l210_210012


namespace pyramid_height_correct_l210_210358

noncomputable def pyramid_height (a α : ℝ) : ℝ :=
  a / (Real.sqrt (2 * (Real.tan (α / 2))^2 - 2))

theorem pyramid_height_correct (a α : ℝ) (hα : α ≠ 0 ∧ α ≠ π) :
  ∃ m : ℝ, m = pyramid_height a α := 
by
  use a / (Real.sqrt (2 * (Real.tan (α / 2))^2 - 2))
  sorry

end pyramid_height_correct_l210_210358


namespace S_9_equals_27_l210_210445

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def a_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  a n

def sum_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 4 + a 8 = 9

def S_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem S_9_equals_27
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : sum_condition a) :
  S_n a 9 = 27 :=
sorry

end S_9_equals_27_l210_210445


namespace max_expression_value_l210_210417

theorem max_expression_value (a b c d e f g h k : ℤ) 
  (ha : a = 1 ∨ a = -1)
  (hb : b = 1 ∨ b = -1)
  (hc : c = 1 ∨ c = -1)
  (hd : d = 1 ∨ d = -1)
  (he : e = 1 ∨ e = -1)
  (hf : f = 1 ∨ f = -1)
  (hg : g = 1 ∨ g = -1)
  (hh : h = 1 ∨ h = -1)
  (hk : k = 1 ∨ k = -1) :
  a * e * k - a * f * h + b * f * g - b * d * k + c * d * h - c * e * g ≤ 4 :=
sorry

end max_expression_value_l210_210417


namespace range_of_k_l210_210464

theorem range_of_k (k : ℝ) (h : ∀ x y : ℝ, x ∈ Ioc (π / 4) (π / 3) → y ∈ Ioc (π / 4) (π / 3) → x < y → k * cos (k * x) > k * cos (k * y)) :
  k ∈ Icc (-6) (-4) ∨ k ∈ Ioc 0 3 ∨ k ∈ Icc 8 9 ∨ k = -12 :=
  sorry

end range_of_k_l210_210464


namespace complex_product_l210_210440

theorem complex_product (z1 z2 : ℂ) (h1 : Complex.abs z1 = 1) (h2 : Complex.abs z2 = 1) 
(h3 : z1 + z2 = -7/5 + (1/5) * Complex.I) : 
  z1 * z2 = 24/25 - (7/25) * Complex.I :=
by
  sorry

end complex_product_l210_210440


namespace log_inequality_solution_l210_210398

theorem log_inequality_solution (x : ℝ) :
  (log 10 (log 9 (log 8 ((log 7 x)^2))) > 1) ↔ x > 7^(8^(9^(10))) :=
sorry

end log_inequality_solution_l210_210398


namespace max_girls_in_ballet_l210_210148

theorem max_girls_in_ballet (boys girls : ℕ) (h1 : boys = 5)
  (h2 : ∀ g, g < girls → ∃ b1 b2 : ℕ, b1 ≠ b2 ∧ b1 < boys ∧ b2 < boys ∧ distance(b1, g) = 5 ∧ distance(b2, g) = 5) : 
  girls ≤ 20 :=
sorry

end max_girls_in_ballet_l210_210148


namespace cartesian_eq_C1_rectangular_eq_C2_max_dist_PQ_l210_210095

noncomputable def parametric_eq_C1 := (θ : ℝ) → (x y : ℝ) × ℕ :=
  (⟨√3 * cos θ, sin θ⟩, 1)

noncomputable def polar_eq_C2 := (ρ θ : ℝ) → Prop :=
  ρ * cos θ - ρ * sin θ - 4 = 0

theorem cartesian_eq_C1 :
  (θ : ℝ) →
  let P := parametric_eq_C1 θ in
  ∃ (x y : ℝ), (P.fst.1 : ℝ) ^ 2 / 3 + ((P.fst.2 : ℝ)) ^ 2 = 1
:= sorry

theorem rectangular_eq_C2 :
  ∀ (ρ θ : ℝ),
  polar_eq_C2 ρ θ →
  ∃ (x y : ℝ), 
    x - y - 4 = 0 ∧
    x = ρ * cos θ ∧ 
    y = ρ * sin θ
:= sorry

theorem max_dist_PQ :
  (θ : ℝ) → let P := parametric_eq_C1 θ in
  ∃ (d : ℝ), 
    let Q := λ (ρ θ' : ℝ), ∃ y, polar_eq_C2 ρ θ' in
    ∀ (Q' : ℝ × ℝ), Q' = (ρ, θ') 
→ (d = (|P.1.1 - P.1.2 - 4| / √2)) ∧ 
     d = 3 * √2
:= sorry

end cartesian_eq_C1_rectangular_eq_C2_max_dist_PQ_l210_210095


namespace time_wandered_l210_210124

-- Definitions and Hypotheses
def distance : ℝ := 4
def speed : ℝ := 2

-- Proof statement
theorem time_wandered : distance / speed = 2 := by
  sorry

end time_wandered_l210_210124


namespace g_neither_even_nor_odd_l210_210106

def g (x : ℝ) : ℝ := ⌊2 * x⌋ + 1 / 3

theorem g_neither_even_nor_odd : ¬ (∀ x, g (-x) = g x) ∧ ¬ (∀ x, g (-x) = -g x) :=
by
  sorry

end g_neither_even_nor_odd_l210_210106


namespace max_apartment_size_l210_210709

theorem max_apartment_size (rental_price_per_sqft : ℝ) (budget : ℝ) (h1 : rental_price_per_sqft = 1.20) (h2 : budget = 720) : 
  budget / rental_price_per_sqft = 600 :=
by 
  sorry

end max_apartment_size_l210_210709


namespace bubble_sort_probability_l210_210963

/-- Given n = 50 and the terms r₁, r₂, ..., r₅₀ are distinct from one another and are in random order,
let p/q, in lowest terms, be the probability that the number that begins as r₅ completₑs to thₑ 35th
 place after one bubble pass. Prove that if p/q = 1/1190,
 then p + q = 1191. --/

theorem bubble_sort_probability (n : ℕ) (r : fin n → ℕ) (h1 : n = 50) 
    (h2 : function.injective r) (h3 : ∀ a, a < 50 → a < fin.last n) :
    ∃ (p q : ℕ), (nat.coprime p q ∧ p / q = 1 / 1190) → (p + q = 1191) := sorry

end bubble_sort_probability_l210_210963


namespace repeating_decimal_to_fraction_l210_210408

noncomputable def repeating_decimal := 0.6 + 3 / 100

theorem repeating_decimal_to_fraction :
  repeating_decimal = 19 / 30 :=
  sorry

end repeating_decimal_to_fraction_l210_210408


namespace probability_distribution_l210_210119

noncomputable def mean_variance_binomial (n : ℕ) (p : ℝ) : ℝ × ℝ :=
  (n * p, n * p * (1 - p))

theorem probability_distribution (P : ℕ → ℝ) :
  (∀ n : ℕ, n > 2 → P n = 1/3 * P (n - 1) + 2/3 * P (n - 2)) →
  P 1 = 1/3 →
  P 2 = 7/9 →
  ∀ n : ℕ, n ≥ 1 →
  P n = 3/5 - 4/15 * (-2/3)^(n-1) :=
sorry

example : mean_variance_binomial 5 (2/3) = (10/3, 10/9) :=
begin
  simp [mean_variance_binomial],
  norm_num,
end

end probability_distribution_l210_210119


namespace solutions_eq_l210_210392

theorem solutions_eq :
  { (a, b, c) : ℕ × ℕ × ℕ | a * b + b * c + c * a = 2 * (a + b + c) } =
  { (2, 2, 2),
    (1, 2, 4), (1, 4, 2), 
    (2, 1, 4), (2, 4, 1),
    (4, 1, 2), (4, 2, 1) } :=
by sorry

end solutions_eq_l210_210392


namespace chocolate_cost_is_75_l210_210357

def candy_bar_cost : ℕ := 25
def juice_pack_cost : ℕ := 50
def num_quarters : ℕ := 11
def total_cost_in_cents : ℕ := num_quarters * candy_bar_cost
def num_candy_bars : ℕ := 3
def num_pieces_of_chocolate : ℕ := 2

def chocolate_cost_in_cents (x : ℕ) : Prop :=
  (num_candy_bars * candy_bar_cost) + (num_pieces_of_chocolate * x) + juice_pack_cost = total_cost_in_cents

theorem chocolate_cost_is_75 : chocolate_cost_in_cents 75 :=
  sorry

end chocolate_cost_is_75_l210_210357


namespace sin_cos_cubed_sum_l210_210907

theorem sin_cos_cubed_sum
  (x : ℝ)
  (hx : 0 < x ∧ x < π / 2)
  (h : sin x - cos x = 1 / 2) :
  ∃ (m n p : ℕ), 
    m.gcd n = 1 ∧ n.gcd p = 1 ∧ m.gcd p = 1 ∧
    ¬ ∃ q : ℕ, prime q ∧ q ^ 2 ∣ p ∧
    (sin x) ^ 3 + (cos x) ^ 3 = (m * real.sqrt p) / n ∧ 
    m + n + p = 28 := 
sorry

end sin_cos_cubed_sum_l210_210907


namespace simplify_complex_division_l210_210954

theorem simplify_complex_division :
  (5 + 7 * Complex.i) / (3 + 4 * Complex.i) = (43 / 25) + (1 / 25) * Complex.i :=
by
  sorry

end simplify_complex_division_l210_210954


namespace perpendicular_lines_a_eq_1_l210_210044

-- Definitions for the given conditions
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + y + 3 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := x + (2 * a - 3) * y = 4

-- Condition that the lines are perpendicular
def perpendicular_lines (a : ℝ) : Prop := a + (2 * a - 3) = 0

-- Proof problem to be solved
theorem perpendicular_lines_a_eq_1 (a : ℝ) (h : perpendicular_lines a) : a = 1 :=
by
  sorry

end perpendicular_lines_a_eq_1_l210_210044


namespace greatest_value_of_a_l210_210775

theorem greatest_value_of_a :
  (∃ a : ℝ, (5 * real.sqrt ((2 * a)^2 + 1) - 4 * a^2 - 2 * a) / (real.sqrt (1 + 4 * a^2) + 5) = 1) →
  ∃ a : ℝ, (a = real.sqrt 6) :=
by sorry

end greatest_value_of_a_l210_210775


namespace complex_abs_sqrt_five_l210_210322

open Complex

theorem complex_abs_sqrt_five : abs (2 + (-1 : ℂ) + 2 * (-I : ℂ)) = Real.sqrt 5 := 
by
  sorry

end complex_abs_sqrt_five_l210_210322


namespace find_circle_equation_l210_210821

noncomputable def circle_equation (D E F : ℝ) : Prop :=
  ∀ (x y : ℝ), (x, y) = (-1, 3) ∨ (x, y) = (0, 0) ∨ (x, y) = (0, 2) →
  x^2 + y^2 + D * x + E * y + F = 0

theorem find_circle_equation :
  ∃ D E F : ℝ, circle_equation D E F ∧
               (∀ x y, x^2 + y^2 + D * x + E * y + F = x^2 + y^2 + 4 * x - 2 * y) :=
sorry

end find_circle_equation_l210_210821


namespace polar_to_rectangular_conversion_l210_210668

theorem polar_to_rectangular_conversion (x y : ℝ) (hx : x = 10) (hy : y = -3) :
  let r := Real.sqrt (x^2 + y^2) in
  let θ := Real.arctan (y / x) in
  let r' := 2 * r^2 in
  let θ' := 2 * θ + Real.pi / 4 in
  (r' * Real.cos θ', r' * Real.sin θ') = ((218 * (100 + 60) / 109) * Real.sqrt 2 / 2, (218 * (100 - 60) / 109) * Real.sqrt 2 / 2) :=
by
  sorry

end polar_to_rectangular_conversion_l210_210668


namespace truck_distance_l210_210286

theorem truck_distance :
  ∀ (a b c d : ℕ),
    a = 20 →
    b = 30 →
    c = 20 →
    d = a + c →
    (d^2 + b^2 = 50^2) :=
by
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  norm_num
  sorry

end truck_distance_l210_210286


namespace find_m_n_l210_210123

def speed_jon : ℚ := 1 / 3

def speed_steve : ℚ := 1 / 3

def speed_east_train (r1 : ℚ) : Prop := 
  ∀ d : ℚ, d = (r1 - speed_jon) ∧ d = (r2 + speed_jon)

def speed_west_train (r2 : ℚ) : Prop := 
  ∀ d : ℚ, d = (r2 + speed_jon) ∧ d = r1 - speed_jon

def time_east_train (r1 r2 : ℚ) : Prop := 
  r1 = r2 + 2 / 3

def time_west_train (r1 r2 : ℚ) : Prop :=
  ∀ d : ℚ, 
  1 / (r2 - speed_steve) = 10 * (1 / (r1 + speed_steve))

def train_length (r2 : ℚ) : ℚ :=
  r2 + speed_jon

theorem find_m_n :
  ∃ m n : ℕ, nat.gcd m n = 1 ∧
  (train_length (13 / 27) = 22 / 27) → (m + n) = 49 :=
by
  sorry

end find_m_n_l210_210123


namespace bus_arrival_time_at_first_station_l210_210147

noncomputable def time_to_first_station (start_time end_time first_station_to_work: ℕ) : ℕ :=
  (end_time - start_time) - first_station_to_work

theorem bus_arrival_time_at_first_station :
  time_to_first_station 360 540 140 = 40 :=
by
  -- provide the proof here, which has been omitted per the instructions
  sorry

end bus_arrival_time_at_first_station_l210_210147


namespace maximize_dragon_resilience_l210_210754

noncomputable def p_s (x : ℝ) (s : ℕ) : ℝ :=
  x^s / (1 + x + x^2)

def K : List ℕ :=
  [1, 2, 2, 1, 0, 2, 1, 0, 1, 2]

def P_K (x : ℝ) : ℝ :=
  List.foldr (λ s acc => acc * p_s x s) 1 K

theorem maximize_dragon_resilience :
  let x_opt := (Real.sqrt 97 + 1) / 8 in
  ∀ x > 0, P_K x ≤ P_K x_opt :=
sorry

end maximize_dragon_resilience_l210_210754


namespace correctStatement_l210_210018

variable (U : Set ℕ) (M : Set ℕ)

namespace Proof

-- Given conditions
def universalSet := {1, 2, 3, 4, 5}
def complementM := {1, 3}
def isComplement (M : Set ℕ) : Prop := U \ M = complementM

-- Target statement to be proved
theorem correctStatement (h1 : U = universalSet) (h2 : isComplement M) : 2 ∈ M := by
  sorry

end Proof

end correctStatement_l210_210018


namespace no_such_ab_exists_l210_210401

theorem no_such_ab_exists : ¬ ∃ (a b : ℝ), ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 * Real.pi → (a * x + b)^2 - Real.cos x * (a * x + b) < (1 / 4) * (Real.sin x)^2 :=
by
  sorry

end no_such_ab_exists_l210_210401


namespace angles_of_isosceles_triangle_l210_210090

def is_isosceles_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
  dist A B = dist A C

def interior_angles_sum_to_180 (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (angle_A angle_B angle_C : ℝ) : Prop :=
  angle_A + angle_B + angle_C = 180

def bisector_condition (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  [MetricSpace D] [MetricSpace E] (AD BE : ℝ) : Prop :=
  AD = BE / 2

theorem angles_of_isosceles_triangle 
  (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] 
  (angle_A angle_B angle_C : ℝ)
  (AD BE : ℝ)
  (isosceles : is_isosceles_triangle A B C)
  (angles_sum : interior_angles_sum_to_180 A B C angle_A angle_B angle_C)
  (bisectors : bisector_condition A B C D E AD BE) :
  angle_A = 108 ∧ angle_B = 36 ∧ angle_C = 36 :=
by
  sorry

end angles_of_isosceles_triangle_l210_210090


namespace find_f_7_l210_210791

noncomputable theory

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(x) = f(-x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f(x + p) = f(x)

def f (x: ℝ) : ℝ :=
  if 0 < x ∧ x < 2 then 2 * x^2
  else if even_function (λ y, if y = x then 1 else 0) then f (-x)
  else if periodic_function (λ y, if y = x then 1 else 0) 4 then f (x - 4)
  else 0

theorem find_f_7 :
  f 7 = 2 :=
sorry

end find_f_7_l210_210791


namespace range_of_a_l210_210871

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x - a| - |x| < 2 - a^2) ↔ a ∈ set.Ioo (-1 : ℝ) (1 : ℝ) :=
by
  sorry

end range_of_a_l210_210871


namespace angle_B_measure_l210_210083

-- Given conditions
variables (a b c A B C : ℝ)
hypotheses
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hA : 0 < A ∧ A < 2 * π)
  (hB : 0 < B ∧ B < 2 * π)
  (hC : 0 < C ∧ C < 2 * π)
  (h1 : a = b * (cos B) / (cos C)) -- This implies a = b / 2(a + c)

-- Statement to prove
theorem angle_B_measure : B = 2 * π / 3 :=
sorry

end angle_B_measure_l210_210083


namespace frequency_of_sample_l210_210354

noncomputable def sampleSize : ℕ := 20

noncomputable def groupFrequencies : list (ℕ × ℕ) := [(10, 20), 2, (20, 30), 3, (30, 40), 4, (40, 50), 5, (50, 60), 4, (60, 70), 2]

noncomputable def frequencyInInterval (intervals: list (ℕ × ℕ)) (frequencies: list ℕ) (bound: ℕ) : ℝ :=
  ((intervals.zip frequencies).filter (fun ⟨(a, b), _⟩ => b <= bound)).sum (·.snd) / sampleSize

theorem frequency_of_sample (intervals frequencies : list (ℕ × ℕ)) : frequencyInInterval intervals frequencies 50 = 0.7 :=
  by
    sorry

end frequency_of_sample_l210_210354


namespace parliament_committees_l210_210920

theorem parliament_committees (n : ℕ) (h : n ≥ 4)
  (enemies : fin n → finset (fin n))
  (h_enemies : ∀ a : fin n, (enemies a).card = 3)
  (h_symmetric : ∀ a b : fin n, b ∈ enemies a → a ∈ enemies b) :
  ∃ C1 C2 : finset (fin n), C1 ∪ C2 = finset.univ ∧ C1 ∩ C2 = ∅ ∧ 
  (∀ a : fin n, (enemies a).filter (λ b, b ∈ C1).card ≤ 1) ∧
  (∀ a : fin n, (enemies a).filter (λ b, b ∈ C2).card ≤ 1) := 
sorry

end parliament_committees_l210_210920


namespace total_number_of_lives_l210_210615

theorem total_number_of_lives (initial_players : ℕ) (additional_players : ℕ) (lives_per_player : ℕ) 
                              (h1 : initial_players = 7) (h2 : additional_players = 2) (h3 : lives_per_player = 7) : 
                              initial_players + additional_players * lives_per_player = 63 :=
by
  sorry

end total_number_of_lives_l210_210615


namespace part_number_in_fourth_selection_l210_210556

theorem part_number_in_fourth_selection
  (total_parts : ℕ)
  (num_samples : ℕ)
  (initial_selection : ℕ)
  (sampling_interval : ℕ)
  (fourth_selection : ℕ)
  (selection_eq : ∀ n : ℕ, n > 0 → n ≤ num_samples → initial_selection + (n - 1) * sampling_interval = nth_selection n)
  (total_parts_eq : total_parts = 200)
  (num_samples_eq : num_samples = 10)
  (initial_selection_eq : initial_selection = 5)
  (sampling_interval_eq : sampling_interval = total_parts / num_samples)
  (fourth_selection_eq : fourth_selection = nth_selection 4)
: fourth_selection = 65 := sorry

end part_number_in_fourth_selection_l210_210556


namespace dragon_resilience_maximized_l210_210758

noncomputable def probability (x : ℝ) (s : ℕ) : ℝ :=
  x^s / (1 + x + x^2)

noncomputable def prob_vec (x : ℝ) : ℝ :=
  let K := [1, 2, 2, 1, 0, 2, 1, 0, 1, 2]
  K.foldr (λ s acc, acc * probability x s) 1

theorem dragon_resilience_maximized (x : ℝ) : 
  (x = (Real.sqrt 97 + 1) / 8) → 
  (0 < x) →
  ∀ K, K = [1, 2, 2, 1, 0, 2, 1, 0, 1, 2] →
  prob_vec x = (x^12 / (1 + x + x^2)^10) :=
begin
  sorry
end

end dragon_resilience_maximized_l210_210758


namespace common_points_count_l210_210418

open Set

noncomputable def eq1 : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), (p.1 - 2 * p.2 + 3) * (4 * p.1 + p.2 - 1) = 0
noncomputable def eq2 : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), (2 * p.1 - p.2 - 5) * (p.1 + 3 * p.2 - 4) = 0

theorem common_points_count : 
  ∃ S : Finite (ℝ × ℝ), (∀ p, p ∈ S ↔ (eq1 p ∧ eq2 p)) ∧ S.finset.card = 4 := 
  sorry

end common_points_count_l210_210418


namespace max_angles_greater_than_45_in_acute_triangle_l210_210061

theorem max_angles_greater_than_45_in_acute_triangle :
  ∀ (a b c : ℝ), a + b + c = 180 ∧ 0 < a ∧ a < 90 ∧ 0 < b ∧ b < 90 ∧ 0 < c ∧ c < 90 →
  ( (45 < a → 1 else 0) + (45 < b → 1 else 0) + (45 < c → 1 else 0) ) ≤ 3 :=
by
  intros a b c h
  sorry

end max_angles_greater_than_45_in_acute_triangle_l210_210061


namespace set_properties_P_l210_210575

theorem set_properties_P (P : Set ℤ)
  (h₁ : ∀ x ∈ P, x ∈ ℤ)
  (h₂ : ∃ x y ∈ P, x > 0 ∧ y < 0)
  (h₃ : ∃ x y ∈ P, ∃ a b : ℤ, x = 2 * a + 1 ∧ y = 2 * b)
  (h₄ : -1 ∉ P)
  (h₅ : ∀ x y ∈ P, x + y ∈ P) :
  0 ∉ P ∧ 2 ∈ P :=
sorry

end set_properties_P_l210_210575


namespace simple_interest_years_l210_210694

theorem simple_interest_years 
  (R : ℝ) 
  (N : ℕ) 
  (h_sum : 400 * R * N / 100 + 200 = 400 * (R + 5) * N / 100) 
  : N = 10 := 
by 
  calc 
    400 * R * N / 100 + 200
    = 400 * (R + 5) * N / 100 : h_sum
  ... 
  sorry

end simple_interest_years_l210_210694


namespace air_quality_conditional_probability_l210_210361

theorem air_quality_conditional_probability :
  (P(A) = 0.8) → (P(A) ∧ P(B) = 0.6) → (P(B | A) = 0.75) := 
by
  intros h1 h2
  sorry

end air_quality_conditional_probability_l210_210361


namespace zeros_in_interval_l210_210918

noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x - Real.pi / 4)

theorem zeros_in_interval : 
  ∃ (s : Set ℝ) (n : ℕ), 
    s = {x | f x = 0} ∧ 
    s ∩ Icc (-Real.pi) (Real.pi) = {x | x = Real.pi / 12 + k * Real.pi / 3 ∧ k ∈ { -3, -2, -1, 0, 1, 2}}.to_finset ∧ 
    n = 6 := 
sorry

end zeros_in_interval_l210_210918


namespace chen_family_temple_visitor_l210_210702

-- Defining the propositions for each person's statement
def A_statement : Prop := ∀ (C_visited : Bool), ¬C_visited
def B_statement : Prop := ∀ (B_visited : Bool), B_visited
def C_statement : Prop := ∀ (A_statement_truth : Prop), A_statement_truth

-- Define the main hypothesis: only one person is lying
def exactly_one_lying (A_statement : Prop) (B_statement : Prop) (C_statement : Prop) (A_lying : Bool) (B_lying : Bool) (C_lying : Bool) : Prop :=
  (A_lying && ¬B_lying && ¬C_lying) ∨ (¬A_lying && B_lying && ¬C_lying) ∨ (¬A_lying && ¬B_lying && C_lying)

-- Main theorem 
theorem chen_family_temple_visitor : ∃ (A_visited B_visited C_visited : Bool), 
  (A_visited = true) ∧ 
  (B_visited = false) ∧ 
  (C_visited = false) ∧ 
  (exactly_one_lying (A_statement C_visited) (B_statement B_visited) (C_statement (A_statement C_visited)) ¬A_visited ¬B_visited (¬A_statement C_visited)) :=
by
  sorry

end chen_family_temple_visitor_l210_210702


namespace range_of_values_l210_210452

theorem range_of_values 
  (f : ℝ → ℝ) 
  (hf_even : ∀ x, f(x) = f(-x))
  (hf_monotone : ∀ a b, 0 ≤ a → a ≤ b → f(b) ≤ f(a))
  (hf_at_neg_two : f(-2) = 0) :
  (∀ x, f(x - 2) > 0 ↔ 0 < x ∧ x < 4) :=
by
  sorry

end range_of_values_l210_210452


namespace number_of_three_digit_numbers_is_48_l210_210618

-- Define the problem: the cards and their constraints
def card1 := (1, 2)
def card2 := (3, 4)
def card3 := (5, 6)

-- The condition given is that 6 cannot be used as 9

-- Define the function to compute the number of different three-digit numbers
def number_of_three_digit_numbers : Nat := 6 * 4 * 2

/- Prove that the number of different three-digit numbers that can be formed is 48 -/
theorem number_of_three_digit_numbers_is_48 : number_of_three_digit_numbers = 48 :=
by
  -- We skip the proof here
  sorry

end number_of_three_digit_numbers_is_48_l210_210618


namespace problem_statement_l210_210046

noncomputable def angle_between_vectors
  (a b : EuclideanSpace ℝ (Fin 3))
  (h1 : ‖a‖ = 2)
  (h2 : ‖b‖ = 4)
  (h3 : (b - a) ⬝ a = 0) : ℝ :=
Real.arccos ((a ⬝ b) / (‖a‖ * ‖b‖))

theorem problem_statement {a b : EuclideanSpace ℝ (Fin 3)}
  (h1 : ‖a‖ = 2)
  (h2 : ‖b‖ = 4)
  (h3 : (b - a) ⬝ a = 0) : angle_between_vectors a b h1 h2 h3 = Real.pi / 3 :=
sorry

end problem_statement_l210_210046


namespace steven_owes_jeremy_l210_210897

-- Definitions for the conditions
def base_payment_per_room := (13 : ℚ) / 3
def rooms_cleaned := (5 : ℚ) / 2
def additional_payment_per_room := (1 : ℚ) / 2

-- Define the total amount of money Steven owes Jeremy
def total_payment (base_payment_per_room rooms_cleaned additional_payment_per_room : ℚ) : ℚ :=
  let base_payment := base_payment_per_room * rooms_cleaned
  let additional_payment := if rooms_cleaned > 2 then additional_payment_per_room * rooms_cleaned else 0
  base_payment + additional_payment

-- The statement to prove
theorem steven_owes_jeremy :
  total_payment base_payment_per_room rooms_cleaned additional_payment_per_room = 145 / 12 :=
by
  sorry

end steven_owes_jeremy_l210_210897


namespace permutation_count_3070_l210_210925

def is_permutation (l1 l2 : list ℕ) : Prop :=
  l1 ~ l2

def satisfies_condition (a : list ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 20 → 
    |a.nth_le (i-1) (by linarith) - a.nth_le (20) (by linarith)| ≥ 
    |a.nth_le (j-1) (by linarith) - a.nth_le (20) (by linarith)|

theorem permutation_count_3070 : 
  ∃ (a : list ℕ), is_permutation a (list.range 21).map(λ n, n+1) ∧ satisfies_condition a ↔ 
    list.countp (λ a, is_permutation a (list.range 21).map(λ n, n+1) ∧ satisfies_condition a) (list.permutations (list.range 21).map(λ n, n+1)) = 3070 :=
by sorry

end permutation_count_3070_l210_210925


namespace select_5_balls_odd_sum_l210_210426

theorem select_5_balls_odd_sum :
  (finset.card {s : finset ℕ | s.card = 5 ∧ 
                  (∀ x ∈ s, x ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}) ∧ 
                  s.sum id % 2 = 1} = 206) :=
by {
  -- the solution steps and proof would go here
  sorry
}

end select_5_balls_odd_sum_l210_210426


namespace counts_of_perfect_cubes_and_fourth_powers_l210_210071

theorem counts_of_perfect_cubes_and_fourth_powers : 
  finset.card ({n | (∃ k, n = k^3) ∨ (∃ k, n = k^4) ∧ 0 < n ∧ n < 1000}) = 14 :=
begin
  sorry
end

end counts_of_perfect_cubes_and_fourth_powers_l210_210071


namespace part1_inequality_part2_inequality_case1_part2_inequality_case2_part2_inequality_case3_l210_210656

-- Part (1)
theorem part1_inequality (m : ℝ) : (∀ x : ℝ, (m^2 + 1)*x^2 - (2*m - 1)*x + 1 > 0) ↔ m > -3/4 := sorry

-- Part (2)
theorem part2_inequality_case1 (a : ℝ) (h : 0 < a ∧ a < 1) : 
  (∀ x : ℝ, (x - 1)*(a*x - 1) > 0 ↔ x < 1 ∨ x > 1/a) := sorry

theorem part2_inequality_case2 : 
  (∀ x : ℝ, (x - 1)*(0*x - 1) > 0 ↔ x < 1) := sorry

theorem part2_inequality_case3 (a : ℝ) (h : a < 0) : 
  (∀ x : ℝ, (x - 1)*(a*x - 1) > 0 ↔ 1/a < x ∧ x < 1) := sorry

end part1_inequality_part2_inequality_case1_part2_inequality_case2_part2_inequality_case3_l210_210656


namespace price_difference_l210_210872

-- Definitions of conditions
def market_price : ℝ := 15400
def initial_sales_tax_rate : ℝ := 0.076
def new_sales_tax_rate : ℝ := 0.0667
def discount_rate : ℝ := 0.05
def handling_fee : ℝ := 200

-- Calculation of original sales tax
def original_sales_tax_amount : ℝ := market_price * initial_sales_tax_rate
-- Calculation of price after discount
def discount_amount : ℝ := market_price * discount_rate
def price_after_discount : ℝ := market_price - discount_amount
-- Calculation of new sales tax
def new_sales_tax_amount : ℝ := price_after_discount * new_sales_tax_rate
-- Calculation of total price with new sales tax and handling fee
def total_price_new : ℝ := price_after_discount + new_sales_tax_amount + handling_fee
-- Calculation of original total price with handling fee
def original_total_price : ℝ := market_price + original_sales_tax_amount + handling_fee

-- Expected difference in total cost
def expected_difference : ℝ := 964.60

-- Lean 4 statement to prove the difference
theorem price_difference :
  original_total_price - total_price_new = expected_difference :=
by
  sorry

end price_difference_l210_210872


namespace sum_and_round_l210_210698

theorem sum_and_round :
  let a := 53.463
  let b := 12.98734
  let c := 0.5697
  let sum := a + b + c
  round (sum * 100) / 100 = 67.02 :=
by
  sorry

end sum_and_round_l210_210698


namespace distance_traveled_on_fifth_day_equals_12_li_l210_210246

theorem distance_traveled_on_fifth_day_equals_12_li:
  ∀ {a_1 : ℝ},
    (a_1 * ((1 - (1 / 2) ^ 6) / (1 - 1 / 2)) = 378) →
    (a_1 * (1 / 2) ^ 4 = 12) :=
by
  intros a_1 h
  sorry

end distance_traveled_on_fifth_day_equals_12_li_l210_210246


namespace coraline_number_l210_210145

theorem coraline_number (C : ℤ) (H1 : ∃(M J : ℤ), M = J + 20 ∧ J = C - 40 ∧ C + J + M = 180) : C = 80 :=
by
  cases H1 with M H1
  cases H1 with J H1
  cases H1 with H2 H3
  cases H3 with H4 H5
  sorry

end coraline_number_l210_210145


namespace fibonacci_seven_fibonacci_sum_2017_l210_210177

noncomputable def fibonacci (n : ℕ) : ℕ :=
  if h : n = 0 then 0
  else if h : n = 1 then 1
  else fibonacci (n - 1) + fibonacci (n - 2)

theorem fibonacci_seven : fibonacci 7 = 13 :=
  by sorry

theorem fibonacci_sum_2017 (m : ℕ) (h : fibonacci 2019 = m) : 
  (finset.range 2017).sum fibonacci = m - 1 :=
  by sorry

end fibonacci_seven_fibonacci_sum_2017_l210_210177


namespace complex_abs_sqrt_five_l210_210323

open Complex

theorem complex_abs_sqrt_five : abs (2 + (-1 : ℂ) + 2 * (-I : ℂ)) = Real.sqrt 5 := 
by
  sorry

end complex_abs_sqrt_five_l210_210323


namespace problem1_problem2_l210_210293
  
theorem problem1 : (sqrt 12 + | -4 | - (2003 - real.pi)^0 - 2 * real.cos (real.pi / 6)) = (sqrt 3 + 3) := 
by 
  -- skipping the proof
  sorry

theorem problem2 (a : ℤ) (h1 : 0 < a) (h2 : a < 4) (h3 : a ≠ 2) (h4 : a ≠ 3) : 
  (a + 2 - 5 / (a - 2)) / ((3 - a) / (2 * a - 4)) = -8 := 
by 
  -- skipping the proof
  sorry

end problem1_problem2_l210_210293


namespace geometric_sequence_max_product_l210_210509

theorem geometric_sequence_max_product
  (b : ℕ → ℝ) (q : ℝ) (b1 : ℝ)
  (h_b1_pos : b1 > 0)
  (h_q : 0 < q ∧ q < 1)
  (h_b : ∀ n, b (n + 1) = b n * q)
  (h_b7_gt_1 : b 7 > 1)
  (h_b8_lt_1 : b 8 < 1) :
  (∀ (n : ℕ), n = 7 → b 1 * b 2 * b 3 * b 4 * b 5 * b 6 * b 7 = b 1 * b 2 * b 3 * b 4 * b 5 * b 6 * b 7) :=
by {
  sorry
}

end geometric_sequence_max_product_l210_210509


namespace total_stocking_stuffers_total_stocking_stuffers_hannah_buys_l210_210052

def candy_canes_per_kid : ℕ := 4
def beanie_babies_per_kid : ℕ := 2
def books_per_kid : ℕ := 1
def kids : ℕ := 3

theorem total_stocking_stuffers : candy_canes_per_kid + beanie_babies_per_kid + books_per_kid = 7 :=
by { 
  -- by trusted computation
  sorry
}

theorem total_stocking_stuffers_hannah_buys : 3 * (candy_canes_per_kid + beanie_babies_per_kid + books_per_kid) = 21 :=
by {
  have h : candy_canes_per_kid + beanie_babies_per_kid + books_per_kid = 7 := total_stocking_stuffers,
  rw h,
  norm_num,
}

end total_stocking_stuffers_total_stocking_stuffers_hannah_buys_l210_210052


namespace periodic_sequence_u_16_eq_a_l210_210730

noncomputable def periodic_sequence (a : ℝ) : ℕ → ℝ
| 0     := a
| (n+1) := -1 / (periodic_sequence n + 1)

theorem periodic_sequence_u_16_eq_a (a : ℝ) (h : 0 < a) : (periodic_sequence a 15) = a := sorry

end periodic_sequence_u_16_eq_a_l210_210730


namespace problem_solution_l210_210007

universe u

variable (U : Set Nat) (M : Set Nat)
variable (complement_U_M : Set Nat)

axiom U_def : U = {1, 2, 3, 4, 5}
axiom complement_U_M_def : complement_U_M = {1, 3}
axiom M_def : M = U \ complement_U_M

theorem problem_solution : 2 ∈ M := by
  sorry

end problem_solution_l210_210007


namespace perp_BQ_CN_l210_210373

open EuclideanGeometry

noncomputable def triangleABC (A B C M N P Q: Point) : Prop := 
  isTriangle A B C ∧ isSquare A B M N ∧ isSquare A C P Q

theorem perp_BQ_CN (A B C M N P Q : Point) :
  triangleABC A B C M N P Q →
  perp (line_through B Q) (line_through C N) := sorry

end perp_BQ_CN_l210_210373


namespace min_changes_to_make_sums_distinct_l210_210232

-- Definition and initial conditions
def matrix3x3 : Type := list (list ℕ)

def initial_matrix : matrix3x3 :=
  [[4, 9, 2], 
   [8, 1, 6], 
   [3, 5, 7]]

def row_sum (m : matrix3x3) (r : ℕ) : ℕ :=
  m[r].sum

def col_sum (m : matrix3x3) (c : ℕ) : ℕ :=
  m.map (λ row, row[c]).sum

-- The target properties
def distinct_sums (sums : list ℕ) : Prop :=
  sums.nodup

-- Main statement
theorem min_changes_to_make_sums_distinct : 
  ∃ (new_matrix : matrix3x3), 
    (∀ i < 3, row_sum new_matrix i ≠ row_sum initial_matrix i) ∧
    (∀ j < 3, col_sum new_matrix j ≠ col_sum initial_matrix j) ∧
    ∃ (changed_cells : list (ℕ × ℕ)), 
      changed_cells.length = 4 ∧
      (∀ (i j : ℕ), (i, j) ∈ changed_cells → new_matrix[i][j] ≠ initial_matrix[i][j]) ∧
      distinct_sums (list.map (row_sum new_matrix) [0, 1, 2] ++ list.map (col_sum new_matrix) [0, 1, 2]) := 
sorry

end min_changes_to_make_sums_distinct_l210_210232


namespace power_func_odd_real_domain_l210_210130

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f (x)

noncomputable def correct_values : set ℝ := {1, 3}

theorem power_func_odd_real_domain (a : ℝ) (h : a ∈ {-1, 0, 1 / 2, 1, 2, 3}) :
  (∀ x : ℝ, x ^ a ∈ ℝ ∧ is_odd_function (λ x, x ^ a)) ↔ a ∈ correct_values := 
sorry

end power_func_odd_real_domain_l210_210130


namespace count_perfect_cubes_or_fourth_powers_l210_210068

theorem count_perfect_cubes_or_fourth_powers :
  ∃ n : ℕ, n = 14 ∧ ∀ x : ℕ,
  0 < x ∧ x < 1000 → (∃ k : ℕ, x = k^3) ∨ (∃ j : ℕ, x = j^4) ↔ n = 14 :=
begin
  sorry,
end

end count_perfect_cubes_or_fourth_powers_l210_210068


namespace sum_of_possible_values_l210_210924

theorem sum_of_possible_values (a b c d : ℝ) (h1 : |a - b| = 2) (h2 : |b - c| = 4) (h3 : |c - d| = 5) :
  let S := { |a - d| | a b c d : ℝ, |a - b| = 2, |b - c| = 4, |c - d| = 5 } in
  S.sum = 22 :=
sorry

end sum_of_possible_values_l210_210924


namespace correctStatement_l210_210021

variable (U : Set ℕ) (M : Set ℕ)

namespace Proof

-- Given conditions
def universalSet := {1, 2, 3, 4, 5}
def complementM := {1, 3}
def isComplement (M : Set ℕ) : Prop := U \ M = complementM

-- Target statement to be proved
theorem correctStatement (h1 : U = universalSet) (h2 : isComplement M) : 2 ∈ M := by
  sorry

end Proof

end correctStatement_l210_210021


namespace probability_two_shots_l210_210964

open ProbabilityTheory

noncomputable def prob_A : ℝ := 3 / 4
noncomputable def prob_B : ℝ := 4 / 5

def event_A : Event := sorry
def event_B : Event := sorry

axiom independent_events : ∀ (e1 e2: Event), e1.independent e2 ↔ (P[e1 ∩ e2] = P[e1] * P[e2])

theorem probability_two_shots :
  by
    let outcome_1 := (1 - prob_A) * (1 - prob_B) * prob_A
    let outcome_2 := (1 - prob_A) * (1 - prob_B) * (1 - prob_A) * prob_B
    let P := outcome_1 + outcome_2
    exact (P = 19 / 400)
:= sorry

end probability_two_shots_l210_210964


namespace quadratic_root_and_coefficient_l210_210443

theorem quadratic_root_and_coefficient (c t : ℝ) (h : 2 + sqrt 3 = t ∨ 2 + sqrt 3 = 4 - t) :
  (2 + sqrt 3) * t = c ∧ t = 2 - sqrt 3 ∧ c = 1 :=
by
    sorry

end quadratic_root_and_coefficient_l210_210443


namespace complex_multiplication_l210_210387

theorem complex_multiplication {i : ℂ} (h : i^2 = -1) : i * (1 - i) = 1 + i := 
by 
  sorry

end complex_multiplication_l210_210387


namespace find_n_of_sum_of_evens_l210_210652

-- Definitions based on conditions in part (a)
def is_odd (n : ℕ) : Prop := n % 2 = 1

def sum_of_evens_up_to (n : ℕ) : ℕ :=
  let k := (n - 1) / 2
  (k / 2) * (2 + (n - 1))

-- Problem statement in Lean
theorem find_n_of_sum_of_evens : 
  ∃ n : ℕ, is_odd n ∧ sum_of_evens_up_to n = 81 * 82 ∧ n = 163 :=
by
  sorry

end find_n_of_sum_of_evens_l210_210652


namespace minimum_value_of_f_on_neg_interval_l210_210139

theorem minimum_value_of_f_on_neg_interval (f : ℝ → ℝ) 
    (h_even : ∀ x, f (-x) = f x) 
    (h_increasing : ∀ x y, 1 ≤ x → x ≤ y → y ≤ 2 → f x ≤ f y) 
  : ∀ x, -2 ≤ x → x ≤ -1 → f (-1) ≤ f x := 
by
  sorry

end minimum_value_of_f_on_neg_interval_l210_210139


namespace naomi_number_of_ways_to_1000_l210_210553

-- Define the initial condition and operations

def start : ℕ := 2

def add1 (n : ℕ) : ℕ := n + 1

def square (n : ℕ) : ℕ := n * n

-- Define a proposition that counts the number of ways to reach 1000 from 2 using these operations
def count_ways (start target : ℕ) : ℕ := sorry  -- We'll need a complex function to literally count the paths, but we'll abstract this here.

-- Theorem stating the number of ways to reach 1000
theorem naomi_number_of_ways_to_1000 : count_ways start 1000 = 128 := 
sorry

end naomi_number_of_ways_to_1000_l210_210553


namespace complex_abs_sqrt_five_l210_210319

open Complex

theorem complex_abs_sqrt_five : abs (2 + (-1 : ℂ) + 2 * (-I : ℂ)) = Real.sqrt 5 := 
by
  sorry

end complex_abs_sqrt_five_l210_210319


namespace arithmetic_sequence_a5_l210_210543

variables {a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℤ}

/-- The sum of the first 9 terms of an arithmetic sequence {a_n}, denoted by S_9, equals 45. 
    Prove that the 5th term, a_5, is 5. -/
theorem arithmetic_sequence_a5 {S_9 : ℤ} (h : S_9 = 45)
  (H : S_9 = (9 * (a_1 + a_9)) / 2) (H2 : a_1 + a_9 = 2 * a_5)
  : a_5 = 5 :=
by {
  /- Proof will be filled in later -/
  sorry
}

end arithmetic_sequence_a5_l210_210543


namespace C_n_equals_D_n_iff_n_equals_3_l210_210127

noncomputable def geometric_series_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

def C_n (n : ℕ) : ℝ :=
  geometric_series_sum 512 (1 / 2) n

def D_n (n : ℕ) : ℝ :=
  geometric_series_sum 1536 (-1 / 2) n

theorem C_n_equals_D_n_iff_n_equals_3 (n : ℕ) (h : n ≥ 1) : C_n n = D_n n ↔ n = 3 :=
by
  sorry

end C_n_equals_D_n_iff_n_equals_3_l210_210127


namespace shaded_areas_relation_l210_210999
noncomputable theory

def square_area := 1

def areaI : ℝ := 1/2
def areaII : ℝ := 1/2
def areaIII : ℝ := 2/9

theorem shaded_areas_relation :
  areaI = areaII ∧ areaI ≠ areaIII ∧ areaII ≠ areaIII := 
by sorry

end shaded_areas_relation_l210_210999


namespace arithmetic_sequence_eighth_term_l210_210228

theorem arithmetic_sequence_eighth_term 
  (a d : ℤ)
  (h1 : a + d = 17) 
  (h2 : a + 4d = 19) : 
  a + 7d = 21 :=
sorry

end arithmetic_sequence_eighth_term_l210_210228


namespace complex_conjugate_proof_l210_210929

noncomputable def complex_conjugate_lhs : ℂ := (3 - complex.I) / (1 - complex.I)
noncomputable def complex_conjugate_rhs : ℂ := 2 - complex.I

theorem complex_conjugate_proof : complex.conj complex_conjugate_lhs = complex_conjugate_rhs :=
sorry

end complex_conjugate_proof_l210_210929


namespace popcorn_kernels_total_l210_210120

def kernel_needed (cups_needed : ℕ) (kernel_ratio : ℕ) (popcorn_ratio : ℕ) : ℝ :=
  (cups_needed : ℝ) * (kernel_ratio : ℝ) / (popcorn_ratio : ℝ)

def joanie_kernels_needed : ℝ := kernel_needed 3 3 6
def mitchell_kernels_needed : ℝ := kernel_needed 4 2 4
def miles_davis_kernels_needed : ℝ := kernel_needed 6 4 8
def cliff_kernels_needed : ℝ := kernel_needed 3 1 3

def total_kernels_needed : ℝ :=
  joanie_kernels_needed + mitchell_kernels_needed + miles_davis_kernels_needed + cliff_kernels_needed

theorem popcorn_kernels_total : total_kernels_needed = 7.5 := by
  sorry

end popcorn_kernels_total_l210_210120


namespace number_of_integers_in_list_l210_210427

/-- The number of integers in the infinite list sqrt[n]{46656} for natural numbers n --/
theorem number_of_integers_in_list : 
  (nat.filter (λ n, (∃ (k1 k2 : ℕ), 46656 = 2^k1 * 3^k2 ∧ k1 = 6 / n ∧ k2 = 6 / n)).length = 4) := by
  sorry

end number_of_integers_in_list_l210_210427


namespace baseball_games_season_duration_l210_210251

theorem baseball_games_season_duration 
    (games_per_month : ℕ) 
    (games_per_season : ℕ) 
    (H1 : games_per_month = 7)
    (H2 : games_per_season = 14) :
    games_per_season / games_per_month = 2 :=
begin
  sorry
end

end baseball_games_season_duration_l210_210251


namespace kombucha_half_fill_l210_210969

-- Define the conditions
def doubles_every_day (n : ℕ) : Prop :=
  ∀ t : ℕ, n t = 2 * n (t - 1)

def fills_entire_jar_in (n : ℕ) (days : ℕ) : Prop :=
  n days = full_jar_area

-- Define the problem statement
theorem kombucha_half_fill (days : ℕ) :
  (doubles_every_day n) → (fills_entire_jar_in n 19) → (fills_entire_jar_in n days → days = 18) :=
  sorry

end kombucha_half_fill_l210_210969


namespace circumcircle_tangency_of_ABQ_and_CDQ_l210_210185

theorem circumcircle_tangency_of_ABQ_and_CDQ
  (A B C D P Q M N : Point)
  (h_trapezoid: Trapezoid A B C D)
  (h_perpendicular: ∠ACD = 90 ∧ ∠BD = 90)
  (h_midline: IsMidline A B C D Q)
  (h_reflection: ∃ R, Reflect P R Q)
  (h_thales: ThalesCircle P A D)
  (h_M_mid: IsMidpoint A D M)
  (h_N_mid: IsMidpoint B C N)
  :
  Tangent (Circumcircle A B Q) (Circumcircle C D Q) := 
sorry

end circumcircle_tangency_of_ABQ_and_CDQ_l210_210185


namespace arcsin_one_eq_pi_div_two_l210_210723

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = Real.pi / 2 :=
by
  -- proof steps here
  sorry

end arcsin_one_eq_pi_div_two_l210_210723


namespace money_left_after_deductions_l210_210365

-- Define the weekly income
def weekly_income : ℕ := 500

-- Define the tax deduction as 10% of the weekly income
def tax : ℕ := (10 * weekly_income) / 100

-- Define the weekly water bill
def water_bill : ℕ := 55

-- Define the tithe as 10% of the weekly income
def tithe : ℕ := (10 * weekly_income) / 100

-- Define the total deductions
def total_deductions : ℕ := tax + water_bill + tithe

-- Define the money left
def money_left : ℕ := weekly_income - total_deductions

-- The statement to prove
theorem money_left_after_deductions : money_left = 345 := by
  sorry

end money_left_after_deductions_l210_210365


namespace find_sequence_formula_l210_210438

variable (a : ℕ → ℝ)

noncomputable def sequence_formula := ∀ n : ℕ, a n = Real.sqrt n

lemma sequence_initial : a 1 = 1 :=
sorry

lemma sequence_recursive (n : ℕ) : a (n+1)^2 - a n^2 = 1 :=
sorry

theorem find_sequence_formula : sequence_formula a :=
sorry

end find_sequence_formula_l210_210438


namespace shift_cos_to_sin_shift_l210_210873

theorem shift_cos_to_sin_shift : 
  ∀ x : ℝ, sin (2 * (x + π / 8) - π / 4) = cos (π / 2 - 2 * x) := 
by
  intros x
  sorry

end shift_cos_to_sin_shift_l210_210873


namespace g_neither_even_nor_odd_l210_210105

def g (x : ℝ) : ℝ := ⌊2 * x⌋ + 1 / 3

theorem g_neither_even_nor_odd : ¬ (∀ x, g (-x) = g x) ∧ ¬ (∀ x, g (-x) = -g x) :=
by
  sorry

end g_neither_even_nor_odd_l210_210105


namespace trig_eqn_solution_l210_210282

theorem trig_eqn_solution {x : Real} (hx1 : sin x > 0) (hx2 : cos x > 0) :
  5.57 * (sin x)^3 * (1 + cos x / sin x) + (cos x)^3 * (1 + sin x / cos x) = 2 * sqrt (sin x * cos x)
  -> ∃ k : ℤ, x = Real.pi / 4 + 2 * Real.pi * k :=
by
  sorry

end trig_eqn_solution_l210_210282


namespace hemisphere_surface_area_l210_210379

theorem hemisphere_surface_area (r : ℝ) (h : r = 9) : 
  let A := π * r^2 in 
  let S := 4 * π * r^2 in 
  let half_S := 1/2 * S in
  A + half_S = 243 * π :=
by 
  sorry

end hemisphere_surface_area_l210_210379


namespace BP_equals_SD_l210_210858

noncomputable def isMidpoint (M A B : Point) : Prop :=
  dist A M = dist M B

noncomputable def intersection (L1 L2 : Line) : Point :=
  classical.some (exists_point_of_lines_intersect L1 L2)

variables {A B C D M P S : Point}
variables {AB CD BD MC AP : Line}
variables [inscribed_quad ABCD]
variables [isMidpoint M A B]
variables [P = intersection MC BD]
variables [S = intersection (line_through C P.parallel) BD]
variables [angles_eq CAD PAB (BMC / 2)]

theorem BP_equals_SD :
  dist B P = dist S D := 
  sorry

end BP_equals_SD_l210_210858


namespace count_valid_three_digit_numbers_l210_210072

theorem count_valid_three_digit_numbers : 
  ∃ n : ℕ, 
    (∀ (d1 d2 d3 : ℕ), 
      (d1 > 5 ∧ d2 > 5 ∧ d3 > 5) ∧ 
      (d1 ≠ d2 ∧ d2 ≠ d3) ∧ 
      (d3 % 2 = 0) ∧ 
      ((d1 + d2 + d3) % 3 = 0) ∧ 
      (n = d1 * 100 + d2 * 10 + d3) → 
      n ∈ {678, 792, 896, 918}) ∧
  ∀ m : ℕ, 
    m ∈ {678, 792, 896, 918} → 
    (∀ (d₁ d₂ d₃ : ℕ), (m = d₁ * 100 + d₂ * 10 + d₃) → 
      (d₁ > 5 ∧ d₂ > 5 ∧ d₃ > 5) ∧
      (d₁ ≠ d₂ ∧ d₂ ≠ d₃) ∧
      (d₃ % 2 = 0) ∧
      ((d₁ + d₂ + d₃) % 3 = 0)) :=
by sorry

end count_valid_three_digit_numbers_l210_210072


namespace ff_x_eq_7_has_4_solutions_l210_210077

def f (x : ℝ) : ℝ := 
  if x ≥ -2 then x^2 - 2 
  else x + 4

theorem ff_x_eq_7_has_4_solutions : 
  {x : ℝ | f (f x) = 7}.card = 4 :=
sorry

end ff_x_eq_7_has_4_solutions_l210_210077


namespace student_score_correct_answer_l210_210510

theorem student_score_correct_answer :
  ∃ x : ℕ, (42 * x) - 38 = 130 ∧ x = 4 := 
by
  use 4
  split
  {
    rw (mul_comm 42 4),
    norm_num,
  }
  {
    norm_num,
  }

end student_score_correct_answer_l210_210510


namespace integral_3_minus_2x_sq_integral_sec_sq_integral_tan_phi_l210_210776

-- Proof problem for integral of (3 - 2x)^2
theorem integral_3_minus_2x_sq (C : ℝ) : 
  ∫ (3 - 2 * x) ^ 2 = 9 * x - 6 * x ^ 2 + (4 / 3) * x ^ 3 + C :=
by sorry

-- Proof problem for integral of sec^2(m - nx)
theorem integral_sec_sq (m n C : ℝ) :
  ∫ sec^2 (m - n * x) = - (1 / n) * tan (m - n * x) + C :=
by sorry

-- Proof problem for integral of tan(ϕ)
theorem integral_tan_phi(C: ℝ):
  ∫ tan(ϕ) = - ln(abs(cos(ϕ))) + C :=
by sorry

end integral_3_minus_2x_sq_integral_sec_sq_integral_tan_phi_l210_210776


namespace inequality_proof_l210_210946

variables {x y : ℝ}

theorem inequality_proof (hx_pos : x > 0) (hy_pos : y > 0) (h1 : x^2 > x + y) (h2 : x^4 > x^3 + y) : x^3 > x^2 + y := 
by 
  sorry

end inequality_proof_l210_210946


namespace combination_solution_l210_210788

theorem combination_solution (x : ℝ) (hx : nat.choose 15 (nat.floor (2*x + 1)) = nat.choose 15 (nat.floor (x + 2))) :
  x = 1 ∨ x = 4 :=
sorry

end combination_solution_l210_210788


namespace expression_evaluation_l210_210294

theorem expression_evaluation :
  (Real.sqrt (1 / 4)) - (abs (Real.cbrt (-8))) + (abs (1 - Real.sqrt 2)) = (Real.sqrt 2 - 5 / 2) := 
  by
  sorry

end expression_evaluation_l210_210294


namespace complex_expression_magnitude_l210_210300

def i := Complex.I

theorem complex_expression_magnitude :
  |2 + i^2 + 2 * i^3| = Real.sqrt 5 :=
by
  sorry

end complex_expression_magnitude_l210_210300


namespace proof_2_in_M_l210_210039

def U : Set ℕ := {1, 2, 3, 4, 5}

def M : Set ℕ := { x | x ∈ U ∧ x ≠ 1 ∧ x ≠ 3 }

theorem proof_2_in_M : 2 ∈ M :=
by
  sorry

end proof_2_in_M_l210_210039


namespace solve_recursive_fraction_l210_210168

noncomputable def recursive_fraction (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0     => x
  | (n+1) => 1 + 1 / (recursive_fraction n x)

theorem solve_recursive_fraction (x : ℝ) (n : ℕ) :
  (recursive_fraction n x = x) ↔ (x = (1 + Real.sqrt 5) / 2 ∨ x = (1 - Real.sqrt 5) / 2) :=
sorry

end solve_recursive_fraction_l210_210168


namespace opposite_of_five_is_neg_five_l210_210211

theorem opposite_of_five_is_neg_five :
  ∃ (x : ℤ), (5 + x = 0) ∧ x = -5 :=
by
  use -5
  split
  · simp
  · rfl

end opposite_of_five_is_neg_five_l210_210211


namespace share_of_a_l210_210545

variables {a b c d : ℝ}
variables {total : ℝ}

-- Conditions
def condition1 (a b c d : ℝ) := a = (3/5) * (b + c + d)
def condition2 (a b c d : ℝ) := b = (2/3) * (a + c + d)
def condition3 (a b c d : ℝ) := c = (4/7) * (a + b + d)
def total_distributed (a b c d : ℝ) := a + b + c + d = 1200

-- Theorem to prove
theorem share_of_a (a b c d : ℝ) (h1 : condition1 a b c d) (h2 : condition2 a b c d) (h3 : condition3 a b c d) (h4 : total_distributed a b c d) : 
  a = 247.5 :=
sorry

end share_of_a_l210_210545


namespace shells_needed_l210_210901

theorem shells_needed (current_shells : ℕ) (total_shells : ℕ) (difference : ℕ) :
  current_shells = 5 → total_shells = 17 → difference = total_shells - current_shells → difference = 12 :=
by
  intros h1 h2 h3
  sorry

end shells_needed_l210_210901


namespace color_divisors_with_conditions_l210_210529

/-- Define the primes, product of the first 100 primes, and set S -/
def first_100_primes : List Nat := sorry -- Assume we have the list of first 100 primes
def product_of_first_100_primes : Nat := first_100_primes.foldr (· * ·) 1
def S := {d : Nat | d > 1 ∧ ∃ m, product_of_first_100_primes = m * d}

/-- Statement of the problem in Lean 4 -/
theorem color_divisors_with_conditions :
  (∃ (k : Nat), (∀ (coloring : S → Fin k), 
    (∀ s1 s2 s3 : S, (s1 * s2 * s3 = product_of_first_100_primes) → (coloring s1 = coloring s2 ∨ coloring s1 = coloring s3 ∨ coloring s2 = coloring s3)) ∧
    (∀ c : Fin k, ∃ s : S, coloring s = c))) ↔ k = 100 := 
by
  sorry

end color_divisors_with_conditions_l210_210529


namespace proof_2_in_M_l210_210037

def U : Set ℕ := {1, 2, 3, 4, 5}

def M : Set ℕ := { x | x ∈ U ∧ x ≠ 1 ∧ x ≠ 3 }

theorem proof_2_in_M : 2 ∈ M :=
by
  sorry

end proof_2_in_M_l210_210037


namespace angle_half_in_second_or_fourth_quadrant_l210_210444
noncomputable theory
open Real

-- Given conditions of the angle alpha being in the fourth quadrant
def angle_in_fourth_quadrant (α : ℝ) (k : ℤ) := (2 * k * π - π / 2 < α ∧ α < 2 * k * π)

-- Lean theorem statement
theorem angle_half_in_second_or_fourth_quadrant (α : ℝ) (k : ℤ)
  (h : angle_in_fourth_quadrant α k) : (∃ m : ℤ, (m * π - π / 4 < α / 2 ∧ α / 2 < (m + 1) * π / 2) ∨ ((m + 1) * π / 2 < α / 2 ∧ α / 2 < (m + 1) * π)) :=
  sorry

end angle_half_in_second_or_fourth_quadrant_l210_210444


namespace function_even_and_decreasing_l210_210703

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_strictly_decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, x ∈ I → y ∈ I → x < y → f y < f x

theorem function_even_and_decreasing : 
  is_even_function (λ x : ℝ, x ^ -2) ∧ is_strictly_decreasing_on (λ x : ℝ, x ^ -2) (set.Ioi 0) :=
by
  sorry

end function_even_and_decreasing_l210_210703


namespace max_area_triangle_BPC_l210_210890

namespace TriangleProblem

open Real

-- Define the given constants
def AB := 12
def BC := 12
def CA := 20

-- Define the derived constants based on the Law of Cosines
noncomputable def cosBAC : ℝ := (AB^2 + CA^2 - BC^2) / (2 * AB * CA)
noncomputable def angleBAC : ℝ := Real.arccos cosBAC
noncomputable def angleCPB : ℝ := pi - (angleBAC / 2)

-- Define the maximum possible area of triangle BPC
noncomputable def maxAreaBPC : ℝ := 72 * sin (angleCPB / 2)

-- The theorem we need to prove: the maximum possible area of ΔBPC
theorem max_area_triangle_BPC : maxAreaBPC = 72 * sin (Real.arccos (5/6) / 2) :=
sorry

end TriangleProblem

end max_area_triangle_BPC_l210_210890


namespace deposit_amount_l210_210179

theorem deposit_amount (x : ℝ) (net_interest : ℝ) (rate : ℝ) (tax : ℝ) :
  net_interest = (x * rate / 100) - (x * rate / 100 * tax / 100) →
  x = 30000 :=
by
  sorry

example : deposit_amount 30000 540 2.25 20 :=
by
  sorry

end deposit_amount_l210_210179


namespace sacks_after_days_l210_210850

-- Define the number of sacks harvested per day
def harvest_per_day : ℕ := 74

-- Define the number of sacks discarded per day
def discard_per_day : ℕ := 71

-- Define the days of harvest
def days_of_harvest : ℕ := 51

-- Define the number of sacks that are not discarded per day
def net_sacks_per_day : ℕ := harvest_per_day - discard_per_day

-- Define the total number of sacks after the specified days of harvest
def total_sacks : ℕ := days_of_harvest * net_sacks_per_day

theorem sacks_after_days :
  total_sacks = 153 := by
  sorry

end sacks_after_days_l210_210850


namespace find_N_l210_210231

theorem find_N (a b c N : ℝ) (h1 : a + b + c = 120) (h2 : a - 10 = N) 
               (h3 : b + 10 = N) (h4 : 7 * c = N): N = 56 :=
by
  sorry

end find_N_l210_210231


namespace correct_statement_l210_210015

universe u
variable (α : Type u)

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5}

noncomputable def complement_U_M : Set ℕ := {1, 3}

noncomputable def M : Set ℕ := U α \ complement_U_M

theorem correct_statement : 2 ∈ M :=
by {
  sorry
}

end correct_statement_l210_210015


namespace subset_proportion_odd_smallest_l210_210797

theorem subset_proportion_odd_smallest (n : ℕ) (hn : 0 < n) :
  let total_subsets := 2 ^ (2 * n)
  let odd_smallest_subsets := ∑ i in finset.range n, 2 ^ (2 * n - 2 * i - 1)
  let proportion := odd_smallest_subsets / total_subsets in
  proportion = (1 - 4 ^ (-n : ℤ)) / 3 :=
by
  sorry

end subset_proportion_odd_smallest_l210_210797


namespace right_triangle_acute_angles_l210_210088

theorem right_triangle_acute_angles (a b : ℝ)
  (h_right_triangle : a + b = 90)
  (h_ratio : a / b = 3 / 2) :
  (a = 54) ∧ (b = 36) :=
by
  sorry

end right_triangle_acute_angles_l210_210088


namespace cos_dihedral_angle_SND_A_distance_from_A_to_plane_SND_l210_210711

-- Define points and conditions
structure Point3D :=
(x: ℝ) (y: ℝ) (z: ℝ)

def A := Point3D.mk 0 0 0
def B := Point3D.mk 4 0 0
def C := Point3D.mk 0 2 0
def S := Point3D.mk 0 0 2

def midpoint (p1 p2: Point3D) : Point3D := 
  Point3D.mk ((p1.x + p2.x) / 2) ((p1.y + p2.y) / 2) ((p1.z + p2.z) / 2)

def N := midpoint A B
def D := midpoint B C

-- Lean 4 statement for question 1 and question 2
theorem cos_dihedral_angle_SND_A :
  cos_dihedral_angle (S, N, D, A) = 1 / sqrt 2 := sorry

theorem distance_from_A_to_plane_SND :
  distance_point_to_plane A (S, N, D) = 2 * sqrt 2 := sorry

end cos_dihedral_angle_SND_A_distance_from_A_to_plane_SND_l210_210711


namespace distance_to_other_focus_l210_210865

theorem distance_to_other_focus 
  (P : ℝ × ℝ) 
  (hP_ellipse : (P.1^2) / 25 + (P.2^2) / 9 = 1) 
  (d_to_one_focus : ℝ) 
  (h_d : d_to_one_focus = 3) 
  : ∃ (d_to_other_focus : ℝ), d_to_other_focus = 7 :=
by 
  use 7
  sorry

end distance_to_other_focus_l210_210865


namespace cube_volume_in_pyramid_and_cone_l210_210685

noncomputable def volume_of_cube
  (base_side : ℝ)
  (pyramid_height : ℝ)
  (cone_radius : ℝ)
  (cone_height : ℝ)
  (cube_side_length : ℝ) : ℝ := 
  cube_side_length^3

theorem cube_volume_in_pyramid_and_cone :
  let base_side := 2
  let pyramid_height := Real.sqrt 3
  let cone_radius := Real.sqrt 2
  let cone_height := Real.sqrt 3
  let cube_side_length := (Real.sqrt 6) / (Real.sqrt 2 + Real.sqrt 3)
  volume_of_cube base_side pyramid_height cone_radius cone_height cube_side_length = (6 * Real.sqrt 6) / 17 :=
by sorry

end cube_volume_in_pyramid_and_cone_l210_210685


namespace proof_f_prime_at_2_l210_210184

noncomputable def f_prime (x : ℝ) (f_prime_2 : ℝ) : ℝ :=
  2 * x + 2 * f_prime_2 - (1 / x)

theorem proof_f_prime_at_2 :
  ∃ (f_prime_2 : ℝ), f_prime 2 f_prime_2 = -7 / 2 :=
by
  sorry

end proof_f_prime_at_2_l210_210184


namespace option_A_option_B_option_D_l210_210269

-- Option A
theorem option_A (a b : ℤ × ℤ) (h₁ : a = (1, 2)) (h₂ : b = (3, 1)) : ¬ collinear a b :=
by sorry

-- Option B
theorem option_B (A B C D : ℤ × ℤ) 
  (h₁ : A = (5, -1)) (h₂ : B = (-1, 7)) (h₃ : C = (1, 2)) : 
  parallelogram A B C D → D = (7, -6) :=
by sorry

-- Option D
theorem option_D (a b : ℝ × ℝ) (h₁ : a = (1, 1)) 
  (h₂ : |b| = 4) (θ : ℝ) (h₃ : θ = π / 4) : 
  projection a b = (2, 2) :=
by sorry

end option_A_option_B_option_D_l210_210269


namespace maximize_probability_resilience_l210_210761

theorem maximize_probability_resilience (x : ℝ) (h : x > 0)
  (p : ℕ → ℝ) (K : vector ℕ 10)
  (p_def : ∀ s, p s = x^s / (1 + x + x^2))
  (K_def : K = ⟨[1, 2, 2, 1, 0, 2, 1, 0, 1, 2], by simp [vector.length]⟩) :
  x = (Real.sqrt 97 + 1) / 8 := 
sorry

end maximize_probability_resilience_l210_210761


namespace hannah_stocking_stuffers_l210_210057

theorem hannah_stocking_stuffers (candy_caness : ℕ) (beanie_babies : ℕ) (books : ℕ) (kids : ℕ) : 
  candy_caness = 4 → 
  beanie_babies = 2 → 
  books = 1 → 
  kids = 3 → 
  candy_caness + beanie_babies + books = 7 → 
  7 * kids = 21 := 
by sorry

end hannah_stocking_stuffers_l210_210057


namespace money_left_after_purchase_l210_210734

noncomputable def initial_money : ℝ := 200
noncomputable def candy_bars : ℝ := 25
noncomputable def bags_of_chips : ℝ := 10
noncomputable def soft_drinks : ℝ := 15

noncomputable def cost_per_candy_bar : ℝ := 3
noncomputable def cost_per_bag_of_chips : ℝ := 2.5
noncomputable def cost_per_soft_drink : ℝ := 1.75

noncomputable def discount_candy_bars : ℝ := 0.10
noncomputable def discount_bags_of_chips : ℝ := 0.05
noncomputable def sales_tax : ℝ := 0.06

theorem money_left_after_purchase : initial_money - 
  ( ((candy_bars * cost_per_candy_bar * (1 - discount_candy_bars)) + 
    (bags_of_chips * cost_per_bag_of_chips * (1 - discount_bags_of_chips)) + 
    (soft_drinks * cost_per_soft_drink)) * 
    (1 + sales_tax)) = 75.45 := by
  sorry

end money_left_after_purchase_l210_210734


namespace doughnut_machine_completion_time_l210_210334

/-- The machine starts at 9:00 AM and by 12:00 PM it has finished one fourth of the day's job.
    Prove that the doughnut machine will complete the job at 9:00 PM. -/
theorem doughnut_machine_completion_time :
  ∀ (start finish : Time),
  start = (Time.mk 9 0 0) ∧
  finish = (Time.mk 12 0 0) ∧
  (finish - start).to_hours = 3 →
  start.to_hours + 12 = finish.to_hours + 9 :=
by
  intros start finish,
  intro h,
  cases h with start_eq h,
  cases h with finish_eq h,
  cases h,
  sorry

end doughnut_machine_completion_time_l210_210334


namespace power_identity_l210_210848

theorem power_identity {a n m k : ℝ} (h1: a^n = 2) (h2: a^m = 3) (h3: a^k = 4) :
  a^(2 * n + m - 2 * k) = 3 / 4 :=
by
  sorry

end power_identity_l210_210848


namespace f_lt_1_l210_210349

noncomputable def f (x : ℝ) : ℝ := sorry

lemma non_negative_f : ∀ x : ℝ, f(x) ≥ 0 := sorry

lemma condition_inequality : ∀ x : ℝ, f(x+1)^2 + f(x+1) - 1 ≤ f(x)^2 := sorry

lemma initial_interval (x : ℝ) (hx : x ∈ set.Icc 0 1) : f(x) = abs (x - 1/2) := sorry

theorem f_lt_1 (x : ℝ) (hx : 0 ≤ x) : f(x) < 1 :=
by
  sorry

end f_lt_1_l210_210349


namespace average_glasses_per_box_l210_210375

-- Definitions of the given variables and conditions
variables {smallerBox largerBox totalGlasses : ℕ} {x : ℕ}

-- Conditions in the problem
def smallerBox := 12
def largerBox := 16
def totalGlasses := 480
def numSmallerBoxes := x
def numLargerBoxes := x + 16

-- The total number of glasses is given by this equation
def totalExpression := smallerBox * x + largerBox * (x + 16)

-- Statement to prove the average
theorem average_glasses_per_box : totalExpression = totalGlasses → (totalGlasses / (numSmallerBoxes + numLargerBoxes) = 15) := by
  sorry

end average_glasses_per_box_l210_210375


namespace exists_XY_contains_or_neither_l210_210928

/-- Let \(A_{1}, A_{2}, \ldots, A_{n}\) be subsets of the set of natural numbers. 
Prove that there exist natural numbers \(X\) and \(Y\) such that each of these subsets either contains 
both \(X\) and \(Y\), or contains neither \(X\) nor \(Y\). -/
theorem exists_XY_contains_or_neither (n : ℕ) (A : fin n → set ℕ) : 
  ∃ (X Y : ℕ), ∀ i : fin n, (X ∈ A i ∧ Y ∈ A i) ∨ (X ∉ A i ∧ Y ∉ A i) :=
sorry

end exists_XY_contains_or_neither_l210_210928


namespace complex_magnitude_l210_210325

open Complex

theorem complex_magnitude :
  ∀ (i : ℂ), i^2 = -1 → i^3 = -i → |2 + i^2 + 2 * i^3| = Real.sqrt 5 :=
by
  intros i h1 h2
  -- skipping proof with sorry
  sorry

end complex_magnitude_l210_210325


namespace find_value_of_f2_l210_210446

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)

theorem find_value_of_f2 : f 2 = 101 / 99 :=
  sorry

end find_value_of_f2_l210_210446


namespace proposition_A_necessary_for_B_proposition_A_not_sufficient_for_B_l210_210792

theorem proposition_A_necessary_for_B (h : ℝ) (a b : ℝ) (h_pos : h > 0) :
  (|a - 1| < h ∧ |b - 1| < h) → |a - b| < 2 * h :=
by {
  assume B : |a - 1| < h ∧ |b - 1| < h,
  cases B with ha hb,
  calc
    |a - b| = |a - 1 + (1 - b)| : by ring
        ... ≤ |a - 1| + |1 - b| : abs_add
        ... < h + h : by { rw [abs_lt] at ha hb, exact add_lt_add ha.2 hb.2 }
        ... = 2 * h : by ring,
}

theorem proposition_A_not_sufficient_for_B (h : ℝ) (h_pos : h > 0) :
  ¬(∀ a b : ℝ, |a - b| < 2 * h → (|a - 1| < h ∧ |b - 1| < h)) :=
by {
  assume h1 : ∀ a b : ℝ, |a - b| < 2 * h → (|a - 1| < h ∧ |b - 1| < h),
  let a := 0.5,
  let b := -0.3,
  have hab : |a - b| < 2 * h,
  { calc
    |a - b| = |0.5 - -0.3| : by ring
        ... = 0.8 : by norm_num
        ... < 2 * 1 : by norm_num,
  },
  have Bfalse : ¬(|a - 1| < h ∧ |b - 1| < h),
  { calc
    |a - 1| = |0.5 - 1| : by norm_num
        ... = 0.5 : by norm_num,
    have hb : ¬(0.3 + 1 < h),
    calc
    |b - 1| = |-0.3 - 1| : by norm_num
        ... = 1.3 : by norm_num,
  },
  contradiction,
}

end proposition_A_necessary_for_B_proposition_A_not_sufficient_for_B_l210_210792


namespace fraction_subtraction_l210_210265

theorem fraction_subtraction : (5 / 6) - (1 / 12) = (3 / 4) := 
by 
  sorry

end fraction_subtraction_l210_210265


namespace probability_odd_divisor_of_15_fac_l210_210732

theorem probability_odd_divisor_of_15_fac (n : ℕ) (h : n = 15!) :
  (probability (λ d, d ∣ n ∧ odd d)) = 1 / 12 :=
by sorry

end probability_odd_divisor_of_15_fac_l210_210732


namespace total_crayons_l210_210995

theorem total_crayons (orange_boxes : ℕ) (orange_per_box : ℕ) (blue_boxes : ℕ) (blue_per_box : ℕ) (red_boxes : ℕ) (red_per_box : ℕ) : 
  orange_boxes = 6 → orange_per_box = 8 → 
  blue_boxes = 7 → blue_per_box = 5 →
  red_boxes = 1 → red_per_box = 11 → 
  orange_boxes * orange_per_box + blue_boxes * blue_per_box + red_boxes * red_per_box = 94 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  exact sorry

end total_crayons_l210_210995


namespace chord_length_correct_l210_210670

noncomputable def length_of_chord(parametric_line : ℝ → ℝ × ℝ) (circle_radius : ℝ) : ℝ :=
  let line_standard_form (t : ℝ) := 
    let (x, y) := parametric_line t
    (x - 1) - 2 * (y - 2)
  (2 * sqrt (circle_radius^2 - (3 / sqrt(5))^2))

theorem chord_length_correct :
  ∀ t : ℝ,
  length_of_chord 
    (λ t, (1 + 2*t, 2 + t))
    3 = 12 * sqrt(5) / 5 := 
by
  sorry

end chord_length_correct_l210_210670


namespace younger_students_count_l210_210507

theorem younger_students_count (Y O : ℕ) (total_students : Y + O = 35)
    (younger_students_team : 2 * Y = 5 * younger_students_team)
    (older_students_team : 4 * O = older_students_team)
    (equal_team_members : younger_students_team = older_students_team) :
  Y = 14 := by
  sorry

end younger_students_count_l210_210507


namespace remainder_when_divided_by_500_l210_210909

def is_greatest_integer_multiple_of_9_with_unique_digits (M : ℕ) : Prop :=
  (∀ n : ℕ, n ≠ M → n < M → ¬ (n % 9 = 0 ∧ (∃ digits : set ℕ, digits = set.range 10 ∧ ∀ x ∈ digits, digits.count x ≤ 1)))

theorem remainder_when_divided_by_500 (M : ℕ) (h1 : is_greatest_integer_multiple_of_9_with_unique_digits M) :
  M % 500 = 190 :=
sorry

end remainder_when_divided_by_500_l210_210909


namespace solve_system_exists_l210_210956

theorem solve_system_exists (x y z : ℝ) 
  (h1 : x + y + z = 3) 
  (h2 : 1 / x + 1 / y + 1 / z = 5 / 12) 
  (h3 : x^3 + y^3 + z^3 = 45) 
  : (x, y, z) = (2, -3, 4) ∨ (x, y, z) = (2, 4, -3) ∨ (x, y, z) = (-3, 2, 4) ∨ (x, y, z) = (-3, 4, 2) ∨ (x, y, z) = (4, 2, -3) ∨ (x, y, z) = (4, -3, 2) := 
sorry

end solve_system_exists_l210_210956


namespace problem_statement_l210_210511

-- Define the isosceles triangle with properties given
def is_isosceles (A B C : ℝ) := A = B

-- Define the angles with relation to each other
def angle_relation (A B : ℝ) := B = 2 * A

-- Relative angles from altitude
def altitude_division (A B C C1 C2 : ℝ) (h_isosceles : is_isosceles A C)
  (h_angle_relation : angle_relation A B) : Prop :=
  C1 - C2 = 30

theorem problem_statement (A B C C1 C2 : ℝ) (h_isosceles : is_isosceles A C)
  (h_angle_relation : angle_relation A B) (h_altitude : altitude_division A C C C1 C2 h_isosceles h_angle_relation) :
  C1 - C2 = 30 := sorry

end problem_statement_l210_210511


namespace correctStatement_l210_210025

variable (U : Set ℕ) (M : Set ℕ)

namespace Proof

-- Given conditions
def universalSet := {1, 2, 3, 4, 5}
def complementM := {1, 3}
def isComplement (M : Set ℕ) : Prop := U \ M = complementM

-- Target statement to be proved
theorem correctStatement (h1 : U = universalSet) (h2 : isComplement M) : 2 ∈ M := by
  sorry

end Proof

end correctStatement_l210_210025


namespace train_length_l210_210360

variable L : ℝ
variable time_pole : ℝ := 24
variable time_platform : ℝ := 39
variable length_platform : ℝ := 187.5

theorem train_length
  (h1 : L = (L / time_pole) * time_pole)
  (h2 : L + length_platform = (L / time_pole) * time_platform) :
  L = 300 := by
  sorry

end train_length_l210_210360


namespace inequality_equality_condition_l210_210137

-- Definitions and conditions
variable (n : ℕ) (n_pos : 0 < n)
variable (x : Fin n → ℝ)
variable (hx : ∀ i j : Fin n, i ≤ j → x i ≤ x j)

-- The inequality part
theorem inequality (n_pos : 0 < n) (hx : ∀ i j : Fin n, i ≤ j → x i ≤ x j) : 
  (∑ i j, |x i - x j|) ^ 2 ≤ (2 * (n^2 - 1) / 3) * ∑ i j, (x i - x j) ^ 2 := 
sorry

-- The equality condition part
theorem equality_condition (n_pos : 0 < n) (hx : ∀ i j : Fin n, i ≤ j → x i ≤ x j) : 
  (∑ i j, |x i - x j|) ^ 2 = (2 * (n^2 - 1) / 3) * ∑ i j, (x i - x j) ^ 2 ↔
    ∃ d : ℝ, ∀ k : Fin n, x k = x 0 + d * k := 
sorry

end inequality_equality_condition_l210_210137


namespace opposite_of_five_l210_210208

theorem opposite_of_five : ∃ y : ℤ, 5 + y = 0 ∧ y = -5 := by
  use -5
  constructor
  . exact rfl
  . sorry

end opposite_of_five_l210_210208


namespace sum_of_all_numbers_eq_one_l210_210883

def pos_matrix (m n : ℕ) := (fin m → fin n → ℝ) -- Define a positive matrix

variables {m n : ℕ}
variables (A : pos_matrix m n)

theorem sum_of_all_numbers_eq_one 
  (row_sum col_sum : fin m → ℝ) (column_sum : fin n → ℝ)
  (h_intersection : ∀ i j, A i j = row_sum i * column_sum j) 
  (h_row_sum : ∀ i, row_sum i = ∑ j, A i j)
  (h_col_sum : ∀ j, column_sum j = ∑ i, A i j) :
  (∑ i j, A i j) = 1 :=
by 
  sorry

end sum_of_all_numbers_eq_one_l210_210883


namespace engineers_to_designers_ratio_l210_210181

-- Define the given conditions for the problem
variables (e d : ℕ) -- e is the number of engineers, d is the number of designers
variables (h1 : (48 * e + 60 * d) / (e + d) = 52)

-- Theorem statement: The ratio of the number of engineers to the number of designers is 2:1
theorem engineers_to_designers_ratio (h1 : (48 * e + 60 * d) / (e + d) = 52) : e = 2 * d :=
by {
  sorry  
}

end engineers_to_designers_ratio_l210_210181


namespace ratio_of_liquid_level_rise_l210_210258

theorem ratio_of_liquid_level_rise
  (V_narrow V_wide : ℝ)
  (r1 r2 h1 h2 x y : ℝ)
  (marble_radius : ℝ)
  (marble_volume : ℝ)
  (initial_height_ratio : h1 = 4 * h2)
  (volume_equality : V_narrow = V_wide)
  (r1_def : r1 = 4)
  (r2_def : r2 = 8)
  (marble_radius_def : marble_radius = 2)
  (marble_volume_def : marble_volume = (4 / 3) * real.pi * marble_radius^3)
  (post_marble_narrow : (1/3) * real.pi * (r1 * x)^2 * (h1 * x) = V_narrow + marble_volume)
  (post_marble_wide : (1/3) * real.pi * (r2 * y)^2 * (h2 * y) = V_wide + marble_volume) 
  (x3_def : x^3 = 1 + (2 / h1))
  (y3_def : y^3 = 1 + (2 / h2)) :
  (x - 1) / (y - 1) = 4 :=
sorry

end ratio_of_liquid_level_rise_l210_210258


namespace top_grade_prob_in_range_l210_210237

-- Define the probabilities of each part being top-grade
def top_grade_prob (i : ℕ) : ℝ :=
  if 1 ≤ i ∧ i ≤ 100 then 0.9 * (0.99^(i-1)) else 0

-- Calculate the number of top-grade parts
def num_top_grade_parts : ℝ :=
  (Finset.range 100).sum (λ i, top_grade_prob (i+1))

-- Define the Poisson theorem probability range
def pos_range_min : ℝ := 47
def pos_range_max : ℝ := 67

-- Define the probability bound
def probability_bound : ℝ := 0.8

theorem top_grade_prob_in_range :
  (num_top_grade_parts ≥ pos_range_min ∧ num_top_grade_parts ≤ pos_range_max) → probability_bound := 
by 
  sorry

end top_grade_prob_in_range_l210_210237


namespace minimum_sum_l210_210394

theorem minimum_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
    (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) + ((a^2 * b) / (18 * b * c)) ≥ 4 / 9 :=
sorry

end minimum_sum_l210_210394


namespace same_number_assigned_to_each_point_l210_210224

namespace EqualNumberAssignment

def is_arithmetic_mean (f : ℤ × ℤ → ℕ) (p : ℤ × ℤ) : Prop :=
  let (x, y) := p
  f (x, y) = (f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1)) / 4

theorem same_number_assigned_to_each_point (f : ℤ × ℤ → ℕ) :
  (∀ p : ℤ × ℤ, is_arithmetic_mean f p) → ∃ m : ℕ, ∀ p : ℤ × ℤ, f p = m :=
by
  intros h
  sorry

end EqualNumberAssignment

end same_number_assigned_to_each_point_l210_210224


namespace find_a_plus_b_l210_210203

-- Definitions of the conditions
def line1 (a : ℝ) : ℝ → ℝ := λ y, (1/4) * y + a
def line2 (b : ℝ) : ℝ → ℝ := λ x, (1/4) * x + b

def intersects_at (x y a b : ℝ) : Prop :=
  x = line1 a y ∧ y = line2 b x

-- The main theorem statement
theorem find_a_plus_b (a b : ℝ) (h : intersects_at 1 2 a b) : a + b = 9/4 :=
by
  sorry

end find_a_plus_b_l210_210203


namespace max_plus_min_times_ten_l210_210950

theorem max_plus_min_times_ten (x y z : ℝ) (h : 4 * (x + y + z) = x^2 + y^2 + z^2) :
  let M := max (xy + xz + yz : ℝ) in
  let m := min (xy + xz + yz : ℝ) in
  M + 10 * m = 28 :=
by
  sorry

end max_plus_min_times_ten_l210_210950


namespace total_scissors_l210_210242

def initial_scissors : ℕ := 54
def added_scissors : ℕ := 22

theorem total_scissors : initial_scissors + added_scissors = 76 :=
by
  sorry

end total_scissors_l210_210242


namespace negative_solutions_iff_l210_210048

theorem negative_solutions_iff (m x y : ℝ) (h1 : x - y = 2 * m + 7) (h2 : x + y = 4 * m - 3) :
  (x < 0 ∧ y < 0) ↔ m < -2 / 3 :=
by
  sorry

end negative_solutions_iff_l210_210048


namespace linear_dependent_m_value_l210_210389

open Matrix

variable {R : Type*} [Field R]

def vectors := (λ m : R, ![![2], ![5]] : Matrix (Fin 2) (Fin 1) R) : (λ m : R, Matrix (Fin 2) (Fin 1) R)

theorem linear_dependent_m_value (m : R) (a b : R) (hab : a ≠ 0 ∨ b ≠ 0) :
  ∃ (k : R), vectors m = k • ![![4], ![m]] → m = 10 :=
by
  sorry

end linear_dependent_m_value_l210_210389


namespace chicken_problem_l210_210583

theorem chicken_problem (x y z : ℕ) :
  x + y + z = 100 ∧ 5 * x + 3 * y + z / 3 = 100 → 
  (x = 0 ∧ y = 25 ∧ z = 75) ∨ 
  (x = 12 ∧ y = 4 ∧ z = 84) ∨ 
  (x = 8 ∧ y = 11 ∧ z = 81) ∨ 
  (x = 4 ∧ y = 18 ∧ z = 78) := 
sorry

end chicken_problem_l210_210583


namespace vec_calculation_l210_210045

-- Define vectors a and b
def vec_a : ℝ × ℝ := (1, 1)
def vec_b : ℝ × ℝ := (1, -1)

-- Define the vector calculation
theorem vec_calculation : (1/2) • vec_a - (3/2) • vec_b = (-2, 1) := by
  sorry

end vec_calculation_l210_210045


namespace perimeter_BB1C_greater_ABC_l210_210135

theorem perimeter_BB1C_greater_ABC
  (A B C D B1 : Point)
  (h1 : is_angle_bisector A D (triangle.mk A B C))
  (h2 : is_perpendicular (line.mk A (line.through (points.mk A D))))
  (h3 : is_perpendicular (line.mk B B1) (line.mk A (line.kt_through A (line.with_perpendicular_h (line.mk A D)))))
  : perimeter (triangle.mk B B1 C) > perimeter (triangle.mk A B C) :=
sorry

end perimeter_BB1C_greater_ABC_l210_210135


namespace bowling_ball_weight_l210_210403

theorem bowling_ball_weight (b k : ℕ) (h1 : 8 * b = 4 * k) (h2 : 3 * k = 84) : b = 14 := by
  sorry

end bowling_ball_weight_l210_210403


namespace minimum_squares_required_l210_210704

theorem minimum_squares_required (length : ℚ) (width : ℚ) (M N : ℕ) :
  (length = 121 / 2) → (width = 143 / 3) → (M / N = 33 / 26) → (M * N = 858) :=
by
  intros hL hW hMN
  -- Proof skipped
  sorry

end minimum_squares_required_l210_210704


namespace decreasing_function_in_interval_l210_210189

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def is_decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x > f y

def is_minimum_on (f : ℝ → ℝ) (I : set ℝ) (m : ℝ) : Prop :=
  ∀ x ∈ I, f x ≥ m

def is_maximum_on (f : ℝ → ℝ) (I : set ℝ) (M : ℝ) : Prop :=
  ∀ x ∈ I, f x ≤ M

theorem decreasing_function_in_interval :
  is_odd f →
  is_decreasing_on f (set.Icc 2 5) →
  is_minimum_on f (set.Icc 2 5) 2 →
  is_decreasing_on f (set.Icc (-5) (-2)) ∧
  is_maximum_on f (set.Icc (-5) (-2)) (-2) :=
by
  intro h1 h2 h3
  sorry

end decreasing_function_in_interval_l210_210189


namespace determine_odd_functions_l210_210604

-- Define the functions
def f (x : ℝ) : ℝ := -4 / x

def g (x : ℝ) : ℝ :=
  if x < 0 then x^3 - 7 * x + 1
  else x^3 - 7 * x - 1

def h (x : ℝ) : ℝ := (sqrt (2 - x^2)) / (2 - abs (x + 2))

def φ (x : ℝ) : ℝ := sqrt (9 - x^2) - sqrt (x^2 - 9)

-- Define a predicate for an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- State the theorem to prove which functions are odd
theorem determine_odd_functions :
  is_odd_function f ∧ ¬is_odd_function g ∧ is_odd_function h ∧ ¬is_odd_function φ :=
by
  sorry

end determine_odd_functions_l210_210604


namespace ellipse_foci_distance_l210_210705

theorem ellipse_foci_distance :
  let a : ℝ := 5
  let b : ℝ := 2
  √(a^2 - b^2) = √21 :=
by
  -- Defining the semi-major axis and semi-minor axis lengths
  let a := 5 : ℝ
  let b := 2 : ℝ
  -- Calculate the distance between the foci
  calc
    √(a^2 - b^2)
    = √(5^2 - 2^2) : by rfl
    = √(25 - 4) : by rfl
    = √21 : by rfl

end ellipse_foci_distance_l210_210705


namespace complex_magnitude_problem_l210_210314

-- Define the imaginary unit with the property i^2 = -1
def i : ℂ := complex.I

-- Prove that the magnitude of the complex number 2 + i² + 2i³ is √5
theorem complex_magnitude_problem : 
  complex.abs (2 + i^2 + 2 * i^3) = real.sqrt 5 := 
by
  -- Use the provided condition i² = -1
  have h : i^2 = -1 := by sorry,
  sorry

end complex_magnitude_problem_l210_210314


namespace vector_properties_l210_210820

open Real

variables (a b : ℝ^3)

def magnitude (v : ℝ^3) : ℝ := real.sqrt (v.dot v)

def angle_between_vectors (v w : ℝ^3) : ℝ :=
real.acos (v.dot w / (magnitude v * magnitude w))

theorem vector_properties
    (h1 : magnitude a = 4)
    (h2 : magnitude b = 3)
    (h3 : (2 • a - 3 • b).dot (2 • a + b) = 61) :
  angle_between_vectors a b = 2 * real.pi / 3 ∧
  magnitude (a + 2 • b) = 2 * real.sqrt 7 := 
sorry

end vector_properties_l210_210820


namespace empire_state_building_height_l210_210176

variable (height_top_floor height_antenna_spire total_height: ℝ)

theorem empire_state_building_height (h1 : height_top_floor = 1250) 
                                   (h2 : height_antenna_spire = 204) 
                                   (h3 : total_height = height_top_floor + height_antenna_spire) :
                                   total_height = 1454 := 
by 
  rw [h3, h1, h2]
  exact rfl

end empire_state_building_height_l210_210176


namespace find_wrongly_written_height_l210_210587

def wrongly_written_height
  (n : ℕ)
  (avg_height_incorrect : ℝ)
  (actual_height : ℝ)
  (avg_height_correct : ℝ) : ℝ :=
  let total_height_incorrect := n * avg_height_incorrect
  let total_height_correct := n * avg_height_correct
  let height_difference := total_height_incorrect - total_height_correct
  actual_height + height_difference

theorem find_wrongly_written_height :
  wrongly_written_height 35 182 106 180 = 176 :=
by
  sorry

end find_wrongly_written_height_l210_210587


namespace complex_expression_magnitude_l210_210296

def i := Complex.I

theorem complex_expression_magnitude :
  |2 + i^2 + 2 * i^3| = Real.sqrt 5 :=
by
  sorry

end complex_expression_magnitude_l210_210296


namespace problem_solution_l210_210003

universe u

variable (U : Set Nat) (M : Set Nat)
variable (complement_U_M : Set Nat)

axiom U_def : U = {1, 2, 3, 4, 5}
axiom complement_U_M_def : complement_U_M = {1, 3}
axiom M_def : M = U \ complement_U_M

theorem problem_solution : 2 ∈ M := by
  sorry

end problem_solution_l210_210003


namespace number_of_girls_in_class_l210_210558

variable (G B : ℕ)

-- Condition 1: total number of students is 250
def total_students_girls_boys : Prop := G + B = 250

-- Condition 2: there were twice as many girls as boys present on that day
def twice_as_many_girls_present : Prop := ∀ (G_present B_present : ℕ), G_present = G ∧ B_present = G / 2

-- Condition 3: all the girls were present
def all_girls_present : Prop := ∀ (G_present : ℕ), G_present = G

-- Condition 4: all the absent students were boys
def absent_students_boys (G_present B_present : ℕ) : Prop :=
  ∀ (B_absent : ℕ), B_absent = B - B_present

-- Final statement: to prove there are 100 girls in the class
theorem number_of_girls_in_class :
  total_students_girls_boys G B ∧
  twice_as_many_girls_present G B ∧
  all_girls_present G ∧
  ∃ (G_present B_present : ℕ), absent_students_boys G_present B_present G B →
  G = 100 :=
by
  sorry

end number_of_girls_in_class_l210_210558


namespace parallel_and_perpendicular_iff_l210_210905

variables (A B C G I O : Type)
variables [triangle ABC : (A B C : Type), AC_ne_BC : A ≠ B]
variables [centroid G ABC : (A B C G : Type), incenter I ABC : (A B C I : Type), circumcenter O ABC : (A B C O : Type)]

theorem parallel_and_perpendicular_iff (AC_ne_BC : A ≠ B) 
    (centroid_G : centroid ABC G)
    (incenter_I : incenter ABC I)
    (circumcenter_O : circumcenter ABC O)
    : (parallel (IG) (AB) ↔ perpendicular (CI) (IO)) → c^2 = -2 (ab + ac + bc) :=
begin
  sorry
end

end parallel_and_perpendicular_iff_l210_210905


namespace particle_hops_5_particle_hops_20_l210_210174

-- Define the conditions for the particle's motion
def hops (start finish : ℤ) (hop_count : ℕ) : ℕ :=
  if 2 * finish = hop_count + start then 
    Nat.choose hop_count ((hop_count + finish - start) / 2) 
  else 
    0

-- Part 1: Prove total distinct methods to land at (3, 0) after 5 hops is 5
theorem particle_hops_5 : hops 0 3 5 = 5 := 
  by
    sorry

-- Part 2: Prove total distinct methods to land at (16, 0) after 20 hops is 190
theorem particle_hops_20 : hops 0 16 20 = 190 :=
  by
    sorry

end particle_hops_5_particle_hops_20_l210_210174


namespace sequence_solution_l210_210843

noncomputable def seq (n : ℕ) : ℕ → ℝ
| 0 := 1 -- This is treated as a_1
| (n + 1) := let a_n := seq n in a_n + 1 + 2 * real.sqrt (1 + a_n)

theorem sequence_solution (n : ℕ) : seq (n - 1) = (n^2 - 1 : ℝ) :=
by
  sorry

end sequence_solution_l210_210843


namespace complex_magnitude_problem_l210_210316

-- Define the imaginary unit with the property i^2 = -1
def i : ℂ := complex.I

-- Prove that the magnitude of the complex number 2 + i² + 2i³ is √5
theorem complex_magnitude_problem : 
  complex.abs (2 + i^2 + 2 * i^3) = real.sqrt 5 := 
by
  -- Use the provided condition i² = -1
  have h : i^2 = -1 := by sorry,
  sorry

end complex_magnitude_problem_l210_210316


namespace cut_square_into_obtuse_triangle_l210_210390

theorem cut_square_into_obtuse_triangle (a : ℝ) (h : a > 0) :
  ∃ (part1 part2 part3 : set (ℝ × ℝ)), 
  (part1 ∪ part2 ∪ part3 = {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ a ∧ 0 ≤ p.2 ∧ p.2 ≤ a}) ∧
  is_triangle part1 ∧
  is_triangle part2 ∧
  is_triangle part3 ∧
  forms_obtuse_triangle part1 part2 part3 :=
sorry

end cut_square_into_obtuse_triangle_l210_210390


namespace euler_line_through_circumcenter_l210_210597

noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def incenter (A B C : Point) : Point := sorry
noncomputable def orthocenter (A B C : Point) : Point := sorry
noncomputable def euler_line (A B C : Point) : Line := sorry
noncomputable def touches (c : Circle) (p : Point) : Prop := sorry

variables {A B C A1 B1 C1 : Point}
variables (circumABC : Circle) (inABC : Circle)

axiom circumscribed_circle (circABC : Circle) :
  circABC = circumABC

axiom inscribed_circle (insABC : Circle) :
  insABC = inABC ∧ touches inABC A1 ∧ touches inABC B1 ∧ touches inABC C1

theorem euler_line_through_circumcenter :
  euler_line A1 B1 C1 = Line.mk (circumcenter A B C) (orthocenter A1 B1 C1) :=
sorry

end euler_line_through_circumcenter_l210_210597


namespace like_terms_exponents_l210_210824

theorem like_terms_exponents (m n : ℤ) 
  (h1 : m - 1 = 1) 
  (h2 : m + n = 3) : 
  m = 2 ∧ n = 1 :=
by 
  sorry

end like_terms_exponents_l210_210824


namespace minimal_sum_Sn_at_n_24_l210_210595

def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a(n + 1) - a(n) = d

def a_n (n : ℕ) : ℤ := 2 * n - 49

theorem minimal_sum_Sn_at_n_24 :
  ∀ (S_n : ℕ → ℤ), (∀ n, S_n n = ∑ i in range(n+1), a_n i) →
  ∃ n : ℕ, arithmetic_seq a_n ∧ (S_n n = ∑ i in range(25), a_n i) ∧ n = 24 :=
by
  intro S_n h_Sn,
  use 24,
  split,
  { unfold arithmetic_seq,
    use 2,
    intro n,
    dsimp [a_n],
    ring },
  split,
  { rw h_Sn,
    have h_sum_24: S_n 24 = ∑ i in range(25), a_n i, by sorry,
    exact h_sum_24 },
  { refl }

end minimal_sum_Sn_at_n_24_l210_210595


namespace ship_length_is_correct_l210_210404

-- Define the variables
variables (L E S C : ℝ)

-- Define the given conditions
def condition1 (L E S C : ℝ) : Prop := 320 * E = L + 320 * (S - C)
def condition2 (L E S C : ℝ) : Prop := 80 * E = L - 80 * (S + C)

-- Mathematical statement to be proven
theorem ship_length_is_correct
  (L E S C : ℝ)
  (h1 : condition1 L E S C)
  (h2 : condition2 L E S C) :
  L = 26 * E + (2 / 3) * E :=
sorry

end ship_length_is_correct_l210_210404


namespace find_number_l210_210666

theorem find_number (x : ℝ) :
  10 * x - 10 = 50 ↔ x = 6 := by
  sorry

end find_number_l210_210666


namespace cos_double_angle_l210_210489

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (2 * θ) = -7 / 25 :=
by
  sorry

end cos_double_angle_l210_210489


namespace job_completion_time_l210_210943

theorem job_completion_time (P : ℝ) (hP : 0 < P) :
  (∀ r : ℝ, (r > 0 → P ≠ r)) →   -- Ensures the uniqueness of P
  P + (5/12) = 4 :=               -- Given conditions and conclusion
by
  -- conditions encoding:
  have work_done_by_P : ℝ := 1 / P,                            -- P's rate of work
  have work_done_by_Q : ℝ := 1 / 15,                           -- Q's rate of work
  have together_work : ℝ := (3 / P) + (3 / 15),                -- P + Q together for 3 hours
  have P_remaining_work : ℝ := (1 / 5) * (1 / P),              -- P's remaining work in 12 mins
  have total_work_done : together_work + P_remaining_work = 1, -- Complete job
  have equation_with_job : (1 / P) := (1 + P_remaining_work),  -- Comparing equality for completeness
  have solve_for_P : P := 4,

  have : sorry,

end job_completion_time_l210_210943


namespace olivia_insurance_premium_l210_210940

theorem olivia_insurance_premium :
  ∀ (P : ℕ) (base_premium accident_percentage ticket_cost : ℤ) (tickets accidents : ℕ),
    base_premium = 50 →
    accident_percentage = P →
    ticket_cost = 5 →
    tickets = 3 →
    accidents = 1 →
    (base_premium + (accidents * base_premium * P / 100) + (tickets * ticket_cost) = 70) →
    P = 10 :=
by
  intros P base_premium accident_percentage ticket_cost tickets accidents
  intro h1 h2 h3 h4 h5 h6
  sorry

end olivia_insurance_premium_l210_210940


namespace find_g_75_l210_210190

variable (g : ℝ → ℝ)

def prop_1 := ∀ x y : ℝ, x > 0 → y > 0 → g (x * y) = g x / y
def prop_2 := g 50 = 30

theorem find_g_75 (h1 : prop_1 g) (h2 : prop_2 g) : g 75 = 20 :=
by
  sorry

end find_g_75_l210_210190


namespace question_l210_210026

variable (U : Set ℕ) (M : Set ℕ)

theorem question :
  U = {1, 2, 3, 4, 5} →
  (U \ M = {1, 3}) →
  2 ∈ M :=
by
  intros
  sorry

end question_l210_210026


namespace line_ellipse_intersection_l210_210132

theorem line_ellipse_intersection (a b : ℝ) (h1 : a^2 + b^2 = 1) (h2 : b ≠ 0)
    (h3 : ∃ (x y : ℝ), ax + by = 2 ∧ x^2/6 + y^2/2 = 1) :
    (a / b ∈ Icc (1:ℝ) ∞ ∪ Icc (-∞:ℝ) (-1:ℝ)) := sorry

end line_ellipse_intersection_l210_210132


namespace Euler_circle_of_triangle_l210_210601

noncomputable def mapping_f_assigns_circle : Prop :=
  ∀ (Δ : Triangle) (A B C D : Point),
    (∀ (σ : Similarity) (Δ1 Δ2 : Triangle), σ Δ1 = Δ2 → σ (f(Δ1)) = f(Δ2)) ∧
    (ExistsCommonPoint (f(Δ(ABC)), f(Δ(BCD)), f(Δ(CDA)), f(Δ(DAB))))
    → isEulerCircle (f(Δ))

theorem Euler_circle_of_triangle (Δ : Triangle) :
  mapping_f_assigns_circle Δ :=
sorry

end Euler_circle_of_triangle_l210_210601


namespace simplified_expression_is_zero_l210_210913

variable (a b c : ℝ)

theorem simplified_expression_is_zero (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : a + 2 * b + 2 * c = 0) :
  (1 / (b^2 + c^2 - a^2) + 1 / (a^2 + c^2 - b^2) + 1 / (a^2 + b^2 - c^2)) = 0 := 
sorry

end simplified_expression_is_zero_l210_210913


namespace fraction_of_yard_occupied_by_flower_beds_l210_210686

noncomputable def yard_dimensions := (30 : ℝ, 10 : ℝ)
noncomputable def triangle_side_length := (10 : ℝ)
noncomputable def trapezoid_parallel_sides := (40 : ℝ, 20 : ℝ)

theorem fraction_of_yard_occupied_by_flower_beds : 
  let yard_area := yard_dimensions.1 * yard_dimensions.2 in
  let triangle_area := (sqrt 3 / 4) * triangle_side_length^2 in
  let flower_beds_area := 2 * triangle_area in
  flower_beds_area / yard_area = 5 * sqrt 3 / 30 :=
  by sorry

end fraction_of_yard_occupied_by_flower_beds_l210_210686


namespace correct_statement_l210_210001

open Set

variable (U : Set ℕ) (M : Set ℕ)
variables (hU : U = {1, 2, 3, 4, 5}) (hM : U \ M = {1, 3})

theorem correct_statement : 2 ∈ M :=
by
  sorry

end correct_statement_l210_210001


namespace demonstrate_subsets_l210_210528

open Set

def M : Set ℕ := {x | x ∈ Finset.range 10000}

theorem demonstrate_subsets :
  ∃ (subsets : Finset (Finset ℕ)), subsets.card = 16 ∧ ∀ a ∈ M, (Finset.filter (λ S, S ∩ M = {a}) subsets).card = 8 :=
by
  sorry

end demonstrate_subsets_l210_210528


namespace problem_l210_210468

noncomputable def f (x a : ℝ) := x^2 * Real.log x + a * x
noncomputable def g (x b : ℝ) := -x^2 + b * x - 3

theorem problem (a b : ℝ) (log2 : Real.log 2 = 0.69) :
  (∃ k, (∀ x, f x a = k.to_smul * x + k.to_smul * 1) → a = -1/2)
  ∧ (a = 0 → (∀ x, x * g x b = 2 * f x 0 → 
     ∃ x1 x2 ∈ (set.Ioo (1/2 : ℝ) (2 : ℝ)), x1 ≠ x2 → 
     4 < b ∧ b < 7 / 2 + 2 * log2)) :=
by
  sorry

end problem_l210_210468


namespace minimum_perimeter_of_polygon_with_Q_zeros_is_8_sqrt_2_l210_210532

def Q (z : Complex) : Complex :=
  z^8 + (6 * Real.sqrt 2 + 8) * z^4 - (6 * Real.sqrt 2 + 9)

theorem minimum_perimeter_of_polygon_with_Q_zeros_is_8_sqrt_2 :
  let zeros := {z : Complex | Q z = 0}
  minimum_perimeter_among_all_8_sided_polygons_with_these_vertices zeros = 8 * Real.sqrt 2 :=
sorry

end minimum_perimeter_of_polygon_with_Q_zeros_is_8_sqrt_2_l210_210532


namespace fundamental_disagreement_l210_210594

-- Definitions based on conditions
def represents_materialism (s : String) : Prop :=
  s = "Without scenery, where does emotion come from?"

def represents_idealism (s : String) : Prop :=
  s = "Without emotion, where does scenery come from?"

-- Theorem statement
theorem fundamental_disagreement :
  ∀ (s1 s2 : String),
  (represents_materialism s1 ∧ represents_idealism s2) →
  (∃ disagreement : String,
    disagreement = "Acknowledging whether the essence of the world is material or consciousness") :=
by
  intros s1 s2 h
  existsi "Acknowledging whether the essence of the world is material or consciousness"
  sorry

end fundamental_disagreement_l210_210594


namespace line_equation_l210_210415

theorem line_equation (x y : ℝ) (c : ℝ)
  (h1 : 2 * x - y + 3 = 0)
  (h2 : 4 * x + 3 * y + 1 = 0)
  (h3 : 3 * x + 2 * y + c = 0) :
  c = 1 := sorry

end line_equation_l210_210415


namespace compute_value_N_l210_210728

theorem compute_value_N : 
  let N := ∑ i in (finset.range 201 | i > 0), if i % 4 < 2 then i^2 else -i^2 in
  N = 20098 :=
begin
  sorry
end

end compute_value_N_l210_210728


namespace unopened_box_contains_40_cards_l210_210425

theorem unopened_box_contains_40_cards :
  ∀ (initial_cards : ℕ)
    (given_away : ℕ)
    (total_cards_after_finding_box : ℕ),
    initial_cards = 26 →
    given_away = 18 →
    total_cards_after_finding_box = 48 →
  total_cards_after_finding_box - (initial_cards - given_away) = 40 :=
begin
  intros,
  sorry,
end

end unopened_box_contains_40_cards_l210_210425


namespace arithmetic_seq_fifth_term_l210_210731

theorem arithmetic_seq_fifth_term (x y : ℝ) 
  (a1 a2 a3 a4 : ℝ) 
  (h1 : a1 = 2 * x^2 + 3 * y^2) 
  (h2 : a2 = x^2 + 2 * y^2) 
  (h3 : a3 = 2 * x^2 - y^2) 
  (h4 : a4 = x^2 - y^2) 
  (d : ℝ) 
  (hd : d = -x^2 - y^2) 
  (h_arith: ∀ i j k : ℕ, i < j ∧ j < k → a2 - a1 = d ∧ a3 - a2 = d ∧ a4 - a3 = d) : 
  a4 + d = -2 * y^2 := 
by 
  sorry

end arithmetic_seq_fifth_term_l210_210731


namespace baseball_games_season_duration_l210_210250

theorem baseball_games_season_duration 
    (games_per_month : ℕ) 
    (games_per_season : ℕ) 
    (H1 : games_per_month = 7)
    (H2 : games_per_season = 14) :
    games_per_season / games_per_month = 2 :=
begin
  sorry
end

end baseball_games_season_duration_l210_210250


namespace length_GP_l210_210099

open Triangle

-- Definitions for triangle, centroids, and altitudes
noncomputable def triangleABC : Triangle :=
{ a := (11 : ℝ), b := (13 : ℝ), c := (20 : ℝ), A := (0, 0), B := (11, 0), C := (5, nonneg_of_real 13), }

noncomputable def G : Point := centroid triangleABC

noncomputable def P : Point := foot_of_altitude G triangleABC.bc

-- Statement to prove the length of GP is 11/5
theorem length_GP : length (line_segment G P) = 11 / 5 := by
  sorry

end length_GP_l210_210099


namespace maximize_probability_resilience_l210_210762

theorem maximize_probability_resilience (x : ℝ) (h : x > 0)
  (p : ℕ → ℝ) (K : vector ℕ 10)
  (p_def : ∀ s, p s = x^s / (1 + x + x^2))
  (K_def : K = ⟨[1, 2, 2, 1, 0, 2, 1, 0, 1, 2], by simp [vector.length]⟩) :
  x = (Real.sqrt 97 + 1) / 8 := 
sorry

end maximize_probability_resilience_l210_210762


namespace perfect_cubes_and_fourths_below_1000_l210_210064

def is_perfect_cube (n: ℕ) : Prop :=
  ∃ k: ℕ, k ^ 3 = n

def is_perfect_fourth (n: ℕ) : Prop :=
  ∃ k: ℕ, k ^ 4 = n

def is_perfect_twelfth (n: ℕ) : Prop :=
  ∃ k: ℕ, k ^ 12 = n

def count_perfect_cubes_below (limit: ℕ) : ℕ :=
  Finset.card (Finset.filter (λ n, is_perfect_cube n) (Finset.range limit))

def count_perfect_fourths_below (limit: ℕ) : ℕ :=
  Finset.card (Finset.filter (λ n, is_perfect_fourth n) (Finset.range limit))

def count_perfect_twelfths_below (limit: ℕ) : ℕ :=
  Finset.card (Finset.filter (λ n, is_perfect_twelfth n) (Finset.range limit))

theorem perfect_cubes_and_fourths_below_1000 : 
  let total := count_perfect_cubes_below 1000 + count_perfect_fourths_below 1000 - count_perfect_twelfths_below 1000 in
  total = 14 :=
by sorry

end perfect_cubes_and_fourths_below_1000_l210_210064


namespace sandy_earnings_correct_l210_210572

def hourly_rate : ℕ := 15
def hours_worked_friday : ℕ := 10
def hours_worked_saturday : ℕ := 6
def hours_worked_sunday : ℕ := 14

def earnings_friday : ℕ := hours_worked_friday * hourly_rate
def earnings_saturday : ℕ := hours_worked_saturday * hourly_rate
def earnings_sunday : ℕ := hours_worked_sunday * hourly_rate

def total_earnings : ℕ := earnings_friday + earnings_saturday + earnings_sunday

theorem sandy_earnings_correct : total_earnings = 450 := by
  sorry

end sandy_earnings_correct_l210_210572


namespace domain_of_rational_function_l210_210837

def rational_function (x : ℝ) : ℝ := (x + 2) / (x - 3)

theorem domain_of_rational_function :
  ∀ y : ℝ, (∃ x : ℝ, y = rational_function x) ↔ y ≠ 1 :=
sorry

end domain_of_rational_function_l210_210837


namespace evaluate_expression_l210_210640

theorem evaluate_expression :
  (3^2 - 2 + 4^1 + 7)⁻¹ * 3 = 1 / 6 :=
by
  sorry

end evaluate_expression_l210_210640


namespace probability_no_adjacent_standing_correct_l210_210965

-- Define the context and conditions
def num_people := 10
def total_outcomes := 2^num_people

-- Define the recursive sequence counting the acceptable outcomes
def favorable_outcomes : ℕ → ℕ
| 0       := 1
| 1       := 2
| n       := favorable_outcomes (n - 1) + favorable_outcomes (n - 2)

-- Calculate the probability
def probability_no_adjacent_standing : ℚ :=
  favorable_outcomes num_people / total_outcomes

-- The theorem statement
theorem probability_no_adjacent_standing_correct :
  probability_no_adjacent_standing = 123 / 1024 := by
  sorry

end probability_no_adjacent_standing_correct_l210_210965


namespace tangent_circle_ratio_one_to_one_l210_210673

theorem tangent_circle_ratio_one_to_one
    {A C B D K : Point}
    (h1 : Circle (Segment.mk A C))
    (h2 : Tangent (Line.mk B C) (Circle (Segment.mk A C)) C)
    (h3 : Intersects (Segment.mk A B) (Circle (Segment.mk A C)) D)
    (h4 : Tangent_through_point Circle D (K : Point) (Line.mk B C)) :
    divides_segment K (Segment.mk B C) (1 : ℚ) (1 : ℚ) :=
sorry

end tangent_circle_ratio_one_to_one_l210_210673


namespace average_of_six_numbers_l210_210970

theorem average_of_six_numbers (a b c d e f : ℝ)
  (h1 : (a + b) / 2 = 3.4)
  (h2 : (c + d) / 2 = 3.8)
  (h3 : (e + f) / 2 = 6.6) :
  (a + b + c + d + e + f) / 6 = 4.6 :=
by sorry

end average_of_six_numbers_l210_210970


namespace correct_choice_l210_210153

-- Define proposition p
def p : Prop := ∃ x : ℕ, x^3 < x^2

-- Define proposition q
def q : Prop := ∀ a : ℝ, (0 < a ∨ 1 < a) → (∃ c : ℝ, c * a = 1) → (log a (2 - 1)) = 0

-- Given conditions:
-- proof that p is false
def not_p : ¬p := sorry

-- proof that q is true
def is_q : q := sorry

-- The original problem
def problem (p q : Prop) := ¬p ∧ q

-- The statement to be proved
theorem correct_choice : problem p q :=
begin
  split,
  { exact not_p },
  { exact is_q }
end

end correct_choice_l210_210153


namespace shared_vertex_angle_of_triangle_and_square_l210_210259

theorem shared_vertex_angle_of_triangle_and_square (α β γ δ ε ζ η θ : ℝ) :
  (α = 60 ∧ β = 60 ∧ γ = 60 ∧ δ = 90 ∧ ε = 90 ∧ ζ = 90 ∧ η = 90 ∧ θ = 90) →
  θ = 90 :=
by
  sorry

end shared_vertex_angle_of_triangle_and_square_l210_210259


namespace weight_of_raisins_l210_210378

theorem weight_of_raisins
  (weight_peanuts weight_chocolate_chips weight_trail_mix weight_raisins : ℝ)
  (h1 : weight_peanuts = 0.17)
  (h2 : weight_chocolate_chips = 0.17)
  (h3 : weight_trail_mix = 0.42)
  (h4 : weight_trail_mix = weight_peanuts + weight_chocolate_chips + weight_raisins) :
  weight_raisins = 0.08 :=
by
  rw [h1, h2] at h4
  linarith

end weight_of_raisins_l210_210378


namespace counts_of_perfect_cubes_and_fourth_powers_l210_210070

theorem counts_of_perfect_cubes_and_fourth_powers : 
  finset.card ({n | (∃ k, n = k^3) ∨ (∃ k, n = k^4) ∧ 0 < n ∧ n < 1000}) = 14 :=
begin
  sorry
end

end counts_of_perfect_cubes_and_fourth_powers_l210_210070


namespace train_crossing_time_approx_l210_210288

def length_of_train : ℝ := 120
def speed_of_train_kmph : ℝ := 60
def length_of_bridge : ℝ := 170

noncomputable def speed_of_train_mps : ℝ := (speed_of_train_kmph * 1000) / 3600
noncomputable def total_distance : ℝ := length_of_train + length_of_bridge
noncomputable def crossing_time : ℝ := total_distance / speed_of_train_mps

theorem train_crossing_time_approx :
  crossing_time ≈ 17.4 := by
  sorry

end train_crossing_time_approx_l210_210288


namespace proof_2_in_M_l210_210040

def U : Set ℕ := {1, 2, 3, 4, 5}

def M : Set ℕ := { x | x ∈ U ∧ x ≠ 1 ∧ x ≠ 3 }

theorem proof_2_in_M : 2 ∈ M :=
by
  sorry

end proof_2_in_M_l210_210040


namespace triangle_angle_tangent_condition_l210_210502

theorem triangle_angle_tangent_condition
  (A B C : ℝ)
  (h1 : A + C = 2 * B)
  (h2 : Real.tan A * Real.tan C = 2 + Real.sqrt 3) :
  (A = Real.pi / 4 ∧ B = Real.pi / 3 ∧ C = 5 * Real.pi / 12) ∨
  (A = 5 * Real.pi / 12 ∧ B = Real.pi / 3 ∧ C = Real.pi / 4) :=
  sorry

end triangle_angle_tangent_condition_l210_210502


namespace totalNumberOfCrayons_l210_210992

def numOrangeCrayons (numBoxes : ℕ) (crayonsPerBox : ℕ) : ℕ :=
  numBoxes * crayonsPerBox

def numBlueCrayons (numBoxes : ℕ) (crayonsPerBox : ℕ) : ℕ :=
  numBoxes * crayonsPerBox

def numRedCrayons (numBoxes : ℕ) (crayonsPerBox : ℕ) : ℕ :=
  numBoxes * crayonsPerBox

theorem totalNumberOfCrayons :
  numOrangeCrayons 6 8 + numBlueCrayons 7 5 + numRedCrayons 1 11 = 94 :=
by
  sorry

end totalNumberOfCrayons_l210_210992


namespace mark_sprinted_distance_l210_210934

def speed := 6 -- miles per hour
def time := 4 -- hours

/-- Mark sprinted exactly 24 miles. -/
theorem mark_sprinted_distance : speed * time = 24 := by
  sorry

end mark_sprinted_distance_l210_210934


namespace log_difference_l210_210491

theorem log_difference {x y a : ℝ} (h : Real.log x - Real.log y = a) :
  Real.log ((x / 2)^3) - Real.log ((y / 2)^3) = 3 * a :=
by 
  sorry

end log_difference_l210_210491


namespace no_positive_integer_n_l210_210568

def leftmost_digit (m : ℕ) : ℕ := 
  m / 10 ^ (m.digits 10).length.pred -- Function to obtain the leftmost digit of a number.

theorem no_positive_integer_n :
  ¬ ∃ (n : ℕ), n > 0 ∧
  (∀ k : ℕ, k ∈ finset.range 10 \ {0} → leftmost_digit ((n + k)!) = k) :=
by
  sorry

end no_positive_integer_n_l210_210568


namespace Mark_jump_rope_hours_l210_210548

theorem Mark_jump_rope_hours 
  (record : ℕ) 
  (jump_rate : ℕ) 
  (seconds_per_hour : ℕ) 
  (h_record : record = 54000) 
  (h_jump_rate : jump_rate = 3) 
  (h_seconds_per_hour : seconds_per_hour = 3600) 
  : (record / jump_rate) / seconds_per_hour = 5 := 
by
  sorry

end Mark_jump_rope_hours_l210_210548


namespace raft_drift_time_l210_210235

-- Define the conditions from the problem
variable (distance : ℝ := 1)
variable (steamboat_time : ℝ := 1) -- in hours
variable (motorboat_time : ℝ := 3 / 4) -- 45 minutes in hours
variable (motorboat_speed_ratio : ℝ := 2)

-- Variables for speeds
variable (vs vf : ℝ)

-- Conditions: the speeds and conditions of traveling from one village to another
variable (steamboat_eqn : vs + vf = distance / steamboat_time := by sorry)
variable (motorboat_eqn : (2 * vs) + vf = distance / motorboat_time := by sorry)

-- Time for the raft to travel the distance
theorem raft_drift_time : 90 = (distance / vf) * 60 := by
  -- Proof comes here
  sorry

end raft_drift_time_l210_210235


namespace number_of_sections_l210_210612

def total_seats : ℕ := 270
def seats_per_section : ℕ := 30

theorem number_of_sections : total_seats / seats_per_section = 9 := 
by sorry

end number_of_sections_l210_210612


namespace largest_divisor_of_expression_l210_210777

theorem largest_divisor_of_expression :
  ∃ x : ℕ, (∀ y : ℕ, x ∣ (7^y + 12*y - 1)) ∧ (∀ z : ℕ, (∀ y : ℕ, z ∣ (7^y + 12*y - 1)) → z ≤ x) :=
sorry

end largest_divisor_of_expression_l210_210777


namespace area_parallelogram_18_10_l210_210412

def base : ℝ := 18
def height : ℝ := 10
def area_of_parallelogram (b h : ℝ) : ℝ := b * h

theorem area_parallelogram_18_10 : area_of_parallelogram base height = 180 := by
  sorry

end area_parallelogram_18_10_l210_210412


namespace seating_arrangement_l210_210178

-- Define the conditions first
def martians : ℕ := 4
def venusians : ℕ := 4
def earthlings : ℕ := 4
def chairs : ℕ := 12
def chair1_occ := "Martian"
def chair12_occ := "Earthling"
def no_earthling_left_of_martian := true
def no_martian_left_of_venusian := true
def no_venusian_left_of_earthling := true

-- Assertion of the problem statement
theorem seating_arrangement (chairs = 12) 
    (martians = 4) 
    (venusians = 4) 
    (earthlings = 4)
    (chair1_occ = "Martian") 
    (chair12_occ = "Earthling") 
    (no_earthling_left_of_martian = true)
    (no_martian_left_of_venusian = true) 
    (no_venusian_left_of_earthling = true) : 
    ∃ N : ℕ, 
    (N * (factorial martians) * (factorial venusians) * (factorial earthlings)) = N * (4!)^3 
    ∧ N = 50 :=
by 
    sorry

end seating_arrangement_l210_210178


namespace saturday_price_of_coat_l210_210713

theorem saturday_price_of_coat (original_price : ℝ) (discount_1 : ℝ) (discount_2 : ℝ) :
  original_price = 200 ∧ discount_1 = 0.6 ∧ discount_2 = 0.3 →
  let first_discounted_price := original_price * (1 - discount_1) in
  let saturday_price := first_discounted_price * (1 - discount_2) in
  saturday_price = 56 :=
by intro h; cases h with h1 h2; cases h2 with h3 h4; sorry

end saturday_price_of_coat_l210_210713


namespace exists_two_participants_with_same_known_count_l210_210953

theorem exists_two_participants_with_same_known_count (n : ℕ) 
  (h_symmetric : ∀ (A B : ℕ), A < n → B < n → A ≠ B → knows A B = knows B A) :
  ∃ (A B : ℕ), A < n ∧ B < n ∧ A ≠ B ∧ (counts_friends A = counts_friends B) :=
sorry

end exists_two_participants_with_same_known_count_l210_210953


namespace number_of_incorrect_propositions_l210_210835

-- Definitions of propositions
def prop1 : Prop := (¬ (p ∧ q) → (¬ p ∧ ¬ q))
def prop2 : Prop := (∀ a b : ℝ, ¬ (a > b → 2^a > 2^b - 1) ↔ (a ≤ b → 2^a ≤ 2^b - 1))
def prop3 : Prop := (∀ (A B : ℝ) (a b R : ℝ), 
                      (A > B ↔ 2 * R * sin A = a ∧ 2 * R * sin B = b → (sin A > sin B ↔ a > b)))

-- Main theorem: There is exactly one incorrect proposition
theorem number_of_incorrect_propositions : 
  (¬ prop1 ∧ prop2 ∧ prop3) →
  (1 = (if ¬ prop1 then 1 else 0) + (if ¬ prop2 then 1 else 0) + (if ¬ prop3 then 1 else 0)) :=
by
  intros h
  sorry

end number_of_incorrect_propositions_l210_210835


namespace sodium_hypochlorite_percentage_composition_l210_210396

theorem sodium_hypochlorite_percentage_composition :
  let Na_mass := 22.99
  let O_mass := 16.00
  let Cl_mass := 35.45
  let NaOCl_molar_mass := Na_mass + O_mass + Cl_mass in
  NaOCl_molar_mass = 74.44 ∧
  (Na_mass / NaOCl_molar_mass * 100) ≈ 30.88 ∧
  (O_mass / NaOCl_molar_mass * 100) ≈ 21.49 ∧
  (Cl_mass / NaOCl_molar_mass * 100) ≈ 47.63 :=
by {
  sorry
}

end sodium_hypochlorite_percentage_composition_l210_210396


namespace unique_markings_count_l210_210284

/-- 
We are given a one-foot stick marked in 1/4 and 1/5 portions.
We need to prove that the total number of unique markings,
including the endpoints, is 9. 
-/
theorem unique_markings_count : 
  (∀ x : ℝ, x ∈ {i / 4 | i ∈ Finset.range 5} ∪ {j / 5 | j ∈ Finset.range 6}) → 
  Finset.card ({x : ℝ | ∃ (i : ℕ) (H1 : i ≤ 4), x = i / 4} 
               ∪ {x : ℝ | ∃ (j : ℕ) (H2 : j ≤ 5), x = j / 5}) = 9 :=
sorry

end unique_markings_count_l210_210284


namespace greatest_integer_in_set_l210_210290

noncomputable def median_set_x (x : List ℝ) : ℝ :=
  (x.nth_le 9 sorry + x.nth_le 10 sorry) / 2

noncomputable def range_set_x (x : List ℝ) : ℝ :=
  x.maximum sorry - x.minimum sorry

theorem greatest_integer_in_set (x : List ℝ) (h_len : x.length = 20) 
  (h_median : median_set_x x = 50) (h_range : range_set_x x = 40) : 
  (x.maximum sorry = 81) :=
sorry

end greatest_integer_in_set_l210_210290


namespace complex_magnitude_problem_l210_210310

-- Define the imaginary unit with the property i^2 = -1
def i : ℂ := complex.I

-- Prove that the magnitude of the complex number 2 + i² + 2i³ is √5
theorem complex_magnitude_problem : 
  complex.abs (2 + i^2 + 2 * i^3) = real.sqrt 5 := 
by
  -- Use the provided condition i² = -1
  have h : i^2 = -1 := by sorry,
  sorry

end complex_magnitude_problem_l210_210310


namespace find_e_l210_210997

theorem find_e (d e f : ℕ) (hd : d > 1) (he : e > 1) (hf : f > 1) :
  (∀ M : ℝ, M ≠ 1 → (M^(1/d) * (M^(1/e) * (M^(1/f)))^(1/e)^(1/d)) = (M^(17/24))^(1/24)) → e = 4 :=
by
  sorry

end find_e_l210_210997


namespace choose_officers_count_l210_210560

theorem choose_officers_count :
  let count_ways (n m: ℕ) :=
    n * m * (m - 1)
  in count_ways 15 15 + count_ways 15 15 = 6300 :=
sorry

end choose_officers_count_l210_210560


namespace probability_selecting_both_types_X_distribution_correct_E_X_correct_l210_210749

section DragonBoatFestival

/-- The total number of zongzi on the plate -/
def total_zongzi : ℕ := 10

/-- The total number of red bean zongzi -/
def red_bean_zongzi : ℕ := 2

/-- The total number of plain zongzi -/
def plain_zongzi : ℕ := 8

/-- The number of zongzi to select -/
def zongzi_to_select : ℕ := 3

/-- Probability of selecting at least one red bean zongzi and at least one plain zongzi -/
def probability_selecting_both : ℚ := 8 / 15

/-- Distribution of the number of red bean zongzi selected (X) -/
def X_distribution : ℕ → ℚ
| 0 => 7 / 15
| 1 => 7 / 15
| 2 => 1 / 15
| _ => 0

/-- Mathematical expectation of the number of red bean zongzi selected (E(X)) -/
def E_X : ℚ := 3 / 5

/-- Theorem stating the probability of selecting both types of zongzi -/
theorem probability_selecting_both_types :
  let p := probability_selecting_both
  p = 8 / 15 :=
by
  let p := probability_selecting_both
  sorry

/-- Theorem stating the probability distribution of the number of red bean zongzi selected -/
theorem X_distribution_correct :
  (X_distribution 0 = 7 / 15) ∧
  (X_distribution 1 = 7 / 15) ∧
  (X_distribution 2 = 1 / 15) :=
by
  sorry

/-- Theorem stating the mathematical expectation of the number of red bean zongzi selected -/
theorem E_X_correct :
  let E := E_X
  E = 3 / 5 :=
by
  let E := E_X
  sorry

end DragonBoatFestival

end probability_selecting_both_types_X_distribution_correct_E_X_correct_l210_210749


namespace symmetric_points_x_axis_l210_210809

theorem symmetric_points_x_axis (a b : ℝ) (P Q : ℝ × ℝ)
  (hP : P = (a + 2, -2))
  (hQ : Q = (4, b))
  (hx : (a + 2) = 4)
  (hy : b = 2) :
  (a^b) = 4 := by
sorry

end symmetric_points_x_axis_l210_210809


namespace intercepts_of_line_eq1_parallel_lines_distance_perpendicular_bisector_l210_210383

-- Declare conditions for problem 1
def line_eq1 (x y : ℝ) : Prop := 2 * x + 5 * y - 20 = 0

-- Problem 1
theorem intercepts_of_line_eq1 :
  (∃ x, line_eq1 x 0 ∧ x = 10) ∧ (∃ y, line_eq1 0 y ∧ y = 4) :=
sorry

-- Declare conditions for problem 2
def line_eq2 (x y : ℝ) : Prop := x - y + 2 = 0
def distance : ℝ := real.sqrt 2 

-- Problem 2
theorem parallel_lines_distance :
  (∃ c, ∀ x y : ℝ, (x - y + c = 0) ∧ (real.abs (c - 2) / real.sqrt (1^2 + (-1)^2) = distance)) ↔
  (c = 0 ∨ c = 4)
:= sorry

-- Problem 3
def point_M : ℝ × ℝ := (7, -1)
def point_N : ℝ × ℝ := (-5, 4)

-- Problem 3
theorem perpendicular_bisector :
  (∃ k b : ℝ, ∀ x y : ℝ, (y - (3/2) = k * (x - 1)) ∧ (24 * x - 10 * y - 9 = 0)) :=
sorry

end intercepts_of_line_eq1_parallel_lines_distance_perpendicular_bisector_l210_210383


namespace range_of_p_l210_210079

theorem range_of_p (p : ℝ) (h : p > 0) : (∀ x : ℝ, x > 0 → log x ≤ p * x - 1) ↔ p ∈ set.Ici 1 := 
sorry

end range_of_p_l210_210079


namespace constant_term_expansion_l210_210632

theorem constant_term_expansion (y: ℝ) :
  ((5 * y + 2 / y) ^ 8).expand_term = 700000 :=
by
  sorry

end constant_term_expansion_l210_210632


namespace smallest_possible_a_plus_b_l210_210448

theorem smallest_possible_a_plus_b :
  ∃ (a b : ℕ), (0 < a ∧ 0 < b) ∧ (2^10 * 7^3 = a^b) ∧ (a + b = 350753) :=
sorry

end smallest_possible_a_plus_b_l210_210448


namespace exists_k_plus_1_element_subset_all_red_l210_210533

variables (X : Set α) (n k m : ℕ)
variable [encodable X]
variables (A : finset (finset X))
variable (red : A.card = m)
variable (num_elements : X.card = n)
variable (num_red_subsets : A.card = m)
variables (k_pos : 0 < k)
variables (upper_bound : m > ((k - 1) * (n - k) + k) / k^2 * (nat.choose n (k - 1)))

theorem exists_k_plus_1_element_subset_all_red (h : (X.card = n) ∧ (A.card = m) ∧ (m > ((k - 1) * (n - k) + k) / k^2 * (nat.choose n (k - 1)))) :
  ∃ (S : finset X), S.card = k + 1 ∧ (∀ T ⊆ S, T.card = k → T ∈ A) :=
sorry

end exists_k_plus_1_element_subset_all_red_l210_210533


namespace isosceles_triangle_of_midsegment_l210_210602

theorem isosceles_triangle_of_midsegment (A B C M N : Point) : 
  midpoint B C = M → midpoint A C = N →
  segment M N ∥ segment A C → segment M N = (1/2) * segment A C →
  segment M N = (1/2) * segment A B →
  segment A C = segment A B :=
by
  intros h1 h2 h3 h4 h5
  sorry

end isosceles_triangle_of_midsegment_l210_210602


namespace inverse_passes_through_fixed_point_l210_210976

-- Given conditions for the function and its domain
variable {a : ℝ} (ha_pos : a > 0) (ha_ne_one : a ≠ 1)

-- Define the function
def f (x : ℝ) : ℝ := log a (x - 1)

-- Define the inverse function
noncomputable def f_inv (y : ℝ) : ℝ := a^y + 1

-- Theorem: The inverse function of f passes through the point (0, 2)
theorem inverse_passes_through_fixed_point :
  f_inv a 2 = 0 :=
sorry

end inverse_passes_through_fixed_point_l210_210976


namespace sum_b_n_l210_210830

-- Define the arithmetic sequence a_n with initial conditions
def a (n : ℕ) : ℤ := 2 * n - 1

-- Define the sequence b_n based on a_n and a_(n+1)
def b (n : ℕ) : ℚ := 2 / (a n * a (n + 1))

-- Proving the sum of the first n terms of b_n is (2n)/(2n + 1)
theorem sum_b_n (n : ℕ) : (finset.range n).sum (λ i, b i) = 2 * n / (2 * n + 1) := sorry

end sum_b_n_l210_210830


namespace johns_weight_l210_210898

theorem johns_weight (j m : ℝ) (h1 : j + m = 240) (h2 : j - m = j / 3) : j = 144 :=
by
  sorry

end johns_weight_l210_210898


namespace bridge_length_l210_210695

noncomputable def speed_kmph_to_mps (kmph : ℝ) : ℝ := kmph * 1000 / 3600

noncomputable def distance (speed_mps time_s : ℝ) : ℝ := speed_mps * time_s

noncomputable def length_of_bridge (distance total_length : ℝ) : ℝ := 
  distance - total_length

theorem bridge_length:
  let speed_mps := speed_kmph_to_mps 36 in
  let total_distance := distance speed_mps 27.997760179185665 in
  let length_train := 130 in
  length_of_bridge total_distance length_train = 149.97760179185665 :=
by
  sorry

end bridge_length_l210_210695


namespace days_in_year_l210_210091

theorem days_in_year (H_total : ℕ) (H_day : ℕ) : H_total = 8760 ∧ H_day = 24 → H_total / H_day = 365 := by
  intro h
  cases h with H_total_eq H_day_eq
  rw [H_total_eq, H_day_eq]
  norm_num
  sorry

end days_in_year_l210_210091


namespace candy_groups_l210_210240

theorem candy_groups (total_candies group_size : Nat) (h1 : total_candies = 30) (h2 : group_size = 3) : total_candies / group_size = 10 := by
  sorry

end candy_groups_l210_210240


namespace impossible_length_AC_l210_210892

theorem impossible_length_AC (A B C : Type) [metric_space A] 
  (AB BC : ℝ) (h_AB : AB = 3) (h_BC : BC = 2) :
  ∀ AC : ℝ, (1 < AC ∧ AC < 5) ↔ (AC ≠ 5) :=
by 
  sorry

end impossible_length_AC_l210_210892


namespace original_number_exists_l210_210146

theorem original_number_exists :
  ∃ x : ℝ, 10 * x = x + 2.7 ∧ x = 0.3 :=
by {
  sorry
}

end original_number_exists_l210_210146


namespace complex_magnitude_l210_210324

open Complex

theorem complex_magnitude :
  ∀ (i : ℂ), i^2 = -1 → i^3 = -i → |2 + i^2 + 2 * i^3| = Real.sqrt 5 :=
by
  intros i h1 h2
  -- skipping proof with sorry
  sorry

end complex_magnitude_l210_210324


namespace number_of_ordered_triples_l210_210129

noncomputable def count_triples : Nat := 50

theorem number_of_ordered_triples 
    (x y z : Nat)
    (hx : x > 0)
    (hy : y > 0)
    (hz : z > 0)
    (H1 : Nat.lcm x y = 500)
    (H2 : Nat.lcm y z = 1000)
    (H3 : Nat.lcm z x = 1000) :
    ∃ (n : Nat), n = count_triples := 
by
    use 50
    sorry

end number_of_ordered_triples_l210_210129


namespace percentage_orange_juice_in_blend_l210_210937

theorem percentage_orange_juice_in_blend :
  let pear_juice_per_pear := 10 / 2
  let orange_juice_per_orange := 8 / 2
  let pear_juice := 2 * pear_juice_per_pear
  let orange_juice := 3 * orange_juice_per_orange
  let total_juice := pear_juice + orange_juice
  (orange_juice / total_juice) = (6 / 11) := 
by
  sorry

end percentage_orange_juice_in_blend_l210_210937


namespace number_of_students_accommodated_l210_210084

noncomputable def original_average_expenditure_per_student : ℝ := 100 * A
noncomputable def increase_in_total_expenditure : ℝ := 400
noncomputable def new_total_expenditure : ℝ := 5400
noncomputable def decrease_in_average_expenditure_per_student : ℝ := 5
noncomputable def new_average_expenditure_per_student : ℝ := A - 5
noncomputable def additional_students : ℝ := x
noncomputable def total_students : ℝ := 100
noncomputable def total_students_now : ℝ := 100 + x

theorem number_of_students_accommodated (A x : ℝ) :
  (100 * A + 400 = 5400) →
  (100 * A = 5000) →
  ((A - 5) * (100 + x) = 5400) →
  x = 20 := sorry

end number_of_students_accommodated_l210_210084


namespace equation_of_locus_C_max_area_OANB_l210_210477

-- Defining conditions
def E : Point := ⟨-2, 0⟩
def F : Point := ⟨2, 0⟩

def satisfies_perpendicular (P : Point) :=
  (P.x + 2) * (P.x - 2) + P.y * P.y = 0

def locus_C := λ (M : Point),
  ∃ P : Point, satisfies_perpendicular P ∧ M.x = P.x ∧ M.y = P.y / 2

noncomputable def line_l (k : ℝ) :=
  λ (x : ℝ), k * x - 2

-- Problem 1: Equation of the locus C
theorem equation_of_locus_C :
  (∀ M : Point, locus_C M → (M.x ^ 2) / 4 + (M.y ^ 2) = 1) :=
sorry

-- Problem 2: Maximum area of quadrilateral OANB and equations of line l
theorem max_area_OANB :
  ∃ A B : Point, 
    locus_C A ∧ locus_C B ∧ 
    (∃ N : Point, N.x = A.x + B.x ∧ N.y = A.y + B.y) ∧
    (∀ k : ℝ, k ∈ { k | k = (sqrt 7) / 2 ∨ k = -(sqrt 7) / 2 } → 
      let l := line_l k in 
      let area := 2 in
      let l_eq := (l.fst * x - 2 ∨ l.snd * x - 2) in
      (area) = 2 ∧ (l_eq)) :=
sorry

end equation_of_locus_C_max_area_OANB_l210_210477


namespace base_prime_representation_196_l210_210260

-- Conditions as definitions
def is_prime (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def prime_factors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ p, is_prime p ∧ p ∣ n)

def prime_exponents (n : ℕ) : List ℕ :=
  prime_factors n |>.map (λ p, (Nat.factorize n).getOrElse p 0)

-- The main theorem to prove
theorem base_prime_representation_196 : prime_exponents 196 = [2, 0, 0, 2] :=
  sorry

end base_prime_representation_196_l210_210260


namespace circle_equation_polar_to_rectangular_line_parametric_to_standard_area_of_triangle_ABC_l210_210096

theorem circle_equation_polar_to_rectangular :
  ∀ (x y : ℝ), 
  (x^2 + y^2)^2 = 16 * (x^2 + y^2) - 32 * x * y → 
  (x - 2)^2 + (y + 2)^2 = 8 :=
by sorry

theorem line_parametric_to_standard :
  ∀ (t : ℝ) (x y : ℝ), 
  (x = t + 1) ∧ (y = t - 1) → 
  x - y = 2 :=
by sorry

theorem area_of_triangle_ABC :
  ∀ (A B : (ℝ × ℝ)) (x y : ℝ),
  (A = (2 + sqrt(2), -2 - sqrt(2))) ∧ (B = (2 - sqrt(2), -2 + sqrt(2))) →
  ((x - 2)^2 + (y + 2)^2 = 8 ∧ x - y = 2) →
  let d := sqrt(2) in
  let h := sqrt(8) - d in
  let AB := 2 * sqrt(6) in
  let S := (1 / 2) * AB * h in
  S = 2 * sqrt(3) :=
by sorry

end circle_equation_polar_to_rectangular_line_parametric_to_standard_area_of_triangle_ABC_l210_210096


namespace dragon_resilience_maximized_l210_210756

noncomputable def probability (x : ℝ) (s : ℕ) : ℝ :=
  x^s / (1 + x + x^2)

noncomputable def prob_vec (x : ℝ) : ℝ :=
  let K := [1, 2, 2, 1, 0, 2, 1, 0, 1, 2]
  K.foldr (λ s acc, acc * probability x s) 1

theorem dragon_resilience_maximized (x : ℝ) : 
  (x = (Real.sqrt 97 + 1) / 8) → 
  (0 < x) →
  ∀ K, K = [1, 2, 2, 1, 0, 2, 1, 0, 1, 2] →
  prob_vec x = (x^12 / (1 + x + x^2)^10) :=
begin
  sorry
end

end dragon_resilience_maximized_l210_210756


namespace red_paint_intensity_l210_210170

theorem red_paint_intensity (x : ℝ) (h1 : 0.5 * 10 + 0.5 * x = 15) : x = 20 :=
sorry

end red_paint_intensity_l210_210170


namespace total_amount_spent_l210_210254

-- Define constants
def price_trick_deck := 8.0
def price_gimmick_coin := 12.0
def discount_trick_decks := 0.10
def discount_gimmick_coins := 0.05
def sales_tax := 0.07
def num_trick_decks := 6 -- 3 for Tom and 3 for his friend
def num_gimmick_coins := 8 -- 4 for Tom and 4 for his friend

-- Helper functions
def total_cost (price : Float) (quantity : Int) : Float := price * quantity.toFloat
def apply_discount (cost : Float) (discount : Float) : Float := cost * (1.0 - discount)
def apply_sales_tax (cost : Float) (tax : Float) : Float := cost * (1.0 + tax)

-- Calculate costs
def trick_decks_cost := total_cost price_trick_deck num_trick_decks
def gimmick_coins_cost := total_cost price_gimmick_coin num_gimmick_coins

def discounted_trick_decks_cost := apply_discount trick_decks_cost discount_trick_decks
def discounted_gimmick_coins_cost := apply_discount gimmick_coins_cost discount_gimmick_coins

def total_cost_after_discounts := discounted_trick_decks_cost + discounted_gimmick_coins_cost
def total_cost_with_tax := apply_sales_tax total_cost_after_discounts sales_tax

-- Prove the total amount spent is approximately $143.81
theorem total_amount_spent : total_cost_with_tax = 143.81 := by
  sorry

end total_amount_spent_l210_210254


namespace smallest_k_for_mutual_criticism_l210_210504

-- Define a predicate that checks if a given configuration of criticisms lead to mutual criticism
def mutual_criticism_exists (deputies : ℕ) (k : ℕ) : Prop :=
  k ≥ 8 -- This is derived from the problem where k = 8 is the smallest k ensuring a mutual criticism

theorem smallest_k_for_mutual_criticism:
  mutual_criticism_exists 15 8 :=
by
  -- This is the theorem statement with the conditions and correct answer. The proof is omitted.
  sorry

end smallest_k_for_mutual_criticism_l210_210504


namespace packs_of_red_balls_l210_210902

/-
Julia bought some packs of red balls, R packs.
Julia bought 10 packs of yellow balls.
Julia bought 8 packs of green balls.
There were 19 balls in each package.
Julia bought 399 balls in total.
The goal is to prove that the number of packs of red balls Julia bought, R, is equal to 3.
-/

theorem packs_of_red_balls (R : ℕ) (balls_per_pack : ℕ) (packs_yellow : ℕ) (packs_green : ℕ) (total_balls : ℕ) 
  (h1 : balls_per_pack = 19) (h2 : packs_yellow = 10) (h3 : packs_green = 8) (h4 : total_balls = 399) 
  (h5 : total_balls = R * balls_per_pack + (packs_yellow + packs_green) * balls_per_pack) : 
  R = 3 :=
by
  -- Proof goes here
  sorry

end packs_of_red_balls_l210_210902


namespace card_collection_average_l210_210338

theorem card_collection_average (n : ℕ) (h : (2 * n + 1) / 3 = 2017) : n = 3025 :=
by
  sorry

end card_collection_average_l210_210338


namespace inequality_l210_210790

noncomputable def a : ℝ := Real.logBase 2 3.6
noncomputable def b : ℝ := Real.logBase 4 3.2
noncomputable def c : ℝ := Real.logBase 4 3.6

theorem inequality : a > c ∧ c > b := by
  sorry

end inequality_l210_210790


namespace prime_divides_sequence_l210_210565

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ n ∣ p, n = 1 ∨ n = p

def sequence_S (p : ℕ) : ℕ :=
  -- Implementation of sequence S(p) here.
  sorry

def sequence_123_9 : ℕ := 
  -- Implementation of sequence 123...9
  sorry

theorem prime_divides_sequence (p : ℕ) (hp : is_prime p) :
  p ∣ (sequence_S p - sequence_123_9) :=
sorry

end prime_divides_sequence_l210_210565


namespace Claire_takes_6_photos_l210_210931

-- Define the number of photos Claire has taken
variable (C : ℕ)

-- Define the conditions as stated in the problem
def Lisa_photos := 3 * C
def Robert_photos := C + 12
def same_number_photos := Lisa_photos C = Robert_photos C

-- The goal is to prove that C = 6
theorem Claire_takes_6_photos (h : same_number_photos C) : C = 6 := by
  sorry

end Claire_takes_6_photos_l210_210931


namespace basis_plane_vectors_parallelogram_vertex_D_equilateral_triangle_dot_product_projection_coordinates_l210_210276

-- Problem 1: Basis for Plane Vectors
theorem basis_plane_vectors (a b : ℝ × ℝ) (ha : a = (1, 2)) (hb : b = (3, 1)) :
  ∃ (u v : ℝ) (hu : u ≠ 0) (hv : v ≠ 0), u * a + v * b = (1, 0) ∧ u * a + v * b = (0, 1) := 
sorry

-- Problem 2: Coordinates of Vertex D
theorem parallelogram_vertex_D (A B C D : ℝ × ℝ) (hA : A = (5, -1)) (hB : B = (-1, 7)) (hC : C = (1, 2)) :
  (D = (7, -6)) :=
sorry

-- Problem 3: Equilateral Triangle Dot Product
theorem equilateral_triangle_dot_product (A B C : ℝ × ℝ) (hABC : (equilateral_triangle A B C )) :
  〈A - B, B - C〉 ≠ (π / 3) :=
sorry

-- Problem 4: Projection Coordinates
theorem projection_coordinates (a b : ℝ × ℝ) (ha : a = (1, 1)) (hb : abs b = 4) (hd : ∠(a, b) = π / 4) :
   ∃ (p : ℝ × ℝ), projection p a b = (2, 2) :=
sorry

end basis_plane_vectors_parallelogram_vertex_D_equilateral_triangle_dot_product_projection_coordinates_l210_210276


namespace correlation_problem_solution_l210_210279

theorem correlation_problem_solution :
  let A := "Grain yield and the amount of fertilizer used"
  let B := "College entrance examination scores and the time spent on review"
  let C := "Sales of goods and advertising expenses"
  let D := "The number of books sold at a fixed price of 5 yuan and sales revenue"
  (¬ correlated A) ∧ (¬ correlated B) ∧ (¬ correlated C) ∧ (¬ correlated D) → D :=
sorry

end correlation_problem_solution_l210_210279


namespace general_formula_for_a_n_l210_210844

noncomputable def λ := (-1 + Real.sqrt 5) / 2
noncomputable def μ := (-1 - Real.sqrt 5) / 2

def a (n : ℕ) : ℝ :=
if n = 1 then 1 else if n = 2 then 2 else a (n - 2) / a (n - 1)

theorem general_formula_for_a_n (n : ℕ) :
  a n = 2 ^ ((λ ^ n - μ ^ n) / (λ - μ)) :=
by sorry

end general_formula_for_a_n_l210_210844


namespace complex_magnitude_l210_210330

open Complex

theorem complex_magnitude :
  ∀ (i : ℂ), i^2 = -1 → i^3 = -i → |2 + i^2 + 2 * i^3| = Real.sqrt 5 :=
by
  intros i h1 h2
  -- skipping proof with sorry
  sorry

end complex_magnitude_l210_210330


namespace problem_I_problem_II_l210_210461

def f (x : ℝ) : ℝ := abs(2 * x - 1) - abs(2 * x - 2)
def k : ℝ := 1

theorem problem_I (x : ℝ) : (f x ≥ x) ↔ (x ≤ -1 ∨ x = 1) :=
sorry

theorem problem_II (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : (a + 2 * b = k) ∧ (2 / a + 1 / b = 4 - 1 / (a * b)) → false :=
sorry

end problem_I_problem_II_l210_210461


namespace path_traveled_is_correct_l210_210281

-- Define the original triangle and the circle.
def side_a : ℝ := 8
def side_b : ℝ := 10
def side_c : ℝ := 12.5
def radius : ℝ := 1.5

-- Define the condition that the circle is rolling inside the triangle.
def new_side (original_side : ℝ) (r : ℝ) : ℝ := original_side - 2 * r

-- Calculate the new sides of the smaller triangle path.
def new_side_a := new_side side_a radius
def new_side_b := new_side side_b radius
def new_side_c := new_side side_c radius

-- Calculate the perimeter of the path traced by the circle's center.
def path_perimeter := new_side_a + new_side_b + new_side_c

-- Prove that this perimeter equals 21.5 units under given conditions.
theorem path_traveled_is_correct : path_perimeter = 21.5 := by
  simp [new_side, new_side_a, new_side_b, new_side_c, path_perimeter]
  sorry

end path_traveled_is_correct_l210_210281


namespace proof_2_in_M_l210_210038

def U : Set ℕ := {1, 2, 3, 4, 5}

def M : Set ℕ := { x | x ∈ U ∧ x ≠ 1 ∧ x ≠ 3 }

theorem proof_2_in_M : 2 ∈ M :=
by
  sorry

end proof_2_in_M_l210_210038


namespace num_factors_of_M_l210_210853

-- Define the integer M
def M : ℕ := 2^4 * 3^3 * 7^2

-- Prove that M has 60 natural-number factors
theorem num_factors_of_M : (∀ d : ℕ, d ∣ M → d ≠ 0) → (∑ d in (range (M + 1)), if d ∣ M then 1 else 0) = 60 :=
by
  -- Proof omitted
  sorry

end num_factors_of_M_l210_210853


namespace sum_consecutive_numbers_equiv_l210_210614

theorem sum_consecutive_numbers_equiv (a b : ℕ) :
  (∃ a : ℕ, ∃ b : ℕ, 5 * (a + 2) = 2 * b + 1) :=
by {
  existsi (1 : ℕ),
  existsi ((5 * 1 + 9) / 2 : ℕ),
  sorry
}

end sum_consecutive_numbers_equiv_l210_210614


namespace find_f_neg_a_l210_210662

noncomputable def f (x : ℝ) : ℝ := x^3 * Real.cos x + 1

variable (a : ℝ)

-- Given condition
axiom h_fa : f a = 11

-- Statement to prove
theorem find_f_neg_a : f (-a) = -9 :=
by
  sorry

end find_f_neg_a_l210_210662


namespace volume_ratio_l210_210086

-- Define the original tetrahedron vertices
structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def A : Point3D := ⟨1, 0, 0⟩
def B : Point3D := ⟨0, 1, 0⟩
def C : Point3D := ⟨0, 0, 1⟩
def D : Point3D := ⟨0, 0, 0⟩

-- Define the centers of the faces
def CenterA : Point3D := ⟨1 / 3, 1 / 3, 0⟩
def CenterB : Point3D := ⟨1 / 3, 0, 1 / 3⟩
def CenterC : Point3D := ⟨0, 1 / 3, 1 / 3⟩
def CenterD : Point3D := ⟨1 / 3, 1 / 3, 1 / 3⟩

-- Define a function to calculate distance between two points (Euclidean distance)
def distance (P Q : Point3D) : ℝ :=
  Real.sqrt ((Q.x - P.x) ^ 2 + (Q.y - P.y) ^ 2 + (Q.z - P.z) ^ 2)

-- Define the volume function for a regular tetrahedron by its side length
def volume_of_regular_tetrahedron (side_length : ℝ) : ℝ :=
  (Real.sqrt 2 / 12) * side_length ^ 3

-- Calculate the side lengths of the original and smaller tetrahedron
def original_side_length : ℝ := distance A B
def smaller_side_length : ℝ := distance CenterA CenterB

theorem volume_ratio : 
  volume_of_regular_tetrahedron smaller_side_length 
  / volume_of_regular_tetrahedron original_side_length = 1 / 27 :=
by
  sorry

end volume_ratio_l210_210086


namespace geometric_mean_of_perpendiculars_l210_210787

-- Definitions for use in the proof statement
noncomputable def perpendicular_from_A_to_tangent : ℝ := sorry
noncomputable def perpendicular_from_P_to_tangent : ℝ := sorry
noncomputable def perpendicular_from_B_to_tangent : ℝ := sorry
noncomputable def perpendicular_from_P_to_AB : ℝ := sorry

-- The statement of the theorem
theorem geometric_mean_of_perpendiculars
  (AT1 PT2 BT3 : ℝ) 
  (h_PT2 : PT2 = perpendicular_from_P_to_AB)
  (h_AT1 : AT1 = perpendicular_from_A_to_tangent)
  (h_BT3 : BT3 = perpendicular_from_B_to_tangent) :
  PT2 = real.sqrt (AT1 * BT3) := sorry

end geometric_mean_of_perpendiculars_l210_210787


namespace parallelogram_above_x_axis_l210_210561

-- Define vertices of the parallelogram
def P := (-4, 4)
def Q := (4, 2)
def R := (2, -2)
def S := (-6, -4)

-- Define the probability function
def probability_not_below_x_axis (vertices : List (ℝ × ℝ)) : ℝ := 1

-- The theorem statement
theorem parallelogram_above_x_axis :
  probability_not_below_x_axis [P, Q, R, S] = 1 :=
sorry

end parallelogram_above_x_axis_l210_210561


namespace plane_section_area_l210_210683

-- Define the conditions in Lean 4
axiom radius (r : ℝ) : r = 2
axiom point_A_on_surface {A : ℝ × ℝ × ℝ} : ∃ O : ℝ × ℝ × ℝ, dist A O = 2
axiom angle_OA_plane (A : ℝ × ℝ × ℝ) (plane : ℝ × ℝ × ℝ → Prop) : ∃ θ : ℝ, θ = (real.cos θ = 1/2)

-- Define the plane section's area to be equivalent to π
theorem plane_section_area (A : ℝ × ℝ × ℝ) (plane : ℝ × ℝ × ℝ → Prop) (Q : ℝ × ℝ × ℝ) :
  (radius 2) →
  (point_A_on_surface) →
  (angle_OA_plane A plane) →
  ∃ S : ℝ, S = π :=
by
  intro h1 h2 h3
  use π
  sorry

end plane_section_area_l210_210683


namespace total_students_l210_210613

noncomputable def general_study_hall_students : ℕ := 30
noncomputable def biology_hall_students : ℕ := 2 * general_study_hall_students
noncomputable def chemistry_hall_students : ℕ := general_study_hall_students + 10
noncomputable def math_hall_students : ℕ := (3 / 5) * (general_study_hall_students + biology_hall_students + chemistry_hall_students)
noncomputable def arts_hall_students : ℕ := general_study_hall_students / 0.20

theorem total_students :
  general_study_hall_students + biology_hall_students + chemistry_hall_students + math_hall_students + arts_hall_students = 358 :=
by
  let gen := general_study_hall_students
  let bio := biology_hall_students
  let chem := chemistry_hall_students
  let math := math_hall_students
  let arts := arts_hall_students
  show gen + bio + chem + math + arts = 358
  sorry

end total_students_l210_210613


namespace average_speed_palindrome_l210_210642

theorem average_speed_palindrome :
  ∀ (initial_odometer final_odometer : ℕ) (hours : ℕ),
  initial_odometer = 123321 →
  final_odometer = 124421 →
  hours = 4 →
  (final_odometer - initial_odometer) / hours = 275 :=
by
  intros initial_odometer final_odometer hours h1 h2 h3
  sorry

end average_speed_palindrome_l210_210642


namespace sandwich_meal_plate_combo_count_l210_210355

theorem sandwich_meal_plate_combo_count :
  (Finset.card ((Finset.Icc (0, 0) (18, 10)).filter (λ (sp : ℕ × ℕ), 5 * sp.1 + 7 * sp.2 = 90)) = 3) :=
by sorry

end sandwich_meal_plate_combo_count_l210_210355


namespace silk_original_amount_l210_210089

theorem silk_original_amount (s r : ℕ) (l d x : ℚ)
  (h1 : s = 30)
  (h2 : r = 3)
  (h3 : d = 12)
  (h4 : 30 - 3 = 27)
  (h5 : x / 12 = 30 / 27):
  x = 40 / 3 :=
by
  sorry

end silk_original_amount_l210_210089


namespace problem_statement_l210_210915

theorem problem_statement : 
  let f := λ x : ℤ, x^2 - x + 2023 in 
  Int.gcd (f 202) (f 203) = 17 :=
by
  sorry

end problem_statement_l210_210915


namespace trigonometric_identity_l210_210793

theorem trigonometric_identity 
  (α β γ : ℝ) 
  (h : (sin (β + γ) * sin (γ + α)) / (cos α * cos γ) = 4 / 9) : 
  (sin (β + γ) * sin (γ + α)) / (cos (α + β + γ) * cos γ) = 4 / 5 := 
sorry

end trigonometric_identity_l210_210793


namespace rectangle_diagonal_division_l210_210352

-- Definition and proof problem statement
theorem rectangle_diagonal_division (m n : ℕ) (h₁ : m = 1000) (h₂ : n = 1979) :
  (m + n - Nat.gcd m n = 2978) :=
by
  -- Substituting the given values to evaluate the result
  rw [h₁, h₂]
  -- Proving the result using the given formula
  have gcd_result : Nat.gcd 1000 1979 = 1 := by
    -- Calculating gcd using the properties of gcd
    -- Skipping detailed steps of Euclidean algorithm, as the gcd value is known.
    sorry
  rw [gcd_result]
  -- Finish the calculation, stating the result is true
  norm_num
  sorry

end rectangle_diagonal_division_l210_210352


namespace model_A_selected_count_l210_210675

def production_A := 1200
def production_B := 6000
def production_C := 2000
def total_selected := 46

def total_production := production_A + production_B + production_C

theorem model_A_selected_count :
  (production_A / total_production) * total_selected = 6 := by
  sorry

end model_A_selected_count_l210_210675


namespace square_area_l210_210692

noncomputable def square_area_condition (x : ℝ) : ℝ :=
  x^2 + 2 * x + 1

theorem square_area :
  let x₁ := -1 + Real.sqrt 7,
      x₂ := -1 - Real.sqrt 7,
      side_length := abs (x₁ - x₂)
  in square_area_condition x₁ = 7 ∧ square_area_condition x₂ = 7 ∧ 
     side_length^2 = 28 := 
by
  have h₁ : square_area_condition (-1 + Real.sqrt 7) = 7 := 
    by sorry
  have h₂ : square_area_condition (-1 - Real.sqrt 7) = 7 := 
    by sorry
  have h_side_length : abs ((-1 + Real.sqrt 7) - (-1 - Real.sqrt 7)) = 2 * Real.sqrt 7 := 
    by sorry
  have h_area : (2 * Real.sqrt 7)^2 = 28 := 
    by sorry
  exact ⟨h₁, h₂, h_area⟩

end square_area_l210_210692


namespace angle_same_terminal_side_l210_210585

theorem angle_same_terminal_side (k : ℤ) : ∃ α : ℝ, α = k * 360 - 30 ∧ 0 ≤ α ∧ α < 360 → α = 330 :=
by
  sorry

end angle_same_terminal_side_l210_210585


namespace jacob_fraction_of_phoebe_age_l210_210885

-- Definitions
def Rehana_current_age := 25
def Rehana_future_age (years : Nat) := Rehana_current_age + years
def Phoebe_future_age (years : Nat) := (Rehana_future_age years) / 3
def Phoebe_current_age := Phoebe_future_age 5 - 5
def Jacob_age := 3
def fraction_of_Phoebe_age := Jacob_age / Phoebe_current_age

-- Theorem statement
theorem jacob_fraction_of_phoebe_age :
  fraction_of_Phoebe_age = 3 / 5 :=
  sorry

end jacob_fraction_of_phoebe_age_l210_210885


namespace fraction_identity_l210_210490

theorem fraction_identity (a b : ℚ) (h : (a - 2 * b) / b = 3 / 5) : a / b = 13 / 5 :=
sorry

end fraction_identity_l210_210490


namespace puppies_per_dog_l210_210717

/--
Chuck breeds dogs. He has 3 pregnant dogs.
They each give birth to some puppies. Each puppy needs 2 shots and each shot costs $5.
The total cost of the shots is $120. Prove that each pregnant dog gives birth to 4 puppies.
-/
theorem puppies_per_dog :
  let num_dogs := 3
  let cost_per_shot := 5
  let shots_per_puppy := 2
  let total_cost := 120
  let cost_per_puppy := shots_per_puppy * cost_per_shot
  let total_puppies := total_cost / cost_per_puppy
  (total_puppies / num_dogs) = 4 := by
  sorry

end puppies_per_dog_l210_210717


namespace count_sufficient_conditions_l210_210449

variables {α β : Type*} [plane α] [plane β]
variable {ℓ : Type*} [line ℓ]
variables {a b c : ℓ}

def parallel (x y : ℓ) : Prop := ∀ p ∈ x, p ∈ y
def perpendicular (x y : ℓ) : Prop := ∃ p, p ∈ x ∩ y ∧ ∀ z, z ∈ y → p ≠ z

theorem count_sufficient_conditions :
  (parallel α β ∧ a ⊆ α ∧ parallel b β) →
  (parallel a c ∧ parallel b c) →
  (α ∩ β = c ∧ a ⊆ α ∧ b ⊆ β ∧ parallel a β ∧ parallel b α) →
  (perpendicular a c ∧ perpendicular b c) →
  2 :=
by
  sorry

end count_sufficient_conditions_l210_210449


namespace opposite_of_five_l210_210216

theorem opposite_of_five : -5 = -5 :=
by
sorry

end opposite_of_five_l210_210216


namespace total_people_in_line_l210_210616

theorem total_people_in_line (initial_people : ℕ) (additional_people : ℕ) 
  (h1 : initial_people = 61) (h2 : additional_people = 22) : initial_people + additional_people = 83 := 
by
  rw [h1, h2]
  exact rfl

end total_people_in_line_l210_210616


namespace decreasing_interval_of_g_l210_210627

theorem decreasing_interval_of_g :
  ∀ x : ℝ, 
    let f := λ x, sqrt 2 * sin (2 * x - π / 4) in
    let g := λ x, sqrt 2 * sin (0.5 * x - π / 12) in
    (function.is_decreasing_on (Icc (7 * π / 6) (19 * π / 6)) g) :=
by sorry

end decreasing_interval_of_g_l210_210627


namespace quadratic_range_l210_210419

theorem quadratic_range : 
  let f (x : ℝ) := 2 * x ^ 2 + 4 * x - 5 in
  ∀ y : ℝ, (∃ x : ℝ, -3 ≤ x ∧ x < 2 ∧ f x = y) ↔ (-7 ≤ y ∧ y < 11) :=
by
  sorry

end quadratic_range_l210_210419


namespace equilateral_triangle_count_l210_210193

theorem equilateral_triangle_count :
  let S := {k : ℤ | -12 ≤ k ∧ k ≤ 12}
  ∃ (n : ℕ), n = 1152 ∧
  ∀ k ∈ S, ∀ x y : ℝ, y = k ∨ y = sqrt 3 * x + 2 * k ∨ y = -sqrt 3 * x + 2 * k →
  ∃ (triangles : set (ℝ × ℝ × ℝ × ℝ × ℝ × ℝ)),
  ∀ t ∈ triangles, is_equilateral t ∧ side_length t = 1 ∧ triangles.count = n
:=
  sorry

end equilateral_triangle_count_l210_210193


namespace correctStatement_l210_210023

variable (U : Set ℕ) (M : Set ℕ)

namespace Proof

-- Given conditions
def universalSet := {1, 2, 3, 4, 5}
def complementM := {1, 3}
def isComplement (M : Set ℕ) : Prop := U \ M = complementM

-- Target statement to be proved
theorem correctStatement (h1 : U = universalSet) (h2 : isComplement M) : 2 ∈ M := by
  sorry

end Proof

end correctStatement_l210_210023


namespace range_of_a_l210_210080

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, x^2 + 4 * (y - a)^2 = 4 ∧ x^2 = 2 * y) ↔ -1 ≤ a ∧ a ≤ 17 / 8 :=
by
  sorry

end range_of_a_l210_210080


namespace max_sqrt2_l210_210204

theorem max_sqrt2 (x : ℝ) : 
  let y := sin(2 * x) - 2 * (sin x)^2 + 1 in 
  y ≤ sqrt(2) :=
sorry

end max_sqrt2_l210_210204


namespace solve_transformed_quadratic_eq_l210_210630

theorem solve_transformed_quadratic_eq :
  (∀ x : ℝ, x^2 - 2 * x - 3 = 0 ↔ (x = -1 ∨ x = 3)) →
  (∀ x : ℝ, (2 * x + 1)^2 - 2 * (2 * x + 1) - 3 = 0 ↔ (x = 1 ∨ x = -1)) :=
begin
  intros h,
  -- The proof starts here, but marked as sorry since we're generating only the statement
  sorry
end

end solve_transformed_quadratic_eq_l210_210630


namespace car_speed_first_hour_l210_210989

theorem car_speed_first_hour :
  ∃ x : ℕ, let total_distance := (x + 30) in let total_time := 2 in (total_distance / total_time = 25) → x = 20 := 
by
  sorry

end car_speed_first_hour_l210_210989


namespace ellipse_condition_sufficient_not_necessary_l210_210186

theorem ellipse_condition_sufficient_not_necessary (n : ℝ) :
  (-1 < n) ∧ (n < 2) → 
  (2 - n > 0) ∧ (n + 1 > 0) ∧ (2 - n > n + 1) :=
by
  intro h
  sorry

end ellipse_condition_sufficient_not_necessary_l210_210186


namespace log_expression_evaluation_l210_210716

theorem log_expression_evaluation :
  1 + log 10 2 * log 10 5 - log 10 2 * log 10 50 - log 3 5 * log 25 9 * log 10 5 = 0 :=
sorry

end log_expression_evaluation_l210_210716


namespace range_independent_variable_l210_210986

theorem range_independent_variable (x : ℝ) :
  (sqrt (x - 1) ≠ 0) ∧ (x - 1 ≥ 0) → x > 1 :=
by
  intro h
  have h1 : sqrt (x - 1) ≠ 0 := h.1
  have h2 : x - 1 ≥ 0 := h.2
  sorry

end range_independent_variable_l210_210986


namespace solve_sin_cos_eq_l210_210578

namespace MathProof

open Real Int

-- Define the integer part (floor) of real functions
noncomputable def int_part (a : ℝ) : ℤ := int.floor a

-- Problem statement
theorem solve_sin_cos_eq (x : ℝ) :
  int_part (sin (2 * x)) - 2 * int_part (cos x) = 3 * int_part (sin (3 * x)) ↔
  ∃ n : ℤ, x ∈ (2 * π * n, π / 6 + 2 * π * n) ∪ (π / 6 + 2 * π * n, π / 4 + 2 * π * n) ∪ (π / 4 + 2 * π * n, π / 3 + 2 * π * n) := by
  sorry

end MathProof

end solve_sin_cos_eq_l210_210578


namespace largest_multiple_of_12_less_than_350_l210_210263

theorem largest_multiple_of_12_less_than_350 : ∃ n, n < 350 ∧ n % 12 = 0 ∧ ∀ m, m < 350 ∧ m % 12 = 0 → m ≤ 348 :=
begin
  use 348,
  split,
  { norm_num },
  split,
  { norm_num },
  { intros m hm1 hm2,
    rw ←nat.le_div_iff_mul_le (nat.pos_of_ne_zero (nat.mod_ne_zero_of_pos hm1.zero_lt)) at hm1,
    norm_cast at hm1,
    norm_num at hm1,
    exact le_trans hm1 (by norm_num) }
end

end largest_multiple_of_12_less_than_350_l210_210263


namespace bubble_pass_1191_l210_210961

/-- 
  Given a sequence of 50 distinct elements, the probability that the element initially 
  at the 25th position ends up at the 35th position after one bubble pass is \( \frac{1}{1190} \). 
  Therefore, \( p + q = 1191 \) if \( \frac{p}{q} \) is in the lowest terms.
-/
theorem bubble_pass_1191 (n : ℕ) (r : Fin n → ℕ) (hn : n = 50) (h_distinct : Function.Injective r)
  (h_random_order : ∀ (i j : Fin n), i ≤ j -> (r i ≤ r j) → r i = r j) : 
  let p := 1 in let q := 1190 in p + q = 1191 :=
by
  sorry

end bubble_pass_1191_l210_210961


namespace basis_plane_vectors_parallelogram_vertex_D_equilateral_triangle_dot_product_projection_coordinates_l210_210274

-- Problem 1: Basis for Plane Vectors
theorem basis_plane_vectors (a b : ℝ × ℝ) (ha : a = (1, 2)) (hb : b = (3, 1)) :
  ∃ (u v : ℝ) (hu : u ≠ 0) (hv : v ≠ 0), u * a + v * b = (1, 0) ∧ u * a + v * b = (0, 1) := 
sorry

-- Problem 2: Coordinates of Vertex D
theorem parallelogram_vertex_D (A B C D : ℝ × ℝ) (hA : A = (5, -1)) (hB : B = (-1, 7)) (hC : C = (1, 2)) :
  (D = (7, -6)) :=
sorry

-- Problem 3: Equilateral Triangle Dot Product
theorem equilateral_triangle_dot_product (A B C : ℝ × ℝ) (hABC : (equilateral_triangle A B C )) :
  〈A - B, B - C〉 ≠ (π / 3) :=
sorry

-- Problem 4: Projection Coordinates
theorem projection_coordinates (a b : ℝ × ℝ) (ha : a = (1, 1)) (hb : abs b = 4) (hd : ∠(a, b) = π / 4) :
   ∃ (p : ℝ × ℝ), projection p a b = (2, 2) :=
sorry

end basis_plane_vectors_parallelogram_vertex_D_equilateral_triangle_dot_product_projection_coordinates_l210_210274


namespace count_correct_l210_210484

noncomputable def count_four_digit_numbers : ℕ :=
  let odd_primes := {x | x ∈ [3, 5, 7]};
  let multiples_of_three := {y | y ∈ [0, 3, 6, 9]};
  let valid_first_digits := odd_primes.to_finset;
  let valid_second_digits (first_digit : ℕ) :=
    (multiples_of_three.to_finset \ {first_digit});
  let remaining_choices (digit1 digit2 : ℕ) :=
    ((Finset.range 10) \ ({digit1, digit2}.to_finset)).card;
  (valid_first_digits.sum (λ d1, (valid_second_digits d1).sum (λ d2,
    remaining_choices d1 d2 * (remaining_choices d1 d2 - 1))))

theorem count_correct : count_four_digit_numbers = 616 :=
by
  sorry

end count_correct_l210_210484


namespace cost_of_milk_l210_210143

theorem cost_of_milk (num_cups_per_day : ℕ) (ounces_per_cup : ℕ) (coffee_cost : ℕ) 
    (coffee_ounces_per_bag : ℕ) (milk_used_per_week : ℚ) (total_spent_per_week : ℕ) 
    (h1 : num_cups_per_day = 2) (h2 : ounces_per_cup = 1.5) 
    (h3 : coffee_cost = 8) (h4 : coffee_ounces_per_bag = 10.5) 
    (h5 : milk_used_per_week = 0.5) (h6 : total_spent_per_week = 18) : 
    4 = total_spent_per_week - (num_cups_per_day * 7 * ounces_per_cup / coffee_ounces_per_bag * coffee_cost) / milk_used_per_week :=
by 
  sorry

end cost_of_milk_l210_210143


namespace solution_set_of_quadratic_inequality_l210_210230

theorem solution_set_of_quadratic_inequality :
  {x : ℝ | x^2 + 2 * x - 3 < 0} = {x : ℝ | -3 < x ∧ x < 1} :=
sorry

end solution_set_of_quadratic_inequality_l210_210230


namespace ratio_of_c_to_b_l210_210180

    theorem ratio_of_c_to_b (a b c : ℤ) (h0 : a = 0) (h1 : a < b) (h2 : b < c)
      (h3 : (a + b + c) / 3 = b / 2) : c / b = 1 / 2 :=
    by
      -- proof steps go here
      sorry
    
end ratio_of_c_to_b_l210_210180


namespace f_neg_a_l210_210659

-- Definition of the function f
def f (x : ℝ) : ℝ := x^3 * Real.cos x + 1

-- Given condition
variable (a : ℝ)
axiom f_a : f a = 11

-- The goal is to prove f(-a) = -9
theorem f_neg_a : f (-a) = -9 := 
sorry

end f_neg_a_l210_210659


namespace evaluate_expression_l210_210423

theorem evaluate_expression (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 12 * x + 2 := 
by 
  sorry

end evaluate_expression_l210_210423


namespace cos_double_angle_l210_210488

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (2 * θ) = -7 / 25 :=
by
  sorry

end cos_double_angle_l210_210488


namespace g_neither_even_nor_odd_l210_210116

def g (x : ℝ) : ℝ := ⌊2 * x⌋ + (1 / 3 : ℝ)

theorem g_neither_even_nor_odd :
  ¬ (∀ x : ℝ, g x = g (-x)) ∧ ¬ (∀ x : ℝ, g x = - g (-x)) :=
by
  sorry

end g_neither_even_nor_odd_l210_210116


namespace lucy_groceries_total_l210_210654

theorem lucy_groceries_total (cookies noodles : ℕ) (h1 : cookies = 12) (h2 : noodles = 16) : cookies + noodles = 28 :=
by
  sorry

end lucy_groceries_total_l210_210654


namespace sequence_is_arithmetic_sequence_l210_210914

theorem sequence_is_arithmetic_sequence (a : ℕ+ → ℤ) (p q : ℕ+) (h : a p = a q + 2003 * (p - q)) : 
  ∃ d : ℤ, ∀ p q : ℕ+, a p = a q + d * (p - q) :=
by
  use 2003
  intros p q
  exact h

end sequence_is_arithmetic_sequence_l210_210914


namespace water_height_in_rectangular_tank_l210_210706

/-- Proof problem setup: Given a pyramid with specific dimensions and the volume of water, calculate the height of the water in a rectangular tank. -/
theorem water_height_in_rectangular_tank :
  let base_side_pyramid := 16 -- side length of the square base of the pyramid in cm
  let height_pyramid := 24 -- height of the pyramid in cm
  let base_length_tank := 32 -- base length of the rectangular tank in cm
  let width_tank := 24 -- width of the rectangular tank in cm
  let volume_pyramid := (1/3 : ℝ) * (base_side_pyramid * base_side_pyramid) * height_pyramid in
  let base_area_tank := base_length_tank * width_tank in
  let height_water := volume_pyramid / base_area_tank in
  height_water = 2.67 := 
by
  sorry

end water_height_in_rectangular_tank_l210_210706
