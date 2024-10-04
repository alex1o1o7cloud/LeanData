import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.BigOperators.Order
import Mathlib.Algebra.Combinatorics.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Polynomial.BigOperators
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.Trigonometric.Basic
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Pigeonhole
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Median
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Probability
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Logic.Basic
import Mathlib.NumberTheory.ModularArithmetic
import Mathlib.Probability.Distribution.Normal
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.LinearCombination
import algebra.group_power
import data.real.basic

namespace cauchy_problem_solution_l462_462807

-- Define the differential equation
def differential_eq (y : ℝ → ℝ) (y' : ℝ → ℝ) (y'' : ℝ → ℝ) : Prop :=
  ∀ x, y'' x + y x = (1 / Real.cos x)

-- Define the initial conditions
def initial_conditions (y : ℝ → ℝ) (y' : ℝ → ℝ) : Prop :=
  y 0 = 1 ∧ y' 0 = 0

-- Define the proposed solution
def proposed_solution (y : ℝ → ℝ) : Prop :=
  ∀ x, y x = Real.cos x * (Real.ln (|Real.cos x|) + 1) + x * Real.sin x

-- The main statement to prove
theorem cauchy_problem_solution :
  ∃ y : ℝ → ℝ, 
    (∃ y' : ℝ → ℝ, ∃ y'' : ℝ → ℝ, differential_eq y y' y'' ∧ initial_conditions y y') 
    ∧ proposed_solution y :=
by
  sorry

end cauchy_problem_solution_l462_462807


namespace no_common_points_if_and_only_if_chord_length_is_two_if_and_only_if_l462_462540

theorem no_common_points_if_and_only_if (m : ℝ) :
  (∀ (x y : ℝ), (x^2 + y^2 = 5) → (2 * x - y + m ≠ 0)) ↔ (m > 5 ∨ m < -5) :=
by
  sorry

theorem chord_length_is_two_if_and_only_if (m : ℝ) :
  (∃ (p1 p2 : ℝ × ℝ),
    (p1.1^2 + p1.2^2 = 5) ∧ (p2.1^2 + p2.2^2 = 5) ∧
    (2 * p1.1 - p1.2 + m = 0) ∧ (2 * p2.1 - p2.2 + m = 0) ∧
    (∥(p2.1 - p1.1, p2.2 - p1.2)∥ = 2)) ↔
  (m = 2 * Real.sqrt 5 ∨ m = -2 * Real.sqrt 5) :=
by
  sorry

end no_common_points_if_and_only_if_chord_length_is_two_if_and_only_if_l462_462540


namespace line_intersects_x_axis_l462_462011

theorem line_intersects_x_axis :
  ∃ x, ∃ y, y = 0 ∧ line_through (8, 2) (4, 6) x y :=
begin
  use 10,
  use 0,
  split,
  { refl, },
  { sorry, }
end

def line_through (p1 p2 : ℝ × ℝ) (x y : ℝ) : Prop :=
  let m := (p2.2 - p1.2) / (p2.1 - p1.1) in
  y = m * (x - p1.1) + p1.2

end line_intersects_x_axis_l462_462011


namespace perimeter_bisectors_intersect_at_single_point_l462_462251

noncomputable theory
open_locale big_operators

variables {a b c : ℝ}

-- Given a triangle with sides a, b, and c,
-- Prove that lines passing through the vertices and bisecting the perimeter intersect at a single point
theorem perimeter_bisectors_intersect_at_single_point
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  ∃ (P : ℝ × ℝ), true := -- This states that such point P exists where those lines intersect
sorry

end perimeter_bisectors_intersect_at_single_point_l462_462251


namespace right_triangle_identity_l462_462593

-- Define the right-angled triangle and its properties
variables (a b c : ℝ) (h : a^2 + b^2 = c^2)

-- Define the radius of the inscribed circle
def inradius := (a * b) / (a + b + c)

theorem right_triangle_identity (h : a^2 + b^2 = c^2) :
  c + 2 * inradius a b c = a + b :=
by {
  have radius_def := inradius,
  sorry
}

end right_triangle_identity_l462_462593


namespace number_of_nephews_l462_462451

def total_jellybeans : ℕ := 70
def jellybeans_per_child : ℕ := 14
def number_of_nieces : ℕ := 2

theorem number_of_nephews : total_jellybeans / jellybeans_per_child - number_of_nieces = 3 := by
  sorry

end number_of_nephews_l462_462451


namespace polygon_interior_exterior_relation_l462_462289

theorem polygon_interior_exterior_relation :
  ∃ n : ℕ, (n ≥ 3) ∧ ((n - 2) * 180 = 2 * 360) → n = 6 :=
begin
  sorry
end

end polygon_interior_exterior_relation_l462_462289


namespace maximum_value_of_func_l462_462501

noncomputable def func (x y : ℝ) : ℝ := (x * y) / (x^2 + y^2)

def domain_x (x : ℝ) : Prop := (1/3 : ℝ) ≤ x ∧ x ≤ (2/5 : ℝ)
def domain_y (y : ℝ) : Prop := (1/2 : ℝ) ≤ y ∧ y ≤ (5/8 : ℝ)

theorem maximum_value_of_func :
  ∀ (x y : ℝ), domain_x x → domain_y y → func x y ≤ (20 / 21 : ℝ) ∧ 
  (∃ (x y : ℝ), domain_x x ∧ domain_y y ∧ func x y = (20 / 21 : ℝ)) :=
by sorry

end maximum_value_of_func_l462_462501


namespace even_factors_of_1386_l462_462280

theorem even_factors_of_1386 : 
  (∃ n : ℕ, prime_factorization 1386 = {2: 1, 3: 2, 7: 1, 11: 1}) →
  (∃ count : ℕ, count = 12 ∧ 
  (∀ (x y : ℕ), x * y = 1386 ∧ even x ∧ even y → count = 12)) :=
sorry -- Proof to be provided later

end even_factors_of_1386_l462_462280


namespace paint_per_color_equal_l462_462600

theorem paint_per_color_equal (total_paint : ℕ) (num_colors : ℕ) (paint_per_color : ℕ) : 
  total_paint = 15 ∧ num_colors = 3 → paint_per_color = 5 := by
  sorry

end paint_per_color_equal_l462_462600


namespace obtuse_angle_A_DB_of_bisectors_l462_462912

theorem obtuse_angle_A_DB_of_bisectors
  (ABC: Triangle)
  (A B C D : Point)
  (h_right_triangle : is_right_triangle ABC)
  (h_angle_A_45 : angle A = 45)
  (h_angle_B_45 : angle B = 45)
  (h_bisectors_intersect_D : intersects_angle_bisectors A B D) :
  angle ADB = 135 :=
by sorry

end obtuse_angle_A_DB_of_bisectors_l462_462912


namespace natural_number_operations_l462_462704

noncomputable theory

variable {a b : ℕ}

theorem natural_number_operations :
  (a + b ∈ ℕ) ∧ (a * b ∈ ℕ) ∧ (a ^ b ∈ ℕ) :=
by
  split; sorry

end natural_number_operations_l462_462704


namespace find_N_l462_462092

theorem find_N (a b c d : ℝ) :
  ∃ N : Matrix (Fin 2) (Fin 2) ℝ, N ⬝ (Matrix.of [[a, b], [c, d]]) = (Matrix.of [[2 * a, 3 * b], [2 * c, 3 * d]]) ∧
    N = Matrix.of [[2, 0], [0, 3]] :=
  sorry

end find_N_l462_462092


namespace round_to_nearest_integer_l462_462969

theorem round_to_nearest_integer (x : ℝ) (hx : x = 8542137.8790345) : Int.round x = 8542138 :=
by
  -- Since the exact proof is not required, we'll leave it as an axiom placeholder
  sorry

end round_to_nearest_integer_l462_462969


namespace measure_angledb_l462_462917

-- Definitions and conditions:
def right_triangle (A B C : Type) : Prop :=
  ∃ (a b c : A), ∠A = 45 ∧ ∠B = 45 ∧ ∠C = 90

def angle_bisector (A B : Type) (D : Type) : Prop :=
  ∃ (AD BD : A), AD bisects ∠A ∧ BD bisects ∠B

-- Prove:
theorem measure_angledb (A B C D : Type) 
  (hABC : right_triangle A B C)
  (hAngleBisectors : angle_bisector A B D) : 
  ∠ ADB = 135 := 
sorry

end measure_angledb_l462_462917


namespace probability_of_at_least_one_two_l462_462006

theorem probability_of_at_least_one_two :
  (∀ (X1 X2 X3 : ℕ), (2 ≤ X1 ∧ X1 ≤ 6) ∧ (2 ≤ X2 ∧ X2 ≤ 6) ∧ (1 ≤ X3 ∧ X3 ≤ 5) ∧ (X1 + X2 = X3 + 1)) →
  let outcomes := {(X1, X2, X3) | (2 ≤ X1 ∧ X1 ≤ 6) ∧ (2 ≤ X2 ∧ X2 ≤ 6) ∧ (1 ≤ X3 ∧ X3 ≤ 5) ∧ (X1 + X2 = X3 + 1)} in
  ∃ (favorableOutcomes : ℕ), 
    (favorableOutcomes = finset.card {o ∈ outcomes | o.1 = 2 ∨ o.2 = 2}) ∧
    (favorableOutcomes = 7) ∧
    (finset.card outcomes = 15) ∧
    (Real.toRational favorableOutcomes / Real.toRational (finset.card outcomes) = 7 / 15) :=
begin
  sorry
end

end probability_of_at_least_one_two_l462_462006


namespace Pyarelal_loss_l462_462446

variables (capital_of_pyarelal capital_of_ashok : ℝ) (total_loss : ℝ)

def is_ninth (a b : ℝ) : Prop := a = b / 9

def applied_loss (loss : ℝ) (ratio : ℝ) : ℝ := ratio * loss

theorem Pyarelal_loss (h1: is_ninth capital_of_ashok capital_of_pyarelal) 
                        (h2: total_loss = 1600) : 
                        applied_loss total_loss (9/10) = 1440 :=
by 
  unfold is_ninth at h1
  sorry

end Pyarelal_loss_l462_462446


namespace points_enclosed_in_circle_l462_462403

theorem points_enclosed_in_circle (P : set (ℝ × ℝ)) (hP : ∀ A B C ∈ P, ∃ (c : ℝ × ℝ) (r : ℝ), r = 1 ∧ dist c A ≤ r ∧ dist c B ≤ r ∧ dist c C ≤ r) :
  ∃ (c : ℝ × ℝ) (r : ℝ), r ≤ 1 ∧ ∀ p ∈ P, dist c p ≤ r := 
  sorry

end points_enclosed_in_circle_l462_462403


namespace smallest_positive_period_range_of_f_triangle_area_l462_462947
noncomputable theory

-- Define the function f
def f (x : ℝ) : ℝ := cos x ^ 2 - sqrt 3 * sin x * cos x + 1/2

-- Problem 1: Prove the smallest positive period of f(x) is π
theorem smallest_positive_period : ∀ x, f(x) = f(x + π) := by sorry

-- Problem 2: Prove the range of f(x) is [0,2]
theorem range_of_f : ∀ y, 0 ≤ y ∧ y ≤ 2 ↔ ∃ x, f(x) = y := by sorry

-- Problem 3: Given conditions and prove the area of triangle ABC is sqrt(3)/2
variables {A B C a b c : ℝ}
theorem triangle_area :
  ∀ (A B C a b c : ℝ), 
  A + B + C = π ∧ 
  f(B + C) = 3/2 ∧ 
  a = sqrt 3 ∧ 
  b + c = 3
  → (1/2 * b * c * sin A = sqrt 3 / 2) := 
by
  intros A B C a b c h,
  sorry

end smallest_positive_period_range_of_f_triangle_area_l462_462947


namespace scientific_notation_of_sesame_mass_l462_462971

theorem scientific_notation_of_sesame_mass :
  0.00000201 = 2.01 * 10^(-6) :=
sorry

end scientific_notation_of_sesame_mass_l462_462971


namespace company_fund_initial_amount_l462_462995

theorem company_fund_initial_amount
  (n : ℕ) -- number of employees
  (initial_bonus_per_employee : ℕ := 60)
  (shortfall : ℕ := 10)
  (revised_bonus_per_employee : ℕ := 50)
  (fund_remaining : ℕ := 150)
  (initial_fund : ℕ := initial_bonus_per_employee * n - shortfall) -- condition that the fund was $10 short when planning the initial bonus
  (revised_fund : ℕ := revised_bonus_per_employee * n + fund_remaining) -- condition after distributing the $50 bonuses

  (eqn : initial_fund = revised_fund) -- equating initial and revised budget calculations
  
  : initial_fund = 950 := 
sorry

end company_fund_initial_amount_l462_462995


namespace equations_of_line_l462_462090

variables (x y : ℝ)

-- Given conditions
def passes_through_point (P : ℝ × ℝ) (x y : ℝ) := (x, y) = P

def has_equal_intercepts_on_axes (f : ℝ → ℝ) :=
  ∃ z : ℝ, z ≠ 0 ∧ f z = 0 ∧ f 0 = z

-- The proof problem statement
theorem equations_of_line (P : ℝ × ℝ) (hP : passes_through_point P 2 (-3)) (h : has_equal_intercepts_on_axes (λ x => -x / (x / 2))) :
  (x + y + 1 = 0) ∨ (3 * x + 2 * y = 0) := 
sorry

end equations_of_line_l462_462090


namespace situps_ratio_l462_462216

theorem situps_ratio (ken_situps : ℕ) (nathan_situps : ℕ) (bob_situps : ℕ) :
  ken_situps = 20 →
  nathan_situps = 2 * ken_situps →
  bob_situps = ken_situps + 10 →
  (bob_situps : ℚ) / (ken_situps + nathan_situps : ℚ) = 1 / 2 :=
by
  sorry

end situps_ratio_l462_462216


namespace number_of_mappings_l462_462821

theorem number_of_mappings (P Q : Set) (f : P → Q) (a : P) (hP : P = {a, b}) (hQ : Q = {-1, 0, 1})
  (hf : f(a) = 0) : 
  (∀ f, f(a) = 0 → (∃! b_image, b_image ∈ Q ∧ f = λ x : P, if x = a then 0 else b_image)) →
  (3 = ∑ f : P → Q, if f(a) = 0 then 1 else 0) :=
by
  -- Here we skip the proof.
  sorry

end number_of_mappings_l462_462821


namespace obtuse_angle_A_DB_of_bisectors_l462_462911

theorem obtuse_angle_A_DB_of_bisectors
  (ABC: Triangle)
  (A B C D : Point)
  (h_right_triangle : is_right_triangle ABC)
  (h_angle_A_45 : angle A = 45)
  (h_angle_B_45 : angle B = 45)
  (h_bisectors_intersect_D : intersects_angle_bisectors A B D) :
  angle ADB = 135 :=
by sorry

end obtuse_angle_A_DB_of_bisectors_l462_462911


namespace lois_initial_books_l462_462950

-- Define the initial number of books
def initial_books (B : ℕ) : Prop :=
  -- Condition 1: Giving away a fourth of her books
  let after_nephew := (3 * B) / 4 in
  -- Condition 2: Donating a third of the remaining books
  let after_library := after_nephew - after_nephew / 3 in
  -- Condition 3: Purchasing 3 new books
  let final_books := after_library + 3 in
  -- Final condition: Lois now has 23 books
  final_books = 23

-- The theorem to prove
theorem lois_initial_books : ∃ B : ℕ, initial_books B ∧ B = 40 :=
by
  existsi 40
  unfold initial_books
  sorry

end lois_initial_books_l462_462950


namespace monthly_increase_per_ticket_l462_462631

variable (x : ℝ)

theorem monthly_increase_per_ticket
    (initial_premium : ℝ := 50)
    (percent_increase_per_accident : ℝ := 0.10)
    (tickets : ℕ := 3)
    (final_premium : ℝ := 70) :
    initial_premium * (1 + percent_increase_per_accident) + tickets * x = final_premium → x = 5 :=
by
  intro h
  sorry

end monthly_increase_per_ticket_l462_462631


namespace more_knights_than_liars_l462_462243

theorem more_knights_than_liars (K L : ℕ) (h_odd : odd (K + L))
  (h_knights : ∀ k, k < K → ∃! l, l < L)
  (h_liars : ∀ l, l < L → ¬ ∃ k, k < K) : K > L := 
sorry

end more_knights_than_liars_l462_462243


namespace roots_of_equation_in_interval_l462_462164

theorem roots_of_equation_in_interval (f : ℝ → ℝ) (interval : Set ℝ) (n_roots : ℕ) :
  (∀ x ∈ interval, f x = 8 * x * (1 - 2 * x^2) * (8 * x^4 - 8 * x^2 + 1) - 1) →
  (interval = Set.Icc 0 1) →
  (n_roots = 4) :=
by
  intros f_eq interval_eq
  sorry

end roots_of_equation_in_interval_l462_462164


namespace expression_incorrect_l462_462270

theorem expression_incorrect (x : ℝ) : 5 * (x + 7) ≠ 5 * x + 7 := 
by 
  sorry

end expression_incorrect_l462_462270


namespace smallest_integer_with_eight_divisors_l462_462376

-- Definitions and conditions:
def prime_factors (n : ℕ) : list (ℕ × ℕ) := sorry
def number_of_divisors (n : ℕ) := (prime_factors n).map (λ (p: ℕ × ℕ), p.2 + 1).prod

-- Proof statement:
theorem smallest_integer_with_eight_divisors 
  (n = 2^7 ∨ n = 2^3 * 3 ∨ n = 2 * 3 * 5) :
  (number_of_divisors n = 8 → n = 24) :=
sorry

end smallest_integer_with_eight_divisors_l462_462376


namespace multiplication_example_l462_462048

theorem multiplication_example : 28 * (9 + 2 - 5) * 3 = 504 := by 
  sorry

end multiplication_example_l462_462048


namespace reflection_coefficient_l462_462726

theorem reflection_coefficient (I_0 : ℝ) (I_4 : ℝ) (k : ℝ) 
  (h1 : I_4 = I_0 * (1 - k)^4) 
  (h2 : I_4 = I_0 / 256) : 
  k = 0.75 :=
by 
  -- Proof omitted
  sorry

end reflection_coefficient_l462_462726


namespace casey_saving_l462_462053

-- Define the conditions
def cost_per_hour_first_employee : ℝ := 20
def cost_per_hour_second_employee : ℝ := 22
def subsidy_per_hour : ℝ := 6
def hours_per_week : ℝ := 40

-- Define the weekly cost calculations
def weekly_cost_first_employee := cost_per_hour_first_employee * hours_per_week
def effective_cost_per_hour_second_employee := cost_per_hour_second_employee - subsidy_per_hour
def weekly_cost_second_employee := effective_cost_per_hour_second_employee * hours_per_week

-- State the theorem
theorem casey_saving :
    weekly_cost_first_employee - weekly_cost_second_employee = 160 := 
by
  sorry

end casey_saving_l462_462053


namespace domain_sqrt_div_l462_462665

theorem domain_sqrt_div (x : ℝ) : 
  (∃ y : ℝ, y = sqrt (x + 3) / x) ↔ (x ≥ -3 ∧ x ≠ 0) :=
by
  split
  {
    rintro ⟨y, h_eq⟩
    split
    {
      by_cases (x < -3)
      { -- if x < -3, contradiction arises because sqrt (x + 3) is not well-defined
        linarith
      },
      { -- if x ≥ -3 condition is satisfied
        by_contradiction Hx0
        exact Hx0 (by linarith [sqrt (x + 3)])
      ]
    },
    {
      intro hx
      use sqrt (x + 3) / x
      field_simp
      exact hx.2
    }
  }
  sorry

end domain_sqrt_div_l462_462665


namespace number_of_solutions_l462_462240

theorem number_of_solutions (n : ℕ) : 
  (∃ a d : ℕ, ∀ k : ℕ, (k > 0) → (|x| + |y| = k → number_of_solutions = a + (k-1) * d)) →
   (∀ k : ℕ, (|x| + |y| = k → number_of_solutions = 4 * k)) →
    (number_of_solutions 20 = 80) :=
begin
   sorry
end

end number_of_solutions_l462_462240


namespace fractional_eq_k_l462_462814

open Real

theorem fractional_eq_k (x k : ℝ) (hx0 : x ≠ 0) (hx1 : x ≠ 1) :
  (3 / x + 6 / (x - 1) - (x + k) / (x * (x - 1)) = 0) ↔ k ≠ -3 ∧ k ≠ 5 := 
sorry

end fractional_eq_k_l462_462814


namespace no_real_f_zero_l462_462234

theorem no_real_f_zero (f : ℝ → ℝ) 
  (h : ∀ x y, f(x + y) = f(x) * f(y) + 1) : 
  ¬ ∃ c : ℝ, f(0) = c :=
by
  sorry

end no_real_f_zero_l462_462234


namespace donna_weekly_earnings_l462_462477

def hourly_rate_dog_walking := 10.0
def hours_per_day_dog_walking := 2
def days_per_week_dog_walking := 7
def hourly_rate_card_shop := 12.5
def hours_per_day_card_shop := 2
def days_per_week_card_shop := 5
def hourly_rate_babysitting := 10.0
def hours_per_weekend_babysitting := 4

theorem donna_weekly_earnings : 
  (hourly_rate_dog_walking * hours_per_day_dog_walking * days_per_week_dog_walking) + 
  (hourly_rate_card_shop * hours_per_day_card_shop * days_per_week_card_shop) + 
  (hourly_rate_babysitting * hours_per_weekend_babysitting) = 305.0 := 
by 
  sorry

end donna_weekly_earnings_l462_462477


namespace proposition_p_is_false_l462_462123

theorem proposition_p_is_false : ¬ (∃ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
by {
  intro h,
  cases h with x hx,
  have h1: x^3 - x^2 + 1 > 0 
  by {
    -- Here needs to assert that x^3 - x^2 + 1 > 0 for any real number x
    sorry
  },
  exact lt_irrefl _ (hx.trans_lt h1),
}

end proposition_p_is_false_l462_462123


namespace min_positive_period_l462_462672

theorem min_positive_period (x : ℝ) : 
  ∃ T > 0, ∀ x, 3 * Real.cos (2 * x + (Real.pi / 6)) = 3 * Real.cos (2 * (x + T) + (Real.pi / 6)) ∧ T = Real.pi :=
begin
  use Real.pi,
  split,
  { exact Real.pi_pos, },
  sorry
end

end min_positive_period_l462_462672


namespace F_is_fixed_point_l462_462713

-- Definitions and conditions
def Circle (O : Type*) := {P : Type*}
def Line (l : Type*) := {P : Type*}
def Point (E : Type*) := {l : Type*}

variables {P : Type*} [MetricSpace P] [InnerProductSpace ℝ P]
variables {O l : Set P} {E M A B C D F : P}

-- Circle O is given
axiom O_is_circle : Circle O

-- Line l is given and outside the circle O
axiom l_is_line : Line l
axiom l_outside_circle : ∀ x ∈ l, x ∉ O

-- OE is perpendicular to l at E
axiom OE_perpendicular_l : E ∈ l ∧ (∀ x : P, x ∈ l → innerProduct (x - E) (E - O) = 0)

-- M is any point on l except E
axiom M_on_l : M ∈ l ∧ M ≠ E

-- Tangents from M to circle O touch at A and B
axiom tangents_from_M : isTangent M A O ∧ isTangent M B O

-- EC ⊥ MA at C
axiom EC_perpendicular_MA : isPerpendicular E C M A

-- ED ⊥ MB at D
axiom ED_perpendicular_MB : isPerpendicular E D M B

-- CD extended intersects OE at F
axiom CD_intersects_OE_at_F : ∃ F : P, LineExtended C D ∩ LineExtended O E = {F}

-- Goal: Prove F is a fixed point
theorem F_is_fixed_point : isFixedPoint F :=
sorry

end F_is_fixed_point_l462_462713


namespace average_minutes_per_day_l462_462751

theorem average_minutes_per_day (f : ℕ) 
    (third_grade_avg : ℕ := 10) (fourth_grade_avg : ℕ := 18) (fifth_grade_avg : ℕ := 12)
    (third_grade_students : ℕ := 9 * f) (fourth_grade_students : ℕ := 3 * f) (fifth_grade_students : ℕ := f) :
  let total_minutes_run := third_grade_avg * third_grade_students + fourth_grade_avg * fourth_grade_students + fifth_grade_avg * fifth_grade_students
  let total_students := third_grade_students + fourth_grade_students + fifth_grade_students
  (total_minutes_run / total_students = 12) :=
begin
  sorry,
end

end average_minutes_per_day_l462_462751


namespace center_of_mass_on_diagonal_l462_462961

variables {A B C D K L : Type} [AddGroup K] [Module ℝ K]

-- Definitions of points A, B, C, D, K, and L in a parallelogram
variable (parallelogram : A × B × C × D)
variable (BC_CD_ratio : ℝ)
variable (K_on_BC : B → C → K)
variable (L_on_CD : C → D → L)

-- Prove that the center of mass of the triangle AKL lies on the diagonal BD
theorem center_of_mass_on_diagonal 
  (h1 : parallelogram (A, B, C, D))
  (hK : K_on_BC B C = K)
  (hL : L_on_CD C D = L)
  (h_ratio : (K - B) / (C - B) = BC_CD_ratio ∧ (L - C) / (D - C) = BC_CD_ratio) :
  ∃ M : D, M ∈ line(B, D) ∧ center_mass(A, K, L) = M := sorry

end center_of_mass_on_diagonal_l462_462961


namespace f_pi_over_4_eq_zero_l462_462668

noncomputable def f (x : ℝ) : ℝ := -Math.sin x + Math.cos x

theorem f_pi_over_4_eq_zero : f (Real.pi / 4) = 0 := by
  sorry

end f_pi_over_4_eq_zero_l462_462668


namespace conic_eccentricity_l462_462168

theorem conic_eccentricity (m : ℝ) (h : m = Real.sqrt (2 * 8) ∨ m = -Real.sqrt (2 * 8)) :
  (∃ e : ℝ, (e = Real.sqrt 3 / 2 ∨ e = Real.sqrt 5) ∧ 
  (∀ x y : ℝ, x^2 + y^2 / m = 1 → ((x ≠ 0 ∨ y ≠ 0) → (Real.eccentricity_of_conic x y m e)))) :=
sorry

end conic_eccentricity_l462_462168


namespace positive_difference_l462_462246

/-- Pauline deposits 10,000 dollars into an account with 4% compound interest annually. -/
def Pauline_initial_deposit : ℝ := 10000
def Pauline_interest_rate : ℝ := 0.04
def Pauline_years : ℕ := 12

/-- Quinn deposits 10,000 dollars into an account with 6% simple interest annually. -/
def Quinn_initial_deposit : ℝ := 10000
def Quinn_interest_rate : ℝ := 0.06
def Quinn_years : ℕ := 12

/-- Pauline's balance after 12 years -/
def Pauline_balance : ℝ := Pauline_initial_deposit * (1 + Pauline_interest_rate) ^ Pauline_years

/-- Quinn's balance after 12 years -/
def Quinn_balance : ℝ := Quinn_initial_deposit * (1 + Quinn_interest_rate * Quinn_years)

/-- The positive difference between Pauline's and Quinn's balances after 12 years is $1189 -/
theorem positive_difference :
  |Quinn_balance - Pauline_balance| = 1189 := 
sorry

end positive_difference_l462_462246


namespace student_arrangements_l462_462033

theorem student_arrangements (students : Finset ℕ) (hcard : students.card = 7) :
  ∃ (A B : Finset ℕ), 
    A.card + B.card = 6 ∧ 
    2 ≤ A.card ∧ 2 ≤ B.card ∧ 
    (A ∪ B = students ∧ A ∩ B = ∅) →
    A.card = 2 ∨ A.card = 3 ∨ A.card = 4 ∨ B.card = 2 ∨ B.card = 3 ∨ B.card = 4 →
  (choose 6 2 * choose 4 4 + choose 6 3 * choose 3 3 + choose 6 4 * choose 2 2) * 7 = 350 :=
by
  sorry

end student_arrangements_l462_462033


namespace integers_equal_zero_l462_462966

theorem integers_equal_zero 
  (a b c : ℤ) 
  (h : (Real.cbrt 4) * a + (Real.cbrt 2) * b + c = 0) : 
  a = 0 ∧ b = 0 ∧ c = 0 :=
sorry

end integers_equal_zero_l462_462966


namespace system_unique_solution_l462_462236

theorem system_unique_solution
  (x y z : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (eq1 : x + y^2 + z^3 = 3)
  (eq2 : y + z^2 + x^3 = 3)
  (eq3 : z + x^2 + y^3 = 3) :
  x = 1 ∧ y = 1 ∧ z = 1 :=
begin
  sorry
end

end system_unique_solution_l462_462236


namespace length_of_AB_l462_462586

open Real

-- Definitions based on the conditions:
def parametric_line_x (t : ℝ) : ℝ := 1 - (sqrt 2) / 2 * t
def parametric_line_y (t : ℝ) : ℝ := 2 + (sqrt 2) / 2 * t

def parabola_y_squared (x : ℝ) : ℝ := 4 * x

-- Theorem statement: Given the parametric equations of the line intersecting the parabola,
-- prove that the length of the line segment AB is 8 * sqrt 2.
theorem length_of_AB : ∃ (A B : ℝ × ℝ), 
  (A.1 = 1 ∧ A.2 = 2) ∧
  (B.1 = 9 ∧ B.2 = -6) ∧
  (y_sq_A : parabola_y_squared A.1 = A.2^2) ∧ 
  (y_sq_B : parabola_y_squared B.1 = B.2^2) ∧ 
  ((sqrt (A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 * sqrt 2) :=
begin
  sorry
end

end length_of_AB_l462_462586


namespace team_air_conditioner_installation_l462_462200

theorem team_air_conditioner_installation (x : ℕ) (y : ℕ) 
  (h1 : 66 % x = 0) 
  (h2 : 60 % y = 0) 
  (h3 : x = y + 2) 
  (h4 : 66 / x = 60 / y) 
  : x = 22 ∧ y = 20 :=
by
  have h5 : x = 22 := sorry
  have h6 : y = 20 := sorry
  exact ⟨h5, h6⟩

end team_air_conditioner_installation_l462_462200


namespace find_value_l462_462134

def f (x : ℝ) : ℝ := if 0 ≤ x ∧ x < 1 then 2^x - 1 else sorry

theorem find_value :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x, f (x + 2) = f x) →  -- f has a period of 2
  f (Real.log 24 / Real.log (1 / 2)) = -1 / 2 := sorry

end find_value_l462_462134


namespace max_cuboid_surface_area_l462_462516

theorem max_cuboid_surface_area (R : ℝ) (a b c : ℝ) :
  4 * real.pi * R^2 = 25 * real.pi ∧ a^2 + b^2 + c^2 = (2 * R)^2 →
  2 * (a * b + a * c + b * c) ≤ 50 :=
by
  sorry

end max_cuboid_surface_area_l462_462516


namespace evaluate_expression_l462_462065

-- Definition of the function f
def f (x : ℤ) : ℤ := 3 * x^2 - 5 * x + 8

-- Theorems and lemmas
theorem evaluate_expression : 3 * f 4 + 2 * f (-4) = 260 := by
  sorry

end evaluate_expression_l462_462065


namespace norm_sum_eq_l462_462492

-- Define the complex numbers
def c1 : ℂ := 3 - 5 * complex.I
def c2 : ℂ := 3 + 5 * complex.I

-- Define the norm (magnitude) of complex numbers
def norm_c1 : ℝ := complex.abs c1
def norm_c2 : ℝ := complex.abs c2

-- The statement to prove
theorem norm_sum_eq : norm_c1 + norm_c2 = 2 * real.sqrt 34 :=
by sorry

end norm_sum_eq_l462_462492


namespace binomial_coefficient_ratio_l462_462986

theorem binomial_coefficient_ratio (n k : ℕ) (h₁ : n = 4 * k + 3) (h₂ : n = 3 * k + 5) : n + k = 13 :=
by
  sorry

end binomial_coefficient_ratio_l462_462986


namespace pizza_volume_one_piece_l462_462423

theorem pizza_volume_one_piece
  (thickness : ℝ)
  (diameter : ℝ)
  (pieces : ℝ)
  (h : thickness = 1/2)
  (d : diameter = 16)
  (p : pieces = 8) :
  ∃ (volume_one_piece : ℝ), volume_one_piece = 4 * Real.pi :=
by 
  rcases (pi * (d / 2) ^ 2 * h) / p with v;
  use v;
  sorry

end pizza_volume_one_piece_l462_462423


namespace cubed_identity_l462_462173

variable (x : ℝ)

theorem cubed_identity (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := 
by
  sorry

end cubed_identity_l462_462173


namespace smallest_even_integer_cube_mod_1000_l462_462502

theorem smallest_even_integer_cube_mod_1000 :
  ∃ n : ℕ, (n % 2 = 0) ∧ (n > 0) ∧ (n^3 % 1000 = 392) ∧ (∀ m : ℕ, (m % 2 = 0) ∧ (m > 0) ∧ (m^3 % 1000 = 392) → n ≤ m) ∧ n = 892 := 
sorry

end smallest_even_integer_cube_mod_1000_l462_462502


namespace greatest_divisor_sum_of_first_fifteen_terms_l462_462319

theorem greatest_divisor_sum_of_first_fifteen_terms 
  (x c : ℕ) (hx : x > 0) (hc : c > 0):
  ∃ d, d = 15 ∧ d ∣ (15*x + 105*c) :=
by
  existsi 15
  split
  . refl
  . apply Nat.dvd.intro
    existsi (x + 7*c)
    refl
  sorry

end greatest_divisor_sum_of_first_fifteen_terms_l462_462319


namespace number_of_C_atoms_in_compound_is_4_l462_462005

def atomic_weight_C : ℕ := 12
def atomic_weight_H : ℕ := 1
def atomic_weight_O : ℕ := 16

def molecular_weight : ℕ := 65

def weight_contributed_by_H_O : ℕ := atomic_weight_H + atomic_weight_O -- 17 amu

def weight_contributed_by_C : ℕ := molecular_weight - weight_contributed_by_H_O -- 48 amu

def number_of_C_atoms := weight_contributed_by_C / atomic_weight_C -- The quotient of 48 amu divided by 12 amu per C atom

theorem number_of_C_atoms_in_compound_is_4 : number_of_C_atoms = 4 :=
by
  sorry -- This is where the proof would go, but it's omitted as per instructions.

end number_of_C_atoms_in_compound_is_4_l462_462005


namespace molecular_weight_correct_l462_462694

noncomputable def molecular_weight_compound : ℝ :=
  (3 * 12.01) + (6 * 1.008) + (1 * 16.00)

theorem molecular_weight_correct :
  molecular_weight_compound = 58.078 := by
  sorry

end molecular_weight_correct_l462_462694


namespace length_segment_AB_l462_462857

-- Define the parametric equations
def parametric_x (α : ℝ) : ℝ := 2 * sqrt 3 * cos α
def parametric_y (α : ℝ) : ℝ := 2 * sin α

-- Define the points A and B in polar coordinates
def A_rho (ρ₁ : ℝ) (θ_a : ℝ := π/6) : Prop :=
  ρ₁^2 = 8

def B_rho (ρ₂ : ℝ) (θ_b : ℝ := 2 * π / 3) : Prop :=
  ρ₂^2 = 24 / 5

-- Cartesian form of the curve
def cartesian_form (x y : ℝ) : Prop :=
  x^2 / 12 + y^2 / 4 = 1

-- Polar form of the curve
def polar_form (ρ θ : ℝ) : Prop :=
  1 / ρ^2 = (cos θ)^2 / 12 + (sin θ)^2 / 4

-- Length of the line segment AB
def length_AB (ρ₁ ρ₂ : ℝ) (θ_a θ_b : ℝ) : ℝ :=
  sqrt (ρ₁^2 + ρ₂^2 - 2 * ρ₁ * ρ₂ * cos (θ_b - θ_a))

-- Theorem
theorem length_segment_AB
  (ρ₁ ρ₂ : ℝ)
  (θ_a θ_b : ℝ)
  (hA : A_rho ρ₁ θ_a)
  (hB : B_rho ρ₂ θ_b) :
  length_AB ρ₁ ρ₂ θ_a θ_b = 8 * sqrt 5 / 5 :=
sorry

end length_segment_AB_l462_462857


namespace remainder_eq_one_l462_462800

-- Define the expression
def expr := ∑ k in Finset.range 11, ((-1) ^ k) * (90 ^ k) * Nat.choose 10 k

-- Statement to prove
theorem remainder_eq_one : expr % 88 = 1 := 
sorry

end remainder_eq_one_l462_462800


namespace car_speed_l462_462715

-- Definitions based on the conditions
def distance : ℕ := 375
def time : ℕ := 5

-- Mathematically equivalent proof statement
theorem car_speed : distance / time = 75 := 
  by
  -- The actual proof will be placed here, but we'll skip it for now.
  sorry

end car_speed_l462_462715


namespace min_score_partition_S_l462_462605

-- Define the set S
def S : Set ℕ := { x | 1 ≤ x ∧ x ≤ 100 }

-- Define what a partition of S is
def is_partition (n : ℕ) (S: Set ℕ) (part : Fin n → Set ℕ) : Prop :=
  (∀ i, part i ≠ ∅) ∧
  (∀ i j, i ≠ j → disjoint (part i) (part j)) ∧
  (∀ x ∈ S, ∃ i, x ∈ part i) ∧
  (∀ i, ∀ x ∈ part i, x ∈ S)

-- Define the average of a set of elements
def avg (s: Set ℕ) : ℝ :=
  if s = ∅ then 0 else (s.to_finite.to_finset.sum id : ℝ) / (s.to_finite.to_finset.card : ℝ)

-- Define the score of a partition
def score {n : ℕ} (part : Fin n → Set ℕ) : ℝ :=
  (finset.sum finset.univ (λ i, avg (part i)) : ℝ) / n

-- The original problem: Prove the minimum score of any partition of S is 10
theorem min_score_partition_S : ∀ (n : ℕ) (part : Fin n → Set ℕ), is_partition n S part → score part ≥ 10 :=
sorry

end min_score_partition_S_l462_462605


namespace angle_between_vectors_l462_462880

-- Vector definitions and conditions
variables (a b : ℝ → ℝ → ℝ)
variable (x : ℝ)
-- We assume the existence of a norm function 
-- and a dot product for the sake of this proof
noncomputable def norm (v : ℝ → ℝ → ℝ) : ℝ := sorry
noncomputable def dot (v w : ℝ → ℝ → ℝ) : ℝ := sorry

-- Length conditions
axiom norm_a : norm a = sqrt 2
axiom norm_b : norm b = 1

-- Inequality condition
axiom inequality_condition : ∀ x : ℝ, norm (λ (t : ℝ), a t + x * b t) ≥ norm (λ (t : ℝ), a t + b t)

-- Target theorem
theorem angle_between_vectors : 
∃ θ : ℝ, θ = (3 * real.pi) / 4 ∧ 
    ∃ cos_θ : ℝ, cos_θ = -real.sqrt 2 / 2 ∧ 
        dot a b = norm a * norm b * cos_θ :=
sorry

end angle_between_vectors_l462_462880


namespace sin_cos_cubed_inequality_l462_462507

-- We state the problem directly in Lean

theorem sin_cos_cubed_inequality (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  (sin α) ^ 3 * (cos β) ^ 3 + (sin α) ^ 3 * (sin β) ^ 3 + (cos α) ^ 3 ≥ (real.sqrt 3) / 3 := 
sorry

end sin_cos_cubed_inequality_l462_462507


namespace inequality_proof_l462_462648

theorem inequality_proof (a b : ℝ) : a^2 + b^2 + 2 * (a - 1) * (b - 1) ≥ 1 :=
by
  sorry

end inequality_proof_l462_462648


namespace conic_sections_l462_462273

theorem conic_sections (x y : ℝ) :
  y^4 - 8 * x^4 = 4 * y^2 - 4 →
  (∃ a b : ℝ, y^2 - a * x^2 = 2 ∧ b = 2√2 ∧ 
    ((a = b ∧ y^2 + a * x^2 = 2) ∧ y^2 - a * x^2 = 2)) :=
sorry

end conic_sections_l462_462273


namespace find_d_to_nearest_tenth_l462_462017

-- Define the problem parameters based on the given conditions
def largeSquareVertices : set (ℝ × ℝ) := {(0, 0), (4040, 0), (4040, 4040), (0, 4040)}
def probability := 3 / 5

-- State the theorem
theorem find_d_to_nearest_tenth :
  ∀ d : ℝ, (π * d^2 = probability) → (float.round (d * 10) / 10 = 0.4) :=
by
  intro d h
  sorry

end find_d_to_nearest_tenth_l462_462017


namespace function_is_increasing_on_interval_l462_462146

noncomputable def f (m x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * m * x^2 + 4 * x - 3

theorem function_is_increasing_on_interval {m : ℝ} :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → (1/3) * x^3 - (1/2) * m * x^2 + 4 * x - 3 ≥ (1/3) * (x - dx)^3 - (1/2) * m * (x - dx)^2 + 4 * (x - dx) - 3)
  ↔ m ≤ 4 :=
sorry

end function_is_increasing_on_interval_l462_462146


namespace prove_f_monotone_odd_arith_sequence_l462_462852

variables {α : Type*} [linear_ordered_field α]

-- Definitions as conditions
def monotone_function (f : α → α) := ∀ x y, x < y → f(x) < f(y)
def odd_function (f : α → α) := ∀ x, f(-x) = -f(x)
def arithmetic_sequence (a : ℕ → α) (a₃ : α) := a₃ > 0 ∧ ∀ n, a n = a 0 + n * ((a 3 - a 0) / 3)

-- The math problem statement
theorem prove_f_monotone_odd_arith_sequence
  (f : α → α)
  (a : ℕ → α)
  (a₃ a₁ a₅ : α)
  (h₁ : monotone_function f)
  (h₂ : odd_function f)
  (h₃ : arithmetic_sequence a a₃)
  (h₄ : a 3 = a₃)
  (h₅ : a 1 = a₁)
  (h₆ : a 5 = a₅) :
  f(a₁) + f(a₃) + f(a₅) > 0 := 
sorry

end prove_f_monotone_odd_arith_sequence_l462_462852


namespace tangent_secant_parallel_l462_462110

/-- 
Given:
1. Point P is outside a circle.
2. Tangents PA and PB are drawn from P, touching the circle at points A and B.
3. A secant line through P intersects the circle at points C and D.
4. A line through point B is drawn parallel to PA, intersecting lines AC and AD at points E and F.
Prove that BE = BF.
-/
theorem tangent_secant_parallel (P A B C D E F : Point)
  (h_circle : circle)
  (hP_outside : P ∉ h_circle)
  (hPA_tangent : tangent_to_circle P A h_circle)
  (hPB_tangent : tangent_to_circle P B h_circle)
  (hP_secant : secant_line P C D h_circle)
  (hBE_parallel : parallel (line_through B E) (line_through P A))
  (hE_on_AC : point_on_line E (line_through A C))
  (hF_on_AD : point_on_line F (line_through A D))
  (hF_B_on_line : point_on_line B (line_through F E)) :
  distance B E = distance B F := 
sorry

end tangent_secant_parallel_l462_462110


namespace diagonal_AC_possibilities_l462_462105

/-
In a quadrilateral with sides AB, BC, CD, and DA, the length of diagonal AC must 
satisfy the inequalities determined by the triangle inequalities for triangles 
ABC and CDA. Prove the number of different whole numbers that could be the 
length of diagonal AC is 13.
-/

def number_of_whole_numbers_AC (AB BC CD DA : ℕ) : ℕ :=
  if 6 < AB ∧ AB < 20 then 19 - 7 + 1 else sorry

theorem diagonal_AC_possibilities : number_of_whole_numbers_AC 7 13 15 10 = 13 :=
  by
    sorry

end diagonal_AC_possibilities_l462_462105


namespace asymptotes_of_hyperbola_l462_462139

theorem asymptotes_of_hyperbola (a : ℝ) :
  (∃ a : ℝ, 9 + a = 13) →
  (∀ x y : ℝ, (x^2 / 9 - y^2 / a = 1) → (a = 4)) →
  (forall (x y : ℝ), (x^2 / 9 - y^2 / 4 = 0) → 
    (y = (2/3) * x) ∨ (y = -(2/3) * x)) :=
by
  sorry

end asymptotes_of_hyperbola_l462_462139


namespace distance_between_parallel_lines_l462_462984

theorem distance_between_parallel_lines (k : ℝ) (h_k : k = -8) :
  let line1 := λ x y : ℝ, k * x + 6 * y + 2 = 0,
      line2 := λ x y : ℝ, 4 * x - 3 * y + 4 = 0 in
  ∃ d : ℝ, d = 1 :=
by
  sorry

end distance_between_parallel_lines_l462_462984


namespace constant_term_binomial_expansion_l462_462089

theorem constant_term_binomial_expansion :
  -- Given the expansion of the binomial (x + 2/x)^6
  (∑ r in Finset.range (6+1), (Nat.choose 6 r) * (2 / x)^(6-r) * x^r).eval 0 = 160 :=
by
  sorry

end constant_term_binomial_expansion_l462_462089


namespace sum_exterior_angles_pentagon_l462_462677

/-- The sum of the exterior angles of a pentagon is equal to 360 degrees. -/
theorem sum_exterior_angles_pentagon (h : ∀ (n : ℕ), n ≥ 3 → sum_exterior_angles n = 360) : 
  sum_exterior_angles 5 = 360 := 
by 
  sorry

end sum_exterior_angles_pentagon_l462_462677


namespace price_increase_ratio_l462_462565

theorem price_increase_ratio 
  (c : ℝ)
  (h1 : 351 = c * 1.30) :
  (c + 351) / c = 2.3 :=
sorry

end price_increase_ratio_l462_462565


namespace combinations_of_4_blocks_no_same_row_col_in_6x6_is_5400_l462_462732

noncomputable def num_combinations_4_blocks_no_same_row_col :=
  (Nat.choose 6 4) * (Nat.choose 6 4) * (Nat.factorial 4)

theorem combinations_of_4_blocks_no_same_row_col_in_6x6_is_5400 :
  num_combinations_4_blocks_no_same_row_col = 5400 := 
by
  sorry

end combinations_of_4_blocks_no_same_row_col_in_6x6_is_5400_l462_462732


namespace pizza_volume_one_piece_l462_462422

theorem pizza_volume_one_piece :
  ∀ (h t: ℝ) (d: ℝ) (n: ℕ), d = 16 → t = 1/2 → n = 8 → h = 8 → 
  ( (π * (d / 2)^2 * t) / n = 4 * π ) :=
by 
  intros h t d n hd ht hn hh
  sorry

end pizza_volume_one_piece_l462_462422


namespace greatest_common_divisor_sum_arithmetic_sequence_l462_462346

theorem greatest_common_divisor_sum_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, d = 15 ∧ ∀ (n : ℕ), n = 15 → ∀ k : ℕ, k = 15 ∧ 15 ∣ (15 * (x + 7 * c)) :=
by
  sorry

end greatest_common_divisor_sum_arithmetic_sequence_l462_462346


namespace eccentricity_equals_2_l462_462768

variables (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) (A : ℝ × ℝ) (F : ℝ × ℝ) (B : ℝ × ℝ)
variables (eqn_hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
variables (focus_F : F = (c, 0)) (imaginary_axis_B : B = (0, b))
variables (intersect_A : A = (c / 3, 2 * b / 3))
variables (vector_eqn : 3 * (A.1, A.2) = (F.1 + 2 * B.1, F.2 + 2 * B.2))
variables (asymptote_eqn : ∀ A1 A2 : ℝ, A2 = (b / a) * A1 → A = (A1, A2))

theorem eccentricity_equals_2 : (c / a = 2) :=
sorry

end eccentricity_equals_2_l462_462768


namespace system_of_equations_solution_l462_462263

theorem system_of_equations_solution :
  (∃ x₁ x₂ x₃ ... x₉₉ x₁₀₀ : ℝ,
  x₁ + x₁ * x₂ = 1 ∧
  x₂ + x₂ * x₃ = 1 ∧
  ⋮
  x₉₉ + x₉₉ * x₁₀₀ = 1 ∧
  x₁₀₀ + x₁₀₀ * x₁ = 1) ↔
  ∃ x : ℝ, (x = (1 - x) / x) ∧ (x = (2: ℝ)/(1 + sqrt 5)) ∧ (x = (2: ℝ)/(1 - sqrt 5)) :=
by
  sorry

end system_of_equations_solution_l462_462263


namespace value_always_positive_l462_462849

variable {α : Type*}
variable [LinearOrder α]
variable (f : α → ℝ)
variable (a_n : ℕ → α)

-- Definitions for conditions
def monotonically_increasing (f : α → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

def odd_function (f : α → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_arithmetic_sequence (a_n : ℕ → α) : Prop :=
  ∃ d, ∀ n, a_n (n + 1) = a_n n + d

-- Given conditions
axiom f_monotonic_increasing : monotonically_increasing f
axiom f_odd : odd_function f
axiom a_n_arithmetic : is_arithmetic_sequence a_n
axiom a_3_pos : a_n 3 > 0

-- Proof problem statement
theorem value_always_positive : f (a_n 1) + f (a_n 3) + f (a_n 5) > 0 :=
sorry

end value_always_positive_l462_462849


namespace equal_distances_necessary_not_sufficient_l462_462397

variable {Point Line Plane : Type}
variable distance : Point → Plane → ℝ
variable onLine : Point → Line → Prop
variable parallel : Line → Plane → Prop

theorem equal_distances_necessary_not_sufficient
  (l : Line) (α : Plane) (p1 p2 : Point):
  onLine p1 l → onLine p2 l → distance p1 α = distance p2 α →
  ¬ (parallel l α) ∧ (∀ (l : Line) (α : Plane), parallel l α → ∀ p1 p2 : Point, onLine p1 l → onLine p2 l → distance p1 α = distance p2 α) :=
by 
  sorry

end equal_distances_necessary_not_sufficient_l462_462397


namespace Milburg_adults_l462_462295

theorem Milburg_adults:
  ∀ (P C A: ℝ), P = 5256.0 → C = 2987.0 → A = P - C → A = 2269.0 :=
by
  intro P C A
  intro hP hC hA
  rw [hP, hC] at hA
  linarith

end Milburg_adults_l462_462295


namespace position_of_point_1_5_1_l462_462779

theorem position_of_point_1_5_1 : 
  let S := [(1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1), (1, 1, 3), (1, 3, 1), (3, 1, 1), 
           (1, 2, 2), (2, 1, 2), (2, 2, 1), (1, 1, 4), (1, 4, 1), (4, 1, 1), (1, 2, 3),
           (1, 3, 2), (1, 2, 4), (1, 5, 1)]
  in (S.index_of (1, 5, 1) + 1) = 22 := 
by
  sorry

end position_of_point_1_5_1_l462_462779


namespace find_g_at_4_l462_462272

theorem find_g_at_4 (g : ℝ → ℝ) (h : ∀ x, 2 * g x + 3 * g (1 - x) = 4 * x^3 - x) : g 4 = 193.2 :=
sorry

end find_g_at_4_l462_462272


namespace isosceles_triangle_perimeter_l462_462577

noncomputable def perimeter_of_isosceles_triangle : ℝ :=
  let BC := 10
  let height := 6
  let half_base := BC / 2
  let side := Real.sqrt (height^2 + half_base^2)
  let perimeter := 2 * side + BC
  perimeter

theorem isosceles_triangle_perimeter :
  let BC := 10
  let height := 6
  perimeter_of_isosceles_triangle = 2 * Real.sqrt (height^2 + (BC / 2)^2) + BC := by
  sorry

end isosceles_triangle_perimeter_l462_462577


namespace schedule_volunteers_l462_462682

theorem schedule_volunteers :
  let volunteers : List (Char) := ['A', 'B', 'C'],
      days : List (String) := ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
      schedules := {s : List (Char × String) | ∀ v ∈ volunteers, ∃ d ∈ days, (v, d) ∈ s} in
  ∀ s ∈ schedules, (A < B) ∧ (A < C) → (number_of_schedules s == 20) :=
by 
  sorry

end schedule_volunteers_l462_462682


namespace sin_cos_105_eq_sqrt6_div_2_l462_462763

noncomputable def sin_cos_identity : Real :=
  sin (Real.pi * 105 / 180) - cos (Real.pi * 105 / 180)

theorem sin_cos_105_eq_sqrt6_div_2 :
  sin_cos_identity = (Real.sqrt 6) / 2 := by
sorry

end sin_cos_105_eq_sqrt6_div_2_l462_462763


namespace triangle_inequality_inequality_l462_462640

theorem triangle_inequality_inequality {a b c p q r : ℝ}
  (h1 : a + b > c)
  (h2 : b + c > a)
  (h3 : c + a > b)
  (h4 : p + q + r = 0) :
  a^2 * p * q + b^2 * q * r + c^2 * r * p ≤ 0 :=
sorry

end triangle_inequality_inequality_l462_462640


namespace permutation_square_sum_distinct_values_l462_462615

theorem permutation_square_sum_distinct_values 
  {a b c d : ℝ} (h : a < b) (h' : b < c) (h'' : c < d) 
  (x y z t : ℝ) (h₁ : x = a ∨ x = b ∨ x = c ∨ x = d) 
  (h₂ : y = a ∨ y = b ∨ y = c ∨ y = d) 
  (h₃ : z = a ∨ z = b ∨ z = c ∨ z = d) 
  (h₄ : t = a ∨ t = b ∨ t = c ∨ t = d) 
  (h_perm : list.nodup [x, y, z, t] ∧ list.perm [x, y, z, t] [a, b, c, d]) :
  (∃ n₁ n₂ n₃ : ℝ, 
  n₁ ≠ n₂ ∧ n₂ ≠ n₃ ∧ n₁ ≠ n₃ ∧ 
  ∀ (x y z t : ℝ), 
  (x = a ∨ x = b ∨ x = c ∨ x = d) ∧ 
  (y = a ∨ y = b ∨ y = c ∨ y = d) ∧ 
  (z = a ∨ z = b ∨ z = c ∨ z = d) ∧ 
  (t = a ∨ t = b ∨ t = c ∨ t = d) ∧ 
  list.nodup [x, y, z, t] ∧ 
  list.perm [x, y, z, t] [a, b, c, d] → 
  (x - y) ^ 2 + (y - z) ^ 2 + (z - t) ^ 2 + (t - x) ^ 2 ∈ {n₁, n₂, n₃}) := 
sorry

end permutation_square_sum_distinct_values_l462_462615


namespace number_of_3_before_4_in_arrangements_l462_462268

theorem number_of_3_before_4_in_arrangements : 
  let digits := [3, 4, 5, 6, 7]
  let permutations := digits.permutations
  let count_3_before_4 := permutations.count (λ p, p.index_of 3 < p.index_of 4)
  count_3_before_4 = 60 :=
by
  sorry

end number_of_3_before_4_in_arrangements_l462_462268


namespace measure_of_obtuse_angle_ADB_l462_462908

-- Define the right triangle ABC with specific angles
def triangle_ABC (A B C D : Type) [AddAngle A B C] :=
  (rightTriangle : (A B C) ∧ 
   angle_A_is_45_degrees : (angle A = 45) ∧ 
   angle_B_is_45_degrees : (angle B = 45) ∧ 
   AD_bisects_A : (bisects A D) ∧ 
   BD_bisects_B : (bisects B D))

-- Statement of the proof problem
theorem measure_of_obtuse_angle_ADB {A B C D : Type} [AddAngle A B C] 
  (h_ABC : triangle_ABC A B C) : 
  measure_obtuse_angle A B D ADB = 135 :=
sorry

end measure_of_obtuse_angle_ADB_l462_462908


namespace probability_same_color_correct_l462_462045

-- Defining the contents of Bag A and Bag B
def bagA : List (String × ℕ) := [("white", 1), ("red", 2), ("black", 3)]
def bagB : List (String × ℕ) := [("white", 2), ("red", 3), ("black", 1)]

-- The probability calculation
noncomputable def probability_same_color (bagA bagB : List (String × ℕ)) : ℚ :=
  let p_white := (1 / 6 : ℚ) * (1 / 3 : ℚ)
  let p_red := (1 / 3 : ℚ) * (1 / 2 : ℚ)
  let p_black := (1 / 2 : ℚ) * (1 / 6 : ℚ)
  p_white + p_red + p_black

-- Proof problem statement
theorem probability_same_color_correct :
  probability_same_color bagA bagB = 11 / 36 := 
by 
  sorry

end probability_same_color_correct_l462_462045


namespace sum_first_13_terms_l462_462838

def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in range n, a (i + 1)

def sum_3 := S_3 (a : ℕ → ℤ) (d : ℤ) : Prop :=
  sum_of_first_n a 3 = 6

def sum_terms_9_11_13 (a : ℕ → ℤ) (d : ℤ) : Prop :=
  a 9 + a 11 + a 13 = 60

theorem sum_first_13_terms (a : ℕ → ℤ) (d : ℤ) :
  sum_3 a d →
  sum_terms_9_11_13 a d →
  sum_of_first_n a 13 = 156 :=
sorry

end sum_first_13_terms_l462_462838


namespace unique_not_in_range_of_g_l462_462767

noncomputable def g (p q r s : ℝ) (x : ℝ) : ℝ := (p * x + q) / (r * x + s)

theorem unique_not_in_range_of_g
  (p q r s : ℝ) (h₀ : p ≠ 0) (h₁ : q ≠ 0) (h₂ : r ≠ 0) (h₃ : s ≠ 0)
  (h₄ : g p q r s 17 = 17) (h₅ : g p q r s 89 = 89)
  (h₆ : ∀ x, x ≠ -s / r → g p q r s (g p q r s x) = x) :
  53 ∉ set.range (g p q r s) :=
begin
  sorry
end

end unique_not_in_range_of_g_l462_462767


namespace parallelogram_sides_l462_462095

section ParallelogramSides

variables {A B C D : Type} 

-- Given: the circumradius of triangle ABC
def circumradius_ABC : ℝ := 5

-- Given: the circumradius of triangle ABD
def circumradius_ABD : ℝ := Real.sqrt 13

-- Given: the distance between the centers of the circumcircles of triangles ABC and ABD
def distance_centers : ℝ := 2

-- Prove: The sides of the parallelogram ABCD
theorem parallelogram_sides 
  (AB AD : ℝ) 
  (h : AB = 6 ∧ AD = 9 * Real.sqrt(2/5)) 
  : True := 
by 
  sorry

end ParallelogramSides

end parallelogram_sides_l462_462095


namespace isosceles_triangle_legs_not_possible_isosceles_possible_isosceles_l462_462016

-- Part ①: Proving the length of each leg
theorem isosceles_triangle_legs (x : ℝ) (h1 : 5 * x = 24) : 2 * x = 48 / 5 :=
by 
  have h2 : x = 24 / 5 := by linarith
  rw [h2]
  linarith

-- Part ②: Checking possibility of forming an isosceles triangle
-- Case 1: Not possible if each leg is 6 cm.
theorem not_possible_isosceles (P : ℝ) (leg : ℝ) (base : ℝ) (h1 : P = 24) (h2 : leg = 6) (h3 : base = P - 2 * leg) :
  ¬(leg + leg > base ∧ leg + base > base ∧ base + base > leg) :=
by 
  have h4 : base = 12 := by linarith
  have h5 : ¬(6 + 6 > 12) := by linarith
  simp [h4, h5]

-- Case 2: Possible if base is 6 cm, resulting in each leg being 9 cm.
theorem possible_isosceles (P : ℝ) (leg : ℝ) (base : ℝ) (h1 : P = 24) (h2 : base = 6) (h3 : leg = (P - base) / 2) : 
  leg = 9 ∧ (leg + leg > base ∧ leg + base > base ∧ base + base > leg) :=
by 
  have h4 : leg = 9 := by linarith
  simp [h4]
  linarith


end isosceles_triangle_legs_not_possible_isosceles_possible_isosceles_l462_462016


namespace vision_standard_rates_l462_462440

-- Definitions for initial conditions
def science_rate_A := 60 / 100.0
def liberal_arts_rate_A := 70 / 100.0
def science_rate_B := 65 / 100.0
def liberal_arts_rate_B := 75 / 100.0

-- Proof Problem Lean 4 Statement
theorem vision_standard_rates :
  (science_rate_B > science_rate_A ∧ liberal_arts_rate_B > liberal_arts_rate_A)
  ∧ (liberal_arts_rate_A > science_rate_A ∧ liberal_arts_rate_B > science_rate_B)
  ∧ ∀ (a b : ℕ), (0.6 * a = 0.7 * b → (0.6 * a + 0.7 * b) / (a + b) ≠ 0.65)
  ∧ (∃ (a1 a2 b1 b2 : ℕ), ((0.6 * a1 + 0.7 * a2) / (a1 + a2) > (0.65 * b1 + 0.75 * b2) / (b1 + b2))) :=
by sorry

end vision_standard_rates_l462_462440


namespace min_m_for_partition_l462_462612

open Set

theorem min_m_for_partition (m : ℕ) (S : Set ℕ) (h1 : m ≥ 5) (h2 : S = \{n | 5 ≤ n ∧ n ≤ m\}) :
  (∀ A B: Set ℕ, A ∪ B = S ∧ A ∩ B = ∅ → (∃ a b c ∈ S, a * b = c ∧ (a ∈ A ∧ b ∈ A ∧ c ∈ A) ∨ (a ∈ B ∧ b ∈ B ∧ c ∈ B))) ↔ m = 3125 :=
by
  sorry

end min_m_for_partition_l462_462612


namespace selection_methods_l462_462901

-- Define the conditions
def total_athletes : Nat := 10
def right_only : Nat := 3
def left_only : Nat := 2
def both_sides : Nat := 5
def selected_rowers : Nat := 6

-- The statement to prove
theorem selection_methods :
  ∑ i in range (left_only + 1), 
  (Nat.choose both_sides (3 - i) * Nat.choose both_sides (3 - i) * 
  Nat.choose (right_only + both_sides + left_only - 2 * i) (6 - 2 * (3 - i))) = 675 := 
sorry

end selection_methods_l462_462901


namespace interest_from_20000_l462_462389

variables (P_1 P_2 I_1 I_2 r t: ℝ)

-- Define conditions
def condition_1 : Prop := P_1 = 5000
def condition_2 : Prop := I_1 = 250
def condition_3 : Prop := P_2 = 20000
def condition_4 : Prop := t = 1
def simple_interest (P r t: ℝ) := P * r * t

-- Proof goal: Interest from $20,000 investment
theorem interest_from_20000
  (h1 : condition_1)
  (h2 : condition_2)
  (h3 : condition_3)
  (h4 : condition_4)
  (I1_def : I_1 = simple_interest P_1 r t) :
  I_2 = 1000 :=
sorry

end interest_from_20000_l462_462389


namespace age_difference_l462_462654

-- Let D denote the daughter's age and M denote the mother's age
variable (D M : ℕ)

-- Conditions given in the problem
axiom h1 : M = 11 * D
axiom h2 : M + 13 = 2 * (D + 13)

-- The main proof statement to show the difference in their current ages
theorem age_difference : M - D = 40 :=
by
  sorry

end age_difference_l462_462654


namespace range_of_a_l462_462889

-- Define the piecewise function f
def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x ≤ 1 then (3 - a) * x - a else Real.log x / Real.log a

-- Define the increasing property for a function on ℝ
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- The theorem to be proved
theorem range_of_a (a : ℝ) (h : is_increasing (f a)) : a ∈ Ioo (3 / 2) 3 :=
  sorry

end range_of_a_l462_462889


namespace ellipse_standard_equation_l462_462532

theorem ellipse_standard_equation (focal_length : ℝ) (eccentricity : ℝ) (a b c : ℝ)
  (h1 : 2 * c = focal_length)
  (h2 : c / a = eccentricity)
  (h3 : focal_length = 8)
  (h4 : eccentricity = 0.8)
  : (a = 5 ∧ b = 3 ∧ c = 4) ∧
    ((∀ x y : ℝ, (x^2)/(5^2) + (y^2)/(3^2) = 1) ∨ (∀ x y : ℝ, (y^2)/(5^2) + (x^2)/(3^2) = 1)) :=
begin
  sorry
end

end ellipse_standard_equation_l462_462532


namespace tangency_problem_l462_462937

open EuclideanGeometry

variables (A B C H M N K L F J : Point)
variables [IsTriangle A B C]
variables [IsOrthocenter H A B C]
variables [IsMidpoint M A B]
variables [IsMidpoint N A C]
variables [InRegion H B M N C]
variables [AreTangent (Circumcircle B M H) (Circumcircle C N H)]
variables [IsParallel (Line.through H ⟨BC⟩) (Line.through H K)]
variables [IsParallel (Line.through H ⟨BC⟩) (Line.through H L)]
variables [Intersection F (Line.through M K) (Line.through N L)]
variables [IsIncenter J (Triangle.mk M H N)]

theorem tangency_problem : dist F J = dist F A :=
sorry

end tangency_problem_l462_462937


namespace justin_tim_same_game_l462_462447

noncomputable def total_players : ℕ := 12
noncomputable def game_size : ℕ := 6
noncomputable def justin : ℕ := 0
noncomputable def tim : ℕ := 1

theorem justin_tim_same_game : ∀ (total_games : ℕ), 
  total_games = Nat.choose total_players game_size → 
  range (Mathlib.combinatorics.pigeonhole.total_game_combinations total_players) / 2 →
  Mathlib.combinatorics.pigeonhole.calculate_combinations (total_players - 2) (game_size - 2) = 
  (210 : ℕ) :=
begin
  sorry
end

end justin_tim_same_game_l462_462447


namespace greatest_divisor_arithmetic_sequence_sum_l462_462331

theorem greatest_divisor_arithmetic_sequence_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ k, (15 * (x + 7 * c)) = 15 * k :=
sorry

end greatest_divisor_arithmetic_sequence_sum_l462_462331


namespace sum_c_n_l462_462834

noncomputable theory
open_locale big_operators

def a : ℕ+ → ℝ
| 2 := 5
| (nat.succ_pos' n) := a n ^ 2 - 2 * (nat.pred n) * a n + 2

def b (n : ℕ) : ℝ := 2 ^ (n - 1)

def c (n : ℕ) : ℝ := a n + b n

def T (n : ℕ) : ℝ := ∑ i in finset.range (n+1), c (i+1)

theorem sum_c_n (n : ℕ) : T n = 2^n + n^2 + 2*n - 1 :=
sorry

end sum_c_n_l462_462834


namespace graphs_intersect_at_one_point_l462_462872

theorem graphs_intersect_at_one_point (m : ℝ) (e := Real.exp 1) :
  (∀ f g : ℝ → ℝ,
    (∀ x, f x = x + Real.log x - 2 / e) ∧ (∀ x, g x = m / x) →
    ∃! x, f x = g x) ↔ (m ≥ 0 ∨ m = - (e + 1) / (e ^ 2)) :=
by sorry

end graphs_intersect_at_one_point_l462_462872


namespace problem_statement_l462_462194

-- Define the problem context
variables {a b c d : ℝ}

-- Define the conditions
def unit_square_condition (a b c d : ℝ) : Prop :=
  a^2 + b^2 + c^2 + d^2 ≥ 2 ∧ a^2 + b^2 + c^2 + d^2 ≤ 4 ∧ 
  a + b + c + d ≥ 2 * Real.sqrt 2 ∧ a + b + c + d ≤ 4

-- Provide the main theorem
theorem problem_statement (h : unit_square_condition a b c d) : 
  2 ≤ a^2 + b^2 + c^2 + d^2 ∧ a^2 + b^2 + c^2 + d^2 ≤ 4 ∧ 
  2 * Real.sqrt 2 ≤ a + b + c + d ∧ a + b + c + d ≤ 4 :=
  by 
  { sorry }  -- Proof to be completed

end problem_statement_l462_462194


namespace proposition_2_proposition_4_l462_462113

-- Definitions from conditions.
def circle_M (x y q : ℝ) : Prop := (x + Real.cos q)^2 + (y - Real.sin q)^2 = 1
def line_l (y k x : ℝ) : Prop := y = k * x

-- Prove that the line l and circle M always intersect for any real k and q.
theorem proposition_2 : ∀ (k q : ℝ), ∃ (x y : ℝ), circle_M x y q ∧ line_l y k x := sorry

-- Prove that for any real k, there exists a real q such that the line l is tangent to the circle M.
theorem proposition_4 : ∀ (k : ℝ), ∃ (q x y : ℝ), circle_M x y q ∧ line_l y k x ∧
  (abs (Real.sin q + k * Real.cos q) = 1 / Real.sqrt (1 + k^2)) := sorry

end proposition_2_proposition_4_l462_462113


namespace pizza_volume_one_piece_l462_462425

theorem pizza_volume_one_piece
  (thickness : ℝ)
  (diameter : ℝ)
  (pieces : ℝ)
  (h : thickness = 1/2)
  (d : diameter = 16)
  (p : pieces = 8) :
  ∃ (volume_one_piece : ℝ), volume_one_piece = 4 * Real.pi :=
by 
  rcases (pi * (d / 2) ^ 2 * h) / p with v;
  use v;
  sorry

end pizza_volume_one_piece_l462_462425


namespace total_distance_flown_l462_462740

/-- 
An eagle can fly 15 miles per hour; 
a falcon can fly 46 miles per hour; 
a pelican can fly 33 miles per hour; 
and a hummingbird can fly 30 miles per hour. 
All the birds flew for 2 hours straight.
Prove that the total distance flown by all the birds is 248 miles.
-/
theorem total_distance_flown :
  let eagle_speed := 15
      falcon_speed := 46
      pelican_speed := 33
      hummingbird_speed := 30
      hours_flown := 2
      eagle_distance := eagle_speed * hours_flown 
      falcon_distance := falcon_speed * hours_flown 
      pelican_distance := pelican_speed * hours_flown 
      hummingbird_distance := hummingbird_speed * hours_flown 
  in eagle_distance + falcon_distance + pelican_distance + hummingbird_distance = 248 := 
sorry

end total_distance_flown_l462_462740


namespace remainder_3_pow_20_div_5_l462_462283

theorem remainder_3_pow_20_div_5 : (3 ^ 20) % 5 = 1 := 
by {
  sorry
}

end remainder_3_pow_20_div_5_l462_462283


namespace stadium_fee_difference_l462_462686

theorem stadium_fee_difference :
  let capacity := 2000
  let entry_fee := 20
  let full_fees := capacity * entry_fee
  let three_quarters_fees := (capacity * 3 / 4) * entry_fee
  full_fees - three_quarters_fees = 10000 :=
by
  sorry

end stadium_fee_difference_l462_462686


namespace no_polyhedron_with_7_edges_l462_462642

theorem no_polyhedron_with_7_edges :
  ∀ (V F E : ℕ), V - E + F = 2 ∧ 2 * E = 3 * F → E ≠ 7 :=
by
  intros V F E h
  cases h with h1 h2
  sorry

end no_polyhedron_with_7_edges_l462_462642


namespace missing_angle_correct_l462_462949

theorem missing_angle_correct (n : ℕ) (h1 : n ≥ 3) (angles_sum : ℕ) (h2 : angles_sum = 2017) 
    (sum_interior_angles : ℕ) (h3 : sum_interior_angles = 180 * (n - 2)) :
    (sum_interior_angles - angles_sum) = 143 :=
by
  sorry

end missing_angle_correct_l462_462949


namespace symmetry_about_half_pi_l462_462147

def f (x : ℝ) : ℝ := Real.sin x + 1 / Real.sin x

noncomputable def domain_of_f (x : ℝ) : Prop := ∀ k : ℤ, x ≠ k * Real.pi

theorem symmetry_about_half_pi (x : ℝ) (h : domain_of_f x) (h1 : domain_of_f (Real.pi / 2 - x)) :
  f (Real.pi / 2 + x) = f (Real.pi / 2 - x) :=
by
  sorry

end symmetry_about_half_pi_l462_462147


namespace construction_cost_is_correct_l462_462728

def land_cost (cost_per_sqm : ℕ) (area : ℕ) : ℕ :=
  cost_per_sqm * area

def bricks_cost (cost_per_1000 : ℕ) (quantity : ℕ) : ℕ :=
  (cost_per_1000 * quantity) / 1000

def roof_tiles_cost (cost_per_tile : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_tile * quantity

def cement_bags_cost (cost_per_bag : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_bag * quantity

def wooden_beams_cost (cost_per_meter : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_meter * quantity

def steel_bars_cost (cost_per_meter : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_meter * quantity

def electrical_wiring_cost (cost_per_meter : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_meter * quantity

def plumbing_pipes_cost (cost_per_meter : ℕ) (quantity : ℕ) : ℕ :=
  cost_per_meter * quantity

def total_cost : ℕ :=
  land_cost 60 2500 +
  bricks_cost 120 15000 +
  roof_tiles_cost 12 800 +
  cement_bags_cost 8 250 +
  wooden_beams_cost 25 1000 +
  steel_bars_cost 15 500 +
  electrical_wiring_cost 2 2000 +
  plumbing_pipes_cost 4 3000

theorem construction_cost_is_correct : total_cost = 212900 :=
  by
    sorry

end construction_cost_is_correct_l462_462728


namespace binomial_theorem_coeff_a_l462_462184

theorem binomial_theorem_coeff_a (a : ℝ) :
  let expansion_term (r : ℕ) := (Nat.choose 7 r) * (2^(r:ℕ)) * (a^(7-r)) * (x^(-7 + 2*r))
  in (∃ r : ℕ, -7 + 2 * r = -3) → 
     ((Nat.choose 7 2) * (2^2) * (a^5) = 84) → 
     a = 1 :=
by
  sorry

end binomial_theorem_coeff_a_l462_462184


namespace greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l462_462362

theorem greatest_divisor_of_sum_first_15_terms_arithmetic_sequence
  (x c : ℕ) -- where x and c are positive integers
  (h_pos_x : 0 < x) -- x is positive
  (h_pos_c : 0 < c) -- c is positive
  : ∃ (d : ℕ), d = 15 ∧ ∀ (S : ℕ), S = 15 * x + 105 * c → d ∣ S :=
by
  sorry

end greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l462_462362


namespace minimum_value_l462_462806

noncomputable def smallest_value_expression (x y : ℝ) := x^4 + y^4 - x^2 * y - x * y^2

theorem minimum_value (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + y ≤ 1) :
  (smallest_value_expression x y) ≥ -1 / 8 :=
sorry

end minimum_value_l462_462806


namespace general_formula_for_a_n_sum_of_first_n_terms_l462_462465

-- Definitions from conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m, a (n + 1) - a n = a (m + 1) - a m

def sum_of_first_five_terms (a : ℕ → ℝ) : Prop :=
∑ i in finset.range 5, a (i + 1) = 15

def geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ r > 1, 2 * a 2 = r * a 6 ∧ a 6 = r * (a 8 + 1)

-- Proof problem
theorem general_formula_for_a_n (a : ℕ → ℝ) :
  arithmetic_sequence a ∧
  sum_of_first_five_terms a ∧
  geometric_sequence a
  → ∀ n, a n = n :=
begin
  sorry
end

theorem sum_of_first_n_terms (a b : ℕ → ℝ) (n : ℕ) :
  (∀ n, a n = n) → (∀ n, b n = 2^n * a n) 
  → ∑ i in finset.range n, b (i + 1) = (n - 1) * 2^(n + 1) + 2 :=
begin
  sorry
end

end general_formula_for_a_n_sum_of_first_n_terms_l462_462465


namespace evaluate_g_at_neg2_l462_462616

def g (x : ℝ) : ℝ := 3 * x^4 - 20 * x^3 + 35 * x^2 - 28 * x - 84

theorem evaluate_g_at_neg2 : g (-2) = 320 := by
  sorry

end evaluate_g_at_neg2_l462_462616


namespace angle_BGE_half_angle_AED_l462_462957

-- Define the problem setup
def square {α : Type} [linear_ordered_field α] (A B C D E F G : α → α) : Prop := sorry
def isosceles_obtuse_triangle {α : Type} [linear_ordered_field α] (A E D : α → α) : Prop := sorry
def circle_circumscribed {α : Type} [linear_ordered_field α] (A E D F : α → α) : Prop := sorry
def point_on_side {α : Type} [linear_ordered_field α] (G C D : α → α) : Prop := sorry
def segment_equality {α : Type} [linear_ordered_field α] (CG DF : α → α) : Prop := sorry

-- Define the main statement
theorem angle_BGE_half_angle_AED
  {α : Type} [linear_ordered_field α]
  (A B C D E F G : α → α)
  (h1 : square A B C D)
  (h2 : isosceles_obtuse_triangle A E D)
  (h3 : circle_circumscribed A E D F)
  (h4 : point_on_side G C D)
  (h5 : segment_equality (line C G) (line D F)) :
  angle (line B G) (line E G) < (1 / 2) * angle (line A E) (line E D) := sorry

end angle_BGE_half_angle_AED_l462_462957


namespace result_of_4_times_3_l462_462556

def operation (a b : ℕ) : ℕ :=
  a^2 + a * Nat.factorial b - b^2

theorem result_of_4_times_3 : operation 4 3 = 31 := by
  sorry

end result_of_4_times_3_l462_462556


namespace maintenance_interval_increase_total_l462_462746

def base_interval : ℝ := 25
def additive_A_increase : ℝ := 0.10
def additive_B_increase : ℝ := 0.15
def additive_C_increase : ℝ := 0.05
def harsh_conditions_decrease : ℝ := 0.05
def manufacturer_recommendation_increase : ℝ := 0.03

theorem maintenance_interval_increase_total :
  let final_interval :=
    (base_interval * (1 + additive_A_increase))
    * (1 + additive_B_increase)
    * (1 + additive_C_increase)
    * (1 - harsh_conditions_decrease)
    * (1 + manufacturer_recommendation_increase) in
  ((final_interval - base_interval) / base_interval) * 100 ≈ 29.97 :=
by
  sorry

end maintenance_interval_increase_total_l462_462746


namespace birds_total_distance_l462_462742

-- Define the speeds of the birds
def eagle_speed : ℕ := 15
def falcon_speed : ℕ := 46
def pelican_speed : ℕ := 33
def hummingbird_speed : ℕ := 30

-- Define the flying time for each bird
def flying_time : ℕ := 2

-- Calculate the total distance flown by all birds
def total_distance_flown : ℕ := (eagle_speed * flying_time) +
                                 (falcon_speed * flying_time) +
                                 (pelican_speed * flying_time) +
                                 (hummingbird_speed * flying_time)

-- The goal is to prove that the total distance flown by all birds is 248 miles
theorem birds_total_distance : total_distance_flown = 248 := by
  -- Proof here
  sorry

end birds_total_distance_l462_462742


namespace find_size_dihedral_angle_l462_462588

-- Define the conditions
variables (AB AD AA1 : ℝ)
variables (E midpoint_A_B : Prop)
variables (D1_E_C_D_dihedral_angle : Prop)

-- These are our given conditions
axiom condition1 : AB = 2
axiom condition2 : AD = 1
axiom condition3 : AA1 = 1

-- State that point E is the midpoint of edge AB
axiom midpoint_condition : E ∧ midpoint_A_B

-- Define the target dihedral angle and the answer
def size_dihedral_angle : ℝ := arctan (sqrt 2 / 2)

-- The math proof problem in Lean statement
theorem find_size_dihedral_angle :
  D1_E_C_D_dihedral_angle → size_dihedral_angle = arctan (sqrt 2 / 2) :=
by
  sorry

end find_size_dihedral_angle_l462_462588


namespace digit_after_decimal_l462_462308

theorem digit_after_decimal (n : ℕ) (h : n = 20) : 
  let d₁ := "142857"
      d₂ := "3"
      d₃ := "476190"
  in d₃[(h % 6)] = '7' := 
by
  -- Using Lean syntax to represent repeating decimals and positions
  let d₃ := "476190"
  have h_mod : h % 6 = 2 := by norm_num
  show d₃[h_mod] = '7'
  sorry

end digit_after_decimal_l462_462308


namespace isosceles_trapezoid_area_l462_462499

def trapezoid_area (l : ℝ) (α : ℝ) : ℝ :=
  (1 / 2) * l^2 * Real.sin (2 * α)

theorem isosceles_trapezoid_area (l : ℝ) (α : ℝ) :
  (∃ area : ℝ, area = trapezoid_area l α) :=
by
  let area := trapezoid_area l α
  use area
  sorry

end isosceles_trapezoid_area_l462_462499


namespace mans_rate_in_still_water_l462_462710

theorem mans_rate_in_still_water
  (V_m V_s : ℝ)
  (h_with_stream : V_m + V_s = 26)
  (h_against_stream : V_m - V_s = 4) :
  V_m = 15 :=
by {
  sorry
}

end mans_rate_in_still_water_l462_462710


namespace casey_saving_l462_462054

-- Define the conditions
def cost_per_hour_first_employee : ℝ := 20
def cost_per_hour_second_employee : ℝ := 22
def subsidy_per_hour : ℝ := 6
def hours_per_week : ℝ := 40

-- Define the weekly cost calculations
def weekly_cost_first_employee := cost_per_hour_first_employee * hours_per_week
def effective_cost_per_hour_second_employee := cost_per_hour_second_employee - subsidy_per_hour
def weekly_cost_second_employee := effective_cost_per_hour_second_employee * hours_per_week

-- State the theorem
theorem casey_saving :
    weekly_cost_first_employee - weekly_cost_second_employee = 160 := 
by
  sorry

end casey_saving_l462_462054


namespace arithmetic_sequence_iff_condition_l462_462673

-- Definitions: A sequence and the condition
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_iff_condition (a : ℕ → ℝ) :
  is_arithmetic_sequence a ↔ (∀ n : ℕ, 2 * a (n + 1) = a n + a (n + 2)) :=
by
  -- Proof is omitted.
  sorry

end arithmetic_sequence_iff_condition_l462_462673


namespace minimize_MA_plus_MF_l462_462529

open Real

noncomputable def parabola (M : ℝ × ℝ) : Prop :=
  M.snd ^ 2 = 4 * M.fst

noncomputable def circle (A : ℝ × ℝ) : Prop :=
  (A.fst - 4) ^ 2 + (A.snd - 1) ^ 2 = 1

noncomputable def focus : ℝ × ℝ := (1, 0)

theorem minimize_MA_plus_MF (M A : ℝ × ℝ) (hM : parabola M) (hA : circle A) :
    ∃ min_val : ℝ, min_val = 4 ∧ (min_val ≤ dist M A + dist M focus) :=
sorry

end minimize_MA_plus_MF_l462_462529


namespace chord_line_parabola_l462_462792

theorem chord_line_parabola (x1 x2 y1 y2 : ℝ) (hx1 : y1^2 = 8*x1) (hx2 : y2^2 = 8*x2)
  (hmid : (x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = -1) : 4*(1/2*(x1 + x2)) + (1/2*(y1 + y2)) - 3 = 0 :=
by
  sorry

end chord_line_parabola_l462_462792


namespace martins_pentagon_perimeter_l462_462952

theorem martins_pentagon_perimeter :
  ∀ (y : ℝ) (a b diagonal : ℝ),
  a = 3 * y → b = 4 * y → 
  diagonal = Real.sqrt (a^2 + b^2) → 
  5 * y = 16 →
  y = 16 / 5 →
  a = 9.6 →
  b = 12.8 → 
  ∀ (x : ℝ),
  (12.8 - x)^2 = x^2 + 9.6^2 →
  x = 2.8 →
  let o := 16 + a + x + x + a in
  o = 40.8 :=
by
  intros
  sorry

end martins_pentagon_perimeter_l462_462952


namespace smallest_odd_n_l462_462395

theorem smallest_odd_n (n : ℕ) (hn : Odd n) : 
  (2 ^ (((n + 1) ^ 2) / 7) > 1000) → n = 9 :=
begin
  sorry, -- Proof of the theorem
end

end smallest_odd_n_l462_462395


namespace square_of_1085_l462_462764

theorem square_of_1085 :
  (1085 : ℕ) = 1000 + 85 → 
  1085^2 = 1000^2 + 2 * 1000 * 85 + 85^2 → 
  1085^2 = 1177225 :=
by
  intro h,
  rw h,
  sorry

end square_of_1085_l462_462764


namespace solution_count_l462_462527

noncomputable def count_positive_integer_solutions : Nat :=
  ∑' (x y z : ℕ) in {
      (x, y, z) |
      x > 0 ∧ 
      y > 0 ∧ 
      z > 0 ∧ 
      x + y + z = 15
  }, 1

theorem solution_count : count_positive_integer_solutions = 91 :=
by
  -- Proof to be provided
  sorry

end solution_count_l462_462527


namespace product_of_ratios_l462_462298

theorem product_of_ratios 
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (hx1 : x1^3 - 3 * x1 * y1^2 = 2005)
  (hy1 : y1^3 - 3 * x1^2 * y1 = 2004)
  (hx2 : x2^3 - 3 * x2 * y2^2 = 2005)
  (hy2 : y2^3 - 3 * x2^2 * y2 = 2004)
  (hx3 : x3^3 - 3 * x3 * y3^2 = 2005)
  (hy3 : y3^3 - 3 * x3^2 * y3 = 2004) :
  (1 - x1/y1) * (1 - x2/y2) * (1 - x3/y3) = 1/1002 := 
sorry

end product_of_ratios_l462_462298


namespace minimum_balls_drawn_to_get_2_blue_and_1_red_is_8_l462_462299

-- Define the initial condition of the balls in the box
structure BoxCondition where
  blue_balls : Nat
  red_balls : Nat

-- Initial box condition as per the problem statement
def initialBoxCondition : BoxCondition :=
  { blue_balls := 7, red_balls := 5 }

-- The theorem we want to prove
theorem minimum_balls_drawn_to_get_2_blue_and_1_red_is_8
  (box : BoxCondition)
  (h1 : box.blue_balls = 7)
  (h2 : box.red_balls = 5) :
  ∃ n, n = 8 ∧
    ∀ drawn_balls, 
    (drawn_balls.length = n → 
     (drawn_balls.count (λ b, b = blue) ≥ 2 ∧
      drawn_balls.count (λ b, b = red) ≥ 1)) :=
sorry

end minimum_balls_drawn_to_get_2_blue_and_1_red_is_8_l462_462299


namespace combined_area_circle_square_l462_462462

/-- Consider a geometric configuration where there is a circle with diameter 12 meters
    and a square with one of its vertices at the circle's center. The side length
    of the square is equal to the diameter of the circle. -/
theorem combined_area_circle_square :
  let diameter := 12
  let radius := diameter / 2
  let side_length := diameter
  let area_circle := Real.pi * radius^2
  let area_square := side_length^2
  area_circle + area_square = 36 * Real.pi + 144 :=
by
  let diameter := 12
  let radius := diameter / 2
  let side_length := diameter
  let area_circle := Real.pi * radius^2
  let area_square := side_length^2
  show area_circle + area_square = 36 * Real.pi + 144
  sorry

end combined_area_circle_square_l462_462462


namespace rotation_problem_l462_462457

theorem rotation_problem (y : ℝ) (hy : y < 360) :
  (450 % 360 == 90) ∧ (y == 360 - 90) ∧ (90 + (360 - y) % 360 == 0) → y == 270 :=
by {
  -- Proof steps go here
  sorry
}

end rotation_problem_l462_462457


namespace quadratic_negative_roots_pq_value_l462_462467

theorem quadratic_negative_roots_pq_value (r : ℝ) :
  (∃ p q : ℝ, p = -87 ∧ q = -23 ∧ x^2 - (r + 7)*x + r + 87 = 0 ∧ p < r ∧ r < q)
  → ((-87)^2 + (-23)^2 = 8098) :=
by
  sorry

end quadratic_negative_roots_pq_value_l462_462467


namespace measure_angledb_l462_462915

-- Definitions and conditions:
def right_triangle (A B C : Type) : Prop :=
  ∃ (a b c : A), ∠A = 45 ∧ ∠B = 45 ∧ ∠C = 90

def angle_bisector (A B : Type) (D : Type) : Prop :=
  ∃ (AD BD : A), AD bisects ∠A ∧ BD bisects ∠B

-- Prove:
theorem measure_angledb (A B C D : Type) 
  (hABC : right_triangle A B C)
  (hAngleBisectors : angle_bisector A B D) : 
  ∠ ADB = 135 := 
sorry

end measure_angledb_l462_462915


namespace locus_of_Q_l462_462608

theorem locus_of_Q (P : ℝ × ℝ) (H1 : P.2 ^ 2 = 2 * P.1) 
    (M N : ℝ × ℝ) (H2 : (M.1 ^ 2 + M.2 ^ 2 = 1) ∧ (N.1 ^ 2 + N.2 ^ 2 = 1))
    (Q : ℝ × ℝ) (H3 : ∃ t > 0, P = (2 * t^2, t) ∧ 
                                (∀ y, y = 1 / t ↔ M.1 = t ^ (-2) ∧ M.2 = -y) ∧
                                (∀ y, y = 1 / t ↔ N.1 = t ^ (-2) ∧ N.2 = y) ∧ 
                                Q = (M.1 + N.1, M.2 + N.2)) 
    : ∀ (x y : ℝ), (y ^ 2 = -2 * x) → x ≤ 1 - real.sqrt(2) := 
by
  sorry

end locus_of_Q_l462_462608


namespace jongkook_second_day_milk_l462_462926

open Nat

def liters_to_milliliters (L : ℕ) : ℕ := L * 1000

def total_milk : ℕ := liters_to_milliliters 6 + 30
def first_day_milk : ℕ := liters_to_milliliters 3 + 7
def third_day_milk : ℕ := 840

def second_day_milk := total_milk - first_day_milk - third_day_milk

theorem jongkook_second_day_milk :
  second_day_milk = 2183 :=
by
  unfold second_day_milk
  unfold total_milk
  unfold first_day_milk
  unfold third_day_milk
  rw [←add_assoc, ←add_assoc, mul_comm 6 1000, mul_comm 3 1000]
  simp [liters_to_milliliters]
  sorry

end jongkook_second_day_milk_l462_462926


namespace min_sum_x1_x2_x3_x4_l462_462282

variables (x1 x2 x3 x4 : ℝ)

theorem min_sum_x1_x2_x3_x4 : 
  (x1 + x2 ≥ 12) → 
  (x1 + x3 ≥ 13) → 
  (x1 + x4 ≥ 14) → 
  (x3 + x4 ≥ 22) → 
  (x2 + x3 ≥ 23) → 
  (x2 + x4 ≥ 24) → 
  (x1 + x2 + x3 + x4 = 37) := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end min_sum_x1_x2_x3_x4_l462_462282


namespace sequence_expression_l462_462922

theorem sequence_expression (a : ℕ → ℚ)
  (h1 : a 1 = 2 / 3)
  (h2 : ∀ n : ℕ, a (n + 1) = (n / (n + 1)) * a n) :
  ∀ n : ℕ, a n = 2 / (3 * n) :=
sorry

end sequence_expression_l462_462922


namespace next_divisible_by_sum_of_digits_l462_462674

/-- The function to compute the sum of digits of a number -/
def sum_of_digits(n : ℕ) : ℕ := (to_digits 10 n).sum

/-- The problem statement -/
theorem next_divisible_by_sum_of_digits :
  ∃ n : ℕ, n > 1232 ∧ sum_of_digits n ∣ n ∧
           ∀ m : ℕ, 1232 < m < n → ¬ (sum_of_digits m ∣ m) :=
begin
  sorry
end

end next_divisible_by_sum_of_digits_l462_462674


namespace vector_equation_proof_l462_462819

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C P : V)

/-- The given condition. -/
def given_condition : Prop :=
  (P - A) + 2 • (P - B) + 3 • (P - C) = 0

/-- The target equality we want to prove. -/
theorem vector_equation_proof (h : given_condition A B C P) :
  P - A = (1 / 3 : ℝ) • (B - A) + (1 / 2 : ℝ) • (C - A) :=
sorry

end vector_equation_proof_l462_462819


namespace stadium_fee_difference_l462_462687

theorem stadium_fee_difference :
  let capacity := 2000
  let entry_fee := 20
  let full_fees := capacity * entry_fee
  let three_quarters_fees := (capacity * 3 / 4) * entry_fee
  full_fees - three_quarters_fees = 10000 :=
by
  sorry

end stadium_fee_difference_l462_462687


namespace ammonia_formation_l462_462498

theorem ammonia_formation (Li3N H2O LiOH NH3 : ℕ) (h₁ : Li3N = 1) (h₂ : H2O = 54) (h₃ : Li3N + 3 * H2O = 3 * LiOH + NH3) :
  NH3 = 1 :=
by
  sorry

end ammonia_formation_l462_462498


namespace part1_part2_combined_theorem_l462_462120

noncomputable def z1 (a : ℝ) : ℂ := (2 * a / (a - 1)) + (a^2 - 1) * complex.I
noncomputable def z2 (m : ℝ) : ℂ := m + (m - 1) * complex.I

theorem part1 (a : ℝ) (h1: z1 a = real_of_complex (z1 a)) : a = -1 :=
by sorry

theorem part2 (m : ℝ) (h2: |1| < |z2 m|) : m < 0 ∨ m > 1 :=
by sorry

theorem combined_theorem (a m : ℝ) (h1: z1 a = real_of_complex (z1 a)) (h2: |1| < |z2 m|) : a = -1 ∧ (m < 0 ∨ m > 1) :=
by sorry

end part1_part2_combined_theorem_l462_462120


namespace length_of_rhombus_side_proof_l462_462286

noncomputable def length_of_rhombus_side (d1 : ℝ) (area : ℝ) : ℝ :=
  let d2 := (area * 2) / d1
  let s := 2 * Real.sqrt ((d2 / 2) ^ 2 - 64)
  s

theorem length_of_rhombus_side_proof
  (d1 : ℝ := 16)
  (area : ℝ := 293.28484447717375)
  (side : ℝ := 32.984845004941284) :
  length_of_rhombus_side d1 area ≈ side := sorry

end length_of_rhombus_side_proof_l462_462286


namespace greatest_divisor_sum_of_first_fifteen_terms_l462_462321

theorem greatest_divisor_sum_of_first_fifteen_terms 
  (x c : ℕ) (hx : x > 0) (hc : c > 0):
  ∃ d, d = 15 ∧ d ∣ (15*x + 105*c) :=
by
  existsi 15
  split
  . refl
  . apply Nat.dvd.intro
    existsi (x + 7*c)
    refl
  sorry

end greatest_divisor_sum_of_first_fifteen_terms_l462_462321


namespace chord_slope_l462_462859

def ellipse := {P : ℝ × ℝ | 4 * P.1^2 + 9 * P.2^2 = 144}

noncomputable def midpoint := (3 : ℝ, 2 : ℝ)

theorem chord_slope (A B : ℝ × ℝ)
  (hA : 4 * A.1^2 + 9 * A.2^2 = 144)
  (hB : 4 * B.1^2 + 9 * B.2^2 = 144)
  (hM : (A.1 + B.1) / 2 = 3 ∧ (A.2 + B.2) / 2 = 2) :
  (A.2 - B.2) / (A.1 - B.1) = -2 / 3 :=
sorry

end chord_slope_l462_462859


namespace greatest_common_divisor_sum_arithmetic_sequence_l462_462341

theorem greatest_common_divisor_sum_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, d = 15 ∧ ∀ (n : ℕ), n = 15 → ∀ k : ℕ, k = 15 ∧ 15 ∣ (15 * (x + 7 * c)) :=
by
  sorry

end greatest_common_divisor_sum_arithmetic_sequence_l462_462341


namespace line_intersects_x_axis_l462_462010

theorem line_intersects_x_axis :
  ∃ x, ∃ y, y = 0 ∧ line_through (8, 2) (4, 6) x y :=
begin
  use 10,
  use 0,
  split,
  { refl, },
  { sorry, }
end

def line_through (p1 p2 : ℝ × ℝ) (x y : ℝ) : Prop :=
  let m := (p2.2 - p1.2) / (p2.1 - p1.1) in
  y = m * (x - p1.1) + p1.2

end line_intersects_x_axis_l462_462010


namespace exists_arith_prog_5_primes_exists_arith_prog_6_primes_l462_462076

-- Define the condition of being an arithmetic progression
def is_arith_prog (seq : List ℕ) : Prop :=
  ∀ (i : ℕ), i < seq.length - 1 → seq.get! (i + 1) - seq.get! i = seq.get! 1 - seq.get! 0

-- Define the condition of being prime
def all_prime (seq : List ℕ) : Prop :=
  ∀ (n : ℕ), n ∈ seq → Nat.Prime n

-- The main statements
theorem exists_arith_prog_5_primes :
  ∃ (seq : List ℕ), seq.length = 5 ∧ is_arith_prog seq ∧ all_prime seq := 
sorry

theorem exists_arith_prog_6_primes :
  ∃ (seq : List ℕ), seq.length = 6 ∧ is_arith_prog seq ∧ all_prime seq := 
sorry

end exists_arith_prog_5_primes_exists_arith_prog_6_primes_l462_462076


namespace greatest_divisor_of_sum_of_arithmetic_sequence_l462_462350

theorem greatest_divisor_of_sum_of_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ k : ℕ, (sum (λ n, x + n * c) (range 15)) = 15 * k :=
by sorry

end greatest_divisor_of_sum_of_arithmetic_sequence_l462_462350


namespace greatest_divisor_arithmetic_sequence_sum_l462_462327

theorem greatest_divisor_arithmetic_sequence_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ k, (15 * (x + 7 * c)) = 15 * k :=
sorry

end greatest_divisor_arithmetic_sequence_sum_l462_462327


namespace remainder_of_95_times_97_div_12_l462_462695

theorem remainder_of_95_times_97_div_12 : 
  (95 * 97) % 12 = 11 := by
  sorry

end remainder_of_95_times_97_div_12_l462_462695


namespace passes_through_fixed_point_l462_462036

def f1 (x : ℝ) : ℝ := 2^x
def f2 (x : ℝ) : ℝ := log x / log 2 -- since we don't have log_2, this is equivalent
def f3 (x : ℝ) : ℝ := x^(1/2)
def f4 (x : ℝ) : ℝ := x^2

theorem passes_through_fixed_point : f1 0 = 1 :=
by {
  sorry
}

end passes_through_fixed_point_l462_462036


namespace percentage_of_sum_l462_462177

theorem percentage_of_sum (x y P : ℝ) (h1 : 0.50 * (x - y) = (P / 100) * (x + y)) (h2 : y = 0.25 * x) : P = 30 :=
by
  sorry

end percentage_of_sum_l462_462177


namespace decode_rebus_l462_462771

theorem decode_rebus :
  ∃ (A B C : Nat), 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    (A + B * 10 + B = C * 100 + A) ∧ 
    A = 6 ∧ B = 9 ∧ C = 1 :=
by
  exists 6 9 1
  sorry

end decode_rebus_l462_462771


namespace calculate_volume_of_tetrahedron_l462_462979

noncomputable def volume_of_tetrahedron (PQ PR PS QR QS RS MQ : ℝ) : ℝ :=
  if h : MQ ≠ 0 then 12 / MQ else 0

theorem calculate_volume_of_tetrahedron
  (PQ PR PS QR QS RS : ℝ)
  (hPQ : PQ = 6)
  (hPR : PR = 3.5)
  (hPS : PS = 4)
  (hQR : QR = 5)
  (hQS : QS = 4.5)
  (hRS : RS = (9 / 2) * Real.sqrt 2)
  (MQ : ℝ) :
  volume_of_tetrahedron PQ PR PS QR QS RS MQ = 12 / MQ :=
begin
  sorry -- The proof is omitted
end

end calculate_volume_of_tetrahedron_l462_462979


namespace count_integer_values_sqrt_floor_eq_eight_l462_462553

theorem count_integer_values_sqrt_floor_eq_eight :
  {x : ℕ | nat.floor (real.sqrt x) = 8}.to_finset.card = 17 := 
sorry

end count_integer_values_sqrt_floor_eq_eight_l462_462553


namespace prove_difference_l462_462730

-- Definitions and assumptions: Dimensions of the rectangular prism
variable (x y z P D : ℝ)
variable (hx : x ≥ y)
variable (hz : z = y)
variable (hP : 4 * (x + y + z) = P)
variable (hD : sqrt (x^2 + y^2 + z^2) = D)

-- The statement we want to prove
theorem prove_difference (d : ℝ) (hd : d = x - y) :
  d = sqrt ((P^2 / 16) - D^2 + 2 * y^2) :=
sorry

end prove_difference_l462_462730


namespace donna_weekly_earnings_l462_462478

def hourly_rate_dog_walking := 10.0
def hours_per_day_dog_walking := 2
def days_per_week_dog_walking := 7
def hourly_rate_card_shop := 12.5
def hours_per_day_card_shop := 2
def days_per_week_card_shop := 5
def hourly_rate_babysitting := 10.0
def hours_per_weekend_babysitting := 4

theorem donna_weekly_earnings : 
  (hourly_rate_dog_walking * hours_per_day_dog_walking * days_per_week_dog_walking) + 
  (hourly_rate_card_shop * hours_per_day_card_shop * days_per_week_card_shop) + 
  (hourly_rate_babysitting * hours_per_weekend_babysitting) = 305.0 := 
by 
  sorry

end donna_weekly_earnings_l462_462478


namespace probability_even_spin_product_l462_462707

def first_spinner_values : list ℕ := [5, 6, 7, 8, 9]
def second_spinner_values : list ℕ := [2, 3, 4, 5, 6, 7]

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

def probability_odd_results (spinner : list ℕ) : ℚ :=
  let count_odd := (spinner.filter is_odd).length
  count_odd / spinner.length

def probability_even_product : ℚ :=
  1 - (probability_odd_results first_spinner_values * probability_odd_results second_spinner_values)

theorem probability_even_spin_product :
  probability_even_product = 7 / 10 :=
by
  sorry

end probability_even_spin_product_l462_462707


namespace circle_inscribed_square_area_l462_462405

noncomputable def square_area (s : ℝ) : ℝ := s^2

theorem circle_inscribed_square_area (r s : ℝ) (h1 : s = 2 * r) (h2 : π * r^2 = 9 * π) : square_area s = 36 :=
by
  -- Definition of the radius from the given area of the circle
  have r_eq : r = 3 := by
    sorry -- Placeholder for the proof r = 3

  -- Applying the side length of the square
  have s_eq : s = 6 := by
    rw [r_eq] at h1
    exact h1

  -- Use the definition of square_area to conclude the area
  rw [s_eq]
  exact eq.refl 36

end circle_inscribed_square_area_l462_462405


namespace total_number_of_feet_l462_462412

theorem total_number_of_feet 
  (H C F : ℕ)
  (h1 : H + C = 44)
  (h2 : H = 24)
  (h3 : F = 2 * H + 4 * C) : 
  F = 128 :=
by
  sorry

end total_number_of_feet_l462_462412


namespace greatest_divisor_of_sum_of_arithmetic_sequence_l462_462352

theorem greatest_divisor_of_sum_of_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ k : ℕ, (sum (λ n, x + n * c) (range 15)) = 15 * k :=
by sorry

end greatest_divisor_of_sum_of_arithmetic_sequence_l462_462352


namespace cos_B_eq_one_third_l462_462187

noncomputable def Triangle (A B C : Type) := (a b c : ℝ) (α β γ : ℝ)

variables {A B C : Type} [Triangle A B C]

variables {a b c α β γ: ℝ}

axiom sides_opposite_angles (T : Triangle A B C) : (b = T.b) ∧ (c = T.c) ∧ (a = T.a)
axiom condition (b c B C : ℝ) : b * (Real.cos C) = (3 * a - c) * (Real.cos B)

theorem cos_B_eq_one_third
  {b c a : ℝ} {B C: ℝ} (h : b * (Real.cos C) = (3 * a - c) * (Real.cos B)) : 
  Real.cos B = (1 / 3) :=
sorry

end cos_B_eq_one_third_l462_462187


namespace locus_of_midpoints_circle_l462_462228

noncomputable def midpoint (A B : Point) : Point := sorry

theorem locus_of_midpoints_circle
  (K : Circle)
  (O : Point) (r : ℝ) (P : Point)
  (h1 : K.center = O)
  (h2 : K.radius = r)
  (h3 : dist O P = (1/3) * r) : 
  exists (C : Circle),
    C.center = midpoint O P ∧ 
    C.radius = (1/6) * r ∧ 
    (∀ M : Point, (M ∈ (set_of (λ (M : Point), ∃ (A B : Point), (A ≠ B) ∧ (A ∈ K) ∧ (B ∈ K) ∧ (P ∈ line_through A B) ∧ (M = midpoint A B))) → (M ∈ C)) := 
sorry

end locus_of_midpoints_circle_l462_462228


namespace num_valid_n_values_l462_462936

theorem num_valid_n_values : 
  let n_range := {n | ∃ (q r : ℕ), 200 ≤ q ∧ q ≤ 1999 ∧ 0 ≤ r ∧ r ≤ 49 ∧ n = 50 * q + r}
  (count (λ n, ∃ (q r : ℕ), 200 ≤ q ∧ q ≤ 1999 ∧ 0 ≤ r ∧ r ≤ 49 ∧ n = 50 * q + r ∧ (q + r) % 7 = 0)) = 14400 :=
sorry

end num_valid_n_values_l462_462936


namespace increasing_interval_l462_462776

noncomputable def f (x : ℝ) := Real.logb 2 (5 - 4 * x - x^2)

theorem increasing_interval : ∀ {x : ℝ}, (-5 < x ∧ x ≤ -2) → f x = Real.logb 2 (5 - 4 * x - x^2) := by
  sorry

end increasing_interval_l462_462776


namespace sequence_general_term_l462_462141

theorem sequence_general_term (n : ℕ) (S : ℕ → ℤ) (a : ℕ → ℤ)
    (h₁ : S n = 2 * n ^ 2 - 3)
    (h₂ : ∀ n > 0, a n = S n - S (n - 1))
    (h₃ : a 1 = -1) :
    a n = if n = 1 then -1 else 4 * n - 2 := 
sorry

end sequence_general_term_l462_462141


namespace breakfast_calories_l462_462595

theorem breakfast_calories : ∀ (planned_calories : ℕ) (B : ℕ),
  planned_calories < 1800 →
  B + 900 + 1100 = planned_calories + 600 →
  B = 400 :=
by
  intros
  sorry

end breakfast_calories_l462_462595


namespace jack_marbles_l462_462212

theorem jack_marbles (marbles_start marbles_shared marbles_end : ℕ) 
  (h_start : marbles_start = 62)
  (h_shared : marbles_shared = 33) :
  marbles_end = marbles_start - marbles_shared :=
by
  -- Assume the given facts
  rw [h_start, h_shared]
  -- Use the facts to conclude
  show 29 = 62 - 33
  -- Evaluate the arithmetic
  exact rfl

end jack_marbles_l462_462212


namespace prime_dates_in_2008_l462_462775

/-- Define a list of prime numbers for use in our problem. --/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

/-- The list of prime months in a year. --/
def prime_months := [2, 3, 5, 7, 11]

/-- The list of prime days that a month can have. --/
def prime_days (is_leap_year : Bool) : List (ℕ × List ℕ) :=
  if is_leap_year then
    [(2, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]),
     (3, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]),
     (5, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]),
     (7, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]),
     (11, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29])]
  else
    []

/-- The final problem statement: Prove there are exactly 53 prime dates in the leap year 2008. --/
theorem prime_dates_in_2008 : ∑ (days : List ℕ) in (prime_days true).map Prod.snd, days.length = 53 := by
  sorry

end prime_dates_in_2008_l462_462775


namespace car_miles_per_gallon_in_city_l462_462384

-- Define the conditions and the problem
theorem car_miles_per_gallon_in_city :
  ∃ C H T : ℝ, 
    H = 462 / T ∧ 
    C = 336 / T ∧ 
    C = H - 12 ∧ 
    C = 32 :=
by
  sorry

end car_miles_per_gallon_in_city_l462_462384


namespace greatest_divisor_arithmetic_sequence_sum_l462_462328

theorem greatest_divisor_arithmetic_sequence_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ k, (15 * (x + 7 * c)) = 15 * k :=
sorry

end greatest_divisor_arithmetic_sequence_sum_l462_462328


namespace surface_area_堑堵_is_4sqrt2_plus_6_l462_462196

-- Define the surface area function
def surface_area_堑堵 (leg height : ℝ) : ℝ :=
  2 * (height * leg) + 2 * (leg * leg / 2) + 2 * (height * leg / 2)

-- Define the specific variables given in the problem
def hypotenuse : ℝ := 2
def height : ℝ := 2
def leg : ℝ := hypotenuse / real.sqrt 2

-- State the theorem to prove the surface area
theorem surface_area_堑堵_is_4sqrt2_plus_6 :
  surface_area_堑堵 leg height = 4 * real.sqrt 2 + 6 :=
by 
  sorry

end surface_area_堑堵_is_4sqrt2_plus_6_l462_462196


namespace sequence_property_l462_462086

theorem sequence_property (n : ℕ) :
  (∃ seq : list ℕ, seq.length = 2 * n ∧ ∀ k, (1 ≤ k ∧ k ≤ n) → 
    (∃ i j, i < j ∧ j - i - 1 = k ∧ seq.nth i = some k ∧ seq.nth j = some k)) ↔
  (∃ l : ℕ, n = 4 * l ∨ n = 4 * l - 1) :=
by sorry

end sequence_property_l462_462086


namespace number_in_original_position_l462_462714

theorem number_in_original_position :
  ∀ (n : ℕ) (a : Fin n.succ → ℕ) (j : ℕ),
  (∀ i, a ⟨i, Nat.lt_trans (Nat.lt_succ_self n) (Nat.succ_pos 0)⟩ = i.succ) →
  (∀ k, k < n.tsub 2 →
    (∀ op : Fin 2,
      if op = 0 then
        a ⟨k, Nat.lt_trans (Nat.lt_succ_self n) (Nat.succ_pos 0)⟩ -
        1 = a ⟨k + 1, Nat.lt_of_lt_of_le (Nat.lt_succ_self k) (Nat.le_of_lt k.lt_succ_self)⟩ - 1 ∧ 
        a ⟨k + 1, Nat.lt_of_lt_of_le (Nat.lt_succ_self k) (Nat.le_of_lt k.lt_succ_self)⟩ + 
        2 = a ⟨k + 2, Nat.lt_of_lt_of_le (add_lt_add_of_lt_of_le k.lt_succ_self (Nat.le_of_lt k.lt_succ_self)) (Nat.le_add_right 2)⟩ - 1
      else
        a ⟨k, Nat.lt_trans (Nat.lt_succ_self n) (Nat.succ_pos 0)⟩ + 
        1 = a ⟨k + 1, Nat.lt_of_lt_of_le (Nat.lt_succ_self k) (Nat.le_of_lt k.lt_succ_self)⟩ - 2 ∧ 
        a ⟨k + 1, Nat.lt_of_lt_of_le (Nat.lt_succ_self k) (Nat.le_of_lt k.lt_succ_self)⟩ - 
        2 = a ⟨k + 2, Nat.lt_of_lt_of_le (add_lt_add_of_lt_of_le k.lt_succ_self (Nat.le_of_lt k.lt_succ_self)) (Nat.le_add_right 2)⟩ + 1)) →
  (∀ i, a ⟨i, Nat.lt_trans (Nat.lt_succ_self n) (Nat.succ_pos 0)⟩ = i.succ) :=
by
  sorry

end number_in_original_position_l462_462714


namespace casey_saves_money_l462_462060

def first_employee_hourly_wage : ℕ := 20
def second_employee_hourly_wage : ℕ := 22
def subsidy_per_hour : ℕ := 6
def weekly_work_hours : ℕ := 40

theorem casey_saves_money :
  let first_employee_weekly_cost := first_employee_hourly_wage * weekly_work_hours
  let second_employee_effective_hourly_wage := second_employee_hourly_wage - subsidy_per_hour
  let second_employee_weekly_cost := second_employee_effective_hourly_wage * weekly_work_hours
  let savings := first_employee_weekly_cost - second_employee_weekly_cost
  savings = 160 :=
by
  sorry

end casey_saves_money_l462_462060


namespace necessary_but_not_sufficient_condition_l462_462157

example (x : ℝ) (k : ℤ) : 
  (sin x = 1/2 ↔ (x = π/6 + 2 * k * π)) ↔ false :=
by 
sorry

theorem necessary_but_not_sufficient_condition (k : ℤ) (x : ℝ) : 
  (∀ x, q x → p x) ∧ ¬(∀ x, p x → q x) :=
by 
  have p : sin x = 1/2 := sorry
  have q : x = π/6 + 2 * k * π := sorry
  sorry

end necessary_but_not_sufficient_condition_l462_462157


namespace molecular_weight_calculation_l462_462049

def molecular_weight (n_Ar n_Si n_H n_O : ℕ) (w_Ar w_Si w_H w_O : ℝ) : ℝ :=
  n_Ar * w_Ar + n_Si * w_Si + n_H * w_H + n_O * w_O

theorem molecular_weight_calculation :
  molecular_weight 2 3 12 8 39.948 28.085 1.008 15.999 = 304.239 :=
by
  sorry

end molecular_weight_calculation_l462_462049


namespace ride_time_l462_462597

theorem ride_time (d v : ℕ) (h_dist : d = 80) (h_speed : v = 16) : d / v = 5 :=
by
  rw [h_dist, h_speed]
  norm_num

end ride_time_l462_462597


namespace problem_statement_l462_462145

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x * (x + 4) else x * (x - 4)

theorem problem_statement (a : ℝ) (h : f a > f (8 - a)) : 4 < a :=
by sorry

end problem_statement_l462_462145


namespace min_packages_l462_462762

theorem min_packages (p : ℕ) (N : ℕ) :
  (N = 19 * p) →
  (N % 7 = 4) →
  (N % 11 = 1) →
  p = 40 :=
by
  sorry

end min_packages_l462_462762


namespace true_proposition_among_choices_l462_462124

def p : Prop := ∀ x : ℝ, 2^x > x^2

def q : Prop := ∀ a b : ℝ, (a > 1 ∧ b > 1) → ab > 1

theorem true_proposition_among_choices : ¬p ∧ ¬q :=
by
  unfold p q
  sorry

end true_proposition_among_choices_l462_462124


namespace area_percentage_l462_462390

variable (Ds Dr Rs Rr As Ar: ℝ)

-- Conditions
def diam_relation (Ds Dr : ℝ) : Prop := Dr = 0.6 * Ds
def radius_s (Ds : ℝ) : ℝ := Ds / 2
def radius_r (Dr : ℝ) : ℝ := Dr / 2
def area_s (Rs : ℝ) : ℝ := Real.pi * Rs^2
def area_r (Rr : ℝ) : ℝ := Real.pi * Rr^2

-- Proof
theorem area_percentage (h : diam_relation Ds Dr) : 
  let Rs := radius_s Ds,
      Rr := radius_r Dr,
      As := area_s Rs,
      Ar := area_r Rr in
  Ar / As = 0.36 :=
by
  sorry

end area_percentage_l462_462390


namespace find_roots_l462_462801

theorem find_roots : 
  ∀ x : ℝ, (x^2 - 5*x + 6) * (x - 3) * (x + 2) = 0 ↔ (x = -2 ∨ x = 2 ∨ x = 3) := by
  sorry

end find_roots_l462_462801


namespace average_age_of_all_individuals_l462_462661

-- Define ages and counts
def sixthGradersCount : ℕ := 40
def sixthGradersAvgAge : ℝ := 12

def parentsCount : ℕ := 60
def parentsAvgAge : ℝ := 40

-- Define total number of individuals and overall average age
def totalIndividuals : ℕ := sixthGradersCount + parentsCount
def totalAge : ℝ := (sixthGradersCount * sixthGradersAvgAge) + (parentsCount * parentsAvgAge)
def overallAvgAge : ℝ := totalAge / totalIndividuals

-- Theorem stating the result
theorem average_age_of_all_individuals :
  overallAvgAge = 28.8 :=
sorry

end average_age_of_all_individuals_l462_462661


namespace hyperprimes_correct_l462_462759

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def is_hyperprime (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ digits.length →
    ∀ i : ℕ, 0 ≤ i ∧ i + k ≤ digits.length →
      is_prime (digits.drop i).take k).foldl (λ acc d, acc * 10 + d) 0

theorem hyperprimes_correct :
  {n : ℕ | is_hyperprime n} = {2, 3, 5, 7, 23, 37, 53, 73, 373} :=
sorry

end hyperprimes_correct_l462_462759


namespace angle_DAB_in_regular_hexagon_is_60_degrees_l462_462581

noncomputable def regular_hexagon_internal_angle (n : ℕ) : ℝ :=
  if n = 6 then 120 else 0 -- Regular hexagon internal angle condition

theorem angle_DAB_in_regular_hexagon_is_60_degrees :
  (regular_hexagon_internal_angle 6 = 120) →
  ∀ (A B C D E F : Type) (AD : AD_diagonal),
  -- Conditions: ABCDEF is a regular hexagon and drawing diagonal AD
  let ABCDEF := regular_hexagon A B C D E F in
  -- Question and its proven answer
  ∃ (DAB : ℝ), DAB = 60 :=
begin
  intros,
  sorry -- Proof to be filled in
end

end angle_DAB_in_regular_hexagon_is_60_degrees_l462_462581


namespace quadratic_function_properties_max_min_f_2x_on_interval_l462_462115

noncomputable def f (x : ℝ) : ℝ := x^2 - x + 1

-- Prove the function satisfies given conditions
theorem quadratic_function_properties : 
  (f 0 = 1) ∧ (∀ x : ℝ, f (x + 1) - f x = 2 * x) :=
by {
  sorry
}

-- Prove max and min values of f(2^x) on the interval [-1, 1]
theorem max_min_f_2x_on_interval :
  let g (x : ℝ) := f (2^x) in 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → g x ≥ 3/4) ∧ (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ g x = 3) :=
by {
  sorry
}

end quadratic_function_properties_max_min_f_2x_on_interval_l462_462115


namespace angle_diff_l462_462749

-- Given conditions as definitions
def angle_A : ℝ := 120
def angle_B : ℝ := 50
def angle_D : ℝ := 60
def angle_E : ℝ := 140

-- Prove the difference between angle BCD and angle AFE is 10 degrees
theorem angle_diff (AB_parallel_DE : ∀ (A B D E : ℝ), AB_parallel_DE)
                 (angle_A_def : angle_A = 120)
                 (angle_B_def : angle_B = 50)
                 (angle_D_def : angle_D = 60)
                 (angle_E_def : angle_E = 140) :
    let angle_3 : ℝ := 180 - angle_A
    let angle_4 : ℝ := 180 - angle_E
    let angle_BCD : ℝ := angle_B + angle_D
    let angle_AFE : ℝ := angle_3 + angle_4
    angle_BCD - angle_AFE = 10 :=
by {
  sorry
}

end angle_diff_l462_462749


namespace problem1_problem2_l462_462675

variable {y : ℝ}

-- Condition given in the problem
def condition : Prop := 6 * y^2 + 7 = 5 * y + 12

-- First proof problem
theorem problem1 (h : condition) : (12 * y - 5)^2 = 145 :=
sorry

-- Second proof problem
theorem problem2 (h : condition) : (5 * y + 2)^2 = (4801 + 490 * Real.sqrt 145 + 3625) / 144 ∨ 
                                      (5 * y + 2)^2 = (4801 - 490 * Real.sqrt 145 + 3625) / 144 :=
sorry

end problem1_problem2_l462_462675


namespace min_questions_to_determine_numbers_min_remaining_numbers_l462_462386

namespace GameNumbers

-- Part (a)
theorem min_questions_to_determine_numbers (a b : Fin 10 → ℕ) (h : ∀ i j : Fin 10, a i + b j = a j + b i) :
  ∃ n : ℕ, n ≤ 11 :=
sorry

end GameNumbers

namespace TableNumbers

-- Part (b)
theorem min_remaining_numbers (m n : ℕ) (table : Fin m → Fin n → ℕ)
  (h : ∀ (i₁ i₂ : Fin m) (j₁ j₂ : Fin n), 
       table i₁ j₁ + table i₂ j₂ = table i₁ j₂ + table i₂ j₁) :
  ∃ remaining : Fin (m * n) → ℕ,
    set.toFinset (set.range remaining).card ≥ m + n - 1 :=
sorry

end TableNumbers

end min_questions_to_determine_numbers_min_remaining_numbers_l462_462386


namespace pipe_fill_ratio_l462_462963

theorem pipe_fill_ratio :
  let A := (1 : ℝ) / 6 in
  let B := (1 : ℝ) / 2 - A in
  B / A = 2 :=
by
  sorry

end pipe_fill_ratio_l462_462963


namespace dvd_cost_is_six_l462_462596

theorem dvd_cost_is_six (C : ℝ)
  (production_cost : ℝ = 2000)
  (selling_ratio : ℝ = 2.5)
  (dvd_sold_per_day : ℕ = 500)
  (working_days_week : ℕ = 5)
  (weeks : ℕ = 20)
  (profit : ℝ = 448000) :
  C = 6 :=
by
  sorry

end dvd_cost_is_six_l462_462596


namespace fraction_series_l462_462629

theorem fraction_series : ∀ n : ℕ, n > 1 → 
  ∃ (i j : ℕ), (i = n - 1) ∧ (∑ k in Finset.range (j - i + 1) \Finset.singleton 0, (1 / ((i + k :ℕ) * (i + k + 1 :ℕ)) : ℝ) = 1 / n) :=
by 
  sorry

end fraction_series_l462_462629


namespace greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l462_462357

theorem greatest_divisor_of_sum_first_15_terms_arithmetic_sequence
  (x c : ℕ) -- where x and c are positive integers
  (h_pos_x : 0 < x) -- x is positive
  (h_pos_c : 0 < c) -- c is positive
  : ∃ (d : ℕ), d = 15 ∧ ∀ (S : ℕ), S = 15 * x + 105 * c → d ∣ S :=
by
  sorry

end greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l462_462357


namespace smallest_binary_720_divisible_l462_462803

theorem smallest_binary_720_divisible:
  ∃ (n : ℕ), (∀ (d : ℕ), d ∈ List.of_digits n → d = 0 ∨ d = 1) ∧ (720 ∣ n) ∧ n = 1111111110000 :=
by
  sorry

end smallest_binary_720_divisible_l462_462803


namespace systematic_sample_contains_18_l462_462439

theorem systematic_sample_contains_18 (employees : Finset ℕ) (sample : Finset ℕ)
    (h1 : employees = Finset.range 52)
    (h2 : sample.card = 4)
    (h3 : ∀ n ∈ sample, n ∈ employees)
    (h4 : 5 ∈ sample)
    (h5 : 31 ∈ sample)
    (h6 : 44 ∈ sample) :
  18 ∈ sample :=
sorry

end systematic_sample_contains_18_l462_462439


namespace total_distance_travelled_l462_462413

theorem total_distance_travelled (total_time hours_foot hours_bicycle speed_foot speed_bicycle distance_foot : ℕ)
  (h1 : total_time = 7)
  (h2 : speed_foot = 8)
  (h3 : speed_bicycle = 16)
  (h4 : distance_foot = 32)
  (h5 : hours_foot = distance_foot / speed_foot)
  (h6 : hours_bicycle = total_time - hours_foot)
  (distance_bicycle := speed_bicycle * hours_bicycle) :
  distance_foot + distance_bicycle = 80 := 
by
  sorry

end total_distance_travelled_l462_462413


namespace solve_for_z_l462_462650

theorem solve_for_z (z : ℂ) (h : 5 - 3 * (I * z) = 3 + 5 * (I * z)) : z = I / 4 :=
sorry

end solve_for_z_l462_462650


namespace Billy_is_45_l462_462613

variable (B J : ℕ)

-- Condition 1: Billy's age is three times Joe's age
def condition1 : Prop := B = 3 * J

-- Condition 2: The sum of their ages is 60
def condition2 : Prop := B + J = 60

-- The theorem we want to prove: Billy's age is 45
theorem Billy_is_45 (h1 : condition1 B J) (h2 : condition2 B J) : B = 45 := 
sorry

end Billy_is_45_l462_462613


namespace find_compound_l462_462087

def compound (x : Type) := x

axiom produces_HCl_and_NH4OH (c : Type) (h2o : Type) (hcl : Type) (nh4oh : Type) :
  ReactsWithOneMoleToProducesOneMole h2o c hcl nh4oh → c = chloramine

def chloramine := NH2Cl

noncomputable def main_compound : Type := 
  ∃ (c : Type), produces_HCl_and_NH4OH c H2O HCl NH4OH

theorem find_compound (c : Type) (h2o : Type) :
  produces_HCl_and_NH4OH c h2o HCl NH4OH → c = chloramine :=
by sorry

end find_compound_l462_462087


namespace exists_unique_c_l462_462219

noncomputable def a_sequence : ℕ → ℝ
| 0       := 1
| (n + 1) := a_sequence n + b_sequence n

noncomputable def b_sequence : ℕ → ℝ
| 0       := 1
| (n + 1) := 3 * a_sequence n + b_sequence n

theorem exists_unique_c (c : ℝ) : c = Real.sqrt 3 ↔ ∀ n : ℕ, n * |c * a_sequence n - b_sequence n| < 2 :=
by
  sorry

end exists_unique_c_l462_462219


namespace decimal_to_binary_thirteen_l462_462470

theorem decimal_to_binary_thirteen : (13 : ℕ) = 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 :=
by
  sorry

end decimal_to_binary_thirteen_l462_462470


namespace yellow_ball_probability_l462_462900

theorem yellow_ball_probability (y w : ℕ) (h_y : y = 6) (h_w : w = 4) :
  let total_balls := y + w,
      yellow_first_draw := y - 1,
      remaining_balls := total_balls - 1 in
  (1 / total_balls) * ((yellow_first_draw:ℚ) / remaining_balls) = 5 / 9 :=
by
  sorry

end yellow_ball_probability_l462_462900


namespace pizza_volume_piece_l462_462426

theorem pizza_volume_piece (h : ℝ) (d : ℝ) (n : ℝ) (V_piece : ℝ) 
  (h_eq : h = 1 / 2) (d_eq : d = 16) (n_eq : n = 8) : 
  V_piece = 4 * Real.pi :=
by
  sorry

end pizza_volume_piece_l462_462426


namespace find_percentage_l462_462891

theorem find_percentage (x p : ℝ) (h1 : 0.25 * x = p * 10 - 30) (h2 : x = 680) : p = 20 := 
sorry

end find_percentage_l462_462891


namespace double_rooms_booked_l462_462044

theorem double_rooms_booked (S D : ℕ) 
  (h1 : S + D = 260) 
  (h2 : 35 * S + 60 * D = 14000) : 
  D = 196 :=
by
  sorry

end double_rooms_booked_l462_462044


namespace exists_rectangle_ABCD_l462_462117

variable {K L M A : Point}

-- Definitions based on the problem conditions
def IsOnLine (p : Point) (line : Line) : Prop := sorry -- Definition for point on line
def Triangle (A B C : Point) : Prop := sorry           -- Definition for a triangle
def Rectangle (A B C D : Point) : Prop := sorry        -- Definition for a rectangle

-- Assumptions about the triangle and the location of point A
axiom triangle_KLM : Triangle K L M
axiom A_on_extLK : ∃ p, IsOnLine A (line_through L K) ∧ p ≠ K ∧ IsOnLine p (segment_ext k L)

-- Problem statement: construct rectangle ABCD with specified points lying on specific lines
theorem exists_rectangle_ABCD :
  ∃ (B C D : Point), Rectangle A B C D ∧
    IsOnLine B (line_through K M) ∧
    IsOnLine C (line_through K L) ∧
    IsOnLine D (line_through L M) :=
sorry

end exists_rectangle_ABCD_l462_462117


namespace valid_triplets_l462_462085

-- defining primality check
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- defining ∈ℕ* properly for validity beyond
def proper_natural_num (n: ℕ) : Prop := 
  n ∈ ℕ
  n > 0

theorem valid_triplets (p q r: ℕ)
  (hpq⟩ proper_natural_num p) 
  (hrq⟩ proper_natural_num r) 
  (hqq⟩ proper_natural_num q) 
  (hge : p ≥ q ≥ r)
  (primes : (is_prime p ∧ is_prime q ) 
              ∨ (is_prime p ∧ is_prime r) 
              ∨ (is_prime q ∧ is_prime r))
  (hpositive : (p + q + r)^2 % (p * q * r) = 0) :
  (p = 3 ∧ q = 3 ∧ r = 3) 
  ∨ (p = 2 ∧ q = 2 ∧ r = 4) 
  ∨ (p = 3 ∧ q = 3 ∧ r = 12) 
  ∨ (p = 3 ∧ q = 2 ∧ r = 1) 
  ∨ (p = 3 ∧ q = 2 ∧ r = 25) := 
sorry

end valid_triplets_l462_462085


namespace balls_into_boxes_l462_462552

theorem balls_into_boxes :
  ∃ ways, 
    ways = (λ n m k, multiset.card (finset.filter (λ p, ∀ x, multiset.mem x p → multiset.card x > 0) (finset.partition n)).keys.card) 6 4 1 
    ∧ ways = 2 :=
sorry

end balls_into_boxes_l462_462552


namespace total_broken_marbles_correct_l462_462818

-- Definitions for the conditions
def first_set_total : ℕ := 50
def second_set_total : ℕ := 60
def first_set_broken_percentage : ℝ := 10 / 100
def second_set_broken_percentage : ℝ := 20 / 100

-- Definition of the total broken marbles
def total_broken_marbles : ℕ := 
  (first_set_total * first_set_broken_percentage).to_nat + 
  (second_set_total * second_set_broken_percentage).to_nat

-- The proof statement
theorem total_broken_marbles_correct : total_broken_marbles = 17 := 
  sorry

end total_broken_marbles_correct_l462_462818


namespace function_identity_l462_462940

theorem function_identity (f : ℕ → ℕ) (h₁ : ∀ n, 0 < f n)
  (h₂ : ∀ n, f (n + 1) > f (f n)) :
∀ n, f n = n :=
sorry

end function_identity_l462_462940


namespace circle_equation_l462_462269

theorem circle_equation (x y : ℝ) :
    (x - 1) ^ 2 + (y - 1) ^ 2 = 1 ↔ (∃ (C : ℝ × ℝ), C = (1, 1) ∧ ∃ (r : ℝ), r = 1 ∧ (x - C.1) ^ 2 + (y - C.2) ^ 2 = r ^ 2) :=
by
  sorry

end circle_equation_l462_462269


namespace tangent_line_at_a_eq_1_monotonicity_of_g_ln_intersection_slope_l462_462868

open Real

noncomputable def f (x : ℝ) := log x

noncomputable def g (x : ℝ) (a : ℝ) := log x + a * x^2 - (2 * a + 1) * x

-- 1.
theorem tangent_line_at_a_eq_1 : 
  let a := 1
  ∃ m b, (m * 1 + b = g 1 a) ∧ (y = m * x + b) = (y = -2) := 
by 
  sorry

-- 2.
theorem monotonicity_of_g (a : ℝ) (h : 0 < a) : 
  (0 < a ∧ a < 1 / 2) → 
  (∀ x, (0 < x ∧ x < 1) ∨ ((1 / (2 * a)) < x) → (0 < deriv (λ x, g x a) x)) ∧ 
  (∀ x, (1 / (2 * a) < x ∧ x < 1) → deriv (λ x, g x a) x < 0) ∧ 
  (a = 1 / 2 ∧ ∀ x, (0 < x) → deriv (λ x, g x a) x > 0) ∧ 
  (a > 1 / 2 ∧ ∀ x, ((0 < x ∧ x < 1 / (2 * a)) ∨ (1 < x)) → deriv (λ x, g x a) x > 0) ∧ 
  (∀ x, (1 / (2 * a) < x ∧ x < 1) → deriv (λ x, g x a) x < 0) :=
by
  sorry

-- 3.
theorem ln_intersection_slope (x₁ x₂ k : ℝ) (h₁ : x₁ < x₂) (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) :
  k = (log x₂ - log x₁) / (x₂ - x₁) → 1 / x₂ < k ∧ k < 1 / x₁ :=
by
  sorry

end tangent_line_at_a_eq_1_monotonicity_of_g_ln_intersection_slope_l462_462868


namespace total_sum_approx_l462_462435

noncomputable def totalMoney (B G : ℕ) (boys_amount girls_amount : ℝ) : ℝ :=
  B * boys_amount + G * girls_amount

theorem total_sum_approx (B G : ℕ) (boys_amount girls_amount : ℝ) (h1 : B + G = 100)
  (h2 : boys_amount = 3.60) (h3 : girls_amount = 2.40) (h4 : B = 60) :
  totalMoney B G boys_amount girls_amount ≈ 312 :=
by
  rw [h2, h3, h4]
  simp [totalMoney]
  norm_num
  sorry

end total_sum_approx_l462_462435


namespace f_solution_l462_462500

noncomputable def f : ℝ → ℝ := 
λ x, if x = 2 then 0 else x * f ((2 * x + 3) / (x - 2)) + 3

theorem f_solution :
  (∀ x, (x ≠ 2 → (f x = x * f ((2 * x + 3) / (x - 2)) + 3)) ∧ (x = 2 → f x = 0)) →
  ∀ x, f x = (3 * (x + 1) * (2 - x)) / (2 * (x ^ 2 + x + 1)) :=
sorry

end f_solution_l462_462500


namespace characterize_superinvariant_sets_l462_462307

def superinvariant_set (S : set ℝ) : Prop :=
∀ (x0 : ℝ) (a : ℝ) (h : a > 0), ∃ (b : ℝ), ∀ x : ℝ, 
(x ∈ S ↔ (x0 + a * (x - x0)) ∈ (λ x, x + b) '' S) ∧
((x0 + a * (x - x0)) ∈ S ↔ (∃ y : ℝ, y ∈ S ∧ (y + b = x0 + a * (x - x0))))

theorem characterize_superinvariant_sets (S : set ℝ) :
  superinvariant_set S ↔ 
  (S = set.univ ∨ (∃ x0 : ℝ, S = {x0}) ∨ 
   (∃ x0 : ℝ, S = set.univ \ {x0}) ∨ 
   (∃ x0 : ℝ, S = set.Iic x0) ∨ 
   (∃ x0 : ℝ, S = set.Ici x0) ∨ 
   (∃ x0 : ℝ, S = set.Iio x0) ∨ 
   (∃ x0 : ℝ, S = set.Ioi x0)) :=
sorry

end characterize_superinvariant_sets_l462_462307


namespace sandwiches_difference_l462_462635

-- Conditions definitions
def sandwiches_at_lunch_monday : ℤ := 3
def sandwiches_at_dinner_monday : ℤ := 2 * sandwiches_at_lunch_monday
def total_sandwiches_monday : ℤ := sandwiches_at_lunch_monday + sandwiches_at_dinner_monday
def sandwiches_on_tuesday : ℤ := 1

-- Proof goal
theorem sandwiches_difference :
  total_sandwiches_monday - sandwiches_on_tuesday = 8 :=
  by
  sorry

end sandwiches_difference_l462_462635


namespace subtracted_value_l462_462734

-- Given conditions
def chosen_number : ℕ := 110
def result_number : ℕ := 110

-- Statement to prove
theorem subtracted_value : ∃ y : ℕ, 3 * chosen_number - y = result_number ∧ y = 220 :=
by
  sorry

end subtracted_value_l462_462734


namespace parallel_vectors_not_same_direction_l462_462983

def parallel_vectors_must_have_same_direction : Prop :=
  ∀ (u v : Vector ℝ), (∃ c : ℝ, c ≠ 0 ∧ u = c • v) → (direction_of u = direction_of v)

theorem parallel_vectors_not_same_direction : ¬parallel_vectors_must_have_same_direction :=
by
  sorry

end parallel_vectors_not_same_direction_l462_462983


namespace Donna_total_earnings_l462_462480

theorem Donna_total_earnings :
  let walking_earnings := 10 * 2 * 7 in
  let card_shop_earnings := 12.5 * 2 * 5 in
  let babysitting_earnings := 10 * 4 in
  walking_earnings + card_shop_earnings + babysitting_earnings = 305 := 
by 
  let walking_earnings := 10 * 2 * 7
  let card_shop_earnings := 12.5 * 2 * 5
  let babysitting_earnings := 10 * 4
  show walking_earnings + card_shop_earnings + babysitting_earnings = 305
  sorry

end Donna_total_earnings_l462_462480


namespace range_of_a_l462_462544

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 < 1 ∧ x2 > 1 ∧ x1 * x1 + (a * a - 1) * x1 + a - 2 = 0 ∧ x2 * x2 + (a * a - 1) * x2 + a - 2 = 0) ↔ -2 < a ∧ a < 1 :=
sorry

end range_of_a_l462_462544


namespace determine_placement_of_single_tile_l462_462039

theorem determine_placement_of_single_tile :
  ∃ squares : set (ℕ × ℕ), 
  (∀ (r c : ℕ), r < 11 → c < 11 → ((\[colored\] pattern)) → squares ((r, c)) ∧ 
  ((commonality of placement conditions for both patterns applying \(\boxed{}\))) :=
begin
  sorry
end

end determine_placement_of_single_tile_l462_462039


namespace part_one_part_two_part_three_l462_462870

variable {a b : ℝ}

def f (x : ℝ) := a * x + b / x

theorem part_one (h : (a, b) ∈ ℝ) (h_slope : (deriv f 1) = 1) : b = a - 1 := sorry

def g (x : ℝ) := a * x + (a - 1) / x - log x

theorem part_two (h : a ∈ ℝ) (h_g : ∀ x > 0, g x ≥ 1) : a ≥ 1 := sorry

theorem part_three (h : a ≥ 1) (x1 x2 : ℝ) (h_g_eq : g x1 = g x2) : x1 + x2 ≥ 2 := sorry

end part_one_part_two_part_three_l462_462870


namespace sandwiches_difference_l462_462634

-- Conditions definitions
def sandwiches_at_lunch_monday : ℤ := 3
def sandwiches_at_dinner_monday : ℤ := 2 * sandwiches_at_lunch_monday
def total_sandwiches_monday : ℤ := sandwiches_at_lunch_monday + sandwiches_at_dinner_monday
def sandwiches_on_tuesday : ℤ := 1

-- Proof goal
theorem sandwiches_difference :
  total_sandwiches_monday - sandwiches_on_tuesday = 8 :=
  by
  sorry

end sandwiches_difference_l462_462634


namespace polygon_sides_eq_six_l462_462291

theorem polygon_sides_eq_six (n : ℕ) (h1 : (n - 2) * 180 = 2 * 360) : n = 6 :=
by
  sorry

end polygon_sides_eq_six_l462_462291


namespace greatest_divisor_sum_of_first_fifteen_terms_l462_462318

theorem greatest_divisor_sum_of_first_fifteen_terms 
  (x c : ℕ) (hx : x > 0) (hc : c > 0):
  ∃ d, d = 15 ∧ d ∣ (15*x + 105*c) :=
by
  existsi 15
  split
  . refl
  . apply Nat.dvd.intro
    existsi (x + 7*c)
    refl
  sorry

end greatest_divisor_sum_of_first_fifteen_terms_l462_462318


namespace ten_years_less_average_age_l462_462575

-- Defining the conditions formally
def lukeAge : ℕ := 20
def mrBernardAgeInEightYears : ℕ := 3 * lukeAge

-- Lean statement to prove the problem
theorem ten_years_less_average_age : 
  mrBernardAgeInEightYears - 8 = 52 → (lukeAge + (mrBernardAgeInEightYears - 8)) / 2 - 10 = 26 := 
by
  intros h
  sorry

end ten_years_less_average_age_l462_462575


namespace initial_amount_calculation_l462_462448

theorem initial_amount_calculation :
  ∃ P : ℝ, 
  let A := 2000 
  let R := 4.761904761904762 
  let T := 3 
  in P = A / (1 + (R * T / 100)) ∧ P = 1750 :=
by 
  sorry

end initial_amount_calculation_l462_462448


namespace greatest_divisor_arithmetic_sum_l462_462369

theorem greatest_divisor_arithmetic_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ d, d = 15 ∧ ∀ S : ℕ, S = 15 * x + 105 * c → d ∣ S :=
by 
  sorry

end greatest_divisor_arithmetic_sum_l462_462369


namespace probability_at_least_one_passes_l462_462297

theorem probability_at_least_one_passes (prob_pass : ℚ) (prob_fail : ℚ) (p_all_fail: ℚ):
  (prob_pass = 1/3) →
  (prob_fail = 1 - prob_pass) →
  (p_all_fail = prob_fail ^ 3) →
  (1 - p_all_fail = 19/27) :=
by
  intros hpp hpf hpaf
  sorry

end probability_at_least_one_passes_l462_462297


namespace truck_distance_on_rough_terrain_l462_462027

theorem truck_distance_on_rough_terrain :
  ∀ (miles_on_smooth : ℝ) (gallons_on_smooth : ℝ) (gallons_on_rough : ℝ) (efficiency_drop : ℝ),
  miles_on_smooth = 300 ∧ gallons_on_smooth = 10 ∧ gallons_on_rough = 15 ∧ efficiency_drop = 0.1 →
  let mileage_smooth := miles_on_smooth / gallons_on_smooth in
  let mileage_rough := mileage_smooth * (1 - efficiency_drop) in
  mileage_rough * gallons_on_rough = 405 := 
by 
  sorry

end truck_distance_on_rough_terrain_l462_462027


namespace average_score_for_girls_at_both_schools_combined_l462_462750

/-
  The following conditions are given:
  - Average score for boys at Lincoln HS = 75
  - Average score for boys at Monroe HS = 85
  - Average score for boys at both schools combined = 82
  - Average score for girls at Lincoln HS = 78
  - Average score for girls at Monroe HS = 92
  - Average score for boys and girls combined at Lincoln HS = 76
  - Average score for boys and girls combined at Monroe HS = 88

  The goal is to prove that the average score for the girls at both schools combined is 89.
-/
theorem average_score_for_girls_at_both_schools_combined 
  (L l M m : ℕ)
  (h1 : (75 * L + 78 * l) / (L + l) = 76)
  (h2 : (85 * M + 92 * m) / (M + m) = 88)
  (h3 : (75 * L + 85 * M) / (L + M) = 82)
  : (78 * l + 92 * m) / (l + m) = 89 := 
sorry

end average_score_for_girls_at_both_schools_combined_l462_462750


namespace greatest_common_divisor_sum_arithmetic_sequence_l462_462348

theorem greatest_common_divisor_sum_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, d = 15 ∧ ∀ (n : ℕ), n = 15 → ∀ k : ℕ, k = 15 ∧ 15 ∣ (15 * (x + 7 * c)) :=
by
  sorry

end greatest_common_divisor_sum_arithmetic_sequence_l462_462348


namespace line_equation_through_point_l462_462791

noncomputable def line_p (a b : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (1 / 2 * a * b = 6) ∧ (2 / a + 3 / (2 * b) = 1)

theorem line_equation_through_point (a b : ℝ) (h : line_p a b) : 
  ∃ l : ℝ -> ℝ -> Prop, ∀ x y : ℝ, l x y ↔ 3 * x + 4 * y = 12 := 
by 
  use λ x y, 3 * x + 4 * y = 12
  sorry

end line_equation_through_point_l462_462791


namespace _l462_462063

noncomputable theorem ellipse_chord_focus_BF :
  ∀ (x y : ℝ),
  (x^2 / 36 + y^2 / 16 = 1) →
  ((x - 2 * Real.sqrt 5)^2 + y^2 = 4) →
  ∃ Bx : ℝ, Bx = 1.8 * Real.sqrt 5 ∧
  Real.sqrt ((Bx - 2 * Real.sqrt 5)^2) = 0.4 * Real.sqrt 5 := by
  intro x y h_ellipse h_distance
  sorry

end _l462_462063


namespace correct_value_l462_462735

-- Define a function representing the original expression
def original_expression (a b : ℝ) : ℝ :=
  (a + 2 * b) ^ 2 - (a + b) * (a - b)

-- State the theorem with a proof obligation skipped
theorem correct_value (a b : ℝ) (h1 : a = -1/2) (h2 : b = 2) : original_expression a b = 16 := 
by
  rw [h1, h2]
  -- Expand and simplify the expression mathematically and show the final value matches 16
  sorry

end correct_value_l462_462735


namespace range_of_x_squared_plus_y_squared_l462_462667

def increasing (f : ℝ → ℝ) := ∀ x y, x < y → f x < f y
def symmetric_about_origin (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem range_of_x_squared_plus_y_squared 
  (f : ℝ → ℝ) 
  (h_incr : increasing f) 
  (h_symm : symmetric_about_origin f) 
  (h_ineq : ∀ x y, f (x^2 - 6 * x) + f (y^2 - 8 * y + 24) < 0) : 
  ∀ x y, 16 < x^2 + y^2 ∧ x^2 + y^2 < 36 := 
sorry

end range_of_x_squared_plus_y_squared_l462_462667


namespace Calculate_BF_l462_462205

variables (A B C D E F : Type) [Triangle ABC]
           (BC AC BD BE BF : ℝ)
           (angle_C_is_right : ∠ C = Real.pi / 2)
           (BC_val : BC = 1)
           (AC_val : AC = 7 / 8)
           (D_on_AB : Point_on_segment D A B)
           (BD_val : BD = 1 / 2)
           (ED_perpendicular_BC : ∃ E, E ∈ Line_perpendicular_to ED BC)
           (DF_parallel_AE : ∀ DF AE, Parallel DF AE)
          
theorem Calculate_BF : BF = 355 / 113 - 3 :=
by
  sorry

end Calculate_BF_l462_462205


namespace inequality_implies_double_l462_462559

-- Define the condition
variables {x y : ℝ}

theorem inequality_implies_double (h : x < y) : 2 * x < 2 * y :=
  sorry

end inequality_implies_double_l462_462559


namespace complex_magnitude_sum_l462_462490

theorem complex_magnitude_sum : |(3 - 5 * complex.I)| + |(3 + 5 * complex.I)| = 2 * real.sqrt 34 :=
by
  sorry

end complex_magnitude_sum_l462_462490


namespace expectation_and_stddev_sum_l462_462854

variable (X : ℝ)
variable (Y : ℝ := 2 * X - 5)
variable (EX : ℝ := 5)
variable (DX : ℝ := 9)

theorem expectation_and_stddev_sum :
  (let EY := 2 * EX - 5 in
   let DY := 4 * DX in
   let sigmaY := Real.sqrt DY in
   EY + sigmaY = 11) :=
by
  sorry

end expectation_and_stddev_sum_l462_462854


namespace area_of_four_union_triangles_l462_462817

-- Definition of the problem's conditions
structure Triangle :=
  (leg1 : ℝ)
  (leg2 : ℝ)
  (is_right_angle : Bool)
  (is_isosceles : Bool)

-- Definition of our specific triangle setup
def T : Triangle := {
  leg1 := 2,
  leg2 := 2,
  is_right_angle := true,
  is_isosceles := true
}

-- The main theorem to be proven
theorem area_of_four_union_triangles (T1 T2 T3 T4 : Triangle) : 
  T1 = T ∧ T2 = T ∧ T3 = T ∧ T4 = T →
  ∃ area : ℝ, area = 5 :=
by sorry

-- The area of the region covered by the union of these four triangular regions is 5
example : area_of_four_union_triangles T T T T := 
by sorry

end area_of_four_union_triangles_l462_462817


namespace casey_savings_l462_462056

-- Define the constants given in the problem conditions
def wage_employee_1 : ℝ := 20
def wage_employee_2 : ℝ := 22
def subsidy : ℝ := 6
def hours_per_week : ℝ := 40

-- Define the weekly cost of each employee
def weekly_cost_employee_1 := wage_employee_1 * hours_per_week
def weekly_cost_employee_2 := (wage_employee_2 - subsidy) * hours_per_week

-- Define the savings by hiring the cheaper employee
def savings := weekly_cost_employee_1 - weekly_cost_employee_2

-- Theorem stating the expected savings
theorem casey_savings : savings = 160 := by
  -- Proof is not included
  sorry

end casey_savings_l462_462056


namespace nguyen_fabric_yards_l462_462637

open Nat

theorem nguyen_fabric_yards :
  let fabric_per_pair := 8.5
  let pairs_needed := 7
  let fabric_still_needed := 49
  let total_fabric_needed := pairs_needed * fabric_per_pair
  let fabric_already_have := total_fabric_needed - fabric_still_needed
  let yards_of_fabric := fabric_already_have / 3
  yards_of_fabric = 3.5 := by
    sorry

end nguyen_fabric_yards_l462_462637


namespace greatest_divisor_arithmetic_sequence_sum_l462_462330

theorem greatest_divisor_arithmetic_sequence_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ k, (15 * (x + 7 * c)) = 15 * k :=
sorry

end greatest_divisor_arithmetic_sequence_sum_l462_462330


namespace probability_of_winning_pair_is_5_over_11_l462_462733

def card := (color : String) × (label : String)

def deck : List card := 
[("red", "A"), ("red", "B"), ("red", "C"), ("red", "D"),
 ("green", "A"), ("green", "B"), ("green", "C"), ("green", "D"),
 ("blue", "A"), ("blue", "B"), ("blue", "C"), ("blue", "D")]

def winning_pair (c1 c2 : card) : Prop :=
(c1.1 = c2.1) ∨ (c1.2 = c2.2)

def count_win_pairs (deck : List card) : Nat :=
(deck.combinations 2).count (λ pair, winning_pair pair.head pair.tail.head)

def total_possible_pairs (deck : List card) : Nat :=
(deck.combinations 2).length

theorem probability_of_winning_pair_is_5_over_11 :
  (count_win_pairs deck : ℚ) / (total_possible_pairs deck : ℚ) = 5 / 11 :=
by
  sorry

end probability_of_winning_pair_is_5_over_11_l462_462733


namespace isosceles_triangle_fraction_l462_462119

theorem isosceles_triangle_fraction (a b : ℝ) (h1 : ∃ 𝜃: ℝ, 𝜃 = 20*π/180) 
  (h2 : a = 2 * b * sin (10*π/180)) :
  (a^3 + b^3) / (a * b^2) = 3 :=
by
  sorry

end isosceles_triangle_fraction_l462_462119


namespace max_value_of_y_l462_462996

theorem max_value_of_y : ∃ x : ℝ, (λ x, (sin x)^2 - 4 * (cos x) + 2).max = 6 :=
by
  sorry

end max_value_of_y_l462_462996


namespace food_cost_approx_l462_462719

-- Define the given constants
def total_paid : ℝ := 35.75
def sales_tax_rate : ℝ := 0.095
def tip_rate : ℝ := 0.10

-- Define the cost of the food
noncomputable def food_cost : ℝ := total_paid / (1 + sales_tax_rate + tip_rate)

-- Prove food_cost is approximately 29.92
theorem food_cost_approx : food_cost ≈ 29.92 :=
by
  sorry

end food_cost_approx_l462_462719


namespace hexagon_diagonal_angle_l462_462578

theorem hexagon_diagonal_angle (A B C D E F : Type) [hexagon: regular_hexagon A B C D E F]
  (interior_angle : ∀ {a b c}, a ≠ b ∧ b ≠ c ∧ c ≠ a → angle a b c = 120) :
  angle D A B = 30 :=
  sorry

end hexagon_diagonal_angle_l462_462578


namespace polynomial_image_ranges_l462_462709

theorem polynomial_image_ranges (p: ℝ^2 → ℝ) (hp: polynomial ℝ p):
  ∃ k: ℝ, p '' (set.univ : set (ℝ × ℝ)) = {k} ∨ p '' (set.univ : set (ℝ × ℝ)) = set.Ici k ∨ 
                                       p '' (set.univ : set (ℝ × ℝ)) = set.Iic k ∨ 
                                       p '' (set.univ : set (ℝ × ℝ)) = set.univ ∨ 
                                       p '' (set.univ : set (ℝ × ℝ)) = set.Ioi k ∨ 
                                       p '' (set.univ : set (ℝ × ℝ)) = set.Iio k :=
sorry

end polynomial_image_ranges_l462_462709


namespace no_primes_between_factorial_and_double_l462_462811

theorem no_primes_between_factorial_and_double (n : ℕ) (h : n > 1) : 
  ∀ p : ℕ, prime p → p > n! + 1 → p < n! + 2 * n → false :=
by
  sorry

end no_primes_between_factorial_and_double_l462_462811


namespace right_angled_triangles_count_l462_462163

theorem right_angled_triangles_count : 
  (∃ (a b : ℕ), a^2 + (sqrt 1001)^2 = b^2 ∧ (b - a) * (b + a) = 1001) → ∃ n, n = 4 :=
by
  sorry

end right_angled_triangles_count_l462_462163


namespace odd_function_value_at_neg_one_l462_462398

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then 4 * x + b else -(4 * (-x) + b)

theorem odd_function_value_at_neg_one (b : ℝ) : f (-1) = -3 := 
by 
  have h₁ := if_neg (not_le.mpr (neg_one_lt_zero)) -- Case where x < 0
  simp [f, (* (4 * 1) + b)] at h₁,
  sorry -- Placeholder for the remainder of the proof

end odd_function_value_at_neg_one_l462_462398


namespace casey_savings_l462_462057

-- Define the constants given in the problem conditions
def wage_employee_1 : ℝ := 20
def wage_employee_2 : ℝ := 22
def subsidy : ℝ := 6
def hours_per_week : ℝ := 40

-- Define the weekly cost of each employee
def weekly_cost_employee_1 := wage_employee_1 * hours_per_week
def weekly_cost_employee_2 := (wage_employee_2 - subsidy) * hours_per_week

-- Define the savings by hiring the cheaper employee
def savings := weekly_cost_employee_1 - weekly_cost_employee_2

-- Theorem stating the expected savings
theorem casey_savings : savings = 160 := by
  -- Proof is not included
  sorry

end casey_savings_l462_462057


namespace nested_expression_value_l462_462461

theorem nested_expression_value : 
  4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4)))))))) = 87380 :=
by 
  sorry

end nested_expression_value_l462_462461


namespace max_value_point_is_one_l462_462150

noncomputable def f (x : ℝ) := 3 * x - x^3

-- The Lean statement asserting that the maximum value point is x = 1
theorem max_value_point_is_one : 
  ∀ x : ℝ, (f' x = 3 - 3 * x^2) → (∀ x < -1, f' x < 0) → (∀ -1 < x < 1, f' x > 0) → (∀ x > 1, f' x < 0) → 1 = argmax f :=
sorry

end max_value_point_is_one_l462_462150


namespace average_age_of_all_individuals_l462_462660

-- Define ages and counts
def sixthGradersCount : ℕ := 40
def sixthGradersAvgAge : ℝ := 12

def parentsCount : ℕ := 60
def parentsAvgAge : ℝ := 40

-- Define total number of individuals and overall average age
def totalIndividuals : ℕ := sixthGradersCount + parentsCount
def totalAge : ℝ := (sixthGradersCount * sixthGradersAvgAge) + (parentsCount * parentsAvgAge)
def overallAvgAge : ℝ := totalAge / totalIndividuals

-- Theorem stating the result
theorem average_age_of_all_individuals :
  overallAvgAge = 28.8 :=
sorry

end average_age_of_all_individuals_l462_462660


namespace number_of_digits_3_pow_20_mul_7_pow_15_l462_462798

noncomputable def num_digits (x : ℕ) : ℕ :=
  (Real.log10 (x)).toNat + 1

theorem number_of_digits_3_pow_20_mul_7_pow_15 :
  let log10_3 := 0.4771
  let log10_7 := 0.8451
  let term_1 := 20 * log10_3
  let term_2 := 15 * log10_7
  let total_log := term_1 + term_2
  num_digits (3^20 * 7^15) = 23 :=
by
  have log_3_pow_20 : Real.log10 (3^20) ≈ 20 * log10_3 := by sorry
  have log_7_pow_15 : Real.log10 (7^15) ≈ 15 * log10_7 := by sorry
  have combined_log : Real.log10 (3^20 * 7^15) ≈ total_log := by
    rw [Real.log10_mul (3^20) (7^15)]
    rw log_3_pow_20
    rw log_7_pow_15
    sorry
  have num_digits_calc : num_digits (3^20 * 7^15) = 23 := by
    rw [num_digits]
    have floor_value : total_log.toNat = 22 := by sorry
    rw [floor_value]
    done
  exact num_digits_calc

end number_of_digits_3_pow_20_mul_7_pow_15_l462_462798


namespace find_number_l462_462183

theorem find_number (r : ℕ) (hr : 0 < r) (n : ℝ) (hn : r / n = 8.2) (rem : ℝ) (hrem : rem = r % n) (hrem_app : rem ≈ 3) : n ≈ 15 :=
sorry

end find_number_l462_462183


namespace tangent_line_circle_l462_462582

noncomputable def parametric_circle (a θ : ℝ) : ℝ × ℝ := (a + Real.cos θ, Real.sin θ)

noncomputable def polar_line (ρ θ : ℝ) : Prop := ρ * Real.sin (θ - π / 4) = sqrt 2 / 2

theorem tangent_line_circle (a : ℝ) :
  (∀ θ ρ : ℝ, polar_line ρ θ → ∃ θ, parametric_circle a θ = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ ρ = 1) →
  a = -1 + sqrt 2 ∨ a = -1 - sqrt 2 :=
by
  sorry

end tangent_line_circle_l462_462582


namespace norm_sum_eq_l462_462491

-- Define the complex numbers
def c1 : ℂ := 3 - 5 * complex.I
def c2 : ℂ := 3 + 5 * complex.I

-- Define the norm (magnitude) of complex numbers
def norm_c1 : ℝ := complex.abs c1
def norm_c2 : ℝ := complex.abs c2

-- The statement to prove
theorem norm_sum_eq : norm_c1 + norm_c2 = 2 * real.sqrt 34 :=
by sorry

end norm_sum_eq_l462_462491


namespace alice_additional_plates_l462_462030

theorem alice_additional_plates (initial_stack : ℕ) (first_addition : ℕ) (total_when_crashed : ℕ) 
  (h1 : initial_stack = 27) (h2 : first_addition = 37) (h3 : total_when_crashed = 83) : 
  total_when_crashed - (initial_stack + first_addition) = 19 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end alice_additional_plates_l462_462030


namespace min_value_l462_462610

noncomputable def min_value_expr (a b c d : ℝ) : ℝ :=
  (a + b) / c + (b + c) / a + (c + d) / b

theorem min_value 
  (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  min_value_expr a b c d ≥ 6 
  := sorry

end min_value_l462_462610


namespace max_value_of_x_squared_plus_xy_plus_y_squared_l462_462978

theorem max_value_of_x_squared_plus_xy_plus_y_squared
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 - x * y + y^2 = 9) : 
  (x^2 + x * y + y^2) ≤ 27 :=
sorry

end max_value_of_x_squared_plus_xy_plus_y_squared_l462_462978


namespace find_t_l462_462093

open Complex Real

theorem find_t (a b : ℂ) (t : ℝ) (h₁ : abs a = 3) (h₂ : abs b = 5) (h₃ : a * b = t - 3 * I) :
  t = 6 * Real.sqrt 6 := by
  sorry

end find_t_l462_462093


namespace sum_c_eq_l462_462126

-- Defining the arithmetic and geometric sequences
def a (n : ℕ) : ℕ := n + 1
def b (n : ℕ) : ℕ := 2 ^ n
def c (n : ℕ) : ℕ := (a n) * (b n)

-- The sum of the first n terms of a sequence
def S (f : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in Finset.range n, f (i + 1)

-- Problem statement
theorem sum_c_eq (n : ℕ) : S c n = n * 2^(n + 1) := by
  sorry

end sum_c_eq_l462_462126


namespace card_transformation_general_card_transformation_l462_462300

def machine1 (a b : ℕ) : ℕ × ℕ := (a + 1, b + 1)

def machine2 (a b : ℕ) (h : a % 2 = 0 ∧ b % 2 = 0) : ℕ × ℕ := (a / 2, b / 2)

def machine3 (a b c : ℕ) : ℕ × ℕ := (a, c)

def can_produce_card (initial target : ℕ × ℕ) : Prop := sorry

theorem card_transformation (initial : ℕ × ℕ) :
  initial = (5, 19) →
  (can_produce_card initial (1, 50) ∧ ¬can_produce_card initial (1, 100)) :=
begin
  sorry
end

theorem general_card_transformation (a b : ℕ) (h : a < b) :
  ∃ d : ℕ, (∀ n : ℕ, n = 1 + k * d → can_produce_card (a, b) (1, n)) :=
begin
  sorry
end

end card_transformation_general_card_transformation_l462_462300


namespace skateboard_price_after_discounts_l462_462242

-- Defining all necessary conditions based on the given problem.
def original_price : ℝ := 150
def discount1 : ℝ := 0.40 * original_price
def price_after_discount1 : ℝ := original_price - discount1
def discount2 : ℝ := 0.25 * price_after_discount1
def final_price : ℝ := price_after_discount1 - discount2

-- Goal: Prove that the final price after both discounts is $67.50.
theorem skateboard_price_after_discounts : final_price = 67.50 := by
  sorry

end skateboard_price_after_discounts_l462_462242


namespace range_of_a_l462_462518

open Classical Real

variables (a : ℝ) (p q : Prop)

noncomputable def proposition_p : Prop :=
  ∀ x ∈ set.Icc 1 2, x^2 - a ≥ 0

noncomputable def proposition_q : Prop :=
  ∃ x : ℝ, x^2 + 2 * a * x + (2 - a) = 0

theorem range_of_a (h : proposition_p a ∧ proposition_q a) : a ≤ -2 ∨ a = 1 :=
by
  sorry

end range_of_a_l462_462518


namespace expansion_properties_l462_462167

open BigOperators

def binomial_expansion (x : ℕ → ℕ) :=
  ∑ i in Finset.range 11, x i

theorem expansion_properties :
  let a : ℕ → ℕ := λ i, 2^i * Nat.choose 10 i in
  (a 0 = 1) ∧
  (∑ i in Finset.range 1 \u 10, a i = 3^10 - 1) ∧
  (∃ i, Nat.choose 10 i = Nat.choose 10 5) ∧
  (a 2 = 9 * a 1) :=
by {
  sorry
}

end expansion_properties_l462_462167


namespace smallest_prime_factor_2379_l462_462701

-- Define the given number
def n : ℕ := 2379

-- Define the condition that 3 is a prime number.
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

-- Define the smallest prime factor
def smallest_prime_factor (n p : ℕ) : Prop :=
  is_prime p ∧ p ∣ n ∧ (∀ q, is_prime q → q ∣ n → p ≤ q)

-- The statement that 3 is the smallest prime factor of 2379
theorem smallest_prime_factor_2379 : smallest_prime_factor n 3 :=
sorry

end smallest_prime_factor_2379_l462_462701


namespace area_of_triangle_ABC_l462_462088

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  let cross_product := v.1 * w.2 - v.2 * w.1
  (Real.abs cross_product) / 2

theorem area_of_triangle_ABC (A B C : ℝ × ℝ)
  (hA : A = (2, -3))
  (hB : B = (1, 4))
  (hC : C = (-3, -2)) :
  area_of_triangle A B C = 17 :=
by
  unfold area_of_triangle
  rw [hA, hB, hC]
  simp
  sorry

end area_of_triangle_ABC_l462_462088


namespace quadratic_roots_condition_l462_462815

theorem quadratic_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ k*x^2 + 2*x + 1 = 0 ∧ k*y^2 + 2*y + 1 = 0) ↔ (k < 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_roots_condition_l462_462815


namespace ratio_of_areas_l462_462748

lemma line_DE_intersects_parabola_once 
  (A A' B C D E F : Point) 
  (hA : A.y = 0)
  (hA' : A'.x = -A.x ∧ A'.y = 0)
  (hB : B ∈ parabola)
  (hC : C ∈ parabola)
  (hA'B : line_through A' B)
  (hA'C : line_through A' C)
  (hD : D ∈ segment A B)
  (hE : E ∈ segment A C)
  (h_ratio : (|CE| / |CA| = |AD| / |AB|)) :
  ∃! G, G ∈ line_through D E ∧ G ∈ parabola :=
sorry

theorem ratio_of_areas 
  {A A' B C D E F : Point} 
  (hA : A.y = 0)
  (hA' : A'.x = -A.x ∧ A'.y = 0)
  (hB : B ∈ parabola)
  (hC : C ∈ parabola)
  (hA'B : line_through A' B)
  (hA'C : line_through A' C)
  (hD : D ∈ segment A B)
  (hE : E ∈ segment A C)
  (h_ratio : (|CE| / |CA| = |AD| / |AB|))
  (hF : F ∈ line_through D E ∧ F ∈ parabola) 
  (S1 : ℝ) (S2 : ℝ)
  (hS1 : area B C F = S1)
  (hS2 : area A D E = S2) :
  S1 / S2 = 1 :=
sorry

end ratio_of_areas_l462_462748


namespace minimal_benches_l462_462430

theorem minimal_benches (x : ℕ) 
  (standard_adults : ℕ := x * 8) (standard_children : ℕ := x * 12)
  (extended_adults : ℕ := x * 8) (extended_children : ℕ := x * 16) 
  (hx : standard_adults + extended_adults = standard_children + extended_children) :
  x = 1 :=
by
  sorry

end minimal_benches_l462_462430


namespace derivative_y_l462_462520

noncomputable theory
open Classical

axiom y_def : ∀ (x : ℝ), y = -2 * exp(x) * sin(x)

theorem derivative_y : ∀ (x : ℝ), (deriv (λ x, -2 * exp(x) * sin(x))) = -2 * exp(x) * (sin(x) + cos(x)) :=
by
  intro x
  have : y = -2 * exp(x) * sin(x), from y_def x
  sorry

end derivative_y_l462_462520


namespace hexagon_cosine_relationship_l462_462607

theorem hexagon_cosine_relationship
  (ABCDEF : Hexagon) (inscribed_in_circle : ABCDEF.InscribedInCircle)
  (h_AB : ABCDEF.side_length AB = 3)
  (h_BC : ABCDEF.side_length BC = 3)
  (h_CD : ABCDEF.side_length CD = 3)
  (h_DE : ABCDEF.side_length DE = 3)
  (h_EF : ABCDEF.side_length EF = 3)
  (h_FA : ABCDEF.side_length FA = 2) :
  (1 - Real.cos (ABCDEF.angle_at B)) * (1 - Real.cos (ABCDEF.angle_at ACF)) = 1 / 9 := 
sorry

end hexagon_cosine_relationship_l462_462607


namespace minimum_distance_sum_l462_462517

open Real

structure Point :=
  (x : ℝ)
  (y : ℝ)

noncomputable def distance (P1 P2 : Point) : ℝ :=
  sqrt ((P2.x - P1.x) ^ 2 + (P2.y - P1.y) ^ 2)

def A : Point := ⟨0, 0⟩
def B : Point := ⟨1, 0⟩
def C : Point := ⟨0, 2⟩
def D : Point := ⟨3, 3⟩

theorem minimum_distance_sum (P : Point) :
  isMinimum (distance P A + distance P B + distance P C + distance P D) (2 * sqrt 3 + sqrt 5) := 
sorry

end minimum_distance_sum_l462_462517


namespace sum_unseen_faces_of_two_dice_eq_21_l462_462278

theorem sum_unseen_faces_of_two_dice_eq_21 :
  (∀ (x y : ℕ), x + y = 7 → x ∈ {1, 2, 3, 4, 5, 6} ∧ y ∈ {1, 2, 3, 4, 5, 6}) → 
  ((∀ x ∈ {6, 2, 3}, ∃ y, x + y = 7) ∧ (∀ z ∈ {1, 4, 5}, ∃ w, z + w = 7)) →
  (∃ (x₁ x₂ x₃ y₁ y₂ y₃ : ℕ), x₁ + x₂ + x₃ + y₁ + y₂ + y₃ = 21) :=
sorry

end sum_unseen_faces_of_two_dice_eq_21_l462_462278


namespace sum_squares_first_15_pos_ints_l462_462293

theorem sum_squares_first_15_pos_ints :
  (∑ k in finset.range 15, (k + 1) ^ 2) = 1240 :=
by
  sorry

end sum_squares_first_15_pos_ints_l462_462293


namespace divisor_ratio_l462_462855

-- Define the number of positive divisors
def num_divisors (n : Nat) : Nat :=
  ∏ p in (Multiset.nodupPowerset (Multiset.ofList (Nat.factors n))).map Multiset.card, p + 1

-- Define specific values for 3600 and 36 divisors
def m := num_divisors 3600
def n := num_divisors 36

theorem divisor_ratio :
  m / n = 5 :=
by
  sorry

end divisor_ratio_l462_462855


namespace sum_of_roots_in_interval_l462_462138

noncomputable def f (x : ℝ) : ℝ := sorry

def is_even_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ (∀ x, f x = f (x + p))

def single_root_in_interval (f : ℝ → ℝ) (a b r : ℝ) : Prop :=
  (f r = 0) ∧ (∀ x, a ≤ x ∧ x ≤ b → (f x = 0 → x = r))

theorem sum_of_roots_in_interval
  (f : ℝ → ℝ) (p : ℝ) (a b : ℝ) (r : ℝ)
  (hf_even_periodic : is_even_periodic f p)
  (hf_single_root : single_root_in_interval f 0 2 r)
  (interval_length_17 : b = 17)
  (period_4 : p = 4)
  (root_r : r = 1):
  let S := (Σ k : ℕ in finset.range 5, r + k * 4) in
  S = 45 :=
by 
  sorry

end sum_of_roots_in_interval_l462_462138


namespace max_sum_arith_seq_l462_462839

theorem max_sum_arith_seq :
  let a1 := 29
  let d := 2
  let a_n (n : ℕ) := a1 + (n - 1) * d
  let S_n (n : ℕ) := n / 2 * (a1 + a_n n)
  S_n 10 = S_n 20 → S_n 20 = 960 := by
sorry

end max_sum_arith_seq_l462_462839


namespace justin_tim_games_l462_462070

theorem justin_tim_games :
  let n := 8 in
  let count := (choose (n - 2) 2) in
  count = 15 := by 
  sorry

end justin_tim_games_l462_462070


namespace probability_of_closer_to_6_l462_462021

noncomputable def probability_closer_to_6 (x : ℝ) : Prop :=
  x ∈ set.Icc (0 : ℝ) 8 ∧ abs (x - 6) < abs (x - 0)

theorem probability_of_closer_to_6 : 
  ∃ p : ℝ, p = 0.6 ∧ measure_theory.measure.probability_closer_to_6 p :=
sorry

end probability_of_closer_to_6_l462_462021


namespace area_relationship_l462_462404

def sides : ℕ × ℕ × ℕ := (12, 35, 37)
def triangle_area (base height : ℕ) : ℕ := (base * height) / 2
def circle_area (radius : ℝ) : ℝ := π * radius^2
def diameter (hypotenuse : ℕ) : ℝ := hypotenuse / 2

theorem area_relationship (A B C : ℝ) : 
  let (a, b, c) := sides in
  let triangle_area := triangle_area a b in
  let circle_radius := diameter c in
  let x := circle_area circle_radius in
  2 * C = x ∧ C - 210 = A + B :=
  sorry

end area_relationship_l462_462404


namespace find_set_l462_462946

/-- Definition of set A -/
def setA : Set ℝ := { x : ℝ | abs x < 4 }

/-- Definition of set B -/
def setB : Set ℝ := { x : ℝ | x^2 - 4 * x + 3 > 0 }

/-- Definition of the intersection A ∩ B -/
def intersectionAB : Set ℝ := { x : ℝ | abs x < 4 ∧ (x > 3 ∨ x < 1) }

/-- Definition of the set we want to find -/
def setDesired : Set ℝ := { x : ℝ | abs x < 4 ∧ ¬(abs x < 4 ∧ (x > 3 ∨ x < 1)) }

/-- The statement to prove -/
theorem find_set :
  setDesired = { x : ℝ | 1 ≤ x ∧ x ≤ 3 } :=
sorry

end find_set_l462_462946


namespace smallest_positive_period_pi_symmetry_axis_center_solution_set_nonpositive_max_min_values_l462_462867

noncomputable def f (x : ℝ) : ℝ := 4 * cos x * sin (x + π / 6)

theorem smallest_positive_period_pi :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') :=
sorry

theorem symmetry_axis_center :
  ∃ k ∈ (Z : set ℤ), (∀ x, f x = f (π / 6 + (k * π) / 2)) ∧ (f (-π / 12 + (k * π) / 2) = 1) :=
sorry

theorem solution_set_nonpositive :
  {x : ℝ | f x ≤ 0} = {x : ℝ | ∃ k ∈ (Z : set ℤ), k * π - π / 2 ≤ x ∧ x ≤ k * π - π / 6} :=
sorry

theorem max_min_values :
  ∃ a b ∈ ℝ, (a = -π / 6 ∧ b = π / 4) ∧ (∀ x ∈ set.Icc (-π / 6) (π / 4), f x ≤ 3) ∧ (∀ x ∈ set.Icc (-π / 6) (π / 4), 0 ≤ f x) :=
sorry

end smallest_positive_period_pi_symmetry_axis_center_solution_set_nonpositive_max_min_values_l462_462867


namespace giyoon_chocolates_l462_462548

theorem giyoon_chocolates (C X : ℕ) (h1 : C = 8 * X) (h2 : C = 6 * (X + 1) + 4) : C = 40 :=
by sorry

end giyoon_chocolates_l462_462548


namespace pizza_volume_one_piece_l462_462420

theorem pizza_volume_one_piece :
  ∀ (h t: ℝ) (d: ℝ) (n: ℕ), d = 16 → t = 1/2 → n = 8 → h = 8 → 
  ( (π * (d / 2)^2 * t) / n = 4 * π ) :=
by 
  intros h t d n hd ht hn hh
  sorry

end pizza_volume_one_piece_l462_462420


namespace constant_term_of_binomial_expansion_is_9_l462_462920

open Nat

theorem constant_term_of_binomial_expansion_is_9 :
  (∃ n : ℕ, 4 ^ n + 2 ^ n = 72 ∧ 
  ∃ T : ℕ → ℝ, T 2 = binom n 1 * 3 ^ 1 ∧ T 2 = 9) := 
begin
  sorry
end

end constant_term_of_binomial_expansion_is_9_l462_462920


namespace largest_divisor_even_composite_l462_462810

noncomputable def even_composite_nat : Type := { n : ℕ // even n ∧ (∃ p q : ℕ, 2 ≤ p ∧ 2 ≤ q ∧ n = p * q) }

theorem largest_divisor_even_composite (n : even_composite_nat) : n.val ∣ n.val! - n.val :=
by
  sorry

end largest_divisor_even_composite_l462_462810


namespace find_a_minus_c_l462_462611

theorem find_a_minus_c (a c : ℝ)
  (h1 : ∀ x, f x = a * x + c)
  (h2 : ∀ x, g x = -4 * x + 6)
  (h3 : ∀ x, h x = f (g x))
  (h4 : ∀ y, h⁻¹ y = y + 8) :
  a - c = 25 / 4 := by
  sorry

end find_a_minus_c_l462_462611


namespace math_proof_problem_l462_462034

theorem math_proof_problem (a : ℝ) : 
  (a^8 / a^4 ≠ a^4) ∧ ((a^2)^3 ≠ a^6) ∧ ((3*a)^3 ≠ 9*a^3) ∧ ((-a)^3 * (-a)^5 = a^8) := 
by 
  sorry

end math_proof_problem_l462_462034


namespace tangent_distance_l462_462436

theorem tangent_distance (x y : ℝ) (hx : (x - 2) ^ 2 + (y - 3) ^ 2 = 1) : 
  ∃ (A : ℝ × ℝ), A = (-3, 4) ∧
  ∀ (C : ℝ × ℝ), C = (2, 3) ∧
  ∀ (r : ℝ), r = 1 → 
  ∀ (d : ℝ), d = real.sqrt ( (-3 - 2) ^ 2 + (4 - 3) ^ 2 ) →
  real.sqrt (d^2 - r^2) = 5 :=
begin
  sorry
end

end tangent_distance_l462_462436


namespace value_of_k_l462_462159

theorem value_of_k :
  (∀ x : ℝ, x ^ 2 - x - 2 > 0 → 2 * x ^ 2 + (5 + 2 * k) * x + 5 * k < 0 → x = -2) ↔ -3 ≤ k ∧ k < 2 :=
sorry

end value_of_k_l462_462159


namespace cos_2beta_l462_462822

open Real

theorem cos_2beta (α β : ℝ)
  (h1 : sin (α - β) = 3 / 5)
  (h2 : sin (α + β) = -3 / 5)
  (h3 : α - β ∈ Ioc (π / 2) π)
  (h4 : α + β ∈ Ioc (3 * π / 2) (2 * π)) : 
  cos (2 * β) = -1 := 
by
  sorry

end cos_2beta_l462_462822


namespace card_arrangement_ways_l462_462482

theorem card_arrangement_ways : 
  ∃ (arrangements : Finset (List ℕ)), arrangements.card = 16 ∧
  ∀ (cards : Fin 8 → ℕ) (arr : List ℕ) (h_arr : arr ∈ arrangements) (i : Fin 8),
  (arr.remove_nth i).sorted (<) ∨ (arr.remove_nth i).sorted (>) :=
begin
  sorry
end

end card_arrangement_ways_l462_462482


namespace bert_profit_l462_462452

-- Definitions from the conditions
def purchase_price (sale_price : ℤ) (markup : ℤ) : ℤ :=
  sale_price - markup

def tax (sale_price : ℤ) (tax_rate : ℤ) : ℤ :=
  sale_price * tax_rate / 100

def profit (sale_price : ℤ) (purchase_price : ℤ) (tax : ℤ) : ℤ :=
  sale_price - purchase_price - tax

-- Constants specific to the problem
def sale_price : ℤ := 90
def markup : ℤ := 10
def tax_rate : ℤ := 10

-- Proof problem
theorem bert_profit :
  let p := purchase_price sale_price markup,
      t := tax sale_price tax_rate,
      pr := profit sale_price p t
  in pr = 1 :=
by
  sorry

end bert_profit_l462_462452


namespace sequence_formula_sum_of_b_m_l462_462515

noncomputable def sequence_sum (n : ℕ) : ℚ :=
  (9 * n * n - 7 * n) / 2

def general_term (n : ℕ) : ℤ :=
  (9 * n - 8 : ℤ)

def terms_in_interval (m : ℕ) : ℕ :=
  9^(2*m - 1) - 9^(m - 1)

def sum_b_m (m : ℕ) : ℚ :=
  (9^(2*m + 1) + 1 - 10 * 9^m + 1) / 80

theorem sequence_formula (n : ℕ) (hn : 0 < n) :
  general_term n = 9 * n - 8 := sorry

theorem sum_of_b_m (m : ℕ) (hm : 0 < m) :
  (∀ k, 0 < k ∧ k ≤ m → terms_in_interval k) →
  ∑ k in (finset.range m).succ, (terms_in_interval k) = sum_b_m m := sorry

end sequence_formula_sum_of_b_m_l462_462515


namespace number_of_non_congruent_triangles_l462_462192

-- Conditions
def point (x y : ℝ) := (x, y)

def vertex1 := point 0 0
def vertex2 := point 1 0
def vertex3 := point 1.5 0.5
def vertex4 := point 2.5 0.5
def midpoint1 := point 0.5 0.25 -- (midpoint between vertex 1 and 3)
def midpoint2 := point 1.75 0.25 -- (midpoint between vertex 2 and 4)
def midpoint3 := point 1.75 0 -- (midpoint between vertex 2 and 4 on the bottom)
def center := point 1.25 0.25 -- (center of the parallelogram)

-- Question: Verifying the number of non-congruent triangles
def count_non_congruent_triangles : ℕ :=
  9

theorem number_of_non_congruent_triangles :
  ∃ n : ℕ, n = count_non_congruent_triangles :=
begin
  use 9,
  refl,
end

end number_of_non_congruent_triangles_l462_462192


namespace greatest_divisor_arithmetic_sum_l462_462368

theorem greatest_divisor_arithmetic_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ d, d = 15 ∧ ∀ S : ℕ, S = 15 * x + 105 * c → d ∣ S :=
by 
  sorry

end greatest_divisor_arithmetic_sum_l462_462368


namespace minimize_quadratic_expression_l462_462106

theorem minimize_quadratic_expression : ∃ x : ℝ, (∀ y : ℝ, 3 * x^2 - 18 * x + 7 ≤ 3 * y^2 - 18 * y + 7) ∧ x = 3 :=
by
  sorry

end minimize_quadratic_expression_l462_462106


namespace product_modulo_7_l462_462094

theorem product_modulo_7 : (1729 * 1865 * 1912 * 2023) % 7 = 6 :=
by
  sorry

end product_modulo_7_l462_462094


namespace greatest_divisor_of_sum_of_arithmetic_sequence_l462_462356

theorem greatest_divisor_of_sum_of_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ k : ℕ, (sum (λ n, x + n * c) (range 15)) = 15 * k :=
by sorry

end greatest_divisor_of_sum_of_arithmetic_sequence_l462_462356


namespace greatest_common_divisor_sum_arithmetic_sequence_l462_462342

theorem greatest_common_divisor_sum_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, d = 15 ∧ ∀ (n : ℕ), n = 15 → ∀ k : ℕ, k = 15 ∧ 15 ∣ (15 * (x + 7 * c)) :=
by
  sorry

end greatest_common_divisor_sum_arithmetic_sequence_l462_462342


namespace milk_transfer_proof_l462_462441

theorem milk_transfer_proof :
  ∀ (A B C x : ℝ), 
  A = 1232 →
  B = A - 0.625 * A → 
  C = A - B → 
  B + x = C - x → 
  x = 154 :=
by
  intros A B C x hA hB hC hEqual
  sorry

end milk_transfer_proof_l462_462441


namespace cos_B_equals_3_over_4_l462_462204

variables {A B C : ℝ} {a b c R : ℝ} (h₁ : b * Real.sin B - a * Real.sin A = (1/2) * a * Real.sin C)
  (h₂ :  2 * R ^ 2 * Real.sin B * (1 - Real.cos (2 * A)) = (1 / 2) * a * b * Real.sin C)

theorem cos_B_equals_3_over_4 : Real.cos B = 3 / 4 := by
  sorry

end cos_B_equals_3_over_4_l462_462204


namespace casey_saving_l462_462052

-- Define the conditions
def cost_per_hour_first_employee : ℝ := 20
def cost_per_hour_second_employee : ℝ := 22
def subsidy_per_hour : ℝ := 6
def hours_per_week : ℝ := 40

-- Define the weekly cost calculations
def weekly_cost_first_employee := cost_per_hour_first_employee * hours_per_week
def effective_cost_per_hour_second_employee := cost_per_hour_second_employee - subsidy_per_hour
def weekly_cost_second_employee := effective_cost_per_hour_second_employee * hours_per_week

-- State the theorem
theorem casey_saving :
    weekly_cost_first_employee - weekly_cost_second_employee = 160 := 
by
  sorry

end casey_saving_l462_462052


namespace solve_remainder_problem_l462_462014

def remainder_problem : Prop :=
  ∃ (n : ℕ), 
    (n % 481 = 179) ∧ 
    (n % 752 = 231) ∧ 
    (n % 1063 = 359) ∧ 
    (((179 + 231 - 359) % 37) = 14)

theorem solve_remainder_problem : remainder_problem :=
by
  sorry

end solve_remainder_problem_l462_462014


namespace complex_square_eq_l462_462131

theorem complex_square_eq (i : ℂ) (hi : i * i = -1) : (1 + i)^2 = 2 * i := 
by {
  -- marking the end of existing code for clarity
  sorry
}

end complex_square_eq_l462_462131


namespace prove_f_monotone_odd_arith_sequence_l462_462851

variables {α : Type*} [linear_ordered_field α]

-- Definitions as conditions
def monotone_function (f : α → α) := ∀ x y, x < y → f(x) < f(y)
def odd_function (f : α → α) := ∀ x, f(-x) = -f(x)
def arithmetic_sequence (a : ℕ → α) (a₃ : α) := a₃ > 0 ∧ ∀ n, a n = a 0 + n * ((a 3 - a 0) / 3)

-- The math problem statement
theorem prove_f_monotone_odd_arith_sequence
  (f : α → α)
  (a : ℕ → α)
  (a₃ a₁ a₅ : α)
  (h₁ : monotone_function f)
  (h₂ : odd_function f)
  (h₃ : arithmetic_sequence a a₃)
  (h₄ : a 3 = a₃)
  (h₅ : a 1 = a₁)
  (h₆ : a 5 = a₅) :
  f(a₁) + f(a₃) + f(a₅) > 0 := 
sorry

end prove_f_monotone_odd_arith_sequence_l462_462851


namespace distance_DE_l462_462683

-- Define the points A, B, C, and P, and their given relationships.
def A : Type := (ℝ × ℝ)
def B : Type := (ℝ × ℝ)
def C : Type := (ℝ × ℝ)
def P : Type := (ℝ × ℝ)

-- Assume the lengths given in the problem.
def AB := 15
def BC := 18
def AC := 21
def PC := 15

-- Given the quadrilaterals are trapezoids and the distance DE.
theorem distance_DE (A B C P D E : Type) 
    (h_AB : distance A B = 15) 
    (h_BC : distance B C = 18) 
    (h_AC : distance A C = 21) 
    (h_PC : distance P C = 15) 
    (h_trapezoid_ABCD : is_trapezoid A B C D) 
    (h_trapezoid_ABCE : is_trapezoid A B C E) : 
  distance D E = 35 :=
sorry

end distance_DE_l462_462683


namespace min_value_expression_l462_462511

theorem min_value_expression (x : ℝ) (hx : x > 0) : x + 4/x ≥ 4 :=
sorry

end min_value_expression_l462_462511


namespace maximum_value_of_f_l462_462847

noncomputable def f (x : ℝ) : ℝ := -- assume f is defined
 sorry

axiom f_derivative (x : ℝ) : deriv f x = deriv f.derivative x

axiom differential_equation (x : ℝ) : x * (deriv (deriv f) x) + 2 * f(x) = 1 / (x^2)

axiom f_at_1 : f 1 = 1

theorem maximum_value_of_f :
   ∃ c, (∀ x, f x ≤ f c) ∧ f c = e / 2 :=
 sorry

end maximum_value_of_f_l462_462847


namespace bowling_playoff_orders_l462_462449

theorem bowling_playoff_orders :
  let num_bowlers := 6
  let choices_per_game := 2
  let total_choices := choices_per_game ^ (num_bowlers - 1)
  total_choices = 32 :=
by
  let num_bowlers := 6
  let choices_per_game := 2
  let total_choices := choices_per_game ^ (num_bowlers - 1)
  have : total_choices = 2 ^ 5 := rfl
  show 2 ^ 5 = 32, by decide

end bowling_playoff_orders_l462_462449


namespace greatest_divisor_arithmetic_sum_l462_462371

theorem greatest_divisor_arithmetic_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ d, d = 15 ∧ ∀ S : ℕ, S = 15 * x + 105 * c → d ∣ S :=
by 
  sorry

end greatest_divisor_arithmetic_sum_l462_462371


namespace cost_to_marked_price_percentage_l462_462267

variable (MP CP : ℝ)
variable (discount : ℝ := 0.15)
variable (gain_percent : ℝ := 54.54545454545454)

theorem cost_to_marked_price_percentage :
  (SP CP MP discount gain_percent: ℝ) : (CP / MP = 0.55) :=
by
  have gain_ratio : ℝ := 6 / 11
  have gain := CP * gain_ratio
  have SP1 := MP * (1 - discount)
  have SP2 := CP * (1 + gain_ratio)
  have h : SP1 = SP2 := by
    sorry
  sorry

end cost_to_marked_price_percentage_l462_462267


namespace find_vertex_X_l462_462591

def midpoint (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

-- Conditions
axiom YZ_midpoint : midpoint Y Z = (2, 3, 1)
axiom XZ_midpoint : midpoint X Z = (1, 2, 3)
axiom XY_midpoint : midpoint X Y = (3, 1, 4)

-- Theorem to be proved
theorem find_vertex_X (X Y Z : ℝ × ℝ × ℝ) : X = (2, 0, 6) :=
begin
  sorry  -- proof to be completed
end

end find_vertex_X_l462_462591


namespace measure_of_obtuse_angle_ADB_l462_462905

-- Define the right triangle ABC with specific angles
def triangle_ABC (A B C D : Type) [AddAngle A B C] :=
  (rightTriangle : (A B C) ∧ 
   angle_A_is_45_degrees : (angle A = 45) ∧ 
   angle_B_is_45_degrees : (angle B = 45) ∧ 
   AD_bisects_A : (bisects A D) ∧ 
   BD_bisects_B : (bisects B D))

-- Statement of the proof problem
theorem measure_of_obtuse_angle_ADB {A B C D : Type} [AddAngle A B C] 
  (h_ABC : triangle_ABC A B C) : 
  measure_obtuse_angle A B D ADB = 135 :=
sorry

end measure_of_obtuse_angle_ADB_l462_462905


namespace eval_a4_minus_a_neg4_l462_462080

theorem eval_a4_minus_a_neg4 (a : ℝ) (ha : a ≠ 0) : 
  a^4 - a^(-4) = (a - a^(-1))^2 * ((a - a^(-1))^2 + 2) := 
sorry

end eval_a4_minus_a_neg4_l462_462080


namespace geometric_series_first_two_terms_sum_l462_462181

theorem geometric_series_first_two_terms_sum (a n : ℕ) (h_pos : 0 < a) (h_sum : (a / (1 - 1 / n.to_rational)) = 3) (hn_pos : 0 < n) : 
    a + a * (1 / n.to_rational) = 8 / 3 := 
by 
    sorry

end geometric_series_first_two_terms_sum_l462_462181


namespace average_age_l462_462659

theorem average_age (avg_age_sixth_graders avg_age_parents : ℕ) 
    (num_sixth_graders num_parents : ℕ)
    (h1 : avg_age_sixth_graders = 12) 
    (h2 : avg_age_parents = 40) 
    (h3 : num_sixth_graders = 40) 
    (h4 : num_parents = 60) :
    (num_sixth_graders * avg_age_sixth_graders + num_parents * avg_age_parents) 
    / (num_sixth_graders + num_parents) = 28.8 := 
by
  sorry

end average_age_l462_462659


namespace batsman_average_l462_462982

theorem batsman_average
  (avg_20_matches : ℕ → ℕ → ℕ)
  (avg_10_matches : ℕ → ℕ → ℕ)
  (total_1st_20 : ℕ := avg_20_matches 20 30)
  (total_next_10 : ℕ := avg_10_matches 10 15) :
  (total_1st_20 + total_next_10) / 30 = 25 :=
by
  sorry

end batsman_average_l462_462982


namespace equal_roots_condition_l462_462262

theorem equal_roots_condition (m : ℝ) :
  (m = 2 ∨ m = (9 + Real.sqrt 57) / 8 ∨ m = (9 - Real.sqrt 57) / 8) →
  ∃ a b c : ℝ, 
  (∀ x : ℝ, (a * x ^ 2 + b * x + c = 0) ↔
  (x * (x - 3) - (m + 2)) / ((x - 3) * (m - 2)) = x / m) ∧
  (b^2 - 4 * a * c = 0) :=
sorry

end equal_roots_condition_l462_462262


namespace Ann_initial_money_l462_462043

variable (A : ℕ)

theorem Ann_initial_money (h : 1111 - 167 = 944) (h₁ : A + 167 = 944) : A = 777 := 
by
  have h₂ : 1111 - 167 = 944 := by exact h
  have h₃ : A + 167 = 944 := by exact h₁
  have h₄ : A = 944 - 167 := by sorry
  have h₅ : 944 - 167 = 777 := by sorry
  show A = 777 from by sorry

end Ann_initial_money_l462_462043


namespace inverse_proportion_point_passes_through_l462_462897

theorem inverse_proportion_point_passes_through
  (m : ℝ) (h1 : (4, 6) ∈ {p : ℝ × ℝ | p.snd = (m^2 + 2 * m - 1) / p.fst})
  : (-4, -6) ∈ {p : ℝ × ℝ | p.snd = (m^2 + 2 * m - 1) / p.fst} :=
sorry

end inverse_proportion_point_passes_through_l462_462897


namespace option_b_option_c_option_d_l462_462845

variable {a b c : ℝ}
variable (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)

-- Statement for Option B
theorem option_b : 
  0 < a → 0 < b → 0 < c → (1 / (a * c) + (a / (b^2 * c)) + (b * c) ≥ 2 * (sqrt 2)) := 
by
  intro a_pos b_pos c_pos
  sorry

-- Statement for Option C
theorem option_c : 
  0 < a → 0 < b → 0 < c → (a + b + c ≥ sqrt (2 * a * b) + sqrt (2 * a * c)) := 
by
  intro a_pos b_pos c_pos
  sorry

-- Statement for Option D
theorem option_d : 
  0 < a → 0 < b → 0 < c → (a^2 + b^2 + c^2 ≥ 2*a*b + 2*b*c - 2*a*c) := 
by
  intro a_pos b_pos c_pos
  sorry

end option_b_option_c_option_d_l462_462845


namespace measure_of_obtuse_angle_ADB_l462_462907

-- Define the right triangle ABC with specific angles
def triangle_ABC (A B C D : Type) [AddAngle A B C] :=
  (rightTriangle : (A B C) ∧ 
   angle_A_is_45_degrees : (angle A = 45) ∧ 
   angle_B_is_45_degrees : (angle B = 45) ∧ 
   AD_bisects_A : (bisects A D) ∧ 
   BD_bisects_B : (bisects B D))

-- Statement of the proof problem
theorem measure_of_obtuse_angle_ADB {A B C D : Type} [AddAngle A B C] 
  (h_ABC : triangle_ABC A B C) : 
  measure_obtuse_angle A B D ADB = 135 :=
sorry

end measure_of_obtuse_angle_ADB_l462_462907


namespace find_number_eq_36_l462_462393

theorem find_number_eq_36 (n : ℝ) (h : (n / 18) * (n / 72) = 1) : n = 36 :=
sorry

end find_number_eq_36_l462_462393


namespace tangent_line_mn_l462_462896

noncomputable def is_tangent (m n t : ℝ) : Prop :=
(∀ x : ℝ, (f : ℝ → ℝ), (f x = sqrt x) → deriv f = (λ x, 1 / (2 * sqrt x)) ∧
y = mx + n → y = (1 / (2 * sqrt t)) * x + (sqrt t / 2) ∧ x = t)

theorem tangent_line_mn
    (m n t : ℝ)
    (h_tangent : is_tangent m n t) :
    m * n = 1 / 4 :=
by
  sorry

end tangent_line_mn_l462_462896


namespace greatest_divisor_of_sum_of_arith_seq_l462_462313

theorem greatest_divisor_of_sum_of_arith_seq (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → d ∣ (15 * (x + 7 * c))) ∧
    (∀ k : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → k ∣ (15 * (x + 7 * c))) → k ≤ d) ∧ 
    d = 15 :=
sorry

end greatest_divisor_of_sum_of_arith_seq_l462_462313


namespace parabola_distance_focus_l462_462563

theorem parabola_distance_focus (x y : ℝ) (h1 : y^2 = 4 * x) (h2 : (x - 1)^2 + y^2 = 16) : x = 3 := by
  sorry

end parabola_distance_focus_l462_462563


namespace arrange_cubes_bound_l462_462306

def num_ways_to_arrange_cubes_into_solids (n : ℕ) : ℕ := sorry

theorem arrange_cubes_bound (n : ℕ) (h : n = (2015^100)) :
  10^14 < num_ways_to_arrange_cubes_into_solids n ∧
  num_ways_to_arrange_cubes_into_solids n < 10^15 := sorry

end arrange_cubes_bound_l462_462306


namespace segment_length_F_F_l462_462684

def point_reflect_y_axis (P : ℝ × ℝ) : ℝ × ℝ := (-P.1, P.2)

theorem segment_length_F_F' :
  let F := (-4, 5)
  let F' := point_reflect_y_axis F
  F' = (4, 5) ∧ (∥F'.1 - F.1∥ = 8) :=
by 
  let F := (-4, 5)
  let F' := point_reflect_y_axis F
  have hF' : F' = (4, 5) := rfl
  split 
  · exact hF'
  · dsimp [F, F']
    norm_num
  sorry

end segment_length_F_F_l462_462684


namespace abs_value_condition_l462_462176

theorem abs_value_condition (m : ℝ) (h : |m - 1| = m - 1) : m ≥ 1 :=
by {
  sorry
}

end abs_value_condition_l462_462176


namespace hexagon_distortion_convex_l462_462218

noncomputable def is_convex_distortion (H : Hexagon) (x : ℝ) : Prop :=
  ∀ (dH : Hexagon), 
    (∀ v ∈ vertices H, dist v (vertices dH v) < 1) → 
    is_convex dH

theorem hexagon_distortion_convex (H : Hexagon) (x : ℝ) (h : H.side_length = x) :
  x ≥ 2 → is_convex_distortion H x :=
by
  sorry

end hexagon_distortion_convex_l462_462218


namespace coconut_grove_nut_yield_l462_462570

theorem coconut_grove_nut_yield (x : ℕ) (Y : ℕ) 
  (h1 : (x + 4) * 60 + x * 120 + (x - 4) * Y = 3 * x * 100)
  (h2 : x = 8) : Y = 180 := 
by
  sorry

end coconut_grove_nut_yield_l462_462570


namespace sqrt_expression_simplified_l462_462766

-- Define the conditions and the expected answer
def simplExp (y : ℝ) : ℝ :=
  √(4 + ( (y^6 - 4) / (3 * y^3) )^2)

def expected (y : ℝ) : ℝ :=
  ( √(y^12 + 28 * y^6 + 16) ) / (3 * y^3)

theorem sqrt_expression_simplified (y : ℝ) :
  simplExp y = expected y :=
by
  -- Proof omitted
  sorry

end sqrt_expression_simplified_l462_462766


namespace min_f_l462_462843

open Real

noncomputable def f (θ a b n : ℝ) : ℝ := (a / (sin θ) ^ n) + (b / (cos θ) ^ n)

variable (a b n : ℝ) (ha : 0 < a) (hb : 0 < b) (hn : nat.prime n)

theorem min_f (hθ1 : 0 < θ) (hθ2 : θ < π / 2) :
  ∃ (θ : ℝ), ∀ θ θa hθb hθn, 0 < θ ∧ θ < π / 2 ∧ prime n -> 
  f θ a b n = (a ^ (2 / (n + 2)) + b ^ (2 / (n + 2))) ^ ((n + 2) / 2) :=
begin 
  sorry
end

end min_f_l462_462843


namespace real_part_of_z_l462_462509

-- Define the given condition and required proof statement
theorem real_part_of_z {z : ℂ} (h : (1 - (complex.I : ℂ)) * conj z = complex.abs (1 + complex.I)) : 
  complex.re z = real.sqrt 2 / 2 :=
sorry

end real_part_of_z_l462_462509


namespace parallelogram_area_ratio_l462_462247

def Parallelogram (A B C D : Type) := -- Definition placeholder for parallelogram
  sorry

def Midpoint (E : Type) (A B : Type) := -- Definition placeholder for midpoint
  sorry

noncomputable def Area_ratio (A B C D E F : Type) (S1 S2 : ℕ) [Parallelogram A B C D] [Midpoint E A B] [Midpoint F B C] : Prop :=
  S1 = 2 * S2

theorem parallelogram_area_ratio (A B C D E F : Type) [Parallelogram A B C D] [Midpoint E A B] [Midpoint F B C] (S1 S2 : ℕ) : 
  Area_ratio A B C D E F S1 S2 := by
  sorry

end parallelogram_area_ratio_l462_462247


namespace corrected_mean_l462_462671

theorem corrected_mean (
  mean : ℝ,
  n : ℕ,
  mean_is_200_3 : mean = 200.3,
  n_is_500 : n = 500,
  incorrect_vals : List ℝ,
  correct_vals : List ℝ,
  incorrect_vals_is_correct : incorrect_vals = [95.3, -15.6, 405.5, 270.7, 300.5],
  correct_vals_is_correct : correct_vals = [57.8, -28.9, 450, 250, 100.1]
) : let original_sum := mean * n,
       errors := List.zipWith (-) correct_vals incorrect_vals, 
       total_error := List.sum errors,
       corrected_sum := original_sum - total_error,
       corrected_mean := corrected_sum / n
  in corrected_mean = 199.845 := by
  sorry

end corrected_mean_l462_462671


namespace obtuse_angle_A_DB_of_bisectors_l462_462910

theorem obtuse_angle_A_DB_of_bisectors
  (ABC: Triangle)
  (A B C D : Point)
  (h_right_triangle : is_right_triangle ABC)
  (h_angle_A_45 : angle A = 45)
  (h_angle_B_45 : angle B = 45)
  (h_bisectors_intersect_D : intersects_angle_bisectors A B D) :
  angle ADB = 135 :=
by sorry

end obtuse_angle_A_DB_of_bisectors_l462_462910


namespace colton_sticker_problem_l462_462061

theorem colton_sticker_problem : 
  let initial_stickers := 85
      friends_given := 5 * 4
      mandy_given := friends_given + 5
      total_given_initial := friends_given + mandy_given
      remaining_after_initial := initial_stickers - total_given_initial
      justin_given := 0.20 * remaining_after_initial
      remaining_after_justin := remaining_after_initial - justin_given
      karen_given := remaining_after_justin / 4
      final_remaining := remaining_after_justin - karen_given
  in final_remaining = 24 :=
by sorry

end colton_sticker_problem_l462_462061


namespace pizza_volume_piece_l462_462428

theorem pizza_volume_piece (h : ℝ) (d : ℝ) (n : ℝ) (V_piece : ℝ) 
  (h_eq : h = 1 / 2) (d_eq : d = 16) (n_eq : n = 8) : 
  V_piece = 4 * Real.pi :=
by
  sorry

end pizza_volume_piece_l462_462428


namespace solution_range_a_l462_462100

-- Define the set of valid x values
def valid_x (x : ℝ) : Prop := x ∈ set.Icc (-real.pi / 6) (real.pi / 2)

-- Define the function f(x)
noncomputable def f (x a : ℝ) : ℝ := (real.sin x)^2 + a * real.sin x + a + 3

-- State the theorem
theorem solution_range_a (a : ℝ) (hx : ∀ x, valid_x x → 0 ≤ f x a) :
  ∀ x, valid_x x → 0 ≤ f x a := sorry

end solution_range_a_l462_462100


namespace conjugate_of_complex_number_l462_462858

theorem conjugate_of_complex_number (z : ℂ) (h : z = 2 * complex.I / (1 + complex.I)) :
  complex.conj z = 1 - complex.I :=
by sorry

end conjugate_of_complex_number_l462_462858


namespace bert_runs_per_day_l462_462046

theorem bert_runs_per_day (total_miles : ℕ) (days_per_week : ℕ) (weeks : ℕ) (daily_miles : ℕ) :
  total_miles = 42 →
  days_per_week = 7 →
  weeks = 3 →
  daily_miles = total_miles / (days_per_week * weeks) →
  daily_miles = 2 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3]
  simp at h4
  exact h4

end bert_runs_per_day_l462_462046


namespace smallest_M_convex_quadrilateral_l462_462805

section ConvexQuadrilateral

-- Let a, b, c, d be the sides of a convex quadrilateral
variables {a b c d M : ℝ}

-- Condition to ensure that a, b, c, d are the sides of a convex quadrilateral
def is_convex_quadrilateral (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a + b + c + d < 360

-- The theorem statement
theorem smallest_M_convex_quadrilateral (hconvex : is_convex_quadrilateral a b c d) : ∃ M, (∀ a b c d, is_convex_quadrilateral a b c d → (a^2 + b^2) / (c^2 + d^2) > M) ∧ M = 1/2 :=
by sorry

end ConvexQuadrilateral

end smallest_M_convex_quadrilateral_l462_462805


namespace polygon_sides_eq_six_l462_462292

theorem polygon_sides_eq_six (n : ℕ) (h1 : (n - 2) * 180 = 2 * 360) : n = 6 :=
by
  sorry

end polygon_sides_eq_six_l462_462292


namespace parakeets_in_each_cage_l462_462725

variable (num_cages : ℕ) (parrots_per_cage : ℕ) (total_birds : ℕ)

-- Given conditions
def total_parrots (num_cages parrots_per_cage : ℕ) : ℕ := num_cages * parrots_per_cage
def total_parakeets (total_birds total_parrots : ℕ) : ℕ := total_birds - total_parrots
def parakeets_per_cage (total_parakeets num_cages : ℕ) : ℕ := total_parakeets / num_cages

-- Theorem: Number of parakeets in each cage is 7
theorem parakeets_in_each_cage (h1 : num_cages = 8) (h2 : parrots_per_cage = 2) (h3 : total_birds = 72) : 
  parakeets_per_cage (total_parakeets total_birds (total_parrots num_cages parrots_per_cage)) num_cages = 7 :=
by
  sorry

end parakeets_in_each_cage_l462_462725


namespace max_reached_at_2001_l462_462505

noncomputable def a (n : ℕ) : ℝ := n^2 / 1.001^n

theorem max_reached_at_2001 : ∀ n : ℕ, a 2001 ≥ a n := 
sorry

end max_reached_at_2001_l462_462505


namespace pq_product_of_quadratic_eq_l462_462894

theorem pq_product_of_quadratic_eq (p q : ℝ) (h : ∀ (x : ℂ), x^2 + p * x + q = 0 → (x = 3 - 4 * complex.I) ∨ (x = 3 + 4 * complex.I)) : 
  p * q = -150 :=
sorry

end pq_product_of_quadratic_eq_l462_462894


namespace sequence_product_sum_l462_462567

-- Sequence definition
noncomputable def a (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 2^(n-1)

-- Sum of the first n terms Sn
noncomputable def S_n (n : ℕ) : ℕ := 2^n - 1

-- Term a_n a_{n+1}
noncomputable def product (n : ℕ) : ℕ := a n * a (n+1)

-- The statement to prove
theorem sequence_product_sum (n : ℕ) : ∑ i in Finset.range n, product i = (2 * 4^n - 2) / 3 := 
sorry

end sequence_product_sum_l462_462567


namespace determinant_matrix_A_l462_462785

open LinearAlgebra

variables (y : ℝ)

def matrix_A : Matrix (Fin 3) (Fin 3) ℝ :=
  !![ 2 * y + 1, 2 * y, 2 * y,
      2 * y, 2 * y + 1, 2 * y,
      2 * y, 2 * y, 2 * y + 1]

theorem determinant_matrix_A : matrix.det (matrix_A y) = 6 * y + 1 :=
by sorry

end determinant_matrix_A_l462_462785


namespace integral_abs_x_plus_1_eq_half_integral_piecewise_function_eq_five_sixths_l462_462754

noncomputable def integral_abs_x_plus_1 : ℝ :=
  ∫ x in -3..2, |x+1|

theorem integral_abs_x_plus_1_eq_half :
  integral_abs_x_plus_1 = 13 / 2 :=
sorry

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then x^2 else 2 - x

noncomputable def integral_piecewise_function : ℝ :=
  ∫ x in 0..2, f x

theorem integral_piecewise_function_eq_five_sixths :
  integral_piecewise_function = 5 / 6 :=
sorry

end integral_abs_x_plus_1_eq_half_integral_piecewise_function_eq_five_sixths_l462_462754


namespace closest_points_to_A_l462_462959

noncomputable def distance_squared (x y : ℝ) : ℝ :=
  x^2 + (y + 3)^2

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 = 9

theorem closest_points_to_A :
  ∃ (x y : ℝ),
    hyperbola x y ∧
    (distance_squared x y = distance_squared (-3 * Real.sqrt 5 / 2) (-3/2) ∨
     distance_squared x y = distance_squared (3 * Real.sqrt 5 / 2) (-3/2)) :=
sorry

end closest_points_to_A_l462_462959


namespace digit_possibilities_757_l462_462408

theorem digit_possibilities_757
  (N : ℕ)
  (h : N < 10) :
  (∃ d₀ d₁ d₂ : ℕ, (d₀ = 2 ∨ d₀ = 5 ∨ d₀ = 8) ∧
  (d₁ = 2 ∨ d₁ = 5 ∨ d₁ = 8) ∧
  (d₂ = 2 ∨ d₂ = 5 ∨ d₂ = 8) ∧
  (d₀ ≠ d₁) ∧
  (d₀ ≠ d₂) ∧
  (d₁ ≠ d₂)) :=
by
  sorry

end digit_possibilities_757_l462_462408


namespace geometric_sequence_formula_l462_462848

variable {a : ℚ}

theorem geometric_sequence_formula (h : (a + 1)^2 = (a - 1) * (a + 4)) :
  ∃ f : ℕ → ℚ, f(n) = 4 * (3 / 2)^(n - 1) :=
by sorry

end geometric_sequence_formula_l462_462848


namespace range_of_m_l462_462533

theorem range_of_m 
  (f : ℝ → ℝ)
  (hf : ∀ x y : ℝ, x < y → f(x) < f(y))
  (m : ℝ)
  (h : f(m^2) > f(-m)) :
  m < -1 ∨ m > 0 :=
begin
  sorry
end

end range_of_m_l462_462533


namespace bryan_samples_l462_462047

noncomputable def initial_samples_per_shelf : ℕ := 128
noncomputable def shelves : ℕ := 13
noncomputable def samples_removed_per_shelf : ℕ := 2
noncomputable def remaining_samples_per_shelf := initial_samples_per_shelf - samples_removed_per_shelf
noncomputable def total_remaining_samples := remaining_samples_per_shelf * shelves

theorem bryan_samples : total_remaining_samples = 1638 := 
by 
  sorry

end bryan_samples_l462_462047


namespace student_a_score_l462_462437

def total_questions : ℕ := 100
def correct_responses : ℕ := 87
def incorrect_responses : ℕ := total_questions - correct_responses
def score : ℕ := correct_responses - 2 * incorrect_responses

theorem student_a_score : score = 61 := by
  unfold score
  unfold correct_responses
  unfold incorrect_responses
  norm_num
  -- At this point, the theorem is stated, but we insert sorry to satisfy the requirement of not providing the proof.
  sorry

end student_a_score_l462_462437


namespace number_of_factors_of_n_l462_462941

noncomputable def n : ℕ := 2^3 * 3^6 * 5^7 * 7^8

theorem number_of_factors_of_n :
  nat.factors_count n = 2016 :=
sorry

end number_of_factors_of_n_l462_462941


namespace different_graphs_l462_462782

theorem different_graphs :
  ∀ (x y : ℝ), (y = 2 * x - 3) ↔ (y = (x - 3)) ↔ ((x = -3) ∨ y = x - 3) ↔
  x = -3 ∧ (∃ y : ℝ, ((x + 3) * y = x^2 - 9) → y = (x - 3)) ↔ false :=
sorry

end different_graphs_l462_462782


namespace trigonometric_identity_l462_462808

variable {α : ℝ}

theorem trigonometric_identity :
  (2 * (sin α)^2 / sin (2 * α)) * (2 * (cos α)^2 / cos (2 * α)) = tan (2 * α) := by
  sorry

end trigonometric_identity_l462_462808


namespace complex_magnitude_sum_l462_462489

theorem complex_magnitude_sum : |(3 - 5 * complex.I)| + |(3 + 5 * complex.I)| = 2 * real.sqrt 34 :=
by
  sorry

end complex_magnitude_sum_l462_462489


namespace unique_diagonal_numbers_l462_462692

-- Define the 101 x 101 matrix
def M : Matrix (Fin 101) (Fin 101) (Fin 101 → ℕ) := sorry

theorem unique_diagonal_numbers (M : Matrix (Fin 101) (Fin 101) ℕ)
  (h1 : ∀ i : Fin 101, ∀ j₁ j₂ : Fin 101, (M i j₁ = M i j₂) → j₁ = j₂)
  (h2 : ∀ i j : Fin 101, M i j = M j i) :
  ∀ i₁ i₂ : Fin 101, i₁ ≠ i₂ → M i₁ i₁ ≠ M i₂ i₂ :=
begin
  sorry
end

end unique_diagonal_numbers_l462_462692


namespace problem1_problem2_l462_462135

section ProofProblems

variables {a b : ℝ}

-- Given that a and b are distinct positive numbers
axiom a_pos : a > 0
axiom b_pos : b > 0
axiom a_neq_b : a ≠ b

-- Problem (i): Prove that a^4 + b^4 > a^3 * b + a * b^3
theorem problem1 : a^4 + b^4 > a^3 * b + a * b^3 :=
by {
  sorry
}

-- Problem (ii): Prove that a^5 + b^5 > a^3 * b^2 + a^2 * b^3
theorem problem2 : a^5 + b^5 > a^3 * b^2 + a^2 * b^3 :=
by {
  sorry
}

end ProofProblems

end problem1_problem2_l462_462135


namespace positive_difference_of_solutions_l462_462281

theorem positive_difference_of_solutions :
  let a := 1, b := -4, c := -32
  ∃ x1 x2 : ℝ, (x^2 - 3*x + 9 = x + 41) ∧
              (x1 = 8 ∨ x1 = -4) ∧
              (x2 = 8 ∨ x2 = -4) ∧
              (x1 ≠ x2) ∧
              (abs(x1 - x2) = 12) :=
by
  sorry

end positive_difference_of_solutions_l462_462281


namespace A_values_A_eigenvalues_l462_462541

noncomputable def A (a : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := 
  ![![3, 0], ![2, a]]

noncomputable def A_inv (b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1 / 3, 0], ![b, 1]]

theorem A_values :
  ∃ (a b : ℝ), 
    A a * A_inv b = 1 ∧
    a = 1 ∧
    b = -2 / 3 :=
by
  sorry

theorem A_eigenvalues (a b : ℝ) (hA : A a * A_inv b = 1)
  (ha : a = 1) (hb : b = -2 / 3) :
    let A_val := A 1
    let char_poly := A_val.charPoly
    (char_poly.roots = ![1, 3]) :=
by
  sorry

end A_values_A_eigenvalues_l462_462541


namespace maximize_car_sales_l462_462001

theorem maximize_car_sales :
  ∃ x : ℕ, 0 < x ∧ x ≤ 80 ∧
  (∀ y : ℕ, (y = 10 * 1000 * (1 - 0.5 * x)) → y ≤ 11250) ∧
  (10 * 1000 * (1 - 0.5 * 50) = 11250) :=
by
  sorry

end maximize_car_sales_l462_462001


namespace square_dissection_l462_462471

theorem square_dissection (m : ℤ) (r1 r2 r3 r4 r5 : ℤ × ℤ)
  (h_side_lengths : {r1.fst, r1.snd, r2.fst, r2.snd, r3.fst, r3.snd, r4.fst, r4.snd, r5.fst, r5.snd} = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) :
  m = 11 ∨ m = 13 :=
by
  sorry

end square_dissection_l462_462471


namespace trigonometric_inequality_l462_462944

open Real

theorem trigonometric_inequality
  (x y z : ℝ)
  (h1 : 0 < x)
  (h2 : x < y)
  (h3 : y < z)
  (h4 : z < π / 2) :
  π / 2 + 2 * sin x * cos y + 2 * sin y * cos z > sin (2 * x) + sin (2 * y) + sin (2 * z) :=
by
  sorry

end trigonometric_inequality_l462_462944


namespace third_island_not_maya_l462_462627

def is_knight (P : Prop) : Prop := P
def is_knave (P : Prop) : Prop := ¬ P

-- Define the inhabitants' statements
def A_statement (A_liar B_liar : Prop) : Prop :=
  (A_liar ∨ B_liar) ∧ (∀ A_knight ∀ B_knight, this_island_called_Maya → A_knight → ¬A_knight)

def B_statement {A_statement : Prop} : Prop := A_statement

-- Hypotheses
variables (A_liar B_liar : Prop) (this_island_called_Maya : Prop)

-- The problem
theorem third_island_not_maya : A_statement A_liar B_liar = B_statement → ¬ this_island_called_Maya :=
sorry

end third_island_not_maya_l462_462627


namespace area_triangle_QDB_l462_462774

theorem area_triangle_QDB (q : ℝ) : 
  let Q := (0, q), D := (3, q), B := (12, 0) in
  let area : ℝ := (1 / 2) * (3 * q) in
  area = 3 * q / 2 := 
by 
  sorry

end area_triangle_QDB_l462_462774


namespace petri_dish_count_l462_462585

theorem petri_dish_count (total_germs : ℝ) (germs_per_dish : ℝ) (h1 : total_germs = 0.036 * 10^5) (h2 : germs_per_dish = 199.99999999999997) :
  total_germs / germs_per_dish = 18 :=
by
  sorry

end petri_dish_count_l462_462585


namespace number_of_possible_values_of_abs_z_l462_462175

theorem number_of_possible_values_of_abs_z (z : ℂ) (h : z^2 - 10 * z + 52 = 0): 
  ∃! r : ℝ, |z| = r :=
sorry

end number_of_possible_values_of_abs_z_l462_462175


namespace correct_statements_l462_462038

theorem correct_statements :
  (∀ (f : ℝ → ℝ), Monotone f → ∃! (x : ℝ), f x = 0) ∧
  (∀ (f : Polynomial ℝ), degree f = 2 → ∃ (x y : ℝ), f x = 0 ∧ f y = 0 → x = y) ∧
  (∀ (a : ℝ), a > 0 ∧ a ≠ 1 → ∀ (x : ℝ), a^x ≠ 0) ∧
  (∀ (a : ℝ), a > 0 → ∀ (x : ℝ), a ≠ 1 → log x = 0 → x = 1) ∧
  ((∃ (f : ℝ → ℝ), (∃ (x : ℝ), f x = 0)) ∧ (∃ (f : ℝ → ℝ), ¬∃ (x : ℝ), f x = 0)) :=
by
sorry

end correct_statements_l462_462038


namespace victor_earnings_l462_462396

variable (wage hours_mon hours_tue : ℕ)

def hourly_wage : ℕ := 6
def hours_worked_monday : ℕ := 5
def hours_worked_tuesday : ℕ := 5

theorem victor_earnings :
  (hours_worked_monday + hours_worked_tuesday) * hourly_wage = 60 :=
by
  sorry

end victor_earnings_l462_462396


namespace units_digit_of_sum_l462_462703

-- Definitions of the sequence terms
def a1 := 1! + 3
def a2 := 2! + 3
def a3 := 3! + 3
def a4 := 4! + 3
def a5 := 5! + 3
def a6 := 6! + 3

-- Sum of the terms in the sequence
def sum_seq := a1 + a2 + a3 + a4 + a5 + a6

-- The proof statement
theorem units_digit_of_sum : (sum_seq % 10) = 1 := 
by {
   sorry
}

end units_digit_of_sum_l462_462703


namespace count_positive_integers_satisfying_inequality_l462_462885

theorem count_positive_integers_satisfying_inequality :
  {n : ℕ | (n + 6) * (n - 5) * (n - 10) * (n - 15) < 0}.finite.card = 8 := by
  sorry

end count_positive_integers_satisfying_inequality_l462_462885


namespace greatest_divisor_of_sum_of_arith_seq_l462_462315

theorem greatest_divisor_of_sum_of_arith_seq (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → d ∣ (15 * (x + 7 * c))) ∧
    (∀ k : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → k ∣ (15 * (x + 7 * c))) → k ≤ d) ∧ 
    d = 15 :=
sorry

end greatest_divisor_of_sum_of_arith_seq_l462_462315


namespace calc_expression_l462_462757

theorem calc_expression : 
  3 * Real.tan (Real.pi / 6) - Real.sqrt 9 + (1 / 3)⁻¹ = Real.sqrt 3 :=
by
  have h1 : Real.tan (Real.pi / 6) = Real.sqrt 3 / 3 := Real.tan_pi_div_six
  have h2 : Real.sqrt 9 = 3 := Real.sqrt_sq 3 (by norm_num)
  have h3 : (1 / 3)⁻¹ = 3 := one_div_inv 3
  rw [h1, h2, h3]
  norm_num
  ring
  sorry

end calc_expression_l462_462757


namespace determine_F_l462_462617

theorem determine_F (n : ℕ) (n_pos : 0 < n) (F : set (set (fin n)))
  (h : ∀ (X : set (fin n)), X ≠ ∅ → (∑ A in F, (if fin.card (A ∩ X) % 2 = 0 then 1 else 0)) = 
  (∑ A in F, (if fin.card (A ∩ X) % 2 = 1 then 1 else 0))) :
  F = ∅ ∨ F = set.univ :=
begin
  sorry
end

end determine_F_l462_462617


namespace perpendicular_lines_b_value_l462_462990

theorem perpendicular_lines_b_value : 
  (∀ b : ℝ, let slope1 := -2 / 3 in
            let slope2 := -b / 3 in
            slope1 * slope2 = -1 → b = -9 / 2) :=
begin
  sorry
end

end perpendicular_lines_b_value_l462_462990


namespace range_independent_variable_l462_462199

noncomputable def range_of_independent_variable (x : ℝ) : Prop :=
  x ≠ 3

theorem range_independent_variable (x : ℝ) :
  (∃ y : ℝ, y = 1 / (x - 3)) → x ≠ 3 :=
by
  intro h
  sorry

end range_independent_variable_l462_462199


namespace norm_sum_eq_l462_462494

-- Define the complex numbers
def c1 : ℂ := 3 - 5 * complex.I
def c2 : ℂ := 3 + 5 * complex.I

-- Define the norm (magnitude) of complex numbers
def norm_c1 : ℝ := complex.abs c1
def norm_c2 : ℝ := complex.abs c2

-- The statement to prove
theorem norm_sum_eq : norm_c1 + norm_c2 = 2 * real.sqrt 34 :=
by sorry

end norm_sum_eq_l462_462494


namespace cylindrical_to_rectangular_l462_462067

theorem cylindrical_to_rectangular :
  ∀ (r θ z : ℝ), r = 5 → θ = (3 * Real.pi) / 4 → z = 2 →
    (r * Real.cos θ, r * Real.sin θ, z) = (-5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2, 2) :=
by
  intros r θ z hr hθ hz
  rw [hr, hθ, hz]
  -- Proof steps would go here, but are omitted as they are not required.
  sorry

end cylindrical_to_rectangular_l462_462067


namespace pies_from_apples_l462_462266

theorem pies_from_apples (total_apples : ℕ) (percent_handout : ℝ) (apples_per_pie : ℕ) 
  (h_total : total_apples = 800) (h_percent : percent_handout = 0.65) (h_per_pie : apples_per_pie = 15) : 
  (total_apples * (1 - percent_handout)) / apples_per_pie = 18 := 
by 
  sorry

end pies_from_apples_l462_462266


namespace find_radius_and_diameter_l462_462003

theorem find_radius_and_diameter (M N r d : ℝ) (h1 : M = π * r^2) (h2 : N = 2 * π * r) (h3 : M / N = 15) : 
  (r = 30) ∧ (d = 60) := by
  sorry

end find_radius_and_diameter_l462_462003


namespace part1_values_correct_estimated_students_correct_l462_462002

def students_data : List ℕ :=
  [30, 60, 70, 10, 30, 115, 70, 60, 75, 90, 15, 70, 40, 75, 105, 80, 60, 30, 70, 45]

def total_students := 200

def categorized_counts := (2, 5, 10, 3) -- (0 ≤ t < 30, 30 ≤ t < 60, 60 ≤ t < 90, 90 ≤ t < 120)

def mean := 60

def median := 65

def mode := 70

theorem part1_values_correct :
  let a := 5
  let b := 3
  let c := 65
  let d := 70
  categorized_counts = (2, a, 10, b) ∧ mean = 60 ∧ median = c ∧ mode = d := by {
  -- Proof will be provided here
  sorry
}

theorem estimated_students_correct :
  let at_least_avg := 130
  at_least_avg = (total_students * 13 / 20) := by {
  -- Proof will be provided here
  sorry
}

end part1_values_correct_estimated_students_correct_l462_462002


namespace part1_A_union_B_when_a_eq_2_l462_462841

variable {a : ℝ}

def A : Set ℝ := {x | (x - 1) / (x - 2) ≤ 1 / 2}

def B (a : ℝ) : Set ℝ := {x | x^2 - (a + 2) * x + 2 * a ≤ 0}

def A_Union_B_when_a_eq_2 : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem part1_A_union_B_when_a_eq_2 : A ∪ B 2 = A_Union_B_when_a_eq_2 :=
  sorry

end part1_A_union_B_when_a_eq_2_l462_462841


namespace intersection_sets_subset_sets_l462_462842

-- Part 1: Intersection of sets A and B when m = 5
theorem intersection_sets (m : ℝ) (A B : Set ℝ)
  (hA : A = {x | m + 1 ≤ x ∧ x ≤ 3 * m - 1})
  (hB : B = {x | 1 ≤ x ∧ x ≤ 10}) :
  (m = 5 → A ∩ B = {x | 6 ≤ x ∧ x ≤ 10}) :=
begin
  sorry
end

-- Part 2: Range of m for A ⊆ B
theorem subset_sets (m : ℝ) (A B : Set ℝ)
  (hA : A = {x | m + 1 ≤ x ∧ x ≤ 3 * m - 1})
  (hB : B = {x | 1 ≤ x ∧ x ≤ 10}) :
  (A ⊆ B ↔ m ∈ Set.Iic (11 / 3)) :=
begin
  sorry
end

end intersection_sets_subset_sets_l462_462842


namespace binomial_coefficient_ratio_l462_462985

theorem binomial_coefficient_ratio (n k : ℕ) (h₁ : n = 4 * k + 3) (h₂ : n = 3 * k + 5) : n + k = 13 :=
by
  sorry

end binomial_coefficient_ratio_l462_462985


namespace greatest_divisor_arithmetic_sum_l462_462367

theorem greatest_divisor_arithmetic_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ d, d = 15 ∧ ∀ S : ℕ, S = 15 * x + 105 * c → d ∣ S :=
by 
  sorry

end greatest_divisor_arithmetic_sum_l462_462367


namespace pizza_volume_piece_l462_462427

theorem pizza_volume_piece (h : ℝ) (d : ℝ) (n : ℝ) (V_piece : ℝ) 
  (h_eq : h = 1 / 2) (d_eq : d = 16) (n_eq : n = 8) : 
  V_piece = 4 * Real.pi :=
by
  sorry

end pizza_volume_piece_l462_462427


namespace count_scenarios_one_topic_not_chosen_l462_462188

theorem count_scenarios_one_topic_not_chosen
  (finalists : ℕ) (topics : ℕ) 
  (condition1 : finalists = 4) 
  (condition2 : topics = 4) : 
  number_of_scenarios_with_exactly_one_topic_not_chosen finalists topics = 144 := 
by 
  -- Definitions from conditions
  -- proving the expected number
  sorry

end count_scenarios_one_topic_not_chosen_l462_462188


namespace count_real_solutions_l462_462551
open Real

theorem count_real_solutions (h : ∀ x : ℝ, 2^(2*x + 1) - 4 * 2^(x + 1) - 2^x + 8 = 0) : 
  ∃ x₁ x₂ : ℝ, (x₁ ≠ x₂) ∧ ∀ y : ℝ, (2^(2*y + 1) - 4 * 2^(y + 1) - 2^y + 8 = 0) → (y = x₁ ∨ y = x₂) :=
sorry

end count_real_solutions_l462_462551


namespace log_base_change_l462_462554

theorem log_base_change
  (r s : ℝ)
  (hr : log 5 2 = r)
  (hs : log 2 7 = s) :
  log 10 7 = (s * r) / (r + 1) :=
sorry

end log_base_change_l462_462554


namespace min_norm_OA_tOB_l462_462132

variables {V : Type*} [InnerProductSpace ℝ V] (O A B C : V)
variables (t : ℝ)

-- Assuming ||OA|| = ||OB|| = 2
def norm_eq_two (v : V) := ∥v∥ = 2
def OA := (A - O)
def OB := (B - O)
axiom hOA : norm_eq_two OA
axiom hOB : norm_eq_two OB

-- Assuming Point C is on the line segment AB
axiom C_on_line_AB : ∃ (α : ℝ), 0 ≤ α ∧ α ≤ 1 ∧ C = α • A + (1 - α) • B

-- Assuming the minimum value of ||OC|| is 1
def OC := (C - O)
axiom hOC_min : ∥OC∥ = 1

-- Our goal is to find the minimum value of ||OA - tOB||
theorem min_norm_OA_tOB : ∃ (m : ℝ), m = sqrt 3 ∧ (∀ t : ℝ, ∥OA - t • OB∥ ≥ m) :=
sorry

end min_norm_OA_tOB_l462_462132


namespace downstream_speed_l462_462013

def Vm : ℝ := 31  -- speed in still water
def Vu : ℝ := 25  -- speed upstream
def Vs := Vm - Vu  -- speed of stream

theorem downstream_speed : Vm + Vs = 37 := 
by
  sorry

end downstream_speed_l462_462013


namespace divide_non_convex_pentagon_l462_462208

-- Definitions based on the conditions of the problem.

structure Pentagon :=
  (vertices : Fin 5 → ℝ × ℝ)
  (non_convex : ∃ i: Fin 5, interior_angle vertices i > 180)

-- The statement we want to prove.
theorem divide_non_convex_pentagon (P : Pentagon) : 
  ∃ P1 P2 : Pentagon, area P1 = area P2 ∧ non_convex P1 ∧ non_convex P2 :=
sorry

end divide_non_convex_pentagon_l462_462208


namespace greatest_divisor_of_sum_of_arithmetic_sequence_l462_462351

theorem greatest_divisor_of_sum_of_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ k : ℕ, (sum (λ n, x + n * c) (range 15)) = 15 * k :=
by sorry

end greatest_divisor_of_sum_of_arithmetic_sequence_l462_462351


namespace max_necklaces_is_5_l462_462402

def green_beads_per_necklace := 9
def white_beads_per_necklace := 6
def orange_beads_per_necklace := 3

def total_green_beads := 45
def total_white_beads := 45
def total_orange_beads := 45

def max_necklaces := Nat.min ((total_green_beads / green_beads_per_necklace), 
                           Nat.min ((total_white_beads / white_beads_per_necklace), 
                                    (total_orange_beads / orange_beads_per_necklace)))

theorem max_necklaces_is_5 :
  max_necklaces = 5 :=
sorry

end max_necklaces_is_5_l462_462402


namespace sec_minus_tan_l462_462555

theorem sec_minus_tan (x : ℝ) (h : (sec x + tan x) = 5 / 3) : (sec x - tan x) = 3 / 5 :=
by
  sorry

end sec_minus_tan_l462_462555


namespace shoes_ratio_l462_462970

theorem shoes_ratio (Scott_shoes : ℕ) (m : ℕ) (h1 : Scott_shoes = 7)
  (h2 : ∀ Anthony_shoes, Anthony_shoes = m * Scott_shoes)
  (h3 : ∀ Jim_shoes, Jim_shoes = Anthony_shoes - 2)
  (h4 : ∀ Anthony_shoes Jim_shoes, Anthony_shoes = Jim_shoes + 2) : 
  ∃ m : ℕ, (Anthony_shoes / Scott_shoes) = m := 
by 
  sorry

end shoes_ratio_l462_462970


namespace wrapping_paper_area_l462_462406

theorem wrapping_paper_area (l w h : ℝ) (hlw : l > w) (hwh : w > h) (hl : l = 2 * w) : 
    (∃ a : ℝ, a = 5 * w^2 + h^2) :=
by 
  sorry

end wrapping_paper_area_l462_462406


namespace foci_hyperbola_l462_462155

theorem foci_hyperbola
  (m : ℝ) 
  (h1 : ∀ x y : ℝ, (y = (sqrt 3 / 2) * x) ∨ (y = -(sqrt 3 / 2) * x))
  (h2 : ∀ (x y : ℝ), (x^2 / 4) - (y^2 / m) = 1) :
  ∃ (c : ℝ), (c = sqrt 7) ∧ (foci : {p : ℝ × ℝ // p = (c, 0) ∨ p = (-c, 0)}) :=
sorry

end foci_hyperbola_l462_462155


namespace solve_problem_l462_462928

noncomputable def problem_statement : Prop :=
  ∃ (a b : ℕ), (a.gcd b = 1) ∧ (100 * a + b = 5300) ∧
    let AEF_area := 1 in  -- We can normalize the area of ΔAEF to 1 for simplicity.
    let ABCD_area := (49 / 400 : ℚ) * AEF_area in
    (ABCD_area = 49 / 400)

theorem solve_problem : problem_statement :=
sorry

end solve_problem_l462_462928


namespace polygon_sides_eq_13_l462_462998

theorem polygon_sides_eq_13 (n : ℕ) (h : n * (n - 3) = 5 * n) : n = 13 := by
  sorry

end polygon_sides_eq_13_l462_462998


namespace sum_even_factors_720_l462_462702

theorem sum_even_factors_720 : 
  let sum_factors := (∑ i in finset.range 5, 2^i) *
                     (∑ i in finset.range 3, 3^i) *
                     (∑ i in finset.range 2 + 1, 5^i)
  in sum_factors = 2340 := 
begin
  have h1 : (∑ i in finset.range 5, 2^i) = 30,
  { simp [nat.sum]) sorry,
  have h2 : (∑ i in finset.range 3, 3^i) = 13,
  { simp [nat.sum]) sorry,
  have h3 : (∑ i in finset.range 2 + 1, 5^i) = 6,
  { simp [nat.sum]) sorry,
  sorry
end

end sum_even_factors_720_l462_462702


namespace intersection_A_B_range_of_a_l462_462866

def A := {x : ℝ | x ≥ 2}
def B := {x : ℝ | 1 ≤ x ∧ x ≤ 2}
def C (a : ℝ) := set.Icc a (2 * a - 1)

theorem intersection_A_B : A ∩ B = {2} :=
  sorry

theorem range_of_a (a : ℝ) : (C a ∪ B = B) ↔ (1 < a ∧ a ≤ 3 / 2) :=
  sorry

end intersection_A_B_range_of_a_l462_462866


namespace rate_of_interest_l462_462736

-- Define the given conditions
def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) := P * R * T / 100

-- The problem statement
theorem rate_of_interest {P : ℝ} (h : simple_interest P R 4 = P * (7 / 6 - 1)) : R = 25 / 6 :=
by sorry


end rate_of_interest_l462_462736


namespace relationship_between_x_and_y_l462_462826

variable (u : ℝ)

theorem relationship_between_x_and_y (h : u > 0) (hx : x = (u + 1)^(1 / u)) (hy : y = (u + 1)^((u + 1) / u)) :
  y^x = x^y :=
by
  sorry

end relationship_between_x_and_y_l462_462826


namespace smallest_common_multiple_l462_462700

open Nat

theorem smallest_common_multiple : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))) = 2520 :=
by
  sorry

end smallest_common_multiple_l462_462700


namespace total_profit_is_36000_l462_462254

-- Definitions for the given conditions
def ratio_Ramesh_XYZ := (5, 4)
def ratio_XYZ_Rajeev := (8, 9)
def Rajeev_share := 12000

-- Theorem statement for the total profit
theorem total_profit_is_36000
  (h_ratio_RX : ratio_Ramesh_XYZ = (5, 4))
  (h_ratio_XR : ratio_XYZ_Rajeev = (8, 9))
  (h_Rajeev_share : Rajeev_share = 12000) :
  let LCM_X := 8 in
  let new_ratio_R_X := (5 * 2, 4 * 2) in
  let combined_ratio := (10, 8, 9) in
  let part_value := 12000 / 9 in
  let total_parts := 10 + 8 + 9 in
  let total_profit := part_value * total_parts in
  total_profit = 36000 :=
sorry

end total_profit_is_36000_l462_462254


namespace find_second_discount_percentage_l462_462392

variable (list_price final_price discount1 : ℝ)
variable (second_discount : ℝ)

def condition_list_price := list_price = 65
def condition_final_price := final_price = 57.33
def condition_discount1 := discount1 = 0.1

def first_discount := list_price * discount1
def price_after_first_discount := list_price - first_discount
def second_discount_amount := price_after_first_discount - final_price
def second_discount_percentage := (second_discount_amount / price_after_first_discount) * 100

theorem find_second_discount_percentage :
  condition_list_price →
  condition_final_price →
  condition_discount1 →
  second_discount_percentage = 2 :=
by
  intros
  sorry

end find_second_discount_percentage_l462_462392


namespace kendra_shirts_l462_462217

-- Define weekly shirt requirements
def school_shirts_per_week : Nat := 5
def club_shirts_per_week : Nat := 3
def saturday_shirts_per_week : Nat := 3
def sunday_shirts_per_week : Nat := 3

-- Total shirts per week
def shirts_per_week : Nat :=
  school_shirts_per_week + club_shirts_per_week + saturday_shirts_per_week + sunday_shirts_per_week

-- Total shirts needed for three weeks
def total_shirts_needed : Nat :=
  shirts_per_week * 3

-- Theorem to prove
theorem kendra_shirts : total_shirts_needed = 42 :=
by
  def shirts_per_week_calc : Nat := 14
  def total_shirts_needed_calc : Nat := shirts_per_week_calc * 3
  -- With the given conditions, calculate the total should be 42
  show total_shirts_needed_calc = 42 from rfl

end kendra_shirts_l462_462217


namespace ticket_purchase_ways_l462_462305

theorem ticket_purchase_ways : (∀ (person1_choice person2_choice : Fin 3), true) → 
  (∃ (ways : Nat), ways = 9) :=
by
  intro h
  exists 9
  sorry

end ticket_purchase_ways_l462_462305


namespace average_first_21_multiples_of_8_l462_462712

noncomputable def average_of_multiples (n : ℕ) (a : ℕ) : ℕ :=
  let sum := (n * (a + a * n)) / 2
  sum / n

theorem average_first_21_multiples_of_8 : average_of_multiples 21 8 = 88 :=
by
  sorry

end average_first_21_multiples_of_8_l462_462712


namespace real_roots_P_n_p_l462_462965

open Polynomial

-- Define the polynomial P_{n,p}(x)
noncomputable def P_n_p (n p : ℕ) : Polynomial ℝ :=
  ∑ j in Finset.range (n + 1), (Polynomial.C ((Nat.choose p j) * (Nat.choose p (n - j))) * Polynomial.X ^ j)

theorem real_roots_P_n_p (n p : ℕ) (hn : 0 < n) (hn_le_p : n ≤ p) : 
  ∀ (x : ℝ), (Polynomial.eval x (P_n_p n p) = 0 → is_real_root (P_n_p n p) x) := 
sorry

end real_roots_P_n_p_l462_462965


namespace greatest_common_divisor_sum_arithmetic_sequence_l462_462343

theorem greatest_common_divisor_sum_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, d = 15 ∧ ∀ (n : ℕ), n = 15 → ∀ k : ℕ, k = 15 ∧ 15 ∣ (15 * (x + 7 * c)) :=
by
  sorry

end greatest_common_divisor_sum_arithmetic_sequence_l462_462343


namespace find_g_neg_3_l462_462233

def g (x : ℤ) : ℤ :=
if x < 1 then 3 * x - 4 else x + 6

theorem find_g_neg_3 : g (-3) = -13 :=
by
  -- proof omitted: sorry
  sorry

end find_g_neg_3_l462_462233


namespace longest_side_length_quadrilateral_l462_462285

theorem longest_side_length_quadrilateral : 
  (∃ (x y : ℝ), x + y ≤ 4 ∧ 2 * x + y ≥ 1 ∧ x ≥ 0 ∧ y ≥ 0)
  → ∃ a b c d: ℝ, 
    (a, b), (c,d) in {(x,y) | (x + y ≤ 4 ∧ 2 * x + y ≥ 1 ∧ x ≥ 0 ∧ y ≥ 0)}
    ∧ dist (a, b) (c, d) = 7 * sqrt 2 / 2 :=
sorry

end longest_side_length_quadrilateral_l462_462285


namespace hypotenuse_of_isosceles_right_triangle_l462_462244

theorem hypotenuse_of_isosceles_right_triangle (a : ℝ) (hyp : a = 8) : 
  ∃ c : ℝ, c = a * Real.sqrt 2 :=
by
  use 8 * Real.sqrt 2
  sorry

end hypotenuse_of_isosceles_right_triangle_l462_462244


namespace bc_sum_eq_twelve_l462_462888

theorem bc_sum_eq_twelve (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hb_lt : b < 12) (hc_lt : c < 12) 
  (h_eq : (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b * c) : b + c = 12 :=
by
  sorry

end bc_sum_eq_twelve_l462_462888


namespace coat_cost_is_correct_l462_462444
noncomputable theory

-- Let x be the cost of the coat in rubles
def coat_cost (x : ℝ) : Prop :=
  let monthly_compensation := (12 + x) / 12 in
  let total_compensation_7_months := 7 * monthly_compensation in
  total_compensation_7_months = 5 + x

theorem coat_cost_is_correct (x : ℝ) : coat_cost x → x = 4.8 :=
  by
  sorry

end coat_cost_is_correct_l462_462444


namespace find_gamma_l462_462609

noncomputable def gamma_of_delta (c : ℝ) (delta : ℝ) : ℝ :=
  c * delta^2

theorem find_gamma (delta gamma : ℝ) (c : ℝ) :
  (gamma_of_delta c 5 = 25) →
  (gamma_of_delta c 8 = 64) :=
by
  assume h1 : gamma_of_delta c 5 = 25
  sorry

end find_gamma_l462_462609


namespace greatest_divisor_arithmetic_sequence_sum_l462_462329

theorem greatest_divisor_arithmetic_sequence_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ k, (15 * (x + 7 * c)) = 15 * k :=
sorry

end greatest_divisor_arithmetic_sequence_sum_l462_462329


namespace solve_inequality_l462_462773

theorem solve_inequality (a : ℝ) : (∀ x : ℝ, |x^2 + 2*a*x + 3*a| ≤ 2 ↔ x = -a) ↔ (a = 1 ∨ a = 2) :=
sorry

end solve_inequality_l462_462773


namespace zeros_of_f_l462_462296

def f (x : ℝ) : ℝ := (x^2 - 3 * x) * (x + 4)

theorem zeros_of_f : ∀ x, f x = 0 ↔ x = 0 ∨ x = 3 ∨ x = -4 := by
  sorry

end zeros_of_f_l462_462296


namespace cubed_identity_l462_462172

variable (x : ℝ)

theorem cubed_identity (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := 
by
  sorry

end cubed_identity_l462_462172


namespace faucet_flow_rate_l462_462245

noncomputable def flow_rate (barrels : ℕ) (gallons_per_barrel : ℕ) (total_time_minutes : ℕ) : ℕ :=
  (barrels * gallons_per_barrel) / total_time_minutes

theorem faucet_flow_rate :
  flow_rate 4 7 8 = 3.5 := by
  sorry

end faucet_flow_rate_l462_462245


namespace no_valid_digits_l462_462772

theorem no_valid_digits :
  ∀ z, (z ∈ {0, 1, 3, 7, 9}) → 
  ¬ (∀ k ≥ 1, ∃ n ≥ 1, ∀ m, n^9 % 10^m = z % 10^m) := 
begin
  intros,
  sorry
end

end no_valid_digits_l462_462772


namespace decreasing_function_probability_function_decreasing_probability_l462_462434

theorem decreasing_function_probability :
  (∑ a in (finset.range 6).map (λ n, n+1), ∑ b in (finset.range 6).map (λ n, n+1),
    if (b : ℝ) / (a : ℝ) ≥ 1/2 then 1 else 0) = 30 :=
begin
  sorry
end

#eval decreasing_function_probability

theorem function_decreasing_probability :
  (30 : ℝ) / 36 = 5 / 6 :=
begin
  norm_num,
  exact rfl,
end

end decreasing_function_probability_function_decreasing_probability_l462_462434


namespace new_volume_eq_7352_l462_462024

variable (l w h : ℝ)

-- Given conditions
def volume_eq : Prop := l * w * h = 5184
def surface_area_eq : Prop := l * w + w * h + h * l = 972
def edge_sum_eq : Prop := l + w + h = 54

-- Question: New volume when dimensions are increased by two inches
def new_volume : ℝ := (l + 2) * (w + 2) * (h + 2)

-- Correct Answer: Prove that the new volume equals 7352
theorem new_volume_eq_7352 (h_vol : volume_eq l w h) (h_surf : surface_area_eq l w h) (h_edge : edge_sum_eq l w h) 
    : new_volume l w h = 7352 :=
by
  -- Proof omitted
  sorry

#check new_volume_eq_7352

end new_volume_eq_7352_l462_462024


namespace axis_of_symmetry_range_l462_462662

theorem axis_of_symmetry_range (a : ℝ) : (-(a + 2) / (3 - 4 * a) > 0) ↔ (a < -2 ∨ a > 3 / 4) :=
by
  sorry

end axis_of_symmetry_range_l462_462662


namespace monotonic_increase_intervals_max_min_values_l462_462860

-- Define the function f
def f (x : ℝ) : ℝ := 2 * sin x * cos x + sqrt 3 * cos (2 * x) + 2

-- Proof of monotonic increase intervals
theorem monotonic_increase_intervals (k : ℤ) :
  ∃ x : ℝ, -5 * pi / 12 + k * pi ≤ x ∧ x ≤ pi / 12 + k * pi ∧ 
  ∀ x1 x2, -5 * pi / 12 + k * pi ≤ x1 ∧ x1 < x2 ∧ x2 ≤ pi / 12 + k * pi → f x1 ≤ f x2 :=
sorry

-- Define the interval for max and min value search
def interval : set ℝ := {x | -pi / 3 ≤ x ∧ x ≤ pi / 3}

-- Proof of maximum and minimum values in the interval
theorem max_min_values :
  ∃ max min : ℝ, 
  max = 4 ∧ min = 2 - sqrt 3 ∧
  ∀ x ∈ interval, f x ≤ max ∧ min ≤ f x :=
sorry

end monotonic_increase_intervals_max_min_values_l462_462860


namespace tangent_line_k_value_l462_462853

noncomputable def k_tangent_ln_curve (k : ℝ) : Prop :=
  ∃ m : ℝ, m > 0 ∧ (∀ x : ℝ, (x = 0) → (k * m = 1) ∧ (m = real.exp k))

theorem tangent_line_k_value (k : ℝ) : k_tangent_ln_curve k → k = 1 / real.exp 1 := by
  sorry

end tangent_line_k_value_l462_462853


namespace log_c_comparison_l462_462557

theorem log_c_comparison (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 0 < c) (h4 : c < 1) : 
  log c a < log c b :=
sorry

end log_c_comparison_l462_462557


namespace smallest_positive_period_l462_462096

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.sin (x / 2) * Real.sin (x / 3)

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T' :=
begin
  use 12 * Real.pi,
  split,
  { exact lt_mul_of_one_lt_left (Real.pi_pos) (show 0 < 12, by norm_num) },
  { intros x, sorry },
  { intros T' hT' hf',
    have hT'positive : 0 < T',
    { sorry },
    by_contra hT'lt12pi,
    { have hf'λ : T' = 12 * Real.pi,
      { sorry },
      exact lt_irrefl _ (hT'lt12pi.trans_le hf'λ.ge) },
    sorry }
end

end smallest_positive_period_l462_462096


namespace trig_identity_l462_462383

theorem trig_identity
  (A B C : ℝ)
  (h1 : ∀ (A B C : ℝ), A + B + C = π)
  (h2 : ∀ (A B C : ℝ), sin A + sin B + sin C = 4 * cos (A / 2) * cos (B / 2) * cos (C / 2))
  (h3 : ∀ (A B : ℝ), sin A + sin B = 2 * sin ((A + B) / 2) * cos ((A - B) / 2))
  (h4 : ∀ (A B : ℝ), sin (π - (A + B)) = sin C)
  (h5 : ∀ (A B : ℝ), sin (A + B) = 2 * sin ((A + B) / 2) * cos ((A + B) / 2))
  (h6 : ∀ (x y : ℝ), cos x - cos y = -2 * sin ((x + y) / 2) * sin ((x - y) / 2)) :
  (sin A + sin B + sin C) / (sin A + sin B - sin C) = cot (A / 2) * cot (B / 2) := by
  sorry

end trig_identity_l462_462383


namespace parabola_equation_line_equations_l462_462513

/-
Conditions:
1. The parabola has its vertex at the origin.
2. The focus of the parabola is on the y-axis.
3. The parabola passes through the point P(2,1).
-/

/-- Part 1: Proof the standard equation of the parabola -/
theorem parabola_equation (x y : ℝ) (h_vertex : x^2 = 2 * (2 * y)) (h_point : (2, 1) ∈ set_of (λ p : ℝ × ℝ, p.fst^2 = 4 * p.snd)) :
  x^2 = 4 * y :=
by
  sorry

/-- Part 2: Proof the standard equations of the line intersecting the parabola at exactly one point -/
theorem line_equations (k : ℝ) (h_vertex : x^2 = 4 * y) (h_point : (2, 1) ∈ set_of (λ p : ℝ × ℝ, p.fst^2 = 4 * p.snd)) :
  (x - y = 1 ∨ x = 2) :=
by
  sorry

end parabola_equation_line_equations_l462_462513


namespace equivalent_math_problem_l462_462840

-- Condition Definitions
def center_origin (C : set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ C → (-x, y) ∈ C ∧ (x, -y) ∈ C

def foci_on_x_axis (C : set (ℝ × ℝ)) : Prop := ∃ a b : ℝ, a > b ∧ C = {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}

def passes_through_points (C : set (ℝ × ℝ)) : Prop :=
  (-2, 0) ∈ C ∧ (sqrt 2, sqrt 2 / 2) ∈ C

def focus_of_parabola : ℝ × ℝ := (1, 0)

def line_pass_through_focus (l : ℝ → ℝ) : Prop :=
  l 0 = 1

def intersects_ellipse_at_two_points (C : set (ℝ × ℝ)) (l : ℝ → ℝ) : Prop :=
  ∃ M N : ℝ × ℝ, M ≠ N ∧ (M ∈ C ∧ N ∈ C) ∧ ∃ y, l y = M.1 ∧ l y = N.1

def perpendicular_vectors (M N : ℝ × ℝ) : Prop := M.1 * N.1 + M.2 * N.2 = 0 

-- Theorem Proving the Equivalent Problem
theorem equivalent_math_problem :
  (∃ C : set (ℝ × ℝ), center_origin C ∧ foci_on_x_axis C ∧ passes_through_points C ∧
  ∀ l : ℝ → ℝ, line_pass_through_focus l →
    ∃ M N : ℝ × ℝ, intersects_ellipse_at_two_points C l ∧ perpendicular_vectors M N ∧
    (l = (λ y, 1 - y/2) ∨ l = (λ y, 1 + y/2))) :=
sorry

end equivalent_math_problem_l462_462840


namespace median_of_special_list_l462_462198

theorem median_of_special_list : 
  let lst := List.join (List.map (λ n : ℕ, List.repeat n n) (List.range 301).tail)
  in median lst = 212 :=
by
  sorry

end median_of_special_list_l462_462198


namespace anna_score_correct_l462_462190

-- Given conditions
def correct_answers : ℕ := 17
def incorrect_answers : ℕ := 6
def unanswered_questions : ℕ := 7
def point_per_correct : ℕ := 1
def point_per_incorrect : ℕ := 0
def deduction_per_unanswered : ℤ := -1 / 2

-- Proving the score
theorem anna_score_correct : 
  correct_answers * point_per_correct + incorrect_answers * point_per_incorrect + unanswered_questions * deduction_per_unanswered = 27 / 2 :=
by
  sorry

end anna_score_correct_l462_462190


namespace incorrect_statement_l462_462241

theorem incorrect_statement :
  (∀ (b h : ℝ), let A := b * h in let A' := b * (3 * h) in A' = 3 * A) ∧
  (∀ (b h : ℝ), let A := (1/2) * b * h in let A' := (1/2) * (3 * b) * h in A' = 3 * A) ∧
  (∀ (r : ℝ), let A := Real.pi * r^2 in let A' := Real.pi * (3 * r)^2 in A' ≠ 3 * A) ∧
  (∀ (a b : ℝ), let A := a / b in let A' := (a / 3) / (3 * b) in A' ≠ A) ∧
  (∀ (x : ℝ), x < 0 → 3 * x < x)
sorry

end incorrect_statement_l462_462241


namespace min_value_expression_l462_462374

theorem min_value_expression : ∀ (x y : ℝ), (∃ a b : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ 2*(x-2)^2 = a ∧ 3*(y+1)^2 = b ∧ 2*x^2 + 3*y^2 - 8*x + 6*y + 25 = a + b + 10) :=
by
  intros x y
  use (2*(x-2)^2), (3*(y+1)^2)
  split
  { exact mul_nonneg (by norm_num) (by linarith) }
  split
  { exact mul_nonneg (by norm_num) (by linarith) }
  split
  { refl }
  split
  { refl }
  sorry

end min_value_expression_l462_462374


namespace lcm_of_1_to_10_is_2520_l462_462697

noncomputable def lcm_of_1_to_10 : ℕ := 
  Nat.lcm (List.range 10).succ

theorem lcm_of_1_to_10_is_2520 : lcm_of_1_to_10 = 2520 :=
by
  sorry

end lcm_of_1_to_10_is_2520_l462_462697


namespace number_of_distinct_pairs_l462_462777

theorem number_of_distinct_pairs (x y : ℕ) (h₁ : 0 < x) (h₂ : x < y) (h₃ : real.sqrt 1156 = real.sqrt x + real.sqrt y) :
  ∃ S : finset (ℕ × ℕ), (∀ p ∈ S, ∃ x y, p = (x, y) ∧ 0 < x ∧ x < y ∧ real.sqrt 1156 = real.sqrt x + real.sqrt y) ∧ S.card = 16 :=
begin
  sorry
end

end number_of_distinct_pairs_l462_462777


namespace number_of_correct_propositions_is_one_l462_462144

-- Define the conditions as given in the problem
def proposition1 (l α β : Type) [linear α l] [linear β l] (h1 : perpendicular l α) (h2 : parallel l β) : perpendicular α β :=
sorry

def proposition2 (α β : Type) (h : ∃ (A B C : Type), non_collinear (A B C) ∧ equidistant_from_plane B β ∧ equidistant_from_plane C β ∧ equidistant_from_plane A β ∧ in_plane A α ∧ in_plane B α ∧ in_plane C α) : parallel α β :=
sorry

def proposition3 (α β : Type) (h : ∃ (P Q : Type), half_planes_of_dihedral_angle P α ∧ half_planes_of_dihedral_angle Q β ∧ perpendicular_planes P Q) : plane_angles_equal_or_complementary α β :=
sorry

def proposition4 (points_space : Type) (l1 l2 α : Type) (h1 : non_coplanar l1 l2) (h2 : parallel α l1) (h3 : parallel α l2) : ∃ (p : points_space), in_plane p α :=
sorry

-- Main theorem to be proved
theorem number_of_correct_propositions_is_one : 
    number_correct_propositions = 1 :=
sorry

end number_of_correct_propositions_is_one_l462_462144


namespace radius_of_circumcircle_l462_462275

def hyperbola (x y a : ℝ) : Prop := (x^2 / a^2) - (y^2 / 16) = 1

def foci (A B : ℝ × ℝ) (a : ℝ) : Prop := 
  A = (-a,  0) ∧ B = (a, 0)

def on_hyperbola (P : ℝ × ℝ) (a : ℝ) : Prop :=
  hyperbola P.1 P.2 a

def incenter (P A B : ℝ × ℝ) : ℝ × ℝ := (3, 1)

def circumradius (P A B : ℝ × ℝ) : ℝ := 65 / 12

theorem radius_of_circumcircle (a : ℝ) (A B P : ℝ × ℝ) 
  (h_hyperbola : hyperbola P.1 P.2 a)
  (h_foci : foci A B a)
  (h_incenter : incenter P A B = (3, 1)) :
  circumradius P A B = 65 / 12 :=
sorry

end radius_of_circumcircle_l462_462275


namespace casey_saves_money_l462_462058

def first_employee_hourly_wage : ℕ := 20
def second_employee_hourly_wage : ℕ := 22
def subsidy_per_hour : ℕ := 6
def weekly_work_hours : ℕ := 40

theorem casey_saves_money :
  let first_employee_weekly_cost := first_employee_hourly_wage * weekly_work_hours
  let second_employee_effective_hourly_wage := second_employee_hourly_wage - subsidy_per_hour
  let second_employee_weekly_cost := second_employee_effective_hourly_wage * weekly_work_hours
  let savings := first_employee_weekly_cost - second_employee_weekly_cost
  savings = 160 :=
by
  sorry

end casey_saves_money_l462_462058


namespace sally_sewed_2_shirts_on_wednesday_l462_462647

def shirts_on_monday : ℕ := 4
def shirts_on_tuesday : ℕ := 3
def buttons_per_shirt : ℕ := 5
def total_buttons : ℕ := 45

def buttons_needed_for_wednesday : ℕ := total_buttons - (shirts_on_monday * buttons_per_shirt + shirts_on_tuesday * buttons_per_shirt)
def shirts_on_wednesday : ℕ := buttons_needed_for_wednesday / buttons_per_shirt

theorem sally_sewed_2_shirts_on_wednesday : shirts_on_wednesday = 2 :=
by
  rw [shirts_on_wednesday, buttons_needed_for_wednesday]
  simp [total_buttons, shirts_on_monday, shirts_on_tuesday, buttons_per_shirt]
  -- Automatically simplified Lean term: (45 - (4 * 5 + 3 * 5)) / 5 = 2
  sorry

end sally_sewed_2_shirts_on_wednesday_l462_462647


namespace find_f_f_3_l462_462865

noncomputable def f : ℝ → ℝ
| x => if x ≤ 1 then 2^x else f (x-1)

theorem find_f_f_3 : f (f 3) = 2 := 
by
  sorry

end find_f_f_3_l462_462865


namespace surface_area_cubic_parabola_rotation_l462_462655

noncomputable def surfaceAreaOfRotation (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  2 * Real.pi * ∫ x in a..b, f x * Real.sqrt (1 + (deriv f x)^2)

def cubicParabola (x : ℝ) : ℝ := (1/3) * x^3

theorem surface_area_cubic_parabola_rotation : 
  surfaceAreaOfRotation cubicParabola 0 1 = Real.pi / 9 * (2 * Real.sqrt 2 - 1) :=
by
  sorry

end surface_area_cubic_parabola_rotation_l462_462655


namespace real_part_of_z_l462_462831

theorem real_part_of_z (z : ℂ) (h : z - abs z = -8 + 12 * complex.I) : z.re = 5 :=
sorry

end real_part_of_z_l462_462831


namespace tulip_area_of_flower_bed_l462_462410

theorem tulip_area_of_flower_bed 
  (CD CF : ℝ) (DE : ℝ := 4) (EF : ℝ := 3) 
  (triangle : ∀ (A B C : ℝ), A = B + C) : 
  CD * CF = 12 :=
by sorry

end tulip_area_of_flower_bed_l462_462410


namespace find_a6_l462_462589

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 2 ∧ a 2 = 5 ∧ ∀ n : ℕ, a (n + 1) = a (n + 2) + a n

theorem find_a6 (a : ℕ → ℤ) (h : seq a) : a 6 = -3 :=
by
  sorry

end find_a6_l462_462589


namespace triangle_areas_ratio_l462_462836

variables {A B C I I1 L M N : Type*}
variables {a b c : ℝ}
variables [DecidableEq L] [DecidableEq M] [DecidableEq N] 

-- Definitions of the triangle ABC and properties
variable (ABC : Triangle A B C)
variable (acute_ABC : ABC.isAcute)
variable (AB_gt_AC : a > b)

-- Definitions of the incenter, A-excenter, midpoints, and intersections
variable (I : Incenter ABC)
variable (I1 : AExcenter ABC)
variable (L : Midpoint (B, C))
variable (LI_intersects_AC_at_M : LineThrough (L, I).Intersects (C, A) = M)
variable (I1L_intersects_AB_at_N : LineThrough (I1, L).Intersects (A, B) = N)

-- Lengths of the sides of the triangle
variable (BC_len_a : SegmentLength (B, C) = a)
variable (CA_len_b : SegmentLength (C, A) = b)
variable (AB_len_c : SegmentLength (A, B) = c)

-- Main statement with the desired ratio of areas
theorem triangle_areas_ratio :
  (AreaRatio (Triangle M N I) (Triangle A B C)) = a * (c - b) / (a + c - b) ^ 2 :=
sorry

end triangle_areas_ratio_l462_462836


namespace number_of_non_empty_subsets_l462_462997

theorem number_of_non_empty_subsets (s : Finset ℕ) (h : s = {1, 2, 3}) : 
  (s.powerset.filter (λ x, x.nonempty)).card = 7 :=
by
  -- Proof omitted
  sorry

end number_of_non_empty_subsets_l462_462997


namespace divides_expression_l462_462248

theorem divides_expression (M : ℕ) : 1992 ∣ (82.factorial * (Finset.sum (Finset.range 83) (λ i, 1 / (i + 1 : ℚ)))) := 
sorry

end divides_expression_l462_462248


namespace swim_team_girls_l462_462737

-- Definitions using the given conditions
variables (B G : ℕ)
theorem swim_team_girls (h1 : G = 5 * B) (h2 : G + B = 96) : G = 80 :=
sorry

end swim_team_girls_l462_462737


namespace existence_of_invisible_polyhedron_l462_462077

theorem existence_of_invisible_polyhedron :
  ∃ (P : Polyhedron) (p : Point), p ∉ (convex_hull P) ∧ ¬(∃ v ∈ P.vertices, segment p v ∩ face P v = ∅ ) :=
sorry

end existence_of_invisible_polyhedron_l462_462077


namespace cubed_identity_l462_462174

variable (x : ℝ)

theorem cubed_identity (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := 
by
  sorry

end cubed_identity_l462_462174


namespace arc_length_l462_462569

-- Define the radius and central angle
def radius : ℝ := 10
def central_angle : ℝ := 240

-- Theorem to prove the arc length is (40 * π) / 3
theorem arc_length (r : ℝ) (n : ℝ) (h_r : r = radius) (h_n : n = central_angle) : 
  (n * π * r) / 180 = (40 * π) / 3 :=
by
  -- Proof omitted
  sorry

end arc_length_l462_462569


namespace smallest_sum_l462_462377

-- Define the digits and constraints
def digits : List ℕ := [1, 2, 3, 7, 8, 9]

-- Each number must include one digit from {1, 2, 3} and one from {7, 8, 9}
def valid_assignment (a b c d e f : ℕ) : Prop :=
  a ∈ [1, 2, 3] ∧ b ∈ [1, 2, 3] ∧ c ∈ [1, 2, 3] ∧
  d ∈ [7, 8, 9] ∧ e ∈ [7, 8, 9] ∧ f ∈ [7, 8, 9] ∧
  List.nodup [a, b, c, d, e, f]

-- Function to calculate the sum
def calculate_sum (a b c d e f : ℕ) : ℕ :=
  100 * (a + d) + 10 * (b + e) + (c + f)

-- The final theorem stating the smallest sum
theorem smallest_sum : ∃ a b c d e f, valid_assignment a b c d e f ∧ calculate_sum a b c d e f = 417 :=
by
  sorry

end smallest_sum_l462_462377


namespace variance_Y_is_8_l462_462545

-- Define the binomial random variable X
noncomputable def X : ℕ → ℕ → Type :=
  sorry -- placeholder for the binomial distribution definition

-- Define the first variable Y based on X
def Y (X : Type) : Type :=
  2 * X - 1

-- Define the expectation and variance operations
noncomputable def expectation (Y : Type) : ℝ :=
  sorry -- placeholder for the expectation computation

noncomputable def variance (Y : Type) : ℝ :=
  sorry -- placeholder for the variance computation

-- Given conditions
variable (X : Type) [binomial X 9 (2 / 3)]

-- Proof statement
theorem variance_Y_is_8 : variance (Y X) = 8 :=
by
  sorry

end variance_Y_is_8_l462_462545


namespace original_length_of_field_l462_462656

theorem original_length_of_field (L W : ℕ) 
  (h1 : L * W = 144) 
  (h2 : (L + 6) * W = 198) : 
  L = 16 := 
by 
  sorry

end original_length_of_field_l462_462656


namespace magicians_can_determine_chosen_cards_l462_462189

noncomputable def card_circle : list ℕ := list.range 29  -- A circular arrangement of cards

def magicians_strategy (chosen_cards : set ℕ) (received_cards : set ℕ) : Prop :=
  ∃ assistant_cards : set ℕ,
  -- Assistant selects appropriate cards
  assistant_cards ⊆ (set.of_list (card_circle.to_finset.filter (λ x, ¬ chosen_cards x))) ∧
  -- Assistant's choice depends on whether the chosen cards are consecutive
  ((∀ c1 c2 ∈ chosen_cards, ∃ next_c1 next_c2 ∈ assistant_cards, next_c1 = c1 + 1 ∧ next_c2 = c2 + 1) ∨
  (∀ c1 c2 ∈ chosen_cards, ∃ next_c1 next_c2 ∈ assistant_cards, next_c1 = c1 + 1 ∧ next_c2 = c2 + 1))

theorem magicians_can_determine_chosen_cards (chosen_cards : set ℕ) : ∀ received_cards, 
chimneys_strategy chosen_cards received_cards → ∃ original_cards, magicians_strategy original_cards received_cards :=
sorry

end magicians_can_determine_chosen_cards_l462_462189


namespace songs_owned_initially_l462_462484

theorem songs_owned_initially (a b c : ℕ) (hc : c = a + b) (hb : b = 7) (hc_total : c = 13) :
  a = 6 :=
by
  -- Direct usage of the given conditions to conclude the proof goes here.
  sorry

end songs_owned_initially_l462_462484


namespace greatest_divisor_of_arithmetic_sequence_sum_l462_462339

theorem greatest_divisor_of_arithmetic_sequence_sum :
  ∀ (x c : ℕ), ∃ k : ℕ, k = 15 ∧ 15 ∣ (15 * x + 105 * c) :=
by
  intro x c
  exists 15
  split
  . rfl
  . sorry

end greatest_divisor_of_arithmetic_sequence_sum_l462_462339


namespace volume_solid_T_l462_462809

open Real

def solid_T (x y z : ℝ) : Prop :=
  abs x + abs y ≤ 2 ∧ abs x + abs z ≤ 2 ∧ abs y + abs z ≤ 2

theorem volume_solid_T : 
  ∃ (V : ℝ), (∀ (x y z : ℝ), solid_T x y z → V = ∫∫∫_{solid T} dx dy dz) ∧ V = 32 / 3 := 
sorry

end volume_solid_T_l462_462809


namespace ten_years_less_average_age_l462_462574

-- Defining the conditions formally
def lukeAge : ℕ := 20
def mrBernardAgeInEightYears : ℕ := 3 * lukeAge

-- Lean statement to prove the problem
theorem ten_years_less_average_age : 
  mrBernardAgeInEightYears - 8 = 52 → (lukeAge + (mrBernardAgeInEightYears - 8)) / 2 - 10 = 26 := 
by
  intros h
  sorry

end ten_years_less_average_age_l462_462574


namespace incorrect_algorithm_property_C_l462_462706

def algorithm_property_A : Prop := 
  ∀ algorithm : String, 
  (is_executed_step_by_step algorithm ∧ yields_unique_result_each_step algorithm)

def algorithm_property_B : Prop := 
  ∀ algorithm : String, 
  (effective_for_class_of_problems algorithm)

def algorithm_property_D : Prop := 
  ∀ algorithm : String, 
  (mechanical_repetitive_calculation algorithm ∧ universal_method algorithm)

def algorithm_property_C : Prop := 
  ∀ problem : String, 
  ∃ algorithm : String, 
  solves_problem algorithm problem

theorem incorrect_algorithm_property_C : 
  algorithm_property_A ∧ algorithm_property_B ∧ algorithm_property_D → ¬ algorithm_property_C :=
  by sorry

end incorrect_algorithm_property_C_l462_462706


namespace odd_function_extension_monotonicity_range_m_l462_462522

noncomputable def f (x : ℝ) : ℝ := if x ≤ 0 then x^2 + 2*x else -x^2 + 2*x

theorem odd_function_extension :
  ∀ x : ℝ, f(x) = (if x ≤ 0 then x^2 + 2*x else -x^2 + 2*x) := by
sorry

theorem monotonicity_range_m (m : ℝ) :
  (∀ x : ℝ, x ∈ Icc (-1 : ℝ) (m - 1) → (differentiable ℝ f) ∧ (deriv f x > 0)) →
  0 < m ∧ m ≤ 2 := by
sorry

end odd_function_extension_monotonicity_range_m_l462_462522


namespace ratio_a_to_b_l462_462271

variable (a x c d b : ℝ)
variable (h1 : d = 3 * x + c)
variable (h2 : b = 4 * x)

theorem ratio_a_to_b : a / b = -1 / 4 := by 
  sorry

end ratio_a_to_b_l462_462271


namespace sample_size_eq_100_l462_462429

variables (frequency : ℕ) (frequency_rate : ℚ)

theorem sample_size_eq_100 (h1 : frequency = 50) (h2 : frequency_rate = 0.5) :
  frequency / frequency_rate = 100 :=
by
  sorry

end sample_size_eq_100_l462_462429


namespace james_two_point_shots_l462_462598

-- Definitions based on conditions
def field_goals := 13
def field_goal_points := 3
def total_points := 79

-- Statement to be proven
theorem james_two_point_shots :
  ∃ x : ℕ, 79 = (field_goals * field_goal_points) + (2 * x) ∧ x = 20 :=
by
  sorry

end james_two_point_shots_l462_462598


namespace polynomial_remainder_l462_462380

theorem polynomial_remainder (p : ℚ[X]) (h1 : p.eval (-2) = 4) (h2 : p.eval (-4) = -8) :
  ∃ (q : ℚ[X]), p = (X + 2) * (X + 4) * q + (6 * X + 16) :=
by
  sorry

end polynomial_remainder_l462_462380


namespace law_of_motion_l462_462723

-- Input Conditions
variables (m ω : ℝ) (x : ℝ → ℝ)

-- Assumptions
axiom (h1 : ∃ C₁ C₂ t : ℝ, x t = C₁ * Real.cos (ω * t) + C₂ * Real.sin (ω * t))
axiom (h2 : m ≠ 0)
axiom (h3 : ∀ t, x'' t = - ω^2 * x t)

-- Required to Prove
theorem law_of_motion (x : ℝ → ℝ) (R α : ℝ) :
  x = (λ t, R * Real.sin (ω * t + α)) :=
sorry

end law_of_motion_l462_462723


namespace eating_time_proof_l462_462625

noncomputable def combined_eating_time (time_fat time_thin weight : ℝ) : ℝ :=
  let rate_fat := 1 / time_fat
  let rate_thin := 1 / time_thin
  let combined_rate := rate_fat + rate_thin
  weight / combined_rate

theorem eating_time_proof :
  let time_fat := 12
  let time_thin := 40
  let weight := 5
  combined_eating_time time_fat time_thin weight = (600 / 13) :=
by
  -- placeholder for the proof
  sorry

end eating_time_proof_l462_462625


namespace polynomial_factorization_l462_462443

lemma correct_factorization :
  (x^2 - 4) = (x + 2) * (x - 2) :=
by
  sorry

theorem polynomial_factorization (x y : ℝ) :
  ∃ (option : ∃ (A : Prop), ∃ (B : Prop), ∃ (C : Prop), ∃ (D : Prop), C),
  let A := (x ^ 2 + 3 * x - 4 = x * (x + 3)),
      B := (x ^ 2 - 4 + 3 * x = (x + 2) * (x - 2)),
      C := (x ^ 2 - 4 = (x + 2) * (x - 2)),
      D := (x ^ 2 - 2 * x * y + y ^ 2 = (x - y) ^ 2)
  in option :=
begin
  use exists.intro _ _,
  exact ⟨A, B, correct_factorization, D⟩,
  sorry
end

end polynomial_factorization_l462_462443


namespace option_A_option_B_option_C_option_D_l462_462111

-- Define the sequence and its sum
def geom_seq (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

def arith_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_seq (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (finset.range n).sum a

-- Define the problems
theorem option_A (a : ℕ → ℝ) (h1 : geom_seq a) (h2 : ∀ n, a n > 0) : arith_seq (λ n, real.log (a n) / real.log 3) := sorry

theorem option_B (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : arith_seq a) (h2 : sum_seq a S) (h3 : S 3 / S 6 = 1 / 4) : S 6 / S 12 = 1 / 4 := sorry

theorem option_C (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : arith_seq a) (h2 : sum_seq a S) (h3 : (finset.range (2 * 5)).sum (λ n, if n % 2 = 0 then a (n / 2) else 0) / 
                       (finset.range (2 * 5)).sum (λ n, if n % 2 = 1 then a (n / 2) else 0) = 9 / 8) (h4 : S 10 = 170) : 
  ∃ d : ℝ, ∀ n, a (n + 1) = a n + d ∧ d = 2 := sorry

theorem option_D (a : ℕ → ℝ) (h : a 1 = 2 ∧ a 2 = 1 ∧ ∀ n, a (n + 2) = abs (a (n + 1) - a n)) : 
  ¬ ((finset.range 100).sum a = 67) := sorry

end option_A_option_B_option_C_option_D_l462_462111


namespace measure_angledb_l462_462916

-- Definitions and conditions:
def right_triangle (A B C : Type) : Prop :=
  ∃ (a b c : A), ∠A = 45 ∧ ∠B = 45 ∧ ∠C = 90

def angle_bisector (A B : Type) (D : Type) : Prop :=
  ∃ (AD BD : A), AD bisects ∠A ∧ BD bisects ∠B

-- Prove:
theorem measure_angledb (A B C D : Type) 
  (hABC : right_triangle A B C)
  (hAngleBisectors : angle_bisector A B D) : 
  ∠ ADB = 135 := 
sorry

end measure_angledb_l462_462916


namespace num_detestable_below_10000_l462_462022

-- Define what it means for a positive integer n to be detestable
def is_detestable (n : ℕ) : Prop :=
  (∑ d in n.digits, d) % 11 = 0

-- Define the set of positive integers below 10000
def below_10000 := {n : ℕ | n < 10000}

-- The main statement we want to prove
theorem num_detestable_below_10000 : 
  {n : ℕ | n ∈ below_10000 ∧ is_detestable n}.card = 1008 := 
sorry

end num_detestable_below_10000_l462_462022


namespace twentieth_number_is_40_base_5_l462_462903

-- Define the conversion from base 10 to base 5
def convert_to_base_5 (n : ℕ) : list ℕ :=
  if n = 0 then [0] else
  let rec convert_aux : ℕ → list ℕ
    | 0 => []
    | m => (m % 5) :: convert_aux (m / 5)
  in convert_aux n

-- Define the twentieth number in base 5
def twentieth_number_base_5 : list ℕ := convert_to_base_5 20

-- Theorem stating the twentieth number in base 5 is [4, 0]
theorem twentieth_number_is_40_base_5 : twentieth_number_base_5 = [0, 4] := 
  sorry

end twentieth_number_is_40_base_5_l462_462903


namespace complex_magnitude_sum_l462_462487

theorem complex_magnitude_sum : |(3 - 5 * complex.I)| + |(3 + 5 * complex.I)| = 2 * real.sqrt 34 :=
by
  sorry

end complex_magnitude_sum_l462_462487


namespace cost_per_mile_proof_l462_462000

noncomputable def daily_rental_cost : ℝ := 50
noncomputable def daily_budget : ℝ := 88
noncomputable def max_miles : ℝ := 190.0

theorem cost_per_mile_proof : 
  (daily_budget - daily_rental_cost) / max_miles = 0.20 := 
by
  sorry

end cost_per_mile_proof_l462_462000


namespace find_z_proportional_l462_462893

theorem find_z_proportional (k : ℝ) (y x z : ℝ) 
  (h₁ : y = 8) (h₂ : x = 2) (h₃ : z = 4) (relationship : y = (k * x^2) / z)
  (y' x' z' : ℝ) (h₄ : y' = 72) (h₅ : x' = 4) : 
  z' = 16 / 9 := by
  sorry

end find_z_proportional_l462_462893


namespace solve_m_problem_l462_462543

theorem solve_m_problem :
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ x^2 - x - m = 0) →
  m ∈ Set.Ico (-1/4 : ℝ) 2 :=
sorry

end solve_m_problem_l462_462543


namespace total_distance_flown_l462_462741

/-- 
An eagle can fly 15 miles per hour; 
a falcon can fly 46 miles per hour; 
a pelican can fly 33 miles per hour; 
and a hummingbird can fly 30 miles per hour. 
All the birds flew for 2 hours straight.
Prove that the total distance flown by all the birds is 248 miles.
-/
theorem total_distance_flown :
  let eagle_speed := 15
      falcon_speed := 46
      pelican_speed := 33
      hummingbird_speed := 30
      hours_flown := 2
      eagle_distance := eagle_speed * hours_flown 
      falcon_distance := falcon_speed * hours_flown 
      pelican_distance := pelican_speed * hours_flown 
      hummingbird_distance := hummingbird_speed * hours_flown 
  in eagle_distance + falcon_distance + pelican_distance + hummingbird_distance = 248 := 
sorry

end total_distance_flown_l462_462741


namespace probability_of_valid_pairs_l462_462972

-- Define the set of balls
def balls : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the condition to check if a product is odd and less than 20
def valid_product (x y : ℕ) : Prop := odd (x * y) ∧ x * y < 20

-- Define the set of valid pairs
def valid_pairs : Finset (ℕ × ℕ) :=
  {p | p.1 ∈ balls ∧ p.2 ∈ balls ∧ valid_product p.1 p.2}.to_finset

-- Define the proof statement for the the probability being 13/49
theorem probability_of_valid_pairs : valid_pairs.card = 13 ∧ (balls.card * balls.card = 49) → (13/49 : ℚ) = 13 / 49 :=
by
  sorry

end probability_of_valid_pairs_l462_462972


namespace cosine_smallest_angle_l462_462670

theorem cosine_smallest_angle (n : ℕ) (x y : ℝ) 
  (h₁ : ∃ (n : ℕ), true) 
  (h₂ : y = 1.5 * x) 
  (h₃ : ∃ (x : ℝ), true) 
  (h₄ : cos x = (n^2 + 7 * n + 5) / (2 * (n^2 + 6 * n + 2))) : 
  cos x = 65 / 114 :=
begin
  sorry
end

end cosine_smallest_angle_l462_462670


namespace greatest_divisor_sum_of_first_fifteen_terms_l462_462320

theorem greatest_divisor_sum_of_first_fifteen_terms 
  (x c : ℕ) (hx : x > 0) (hc : c > 0):
  ∃ d, d = 15 ∧ d ∣ (15*x + 105*c) :=
by
  existsi 15
  split
  . refl
  . apply Nat.dvd.intro
    existsi (x + 7*c)
    refl
  sorry

end greatest_divisor_sum_of_first_fifteen_terms_l462_462320


namespace greatest_divisor_of_arithmetic_sequence_sum_l462_462338

theorem greatest_divisor_of_arithmetic_sequence_sum :
  ∀ (x c : ℕ), ∃ k : ℕ, k = 15 ∧ 15 ∣ (15 * x + 105 * c) :=
by
  intro x c
  exists 15
  split
  . rfl
  . sorry

end greatest_divisor_of_arithmetic_sequence_sum_l462_462338


namespace intersection_point_divides_chord_l462_462664

theorem intersection_point_divides_chord (R AB PO : ℝ)
    (hR: R = 11) (hAB: AB = 18) (hPO: PO = 7) :
    ∃ (AP PB : ℝ), (AP / PB = 2 ∨ AP / PB = 1 / 2) ∧ (AP + PB = AB) := by
  sorry

end intersection_point_divides_chord_l462_462664


namespace probability_point_closer_to_six_l462_462019

theorem probability_point_closer_to_six :
  let segment := (0, 8)
  let midpoint := (0 + 6) / 2
  let closer_interval := (3, 8)
  let probability := (8 - 3) / (8 - 0) in
  (Real.floor (probability * 10)) / 10 = 0.6 :=
by
  let segment := (0, 8)
  let midpoint := (0 + 6) / 2
  let closer_interval := (3, 8)
  let probability := (8 - 3) / (8 - 0)
  show (Real.floor (probability * 10)) / 10 = 0.6
  sorry -- proof to be completed

end probability_point_closer_to_six_l462_462019


namespace increasing_condition_l462_462934

noncomputable def f (x a : ℝ) : ℝ := (Real.exp x) + a * (Real.exp (-x))

theorem increasing_condition (a : ℝ) : (∀ x : ℝ, 0 ≤ (Real.exp (2 * x) - a) / (Real.exp x)) ↔ a ≤ 0 :=
by
  sorry

end increasing_condition_l462_462934


namespace eggs_total_l462_462481

-- Definitions based on conditions
def isPackageSize (n : Nat) : Prop :=
  n = 6 ∨ n = 11

def numLargePacks : Nat := 5

def largePackSize : Nat := 11

-- Mathematical statement to prove
theorem eggs_total : ∃ totalEggs : Nat, totalEggs = numLargePacks * largePackSize :=
  by sorry

end eggs_total_l462_462481


namespace inspection_probability_l462_462524

noncomputable def defective_items : ℕ := 2
noncomputable def good_items : ℕ := 3
noncomputable def total_items : ℕ := defective_items + good_items

/-- Given 2 defective items and 3 good items mixed together,
the probability that the inspection stops exactly after
four inspections is 3/5 --/
theorem inspection_probability :
  (2 * (total_items - 1) * total_items / (total_items * (total_items - 1) * (total_items - 2) * (total_items - 3))) = (3 / 5) :=
by
  sorry

end inspection_probability_l462_462524


namespace Sn_2011_l462_462584

def arithmetic_sequence (a : ℕ → ℤ) := ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

variables (a S : ℕ → ℤ)
variable h1 : a 1 = -2011
variable h2 : (S 2010) / 2010 - (S 2008) / 2008 = 2

theorem Sn_2011 : S 2011 = -2011 :=
sorry

end Sn_2011_l462_462584


namespace f_periodic_and_value_l462_462531

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then x else -f(x-1)

theorem f_periodic_and_value :
  (∀ x : ℝ, f(x+1) = -f(x)) ∧
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ 1) → f(x) = x) ∧
  (f 8.5 = 0.5) :=
by
  sorry

end f_periodic_and_value_l462_462531


namespace polar_equation_of_C_MA_MB_product_l462_462201

-- Define the parameter equations of line l and curve C
def line_l (t : ℝ) : ℝ × ℝ := (2 + t, 1 + t * Real.sqrt 3)
def curve_C (θ : ℝ) : ℝ × ℝ := (4 + 2 * Real.cos θ, 3 + 2 * Real.sin θ)
def M : ℝ × ℝ := (2, 1)

-- Question (I): Find the polar equation of curve C
theorem polar_equation_of_C : 
  ∃ (ρ θ : ℝ), ρ^2 - 8 * ρ * Real.cos θ - 6 * ρ * Real.sin θ + 21 = 0 := 
sorry

-- Question (II): Prove the value of |MA| * |MB| given the intersection points A and B
theorem MA_MB_product : 
∃ (t₁ t₂ : ℝ), 
  let A := line_l t₁,
  let B := line_l t₂,
  let dist (P Q : ℝ × ℝ) := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) in
  dist M A * dist M B = 4 :=
sorry

end polar_equation_of_C_MA_MB_product_l462_462201


namespace move_left_is_negative_l462_462041

theorem move_left_is_negative (movement_right : ℝ) (h : movement_right = 3) : -movement_right = -3 := 
by 
  sorry

end move_left_is_negative_l462_462041


namespace petya_cannot_form_figure_c_l462_462962

-- Define the rhombus and its properties, including rotation
noncomputable def is_rotatable_rhombus (r : ℕ) : Prop := sorry

-- Define the larger shapes and their properties in terms of whether they can be formed using rotations of the rhombus.
noncomputable def can_form_figure_a (rhombus : ℕ) : Prop := sorry
noncomputable def can_form_figure_b (rhombus : ℕ) : Prop := sorry
noncomputable def can_form_figure_c (rhombus : ℕ) : Prop := sorry
noncomputable def can_form_figure_d (rhombus : ℕ) : Prop := sorry

-- Statement: Petya cannot form the figure (c) using the rhombus and allowed transformations.
theorem petya_cannot_form_figure_c (rhombus : ℕ) (h : is_rotatable_rhombus rhombus) :
  ¬ can_form_figure_c rhombus := sorry

end petya_cannot_form_figure_c_l462_462962


namespace cornbread_pieces_l462_462601

theorem cornbread_pieces (pan_length pan_width piece_length piece_width : ℕ)
  (h₁ : pan_length = 24) (h₂ : pan_width = 20) 
  (h₃ : piece_length = 3) (h₄ : piece_width = 2) :
  (pan_length * pan_width) / (piece_length * piece_width) = 80 := by
  sorry

end cornbread_pieces_l462_462601


namespace sum_inequality_nonnegative_reals_l462_462945

theorem sum_inequality_nonnegative_reals
  (x : ℕ → ℝ)
  (hx_nonneg : ∀ i, 0 ≤ x i)
  (hx_sum : ∑ i in Finset.range 5, 1 / (1 + x i) = 1) :
  ∑ i in Finset.range 5, x i / (4 + (x i)^2) ≤ 1 :=
sorry

end sum_inequality_nonnegative_reals_l462_462945


namespace inverse_of_matrixA_l462_462091

open Matrix

def matrixA : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![2, -1],
  ![4, 3]
]

def matrixAInv : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![3/10, 1/10],
  ![-2/5, 1/5]
]

theorem inverse_of_matrixA : matrixA ⬝ matrixAInv = 1 :=
by {
  sorry
}

end inverse_of_matrixA_l462_462091


namespace tire_price_l462_462676

theorem tire_price (x : ℕ) (h : 4 * x + 5 = 485) : x = 120 :=
by
  sorry

end tire_price_l462_462676


namespace function_satisfies_conditions_increasing_intervals_and_axis_of_symmetry_l462_462537

noncomputable def analyt_expression (f : ℝ → ℝ) (x : ℝ) : ℝ := 2 * Real.sin (2 * x + π/6)

def highest_point (f : ℝ → ℝ) : Prop :=
  f (π/6) = 2

def lowest_point (f : ℝ → ℝ) : Prop :=
  f (2*π/3) = -2

-- The function definition
def f := λ x : ℝ, 2 * Real.sin (2 * x + π/6)

-- Proving that the function satisfies the given conditions
theorem function_satisfies_conditions :
  highest_point f ∧ lowest_point f :=
by
  split
  · -- Proof for highest point
    unfold highest_point
    dsimp [f]
    rw [Real.sin_add, Real.sin_mul, Real.cos_mul]
    sorry
  · -- Proof for lowest point
    unfold lowest_point
    dsimp [f]
    rw [Real.sin_add, Real.sin_mul, Real.cos_mul]
    sorry

-- Increasing intervals and axis of symmetry statements to be proved
theorem increasing_intervals_and_axis_of_symmetry :
  (∀ k : ℤ, ∀ x : ℝ, k * π - π/3 ≤ x ∧ x ≤ k * π + π/6 → f x ≥ 0) ∧
  (∀ k : ℤ, (2 * k * π + π/2)/2 + π/6 = k * π/2 + π/6) :=
by
  split
  · -- Proof for increasing intervals
    intro k x
    sorry
  · -- Proof for axis of symmetry
    intro k
    sorry

end function_satisfies_conditions_increasing_intervals_and_axis_of_symmetry_l462_462537


namespace num_partitions_with_two_internal_triangles_l462_462506

-- Define binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

theorem num_partitions_with_two_internal_triangles (n : ℕ) (h_n : n > 7) :
  (∃ P : Type, is_convex_polygon_with_n_sides P n) →
  (∃ (f : P → finset P), num_internal_triangles f = 2) →
  partitions_with_two_internal_triangles n = n * binom (n - 4) 4 * 2^(n-9) :=
sorry

end num_partitions_with_two_internal_triangles_l462_462506


namespace greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l462_462358

theorem greatest_divisor_of_sum_first_15_terms_arithmetic_sequence
  (x c : ℕ) -- where x and c are positive integers
  (h_pos_x : 0 < x) -- x is positive
  (h_pos_c : 0 < c) -- c is positive
  : ∃ (d : ℕ), d = 15 ∧ ∀ (S : ℕ), S = 15 * x + 105 * c → d ∣ S :=
by
  sorry

end greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l462_462358


namespace proof_problem_l462_462844

-- Define the given hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2 / 4) - y^2 = 1

-- Define the equation of the circle with diameter F1F2
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 = 5

-- Define the asymptotes of the hyperbola
def asymptote_eq (x y : ℝ) : Prop :=
  y = (1 / 2) * x ∨ y = -(1 / 2) * x

-- The main theorem to prove:
theorem proof_problem :
  (∀ M : ℝ × ℝ, asymptote_eq M.1 M.2 → circle_eq M.1 M.2 →
    (M.1 = 2 ∨ M.1 = -2) ∧
    let [F1, F2] := [(-sqrt 5, 0), (sqrt 5, 0)]
    in (1 / 2) * |2 * sqrt 5| * |1| = sqrt 5) :=
by
  sorry

end proof_problem_l462_462844


namespace product_of_two_numbers_l462_462294

-- Define the conditions
def two_numbers (x y : ℝ) : Prop :=
  x + y = 27 ∧ x - y = 7

-- Define the product function
def product_two_numbers (x y : ℝ) : ℝ := x * y

-- State the theorem
theorem product_of_two_numbers : ∃ x y : ℝ, two_numbers x y ∧ product_two_numbers x y = 170 := by
  sorry

end product_of_two_numbers_l462_462294


namespace Donna_total_earnings_l462_462479

theorem Donna_total_earnings :
  let walking_earnings := 10 * 2 * 7 in
  let card_shop_earnings := 12.5 * 2 * 5 in
  let babysitting_earnings := 10 * 4 in
  walking_earnings + card_shop_earnings + babysitting_earnings = 305 := 
by 
  let walking_earnings := 10 * 2 * 7
  let card_shop_earnings := 12.5 * 2 * 5
  let babysitting_earnings := 10 * 4
  show walking_earnings + card_shop_earnings + babysitting_earnings = 305
  sorry

end Donna_total_earnings_l462_462479


namespace greatest_divisor_arithmetic_sum_l462_462366

theorem greatest_divisor_arithmetic_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ d, d = 15 ∧ ∀ S : ℕ, S = 15 * x + 105 * c → d ∣ S :=
by 
  sorry

end greatest_divisor_arithmetic_sum_l462_462366


namespace number_of_rectangular_arrays_of_chairs_l462_462829

/-- 
Given a classroom that contains 45 chairs, prove that 
the number of rectangular arrays of chairs that can be made such that 
each row contains at least 3 chairs and each column contains at least 3 chairs is 4.
-/
theorem number_of_rectangular_arrays_of_chairs : 
  ∃ (n : ℕ), n = 4 ∧ 
    ∀ (a b : ℕ), (a * b = 45) → 
      (a ≥ 3) → (b ≥ 3) → 
      (n = 4) := 
sorry

end number_of_rectangular_arrays_of_chairs_l462_462829


namespace arithmetic_sequence_problem_l462_462877

theorem arithmetic_sequence_problem 
  (a_n b_n : ℕ → ℕ) 
  (S_n T_n : ℕ → ℕ) 
  (h1: ∀ n, S_n n = (n * (a_n n + a_n (n-1))) / 2)
  (h2: ∀ n, T_n n = (n * (b_n n + b_n (n-1))) / 2)
  (h3: ∀ n, (S_n n) / (T_n n) = (7 * n + 2) / (n + 3)):
  (a_n 4) / (b_n 4) = 51 / 10 := 
sorry

end arithmetic_sequence_problem_l462_462877


namespace unique_representation_l462_462249

theorem unique_representation (N : ℕ) (hN : N > 0) :
  ∃! (k : ℕ) (d : ℕ → ℕ), (∀ i ≤ k, d i = 1 ∨ d i = 2) ∧ N = ∑ i in finset.range (k + 1), (d i) * 2^i := sorry

end unique_representation_l462_462249


namespace conditional_probability_l462_462716

-- Definitions of conditions and events
def P : Set (Set α) → ℝ → Prop := λ X p, sorry  -- Assume existence of P(X, p) representing P(X) = p
def A : Set α := sorry  -- Event A: red light flashing after the first closure
def B : Set α := sorry  -- Event B: red light flashing after the second closure

-- Given conditions
axiom P_A : P A (1 / 2)
axiom P_A_and_B : P (A ∩ B) (1 / 3)

-- Goal statement
theorem conditional_probability : ∃ PB_A, PB_A = (1 / 3) / (1 / 2) → PB_A = 2 / 3 :=
by
  use (1 / 3) / (1 / 2)
  intro h
  rw h
  norm_num

end conditional_probability_l462_462716


namespace unique_real_solution_k_l462_462379

theorem unique_real_solution_k (k : ℝ) :
  ∃! x : ℝ, (3 * x + 8) * (x - 6) = -62 + k * x ↔ k = -10 + 12 * Real.sqrt 1.5 ∨ k = -10 - 12 * Real.sqrt 1.5 := by
  sorry

end unique_real_solution_k_l462_462379


namespace count_valid_numbers_l462_462550

open Nat

-- Definitions corresponding to the conditions in part a
def digits : List ℕ := [0, 1, 2, 3, 4, 5]
def isFourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def hasNoRepeatingDigits (n : ℕ) : Prop := (n.digits : List ℕ).nodup
def isDivisibleBy25 (n : ℕ) : Prop := n % 25 = 0

-- Main statement: prove the count of such numbers is 21
theorem count_valid_numbers : 
  Nat.card {n // isFourDigitNumber n ∧ hasNoRepeatingDigits n ∧ isDivisibleBy25 n} = 21 := sorry

end count_valid_numbers_l462_462550


namespace measure_angledb_l462_462914

-- Definitions and conditions:
def right_triangle (A B C : Type) : Prop :=
  ∃ (a b c : A), ∠A = 45 ∧ ∠B = 45 ∧ ∠C = 90

def angle_bisector (A B : Type) (D : Type) : Prop :=
  ∃ (AD BD : A), AD bisects ∠A ∧ BD bisects ∠B

-- Prove:
theorem measure_angledb (A B C D : Type) 
  (hABC : right_triangle A B C)
  (hAngleBisectors : angle_bisector A B D) : 
  ∠ ADB = 135 := 
sorry

end measure_angledb_l462_462914


namespace order_of_trig_values_l462_462508

-- Define the given conditions
def a : ℝ := Real.tan (70 * Real.pi / 180)
def b : ℝ := Real.sin (25 * Real.pi / 180)
def c : ℝ := Real.cos (25 * Real.pi / 180)

-- Statement of the proof problem
theorem order_of_trig_values : b < c ∧ c < a := 
by
  -- Proof placeholder
  sorry

end order_of_trig_values_l462_462508


namespace greatest_divisor_of_sum_of_arith_seq_l462_462314

theorem greatest_divisor_of_sum_of_arith_seq (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → d ∣ (15 * (x + 7 * c))) ∧
    (∀ k : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → k ∣ (15 * (x + 7 * c))) → k ≤ d) ∧ 
    d = 15 :=
sorry

end greatest_divisor_of_sum_of_arith_seq_l462_462314


namespace hari_digs_well_alone_in_48_days_l462_462213

theorem hari_digs_well_alone_in_48_days :
  (1 / 16 + 1 / 24 + 1 / (Hari_days)) = 1 / 8 → Hari_days = 48 :=
by
  intro h
  sorry

end hari_digs_well_alone_in_48_days_l462_462213


namespace sum_of_integer_solutions_l462_462780

theorem sum_of_integer_solutions : 
  (Finset.sum ((Finset.filter (λ x : ℤ, 1 < (x-3)^2 ∧ (x-3)^2 < 36) (Finset.Icc (-100) 100).val).map (λ x, x)).val) = 30 := 
  sorry

end sum_of_integer_solutions_l462_462780


namespace decreasing_interval_l462_462991

-- Definitions based on the given conditions
def t (x : ℝ) : ℝ := x^2 - 5 * x - 6
def y (x : ℝ) : ℝ := 2 ^ t x

-- Statement to prove the function y is monotonically decreasing
theorem decreasing_interval : ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ ≤ 5/2 → y x₁ ≥ y x₂ := by
  sorry

end decreasing_interval_l462_462991


namespace find_pairs_satisfying_conditions_l462_462072

theorem find_pairs_satisfying_conditions (x y : ℝ) :
    abs (x + y) = 3 ∧ x * y = -10 →
    (x = 5 ∧ y = -2) ∨ (x = -2 ∧ y = 5) ∨ (x = 2 ∧ y = -5) ∨ (x = -5 ∧ y = 2) :=
by
  sorry

end find_pairs_satisfying_conditions_l462_462072


namespace greatest_divisor_of_sum_of_arith_seq_l462_462309

theorem greatest_divisor_of_sum_of_arith_seq (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → d ∣ (15 * (x + 7 * c))) ∧
    (∀ k : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → k ∣ (15 * (x + 7 * c))) → k ≤ d) ∧ 
    d = 15 :=
sorry

end greatest_divisor_of_sum_of_arith_seq_l462_462309


namespace complex_identity_l462_462503

theorem complex_identity : (2 / (1 + Complex.i)^2) = -Complex.i :=
by
  sorry

end complex_identity_l462_462503


namespace det_matrixB_eq_neg_one_l462_462932

variable (x y : ℝ)

def matrixB : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![x, 3],
  ![-4, y]
]

theorem det_matrixB_eq_neg_one 
  (h : matrixB x y - (matrixB x y)⁻¹ = 2 • (1 : Matrix (Fin 2) (Fin 2) ℝ)) :
  Matrix.det (matrixB x y) = -1 := sorry

end det_matrixB_eq_neg_one_l462_462932


namespace line_segment_AB_passes_through_focus_l462_462722

theorem line_segment_AB_passes_through_focus
  (p : ℝ) (h_pos : p > 0)
  (A B : ℝ × ℝ) (h_parabola_A : A.2^2 = 2 * p * A.1) (h_parabola_B : B.2^2 = 2 * p * B.1)
  (h_diff_quadrant : A.1 * B.1 < 0)
  (N : ℝ × ℝ) (h_directrix_N : N = (0, -p / 2))
  (h_angle_bisected : ∃ k : ℝ, A.2 / (A.1 - k) = -B.2 / (B.1 - k))
  : ∃ F : ℝ × ℝ, (F.1 = p / 2 ∧ F.2 = 0) ∧ line_through A B F :=
sorry

end line_segment_AB_passes_through_focus_l462_462722


namespace splitting_divisible_by_3_l462_462023

def number_of_ways : ℕ → ℕ
| 0     := 1
| 2     := 3
| 4     := 11
| (n+6) := 4 * number_of_ways n - number_of_ways (n+2)

/-- Prove that the number of ways of splitting is divisible by 3 -/ 
theorem splitting_divisible_by_3 :
  number_of_ways 200 % 3 = 0 :=
sorry

end splitting_divisible_by_3_l462_462023


namespace evaluate_f_at_2_l462_462455

-- Define the polynomial function f(x)
def f (x : ℝ) : ℝ := 3 * x^6 - 2 * x^5 + x^3 + 1

theorem evaluate_f_at_2 : f 2 = 34 :=
by
  -- Insert proof here
  sorry

end evaluate_f_at_2_l462_462455


namespace sandwiches_difference_l462_462632

-- Define the number of sandwiches Samson ate at lunch on Monday
def sandwichesLunchMonday : ℕ := 3

-- Define the number of sandwiches Samson ate at dinner on Monday (twice as many as lunch)
def sandwichesDinnerMonday : ℕ := 2 * sandwichesLunchMonday

-- Define the total number of sandwiches Samson ate on Monday
def totalSandwichesMonday : ℕ := sandwichesLunchMonday + sandwichesDinnerMonday

-- Define the number of sandwiches Samson ate for breakfast on Tuesday
def sandwichesBreakfastTuesday : ℕ := 1

-- Define the total number of sandwiches Samson ate on Tuesday
def totalSandwichesTuesday : ℕ := sandwichesBreakfastTuesday

-- Define the number of more sandwiches Samson ate on Monday than on Tuesday
theorem sandwiches_difference : totalSandwichesMonday - totalSandwichesTuesday = 8 :=
by
  sorry

end sandwiches_difference_l462_462632


namespace integer_roots_iff_floor_square_l462_462927

variable (α β : ℝ)
variable (m n : ℕ)
variable (real_roots : α^2 - m*α + n = 0 ∧ β^2 - m*β + n = 0)

noncomputable def are_integers (α β : ℝ) : Prop := (∃ (a b : ℤ), α = a ∧ β = b)

theorem integer_roots_iff_floor_square (m n : ℕ) (α β : ℝ)
  (hmn : 0 ≤ m ∧ 0 ≤ n)
  (roots_real : α^2 - m*α + n = 0 ∧ β^2 - m*β + n = 0) :
  (are_integers α β) ↔ (∃ k : ℤ, (⌊m * α⌋ + ⌊m * β⌋) = k^2) :=
sorry

end integer_roots_iff_floor_square_l462_462927


namespace min_dist_PQ_l462_462182

-- Define circles C1 and C2
def C1 (P : ℝ × ℝ) : Prop := (P.1 - 2)^2 + (P.2 - 2)^2 = 1
def C2 (Q : ℝ × ℝ) : Prop := (Q.1 + 2)^2 + (Q.2 + 1)^2 = 4

-- Define the distance between two points
def dist (P Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Prove the minimum value of the distance between any points P on C1 and Q on C2
theorem min_dist_PQ : ∀ P Q, C1 P → C2 Q → dist P Q = 2 :=
by
  intros P Q hP hQ
  sorry

end min_dist_PQ_l462_462182


namespace valid_parameterizations_l462_462993

noncomputable def is_scalar_multiple (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

def lies_on_line (p : ℝ × ℝ) (m b : ℝ) : Prop :=
  p.2 = m * p.1 + b

def is_valid_parameterization (p d : ℝ × ℝ) (m b : ℝ) : Prop :=
  lies_on_line p m b ∧ is_scalar_multiple d (2, 1)

theorem valid_parameterizations :
  (is_valid_parameterization (7, 18) (-1, -2) 2 4) ∧
  (is_valid_parameterization (1, 6) (5, 10) 2 4) ∧
  (is_valid_parameterization (2, 8) (20, 40) 2 4) ∧
  ¬ (is_valid_parameterization (-4, -4) (1, -1) 2 4) ∧
  ¬ (is_valid_parameterization (-3, -2) (0.5, 1) 2 4) :=
by {
  sorry
}

end valid_parameterizations_l462_462993


namespace diamonds_in_G5_l462_462463

-- Define the sequence of figures G_n
def G : ℕ → ℕ
| 1     := 1
| 2     := 9
| 3     := 21
| n + 1 := G n + 4 * (2 * (G n - G (n - 1)) + 2) if n = 2
                     else G n + 4 * (2 * ((G n - G (n - 1)) / 4) + (G n - G (n - 1)) / 4)

-- Test if G5 equals 129
theorem diamonds_in_G5 : G 5 = 129 := by
  sorry

end diamonds_in_G5_l462_462463


namespace decreasing_intervals_max_min_values_l462_462869

noncomputable def f (x : ℝ) : ℝ := Math.sin (2 * x) + Math.cos (2 * x) + 1

theorem decreasing_intervals (k : ℤ) :
  ∃ a b : ℝ, (a = k * Real.pi + Real.pi / 8) ∧ (b = k * Real.pi + 5 * Real.pi / 8) ∧ 
  ∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x :=
sorry

theorem max_min_values :
  (∃ x : ℝ, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 ∧ f x = 0 ∧ x = -Real.pi / 4) ∧
  (∃ x : ℝ, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 ∧ f x = 1 + Real.sqrt 2 ∧ x = Real.pi / 8) :=
sorry

end decreasing_intervals_max_min_values_l462_462869


namespace equipment_total_cost_l462_462679

def cost_jersey : ℝ := 25
def cost_shorts : ℝ := 15.20
def cost_socks : ℝ := 6.80
def cost_cleats : ℝ := 40
def cost_water_bottle : ℝ := 12
def cost_one_player := cost_jersey + cost_shorts + cost_socks + cost_cleats + cost_water_bottle
def num_players : ℕ := 25
def total_cost_for_team : ℝ := cost_one_player * num_players

theorem equipment_total_cost :
  total_cost_for_team = 2475 := by
  sorry

end equipment_total_cost_l462_462679


namespace final_amount_after_45_days_l462_462411

-- The following definitions are given according to the conditions in the problem.
def initial_amount := 350
def evaporation_rate := 1
def refill_rate := 0.4
def duration := 45

-- The goal is to prove that the final amount of water in the trough after 45 days is 323 gallons.
theorem final_amount_after_45_days :
  initial_amount - duration * (evaporation_rate - refill_rate) = 323 :=
by
  sorry

end final_amount_after_45_days_l462_462411


namespace time_to_paint_wall_l462_462015

noncomputable def paint_time (l : ℝ) (d : ℝ) (L : ℝ) (v : ℝ) : ℝ :=
  let r := d / 2
  let area_per_revolution := 2 * Real.pi * r * l
  let area_per_second := area_per_revolution * v
  let total_wall_area := L^2
  total_wall_area / area_per_second

theorem time_to_paint_wall :
  paint_time 20 15 300 2 ≈ 47.7 :=
by
  let radius := 15 / 2
  let lateral_area := 2 * Real.pi * radius * 20
  let area_per_second := lateral_area * 2
  let wall_area := 300^2
  let time_required := wall_area / area_per_second
  have h : time_required ≈ 47.7 := by norm_num
  exact h
  sorry

end time_to_paint_wall_l462_462015


namespace sufficient_but_not_necessary_condition_l462_462128

theorem sufficient_but_not_necessary_condition (a : ℝ) (h : a ≠ 0) :
  (a > 2 ↔ |a - 1| > 1) ↔ (a > 2 → |a - 1| > 1) ∧ (a < 0 → |a - 1| > 1) ∧ (∃ x : ℝ, (|x - 1| > 1) ∧ x < 0 ∧ x ≠ a) :=
by
  sorry

end sufficient_but_not_necessary_condition_l462_462128


namespace geometric_sequence_vertex_property_l462_462129

theorem geometric_sequence_vertex_property (a b c d : ℝ) 
  (h_geom : ∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r)
  (h_vertex : b = 1 ∧ c = 2) : a * d = b * c :=
by sorry

end geometric_sequence_vertex_property_l462_462129


namespace is_linear_equation_with_one_var_l462_462381

-- Definitions
def eqA := ∀ (x : ℝ), x^2 + 1 = 5
def eqB := ∀ (x y : ℝ), x + 2 = y - 3
def eqC := ∀ (x : ℝ), 1 / (2 * x) = 10
def eqD := ∀ (x : ℝ), x = 4

-- Theorem stating which equation represents a linear equation in one variable
theorem is_linear_equation_with_one_var : eqD :=
by
  -- Proof skipped
  sorry

end is_linear_equation_with_one_var_l462_462381


namespace elle_practices_hours_l462_462109

variable (practice_time_weekday : ℕ) (days_weekday : ℕ) (multiplier_saturday : ℕ) (minutes_in_an_hour : ℕ) 
          (total_minutes_weekdays : ℕ) (total_minutes_saturday : ℕ) (total_minutes_week : ℕ) (total_hours : ℕ)

theorem elle_practices_hours :
  practice_time_weekday = 30 ∧
  days_weekday = 5 ∧
  multiplier_saturday = 3 ∧
  minutes_in_an_hour = 60 →
  total_minutes_weekdays = practice_time_weekday * days_weekday →
  total_minutes_saturday = practice_time_weekday * multiplier_saturday →
  total_minutes_week = total_minutes_weekdays + total_minutes_saturday →
  total_hours = total_minutes_week / minutes_in_an_hour →
  total_hours = 4 :=
by
  intros
  sorry

end elle_practices_hours_l462_462109


namespace playerA_winning_conditions_l462_462514

def playerA_has_winning_strategy (n : ℕ) : Prop :=
  (n % 4 = 0) ∨ (n % 4 = 3)

theorem playerA_winning_conditions (n : ℕ) (h : n ≥ 2) : 
  playerA_has_winning_strategy n ↔ (n % 4 = 0 ∨ n % 4 = 3) :=
by sorry

end playerA_winning_conditions_l462_462514


namespace chuck_team_score_proof_chuck_team_score_l462_462082

-- Define the conditions
def yellow_team_score : ℕ := 55
def lead : ℕ := 17

-- State the main proposition
theorem chuck_team_score (yellow_team_score : ℕ) (lead : ℕ) : ℕ :=
yellow_team_score + lead

-- Formulate the final proof goal
theorem proof_chuck_team_score : chuck_team_score yellow_team_score lead = 72 :=
by {
  -- This is the place where the proof should go
  sorry
}

end chuck_team_score_proof_chuck_team_score_l462_462082


namespace determine_x_value_l462_462074

theorem determine_x_value (a b c x : ℕ) (h1 : x = a + 7) (h2 : a = b + 12) (h3 : b = c + 25) (h4 : c = 95) : x = 139 := by
  sorry

end determine_x_value_l462_462074


namespace functional_eq_solution_l462_462786

theorem functional_eq_solution (f : ℝ → ℝ) 
  (h_diff : ∀ t, differentiable ℝ (λ x, f x)) 
  (h_diff2 : ∀ t, continuous (λ x, (differentiable 2 ℝ (λ x, f x))) || f'' x ≠ none) 
  (h_eq : ∀ t, f(t)^2 = f(t * real.sqrt 2)) : 
  ∃ c : ℝ, ∀ x, f(x) = real.exp (c * x^2) :=
sorry

end functional_eq_solution_l462_462786


namespace max_sum_of_sines_l462_462796

theorem max_sum_of_sines (A B C : ℝ) (h₁ : A + B + C = Real.pi) (h₂ : 0 < A) (h₃ : A < Real.pi) (h₄ : 0 < B) (h₅ : B < Real.pi) (h₆ : 0 < C) (h₇ : C < Real.pi) :
  (Real.sin A + Real.sin B + Real.sin C) ≤ (3 * Real.sqrt(3) / 2) ∧ (A = B ∧ B = C → Real.sin A + Real.sin B + Real.sin C = 3 * Real.sqrt(3) / 2) :=
by
  sorry

end max_sum_of_sines_l462_462796


namespace modulus_of_z_l462_462185

noncomputable def z (h : z * (1 + complex.I) = 4 - 2 * complex.I) : ℂ := 
  (4 - 2 * complex.I) / (1 + complex.I)

theorem modulus_of_z (h : z * (1 + complex.I) = 4 - 2 * complex.I) : complex.abs z = real.sqrt 10 :=
sorry

end modulus_of_z_l462_462185


namespace find_reflection_point_of_light_l462_462417

variables (A C B : ℝ × ℝ × ℝ)
variables (n : ℝ × ℝ × ℝ)
variable  (plane : ℝ → ℝ → ℝ → Prop)

def is_reflection (A B C : ℝ × ℝ × ℝ) (plane : ℝ → ℝ → ℝ → Prop) : Prop := 
  let D := (2 * ((B.1 - A.1) + (B.2 - A.2) + (B.3 - A.3))) in
  plane D.1 D.2 D.3 ∧ (C = (-2 * D.1 + A.1, -2 * D.2 + A.2, -2 * D.3 + A.3))

noncomputable def plane_eq := λ (x y z : ℝ), x + y + z = 10

theorem find_reflection_point_of_light :
  A = (-2, 8, 10) →
  C = (4, 4, 8) →
  n = (1, 1, 1) →
  plane = plane_eq →
  is_reflection A B C plane →
  B = (70 / 29, 61 / 29, 130 / 29) :=
sorry

end find_reflection_point_of_light_l462_462417


namespace no_integers_satisfy_PP_x_eq_x_squared_l462_462622

-- Define the polynomial P(x) with integer coefficients
def P (x : ℤ) : ℤ := sorry

-- The given conditions
axiom h1 : P (-1) = -4
axiom h2 : P (-3) = -40
axiom h3 : P (-5) = -156

-- The main statement we want to prove
theorem no_integers_satisfy_PP_x_eq_x_squared : ∃ (n : ℕ), ∀ (x : ℤ), n = 0 ∧ (P (P x) = x^2 → false) :=
by
  use 0
  intros x h
  cases h
  sorry

end no_integers_satisfy_PP_x_eq_x_squared_l462_462622


namespace greatest_common_divisor_sum_arithmetic_sequence_l462_462345

theorem greatest_common_divisor_sum_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, d = 15 ∧ ∀ (n : ℕ), n = 15 → ∀ k : ℕ, k = 15 ∧ 15 ∣ (15 * (x + 7 * c)) :=
by
  sorry

end greatest_common_divisor_sum_arithmetic_sequence_l462_462345


namespace angle_DAB_in_regular_hexagon_is_60_degrees_l462_462580

noncomputable def regular_hexagon_internal_angle (n : ℕ) : ℝ :=
  if n = 6 then 120 else 0 -- Regular hexagon internal angle condition

theorem angle_DAB_in_regular_hexagon_is_60_degrees :
  (regular_hexagon_internal_angle 6 = 120) →
  ∀ (A B C D E F : Type) (AD : AD_diagonal),
  -- Conditions: ABCDEF is a regular hexagon and drawing diagonal AD
  let ABCDEF := regular_hexagon A B C D E F in
  -- Question and its proven answer
  ∃ (DAB : ℝ), DAB = 60 :=
begin
  intros,
  sorry -- Proof to be filled in
end

end angle_DAB_in_regular_hexagon_is_60_degrees_l462_462580


namespace range_of_a_l462_462186

theorem range_of_a (a : ℝ) :
  (∀ (x y : ℝ), 3 * a * x + (a^2 - 3 * a + 2) * y - 9 < 0 → (3 * a * x + (a^2 - 3 * a + 2) * y - 9 = 0 → y > 0)) ↔ (1 < a ∧ a < 2) :=
by
  sorry

end range_of_a_l462_462186


namespace find_a_plus_c_l462_462274

def parabola_intersection_problem (a b c d : ℝ) : Prop :=
  (∀ x : ℝ, y = -|x - a|^2 + b → y = 8 → x = 1) ∧
  (∀ x : ℝ, y = -|x - a|^2 + b → y = 4 → x = 9) ∧
  (∀ x : ℝ, y = |x - c|^2 + d → y = 8 → x = 1) ∧
  (∀ x : ℝ, y = |x - c|^2 + d → y = 4 → x = 9)

theorem find_a_plus_c (a b c d : ℝ) (h : parabola_intersection_problem a b c d) : a + c = 10 :=
sorry

end find_a_plus_c_l462_462274


namespace proof_problem_l462_462158

-- Definitions for propositions p and q
def is_obtuse_triangle (A B : ℝ) : Prop :=
  π > A + B ∧ A > π / 2 - B ∧ A > 0

def p : Prop :=
  ∀ (A B C : ℝ), is_obtuse_triangle A B → sin A < cos B

def q : Prop :=
  ∀ (x y : ℝ), x + y ≠ 2 → x ≠ -1 ∨ y ≠ 3

-- Theorem statement
theorem proof_problem : ¬p ∧ q :=
by
  sorry

end proof_problem_l462_462158


namespace four_digit_numbers_count_l462_462999

theorem four_digit_numbers_count : 
  (∀ d1 d2 d3 d4 : Fin 4, 
    (d1 = 1 ∨ d1 = 2 ∨ d1 = 3) ∧ 
    d2 ≠ d1 ∧ d2 ≠ 0 ∧ 
    d3 ≠ d1 ∧ d3 ≠ d2 ∧ 
    d4 ≠ d1 ∧ d4 ≠ d2 ∧ d4 ≠ d3) →
  3 * 6 = 18 := 
by
  sorry

end four_digit_numbers_count_l462_462999


namespace sector_area_l462_462837

-- Definitions of conditions
def arc_length : ℝ := 28 -- given arc length in cm
def central_angle : ℝ := 240 -- given central angle in degrees

-- Radius calculated from arc length and central angle
def radius : ℝ := arc_length / (central_angle / 360 * 2 * Real.pi)

-- Theorem to prove the area of the sector given the conditions
theorem sector_area : (1/2 * arc_length * radius = 294 / Real.pi) :=
sorry

end sector_area_l462_462837


namespace greatest_common_divisor_sum_arithmetic_sequence_l462_462344

theorem greatest_common_divisor_sum_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, d = 15 ∧ ∀ (n : ℕ), n = 15 → ∀ k : ℕ, k = 15 ∧ 15 ∣ (15 * (x + 7 * c)) :=
by
  sorry

end greatest_common_divisor_sum_arithmetic_sequence_l462_462344


namespace sum_of_100th_bracket_l462_462954

theorem sum_of_100th_bracket : 
  let seq := (λ n, 3 + 2 * n) in
  let bracket (k : ℕ) := 
    if k % 4 = 0 then {seq (4 * k - 3), seq (4 * k - 2), seq (4 * k - 1), seq (4 * k)} else
    if k % 4 = 1 then {seq (4 * k - 3)} else
    if k % 4 = 2 then {seq (4 * k - 4), seq (4 * k - 3)} else
    {seq (4 * k - 5), seq (4 * k - 4), seq (4 * k - 3)} in
  let sum_bracket (nums : set ℕ) := nums.foldl (λ (x y : ℕ), x + y) 0 in
  sum_bracket (bracket 100) = 1992 :=
by sorry

end sum_of_100th_bracket_l462_462954


namespace intersection_M_N_l462_462235

def M : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ x^2 + y^2 = 1 }
def N : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ y = x^2 }

theorem intersection_M_N :
  { p : ℝ × ℝ | (p ∈ M) ∧ (p ∈ N) } = { p : ℝ × ℝ | ∃ y, y ∈ (Icc 0 1) ∧ p = (sqrt y, y) ∨ p = (-sqrt y, y) } :=
by
  sorry

end intersection_M_N_l462_462235


namespace common_chord_length_l462_462304

theorem common_chord_length (r d c : ℝ) (hr : r = 15) (hd : d = 25) :
  c ≈ 17 := sorry

end common_chord_length_l462_462304


namespace remainder_4_exp_4_exp_4_exp_4_mod_500_l462_462454

theorem remainder_4_exp_4_exp_4_exp_4_mod_500 :
  (4 ^ 4 ^ 4 ^ 4) % 500 = 36 :=
by
  sorry

end remainder_4_exp_4_exp_4_exp_4_mod_500_l462_462454


namespace greatest_divisor_arithmetic_sum_l462_462372

theorem greatest_divisor_arithmetic_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ d, d = 15 ∧ ∀ S : ℕ, S = 15 * x + 105 * c → d ∣ S :=
by 
  sorry

end greatest_divisor_arithmetic_sum_l462_462372


namespace radius_middle_circle_l462_462098

theorem radius_middle_circle 
  {r₁ r₂ r₃ r₄ r₅ : ℝ} 
  (h_r₁ : r₁ = 6) 
  (h_r₅ : r₅ = 20) 
  (h_seq : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 5 → r₁ + (j - i) * (r₂ - r₁) = r_{j}) 
  : r₃ = 13 :=
  sorry

end radius_middle_circle_l462_462098


namespace trapezoid_sides_l462_462923

-- Definitions of the given conditions in Lean 4
variables {A B C D : Type*} [normed_group A] [normed_space ℝ A] [inner_product_space ℝ A]
variables (a b : ℝ) 

-- The theorem statement
theorem trapezoid_sides :
  ∀ (AD BD BC : ℝ), -- lengths of the sides
  AD = a → BC = b → -- given lengths of bases
  ∃ (AB CD : ℝ), 
  (BD ^ 2 = BC * AD) → -- derived BD from similarity
  (AB = sqrt(a * (a + b))) ∧ (CD = sqrt(b * (a + b))) := 
begin
  sorry
end

end trapezoid_sides_l462_462923


namespace three_circles_cover_horizon_two_circles_cannot_cover_horizon_l462_462210

/-- A point outside three non-overlapping non-touching circles can see the entire horizon -/
theorem three_circles_cover_horizon (C1 C2 C3 : Set Point) (P : Point) :
  ∀ θ : ℝ, ∃ r : ℝ, ∀ (ray : ℝ → Point), (θ = ray 1 → ray r ∈ C1 ∪ C2 ∪ C3) := sorry

/-- A point outside two non-overlapping non-touching circles cannot see the entire horizon -/
theorem two_circles_cannot_cover_horizon (C1 C2 : Set Point) (P : Point) :
  ∃ θ : ℝ, ∀ r : ℝ, (∀ (ray : ℝ → Point), ¬(θ = ray 1 → ray r ∈ C1 ∪ C2)) := sorry

end three_circles_cover_horizon_two_circles_cannot_cover_horizon_l462_462210


namespace probability_diff_greater_than_one_eq_seven_eighths_l462_462644

noncomputable def prob_greater_than_one : ℝ :=
  let heads_p := 1 / 2
  let tails_p := 1 / 2
  let uniform_dist := @MeasureTheory.Measure.uniform _ _ LinearOrder.IntervalOrdering.NonnegOrderedAddCommGroup.interval ∅ set.Icc 0 2
  let coin_flip (p : ℝ) := (ennreal.of_real p) * uniform_dist

  -- Event: Both flips are tails; both numbers chosen uniformly from [0,2]
  let case1 := (tails_p) ^ 2 * ∫ x in 0..2, ∫ y in 0..2, if (x - y > 1) then 1 else 0

  -- Event: First flip tails for y, second flip heads-tails (prob chosen 2)
  let case2 := (tails_p) * heads_p * (∫ y in 0..2, if (2 - y > 1) then 1 else 0)

  -- Event: First flip tails for x, second flip heads-heads (prob chosen 0)
  let case3 := (heads_p * tails_p) * (∫ x in 0..2, if (x - 0 > 1) then 1 else 0)

  -- Event: First flip heads-tails for both x and y; chose 2 and 0
  let case4 := (heads_p) ^ 2

  -- Sum of all cases for P(x - y > 1)
  let prob_xy_diff_gt_1 := case1 + case2 + case3 + case4
  -- P(|x - y| > 1) is twice that due to symmetry
  2 * prob_xy_diff_gt_1

theorem probability_diff_greater_than_one_eq_seven_eighths :
  prob_greater_than_one = 7 / 8 :=
sorry

end probability_diff_greater_than_one_eq_seven_eighths_l462_462644


namespace continuous_limit_at_x_eq_1_l462_462781

theorem continuous_limit_at_x_eq_1 :
  ∀ (f : ℝ → ℝ),(∀ x, f x = (x^3 - 1) / (x^2 - 1)) → 
  tendsto f (𝓝[≠] 1) (𝓝 (3 / 2)) :=
by {
    intro f,
    intro hf,
    sorry
  }

end continuous_limit_at_x_eq_1_l462_462781


namespace anne_wandered_hours_l462_462747

noncomputable def speed : ℝ := 2 -- miles per hour
noncomputable def distance : ℝ := 6 -- miles

theorem anne_wandered_hours (t : ℝ) (h : distance = speed * t) : t = 3 := by
  sorry

end anne_wandered_hours_l462_462747


namespace jacob_has_winning_strategy_l462_462594

def jacob_winning_strategy (m n : ℕ) : Prop :=
  (m ≠ n → ∃ strategy_for_jacob : (ℕ × ℕ) → (ℕ × ℕ), 
      ∀ pos : ℕ × ℕ, strategy_for_jacob pos ∈ ({(i, j) : ℕ × ℕ | i <= m ∧ j <= n}))

theorem jacob_has_winning_strategy (m n : ℕ) : jacob_winning_strategy m n ↔ m ≠ n :=
begin
  sorry
end

end jacob_has_winning_strategy_l462_462594


namespace monotonically_increasing_interval_l462_462152

noncomputable def f (x : ℝ) : ℝ := Real.log (-3 * x^2 + 4 * x + 4)

theorem monotonically_increasing_interval :
  ∀ x, x ∈ Set.Ioc (-2/3 : ℝ) (2/3 : ℝ) → MonotoneOn f (Set.Ioc (-2/3) (2/3)) :=
sorry

end monotonically_increasing_interval_l462_462152


namespace mapping_problem_l462_462663

open Set

noncomputable def f₁ (x : ℝ) : ℝ := Real.sqrt x
noncomputable def f₂ (x : ℝ) : ℝ := 1 / x
def f₃ (x : ℝ) : ℝ := x^2 - 2
def f₄ (x : ℝ) : ℝ := x^2

def A₁ : Set ℝ := {1, 4, 9}
def B₁ : Set ℝ := {-3, -2, -1, 1, 2, 3}
def A₂ : Set ℝ := univ
def B₂ : Set ℝ := univ
def A₃ : Set ℝ := univ
def B₃ : Set ℝ := univ
def A₄ : Set ℝ := {-1, 0, 1}
def B₄ : Set ℝ := {-1, 0, 1}

theorem mapping_problem : 
  ¬ (∀ x ∈ A₁, f₁ x ∈ B₁) ∧
  ¬ (∀ x ∈ A₂, x ≠ 0 → f₂ x ∈ B₂) ∧
  (∀ x ∈ A₃, f₃ x ∈ B₃) ∧
  (∀ x ∈ A₄, f₄ x ∈ B₄) :=
by
  sorry

end mapping_problem_l462_462663


namespace average_is_0_1667X_plus_3_l462_462770

noncomputable def average_of_three_numbers (X Y Z : ℝ) : ℝ := (X + Y + Z) / 3

theorem average_is_0_1667X_plus_3 (X Y Z : ℝ) 
  (h1 : 2001 * Z - 4002 * X = 8008) 
  (h2 : 2001 * Y + 5005 * X = 10010) : 
  average_of_three_numbers X Y Z = 0.1667 * X + 3 := 
sorry

end average_is_0_1667X_plus_3_l462_462770


namespace cubed_identity_l462_462169

theorem cubed_identity (x : ℝ) (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := 
sorry

end cubed_identity_l462_462169


namespace monday_time_increase_25_percent_l462_462387

variable (x : ℝ) (h1 : x ≠ 0) 

def T_sunday := (64 / x)
def T_monday1 := (32 / (2 * x))
def T_monday2 := (32 / (x / 2))
def T_monday := T_monday1 + T_monday2
def percent_increase := ((T_monday - T_sunday) / T_sunday) * 100

theorem monday_time_increase_25_percent :
    percent_increase x h1 = 25 := by
  sorry

end monday_time_increase_25_percent_l462_462387


namespace shortest_altitude_right_triangle_l462_462062

theorem shortest_altitude_right_triangle (a b c : ℕ) (h1: a = 8) (h2: b = 15) (h3: c = 17) 
  (h4: a ^ 2 + b ^ 2 = c ^ 2) : 
  let area := (1 / 2 : ℚ) * (a : ℚ) * (b : ℚ),
      hypotenuse := (c : ℚ),
      altitude := (2 * area) / hypotenuse in
  altitude = 120 / 17 :=
by
  sorry

end shortest_altitude_right_triangle_l462_462062


namespace neg_p_l462_462156

namespace Negation

open Classical

variable (x : ℝ)

def p : Prop := ∃ x : ℝ, x^2 - 3 * x + 2 = 0

theorem neg_p : ¬p ↔ ∀ x : ℝ, x^2 - 3 * x + 2 ≠ 0 := by
  simp [p]
  sorry

end Negation

end neg_p_l462_462156


namespace smallest_integer_l462_462626

theorem smallest_integer (
  n : ℕ)
  (h1 : n > 50)
  (h2 : ¬ (∃ k : ℕ, k * k = n))
  (h3 : ∃ k : ℕ, 3 * k = n)
  (h4 : even (nat.divisors n).length)
  : n = 51 :=
sorry

end smallest_integer_l462_462626


namespace length_of_first_platform_l462_462438

noncomputable def speed (distance time : ℕ) :=
  distance / time

theorem length_of_first_platform 
  (L : ℕ) (train_length : ℕ) (time1 time2 : ℕ) (platform2_length : ℕ) (speed : ℕ) 
  (H1 : L + train_length = speed * time1) 
  (H2 : platform2_length + train_length = speed * time2) 
  (train_length_eq : train_length = 30) 
  (time1_eq : time1 = 12) 
  (time2_eq : time2 = 15) 
  (platform2_length_eq : platform2_length = 120) 
  (speed_eq : speed = 10) : L = 90 :=
by
  sorry

end length_of_first_platform_l462_462438


namespace bounded_int_exists_l462_462652

noncomputable def exists_ints_bounded (x : ℕ → ℝ) (n : ℕ) (k : ℤ) : Prop :=
  ∃ (a : ℕ → ℤ), (∀ i : ℕ, i < n → |a i| ≤ k - 1) ∧ 
  (∀ i : ℕ, i < n → a i ≠ 0) ∧ (|∑ i in Finset.range n, a i * x i| ≤ (k - 1) * Real.sqrt n / (k ^ n - 1))

theorem bounded_int_exists {n : ℕ} {k : ℤ} (x : ℕ → ℝ) 
  (hx : (∑ i in Finset.range n, x i ^ 2 = 1)) (hk : k ≥ 2) : exists_ints_bounded x n k :=
sorry

end bounded_int_exists_l462_462652


namespace maker_wins_if_grid_is_odd_and_large_l462_462623

def is_odd (x : ℕ) : Prop := x % 2 = 1

theorem maker_wins_if_grid_is_odd_and_large (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) (hmo : is_odd m) (hno : is_odd n) :
  ∃ y, ∀ x, (x ≤ m → y ≤ n) → green_block_at (x, y) :=
sorry

end maker_wins_if_grid_is_odd_and_large_l462_462623


namespace measure_of_obtuse_angle_ADB_l462_462904

-- Define the right triangle ABC with specific angles
def triangle_ABC (A B C D : Type) [AddAngle A B C] :=
  (rightTriangle : (A B C) ∧ 
   angle_A_is_45_degrees : (angle A = 45) ∧ 
   angle_B_is_45_degrees : (angle B = 45) ∧ 
   AD_bisects_A : (bisects A D) ∧ 
   BD_bisects_B : (bisects B D))

-- Statement of the proof problem
theorem measure_of_obtuse_angle_ADB {A B C D : Type} [AddAngle A B C] 
  (h_ABC : triangle_ABC A B C) : 
  measure_obtuse_angle A B D ADB = 135 :=
sorry

end measure_of_obtuse_angle_ADB_l462_462904


namespace max_number_plate_upside_down_l462_462989

-- Define the condition for a character looking the same when upside down
def is_symmetric_upside_down (c : Char) : Prop :=
  c = '0' ∨ c = '1' ∨ c = '8' ∨ c = 'H' ∨ c = 'O' ∨ c = 'X'

-- Define the condition for a character converting to another valid character when upside down
def is_valid_upside_down (c d : Char) : Prop :=
  (c = '6' ∧ d = '9') ∨ (c = '9' ∧ d = '6') ∨ (is_symmetric_upside_down c ∧ c = d)

-- Define a function to check if a string satisfies the upside-down condition
def valid_number_plate (s : String) : Prop :=
  s.length = 8 ∧ 
  is_valid_upside_down (s.get 0) (s.get 7) ∧
  is_valid_upside_down (s.get 1) (s.get 6) ∧
  is_valid_upside_down (s.get 2) (s.get 5) ∧
  is_valid_upside_down (s.get 3) (s.get 4)

-- Specify the problem statement
theorem max_number_plate_upside_down : valid_number_plate "60HOH09" :=
by
  unfold valid_number_plate is_valid_upside_down is_symmetric_upside_down
  split; norm_num; tauto
  sorry -- Additional steps needed to fully complete the proof

end max_number_plate_upside_down_l462_462989


namespace alice_additional_plates_l462_462029

theorem alice_additional_plates (initial_stack : ℕ) (first_addition : ℕ) (total_when_crashed : ℕ) 
  (h1 : initial_stack = 27) (h2 : first_addition = 37) (h3 : total_when_crashed = 83) : 
  total_when_crashed - (initial_stack + first_addition) = 19 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end alice_additional_plates_l462_462029


namespace same_function_pairs_l462_462442

noncomputable def f1 (x : ℝ) := Real.sqrt (-2 * x^3)
noncomputable def g1 (x : ℝ) := x * Real.sqrt (-2 * x)

def f2 (x : ℝ) := abs x
def g2 (x : ℝ) := (Real.sqrt x) ^ 2

def f3 (x : ℝ) := x^0
def g3 (x : ℝ) := 1 / x^0

def f4 (x : ℝ) := x^2 - 2 * x - 1
def g4 (t : ℝ) := t^2 - 2 * t - 1

theorem same_function_pairs : 
  (∀ x : ℝ, x ≠ 0 → f3 x = g3 x) ∧ (∀ x : ℝ, f4 x = g4 x) :=
by
  sorry

end same_function_pairs_l462_462442


namespace sqrt_cuberoot_abs_value_sum_l462_462456

theorem sqrt_cuberoot_abs_value_sum :
  (Real.sqrt 9) + (Real.cbrt (-8)) + (abs (Real.sqrt 2 - 1)) = Real.sqrt 2 :=
by
  sorry

end sqrt_cuberoot_abs_value_sum_l462_462456


namespace cosine_difference_l462_462519

theorem cosine_difference (α : ℝ) (h₁ : cos α = 5 / 13) (h₂ : 3 * π / 2 < α ∧ α < 2 * π) :
  cos (α - π / 4) = -7 * real.sqrt 2 / 26 :=
by
  sorry

end cosine_difference_l462_462519


namespace tourist_initial_money_l462_462738

-- Define the amounts left each day recursively
def amount_after_day (initial : ℝ) (day : ℕ) : ℝ :=
  if day = 0 then initial else
  (amount_after_day initial (day - 1)) / 2 - 100

-- The main statement to prove
theorem tourist_initial_money (x : ℝ) :
  amount_after_day x 5 = 0 → x = 6200 :=
by
  sorry

end tourist_initial_money_l462_462738


namespace min_dimensions_l462_462951

noncomputable def width_and_length_min : ℝ × ℝ :=
let w := real.sqrt 250 / real.sqrt 2 in
let l := 2 * w in
(w, l)

theorem min_dimensions :
  ∃ (w l : ℝ), width_and_length_min = (w, l) ∧ (l = 2 * w) ∧ (w * l >= 500) ∧ (w = real.sqrt 250) :=
begin
  use (5 * real.sqrt 10),
  use (10 * real.sqrt 10),
  split,
  { simp [width_and_length_min, (∘)], },
  split,
  { linarith, },
  split,
  { nlinarith [sq (5 * real.sqrt 10), mul_self_le_mul_self_iff, real.sqrt_nonneg], },
  { nlinarith, }
end

end min_dimensions_l462_462951


namespace find_segment_AD_length_l462_462207

noncomputable def segment_length_AD (A B C D X : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace X] :=
  ∃ (angle_BAD angle_ABC angle_BCD : Real)
    (length_AB length_CD : Real)
    (perpendicular : X) (angle_BAX angle_ABX : Real)
    (length_AX length_DX length_AD : Real),
    angle_BAD = 60 ∧
    angle_ABC = 30 ∧
    angle_BCD = 30 ∧
    length_AB = 15 ∧
    length_CD = 8 ∧
    angle_BAX = 30 ∧
    angle_ABX = 60 ∧
    length_AX = length_AB / 2 ∧
    length_DX = length_CD / 2 ∧
    length_AD = length_AX - length_DX ∧
    length_AD = 3.5

theorem find_segment_AD_length (A B C D X : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace X] : segment_length_AD A B C D X :=
by
  sorry

end find_segment_AD_length_l462_462207


namespace max_profit_at_60_l462_462026

variable (x : ℕ) (y W : ℝ)

def charter_fee : ℝ := 15000
def max_group_size : ℕ := 75

def ticket_price (x : ℕ) : ℝ :=
  if x ≤ 30 then 900
  else if 30 < x ∧ x ≤ max_group_size then -10 * (x - 30) + 900
  else 0

def profit (x : ℕ) : ℝ :=
  if x ≤ 30 then 900 * x - charter_fee
  else if 30 < x ∧ x ≤ max_group_size then (-10 * x + 1200) * x - charter_fee
  else 0

theorem max_profit_at_60 : x = 60 → profit x = 21000 := by
  sorry

end max_profit_at_60_l462_462026


namespace normal_distribution_symmetry_l462_462621

open MeasureTheory

variable (a : ℝ)

theorem normal_distribution_symmetry (X : MeasureTheory.ProbabilityDistributions.Normal 3 2) :
  (∀ a : ℝ, Probability (X < 2 * a + 3) = Probability (X > a - 2)) → a = 5 / 3 :=
by
  intros h
  sorry

end normal_distribution_symmetry_l462_462621


namespace vasya_running_time_upwards_l462_462753

noncomputable def time_to_run_up_and_down_upwards (x y : ℝ) (t_stationary t_downward: ℝ) : ℝ :=
    (1 / (x - y) + 1 / ((1 / 2 / x) + y)) * 60

theorem vasya_running_time_upwards :
  ∀ (t_stationary t_downward : ℝ),
    t_stationary = 6 →
    t_downward = 13.5 →
    time_to_run_up_and_down_upwards (1 / 2) (1 / 6) t_stationary t_downward = 324 :=
by
  assume t_stationary t_downward
  assume h1 : t_stationary = 6
  assume h2 : t_downward = 13.5
  sorry

end vasya_running_time_upwards_l462_462753


namespace symmetric_point_l462_462125

theorem symmetric_point (m : ℝ) (hA : (1 : ℝ, 2 : ℝ) ∈ {p : ℝ × ℝ | p.snd = p.fst^2 + 4 * p.fst - m}) :
  ∃ B : ℝ × ℝ, B = (-5, 2) ∧ (B.snd = 2) :=
by 
  use (-5, 2)
  split
  all_goals sorry

end symmetric_point_l462_462125


namespace graph_of_equation_is_two_intersecting_lines_l462_462064

theorem graph_of_equation_is_two_intersecting_lines :
  ∀ x y : ℝ, (x + 3 * y) ^ 3 = x ^ 3 + 9 * y ^ 3 ↔ (x = 0 ∨ y = 0 ∨ x + 3 * y = 0) :=
by
  sorry

end graph_of_equation_is_two_intersecting_lines_l462_462064


namespace greatest_divisor_sum_of_first_fifteen_terms_l462_462323

theorem greatest_divisor_sum_of_first_fifteen_terms 
  (x c : ℕ) (hx : x > 0) (hc : c > 0):
  ∃ d, d = 15 ∧ d ∣ (15*x + 105*c) :=
by
  existsi 15
  split
  . refl
  . apply Nat.dvd.intro
    existsi (x + 7*c)
    refl
  sorry

end greatest_divisor_sum_of_first_fifteen_terms_l462_462323


namespace casey_savings_l462_462055

-- Define the constants given in the problem conditions
def wage_employee_1 : ℝ := 20
def wage_employee_2 : ℝ := 22
def subsidy : ℝ := 6
def hours_per_week : ℝ := 40

-- Define the weekly cost of each employee
def weekly_cost_employee_1 := wage_employee_1 * hours_per_week
def weekly_cost_employee_2 := (wage_employee_2 - subsidy) * hours_per_week

-- Define the savings by hiring the cheaper employee
def savings := weekly_cost_employee_1 - weekly_cost_employee_2

-- Theorem stating the expected savings
theorem casey_savings : savings = 160 := by
  -- Proof is not included
  sorry

end casey_savings_l462_462055


namespace prime_iff_all_coeffs_divisible_l462_462930

theorem prime_iff_all_coeffs_divisible {p a : ℕ} (hp_pos : 0 < p) (ha_pos : 0 < a) (h_coprime : Nat.gcd p a = 1) :
  (Nat.Prime p) ↔ (∀ k : ℕ, k < p → (binom p k * a^k : ℤ) % (p * (a^p : ℤ) = 0 :=
sorry

end prime_iff_all_coeffs_divisible_l462_462930


namespace n1_prime_n2_not_prime_l462_462549

def n1 := 1163
def n2 := 16424
def N := 19101112
def N_eq : N = n1 * n2 := by decide

theorem n1_prime : Prime n1 := 
sorry

theorem n2_not_prime : ¬ Prime n2 :=
sorry

end n1_prime_n2_not_prime_l462_462549


namespace relation_among_abc_l462_462151

theorem relation_among_abc :
  let f (x : ℝ) := x^2 + 5
  let a := f (-Real.log2 5)
  let b := f (Real.log2 3)
  let c := f (-1)
  in a > b ∧ b > c :=
by
  sorry

end relation_among_abc_l462_462151


namespace cross_number_puzzle_hundreds_digit_l462_462765

theorem cross_number_puzzle_hundreds_digit :
  ∃ a b : ℕ, a ≥ 5 ∧ a ≤ 6 ∧ b = 3 ∧ (3^a / 100 = 7 ∨ 7^b / 100 = 7) :=
sorry

end cross_number_puzzle_hundreds_digit_l462_462765


namespace find_a_l462_462871

-- Definition of the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x^2 - 2 * x - 1

-- Definition of the derivative of f(x)
def f_prime (x : ℝ) (a : ℝ) : ℝ := Real.exp x - 2 * a * x - 2

-- Tangent line at the point (1, f(1)) and y-intercept condition
theorem find_a (a : ℝ) : 
  (f_prime 1 a = (f 1 a - (-2)) / (1 - 0)) → 
  (1, f 1 a) ∈ (λ x : ℝ, f_prime 1 a * x + (-2)) → 
  a = -1 := 
by 
  sorry

end find_a_l462_462871


namespace find_matrix_N_l462_462793

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![2, -5],
  ![4, -3]
]

def B : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![-20, -8],
  ![9, 4]
]

def N : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![(46 / 7 : ℚ), (-58 / 7 : ℚ)],
  ![(-43 / 14 : ℚ), (53 / 14 : ℚ)]
]

theorem find_matrix_N : N ⬝ A = B := by
  sorry

end find_matrix_N_l462_462793


namespace pencils_difference_l462_462638

theorem pencils_difference
  (pencils_in_backpack : ℕ := 2)
  (pencils_at_home : ℕ := 15) :
  pencils_at_home - pencils_in_backpack = 13 := by
  sorry

end pencils_difference_l462_462638


namespace mean_three_numbers_l462_462657

open BigOperators

theorem mean_three_numbers (a b c : ℝ) (s : Finset ℝ) (h₀ : s.card = 20)
  (h₁ : (∑ x in s, x) / 20 = 45) 
  (h₂ : (∑ x in s ∪ {a, b, c}, x) / 23 = 50) : 
  (a + b + c) / 3 = 250 / 3 :=
by
  sorry

end mean_three_numbers_l462_462657


namespace selection_methods_l462_462259

-- Conditions
def volunteers : ℕ := 5
def friday_slots : ℕ := 1
def saturday_slots : ℕ := 2
def sunday_slots : ℕ := 1

-- Function to calculate combinatorial n choose k
def choose (n k : ℕ) : ℕ :=
(n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Function to calculate permutations of n P k
def perm (n k : ℕ) : ℕ :=
(n.factorial) / ((n - k).factorial)

-- The target proposition
theorem selection_methods : choose volunteers saturday_slots * perm (volunteers - saturday_slots) (friday_slots + sunday_slots) = 60 :=
by
  -- assumption here leads to the property required, usually this would be more detailed computation.
  sorry

end selection_methods_l462_462259


namespace max_value_of_abs_z_minus_one_l462_462890

noncomputable def max_distance (z : ℂ) : ℝ :=
  complex.abs (z - 1)

theorem max_value_of_abs_z_minus_one :
  ∀ (z : ℂ), complex.abs (z - (4 - 4 * complex.I)) ≤ 2 → max_distance z ≤ 7 :=
by
  sorry

end max_value_of_abs_z_minus_one_l462_462890


namespace modulus_of_Z_l462_462223

open Complex

def given_conditions : Prop :=
  ∃ (Z : ℂ), ∃ (conjZ : ℂ), conjZ = (-1 + Complex.i) * (2 * Complex.i ^ 3 / (1 + Complex.i)) ∧ |Z| = |conjZ|

theorem modulus_of_Z : given_conditions → |Z| = 2 :=
begin
  assume h,
  obtain ⟨Z, conjZ, conjZ_eq, Z_modulus⟩ := h,
  have conjZ_value : conjZ = 2, from by { rw [pow_succ, pow_one, mul_assoc, mul_comm Complex.i, mul_div_cancel', mul_comm, mul_neg_1, mul_comm], ring, field_simp },
  rw conjZ_value at Z_modulus,
  exact Z_modulus,
end

end modulus_of_Z_l462_462223


namespace points_opposite_sides_l462_462530

theorem points_opposite_sides (a : ℝ) :
  (2 * 1 - 3 * (-a) + 1) * (2 * a - 3 * 1 + 1) < 0 ↔ -1 < a ∧ a < 1 :=
by
  calc (2 * 1 - 3 * (-a) + 1) * (2 * a - 3 * 1 + 1)
    = (2 + 3a + 1) * (2a - 3 + 1)                      : by simp
    ... = (3a + 3) * (2a - 2)                          : by simp
    ... = ((3 (a + 1)) * (2 (a - 1)))                  : by simp
    ... = (6 (a + 1) * (a - 1))                        : by ring
    ... = ((6 * a*a) - (1 * 6))                        : by ring
    ... < 0                                           : sorry

end points_opposite_sides_l462_462530


namespace right_angled_triangles_count_l462_462162

theorem right_angled_triangles_count : 
  (∃ (a b : ℕ), a^2 + (sqrt 1001)^2 = b^2 ∧ (b - a) * (b + a) = 1001) → ∃ n, n = 4 :=
by
  sorry

end right_angled_triangles_count_l462_462162


namespace trajectory_of_M_minimum_area_of_triangle_l462_462614

open Real

-- Define the points and necessary conditions
def F := (1 / 2, 0)

axiom A_on_x_axis : ∃ x : ℝ, (x, 0) ∈ ℝ × ℝ
axiom B_on_y_axis : ∀ B : ℝ × ℝ, ¬ (B.1 = 0 → B.2 ≠ 0) → B.2 > 0

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

axiom condition_AM_eq_2AB (A B M : ℝ × ℝ) : M = (2 * B.1 - A.1, 2 * B.2 - A.2)
axiom condition_BA_dot_BF (A B : ℝ × ℝ) (F : ℝ × ℝ) : (A.1 - B.1) * (F.1 - B.1) + (A.2 - B.2) * (F.2 - B.2) = 0

-- Formalization of the theorem for the trajectory E of point M
theorem trajectory_of_M (x y : ℝ) (A B M : ℝ × ℝ) (F := (1 / 2, 0)) :
  A_on_x_axis → B_on_y_axis B → condition_AM_eq_2AB A B M → condition_BA_dot_BF A B F → (y^2 = 2*x) :=
by sorry

-- Define other necessary points and conditions for the second part
axiom P_on_E (P : ℝ × ℝ) : P.2 ^ 2 = 2 * P.1
axiom R_N_on_y_axis : ∀ R N : ℝ × ℝ, R.2 ≠ N.2 → R.1 = 0 ∧ N.1 = 0

-- Formalization of the theorem for the minimum area of triangle PRN
theorem minimum_area_of_triangle (P R N : ℝ × ℝ) :
  P_on_E P → R_N_on_y_axis R N → (∃ x y : ℝ, (x - 1)^2 + y^2 = 1) →
  (∀ x_0 : ℝ, x_0 > 2 → P = (x_0, sqrt (2 * x_0))) →
  ∃ (min_area : ℝ), min_area = 8 :=
by sorry

end trajectory_of_M_minimum_area_of_triangle_l462_462614


namespace greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l462_462363

theorem greatest_divisor_of_sum_first_15_terms_arithmetic_sequence
  (x c : ℕ) -- where x and c are positive integers
  (h_pos_x : 0 < x) -- x is positive
  (h_pos_c : 0 < c) -- c is positive
  : ∃ (d : ℕ), d = 15 ∧ ∀ (S : ℕ), S = 15 * x + 105 * c → d ∣ S :=
by
  sorry

end greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l462_462363


namespace hyperbola_eccentricity_l462_462136

theorem hyperbola_eccentricity (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  let c := (3 * real.sqrt 2) / 4 * a in
  let P := (-c, -b * c / (3 * a)) in
  let F1 := (-c, 0) in
  ∃ e : ℝ, e = (c / a) ∧ e = (3 * real.sqrt 2) / 4 :=
by
  sorry

end hyperbola_eccentricity_l462_462136


namespace vector_satisfies_condition_l462_462468

def line_l (t : ℝ) : ℝ × ℝ := (2 + 3 * t, 5 + 2 * t)
def line_m (s : ℝ) : ℝ × ℝ := (1 + 2 * s, 3 + 2 * s)

variable (A B P : ℝ × ℝ)

def vector_BA (B A : ℝ × ℝ) : ℝ × ℝ := (A.1 - B.1, A.2 - B.2)
def vector_v : ℝ × ℝ := (1, -1)

theorem vector_satisfies_condition : 
  2 * vector_v.1 - vector_v.2 = 3 := by
  sorry

end vector_satisfies_condition_l462_462468


namespace max_sum_inverse_distinct_subsets_l462_462943

open Finset

theorem max_sum_inverse_distinct_subsets (n : ℕ) (a : Fin n → ℕ) 
  (h_n : 5 ≤ n)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_subsets : ∀ A B : Finset (Fin n), A ≠ B → A.nonempty → B.nonempty → (∑ x in A, a x) ≠ (∑ x in B, a x)) :
  (∑ x, 1 / (a x : ℝ)) ≤ 2 - 1 / 2 ^ (n - 1) :=
sorry

end max_sum_inverse_distinct_subsets_l462_462943


namespace plates_added_before_topple_l462_462031

theorem plates_added_before_topple (init_plates add_first add_total : ℕ) (h : init_plates = 27) (h1 : add_first = 37) (h2 : add_total = 83) : 
  add_total - (init_plates + add_first) = 19 :=
by
  -- proof goes here
  sorry

end plates_added_before_topple_l462_462031


namespace arrangement_of_mississippi_no_adjacent_s_l462_462473

-- Conditions: The word "MISSISSIPPI" has 11 letters with specific frequencies: 1 M, 4 I's, 4 S's, 2 P's.
-- No two S's can be adjacent.
def ways_to_arrange_mississippi_no_adjacent_s: Nat :=
  let total_non_s_arrangements := Nat.factorial 7 / (Nat.factorial 4 * Nat.factorial 2)
  let gaps_for_s := Nat.choose 8 4
  total_non_s_arrangements * gaps_for_s

theorem arrangement_of_mississippi_no_adjacent_s : ways_to_arrange_mississippi_no_adjacent_s = 7350 :=
by
  unfold ways_to_arrange_mississippi_no_adjacent_s
  sorry

end arrangement_of_mississippi_no_adjacent_s_l462_462473


namespace greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l462_462359

theorem greatest_divisor_of_sum_first_15_terms_arithmetic_sequence
  (x c : ℕ) -- where x and c are positive integers
  (h_pos_x : 0 < x) -- x is positive
  (h_pos_c : 0 < c) -- c is positive
  : ∃ (d : ℕ), d = 15 ∧ ∀ (S : ℕ), S = 15 * x + 105 * c → d ∣ S :=
by
  sorry

end greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l462_462359


namespace neg_fractions_comparison_l462_462458

theorem neg_fractions_comparison : (- (5 / 6: ℚ) > - (6 / 7: ℚ)) :=
by {
  have h1 : abs (- (5 / 6: ℚ)) = 5 / 6 := abs_neg (5 / 6),
  have h2 : abs (- (6 / 7: ℚ)) = 6 / 7 := abs_neg (6 / 7),
  have h3 : (5 / 6: ℚ) < (6 / 7: ℚ) := by norm_num,
  sorry
}

end neg_fractions_comparison_l462_462458


namespace smallest_possible_n_l462_462483

theorem smallest_possible_n (n : ℕ) (h : ∃ k : ℕ, 15 * n - 2 = 11 * k) : n % 11 = 6 :=
by
  sorry

end smallest_possible_n_l462_462483


namespace rounding_indeterminacy_l462_462562

-- Define necessary rounding functions
def round_to_thousandth (x : ℝ) : ℝ := (real.round (x * 1000)) / 1000
def round_to_hundredth (x : ℝ) : ℝ := (real.round (x * 100)) / 100
def round_to_tenth (x : ℝ) : ℝ := (real.round (x * 10)) / 10

-- Specify the conditions and goals
theorem rounding_indeterminacy (x : ℝ) :
  let a := round_to_thousandth x in
  let b := round_to_hundredth a in
  let c := round_to_tenth b in
  let d := round_to_tenth x in
  ¬ ((a ≥ b ∧ b ≥ c ∧ c ≥ d) ∨ (a ≤ b ∧ b ≤ c ∧ c ≤ d)) :=
by
  sorry -- Proof to be provided

end rounding_indeterminacy_l462_462562


namespace inequality_solution_cosine_range_l462_462399

-- Problem 1
theorem inequality_solution (x : ℝ) : 81 * 3^(2 * x) > (1 / 9) ^ (x + 2) ↔ x > -4 / 3 :=
sorry

-- Problem 2
theorem cosine_range :
  ∀ x ∈ Icc (0 : ℝ) (π / 2), 
    -3 * Real.sqrt 2 / 2 ≤ 3 * Real.cos (2 * x + π / 4) ∧
    3 * Real.cos (2 * x + π / 4) ≤ 3 * Real.sqrt 2 / 2 :=
sorry

end inequality_solution_cosine_range_l462_462399


namespace simplify_expr_l462_462261

theorem simplify_expr :
  (sqrt 450 / sqrt 200) + (sqrt 98 / sqrt 49) = (3 + 2 * sqrt 2) / 2 :=
by
  sorry

end simplify_expr_l462_462261


namespace function_increasing_l462_462564

variable {α : Type*} [LinearOrderedField α]

def is_increasing (f : α → α) : Prop :=
  ∀ x y : α, x < y → f x < f y

theorem function_increasing (f : α → α) (h : ∀ x1 x2 : α, x1 ≠ x2 → x1 * f x1 + x2 * f x2 > x1 * f x2 + x2 * f x1) :
  is_increasing f :=
by
  sorry

end function_increasing_l462_462564


namespace sum_of_first_15_terms_l462_462566

theorem sum_of_first_15_terms (a d : ℝ) (h : a + 7 * d = 4) : 
  let S₁₅ := 15 / 2 * (2 * a + 14 * d) in S₁₅ = 60 :=
by
  have h₁ : 2 * a + 14 * d = 8,
    from calc
      2 * a + 14 * d = 2 * (a + 7 * d) : by ring
      ... = 2 * 4 : by rw [h]
      ... = 8 : by norm_num,
  sorry

end sum_of_first_15_terms_l462_462566


namespace PQ_R_exist_l462_462789

theorem PQ_R_exist :
  ∃ P Q R : ℚ, 
    (P = -3/5) ∧ (Q = -1) ∧ (R = 13/5) ∧
    (∀ x : ℚ, x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 → 
    (x^2 - 10)/((x - 1)*(x - 4)*(x - 6)) = P/(x - 1) + Q/(x - 4) + R/(x - 6)) :=
by
  sorry

end PQ_R_exist_l462_462789


namespace coplanarity_of_intersection_lines_l462_462618

-- Definitions for points and circumsphere.
variable {α β γ δ : Plane}
variable {A B C D O : Point}
variable {R : ℝ}
variable (h0 : α.tangent_to_circumsphere_tetrahedron_at A)
variable (h1 : β.tangent_to_circumsphere_tetrahedron_at B)
variable (h2 : γ.tangent_to_circumsphere_tetrahedron_at C)
variable (h3 : δ.tangent_to_circumsphere_tetrahedron_at D)
variable (h4 : ∃ l1, is_intersection_line α β l1 ∧ is_coplanar l1 (Line_through C D))

-- Proof statement
theorem coplanarity_of_intersection_lines (h0 : α.tangent_to_circumsphere_tetrahedron_at A) 
                                         (h1 : β.tangent_to_circumsphere_tetrahedron_at B)
                                         (h2 : γ.tangent_to_circumsphere_tetrahedron_at C)
                                         (h3 : δ.tangent_to_circumsphere_tetrahedron_at D)
                                         (h4 : ∃ l1, is_intersection_line α β l1 ∧ is_coplanar l1 (Line_through C D)) :
  ∃ l2, is_intersection_line γ δ l2 ∧ is_coplanar l2 (Line_through A B) :=
by
  sorry

end coplanarity_of_intersection_lines_l462_462618


namespace find_value_of_P_l462_462526

def f (x : ℝ) : ℝ := (x^2 + x - 2)^2002 + 3

theorem find_value_of_P :
  f ( (Real.sqrt 5) / 2 - 1 / 2 ) = 4 := by
  sorry

end find_value_of_P_l462_462526


namespace proportional_function_property_l462_462255

theorem proportional_function_property :
  (∀ x, ∃ y, y = -3 * x ∧
  (x = 0 → y = 0) ∧
  (x > 0 → y < 0) ∧
  (x < 0 → y > 0) ∧
  (x = 1 → y = -3) ∧
  (∀ x, y = -3 * x → (x > 0 ∧ y < 0 ∨ x < 0 ∧ y > 0))) :=
by
  sorry

end proportional_function_property_l462_462255


namespace pages_left_to_read_l462_462953

def total_pages : ℕ := 17
def pages_read : ℕ := 11

theorem pages_left_to_read : total_pages - pages_read = 6 := by
  sorry

end pages_left_to_read_l462_462953


namespace number_of_scenarios_l462_462028

theorem number_of_scenarios (n k : ℕ) (h₁ : n = 6) (h₂ : k = 6) :
    ∃ m: ℕ, m = Nat.choose 6 2 * 5^4
:= by
  use Nat.choose 6 2 * 5^4
  sorry

end number_of_scenarios_l462_462028


namespace greatest_divisor_of_arithmetic_sequence_sum_l462_462335

theorem greatest_divisor_of_arithmetic_sequence_sum :
  ∀ (x c : ℕ), ∃ k : ℕ, k = 15 ∧ 15 ∣ (15 * x + 105 * c) :=
by
  intro x c
  exists 15
  split
  . rfl
  . sorry

end greatest_divisor_of_arithmetic_sequence_sum_l462_462335


namespace find_m_l462_462685

def circle1 (x y m : ℝ) : Prop := (x + 2)^2 + (y - m)^2 = 9
def circle2 (x y m : ℝ) : Prop := (x - m)^2 + (y + 1)^2 = 4

theorem find_m (m : ℝ) : 
  ∃ x1 y1 x2 y2 : ℝ, 
    circle1 x1 y1 m ∧ 
    circle2 x2 y2 m ∧ 
    (m + 2)^2 + (-1 - m)^2 = 25 → 
    m = 2 :=
by
  sorry

end find_m_l462_462685


namespace min_abs_sum_sequence_l462_462731

def sequence (x : ℕ → ℤ) := (x 0 = 0) ∧ (∀ n : ℕ, |x (n + 1)| = |x n + 1|)

theorem min_abs_sum_sequence : ∀ (x : ℕ → ℤ), sequence x → ∃ S : ℤ, S = | (Finset.range 1975).sum (λ n, x (n + 1)) | ∧ S = 20 :=
by
  intro x seq_x
  sorry

end min_abs_sum_sequence_l462_462731


namespace sum_abc_l462_462075

noncomputable def a : ℝ := 5^64
noncomputable def b : ℝ := 3^(5^16)
noncomputable def c : ℝ := 4^(3^125)

theorem sum_abc :
  (log 3) (log 4) (log 5 a) = 1 ∧
  (log 4) (log 5) (log 3 b) = 2 ∧
  (log 5) (log 3) (log 4 c) = 3 →
  a + b + c = 5^64 + 3^(5^16) + 4^(3^125) :=
by
  sorry

end sum_abc_l462_462075


namespace calculate_weight_of_6_moles_HClO2_l462_462143

noncomputable def weight_of_6_moles_HClO2 := 
  let molar_mass_H := 1.01
  let molar_mass_Cl := 35.45
  let molar_mass_O := 16.00
  let molar_mass_HClO2 := molar_mass_H + molar_mass_Cl + 2 * molar_mass_O
  let moles_HClO2 := 6
  moles_HClO2 * molar_mass_HClO2

theorem calculate_weight_of_6_moles_HClO2 : weight_of_6_moles_HClO2 = 410.76 :=
by
  sorry

end calculate_weight_of_6_moles_HClO2_l462_462143


namespace cos_arcsin_five_thirteen_l462_462459

theorem cos_arcsin_five_thirteen : cos (arcsin (5 / 13)) = 12 / 13 := 
by
  sorry

end cos_arcsin_five_thirteen_l462_462459


namespace grid_second_row_531_l462_462083

def valid_grid (grid : Matrix (Fin 5) (Fin 5) ℕ) : Prop :=
  (∀ i, Set.toFinset (Set.range (fun j => grid i j)).card = 5) ∧  -- Each row contains unique numbers
  (∀ j, Set.toFinset (Set.range (fun i => grid i j)).card = 5) ∧  -- Each column contains unique numbers
  (∀ i j, ∀ i' < 5, ∀ j' < 5, grid i j ≠ grid i' j' → (grid i j, grid i' j').snd ∣ (grid i j, grid i' j').fst)  -- Division operation results in integer

theorem grid_second_row_531 (grid : Matrix (Fin 5) (Fin 5) ℕ) (h_valid : valid_grid grid) :
  (100 * grid 1 0 + 10 * grid 1 1 + grid 1 2) = 531 :=
sorry

end grid_second_row_531_l462_462083


namespace find_extrema_l462_462795

def f (x : ℝ) : ℝ := (2 * x + 1) / (x + 1)

theorem find_extrema :
  let a := 1
  let b := 4
  (∀ x, x ∈ set.Icc a b → f x ≤ f b) ∧
  (∀ x, x ∈ set.Icc a b → f x ≥ f a) ∧
  f a = 3 / 2 ∧ f b = 9 / 5 :=
by
  sorry

end find_extrema_l462_462795


namespace range_of_m_l462_462510

theorem range_of_m (m : ℝ) (x : ℝ) (hp : (x + 2) * (x - 10) ≤ 0)
  (hq : x^2 - 2 * x + 1 - m^2 ≤ 0) (hm : m > 0) : 0 < m ∧ m ≤ 3 :=
sorry

end range_of_m_l462_462510


namespace number_of_elements_in_A_inter_B_is_zero_l462_462619

def A : set ℕ := {0, 1, 2}
def B : set ℝ := {x | (x + 1) * (x + 2) < 0}

theorem number_of_elements_in_A_inter_B_is_zero : (A ∩ B).finite ∧ (A ∩ B).to_finset.card = 0 := by
  sorry

end number_of_elements_in_A_inter_B_is_zero_l462_462619


namespace disjoint_pairs_div_mod_l462_462222

def set_T : Set ℕ := { x | x ∈ Finset.range 16 ∧ x ≠ 0 }

def num_disjoint_pairs (T : Set ℕ) : ℕ := 
  (3^15 - 2 * 2^15 + 1) / 2

theorem disjoint_pairs_div_mod :
  let T := set_T in
  m = num_disjoint_pairs T →
  m % 500 = 186 := by
    intros T m h1
    sorry

end disjoint_pairs_div_mod_l462_462222


namespace inequality_solution_l462_462084

theorem inequality_solution (x : ℝ) :
  (x ∈ Ioo (-2:ℝ) (2:ℝ) ∨ x ∈ Ioo (2:ℝ) (10:ℝ)) → 
  ((x^2 + 1) / (x - 2) ≥ (2 * x + 13) / (3 * (x + 2))) := 
begin
  sorry
end

end inequality_solution_l462_462084


namespace number_of_incorrect_inferences_l462_462472

theorem number_of_incorrect_inferences :
  let stmt1 := (∀ x : ℝ, (x ≠ 1 → x^2 - 3 * x + 2 ≠ 0) ↔ ∀ x : ℝ, (x^2 - 3 * x + 2 = 0 → x = 1))
  let stmt2 := (¬ (∀ x : ℝ, (x^2 = 1 → x = 1)) ↔ ∀ x : ℝ, (x^2 = 1 → x ≠ 1))
  let stmt3 := (∀ x : ℝ, (x < 1 → x^2 - 3 * x + 2 > 0) ∧ (¬ (x^2 - 3 * x + 2 > 0 → x < 1)))
  let stmt4 := (∀ p q : Prop, ¬(p ∧ q) ↔ (¬p ∧ ¬q))
  in 2 = (cond (stmt1 ∧ ¬stmt2 ∧ stmt3 ∧ ¬stmt4) (1) (0)) + 
          (cond (¬stmt1) (1) (0)) +
          (cond (stmt2) (1) (0)) +
          (cond (¬stmt3) (1) (0)) +
          (cond (stmt4) (1) (0)) 
:= sorry

end number_of_incorrect_inferences_l462_462472


namespace tax_percentage_other_items_l462_462636

theorem tax_percentage_other_items (total_amt before_tax : ℝ) (clothing_pct food_pct other_items_pct : ℝ) (tax_clothing_pct tax_total_pct : ℝ) (tax_other_items_pct : ℝ) :
  clothing_pct = 0.50 ∧
  food_pct = 0.20 ∧
  other_items_pct = 0.30 ∧
  tax_clothing_pct = 0.04 ∧
  tax_total_pct = 0.044 ∧
  total_amt > 0 → 
  let clothing_amt := clothing_pct * total_amt in
  let food_amt := food_pct * total_amt in
  let other_items_amt := other_items_pct * total_amt in
  let tax_clothing := tax_clothing_pct * clothing_amt in
  let total_tax := tax_total_pct * total_amt in
  let tax_other_items := total_tax - tax_clothing in
  tax_other_items_pct = (tax_other_items / other_items_amt) * 100 :=
begin
  intros h,
  sorry
end

end tax_percentage_other_items_l462_462636


namespace total_machines_sold_l462_462385

noncomputable def commission_rate_first_hundred : ℝ := 0.03
noncomputable def commission_rate_after_hundred : ℝ := 0.04
noncomputable def sale_price_per_machine : ℝ := 10000
noncomputable def total_commission_received : ℝ := 42000

theorem total_machines_sold :
  let commission_first_hundred := commission_rate_first_hundred * sale_price_per_machine * 100 in
  let remaining_commission := total_commission_received - commission_first_hundred in
  let commission_per_machine_after_hundred := commission_rate_after_hundred * sale_price_per_machine in
  let machines_sold_after_hundred := remaining_commission / commission_per_machine_after_hundred in
  let total_machines := 100 + machines_sold_after_hundred in
  total_machines = 130 :=
by
  sorry

end total_machines_sold_l462_462385


namespace greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l462_462364

theorem greatest_divisor_of_sum_first_15_terms_arithmetic_sequence
  (x c : ℕ) -- where x and c are positive integers
  (h_pos_x : 0 < x) -- x is positive
  (h_pos_c : 0 < c) -- c is positive
  : ∃ (d : ℕ), d = 15 ∧ ∀ (S : ℕ), S = 15 * x + 105 * c → d ∣ S :=
by
  sorry

end greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l462_462364


namespace infinite_g_eq_1_l462_462833

noncomputable def g : ℝ -> ℝ := sorry

def conditions (g : ℝ → ℝ) : Prop :=
  ∀ A B : set ℝ, 
    (∀ x : ℝ, (x ∈ A ∨ x ∈ B) ∧ (x ∈ [0, 1])) →
    (A ≠ ∅ ∧ B ≠ ∅) →
    (∃ x ∈ A, g x ∈ B ∨ ∃ x ∈ B, g x ∈ A) ∧ 
    ∀ x : [0, 1], g x > x

theorem infinite_g_eq_1 (g : ℝ → ℝ) (h : conditions g) : 
  ∃^∞ x ∈ [0, 1], g x = 1 := 
BY
  sorry

end infinite_g_eq_1_l462_462833


namespace min_f_max_f_l462_462101

-- Defining the quadratic equation and the function f(k) based on its larger root.
noncomputable def f (k : ℝ) : ℝ :=
  let a := k^2 + 1
  let b := 10 * k
  let c := -6 * (9 * k^2 + 1)
  let discriminant := b^2 - 4 * a * c
  (-b + Real.sqrt discriminant) / (2 * a)

-- Stating the proofs for the minimum and maximum values of f(k).
theorem min_f (k : ℝ) : f(k) ≥ 2 := sorry
theorem max_f (k : ℝ) : f(k) ≤ 9 := sorry

end min_f_max_f_l462_462101


namespace solve_inequality_l462_462651

theorem solve_inequality (x : ℝ) :
  (0 ≤ x^2 - x - 2 ∧ x^2 - x - 2 ≤ 4) ↔
  (-2 ≤ x ∧ x ≤ -1) ∨ (2 ≤ x ∧ x ≤ 3) :=
by sorry

end solve_inequality_l462_462651


namespace mn_parallel_bc_l462_462835

open EuclideanGeometry

variables {ω : Circle} {A B C M N : Point} 

/-- Given a triangle ABC inscribed in a circle ω,
    M is the foot of the perpendicular from B to AC,
    and N is the foot of the perpendicular from A to the tangent to ω at B,
    prove that MN is parallel to BC. -/
theorem mn_parallel_bc (h_triangle_ABC : InscribedTriangle ω A B C)
  (h_M_perp : FootOfPerpendicular B AC M)
  (h_N_perp : FootOfPerpendicular A (TangentAtB ω B) N) :
  Parallel MN BC := sorry

end mn_parallel_bc_l462_462835


namespace trig_identity_example_l462_462486

theorem trig_identity_example :
  cos (40 : ℝ) * cos (160 : ℝ) + sin (40 : ℝ) * sin (20 : ℝ) = - (1 / 2) :=
by
  sorry  -- This denotes that the proof is omitted

end trig_identity_example_l462_462486


namespace xyz_value_l462_462560

theorem xyz_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x + 1/y = 5) (h2 : y + 1/z = 3) (h3 : z + 1/x = 2) :
  x * y * z = 10 + 3 * Real.sqrt 11 :=
by
  sorry

end xyz_value_l462_462560


namespace part1_part2_l462_462824

theorem part1 (m n : ℤ) (h : m + 4 * n - 3 = 0) : 2^m * 16^n = 8 := 
by
  sorry

theorem part2 (x : ℝ) (n : ℤ) (hn : 0 < n) (hx : x^(2 * n) = 4) : 
  (x^(3 * n))^2 - 2 * (x^2)^(2 * n) = 32 := 
by
  sorry

end part1_part2_l462_462824


namespace scientific_notation_of_number_l462_462628

theorem scientific_notation_of_number :
  1214000 = 1.214 * 10^6 :=
by
  sorry

end scientific_notation_of_number_l462_462628


namespace erik_orange_juice_count_l462_462485

theorem erik_orange_juice_count (initial_money bread_loaves bread_cost orange_juice_cost remaining_money : ℤ)
  (h₁ : initial_money = 86)
  (h₂ : bread_loaves = 3)
  (h₃ : bread_cost = 3)
  (h₄ : orange_juice_cost = 6)
  (h₅ : remaining_money = 59) :
  (initial_money - remaining_money - (bread_loaves * bread_cost)) / orange_juice_cost = 3 :=
by
  sorry

end erik_orange_juice_count_l462_462485


namespace measure_angle_XZY_l462_462197

theorem measure_angle_XZY {X Y Z M : Type} [affine_space ℝ (triangle X Y Z)] 
(h_mid : midpoint M Y Z)
(h_angle1 : angle X M Z = 30)
(h_angle2 : angle X Y Z = 15) : 
angle X Z Y = 75 := 
sorry

end measure_angle_XZY_l462_462197


namespace polygon_interior_exterior_relation_l462_462290

theorem polygon_interior_exterior_relation :
  ∃ n : ℕ, (n ≥ 3) ∧ ((n - 2) * 180 = 2 * 360) → n = 6 :=
begin
  sorry
end

end polygon_interior_exterior_relation_l462_462290


namespace find_circle_equation_find_line_equation_l462_462828

-- Given definitions
variables {x y a b r : ℝ}

-- Conditions
def condition1 : Prop := (1 - a)^2 + (sqrt 3 - b)^2 = r^2
def condition2 : Prop := b = a
def condition3 : Prop := r^2 = (a + a - 2)^2 / 2 + (sqrt 2)^2

-- Equation of the circle
def circle_eq : Prop := x^2 + y^2 = 4

-- Proving the circle equation
theorem find_circle_equation : condition1 ∧ condition2 ∧ condition3 → circle_eq :=
begin
  sorry
end

-- Line conditions
variables {m x1 y1 x2 y2 : ℝ}

def line_eq (m : ℝ) : Prop := 2 * x + m * y - 3 = 0

def point_condition : Prop := ∃ (m : ℝ), line_eq m

def dot_product_condition : Prop := (m^2 + 1) * y1 * y2 + (3 / 2) * m * (y1 + y2) + 9 / 4 = -2

-- Proving the line equation
theorem find_line_equation : point_condition ∧ dot_product_condition → (line_eq (sqrt 5 / 2) ∨ line_eq (-sqrt 5 / 2)) :=
begin
  sorry
end

end find_circle_equation_find_line_equation_l462_462828


namespace exists_nonagon_distinct_diagonals_l462_462202

-- Define what it means to place numbers on vertices of a nonagon such that all diagonal products are distinct
structure NonagonPlacement (vertices : Fin 9 → ℕ) :=
(is_distinct_diagonal_products : (∀ i j : Fin 9, i ≠ j → vertices i ≠ vertices j) → 
  ∀ (i j k l m n : Fin 9), 
  (i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ j ≠ l ∧ i ≠ l ∧ k ≠ l ∧ m ≠ n ∧ m ≠ k ∧ n ≠ k) → 
  (vertices i * vertices j ≠ vertices m * vertices n))

-- State the main theorem
theorem exists_nonagon_distinct_diagonals : 
  ∃ placement : Fin 9 → ℕ, NonagonPlacement placement :=
sorry

end exists_nonagon_distinct_diagonals_l462_462202


namespace john_paid_8000_l462_462599

-- Define the variables according to the conditions
def upfront_fee : ℕ := 1000
def hourly_rate : ℕ := 100
def court_hours : ℕ := 50
def prep_hours : ℕ := 2 * court_hours
def total_hours : ℕ := court_hours + prep_hours
def total_fee : ℕ := upfront_fee + total_hours * hourly_rate
def john_share : ℕ := total_fee / 2

-- Prove that John's share is $8,000
theorem john_paid_8000 : john_share = 8000 :=
by sorry

end john_paid_8000_l462_462599


namespace total_chairs_calc_l462_462813

-- Defining the condition of having 27 rows
def rows : ℕ := 27

-- Defining the condition of having 16 chairs per row
def chairs_per_row : ℕ := 16

-- Stating the theorem that the total number of chairs is 432
theorem total_chairs_calc : rows * chairs_per_row = 432 :=
by
  sorry

end total_chairs_calc_l462_462813


namespace lcm_of_1_to_10_is_2520_l462_462698

noncomputable def lcm_of_1_to_10 : ℕ := 
  Nat.lcm (List.range 10).succ

theorem lcm_of_1_to_10_is_2520 : lcm_of_1_to_10 = 2520 :=
by
  sorry

end lcm_of_1_to_10_is_2520_l462_462698


namespace standard_equation_of_ellipse_distance_to_directrix_l462_462195

noncomputable def ellipse (a b : ℝ) : Prop := 
  a > b ∧ b > 0 ∧ ∀ (x y : ℝ), x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1

theorem standard_equation_of_ellipse 
  (a b : ℝ) 
  (h : ellipse a b)
  (F : ℝ × ℝ := (-1, 0)) 
  (c : ℝ := 1)
  (A : ℝ × ℝ := (-a, 0)) 
  (B : ℝ × ℝ := (0, b)) 
  (C : ℝ × ℝ := (0, -b)) 
  (M : ℝ × ℝ := (-a / 2, 0))
  (h1 : ∀ (x y : ℝ), y = x + b → (x, y) = M)
  (h2 : ∀ (x y : ℝ), x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1) :
  a = 3 ∧ b ^ 2 = 8 := 
sorry

theorem distance_to_directrix
  (a b : ℝ)
  (h : ellipse a b)
  (h1 : a ^ 2 = b ^ 2 + 1)
  (slope_eq : 1)
  (D : ℝ × ℝ := (-4 / 3, -1 / 3))
  (directrix_x : ℝ := 2) 
  (dist_D_to_directrix : ℝ := abs (D.1 - directrix_x)) :
  dist_D_to_directrix = 10 / 3 :=
sorry

end standard_equation_of_ellipse_distance_to_directrix_l462_462195


namespace max_area_of_2023_sided_polygon_l462_462227

noncomputable def maximum_area_2023_sided_polygon : ℝ :=
  1011 / 2 * Real.cot (Real.pi / 4044)

theorem max_area_of_2023_sided_polygon (P : Polygon) (h1 : P.sides = 2023)
  (h2 : ∀ (i : Fin (2023 - 1)), P.side_length i = 1) :
  P.area ≤ maximum_area_2023_sided_polygon :=
sorry

end max_area_of_2023_sided_polygon_l462_462227


namespace total_oranges_l462_462681

theorem total_oranges (capacity1 capacity2 capacity3 : ℕ) (fill1 fill2 fill3 : ℚ)
  (h1 : capacity1 = 80) (h2 : capacity2 = 50) (h3 : capacity3 = 60)
  (hf1 : fill1 = 3/4) (hf2 : fill2 = 3/5) (hf3 : fill3 = 2/3) :
  let oranges1 := fill1 * capacity1,
      oranges2 := fill2 * capacity2,
      oranges3 := fill3 * capacity3,
      total_oranges := oranges1 + oranges2 + oranges3
  in total_oranges = 130 := 
by {
  have oranges1_eq : oranges1 = 60 := by sorry,
  have oranges2_eq : oranges2 = 30 := by sorry,
  have oranges3_eq : oranges3 = 40 := by sorry,
  have total_eq : total_oranges = 130 := by sorry,
  exact total_eq
}

end total_oranges_l462_462681


namespace lottery_probability_correct_l462_462994

def number_of_winnerballs_ways : ℕ := Nat.choose 50 6

def probability_megaBall : ℚ := 1 / 30

def probability_winnerBalls : ℚ := 1 / number_of_winnerballs_ways

def combined_probability : ℚ := probability_megaBall * probability_winnerBalls

theorem lottery_probability_correct : combined_probability = 1 / 476721000 := by
  sorry

end lottery_probability_correct_l462_462994


namespace range_of_a_l462_462166

theorem range_of_a (a : ℝ) :
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, | (3^n / (3^(n+1) + (a+1)^n)) - (1/3) | < ε ) →
  a ∈ set.Ioo (-4 : ℝ) (2 : ℝ) :=
by 
  intros h
  -- Formal proof goes here
  sorry

end range_of_a_l462_462166


namespace greatest_divisor_arithmetic_sum_l462_462370

theorem greatest_divisor_arithmetic_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ d, d = 15 ∧ ∀ S : ℕ, S = 15 * x + 105 * c → d ∣ S :=
by 
  sorry

end greatest_divisor_arithmetic_sum_l462_462370


namespace solve_for_y_l462_462976

theorem solve_for_y (y : ℝ) (h : log 3 y + 2 * (log 3 y / log 3 9) + (log 3 y / log 3 27) = 5) : 
  y = 3 ^ (15 / 7) :=
sorry

end solve_for_y_l462_462976


namespace probability_of_fourth_three_is_correct_l462_462257

noncomputable def p_plus_q : ℚ := 41 + 84

theorem probability_of_fourth_three_is_correct :
  let fair_die_prob := (1 / 6 : ℚ)
  let biased_die_prob := (1 / 2 : ℚ)
  -- Probability of rolling three threes with the fair die:
  let fair_die_three_three_prob := fair_die_prob ^ 3
  -- Probability of rolling three threes with the biased die:
  let biased_die_three_three_prob := biased_die_prob ^ 3
  -- Probability of rolling three threes in total:
  let total_three_three_prob := fair_die_three_three_prob + biased_die_three_three_prob
  -- Probability of using the fair die given three threes
  let fair_die_given_three := fair_die_three_three_prob / total_three_three_prob
  -- Probability of using the biased die given three threes
  let biased_die_given_three := biased_die_three_three_prob / total_three_three_prob
  -- Probability of rolling another three:
  let fourth_three_prob := fair_die_given_three * fair_die_prob + biased_die_given_three * biased_die_prob
  -- Simplifying fraction
  let result_fraction := (41 / 84 : ℚ)
  -- Final answer p + q is 125
  p_plus_q = 125 ∧ fourth_three_prob = result_fraction
:= by
  sorry

end probability_of_fourth_three_is_correct_l462_462257


namespace water_tank_capacity_l462_462705

theorem water_tank_capacity (C : ℝ) :
  0.4 * C - 0.1 * C = 36 → C = 120 :=
by sorry

end water_tank_capacity_l462_462705


namespace find_m_values_l462_462122

theorem find_m_values (m : ℝ) :
  let l1 := (m + 2) * x + (m + 3) * y - 5,
      l2 := 6 * x + (2 * m - 1) * y = 5 in
  (l1 // l2 -> m = -5 / 2) ∧ 
  (l1 ⊥ l2 -> m = -1 ∨ m = -9 / 2) :=
sorry

end find_m_values_l462_462122


namespace females_in_soccer_not_in_basketball_l462_462980

/-- The Pythagoras High School soccer team and basketball team members statistics -/
variables (soccer_male soccer_female basketball_male basketball_female both_male total_students : ℕ)
variables (both_female soccer_only_female : ℕ)

/-- Given conditions -/
def conditions : Prop :=
  soccer_male = 120 ∧
  soccer_female = 60 ∧
  basketball_male = 100 ∧
  basketball_female = 80 ∧
  both_male = 70 ∧
  total_students = 260 ∧
  both_female = 60 + 80 - (260 - (120 + 100 - 70)) ∧
  soccer_only_female = 60 - both_female

/-- Prove the number of females in the soccer team who are NOT in the basketball team -/
theorem females_in_soccer_not_in_basketball : soccer_only_female = 30 :=
by
  intro h:
    sorry
    sorry  



end females_in_soccer_not_in_basketball_l462_462980


namespace probability_a_sub_b_gt_0_is_zero_l462_462590

open set

noncomputable def triangle : set (ℝ × ℝ) :=
  {p | ∃ (a b : ℝ), (a, b) = (0, 0) ∨ (a, b) = (4, 0) ∨ (a, b) = (4, 10)}

theorem probability_a_sub_b_gt_0_is_zero :
  ∀ (a b : ℝ), (a, b) ∈ triangle → ¬(a - b > 0) := 
sorry

end probability_a_sub_b_gt_0_is_zero_l462_462590


namespace problem1_problem2_l462_462224

-- Define the required conditions
variables {a b : ℤ}
-- Conditions
axiom h1 : a ≥ 1
axiom h2 : b ≥ 1

-- Proof statement for question 1
theorem problem1 : ¬ (a ∣ b^2 ↔ a ∣ b) := by
  sorry

-- Proof statement for question 2
theorem problem2 : (a^2 ∣ b^2 ↔ a ∣ b) := by
  sorry

end problem1_problem2_l462_462224


namespace sin_angle_BAC_l462_462587

open Real -- Open the real number space for real number operations

-- Define the coordinates of points A, B, and C
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (1, 0, 0)
def C : ℝ × ℝ × ℝ := (1, 1, 0)

-- Define the vectors AB and AC
def vector_AB : ℝ × ℝ × ℝ := (B.1 - A.1, B.2 - A.2, B.3 - A.3) -- (1, 0, 0)
def vector_AC : ℝ × ℝ × ℝ := (C.1 - A.1, C.2 - A.2, C.3 - A.3) -- (1, 1, 0)

-- Define the dot product of two vectors
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Define the magnitude of a vector
def magnitude (u : ℝ × ℝ × ℝ) : ℝ := sqrt (u.1 ^ 2 + u.2 ^ 2 + u.3 ^ 2)

-- Define the vector component of AC perpendicular to AB
def vector_AC_prime : ℝ × ℝ × ℝ :=
  let proj_len := dot_product vector_AC vector_AB / (dot_product vector_AB vector_AB)
  in (vector_AC.1 - proj_len * vector_AB.1, vector_AC.2 - proj_len * vector_AB.2, vector_AC.3 - proj_len * vector_AB.3)

-- Define the Lean theorem statement
theorem sin_angle_BAC : sin (angle ⟨vector_AB⟩ ⟨vector_AC⟩) = (√2) / 2 :=
by 
  sorry

end sin_angle_BAC_l462_462587


namespace greatest_divisor_of_arithmetic_sequence_sum_l462_462336

theorem greatest_divisor_of_arithmetic_sequence_sum :
  ∀ (x c : ℕ), ∃ k : ℕ, k = 15 ∧ 15 ∣ (15 * x + 105 * c) :=
by
  intro x c
  exists 15
  split
  . rfl
  . sorry

end greatest_divisor_of_arithmetic_sequence_sum_l462_462336


namespace question1_question2_l462_462878

-- Define the sets A and B as given in the problem
def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}
def B : Set ℝ := {x | x < -2 ∨ x > 5}

-- Lean statement for (1)
theorem question1 (m : ℝ) : 
  (A m ⊆ B) ↔ (m < 2 ∨ m > 4) :=
by
  sorry

-- Lean statement for (2)
theorem question2 (m : ℝ) : 
  (A m ∩ B = ∅) ↔ (m ≤ 3) :=
by
  sorry

end question1_question2_l462_462878


namespace obtuse_angle_A_DB_of_bisectors_l462_462909

theorem obtuse_angle_A_DB_of_bisectors
  (ABC: Triangle)
  (A B C D : Point)
  (h_right_triangle : is_right_triangle ABC)
  (h_angle_A_45 : angle A = 45)
  (h_angle_B_45 : angle B = 45)
  (h_bisectors_intersect_D : intersects_angle_bisectors A B D) :
  angle ADB = 135 :=
by sorry

end obtuse_angle_A_DB_of_bisectors_l462_462909


namespace angle_is_2pi_over_3_l462_462142

variables (a b : ℝ) 

-- Provided definitions and conditions
def vec_a_dot_vec_a_plus_2_vec_b : Prop := (a • a + 2 • b) = (0 : vector ℝ 3)
def norm_a : Prop := ∥a∥ = 2
def norm_b : Prop := ∥b∥ = 2

-- Angle to prove
def angle_between_vectors (a b : vector ℝ 3) : ℝ := real.arccos ((a • b) / (∥a∥ * ∥b∥))

theorem angle_is_2pi_over_3 (a b : vector ℝ 3)
  (h1 : vec_a_dot_vec_a_plus_2_vec_b a b)
  (h2 : norm_a a)
  (h3 : norm_b b) :
  angle_between_vectors a b = (2 * real.pi / 3) :=
sorry

end angle_is_2pi_over_3_l462_462142


namespace tangent_line_to_parabola_l462_462475

theorem tangent_line_to_parabola (r : ℝ) :
  (∃ x : ℝ, 2 * x^2 - x - r = 0) ∧
  (∀ x1 x2 : ℝ, (2 * x1^2 - x1 - r = 0) ∧ (2 * x2^2 - x2 - r = 0) → x1 = x2) →
  r = -1 / 8 :=
sorry

end tangent_line_to_parabola_l462_462475


namespace sandwiches_difference_l462_462633

-- Define the number of sandwiches Samson ate at lunch on Monday
def sandwichesLunchMonday : ℕ := 3

-- Define the number of sandwiches Samson ate at dinner on Monday (twice as many as lunch)
def sandwichesDinnerMonday : ℕ := 2 * sandwichesLunchMonday

-- Define the total number of sandwiches Samson ate on Monday
def totalSandwichesMonday : ℕ := sandwichesLunchMonday + sandwichesDinnerMonday

-- Define the number of sandwiches Samson ate for breakfast on Tuesday
def sandwichesBreakfastTuesday : ℕ := 1

-- Define the total number of sandwiches Samson ate on Tuesday
def totalSandwichesTuesday : ℕ := sandwichesBreakfastTuesday

-- Define the number of more sandwiches Samson ate on Monday than on Tuesday
theorem sandwiches_difference : totalSandwichesMonday - totalSandwichesTuesday = 8 :=
by
  sorry

end sandwiches_difference_l462_462633


namespace fermat_point_distances_sum_l462_462237

/-- Fermat point P of triangle ABC calculation -/
theorem fermat_point_distances_sum :
  let A := (0 : ℝ, 0 : ℝ)
  let B := (12 : ℝ, 0 : ℝ)
  let C := (4 : ℝ, 6 : ℝ)
  let P := (5 : ℝ, 3 : ℝ)
  let dist (p1 p2 : ℝ × ℝ) := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let AP := dist A P
  let BP := dist B P
  let CP := dist C P
  let total_distance := AP + BP + CP
  x + y = 3 :=
by
  have h1 : AP = real.sqrt 34 := sorry
  have h2 : BP = real.sqrt 58 := sorry
  have h3 : CP = real.sqrt 10 := sorry
  have h_total : total_distance = real.sqrt 34 + real.sqrt 58 + real.sqrt 10 := sorry
  let x := 1
  let y := 2
  have h_xy : x + y = 3 := by linarith
  exact h_xy

end fermat_point_distances_sum_l462_462237


namespace local_min_c_value_l462_462539

-- Definition of the function f(x) with its local minimum condition
def f (x c : ℝ) := x * (x - c)^2

-- Theorem stating that for the given function f(x) to have a local minimum at x = 1, the value of c must be 1
theorem local_min_c_value (c : ℝ) (h : ∀ ε > 0, f 1 ε < f c ε) : c = 1 := sorry

end local_min_c_value_l462_462539


namespace arithmetic_operation_equals_l462_462696

theorem arithmetic_operation_equals :
  12.1212 + 17.0005 - 9.1103 = 20.0114 := 
by 
  sorry

end arithmetic_operation_equals_l462_462696


namespace quarter_sphere_surface_area_l462_462981

-- Given condition
def base_area_quarter_sphere (r : ℝ) : Prop :=
  (1 / 4) * Real.pi * r ^ 2 = 144 * Real.pi

-- Question to answer (problem to prove)
def total_surface_area_quarter_sphere (r : ℝ) : ℝ :=
  (1 / 4) * (4 * Real.pi * r ^ 2) + (1 / 2) * Real.pi * r ^ 2 + (1 / 4) * Real.pi * r ^ 2

theorem quarter_sphere_surface_area (r : ℝ) (h : base_area_quarter_sphere r) :
  total_surface_area_quarter_sphere r = 1008 * Real.pi := by
  sorry

end quarter_sphere_surface_area_l462_462981


namespace greatest_divisor_of_arithmetic_sequence_sum_l462_462333

theorem greatest_divisor_of_arithmetic_sequence_sum :
  ∀ (x c : ℕ), ∃ k : ℕ, k = 15 ∧ 15 ∣ (15 * x + 105 * c) :=
by
  intro x c
  exists 15
  split
  . rfl
  . sorry

end greatest_divisor_of_arithmetic_sequence_sum_l462_462333


namespace none_of_these_l462_462203

-- Definitions of basic conditions from step a)
variables (O A B C D E P : Point)
variables (r : ℝ)
variables (circle : Circle O r)

-- Additional conditions
axiom AB_perp_BC : perp (line A B) (line B C)
axiom ADOE_line : collinear [A, D, O, E]
axiom AP_eq_2AD : distance A P = 2 * distance A D
axiom AB_length : distance A B = 3 * r

-- Theorem statement that none of the above equations hold true
theorem none_of_these :
  ¬ (distance A P ^ 2 = distance P B * distance A B ∨
  distance A P * distance P D = distance P B * distance A D ∨
  distance A B ^ 2 = distance A D * distance D E ∨
  distance A B * distance A P = distance O B * distance A O) :=
sorry

end none_of_these_l462_462203


namespace greatest_divisor_sum_of_first_fifteen_terms_l462_462322

theorem greatest_divisor_sum_of_first_fifteen_terms 
  (x c : ℕ) (hx : x > 0) (hc : c > 0):
  ∃ d, d = 15 ∧ d ∣ (15*x + 105*c) :=
by
  existsi 15
  split
  . refl
  . apply Nat.dvd.intro
    existsi (x + 7*c)
    refl
  sorry

end greatest_divisor_sum_of_first_fifteen_terms_l462_462322


namespace max_points_on_line_four_circles_l462_462107

theorem max_points_on_line_four_circles (C1 C2 C3 C4 : Set Point) (hC : ∀ i ∈ {C1, C2, C3, C4}, circle i) : 
  ∃ L : Set Point, line L ∧ ∀ p ∈ L, p ∈ ⋃ i ∈ {C1, C2, C3, C4}, i → 
  (L ∩ ⋃ i ∈ {C1, C2, C3, C4}, i).card ≤ 8 :=
sorry

end max_points_on_line_four_circles_l462_462107


namespace sum_consecutive_odds_l462_462630

theorem sum_consecutive_odds (n : ℕ) : 
  ∑ k in finset.range n, (n^2 - n + 2 * k + 1) = n^3 :=
sorry

end sum_consecutive_odds_l462_462630


namespace f_eq_g_symmetry_point_l462_462645

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + π / 3)

noncomputable def g (x : ℝ) : ℝ := 4 * Real.cos (2 * x - π / 6)

theorem f_eq_g : ∀ x, f x = g x := by
  sorry

theorem symmetry_point : f (-π / 6) = 0 := by
  sorry

end f_eq_g_symmetry_point_l462_462645


namespace value_always_positive_l462_462850

variable {α : Type*}
variable [LinearOrder α]
variable (f : α → ℝ)
variable (a_n : ℕ → α)

-- Definitions for conditions
def monotonically_increasing (f : α → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

def odd_function (f : α → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_arithmetic_sequence (a_n : ℕ → α) : Prop :=
  ∃ d, ∀ n, a_n (n + 1) = a_n n + d

-- Given conditions
axiom f_monotonic_increasing : monotonically_increasing f
axiom f_odd : odd_function f
axiom a_n_arithmetic : is_arithmetic_sequence a_n
axiom a_3_pos : a_n 3 > 0

-- Proof problem statement
theorem value_always_positive : f (a_n 1) + f (a_n 3) + f (a_n 5) > 0 :=
sorry

end value_always_positive_l462_462850


namespace sum_of_roots_l462_462931

theorem sum_of_roots (a b c d : ℝ) (h : ∀ x : ℝ, 
  a * (x ^ 3 - x) ^ 3 + b * (x ^ 3 - x) ^ 2 + c * (x ^ 3 - x) + d 
  ≥ a * (x ^ 2 + x + 1) ^ 3 + b * (x ^ 2 + x + 1) ^ 2 + c * (x ^ 2 + x + 1) + d) :
  b / a = -6 :=
sorry

end sum_of_roots_l462_462931


namespace t_shirt_cost_l462_462761

theorem t_shirt_cost (total_amount_spent : ℝ) (number_of_t_shirts : ℕ) (cost_per_t_shirt : ℝ)
  (h0 : total_amount_spent = 201) 
  (h1 : number_of_t_shirts = 22)
  (h2 : cost_per_t_shirt = total_amount_spent / number_of_t_shirts) :
  cost_per_t_shirt = 9.14 := 
sorry

end t_shirt_cost_l462_462761


namespace probability_of_closer_to_6_l462_462020

noncomputable def probability_closer_to_6 (x : ℝ) : Prop :=
  x ∈ set.Icc (0 : ℝ) 8 ∧ abs (x - 6) < abs (x - 0)

theorem probability_of_closer_to_6 : 
  ∃ p : ℝ, p = 0.6 ∧ measure_theory.measure.probability_closer_to_6 p :=
sorry

end probability_of_closer_to_6_l462_462020


namespace triangle_properties_l462_462898

noncomputable def triangle := 
  {a b c : ℝ // b = 2 ∧ c = 3 ∧ ∃ θ : ℝ, cos θ = 1 / 3}

theorem triangle_properties {a b c : ℝ} (h : triangle) :
  let θ := classical.some h.2.2 in
  a = 3 ∧ 
  (1/2) * a * b * real.sin θ = 2 * real.sqrt 2 ∧ 
  cos (classical.choose (λ θ₁, cos θ₁ = cos θ)) - 
      cos (θ - classical.choose (λ θ₂, cos θ₂ = cos θ)) = 23 / 27 :=
by
  -- omitted proof
  sorry

end triangle_properties_l462_462898


namespace tetrahedron_condition_proof_l462_462935

/-- Define the conditions for the necessary and sufficient condition for each k -/
def tetrahedron_condition (a : ℝ) (k : ℕ) : Prop :=
  match k with
  | 1 => a < Real.sqrt 3
  | 2 => Real.sqrt (2 - Real.sqrt 3) < a ∧ a < Real.sqrt (2 + Real.sqrt 3)
  | 3 => a < Real.sqrt 3
  | 4 => a > Real.sqrt (2 - Real.sqrt 3)
  | 5 => a > 1 / Real.sqrt 3
  | _ => False -- not applicable for other values of k

/-- Prove that the condition is valid for given a and k -/
theorem tetrahedron_condition_proof (a : ℝ) (k : ℕ) : tetrahedron_condition a k := 
  by
  sorry

end tetrahedron_condition_proof_l462_462935


namespace find_johns_allowance_l462_462388

variable (A : ℝ)  -- John's weekly allowance

noncomputable def johns_allowance : Prop :=
  let arcade_spent := (3 / 5) * A
  let remaining_after_arcade := (2 / 5) * A
  let toy_store_spent := (1 / 3) * remaining_after_arcade
  let remaining_after_toy_store := remaining_after_arcade - toy_store_spent
  let final_spent := 0.88
  final_spent = remaining_after_toy_store → A = 3.30

theorem find_johns_allowance : johns_allowance A := by
  sorry

end find_johns_allowance_l462_462388


namespace greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l462_462360

theorem greatest_divisor_of_sum_first_15_terms_arithmetic_sequence
  (x c : ℕ) -- where x and c are positive integers
  (h_pos_x : 0 < x) -- x is positive
  (h_pos_c : 0 < c) -- c is positive
  : ∃ (d : ℕ), d = 15 ∧ ∀ (S : ℕ), S = 15 * x + 105 * c → d ∣ S :=
by
  sorry

end greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l462_462360


namespace angelina_speed_from_library_to_gym_l462_462042

theorem angelina_speed_from_library_to_gym :
  ∃ (v : ℝ), 
    (840 / v - 510 / (1.5 * v) = 40) ∧
    (510 / (1.5 * v) - 480 / (2 * v) = 20) ∧
    (2 * v = 25) :=
by
  sorry

end angelina_speed_from_library_to_gym_l462_462042


namespace chrysler_floors_difference_l462_462653

theorem chrysler_floors_difference (C L : ℕ) (h1 : C = 23) (h2 : C + L = 35) : C - L = 11 := by
  sorry

end chrysler_floors_difference_l462_462653


namespace cubed_identity_l462_462170

theorem cubed_identity (x : ℝ) (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := 
sorry

end cubed_identity_l462_462170


namespace obtuse_angle_A_DB_of_bisectors_l462_462913

theorem obtuse_angle_A_DB_of_bisectors
  (ABC: Triangle)
  (A B C D : Point)
  (h_right_triangle : is_right_triangle ABC)
  (h_angle_A_45 : angle A = 45)
  (h_angle_B_45 : angle B = 45)
  (h_bisectors_intersect_D : intersects_angle_bisectors A B D) :
  angle ADB = 135 :=
by sorry

end obtuse_angle_A_DB_of_bisectors_l462_462913


namespace candy_cost_correct_l462_462602

-- Given conditions:
def given_amount : ℝ := 1.00
def change_received : ℝ := 0.46

-- Define candy cost based on given conditions
def candy_cost : ℝ := given_amount - change_received

-- Statement to be proved
theorem candy_cost_correct : candy_cost = 0.54 := 
by
  sorry

end candy_cost_correct_l462_462602


namespace hexagon_diagonal_angle_l462_462579

theorem hexagon_diagonal_angle (A B C D E F : Type) [hexagon: regular_hexagon A B C D E F]
  (interior_angle : ∀ {a b c}, a ≠ b ∧ b ≠ c ∧ c ≠ a → angle a b c = 120) :
  angle D A B = 30 :=
  sorry

end hexagon_diagonal_angle_l462_462579


namespace ellipse_properties_l462_462040

theorem ellipse_properties :
  ∃ (a b h k : ℝ), 
  a > 0 ∧ b > 0 ∧ 
  (∀ x y : ℝ, (x, y) = (5, -2) → (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 - h)^2 / a^2 + (p.2 - k)^2 / b^2 = 1)) ∧ 
  h = 0 ∧ 
  k = -2 ∧ 
  a + k = 3 :=
by
  sorry

end ellipse_properties_l462_462040


namespace cyclic_quad_l462_462639

variables {A B C D E : Type}

-- Let \(A, B, C, D\) be points on a circle and \(E\) be the intersection of \(AC\) and \(BD\).
variable (cyclic : ∀ (A B C D : Type), A ∈ circle → B ∈ circle → C ∈ circle → D ∈ circle → Prop)
variable (intersection : E ∈ AC ∧ E ∈ BD)
variable (area_condition : ∀ (A B C E : Type), [ABE] ∙ [CDE] = 36)

theorem cyclic_quad (A B C D E : Type) (h_cyclic : cyclic A B C D) (h_intersection : intersection A B C D E) (h_area_condition : area_condition A B C D) :
  [ADE] ∙ [BCE] = 36 :=
by sorry

end cyclic_quad_l462_462639


namespace casey_saves_money_l462_462059

def first_employee_hourly_wage : ℕ := 20
def second_employee_hourly_wage : ℕ := 22
def subsidy_per_hour : ℕ := 6
def weekly_work_hours : ℕ := 40

theorem casey_saves_money :
  let first_employee_weekly_cost := first_employee_hourly_wage * weekly_work_hours
  let second_employee_effective_hourly_wage := second_employee_hourly_wage - subsidy_per_hour
  let second_employee_weekly_cost := second_employee_effective_hourly_wage * weekly_work_hours
  let savings := first_employee_weekly_cost - second_employee_weekly_cost
  savings = 160 :=
by
  sorry

end casey_saves_money_l462_462059


namespace area_BEIH_l462_462708

noncomputable def point : Type := ℝ × ℝ

-- Define the points
def B : point := (0, 0)
def A : point := (0, 3)
def D : point := (3, 3)
def C : point := (3, 0)
def E : point := (0, 1.5) -- midpoint of AB
def F : point := (1.5, 0) -- midpoint of BC

-- Lines intersection
def line_AF (p : point) : Prop := p.2 = -2 * p.1 + 3
def line_DE (p : point) : Prop := p.2 = 1 / 2 * p.1 + 1.5

-- Points I and H
def I : point := (3 / 5, 9 / 5) -- intersection of AF and DE
def line_BD (p : point) : Prop := p.2 = p.1
def H : point := (1, 1) -- intersection of BD and AF

-- Quadrilateral BEIH
def quadrilateral_area (a b c d : point) : ℝ :=
  0.5 * ((a.1 * b.2 + b.1 * c.2 + c.1 * d.2 + d.1 * a.2) -
         (a.2 * b.1 + b.2 * c.1 + c.2 * d.1 + d.2 * a.1))

-- Proof statement
theorem area_BEIH : quadrilateral_area B E I H = 6 / 5 := by {
  sorry
}

end area_BEIH_l462_462708


namespace norm_sum_eq_l462_462493

-- Define the complex numbers
def c1 : ℂ := 3 - 5 * complex.I
def c2 : ℂ := 3 + 5 * complex.I

-- Define the norm (magnitude) of complex numbers
def norm_c1 : ℝ := complex.abs c1
def norm_c2 : ℝ := complex.abs c2

-- The statement to prove
theorem norm_sum_eq : norm_c1 + norm_c2 = 2 * real.sqrt 34 :=
by sorry

end norm_sum_eq_l462_462493


namespace a8_eq_64_l462_462140

variable (S : ℕ → ℕ)
variable (a : ℕ → ℕ)

axiom a1_eq_2 : a 1 = 2
axiom S_recurrence : ∀ (n : ℕ), S (n + 1) = 2 * S n - 1

theorem a8_eq_64 : a 8 = 64 := 
by
sorry

end a8_eq_64_l462_462140


namespace first_four_terms_l462_462875

-- Sequence definition
def sequence : ℕ → ℚ
| n := (1 + (-1)^(n+1)) / 2

-- Statement to prove the first four terms of the sequence
theorem first_four_terms :
  sequence 1 = 1 ∧
  sequence 2 = 0 ∧
  sequence 3 = 1 ∧
  sequence 4 = 0 := 
by
  -- Proof of each term is omitted and left as a placeholder for now
  sorry

end first_four_terms_l462_462875


namespace decomposition_of_cube_l462_462104

theorem decomposition_of_cube (m : ℕ) (h : m^2 - m + 1 = 73) : m = 9 :=
sorry

end decomposition_of_cube_l462_462104


namespace divide_into_flags_l462_462693

/- A structure to represent a flag on a square, which is any pentagon formed by the vertices of a square and its center -/
structure Flag (square : Type) :=
(center : square)
(vertices : fin 4 → square)

/- Definition of each figure -/
def FigureA (square : Type) := ∃ (flag : Flag square), true
def FigureB (outerSq innerSq : Type) := ∃ (flag : Flag outerSq), ∃ (flag : Flag innerSq), true
def FigureC (smallSquares : fin 9 → Type) := ∃ (flag : fin 9 → Flag (smallSquares _)), true

/- The main theorem which claims that we can divide the figures into flags as defined -/
theorem divide_into_flags (square : Type) :
  (FigureA square) ∧
  (∃ (outerSq innerSq : Type), FigureB outerSq innerSq) ∧
  (∃ (smallSquares : fin 9 → Type), FigureC smallSquares) :=
sorry

end divide_into_flags_l462_462693


namespace rectangular_coordinates_of_transformed_point_l462_462727

theorem rectangular_coordinates_of_transformed_point :
  ∀ (x y : ℝ), sqrt (x * x + y * y) = r →
  r = 10 → 
  θ = real.arccos (x / r) →
  θ = real.arcsin (y / r) →
  x = 8 →
  y = 6 →
  (2 * cos(θ + (real.pi / 4)), 2 * sin(θ + (real.pi / 4))) = (2 * sqrt 2, 14 * sqrt 2) :=
by 
  sorry

end rectangular_coordinates_of_transformed_point_l462_462727


namespace max_area_of_triangle_l462_462924

variable (A B C : Type) [InnerProductSpace ℝ (Type)]
variable (S : ℝ)

def is_triangle (A B C : Type) [InnerProductSpace ℝ (Type)] (BC AB AC : ℝ) : Prop :=
  BC = 2 ∧ (AB * AC * real.cosAngle A B C) = 1

theorem max_area_of_triangle 
  (h : is_triangle A B C 2 (dist A B) (dist A C)) : 
  S ≤ real.sqrt 2 :=
begin
  sorry
end

end max_area_of_triangle_l462_462924


namespace largest_angle_of_triangle_l462_462678

noncomputable def a (m : ℝ) : ℝ := m^2 + m + 1
noncomputable def b (m : ℝ) : ℝ := 2m + 1
noncomputable def c (m : ℝ) : ℝ := m^2 - 1

theorem largest_angle_of_triangle (m : ℝ) (hm : m > 1) :
  ∀ (a = m^2 + m + 1) (b = 2m + 1) (c = m^2 - 1),
  angle_opposite_largest_side a b c = 2 * π / 3 := sorry

end largest_angle_of_triangle_l462_462678


namespace fee_difference_l462_462688

-- Defining the given conditions
def stadium_capacity : ℕ := 2000
def fraction_full : ℚ := 3 / 4
def entry_fee : ℚ := 20

-- Statement to prove
theorem fee_difference :
  let people_at_three_quarters := stadium_capacity * fraction_full
  let total_fees_at_three_quarters := people_at_three_quarters * entry_fee
  let total_fees_full := stadium_capacity * entry_fee
  total_fees_full - total_fees_at_three_quarters = 10000 :=
by
  sorry

end fee_difference_l462_462688


namespace frac_two_over_x_values_l462_462179

theorem frac_two_over_x_values (x : ℝ) (h : 1 - 9 / x + 20 / (x ^ 2) = 0) :
  (2 / x = 1 / 2 ∨ 2 / x = 0.4) :=
sorry

end frac_two_over_x_values_l462_462179


namespace exists_subset_sum_mod_p_l462_462220

theorem exists_subset_sum_mod_p (p : ℕ) (hp : Nat.Prime p) (A : Finset ℕ)
  (hA_card : A.card = p - 1) (hA : ∀ a ∈ A, a % p ≠ 0) : 
  ∀ n : ℕ, n < p → ∃ B ⊆ A, (B.sum id) % p = n :=
by
  sorry

end exists_subset_sum_mod_p_l462_462220


namespace parabola_and_lines_l462_462137

noncomputable def parabola_equation_proof : Prop :=
  ∃ (p y₀ : ℝ), (p > 0) ∧ (x : ℝ) (A : ℝ × ℝ),
  (A = (2 * real.sqrt 3, y₀)) ∧ (p < y₀) ∧
  (dist (2 * real.sqrt 3, y₀) (0, p / 2) = 4) ∧
  (2 * p * y₀ = 12) ∧
  (x ^ 2 = 4 * y₀)

noncomputable def line_passes_through_point_and_area : Prop :=
  ∃ (k b : ℝ), (l₁ : ℝ → ℝ),
  (l₁(x) = k * x + b) ∧
  ∃ D E : ℝ × ℝ, (D ≠ E) ∧
  (D ∈ { (x : ℝ, y : ℝ) | x^2 = 4 * y }) ∧
  (E ∈ { (x : ℝ, y : ℝ) | x^2 = 4 * y }) ∧
  (vector.dot_product D E = -4) ∧
  (b = 2) ∧ (D = (0, 2)) ∧
  ∃ (m : ℝ), (Q : ℝ × ℝ),
  (Q = (-2, 3)) ∧
  ∃ (l₂ : ℝ → ℝ),
  (l₂(x) = (x - (m * 3) + 3 * m + 2)) ∧
  (l₂ Q.1 = Q.2) ∧
  ∃ (F : ℝ × ℝ), (F = (0, 1)) ∧
  (area (F P Q) = 1)

theorem parabola_and_lines :
  parabola_equation_proof ∧ line_passes_through_point_and_area :=
sorry

end parabola_and_lines_l462_462137


namespace max_profit_correct_l462_462409

noncomputable def max_profit (units_A units_B : ℕ) : ℕ :=
  20000 * units_A + 30000 * units_B

theorem max_profit_correct : 
  ∃ units_A units_B, 
  4 * units_A ≤ 16 ∧
  4 * units_B ≤ 12 ∧
  units_A + 2 * units_B ≤ 8 ∧
  max_profit units_A units_B = 140000 ∧
  ∀ units_A' units_B', 
    4 * units_A' ≤ 16 ∧
    4 * units_B' ≤ 12 ∧
    units_A' + 2 * units_B' ≤ 8 →
    max_profit units_A' units_B' ≤ max_profit units_A units_B :=
begin
  use [4, 2],
  iterate 3 { split; linarith },
  split,
  { simp [max_profit] },
  { intros a b h1 h2 h3,
    rw max_profit,
    rw max_profit,
    linarith
  },
  sorry
end

end max_profit_correct_l462_462409


namespace find_y_minus_x_l462_462568

theorem find_y_minus_x (x y : ℝ) (h1 : x + y = 8) (h2 : y - 3 * x = 7) : y - x = 7.5 :=
by
  sorry

end find_y_minus_x_l462_462568


namespace greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l462_462361

theorem greatest_divisor_of_sum_first_15_terms_arithmetic_sequence
  (x c : ℕ) -- where x and c are positive integers
  (h_pos_x : 0 < x) -- x is positive
  (h_pos_c : 0 < c) -- c is positive
  : ∃ (d : ℕ), d = 15 ∧ ∀ (S : ℕ), S = 15 * x + 105 * c → d ∣ S :=
by
  sorry

end greatest_divisor_of_sum_first_15_terms_arithmetic_sequence_l462_462361


namespace infinitely_many_integers_l462_462606

variable (x : ℕ → ℝ)

def recurrence_relation (n : ℕ) : Prop :=
  ∀ (n : ℕ), n ≥ 3 → x n = (x (n-2) * x (n-1)) / (2 * x (n-2) - x (n-1))

theorem infinitely_many_integers
  (h1 : x 1 ≠ 0) (h2 : x 2 ≠ 0) :
  ( ∀ n, n ≥ 3 → recurrence_relation x n) →
  ( ∃∞ n, x n ∈ ℤ ) ↔ (x 1 = x 2 ∧ x 1 ∈ ℤ ∧ x 1 ≠ 0) :=
sorry

end infinitely_many_integers_l462_462606


namespace convex_partition_with_circles_l462_462925

theorem convex_partition_with_circles (Polygon : Type) (Circles : list (Polygon → Prop)) [convex Polygon] [∀ C ∈ Circles, convex C] : 
  (∃ partitions : list (Polygon → Prop), (∀ part ∈ partitions, convex part) ∧ (∀ C ∈ Circles, ∃ part ∈ partitions, C part) ∧ 
  (∃! C part, part ∈ partitions ∧ C ∈ Circles ∧ C part)) :=
sorry

end convex_partition_with_circles_l462_462925


namespace pizza_volume_one_piece_l462_462424

theorem pizza_volume_one_piece
  (thickness : ℝ)
  (diameter : ℝ)
  (pieces : ℝ)
  (h : thickness = 1/2)
  (d : diameter = 16)
  (p : pieces = 8) :
  ∃ (volume_one_piece : ℝ), volume_one_piece = 4 * Real.pi :=
by 
  rcases (pi * (d / 2) ^ 2 * h) / p with v;
  use v;
  sorry

end pizza_volume_one_piece_l462_462424


namespace highest_monthly_profit_max_average_profit_l462_462401

noncomputable def profit (x : ℕ) : ℤ :=
if 1 ≤ x ∧ x ≤ 5 then 26 * x - 56
else if 5 < x ∧ x ≤ 12 then 210 - 20 * x
else 0

noncomputable def average_profit (x : ℕ) : ℝ :=
if 1 ≤ x ∧ x ≤ 5 then (13 * ↑x - 43 : ℤ) / ↑x
else if 5 < x ∧ x ≤ 12 then (-10 * ↑x + 200 - 640 / ↑x : ℝ)
else 0

theorem highest_monthly_profit :
  ∃ m p, m = 6 ∧ p = 90 ∧ profit m = p :=
by sorry

theorem max_average_profit (x : ℕ) :
  1 ≤ x ∧ x ≤ 12 →
  average_profit x ≤ 40 ∧ (average_profit 8 = 40 → x = 8) :=
by sorry

end highest_monthly_profit_max_average_profit_l462_462401


namespace second_derivative_at_pi_over_3_l462_462620

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x) * (Real.cos x)

theorem second_derivative_at_pi_over_3 : 
  (deriv (deriv f)) (Real.pi / 3) = -1 :=
  sorry

end second_derivative_at_pi_over_3_l462_462620


namespace proof_problem1_proof_problem2_l462_462051

noncomputable def problem1_lhs : ℝ :=
-(1^(2022 : ℝ)) - |(2 : ℝ) - real.sqrt (2 : ℝ)| - real.cbrt (-8 : ℝ)

noncomputable def problem1_rhs : ℝ :=
real.sqrt 2 - 1

noncomputable def problem2_lhs : ℝ :=
real.sqrt ((-2 : ℝ)^2) + |1 - real.sqrt (2 : ℝ)| - (2 * real.sqrt (2 : ℝ) - 1)

noncomputable def problem2_rhs : ℝ :=
2 - real.sqrt 2

theorem proof_problem1 : problem1_lhs = problem1_rhs :=
by sorry

theorem proof_problem2 : problem2_lhs = problem2_rhs :=
by sorry

end proof_problem1_proof_problem2_l462_462051


namespace product_slope_y_intercept_eq_zero_l462_462964

variables (x1 x2 : ℝ) (h_diff : x1 ≠ x2)

def point_y (y : ℝ) := ∀ x, ∃ C D : ℝ, C = y ∧ D = y
      
theorem product_slope_y_intercept_eq_zero (hC : point_y 20 x1) (hD : point_y 20 x2) : 
  let slope := (0 : ℝ)
  let y_intercept := (20 : ℝ) in
  slope * y_intercept = 0 :=
by 
  sorry

end product_slope_y_intercept_eq_zero_l462_462964


namespace find_z_l462_462497

theorem find_z (z : ℝ) (h : (z^2 - 5 * z + 6) / (z - 2) + (5 * z^2 + 11 * z - 32) / (5 * z - 16) = 1) : z = 1 :=
sorry

end find_z_l462_462497


namespace area_PQRS_l462_462114

variables {ABCD : Type*} [ConvexQuadrilateral ABCD] 
variables (M : Point) 
variables (P Q R S : Point)
variables (s : ℝ)

-- Convex quadrilateral ABCD with area s
def area_ABCD (h : ConvexQuadrilateral ABCD) : ℝ := s

-- Points P, Q, R, and S are symmetric to point M with respect to midpoints of sides of ABCD
def symmetric_points {ABCD : Type*} [ConvexQuadrilateral ABCD]
  (M P Q R S : Point) : Prop := 
  ∃ P1 Q1 R1 S1 : Point, 
  IsMidpoint P1 A B ∧ 
  IsMidpoint Q1 B C ∧ 
  IsMidpoint R1 C D ∧ 
  IsMidpoint S1 D A ∧ 
  SymmetricAbout M P P1 ∧
  SymmetricAbout M Q Q1 ∧
  SymmetricAbout M R R1 ∧
  SymmetricAbout M S S1

-- The theorem stating the area of the quadrilateral formed by symmetric points
theorem area_PQRS {ABCD : Type*} [ConvexQuadrilateral ABCD] 
  (M P Q R S : Point) (s : ℝ)
  (h : ConvexQuadrilateral ABCD) 
  (h_area : area_ABCD h = s) 
  (h_symmetrical : symmetric_points M P Q R S) :
  area_PQRS P Q R S = 2 * s :=
sorry

end area_PQRS_l462_462114


namespace integral_solution_l462_462495

noncomputable def integral_problem : ℝ :=
  ∫ x in -1..1, (real.sqrt (1 - x^2) + x * real.cos x)

theorem integral_solution :
  integral_problem = (real.pi / 2) :=
by
  sorry

end integral_solution_l462_462495


namespace find_k_l462_462474

theorem find_k : ∃ b k : ℝ, (∀ x : ℝ, (x + b)^2 = x^2 - 20 * x + k) ∧ k = 100 := by
  sorry

end find_k_l462_462474


namespace part1_part2_l462_462154

noncomputable def f := fun x : Real => sin (x - π / 6) + cos (x - π / 3)
noncomputable def g := fun x : Real => 2 * sin (x / 2) ^ 2

theorem part1 (θ : Real) (h : 0 < θ ∧ θ < π / 2) (h₀ : f θ = 3 * sqrt 3 / 5) : g θ = 1 / 5 := 
sorry

theorem part2 :
  {x | ∀ k : Int, (2 * k * π ≤ x ∧ x ≤ 2 * k * π + 2 * π / 3) ↔ f x ≥ g x} :=
sorry

end part1_part2_l462_462154


namespace twenty_five_ounces_of_forty_percent_salt_solution_needed_l462_462884

theorem twenty_five_ounces_of_forty_percent_salt_solution_needed
  (V₁ : ℝ)
  (C₁ C₂ C_desired : ℝ)
  (s : ℝ) :
  V₁ = 50 →
  C₁ = 0.10 →
  C₂ = 0.40 →
  C_desired = 0.20 →
  let total_salt := 0.1 * V₁ + 0.4 * s in
  let total_volume := V₁ + s in
  (total_salt / total_volume = C_desired) →
  s = 25 := sorry

end twenty_five_ounces_of_forty_percent_salt_solution_needed_l462_462884


namespace possible_values_of_a_l462_462546

theorem possible_values_of_a :
  (∀ x, (x^2 - 3 * x + 2 = 0) → (ax - 2 = 0)) → (a = 0 ∨ a = 1 ∨ a = 2) :=
by
  intro h
  sorry

end possible_values_of_a_l462_462546


namespace sufficient_for_orthogonality_A_sufficient_for_orthogonality_C_sufficient_for_orthogonality_D_l462_462879

-- Definitions of orthogonality and parallelism in space
variables (Line Plane : Type) [HasSubset Line Plane] [HasParallel Line Plane] [HasOrthogonal Line Plane Plane]

-- Definitions for lines and planes
variables (m n : Line) (α β : Plane)

-- Condition A
def condition_A : Prop := m ⊥ α ∧ m ∥ β

-- Condition B
def condition_B : Prop := m ⊂ α ∧ n ⊂ β ∧ m ⊥ n

-- Condition C
def condition_C : Prop := m ⊂ α ∧ m ∥ n ∧ n ⊥ β

-- Condition D
def condition_D : Prop := m ⊥ n ∧ m ⊥ α ∧ n ⊥ β

-- Statement to prove each sufficient condition
theorem sufficient_for_orthogonality_A : condition_A m α β → α ⊥ β := sorry

theorem sufficient_for_orthogonality_C : condition_C m n α β → α ⊥ β := sorry

theorem sufficient_for_orthogonality_D : condition_D m n α β → α ⊥ β := sorry

end sufficient_for_orthogonality_A_sufficient_for_orthogonality_C_sufficient_for_orthogonality_D_l462_462879


namespace fraction_to_decimal_l462_462081

theorem fraction_to_decimal : (7 / 12 : ℝ) = 0.5833 + (3 / 10000) * (1 / (1 - (1 / 10))) := 
by sorry

end fraction_to_decimal_l462_462081


namespace area_sum_equal_l462_462604

variable (a x1 y1 x2 y2 x3 y3 x4 y4 : ℝ)

def area_ABB'A' : ℝ := 
  1 / 2 * abs (a * y2 + x2 * y1 - y2 * x1)

def area_CDD'C' : ℝ := 
  1 / 2 * abs (a^2 + x4 * y3 + a * x3 - a * x4 - y4 * x3 - a * y3)

def area_BCC'B' : ℝ := 
  1 / 2 * abs (a^2 + a * y3 + x3 * y2 - a * x3 - y3 * x2 - y2 * a)

def area_DAA'D' : ℝ := 
  1 / 2 * abs (x1 * y4 + a * x4 - y1 * x4)

theorem area_sum_equal :
  area_ABB'A' a x1 y1 x2 y2 = 
  area_BCC'B' a y3 x3 y2 + 
  area_DAA'D' a x4 x4 y1 y4 :=
sorry

end area_sum_equal_l462_462604


namespace find_p_l462_462279

def parabola (p : ℝ) (y x : ℝ) : Prop := y ^ 2 = p * x ∧ p > 0
def line (y x : ℝ) : Prop := y = x - 1

theorem find_p (p : ℝ) (h_chord_length : ¬∃ y1 y2 : ℝ, 
    let x1 := y1 + 1 in
    let x2 := y2 + 1 in
    let chord_length := sqrt (2 * (p ^ 2 + 4 * p)) in
    parabola p y1 x1 ∧ parabola p y2 x2 ∧ chord_length = sqrt 10) : 
  p = 1 :=
sorry

end find_p_l462_462279


namespace greatest_divisor_arithmetic_sequence_sum_l462_462326

theorem greatest_divisor_arithmetic_sequence_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ k, (15 * (x + 7 * c)) = 15 * k :=
sorry

end greatest_divisor_arithmetic_sequence_sum_l462_462326


namespace number_of_ways_to_select_numbers_with_even_sum_l462_462892

theorem number_of_ways_to_select_numbers_with_even_sum :
  ∃ (S : Finset ℕ) (n : ℕ), S ⊆ Finset.range 10 ∧ S.card = 4 ∧ (S.sum id) % 2 = 0 ∧ (Finset.card (Finset.filter (λ S, (S.sum id) % 2 = 0) (Finset.powersetLen 4 (Finset.range 10)))) = 66 :=
by
  let S := Finset.range 10
  existsi S, 4
  split
  { exact Finset.subset.refl _ }
  split
  { exact Finset.card_range 10 }
  split
  { sorry }
  { sorry }


end number_of_ways_to_select_numbers_with_even_sum_l462_462892


namespace part1_inequality_part2_inequality_l462_462977

theorem part1_inequality (x : ℝ) : 
  (3 * x - 2) / (x - 1) > 1 ↔ x > 1 ∨ x < 1 / 2 := 
by sorry

theorem part2_inequality (x a : ℝ) : 
  x^2 - a * x - 2 * a^2 < 0 ↔ 
  (a = 0 → False) ∧ 
  (a > 0 → -a < x ∧ x < 2 * a) ∧ 
  (a < 0 → 2 * a < x ∧ x < -a) := 
by sorry

end part1_inequality_part2_inequality_l462_462977


namespace count_a_values_l462_462933

def d1 (a : ℤ) : ℤ := a^2 + 3^a + a * 3^((a + 1) / 2)
def d2 (a : ℤ) : ℤ := a^2 + 3^a - a * 3^((a + 1) / 2)

def is_multiple_of_3 (n : ℤ) : Prop := ∃ k : ℤ, n = 3 * k

theorem count_a_values (count : ℤ) : count = 34 :=
  count = finset.card (finset.filter (λ a, is_multiple_of_3 (d1 a * d2 a))
    (finset.Icc 1 101)) sorry

end count_a_values_l462_462933


namespace greatest_divisor_of_sum_of_arithmetic_sequence_l462_462353

theorem greatest_divisor_of_sum_of_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ k : ℕ, (sum (λ n, x + n * c) (range 15)) = 15 * k :=
by sorry

end greatest_divisor_of_sum_of_arithmetic_sequence_l462_462353


namespace intersection_A_B_l462_462547

def A : Set ℝ := {x | (2 * x - 1) / (x - 2) < 0}
def B : Set ℕ := Set.univ

theorem intersection_A_B :
  {x : ℕ | x ∈ A} = {1} :=
by
  sorry

end intersection_A_B_l462_462547


namespace possible_age_of_youngest_child_l462_462007

noncomputable def valid_youngest_age (father_fee : ℝ) (child_fee_per_year : ℝ) (total_bill : ℝ) (triplet_age : ℝ) : ℝ :=
  total_bill - father_fee -  (3 * triplet_age * child_fee_per_year)

theorem possible_age_of_youngest_child (father_fee : ℝ) (child_fee_per_year : ℝ) (total_bill : ℝ) (t y : ℝ)
  (h1 : father_fee = 16)
  (h2 : child_fee_per_year = 0.8)
  (h3 : total_bill = 43.2)
  (age_condition : y = (total_bill - father_fee) / child_fee_per_year - 3 * t) :
  y = 1 ∨ y = 4 :=
by
  sorry

end possible_age_of_youngest_child_l462_462007


namespace min_period_sin_squared_l462_462277

theorem min_period_sin_squared :
  ∀ x : ℝ, ∀ T > 0, (∀ x, 2 * sin (x + π / 6) ^ 2 = 2 * sin (x + T + π / 6) ^ 2) ↔ T = π :=
by
  sorry

end min_period_sin_squared_l462_462277


namespace part1_part2_l462_462133

section Problem

variables {α : Type*} [EuclideanGeometry α] [nonempty α]
variables (A B C M N I P T Q I₁ I₂ : α)
variables (hAcute : ∠A < ∠B)
variables (hMidpts : is_midpoint M (arc BC) ∧ is_midpoint N (arc AC)) -- M and N are midpoints
variables (hCircumcircle : is_circumcircle Γ (triangle ABC)) -- circumcircle Γ
variables (hP : line_through C ∥ line_through M N ∧ P ∈ Γ)
variables (hI : incenter I (triangle ABC))
variables (hT : T ∈ Γ ∧ line_through P I ∋ T)
variables (hQ : Q ∈ (arc_sub AB \ {A, T, B}) ∧ incenter I₁ (triangle AQC) ∧ incenter I₂ (triangle QCB))

-- Statement for Part 1
theorem part1 : dist M P * dist M T = dist N P * dist N T := sorry

-- Statement for Part 2
theorem part2 : cyclic Q I₁ I₂ T := sorry

end Problem

end part1_part2_l462_462133


namespace find_lambda_l462_462881

def vector_a (λ : ℝ) : ℝ × ℝ × ℝ := (λ, 0, -1)
def vector_b (λ : ℝ) : ℝ × ℝ × ℝ := (2, 5, λ^2)

def perpendicular (a b : ℝ × ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 0

theorem find_lambda (λ : ℝ) :
  perpendicular (vector_a λ) (vector_b λ) → (λ = 0 ∨ λ = 2) :=
by 
  unfold perpendicular vector_a vector_b
  sorry

end find_lambda_l462_462881


namespace radius_of_spheres_l462_462718

/--
If a cylindrical container is filled with water to a height of 8 cm, and three identical
spheres are placed into it causing the water level to rise and cover the topmost sphere,
then the radius of the spheres is 4 cm.
-/
theorem radius_of_spheres (h_initial : ℝ) 
  (sphere_vol : ℝ → ℝ )
  (h_final : ℝ)
  (height_increase : ℝ)
  (r : ℝ)
  (V_sphere : ℝ → ℝ) 
  (cover_spheres : r = height_increase / 2) 
  (height_increase_eq : height_increase = h_final - h_initial)
  (vol : sphere_vol r) :
  h_initial = 8 → 
  height_increase = 8 → 
  sphere_vol = 3 * (4 / 3 * real.pi * r ^ 3) →
  h_final = h_initial + height_increase →
  r = 4 :=
by
  sorry

end radius_of_spheres_l462_462718


namespace inequalities_lemma_l462_462690

-- Defining the main theorem within the context of natural numbers excluding zero
theorem inequalities_lemma (n : ℕ) (h : n > 0) :
  2 * (Real.sqrt (↑n + 1) - 1) < (1 + ∑ k in Finset.range n, 1 / Real.sqrt (↑k + 2)) ∧ 
  (1 + ∑ k in Finset.range n, 1 / Real.sqrt (↑k + 2)) < 2 * Real.sqrt n :=
sorry

end inequalities_lemma_l462_462690


namespace boundary_length_of_figure_l462_462729

theorem boundary_length_of_figure (a : ℝ) (h_a : a * (3 * a) = 72) : 
  let shorter_side := 2 * Real.sqrt 6,
      longer_side := 6 * Real.sqrt 6,
      segment_short := (Real.sqrt 6) / 2,
      segment_long := (3 * Real.sqrt 6) / 2,
      arc_length := Real.pi * (Real.sqrt 6),
      straight_length := 8 * Real.sqrt 6,
      boundary_length := 8 * Real.sqrt 6 + Real.pi * Real.sqrt 6,
      approximation := boundary_length ≈ 27.3 in
  approximation :=
begin
  sorry
end

end boundary_length_of_figure_l462_462729


namespace center_of_circle_is_correct_minimum_tangent_length_from_line_to_circle_l462_462583

noncomputable def line_parametric_eq (t : ℝ) : ℝ × ℝ :=
  ( (sqrt 2 / 2) * t, (sqrt 2 / 2) * t + 4 * sqrt 2 )

noncomputable def circle_polar_eq (θ : ℝ) : ℝ :=
  2 * cos (θ + (π/4))

def circle_center_cartesian : ℝ × ℝ :=
  (sqrt 2 / 2, - sqrt 2 / 2)

theorem center_of_circle_is_correct :
  ∃ center : ℝ × ℝ, 
    center = circle_center_cartesian :=
sorry

noncomputable def tangent_length (t : ℝ) : ℝ :=
  sqrt ((t^2 + 8*t + 40))

theorem minimum_tangent_length_from_line_to_circle :
  ∃ min_length : ℝ,
    min_length = 2 * sqrt 6 :=
sorry

end center_of_circle_is_correct_minimum_tangent_length_from_line_to_circle_l462_462583


namespace price_increase_l462_462711

-- Definitions of the conditions
variable {P : ℝ} (h_pos : 0 < P)

-- We need to prove that the percentage increase needed to make the prices of the two items equal is 25%
theorem price_increase : 
  let cheaper_item_price := 0.80 * P in
  let increase_needed := P - cheaper_item_price in
  let percentage_increase := (increase_needed / cheaper_item_price) * 100 in
  percentage_increase = 25 :=
sorry

end price_increase_l462_462711


namespace fred_money_last_week_l462_462603

-- Definitions for the conditions in the problem
variables {f j : ℕ} (current_fred : ℕ) (current_jason : ℕ) (last_week_jason : ℕ)
variable (earning : ℕ)

-- Conditions
axiom Fred_current_money : current_fred = 115
axiom Jason_current_money : current_jason = 44
axiom Jason_last_week_money : last_week_jason = 40
axiom Earning_amount : earning = 4

-- Theorem statement: prove Fred's money last week
theorem fred_money_last_week (current_fred last_week_jason current_jason earning : ℕ)
  (Fred_current_money : current_fred = 115)
  (Jason_current_money : current_jason = 44)
  (Jason_last_week_money : last_week_jason = 40)
  (Earning_amount : earning = 4)
  : current_fred - earning = 111 :=
sorry

end fred_money_last_week_l462_462603


namespace slope_tangent_exp_at_A_l462_462287

theorem slope_tangent_exp_at_A : 
  ∀ (x : ℝ), 
  (0 : ℝ) = 0 → 
  y = (λ x, exp x) → 
  y' (0) = 1 := 
by
  sorry

end slope_tangent_exp_at_A_l462_462287


namespace odd_function_extension_monotonicity_range_m_l462_462521

noncomputable def f (x : ℝ) : ℝ := if x ≤ 0 then x^2 + 2*x else -x^2 + 2*x

theorem odd_function_extension :
  ∀ x : ℝ, f(x) = (if x ≤ 0 then x^2 + 2*x else -x^2 + 2*x) := by
sorry

theorem monotonicity_range_m (m : ℝ) :
  (∀ x : ℝ, x ∈ Icc (-1 : ℝ) (m - 1) → (differentiable ℝ f) ∧ (deriv f x > 0)) →
  0 < m ∧ m ≤ 2 := by
sorry

end odd_function_extension_monotonicity_range_m_l462_462521


namespace length_of_bridge_l462_462276

-- Define the necessary conditions
def train_length : ℝ := 150 -- in meters
def train_speed_km_hr : ℝ := 45 -- in km/hr
def train_speed_m_s : ℝ := train_speed_km_hr * 1000 / 3600 -- converting km/hr to m/s
def crossing_time : ℝ := 30 -- in seconds

-- Define the problem to prove the length of the bridge
theorem length_of_bridge :
  let distance_covered := train_speed_m_s * crossing_time in
  ∃ L : ℝ, train_length + L = distance_covered ∧ L = 225 := 
by
  sorry

end length_of_bridge_l462_462276


namespace total_volume_of_removed_tetrahedra_l462_462068

-- Define the unit cube and the transformation of its faces into regular hexagons
structure UnitCube := (side : ℝ) (volume : ℝ)
def unit_cube := UnitCube 1 1

-- Define the tetrahedron that is sliced off from each corner of the cube
structure Tetrahedron := (volume : ℝ)

-- Function to compute total volume of removed tetrahedra
def total_removed_volume (unit_cube : UnitCube) (num_tetrahedra : ℕ) (tetrahedron_volume : ℝ) : ℝ :=
  num_tetrahedra * tetrahedron_volume

-- Define the number of tetrahedra removed from the unit cube
def num_tetrahedra_removed : ℕ := 8

-- Volume of a single removed tetrahedron
noncomputable def removed_tetrahedron_volume : ℝ := (6 - 4 * real.sqrt 3) / 5

-- The statement we need to prove
theorem total_volume_of_removed_tetrahedra :
  total_removed_volume unit_cube num_tetrahedra_removed removed_tetrahedron_volume = 8 * ((6 - 4 * real.sqrt 3) / 5) :=
by sorry

end total_volume_of_removed_tetrahedra_l462_462068


namespace f_monotonicity_m_range_l462_462862

noncomputable def f (x : ℝ) := log (2 ^ x - 1) / log 2
noncomputable def g (x : ℝ) := log (2 ^ x + 1) / log 2

-- Define the monotonicity proof problem for f(x)
theorem f_monotonicity : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f(x) < f(y) :=
by
  intro x y hx hy hxy
  sorry  -- Proof that f(x) is monotonically increasing on (0, +∞)

-- Define the range proof problem for m
theorem m_range : ∃ x : ℝ, x ∈ Icc 1 2 → ∃ m : ℝ, m ∈ Icc (log (5 / 3) / log 2) (log 3 / log 2) ∧ g(x) = m + f(x) :=
by
  intro x hx
  use g(x) - f(x)
  split
  . interval
  sorry  -- Proof that there exists a solution m in the range given the conditions

#check f_monotonicity
#check m_range

end f_monotonicity_m_range_l462_462862


namespace birds_total_distance_l462_462743

-- Define the speeds of the birds
def eagle_speed : ℕ := 15
def falcon_speed : ℕ := 46
def pelican_speed : ℕ := 33
def hummingbird_speed : ℕ := 30

-- Define the flying time for each bird
def flying_time : ℕ := 2

-- Calculate the total distance flown by all birds
def total_distance_flown : ℕ := (eagle_speed * flying_time) +
                                 (falcon_speed * flying_time) +
                                 (pelican_speed * flying_time) +
                                 (hummingbird_speed * flying_time)

-- The goal is to prove that the total distance flown by all birds is 248 miles
theorem birds_total_distance : total_distance_flown = 248 := by
  -- Proof here
  sorry

end birds_total_distance_l462_462743


namespace find_a_plus_b_l462_462221

variable {a b c m n : ℝ}
variable {F1 P F2 : ℝ}
variable (hyperbola_eq : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
variable (a_pos : a > 0)
variable (b_pos : b > 0)
variable (eccentricity : c / a = 5 / 4)
variable (angle_F1PF2 : ∠F1 P F2 = Real.pi / 2)
variable (area_triangle : 1/2 * (F1 P * P F2 * sin (angle_F1PF2)) = 9)

theorem find_a_plus_b : a + b = 7 := by
  sorry

end find_a_plus_b_l462_462221


namespace polynomial_b_value_l462_462504

noncomputable def find_b (a b c d : ℝ) : ℂ := 
  let f := λ x : ℂ, x^4 + a*x^3 + b*x^2 + c*x + d
  in if h : f.has_root_complex_nonreal (13 + 4*i) then 
       b else 0

theorem polynomial_b_value (a b c d : ℝ) (z w : ℂ) (hz : z * w = 13 + i) 
    (hw : z + w = 3 + 4*i) (hconjz : conj z = z) (hconjw : conj w = w)
    (hroots : polynomial.has_roots_real_coeff f (z, w) (conj z, conj w)) : 
  find_b a b c d = 51 :=
by 
  sorry

end polynomial_b_value_l462_462504


namespace subtraction_in_base8_correct_l462_462496

def base8_to_nat (digits : List ℕ) : ℕ :=
  digits.reverse.foldl (λ acc d, acc * 8 + d) 0

def 4725₈ := base8_to_nat [4, 7, 2, 5]
def 2367₈ := base8_to_nat [2, 3, 6, 7]

def result_in_base8 := base8_to_nat [2, 3, 3, 6]

theorem subtraction_in_base8_correct :
  4725₈ - 2367₈ = 1246 := by
  sorry

end subtraction_in_base8_correct_l462_462496


namespace greatest_divisor_of_sum_of_arithmetic_sequence_l462_462349

theorem greatest_divisor_of_sum_of_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ k : ℕ, (sum (λ n, x + n * c) (range 15)) = 15 * k :=
by sorry

end greatest_divisor_of_sum_of_arithmetic_sequence_l462_462349


namespace greatest_divisor_of_arithmetic_sequence_sum_l462_462340

theorem greatest_divisor_of_arithmetic_sequence_sum :
  ∀ (x c : ℕ), ∃ k : ℕ, k = 15 ∧ 15 ∣ (15 * x + 105 * c) :=
by
  intro x c
  exists 15
  split
  . rfl
  . sorry

end greatest_divisor_of_arithmetic_sequence_sum_l462_462340


namespace range_of_x_l462_462827

  variable (a x : Real)

  def p := x^2 - 4 * a * x + 3 * a^2 < 0
  def q := 2 < x ∧ x ≤ 3

  theorem range_of_x (h₁ : a = 1) (h₂ : p ∧ q) : 2 < x ∧ x < 3 :=
  by sorry
  
end range_of_x_l462_462827


namespace medians_perpendicular_DE_l462_462624

theorem medians_perpendicular_DE (D E F : Type)
  [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (DP EQ : ℝ)
  (h_DP : DP = 15)
  (h_EQ : EQ = 20)
  (h_perpendicular : ∀ G : MetricSpace.point, ∀ D E : Set (MetricSpace.point G),
    D ≠ E → ∀ P Q : G, MetricSpace.distance D P = DP → MetricSpace.distance E Q = EQ →
    MetricSpace.angle (D - P) (E - Q) = π / 2) :
  MetricSpace.distance D E = 50 / 3 :=
by
  sorry

end medians_perpendicular_DE_l462_462624


namespace how_many_integers_l462_462103

theorem how_many_integers (x : ℤ) : 
  {x : ℤ | x^4 - 62 * x^2 + 60 < 0}.finite.to_finset.card = 12 := 
by 
  sorry

end how_many_integers_l462_462103


namespace midpoint_of_translated_segment_l462_462258

def Point : Type := (ℤ × ℤ)

def midpoint (p1 p2 : Point) : Point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def translate (p : Point) (dx dy : ℤ) : Point :=
  (p.1 + dx, p.2 + dy)

theorem midpoint_of_translated_segment :
  midpoint (translate (midpoint (4, 1) (-8, 5)) 2 3) (translate (midpoint (4, 1) (-8, 5)) 2 3) = (0, 6) :=
by
  sorry

end midpoint_of_translated_segment_l462_462258


namespace probability_four_of_six_show_same_value_l462_462975

open Nat

theorem probability_four_of_six_show_same_value :
  (let total_outcomes := 6^4 in
   let successful_outcomes := 676 in
   (successful_outcomes.toRational / total_outcomes.toRational) = (169/324)) :=
by
  let total_outcomes := 6^4
  let successful_outcomes := 676
  show (successful_outcomes.toRational / total_outcomes.toRational) = (169 / 324)
  sorry

end probability_four_of_six_show_same_value_l462_462975


namespace find_modulus_of_z_l462_462523

theorem find_modulus_of_z (z : ℂ) (h : z * (1 + complex.I) = complex.I) : 
  |z| = real.sqrt 2 / 2 :=
sorry

end find_modulus_of_z_l462_462523


namespace normal_pdf_integral_l462_462250

-- Define the probability density function for a normal distribution
def normal_pdf (x a σ : ℝ) : ℝ :=
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(x - a)^2 / (2 * σ^2))

-- Prove the function defining the density of the normal distribution
-- integrates to 1 over the entire real line.
theorem normal_pdf_integral {a σ : ℝ} (hσ : σ > 0) :
  ∫ (x : ℝ) in -∞..∞, normal_pdf x a σ = 1 :=
  sorry

end normal_pdf_integral_l462_462250


namespace g_neg_x_eq_neg_g_x_l462_462229

-- Define the function g(x)
def g (x : ℝ) : ℝ := (x^2 + 1) / (x - 1)

-- The main statement to be proved
theorem g_neg_x_eq_neg_g_x (x : ℝ) (h : x^2 ≠ 1) : g (-x) = -g x :=
by
  -- details of the proof go here
  sorry

end g_neg_x_eq_neg_g_x_l462_462229


namespace area_parallelogram_is_960_l462_462790

-- Define the base and height
def base : ℝ := 60
def height : ℝ := 16

-- Define the area formula for a parallelogram
def area_of_parallelogram (b h : ℝ) : ℝ := b * h

-- Prove the area is 960 cm² given the base and height
theorem area_parallelogram_is_960 : area_of_parallelogram base height = 960 :=
by
  unfold area_of_parallelogram
  sorry

end area_parallelogram_is_960_l462_462790


namespace number_of_incorrect_conditions_l462_462035

theorem number_of_incorrect_conditions :
  (1 ∈ ({0, 1, 2} : Set ℕ)) ∧
  (∅ ⊆ ({0, 1, 2} : Set ℕ)) ∧
  (¬ ({1} ∈ ({0, 1, 2} : Set ℕ))) ∧
  ({0, 1, 2} = {2, 0, 1}) →
  ∑ b in ({(1 ∈ ({0, 1, 2} : Set ℕ)),
            (∅ ⊆ ({0, 1, 2} : Set ℕ)),
            (¬ ({1} ∈ ({0, 1, 2} : Set ℕ))),
            ({0, 1, 2} = {2, 0, 1})} : Finset Bool), if b then 1 else 0) = 1 := by
  sorry

end number_of_incorrect_conditions_l462_462035


namespace inequality_solution_sets_l462_462127

theorem inequality_solution_sets (a : ℝ) (h : a > 1) :
  ∀ x : ℝ, ((a = 2 → (x ≠ 1 → (a-1)*x*x - a*x + 1 > 0)) ∧
            (1 < a ∧ a < 2 → (x < 1 ∨ x > 1/(a-1) → (a-1)*x*x - a*x + 1 > 0)) ∧
            (a > 2 → (x < 1/(a-1) ∨ x > 1 → (a-1)*x*x - a*x + 1 > 0))) :=
by
  sorry

end inequality_solution_sets_l462_462127


namespace cyclic_quadrilateral_angle_correct_l462_462876

noncomputable def cyclic_quadrilateral_angle (a b c d : ℝ) : ℝ :=
  arccos ((a^2 + b^2 - d^2 - c^2) / (2 * (a * b + d * c)))

theorem cyclic_quadrilateral_angle_correct (a b c d : ℝ) (h1 : a^2 + b^2 - d^2 - c^2 ≠ 0) (h2 : 2 * (a * b + d * c) ≠ 0) :
  ∃ θ : ℝ, θ = cyclic_quadrilateral_angle a b c d := by
  sorry

end cyclic_quadrilateral_angle_correct_l462_462876


namespace magnitude_of_b_l462_462846

variable {a b : EuclideanSpace ℝ (Fin 2)} -- Assuming a, b are vectors in a 2D Euclidean space for simplicity

axiom abs_a_eq_2 : ∥a∥ = 2
axiom perpendicular : (a - b) ⬝ a = 0
axiom angle_pi_over_6 : angle a b = π / 6

noncomputable def abs_b : ℝ := ∥b∥

theorem magnitude_of_b : abs_b = 4 * sqrt(3) / 3 :=
by
  sorry

end magnitude_of_b_l462_462846


namespace product_of_non_primitive_roots_eq_one_mod_p_squared_l462_462165

theorem product_of_non_primitive_roots_eq_one_mod_p_squared (p : ℕ) (hp : Nat.prime p) :
  let S := {g | 1 ≤ g ∧ g ≤ p^2 ∧ IsPrimitiveRoot g p ∧ ¬IsPrimitiveRoot g (p^2)} in
  (∏ g in S, g) ≡ 1 [MOD p^2] := 
by
  sorry

end product_of_non_primitive_roots_eq_one_mod_p_squared_l462_462165


namespace sum_of_n_and_k_l462_462988

open Nat

theorem sum_of_n_and_k (n k : ℕ) (h1 : (n.choose (k + 1)) = 3 * (n.choose k))
                      (h2 : (n.choose (k + 2)) = 2 * (n.choose (k + 1))) :
    n + k = 7 := by
  sorry

end sum_of_n_and_k_l462_462988


namespace range_f_area_ABC_l462_462823

-- Part (I)
def f (x : ℝ) : ℝ := sin (2 * x) - 2 * sqrt 3 * (sin x) ^ 2 + 2 * sqrt 3

theorem range_f (x : ℝ) (hx : x ∈ Icc (-π / 3) (π / 6)) : 
  f x ∈ Icc 0 (2 + sqrt 3) :=
sorry

-- Part (II)
variables (A B C : ℝ) (a b c : ℝ)
variables (hA : f A = sqrt 3)
variables (hB : sin B = 3 / 5) (hb : b = 2)
variables (acute : A + B + C = π ∧ A < π / 2 ∧ B < π / 2 ∧ C < π / 2)

theorem area_ABC : 
  let sinA := sin (π / 12),
      cosA := cos (π / 12),
      sinC := sin (π / 12 + B) in
  1 / 2 * b * (b * sinA / sin B) * sinC = (11 - 4 * sqrt 3) / 6 :=
sorry

end range_f_area_ABC_l462_462823


namespace min_value_sqrt_expression_l462_462112

--Define the conditions and the main statement
theorem min_value_sqrt_expression (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 1) : 
    √(x + 1/x) + √(y + 1/y) ≥ √10 := 
sorry -- Proof is omitted

end min_value_sqrt_expression_l462_462112


namespace percentage_of_salt_in_solution_A_l462_462432

theorem percentage_of_salt_in_solution_A (x : ℝ) :
  (28 * x + 112 * 0.9 = 140 * 0.8) → (x = 0.4) :=
begin
  intros h,
  sorry
end

end percentage_of_salt_in_solution_A_l462_462432


namespace count_18_tuples_l462_462102

theorem count_18_tuples : 
  (∃ (n : ℕ), 
    n = 8 ∧ 
    (∀ (a : Fin 18 → {-1, 0, 1}), 
      (∑ i, (2^i : ℤ) * a i = 2^10 → 
       n = ∃ (count : ℕ) (tuples : Fin (count) → Fin 18 → {-1, 0, 1}), 
         (∀ j, (∑ i, (2^i : ℤ) * tuples j i = 2^10))))) :=
sorry

end count_18_tuples_l462_462102


namespace second_square_area_l462_462902

theorem second_square_area (h : isosceles_right_triangle)
                           (area_first_square : 484 = 484) :
  let s := 22
  let leg := 2 * s
  let hypotenuse := leg * Real.sqrt 2
  let S := hypotenuse / 3 in
  (S * S) = 3872 / 9 :=
by
  sorry

end second_square_area_l462_462902


namespace angle_relationship_l462_462929

noncomputable theory

variables {A B C O P1 P2 R S Q1 Q2 U : Point}
-- Definitions for equilateral triangle center and circle
def is_center (O : Point) (A B C : Point) : Prop :=
  equilateral_triangle A B C ∧ centroid O A B C

-- Definitions for the circle
def on_circle (P : Point) (B O C : Point) : Prop := 
  circle_path (B O C) P

-- Given points P1 and P2 on circle BOC, not B, O, and C
axiom P1_on_circle : on_circle P1 B O C ∧ P1 ≠ B ∧ P1 ≠ O ∧ P1 ≠ C
axiom P2_on_circle : on_circle P2 B O C ∧ P2 ≠ B ∧ P2 ≠ O ∧ P2 ≠ C

-- Given the order of points on circle
axiom ordered_points_on_circle (P : Point) :
  ∃ point_list : list Point, point_list = [B, P1, P2, O, C] ∧ (∀ i, (i < (length_point_list )) → (P = ith_point point_list i))

-- Given extensions intersect at R and S
axiom extensions_intersect (BP1 CP1 : Line) :
  ∃ R S : Point, 
    BP1_C_extension BP1 (C A) R ∧
    BP1_B_extension CP1 (A B) S

-- Given definitions of Q1 and Q2
axiom A_P1_line (A P1 : Point) : Line AP1 
axiom RS_line (R S : Point) : Line RS 
axiom Q1_intersection (AP1_line RS_line : Line) :
  ∃ Q1 : Point, intersection_point AP1_line RS_line Q1

axiom Q2_definition (analogous_to_Q1 P2 : Point) :
  ∃ Q2 : Point, definition_based_on_Q1 P2 Q2

-- Given circle intersections
axiom circle_intersections (O Q1 Q2 P1 P2 : Point) :
  ∃ U : Point, 
    circle_intersection (O P1 Q1) (O P2 Q2) U ∧ U ≠ O

-- Prove the required angle relationship
theorem angle_relationship (A B C O P1 P2 R S Q1 Q2 U : Point)
  [is_center O A B C]
  [P1_on_circle P1] 
  [P2_on_circle P2]
  [ordered_points_on_circle P]
  [extensions_intersect BP1 CP1]
  [Q1_intersection AP1_line RS_line]
  [Q2_definition P2]
  [circle_intersections O Q1 Q2 P1 P2] :
  2 * ∠ Q2 U Q1 + ∠ Q2 O Q1 = 360 :=
sorry

end angle_relationship_l462_462929


namespace unique_solution_for_x_3_l462_462812

theorem unique_solution_for_x_3 :
  ∃! (a : ℕ), (16 ≤ a ∧ a < 20) ∧ ∀ (x : ℕ), x > 0 → (3 * x > 4 * x - 4 ∧ 4 * x - a > -8) → x = 3 :=
sorry

end unique_solution_for_x_3_l462_462812


namespace expression_defined_iff_l462_462816

theorem expression_defined_iff (x : ℝ) : 
  (∃ y, y = \frac{\log{(9-x^2)}}{\sqrt{2x-1}}) ↔ (\frac{1}{2} < x ∧ x < 3) := 
by {
  sorry
}

end expression_defined_iff_l462_462816


namespace compute_expression_l462_462460

theorem compute_expression : 45 * 28 + 72 * 45 = 4500 :=
by
  sorry

end compute_expression_l462_462460


namespace percentage_calculation_l462_462799

theorem percentage_calculation (P : ℝ) : 
    (P / 100) * 24 + 0.10 * 40 = 5.92 ↔ P = 8 :=
by 
    sorry

end percentage_calculation_l462_462799


namespace parabola_intersections_l462_462066

def conditions (a b : ℤ) : Prop :=
  a ∈ {-2, -1, 0, 1, 2} ∧ b ∈ {-3, -2, -1, 1, 2, 3}

theorem parabola_intersections : 
  (∑ a in {-2, -1, 0, 1, 2}, 
   ∑ b in {-3, -2, -1, 1, 2, 3}, ∑ a' in {-2, -1, 0, 1, 2}, 
   ∑ b' in {-3, -2, -1, 1, 2, 3}, if a ≠ a' ∨ b ≠ b' then 2 else 0)
   = 810 :=
by
  sorry

end parabola_intersections_l462_462066


namespace symmetry_center_sum_l462_462264

noncomputable def f (x : ℝ) : ℝ := x + Real.sin (Real.pi * x) - 3

theorem symmetry_center_sum : 
  (∑ k in Finset.range (4032), f (k / 2016)) = -8062 :=
by
  sorry

end symmetry_center_sum_l462_462264


namespace greatest_divisor_sum_of_first_fifteen_terms_l462_462324

theorem greatest_divisor_sum_of_first_fifteen_terms 
  (x c : ℕ) (hx : x > 0) (hc : c > 0):
  ∃ d, d = 15 ∧ d ∣ (15*x + 105*c) :=
by
  existsi 15
  split
  . refl
  . apply Nat.dvd.intro
    existsi (x + 7*c)
    refl
  sorry

end greatest_divisor_sum_of_first_fifteen_terms_l462_462324


namespace range_of_f_range_of_ac_plus_bd_l462_462149

section
-- Define the function f(x)
def f (x : ℝ) : ℝ := Real.sqrt (|x + 3| - |x - 1| + 5)

-- Prove the range of f(x)
theorem range_of_f : Set.range f = Set.Icc 1 3 := sorry

-- Given conditions
variables (a b c d : ℝ)
variable (cond1 : a^2 + b^2 = 1)
variable (cond2 : c^2 + d^2 = 3)

-- Prove the range of ac + bd
theorem range_of_ac_plus_bd : ac + bd ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) := 
  sorry
end

end range_of_f_range_of_ac_plus_bd_l462_462149


namespace part1_part2_l462_462863

noncomputable def f (x : ℝ) : ℝ := Real.log2 ((1 + x) / (1 - x))

theorem part1 (h1 : f (3/5) = Real.log2 4) (h2 : f (-3/5) = -Real.log2 4) : f (3/5) = 2 ∧ f (-3/5) = -2 := by
  sorry

theorem part2 : ∀ x : ℝ, f (-x) = -f x := by
  sorry

end part1_part2_l462_462863


namespace all_liars_l462_462956

def num_players : ℕ := 11

def is_knight (n : ℕ) : Prop := sorry -- To be defined based on player behavior

def num_knights := {n : ℕ // is_knight n}.subtype.count

def statement_1 := num_knights = num_players - num_knights
def statement_2 := (num_knights - (num_players - num_knights)).nat_abs = 1
def statement_11 := (num_knights - (num_players - num_knights)).nat_abs = 10

theorem all_liars : 
    ¬ statement_1 ∧ ¬ statement_2 ∧ ¬ statement_11 → num_knights = 0 :=
sorry

end all_liars_l462_462956


namespace number_of_young_teachers_selected_l462_462431

theorem number_of_young_teachers_selected 
  (total_teachers elderly_teachers middle_aged_teachers young_teachers sample_size : ℕ)
  (h_total: total_teachers = 200)
  (h_elderly: elderly_teachers = 25)
  (h_middle_aged: middle_aged_teachers = 75)
  (h_young: young_teachers = 100)
  (h_sample_size: sample_size = 40)
  : young_teachers * sample_size / total_teachers = 20 := 
sorry

end number_of_young_teachers_selected_l462_462431


namespace log_expression_equality_l462_462079

noncomputable def evaluate_log_expression : Real :=
  let log4_8 := (Real.log 8) / (Real.log 4)
  let log5_10 := (Real.log 10) / (Real.log 5)
  Real.sqrt (log4_8 + log5_10)

theorem log_expression_equality : 
  evaluate_log_expression = Real.sqrt ((5 / 2) + (Real.log 2 / Real.log 5)) :=
by
  sorry

end log_expression_equality_l462_462079


namespace b_coordinates_bc_equation_l462_462534

section GeometryProof

-- Define point A
def A : ℝ × ℝ := (1, 1)

-- Altitude CD has the equation: 3x + y - 12 = 0
def altitude_CD (x y : ℝ) : Prop := 3 * x + y - 12 = 0

-- Angle bisector BE has the equation: x - 2y + 4 = 0
def angle_bisector_BE (x y : ℝ) : Prop := x - 2 * y + 4 = 0

-- Coordinates of point B
def B : ℝ × ℝ := (-8, -2)

-- Equation of line BC
def line_BC (x y : ℝ) : Prop := 9 * x - 13 * y + 46 = 0

-- Proof statement for the coordinates of point B
theorem b_coordinates : ∃ x y : ℝ, (x, y) = B :=
by sorry

-- Proof statement for the equation of line BC
theorem bc_equation : ∃ (f : ℝ → ℝ → Prop), f = line_BC :=
by sorry

end GeometryProof

end b_coordinates_bc_equation_l462_462534


namespace smallest_common_multiple_l462_462699

open Nat

theorem smallest_common_multiple : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))) = 2520 :=
by
  sorry

end smallest_common_multiple_l462_462699


namespace kelly_needs_to_give_away_l462_462215

-- Definition of initial number of Sony games and desired number of Sony games left
def initial_sony_games : ℕ := 132
def desired_remaining_sony_games : ℕ := 31

-- The main theorem: The number of Sony games Kelly needs to give away to have 31 left
theorem kelly_needs_to_give_away : initial_sony_games - desired_remaining_sony_games = 101 := by
  sorry

end kelly_needs_to_give_away_l462_462215


namespace find_d_l462_462745

def ellipse_tangent_problem (d : ℝ) : Prop :=
  let F₁ := (7, 3)
  let C := (d, 3)
  ∃ a b : ℝ, 
    a = d - 7 ∧ b = 3 ∧
    ∀ T : ℝ × ℝ, 
      (T = (d, 0) → T.1 ≠ F₁.1 → 
        (|val (T.1 - F₁.1) + |val (T.2 - F₁.2) = 2 * abs a)) 

theorem find_d (d : ℝ) (h : ellipse_tangent_problem d) : d = 11 :=
by
  sorry

end find_d_l462_462745


namespace sum_pq_l462_462415

noncomputable def fold_coordinates (p q : ℕ) : ℕ := p + q

def condition_1 (a b c d : ℕ) := 
    (a = 1) ∧ (b = 3) ∧ (c = 5) ∧ (d = 1)

def condition_2 (a b c d : ℕ) := 
    (a = 6) ∧ (b = 4)

theorem sum_pq (p q : ℕ) 
    (h1 : condition_1 1 3 5 1) 
    (h2 : condition_2 6 4 (6+14-2*q) q) :
    fold_coordinates p q = 6.7 :=
sorry

end sum_pq_l462_462415


namespace grape_rate_per_kg_l462_462752

theorem grape_rate_per_kg (G : ℝ) : 
    (8 * G) + (9 * 55) = 1055 → G = 70 := by
  sorry

end grape_rate_per_kg_l462_462752


namespace divide_plane_into_symmetric_polygons_l462_462209

-- Define a condition for a polygon to be symmetric by a specific rotation
def symmetric_polygon (p : Polygon) (θ : ℝ) : Prop :=
  ∃ c : Point, ∀ v ∈ p.vertices, rotate_about c θ v ∈ p.vertices

-- Define a condition for all sides to be greater than a specific length
def all_sides_longer_than (p : Polygon) (length : ℝ) : Prop :=
  ∀ e ∈ p.edges, e.length > length

-- Define the main proposition
theorem divide_plane_into_symmetric_polygons :
  ∃ decomposition : set (set Point), 
  (∀ p ∈ decomposition, symmetric_polygon p (2 * Real.pi / 7) ∧ all_sides_longer_than p 1) ∧
  (⋃₀ decomposition = set.univ) := 
sorry

end divide_plane_into_symmetric_polygons_l462_462209


namespace original_rectangle_perimeter_proof_l462_462419

-- Define the condition that each smaller rectangle has an integer perimeter
def has_integer_perimeter (w l : ℕ) : Prop :=
  2 * (w + l) ∈ ℕ

-- Define the original problem setup function
noncomputable def original_rectangle_perimeter_integer (L W : ℕ) 
  (h : ∀ (i j : ℕ) (h1: 0 ≤ i < 7) (h2: 0 ≤ j < 7), has_integer_perimeter (L / 7) (W / 7)) 
  : Prop := 
  2 * (L + W) ∈ ℕ

-- Final theorem to prove the problem statement
theorem original_rectangle_perimeter_proof (L W : ℕ) 
  (h : ∀ (i j : ℕ) (h1: 0 ≤ i < 7) (h2: 0 ≤ j < 7), has_integer_perimeter (L / 7) (W / 7)) 
  : original_rectangle_perimeter_integer L W h := 
sorry

end original_rectangle_perimeter_proof_l462_462419


namespace hyperbola_standard_equation_l462_462873

noncomputable def parabola_focus (k : ℝ) : (ℝ × ℝ) :=
(0, 1 / (4 * k))

theorem hyperbola_standard_equation :
  ∃ a b : ℝ,
    a > 0 ∧ b > 0 ∧
    (∀ (x y : ℝ), ((y^2 / a^2) - (x^2 / b^2) = 1) ↔ (
      let F := (0, 5)
      let E := (0, -5)
      (|((x - 0)^2 + (y - 5)^2)^0.5 - ((x - 0)^2 + (y + 5)^2)^0.5| = 6))) :=
begin
  use [3, 4],
  split,
  { norm_num },
  split,
  { norm_num },
  { sorry } -- This is where the proof would go
end

end hyperbola_standard_equation_l462_462873


namespace largest_odd_number_hundreds_digit_l462_462691

theorem largest_odd_number_hundreds_digit :
  ∃ n : ℕ, (∃ (digits : ℕ × ℕ × ℕ × ℕ × ℕ),
    let ⟨a, b, c, d, e⟩ := digits in
    (n = a * 10000 + b * 1000 + c * 100 + d * 10 + e) ∧
    (a ∈ {1, 2, 3, 5, 8}) ∧
    (b ∈ {1, 2, 3, 5, 8}) ∧
    (c ∈ {1, 2, 3, 5, 8}) ∧
    (d ∈ {1, 2, 3, 5, 8}) ∧
    (e ∈ {1, 2, 3, 5, 8}) ∧
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
     c ≠ d ∧ c ≠ e ∧
     d ≠ e) ∧
    (e % 2 = 1)) ∧
  (∃ h, h = (n / 100) % 10 ∧ h = 2) :=
by sorry

end largest_odd_number_hundreds_digit_l462_462691


namespace log_equations_l462_462558

variable (x : Real)

theorem log_equations (h : x * Real.log 4 / Real.log 3 = 1) : 
  x = Real.log 3 / Real.log 4 ∧ 4 ^ x + 4 ^ -x = 10 / 3 := 
by
  sorry

end log_equations_l462_462558


namespace Q_over_Q_neg_eq_1_l462_462769

theorem Q_over_Q_neg_eq_1 {g : Polynomial ℝ} {Q : Polynomial ℝ} 
  (h_g : g = X^2009 + 19 * X^2008 + 1)
  (h_roots : ∀ z, IsRoot g z → ∃ s, z = s ∧ IsRoot (C s * (X - s)) s)
  (h_Q : ∀ j ∈ (Finset.range 2009).image (λ i, s_i + 1 / s_i), Q j = 0) :
  Q 1 / Q (-1) = 1 :=
sorry

end Q_over_Q_neg_eq_1_l462_462769


namespace lattice_points_count_l462_462883

theorem lattice_points_count :
  (∃ S : Finset (ℤ × ℤ), S.card = 8 ∧
    ∀ (p : ℤ × ℤ),
      p ∈ S ↔
        let a := p.1
        let b := p.2 in
        a^2 + b^2 < 25 ∧
        a^2 + b^2 < 10 * a ∧
        a^2 + b^2 < 10 * b) :=
by sorry

end lattice_points_count_l462_462883


namespace find_x_l462_462012

theorem find_x (x : ℝ) (h1 : sqrt ((x - 3) ^ 2 + (7 - -2) ^ 2) = 12) (h2 : x < 0) : 
  x = 3 - sqrt 63 :=
by
  sorry

end find_x_l462_462012


namespace order_relation_l462_462225

theorem order_relation (a b c : ℝ) 
  (h1 : a = Real.log 2)
  (h2 : b = Real.cos 2)
  (h3 : c = 2^0.2) : b < a ∧ a < c :=
by
  sorry

end order_relation_l462_462225


namespace equilateral_triangle_midpoints_l462_462646

variables {A B C A' B' C' P Q R : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space A'] 
[metric_space B'] [metric_space C'] [metric_space P] [metric_space Q] [metric_space R]
variables (A B C A' B' C' P Q R : ℝ × ℝ)

noncomputable def rotate60 (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
let (x, y) := (p.1 - center.1, p.2 - center.2) in
(center.1 + x * (1 / 2) - y * (√3 / 2), center.2 + x * (√3 / 2) + y * (1 / 2))

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def is_equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
dist A B = dist B C ∧ dist B C = dist C A ∧ ∡ A B C = 60 ∧ ∡ B C A = 60

theorem equilateral_triangle_midpoints :
  rotate60 C = C' →
  A' = rotate60 A C →
  B' = rotate60 B C →
  P = midpoint A' B →
  Q = midpoint B' C →
  R = midpoint C' A →
  is_equilateral_triangle P Q R :=
begin
  sorry
end

end equilateral_triangle_midpoints_l462_462646


namespace equilateral_triangle_perimeter_l462_462576

theorem equilateral_triangle_perimeter (A B C M : Point) (r : ℝ) 
  (h_equilateral : equilateral_triangle A B C)
  (h_midpoint : midpoint M B C)
  (h_circumcircle_area : pi * r^2 = 36 * pi) :
  perimeter A B C = 36 :=
by
  sorry

end equilateral_triangle_perimeter_l462_462576


namespace color_of_2004th_light_l462_462783

def color_sequence := ["G", "Y", "Y", "R", "B", "R", "R"]

def nth_color (n : ℕ) : String :=
  color_sequence[(n % 7)]

theorem color_of_2004th_light : nth_color 2004 = "Y" := by
  -- Placeholder for the actual proof
  sorry

end color_of_2004th_light_l462_462783


namespace find_equation_of_ellipse_l462_462118

noncomputable def equation_of_ellipse (a b c : ℝ) :=
  ∃ (x y : ℝ), let lhs := (x^2 / a^2) + (y^2 / b^2) in lhs = 1

theorem find_equation_of_ellipse :
  ∃ (a b : ℝ) (x y : ℝ),
  (x^2 / a^2) + (y^2 / b^2) = 1 ∧
  (a > b) ∧
  (a^2 = b^2 + 4) ∧
  (a / b = 2 / (sqrt 3)) ∧ 
  equation_of_ellipse (4:ℝ) (2*√3) 2 :=
by
  existsi (4 : ℝ)
  existsi (2 * (√3) : ℝ)
  existsi (0 : ℝ)
  existsi (0 : ℝ)
  split
    by
      calc (0^2 / 4^2) + (0^2 / (2 * √3)^2) = 0
      ... = 1
  sorry
  sorry
  sorry
  sorry

end find_equation_of_ellipse_l462_462118


namespace recurring_decimal_product_l462_462050

theorem recurring_decimal_product :
  (let x := (8 / 99) in let y := (4 / 11) in x * y = 32 / 1089) :=
by
  let x : ℚ := 8 / 99
  let y : ℚ := 4 / 11
  show x * y = 32 / 1089
  sorry

end recurring_decimal_product_l462_462050


namespace incorrect_directions_of_opening_l462_462382

-- Define the functions
def f (x : ℝ) : ℝ := 2 * (x - 3)^2
def g (x : ℝ) : ℝ := -2 * (x - 3)^2

-- The theorem (statement) to prove
theorem incorrect_directions_of_opening :
  ¬(∀ x, (f x > 0 ∧ g x > 0) ∨ (f x < 0 ∧ g x < 0)) :=
sorry

end incorrect_directions_of_opening_l462_462382


namespace infinite_coprime_pairs_with_cubic_root_property_l462_462260

theorem infinite_coprime_pairs_with_cubic_root_property :
  ∃ᶠ (m n : ℤ) in at_top, gcd m n = 1 ∧
  ∃ (x1 x2 x3 : ℤ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (x1 + m)^3 = n * x1 ∧ (x2 + m)^3 = n * x2 ∧ (x3 + m)^3 = n * x3 := 
sorry

end infinite_coprime_pairs_with_cubic_root_property_l462_462260


namespace integer_solutions_l462_462787

theorem integer_solutions (x y : ℤ) (h : y^2 = x^3 + 2 * x^2 + 2 * x + 1) :
  (x = -1 ∧ y = 0) ∨ (x = 0 ∧ y = -1) ∨ (x = 0 ∧ y = 1) :=
begin
  sorry
end

end integer_solutions_l462_462787


namespace intersect_at_one_point_l462_462450

-- Define the equations as given in the conditions
def equation1 (b : ℝ) (x : ℝ) : ℝ := b * x ^ 2 + 2 * x + 2
def equation2 (x : ℝ) : ℝ := -2 * x - 2

-- Statement of the theorem
theorem intersect_at_one_point (b : ℝ) :
  (∀ x : ℝ, equation1 b x = equation2 x → x = 1) ↔ b = 1 := sorry

end intersect_at_one_point_l462_462450


namespace measure_angledb_l462_462918

-- Definitions and conditions:
def right_triangle (A B C : Type) : Prop :=
  ∃ (a b c : A), ∠A = 45 ∧ ∠B = 45 ∧ ∠C = 90

def angle_bisector (A B : Type) (D : Type) : Prop :=
  ∃ (AD BD : A), AD bisects ∠A ∧ BD bisects ∠B

-- Prove:
theorem measure_angledb (A B C D : Type) 
  (hABC : right_triangle A B C)
  (hAngleBisectors : angle_bisector A B D) : 
  ∠ ADB = 135 := 
sorry

end measure_angledb_l462_462918


namespace right_triangle_probability_bd_greater_than_10_l462_462303

theorem right_triangle_probability_bd_greater_than_10:
  ∀ (A B C P D : Type)
  (angle_ACB : ACB = 90)
  (angle_ABC : ABC = 45)
  (length_AB : AB = 20)
  (within_triangle : P ∈ triangle A B C)
  (D_on_AC : line BP meets AC at D), 
  (∃ (length_BD: BD > 10), probability = 1) :=
by
  sorry

end right_triangle_probability_bd_greater_than_10_l462_462303


namespace distance_Z1_Z2_l462_462512

def Z1_coord : ℝ × ℝ := (1, -1)
def Z2_coord : ℝ × ℝ := (3, -5)

theorem distance_Z1_Z2 : dist (Z1_coord) (Z2_coord) = 2 * Real.sqrt 5 :=
by
  rw [dist_eq, Real.sqrt_eq_rpow]
  repeat { rw ← abs_eq_mul_self (of_real_nonneg.mpr abs_nonneg) }
  sorry

end distance_Z1_Z2_l462_462512


namespace meeting_probability_proof_l462_462414

noncomputable def successful_meeting_probability : Prop :=
  let cube_volume := 8
  let desired_volume := (1 / 3) * ∫(0..2), (2 - z)^2 * z d z
  let probability := desired_volume / cube_volume
  probability = 1 / 9

theorem meeting_probability_proof : successful_meeting_probability :=
  sorry

end meeting_probability_proof_l462_462414


namespace probability_point_closer_to_six_l462_462018

theorem probability_point_closer_to_six :
  let segment := (0, 8)
  let midpoint := (0 + 6) / 2
  let closer_interval := (3, 8)
  let probability := (8 - 3) / (8 - 0) in
  (Real.floor (probability * 10)) / 10 = 0.6 :=
by
  let segment := (0, 8)
  let midpoint := (0 + 6) / 2
  let closer_interval := (3, 8)
  let probability := (8 - 3) / (8 - 0)
  show (Real.floor (probability * 10)) / 10 = 0.6
  sorry -- proof to be completed

end probability_point_closer_to_six_l462_462018


namespace properties_of_f_l462_462861

def f (x : ℝ) : ℝ := cos x * sin (2 * x)

theorem properties_of_f :
  (∀ x, f (-x) = -f x) ∧
  (∀ x, f (x + 2 * π) = f x) ∧
  (∀ x, f (π - x) = f x) ∧
  (∃ x, ∀ y, f y ≤ f x) ∧
  (∀ x ∈ Icc (-π/6) (π/6), ∃ δ > 0, ∀ h, abs h < δ → f (x + h) ≥ f x) :=
by
  sorry

end properties_of_f_l462_462861


namespace root_of_inverse_f_plus_x_eq_k_l462_462525

variable {α : Type*} [Nonempty α] [Field α]
variable (f : α → α)
variable (f_inv : α → α)
variable (k : α)

def root_of_f_plus_x_eq_k (x : α) : Prop :=
  f x + x = k

def inverse_function (f : α → α) (f_inv : α → α) : Prop :=
  ∀ y : α, f (f_inv y) = y ∧ f_inv (f y) = y

theorem root_of_inverse_f_plus_x_eq_k
  (h1 : root_of_f_plus_x_eq_k f 5 k)
  (h2 : inverse_function f f_inv) :
  f_inv (k - 5) + (k - 5) = k :=
by
  sorry

end root_of_inverse_f_plus_x_eq_k_l462_462525


namespace sum_of_terms_l462_462288

def sequence_sum (n : ℕ) : ℕ :=
  n^2 + 2*n + 5

theorem sum_of_terms : sequence_sum 9 - sequence_sum 6 = 51 :=
by
  sorry

end sum_of_terms_l462_462288


namespace tan_alpha_minus_pi_over_4_eq_negative_seven_l462_462130

open Real

theorem tan_alpha_minus_pi_over_4_eq_negative_seven (α : ℝ) (h1 : α ∈ Ioo (-π) 0) (h2 : cos α = -4/5) : 
  tan (α - π/4) = -7 := sorry

end tan_alpha_minus_pi_over_4_eq_negative_seven_l462_462130


namespace no_pos_ints_permutations_exceed_n_l462_462476

theorem no_pos_ints_permutations_exceed_n (b n : ℕ) (hb : b > 1) (hn : n > 1) :
  ¬ (∃ m, (0 < m) ∧ (m > n) ∧ (m = (n.digitPermutations b).length)) :=
by
  sorry

end no_pos_ints_permutations_exceed_n_l462_462476


namespace no_confidence_expected_value_range_of_p_l462_462302

-- Definitions for conditions
def requiredQuestions_old := 3
def buzzerQuestions_old := 3
def requiredQuestions_new := 4
def buzzerQuestions_new := 2

def score_required_correct := 1
def score_required_wrong := 0
def score_buzzer_correct := 2
def score_buzzer_wrong := -1

def contingency_table := 
  {a := 6, b := 9, c := 4, d := 1, n := 20}

def chi_squared (a b c d n : ℕ) : ℝ :=
  (n * ((a * d - b * c)^2)) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Given values for calculations
noncomputable def K_sq : ℝ := chi_squared contingency_table.a contingency_table.b contingency_table.c contingency_table.d contingency_table.n

-- Vaule of 0.05 significance level
def critical_value := 3.841

-- Theorem stating K_sq < critical_value implies no 95% confidence
theorem no_confidence : K_sq < critical_value := by sorry

-- Student C's probabilities and calculations
def p_buzzer : ℝ := 0.6
def p_correct_buzzer : ℝ := 0.8

def P (X : ℝ) : ℝ :=
  if X = -1 then p_buzzer * (1 - p_correct_buzzer)
  else if X = 0 then 1 - p_buzzer
  else if X = 2 then p_buzzer * p_correct_buzzer
  else 0

def E (X : ℝ) : ℝ := 
  (-1) * P(-1) + 0 * P(0) + 2 * P(2)

-- The expected value 
theorem expected_value : E(∞) = 0.84 := by sorry

-- Range of p for expected total scores condition
def old_format_score (p : ℝ) : ℝ := 3 * p + 3 * E(∞)
def new_format_score (p : ℝ) : ℝ := 4 * p + 2 * E(∞)
def expected_score_diff (p : ℝ) : ℝ := abs (old_format_score p - new_format_score p)

theorem range_of_p (p : ℝ) : expected_score_diff p ≤ 0.1 ↔ 0.74 ≤ p ∧ p ≤ 0.94 := by sorry

end no_confidence_expected_value_range_of_p_l462_462302


namespace greatest_divisor_arithmetic_sum_l462_462365

theorem greatest_divisor_arithmetic_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ d, d = 15 ∧ ∀ S : ℕ, S = 15 * x + 105 * c → d ∣ S :=
by 
  sorry

end greatest_divisor_arithmetic_sum_l462_462365


namespace line_intersects_midpoint_l462_462669

theorem line_intersects_midpoint (c : ℝ) :
  let x1 := 1
  let y1 := 4
  let x2 := 3
  let y2 := 8
  let xm := (x1 + x2) / 2
  let ym := (y1 + y2) / 2
  (xm + ym = c) →
  c = 8 :=
by
  let x1 := 1
  let y1 := 4
  let x2 := 3
  let y2 := 8
  let xm := (x1 + x2) / 2
  let ym := (y1 + y2) / 2
  have midpoint_coords : xm = 2 ∧ ym = 6 :=
    by simp [xm, ym]
  intro h
  have c_val : c = 8 :=
    by
      rw [xm, ym] at h
      simp at h
      exact h
  exact c_val

end line_intersects_midpoint_l462_462669


namespace series_sum_108_l462_462755

theorem series_sum_108 : 
  (∑ n in Finset.range 9, (1/(n + 1) - 1/(n + 3))/(1/(n + 1) * 1/(n + 2) * 1/(n + 3))) = 108 := 
sorry

end series_sum_108_l462_462755


namespace greatest_common_divisor_sum_arithmetic_sequence_l462_462347

theorem greatest_common_divisor_sum_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, d = 15 ∧ ∀ (n : ℕ), n = 15 → ∀ k : ℕ, k = 15 ∧ 15 ∣ (15 * (x + 7 * c)) :=
by
  sorry

end greatest_common_divisor_sum_arithmetic_sequence_l462_462347


namespace simplify_expr_C_l462_462973

theorem simplify_expr_C (x y : ℝ) : 5 * x - (x - 2 * y) = 4 * x + 2 * y :=
by
  sorry

end simplify_expr_C_l462_462973


namespace monotonicity_of_f_g_greater_than_f_plus_kx_minus_1_l462_462538

noncomputable def f (x k : ℝ) : ℝ := log x - k * x + 1
noncomputable def g (x a : ℝ) : ℝ := exp x / (a * x)

theorem monotonicity_of_f (k : ℝ) (x : ℝ) (h₁ : 0 < x) : 
  (k ≤ 0 ∧ (∀ x, differentiable_at ℝ (λ x, f x k) x)) ∨ ((k > 0) ∧ (∀ x, differentiable_at ℝ (λ x, f x k) x)) := 
sorry

theorem g_greater_than_f_plus_kx_minus_1 (a : ℝ) (h₁ : 0 < a) (h₂ : a ≤ exp 2 / 2) (x : ℝ) (h₃ : 0 < x) (k : ℝ) : 
  g x a > f x k + k * x - 1 :=
sorry

end monotonicity_of_f_g_greater_than_f_plus_kx_minus_1_l462_462538


namespace smallest_positive_multiple_smallest_positive_multiple_is_30_l462_462804

theorem smallest_positive_multiple (b : ℕ) (hb1 : b % 6 = 0) (hb2 : b % 15 = 0) : b ≥ 30 :=
by sorry

theorem smallest_positive_multiple_is_30 : ∃ b, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 ∧ ∀ b', b' > 0 ∧ b' % 6 = 0 ∧ b' % 15 = 0 → b' ≥ b :=
by {
  use 30,
  split,
  { exact 30,
  { split,
  { exact 30 % 6 = 0,
  { split,
  { exact 30 % 15 = 0,
  { intros b' hb'0 hb'1 hb'2,
    exact b' ≥ 30, sorry }}}
}

end smallest_positive_multiple_smallest_positive_multiple_is_30_l462_462804


namespace f_at_2_is_neg_1_l462_462536

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^5 + b * x^3 - x + 2

-- Given condition: f(-2) = 5
axiom h : ∀ (a b : ℝ), f a b (-2) = 5

-- Prove that f(2) = -1 given the above conditions
theorem f_at_2_is_neg_1 (a b : ℝ) (h_ab : f a b (-2) = 5) : f a b 2 = -1 := by
  sorry

end f_at_2_is_neg_1_l462_462536


namespace measure_of_obtuse_angle_ADB_l462_462906

-- Define the right triangle ABC with specific angles
def triangle_ABC (A B C D : Type) [AddAngle A B C] :=
  (rightTriangle : (A B C) ∧ 
   angle_A_is_45_degrees : (angle A = 45) ∧ 
   angle_B_is_45_degrees : (angle B = 45) ∧ 
   AD_bisects_A : (bisects A D) ∧ 
   BD_bisects_B : (bisects B D))

-- Statement of the proof problem
theorem measure_of_obtuse_angle_ADB {A B C D : Type} [AddAngle A B C] 
  (h_ABC : triangle_ABC A B C) : 
  measure_obtuse_angle A B D ADB = 135 :=
sorry

end measure_of_obtuse_angle_ADB_l462_462906


namespace chord_line_eq_parabola_midpoint_l462_462542

theorem chord_line_eq_parabola_midpoint :
  (∀ {x y : ℝ}, x^2 = -2 * y → ∃ m b : ℝ, y = m * x + b) →
  ∀ {x1 x2 y1 y2 : ℝ}, (x1 + x2 = -2) →
  (x1^2 = -2 * y1) →
  (x2^2 = -2 * y2) →
  let midpoint : ℝ × ℝ := (-1, -5) in
  midpoint = ((x1 + x2) / 2, (y1 + y2) / 2) →
  (∃ k : ℝ, k = 1) →
  (∃ b : ℝ, b = -4) →
  y = x - 4 :=
by
  intro h chord_cond x1 x2 y1 y2 hx1x2 hx1y1 hx2y2 hmidpoint hk hb
  sorry

end chord_line_eq_parabola_midpoint_l462_462542


namespace solve_x_in_equation_l462_462378

theorem solve_x_in_equation (x : ℕ) (h : x + (x + 1) + (x + 2) + (x + 3) = 18) : x = 3 :=
by
  sorry

end solve_x_in_equation_l462_462378


namespace cubed_identity_l462_462171

theorem cubed_identity (x : ℝ) (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := 
sorry

end cubed_identity_l462_462171


namespace f_sum_always_positive_l462_462071

-- Define the function on ℝ
variable (f : ℝ → ℝ)

-- Conditions
def is_increasing_on_neg_infty_2 (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → x < 2 → y < 2 → f(x) < f(y)

def is_odd_function (f : ℝ → ℝ)  : Prop :=
  ∀ x, f(x + 2) = -f(-(x + 2))

-- Given x1 < 2, x2 > 2, and |x1 - 2| < |x2 - 2|
def in_domain (x1 x2 : ℝ) : Prop :=
  x1 < 2 ∧ x2 > 2 ∧ abs(x1 - 2) < abs(x2 - 2)

-- Proof problem statement
theorem f_sum_always_positive (f : ℝ → ℝ) (x1 x2 : ℝ) (h1: is_increasing_on_neg_infty_2 f) (h2: is_odd_function f) (h3: in_domain x1 x2) :
    f(x1) + f(x2) > 0 := sorry

end f_sum_always_positive_l462_462071


namespace sum_of_values_l462_462231

noncomputable def f (x : ℝ) : ℝ :=
if x < 3 then 5 * x + 20 else 3 * x - 21

theorem sum_of_values (h₁ : ∃ x, x < 3 ∧ f x = 4) (h₂ : ∃ x, x ≥ 3 ∧ f x = 4) :
  ∃a b : ℝ, a = -16 / 5 ∧ b = 25 / 3 ∧ (a + b = 77 / 15) :=
by {
  sorry
}

end sum_of_values_l462_462231


namespace greatest_divisor_of_sum_of_arithmetic_sequence_l462_462355

theorem greatest_divisor_of_sum_of_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ k : ℕ, (sum (λ n, x + n * c) (range 15)) = 15 * k :=
by sorry

end greatest_divisor_of_sum_of_arithmetic_sequence_l462_462355


namespace probability_double_equals_one_fourth_l462_462464

-- Define the standard set of dominos as a list of pairs (a, b) where 0 ≤ a ≤ b ≤ 6
def standardSetOfDominos : List (ℕ × ℕ) :=
  [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6),
   (1,1), (1,2), (1,3), (1,4), (1,5), (1,6),
   (2,2), (2,3), (2,4), (2,5), (2,6),
   (3,3), (3,4), (3,5), (3,6),
   (4,4), (4,5), (4,6),
   (5,5), (5,6),
   (6,6)]

-- Define what a double domino is
def isDouble (d : ℕ × ℕ) : Prop := d.1 = d.2

-- Count the number of double dominos
def numberOfDoubles : ℕ := standardSetOfDominos.countp isDouble

-- Count the total number of dominos
def totalNumberOfDominos : ℕ := standardSetOfDominos.length

-- Define the probability of selecting a double
def probabilityOfDouble : ℚ := numberOfDoubles / totalNumberOfDominos

-- Statement of the problem: Prove that the probability of a double equals 1/4
theorem probability_double_equals_one_fourth : probabilityOfDouble = 1 / 4 := by
  sorry

end probability_double_equals_one_fourth_l462_462464


namespace symmetric_min_value_l462_462528

noncomputable def circle_center := (-1, 2)
def is_symmetric (a b : ℝ) (p : ℝ × ℝ) : Prop :=
  ∃ q : ℝ × ℝ, 2 * a * p.1 - b * p.2 + 2 = 0 ∧ q.1 = p.1 ∧ q.2 = p.2

theorem symmetric_min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 1) :
  ∀ p : ℝ × ℝ, is_symmetric a b p → ∀ x y : ℝ, (x^2 + y^2 + 2 * x - 4 * y + 1 = 0) → 
  (x - y = -1) → (point_on_circle center) → 
  (p = q) → (q = p) → 
  (p ∈ circle ∧ q ∈ circle) → 
  (b = 
      3 + 2 * real.sqrt(2)) :=
begin
  sorry
end

end symmetric_min_value_l462_462528


namespace plates_added_before_topple_l462_462032

theorem plates_added_before_topple (init_plates add_first add_total : ℕ) (h : init_plates = 27) (h1 : add_first = 37) (h2 : add_total = 83) : 
  add_total - (init_plates + add_first) = 19 :=
by
  -- proof goes here
  sorry

end plates_added_before_topple_l462_462032


namespace sum_mod_1000_l462_462025

def a : ℕ → ℤ
| 0     := 2
| 1     := 2
| 2     := 2
| (n+3) := a n + a (n+1) + a (n+2)

theorem sum_mod_1000 :
  a 28 = 4091570 →
  a 29 = 22403642 →
  a 30 = 41206722 →
  (∑ k in Finset.range 29, a k) % 1000 = 189 :=
by
  intros h28 h29 h30
  sorry

end sum_mod_1000_l462_462025


namespace distance_inequality_l462_462955

variables {A B C D E : Type}
variable [metric_space E] -- ensuring a metric space context to define distances

theorem distance_inequality
  (on_line : A ≠ B → C ≠ D → (∀ x : E, (x = A ∨ x = B ∨ x = C ∨ x = D) → x ≠ E))
  (dist_AB_CD : dist A B = dist C D) :
  dist E A + dist E D > dist E B + dist E C :=
sorry

end distance_inequality_l462_462955


namespace sale_in_fifth_month_l462_462009

def sales_month_1 := 6635
def sales_month_2 := 6927
def sales_month_3 := 6855
def sales_month_4 := 7230
def sales_month_6 := 4791
def target_average := 6500
def number_of_months := 6

def total_sales := sales_month_1 + sales_month_2 + sales_month_3 + sales_month_4 + sales_month_6

theorem sale_in_fifth_month :
  (target_average * number_of_months) - total_sales = 6562 :=
by
  sorry

end sale_in_fifth_month_l462_462009


namespace determine_f_values_l462_462720

noncomputable def f : ℤ → ℤ := sorry

theorem determine_f_values :
  (f 0 = 0 ∧ f 1 = 1) ∨ (f 0 = 1 ∧ f 1 = 1) :=
begin  
  -- Given that the conditions are satisfied
  have h1 : ∀ x : ℤ, f (x + 5) - f x = 10 * x + 25, sorry,
  have h2 : ∀ x : ℤ, f (x^2) = (f x - x)^2 + x^2, sorry,
  
  -- Prove the resulting ordered pairs
  sorry
end

end determine_f_values_l462_462720


namespace greatest_divisor_of_sum_of_arith_seq_l462_462311

theorem greatest_divisor_of_sum_of_arith_seq (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → d ∣ (15 * (x + 7 * c))) ∧
    (∀ k : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → k ∣ (15 * (x + 7 * c))) → k ≤ d) ∧ 
    d = 15 :=
sorry

end greatest_divisor_of_sum_of_arith_seq_l462_462311


namespace min_value_PF_plus_PA_exists_l462_462895

structure Point where
  x : ℝ
  y : ℝ

def hyperbola (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 12) = 1

def distance (P Q : Point) : ℝ := real.sqrt((P.x - Q.x)^2 + (P.y - Q.y)^2)

def focus_left : Point := ⟨-2, 0⟩
def focus_right : Point := ⟨2, 0⟩
def A : Point := ⟨1, 4⟩

theorem min_value_PF_plus_PA_exists :
  ∃ P : Point, hyperbola P.x P.y ∧
    (∀ Q : Point, hyperbola Q.x Q.y → distance Q focus_left + distance Q A ≥ distance P focus_left + distance P A) ∧
    (distance P focus_left + distance P A = 9) := by
  sorry

end min_value_PF_plus_PA_exists_l462_462895


namespace smallest_C_inequality_l462_462802

theorem smallest_C_inequality :
  ∃ C ≥ 0, (∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z)) ∧ C = Real.sqrt (4 / 3) :=
begin
  use Real.sqrt (4 / 3),
  split,
  { exact Real.sqrt_nonneg (4 / 3), },
  { intros x y z,
    sorry,  -- Proof to be completed
  },
end

end smallest_C_inequality_l462_462802


namespace greatest_divisor_of_sum_of_arithmetic_sequence_l462_462354

theorem greatest_divisor_of_sum_of_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ k : ℕ, (sum (λ n, x + n * c) (range 15)) = 15 * k :=
by sorry

end greatest_divisor_of_sum_of_arithmetic_sequence_l462_462354


namespace smallest_crate_side_l462_462407

/-- 
A crate measures some feet by 8 feet by 12 feet on the inside. 
A stone pillar in the shape of a right circular cylinder must fit into the crate for shipping so that 
it rests upright when the crate sits on at least one of its six sides. 
The radius of the pillar is 7 feet. 
Prove that the length of the crate's smallest side is 8 feet.
-/
theorem smallest_crate_side (x : ℕ) (hx : x >= 14) : min (min x 8) 12 = 8 :=
by {
  sorry
}

end smallest_crate_side_l462_462407


namespace candidate_majority_votes_l462_462573

theorem candidate_majority_votes (total_votes : ℕ) (candidate_percentage other_percentage : ℕ) 
  (h_total_votes : total_votes = 5200)
  (h_candidate_percentage : candidate_percentage = 60)
  (h_other_percentage : other_percentage = 40) :
  (candidate_percentage * total_votes / 100) - (other_percentage * total_votes / 100) = 1040 := 
by
  sorry

end candidate_majority_votes_l462_462573


namespace fourth_equation_general_expression_l462_462239

theorem fourth_equation :
  (10 : ℕ)^2 - 4 * (4 : ℕ)^2 = 36 := 
sorry

theorem general_expression (n : ℕ) (hn : n > 0) :
  (2 * n + 2)^2 - 4 * n^2 = 8 * n + 4 :=
sorry

end fourth_equation_general_expression_l462_462239


namespace selling_price_correct_l462_462744

-- Conditions given in the problem
def original_price : ℝ := 120
def initial_discount_rate : ℝ := 0.30
def additional_discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.12

-- Intermediate calculations
def initial_sale_price : ℝ := original_price * (1 - initial_discount_rate)
def additional_discount : ℝ := if initial_sale_price > 80 then initial_sale_price * additional_discount_rate else 0
def final_sale_price : ℝ := initial_sale_price - additional_discount
def tax : ℝ := final_sale_price * tax_rate

-- The total selling price, including tax
def total_selling_price : ℝ := final_sale_price + tax

-- Proof that the total selling price is 84.672 dollars
theorem selling_price_correct : total_selling_price = 84.672 := by
  sorry

end selling_price_correct_l462_462744


namespace problem1_problem2_l462_462864

-- Definition of the function f(x)
def f (x a : ℝ) : ℝ := |2 * x - a| + |2 * x - 1|

-- 1st problem: Prove the solution set for f(x) ≤ 2 when a = -1 is { x | x = ± 1/2 }
theorem problem1 : (∀ x : ℝ, f x (-1) ≤ 2 ↔ x = 1/2 ∨ x = -1/2) :=
by sorry

-- 2nd problem: Prove the range of real number a is [0, 3]
theorem problem2 : (∃ a : ℝ, (∀ x ∈ Set.Icc (1/2:ℝ) 1, f x a ≤ |2 * x + 1| ) ↔ 0 ≤ a ∧ a ≤ 3) :=
by sorry

end problem1_problem2_l462_462864


namespace constant_term_in_expansion_l462_462561

theorem constant_term_in_expansion : 
  let n := (5 + 7) / 2
  in n = 6 → 
  let binom_term := (1 / x - 2 * x) ^ n 
  in (∃ c : ℚ, constant_term binom_term = -160) :=
by 
  sorry

end constant_term_in_expansion_l462_462561


namespace compute_C_l462_462794

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![[2, 1], [3, 4]]

def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![[0, -5], [-1, 6]]

theorem compute_C : 
  (fun C => C = 2 • A + B) 
    ![[4, -3], [5, 14]] := by
  sorry

end compute_C_l462_462794


namespace number_of_good_carrots_l462_462238

def total_carrots (nancy_picked : ℕ) (mom_picked : ℕ) : ℕ :=
  nancy_picked + mom_picked

def bad_carrots := 14

def good_carrots (total : ℕ) (bad : ℕ) : ℕ :=
  total - bad

theorem number_of_good_carrots :
  good_carrots (total_carrots 38 47) bad_carrots = 71 := by
  sorry

end number_of_good_carrots_l462_462238


namespace no_sphere_inscribed_l462_462571

theorem no_sphere_inscribed
  (P : Type) [polyhedron P] 
  (black_faces : set (face P))
  (white_faces : set (face P))
  (num_black_faces : ℕ)
  (num_white_faces : ℕ)
  (no_two_black_faces_share_edge : ∀ f1 f2: face P, f1 ∈ black_faces → f2 ∈ black_faces → ¬ (share_edge f1 f2))
  (more_black_than_half : 2 * card black_faces > card (faces P))
  (sum_black_greater_than_sum_white : sum_area black_faces > sum_area white_faces) :
  ¬ (sphere_inscribed P) := 
by 
  sorry

end no_sphere_inscribed_l462_462571


namespace solve_z_cubed_eq_neg_one_l462_462097

-- The statement of the problem in Lean 4:
theorem solve_z_cubed_eq_neg_one (z : ℂ) :
  (z ^ 3 = -1) ↔ (z = -1 ∨ z = (1 + complex.I * real.sqrt 3) / 2 ∨ z = (1 - complex.I * real.sqrt 3) / 2) :=
by
  sorry

end solve_z_cubed_eq_neg_one_l462_462097


namespace greatest_integer_l462_462256

theorem greatest_integer (m : ℕ) (h1 : 0 < m) (h2 : m < 150)
  (h3 : ∃ a : ℤ, m = 9 * a - 2) (h4 : ∃ b : ℤ, m = 5 * b + 4) :
  m = 124 := 
sorry

end greatest_integer_l462_462256


namespace length_platform_equals_length_train_l462_462992

def speed_km_per_hr : ℝ := 126
def time_seconds : ℝ := 60
def length_train : ℝ := 1050

def speed_m_per_s : ℝ := speed_km_per_hr * 1000 / 3600
def total_distance : ℝ := speed_m_per_s * time_seconds
def length_platform : ℝ := total_distance - length_train

theorem length_platform_equals_length_train : length_platform = length_train := by
  sorry

end length_platform_equals_length_train_l462_462992


namespace prime_divisors_ge_n_add_k_l462_462649

theorem prime_divisors_ge_n_add_k
  (n k : ℕ) 
  (p : ℕ → ℕ) 
  (h : ∀ i, 1 ≤ i → i ≤ n → prime (p i ∧ ¬ even (p i))) 
  (unique_odd_primes : (∀ i j, 1 ≤ i → i ≤ n → 1 ≤ j → j ≤ n → p i = p j → i = j)) 
  (num_odd_primes : ∀ i : ℕ, odd i -> ∃ j, p i = j ∧ j, ith_odd_prime) 
   :
  ∃ N, N =  ( ∏ i in (range n), (p i) + 1 ) ^ (2^(k)) - 1 ∧  
  (∃ A, finite A ∧ ∃ u : ∣ (, A → prime u) ∧  prime small_primes:

  sorry
  
end prime_divisors_ge_n_add_k_l462_462649


namespace smallest_five_sequential_number_greater_than_2000_is_2004_l462_462099

def fiveSequentialNumber (N : ℕ) : Prop :=
  (if 1 ∣ N then 1 else 0) + 
  (if 2 ∣ N then 1 else 0) + 
  (if 3 ∣ N then 1 else 0) + 
  (if 4 ∣ N then 1 else 0) + 
  (if 5 ∣ N then 1 else 0) + 
  (if 6 ∣ N then 1 else 0) + 
  (if 7 ∣ N then 1 else 0) + 
  (if 8 ∣ N then 1 else 0) + 
  (if 9 ∣ N then 1 else 0) ≥ 5

theorem smallest_five_sequential_number_greater_than_2000_is_2004 :
  ∀ N > 2000, fiveSequentialNumber N → N = 2004 :=
by
  intros N hn hfsn
  have hN : N = 2004 := sorry
  exact hN

end smallest_five_sequential_number_greater_than_2000_is_2004_l462_462099


namespace prob_six_digit_palindrome_div_11_l462_462433

theorem prob_six_digit_palindrome_div_11 :
  let six_digit_palindrome := {n : ℕ // 100000 ≤ n ∧ n < 1000000 ∧ (∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 → n = 100001 * a + 10010 * b + 1100 * c)}
  let num_divisible_11 := {n ∈ six_digit_palindrome | n % 11 = 0}
  ∃ (num_total : ℕ) (num_favorable : ℕ), num_total = 900 ∧ num_favorable = 25 ∧ num_divisible_11.card = num_favorable ∧ (num_favorable : ℚ) / num_total = 1 / 36 :=
by
  sorry

end prob_six_digit_palindrome_div_11_l462_462433


namespace greatest_divisor_of_sum_of_arith_seq_l462_462312

theorem greatest_divisor_of_sum_of_arith_seq (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → d ∣ (15 * (x + 7 * c))) ∧
    (∀ k : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → k ∣ (15 * (x + 7 * c))) → k ≤ d) ∧ 
    d = 15 :=
sorry

end greatest_divisor_of_sum_of_arith_seq_l462_462312


namespace number_of_human_family_members_l462_462572

-- Definitions for the problem
def num_birds := 4
def num_dogs := 3
def num_cats := 18
def bird_feet := 2
def dog_feet := 4
def cat_feet := 4
def human_feet := 2
def human_heads := 1

def animal_feet := (num_birds * bird_feet) + (num_dogs * dog_feet) + (num_cats * cat_feet)
def animal_heads := num_birds + num_dogs + num_cats

def total_feet (H : Nat) := animal_feet + (H * human_feet)
def total_heads (H : Nat) := animal_heads + (H * human_heads)

-- The problem statement translated to Lean
theorem number_of_human_family_members (H : Nat) : (total_feet H) = (total_heads H) + 74 → H = 7 :=
by
  sorry

end number_of_human_family_members_l462_462572


namespace construct_quadrilateral_l462_462469

variables {AB BC CD DA AC : ℝ}
variables (h1 : AB > 0) (h2 : BC > 0) (h3 : CD > 0) (h4 : DA > 0)
variable (h5 : AC > 0)

-- Statement of the theorem for constructing the quadrilateral ABCD
theorem construct_quadrilateral
(h_bisect : ∀ (A B C D : Type) (AB BC CD DA AC : ℝ), AC ∈ bisects_angle A B C D)
: ∃ (A B C D : Type), quadrilateral A B C D ∧ side_lengths A B C D AB BC CD DA ∧ diagonal_bisects_angle A B C D AC :=
by
  sorry

end construct_quadrilateral_l462_462469


namespace no_positive_int_solutions_l462_462967

theorem no_positive_int_solutions
  (x y z t : ℕ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (ht : 0 < t)
  (h1 : x^2 + 2 * y^2 = z^2)
  (h2 : 2 * x^2 + y^2 = t^2) : false :=
by
  sorry

end no_positive_int_solutions_l462_462967


namespace value_of_fraction_l462_462856

variables {a b c : ℝ}

-- Conditions
def quadratic_has_no_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c < 0

def person_A_roots (a' b c : ℝ) : Prop :=
  b = -6 * a' ∧ c = 8 * a'

def person_B_roots (a b' c : ℝ) : Prop :=
  b' = -3 * a ∧ c = -4 * a

-- Proof Statement
theorem value_of_fraction (a b c a' b' : ℝ)
  (hnr : quadratic_has_no_real_roots a b c)
  (hA : person_A_roots a' b c)
  (hB : person_B_roots a b' c) :
  (2 * b + 3 * c) / a = 6 :=
by
  sorry

end value_of_fraction_l462_462856


namespace sin_beta_value_l462_462887

theorem sin_beta_value (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : Real.cos α = 4 / 5) 
  (h4 : Real.cos (α + β) = 5 / 13) : 
  Real.sin β = 33 / 65 := 
by 
  sorry

end sin_beta_value_l462_462887


namespace simplify_expr1_simplify_expr2_l462_462974

noncomputable def expr1 : ℝ := (2 + 7/9) ^ 0.5 + 0.1 ^ (-2) + (2 + 10/27) ^ (-2 / 3) - 3 * (Real.pi ^ 0) + 37/48

theorem simplify_expr1 : expr1 = 100 := by
  sorry

variable (a : ℝ)

noncomputable def expr2 : ℝ := 3 * a ^ (7 / 2) * sqrt (a ^ (-3)) / (cbrt (sqrt (a ^ (-3)) * sqrt (a ^ (-1))))

theorem simplify_expr2 : expr2 = a ^ (4 / 3) := by
  sorry

end simplify_expr1_simplify_expr2_l462_462974


namespace sum_of_n_and_k_l462_462987

open Nat

theorem sum_of_n_and_k (n k : ℕ) (h1 : (n.choose (k + 1)) = 3 * (n.choose k))
                      (h2 : (n.choose (k + 2)) = 2 * (n.choose (k + 1))) :
    n + k = 7 := by
  sorry

end sum_of_n_and_k_l462_462987


namespace incenter_circumcenter_circle_collinear_l462_462445

-- Definitions of the conditions
structure Triangle (α : Type) :=
(A B C : α)

variables {α : Type} [EuclideanGeometry α]

structure Circle (α : Type) :=
(center : α)
(radius : ℝ)

def incenter (T : Triangle α) : α := sorry
def circumcenter (T : Triangle α) : α := sorry

def externally_tangent (c₁ c₂ : Circle α) : Prop := sorry
def tangent_to_sides (T : Triangle α) (c : Circle α) : Prop := sorry

-- The circles K1, K2, K3, and K4 with equal radii, positioned as described
variables (T : Triangle α) (K₁ K₂ K₃ K₄ : Circle α) 
(h_rad : K₁.radius = K₂.radius ∧ K₂.radius = K₃.radius ∧ K₃.radius = K₄.radius)
(h_tangent_sides_K₁ : tangent_to_sides T K₁)
(h_tangent_sides_K₂ : tangent_to_sides T K₂)
(h_tangent_sides_K₃ : tangent_to_sides T K₃)
(h_ext_tangent_K₁_K₄ : externally_tangent K₁ K₄)
(h_ext_tangent_K₂_K₄ : externally_tangent K₂ K₄)
(h_ext_tangent_K₃_K₄ : externally_tangent K₃ K₄)

-- The incenter I, circumcenter O of triangle ABC, and the center of circle K₄ are collinear
theorem incenter_circumcenter_circle_collinear : 
  collinear [incenter T, circumcenter T, K₄.center] :=
sorry

end incenter_circumcenter_circle_collinear_l462_462445


namespace min_value_expression_l462_462797

theorem min_value_expression :
  ∀ x y : ℝ, ∃ m : ℝ, m = x^2 - x * y + 4 * y^2 ∧ m ≥ 0 :=
begin
  intros x y,
  use (x^2 - x * y + 4 * y^2),
  split,
  { refl, },
  { sorry, }  -- Proof that the minimum value is 0
end

end min_value_expression_l462_462797


namespace gcd_le_sqrt_sum_l462_462938

theorem gcd_le_sqrt_sum {a b : ℕ} (ha : 0 < a) (hb : 0 < b)
  (h : (a + 1) / b + (b + 1) / a ∈ ℤ) : Nat.gcd a b ≤ Nat.sqrt (a + b) :=
sorry

end gcd_le_sqrt_sum_l462_462938


namespace trig_identity_example_l462_462073

noncomputable def cos24 := Real.cos (24 * Real.pi / 180)
noncomputable def cos36 := Real.cos (36 * Real.pi / 180)
noncomputable def sin24 := Real.sin (24 * Real.pi / 180)
noncomputable def sin36 := Real.sin (36 * Real.pi / 180)
noncomputable def cos60 := Real.cos (60 * Real.pi / 180)

theorem trig_identity_example :
  cos24 * cos36 - sin24 * sin36 = cos60 :=
by
  sorry

end trig_identity_example_l462_462073


namespace greatest_divisor_arithmetic_sequence_sum_l462_462325

theorem greatest_divisor_arithmetic_sequence_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ k, (15 * (x + 7 * c)) = 15 * k :=
sorry

end greatest_divisor_arithmetic_sequence_sum_l462_462325


namespace geometric_sequence_identity_l462_462886

variables {b : ℕ → ℝ} {m n p : ℕ}

def is_geometric_sequence (b : ℕ → ℝ) :=
  ∀ i j k : ℕ, i < j → j < k → b j^2 = b i * b k

noncomputable def distinct_pos_ints (m n p : ℕ) :=
  0 < m ∧ 0 < n ∧ 0 < p ∧ m ≠ n ∧ n ≠ p ∧ p ≠ m

theorem geometric_sequence_identity 
  (h_geom : is_geometric_sequence b) 
  (h_distinct : distinct_pos_ints m n p) : 
  b p ^ (m - n) * b m ^ (n - p) * b n ^ (p - m) = 1 :=
sorry

end geometric_sequence_identity_l462_462886


namespace second_question_correct_l462_462178

/-- Define the problem conditions -/
variables (A B : ℝ) (n : ℝ) (both : ℝ) (neither : ℝ)

/-- Given conditions -/
def conditions :=
  A = 0.75 ∧ n = 0.20 ∧ both = 0.60

/-- The problem statement -/
theorem second_question_correct (A B n both x : ℝ) (h : conditions A B n both) :
  A + B - both = 1 - n → x = 0.65 :=
by
  intro h1
  have h2 : A + B - both = 0.80 := by { rw h1, sorry }
  sorry

end second_question_correct_l462_462178


namespace real_part_of_z_l462_462830

theorem real_part_of_z (z : ℂ) (h : z - abs z = -8 + 12 * complex.I) : z.re = 5 :=
sorry

end real_part_of_z_l462_462830


namespace circumcircle_CDF_touches_AE_l462_462960

-- Here are the conditions:
variables {A B C D E F P Q : Point}
variables (parallelogram_ABCD : parallelogram A B C D)
variables (E_on_BC : E ∈ line B C)
variables (F_on_AD : F ∈ line A D)
variables (circumcircle_ABE_touches_CF : ∃ P, P ∈ line C F ∧ touches (circumcircle A B E) P)

-- The statement to prove:
theorem circumcircle_CDF_touches_AE (parallelogram_ABCD : parallelogram A B C D) 
  (E_on_BC : E ∈ line B C) (F_on_AD : F ∈ line A D)
  (circumcircle_ABE_touches_CF : ∃ P, P ∈ line C F ∧ touches (circumcircle A B E) P) :
  ∃ Q, Q ∈ line A E ∧ touches (circumcircle C D F) Q := 
sorry

end circumcircle_CDF_touches_AE_l462_462960


namespace remainder_76_pow_77_mod_7_l462_462375

/-- Statement of the problem:
Prove that the remainder of \(76^{77}\) divided by 7 is 6.
-/
theorem remainder_76_pow_77_mod_7 :
  (76 ^ 77) % 7 = 6 := 
by
  sorry

end remainder_76_pow_77_mod_7_l462_462375


namespace axis_of_symmetry_l462_462265

theorem axis_of_symmetry (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) :
  (y = f (2x + 1)) → is_symmetrical (x = -1) :=
by
  sorry

def is_symmetrical (l : ℝ) (p : ℝ → ℝ) : Prop :=
  ∀ x, p (2 * x + 1) = p (2 * (-x) + 1)

end axis_of_symmetry_l462_462265


namespace logs_sum_eq_neg_one_l462_462535

theorem logs_sum_eq_neg_one : 
  (∑ i in (finset.range 2009), real.log 2010 (↑i / (i + 1 : ℝ))) = -1 := by
  sorry

end logs_sum_eq_neg_one_l462_462535


namespace greatest_divisor_of_arithmetic_sequence_sum_l462_462334

theorem greatest_divisor_of_arithmetic_sequence_sum :
  ∀ (x c : ℕ), ∃ k : ℕ, k = 15 ∧ 15 ∣ (15 * x + 105 * c) :=
by
  intro x c
  exists 15
  split
  . rfl
  . sorry

end greatest_divisor_of_arithmetic_sequence_sum_l462_462334


namespace sine_expression_evaluation_l462_462939

theorem sine_expression_evaluation (b : ℝ) (h : b = 2 * π / 13) :
  (sin (4 * b) * sin (8 * b) * sin (10 * b) * sin (12 * b) * sin (14 * b)) / 
  (sin b * sin (2 * b) * sin (4 * b) * sin (6 * b) * sin (10 * b)) = 
  sin (10 * π / 13) / sin (4 * π / 13) :=
by
  cases h
  sorry

end sine_expression_evaluation_l462_462939


namespace cube_of_z_l462_462226

def z : ℂ := 2 + 5 * Complex.i

theorem cube_of_z : z^3 = -142 - 65 * Complex.i :=
by
  sorry

end cube_of_z_l462_462226


namespace longest_side_length_quadrilateral_l462_462284

theorem longest_side_length_quadrilateral : 
  (∃ (x y : ℝ), x + y ≤ 4 ∧ 2 * x + y ≥ 1 ∧ x ≥ 0 ∧ y ≥ 0)
  → ∃ a b c d: ℝ, 
    (a, b), (c,d) in {(x,y) | (x + y ≤ 4 ∧ 2 * x + y ≥ 1 ∧ x ≥ 0 ∧ y ≥ 0)}
    ∧ dist (a, b) (c, d) = 7 * sqrt 2 / 2 :=
sorry

end longest_side_length_quadrilateral_l462_462284


namespace sum_mean_median_mode_eq_l462_462756

open List

def the_list : List ℕ := [1, 2, 2, 4, 5, 5, 5, 7, 8]

def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

def mode (l : List ℕ) : ℕ :=
  l.mode

def median (l : List ℕ) : ℕ :=
  l.median

theorem sum_mean_median_mode_eq :
  mean the_list + median the_list + mode the_list = 43 / 3 := by sorry

end sum_mean_median_mode_eq_l462_462756


namespace problem_1_l462_462758

theorem problem_1 :
  (5 / ((1 / (1 * 2)) + (1 / (2 * 3)) + (1 / (3 * 4)) + (1 / (4 * 5)) + (1 / (5 * 6)))) = 6 := by
  sorry

end problem_1_l462_462758


namespace greatest_divisor_of_arithmetic_sequence_sum_l462_462337

theorem greatest_divisor_of_arithmetic_sequence_sum :
  ∀ (x c : ℕ), ∃ k : ℕ, k = 15 ∧ 15 ∣ (15 * x + 105 * c) :=
by
  intro x c
  exists 15
  split
  . rfl
  . sorry

end greatest_divisor_of_arithmetic_sequence_sum_l462_462337


namespace average_expenditure_Feb_to_July_l462_462739

theorem average_expenditure_Feb_to_July (avg_Jan_to_Jun : ℝ) (spend_Jan : ℝ) (spend_July : ℝ) 
    (total_Jan_to_Jun : avg_Jan_to_Jun = 4200) (spend_Jan_eq : spend_Jan = 1200) (spend_July_eq : spend_July = 1500) :
    (4200 * 6 - 1200 + 1500) / 6 = 4250 :=
by
  sorry

end average_expenditure_Feb_to_July_l462_462739


namespace altitudes_perpendicular_to_base_l462_462193

-- Definition of perpendicularity
def is_perpendicular (l₁ l₂ : Line) : Prop := ∃ p : Point, p ∈ l₁ ∧ p ∈ l₂ ∧ l₁.slope * l₂.slope = -1

-- Definition of altitude
structure Triangle :=
(A B C : Point)
(base : Line := Line_through A B)
[altitude_AC : is_perpendicular base (Line_through B C)]
[altitude_BC : is_perpendicular base (Line_through C A)]
[altitude_AB : is_perpendicular base (Line_through A C)]

-- Proposition to be proven
theorem altitudes_perpendicular_to_base (T : Triangle) : 
  ∀ (l1 l2 : Line), (is_perpendicular l1 l2) → (l1 = T.base ∨ l2 = T.base) := 
by
  sorry

end altitudes_perpendicular_to_base_l462_462193


namespace complex_magnitude_sum_l462_462488

theorem complex_magnitude_sum : |(3 - 5 * complex.I)| + |(3 + 5 * complex.I)| = 2 * real.sqrt 34 :=
by
  sorry

end complex_magnitude_sum_l462_462488


namespace greatest_divisor_of_sum_of_arith_seq_l462_462310

theorem greatest_divisor_of_sum_of_arith_seq (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → d ∣ (15 * (x + 7 * c))) ∧
    (∀ k : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → k ∣ (15 * (x + 7 * c))) → k ≤ d) ∧ 
    d = 15 :=
sorry

end greatest_divisor_of_sum_of_arith_seq_l462_462310


namespace sum_first_100_terms_l462_462116

noncomputable def a : ℕ → ℝ

axiom seq_relation : ∀ n : ℕ, a (n + 1) + (-1 : ℝ)^(n + 1) * a n = 2

theorem sum_first_100_terms : (Finset.range 100).sum a = 100 := 
by
  sorry

end sum_first_100_terms_l462_462116


namespace a_3_eq_3_l462_462121

-- Given conditions
variables {a : ℕ → ℝ}
variable (q : ℝ)
hypothesis h1 : a 1 = 1
hypothesis h5 : a 5 = 9
hypothesis geom_seq : ∀ n : ℕ, a (n + 1) = a n * q

-- What we need to prove
theorem a_3_eq_3 : a 3 = 3 :=
sorry

end a_3_eq_3_l462_462121


namespace perpendicular_vector_solution_parallel_vector_solution_l462_462160

-- Definitions of vectors
def a : ℝ × ℝ × ℝ := (2, -1, 3)
def b (x : ℝ) : ℝ × ℝ × ℝ := (-4, 2, x)

-- Condition for perpendicular vectors
def perpendicular (a b : ℝ × ℝ × ℝ) : Prop := 
  (a.1 * b.1 + a.2 * b.2 + a.3 * b.3) = 0

-- Condition for parallel vectors
def proportional (a b : ℝ × ℝ × ℝ) : Prop := (∃ k : ℝ, k ≠ 0 ∧ (a.1 = k * b.1) ∧ (a.2 = k * b.2) ∧ (a.3 = k * b.3))

-- Proof problem 1: Perpendicular vector condition implies x = 10/3
theorem perpendicular_vector_solution :
  perpendicular a (b x) → x = 10 / 3 :=
by sorry

-- Proof problem 2: Parallel vector condition implies x = -6
theorem parallel_vector_solution :
  proportional a (b x) → x = -6 :=
by sorry

end perpendicular_vector_solution_parallel_vector_solution_l462_462160


namespace fruit_basket_l462_462680

theorem fruit_basket :
  ∀ (oranges apples bananas peaches : ℕ),
  oranges = 6 →
  apples = oranges - 2 →
  bananas = 3 * apples →
  peaches = bananas / 2 →
  oranges + apples + bananas + peaches = 28 :=
by
  intros oranges apples bananas peaches h_oranges h_apples h_bananas h_peaches
  rw [h_oranges, h_apples, h_bananas, h_peaches]
  sorry

end fruit_basket_l462_462680


namespace average_age_l462_462658

theorem average_age (avg_age_sixth_graders avg_age_parents : ℕ) 
    (num_sixth_graders num_parents : ℕ)
    (h1 : avg_age_sixth_graders = 12) 
    (h2 : avg_age_parents = 40) 
    (h3 : num_sixth_graders = 40) 
    (h4 : num_parents = 60) :
    (num_sixth_graders * avg_age_sixth_graders + num_parents * avg_age_parents) 
    / (num_sixth_graders + num_parents) = 28.8 := 
by
  sorry

end average_age_l462_462658


namespace greatest_divisor_sum_of_first_fifteen_terms_l462_462317

theorem greatest_divisor_sum_of_first_fifteen_terms 
  (x c : ℕ) (hx : x > 0) (hc : c > 0):
  ∃ d, d = 15 ∧ d ∣ (15*x + 105*c) :=
by
  existsi 15
  split
  . refl
  . apply Nat.dvd.intro
    existsi (x + 7*c)
    refl
  sorry

end greatest_divisor_sum_of_first_fifteen_terms_l462_462317


namespace determine_plane_l462_462400

-- Given conditions directly from the problem
structure Line (P : Type*) :=
(intersect : P → P → Prop)

variables {P : Type*} [Nonempty P]

-- Definitions corresponding to the options in the problem

/-- Three lines that intersect pairwise but not at the same point. -/
def option_A (l1 l2 l3 : Line P) : Prop := 
∃ p1 p2 p3 : P, 
  p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
  l1.intersect p1 p2 ∧ l2.intersect p2 p3 ∧ l3.intersect p3 p1

/-- Statement to prove -/
theorem determine_plane (l1 l2 l3 : Line P) :
  option_A l1 l2 l3 → ∃ plane : set P, ∀ p ∈ plane, ∀ q ∈ plane, ∃ r ∈ plane, r ≠ p ∧ r ≠ q :=
sorry

end determine_plane_l462_462400


namespace tangents_at_diameter_endpoints_parallel_l462_462641

theorem tangents_at_diameter_endpoints_parallel
  (O A B : Point)
  (circle : Circle)
  (A_on_circle : A ∈ circle)
  (B_on_circle : B ∈ circle)
  (diameter : Line)
  (diam_oa : diameter.contains A)
  (diam_ob : diameter.contains B)
  (radius_OA : is_radius O A)
  (radius_OB : is_radius O B)
  (tangent_tA : Line)
  (tangent_tA_perp : tangent_tA ⊥ Line.through O A)
  (tangent_tB : Line)
  (tangent_tB_perp : tangent_tB ⊥ Line.through O B)
  (OA_straight : Line.through O A = diameter)
  (OB_straight : Line.through O B = diameter) :
  tangent_tA ∥ tangent_tB := by
  sorry

end tangents_at_diameter_endpoints_parallel_l462_462641


namespace find_expression_l462_462592

theorem find_expression (x : ℝ) (h : (1 / Real.cos (2022 * x)) + Real.tan (2022 * x) = 1 / 2022) :
  (1 / Real.cos (2022 * x)) - Real.tan (2022 * x) = 2022 :=
by
  sorry

end find_expression_l462_462592


namespace rationalize_denominator_l462_462643

theorem rationalize_denominator :
  ∃ A B C D E F : ℤ,
  F > 0 ∧
  (1 / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11)) =
  ((A * Real.sqrt 2 + B * Real.sqrt 3 + C * Real.sqrt 5 + D * Real.sqrt 11 + E * Real.sqrt (some_val X)) / F) ∧
  (A + B + C + D + E + F) = 20 := 
sorry

end rationalize_denominator_l462_462643


namespace number_of_product_free_subsets_l462_462760

def product_free (S : Finset ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → a * b ≠ c

def elements := Finset.range (10 + 1)

theorem number_of_product_free_subsets :
  (Finset.filter product_free (Finset.powerset elements)).card = 252 :=
by {
  sorry
}

end number_of_product_free_subsets_l462_462760


namespace range_of_a_l462_462148

noncomputable def f (x a : ℝ) : ℝ := real.sqrt (real.exp x + x - a)

theorem range_of_a :
  (∃ (y_0 : ℝ), -1 ≤ y_0 ∧ y_0 ≤ 1 ∧ f (f y_0 a) a = y_0) ↔ 1 ≤ a ∧ a ≤ real.exp 1 :=
sorry

end range_of_a_l462_462148


namespace convexity_of_function_range_of_a_l462_462008

-- Define a function f(x) based on given conditions
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x

-- Prove convexity of f when a < 0
theorem convexity_of_function (a : ℝ) (h : a < 0) :
  (∀ x1 x2 : ℝ, f a ((x1 + x2) / 2) ≥ (f a x1 + f a x2) / 2) :=
sorry

-- Prove the range of a given the constraints on f(x) in [0, 1]
theorem range_of_a (a : ℝ) :
  ( (∀ x ∈ Icc (0 : ℝ) (1 : ℝ), |f a x| ≤ 1) → (-2 ≤ a ∧ a < 0) ) :=
sorry

end convexity_of_function_range_of_a_l462_462008


namespace largest_unique_digit_sum_eq_sixteen_l462_462373

theorem largest_unique_digit_sum_eq_sixteen :
  ∀ (n : ℕ), (∀ d ∈ digits n, d ≠ (d' : ℕ, d' ∈ digits n → d ≠ d')) ∧ (digits n).sum = 16 → n = 643210 :=
by
  sorry

end largest_unique_digit_sum_eq_sixteen_l462_462373


namespace total_population_l462_462301

-- Define the conditions
variables (T G Td Lb : ℝ)

-- Given conditions and the result
def conditions : Prop :=
  G = 1 / 2 * T ∧
  Td = 0.60 * G ∧
  Lb = 16000 ∧
  T = Td + G + Lb

-- Problem statement: Prove that the total population T is 80000
theorem total_population (h : conditions T G Td Lb) : T = 80000 :=
by
  sorry

end total_population_l462_462301


namespace each_brother_pays_19_80_l462_462214

noncomputable def john_smith_payment : ℝ :=
let cake_price := 12
let num_cakes := 3
let tax_rate := 0.10
let cost_before_tax := num_cakes * cake_price
let tax_amount := tax_rate * cost_before_tax
let total_cost := cost_before_tax + tax_amount
in total_cost / 2

theorem each_brother_pays_19_80 : john_smith_payment = 19.80 :=
by
  -- Proof would go here
  sorry

end each_brother_pays_19_80_l462_462214


namespace max_abs_z_l462_462832

theorem max_abs_z (z : ℂ) (h : abs (z + 2 - 2 * complex.i) = 1) : abs (z - 2 - 2 * complex.i) ≤ 5 :=
sorry

end max_abs_z_l462_462832


namespace probability_plane_intersects_interior_rect_prism_l462_462108

theorem probability_plane_intersects_interior_rect_prism : 
  let total_ways := Nat.choose 8 4,
      non_intersecting_cases := 6 in
    (total_ways - non_intersecting_cases) / total_ways = (32 : ℚ) / 35 := by
sorry

end probability_plane_intersects_interior_rect_prism_l462_462108


namespace total_number_of_flags_is_12_l462_462069

def number_of_flags : Nat :=
  3 * 2 * 2

theorem total_number_of_flags_is_12 : number_of_flags = 12 := by
  sorry

end total_number_of_flags_is_12_l462_462069


namespace angle_bisector_length_l462_462206

theorem angle_bisector_length (a b : ℝ) (h : ∠BAC = 120) (AB AC : LineSegment) (hAB : length AB = a) (hAC : length AC = b) :
  ∃ AM : LineSegment, length AM = a * b / (a + b) :=
by
  sorry

end angle_bisector_length_l462_462206


namespace total_cats_left_l462_462724

theorem total_cats_left 
           (s p h s_s p_s h_s : ℕ) 
           (hs : s = 20) 
           (hp : p = 12) 
           (hh : h = 8)
           (h_s_s : s_s = 8) 
           (h_p_s : p_s = 5) 
           (h_h_s : h_s = 3) : 
           (s + p + h) - (s_s + p_s + h_s) = 24 :=
by 
  rw [hs, hp, hh, h_s_s, h_p_s, h_h_s]
  simp
  sorry

end total_cats_left_l462_462724


namespace find_n_l462_462942

noncomputable def n : ℕ := sorry -- Explicitly define n as a variable, but the value is not yet provided.

theorem find_n (h₁ : n > 0)
    (h₂ : Real.sqrt 3 > (n + 4) / (n + 1))
    (h₃ : Real.sqrt 3 < (n + 3) / n) : 
    n = 4 :=
sorry

end find_n_l462_462942


namespace sum_of_continuous_n_equals_zero_l462_462232

def f (x n : ℝ) : ℝ :=
if x < n then x^2 + 3*x + 1
else 3*x + 7

theorem sum_of_continuous_n_equals_zero :
  (∑ n in ({n : ℝ | f n n = f (n - 1) n })) = 0 :=
by
  sorry

end sum_of_continuous_n_equals_zero_l462_462232


namespace problem_isosceles_right_triangle_l462_462921

/-- Given an isosceles right triangle ABC with AB = AC = 3 and ∠A = 90°,
M is the midpoint of BC, and points I and E lie on AC and AB respectively,
with AI > AE and A, I, M, E concyclic. If the area of ∆EMI is 2 and
CI = (a - √b) / c where a, b, and c are positive integers and b is not a perfect square,
then a + b + c = 12. -/
theorem problem_isosceles_right_triangle :
  ∃ (a b c : ℕ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧ (¬ ∃ (k : ℕ), k * k = b) ∧ 
    (let CI := (a - Real.sqrt b) / c in 
    let areaEMI := 2 in 
    (∃ (AB AC : ℝ), AB = 3 ∧ AC = 3 ∧ angle A = π / 2) ∧
    (let M := midpoint BC in 
    let is_concyclic := cyclic A I M E in
    ∃ (AI AE : ℝ), AI > AE ∧ 
    area (triangle EMI) = 2 ∧ 
    a + b + c = 12)) := sorry

end problem_isosceles_right_triangle_l462_462921


namespace parabola_directrix_l462_462666

theorem parabola_directrix (x y : ℝ) (h : x^2 = 8 * y) : y = -2 :=
sorry

end parabola_directrix_l462_462666


namespace average_speed_whole_journey_l462_462394

-- Definitions based on conditions
def speed_XY : ℝ := 44
def speed_YX : ℝ := 36
def distance : ℝ := 1  -- We will assume a unit distance for simplicity

-- The proof problem
theorem average_speed_whole_journey :
  let T1 := distance / speed_XY in
  let T2 := distance / speed_YX in
  let total_time := T1 + T2 in
  let total_distance := 2 * distance in
  let avg_speed := total_distance / total_time in
  avg_speed = 39.6 := by
  sorry

end average_speed_whole_journey_l462_462394


namespace evaluate_expression_l462_462466

def g (x : ℝ) : ℝ := x^2 - 3 * real.sqrt x

theorem evaluate_expression : 3 * g 3 - g 27 = -702 := by
  sorry

end evaluate_expression_l462_462466


namespace greatest_divisor_arithmetic_sequence_sum_l462_462332

theorem greatest_divisor_arithmetic_sequence_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ k, (15 * (x + 7 * c)) = 15 * k :=
sorry

end greatest_divisor_arithmetic_sequence_sum_l462_462332


namespace side_length_of_square_l462_462418

theorem side_length_of_square (l b : ℕ) (ratio_cond : l / b = 25 / 16) (rect_area_cond : l = 250 ∧ b = 160) : ∃ s : ℕ, s * s = l * b ∧ s = 200 :=
by
  have h1 : l * b = 250 * 160 := by
    sorry
  have h2 : 250 * 160 = 40000 := by
    sorry
  have h3 : ∃ s : ℕ, s * s = 40000 := by
    sorry
  have h4 : s = 200 := by
    sorry
  sorry

end side_length_of_square_l462_462418


namespace stratified_sampling_correct_l462_462717

noncomputable def stratified_sampling (total employees_over50 employees_between35_49 employees_under35 sample_size : ℕ) : (ℕ × ℕ × ℕ) :=
  let proportion_over50 := employees_over50 / (total : ℚ)
  let proportion_between35_49 := employees_between35_49 / (total : ℚ)
  let proportion_under35 := employees_under35 / (total : ℚ)
  let selected_over50 := (sample_size : ℚ) * proportion_over50
  let selected_between35_49 := (sample_size : ℚ) * proportion_between35_49
  let selected_under35 := (sample_size : ℚ) * proportion_under35
  (selected_over50.toNat, selected_between35_49.toNat, selected_under35.toNat)

theorem stratified_sampling_correct :
  stratified_sampling 150 15 45 90 30 = (3, 9, 18) :=
by
  sorry

end stratified_sampling_correct_l462_462717


namespace range_of_m_l462_462820

variable {α : Type*} [LinearOrder α]
  
def A := {x | x < 2}
def B (m : α) := {x | x < m}

theorem range_of_m (m : α) (h : B m ⊆ A) : m ≤ 2 :=
sorry

end range_of_m_l462_462820


namespace no_piece_reaches_y5_l462_462958

noncomputable def omega : ℝ := (-1 + real.sqrt 5) / 2

-- Define a value assignment function for lattice points
def value (x y : ℤ) : ℝ := omega ^ (abs x - y)

-- Prove that given the initial conditions and jumping rules, no piece can reach the line y = 5
theorem no_piece_reaches_y5 
  (initial_positions : set (ℤ × ℤ)) 
  (h_initial : ∀ (x y : ℤ), (x, y) ∈ initial_positions → y ≤ 0) 
  (jump_rule : (ℤ × ℤ) → (ℤ × ℤ) → Prop) 
  (jump_cond : ∀ p1 p2, jump_rule p1 p2 → 
                ∃ (x : ℤ), (x, |p1.1 - p2.1| = 1 ∨ x = |p1.2 - p2.2| = 1) ∧
                           p2.1 = p1.1 + 1 ∨ p2.1 = p1.1 - 1 ∨
                           p2.2 = p1.2 + 1 ∨ p2.2 = p1.2 - 1) :
  ¬ ∃ (x : ℤ), (x, 5) ∈ initial_positions :=
begin
  sorry
end

end no_piece_reaches_y5_l462_462958


namespace greatest_divisor_of_sum_of_arith_seq_l462_462316

theorem greatest_divisor_of_sum_of_arith_seq (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → d ∣ (15 * (x + 7 * c))) ∧
    (∀ k : ℕ, (∀ x c : ℕ, x > 0 ∧ c > 0 → k ∣ (15 * (x + 7 * c))) → k ≤ d) ∧ 
    d = 15 :=
sorry

end greatest_divisor_of_sum_of_arith_seq_l462_462316


namespace min_sugar_l462_462453

variable (f s : ℝ)

theorem min_sugar (h1 : f ≥ 10 + 3 * s) (h2 : f ≤ 4 * s) : s ≥ 10 := by
  sorry

end min_sugar_l462_462453


namespace true_propositions_l462_462037

theorem true_propositions :
  (∀ x : ℝ, x^2 - x ≥ x - 1) ∧ 
  (∃ x : ℝ, x > 1 ∧ x + 4 / (x - 1) = 6) ∧ 
  (∀ a b : ℝ, a > b ∧ b > 0 → b / a < (b + 1) / (a + 1)) ∧ 
  ¬ (∃ x : ℝ, (x^2 + 10) / real.sqrt (x^2 + 9) = 2) :=
by
  -- (1) Prove ∀ x ∈ ℝ, x^2 - x ≥ x - 1
  have hA : ∀ x : ℝ, x^2 - x ≥ x - 1 :=
    fun x => by
      calc
        x^2 - x = (x - 1)^2 + (x - 1) : by ring_exp
        _ ≥ x - 1 : by
          let y := x - 1
          linarith [square_nonneg y]
  -- (2) Prove ∃ x > 1, x + 4 / (x - 1) = 6
  have hB : ∃ x : ℝ, x > 1 ∧ x + 4 / (x - 1) = 6 :=
    have h : ∀ x, x^2 - 7 * x + 10 = 0 → (x = 2 ∨ x = 5) :=
      fun x h =>
        quadratic_formula ℝ x (by linarith[1, -7]) (by linarith[10])
        |> fun ⟨w₁, w₂⟩ =>
          and.intro w₁.left w₂.right
    exists.intro 2 (and.intro (by linarith) (by norm_num[quadratic_formula]; linarith))
  -- (3) Prove ∀ a b : ℝ, a > b > 0 → b / a < (b + 1) / (a + 1)
  have hC : ∀ a b : ℝ, a > b ∧ b > 0 → b / a < (b + 1) / (a + 1) :=
    fun a b h =>
      let h := h.left (and.right h)
      rat_lt_ofWneq (fraction_lt (show a * (a + 1) > 0 by exactly_nonzero)
  -- (4) Prove ¬ ∃ x, (x^2 + 10) / real.sqrt (x^2 + 9) = 2
  have hD : ¬ ∃ x : ℝ, (x^2 + 10) / real.sqrt (x^2 + 9) = 2 :=
    have h : (x^2 + 10 / real.sqrt (x^2 + 9) ≥ _, by (fun x => AM_GM_inequality)
      and.intro (by linarith) (by linarith [real.sqrt_pos _])
  exactly ⟨hA, hB, hC, hD⟩

end true_propositions_l462_462037


namespace pizza_volume_one_piece_l462_462421

theorem pizza_volume_one_piece :
  ∀ (h t: ℝ) (d: ℝ) (n: ℕ), d = 16 → t = 1/2 → n = 8 → h = 8 → 
  ( (π * (d / 2)^2 * t) / n = 4 * π ) :=
by 
  intros h t d n hd ht hn hh
  sorry

end pizza_volume_one_piece_l462_462421


namespace only_one_expression_equals_l462_462882

theorem only_one_expression_equals (x : ℝ) (h : x > 0) :
  (2x^{x + 1} = 2x^{x + 1}) ∧
  (x^{2x + 2} ≠ 2x^{x + 1}) ∧
  ((x + 1)^{x + 1} ≠ 2x^{x + 1}) ∧
  ((2x)^{x + 1} ≠ 2x^{x + 1}) :=
by
  sorry

end only_one_expression_equals_l462_462882


namespace least_number_div_by_24_32_36_54_increased_by_3_eq_861_l462_462391

theorem least_number_div_by_24_32_36_54_increased_by_3_eq_861 :
  let lcm := Nat.lcm (Nat.lcm 24 32) (Nat.lcm 36 54)
  in lcm - 3 = 861 :=
by
  -- Definition setup according to conditions found in a)
  have lcm_def : Nat.lcm (Nat.lcm 24 32) (Nat.lcm 36 54) = 864 :=
    by norm_num
  -- Least number setup according to steps in b)
  show 864 - 3 = 861 from
    by norm_num
  done

end least_number_div_by_24_32_36_54_increased_by_3_eq_861_l462_462391


namespace red_marble_fraction_l462_462899

theorem red_marble_fraction (x : ℕ) (hx : x > 0) :
  let blue_initial := (2 / 3) * x,
      red_initial := (1 / 3) * x,
      blue_tripled := 3 * blue_initial,
      total_new := blue_tripled + red_initial in
  red_initial / total_new = 1 / 7 :=
by
  sorry

end red_marble_fraction_l462_462899


namespace number_of_distinct_intersections_l462_462778

/-- The problem is to prove that the number of distinct intersection points
in the xy-plane for the graphs of the given equations is exactly 4. -/
theorem number_of_distinct_intersections :
  ∃ (S : Finset (ℝ × ℝ)), 
  (∀ p : ℝ × ℝ, p ∈ S ↔
    ((p.1 + p.2 = 7 ∨ 2 * p.1 - 3 * p.2 + 1 = 0) ∧
     (p.1 - p.2 - 2 = 0 ∨ 3 * p.1 + 2 * p.2 - 10 = 0))) ∧
  S.card = 4 :=
sorry

end number_of_distinct_intersections_l462_462778


namespace lawn_length_l462_462721

theorem lawn_length (area width : ℝ) (h_area : area = 20) (h_width : width = 5) : 
  ∃ length : ℝ, length = 4 :=
by
  have h : 20 = 4 * 5 := by norm_num
  existsi 4
  exact h

end lawn_length_l462_462721


namespace max_principals_in_10_years_l462_462078

theorem max_principals_in_10_years (h : ∀ p : ℕ, 4 * p ≤ 10) :
  ∃ n : ℕ, n ≤ 3 ∧ n = 3 :=
sorry

end max_principals_in_10_years_l462_462078


namespace bricks_needed_for_wall_l462_462161

noncomputable def brick_volume (length : ℝ) (height : ℝ) (thickness : ℝ) : ℝ :=
  length * height * thickness

noncomputable def wall_volume (length : ℝ) (height : ℝ) (average_thickness : ℝ) : ℝ :=
  length * height * average_thickness

noncomputable def number_of_bricks (wall_vol : ℝ) (brick_vol : ℝ) : ℝ :=
  wall_vol / brick_vol

theorem bricks_needed_for_wall : 
  let length_wall := 800
  let height_wall := 660
  let avg_thickness_wall := (25 + 22.5) / 2 -- in cm
  let length_brick := 25
  let height_brick := 11.25
  let thickness_brick := 6
  let mortar_thickness := 1

  let adjusted_length_brick := length_brick + mortar_thickness
  let adjusted_height_brick := height_brick + mortar_thickness

  let volume_wall := wall_volume length_wall height_wall avg_thickness_wall
  let volume_brick_with_mortar := brick_volume adjusted_length_brick adjusted_height_brick thickness_brick

  number_of_bricks volume_wall volume_brick_with_mortar = 6565 :=
by
  sorry

end bricks_needed_for_wall_l462_462161


namespace probability_event_l462_462968

theorem probability_event (x : ℝ) (h_interval : x ∈ set.Icc (-1 : ℝ) 2) :
  measure_theory.measure_space.probability (set_of (λ x, 1 ≤ 2^x ∧ 2^x ≤ 2)) = 1 / 3 :=
sorry

end probability_event_l462_462968


namespace length_of_chord_l462_462874

variables {t : ℝ}

def curve_C : set (ℝ × ℝ) := {p | p.1 ^ 2 + p.2 ^ 2 = 1}
def line_l : set (ℝ × ℝ) := {p | ∃ t, p.1 = -1 + 4 * t ∧ p.2 = 3 * t}

theorem length_of_chord : 
  let d := abs (3 * 0 - 4 * 0 + 3) / sqrt (3 ^ 2 + (-4) ^ 2) in
  2 * sqrt (1 ^ 2 - d ^ 2) = 8 / 5 := 
by
  let d := abs (3 * 0 - 4 * 0 + 3) / sqrt (3 ^ 2 + (-4) ^ 2)
  show 2 * sqrt (1 ^ 2 - d ^ 2) = 8 / 5
  sorry

end length_of_chord_l462_462874


namespace find_real_k_l462_462788

theorem find_real_k :
  ∃ k : ℝ, 
  ∥ k • (3 : ℝ, -4 : ℝ, 1 : ℝ) - (6 : ℝ, 2 : ℝ, -3 : ℝ) ∥ = 3 * Real.sqrt 26 ∧ 
  (k = 151 / 52 ∨ k = -123 / 52) :=
sorry

end find_real_k_l462_462788


namespace probability_of_spade_or_king_is_4_over_13_l462_462180

def total_cards : ℕ := 52
def spades_count : ℕ := 13
def kings_count : ℕ := 4
def king_spades_overlap : ℕ := 1
def favorable_outcomes : ℕ := spades_count + kings_count - king_spades_overlap := by rfl
def probability_of_spade_or_king := (favorable_outcomes, total_cards) → ℚ := (favorable_outcomes / total_cards) by rfl

theorem probability_of_spade_or_king_is_4_over_13 : probability_of_spade_or_king = 4 / 13 :=
by
  sorry

end probability_of_spade_or_king_is_4_over_13_l462_462180


namespace new_pyramid_volume_l462_462416

theorem new_pyramid_volume
  (vol_original : ℝ) (h_original : ℝ) (w_original : ℝ) (l_original : ℝ)
  (h_reduction : ℝ) (w_reduction : ℝ) (new_volume : ℝ)
  (h : vol_original = 72)
  (h_base : h_original = 9)
  (w_base : w_original = 2)
  (l_base : l_original = 2)
  (h_red : h_reduction = 0.30)
  (w_red : w_reduction = 0.10)
  (new_h : ℝ := (1 - h_reduction) * h_original)
  (new_w : ℝ := (1 - w_reduction) * w_original)
  (new_l : ℝ := l_original)
  (calc_volume : new_volume = (1 / 3) * new_l * new_w * new_h) :
  new_volume = 7.56 :=
begin
  rw [h, h_base, w_base, l_base, h_red, w_red] at *,
  simp [new_h, new_w, new_l],
  norm_num,
end

end new_pyramid_volume_l462_462416


namespace no_both_squares_l462_462230

theorem no_both_squares {x y : ℕ} (hx : x > 0) (hy : y > 0) : ¬ (∃ a b : ℕ, a^2 = x^2 + 2 * y ∧ b^2 = y^2 + 2 * x) :=
by
  sorry

end no_both_squares_l462_462230


namespace odd_nat_composite_iff_exists_a_l462_462252

theorem odd_nat_composite_iff_exists_a (c : ℕ) (h_odd : c % 2 = 1) :
  (∃ a : ℕ, a ≤ c / 3 - 1 ∧ ∃ k : ℕ, (2*a - 1)^2 + 8*c = k^2) ↔
  ∃ d : ℕ, ∃ e : ℕ, d > 1 ∧ e > 1 ∧ d * e = c := 
sorry

end odd_nat_composite_iff_exists_a_l462_462252


namespace alberto_vs_bjorn_alberto_vs_carlos_l462_462191

-- Define the distances calculated by each biker
def alberto_distance : ℕ := 90
def bjorn_distance : ℕ := 72
def carlos_distance : ℕ := 60

-- Define the difference in distances between Albert and Bjorn after 6 hours
theorem alberto_vs_bjorn : alberto_distance - bjorn_distance = 18 :=
by
  have h : 90 - 72 = 18 := rfl
  exact h

-- Define the difference in distances between Albert and Carlos after 6 hours
theorem alberto_vs_carlos : alberto_distance - carlos_distance = 30 :=
by
  have h : 90 - 60 = 30 := rfl
  exact h

end alberto_vs_bjorn_alberto_vs_carlos_l462_462191


namespace fee_difference_l462_462689

-- Defining the given conditions
def stadium_capacity : ℕ := 2000
def fraction_full : ℚ := 3 / 4
def entry_fee : ℚ := 20

-- Statement to prove
theorem fee_difference :
  let people_at_three_quarters := stadium_capacity * fraction_full
  let total_fees_at_three_quarters := people_at_three_quarters * entry_fee
  let total_fees_full := stadium_capacity * entry_fee
  total_fees_full - total_fees_at_three_quarters = 10000 :=
by
  sorry

end fee_difference_l462_462689


namespace cos_two_x_increases_on_interval_l462_462153

noncomputable def cos_two_x_increasing_interval : set ℝ :=
  {x | ∃ k : ℤ, (k * real.pi + real.pi / 2 ≤ x ∧ x ≤ k * real.pi + real.pi)}

theorem cos_two_x_increases_on_interval :
  ∀ x ∈ Icc (0 : ℝ) real.pi, x ∈ cos_two_x_increasing_interval -> (1/2 : ℝ).bit0 * real.pi ≤ x ∧ x ≤ real.pi :=
by sorry

end cos_two_x_increases_on_interval_l462_462153


namespace part1_part2_l462_462825

theorem part1 (m n : ℤ) (h : m + 4 * n - 3 = 0) : 2^m * 16^n = 8 := 
by
  sorry

theorem part2 (x : ℝ) (n : ℤ) (hn : 0 < n) (hx : x^(2 * n) = 4) : 
  (x^(3 * n))^2 - 2 * (x^2)^(2 * n) = 32 := 
by
  sorry

end part1_part2_l462_462825


namespace isla_capsules_days_l462_462211

theorem isla_capsules_days (days_in_july : ℕ) (days_forgot : ℕ) (known_days_in_july : days_in_july = 31) (known_days_forgot : days_forgot = 2) : days_in_july - days_forgot = 29 := 
by
  -- Placeholder for proof, not required in the response.
  sorry

end isla_capsules_days_l462_462211


namespace fraction_books_sold_l462_462004

theorem fraction_books_sold (B : ℕ) (F : ℚ) (h1 : 36 = B - F * B) (h2 : 252 = 3.50 * F * B) : F = 2 / 3 := by
  -- Proof omitted
  sorry

end fraction_books_sold_l462_462004


namespace color_distribution_l462_462784

theorem color_distribution (S : Fin 100 -> Finset (Fin 100 × Fin 100)) 
  (hS : ∀ i, (S i).card = 100) :
  ∃ r : Fin 100, ∃ c : Fin 100, 
    (S ∩ { p | p.1 = r } ).card ≥ 10 ∨ (S ∩ { p | p.2 = c }).card ≥ 10 := by
  sorry

end color_distribution_l462_462784


namespace area_triangle_PCR_l462_462253

/-- Given a parallelogram ABCD with area 48 square units. Points P and Q lie on sides AB and CD respectively such that AP = PB and CQ = QD. Point R is the midpoint of side BC. Prove that the area of triangle PCR is 12 square units. -/
theorem area_triangle_PCR 
  (A B C D P Q R : Type)
  (h₁ : parallelogram A B C D)
  (h₂ : area A B C D = 48)
  (h₃ : P ∈ segment A B)
  (h₄ : Q ∈ segment C D)
  (h₅ : AP = PB)
  (h₆ : CQ = QD)
  (h₇ : R = midpoint C B) :
  area_triangle P C R = 12 := 
sorry

end area_triangle_PCR_l462_462253


namespace lines_through_P_intersection_count_l462_462948

-- Definitions of the hyperbola and point P
def hyperbola (x y : ℝ) : Prop := (x^2 / 16) - (y^2 / 9) = 1
def point_P : ℝ × ℝ := (4, 4)

-- The main statement asserting there are exactly 4 such lines
theorem lines_through_P_intersection_count :
  ∃ l : set (ℝ × ℝ), 
    (∀ p ∈ l, ∃ m b : ℝ, ∀ x : ℝ, p = (x, m * x + b)) ∧
    (∀ p ∈ l, ∃! (x y : ℝ), hyperbola x y ∧ ((x, y) = p)) ∧
    (∀ p ∈ l, (p = (4, 4))) ∧
    (Fintype.card l = 4) := sorry

end lines_through_P_intersection_count_l462_462948


namespace boys_in_classroom_l462_462919

theorem boys_in_classroom :
  ∃ boys : ℕ,
  let desks_with_boy_and_girl := 2 in
  let desks_with_two_girls := 2 * desks_with_boy_and_girl in
  let desks_with_two_boys := 2 * desks_with_two_girls in
  let total_girls := 10 in
  let calculated_girls := 2 * desks_with_two_girls + desks_with_boy_and_girl in
  calculated_girls = total_girls →
  boys = (2 * desks_with_two_boys + desks_with_boy_and_girl):=
by
  use 18
  intro
  sorry

end boys_in_classroom_l462_462919
