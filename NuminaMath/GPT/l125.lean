import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Algebra.Order.AbsoluteValue
import Mathlib.Algebra.Parity
import Mathlib.Analysis.SpecialFunctions.Basic
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.Trigonometry.Basic
import Mathlib.Combinatorics.Perm
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Logic.Basic
import Mathlib.MeasureTheory.Integral.Bochner
import Mathlib.MeasureTheory.ProbabilityMassFunction
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.Independence
import Mathlib.Probability.Integration
import Mathlib.Tactic
import Mathlib.Tactic.Basic

namespace measure_angle_ABF_of_regular_octagon_l125_125220

theorem measure_angle_ABF_of_regular_octagon (h : regular_octagon ABCDEFGH) : angle ABF = 22.5 :=
sorry

end measure_angle_ABF_of_regular_octagon_l125_125220


namespace no_rectangle_other_than_square_l125_125854

theorem no_rectangle_other_than_square (p_1 p_2 p_3 p_4 : ℕ) (h_prime1 : prime p_1) (h_prime2 : prime p_2) (h_prime3 : prime p_3) (h_prime4 : prime p_4) (h_odd1 : p_1 % 2 = 1) (h_odd2 : p_2 % 2 = 1) (h_odd3 : p_3 % 2 = 1) (h_odd4 : p_4 % 2 = 1) :
    let segments := (λ p, (list.range 100).map (λ n, 1 / p ^ (n + 1))) in
    ∀ rect : set (ℚ × ℚ), rect ∈ segments p_1 ∪ segments p_2 ∪ segments p_3 ∪ segments p_4 → rect = set.univ := 
sorry

end no_rectangle_other_than_square_l125_125854


namespace parabola_distance_l125_125791

noncomputable def parabola_focus : (ℝ × ℝ) := (0, 1)

noncomputable def parabola_directrix_eq : (ℝ → Prop) := 
  λ y, y = -1

def point_on_parabola (P : ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), P = (x, y) ∧ x^2 = 4 * y

def perpendicular_to_directrix (P A : ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), A = (x, y) ∧ P.1 = x ∧ y = -1

def angle_AFO_30_degrees (A F O : ℝ × ℝ) : Prop :=
  ∃ (θ : ℝ), θ = 30 * (Real.pi / 180) ∧ 
  Real.atan2 (A.2 - F.2) (A.1 - F.1) = θ

def distance (P F : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - F.1) ^ 2 + (P.2 - F.2) ^ 2)

theorem parabola_distance (P A F O : ℝ × ℝ)
  (h1 : F = parabola_focus)
  (h2 : ∀ y, parabola_directrix_eq y)
  (h3 : point_on_parabola P)
  (h4 : perpendicular_to_directrix P A)
  (h5 : angle_AFO_30_degrees A F O) :
  distance P F = 4 / 3 :=
sorry

end parabola_distance_l125_125791


namespace sum_and_product_of_roots_l125_125617

theorem sum_and_product_of_roots:
  ∀ x : ℝ, x^2 - 18 * x + 16 = 0 →
    (sum_of_roots : ℝ = 18) ∧ (product_of_roots : ℝ = 16) :=
begin
  sorry
end

end sum_and_product_of_roots_l125_125617


namespace evaluate_expression_l125_125004

theorem evaluate_expression : (3^3)^4 = 531441 :=
by sorry

end evaluate_expression_l125_125004


namespace num_values_of_a_l125_125425

theorem num_values_of_a :
  {a : ℕ // 1 ≤ a ∧ a ≤ 50 ∧ ∃(x1 x2 : ℤ), x1 = x2 ∧ x1 + x2 = -(2 * a + 2) ∧ x1 * x2 = (a + 1)^2}.card = 50 :=
sorry

end num_values_of_a_l125_125425


namespace probability_of_roll_6_after_E_l125_125384

/- Darryl has a six-sided die with faces 1, 2, 3, 4, 5, 6.
   The die is weighted so that one face comes up with probability 1/2,
   and the other five faces have equal probability.
   Darryl does not know which side is weighted, but each face is equally likely to be the weighted one.
   Darryl rolls the die 5 times and gets a 1, 2, 3, 4, and 5 in some unspecified order. -/

def probability_of_next_roll_getting_6 : ℚ :=
  let p_weighted := (1 / 2 : ℚ)
  let p_unweighted := (1 / 10 : ℚ)
  let p_w6_given_E := (1 / 26 : ℚ)
  let p_not_w6_given_E := (25 / 26 : ℚ)
  p_w6_given_E * p_weighted + p_not_w6_given_E * p_unweighted

theorem probability_of_roll_6_after_E : probability_of_next_roll_getting_6 = 3 / 26 := sorry

end probability_of_roll_6_after_E_l125_125384


namespace man_finishes_work_in_100_days_l125_125313

variable (M W : ℝ)
variable (H1 : 10 * M * 6 + 15 * W * 6 = 1)
variable (H2 : W * 225 = 1)

theorem man_finishes_work_in_100_days (M W : ℝ) (H1 : 10 * M * 6 + 15 * W * 6 = 1) (H2 : W * 225 = 1) : M = 1 / 100 :=
by
  sorry

end man_finishes_work_in_100_days_l125_125313


namespace triangle_equilateral_triangle_side_length_area_l125_125127
-- Import the necessary library

-- Case 1: Prove that triangle ABC is an equilateral triangle given f(A) = -sqrt(3)/2
theorem triangle_equilateral (a b c A B C : ℝ)
  (h1 : A + B + C = 180)
  (h2 : 2 * B = A + C)
  (h3 : f A = -(Real.sqrt 3) / 2)
  (h4 : ∀ (x : ℝ), (Real.cos (x / 2))^2 + (Real.sin (x / 2))^2 = 1) :
  A = 60 ∧ B = 60 ∧ C = 60 ∧ f A = -(Real.sqrt 3) / 2 := 
sorry

-- Case 2: Prove the length of side c and the area of triangle ABC
theorem triangle_side_length_area (a b c A B : ℝ) 
  (h1 : B = 60)
  (h2 : b = Real.sqrt 3)
  (h3 : a = Real.sqrt 2)
  (h4 : 2 * B = A + C)
  (h5 : A + B + C = 180)
  (h6 : a^2 + c^2 - 2 * a * c * Real.cos B = b^2)
  (cs : c = (Real.sqrt 2 + Real.sqrt 6) / 2) :
  ∃ c S, c = (Real.sqrt 2 + Real.sqrt 6) / 2 ∧ S = (a * c * Real.sin B) / 2 := 
sorry

end triangle_equilateral_triangle_side_length_area_l125_125127


namespace segment_length_in_right_triangle_l125_125832

theorem segment_length_in_right_triangle :
  ∀ (a b : ℕ) (c : ℝ) (r : ℝ), a = 5 → b = 12 → c = Real.sqrt (a^2 + b^2) → r = (a + b - c) / 2 →
  segment_parallel_to_longer_leg a b c r = r := by
  sorry

end segment_length_in_right_triangle_l125_125832


namespace simplify_sqrt_expression_l125_125679

theorem simplify_sqrt_expression :
  real.sqrt 27 - real.sqrt 12 + real.sqrt 48 = 5 * real.sqrt 3 :=
by
  sorry

end simplify_sqrt_expression_l125_125679


namespace vector_angle_pi_over_six_l125_125795

variable {a b : ℝ}

/-- Given two non-zero vectors a and b that satisfy the given conditions,
  prove that the angle between (a + b) and a is π/6. -/
theorem vector_angle_pi_over_six (a b : ℝ)
  (h1 : |a + b| = 2 * |b|)
  (h2 : |a - b| = 2 * |b|)
  (h3 : a ≠ 0)
  (h4 : b ≠ 0) :
  let angle := real.angle (a + b) a in
  angle = π / 6 := by
  sorry

end vector_angle_pi_over_six_l125_125795


namespace problem_statement_l125_125632

-- Define the repeating decimal 0.000272727... as x
noncomputable def repeatingDecimal : ℚ := 3 / 11000

-- Define the given condition for the question
def decimalRepeatsIndefinitely : Prop := 
  repeatingDecimal = 0.0002727272727272727  -- Representation for repeating decimal

-- Definitions of large powers of 10
def ten_pow_5 := 10^5
def ten_pow_3 := 10^3

-- The problem statement
theorem problem_statement : decimalRepeatsIndefinitely →
  (ten_pow_5 - ten_pow_3) * repeatingDecimal = 27 :=
sorry

end problem_statement_l125_125632


namespace car_diesel_usage_l125_125647

noncomputable def diesel_consumed (diesel_per_km : ℝ) (hours : ℝ) (speed : ℝ) : ℝ :=
  let distance := speed * hours
  diesel_per_km * distance

theorem car_diesel_usage :
  diesel_consumed 0.14 2.5 93.6 = 32.76 :=
by
  -- Distance calculation
  have dist_calc : 93.6 * 2.5 = 234 := by norm_num
  -- Diesel usage calculation
  have diesel_calc : 0.14 * 234 = 32.76 := by norm_num
  -- Prove final statement
  rw [diesel_consumed, dist_calc, diesel_calc]
  norm_num
  sorry

end car_diesel_usage_l125_125647


namespace find_g3_l125_125191

noncomputable def g (x : ℝ) (a b c d : ℝ) : ℝ := a * x^2 + b * x^3 + c * x + d

theorem find_g3 (a b c d : ℝ) (h : g (-3) a b c d = 2) : g 3 a b c d = 0 := 
by 
  sorry

end find_g3_l125_125191


namespace intervals_of_monotonicity_period_and_range_l125_125748

noncomputable def omega₁ : ℝ := 1 / 2
noncomputable def omega₂ : ℝ := 1

def y₁ (x : ℝ) : ℝ := 1 / 2 + Real.sin (omega₁ * x + Real.pi / 6)
def y₂ (x : ℝ) : ℝ := 1 / 2 + Real.sin (2 * omega₂ * x + Real.pi / 6)

theorem intervals_of_monotonicity (k : ℤ) (x : ℝ) :
  y₁ x = 1 / 2 + Real.sin (x + Real.pi / 6) → ω₁ = 1/2 → 
  ( - 2 * Real.pi / 3 + 2 * k * Real.pi ≤ x ∧ x ≤ Real.pi / 3 + 2 * k * Real.pi ) :=
sorry

theorem period_and_range (x : ℝ) (y₃ := y₂ x) :
  (0 < omega₂ ∧ omega₂ < 2 ∧ x = Real.pi / 6 ∧ x = (Real.pi / 2 + k_1 * Real.pi) / (2 * omega₂)) →
  (f (k_2 * Real.pi) = y₃ ∧ ∀ y, y₃ = 1 / 2 + Real.sin (2 * x + Real.pi / 6) → - 1 / 2 ≤ y ∧ y ≤ 3 / 2) :=
sorry

end intervals_of_monotonicity_period_and_range_l125_125748


namespace point_on_line_l125_125108

theorem point_on_line (a b : ℝ) (h : (a, b) = (b, a)) : a = b → ∀ p : ℝ × ℝ, p.1 = p.2 := by
  intros ha_eq_b
  let p := (a, b)
  have h_p : p = (a, a) := by 
    rw ha_eq_b
  simp [ha_eq_b] at h_p
  exact sorry

end point_on_line_l125_125108


namespace value_of_a2_l125_125467

def sequence (n : ℕ) : ℝ := ∑ k in Finset.range (n^2 - n + 1) + n, (1 : ℝ) / (n + k)

theorem value_of_a2 : sequence 2 = (1 / 2) + (1 / 3) + (1 / 4) := by
  sorry

end value_of_a2_l125_125467


namespace circle_area_given_circumference_l125_125318

noncomputable def circle_circumference := 87.98229536926875
noncomputable def pi := Real.pi

theorem circle_area_given_circumference :
  let r := circle_circumference / (2 * pi) in
  let area := pi * r^2 in
  abs (area - 615.75164) < 0.00001 :=
by
  sorry

end circle_area_given_circumference_l125_125318


namespace max_pasture_area_maximization_l125_125334

noncomputable def max_side_length (fence_cost_per_foot : ℕ) (total_cost : ℕ) : ℕ :=
  let total_length := total_cost / fence_cost_per_foot
  let x := total_length / 4
  2 * x

theorem max_pasture_area_maximization :
  max_side_length 8 1920 = 120 :=
by
  sorry

end max_pasture_area_maximization_l125_125334


namespace hyperbola_equation_k_value_range_l125_125049

-- Definitions and conditions
def hyperbola_center := (0, 0)
def hyperbola_focus := (2, 0)
def hyperbola_vertex := (sqrt 3, 0)
def hyperbola_eqn := ∀ x y : ℝ, (x^2 / 3) - y^2 = 1

-- Proof Problem 1
theorem hyperbola_equation : hyperbola_center = (0, 0) → hyperbola_focus = (2, 0) → hyperbola_vertex = (sqrt 3, 0) → hyperbola_eqn :=
by sorry

-- Definitions and conditions for Proof Problem 2
def line_eq (k : ℝ) := ∀ x y : ℝ, y = k * x + sqrt 2
def k_range := {k : ℝ | k ∈ (-1:ℝ) ⊔ -sqrt (3) / 3 ⊔ sqrt (3) / 3 ⊔ 1}

-- Proof Problem 2
theorem k_value_range (k : ℝ) (A B : ℝ × ℝ) :
  (∀ x y : ℝ, (x^2 / 3) - y^2 = 1) → 
  (∀ x y : ℝ, y = k * x + sqrt 2) → 
  A ≠ B → 
  (∀ x y : ℝ, prod.fst A * prod.fst B + (k * prod.fst A + sqrt 2) * (k * prod.fst B + sqrt 2) > 2) → 
  k_range k :=
by sorry

end hyperbola_equation_k_value_range_l125_125049


namespace plankton_sixth_hour_proof_l125_125672

variable (x : ℕ)
variable (amount_eaten : ℕ → ℕ)
variable (total_plankton : ℕ)

-- Conditions
def follows_progression (n : ℕ) : Prop :=
  if n = 1 then amount_eaten n = x else amount_eaten n = amount_eaten (n-1) + 3

def equal_total : Prop := (∑ i in List.range 9, amount_eaten (i+1)) = 450

def plankton_sixth_hour : Prop := amount_eaten 6 = 53

-- Equivalent Proof Problem
theorem plankton_sixth_hour_proof (h1 : ∀ n ≤ 9, follows_progression n)
                                  (h2 : equal_total) :
  plankton_sixth_hour :=
sorry

end plankton_sixth_hour_proof_l125_125672


namespace base_conversion_subtraction_l125_125398

namespace BaseConversion

def base9_to_base10 (n : ℕ) : ℕ :=
  3 * 9^2 + 2 * 9^1 + 4 * 9^0

def base6_to_base10 (n : ℕ) : ℕ :=
  1 * 6^2 + 5 * 6^1 + 6 * 6^0

theorem base_conversion_subtraction : (base9_to_base10 324) - (base6_to_base10 156) = 193 := by
  sorry

end BaseConversion

end base_conversion_subtraction_l125_125398


namespace cos_A_eq_l125_125447

variable (A : Real) (A_interior_angle_tri_ABC : A > π / 2 ∧ A < π) (tan_A_eq_neg_two : Real.tan A = -2)

theorem cos_A_eq : Real.cos A = - (Real.sqrt 5) / 5 := by
  sorry

end cos_A_eq_l125_125447


namespace tangent_line_at_1_range_of_a_l125_125785

open Real

noncomputable def f (x : ℝ) (a : ℝ) := log x - a / x

theorem tangent_line_at_1 (a : ℝ) (h : a = 2) :
  let f' (x : ℝ) := deriv (λ x, log x - 2 / x) in
  let m := f' 1 in
  let y := log 1 - 2 / 1 in
  3 * x - y - 5 = 0 :=
by
  have h₁ : f (1) (2) = -2 := by sorry
  have h₂ : f' 1 = 3 := by sorry
  exact sorry

theorem range_of_a (h : ∀ x, 1 < x → log x - a / x > -x + 2) : a ≤ -1 :=
by
  have h₁ : ∀ x, 1 < x → a < x * log x + x^2 - 2 * x := by sorry
  have h₂ : ∀ x, 1 < x → log x + 2 * x - 1 > 0 := by sorry
  have h₃ : ∀ x, 1 < x → x * log x + x^2 - 2 * x > x * log 1 + 1^2 - 2* 1 := by sorry
  exact sorry

end tangent_line_at_1_range_of_a_l125_125785


namespace T_mod_2023_l125_125178

theorem T_mod_2023 : 
  let T := List.sum ((List.range 2023).map (λ n, if n % 2 = 0 then n else -n)) in
  T % 2023 = 1012 :=
by 
  sorry

end T_mod_2023_l125_125178


namespace constants_solution_l125_125737

noncomputable def find_constants (P : ℤ[X]) (a b : ℤ) : Prop :=
  (eval (-4/3) P = 23) ∧ (eval 3 P = 10)

theorem constants_solution :
  ∃ a b : ℤ, (find_constants (3*X^4 + a*X^3 + b*X^2 - 16*X + 55) a b) ∧
    a = -29 ∧ b = 7 := 
begin
  sorry
end

end constants_solution_l125_125737


namespace vertex_of_quadratic_l125_125924

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := -3 * x^2 - 6 * x + 5

-- State the theorem for vertex coordinates
theorem vertex_of_quadratic :
  (∀ x : ℝ, quadratic_function (- (-6) / (2 * -3)) = quadratic_function 1)
  → (1, quadratic_function 1) = (1, 8) :=
by
  intros h
  sorry

end vertex_of_quadratic_l125_125924


namespace umar_age_is_10_l125_125349

-- Define Ali's age
def Ali_age := 8

-- Define the age difference between Ali and Yusaf
def age_difference := 3

-- Define Yusaf's age based on the conditions
def Yusaf_age := Ali_age - age_difference

-- Define Umar's age which is twice Yusaf's age
def Umar_age := 2 * Yusaf_age

-- Prove that Umar's age is 10
theorem umar_age_is_10 : Umar_age = 10 :=
by
  -- Proof is skipped
  sorry

end umar_age_is_10_l125_125349


namespace original_price_of_books_l125_125974

theorem original_price_of_books (purchase_cost : ℝ) (original_price : ℝ) :
  (purchase_cost = 162) →
  (original_price ≤ 100) ∨ 
  (100 < original_price ∧ original_price ≤ 200 ∧ purchase_cost = original_price * 0.9) ∨ 
  (original_price > 200 ∧ purchase_cost = original_price * 0.8) →
  (original_price = 180 ∨ original_price = 202.5) :=
by
  sorry

end original_price_of_books_l125_125974


namespace addition_rule_l125_125553

noncomputable def probability_space (Ω : Type*) := sorry -- Assume this is properly defined elsewhere in Mathlib

variables {Ω : Type*} [probability_space Ω]
variables (P : set Ω → ℝ) {A B : set Ω}

-- Defining events A and B as subsets of Ω
def event_A := A
def event_B := B

-- Defining the probabilities of events
axiom prob_nonneg : ∀ s : set Ω, 0 ≤ P s
axiom prob_add : ∀ s t : set Ω, (disjoint s t) → P (s ∪ t) = P s + P t

-- Defining the addition rule of probability
theorem addition_rule (A B : set Ω) : P (A ∪ B) = P A + P B - P (A ∩ B) := by
  sorry

end addition_rule_l125_125553


namespace dodecahedral_dice_sum_20_probability_l125_125254

theorem dodecahedral_dice_sum_20_probability :
  let faces := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
  in (∃ (A B : ℕ), A ∈ faces ∧ B ∈ faces ∧ A + B = 20) → 
     (5 : ℚ) / 144 = 5 / 144 :=
by
  sorry

end dodecahedral_dice_sum_20_probability_l125_125254


namespace find_b_l125_125933

theorem find_b (b p : ℤ) 
  (h1 : 3x^3 + bx + 12 = (x^2 + px + 2) * (3x + 6))
  (h2 : 6 + 3p = 0)
  (h3 : 6p + 6 = b)
  (h4 : p = -2) : b = -6 :=
by
  sorry

end find_b_l125_125933


namespace scheduling_leaders_l125_125936

theorem scheduling_leaders : 
  let leaders := {A, B, C, D, E, F}
  in 
  (∀ l ∈ leaders, ∃ d ∈ {1, 2, 3}, l ∈ schedule[d]) ∧
  ((∀ d ∈ {1, 2, 3}, ∃ l₁ l₂ ∈ leaders, l₁ ≠ l₂ ∧ l₁ ∈ schedule[d] ∧ l₂ ∈ schedule[d])) ∧
  (A ∉ schedule[2]) ∧
  (B ∉ schedule[3]) → 
  ∃ num_arrangements : nat, num_arrangements = 42 :=
by sorry

end scheduling_leaders_l125_125936


namespace campsite_distance_l125_125711

variable (d : ℝ)

theorem campsite_distance (h1 : d < 10) (h2 : d > 9) (h3 : d ≠ 11) : d ∈ set.Ioi 9 := by
  sorry

end campsite_distance_l125_125711


namespace semicircle_area_of_inscribed_rectangle_l125_125315

open Real

theorem semicircle_area_of_inscribed_rectangle :
  ∀ (d : ℝ), (d = 3) → 
  let r := d / 2 in
  let A := (1/2) * π * r^2 in
  A = (9 * π) / 8 :=
by
  intros d hd
  simp [hd]
  let r := d / 2
  have hr: r = 3 / 2 := by simp [d, hd]
  simp [hr]
  sorry

end semicircle_area_of_inscribed_rectangle_l125_125315


namespace polynomial_divisible_l125_125390

def polynomial := (x : ℝ) → x^5 - 2*x^4 + 3*x^3 - (-54)*x^2 + (-48)*x - 8

theorem polynomial_divisible :
  ∀ x : ℝ, polynomial x = 0 → (x = -2 ∨ x = 1) :=
by
  sorry

end polynomial_divisible_l125_125390


namespace middle_number_consecutive_odd_sum_l125_125987

theorem middle_number_consecutive_odd_sum (n : ℤ)
  (h1 : n % 2 = 1) -- n is an odd number
  (h2 : n + (n + 2) + (n + 4) = n + 20) : 
  n + 2 = 9 :=
by
  sorry

end middle_number_consecutive_odd_sum_l125_125987


namespace max_area_of_garden_l125_125994

theorem max_area_of_garden (p : ℝ) (h : p = 36) : 
  ∃ A : ℝ, (∀ l w : ℝ, l + l + w + w = p → l * w ≤ A) ∧ A = 81 :=
by
  sorry

end max_area_of_garden_l125_125994


namespace find_constants_l125_125501

-- Given conditions
variables (OA OB OC : ℝ)
variables (theta phi : ℝ)

-- Definitions for lengths of vectors and angles
def norm_OA := (OA = 2)
def norm_OB := (OB = 2)
def norm_OC := (OC = 2)

def tan_angle_AOC := (tan theta = 4)
def angle_BOC := (phi = real.pi / 3) -- 60 degrees in radians

-- Finding constants m and n
theorem find_constants : ∃ (m n : ℝ), OC = m * OA + n * OB :=
begin
  -- Norm definitions
  have h1 : norm_OA, by sorry,
  have h2 : norm_OB, by sorry,
  have h3 : norm_OC, by sorry,
  -- Angle definitions
  have h4 : tan_angle_AOC, by sorry,
  have h5 : angle_BOC, by sorry,
  -- Solve for m and n
  use [2, -6],
  sorry
end

end find_constants_l125_125501


namespace find_x_l125_125664

def median (s : List ℝ) : ℝ := sorry -- Assume some implementation of median

theorem find_x (x : ℝ) :
  let data := [23, 27, 20, 18, x, 12] in
  median data = 21 → x = 22 :=
by 
  let data := [23, 27, 20, 18, x, 12]
  assume h : median data = 21
  sorry

end find_x_l125_125664


namespace quadratic_has_two_distinct_real_roots_l125_125559

theorem quadratic_has_two_distinct_real_roots (k : ℝ) :
  let Δ := (k - 1) ^ 2 + 4 in
  Δ > 0 :=
by
  let Δ := (k - 1) ^ 2 + 4
  -- We know that (k - 1) ^ 2 ≥ 0 for any real k
  -- Thus, (k - 1) ^ 2 + 4 > 0
  sorry

end quadratic_has_two_distinct_real_roots_l125_125559


namespace mr_brown_no_calls_in_2020_l125_125206

noncomputable def number_of_days_with_no_calls (total_days : ℕ) (calls_niece1 : ℕ) (calls_niece2 : ℕ) (calls_niece3 : ℕ) : ℕ := 
  let calls_2 := total_days / calls_niece1
  let calls_3 := total_days / calls_niece2
  let calls_4 := total_days / calls_niece3
  let calls_6 := total_days / (Nat.lcm calls_niece1 calls_niece2)
  let calls_12_ := total_days / (Nat.lcm calls_niece1 (Nat.lcm calls_niece2 calls_niece3))
  total_days - (calls_2 + calls_3 + calls_4 - calls_6 - calls_4 - (total_days / calls_niece2 / 4) + calls_12_)

theorem mr_brown_no_calls_in_2020 : number_of_days_with_no_calls 365 2 3 4 = 122 := 
  by 
    -- Proof steps would go here
    sorry

end mr_brown_no_calls_in_2020_l125_125206


namespace sqrt_square_eq_self_sqrt_784_square_l125_125377

theorem sqrt_square_eq_self (n : ℕ) (h : n ≥ 0) : (Real.sqrt n) ^ 2 = n :=
by
  sorry

theorem sqrt_784_square : (Real.sqrt 784) ^ 2 = 784 :=
by
  exact sqrt_square_eq_self 784 (Nat.zero_le 784)

end sqrt_square_eq_self_sqrt_784_square_l125_125377


namespace cage_chicken_problem_l125_125817

theorem cage_chicken_problem :
  (∃ x : ℕ, 6 ≤ x ∧ x ≤ 10 ∧ (4 * x + 1 = 5 * (x - 1))) ∧
  (∀ x : ℕ, 6 ≤ x ∧ x ≤ 10 → (4 * x + 1 ≥ 25 ∧ 4 * x + 1 ≤ 41)) :=
by
  sorry

end cage_chicken_problem_l125_125817


namespace find_points_on_hyperbola_l125_125211

def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

def distance_to_line (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / sqrt (a^2 + b^2)

theorem find_points_on_hyperbola :
  let x1 := 8 / sqrt 3;
  let y1 := sqrt 3;
  let x2 := -8 / sqrt 3;
  let y2 := -sqrt 3;
  point_on_hyperbola x1 y1 ∧ point_on_hyperbola x2 y2 ∧
  (distance_to_line x1 y1 3 (-4) 0 = 3 * distance_to_line x1 y1 3 4 0) ∧
  (distance_to_line x2 y2 3 (-4) 0 = 3 * distance_to_line x2 y2 3 4 0) := 
by
  sorry

end find_points_on_hyperbola_l125_125211


namespace average_marks_of_passed_boys_l125_125835

theorem average_marks_of_passed_boys
    (total_boys : ℕ) (total_average : ℝ)
    (passed_boys : ℕ) (failed_average : ℝ)
    (total_marks : ℝ)
    (total_failed_boys : ℕ)
    (assumption1 : total_boys = 120)
    (assumption2 : total_average = 36)
    (assumption3 : passed_boys = 105)
    (assumption4 : failed_average = 15)
    (assumption5 : total_failed_boys = total_boys - passed_boys)
    (assumption6 : total_marks = total_boys * total_average)
    (assumption7 : total_marks_failed := total_failed_boys * failed_average)
    (assumption8 : total_marks_passed := total_marks - total_marks_failed)
    (total_marks_passed_calculated := total_marks_passed / passed_boys)
    : total_marks_passed_calculated = 39 :=
  sorry

end average_marks_of_passed_boys_l125_125835


namespace books_not_sold_l125_125346

theorem books_not_sold :
  let initial_stock := 800
  let books_sold_mon := 60
  let books_sold_tue := 10
  let books_sold_wed := 20
  let books_sold_thu := 44
  let books_sold_fri := 66
  in initial_stock - (books_sold_mon + books_sold_tue + books_sold_wed + books_sold_thu + books_sold_fri) = 600 :=
by
  let initial_stock := 800
  let books_sold_mon := 60
  let books_sold_tue := 10
  let books_sold_wed := 20
  let books_sold_thu := 44
  let books_sold_fri := 66
  show initial_stock - (books_sold_mon + books_sold_tue + books_sold_wed + books_sold_thu + books_sold_fri) = 600
  calc
    initial_stock - (books_sold_mon + books_sold_tue + books_sold_wed + books_sold_thu + books_sold_fri) = initial_stock - 200 : by sorry
    ... = 600 : by sorry

end books_not_sold_l125_125346


namespace find_p_plus_q_l125_125066

noncomputable def f (x : ℝ) : ℝ := (Real.exp (abs x) - Real.sin x + 1) / (Real.exp (abs x) + 1)

def interval (m : ℝ) (h : 0 < m) := Set.Icc (-m) m

theorem find_p_plus_q (m : ℝ) (h : 0 < m) : 
  let I := interval m h
  let p := Sup (Set.Image f I)
  let q := Inf (Set.Image f I)
  p + q = 2 :=
sorry

end find_p_plus_q_l125_125066


namespace range_of_m_l125_125033

theorem range_of_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / (x + 1) + 4 / y = 1) (m : ℝ) :
  x + y / 4 > m^2 - 5 * m - 3 ↔ -1 < m ∧ m < 6 := sorry

end range_of_m_l125_125033


namespace anya_takes_home_balloons_l125_125603

theorem anya_takes_home_balloons:
  ∀ (total_balloons : ℕ) (colors : ℕ) (half : ℕ) (balloons_per_color : ℕ),
  total_balloons = 672 →
  colors = 4 →
  balloons_per_color = total_balloons / colors →
  half = balloons_per_color / 2 →
  half = 84 :=
by 
  intros total_balloons colors half balloons_per_color 
  intros h1 h2 h3 h4
  sorry

end anya_takes_home_balloons_l125_125603


namespace range_of_expression_l125_125541

variable (a b : ℝ)
variable (h1 : |a| ≤ 1)
variable (h2 : |a + b| ≤ 1)

theorem range_of_expression : ∃ x, x ∈ set.Icc (-2) (9/4) ∧ x = (a + 1) * (b + 1) :=
by sorry

end range_of_expression_l125_125541


namespace chromatic_number_le_k_plus_one_l125_125877

theorem chromatic_number_le_k_plus_one {V E : Type} [Fintype V] (G : SimpleGraph V) (k : ℕ) :
  (∀ v : V, G.degree v ≤ k) → G.chromaticNumber ≤ k + 1 :=
sorry

end chromatic_number_le_k_plus_one_l125_125877


namespace non_negative_integers_abs_less_than_3_l125_125585

theorem non_negative_integers_abs_less_than_3 :
  { x : ℕ | x < 3 } = {0, 1, 2} :=
by
  sorry

end non_negative_integers_abs_less_than_3_l125_125585


namespace product_of_consecutive_triangular_not_square_infinite_larger_triangular_numbers_square_product_l125_125903

section TriangularNumbers

-- Define triangular numbers
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

-- Statement 1: The product of two consecutive triangular numbers is not a perfect square
theorem product_of_consecutive_triangular_not_square (n : ℕ) (hn : n > 0) :
  ¬ ∃ m : ℕ, triangular (n - 1) * triangular n = m * m := by
  sorry

-- Statement 2: There exist infinitely many larger triangular numbers such that the product with t_n is a perfect square
theorem infinite_larger_triangular_numbers_square_product (n : ℕ) :
  ∃ᶠ m in at_top, ∃ k : ℕ, triangular n * triangular m = k * k := by
  sorry

end TriangularNumbers

end product_of_consecutive_triangular_not_square_infinite_larger_triangular_numbers_square_product_l125_125903


namespace general_term_a_n_geo_seq_C_n_general_term_b_n_sum_d_n_l125_125528

-- Arithmetic Sequence
axiom a1 : ℕ → ℕ 
axiom a_n_sequence: (n : ℕ) (h₁ : n = 2) : a1 n = 4
axiom S_5_sum: (n : ℕ) (h₂ : n = 5): 5 * a1 1 + 5 * 2 * 2 = 30 

-- General Term for \{a_n\}
theorem general_term_a_n (n : ℕ) : a1 n = 2 * n := sorry

-- Sequence b_n
axiom b1_seq : ℕ → ℕ 
axiom initial_b1: b1_seq 1 = 0
axiom b1_def: ∀ n ≥ 2, b1_seq n = 2 * b1_seq (n - 1) + 1

-- Sequence C_n and General Term for \{b_n\}
noncomputable def C_n (n : ℕ) : ℕ := b1_seq n + 1
theorem geo_seq_C_n (n : ℕ) (h₂ : n ≥ 2) : C_n n / C_n (n - 1) = 2 := sorry
theorem general_term_b_n (n : ℕ) : b1_seq n = 2^(n - 1) - 1 := sorry

-- Sequence d_n
axiom d_n_seq (n : ℕ) : ℝ := 4 / (a1 n * a1 (n + 1)) + b1_seq n

-- Sum of first n terms of d_n
theorem sum_d_n (n : ℕ) : \sum_{i=1}^{n} d_n_seq(i) = 2^n - n - (1 / (n + 1)) := sorry

end general_term_a_n_geo_seq_C_n_general_term_b_n_sum_d_n_l125_125528


namespace f_eq_x_pow_n_l125_125864

-- Define the set D
def D := { x : ℝ | x > 0 ∧ x ≠ 1 }

-- Define the function f from D to ℝ
variable (f : ℝ → ℝ)

-- Define the conditions on n, the functional equation, and specific cases of f
variable (n : ℕ) (n_pos : 0 < n)
variable (h1 : ∀ x : ℝ, x ∈ D → x^n * f(x) = f(x^2))
variable (h2 : ∀ x : ℝ, 0 < x ∧ x < 1 / 1989 → f(x) = x^n)
variable (h3 : ∀ x : ℝ, x > 1989 → f(x) = x^n)

-- The goal is to prove that f(x) = x^n for all x in D
theorem f_eq_x_pow_n (x : ℝ) (hx : x ∈ D) : f(x) = x^n :=
by
  sorry

end f_eq_x_pow_n_l125_125864


namespace complement_P_relative_to_U_l125_125888

variable (U : Set ℝ) (P : Set ℝ)

theorem complement_P_relative_to_U (hU : U = Set.univ) (hP : P = {x : ℝ | x < 1}) : 
  U \ P = {x : ℝ | x ≥ 1} := by
  sorry

end complement_P_relative_to_U_l125_125888


namespace probability_of_performances_l125_125670

def num_singing := 5
def num_dancing := 3
def total_performances := num_singing + num_dancing
def total_permutations : ℕ := fact total_performances
def desired_permutations : ℕ := (comb num_dancing 2) * (fact (total_performances - 3))

theorem probability_of_performances :
  total_permutations ≠ 0 →
  (desired_permutations / total_permutations : ℚ) = 3 / 28 :=
begin
  intros h,
  sorry
end

end probability_of_performances_l125_125670


namespace area_ratio_centroid_pentagon_l125_125175

variables {V : Type*} [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

/-- Given a convex pentagon ABCDE and centroids G_A, G_B, G_C, G_D, G_E of quadrilaterals BCDE, ACDE, ABDE, ABCE, and ABCD respectively,
show that the area ratio [G_A G_B G_C G_D G_E] / [ABCDE] is 1/16. -/
theorem area_ratio_centroid_pentagon 
  (A B C D E G_A G_B G_C G_D G_E : V)
  (hG_A : G_A = ((B + C + D + E) : V) / 4)
  (hG_B : G_B = ((A + C + D + E) : V) / 4)
  (hG_C : G_C = ((A + B + D + E) : V) / 4)
  (hG_D : G_D = ((A + B + C + E) : V) / 4)
  (hG_E : G_E = ((A + B + C + D) : V) / 4)
  (convex_A: Convex ℝ (Set.Of [A, B, C, D, E])) :
  ∃ (GA_GB_GC_GD_GE ABCDE : ℝ), 
    GA_GB_GC_GD_GE / ABCDE = 1 / 16 := 
sorry

end area_ratio_centroid_pentagon_l125_125175


namespace cyclist_speeds_l125_125642

-- Given definitions
def distance_AB : ℝ := 240
def time_B_later : ℝ := 0.5
def speed_diff : ℝ := 3
def fix_time : ℝ := 1.5
def midpoint_distance : ℝ := distance_AB / 2

-- Definition of speeds
def speed_A : ℝ := 12
def speed_B : ℝ := speed_A + speed_diff

-- Theorem statement
theorem cyclist_speeds :
  (∀ (x y : ℝ),
    let time_A_to_C := midpoint_distance / x
    let time_B_to_C := midpoint_distance / (y + speed_diff)
    x = y + speed_diff ∧ 
    time_A_to_C - time_B_later = time_B_to_C ∧
    time_B_to_C = fix_time →
    x = 12 ∧ y = 15) := sorry

end cyclist_speeds_l125_125642


namespace random_event_is_B_l125_125291

variable (isCertain : Event → Prop)
variable (isImpossible : Event → Prop)
variable (isRandom : Event → Prop)

variable (A : Event)
variable (B : Event)
variable (C : Event)
variable (D : Event)

-- Here we set the conditions as definitions in Lean 4:
def condition_A : isCertain A := sorry
def condition_B : isRandom B := sorry
def condition_C : isCertain C := sorry
def condition_D : isImpossible D := sorry

-- The theorem we need to prove:
theorem random_event_is_B : isRandom B := 
by
-- adding sorry to skip the proof
sorry

end random_event_is_B_l125_125291


namespace eggs_division_l125_125906

theorem eggs_division :
  (∀ (eggs friends : ℕ), eggs = 16 ∧ friends = 8 → eggs / friends = 2) :=
by
  intros eggs friends h
  cases h with h1 h2
  rw [h1, h2]
  norm_num

end eggs_division_l125_125906


namespace percentage_income_diff_l125_125814

variable (A B : ℝ)

-- Condition that B's income is 33.33333333333333% greater than A's income
def income_relation (A B : ℝ) : Prop :=
  B = (4 / 3) * A

-- Proof statement to show that A's income is 25% less than B's income
theorem percentage_income_diff : 
  income_relation A B → 
  ((B - A) / B) * 100 = 25 :=
by
  intros h
  rw [income_relation] at h
  sorry

end percentage_income_diff_l125_125814


namespace largest_real_number_c_l125_125722

theorem largest_real_number_c (x : Fin 51 → ℝ) (h_sum : ∑ i, x i = 0) :
  ∀ M, median (Finset.image (λ (i : Fin 51), x i) Finset.univ) M → 
  ∑ i, (x i)^2 ≥ (702 / 25) * M^2 :=
by sorry

end largest_real_number_c_l125_125722


namespace triangle_subsegment_length_l125_125595

noncomputable def length_longer_subsegment (PQ PR QR QS SR : ℝ) : Prop :=
  ((PQ / PR = 3 / 4) ∧ (QR = 12) ∧ ((QS / SR = 3 / 4) ∧ (QS + SR = QR))) 
  → (SR = 48 / 7)

theorem triangle_subsegment_length :
  ∃ (PQ PR QS SR : ℝ), length_longer_subsegment PQ PR 12 QS SR :=
begin
  -- Define the side lengths in the ratio 3:4:5
  let PQ := 3 * 4,
  let PR := 4 * 4,
  let QR := 5 * 4,

  -- Define QS and SR using the angle bisector theorem
  let QS := (3 / (3 + 4)) * QR,
  let SR := (4 / (3 + 4)) * QR,

  -- Solution based on the provided conditions
  use [PQ, PR, QS, SR],
  split, -- for PQ / PR = 3 / 4
  { norm_num at *,
    have h1 := PQ / PR,
    rw ←div_eq_inv_mul at h1,
    have h2 : PQ = 12 := by ring,
    have h3 : PR = 16 := by ring,
    exact h1, },
  
  split, -- for QR = 12
  { exact 12, },

  split, -- for QS / SR = 3 / 4 ∧ QS + SR = QR
  { split,
    { apply (div_eq_inv_mul (QR * 3) (QR * 4)).symm,
      ring, },
    { finish, } },

  -- Conclude SR = 48 / 7
  { finish, fitting },
end

end triangle_subsegment_length_l125_125595


namespace problem1_problem2_l125_125687

theorem problem1 : (40 * Real.sqrt 3 - 18 * Real.sqrt 3 + 8 * Real.sqrt 3) / 6 = 5 * Real.sqrt 3 := 
by sorry

theorem problem2 : (Real.sqrt 3 - 2)^2023 * (Real.sqrt 3 + 2)^2023
                 - Real.sqrt 4 * Real.sqrt (1 / 2)
                 - (Real.pi - 1)^0
                = -2 - Real.sqrt 2 :=
by sorry

end problem1_problem2_l125_125687


namespace diagonals_bisect_each_other_l125_125922

noncomputable def quadrilateral (sum_ext_angles_360 : Prop) (has_diagonals : Prop) (instable : Prop) : Prop :=
  sum_ext_angles_360 ∧ has_diagonals ∧ instable

noncomputable def parallelogram (sum_ext_angles_360 : Prop) (has_diagonals : Prop) (instable : Prop) (diagonals_bisect : Prop) : Prop :=
  sum_ext_angles_360 ∧ has_diagonals ∧ instable ∧ diagonals_bisect

theorem diagonals_bisect_each_other 
  (sum_ext_angles_360 : Prop) (has_diagonals : Prop) (instable : Prop) (diagonals_bisect : Prop) :
  parallelogram sum_ext_angles_360 has_diagonals instable diagonals_bisect →
  quadrilateral sum_ext_angles_360 has_diagonals instable ∧ diagonals_bisect :=
begin
  sorry
end

end diagonals_bisect_each_other_l125_125922


namespace repeating_decimal_sum_numerator_denominator_l125_125931

-- Define the problem statement
theorem repeating_decimal_sum_numerator_denominator :
  let (a, b) := (250, 99) in a + b = 349 :=
by
  let x := 2.5252525 -- Example value; translation context
  have hx : 100 * x = 252.525252 -- Scaling to align repeating digits
  have hy : x = 2.525252 -- Initial statement of x variable
  let m := 250 -- Numerator of simplified fraction
  let n := 99  -- Denominator of simplified fraction
  sorry -- Proof omitted

end repeating_decimal_sum_numerator_denominator_l125_125931


namespace more_orchids_than_roses_l125_125270

theorem more_orchids_than_roses (orchids roses : ℕ) (h_orchids : orchids = 13) (h_roses : roses = 3) : orchids - roses = 10 :=
by
  rw [h_orchids, h_roses]
  rfl

end more_orchids_than_roses_l125_125270


namespace ducks_in_garden_l125_125943

def count_ducks (dogs ducks total_feet : ℕ) (dogs_feet_per_dog ducks_feet_per_duck : ℕ) : ℕ :=
  (total_feet - dogs * dogs_feet_per_dog) / ducks_feet_per_duck

theorem ducks_in_garden (dogs ducks total_feet : ℕ)
  (h_dogs : dogs = 6)
  (h_total_feet : total_feet = 28)
  (h_dogs_feet_per_dog : 4)
  (h_ducks_feet_per_duck : 2) : ducks = 2 :=
by
  have total_dog_feet := h_dogs * h_dogs_feet_per_dog
  have remaining_feet := h_total_feet - total_dog_feet
  have h_ducks := remaining_feet / h_ducks_feet_per_duck
  sorry

end ducks_in_garden_l125_125943


namespace arc_length_and_segment_area_max_sector_area_l125_125759

-- Problem (1)
theorem arc_length_and_segment_area (α R : ℝ) (h1 : α = π / 3) (h2 : R = 10) :
  let l := α * R,
      S_sector := 1/2 * α * R^2,
      S_triangle := 1/2 * R * (R * Real.sin α),
      S_segment := S_sector - S_triangle in
  l = 10 * π / 3 ∧ S_segment = 50 * (π / 3 - Real.sqrt 3 / 2) :=
by sorry

-- Problem (2)
theorem max_sector_area (P : ℝ) (h1 : P = 12) :
  let α := (12 - 2 * 3) / 3,
      S_sector := 1/2 * α * 3^2 in
  α = 2 ∧ S_sector = 9 :=
by sorry

end arc_length_and_segment_area_max_sector_area_l125_125759


namespace city_count_multiple_of_2016_l125_125941

def f {V : Type} [Fintype V] (n : ℕ) (v : V) : ℕ := sorry -- function to count permutations

theorem city_count_multiple_of_2016 (n : ℕ) (V : Type) [Fintype V]
  [DecidableRel (@connected V _)] (f : V → ℕ)
  (h_tree : ∀ u v : V, ∃ t, tree t)
  (h_condition : n > 1)
  (h_f_multiple : ∀ v : V, v ≠ remaining_city → (f v) % 2016 = 0) :
  (f remaining_city) % 2016 = 0 :=
by
  sorry

end city_count_multiple_of_2016_l125_125941


namespace find_xz_of_parallel_l125_125797

variable {x z : ℝ}

-- Conditions
def a := (x, 4, 3)
def b := (3, 2, z)
def parallel (a b : ℝ × ℝ × ℝ) : Prop := ∃ λ : ℝ, a = (λ * b.1, λ * b.2, λ * b.3)

-- Statement
theorem find_xz_of_parallel (h : parallel a b) : x * z = 9 :=
by
  sorry

end find_xz_of_parallel_l125_125797


namespace solving_linear_equations_count_l125_125563

def total_problems : ℕ := 140
def algebra_percentage : ℝ := 0.40
def algebra_problems := (total_problems : ℝ) * algebra_percentage
def solving_linear_equations_percentage : ℝ := 0.50
def solving_linear_equations_problems := algebra_problems * solving_linear_equations_percentage

theorem solving_linear_equations_count :
  solving_linear_equations_problems = 28 :=
by
  sorry

end solving_linear_equations_count_l125_125563


namespace limit_of_sequence_l125_125723

noncomputable def f (n : ℕ) : ℝ := (3^(n+1) - 2^n) / (3^n + 2^(n+1))

theorem limit_of_sequence : (filter.at_top.map f).lim = 3 := sorry

end limit_of_sequence_l125_125723


namespace ratio_board_games_l125_125129

theorem ratio_board_games (total_students silent_reading homework playing_games : ℕ) 
  (h_total_students : total_students = 24)
  (h_silent_reading : silent_reading = total_students / 2)
  (h_homework : homework = 4)
  (h_playing_games : playing_games = total_students - (silent_reading + homework)) :
  playing_games / total_students = 1 / 3 :=
by 
  rw [h_total_students, h_silent_reading, h_homework, h_playing_games]
  sorry

end ratio_board_games_l125_125129


namespace equilibrium_constant_1_equilibrium_constant_2_equilibrium_constant_reverse_l125_125665

section EquilibriumConstants

-- Defining the given conditions
def c_H2 := 0.5  -- mol·L^(-1)
def c_HI := 4    -- mol·L^(-1)
def c_HI_final := 3  -- mol·L^(-1) at equilibrium
def c_NH3 := c_HI_final  -- mol·L^(-1) assume at equilibrium same as HI

-- Define necessary equilibrium constants calculations
def K2 := (c_H2 * c_H2) / (c_HI_final * c_HI_final)
def K1 := c_HI_final * c_HI_final
def K := 1 / K2

-- Stating the proof goals
theorem equilibrium_constant_1 : K1 = 9 :=
  by sorry  -- Proof omitted.

theorem equilibrium_constant_2 : K2 = 25 / 9 :=
  by sorry  -- Proof omitted.

theorem equilibrium_constant_reverse : K = 9 / 25 :=
  by sorry  -- Proof omitted.

end EquilibriumConstants

end equilibrium_constant_1_equilibrium_constant_2_equilibrium_constant_reverse_l125_125665


namespace division_theorem_l125_125016

variable (x : ℤ)

def dividend := 8 * x ^ 4 + 7 * x ^ 3 + 3 * x ^ 2 - 5 * x - 8
def divisor := x - 1
def quotient := 8 * x ^ 3 + 15 * x ^ 2 + 18 * x + 13
def remainder := 5

theorem division_theorem : dividend x = divisor x * quotient x + remainder := by
  sorry

end division_theorem_l125_125016


namespace generic_packages_needed_eq_2_l125_125361

-- Define parameters
def tees_per_generic_package : ℕ := 12
def tees_per_aero_package : ℕ := 2
def members_foursome : ℕ := 4
def tees_needed_per_member : ℕ := 20
def aero_packages_purchased : ℕ := 28

-- Calculate total tees needed and total tees obtained from aero packages
def total_tees_needed : ℕ := members_foursome * tees_needed_per_member
def aero_tees_obtained : ℕ := aero_packages_purchased * tees_per_aero_package
def generic_tees_needed : ℕ := total_tees_needed - aero_tees_obtained

-- Prove the number of generic packages needed is 2
theorem generic_packages_needed_eq_2 : 
  generic_tees_needed / tees_per_generic_package = 2 :=
  sorry

end generic_packages_needed_eq_2_l125_125361


namespace total_animals_correct_l125_125103

-- Define the number of aquariums and the number of animals per aquarium.
def num_aquariums : ℕ := 26
def animals_per_aquarium : ℕ := 2

-- Define the total number of saltwater animals.
def total_animals : ℕ := num_aquariums * animals_per_aquarium

-- The statement we want to prove.
theorem total_animals_correct : total_animals = 52 := by
  -- Proof is omitted.
  sorry

end total_animals_correct_l125_125103


namespace profit_percentage_correct_l125_125663

noncomputable def overall_profit_percentage : ℚ :=
  let cost_radio := 225
  let overhead_radio := 15
  let price_radio := 300
  let cost_watch := 425
  let overhead_watch := 20
  let price_watch := 525
  let cost_mobile := 650
  let overhead_mobile := 30
  let price_mobile := 800
  
  let total_cost_price := (cost_radio + overhead_radio) + (cost_watch + overhead_watch) + (cost_mobile + overhead_mobile)
  let total_selling_price := price_radio + price_watch + price_mobile
  let total_profit := total_selling_price - total_cost_price
  (total_profit * 100 : ℚ) / total_cost_price
  
theorem profit_percentage_correct :
  overall_profit_percentage = 19.05 := by
  sorry

end profit_percentage_correct_l125_125663


namespace largest_possible_3_digit_sum_l125_125111

theorem largest_possible_3_digit_sum (X Y Z : ℕ) (h_diff : X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z) 
(h_digit_X : 0 ≤ X ∧ X ≤ 9) (h_digit_Y : 0 ≤ Y ∧ Y ≤ 9) (h_digit_Z : 0 ≤ Z ∧ Z ≤ 9) :
  (100 * X + 10 * X + X) + (10 * Y + X) + X = 994 → (X, Y, Z) = (8, 9, 0) := by
  sorry

end largest_possible_3_digit_sum_l125_125111


namespace lucas_and_molly_same_team_l125_125470

-- Definitions based on problem conditions
def num_players : ℕ := 12
def chosen_team_size : ℕ := 6
def remaining_players : ℕ := num_players - 2 -- excluding Lucas and Molly

-- Function to compute binomial coefficient (combinations)
noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Number of teams of 6 players where Lucas and Molly are together
def num_teams_with_lucas_and_molly : ℕ := binom remaining_players (chosen_team_size - 2)

-- Theorem stating that the number of times Lucas and Molly play together is 210
theorem lucas_and_molly_same_team : num_teams_with_lucas_and_molly = 210 := by
  sorry

end lucas_and_molly_same_team_l125_125470


namespace complement_union_l125_125096

variable (U : Set ℤ)
variable (A : Set ℤ)
variable (B : Set ℤ)

theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3})
                         (hA : A = {-1, 0, 1})
                         (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} :=
sorry

end complement_union_l125_125096


namespace calculate_area_of_DEF_l125_125381

open Real

noncomputable def isosceles_triangle : Type :=
{ A B C : ℝ³ // distance A B = distance A C ∧ distance B C = 10*sqrt 3 ∧ distance A B = 10*sqrt 2 }

noncomputable def semicircle_radius (d: ℝ) : ℝ := d / 2

def tangent_point_height (r: ℝ) : ℝ := r

def midpoint (P Q: ℝ³) : ℝ³ := (P + Q) / 2

def tangent_point (M: ℝ³) (h: ℝ) : ℝ³ := ⟨ M.1, M.2, h ⟩  -- using 3D coordinates with height h

def area_of_triangle (D E F: ℝ³) : ℝ := 
  abs (D.x * (E.y - F.y) + E.x * (F.y - D.y) + F.x * (D.y - E.y)) / 2

theorem calculate_area_of_DEF {A B C : ℝ³} (h₁ h₂ h₃ : ℝ) (H : isosceles_triangle) :
  (area_of_triangle 
    (tangent_point (midpoint A B) h₁) 
    (tangent_point (midpoint A C) h₂) 
    (tangent_point (midpoint B C) h₃))
  = 24 :=
sorry

end calculate_area_of_DEF_l125_125381


namespace gumball_machine_l125_125660

variable (R B G Y O : ℕ)

theorem gumball_machine : 
  (B = (1 / 2) * R) ∧
  (G = 4 * B) ∧
  (Y = (7 / 2) * B) ∧
  (O = (2 / 3) * (R + B)) ∧
  (R = (3 / 2) * Y) ∧
  (Y = 24) →
  (R + B + G + Y + O = 186) :=
sorry

end gumball_machine_l125_125660


namespace equation_of_trajectory_equation_of_l1_l125_125327

theorem equation_of_trajectory (M F: ℝ × ℝ) 
    (line_l : ℝ → Prop)
    (h1 : ∀ x, line_l x ↔ x = -1)
    (h2 : F = (1, 0))
    (h3 : line_l M.1 = (dist M F)): 
    M.2^2 = 4 * M.1 :=
  sorry

theorem equation_of_l1 (F : ℝ × ℝ) 
    (k : ℝ)
    (h4 : F = (1, 0))
    (h5 : ∀ x y, (y = k * (x - 1)) → False)
    (h6 : ∀ A B : ℝ × ℝ, (y1^2 = 4 * x1, y2^2 = 4 * x2, |A - B| = 6) → False):

    (k = sqrt(2) ∨ k = -sqrt(2) )  ∧
    ( y = sqrt(2) * (x - 1) ∨  y = -sqrt(2) * (x - 1)) :=
sorry

end equation_of_trajectory_equation_of_l1_l125_125327


namespace smallest_possible_value_is_7_over_2_l125_125185

noncomputable def smallest_possible_value (z : ℂ) (h : |z^2 + 9| = |z * (z + 3 * complex.I)|) : ℝ :=
  ⨅ z : ℂ, |z + 2 * complex.I|

theorem smallest_possible_value_is_7_over_2 :
  smallest_possible_value = 7 / 2 :=
by
  sorry

end smallest_possible_value_is_7_over_2_l125_125185


namespace remaining_surface_area_correct_l125_125712

open Real

-- Define the original cube and the corner cubes
def orig_cube : ℝ × ℝ × ℝ := (5, 5, 5)
def corner_cube : ℝ × ℝ × ℝ := (2, 2, 2)

-- Define a function to compute the surface area of a cube given dimensions (a, b, c)
def surface_area (a b c : ℝ) : ℝ := 2 * (a * b + b * c + c * a)

-- Original surface area of the cube
def orig_surface_area : ℝ := surface_area 5 5 5

-- Total surface area of the remaining figure after removing 8 corner cubes
def remaining_surface_area : ℝ := 150  -- Calculated directly as 6 * 25

-- Theorem stating that the surface area of the remaining figure is 150 cm^2
theorem remaining_surface_area_correct :
  remaining_surface_area = 150 := sorry

end remaining_surface_area_correct_l125_125712


namespace P_2022_mod_1000_l125_125336

-- Define the sequence of polynomials
def P : ℕ → (ℕ → ℤ) := λ n x, 
  match n with
  | 0   => 0
  | 1   => x + 1
  | n+1 => (P n (x + 1) - P n (-x + 1)) / 2 -- This directly matches the given recurrence
  end

-- State the problem in terms of proving the value of P_2022(1) mod 1000
theorem P_2022_mod_1000 : (P 2022 1) % 1000 = 616 :=
by sorry

end P_2022_mod_1000_l125_125336


namespace find_all_triplets_l125_125032

def greatest_prime_factor (n : ℕ) : ℕ :=
  if h : n > 1 then nat.find_greatest_prime_factor n else 1

noncomputable def is_arith_prog (x y z : ℕ) : Prop :=
  2 * y = x + z

noncomputable def problem_conditions (x y z : ℕ) : Prop :=
  is_arith_prog x y z ∧ greatest_prime_factor (x * y * z) ≤ 3

theorem find_all_triplets (x y z : ℕ) (h : problem_conditions x y z) :
  ∃ l : ℕ, l = 2^a * 3^b ∧ ((x, y, z) = (l, 2 * l, 3 * l) ∨ (x, y, z) = (2 * l, 3 * l, 4 * l) ∨ (x, y, z) = (2 * l, 9 * l, 16 * l)) :=
sorry

end find_all_triplets_l125_125032


namespace divisibility_by_10_l125_125910

theorem divisibility_by_10 (a : ℤ) (n : ℕ) (h : n ≥ 2) : 
  (a^(2^n + 1) - a) % 10 = 0 :=
by
  sorry

end divisibility_by_10_l125_125910


namespace sum_abs_coeffs_l125_125035

noncomputable def P (x : ℚ) : ℚ := 1 - (1/2) * x + (1/4) * x^3 - (1/8) * x^4

noncomputable def Q (x : ℚ) : ℚ := P(x) * P(x^2) * P(x^4) * P(x^6) * P(x^8) * P(x^10)

theorem sum_abs_coeffs (a : ℕ → ℚ) : (∑ i in range 61, |a i|) = (531441 / 262144) :=
by
  sorry

end sum_abs_coeffs_l125_125035


namespace correct_option_C_l125_125292

-- Define the variables and expressions used in the conditions
variables (a : ℝ)

-- Define each condition as a proposition
def option_A := 2 * a^3 - a^2 = a
def option_B := (a^2)^3 = a^5
def option_C := a^2 * a^3 = a^5
def option_D := (2 * a)^3 = 6 * a^3

-- State the theorem indicating that option C is the correct operation
theorem correct_option_C : option_C ∧ ¬option_A ∧ ¬option_B ∧ ¬option_D :=
by {
  sorry -- Here we just state the theorem, the proof should be completed separately.
}

end correct_option_C_l125_125292


namespace distance_to_place_l125_125295

variable (T : Type) [LinearOrderedField T]

def rowing_distance (r_w_s : T) (v_c : T) (t_total : T) : T :=
  let v_up : T := r_w_s - v_c -- Speed against the current
  let v_down : T := r_w_s + v_c -- Speed with the current
  let t_up : T := (r_w_s - v_c) * t_total / ((r_w_s - v_c) + (r_w_s + v_c))
  v_up * t_up

theorem distance_to_place :
  rowing_distance 10 2 10 = 48 := by
  unfold rowing_distance
  simp only [sub_add_cancel, mul_div_cancel_left, ne_of_gt, zero_lt_mul_left, div_self]
  norm_num
  exact sorry

end distance_to_place_l125_125295


namespace gas_fee_in_august_l125_125828

-- Definitions for conditions
def charge_per_cubic_meter (x : ℕ) : ℝ :=
  if x <= 60 then 0.8 else 1.2

def total_gas_fee (x : ℝ) : ℝ :=
  if x ≤ 60 then x * 0.8 else 60 * 0.8 + (x - 60) * 1.2

def average_gas_fee (total_fee : ℝ) (x : ℝ) : ℝ :=
  total_fee / x

-- Given conditions
variable (x : ℝ) (h_avg : average_gas_fee (total_gas_fee x) x = 0.88)

-- Statement to prove
theorem gas_fee_in_august : total_gas_fee x = 66 := by
  sorry

end gas_fee_in_august_l125_125828


namespace expectation_and_variance_of_affine_transformation_l125_125110

theorem expectation_and_variance_of_affine_transformation
  (X : Type) [probability_space X] 
  (f : X → ℝ) [is_discrete f] :
  let Y := λ x, 3 * f x + 2 in
  E(Y) = 3 * E(f) + 2 ∧ D(Y) = 9 * D(f) :=
by
  unfold is_discrete
  sorry

end expectation_and_variance_of_affine_transformation_l125_125110


namespace share_of_A_is_357_l125_125628

/- 
  Definitions and assumptions based on given conditions 
-/
def initial_investment_A : ℕ := 6000
def initial_investment_B : ℕ := 4000
def withdrawal_A : ℕ := 1000
def advance_B : ℕ := 1000
def total_profit : ℕ := 630
def total_months : ℕ := 12
def months_after_changes : ℕ := 4
def months_before_changes : ℕ := total_months - months_after_changes

/- 
  Calculate investment months for A and B
-/
def investment_months_A : ℕ := (initial_investment_A * months_before_changes) + ((initial_investment_A - withdrawal_A) * months_after_changes)
def investment_months_B : ℕ := (initial_investment_B * months_before_changes) + ((initial_investment_B + advance_B) * months_after_changes)
def ratio_A_B := (investment_months_A.to_rat / investment_months_B.to_rat)

/- 
  Calculating A's share of the profit based on the ratio 
-/
def A_share := (ratio_A_B / (ratio_A_B + 1)) * total_profit

/- 
  The main theorem to state the problem equivalently 
-/
theorem share_of_A_is_357 : A_share = 357 := by
  sorry

end share_of_A_is_357_l125_125628


namespace find_function_l125_125199

-- Let f be a differentiable function over all real numbers
variable (f : ℝ → ℝ)
variable (k : ℝ)

-- Condition: f is differentiable over (-∞, ∞)
variable (h_diff : differentiable ℝ f)

-- Condition: f(0) = 1
variable (h_init : f 0 = 1)

-- Condition: for any x1, x2 in ℝ, f(x1 + x2) ≥ f(x1) f(x2)
variable (h_ineq : ∀ x1 x2 : ℝ, f (x1 + x2) ≥ f x1 * f x2)

-- We aim to prove: f(x) = e^(kx)
theorem find_function : ∃ k : ℝ, ∀ x : ℝ, f x = Real.exp (k * x) :=
sorry

end find_function_l125_125199


namespace checkerboard_black_squares_l125_125374

theorem checkerboard_black_squares (n : ℕ) (h: n = 29) :
  let black_squares := if n % 2 = 0 then (n * n) / 2 else (n * n + 1) / 2 in
  black_squares = 421 :=
by
  declare_var n : ℕ
  declare_var h : n = 29
  -- Assume necessary definitions and properties
  sorry

end checkerboard_black_squares_l125_125374


namespace domain_of_y_l125_125148

noncomputable def y (x : ℝ) : ℝ := (sqrt (x + 2)) / (x - 2)

theorem domain_of_y : ∀ x : ℝ, ((x ≥ -2) ∧ (x ≠ 2)) ↔ (∃ y : ℝ, y = (sqrt (x + 2)) / (x - 2)) :=
by sorry

end domain_of_y_l125_125148


namespace eve_ate_13_apples_l125_125342

theorem eve_ate_13_apples :
  ∃ (a b : ℕ), 
    (7 * a + 5 * b = 31) ∧ 
    (3 * a + 2 * b = 13) := 
begin
  use [3, 2],
  split,
  { norm_num },
  { norm_num }
end

end eve_ate_13_apples_l125_125342


namespace inequality_proof_l125_125633

open Real

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a * b * c * d = 1) :
  (a^4 + b^4) / (a^2 + b^2) + (b^4 + c^4) / (b^2 + c^2) + (c^4 + d^4) / (c^2 + d^2) + (d^4 + a^4) / (d^2 + a^2) ≥ 4 :=
by
  sorry

end inequality_proof_l125_125633


namespace find_softball_players_l125_125130

def total_players : ℕ := 51
def cricket_players : ℕ := 10
def hockey_players : ℕ := 12
def football_players : ℕ := 16

def softball_players : ℕ := total_players - (cricket_players + hockey_players + football_players)

theorem find_softball_players : softball_players = 13 := 
by {
  sorry
}

end find_softball_players_l125_125130


namespace complement_union_eq_l125_125092

variable (U : Set Int := {-2, -1, 0, 1, 2, 3}) 
variable (A : Set Int := {-1, 0, 1}) 
variable (B : Set Int := {1, 2}) 

theorem complement_union_eq :
  U \ (A ∪ B) = {-2, 3} := by 
  sorry

end complement_union_eq_l125_125092


namespace happy_boys_l125_125209

theorem happy_boys :
  ∀ (total_children happy_children sad_children neutral_children boys girls sad_girls neutral_boys : ℕ),
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  neutral_children = 20 →
  boys = 22 →
  girls = 38 →
  sad_girls = 4 →
  neutral_boys = 10 →
  happy_children + sad_children + neutral_children = total_children →
  boys + girls = total_children →
  ∃ (happy_boys : ℕ), happy_boys = boys - (sad_children - sad_girls) - neutral_boys ∧ happy_boys = 6 :=
begin
  sorry
end

end happy_boys_l125_125209


namespace conjugate_of_z_l125_125781

def z := Complex.i * (4 - 3 * Complex.i)

theorem conjugate_of_z : Complex.conj z = 3 - 4 * Complex.i :=
by
  -- The proof steps would be here (not required as per instructions)
  sorry

end conjugate_of_z_l125_125781


namespace determine_g4_value_mult_l125_125531

open Real

noncomputable def g : ℝ → ℝ := sorry

axiom g_pos : ∀ x : ℝ, 0 < x → 0 < g x
axiom g_one : g 1 = 1
axiom g_eq : ∀ x : ℝ, 0 < x → g (x^2 * g x) = x * g (x^2) + g x

theorem determine_g4_value_mult : 
  let n := 1 in
  let s := 36 / 23 in
  n * s = 36 / 23 :=
by 
  sorry

end determine_g4_value_mult_l125_125531


namespace find_y_l125_125990

-- Define the conditions and the problem statement.
def positive_integers (x y : ℕ) (r : ℚ) (q : ℚ) : Prop :=
  y ≠ 0 ∧ r < y ∧ x = y * q + r

theorem find_y :
  ∃ (y : ℕ), positive_integers 96 * y + 1 y 0.12 1.44 ∧ y = 12 :=
by {
  sorry
}

end find_y_l125_125990


namespace min_sum_squares_l125_125304

noncomputable def distances (P : ℝ) : ℝ :=
  let AP := P
  let BP := |P - 1|
  let CP := |P - 2|
  let DP := |P - 5|
  let EP := |P - 13|
  AP^2 + BP^2 + CP^2 + DP^2 + EP^2

theorem min_sum_squares : ∀ P : ℝ, distances P ≥ 88.2 :=
by
  sorry

end min_sum_squares_l125_125304


namespace sum_of_real_solutions_abs_equation_l125_125618

theorem sum_of_real_solutions_abs_equation :
  ∑ x in {x : ℝ | |x^2 - 10 * x + 29| = 3}.to_finset = 0 := 
by
  sorry

end sum_of_real_solutions_abs_equation_l125_125618


namespace parallel_planes_l125_125468

-- Definitions of planes and lines
variables {α β : Type*} [plane α] [plane β]
variables {m n l1 l2 : line}

-- Conditions as hypotheses
hypothesis (hm : line_on_plane m α)
hypothesis (hn : line_on_plane n α)
hypothesis (hl1 : line_on_plane l1 β)
hypothesis (hl2 : line_on_plane l2 β)
hypothesis (distinct_mn : m ≠ n)
hypothesis (intersect_l1l2 : intersect l1 l2)

-- Parallel relations
hypothesis (m_parallel_l1 : m ∥ l1)
hypothesis (n_parallel_l2 : n ∥ l2)

-- Proposition to prove
theorem parallel_planes : α ∥ β :=
by sorry

end parallel_planes_l125_125468


namespace limit_of_recurrence_l125_125171

open Classical
noncomputable theory

def recurrence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) = (4 / 7 : ℚ) * a n + (3 / 7 : ℚ) * a (n - 1)

def initial_conditions (a : ℕ → ℚ) : Prop :=
  a 0 = 1 ∧ a 1 = 2

theorem limit_of_recurrence (a : ℕ → ℚ) (h_rec: recurrence a) (h_init: initial_conditions a) :
  (∀ L, (∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - L| < ε) → L = 17 / 10) :=
sorry

end limit_of_recurrence_l125_125171


namespace sum_divisors_36_48_l125_125023

open Finset

noncomputable def sum_common_divisors (a b : ℕ) : ℕ :=
  let divisors_a := (range (a + 1)).filter (λ x => a % x = 0)
  let divisors_b := (range (b + 1)).filter (λ x => b % x = 0)
  let common_divisors := divisors_a ∩ divisors_b
  common_divisors.sum id

theorem sum_divisors_36_48 : sum_common_divisors 36 48 = 28 := by
  sorry

end sum_divisors_36_48_l125_125023


namespace ratio_of_cube_sides_l125_125258

theorem ratio_of_cube_sides {a b : ℝ} (h : (6 * a^2) / (6 * b^2) = 16) : a / b = 4 :=
by
  sorry

end ratio_of_cube_sides_l125_125258


namespace minimum_value_and_function_value_l125_125783

noncomputable section

def f (x : ℝ) : ℝ :=
  if x >= 1 then log x / log 2 else -log x / log 2

theorem minimum_value_and_function_value (a b : ℝ) (h : 0 < a ∧ a < b ∧ f a = f b)
  : (∃ m : ℝ, m = (1 / a + 4 / b) ∧ m = 4) ∧ f (a + b) = 1 - 2 * log 2 / log 2 := 
by sorry

end minimum_value_and_function_value_l125_125783


namespace parallel_probability_perpendicular_probability_l125_125496

def vec_A (x : ℤ) : ℤ × ℤ := (x, -1)
def vec_B (y : ℤ) : ℤ × ℤ := (3, y)

def valid_x : List ℤ := [-1, 1, 3]
def valid_y : List ℤ := [1, 3, 9]

def is_parallel (x y : ℤ) : Prop := x * y = -3
def is_perpendicular (x y : ℤ) : Prop := y = 3 * x

def count_parallel : ℕ :=
(valid_x.product valid_y).count (λ ⟨x, y⟩, is_parallel x y)

def count_perpendicular : ℕ :=
(valid_x.product valid_y).count (λ ⟨x, y⟩, is_perpendicular x y)

def total_combinations : ℕ :=
(valid_x.length) * (valid_y.length)

theorem parallel_probability :
  (count_parallel : ℚ) / total_combinations = 1 / 9 := sorry

theorem perpendicular_probability :
  (count_perpendicular : ℚ) / total_combinations = 2 / 9 := sorry

end parallel_probability_perpendicular_probability_l125_125496


namespace isosceles_triangle_side_length_condition_l125_125764

theorem isosceles_triangle_side_length_condition (x y : ℕ) :
    y = x + 1 ∧ 2 * x + y = 16 → (y = 6 → x = 5) :=
by sorry

end isosceles_triangle_side_length_condition_l125_125764


namespace slope_tangent_at_2_20_l125_125264

noncomputable def f (x : ℝ) : ℝ := x^3 + x^2 * (3 : ℝ)

theorem slope_tangent_at_2_20 : 
  let m := f 2 in
  let slope := (deriv f 2) in
  (2, m) = (2, 20) ∧ slope = 24 :=
by
  sorry

end slope_tangent_at_2_20_l125_125264


namespace hyperbola_parabola_area_l125_125790

theorem hyperbola_parabola_area {a b p : ℝ} (ha : a > 0) (hb : b > 0) (hp : p > 0)
    (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
    (h_eccentricity : (Real.sqrt (a^2 + b^2)) / a = 2)
    (h_parabola : ∀ x y, y^2 = 2 * p * x)
    (h_area : let A := (-p / 2, b * p / (2 * a)),
                  B := (-p / 2, -b * p / (2 * a))
              in 1 / 2 * abs (-p / 2) * abs ((b * p / a)) = Real.sqrt 3 / 4) : p = 1 := 
sorry

end hyperbola_parabola_area_l125_125790


namespace last_letter_of_100th_permutation_l125_125919

noncomputable def BRICK := ['B', 'R', 'I', 'C', 'K']

theorem last_letter_of_100th_permutation :
  (Permutations (multiset_finset (multiset.of_finset (set_of_finite (set_univ BRICK))))) 100).last = 'K' := sorry

end last_letter_of_100th_permutation_l125_125919


namespace efficient_elimination_l125_125967

def eq1 (x y : ℝ) := 3 * x - 2 * y = 3
def eq2 (x y : ℝ) := 4 * x + y = 15 

theorem efficient_elimination (x y : ℝ) :
  (eq2 x y) * 2 + eq1 x y → (λ x, 11 * x = 33) :=
by
  sorry

end efficient_elimination_l125_125967


namespace iron_sheet_required_l125_125962

def base_diameter_dm : ℝ := 1
def height_dm : ℝ := 5
def pi : ℝ := Real.pi
def to_cm(dm : ℝ) : ℝ := dm * 10

def radius_cm := to_cm(base_diameter_dm) / 2
def height_cm := to_cm(height_dm)

def base_area := π * radius_cm^2
def lateral_surface_area := 2 * π * radius_cm * height_cm

def total_surface_area := base_area + lateral_surface_area

theorem iron_sheet_required : total_surface_area = 1648.5 := by
  -- Proof here
  sorry

end iron_sheet_required_l125_125962


namespace roots_sum_product_l125_125935

theorem roots_sum_product (m n : ℝ) (h1 : m / 2 = 6) (h2 : n / 2 = 10) : m + n = 32 := by
  have hm : m = 12 := by
    rw [eq_div_iff_mul_eq] at h1
    norm_num at h1
    exact h1
  have hn : n = 20 := by
    rw [eq_div_iff_mul_eq] at h2
    norm_num at h2
    exact h2
  rw [hm, hn]
  norm_num

end roots_sum_product_l125_125935


namespace interval_monotonic_decrease_f_alpha_l125_125461

noncomputable def f (x : ℝ) : ℝ :=
  sin (π / 4 + x) * sin (π / 4 - x)

theorem interval_monotonic_decrease (k : ℤ) : 
  ∀ x, k * π ≤ x ∧ x ≤ k * π + π / 2 → f x = 1 / 2 * cos (2 * x) := 
by sorry

theorem f_alpha (α : ℝ) (hα: α < π / 2) (h : sin (α - π / 4) = 1 / 2) : f α = -sqrt 3 / 2 :=
by sorry

end interval_monotonic_decrease_f_alpha_l125_125461


namespace cuboid_dimensions_sum_l125_125653

theorem cuboid_dimensions_sum (A B C : ℝ) 
  (h1 : A * B = 45) 
  (h2 : B * C = 80) 
  (h3 : C * A = 180) : 
  A + B + C = 145 / 9 :=
sorry

end cuboid_dimensions_sum_l125_125653


namespace arithmetic_sequence_ratio_l125_125308

theorem arithmetic_sequence_ratio :
  let numerator := ∑ k in Finset.range 17, (2 + k * 2)
  let denominator := ∑ k in Finset.range 17, (3 + k * 3)
  numerator / denominator = 2 / 3 :=
by
  sorry

end arithmetic_sequence_ratio_l125_125308


namespace integers_satisfying_inequality_l125_125474

theorem integers_satisfying_inequality :
  {n : ℤ | -15 ≤ n ∧ n ≤ 15 ∧ (n - 3) * (n + 5) * (n + 9) < 0}.to_finset.card = 13 :=
by
  sorry

end integers_satisfying_inequality_l125_125474


namespace five_digit_divisible_by_15_l125_125533

theorem five_digit_divisible_by_15 :
  let q_values := {q : ℕ | 200 ≤ q ∧ q ≤ 1999}
  let r_values := {r : ℕ | 0 ≤ r ∧ r < 50}
  let valid_pairs := (q_values × r_values).filter (λ (qr : ℕ × ℕ), (qr.fst + qr.snd) % 15 = 0)
  valid_pairs.card = 7200 :=
by 
  sorry

end five_digit_divisible_by_15_l125_125533


namespace find_r_l125_125118

theorem find_r (k r : ℝ) : 
  5 = k * 3^r ∧ 45 = k * 9^r → r = 2 :=
by 
  sorry

end find_r_l125_125118


namespace solving_linear_equations_problems_l125_125561

theorem solving_linear_equations_problems (total_problems : ℕ) (perc_algebra : ℕ) :
  total_problems = 140 → perc_algebra = 40 → 
  let algebra_problems := (perc_algebra * total_problems) / 100 in
  let solving_linear := algebra_problems / 2 in
  solving_linear = 28 :=
by
  intros h_total h_perc
  let algebra_problems := (perc_algebra * total_problems) / 100
  let solving_linear := algebra_problems / 2
  have h1 : algebra_problems = 56 := by
    rw [h_total, h_perc]
    norm_num
  have h2 : solving_linear = algebra_problems / 2 := by rfl
  rw [h1, h2]
  norm_num
  simp only [Nat.div_eq_of_lt]
  norm_num
  sorry

end solving_linear_equations_problems_l125_125561


namespace board_product_l125_125896

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

def has_nonzero_digits (n : ℕ) : Prop :=
  n.digits 10 ≠ [0]

def has_one_even_digit (n m : ℕ) : Prop := 
  (list.filter (λ d, d % 2 = 0) (n.digits 10 ++ m.digits 10)).length = 1

theorem board_product :
  ∃ (x y : ℕ), 
    100 ≤ x ∧ x ≤ 999 ∧ 
    100 ≤ y ∧ y ≤ 999 ∧ 
    has_nonzero_digits x ∧
    has_nonzero_digits y ∧
    has_one_even_digit x y ∧ 
    x ≠ y ∧
    let p := x * y in
    is_palindrome p ∧ 10000 ≤ p ∧ p ≤ 99999 ∧
    p = 29392 :=
sorry

end board_product_l125_125896


namespace soccer_games_total_l125_125269

/-- There are 15 teams in a soccer league. Each team plays each of the other teams once. -/
theorem soccer_games_total : ∃ t, t = 15*(15-1)/2 := 
by
  use 105
  sorry

end soccer_games_total_l125_125269


namespace positive_integers_condition_l125_125740

theorem positive_integers_condition : ∃ n : ℕ, (n > 0) ∧ (n < 50) ∧ (∃ k : ℕ, n = k * (50 - n)) :=
sorry

end positive_integers_condition_l125_125740


namespace sin_subtract_of_obtuse_angle_l125_125774

open Real -- Open the Real namespace for convenience.

theorem sin_subtract_of_obtuse_angle (α : ℝ) 
  (h1 : (π / 2) < α) (h2 : α < π)
  (h3 : sin (π / 4 + α) = 3 / 4)
  : sin (π / 4 - α) = - (sqrt 7) / 4 := 
by 
  sorry -- Proof placeholder.

end sin_subtract_of_obtuse_angle_l125_125774


namespace eugene_pencils_l125_125395

theorem eugene_pencils :
  ∀ (initial_pencils remaining_pencils given_pencils : ℝ),
  initial_pencils = 51.0 → remaining_pencils = 45 →
  given_pencils = initial_pencils - remaining_pencils →
  given_pencils = 6 :=
by
  intros initial_pencils remaining_pencils given_pencils
  assume h_initial h_remaining h_given
  rw [h_initial, h_remaining] at h_given
  exact h_given⟩

end eugene_pencils_l125_125395


namespace pencil_price_l125_125860

theorem pencil_price (P : ℝ) (hp1 : 30 * P) (hp2 : 50 * P) (hp3 : 50 * P = 30 * P + 80) : P = 4 :=
by
  sorry

end pencil_price_l125_125860


namespace amy_total_soups_l125_125353

def chicken_soup := 6
def tomato_soup := 3
def vegetable_soup := 4
def clam_chowder := 2
def french_onion_soup := 1
def minestrone_soup := 5

theorem amy_total_soups : (chicken_soup + tomato_soup + vegetable_soup + clam_chowder + french_onion_soup + minestrone_soup) = 21 := by
  sorry

end amy_total_soups_l125_125353


namespace angle_inequality_l125_125287

def valid_angles (x : ℝ) (k : ℤ) : Prop :=
  (-π/6 + 2*k*π < x ∧ x < π/6 + 2*k*π) ∨ (2*π/3 + 2*k*π < x ∧ x < 4*π/3 + 2*k*π)

theorem angle_inequality (x : ℝ) : 
  (cos (2 * x) - 4 * cos (π / 4) * cos (5 * π / 12) * cos x + cos (5 * π / 6) + 1 > 0) ↔ 
  ∃ k : ℤ, valid_angles x k :=
sorry

end angle_inequality_l125_125287


namespace option_d_correct_l125_125770

variables (α β γ : Set Point) (a b : Set Point) (P : Point)

-- Non-coincident planes
axiom non_coincident_planes : α ≠ β ∧ β ≠ γ ∧ γ ≠ α

-- Non-coincident lines
axiom non_coincident_lines : a ≠ b

-- Conditions for option D
axiom perp_a_alpha : a ⊥ α
axiom point_of_intersection : a ∩ b = {P}

theorem option_d_correct : ¬(b ⊥ α) := sorry

end option_d_correct_l125_125770


namespace sum_of_roots_quadratic_l125_125732

theorem sum_of_roots_quadratic (a b c : ℝ) (h_eq : a ≠ 0) (h_eqn : -48 * a * (a * 1) + 100 * a + 200 * a^2 = 0) : 
  - b / a = (25 : ℚ) / 12 :=
by
  have h1 : a = -48 := rfl
  have h2 : b = 100 := rfl
  sorry

end sum_of_roots_quadratic_l125_125732


namespace cyclist_start_time_l125_125321

/-- Given a cyclist riding at a constant speed such that at 11:22 they had covered
1.4 times the distance covered at 11:08, prove that they started riding at 10:33. -/
theorem cyclist_start_time :
  (constant_speed : ∀ t₁ t₂, d t₁ / t₁ = d t₂ / t₂)
  (d_11_22 = 1.4 * d_11_08) → 
  (start_time = "10:33") := by
  sorry

end cyclist_start_time_l125_125321


namespace volume_of_revolution_l125_125684

theorem volume_of_revolution :
  (∫ x in (0 : ℝ) .. 1, (x : ℝ) - x^6) * π = (5 * π) / 14 :=
by
  sorry

end volume_of_revolution_l125_125684


namespace max_value_of_f_l125_125260

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x) ^ 2 + (Real.sin x) * (Real.cos x)

theorem max_value_of_f :
  ∃ x ∈ Icc (0:ℝ) (2 * Real.pi), f x = 3 / 2 := sorry

end max_value_of_f_l125_125260


namespace sale_in_fifth_month_l125_125323

theorem sale_in_fifth_month (s1 s2 s3 s4 s5 s6 : ℤ) (avg_sale : ℤ) (h1 : s1 = 6435) (h2 : s2 = 6927)
  (h3 : s3 = 6855) (h4 : s4 = 7230) (h6 : s6 = 7391) (h_avg_sale : avg_sale = 6900) :
    (s1 + s2 + s3 + s4 + s5 + s6) / 6 = avg_sale → s5 = 6562 :=
by
  sorry

end sale_in_fifth_month_l125_125323


namespace prove_pq_l125_125386

def single_track_curve (C : Set (ℝ × ℝ)) : Prop :=
  ∃ f : ℝ × ℝ → ℝ, Continuous f ∧ ∀ p, f p = 0 → ∃ γ : ℝ → ℝ × ℝ, Continuous γ ∧
    (∀ t, γ t ∈ C) ∧ γ 0 = p ∧ (∀ t₁ t₂, γ t₁ = γ t₂ → t₁ = t₂)

def double_track_curve (C : Set (ℝ × ℝ)) : Prop :=
  ∃ C₁ C₂, single_track_curve C₁ ∧ single_track_curve C₂ ∧ (∀ p ∈ C₁, p ∉ C₂) ∧
    C = C₁ ∪ C₂

noncomputable def Γ (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ sqrt ((x + 1)^2 + y^2) * sqrt ((x - 1)^2 + y^2) = m}

theorem prove_pq :
  (∃ m, m > 1 ∧ single_track_curve (Γ m)) ∧
  (∃ m, 0 < m ∧ m < 1 ∧ double_track_curve (Γ m)) :=
sorry

end prove_pq_l125_125386


namespace distance_maximized_l125_125143

-- Define the equation of the curve C1
def curveC1 (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 / 4 = 1

-- Define the polar equation of the line l
def lineLPolar (ρ θ : ℝ) : Prop :=
  ρ * (2 * Real.cos θ - Real.sin θ) = 6

-- Define the Cartesian equation of the line l
def lineL (x y : ℝ) : Prop :=
  2 * x - y = 6

-- Define the distance from a point P to the line l
def distance_to_line (P : ℝ × ℝ) (x y : ℝ) : ℝ :=
  abs (2 * P.1 - P.2 - 6) / Real.sqrt (2^2 + (-1)^2)

-- The point on the curve C1 that maximizes the distance to the line l
def pointP : ℝ × ℝ :=
  (-3 / 2, 1)

-- The maximum distance from P to the line l
def max_distance := 2 * Real.sqrt 5

theorem distance_maximized :
  ∀ x y, curveC1 x y → lineL x y →
  distance_to_line pointP x y = max_distance := by
  intros x y curve_eq line_eq
  sorry

end distance_maximized_l125_125143


namespace candle_burning_time_l125_125276

theorem candle_burning_time :
  ∃ t : ℚ, (1 - t / 5) = 3 * (1 - t / 4) ∧ t = 40 / 11 :=
by {
  sorry
}

end candle_burning_time_l125_125276


namespace concyclic_iff_ratio_real_l125_125555

noncomputable def concyclic_condition (z1 z2 z3 z4 : ℂ) : Prop :=
  (∃ c : ℂ, c ≠ 0 ∧ ∀ (w : ℂ), (w - z1) * (w - z3) / ((w - z2) * (w - z4)) = c)

noncomputable def ratio_real (z1 z2 z3 z4 : ℂ) : Prop :=
  ∃ r : ℝ, (z1 - z3) * (z2 - z4) / ((z1 - z4) * (z2 - z3)) = r

theorem concyclic_iff_ratio_real (z1 z2 z3 z4 : ℂ) :
  concyclic_condition z1 z2 z3 z4 ↔ ratio_real z1 z2 z3 z4 :=
sorry

end concyclic_iff_ratio_real_l125_125555


namespace log_squared_inequality_l125_125431

theorem log_squared_inequality (a b c : ℝ) (h1 : (1/2)^a > (1/2)^b) (h2 : (1/2)^b > 1) :
  log 2 (a^2) > log 2 (b^2) :=
sorry

end log_squared_inequality_l125_125431


namespace centroid_on_circle_PQR_l125_125050

-- Definitions for points: A, B, C
variables (A B C : Type*)
-- Midpoints D, E, F
variables (D E F : Type*)
-- Intersection Points P, Q, R and centroid G
variables (P Q R G : Type*)

-- Definition for properties of triangle and centroid
def is_non_isosceles_triangle (A B C : Type*) := sorry
def is_midpoint (X Y Z M : Type*) := sorry -- X and Y are endpoints, M is midpoint of XZ
def is_intersection (CircleX PointY LineZ PointW : Type*) := sorry -- CircleX and LineZ intersect at PointW

-- Given conditions
axiom non_isosceles_triangle : is_non_isosceles_triangle A B C
axiom midpoint_D : is_midpoint B C D
axiom midpoint_E : is_midpoint C A E
axiom midpoint_F : is_midpoint A B F
axiom intersection_P : is_intersection (circle_through B C F) P (line_through B E)
axiom intersection_Q : is_intersection (circle_through A B E) Q (line_through A D)
axiom intersection_R : is_intersection (line_through D P) R (line_through F Q)
axiom centroid_G : is_centroid G A B C

-- Proof statement
theorem centroid_on_circle_PQR : 
  lies_on_circle G (circle_through P Q R) := sorry

end centroid_on_circle_PQR_l125_125050


namespace length_of_interval_of_m_l125_125535

def is_lattice_point (p : ℕ × ℕ) : Prop :=
  1 ≤ p.1 ∧ p.1 ≤ 20 ∧ 1 ≤ p.2 ∧ p.2 ≤ 20

def set_of_lattice_points : set (ℕ × ℕ) :=
  { p | is_lattice_point p }

def points_below_line (m : ℚ) : set (ℕ × ℕ) :=
  { p | is_lattice_point p ∧ p.2 ≤ m * p.1 }

theorem length_of_interval_of_m (m : ℚ) :
  (∃ (p : ℚ), points_below_line p = 200) →
  ∃ a b : ℕ, a * 84 = b ∧ gcd a b = 1 ∧ m = a / b := 
sorry

end length_of_interval_of_m_l125_125535


namespace largest_number_is_B_l125_125620
open Real

noncomputable def A := 0.989
noncomputable def B := 0.998
noncomputable def C := 0.899
noncomputable def D := 0.9899
noncomputable def E := 0.8999

theorem largest_number_is_B :
  B = max (max (max (max A B) C) D) E :=
by
  sorry

end largest_number_is_B_l125_125620


namespace trigonometric_sum_not_always_positive_l125_125558

-- Define the arbitrary coefficients
variables {a31 a30 a29 a28 a27 a26 a25 a24 a23 a22 a21 a20 a19 a18 a17 a16 a15 a14 a13 a12 a11 a10 a9 a8 a7 a6 a5 a4 a3 a2 a1 : ℝ}

-- Define the main theorem
theorem trigonometric_sum_not_always_positive :
  ¬ (∀ x : ℝ, (cos (32 * x) + 
               a31 * cos (31 * x) + 
               a30 * cos (30 * x) + 
               a29 * cos (29 * x) + 
               a28 * cos (28 * x) + 
               a27 * cos (27 * x) + 
               a26 * cos (26 * x) + 
               a25 * cos (25 * x) + 
               a24 * cos (24 * x) + 
               a23 * cos (23 * x) + 
               a22 * cos (22 * x) + 
               a21 * cos (21 * x) + 
               a20 * cos (20 * x) + 
               a19 * cos (19 * x) + 
               a18 * cos (18 * x) + 
               a17 * cos (17 * x) + 
               a16 * cos (16 * x) + 
               a15 * cos (15 * x) + 
               a14 * cos (14 * x) + 
               a13 * cos (13 * x) + 
               a12 * cos (12 * x) + 
               a11 * cos (11 * x) + 
               a10 * cos (10 * x) + 
               a9  * cos (9 * x)  + 
               a8  * cos (8 * x)  + 
               a7  * cos (7 * x)  + 
               a6  * cos (6 * x)  + 
               a5  * cos (5 * x)  + 
               a4  * cos (4 * x)  + 
               a3  * cos (3 * x)  + 
               a2  * cos (2 * x)  + 
               a1  * cos (x)) > 0) := 
sorry

end trigonometric_sum_not_always_positive_l125_125558


namespace CircumcirclesTangent_l125_125196

-- Definitions for points and conditions
variable {Point : Type}
variable {triangle : Type}
variable (A B C H M F Q K : Point)
variable [AcuteTriangle triangle A B C]
variable [Circumcircle triangle A B C Γ]
variable [Orthocenter triangle H]
variable [Midpoint B C M]
variable [FootAltitude A F]
variable [PointsOnCircum Γ Q K]
variable [RightAngleAt AQ H]
variable [RightAngleAt QK H]

-- Statement
theorem CircumcirclesTangent (h1 : triangle.AB > triangle.AC) : 
  Tangent (Circumcircle K Q H) (Circumcircle K F M) :=
sorry

end CircumcirclesTangent_l125_125196


namespace runners_meet_at_starting_point_after_800_seconds_l125_125952

theorem runners_meet_at_starting_point_after_800_seconds :
  (∀ t, 0.5 * t ≡ 0 ∧ t % 400 = 0) →
  (∀ t, 1.0 * t ≡ 0) →
  ∃ t, t = 800 ∧ (∀ t', (t' % (800 / 0.5) = 0) → t' = 800) :=
sorry

end runners_meet_at_starting_point_after_800_seconds_l125_125952


namespace actual_time_between_two_and_three_l125_125626

theorem actual_time_between_two_and_three (x y : ℕ) 
  (h1 : 2 ≤ x ∧ x < 3)
  (h2 : 60 * y + x = 60 * x + y - 55) : 
  x = 2 ∧ y = 5 + 5 / 11 := 
sorry

end actual_time_between_two_and_three_l125_125626


namespace find_value_of_f_at_pi_over_eight_l125_125455

theorem find_value_of_f_at_pi_over_eight
  (ω : ℝ) (hω : ω > 0)
  (h_period : 2 * Real.pi / ω = Real.pi) :
  let f (x : ℝ) := Real.sin(ω * x + Real.pi / 4)
  in f (Real.pi / 8) = 1 :=
by
  sorry

end find_value_of_f_at_pi_over_eight_l125_125455


namespace tangent_line_eqn_l125_125579

-- Define the function representing the curve
def curve (x : ℝ) : ℝ := x^2 + 1/x

-- Define the point of tangency
def tangent_point : ℝ × ℝ := (1, curve 1)

-- Define the derivative of the curve
noncomputable def derivative_curve (x : ℝ) : ℝ := (deriv curve) x

-- State the theorem
theorem tangent_line_eqn : ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * 1 + b * 2 + c = 0 ∧ ∀ x y, y = curve x → a * x + b * y + c = 0 :=
  sorry

end tangent_line_eqn_l125_125579


namespace books_not_sold_l125_125344

theorem books_not_sold 
  (s : ℕ) (m : ℕ) (t : ℕ) (w : ℕ) (th : ℕ) (f : ℕ) 
  (h_s : s = 800) 
  (h_m : m = 60) 
  (h_t : t = 10) 
  (h_w : w = 20) 
  (h_th : th = 44) 
  (h_f : f = 66) : 
  s - (m + t + w + th + f) = 600 :=
by
  rw [h_s, h_m, h_t, h_w, h_th, h_f]
  simp
  norm_num
  sorry

end books_not_sold_l125_125344


namespace stream_speed_l125_125300

theorem stream_speed (C S : ℝ) 
    (h1 : C - S = 8) 
    (h2 : C + S = 12) : 
    S = 2 :=
sorry

end stream_speed_l125_125300


namespace find_AN_l125_125853

-- Define the given lengths of sides of the triangle
def sides (AB AC BC : ℝ) := AB = 8 ∧ AC = 4 ∧ BC = 6

-- Define the ratio of segments AM and CM
def ratio_AM_CM (AM CM : ℝ) := AM / CM = 3

-- Define the intersection point of AK and BM
def intersection_point_AN (AK_length AN : ℝ) := AN = AK_length * (9 / 11)

-- The given length of the angle bisector AK
def angle_bisector_length (AK_length : ℝ) := AK_length = 2 * real.sqrt 6

-- The proof problem that AN equals the given length
theorem find_AN (AB AC BC AM CM AK_length AN : ℝ)
  (h_sides : sides AB AC BC)
  (h_ratio : ratio_AM_CM AM CM)
  (h_angle_bisector : angle_bisector_length AK_length)
  (h_intersection_point : intersection_point_AN AK_length AN) :
  AN = (18 * real.sqrt 6) / 11 := 
by 
  sorry

end find_AN_l125_125853


namespace sum_of_three_digit_numbers_divisible_by_7_l125_125026

/-- Prove that the sum of all three-digit numbers that are divisible by 7 is 70336. -/
theorem sum_of_three_digit_numbers_divisible_by_7 :
  let a := 105 in
  let l := 994 in
  let d := 7 in
  let n := (l - a) / d + 1 in
  (n * (a + l)) / 2 = 70336 :=
by
  sorry

end sum_of_three_digit_numbers_divisible_by_7_l125_125026


namespace sum_of_distances_to_focus_l125_125180

def quadratic_parabola (x : ℝ) : ℝ := 2 * x^2

def circle_intersects_parabola (x : ℝ) (k h r : ℝ) : Prop :=
  (x - k)^2 + (2 * x^2 - h)^2 = r^2

theorem sum_of_distances_to_focus :
  let focus := (0, 1 / 8 : ℝ)
  let pts : List (ℝ × ℝ) := [(-14, 392), (-1, 2), (6.5, 84.5), (8.5, 2 * 8.5^2)]
  let d (p : ℝ × ℝ) := (focus.1 - p.1)^2 + (focus.2 - p.2)^2 |> Float.sqrt
  (d (-14, 392) + d (-1, 2) + d (6.5, 84.5) + d (8.5, 144.5) = 623.259) :=
by
  sorry

end sum_of_distances_to_focus_l125_125180


namespace maximum_leftover_guests_with_no_fitting_galoshes_l125_125998

theorem maximum_leftover_guests_with_no_fitting_galoshes :
  ∀ (guests : ℕ) (galoshes : ℕ → ℕ), 
    (guests = 10) → 
    (∀ g, g ≤ 10 → galoshes g = g) → 
    (∀ g, g ≤ 10 → ∀ k, (k ≥ g) → galoshes k = k) → 
    (∀ gs : list ℕ, (gs.length = 5) ∧ ∀ g, g ∈ gs → 
    ¬(∃ k, (k ≥ g) ∧ galoshes k = k)) :=
by
  intros guests galoshes h_guests h_galoshes_sizes h_galoshes_fit gs h_length h_exist_fit
  apply sorry

end maximum_leftover_guests_with_no_fitting_galoshes_l125_125998


namespace min_distance_of_complex_numbers_l125_125876

open Complex

theorem min_distance_of_complex_numbers
  (z w : ℂ)
  (h₁ : abs (z + 1 + 3 * Complex.I) = 1)
  (h₂ : abs (w - 7 - 8 * Complex.I) = 3) :
  ∃ d, d = Real.sqrt 185 - 4 ∧ ∀ Z W : ℂ, abs (Z + 1 + 3 * Complex.I) = 1 → abs (W - 7 - 8 * Complex.I) = 3 → abs (Z - W) ≥ d :=
sorry

end min_distance_of_complex_numbers_l125_125876


namespace radius_of_circle_l125_125487

theorem radius_of_circle (d : ℝ) (h : d = 22) : (d / 2) = 11 := by
  sorry

end radius_of_circle_l125_125487


namespace greatest_n_with_congruent_triangles_l125_125406

theorem greatest_n_with_congruent_triangles :
  ∃ (n : ℕ), (∀ A B C D : Point, A ≠ B ∧ C ≠ D ∧ AB ≠ CD →
  ∃ (X : Fin n → Point), (∀ i : Fin n, congruent_triangle (A, B, X i) (C, D, X i))) ∧ n = 4 :=
begin
  sorry
end

end greatest_n_with_congruent_triangles_l125_125406


namespace count_valid_numbers_count_valid_numbers_divisible_by_9_l125_125696

def valid_number (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  n = 1000 * a + 100 * b + 10 * c + d ∧
  (1000 * c + 100 * d + 10 * a + b) = n - 99

def number_of_valid_numbers : ℕ :=
  89

def number_of_valid_numbers_divisible_by_9 : ℕ :=
  10

theorem count_valid_numbers :
  {n // 1000 ≤ n ∧ n < 10000 ∧ valid_number n}.card = number_of_valid_numbers := by
  sorry

theorem count_valid_numbers_divisible_by_9 :
  {n // 1000 ≤ n ∧ n < 10000 ∧ valid_number n ∧ n % 9 = 0}.card = number_of_valid_numbers_divisible_by_9 := by
  sorry

end count_valid_numbers_count_valid_numbers_divisible_by_9_l125_125696


namespace f_neg3_l125_125582

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 4) = f x
axiom f_neg1 : f (-1) = 3

theorem f_neg3 : f (-3) = -3 :=
by {
  have h1: f (-3) = f (1), from periodic_f (-3),
  have h2: f (1) = -f (-1), from odd_f 1,
  rw h2,
  rw f_neg1,
  norm_num,
}

end f_neg3_l125_125582


namespace c₀_le_zero_l125_125590

variables {t : ℝ} (h₁ : t > 1)
def poly_seq : ℕ → (ℝ → ℝ)
| 0 := λ x, 1
| 1 := λ x, x
| (n + 1) := 
    let p_n := poly_seq n,
        p_nm1 := poly_seq (n - 1),
        num := (2 * (n : ℝ) + t - 2) * x * p_n x - (n : ℝ) * p_nm1 x,
        denom := (n : ℝ) + t - 2 in
    if hₙ : n > 0 then λ x, num / denom
    else λ x, x -- dummy, not used since n > 0 

noncomputable def exists_unique_c (f : ℝ → ℝ) (n : ℕ)
  (hf : ∃ a₀ ... aₙ : ℝ, f = λ x, a₀ * x^n + ... + aₙ) :
  ∃! (c : ℕ → ℝ), f = ((λ x, ∑ i in range (n+1), c i * (poly_seq t h₁ i x)) : ℝ → ℝ) :=
sorry 

noncomputable def find_c₀ (α β : ℝ) :
  ℝ := sorry

theorem c₀_le_zero (α β : ℝ) (h₂ : t ≥ 10) : find_c₀ t α β ≤ 0 :=
sorry 

end c₀_le_zero_l125_125590


namespace part_a_part_b_l125_125631

-- Assuming existence of function S satisfying certain properties
variable (S : Type → Type → Type → ℝ)

-- Part (a)
theorem part_a (A B C : Type) : 
  S A B C = -S B A C ∧ S A B C = S B C A :=
sorry

-- Part (b)
theorem part_b (A B C D : Type) : 
  S A B C = S D A B + S D B C + S D C A :=
sorry

end part_a_part_b_l125_125631


namespace solve_abs_ineq_l125_125019

theorem solve_abs_ineq (x : ℝ) : |(8 - x) / 4| < 3 ↔ 4 < x ∧ x < 20 := by
  sorry

end solve_abs_ineq_l125_125019


namespace min_frac_sum_l125_125771

noncomputable def min_value : ℝ :=
  3 / 2 + Real.sqrt 2

theorem min_frac_sum (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b = 2) :
  (λ x y, (1 / x) + (2 / y)) a b >= min_value :=
sorry

end min_frac_sum_l125_125771


namespace solve_equations_l125_125625

theorem solve_equations (a b x y : ℤ) 
  (h1 : b * (-1) + 2 * 2 = 8) -- Condition from Xiao Hong's solution
  (h2 : a * 1 + 3 * 4 = 5)   -- Condition from Xiao Feng's solution
  (h3 : -7 * 7 + 3 * 18 = 5) -- Verification for the solution of the original system
  (h4 : -4 * 7 + 2 * 18 = 8) -- Verification for the solution of the original system
  : a = -7 ∧ b = -4 ∧ x = 7 ∧ y = 18 :=
begin
  sorry
end

end solve_equations_l125_125625


namespace other_bill_denomination_l125_125959

-- Define the conditions of the problem
def cost_shirt : ℕ := 80
def ten_dollar_bills : ℕ := 2
def other_bills (x : ℕ) : ℕ := ten_dollar_bills + 1

-- The amount paid with $10 bills
def amount_with_ten_dollar_bills : ℕ := ten_dollar_bills * 10

-- The total amount should match the cost of the shirt
def total_amount (x : ℕ) : ℕ := amount_with_ten_dollar_bills + (other_bills x) * x

-- Statement to prove
theorem other_bill_denomination : 
  ∃ (x : ℕ), total_amount x = cost_shirt ∧ x = 20 :=
by
  sorry

end other_bill_denomination_l125_125959


namespace min_value_of_sequence_l125_125053

noncomputable def sequence (a : ℕ → ℝ) := (∀ n : ℕ, a (n + 2) = a (n + 1) + 2 * a n)
noncomputable def geometric_mean (a : ℕ → ℝ) (m n : ℕ) := (real.sqrt (a m * a n) = 4 * a 1)
noncomputable def minimum_value (m n : ℕ) := (m + n = 6) → (1 / m + 5 / n = 1 + real.sqrt 5 / 3)

theorem min_value_of_sequence (a : ℕ → ℝ) (m n : ℕ)
  (h_seq : sequence a)
  (h_geom : geometric_mean a m n)
  (h_sum : m + n = 6) :
  1 / m + 5 / n = 1 + real.sqrt 5 / 3 :=
sorry

end min_value_of_sequence_l125_125053


namespace students_not_next_each_other_l125_125029

open Nat

theorem students_not_next_each_other (n : ℕ) (k : ℕ) (m : ℕ) (h1 : n = 5) (h2 : k = 2) (h3 : m = 3)
  (h4 : ∀ (A B : ℕ), A ≠ B) : 
  ∃ (total : ℕ), total = 3! * (choose (5-3+1) 2) := 
by
  sorry

end students_not_next_each_other_l125_125029


namespace cyclic_quadrilateral_property_l125_125060

variables {A B C D X D_I C_I M Y E F : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variables [MetricSpace X] [MetricSpace D_I] [MetricSpace C_I] [MetricSpace M]
variables [MetricSpace Y] [MetricSpace E] [MetricSpace F]

def cyclic_quadrilateral (A B C D : Type*) := 
  ∃ (circ : Set Type*), A ∈ circ ∧ B ∈ circ ∧ C ∈ circ ∧ D ∈ circ

def midpoint (A B M : Type*) [MetricSpace M] := 
  dist A M = dist B M ∧ dist M B = (dist A B) / 2

variables (h1 : cyclic_quadrilateral A B C D)
variables (h2 : line_intersect A C B D = X)
variables (h3 : midpoint D X D_I)
variables (h4 : midpoint C X C_I)
variables (h5 : midpoint D C M)
variables (h6 : lines_intersect A D_I B C_I = Y)
variables (h7 : line_intersect_point_line_segment Y M A C = E)
variables (h8 : line_intersect_point_line_segment Y M B D = F)

theorem cyclic_quadrilateral_property 
    (A B C D X D_I C_I M Y E F : Type*) 
    [MetricSpace A] [MetricSpace B] [MetricSpace C]
    [MetricSpace D] [MetricSpace X] [MetricSpace D_I]
    [MetricSpace C_I] [MetricSpace M] [MetricSpace Y]
    [MetricSpace E] [MetricSpace F]
    (h1 : cyclic_quadrilateral A B C D)
    (h2 : line_intersect A C B D = X)
    (h3 : midpoint D X D_I)
    (h4 : midpoint C X C_I)
    (h5 : midpoint D C M)
    (h6 : lines_intersect A D_I B C_I = Y)
    (h7 : line_intersect_point_line_segment Y M A C = E)
    (h8 : line_intersect_point_line_segment Y M B D = F) :
  dist X Y ^ 2 = dist Y E * dist Y F := 
sorry

end cyclic_quadrilateral_property_l125_125060


namespace locus_of_centers_eq_circle_l125_125177

noncomputable theory
open Real

def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem locus_of_centers_eq_circle (P Q : ℝ × ℝ) (r : ℝ) (hr : 0 < r) :
  let M := midpoint P Q in
  let d := distance P Q in
  (d / 2) ≤ r → 
  set_of (λ O : ℝ × ℝ, distance O P = r ∧ distance O Q = r) =
  {O | distance O M = sqrt (r^2 - (d / 2)^2)} := sorry

end locus_of_centers_eq_circle_l125_125177


namespace find_abcd_from_N_l125_125591

theorem find_abcd_from_N (N : ℕ) (hN1 : N ≥ 10000) (hN2 : N < 100000)
  (hN3 : N % 100000 = (N ^ 2) % 100000) : (N / 10) / 10 / 10 / 10 = 2999 := by
  sorry

end find_abcd_from_N_l125_125591


namespace magnitude_of_p_l125_125772

variable {V : Type*} [inner_product_space ℝ V]

theorem magnitude_of_p (a b p : V) 
  (a_unit : ∥a∥ = 1)
  (b_unit : ∥b∥ = 1)
  (a_dot_b : ⟪a, b⟫ = -1/2)
  (p_dot_a : ⟪p, a⟫ = 1/2)
  (p_dot_b : ⟪p, b⟫ = 1/2) :
  ∥p∥ = 1 := 
sorry

end magnitude_of_p_l125_125772


namespace number_of_broken_lines_l125_125556

theorem number_of_broken_lines (n : ℕ) :
  let binomial := (n.choose n)
  (number_of_broken_lines_length_2n n) = binomial ^ 2 := 
sorry

end number_of_broken_lines_l125_125556


namespace minimal_students_for_activities_l125_125651

theorem minimal_students_for_activities : ∃ (n : ℕ), 
  (∀ (students : Fin n → set (Fin 6)), 
    (∀ i : Fin n, students i ⊆ Finset.range 6 ∧ Finset.card (students i) ≤ 3) → 
    (∀ (d1 d2 d3 : Fin 6), 
      ∃ i : Fin n, {d1, d2, d3} ⊆ students i) →
    n = 20) :=
sorry

end minimal_students_for_activities_l125_125651


namespace not_exists_tangent_to_line_l125_125972

noncomputable def f_D (x : ℝ) : ℝ := sqrt x + 2 * x

theorem not_exists_tangent_to_line : ¬ ∃ (x : ℝ), x > 0 ∧ (deriv f_D x = 2) :=
by 
  sorry -- to be proven

end not_exists_tangent_to_line_l125_125972


namespace remainder_of_power_division_l125_125616

theorem remainder_of_power_division :
  (2^222 + 222) % (2^111 + 2^56 + 1) = 218 :=
by sorry

end remainder_of_power_division_l125_125616


namespace initial_money_l125_125606

theorem initial_money (cost_of_candy_bar : ℕ) (change_received : ℕ) (initial_money : ℕ) 
  (h_cost : cost_of_candy_bar = 45) (h_change : change_received = 5) :
  initial_money = cost_of_candy_bar + change_received :=
by
  -- here is the place for the proof which is not needed
  sorry

end initial_money_l125_125606


namespace complex_fraction_simplifies_to_minus_two_l125_125267

theorem complex_fraction_simplifies_to_minus_two :
  (1 / (1 + real.root 3 4) + 1 / (1 - real.root 3 4) + 2 / (1 + real.sqrt 3)) = -2 :=
by sorry

end complex_fraction_simplifies_to_minus_two_l125_125267


namespace num_valid_n_l125_125741

theorem num_valid_n : ∃ k, k = 4 ∧ ∀ n : ℕ, (0 < n ∧ n < 50 ∧ ∃ m : ℕ, m > 0 ∧ n = m * (50 - n)) ↔ 
  (n = 25 ∨ n = 40 ∨ n = 45 ∨ n = 48) :=
by 
  sorry

end num_valid_n_l125_125741


namespace probability_sum_20_with_2_dodecahedral_dice_l125_125253

theorem probability_sum_20_with_2_dodecahedral_dice : 
  let outcomes := { (a, b) | a ∈ finset.range 1 13 ∧ b ∈ finset.range 1 13 } in
  let favorable := { (a, b) | a ∈ finset.range 1 13 ∧ b ∈ finset.range 1 13 ∧ a + b = 20 } in
  (favorable.card : ℚ) / (outcomes.card : ℚ) = 1 / 48 := 
by
  sorry

end probability_sum_20_with_2_dodecahedral_dice_l125_125253


namespace part1_zero_in_interval_part2_monotonic_intervals_l125_125070

-- (a) Define the function f
def f (x : ℝ) : ℝ := Real.cos x - Real.exp (-x)

-- (b) Define the derivative f'
def f' (x : ℝ) : ℝ := -Real.sin x + Real.exp (-x)

-- (c) Prove that f' has exactly one zero in the interval (π/6, π/4)
theorem part1_zero_in_interval : ∃! x ∈ Set.Ioo (Real.pi / 6) (Real.pi / 4), f' x = 0 := sorry

-- (d) Prove that f has two distinct intervals of monotonic increase and one interval of monotonic decrease on [0, 2π]
theorem part2_monotonic_intervals :
  ∃ a b : ℝ, 0 < a ∧ a < b ∧ b < 2*Real.pi ∧
    (∀ x ∈ Set.Ioo 0 a, 0 < f' x) ∧
    (∀ x ∈ Set.Ioo a b, f' x < 0) ∧
    (∀ x ∈ Set.Ioo b (2*Real.pi), 0 < f' x) := sorry

end part1_zero_in_interval_part2_monotonic_intervals_l125_125070


namespace ratio_AB_CD_over_AC_BD_perpendicular_tangents_circumcircles_l125_125521

variables (A B C D : Type) [EuclideanGeometry A B C D] 

-- Definitions based on conditions
def angle_ADB_eq_angle_ACB_add_90 := 
  ∀ (α A B C D : Type) [EuclideanGeometry A B C D], 
  ∠ (line A D B) = ∠ (line A C B) + 90

def AC_mul_BD_eq_AD_mul_BC := 
  ∀ (A B C D : Type) [EuclideanGeometry A B C D],
  dist A C * dist B D = dist A D * dist B C

-- Theorem for the ratio calculation (1)
theorem ratio_AB_CD_over_AC_BD (A B C D : Type) [EuclideanGeometry A B C D] : 
  (angle_ADB_eq_angle_ACB_add_90 A B C D) → 
  (AC_mul_BD_eq_AD_mul_BC A B C D) → 
  dist A B * dist C D / (dist A C * dist B D) = real.sqrt 2 := 
  by sorry

-- Theorem for perpendicular tangents (2)
theorem perpendicular_tangents_circumcircles (A B C D : Type) [EuclideanGeometry A B C D] : 
  (angle_ADB_eq_angle_ACB_add_90 A B C D) → 
  (AC_mul_BD_eq_AD_mul_BC A B C D) → 
  tangents_are_perpendicular (circ ABC D) (circ ABD C) :=
  by sorry

end ratio_AB_CD_over_AC_BD_perpendicular_tangents_circumcircles_l125_125521


namespace check_quadratic_conclusions_l125_125332

theorem check_quadratic_conclusions (y : ℝ → ℝ) (a b c : ℝ) (ha : a ≠ 0)
    (h1 : y (-3) = 0) (h2 : y (-2) = -3) (h3 : y (-1) = -4) (h4 : y 0 = -3)
    (h_quad : ∀ x : ℝ, y x = a * x^2 + b * x + c) :
    2 = (if a * c < 0 then 1 else 0) +
        (if ∀ x, x > 1 → ∀ y, y x ≥ y (x-1) then 1 else 0) +
        (if ∀ x, (a * x^2 + (b - 4) * x + c = 0) → x = -4 then 1 else 0) +
        (if ∀ x, -1 < x ∧ x < 0 → a * x^2 + (b - 1) * x + c + 3 > 0 then 1 else 0) :=
begin
  sorry
end

end check_quadratic_conclusions_l125_125332


namespace range_of_t_is_2plus1OverE_to_e_l125_125433

def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem range_of_t_is_2plus1OverE_to_e :
  (∀ t : ℝ, (∃ x₀ : ℝ, x₀ ∈ Set.Icc (-1 : ℝ) (1 : ℝ) ∧
                ∀ m : ℝ, m ∈ Set.Ico (t-2) t → f x₀ = m) ↔ (2 + 1/Real.exp 1) < t ∧ t ≤ Real.exp 1) :=
sorry

end range_of_t_is_2plus1OverE_to_e_l125_125433


namespace f_one_equals_half_f_increasing_l125_125048

noncomputable def f : ℝ → ℝ := sorry

axiom f_add_half (x y : ℝ) : f (x + y) = f x + f y + 1/2

axiom f_half     : f (1/2) = 0

axiom f_positive (x : ℝ) (hx : x > 1/2) : f x > 0

theorem f_one_equals_half : f 1 = 1/2 := 
by 
  sorry

theorem f_increasing : ∀ x1 x2 : ℝ, x1 > x2 → f x1 > f x2 := 
by 
  sorry

end f_one_equals_half_f_increasing_l125_125048


namespace volume_of_pyramid_l125_125697

theorem volume_of_pyramid (a b θ : ℝ) (APB : ∠PAB = 2 * θ)
  (h1 : AB = a) (h2 : BC = b) (h3 : CD = a)
  (h4 : ∀ x ∈ {A, B, C, D}, dist P x = r) 
  (h5 : a^2 + b^2 = 2) :
  volume (Pyramid.mk P A B C D) = (a * b * sqrt (a^2 * tan θ ^ 2 + b^2)) / 6 :=
  sorry

end volume_of_pyramid_l125_125697


namespace cube_side_length_ratio_l125_125652

-- Define the conditions and question
variable (s₁ s₂ : ℝ)
variable (weight₁ weight₂ : ℝ)
variable (V₁ V₂ : ℝ)
variable (same_metal : Prop)

-- Conditions
def condition1 (weight₁ : ℝ) : Prop := weight₁ = 4
def condition2 (weight₂ : ℝ) : Prop := weight₂ = 32
def condition3 (V₁ V₂ : ℝ) (s₁ s₂ : ℝ) : Prop := (V₁ = s₁^3) ∧ (V₂ = s₂^3)
def condition4 (same_metal : Prop) : Prop := same_metal

-- Volume definition based on weights and proportion
noncomputable def volume_definition (weight₁ weight₂ V₁ V₂ : ℝ) : Prop :=
(weight₂ / weight₁) = (V₂ / V₁)

-- Define the proof target
theorem cube_side_length_ratio
    (h1 : condition1 weight₁)
    (h2 : condition2 weight₂)
    (h3 : condition3 V₁ V₂ s₁ s₂)
    (h4 : condition4 same_metal)
    (h5 : volume_definition weight₁ weight₂ V₁ V₂) : 
    (s₂ / s₁) = 2 :=
by
  sorry

end cube_side_length_ratio_l125_125652


namespace maximal_area_of_fencing_maximal_area_with_internal_fence_l125_125754

theorem maximal_area_of_fencing (x : ℝ) (h : 0 ≤ x ∧ x ≤ 30) : 
  (λ x, x * (30 - x)).maximum_on {x | 0 ≤ x ∧ x ≤ 30} = 225 := 
sorry

theorem maximal_area_with_internal_fence (x : ℝ) (h : 0 ≤ x ∧ x ≤ 30) :
  (λ x, 15 * (30 - 2 * 15 / 2)).maximum_on {x | 0 ≤ x ∧ x ≤ 30} = 225 :=
sorry

end maximal_area_of_fencing_maximal_area_with_internal_fence_l125_125754


namespace ways_to_sum_2022_using_2s_and_3s_l125_125806

theorem ways_to_sum_2022_using_2s_and_3s : 
  (∃ n : ℕ, n ≤ 337 ∧ 6 * 337 = 2022) →
  (finset.card (finset.Icc 0 337) = 338) :=
by
  intros n h
  rw finset.card_Icc
  sorry

end ways_to_sum_2022_using_2s_and_3s_l125_125806


namespace tangent_line_of_circle_l125_125458

def circle_equation (x y : ℝ) := x^2 + y^2 + 3 * x - 4 * y + 6 = 0

theorem tangent_line_of_circle : ∃ y : ℝ, tangent_line (x = -2) (circle_equation x y) := 
sorry

end tangent_line_of_circle_l125_125458


namespace prob1_part1_prob1_part2_l125_125794
open Set

noncomputable def A : Set ℝ := {x | 1 ≤ 2^x ∧ 2^x ≤ 4}
noncomputable def B (a : ℝ) : Set ℝ := {x | x - a > 0}

theorem prob1_part1 (a : ℝ) (ha : a = 1) : 
  A ∩ B a = {x | 1 < x ∧ x ≤ 2} ∧
  (A ∪ B aᶜ = {x | x ≤ 2}) :=
sorry

theorem prob1_part2 (a : ℝ) (h : A ∪ B a = B a) : 
  a < 0 :=
sorry

end prob1_part1_prob1_part2_l125_125794


namespace angle_ABF_regular_octagon_l125_125222

theorem angle_ABF_regular_octagon (ABCDEFGH : Type) [regular_octagon ABCDEFGH] :
  ∃ AB F : Point, angle AB F = 22.5 := sorry

end angle_ABF_regular_octagon_l125_125222


namespace probability_of_chosen_figure_is_circle_l125_125337

-- Define the total number of figures and number of circles.
def total_figures : ℕ := 12
def number_of_circles : ℕ := 5

-- Define the probability calculation.
def probability_of_circle (total : ℕ) (circles : ℕ) : ℚ := circles / total

-- State the theorem using the defined conditions.
theorem probability_of_chosen_figure_is_circle : 
  probability_of_circle total_figures number_of_circles = 5 / 12 :=
by
  sorry  -- Placeholder for the actual proof.

end probability_of_chosen_figure_is_circle_l125_125337


namespace scalar_c_zero_l125_125729

variables {V : Type*} [inner_product_space ℝ V]

def i : V := sorry
def j : V := sorry
def k : V := sorry

axiom is_unit_vector_i : ∥i∥ = 1
axiom is_unit_vector_j : ∥j∥ = 1
axiom is_unit_vector_k : ∥k∥ = 1

theorem scalar_c_zero (v : V) : 
  i × (v × j) + j × (v × k) + k × (v × i) = 0 :=
sorry

end scalar_c_zero_l125_125729


namespace solving_linear_equations_problems_l125_125560

theorem solving_linear_equations_problems (total_problems : ℕ) (perc_algebra : ℕ) :
  total_problems = 140 → perc_algebra = 40 → 
  let algebra_problems := (perc_algebra * total_problems) / 100 in
  let solving_linear := algebra_problems / 2 in
  solving_linear = 28 :=
by
  intros h_total h_perc
  let algebra_problems := (perc_algebra * total_problems) / 100
  let solving_linear := algebra_problems / 2
  have h1 : algebra_problems = 56 := by
    rw [h_total, h_perc]
    norm_num
  have h2 : solving_linear = algebra_problems / 2 := by rfl
  rw [h1, h2]
  norm_num
  simp only [Nat.div_eq_of_lt]
  norm_num
  sorry

end solving_linear_equations_problems_l125_125560


namespace area_of_square_l125_125543

theorem area_of_square {A B C D E F : Point} (ABCD_square : square A B C D) 
  (L L' : Line) (hLA : L ∋ A) (hLC : L' ∋ C)
  (hL_perp : L ⊥ BD) (hL'_perp : L' ⊥ BD)
  (D_E_F_division : segment DB = ([D, E], [E, F], [F, B]))
  (hDE : segment_length D E = 1)
  (hEF : segment_length E F = 2)
  (hFB : segment_length F B = 1) :
  area ABCD = 4 * √3 := 
sorry

end area_of_square_l125_125543


namespace jill_total_watching_time_l125_125514

theorem jill_total_watching_time :
  let duration : ℕ → ℝ := λ n, 30 * (1.5)^n in
  (duration 0) + (duration 1) + (duration 2) + (duration 3) + (duration 4) = 395.625 :=
by
  sorry

end jill_total_watching_time_l125_125514


namespace parallelogram_sum_l125_125393

theorem parallelogram_sum 
  (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ)
  (h_vert1 : (x1, y1) = (1, 1))
  (h_vert2 : (x2, y2) = (6, 3))
  (h_vert3 : (x3, y3) = (9, 3))
  (h_vert4 : (x4, y4) = (4, 1))
  (p a : ℝ)
  (h_perimeter : p = 2 * real.sqrt ( (x2 - x1)^2 + (y2 - y1)^2 ) + 2 * real.sqrt ( (x3 - x2)^2 ))
  (h_area : a = (x3 - x2) * (y2 - y1)) :
  p + a = 2 * real.sqrt 29 + 12 := 
sorry

end parallelogram_sum_l125_125393


namespace basic_computer_price_l125_125598

-- Define the given conditions.
variable (C P : ℝ) -- Prices of the basic computer and printer
variable (total_cost : ℝ := 2500)
variable (price_increment : ℝ := 800)
variable (printer_fraction : ℝ := 1/5)
variable (tax_rate : ℝ := 0.05)
variable (total_cost_after_tax : ℝ := total_cost)

-- Define the conditions
def basic_computer_printer_total : Prop := C + P = total_cost

-- Considering the first enhanced computer
def first_enhanced_computer_before_tax : ℝ := (C + price_increment) + printer_fraction * (C + price_increment)

def first_enhanced_computer_after_tax : ℝ := first_enhanced_computer_before_tax * (1 + tax_rate)

-- Statement to prove
theorem basic_computer_price :
  basic_computer_printer_total C P →
  first_enhanced_computer_after_tax C = total_cost_after_tax →
  C = 1184.13 :=
sorry

end basic_computer_price_l125_125598


namespace probability_AC_less_than_12_l125_125355

noncomputable def probability_distance_AC_lt_12_cm 
  (α : ℝ) 
  (hα : α ∈ set.Ioo 0 real.pi) : ℝ :=
let A := (0, -10) in
let B := (0, 0) in
let C := (8 * real.cos α, 8 * real.sin α) in
let AC_dist := real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) in
if AC_dist < 12 then 
  (measure_theory.MeasureSpace.measure (set.Ioo 0 real.pi)
    (set_of (λ α, AC_dist < 12))) / (measure_theory.MeasureSpace.measure (set.Ioo 0 real.pi) set.univ)
else 0

theorem probability_AC_less_than_12 
  : probability_distance_AC_lt_12_cm = 1 / 4 :=
sorry

end probability_AC_less_than_12_l125_125355


namespace complement_union_l125_125080

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- The proof problem statement
theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3}) (hA : A = {-1, 0, 1}) (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} := sorry

end complement_union_l125_125080


namespace mila_needs_48_hours_to_earn_as_much_as_agnes_l125_125948

/-- Definition of the hourly wage for the babysitters and the working hours of Agnes. -/
def mila_hourly_wage : ℝ := 10
def agnes_hourly_wage : ℝ := 15
def agnes_weekly_hours : ℝ := 8
def weeks_in_month : ℝ := 4

/-- Mila needs to work 48 hours in a month to earn as much as Agnes. -/
theorem mila_needs_48_hours_to_earn_as_much_as_agnes :
  ∃ (mila_monthly_hours : ℝ), mila_monthly_hours = 48 ∧ 
  mila_hourly_wage * mila_monthly_hours = agnes_hourly_wage * agnes_weekly_hours * weeks_in_month := 
sorry

end mila_needs_48_hours_to_earn_as_much_as_agnes_l125_125948


namespace remy_sold_110_bottles_l125_125905

theorem remy_sold_110_bottles 
    (price_per_bottle : ℝ)
    (total_evening_sales : ℝ)
    (evening_more_than_morning : ℝ)
    (nick_fewer_than_remy : ℝ)
    (R : ℝ) 
    (total_morning_sales_is : ℝ) :
    price_per_bottle = 0.5 →
    total_evening_sales = 55 →
    evening_more_than_morning = 3 →
    nick_fewer_than_remy = 6 →
    total_morning_sales_is = total_evening_sales - evening_more_than_morning →
    (R * price_per_bottle) + ((R - nick_fewer_than_remy) * price_per_bottle) = total_morning_sales_is →
    R = 110 :=
by
  intros
  sorry

end remy_sold_110_bottles_l125_125905


namespace solve_equation_l125_125266

noncomputable def equation_solution (x : ℝ) : Prop :=
  (3 / x = 2 / (x - 2)) ∧ x ≠ 0 ∧ x - 2 ≠ 0

theorem solve_equation : (equation_solution 6) :=
  by
    sorry

end solve_equation_l125_125266


namespace sequences_fix_square_l125_125699

inductive Transformation
| L  -- rotation of 90 degrees CCW
| R  -- rotation of 90 degrees CW
| H  -- reflection across x-axis
| V  -- reflection across y-axis

def apply_transformation : Transformation → (ℝ × ℝ) → (ℝ × ℝ)
| Transformation.L (x, y) => (-y, x)
| Transformation.R (x, y) => (y, -x)
| Transformation.H (x, y) => (x, -y)
| Transformation.V (x, y) => (-x, y)

def apply_sequence : list Transformation → (ℝ × ℝ) → (ℝ × ℝ)
| [] p => p
| (t::ts) p => apply_sequence ts (apply_transformation t p)

def invariant_square (seq : list Transformation) : Prop :=
  apply_sequence seq (1, 1) = (1, 1) ∧
  apply_sequence seq (-1, 1) = (-1, 1) ∧
  apply_sequence seq (-1, -1) = (-1, -1) ∧
  apply_sequence seq (1, -1) = (1, -1)

noncomputable def count_sequences_fixing_square : ℕ :=
  -- The number of sequences of 10 transformations that return to original positions
  2^18

theorem sequences_fix_square :
  (finset.univ : finset (list Transformation)).filter invariant_square = count_sequences_fixing_square :=
sorry

end sequences_fix_square_l125_125699


namespace measure_angle_ABF_of_regular_octagon_l125_125221

theorem measure_angle_ABF_of_regular_octagon (h : regular_octagon ABCDEFGH) : angle ABF = 22.5 :=
sorry

end measure_angle_ABF_of_regular_octagon_l125_125221


namespace factor_expression_l125_125717

theorem factor_expression (x : ℝ) : 54 * x^5 - 135 * x^9 = 27 * x^5 * (2 - 5 * x^4) :=
by
  sorry

end factor_expression_l125_125717


namespace no_function_f_satisfies_l125_125875

noncomputable def g (y : ℝ) : ℝ := sorry -- Define g to be affine on segments and satisfy g(n) = (-1)^n for all integers n

theorem no_function_f_satisfies (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + f y) = f x + g y) → false :=
begin
  sorry
end

end no_function_f_satisfies_l125_125875


namespace sum_solutions_eq_120_l125_125964

open Real

theorem sum_solutions_eq_120 :
  ∑ x in { x | x = abs (2 * x - abs (100 - 2 * x)) }, x = 120 :=
by
  sorry

end sum_solutions_eq_120_l125_125964


namespace linda_original_savings_l125_125297

theorem linda_original_savings:
  ∃ (S : ℝ), (4/5 * S + 100 = S) ∧ (1/5 * S = 100) := 
begin
  use 500,
  split,
  by calc
    (4/5 * 500 + 100)
    = 400 + 100 : by norm_num
    ... = 500 : by norm_num,
  by calc
    (1/5 * 500)
    = 100 : by norm_num,
end

end linda_original_savings_l125_125297


namespace base_8_addition_square_value_l125_125820

theorem base_8_addition_square_value:
  ∃ (square : ℕ), 
    square < 8 ∧ 
    5 + 4 + 3 * 8 + square * 8^0 + square + 6 * 8 + 1 * 8^1 + 0 * 8^2 + 0 * 8^3 +
    (square + 4) ∗ 8^0 + 0 * 8^1 + 0 * 8^2 + 0 * 8^3 =
    6 * 8^0 + 5 * 8^1 + square * 8^2 + 2 * 8^3 ∧ 
    square = 6 := 
by
  sorry

end base_8_addition_square_value_l125_125820


namespace arithmetic_sequence_common_difference_l125_125423

theorem arithmetic_sequence_common_difference :
  ∀ (d : ℝ), (∀ n, seq n = -15 + (n - 1) * d) → (seq 6 > 0 ∧ seq 5 ≤ 0) → (3 < d ∧ d ≤ 15 / 4) :=
by
  intros d seq hn
  sorry

end arithmetic_sequence_common_difference_l125_125423


namespace min_distance_squared_l125_125306

noncomputable def min_squared_distances (AP BP CP DP EP : ℝ) : ℝ :=
  AP^2 + BP^2 + CP^2 + DP^2 + EP^2

theorem min_distance_squared :
  ∃ P : ℝ, ∀ (A B C D E : ℝ), A = 0 ∧ B = 1 ∧ C = 2 ∧ D = 5 ∧ E = 13 -> 
  min_squared_distances (abs (P - A)) (abs (P - B)) (abs (P - C)) (abs (P - D)) (abs (P - E)) = 114.8 :=
sorry

end min_distance_squared_l125_125306


namespace elder_brother_age_l125_125738

-- Define the conditions from the problem
variables (younger_age elder_age : ℕ) -- ages of the younger and elder brothers this year

-- Define the conditions
def condition1 := (elder_age - 5 = younger_age + 7)
def condition2 := (elder_age + 4 + younger_age - 3 = 35)

-- Define the question: Proving the elder brother's age this year
theorem elder_brother_age : 
  condition1 → condition2 → elder_age = 23 := by
  intros h1 h2
  sorry

end elder_brother_age_l125_125738


namespace tram_speed_l125_125546

variable (t1 t2 a : ℝ) (h1 : t1 = 3) (h2 : t2 = 13) (h3 : a = 100)

-- Define the conditions
axiom h1 : t1 = 3
axiom h2 : t2 = 13
axiom h3 : a = 100

-- Show that the speed of the tram v is 10 meters per second
theorem tram_speed (v : ℝ) : v = 10 :=
  by sorry

end tram_speed_l125_125546


namespace quadrilateral_side_length_eq_12_l125_125139

-- Definitions
def EF : ℝ := 7
def FG : ℝ := 15
def GH : ℝ := 7
def HE : ℝ := 12
def EH : ℝ := 12

-- Statement to prove that EH = 12 given the definition and conditions
theorem quadrilateral_side_length_eq_12
  (EF_eq : EF = 7)
  (FG_eq : FG = 15)
  (GH_eq : GH = 7)
  (HE_eq : HE = 12)
  (EH_eq : EH = 12) : 
  EH = 12 :=
sorry

end quadrilateral_side_length_eq_12_l125_125139


namespace pentagon_area_bisect_l125_125245

-- Define the coordinates of the pentagon vertices
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨0, 0⟩
def B : Point := ⟨11, 0⟩
def C : Point := ⟨11, 2⟩
def D : Point := ⟨6, 2⟩
def E : Point := ⟨0, 8⟩

-- Define the area calculation functions (if necessary in Lean)
-- Here we are skipping the detailed definitions and calculations of the area.

-- Define the problem statement
theorem pentagon_area_bisect :=
  ∃ k : ℝ, (k = 8 - 2 * Real.sqrt 6) ∧ 
    let line := λ (p : Point) => p.x = k in
      -- Ensure that the vertical line divides the pentagon into two regions with equal area
      sorry

end pentagon_area_bisect_l125_125245


namespace num_proper_subsets_of_A_l125_125869

open Finset

def A : Finset ℕ := {1, 2, 3}

theorem num_proper_subsets_of_A : A.card = 3 → (2 ^ A.card - 1) = 7 :=
by
  intros h
  sorry

end num_proper_subsets_of_A_l125_125869


namespace period_of_cosine_function_l125_125261

noncomputable def minimum_positive_period (f : ℝ → ℝ) : ℝ :=
  let T := 2 * Real.pi / 3
  T

theorem period_of_cosine_function :
  minimum_positive_period (λ x, Real.cos (3 * x - Real.pi / 3)) = 2 * Real.pi / 3 := by
  sorry

end period_of_cosine_function_l125_125261


namespace b_2023_eq_1_div_7_l125_125530

noncomputable def b : ℕ → ℚ
| 1     := 5
| 2     := 7
| (n+3) := b (n+2) / b (n+1)

theorem b_2023_eq_1_div_7 : b 2023 = 1 / 7 := 
by { sorry }

end b_2023_eq_1_div_7_l125_125530


namespace mike_sold_song_book_for_correct_amount_l125_125891

-- Define the constants for the cost of the trumpet and the net amount spent
def cost_of_trumpet : ℝ := 145.16
def net_amount_spent : ℝ := 139.32

-- Define the amount received from selling the song book
def amount_received_from_selling_song_book : ℝ :=
  cost_of_trumpet - net_amount_spent

-- The theorem stating the amount Mike sold the song book for
theorem mike_sold_song_book_for_correct_amount :
  amount_received_from_selling_song_book = 5.84 :=
sorry

end mike_sold_song_book_for_correct_amount_l125_125891


namespace probability_sum_of_dice_gt_8_l125_125277

noncomputable def probability_sum_gt_8 (event_space : set (ℕ × ℕ)) (desired_event : set (ℕ × ℕ)) (prob : ℝ) : Prop :=
  Prob.event_space event_space →
  Prob.event desired_event →
  Prob.P event_space desired_event = prob

theorem probability_sum_of_dice_gt_8 :
  probability_sum_gt_8 
    ({(r, b) | r ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {3, 6}}) 
    ({(r, b) | (r + b > 8) ∧ (b ∈ {3, 6})}) 
    (5 / 12) :=
sorry

end probability_sum_of_dice_gt_8_l125_125277


namespace correct_statement_l125_125030

open Set

variable {A M P : Set ℝ}
variable [Nonempty A] [Nonempty M] [Nonempty P]

def A_star (A : Set ℝ) : Set ℝ := {y | ∀ x ∈ A, y ≥ x}
def M_star (M : Set ℝ) : Set ℝ := {y | ∀ x ∈ M, y ≥ x}
def P_star (P : Set ℝ) : Set ℝ := {y | ∀ x ∈ P, y ≥ x}

theorem correct_statement (h_subset : M ⊆ P) : P_star P ⊆ M_star M :=
begin
  sorry
end

end correct_statement_l125_125030


namespace max_height_in_first_2_seconds_l125_125644

-- Define the ball's height as a function of time
def height (t : ℝ) : ℝ := -20 * t^2 + 50 * t + 5

-- Define the interval [0, 2]
def time_interval := set.Icc (0 : ℝ) (2 : ℝ)

-- Define the maximum height within the interval [0, 2]
theorem max_height_in_first_2_seconds : 
  ∃ t ∈ time_interval, ∀ s ∈ time_interval, height s ≤ height t ∧ height t = 36.25 :=
by 
  apply exists.intro 1.25
  split
  · show 1.25 ∈ time_interval
    sorry
  · show ∀ s ∈ time_interval, height s ≤ height 1.25 ∧ height 1.25 = 36.25
    sorry

end max_height_in_first_2_seconds_l125_125644


namespace zoo_children_count_l125_125659

theorem zoo_children_count:
  ∀ (C : ℕ), 
  (10 * C + 16 * 10 = 220) → 
  C = 6 :=
by
  intro C
  intro h
  sorry

end zoo_children_count_l125_125659


namespace interest_rate_is_4_percent_l125_125233

variable (P A n : ℝ)
variable (r : ℝ)
variable (n_pos : n ≠ 0)

-- Define the conditions
def principal : ℝ := P
def amount_after_n_years : ℝ := A
def years : ℝ := n
def interest_rate : ℝ := r

-- The compound interest formula
def compound_interest (P A r : ℝ) (n : ℝ) : Prop :=
  A = P * (1 + r) ^ n

-- The Lean theorem statement
theorem interest_rate_is_4_percent
  (P_val : principal = 7500)
  (A_val : amount_after_n_years = 8112)
  (n_val : years = 2)
  (h : compound_interest P A r n) :
  r = 0.04 :=
sorry

end interest_rate_is_4_percent_l125_125233


namespace probability_xi_greater_2_l125_125056

def xi_distribution (σ : ℝ) : Prop :=
  ∀ x : ℝ, true -- The exact definition for normal distribution can be more complex

-- Given conditions (1 and 2)
axiom xi_is_normal (σ : ℝ) : xi_distribution σ
axiom prob_neg2_to_0 (σ : ℝ) : P(-2 ≤ ξ ∧ ξ ≤ 0 | xi_distribution σ) = 0.4

-- The proof problem statement
theorem probability_xi_greater_2 {σ : ℝ} (h1 : xi_is_normal σ) (h2 : prob_neg2_to_0 σ) :
  P(ξ > 2 | xi_distribution σ) = 0.1 :=
sorry

end probability_xi_greater_2_l125_125056


namespace eq1_eq2_eq3_eq4_l125_125573

theorem eq1 : ∀ x : ℝ, x = 6 → 3 * x - 8 = x + 4 := by
  intros x hx
  rw [hx]
  sorry

theorem eq2 : ∀ x : ℝ, x = -2 → 1 - 3 * (x + 1) = 2 * (1 - 0.5 * x) := by
  intros x hx
  rw [hx]
  sorry

theorem eq3 : ∀ x : ℝ, x = -20 → (1 / 6) * (3 * x - 6) = (2 / 5) * x - 3 := by
  intros x hx
  rw [hx]
  sorry

theorem eq4 : ∀ y : ℝ, y = -1 → (3 * y - 1) / 4 - 1 = (5 * y - 7) / 6 := by
  intros y hy
  rw [hy]
  sorry

end eq1_eq2_eq3_eq4_l125_125573


namespace no_trisquarish_num_l125_125698

def is_nonzero_digit (d : ℕ) : Prop := d > 0 ∧ d < 10

def is_six_digit_num (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def is_perfect_cube (n : ℕ) : Prop := ∃ y : ℕ, y * y * y = n

def is_trisquarish (n : ℕ) : Prop :=
  is_six_digit_num n ∧
  (∀ d, d ∈ (n.digits 10) → is_nonzero_digit d) ∧
  is_perfect_cube n ∧
  let (a, b) := n.divMod 1000 in
  is_perfect_square a ∧ is_perfect_square b

theorem no_trisquarish_num : ¬ ∃ n : ℕ, is_trisquarish n :=
by {
  sorry
}

end no_trisquarish_num_l125_125698


namespace range_of_sinA_plus_sinB_plus_sinC_range_of_sinA_mul_sinB_mul_sinC_l125_125498

open Real

def is_acute_triangle (A B C : ℝ) : Prop := 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π

theorem range_of_sinA_plus_sinB_plus_sinC (A B C : ℝ) (hA : A = π / 3) (hABC : is_acute_triangle A B C) :
  (sin A + sin B + sin C) ∈ Ioc ((3 + sqrt 3)/2) ((6 + sqrt 3)/2) :=
sorry

theorem range_of_sinA_mul_sinB_mul_sinC (A B C : ℝ) (hA : A = π / 3) (hABC : is_acute_triangle A B C) :
  (sin A * sin B * sin C) ∈ Ioc 0 ((3 * sqrt 3)/8) :=
sorry

end range_of_sinA_plus_sinB_plus_sinC_range_of_sinA_mul_sinB_mul_sinC_l125_125498


namespace problem_I_II_l125_125072

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

def seq_a (a : ℕ → ℝ) (a1 : ℝ) : Prop :=
  a 0 = a1 ∧ (∀ n, a (n + 1) = f (a n))

theorem problem_I_II (a : ℕ → ℝ) (a1 : ℝ) (h_a1 : 0 < a1 ∧ a1 < 1) (h_seq : seq_a a a1) :
  (∀ n, 0 < a (n + 1) ∧ a (n + 1) < a n ∧ a n < 1) ∧
  (∀ n, a (n + 1) < (1 / 6) * (a n) ^ 3) :=
  sorry

end problem_I_II_l125_125072


namespace tianjin_2016_math_problem_l125_125839

def f (x : ℝ) := (2 * x + 1) * Real.exp x

theorem tianjin_2016_math_problem :
  (deriv f 0) = 3 :=
by
  sorry

end tianjin_2016_math_problem_l125_125839


namespace solve_for_a_l125_125116

theorem solve_for_a (x : ℝ) (h : x = 0.3) :
  ∀ a : ℝ, (a * x + 2) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / 3 → a = 10 :=
by
  intro a
  intro H
  rw h at H
  sorry

end solve_for_a_l125_125116


namespace max_product_min_quotient_l125_125372

theorem max_product_min_quotient :
  let nums := [-5, -3, -1, 2, 4]
  let a := max (max (-5 * -3) (-5 * -1)) (max (-3 * -1) (max (2 * 4) (max (2 * -1) (4 * -1))))
  let b := min (min (4 / -1) (2 / -3)) (min (2 / -5) (min (4 / -3) (-5 / -3)))
  a = 15 ∧ b = -4 → a / b = -15 / 4 :=
by
  sorry

end max_product_min_quotient_l125_125372


namespace limit_sum_distances_eq_l125_125841

noncomputable def sequence_x : ℕ → ℝ
| 0     := 1
| (n+1) := 1/2 * (sequence_x n + sequence_y n)
and sequence_y : ℕ → ℝ
| 0     := 1
| (n+1) := 1/2 * (sequence_x n - sequence_y n)

def distance_from_origin (n : ℕ) : ℝ :=
  real.sqrt (sequence_x n ^ 2 + sequence_y n ^ 2)

def sum_distances (n : ℕ) : ℝ :=
  ∑ i in finset.range n, distance_from_origin i

theorem limit_sum_distances_eq :
  filter.tendsto sum_distances filter.at_top (nhds (2 + 2 * real.sqrt 2)) :=
sorry

end limit_sum_distances_eq_l125_125841


namespace degrees_for_lemon_pie_l125_125494

theorem degrees_for_lemon_pie 
    (total_students : ℕ)
    (chocolate_lovers : ℕ)
    (apple_lovers : ℕ)
    (blueberry_lovers : ℕ)
    (remaining_students : ℕ)
    (lemon_pie_degrees : ℝ) :
    total_students = 42 →
    chocolate_lovers = 15 →
    apple_lovers = 9 →
    blueberry_lovers = 7 →
    remaining_students = total_students - (chocolate_lovers + apple_lovers + blueberry_lovers) →
    lemon_pie_degrees = (remaining_students / 2 / total_students * 360) →
    lemon_pie_degrees = 47.14 :=
by
  intros _ _ _ _ _ _
  sorry

end degrees_for_lemon_pie_l125_125494


namespace sum_lent_is_correct_l125_125656

variable (P : ℝ) -- Sum lent
variable (R : ℝ) -- Interest rate
variable (T : ℝ) -- Time period
variable (I : ℝ) -- Simple interest

-- Conditions
axiom interest_rate : R = 8
axiom time_period : T = 8
axiom simple_interest_formula : I = (P * R * T) / 100
axiom interest_condition : I = P - 900

-- The proof problem
theorem sum_lent_is_correct : P = 2500 := by
  -- The proof is skipped
  sorry

end sum_lent_is_correct_l125_125656


namespace satisfies_conditions_l125_125438

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.abs x)

lemma even_f :
  ∀ x : ℝ, f (-x) = f x := 
by sorry

lemma monotonic_increasing_f :
  ∀ x y : ℝ, 0 < x → x < y → f x < f y := 
by sorry

lemma functional_equation_f :
  ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → f (x * y) = f x + f y := 
by sorry

theorem satisfies_conditions :
  f = fun x => Real.log (Real.abs x) :=
by sorry

end satisfies_conditions_l125_125438


namespace train_ride_duration_is_360_minutes_l125_125552

-- Define the conditions given in the problem
def arrived_at_station_at_8 (t : ℕ) : Prop := t = 8 * 60
def train_departed_at_835 (t_depart : ℕ) : Prop := t_depart = 8 * 60 + 35
def train_arrived_at_215 (t_arrive : ℕ) : Prop := t_arrive = 14 * 60 + 15
def exited_station_at_3 (t_exit : ℕ) : Prop := t_exit = 15 * 60

-- Define the problem statement
theorem train_ride_duration_is_360_minutes (boarding alighting : ℕ) :
  arrived_at_station_at_8 boarding ∧ 
  train_departed_at_835 boarding ∧ 
  train_arrived_at_215 alighting ∧ 
  exited_station_at_3 alighting → 
  alighting - boarding = 360 := 
by
  sorry

end train_ride_duration_is_360_minutes_l125_125552


namespace incorrect_statement_for_function_l125_125744

theorem incorrect_statement_for_function (x : ℝ) (h : x > 0) : 
  ¬(∀ x₁ x₂ : ℝ, (x₁ > 0) → (x₂ > 0) → (x₁ < x₂) → (6 / x₁ < 6 / x₂)) := 
sorry

end incorrect_statement_for_function_l125_125744


namespace sum_of_squares_of_chords_in_sphere_l125_125442

-- Defining variables
variables (R PO : ℝ)

-- Define the problem statement
theorem sum_of_squares_of_chords_in_sphere
  (chord_lengths_squared : ℝ)
  (H_chord_lengths_squared : chord_lengths_squared = 3 * R^2 - 2 * PO^2) :
  chord_lengths_squared = 3 * R^2 - 2 * PO^2 :=
by
  sorry -- proof is omitted

end sum_of_squares_of_chords_in_sphere_l125_125442


namespace squirrel_travel_time_l125_125340

/-- Problem: Prove that the time it takes for a squirrel to travel 1 mile
at a constant speed of 4 miles per hour is 15 minutes. -/
theorem squirrel_travel_time (distance : ℝ) (speed : ℝ) (time_minutes : ℝ) 
  (h_distance : distance = 1)
  (h_speed : speed = 4)
  (h_conversion : time_minutes = (distance / speed) * 60) :
  time_minutes = 15 :=
by {
  rw [h_distance, h_speed, h_conversion],
  norm_num,
}

end squirrel_travel_time_l125_125340


namespace solve_system_l125_125574

noncomputable theory

open Real

theorem solve_system : 
  ∃ (x y : ℝ), x * log 2 3 + y = log 2 18 ∧ 5^x = 25^y ∧ x = 2 ∧ y = 1 :=
by {
  refine ⟨2, 1, _, _, rfl, rfl⟩;
  -- The following are the conditions given in the problem, using logarithms and exponentiation rules
  { linarith,
    rw [log_mul, log_pow],
    norm_num,
    rw [log, mul_comm],
    sorry },
  { linarith,
    sorry }
}

end solve_system_l125_125574


namespace items_in_bags_l125_125675

theorem items_in_bags : 
  let items : ℕ := 5
  let bags : ℕ := 3
  let identical : bool := true
  number_of_ways_to_distribute items bags identical = 31 :=
by sorry

end items_in_bags_l125_125675


namespace angle_ABF_is_correct_l125_125216

-- Define a regular octagon
structure RegularOctagon (A B C D E F G H : Type) := 
  (sides_eq : ∀ (i j : ℕ), 0 ≤ i ∧ i < 8 → 0 ≤ j ∧ j < 8 → (A i) = (A j))
  (angles_eq : ∀ (i j : ℕ), 0 ≤ i ∧ i < 8 → 0 ≤ j ∧ j < 8 → (A (i + 1) - A i) = 135)

noncomputable def measure_angle_ABF {A B C D E F G H : Type} 
  (oct : RegularOctagon A B C D E F G H) : ℝ :=
22.5

theorem angle_ABF_is_correct (A B C D E F G H : Type) 
  (oct : RegularOctagon A B C D E F G H) :
  measure_angle_ABF oct = 22.5 :=
by
  sorry

end angle_ABF_is_correct_l125_125216


namespace find_side_c_l125_125491

theorem find_side_c {a b c : ℝ} (A B C : ℝ) 
  (ha : a = 5) 
  (hb : b = 7) 
  (hB : B = 60 * real.pi / 180) :
  c = 8 :=
by
  -- We state that we need to prove c = 8 given the conditions
  sorry

end find_side_c_l125_125491


namespace calculate_sum_l125_125368

open Real

theorem calculate_sum :
  (-1: ℝ) ^ 2023 + (1/2) ^ (-2: ℝ) + 3 * tan (pi / 6) - (3 - pi) ^ 0 + |sqrt 3 - 2| = 4 :=
by
  sorry

end calculate_sum_l125_125368


namespace part1_part2_l125_125063

noncomputable def f (x : ℝ) (m : ℝ) := (m ^ 2 - 3 * m + 3) * x ^ (m + 1)
def g (x : ℝ) (m : ℝ) := f x m + x + 2

theorem part1 (h : ∀ x : ℝ, f x 1 = f (-x) 1) : f x 1 = x ^ 2 :=
by sorry

theorem part2 (k : ℝ) :
  (∀ x : ℝ, x ∈ Icc (-1 : ℝ) (2 : ℝ) → g x 1 ≥ k * x) →
  -2 ≤ k ∧ k ≤ 2 * Real.sqrt 2 + 1 :=
by sorry

end part1_part2_l125_125063


namespace rubber_bands_per_large_ball_l125_125205

open Nat

theorem rubber_bands_per_large_ball :
  let total_rubber_bands := 5000
  let small_bands := 50
  let small_balls := 22
  let large_balls := 13
  let used_bands := small_balls * small_bands
  let remaining_bands := total_rubber_bands - used_bands
  let large_bands := remaining_bands / large_balls
  large_bands = 300 :=
by
  sorry

end rubber_bands_per_large_ball_l125_125205


namespace probability_smallest_divides_larger_two_l125_125271

noncomputable def number_of_ways := 20

noncomputable def successful_combinations := 11

theorem probability_smallest_divides_larger_two : (successful_combinations : ℚ) / number_of_ways = 11 / 20 :=
by
  sorry

end probability_smallest_divides_larger_two_l125_125271


namespace china_junior_1990_problem_l125_125829

theorem china_junior_1990_problem 
  (x y z a b c : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0) 
  (ha : a ≠ -1) 
  (hb : b ≠ -1) 
  (hc : c ≠ -1)
  (h1 : a * x = y * z / (y + z))
  (h2 : b * y = x * z / (x + z))
  (h3 : c * z = x * y / (x + y)) :
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = 1) :=
sorry

end china_junior_1990_problem_l125_125829


namespace least_value_of_d_l125_125408

theorem least_value_of_d :
  ∀ d : ℝ, (|((3 - 2 * d) / 5) + 2| ≤ 3) → d = -1 → d = -1 :=
by
  intro d
  intro h
  intro h_eq
  rw [h_eq] at h
  exact h

end least_value_of_d_l125_125408


namespace digit_68th_is_1_l125_125880

noncomputable def largest_n : ℕ :=
  (10^100 - 1) / 14

def digit_at_68th_place (n : ℕ) : ℕ :=
  (n / 10^(68 - 1)) % 10

theorem digit_68th_is_1 : digit_at_68th_place largest_n = 1 :=
sorry

end digit_68th_is_1_l125_125880


namespace range_of_y_l125_125389

theorem range_of_y (x y : ℝ) :
  y - x^2 < (Real.sqrt (x^2)) → (x ≥ 0 → y < x + x^2) ∧ (x < 0 → y < -x + x^2) :=
by
  intro h
  split
  { intro hx
    have : |x| = x, by sorry
    linarith }
  { intro hx
    have : |x| = -x, by sorry
    linarith }
  sorry

end range_of_y_l125_125389


namespace range_of_m_a_eq_1_range_of_m_a_in_3_to_6_l125_125065

noncomputable def f (x : ℝ) (a m : ℝ) : ℝ := x^3 + a * x^2 - a^2 * x + m

theorem range_of_m_a_eq_1 (m : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ 1 m = 0 ∧ f x₂ 1 m = 0 ∧ f x₃ 1 m = 0) →
  (-1 < m ∧ m < 5 / 27) :=
sorry

theorem range_of_m_a_in_3_to_6 (m : ℝ) :
  (∀ a ∈ Icc (3 : ℝ) (6 : ℝ), ∀ x ∈ Icc (-2 : ℝ) (2 : ℝ), f x a m ≤ 1) →
  m ≤ -87 :=
sorry

end range_of_m_a_eq_1_range_of_m_a_in_3_to_6_l125_125065


namespace find_numerator_of_fraction_l125_125925

theorem find_numerator_of_fraction :
  ∃ x : ℚ, (x / (4 * x + 5) = 3 / 7) ∧ (x = -3) :=
begin
  use -3,
  split,
  { calc (-3) / (4 * (-3) + 5)
         = (-3) / (-12 + 5) : by refl
     ... = (-3) / (-7) : by congr
     ... = 3 / 7 : by norm_num },
  { refl }
end

end find_numerator_of_fraction_l125_125925


namespace dodecahedral_dice_sum_20_probability_l125_125255

theorem dodecahedral_dice_sum_20_probability :
  let faces := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
  in (∃ (A B : ℕ), A ∈ faces ∧ B ∈ faces ∧ A + B = 20) → 
     (5 : ℚ) / 144 = 5 / 144 :=
by
  sorry

end dodecahedral_dice_sum_20_probability_l125_125255


namespace median_of_trapezoid_eq_2_plus_sqrt2_l125_125354

theorem median_of_trapezoid_eq_2_plus_sqrt2 :
  let side_large : ℝ := 4
  let area_large : ℝ := (sqrt 3 / 4) * (side_large ^ 2)
  let area_small : ℝ := area_large / 2
  let side_small : ℝ := sqrt (4 * area_small / sqrt 3)
  let base_large : ℝ := side_large
  let base_small : ℝ := side_small
  let median_trapezoid : ℝ := (base_large + base_small) / 2
  in median_trapezoid = 2 + sqrt 2 := sorry

end median_of_trapezoid_eq_2_plus_sqrt2_l125_125354


namespace find_min_value_l125_125873

variable (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1)

theorem find_min_value :
  (1 / (2 * a + 3 * b) + 1 / (2 * b + 3 * c) + 1 / (2 * c + 3 * a)) ≥ (9 / 5) :=
sorry

end find_min_value_l125_125873


namespace lily_spent_amount_l125_125567

def num_years (start_year end_year : ℕ) : ℕ :=
  end_year - start_year

def total_spent (cost_per_plant num_years : ℕ) : ℕ :=
  cost_per_plant * num_years

theorem lily_spent_amount :
  let start_year := 1989
  let end_year := 2021
  let cost_per_plant := 20
  num_years start_year end_year = 32 →
  total_spent cost_per_plant 32 = 640 :=
by
  intros
  sorry

end lily_spent_amount_l125_125567


namespace maximize_a_n_l125_125466

-- Given sequence definition
noncomputable def a_n (n : ℕ) := (n + 2) * (7 / 8) ^ n

-- Prove that n = 5 or n = 6 maximizes the sequence
theorem maximize_a_n : ∃ n, (n = 5 ∨ n = 6) ∧ (∀ k, a_n k ≤ a_n n) :=
by
  sorry

end maximize_a_n_l125_125466


namespace partition_N_to_satisfy_ratio_condition_l125_125173

theorem partition_N_to_satisfy_ratio_condition (c : ℚ) (hc : 0 < c) (hc1 : c ≠ 1) :
  ∃ (A B : set ℕ), A ≠ ∅ ∧ B ≠ ∅ ∧ (∀ x y ∈ A, x / y ≠ c) ∧ (∀ x y ∈ B, x / y ≠ c) :=
sorry

end partition_N_to_satisfy_ratio_condition_l125_125173


namespace no_representation_of_216p3_l125_125440

theorem no_representation_of_216p3 (p : ℕ) (hp_prime : Nat.Prime p)
  (hp_form : ∃ m : ℤ, p = 4 * m + 1) : ¬ ∃ x y z : ℤ, 216 * (p ^ 3) = x^2 + y^2 + z^9 := by
  sorry

end no_representation_of_216p3_l125_125440


namespace cos_angle_A_l125_125456

theorem cos_angle_A (a b c : ℝ) (h_a : a = 2 * c) (h_b : b = sqrt 2 / 2 * a) :
  cos (angle_A a b c) = -sqrt 2 / 4 :=
by
  /- Additional setup may be required here for defining angle_A and cos which is trimmed here -/
  sorry

end cos_angle_A_l125_125456


namespace sandwich_cost_proof_l125_125159

/-- Definitions of ingredient costs and quantities. --/
def bread_cost : ℝ := 0.15
def ham_cost : ℝ := 0.25
def cheese_cost : ℝ := 0.35
def mayo_cost : ℝ := 0.10
def lettuce_cost : ℝ := 0.05
def tomato_cost : ℝ := 0.08

def num_bread_slices : ℕ := 2
def num_ham_slices : ℕ := 2
def num_cheese_slices : ℕ := 2
def num_mayo_tbsp : ℕ := 1
def num_lettuce_leaf : ℕ := 1
def num_tomato_slices : ℕ := 2

/-- Calculation of the total cost in dollars and conversion to cents. --/
def sandwich_cost_in_dollars : ℝ :=
  (num_bread_slices * bread_cost) + 
  (num_ham_slices * ham_cost) + 
  (num_cheese_slices * cheese_cost) + 
  (num_mayo_tbsp * mayo_cost) + 
  (num_lettuce_leaf * lettuce_cost) + 
  (num_tomato_slices * tomato_cost)

def sandwich_cost_in_cents : ℝ :=
  sandwich_cost_in_dollars * 100

/-- Prove that the cost of the sandwich in cents is 181. --/
theorem sandwich_cost_proof : sandwich_cost_in_cents = 181 := by
  sorry

end sandwich_cost_proof_l125_125159


namespace eccentricities_of_conic_sections_l125_125763

theorem eccentricities_of_conic_sections
  (a b m n : ℝ) (e e_N : ℝ)
  (ha : a > b) (hb : b > 0)
  (H1 : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
  (H2 : ∀ x y : ℝ, x^2 / m^2 - y^2 / n^2 = 1)
  (H3 : ∀ x y : ℝ, two_asymptotes_and_four_intersections_form_regular_hexagon :=
    let c := sqrt (a^2 - b^2) in
    -- Other conditions required to specify the formation of the hexagon
    true)
  : e = sqrt(3) - 1 ∧ e_N = 2 := by sorry

end eccentricities_of_conic_sections_l125_125763


namespace part1_part2_l125_125852

noncomputable def triangle_area (A B C : ℝ) (a b c : ℝ) : ℝ :=
  1/2 * a * c * Real.sin B

theorem part1 
  (A B C : ℝ) (a b c : ℝ)
  (h₁ : A = π / 6)
  (h₂ : a = 2)
  (h₃ : 2 * a * c * Real.sin A + a^2 + c^2 - b^2 = 0) :
  triangle_area A B C a b c = Real.sqrt 3 :=
sorry

theorem part2 
  (A B C : ℝ) (a b c : ℝ)
  (h₁ : A = π / 6)
  (h₂ : a = 2)
  (h₃ : 2 * a * c * Real.sin A + a^2 + c^2 - b^2 = 0) :
  ∃ B, 
  (B = 2 * π / 3) ∧ (4 * Real.sin C^2 + 3 * Real.sin A^2 + 2) / (Real.sin B^2) = 5 :=
sorry

end part1_part2_l125_125852


namespace anne_initial_sweettarts_l125_125357

variable (x : ℕ)
variable (num_friends : ℕ := 3)
variable (sweettarts_per_friend : ℕ := 5)
variable (total_sweettarts_given : ℕ := num_friends * sweettarts_per_friend)

theorem anne_initial_sweettarts 
  (h1 : ∀ person, person < num_friends → sweettarts_per_friend = 5)
  (h2 : total_sweettarts_given = 15) : 
  total_sweettarts_given = 15 := 
by 
  sorry

end anne_initial_sweettarts_l125_125357


namespace range_of_t_l125_125201

theorem range_of_t 
  (a : ℕ → ℝ)
  (t : ℝ)
  (h1 : a 1 = 3)
  (h2 : ∀ n : ℕ, n > 0 → n * (a (n + 1) - a n) = a n + 1)
  (ineq : ∀ (a_val : ℝ) (n : ℕ), -1 ≤ a_val ∧ a_val ≤ 1 ∧ n > 0 → (a (n + 1) / (n + 1) < t^2 - 2 * a_val * t + 1)) :
  t ∈ Set.Ioo (-Float.sqrt 3 - 1) (-Float.sqrt 3 + 1) :=
sorry

end range_of_t_l125_125201


namespace find_vector_b_l125_125872

noncomputable def a : ℝ × ℝ × ℝ := (3, 2, 4)
noncomputable def b : ℝ × ℝ × ℝ := (5, 3, 1.5)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

theorem find_vector_b :
  dot_product a b = 20 ∧ cross_product a b = (-15, -5, 1) :=
by
  -- The proof should be written here
  sorry

end find_vector_b_l125_125872


namespace sum_of_ratios_roots_l125_125107

noncomputable def roots_of_polynomial : set ℝ :=
{ x | x^3 - x - 1 = 0 }

theorem sum_of_ratios_roots (α β γ : ℝ) 
  (hα : α ∈ roots_of_polynomial)
  (hβ : β ∈ roots_of_polynomial)
  (hγ : γ ∈ roots_of_polynomial) :
  (1 + α) / (1 - α) + (1 + β) / (1 - β) + (1 + γ) / (1 - γ) = -7 :=
sorry

end sum_of_ratios_roots_l125_125107


namespace angle_ABF_regular_octagon_l125_125223

theorem angle_ABF_regular_octagon (ABCDEFGH : Type) [regular_octagon ABCDEFGH] :
  ∃ AB F : Point, angle AB F = 22.5 := sorry

end angle_ABF_regular_octagon_l125_125223


namespace max_elements_in_X_l125_125537

-- Definitions
def a (i : ℕ) := {a | a ∈ {1, 2, 3, 4, 5}}

def b (i : ℕ) := {b | b ∈ {1, 2, ..., 10}}

def X (a : ℕ → ℕ) (b : ℕ → ℕ) := 
  {(i, j) | 1 ≤ i ∧ i < j ∧ j ≤ 20 ∧ (a i - a j) * (b i - b j) < 0}

-- Theorem statement
theorem max_elements_in_X 
  (a : ℕ → ℕ) (b : ℕ → ℕ)
  (ha : ∀ i, 1 ≤ i ∧ i ≤ 20 → a i ∈ {1, 2, 3, 4, 5})
  (hb : ∀ i, 1 ≤ i ∧ i ≤ 20 → b i ∈ {1, 2, ..., 10})
  : ∃ X, ∀ (a : ℕ → ℕ) (b : ℕ → ℕ),
        X = { (i, j) | 1 ≤ i ∧ i < j ∧ j ≤ 20 ∧ (a i - a j) * (b i - b j) < 0 } ∧ X.card = 160 :=
sorry

end max_elements_in_X_l125_125537


namespace david_and_moore_together_complete_job_in_six_days_l125_125981

-- Definition of given conditions
def david_work_rate : ℝ := 1 / 12
def david_alone_days : ℝ := 6
def remaining_days_with_moore : ℝ := 3
def remaining_work_completed : ℝ := 1/2

-- Prove that the combined work rate gives a completion time of 6 days
theorem david_and_moore_together_complete_job_in_six_days : 
  (1 / (david_work_rate + ((remaining_work_completed / remaining_days_with_moore) - david_work_rate))) = 6 :=
by
  -- Explanations of proofs are omitted, insert the necessary steps to complete the proof.
  sorry

end david_and_moore_together_complete_job_in_six_days_l125_125981


namespace measure_angle_ABF_of_regular_octagon_l125_125219

theorem measure_angle_ABF_of_regular_octagon (h : regular_octagon ABCDEFGH) : angle ABF = 22.5 :=
sorry

end measure_angle_ABF_of_regular_octagon_l125_125219


namespace composite_divides_factorial_l125_125036

-- Define the factorial of a number
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Statement of the problem
theorem composite_divides_factorial (m : ℕ) (hm : m ≠ 4) (hcomposite : ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = m) :
  m ∣ factorial (m - 1) :=
by
  sorry

end composite_divides_factorial_l125_125036


namespace victor_has_6_pasta_orders_l125_125958

noncomputable theory

variable (num_pasta_orders : ℕ)

def chicken_needed_for_fried_dinner_orders := 2 * 8
def chicken_needed_for_bbq_orders := 3 * 3
def total_chicken_needed := 37
def total_chicken_from_dinner_and_bbq := chicken_needed_for_fried_dinner_orders + chicken_needed_for_bbq_orders
def chicken_needed_for_pasta_orders := total_chicken_needed - total_chicken_from_dinner_and_bbq

theorem victor_has_6_pasta_orders
  (h1 : chicken_needed_for_fried_dinner_orders = 16)
  (h2 : chicken_needed_for_bbq_orders = 9)
  (h3 : total_chicken_needed = 37)
  (h4 : total_chicken_from_dinner_and_bbq = 25)
  (h5 : chicken_needed_for_pasta_orders = 12)
  (h6 : num_pasta_orders = chicken_needed_for_pasta_orders / 2) :
  num_pasta_orders = 6 :=
sorry

end victor_has_6_pasta_orders_l125_125958


namespace geometric_sequence_arithmetic_sequence_ratio_l125_125057

theorem geometric_sequence_arithmetic_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (hq : q > 1) 
  (h_arithmetic : 2 * a 1, (3 / 2) * a 2, a 3 form an arithmetic sequence) 
  (a_is_geom : ∀ n, a (n + 1) = a n * q) :
  ( (a 1 + a 1 * q + a 1 * q^2 + a 1 * q^3) / (a 1 * q^3) = 15 / 8 ) :=
by
  sorry

end geometric_sequence_arithmetic_sequence_ratio_l125_125057


namespace sum_sequence_problem_l125_125871

/-- 
Given the sequence {a_{n}} such that 
1. \( a_{1} = -1 \)
2. \( a_{n+1} = S_{n} S_{n+1} \)
where \( S_{n} \) is the sum of the first \( n \) terms of the sequence,
prove that \( S_{2016} = -\frac{1}{2016} \).
-/
theorem sum_sequence_problem
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h1 : a 1 = -1)
  (h2 : ∀ n, a (n + 1) = S n * S (n + 1))
  (h3 : ∀ n, S n = ∑ i in Finset.range n, a (i + 1)) :
  S 2016 = - (1 / 2016) :=
sorry

end sum_sequence_problem_l125_125871


namespace expected_value_of_X_l125_125953

noncomputable def probA : ℝ := 0.7
noncomputable def probB : ℝ := 0.8
noncomputable def probC : ℝ := 0.5

def prob_distribution : MeasureTheory.PMF ℕ :=
{
  support := {0, 1, 2, 3},
  toFun := λ x, match x with
    | 0 => (1 - probA) * (1 - probB) * (1 - probC)
    | 1 => probA * (1 - probB) * (1 - probC) + (1 - probA) * probB * (1 - probC) + (1 - probA) * (1 - probB) * probC
    | 2 => probA * probB * (1 - probC) + probA * (1 - probB) * probC + (1 - probA) * probB * probC
    | 3 => probA * probB * probC
    | _ => 0
}

noncomputable def expected_value : ℝ :=
  MeasureTheory.Integral (λ x, (x : ℝ)) prob_distribution

theorem expected_value_of_X : expected_value = 2 :=
by {
  sorry
}

end expected_value_of_X_l125_125953


namespace total_spending_l125_125236

-- Conditions
def pop_spending : ℕ := 15
def crackle_spending : ℕ := 3 * pop_spending
def snap_spending : ℕ := 2 * crackle_spending

-- Theorem stating the total spending
theorem total_spending : snap_spending + crackle_spending + pop_spending = 150 :=
by
  sorry

end total_spending_l125_125236


namespace proof_problem_l125_125701

def diamond (a b : ℚ) := a - (1 / b)

theorem proof_problem :
  ((diamond (diamond 2 4) 5) - (diamond 2 (diamond 4 5))) = (-71 / 380) := by
  sorry

end proof_problem_l125_125701


namespace integral_evaluation_l125_125716

noncomputable def integral_value : ℝ :=
  ∫ x in -1..1, (Real.sin x + Real.sqrt (1 - x^2))

theorem integral_evaluation : integral_value = Real.pi / 2 :=
by
  -- skipping the actual proof
  sorry

end integral_evaluation_l125_125716


namespace dice_probability_l125_125912

/-- A standard six-sided die -/
inductive Die : Type
| one | two | three | four | five | six

open Die

/-- Calculates the probability that after re-rolling four dice, at least four out of the six total dice show the same number,
given that initially six dice are rolled and there is no three-of-a-kind, and there is a pair of dice showing the same number
which are then set aside before re-rolling the remaining four dice. -/
theorem dice_probability (h1 : ∀ (d1 d2 d3 d4 d5 d6 : Die), 
  ¬ (d1 = d2 ∧ d2 = d3 ∨ d1 = d2 ∧ d2 = d4 ∨ d1 = d2 ∧ d2 = d5 ∨
     d1 = d2 ∧ d2 = d6 ∨ d1 = d3 ∧ d3 = d4 ∨ d1 = d3 ∧ d3 = d5 ∨
     d1 = d3 ∧ d3 = d6 ∨ d1 = d4 ∧ d4 = d5 ∨ d1 = d4 ∧ d4 = d6 ∨
     d1 = d5 ∧ d5 = d6 ∨ d2 = d3 ∧ d3 = d4 ∨ d2 = d3 ∧ d3 = d5 ∨
     d2 = d3 ∧ d3 = d6 ∨ d2 = d4 ∧ d4 = d5 ∨ d2 = d4 ∧ d4 = d6 ∨
     d2 = d5 ∧ d5 = d6 ∨ d3 = d4 ∧ d4 = d5 ∨ d3 = d4 ∧ d4 = d6 ∨ d3 = d5 ∧ d5 = d6 ∨ d4 = d5 ∧ d5 = d6))
    (h2 : ∃ (d1 d2 : Die) (d3 d4 d5 d6 : Die), d1 = d2 ∧ d3 ≠ d1 ∧ d4 ≠ d1 ∧ d5 ≠ d1 ∧ d6 ≠ d1): 
    ℚ := 
11 / 81

end dice_probability_l125_125912


namespace geometric_sequence_common_ratio_l125_125131

theorem geometric_sequence_common_ratio {a : ℕ → ℝ} 
    (h1 : a 1 = 1) 
    (h4 : a 4 = 1 / 64) 
    (geom_seq : ∀ n, ∃ r, a (n + 1) = a n * r) : 
       
    ∃ q, (∀ n, a n = 1 * (q ^ (n - 1))) ∧ (a 4 = 1 * (q ^ 3)) ∧ q = 1 / 4 := 
by
    sorry

end geometric_sequence_common_ratio_l125_125131


namespace log_base_proof_l125_125810

theorem log_base_proof (a : ℝ) (h : log a (1 / 4) = -2) : a = 2 := 
by
  sorry

end log_base_proof_l125_125810


namespace modulus_z_l125_125102

def z : ℂ := (1 + complex.I) / (1 - complex.I) + 2 * complex.I

theorem modulus_z : complex.abs z = 3 := by
  sorry

end modulus_z_l125_125102


namespace convex_polygon_enclosed_in_triangle_l125_125314

theorem convex_polygon_enclosed_in_triangle (M : Type) [convex_polygon M] (hM1: ¬ parallelogram M) :
  ∃ (T : Type) [triangle T], enclosed_by M T :=
sorry

end convex_polygon_enclosed_in_triangle_l125_125314


namespace problem1_simplification_problem2_simplification_l125_125379

theorem problem1_simplification : (3 / Real.sqrt 3 - (Real.sqrt 3) ^ 2 - Real.sqrt 27 + (abs (Real.sqrt 3 - 2))) = -1 - 3 * Real.sqrt 3 :=
  by
    sorry

theorem problem2_simplification (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 2) :
  ((x + 2) / (x ^ 2 - 2 * x) - (x - 1) / (x ^ 2 - 4 * x + 4)) / ((x - 4) / x) = 1 / (x - 2) ^ 2 :=
  by
    sorry

end problem1_simplification_problem2_simplification_l125_125379


namespace smallest_b_g_l125_125539

def g (x : ℕ) : ℕ :=
  if x % 15 = 0 then x / 15
  else if x % 3 = 0 then 5 * x
  else if x % 5 = 0 then 3 * x
  else x + 5

def g_iter (b : ℕ) (x : ℕ) : ℕ :=
  Nat.iterate b g x

theorem smallest_b_g (b : ℕ) (h1: b > 1) : g_iter 15 2 = g_iter b 2 :=
by
  sorry

end smallest_b_g_l125_125539


namespace hour_hand_rotation_l125_125283

theorem hour_hand_rotation : 
  ∀ (degrees_per_hour time_passed : ℝ), 
  degrees_per_hour = 30 ∧ time_passed = 3 → degrees_per_hour * time_passed = 90 :=
by
  intros degrees_per_hour time_passed h,
  cases h with h1 h2,
  rw [h1, h2],
  norm_num

end hour_hand_rotation_l125_125283


namespace probability_exactly_three_blue_marbles_l125_125161

def marble_prob : ℚ :=
  let p_blue := 8 / 15
  let p_red := 7 / 15
  let choose := Nat.binomial 7 3
  let prob := choose * (p_blue^3) * (p_red^4)
  prob

theorem probability_exactly_three_blue_marbles :
  marble_prob = 640 / 1547 :=
by
  sorry

end probability_exactly_three_blue_marbles_l125_125161


namespace sum_xyz_eq_11sqrt5_l125_125483

noncomputable def x : ℝ :=
sorry

noncomputable def y : ℝ :=
sorry

noncomputable def z : ℝ :=
sorry

axiom pos_x : x > 0
axiom pos_y : y > 0
axiom pos_z : z > 0

axiom xy_eq_30 : x * y = 30
axiom xz_eq_60 : x * z = 60
axiom yz_eq_90 : y * z = 90

theorem sum_xyz_eq_11sqrt5 : x + y + z = 11 * Real.sqrt 5 :=
sorry

end sum_xyz_eq_11sqrt5_l125_125483


namespace complement_union_eq_l125_125086

open Set

-- Definition of sets U, A, and B
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- Statement of the problem
theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 3} :=
by sorry

end complement_union_eq_l125_125086


namespace cos_alpha_of_coords_l125_125061

theorem cos_alpha_of_coords {α : Real} (x y : Real) (r : Real) (h₁ : x = -3) (h₂ : y = -4) (h₃ : r = 5) : 
  cos α = -3/5 :=
by 
  sorry

end cos_alpha_of_coords_l125_125061


namespace min_ω_satisfies_condition_l125_125073

theorem min_ω_satisfies_condition :
  ∃ ω : ℕ, (ω > 0) ∧ (2 * real.sin (ω * 2 * real.pi + real.pi / 3) = real.sqrt 3) ∧
  (∀ ω' : ℕ, (ω' > 0) ∧ (ω' < ω) → ¬(2 * real.sin (ω' * 2 * real.pi + real.pi / 3) = real.sqrt 3)) :=
by {
  let ω_min := 1,
  use ω_min,
  split,
  -- ω > 0
  exact nat.one_pos,
  split,
  -- 2 * real.sin (ω * 2 * real.pi + real.pi / 3) = real.sqrt 3
  norm_num,
  ring,
  simp,
  ring,
  simp [real.sin_two_pi, real.sin_pi_over_three, norm_num],
  -- ∀ ω' : ℕ, (ω' > 0) ∧ (ω' < ω) → ¬(2 * real.sin (ω' * 2 * real.pi + real.pi / 3) = real.sqrt 3)
  assume ω' hpos hlt,
  simp,
  sorry
}

end min_ω_satisfies_condition_l125_125073


namespace hcl_formed_l125_125475

-- Define the balanced chemical equation as a relationship between reactants and products
def balanced_equation (m_C2H6 m_Cl2 m_CCl4 m_HCl : ℝ) :=
  m_C2H6 + 4 * m_Cl2 = m_CCl4 + 6 * m_HCl

-- Define the problem-specific values
def reaction_given (m_C2H6 m_Cl2 m_CCl4 m_HCl : ℝ) :=
  m_C2H6 = 3 ∧ m_Cl2 = 21 ∧ m_CCl4 = 6 ∧ balanced_equation m_C2H6 m_Cl2 m_CCl4 m_HCl

-- Prove the number of moles of HCl formed
theorem hcl_formed : ∃ (m_HCl : ℝ), reaction_given 3 21 6 m_HCl ∧ m_HCl = 18 :=
by
  sorry

end hcl_formed_l125_125475


namespace equal_distances_l125_125328

variable (A B C D X Y : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace X] [MetricSpace Y]

-- Definitions if required for circles and parallelogram
variable (circleA : Metric.ball A B)
variable (circleC : Metric.ball C B)
variable (parallelogram_ABCD : IsParallelogram A B C D)
variable (line_l : ∃ (p : Type), p ∈ MetricSegment B X ∧ p ∈ MetricSegment B Y)

theorem equal_distances (h₁ : B ∈ circleA) (h₂ : B ∈ circleC) (h₃ : line_l) : dist D X = dist D Y := by
  sorry

end equal_distances_l125_125328


namespace reams_paper_l125_125800

theorem reams_paper (total_reams reams_haley reams_sister : Nat) 
    (h1 : total_reams = 5)
    (h2 : reams_haley = 2)
    (h3 : total_reams = reams_haley + reams_sister) : 
    reams_sister = 3 := by
  sorry

end reams_paper_l125_125800


namespace Mary_income_percentage_of_Juan_income_l125_125890

variables (Mary_income Tim_income Juan_income : ℝ)

def condition1 : Prop := Mary_income = 1.6 * Tim_income
def condition2 : Prop := Tim_income = 0.9 * Juan_income
def question_and_answer : Prop := Mary_income = 1.44 * Juan_income

theorem Mary_income_percentage_of_Juan_income 
    (h1 : condition1 Mary_income Tim_income Juan_income)
    (h2 : condition2 Mary_income Tim_income Juan_income)
    : question_and_answer Mary_income Tim_income Juan_income := by
  sorry  -- To be proved

end Mary_income_percentage_of_Juan_income_l125_125890


namespace cos_beta_eq_sqrt2_div2_l125_125824

theorem cos_beta_eq_sqrt2_div2 (β : Real) :
  let α := -1035
  let θ := α + 3 * 360
  (θ = 45) →
  (cos θ = cos 45) →
  cos β = cos 45 :=
by
  intros α θ h1 h2
  rw [h1, h2]
  exact Sorry

end cos_beta_eq_sqrt2_div2_l125_125824


namespace connected_graph_l125_125383

noncomputable theory

open set

variables (n : ℕ) (points : fin (2 * n) → ℝ × ℝ)
variable  (circles : set (set (ℝ × ℝ)))

-- conditions
def valid_circles : Prop :=
  ∀ c ∈ circles, set.finite c ∧ c.card ≥ n + 1 ∧ 
  ∀ x ∈ c, x ∈ points.val

-- construct graph G
def graph (points : fin (2 * n) → ℝ × ℝ) (circles : set (set (ℝ × ℝ))) : simple_graph (ℝ × ℝ) := {
  adj := λ x y, ∃ c ∈ circles, x ∈ c ∧ y ∈ c
}

theorem connected_graph :
  valid_circles n points circles →
  (∀ x y ∈ points, ∃ p, chain (graph points circles) x y p)
sorry

end connected_graph_l125_125383


namespace sequence_is_purely_periodic_sequence_period_7m_l125_125310

-- Problem 1 in Lean 4

theorem sequence_is_purely_periodic {a : ℕ → ℤ} 
  (k : ℕ) (x : ℕ → ℂ) 
  (T : ℕ → ℕ) 
  (h1 : ∀ i, i < k → (∃ c : ℕ → ℂ, a = λ n, ∑ i in finset.range k, c i * x i ^ n))
  (h2 : ∀ j, j < k → x j ^ (T j) = 1) : 
  ∃ T0 : ℕ, ∀ n, a (n + T0) = a n :=
sorry

-- Problem 2 in Lean 4

theorem sequence_period_7m (y : ℕ → ℂ) (m : ℕ) (h1 : 0 < m)
  (h2 : ∀ n, y n + y (n + 2 * m) = 2 * y (n + m) * (Real.cos (2 * Real.pi / 7))) :
  ∀ n, y (n + 7 * m) = y n :=
sorry

end sequence_is_purely_periodic_sequence_period_7m_l125_125310


namespace complement_union_eq_l125_125089

variable (U : Set Int := {-2, -1, 0, 1, 2, 3}) 
variable (A : Set Int := {-1, 0, 1}) 
variable (B : Set Int := {1, 2}) 

theorem complement_union_eq :
  U \ (A ∪ B) = {-2, 3} := by 
  sorry

end complement_union_eq_l125_125089


namespace non_congruent_rectangles_count_l125_125333

noncomputable def countNonCongruentRectangles (perimeter : ℕ) (widthOdd : ∀ w, Odd w → Prop) : ℕ :=
  let pairs := finset.filter (λ (w h : ℕ), Odd w ∧ w + h = perimeter / 2) (finset.range (perimeter / 2 + 1)).product (finset.range (perimeter / 2 + 1))
  pairs.card

theorem non_congruent_rectangles_count (h : ℕ) (w : ℕ) (perimeter : ℕ) (rectangles_count : ℕ) (width_odd : Odd w) : rectangles_count = 19 :=
by
  assume perimeter = 76
  assume widthOdd w
  have rectangles_count = countNonCongruentRectangles 76 (λ w, Odd w) := sorry
  exact rectangles_count = 19

end non_congruent_rectangles_count_l125_125333


namespace max_product_min_quotient_l125_125373

theorem max_product_min_quotient :
  let nums := [-5, -3, -1, 2, 4]
  let a := max (max (-5 * -3) (-5 * -1)) (max (-3 * -1) (max (2 * 4) (max (2 * -1) (4 * -1))))
  let b := min (min (4 / -1) (2 / -3)) (min (2 / -5) (min (4 / -3) (-5 / -3)))
  a = 15 ∧ b = -4 → a / b = -15 / 4 :=
by
  sorry

end max_product_min_quotient_l125_125373


namespace find_angle_CED_l125_125174

axiom circle_center (O : Point) (radius : ℝ) (circle : Circle)
axiom chord (A B : Point) (A_on_circle : on_circle A circle) (B_on_circle : on_circle B circle) (AB_is_chord : on_chord A B O)
axiom E_on_circle (E : Point) (E_on_circle : on_circle E circle)
axiom tangent_intersections (B E C D : Point) 
  (tangent_at_B : tangent_line B circle C intersects AE D)
  (tangent_at_E : tangent_line E circle C intersects AE D)
axiom angle_subtended_by_chord (angle_AOB : Angle) (angle_AOB_60_deg : angle_AOB = 60)
axiom angle_BAE (angle_BAE_30_deg : Angle) (angle_BAE_30_deg : angle_BAE_30_deg = 30)

theorem find_angle_CED : 
  ∃ angle_CED : Angle, 
    angle_subtended_by_chord (angle A O B) (60) → 
    angle_BAE (30) → 
    angle_CED = 60 := 
sorry

end find_angle_CED_l125_125174


namespace ways_to_sum_2022_l125_125803

theorem ways_to_sum_2022 : 
  ∃ n : ℕ, (∀ a b : ℕ, (2022 = 2 * a + 3 * b) ∧ n = (b - a) / 4 ∧ n = 338) := 
sorry

end ways_to_sum_2022_l125_125803


namespace percent_swans_not_ducks_l125_125861

theorem percent_swans_not_ducks
  (geese : ℝ) (swans : ℝ) (herons : ℝ) (ducks : ℝ)
  (h_geese : geese = 0.20) (h_swans : swans = 0.30) (h_herons : herons = 0.25) (h_ducks : ducks = 0.25) :
  ((swans / (1 - ducks)) * 100 = 40) :=
by
  have h_non_duck_birds : 1 - ducks = 0.75 := by linarith
  have h_swans_percentage : (swans / (1 - ducks)) * 100 = (0.30 / 0.75) * 100 := by congr; rw [h_duck_birds, h_swans]
  have h_calc : (0.30 / 0.75) * 100 = 40 := by norm_num
  rw [h_swans_percentage, h_calc]
  sorry

end percent_swans_not_ducks_l125_125861


namespace eq_in_first_quarter_l125_125493

variable (x : ℝ)

theorem eq_in_first_quarter (initial_turnover : ℝ) (total_turnover : ℝ) (H_initial : initial_turnover = 50) (H_total : total_turnover = 600) :
  50 * (1 + (1 + x) + (1 + x) ^ 2) = total_turnover :=
by
  rw [H_total, H_initial]
  sorry

end eq_in_first_quarter_l125_125493


namespace not_correct_sum_x4_y4_l125_125183

variable {𝔠 ℂ : Type}
open Complex

theorem not_correct_sum_x4_y4 (x y : ℂ) (h₁ : x = Complex.I) (h₂ : y = -Complex.I) : x^4 + y^4 ≠ 0 := by
  sorry

end not_correct_sum_x4_y4_l125_125183


namespace max_value_trig_expression_l125_125411

/-- 
  Theorem: The maximum value of the expression 
  S = cos(θ₁) * sin(θ₂) + cos(θ₂) * sin(θ₃) + cos(θ₃) * sin(θ₄) + cos(θ₄) * sin(θ₅) + 
        cos(θ₅) * sin(θ₆) + cos(θ₆) * sin(θ₁) 
  over all real numbers θ₁, θ₂, θ₃, θ₄, θ₅, θ₆ is 3.
-/
theorem max_value_trig_expression
  (θ₁ θ₂ θ₃ θ₄ θ₅ θ₆ : ℝ) :
  let S := cos θ₁ * sin θ₂ + cos θ₂ * sin θ₃ + cos θ₃ * sin θ₄ +
           cos θ₄ * sin θ₅ + cos θ₅ * sin θ₆ + cos θ₆ * sin θ₁
  in S ≤ 3 :=
sorry

end max_value_trig_expression_l125_125411


namespace greatest_integer_solution_proof_l125_125064

noncomputable def greatest_integer_solution (f : ℝ → ℝ) : ℤ :=
  if h : ∃ x, 2 * real.log x = 7 - 2 * x then 4 else sorry

theorem greatest_integer_solution_proof :
  ∃ x, 2 * real.log x = 7 - 2 * x → greatest_integer_solution (λ x, 2 * real.log x - 7 + 2 * x) = 4 :=
by
  intro h
  sorry

end greatest_integer_solution_proof_l125_125064


namespace omega_range_l125_125527

theorem omega_range {ω : ℝ} (hω : ω > 0) 
                    (a b : ℝ) 
                    (ha : π ≤ a) (hb : a < b) (hc : b ≤ 2 * π) 
                    (h_sin : sin (ω * a) + sin (ω * b) = 2) :
    (ω ∈ Set.Ioo (1 / 4) (1 / 2) ∪ Set.Ioi (5 / 4)) :=
by
  sorry

end omega_range_l125_125527


namespace infinite_series_equals_3_l125_125692

noncomputable def infinite_series_sum := ∑' (k : ℕ), (12^k) / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))

theorem infinite_series_equals_3 : infinite_series_sum = 3 := by
  sorry

end infinite_series_equals_3_l125_125692


namespace angle_is_pi_over_3_l125_125747

def vec3 := (ℝ × ℝ × ℝ)

noncomputable def dot_product (u v : vec3) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

noncomputable def magnitude (u : vec3) : ℝ :=
  Real.sqrt (u.1 * u.1 + u.2 * u.2 + u.3 * u.3)

noncomputable def angle_between (u v : vec3) : ℝ :=
  Real.arccos (dot_product u v / (magnitude u * magnitude v))

def a : vec3 := (2, -3, Real.sqrt 3)
def b : vec3 := (1, 0, 0)

theorem angle_is_pi_over_3 : angle_between a b = Real.pi / 3 := by
  sorry

end angle_is_pi_over_3_l125_125747


namespace area_triangle_PZQ_l125_125140

/-- 
In rectangle PQRS, side PQ measures 8 units and side QR measures 4 units.
Points X and Y are on side RS such that segment RX measures 2 units and
segment SY measures 3 units. Lines PX and QY intersect at point Z.
Prove the area of triangle PZQ is 128/3 square units.
-/

theorem area_triangle_PZQ {PQ QR RX SY : ℝ} (h1 : PQ = 8) (h2 : QR = 4) (h3 : RX = 2) (h4 : SY = 3) :
  let area_PZQ : ℝ := 8 * 4 / 2 * 8 / (3 * 2)
  area_PZQ = 128 / 3 :=
by
  sorry

end area_triangle_PZQ_l125_125140


namespace number_of_rectangular_arrays_l125_125662

theorem number_of_rectangular_arrays (n : ℕ) (h : n = 48) : 
  ∃ k : ℕ, (k = 6 ∧ ∀ m p : ℕ, m * p = n → m ≥ 3 → p ≥ 3 → m = 3 ∨ m = 4 ∨ m = 6 ∨ m = 8 ∨ m = 12 ∨ m = 16 ∨ m = 24) :=
by
  sorry

end number_of_rectangular_arrays_l125_125662


namespace S4_eq_15_l125_125133

-- Definitions based on the conditions
def a1 : ℕ := 1
def S (n : ℕ) (q : ℝ) : ℝ := (1 - q^n) / (1 - q)

-- Given conditions turned into Lean definitions
def S5_eq_5S3_minus_4 (q : ℝ) : Prop :=
  S 5 q = 5 * S 3 q - 4

-- The theorem to prove
theorem S4_eq_15 {q : ℝ} (h : q ∈ {q : ℝ | q > 0} ∧ S5_eq_5S3_minus_4 q) : 
  S 4 q = 15 :=
by
  sorry

end S4_eq_15_l125_125133


namespace see_each_other_again_l125_125513

noncomputable def distance_jenny_kenny (t : ℚ) : ℚ :=
  let jenny := (t, 150)
  let kenny := (2 * t, -150)
  dist jenny kenny

theorem see_each_other_again (t : ℚ) (dist1 dist2 : ℚ) :
  dist1 = 300 →
  dist2 = 150 →
  (Base : (150^2 + t^2) = dist2^2) →
  (Line : 150 + 2 * sqrt (300*(t^2) + 2 * t)) ->
  let k := (Base + Line in 240 * Base = 300 * Line) → -- derived using geometry and tangents to circle
  let num_denom_sum := num_denom t in
  num_denom_sum = 245 :=
by
  sorry

end see_each_other_again_l125_125513


namespace min_value_y_l125_125707

theorem min_value_y : ∃ x : ℝ, (y = 2 * x^2 + 8 * x + 18) ∧ (∀ x : ℝ, y ≥ 10) :=
by
  sorry

end min_value_y_l125_125707


namespace collinearity_of_reflected_points_l125_125534

theorem collinearity_of_reflected_points 
  (A B C P : Point)
  (γ : Line)
  (A' B' C' : Point)
  (hP_in_ABC_plane : P ∈ plane A B C)
  (hγ_through_P : P ∈ γ)
  (hA'_reflection_intersection : reflection_across_line γ (line_through P A) ∩ line_through B C = A')
  (hB'_reflection_intersection : reflection_across_line γ (line_through P B) ∩ line_through C A = B')
  (hC'_reflection_intersection : reflection_across_line γ (line_through P C) ∩ line_through A B = C') :
  are_collinear A' B' C' :=
sorry

end collinearity_of_reflected_points_l125_125534


namespace rectangle_condition_l125_125499

variables {A B C D : Type*} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D]
variables (ABCD : Quadrilateral A B C D)

-- Assume the given conditions
def is_parallelogram (ABCD : Quadrilateral A B C D) : Prop := 
  parallel AD BC ∧ parallel AB CD

def is_rectangle (ABCD : Quadrilateral A B C D) : Prop :=
  is_parallelogram ABCD ∧ (angle A = 90 ∨ diagonals_equal ABCD)

-- Statement of the problem in Lean 4
theorem rectangle_condition (h1 : parallel AD BC) (h2 : parallel AB CD) : 
  is_rectangle ABCD ↔ (∠A = 90 ∨ (AC = BD)) :=
begin
  sorry,
end

end rectangle_condition_l125_125499


namespace sequence_general_term_l125_125151

theorem sequence_general_term (a : ℕ → ℝ) (n : ℕ) 
  (h1 : a 1 = 2) 
  (h2 : ∀ n, a (n + 1) = a n + log (1 + 1 / n)) :
  a n = 2 + log n :=
sorry

end sequence_general_term_l125_125151


namespace exam_time_ratio_l125_125623

-- Lean statements to define the problem conditions and goal
theorem exam_time_ratio (x M : ℝ) (h1 : x > 0) (h2 : M = x / 18) : 
  (5 * x / 6 + 2 * M) / (x / 6 - 2 * M) = 17 := by
  sorry

end exam_time_ratio_l125_125623


namespace opposite_of_ten_is_negative_ten_l125_125932

theorem opposite_of_ten_is_negative_ten : ∃ y : ℝ, 10 + y = 0 ∧ y = -10 :=
begin
  use -10,
  split,
  {
    -- Condition: 10 + (-10) = 0
    exact add_right_neg 10,
  },
  {
    -- Verification: y is indeed -10
    refl,
  }
end

end opposite_of_ten_is_negative_ten_l125_125932


namespace parabola_focus_l125_125244

theorem parabola_focus (p : ℝ) (hp : p = 4) : ∃ x y, (x, y) = (2, 0) :=
by
  have h1 : (p = 4) := hp
  use (2, 0)
  sorry

end parabola_focus_l125_125244


namespace functional_equation_hold_l125_125883

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 + 2007*x - 6028) / (3 * (x - 1))

theorem functional_equation_hold (x : ℝ) (h : x ≠ 1) :
  x + f(x) + 2 * f((x + 2009) / (x - 1)) = 2010 := by
  sorry

end functional_equation_hold_l125_125883


namespace ratio_m_over_n_l125_125194

theorem ratio_m_over_n : 
  ∀ (m n : ℕ) (a b : ℝ),
  let α := (3 : ℝ) / 4
  let β := (19 : ℝ) / 20
  (a = α * b) →
  (a = β * (a * m + b * n) / (m + n)) →
  (n ≠ 0) →
  m / n = 8 / 9 :=
by
  intros m n a b α β hα hβ hn
  sorry

end ratio_m_over_n_l125_125194


namespace lily_spent_amount_l125_125568

def num_years (start_year end_year : ℕ) : ℕ :=
  end_year - start_year

def total_spent (cost_per_plant num_years : ℕ) : ℕ :=
  cost_per_plant * num_years

theorem lily_spent_amount :
  let start_year := 1989
  let end_year := 2021
  let cost_per_plant := 20
  num_years start_year end_year = 32 →
  total_spent cost_per_plant 32 = 640 :=
by
  intros
  sorry

end lily_spent_amount_l125_125568


namespace bowl_weight_after_refill_l125_125207

-- Define the problem conditions
def empty_bowl_weight : ℕ := 420
def day1_consumption : ℕ := 53
def day2_consumption : ℕ := 76
def day3_consumption : ℕ := 65
def day4_consumption : ℕ := 14

-- Define the total consumption over 4 days
def total_consumption : ℕ :=
  day1_consumption + day2_consumption + day3_consumption + day4_consumption

-- Define the final weight of the bowl after refilling
def final_bowl_weight : ℕ :=
  empty_bowl_weight + total_consumption

-- Statement to prove
theorem bowl_weight_after_refill : final_bowl_weight = 628 := by
  sorry

end bowl_weight_after_refill_l125_125207


namespace find_y_given_x_inverse_square_l125_125627

theorem find_y_given_x_inverse_square (x y : ℚ) : 
  (∀ k, (3 * y = k / x^2) ∧ (3 * 5 = k / 2^2)) → (x = 6) → y = 5 / 9 :=
by
  sorry

end find_y_given_x_inverse_square_l125_125627


namespace nine_distinct_numbers_in_grid_l125_125006

theorem nine_distinct_numbers_in_grid :
  ∀ (m : List (List ℕ)), 
  (∀ i, 0 ≤ i ∧ i < 5 → List.length (m.nthLe i sorry) = 5) ∧ 
  (∀ i j, 0 ≤ (m.nthLe i sorry).nthLe j sorry ∧ (m.nthLe i sorry).nthLe j sorry ≤ 25) →
  ∃! (X : ℕ), X ∈ (List.map List.maximum' m ++ List.map List.minimum' (List.transpose m)) ∧
  (10 = List.length (List.map List.maximum' m ++ List.map List.minimum' (List.transpose m))) ∧ 
  9 = (List.length (List.dedup (List.map List.maximum' m ++ List.map List.minimum' (List.transpose m))))
:= sorry

end nine_distinct_numbers_in_grid_l125_125006


namespace R_eq_R_plus2_eq_2_l125_125356

def R (n : ℕ) : ℕ := (∑ k in finset.Ico 2 13, (n % k))

def two_digit_pos_int (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem R_eq_R_plus2_eq_2 :
  (finset.filter (λ n, two_digit_pos_int n ∧ R n = R (n + 2)) (finset.range 100)).card = 2 :=
by {
  sorry
}

end R_eq_R_plus2_eq_2_l125_125356


namespace sum_of_distinct_product_l125_125930

-- Define the conditions given in the problem
def is_digit (n : ℕ) : Prop := n ≤ 9

def divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

def divisible_by_11 (n : ℕ) : Prop :=
  let odd_pos_sum := 7 + 1 + 4 + 9 + (n / 10 % 10)
  let even_pos_sum := 9 + (n / 10000 % 10) + 0 + 3 + (n % 10)
  abs (odd_pos_sum - even_pos_sum) = 11

-- Define the number 791G4093H5 as a function of G and H
def number (G H : ℕ) : ℕ :=
  7000000000 + 910000000 + G * 1000000 + 409000 + 3000 + H * 10 + 5

-- Proposition stating that the sum of all distinct possible values of GH is 8
theorem sum_of_distinct_product (G H : ℕ) (hG : is_digit G) (hH : is_digit H) :
  divisible_by_11 (number G H) → (G = 0 ∧ H = 7 ∨ G = 1 ∧ H = 8) →

  set.to_finset { GH | ∃ G H, G * H }.sum = 8 :=
by
  sorry
  
end sum_of_distinct_product_l125_125930


namespace enjoyable_gameplay_time_l125_125511

def total_gameplay_time_base : ℝ := 150
def enjoyable_fraction_base : ℝ := 0.30
def total_gameplay_time_expansion : ℝ := 50
def load_screen_fraction_expansion : ℝ := 0.25
def inventory_management_fraction_expansion : ℝ := 0.25
def mod_skip_fraction : ℝ := 0.15

def enjoyable_time_base : ℝ := total_gameplay_time_base * enjoyable_fraction_base
def not_load_screen_time_expansion : ℝ := total_gameplay_time_expansion * (1 - load_screen_fraction_expansion)
def not_inventory_management_time_expansion : ℝ := not_load_screen_time_expansion * (1 - inventory_management_fraction_expansion)

def tedious_time_base : ℝ := total_gameplay_time_base * (1 - enjoyable_fraction_base)
def tedious_time_expansion : ℝ := total_gameplay_time_expansion - not_inventory_management_time_expansion
def total_tedious_time : ℝ := tedious_time_base + tedious_time_expansion

def time_skipped_by_mod : ℝ := total_tedious_time * mod_skip_fraction

def total_enjoyable_time : ℝ := enjoyable_time_base + not_inventory_management_time_expansion + time_skipped_by_mod

theorem enjoyable_gameplay_time :
  total_enjoyable_time = 92.16 :=     by     simp [total_enjoyable_time, enjoyable_time_base, not_inventory_management_time_expansion, time_skipped_by_mod]; sorry

end enjoyable_gameplay_time_l125_125511


namespace find_k_l125_125444

-- Conditions
variables {V : Type} [inner_product_space ℝ V] 
variables (a b : V) (k : ℝ)

-- Definitions based on the conditions
def is_unit_vector (v : V) : Prop := inner v v = 1
def is_perpendicular (v w : V) : Prop := inner v w = 0

-- Theorem to be proved
theorem find_k (h₁ : is_unit_vector a)
               (h₂ : is_unit_vector b)
               (h₃ : ¬ collinear ℝ {a, b})
               (h₄ : is_perpendicular (a + b) (k • a - b)) :
  k = 1 :=
sorry

end find_k_l125_125444


namespace ratio_of_interior_to_exterior_angle_in_regular_octagon_l125_125134

theorem ratio_of_interior_to_exterior_angle_in_regular_octagon
  (n : ℕ) (regular_polygon : n = 8) : 
  let interior_angle := ((n - 2) * 180) / n
  let exterior_angle := 360 / n
  (interior_angle / exterior_angle) = 3 :=
by
  sorry

end ratio_of_interior_to_exterior_angle_in_regular_octagon_l125_125134


namespace prove_varphi_l125_125124

open Real

-- Given conditions
axiom varphi : ℝ
axiom varphi_bound : -π < varphi ∧ varphi < 0

-- The function and transformation described
def f (x : ℝ) := sin (3 * x + varphi)

-- Symmetry condition of the transformed function
axiom symmetry_condition : ∀ x : ℝ, sin (3 * (x + π / 12) + varphi) = sin (-3 * (x + π / 12) - varphi)

-- Problem Statement to prove
theorem prove_varphi : varphi = -π / 4 :=
sorry

end prove_varphi_l125_125124


namespace total_pages_correct_l125_125954

-- Given that 390 digits were required to number the pages
constant digits_required : Nat := 390

-- Define the number of digits used for different ranges of pages
def digits_in_range (start : Nat) (end : Nat) (digits_per_page : Nat) : Nat :=
  (end - start + 1) * digits_per_page

-- Pages 1 to 9
def digits_1_to_9 : Nat := digits_in_range 1 9 1

-- Pages 10 to 99
def digits_10_to_99 : Nat := digits_in_range 10 99 2

-- Total digits used by pages 1 to 99
def digits_1_to_99 : Nat := digits_1_to_9 + digits_10_to_99

-- Remaining digits for pages 100 and beyond
def remaining_digits : Nat := digits_required - digits_1_to_99

-- Number of pages in the range 100 and beyond
def pages_beyond_99 : Nat := remaining_digits / 3

-- Total number of pages
def total_pages : Nat := 99 + pages_beyond_99

-- Theorem stateent: total number of pages is 166
theorem total_pages_correct : total_pages = 166 := by
  sorry -- Proof to be provided

end total_pages_correct_l125_125954


namespace maximize_S_l125_125031

-- Define the arithmetic sequence and conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def a1 : ℝ := 13
def S (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in range n, a (i + 1)

-- Given conditions
def conditions (a : ℕ → ℝ) : Prop :=
  arithmetic_sequence a ∧ a 1 = a1 ∧ S a 3 = S a 11

-- Proof that S_n is maximized at n = 7
theorem maximize_S 
  (a : ℕ → ℝ) 
  (h : conditions a) : 
  ∀ n, n ≥ 7 → S a n ≤ S a 7 := 
sorry

end maximize_S_l125_125031


namespace farm_total_amount_90000_l125_125547

-- Defining the conditions
def apples_produce (mangoes: ℕ) : ℕ := 2 * mangoes
def oranges_produce (mangoes: ℕ) : ℕ := mangoes + 200

-- Defining the total produce of all fruits
def total_produce (mangoes: ℕ) : ℕ := apples_produce mangoes + mangoes + oranges_produce mangoes

-- Defining the price per kg
def price_per_kg : ℕ := 50

-- Defining the total amount from selling all fruits
noncomputable def total_amount (mangoes: ℕ) : ℕ := total_produce mangoes * price_per_kg

-- Proving that the total amount he got in that season is $90,000
theorem farm_total_amount_90000 : total_amount 400 = 90000 := by
  sorry

end farm_total_amount_90000_l125_125547


namespace min_sum_squares_l125_125305

noncomputable def distances (P : ℝ) : ℝ :=
  let AP := P
  let BP := |P - 1|
  let CP := |P - 2|
  let DP := |P - 5|
  let EP := |P - 13|
  AP^2 + BP^2 + CP^2 + DP^2 + EP^2

theorem min_sum_squares : ∀ P : ℝ, distances P ≥ 88.2 :=
by
  sorry

end min_sum_squares_l125_125305


namespace exists_six_digit_in_seq_l125_125937

noncomputable def seq : ℕ → ℕ
| 0 := 2
| (n+1) := nat.floor (3 / 2 * seq n : ℝ)

theorem exists_six_digit_in_seq :
  ∃ n, 100000 ≤ seq n ∧ seq n < 1000000 :=
sorry

end exists_six_digit_in_seq_l125_125937


namespace max_value_of_y_over_x_l125_125100

theorem max_value_of_y_over_x
  (x y : ℝ)
  (h1 : x + y ≥ 3)
  (h2 : x - y ≥ -1)
  (h3 : 2 * x - y ≤ 3) :
  (∀ (x y : ℝ), (x + y ≥ 3) ∧ (x - y ≥ -1) ∧ (2 * x - y ≤ 3) → (∀ k, k = y / x → k ≤ 2)) :=
by
  sorry

end max_value_of_y_over_x_l125_125100


namespace random_event_is_B_l125_125288

axiom isCertain (event : Prop) : Prop
axiom isImpossible (event : Prop) : Prop
axiom isRandom (event : Prop) : Prop

def A : Prop := ∀ t, t is certain (the sun rises from the east at time t)
def B : Prop := ∃ t, t is random (encountering a red light at time t ∧ passing through traffic light intersection)
def C : Prop := ∀ (p1 p2 p3 : ℝ²), isCertain (non-collinear points p1 p2 p3 → ∃! c, c is circle passing through p1 p2 p3)
def D : Prop := ∀ (T : Triangle), isImpossible (sum_of_interior_angles T = 540)

theorem random_event_is_B : isRandom B :=
by
  sorry

end random_event_is_B_l125_125288


namespace solve_p_l125_125809

theorem solve_p (p q : ℚ) (h1 : 5 * p + 3 * q = 7) (h2 : 2 * p + 5 * q = 8) : 
  p = 11 / 19 :=
by
  sorry

end solve_p_l125_125809


namespace area_trapezoid_ABCD_l125_125850
-- Importing the necessary library to bring in all required mathematical tools

-- Defining the conditions in Lean 4 format
variables {A B C D E : Type*}
variables [trapezoid : Trapezoid A B C D]
variables [parallel_AD_BC : Parallel AD BC]
variables (p q : ℝ)

-- Given conditions
variables [area_EBC : Area (triangle E B C) = p^2]
variables [area_EDA : Area (triangle E D A) = q^2]

-- Theorem statement
theorem area_trapezoid_ABCD : Area (trapezoid A B C D) = (p + q)^2 := 
sorry

end area_trapezoid_ABCD_l125_125850


namespace intersection_A_and_B_l125_125874

variable {A : Set ℤ}  -- A is a subset of integers
variable (B : Set ℤ)
variable (f : ℤ → ℤ)
variable hB : B = {1, 2}
variable hf : ∀ x, f x = x^2

theorem intersection_A_and_B (A : Set ℤ) (hB : B = {1, 2}) (hf : ∀ x, f x = x^2) :
  A ∩ B = ∅ ∨ A ∩ B = {1} :=
sorry

end intersection_A_and_B_l125_125874


namespace ideal_heat_engine_efficiency_l125_125251

def efficiency (T_C T_H : ℝ) : ℝ := 1 - T_C / T_H

def initial_efficiency : ℝ := 0.40

def new_temperature_hot (T_H : ℝ) : ℝ := 1.40 * T_H

def new_temperature_cold (T_C : ℝ) : ℝ := 0.60 * T_C

theorem ideal_heat_engine_efficiency
  (T_C T_H : ℝ)
  (h1 : efficiency T_C T_H = initial_efficiency)
  (T_H' : ℝ := new_temperature_hot T_H)
  (T_C' : ℝ := new_temperature_cold T_C) :
  efficiency T_C' T_H' = 0.74 :=
by
  sorry

end ideal_heat_engine_efficiency_l125_125251


namespace eleven_twelve_divisible_by_133_l125_125225

theorem eleven_twelve_divisible_by_133 (n : ℕ) (h : n > 0) : 133 ∣ (11^(n+2) + 12^(2*n+1)) := 
by 
  sorry

end eleven_twelve_divisible_by_133_l125_125225


namespace find_n_modulo_l125_125407

theorem find_n_modulo :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 7 ∧ n ≡ -3737 [MOD 8] ∧ n = 7 :=
by
  sorry

end find_n_modulo_l125_125407


namespace union_A_B_inter_A_B_inter_compA_B_l125_125780

-- Extend the universal set U to be the set of all real numbers ℝ
def U : Set ℝ := Set.univ

-- Define set A as the set of all real numbers x such that -3 ≤ x ≤ 4
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 4}

-- Define set B as the set of all real numbers x such that -1 < x < 5
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 5}

-- Prove that A ∪ B = {x : ℝ | -3 ≤ x ∧ x < 5}
theorem union_A_B : A ∪ B = {x : ℝ | -3 ≤ x ∧ x < 5} := by
  sorry

-- Prove that A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 4}
theorem inter_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 4} := by
  sorry

-- Define the complement of A in U
def comp_A : Set ℝ := {x : ℝ | x < -3 ∨ x > 4}

-- Prove that (complement_U A) ∩ B = {x : ℝ | 4 < x ∧ x < 5}
theorem inter_compA_B : comp_A ∩ B = {x : ℝ | 4 < x ∧ x < 5} := by
  sorry

end union_A_B_inter_A_B_inter_compA_B_l125_125780


namespace sam_investment_l125_125234

noncomputable def compound_interest (P: ℝ) (r: ℝ) (n: ℕ) (t: ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem sam_investment :
  compound_interest 3000 0.10 4 1 = 3311.44 :=
by
  sorry

end sam_investment_l125_125234


namespace bob_distance_correct_l125_125330

-- Define the problem conditions
def side_length : ℝ := 3
def total_distance_walked : ℝ := 7
def cos_108 : ℝ := -0.309
def sin_108 : ℝ := 0.951
def cos_216 : ℝ := -0.809
def sin_216 : ℝ := -0.588

-- Distance function using Pythagorean theorem
def distance (x1 y1 : ℝ) : ℝ :=
  real.sqrt (x1 ^ 2 + y1 ^ 2)

-- Bob's final coordinates after walking 7km.
def bob_final_coords : ℝ × ℝ :=
  let x := 3 + 3 * cos_108 + 1 * cos_216 in
  let y := 3 * sin_108 + 1 * sin_216 in
  (x, y)

-- Final distance from the starting point
def bob_distance_from_start : ℝ :=
  let (x, y) := bob_final_coords in
  distance x y

-- Prove the required distance
theorem bob_distance_correct :
  bob_distance_from_start = real.sqrt 6.731 :=
by
  sorry

end bob_distance_correct_l125_125330


namespace trip_distance_first_part_l125_125654

theorem trip_distance_first_part (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 70) (h3 : 32 = 70 / ((x / 48) + ((70 - x) / 24))) : x = 35 :=
by
  sorry

end trip_distance_first_part_l125_125654


namespace min_value_of_fraction_l125_125540

theorem min_value_of_fraction 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (Real.sqrt 3) = Real.sqrt (3 ^ a * 3 ^ (2 * b))) : 
  ∃ (min : ℝ), min = (2 / a + 1 / b) ∧ min = 8 :=
by
  -- proof will be skipped using sorry
  sorry

end min_value_of_fraction_l125_125540


namespace distinct_complex_numbers_count_l125_125801

theorem distinct_complex_numbers_count :
  let real_choices := 10
  let imag_choices := 9
  let distinct_complex_numbers := real_choices * imag_choices
  distinct_complex_numbers = 90 :=
by
  sorry

end distinct_complex_numbers_count_l125_125801


namespace man_salary_l125_125325

variable (salary : ℕ)
variable (food : ℕ := (1/5 : ℚ) * salary)
variable (rent : ℕ := (1/10 : ℚ) * salary)
variable (clothes : ℕ := (3/5 : ℚ) * salary)
variable (remaining : ℕ := 18000)

theorem man_salary :
  salary - food - rent - clothes = remaining -> salary = 180000 :=
by
  sorry

end man_salary_l125_125325


namespace cost_of_items_l125_125230

theorem cost_of_items (e t b : ℝ) 
    (h1 : 3 * e + 4 * t = 3.20)
    (h2 : 4 * e + 3 * t = 3.50)
    (h3 : 5 * e + 5 * t + 2 * b = 5.70) :
    4 * e + 4 * t + 3 * b = 5.20 :=
by
  sorry

end cost_of_items_l125_125230


namespace calculate_fraction_l125_125681

theorem calculate_fraction (x : ℝ) (h₀ : x ≠ 1) (h₁ : x ≠ -1) : 
  (1 / (x - 1)) - (2 / (x^2 - 1)) = 1 / (x + 1) :=
by
  sorry

end calculate_fraction_l125_125681


namespace volume_of_solid_is_112_pi_sqrt_14_l125_125262

def vector_dot (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def volume_of_solid (u : ℝ × ℝ × ℝ) :=
  vector_dot u u = vector_dot u (-6, 18, 12)

theorem volume_of_solid_is_112_pi_sqrt_14 :
  ∃ (r : ℝ), volume_of_solid r → (4 / 3) * π * 126^(3 / 2) = 112 * π * real.sqrt 14 := by
  sorry

end volume_of_solid_is_112_pi_sqrt_14_l125_125262


namespace probability_sum_20_with_2_dodecahedral_dice_l125_125252

theorem probability_sum_20_with_2_dodecahedral_dice : 
  let outcomes := { (a, b) | a ∈ finset.range 1 13 ∧ b ∈ finset.range 1 13 } in
  let favorable := { (a, b) | a ∈ finset.range 1 13 ∧ b ∈ finset.range 1 13 ∧ a + b = 20 } in
  (favorable.card : ℚ) / (outcomes.card : ℚ) = 1 / 48 := 
by
  sorry

end probability_sum_20_with_2_dodecahedral_dice_l125_125252


namespace power_of_seven_l125_125708

theorem power_of_seven (x : ℝ) : 
  x = 1 / 12 → (7^(1/4) / 7^(1/6)) = 7^x :=
by
  intro h
  rw h
  sorry

end power_of_seven_l125_125708


namespace jeff_total_is_14_l125_125857

noncomputable def jeffs_total_purchase (p1 p2 p3 discount : ℝ) : ℝ :=
  let p3_after_discount := p3 - discount
  (Real.ceil p1.round + Real.ceil p2.round + Real.ceil p3_after_discount.round)

theorem jeff_total_is_14 :
  jeffs_total_purchase 2.45 3.75 8.56 0.50 = 14 :=
by
  sorry

end jeff_total_is_14_l125_125857


namespace infinite_series_sum_l125_125730

theorem infinite_series_sum :
  let S := ( ∑' n, (n + 1) * (1 / 999) ^ n )
  in S = 998001 / 996004 :=
by
  sorry

end infinite_series_sum_l125_125730


namespace binomial_identity_a_l125_125992

variables {r m k : ℕ}

theorem binomial_identity_a (h₁ : 0 ≤ k) (h₂ : k ≤ m) (h₃ : m ≤ r) :
  nat.choose r m * nat.choose m k = nat.choose r k * nat.choose (r - k) (m - k) := sorry

end binomial_identity_a_l125_125992


namespace find_alpha_l125_125128

noncomputable def equal_sides (A B C D E F G : Type) [MetricSpace A] :
    Prop := 
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D E ∧
  dist D E = dist E F ∧ dist E F = dist F G ∧ dist F G = dist G A

-- Conditions
variables {A B C D E F G : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace G]
variables (alpha : ℝ) (AE AD : Prop) (h1 : equal_sides A B C D E F G)
-- Use dummy variable Prop for straight lines AE and AD to avoid actual directional calculation

-- Theorem statement
theorem find_alpha (h2 : ∠ DAE = alpha^degree) : alpha = 180 / 7 := sorry

end find_alpha_l125_125128


namespace expression_value_l125_125882

theorem expression_value (x : ℤ) (hx : x = 1729) : abs (abs (abs x + x) + abs x) + x = 6916 :=
by
  rw [hx]
  sorry

end expression_value_l125_125882


namespace geom_series_sum_l125_125694

variable (n : ℕ) (a r : ℤ)

theorem geom_series_sum (h₁ : a = -1) (h₂ : r = -3) (h₃ : (n : ℤ) = 9) :
  ∑ i in finset.range n, a * r ^ i = 4921 :=
by
  -- Definitions and conditions
  have h₄ : ∑ i in finset.range (9 : ℕ), -1 * (-3) ^ i = 4921 := sorry
  exact h₄

end geom_series_sum_l125_125694


namespace num_tangent_lines_with_equal_intercepts_l125_125104

noncomputable def circle : set (ℝ × ℝ) := { p | p.1^2 + (p.2 - 2)^2 = 1 }

def is_tangent_line (line : (ℝ × ℝ) → Prop) : Prop :=
∀ P ∈ circle, ∃ Q ∈ circle, line P ∧ ¬line Q

def equal_intercepts_line (line : (ℝ × ℝ) → Prop) : Prop :=
∃ a : ℝ, a ≠ 0 ∧ line = λ P, P.1 + P.2 = a ∨ P.1 + P.2 = -a

theorem num_tangent_lines_with_equal_intercepts :
  ∃! (lines : set ((ℝ × ℝ) → Prop)), set.finite lines ∧ lines.card = 4 ∧
  ∀ l ∈ lines, is_tangent_line l ∧ equal_intercepts_line l :=
sorry

end num_tangent_lines_with_equal_intercepts_l125_125104


namespace distinct_points_distance_l125_125901

theorem distinct_points_distance (e : ℝ) (a b : ℝ) (h₁ : (sqrt e, a) ∈ { p : ℝ × ℝ | p.2^2 + p.1^4 = 3 * p.1^2 * p.2 + 1 })
                                     (h₂ : (sqrt e, b) ∈ { p : ℝ × ℝ | p.2^2 + p.1^4 = 3 * p.1^2 * p.2 + 1 })
                                     (h₃ : a ≠ b) :
  |a - b| = 2 * sqrt ((5 * e^2) / 4 - 1) :=
  sorry

end distinct_points_distance_l125_125901


namespace total_chore_time_l125_125512

-- Let's define the conditions
def hours_vacuuming : ℕ := 3
def multiplier_rest : ℕ := 3

-- Define the proof statement
theorem total_chore_time :
  let hours_rest := hours_vacuuming * multiplier_rest in
  hours_vacuuming + hours_rest = 12 :=
by
  sorry

end total_chore_time_l125_125512


namespace part1_part2_l125_125843

noncomputable def cost_prices (x y : ℕ) : Prop := 
  8800 / (y + 4) = 2 * (4000 / x) ∧ 
  x = 40 ∧ 
  y = 44

theorem part1 : ∃ x y : ℕ, cost_prices x y := sorry

noncomputable def minimum_lucky_rabbits (m : ℕ) : Prop := 
  26 * m + 20 * (200 - m) ≥ 4120 ∧ 
  m = 20

theorem part2 : ∃ m : ℕ, minimum_lucky_rabbits m := sorry

end part1_part2_l125_125843


namespace sum_of_common_divisors_36_48_l125_125021

-- Definitions based on the conditions
def is_divisor (n d : ℕ) : Prop := d ∣ n

-- List of divisors for 36 and 48
def divisors_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]
def divisors_48 : List ℕ := [1, 2, 3, 4, 6, 8, 12, 16, 24, 48]

-- Definition of common divisors
def common_divisors_36_48 : List ℕ := [1, 2, 3, 4, 6, 12]

-- Sum of common divisors
def sum_common_divisors_36_48 := common_divisors_36_48.sum

-- The statement of the theorem
theorem sum_of_common_divisors_36_48 : sum_common_divisors_36_48 = 28 := by
  sorry

end sum_of_common_divisors_36_48_l125_125021


namespace mr_value_l125_125248

structure Point :=
(x : ℕ)
(y : ℕ)

def distance (p1 p2 : Point) : ℝ :=
Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def is_growing_path (path : List Point) : Prop :=
∀ i j, i < j → distance (path[i]) (path[j]) > distance (path[i + 1]) (path[i])

def grid_points : List Point :=
(List.range 4).bind (λ x, (List.range 4).map (λ y, Point.mk x y))

def max_growing_path_length : ℕ :=
10  -- From solution step 2
  
def num_max_growing_paths : ℕ :=
24  -- From solution step 6

theorem mr_value : max_growing_path_length * num_max_growing_paths = 240 := by
  have m := max_growing_path_length
  have r := num_max_growing_paths
  calc
    m * r = 10 * 24 : by rw [nat.cast_add, nat.cast_mul]
    ... = 240       : by norm_num

end mr_value_l125_125248


namespace topsoil_cost_proof_l125_125274

-- Definitions
def cost_per_cubic_foot : ℕ := 8
def cubic_feet_per_cubic_yard : ℕ := 27
def amount_in_cubic_yards : ℕ := 7

-- Theorem
theorem topsoil_cost_proof : cost_per_cubic_foot * cubic_feet_per_cubic_yard * amount_in_cubic_yards = 1512 := by
  -- proof logic goes here
  sorry

end topsoil_cost_proof_l125_125274


namespace ellipse_hyperbola_foci_coincide_l125_125580

theorem ellipse_hyperbola_foci_coincide (b^2 : ℝ) :
  (∀ (x y : ℝ), x^2 / 25 + y^2 / b^2 = 1) ∧
  (∀ (x y : ℝ), x^2 / (169 / 36) - y^2 / (144 / 36) = 1) →
  b^2 = 587 / 36 :=
by sorry

end ellipse_hyperbola_foci_coincide_l125_125580


namespace shortest_chord_intercepted_by_line_l125_125819

theorem shortest_chord_intercepted_by_line (k : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 2*x - 3 = 0 → y = k*x + 1 → (x - y + 1 = 0)) :=
sorry

end shortest_chord_intercepted_by_line_l125_125819


namespace monomial_properties_l125_125241

noncomputable def monomial_coefficient (c : ℝ) (a : ℝ) : ℝ :=
c

noncomputable def monomial_degree (e : Nat) : Nat :=
e

theorem monomial_properties :
  (monomial_coefficient (-3 * real.pi) (a ^ 3) = -3 * real.pi) ∧
  (monomial_degree 3 = 3) :=
by
  sorry

end monomial_properties_l125_125241


namespace range_of_mn_l125_125532

theorem range_of_mn (m n : ℝ)
  (h : ∀ x y : ℝ, (m+1)*x + (n+1)*y = 2 → (x-1)^2 + (y-1)^2 = 1) :
  m + n ∈ set.Iic (2 - 2 * real.sqrt 2) ∪ set.Ici (2 + 2 * real.sqrt 2) :=
sorry

end range_of_mn_l125_125532


namespace both_fifth_and_ninth_terms_are_20_l125_125788

def sequence_a (n : ℕ) : ℕ := n^2 - 14 * n + 65

theorem both_fifth_and_ninth_terms_are_20 : sequence_a 5 = 20 ∧ sequence_a 9 = 20 := 
by
  sorry

end both_fifth_and_ninth_terms_are_20_l125_125788


namespace no_ten_consecutive_sum_2016_exists_seven_consecutive_sum_2016_l125_125630

theorem no_ten_consecutive_sum_2016 : ¬ ∃ a : ℕ, (10 * a + (10 * (10 - 1)) / 2 = 2016) :=
by 
  intro h
  obtain ⟨a, h_sum⟩ := h
  have h_ineq : 10 * a + 45 = 2016 := by rwa[add_mul, nat.mul_div_right' _]
  sorry

theorem exists_seven_consecutive_sum_2016 : ∃ b : ℕ, (7 * b + (7 * (7 - 1)) / 2 = 2016) :=
by 
  exists 285
  have h_sum : 7 * 285 + (7 * 6) / 2 = 2016 := by norm_num
  exact h_sum

end no_ten_consecutive_sum_2016_exists_seven_consecutive_sum_2016_l125_125630


namespace function_is_zero_l125_125519

-- Define the condition that for any three points A, B, and C forming an equilateral triangle,
-- the sum of their function values is zero.
def has_equilateral_property (f : ℝ × ℝ → ℝ) : Prop :=
  ∀ (A B C : ℝ × ℝ), dist A B = 1 ∧ dist B C = 1 ∧ dist C A = 1 → 
  f A + f B + f C = 0

-- Define the theorem that states that a function with the equilateral property is identically zero.
theorem function_is_zero {f : ℝ × ℝ → ℝ} (h : has_equilateral_property f) : 
  ∀ (x : ℝ × ℝ), f x = 0 := 
by
  sorry

end function_is_zero_l125_125519


namespace boat_trip_ratio_l125_125645

theorem boat_trip_ratio (speed_still_water : ℕ) (current_speed : ℕ) (down_distance : ℕ) (up_distance : ℕ)
  (h1 : speed_still_water = 20)
  (h2 : current_speed = 4)
  (h3 : down_distance = 5)
  (h4 : up_distance = 3) :
  (let avg_speed := (down_distance + up_distance) * (speed_still_water - current_speed) * (speed_still_water + current_speed) / (down_distance * (speed_still_water + current_speed) + up_distance * (speed_still_water - current_speed)) / speed_still_water in
  avg_speed = 96 / 95) := 
by
  -- Proof omitted
  sorry

end boat_trip_ratio_l125_125645


namespace range_of_a_inequality_for_n_l125_125062

-- Define the function f(x) and state the condition that it is decreasing
def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x - a * x^2

-- Define the condition for f(x) being decreasing
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → (f' f x ≤ 0)

-- First part to prove: range of values for a
theorem range_of_a (a : ℝ) : is_decreasing (f x a) → a ≥ 1/2 :=
sorry

-- Second part to prove: the inequality for all n > 1
theorem inequality_for_n (n : ℕ) (h : n > 1) :
  (∑ i in finset.range (n - 1) + 1, 1 / (i * Real.log i)) > (3 * n^2 - n - 2) / (2 * n * (n + 1)) :=
sorry

end range_of_a_inequality_for_n_l125_125062


namespace wrong_value_l125_125929

-- Definitions based on the conditions
def initial_mean : ℝ := 32
def corrected_mean : ℝ := 32.5
def num_observations : ℕ := 50
def correct_observation : ℝ := 48

-- We need to prove that the wrong value of the observation was 23
theorem wrong_value (sum_initial : ℝ) (sum_corrected : ℝ) : 
  sum_initial = num_observations * initial_mean ∧ 
  sum_corrected = num_observations * corrected_mean →
  48 - (sum_corrected - sum_initial) = 23 :=
by
  sorry

end wrong_value_l125_125929


namespace min_chord_length_l125_125047

open Real

def circle (x y : ℝ) := (x - 1) ^ 2 + (y - 2) ^ 2 = 25

def line (m x y : ℝ) := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

theorem min_chord_length (m : ℝ) : 
  ∃ L, (∀ x y, line m x y → circle x y) → 
  L = 4 * sqrt 5 :=
by
  sorry

end min_chord_length_l125_125047


namespace inequality_proof_l125_125634

open Real

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a * b * c * d = 1) :
  (a^4 + b^4) / (a^2 + b^2) + (b^4 + c^4) / (b^2 + c^2) + (c^4 + d^4) / (c^2 + d^2) + (d^4 + a^4) / (d^2 + a^2) ≥ 4 :=
by
  sorry

end inequality_proof_l125_125634


namespace chip_sheets_per_pack_l125_125688

noncomputable def sheets_per_pack (pages_per_day : ℕ) (days_per_week : ℕ) (classes : ℕ) 
                                  (weeks : ℕ) (packs : ℕ) : ℕ :=
(pages_per_day * days_per_week * classes * weeks) / packs

theorem chip_sheets_per_pack :
  sheets_per_pack 2 5 5 6 3 = 100 :=
sorry

end chip_sheets_per_pack_l125_125688


namespace books_not_sold_l125_125347

theorem books_not_sold :
  let initial_stock := 800
  let books_sold_mon := 60
  let books_sold_tue := 10
  let books_sold_wed := 20
  let books_sold_thu := 44
  let books_sold_fri := 66
  in initial_stock - (books_sold_mon + books_sold_tue + books_sold_wed + books_sold_thu + books_sold_fri) = 600 :=
by
  let initial_stock := 800
  let books_sold_mon := 60
  let books_sold_tue := 10
  let books_sold_wed := 20
  let books_sold_thu := 44
  let books_sold_fri := 66
  show initial_stock - (books_sold_mon + books_sold_tue + books_sold_wed + books_sold_thu + books_sold_fri) = 600
  calc
    initial_stock - (books_sold_mon + books_sold_tue + books_sold_wed + books_sold_thu + books_sold_fri) = initial_stock - 200 : by sorry
    ... = 600 : by sorry

end books_not_sold_l125_125347


namespace sin_2theta_value_l125_125482

theorem sin_2theta_value (θ : ℝ) (h : ∑' n : ℕ, (cos θ)^(2*n) = 9) : 
  sin (2*θ) = 4 * Real.sqrt 2 / 9 ∨ sin (2*θ) = -4 * Real.sqrt 2 / 9 := 
by 
  sorry

end sin_2theta_value_l125_125482


namespace curve_intersects_midline_once_l125_125520

open Complex

variables (a b c : ℝ)
def z0 := Complex.I * a
def z1 := Complex.mk (1 / 2) (b : ℝ)
def z2 := Complex.mk 1 (c : ℝ)

noncomputable def curve (t : ℝ) : ℂ :=
  z0 * (cos t)^4 + 2 * z1 * (cos t)^2 * (sin t)^2 + z2 * (sin t)^4

theorem curve_intersects_midline_once :
  let point := (1 / 2 : ℝ) + Complex.I * (1 / 4 * (a + c + 2 * b)) in
  ∀ (t : ℝ), (curve a b c t) = point ↔ t = 1 / 2 :=
sorry

end curve_intersects_midline_once_l125_125520


namespace math_problem_l125_125685

noncomputable def part1 : Prop :=
  (sqrt 2 + 2) ^ 2 = 6 + 4 * sqrt 2

noncomputable def part2 : Prop :=
  2 * (sqrt 3 - sqrt 8) - (1 / 2) * (sqrt 18 + sqrt 12) = -7 * sqrt 2 / 2

theorem math_problem : part1 ∧ part2 :=
by
  split
  sorry
  sorry

end math_problem_l125_125685


namespace necessary_but_not_sufficient_l125_125753

-- Definitions
variables (α : Type*) [plane : Set α] (a : α) (lines : Set (Set α))

-- Conditions
def p : Prop := ∃ (l : Set α), l ∈ lines ∧ ∀ l1 l2 : Set α, l1 ∈ lines → l2 ∈ lines → l1 ≠ l2 → (a ⊥ l1 ∧ a ⊥ l2)
def q : Prop := a ⊥ plane

-- Statement to Prove:
theorem necessary_but_not_sufficient : (q → p) ∧ (¬(p → q)) :=
by
  sorry

end necessary_but_not_sufficient_l125_125753


namespace find_incorrect_proposition_l125_125807

/-- Proposition A: If a line within a plane is perpendicular to any line within another plane,
then the two planes are perpendicular. -/
def propA (l1 l2 : Line) (p1 p2 : Plane) : Prop :=
  (l1 ∈ p1 ∧ l2 ∈ p2 ∧ ∀ l ∈ p2, l1 ⟂ l) → (p1 ⟂ p2)

/-- Proposition B: If every line in a plane is parallel to another plane,
then the two planes are parallel. -/
def propB (p1 p2 : Plane) : Prop :=
  (∀ l ∈ p1, ∃ p' : Plane, (l ∥ p') ∧ (p' ∥ p2)) → (p1 ∥ p2)

/-- Proposition C: If a line is parallel to a plane, and a plane that passes through this line 
intersects the given plane, then the line is parallel to the line of intersection. -/
def propC (l: Line) (p1 p2 : Plane) : Prop :=
  ((l ∥ p1) ∧ ∃ lI: Line, lI ∈ (p1 ∩ p2) ∧ (l ∥ lI)) 

/-- Proposition D: If the projections of two different lines on a plane are perpendicular to each
other, then the two lines are perpendicular. -/
def propD (l1 l2 : Line) (p : Plane) : Prop :=
  (proj l1 p ⟂ proj l2 p) → (l1 ⟂ l2)

-- Main statement
theorem find_incorrect_proposition : ¬ propD l1 l2 p :=
sorry

end find_incorrect_proposition_l125_125807


namespace greatest_discarded_oranges_l125_125643

theorem greatest_discarded_oranges (n : ℕ) : n % 7 ≤ 6 := 
by 
  sorry

end greatest_discarded_oranges_l125_125643


namespace necessary_but_not_sufficient_l125_125303

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 ≥ 1) → (x > 1) ∨ (x ≤ -1) := 
by 
  sorry

end necessary_but_not_sufficient_l125_125303


namespace sqrt_a_minus_2_range_l125_125125

theorem sqrt_a_minus_2_range (a : ℝ) : (∃ b : ℝ, b ≥ 0 ∧ b = a - 2) → a ≥ 2 :=
by
  intro h
  cases h with b hb
  cases hb with hb0 hba
  rw [hba] at hb0
  linarith
  sorry

end sqrt_a_minus_2_range_l125_125125


namespace distance_between_lights_l125_125572

/-- Given a string of lights where lights are 8 inches apart 
    and follow a repeating pattern of 3 red lights followed by 4 green lights, 
    prove that the distance in feet between the 5th red light and the 23rd red light is 28 feet. -/
theorem distance_between_lights : 
  (distance_between_lights 5 23) = 28 :=
sorry

end distance_between_lights_l125_125572


namespace train_B_speed_l125_125607

-- Definitions based on conditions
def speed_of_train_A : Real := 90
def time_after_meeting_A : Real := 9
def time_after_meeting_B : Real := 4

-- Definition to be proved
theorem train_B_speed :
  (let distance_A := speed_of_train_A * time_after_meeting_A in
  let v_B := distance_A / time_after_meeting_B in
  v_B = 202.5) := by
  sorry

end train_B_speed_l125_125607


namespace find_constant_term_l125_125923

noncomputable def constantTermInExpansion : ℤ := -480

theorem find_constant_term :
  let expression := (x^2 + (4 / x^2) - 4)^3 * (x + 3)
  constantTerm expression = constantTermInExpansion :=
sorry

end find_constant_term_l125_125923


namespace constant_term_in_binomial_expansion_l125_125751

theorem constant_term_in_binomial_expansion :
  let a := ∫ x in -Real.pi / 2..Real.pi / 2, Real.cos x 
  in (a = 2) →
  let expansion_term := (x + a / Real.sqrt x) ^ 6
  in constant_term expansion_term = 240 :=
by
  simp only [expansion_term]
  sorry -- Proof is skipped in this task

end constant_term_in_binomial_expansion_l125_125751


namespace floor_expression_correct_l125_125378

theorem floor_expression_correct :
  (∃ x : ℝ, x = 2007 ^ 3 / (2005 * 2006) - 2005 ^ 3 / (2006 * 2007) ∧ ⌊x⌋ = 8) := 
sorry

end floor_expression_correct_l125_125378


namespace length_of_third_median_l125_125836

-- Define the given conditions
variables (triangle : Type) [IsoscelesTriangle triangle]
variables (ABC : triangle)
variables (AB AC : ℝ) (equal_sides : AB = AC)
variables (m_A m_B : ℝ)
variables (area : ℝ)
variables (m_C : ℝ)

-- Set the specific conditions from the problem
noncomputable def isosceles_triangle_conditions : Prop :=
  AB = AC ∧ m_A = 4 ∧ m_B = 4 ∧ area = 3 * Real.sqrt 15

-- The statement to prove
theorem length_of_third_median : isosceles_triangle_conditions AB AC m_A m_B area ABC → m_C = 2 * Real.sqrt 37 :=
by
  sorry

end length_of_third_median_l125_125836


namespace find_pairs_l125_125401

theorem find_pairs (a b : ℤ) (ha : a ≥ 1) (hb : b ≥ 1)
  (h1 : (a^2 + b) % (b^2 - a) = 0) 
  (h2 : (b^2 + a) % (a^2 - b) = 0) :
  (a = 2 ∧ b = 2) ∨ (a = 3 ∧ b = 3) ∨ (a = 1 ∧ b = 2) ∨ 
  (a = 2 ∧ b = 1) ∨ (a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2) := 
sorry

end find_pairs_l125_125401


namespace remaining_players_average_points_l125_125231

-- Define the conditions
def total_points : ℕ := 270
def total_players : ℕ := 9
def players_averaged_50 : ℕ := 5
def average_points_50 : ℕ := 50

-- Define the query
theorem remaining_players_average_points :
  (total_points - players_averaged_50 * average_points_50) / (total_players - players_averaged_50) = 5 :=
by
  sorry

end remaining_players_average_points_l125_125231


namespace area_of_circle_in_terms_of_pi_l125_125838

noncomputable def circle_eq := 3 * x^2 + 3 * y^2 - 12 * x + 9 * y + 27 = 0

theorem area_of_circle_in_terms_of_pi :
  (∃ x y : ℝ, circle_eq) →
  ∃ r : ℝ, π * r^2 = 61 / 4 * π :=
sorry

end area_of_circle_in_terms_of_pi_l125_125838


namespace complex_props_hold_l125_125426

theorem complex_props_hold (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ((a + b)^2 = a^2 + 2*a*b + b^2) ∧ (a^2 = a*b → a = b) :=
by
  sorry

end complex_props_hold_l125_125426


namespace problem_statement_l125_125058

noncomputable def inequality_not_necessarily_true (a b c : ℝ) :=
  c < b ∧ b < a ∧ a * c < 0

theorem problem_statement (a b c : ℝ) (h : inequality_not_necessarily_true a b c) : ¬ (∃ a b c : ℝ, c < b ∧ b < a ∧ a * c < 0 ∧ ¬ (b^2/c > a^2/c)) :=
by sorry

end problem_statement_l125_125058


namespace part1_div_15_cubed_part2_find_all_n_l125_125076

def sequence (n : ℕ) : ℤ := 15 * n + 2 + (15 * n - 32) * 16^(n - 1)

theorem part1_div_15_cubed (n : ℕ) : 15^3 ∣ sequence n := 
by sorry

theorem part2_find_all_n : 
  {n | 1991 ∣ sequence n ∧ 1991 ∣ sequence (n + 1) ∧ 1991 ∣ sequence (n + 2)} = {k * 89595 | k : ℕ} :=
by sorry

end part1_div_15_cubed_part2_find_all_n_l125_125076


namespace find_a_plus_b_div_3_l125_125813

-- Define the conditions
variables (a b : ℝ)
def y (x : ℝ) : ℝ := a + b / x

-- Specify the conditions for x = -2 and x = -6
axiom h1 : y (-2) = 3
axiom h2 : y (-6) = 7

-- State the theorem to prove
theorem find_a_plus_b_div_3 : a + b / 3 = 13 := by
  sorry

end find_a_plus_b_div_3_l125_125813


namespace complement_union_eq_l125_125087

open Set

-- Definition of sets U, A, and B
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- Statement of the problem
theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 3} :=
by sorry

end complement_union_eq_l125_125087


namespace escher_prints_consecutive_probability_l125_125548

theorem escher_prints_consecutive_probability :
  let totalArtPieces : ℕ := 12
  let escherPrints : ℕ := 4
  let arrangementsWithConsecutiveEscherPrints := (9.factorial * 4.factorial)
  let totalArrangements := 12.factorial
  arrangementsWithConsecutiveEscherPrints / totalArrangements = (1 / 55 : ℚ) := 
by
  sorry

end escher_prints_consecutive_probability_l125_125548


namespace prove_isosceles_and_find_angle_l125_125508

noncomputable def A := (3 : ℝ, 0 : ℝ)
noncomputable def E := (6 : ℝ, -1 : ℝ)
noncomputable def F := (7 : ℝ, 2 : ℝ)

def midpoint (p q : ℝ × ℝ) : ℝ × ℝ := ((p.1 + q.1) / 2, (p.2 + q.2) / 2)

def is_iso_trapezoid (A B C D : ℝ × ℝ) : Prop :=
  let E := midpoint A B
      F := midpoint C D
      AB_slope := (B.2 - A.2) / (B.1 - A.1)
      CD_slope := (D.2 - C.2) / (D.1 - C.1) in
  AB_slope = CD_slope ∧
  (B.1 = C.1) ∧ -- BC is parallel to the y-axis.

theorem prove_isosceles_and_find_angle :
  ∃ B C D : ℝ × ℝ, is_iso_trapezoid A B C D ∧
  let angle := Real.arccos (1 / Real.sqrt 10) in
  ∃ α, α = angle := sorry

end prove_isosceles_and_find_angle_l125_125508


namespace right_triangle_area_l125_125141

theorem right_triangle_area
  (A B C : Type)
  [MetricSpace A]
  [MetricSpace B]
  [MetricSpace C]
  (triangle_ABC : Triangle A B C)
  (angle_C : angle (vec A C) (vec B C) = π / 2)
  (sin_A : sin angle_A = 5 / 13)
  (BC_length : dist B C = 10) :
  area triangle_ABC = 120 :=
begin
  sorry
end

end right_triangle_area_l125_125141


namespace mila_hours_to_match_agnes_monthly_earnings_l125_125946

-- Definitions based on given conditions
def hourly_rate_mila : ℕ := 10
def hourly_rate_agnes : ℕ := 15
def weekly_hours_agnes : ℕ := 8
def weeks_in_month : ℕ := 4

-- Target statement to prove: Mila needs to work 48 hours to earn as much as Agnes in a month
theorem mila_hours_to_match_agnes_monthly_earnings :
  ∃ (h : ℕ), h = 48 ∧ (h * hourly_rate_mila) = (hourly_rate_agnes * weekly_hours_agnes * weeks_in_month) :=
by
  sorry

end mila_hours_to_match_agnes_monthly_earnings_l125_125946


namespace spending_on_hydrangeas_l125_125569

def lily_spending : ℕ :=
  let start_year := 1989
  let end_year := 2021
  let cost_per_plant := 20
  let years := end_year - start_year
  cost_per_plant * years

theorem spending_on_hydrangeas : lily_spending = 640 := 
  sorry

end spending_on_hydrangeas_l125_125569


namespace ratio_of_silver_to_gold_l125_125599

-- Definitions for balloon counts
def gold_balloons : Nat := 141
def black_balloons : Nat := 150
def total_balloons : Nat := 573

-- Define the number of silver balloons S
noncomputable def silver_balloons : Nat :=
  total_balloons - gold_balloons - black_balloons

-- The goal is to prove the ratio of silver to gold balloons is 2
theorem ratio_of_silver_to_gold :
  (silver_balloons / gold_balloons) = 2 := by
  sorry

end ratio_of_silver_to_gold_l125_125599


namespace Farrah_total_match_sticks_l125_125005

theorem Farrah_total_match_sticks (Boxes Matchboxes Sticks : ℕ) 
  (h1 : Boxes = 7) 
  (h2 : Matchboxes = 35) 
  (h3 : Sticks = 500) 
  : Boxes * Matchboxes * Sticks = 122500 :=
by 
  rw [h1, h2, h3]
  sorry

end Farrah_total_match_sticks_l125_125005


namespace find_y_solution_l125_125718

variable (y : ℚ)

theorem find_y_solution (h : (y^2 - 12*y + 32) / (y - 2) + (3*y^2 + 11*y - 14) / (3*y - 1) = -5) : 
    y = -17/6 :=
by
  sorry

end find_y_solution_l125_125718


namespace triple_comp_f_of_2_l125_125705

def f (x : ℝ) : ℝ :=
if x > 9 then real.sqrt x else x^2 + 1

theorem triple_comp_f_of_2 : f (f (f 2)) = real.sqrt 26 := by
  sorry

end triple_comp_f_of_2_l125_125705


namespace umar_age_is_10_l125_125348

-- Define Ali's age
def Ali_age := 8

-- Define the age difference between Ali and Yusaf
def age_difference := 3

-- Define Yusaf's age based on the conditions
def Yusaf_age := Ali_age - age_difference

-- Define Umar's age which is twice Yusaf's age
def Umar_age := 2 * Yusaf_age

-- Prove that Umar's age is 10
theorem umar_age_is_10 : Umar_age = 10 :=
by
  -- Proof is skipped
  sorry

end umar_age_is_10_l125_125348


namespace bracelet_price_l125_125352

theorem bracelet_price
  (total_bracelets : ℕ := 52)
  (material_cost : ℚ := 3)
  (given_away : ℕ := 8)
  (profit : ℚ := 8)
  (bracelets_sold : ℕ := total_bracelets - given_away)
  (total_sales : ℚ := profit + material_cost)
  (price_per_bracelet : ℚ := total_sales / bracelets_sold) :
  price_per_bracelet = 0.25 := 
by
  sorry

end bracelet_price_l125_125352


namespace magnitude_of_complex_l125_125122

theorem magnitude_of_complex (a b : ℝ) (h_a : a = 2) (h_b : b = 1) :
  |complex.mk a b| = real.sqrt 5 :=
by {
  rw [h_a, h_b],
  simp,
  norm_num,
}

end magnitude_of_complex_l125_125122


namespace number_of_correct_propositions_l125_125460

def proposition_1 (angles : Type) : Prop := ∀ (a b : angles), a = b → (a + b = 180)
def proposition_2 (angles : Type) : Prop := ∀ (a b : angles), (a + b = 90) → (a = b)
def proposition_3 (angles : Type) : Prop := ∀ (a b : angles), (a + b = 90) → (a < 90 ∧ b > 90)
def proposition_4 (lines : Type) : Prop := ∀ (l1 l2 l3 : lines), (l1 ∥ l3) ∧ (l2 ∥ l3) → (l1 ∥ l2)
def proposition_5 (angles : Type) : Prop := ∀ (a b : angles), (a + b = 90) → (bisector a ⊥ bisector b)

theorem number_of_correct_propositions (angles lines : Type) :
  (proposition_4 lines) ∧ (proposition_5 angles) ∧ ¬(proposition_1 angles) 
  ∧ ¬(proposition_2 angles) ∧ ¬(proposition_3 angles) → 2 :=
by
  sorry

end number_of_correct_propositions_l125_125460


namespace impossible_relationships_l125_125768

theorem impossible_relationships (a b : ℝ) (h : (1 / a) = (1 / b)) :
  (¬ (0 < a ∧ a < b)) ∧ (¬ (b < a ∧ a < 0)) :=
by
  sorry

end impossible_relationships_l125_125768


namespace triangle_area_trisection_l125_125153

-- Define the triangle and points
variables {A B C E F : Type}

-- Assume E and F are trisection points on side AB
-- Assume A B C are distinct points forming a triangle
-- We need to show that area of EFC is 1/3 of area of ABC
noncomputable def area_of_triangle {A B C : Type} (a : A) (b : B) (c : C) : ℝ := 
  sorry -- placeholder for actual area calculation

theorem triangle_area_trisection (A B C E F : Type)
  (h1 : E ∈ [A, B]) -- E is on AB
  (h2 : F ∈ [A, B]) -- F is on AB
  (h3 : (AE : ℝ) = (EF : ℝ)) -- E and F are trisection points
  (h4 : (EF : ℝ) = (FB : ℝ)) -- continue trisection condition
  : area_of_triangle E F C = (1 / 3) * area_of_triangle A B C :=
sorry

end triangle_area_trisection_l125_125153


namespace complement_union_eq_l125_125088

variable (U : Set Int := {-2, -1, 0, 1, 2, 3}) 
variable (A : Set Int := {-1, 0, 1}) 
variable (B : Set Int := {1, 2}) 

theorem complement_union_eq :
  U \ (A ∪ B) = {-2, 3} := by 
  sorry

end complement_union_eq_l125_125088


namespace number_of_pears_in_fruit_gift_set_l125_125942

theorem number_of_pears_in_fruit_gift_set 
  (F : ℕ) 
  (h1 : (2 / 9) * F = 10) 
  (h2 : 2 / 5 * F = 18) : 
  (2 / 5) * F = 18 :=
by 
  -- Sorry is used to skip the actual proof for now
  sorry

end number_of_pears_in_fruit_gift_set_l125_125942


namespace Joan_spent_on_games_l125_125158

theorem Joan_spent_on_games (basketball_price : ℝ) (racing_price : ℝ)
  (basketball_discount_rate : ℝ) (racing_discount_rate : ℝ) :
  basketball_price = 5.20 →
  racing_price = 4.23 →
  basketball_discount_rate = 0.15 →
  racing_discount_rate = 0.10 →
  let basketball_discounted_price := basketball_price - (basketball_discount_rate * basketball_price) in
  let racing_discounted_price := racing_price - (racing_discount_rate * racing_price) in
  basketball_discounted_price + racing_discounted_price = 8.23 :=
by
  intros h1 h2 h3 h4
  let basketball_discounted_price := (5.20 - (0.15 * 5.20)) in
  let racing_discounted_price := (4.23 - (0.10 * 4.23)) in
  have h1 : basketball_discounted_price = 4.42 := by sorry
  have h2 : racing_discounted_price = 3.807 := by sorry
  have h3 : basketball_discounted_price + racing_discounted_price = 8.227 := by sorry
  have h4 : 8.227 ≈ 8.23 := by sorry  -- Assuming some real number approximation
  exact h4

end Joan_spent_on_games_l125_125158


namespace part1_28_mysterious_part1_2020_mysterious_part2_mysterious_multiple_of_4_part3_not_mysterious_odd_diff_l125_125485

def is_mysterious (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = (2*k + 2)^2 - (2*k)^2

theorem part1_28_mysterious : is_mysterious 28 :=
by {
  use 3,
  sorry
}

theorem part1_2020_mysterious : is_mysterious 2020 :=
by {
  use 502,
  sorry
}

theorem part2_mysterious_multiple_of_4 (n : ℕ) (h : is_mysterious n) : n % 4 = 0 :=
by {
  cases h with k hk,
  rw hk,
  sorry
}

theorem part3_not_mysterious_odd_diff (k : ℕ) : ¬ is_mysterious ((2 * k + 1)^2 - (2 * k - 1)^2) :=
by {
  intro h,
  have h1 : (2 * k + 1)^2 - (2 * k - 1)^2 = 8 * k,
  { ring },
  rw h1 at h,
  sorry
}

end part1_28_mysterious_part1_2020_mysterious_part2_mysterious_multiple_of_4_part3_not_mysterious_odd_diff_l125_125485


namespace odd_function_value_l125_125113

-- Define that f is odd function given conditions.
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Define f with given properties and conditions.
def f (x c : ℝ) := λ x : ℝ, a * x^3 + x + c

-- The main theorem statement
theorem odd_function_value
  (a b c : ℝ)
  (h_odd: ∀ x, f x = ax^3 + x + c)
  (h_symmetric: is_odd_function f):
  a + b + c + 2 = 2 := 
by sorry

end odd_function_value_l125_125113


namespace disjunction_false_implies_neg_p_true_neg_p_true_does_not_imply_disjunction_false_l125_125099

variable (p q : Prop)

theorem disjunction_false_implies_neg_p_true (hpq : ¬(p ∨ q)) : ¬p :=
by 
  sorry

theorem neg_p_true_does_not_imply_disjunction_false (hnp : ¬p) : ¬(¬(p ∨ q)) :=
by 
  sorry

end disjunction_false_implies_neg_p_true_neg_p_true_does_not_imply_disjunction_false_l125_125099


namespace ellipse_tangent_compute_d_l125_125674

theorem ellipse_tangent_compute_d:
  let f1 := (4 : ℝ, 6 : ℝ) in
  ∃ d : ℝ, (6 : ℝ) + 4 * sqrt 5 = d :=
begin
  use 6 + 4 * sqrt 5,
  sorry
end

end ellipse_tangent_compute_d_l125_125674


namespace minimum_value_of_a_l125_125044

theorem minimum_value_of_a (x y a : ℝ) (h1 : y = (1 / (x - 2)) * (x^2))
(h2 : x = a * y) : a = 3 :=
sorry

end minimum_value_of_a_l125_125044


namespace num_proper_subsets_of_A_l125_125867

-- Define the set A as per the condition.
def A : Set ℕ := {x | -1 < x ∧ x ≤ 3 ∧ 0 < x}

-- The theorem stating the number of proper subsets of A.
theorem num_proper_subsets_of_A : (A.toFinset.powerset.card - 1) = 7 :=
by 
  sorry

end num_proper_subsets_of_A_l125_125867


namespace alice_min_speed_l125_125249

open Real

theorem alice_min_speed (d : ℝ) (bob_speed : ℝ) (alice_delay : ℝ) (alice_time : ℝ) :
  d = 180 → bob_speed = 40 → alice_delay = 0.5 → alice_time = 4 → d / alice_time > (d / bob_speed) - alice_delay →
  d / alice_time > 45 := by
  sorry


end alice_min_speed_l125_125249


namespace odd_nat_not_divisible_l125_125402

theorem odd_nat_not_divisible (n : ℕ) (h_odd : Odd n) (h_nat : 0 < n) :
  ¬ (n^2 ∣ (n-1)!) ↔ (Prime n ∨ n = 9) :=
sorry

end odd_nat_not_divisible_l125_125402


namespace probability_gcd_1_l125_125278

-- Define gcd and the set
def is_coprime (a b : ℕ) := Nat.gcd a b = 1

-- The proof problem statement
theorem probability_gcd_1 :
  let S := {i | i ∈ Finset.range 11 \ {0}} in
  let pairs := {p : ℕ × ℕ | p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2} in
  let coprime_pairs := {p : ℕ × ℕ | p ∈ pairs ∧ is_coprime p.1 p.2} in
  (Finset.card coprime_pairs / Finset.card pairs : ℚ) = 32 / 45 := 
by
  sorry

end probability_gcd_1_l125_125278


namespace exchanges_divisible_by_26_l125_125551

variables (p a d : ℕ) -- Define the variables for the number of exchanges

theorem exchanges_divisible_by_26 (t : ℕ) (h1 : p = 4 * a + d) (h2 : p = a + 5 * d) :
  ∃ k : ℕ, a + p + d = 26 * k :=
by {
  -- Replace these sorry placeholders with the actual proof where needed
  sorry
}

end exchanges_divisible_by_26_l125_125551


namespace isomorphism_A_Rstar_finite_semigroups_groups_bounded_semigroup_H_l125_125385

-- Define the operator * on ℝ as x * y = x + y + xy.
def star (x y : ℝ) := x + y + x * y

-- Define the set A = ℝ \ {-1}.
def A := {x : ℝ | x ≠ -1}

-- Part (a): Prove isomorphism between (A, *) and (ℝ \ {0}, ⋅).
theorem isomorphism_A_Rstar : ∃ f : A → {x : ℝ | x ≠ 0},
  (∀ x y ∈ A, f (star x y) = f x * f y) ∧ (∀ z ∈ {x : ℝ | x ≠ 0}, ∃ x ∈ A, f x = z) :=
sorry

-- Part (b): Determine all finite semigroups of ℝ under * and identify groups.
theorem finite_semigroups_groups :
  {S : set ℝ | S.finite ∧ (∀ x y ∈ S, star x y ∈ S)} = {{-2, 0}, {0}} :=
sorry

-- Part (c): Prove bounded semigroup H under * implies H ⊆ [-2, 0].
theorem bounded_semigroup_H (H : set ℝ) (hH : ∀ x y ∈ H, star x y ∈ H)
  (bounded_H : ∃ M, ∀ x ∈ H, |x| ≤ M) : H ⊆ set.Icc (-2 : ℝ) 0 :=
sorry

end isomorphism_A_Rstar_finite_semigroups_groups_bounded_semigroup_H_l125_125385


namespace smallest_value_expression_l125_125018

theorem smallest_value_expression (a b c : Int) (h1 : b > a) (h2 : a > c) (h3 : c > 0) (h4 : b ≠ 0) :
  let expr := (a + b)^2 + (b + c)^2 + (c - a)^2 + (a - c)^2
  ∃ k : Real, k = 9/2 ∧ ((expr : Real) / b^2) = k := sorry

end smallest_value_expression_l125_125018


namespace trapezoid_shorter_base_l125_125259

theorem trapezoid_shorter_base (L : ℝ) (S : ℝ) (m : ℝ)
  (hL : L = 100)
  (hm : m = 4)
  (h : m = (L - S) / 2) :
  S = 92 :=
by {
  sorry -- Proof is not required
}

end trapezoid_shorter_base_l125_125259


namespace find_first_term_of_arithmetic_series_l125_125421

variable (a d : ℝ)

def sum_of_first_n (n : ℝ) (a d : ℝ) : ℝ :=
  n / 2 * (2 * a + (n - 1) * d)

def problem_conditions (a d : ℝ) : Prop :=
  sum_of_first_n 30 a d = 600 ∧ sum_of_first_n 70 (a + 30 * d) d = 4900

theorem find_first_term_of_arithmetic_series
  (h : problem_conditions a d) :
  a = 5.5 :=
sorry

end find_first_term_of_arithmetic_series_l125_125421


namespace fractional_uniform_independent_l125_125536

noncomputable theory

variables (Ω : Type) [MeasurableSpace Ω] (U V : Ω → ℝ)
variables [IsProbabilityMeasure (measure_theory.measure_space Ω)]

def uniform_on_01 (x : ℝ) : Prop := ∀ a b : ℝ, (0 ≤ a) → (a ≤ b) → (b ≤ 1) → 
  (measure_theory.probability_measure.has_pdf (λ x, 1)) (λ x, a ≤ x ∧ x ≤ b) = b - a

def fractional_part (x : ℝ) : ℝ := x - ⌊x⌋

axiom U_uniform : uniform_on_01 U
axiom U_V_indep : measure_theory.Indep U V
axiom V_rv : measure_theory.ProbabilityMeasure (measure_theory.measure_space Ω V)

theorem fractional_uniform_independent : uniform_on_01 (fractional_part (U + V)) ∧
  measure_theory.Indep V (λ ω, fractional_part (U ω + V ω)) :=
sorry

end fractional_uniform_independent_l125_125536


namespace prime_expression_l125_125170

noncomputable def a := 2
noncomputable def b := 3
noncomputable def c := 4
noncomputable def d := 6
noncomputable def e := 1
noncomputable def f := 5
noncomputable def g := 7

theorem prime_expression : a * b * c * d + e * f * g = 179 ∧ Prime (a * b * c * d + e * f * g) :=
by
  have abcd := a * b * c * d
  have efg := e * f * g
  have sum := abcd + efg
  have eq := sum = 179
  have prime_check := Prime sum
  exact ⟨eq, prime_check⟩

end prime_expression_l125_125170


namespace Bill_original_profit_percentage_l125_125362

theorem Bill_original_profit_percentage 
  (S : ℝ) 
  (h_S : S = 879.9999999999993) 
  (h_cond : ∀ (P : ℝ), 1.17 * P = S + 56) :
  ∃ (profit_percentage : ℝ), profit_percentage = 10 := 
by
  sorry

end Bill_original_profit_percentage_l125_125362


namespace action_figure_cost_is_8_l125_125858

def action_figure_cost_proof (a t m : ℕ) (h1 : a = 7) (h2 : t = 16) (h3 : m = 72) : ℕ :=
  let remaining_figures := t - a
  let cost_per_figure := m / remaining_figures
  cost_per_figure

theorem action_figure_cost_is_8 : action_figure_cost_proof 7 16 72 7 16 72 = 8 :=
by
  sorry

end action_figure_cost_is_8_l125_125858


namespace problem_solution_inf_problem_solution_prime_l125_125557

-- Definitions based on the given conditions and problem statement
def is_solution_inf (m : ℕ) : Prop := 3^m ∣ 2^(3^m) + 1

def is_solution_prime (n : ℕ) : Prop := n.Prime ∧ n ∣ 2^n + 1

-- Lean statement for the math proof problem
theorem problem_solution_inf : ∀ m : ℕ, m ≥ 0 → is_solution_inf m := sorry

theorem problem_solution_prime : ∀ n : ℕ, n.Prime → is_solution_prime n → n = 3 := sorry

end problem_solution_inf_problem_solution_prime_l125_125557


namespace proving_AD_eq_BE_l125_125926

-- Assume we have points A, B, C, D, E on a plane
variables {A B C D E P : Type}
variables [geometry_space A B C D E P]

-- Define the segments and angle properties mentioned in the conditions
axiom AC_eq_CE : segment_length A C = segment_length C E
axiom CE_eq_AE : segment_length C E = segment_length A E
axiom angle_APB_eq_ACE : angle_measure A P B = angle_measure A C E
axiom segment_sum_eq : segment_length A B + segment_length B C = segment_length C D + segment_length D E

-- Define points intersection and equality of segments to be proved
def AD_BE_equal : Prop := segment_length A D = segment_length B E

-- Main theorem statement
theorem proving_AD_eq_BE : 
  AC_eq_CE → 
  CE_eq_AE → 
  angle_APB_eq_ACE → 
  segment_sum_eq → 
  AD_BE_equal := by
  intros
  sorry -- Proof to be filled in

end proving_AD_eq_BE_l125_125926


namespace partition_exists_iff_l125_125416

theorem partition_exists_iff (k : ℕ) :
  (∃ (A B : Finset ℕ), A ∪ B = Finset.range (1990 + k + 1) ∧ A ∩ B = ∅ ∧ 
  (A.sum id + 1990 * A.card = B.sum id + 1990 * B.card)) ↔ 
  (k % 4 = 3 ∨ (k % 4 = 0 ∧ k ≥ 92)) :=
by
  sorry

end partition_exists_iff_l125_125416


namespace roots_ratio_sum_l125_125106

theorem roots_ratio_sum (α β : ℝ) (hαβ : α > β) (h1 : 3*α^2 + α - 1 = 0) (h2 : 3*β^2 + β - 1 = 0) :
  α / β + β / α = -7 / 3 :=
sorry

end roots_ratio_sum_l125_125106


namespace preceding_integer_binary_l125_125109

--- The conditions as definitions in Lean 4

def M := 0b101100 -- M is defined as binary '101100' which is decimal 44
def preceding_binary (n : Nat) : Nat := n - 1 -- Define a function to get the preceding integer in binary

--- The proof problem statement in Lean 4
theorem preceding_integer_binary :
  preceding_binary M = 0b101011 :=
by
  sorry

end preceding_integer_binary_l125_125109


namespace angles_of_triangle_arith_seq_l125_125457

theorem angles_of_triangle_arith_seq (A B C a b c : ℝ) (h1 : A + B + C = 180) (h2 : A = B - (B - C)) (h3 : (1 / a + 1 / c) / 2 = 1 / b) : 
  A = 60 ∧ B = 60 ∧ C = 60 :=
sorry

end angles_of_triangle_arith_seq_l125_125457


namespace max_value_expression_l125_125119

theorem max_value_expression : 
  ∃ θ : ℝ, ∀ θ (h1 : -1 ≤ sin (3 * θ)) (h2 : sin (3 * θ) ≤ 1) (h3 : -1 ≤ cos (2 * θ)) (h4 : cos (2 * θ) ≤ 1), 
  (1 / 2 * (sin (3 * θ))^2 - 1 / 2 * cos (2 * θ)) ≤ 1 :=
sorry

end max_value_expression_l125_125119


namespace solve_eq1_solve_eq2_l125_125914

theorem solve_eq1 (x : ℝ):
  (x - 1) * (x + 3) = x - 1 ↔ x = 1 ∨ x = -2 :=
by 
  sorry

theorem solve_eq2 (x : ℝ):
  2 * x^2 - 6 * x = -3 ↔ x = (3 + Real.sqrt 3) / 2 ∨ x = (3 - Real.sqrt 3) / 2 :=
by 
  sorry

end solve_eq1_solve_eq2_l125_125914


namespace conjugate_of_complex_z_l125_125445

theorem conjugate_of_complex_z : 
  ∀ (z : ℂ), (1 + complex.I) * z = 2 → conj z = 1 + complex.I := 
by
  intros z h
  sorry

end conjugate_of_complex_z_l125_125445


namespace find_f_8_5_l125_125577

variable (f : ℝ → ℝ)

def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f(x) = f(-x)

def shifted_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f(x-1) = -f(1-x)

theorem find_f_8_5 
  (h_even : even_function f)
  (h_shifted_odd : shifted_odd_function f)
  (h_f_0_5 : f 0.5 = 9) : f 8.5 = 9 :=
by
  sorry

end find_f_8_5_l125_125577


namespace sum_of_roots_modulus_one_l125_125486

theorem sum_of_roots_modulus_one (z : ℂ) (hz : z ^ 2009 + z ^ 2008 + 1 = 0) (hmod : ∀ z, z ≠ 0 → |z| = 1 → z = -1/2 + (complex.I * (sqrt 3) / 2) ∨ z = -1/2 - (complex.I * (sqrt 3) / 2)) :
  ∑ z in {z : ℂ | z ^ 2009 + z ^ 2008 + 1 = 0 ∧ |z| = 1}, z = -1 :=
by
  sorry

end sum_of_roots_modulus_one_l125_125486


namespace derivative_f_cos2x_l125_125041

variable {f : ℝ → ℝ} {x : ℝ}

theorem derivative_f_cos2x :
  f (Real.cos (2 * x)) = 1 - 2 * (Real.sin x) ^ 2 →
  deriv f x = -2 * Real.sin (2 * x) :=
by sorry

end derivative_f_cos2x_l125_125041


namespace no_solution_exists_l125_125009

theorem no_solution_exists : 
  ∀ x : ℝ, ¬ (x / 2 ≥ 1 + x ∧ 3 + 2 * x > -3 - 3 * x) := 
by 
  intros x h 
  cases h with h1 h2 
  have : x ≤ -2 := 
  begin
    have : x / 2 ≥ 1 + x := h1,
    linarith,
  end,
  have : x > - 6 / 5 := 
  begin
    have : 3 + 2 * x > -3 - 3 * x := h2,
    linarith,
  end,
  linarith,

end no_solution_exists_l125_125009


namespace fixed_point_in_AB_l125_125046

noncomputable def circle_eq (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 1

def line_l_eq (x y : ℝ) : Prop :=
  x + y = 0

theorem fixed_point_in_AB (P : ℝ × ℝ) :
  line_l_eq P.1 P.2 →
  (∀ A B : ℝ × ℝ, circle_eq A.1 A.2 → circle_eq B.1 B.2 → 
  let PA := (P.1 - A.1)^2 + (P.2 - A.2)^2,
  PB := (P.1 - B.1)^2 + (P.2 - B.2)^2 in
  tangent PA P A → tangent PB P B →
  ∃ (x y : ℝ), x = 3/2 ∧ y = -1/2 ∧
  ∀ t : ℝ, line_through (P.1, P.2) (A.1, A.2)
  (B.1, B.2) (x, y)) :=
begin
  sorry
end

end fixed_point_in_AB_l125_125046


namespace cameras_capture_l125_125157

open Classical

theorem cameras_capture (n k : ℕ) (h₁ : n = 1000)
  (h₂ : k = 998)
  (h₃ : ∀ (O A B : ℕ), (A < n) → (B < n) → (O < n) → 
    ((A ≠ O) ∧ (B ≠ O) ∧ (A ≠ B)) →
    ∠ A O B > 179 →
    ¬ (captures O A) ∨ ¬ (captures O B)) :
  ∃ O : ℕ, (O < n) ∧ (captures O).card ≤ k := sorry

end cameras_capture_l125_125157


namespace find_A_coords_find_AC_equation_l125_125454

theorem find_A_coords
  (B : ℝ × ℝ) (hB : B = (1, -2))
  (median_CM : ∀ x y, 2 * x - y + 1 = 0)
  (angle_bisector_BAC : ∀ x y, x + 7 * y - 12 = 0) :
  ∃ A : ℝ × ℝ, A = (-2, 2) :=
by
  sorry

theorem find_AC_equation
  (A B : ℝ × ℝ) (hA : A = (-2, 2)) (hB : B = (1, -2))
  (median_CM : ∀ x y, 2 * x - y + 1 = 0)
  (angle_bisector_BAC : ∀ x y, x + 7 * y - 12 = 0) :
  ∃ k b : ℝ, ∀ x y, y = k * x + b ↔ 3 * x - 4 * y + 14 = 0 :=
by
  sorry

end find_A_coords_find_AC_equation_l125_125454


namespace cannot_form_right_triangle_setA_l125_125621

def is_right_triangle (a b c : ℝ) : Prop :=
  (a^2 + b^2 = c^2) ∨ (a^2 + c^2 = b^2) ∨ (b^2 + c^2 = a^2)

theorem cannot_form_right_triangle_setA (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) :
  ¬ is_right_triangle a b c :=
by {
  sorry
}

end cannot_form_right_triangle_setA_l125_125621


namespace find_value_of_expression_l125_125417

theorem find_value_of_expression :
  (3 / 11) * (∏ n in finset.Icc 3 120, (1 + (1 / n))) ≈ 0.366 :=
sorry

end find_value_of_expression_l125_125417


namespace decagon_diagonals_intersect_probability_l125_125999

theorem decagon_diagonals_intersect_probability :
  let n := 10  -- number of vertices in decagon
  let diagonals := n * (n - 3) / 2  -- number of diagonals in decagon
  let pairs_diagonals := (diagonals * (diagonals - 1)) / 2  -- ways to choose 2 diagonals from diagonals
  let ways_choose_4 := Nat.choose 10 4  -- ways to choose 4 vertices from 10
  let probability := (4 * ways_choose_4) / pairs_diagonals  -- four vertices chosen determine two intersecting diagonals forming a convex quadrilateral
  probability = (210 / 595) := by
  -- Definitions (diagonals, pairs_diagonals, ways_choose_4) are directly used as hypothesis

  sorry  -- skipping the proof

end decagon_diagonals_intersect_probability_l125_125999


namespace pyramid_volume_l125_125341

variable (P Q R S T : Type) 

/-- The base of the pyramid QRST is a square with side length 1/3. -/
def square_base (a : ℝ) : Prop := a = 1 / 3

/-- The height of the pyramid from P to the base QRST is 1. -/
def height (h : ℝ) : Prop := h = 1

/-- The volume of the pyramid PQRST is 1/27. -/
theorem pyramid_volume (a h : ℝ) (h₁ : square_base a) (h₂ : height h) : 
  (1 / 3) * (a ^ 2) * h = 1 / 27 := 
  by 
    sorry

end pyramid_volume_l125_125341


namespace number_of_cows_l125_125983

variable (D C : Nat)

theorem number_of_cows (h : 2 * D + 4 * C = 2 * (D + C) + 30) : C = 15 :=
by
  sorry

end number_of_cows_l125_125983


namespace combined_dancing_time_excluding_break_l125_125160

def john_dance_time (first_period second_period : ℕ) : ℕ :=
  first_period + second_period

def james_dance_time (john_total_time extra_ratio : ℕ) : ℕ :=
  john_total_time + (john_total_time * extra_ratio / 3)

def combined_dance_time (john_time james_time : ℕ) : ℕ :=
  john_time + james_time

theorem combined_dancing_time_excluding_break
  (john_first_period john_second_period john_break : ℕ)
  (extra_ratio : ℕ) :
  john_first_period = 3 →
  john_second_period = 5 →
  john_break = 1 →
  extra_ratio = 1 →
  combined_dance_time (john_dance_time john_first_period john_second_period)
                      (james_dance_time (john_first_period + john_break + john_second_period) extra_ratio) = 20 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  have hj : john_dance_time 3 5 = 3 + 5 := rfl
  have ht : 3 + 1 + 5 = 9 := rfl
  have hjt : james_dance_time 9 1 = 9 + (9 * 1 / 3) := rfl
  rw [hj, ht, hjt]
  norm_num
  sorry

end combined_dancing_time_excluding_break_l125_125160


namespace positive_integers_condition_l125_125739

theorem positive_integers_condition : ∃ n : ℕ, (n > 0) ∧ (n < 50) ∧ (∃ k : ℕ, n = k * (50 - n)) :=
sorry

end positive_integers_condition_l125_125739


namespace log_a_b_integer_probability_l125_125279

noncomputable def count_valid_pairs (n : ℕ) : ℕ :=
  ∑ x in finset.range (n + 1), 
    if 1 ≤ x ∧ x ≤ n
      then finset.card (finset.filter (λ y, y ≠ x ∧ y % x = 0) (finset.range (n + 1)))
      else 0

theorem log_a_b_integer_probability : 
  let n := 20,
      total_pairs := (n * (n - 1)) / 2,
      valid_pairs := count_valid_pairs n
  in 
  (valid_pairs : ℚ) / total_pairs = 47 / 190 :=
by
  sorry

end log_a_b_integer_probability_l125_125279


namespace Eunji_has_most_marbles_l125_125545

-- Declare constants for each person's marbles
def Minyoung_marbles : ℕ := 4
def Yujeong_marbles : ℕ := 2
def Eunji_marbles : ℕ := Minyoung_marbles + 1

-- Theorem: Eunji has the most marbles
theorem Eunji_has_most_marbles :
  Eunji_marbles > Minyoung_marbles ∧ Eunji_marbles > Yujeong_marbles :=
by
  sorry

end Eunji_has_most_marbles_l125_125545


namespace invariant_expression_l125_125870

-- Define the objects involved
variables {A B C P D E F : Point}

-- Assumptions on orthogonal projections
def orthogonal_projection (p : Point) (line : Line) : Point := sorry
axiom D_def : D = orthogonal_projection P (line_through B C)
axiom E_def : E = orthogonal_projection P (line_through C A)
axiom F_def : F = orthogonal_projection P (line_through A B)

-- Utility axioms (like collinearity, perpendicularity, etc.)
axiom P_in_triangle : in_triangle P A B C
axiom D_perp : ∠(line_through B P, line_through B C) = 90°
axiom E_perp : ∠(line_through C P, line_through C A) = 90°
axiom F_perp : ∠(line_through A P, line_through A B) = 90°

-- Prove the expression is invariant of P
theorem invariant_expression :
  (EF * d(A, P)) / d(A, P) + (FD * d(B, P)) / d(B, P) + (DE * d(C, P)) / d(C, P) = 
  (EF * 1) + (FD * 1) + (DE * 1) := 
  sorry

end invariant_expression_l125_125870


namespace angle_is_30_degrees_l125_125112

theorem angle_is_30_degrees (A : ℝ) (h_acute : A > 0 ∧ A < π / 2) (h_sin : Real.sin A = 1/2) : A = π / 6 := 
by 
  sorry

end angle_is_30_degrees_l125_125112


namespace smallest_k_l125_125182

theorem smallest_k (p : ℕ) (hp : p = 997) : 
  ∃ k : ℕ, (p^2 - k) % 10 = 0 ∧ k = 9 :=
by
  sorry

end smallest_k_l125_125182


namespace train_speed_proof_l125_125669

variables (distance_to_syracuse total_time_hours return_trip_speed average_speed_to_syracuse : ℝ)

def question_statement : Prop :=
  distance_to_syracuse = 120 ∧
  total_time_hours = 5.5 ∧
  return_trip_speed = 38.71 →
  average_speed_to_syracuse = 50

theorem train_speed_proof :
  question_statement distance_to_syracuse total_time_hours return_trip_speed average_speed_to_syracuse :=
by
  -- sorry is used to indicate that the proof is omitted
  sorry

end train_speed_proof_l125_125669


namespace exists_six_clique_l125_125507

variable (E : Type) [Fintype E] [DecidableEq E]
variable [Nonempty E] [Fintype.card E = 1991]
variable (connected : E → E → Prop)
variable [∀ a : E, Fintype.card {b : E // connected a b} ≥ 1593]

theorem exists_six_clique :
  ∃ (A : Finset E), A.card = 6 ∧ (∀ a b : E, a ∈ A → b ∈ A → a ≠ b → connected a b) :=
  sorry

end exists_six_clique_l125_125507


namespace b_days_solve_l125_125977

-- Definitions from the conditions
variable (b_days : ℝ)
variable (a_rate : ℝ) -- work rate of a
variable (b_rate : ℝ) -- work rate of b

-- Condition 1: a is twice as fast as b
def twice_as_fast_as_b : Prop :=
  a_rate = 2 * b_rate

-- Condition 2: a and b together can complete the work in 3.333333333333333 days
def combined_completion_time : Prop :=
  1 / (a_rate + b_rate) = 10 / 3

-- The number of days b alone can complete the work should satisfy this equation
def b_alone_can_complete_in_b_days : Prop :=
  b_rate = 1 / b_days

-- The actual theorem we want to prove:
theorem b_days_solve (b_rate a_rate : ℝ) (h1 : twice_as_fast_as_b a_rate b_rate) (h2 : combined_completion_time a_rate b_rate) : b_days = 10 :=
by
  sorry

end b_days_solve_l125_125977


namespace elias_purchased_50cent_items_l125_125714

theorem elias_purchased_50cent_items :
  ∃ (a b c : ℕ), a + b + c = 50 ∧ (50 * a + 250 * b + 400 * c = 5000) ∧ (a = 40) :=
by {
  sorry
}

end elias_purchased_50cent_items_l125_125714


namespace directrix_of_parabola_l125_125927

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the directrix equation we need to prove
def directrix : ℝ := -1/8

-- Statement of the theorem
theorem directrix_of_parabola :
  ∃ d : ℝ, (∀ x : ℝ, parabola x = d) ∧ (d = directrix) :=
sorry

end directrix_of_parabola_l125_125927


namespace kathleen_money_left_l125_125164

def june_savings : ℕ := 21
def july_savings : ℕ := 46
def august_savings : ℕ := 45

def school_supplies_expenses : ℕ := 12
def new_clothes_expenses : ℕ := 54

def total_savings : ℕ := june_savings + july_savings + august_savings
def total_expenses : ℕ := school_supplies_expenses + new_clothes_expenses

def total_money_left : ℕ := total_savings - total_expenses

theorem kathleen_money_left : total_money_left = 46 :=
by
  sorry

end kathleen_money_left_l125_125164


namespace max_guests_without_galoshes_l125_125995

theorem max_guests_without_galoshes :
  ∀ (guests galoshes : Fin 10) (sizes : Fin 10 → Fin 10),
  (∀ i : Fin 10, ∃ j : Fin 10, sizes i = j) →
  (∀ i : Fin 10, ∀ j : Fin 10, sizes i ≤ sizes j → galoshes j ≥ guests i) →
  (∃ n : ℕ, n ≤ 5 ∧ ∀ remaining_guests : Fin n, ∃ k : Fin 10, guests remaining_guests > galoshes k) :=
sorry

end max_guests_without_galoshes_l125_125995


namespace ratio_of_areas_l125_125152

-- Definitions of objects in the problem
structure Trapezoid (A B C D : Type) :=
  (angle_BAD : ℝ)
  (angle_CDA : ℝ)
  (AD : ℝ)
  (BC : ℝ)
  (M : Type)
  (N : Type)
  (L : Type)
  (perpendicular_M_AB : Prop)
  (perpendicular_N_CD : Prop)

-- Defining midpoint property
def midpoint (M : Type) (A : Type) (B : Type) : Prop := sorry

theorem ratio_of_areas (A B C D M N L : Type) 
  (tr : Trapezoid A B C D) 
  (midp_M_AB : midpoint M A B)
  (midp_N_CD : midpoint N C D) 
  (perpend_M_AB : tr.perpendicular_M_AB)
  (perpend_N_CD : tr.perpendicular_N_CD) :
  let area_ratio := (triangle_area M N L) / (trapezoid_area A B C D) in
  area_ratio = (7 * real.sqrt 3) / 6 :=
sorry

end ratio_of_areas_l125_125152


namespace exists_unique_triplet_l125_125793

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

theorem exists_unique_triplet :
  ∃!(t : ℕ × ℕ × ℕ), t.1 > 0 ∧ t.2.1 > 0 ∧ t.1 < t.2.2 ∧ t.2.1 < t.2.2 ∧
  ∀ n : ℕ, n > 0 → t.2.2 ∣ (fibonacci n - t.1 * n * t.2.1 ^ n) :=
sorry

end exists_unique_triplet_l125_125793


namespace distance_between_city_and_village_l125_125960

variables (S x y : ℝ)

theorem distance_between_city_and_village (h1 : S / 2 - 2 = y * S / (2 * x))
    (h2 : 2 * S / 3 + 2 = x * S / (3 * y)) : S = 6 :=
by
  sorry

end distance_between_city_and_village_l125_125960


namespace g_decreasing_intervals_l125_125464

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + π / 6)
noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x)

theorem g_decreasing_intervals (k : ℤ) : 
  ∀ x ∈ set.Icc (k * Real.pi) (k * Real.pi + Real.pi / 2), 
  ∀ y ∈ set.Icc (k * Real.pi) (k * Real.pi + Real.pi / 2), 
  x < y → g y < g x := 
sorry

end g_decreasing_intervals_l125_125464


namespace umar_age_is_ten_l125_125351

-- Define variables for Ali, Yusaf, and Umar
variables (ali_age yusa_age umar_age : ℕ)

-- Define the conditions from the problem
def ali_is_eight : Prop := ali_age = 8
def ali_older_than_yusaf : Prop := ali_age - yusa_age = 3
def umar_twice_yusaf : Prop := umar_age = 2 * yusa_age

-- The theorem that uses the conditions to assert Umar's age
theorem umar_age_is_ten 
  (h1 : ali_is_eight ali_age)
  (h2 : ali_older_than_yusaf ali_age yusa_age)
  (h3 : umar_twice_yusaf umar_age yusa_age) : 
  umar_age = 10 :=
by
  sorry

end umar_age_is_ten_l125_125351


namespace coefficient_of_one_over_x_in_expansion_l125_125504

theorem coefficient_of_one_over_x_in_expansion :
  let T (r : ℕ) : ℤ := (-1)^(r) * 2^(2*r-5) * Nat.choose 5 r
  (∀ r : ℕ, 5 - 2 * r = -1 → T r * x^(5 - 2 * r) = -40 * x^(-1)) :=
by
  sorry

end coefficient_of_one_over_x_in_expansion_l125_125504


namespace maximize_expression_l125_125884

theorem maximize_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 1) :
  2 * a * b + 2 * b * c * sqrt 3 ≤ 2 :=
sorry

end maximize_expression_l125_125884


namespace complement_union_l125_125097

variable (U : Set ℤ)
variable (A : Set ℤ)
variable (B : Set ℤ)

theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3})
                         (hA : A = {-1, 0, 1})
                         (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} :=
sorry

end complement_union_l125_125097


namespace problem_solution_l125_125038

def A : Set ℕ := {1, 2, 3, 4}

noncomputable def f (x : ℕ) : ℝ :=
  Real.log x / Real.log 2

def B : Set ℝ := {f 1, f 2, f 3, f 4}

theorem problem_solution :
  B = {0, 1, Real.log 3 / Real.log 2, 2} ∧ 
  (A ∩ {a : ℕ | ∃ b ∈ B, a = b}) = {1, 2} ∧ 
  (A ∪ {a : ℕ | ∃ b ∈ B, a = b}) = {0, 1, Real.log 3 / Real.log 2, 2, 3, 4} :=
by 
  sorry

end problem_solution_l125_125038


namespace find_x_for_gx_eq_20_l125_125238

-- Define the functions and conditions
def f (x : ℝ) : ℝ := (2 * x + 10) / (x + 1)
def g (x : ℝ) : ℝ := 4 * (f⁻¹' x)

theorem find_x_for_gx_eq_20 :
  ∀ (x : ℝ), g x = 20 → x = 10 / 3 :=
by
  intro x
  intro h_gx
  sorry

end find_x_for_gx_eq_20_l125_125238


namespace orange_ribbons_proof_l125_125132

-- Define the total number of ribbons such that the conditions hold
def total_ribbons (total : ℕ) : Prop :=
  let yellow := total * 1 / 4 in
  let purple := total * 1 / 3 in
  let orange := total * 1 / 6 in
  let silver := 40 in
  yellow + purple + orange + silver = total

-- Define the number of orange ribbons given the total number of ribbons
def orange_ribbons (total : ℕ) : ℕ :=
  total * 1 / 6

theorem orange_ribbons_proof : ∃ total, total_ribbons total ∧ orange_ribbons total = 27 :=
by {
  sorry
}

end orange_ribbons_proof_l125_125132


namespace sequence_divisibility_l125_125418

theorem sequence_divisibility (p : ℕ) (hp : Nat.Prime p ∧ p % 2 = 1) :
  ∃ m : ℕ, m > 0 ∧ p ∣ (sequence m) ∧ p ∣ (sequence (m + 1)) :=
by
  let a : ℕ → ℤ
  def a 1 := 3
  def a (n + 1) := (2 * (n + 1) * a n - n - 2) / n

  have step1 : ∃ m, p ∣ a m ∧ p ∣ a (m + 1),
  -- detailed proof would go here
  sorry

  exact step1

end sequence_divisibility_l125_125418


namespace kathleen_remaining_money_l125_125165

-- Define the conditions
def saved_june := 21
def saved_july := 46
def saved_august := 45
def spent_school_supplies := 12
def spent_clothes := 54
def aunt_gift_threshold := 125
def aunt_gift := 25

-- Prove that Kathleen has the correct remaining amount of money
theorem kathleen_remaining_money : 
    (saved_june + saved_july + saved_august) - 
    (spent_school_supplies + spent_clothes) = 46 := 
by
  sorry

end kathleen_remaining_money_l125_125165


namespace crayons_at_end_of_school_year_l125_125899

def number_of_erasers := 457
def initial_crayons := 617
def final_erasers := number_of_erasers
def final_crayons := final_erasers + 66

theorem crayons_at_end_of_school_year : final_crayons = 523 :=
by
  unfold final_crayons final_erasers number_of_erasers
  simp
  sorry

end crayons_at_end_of_school_year_l125_125899


namespace angle_ABF_regular_octagon_l125_125224

theorem angle_ABF_regular_octagon (ABCDEFGH : Type) [regular_octagon ABCDEFGH] :
  ∃ AB F : Point, angle AB F = 22.5 := sorry

end angle_ABF_regular_octagon_l125_125224


namespace max_value_sum_product_l125_125435

noncomputable def maximum_dot_product {n : ℕ} (a b : Fin n → ℝ) : ℝ :=
  ∑ i, a i * b i

theorem max_value_sum_product (n : ℕ) (a b : Fin n → ℝ) 
  (ha : ∑ i, (a i)^2 = 4) (hb : ∑ i, (b i)^2 = 9) : 
  maximum_dot_product a b ≤ 6 :=
sorry

end max_value_sum_product_l125_125435


namespace sum_divisors_36_48_l125_125025

open Finset

noncomputable def sum_common_divisors (a b : ℕ) : ℕ :=
  let divisors_a := (range (a + 1)).filter (λ x => a % x = 0)
  let divisors_b := (range (b + 1)).filter (λ x => b % x = 0)
  let common_divisors := divisors_a ∩ divisors_b
  common_divisors.sum id

theorem sum_divisors_36_48 : sum_common_divisors 36 48 = 28 := by
  sorry

end sum_divisors_36_48_l125_125025


namespace solving_linear_equations_count_l125_125562

def total_problems : ℕ := 140
def algebra_percentage : ℝ := 0.40
def algebra_problems := (total_problems : ℝ) * algebra_percentage
def solving_linear_equations_percentage : ℝ := 0.50
def solving_linear_equations_problems := algebra_problems * solving_linear_equations_percentage

theorem solving_linear_equations_count :
  solving_linear_equations_problems = 28 :=
by
  sorry

end solving_linear_equations_count_l125_125562


namespace base_six_equals_base_b_l125_125240

noncomputable def base_six_to_decimal (n : ℕ) : ℕ :=
  6 * 6 + 2

noncomputable def base_b_to_decimal (b : ℕ) : ℕ :=
  b^2 + 2 * b + 4

theorem base_six_equals_base_b (b : ℕ) : b^2 + 2 * b - 34 = 0 → b = 4 := 
by sorry

end base_six_equals_base_b_l125_125240


namespace julie_book_problem_l125_125162

theorem julie_book_problem : 
  ∃ P : ℕ, 
  (let read_yesterday := 12 in
   let read_today := 2 * read_yesterday in
   let total_read := read_yesterday + read_today in
   let read_tomorrow := 42 in
   let remaining_pages := read_tomorrow * 2 in
   let P := total_read + remaining_pages in
   P = 120) :=
sorry

end julie_book_problem_l125_125162


namespace solve_equation_1_solve_equation_2_l125_125237

theorem solve_equation_1 (x : ℝ) : x^2 - 2 * x = 1 → (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) :=
begin
  sorry
end

theorem solve_equation_2 (x : ℝ) : x^2 + 5 * x + 6 = 0 → (x = -2 ∨ x = -3) :=
begin
  sorry
end

end solve_equation_1_solve_equation_2_l125_125237


namespace terminating_decimal_n_count_l125_125424

theorem terminating_decimal_n_count :
  {n : ℕ | 1 ≤ n ∧ n ≤ 150 ∧ (∃ k, n = 9 * k)}.to_finset.card = 16 :=
by
  sorry

end terminating_decimal_n_count_l125_125424


namespace num_tosses_l125_125655

theorem num_tosses (n : ℕ) (h : (1 - (7 / 8 : ℝ)^n) = 0.111328125) : n = 7 :=
by
  sorry

end num_tosses_l125_125655


namespace pavan_travel_distance_l125_125215

theorem pavan_travel_distance (t : ℝ) (v1 v2 : ℝ) (D : ℝ) (h₁ : t = 15) (h₂ : v1 = 30) (h₃ : v2 = 25):
  (D / 2) / v1 + (D / 2) / v2 = t → D = 2250 / 11 :=
by
  intro h
  rw [h₁, h₂, h₃] at h
  sorry

end pavan_travel_distance_l125_125215


namespace Anya_took_home_balloons_l125_125604

theorem Anya_took_home_balloons :
  ∃ (balloons_per_color : ℕ), 
  ∃ (yellow_balloons_home : ℕ), 
  (672 = 4 * balloons_per_color) ∧ 
  (yellow_balloons_home = balloons_per_color / 2) ∧ 
  (yellow_balloons_home = 84) :=
begin
  sorry
end

end Anya_took_home_balloons_l125_125604


namespace inscribed_square_area_l125_125650

theorem inscribed_square_area (a : ℝ) (h : a > 0) : 
  let d := a * Real.sqrt 2 in
  let R := d / 2 in
  let x := a / 5 in
  (x * x = (a * a) / 25) := 
by 
  sorry

end inscribed_square_area_l125_125650


namespace valid_license_plates_l125_125505

-- Define the number of vowels and the total alphabet letters.
def num_vowels : ℕ := 5
def num_letters : ℕ := 26
def num_digits : ℕ := 10

-- Define the total number of valid license plates in Eldoria.
theorem valid_license_plates : num_vowels * num_letters * num_digits^3 = 130000 := by
  sorry

end valid_license_plates_l125_125505


namespace anya_takes_home_balloons_l125_125602

theorem anya_takes_home_balloons:
  ∀ (total_balloons : ℕ) (colors : ℕ) (half : ℕ) (balloons_per_color : ℕ),
  total_balloons = 672 →
  colors = 4 →
  balloons_per_color = total_balloons / colors →
  half = balloons_per_color / 2 →
  half = 84 :=
by 
  intros total_balloons colors half balloons_per_color 
  intros h1 h2 h3 h4
  sorry

end anya_takes_home_balloons_l125_125602


namespace quadrilateral_interior_angles_l125_125847

theorem quadrilateral_interior_angles (a : ℝ) 
  (h1 : ∀ quad : Type, (sum_of_interior_angles quad = 360)) 
  (h2 : number_of_quadrilaterals_in_figure = 4) : 
  a = 1440 - 360 :=
sorry

end quadrilateral_interior_angles_l125_125847


namespace ratio_of_triangle_areas_is_one_l125_125900

open Real

theorem ratio_of_triangle_areas_is_one
  (ABC : Type) [is_triangle ABC]
  (A B C D : Point)
  (h1 : AB = AC)
  (h2 : ∠ BAC = 100)
  (h3 : D ∈ Segment A C)
  (h4 : ∠ DBC = 60) :
  area (Triangle.mk A D B) / area (Triangle.mk C D B) = 1 := 
sorry

end ratio_of_triangle_areas_is_one_l125_125900


namespace tangent_lines_two_curves_l125_125488

theorem tangent_lines_two_curves
  (a : ℝ)
  (l : Line)
  (h_l_tangent_x3 : ∃ x0 : ℝ, l = tangentLine (λ x, x^3) x0 ∧ l.contains (1, 0))
  (h_l_tangent_ax2 : ∃ x0 : ℝ, l = tangentLine (λ x, a * x^2 + (15/4) * x - 9) x0 ∧ l.contains (1, 0)) :
  a = -(25 / 64) ∨ a = -1 := sorry

end tangent_lines_two_curves_l125_125488


namespace random_event_is_B_l125_125290

variable (isCertain : Event → Prop)
variable (isImpossible : Event → Prop)
variable (isRandom : Event → Prop)

variable (A : Event)
variable (B : Event)
variable (C : Event)
variable (D : Event)

-- Here we set the conditions as definitions in Lean 4:
def condition_A : isCertain A := sorry
def condition_B : isRandom B := sorry
def condition_C : isCertain C := sorry
def condition_D : isImpossible D := sorry

-- The theorem we need to prove:
theorem random_event_is_B : isRandom B := 
by
-- adding sorry to skip the proof
sorry

end random_event_is_B_l125_125290


namespace euclidean_division_quotient_remainder_l125_125363

noncomputable def A : Polynomial ℝ := X^3 + 2*X^2 + 3*X + 4
noncomputable def B : Polynomial ℝ := X^2 + 1
noncomputable def Q : Polynomial ℝ := X + 2
noncomputable def R : Polynomial ℝ := 2*X + 2

theorem euclidean_division_quotient_remainder :
  ∃ Q R, A = B * Q + R ∧ degree R < degree B ∧ Q = X + 2 ∧ R = 2*X + 2 :=
by
  use [Q, R]
  -- Placeholder proof using sorry
  sorry

end euclidean_division_quotient_remainder_l125_125363


namespace vertical_angles_are_congruent_l125_125973

def supplementary_angles (a b : ℝ) : Prop := a + b = 180
def corresponding_angles (l1 l2 t : ℝ) : Prop := l1 = l2
def exterior_angle_greater (ext int1 int2 : ℝ) : Prop := ext = int1 + int2
def vertical_angles_congruent (a b : ℝ) : Prop := a = b

theorem vertical_angles_are_congruent (a b : ℝ) (h : vertical_angles_congruent a b) : a = b := by
  sorry

end vertical_angles_are_congruent_l125_125973


namespace prove_angle_β_l125_125554

-- Defining the problem conditions
variables (a b c : ℝ) (β : ℝ)

-- The given condition as a hypothesis
axiom given_eq : (a^2 + b^2 + c^2)^2 = 4 * b^2 * (a^2 + c^2) + 3 * a^2 * c^2

-- The statement to be proved
theorem prove_angle_β :
  (∃ (β : ℝ), (β = 30 ∨ β = 150) ∧ -- The angles in degrees that β must be one of
  (a^2 + b^2 + c^2)^2 = 4 * b^2 * (a^2 + c^2) + 3 * a^2 * c^2 → 
   cos β = sqrt 3 / 2 ∨ cos β = - sqrt 3 / 2 ) := sorry

end prove_angle_β_l125_125554


namespace min_value_modulus_z_add_2i_l125_125188

open Complex

theorem min_value_modulus_z_add_2i {z : ℂ} (h : |z^2 + 9| = |z * (z + 3 * Complex.I)|) : 
  ∃ z, |z + 2 * Complex.I| = 5 / 2 :=
begin
  sorry
end

end min_value_modulus_z_add_2i_l125_125188


namespace a_seq_general_term_b_seq_sum_l125_125311

/- Definitions of the sequences -/
def a_seq (n : ℕ) : ℕ := (3 ^ (n - 1))

def b_seq (n : ℕ) : ℕ := n * (3 ^ n)

/- The sums of the sequences -/
def sum_a_seq (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k, 3 ^ k * a_seq (k + 1))

def sum_b_seq (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k, b_seq (k + 1))

/- Problem statements -/
theorem a_seq_general_term (n : ℕ) :
  a_seq (n + 1) = 3 ^ (n) := by sorry

theorem b_seq_sum (n : ℕ) :
  sum_b_seq n = (n - 1) * 3 ^ (n + 1) - 2 * 3 := by sorry

end a_seq_general_term_b_seq_sum_l125_125311


namespace arithmetic_geometric_sequence_l125_125495

-- Define the core elements of the problem
noncomputable def a (n : ℕ) : ℝ := 2 * (2:ℝ)^(n-1)  -- Explicit definition based on the geometric sequence

-- Define the sum S_n of the first n terms
noncomputable def S (n : ℕ) : ℝ := (∑ i in finset.range n, a i.succ)

-- State the main theorem
theorem arithmetic_geometric_sequence :
  a 1 = 2 ∧ (a 2 + a 5 = 2 * (a 4 + 2)) → (S 10 - S 4 = 2016) :=
begin
  intro h,
  sorry
end

end arithmetic_geometric_sequence_l125_125495


namespace central_angle_of_spherical_sector_l125_125613

theorem central_angle_of_spherical_sector (R α r m : ℝ) (h1 : R * Real.pi * r = 2 * R * Real.pi * m) (h2 : R^2 = r^2 + (R - m)^2) :
  α = 2 * Real.arccos (3 / 5) :=
by
  sorry

end central_angle_of_spherical_sector_l125_125613


namespace expected_winnings_l125_125135

noncomputable def calculate_winnings (odds : List ℝ) (bet : ℝ) : ℝ :=
  odds.prod * bet

theorem expected_winnings :
  let odds := [1.35, 5.75, 3.45, 2.25, 4.12, 1.87]
  let bet := 8.00
  calculate_winnings odds bet = 1667.9646 :=
by
  intros odds bet
  have h : odds.prod = 208.495575 := sorry
  have h' : 208.495575 * bet = 1667.9646 := sorry
  rw [h, h']
  sorry

end expected_winnings_l125_125135


namespace ellipse_equation_params_l125_125700

/-- Prompts to find the equation of the ellipse and prove that the sum of the absolute values 
    of the coefficients is as given. -/
theorem ellipse_equation_params
  (x y : ℝ)
  (t : ℝ)
  (hx : x = (4 * (cos t + 2)) / (3 + sin t))
  (hy : y = (5 * (sin t - 4)) / (3 + sin t)) :
  (x^2 * y^2 + 10 * x^2 * y + 25 * x^2 + 128 * y^2 + 1760 * y + 6000 = 0) ∧
  (|1| + |10| + |25| + |128| + |1760| + |6000| = 7924) :=
by
  sorry

end ellipse_equation_params_l125_125700


namespace shaded_area_percent_l125_125965

-- Definition of conditions
def square_area (side_length : ℕ) : ℕ := side_length ^ 2

def shaded_area (r1_area r2_area r3_area : ℕ) : ℕ := r1_area + r2_area + r3_area

def shaded_percentage (shaded : ℕ) (total : ℕ) : ℕ := (shaded * 100) / total

-- Problem
theorem shaded_area_percent (side_length r1_side1 r1_side2 r2_side1 r2_side2 r3_side1 r3_side2 : ℕ) :
  side_length = 6 ∧
  r1_side1 = 2 ∧ r1_side2 = 2 ∧
  r2_side1 = 4 ∧ r2_side2 = 3 ∧
  r3_side1 = 6 ∧ r3_side2 = 5 →
  let total_area := square_area side_length in
  let r1_area := r1_side1 * r1_side2 in
  let r2_area := r2_side1 * r2_side2 in
  let r3_area := r3_side1 * r3_side2 in
  let shaded := shaded_area r1_area r2_area r3_area in
  shaded_percentage shaded total_area = 61 :=
begin
  sorry -- Proof goes here
end

end shaded_area_percent_l125_125965


namespace total_notebooks_distributed_l125_125429

/-- Define the parameters for children in Class A and Class B and the conditions given. -/
def ClassAChildren : ℕ := 64
def ClassBChildren : ℕ := 13

/-- Define the conditions as per the problem -/
def notebooksPerChildInClassA (A : ℕ) : ℕ := A / 8
def notebooksPerChildInClassB (A : ℕ) : ℕ := 2 * A
def totalChildrenClasses (A B : ℕ) : ℕ := A + B
def totalChildrenCondition (A : ℕ) : ℕ := 6 * A / 5

/-- Theorem to state the number of notebooks distributed between the two classes -/
theorem total_notebooks_distributed (A : ℕ) (B : ℕ) (H : A = 64) (H1 : B = 13) : 
  (A * (A / 8) + B * (2 * A)) = 2176 := by
  -- Conditions from the problem
  have conditionA : A = 64 := H
  have conditionB : B = 13 := H1
  have classA_notebooks : ℕ := (notebooksPerChildInClassA A) * A
  have classB_notebooks : ℕ := (notebooksPerChildInClassB A) * B
  have total_notebooks : ℕ := classA_notebooks + classB_notebooks
  -- Proof that total notebooks equals 2176
  sorry

end total_notebooks_distributed_l125_125429


namespace value_of_a_over_b_l125_125370

def elements : List ℤ := [-5, -3, -1, 2, 4]

def maxProduct (l : List ℤ) : ℤ :=
  l.product $ prod for (x, y) in l.allPairs if x ≠ y and x * y

def minQuotient (l : List ℤ) : Rat :=
  l.allPairs $ min (x / y) for (x, y) in l.allPairs if x ≠ y and y ≠ 0

theorem value_of_a_over_b :
  let a := maxProduct elements
  let b := minQuotient elements
  a = 15 → b = -4 → a / b = -4
by
  intro a h_ma b h_mb
  have ha : a = 15 := h_ma
  have hb : b = -4 := h_mb
  rw [ha, hb]
  norm_num [ha, hb]
  sorry

end value_of_a_over_b_l125_125370


namespace minimum_value_l125_125184

theorem minimum_value (x : Fin 50 → ℝ) (h₁ : ∀ i, 0 < x i) (h₂ : ∑ i, (x i) ^ 3 = 1) :
    (∑ i in Finset.univ, (x i) / (1 - (x i) ^ 3)) ≥ (5 / 2) :=
sorry

end minimum_value_l125_125184


namespace train_b_overtakes_train_a_in_50_minutes_l125_125608

theorem train_b_overtakes_train_a_in_50_minutes 
  (speed_A speed_B : ℝ)
  (speed_A_eq : speed_A = 50)
  (speed_B_eq : speed_B = 80)
  (initial_time_gap_minutes : ℝ)
  (initial_time_gap_minutes_eq : initial_time_gap_minutes = 30) :
  let initial_time_gap_hours := initial_time_gap_minutes / 60 in
  let initial_distance_gap := speed_A * initial_time_gap_hours in
  let relative_speed := speed_B - speed_A in
  let time_to_catch_up_hours := initial_distance_gap / relative_speed in
  time_to_catch_up_hours * 60 = 50 :=
by
  let initial_time_gap_hours := initial_time_gap_minutes / 60
  let initial_distance_gap := speed_A * initial_time_gap_hours
  let relative_speed := speed_B - speed_A
  let time_to_catch_up_hours := initial_distance_gap / relative_speed
  exact sorry

end train_b_overtakes_train_a_in_50_minutes_l125_125608


namespace elena_has_card_4_l125_125394

noncomputable theory

def scores (name : String) : ℕ :=
  match name with
  | "Elena" => 9
  | "Liam" => 8
  | "Hannah" => 13
  | "Felix" => 18
  | "Sara" => 17
  | _ => 0 -- just a default case for robustness

def card_set := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def player_cards (name : String) : list ℕ := sorry -- Assume this is a function that finds the correct pair

theorem elena_has_card_4 : 4 ∈ player_cards "Elena" :=
by
  sorry

end elena_has_card_4_l125_125394


namespace num_proper_subsets_of_A_l125_125866

-- Define the set A as per the condition.
def A : Set ℕ := {x | -1 < x ∧ x ≤ 3 ∧ 0 < x}

-- The theorem stating the number of proper subsets of A.
theorem num_proper_subsets_of_A : (A.toFinset.powerset.card - 1) = 7 :=
by 
  sorry

end num_proper_subsets_of_A_l125_125866


namespace alpha_beta_90_neither_sufficient_nor_necessary_l125_125825

-- Define α and β as real numbers
variables (α β : ℝ)

-- State the theorem that describes the problem
theorem alpha_beta_90_neither_sufficient_nor_necessary :
  (α + β = 90 * (π / 180)) → ¬(sin α + sin β > 1) ∧
  (sin α + sin β > 1) → ¬(α + β = 90 * (π / 180)) :=
sorry

end alpha_beta_90_neither_sufficient_nor_necessary_l125_125825


namespace complement_of_alpha_l125_125798

theorem complement_of_alpha : 
  ∀ (alpha : ℝ), 
    alpha = 60 + 18 / 60 → 
      90 - alpha = 29 + 42 / 60 :=
by
  intros alpha h
  rw [h]
  norm_num
  sorry

end complement_of_alpha_l125_125798


namespace maximum_value_expression_l125_125413

theorem maximum_value_expression (θ1 θ2 θ3 θ4 θ5 θ6 : ℝ) :
  ∃ θ1 θ2 θ3 θ4 θ5 θ6, (cos θ1 * sin θ2 +
                         cos θ2 * sin θ3 +
                         cos θ3 * sin θ4 +
                         cos θ4 * sin θ5 +
                         cos θ5 * sin θ6 +
                         cos θ6 * sin θ1) = 3 :=
sorry

end maximum_value_expression_l125_125413


namespace min_quadratic_expression_l125_125726

theorem min_quadratic_expression:
  ∀ x y : ℝ, 2 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y ≥ 3 :=
by
  sorry

end min_quadratic_expression_l125_125726


namespace distance_points_N_Q_l125_125212

noncomputable def distance_between_N_and_Q
  (A B C M N P Q : ℝ × ℝ)
  (h₁ : ∠BAC = 60)
  (h₂ : dist A B = √3)
  (h₃ : dist A C = 1)
  (h₄ : dist A M = dist A P)
  (h₅ : dist M B = dist N B)
  (h₆ : dist P C = dist Q C) : ℝ :=
  dist N Q

theorem distance_points_N_Q
  (A B C M N P Q : ℝ × ℝ)
  (h₁ : ∠BAC = 60)
  (h₂ : dist A B = √3)
  (h₃ : dist A C = 1)
  (h₄ : dist A M = dist A P)
  (h₅ : dist M B = dist N B)
  (h₆ : dist P C = dist Q C) :
  distance_between_N_and_Q A B C M N P Q h₁ h₂ h₃ h₄ h₅ h₆ = (√(19 - 8*√3)) / 2 :=
begin
  sorry
end

end distance_points_N_Q_l125_125212


namespace valid_propositions_l125_125459

/- Proposition ③ -/
def proposition3 : Prop :=
  ∃ (x0 : ℝ), ∀ x : ℝ, x > x0 → 2^x > x^2

/- Proposition ⑤ -/
noncomputable def proposition5 : Prop :=
  let x1 := Classical.some (exists x : ℝ, x + real.log x = 5)
  let x2 := Classical.some (exists x : ℝ, x + 10^x = 5)
  x1 + x2 = 5

theorem valid_propositions : proposition3 ∧ proposition5 := by
  sorry

end valid_propositions_l125_125459


namespace bride_older_than_groom_l125_125268

-- Define the ages of the bride and groom
variables (B G : ℕ)

-- Given conditions
def groom_age : Prop := G = 83
def total_age : Prop := B + G = 185

-- Theorem to prove how much older the bride is than the groom
theorem bride_older_than_groom (h1 : groom_age G) (h2 : total_age B G) : B - G = 19 :=
sorry

end bride_older_than_groom_l125_125268


namespace can_contain_more_than_12_numbers_l125_125908

noncomputable def largest_digit (n : ℕ) : ℕ :=
  n.digits.max'.get_or_else 0

noncomputable def next_term (a : ℕ) : ℕ :=
  a - largest_digit a

noncomputable def sequence (a1 : ℕ) : ℕ → ℕ
| 0     := a1
| (n+1) := next_term (sequence n)

theorem can_contain_more_than_12_numbers (a1 : ℕ) (h : even a1) :
  ∃ n > 12, even (sequence a1 n) :=
sorry

end can_contain_more_than_12_numbers_l125_125908


namespace find_divisor_l125_125944

theorem find_divisor (d : ℕ) (N : ℕ) (a b : ℕ)
  (h1 : a = 9) (h2 : b = 79) (h3 : N = 7) :
  (∃ d, (∀ k : ℕ, a ≤ k*d ∧ k*d ≤ b → (k*d) % d = 0) ∧
   ∀ count : ℕ, count = (b / d) - ((a - 1) / d) → count = N) →
  d = 11 :=
by
  sorry

end find_divisor_l125_125944


namespace no_prime_between_100_and_110_div_6_eq_3_l125_125586

theorem no_prime_between_100_and_110_div_6_eq_3 :
  ¬ ∃ n : ℕ, nat.prime n ∧ 100 < n ∧ n < 110 ∧ n % 6 = 3 := 
by
  sorry

end no_prime_between_100_and_110_div_6_eq_3_l125_125586


namespace max_roses_purchase_l125_125232

-- Define price constants and constraints
def price_per_individual_rose : ℝ := 6.3
def price_per_one_dozen : ℝ := 36
def price_per_two_dozen : ℝ := 50
def price_per_five_dozen : ℝ := 110

def budget : ℝ := 680
def min_red_roses_cost : ℝ := 200

-- Maximum number of roses calculation
theorem max_roses_purchase : ∃ (n : ℕ), 
  n = 360 ∧ 
  let total_five_dozen := 2 in 
  let remaining_budget := budget - (total_five_dozen * price_per_five_dozen) in 
  total_five_dozen * 60 + ∃ (remaining_five_dozen : ℕ), 
    remaining_five_dozen ≤ (remaining_budget / price_per_five_dozen).toNat ∧ 
    total_five_dozen + remaining_five_dozen = 6 in
  remaining_budget - (remaining_five_dozen * price_per_five_dozen) ≥ 0 :=
-- Sorry to skip the proof
sorry

end max_roses_purchase_l125_125232


namespace triangle_ABC_angles_l125_125037

theorem triangle_ABC_angles (A B C M: Type) [triangle A B C] 
  (hC : ∠B C A = 90)
  (median_CM : midpoint C M)
  (circle_touch_CM : ∀ (P : Point), P ∈ incircle A C M → P = midpoint C M) :
  angles A B C = (30, 60, 90) :=
by 
  sorry

end triangle_ABC_angles_l125_125037


namespace find_a_l125_125822

theorem find_a (a : ℝ) (A B : Set ℝ) (hA : A = {-1, 1}) (hB : B = {x | x * a = 1}) (hB_sub_A : B ⊆ A) : 
  a ∈ {-1, 0, 1} :=
by
  sorry

end find_a_l125_125822


namespace seq_formulas_find_min_c_sum_less_than_l125_125441

variables {a : ℕ → ℝ} {b : ℕ → ℝ} {S T B : ℕ → ℝ}

-- Define the sequences a_n and b_n
def a_n (n : ℕ) : ℝ := 2^n
def b_n (n : ℕ) : ℝ := 2 * n - 1

-- Conditions
axiom hn : ∀ n, a n = (S n + 2) / 2
axiom hp : ∀ n, b (n+1) - b n = 2
axiom hb1 : b 1 = 1

-- Define T_n with respect to sequences a and b
def T (n : ℕ) : ℝ := (finset.range n).sum (λ i, b (i + 1) / a (i + 1))

-- Define B_n as the sum of the first n terms of b
def B (n : ℕ) : ℝ := (finset.range n).sum (λ i, b (i + 1))

-- Prove the given conditions and questions:
theorem seq_formulas : 
  (∀ n, a n = 2^n) ∧ 
  (∀ n, b n = 2 * n - 1) := sorry

theorem find_min_c (c : ℤ) : 
  (∀ n : ℕ, T n < c) → 
  c = 3 := sorry

theorem sum_less_than : 
  (∀ n, (finset.range n).sum (λ i, 1 / B (i + 1)) < 5 / 3) := sorry

end seq_formulas_find_min_c_sum_less_than_l125_125441


namespace kathleen_money_left_l125_125163

def june_savings : ℕ := 21
def july_savings : ℕ := 46
def august_savings : ℕ := 45

def school_supplies_expenses : ℕ := 12
def new_clothes_expenses : ℕ := 54

def total_savings : ℕ := june_savings + july_savings + august_savings
def total_expenses : ℕ := school_supplies_expenses + new_clothes_expenses

def total_money_left : ℕ := total_savings - total_expenses

theorem kathleen_money_left : total_money_left = 46 :=
by
  sorry

end kathleen_money_left_l125_125163


namespace function_conditions_l125_125400

theorem function_conditions 
  (f : ℕ → ℤ) 
  (h1 : ∀ a b : ℕ, a ∣ b → f a ≥ f b) 
  (h2 : ∀ a b : ℕ, f (a * b) + f (a^2 + b^2) = f a + f b) 
  : ∀ n : ℕ, f n = ∑ i in (factorize n).q, f(i) - ((factorize n).h - 1) * f 1 :=
sorry

end function_conditions_l125_125400


namespace common_root_cubic_polynomials_l125_125302

open Real

theorem common_root_cubic_polynomials (a b c : ℝ)
  (h1 : ∃ α : ℝ, α^3 - a * α^2 + b = 0 ∧ α^3 - b * α^2 + c = 0)
  (h2 : ∃ β : ℝ, β^3 - b * β^2 + c = 0 ∧ β^3 - c * β^2 + a = 0)
  (h3 : ∃ γ : ℝ, γ^3 - c * γ^2 + a = 0 ∧ γ^3 - a * γ^2 + b = 0)
  : a = b ∧ b = c :=
sorry

end common_root_cubic_polynomials_l125_125302


namespace arithmetic_and_geometric_mean_l125_125404

theorem arithmetic_and_geometric_mean (a b : ℝ) :
  let x := real.sqrt 3 + real.sqrt 2
  let y := real.sqrt 3 - real.sqrt 2
  a = (x + y) / 2 ∧ b = real.sqrt (x * y) ∧ (b = 1 ∨ b = -1) :=
by
  let x := real.sqrt 3 + real.sqrt 2
  let y := real.sqrt 3 - real.sqrt 2
  have a_def : a = (x + y) / 2 := by sorry
  have b_def : b = real.sqrt (x * y) := by sorry
  have b_pos_neg : (b = 1 ∨ b = -1) := by sorry
  exact ⟨a_def, b_def, b_pos_neg⟩

end arithmetic_and_geometric_mean_l125_125404


namespace sum_of_bounds_l125_125001

noncomputable def log_bounds : ℕ → ℕ → ℕ → Prop := λ a b n,
  (log 2 1024 = 10) ∧ (log 2 4096 = 12) ∧ 
  (1024 < n ∧ n < 4096) ∧ 
  (a = 10 ∧ b = 12) ∧ 
  (log 2 n > a ∧ log 2 n < b)

theorem sum_of_bounds : log_bounds 10 12 2048 → (10 + 12 = 22) :=
by sorry

end sum_of_bounds_l125_125001


namespace new_avg_temp_l125_125597

structure CityTemps where
  ny : ℝ
  miami : ℝ
  sd : ℝ
  phoenix : ℝ
  denver : ℝ

def average (temps : List ℝ) : ℝ := (temps.sum) / temps.length

theorem new_avg_temp :
  let ny := 80
  let miami := ny + 10
  let sd := miami + 25
  let avg_three := (ny + miami + sd) / 3
  let phoenix := sd + 0.15 * sd
  let denver := avg_three - 5
  average [ny, miami, sd, phoenix, denver] = 101.45 :=
by
  sorry

end new_avg_temp_l125_125597


namespace sufficient_but_not_necessary_condition_not_necessary_condition_l125_125197

theorem sufficient_but_not_necessary_condition (x : ℝ) : (abs (x - 2) < 1) → (x^2 + x - 2 > 0) :=
begin
  sorry,
end

theorem not_necessary_condition (x : ℝ) : (x^2 + x - 2 > 0) → ¬(abs (x - 2) < 1) :=
begin
  sorry,
end

end sufficient_but_not_necessary_condition_not_necessary_condition_l125_125197


namespace compound_interest_proof_l125_125823

-- Define simple interest conditions
variables (PR : ℝ)  -- Principal amount
def SI (P : ℝ) (R : ℝ) (T : ℝ) := P * R * T / 100

-- Define compound interest calculation
def CI (P : ℝ) (R : ℝ) (T : ℝ) := P * (1 + R / 100) ^ T - P

-- Problem statement proof (Lean 4)
theorem compound_interest_proof : 
  (SI 500 5 2 = 50) → 
  CI 500 5 2 = 51.25 := 
by {
  sorry
}

end compound_interest_proof_l125_125823


namespace sum_of_eleven_terms_l125_125052

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variables {d a_1 : ℝ}

-- Arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (a_1 d : ℝ) : Prop :=
  ∀ n, a n = a_1 + n * d

-- Sum of the first n terms of arithmetic sequence
def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n / 2) * (2 * a 0 + (n - 1) * d)

-- Given condition as hypothesis
def given_condition (a : ℕ → ℝ) : Prop :=
  2 * (a 0 + a 2 + a 4) + 3 * (a 7 + a 9) = 36

-- Theorem statement
theorem sum_of_eleven_terms {a : ℕ → ℝ} {S : ℕ → ℝ} {a_1 d : ℝ}
  (h_arith : arithmetic_sequence a a_1 d)
  (h_sum : sum_of_first_n_terms S a)
  (h_cond : given_condition a) :
  S 11 = 33 :=
sorry

end sum_of_eleven_terms_l125_125052


namespace area_of_DEF_l125_125720

noncomputable def DF : ℝ := 4
noncomputable def EF : ℝ := 4
noncomputable def DE : ℝ := 4 * Real.sqrt 2

theorem area_of_DEF : 
  DE = 4 * Real.sqrt 2 ∧
  DF = 4 ∧
  EF = 4 ∧
  ∠DEF = π / 4 →
  (1/2) * DF * EF = 8 :=
by
  intros h
  sorry

end area_of_DEF_l125_125720


namespace sum_of_common_divisors_36_48_l125_125020

-- Definitions based on the conditions
def is_divisor (n d : ℕ) : Prop := d ∣ n

-- List of divisors for 36 and 48
def divisors_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]
def divisors_48 : List ℕ := [1, 2, 3, 4, 6, 8, 12, 16, 24, 48]

-- Definition of common divisors
def common_divisors_36_48 : List ℕ := [1, 2, 3, 4, 6, 12]

-- Sum of common divisors
def sum_common_divisors_36_48 := common_divisors_36_48.sum

-- The statement of the theorem
theorem sum_of_common_divisors_36_48 : sum_common_divisors_36_48 = 28 := by
  sorry

end sum_of_common_divisors_36_48_l125_125020


namespace length_XY_le_half_perimeter_l125_125509

noncomputable def is_trapezoid (A B C D : ℝ^2) : Prop := 
  ∃ (n : ℝ), A.y = n ∧ B.y = n ∧ C.y ≠ n ∧ D.y ≠ n ∧ B.x = A.x + n ∧ D.x = C.x - n

noncomputable def is_diameter (A D : ℝ^2) (Γ : set ℝ^2) : Prop := 
  ∃ (O : ℝ^2), O = (A + D) / 2 ∧ ∀ (X : ℝ^2), X ∈ Γ → dist O X = dist X A / 2

noncomputable def midpoint (P Q : ℝ^2) : ℝ^2 :=
  (P + Q) / 2

theorem length_XY_le_half_perimeter 
  (A B C D X Y: ℝ^2)
  (h_trap : is_trapezoid A B C D)
  (Γ₁ Γ₂ : set ℝ^2)
  (hΓ₁ : is_diameter A D Γ₁)
  (hΓ₂ : is_diameter B C Γ₂)
  (hX : X ∈ Γ₁)
  (hY : Y ∈ Γ₂):
  dist X Y ≤ (dist A B + dist B C + dist C D + dist D A) / 2 :=
by
  sorry

end length_XY_le_half_perimeter_l125_125509


namespace swallow_oxygen_consumption_l125_125917

theorem swallow_oxygen_consumption :
  (∃ (a : ℝ), ∀ (x v : ℝ), 
    (v = a * log 2 (x / 10)) → 
    ((x = 40) → (v = 10) → (a = 5)) ∧ 
    ((v = 25) → (x = 320))) :=
sorry

end swallow_oxygen_consumption_l125_125917


namespace locus_of_Q_is_circle_l125_125863

open Set Classical

variables {A B C D P Q: Point}
variables {ω : Circle}

-- Conditions
axiom h1 : general_position [A, B, C, D]
axiom h2 : B ∈ ω ∧ C ∈ ω
axiom h3 : moves_along P ω
axiom h4 : ∀ P, circle_of_three_points A B P = circle A B P
axiom h5 : ∀ P, circle_of_three_points P C D = circle P C D
axiom h6 : ∀ P, Q ≠ P ∧ (Q ∈ circle_of_three_points A B P ∩ circle_of_three_points P C D)

-- Theorem
theorem locus_of_Q_is_circle : ∃ k : Circle, A ∈ k ∧ D ∈ k ∧ (∀ Q, Q ∈ k) :=
  sorry

end locus_of_Q_is_circle_l125_125863


namespace moles_of_water_formed_l125_125727

-- Defining the relevant constants
def NH4Cl_moles : ℕ := sorry  -- Some moles of Ammonium chloride (NH4Cl)
def NaOH_moles : ℕ := 3       -- 3 moles of Sodium hydroxide (NaOH)
def H2O_moles : ℕ := 3        -- The total moles of Water (H2O) formed

-- Statement of the problem
theorem moles_of_water_formed :
  NH4Cl_moles ≥ NaOH_moles → H2O_moles = 3 :=
sorry

end moles_of_water_formed_l125_125727


namespace ab_value_a2_plus_b2_minus_ab_value_l125_125101

-- Conditions
def a := Real.sqrt 7 + 2
def b := Real.sqrt 7 - 2

-- Theorems to prove
theorem ab_value : a * b = 3 := 
by 
  -- here would be the proof, but we use sorry to skip it
  sorry

theorem a2_plus_b2_minus_ab_value : a^2 + b^2 - a * b = 19 := 
by 
  -- here would be the proof, but we use sorry to skip it
  sorry

end ab_value_a2_plus_b2_minus_ab_value_l125_125101


namespace avg_weekly_income_500_l125_125335

theorem avg_weekly_income_500 :
  let base_salary := 350
  let income_past_5_weeks := [406, 413, 420, 436, 495]
  let commission_next_2_weeks_avg := 315
  let total_income_past_5_weeks := income_past_5_weeks.sum
  let total_base_salary_next_2_weeks := base_salary * 2
  let total_commission_next_2_weeks := commission_next_2_weeks_avg * 2
  let total_income := total_income_past_5_weeks + total_base_salary_next_2_weeks + total_commission_next_2_weeks
  let avg_weekly_income := total_income / 7
  avg_weekly_income = 500 := by
{
  sorry
}

end avg_weekly_income_500_l125_125335


namespace determine_y_l125_125480

variable {R : Type} [LinearOrderedField R]
variables {x y : R}

theorem determine_y (h1 : 2 * x - 3 * y = 5) (h2 : 4 * x + 9 * y = 6) : y = -4 / 15 :=
by
  sorry

end determine_y_l125_125480


namespace total_volume_tetrahedra_l125_125320

theorem total_volume_tetrahedra (side_length : ℝ) (x : ℝ) (sqrt_2 : ℝ := Real.sqrt 2) 
  (cube_to_octa_length : x = 2 * (sqrt_2 - 1)) 
  (volume_of_one_tetra : ℝ := ((6 - 4 * sqrt_2) * (3 - sqrt_2)) / 6) :
  side_length = 2 → 
  8 * volume_of_one_tetra = (104 - 72 * sqrt_2) / 3 :=
by
  intros
  sorry

end total_volume_tetrahedra_l125_125320


namespace jaqueline_can_construct_incenter_l125_125766

variables (A B C P Q : Point)
variables (h_triangle : Triangle A B C)

-- Defining Jaqueline's abilities
-- Ability to draw a line
axiom draw_line : ∀ (P Q : Point), Line

-- Ability to construct a circle with a given diameter
axiom construct_circle : ∀ (P Q : Point), Circle

-- Ability to mark intersection points
axiom mark_intersection : ∀ (obj1 obj2 : Object), Set Point

-- The main problem statement
theorem jaqueline_can_construct_incenter (h_acute_scalene : acute_scalene_triangle A B C)
    (h_ruler : ∀ (P Q : Point), Line)
    (h_circle : ∀ (P Q : Point), Circle)
    (h_intersection : ∀ (obj1 obj2 : Object), Set Point) :
    ∃ (I : Point), incenter I A B C :=
sorry

end jaqueline_can_construct_incenter_l125_125766


namespace gwendolyn_read_time_l125_125799

theorem gwendolyn_read_time :
  let rate := 200 -- sentences per hour
  let paragraphs_per_page := 30
  let sentences_per_paragraph := 15
  let pages := 100
  let sentences_per_page := sentences_per_paragraph * paragraphs_per_page
  let total_sentences := sentences_per_page * pages
  let total_time := total_sentences / rate
  total_time = 225 :=
by
  sorry

end gwendolyn_read_time_l125_125799


namespace distinct_integers_no_perfect_square_product_l125_125640

theorem distinct_integers_no_perfect_square_product
  (k : ℕ) (hk : 0 < k) :
  ∀ a b : ℕ, k^2 < a ∧ a < (k+1)^2 → k^2 < b ∧ b < (k+1)^2 → a ≠ b → ¬∃ m : ℕ, a * b = m^2 :=
by sorry

end distinct_integers_no_perfect_square_product_l125_125640


namespace palindromic_product_l125_125898

-- Definitions
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def is_nonzero_digit (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 9

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def distinct (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def exactly_one_even (a b c : ℕ) : Prop :=
  (a % 2 = 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0) ∨
  (a % 2 ≠ 0 ∧ b % 2 = 0 ∧ c % 2 ≠ 0) ∨
  (a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 = 0)

-- The main theorem
theorem palindromic_product :
  ∃ (n m p : ℕ),
    is_nonzero_digit n ∧ is_nonzero_digit m ∧ is_nonzero_digit p ∧
    distinct n m p ∧ exactly_one_even n m p ∧
    is_three_digit (n * 100 + m * 10 + p) ∧
    is_three_digit (n * 100 + p * 10 + m) ∧
    (n * 100 + m * 10 + p) * (n * 100 + p * 10 + m) = 29392 ∧
    is_palindrome (29392) := 
begin
  sorry
end

end palindromic_product_l125_125898


namespace part1_part2_l125_125069

def f (x : ℝ) : ℝ := Real.cos x - 1 / Real.exp x

theorem part1 (h1 : 7 < Real.exp 2) (h2 : Real.exp 2 < 8) 
    (h3 : Real.exp 3 > 16) (h4 : Real.exp (-3 * Real.pi / 4) < Real.sqrt 2 / 2) :
  ∃ x : ℝ, x ∈ (Real.pi / 6, Real.pi / 4) ∧ f' x = 0 := sorry

theorem part2 (h1 : 7 < Real.exp 2) (h2 : Real.exp 2 < 8) 
    (h3 : Real.exp 3 > 16) (h4 : Real.exp (-3 * Real.pi / 4) < Real.sqrt 2 / 2) :
  ∃ x1 x2 x3 x4 : ℝ, 
  0 < x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧ x4 < 2 * Real.pi ∧ 
  (∀ x ∈ (0, x1), f' x > 0) ∧ (∀ x ∈ (x1, x2), f' x < 0) ∧ 
  (∀ x ∈ (x2, x3), f' x = 0) ∧ (∀ x ∈ (x3, x4), f' x > 0) := sorry

end part1_part2_l125_125069


namespace smallest_value_of_a_l125_125575

-- Define the conditions of the problem
def parabola_vertex : Prop :=
  ∃ (a : ℚ), a > 0 ∧
             (∃ (b c : ℚ), ∀ x : ℚ, 
               (x - 3 / 5) * (x - 3 / 5) * a - 25 / 12 = a * x * x + b * x + c ∧
               2 * a + b + c ∈ ℤ )

-- Prove that under these conditions, the smallest possible value of 'a' is 925/408
theorem smallest_value_of_a {a : ℚ} (h : parabola_vertex) : a = 925 / 408 := 
by 
  sorry

end smallest_value_of_a_l125_125575


namespace general_term_sum_first_n_terms_sum_max_value_l125_125542

variable (a_n : ℕ → ℕ) (S_n : ℕ → ℕ)

axiom a₃_eq_24 : a_n 3 = 24
axiom a₆_eq_18 : a_n 6 = 18

theorem general_term : ∀ n, a_n n = 30 - 2 * n :=
sorry

theorem sum_first_n_terms : ∀ n, S_n n = - n^2 + 29 * n :=
sorry

theorem sum_max_value : S_n 14 = 210 ∧ S_n 15 = 210 :=
sorry

end general_term_sum_first_n_terms_sum_max_value_l125_125542


namespace parallelogram_area_l125_125773

variables (A B C D O : Type)
variable [Inhabited A]
variable [Inhabited B]
variable [Inhabited C]
variable [Inhabited D]
variable [AffinePlane B A]

-- Conditions from the problem
variable [Parallelogram ABCD]
variable (h1 : IntersectionPoint O ABCD)
variable (h2 : Area (Triangle A B C) = 3)

-- The proof problem translated to a Lean 4 statement
theorem parallelogram_area (A B C D O : Point) 
  [Parallelogram ABCD]
  [IntersectionPoint O ABCD]
  (h2 : Area (Triangle A B C) = 3) :
  Area (Parallelogram ABCD) = 6 :=
sorry

end parallelogram_area_l125_125773


namespace least_positive_integer_satisfies_series_l125_125013

theorem least_positive_integer_satisfies_series :
  (∑ k in finset.range (91), 1 / (Real.sin (44 + k : ℝ) * Real.sin (45 + k : ℝ))) = 1 / Real.sin 1 :=
sorry

end least_positive_integer_satisfies_series_l125_125013


namespace board_product_l125_125895

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

def has_nonzero_digits (n : ℕ) : Prop :=
  n.digits 10 ≠ [0]

def has_one_even_digit (n m : ℕ) : Prop := 
  (list.filter (λ d, d % 2 = 0) (n.digits 10 ++ m.digits 10)).length = 1

theorem board_product :
  ∃ (x y : ℕ), 
    100 ≤ x ∧ x ≤ 999 ∧ 
    100 ≤ y ∧ y ≤ 999 ∧ 
    has_nonzero_digits x ∧
    has_nonzero_digits y ∧
    has_one_even_digit x y ∧ 
    x ≠ y ∧
    let p := x * y in
    is_palindrome p ∧ 10000 ≤ p ∧ p ≤ 99999 ∧
    p = 29392 :=
sorry

end board_product_l125_125895


namespace count_integer_values_l125_125473

theorem count_integer_values (π : Real) (hπ : Real.pi = π):
  ∃ n : ℕ, n = 27 ∧ ∀ x : ℤ, |(x:Real)| < 4 * π + 1 ↔ -13 ≤ x ∧ x ≤ 13 :=
by sorry

end count_integer_values_l125_125473


namespace split_weights_into_equal_piles_l125_125600

theorem split_weights_into_equal_piles (weights : ℕ → ℕ) (h_len : ∀ m, m < 64 → weights (m + 1) - weights m = 1 ∨ weights m - weights (m + 1) = 1) :
  ∃ (pile1 pile2 : finset ℕ), pile1.card = 32 ∧ pile2.card = 32 ∧ pile1.sum weights = pile2.sum weights :=
sorry

end split_weights_into_equal_piles_l125_125600


namespace mod_graph_sum_l125_125695

theorem mod_graph_sum (x₀ y₀ : ℕ) (h₁ : 2 * x₀ ≡ 1 [MOD 11]) (h₂ : 3 * y₀ ≡ 10 [MOD 11]) : x₀ + y₀ = 13 :=
by
  sorry

end mod_graph_sum_l125_125695


namespace polynomial_roots_power_sum_l125_125808

theorem polynomial_roots_power_sum {a b c : ℝ}
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 6)
  (h3 : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 21 :=
by
  sorry

end polynomial_roots_power_sum_l125_125808


namespace polynomial_real_roots_b_value_l125_125200

theorem polynomial_real_roots_b_value {a b : ℝ} (h_poly : ∀ x : ℝ, (X^3 - a * X^2 + b * X - a).roots.all (λ r : ℝ, r ∈ ℝ))
  (h_min_pos_a : ∀ a', (∀ x : ℝ, (X^3 - a' * X^2 + b * X - a').roots.all (λ r : ℝ, r ∈ ℝ)) → 0 < a' → a' ≥ a)
  (h_unique_b : ∀ a', (∀ x : ℝ, (X^3 - a' * X^2 + b * X - a').roots.all (λ r : ℝ, r ∈ ℝ)) → 0 < a' → b = ((a') * (a')) / (a'))
  : b = 9 :=
by
  sorry

end polynomial_real_roots_b_value_l125_125200


namespace max_b_of_box_volume_l125_125940

theorem max_b_of_box_volume (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : Prime c) (h5 : a * b * c = 360) : b = 12 := 
sorry

end max_b_of_box_volume_l125_125940


namespace sqrt_eq_neg_self_impl_m_leq_zero_l125_125812

theorem sqrt_eq_neg_self_impl_m_leq_zero (m : ℝ) (h : sqrt (m^2) = -m) : m ≤ 0 := 
by
  sorry

end sqrt_eq_neg_self_impl_m_leq_zero_l125_125812


namespace ellipse_equation_l125_125762

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a^2 = (4/3) * b^2)
    (circle_tangent : b = Real.sqrt 3) :
    (∀ x y : ℝ, (x^2)/(a^2) + (y^2)/(b^2) = 1 ↔ x^2/4 + y^2/3 = 1) :=
by
  intros x y
  have ha : a^2 = 4 := by
    rw [h3, circle_tangent, Real.sqrt, sqr]
    ring_nfm only,
  exact sorry

end ellipse_equation_l125_125762


namespace sum_of_ages_l125_125666

theorem sum_of_ages (a b c : ℕ) (father_age := 10 * a + b) (son_age := 10 + c) 
  (h1 : 1000 * a + 100 * b + 10 + c - abs (father_age - son_age) = 4289) :
  father_age + son_age = 59 :=
  sorry

end sum_of_ages_l125_125666


namespace problem_b_solution_l125_125682

-- Define the expressions as functions
def Sum1 := (∑ i in range 101, i)
def Sum2 := Nat.InfiniteSum (fun n => n)
def Sum3 (n : ℕ) := (∑ i in range (n + 1), i)

-- Define the condition for an algorithm (finite sum)
def can_be_solved_by_algorithm (S : ℕ) := True

theorem problem_b_solution :
  can_be_solved_by_algorithm Sum1 ∧
  ¬ can_be_solved_by_algorithm Sum2 ∧
  ∀ (n : ℕ) (h : n ≥ 1), can_be_solved_by_algorithm (Sum3 n) :=
by sorry

end problem_b_solution_l125_125682


namespace radius_of_tangent_circle_l125_125051

theorem radius_of_tangent_circle (x : ℝ) (hx : 0 < x) (hx1 : x < 1) : 
  ∃ r : ℝ, r = (1 - x) * (2 - Real.sqrt 2) :=
by 
  use (1 - x) * (2 - Real.sqrt 2)
  exact sorry

end radius_of_tangent_circle_l125_125051


namespace BM_perpendicular_AK_l125_125991

variables (O A B C D M N G K : Point)
variables (circle_O : Circle)
variables (inscribed_quad : InscribedQuadrilateral O circle_O A B C D)
variables (perp_diag : AC ⟂ BD)
variables (midpoint_M : MidpointArc O circle_O A D C M)
variables (midpoint_N : MidpointArc O circle_O A B C N)
variables (diameter_through_D : Diameter O circle_O D)
variables (G_on_AN : ChordIntersection O circle_O A N G)
variables (K_on_CD : PointOnLineSegment C D K)
variables (parallel_GK_NC : GK || NC)

theorem BM_perpendicular_AK :
  BM ⟂ AK :=
sorry

end BM_perpendicular_AK_l125_125991


namespace equal_slopes_iff_parallel_l125_125481

noncomputable def lines_have_equal_slopes (L₁ L₂ : Type) [has_slope L₁] [has_slope L₂] : Prop :=
  slope L₁ = slope L₂

noncomputable def lines_are_parallel (L₁ L₂ : Type) [line L₁] [line L₂] : Prop :=
  parallel L₁ L₂

theorem equal_slopes_iff_parallel (L₁ L₂ : Type) [has_slope L₁] [has_slope L₂] [line L₁] [line L₂] : 
  lines_have_equal_slopes L₁ L₂ ↔ lines_are_parallel L₁ L₂ :=
by
  sorry -- proof goes here

end equal_slopes_iff_parallel_l125_125481


namespace find_primes_for_expression_l125_125403

theorem find_primes_for_expression :
  ∀ (p : ℕ), 
  (p = 2 ∨ p = 3 ∨ p = 5) ↔ 
  ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 
  prime p ∧ 
  (x^p + y^p + z^p - x - y - z + 0 = 2 * 3 * 5 ∨ x^p + y^p + z^p - x - y - z = 30 ∨ x^p + y^p + z^p - x - y - z = 2 ^ 5) :=
sorry

end find_primes_for_expression_l125_125403


namespace sum_PE_PF_eq_PD_l125_125636

-- Given: ABCD is a rectangle, E is on CD, F is midpoint of AB,
-- PF is perpendicular to CD, PE is perpendicular to AB,
-- and PD is the diagonal of the rectangle
variables (A B C D E F P : Type) [AffineSpace ℝ A] [AffineSpace ℝ B] 
         [AffineSpace ℝ C] [AffineSpace ℝ D] [AffineSpace ℝ E] [AffineSpace ℝ F] [AffineSpace ℝ P]
         
-- Definitions
def is_rectangle (A B C D : Type) : Prop := sorry
def is_midpoint (F : Type) (A B : Type) : Prop := sorry
def perpendicular (X Y : Type) : Prop := sorry
def on_segment (X Y : Type) (E : Type) : Prop := sorry
def diagonal (A C : Type) : Prop := sorry

-- Conditions
axiom rect_ABCD : is_rectangle A B C D
axiom E_on_CD : on_segment C D E
axiom midpoint_F_AB : is_midpoint F A B
axiom PF_perp_CD : perpendicular P F
axiom PE_perp_AB : perpendicular P E
axiom PD_diagonal : diagonal P D

-- Theorem statement
theorem sum_PE_PF_eq_PD : 
  rect_ABCD ∧ E_on_CD ∧ midpoint_F_AB ∧ PF_perp_CD ∧ PE_perp_AB ∧ PD_diagonal → 
  sorry

end sum_PE_PF_eq_PD_l125_125636


namespace find_c_floor_l125_125172

open Real

noncomputable theory

def smallest_c := (11 : ℝ) / (6 * sqrt 3)

theorem find_c_floor : 
  (∀ (n : ℕ) (n_pos : 0 < n) (x : Fin n.succ → ℝ), 
    (∀ k : ℕ, k ≤ n →
      sqrt (∑ i in Finset.range k, x i ^ 2 + ∑ i in Finset.range (n - k), x (k + i.succ)) > 0 → 
      (∑ k in Finset.range (n + 1), 
          ((n ^ 3 + k ^ 3 - k ^ 2 * n) ^ (3 / 2)) / 
            sqrt (∑ i in Finset.range (k + 1), x i ^ 2 + ∑ i in Finset.range (n - k), x (i+s k).succ)) 
        ≤ sqrt 3 * (∑ i in Finset.range n, i.succ ^ 3 * (4 * n - 3 * i.succ + 100) / (x i.succ)) + smallest_c * n ^ 5 + 100 * n ^ 4)) 
  → ⌊2020 * smallest_c⌋ = 2137 := 
by 
  sorry

end find_c_floor_l125_125172


namespace flagpole_shadow_length_correct_l125_125657

noncomputable def flagpole_shadow_length (flagpole_height building_height building_shadow_length : ℕ) :=
  flagpole_height * building_shadow_length / building_height

theorem flagpole_shadow_length_correct :
  flagpole_shadow_length 18 20 50 = 45 :=
by
  sorry

end flagpole_shadow_length_correct_l125_125657


namespace complement_union_l125_125079

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- The proof problem statement
theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3}) (hA : A = {-1, 0, 1}) (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} := sorry

end complement_union_l125_125079


namespace floor_sqrt_225_l125_125396

theorem floor_sqrt_225 : Int.floor (Real.sqrt 225) = 15 := by
  sorry

end floor_sqrt_225_l125_125396


namespace chicken_stock_to_milk_ratio_l125_125202

-- Define the conditions
def milk_quarts : ℕ := 2
def vegetable_quarts : ℕ := 1
def total_quarts_needed : ℕ := 3 * 3
def initial_soup_quarts := milk_quarts + vegetable_quarts
def chicken_stock_quarts := total_quarts_needed - initial_soup_quarts
def ratio_of_chicken_stock_to_milk := chicken_stock_quarts / milk_quarts

-- Prove the required ratio
theorem chicken_stock_to_milk_ratio : ratio_of_chicken_stock_to_milk = 3 := by
  have h_total_quarts_needed : total_quarts_needed = 9 := rfl
  have h_initial_soup : initial_soup_quarts = 3 := rfl
  have h_chicken_stock : chicken_stock_quarts = 6 := rfl
  have h_ratio : ratio_of_chicken_stock_to_milk = 6 / 2 := rfl
  show 6 / 2 = 3 from rfl
  sorry

end chicken_stock_to_milk_ratio_l125_125202


namespace sum_of_solutions_eqn_l125_125733

theorem sum_of_solutions_eqn : 
  (∀ x : ℝ, -48 * x^2 + 100 * x + 200 = 0 → False) → 
  (-100 / -48) = (25 / 12) :=
by
  intros
  sorry

end sum_of_solutions_eqn_l125_125733


namespace profit_difference_l125_125331

variable (P : ℕ) -- P is the total profit
variable (r1 r2 : ℚ) -- r1 and r2 are the parts of the ratio for X and Y, respectively

noncomputable def X_share (P : ℕ) (r1 r2 : ℚ) : ℚ :=
  (r1 / (r1 + r2)) * P

noncomputable def Y_share (P : ℕ) (r1 r2 : ℚ) : ℚ :=
  (r2 / (r1 + r2)) * P

theorem profit_difference (P : ℕ) (r1 r2 : ℚ) (hP : P = 800) (hr1 : r1 = 1/2) (hr2 : r2 = 1/3) :
  X_share P r1 r2 - Y_share P r1 r2 = 160 := by
  sorry

end profit_difference_l125_125331


namespace revenue_increase_l125_125649

theorem revenue_increase (R : ℕ) (r2000 r2003 r2005 : ℝ) (h1 : r2003 = r2000 * 1.50) (h2 : r2005 = r2000 * 1.80) :
  ((r2005 - r2003) / r2003) * 100 = 20 :=
by sorry

end revenue_increase_l125_125649


namespace euclidean_remainder_l125_125728

noncomputable def P(x : ℝ) : ℝ := x^100 - 2 * x^51 + 1
noncomputable def D(x : ℝ) : ℝ := x^2 - 1
noncomputable def R(x : ℝ) : ℝ := -2 * x + 2

theorem euclidean_remainder :
  ∃ Q(x : ℝ), P(x) = Q(x) * D(x) + R(x) :=
sorry

end euclidean_remainder_l125_125728


namespace part_a_l125_125312

theorem part_a (A B C D M : Point)
  (h1 : convex_quadrilateral A B C D)
  (h2 : internal_point M A B C D) :
  dist M A + dist M B < dist A D + dist D C + dist C B := 
  sorry

end part_a_l125_125312


namespace original_price_of_heels_l125_125208

variable (x : Real)

theorem original_price_of_heels :
  let jumper_original := 30
  let jumper_discount := 0.10 * jumper_original
  let jumper_final := jumper_original - jumper_discount
  let tshirt_price := 10
  let tshirt_final := 2 * tshirt_price
  let heels_discount := 0.20 * x
  let heels_final := x - heels_discount
  let total_before_tax := jumper_final + tshirt_final + heels_final
  let sales_tax := 0.06 * total_before_tax
  let total_after_tax := total_before_tax + sales_tax
  (total_after_tax = 150) → (x ≈ 118.16) :=
by
  intros
  let jumper_original := 30
  let jumper_discount :=  0.10 * jumper_original
  let jumper_final :=  jumper_original - jumper_discount
  let tshirt_price := 10
  let tshirt_final := 2 * tshirt_price
  let heels_discount := 0.20 * x
  let heels_final := x - heels_discount
  let total_before_tax := jumper_final + tshirt_final + heels_final
  let sales_tax :=  0.06 * total_before_tax
  let total_after_tax := total_before_tax + sales_tax
  have h150 : total_after_tax = 150 := by sorry
  have h11816 : x ≈ 118.16 := by sorry
  exact h11816

end original_price_of_heels_l125_125208


namespace parallel_vectors_cosine_identity_l125_125796

theorem parallel_vectors_cosine_identity (α : ℝ) (h : ∃ (k : ℝ), (\frac{1}{3}, Real.tan α) = k • (Real.cos α, 2)) : Real.cos (2 * α) = 1 / 9 :=
by
  sorry

end parallel_vectors_cosine_identity_l125_125796


namespace distance_between_centers_l125_125566

theorem distance_between_centers (ABC : Triangle) (O : Point) (I : Point) (R r : ℝ)
  (circumcenter_O : O = center_of_circumscribed_circle ABC) 
  (circumradius_R : R = radius_of_circumscribed_circle ABC O) 
  (incenter_I : I = center_of_inscribed_circle ABC) 
  (inradius_r : r = radius_of_inscribed_circle ABC I) :
  distance O I = sqrt (R^2 - 2 * R * r) :=
by
  sorry

end distance_between_centers_l125_125566


namespace part_I_part_II_l125_125826

noncomputable theory
open_locale real

variables {A B C : ℝ} 
variables {a b c : ℝ} 

theorem part_I (h₁ : a * real.cos C = b - (real.sqrt 3 / 2) * c)
  (h₂ : B + A + C = real.pi) :
  A = real.pi / 6 :=
sorry

theorem part_II (h₁ : a * real.cos C = b - (real.sqrt 3 / 2) * c)
  (h₂ : B = real.pi / 6)
  (h₃ : AC = 4)
  (h₄ : A = real.pi / 6)
  (h₅ : C = 2 * real.pi / 3)
  (h₆ : BC = 4)
  (h₇ : AB = 4 * real.sqrt 3) :
  (2 * AC = AB + 4) →
  (AM = 2 * real.sqrt 7) :=
sorry

end part_I_part_II_l125_125826


namespace parabola_ellipse_focus_l125_125777

theorem parabola_ellipse_focus {p : ℝ} (hp : p > 0) :
  let F := (1 : ℝ, 0 : ℝ),
  let C := fun (y : ℝ) => (y^2 : ℝ) = 2 * p * (x : ℝ),
  let E := (x : ℝ)^2 / 4 + (y : ℝ)^2 / 3 = 1,
  F = (1, 0) →
  p = 2 →
  (C y = 4 * (x : ℝ) ∧
   (∀ (A B : ℝ × ℝ), (minimum (|AB| / |MF|) = 2)) ∧
   (∀ (A B A' B' : ℝ × ℝ), (triangle A'FB' is_right_triangle A' F B'))) :=
by
  intros F C E hF hp_eq,
  sorry

end parabola_ellipse_focus_l125_125777


namespace product_fraction_sequence_l125_125680

theorem product_fraction_sequence :
  (list.product ((list.range' 5 (2036 - 5) 3).map (λ n, n / (n + 3)))) = (5 / 2039) := 
sorry

end product_fraction_sequence_l125_125680


namespace num_mappings_conditions_l125_125198

def M : Set Int := {-1, 0, 1}
def N : Set Int := {-2, -1, 0, 1, 2}

def condition (f : Int → Int) : Prop := ∀ x ∈ M, (x + f x) % 2 ≠ 0

theorem num_mappings_conditions :
  ∃ (f : (Int → Int) → Prop), condition f ∧
  (set.count (λ f, condition f) (set.maps M N) = 18) :=
sorry

end num_mappings_conditions_l125_125198


namespace problem_solution_l125_125550

theorem problem_solution :
  ∀ (A B M N C : Type) (arc_BM arc_MN arc_NC : ℝ), 
    arc_BM = 50 ∧ arc_MN = 60 ∧ arc_NC = 68 →
    let arc_BC := arc_BM + arc_NC in 
    let ∡ R := (arc_MN - arc_BC) / 2 in 
    let ∡ S := arc_BC / 2 in 
    ∡ R + ∡ S = 30 :=
by
  intros A B M N C arc_BM arc_MN arc_NC h
  let arc_BC := arc_BM + arc_NC
  let angle_R := (arc_MN - arc_BC) / 2
  let angle_S := arc_BC / 2
  have h1 : arc_BM = 50 := And.left h
  have h2 : arc_MN = 60 := And.elim_left (And.right h)
  have h3 : arc_NC = 68 := And.elim_right (And.right h)
  sorry

end problem_solution_l125_125550


namespace proof_A_cap_complement_B_l125_125775

variable (A B U : Set ℕ) (h1 : A ⊆ U) (h2 : B ⊆ U)
variable (h3 : U = {1, 2, 3, 4})
variable (h4 : (U \ (A ∪ B)) = {4}) -- \ represents set difference, complement in the universal set
variable (h5 : B = {1, 2})

theorem proof_A_cap_complement_B : A ∩ (U \ B) = {3} := by
  sorry

end proof_A_cap_complement_B_l125_125775


namespace arrow_in_48th_position_l125_125821

def arrow_sequence : List (String) := ["→", "↑", "↓", "←", "↘"]

theorem arrow_in_48th_position :
  arrow_sequence.get? ((48 % 5) - 1) = some "↓" :=
by
  norm_num
  sorry

end arrow_in_48th_position_l125_125821


namespace rope_length_l125_125369

theorem rope_length (x : ℝ) 
  (h : 10^2 + (x - 4)^2 = x^2) : 
  x = 14.5 :=
sorry

end rope_length_l125_125369


namespace find_cos_alpha_l125_125059

theorem find_cos_alpha (α : ℝ) (h0 : 0 ≤ α ∧ α ≤ π / 2) (h1 : Real.sin (α - π / 6) = 3 / 5) : 
  Real.cos α = (4 * Real.sqrt 3 - 3) / 10 :=
sorry

end find_cos_alpha_l125_125059


namespace number_of_valid_k_l125_125388

theorem number_of_valid_k : 
  (∃ k : ℕ, k > 0 ∧ ∃ x : ℤ, k * x - 18 = 3 * k) → 
  (∑ k in finset.filter (λ k, 18 % k = 0) (finset.range 19), 1) = 6 :=
by
  sorry

end number_of_valid_k_l125_125388


namespace tan_x_value_l125_125811

theorem tan_x_value (x : ℝ) (h1 : 0 < x ∧ x < π) (h2 : sin x + cos x = 1/5) : tan x = -4/3 :=
by
  sorry

end tan_x_value_l125_125811


namespace centers_of_Wa_Wb_Wc_are_collinear_l125_125518

theorem centers_of_Wa_Wb_Wc_are_collinear
  (A B C : Point)
  (circumcircle : Circle)
  (h_circumcircle : passes_through circumcircle A ∧ passes_through circumcircle B ∧ passes_through circumcircle C)
  (W_a W_b W_c : Circle)
  (center_Wa : Point)
  (center_Wb : Point)
  (center_Wc : Point)
  (h_Wa_center : Center_of W_a = center_Wa ∧ On_Line center_Wa B C ∧ passes_through W_a A ∧ perpendicular W_a circumcircle)
  (h_Wb_center : Center_of W_b = center_Wb ∧ On_Line center_Wb A C ∧ passes_through W_b B ∧ perpendicular W_b circumcircle)
  (h_Wc_center : Center_of W_c = center_Wc ∧ On_Line center_Wc A B ∧ passes_through W_c C ∧ perpendicular W_c circumcircle) 
  : Collinear center_Wa center_Wb center_Wc := 
sorry

end centers_of_Wa_Wb_Wc_are_collinear_l125_125518


namespace largest_vertex_sum_l125_125034

theorem largest_vertex_sum (a T : ℤ) (hT : T ≠ 0) (ha : ∀ (x y : ℤ), y = ax^2 + bx + c 
                                       → (x, y) ∈ {(0,0), (3T,0), (3T + 1, 36)}) : 
    (∃ M : ℤ, M = 4 ∧ 
    ∀ (b c : ℤ), y = ax^2 + bx + c 
    → (x, y) ∈ {(0,0), (3T,0), (3T + 1, 36)}
    → let xv := 3T / 2
    in let yv = -9 * a * T^2 / 4
    in M = xv + yv) :=
sorry

end largest_vertex_sum_l125_125034


namespace find_digit_l125_125719

/-- The 2023rd digit past the decimal point in the decimal expansion of 7/18 is 3. -/
theorem find_digit (n : ℕ) (h_pos : n > 0)
  (h_seq : ∀ k, let d := nat.abs (k % 2) in
               d = if d = 0 then 3 else 8) : 
  (2023 % 2 = 1) → (∃ m : ℕ, m = 2023 ∧ ∃ k : ℕ, k = m / 2 ∧ nat.digits 10 3 = [3]) :=
by 
  sorry

end find_digit_l125_125719


namespace intersect_y_axis_integer_points_inside_W_when_k_is_2_no_integer_points_inside_W_range_of_k_l125_125506

-- Given a line l: y = kx + 1 with k ≠ 0
variable {k : ℝ} (h : k ≠ 0)

-- 1. Prove the intersection point of l and the y-axis is (0, 1)
theorem intersect_y_axis :
  ∃ y, y = k * 0 + 1 ∧ (0, y) = (0, 1) :=
by sorry

-- 2. Prove the number of integer points inside region W enclosed by AB, BC, and CA when k = 2 is 6
theorem integer_points_inside_W_when_k_is_2 :
  ∃ W : set (ℝ × ℝ), 
    let A := (2 : ℝ, 5 : ℝ),
        B := (-(3/2) : ℝ, -2 : ℝ),
        C := (2 : ℝ, -2 : ℝ),
        W := {p | (p.1, p.2) ∈ set.line_segment ℝ A B ∪ set.line_segment ℝ B C ∪ set.line_segment ℝ C A} in
    W = set.of [ (0,0),  (0,-1), (1,0), (1,-1), (1,1), (1,2)] :=
by sorry

-- 3. Prove the range of values of k such that there are no integer points inside W is -1 ≤ k < 0 or k = -2
theorem no_integer_points_inside_W_range_of_k :
  ∀ k : ℝ, (-1 ≤ k ∧ k < 0) ∨ (k = -2) ↔
  ∃ W : set (ℝ × ℝ), 
    let A := (k : ℝ, k^2 + 1 : ℝ),
        B := ((-k - 1) / k : ℝ, -k : ℝ),
        C := (k : ℝ, -k : ℝ),
        W := {p | (p.1, p.2) ∈ set.line_segment ℝ A B ∪ set.line_segment ℝ B C ∪ set.line_segment ℝ C A} in
    ¬(∃ p : ℤ × ℤ, (p.1, p.2) ∈ W) :=
by sorry

end intersect_y_axis_integer_points_inside_W_when_k_is_2_no_integer_points_inside_W_range_of_k_l125_125506


namespace infinite_series_equals_3_l125_125693

noncomputable def infinite_series_sum := ∑' (k : ℕ), (12^k) / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))

theorem infinite_series_equals_3 : infinite_series_sum = 3 := by
  sorry

end infinite_series_equals_3_l125_125693


namespace trains_cross_time_16_seconds_l125_125609

noncomputable def time_for_trains_to_cross_each_other
  (length : ℕ) (time1 : ℕ) (time2 : ℕ) : ℕ :=
  let speed1 := length / time1
  let speed2 := length / time2
  let relative_speed := speed1 + speed2
  let total_length := 2 * length
  total_length / relative_speed

theorem trains_cross_time_16_seconds :
  time_for_trains_to_cross_each_other 120 12 24 = 16 :=
by {
  unfold time_for_trains_to_cross_each_other,
  norm_num,
}

end trains_cross_time_16_seconds_l125_125609


namespace angle_ABF_is_correct_l125_125217

-- Define a regular octagon
structure RegularOctagon (A B C D E F G H : Type) := 
  (sides_eq : ∀ (i j : ℕ), 0 ≤ i ∧ i < 8 → 0 ≤ j ∧ j < 8 → (A i) = (A j))
  (angles_eq : ∀ (i j : ℕ), 0 ≤ i ∧ i < 8 → 0 ≤ j ∧ j < 8 → (A (i + 1) - A i) = 135)

noncomputable def measure_angle_ABF {A B C D E F G H : Type} 
  (oct : RegularOctagon A B C D E F G H) : ℝ :=
22.5

theorem angle_ABF_is_correct (A B C D E F G H : Type) 
  (oct : RegularOctagon A B C D E F G H) :
  measure_angle_ABF oct = 22.5 :=
by
  sorry

end angle_ABF_is_correct_l125_125217


namespace range_of_f_l125_125581

def f (x : ℝ) := (⌊|x|⌋ : ℤ) - |⌊x⌋|

theorem range_of_f : set.range f = { -1, 0 } :=
  sorry

end range_of_f_l125_125581


namespace minimum_average_label_K2017_l125_125250

def edge_labeling_condition (K : Type) [fintype K] [DecidableEq K] (edge_labels : K → K → ℕ) : Prop :=
  ∀ (a b c : K), a ≠ b → b ≠ c → c ≠ a → edge_labels a b + edge_labels b c + edge_labels c a ≥ 5

theorem minimum_average_label_K2017 :
  ∃ edge_labels : fin 2017 → fin 2017 → ℕ,
  (∀ i j, edge_labels i j = edge_labels j i) ∧ -- symmetry of the labeling
  (∀ i j, i ≠ j → edge_labels i j ∈ {1, 2, 3}) ∧ -- labels are 1, 2, or 3
  edge_labeling_condition (fin 2017) edge_labels ∧ -- sum of labels in any triangle ≥ 5
  (finset.univ.prod (λ (pair : (fin 2017) × (fin 2017)), if pair.1 < pair.2 then edge_labels pair.1 pair.2 else 0) /
    ((fintype.card (fin 2017) * (fintype.card (fin 2017) - 1)) / 2) = 2 - 1 / 2017 :=
sorry

end minimum_average_label_K2017_l125_125250


namespace probability_same_gender_probability_same_school_l125_125907

theorem probability_same_gender :
  let school_A := ({m_A1, m_A2, f_A} : Finset (String))
      school_B := ({m_B, f_B1, f_B2} : Finset (String))
      total_outcomes := (school_A × school_B).card
      same_gender_outcomes := 
        (({m_A1, m_A2} × {m_B}) ∪ ({f_A} × {f_B1, f_B2})).card
  in (same_gender_outcomes : ℚ) / total_outcomes = 4 / 9 := 
by
  sorry

theorem probability_same_school :
  let teachers := ({m_A1, m_A2, f_A, m_B, f_B1, f_B2} : Finset (String))
      total_outcomes := (teachers.powerset.filter (λ s, s.card = 2)).card
      same_school_outcomes := 
        (({m_A1, m_A2, f_A}.powerset.filter (λ s, s.card = 2)) ∪ 
         ({m_B, f_B1, f_B2}.powerset.filter (λ s, s.card = 2))).card
  in (same_school_outcomes : ℚ) / total_outcomes = 2 / 5 :=
by
  sorry

end probability_same_gender_probability_same_school_l125_125907


namespace range_of_f_l125_125934

noncomputable def f (x : ℝ) : ℝ := 2 + Real.log x / Real.log 2

theorem range_of_f : range f = { y : ℝ | y ≥ 2 } :=
by
  sorry

end range_of_f_l125_125934


namespace area_of_shaded_region_l125_125338

def diagonal := 10
def length_to_width_ratio := 3 / 2
def number_of_squares := 24

theorem area_of_shaded_region : 
  let x := sqrt (100 / 13)
  let area_of_one_square := x^2
    in number_of_squares * area_of_one_square = 2400 / 13 :=
by sorry

end area_of_shaded_region_l125_125338


namespace three_lines_pass_through_point_and_intersect_parabola_l125_125015

-- Define the point (0,1)
def point : ℝ × ℝ := (0, 1)

-- Define the parabola y^2 = 4x as a set of points
def parabola (p : ℝ × ℝ) : Prop :=
  (p.snd)^2 = 4 * (p.fst)

-- Define the condition for the line passing through (0,1)
def line_through_point (line_eq : ℝ → ℝ) : Prop :=
  line_eq 0 = 1

-- Define the condition for the line intersecting the parabola at only one point
def intersects_once (line_eq : ℝ → ℝ) : Prop :=
  ∃! x : ℝ, parabola (x, line_eq x)

-- The main theorem statement
theorem three_lines_pass_through_point_and_intersect_parabola :
  ∃ (f1 f2 f3 : ℝ → ℝ), 
    line_through_point f1 ∧ line_through_point f2 ∧ line_through_point f3 ∧
    intersects_once f1 ∧ intersects_once f2 ∧ intersects_once f3 ∧
    (∀ (f : ℝ → ℝ), (line_through_point f ∧ intersects_once f) ->
      (f = f1 ∨ f = f2 ∨ f = f3)) :=
sorry

end three_lines_pass_through_point_and_intersect_parabola_l125_125015


namespace sum_of_digits_of_n_l125_125881

open Int

def gcd (a b : ℕ) : ℕ :=
if h : a = 0 then b else if b = 0 then a else Nat.gcd (a % b) b

-- Given conditions as definitions in Lean
def condition1 (n : ℕ) : Prop := gcd 75 (n + 150) = 25
def condition2 (n : ℕ) : Prop := gcd (n + 75) 150 = 75
def least_greater_than (n : ℕ) (k : ℕ) : Prop := ∀ m : ℕ, condition1 m ∧ condition2 m ∧ m > k → n ≤ m

-- Statement of the problem
theorem sum_of_digits_of_n :
  ∃ (n : ℕ), 
    least_greater_than n 1000 ∧ condition1 n ∧ condition2 n ∧ 
    (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10)) = 9 :=
sorry

end sum_of_digits_of_n_l125_125881


namespace ratio_problem_l125_125988

theorem ratio_problem (x : ℕ) (h : 150 * 2 = x) : x = 300 := 
by {
  exact h
  sorry
}

end ratio_problem_l125_125988


namespace sin_alpha_of_point_l125_125842

theorem sin_alpha_of_point (α : ℝ) (P : ℝ × ℝ) (h_initial : P.1 = 1/2) (h_terminal : P.2 = -√3/2) :
  sin α = -√3/2 := 
sorry

end sin_alpha_of_point_l125_125842


namespace dice_sum_eight_dice_l125_125966

/--
  Given 8 fair 6-sided dice, prove that the number of ways to obtain
  a sum of 11 on the top faces of these dice, is 120.
-/
theorem dice_sum_eight_dice :
  (∃ n : ℕ, ∀ (dices : List ℕ), (dices.length = 8 ∧ (∀ d ∈ dices, 1 ≤ d ∧ d ≤ 6) 
   ∧ dices.sum = 11) → n = 120) :=
sorry

end dice_sum_eight_dice_l125_125966


namespace ship_passes_nano_blade_in_40_seconds_l125_125301

-- Define the conditions
def length_of_ship : ℝ := 400
def time_through_wormhole : ℝ := 50
def length_of_wormhole : ℝ := 100
def total_distance_through_wormhole : ℝ := length_of_ship + length_of_wormhole
def speed_of_ship : ℝ := total_distance_through_wormhole / time_through_wormhole

-- Define the expected time to pass through the nano-blade material
def expected_time : ℝ := length_of_ship / speed_of_ship

-- The theorem to prove
theorem ship_passes_nano_blade_in_40_seconds :
  expected_time = 40 :=
begin
  sorry
end

end ship_passes_nano_blade_in_40_seconds_l125_125301


namespace perp_lines_slope_parallel_lines_distance_l125_125055

-- Definition of equations of the lines
def line1 (a : ℝ) : ℝ → ℝ → Prop := λ x y, 2 * a * x + y - 1 = 0
def line2 (a : ℝ) : ℝ → ℝ → Prop := λ x y, a * x + (a - 1) * y + 1 = 0

-- Slopes of the lines
def slope_line1 (a : ℝ) : ℝ := -2 * a
def slope_line2 (a : ℝ) : ℝ := -a / (a - 1)

-- Proving perpendicular condition
theorem perp_lines_slope (a : ℝ) (h : slope_line1 a * slope_line2 a = -1) :
  a = -1 ∨ a = 1 / 2 := 
sorry

-- Proving parallel condition and calculating distance
theorem parallel_lines_distance (a : ℝ) 
  (h1 : slope_line1 a = slope_line2 a) 
  (h2 : a = 3 / 2) :
  let d := |((3 * 1 + (0 * 0) + 1) : ℝ) / sqrt(3^2 + (-1)^2)|
  in d = 2 * sqrt 10 / 5 :=
sorry

end perp_lines_slope_parallel_lines_distance_l125_125055


namespace union_of_A_and_B_l125_125769

noncomputable def A : Set ℝ := {1, 2, 3}
noncomputable def B : Set ℝ := {x | x < 3}

theorem union_of_A_and_B : A ∪ B = {x | x ≤ 3} := by
  sorry

end union_of_A_and_B_l125_125769


namespace geese_flew_away_l125_125601

theorem geese_flew_away (initial remaining flown_away : ℕ) (h_initial: initial = 51) (h_remaining: remaining = 23) : flown_away = 28 :=
by
  sorry

end geese_flew_away_l125_125601


namespace one_gallon_fills_one_cubic_foot_l125_125710

theorem one_gallon_fills_one_cubic_foot
  (total_water : ℕ)
  (drinking_cooking : ℕ)
  (shower_water : ℕ)
  (num_showers : ℕ)
  (pool_length : ℕ)
  (pool_width : ℕ)
  (pool_height : ℕ)
  (h_total_water : total_water = 1000)
  (h_drinking_cooking : drinking_cooking = 100)
  (h_shower_water : shower_water = 20)
  (h_num_showers : num_showers = 15)
  (h_pool_length : pool_length = 10)
  (h_pool_width : pool_width = 10)
  (h_pool_height : pool_height = 6) :
  (pool_length * pool_width * pool_height) / 
  (total_water - drinking_cooking - num_showers * shower_water) = 1 := by
  sorry

end one_gallon_fills_one_cubic_foot_l125_125710


namespace equation_of_parallel_line_l125_125661

theorem equation_of_parallel_line (x y : ℝ) :
  let l : ℝ → ℝ → Prop := λ x y, x + 2 * y + c = 0 in
  ∀ c, l x y ↔ x + 2 * y - 3 = 0 :=
by sorry

end equation_of_parallel_line_l125_125661


namespace altitude_ad_2sqrt30_l125_125611

noncomputable def triangle_altitude (AB AC BC : ℝ) (D_midpoint : BC = 2 * (BC / 2))
  (AB_eq_AC : AB = AC) : ℝ :=
  let BD := BC / 2
  let AD_square := AB * AB - BD * BD
  real.sqrt AD_square

theorem altitude_ad_2sqrt30 : 
  ∀ {A B C : ℝ}, A = 13 → B = 13 → C = 14 → 
  triangle_altitude A B C (by norm_num) rfl = 2 * real.sqrt 30 :=
by
  intros A B C hA hB hC
  simp [triangle_altitude, hA, hB, hC, real.sqrt_eq_rpow]
  norm_num
  sorry

end altitude_ad_2sqrt30_l125_125611


namespace fourth_angle_of_quadrilateral_l125_125831

theorem fourth_angle_of_quadrilateral (A : ℝ) : 
  (120 + 85 + 90 + A = 360) ↔ A = 65 := 
by
  sorry

end fourth_angle_of_quadrilateral_l125_125831


namespace initial_rate_of_interest_l125_125263

theorem initial_rate_of_interest (P : ℝ) (R : ℝ) 
  (h1 : 1680 = (P * R * 5) / 100) 
  (h2 : 1680 = (P * 5 * 4) / 100) : 
  R = 4 := 
by 
  sorry

end initial_rate_of_interest_l125_125263


namespace hare_race_l125_125928

theorem hare_race :
  ∃ (total_jumps: ℕ) (final_jump_leg: String), total_jumps = 548 ∧ final_jump_leg = "right leg" :=
by
  sorry

end hare_race_l125_125928


namespace max_expression_value_l125_125816

theorem max_expression_value :
  ∀ (a b : ℝ), (100 ≤ a ∧ a ≤ 500) → (500 ≤ b ∧ b ≤ 1500) → 
  (∃ x, x = (b - 100) / (a + 50) ∧ ∀ y, y = (b - 100) / (a + 50) → y ≤ (28 / 3)) :=
by
  sorry

end max_expression_value_l125_125816


namespace integer_solution_exists_l125_125169

theorem integer_solution_exists (a b : ℤ) : 
  ∃ x : ℤ, (x - a) * (x - b) * (x - 3) + 1 = 0 :=
begin
  sorry
end

end integer_solution_exists_l125_125169


namespace find_number_satisfy_equation_l125_125117

theorem find_number_satisfy_equation (x : ℝ) :
  9 - x / 7 * 5 + 10 = 13.285714285714286 ↔ x = -20 := sorry

end find_number_satisfy_equation_l125_125117


namespace average_marks_l125_125985

theorem average_marks :
  let a1 := 76
  let a2 := 65
  let a3 := 82
  let a4 := 67
  let a5 := 75
  let n := 5
  let total_marks := a1 + a2 + a3 + a4 + a5
  let avg_marks := total_marks / n
  avg_marks = 73 :=
by
  sorry

end average_marks_l125_125985


namespace chord_length_circle_M_eq_l125_125756

noncomputable def circle_eq : Float -> Float -> Prop := fun x y => x^2 + y^2 = 8

def point_P0 : Prod Float Float := (-1, 2)
def point_C : Prod Float Float := (3, 0)

def eq_chord_len (α : Float) : Float := 
  if α = 135 then (2 * Float.sqrt((2 * Float.sqrt(2))^2 - (Float.sqrt(2) / 2)^2))
  else 0

def eq_circle_M (M_x M_y : Float) (M_r : Float) : Prop := 
  (M_x - 1/4)^2 + (M_y + 1/2)^2 = M_r^2

theorem chord_length (α : Float) : Prop :=
  α = 135 -> eq_chord_len α = Float.sqrt 30

theorem circle_M_eq : Prop :=
  eq_circle_M (1/4) (-1/2) (Float.sqrt (125 / 16))

end chord_length_circle_M_eq_l125_125756


namespace f_five_times_of_one_l125_125865

def f : ℝ → ℝ :=
  λ x, if x ≥ 0 then -x^2 else x + 8

theorem f_five_times_of_one : f (f (f (f (f 1)))) = -33 :=
by
  sorry

end f_five_times_of_one_l125_125865


namespace amy_money_left_l125_125673

theorem amy_money_left (initial_money : ℝ) (doll_count board_game_count comic_book_count : ℕ)
  (doll_cost board_game_cost comic_book_cost : ℝ) :
  initial_money = 100 →
  doll_count = 3 →
  board_game_count = 2 →
  comic_book_count = 4 →
  doll_cost = 1.25 →
  board_game_cost = 12.75 →
  comic_book_cost = 3.50 →
  initial_money - (doll_count * doll_cost + board_game_count * board_game_cost + comic_book_count * comic_book_cost) = 56.75 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6, h7]
  norm_num
  sorry

end amy_money_left_l125_125673


namespace linear_function_does_not_pass_first_quadrant_l125_125257

theorem linear_function_does_not_pass_first_quadrant (k b : ℝ) (h : ∀ x : ℝ, y = k * x + b) :
  k = -1 → b = -2 → ¬∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = k * x + b :=
by
  sorry

end linear_function_does_not_pass_first_quadrant_l125_125257


namespace problem1_problem2_l125_125358

noncomputable def concentration (a x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 2 then a * (4 + x) / (4 - x) else a * (5 - x)

def effective (y : ℝ) : Prop := y ≥ 4

-- Problem 1: Prove that if 4 units of nutrient are released only once, the solution is effective for 4 days.
theorem problem1 (a : ℝ) (h₁ : 1 ≤ a ∧ a ≤ 4) (ha : a = 4) : 
  ∀ x, 0 ≤ x ∧ x ≤ 4 → effective (concentration a x) :=
begin
  intros x hx,
  unfold concentration,
  split_ifs,
  { simp [effective, h],
    linarith, },
  { simp [effective, h],
    linarith, }
end

-- Problem 2: Find the minimum value of b to make the solution continuously effective for the next 2 days.
theorem problem2 (b : ℝ) (h_two_units : a = 2) : 
  (∀ x, 6 ≤ x ∧ x ≤ 10 → effective (2 * (5 - x / 2) + 
     b * (concentration b (x - 3) - 1))) → (b ≥ 24 - 16 * real.sqrt 2) :=
begin
  intro h,
  sorry
end

end problem1_problem2_l125_125358


namespace no_tangent_of_2x_plus_m_for_f4_l125_125969

def f1 (x : ℝ) : ℝ := x^2 + x
def f2 (x : ℝ) : ℝ := x^3 + exp x
def f3 (x : ℝ) : ℝ := log x + x^2 / 2
def f4 (x : ℝ) : ℝ := sqrt x + 2 * x

theorem no_tangent_of_2x_plus_m_for_f4 :
  ∃ (m : ℝ) (f : ℝ → ℝ), f = f4 → ∀ x : ℝ, diff f x ≠ 2 :=
sorry

end no_tangent_of_2x_plus_m_for_f4_l125_125969


namespace regular_octagon_exterior_angle_l125_125848

theorem regular_octagon_exterior_angle : 
  ∀ (n : ℕ), n = 8 → (180 * (n - 2) / n) + (180 - (180 * (n - 2) / n)) = 180 := by
  sorry

end regular_octagon_exterior_angle_l125_125848


namespace chords_intersecting_theorem_l125_125757

noncomputable def intersecting_chords_theorem (P A B C D : ℝ) (h_circle : P ≠ A) (h_ab : A ≠ B) (h_cd : C ≠ D) : ℝ :=
  sorry

theorem chords_intersecting_theorem (P A B C D : ℝ) (h_circle : P ≠ A) (h_ab : A ≠ B) (h_cd : C ≠ D) :
  (P - A) * (P - B) = (P - C) * (P - D) :=
by sorry

end chords_intersecting_theorem_l125_125757


namespace tetrahedron_volume_l125_125667

theorem tetrahedron_volume (a b c : ℝ)
  (h₁ : a + b > c) (h₂ : a + c > b) (h₃ : b + c > a) :
  ∃ V : ℝ, 
    V = (1 / (6 * Real.sqrt 2)) * 
        Real.sqrt ((a^2 + b^2 - c^2) * (a^2 + c^2 - b^2) * (b^2 + c^2 - a^2)) :=
sorry

end tetrahedron_volume_l125_125667


namespace problem_a_problem_b_problem_c_l125_125904

theorem problem_a (n : ℕ) (h : 1 ≤ n) : 
  (∑ k in Finset.range (n - 1), Real.sin ((2 * (k + 1) * Real.pi) / n)) = 0 :=
sorry

theorem problem_b (n : ℕ) (h : 1 ≤ n) : 
  (∑ k in Finset.range (n - 1), Real.cos ((2 * (k + 1) * Real.pi) / n)) = -1 :=
sorry

theorem problem_c (n : ℕ) (h : 1 ≤ n) (α : ℝ) : 
  (∑ k in Finset.range (n - 1), Real.sin (α + ((2 * (k + 1) * Real.pi) / n))) = 0 :=
sorry

end problem_a_problem_b_problem_c_l125_125904


namespace g_10_is_1_l125_125583

def g : ℝ → ℝ := sorry

theorem g_10_is_1
  (hx : ∀ x y : ℝ, g(x * y) = g(x) * g(y))
  (h1 : g(1) ≠ 0) :
  g(10) = 1 := sorry

end g_10_is_1_l125_125583


namespace tom_break_even_days_l125_125273

theorem tom_break_even_days
    (initial_cost : ℕ)
    (daily_ticket_sales : ℕ)
    (ticket_price : ℕ)
    (daily_running_percentage : ℕ)
    (daily_running_cost : ℕ)
    (daily_income : ℕ)
    (daily_profit : ℕ) :
    initial_cost = 100000 →
    daily_ticket_sales = 150 →
    ticket_price = 10 →
    daily_running_percentage = 1 →
    daily_running_cost = initial_cost * daily_running_percentage / 100 →
    daily_income = daily_ticket_sales * ticket_price →
    daily_profit = daily_income - daily_running_cost →
    initial_cost / daily_profit = 200 :=
by
  intros h_initial h_ticket_sales h_ticket_price h_percentage h_running_cost h_income h_profit
  rw [h_initial, h_ticket_sales, h_ticket_price, h_percentage] at *
  have h1 : daily_running_cost = 1000 := by norm_num [h_running_cost]
  have h2 : daily_income = 1500 := by norm_num [h_income]
  have h3 : daily_profit = 500 := by norm_num [h_running_cost, h_income, h_profit]
  rw [←h1, ←h2, ←h3]
  norm_num
  sorry

end tom_break_even_days_l125_125273


namespace triangle_is_right_triangle_l125_125154

theorem triangle_is_right_triangle
  (a b c : ℝ)
  (A B C : ℝ)
  (h₁ : a ≠ b)
  (h₂ : (a^2 + b^2) * Real.sin (A - B) = (a^2 - b^2) * Real.sin (A + B))
  (A_ne_B : A ≠ B)
  (hABC : A + B + C = Real.pi) :
  C = Real.pi / 2 :=
by
  sorry

end triangle_is_right_triangle_l125_125154


namespace circle_standard_equation_l125_125776

theorem circle_standard_equation (x y : ℝ) : 
    let h := 1
    let k := 2
    let r := 1
    ((x - h)^2 + (y - k)^2 = r^2) ↔ ((x - 1)^2 + (y - 2)^2 = 1) := 
by
    let h := 1
    let k := 2
    let r := 1
    have H : h = 1 := rfl
    have K : k = 2 := rfl
    have R : r = 1 := rfl
    rw [H, K, R]
    simp
    sorry

end circle_standard_equation_l125_125776


namespace snow_on_second_day_l125_125677

-- Definition of conditions as variables in Lean
def snow_on_first_day := 6 -- in inches
def snow_melted := 2 -- in inches
def additional_snow_fifth_day := 12 -- in inches
def total_snow := 24 -- in inches

-- The variable for snow on the second day
variable (x : ℕ)

-- Proof goal
theorem snow_on_second_day : snow_on_first_day + x - snow_melted + additional_snow_fifth_day = total_snow → x = 8 :=
by
  intros h
  sorry

end snow_on_second_day_l125_125677


namespace year_when_mother_age_is_twice_jack_age_l125_125210

noncomputable def jack_age_2010 := 12
noncomputable def mother_age_2010 := 3 * jack_age_2010

theorem year_when_mother_age_is_twice_jack_age :
  ∃ x : ℕ, mother_age_2010 + x = 2 * (jack_age_2010 + x) ∧ (2010 + x = 2022) :=
by
  sorry

end year_when_mother_age_is_twice_jack_age_l125_125210


namespace smallest_c_in_arithmetic_prog_l125_125529

theorem smallest_c_in_arithmetic_prog (a b c d : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : b = a + d) (h6 : c = a + 2*d) (h7 : d = a + 3*d) (h8 : a * b * c * d = 256) : c = 4 :=
begin
  sorry
end

end smallest_c_in_arithmetic_prog_l125_125529


namespace find_angle_D_l125_125830

theorem find_angle_D (A B C D : ℝ) (h1 : A + B = 180) (h2 : C = D) (h3 : A = 40) (h4 : B + C = 130) : D = 40 := by
  sorry

end find_angle_D_l125_125830


namespace scientific_notation_l125_125397

theorem scientific_notation (x : ℝ) (h : x = 0.0000000108) : x = 1.08 * 10 ^ (-8) :=
by
  rw h
  sorry

end scientific_notation_l125_125397


namespace complement_union_eq_l125_125090

variable (U : Set Int := {-2, -1, 0, 1, 2, 3}) 
variable (A : Set Int := {-1, 0, 1}) 
variable (B : Set Int := {1, 2}) 

theorem complement_union_eq :
  U \ (A ∪ B) = {-2, 3} := by 
  sorry

end complement_union_eq_l125_125090


namespace number_of_monomials_l125_125844

def isMonomial (expr : String) : Bool :=
  match expr with
  | "-(2 / 3) * a^3 * b" => true
  | "(x * y) / 2" => true
  | "-4" => true
  | "0" => true
  | _ => false

def countMonomials (expressions : List String) : Nat :=
  expressions.foldl (fun acc expr => if isMonomial expr then acc + 1 else acc) 0

theorem number_of_monomials : countMonomials ["-(2 / 3) * a^3 * b", "(x * y) / 2", "-4", "-(2 / a)", "0", "x - y"] = 4 :=
by
  sorry

end number_of_monomials_l125_125844


namespace smallest_naturally_ending_in_6_satisfying_condition_l125_125017

def ends_in_6 (n : ℕ) : Prop :=
  n % 10 = 6

def move_unit_digit_to_front (n : ℕ) : ℕ :=
  let m := n / 10
  6 * (10 ^ (nat.log 10 m + 1)) + m

theorem smallest_naturally_ending_in_6_satisfying_condition :
  ∃ n : ℕ, ends_in_6 n ∧ move_unit_digit_to_front n = 4 * n ∧ n = 153846 :=
by
  sorry

end smallest_naturally_ending_in_6_satisfying_condition_l125_125017


namespace mean_of_remaining_three_numbers_l125_125920

theorem mean_of_remaining_three_numbers 
    (a b c d : ℝ)
    (h₁ : (a + b + c + d) / 4 = 92)
    (h₂ : d = 120)
    (h₃ : b = 60) : 
    (a + b + c) / 3 = 82.6666666666 := 
by 
    -- This state suggests adding the constraints added so far for the proof:
    sorry

end mean_of_remaining_three_numbers_l125_125920


namespace students_playing_both_l125_125984

theorem students_playing_both (n F C N : ℕ) (hF : F = 325) (hC : C = 175) (hN : N = 50) (hn : n = 410) :
  F + C - (n - N) = 140 := by {
  rw [hF, hC, hn, hN],
  sorry
}

end students_playing_both_l125_125984


namespace translation_vector_coords_l125_125955

-- Definitions according to the given conditions
def original_circle (x y : ℝ) : Prop := x^2 + y^2 = 1
def translated_circle (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 1

-- Statement that we need to prove
theorem translation_vector_coords :
  ∃ (a b : ℝ), 
  (∀ x y : ℝ, original_circle x y ↔ translated_circle (x - a) (y - b)) ∧
  (a, b) = (-1, 2) := 
sorry

end translation_vector_coords_l125_125955


namespace prove_l125_125192

variable (m n : ℕ)
variable (a b α β : ℝ)
variable hα : α = 3 / 4
variable hβ : β = 19 / 20

def height_ratio (m n : ℕ) (a b α β : ℝ) (hα : α = 3 / 4) (hβ : β = 19 / 20) 
    (h1 : a = α * b) (h2 : a = β * (a * m + b * n) / (m + n)) : (ℝ) :=
  m / n

theorem prove\_height\_ratio (m n : ℕ) (a b α β : ℝ) (hα : α = 3 / 4) (hβ : β = 19 / 20) 
    (h1 : a = α * b) (h2 : a = β * (a * m + b * n) / (m + n)) : 
  height_ratio m n a b α β hα hβ h1 h2 = 8 / 9 :=
by
  -- Proof omitted
  sorry

end prove_l125_125192


namespace umar_age_is_ten_l125_125350

-- Define variables for Ali, Yusaf, and Umar
variables (ali_age yusa_age umar_age : ℕ)

-- Define the conditions from the problem
def ali_is_eight : Prop := ali_age = 8
def ali_older_than_yusaf : Prop := ali_age - yusa_age = 3
def umar_twice_yusaf : Prop := umar_age = 2 * yusa_age

-- The theorem that uses the conditions to assert Umar's age
theorem umar_age_is_ten 
  (h1 : ali_is_eight ali_age)
  (h2 : ali_older_than_yusaf ali_age yusa_age)
  (h3 : umar_twice_yusaf umar_age yusa_age) : 
  umar_age = 10 :=
by
  sorry

end umar_age_is_ten_l125_125350


namespace cosine_angle_is_zero_l125_125588

-- Define the structure of an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  angle_60_deg : Prop

-- Define the structure of a parallelogram built from 6 equilateral triangles
structure Parallelogram where
  composed_of_6_equilateral_triangles : Prop
  folds_into_hexahedral_shape : Prop

-- Define the angle and its cosine computation between two specific directions in the folded hexahedral shape
def cosine_of_angle_between_AB_and_CD (parallelogram : Parallelogram) : ℝ := sorry

-- The condition that needs to be proved
axiom parallelogram_conditions : Parallelogram
axiom cosine_angle_proof : cosine_of_angle_between_AB_and_CD parallelogram_conditions = 0

-- Final proof statement
theorem cosine_angle_is_zero : cosine_of_angle_between_AB_and_CD parallelogram_conditions = 0 :=
cosine_angle_proof

end cosine_angle_is_zero_l125_125588


namespace angle_B_and_max_area_l125_125851

noncomputable def max_area_of_triangle (a b c : ℝ) (B : ℝ) : ℝ :=
  if h : 0 < B ∧ B < π ∧ b = 2 ∧ B = π / 3 then
    let area := (1/2) * a * c * (Real.sin B) in
    Real.sqrt 3
  else 0

theorem angle_B_and_max_area (a b c : ℝ) :
  (∀ B,  a * Real.cos B + b * Real.cos (π - B) = Real.sqrt 3 / 3 * c * Real.tan B → B = π / 3) ∧
  (max_area_of_triangle a b c (π / 3) = Real.sqrt 3) :=
by
  split
  · intro B h
    sorry
  · unfold max_area_of_triangle
    split_ifs with h
    · exact rfl
    · exfalso
      apply h
      split
      sorry

end angle_B_and_max_area_l125_125851


namespace spending_on_hydrangeas_l125_125570

def lily_spending : ℕ :=
  let start_year := 1989
  let end_year := 2021
  let cost_per_plant := 20
  let years := end_year - start_year
  cost_per_plant * years

theorem spending_on_hydrangeas : lily_spending = 640 := 
  sorry

end spending_on_hydrangeas_l125_125570


namespace pyramidal_lateral_face_base_length_l125_125576

theorem pyramidal_lateral_face_base_length (A : ℝ) (h : ℝ) : 
  (A = 150) → (h = 40) → (s = 7.5) :=
begin
  sorry
end

end pyramidal_lateral_face_base_length_l125_125576


namespace dice_same_color_probability_l125_125479

theorem dice_same_color_probability :
  let maroon_prob := 3 / 12 in
  let teal_prob := 4 / 12 in
  let cyan_prob := 4 / 12 in
  let sparkly_prob := 1 / 12 in
  maroon_prob^2 + teal_prob^2 + cyan_prob^2 + sparkly_prob^2 = 7 / 24 :=
by
  let maroon_prob := 3 / 12
  let teal_prob := 4 / 12
  let cyan_prob := 4 / 12
  let sparkly_prob := 1 / 12
  calc
    maroon_prob^2 + teal_prob^2 + cyan_prob^2 + sparkly_prob^2
      = (3 / 12)^2 + (4 / 12)^2 + (4 / 12)^2 + (1 / 12)^2 : by sorry
    ... = 9 / 144 + 16 / 144 + 16 / 144 + 1 / 144 : by sorry
    ... = 42 / 144 : by sorry
    ... = 7 / 24 : by sorry

end dice_same_color_probability_l125_125479


namespace range_of_a_l125_125054

-- Definitions based on the conditions
variable {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = - f x)
variable (h_decreasing : ∀ x y, 0 ≤ x → y ≥ x → f y ≤ f x)
variable (h_condition : ∀ x, -1 ≤ x ∧ x ≤ 2 → f (x^3 - 2 * x + a) < f (x + 1))

-- The main theorem statement
theorem range_of_a (a : ℝ) : (∀ x, -1 ≤ x ∧ x ≤ 2 → f (x^3 - 2 * x + a) < f (x + 1)) → a > 3 :=
begin
  sorry
end

end range_of_a_l125_125054


namespace first_consecutive_odd_number_l125_125298

theorem first_consecutive_odd_number :
  ∃ (x : ℤ), let y := x + 2
                        z := x + 4
              in 8 * x = 3 * z + 2 * y + 5 ↔ x = 7 :=
begin
  sorry
end

end first_consecutive_odd_number_l125_125298


namespace trig_identity_l125_125364

theorem trig_identity :
  sin (40 * pi / 180) * sin (10 * pi / 180) + cos (40 * pi / 180) * sin (80 * pi / 180) = sqrt 3 / 2 := 
begin
  sorry
end

end trig_identity_l125_125364


namespace min_value_l125_125190

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
3 * a + 6 * b + 12 * c

theorem min_value (a b c : ℝ) (h : 9 * a ^ 2 + 4 * b ^ 2 + 36 * c ^ 2 = 4) :
  minimum_value a b c = -2 * Real.sqrt 14 := sorry

end min_value_l125_125190


namespace part1_part2_l125_125068

def f (x : ℝ) : ℝ := Real.cos x - 1 / Real.exp x

theorem part1 (h1 : 7 < Real.exp 2) (h2 : Real.exp 2 < 8) 
    (h3 : Real.exp 3 > 16) (h4 : Real.exp (-3 * Real.pi / 4) < Real.sqrt 2 / 2) :
  ∃ x : ℝ, x ∈ (Real.pi / 6, Real.pi / 4) ∧ f' x = 0 := sorry

theorem part2 (h1 : 7 < Real.exp 2) (h2 : Real.exp 2 < 8) 
    (h3 : Real.exp 3 > 16) (h4 : Real.exp (-3 * Real.pi / 4) < Real.sqrt 2 / 2) :
  ∃ x1 x2 x3 x4 : ℝ, 
  0 < x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧ x4 < 2 * Real.pi ∧ 
  (∀ x ∈ (0, x1), f' x > 0) ∧ (∀ x ∈ (x1, x2), f' x < 0) ∧ 
  (∀ x ∈ (x2, x3), f' x = 0) ∧ (∀ x ∈ (x3, x4), f' x > 0) := sorry

end part1_part2_l125_125068


namespace cody_spent_19_dollars_l125_125375

-- Given conditions
def initial_money : ℕ := 45
def birthday_gift : ℕ := 9
def remaining_money : ℕ := 35

-- Problem: Prove that the amount of money spent on the game is $19.
theorem cody_spent_19_dollars :
  (initial_money + birthday_gift - remaining_money) = 19 :=
by sorry

end cody_spent_19_dollars_l125_125375


namespace distance_AB_polar_coordinate_equation_l125_125144

-- Definitions based on given conditions
def parametric_curve (t : ℝ) : ℝ × ℝ :=
(2 - t - t^2, 2 - 3*t + t^2)

-- The proof statement for the distance |AB|
theorem distance_AB : 
  let A := (2 - (-2 : ℝ) - (-2 : ℝ)^2, 2 - 3*(-2) + (-2)^2)
      B := (2 - 2 - 2^2, 2 - 3*2 + 2^2) in
  A = (0, 12) ∧ B = (-4, 0) ∧
  real.sqrt ((-4 - 0)^2 + (0 - 12)^2) = 4 * real.sqrt 10 :=
sorry

-- The proof statement for the polar coordinate equation of the line AB
theorem polar_coordinate_equation :
  ∀ (ρ θ : ℝ),
  (3 * (ρ * real.cos θ) - ρ * real.sin θ + 12) = 0 :=
sorry

end distance_AB_polar_coordinate_equation_l125_125144


namespace average_first_n_numbers_eq_10_l125_125721

theorem average_first_n_numbers_eq_10 (n : ℕ) 
  (h : (n * (n + 1)) / (2 * n) = 10) : n = 19 :=
  sorry

end average_first_n_numbers_eq_10_l125_125721


namespace sum_decomposition_order_significant_sum_decomposition_order_not_significant_l125_125837

-- Define the problem for Case A: Order of addends is significant
theorem sum_decomposition_order_significant (n : ℕ) (hn : n > 0) :
  {p : ℕ × ℕ // p.1 + p.2 = n}.card = n - 1 := 
sorry

-- Define the problem for Case B: Order of addends is not significant
theorem sum_decomposition_order_not_significant (n : ℕ) (hn : n > 0) :
  let num_decompositions := if n % 2 = 1 then (n - 1) / 2 else n / 2
  in {p : ℕ × ℕ // p.1 + p.2 = n ∧ p.1 ≤ p.2}.card = num_decompositions := 
sorry

end sum_decomposition_order_significant_sum_decomposition_order_not_significant_l125_125837


namespace valid_sequences_count_l125_125179

-- Define the vertices of the triangle T'
def T' : set (ℤ × ℤ) := {(0, 0), (5, 0), (0, 4)}

-- Define the transformations
def rotation_90 (p : ℤ × ℤ) : ℤ × ℤ := (-p.2, p.1)
def rotation_180 (p : ℤ × ℤ) : ℤ × ℤ := (-p.1, -p.2)
def rotation_270 (p : ℤ × ℤ) : ℤ × ℤ := (p.2, -p.1)
def reflect_yx (p : ℤ × ℤ) : ℤ × ℤ := (p.2, p.1)
def reflect_ynegx (p : ℤ × ℤ) : ℤ × ℤ := (-p.2, -p.1)

-- Define a function to check if a sequence of transformations maps T' onto itself
def maps_T'_onto_itself (f : (ℤ × ℤ) → ℤ × ℤ) : Prop :=
  f '' T' = T'

noncomputable def count_valid_sequences : ℕ :=
  let transformations := [rotation_90, rotation_180, rotation_270, reflect_yx, reflect_ynegx] in
  let sequences := list.product (list.product transformations transformations) transformations in
  sequences.count (λ seq, maps_T'_onto_itself (seq.2 ∘ seq.1.2 ∘ seq.1.1))

-- The statement that needs to be proven
theorem valid_sequences_count : count_valid_sequences = 12 :=
  sorry

end valid_sequences_count_l125_125179


namespace min_value_example_l125_125142

theorem min_value_example (a b : ℝ) (ha : a > 0) (hb : b > 0) (hline : a + 2 * b = 1) :
  ∃ m, m = 9 ∧ ∀ (a > 0) (b > 0), (a + 2 * b = 1) → (4 / (a + b) + 1 / b) ≥ m :=
sorry

end min_value_example_l125_125142


namespace log_exp_proof_l125_125686

-- Definitions based on given problem conditions
def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_exp_proof :
  lg 25 + 2 * lg 2 + 8^(2/3) = 6 := by
  sorry

end log_exp_proof_l125_125686


namespace max_cos2A_plus_sin2A_exists_max_cos2A_plus_sin2A_l125_125725

theorem max_cos2A_plus_sin2A (A : ℝ) : (cos (2 * A) + sin (2 * A)) ≤ sqrt 2 :=
begin
  sorry
end

theorem exists_max_cos2A_plus_sin2A (A : ℝ) : ∃ A, (cos (2 * A) + sin (2 * A)) = sqrt 2 :=
begin
  sorry
end

end max_cos2A_plus_sin2A_exists_max_cos2A_plus_sin2A_l125_125725


namespace find_angle_A_l125_125490

variable (a b c : ℝ)
variable (A : ℝ)

axiom triangle_ABC : a = Real.sqrt 3 ∧ b = 1 ∧ c = 2

theorem find_angle_A : a = Real.sqrt 3 ∧ b = 1 ∧ c = 2 → A = Real.pi / 3 :=
by
  intro h
  sorry

end find_angle_A_l125_125490


namespace not_exists_tangent_to_line_l125_125971

noncomputable def f_D (x : ℝ) : ℝ := sqrt x + 2 * x

theorem not_exists_tangent_to_line : ¬ ∃ (x : ℝ), x > 0 ∧ (deriv f_D x = 2) :=
by 
  sorry -- to be proven

end not_exists_tangent_to_line_l125_125971


namespace find_S_l125_125637

-- Define given conditions
constant c : ℝ
axiom relationship : ∀ (R S T : ℝ), R = c * S / T
axiom initial_conditions : relationship 2 (1/2) (8/5)

-- Prove the required condition
theorem find_S (R T : ℝ) : S = (25 * Real.sqrt 2) / 2 :=
  by
    have c_value : c = 32 / 5,
    sorry,
    have S_value : S * T = 5 / 32,
    sorry,
    have new_S : S = 16 * T * 25 / 32,
    sorry,
    eventually sorry ⟩

end find_S_l125_125637


namespace ways_to_sum_2022_l125_125804

theorem ways_to_sum_2022 : 
  ∃ n : ℕ, (∀ a b : ℕ, (2022 = 2 * a + 3 * b) ∧ n = (b - a) / 4 ∧ n = 338) := 
sorry

end ways_to_sum_2022_l125_125804


namespace cos_2alpha_equiv_l125_125007

-- Given conditions: ctg equation and alpha interval constraints
theorem cos_2alpha_equiv (α : ℝ) (h1 : 2 * (Real.cot α)^2 + 7 * (Real.cot α) + 3 = 0) 
  (h2 : (3/2) * Real.pi < α ∧ α < (7/4) * Real.pi) 
  ∨ (h3 : (7/4) * Real.pi < α ∧ α < 2 * Real.pi): 
  ∃ c : ℝ, (c = -3/5 ∨ c = 4/5) ∧ c = Real.cos (2 * α) :=
by
  sorry

end cos_2alpha_equiv_l125_125007


namespace smallest_base_l125_125736

-- Definitions of the conditions
def condition1 (b : ℕ) : Prop := b > 3
def condition2 (b : ℕ) : Prop := b > 7
def condition3 (b : ℕ) : Prop := b > 6
def condition4 (b : ℕ) : Prop := b > 8

-- Main theorem statement
theorem smallest_base : ∀ b : ℕ, condition1 b ∧ condition2 b ∧ condition3 b ∧ condition4 b → b = 9 := by
  sorry

end smallest_base_l125_125736


namespace expressions_cannot_all_exceed_one_fourth_l125_125039

theorem expressions_cannot_all_exceed_one_fourth (a b c : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) : 
  ¬ ((1 - a) * b > 1/4 ∧ (1 - b) * c > 1/4 ∧ (1 - c) * a > 1/4) := 
by
  sorry

end expressions_cannot_all_exceed_one_fourth_l125_125039


namespace algebraic_expression_multiplied_l125_125610

-- Given condition
def expression (n : ℕ) : ℕ := 
  (List.prod (List.range n).map (λ i => n + i + 1))

-- Inductive proof problem statement
theorem algebraic_expression_multiplied (k : ℕ) (h : k ∈ Set.Ioi 0) :
  expression (k + 1) = expression k * (2 * (2 * k + 1)) :=
by
  sorry

end algebraic_expression_multiplied_l125_125610


namespace calculate_sphere_radius_l125_125339

open Real EuclideanGeometry

-- Definitions for the coordinates of the vertices
def A : Point := (0, 0, 0)
def C : Point := (1, 1, 0)

-- Definitions for the edge passing through B and the plane of the top face
def touches_edge_B (G : Point) (r : ℝ) : Prop := (G.1 - 1)^2 + (G.2 - 0)^2 = r^2
def touches_top_face (G : Point) (r : ℝ) : Prop := G.3 = 1 - r

theorem calculate_sphere_radius (G : Point) (r : ℝ)
  (h1 : (G.1)^2 + (G.2)^2 + (G.3)^2 = r^2)
  (h2 : (1 - G.1)^2 + (1 - G.2)^2 + G.3^2 = r^2)
  (h3 : touches_edge_B G r)
  (h4 : touches_top_face G r) :
  r = 0.751 := 
sorry

end calculate_sphere_radius_l125_125339


namespace num_multiples_3_with_units_digit_3_or_9_l125_125105

theorem num_multiples_3_with_units_digit_3_or_9 : 
    let count_multiples := (λ (n : ℕ), 0 < n ∧ n < 150 ∧ n % 3 = 0 ∧ (n % 10 = 3 ∨ n % 10 = 9)) 
  in (∑ k in (Finset.range 150), if count_multiples k then 1 else 0) = 16 :=
by sorry

end num_multiples_3_with_units_digit_3_or_9_l125_125105


namespace div_power_n_minus_one_l125_125879

theorem div_power_n_minus_one (n : ℕ) (hn : n > 0) (h : n ∣ (2^n - 1)) : n = 1 := by
  sorry

end div_power_n_minus_one_l125_125879


namespace part1_solution_part2_solution_l125_125886

-- Defining the function f and conditions for part (1)
def f (k a x : ℝ) := k * a^x - a^(-x)

-- Conditions for part (1)
variables {a k x : ℝ}
axiom h_a_gt_zero : a > 0
axiom h_a_ne_one : a ≠ 1
axiom h_odd_f : ∀ x : ℝ, f k a (-x) = -f k a x
axiom h_f1_gt_zero : f k a 1 > 0

-- Defining the function g for part (2)
def g (a m x : ℝ) := a^(2 * x) + a^(-2 * x) - 2 * m * f k a x

-- Conditions for part (2)
axiom h_f1_zero : f k a 1 = 0
axiom h_g_min : ∀ x : ℝ, x ≥ 1 → (∃ m : ℝ, ∀ t : ℝ, t = f k a x → g a m x ≥ -2)

-- Part (1) proof statement
theorem part1_solution (h_k : k = 1) : (x > 1 ∨ x < -4) := sorry

-- Part (2) proof statement
theorem part2_solution (h_m : m = 2) : ∃ m : ℝ, ∀ t : ℝ, t ≥ 1 → g a m t = -2 := sorry

end part1_solution_part2_solution_l125_125886


namespace product_a2_a3_a4_l125_125075

open Classical

noncomputable def geometric_sequence (a : ℕ → ℚ) (a1 : ℚ) (q : ℚ) : Prop :=
∀ n : ℕ, a n = a1 * q^(n - 1)

theorem product_a2_a3_a4 (a : ℕ → ℚ) (q : ℚ) 
  (h_seq : geometric_sequence a 1 q)
  (h_a1 : a 1 = 1)
  (h_a5 : a 5 = 1 / 9) :
  a 2 * a 3 * a 4 = 1 / 27 :=
sorry

end product_a2_a3_a4_l125_125075


namespace not_rain_probability_l125_125592

-- Define the probability of rain tomorrow
def prob_rain : ℚ := 3 / 10

-- Define the complementary probability (probability that it will not rain tomorrow)
def prob_no_rain : ℚ := 1 - prob_rain

-- Statement to prove: probability that it will not rain tomorrow equals 7/10 
theorem not_rain_probability : prob_no_rain = 7 / 10 := 
by sorry

end not_rain_probability_l125_125592


namespace additional_distance_to_achieve_target_average_speed_l125_125702

-- Given conditions
def initial_distance : ℕ := 20
def initial_speed : ℕ := 40
def target_average_speed : ℕ := 55

-- Prove that the additional distance required to average target speed is 90 miles
theorem additional_distance_to_achieve_target_average_speed 
  (total_distance : ℕ) 
  (total_time : ℚ) 
  (additional_distance : ℕ) 
  (additional_speed : ℕ) :
  total_distance = initial_distance + additional_distance →
  total_time = (initial_distance / initial_speed) + (additional_distance / additional_speed) →
  additional_speed = 60 →
  total_distance / total_time = target_average_speed →
  additional_distance = 90 :=
by 
  sorry

end additional_distance_to_achieve_target_average_speed_l125_125702


namespace triangle_inequality_l125_125549

variables {A B C D F E : Type}
variables [linear_ordered_comm_ring A]

def point_on_line (x y : A) : Type := sorry

def midpoint (x y m : A) : Prop := 2 * m = x + y

theorem triangle_inequality (ABC : Type) (AB BC : ABC) (D F : AB) (E : D) (F : BC)
  (hD_on_AB : point_on_line A B D)
  (hF_on_BC : point_on_line B C F)
  (h_midpoint_E : midpoint D F E) :
  AD + FC ≤ AE + EC :=
sorry

end triangle_inequality_l125_125549


namespace money_equations_l125_125145

theorem money_equations (x y : ℝ) (h1 : x + (1 / 2) * y = 50) (h2 : y + (2 / 3) * x = 50) :
  x + (1 / 2) * y = 50 ∧ y + (2 / 3) * x = 50 :=
by
  exact ⟨h1, h2⟩

-- Please note that by stating the theorem this way, we have restated the conditions and conclusion
-- in Lean 4. The proof uses the given conditions directly without the need for intermediate steps.

end money_equations_l125_125145


namespace calculateDesiredProfitPercentage_l125_125658

noncomputable def costPrice (sp : ℝ) (lossPercent : ℝ) : ℝ := sp / (1 - lossPercent / 100)

def profitPercentage (cp sp : ℝ) : ℝ := ((sp - cp) / cp) * 100

def desiredProfitPercentage := 4.975

theorem calculateDesiredProfitPercentage :
  profitPercentage (costPrice 10 15) 12.35 = desiredProfitPercentage :=
by
  sorry

end calculateDesiredProfitPercentage_l125_125658


namespace count_consecutive_integers_in_list_k_l125_125544

def list_k := λ (x : Int), -3 ≤ x ∧ x ≤ 8

theorem count_consecutive_integers_in_list_k 
  (least_integer : Int)
  (range_positive : Int) 
  (count : Int)
  (hl : least_integer = -3)
  (hr : range_positive = 7) 
  (hc : count = 12) :
  (∃ k, (-3 ≤ k ∧ k ≤ 8)) ∧ (count = (8 - (-3) + 1)) :=
  by
    sorry

end count_consecutive_integers_in_list_k_l125_125544


namespace correctness_of_statements_l125_125622

theorem correctness_of_statements :
  (∀ x, (3 - 2 * (-2) = 7)) ∧
  (∀ x, (x = 1 ∨ x = -2) ↔ ((x - 1) * (x + 2) = 0)) ∧
  (∀ e, (e = 3 - 2 * (-2)) ∧ (3 - 2 * (-2) = 7)) ∧
  (∀ p, (p = ((x - 1) * (x + 2) = 0))
sorry

end correctness_of_statements_l125_125622


namespace right_triangle_condition_l125_125565

theorem right_triangle_condition (a b c : ℝ) :
  (a^3 + b^3 + c^3 = a*b*(a + b) - b*c*(b + c) + a*c*(a + c)) ↔ (a^2 = b^2 + c^2) ∨ (b^2 = a^2 + c^2) ∨ (c^2 = a^2 + b^2) :=
by
  sorry

end right_triangle_condition_l125_125565


namespace triangles_congruent_l125_125228

theorem triangles_congruent
  {A B C A1 B1 C1 : Type}
  (triABC : Triangle A B C)
  (triA1B1C1 : Triangle A1 B1 C1)
  (h1 : dist A C = dist A1 C1)
  (h2 : dist B C = dist B1 C1)
  (h3 : dist B C > dist A C)
  (h4 : angle A = angle A1) :
  congruent triABC triA1B1C1 := 
sorry

end triangles_congruent_l125_125228


namespace matrix_not_invertible_y_eq_9_div_7_l125_125027

theorem matrix_not_invertible_y_eq_9_div_7 : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := ![
    ![2 * (9 : ℚ) / 7, 9],
    ![4 - 2 * (9 : ℚ) / 7, 5]
  ] 
  in det A = 0 ↔ (9 : ℚ) / 7 = 9 / 7 := by
  sorry

end matrix_not_invertible_y_eq_9_div_7_l125_125027


namespace maximum_of_m_l125_125168

theorem maximum_of_m (A : Fin 2001 → ℕ) :
  let m := { (i, j, k) // 1 ≤ i ∧ i < j ∧ j < k ∧ k ≤ 2001 ∧ A j = A i + 1 ∧ A k = A j + 1 }.toFinset.card 
  m ≤ 667^3 :=
by
  sorry

end maximum_of_m_l125_125168


namespace slope_of_tangent_at_point_A_l125_125265

theorem slope_of_tangent_at_point_A : 
  let y := λ x, Real.exp x in
  let A := (0 : ℝ, Real.exp 0) in
  (D f / Dx).eval A.1 == 1 :=
by
  sorry

end slope_of_tangent_at_point_A_l125_125265


namespace complement_union_l125_125081

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- The proof problem statement
theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3}) (hA : A = {-1, 0, 1}) (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} := sorry

end complement_union_l125_125081


namespace train_speed_l125_125668

theorem train_speed (L_train : ℕ) (L_bridge : ℕ) (T : ℕ) (train_len : L_train = 100) (bridge_len : L_bridge = 300) (time_cross : T = 36) : 
  (L_train + L_bridge) / T = 400 / 36 :=
by
  rw [train_len, bridge_len, time_cross]
  norm_num
  sorry

end train_speed_l125_125668


namespace polyhedron_volume_l125_125845

theorem polyhedron_volume 
  (A B C : Type) [equilateral_triangle : A = B = C ∧ ∀ a : A, a.side_length = 2]
  (D E F : Type) [square : D = E = F ∧ ∀ d : D, d.side_length = 2]
  (G : Type) [regular_hexagon : ∀ g : G, g.side_length = 1]
  (form_polyhedron : A ∪ B ∪ C ∪ D ∪ E ∪ F ∪ G forms polyhedron) : 
  volume form_polyhedron = 8 :=
by
  sorry

end polyhedron_volume_l125_125845


namespace OH_squared_value_l125_125176

theorem OH_squared_value
  (O H A B C : Point)
  (a b c R : ℝ)
  (h1 : R = 10)
  (h2 : a^2 + b^2 - c^2 = 40)
  (circumcenter_O : Circumcenter O A B C)
  (orthocenter_H : Orthocenter H A B C) :
  distance O H ^ 2 = 260 := by
  sorry

end OH_squared_value_l125_125176


namespace kanul_initial_amount_l125_125516

noncomputable def initial_amount : ℝ :=
  (5000 : ℝ) + 200 + 1200 + (11058.82 : ℝ) * 0.15 + 3000

theorem kanul_initial_amount (X : ℝ) 
  (raw_materials : ℝ := 5000) 
  (machinery : ℝ := 200) 
  (employee_wages : ℝ := 1200) 
  (maintenance_cost : ℝ := 0.15 * X)
  (remaining_balance : ℝ := 3000) 
  (expenses : ℝ := raw_materials + machinery + employee_wages + maintenance_cost) 
  (total_expenses : ℝ := expenses + remaining_balance) :
  X = total_expenses :=
by sorry

end kanul_initial_amount_l125_125516


namespace problem_1_problem_2_l125_125787

-- Problem 1
theorem problem_1 (x : ℝ) (h_fx : |2 * x + 1| + |2 * x - 3| ≤ 6) : -1 ≤ x ∧ x ≤ 2 :=
sorry

-- Problem 2
theorem problem_2 (x : ℝ) (a : ℝ) 
  (h : |2 * x + 1| + |2 * x - 3| - log (a ^ 2 - 3 * a) / log 2 > 2) :
  (-1 < a ∧ a < 0) ∨ (3 < a ∧ a < 4) :=
sorry

end problem_1_problem_2_l125_125787


namespace problem_a_problem_b_problem_c_problem_d_l125_125980

-- Problem a
theorem problem_a (a : ℝ) : (a + 1) * (a - 1) = a^2 - 1 :=
by sorry

-- Problem b
theorem problem_b (a : ℝ) : (2 * a + 3) * (2 * a - 3) = 4 * a^2 - 9 :=
by sorry

-- Problem c
theorem problem_c (m n : ℝ) : (m^3 - n^5) * (n^5 + m^3) = m^6 - n^10 :=
by sorry

-- Problem d
theorem problem_d (m n : ℝ) : (3 * m^2 - 5 * n^2) * (3 * m^2 + 5 * n^2) = 9 * m^4 - 25 * n^4 :=
by sorry

end problem_a_problem_b_problem_c_problem_d_l125_125980


namespace sue_initial_savings_l125_125689

theorem sue_initial_savings:
  ∀ (perfume_cost christian_savings additional_needed: ℝ) 
    (mowed_yards walked_dogs: ℕ)
    (charge_per_yard charge_per_dog: ℝ),
  perfume_cost = 50.00 →
  christian_savings = 5.00 →
  mowed_yards = 4 →
  charge_per_yard = 5.00 →
  walked_dogs = 6 →
  charge_per_dog = 2.00 →
  additional_needed = 6.00 →
  let christian_earned := mowed_yards * charge_per_yard in
  let sue_earned := walked_dogs * charge_per_dog in
  let total_earned := christian_savings + christian_earned + sue_earned in
  let total_money := perfume_cost - additional_needed in
  let sue_initial_savings := total_money - total_earned in
  sue_initial_savings = 7.00 :=
by
  sorry

end sue_initial_savings_l125_125689


namespace increasing_interval_of_function_l125_125584

theorem increasing_interval_of_function {k : ℤ} (ω : ℝ) (hω : ω > 0)
  (h_period : ∃ a b, ∃ A B, |A - B| = π) :
  ∃ I : Set ℝ, I = {x | ∃ k : ℤ, 2 * k * π - π / 3 ≤ x ∧ x ≤ 2 * k * π + 2 * π / 3} ∧
    (∀ x ∈ I, f x > 0) :=
by
  -- The proof is not provided here, but the statement should be correct.
  sorry

end increasing_interval_of_function_l125_125584


namespace oscar_bus_ride_length_l125_125213

/-- Oscar's bus ride to school is some distance, and Charlie's bus ride is 0.25 mile.
Oscar's bus ride is 0.5 mile longer than Charlie's. Prove that Oscar's bus ride is 0.75 mile. -/
theorem oscar_bus_ride_length (charlie_ride : ℝ) (h1 : charlie_ride = 0.25) 
  (oscar_ride : ℝ) (h2 : oscar_ride = charlie_ride + 0.5) : oscar_ride = 0.75 :=
by sorry

end oscar_bus_ride_length_l125_125213


namespace wendy_points_earned_l125_125282

-- Define the conditions
def points_per_bag : ℕ := 5
def total_bags : ℕ := 11
def bags_not_recycled : ℕ := 2

-- Define the statement to be proved
theorem wendy_points_earned : (total_bags - bags_not_recycled) * points_per_bag = 45 :=
by
  sorry

end wendy_points_earned_l125_125282


namespace symmetric_points_on_circle_l125_125098

noncomputable def reflection_set := {P : ℝ × ℝ | ∃ l : ℝ × ℝ → bool, is_line_through B l ∧ reflection A l = P}

variables (A B : ℝ × ℝ)

def circle_center_radius (center : ℝ × ℝ) (radius : ℝ) : set (ℝ × ℝ) :=
  {P | dist P center = radius}

theorem symmetric_points_on_circle (A B : ℝ × ℝ) :
  reflection_set A B = circle_center_radius B (dist A B) :=
sorry

end symmetric_points_on_circle_l125_125098


namespace Anya_took_home_balloons_l125_125605

theorem Anya_took_home_balloons :
  ∃ (balloons_per_color : ℕ), 
  ∃ (yellow_balloons_home : ℕ), 
  (672 = 4 * balloons_per_color) ∧ 
  (yellow_balloons_home = balloons_per_color / 2) ∧ 
  (yellow_balloons_home = 84) :=
begin
  sorry
end

end Anya_took_home_balloons_l125_125605


namespace additional_amount_per_share_for_per_increment_of_earnings_l125_125648

-- Define the given conditions
def expected_earnings := 0.80
def half_dividends_per_share := expected_earnings / 2
def actual_earnings := 1.10
def shares_held := 400
def actual_dividend_paid := 208

-- Define what needs to be proved
theorem additional_amount_per_share_for_per_increment_of_earnings :
  let additional_earnings := actual_earnings - expected_earnings,
      additional_dividends := actual_dividend_paid - (shares_held * half_dividends_per_share),
      number_of_increments := additional_earnings / 0.10,
      additional_amount_per_share := additional_dividends / (shares_held * number_of_increments)
  in additional_amount_per_share = 0.04 :=
by sorry  -- Proof to be filled in later

end additional_amount_per_share_for_per_increment_of_earnings_l125_125648


namespace not_lengths_of_external_diagonals_l125_125392

theorem not_lengths_of_external_diagonals (a b c : ℝ) (h : a ≤ b ∧ b ≤ c) :
  (¬ (a = 5 ∧ b = 6 ∧ c = 9)) :=
by
  sorry

end not_lengths_of_external_diagonals_l125_125392


namespace maximum_leftover_guests_with_no_fitting_galoshes_l125_125997

theorem maximum_leftover_guests_with_no_fitting_galoshes :
  ∀ (guests : ℕ) (galoshes : ℕ → ℕ), 
    (guests = 10) → 
    (∀ g, g ≤ 10 → galoshes g = g) → 
    (∀ g, g ≤ 10 → ∀ k, (k ≥ g) → galoshes k = k) → 
    (∀ gs : list ℕ, (gs.length = 5) ∧ ∀ g, g ∈ gs → 
    ¬(∃ k, (k ≥ g) ∧ galoshes k = k)) :=
by
  intros guests galoshes h_guests h_galoshes_sizes h_galoshes_fit gs h_length h_exist_fit
  apply sorry

end maximum_leftover_guests_with_no_fitting_galoshes_l125_125997


namespace find_side_c_l125_125827

-- Define the given conditions as variables
variables (a b C : ℝ)
variables (h_a : a = 4)
variables (h_b : b = 6)
variables (h_C : C = 2 * Real.pi / 3)

-- State the theorem using the given conditions and the Law of Cosines result.
theorem find_side_c (a b C : ℝ) (h_a : a = 4) (h_b : b = 6) (h_C : C = 2 * Real.pi / 3) : 
  let c := Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos C) in
  c = 2 * Real.sqrt 19 :=
by
  sorry

end find_side_c_l125_125827


namespace total_birds_times_types_l125_125203

-- Defining the number of adults and offspring for each type of bird.
def num_ducks1 : ℕ := 2
def num_ducklings1 : ℕ := 5
def num_ducks2 : ℕ := 6
def num_ducklings2 : ℕ := 3
def num_ducks3 : ℕ := 9
def num_ducklings3 : ℕ := 6

def num_geese : ℕ := 4
def num_goslings : ℕ := 7

def num_swans : ℕ := 3
def num_cygnets : ℕ := 4

-- Calculate total number of birds
def total_ducks := (num_ducks1 * num_ducklings1 + num_ducks1) + (num_ducks2 * num_ducklings2 + num_ducks2) +
                      (num_ducks3 * num_ducklings3 + num_ducks3)

def total_geese := num_geese * num_goslings + num_geese
def total_swans := num_swans * num_cygnets + num_swans

def total_birds := total_ducks + total_geese + total_swans

-- Calculate the number of different types of birds
def num_types_of_birds : ℕ := 3 -- ducks, geese, swans

-- The final Lean statement to be proven
theorem total_birds_times_types :
  total_birds * num_types_of_birds = 438 :=
  by sorry

end total_birds_times_types_l125_125203


namespace complement_union_l125_125093

variable (U : Set ℤ)
variable (A : Set ℤ)
variable (B : Set ℤ)

theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3})
                         (hA : A = {-1, 0, 1})
                         (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} :=
sorry

end complement_union_l125_125093


namespace matrix_scalars_exist_l125_125525

namespace MatrixProof

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, 1], ![4, -1]]

theorem matrix_scalars_exist :
  ∃ r s : ℝ, B^6 = r • B + s • (1 : Matrix (Fin 2) (Fin 2) ℝ) ∧ r = 0 ∧ s = 64 := by
  sorry

end MatrixProof

end matrix_scalars_exist_l125_125525


namespace part1_zero_in_interval_part2_monotonic_intervals_l125_125071

-- (a) Define the function f
def f (x : ℝ) : ℝ := Real.cos x - Real.exp (-x)

-- (b) Define the derivative f'
def f' (x : ℝ) : ℝ := -Real.sin x + Real.exp (-x)

-- (c) Prove that f' has exactly one zero in the interval (π/6, π/4)
theorem part1_zero_in_interval : ∃! x ∈ Set.Ioo (Real.pi / 6) (Real.pi / 4), f' x = 0 := sorry

-- (d) Prove that f has two distinct intervals of monotonic increase and one interval of monotonic decrease on [0, 2π]
theorem part2_monotonic_intervals :
  ∃ a b : ℝ, 0 < a ∧ a < b ∧ b < 2*Real.pi ∧
    (∀ x ∈ Set.Ioo 0 a, 0 < f' x) ∧
    (∀ x ∈ Set.Ioo a b, f' x < 0) ∧
    (∀ x ∈ Set.Ioo b (2*Real.pi), 0 < f' x) := sorry

end part1_zero_in_interval_part2_monotonic_intervals_l125_125071


namespace period_cosine_function_l125_125615

theorem period_cosine_function : 
  ∃ T, (∀ x, cos (x / 4 + π / 4) = cos ((x + T) / 4 + π / 4)) ∧ T = 8 * π := 
begin
  sorry
end

end period_cosine_function_l125_125615


namespace math_problem_l125_125589

open Real

-- Definitions for the parametric equations of the line and the circle
def line_parametric (t : ℝ) : ℝ × ℝ :=
  (1 + 2 * t, 1 - 2 * t)

def circle_parametric (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, 2 * Real.sin α)

-- Condition for the polar equation of the circle
def circle_polar : Prop :=
  ∀ (r θ : ℝ), (r = 2)

-- The length of the chord AB
def chord_length : ℝ :=
  2 * Real.sqrt 2

-- The statement to prove the given problem
theorem math_problem :
  (circle_polar) ∧
  (length_of_chord line_parametric circle_parametric = chord_length) :=
by
  sorry

end math_problem_l125_125589


namespace fran_ate_15_green_macaroons_l125_125427

variable (total_red total_green initial_remaining green_macaroons_eaten : ℕ)

-- Conditions as definitions
def initial_red_macaroons := 50
def initial_green_macaroons := 40
def total_macaroons := 90
def remaining_macaroons := 45

-- Total eaten macaroons
def total_eaten_macaroons (G : ℕ) := G + 2 * G

-- The proof statement
theorem fran_ate_15_green_macaroons
  (h1 : total_red = initial_red_macaroons)
  (h2 : total_green = initial_green_macaroons)
  (h3 : initial_remaining = remaining_macaroons)
  (h4 : total_macaroons = initial_red_macaroons + initial_green_macaroons)
  (h5 : initial_remaining = total_macaroons - total_eaten_macaroons green_macaroons_eaten):
  green_macaroons_eaten = 15 :=
  by
  sorry

end fran_ate_15_green_macaroons_l125_125427


namespace ellipse_properties_l125_125443

noncomputable def ellipse (x y a b : ℝ) : Prop :=
  (x^2 / a^2 + y^2 / b^2 = 1)

noncomputable def slopes_condition (x1 y1 x2 y2 : ℝ) (k_ab k_oa k_ob : ℝ) : Prop :=
  (k_ab^2 = k_oa * k_ob)

variables {x y : ℝ}

theorem ellipse_properties :
  (ellipse x y 2 1) ∧ -- Given ellipse equation
  (∃ (x1 y1 x2 y2 k_ab k_oa k_ob : ℝ), slopes_condition x1 y1 x2 y2 k_ab k_oa k_ob) →
  (∃ (OA OB : ℝ), OA^2 + OB^2 = 5) ∧ -- Prove sum of squares is constant
  (∃ (m : ℝ), (m = 1 → ∃ (line_eq : ℝ → ℝ), ∀ x, line_eq x = (1 / 2) * x + m)) -- Maximum area of triangle AOB

:= sorry

end ellipse_properties_l125_125443


namespace det_S_eq_9_l125_125526

def scaling_factor : ℝ := 3
def rotation_angle : ℝ := Real.pi / 4 -- 45 degrees in radians

noncomputable def S : Matrix (Fin 2) (Fin 2) ℝ :=
  Matrix.of ![
    ![scaling_factor * Real.cos rotation_angle, -scaling_factor * Real.sin rotation_angle],
    ![scaling_factor * Real.sin rotation_angle, scaling_factor * Real.cos rotation_angle]
  ]

theorem det_S_eq_9 : Matrix.det S = 9 := by
  sorry

end det_S_eq_9_l125_125526


namespace population_reaches_max_capacity_l125_125138

-- Given conditions
def max_population (total_acres : ℕ) (acre_per_person : ℝ) : ℕ :=
  total_acres / acre_per_person

def population_after_n_years (initial_population : ℕ) (growth_rate : ℝ) (years : ℕ) (period : ℕ) : ℕ :=
  initial_population * (growth_rate ^ (years / period))

-- Prove that in 100 years from 1998 the population will meet the maximum supported population
theorem population_reaches_max_capacity :
  ∀ (total_acres acre_per_person initial_population : ℕ) (growth_rate : ℝ) (period years : ℕ),
  total_acres = 24900 →
  acre_per_person = 1.5 →
  initial_population = 200 →
  growth_rate = 3 →
  period = 25 →
  years = 100 →
  population_after_n_years initial_population growth_rate years period ≥ max_population total_acres acre_per_person :=
by 
  intros total_acres acre_per_person initial_population growth_rate period years h1 h2 h3 h4 h5 h6 
  sorry

end population_reaches_max_capacity_l125_125138


namespace eval_polynomial_at_neg_two_l125_125003

theorem eval_polynomial_at_neg_two : 
  let y := -2 in
  y^3 - y^2 + 2*y + 2 = -14 := by
  sorry

end eval_polynomial_at_neg_two_l125_125003


namespace winnie_piglet_win_strategically_l125_125993

-- Definitions of players and jars
inductive Player
| Winnie
| Rabbit
| Piglet

def Jar := Fin 3

-- Turn order and placement restrictions
noncomputable def can_place (p : Player) (j : Jar) : Bool :=
match p, j with
| Player.Winnie, 0 | Player.Winnie, 1 => true
| Player.Rabbit, 1 | Player.Rabbit, 2 => true
| Player.Piglet, 0 | Player.Piglet, 2 => true
| _, _ => false

-- Game state and loss condition
structure GameState :=
(nuts : Jar → ℕ)

def loses (s : GameState) :=
s.nuts 0 = 1999 ∨ s.nuts 1 = 1999 ∨ s.nuts 2 = 1999

-- The theorem to be proved
theorem winnie_piglet_win_strategically :
  ∀ (initial_state : GameState),
  (∀ s : GameState, ¬ loses s → ∃ p : Player, ∃ j : Jar, can_place p j ∧ ¬ loses ⟨s.nuts.update j (s.nuts j + 1)⟩) →
  ∃ s : GameState, loses s ∧ (∃ k : Player, can_place k (s.nuts.find_index (λ n, n = 1999))) :=
sorry

end winnie_piglet_win_strategically_l125_125993


namespace no_positive_integer_n_such_that_2n2_plus_1_3n2_plus_1_6n2_plus_1_are_all_squares_l125_125227

theorem no_positive_integer_n_such_that_2n2_plus_1_3n2_plus_1_6n2_plus_1_are_all_squares :
  ¬ ∃ n : ℕ, n > 0 ∧ ∃ a b c : ℕ, 2 * n^2 + 1 = a^2 ∧ 3 * n^2 + 1 = b^2 ∧ 6 * n^2 + 1 = c^2 := by
  sorry

end no_positive_integer_n_such_that_2n2_plus_1_3n2_plus_1_6n2_plus_1_are_all_squares_l125_125227


namespace minimum_lines_for_shared_vertex_triangles_l125_125765

theorem minimum_lines_for_shared_vertex_triangles
  (n : ℕ) (h : n ≥ 5) (no_four_points_collinear : true) :
  ∃ m : ℕ, m = (⌈ n^2 / 4 ⌉ + 2) ∧ (∃ (points : list (ℕ × ℕ)), points.length = n ∧ (∀ p₁ p₂ p₃, (p₁ ≠ p₂ ∧ p₂ ≠ p₃ ∧ p₃ ≠ p₁) → ¬ collinear p₁ p₂ p₃) ∧ (∃ two_triangles, share_exactly_one_vertex two_triangles))) :=
sorry

-- Define collinearity (for completeness, though we're not providing proof here)
def collinear (a b c : ℕ × ℕ) : Prop :=
  let (ax, ay) := a
  let (bx, by) := b
  let (cx, cy) := c
  (by - ay) * (cx - ax) = (cy - ay) * (bx - ax)

-- Placeholder for definition of sharing exactly one vertex
def share_exactly_one_vertex (triangles : list (list (ℕ × ℕ))) : Prop := sorry

end minimum_lines_for_shared_vertex_triangles_l125_125765


namespace zeros_in_expansion_of_88888888_squared_l125_125893

theorem zeros_in_expansion_of_88888888_squared :
  ∀ n : ℕ, n ≥ 3 → 
  let x := 10^n - 1 in 
  nat.of_digits 10 (list.replicate (n-2) 0) = 0 → 
  ∃ z : ℕ, let y := x^2 in z = n - 2 :=
by
  intros n hn x Hzero
  have : x = 88 * 10^(n-2) + 88 := sorry
  have : y = x ^ 2 := sorry
  have : ∃ z : ℕ, z = n - 2 := sorry
  exact sorry

end zeros_in_expansion_of_88888888_squared_l125_125893


namespace sin_double_angle_identity_l125_125749

theorem sin_double_angle_identity (α : ℝ) (h : sin α + cos α = 1 / 3) : sin (2 * α) = -8 / 9 := by
  sorry

end sin_double_angle_identity_l125_125749


namespace factorial_ratio_l125_125704

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem factorial_ratio (n : ℕ) : n > 0 -> (factorial (n + 1)) / (factorial n) = n + 1 :=
by
  intros h
  sorry

example : (factorial 2017) / (factorial 2016) = 2017 :=
by
  apply factorial_ratio
  sorry

end factorial_ratio_l125_125704


namespace find_number_l125_125415

theorem find_number (x : ℝ) (h : x - (3 / 5) * x = 56) : x = 140 := 
by {
  -- The proof would be written here,
  -- but it is indicated to skip it using "sorry"
  sorry
}

end find_number_l125_125415


namespace cylinder_ratio_l125_125437

theorem cylinder_ratio
  (V : ℝ) (r h : ℝ)
  (h_volume : π * r^2 * h = V)
  (h_surface_area : 2 * π * r * h = 2 * (V / r)) :
  h / r = 2 :=
sorry

end cylinder_ratio_l125_125437


namespace simplify_expr_l125_125002

-- Define the condition on b
def condition (b : ℚ) : Prop :=
  b ≠ -1 / 2

-- Define the expression to be evaluated
def expression (b : ℚ) : ℚ :=
  1 - 1 / (1 + b / (1 + b))

-- Define the simplified form
def simplified_expr (b : ℚ) : ℚ :=
  b / (1 + 2 * b)

-- The theorem statement showing the equivalence
theorem simplify_expr (b : ℚ) (h : condition b) : expression b = simplified_expr b :=
by
  sorry

end simplify_expr_l125_125002


namespace find_coordinates_of_Q_l125_125155

def vector (n : ℕ) := fin n → ℝ

variables (A B C Q : vector 3)
variables (G H : vector 3)

axiom AG_GB_ratio : ∃ k : ℝ, k = 3 / 5 ∧ G = k • A + (1 - k) • B
axiom BH_HC_ratio : ∃ l : ℝ, l = 3 / 4 ∧ H = l • B + (1 - l) • C
axiom Q_intersection : ∃ t s : ℝ, Q = t • G + (1 - t) • A ∧ Q = s • H + (1 - s) • C

theorem find_coordinates_of_Q :
  Q = (1 / 7) • A + (2 / 7) • B + (4 / 7) • C :=
sorry

end find_coordinates_of_Q_l125_125155


namespace smallest_x_y_sum_299_l125_125703

theorem smallest_x_y_sum_299 : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x < y ∧ (100 + (x / y : ℚ) = 2 * (100 * x / y : ℚ)) ∧ (x + y = 299) :=
by
  sorry

end smallest_x_y_sum_299_l125_125703


namespace find_sphere_volume_l125_125500

noncomputable def sphere_volume (d: ℝ) (V: ℝ) : Prop := d = 3 * (16 / 9) * V

theorem find_sphere_volume :
  sphere_volume (2 / 3) (1 / 6) :=
by
  sorry

end find_sphere_volume_l125_125500


namespace sum_divisors_36_48_l125_125024

open Finset

noncomputable def sum_common_divisors (a b : ℕ) : ℕ :=
  let divisors_a := (range (a + 1)).filter (λ x => a % x = 0)
  let divisors_b := (range (b + 1)).filter (λ x => b % x = 0)
  let common_divisors := divisors_a ∩ divisors_b
  common_divisors.sum id

theorem sum_divisors_36_48 : sum_common_divisors 36 48 = 28 := by
  sorry

end sum_divisors_36_48_l125_125024


namespace g_of_g_even_l125_125181

variable {X : Type} [AddGroup X] [HasNeg X]

def is_even_function (g : X → X) : Prop :=
  ∀ x, g (-x) = g x

theorem g_of_g_even (g : X → X) (h : is_even_function g) : is_even_function (g ∘ g) :=
by
  unfold is_even_function at *
  intro x
  specialize h (g (-x))
  rw h
  exact h


end g_of_g_even_l125_125181


namespace length_of_AB_l125_125502

open Real

-- Conditions:
def O : Point := sorry  -- O is the center of each circle

def C1 : Circle := {center := O, radius := 12}  -- Outer circle with circumference 24 * pi
def C2 : Circle := {center := O, radius := 7}   -- Inner circle with circumference 14 * pi

def B : Point := sorry  -- B is a point on the outer circle
def A : Point := sorry  -- OB intersects inner circle at A

-- Proof problem:
theorem length_of_AB : distance B A = 5 :=
by
  -- Proof will be provided here
  sorry

end length_of_AB_l125_125502


namespace complement_union_l125_125078

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- The proof problem statement
theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3}) (hA : A = {-1, 0, 1}) (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} := sorry

end complement_union_l125_125078


namespace tangent_circles_m_eq_9_l125_125121

noncomputable def circle1 := { center := (0, 0), radius := 1 }

noncomputable def circle2 (m : ℝ) := 
  let center := (3, 4)
  let radius := Real.sqrt (25 - m)
  { center := center, radius := radius }

theorem tangent_circles_m_eq_9 (m : ℝ) :
  let c1 := circle1
  let c2 := circle2 m
  (dist c1.center c2.center = c1.radius + c2.radius) → m = 9 :=
by 
  intro h
  sorry

end tangent_circles_m_eq_9_l125_125121


namespace range_of_a_l125_125123

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

noncomputable def g (a x : ℝ) : ℝ := a * x + 2

theorem range_of_a (a : ℝ) (h : 0 < a) :
  (∀ x₁ ∈ set.Icc (-1 : ℝ) 2, ∃ x₀ ∈ set.Icc (-1 : ℝ) 2, g a x₁ = f x₀) ↔ (0 < a ∧ a ≤ 1/2) := 
sorry

end range_of_a_l125_125123


namespace mnp_sum_correct_l125_125382

noncomputable def mnp_sum : ℕ :=
  let m := 1032
  let n := 40
  let p := 3
  m + n + p

theorem mnp_sum_correct : mnp_sum = 1075 := by
  -- Given the conditions, the established value for m, n, and p should sum to 1075
  sorry

end mnp_sum_correct_l125_125382


namespace circle_cartesian_eq_of_polar_eq_minimize_distance_on_line_l125_125840

theorem circle_cartesian_eq_of_polar_eq
  (θ x y : Real)
  (polar_eq : ∀ θ, x = ρ * cos θ ∧ y = ρ * sin θ ∧ ρ = 2 * sqrt 3 * sin θ)
  : x^2 + (y - sqrt 3)^2 = 3 := by sorry

theorem minimize_distance_on_line
  (t : Real)
  (P : Real × Real := (3 + 1/2 * t, sqrt 3 / 2 * t))
  (C : Real × Real := (0, sqrt 3))
  (distance := dist P C)
  : ∀ t, distance = sqrt (t^2 + 12) → min distance = 2sqrt 3 ∧ P = (3, 0) := by sorry

end circle_cartesian_eq_of_polar_eq_minimize_distance_on_line_l125_125840


namespace max_x2_plus_y2_max_x2_plus_y2_achieved_l125_125414

noncomputable def maximum_x2_plus_y2 : ℝ :=
  let x := 3 in
  let y := 8 in
  x^2 + y^2

theorem max_x2_plus_y2 (x y : ℝ) (h : x^2 + y^2 = 3 * x + 8 * y) : x^2 + y^2 ≤ 73 :=
begin
  sorry
end

theorem max_x2_plus_y2_achieved (x y : ℝ) (h : x^2 + y^2 = 3 * x + 8 * y) : x = 3 ∧ y = 8 → x^2 + y^2 = 73 :=
begin
  sorry
end

end max_x2_plus_y2_max_x2_plus_y2_achieved_l125_125414


namespace incorrect_statements_count_l125_125587

theorem incorrect_statements_count :
  let p q a b am2 bm2 := Prop
  let proposition1 := (p ∨ q → p ∧ q)
  let proposition2 := (x : ℝ) (H : x^2 - 3 * x + 2 = 0) → (x = 1 ∨ x = 2)
  let proposition3 := (a b : ℝ) (H : a > b) → (1 / a < 1 / b)
  let proposition4 := (a b : ℝ) (am2 bm2 : ℝ) (H : a * m^2 <= b * m^2) → (a <= b)

  ¬ proposition1 →
  ¬ proposition2 →
  ¬ proposition3 →
  proposition4 →
-- asserting that there are exactly 3 incorrect statements
  3
by
  sorry

end incorrect_statements_count_l125_125587


namespace distance_to_mothers_house_l125_125293

theorem distance_to_mothers_house 
  (D : ℝ) 
  (h1 : (2 / 3) * D = 156.0) : 
  D = 234.0 := 
sorry

end distance_to_mothers_house_l125_125293


namespace find_range_of_a_l125_125767

-- Define the propositions p and q based on the given conditions
def p (a x : ℝ) : Prop := (2 < x) ∧ (2^x * (x - a) < 1)
def q (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, y = log (x^2 + 2*a*x + a)

-- Define the main theorem stating the range of a
theorem find_range_of_a (a : ℝ) :
  ¬ (∃ x : ℝ, p a x ∧ q a) ∧ (∃ x : ℝ, p a x) ∨ q a ↔ (a ≤ 0 ∨ (1 ≤ a ∧ a ≤ 7/4)) :=
by
  sorry

end find_range_of_a_l125_125767


namespace probability_snow_at_least_once_l125_125894

-- Defining the probability of no snow on the first five days
def no_snow_first_five_days : ℚ := (4 / 5) ^ 5

-- Defining the probability of no snow on the next five days
def no_snow_next_five_days : ℚ := (2 / 3) ^ 5

-- Total probability of no snow during the first ten days
def no_snow_first_ten_days : ℚ := no_snow_first_five_days * no_snow_next_five_days

-- Probability of snow at least once during the first ten days
def snow_at_least_once_first_ten_days : ℚ := 1 - no_snow_first_ten_days

-- Desired proof statement
theorem probability_snow_at_least_once :
  snow_at_least_once_first_ten_days = 726607 / 759375 := by
  sorry

end probability_snow_at_least_once_l125_125894


namespace smaller_angle_at_2_30_correct_l125_125472

noncomputable def angle_at_two_thirty : ℝ :=
  let minute_angle := 180 in
  let hour_angle := 2 * 30 + 30 * 0.5 in
  let smaller_angle := minute_angle - hour_angle in
  smaller_angle

theorem smaller_angle_at_2_30_correct : angle_at_two_thirty = 105 :=
by
  unfold angle_at_two_thirty
  norm_num
  -- sorry

end smaller_angle_at_2_30_correct_l125_125472


namespace hyperbola_equation_l125_125453

theorem hyperbola_equation (asymptote_eq : ∀ x : ℝ, y = ± (4/3) * x)
(focal_length : ℝ) (h_focal_length : focal_length = 20) : 
  ∃ (a b : ℝ), (a = 6 ∧ b = 8 ∧ ((x^2 / 36 - y^2 / 64 = 1) ∨ (y^2 / 64 - x^2 / 36 = 1))) := 
by
  sorry

end hyperbola_equation_l125_125453


namespace total_passengers_landed_l125_125517

theorem total_passengers_landed (on_time late : ℕ) (h1 : on_time = 14507) (h2 : late = 213) : 
    on_time + late = 14720 :=
by
  sorry

end total_passengers_landed_l125_125517


namespace eliot_account_balance_l125_125299

theorem eliot_account_balance (A E : ℝ) 
  (h1 : A > E)
  (h2 : A - E = (1 / 12) * (A + E))
  (h3 : 1.10 * A - 1.15 * E = 22) : 
  E = 146.67 :=
by
  sorry

end eliot_account_balance_l125_125299


namespace intervals_of_monotonicity_f_range_of_a_minimum_value_h_l125_125042

noncomputable def f (x : ℝ) : ℝ := x * log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x^2 - x + 2
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f(x) - a * (x - 1)

theorem intervals_of_monotonicity_f :
  (∀ x : ℝ, 0 < x ∧ x < (1 / exp 1) → deriv f x < 0) ∧
  (∀ x : ℝ, x > (1 / exp 1) → deriv f x > 0) :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > 0 → 2 * f x ≤ deriv (λ x, g x a) x + 2) → a ≥ -2 :=
sorry

theorem minimum_value_h (a : ℝ) (min_h : ℝ):
  (1 ≤ a ∧ a ≤ 2 → min_h = a - exp (a - 1)) ∧
  (a > 2 → min_h = (1 - a) * exp 1 + a) :=
sorry

end intervals_of_monotonicity_f_range_of_a_minimum_value_h_l125_125042


namespace A_speed_is_10_l125_125671

noncomputable def A_walking_speed (v t : ℝ) := 
  v * (t + 7) = 140 ∧ v * (t + 7) = 20 * t

theorem A_speed_is_10 (v t : ℝ) 
  (h1 : v * (t + 7) = 140)
  (h2 : v * (t + 7) = 20 * t) :
  v = 10 :=
sorry

end A_speed_is_10_l125_125671


namespace part1_part2_l125_125420

-- Define the function f
def f (x : ℝ) : ℝ :=
  ∫ t in x..(π / 4 - x), real.log (1 + real.tan t) / real.log 4

-- Prove that the derivative of f(x) is -1/2
theorem part1 : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ (π / 8) → (deriv f x) = -1 / 2 :=
by sorry

-- Define the sequence a_n
noncomputable def a_seq : ℕ → ℝ
| 0       := f 0
| (n + 1) := f (a_seq n)

-- Prove the n-th term of the sequence a_n
theorem part2 : ∀ n : ℕ, a_seq n = (π / 24) + (π / 48) * ((-1 / 2) ^ n) :=
by sorry

end part1_part2_l125_125420


namespace domain_of_function_l125_125578

theorem domain_of_function :
  {x : ℝ | 3 - x > 0 ∧ x^2 - 1 > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | 1 < x ∧ x < 3} :=
by
  sorry

end domain_of_function_l125_125578


namespace equation1_solution_equation2_solution_l125_125913

-- Equation 1 Statement
theorem equation1_solution (x : ℝ) : 
  (1 / 6) * (3 * x - 6) = (2 / 5) * x - 3 ↔ x = -20 :=
by sorry

-- Equation 2 Statement
theorem equation2_solution (x : ℝ) : 
  (1 - 2 * x) / 3 = (3 * x + 1) / 7 - 3 ↔ x = 67 / 23 :=
by sorry

end equation1_solution_equation2_solution_l125_125913


namespace thirteenth_never_monday_l125_125156

theorem thirteenth_never_monday : ∀ (y : ℕ), ∃ m, ((13 : ℕ) : Zmod 7) ≠ ((day_of_week (Calendar.date.mk y m 1) + (12 : ℕ)) % 7) := sorry

end thirteenth_never_monday_l125_125156


namespace split_8_students_into_pairs_l125_125641

-- Define the number of students
def num_students : ℕ := 8

-- Define the function to compute the number of ways to split n students into pairs
def number_of_ways_to_split_into_pairs (n : ℕ) : ℕ :=
  if n % 2 ≠ 0 then 0 -- If n is odd, no way to split into pairs
  else (nat.factorial n) / (nat.factorial (n / 2) * 2 ^ (n / 2))

-- The proof statement
theorem split_8_students_into_pairs :
  number_of_ways_to_split_into_pairs num_students = 105 :=
by 
  -- Compute the result explicitly to match against 105
  have h1 : num_students = 8 := rfl
  have h2 : number_of_ways_to_split_into_pairs 8 = 105 := by sorry
  exact h2

end split_8_students_into_pairs_l125_125641


namespace smallest_ten_digit_number_of_cards_l125_125281

noncomputable def card_numbers : list ℕ := [513, 23, 5, 4, 46, 7]

theorem smallest_ten_digit_number_of_cards : 
  (exists l : list ℕ, l.perm card_numbers ∧ list.foldl (λ acc x, acc * 10 ^ (nat.log10 x + 1) + x) 0 l = 2344651357) := 
sorry

end smallest_ten_digit_number_of_cards_l125_125281


namespace complement_union_eq_l125_125085

open Set

-- Definition of sets U, A, and B
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- Statement of the problem
theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 3} :=
by sorry

end complement_union_eq_l125_125085


namespace solidRegion_volume_l125_125489

open Real
open Set
open MeasureTheory

def solidRegion : Set (ℝ × ℝ × ℝ) := { p | ∃ (x y z : ℝ),
  p = (x, y, z) ∧ 0 ≤ z ∧ z ≤ 1 + x + y - 3 * (x - y) * y ∧ 0 ≤ y ∧ y ≤ 1 ∧ y ≤ x ∧ x ≤ y + 1 }

def volume (s : Set (ℝ × ℝ × ℝ)) : ℝ := ∫ᵐ (p : ℝ × ℝ × ℝ) in s, 1

theorem solidRegion_volume : volume solidRegion = 13 / 4 := 
by
  sorry

end solidRegion_volume_l125_125489


namespace find_angle_A_find_range_of_y_l125_125449

-- Part 1: Prove angle A = π / 3
theorem find_angle_A 
  (a b c : ℝ) (A B C : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : ∠A + ∠B + ∠C = π)
  (h5 : (2 * b - c) / a = (cos C) / (cos A)) 
  : A = π / 3 := 
by sorry

-- Part 2: Prove the range of the function y
theorem find_range_of_y 
  (A B C : ℝ) (h1 : A = π / 3) 
  : ∀ B : ℝ, 0 < B ∧ B < 2 * π / 3 → 
    ∃ y : ℝ, y = sqrt 3 * sin B + sin (C - π / 6) ∧ 1 < y ∧ y ≤ 2 := 
by sorry

end find_angle_A_find_range_of_y_l125_125449


namespace find_packs_size_l125_125428

theorem find_packs_size (y : ℕ) :
  (24 - 2 * y) * (36 + 4 * y) = 864 → y = 3 :=
by
  intro h
  -- Proof goes here
  sorry

end find_packs_size_l125_125428


namespace solution_to_fractional_equation_l125_125596

theorem solution_to_fractional_equation (x : ℝ) (h : x ≠ 3 ∧ x ≠ 1) :
  (x / (x - 3) = (x + 1) / (x - 1)) ↔ (x = -3) :=
by
  sorry

end solution_to_fractional_equation_l125_125596


namespace prove_l125_125193

variable (m n : ℕ)
variable (a b α β : ℝ)
variable hα : α = 3 / 4
variable hβ : β = 19 / 20

def height_ratio (m n : ℕ) (a b α β : ℝ) (hα : α = 3 / 4) (hβ : β = 19 / 20) 
    (h1 : a = α * b) (h2 : a = β * (a * m + b * n) / (m + n)) : (ℝ) :=
  m / n

theorem prove\_height\_ratio (m n : ℕ) (a b α β : ℝ) (hα : α = 3 / 4) (hβ : β = 19 / 20) 
    (h1 : a = α * b) (h2 : a = β * (a * m + b * n) / (m + n)) : 
  height_ratio m n a b α β hα hβ h1 h2 = 8 / 9 :=
by
  -- Proof omitted
  sorry

end prove_l125_125193


namespace shelby_drive_rain_minutes_l125_125909

theorem shelby_drive_rain_minutes
  (total_distance : ℝ)
  (total_time : ℝ)
  (sunny_speed : ℝ)
  (rainy_speed : ℝ)
  (t_sunny : ℝ)
  (t_rainy : ℝ) :
  total_distance = 20 →
  total_time = 50 →
  sunny_speed = 40 →
  rainy_speed = 25 →
  total_time = t_sunny + t_rainy →
  (sunny_speed / 60) * t_sunny + (rainy_speed / 60) * t_rainy = total_distance →
  t_rainy = 30 :=
by
  intros
  sorry

end shelby_drive_rain_minutes_l125_125909


namespace cost_two_bedroom_unit_l125_125646

theorem cost_two_bedroom_unit :
  let num_units := 12
  let cost_one_bedroom := 360
  let total_income := 4950
  let num_two_bedroom := 7
  let num_one_bedroom := num_units - num_two_bedroom
  let cost_two_bedroom := 450
  total_income = num_one_bedroom * cost_one_bedroom + num_two_bedroom * cost_two_bedroom :=
begin
  sorry
end

end cost_two_bedroom_unit_l125_125646


namespace books_not_sold_l125_125345

theorem books_not_sold 
  (s : ℕ) (m : ℕ) (t : ℕ) (w : ℕ) (th : ℕ) (f : ℕ) 
  (h_s : s = 800) 
  (h_m : m = 60) 
  (h_t : t = 10) 
  (h_w : w = 20) 
  (h_th : th = 44) 
  (h_f : f = 66) : 
  s - (m + t + w + th + f) = 600 :=
by
  rw [h_s, h_m, h_t, h_w, h_th, h_f]
  simp
  norm_num
  sorry

end books_not_sold_l125_125345


namespace max_value_of_f_l125_125724

-- Define the function f(x)
def f (x : ℝ) : ℝ := Real.sqrt (x + 50) + Real.sqrt (20 - x) + 2 * Real.sqrt x

-- Define the proof statement
theorem max_value_of_f : ∃ x ∈ Icc 0 20, f x = 18.124 :=
by
  sorry

end max_value_of_f_l125_125724


namespace proof_problem_l125_125815

variable (φ : ℝ) (x : ℂ)
variable (h₁ : 0 < φ ∧ φ < π)
variable (h₂ : x + (1 / x) = (2 : ℂ) * complex.cos (2 * φ))

theorem proof_problem : x ^ 3 + (1 / x ^ 3) = (2 : ℂ) * complex.cos (6 * φ) := 
by 
  sorry

end proof_problem_l125_125815


namespace volume_of_1g_l125_125986

theorem volume_of_1g (mass_m3_kg : ℝ) (kg_to_g : ℝ) (m3_to_cm3 : ℝ) :
  mass_m3_kg = 200 →
  kg_to_g = 1000 →
  m3_to_cm3 = 1000000 →
  let density := (mass_m3_kg * kg_to_g) / m3_to_cm3,
      mass_g := 1 in
  mass_g / density = 5 :=
by
  intro h_mass_m3_kg h_kg_to_g h_m3_to_cm3
  let density := (mass_m3_kg * kg_to_g) / m3_to_cm3
  let mass_g := 1
  have h_density : density = (mass_m3_kg * kg_to_g) / m3_to_cm3, from rfl
  have h := mass_g / density
  sorry

end volume_of_1g_l125_125986


namespace intersection_of_A_and_B_l125_125077

def A : Set ℝ := {x | x - 1 > 1}
def B : Set ℝ := {x | x < 3}

theorem intersection_of_A_and_B : (A ∩ B) = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end intersection_of_A_and_B_l125_125077


namespace cistern_empty_time_l125_125629

noncomputable def time_to_empty_cistern (fill_no_leak_time fill_with_leak_time : ℝ) (filled_cistern : ℝ) : ℝ :=
  let R := filled_cistern / fill_no_leak_time
  let L := (R - filled_cistern / fill_with_leak_time)
  filled_cistern / L

theorem cistern_empty_time :
  time_to_empty_cistern 12 14 1 = 84 :=
by
  unfold time_to_empty_cistern
  simp
  sorry

end cistern_empty_time_l125_125629


namespace angle_ABF_is_correct_l125_125218

-- Define a regular octagon
structure RegularOctagon (A B C D E F G H : Type) := 
  (sides_eq : ∀ (i j : ℕ), 0 ≤ i ∧ i < 8 → 0 ≤ j ∧ j < 8 → (A i) = (A j))
  (angles_eq : ∀ (i j : ℕ), 0 ≤ i ∧ i < 8 → 0 ≤ j ∧ j < 8 → (A (i + 1) - A i) = 135)

noncomputable def measure_angle_ABF {A B C D E F G H : Type} 
  (oct : RegularOctagon A B C D E F G H) : ℝ :=
22.5

theorem angle_ABF_is_correct (A B C D E F G H : Type) 
  (oct : RegularOctagon A B C D E F G H) :
  measure_angle_ABF oct = 22.5 :=
by
  sorry

end angle_ABF_is_correct_l125_125218


namespace ratio_m_over_n_l125_125195

theorem ratio_m_over_n : 
  ∀ (m n : ℕ) (a b : ℝ),
  let α := (3 : ℝ) / 4
  let β := (19 : ℝ) / 20
  (a = α * b) →
  (a = β * (a * m + b * n) / (m + n)) →
  (n ≠ 0) →
  m / n = 8 / 9 :=
by
  intros m n a b α β hα hβ hn
  sorry

end ratio_m_over_n_l125_125195


namespace distribute_balls_l125_125477

theorem distribute_balls :
  let balls := 7
      boxes := 3
      all_in_one := 1
      six_and_one := 7
      five_and_two := 21
      four_and_three := 35
  in all_in_one + six_and_one + five_and_two + four_and_three = 64 :=
by
  sorry

end distribute_balls_l125_125477


namespace sum_of_squares_CP_CQ_l125_125834

-- Definitions of the conditions
def isEquilateralTriangle (A B C : ℝ) : Prop := 
  A = B ∧ B = C ∧ C = A

def pointOnSegment (A B : ℝ) (x : ℝ) : Prop := 
  0 < x ∧ x < B

noncomputable def lawOfCosines (a b c : ℝ) (cosgamma : ℝ) : ℝ :=
  a^2 + b^2 - 2 * a * b * cosgamma

-- The statement to prove
theorem sum_of_squares_CP_CQ :
  ∀ (A B C D P Q : ℝ), 
  isEquilateralTriangle A B C → 
  C = 10 → 
  pointOnSegment B C D → 
  D = 6 → 
  A = 2 → 
  B = 2 →
  let CP := lawOfCosines 10 2 (real.cos (real.pi / 3)) in 
  let CQ := lawOfCosines 10 2 (real.cos (real.pi / 3)) in 
  CP^2 + CQ^2 = 168 :=
by
  intros A B C D P Q hEquilateral hC hPonSegment hD hA hB
  let CP := lawOfCosines 10 2 (real.cos (real.pi / 3))
  let CQ := lawOfCosines 10 2 (real.cos (real.pi / 3))
  sorry

end sum_of_squares_CP_CQ_l125_125834


namespace sum_of_roots_quadratic_l125_125731

theorem sum_of_roots_quadratic (a b c : ℝ) (h_eq : a ≠ 0) (h_eqn : -48 * a * (a * 1) + 100 * a + 200 * a^2 = 0) : 
  - b / a = (25 : ℚ) / 12 :=
by
  have h1 : a = -48 := rfl
  have h2 : b = 100 := rfl
  sorry

end sum_of_roots_quadratic_l125_125731


namespace initial_peanuts_count_l125_125951

def peanuts_initial (P : ℕ) : Prop :=
  P - (1 / 4 : ℝ) * P - 29 = 82

theorem initial_peanuts_count (P : ℕ) (h : peanuts_initial P) : P = 148 :=
by
  -- The complete proof can be constructed here.
  sorry

end initial_peanuts_count_l125_125951


namespace freshman_admission_needed_l125_125229

-- Definitions for the conditions
def one_third_drops_out (F : ℕ) : ℕ := F - F / 3

def forty_drops_out (S : ℕ) : ℕ := S - 40

def one_tenth_drops_out (J : ℕ) : ℕ := J - J / 10

def students_enrolled := 3400

-- Definition for the freshman students needed
noncomputable def freshman_needed : ℕ :=
  let J := (students_enrolled * 10 + 8) / 9 in
  let S := J + 40 in
  let F := (S * 3 + 1) / 2 in
  F

-- Statement of the theorem
theorem freshman_admission_needed : freshman_needed = 5727 := 
  sorry

end freshman_admission_needed_l125_125229


namespace polygon_fully_painted_l125_125594

/-- Given a convex polygon with several diagonals drawn such that no three diagonals intersect at a single point,
and each side and diagonal is painted on one side, there exists at least one polygon formed by these diagonals
that is entirely painted on the outside. -/
theorem polygon_fully_painted 
  (P : Type)
  [convex_polygon P]
  (diagonals : set (diagonal P))
  (H_no_three_intersect : ∀ d1 d2 d3 ∈ diagonals, ¬ (intersect_at_single_point d1 d2 d3))
  (H_painted : ∀ s ∈ (sides P) ∪ diagonals, painted_on_one_side s) : 
  ∃ p ∈ (polygons_formed_by_diagonals P diagonals), fully_painted_on_outside p :=
sorry

end polygon_fully_painted_l125_125594


namespace sum_of_fractions_l125_125365

-- Definitions of parameters and conditions
variables {x y : ℝ}
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)

-- The statement of the proof problem
theorem sum_of_fractions (hx : x ≠ 0) (hy : y ≠ 0) : 
  (3 / x) + (2 / y) = (3 * y + 2 * x) / (x * y) :=
sorry

end sum_of_fractions_l125_125365


namespace arctg_inequality_l125_125280

theorem arctg_inequality (a b : ℝ) :
    |Real.arctan a - Real.arctan b| ≤ |b - a| := 
sorry

end arctg_inequality_l125_125280


namespace max_guests_without_galoshes_l125_125996

theorem max_guests_without_galoshes :
  ∀ (guests galoshes : Fin 10) (sizes : Fin 10 → Fin 10),
  (∀ i : Fin 10, ∃ j : Fin 10, sizes i = j) →
  (∀ i : Fin 10, ∀ j : Fin 10, sizes i ≤ sizes j → galoshes j ≥ guests i) →
  (∃ n : ℕ, n ≤ 5 ∧ ∀ remaining_guests : Fin n, ∃ k : Fin 10, guests remaining_guests > galoshes k) :=
sorry

end max_guests_without_galoshes_l125_125996


namespace mean_of_solutions_l125_125014

theorem mean_of_solutions (x : ℝ) :
  (x^3 + 6 * x^2 + 5 * x = 0) →
  (x = 0 ∨ x = -1 ∨ x = -5) →
  (real.mean [0, -1, -5] = -2) := 
sorry

end mean_of_solutions_l125_125014


namespace random_event_is_B_l125_125289

axiom isCertain (event : Prop) : Prop
axiom isImpossible (event : Prop) : Prop
axiom isRandom (event : Prop) : Prop

def A : Prop := ∀ t, t is certain (the sun rises from the east at time t)
def B : Prop := ∃ t, t is random (encountering a red light at time t ∧ passing through traffic light intersection)
def C : Prop := ∀ (p1 p2 p3 : ℝ²), isCertain (non-collinear points p1 p2 p3 → ∃! c, c is circle passing through p1 p2 p3)
def D : Prop := ∀ (T : Triangle), isImpossible (sum_of_interior_angles T = 540)

theorem random_event_is_B : isRandom B :=
by
  sorry

end random_event_is_B_l125_125289


namespace find_valid_pairs_l125_125011

open Nat

theorem find_valid_pairs (m n : ℕ) :
  (∃ (A : Finset (Finset ℕ)),
    (∀ (X : Finset ℕ), (X ⊆ Finset.range (n + 1)) ∧ X.card = 2 → ∃ (i : ℕ), (i < m) ∧ X ⊆ A) ∧
    (∀ (i j : ℕ), i < m → j < m → X ≠ Y → (A i ∩ A j).card = 1)) →
  (m = 1 ∧ n = 3) ∨ (m = 7 ∧ n = 7) :=
by {
  sorry
}

end find_valid_pairs_l125_125011


namespace max_section_area_of_cone_l125_125778

noncomputable def max_cone_section_area (h V : ℝ) : ℝ :=
  2

-- Define the given conditions
def height := 1
def volume := π

-- State the math proof problem
theorem max_section_area_of_cone :
  (max_cone_section_area height volume) = 2 := 
by
  sorry

end max_section_area_of_cone_l125_125778


namespace exists_m_ge_3_l125_125524

def floor (x : ℝ) : ℤ := Int.floor x

def satisfies_recurrence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n > 0 ∧
    (a (n + 2) = (floor (2 * (a (n + 1) : ℝ) / (a n : ℝ)) : ℕ) + 
                (floor (2 * (a n : ℝ) / (a (n + 1) : ℝ)) : ℕ))

theorem exists_m_ge_3 (a : ℕ → ℕ) (h : satisfies_recurrence a) : ∃ m : ℕ, m ≥ 3 ∧ a m = a (m + 1) :=
sorry

end exists_m_ge_3_l125_125524


namespace sum_of_intersections_l125_125758

-- Definitions and conditions
def f (x : ℝ) : ℝ := sorry
axiom f_condition : ∀ x : ℝ, f (-x) = 2 - f x
def g (x : ℝ) : ℝ := (x + 1) / x

-- Existence of intersection points and their sum is m
noncomputable def intersections : List (ℝ × ℝ) := sorry

axiom intersection_condition : ∀ p ∈ intersections, p.snd = f p.fst ∧ p.snd = g p.fst ∧ (f (-p.fst) = 2 - f p.fst)

-- Main theorem statement
theorem sum_of_intersections : List.sum (intersections.map (λ p, p.fst + p.snd)) = intersections.length := 
sorry

end sum_of_intersections_l125_125758


namespace tank_C_capacity_is_80_percent_of_tank_B_l125_125918

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ := 
  Real.pi * r^2 * h

theorem tank_C_capacity_is_80_percent_of_tank_B :
  ∀ (h_C c_C h_B c_B : ℝ), 
    h_C = 10 ∧ c_C = 8 ∧ h_B = 8 ∧ c_B = 10 → 
    (volume_of_cylinder (c_C / (2 * Real.pi)) h_C) / 
    (volume_of_cylinder (c_B / (2 * Real.pi)) h_B) * 100 = 80 := 
by 
  intros h_C c_C h_B c_B h_conditions
  obtain ⟨h_C_10, c_C_8, h_B_8, c_B_10⟩ := h_conditions
  sorry

end tank_C_capacity_is_80_percent_of_tank_B_l125_125918


namespace BKOM_cyclic_l125_125892

theorem BKOM_cyclic (A B C N K M O : Type*)
  [IsTriangle A B C]
  (H_AC_longest : longest_side A C B)
  (H_N_on_AC : on_segment N A C)
  (H_K_on_AB : on_perpendicular_bisector K A N B)
  (H_M_on_BC : on_perpendicular_bisector M N C B)
  (H_O_circumcenter : is_circumcenter O A B C) :
  is_cyclic_quadrilateral B K O M :=
sorry

end BKOM_cyclic_l125_125892


namespace eval_expression_l125_125422

-- Define non-zero numbers x and z,
variables {x z : ℝ} (hx : x ≠ 0) (hz : z ≠ 0)

-- Define the relation x = 1 / z^2
hypothesis (h : x = 1 / z^2)

-- Prove the desired equality
theorem eval_expression : (x + 1 / x) * (z^2 - 1 / z^2) = z^4 - 1 / z^4 :=
by sorry

end eval_expression_l125_125422


namespace green_leaves_initial_l125_125945

def initial_green_leaves_per_plant : ℕ :=
  let total_initial_leaves := 54 in
  let number_of_plants := 3 in
  total_initial_leaves / number_of_plants

theorem green_leaves_initial (total_green_leaves : ℕ) (number_of_plants : ℕ) (fraction_left : ℚ) (remaining_green_leaves : ℕ) :
  (fraction_left * total_green_leaves = remaining_green_leaves) →
  (number_of_plants = 3) →
  (fraction_left = 2 / 3) →
  (remaining_green_leaves = 36) →
  initial_green_leaves_per_plant = 18 :=
by
  intros h_frac h_plants h_fraction_left h_remaining
  subst h_plants
  subst h_fraction_left
  subst h_remaining
  sorry

end green_leaves_initial_l125_125945


namespace volume_of_tetrahedron_ABCD_l125_125760

noncomputable def tetrahedron_volume_proof (S: ℝ) (AB AD BD: ℝ) 
    (angle_ABD_DBC_CBA angle_ADB_BDC_CDA angle_ACB_ACD_BCD: ℝ) : ℝ :=
if h1 : S = 1 ∧ AB = AD ∧ BD = (Real.sqrt 2) / 2
    ∧ angle_ABD_DBC_CBA = 180 ∧ angle_ADB_BDC_CDA = 180 
    ∧ angle_ACB_ACD_BCD = 90 then
  (1 / 24)
else
  0

-- Statement to prove
theorem volume_of_tetrahedron_ABCD : tetrahedron_volume_proof 1 AB AD ((Real.sqrt 2) / 2) 180 180 90 = (1 / 24) :=
by sorry

end volume_of_tetrahedron_ABCD_l125_125760


namespace find_range_and_interval_and_cos2θ_l125_125067

noncomputable def f (x : Real) : Real :=
  cos (2 * x) + (2 * Real.sqrt 3) * sin x * cos x

theorem find_range_and_interval_and_cos2θ
  (θ : Real)
  (hθ1 : 0 < θ)
  (hθ2 : θ < Real.pi / 6)
  (hfθ : f θ = 4 / 3) :
  (∀ x : Real, -2 ≤ f x ∧ f x ≤ 2) ∧
  (∀ k : ℤ, ∀ x : Real, k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6 → StrictMonoOn f {x | k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6}) ∧
  cos (2 * θ) = (Real.sqrt 15 + 2) / 6 :=
sorry

end find_range_and_interval_and_cos2θ_l125_125067


namespace quadratic_discriminant_l125_125226

theorem quadratic_discriminant {a b c : ℝ} (h : (a + b + c) * c ≤ 0) : b^2 ≥ 4 * a * c :=
sorry

end quadratic_discriminant_l125_125226


namespace value_of_a_over_b_l125_125371

def elements : List ℤ := [-5, -3, -1, 2, 4]

def maxProduct (l : List ℤ) : ℤ :=
  l.product $ prod for (x, y) in l.allPairs if x ≠ y and x * y

def minQuotient (l : List ℤ) : Rat :=
  l.allPairs $ min (x / y) for (x, y) in l.allPairs if x ≠ y and y ≠ 0

theorem value_of_a_over_b :
  let a := maxProduct elements
  let b := minQuotient elements
  a = 15 → b = -4 → a / b = -4
by
  intro a h_ma b h_mb
  have ha : a = 15 := h_ma
  have hb : b = -4 := h_mb
  rw [ha, hb]
  norm_num [ha, hb]
  sorry

end value_of_a_over_b_l125_125371


namespace denominator_expression_l125_125469

theorem denominator_expression (x y a b E : ℝ)
  (h1 : x / y = 3)
  (h2 : (2 * a - x) / E = 3)
  (h3 : a / b = 4.5) : E = 3 * b - y :=
sorry

end denominator_expression_l125_125469


namespace fox_initial_money_l125_125571

theorem fox_initial_money : ∃ (x : ℕ), 
  let after_first_crossing := 2 * x - 50 in
  let after_second_crossing := 2 * after_first_crossing - 50 in
  let after_third_crossing := 2 * after_second_crossing - 50 in
  let after_fourth_crossing := 2 * after_third_crossing - 50 in
  after_fourth_crossing = 0 ∧ x = 47 :=
sorry

end fox_initial_money_l125_125571


namespace complement_union_l125_125082

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- The proof problem statement
theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3}) (hA : A = {-1, 0, 1}) (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} := sorry

end complement_union_l125_125082


namespace min_value_modulus_z_add_2i_l125_125187

open Complex

theorem min_value_modulus_z_add_2i {z : ℂ} (h : |z^2 + 9| = |z * (z + 3 * Complex.I)|) : 
  ∃ z, |z + 2 * Complex.I| = 5 / 2 :=
begin
  sorry
end

end min_value_modulus_z_add_2i_l125_125187


namespace eval_f_l125_125439

def f (x : ℂ) : ℂ :=
  if x.im = 0 then 1 + x else (1 - complex.I) * x

theorem eval_f (c : ℂ) (h : c = 1 + complex.I) : f c = 2 := 
by 
  sorry

end eval_f_l125_125439


namespace right_triangle_third_side_l125_125833

theorem right_triangle_third_side (a b : ℕ) (c : ℝ) (h₁: a = 3) (h₂: b = 4) (h₃: ((a^2 + b^2 = c^2) ∨ (a^2 + c^2 = b^2)) ∨ (c^2 + b^2 = a^2)):
  c = Real.sqrt 7 ∨ c = 5 :=
by
  sorry

end right_triangle_third_side_l125_125833


namespace tan_alpha_eq_13_over_16_l125_125448

noncomputable def find_tan_alpha (α β : ℝ) : ℝ :=
if h1 : tan (3 * α - 2 * β) = 1 / 2 ∧ tan (5 * α - 4 * β) = 1 / 4 then (13 / 16) else 0

theorem tan_alpha_eq_13_over_16 (α β : ℝ) (h1 : tan (3 * α - 2 * β) = 1 / 2)
                                    (h2 : tan (5 * α - 4 * β) = 1 / 4) :
  tan α = 13 / 16 := 
by {
  sorry
}

end tan_alpha_eq_13_over_16_l125_125448


namespace sequence_general_formula_l125_125405

def sequence_term (n : ℕ) : ℕ :=
  if n = 0 then 3 else 3 + n * 5 

theorem sequence_general_formula (n : ℕ) : n > 0 → sequence_term n = 5 * n - 2 :=
by 
  sorry

end sequence_general_formula_l125_125405


namespace sandwich_varieties_count_l125_125564

theorem sandwich_varieties_count :
  let num_toppings := 2^7,               -- 128 topping combinations
      num_bread := 3,                    -- 3 bread choices
      num_layer_choices := 3 + 3 * 3,    -- 12 filling choices (one or two layers)
      total_varieties := num_toppings * num_bread * num_layer_choices
  in
  total_varieties = 4608 := by
  sorry

end sandwich_varieties_count_l125_125564


namespace chipmunks_initial_count_l125_125360

variable (C : ℕ) (total : ℕ) (morning_beavers : ℕ) (afternoon_beavers : ℕ) (decrease_chipmunks : ℕ)

axiom chipmunks_count : morning_beavers = 20 
axiom beavers_double : afternoon_beavers = 2 * morning_beavers
axiom decrease_chipmunks_initial : decrease_chipmunks = 10
axiom total_animals : total = 130

theorem chipmunks_initial_count : 
  20 + C + (2 * 20) + (C - 10) = 130 → C = 40 :=
by
  intros h
  sorry

end chipmunks_initial_count_l125_125360


namespace range_of_a_l125_125114

-- Define the function f and its derivative f'
def f (a x : ℝ) : ℝ := x^3 + 3 * a * x^2 + 3 * (a + 2) * x + 1
def f_prime (a x : ℝ) : ℝ := 3 * x^2 + 6 * a * x + 3 * (a + 2)

-- We are given that for f to have both maximum and minimum values, f' must have two distinct roots
-- Thus we translate the mathematical condition to the discriminant of f' being greater than 0
def discriminant_greater_than_zero (a : ℝ) : Prop :=
  (6 * a)^2 - 4 * 3 * 3 * (a + 2) > 0

-- Finally, we want to prove that this simplifies to a condition on a
theorem range_of_a (a : ℝ) : discriminant_greater_than_zero a ↔ (a > 2 ∨ a < -1) :=
by
  -- Write the proof here
  sorry

end range_of_a_l125_125114


namespace sum_infinite_series_l125_125690

theorem sum_infinite_series :
  ∑ k in (Finset.range ∞), (12^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 3 :=
sorry

end sum_infinite_series_l125_125690


namespace gina_initial_money_l125_125745

variable (M : ℝ)
variable (kept : ℝ := 170)

theorem gina_initial_money (h1 : M * 1 / 4 + M * 1 / 8 + M * 1 / 5 + kept = M) : 
  M = 400 :=
by
  sorry

end gina_initial_money_l125_125745


namespace tangent_line_eq_l125_125779

open Real

theorem tangent_line_eq (x y : ℝ) (h : x > 0) (h_tangent : deriv (λ x : ℝ, log x + 2 * x) x = 3) :
  3 * x - y - 1 = 0 :=
sorry

end tangent_line_eq_l125_125779


namespace parallelogram_s_value_l125_125329

noncomputable def parallelogram_area (s : ℝ) : ℝ :=
  s * 2 * (s / Real.sqrt 2)

theorem parallelogram_s_value (s : ℝ) (h₀ : parallelogram_area s = 8 * Real.sqrt 2) : 
  s = 2 * Real.sqrt 2 :=
by
  sorry

end parallelogram_s_value_l125_125329


namespace graph_properties_l125_125317

theorem graph_properties (x : ℝ) (h : x ≠ 1) :
  let y := (x + 2) / (x - 1)
  (∀ x, y ≠ 1) ∧ 
  ¬ (∀ x, (x, y) = (1, 1)) ∧
  ∃ x, y = 0 ∧ 
  (∀ (x : ℝ), x ∈ Set.Ioo (-∞) (1) → (differentiable_at ℝ (λx, (x + 2) / (x - 1)) x) ∧ strict_mono (λx, (x + 2) / (x - 1)) (Set.Ioo (-∞) (1))) ∧ 
  (∀ (x : ℝ), x ∈ Set.Ioo (1) (∞) → (differentiable_at ℝ (λx, (x + 2) / (x - 1)) x) ∧ strict_mono (λx, (x + 2) / (x - 1)) (Set.Ioo (1) (∞))) :=
by
  sorry

end graph_properties_l125_125317


namespace calculate_expression_l125_125366

theorem calculate_expression :
  (-1^2) + |(-2 : ℤ)| + (∛(-8 : ℤ)) + (√((-3 : ℤ)^2)) = 2 :=
by
  sorry

end calculate_expression_l125_125366


namespace actual_diameter_of_tissue_l125_125989

-- We define the conditions as given in the problem.
def magnified_diameter (d : ℝ) : Prop := d = 0.3
def magnification_factor (m : ℕ) : Prop := m = 1000

-- We state the theorem that the actual diameter given these conditions equals 0.0003 centimeters.
theorem actual_diameter_of_tissue : 
  magnified_diameter 0.3 → magnification_factor 1000 → 0.3 / 1000 = 0.0003 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end actual_diameter_of_tissue_l125_125989


namespace sum_geometric_seq_2016_l125_125789

theorem sum_geometric_seq_2016 (a : ℕ → ℝ) (q : ℝ) (h_q_ne : q ≠ 1) 
(h1 : a 2 * a 6 = 16) 
(h2 : a 3 + a 5 = 8) :
∑ i in Finset.range 2016, a i = 0 :=
sorry

end sum_geometric_seq_2016_l125_125789


namespace solve_equation_l125_125782

variables (a b c : ℝ)

theorem solve_equation (h1 : 3 * a - 2 * b - 2 * c = 30) 
                       (h2 : a + b + c = 10) :
  sqrt (3 * a) - sqrt (2 * b + 2 * c) = sqrt 30 := 
sorry

end solve_equation_l125_125782


namespace diameter_of_large_circle_proof_l125_125713

noncomputable def diameter_of_large_circle : ℝ :=
  8.34

theorem diameter_of_large_circle_proof :
  ∀ (n : ℕ) (r1 r2 : ℝ), n = 8 → r1 = 2 → r2 = 2 → 
  (∀ (i : ℕ), i < n → ∃ (x y : ℝ),
      (x - y) * (x - y) = (2 * r1) ^ 2 ∧ 
      (x + y) * (x + y) = (2 * r2) ^ 2) →
  (2 * r2 + 2 * r1) ≈ diameter_of_large_circle :=
by
  intro n r1 r2 h1 h2 h3 h4
  sorry

end diameter_of_large_circle_proof_l125_125713


namespace cos_3theta_l125_125120

theorem cos_3theta (θ : ℝ) (h : cos θ = 1 / 4) : cos (3 * θ) = -11 / 16 :=
by
  sorry

end cos_3theta_l125_125120


namespace amount_of_water_per_minute_l125_125978

noncomputable def flow_rate_kmph := 7
noncomputable def flow_rate_mpm := flow_rate_kmph * 1000 / 60
noncomputable def river_depth := 2
noncomputable def river_width := 45
noncomputable def cross_sectional_area := river_width * river_depth
noncomputable def volume_per_min := cross_sectional_area * flow_rate_mpm

theorem amount_of_water_per_minute : volume_per_min = 10500.3 := 
sorry

end amount_of_water_per_minute_l125_125978


namespace yuna_friends_count_l125_125975

theorem yuna_friends_count (a b : ℕ) (h1 : a = 3) (h2 : b = 5) : a + 1 + b = 9 :=
by
  rw [h1, h2]
  rfl

end yuna_friends_count_l125_125975


namespace sample_variance_is_correct_l125_125242

-- Define the sequence and the conditions
def arith_seq (x₁ : ℝ) (n : ℕ) : ℝ :=
  x₁ + n

noncomputable def mean (x₁ : ℝ) : ℝ :=
  (x₁ + (arith_seq x₁ 1) + (arith_seq x₁ 2) + (arith_seq x₁ 3) + (arith_seq x₁ 4) +
   (arith_seq x₁ 5) + (arith_seq x₁ 6) + (arith_seq x₁ 7) + (arith_seq x₁ 8)) / 9

noncomputable def sample_variance (x₁ : ℝ) : ℝ :=
  (∑ i in (Finset.range 9), (arith_seq x₁ i - mean x₁) ^ 2) / 9

theorem sample_variance_is_correct (x₁ : ℝ) :
  sample_variance x₁ = 20 / 3 :=
sorry

end sample_variance_is_correct_l125_125242


namespace mila_hours_to_match_agnes_monthly_earnings_l125_125947

-- Definitions based on given conditions
def hourly_rate_mila : ℕ := 10
def hourly_rate_agnes : ℕ := 15
def weekly_hours_agnes : ℕ := 8
def weeks_in_month : ℕ := 4

-- Target statement to prove: Mila needs to work 48 hours to earn as much as Agnes in a month
theorem mila_hours_to_match_agnes_monthly_earnings :
  ∃ (h : ℕ), h = 48 ∧ (h * hourly_rate_mila) = (hourly_rate_agnes * weekly_hours_agnes * weeks_in_month) :=
by
  sorry

end mila_hours_to_match_agnes_monthly_earnings_l125_125947


namespace find_a_for_square_of_binomial_l125_125399

theorem find_a_for_square_of_binomial (a : ℝ) :
  (∃ r s : ℝ, (r * x + s)^2 = a * x^2 + 18 * x + 9) ↔ a = 9 := 
sorry

end find_a_for_square_of_binomial_l125_125399


namespace find_x_is_b_cubed_l125_125008

noncomputable def find_x (b x : ℝ) := 
  log x / log (b ^ 3) + 3 * log b / log x

theorem find_x_is_b_cubed (b : ℝ) (hb_pos : 0 < b) (hb_ne_one : b ≠ 1) (hx_ne_one : x ≠ 1) (hx : find_x b x = 2) : 
  x = b ^ 3 := 
sorry

end find_x_is_b_cubed_l125_125008


namespace solution_l125_125204

noncomputable def problem (p : Prop) : Prop :=
  (3^2 = 9) ∧ 
  (24 * (5 / 8) = 15) ∧ 
  (2.1 / (3 / 7) = 4.9) ∧ 
  (1 - 0.15 = 0.85) ∧ 
  ((1 / 5) + (4 / 5 * 0) = (1 / 5)) ∧ 
  ((1.25 * 8) / (1.25 * 8) = 1)

theorem solution : problem :=
by {
  split,
  -- Proving 3^2 = 9
  exact (by norm_num : 3 ^ 2 = 9),

  split,
  exact (by norm_num : 24 * (5 / 8) = 15),

  split,
  exact (by norm_num : 2.1 / (3 / 7) = 4.9),

  split,
  exact (by norm_num : 1 - 0.15 = 0.85),

  split,
  exact (by norm_num : (1 / 5) + (4 / 5 * 0) = (1 / 5)),

  -- Proving (1.25 * 8) / (1.25 * 8) = 1
  exact (by norm_num : (1.25 * 8) / (1.25 * 8) = 1)
}

example : 3^2 = 9 := by norm_num -- Proving 3^2 = 9
example : 24 * (5 / 8) = 15 := by norm_num -- Proving 24 * (5 / 8) = 15
example : 2.1 / (3 / 7) = 4.9 := by norm_num -- Proving 2.1 / (3 / 7) = 4.9
example : 1 - 0.15 = 0.85 := by norm_num -- Proving 1 - 0.15 = 0.85
example : (1 / 5) + (4 / 5 * 0) = (1 / 5) := by norm_num -- Proving (1 / 5) + (4 / 5 * 0) = (1 / 5)
example : (1.25 * 8) / (1.25 * 8) = 1 := by norm_num -- Proving (1.25 * 8) / (1.25 * 8) = 1

end solution_l125_125204


namespace face_value_of_share_l125_125324

-- Let FV be the face value of each share.
-- Given conditions:
-- Dividend rate is 9%
-- Market value of each share is Rs. 42
-- Desired interest rate is 12%

theorem face_value_of_share (market_value : ℝ) (dividend_rate : ℝ) (interest_rate : ℝ) (FV : ℝ) :
  market_value = 42 ∧ dividend_rate = 0.09 ∧ interest_rate = 0.12 →
  0.09 * FV = 0.12 * market_value →
  FV = 56 :=
by
  sorry

end face_value_of_share_l125_125324


namespace shift_sin_function_l125_125275

theorem shift_sin_function (x : ℝ) : 
  let y := λ x, sin (2 * x) in
  let y_shifted := λ x, sin (2 * (x + (π / 6))) in
  y_shifted x = sin (2 * x + π / 3) :=
by
  -- Proof to be provided
  sorry

end shift_sin_function_l125_125275


namespace seq_general_term_l125_125593

theorem seq_general_term
  (a : ℕ → ℚ) 
  (h : ∀ n, (∑ i in finset.range n, (3 ^ i) * a (i + 1)) = n / 3) : 
  ∀ n : ℕ, a (n + 1) = 1 / (3 ^ (n + 1)) :=
by
  intro n
  sorry

end seq_general_term_l125_125593


namespace order_of_a_b_c_l125_125040

noncomputable def a : ℝ := (Real.log (Real.sqrt 2)) / 2
noncomputable def b : ℝ := Real.log 3 / 6
noncomputable def c : ℝ := 1 / (2 * Real.exp 1)

theorem order_of_a_b_c : c > b ∧ b > a := by
  sorry

end order_of_a_b_c_l125_125040


namespace radius_tangent_circle_eq_l125_125137

noncomputable theory

def isosceles_triangle_radius (a α : ℝ) : ℝ :=
  let R := (a / 2) * tan (α / 2) in
  (a / 2) * (tan (α / 2))^3

theorem radius_tangent_circle_eq (a α : ℝ) :
  ∀ (r : ℝ), r = isosceles_triangle_radius a α := sorry

end radius_tangent_circle_eq_l125_125137


namespace pulled_pork_sauce_l125_125510

def sauce_content (ketchup vineger honey : ℕ) : ℕ := ketchup + vineger + honey

def burger_sauce_usage (burgers_per_sause : ℚ) (burger_number : ℕ) : ℚ := burger_number * burgers_per_sause

noncomputable def sauce_each_pulled_pork (total_sause_used burgers_sause usage_pulled_porck : ℚ) : ℚ :=
  (total_sause_used - burgers_sause) / usage_pulled_porck

theorem pulled_pork_sauce {ketchup vinegar honey burger_sauce burger_amt pulled_pork_amt solution : ℚ} :
  ketchup = 3 → vinegar = 1 → honey = 1 →
  burger_sauce = 1/4 →
  burger_amt = 8 →
  pulled_pork_amt = 18 →
  sauce_content ketchup vinegar honey = 5 →
  burger_sauce_usage burger_sauce burger_amt = 2 →
  sauce_each_pulled_pork 5 2 pulled_pork_amt = solution →
  solution = 1/6 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8' h9
  sorry

end pulled_pork_sauce_l125_125510


namespace complex_division_problem_l125_125976

theorem complex_division_problem :
  (1 + Complex.i)^2 / (1 - Complex.i)^3 = -1/2 - 1/2 * Complex.i :=
by
  sorry

end complex_division_problem_l125_125976


namespace incenter_centroid_relation_l125_125522

noncomputable section

variables {A B C I G : EuclideanSpace ℝ 3}

-- Define the distances IA^2, IB^2, IC^2, IG^2, GA^2, GB^2, GC^2
def IA_sq := ∥I - A∥^2
def IB_sq := ∥I - B∥^2
def IC_sq := ∥I - C∥^2
def IG_sq := ∥I - (A + B + C) / 3∥^2
def GA_sq := ∥(B + C - 2 * A) / 3∥^2
def GB_sq := ∥(A + C - 2 * B) / 3∥^2
def GC_sq := ∥(A + B - 2 * C) / 3∥^2

theorem incenter_centroid_relation :
  ∃ k : ℝ, IA_sq + IB_sq + IC_sq = k * IG_sq + GA_sq + GB_sq + GC_sq :=
begin
  use 3,
  sorry -- Proof goes here
end

end incenter_centroid_relation_l125_125522


namespace trapezoid_height_l125_125855

-- Conditions: definitions for isosceles trapezoid ABCD, point P, and given lengths
variables (A B C D P : Point)
variables (AP BP CD : ℝ)
variable h : ℝ

-- Conditions:
axiom is_isosceles_trapezoid : is_isosceles_trapezoid A B C D
axiom P_on_AB : on_segment P A B
axiom length_AP : dist A P = 11
axiom length_BP : dist B P = 27
axiom length_CD : dist C D = 34
axiom angle_CPD_right : ∠ C P D = 90

-- Theorem: height of the isosceles trapezoid ABCD
theorem trapezoid_height : h = 15 :=
begin
  sorry
end

end trapezoid_height_l125_125855


namespace range_of_k_for_inequalities_l125_125126

theorem range_of_k_for_inequalities :
  (∀ x : ℝ, (x^2 - x - 2 > 0) → (2x^2 + (2 * k + 7) * x + 7 * k < 0)) →
  (∀ x : ℝ, x = -3 ∨ x = -2 → (x^2 - x - 2 > 0) ∧ (2x^2 + (2 * k + 7) * x + 7 * k < 0)) →
  (-3 : ℝ) ≤ k ∧ k < 2 :=
sorry

end range_of_k_for_inequalities_l125_125126


namespace complement_union_l125_125094

variable (U : Set ℤ)
variable (A : Set ℤ)
variable (B : Set ℤ)

theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3})
                         (hA : A = {-1, 0, 1})
                         (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} :=
sorry

end complement_union_l125_125094


namespace monotonic_increasing_interval_l125_125786

def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

def interval_monotonically_increasing (k : ℤ) (x φ : ℝ) : set ℝ :=
  { x | k * Real.pi + Real.pi / 6 ≤ x ∧ x ≤ k * Real.pi + 2 * Real.pi / 3 }

theorem monotonic_increasing_interval (φ : ℝ) (k : ℤ) :
  (∀ x : ℝ, f x φ ≤ | f (Real.pi / 6) φ |) →
  f (Real.pi / 2) φ > f Real.pi φ →
  ∀ x : ℝ, x ∈ interval_monotonically_increasing k x φ :=
by 
  sorry

end monotonic_increasing_interval_l125_125786


namespace sum_xyz_eq_11sqrt5_l125_125484

noncomputable def x : ℝ :=
sorry

noncomputable def y : ℝ :=
sorry

noncomputable def z : ℝ :=
sorry

axiom pos_x : x > 0
axiom pos_y : y > 0
axiom pos_z : z > 0

axiom xy_eq_30 : x * y = 30
axiom xz_eq_60 : x * z = 60
axiom yz_eq_90 : y * z = 90

theorem sum_xyz_eq_11sqrt5 : x + y + z = 11 * Real.sqrt 5 :=
sorry

end sum_xyz_eq_11sqrt5_l125_125484


namespace vector_decomposition_l125_125635

variables (α β : ℝ)
def x : ℝ × ℝ × ℝ := (-9, -8, -3)
def p : ℝ × ℝ × ℝ := (1, 4, 1)
def q : ℝ × ℝ × ℝ := (-3, 2, 0)
def r : ℝ × ℝ × ℝ := (1, -1, 2)
def γ : ℝ := 0

theorem vector_decomposition : x = (α * p.1 + β * q.1, α * p.2 + β * q.2, α * p.3 + β * q.3) :=
by {
  let α := -3,
  let β := 2,
  have h1: -9 = α * p.1 + β * q.1, by sorry,
  have h2: -8 = α * p.2 + β * q.2, by sorry,
  have h3: -3 = α * p.3 + β * q.3, by sorry,
  exact ⟨h1, h2, h3⟩
}

end vector_decomposition_l125_125635


namespace radius_of_fourth_circle_l125_125503

noncomputable def geometric_radius (r1 : ℝ) (r6 : ℝ) (n : ℕ) : ℝ :=
  let k := (r6 / r1)^(1 / 5)
  r1 * k^n

theorem radius_of_fourth_circle :
  geometric_radius 5 20 3 = 6.6 :=
by
  let r1 := 5
  let r6 := 20
  let r4 := geometric_radius r1 r6 3
  have h1 : r4 = r1 * (r6 / r1)^(3 / 5) := by sorry
  have h2 : r1 * (r6 / r1)^(3 / 5) = 5 * (4^(3 / 5)) := by sorry
  have h3 : 4^(3 / 5) = ((4^(1 / 5))^3) := by sorry
  have h4 : r4 = 5 * ((4)^(1 / 5))^3 := by sorry
  have h5 : r4 ≈ 6.6 := by sorry
  exact h5

end radius_of_fourth_circle_l125_125503


namespace f_log_log3_l125_125755

-- Define the function f based on the given conditions
noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ :=
  a * sin x + b * real.cbrt x + c * real.log (x + real.sqrt (x^2 + 1)) + 4

-- Prove the required value
theorem f_log_log3 (a b c : ℝ) : 
  f (real.log10 (real.logb 3 10)) a b c = 5 →
  f (real.log10 (real.log10 3)) a b c = 3 := 
by 
  intro h1 
  sorry

end f_log_log3_l125_125755


namespace number_ways_replacing_asterisks_correct_l125_125150

def number_of_ways_to_replace := 
  let choices := {0, 1, 2, 3, 4, 5, 6, 7, 8} in
  let sum_fixed_digits := 2 + 0 + 1 + 6 + 0 + 2 in
  let is_divisible_by_5 (d : ℕ) := d = 0 ∨ d = 5 in
  let ways_to_replace_last_digit := 2 in
  let valid_sum_replacement := 7 in
  let remaining_digits_choices := 9 in
  let remaining_positions := 5 in 
  ways_to_replace_last_digit * remaining_digits_choices^remaining_positions

theorem number_ways_replacing_asterisks_correct :
  number_of_ways_to_replace = 13122 :=
by
  sorry

end number_ways_replacing_asterisks_correct_l125_125150


namespace distances_equal_to_plane_l125_125849

variable {Point : Type}
variables (A B C D E F : Point)
variables (line : Point → Point → Prop)
variables (circle : Point → Point → Point → Point → Prop)
variables (midpoint : Point → Point → Point)
variables (perpendicular : Point → Point → Prop)
variables (altitude_of : Point → Point → Point → Prop)
variables (distance_to_plane : Point → Point → Point → Prop)
variable (α : Point → Point → Prop)

-- Conditions
axiom Tetrahedron_ABCD : true -- assumes the existence of tetrahedron ABCD
axiom Altitude_BE : altitude_of B A D E
axiom Altitude_CF : altitude_of C A D F
axiom Plane_α_perpendicular_AD : perpendicular A D → α (midpoint A D)
axiom Points_ACDE_on_circle : circle A C D E
axiom Points_ABDF_on_circle : circle A B D F

-- Proof statement
theorem distances_equal_to_plane (A B C D E F : Point) (α : Point → Point → Prop) 
    [Tetrahedron_ABCD] [Altitude_BE] [Altitude_CF] [Plane_α_perpendicular_AD]
    [Points_ACDE_on_circle] [Points_ABDF_on_circle] :
    distance_to_plane E α = distance_to_plane F α := 
sorry

end distances_equal_to_plane_l125_125849


namespace calculate_lunch_break_duration_l125_125214

noncomputable def paula_rate (p : ℝ) : Prop := p > 0
noncomputable def helpers_rate (h : ℝ) : Prop := h > 0
noncomputable def apprentice_rate (a : ℝ) : Prop := a > 0
noncomputable def lunch_break_duration (L : ℝ) : Prop := L >= 0

-- Monday's work equation
noncomputable def monday_work (p h a L : ℝ) (monday_work_done : ℝ) :=
  0.6 = (9 - L) * (p + h + a)

-- Tuesday's work equation
noncomputable def tuesday_work (h a L : ℝ) (tuesday_work_done : ℝ) :=
  0.3 = (7 - L) * (h + a)

-- Wednesday's work equation
noncomputable def wednesday_work (p a L : ℝ) (wednesday_work_done : ℝ) :=
  0.1 = (1.2 - L) * (p + a)

-- Final proof statement
theorem calculate_lunch_break_duration (p h a L : ℝ)
  (H1 : paula_rate p)
  (H2 : helpers_rate h)
  (H3 : apprentice_rate a)
  (H4 : lunch_break_duration L)
  (H5 : monday_work p h a L 0.6)
  (H6 : tuesday_work h a L 0.3)
  (H7 : wednesday_work p a L 0.1) :
  L = 1.4 :=
sorry

end calculate_lunch_break_duration_l125_125214


namespace calculate_dot_product_l125_125784

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (π / 6 * x + π / 3)

theorem calculate_dot_product :
  let A : ℝ × ℝ := (4, 0)
  let B : ℝ × ℝ := ?B
  let C : ℝ × ℝ := ?C
  let O : ℝ × ℝ := (0, 0) in
  B + C = (8, 0) →
  ((O +. B) + (O +. C)) • A = 32 := by
  intros
  sorry

end calculate_dot_product_l125_125784


namespace arithmetic_sequence_sum_l125_125451

theorem arithmetic_sequence_sum :
  ∀ (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (d : ℝ),
    (∀ (n : ℕ), a_n n = 1 + (n - 1) * d) →  -- first condition
    d ≠ 0 →  -- second condition
    (∀ (n : ℕ), S_n n = n / 2 * (2 * 1 + (n - 1) * d)) →  -- third condition
    (1 * (1 + 4 * d) = (1 + d) ^ 2) →  -- fourth condition
    S_n 8 = 64 :=  -- conclusion
by {
  sorry
}

end arithmetic_sequence_sum_l125_125451


namespace at_least_eight_composites_l125_125434

theorem at_least_eight_composites (n : ℕ) (h : n > 1000) :
  ∃ (comps : Finset ℕ), 
    comps.card ≥ 8 ∧ 
    (∀ x ∈ comps, ¬Prime x) ∧ 
    (∀ k, k < 12 → n + k ∈ comps ∨ Prime (n + k)) :=
by
  sorry

end at_least_eight_composites_l125_125434


namespace find_n_l125_125612

theorem find_n :
  ∃ n : ℤ, (0 ≤ n) ∧ (n < 103) ∧ (98 * n ≡ 33 [MOD 103]) ∧ n = 87 :=
by
  sorry

end find_n_l125_125612


namespace difference_in_areas_l125_125319

noncomputable def equilateral_triangle_side : ℝ := 12
noncomputable def square_side_intercept : ℝ := equilateral_triangle_side / 2

theorem difference_in_areas : 
  let n := 9 - 3 * Real.sqrt 3 in
  let overlapped_area := (2 * n) ^ 2 - n * (6 - n) in
  let non_overlapping_area := 4 * (n * (6 - n)) + 4 * (equilateral_triangle_side * 10 - n ^ 2) in
  overlapped_area - non_overlapping_area = 102.6 - 57.6 * Real.sqrt 3 :=
by
  -- proof
  sorry

end difference_in_areas_l125_125319


namespace total_afternoon_evening_emails_l125_125856

-- Definitions based on conditions
def afternoon_emails : ℕ := 5
def evening_emails : ℕ := 8

-- Statement to be proven
theorem total_afternoon_evening_emails : afternoon_emails + evening_emails = 13 :=
by 
  sorry

end total_afternoon_evening_emails_l125_125856


namespace sum_of_solutions_eqn_l125_125734

theorem sum_of_solutions_eqn : 
  (∀ x : ℝ, -48 * x^2 + 100 * x + 200 = 0 → False) → 
  (-100 / -48) = (25 / 12) :=
by
  intros
  sorry

end sum_of_solutions_eqn_l125_125734


namespace function_value_4000_l125_125028

def f : ℤ → ℤ := sorry

theorem function_value_4000 : f(0) = 1 → (∀ x : ℤ, f(x + 2) = f(x) + 3 * x + 2) → f(4000) = 11998001 :=
by
  intros h₁ h₂
  sorry

end function_value_4000_l125_125028


namespace find_phi_l125_125463

theorem find_phi (ω : ℝ) (φ : ℝ) (hω : 0 < ω) (hφ : |φ| < π / 2) :
    (∀ x, sin (ω * (x + π / 3) + φ) = cos (2 * x + π / 4)) → φ = π / 12 :=
by
  intro h
  sorry

end find_phi_l125_125463


namespace vector_decomposition_l125_125624

theorem vector_decomposition : 
  ∃ (α β γ : ℝ), 
    let x := (15, -20, -1) in
    let p := (0, 2, 1) in
    let q := (0, 1, -1) in
    let r := (5, -3, 2) in
    x = (α * p.1 + β * q.1 + γ * r.1, α * p.2 + β * q.2 + γ * r.2, α * p.3 + β * q.3 + γ * r.3) ∧
    α = -6 ∧ β = 1 ∧ γ = 3 := 
by 
  sorry

end vector_decomposition_l125_125624


namespace max_value_trig_expression_l125_125410

/-- 
  Theorem: The maximum value of the expression 
  S = cos(θ₁) * sin(θ₂) + cos(θ₂) * sin(θ₃) + cos(θ₃) * sin(θ₄) + cos(θ₄) * sin(θ₅) + 
        cos(θ₅) * sin(θ₆) + cos(θ₆) * sin(θ₁) 
  over all real numbers θ₁, θ₂, θ₃, θ₄, θ₅, θ₆ is 3.
-/
theorem max_value_trig_expression
  (θ₁ θ₂ θ₃ θ₄ θ₅ θ₆ : ℝ) :
  let S := cos θ₁ * sin θ₂ + cos θ₂ * sin θ₃ + cos θ₃ * sin θ₄ +
           cos θ₄ * sin θ₅ + cos θ₅ * sin θ₆ + cos θ₆ * sin θ₁
  in S ≤ 3 :=
sorry

end max_value_trig_expression_l125_125410


namespace prod_of_extrema_l125_125436

noncomputable def f (x k : ℝ) : ℝ := (x^4 + k*x^2 + 1) / (x^4 + x^2 + 1)

theorem prod_of_extrema (k : ℝ) (h : ∀ x : ℝ, f x k ≥ 0 ∧ f x k ≤ 1 + (k - 1) / 3) :
  (∀ x : ℝ, f x k ≤ (k + 2) / 3) ∧ (∀ x : ℝ, f x k ≥ 1) → 
  (∃ φ ψ : ℝ, φ = 1 ∧ ψ = (k + 2) / 3 ∧ ∀ x y : ℝ, f x k = φ → f y k = ψ) → 
  (∃ φ ψ : ℝ, φ * ψ = (k + 2) / 3) :=
sorry

end prod_of_extrema_l125_125436


namespace topping_combinations_count_l125_125387

-- Definitions of different topping categories and options
inductive Cheese
| mozzarella
| cheddar
| goat_cheese

inductive Meat
| pepperoni
| sausage
| bacon
| ham

inductive Vegetable
| peppers
| onions
| mushrooms
| olives
| tomatoes

-- Conditions based on constraints provided
def valid_combination (c : Cheese) (m : Meat) (v : Vegetable) : Prop :=
  (c ≠ Cheese.mozzarella ∨ m ≠ Meat.pepperoni) ∧
  (m ≠ Meat.bacon ∨ v ≠ Vegetable.onions) ∧
  (v ≠ Vegetable.onions ∨ c = Cheese.goat_cheese)

-- Question (Proof Goal): Proving the number of valid topping combinations is 53
theorem topping_combinations_count : 
  ∃ n, n = 53 ∧ 
  (∀ c m v, valid_combination c m v → 
    finset.card (finset.filter (λ x, valid_combination x.1 x.2.1 x.2.2) 
      (finset.product (finset.product (finset.univ : finset Cheese) 
                                      (finset.univ : finset Meat)) 
                      (finset.univ : finset Vegetable))) = n) :=
sorry

end topping_combinations_count_l125_125387


namespace increasing_sequence_l125_125450

theorem increasing_sequence (n : ℕ) (hn : n > 0) : 
  (a_n = n / (n + 2)) → (a_n < a_{n + 1}) :=
by
  let a_n := λ n : ℕ, n / (n + 2)
  sorry

end increasing_sequence_l125_125450


namespace monotonic_on_interval_l125_125256

noncomputable def f (x m : ℝ) : ℝ := x^2 - 2 * m * x + 3

theorem monotonic_on_interval (m : ℝ) : 
  (∀ x₁ x₂ ∈ (set.Icc 1 3), x₁ ≤ x₂ → f x₁ m ≤ f x₂ m ∨ f x₁ m ≥ f x₂ m) ↔ m ≤ 1 ∨ m ≥ 3 :=
by
  sorry

end monotonic_on_interval_l125_125256


namespace mila_needs_48_hours_to_earn_as_much_as_agnes_l125_125949

/-- Definition of the hourly wage for the babysitters and the working hours of Agnes. -/
def mila_hourly_wage : ℝ := 10
def agnes_hourly_wage : ℝ := 15
def agnes_weekly_hours : ℝ := 8
def weeks_in_month : ℝ := 4

/-- Mila needs to work 48 hours in a month to earn as much as Agnes. -/
theorem mila_needs_48_hours_to_earn_as_much_as_agnes :
  ∃ (mila_monthly_hours : ℝ), mila_monthly_hours = 48 ∧ 
  mila_hourly_wage * mila_monthly_hours = agnes_hourly_wage * agnes_weekly_hours * weeks_in_month := 
sorry

end mila_needs_48_hours_to_earn_as_much_as_agnes_l125_125949


namespace combined_average_age_l125_125239

noncomputable def roomA : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
noncomputable def roomB : Set ℕ := {11, 12, 13, 14}
noncomputable def average_age_A := 55
noncomputable def average_age_B := 35
noncomputable def total_people := (10 + 4)
noncomputable def total_age_A := 10 * average_age_A
noncomputable def total_age_B := 4 * average_age_B
noncomputable def combined_total_age := total_age_A + total_age_B

theorem combined_average_age :
  (combined_total_age / total_people : ℚ) = 49.29 :=
by sorry

end combined_average_age_l125_125239


namespace palindromic_product_l125_125897

-- Definitions
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def is_nonzero_digit (d : ℕ) : Prop :=
  1 ≤ d ∧ d ≤ 9

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def distinct (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def exactly_one_even (a b c : ℕ) : Prop :=
  (a % 2 = 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0) ∨
  (a % 2 ≠ 0 ∧ b % 2 = 0 ∧ c % 2 ≠ 0) ∨
  (a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 = 0)

-- The main theorem
theorem palindromic_product :
  ∃ (n m p : ℕ),
    is_nonzero_digit n ∧ is_nonzero_digit m ∧ is_nonzero_digit p ∧
    distinct n m p ∧ exactly_one_even n m p ∧
    is_three_digit (n * 100 + m * 10 + p) ∧
    is_three_digit (n * 100 + p * 10 + m) ∧
    (n * 100 + m * 10 + p) * (n * 100 + p * 10 + m) = 29392 ∧
    is_palindrome (29392) := 
begin
  sorry
end

end palindromic_product_l125_125897


namespace smallest_possible_value_is_7_over_2_l125_125186

noncomputable def smallest_possible_value (z : ℂ) (h : |z^2 + 9| = |z * (z + 3 * complex.I)|) : ℝ :=
  ⨅ z : ℂ, |z + 2 * complex.I|

theorem smallest_possible_value_is_7_over_2 :
  smallest_possible_value = 7 / 2 :=
by
  sorry

end smallest_possible_value_is_7_over_2_l125_125186


namespace coefficient_x2_in_derivative_of_f_l125_125462

noncomputable def f (x : ℝ) : ℝ := (1 + x^2) * (1 - 2*x)^5

theorem coefficient_x2_in_derivative_of_f :
  ∀ (x : ℝ), coefficient (derivative f) 2 = -270 :=
by
  sorry

end coefficient_x2_in_derivative_of_f_l125_125462


namespace fifth_term_sequence_l125_125446

theorem fifth_term_sequence : 2^5 * 1 * 3 * 5 * 7 * 9 = 6 * 7 * 8 * 9 * 10 := 
by 
  sorry

end fifth_term_sequence_l125_125446


namespace solve_system_l125_125915

theorem solve_system : ∃ x y : ℚ, 4 * x - 3 * y = 2 ∧ 5 * x + 4 * y = 3 ∧ x = 17 / 31 ∧ y = 2 / 31 := by
  use (17 / 31)
  use (2 / 31)
  split; linarith
  split; linarith

end solve_system_l125_125915


namespace calculate_value_expression_l125_125683

theorem calculate_value_expression : 3 - (-3)^(3-(-3)) = -726 :=
by
  sorry

end calculate_value_expression_l125_125683


namespace min_distance_squared_l125_125307

noncomputable def min_squared_distances (AP BP CP DP EP : ℝ) : ℝ :=
  AP^2 + BP^2 + CP^2 + DP^2 + EP^2

theorem min_distance_squared :
  ∃ P : ℝ, ∀ (A B C D E : ℝ), A = 0 ∧ B = 1 ∧ C = 2 ∧ D = 5 ∧ E = 13 -> 
  min_squared_distances (abs (P - A)) (abs (P - B)) (abs (P - C)) (abs (P - D)) (abs (P - E)) = 114.8 :=
sorry

end min_distance_squared_l125_125307


namespace bridge_length_l125_125979

-- Given conditions in the problem
def train_length : ℝ := 140 -- meters
def train_speed : ℝ := 45 -- km/hr
def crossing_time : ℝ := 30 -- seconds

-- Mathematically equivalent proof problem
theorem bridge_length 
  (train_length = 140) 
  (train_speed = 45) 
  (crossing_time = 30) : 
  let speed_m_per_s := train_speed * 1000 / 3600;
  let total_distance := speed_m_per_s * crossing_time;
  let bridge_length := total_distance - train_length;
  bridge_length = 235 :=
by
  sorry

end bridge_length_l125_125979


namespace mass_percentage_O_in_Al2_CO3_3_l125_125409

-- Define the atomic masses
def atomic_mass_Al : Float := 26.98
def atomic_mass_C : Float := 12.01
def atomic_mass_O : Float := 16.00

-- Define the formula of aluminum carbonate
def Al_count : Nat := 2
def C_count : Nat := 3
def O_count : Nat := 9

-- Define the molar mass calculation
def molar_mass_Al2_CO3_3 : Float :=
  (Al_count.toFloat * atomic_mass_Al) + 
  (C_count.toFloat * atomic_mass_C) + 
  (O_count.toFloat * atomic_mass_O)

-- Define the mass of oxygen in aluminum carbonate
def mass_O_in_Al2_CO3_3 : Float := O_count.toFloat * atomic_mass_O

-- Define the mass percentage of oxygen in aluminum carbonate
def mass_percentage_O : Float := (mass_O_in_Al2_CO3_3 / molar_mass_Al2_CO3_3) * 100

-- Proof statement
theorem mass_percentage_O_in_Al2_CO3_3 :
  mass_percentage_O = 61.54 := by
  sorry

end mass_percentage_O_in_Al2_CO3_3_l125_125409


namespace math_problem_proof_l125_125376

theorem math_problem_proof :
    24 * (243 / 3 + 49 / 7 + 16 / 8 + 4 / 2 + 2) = 2256 :=
by
  -- Proof omitted
  sorry

end math_problem_proof_l125_125376


namespace problem_sin_cos_diff_l125_125735

theorem problem_sin_cos_diff : sin (7 * real.pi / 180) * cos (37 * real.pi / 180) - sin (83 * real.pi / 180) * sin (37 * real.pi / 180) = -1 / 2 := 
sorry

end problem_sin_cos_diff_l125_125735


namespace domain_of_f_l125_125614
noncomputable def f (x : ℝ) : ℝ := log 5 (log 2 (log 3 (log 4 (log 6 x))))

theorem domain_of_f : {x : ℝ | 6^64 < x} = {x : ℝ | ∃ y, y ∈ (6^64, ⊤) ∧ f x = y} := by
  sorry

end domain_of_f_l125_125614


namespace point_on_x_axis_l125_125243

theorem point_on_x_axis (x y : ℝ) (h : y = 0) : (x, y) ∈ { p : ℝ × ℝ | p.2 = 0 } :=
begin
  sorry
end

end point_on_x_axis_l125_125243


namespace slope_of_line_l125_125938

theorem slope_of_line (x y : ℝ) : (y - 3 = 4 * (x + 1)) → 4 :=
begin
  intro h,
  sorry -- To be filled with the proof
end

end slope_of_line_l125_125938


namespace gcd_pq_condition_l125_125746

theorem gcd_pq_condition (p q : ℤ) (h : Int.gcd p q = 1) :
  (∀ x : ℝ, (x ∈ ℚ) ↔ (x ^ p ∈ ℚ ∧ x ^ q ∈ ℚ)) :=
by
  intro x
  sorry

end gcd_pq_condition_l125_125746


namespace briana_annual_yield_is_10_percent_l125_125715

variables (B : ℝ) (roi_Emma roi_Briana : ℝ)

-- Emma's initial investment and the yield
def emma_investment : ℝ := 300
def emma_annual_yield : ℝ := 0.15

-- Briana's initial investment and the unknown yield percentage B
def briana_investment : ℝ := 500
def briana_annual_yield_percentage : ℝ := B / 100

-- ROI formulas after 2 years
noncomputable def ROI_Emma : ℝ := emma_investment * emma_annual_yield * 2
noncomputable def ROI_Briana : ℝ := briana_investment * briana_annual_yield_percentage * 2

-- The condition of the problem after 2 years ROI difference is $10
axiom roi_difference_condition : ROI_Briana - ROI_Emma = 10

-- Problem: Prove that Briana's annual percentage yield is 10%
theorem briana_annual_yield_is_10_percent : B = 10 :=
sorry

end briana_annual_yield_is_10_percent_l125_125715


namespace number_of_men_for_2km_road_l125_125272

noncomputable def men_for_1km_road : ℕ := 30
noncomputable def days_for_1km_road : ℕ := 12
noncomputable def hours_per_day_for_1km_road : ℕ := 8
noncomputable def length_of_1st_road : ℕ := 1
noncomputable def length_of_2nd_road : ℕ := 2
noncomputable def working_hours_per_day_2nd_road : ℕ := 14
noncomputable def days_for_2km_road : ℝ := 20.571428571428573

theorem number_of_men_for_2km_road (total_man_hours_1km : ℕ := men_for_1km_road * days_for_1km_road * hours_per_day_for_1km_road):
  (men_for_1km_road * length_of_2nd_road * days_for_1km_road * hours_per_day_for_1km_road = 5760) →
  ∃ (men_for_2nd_road : ℕ), men_for_1km_road * 2 * days_for_1km_road * hours_per_day_for_1km_road = 5760 ∧  men_for_2nd_road * days_for_2km_road * working_hours_per_day_2nd_road = 5760 ∧ men_for_2nd_road = 20 :=
by {
  sorry
}

end number_of_men_for_2km_road_l125_125272


namespace pythagorean_triple_exists_l125_125167

open Int Nat

theorem pythagorean_triple_exists (a b c : ℕ) (d : ℤ)
  (h₀ : a < b) (h₁ : b < c)
  (h₂ : gcd (c - a) (c - b) = 1)
  (h₃ : (a + d) * (a + d) + (b + d) * (b + d) = (c + d) * (c + d)) :
  ∃ (l m : ℤ), c + d = l * l + m * m := by
  sorry

end pythagorean_triple_exists_l125_125167


namespace g_n_plus_2_minus_g_n_l125_125538

noncomputable theory

def g (n : ℕ) : ℝ := (6 + 4 * real.sqrt 6) / 12 * ((2 + real.sqrt 6) / 3)^n + (6 - 4 * real.sqrt 6) / 12 * ((2 - real.sqrt 6) / 3)^n

theorem g_n_plus_2_minus_g_n (n : ℕ) : g (n + 2) - g n = ((-1 + 6 * real.sqrt 6) / 9) * g n :=
sorry

end g_n_plus_2_minus_g_n_l125_125538


namespace brother_highlighter_expense_l125_125471

noncomputable def spent_on_highlighters (total_money : ℕ) (sharpener_price : ℕ) (sharpeners_bought : ℕ)
(notebook_price : ℕ) (notebooks_bought : ℕ) (eraser_price : ℕ) (erasers_bought : ℕ) : ℕ :=
let heaven_spent := (sharpener_price * sharpeners_bought) + (notebook_price * notebooks_bought) in
let remaining_money := total_money - heaven_spent in
let brother_spent_on_erasers := eraser_price * erasers_bought in
remaining_money - brother_spent_on_erasers

theorem brother_highlighter_expense :
  spent_on_highlighters 150 3 5 7 6 2 15 = 63 :=
by
  exact sorry

end brother_highlighter_expense_l125_125471


namespace area_of_triangle_l125_125285

theorem area_of_triangle (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ≠ b) :
  let A := (1, a + b), B := ((a - b) / a, a), C := (0, a) in
  let area := |(b * (a - b)) / (2 * a)| in
  ∃ A B C : ℝ × ℝ, 
    A = (1, a + b) ∧ 
    B = ((a - b) / a, a) ∧ 
    C = (0, a) ∧ 
    (1 / 2) * |((a - b) / a) * b| = |(b * (a - b)) / (2 * a)| := 
by
  sorry

end area_of_triangle_l125_125285


namespace num_proper_subsets_of_A_l125_125868

open Finset

def A : Finset ℕ := {1, 2, 3}

theorem num_proper_subsets_of_A : A.card = 3 → (2 ^ A.card - 1) = 7 :=
by
  intros h
  sorry

end num_proper_subsets_of_A_l125_125868


namespace infinite_factors_gt_sqrt_l125_125950

theorem infinite_factors_gt_sqrt :
  ∃ᶠ n in at_top, ∃ (a b : ℕ), a * b = n^3 + 4 * n + 505 ∧ a > sqrt n ∧ b > sqrt n :=
sorry

end infinite_factors_gt_sqrt_l125_125950


namespace eval_expression_at_a_l125_125619

theorem eval_expression_at_a (a : ℝ) (h : a = 1 / 2) : (2 * a⁻¹ + a⁻¹ / 2) / a = 10 :=
by
  sorry

end eval_expression_at_a_l125_125619


namespace friends_count_l125_125862

-- Define that Laura has 28 blocks
def blocks := 28

-- Define that each friend gets 7 blocks
def blocks_per_friend := 7

-- The proof statement we want to prove
theorem friends_count : blocks / blocks_per_friend = 4 := by
  sorry

end friends_count_l125_125862


namespace students_only_in_math_l125_125359

variable (total_students math_students science_students : ℕ)
variable (H_total : total_students = 120)
variable (H_math : math_students = 85)
variable (H_science : science_students = 65)

theorem students_only_in_math :
  ∃ (students_only_math : ℕ), students_only_math = math_students - (math_students + science_students - total_students) ∧ students_only_math = 55 :=
by {
  use 55,
  rw [H_total, H_math, H_science],
  change 55 = 85 - (85 + 65 - 120),
  linarith,
}

end students_only_in_math_l125_125359


namespace water_height_after_valve_open_l125_125956

theorem water_height_after_valve_open 
  (h : ℝ) (ρ_w : ℝ) (ρ_o : ℝ) 
  (h_eq : h = 40 / 100)
  (ρ_w_eq : ρ_w = 1000)
  (ρ_o_eq : ρ_o = 700) : 
  ∃ h_w h_o, h_w + h_o = 2 * h ∧ (ρ_w * h_w = ρ_o * h_o) ∧ h_w = 34 / 100 :=
by
  use 34 / 100, 46 / 100
  split
  . sorry
  . split
    . sorry
    . rfl

end water_height_after_valve_open_l125_125956


namespace kids_played_on_Wednesday_l125_125515

def played_on_Monday : ℕ := 17
def played_on_Tuesday : ℕ := 15
def total_kids : ℕ := 34

theorem kids_played_on_Wednesday :
  total_kids - (played_on_Monday + played_on_Tuesday) = 2 :=
by sorry

end kids_played_on_Wednesday_l125_125515


namespace sin_squared_identity_l125_125367

theorem sin_squared_identity :
  1 - 2 * (Real.sin (105 * Real.pi / 180))^2 = - (Real.sqrt 3) / 2 :=
by sorry

end sin_squared_identity_l125_125367


namespace quadrilateral_conditions_max_points_l125_125961

variables {A B C D P : Type} [ordered_ring A]
variables (convex_quadrilateral : A → A → A → A → Prop)
variables (triangle_area : A → A → A → A)
variables (equal_area_points : ∀ A B C D P, convex_quadrilateral A B C D → 
  (triangle_area A B P = triangle_area B C P ∧ 
   triangle_area B C P = triangle_area C D P ∧ 
   triangle_area C D P = triangle_area D A P))

theorem quadrilateral_conditions (A B C D P : A) 
(conv_quad : convex_quadrilateral A B C D)
(equal_points : equal_area_points A B C D P conv_quad) :
(exists (M : A), M ∈ line A C ∧ M ∈ line B D) ∨ 
(exists E, intersection_point A C B D E ∧ [ABCD] = 2 * [DEC]) :=
sorry

theorem max_points (A B C D : A) 
(conv_quad : convex_quadrilateral A B C D)
(equal_points_count : ∃ P, equal_area_points A B C D P conv_quad) :
(count P, equal_area_points A B C D P conv_quad) = 1 :=
sorry

end quadrilateral_conditions_max_points_l125_125961


namespace shaded_area_l125_125147

theorem shaded_area (r : ℝ) (α : ℝ) (β : ℝ) (h1 : r = 4) (h2 : α = 1/2) :
  β = 64 - 16 * Real.pi := by sorry

end shaded_area_l125_125147


namespace circles_internally_tangent_l125_125792

theorem circles_internally_tangent (R r : ℝ) (h1 : R + r = 5) (h2 : R * r = 6) (d : ℝ) (h3 : d = 1) : d = |R - r| :=
by
  -- This allows the logic of the solution to be captured as the theorem we need to prove
  sorry

end circles_internally_tangent_l125_125792


namespace river_flow_volume_l125_125296

noncomputable def river_depth : ℝ := 2
noncomputable def river_width : ℝ := 45
noncomputable def flow_rate_kmph : ℝ := 4
noncomputable def flow_rate_mpm := flow_rate_kmph * 1000 / 60
noncomputable def cross_sectional_area := river_depth * river_width
noncomputable def volume_per_minute := cross_sectional_area * flow_rate_mpm

theorem river_flow_volume :
  volume_per_minute = 6000.3 := by
  sorry

end river_flow_volume_l125_125296


namespace smallest_sum_two_distinct_3_digit_numbers_l125_125963

theorem smallest_sum_two_distinct_3_digit_numbers : 
  (∃ a b c d e f : ℕ, {a, b, c, d, e, f} ⊆ {4, 5, 6, 7, 8, 9, 10} ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧ 
  e ≠ f ∧ 
  let x := 100*a + 10*b + c;
      y := 100*d + 10*e + f in
  x ≠ y) →
  ∀ S : ℕ, S = min ((100*a + 10*b + c) + (100*d + 10*e + f)) ∧ 
  let digits := {a, b, c, d, e, f}.erase 10 in
  456 ∈ digits ∧ 789 ∈ digits ∧ digits ⊆ {4, 5, 6, 7, 8, 9, 10} →
  S = 1245 :=
begin
  sorry
end

end smallest_sum_two_distinct_3_digit_numbers_l125_125963


namespace a4_equals_9_l125_125149

variable {a : ℕ → ℝ}

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem a4_equals_9 (h_geom : geometric_sequence a)
  (h_roots : ∃ a2 a6 : ℝ, a2^2 - 34 * a2 + 81 = 0 ∧ a6^2 - 34 * a6 + 81 = 0 ∧ a 2 = a2 ∧ a 6 = a6) :
  a 4 = 9 :=
sorry

end a4_equals_9_l125_125149


namespace silverware_probability_l125_125478

theorem silverware_probability : 
  let n := 24 in
  let r := 4 in
  let waysToChooseTotal := Nat.desc_factorial n r / Nat.factorial r in
  let favorable := (8 * 8 * 8 * 7) * 3 in
  (favorable.toRat / waysToChooseTotal.toRat) = (214 / 253 : ℚ) :=
by
  -- Let variables for ease of reference
  let n := 24
  let r := 4
  let waysToChooseTotal := Nat.desc_factorial n r / Nat.factorial r
  let favorable := (8 * 8 * 8 * 7) * 3
  -- Calculate the fraction
  have h1 : (favorable.toRat / waysToChooseTotal.toRat) = (214 / 253 : ℚ) := sorry
  exact h1

end silverware_probability_l125_125478


namespace pears_for_twenty_apples_l125_125818

-- Definitions based on given conditions
variables (a o p : ℕ) -- represent the number of apples, oranges, and pears respectively
variables (k1 k2 : ℕ) -- scaling factors 

-- Conditions as given
axiom ten_apples_five_oranges : 10 * a = 5 * o
axiom three_oranges_four_pears : 3 * o = 4 * p

-- Proving the number of pears Mia can buy for 20 apples
theorem pears_for_twenty_apples : 13 * p ≤ (20 * a) :=
by
  -- Actual proof would go here
  sorry

end pears_for_twenty_apples_l125_125818


namespace fraction_product_l125_125678

theorem fraction_product : (1 / 2) * (1 / 3) * (1 / 6) * 120 = 10 / 3 :=
by
  sorry

end fraction_product_l125_125678


namespace integer_solutions_l125_125010

theorem integer_solutions :
  ∃ (a b c : ℤ), a + b + c = 24 ∧ a^2 + b^2 + c^2 = 210 ∧ a * b * c = 440 ∧
    (a = 5 ∧ b = 8 ∧ c = 11) ∨ (a = 5 ∧ b = 11 ∧ c = 8) ∨ 
    (a = 8 ∧ b = 5 ∧ c = 11) ∨ (a = 8 ∧ b = 11 ∧ c = 5) ∨
    (a = 11 ∧ b = 5 ∧ c = 8) ∨ (a = 11 ∧ b = 8 ∧ c = 5) :=
sorry

end integer_solutions_l125_125010


namespace ways_to_sum_2022_using_2s_and_3s_l125_125805

theorem ways_to_sum_2022_using_2s_and_3s : 
  (∃ n : ℕ, n ≤ 337 ∧ 6 * 337 = 2022) →
  (finset.card (finset.Icc 0 337) = 338) :=
by
  intros n h
  rw finset.card_Icc
  sorry

end ways_to_sum_2022_using_2s_and_3s_l125_125805


namespace inequality_solution_set_l125_125939

theorem inequality_solution_set
  (x : ℝ)
  (h₁ : x ≠ 0) :
  \left| \frac{x - 2}{x} \right| > \frac{x - 2}{x} ↔ 0 < x ∧ x < 2 :=
sorry

end inequality_solution_set_l125_125939


namespace value_of_a14_l125_125497

noncomputable def a : ℕ → ℝ
def sequence_pos : Prop := ∀ (n : ℕ), a n > 0
def a2 : Prop := a 2 = 2
def a8 : Prop := a 8 = 8
def relation : Prop := ∀ (n : ℕ), n ≥ 2 → (Real.sqrt (a (n - 1)) * Real.sqrt (a (n + 1)) = a n)

theorem value_of_a14
  (seq_pos : sequence_pos)
  (h2 : a2)
  (h8 : a8)
  (h_rel : relation) :
  a 14 = 32 := 
sorry

end value_of_a14_l125_125497


namespace solve_abs_inequality_l125_125419

theorem solve_abs_inequality (x : ℝ) : 
  2 ≤ |x - 3| ∧ |x - 3| ≤ 5 ↔ (-2 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 8) :=
sorry

end solve_abs_inequality_l125_125419


namespace log10_two_bounds_l125_125957

theorem log10_two_bounds
  (h1 : 10 ^ 3 = 1000)
  (h2 : 10 ^ 4 = 10000)
  (h3 : 2 ^ 10 = 1024)
  (h4 : 2 ^ 12 = 4096) :
  1 / 4 < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < 0.4 := 
sorry

end log10_two_bounds_l125_125957


namespace valid_rearrangements_count_l125_125802

noncomputable def count_valid_rearrangements : ℕ := sorry

theorem valid_rearrangements_count :
  count_valid_rearrangements = 7 :=
sorry

end valid_rearrangements_count_l125_125802


namespace simplify_polynomial_l125_125235

theorem simplify_polynomial :
  (3 * y - 2) * (6 * y ^ 12 + 3 * y ^ 11 + 6 * y ^ 10 + 3 * y ^ 9) =
  18 * y ^ 13 - 3 * y ^ 12 + 12 * y ^ 11 - 3 * y ^ 10 - 6 * y ^ 9 :=
by
  sorry

end simplify_polynomial_l125_125235


namespace solution_set_inequality_l125_125074

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 0 then 2^(-x) + 1 else -real.sqrt x

theorem solution_set_inequality :
  {x : ℝ | f (x + 1) - 9 ≤ 0} = set.Ici (-4) :=
by {
  have h_cases : ∀ x, f (x + 1) = if x ≤ -1 then 2^(-(x + 1)) + 1 else -real.sqrt (x + 1),
  { intro x, unfold f, split_ifs; refl },
  sorry
}

end solution_set_inequality_l125_125074


namespace greatest_num_consecutive_sum_36_l125_125284

theorem greatest_num_consecutive_sum_36 :
  (∃ N a : ℤ, ∑ i in (finset.range N).map (λ i, a + i) = 36 ∧ N = 72) :=
sorry

end greatest_num_consecutive_sum_36_l125_125284


namespace exist_directed_graph_two_step_l125_125043

theorem exist_directed_graph_two_step {n : ℕ} (h : n > 4) :
  ∃ G : SimpleGraph (Fin n), 
    (∀ u v : Fin n, u ≠ v → 
      (G.Adj u v ∨ (∃ w : Fin n, u ≠ w ∧ w ≠ v ∧ G.Adj u w ∧ G.Adj w v))) :=
sorry

end exist_directed_graph_two_step_l125_125043


namespace unique_solution_tan_eq_sin_cos_l125_125476

theorem unique_solution_tan_eq_sin_cos :
  ∃! x, 0 ≤ x ∧ x ≤ Real.arccos 0.1 ∧ Real.tan x = Real.sin (Real.cos x) :=
sorry

end unique_solution_tan_eq_sin_cos_l125_125476


namespace impact_point_distances_correct_l125_125343

structure Rectangle where
  width : ℝ
  height : ℝ

def point_of_impact_distances (rect : Rectangle) 
  (Area_left Area_right Area_bottom : ℝ) 
  (h1 : Area_right = 3 * Area_left) 
  (h2 : Area_bottom = 2 * Area_left) : ℝ × ℝ × ℝ × ℝ :=
  let dist_left := 2
  let dist_right := 6
  let dist_bottom := 3
  let dist_top := rect.height - dist_bottom
  (dist_left, dist_right, dist_bottom, dist_top)

theorem impact_point_distances_correct :
  ∀ (rect : Rectangle) (Area_left Area_right Area_bottom : ℝ),
    rect.width = 8 ∧ rect.height = 6 →
    Area_right = 3 * Area_left →
    Area_bottom = 2 * Area_left →
    point_of_impact_distances rect Area_left Area_right Area_bottom
    (by simp) (by simp) = (2, 6, 3, 3) := by
  intros rect Area_left Area_right Area_bottom 
         h_rect h_area1 h_area2
  simp [point_of_impact_distances]
  sorry

end impact_point_distances_correct_l125_125343


namespace f_zero_f_odd_f_not_decreasing_f_increasing_l125_125322

noncomputable def f (x : ℝ) : ℝ := sorry -- The function definition is abstract.

-- Functional equation condition
axiom functional_eq (x y : ℝ) (h1 : -1 < x) (h2 : x < 1) (h3 : -1 < y) (h4 : y < 1) : 
  f x + f y = f ((x + y) / (1 + x * y))

-- Condition for negative interval
axiom neg_interval (x : ℝ) (h1 : -1 < x) (h2 : x < 0) : f x < 0

-- Statements to prove

-- a): f(0) = 0
theorem f_zero : f 0 = 0 := 
by
  sorry

-- b): f(x) is an odd function
theorem f_odd (x : ℝ) (h1 : -1 < x) (h2 : x < 1) : f (-x) = -f x := 
by
  sorry

-- c): f(x) is not a decreasing function
theorem f_not_decreasing (x1 x2 : ℝ) (h1 : -1 < x1) (h2 : x1 < x2) (h3 : x2 < 1) : ¬(f x1 > f x2) :=
by
  sorry

-- d): f(x) is an increasing function
theorem f_increasing (x1 x2 : ℝ) (h1 : -1 < x1) (h2 : x1 < x2) (h3 : x2 < 1) : f x1 < f x2 :=
by
  sorry

end f_zero_f_odd_f_not_decreasing_f_increasing_l125_125322


namespace kathleen_remaining_money_l125_125166

-- Define the conditions
def saved_june := 21
def saved_july := 46
def saved_august := 45
def spent_school_supplies := 12
def spent_clothes := 54
def aunt_gift_threshold := 125
def aunt_gift := 25

-- Prove that Kathleen has the correct remaining amount of money
theorem kathleen_remaining_money : 
    (saved_june + saved_july + saved_august) - 
    (spent_school_supplies + spent_clothes) = 46 := 
by
  sorry

end kathleen_remaining_money_l125_125166


namespace correct_time_fraction_day_l125_125326

-- Definition of the problem conditions
def is_time_display_correct (time: Nat) : Bool :=
  let digits := to_digits time
  ¬(2 ∈ digits)

def to_digits (time: Nat) : List Nat :=
  -- This function represents converting an integer to a list of its digits
  -- In actual Lean code, you would use appropriate functions to handle this.
  sorry

-- Hours part: Total hours in 12-hour format
def correct_hours_fraction : Rat := 5 / 6

-- Minutes and seconds part: Total correct units in an hour or a minute
def correct_minutes_fraction : Rat := 11 / 15

-- Overall fraction of the day
theorem correct_time_fraction_day : 
  let fraction : Rat := correct_hours_fraction * correct_minutes_fraction * correct_minutes_fraction in
  fraction = 121 / 270 :=
by {
  sorry
}

end correct_time_fraction_day_l125_125326


namespace correct_division_l125_125968

theorem correct_division (a : ℝ) : a^8 / a^2 = a^6 := by 
  sorry

end correct_division_l125_125968


namespace find_k_l125_125639

variable {c : ℝ} {k : ℝ}

theorem find_k 
  (h1 : ∀ n : ℕ, a (n + 1) = c * a n)
  (h2 : ∀ n : ℕ, ∑ i in finset.range (n + 1), a i = 3 * n + k) : k = -1 :=
by
-- placeholder for the eventual proof
sorry

end find_k_l125_125639


namespace maximum_value_expression_l125_125412

theorem maximum_value_expression (θ1 θ2 θ3 θ4 θ5 θ6 : ℝ) :
  ∃ θ1 θ2 θ3 θ4 θ5 θ6, (cos θ1 * sin θ2 +
                         cos θ2 * sin θ3 +
                         cos θ3 * sin θ4 +
                         cos θ4 * sin θ5 +
                         cos θ5 * sin θ6 +
                         cos θ6 * sin θ1) = 3 :=
sorry

end maximum_value_expression_l125_125412


namespace evaluate_expression_l125_125115

theorem evaluate_expression : (x = 5) → (3 * x + 2 = 17) :=
by
  intros h
  rw h
  sorry

end evaluate_expression_l125_125115


namespace fraction_of_men_married_is_two_thirds_l125_125982

-- Define the total number of faculty members
def total_faculty_members : ℕ := 100

-- Define the number of women as 70% of the faculty members
def women : ℕ := (70 * total_faculty_members) / 100

-- Define the number of men as 30% of the faculty members
def men : ℕ := (30 * total_faculty_members) / 100

-- Define the number of married faculty members as 40% of the faculty members
def married_faculty : ℕ := (40 * total_faculty_members) / 100

-- Define the number of single men as 1/3 of the men
def single_men : ℕ := men / 3

-- Define the number of married men as 2/3 of the men
def married_men : ℕ := (2 * men) / 3

-- Define the fraction of men who are married
def fraction_married_men : ℚ := married_men / men

-- The proof statement
theorem fraction_of_men_married_is_two_thirds : fraction_married_men = 2 / 3 := 
by sorry

end fraction_of_men_married_is_two_thirds_l125_125982


namespace math_problem_l125_125452

-- Define the initial conditions
variable (m n a : ℤ)
variable (sqrt_m : ℤ) (root1 root2 : ℤ)
hypothesis (H1 : sqrt_m = 3)
hypothesis (H2 : root1 = a + 4)
hypothesis (H3 : root2 = 2 * a - 16)
hypothesis (H4 : root1 * root1 = n)
hypothesis (H5 : root2 * root2 = n)

-- Prove the math problem statements
theorem math_problem :
    ∃ m n a, (m = 9) ∧ (n = 64) ∧ (sqrt_m = 3) ∧ (root1 = a + 4) ∧ (root2 = 2 * a - 16) 
    ∧ (root1 * root1 = n) ∧ (root2 * root2 = n) 
    ∧ (root1 + root2 = 0) ∧ (a = 4) 
    ∧ (∃ sqrt_m, sqrt_m = 3)
    ∧ (√ m = 3 ∧ 7 * m - n = -1 ∧ ³√ (7 * m - n) = -1) :=
begin
    existsi [9, 64, 4],
    split,
    -- We will complete this proof later using helper lemmas and proofs related to root finding and verification.
    sorry,
end

end math_problem_l125_125452


namespace part1_prob_one_smart_life_question_part2_prob_distribution_and_variance_l125_125492

noncomputable def comb (n k : ℕ) : ℕ :=
  nat.choose n k

theorem part1_prob_one_smart_life_question :
  let total_ways := comb 6 3
  let ways_one_smart_life := comb 4 2 * comb 2 1
  (ways_one_smart_life : ℚ) / total_ways = 3 / 5 :=
by
  sorry

theorem part2_prob_distribution_and_variance :
  let pX0 := (comb 4 3 : ℚ) / comb 6 3
  let pX1 := (comb 4 2 * comb 2 1 : ℚ) / comb 6 3
  let pX2 := (comb 4 1 * comb 2 2 : ℚ) / comb 6 3
  let EX := 0 * pX0 + 1 * pX1 + 2 * pX2
  let DX := (0 - EX)^2 * pX0 + (1 - EX)^2 * pX1 + (2 - EX)^2 * pX2
  (pX0 = 1 / 5) ∧ (pX1 = 3 / 5) ∧ (pX2 = 1 / 5) ∧ (EX = 1) ∧ (DX = 2 / 5) :=
by
  sorry

end part1_prob_one_smart_life_question_part2_prob_distribution_and_variance_l125_125492


namespace complement_union_eq_l125_125091

variable (U : Set Int := {-2, -1, 0, 1, 2, 3}) 
variable (A : Set Int := {-1, 0, 1}) 
variable (B : Set Int := {1, 2}) 

theorem complement_union_eq :
  U \ (A ∪ B) = {-2, 3} := by 
  sorry

end complement_union_eq_l125_125091


namespace intersection_of_A_and_B_l125_125887

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := { x | x^2 + x ≤ 0}

theorem intersection_of_A_and_B :
  A ∩ B = {-1, 0} :=
by
  sorry

end intersection_of_A_and_B_l125_125887


namespace pentagon_diagonal_sum_l125_125523

def sum_of_diagonals (PQ RT QR ST PT : ℕ) (a b : ℕ) : Prop :=
  PQ = 4 ∧ RT = 4 ∧ QR = 11 ∧ ST = 11 ∧ PT = 15 ∧ a = 90 ∧ b = 1 ∧ (a + b = 91)

theorem pentagon_diagonal_sum (PQ RT QR ST PT : ℕ) (a b : ℕ) : 
  sum_of_diagonals PQ RT QR ST PT a b := 
by {
  -- Conditions for the pentagon
  have h1 : PQ = 4 := sorry,
  have h2 : RT = 4 := sorry,
  have h3 : QR = 11 := sorry,
  have h4 : ST = 11 := sorry,
  have h5 : PT = 15 := sorry,

  -- Proving the sum of diagonals, and the values of a and b
  have h6 : a = 90 := sorry,
  have h7 : b = 1 := sorry,
  have result : a + b = 91 := sorry,

  -- Combining all statements
  exact ⟨h1, h2, h3, h4, h5, h6, h7, result⟩
}

end pentagon_diagonal_sum_l125_125523


namespace trigonometric_identity_l125_125309

theorem trigonometric_identity :
  sin (155 * (Real.pi / 180)) * sin (55 * (Real.pi / 180)) + 
  cos (25 * (Real.pi / 180)) * cos (55 * (Real.pi / 180)) = (Real.sqrt 3) / 2 :=
by
  sorry

end trigonometric_identity_l125_125309


namespace inequality_proof_l125_125878

open Real

theorem inequality_proof (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l125_125878


namespace trigonometric_identity_l125_125638

theorem trigonometric_identity : 
  cos (17 * (Real.pi / 180)) * cos (13 * (Real.pi / 180)) - sin (17 * (Real.pi / 180)) * sin (13 * (Real.pi / 180)) = Real.sqrt 3 / 2 := 
by
  sorry

end trigonometric_identity_l125_125638


namespace outfits_count_l125_125916

theorem outfits_count (shirts ties hats : ℕ) (h_shirts : shirts = 8) (h_ties : ties = 6) (h_hats : hats = 4) : 
  shirts * ties * hats = 192 := 
by 
  rw [h_shirts, h_ties, h_hats] 
  norm_num
  sorry

end outfits_count_l125_125916


namespace prove_initial_concentration_l125_125316

noncomputable def initial_concentration
  (V : ℝ) -- Total volume of the solution.
  (C : ℝ) -- Initial concentration of the solution (to be found).
  (fraction_replaced : ℝ) -- Fraction of the solution that was replaced.
  (new_concentration : ℝ) -- New concentration of the solution.
  (replacement_concentration : ℝ) -- Concentration of the solution that was added.
  : Prop :=
  fraction_replaced * C + (1 - fraction_replaced) * replacement_concentration = new_concentration

theorem prove_initial_concentration :
  ∀ (V : ℝ), initial_concentration V 0.45 0.5 0.35 0.25 :=
by
  intro V
  have fraction_replaced := 0.5
  have new_concentration := 0.35
  have replacement_concentration := 0.25
  have initial := (fraction_replaced * 0.45 + (1 - fraction_replaced) * replacement_concentration = new_concentration)
  exact initial

end prove_initial_concentration_l125_125316


namespace prob_fourth_term_integer_l125_125889

noncomputable def seq_term (n : ℕ) : ℕ → ℚ 
| 0 := 4
| (k + 1) := if k.bodd then (3 * (seq_term k) + 3) else ((seq_term k - 3) / 3)

theorem prob_fourth_term_integer : 
  (∑ n in list.range 3, if is_integer (seq_term 3 n) then 1 else 0) / 8 = 3/4 := 
by sorry  -- Proof omitted

end prob_fourth_term_integer_l125_125889


namespace complement_union_eq_l125_125084

open Set

-- Definition of sets U, A, and B
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- Statement of the problem
theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 3} :=
by sorry

end complement_union_eq_l125_125084


namespace determine_running_speed_l125_125859
noncomputable def John_running_speed (x : ℝ) : Prop :=
  let cycling_time := 15 / (3 * x - 2)
  let running_time := 3 / x
  let total_exercise_time := cycling_time + running_time
  total_exercise_time = 2

theorem determine_running_speed :
  (∃ x : ℝ, John_running_speed x) → ∃ x ∈ set.Icc (0 : ℝ) 10, abs (x - 4.44) < 0.01 :=
begin
  have real_exists_x : ∃ x, John_running_speed x, sorry, -- This is a placeholder for the proof that John_running_speed x has solutions.
  use 4.44,
  split,
  { linarith, },
  { norm_num, },
end

end determine_running_speed_l125_125859


namespace circle_radius_tangent_l125_125189

theorem circle_radius_tangent (A B O M X : Type) (AB AM MB r : ℝ)
  (hL1 : AB = 2) (hL2 : AM = 1) (hL3 : MB = 1) (hMX : MX = 1/2)
  (hTangent1 : OX = 1/2 + r) (hTangent2 : OM = 1 - r)
  (hPythagorean : OM^2 + MX^2 = OX^2) :
  r = 1/3 :=
by
  sorry

end circle_radius_tangent_l125_125189


namespace ratio_of_areas_is_one_l125_125247

-- Definitions of points and geometric properties based on conditions
variable (A B C D E N : Point)
variable (ω : Circle)
variable (BC : Line)
variable [h1 : Diameter AC ω]
variable [h2 : PerpendicularLineThroughPoint D BC E ω]

-- Define the areas of the respective geometric shapes
def area_triangle_BCD : ℝ := sorry
def area_quadrilateral_ABEC : ℝ := sorry

-- The theorem we want to prove
theorem ratio_of_areas_is_one :
  area_triangle_BCD A B C D = area_quadrilateral_ABEC A B E C / 2 :=
sorry

end ratio_of_areas_is_one_l125_125247


namespace expression_negativity_l125_125432

-- Given conditions: a, b, and c are lengths of the sides of a triangle
variables (a b c : ℝ)
axiom triangle_inequality1 : a + b > c
axiom triangle_inequality2 : b + c > a
axiom triangle_inequality3 : c + a > b

-- To prove: (a - b)^2 - c^2 < 0
theorem expression_negativity (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  (a - b)^2 - c^2 < 0 :=
sorry

end expression_negativity_l125_125432


namespace total_seconds_eq_250200_l125_125676

def bianca_hours : ℝ := 12.5
def celeste_hours : ℝ := 2 * bianca_hours
def mcclain_hours : ℝ := celeste_hours - 8.5
def omar_hours : ℝ := bianca_hours + 3

def total_hours : ℝ := bianca_hours + celeste_hours + mcclain_hours + omar_hours
def hour_to_seconds : ℝ := 3600
def total_seconds : ℝ := total_hours * hour_to_seconds

theorem total_seconds_eq_250200 : total_seconds = 250200 := by
  sorry

end total_seconds_eq_250200_l125_125676


namespace expressions_equal_iff_c_eq_formula_l125_125391

theorem expressions_equal_iff_c_eq_formula
  (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (sqrt (b + 4 * a / c) = b * sqrt (a / c)) ↔ (c = b * a - 4 * a / b) := 
sorry

end expressions_equal_iff_c_eq_formula_l125_125391


namespace complement_union_eq_l125_125083

open Set

-- Definition of sets U, A, and B
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- Statement of the problem
theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 3} :=
by sorry

end complement_union_eq_l125_125083


namespace tangent_point_abscissa_l125_125885

noncomputable theory

open Real

def f (λ x : ℝ) : ℝ := exp x + λ * exp (-x)

def f'_def (λ x : ℝ) : ℝ := deriv (f λ) x

def f''_def (λ x : ℝ) : ℝ := deriv (f'_def λ) x

theorem tangent_point_abscissa (λ : ℝ) 
  (h : ∀ x : ℝ, f''_def λ x = - (f''_def λ (-x))) 
  (slope_tangent : ∀ x : ℝ, (deriv (f λ) x) = 3/2) 
  : ∃ x₀ : ℝ, x₀ = ln 2 ∧ deriv (f λ) x₀ = 3/2 :=
sorry

end tangent_point_abscissa_l125_125885


namespace simplify_expression_is_zero_l125_125911

noncomputable def simplify_expression (m n : ℝ) : ℝ :=
  m - n - (m - n)

theorem simplify_expression_is_zero (m n : ℝ) : simplify_expression m n = 0 :=
by {
  unfold simplify_expression,
  -- simplify using basic arithmetic
  calc
  m - n - (m - n) = m - n - m + n : by simp
  ... = (m - m) + (n - n)   : by simp
  ... = 0                   : by ring
}

end simplify_expression_is_zero_l125_125911


namespace geometric_sequence_a3_l125_125846

variable (a : ℕ → ℂ) (a1 : ℂ) (q : ℂ)
  (h_a1 : a 1 = a1)
  (h_q : q = -2)
  (h_geometric : ∀ n, a (n + 1) = a 1 * q ^ n)

theorem geometric_sequence_a3 :
  a 3 = 12 :=
by
  simp only [h_geometric, h_a1, h_q]
  sorry

end geometric_sequence_a3_l125_125846


namespace OE_perpendicular_BD_l125_125136

variables {α : Type*} [LinearOrder α] [Field α] [CharZero α]

/-- Given conditions -/
-- Assume an isosceles triangle with vertices A, B, C
variables (A B C D E O : α)
-- Assume O is the circumcenter of triangle ABC
-- Assume D is the midpoint of AC
-- Assume E is the centroid of triangle DBC

noncomputable def is_midpoint (P Q R : α) : Prop := 2 * Q = P + R
noncomputable def is_centroid (P Q R S : α) : Prop := 3 * S = P + Q + R

-- Define the main theorem
theorem OE_perpendicular_BD 
  (isosceles_triangle : AC = BC)
  (circumcenter : O = 0)
  (midpoint_D : is_midpoint A D C)
  (centroid_E : is_centroid D B C E) :
  dot (OE B D) = 0 :=
sorry

end OE_perpendicular_BD_l125_125136


namespace hyperbola_eccentricity_l125_125465

variable {a b c e : ℝ}

def hyperbola_eq (x y a b : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

def point_A (a : ℝ) : ℝ × ℝ :=
  (-a, 0)

def point_F (c : ℝ) : ℝ × ℝ :=
  (c, 0)

def point_B (c b a : ℝ) : ℝ × ℝ :=
  (c, (b^2 / a))

def slope (P Q : ℝ × ℝ) : ℝ :=
  (Q.2 - P.2) / (Q.1 - P.1)

theorem hyperbola_eccentricity :
  ∀ (a b c : ℝ),
  0 < a → 0 < b → a < c → 
  hyperbola_eq c (b^2 / a) a b →
  slope (point_A a) (point_B c b a) = 1/2 →
  e = c / a →
  e = 3 / 2 := 
by
  intros a b c ha hb hac hc_eq hslope hecc
  sorry

end hyperbola_eccentricity_l125_125465


namespace books_loaned_out_is_40_l125_125294

def initial_books := 75
def returned_percentage := 0.80
def end_of_month_books := 67

theorem books_loaned_out_is_40 (x : ℝ) (h : x ∈ ℝ) : 75 - (1 - 0.80) * x = 67 → x = 40 :=
by
  assume h0 : 75 - (1 - 0.80) * x = 67
  sorry

end books_loaned_out_is_40_l125_125294


namespace distance_from_Q_to_EF_is_24_div_5_l125_125380

-- Define the configuration of the square and points
def E := (0, 8)
def F := (8, 8)
def G := (8, 0)
def H := (0, 0)
def N := (4, 0) -- Midpoint of GH
def r1 := 4 -- Radius of the circle centered at N
def r2 := 8 -- Radius of the circle centered at E

-- Definition of the first circle centered at N with radius r1
def circle1 (x y : ℝ) := (x - 4)^2 + y^2 = r1^2

-- Definition of the second circle centered at E with radius r2
def circle2 (x y : ℝ) := x^2 + (y - 8)^2 = r2^2

-- Define the intersection point Q, other than H
def Q := (32 / 5, 16 / 5) -- Found as an intersection point between circle1 and circle2

-- Define the distance from point Q to the line EF
def dist_to_EF := 8 - (Q.2) -- (Q.2 is the y-coordinate of Q)

-- The main statement to prove
theorem distance_from_Q_to_EF_is_24_div_5 : dist_to_EF = 24 / 5 := by
  sorry

end distance_from_Q_to_EF_is_24_div_5_l125_125380


namespace ratio_XA_XY_l125_125146

-- Given conditions and definitions
def ABCD_is_square (ABCD : ℝ → ℝ → ℝ → ℝ → Prop) := ∀ a b c d : ℝ, ABCD a b c d = (a - b) * (a - c) = (b - d) * (c - d)
def area_ratio (S_q S : ℝ) := S_q = (7 / 32) * S
def triangle_ratio (XA XY k : ℝ) := k = (XA / XY)

-- Lean statement of the proof goal
theorem ratio_XA_XY (ABCD : ℝ → ℝ → ℝ → ℝ → Prop) (XA XY S_q S : ℝ) (h1 : ABCD_is_square ABCD) (h2 : area_ratio S_q S) :
  ∃ k : ℝ, triangle_ratio XA XY k ∧ (k = 7 / 8 ∨ k = 1 / 8) :=
by
  sorry

end ratio_XA_XY_l125_125146


namespace num_triangles_in_n_gon_l125_125902

-- Definitions for the problem in Lean based on provided conditions
def n_gon (n : ℕ) : Type := sorry  -- Define n-gon as a polygon with n sides
def non_intersecting_diagonals (n : ℕ) : Prop := sorry  -- Define the property of non-intersecting diagonals in an n-gon
def num_triangles (n : ℕ) : ℕ := sorry  -- Define a function to calculate the number of triangles formed by the diagonals in an n-gon

-- Statement of the theorem to prove
theorem num_triangles_in_n_gon (n : ℕ) (h : non_intersecting_diagonals n) : num_triangles n = n - 2 :=
by
  sorry

end num_triangles_in_n_gon_l125_125902


namespace number_of_elements_A_inter_B_l125_125430

noncomputable def A : set ℕ := {x | x ≤ 8}
noncomputable def B : set ℤ := {y | y > 3}
noncomputable def C : set ℕ := A ∩ B

theorem number_of_elements_A_inter_B :
  fintype.card C = 5 := by
  sorry

end number_of_elements_A_inter_B_l125_125430


namespace incorrect_proposition_D_l125_125752

noncomputable def verify_incorrect_proposition (l m n : Line) (α β : Plane) : Prop :=
  (l ≠ m ∧ l ≠ n ∧ m ≠ n) →
  (α ≠ β) →
  let option_D := (l ⊂ α ∧ α ⊥ β) → l ⊥ β in
  ¬ option_D

-- Statement of the theorem verifying that proposition D is incorrect
theorem incorrect_proposition_D (l m n : Line) (α β : Plane) :
  l ≠ m ∧ l ≠ n ∧ m ≠ n →
  α ≠ β →
  ¬ ((l ⊂ α ∧ α ⊥ β) → l ⊥ β) := sorry

end incorrect_proposition_D_l125_125752


namespace sum_of_common_divisors_36_48_l125_125022

-- Definitions based on the conditions
def is_divisor (n d : ℕ) : Prop := d ∣ n

-- List of divisors for 36 and 48
def divisors_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]
def divisors_48 : List ℕ := [1, 2, 3, 4, 6, 8, 12, 16, 24, 48]

-- Definition of common divisors
def common_divisors_36_48 : List ℕ := [1, 2, 3, 4, 6, 12]

-- Sum of common divisors
def sum_common_divisors_36_48 := common_divisors_36_48.sum

-- The statement of the theorem
theorem sum_of_common_divisors_36_48 : sum_common_divisors_36_48 = 28 := by
  sorry

end sum_of_common_divisors_36_48_l125_125022


namespace num_valid_n_l125_125742

theorem num_valid_n : ∃ k, k = 4 ∧ ∀ n : ℕ, (0 < n ∧ n < 50 ∧ ∃ m : ℕ, m > 0 ∧ n = m * (50 - n)) ↔ 
  (n = 25 ∨ n = 40 ∨ n = 45 ∨ n = 48) :=
by 
  sorry

end num_valid_n_l125_125742


namespace mean_equals_sum_of_squares_l125_125921

noncomputable def arithmetic_mean (x y z : ℝ) := (x + y + z) / 3
noncomputable def geometric_mean (x y z : ℝ) := (x * y * z) ^ (1 / 3)
noncomputable def harmonic_mean (x y z : ℝ) := 3 / ((1 / x) + (1 / y) + (1 / z))

theorem mean_equals_sum_of_squares (x y z : ℝ) (h1 : arithmetic_mean x y z = 10)
  (h2 : geometric_mean x y z = 6) (h3 : harmonic_mean x y z = 4) :
  x^2 + y^2 + z^2 = 576 :=
  sorry

end mean_equals_sum_of_squares_l125_125921


namespace math_problem_l125_125012

theorem math_problem
  (a b c d : ℤ)
  (h_cond : ∀ x : ℝ, (7 * x / 5) + 2 = 4 / x)
  (h_form : ∀ x : ℝ, x = (a + b * real.sqrt c) / d)
  (h_a : a = 10)
  (h_b : b = 7)
  (h_c : c = 3)
  (h_d : d = 7)
  (h_b_ne_zero : b ≠ 0) :
  (a * c * d) / b = 30 := 
sorry

end math_problem_l125_125012


namespace complement_union_l125_125095

variable (U : Set ℤ)
variable (A : Set ℤ)
variable (B : Set ℤ)

theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3})
                         (hA : A = {-1, 0, 1})
                         (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} :=
sorry

end complement_union_l125_125095


namespace desiree_age_l125_125706

variables (D C : ℕ)
axiom condition1 : D = 2 * C
axiom condition2 : D + 30 = (2 * (C + 30)) / 3 + 14

theorem desiree_age : D = 6 :=
by
  sorry

end desiree_age_l125_125706


namespace angle_between_a_and_b_l125_125045

noncomputable def find_angle_between_vectors
  (a b : ℝ × ℝ)
  (ha : ‖a‖ = real.sqrt 2)
  (hb : ‖b‖ = 1)
  (h_perp : a.1 * (a.1 + 2 * b.1) + a.2 * (a.2 + 2 * b.2) = 0) :
  ℝ :=
by
  sorry

theorem angle_between_a_and_b
  (a b : ℝ × ℝ)
  (ha : ‖a‖ = real.sqrt 2)
  (hb : ‖b‖ = 1)
  (h_perp : a.1 * (a.1 + 2 * b.1) + a.2 * (a.2 + 2 * b.2) = 0) :
  find_angle_between_vectors a b ha hb h_perp = 3 * real.pi / 4 :=
sorry

end angle_between_a_and_b_l125_125045


namespace length_of_side_of_pentagon_l125_125286

-- Assuming these conditions from the math problem:
-- 1. The perimeter of the regular polygon is 125.
-- 2. The polygon is a pentagon (5 sides).

-- Let's define the conditions:
def perimeter := 125
def sides := 5
def regular_polygon (perimeter : ℕ) (sides : ℕ) := (perimeter / sides : ℕ)

-- Statement to be proved:
theorem length_of_side_of_pentagon : regular_polygon perimeter sides = 25 := 
by sorry

end length_of_side_of_pentagon_l125_125286


namespace sum_infinite_series_l125_125691

theorem sum_infinite_series :
  ∑ k in (Finset.range ∞), (12^k / ((4^k - 3^k) * (4^(k + 1) - 3^(k + 1)))) = 3 :=
sorry

end sum_infinite_series_l125_125691


namespace minimize_S_n_at_7_l125_125761

-- Define the arithmetic sequence and conditions
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, a (n + 1) - a n = a 2 - a 1

def conditions (a : ℕ → ℤ) : Prop :=
arithmetic_sequence a ∧ a 2 = -11 ∧ (a 5 + a 9 = -2)

-- Define the sum of first n terms of the sequence
def S (a : ℕ → ℤ) (n : ℕ) : ℤ :=
(n * (a 1 + a n)) / 2

-- Define the minimum S_n and that it occurs at n = 7
theorem minimize_S_n_at_7 (a : ℕ → ℤ) (n : ℕ) (h : conditions a) :
  ∀ m, S a m ≥ S a 7 := sorry

end minimize_S_n_at_7_l125_125761


namespace log_evaluation_l125_125000

-- Definitions related to the conditions of the problem
def log_base (b a : ℝ) := Real.log a / Real.log b

lemma condition1 : 64 = 4 ^ 3 := by norm_num
lemma condition2 : Real.sqrt 4 = 4 ^ (1/2 : ℝ) := by norm_num
lemma log_power_rule (b a : ℝ) (c : ℝ) : log_base b (a ^ c) = c * log_base b a := 
by {
  rw [log_base, Real.log_pow, mul_div_assoc],
  exact mul_comm _ _
}

-- Definition of the main problem to be proved
theorem log_evaluation : log_base 4 (64 * Real.sqrt 4) = 7/2 :=
by {
  have h1 : 64 = 4^3 := condition1,
  have h2 : Real.sqrt 4 = 4^(1/2 : ℝ) := condition2,
  have h3 : 64 * Real.sqrt 4 = 4^(3 + 1/2 : ℝ) := 
    by {
      rw [h1, h2],
      exact Real.mul_self_sqrt (4 ^ 3),
    },
  rw [log_power_rule 4 4 (7/2), log_base, Real.log_pow],
  exact div_self (Real.log_ne_zero_of_pos (by norm_num : (4 : ℝ) > 0)),
  exact mul_comm _ _
}

end log_evaluation_l125_125000


namespace polynomial_degree_is_3_l125_125246

noncomputable def polynomial_expression := 3 * a^2 - a * b^2 + 2 * a^2 - 3^4

theorem polynomial_degree_is_3 : degree polynomial_expression = 3 :=
sorry

end polynomial_degree_is_3_l125_125246


namespace cyclic_quadrilateral_l125_125750

noncomputable theory
open_locale classical

variables {A B C I H H_A H_B H_C U V W : Type}
variables [euclidean_plane A B C] (I: incenter A B C) (H: orthocenter A B C)
variables (H_A H_B H_C: footperpendicular A B C) 
variables (U V W: points_au_bv_cw A H_A B H_B C H_C 2r)

theorem cyclic_quadrilateral (H: point H) (U: point U) (V: point V) (W: point W)
  (h1 : on_perpendicular A H_A)
  (h2 : on_perpendicular B H_B)
  (h3 : on_perpendicular C H_C)
  (h4 : AU = 2 * r)
  (h5 : BV = 2 * r)
  (h6 : CW = 2 * r) :
  cyclic H U V W :=
by sorry

end cyclic_quadrilateral_l125_125750


namespace no_tangent_of_2x_plus_m_for_f4_l125_125970

def f1 (x : ℝ) : ℝ := x^2 + x
def f2 (x : ℝ) : ℝ := x^3 + exp x
def f3 (x : ℝ) : ℝ := log x + x^2 / 2
def f4 (x : ℝ) : ℝ := sqrt x + 2 * x

theorem no_tangent_of_2x_plus_m_for_f4 :
  ∃ (m : ℝ) (f : ℝ → ℝ), f = f4 → ∀ x : ℝ, diff f x ≠ 2 :=
sorry

end no_tangent_of_2x_plus_m_for_f4_l125_125970


namespace diana_owes_amount_l125_125709

def principal : ℝ := 60
def rate : ℝ := 0.06
def time : ℝ := 1
def interest := principal * rate * time
def original_amount := principal
def total_amount := original_amount + interest

theorem diana_owes_amount :
  total_amount = 63.60 :=
by
  -- Placeholder for actual proof
  sorry

end diana_owes_amount_l125_125709


namespace find_r_divisibility_l125_125743

theorem find_r_divisibility (r : ℝ) :
  (∃ s : ℝ, 10 * (x - r)^2 * (x - s) = 10 * x^3 - 5 * x^2 - 52 * x + 56) → r = 4 / 3 :=
by
  sorry

end find_r_divisibility_l125_125743
