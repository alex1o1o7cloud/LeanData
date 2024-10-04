import Mathlib
import Mathlib.Algebra.Combinatorics.Perm
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Order
import Mathlib.Algebra.Trig
import Mathlib.Combinatorics.CombinatorialAiden
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype
import Mathlib.Data.Int.Basic
import Mathlib.Data.List
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.LinearAlgebra.FinVec
import Mathlib.NumberTheory.PerfectSquare
import Mathlib.Tactic
import data.nat.combinatorics

namespace analysis_of_f_and_g_l620_620953

noncomputable def f (x : ℝ) : ℝ :=
  2 * sin ((1 / 3) * x + π / 6)

noncomputable def g (x : ℝ) : ℝ :=
  2 * sin ((4 / 3) * x - 5 * π / 18)

theorem analysis_of_f_and_g :
  (∀ x, f x = 2 * sin ((1 / 3) * x + π / 6)) ∧
  (∀ x, g x = 2 * sin ((4 / 3) * x - 5 * π / 18)) :=
by
  sorry

end analysis_of_f_and_g_l620_620953


namespace fixed_points_circle_l620_620141

theorem fixed_points_circle (m : ℝ) : 
  (√5)^2 + (2 * √5)^2 + 2 * m * (√5) - m * (2 * √5) - 25 = 0 ∧
  (-√5)^2 + (-2 * √5)^2 + 2 * m * (-√5) - m * (-2 * √5) - 25 = 0 := 
by sorry

end fixed_points_circle_l620_620141


namespace ab_equals_six_l620_620039

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end ab_equals_six_l620_620039


namespace sum_first_13_terms_l620_620582

variable {α : Type*} [AddCommGroup α] [Module ℤ α]
variable (a : ℕ → α)

def arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem sum_first_13_terms
  (d : α) (h : arithmetic_sequence a d) 
  (H : a 5 + a 6 + a 7 = 9) :
  let a_1 := a 0,
      a_6 := a 5,
      a_7 := a 6,
      a_13 := a_1 + 12 * d in
  (13 / 2 * (a_1 + a_13) : α) = 39 :=
by
  sorry

end sum_first_13_terms_l620_620582


namespace lcm_of_two_numbers_l620_620837

theorem lcm_of_two_numbers :
  ∀ (a b : ℕ), a = 48 → b = 54 → lcm a b = 432 :=
begin
  intros a b h1 h2,
  rw [h1, h2],
  exact lcm_fresh 48 54 sorry,
end

end lcm_of_two_numbers_l620_620837


namespace cat_food_sufficiency_l620_620515

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end cat_food_sufficiency_l620_620515


namespace magic_square_exists_l620_620457

theorem magic_square_exists : 
  ∃ (a b c d e f g h : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ 
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ 
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ 
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ 
    f ≠ g ∧ f ≠ h ∧
    g ≠ h ∧
    a + b + c = 12 ∧ d + e + f = 12 ∧ g + h + 0 = 12 ∧
    a + d + g = 12 ∧ b + 0 + h = 12 ∧ c + f + 0 = 12 :=
sorry

end magic_square_exists_l620_620457


namespace point_in_second_quadrant_l620_620978

-- Assuming A and B are acute angles of triangle ABC
variables {A B : ℝ}

-- Definition of triangle ABC being acute implies A + B > π / 2
def acute_triangle (A B : ℝ) : Prop :=
  A + B > π / 2

-- Point P defined as (cos B - sin A, sin B - cos A)
def P (A B : ℝ) := (real.cos B - real.sin A, real.sin B - real.cos A)

-- Proof problem statement: Show that P is in the second quadrant
theorem point_in_second_quadrant (ha : 0 < A) (hb : A < π / 2)
  (hc : 0 < B) (hd : B < π / 2) (h_acute : acute_triangle A B) :
  let P := P A B in
  (fst P < 0) ∧ (snd P > 0) :=
sorry

end point_in_second_quadrant_l620_620978


namespace ab_value_l620_620012

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_value_l620_620012


namespace car_speeds_meet_halfway_l620_620706

theorem car_speeds_meet_halfway
  (distance_between : ℕ) (departure_time_car1 : ℕ) (departure_time_car2 : ℕ)
  (speed_car1 : ℕ) :
  ∀ (time_of_meeting : ℕ), 
  distance_between = 600 →
  departure_time_car1 = 7 →
  departure_time_car2 = 8 →
  speed_car1 = 50 →
  (2 * (time_of_meeting * speed_car2)) = distance_between →
  time_of_meeting = 13 →
  speed_car2 = 60 :=
begin
  intros,
  sorry
end

end car_speeds_meet_halfway_l620_620706


namespace right_triangle_AB_EQ_3sqrt13_l620_620089

open Real

noncomputable def AB (A B C : ℝ × ℝ) : ℝ :=
  dist A B

theorem right_triangle_AB_EQ_3sqrt13
  {A B C : ℝ × ℝ}
  (AC_val : dist A C = 6)
  (tan_A_val : tan (angle B A C) = 3 / 2)
  (right_angle : angle A B C = π / 2) :
  AB A B C = 3 * sqrt 13 :=
  sorry

end right_triangle_AB_EQ_3sqrt13_l620_620089


namespace marbles_total_l620_620895

theorem marbles_total (fabian kyle miles : ℕ) (h1 : fabian = 3 * kyle) (h2 : fabian = 5 * miles) (h3 : fabian = 15) : kyle + miles = 8 := by
  sorry

end marbles_total_l620_620895


namespace arranging_plants_in_a_row_l620_620850

/-
  April has four different basil plants, four different tomato plants, and two different pepper plants. 
  In how many ways can she arrange the plants in a row if all tomato plants are grouped together, 
  and all pepper plants are also grouped together?
-/

def num_plants := 4 + 4 + 2 -- total number of individual plants (though we group tomatoes and peppers)

-- The basil plants
def num_basil := 4 

-- The tomato plants
def num_tomato := 4 

-- The pepper plants
def num_pepper := 2 

-- The total number of ways to arrange plants with given conditions
def total_ways := factorial 6 * factorial num_tomato * factorial num_pepper

theorem arranging_plants_in_a_row :
  total_ways = 34560 :=
by 
  sorry

end arranging_plants_in_a_row_l620_620850


namespace greatest_three_digit_multiple_of_17_l620_620301

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 17 ∣ n :=
by 
  sorry

end greatest_three_digit_multiple_of_17_l620_620301


namespace probability_calculation_l620_620200

noncomputable def p (n : ℕ) (a : ℝ) : ℝ := 
  if h : n ∈ {1, 2, 3, 4} then a / ((n + 1) * (n + 2)) else 0

theorem probability_calculation (a : ℝ) :
  (p 1 a + p 2 a + p 3 a + p 4 a = 1) →
  (∀ x, p x a = if x ∈ {1, 2, 3, 4} then a / ((x + 1) * (x + 2)) else 0) →
  P(1, 2 a (λ X, p X a) = ¾ :=
begin
  sorry
end

end probability_calculation_l620_620200


namespace sum_series_75_to_99_l620_620479

theorem sum_series_75_to_99 : 
  let a := 75
  let l := 99
  let n := l - a + 1
  let s := n * (a + l) / 2
  s = 2175 :=
by
  sorry

end sum_series_75_to_99_l620_620479


namespace max_points_for_top_four_teams_l620_620997

-- Definitions based on the conditions
def league : Type := { teams : Fin 8 // True }
def points_for_win : ℕ := 3
def points_for_draw : ℕ := 2
def points_for_loss : ℕ := 0
def total_teams : ℕ := 8
def matches_per_pair : ℕ := 3
def total_matches : ℕ := 63
def total_points_distributed : ℕ := 189

-- Problem statement
theorem max_points_for_top_four_teams : 
  ∃ (p : ℕ), p = 54 ∧ (∃ (A B C D : league), (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ A ≠ C ∧ A ≠ D ∧ B ≠ D) ∧ 
  (∀ x ∈ {A, B, C, D}, x.points = p)) :=
begin
  sorry
end

end max_points_for_top_four_teams_l620_620997


namespace product_even_l620_620524

theorem product_even (a : Fin 7 → Fin 7) (hperm : Function.Bijective a) :
  (∏ i, (a i).val - i.val) % 2 = 0 :=
by 
  sorry

end product_even_l620_620524


namespace shifted_roots_polynomial_l620_620127

def original_polynomial (x : ℝ) : ℝ := x^3 - 4 * x^2 + 9 * x - 7

theorem shifted_roots_polynomial :
  ∀ (p q r : ℝ),
  -- Conditions: p, q and r are roots of the original polynomial
  original_polynomial p = 0 → original_polynomial q = 0 → original_polynomial r = 0 →
  -- Statement: the polynomial with roots p + 3, q + 3, and r + 3 is x^3 - 13x^2 + 60x - 90
  ∀ (x : ℝ), (x - (p + 3)) * (x - (q + 3)) * (x - (r + 3)) = x^3 - 13 * x^2 + 60 * x - 90 := 
by {
  intros p q r hp hq hr x,
  sorry
}

end shifted_roots_polynomial_l620_620127


namespace age_difference_l620_620415

theorem age_difference
  (A B C : ℕ)
  (h1 : B = 12)
  (h2 : B = 2 * C)
  (h3 : A + B + C = 32) :
  A - B = 2 :=
sorry

end age_difference_l620_620415


namespace ab_equals_six_l620_620047

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end ab_equals_six_l620_620047


namespace expression_evaluates_to_minus_99_l620_620894

-- Define the expression given the conditions in the problem statement.
def expr := -( (18 / 3 * 11) - (48 / 4) + (5 * 9) )

-- State the theorem that we need to prove: the evaluated expression equals -99.
theorem expression_evaluates_to_minus_99 : expr = -99 :=
by
  have term1 : 18 / 3 * 11 = 66 := by sorry
  have term2 : 48 / 4 = 12 := by sorry
  have term3 : 5 * 9 = 45 := by sorry
  calc
    expr = -( (18 / 3 * 11) - (48 / 4) + (5 * 9) ) : rfl
      ... = -( 66 - 12 + 45 ) : by rw [term1, term2, term3]
      ... = -( 54 + 45 ) : by sorry
      ... = -99 : by sorry

end expression_evaluates_to_minus_99_l620_620894


namespace parabola_passes_through_0_0_l620_620468

theorem parabola_passes_through_0_0 : 
  ∀ (x : ℝ), (3 * x ^ 2 = 0) ↔ (x = 0) :=
by
  intro x
  constructor
  · intro h
    simp at h
    exact eq_zero_of_mul_eq_self_right _ h
  · intro h
    rw h
    simp

end parabola_passes_through_0_0_l620_620468


namespace greatest_three_digit_multiple_of_17_l620_620296

def is_multiple_of (n : ℕ) (k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, is_three_digit n ∧ is_multiple_of n 17 → n = 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620296


namespace bottle_caps_original_l620_620156

theorem bottle_caps_original (removed left : ℕ) (h_removed : removed = 47) (h_left : left = 40) : (removed + left = 87) :=
by
  rw [h_removed, h_left]
  exact rfl

end bottle_caps_original_l620_620156


namespace lambda_value_l620_620211

theorem lambda_value (λ : ℝ) : 
  let a := (λ, 1)
  let b := (1, -2)
  (a.1 * b.1 + a.2 * b.2 = 0) → λ = 2 :=
by 
  intro h 
  sorry

end lambda_value_l620_620211


namespace domain_of_g_l620_620904

open Real

noncomputable def g (x : ℝ) : ℝ :=
  sqrt (1 - sqrt (2 - sqrt (3 - sqrt (4 - x))))

theorem domain_of_g : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 ↔ ∃ y : ℝ, g y = g x :=
by
  sorry

end domain_of_g_l620_620904


namespace power_of_q_l620_620635

theorem power_of_q (p q : ℕ) (n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hdiv : Nat.numDivisors (p^2 * q^n) = 18) :
  n = 5 := by
  sorry

end power_of_q_l620_620635


namespace sufficient_food_supply_l620_620502

variable {L S : ℝ}

theorem sufficient_food_supply (h1 : L + 4 * S = 14) (h2 : L > S) : L + 3 * S ≥ 11 :=
by
  sorry

end sufficient_food_supply_l620_620502


namespace obtuse_triangle_area_l620_620793

theorem obtuse_triangle_area (a b : ℝ) (C : ℝ) (h₁ : a = 8) (h₂ : b = 12) (h₃ : C = 150 * (Real.pi / 180))
  : (1/2 * a * b * Real.sin C) = 24 := 
by
  rw [h₁, h₂, h₃]
  have sin_C : Real.sin (150 * (Real.pi / 180)) = 1/2 := sorry -- This can be obtained from trigonometric identity
  rw sin_C
  norm_num

end obtuse_triangle_area_l620_620793


namespace greatest_three_digit_multiple_of_17_l620_620334

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

theorem greatest_three_digit_multiple_of_17 : ∀ n, n < 1000 → is_multiple_of n 17 → n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620334


namespace fixed_points_circle_l620_620140

theorem fixed_points_circle (m : ℝ) : 
  (√5)^2 + (2 * √5)^2 + 2 * m * (√5) - m * (2 * √5) - 25 = 0 ∧
  (-√5)^2 + (-2 * √5)^2 + 2 * m * (-√5) - m * (-2 * √5) - 25 = 0 := 
by sorry

end fixed_points_circle_l620_620140


namespace ab_equals_six_l620_620025

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_l620_620025


namespace cat_food_problem_l620_620499

noncomputable def L : ℝ
noncomputable def S : ℝ

theorem cat_food_problem (L S : ℝ) (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
by 
  sorry

end cat_food_problem_l620_620499


namespace greatest_three_digit_multiple_of_17_l620_620318

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → m ≤ n) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620318


namespace product_of_ab_l620_620058

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end product_of_ab_l620_620058


namespace cat_food_problem_l620_620494

noncomputable def L : ℝ
noncomputable def S : ℝ

theorem cat_food_problem (L S : ℝ) (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
by 
  sorry

end cat_food_problem_l620_620494


namespace greatest_three_digit_multiple_of_17_l620_620374

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 100 ≤ n ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l620_620374


namespace geometric_sequence_seventh_term_eq_l620_620906

theorem geometric_sequence_seventh_term_eq :
  ∀ a₁ a₂ : ℝ, a₁ = 3 → a₂ = -3 / 2 →
  let r := a₂ / a₁ in
  let a₇ := a₁ * r^(6) in
  a₇ = 3 / 64 :=
by
  intros a₁ a₂ h₁ h₂
  let r := a₂ / a₁
  let a₇ := a₁ * r^(6)
  have ha₁ : a₁ = 3 := h₁
  have ha₂ : a₂ = -3 / 2 := h₂
  rw [ha₁, ha₂] at *
  sorry

end geometric_sequence_seventh_term_eq_l620_620906


namespace compare_integrals_l620_620918

noncomputable def S1 : ℝ := ∫ x in 1..2, x
noncomputable def S2 : ℝ := ∫ x in 1..2, exp x
noncomputable def S3 : ℝ := ∫ x in 1..2, x^2

theorem compare_integrals : S1 < S3 ∧ S3 < S2 := 
by
  -- S1 = ∫ x in 1..2, x
  have hS1 : S1 = (1/2) * (2^2 - 1^2) := sorry,
  -- S2 = ∫ x in 1..2, exp x
  have hS2 : S2 = exp 2 - exp 1 := sorry,
  -- S3 = ∫ x in 1..2, x^2
  have hS3 : S3 = (1/3) * (2^3 - 1^3) := sorry,
  -- Calculate numerical values and compare
  have hS1_val : S1 = 3/2 := sorry,
  have hS3_val : S3 = 7/3 := sorry,
  have hS2_val : S2 > 4 := sorry,
  -- Prove comparisons
  have h1 : 3/2 < 7/3 := sorry,
  have h2 : 7/3 < exp 2 - exp 1 := sorry,
  exact ⟨h1, h2⟩

end compare_integrals_l620_620918


namespace trapezoid_APBQ_l620_620119

open IncidenceGeometry

variables {A B C I P Q : Point}
variables [IncidenceGeometry]
variables {circumcircle_AIB : Circle}
variables {line_CA line_CB : Line}

def valid_incenter (A B C I : Point) : Prop := 
  is_incenter I A B C

def circumcircle_meets_line_again (circle : Circle) (line : Line) (P : Point) (A : Point) : Prop := 
  (P ≠ A) ∧ (P ∈ circle ∧ P ∈ line)

theorem trapezoid_APBQ
  (h1 : valid_incenter A B C I)
  (h2 : circumcircle_AIB = Circumcircle A I B)
  (h3 : line_CA = Line C A)
  (h4 : line_CB = Line C B)
  (hP : circumcircle_meets_line_again circumcircle_AIB line_CA P A)
  (hQ : circumcircle_meets_line_again circumcircle_AIB line_CB Q B) :
  ∃ (Q : Point), Q ∈ circumscribedQuad A B P Q ∧ parallel (Line A Q) (Line B P) := 
sorry

end trapezoid_APBQ_l620_620119


namespace greatest_three_digit_multiple_of_17_l620_620297

def is_multiple_of (n : ℕ) (k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, is_three_digit n ∧ is_multiple_of n 17 → n = 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620297


namespace sufficient_food_l620_620493

-- Definitions for a large package and a small package as amounts of food lasting days
def largePackageDay : ℕ := L
def smallPackageDay : ℕ := S

-- Condition: One large package and four small packages last 14 days
axiom condition : largePackageDay + 4 * smallPackageDay = 14

-- Theorem to be proved: One large package and three small packages last at least 11 days
theorem sufficient_food : largePackageDay + 3 * smallPackageDay ≥ 11 :=
by
  sorry

end sufficient_food_l620_620493


namespace mikes_age_is_18_l620_620693

-- Define variables for Mike's age (m) and his uncle's age (u)
variables (m u : ℕ)

-- Condition 1: Mike is 18 years younger than his uncle
def condition1 : Prop := m = u - 18

-- Condition 2: The sum of their ages is 54 years
def condition2 : Prop := m + u = 54

-- Statement: Prove that Mike's age is 18 given the conditions
theorem mikes_age_is_18 (h1 : condition1 m u) (h2 : condition2 m u) : m = 18 :=
by
  -- Proof skipped with sorry
  sorry

end mikes_age_is_18_l620_620693


namespace log_equation_positive_x_l620_620802

theorem log_equation_positive_x (x : ℝ) (hx : 0 < x) (hx1 : x ≠ 1) : 
  (Real.log x / Real.log 2) * (Real.log 9 / Real.log x) = Real.log 9 / Real.log 2 :=
by sorry

end log_equation_positive_x_l620_620802


namespace range_of_m_l620_620956

-- Given function
def f (x : ℝ) : ℝ := x^2 - 1

theorem range_of_m
  (h : ∀ x : ℝ, x ∈ Icc (3/2) (real.top) -> f (x / m) - 4 * m^2 * f x ≤ f (x - 1) + 4 * f m) :
  m ≤ - real.sqrt 3 / 2 ∨ m ≥ real.sqrt 3 / 2 :=
by
  sorry

end range_of_m_l620_620956


namespace largest_three_digit_multiple_of_17_l620_620278

theorem largest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n ≤ 999) ∧ (n ≥ 100) ∧ (17 ∣ n) ∧ ∀ m, ((m > n) ∧ (m ≤ 999) ∧ (m ≥ 100) ∧ (17 ∣ m)) → false :=
by {
  sorry
}

end largest_three_digit_multiple_of_17_l620_620278


namespace range_of_W_l620_620161

def W : Set ℕ := { n | 10 < n ∧ n < 25 ∧ Nat.Prime n }

theorem range_of_W : (set.range W).max - (set.range W).min = 12 := by
  sorry

end range_of_W_l620_620161


namespace cat_food_sufficiency_l620_620509

theorem cat_food_sufficiency (L S : ℝ) (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end cat_food_sufficiency_l620_620509


namespace greatest_three_digit_multiple_of_17_l620_620314

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → m ≤ n) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620314


namespace probability_heads_odd_after_100_flips_l620_620437

def initial_probability_heads : ℚ := 3 / 4
def flips : ℕ := 100
def decrement_heads : ℚ := 1 / 200
def p (i : ℕ) : ℚ := initial_probability_heads - i * decrement_heads

-- Define the recurrence relation function
def recurrence (n : ℕ) (P : ℕ → ℚ) (p : ℕ → ℚ) : ℚ :=
  (1 - 2 * p(n)) * P(n - 1) + p(n)

-- Base case: P_0
def P : ℕ → ℚ
| 0       := 0
| (n + 1) := recurrence (n + 1) P p

theorem probability_heads_odd_after_100_flips : ∃ p100 : ℚ, p100 = P flips :=
begin
  existsi P flips,
  refl,
end

end probability_heads_odd_after_100_flips_l620_620437


namespace jonas_needs_to_buy_socks_to_triple_wardrobe_l620_620101

-- Declare the given conditions as variables
def pairs_of_socks : ℕ := 20
def pairs_of_shoes : ℕ := 5
def pairs_of_pants : ℕ := 10
def tshirts : ℕ := 10
def hats : ℕ := 6
def jackets : ℕ := 4

-- The goal is to prove that to triple the total number of individual items, Jonas needs to buy 80 pairs of socks.
theorem jonas_needs_to_buy_socks_to_triple_wardrobe :
  let total_items := (pairs_of_socks * 2) + (pairs_of_shoes * 2) + pairs_of_pants + tshirts + hats + jackets in
  let target_items := total_items * 3 in
  let additional_items_needed := target_items - total_items in
  let pairs_of_socks_needed := additional_items_needed / 2 in
  pairs_of_socks_needed = 80 :=
by
  sorry

end jonas_needs_to_buy_socks_to_triple_wardrobe_l620_620101


namespace greatest_integer_pi_minus_sqrt_9_l620_620863

def greatest_integer_function (x : ℝ) : ℤ := Int.floor x

theorem greatest_integer_pi_minus_sqrt_9 : greatest_integer_function (Real.pi - Real.sqrt 9) = 0 :=
by
  -- Given π is approximately 3.14
  have h1 : 3 < Real.pi ∧ Real.pi < 4 := sorry -- Assume pi is between 3 and 4
  -- The square root of 9 is 3
  have h2 : Real.sqrt 9 = 3 := by simp [Real.sqrt]
  -- Therefore, 0 < pi - 3 < 1
  have h3 : 0 < Real.pi - 3 ∧ Real.pi - 3 < 1 := by
    split
    case left =>
      linarith [h1.left]
    case right =>
      linarith [h1.right]
  -- Hence, the greatest integer function of (pi - 3) is 0
  exact sorry

end greatest_integer_pi_minus_sqrt_9_l620_620863


namespace john_investment_l620_620671

theorem john_investment (total_interest : ℝ) (first_account : ℝ) (interest_rate_first : ℝ) 
  (interest_rate_diff : ℝ) (interest_first : ℝ) (total_interests_sum : total_interest = 1282) 
  (first_account_investment : first_account = 4000) (interest_rate_first_def : interest_rate_first = 0.11) 
  (interest_rate_diff_def : interest_rate_diff = 0.015) (interest_first_def : interest_first = first_account * interest_rate_first) : 
  ∃ (second_account : ℝ), second_account * (interest_rate_first + interest_rate_diff) = total_interest - interest_first ∧ second_account = 6736 :=
by
  -- Defining the first interest
  have interest_first_calc : 4000 * 0.11 = 440, by norm_num
  
  -- Assuming total interest equation
  assume h : total_interest = interest_first + second_account * (interest_rate_first + interest_rate_diff)
  
  -- Solving for the second account
  use second_account,
  rw [interest_rate_first_def, interest_rate_diff_def],
  rw [show interest_first = 440, from interest_first_calc],
  have second_account_calc : second_account * 0.125 = total_interest - 440, by linarith,
  rw [show total_interest - 440 = 842, by linarith],
  norm_num,
  have final_calc : second_account = 842 / 0.125,
  norm_num,
  exact final_calc

end john_investment_l620_620671


namespace profit_percentage_l620_620203

theorem profit_percentage
    (sale_price_incl_tax : ℚ)
    (tax_rate : ℚ)
    (cost_price : ℚ)
    (hpp: sale_price_incl_tax = 616)
    (htax: tax_rate = 10 / 100)
    (hcp: cost_price = 535.65) :
    ∃ (profit_percentage : ℚ), profit_percentage ≈ 4.54 :=
by
  sorry

end profit_percentage_l620_620203


namespace largest_three_digit_multiple_of_17_l620_620274

theorem largest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n ≤ 999) ∧ (n ≥ 100) ∧ (17 ∣ n) ∧ ∀ m, ((m > n) ∧ (m ≤ 999) ∧ (m ≥ 100) ∧ (17 ∣ m)) → false :=
by {
  sorry
}

end largest_three_digit_multiple_of_17_l620_620274


namespace digital_earth_functionalities_l620_620885

def digital_earth_allows_internet_navigation : Prop := 
  ∀ (f : String), f ∈ ["Receive distance education", "Shop online", "Seek medical advice online"]

def digital_earth_does_not_allow_physical_travel : Prop := 
  ¬ (∀ (f : String), f ∈ ["Travel around the world"])

theorem digital_earth_functionalities :
  digital_earth_allows_internet_navigation ∧ digital_earth_does_not_allow_physical_travel →
  ∀(f : String), f ∈ ["Receive distance education", "Shop online", "Seek medical advice online"] :=
by
  sorry

end digital_earth_functionalities_l620_620885


namespace infinitely_differentiable_function_identity_l620_620900

theorem infinitely_differentiable_function_identity (f : ℝ → ℝ) (hf : ∀ x y : ℝ, f(x + y) = f(x) + f(y) + 2*x*y) :
  (∃ a : ℝ, ∀ x : ℝ, f(x) = x^2 + a * x) :=
by 
  sorry

end infinitely_differentiable_function_identity_l620_620900


namespace triangle_angles_and_area_l620_620661

theorem triangle_angles_and_area 
  (A B C : ℝ) 
  (A_plus_B_eq_5C : A + B = 5 * C)
  (sin_A_minus_C_eq_2sinB : Real.sin (A - C) = 2 * Real.sin B)
  (CM : ℝ) (CM_eq : CM = 2 * Real.sqrt 7)
  (AC AB : ℝ)
  (M_midpoint : M = (AB + AC) / 2) :
  A = 2 * Real.pi / 3 ∧
  let area := (1 / 2) * AB * AC * Real.sin A 
  in area = 4 * Real.sqrt 3 :=
by
  sorry

end triangle_angles_and_area_l620_620661


namespace find_theta_l620_620919

open Real

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b : ℝ × ℝ := (1, -1)

noncomputable def vector_c : ℝ × ℝ := (2 • vector_a.1 + vector_b.1, 2 • vector_a.2 + vector_b.2)
noncomputable def vector_d : ℝ × ℝ := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)

-- Definition of the dot product
noncomputable def dot_prod (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Definition of the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 * v.1 + v.2 * v.2)

-- Definition of cosθ
noncomputable def cos_theta : ℝ := dot_prod vector_c vector_d / (magnitude vector_c * magnitude vector_d)

theorem find_theta : acos cos_theta = π / 4 :=
by
  sorry

end find_theta_l620_620919


namespace sum_abs_complex_roots_eq_8_l620_620541

open Complex Polynomial

noncomputable def sum_abs_roots (p : Polynomial Complex) : ℂ :=
  (root_set p).sum (λ z, abs z)

theorem sum_abs_complex_roots_eq_8 :
  sum_abs_roots (Polynomial.C (20 : ℂ) * Polynomial.X^8 + Polynomial.C (7*Complex.i) * Polynomial.X^7 - Polynomial.C (7*Complex.i) * Polynomial.X + Polynomial.C (20 : ℂ)) = 8 := 
sorry

end sum_abs_complex_roots_eq_8_l620_620541


namespace isosceles_triangle_inscribed_circle_radius_l620_620646

-- Lean structures and definitions for the problem
structure IsoscelesTriangle :=
(base : ℝ)
(height_abcd : ℝ)
(height_bcm : ℝ)

def inscribed_circle_radius (T : IsoscelesTriangle) : ℝ := 
  let ratio_CO_OM := 17 / 15;
  let total_height := T.height_abcd;
  let CO := (17 / (17 + 15)) * total_height;
  let OM := (15 / (17 + 15)) * total_height;
  (base : ℝ) (h_base : T.base = 60) 
  (h_ratio : T.height_abcd / (T.height_bcm) = ratio_CO_OM) : 
  ℝ

theorem isosceles_triangle_inscribed_circle_radius (T : IsoscelesTriangle) 
  (h_base : T.base = 60)
  (h_ratio : T.height_abcd / (T.height_bcm) = 17 / 15) :
  inscribed_circle_radius T = 7.5 :=
sorry

end isosceles_triangle_inscribed_circle_radius_l620_620646


namespace rhombus_diagonal_l620_620175

theorem rhombus_diagonal (d1 d2 area : ℝ) (h1 : d1 = 20) (h2 : area = 160) (h3 : area = (d1 * d2) / 2) :
  d2 = 16 :=
by
  rw [h1, h2] at h3
  linarith

end rhombus_diagonal_l620_620175


namespace problem_one_problem_two_l620_620431

open Classical

noncomputable theory

-- Definitions for the conditions
def D_on_AB (A B D : Point) : Prop := ∃ k, 0 ≤ k ∧ k ≤ 1 ∧ D = (1 - k) • A + k • B
def is_perpendicular (CD : Line) (AB : Line) : Prop := CD ⊥ AB

-- Main Problem (conditions and goals)
theorem problem_one (A B C D : Point) (α β : ℝ) (CD_AB : D_on_AB A B D) 
  (AD_eq_2 : dist A D = 2) (DB_eq_8: dist D B = 8) (angle_ABC : ∠ B A C = α) 
  (angle_CAB : ∠ C A B = β) (perp_CD_AB : is_perpendicular (line_through C D) (line_through A B)) 
  (beta_eq_2alpha : β = 2 * α) : dist C D = 4 * sqrt 2 :=
sorry

theorem problem_two (A B C D : Point) (α β : ℝ) (CD_AB : D_on_AB A B D) 
  (AD_eq_2 : dist A D = 2) (DB_eq_8: dist D B = 8) (angle_ABC : ∠ B A C = α) 
  (angle_CAB : ∠ C A B = β) (alpha_plus_beta : α + β = π / 4) : 
  ∃ M : ℝ, (area_of_triangle A C D ≤ M ∧ M = 5 * (sqrt 2 - 1)) :=
sorry

end problem_one_problem_two_l620_620431


namespace arcsin_equation_solution_l620_620168

theorem arcsin_equation_solution (x : ℝ) (h : Real.arcsin x + Real.arcsin (3 * x) = π / 4) :
  x = sqrt 102 / 51 ∨ x = - sqrt 102 / 51 :=
by
  sorry

end arcsin_equation_solution_l620_620168


namespace second_candidate_more_marks_30_l620_620818

noncomputable def total_marks : ℝ := 600
def passing_marks_approx : ℝ := 240

def candidate_marks (percentage : ℝ) (total : ℝ) : ℝ :=
  percentage * total

def more_marks (second_candidate : ℝ) (passing : ℝ) : ℝ :=
  second_candidate - passing

theorem second_candidate_more_marks_30 :
  more_marks (candidate_marks 0.45 total_marks) passing_marks_approx = 30 := by
  sorry

end second_candidate_more_marks_30_l620_620818


namespace probability_X_eq_Y_l620_620465

def prob_X_eq_Y (X Y : ℝ) : ℝ := 
  if H : (-5 * π ≤ X ∧ X ≤ 5 * π) ∧ (-5 * π ≤ Y ∧ Y ≤ 5 * π) ∧ cos (cos X) = cos (cos Y) 
  then 1/11 
  else 0
  
theorem probability_X_eq_Y :
  ∀ (X Y : ℝ), (-5 * π ≤ X ∧ X ≤ 5 * π) ∧ (-5 * π ≤ Y ∧ Y ≤ 5 * π) → cos (cos X) = cos (cos Y) → prob_X_eq_Y X Y = 1/11 :=
by sorry

end probability_X_eq_Y_l620_620465


namespace sheila_saving_years_l620_620162

theorem sheila_saving_years 
  (initial_amount : ℝ) 
  (monthly_saving : ℝ) 
  (secret_addition : ℝ) 
  (final_amount : ℝ) 
  (years : ℝ) : 
  initial_amount = 3000 ∧ 
  monthly_saving = 276 ∧ 
  secret_addition = 7000 ∧ 
  final_amount = 23248 → 
  years = 4 := 
sorry

end sheila_saving_years_l620_620162


namespace kyle_and_miles_marbles_l620_620897

theorem kyle_and_miles_marbles (f k m : ℕ) 
  (h1 : f = 3 * k) 
  (h2 : f = 5 * m) 
  (h3 : f = 15) : 
  k + m = 8 := 
by 
  sorry

end kyle_and_miles_marbles_l620_620897


namespace ab_equals_six_l620_620037

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end ab_equals_six_l620_620037


namespace ab_eq_six_l620_620069

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_eq_six_l620_620069


namespace total_surface_area_of_pyramid_l620_620838

-- Define the parameters of the problem
def base_length_1 : ℝ := 8
def base_length_2 : ℝ := 6
def height : ℝ := 10

-- Condition for the problem
def rectangular_base : Prop :=
  base_length_1 = 8 ∧ base_length_2 = 6 ∧ height = 10

-- Total surface area calculation to be proven
theorem total_surface_area_of_pyramid :
  rectangular_base →
  (base_length_1 * base_length_2) + 4 * (1 / 2 * (2 * (sqrt (29)) * base_length_2)) = 48 + 24 * sqrt 29 :=
by
  intro h
  sorry

end total_surface_area_of_pyramid_l620_620838


namespace sqrt_factorial_div_l620_620481

theorem sqrt_factorial_div:
  Real.sqrt (↑(Nat.factorial 9) / 90) = 4 * Real.sqrt 42 := 
by
  -- Steps of the proof
  sorry

end sqrt_factorial_div_l620_620481


namespace parametric_to_standard_form_l620_620962

theorem parametric_to_standard_form (t : ℝ) (x y : ℝ)
    (param_eq1 : x = 1 + t)
    (param_eq2 : y = -1 + t) :
    x - y - 2 = 0 :=
sorry

end parametric_to_standard_form_l620_620962


namespace value_of_expression_l620_620852

theorem value_of_expression (x1 x2 x3 x4 x5 x6 x7 : ℝ) 
( h1 : x1 + 4 * x2 + 9 * x3 + 16 * x4 + 25 * x5 + 36 * x6 + 49 * x7 = 1 )
( h2 : 4 * x1 + 9 * x2 + 16 * x3 + 25 * x4 + 36 * x5 + 49 * x6 + 64 * x7 = 8 )
( h3 : 9 * x1 + 16 * x2 + 25 * x3 + 36 * x4 + 49 * x5 + 64 * x6 + 81 * x7 = 81 ) :
  25 * x1 + 36 * x2 + 49 * x3 + 64 * x4 + 81 * x5 + 100 * x6 + 121 * x7 = 425 := 
begin 
  sorry
end

end value_of_expression_l620_620852


namespace greatest_three_digit_multiple_of_17_l620_620375

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 100 ≤ n ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l620_620375


namespace greatest_three_digit_multiple_of_17_l620_620353

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (n % 17 = 0) ∧ ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ (m % 17 = 0) → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620353


namespace smallest_positive_solution_eq_sqrt_29_l620_620796

theorem smallest_positive_solution_eq_sqrt_29 :
  ∃ x : ℝ, 0 < x ∧ x^4 - 58 * x^2 + 841 = 0 ∧ x = Real.sqrt 29 :=
by
  sorry

end smallest_positive_solution_eq_sqrt_29_l620_620796


namespace find_ages_of_siblings_l620_620218

-- Define the ages of the older brother and the younger sister as variables x and y
variables (x y : ℕ)

-- Define the conditions as provided in the problem
def condition1 : Prop := x = 4 * y
def condition2 : Prop := x + 3 = 3 * (y + 3)

-- State that the system of equations defined by condition1 and condition2 is consistent
theorem find_ages_of_siblings (x y : ℕ) (h1 : x = 4 * y) (h2 : x + 3 = 3 * (y + 3)) : 
  (x = 4 * y) ∧ (x + 3 = 3 * (y + 3)) :=
by 
  exact ⟨h1, h2⟩

end find_ages_of_siblings_l620_620218


namespace greatest_three_digit_multiple_of_17_l620_620326

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

theorem greatest_three_digit_multiple_of_17 : ∀ n, n < 1000 → is_multiple_of n 17 → n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620326


namespace greatest_three_digit_multiple_of_17_l620_620246

def is_three_digit (n : Int) : Prop := (n ≥ 100 ∧ n ≤ 999)
def is_multiple_of_17 (n : Int) : Prop := (∃ k : Int, n = 17 * k)

theorem greatest_three_digit_multiple_of_17 : ∀ n : Int, is_three_digit n ∧ is_multiple_of_17 n → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l620_620246


namespace problem1_problem2_l620_620575

section LineEquations

variable {x y : ℝ} (P : ℝ × ℝ) (a b : ℝ)

def line_through_P (P : ℝ × ℝ) (intercept_eq : ℝ) : Prop :=
  ∃ (a : ℝ), (∀ x y, x + y = a) ∧ (P = (2,1)) → a = intercept_eq

def minimum_area_triangle (P : ℝ × ℝ) : Prop :=
  ∃ (a b : ℝ), a * b = 8 ∧ (∀ x, x + 2 * y = 4) ∧ (P = (2,1)) → (a = 4 ∧ b = 2 ∧ (1/2 * a * b) = 4)

theorem problem1 :
  line_through_P (2,1) 3 := 
begin 
  sorry 
end

theorem problem2 :
  minimum_area_triangle (2,1) :=
begin
  sorry 
end

end LineEquations

end problem1_problem2_l620_620575


namespace ab_value_l620_620029

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_value_l620_620029


namespace distance_from_D_to_centroid_ABC_l620_620651

noncomputable def distance_to_centroid (A B C D : ℝ × ℝ × ℝ) : ℝ :=
  let M := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3, (A.3 + B.3 + C.3) / 3)
  in Real.sqrt ((D.1 - M.1) ^ 2 + (D.2 - M.2) ^ 2 + (D.3 - M.3) ^ 2)

theorem distance_from_D_to_centroid_ABC :
  distance_to_centroid (-1, 2, 0) (5, 2, -1) (2, -1, 4) (-2, 2, -1) = Real.sqrt 21 :=
by sorry

end distance_from_D_to_centroid_ABC_l620_620651


namespace frac_equality_l620_620569

variables (a b : ℚ) -- Declare the variables as rational numbers

-- State the theorem with the given condition and the proof goal
theorem frac_equality (h : a / b = 2 / 3) : a / (a + b) = 2 / 5 :=
by
  sorry -- proof goes here

end frac_equality_l620_620569


namespace num_letters_with_line_no_dot_l620_620639

theorem num_letters_with_line_no_dot :
  ∀ (total_letters with_dot_and_line : ℕ) (with_dot_only with_line_only : ℕ),
    (total_letters = 60) →
    (with_dot_and_line = 20) →
    (with_dot_only = 4) →
    (total_letters = with_dot_and_line + with_dot_only + with_line_only) →
    with_line_only = 36 :=
by
  intros total_letters with_dot_and_line with_dot_only with_line_only
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end num_letters_with_line_no_dot_l620_620639


namespace greatest_three_digit_multiple_of_17_l620_620255

def is_three_digit (n : Int) : Prop := (n ≥ 100 ∧ n ≤ 999)
def is_multiple_of_17 (n : Int) : Prop := (∃ k : Int, n = 17 * k)

theorem greatest_three_digit_multiple_of_17 : ∀ n : Int, is_three_digit n ∧ is_multiple_of_17 n → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l620_620255


namespace find_angle_B_find_triangle_area_l620_620660

variable (A B C a b c : ℝ) (triangle_ABC : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
variable (cos_condition : cos B / cos C = -b / (2 * a + c))
variable (b_given : b = Real.sqrt 13) (a_plus_c : a + c = 4)

theorem find_angle_B : 
  B = 2 * π / 3 :=
by
  sorry

theorem find_triangle_area {a c : ℝ} (angle_B : B = 2 * π / 3) (area : ℝ) (h_area : area = 1 / 2 * b * c * sin A) :
  area = 3 * Real.sqrt 3 / 4 :=
by
  sorry

end find_angle_B_find_triangle_area_l620_620660


namespace sufficient_food_l620_620488

-- Definitions for a large package and a small package as amounts of food lasting days
def largePackageDay : ℕ := L
def smallPackageDay : ℕ := S

-- Condition: One large package and four small packages last 14 days
axiom condition : largePackageDay + 4 * smallPackageDay = 14

-- Theorem to be proved: One large package and three small packages last at least 11 days
theorem sufficient_food : largePackageDay + 3 * smallPackageDay ≥ 11 :=
by
  sorry

end sufficient_food_l620_620488


namespace greatest_three_digit_multiple_of_17_l620_620398

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) := 
  exists.intro 986 sorry

end greatest_three_digit_multiple_of_17_l620_620398


namespace average_of_x_values_l620_620070

theorem average_of_x_values (x : ℝ) (h : sqrt (3 * x^2 + 4) = sqrt 28) : 
  (x = 2 * sqrt 2 ∨ x = -2 * sqrt 2) → (2 * sqrt 2 + (-2 * sqrt 2)) / 2 = 0 :=
by
  sorry

end average_of_x_values_l620_620070


namespace person_B_distance_from_A_l620_620149

noncomputable def speed_ratio_A_to_B := 6 / 5
noncomputable def midpoint_distance := 5.0  -- in kilometers

theorem person_B_distance_from_A (total_distance speed_A speed_B t : ℝ)
  (h1: speed_A / speed_B = speed_ratio_A_to_B)
  (h2: 0 < speed_A)
  (h3: 0 < speed_B)
  (h4: total_distance / 2 = midpoint_distance)
  (h5: t = total_distance / (speed_A + speed_B))
  : total_distance - (t * speed_B) = 5 / 3 :=
by
  -- Remaining proof steps go here
  sorry

end person_B_distance_from_A_l620_620149


namespace tom_cars_left_l620_620222

open Nat

theorem tom_cars_left (packages cars_per_package : ℕ) (fraction_given : ℚ) :
  packages = 10 →
  cars_per_package = 5 →
  fraction_given = 1 / 5 →
  2 * (fraction_given * (packages * cars_per_package)) ≤ packages * cars_per_package →
  (packages * cars_per_package) - 2 * (fraction_given * (packages * cars_per_package)).toNat = 30 :=
by
  intros h_packages h_cars_per_package h_fraction_given h_le
  sorry

end tom_cars_left_l620_620222


namespace greatest_three_digit_multiple_of_17_l620_620366

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 100 ≤ n ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l620_620366


namespace cartesian_point_A_ellipse_cartesian_dot_product_range_l620_620655

noncomputable def polar_coordinates (rho theta : ℝ) : ℝ × ℝ :=
  (rho * Math.cos theta, rho * Math.sin theta)

def ellipse_params : (ℝ → ℝ × ℝ) :=
  λ θ, (√3 * Math.cos θ, Math.sin θ)

theorem cartesian_point_A :
  polar_coordinates 2 (Real.pi / 6) = (√3, 1) := 
sorry

theorem ellipse_cartesian :
  ∀ θ, ellipse_params θ = (√3 * Math.cos θ, Math.sin θ) :=
sorry

theorem dot_product_range :
  ∀ θ, 5 - √13 ≤ (-√3, -2).fst * (√3 * Math.cos θ - √3) + (-2) * (Math.sin θ - 1) ∧ 
        (-√3, -2).fst * (√3 * Math.cos θ - √3) + (-2) * (Math.sin θ - 1) ≤ 5 + √13 :=
sorry

end cartesian_point_A_ellipse_cartesian_dot_product_range_l620_620655


namespace distance_from_A_to_y_axis_is_2_l620_620179

-- Define the point A
def point_A : ℝ × ℝ := (-2, 1)

-- Define the distance function to the y-axis
def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  abs p.1

-- The theorem to prove
theorem distance_from_A_to_y_axis_is_2 : distance_to_y_axis point_A = 2 :=
by
  sorry

end distance_from_A_to_y_axis_is_2_l620_620179


namespace find_x_l620_620588

variable (A B : Set ℕ)
variable (x : ℕ)

theorem find_x (hA : A = {1, 3}) (hB : B = {2, x}) (hUnion : A ∪ B = {1, 2, 3, 4}) : x = 4 := by
  sorry

end find_x_l620_620588


namespace tan_sum_pi_div_4_sin_fraction_simplifies_to_1_l620_620940

variable (α : ℝ)
variable (π : ℝ) [Fact (π > 0)]

-- Assume condition
axiom tan_alpha_eq_2 : Real.tan α = 2

-- Goal (1): Prove that tan(α + π/4) = -3
theorem tan_sum_pi_div_4 : Real.tan (α + π / 4) = -3 :=
by
  sorry

-- Goal (2): Prove that (sin(2α) / (sin^2(α) + sin(α) * cos(α) - cos(2α) - 1)) = 1
theorem sin_fraction_simplifies_to_1 :
  (Real.sin (2 * α)) / (Real.sin (α)^2 + Real.sin (α) * Real.cos (α) - Real.cos (2 * α) - 1) = 1 :=
by
  sorry

end tan_sum_pi_div_4_sin_fraction_simplifies_to_1_l620_620940


namespace prove_non_negative_axbycz_l620_620965

variable {a b c x y z : ℝ}

theorem prove_non_negative_axbycz
  (h1 : (a + b + c) * (x + y + z) = 3)
  (h2 : (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) = 4) :
  a * x + b * y + c * z ≥ 0 := 
sorry

end prove_non_negative_axbycz_l620_620965


namespace line_parabola_one_intersection_l620_620988

theorem line_parabola_one_intersection (k : ℝ) : 
  ((∃ (x y : ℝ), y = k * x - 1 ∧ y^2 = 4 * x ∧ (∀ u v : ℝ, u ≠ x → v = k * u - 1 → v^2 ≠ 4 * u)) ↔ (k = 0 ∨ k = 1)) := 
sorry

end line_parabola_one_intersection_l620_620988


namespace compute_G_101_l620_620875

-- Define the sequence G
noncomputable def G : ℕ → ℕ
| 1     := 3
| (n+1) := (3 * G n + 3) / 3

-- Formal statement of the proof problem
theorem compute_G_101 : G 101 = 103 :=
by
  sorry

end compute_G_101_l620_620875


namespace roofs_needed_l620_620100

/-
Problem Statement:
Prove that Jerry needs to shingle 3 roofs given that:
1. Each roof has two slanted rectangular sides measuring 20 feet by 40 feet.
2. He needs 8 shingles to cover one square foot of the roof.
3. He needs a total of 38400 shingles.
-/

def length : ℕ := 20
def width : ℕ := 40
def shingles_per_sqft : ℕ := 8
def total_shingles : ℕ := 38400

theorem roofs_needed (l w sps t_shingles : ℕ) (h1 : l = 20) (h2 : w = 40) (h3 : sps = 8) (h4 : t_shingles = 38400) :
  (t_shingles / (2 * (l * w) * sps)) = 3 :=
by
  rw [h1, h2, h3, h4]
  sorry

end roofs_needed_l620_620100


namespace a_works_alone_in_37_days_l620_620984

variable (Wa Wb Wc : ℝ)

-- Condition: a is thrice as fast as b
def condition1 : Prop := Wa = 3 * Wb

-- Condition: a is twice as fast as c
def condition2 : Prop := Wa = 2 * Wc

-- Condition: Together they can complete the work in 20 days
def condition3 : Prop := Wa + Wb + Wc = 1 / 20

-- Goal: How long for a to do the work alone 
-- Working rate Wa implies Time for a is inverse of working rate
def timeForA : ℝ := 1 / Wa

theorem a_works_alone_in_37_days (h1 : condition1 Wa Wb) 
                                  (h2 : condition2 Wa Wc) 
                                  (h3 : condition3 Wa Wb Wc) :
    timeForA Wa ≈ 37 :=
sorry

end a_works_alone_in_37_days_l620_620984


namespace right_triangle_AB_EQ_3sqrt13_l620_620088

open Real

noncomputable def AB (A B C : ℝ × ℝ) : ℝ :=
  dist A B

theorem right_triangle_AB_EQ_3sqrt13
  {A B C : ℝ × ℝ}
  (AC_val : dist A C = 6)
  (tan_A_val : tan (angle B A C) = 3 / 2)
  (right_angle : angle A B C = π / 2) :
  AB A B C = 3 * sqrt 13 :=
  sorry

end right_triangle_AB_EQ_3sqrt13_l620_620088


namespace find_line_equation_l620_620594

theorem find_line_equation
    (A : Point := ⟨5, 0⟩)
    (P : Point := ⟨2, 1⟩)
    (line1 : ℝ → ℝ → Prop := λ x y, 2 * x + y - 5 = 0)
    (line2 : ℝ → ℝ → Prop := λ x y, x - 2 * y = 0)
    (passes_through_intersection : (∃ x y, line1 x y ∧ line2 x y ∧ l x y))
    (distance_condition : ∀ l : Line, distance_from_point A l = 3) :
  l = λ x y, 4 * x - 3 * y - 5 = 0 ∨ l = λ x y, x = 2 :=
sorry

end find_line_equation_l620_620594


namespace total_payment_is_correct_l620_620534

-- Define the number of friends
def number_of_friends : ℕ := 7

-- Define the amount each friend paid
def amount_per_friend : ℝ := 70.0

-- Define the total amount paid
def total_amount_paid : ℝ := number_of_friends * amount_per_friend

-- Prove that the total amount paid is 490.0
theorem total_payment_is_correct : total_amount_paid = 490.0 := by 
  -- Here, the proof would be filled in
  sorry

end total_payment_is_correct_l620_620534


namespace z_squared_in_second_quadrant_l620_620726
open Complex Real

noncomputable def z : ℂ := exp (π * I / 3)

theorem z_squared_in_second_quadrant : (z^2).re < 0 ∧ (z^2).im > 0 :=
by
  sorry

end z_squared_in_second_quadrant_l620_620726


namespace school_trip_seat_count_l620_620753

theorem school_trip_seat_count :
  ∀ (classrooms students_per_classroom seats_per_bus : ℕ),
  classrooms = 87 →
  students_per_classroom = 58 →
  seats_per_bus = 29 →
  ∀ (total_students total_buses_needed : ℕ),
  total_students = classrooms * students_per_classroom →
  total_buses_needed = (total_students + seats_per_bus - 1) / seats_per_bus →
  seats_per_bus = 29 := by
  intros classrooms students_per_classroom seats_per_bus
  intros h1 h2 h3
  intros total_students total_buses_needed
  intros h4 h5
  sorry

end school_trip_seat_count_l620_620753


namespace sqrt_factorial_div_90_eq_l620_620482

open Real

theorem sqrt_factorial_div_90_eq : sqrt (realOfNat (Nat.factorial 9) / 90) = 24 * sqrt 7 := by
  sorry

end sqrt_factorial_div_90_eq_l620_620482


namespace find_abc_l620_620960

variable {a b c : ℝ}

def parabola_passes_through_1_1 := a + b + c = 1
def parabola_slope_at_2_1 := 4 * a + 2 * b + c = -1
def tangent_slope_condition := 4 * a + b = 1

theorem find_abc 
    (h1 : parabola_passes_through_1_1) 
    (h2 : parabola_slope_at_2_1) 
    (h3 : tangent_slope_condition) : 
    a = 3 ∧ b = -11 ∧ c = 9 := 
by
    sorry

end find_abc_l620_620960


namespace cat_food_sufficiency_l620_620511

theorem cat_food_sufficiency (L S : ℝ) (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end cat_food_sufficiency_l620_620511


namespace susan_spent_total_l620_620147

-- Definitions for the costs and quantities
def pencil_cost : ℝ := 0.25
def pen_cost : ℝ := 0.80
def total_items : ℕ := 36
def pencils_bought : ℕ := 16

-- Question: How much did Susan spend?
theorem susan_spent_total : (pencil_cost * pencils_bought + pen_cost * (total_items - pencils_bought)) = 20 :=
by
    -- definition goes here
    sorry

end susan_spent_total_l620_620147


namespace find_principal_l620_620807

noncomputable def compoundPrincipal (A : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  A / (1 + r / n) ^ (n * t)

theorem find_principal :
  let A := 3969
  let r := 0.05
  let n := 1
  let t := 2
  compoundPrincipal A r n t = 3600 :=
by
  sorry

end find_principal_l620_620807


namespace min_val_at_plus_minus_two_l620_620804

-- Define a fourth-degree polynomial with integer coefficients and a positive leading coefficient
def P (x : ℝ) : ℝ := a * x^4 + b * x^3 + c * x^2 + d * x + e

axiom a_pos : 0 < a
axiom P_3eq5 : P (Real.sqrt 3) = P (Real.sqrt 5)

-- The main theorem stating the minimum values of the polynomial P(x)
theorem min_val_at_plus_minus_two : ∀ (a b c d e : ℝ) (a_pos : 0 < a) (P_3eq5 : P (Real.sqrt 3) = P (Real.sqrt 5)), (∀ x, P x ≥ P 2) ∧ (∀ x, P x ≥ P (-2)) :=
by
  intros a b c d e a_pos P_3eq5
  sorry

end min_val_at_plus_minus_two_l620_620804


namespace total_games_played_l620_620214

-- There are 20 players, each game played by two players, and each player plays each other player exactly once.
theorem total_games_played : ∀ (n : ℕ), n = 20 → C(n, 2) = 190 :=
by {
  intro n,
  intro h,
  rw h,
  simp [nat.choose],
  rfl,
}

end total_games_played_l620_620214


namespace add_neg_two_and_three_l620_620486

theorem add_neg_two_and_three : -2 + 3 = 1 :=
by
  sorry

end add_neg_two_and_three_l620_620486


namespace ab_equals_six_l620_620017

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_l620_620017


namespace greatest_three_digit_multiple_of_17_l620_620395

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) := 
  exists.intro 986 sorry

end greatest_three_digit_multiple_of_17_l620_620395


namespace max_a_if_monotonically_increasing_l620_620185

noncomputable def f (x a : ℝ) : ℝ := x^3 + Real.exp x - a * x

theorem max_a_if_monotonically_increasing (a : ℝ) : 
  (∀ x, 0 ≤ x → 3 * x^2 + Real.exp x - a ≥ 0) ↔ a ≤ 1 :=
by
  sorry

end max_a_if_monotonically_increasing_l620_620185


namespace smallest_k_basis_subset_l620_620576

noncomputable def t (n : ℕ) : ℕ :=
  Nat.find (λ t, (t * t.succ) / 2 < n ∧ n ≤ (t.succ * t.succ.succ) / 2)

def M (n : ℕ) : Set ℤ :=
  {x | -↑n ≤ x ∧ x ≤ ↑n}

def is_basis_subset (n : ℕ) (P : Set ℤ) : Prop :=
  ∀ x ∈ M n, ∃ S ⊆ P, x = S.sum id

theorem smallest_k_basis_subset (n : ℕ) :
  ∃ k, (∀ P : Set ℤ, P.card = k → is_basis_subset n P) ∧ k = n + t n + 1 :=
begin
  sorry
end

end smallest_k_basis_subset_l620_620576


namespace log_det_solution_l620_620939

theorem log_det_solution (x : ℝ) : 
  (log (Real.sqrt 2) (abs (x - 11)) < 0) ↔ (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 2) := 
by 
  sorry

end log_det_solution_l620_620939


namespace compound_interest_rate_l620_620757

theorem compound_interest_rate (SI CI : ℝ) (P1 P2 : ℝ) (T1 T2 : ℝ) (R1 : ℝ) (R : ℝ) 
    (H1 : SI = (P1 * R1 * T1) / 100)
    (H2 : CI = 2 * SI)
    (H3 : CI = P2 * ((1 + R/100)^2 - 1))
    (H4 : P1 = 1272)
    (H5 : P2 = 5000)
    (H6 : T1 = 5)
    (H7 : T2 = 2)
    (H8 : R1 = 10) :
  R = 12 :=
by
  sorry

end compound_interest_rate_l620_620757


namespace largest_three_digit_multiple_of_17_l620_620275

theorem largest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n ≤ 999) ∧ (n ≥ 100) ∧ (17 ∣ n) ∧ ∀ m, ((m > n) ∧ (m ≤ 999) ∧ (m ≥ 100) ∧ (17 ∣ m)) → false :=
by {
  sorry
}

end largest_three_digit_multiple_of_17_l620_620275


namespace probability_blue_now_0_2_l620_620774

variable (totalBalls initialBlueBalls removedBlueBalls : ℕ)
variable (newBlueBalls newTotalBalls : ℕ)

def initial_conditions := (totalBalls = 25) ∧ (initialBlueBalls = 9) ∧ (removedBlueBalls = 5)
def updated_conditions := (newBlueBalls = initialBlueBalls - removedBlueBalls) ∧ (newTotalBalls = totalBalls - removedBlueBalls)
def probability_blue_ball_now := newBlueBalls / newTotalBalls.to_nat.cast

theorem probability_blue_now_0_2 :
  initial_conditions totalBalls initialBlueBalls removedBlueBalls →
  updated_conditions initialBlueBalls removedBlueBalls newBlueBalls totalBalls newTotalBalls →
  probability_blue_ball_now newBlueBalls newTotalBalls = 0.2 := by
  sorry

end probability_blue_now_0_2_l620_620774


namespace product_of_ab_l620_620053

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end product_of_ab_l620_620053


namespace team_selection_ways_l620_620776

/-- There are 6 male athletes and 4 female athletes, including 1 male captain and 1 female captain. 
A team of 5 people is to be selected under these conditions:
- At least 1 female athlete is included.
- The team must include a captain and at least 1 female athlete.
Our goal is to prove there are 191 ways to form such a team. -/
theorem team_selection_ways : 
  let males := 6 in
  let females := 4 in
  let male_captain := 1 in
  let female_captain := 1 in
  (finset.card (finset.filter 
    (λ team, 
      (finset.card (team ∩ finset.range females) ≥ 1) ∧ 
      (male_captain ∈ team ∨ female_captain ∈ team)) 
    (finset.powerset_len 5 (finset.range (males + females)))) = 191) :=
 sorry

end team_selection_ways_l620_620776


namespace linearity_condition_proportionality_condition_l620_620958

def is_linear (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b

def is_proportional (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = a * x

noncomputable def given_function (m n : ℝ) : ℝ → ℝ :=
  λ x, (m+1) * x ^ |m| + n - 3

theorem linearity_condition (m n : ℝ) :
  is_linear (given_function m n) ↔ (m = 1 ∧ ∃ k : ℝ, n = k) :=
sorry

theorem proportionality_condition (m n : ℝ) :
  is_proportional (given_function m n) ↔ (m = 1 ∧ n = 3) :=
sorry

end linearity_condition_proportionality_condition_l620_620958


namespace pos_reals_inequality_l620_620720

theorem pos_reals_inequality {n : ℕ} (a : fin n → ℝ) (h : ∀ i, 0 < a i) :
  (∑ i, a i) * (∑ i, (1 / a i)) ≥ (n : ℝ)^2 :=
by
  sorry

end pos_reals_inequality_l620_620720


namespace rearrangements_count_l620_620930

def rearrangements_with_two_increasing_pairs (n : ℕ) : ℕ :=
  3^n - (n + 1) * (2^n - (n + 1)) - (n + 2).choose 2

theorem rearrangements_count (n : ℕ) :
  rearrangements_with_two_increasing_pairs n = 3^n - (n + 1) * (2^n - (n + 1)) - binom(n + 2, 2) := 
sorry

#inline_bin _ := $(lean 3$n$'').polynomial_exponential_series.lean_binom

#lean 3^n - (n + 1) * (2 ^n - (n+ 1)) - (n +2. choose) 2

end rearrangements_count_l620_620930


namespace largest_three_digit_multiple_of_17_l620_620281

theorem largest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n ≤ 999) ∧ (n ≥ 100) ∧ (17 ∣ n) ∧ ∀ m, ((m > n) ∧ (m ≤ 999) ∧ (m ≥ 100) ∧ (17 ∣ m)) → false :=
by {
  sorry
}

end largest_three_digit_multiple_of_17_l620_620281


namespace greatest_three_digit_multiple_of_17_l620_620303

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 17 ∣ n :=
by 
  sorry

end greatest_three_digit_multiple_of_17_l620_620303


namespace min_people_like_both_l620_620788

theorem min_people_like_both (n m b : ℕ) (h_n : n = 200) (h_m : m = 160) (h_b : b = 145) : 
    ∃ k, k = m + b - n ∧ k = 105 :=
by
  use (m + b - n)
  split
  . refl
  . exact Nat.add_sub_of_le (by linarith)

end min_people_like_both_l620_620788


namespace age_difference_l620_620416

theorem age_difference
  (A B C : ℕ)
  (h1 : B = 12)
  (h2 : B = 2 * C)
  (h3 : A + B + C = 32) :
  A - B = 2 :=
sorry

end age_difference_l620_620416


namespace value_of_m_l620_620543

theorem value_of_m (m : ℝ) : (∀ x : ℝ, (x^2 + 2 * m * x + m > 3 / 16)) ↔ (1 / 4 < m ∧ m < 3 / 4) :=
by sorry

end value_of_m_l620_620543


namespace regular_nonagon_coloring_l620_620212

theorem regular_nonagon_coloring (m n : ℕ) 
  (H1 : m = 3) -- Condition for minimum number of colors needed
  (H2 : n = 18) -- Condition for total number of valid colorings using m colors
  : m * n = 54 :=
by { rw [H1, H2], exact rfl }

end regular_nonagon_coloring_l620_620212


namespace greatest_three_digit_multiple_of_17_l620_620315

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → m ≤ n) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620315


namespace greatest_three_digit_multiple_of_17_l620_620302

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 17 ∣ n :=
by 
  sorry

end greatest_three_digit_multiple_of_17_l620_620302


namespace mat_pow_ne_id_l620_620687
noncomputable theory

variables {A : Matrix (Fin 5) (Fin 5) ℂ}
variable [Invertible (1 - A)]
variable (trA0 : A.trace = 0)

theorem mat_pow_ne_id (A.trace = 0) [invertible (1 - A)] : A ^ 5 ≠ 1 := sorry

end mat_pow_ne_id_l620_620687


namespace sufficient_food_l620_620492

-- Definitions for a large package and a small package as amounts of food lasting days
def largePackageDay : ℕ := L
def smallPackageDay : ℕ := S

-- Condition: One large package and four small packages last 14 days
axiom condition : largePackageDay + 4 * smallPackageDay = 14

-- Theorem to be proved: One large package and three small packages last at least 11 days
theorem sufficient_food : largePackageDay + 3 * smallPackageDay ≥ 11 :=
by
  sorry

end sufficient_food_l620_620492


namespace greatest_3_digit_multiple_of_17_l620_620383

theorem greatest_3_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≥ m) ∧ n = 986 := 
begin
  sorry
end

end greatest_3_digit_multiple_of_17_l620_620383


namespace modified_density_function_example_l620_620914

theorem modified_density_function_example (g fζ : ℝ → ℝ) (n : ℝ) (n_ge_1 : n ≥ 1)
  (h_odd_g : ∀ x, g (-x) = -g x) (h_even_fζ : ∀ x, fζ (-x) = fζ x) :
  (let f := λ x, (g (-x) / 2 + fζ (x) / 2)
   in (∫ x in (-∞ : ℝ)..0, |x|^n * f x) = (∫ x in (0 : ℝ)..∞, |x|^n * f x)) :=
by
  sorry

end modified_density_function_example_l620_620914


namespace g_243_equals_50_l620_620751

def g (x : ℕ) : ℝ := sorry

axiom g_property (x y m : ℕ) (hx : 0 < x) (hy : 0 < y) (hm : 0 < m) (hxy : x + y = 3^m) :
  g(x) + g(y) = 2 * m^2

theorem g_243_equals_50 : g 243 = 50 := by
  sorry

end g_243_equals_50_l620_620751


namespace greatest_three_digit_multiple_of_17_l620_620345

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620345


namespace greatest_three_digit_multiple_of_17_l620_620249

def is_three_digit (n : Int) : Prop := (n ≥ 100 ∧ n ≤ 999)
def is_multiple_of_17 (n : Int) : Prop := (∃ k : Int, n = 17 * k)

theorem greatest_three_digit_multiple_of_17 : ∀ n : Int, is_three_digit n ∧ is_multiple_of_17 n → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l620_620249


namespace find_interest_rate_l620_620695

theorem find_interest_rate
  (P A : ℝ) (n t : ℕ) (r : ℝ)
  (hP : P = 100)
  (hA : A = 121.00000000000001)
  (hn : n = 2)
  (ht : t = 1)
  (compound_interest : A = P * (1 + r / n) ^ (n * t)) :
  r = 0.2 :=
by
  sorry

end find_interest_rate_l620_620695


namespace shaded_area_eq_4_l620_620714

def rectangle_JKLM := {J K L M : (ℝ × ℝ) // 
  J = (0, 0) ∧
  K = (4, 0) ∧
  M = (4, 5) ∧
  L = (0, 5) ∧
  dist J K = 1 ∧ 
  dist K M = 1 ∧
  dist M L = 1}

theorem shaded_area_eq_4 (r : rectangle_JKLM) : 
  let Q := ((2 : ℝ), (2 : ℝ)),
      P := ((4 : ℝ), (4 : ℝ)),
      K := ((4 : ℝ), (0 : ℝ)) in
  1 / 2 * (| (K.1 * P.2 + P.1 * Q.2 + Q.1 * K.2) - (K.2 * P.1 + P.2 * Q.1 + Q.2 * K.1) |) = 4 :=
by
  sorry

end shaded_area_eq_4_l620_620714


namespace square_can_be_divided_into_40_smaller_squares_l620_620808

theorem square_can_be_divided_into_40_smaller_squares 
: ∃ (n : ℕ), n * n = 40 := 
sorry

end square_can_be_divided_into_40_smaller_squares_l620_620808


namespace seating_arrangements_l620_620436

-- Definitions of the conditions:
def democrats : Nat := 7
def republicans : Nat := 5
def total_politicians : Nat := democrats + republicans

-- Theorem statement without the proof.
theorem seating_arrangements : (nat.factorial (total_politicians - 1)) = 39916800 :=
by
  -- The proof is left as an exercise.
  sorry

end seating_arrangements_l620_620436


namespace sum_of_reciprocal_AP_l620_620760

theorem sum_of_reciprocal_AP (a1 a2 a3 : ℝ) (d : ℝ)
  (h1 : a1 + a2 + a3 = 11/18)
  (h2 : 1/a1 + 1/a2 + 1/a3 = 18)
  (h3 : 1/a2 = 1/a1 + d)
  (h4 : 1/a3 = 1/a1 + 2*d) :
  (a1 = 1/9 ∧ a2 = 1/6 ∧ a3 = 1/3) ∨ (a1 = 1/3 ∧ a2 = 1/6 ∧ a3 = 1/9) :=
sorry

end sum_of_reciprocal_AP_l620_620760


namespace sandwich_cost_is_five_l620_620666

-- Define the cost of each sandwich
variables (x : ℝ)

-- Conditions
def jack_orders_sandwiches (cost_per_sandwich : ℝ) : Prop :=
  3 * cost_per_sandwich = 15

-- Proof problem statement (no proof provided)
theorem sandwich_cost_is_five (h : jack_orders_sandwiches x) : x = 5 :=
sorry

end sandwich_cost_is_five_l620_620666


namespace min_value_problem_l620_620113

theorem min_value_problem (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + 3 * b = 1) :
    (1 / a) + (3 / b) ≥ 16 :=
sorry

end min_value_problem_l620_620113


namespace greatest_three_digit_multiple_of_17_l620_620401

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) := 
  exists.intro 986 sorry

end greatest_three_digit_multiple_of_17_l620_620401


namespace de_bruijn_erdos_l620_620122

noncomputable def chromatic_number (G : Type*) [graph G] : ℕ := sorry

variables (V : Type*) (E : set (V × V))

def finite_subgraph (V : Type*) [fintype V] : graph V := sorry

theorem de_bruijn_erdos (G : (V × E) → Type*) (k : ℕ)
  (h : ∀ (G' : Type*) [graph G'],
        (finite G') → chromatic_number G' ≤ k)
    : chromatic_number G ≤ k :=
sorry

end de_bruijn_erdos_l620_620122


namespace ab_value_l620_620009

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_value_l620_620009


namespace sqrt_of_N_base_a_l620_620087

theorem sqrt_of_N_base_a (a n : ℕ) (h_pos : a > 0):
  (let N := (finset.range (2*n-1+1)).sum (λ k, (k+1) * a^(2*n-2-k)) in
  sqrt N = (finset.range n).sum (λ k, a^(n-1-k) + 1)) :=
sorry

end sqrt_of_N_base_a_l620_620087


namespace smaller_cube_volume_in_sphere_in_cube_l620_620456

noncomputable def volume_of_smaller_cube (edge_len : ℝ) : ℝ :=
  let s := (edge_len / (Real.sqrt 3)) in
  s^3

theorem smaller_cube_volume_in_sphere_in_cube :
  volume_of_smaller_cube 16 = (12288 * Real.sqrt 3) / 27 := by
  sorry

end smaller_cube_volume_in_sphere_in_cube_l620_620456


namespace Cora_pages_to_read_on_Thursday_l620_620531

theorem Cora_pages_to_read_on_Thursday
  (total_pages : ℕ)
  (read_monday : ℕ)
  (read_tuesday : ℕ)
  (read_wednesday : ℕ)
  (pages_left : ℕ)
  (read_friday : ℕ)
  (thursday_pages : ℕ) :
  total_pages = 158 →
  read_monday = 23 →
  read_tuesday = 38 →
  read_wednesday = 61 →
  pages_left = total_pages - (read_monday + read_tuesday + read_wednesday) →
  read_friday = 2 * thursday_pages →
  pages_left = thursday_pages + read_friday →
  thursday_pages = 12 :=
begin
  -- Proof is not required
  sorry
end

end Cora_pages_to_read_on_Thursday_l620_620531


namespace greatest_three_digit_multiple_of_17_l620_620362

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (n % 17 = 0) ∧ ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ (m % 17 = 0) → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620362


namespace angle_A_value_perimeter_given_area_perimeter_range_l620_620663

open Real

-- Definitions
variable (a b c : ℝ)
variable (A B C : ℝ) -- Angles in radians

-- Given conditions
def condition1 : Prop := a / cos A = (b + c) / (cos B + cos C)
def condition2 : Prop := a = 3
def condition3 : Prop := (1 / 2) * b * c * sin A = sqrt 3 / 2

-- Proof Problems
theorem angle_A_value (h : condition1 a A b c B C) : A = π / 3 := sorry

theorem perimeter_given_area (h1 : condition1 a A b c B C) (h2 : condition2 a) (h3 : condition3 a b c A) : a + b + c = 3 + sqrt 15 := sorry

theorem perimeter_range (h1 : a = 3) : 6 < a + b + c ∧ a + b + c ≤ 9 := sorry

end angle_A_value_perimeter_given_area_perimeter_range_l620_620663


namespace values_of_m_zero_rain_l620_620665

def f (x y : ℝ) : ℝ := abs (x^3 + 2*x^2*y - 5*x*y^2 - 6*y^3)

theorem values_of_m_zero_rain :
  {m : ℝ | ∀ x : ℝ, f x (m * x) = 0} = {-1, 1/2, -1/3} :=
sorry

end values_of_m_zero_rain_l620_620665


namespace sasha_celebration_min_places_l620_620886

def minimum_celebration_places (n : ℕ) : Prop :=
  ∀ (s : set ℝ^3), (∀ h : set (set ℝ^3), (s ∈ h → (¬ boundary s ∈ h))) →
  (number_of_places (s) == n)

theorem sasha_celebration_min_places :
  minimum_celebration_places 4 
  sorry

end sasha_celebration_min_places_l620_620886


namespace greatest_three_digit_multiple_of_17_l620_620244

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n, (n ∈ set.Icc 100 999) ∧ (n % 17 = 0) ∧ (∀ m, (m ∈ set.Icc 100 999 ∧ m % 17 = 0) → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620244


namespace largest_three_digit_multiple_of_17_l620_620279

theorem largest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n ≤ 999) ∧ (n ≥ 100) ∧ (17 ∣ n) ∧ ∀ m, ((m > n) ∧ (m ≤ 999) ∧ (m ≥ 100) ∧ (17 ∣ m)) → false :=
by {
  sorry
}

end largest_three_digit_multiple_of_17_l620_620279


namespace spelling_bee_contestants_l620_620645

theorem spelling_bee_contestants (n : ℕ) 
  (h1 : 0.025 * n = 15) : 
  n = 600 :=
by sorry

end spelling_bee_contestants_l620_620645


namespace max_f_sequence_l620_620559

noncomputable def f {α : Type} [LinearOrder α] (s : List α) : ℕ :=
  s.pairwise (λ x y, | x - y |)

theorem max_f_sequence (n : ℕ) (h : n > 0) : ∃ s : List ℕ, (s.length = n ∧ s.nodup ∧ s.all (λ x, x ∈ (List.range (n + 1)))) → f s = ⌊n/2⌋ :=
by sorry

end max_f_sequence_l620_620559


namespace room_number_units_digit_l620_620670

/-- A room number is a two-digit number such that exactly three of the following four statements are true:
1. It is divisible by 4.
2. It is odd.
3. The sum of its digits is 12.
4. One of its digits is 8.
Prove that the units digit of the room number is 4. -/
theorem room_number_units_digit : ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧
  (((n % 4 = 0) ∨ (n % 2 = 1) ∨ (n.digits 10).sum = 12 ∨ (8 ∈ n.digits 10)) ∧
  (nat.bodd n = 0 ∨ n % 2 = 1 ∨ (n.digits 10).sum = 12 ∨ (8 ∈ n.digits 10)) ∧
  ((n.digits 10).sum = 12 ∨ n % 2 = 1 ∨ (n % 4 = 0) ∨ (8 ∈ n.digits 10)) ∧
  (8 ∈ n.digits 10 ∨ n % 2 = 1 ∨ (n.digits 10).sum = 12 ∨ (n % 4 = 0)) ∧
  ¬(((n % 4 = 0) ∧ (n % 2 = 1) ∧ (n.digits 10).sum = 12 ∧ (8 ∈ n.digits 10)) ∨
     ((n % 4 != 0) ∧ (n % 2 != 1) ∧ (n.digits 10).sum != 12 ∧ (8 ∉ n.digits 10)))) ∧
  (n % 10 = 4) :=
sorry

end room_number_units_digit_l620_620670


namespace greatest_three_digit_multiple_of_17_l620_620350

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (n % 17 = 0) ∧ ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ (m % 17 = 0) → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620350


namespace sum_of_first_9_terms_l620_620946

variable {a : ℕ → ℤ} -- Define the arithmetic sequence
variable {S : ℕ → ℤ} -- Define the sum of the first n terms of the sequence

-- Define the conditions in Lean
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n

def sum_of_first_n_terms (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
∀ n, S n = (n * (a 1 + a n)) / 2

def condition_a2 : a 2 = -2 :=
sorry

def condition_a8 : a 8 = 6 :=
sorry

-- Define the theorem to prove the sum of the first 9 terms
theorem sum_of_first_9_terms (h_arith_seq : is_arithmetic_sequence a)
    (h_sum_terms : sum_of_first_n_terms S a)
    (h_a2 : condition_a2)
    (h_a8 : condition_a8) :
  S 9 = 18 :=
sorry

end sum_of_first_9_terms_l620_620946


namespace last_digit_appears_l620_620527

def modified_fibonacci : ℕ → ℕ 
| 0 := 2
| 1 := 2
| (n + 2) := (modified_fibonacci (n + 1) + modified_fibonacci n) % 7

theorem last_digit_appears :
  ∃ n, ∀ d ∈ list.range 7, ∃ m ≤ n, (modified_fibonacci m % 7 = d) ∧ 
    ∀ i < n, modified_fibonacci i % 7 ≠ 0 :=
sorry

end last_digit_appears_l620_620527


namespace greatest_three_digit_multiple_of_17_l620_620245

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n, (n ∈ set.Icc 100 999) ∧ (n % 17 = 0) ∧ (∀ m, (m ∈ set.Icc 100 999 ∧ m % 17 = 0) → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620245


namespace triangle_area_l620_620756

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem triangle_area (a b c : ℕ) (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) (h4 : is_right_triangle a b c) :
  (1 / 2 : ℝ) * a * b = 180 :=
by sorry

end triangle_area_l620_620756


namespace triangle_area_l620_620204

/-- Proof that the area of a triangle with side lengths 9 cm, 40 cm, and 41 cm is 180 square centimeters, 
    given that these lengths form a right triangle. -/
theorem triangle_area : ∀ (a b c : ℕ), a = 9 → b = 40 → c = 41 → a^2 + b^2 = c^2 → (a * b) / 2 = 180 := by
  intros a b c ha hb hc hpyth
  sorry

end triangle_area_l620_620204


namespace greatest_three_digit_multiple_of_17_l620_620373

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 100 ≤ n ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l620_620373


namespace baguette_orderings_l620_620434

theorem baguette_orderings :
  ((Finset.card (Finset.powersetLen 3 (Finset.range 4))) * 3.factorial) = 24 :=
by
  sorry

end baguette_orderings_l620_620434


namespace max_ab_l620_620630

theorem max_ab {a b : ℝ} (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 6) : ab ≤ 9 :=
sorry

end max_ab_l620_620630


namespace vector_parallel_l620_620920

theorem vector_parallel (x : ℝ) : 
  let a := (-1 : ℝ, 2 : ℝ), b := (x, -1) in
  (a.1 * b.2 = a.2 * b.1) -> x = 1 / 2 :=
by
  intro h
  sorry

end vector_parallel_l620_620920


namespace Sn_formula_bn_formula_Sn_times_bn_eq_Tn_times_an_when_n_eq_1_Sn_times_bn_gt_Tn_times_an_when_n_ge_2_l620_620935

-- Definitions for conditions
def Sn_sum_first_5_terms := 30
def Sn_sum_first_10_terms := 110
def Tn_formula (n : ℕ) (bn : ℕ) := 3 / 2 * ↑bn - 1 / 2

-- Variables
variables {a₁ d S₅ S₁₀ : ℕ}
variables (n : ℕ)

-- Sequence and sums
def a_n (n : ℕ) : ℕ := a₁ + (n - 1) * d
def Sn (n : ℕ) : ℕ := n * (a₁ + a_n n) / 2

-- Recorded conditions
axiom S₅_eq : Sn 5 = Sn_sum_first_5_terms
axiom S₁₀_eq : Sn 10 = Sn_sum_first_10_terms
axiom Tn_eq (n : ℕ) (bn : ℕ) : Tn_formula n bn = 3 / 2 * bn - 1 / 2

-- Prove that Sn and bn follow the calculated rules
theorem Sn_formula : Sn n = n^2 + n :=
sorry

theorem bn_formula : ∀ (n : ℕ), b_n n = if n ≠ 0 then (3 : ℕ)^(n-1) else 0 :=
sorry

-- Further analysis
def Tn (n : ℕ) : ℕ := (3^n - 1 : ℕ) / 2
def b_n (n : ℕ) : ℕ := 3^(n-1)

theorem Sn_times_bn_eq_Tn_times_an_when_n_eq_1 :
  S_n 1 * b_n 1 = T_n 1 * a_n 1 :=
sorry

theorem Sn_times_bn_gt_Tn_times_an_when_n_ge_2 (n : ℕ) (h : n ≥ 2):
  S_n n * b_n n > T_n n * a_n n :=
sorry

end Sn_formula_bn_formula_Sn_times_bn_eq_Tn_times_an_when_n_eq_1_Sn_times_bn_gt_Tn_times_an_when_n_ge_2_l620_620935


namespace sin_squared_sufficient_condition_l620_620915

theorem sin_squared_sufficient_condition (α β : ℝ) (hα : 0 < α) (hα' : α < π / 2) (hβ : 0 < β) (hβ' : β < π / 2) :
  (α + β = π / 2) ↔ (sin α * sin α + sin β * sin β = sin (α + β) * sin (α + β)) :=
by
  sorry

end sin_squared_sufficient_condition_l620_620915


namespace ab_value_l620_620032

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_value_l620_620032


namespace greatest_three_digit_multiple_of_17_l620_620253

def is_three_digit (n : Int) : Prop := (n ≥ 100 ∧ n ≤ 999)
def is_multiple_of_17 (n : Int) : Prop := (∃ k : Int, n = 17 * k)

theorem greatest_three_digit_multiple_of_17 : ∀ n : Int, is_three_digit n ∧ is_multiple_of_17 n → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l620_620253


namespace cat_food_sufficiency_l620_620519

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S >= 11 :=
sorry

end cat_food_sufficiency_l620_620519


namespace greatest_3_digit_multiple_of_17_l620_620379

theorem greatest_3_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≥ m) ∧ n = 986 := 
begin
  sorry
end

end greatest_3_digit_multiple_of_17_l620_620379


namespace greatest_three_digit_multiple_of_17_l620_620324

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

theorem greatest_three_digit_multiple_of_17 : ∀ n, n < 1000 → is_multiple_of n 17 → n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620324


namespace area_AMK_l620_620698

-- Defining the basic geometry conditions
variables {A B C M K : Type} [EuclideanSpace ℝ A] 
variables (triangle_ABC : Triangle ℝ A B C)
variables (area_ABC : ℝ) (am_mb_ratio : ℝ) (ak_kc_ratio : ℝ)

-- Given conditions
def given_conditions (hs : area_ABC = 50)
(am_mb_ratio : am_mb_ratio = 1 / 6) (ak_kc_ratio : ak_kc_ratio = 3 / 5) : Prop :=
  true

-- Stating the theorem
theorem area_AMK (hs : area_ABC = 50)
(am_mb_ratio : am_mb_ratio = 1 / 6) (ak_kc_ratio : ak_kc_ratio = 3 / 5) : 
  ∃ (area_AMK : ℝ), area_AMK = 5 :=
begin
  -- Proof goes here
  sorry
end

end area_AMK_l620_620698


namespace max_value_sum_first_n_terms_l620_620591

/-- 
Given:
1. All terms of the geometric sequence {a_n} are positive numbers not equal to 1.
2. The sequence {b_n} satisfies b_n = log a_n.
3. b_3 = 18.
4. b_6 = 12.
Prove:
The maximum value of the sum of the first n terms of the sequence {b_n} is 132.
-/
theorem max_value_sum_first_n_terms :
  ∀ (a_n : ℕ → ℝ) (b_n : ℕ → ℝ),
  (∀ n, a_n n > 0) ∧ (∀ n, a_n n ≠ 1) ∧
  (∀ n, b_n n = log (a_n n)) ∧
  b_n 3 = 18 ∧
  b_n 6 = 12 →
  ∃ n, n ∈ set.univ ∧ (∃ S_n, (S_n = ∑ i in finset.range n, b_n i) ∧ S_n ≤ 132) :=
begin
  sorry
end

end max_value_sum_first_n_terms_l620_620591


namespace cat_food_sufficiency_l620_620516

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end cat_food_sufficiency_l620_620516


namespace ab_eq_six_l620_620068

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_eq_six_l620_620068


namespace determine_sequence_for_primes_l620_620683

theorem determine_sequence_for_primes (p : ℕ) [prime p] (h : p > 3) :
  (∀ (a : Fin (p/2)). 
    ∃! (a_seq : Fin (p/2) → Fin p),
    ∀ (i j : Fin (p/2)), (i ≠ j) → (a_seq i * a_seq j) % p = a) ↔ p > 5 :=
by
  sorry

end determine_sequence_for_primes_l620_620683


namespace rotate_circular_digits_divisible_by_27_l620_620764

theorem rotate_circular_digits_divisible_by_27 
  (seq : List ℕ) (h_len : seq.length = 1956)
  (h0: ∀ i : ℕ, 0 ≤ i → i < 1956 → 0 ≤ seq.nth_le i h0 < 10)
  (h_div_27: (seq.foldl (λ acc d, 10 * acc + d) 0) % 27 = 0)
  : ∀ (k : ℕ) (h_k : k < 1956), ((List.rotate seq k).foldl (λ acc d, 10 * acc + d) 0) % 27 = 0 :=
by
  sorry

end rotate_circular_digits_divisible_by_27_l620_620764


namespace greatest_three_digit_multiple_of_17_l620_620389

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) := 
  exists.intro 986 sorry

end greatest_three_digit_multiple_of_17_l620_620389


namespace ab_value_l620_620011

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_value_l620_620011


namespace peacocks_in_zoo_l620_620999

theorem peacocks_in_zoo :
  ∃ p t : ℕ, 2 * p + 4 * t = 54 ∧ p + t = 17 ∧ p = 7 :=
by
  sorry

end peacocks_in_zoo_l620_620999


namespace seating_arrangements_correct_l620_620648

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def num_seating_arrangements (n : ℕ) (fixed_next : ℕ) : ℕ :=
  2 * factorial (n - 2)

theorem seating_arrangements_correct : 
  num_seating_arrangements 6 2 = 48 :=
by
  -- We state the theorem including the proof steps but skip the proof with a sorry.
  sorry

end seating_arrangements_correct_l620_620648


namespace inequalities_true_l620_620570

theorem inequalities_true (a b : ℝ) (h : a > b) (h' : b > 0) :
  a^2 > b^2 ∧ a^3 > b^3 ∧ sqrt (a - b) > sqrt a - sqrt b :=
by
  sorry

end inequalities_true_l620_620570


namespace james_minimum_wage_l620_620099

theorem james_minimum_wage : 
  ∀ (wage meat_pounds fruitveg_pounds bread_pounds janitor_hours janitor_wage james_hours : ℕ),
  meat_pounds = 20 →                     -- 20 pounds of meat
  fruitveg_pounds = 15 →                 -- 15 pounds of fruits and vegetables
  bread_pounds = 60 →                    -- 60 pounds of bread products
  janitor_hours = 10 →                   -- 10 hours for janitorial staff
  janitor_wage = 10 →                    -- $10/hour for janitorial staff (normal wage)
  james_hours = 50 →                     -- James will have to work 50 hours
  wage = 5 →                             -- $5/pound for meat
  fruitveg_cost_per_pound = 4 →          -- $4/pound for fruits and vegetables
  bread_cost_per_pound = 3/2 →           -- $1.50/pound for bread products
  time_and_a_half = 3/2 →                -- time-and-a-half pay multiplier
  let meat_cost := meat_pounds * wage,
  let fruitveg_cost := fruitveg_pounds * fruitveg_cost_per_pound,
  let bread_cost := bread_pounds * bread_cost_per_pound,
  let janitor_cost := janitor_hours * (janitor_wage * time_and_a_half),
  let total_cost := meat_cost + fruitveg_cost + bread_cost + janitor_cost,
  let min_wage := total_cost / james_hours,
  min_wage = 8 := 
by
  sorry

end james_minimum_wage_l620_620099


namespace ab_value_l620_620033

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_value_l620_620033


namespace greatest_3_digit_multiple_of_17_l620_620385

theorem greatest_3_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≥ m) ∧ n = 986 := 
begin
  sorry
end

end greatest_3_digit_multiple_of_17_l620_620385


namespace count_nonempty_sets_l620_620976

theorem count_nonempty_sets : 
  let M := {M : Set ℕ | M ⊆ {1, 2, 3, 4, 5} ∧ ∀ a ∈ M, (6 - a) ∈ M} 
  in M.count = 7 :=
by
  let M := {M : Set ℕ | M ⊆ {1, 2, 3, 4, 5} ∧ ∀ a ∈ M, (6 - a) ∈ M}
  have h1 : {1, 5} ∈ M := by sorry
  have h2 : {2, 4} ∈ M := by sorry
  have h3 : {3} ∈ M := by sorry
  have h4 : {1, 5, 2, 4} ∈ M := by sorry
  have h5 : {1, 5, 3} ∈ M := by sorry
  have h6 : {2, 4, 3} ∈ M := by sorry
  have h7 : {1, 5, 2, 4, 3} ∈ M := by sorry
  have h_subset : ∀ S ∈ M, S = {1, 5} ∨ S = {2, 4} ∨ S = {3} ∨ 
                                 S = {1, 5, 2, 4} ∨ S = {1, 5, 3} ∨ 
                                 S = {2, 4, 3} ∨ S = {1, 5, 2, 4, 3} := by sorry
  exact M.count_eq (by simp only [h1, h2, h3, h4, h5, h6, h7])

end count_nonempty_sets_l620_620976


namespace find_integer_n_l620_620791

theorem find_integer_n :
  ∃ n : ℕ, 0 ≤ n ∧ n < 201 ∧ 200 * n ≡ 144 [MOD 101] ∧ n = 29 := 
by
  sorry

end find_integer_n_l620_620791


namespace sufficient_food_supply_l620_620504

variable {L S : ℝ}

theorem sufficient_food_supply (h1 : L + 4 * S = 14) (h2 : L > S) : L + 3 * S ≥ 11 :=
by
  sorry

end sufficient_food_supply_l620_620504


namespace mark_vs_luke_percentage_difference_l620_620132

-- Variables for the number of pencils each person has
variables {J M L : ℝ}

-- Conditions given in the problem
def condition1 : Prop := M = 1.75 * J
def condition2 : Prop := L = 1.50 * J

-- Theorem to prove the percentage difference
theorem mark_vs_luke_percentage_difference :
  condition1 → condition2 → (M - L) / L = 16.67 / 100 :=
by
  intros h1 h2
  sorry  -- Proof elided

end mark_vs_luke_percentage_difference_l620_620132


namespace Cora_book_reading_problem_l620_620533

theorem Cora_book_reading_problem
  (total_pages: ℕ)
  (read_monday: ℕ)
  (read_tuesday: ℕ)
  (read_wednesday: ℕ)
  (H: total_pages = 158 ∧ read_monday = 23 ∧ read_tuesday = 38 ∧ read_wednesday = 61) :
  ∃ P: ℕ, 23 + 38 + 61 + P + 2 * P = total_pages ∧ P = 12 :=
  sorry

end Cora_book_reading_problem_l620_620533


namespace ellipse_hyperbola_same_directrix_l620_620987

theorem ellipse_hyperbola_same_directrix (k : ℝ) : 
  (∀ x y : ℝ, x^2 + k * y^2 = 1 → x^2 / 4 - y^2 / 5 = 1 → 
  ∀ e : ℝ, e = sqrt(1 + 5 / 4) → ∀ d : ℝ, d = 4 / e → d = 8 / 3) → 
  k = 16 / 7 :=
by
  sorry

end ellipse_hyperbola_same_directrix_l620_620987


namespace ab_equals_six_l620_620020

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_l620_620020


namespace max_profit_l620_620462

noncomputable def C (x : ℝ) : ℝ :=
  if x < 80 then (1/2) * x^2 + 40 * x
  else 101 * x + 8100 / x - 2180

noncomputable def profit (x : ℝ) : ℝ :=
  if x < 80 then 100 * x - C x - 500
  else 100 * x - C x - 500

theorem max_profit :
  (∀ x, (0 < x ∧ x < 80) → profit x = - (1/2) * x^2 + 60 * x - 500) ∧
  (∀ x, (80 ≤ x) → profit x = 1680 - (x + 8100 / x)) ∧
  (∃ x, x = 90 ∧ profit x = 1500) :=
by {
  -- Proof here
  sorry
}

end max_profit_l620_620462


namespace ab_equals_six_l620_620002

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l620_620002


namespace simplify_tangent_cosine_expr_l620_620721

theorem simplify_tangent_cosine_expr :
  (real.tan (20 * real.pi / 180) + real.tan (40 * real.pi / 180) + real.tan (60 * real.pi / 180)) / real.cos (10 * real.pi / 180) =
  real.sqrt 3 * ((1 / 2 * real.cos (10 * real.pi / 180) + real.sqrt 3 / 2) / (real.cos (10 * real.pi / 180) * real.cos (20 * real.pi / 180) * real.cos (40 * real.pi / 180))) :=
sorry

end simplify_tangent_cosine_expr_l620_620721


namespace irreducible_polynomial_l620_620125

theorem irreducible_polynomial {α : Type*} [CommRing α] [IsDomain α] 
  (a : ℕ → α) (n : ℕ) (h : ∀ (i j : ℕ), i ≠ j → a i ≠ a j) : 
  Irreducible (∏ i in finset.range n, (X - C (a i)) - 1) := by
sorry

end irreducible_polynomial_l620_620125


namespace fraction_exponent_eval_l620_620892

theorem fraction_exponent_eval :
  (1 / 3) ^ 6 * (2 / 5) ^ (-4) = 625 / 11664 := by
  sorry

end fraction_exponent_eval_l620_620892


namespace marbles_total_l620_620896

theorem marbles_total (fabian kyle miles : ℕ) (h1 : fabian = 3 * kyle) (h2 : fabian = 5 * miles) (h3 : fabian = 15) : kyle + miles = 8 := by
  sorry

end marbles_total_l620_620896


namespace total_receipts_l620_620840

theorem total_receipts 
  (x y : ℕ) 
  (h1 : x + y = 64)
  (h2 : y ≥ 8) 
  : 3 * x + 4 * y = 200 := 
by
  sorry

end total_receipts_l620_620840


namespace cat_food_sufficiency_l620_620522

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S >= 11 :=
sorry

end cat_food_sufficiency_l620_620522


namespace find_angles_of_triangles_l620_620430

-- Given conditions
variables (A B C D M : Point)
variables (h1 : Segment A C ≠ ∅)
variables (h2 : LineSegment.Intersects AD BC at M)
variables (h3 : ∠B = 40)
variables (h4 : ∠D = 40)
variables (h5 : Distance DB = Distance AB)
variables (h6 : ∠AMC = 70)

-- Conclusion to be proven
theorem find_angles_of_triangles :
  ∠BAC = 110 ∧ ∠BCA = 30 ∧ ∠DCA = 60 ∧ ∠DAC = 80 := 
sorry

end find_angles_of_triangles_l620_620430


namespace greatest_three_digit_multiple_of_17_l620_620349

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620349


namespace max_min_a1_conditions_l620_620428

open Real

noncomputable def max_a1 (u v : ℝ) (n : ℕ) :=
  (u + sqrt ((n - 1) * (n * v - u ^ 2))) / n

noncomputable def min_a1 (u v : ℝ) (n k : ℕ) :=
  if v ≤ u ^ 2 / (k + 1) then
    u / (k + 1)
  else
    (k * u + sqrt (k * (k + 1) * v - u ^ 2)) / (k * (k + 1))

theorem max_min_a1_conditions (u v : ℝ) (n : ℕ) :
  (∃ a : ℕ → ℝ, u > 0 ∧ v > 0 ∧
    (∀ i, (1 ≤ i ∧ i ≤ n) → a i ≥ 0) ∧
    (∑ i in range n, a i = u) ∧
    (∑ i in range n, (a i) ^ 2 = v)) ↔
  (n * v ≥ u ^ 2 ∧ u ^ 2 ≥ v ∧
    ∀ k, (1 ≤ k ∧ k ≤ n - 1) →
      (u ^ 2 / k ≥ v → min_a1 u v n k = (k * u + sqrt (k * (k + 1) * v - u ^ 2)) / (k * (k + 1)) ∧
      u ^ 2 / (k + 1) ≥ v → min_a1 u v n k = u / (k + 1))) ∧
  max_a1 u v n = (u + sqrt ((n - 1) * (n * v - u ^ 2))) / n := by
  sorry

end max_min_a1_conditions_l620_620428


namespace sum_in_base4_l620_620740

def dec_to_base4 (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  let rec convert (n : ℕ) (acc : ℕ) (power : ℕ) :=
    if n = 0 then acc
    else convert (n / 4) (acc + (n % 4) * power) (power * 10)
  convert n 0 1

theorem sum_in_base4 : dec_to_base4 (234 + 78) = 13020 :=
  sorry

end sum_in_base4_l620_620740


namespace Sperner_bound_l620_620677

variables {n r t : ℕ}
variables {X : Type*} [fintype X] [decidable_eq X]

def is_S_family (A : finset (finset X)) : Prop :=
  ∀ {a b : finset X}, a ∈ A → b ∈ A → (a ≠ b → ¬(a ⊆ b) ∧ ¬(b ⊆ a))

theorem Sperner_bound (A : finset (finset X)) :
  (∀ i, (i < t → ∃ A_i ∈ A, A_i.card ≤ r)) →
  (r ≤ n / 2) →
  is_S_family A →
  t ≤ nat.choose (n - 1) (r - 1) :=
sorry

end Sperner_bound_l620_620677


namespace min_value_of_quadratic_l620_620881

theorem min_value_of_quadratic (x : ℝ) : ∃ y, y = x^2 + 14*x + 20 ∧ ∀ z, z = x^2 + 14*x + 20 → z ≥ -29 :=
by
  sorry

end min_value_of_quadratic_l620_620881


namespace dice_faces_combined_least_possible_l620_620787

theorem dice_faces_combined_least_possible (a b : ℕ) (h1 : a ≥ 8) (h2 : b ≥ 8) (h3 : a ≥ b) 
(h_prob_sum_9 : 8 / (a * b) = 1 / 2 * (prob_sum_12 a b)) 
(h_prob_sum_14 : 1 / 6 = prob_sum_14 a b / (a * b)) : 
  a + b = 23 :=
by
  sorry

noncomputable def prob_sum_12 (a b : ℕ) : ℚ :=
  if a < 12 ∧ b < 12 then 0 else 16 / (a * b)

noncomputable def prob_sum_14 (a b : ℕ) : ℚ :=
  (number_of_ways_sum_14 a b) / (a * b)

noncomputable def number_of_ways_sum_14 (a b : ℕ) : ℕ :=
  if a < 14 ∧ b < 14 then 0 else ab / 6

end dice_faces_combined_least_possible_l620_620787


namespace fraction_equality_l620_620567

theorem fraction_equality (a b : ℝ) (h : a / b = 2 / 3) : a / (a + b) = 2 / 5 :=
by
  sorry

end fraction_equality_l620_620567


namespace exists_divisor_in_S_l620_620105

open Set

theorem exists_divisor_in_S (A B : Set ℕ) 
  (hA1 : 2 ≤ A.card) (hB1 : 2 ≤ B.card) 
  (S : Set ℕ)
  (hS : S = {n | ∃ a ∈ A, ∃ b ∈ B, n = a * b} ∧ S.card = A.card + B.card - 1) :
  ∃ x y z ∈ S, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x ∣ (y * z) := 
sorry

end exists_divisor_in_S_l620_620105


namespace fathers_age_after_further_8_years_l620_620809

variable (R F : ℕ)

def age_relation_1 : Prop := F = 4 * R
def age_relation_2 : Prop := F + 8 = (5 * (R + 8)) / 2

theorem fathers_age_after_further_8_years (h1 : age_relation_1 R F) (h2 : age_relation_2 R F) : (F + 16) = 2 * (R + 16) :=
by 
  sorry

end fathers_age_after_further_8_years_l620_620809


namespace flea_return_strategy_l620_620676

def isPrime (p: ℕ) : Prop := p > 1 ∧ ∀ m, m > 1 → m * m ≤ p → p % m ≠ 0

def f (p : ℕ) : ℤ :=
if p = 2 ∨ p = 3 then 0
else if p % 3 = 1 then 1
else -1

theorem flea_return_strategy (p : ℕ) (h_prime : isPrime p) :
  (p = 2 ∨ p = 3 → f(p) = 0) ∧
  (p ≠ 2 ∧ p ≠ 3 → p % 3 = 1 → f(p) = 1) ∧
  (p ≠ 2 ∧ p ≠ 3 → p % 3 = 2 → f(p) = -1) :=
sorry

end flea_return_strategy_l620_620676


namespace problem_statement_l620_620092

variable {Point Line Plane : Type}

-- Definitions for perpendicular and parallel
def perpendicular (l1 l2 : Line) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def perp_to_plane (l : Line) (p : Plane) : Prop := sorry

-- Given variables
variable (a b c d : Line) (α β : Plane)

-- Conditions
axiom a_perp_b : perpendicular a b
axiom c_perp_d : perpendicular c d
axiom a_perp_alpha : perp_to_plane a α
axiom c_perp_alpha : perp_to_plane c α

-- Required proof
theorem problem_statement : perpendicular c b :=
by sorry

end problem_statement_l620_620092


namespace ab_equals_six_l620_620019

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_l620_620019


namespace expression_B_cannot_be_factored_in_reals_l620_620466

-- Define the expressions
def exprA (x y : ℝ) : ℝ := 9 * x^2 + 3 * x * y^2
def exprB (a b : ℝ) : ℝ := a^2 + 2 * a * b - b^2
def exprC (x y : ℝ) : ℝ := -x^2 + 25 * y^2
def exprD (x : ℝ) : ℝ := x^2 - x + 1 / 4

-- State the problem in Lean
theorem expression_B_cannot_be_factored_in_reals :
  ¬ ∃ (f g : ℝ → ℝ), ∀ a b : ℝ, exprB a b = f a * g b := sorry

end expression_B_cannot_be_factored_in_reals_l620_620466


namespace cos_value_given_tan_l620_620921

theorem cos_value_given_tan (θ : Real) (h : tan θ = 3) : 
  cos ((3 * Real.pi / 2) + 2 * θ) = 3 / 5 :=
by 
  sorry

end cos_value_given_tan_l620_620921


namespace solve_for_x_l620_620169

theorem solve_for_x : ∃ (x : ℕ), (1 / 4 : ℚ) - (1 / 5 : ℚ) = 1 / (x : ℚ) ∧ x = 20 :=
by
  use 20
  have h : (1 / 4 : ℚ) - (1 / 5 : ℚ) = (5 : ℚ) / 20 - (4 : ℚ) / 20 := by sorry
  have h' : (5 / 20 : ℚ) - (4 / 20 : ℚ) = 1 / 20 := by sorry
  rw [h, h']
  split
  { exact (1 / 20 : ℚ).symm }
  { refl }
  sorry

end solve_for_x_l620_620169


namespace product_of_ab_l620_620054

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end product_of_ab_l620_620054


namespace kyle_and_miles_marbles_l620_620898

theorem kyle_and_miles_marbles (f k m : ℕ) 
  (h1 : f = 3 * k) 
  (h2 : f = 5 * m) 
  (h3 : f = 15) : 
  k + m = 8 := 
by 
  sorry

end kyle_and_miles_marbles_l620_620898


namespace next_remote_work_together_l620_620085

open Nat

theorem next_remote_work_together (A B C D : ℕ) (hA : A = 5) (hB : B = 3) (hC : C = 8) (hD : D = 9) :
  Nat.lcm (Nat.lcm A B) (Nat.lcm C D) = 360 :=
by
  rw [hA, hB, hC, hD]
  simp [Nat.lcm]
  sorry

end next_remote_work_together_l620_620085


namespace evaluate_dollar_l620_620879

variable {R : Type} [Field R]

def dollar (a b : R) : R := (a - b) ^ 2

theorem evaluate_dollar (x y : R) : dollar (2 * x + 3 * y) (3 * x - 4 * y) = x ^ 2 - 14 * x * y + 49 * y ^ 2 := by
  sorry

end evaluate_dollar_l620_620879


namespace long_fur_brown_dogs_l620_620080

-- Defining the basic parameters given in the problem
def total_dogs : ℕ := 45
def long_fur : ℕ := 26
def brown_dogs : ℕ := 30
def neither_long_fur_nor_brown : ℕ := 8

-- Statement of the theorem
theorem long_fur_brown_dogs : ∃ LB : ℕ, LB = 27 ∧ total_dogs = long_fur + brown_dogs - LB + neither_long_fur_nor_brown :=
by {
  -- skipping the proof
  sorry
}

end long_fur_brown_dogs_l620_620080


namespace count_special_numbers_l620_620628

theorem count_special_numbers :
  ∃ n : ℕ, n = 5 ∧
    (∀ k : ℕ, 0 < k → k < 3000 → k % 4 = 0 ∧ k % 5 = 0 ∧ (∃ m : ℕ, k = m^2) → 
    {x | x < 3000 ∧ x % 4 = 0 ∧ x % 5 = 0 ∧ (∃ m : ℕ, x = m^2)}.card = n) :=
begin
  use 5,
  split,
  { refl, },
  { sorry, }
end

end count_special_numbers_l620_620628


namespace ab_equals_six_l620_620018

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_l620_620018


namespace paint_total_gallons_l620_620444

theorem paint_total_gallons
  (white_paint_gallons : ℕ)
  (blue_paint_gallons : ℕ)
  (h_wp : white_paint_gallons = 660)
  (h_bp : blue_paint_gallons = 6029) :
  white_paint_gallons + blue_paint_gallons = 6689 := 
by
  sorry

end paint_total_gallons_l620_620444


namespace no_intersection_points_l620_620227

theorem no_intersection_points :
  ∀ x y : ℝ, (y = 2*x^2 + 3*x - 4) → (y = 3*x^2 + 12) → false := 
begin
  intros x y h1 h2,
  have h := congr_arg (λ y, y - (2*x^2 + 3*x - 4)) h2,
  simp [h1] at h,
  have : 2*x^2 + 3*x - 4 - (3*x^2 + 12) = -x^2 + 3*x - 16 := by ring,
  rw this at h,
  simp at h,
  let discriminant := -55,
  have h3 : discr < 0 := by norm_num,
  sorry
end

end no_intersection_points_l620_620227


namespace distance_from_A_to_y_axis_is_2_l620_620178

-- Define the point A
def point_A : ℝ × ℝ := (-2, 1)

-- Define the distance function to the y-axis
def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  abs p.1

-- The theorem to prove
theorem distance_from_A_to_y_axis_is_2 : distance_to_y_axis point_A = 2 :=
by
  sorry

end distance_from_A_to_y_axis_is_2_l620_620178


namespace next_time_angle_150_degrees_l620_620975

theorem next_time_angle_150_degrees (x : ℝ) : (∃ x, x = 329/6) :=
by
  let θ := λ H M : ℝ, abs (30 * H - 5.5 * M)
  let initial_angle := θ 5 0
  have eq1 : initial_angle = 150 :=
    by sorry
  let H := 5 + x / 60
  have eq2 : θ H x = 150 :=
    by sorry
  have eq3 : abs (150 - 5 * x) = 150 :=
    by sorry
  have eq4 : abs (150 - 5 * x) = 150 := by sorry
  have solution : x = 54 + 6 / 11 :=
    by sorry
  existsi solution
  sorry

end next_time_angle_150_degrees_l620_620975


namespace product_of_ab_l620_620050

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end product_of_ab_l620_620050


namespace sqrt_factorial_div_90_eq_l620_620483

open Real

theorem sqrt_factorial_div_90_eq : sqrt (realOfNat (Nat.factorial 9) / 90) = 24 * sqrt 7 := by
  sorry

end sqrt_factorial_div_90_eq_l620_620483


namespace rectangle_area_l620_620824

theorem rectangle_area (A1 A2 : ℝ) (h1 : A1 = 40) (h2 : A2 = 10) :
    ∃ n : ℕ, n = 240 ∧ ∃ R : ℝ, R = 2 * Real.sqrt (40 / Real.pi) + 2 * Real.sqrt (10 / Real.pi) ∧ 
               (4 * Real.sqrt (10) / Real.sqrt (Real.pi)) * (6 * Real.sqrt (10) / Real.sqrt (Real.pi)) = n / Real.pi :=
by
  sorry

end rectangle_area_l620_620824


namespace quadratic_has_solution_l620_620709

theorem quadratic_has_solution (a b : ℝ) : ∃ x : ℝ, (a^6 - b^6) * x^2 + 2 * (a^5 - b^5) * x + (a^4 - b^4) = 0 :=
  by sorry

end quadratic_has_solution_l620_620709


namespace x_eq_one_l620_620109

/--
Let \(x, y\) be positive integers. If for each positive integer \(n\) we have that \((ny)^2 + 1\) divides \(x^{\varphi(n)} - 1\), then \(x = 1\).
-/
theorem x_eq_one 
  (x y : ℕ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y) 
  (h : ∀ n : ℕ, 0 < n → ((n * y)^2 + 1) ∣ (x^(nat_totient n) - 1)) : 
  x = 1 := 
sorry

end x_eq_one_l620_620109


namespace line_CE_determined_l620_620937

noncomputable def inscribed_quadrilateral (A B C D : Point) : Prop :=
  cyclic_quadrilateral A B C D

def intersects_lines_AC_CB (A B C D E : Point) : Prop :=
  ∃ X Y, X ∈ line A C ∧ Y ∈ line C B ∧ X ∈ quadrilateral A B C D ∧ Y ∈ quadrilateral A B C D

theorem line_CE_determined (A B C D E : Point) 
  (h1 : inscribed_quadrilateral A B C D)
  (h2 : intersects_lines_AC_CB A B C D E) :
  ∃ CE : Line, CE ∈ line C E ∧ CE ∈ intersections (line A C) (line C B) :=
sorry

end line_CE_determined_l620_620937


namespace relation_among_abc_l620_620922

noncomputable def a : ℝ := Real.logb 2 0.3
noncomputable def b : ℝ := 2^0.1
noncomputable def c : ℝ := 0.2^1.3

theorem relation_among_abc : a < c ∧ c < b :=
by
  sorry

end relation_among_abc_l620_620922


namespace hike_up_days_l620_620832

theorem hike_up_days (R_up R_down D_down D_up : ℝ) 
  (H1 : R_up = 8) 
  (H2 : R_down = 1.5 * R_up)
  (H3 : D_down = 24)
  (H4 : D_up / R_up = D_down / R_down) : 
  D_up / R_up = 2 :=
by
  sorry

end hike_up_days_l620_620832


namespace extreme_value_at_one_inequality_for_positive_values_l620_620609

variable (a : ℝ)

def f (x : ℝ) : ℝ := x ^ 2 + 2 * a * log x

theorem extreme_value_at_one (h : deriv f 1 = 0) : a = -1 := sorry

theorem inequality_for_positive_values
  (h : ∀ x, 1 ≤ x → f x > 0) : a > - Real.exp 1 := sorry

end extreme_value_at_one_inequality_for_positive_values_l620_620609


namespace shorter_piece_length_l620_620814

theorem shorter_piece_length (x : ℝ) (h : 3 * x = 60) : x = 20 :=
by
  sorry

end shorter_piece_length_l620_620814


namespace ellipse_equation_l620_620880

-- Defining the points P(1, sqrt(3)/2) and Q(2, 0)
def P : ℝ × ℝ := (1, Real.sqrt 3 / 2)
def Q : ℝ × ℝ := (2, 0)

-- Assume an equation of the form mx^2 + ny^2 = 1
def m : ℝ := 1/4
def n : ℝ := 1

theorem ellipse_equation :
  (m * (P.1)^2 + n * (P.2)^2 = 1) ∧ (m * (Q.1)^2 + n * (Q.2)^2 = 1) ↔ (∀ x y, m * x^2 + n * y^2 = 1 ↔ x^2 / 4 + y^2 = 1) :=
by 
  sorry

end ellipse_equation_l620_620880


namespace greatest_three_digit_multiple_of_17_l620_620328

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

theorem greatest_three_digit_multiple_of_17 : ∀ n, n < 1000 → is_multiple_of n 17 → n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620328


namespace ab_value_l620_620005

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_value_l620_620005


namespace greatest_3_digit_multiple_of_17_l620_620376

theorem greatest_3_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≥ m) ∧ n = 986 := 
begin
  sorry
end

end greatest_3_digit_multiple_of_17_l620_620376


namespace best_fit_model_l620_620409

theorem best_fit_model (r2_A r2_B r2_C r2_D : ℝ) 
                       (hA : r2_A = 0.60) (hB : r2_B = 0.63) 
                       (hC : r2_C = 0.65) (hD : r2_D = 0.68) :
                       r2_D = max (max r2_A r2_B) (max r2_C r2_D) := 
by
  have h1 : r2_A ≤ r2_B, by linarith [hA, hB]
  have h2 : r2_B ≤ r2_C, by linarith [hB, hC]
  have h3 : r2_C ≤ r2_D, by linarith [hC, hD]
  have h4 : r2_A ≤ r2_C, by linarith [h1, h2]
  have h5 : r2_B ≤ r2_D, by linarith [h2, h3]
  have h6 : r2_A ≤ r2_D, by linarith [h4, h3]
  have h7 : r2_C ≤ r2_D, by linarith [h3]
  sorry

end best_fit_model_l620_620409


namespace greatest_three_digit_multiple_of_17_l620_620390

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) := 
  exists.intro 986 sorry

end greatest_three_digit_multiple_of_17_l620_620390


namespace ab_equals_six_l620_620044

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end ab_equals_six_l620_620044


namespace result_l620_620857

def problem : Float :=
  let sum := 78.652 + 24.3981
  let diff := sum - 0.025
  Float.round (diff * 100) / 100

theorem result :
  problem = 103.03 := by
  sorry

end result_l620_620857


namespace ab_value_l620_620034

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_value_l620_620034


namespace curve_slope_at_neg1_l620_620599

noncomputable def curve : ℝ → ℝ → ℝ := λ x a, x^4 + a * x^2 + 1

noncomputable def derivative (a : ℝ) : ℝ → ℝ := λ x, 4 * x^3 + 2 * a * x

theorem curve_slope_at_neg1 (a : ℝ) : derivative a (-1) = 8 → a = -6 :=
by
  intro h
  sorry

end curve_slope_at_neg1_l620_620599


namespace greatest_three_digit_multiple_of_17_l620_620268

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (17 ∣ n) ∧ ∀ m : ℤ, (100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m) → m ≤ n :=
begin
  use 969,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num, exact dec_trivial, },
  { intros m h_mconds,
    cases h_mconds with hm1 h_mconds,
    cases h_mconds with hm2 h_formula,
    rw h_formula, sorry, }  -- Completing this part is not necessary according to the guidelines
end

end greatest_three_digit_multiple_of_17_l620_620268


namespace clock_90_degree_angle_times_l620_620723

noncomputable def first_time_90_degree_angle (t : ℝ) : Prop := 5.5 * t = 90

noncomputable def second_time_90_degree_angle (t : ℝ) : Prop := 5.5 * t = 270

theorem clock_90_degree_angle_times :
  ∃ t₁ t₂ : ℝ,
  first_time_90_degree_angle t₁ ∧ 
  second_time_90_degree_angle t₂ ∧ 
  t₁ = (180 / 11 : ℝ) ∧ 
  t₂ = (540 / 11 : ℝ) :=
by
  sorry

end clock_90_degree_angle_times_l620_620723


namespace square_of_smallest_integer_square_is_441_l620_620207

theorem square_of_smallest_integer (x : ℕ)
  (h : x^2 + (x + 1)^2 + (x + 2)^2 + (x + 3)^2 = (x + 4)^2 + (x + 5)^2 + (x + 6)^2) :
  x = 21 :=
begin
  -- The necessary steps to prove that x = 21 will go here.
  sorry
end

theorem square_is_441 (x : ℕ) (h : x = 21) : x^2 = 441 :=
begin
  -- The necessary steps to prove that the square of 21 is 441.
  sorry
end

end square_of_smallest_integer_square_is_441_l620_620207


namespace greatest_three_digit_multiple_of_17_l620_620250

def is_three_digit (n : Int) : Prop := (n ≥ 100 ∧ n ≤ 999)
def is_multiple_of_17 (n : Int) : Prop := (∃ k : Int, n = 17 * k)

theorem greatest_three_digit_multiple_of_17 : ∀ n : Int, is_three_digit n ∧ is_multiple_of_17 n → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l620_620250


namespace circle_ratio_new_diameter_circumference_l620_620989

theorem circle_ratio_new_diameter_circumference (r : ℝ) :
  let new_radius := r + 2
  let new_diameter := 2 * new_radius
  let new_circumference := 2 * Real.pi * new_radius
  new_circumference / new_diameter = Real.pi := 
by
  sorry

end circle_ratio_new_diameter_circumference_l620_620989


namespace lattice_point_probability_l620_620833

theorem lattice_point_probability :
  let side_length := 4040
  let total_area := side_length ^ 2
  let unit_square_area := 1
  let desired_probability := 1 / 4
  -- let the radius d be such that the covered area fraction is desired_probability
  let area_of_circle_covered := desired_probability * unit_square_area
  let radius := Real.sqrt (area_of_circle_covered / Real.pi)
  Float.round(radius * 10) / 10 = 0.3 := by
sorry

end lattice_point_probability_l620_620833


namespace total_ingredients_l620_620202

theorem total_ingredients (b f s : ℕ) (h_ratio : 2 * f = 5 * f) (h_flour : f = 15) : b + f + s = 30 :=
by 
  sorry

end total_ingredients_l620_620202


namespace solve_for_m_l620_620166

-- Define the conditions as hypotheses
theorem solve_for_m (m : ℝ) (h : (m - 5)^3 = (1 / 16)^(-1)) : m = 5 + 2^(4 / 3) := by
  sorry

end solve_for_m_l620_620166


namespace board_uniform_color_l620_620927

theorem board_uniform_color {n : ℕ} (h_n : n ≥ 3) :
  (∀ board : (Fin n) × (Fin n) → bool, ∃ steps : list ((Fin n) × (Fin n)), 
    uniform_color (apply_steps board steps)) ↔ 
  ∃ k : ℕ, n = 4 * k :=
by 
  sorry

end board_uniform_color_l620_620927


namespace number_of_elements_in_A_l620_620985

def A : set (ℤ × ℤ) := {(1, 2), (3, 4)}

theorem number_of_elements_in_A : A.size = 2 :=
by
  sorry

end number_of_elements_in_A_l620_620985


namespace find_positive_integers_l620_620902

theorem find_positive_integers :
  ∃ (n : Fin 8 → ℕ),
  (∀ i, 1 ≤ n i) ∧
  ∀ k : ℤ, (-2007 ≤ k ∧ k ≤ 2007) →
  ∃ (α : Fin 8 → ℤ), (∀ i, α i ∈ {-1, 0, 1}) ∧ k = (Finset.univ.sum (λ i, α i * n i)) :=
by {
  sorry
}

end find_positive_integers_l620_620902


namespace part_I_part_II_l620_620610

-- Part I
theorem part_I (x : ℝ) : (|x + 1| + |x - 4| ≤ 2 * |x - 4|) ↔ (x < 1.5) :=
sorry

-- Part II
theorem part_II (a : ℝ) : (∀ x : ℝ, |x + a| + |x - 4| ≥ 3) → (a ≤ -7 ∨ a ≥ -1) :=
sorry

end part_I_part_II_l620_620610


namespace stickers_after_loss_l620_620773

-- Conditions
def stickers_per_page : ℕ := 20
def initial_pages : ℕ := 12
def lost_pages : ℕ := 1

-- Problem statement
theorem stickers_after_loss : (initial_pages - lost_pages) * stickers_per_page = 220 := by
  sorry

end stickers_after_loss_l620_620773


namespace beer_output_per_hour_increase_l620_620425

noncomputable def output_increase_per_hour
  (initial_output : ℚ) (initial_hours : ℚ)
  (increased_output : (initial_output * 1.20))
  (decreased_hours : (initial_hours * 0.70)) : ℚ := 
  ((increased_output / decreased_hours) / (initial_output / initial_hours) - 1) * 100

theorem beer_output_per_hour_increase
  (initial_output : ℚ) (initial_hours : ℚ) :
  output_increase_per_hour initial_output initial_hours
    (initial_output * 1.20) (initial_hours * 0.70) = 71.43 := 
sorry

end beer_output_per_hour_increase_l620_620425


namespace greatest_three_digit_multiple_of_17_l620_620258

def is_three_digit (n : Int) : Prop := (n ≥ 100 ∧ n ≤ 999)
def is_multiple_of_17 (n : Int) : Prop := (∃ k : Int, n = 17 * k)

theorem greatest_three_digit_multiple_of_17 : ∀ n : Int, is_three_digit n ∧ is_multiple_of_17 n → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l620_620258


namespace power_identity_l620_620916

theorem power_identity (x y a b : ℝ) (h1 : 10^x = a) (h2 : 10^y = b) : 10^(3*x + 2*y) = a^3 * b^2 := 
by 
  sorry

end power_identity_l620_620916


namespace curve_is_circle_l620_620883

theorem curve_is_circle (x y : ℝ) : x^2 + y^2 = 16 ↔ ∃ r : ℝ, r = 4 ∧ (x, y) = (r * real.cos θ, r * real.sin θ) :=
sorry

end curve_is_circle_l620_620883


namespace gnomes_remaining_in_ravenswood_l620_620731

theorem gnomes_remaining_in_ravenswood 
  (westerville_gnomes : ℕ)
  (ravenswood_initial_gnomes : ℕ)
  (taken_gnomes : ℕ)
  (remaining_gnomes : ℕ)
  (h1 : westerville_gnomes = 20)
  (h2 : ravenswood_initial_gnomes = 4 * westerville_gnomes)
  (h3 : taken_gnomes = (40 * ravenswood_initial_gnomes) / 100)
  (h4 : remaining_gnomes = ravenswood_initial_gnomes - taken_gnomes) :
  remaining_gnomes = 48 :=
by
  sorry

end gnomes_remaining_in_ravenswood_l620_620731


namespace max_real_part_fifth_power_l620_620696

theorem max_real_part_fifth_power :
  let z1 := complex.mk (-2) 0
  let z2 := complex.mk (-real.sqrt 3) 1
  let z3 := complex.mk (-real.sqrt 2) (real.sqrt 2)
  let z4 := complex.mk (-1) (real.sqrt 3)
  let z5 := complex.mk 0 2 in
  complex.re (z2 ^ 5) > complex.re (z1 ^ 5) ∧
  complex.re (z2 ^ 5) > complex.re (z3 ^ 5) ∧
  complex.re (z2 ^ 5) > complex.re (z4 ^ 5) ∧
  complex.re (z2 ^ 5) > complex.re (z5 ^ 5) :=
by
  -- placeholder for the proof.
  sorry

end max_real_part_fifth_power_l620_620696


namespace greatest_three_digit_multiple_of_17_l620_620248

def is_three_digit (n : Int) : Prop := (n ≥ 100 ∧ n ≤ 999)
def is_multiple_of_17 (n : Int) : Prop := (∃ k : Int, n = 17 * k)

theorem greatest_three_digit_multiple_of_17 : ∀ n : Int, is_three_digit n ∧ is_multiple_of_17 n → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l620_620248


namespace simplify_G_equals_4F_l620_620112

-- Define F as a function of x
def F (x : ℝ) : ℝ := log ((1 + x) / (1 - x))

-- Substitute x in F with (4x - x^3) / (1 + 4x^2) to define G
def G (x : ℝ) : ℝ := 
  let y := (4 * x - x^3) / (1 + 4 * x^2) 
  in log ((1 + y) / (1 - y))

-- Prove that G(x) = 4 * F(x)
theorem simplify_G_equals_4F (x : ℝ) : G x = 4 * F x :=
by
  sorry

end simplify_G_equals_4F_l620_620112


namespace greatest_three_digit_multiple_of_17_l620_620337

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620337


namespace remaining_stickers_l620_620769

def stickers_per_page : ℕ := 20
def pages : ℕ := 12
def lost_pages : ℕ := 1

theorem remaining_stickers : 
  (pages * stickers_per_page - lost_pages * stickers_per_page) = 220 :=
  by
    sorry

end remaining_stickers_l620_620769


namespace prime_sets_subseteq_l620_620427

def S1 (p : ℕ) : set (ℤ × ℤ × ℤ) :=
  { x | ∃ (a b c : ℤ), x = (a, b, c) ∧ (a^2 * b^2 + b^2 * c^2 + c^2 * a^2 + 1) % p = 0 }

def S2 (p : ℕ) : set (ℤ × ℤ × ℤ) :=
  { x | ∃ (a b c : ℤ), x = (a, b, c) ∧ (a^2 * b^2 * c^2 * (a^2 * b^2 * c^2 + a^2 + b^2 + c^2)) % p = 0 }

theorem prime_sets_subseteq (p : ℕ) : 
  S1 p ⊆ S2 p ↔ p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 13 ∨ p = 17 :=
by
  sorry

end prime_sets_subseteq_l620_620427


namespace greatest_three_digit_multiple_of_17_l620_620323

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → m ≤ n) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620323


namespace floor_ceil_sqrt_sum_l620_620890

theorem floor_ceil_sqrt_sum (n : ℕ) (h : n = 50) : 
 (\lfloor Real.sqrt n \rfloor)^2 + (\lceil Real.sqrt n \rceil)^2 = 113 := 
by 
  sorry

end floor_ceil_sqrt_sum_l620_620890


namespace log_eq_solution_l620_620600

open Real

noncomputable def solve_log_eq : Real :=
  let x := 62.5^(1/3)
  x

theorem log_eq_solution (x : Real) (hx : 3 * log x - 4 * log 5 = -1) :
  x = solve_log_eq :=
by
  sorry

end log_eq_solution_l620_620600


namespace ab_equals_six_l620_620041

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end ab_equals_six_l620_620041


namespace determine_a_l620_620954

theorem determine_a (a : ℝ) (f : ℝ → ℝ) (h : f = fun x => a * x^3 - 2 * x) (pt : f (-1) = 4) : a = -2 := by
  sorry

end determine_a_l620_620954


namespace pentagon_area_l620_620876

theorem pentagon_area
  (XY UZ : ℝ) (YZ ZW : ℝ) (XU : ℝ)
  (hXY : XY = 4) (hUZ : UZ = 4)
  (hYZ : YZ = 5) (hZW : ZW = 5)
  (hXU : XU = 6)
  (inscribed_circle : ∀ P Q R S T, 
    tangent_point P XY ∧ tangent_point Q YZ ∧ tangent_point R ZW ∧ tangent_point S WU ∧ tangent_point T UX) :
  area_pentagon XYZWU = 45.65 :=
by
  sorry

end pentagon_area_l620_620876


namespace Simplified_G_l620_620111

noncomputable def F (x : ℝ) : ℝ := log ((1 + x) / (1 - x))

noncomputable def G (x : ℝ) : ℝ := log ((1 + (2 * x) / (1 + x^2)) / (1 - (2 * x) / (1 + x^2)))

theorem Simplified_G (x : ℝ) : G x = 2 * F x := by
  sorry

end Simplified_G_l620_620111


namespace largest_constant_inequality_l620_620536

theorem largest_constant_inequality :
  ∃ C : ℝ, (∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z)) ∧ C = Real.sqrt (4 / 3) :=
by {
  sorry
}

end largest_constant_inequality_l620_620536


namespace sum_reciprocal_s_n_l620_620759

theorem sum_reciprocal_s_n (a_3 : ℕ = 3) (S_4 : ℕ = 10) : 
  (∑ k in finset.range n, 1 / (S k)) = (2 * n) / (n + 1) := 
by sorry

end sum_reciprocal_s_n_l620_620759


namespace vector_proof_l620_620926

-- Define vectors and their magnitudes
variables (a b : ℝ^3)
variables (ha : ‖a‖ = 1)
variables (hb : ‖b‖ = 3)
variables (h_opposite : b = - (3 : ℝ) • a)

-- The target theorem
theorem vector_proof (a b : ℝ^3) (ha : ‖a‖ = 1) (hb : ‖b‖ = 3) (h_opposite : b = -3 • a) : b = -3 • a :=
h_opposite

end vector_proof_l620_620926


namespace distance_from_A_to_y_axis_l620_620181

variable (x y : ℝ)

def distance_to_y_axis (x : ℝ) : ℝ := |x|

theorem distance_from_A_to_y_axis (hx : x = -2) (hy : y = 1) :
  distance_to_y_axis x = 2 :=
by
  rw [hx]
  simp [distance_to_y_axis]
  norm_num

#eval distance_from_A_to_y_axis (by rfl) (by rfl)

end distance_from_A_to_y_axis_l620_620181


namespace probability_of_h_l620_620219

noncomputable def word : String := "Mathematics"

def total_number_of_letters (w : String) : Nat := String.length w

def count_letter_h (w : String) : Nat := w.toList.count ('h' ==)

theorem probability_of_h :
  total_number_of_letters word = 11 → 
  count_letter_h word = 1 →
  (count_letter_h word) / (total_number_of_letters word : ℝ) = (1 / 11 : ℝ) :=
by
  intros
  sorry

end probability_of_h_l620_620219


namespace largest_three_digit_multiple_of_17_l620_620272

theorem largest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n ≤ 999) ∧ (n ≥ 100) ∧ (17 ∣ n) ∧ ∀ m, ((m > n) ∧ (m ≤ 999) ∧ (m ≥ 100) ∧ (17 ∣ m)) → false :=
by {
  sorry
}

end largest_three_digit_multiple_of_17_l620_620272


namespace remainder_of_2_pow_2017_mod_11_l620_620404

theorem remainder_of_2_pow_2017_mod_11 : (2 ^ 2017) % 11 = 7 := by
  sorry

end remainder_of_2_pow_2017_mod_11_l620_620404


namespace ab_equals_six_l620_620022

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_l620_620022


namespace remaining_stickers_l620_620770

def stickers_per_page : ℕ := 20
def pages : ℕ := 12
def lost_pages : ℕ := 1

theorem remaining_stickers : 
  (pages * stickers_per_page - lost_pages * stickers_per_page) = 220 :=
  by
    sorry

end remaining_stickers_l620_620770


namespace train_B_speed_is_80_l620_620228

-- Definitions
def speed_A : ℝ := 60   -- Train A's speed in mph
def time_A_before_B : ℝ := 2 / 3  -- Time in hours Train A left before Train B (40 minutes = 2/3 hours)
def overtaking_time : ℝ := 2  -- Time in hours for Train B to overtake Train A (120 minutes = 2 hours)

-- Proof statement
theorem train_B_speed_is_80
  (speed_A_dist : speed_A * time_A_before_B = 40)  -- Distance covered by Train A in initial 40 minutes
  (additional_A_dist : speed_A * overtaking_time = 120)  -- Additional distance covered by Train A in next 2 hours
  (total_dist : 40 + 120 = 160) -- Total distance Train A covered when overtaken
  : ∃ (speed_B : ℝ), speed_B * overtaking_time = 160 ∧ speed_B = 80 := sorry

end train_B_speed_is_80_l620_620228


namespace popularity_order_l620_620644

noncomputable def frac : ℚ → ℤ × ℤ
| ⟨_, ⟨n, d, h⟩⟩ := (n, d)

theorem popularity_order :
  let projects := [("Park Renovation", 9 / 24), 
                   ("New Library", 10 / 30),
                   ("Street Lighting", 7 / 21),
                   ("Community Garden", 8 / 24)] in
  let simplified_fractions := 
    [("Park Renovation", frac (9 / 24)),
     ("New Library", frac (10 / 30)),
     ("Street Lighting", frac (7 / 21)),
     ("Community Garden", frac (8 / 24))] in
  let ordered_projects := 
    [("Park Renovation", 9 / 24),
     ("Community Garden", 8 / 24),
     ("New Library", 8 / 24),
     ("Street Lighting", 8 / 24)] in
  sorted_by popularity (projects_to_level (simplify_fractions projects)) = ordered_projects :=
by
  sorry

end popularity_order_l620_620644


namespace frequency_of_heads_l620_620783

theorem frequency_of_heads :
    ∃ (W₁ W₂ W₃ : ℝ), 
    W₁ = 2048 / 4040 ∧ W₁ ≈ 0.5069 ∧
    W₂ = 6019 / 12000 ∧ W₂ ≈ 0.5016 ∧
    W₃ = 12012 / 24000 ∧ W₃ ≈ 0.5005 :=
by
    sorry

end frequency_of_heads_l620_620783


namespace simplify_expression_l620_620463

theorem simplify_expression : 
  8 - (-3) + (-5) + (-7) = 3 + 8 - 7 - 5 := 
by
  sorry

end simplify_expression_l620_620463


namespace planned_water_usage_correct_annual_fee_correct_l620_620163

-- Define the planned average monthly water usage as per the given problem
def planned_monthly_water_usage : ℝ :=
  let x := 20 in
  x

-- Define the condition equation for planned average monthly water usage
def condition_eq (x : ℝ) : Prop :=
  (480 / x) - (480 / (x + 4)) = 4

-- Prove that the solution to the above condition is x = 20
theorem planned_water_usage_correct : ∃ x : ℝ, condition_eq x ∧ x = planned_monthly_water_usage :=
begin
  use 20,
  split,
  { -- Verify the condition
    show condition_eq 20,
    -- Simplify the condition to validate it
    unfold condition_eq,
    simp [div_eq_mul_inv, mul_add, mul_comm, mul_assoc, add_mul],
    norm_num },
  { -- Validate that our planned monthly water usage is 20
    refl }
end

-- Definitions for the annual water fee calculation under new tariff conditions
def actual_water_usage (planned : ℝ) : ℝ := planned * 1.4
def tariff : ℝ := 1.9

-- Over quota and excess fee calculations according to the new payment method
noncomputable def annual_water_fee (planned : ℝ) (actual : ℝ) : ℝ :=
  8 * planned * tariff +
  4 * planned * tariff +
  4 * (actual - 20) * (tariff * 1.5)

-- Define that if the planned monthly usage is 20, the total annual fee should be 547.2 yuan
theorem annual_fee_correct :
  annual_water_fee planned_monthly_water_usage (actual_water_usage planned_monthly_water_usage) = 547.2 :=
begin
  unfold planned_monthly_water_usage actual_water_usage annual_water_fee,
  norm_num,
  -- Calculation verification 
  have h1 : 8 * 20 * 1.9 = 304, norm_num,
  have h2 : 4 * 20 * 1.9 = 152, norm_num,
  have h3 : 4 * 8 * (1.9 * 1.5) = 91.2, norm_num,
  rw [h1, h2, h3],
  norm_num
end

end planned_water_usage_correct_annual_fee_correct_l620_620163


namespace number_of_sandwiched_even_numbers_l620_620790

theorem number_of_sandwiched_even_numbers : 
  let digits := {1, 2, 3, 4} in
  let odd := {1, 3} in
  let even := {2, 4} in
  ∃ (n : ℕ), n = 8 ∧ 
    (∀ lst : list ℕ, lst.permutations.count 
      (λ l, ∃ o1 o2 e, o1 ∈ odd ∧ o2 ∈ odd ∧ e ∈ even ∧ 
                      l = [o1, e, o2] ∨ l = [o2, e, o1] ∧ 
                      ∀ n ∈ l, n ∈ digits) = n)
:= sorry -- proof omitted

end number_of_sandwiched_even_numbers_l620_620790


namespace terminating_decimal_fraction_count_l620_620561

/-- 
This theorem states that there are exactly 9 integer values of n 
between 1 and 565 inclusive such that the decimal representation of n / 570 terminates.
-/
theorem terminating_decimal_fraction_count : 
  {n : ℕ | 1 ≤ n ∧ n ≤ 565 ∧ ∃ k : ℕ, n = k * 57}.finite ∧ {n : ℕ | 1 ≤ n ∧ n ≤ 565 ∧ ∃ k : ℕ, n = k * 57}.to_finset.card = 9 :=
by
  sorry

end terminating_decimal_fraction_count_l620_620561


namespace problem_statement_l620_620448

-- Define constants and functions
def a : ℝ := -2
def b : ℝ := 2 * Real.sqrt(3)
def ω : ℝ := 4

def f (x : ℝ) : ℝ := a * Real.cos(ω * x) + b * Real.sin(ω * x)

-- Given conditions
def x_interval (x : ℝ) : Prop := Real.pi / 4 < x ∧ x < 3 * Real.pi / 4
def f_x_plus_pi_6 (x : ℝ) : Prop := f(x + Real.pi / 6) = 4 / 3

-- Prove the statement
theorem problem_statement (x : ℝ) (hx_int : x_interval x) (hx_f : f_x_plus_pi_6 x) : 
  f (x / 2 + Real.pi / 6) = -4 * Real.sqrt(6) / 3 := 
    sorry

end problem_statement_l620_620448


namespace percent_children_with_both_colors_l620_620817

theorem percent_children_with_both_colors
  (F : ℕ) (C : ℕ) 
  (even_F : F % 2 = 0)
  (children_pick_two_flags : C = F / 2)
  (sixty_percent_blue : 6 * C / 10 = 6 * C / 10)
  (fifty_percent_red : 5 * C / 10 = 5 * C / 10)
  : (6 * C / 10) + (5 * C / 10) - C = C / 10 :=
by
  sorry

end percent_children_with_both_colors_l620_620817


namespace dorothy_profit_l620_620887

def doughnut_expenses : ℕ := 53
def rent_utilities : ℕ := 27
def doughnuts_made : ℕ := 25
def price_per_doughnut : ℕ := 3

theorem dorothy_profit :
  let total_expenses := doughnut_expenses + rent_utilities
  let revenue := doughnuts_made * price_per_doughnut
  let profit := revenue - total_expenses
  profit = -5 := by
  sorry

end dorothy_profit_l620_620887


namespace PedrinhoMetOnThursday_l620_620102

def DayOfWeek := 
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

def WhatDayIsIt (d : DayOfWeek) : Prop := 
match d with
| DayOfWeek.Tuesday    => ∀ (s : String), s ≠ "Saturday" ∧ s ≠ "Wednesday"
| DayOfWeek.Thursday   => ∀ (s : String), s ≠ "Saturday" ∧ s ≠ "Wednesday"
| DayOfWeek.Saturday   => ∀ (s : String), s ≠ "Saturday" ∧ s ≠ "Wednesday"
| _                    => True

def LyingBehavior (d : DayOfWeek) : Prop := 
match d with
| DayOfWeek.Tuesday
| DayOfWeek.Thursday
| DayOfWeek.Saturday => True
| _                  => False

def TellingTruthBehavior (d : DayOfWeek) : Prop := ¬ LyingBehavior d

theorem PedrinhoMetOnThursday : 
(∃ d : DayOfWeek, LyingBehavior d ∧ WhatDayIsIt d "Saturday" ∧ WhatDayIsIt (nextDay d) "Wednesday") 
↔ d = DayOfWeek.Thursday :=
by
sorry

end PedrinhoMetOnThursday_l620_620102


namespace no_incorrect_conclusions_l620_620617

theorem no_incorrect_conclusions (a b c : ℝ) (h1 : 2^a = 24) (h2 : 2^b = 6) (h3 : 2^c = 9) :
  (a - b = 2) ∧ (3 * b = a + c) ∧ (2 * b - c = 2) → 0 = 0 :=
by {
  intro h,
  exact sorry,
}

end no_incorrect_conclusions_l620_620617


namespace tangent_line_at_2_tangent_line_at_A_l620_620606

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 4 * x^2 + 5 * x - 4

-- Define the derivative f'
def f' (x : ℝ) : ℝ := 3 * x^2 - 8 * x + 5

-- Equation of the tangent line to f(x) at point (2, f(2))
theorem tangent_line_at_2 :
  ∀ x y : ℝ, (x = 2 → y = f 2) → x - y - 4 = 0 :=
by 
  intros x y h,
  
-- Equation of the tangent line to f(x) that passes through point A(2, -2)
theorem tangent_line_at_A :
  ∃ x0 : ℝ, (x0 = 2 ∨ x0 = 1) →
  ∀ x y : ℝ, (x = 2 → y = -2) →
  (x - y - 4 = 0 ∨ y + 2 = 0) :=
by 
  have h1 : f 2 = -2 := sorry,
  have h2 : f' 2 = 1 := sorry,
  have h3 : f 1 = -2 := sorry,
  have h4 : f' 1 = 0 := sorry,
  intros x0 h x y p,
  sorry 

end tangent_line_at_2_tangent_line_at_A_l620_620606


namespace no_value_of_a_l620_620553

theorem no_value_of_a (a : ℝ) (x y : ℝ) : ¬∃ x1 x2 : ℝ, (x1 ≠ x2) ∧ (x1^2 + y^2 + 2 * x1 = abs (x1 - a) - 1) ∧ (x2^2 + y^2 + 2 * x2 = abs (x2 - a) - 1) := 
by
  sorry

end no_value_of_a_l620_620553


namespace regression_equation_not_meaningful_l620_620634

-- Definitions based on conditions
variable (X Y : Type)
variable [MetricSpace X] [MetricSpace Y]

-- Regression equation meaningfulness is incorrect
theorem regression_equation_not_meaningful (x y : X) (scatter_points : Set X) (h_corr : Correlated x y) (h_linearity : ∀ (p ∈ scatter_points), ∃ (line : X → Y), (metric_space_dist p.line) < ε) : 
  ¬ meaningful (regression_equation scatter_points) := 
sorry

end regression_equation_not_meaningful_l620_620634


namespace ratio_of_r_to_s_l620_620528

-- Defining the right triangle with legs a and b, and hypotenuse c
variables (a b c r s : ℝ)
-- Assuming the given ratio of the legs
variables (h_ab_ratio: a / b = 2 / 5)
-- Assuming the right triangle properties: the hypotenuse and dropping perpendicular
variables (h_c: c = real.sqrt (a^2 + b^2))
variables (h_r: r = a^2 / c)
variables (h_s: s = b^2 / c)

-- Stating the goal to be proven
theorem ratio_of_r_to_s : r / s = 4 / 25 :=
by sorry

end ratio_of_r_to_s_l620_620528


namespace greatest_three_digit_multiple_of_17_l620_620312

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → m ≤ n) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620312


namespace conditional_probability_20_to_25_l620_620945

-- Define events and probabilities
variables (A B : Prop)
variables (P : Prop → ℝ)

-- Given conditions as hypotheses
axiom h1 : P B = 0.9
axiom h2 : P A = 0.5

-- Define conditional probability
noncomputable def conditional_probability (A B : Prop) (P : Prop → ℝ) :=
  P A / P B

-- Formulate the main statement
theorem conditional_probability_20_to_25 (A B : Prop) (P : Prop → ℝ) (h1 : P B = 0.9) (h2 : P A = 0.5) :
  conditional_probability A B P = 5 / 9 :=
by
  unfold conditional_probability
  rw [h1, h2]
  norm_num
  sorry

end conditional_probability_20_to_25_l620_620945


namespace greatest_three_digit_multiple_of_17_l620_620239

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n, (n ∈ set.Icc 100 999) ∧ (n % 17 = 0) ∧ (∀ m, (m ∈ set.Icc 100 999 ∧ m % 17 = 0) → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620239


namespace blankets_warmth_increase_l620_620408

-- Conditions
def blankets_in_closet : ℕ := 14
def blankets_used : ℕ := blankets_in_closet / 2
def degree_per_blanket : ℕ := 3

-- Goal: Prove that the total temperature increase is 21 degrees.
theorem blankets_warmth_increase : blankets_used * degree_per_blanket = 21 :=
by
  sorry

end blankets_warmth_increase_l620_620408


namespace extracurricular_popularity_order_l620_620853

def fraction_likes_drama := 9 / 28
def fraction_likes_music := 13 / 36
def fraction_likes_art := 11 / 24

theorem extracurricular_popularity_order :
  fraction_likes_art > fraction_likes_music ∧ 
  fraction_likes_music > fraction_likes_drama :=
by
  sorry

end extracurricular_popularity_order_l620_620853


namespace blankets_warmth_increase_l620_620407

-- Conditions
def blankets_in_closet : ℕ := 14
def blankets_used : ℕ := blankets_in_closet / 2
def degree_per_blanket : ℕ := 3

-- Goal: Prove that the total temperature increase is 21 degrees.
theorem blankets_warmth_increase : blankets_used * degree_per_blanket = 21 :=
by
  sorry

end blankets_warmth_increase_l620_620407


namespace rawhide_bone_cost_l620_620858

theorem rawhide_bone_cost 
  (cost_per_biscuit : ℝ) 
  (biscuits_per_day : ℕ) 
  (bones_per_day : ℕ) 
  (weekly_cost : ℝ) 
  (bone_cost : ℝ) 
  (daily_biscuit_cost : ℝ) 
  (total_weekly_cost_eq : ∀ bone_cost, weekly_cost = 7 * (daily_biscuit_cost + bones_per_day * bone_cost))
  : bone_cost = 1 := 
by 
  have daily_biscuit_cost := biscuits_per_day * cost_per_biscuit,
  have total_weekly_cost := 7 * (daily_biscuit_cost + bones_per_day * bone_cost),
  sorry

end rawhide_bone_cost_l620_620858


namespace olympic_inequality_l620_620086

variable (a b k : ℕ)

theorem olympic_inequality (h_odd : b % 2 = 1) (h_b_ge_3 : 3 ≤ b) 
  (h_k_condition : ∀ (E1 E2 S : ℕ), E1 ≠ E2 → count (E1, E2, S) ≤ k) :
  (k : ℚ) / a ≥ (b - 1 : ℚ) / (2 * b) :=
by
  sorry

end olympic_inequality_l620_620086


namespace algebraic_expression_value_l620_620713

theorem algebraic_expression_value
  (a b x c d : ℝ)
  (h1 : a + b = 0)
  (h2 : c * d = 1)
  (h3 : |x| = Real.sqrt 3) :
  x^2 + Real.sqrt (a + b + 4) - (27 * c * d)^(1/3) = 2 :=
begin
  sorry
end

end algebraic_expression_value_l620_620713


namespace molecularWeight_correct_l620_620794

noncomputable def molecularWeight (nC nH nO nN: ℤ) 
    (wC wH wO wN : ℚ) : ℚ := nC * wC + nH * wH + nO * wO + nN * wN

theorem molecularWeight_correct : 
    molecularWeight 5 12 3 1 12.01 1.008 16.00 14.01 = 134.156 := by
  sorry

end molecularWeight_correct_l620_620794


namespace midpoint_translation_proof_l620_620746

def Point := (ℝ × ℝ)

def triangle_b := (1 : ℝ, 1 : ℝ)
def triangle_i := (2 : ℝ, 4 : ℝ)
def triangle_g := (5 : ℝ, 1 : ℝ)

def translate_left_up (p : Point) (left_units up_units : ℝ) : Point :=
  (p.1 - left_units, p.2 + up_units)

def midpoint (p1 p2 : Point) : Point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def triangle_big_midpoint := midpoint triangle_b triangle_g

def translation_vector_b' := translate_left_up triangle_big_midpoint 4 3

theorem midpoint_translation_proof :
  translation_vector_b' = (-1, 4) :=
  sorry

end midpoint_translation_proof_l620_620746


namespace arc_length_of_arc_l620_620173

theorem arc_length_of_arc (r : ℝ) (theta : ℝ) (h_r : r = real.pi) (h_theta : theta = 2 * real.pi / 3) :
  let l := abs theta * r in l = (2 / 3) * real.pi ^ 2 :=
by 
  sorry

end arc_length_of_arc_l620_620173


namespace central_angle_probability_l620_620442

theorem central_angle_probability (A : ℝ) (x : ℝ)
  (h1 : A > 0)
  (h2 : (x / 360) * A / A = 1 / 8) : 
  x = 45 := 
by
  sorry

end central_angle_probability_l620_620442


namespace correct_sampling_method_and_senior_count_l620_620449

noncomputable def total_students : ℕ := 1600
noncomputable def freshmen : ℕ := 520
noncomputable def sophomores : ℕ := 500
noncomputable def seniors : ℕ := 580
noncomputable def sample_size : ℕ := 80

theorem correct_sampling_method_and_senior_count :
  (AppropriateSamplingMethod = "stratified") ∧
  (NumberOfSeniorsToSurvey = 29) :=
by
  let AppropriateSamplingMethod := "stratified"
  let NumberOfSeniorsToSurvey := (29 * (sample_size / ({freshmen, sophomores, seniors}.sum / 80)) : ℕ)
  have h1 : (26 + 25 + 29) = 80 := by norm_num
  have h2 : (sample_size / 80) = 1 := by norm_num
  sorry

end correct_sampling_method_and_senior_count_l620_620449


namespace product_of_ab_l620_620051

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end product_of_ab_l620_620051


namespace complex_number_magnitude_l620_620688

theorem complex_number_magnitude (z : ℂ) (h : (1 + complex.i) * z = -3 + complex.i) : complex.abs z = real.sqrt 5 :=
sorry

end complex_number_magnitude_l620_620688


namespace negation_proposition_l620_620708

theorem negation_proposition (a b : ℤ) (h : a ≤ b) : 2^a ≤ 2^b - 1 :=
sorry

end negation_proposition_l620_620708


namespace card_game_fairness_l620_620789

theorem card_game_fairness :
  let deck_size := 52
  let aces := 2
  let total_pairings := Nat.choose deck_size aces  -- Number of ways to choose 2 positions from 52
  let tie_cases := deck_size - 1                  -- Number of ways for consecutive pairs
  let non_tie_outcomes := total_pairings - tie_cases
  non_tie_outcomes / 2 = non_tie_outcomes / 2
:= sorry

end card_game_fairness_l620_620789


namespace Cora_book_reading_problem_l620_620532

theorem Cora_book_reading_problem
  (total_pages: ℕ)
  (read_monday: ℕ)
  (read_tuesday: ℕ)
  (read_wednesday: ℕ)
  (H: total_pages = 158 ∧ read_monday = 23 ∧ read_tuesday = 38 ∧ read_wednesday = 61) :
  ∃ P: ℕ, 23 + 38 + 61 + P + 2 * P = total_pages ∧ P = 12 :=
  sorry

end Cora_book_reading_problem_l620_620532


namespace find_a_for_perpendicular_lines_l620_620637

theorem find_a_for_perpendicular_lines 
  (a : ℝ) 
  (h_perpendicular : ∀ x y : ℝ, ax + 2y + 6 = 0 ∧ x + a*(a+1)*y + (a^2 - 1) = 0) : 
  a = 0 ∨ a = -3/2 :=
sorry

end find_a_for_perpendicular_lines_l620_620637


namespace michael_initial_fish_count_l620_620692

def initial_fish_count (total_fish now : nat) (fish_given_by_ben : nat) : nat :=
  now - fish_given_by_ben

theorem michael_initial_fish_count :
  ∀ (now fish_given_by_ben : nat), 
  now = 49 ∧ fish_given_by_ben = 18 → 
  initial_fish_count now fish_given_by_ben = 31 :=
by
  intros now fish_given_by_ben h
  cases h with h_now h_ben
  rw [h_now, h_ben]
  exact rfl

end michael_initial_fish_count_l620_620692


namespace ab_equals_six_l620_620000

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l620_620000


namespace prime_even_intersection_l620_620679

-- Define P as the set of prime numbers
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def P : Set ℕ := { n | is_prime n }

-- Define Q as the set of even numbers
def Q : Set ℕ := { n | n % 2 = 0 }

-- Statement to prove
theorem prime_even_intersection : P ∩ Q = {2} :=
by
  sorry

end prime_even_intersection_l620_620679


namespace shaded_region_area_l620_620755

theorem shaded_region_area (RS : ℝ) (n_shaded : ℕ)
  (h1 : RS = 10) (h2 : n_shaded = 20) :
  (20 * (RS / (2 * Real.sqrt 2))^2) = 250 :=
by
  sorry

end shaded_region_area_l620_620755


namespace proof_area_of_circle_section_proof_angle_ACB_proof_volume_of_prism_l620_620174

noncomputable def rhombus_base_conditions : Prop :=
    let BD := 12
    let angle_BAC := 60 * Real.pi / 180
    true

noncomputable def sphere_vertices_passed_through : Prop :=
    true

noncomputable def radius_of_sphere_eq_8 : Prop :=
    let radius := 8
    true

noncomputable def area_of_circle_section (BD : ℝ) (angle_BAC : ℝ) : ℝ :=
    let r := 6 * Real.tan (30 * Real.pi / 180)
    Real.pi * (4 * Real.sqrt 3)^2

theorem proof_area_of_circle_section :
  rhombus_base_conditions ∧ sphere_vertices_passed_through →
  area_of_circle_section 12 (60 * Real.pi / 180) = 48 * Real.pi := by
    intros
    sorry

theorem proof_angle_ACB :
  rhombus_base_conditions ∧ sphere_vertices_passed_through →
  ∠ (pointA1 pointC pointB) = 90 := by
    intros
    sorry

noncomputable def volume_of_prism (BD : ℝ) (angle_BAC : ℝ) (radius : ℝ) : ℝ :=
    let AC : ℝ := 2 * 6 * Real.tan (30 * Real.pi / 180)
    let height := 2 * (Real.sqrt (radius^2 - (AC/2)^2))
    let base_area := (1/2) * AC * BD
    base_area * height

theorem proof_volume_of_prism :
  rhombus_base_conditions ∧ sphere_vertices_passed_through ∧ radius_of_sphere_eq_8 →
  volume_of_prism 12 (60 * Real.pi / 180) 8 = 192 * Real.sqrt 3 := by
    intros
    sorry

end proof_area_of_circle_section_proof_angle_ACB_proof_volume_of_prism_l620_620174


namespace car_transport_capacity_calculate_total_goods_l620_620618

theorem car_transport_capacity (x y : ℕ) (h1 : 2 * x + y = 10) (h2 : x + 2 * y = 11) :
  ((x = 3) ∧ (y = 4)) :=
by
  sorry

theorem calculate_total_goods (h : car_transport_capacity 3 4 10 11) : 
  (6 * 3 + 8 * 4 = 50) :=
by
  sorry

end car_transport_capacity_calculate_total_goods_l620_620618


namespace work_completion_time_l620_620784

theorem work_completion_time (A B C : ℝ) (hA : A = 10) (hB : B = 15) (hC : C = 20) :
  (1 / (1 / A + 1 / B + 1 / C)) = 60 / 13 :=
by
  -- Workers' work rates per day
  have work_rate_A := 1 / A,
  have work_rate_B := 1 / B,
  have work_rate_C := 1 / C,

  -- Combined work rate
  have combined_work_rate := work_rate_A + work_rate_B + work_rate_C,

  -- Time to complete the job
  have time_to_complete_job := 1 / combined_work_rate,

  -- Simplifying the expressions
  have work_rate_A_eq := by rw [hA, show (1 : ℝ) / 10 = 0.1, from rfl],
  have work_rate_B_eq := by rw [hB, show (1 : ℝ) / 15 = 1 / 15, from rfl],
  have work_rate_C_eq := by rw [hC, show (1 : ℝ) / 20 = 1 / 20, from rfl],

  have combined_work_rate_eq := by rw [work_rate_A_eq, work_rate_B_eq, work_rate_C_eq,
    calc (0.1 + 1 / 15 + 1 / 20) = (6 / 60) + (4 / 60) + (3 / 60) : by norm_num,
                                 = 13 / 60 : by norm_num],

  show (1 / (1 / A + 1 / B + 1 / C)) = 60 / 13 from by rw [combined_work_rate_eq, ← div_eq_inv_mul,
    (show inv_div, by simp), show inv (13 / 60) = 60 / 13, from rfl, time_to_complete_job_eq],

  sorry

end work_completion_time_l620_620784


namespace quadratic_has_real_roots_find_m_given_roots_l620_620601

-- Define the quadratic equation
def quadratic_eq (x m : ℝ) : ℝ := x^2 + 2 * (2 - m) * x + 3 - 6 * m

-- (1) Prove that no matter what real number $m$ is, the equation always has real roots.
theorem quadratic_has_real_roots (m : ℝ) : ∃ x1 x2 : ℝ, quadratic_eq x1 m = 0 ∧ quadratic_eq x2 m = 0 :=
by {
  let Δ := 4 * (m + 1)^2,
  have h : Δ ≥ 0,
  { linarith },
  -- Using the property of the quadratic discriminant being non-negative
  rw [quadratic_eq],
  sorry
}

-- (2) If the two real roots of the equation, x1 and x2, satisfy x

1 = 3x2, find the value of the real number m.
theorem find_m_given_roots (m : ℝ) (x1 x2 : ℝ) (h_eq : quadratic_eq x1 m = 0 ∧ quadratic_eq x2 m = 0 ∧ x1 = 3 * x2) :
  m = 0 ∨ m = -4 :=
by {
  -- Using the conditions and solving for m
  have h_sum : (x1 + x2 = 2 * m - 4) := sorry,
  have h_prod : (x1 * x2 = 3 * x2^2 = 3 - 6 * m) := sorry,
  -- Equating and solving for m proving that m = 0 or m = -4
  let x2_val := (m - 2) / 2,
  let eq1 := h_sum,
  let eq2 := h_prod,
  have eq_final := eq1 * eq1 = eq2, -- Simplifying the equations
  sorry
}

end quadratic_has_real_roots_find_m_given_roots_l620_620601


namespace distance_correct_l620_620801

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_correct : ∀ a : ℝ, distance (-3, -a) (-3, 2 - a) = 2 :=
by
  sorry

end distance_correct_l620_620801


namespace ab_value_l620_620014

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_value_l620_620014


namespace determine_r_l620_620690

-- Definitions of vectors a and b
def a : ℝ × ℝ × ℝ := (4, 2, -3)
def b : ℝ × ℝ × ℝ := (1, 3, -2)

-- Given vector c and the equation relating c, a, b, and r
def c : ℝ × ℝ × ℝ := (5, 4, -8)

-- Cross product of a and b
def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2.2 * v.2.1 - u.2.1 * v.2.2, u.2.0 * v.2.2 - u.2.2 * v.2.0, u.2.1 * v.2.0 - u.2.0 * v.2.1)

-- Dot product of two vectors
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.2.0 * v.2.0 + u.2.1 * v.2.1 + u.2.2 * v.2.2

-- Cross product of a and b
def cp_ab := cross_product a b

-- Given equation in the problem setup
def given_eq (p q r : ℝ) : Prop :=
  let term1 := (p * a.2.0, p * a.2.1, p * a.2.2)
  let term2 := (q * b.2.0, q * b.2.1, q * b.2.2)
  let term3 := (r * cp_ab.2.0, r * cp_ab.2.1, r * cp_ab.2.2)
  (term1.0 + term2.0 + term3.0 = c.2.0) ∧ (term1.1 + term2.1 + term3.1 = c.2.1) ∧ (term1.2 + term2.2 + term3.2 = c.2.2)

-- Statement to prove
theorem determine_r (p q r : ℝ) (h : given_eq p q r) : r = -4 / 5 :=
  sorry

end determine_r_l620_620690


namespace cat_food_sufficiency_l620_620521

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S >= 11 :=
sorry

end cat_food_sufficiency_l620_620521


namespace max_value_of_function_zero_l620_620077

theorem max_value_of_function_zero (a : ℝ) :
  (∀ x : ℝ, (-2) ≤ 2 * sin (x + π / 6) + a) ∧ (∃ x : ℝ, 2 * sin (x + π / 6) + a = 0) →
  a = -2 := 
by 
  sorry

end max_value_of_function_zero_l620_620077


namespace error_arrangements_l620_620076

-- Definitions based on the conditions
def word := "error"
def total_letters := 5
def r_repeats := 3
def e_repeats := 1
def o_repeats := 1

-- Helper function to calculate factorial
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Define the function for permutation of a multiset
def P (n : ℕ) (n1 n2 n3 : ℕ) : ℕ :=
  fact n / (fact n1 * fact n2 * fact n3)

theorem error_arrangements : P total_letters r_repeats e_repeats o_repeats - 1 = 19 := by
  -- We use the given conditions directly in the theorem
  sorry

end error_arrangements_l620_620076


namespace transformed_log_is_correct_l620_620785

theorem transformed_log_is_correct (f : ℝ → ℝ)
  (h₁ : ∀ x, f (x + 1) = log x)
  (h₂ : ∀ x, f (-x) = log x) :
  ∀ x, f x = log (1 - x) :=
by sorry

end transformed_log_is_correct_l620_620785


namespace infinite_parallel_lines_through_point_l620_620197

noncomputable def number_of_parallel_lines_through_point (P : Set (Set Point)) (A : Point) : ℕ → Prop :=
  ∀ n, n = 0 ∨ n = 1 ∨ n = 2 ∨ ∃ k, k > 2 ∧ n = k ∨ n = ⊤

theorem infinite_parallel_lines_through_point (P : Set (Set Point)) (A : Point) (h1 : A ∉ P): number_of_parallel_lines_through_point P A ⊤ :=
by
  sorry

end infinite_parallel_lines_through_point_l620_620197


namespace correct_average_is_18_l620_620424

theorem correct_average_is_18 (incorrect_avg : ℕ) (incorrect_num : ℕ) (true_num : ℕ) (n : ℕ) 
  (h1 : incorrect_avg = 16) (h2 : incorrect_num = 25) (h3 : true_num = 45) (h4 : n = 10) : 
  (incorrect_avg * n + (true_num - incorrect_num)) / n = 18 :=
by
  sorry

end correct_average_is_18_l620_620424


namespace volunteer_distribution_l620_620471

theorem volunteer_distribution :
  let teachers := 3
  let schools := 4
  let max_per_school := 2
  let plans := 60
  in ∃ plans, (teachers ≤ schools * max_per_school) ∧ plans = 60 :=
by
  sorry

end volunteer_distribution_l620_620471


namespace greatest_three_digit_multiple_of_17_l620_620290

def is_multiple_of (n : ℕ) (k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, is_three_digit n ∧ is_multiple_of n 17 → n = 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620290


namespace ab_equals_six_l620_620023

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_l620_620023


namespace angles_equal_l620_620649

-- Definitions of the given points and lines
variables (A B C D E F G : Point)
variables (γ : Quadrilateral A B C D)

-- Conditions specified in the problem
variable (h1 : Bisects AC (Angle A B D))
variable (h2 : On E (Line C D))
variable (h3 : Intersects BE AC F)
variable (h4 : Intersects (Extension DF) BC G)

-- Target statement to prove
theorem angles_equal : Angle GAC = Angle EAC :=
by
  sorry

end angles_equal_l620_620649


namespace range_of_f_l620_620558

noncomputable def f (x : ℝ) : ℝ := (x^2 - 1/2*x) / (x^2 - x + 1)

theorem range_of_f : set.range f = set.Icc (2/5) 6 := 
begin
  sorry
end

end range_of_f_l620_620558


namespace right_triangle_AB_l620_620091

theorem right_triangle_AB 
  {A B C : Type}
  [EuclideanGeometry A B C]
  (AC BC : ℝ)
  (h1 : angle A C = 90)
  (h2 : AC = 6)
  (h3 : tan(A) = 3/2) :
  AB = 3 * sqrt 13 := 
sorry

end right_triangle_AB_l620_620091


namespace max_popsicles_is_13_l620_620700

/-- Pablo's budgets and prices for buying popsicles. -/
structure PopsicleStore where
  single_popsicle_cost : ℕ
  three_popsicle_box_cost : ℕ
  five_popsicle_box_cost : ℕ
  starting_budget : ℕ

/-- The maximum number of popsicles Pablo can buy given the store's prices and his budget. -/
def maxPopsicles (store : PopsicleStore) : ℕ :=
  let num_five_popsicle_boxes := store.starting_budget / store.five_popsicle_box_cost
  let remaining_after_five_boxes := store.starting_budget % store.five_popsicle_box_cost
  let num_three_popsicle_boxes := remaining_after_five_boxes / store.three_popsicle_box_cost
  let remaining_after_three_boxes := remaining_after_five_boxes % store.three_popsicle_box_cost
  let num_single_popsicles := remaining_after_three_boxes / store.single_popsicle_cost
  num_five_popsicle_boxes * 5 + num_three_popsicle_boxes * 3 + num_single_popsicles

theorem max_popsicles_is_13 :
  maxPopsicles { single_popsicle_cost := 1, 
                 three_popsicle_box_cost := 2, 
                 five_popsicle_box_cost := 3, 
                 starting_budget := 8 } = 13 := by
  sorry

end max_popsicles_is_13_l620_620700


namespace exists_ab_kth_power_l620_620544

theorem exists_ab_kth_power :
  (∃ a b : ℕ, a > b ∧ b > 1 ∧ 
    ∀ k : ℕ, k > 0 → ∃ n : ℕ, (an + b) = (m ^ k) for some m with m ∈ ℕ) ∧
  ((a = 6 ∧ b = 4) ∨ (a = 10 ∧ b = 5)) := 
sorry

end exists_ab_kth_power_l620_620544


namespace min_distance_PQ_l620_620592

noncomputable def minimum_distance (x : ℝ) : ℝ :=
  abs ((1 / 2) * real.exp x - x) / real.sqrt 2

theorem min_distance_PQ :
  ∃ x : ℝ, (P : ℝ × ℝ) := (x, (1 / 2) * real.exp x) →
  (Q : ℝ × ℝ) := (real.exp x / 2, x) →
  is_minimum ((2 * minimum_distance x) = real.sqrt 2 * (1 - real.log 2)) :=
sorry

end min_distance_PQ_l620_620592


namespace greatest_three_digit_multiple_of_17_l620_620238

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n, (n ∈ set.Icc 100 999) ∧ (n % 17 = 0) ∧ (∀ m, (m ∈ set.Icc 100 999 ∧ m % 17 = 0) → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620238


namespace expression_evaluation_l620_620869

def e : Int := -(-1) + 3^2 / (1 - 4) * 2

theorem expression_evaluation : e = -5 := 
by
  unfold e
  sorry

end expression_evaluation_l620_620869


namespace total_elephants_in_two_parks_is_280_l620_620194

def number_of_elephants_we_preserve_for_future : ℕ := 70
def multiple_factor : ℕ := 3

def number_of_elephants_gestures_for_good : ℕ := multiple_factor * number_of_elephants_we_preserve_for_future

def total_number_of_elephants : ℕ := number_of_elephants_we_preserve_for_future + number_of_elephants_gestures_for_good

theorem total_elephants_in_two_parks_is_280 : total_number_of_elephants = 280 :=
by
  sorry

end total_elephants_in_two_parks_is_280_l620_620194


namespace triangle_ratio_AD_AB_example_l620_620992

noncomputable def ratio_AD_AB (A B C D E : Point) (angle_A angle_B angle_ADE : ℝ) 
  (DE_divide_equal_area : divides_into_equal_area ↔ DE_intersect_AC) 
  (is_on_AB : lies_on D AB) : Prop :=
  angle_A = 45 ∧ angle_B = 30 ∧ angle_ADE = 60 ∧
  (∃ r : ℝ, r = 1 / real.sqrt (real.sqrt 12) ∧ AD / AB = r)

-- The main theorem statement:
theorem triangle_ratio_AD_AB_example (A B C D E : Point) 
  (angle_A angle_B angle_ADE : ℝ) (DE_divide_equal_area : divides_into_equal_area ↔ DE_intersect_AC) 
  (is_on_AB : lies_on D AB) :
  angle_A = 45 →
  angle_B = 30 →
  angle_ADE = 60 →
  (∃ r : ℝ, r = 1 / real.sqrt (real.sqrt 12) ∧ AD / AB = r) :=
begin
  -- Structure of proof goes here
  sorry,
end

end triangle_ratio_AD_AB_example_l620_620992


namespace smallest_positive_integer_M_exists_l620_620907

theorem smallest_positive_integer_M_exists : 
  ∃ M: ℕ, (M > 0) ∧ 
  (∃ i ∈ ({0, 1, 2, 3} : set ℕ), (M + i) % 8 = 0) ∧
  (∃ j ∈ ({0, 1, 2, 3} : set ℕ), (M + j) % 9 = 0) ∧
  (∃ k ∈ ({0, 1, 2, 3} : set ℕ), (M + k) % 25 = 0) ∧
  (∃ l ∈ ({0, 1, 2, 3} : set ℕ), (M + l) % 49 = 0) :=
exists.intro 196 (by
  have h0 : (196 + 0) % 49 = 0 := by norm_num,
  have h1 : (198 + 0) % 9 = 0 := by norm_num,
  have h2 : (200 + 0) % 8 = 0 := by norm_num,
  have h3 : (200 + 0) % 25 = 0 := by norm_num,
  simp [h0, h1, h2, h3],
  sorry)

end smallest_positive_integer_M_exists_l620_620907


namespace candle_problem_l620_620911

theorem candle_problem (n : ℕ) :
  (∃ (lights : ℕ → ℕ), (∀ i, i ∈ fin n.succ → lights i = i + 1) ∧ n = 2 * k + 1 → lights (n - i) = 0) → n % 2 = 1 :=
by
  sorry

end candle_problem_l620_620911


namespace greatest_three_digit_multiple_of_17_l620_620336

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

theorem greatest_three_digit_multiple_of_17 : ∀ n, n < 1000 → is_multiple_of n 17 → n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620336


namespace ab_value_l620_620036

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_value_l620_620036


namespace sufficient_food_supply_l620_620501

variable {L S : ℝ}

theorem sufficient_food_supply (h1 : L + 4 * S = 14) (h2 : L > S) : L + 3 * S ≥ 11 :=
by
  sorry

end sufficient_food_supply_l620_620501


namespace greatest_three_digit_multiple_of_17_l620_620269

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (17 ∣ n) ∧ ∀ m : ℤ, (100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m) → m ≤ n :=
begin
  use 969,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num, exact dec_trivial, },
  { intros m h_mconds,
    cases h_mconds with hm1 h_mconds,
    cases h_mconds with hm2 h_formula,
    rw h_formula, sorry, }  -- Completing this part is not necessary according to the guidelines
end

end greatest_three_digit_multiple_of_17_l620_620269


namespace log_limit_l620_620982

theorem log_limit (x : ℝ) (hx_pos : x > 0) :
  tendsto (λ x, log 4 (8 * x - 7) - log 4 (3 * x + 2)) at_top (𝓝 (log 4 (8 / 3))) :=
by
  sorry

end log_limit_l620_620982


namespace nine_digit_palindrome_count_l620_620229

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

def valid_digits : List ℕ := [6, 7, 8, 9]

def count_palindromes (digits : List ℕ) (length : ℕ) : ℕ :=
  (digits.length : ℕ)^((length + 1) / 2)

theorem nine_digit_palindrome_count : 
  count_palindromes valid_digits 9 = 1024 :=
by
  simp [count_palindromes, valid_digits]
  rfl

end nine_digit_palindrome_count_l620_620229


namespace value_of_f_at_5_l620_620632

def f (x : ℝ) : ℝ := 4 * x + 2

theorem value_of_f_at_5 : f 5 = 22 :=
by
  sorry

end value_of_f_at_5_l620_620632


namespace sum_log_divisors_12_pow_n_eq_1296_l620_620205

theorem sum_log_divisors_12_pow_n_eq_1296 (n : ℕ) :
  (∑ a in Finset.range (2 * n + 1), ∑ b in Finset.range (n + 1), Real.logb 12 (2^a * 3^b)) = 1296 →
  n = 8 :=
by
  sorry

end sum_log_divisors_12_pow_n_eq_1296_l620_620205


namespace remove_parentheses_l620_620157

variable (a b c : ℝ)

theorem remove_parentheses :
  -3 * a - (2 * b - c) = -3 * a - 2 * b + c :=
by
  sorry

end remove_parentheses_l620_620157


namespace greatest_three_digit_multiple_of_17_l620_620256

def is_three_digit (n : Int) : Prop := (n ≥ 100 ∧ n ≤ 999)
def is_multiple_of_17 (n : Int) : Prop := (∃ k : Int, n = 17 * k)

theorem greatest_three_digit_multiple_of_17 : ∀ n : Int, is_three_digit n ∧ is_multiple_of_17 n → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l620_620256


namespace ab_equals_six_l620_620043

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end ab_equals_six_l620_620043


namespace ab_value_l620_620004

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_value_l620_620004


namespace projection_of_2a_minus_b_in_direction_of_a_l620_620944

noncomputable def projection_in_direction (a b : EuclideanSpace ℝ 3) (angle_cos : ℝ) (a_mag b_mag : ℝ) : ℝ := 
  let dot_product := (2 * a - b) ⬝ a
  let projection_result := dot_product / a_mag
  projection_result

theorem projection_of_2a_minus_b_in_direction_of_a 
  (a b : EuclideanSpace ℝ 3)
  (h1 : ∥a∥ = 2)
  (h2 : ∥b∥ = 6)
  (h3 : inner a b = ∥a∥ * ∥b∥ * Real.cos (Real.pi / 3)) :
  projection_in_direction a b (Real.cos (Real.pi / 3)) 2 6 = 1 := 
by
  sorry

end projection_of_2a_minus_b_in_direction_of_a_l620_620944


namespace factorization_x8_minus_81_l620_620526

theorem factorization_x8_minus_81 (x : ℝ) : 
  x^8 - 81 = (x^4 + 9) * (x^2 + 3) * (x + real.sqrt 3) * (x - real.sqrt 3) :=
by
  sorry

end factorization_x8_minus_81_l620_620526


namespace angle_ADP_eq_PBA_l620_620110

-- Definitions for the problem
variables (A B C D P : Point)
variable (θAPB θDPC θADP θPBA : Angle)

def isParallelogram (A B C D : Point) : Prop := 
  -- Definition of parallelogram goes here
  sorry

def angleSum180 (θ1 θ2 : Angle) : Prop := 
  θ1 + θ2 = 180

def angleEqual (θ1 θ2 : Angle) : Prop :=
  θ1 = θ2

-- Theorem to prove
theorem angle_ADP_eq_PBA
  (h1 : isParallelogram A B C D)
  (h2 : angleSum180 θAPB θDPC)
  (h3 : θADP = θPBA) : angleEqual θADP θPBA :=
sorry

end angle_ADP_eq_PBA_l620_620110


namespace greatest_three_digit_multiple_of_17_l620_620365

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 100 ≤ n ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l620_620365


namespace probability_diagonals_intersect_l620_620996

theorem probability_diagonals_intersect (n : ℕ) (hn : n > 0) :
  let vertices := 2 * n + 1,
      total_diagonals := vertices.choose 2 - vertices,
      ways_to_select_two_diagonals := (total_diagonals.choose 2 : ℚ),
      intersecting_pairs := (vertices.choose 4 : ℚ)
  in (intersecting_pairs / ways_to_select_two_diagonals) = (n * (2 * n - 1) / (3 * (2 * n^2 - n - 2))) :=
by
  sorry

end probability_diagonals_intersect_l620_620996


namespace minimum_employees_needed_l620_620825

noncomputable def employees_needed (total_days : ℕ) (work_days : ℕ) (rest_days : ℕ) (min_on_duty : ℕ) : ℕ :=
  let comb := (total_days.choose rest_days)
  min_on_duty * comb / work_days

theorem minimum_employees_needed {total_days work_days rest_days min_on_duty : ℕ} (h_total_days: total_days = 7) (h_work_days: work_days = 5) (h_rest_days: rest_days = 2) (h_min_on_duty: min_on_duty = 45) : 
  employees_needed total_days work_days rest_days min_on_duty = 63 := by
  rw [h_total_days, h_work_days, h_rest_days, h_min_on_duty]
  -- detailed computation and proofs steps omitted
  -- the critical part is to ensure 63 is derived correctly based on provided values
  sorry

end minimum_employees_needed_l620_620825


namespace hockey_puck_price_l620_620856

theorem hockey_puck_price (P : ℝ) (hP : 0 < P) : 
  P > 0.99 * P :=
by
  linarith [mul_lt_mul_of_pos_left (by norm_num : 0.99 < 1) hP]

end hockey_puck_price_l620_620856


namespace no_linear_term_in_expansion_l620_620991

theorem no_linear_term_in_expansion (a : ℝ) :
  let expanded_expr := (3 * x + 2) * (3 * x + a) in
  ¬∃ x_term : ℝ, expanded_expr = 9 * x^2 + x_term * x + 2 * a ↔ a = -2 := 
sorry

end no_linear_term_in_expansion_l620_620991


namespace triangle_side_length_l620_620993

variable (A B C : Type) [inner_product_space ℝ A] 
variable (b c a : ℝ) 
variable (cosA : ℝ)

theorem triangle_side_length
  (hb : b = 3)
  (hc : c = 5)
  (hcosA : cosA = -1 / 2)
  (hcos_rule : a^2 = b^2 + c^2 - 2 * b * c * cosA) :
  a = 7 :=
by
  sorry

end triangle_side_length_l620_620993


namespace sum_c_sequence_m_range_l620_620574

noncomputable def a (n : ℕ) : ℝ := 1 / (4^n)

noncomputable def b (n : ℕ) : ℝ := 2 * n

noncomputable def c (n : ℕ) : ℝ := a n * b n

noncomputable def S (n : ℕ) : ℝ :=
  2 * (1 / 4) + 4 * (1 / 4^2) + ... + (2 * n - 2) * (1 / 4^(n-1)) + 2 * n * (1 / 4^n) -- omitting implementation for sum

theorem sum_c_sequence (n : ℕ) :
  S n = 8 / 9 - (6 * n + 8) / (9 * 4^n) := sorry

noncomputable def d (n : ℕ) : ℝ := (8 * n + 4) / ((n + 1)^2 * (b n)^2)

noncomputable def T (n : ℕ) : ℝ := 1 - 1 / ((n + 1)^2)

theorem m_range (m : ℝ) (h : ∀ n : ℕ, T n ≥ (1 / 2 * m^2 + m - 1)) :
  -2 - 3 * real.sqrt 2 ≤ 2 * m ∧ 2 * m ≤ -2 + 3 * real.sqrt 2 := sorry

end sum_c_sequence_m_range_l620_620574


namespace find_B1_l620_620598

noncomputable section

open Set

variables (A1 A2 A3 A4 A5 B1 B3 : Point) (O : Origin)
variables (pentagon : ConvexRegularPentagon A1 A2 A3 A4 A5)
variables (on_line : Collinear A1 B1 A3)

theorem find_B1 (h: B3.on (B1.line_through A2) ∧ B3.on (A3.line_through A5)) :
  (A1.distance B1 / B1.distance A3) = (A3.distance B3 / B3.distance A5) :=
sorry

end find_B1_l620_620598


namespace sam_total_distance_l620_620691

-- Definitions based on conditions
def first_half_distance : ℕ := 120
def first_half_time : ℕ := 3
def second_half_distance : ℕ := 80
def second_half_time : ℕ := 2
def sam_time : ℚ := 5.5

-- Marguerite's overall average speed
def marguerite_average_speed : ℚ := (first_half_distance + second_half_distance) / (first_half_time + second_half_time)

-- Theorem statement: Sam's total distance driven
theorem sam_total_distance : ∀ (d : ℚ), d = (marguerite_average_speed * sam_time) ↔ d = 220 := by
  intro d
  sorry

end sam_total_distance_l620_620691


namespace min_blue_eyes_tablets_l620_620079

theorem min_blue_eyes_tablets (total_students blue_eyed_students tablet_students : ℕ) 
  (h1 : total_students = 20)
  (h2 : blue_eyed_students = 8)
  (h3 : tablet_students = 15) : 
  ∃ min_students : ℕ, min_students = 3 :=
by {
  use (blue_eyed_students + tablet_students - total_students),
  rw [h1, h2, h3],
  norm_num,
  exact eq.refl 3,
}

end min_blue_eyes_tablets_l620_620079


namespace integers_abs_le_3_l620_620741

theorem integers_abs_le_3 :
  {x : ℤ | |x| ≤ 3} = { -3, -2, -1, 0, 1, 2, 3 } :=
by
  sorry

end integers_abs_le_3_l620_620741


namespace magician_card_pairs_l620_620446

theorem magician_card_pairs:
  ∃ (f : Fin 65 → Fin 65 × Fin 65), 
  (∀ m n : Fin 65, ∃ k l : Fin 65, (f m = (k, l) ∧ f n = (l, k))) := 
sorry

end magician_card_pairs_l620_620446


namespace more_roses_than_orchids_l620_620780

theorem more_roses_than_orchids (roses orchids : ℕ) (h1 : roses = 12) (h2 : orchids = 2) : roses - orchids = 10 := by
  sorry

end more_roses_than_orchids_l620_620780


namespace algebraic_expression_value_l620_620917

theorem algebraic_expression_value 
  (x y : ℝ) 
  (h : 2 * x + y = 1) : 
  (y + 1) ^ 2 - (y ^ 2 - 4 * x + 4) = -1 := 
by 
  sorry

end algebraic_expression_value_l620_620917


namespace today_is_thursday_l620_620846

-- Define the days of the week as an enumerated type
inductive DayOfWeek
| Monday : DayOfWeek
| Tuesday : DayOfWeek
| Wednesday : DayOfWeek
| Thursday : DayOfWeek
| Friday : DayOfWeek
| Saturday : DayOfWeek
| Sunday : DayOfWeek

open DayOfWeek

-- Define the conditions for the lion and the unicorn
def lion_truth (d: DayOfWeek) : Bool :=
match d with
| Monday | Tuesday | Wednesday => false
| _ => true

def unicorn_truth (d: DayOfWeek) : Bool :=
match d with
| Thursday | Friday | Saturday => false
| _ => true

-- The statement made by the lion and the unicorn
def lion_statement (today: DayOfWeek) : Bool :=
match today with
| Monday => lion_truth Sunday
| Tuesday => lion_truth Monday
| Wednesday => lion_truth Tuesday
| Thursday => lion_truth Wednesday
| Friday => lion_truth Thursday
| Saturday => lion_truth Friday
| Sunday => lion_truth Saturday

def unicorn_statement (today: DayOfWeek) : Bool :=
match today with
| Monday => unicorn_truth Sunday
| Tuesday => unicorn_truth Monday
| Wednesday => unicorn_truth Tuesday
| Thursday => unicorn_truth Wednesday
| Friday => unicorn_truth Thursday
| Saturday => unicorn_truth Friday
| Sunday => unicorn_truth Saturday

-- Main theorem to prove the current day
theorem today_is_thursday (d: DayOfWeek) (lion_said: lion_statement d = false) (unicorn_said: unicorn_statement d = false) : d = Thursday :=
by
  -- Placeholder for actual proof
  sorry

end today_is_thursday_l620_620846


namespace product_of_ab_l620_620056

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end product_of_ab_l620_620056


namespace remaining_stickers_l620_620768

def stickers_per_page : ℕ := 20
def pages : ℕ := 12
def lost_pages : ℕ := 1

theorem remaining_stickers : 
  (pages * stickers_per_page - lost_pages * stickers_per_page) = 220 :=
  by
    sorry

end remaining_stickers_l620_620768


namespace previous_average_daily_production_l620_620562

theorem previous_average_daily_production (n : ℕ) (P : ℝ) :
  n = 9 -> 
  let today_production := 100 in
  let new_average := 55 in
  let total_production_9_days := n * P in
  let new_total_production := total_production_9_days + today_production in
  let new_total_production_10_days := new_average * (n + 1) in
  new_total_production = new_total_production_10_days -> 
  P = 50 := 
by
  intros h1 h2
  sorry

end previous_average_daily_production_l620_620562


namespace part_a_part_b_l620_620419

-- Definitions for points A, B, their reflection B', and the line l
variables {A B B' X : Point} {l : Line}

-- Part (a)
-- Given A and B on the same side of the line l, and B' is the reflection of B across l,
-- prove that X is the intersection of AB' and l if it minimizes the sum of distances from X to A and B
theorem part_a (h_A_B_same_side: same_side A B l) (h_reflection: is_reflection B l B')
    (h_intersection: is_intersection X (line_through A B') l) :
  minimizes_sum_distances X A B := sorry

-- Part (b)
-- Given A and B on opposite sides of the line l, and B' is the reflection of B across l,
-- prove that X is the intersection of AB' and l if it maximizes the absolute value of the difference in distances
theorem part_b (h_A_B_opposite_sides: opposite_sides A B l) (h_reflection: is_reflection B l B')
    (h_intersection: is_intersection X (line_through A B') l) :
  maximizes_abs_diff_distances X A B := sorry

end part_a_part_b_l620_620419


namespace greatest_three_digit_multiple_of_17_l620_620370

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 100 ≤ n ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l620_620370


namespace count_b1_not_divisible_by_3_le_2023_l620_620115

theorem count_b1_not_divisible_by_3_le_2023 : 
  let count := (2023 - ((2023 / 3))) 
  in count = 1349 := 
by
  sorry

end count_b1_not_divisible_by_3_le_2023_l620_620115


namespace trapezoid_APBQ_l620_620118

open IncidenceGeometry

variables {A B C I P Q : Point}
variables [IncidenceGeometry]
variables {circumcircle_AIB : Circle}
variables {line_CA line_CB : Line}

def valid_incenter (A B C I : Point) : Prop := 
  is_incenter I A B C

def circumcircle_meets_line_again (circle : Circle) (line : Line) (P : Point) (A : Point) : Prop := 
  (P ≠ A) ∧ (P ∈ circle ∧ P ∈ line)

theorem trapezoid_APBQ
  (h1 : valid_incenter A B C I)
  (h2 : circumcircle_AIB = Circumcircle A I B)
  (h3 : line_CA = Line C A)
  (h4 : line_CB = Line C B)
  (hP : circumcircle_meets_line_again circumcircle_AIB line_CA P A)
  (hQ : circumcircle_meets_line_again circumcircle_AIB line_CB Q B) :
  ∃ (Q : Point), Q ∈ circumscribedQuad A B P Q ∧ parallel (Line A Q) (Line B P) := 
sorry

end trapezoid_APBQ_l620_620118


namespace ab_equals_six_l620_620046

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end ab_equals_six_l620_620046


namespace optimal_diagonals_l620_620717

noncomputable def is_optimal_diagonal (polygon : Type) [RegularPolygon polygon 400] (A_0 : polygon) (O : Point) : Prop :=
  ∀ k,
    k ∈ {54, 146, 254, 346} ↔ 
    let A_k := Vertex polygon k in
    let d := |distance A_0 A_k - 2 * distance O A_k| - distance A_0 O in
    ∀ m,
    |distance (Vertex polygon m) (Vertex polygon (200 - m)) - (distance (Vertex polygon m) (Vertex polygon (200 + m)))| ≥ d

theorem optimal_diagonals (polygon : Type) [RegularPolygon polygon 400] (A_0 : polygon) (O : Point) :
  is_optimal_diagonal polygon A_0 O :=
sorry

end optimal_diagonals_l620_620717


namespace greatest_three_digit_multiple_of_17_l620_620261

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (17 ∣ n) ∧ ∀ m : ℤ, (100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m) → m ≤ n :=
begin
  use 969,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num, exact dec_trivial, },
  { intros m h_mconds,
    cases h_mconds with hm1 h_mconds,
    cases h_mconds with hm2 h_formula,
    rw h_formula, sorry, }  -- Completing this part is not necessary according to the guidelines
end

end greatest_three_digit_multiple_of_17_l620_620261


namespace greatest_three_digit_multiple_of_17_l620_620298

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 17 ∣ n :=
by 
  sorry

end greatest_three_digit_multiple_of_17_l620_620298


namespace greatest_three_digit_multiple_of_17_l620_620356

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (n % 17 = 0) ∧ ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ (m % 17 = 0) → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620356


namespace rice_dumpling_costs_l620_620653

theorem rice_dumpling_costs :
  ∃ x, (1300 / (x + 0.6) = 1000 / x) ∧ x = 2 ∧ (x + 0.6 = 2.6) :=
begin
  sorry
end

end rice_dumpling_costs_l620_620653


namespace gnome_count_l620_620728

theorem gnome_count (g_R: ℕ) (g_W: ℕ) (h1: g_R = 4 * g_W) (h2: g_W = 20) : g_R - (40 * g_R / 100) = 48 := by
  sorry

end gnome_count_l620_620728


namespace calculate_area_of_UVWX_l620_620650

-- Define the rectangle PQRS with the given properties
structure Rectangle (P Q R S : ℝ × ℝ) :=
(is_rectangle : true)

-- Define trisect points on PQ and RS
structure TrisectPoints (L M N O : ℝ × ℝ) (P Q R S : ℝ × ℝ) :=
(trisect_PQ : true)
(trisect_RS : true)

-- Given conditions
def P : ℝ × ℝ := (0, 0)
def Q : ℝ × ℝ := (3, 0)
def R : ℝ × ℝ := (0, 3)
def S : ℝ × ℝ := (3, 3)
def L : ℝ × ℝ := (1, 0)
def M : ℝ × ℝ := (2, 0)
def N : ℝ × ℝ := (2, 3)
def O : ℝ × ℝ := (1, 3)

def pqrs : Rectangle P Q R S :=
{ is_rectangle := true }

def trisect : TrisectPoints L M N O P Q R S :=
{ trisect_PQ := true,
  trisect_RS := true }

-- Problem statement in Lean
theorem calculate_area_of_UVWX
  (P Q R S L M N O : ℝ × ℝ)
  (pqrs : Rectangle P Q R S)
  (trisect : TrisectPoints L M N O P Q R S)
  (h1 : P = (0, 0))
  (h2 : Q = (3, 0))
  (h3 : R = (0, 3))
  (h4 : S = (3, 3))
  (h5 : L = (1, 0))
  (h6 : M = (2, 0))
  (h7 : N = (2, 3))
  (h8 : O = (1, 3))
  (h9 : dist P R = 3)
  (h10 : dist P M = 3) :
  area_of_quadrilateral U V W X = 4.5 :=
sorry

end calculate_area_of_UVWX_l620_620650


namespace greatest_3_digit_multiple_of_17_l620_620384

theorem greatest_3_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≥ m) ∧ n = 986 := 
begin
  sorry
end

end greatest_3_digit_multiple_of_17_l620_620384


namespace zero_of_f_l620_620213

noncomputable def f (x : ℝ) : ℝ := Real.logb 5 (x - 1)

theorem zero_of_f :
  ∃ x : ℝ, f x = 0 ∧ x = 2 :=
by
  use 2
  unfold f
  sorry -- Skip the proof steps, as instructed.

end zero_of_f_l620_620213


namespace digits_satisfying_15n_divisible_by_n_l620_620560

theorem digits_satisfying_15n_divisible_by_n :
  {n : ℕ | n < 10 ∧ n ≠ 0 ∧ 15 * n % n = 0}.card = 5 := by
  sorry

end digits_satisfying_15n_divisible_by_n_l620_620560


namespace greatest_three_digit_multiple_of_17_l620_620358

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (n % 17 = 0) ∧ ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ (m % 17 = 0) → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620358


namespace number_of_sequences_l620_620622

/-- Prove that the number of distinct sequences of four letters
  that can be made from the letters in "EQUALS" is 8, given the following conditions:
  1. Each sequence must begin with 'L'.
  2. Each sequence must end with 'Q'.
  3. Each sequence must contain exactly one vowel (either 'E' or 'U', but not both).
  4. No letter can appear in a sequence more than once.
-/
theorem number_of_sequences : 
  let EQUALS := ['E', 'Q', 'U', 'A', 'L', 'S'] in
  let vowels := ['E', 'U'] in
  let consonants := ['Q', 'A', 'L', 'S'] in
  (∀ seq : List Char, seq.length = 4 ∧ seq.head = 'L' ∧ seq.last = 'Q' ∧ (∃ v ∈ vowels, v ∈ seq) ∧ (∀ c1 c2, c1 ∈ seq -> c2 ∈ seq -> c1 = c2 -> c1 = c2)) →
  seq.count = 8 :=
begin
  sorry
end

end number_of_sequences_l620_620622


namespace area_of_triangle_PQR_l620_620620

-- Conditions of the problem
def hex_pyramid_base_side : ℝ := 4
def hex_pyramid_altitude : ℝ := 8

def point_P_fraction : ℝ := 1 / 4
def point_Q_fraction : ℝ := 1 / 4
def point_R_fraction : ℝ := 1 / 2

-- GN, LN, JN are the same in a regular hexagonal base pyramid.
def gn_ln_jn : ℝ := Real.sqrt (hex_pyramid_base_side ^ 2 + hex_pyramid_altitude ^ 2)

-- Points P, Q, R as fractions along specified edges
def GP : ℝ := point_P_fraction * gn_ln_jn
def LQ : ℝ := point_Q_fraction * gn_ln_jn
def JR : ℝ := point_R_fraction * gn_ln_jn

-- Theorem to prove the area of \(\triangle PQR\) is \(4√3\) square centimeters.
theorem area_of_triangle_PQR :
  let area : ℝ := 4 * Real.sqrt 3 in
  (hex_pyramid_base_side = 4) →
  (hex_pyramid_altitude = 8) →
  (point_P_fraction = 1 / 4) →
  (point_Q_fraction = 1 / 4) →
  (point_R_fraction = 1 / 2) →
  ∃ (P Q R : Point), -- Assuming P, Q, R can be represented as points in space
    (area_of_triangle P Q R = 4 * Real.sqrt 3) :=
begin
  sorry
end

end area_of_triangle_PQR_l620_620620


namespace problem_l620_620116

noncomputable def seq (b : ℕ → ℝ) :=
  ∀ n ≥ 2, b n = b (n - 1) * b (n + 1)

theorem problem 
  (b : ℕ → ℝ)
  (initial_cond : b 1 = 2 + Real.sqrt 6)
  (target_cond : b 2023 = 10 + Real.sqrt 6)
  (rec_rel : seq b) :
  b 2050 = 2 + Real.sqrt 6 := 
begin
  sorry
end

end problem_l620_620116


namespace distance_on_fifth_day_l620_620652

-- Define the initial conditions
def first_day_distance : ℕ := 100
def total_distance_nine_days : ℕ := 1260
def number_of_days : ℕ := 9

-- Define the proof goal that the distance on the fifth day is 140 li
theorem distance_on_fifth_day : 
  ∃ d : ℕ, (∑ i in finset.range number_of_days, first_day_distance + i * d = total_distance_nine_days) 
           → (first_day_distance + 4 * d = 140) :=
sorry

end distance_on_fifth_day_l620_620652


namespace gnomes_remaining_in_ravenswood_l620_620730

theorem gnomes_remaining_in_ravenswood 
  (westerville_gnomes : ℕ)
  (ravenswood_initial_gnomes : ℕ)
  (taken_gnomes : ℕ)
  (remaining_gnomes : ℕ)
  (h1 : westerville_gnomes = 20)
  (h2 : ravenswood_initial_gnomes = 4 * westerville_gnomes)
  (h3 : taken_gnomes = (40 * ravenswood_initial_gnomes) / 100)
  (h4 : remaining_gnomes = ravenswood_initial_gnomes - taken_gnomes) :
  remaining_gnomes = 48 :=
by
  sorry

end gnomes_remaining_in_ravenswood_l620_620730


namespace bisector_intersects_l620_620658

structure Trapezoid (A B C D : Type) :=
  (AB : ℝ)
  (BC : ℝ)
  (AB_parallel_CD : Prop)

theorem bisector_intersects {A B C D : Type}
  (trap : Trapezoid A B C D)
  (a b : ℝ)
  (h_AB : trap.AB = a)
  (h_BC : trap.BC = b) :
  (a > b → "intersects CD") ∧ (a < b → "intersects BC") :=
by {
  sorry
}

end bisector_intersects_l620_620658


namespace fewest_cookies_l620_620475

theorem fewest_cookies
  (r a s d1 d2 : ℝ)
  (hr_pos : r > 0)
  (ha_pos : a > 0)
  (hs_pos : s > 0)
  (hd1_pos : d1 > 0)
  (hd2_pos : d2 > 0)
  (h_Alice_cookies : 15 = 15)
  (h_same_dough : true) :
  15 < (15 * (Real.pi * r^2)) / (a^2) ∧
  15 < (15 * (Real.pi * r^2)) / ((3 * Real.sqrt 3 / 2) * s^2) ∧
  15 < (15 * (Real.pi * r^2)) / ((1 / 2) * d1 * d2) :=
by
  sorry

end fewest_cookies_l620_620475


namespace greatest_three_digit_multiple_of_17_l620_620286

def is_multiple_of (n : ℕ) (k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, is_three_digit n ∧ is_multiple_of n 17 → n = 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620286


namespace dealership_sales_prediction_l620_620854

theorem dealership_sales_prediction (sports_cars_sold sedans SUVs : ℕ) 
    (ratio_sc_sedans : 3 * sedans = 5 * sports_cars_sold) 
    (ratio_sc_SUVs : sports_cars_sold = 2 * SUVs) 
    (sports_cars_sold_next_month : sports_cars_sold = 36) :
    (sedans = 60 ∧ SUVs = 72) :=
sorry

end dealership_sales_prediction_l620_620854


namespace total_elephants_in_two_parks_is_280_l620_620193

def number_of_elephants_we_preserve_for_future : ℕ := 70
def multiple_factor : ℕ := 3

def number_of_elephants_gestures_for_good : ℕ := multiple_factor * number_of_elephants_we_preserve_for_future

def total_number_of_elephants : ℕ := number_of_elephants_we_preserve_for_future + number_of_elephants_gestures_for_good

theorem total_elephants_in_two_parks_is_280 : total_number_of_elephants = 280 :=
by
  sorry

end total_elephants_in_two_parks_is_280_l620_620193


namespace greatest_three_digit_multiple_of_17_l620_620357

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (n % 17 = 0) ∧ ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ (m % 17 = 0) → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620357


namespace general_term_is_correct_sum_of_b_n_l620_620581

noncomputable def a_n (n : ℕ) : ℕ := 2 * n + 1

theorem general_term_is_correct :
  (∀ n : ℕ, a_n n = 2 * n + 1) :=
sorry

def b_n (n : ℕ) : ℚ := 1 / (a_n n ^ 2 - 1)

theorem sum_of_b_n (n : ℕ) :
  ∑ k in Finset.range n, b_n (k + 1) = (n : ℚ) / (4 * (n + 1)) :=
sorry

end general_term_is_correct_sum_of_b_n_l620_620581


namespace expected_turns_formula_l620_620828

noncomputable def expected_turns (n : ℕ) : ℝ :=
  n +  1 / 2 - (n - 1 / 2) * (1 / Real.sqrt (Real.pi * (n - 1)))

theorem expected_turns_formula (n : ℕ) (h : n ≥ 1) :
  expected_turns n = n + 1 / 2 - (n - 1 / 2) * (1 / Real.sqrt (Real.pi * (n - 1))) :=
by
  sorry

end expected_turns_formula_l620_620828


namespace greatest_three_digit_multiple_of_17_l620_620254

def is_three_digit (n : Int) : Prop := (n ≥ 100 ∧ n ≤ 999)
def is_multiple_of_17 (n : Int) : Prop := (∃ k : Int, n = 17 * k)

theorem greatest_three_digit_multiple_of_17 : ∀ n : Int, is_three_digit n ∧ is_multiple_of_17 n → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l620_620254


namespace greatest_three_digit_multiple_of_17_l620_620292

def is_multiple_of (n : ℕ) (k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, is_three_digit n ∧ is_multiple_of n 17 → n = 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620292


namespace problem1_problem2_problem3_problem4_l620_620722

section Problem1

variable (x y : ℝ)

theorem problem1 : (x - 2 * y) ^ 2 - x * (x + 3 * y) - 4 * y ^ 2 = -7 * x * y :=
by
  sorry

end Problem1

section Problem2

variable (x y m : ℝ)

theorem problem2 : 3 * x ^ 2 - 6 * x * y + 3 * y ^ 2 - 27 * m ^ 2 = 3 * (x - y + 3 * m) * (x - y - 3 * m) :=
by
  sorry

end Problem2

section Problem3

theorem problem3 : (3 * √12 - 2 * (√(1 / 3)) + √48) / (2 * √3) = 14 / 3 :=
by
  sorry

end Problem3

section Problem4

variable (x : ℝ)

theorem problem4 (h : |x - 3| + |3 * x + 2| < 15) : -7 / 2 < x ∧ x < 4 :=
by
  sorry

end Problem4

end problem1_problem2_problem3_problem4_l620_620722


namespace systematic_sampling_correct_l620_620433

-- Conditions as definitions
def total_bags : ℕ := 50
def num_samples : ℕ := 5
def interval (total num : ℕ) : ℕ := total / num
def correct_sequence : List ℕ := [5, 15, 25, 35, 45]

-- Statement
theorem systematic_sampling_correct :
  ∃ l : List ℕ, (l.length = num_samples) ∧ 
               (∀ i ∈ l, i ≤ total_bags) ∧
               (∀ i j, i < j → l.indexOf i < l.indexOf j → j - i = interval total_bags num_samples) ∧
               l = correct_sequence :=
by
  sorry

end systematic_sampling_correct_l620_620433


namespace problem_1_problem_2_l620_620611

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + 3 * x

-- Problem I
theorem problem_1 (x : ℝ) : (f x 1 ≥ 3 * x + 2) ↔ (x ≥ 3 ∨ x ≤ -1) :=
by sorry

-- Problem II
theorem problem_2 (a : ℝ) (h : ∀ x : ℝ, f x a ≤ 0 → x ≤ -3) : a = 6 :=
by sorry

end problem_1_problem_2_l620_620611


namespace fixed_points_on_circle_l620_620139

theorem fixed_points_on_circle (m : ℝ) : 
  (√5, 2 * √5) ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 + 2 * m * p.1 - m * p.2 - 25 = 0} ∧ 
  (-√5, -2 * √5) ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 + 2 * m * p.1 - m * p.2 - 25 = 0} :=
sorry

end fixed_points_on_circle_l620_620139


namespace ab_equals_six_l620_620040

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end ab_equals_six_l620_620040


namespace sum_digits_from_10e2003_to_10e2004_minus_1_l620_620675

def sum_of_digits (n : ℕ) : ℕ :=
  (n.to_string.data.map (λ c, c.to_nat - '0'.to_nat)).sum

theorem sum_digits_from_10e2003_to_10e2004_minus_1 :
  sum_of_digits (Icc.sum 10^{2003} (λ k, sum_of_digits k) 10^{2004}) = 24 :=
sorry

end sum_digits_from_10e2003_to_10e2004_minus_1_l620_620675


namespace initial_guppies_l620_620464

theorem initial_guppies (total_gups : ℕ) (dozen_gups : ℕ) (extra_gups : ℕ) (baby_gups_initial : ℕ) (baby_gups_later : ℕ) :
  total_gups = 52 → dozen_gups = 12 → extra_gups = 3 → baby_gups_initial = 3 * 12 → baby_gups_later = 9 → 
  total_gups - (baby_gups_initial + baby_gups_later) = 7 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end initial_guppies_l620_620464


namespace freqDistTablePurpose_l620_620877

-- Definition of conditions
def freqDistTable : Type := 
  { size : Nat }

def understandSize (table : freqDistTable) : Prop :=
  true

def estimateOverallSituation (table : freqDistTable) : Prop :=
  true

-- Theorem to prove the purpose of creating frequency distribution table
theorem freqDistTablePurpose (table : freqDistTable) :
  understandSize table → estimateOverallSituation table → (true) :=
begin
  sorry
end

end freqDistTablePurpose_l620_620877


namespace sqrt_factorial_div_l620_620480

theorem sqrt_factorial_div:
  Real.sqrt (↑(Nat.factorial 9) / 90) = 4 * Real.sqrt 42 := 
by
  -- Steps of the proof
  sorry

end sqrt_factorial_div_l620_620480


namespace find_angle_A_find_area_triangle_l620_620095

variable (a b c : ℝ)
variable (A B : ℝ) -- angles in radians

-- Condition from the problem
axiom condition1 : sqrt 3 * b * cos A = a * sin B
-- Additional given conditions
axiom given1 : a = sqrt 7
axiom given2 : b = 2
axiom given3 : A = π / 3

-- Define area of the triangle
noncomputable def area (a b c A : ℝ) : ℝ := (1/2) * b * c * sin A

-- Tasks to prove
theorem find_angle_A : A = π / 3 := by
  sorry

theorem find_area_triangle : area a b c A = (3 * sqrt 3) / 2 := by
  sorry

end find_angle_A_find_area_triangle_l620_620095


namespace purely_imaginary_z_eq_neg2_l620_620986

theorem purely_imaginary_z_eq_neg2 (a : ℝ) :
  (a^2 + a - 2 = 0) ∧ (a^2 - 3a + 2 ≠ 0) → a = -2 :=
by
  sorry

end purely_imaginary_z_eq_neg2_l620_620986


namespace sufficient_food_l620_620489

-- Definitions for a large package and a small package as amounts of food lasting days
def largePackageDay : ℕ := L
def smallPackageDay : ℕ := S

-- Condition: One large package and four small packages last 14 days
axiom condition : largePackageDay + 4 * smallPackageDay = 14

-- Theorem to be proved: One large package and three small packages last at least 11 days
theorem sufficient_food : largePackageDay + 3 * smallPackageDay ≥ 11 :=
by
  sorry

end sufficient_food_l620_620489


namespace greatest_three_digit_multiple_of_17_l620_620360

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (n % 17 = 0) ∧ ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ (m % 17 = 0) → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620360


namespace modulus_of_z_l620_620572

open Complex

noncomputable def z : ℂ := (1 - sqrt 3 * I) / (1 + I)

theorem modulus_of_z :
  Complex.abs z = sqrt 2 :=
sorry

end modulus_of_z_l620_620572


namespace inequality_solution_l620_620901

theorem inequality_solution (x : ℝ) : x ∈ [3, 4) ∪ (4, 64/11] ↔ (x ≠ 4) ∧ (x * (x + 1)) / (x - 4)^2 ≥ 12 :=
by {
  sorry -- Proof not required, only statement needed
}

end inequality_solution_l620_620901


namespace complement_intersection_l620_620689

open Set

variable U : Set ℝ := univ

def A : Set ℝ := {x | x ≥ 1}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5}

theorem complement_intersection (x : ℝ) : x ∈ (U \ A) ∩ B ↔ 0 ≤ x ∧ x < 1 := by
  rw [mem_inter_iff, mem_compl_iff, mem_set_of_eq, mem_set_of_eq]
  exact sorry

end complement_intersection_l620_620689


namespace greatest_three_digit_multiple_of_17_l620_620340

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620340


namespace point_position_after_time_l620_620453

noncomputable def final_position (initial : ℝ × ℝ) (velocity : ℝ × ℝ) (time : ℝ) : ℝ × ℝ :=
  (initial.1 + velocity.1 * time, initial.2 + velocity.2 * time)

theorem point_position_after_time :
  final_position (-10, 10) (4, -3) 5 = (10, -5) :=
by
  sorry

end point_position_after_time_l620_620453


namespace distance_between_house_and_school_l620_620474

/-- Xiaoming leaves his house at 7:00 AM, walks to school at a speed of 52 meters per minute,
and upon arrival, the hour and minute hands on his watch are positioned symmetrically around the number 7.
Prove that the distance between Xiaoming’s house and the school is 1680 meters, given that
the entire walk took less than an hour. -/
theorem distance_between_house_and_school {time_taken : ℕ} (h₁ : time_taken < 60)
  (h₂ : ∀ t, (t = time_taken) → (210 = (6.5 * t))) : 52 * time_taken = 1680 :=
sorry

end distance_between_house_and_school_l620_620474


namespace process_ends_with_2_balls_in_any_urn_process_ends_with_10_balls_in_urns_l620_620420

theorem process_ends_with_2_balls_in_any_urn :
  let urns := 3,
      probability := 1 / 3
  in 
    probability = 1 / 3 := by
  sorry

theorem process_ends_with_10_balls_in_urns :
  let urns := 2014,
      prob_10_balls := (2013 / 2014) * (2012 / 2014) * (2011 / 2014) * (2010 / 2014) * (2009 / 2014) *
                       (2008 / 2014) * (2007 / 2014) * (2006 / 2014) * (9 / 2014)
  in 
    prob_10_balls = (2013 / 2014) * (2012 / 2014) * (2011 / 2014) * (2010 / 2014) * (2009 / 2014) *
                    (2008 / 2014) * (2007 / 2014) * (2006 / 2014) * (9 / 2014) := by
  sorry

end process_ends_with_2_balls_in_any_urn_process_ends_with_10_balls_in_urns_l620_620420


namespace jim_caught_fish_l620_620859

variable (ben judy billy susie jim caught_back total_filets : ℕ)

def caught_fish : ℕ :=
  ben + judy + billy + susie + jim - caught_back

theorem jim_caught_fish (h_ben : ben = 4)
                        (h_judy : judy = 1)
                        (h_billy : billy = 3)
                        (h_susie : susie = 5)
                        (h_caught_back : caught_back = 3)
                        (h_total_filets : total_filets = 24)
                        (h_filets_per_fish : ∀ f : ℕ, total_filets = f * 2 → caught_fish ben judy billy susie jim caught_back = f) :
  jim = 2 :=
by
  -- Proof goes here
  sorry

end jim_caught_fish_l620_620859


namespace AB_length_is_9_l620_620472

noncomputable def equilateral_triangle_side_length : ℝ := 24

noncomputable def number_of_equal_areas : ℕ := 5

def find_AB (side_length : ℝ) (num_areas : ℕ) : ℝ :=
  if side_length = 24 ∧ num_areas = 5 then 9 else 0

theorem AB_length_is_9 :
  find_AB equilateral_triangle_side_length number_of_equal_areas = 9 := by
  sorry

end AB_length_is_9_l620_620472


namespace circumcircle_passes_fixed_point_l620_620847

theorem circumcircle_passes_fixed_point
  (a b : ℝ)
  (h_intersects_axes : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 ≠ 0 ∧ x2 ≠ 0 ∧ (0, b) ≠ (x1, 0) ∧ (0, b) ≠ (x2, 0)) :
  ∃ D : ℝ × ℝ, D = (0, 1) ∧ ∀ (x1 x2 : ℝ), x1 ≠ x2 → (x1, 0), (x2, 0), (0, b) ∈ ({p : ℝ × ℝ | p.2 = p.1 ^ 2 + a * p.1 + b}) → D ∈ ({p : ℝ × ℝ | ∃ r, (r ≠ 0) ∧ distance p (x1, 0) = distance p (x2, 0) ∧ distance p (0, b) = distance p (x1, 0)})
  sorry

end circumcircle_passes_fixed_point_l620_620847


namespace jonah_profit_is_correct_l620_620672

noncomputable def jonah_profit : Real :=
  let pineapples := 6
  let pricePerPineapple := 3
  let pineappleCostWithoutDiscount := pineapples * pricePerPineapple
  let discount := if pineapples > 4 then 0.20 * pineappleCostWithoutDiscount else 0
  let totalCostAfterDiscount := pineappleCostWithoutDiscount - discount
  let ringsPerPineapple := 10
  let totalRings := pineapples * ringsPerPineapple
  let ringsSoldIndividually := 2
  let pricePerIndividualRing := 5
  let revenueFromIndividualRings := ringsSoldIndividually * pricePerIndividualRing
  let ringsLeft := totalRings - ringsSoldIndividually
  let ringsPerSet := 4
  let setsSold := ringsLeft / ringsPerSet -- This should be interpreted as an integer division
  let pricePerSet := 16
  let revenueFromSets := setsSold * pricePerSet
  let totalRevenue := revenueFromIndividualRings + revenueFromSets
  let profit := totalRevenue - totalCostAfterDiscount
  profit
  
theorem jonah_profit_is_correct :
  jonah_profit = 219.60 := by
  sorry

end jonah_profit_is_correct_l620_620672


namespace greatest_three_digit_multiple_of_17_l620_620368

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 100 ≤ n ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l620_620368


namespace sum_series_eq_half_l620_620874

theorem sum_series_eq_half :
  (∑' (n : ℕ) in finset.Ici 2, (3 * (n : ℝ)^3 + 2 * (n : ℝ)^2 - n + 1) / (n^6 + n^5 - n^4 + n^3 - n^2 + n)) = 1/2 := 
sorry

end sum_series_eq_half_l620_620874


namespace harmonic_progression_l620_620782

theorem harmonic_progression (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
(h_harm : 1 / (a : ℝ) + 1 / (c : ℝ) = 2 / (b : ℝ))
(h_div : c % b = 0)
(h_inc : a < b ∧ b < c) :
  a = 20 → 
  (b, c) = (30, 60) ∨ (b, c) = (35, 140) ∨ (b, c) = (36, 180) ∨ (b, c) = (38, 380) ∨ (b, c) = (39, 780) :=
by sorry

end harmonic_progression_l620_620782


namespace customer_difference_l620_620461

theorem customer_difference (before after : ℕ) (h1 : before = 19) (h2 : after = 4) : before - after = 15 :=
by
  sorry

end customer_difference_l620_620461


namespace greatest_3_digit_multiple_of_17_l620_620377

theorem greatest_3_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≥ m) ∧ n = 986 := 
begin
  sorry
end

end greatest_3_digit_multiple_of_17_l620_620377


namespace greatest_three_digit_multiple_of_17_l620_620304

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 17 ∣ n :=
by 
  sorry

end greatest_three_digit_multiple_of_17_l620_620304


namespace sufficient_food_supply_l620_620505

variable {L S : ℝ}

theorem sufficient_food_supply (h1 : L + 4 * S = 14) (h2 : L > S) : L + 3 * S ≥ 11 :=
by
  sorry

end sufficient_food_supply_l620_620505


namespace weight_loss_l620_620668

def initial_weight : ℕ := 69
def current_weight : ℕ := 34

theorem weight_loss :
  initial_weight - current_weight = 35 :=
by
  sorry

end weight_loss_l620_620668


namespace ab_equals_six_l620_620015

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_l620_620015


namespace find_x_plus_y_plus_z_l620_620198

noncomputable def calculate_distance_S_to_triangle_PQR (x y z : ℕ) (P Q R S : ℝ × ℝ × ℝ) 
  (dist_PQ dist_QR dist_RP radius_S : ℝ) : Prop :=
radius_S = 17 ∧ dist_PQ = 12 ∧ dist_QR = 16 ∧ dist_RP = 20 ∧
x = 7 ∧ y = 391 ∧ z = 9 ∧
euclidean_distance S (foot_perpendicular S P Q R) = (x * sqrt y) / z

theorem find_x_plus_y_plus_z : ∃ (x y z : ℕ), calculate_distance_S_to_triangle_PQR x y z P Q R S 17 ∧
  x + y + z = 407 := 
sorry

end find_x_plus_y_plus_z_l620_620198


namespace problem_a_g_pi_l620_620736

-- Define the function f
def f (x : ℝ) : ℝ := sin (2 * x - π / 3)

-- Define the translated function h
def h (x : ℝ) : ℝ := sin (2 * x + π / 3)

-- Define the stretched function g resulting from h
def g (x : ℝ) : ℝ := sin (x + π / 3)

-- State the theorem to prove g(π) = -√3 / 2
theorem problem_a_g_pi : g π = - (sqrt 3) / 2 := 
by 
  sorry

end problem_a_g_pi_l620_620736


namespace square_isosceles_l620_620145

theorem square_isosceles (A B C D E K O S : Point) 
  (h_square : is_square A B C D) 
  (h_iso : is_isosceles_triangle A B E) 
  (h_AE_BE : dist A E = dist B E) 
  (h_K_ced_angles : bisects_angle E O K S) 
  (h_symmetry : K = S) : SK = KO :=
by sorry

end square_isosceles_l620_620145


namespace fraction_of_men_l620_620470

open Real

theorem fraction_of_men (M W : ℕ) (h1 : M + W = 90)
                          (h2 : W + 10 = 0.40 * 100) : 
                          M / (M + W) = 2 / 3 := 
by
  have hW : W = 30 := by
    rw [←Nat.cast_inj]
    exact Eq.symm (nat_cast_inj.2 (eq_of_nat_eq_nat h2).trans (nat_cast_inj.1 (W + 10 = 40)))
  have hM : M = 60 := by
    rw [nat_add_eq_add_nat]
    exact nat_cast_inj.1 (eq_of_nat_eq_nat h1).trans (nat_cast_inj.1 rfl)
  rw [←hM, ←hW]
  norm_num
  sorry

end fraction_of_men_l620_620470


namespace greatest_three_digit_multiple_of_17_l620_620371

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 100 ≤ n ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l620_620371


namespace greatest_three_digit_multiple_of_17_l620_620394

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) := 
  exists.intro 986 sorry

end greatest_three_digit_multiple_of_17_l620_620394


namespace maximum_water_overflow_volume_l620_620217

-- Define the base side length of the equilateral triangle.
def s := 6

-- Define the conditions for the rectangular prism values.
variables (a : ℝ) (h : ℝ)
hypothesis h_a : a < s
hypothesis h_h : h > s

-- Define the height of the equilateral triangle.
def H := (sqrt 3 / 2) * s

-- Define the relationship for height h in terms of a.
def h_relation (a : ℝ) := H - (sqrt 6 / 2) * a

-- Define the volume of water displaced.
def V (a : ℝ) := a^2 * (H - (sqrt 6 / 2) * a)

-- State the theorem to be proven.
theorem maximum_water_overflow_volume : a = 2 * sqrt 2 → 
  ∀ h (h_relation a),
  V a = 
    let max_V := (2 * sqrt 6) * (sqrt 2)^3 in 
    V a = max_V := sorry

end maximum_water_overflow_volume_l620_620217


namespace greatest_three_digit_multiple_of_17_l620_620392

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) := 
  exists.intro 986 sorry

end greatest_three_digit_multiple_of_17_l620_620392


namespace range_of_a_l620_620951

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (x1 x2 : ℝ)
  (h1 : 0 < x1) (h2 : x1 < x2) (h3 : x2 ≤ 1) (f_def : ∀ x, f x = a * x - x^3)
  (condition : f x2 - f x1 > x2 - x1) :
  a ≥ 4 :=
by sorry

end range_of_a_l620_620951


namespace problem_l620_620604

noncomputable def f (x : ℝ) (a : ℝ) (α : ℝ) (b : ℝ) (β : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)

theorem problem {a α b β : ℝ} (h : f 2001 a α b β = 3) : f 2012 a α b β = -3 := by
  sorry

end problem_l620_620604


namespace positive_integer_solutions_eqn_l620_620552

theorem positive_integer_solutions_eqn (n : ℕ) :
  (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ (x^2 + y^2)^n = (x * y)^2016) ↔
    n ∈ {1344, 1728, 1792, 1920, 1984} :=
by
  sorry

end positive_integer_solutions_eqn_l620_620552


namespace largest_three_digit_multiple_of_17_l620_620273

theorem largest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n ≤ 999) ∧ (n ≥ 100) ∧ (17 ∣ n) ∧ ∀ m, ((m > n) ∧ (m ≤ 999) ∧ (m ≥ 100) ∧ (17 ∣ m)) → false :=
by {
  sorry
}

end largest_three_digit_multiple_of_17_l620_620273


namespace vector_expression_equivalence_l620_620602

variables (a b : Type) [AddCommGroup a] [VectorSpace ℝ a]
variables (u v : a)

theorem vector_expression_equivalence (u v : a) : 
  (1/3 : ℝ) • ((1/2 : ℝ) • (2 • u + 8 • v) - (4 • u - 2 • v)) = 2 • v - u :=
sorry

end vector_expression_equivalence_l620_620602


namespace fixed_point_other_than_A_l620_620934

-- Definitions based on conditions
variables (A B C : Point)
variables (B1 : Line A B) (C1 : Line A C) (BC : Line B C)
variables (ω : Circle)

-- Conditions
def not_right_angle (A B C : Point) : Prop :=
  ∠BAC ≠ 90°

def spiral_similarity_exists (A B1 C1 : Point) : Prop :=
  ∃ B2 C2 : Point, on_line B2 BC ∧ on_line C2 BC ∧ (spiral_similarity A B1 C1 B2 C2)

def points_concur_on_circumcircle (D : Point) : Prop :=
  ∃ X Y : Point, on_circumcircle D ω ∧ on_line B1 B2 X ∧ on_line C1 C2 Y ∧ (X = Y ∧ X ≠ B1 ∧ X ≠ C1)

-- Theorem to be proved
theorem fixed_point_other_than_A (A B C B1 C1 B2 C2 D : Point) (ω : Circle)
  [not_right_angle A B C]
  [spiral_similarity_exists A B1 C1]
  [points_concur_on_circumcircle D] :
  ∃ P : Point, on_circle P ω ∧ P ≠ A :=
begin
  sorry
end

end fixed_point_other_than_A_l620_620934


namespace angle_bisector_slope_l620_620172

theorem angle_bisector_slope 
  (m1 m2 : ℝ)
  (h1 : m1 = 2) 
  (h2 : m2 = 4) : 
  let k := (m1 + m2 - real.sqrt (1 + m1 ^ 2 + m2 ^ 2)) / (1 - m1 * m2) in
  k = (real.sqrt 21 - 6) / 7 :=
begin
  sorry
end

end angle_bisector_slope_l620_620172


namespace arithmetic_sequence_condition_sequence_terms_l620_620589

-- Problem I
def periodic_sequence (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 2
  | 1 => 1
  | 2 => 4
  | 3 => 3
  | _ => 0 -- this case will never happen due to the modulo operation

def A (n : ℕ) : ℕ :=
  let seq := List.map periodic_sequence (List.range (n+1))
  List.maximum' seq

def B (n : ℕ) : ℕ :=
  let seq := List.map periodic_sequence (List.filter (λ x => x > n) (List.range 100))
  List.minimum' seq  -- use 100 as an arbitrary large number to simulate "infinite"

def d (n : ℕ) : ℕ :=
  A n - B n

example : d 1 = 1 ∧ d 2 = 1 ∧ d 3 = 3 ∧ d 4 = 3 := by
  unfold d A B periodic_sequence
  have h1 : (List.range (1+1)).map periodic_sequence = [2] := by simp [List.range, periodic_sequence]
  have h2 : (List.range (2+1)).map periodic_sequence = [2, 1] := by simp [List.range, periodic_sequence]
  have h3 : (List.range (3+1)).map periodic_sequence = [2, 1, 4] := by simp [List.range, periodic_sequence]
  have h4 : (List.range (4+1)).map periodic_sequence = [2, 1, 4, 3] := by simp [List.range, periodic_sequence]
  simp [A, B, d] at *
  simp [h1, h2, h3, h4]
  sorry  -- complete the detailed proof

-- Problem II
theorem arithmetic_sequence_condition (d : ℕ) (a : ℕ → ℕ) :
  (∀ n, d n = (-d : ℤ)) ↔ ∃ a1, ∀ n, a (n + 1) = a n + d := sorry

-- Problem III
theorem sequence_terms (a : ℕ → ℕ) :
  a 1 = 2 → (∀ n, d n = 1) → (∀ n, a n = 1 ∨ a n = 2) ∧ ∃ᶠ n in at_top, a n = 1 := sorry

end arithmetic_sequence_condition_sequence_terms_l620_620589


namespace derivative_at_pi_l620_620925

noncomputable def f (x : ℝ) : ℝ := x * sin x

theorem derivative_at_pi : deriv f π = -π :=
by
  -- Proof is omitted
  sorry

end derivative_at_pi_l620_620925


namespace c_seven_eq_3543_l620_620686

-- Define the sequence c_n
def c : ℕ → ℤ
| 1     := 3
| 2     := 9
| (n+1) := 3 * c n + c (n - 1)

-- State the theorem to prove c_7 = 3543
theorem c_seven_eq_3543 : c 7 = 3543 := sorry

end c_seven_eq_3543_l620_620686


namespace ab_eq_six_l620_620059

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_eq_six_l620_620059


namespace problem_solution_l620_620742

-- Define the function f
def f (x : ℝ) : ℝ := (2 * x + 3) / (2 * x - 4)

-- Define points O and P
def O : ℝ × ℝ := (0, 0)
def P : ℝ × ℝ := (2, 1)

-- Define the vectors OP, OA, and OB
def OP : ℝ × ℝ := (P.1 - O.1, P.2 - O.2)
def dot_product : (ℝ × ℝ) → (ℝ × ℝ) → ℝ := λ v1 v2, v1.1 * v2.1 + v1.2 * v2.2

theorem problem_solution :
  ∀ A B : ℝ × ℝ, (line_through P A) ∧ (line_through P B) ∧ (f A.1 = A.2) ∧ (f B.1 = B.2) →
  dot_product OP ((A.1 - O.1, A.2 - O.2) + (B.1 - O.1, B.2 - O.2)) = 10 :=
by
  sorry

end problem_solution_l620_620742


namespace giyun_walk_distance_l620_620969

theorem giyun_walk_distance (step_length : ℝ) (steps_per_minute : ℕ) (minutes : ℕ) 
  (h1 : step_length = 0.75) 
  (h2 : steps_per_minute = 70) 
  (h3 : minutes = 13) : 
  distance (house : location) (school : location) = 682.5 :=
by
  sorry

end giyun_walk_distance_l620_620969


namespace greatest_three_digit_multiple_of_17_l620_620263

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (17 ∣ n) ∧ ∀ m : ℤ, (100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m) → m ≤ n :=
begin
  use 969,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num, exact dec_trivial, },
  { intros m h_mconds,
    cases h_mconds with hm1 h_mconds,
    cases h_mconds with hm2 h_formula,
    rw h_formula, sorry, }  -- Completing this part is not necessary according to the guidelines
end

end greatest_three_digit_multiple_of_17_l620_620263


namespace greatest_three_digit_multiple_of_17_l620_620400

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) := 
  exists.intro 986 sorry

end greatest_three_digit_multiple_of_17_l620_620400


namespace smallest_integer_l620_620075

theorem smallest_integer (k : ℕ) (n : ℕ) (h936 : 936 = 2^3 * 3^1 * 13^1)
  (h2 : 2^5 ∣ 936 * k)
  (h3 : 3^3 ∣ 936 * k)
  (h4 : 12^2 ∣ 936 * k) : 
  k = 36 :=
by {
  sorry
}

end smallest_integer_l620_620075


namespace sqrt_x_minus_4_condition_l620_620220

theorem sqrt_x_minus_4_condition (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 4)) ↔ x ≥ 4 := by
  sorry

end sqrt_x_minus_4_condition_l620_620220


namespace cat_food_sufficiency_l620_620506

theorem cat_food_sufficiency (L S : ℝ) (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end cat_food_sufficiency_l620_620506


namespace cat_food_sufficiency_l620_620517

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end cat_food_sufficiency_l620_620517


namespace average_of_integers_between_l620_620231

theorem average_of_integers_between (N : ℤ) (h1 : 18 < N) (h2 : N < 28) :
  let L : List ℤ := [19, 20, 21, 22, 23, 24, 25, 26, 27] in
  (L.sum / L.length : ℚ) = 23 := 
by
  sorry

end average_of_integers_between_l620_620231


namespace greatest_three_digit_multiple_of_17_l620_620317

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → m ≤ n) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620317


namespace time_to_complete_smaller_field_20_minutes_l620_620452

noncomputable def time_to_complete_smaller_field 
  (w l l_large w_large : ℝ) (time_large : ℝ)
  (h1 : l = 1.5 * w) 
  (h2 : l_large = 3 * l) 
  (h3 : w_large = 4 * w) 
  (h4 : time_large = 68) : ℝ :=
let P_small := 2 * (l + w) in
let P_large := 2 * (l_large + w_large) in
let time_small := (time_large * P_small) / P_large in
time_small

theorem time_to_complete_smaller_field_20_minutes 
  (w l l_large w_large : ℝ)
  (h1 : l = 1.5 * w) 
  (h2 : l_large = 3 * l) 
  (h3 : w_large = 4 * w) 
  (h4 : 68 = 68) :
  time_to_complete_smaller_field w l l_large w_large 68 = 20 := 
begin
  sorry
end

end time_to_complete_smaller_field_20_minutes_l620_620452


namespace train_crossing_time_l620_620459

theorem train_crossing_time
  (distance : ℕ) (speed_kmh : ℕ) (distance_eq_800 : distance = 800)
  (speed_eq_144 : speed_kmh = 144) :
  let speed_ms := speed_kmh * 1000 / 3600 in 
  let time := distance / speed_ms in
  time = 20 := by
  sorry

end train_crossing_time_l620_620459


namespace isogonal_conjugate_curves_isogonal_conjugate_curves_tangent_isogonal_conjugate_curves_intersect_isotomic_conjugate_curves_isotomic_conjugate_curves_tangent_isotomic_conjugate_curves_intersect_l620_620580

-- Given definitions and lean statements for the problem
variables {ABC : Type} [triangle ABC] {l : Type} [line l]
variable (ω : Type) [circumcircle ω ABC] -- circumcircle of triangle ABC
variable (S : Type) [steiner_ellipse S ABC] -- Steiner ellipse of triangle ABC

-- Definitions manifesting the problem conditions
def does_not_intersect (l : Type) (C : Type) := ¬ ∃ (p : Type), (p ∈ l) ∧ (p ∈ C)
def is_tangent (l : Type) (C : Type) := ∃ (p : Type), (p ∈ l) ∧ (p ∈ C) ∧ ∀ (q : Type), ((q ∈ l) ∧ (q ≠ p)) → (q ∉ C)
def intersects_at_two_points (l : Type) (C : Type) := ∃ (p q : Type), (p ≠ q) ∧ (p ∈ l ∧ p ∈ C) ∧ (q ∈ l ∧ q ∈ C)

theorem isogonal_conjugate_curves (h1 : does_not_intersect l ω) :
  is_ellipse (isogonal_conjugate l ABC) := sorry

theorem isogonal_conjugate_curves_tangent (h2 : is_tangent l ω) :
  is_parabola (isogonal_conjugate l ABC) := sorry

theorem isogonal_conjugate_curves_intersect (h3 : intersects_at_two_points l ω) :
  is_hyperbola (isogonal_conjugate l ABC) := sorry

theorem isotomic_conjugate_curves (h4 : does_not_intersect l S) :
  is_ellipse (isotomic_conjugate l ABC) := sorry

theorem isotomic_conjugate_curves_tangent (h5 : is_tangent l S) :
  is_parabola (isotomic_conjugate l ABC) := sorry

theorem isotomic_conjugate_curves_intersect (h6 : intersects_at_two_points l S) :
  is_hyperbola (isotomic_conjugate l ABC) := sorry

end isogonal_conjugate_curves_isogonal_conjugate_curves_tangent_isogonal_conjugate_curves_intersect_isotomic_conjugate_curves_isotomic_conjugate_curves_tangent_isotomic_conjugate_curves_intersect_l620_620580


namespace sufficient_food_l620_620490

-- Definitions for a large package and a small package as amounts of food lasting days
def largePackageDay : ℕ := L
def smallPackageDay : ℕ := S

-- Condition: One large package and four small packages last 14 days
axiom condition : largePackageDay + 4 * smallPackageDay = 14

-- Theorem to be proved: One large package and three small packages last at least 11 days
theorem sufficient_food : largePackageDay + 3 * smallPackageDay ≥ 11 :=
by
  sorry

end sufficient_food_l620_620490


namespace b_sequence_is_constant_l620_620595

noncomputable def b_sequence_formula (a b : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n > 0 → ∃ d q : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ (∀ n : ℕ, b (n + 1) = b n * q)) ∧
  (∀ n : ℕ, n > 0 → a (n + 1) / a n = b n) ∧
  (∀ n : ℕ, n > 0 → b n = 1)

theorem b_sequence_is_constant (a b : ℕ → ℝ) (h : b_sequence_formula a b) : ∀ n : ℕ, n > 0 → b n = 1 :=
  by
    sorry

end b_sequence_is_constant_l620_620595


namespace largest_three_digit_multiple_of_17_l620_620282

theorem largest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n ≤ 999) ∧ (n ≥ 100) ∧ (17 ∣ n) ∧ ∀ m, ((m > n) ∧ (m ≤ 999) ∧ (m ≥ 100) ∧ (17 ∣ m)) → false :=
by {
  sorry
}

end largest_three_digit_multiple_of_17_l620_620282


namespace mel_younger_than_katherine_l620_620134

theorem mel_younger_than_katherine :
  ∀ (k m : ℕ), (k = 24) → (m = 21) → (k - m = 3) :=
by
  intros k m hk hm,
  rw [hk, hm],
  norm_num

end mel_younger_than_katherine_l620_620134


namespace function_is_polynomial_l620_620108

variable (f : ℝ → ℝ)

theorem function_is_polynomial (h : ∀ n : ℕ, n ≥ 2 → Polynomial (f^[n])) : Polynomial f := sorry

end function_is_polynomial_l620_620108


namespace chocolate_bar_cost_l620_620545

-- Definitions based on the conditions given in the problem.
def total_bars : ℕ := 7
def remaining_bars : ℕ := 4
def total_money : ℚ := 9
def bars_sold : ℕ := total_bars - remaining_bars
def cost_per_bar := total_money / bars_sold

-- The theorem that needs to be proven.
theorem chocolate_bar_cost : cost_per_bar = 3 := by
  -- proof placeholder
  sorry

end chocolate_bar_cost_l620_620545


namespace distance_from_A_to_y_axis_l620_620180

variable (x y : ℝ)

def distance_to_y_axis (x : ℝ) : ℝ := |x|

theorem distance_from_A_to_y_axis (hx : x = -2) (hy : y = 1) :
  distance_to_y_axis x = 2 :=
by
  rw [hx]
  simp [distance_to_y_axis]
  norm_num

#eval distance_from_A_to_y_axis (by rfl) (by rfl)

end distance_from_A_to_y_axis_l620_620180


namespace total_sum_of_permutations_l620_620702

theorem total_sum_of_permutations (digits : List ℕ) (h_digits : digits = [2, 0, 1, 8]) :
  let numbers := digits.permutations.map (λ l, l.foldl (λ a b, a * 10 + b) 0)
  ∑ n in numbers.to_finset, n = 78331 := 
by
  sorry

end total_sum_of_permutations_l620_620702


namespace percentage_of_students_in_60_to_69_range_is_20_l620_620830

theorem percentage_of_students_in_60_to_69_range_is_20 :
  let scores := [4, 8, 6, 5, 2]
  let total_students := scores.sum
  let students_in_60_to_69 := 5
  (students_in_60_to_69 * 100 / total_students) = 20 := by
  sorry

end percentage_of_students_in_60_to_69_range_is_20_l620_620830


namespace inequality_proof_l620_620924

-- Define the conditions and the theorem statement
variables {a b c d : ℝ}

theorem inequality_proof (h1 : c < d) (h2 : a > b) (h3 : b > 0) : a - c > b - d :=
by
  sorry

end inequality_proof_l620_620924


namespace problem_correct_l620_620680

noncomputable def S : Set ℕ := {x | x^2 - x = 0}
noncomputable def T : Set ℕ := {x | x ∈ Set.univ ∧ 6 % (x - 2) = 0}

theorem problem_correct : S ∩ T = ∅ :=
by sorry

end problem_correct_l620_620680


namespace problem1_problem2_l620_620864

noncomputable def proof1 : ℝ :=
  let a := (9 / 4)^(1/2)
  let b := (-9.6)^0
  let c := (27 / 8)^(-2 / 3)
  let d := (1.5)^(-2)
  a - b - c + d

theorem problem1 : proof1 = 1 / 2 := by
  sorry
  
noncomputable def proof2 : ℝ :=
  let a := Real.log 3 (sqrt 27)
  let b := Real.log10 25
  let c := Real.log10 4
  let d := 7^(Real.log 7 2)
  let e := (-9.8)^0
  a + b + c + d + e

theorem problem2 : proof2 = 13 / 2 := by
  sorry

end problem1_problem2_l620_620864


namespace jimin_class_students_l620_620888

theorem jimin_class_students 
    (total_distance : ℝ)
    (interval_distance : ℝ)
    (h1 : total_distance = 242)
    (h2 : interval_distance = 5.5) :
    (total_distance / interval_distance) + 1 = 45 :=
by sorry

end jimin_class_students_l620_620888


namespace triangle_not_necessarily_equilateral_l620_620743

def is_equilateral (ABC : Triangle) : Prop :=
  ∀ {A B C : Point}, ABC.angle A B C = 60 ∧ ABC.angle B C A = 60 ∧ ABC.angle C A B = 60

def midpoint (A B : Point) : Point := sorry
def altitude (A B C : Point) : Line := sorry
def base (altitude : Line) : Point := sorry

theorem triangle_not_necessarily_equilateral
  (ABC : Triangle)
  (A B C : Point)
  (M : Point := midpoint A C)
  (H : Point := base (altitude B A C))
  (K : Point := base (altitude C B A))
  (equilateral : is_equilateral (Triangle.mk M H K)) :
  ¬ is_equilateral ABC :=
sorry

end triangle_not_necessarily_equilateral_l620_620743


namespace series_sum_l620_620484

theorem series_sum : 
  let series := List.range' (-3) (10,004 - (-3) + 1) in
  (List.zipWith (fun x y -> x + y) (series |>.filter (fun x => x % 2 ≠ 0)) (series |>.filter (fun x => x % 2 = 0))).sum = 5004 := 
by
  sorry

end series_sum_l620_620484


namespace ab_value_l620_620035

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_value_l620_620035


namespace books_sold_on_friday_correct_l620_620669

def initial_stock : ℕ := 900
def books_sold_monday : ℕ := 75
def books_sold_tuesday : ℕ := 50
def books_sold_wednesday : ℕ := 64
def books_sold_thursday : ℕ := 78
def unsold_percentage : ℝ := 55.333333333333336 / 100

noncomputable def books_unsold : ℕ := (unsold_percentage * initial_stock).round.toNat
def books_sold_mon_to_thu : ℕ := books_sold_monday + books_sold_tuesday + books_sold_wednesday + books_sold_thursday
def books_sold_friday : ℕ := initial_stock - books_unsold - books_sold_mon_to_thu

theorem books_sold_on_friday_correct : books_sold_friday = 135 :=
by
  sorry

end books_sold_on_friday_correct_l620_620669


namespace find_common_difference_l620_620583

-- Define the arithmetic sequence properties
def arithmetic_sequence (a d : ℤ) (n : ℕ) : Prop :=
  ∀ m, a + m * d = a + m * d

-- Define the sum of the first n terms of an arithmetic sequence
def sum_of_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  n * a + (n * (n - 1) / 2) * d

-- The given problem conditions
def first_term : ℤ := 2022
def sum_first_20 : ℤ := 22
def n : ℕ := 20

-- The goal is to prove the common difference d is -20209/95
def common_difference : ℚ := -20209 / 95

-- The theorem statement, no proof required per problem instructions
theorem find_common_difference :
  ∃ d : ℚ, (sum_of_first_n_terms 2022 d 20 = 22) ∧ d = -20209 / 95 :=
begin
  use common_difference,
  split,
  { -- sum_of_first_n_terms 2022 (-20209 / 95) 20 = 22
    sorry
  },
  { -- d = -20209 / 95
    refl
  }
end

end find_common_difference_l620_620583


namespace greatest_three_digit_multiple_of_17_l620_620319

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → m ≤ n) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620319


namespace ab_equals_six_l620_620021

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_l620_620021


namespace greatest_three_digit_multiple_of_17_l620_620242

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n, (n ∈ set.Icc 100 999) ∧ (n % 17 = 0) ∧ (∀ m, (m ∈ set.Icc 100 999 ∧ m % 17 = 0) → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620242


namespace number_of_true_propositions_l620_620964

theorem number_of_true_propositions :
  let original : Prop := ∀ x y : ℝ, x^2 > y^2 → x > y,
      converse : Prop := ∀ x y : ℝ, x > y → x^2 > y^2,
      inverse : Prop := ∀ x y : ℝ, x ≤ y → x^2 ≤ y^2,
      contrapositive : Prop := ∀ x y : ℝ, x ≤ y → x^2 ≤ y^2
  in
  (¬original ∧ ¬converse ∧ ¬inverse ∧ ¬contrapositive) ↔ 0 = 0 :=
by
  -- proof
  sorry

end number_of_true_propositions_l620_620964


namespace at_least_one_musician_l620_620977

theorem at_least_one_musician (total_friends musicians non_musicians choose_friends non_musical_groups : ℕ)
    (h1 : total_friends = 10) (h2 : musicians = 4) (h3 : non_musicians = 6) (h4 : choose_friends = 4)
    (h5 : non_musical_groups = 15) : 
    (Nat.choose total_friends choose_friends) - (Nat.choose non_musicians choose_friends) = 195 :=
begin
  rw [h1, h2, h3, h4],
  have total_ways : Nat.choose 10 4 = 210 := by norm_num,
  rw total_ways,
  rw h5,
  norm_num,
end

end at_least_one_musician_l620_620977


namespace cat_food_sufficiency_l620_620520

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S >= 11 :=
sorry

end cat_food_sufficiency_l620_620520


namespace great_pyramid_sum_height_width_l620_620727

noncomputable def height_m := 170
noncomputable def meter_to_feet := 3.28084
noncomputable def height_f := height_m * meter_to_feet
noncomputable def tangent_theta_ratio := 4 / 3
noncomputable def base_length := 2 * height_f * (3 / 4)
noncomputable def sum_height_width := height_f + base_length

theorem great_pyramid_sum_height_width :
  sum_height_width = 1394.357 := 
by
  sorry

end great_pyramid_sum_height_width_l620_620727


namespace greatest_three_digit_multiple_of_17_l620_620259

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (17 ∣ n) ∧ ∀ m : ℤ, (100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m) → m ≤ n :=
begin
  use 969,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num, exact dec_trivial, },
  { intros m h_mconds,
    cases h_mconds with hm1 h_mconds,
    cases h_mconds with hm2 h_formula,
    rw h_formula, sorry, }  -- Completing this part is not necessary according to the guidelines
end

end greatest_three_digit_multiple_of_17_l620_620259


namespace cars_left_with_tom_l620_620224

theorem cars_left_with_tom : 
  let totalCars := 10 * 5,
      carsGiven := 2 * (1 / 5 * totalCars) 
  in totalCars - carsGiven = 30 := 
by 
  -- Definitions from conditions
  let totalCars : ℕ := 10 * 5
  let carsGiven : ℕ := 2 * (1 / 5 : ℝ) * totalCars

  -- The final statement
  have h1 : totalCars - carsGiven = 50 - 20 := sorry

  -- The known correct answer
  exact h1.trans (by norm_num)

end cars_left_with_tom_l620_620224


namespace ratio_of_men_to_women_l620_620763

theorem ratio_of_men_to_women
  (wage_mt: ℕ)
  (wage_w: ℕ)
  (n_men_first: ℕ)
  (n_women_first: ℕ)
  (total_wages_first: ℕ)
  (n_women_second: ℕ)
  (sum_wages_second: ℕ) :
  wage_mt = 350 → n_men_first = 24 → n_women_first = 16 →
  total_wages_first = 11600 → n_women_second = 37 →
  sum_wages_second = 11600 →
  24 * 350 + 16 * wage_w = 11600 →
  (wage_w = 200 ∧ 
   (m : ℕ), m * wage_mt + n_women_second * wage_w = 11600 → m = 12) → 
  ratio_of_men_to_women = (12:37) :=
begin
  intros,
  -- steps proving the ratio will be added here
  sorry
end

end ratio_of_men_to_women_l620_620763


namespace fraction_equality_l620_620566

theorem fraction_equality (a b : ℝ) (h : a / b = 2 / 3) : a / (a + b) = 2 / 5 :=
by
  sorry

end fraction_equality_l620_620566


namespace number_of_paths_l620_620621

-- Definition of vertices
inductive Vertex
| A | B | C | D | E | F | G

-- Edges based on the description
def edges : List (Vertex × Vertex) := [
  (Vertex.A, Vertex.G), (Vertex.G, Vertex.C), (Vertex.G, Vertex.D), (Vertex.C, Vertex.B),
  (Vertex.D, Vertex.C), (Vertex.D, Vertex.F), (Vertex.D, Vertex.E), (Vertex.E, Vertex.F),
  (Vertex.F, Vertex.B), (Vertex.C, Vertex.F), (Vertex.A, Vertex.C), (Vertex.A, Vertex.D)
]

-- Function to count paths from A to B without revisiting any vertex
def countPaths (start : Vertex) (goal : Vertex) (adj : List (Vertex × Vertex)) : Nat :=
sorry

-- The theorem statement
theorem number_of_paths : countPaths Vertex.A Vertex.B edges = 10 :=
sorry

end number_of_paths_l620_620621


namespace fuel_consumption_per_100_km_l620_620208

-- Defining the conditions
variable (initial_fuel : ℕ) (remaining_fuel : ℕ) (distance_traveled : ℕ)

-- Assuming the conditions provided in the problem
axiom initial_fuel_def : initial_fuel = 47
axiom remaining_fuel_def : remaining_fuel = 14
axiom distance_traveled_def : distance_traveled = 275

-- The statement to prove: fuel consumption per 100 km
theorem fuel_consumption_per_100_km (initial_fuel remaining_fuel distance_traveled : ℕ) :
  initial_fuel = 47 →
  remaining_fuel = 14 →
  distance_traveled = 275 →
  (initial_fuel - remaining_fuel) * 100 / distance_traveled = 12 :=
by
  sorry

end fuel_consumption_per_100_km_l620_620208


namespace quadratic_solution_exists_for_any_a_b_l620_620711

theorem quadratic_solution_exists_for_any_a_b (a b : ℝ) : 
  ∃ x : ℝ, (a^6 - b^6)*x^2 + 2*(a^5 - b^5)*x + (a^4 - b^4) = 0 := 
by
  -- The proof would go here
  sorry

end quadratic_solution_exists_for_any_a_b_l620_620711


namespace nina_savings_weeks_l620_620136

def video_game_cost := 50
def headset_cost := 70
def sales_tax_rate := 0.12
def weekly_allowance := 10
def savings_rate := 0.40
def required_savings := 
  let total_cost := video_game_cost + headset_cost
  let tax := total_cost * sales_tax_rate
  total_cost + tax
def weekly_savings := weekly_allowance * savings_rate
def weeks_to_save := (required_savings / weekly_savings).ceil -- using ceil to round up.

theorem nina_savings_weeks : weeks_to_save = 34 := by
  sorry

end nina_savings_weeks_l620_620136


namespace greatest_three_digit_multiple_of_17_l620_620320

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → m ≤ n) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620320


namespace cat_food_problem_l620_620498

noncomputable def L : ℝ
noncomputable def S : ℝ

theorem cat_food_problem (L S : ℝ) (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
by 
  sorry

end cat_food_problem_l620_620498


namespace greatest_3_digit_multiple_of_17_l620_620380

theorem greatest_3_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≥ m) ∧ n = 986 := 
begin
  sorry
end

end greatest_3_digit_multiple_of_17_l620_620380


namespace penny_net_income_proof_l620_620148

noncomputable def penny_net_income (initial_daily_income : ℝ) (income_increase_rate : ℝ) 
  (tax_rate : ℝ) (monthly_expenses : ℝ) (days_per_month : ℕ) (months : ℕ) : ℝ :=
let month1_income := initial_daily_income * days_per_month
let month1_net := month1_income * (1 - tax_rate) - monthly_expenses
let month2_income := (initial_daily_income * (1 + income_increase_rate)) * days_per_month
let month2_net := month2_income * (1 - tax_rate) - monthly_expenses
let month3_income := ((initial_daily_income * (1 + income_increase_rate)) 
                      * (1 + income_increase_rate)) * days_per_month
let month3_net := month3_income * (1 - tax_rate) - monthly_expenses
in month1_net + month2_net + month3_net

theorem penny_net_income_proof : 
  penny_net_income 10 0.20 0.15 100 30 3 = 628.20 :=
by sorry

end penny_net_income_proof_l620_620148


namespace stratified_sampling_is_reasonable_l620_620826

-- Defining our conditions and stating our theorem
def flat_land := 150
def ditch_land := 30
def sloped_land := 90
def total_acres := 270
def sampled_acres := 18
def sampling_ratio := sampled_acres / total_acres

def flat_land_sampled := flat_land * sampling_ratio
def ditch_land_sampled := ditch_land * sampling_ratio
def sloped_land_sampled := sloped_land * sampling_ratio

theorem stratified_sampling_is_reasonable :
  flat_land_sampled = 10 ∧
  ditch_land_sampled = 2 ∧
  sloped_land_sampled = 6 := 
by
  sorry

end stratified_sampling_is_reasonable_l620_620826


namespace average_annual_cost_reduction_l620_620221

theorem average_annual_cost_reduction (x : ℝ) (h : (1 - x) ^ 2 = 0.64) : x = 0.2 :=
sorry

end average_annual_cost_reduction_l620_620221


namespace a_2021_value_l620_620130

theorem a_2021_value {a : ℕ → ℕ}
  (h1 : ∀ n ≥ 1, a n ≤ n)
  (h2 : ∀ n ≥ 2, (∑ k in Finset.range (n - 1), Real.cos (π * a k / n)) = 0) :
  a 2021 = 2021 :=
sorry

end a_2021_value_l620_620130


namespace greatest_three_digit_multiple_of_17_l620_620307

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 17 ∣ n :=
by 
  sorry

end greatest_three_digit_multiple_of_17_l620_620307


namespace total_outfits_l620_620803

def num_red_shirts : ℕ := 7
def num_green_shirts : ℕ := 7
def num_pairs_of_pants : ℕ := 8
def num_green_hats : ℕ := 10
def num_red_hats : ℕ := 10
def num_blue_hats : ℕ := 5

theorem total_outfits : 
  let total_red_shirts := num_red_shirts * num_pairs_of_pants * (num_green_hats + num_blue_hats),
      total_green_shirts := num_green_shirts * num_pairs_of_pants * (num_red_hats + num_blue_hats)
  in total_red_shirts + total_green_shirts = 1680 :=
by
  let total_red_shirts := num_red_shirts * num_pairs_of_pants * (num_green_hats + num_blue_hats)
  let total_green_shirts := num_green_shirts * num_pairs_of_pants * (num_red_hats + num_blue_hats)
  have h1 : total_red_shirts = 840 := sorry
  have h2 : total_green_shirts = 840 := sorry
  exact h1 + h2

end total_outfits_l620_620803


namespace greatest_3_digit_multiple_of_17_l620_620382

theorem greatest_3_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≥ m) ∧ n = 986 := 
begin
  sorry
end

end greatest_3_digit_multiple_of_17_l620_620382


namespace greatest_three_digit_multiple_of_17_l620_620327

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

theorem greatest_three_digit_multiple_of_17 : ∀ n, n < 1000 → is_multiple_of n 17 → n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620327


namespace find_f7_l620_620957

def f (x : ℝ) : ℝ := x^3 + 3 * x + (2^x - 1) / (2^x + 1) + 5

theorem find_f7 (h : f (-7) = -7) : f 7 = 17 :=
by
  -- Placeholder for the proof.
  sorry

end find_f7_l620_620957


namespace none_of_these_are_perfect_squares_l620_620410

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = n

theorem none_of_these_are_perfect_squares :
  ¬ is_perfect_square (19! * 20! / 2) ∧
  ¬ is_perfect_square (20! * 21! / 2) ∧
  ¬ is_perfect_square (21! * 22! / 2) ∧
  ¬ is_perfect_square (22! * 23! / 2) ∧
  ¬ is_perfect_square (23! * 24! / 2) :=
by
  sorry

end none_of_these_are_perfect_squares_l620_620410


namespace dice_circle_prob_l620_620451

theorem dice_circle_prob :
  let outcomes := [(1,1), (1,2), (2,1), (1,3), (3,1), (2,2)] in
  let valid := [1, 2, 3] in
  let prod_rolls := { (x : ℕ × ℕ) // 1 ≤ x.fst ∧ x.fst ≤ 6 ∧ 1 ≤ x.snd ∧ x.snd ≤ 6 } in
  let prob := (valid.card / prod_rolls.card).toRational in
  prob = 5 / 36 :=
by
  -- The detailed proof would go here.
  sorry

end dice_circle_prob_l620_620451


namespace greatest_three_digit_multiple_of_17_l620_620342

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620342


namespace dihedral_angle_PBD_A_is_60_l620_620851

-- Define the points A, B, C, D, P and the conditions given
variables (A B C D P : Type) [metric_space A]
variables [metric_space B] [metric_space C] [metric_space D] [metric_space P]

-- Given lengths and properties
variables {PA : ℝ} (PA_eq : PA = 3)
variables {AD : ℝ} (AD_eq : AD = 2)
variables {AB : ℝ} (AB_eq : AB = 2 * real.sqrt 3)
variables {BC : ℝ} (BC_eq : BC = 6)

-- Given angle conditions
variables (angle_ABC_eq : ∠ A B C = 90)
variables (parallel_AD_BC : parallel AD BC)
variables (perpendicular_PA_to_plane : ∀ (plane : set (Type)), orthogonal PA plane)

-- Define the dihedral angle between planes PBD and ABD
def dihedral_angle (P : Type) (B : Type) (D : Type) (A : Type) : ℝ := sorry

-- State the problem
theorem dihedral_angle_PBD_A_is_60 :
  dihedral_angle P B D A = 60 :=
sorry

end dihedral_angle_PBD_A_is_60_l620_620851


namespace prob_four_sons_four_daughters_l620_620694

open Finset

theorem prob_four_sons_four_daughters 
  (n : ℕ := 8) 
  (p : ℚ := 1 / 2) 
  (k : ℕ := 4) :
  let total_combinations := 2^n in
  let favorable_combinations := nat.choose n k in
  (favorable_combinations : ℚ) / (total_combinations : ℚ) = 35 / 128 := 
by
  sorry

end prob_four_sons_four_daughters_l620_620694


namespace greatest_three_digit_multiple_of_17_l620_620235

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n, (n ∈ set.Icc 100 999) ∧ (n % 17 = 0) ∧ (∀ m, (m ∈ set.Icc 100 999 ∧ m % 17 = 0) → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620235


namespace cat_food_sufficiency_l620_620512

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end cat_food_sufficiency_l620_620512


namespace coaxial_circle_l620_620455

theorem coaxial_circle
  (circle1 circle2 : Type) 
  (A1 B1 C1 D1 A2 B2 C2 D2 : Type) 
  (secant1 secant2 : Type) 
  (A1_on_circle1 : A1 ∈ circle1) (B1_on_circle1 : B1 ∈ circle1)
  (C1_on_circle1 : C1 ∈ circle1) (D1_on_circle1 : D1 ∈ circle1)
  (A2_on_circle2 : A2 ∈ circle2) (B2_on_circle2 : B2 ∈ circle2)
  (C2_on_circle2 : C2 ∈ circle2) (D2_on_circle2 : D2 ∈ circle2)
  (A1B1_on_secant1 : A1 ∈ secant1) (B1_on_secant1 : B1 ∈ secant1)
  (A2B2_on_secant1 : A2 ∈ secant1) (B2_on_secant1 : B2 ∈ secant1)
  (C1D1_on_secant2 : C1 ∈ secant2) (D1_on_secant2 : D1 ∈ secant2)
  (C2D2_on_secant2 : C2 ∈ secant2) (D2_on_secant2 : D2 ∈ secant2) :
  ∃ P X Q R, P ∈ circle1 ∧ P ∈ circle2 ∧
             X ∈ circle1 ∧ X ∈ circle2 ∧ 
             Q ∈ circle1 ∧ Q ∈ circle2 ∧ 
             R ∈ circle1 ∧ R ∈ circle2 ∧ 
             (P = line_intersection (A1, C1) (B2, D2)) ∧
             (X = line_intersection (A1, C1) (A2, C2)) ∧
             (Q = line_intersection (A2, C2) (B1, D1)) ∧
             (R = line_intersection (B2, D2) (B1, D1)) ∧
             are_concyclic P X Q R :=
by
  sorry

end coaxial_circle_l620_620455


namespace triangle_area_range_l620_620929

-- Defining the given conditions
def parabola : Set (ℝ × ℝ) := {p | p.2^2 = 4 * p.1}

def focus : ℝ × ℝ := (1, 0)
def directrix_intersection : ℝ × ℝ := (-1, 0)

-- Function for the line passing through the focus
def line_passing_through_focus (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 = k * (p.1 - 1)}

-- Formalizing the problem in Lean 4 statement
theorem triangle_area_range :
  ∀ (k : ℝ), let A := (some (parabola ∩ (line_passing_through_focus k))),
                B := (some (parabola ∩ (line_passing_through_focus k))),
                D := directrix_intersection in
          (4 ≤ (1/2) * |D.1 * (A.2 - B.2) + A.1 * (B.2 - D.2) + B.1 * (D.2 - A.2)| ) :=
by
  sorry

end triangle_area_range_l620_620929


namespace cost_of_gravelling_path_l620_620806

def rectangular_plot_length : ℝ := 100
def rectangular_plot_breadth : ℝ := 70
def path_width : ℝ := 2.5
def gravelling_cost_per_sqm: ℝ := 0.90

theorem cost_of_gravelling_path:
  let total_area := rectangular_plot_length * rectangular_plot_breadth in
  let grassy_area_length := rectangular_plot_length - 2 * path_width in
  let grassy_area_breadth := rectangular_plot_breadth - 2 * path_width in
  let grassy_area := grassy_area_length * grassy_area_breadth in
  let path_area := total_area - grassy_area in
  let cost := path_area * gravelling_cost_per_sqm in
  cost = 742.5 :=
by
  sorry

end cost_of_gravelling_path_l620_620806


namespace distance_between_parallel_lines_l620_620176

-- Definitions of the line equations
def line1 (x y : ℝ) : Prop := 2 * x - y + 3 = 0
def line2 (x y : ℝ) : Prop := -4 * x + 2 * y + 5 = 0

-- Statement of the problem
theorem distance_between_parallel_lines :
  let d := |3 - (-5 / 2)| / real.sqrt (2^2 + (-1)^2)
  d = (11 * real.sqrt 5) / 10 := 
sorry

end distance_between_parallel_lines_l620_620176


namespace ratio_of_female_contestants_l620_620476

theorem ratio_of_female_contestants (T M F : ℕ) (hT : T = 18) (hM : M = 12) (hF : F = T - M) :
  F / T = 1 / 3 :=
by
  sorry

end ratio_of_female_contestants_l620_620476


namespace hcf_lcm_product_l620_620748

theorem hcf_lcm_product (a b : ℕ) (H : a * b = 45276) (L : Nat.lcm a b = 2058) : Nat.gcd a b = 22 :=
by 
  -- The proof steps go here
  sorry

end hcf_lcm_product_l620_620748


namespace converse_diagonals_bisect_each_other_l620_620414

theorem converse_diagonals_bisect_each_other (Q : Type) [quadrilateral Q] :
  (∀ A B C D : Q, midpoint (diagonal AC) = midpoint (diagonal BD)) ↔ is_parallelogram Q :=
sorry

end converse_diagonals_bisect_each_other_l620_620414


namespace min_guesses_correct_l620_620429

def min_guesses (n k : ℕ) : ℕ :=
  if n = 2 * k then 2 else 1

theorem min_guesses_correct (n k : ℕ) (h : n > k) :
  (min_guesses n k = 2 ↔ n = 2 * k) ∧ (min_guesses n k = 1 ↔ n ≠ 2 * k) := by
  sorry

end min_guesses_correct_l620_620429


namespace quadratic_polynomial_solution_l620_620905

def q (x : ℝ) : ℝ := (4/5) * x^2 - (1/5) * x + (17/5)

theorem quadratic_polynomial_solution : 
  q (-2) = 7 ∧ q 1 = 4 ∧ q 3 = 10 := by
    -- We are given the polynomial q(x) and need to prove it meets the conditions.
    sorry

end quadratic_polynomial_solution_l620_620905


namespace berengere_needs_to_contribute_zero_l620_620861

-- Define the conversion rate
def exchange_rate : ℝ := 1.1

-- Define the amount Emily has in USD
def emily_usd : ℝ := 10

-- Define the conversion of Emily's money to euros
def emily_euros : ℝ := emily_usd / exchange_rate

-- Define the cost of the cake in euros
def cake_cost_euros : ℝ := 8

-- Define the contribution needed from Berengere in euros
def berengere_contribution : ℝ := max 0 (cake_cost_euros - emily_euros)

-- The theorem that Berengere needs to contribute 0 euros
theorem berengere_needs_to_contribute_zero : berengere_contribution = 0 :=
by
  sorry

end berengere_needs_to_contribute_zero_l620_620861


namespace no_solution_sqrt_eq_l620_620167

theorem no_solution_sqrt_eq :
  ¬ ∃ t : ℝ, -7 ≤ t ∧ t ≤ 7 ∧ sqrt (49 - t^2) + 7 = 0 :=
by sorry

end no_solution_sqrt_eq_l620_620167


namespace ab_eq_six_l620_620064

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_eq_six_l620_620064


namespace ratio_determinant_change_l620_620750

theorem ratio_determinant_change (B D W : ℕ) (B' D' W' : ℕ) :
  (B : ℕ) / D = 2 / 40 ∧ D / W = 40 / 100 ∧ 
  B' / D' = 3 * (B / D) ∧ D' = 40 ∧ W' = 200 → 
  (D' / W') = (1 / 5 : ℝ) ∧ (D / W) = (2 / 5 : ℝ) → 
  change_factor (D / W) (D' / W') = (1 / 2 : ℝ) :=
by 
  sorry

noncomputable def change_factor (x y : ℝ) : ℝ := 
  y / x

end ratio_determinant_change_l620_620750


namespace sufficient_food_l620_620491

-- Definitions for a large package and a small package as amounts of food lasting days
def largePackageDay : ℕ := L
def smallPackageDay : ℕ := S

-- Condition: One large package and four small packages last 14 days
axiom condition : largePackageDay + 4 * smallPackageDay = 14

-- Theorem to be proved: One large package and three small packages last at least 11 days
theorem sufficient_food : largePackageDay + 3 * smallPackageDay ≥ 11 :=
by
  sorry

end sufficient_food_l620_620491


namespace total_elephants_l620_620196

-- Definitions and Hypotheses
def W : ℕ := 70
def G : ℕ := 3 * W

-- Proposition
theorem total_elephants : W + G = 280 := by
  sorry

end total_elephants_l620_620196


namespace greatest_three_digit_multiple_of_17_l620_620347

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620347


namespace ordered_pair_A_B_l620_620188

noncomputable def cubic_function (x : ℝ) : ℝ := x^3 - 2 * x^2 - 3 * x + 6
noncomputable def linear_function (x : ℝ) : ℝ := -2 / 3 * x + 2

noncomputable def points_intersect (x1 x2 x3 y1 y2 y3 : ℝ) : Prop :=
  cubic_function x1 = y1 ∧ cubic_function x2 = y2 ∧ cubic_function x3 = y3 ∧
  2 * x1 + 3 * y1 = 6 ∧ 2 * x2 + 3 * y2 = 6 ∧ 2 * x3 + 3 * y3 = 6

theorem ordered_pair_A_B (x1 x2 x3 y1 y2 y3 A B : ℝ)
  (h_intersect : points_intersect x1 x2 x3 y1 y2 y3) 
  (h_sum_x : x1 + x2 + x3 = A)
  (h_sum_y : y1 + y2 + y3 = B) :
  (A, B) = (2, 14 / 3) :=
by {
  sorry
}

end ordered_pair_A_B_l620_620188


namespace next_time_angle_150_degrees_l620_620974

theorem next_time_angle_150_degrees (x : ℝ) : (∃ x, x = 329/6) :=
by
  let θ := λ H M : ℝ, abs (30 * H - 5.5 * M)
  let initial_angle := θ 5 0
  have eq1 : initial_angle = 150 :=
    by sorry
  let H := 5 + x / 60
  have eq2 : θ H x = 150 :=
    by sorry
  have eq3 : abs (150 - 5 * x) = 150 :=
    by sorry
  have eq4 : abs (150 - 5 * x) = 150 := by sorry
  have solution : x = 54 + 6 / 11 :=
    by sorry
  existsi solution
  sorry

end next_time_angle_150_degrees_l620_620974


namespace hourly_wage_calculation_l620_620171

variable (H : ℝ)
variable (hours_per_week : ℝ := 40)
variable (wage_per_widget : ℝ := 0.16)
variable (widgets_per_week : ℝ := 500)
variable (total_earnings : ℝ := 580)

theorem hourly_wage_calculation :
  (hours_per_week * H + widgets_per_week * wage_per_widget = total_earnings) →
  H = 12.5 :=
by
  intro h_equation
  -- Proof steps would go here
  sorry

end hourly_wage_calculation_l620_620171


namespace greatest_three_digit_multiple_of_17_l620_620262

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (17 ∣ n) ∧ ∀ m : ℤ, (100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m) → m ≤ n :=
begin
  use 969,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num, exact dec_trivial, },
  { intros m h_mconds,
    cases h_mconds with hm1 h_mconds,
    cases h_mconds with hm2 h_formula,
    rw h_formula, sorry, }  -- Completing this part is not necessary according to the guidelines
end

end greatest_three_digit_multiple_of_17_l620_620262


namespace rod_length_difference_l620_620473

theorem rod_length_difference (L₁ L₂ : ℝ) (h1 : L₁ + L₂ = 33)
    (h2 : (∀ x : ℝ, x = (2 / 3) * L₁ ∧ x = (4 / 5) * L₂)) :
    abs (L₁ - L₂) = 3 := by
  sorry

end rod_length_difference_l620_620473


namespace bacteria_reach_target_l620_620443

def bacteria_growth (initial : ℕ) (target : ℕ) (doubling_time : ℕ) (delay : ℕ) : ℕ :=
  let doubling_count := Nat.log2 (target / initial)
  doubling_count * doubling_time + delay

theorem bacteria_reach_target : 
  bacteria_growth 800 25600 5 3 = 28 := by
  sorry

end bacteria_reach_target_l620_620443


namespace CorrectCalculation_l620_620799

theorem CorrectCalculation : 
  ¬ ((-3 * -3 = 6) ∨ (-sqrt 3 * -sqrt 3 = 9) ∨ (3 / x + 3 / x = 6 / 2 * x)) ∧ ((-3 * x) * (-3 * x)^2 = -27 * x^3) := 
by {
  sorry
}

end CorrectCalculation_l620_620799


namespace largest_three_digit_multiple_of_17_l620_620283

theorem largest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n ≤ 999) ∧ (n ≥ 100) ∧ (17 ∣ n) ∧ ∀ m, ((m > n) ∧ (m ≤ 999) ∧ (m ≥ 100) ∧ (17 ∣ m)) → false :=
by {
  sorry
}

end largest_three_digit_multiple_of_17_l620_620283


namespace scoops_for_mom_l620_620151

/-- 
  Each scoop of ice cream costs $2.
  Pierre gets 3 scoops.
  The total bill is $14.
  Prove that Pierre's mom gets 4 scoops.
-/
theorem scoops_for_mom
  (scoop_cost : ℕ)
  (pierre_scoops : ℕ)
  (total_bill : ℕ) :
  scoop_cost = 2 → pierre_scoops = 3 → total_bill = 14 → 
  (total_bill - pierre_scoops * scoop_cost) / scoop_cost = 4 := 
by
  intros h1 h2 h3
  sorry

end scoops_for_mom_l620_620151


namespace greatest_three_digit_multiple_of_17_l620_620264

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (17 ∣ n) ∧ ∀ m : ℤ, (100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m) → m ≤ n :=
begin
  use 969,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num, exact dec_trivial, },
  { intros m h_mconds,
    cases h_mconds with hm1 h_mconds,
    cases h_mconds with hm2 h_formula,
    rw h_formula, sorry, }  -- Completing this part is not necessary according to the guidelines
end

end greatest_three_digit_multiple_of_17_l620_620264


namespace rotation_equivalence_l620_620871

theorem rotation_equivalence
  (A B C : Point)
  (rotated_clockwise_450 : rotate_clockwise A B 450 = C)
  (rotated_counterclockwise_y : rotate_counterclockwise A B y = C)
  (y_lt_360 : y < 360) :
  y = 270 := 
begin
  -- The proof will go here.
  sorry
end

end rotation_equivalence_l620_620871


namespace value_of_f_at_2_l620_620868

def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem value_of_f_at_2 : f 2 = 62 :=
by
  -- The proof will be inserted here, it follows Horner's method steps shown in the solution
  sorry

end value_of_f_at_2_l620_620868


namespace participant_rank_score_l620_620640

-- Define the problem space
def score_bound : ℕ := 7
def problem_count : ℕ := 6
def participant_count : ℕ := score_bound^problem_count
def rank_required : ℕ := (score_bound - 1)^problem_count

-- Define the function relating to the product of scores
noncomputable def participant_score (scores: Fin 8 → ℕ) : ℕ := 
  (Finset.univ : Finset (Fin 8)).prod (λ i => scores i)

-- Assert the conditions and the ranking requirement
theorem participant_rank_score (scores: Fin 8 → ℕ)
  (h: ∀ i, scores i ≤ score_bound ∧ scores i ≥ 0 )
  (unique: ∀ p1 p2: Fin 8 → ℕ, participant_score p1 = participant_score p2 ∧ ∑ i, p1 i = ∑ i, p2 i → p1 = p2)
  (h_rank: rank_required = 117649) :
  ∃ scores, participant_score scores == 1 :=
sorry

end participant_rank_score_l620_620640


namespace proof_equations_l2_l3_l620_620450

open Real

noncomputable def equation_l2 : Prop :=
  ∃ (M M' P : Point),
    M = ⟨-1, 3⟩ ∧
    M' = ⟨M.1, -M.2⟩ ∧
    P = ⟨1, 0⟩ ∧
    (∃ e : AffineLine, e = AffineLine.mk M' P ∧ ∀ x, e.equation x = (3/2) * (x - 1))

noncomputable def equation_l3 : Prop :=
  ∃ (N P' : Point),
    N = ⟨11/5, 9/5⟩ ∧
    P' = ⟨4, 3⟩ ∧
    (∃ e : AffineLine, e = AffineLine.mk N P' ∧ ∀ x, e.equation x = (2/3) * (x - 4))

theorem proof_equations_l2_l3 : equation_l2 ∧ equation_l3 := sorry

end proof_equations_l2_l3_l620_620450


namespace saturday_temperature_l620_620862

def temperatures : List ℝ := [99.1, 98.2, 98.7, 99.3, 99.8, 99.0]

def average_temperature_week : ℝ := 99.0

theorem saturday_temperature : (∑ t in temperatures, t) + x = average_temperature_week * 7 → x = 98.9 :=
by
  intros
  sorry

end saturday_temperature_l620_620862


namespace benny_birthday_money_l620_620860

-- Define conditions
def spent_on_gear : ℕ := 47
def left_over : ℕ := 32

-- Define the total amount Benny received
def total_money_received : ℕ := 79

-- Theorem statement
theorem benny_birthday_money (spent_on_gear : ℕ) (left_over : ℕ) : spent_on_gear + left_over = total_money_received :=
by
  sorry

end benny_birthday_money_l620_620860


namespace product_of_ab_l620_620049

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end product_of_ab_l620_620049


namespace partition_infinite_sets_100_max_intersecting_sets_l620_620682

theorem partition_infinite_sets_100 (A : Set (ℚ × ℚ)) (L : Set (ℚ × ℚ)) :
  (∀ p ∈ A, ∀ q ∈ A, p ≠ q → (∃ l : Set (ℚ × ℚ), Line l ∧ p ∈ l ∧ q ∈ l)) →
  (∃ A_i : Fin 100 → Set (ℚ × ℚ),
    (∀ i, ∃ n : ℕ, A_i i = {p : ℚ × ℚ | 100 * n + i ≤ p.1 ^ 2 + p.2 ^ 2 ∧ p.1 ^ 2 + p.2 ^ 2 < 100 * n + i + 1} ∧
           A_i i ∩ A_i j = ∅ ∀ i ≠ j ∧ (∀ l : Set (ℚ × ℚ), Line l ∧ (∃ p q ∈ A, p ∈ l ∧ q ∈ l) → ∀ i, ∃ p ∈ A_i i, p ∈ l)) :=
by sorry

theorem max_intersecting_sets (A : Set (ℚ × ℚ)) :
  (∀ A_i : Fin 100 → Set (ℚ × ℚ), (∀ i, A_i i ∩ A_i j = ∅ ∀ i ≠ j ∧ ∀ i, set.infinite (A_i i)) →
    (∃ r : ℕ, r ≤ 100 ∧ ∀ l : Set (ℚ × ℚ), Line l ∧ (∃ i j, A_i i ∩ l ≠ ∅ ∧ A_i j ∩ l ≠ ∅) → ∃ i ≤ r, ∀ i, ∃ p ∈ A_i i, p ∈ l)) ∧
  (r = 3) :=
by sorry

end partition_infinite_sets_100_max_intersecting_sets_l620_620682


namespace greatest_three_digit_multiple_of_17_l620_620233

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n, (n ∈ set.Icc 100 999) ∧ (n % 17 = 0) ∧ (∀ m, (m ∈ set.Icc 100 999 ∧ m % 17 = 0) → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620233


namespace measure_angle_BHC_l620_620659

-- Definition of the angles based on the conditions
def angle_ABC := 54
def angle_ACB := 16

-- The statement to prove the measure of angle BHC given the defined conditions
theorem measure_angle_BHC (angle_ABC angle_ACB : ℕ) (angle_ABC = 54) (angle_ACB = 16) : 
  ∠ BHC = 70 :=
sorry

end measure_angle_BHC_l620_620659


namespace license_plates_count_l620_620626

/-- 
A theorem to prove that the number of possible license plates is 1250.
The license plates consist of 2 vowels followed by 2 digits. Both digits 
must be either odd or even numbers.
-/
theorem license_plates_count : 
  let vowels := 5,  -- 5 vowels: A, E, I, O, U
      choices_for_vowels := vowels * vowels,  -- Each of the 2 positions can be filled by any of 5 vowels
      digit_types := 2,  -- 2 types: both odd or both even
      choices_for_digits := 5 * 5  -- Each of the 2 positions can be filled by any of 5 odd or even digits
  in
  choices_for_vowels * digit_types * choices_for_digits = 1250 := 
by
  sorry

end license_plates_count_l620_620626


namespace ab_value_l620_620031

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_value_l620_620031


namespace ab_eq_six_l620_620063

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_eq_six_l620_620063


namespace greatest_three_digit_multiple_of_17_l620_620288

def is_multiple_of (n : ℕ) (k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, is_three_digit n ∧ is_multiple_of n 17 → n = 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620288


namespace factorization_x8_minus_81_l620_620525

theorem factorization_x8_minus_81 (x : ℝ) : 
  x^8 - 81 = (x^4 + 9) * (x^2 + 3) * (x + real.sqrt 3) * (x - real.sqrt 3) :=
by
  sorry

end factorization_x8_minus_81_l620_620525


namespace total_sum_of_numbers_l620_620704

theorem total_sum_of_numbers : 
  let digits := [2, 0, 1, 8]
  let all_permutations := Finset.univ.powerset.filter (λ s, s.card ≠ 0) -- Generating all non-empty subsets
  let all_numbers := all_permutations.map (finset.sum_digits) -- Convert each subset into a number
  let total_sum := all_numbers.sum
  in total_sum = 78311 := 
by
  sorry

end total_sum_of_numbers_l620_620704


namespace cos_half_angle_product_leq_three_sqrt_three_over_eight_l620_620421

theorem cos_half_angle_product_leq_three_sqrt_three_over_eight (α β γ : ℝ) (h1 : α + β + γ = π) 
  (h2 : sin α * sin β * sin γ ≤ 3 * Real.sqrt 3 / 8) : 
  cos (α / 2) * cos (β / 2) * cos (γ / 2) ≤ 3 * Real.sqrt 3 / 8 :=
by sorry

end cos_half_angle_product_leq_three_sqrt_three_over_eight_l620_620421


namespace sum_non_prime_between_50_60_eq_383_start_number_is_50_l620_620206

def is_non_prime (n : ℕ) : Prop := ∃ d, 1 < d ∧ d < n ∧ d ∣ n

def non_prime_numbers_between (start finish : ℕ) : List ℕ :=
  (List.range (finish - start - 1)).map (λ i => start + i + 1) |>.filter is_non_prime

theorem sum_non_prime_between_50_60_eq_383 : (non_prime_numbers_between 50 60).sum = 383 :=
sorry

theorem start_number_is_50 (n : ℕ) (hn : (non_prime_numbers_between n 60).sum = 383) : n = 50 :=
by {
  have h : non_prime_numbers_between 50 60 = [51, 52, 54, 55, 56, 57, 58] := 
  begin 
    sorry
  end,
  have sum_is_383 : (non_prime_numbers_between 50 60).sum = 383 := sum_non_prime_between_50_60_eq_383,
  have h₁ : (non_prime_numbers_between n 60).sum = (non_prime_numbers_between 50 60).sum := by rw hn,
  exact h₁.symm.trans sum_is_383
}

end sum_non_prime_between_50_60_eq_383_start_number_is_50_l620_620206


namespace solve_problem_l620_620685

theorem solve_problem
    (x y z : ℝ)
    (h1 : x > 0)
    (h2 : y > 0)
    (h3 : z > 0)
    (h4 : x^2 + x * y + y^2 = 2)
    (h5 : y^2 + y * z + z^2 = 5)
    (h6 : z^2 + z * x + x^2 = 3) :
    x * y + y * z + z * x = 2 * Real.sqrt 2 := 
by
  sorry

end solve_problem_l620_620685


namespace discount_is_five_l620_620839
-- Importing the needed Lean Math library

-- Defining the problem conditions
def costPrice : ℝ := 100
def profit_percent_with_discount : ℝ := 0.2
def profit_percent_without_discount : ℝ := 0.25

-- Calculating the respective selling prices
def sellingPrice_with_discount := costPrice * (1 + profit_percent_with_discount)
def sellingPrice_without_discount := costPrice * (1 + profit_percent_without_discount)

-- Calculating the discount 
def calculated_discount := sellingPrice_without_discount - sellingPrice_with_discount

-- Proving that the discount is $5
theorem discount_is_five : calculated_discount = 5 := by
  -- Proof omitted
  sorry

end discount_is_five_l620_620839


namespace collinear_if_lambda_mu_eq_one_l620_620596

def vector_not_collinear (a b : ℝ^2) : Prop :=
  ¬ (∃ k : ℝ, a = k • b ∨ b = k • a)

def vectors_collinear (u v : ℝ^2) : Prop :=
  ∃ k : ℝ, u = k • v

theorem collinear_if_lambda_mu_eq_one 
  (a b : ℝ^2) (λ μ : ℝ)
  (h₁ : vector_not_collinear a b)
  (h₂ : vectors_collinear (λ • a + b) (a + μ • b)) :
  λ * μ = 1 :=
sorry

end collinear_if_lambda_mu_eq_one_l620_620596


namespace chemistry_problem_l620_620821

theorem chemistry_problem 
(C : ℝ)  -- concentration of the original salt solution
(h_mix : 1 * C / 100 = 15 * 2 / 100) : 
  C = 30 := 
sorry

end chemistry_problem_l620_620821


namespace ages_problem_l620_620619

theorem ages_problem :
  let
    mother_age := 80
    grandmother_age := 3 * mother_age
    grace_age := (3 / 8 : ℝ) * grandmother_age
    father_age := (7 / 12 : ℝ) * grandmother_age
    brother_age := (2 / 5 : ℝ) * grace_age
    sister_age := (3 / 7 : ℝ) * brother_age
  in
    grace_age = 90 ∧ father_age = 140 ∧ brother_age = 36 ∧ sister_age ≈ 15.43 :=
by
  let mother_age := 80
  let grandmother_age := 3 * mother_age
  let grace_age := (3 / 8 : ℝ) * grandmother_age
  let father_age := (7 / 12 : ℝ) * grandmother_age
  let brother_age := (2 / 5 : ℝ) * grace_age
  let sister_age := (3 / 7 : ℝ) * brother_age
  have h1 : grace_age = 90 := sorry
  have h2 : father_age = 140 := sorry
  have h3 : brother_age = 36 := sorry
  have h4 : sister_age ≈ 15.43 := sorry
  exact ⟨h1, h2, h3, h4⟩

end ages_problem_l620_620619


namespace ab_value_l620_620028

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_value_l620_620028


namespace average_mpg_calc_l620_620477

variable (O1 O2 O3 G1 G2 G3 : ℝ)

theorem average_mpg_calc (hO1 : O1 = 35400) (hO2 : O2 = 35680) (hO3 : O3 = 36000)
    (hG1 : G1 = 8) (hG2 : G2 = 15) (hG3 : G3 = 18) :
    let distance := O3 - O1 in
    let total_gas := G1 + G2 + G3 in
    let avg_mpg := distance / total_gas in
    Float.round (avg_mpg * 10) / 10 = 14.6 :=
by
  sorry

end average_mpg_calc_l620_620477


namespace total_games_in_league_l620_620082

theorem total_games_in_league (n : ℕ) (h : n = 25) : 
  (n * (n - 1)) / 2 = 300 :=
by
  rw h
  norm_num
  sorry

end total_games_in_league_l620_620082


namespace Zuminglish_mod_1000_l620_620995

def a : ℕ → ℕ
| 2 => 4
| (n+1) => 2 * (a n + c n)
with b : ℕ → ℕ
| 2 => 2
| (n+1) => a n
with c : ℕ → ℕ
| 2 => 2
| (n+1) => 2 * b n

def S (n : ℕ) : ℕ := a n + b n + c n

theorem Zuminglish_mod_1000 : S 12 % 1000 = 416 :=
by
  sorry -- Proof omitted

end Zuminglish_mod_1000_l620_620995


namespace triangle_proof_l620_620152

theorem triangle_proof
  (A B C P M D E : Type*)
  [point : ∀ x, Mathlib.point x]
  (h1 : Mathlib.acute_triangle A B C)
  (h2 : Mathlib.inside_triangle P A B C)
  (h3 : Mathlib.angle_eq (Mathlib.angle B A P) (Mathlib.angle C A P))
  (h4 : Mathlib.midpoint M B C)
  (circ_ABP circ_ACP : Mathlib.circumcircle)
  (h5 : Mathlib.on_circumcircle D A B P circ_ABP)
  (h6 : Mathlib.on_circumcircle E A C P circ_ACP)
  (h7 : Mathlib.between P M E)
  (h8 : Mathlib.between P E D)
  (h9 : Mathlib.dist_eq (Mathlib.dist D E) (Mathlib.dist M P)) :
  Mathlib.dist_eq (Mathlib.dist B C) (2 * Mathlib.dist B P) :=
sorry

end triangle_proof_l620_620152


namespace sufficient_food_supply_l620_620503

variable {L S : ℝ}

theorem sufficient_food_supply (h1 : L + 4 * S = 14) (h2 : L > S) : L + 3 * S ≥ 11 :=
by
  sorry

end sufficient_food_supply_l620_620503


namespace decreasing_interval_l620_620586

def f (a x : ℝ) : ℝ := x^2 + 2*(a - 1)*x + 2

theorem decreasing_interval (a : ℝ) : (∀ x y : ℝ, x ≤ y → y ≤ 4 → f a y ≤ f a x) ↔ a < -3 := 
by
  sorry

end decreasing_interval_l620_620586


namespace product_of_numbers_l620_620226

variable (x y : ℕ)

theorem product_of_numbers : x + y = 120 ∧ x - y = 6 → x * y = 3591 := by
  sorry

end product_of_numbers_l620_620226


namespace circle_line_intersects_l620_620752

theorem circle_line_intersects (λ : ℝ) :
    ∃ (A : ℝ × ℝ), (A.1 - 1)^2 + A.2^2 - 7 < 0 ∧ 
                   ((λ + 1) * A.1 - A.2 + 1 - λ = 0) :=
begin
    sorry
end

end circle_line_intersects_l620_620752


namespace ab_eq_six_l620_620065

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_eq_six_l620_620065


namespace proof_expression_C_equals_negative_one_l620_620467

def A : ℤ := abs (-1)
def B : ℤ := -(-1)
def C : ℤ := -(1^2)
def D : ℤ := (-1)^2

theorem proof_expression_C_equals_negative_one : C = -1 :=
by 
  sorry

end proof_expression_C_equals_negative_one_l620_620467


namespace rectangular_prism_edge_properties_l620_620627

-- Define a rectangular prism and the concept of parallel and perpendicular pairs of edges.
structure RectangularPrism :=
  (vertices : Fin 8 → Fin 3 → ℝ)
  -- Additional necessary conditions on the structure could be added here.

-- Define the number of parallel edges in a rectangular prism
def number_of_parallel_edge_pairs (rp : RectangularPrism) : ℕ :=
  -- Formula or logic to count parallel edge pairs.
  8 -- Placeholder for actual logic computation, based on problem conditions.

-- Define the number of perpendicular edges in a rectangular prism
def number_of_perpendicular_edge_pairs (rp : RectangularPrism) : ℕ :=
  -- Formula or logic to count perpendicular edge pairs.
  20 -- Placeholder for actual logic computation, based on problem conditions.

-- Theorem that asserts the requirement based on conditions
theorem rectangular_prism_edge_properties (rp : RectangularPrism) :
  number_of_parallel_edge_pairs rp = 8 ∧ number_of_perpendicular_edge_pairs rp = 20 :=
  by
    -- Placeholder proof that establishes the theorem
    sorry

end rectangular_prism_edge_properties_l620_620627


namespace evaluate_expression_l620_620893

open Int

theorem evaluate_expression : (Int.ceil (7 / 3) + Int.floor (-7 / 3) - 3 = -3) := by
	sorry

end evaluate_expression_l620_620893


namespace circle_table_acquaintance_impossible_l620_620775

theorem circle_table_acquaintance_impossible (P : Finset ℕ) (hP : P.card = 40) :
  ¬ (∀ (a b : ℕ), (a ∈ P) → (b ∈ P) → (∃ k, 2 * k ≠ 0) → (∃ c, c ∈ P) ∧ (a ≠ b) ∧ (c = a ∨ c = b)
       ↔ ¬(∃ k, 2 * k + 1 ≠ 0)) :=
by
  sorry

end circle_table_acquaintance_impossible_l620_620775


namespace greatest_three_digit_multiple_of_17_l620_620352

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (n % 17 = 0) ∧ ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ (m % 17 = 0) → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620352


namespace shawna_wednesday_situps_l620_620718

def shawna_goal : ℕ := 30
def monday_situps : ℕ := 12
def tuesday_situps : ℕ := 19

theorem shawna_wednesday_situps : 
  let monday_deficit := shawna_goal - monday_situps in
  let tuesday_deficit := shawna_goal - tuesday_situps in
  let total_deficit := monday_deficit + tuesday_deficit + shawna_goal in
  total_deficit = 59 := by
  sorry

end shawna_wednesday_situps_l620_620718


namespace greatest_three_digit_multiple_of_17_l620_620397

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) := 
  exists.intro 986 sorry

end greatest_three_digit_multiple_of_17_l620_620397


namespace greatest_three_digit_multiple_of_17_l620_620321

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → m ≤ n) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620321


namespace greatest_three_digit_multiple_of_17_l620_620313

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → m ≤ n) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620313


namespace range_m_l620_620966

theorem range_m (m : ℝ) :
  let A := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
  let B := {x : ℝ | 2 * m - 1 ≤ x ∧ x ≤ m + 1}
  B ⊆ A → (-1 / 2) ≤ m :=
by {
  let A := {x : ℝ | -2 ≤ x ∧ x ≤ 5},
  let B := {x : ℝ | 2 * m - 1 ≤ x ∧ x ≤ m + 1},
  intros h,
  sorry
}

end range_m_l620_620966


namespace football_team_matches_l620_620094

theorem football_team_matches (total_matches loses total_points: ℕ) 
  (points_win points_draw points_lose wins draws: ℕ)
  (h1: total_matches = 15)
  (h2: loses = 4)
  (h3: total_points = 29)
  (h4: points_win = 3)
  (h5: points_draw = 1)
  (h6: points_lose = 0)
  (h7: wins + draws + loses = total_matches)
  (h8: points_win * wins + points_draw * draws = total_points) :
  wins = 9 ∧ draws = 2 :=
sorry


end football_team_matches_l620_620094


namespace greatest_three_digit_multiple_of_17_l620_620333

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

theorem greatest_three_digit_multiple_of_17 : ∀ n, n < 1000 → is_multiple_of n 17 → n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620333


namespace greatest_3_digit_multiple_of_17_l620_620381

theorem greatest_3_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≥ m) ∧ n = 986 := 
begin
  sorry
end

end greatest_3_digit_multiple_of_17_l620_620381


namespace students_height_order_valid_after_rearrangement_l620_620724
open List

variable {n : ℕ} -- number of students in each row
variable (a b : Fin n → ℝ) -- heights of students in each row

/-- Prove Gábor's observation remains valid after rearrangement: 
    each student in the back row is taller than the student in front of them.
    Given:
    - ∀ i, b i < a i (initial condition)
    - ∀ i < j, a i ≤ a j (rearrangement condition)
    Prove:
    - ∀ i, b i < a i (remains valid after rearrangement)
-/
theorem students_height_order_valid_after_rearrangement
  (h₁ : ∀ i : Fin n, b i < a i)
  (h₂ : ∀ (i j : Fin n), i < j → a i ≤ a j) :
  ∀ i : Fin n, b i < a i :=
by sorry

end students_height_order_valid_after_rearrangement_l620_620724


namespace reflection_over_line_y_eq_x_l620_620732

theorem reflection_over_line_y_eq_x {x y x' y' : ℝ} (h_c : (x, y) = (6, -5)) (h_reflect : (x', y') = (y, x)) :
  (x', y') = (-5, 6) :=
  by
    simp [h_c, h_reflect]
    sorry

end reflection_over_line_y_eq_x_l620_620732


namespace trig_identity_l620_620153

theorem trig_identity:
  sin (π / 14) * sin (3 * π / 14) * sin (5 * π / 14) = 1 / 8 :=
by
  sorry

end trig_identity_l620_620153


namespace scientific_notation_of_4800000_l620_620189

theorem scientific_notation_of_4800000 : 
  ∃ (c : ℕ), 4800000 = 4.8 * 10 ^ c ∧ c = 6 := 
sorry

end scientific_notation_of_4800000_l620_620189


namespace cos_sum_identity_1_cos_sum_identity_2_l620_620164

variable (α β γ : ℝ)

theorem cos_sum_identity_1 (h : α + β + γ = 180) :
  cos α + cos β + cos γ = 4 * sin (α / 2) * sin (β / 2) * sin (γ / 2) + 1 :=
sorry

theorem cos_sum_identity_2 (h : α + β + γ = 180) :
  cos α + cos β - cos γ = 4 * cos (α / 2) * cos (β / 2) * sin (γ / 2) - 1 :=
sorry

end cos_sum_identity_1_cos_sum_identity_2_l620_620164


namespace greatest_three_digit_multiple_of_17_l620_620237

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n, (n ∈ set.Icc 100 999) ∧ (n % 17 = 0) ∧ (∀ m, (m ∈ set.Icc 100 999 ∧ m % 17 = 0) → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620237


namespace sum_y_equals_17_l620_620933

def S (a_n b n : ℝ) := a_n^2 + b * n

def f (x : ℝ) := 2 + sin (2 * x) - 2 * (sin (x / 2))^2

def y (a_n : ℝ) := f a_n 

axiom a9_eq_pi_div_2 : a_n 9 = Real.pi / 2

theorem sum_y_equals_17 {a_n b : ℝ} (h1: ∀ n, S (a_n n) b n = ∑ i in range n, a_n i) :
   (∑ i in range 17, y (a_n i)) = 17 := 
sorry

end sum_y_equals_17_l620_620933


namespace university_diploma_percentage_l620_620647

variables (population : ℝ)
          (U : ℝ) -- percentage of people with a university diploma
          (J : ℝ := 0.40) -- percentage of people with the job of their choice
          (S : ℝ := 0.10) -- percentage of people with a secondary school diploma pursuing further education

-- Condition 1: 18% of the people do not have a university diploma but have the job of their choice.
-- Condition 2: 25% of the people who do not have the job of their choice have a university diploma.
-- Condition 3: 10% of the people have a secondary school diploma and are pursuing further education.
-- Condition 4: 60% of the people with secondary school diploma have the job of their choice.
-- Condition 5: 30% of the people in further education have a job of their choice as well.
-- Condition 6: 40% of the people have the job of their choice.

axiom condition_1 : 0.18 * population = (0.18 * (1 - U)) * (population)
axiom condition_2 : 0.25 * (100 - J * 100) = 0.25 * (population - J * population)
axiom condition_3 : S * population = 0.10 * population
axiom condition_4 : 0.60 * S * population = (0.60 * S) * population
axiom condition_5 : 0.30 * S * population = (0.30 * S) * population
axiom condition_6 : J * population = 0.40 * population

theorem university_diploma_percentage : U * 100 = 37 :=
by sorry

end university_diploma_percentage_l620_620647


namespace ellipse_equation_max_area_triangle_l620_620584

open Real

theorem ellipse_equation (a b c : ℝ) (h1 : a^2 = b^2 + c^2) (h2 : b = sqrt 3) (h3 : 0 < b) (h4 : 0 < c) (h5 : a > b > 0) (h6 : a + c = 3) :
  ellipse_eq : ((∀ x y, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 4 + y^2 / 3 = 1)) :=
by sorry

theorem max_area_triangle (a b c : ℝ) (h1 : a^2 = b^2 + c^2) (h2 : b = sqrt 3) (h3 : 0 < b) (h4 : 0 < c) (h5 : a > b > 0) (h6 : a + c = 3) (line_eq : ℝ → ℝ) (F₁ : ℝ × ℝ := (-1, 0)) (F₂ : ℝ × ℝ := (1, 0)) :
  ∃ A B : ℝ × ℝ, max_area : (area (F₂, A, B) ≤ 3) :=
by sorry

end ellipse_equation_max_area_triangle_l620_620584


namespace greatest_three_digit_multiple_of_17_l620_620247

def is_three_digit (n : Int) : Prop := (n ≥ 100 ∧ n ≤ 999)
def is_multiple_of_17 (n : Int) : Prop := (∃ k : Int, n = 17 * k)

theorem greatest_three_digit_multiple_of_17 : ∀ n : Int, is_three_digit n ∧ is_multiple_of_17 n → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l620_620247


namespace cat_food_problem_l620_620497

noncomputable def L : ℝ
noncomputable def S : ℝ

theorem cat_food_problem (L S : ℝ) (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
by 
  sorry

end cat_food_problem_l620_620497


namespace area_of_BCDE_minus_circle_l620_620673

noncomputable def area_inside_quadrilateral_but_outside_circle (hex_side : ℝ) (circle_radius : ℝ) : ℝ :=
  let area_hex : ℝ := 6 * (√3 / 4 * hex_side^2)
  let area_quad : ℝ := 4 * (√3 / 4 * hex_side^2)
  let area_circle : ℝ := π * circle_radius^2
  area_quad - area_circle

theorem area_of_BCDE_minus_circle (hex_side : ℝ) (circle_radius : ℝ) (correct_area : ℝ) :
  hex_side = 2 → circle_radius = 3 → correct_area = 4 * √3 - 9 * π →
  area_inside_quadrilateral_but_outside_circle hex_side circle_radius = correct_area :=
by
  intros h_side h_radius h_correct_area
  sorry

end area_of_BCDE_minus_circle_l620_620673


namespace log_eq_implies_eq_l620_620884

theorem log_eq_implies_eq (p q : ℝ) (hq : q ≠ 1) :
  log p + log (q^2) = log (p + q^2) → p = q^2 / (q^2 - 1) :=
by
  sorry

end log_eq_implies_eq_l620_620884


namespace marble_arrangement_remainder_l620_620889

theorem marble_arrangement_remainder : 
  let m := 19 in 
  let total_marbles := m + 7 in 
  let N := nat.choose total_marbles 7 in 
  N % 1000 = 388 :=
begin
  let m := 19,
  let total_marbles := m + 7,
  let N := nat.choose total_marbles 7,
  have hN : N = 50388,
  {
    -- proof of N = 50388
    sorry
  },
  have h_mod : 50388 % 1000 = 388,
  { 
    -- proof of modulo operation
    sorry 
  },
  exact h_mod,
end

end marble_arrangement_remainder_l620_620889


namespace count_intersections_l620_620738

noncomputable theory

def log2 (x : ℝ) := Real.log x / Real.log 2

def y1 (x : ℝ) := log2 x
def y2 (x : ℝ) := 1 / log2 x
def y3 (x : ℝ) := -log2 x
def y4 (x : ℝ) := -1 / log2 x

theorem count_intersections : 
  {p : ℝ × ℝ | p.1 > 0 ∧ 
    ((p.2 = y1 p.1 ∧ p.2 = y2 p.1) ∨ 
    (p.2 = y1 p.1 ∧ p.2 = y3 p.1) ∨ 
    (p.2 = y1 p.1 ∧ p.2 = y4 p.1) ∨ 
    (p.2 = y2 p.1 ∧ p.2 = y3 p.1) ∨ 
    (p.2 = y2 p.1 ∧ p.2 = y4 p.1) ∨ 
    (p.2 = y3 p.1 ∧ p.2 = y4 p.1))}.toFinset.card = 3 :=
by
  sorry

end count_intersections_l620_620738


namespace edward_levels_beaten_l620_620546

theorem edward_levels_beaten : ∃ B : ℕ, (3 * (32 - B) = B) ∧ B = 24 :=
by
  use 24
  split
  sorry
  refl

end edward_levels_beaten_l620_620546


namespace perpendicular_PK_AB_l620_620107

open Classical

noncomputable section

variables {A B C D K L M P : Type} [InnerProductSpace ℝ A] [InnerProductSpace ℝ B]
variables [InnerProductSpace ℝ C] [InnerProductSpace ℝ D] [InnerProductSpace ℝ K]
variables [InnerProductSpace ℝ L] [InnerProductSpace ℝ M] [InnerProductSpace ℝ P]
variables [NormedSpace ℝ A] [NormedSpace ℝ B] [NormedSpace ℝ C] [NormedSpace ℝ D]
variables [NormedSpace ℝ K] [NormedSpace ℝ L] [NormedSpace ℝ M] [NormedSpace ℝ P]

-- Assumptions/Conditions as per the problem:
variables (h1 : acute_triangle A B C)
variables (h2 : altitude D A B C)
variables (h3 : bisector DAC K B C)
variables (h4 : projection K L A C)
variables (h5 : intersection BL AD M)
variables (h6 : intersection MC DL P)

-- Proposition to prove:
theorem perpendicular_PK_AB : is_perpendicular PK AB :=
by sorry

end perpendicular_PK_AB_l620_620107


namespace best_sampling_method_l620_620441

theorem best_sampling_method :
  let elderly := 27
  let middle_aged := 54
  let young := 81
  let total_population := elderly + middle_aged + young
  let sample_size := 36
  let sampling_methods := ["simple random sampling", "systematic sampling", "stratified sampling"]
  stratified_sampling
:=
by
  sorry

end best_sampling_method_l620_620441


namespace greatest_three_digit_multiple_of_17_l620_620339

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620339


namespace number_of_stickers_after_losing_page_l620_620767

theorem number_of_stickers_after_losing_page (stickers_per_page : ℕ) (initial_pages : ℕ) (pages_lost : ℕ) :
  stickers_per_page = 20 → initial_pages = 12 → pages_lost = 1 → (initial_pages - pages_lost) * stickers_per_page = 220 :=
by
  intros
  sorry

end number_of_stickers_after_losing_page_l620_620767


namespace greatest_three_digit_multiple_of_17_l620_620306

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 17 ∣ n :=
by 
  sorry

end greatest_three_digit_multiple_of_17_l620_620306


namespace greatest_three_digit_multiple_of_17_l620_620332

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

theorem greatest_three_digit_multiple_of_17 : ∀ n, n < 1000 → is_multiple_of n 17 → n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620332


namespace question1_question2_l620_620942

noncomputable def z1 : ℂ := 1 - complex.i
noncomputable def z2 (a : ℝ) : ℂ := 3 + a * complex.i

theorem question1 (a : ℝ) (h : z1 + z2 a = 4) : z1 * z2 a = 4 - 2 * complex.i :=
begin
  have h_real : a = 1,
  sorry,
  rw h_real,
  -- Additional proof steps skipped
  sorry,
end

theorem question2 (a : ℝ) (h : ∀ re : ℝ, z2 a / z1 = complex.i * re) : complex.abs (z2 a) = 3 * real.sqrt 2 :=
begin
  have h_pure_imag : a = 3,
  sorry,
  rw h_pure_imag,
  -- Additional proof steps skipped
  sorry,
end

end question1_question2_l620_620942


namespace find_positive_integer_n_l620_620931

theorem find_positive_integer_n (S : ℕ → ℚ) (hS : ∀ n, S n = n / (n + 1))
  (h : ∃ n : ℕ, S n * S (n + 1) = 3 / 4) : 
  ∃ n : ℕ, n = 6 := 
by {
  sorry
}

end find_positive_integer_n_l620_620931


namespace greatest_three_digit_multiple_of_17_l620_620310

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 17 ∣ n :=
by 
  sorry

end greatest_three_digit_multiple_of_17_l620_620310


namespace intersection_point_of_line_and_y_axis_l620_620754

theorem intersection_point_of_line_and_y_axis :
  {p : ℝ × ℝ | ∃ x, p = (x, 2 * x + 1) ∧ x = 0} = {(0, 1)} :=
by sorry

end intersection_point_of_line_and_y_axis_l620_620754


namespace finite_rational_points_l620_620882

noncomputable def point_count : ℕ := sorry

theorem finite_rational_points :
  ∃ n : ℕ, (∀ (x y : ℚ), (∃ p q r s : ℕ, prime p ∧ prime q ∧ prime r ∧ prime s ∧ x = (p / q) ∧ y = (r / s) ∧ x > 0 ∧ y > 0 ∧ x + y ≤ 7) → true) ∧ point_count = n :=
begin
  sorry
end

end finite_rational_points_l620_620882


namespace locus_is_circle_l620_620983

open Complex

-- Definitions based only on given conditions
def p_locus (θ : ℝ) (z : ℂ) : ℂ := z^2 + 2 * z * Complex.cos θ

-- The goal is to prove that the locus of p is a circle given |z| = 1 and θ is the argument of z
theorem locus_is_circle (z : ℂ) (θ : ℝ) (hz : abs z = 1) : 
  ∃ (C : ℂ) (r : ℝ), ∀ (z : ℂ), abs z = 1 → 
    abs (p_locus θ z - C) = r := 
sorry

end locus_is_circle_l620_620983


namespace correct_operation_l620_620411

variable {R : Type*} [CommRing R] (x y : R)

theorem correct_operation : x * (1 + y) = x + x * y :=
by sorry

end correct_operation_l620_620411


namespace find_x_l620_620967

open Real

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (2, -1)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_x (x : ℝ) : 
  let ab := (a.1 + x * b.1, a.2 + x * b.2)
  let minus_b := (-b.1, -b.2)
  dot_product ab minus_b = 0 
  → x = -2 / 5 :=
by
  intros
  sorry

end find_x_l620_620967


namespace cat_food_problem_l620_620495

noncomputable def L : ℝ
noncomputable def S : ℝ

theorem cat_food_problem (L S : ℝ) (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
by 
  sorry

end cat_food_problem_l620_620495


namespace calculate_value_l620_620867

theorem calculate_value :
  let a := (120 / 15)
  let b := (15 * 18)
  let c := (405 / 9)
  let d := (3^3)
  65 + a + b - 250 - c + d = 75 :=
by
  let a := (120 / 15)
  let b := (15 * 18)
  let c := (405 / 9)
  let d := (3^3)
  have ha : a = 8 := by norm_num
  have hb : b = 270 := by norm_num
  have hc : c = 45 := by norm_num
  have hd : d = 27 := by norm_num
  calc 
    65 + a + b - 250 - c + d = 65 + 8 + 270 - 250 - 45 + 27 := by rw [ha, hb, hc, hd]
    ... = 75 := by norm_num

end calculate_value_l620_620867


namespace cos_sin_identity_l620_620485

theorem cos_sin_identity : 
  (Real.cos (75 * Real.pi / 180) + Real.sin (75 * Real.pi / 180)) * 
  (Real.cos (75 * Real.pi / 180) - Real.sin (75 * Real.pi / 180)) = -Real.sqrt 3 / 2 := 
  sorry

end cos_sin_identity_l620_620485


namespace find_A_l620_620779

-- Condition setup
def A (n : ℕ) : ℕ := (n / 100)

-- The main theorem based on the conditions and correct answer
theorem find_A: ∀ (A : ℕ), 
  A < 10 →
  let n := A * 100 + 27 in
  (n / 100) = 2 →
  A = 2 :=
begin
  intros A h1 hn hround,
  sorry
end

end find_A_l620_620779


namespace example_problem_l620_620613

theorem example_problem 
  (a b : ℝ) 
  (a_pos : a > 0) 
  (b_pos : b > 0) : 
  (\frac{1}{a} + \frac{1}{b} = \sqrt{ab}) → (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by
  sorry

end example_problem_l620_620613


namespace find_angle_BAC_l620_620454

-- Definitions used in the problem
def isRectangle (W XYZ : ℝ) : Prop := true -- assuming some properties of a rectangle

variables {A B C W X Y Z : ℝ}
variables (angleBWZ angleCXY : ℝ)

-- The proof statement
theorem find_angle_BAC
  (h1 : isRectangle W X Y Z)
  (h2 : angleBWZ = 22)
  (h3 : angleCXY = 65) :
  ∃ angleBAC : ℝ, angleBAC = 87 :=
by 
  -- skip the actual proof
  sorry

end find_angle_BAC_l620_620454


namespace greatest_three_digit_multiple_of_17_l620_620363

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 100 ≤ n ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l620_620363


namespace _l620_620662

noncomputable def range_of_angle_A (a b c : ℝ) (A : ℝ) [triangle_ABC : Triangle a b c A] : Prop :=
  (0 < A ∧ A ≤ Real.pi / 3) → 
  (b / (b + c) + c / (a + b) ≥ 1)

noncomputable theorem range_of_angle_A_in_triangle (a b c A : ℝ)
  (hypotheses : b / (b + c) + c / (a + b) ≥ 1 ∧ 0 < A ∧ A ≤ Real.pi / 3)
  (h_triangle : Triangle a b c A) :
   ∃ (A : ℝ), A ∈ (0, Real.pi / 3] :=
sorry

end _l620_620662


namespace vertical_asymptote_condition_l620_620563

theorem vertical_asymptote_condition (c : ℝ) :
  (∀ x : ℝ, (x = 3 ∨ x = -6) → (x^2 - x + c = 0)) → 
  (c = -6 ∨ c = -42) :=
by
  sorry

end vertical_asymptote_condition_l620_620563


namespace sequence_properties_l620_620170

noncomputable def b : ℕ → ℕ 
| 0 := 0
| 1 := 2
| 2 := 3
| (n + 3) := b (n + 1) * b (n + 2) + 1

theorem sequence_properties :
  b 3 = 7 ∧ ∀ n > 0, ∃ k > 0, b (n + 1) ≠ k * b n := 
by
  sorry

end sequence_properties_l620_620170


namespace triangle_area_l620_620230

theorem triangle_area :
  let A := (2, -3)
  let B := (2, 4)
  let C := (8, 0) 
  let base := (4 - (-3))
  let height := (8 - 2)
  let area := (1 / 2) * base * height
  area = 21 := 
by 
  sorry

end triangle_area_l620_620230


namespace complex_number_solution_l620_620947

-- Define the given complex number
def z (a b : ℝ) := a + b * complex.I

-- Define the condition
theorem complex_number_solution (a b : ℝ) (hz : (complex.mk (real.sqrt 3) 3) * (z a b) = complex.I * 3) :
  (z a b) = (z (3 / 4) (real.sqrt 3 / 4)) :=
by sorry

end complex_number_solution_l620_620947


namespace smallest_number_123456789_l620_620744

theorem smallest_number_123456789 : 
  (exists f : ℕ → ℕ, 
    (f 0 = 123456789 ∧ 
     (∀ n, let x := f n, x = 101010101 ∨ (exists i, 0 < i ∧ i < 9 ∧ 
     (∃ a b, (nat.digits 10 x).nth i = some a ∧ (nat.digits 10 x).nth (i+1) = some b ∧ 
      a ≠ 0 ∧ b ≠ 0 ∧ (nat.digits 10 (f (n + 1))).nth i = some (b-1) ∧ 
      (nat.digits 10 (f (n + 1))).nth (i+1) = some (a-1)))))) →
  ∃ n, f n = 101010101 :=
by
  sorry

end smallest_number_123456789_l620_620744


namespace greatest_three_digit_multiple_of_17_l620_620396

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) := 
  exists.intro 986 sorry

end greatest_three_digit_multiple_of_17_l620_620396


namespace stickers_after_loss_l620_620772

-- Conditions
def stickers_per_page : ℕ := 20
def initial_pages : ℕ := 12
def lost_pages : ℕ := 1

-- Problem statement
theorem stickers_after_loss : (initial_pages - lost_pages) * stickers_per_page = 220 := by
  sorry

end stickers_after_loss_l620_620772


namespace charles_share_l620_620438

-- Given conditions
variables (J A C : ℕ)
variable (hJ : J = 3600)
variable (hA_J : A = 1.2 * J)
variable (hA_C : A = 1.25 * C)

-- Prove the question
theorem charles_share (hJ : J = 3600) (hA_J : A = 1.2 * J) (hA_C : A = 1.25 * C) : C = 3456 := 
by
  sorry

end charles_share_l620_620438


namespace ab_eq_six_l620_620062

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_eq_six_l620_620062


namespace evaluate_expression_l620_620550

theorem evaluate_expression : (827 * 827) - (826 * 828) + 2 = 3 := by
  sorry

end evaluate_expression_l620_620550


namespace ab_eq_six_l620_620061

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_eq_six_l620_620061


namespace correct_sunset_time_l620_620143

noncomputable def length_of_daylight : ℕ × ℕ := (9, 48)  -- 9 hours and 48 minutes
noncomputable def sunrise_time_in_pm : ℕ × ℕ := (16, 35)  -- 16:35 in 24-hour format

theorem correct_sunset_time :
  let daylight_hours := length_of_daylight.1,
      daylight_minutes := length_of_daylight.2,
      sunrise_hours := sunrise_time_in_pm.1,
      sunrise_minutes := sunrise_time_in_pm.2,
      total_minutes := sunrise_minutes + daylight_minutes,
      extra_hours := total_minutes / 60,
      sunset_minutes := total_minutes % 60,
      total_hours := sunrise_hours + daylight_hours + extra_hours,
      sunset_hours := if total_hours >= 24 then total_hours - 24 else total_hours
  in (sunset_hours = 2) ∧ (sunset_minutes = 23) := 
by
  sorry

end correct_sunset_time_l620_620143


namespace balls_in_boxes_distribution_l620_620629

theorem balls_in_boxes_distribution :
  ∑ k in Finset.range 4, (Nat.choose 7 k) * 2 ^ (7 - k) = 1808 :=
by
  sorry

end balls_in_boxes_distribution_l620_620629


namespace samuel_book_pages_total_l620_620158

/-- Samuel read an average of 42 pages per day for the first three days,
an average of 39 pages per day for the next three days,
and then finished the book by reading 28 pages on the last day.
Prove that the total number of pages in the book is 271. -/
theorem samuel_book_pages_total
  (avg_first_three_days : ℕ = 42)
  (avg_next_three_days : ℕ = 39)
  (pages_last_day : ℕ = 28)
  (days_first_phase : ℕ = 3)
  (days_second_phase : ℕ = 3)
  : 3 * avg_first_three_days + 3 * avg_next_three_days + pages_last_day = 271 := by
  sorry

end samuel_book_pages_total_l620_620158


namespace greatest_three_digit_multiple_of_17_l620_620294

def is_multiple_of (n : ℕ) (k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, is_three_digit n ∧ is_multiple_of n 17 → n = 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620294


namespace largest_three_digit_multiple_of_17_l620_620284

theorem largest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n ≤ 999) ∧ (n ≥ 100) ∧ (17 ∣ n) ∧ ∀ m, ((m > n) ∧ (m ≤ 999) ∧ (m ≥ 100) ∧ (17 ∣ m)) → false :=
by {
  sorry
}

end largest_three_digit_multiple_of_17_l620_620284


namespace total_teachers_in_all_departments_is_637_l620_620719

noncomputable def total_teachers : ℕ :=
  let major_departments := 9
  let minor_departments := 8
  let teachers_per_major := 45
  let teachers_per_minor := 29
  (major_departments * teachers_per_major) + (minor_departments * teachers_per_minor)

theorem total_teachers_in_all_departments_is_637 : total_teachers = 637 := 
  by
  sorry

end total_teachers_in_all_departments_is_637_l620_620719


namespace greatest_three_digit_multiple_of_17_l620_620266

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (17 ∣ n) ∧ ∀ m : ℤ, (100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m) → m ≤ n :=
begin
  use 969,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num, exact dec_trivial, },
  { intros m h_mconds,
    cases h_mconds with hm1 h_mconds,
    cases h_mconds with hm2 h_formula,
    rw h_formula, sorry, }  -- Completing this part is not necessary according to the guidelines
end

end greatest_three_digit_multiple_of_17_l620_620266


namespace sum_of_squares_over_sum_ge_half_l620_620585

theorem sum_of_squares_over_sum_ge_half 
  {n : ℕ} {a : Fin n → ℝ}
  (hpos : ∀ i, 0 < a i)
  (hsum : (∑ i, a i) = 1) :
  ∑ i : Fin n, (a i) ^ 2 / (a i + a (i + 1) % n) ≥ 1 / 2 :=
sorry

end sum_of_squares_over_sum_ge_half_l620_620585


namespace flowchart_structure_correct_l620_620413

-- Definitions based on conditions
def flowchart_typically_has_one_start : Prop :=
  ∃ (start : Nat), start = 1

def flowchart_typically_has_one_or_more_ends : Prop :=
  ∃ (ends : Nat), ends ≥ 1

-- Theorem for the correct statement
theorem flowchart_structure_correct :
  (flowchart_typically_has_one_start ∧ flowchart_typically_has_one_or_more_ends) →
  (∃ (start : Nat) (ends : Nat), start = 1 ∧ ends ≥ 1) :=
by
  sorry

end flowchart_structure_correct_l620_620413


namespace single_fraction_count_is_two_l620_620638

-- Define the expressions as given
def expr1 := (m - n) / 2
def expr2 := y / Real.pi
def expr3 := (2 * x) / (x + 2)
def expr4 := (x / 7) + (y / 8)
def expr5 := 2 / y

-- Define a predicate to check if an expression is a single fraction
def is_single_fraction (e : ℝ) : Prop := 
  match e with
  | (a / b) => True
  | _       => False

-- Define the number of single fractions among the given expressions
def num_single_fractions : ℕ := 
  [is_single_fraction expr1, is_single_fraction expr2, is_single_fraction expr3, is_single_fraction expr4, is_single_fraction expr5]
  .filter id 
  .length

-- State the theorem
theorem single_fraction_count_is_two : num_single_fractions = 2 := 
by
  sorry

end single_fraction_count_is_two_l620_620638


namespace exists_unique_m0_l620_620614

-- Definition of the sequence and function
def f1 (x : ℤ) : ℤ := 2 * x + 1
def f : ℕ → ℤ → ℤ
| 0, x := x
| (n + 1), x := f1 (f n x)

-- Main theorem to be proved
theorem exists_unique_m0 (n : ℕ) (h : n ≥ 11) : ∃! m0 ∈ { m | 0 ≤ m ∧ m < 1994 }, 1995 ∣ f n m0 :=
by
  sorry

end exists_unique_m0_l620_620614


namespace product_of_distances_l620_620928

noncomputable def point := ℝ × ℝ
def P : point := (1, 1)
def angle := Real.pi / 6

def line_parametric (t : ℝ) : point :=
  (1 + (Real.sqrt 3) / 2 * t, 1 + 1 / 2 * t)

def circle (p : point) : Prop :=
  p.1 ^ 2 + p.2 ^ 2 = 4

def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem product_of_distances :
  ∃ (A B : point), circle A ∧ circle B ∧ 
  (∃ t1 t2 : ℝ, line_parametric t1 = A ∧ line_parametric t2 = B) ∧ 
  (distance P A * distance P B = 2) :=
sorry

end product_of_distances_l620_620928


namespace blankets_warm_nathan_up_l620_620406

theorem blankets_warm_nathan_up :
  (∀ (blankets_added half_blankets: ℕ), blankets_added = (half_blankets) -> half_blankets = 7 -> ∃ warmth: ℕ, warmth = blankets_added * 3  ∧ warmth = 21) :=
begin
  intros,
  sorry
end

end blankets_warm_nathan_up_l620_620406


namespace greatest_three_digit_multiple_of_17_l620_620309

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 17 ∣ n :=
by 
  sorry

end greatest_three_digit_multiple_of_17_l620_620309


namespace sum_first_20_terms_l620_620579

inductive parity
| odd
| even

open parity

def seq_def (n : ℕ) (a : ℕ → ℕ) : ℕ :=
  if n % 2 = 1 then
    2 * (a (n - 1)) - 2
  else
    (a (n - 1)) + 1

def a_seq : ℕ → ℕ
| 1 := 1
| (n + 1) := seq_def (n + 1) a_seq

def S_20 : ℕ := ∑ i in finset.range 20, a_seq (i + 1)

theorem sum_first_20_terms : S_20 = 2056 := 
by
  sorry

end sum_first_20_terms_l620_620579


namespace determine_a_l620_620597

theorem determine_a (a : ℝ) : 
  (let z := (a + 3 * complex.I) / complex.I + a in 
    (a - 3 < 0 ∧ -a > 0) → 
    a = -4) :=
by sorry

end determine_a_l620_620597


namespace range_of_a_l620_620078

theorem range_of_a :
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) ↔ -1 ≤ a ∧ a ≤ 3 := 
by
  sorry

end range_of_a_l620_620078


namespace eat_jar_together_time_l620_620823

-- Define the rate of the child
def child_rate := 1 / 6

-- Define the rate of Karlson who eats twice as fast as the child
def karlson_rate := 2 * child_rate

-- Define the combined rate when both eat together
def combined_rate := child_rate + karlson_rate

-- Prove that the time taken together to eat one jar is 2 minutes
theorem eat_jar_together_time : (1 / combined_rate) = 2 :=
by
  -- Add the proof steps here
  sorry

end eat_jar_together_time_l620_620823


namespace second_reduction_percentage_l620_620747

variable (P : ℝ) -- Original price
variable (x : ℝ) -- Second reduction percentage

-- Condition 1: After a 25% reduction
def first_reduction (P : ℝ) : ℝ := 0.75 * P

-- Condition 3: Combined reduction equivalent to 47.5%
def combined_reduction (P : ℝ) : ℝ := 0.525 * P

-- Question: Given the conditions, prove that the second reduction is 0.3
theorem second_reduction_percentage (P : ℝ) (x : ℝ) :
  (1 - x) * first_reduction P = combined_reduction P → x = 0.3 :=
by
  intro h
  sorry

end second_reduction_percentage_l620_620747


namespace problem_solution_l620_620608

noncomputable def f (x m : ℝ) : ℝ := real.log x - m * x + m

theorem problem_solution (m : ℝ) :
  (∀ x ∈ set.Ioi (0 : ℝ), f x m ≤ 0) ↔ m = 1 := 
sorry

end problem_solution_l620_620608


namespace greatest_three_digit_multiple_of_17_l620_620300

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 17 ∣ n :=
by 
  sorry

end greatest_three_digit_multiple_of_17_l620_620300


namespace trig_identity_example_l620_620542

theorem trig_identity_example :
  (Real.sin (36 * Real.pi / 180) * Real.cos (6 * Real.pi / 180) - 
   Real.sin (54 * Real.pi / 180) * Real.cos (84 * Real.pi / 180)) = 
  1 / 2 :=
by
  sorry

end trig_identity_example_l620_620542


namespace max_value_Q_eq_one_l620_620908

noncomputable def Q (a : ℝ) : ℝ :=
  -- Q(a) is the probability that cos²(π x) + cos²(π y) < a given x ∈ [0, a²] and y ∈ [0, 1]

theorem max_value_Q_eq_one (a : ℝ) (h₀ : 0 ≤ a) (h₁ : a ≤ 1) : 
  ∃ M, M = 1 ∧ ∀ a, Q(a) ≤ M := 
sorry

end max_value_Q_eq_one_l620_620908


namespace ab_eq_six_l620_620060

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_eq_six_l620_620060


namespace colleen_cookies_l620_620873

-- Definitions from the conditions
def number_of_chocolate_chips : ℕ := 108
def number_of_m_and_ms : ℕ := (1 / 3 : ℚ) * number_of_chocolate_chips
def total_chocolate_pieces : ℕ := number_of_chocolate_chips + number_of_m_and_ms.toNat
def average_pieces_per_cookie : ℕ := 3

-- The proof problem
theorem colleen_cookies : total_chocolate_pieces / average_pieces_per_cookie = 48 :=
by
  -- Insert actual proof steps here
  sorry

end colleen_cookies_l620_620873


namespace ab_value_l620_620010

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_value_l620_620010


namespace scientific_notation_141260_million_l620_620144

theorem scientific_notation_141260_million :
  (141260 * 10^6 : ℝ) = 1.4126 * 10^11 := 
sorry

end scientific_notation_141260_million_l620_620144


namespace determine_y_l620_620074

variable (x y : ℝ)

theorem determine_y (h1 : 0.25 * x = 0.15 * y - 15) (h2 : x = 840) : y = 1500 := 
by 
  sorry

end determine_y_l620_620074


namespace family_member_in_same_building_l620_620842

theorem family_member_in_same_building :
  ∃ (person : ℕ) (building : Fin 5),
  let populations := [5, 15, 25, 35, 45] in
  (∀ p ∈ populations, ∃ f1 f2 : ℕ, f1 ≠ f2 ∧ f1 ≠ p ∧ f2 ≠ p) →
  (∃ f : ℕ, f = person ∧ f ∈ populations[building]) :=
by
  sorry

end family_member_in_same_building_l620_620842


namespace Cora_pages_to_read_on_Thursday_l620_620530

theorem Cora_pages_to_read_on_Thursday
  (total_pages : ℕ)
  (read_monday : ℕ)
  (read_tuesday : ℕ)
  (read_wednesday : ℕ)
  (pages_left : ℕ)
  (read_friday : ℕ)
  (thursday_pages : ℕ) :
  total_pages = 158 →
  read_monday = 23 →
  read_tuesday = 38 →
  read_wednesday = 61 →
  pages_left = total_pages - (read_monday + read_tuesday + read_wednesday) →
  read_friday = 2 * thursday_pages →
  pages_left = thursday_pages + read_friday →
  thursday_pages = 12 :=
begin
  -- Proof is not required
  sorry
end

end Cora_pages_to_read_on_Thursday_l620_620530


namespace ab_equals_six_l620_620042

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end ab_equals_six_l620_620042


namespace remainder_degrees_l620_620798

theorem remainder_degrees (P Q : Polynomial ℝ) (hQ : Q = Polynomial.X^4 - 2 * Polynomial.X^3 + Polynomial.X - 5) : 
  ∃ R : Polynomial ℝ, (degree (P % Q) = 0 ∨ degree (P % Q) = 1 ∨ degree (P % Q) = 2 ∨ degree (P % Q) = 3) :=
by
  sorry

end remainder_degrees_l620_620798


namespace evaluate_expression_l620_620812

-- Define the conditions
def num : ℤ := 900^2
def a : ℤ := 306
def b : ℤ := 294
def denom : ℤ := a^2 - b^2

-- State the theorem to be proven
theorem evaluate_expression : (num : ℚ) / denom = 112.5 :=
by
  -- proof is skipped
  sorry

end evaluate_expression_l620_620812


namespace exists_consecutive_happy_years_l620_620469

def split_digits (n : ℕ) : ℕ × ℕ :=
  let digits := (n.toString.toList.map (λ c, c.toNat - '0'.toNat))
  if digits.length < 4 then (0, 0)
  else
    let a := digits.take 2
    let b := digits.drop 2
    (a.foldl (λ x y, x * 10 + y) 0, b.foldl (λ x y, x * 10 + y) 0)

def is_happy (n : ℕ) : Prop :=
  let (a, b) := split_digits n
  n % (a + b) = 0

theorem exists_consecutive_happy_years : ∃ n : ℕ, is_happy n ∧ is_happy (n + 1) :=
by
  have h1 := is_happy 2024
  have h2 := is_happy 2025
  existsi 2024
  split
  exact h1
  exact h2

end exists_consecutive_happy_years_l620_620469


namespace greatest_three_digit_multiple_of_17_l620_620252

def is_three_digit (n : Int) : Prop := (n ≥ 100 ∧ n ≤ 999)
def is_multiple_of_17 (n : Int) : Prop := (∃ k : Int, n = 17 * k)

theorem greatest_three_digit_multiple_of_17 : ∀ n : Int, is_three_digit n ∧ is_multiple_of_17 n → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l620_620252


namespace greatest_three_digit_multiple_of_17_l620_620355

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (n % 17 = 0) ∧ ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ (m % 17 = 0) → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620355


namespace factor_100_minus_16y2_l620_620899

theorem factor_100_minus_16y2 (y : ℝ) : 100 - 16 * y^2 = 4 * (5 - 2 * y) * (5 + 2 * y) := 
by sorry

end factor_100_minus_16y2_l620_620899


namespace greatest_three_digit_multiple_of_17_l620_620241

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n, (n ∈ set.Icc 100 999) ∧ (n % 17 = 0) ∧ (∀ m, (m ∈ set.Icc 100 999 ∧ m % 17 = 0) → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620241


namespace a_in_s_l620_620615

-- Defining the sets and the condition
def S : Set ℕ := {1, 2}
def T (a : ℕ) : Set ℕ := {a}

-- The Lean theorem statement
theorem a_in_s (a : ℕ) (h : S ∪ T a = S) : a = 1 ∨ a = 2 := 
by 
  sorry

end a_in_s_l620_620615


namespace edge_length_of_smaller_cubes_l620_620971

theorem edge_length_of_smaller_cubes 
  (edge_length_of_box : ℝ)
  (num_of_smaller_cubes : ℝ)
  (h1 : edge_length_of_box = 1)
  (h2 : num_of_smaller_cubes = 999.9999999999998) :
  ∃ edge_length_of_smaller_cube : ℝ,
    edge_length_of_smaller_cube = 0.1 :=
by
  let rounded_number := 1000 -- rounding the number of cubes
  have volume_of_box : ℝ := edge_length_of_box ^ 3
  have volume_of_smaller_cube : ℝ := volume_of_box / rounded_number
  have edge_length_of_smaller_cube : ℝ := real.cbrt volume_of_smaller_cube
  have h3 : volume_of_box = 1 := by rw [h1]; norm_num
  have h4 : volume_of_smaller_cube = 0.001 := by rw [volume_of_box, h3]; norm_num
  have h5 : edge_length_of_smaller_cube = 0.1 := by norm_num [real.cbrt]
  use edge_length_of_smaller_cube
  exact h5

end edge_length_of_smaller_cubes_l620_620971


namespace total_elephants_l620_620195

-- Definitions and Hypotheses
def W : ℕ := 70
def G : ℕ := 3 * W

-- Proposition
theorem total_elephants : W + G = 280 := by
  sorry

end total_elephants_l620_620195


namespace locus_M_line_l_l620_620707

-- Definitions and conditions
def circle_O : set (ℝ × ℝ) :=
  {p | p.1 ^ 2 + p.2 ^ 2 = 4}

def foot_perpendicular (P D : ℝ × ℝ) : Prop :=
  D.2 = 0 ∧ P.2 ≠ D.2 ∧ P.1 = D.1

def midpoint (P D M : ℝ × ℝ) : Prop :=
  M.1 = P.1 ∧ M.2 = (P.2 + D.2) / 2

-- Problem 1: Locus of M
theorem locus_M {P D M : ℝ × ℝ} 
  (hP_circle : P ∈ circle_O)
  (hfoot : foot_perpendicular P D)
  (hmid : midpoint P D M) :
  M ∈ {p | (p.1 ^ 2 / 4) + p.2 ^ 2 = 1} :=
sorry

-- Problem 2: Equation of line l
theorem line_l {A B C D : ℝ × ℝ} 
  (h1 : A ∈ circle_O) (h2 : B ∈ circle_O) 
  (h3 : C ∈ {p | (p.1 ^ 2 / 4) + p.2 ^ 2 = 1}) 
  (h4 : D ∈ {p | (p.1 ^ 2 / 4) + p.2 ^ 2 = 1}) 
  (hAB : |(A.1 - B.1)| = 2 ∧ |(A.2 - B.2)| = 0)  -- simplified assuming intersection with x-axis
  (hCD_len : 2 = abs (C.2 - D.2)) -- simplified |CD| length, example
  (hAB_CD : abs(|(A.1 - B.1)| * abs(C.2 - D.2)) = 8 * sqrt 10 / 5) :
  (∃ k, ∀ x, y = k * (x - sqrt 3)) ∨ (∀ x, y = -k * (x + sqrt 3)) :=
sorry

end locus_M_line_l_l620_620707


namespace greatest_three_digit_multiple_of_17_l620_620325

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

theorem greatest_three_digit_multiple_of_17 : ∀ n, n < 1000 → is_multiple_of n 17 → n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620325


namespace element_not_subset_l620_620633

-- Define the propositions
def in_set (a : Type) (A : set Type) : Prop :=
  a ∈ A

def subset_of_set (a : Type) (A : set Type) : Prop :=
  a ⊆ A

-- The condition
variable {a : Type} {A : set Type}

-- The theorem we need to prove
theorem element_not_subset :
  in_set a A → ¬ subset_of_set a A :=
by 
  -- proof goes here
  sorry

end element_not_subset_l620_620633


namespace recurring_decimal_sum_l620_620551

-- Definitions based on the conditions identified
def recurringDecimal (n : ℕ) : ℚ := n / 9
def r8 := recurringDecimal 8
def r2 := recurringDecimal 2
def r6 := recurringDecimal 6
def r6_simplified : ℚ := 2 / 3

-- The theorem to prove
theorem recurring_decimal_sum : r8 + r2 - r6_simplified = 4 / 9 :=
by
  -- Proof steps will go here (but are omitted because of the problem requirements)
  sorry

end recurring_decimal_sum_l620_620551


namespace blankets_warm_nathan_up_l620_620405

theorem blankets_warm_nathan_up :
  (∀ (blankets_added half_blankets: ℕ), blankets_added = (half_blankets) -> half_blankets = 7 -> ∃ warmth: ℕ, warmth = blankets_added * 3  ∧ warmth = 21) :=
begin
  intros,
  sorry
end

end blankets_warm_nathan_up_l620_620405


namespace prove_c_eq_d_l620_620981

-- Define the given logarithmic relationships
def c := log 4 625
def d := log 5 25

-- State the theorem to prove c = d.
theorem prove_c_eq_d : c = d := by
  sorry

end prove_c_eq_d_l620_620981


namespace best_approx_sqrt3_l620_620903

-- Define the specific problem conditions.
def is_best_approximation (p q : ℤ) : Prop :=
  q ≤ 15 ∧ (∀ (a b : ℤ), b ≤ 15 → abs (sqrt 3 - p / q) < abs (sqrt 3 - a / b))

theorem best_approx_sqrt3 : ∃ (p q : ℤ), q ≤ 15 ∧ is_best_approximation 26 15 :=
by
  -- A witness to the theorem is given by (p, q) = (26, 15)
  use 26
  use 15
  -- Insert proof here
  sorry

end best_approx_sqrt3_l620_620903


namespace arrangement_count_l620_620215

theorem arrangement_count :
  let students := ['A', 'B', 'C', 'D', 'E']
  let teachers := ['X', 'Y']
  ∃ (arrangements : list (list char)),
  (∀ (g : list char), g ∈ arrangements → ('A'::'B'::'C'::[]) ⊆ g ∧
    ∀ t ∈ teachers, ¬((t::'X'::'Y'::[]) ⊆ g ∨ ('X'::'Y'::t::[]) ⊆ g)) ∧
  arrangements.length = 504 :=
by
  sorry

end arrangement_count_l620_620215


namespace fill_cistern_time_l620_620805

theorem fill_cistern_time (F E : ℝ) (hF : F = 1/2) (hE : E = 1/4) : 
  (1 / (F - E)) = 4 :=
by
  -- Definitions of F and E are used as hypotheses hF and hE
  -- Prove the actual theorem stating the time to fill the cistern is 4 hours
  sorry

end fill_cistern_time_l620_620805


namespace largest_number_is_option_b_l620_620800

-- Define each number
def option_a : ℝ := 9.12344
def option_b : ℝ := 9.123 + 0.4 / (10 - 1) -- 9.123 + 4 * 10^-3 repeating
def option_c : ℝ := 9.12 + 34 / (100 * (10^2 - 1)) -- 9.12 + 34 * 10^-2 repeating
def option_d : ℝ := 9.1 + 234 / (1000 * (10^3 - 1)) -- 9.1 + 234 * 10^-3 repeating
def option_e : ℝ := 9 + 1234 / (10000 * (10^4 - 1)) -- 9 + 1234 * 10^-4 repeating

-- Proof statement
theorem largest_number_is_option_b :
  option_b > option_a ∧ option_b > option_c ∧ option_b > option_d ∧ option_b > option_e :=
by
  sorry

end largest_number_is_option_b_l620_620800


namespace derivative_at_minus_one_l620_620605

def f (x : ℝ) (fp_minus1 : ℝ) : ℝ := x^2 - 2 * x * fp_minus1

theorem derivative_at_minus_one :
  ∀ fp_minus1 : ℝ, (f' : ℝ → ℝ) (x) = deriv (λ y, f y fp_minus1) x → f'(-1) = -2 - 2 * f'(-1) → f'(-1) = -2 / 3 :=
begin
  intros fp_minus1 f',
  sorry,
end

end derivative_at_minus_one_l620_620605


namespace exists_vectors_l620_620555

noncomputable def vectorProve (a b c : ℝ^3) : Prop :=
  a + b + c = 0 ∧
  ‖a + b - c‖ = 1 ∧
  ‖b + c - a‖ = 1 ∧
  ‖c + a - b‖ = 1

theorem exists_vectors : 
  ∃ (a b c : ℝ^3),
  vectorProve a b c := 
sorry

end exists_vectors_l620_620555


namespace greatest_three_digit_multiple_of_17_l620_620364

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 100 ≤ n ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l620_620364


namespace exactly_one_root_in_interval_l620_620201

theorem exactly_one_root_in_interval (p q : ℝ) (h : q * (q + p + 1) < 0) :
    ∃ x : ℝ, 0 < x ∧ x < 1 ∧ (x^2 + p * x + q = 0) := sorry

end exactly_one_root_in_interval_l620_620201


namespace ab_value_l620_620007

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_value_l620_620007


namespace age_difference_proof_l620_620417

theorem age_difference_proof (A B C : ℕ) (h1 : B = 12) (h2 : B = 2 * C) (h3 : A + B + C = 32) :
  A - B = 2 :=
by
  sorry

end age_difference_proof_l620_620417


namespace probability_of_winning_pair_is_51_over_91_l620_620445

def red_cards : ℕ := 7 -- 5 regular + 2 additional for vowels
def green_cards : ℕ := 7 -- 5 regular + 2 additional for vowels
def total_cards : ℕ := red_cards + green_cards -- 14 cards

def ways_draw_two_cards := nat.choose total_cards 2 -- Combination of 14 taken 2

def same_letter_ways : ℕ := (nat.choose 2 2) * 3 + (nat.choose 3 2) * 2 -- three non-vowels and two vowels
def same_color_ways : ℕ := 2 * (nat.choose red_cards 2) -- both colors are the same

def total_favorable_outcomes : ℕ := same_letter_ways + same_color_ways

def probability_winning_pair : ℚ := total_favorable_outcomes / ways_draw_two_cards

theorem probability_of_winning_pair_is_51_over_91 : probability_winning_pair = 51 / 91 :=
by
  sorry

end probability_of_winning_pair_is_51_over_91_l620_620445


namespace gambler_difference_eq_two_l620_620831

theorem gambler_difference_eq_two (x y : ℕ) (x_lost y_lost : ℕ) :
  20 * x + 100 * y = 3000 ∧
  x + y = 14 ∧
  20 * (14 - y_lost) + 100 * y_lost = 760 →
  (x_lost - y_lost = 2) := sorry

end gambler_difference_eq_two_l620_620831


namespace greatest_three_digit_multiple_of_17_l620_620393

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) := 
  exists.intro 986 sorry

end greatest_three_digit_multiple_of_17_l620_620393


namespace greatest_three_digit_multiple_of_17_l620_620351

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (n % 17 = 0) ∧ ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ (m % 17 = 0) → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620351


namespace smallest_domain_of_g_l620_620535

def g (x : ℤ) : ℤ :=
  if x % 2 = 0 then x / 2 else 3 * x + 2

theorem smallest_domain_of_g : ∃ n : ℕ, n = 9 ∧ ∀ x ∈ {9, 28, 14, 7, 23, 71, 215, 647, 1943}, g x ∈ {28, 14, 7, 23, 71, 215, 647, 1943, 5829} :=
begin
  sorry
end

end smallest_domain_of_g_l620_620535


namespace triangles_isosceles_l620_620844

-- Define the initial conditions
variable (A B C D : Point)
variable (c d g : Line)
variable (proj_plane1 : Plane)
variable (parallel_to_proj_plane1 : g.is_parallel_to proj_plane1)
variable (intersect_pts : (A, B) ∈ c ∧ (C, D) ∈ d)

-- Given condition
variable (AB_eq_CD : AB.length = CD.length)

-- Goal: Prove that the triangles EFM and GHN are isosceles
theorem triangles_isosceles
  (EFM_isosceles : Triangle E F M.is_isosceles)
  (GHN_isosceles : Triangle G H N.is_isosceles) 
  (proj_planes_intersect : proj_plane1.intersects (Plane_through_lines c d)) :
  ∃ E F M G H N : Point, 
  (A B C D ∈ g ∧ parallel_to_proj_plane1) →
  (EFM_isosceles ∧ GHN_isosceles) := sorry

end triangles_isosceles_l620_620844


namespace domain_log_base_2_l620_620182

theorem domain_log_base_2 (x : ℝ) : (1 - x > 0) ↔ (x < 1) := by
  sorry

end domain_log_base_2_l620_620182


namespace greatest_three_digit_multiple_of_17_l620_620270

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (17 ∣ n) ∧ ∀ m : ℤ, (100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m) → m ≤ n :=
begin
  use 969,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num, exact dec_trivial, },
  { intros m h_mconds,
    cases h_mconds with hm1 h_mconds,
    cases h_mconds with hm2 h_formula,
    rw h_formula, sorry, }  -- Completing this part is not necessary according to the guidelines
end

end greatest_three_digit_multiple_of_17_l620_620270


namespace probability_of_consonant_initials_is_10_over_13_l620_620994

def is_vowel (c : Char) : Prop :=
  c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U' ∨ c = 'Y'

def is_consonant (c : Char) : Prop :=
  ¬(is_vowel c) ∧ c ≠ 'W' 

noncomputable def probability_of_consonant_initials : ℚ :=
  let total_letters := 26
  let number_of_vowels := 6
  let number_of_consonants := total_letters - number_of_vowels
  number_of_consonants / total_letters

theorem probability_of_consonant_initials_is_10_over_13 :
  probability_of_consonant_initials = 10 / 13 :=
by
  sorry

end probability_of_consonant_initials_is_10_over_13_l620_620994


namespace greatest_three_digit_multiple_of_17_l620_620267

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (17 ∣ n) ∧ ∀ m : ℤ, (100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m) → m ≤ n :=
begin
  use 969,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num, exact dec_trivial, },
  { intros m h_mconds,
    cases h_mconds with hm1 h_mconds,
    cases h_mconds with hm2 h_formula,
    rw h_formula, sorry, }  -- Completing this part is not necessary according to the guidelines
end

end greatest_three_digit_multiple_of_17_l620_620267


namespace trucks_in_yard_l620_620998

/-- The number of trucks in the yard is 23, given the conditions. -/
theorem trucks_in_yard (T : ℕ) (H1 : ∃ n : ℕ, n > 0)
  (H2 : ∃ k : ℕ, k = 5 * T)
  (H3 : T + 5 * T = 140) : T = 23 :=
sorry

end trucks_in_yard_l620_620998


namespace greatest_3_digit_multiple_of_17_l620_620378

theorem greatest_3_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≥ m) ∧ n = 986 := 
begin
  sorry
end

end greatest_3_digit_multiple_of_17_l620_620378


namespace right_triangle_AB_l620_620090

theorem right_triangle_AB 
  {A B C : Type}
  [EuclideanGeometry A B C]
  (AC BC : ℝ)
  (h1 : angle A C = 90)
  (h2 : AC = 6)
  (h3 : tan(A) = 3/2) :
  AB = 3 * sqrt 13 := 
sorry

end right_triangle_AB_l620_620090


namespace vertices_of_trapezoid_l620_620120

-- Definitions and theorems used in Lean 4 equivalent to the math problem conditions.
variables {A B C I P Q : Type*} [Incircle I A B C] [Circumcircle A I B P] [Circumcircle A I B Q]

theorem vertices_of_trapezoid (h1 : Incircle I A B C)
  (h2 : Circumcircle A I B P)
  (h3 : Circumcircle A I B Q)
  (hP : P ≠ A)
  (hQ : Q ≠ B) :
  is_trapezoid {A, B, P, Q} :=
sorry

end vertices_of_trapezoid_l620_620120


namespace digit_123_of_25_div_700_l620_620792

noncomputable def repeating_sequence : ℕ → ℕ :=
  λ n => "035714".to_list.nth! (n % 6).

theorem digit_123_of_25_div_700 :
  let d := repeating_sequence 123 in
  d = 5 := by
  sorry

end digit_123_of_25_div_700_l620_620792


namespace fixed_points_on_circle_l620_620138

theorem fixed_points_on_circle (m : ℝ) : 
  (√5, 2 * √5) ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 + 2 * m * p.1 - m * p.2 - 25 = 0} ∧ 
  (-√5, -2 * √5) ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 + 2 * m * p.1 - m * p.2 - 25 = 0} :=
sorry

end fixed_points_on_circle_l620_620138


namespace greatest_three_digit_multiple_of_17_l620_620271

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (17 ∣ n) ∧ ∀ m : ℤ, (100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m) → m ≤ n :=
begin
  use 969,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num, exact dec_trivial, },
  { intros m h_mconds,
    cases h_mconds with hm1 h_mconds,
    cases h_mconds with hm2 h_formula,
    rw h_formula, sorry, }  -- Completing this part is not necessary according to the guidelines
end

end greatest_three_digit_multiple_of_17_l620_620271


namespace greatest_three_digit_multiple_of_17_l620_620335

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

theorem greatest_three_digit_multiple_of_17 : ∀ n, n < 1000 → is_multiple_of n 17 → n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620335


namespace max_students_seated_l620_620849

theorem max_students_seated :
  let seats_per_row : ℕ → ℕ := λ i, 11 + i
  let students_per_row : ℕ → ℕ := λ i, (seats_per_row i + 1) / 2
  (∑ i in Finset.range 25, students_per_row (i+1)) = 192 :=
by
  let seats_per_row : ℕ → ℕ := λ i, 11 + i
  let students_per_row : ℕ → ℕ := λ i, (seats_per_row i + 1) / 2
  calc
    ∑ i in Finset.range 25, students_per_row (i+1)
    = 192 : sorry

end max_students_seated_l620_620849


namespace angle_CED_l620_620786

-- Define the geometric constructs with the given conditions
theorem angle_CED {A B C D E : Type*} [metric_space A] [metric_space B]
  (r_A r_B : ℝ) (h₀ : r_A ≠ r_B) 
  (h₁ : dist A B = r_A + r_B)
  (h₂ : dist A E = r_B) (h₃ : dist B E = r_A) 
  (h₄ : dist A C = r_A) (h₅ : dist B D = r_B)
  (h6 : E ∈ (sphere A r_A : set A)) (h7 : E ∈ (sphere B r_B : set B)) 
  : ∠ C E D = 90 :=
sorry

end angle_CED_l620_620786


namespace sufficient_food_supply_l620_620500

variable {L S : ℝ}

theorem sufficient_food_supply (h1 : L + 4 * S = 14) (h2 : L > S) : L + 3 * S ≥ 11 :=
by
  sorry

end sufficient_food_supply_l620_620500


namespace find_percentage_l620_620820

theorem find_percentage (P : ℝ) (h1 : (P / 100) * 200 = 30 + 0.60 * 50) : P = 30 :=
by
  sorry

end find_percentage_l620_620820


namespace minimum_researchers_needed_l620_620816

theorem minimum_researchers_needed
  (G M B : ℕ)
  (hG : G = 120)
  (hM : M = 90)
  (hB : B = 40) :
  ∃ (R : ℕ), R = G + M - B ∧ R = 170 :=
by
  use G + M - B
  simp [hG, hM, hB]
  sorry

end minimum_researchers_needed_l620_620816


namespace greatest_three_digit_multiple_of_17_l620_620391

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) := 
  exists.intro 986 sorry

end greatest_three_digit_multiple_of_17_l620_620391


namespace greatest_3_digit_multiple_of_17_l620_620388

theorem greatest_3_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≥ m) ∧ n = 986 := 
begin
  sorry
end

end greatest_3_digit_multiple_of_17_l620_620388


namespace greatest_three_digit_multiple_of_17_l620_620372

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 100 ≤ n ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l620_620372


namespace approximate_fish_number_l620_620641

theorem approximate_fish_number (N : ℕ) (tagged_initial : ℕ) (catch_size : ℕ) (tagged_catch : ℕ) 
  (h : (tagged_catch : ℚ) / catch_size = (tagged_initial : ℚ) / N) : 
  N = 1750 := 
by
  have h1 : 2 / 50 = 70 / N := h
  have h2 : 2 * N = 50 * 70
  have h3 : 2 * N = 3500 := by rw [mul_comm, h2]
  exact eq_of_mul_eq_mul_left (by norm_num) h3

end approximate_fish_number_l620_620641


namespace problem_part1_problem_part2_l620_620870

theorem problem_part1 : (sqrt 2) / (sqrt (1 / 2)) * (sqrt 8) - (sqrt 8) = 2 * (sqrt 2) := sorry

theorem problem_part2 : (sqrt 48) - (sqrt 27) + (-3 * (sqrt 2))^2 - (3 / (sqrt 3)) = 18 := sorry

end problem_part1_problem_part2_l620_620870


namespace total_years_eq_25_l620_620654

-- Definitions based on conditions
def women_percentage : ℝ := 0.60
def men_percentage : ℝ := 0.40 -- Since 100% - 60% = 40%
def beard_percentage : ℝ := 0.40
def men_with_beards : ℕ := 4

-- Theorem to prove:
theorem total_years_eq_25 : ∃ Y : ℝ, Y * (men_percentage * beard_percentage) = men_with_beards ∧ Y = 25 :=
by
  use 25
  split
  sorry
  sorry

end total_years_eq_25_l620_620654


namespace angle_same_after_minutes_l620_620973

def angle_between_hands (H M : ℝ) : ℝ :=
  abs (30 * H - 5.5 * M)

theorem angle_same_after_minutes (x : ℝ) :
  x = 54 + 6 / 11 → 
  angle_between_hands (5 + (x / 60)) x = 150 :=
by
  sorry

end angle_same_after_minutes_l620_620973


namespace incorrect_statement_c_l620_620432

-- Define even function
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x

-- Define odd function
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

-- Function definitions
def f1 (x : ℝ) : ℝ := x^4 + x^2
def f2 (x : ℝ) : ℝ := x^3 + x^2

-- Main theorem statement
theorem incorrect_statement_c : ¬ is_odd f2 := sorry

end incorrect_statement_c_l620_620432


namespace cat_food_sufficiency_l620_620518

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S >= 11 :=
sorry

end cat_food_sufficiency_l620_620518


namespace greatest_three_digit_multiple_of_17_l620_620260

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (17 ∣ n) ∧ ∀ m : ℤ, (100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m) → m ≤ n :=
begin
  use 969,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num, exact dec_trivial, },
  { intros m h_mconds,
    cases h_mconds with hm1 h_mconds,
    cases h_mconds with hm2 h_formula,
    rw h_formula, sorry, }  -- Completing this part is not necessary according to the guidelines
end

end greatest_three_digit_multiple_of_17_l620_620260


namespace bubbleSort_iter_count_l620_620603

/-- Bubble sort iterates over the list repeatedly, swapping adjacent elements if they are in the wrong order. -/
def bubbleSortSteps (lst : List Int) : List (List Int) :=
sorry -- Implementation of bubble sort to capture each state after each iteration

/-- Prove that sorting [6, -3, 0, 15] in descending order using bubble sort requires exactly 3 iterations. -/
theorem bubbleSort_iter_count : 
  (bubbleSortSteps [6, -3, 0, 15]).length = 3 :=
sorry

end bubbleSort_iter_count_l620_620603


namespace c_completion_days_l620_620422

noncomputable def work_rate (days: ℕ) := (1 : ℝ) / days

theorem c_completion_days : 
  ∀ (W : ℝ) (Ra Rb Rc : ℝ) (Dc : ℕ),
  Ra = work_rate 30 → Rb = work_rate 30 → Rc = work_rate Dc →
  (Ra + Rb + Rc) * 8 + (Ra + Rb) * 4 = W → 
  Dc = 40 :=
by
  intros W Ra Rb Rc Dc hRa hRb hRc hW
  sorry

end c_completion_days_l620_620422


namespace regular_hexagonal_pyramid_surface_area_l620_620593

noncomputable def surface_area_of_pyramid (base_edge side_edge : ℝ) : ℝ :=
  let h := Real.sqrt (side_edge ^ 2 - 1 ^ 2) in
  6 * Real.sqrt 3 / 4 * base_edge ^ 2 + 6 * 1 / 2 * base_edge * h

theorem regular_hexagonal_pyramid_surface_area :
  surface_area_of_pyramid 2 (Real.sqrt 5) = 6 * Real.sqrt 3 + 12 :=
by
  sorry

end regular_hexagonal_pyramid_surface_area_l620_620593


namespace volume_of_cylindrical_block_l620_620642

variable (h_cylindrical : ℕ) (combined_value : ℝ)

theorem volume_of_cylindrical_block (h_cylindrical : ℕ) (combined_value : ℝ):
  h_cylindrical = 3 → combined_value / 5 * h_cylindrical = 15.42 := by
suffices combined_value / 5 = 5.14 from sorry
suffices 5.14 * 3 = 15.42 from sorry
suffices h_cylindrical = 3 from sorry
suffices 25.7 = combined_value from sorry
sorry

end volume_of_cylindrical_block_l620_620642


namespace age_difference_proof_l620_620418

theorem age_difference_proof (A B C : ℕ) (h1 : B = 12) (h2 : B = 2 * C) (h3 : A + B + C = 32) :
  A - B = 2 :=
by
  sorry

end age_difference_proof_l620_620418


namespace total_accidents_l620_620103

noncomputable def A (k x : ℕ) : ℕ := 96 + k * x

theorem total_accidents :
  let k_morning := 1
  let k_evening := 3
  let x_morning := 2000
  let x_evening := 1000
  A k_morning x_morning + A k_evening x_evening = 5192 := by
  sorry

end total_accidents_l620_620103


namespace greatest_three_digit_multiple_of_17_l620_620240

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n, (n ∈ set.Icc 100 999) ∧ (n % 17 = 0) ∧ (∀ m, (m ∈ set.Icc 100 999 ∧ m % 17 = 0) → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620240


namespace probability_of_A_selected_l620_620565

theorem probability_of_A_selected :
  let S : Finset String := { "A", "B", "C", "D", "E" }
  let selected : Finset (Finset String) := (Finset.powersetLen 3 S)
  let selected_with_A : Finset (Finset String) := selected.filter (fun s => "A" ∈ s)
  selected_with_A.card / selected.card = (3 : ℚ) / (5 : ℚ) :=
by
  sorry

end probability_of_A_selected_l620_620565


namespace largest_three_digit_multiple_of_17_l620_620280

theorem largest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n ≤ 999) ∧ (n ≥ 100) ∧ (17 ∣ n) ∧ ∀ m, ((m > n) ∧ (m ≤ 999) ∧ (m ≥ 100) ∧ (17 ∣ m)) → false :=
by {
  sorry
}

end largest_three_digit_multiple_of_17_l620_620280


namespace OH_perp_MN_l620_620097

open Real EuclideanGeometry

theorem OH_perp_MN {A B C O H D E F M N : Point} 
(h_triangle : Triangle A B C)
(h_circumcenter : IsCircumcenter O A B C)
(h_altitudes : IsOrthocenter H A B C D E F)
(h_intersect_ED_AB : Intersects (LineThrough E D) (Segment AB) M)
(h_intersect_FD_AC : Intersects (LineThrough F D) (Segment AC) N) :
Perpendicular (LineThrough O H) (LineThrough M N) := 
sorry

end OH_perp_MN_l620_620097


namespace polynomial_sum_of_coefficients_l620_620128

theorem polynomial_sum_of_coefficients {v : ℕ → ℝ} (h1 : v 1 = 7)
  (h2 : ∀ n : ℕ, v (n + 1) - v n = 5 * n - 2) :
  ∃ (a b c : ℝ), (∀ n : ℕ, v n = a * n^2 + b * n + c) ∧ (a + b + c = 7) :=
by
  sorry

end polynomial_sum_of_coefficients_l620_620128


namespace sum_of_equal_numbers_is_555_l620_620822

-- Define the chessboard dimensions
def Rows : ℕ := 13
def Columns : ℕ := 17

-- Define the row-wise numbering function
def rowNumber (x y : ℕ) : ℕ :=
  if 1 ≤ x ∧ x ≤ Rows ∧ 1 ≤ y ∧ y ≤ Columns then
    (x - 1) * Columns + y
  else 0  -- out of bounds

-- Define the column-wise numbering function
def colNumber (x y : ℕ) : ℕ :=
  if 1 ≤ x ∧ x ≤ Rows ∧ 1 ≤ y ∧ y ≤ Columns then
    (y - 1) * Rows + x
  else 0  -- out of bounds

-- Define a predicate that checks if a number is the same in both numbering systems
def isSameNumber (x y : ℕ) : Prop :=
  rowNumber x y = colNumber x y

-- The sum of numbers that are the same in both grid systems
noncomputable def sumEqualNumbers : ℕ :=
  (List.range Rows).bind (λ x =>
    (List.range Columns).filterMap (λ y =>
      if isSameNumber (x+1) (y+1) then 
        some (rowNumber (x+1) (y+1))
      else 
        none
    )
  ).sum

-- The final theorem stating the sum of these numbers is 555
theorem sum_of_equal_numbers_is_555 : sumEqualNumbers = 555 :=
  by
    sorry

end sum_of_equal_numbers_is_555_l620_620822


namespace slope_of_line_AF_parabola_l620_620177

theorem slope_of_line_AF_parabola (A : ℝ × ℝ)
  (hA_on_parabola : A.snd ^ 2 = 4 * A.fst)
  (h_dist_focus : Real.sqrt ((A.fst - 1) ^ 2 + A.snd ^ 2) = 4) :
  (A.snd / (A.fst - 1) = Real.sqrt 3 ∨ A.snd / (A.fst - 1) = -Real.sqrt 3) :=
sorry

end slope_of_line_AF_parabola_l620_620177


namespace right_triangle_points_inequality_l620_620084

theorem right_triangle_points_inequality
  (A B C : ℝ × ℝ) (P : ℕ → ℝ × ℝ)
  (h_right_triangle : ∠ B A C = 90) 
  (n : ℕ) 
  (h_points_inside_triangle : ∀ i, P i ∈ triangle ABC) :
  ∃ (f : fin n → ℕ),
  ∑ i in finset.range (n - 1), dist2 (P (f i)) (P (f (i + 1))) ≤ dist2 A B := 
begin
  sorry
end

end right_triangle_points_inequality_l620_620084


namespace sum_of_two_numbers_l620_620761

theorem sum_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) (h3 : x * y = 200) : (x + y = 30) :=
by sorry

end sum_of_two_numbers_l620_620761


namespace cat_food_sufficiency_l620_620507

theorem cat_food_sufficiency (L S : ℝ) (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end cat_food_sufficiency_l620_620507


namespace largest_power_of_2_divisor_l620_620556

noncomputable def a := 17
noncomputable def b := 13
noncomputable def n := a^4 - b^4

theorem largest_power_of_2_divisor :
  ∃ k : ℕ, 2^k ∣ n ∧ ∀ m : ℕ, 2^m ∣ n → m ≤ k := 
begin
  use 4,
  split,
  { sorry }, -- Here we need a proof that 2^4 divides n
  { intros m hm,
    sorry }  -- Here we need to show that any power of 2 that divides n is at most 4
end

end largest_power_of_2_divisor_l620_620556


namespace greatest_three_digit_multiple_of_17_l620_620348

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620348


namespace monotonicity_m_eq_zero_range_of_m_l620_620955

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := Real.exp x - m * x^2 - 2 * x

theorem monotonicity_m_eq_zero :
  ∀ x : ℝ, (x < Real.log 2 → f x 0 < f (x + 1) 0) ∧ (x > Real.log 2 → f x 0 > f (x - 1) 0) := 
sorry

theorem range_of_m :
  ∀ x : ℝ, x ∈ Set.Ici 0 → f x m > (Real.exp 1 / 2 - 1) → m < (Real.exp 1 / 2 - 1) := 
sorry

end monotonicity_m_eq_zero_range_of_m_l620_620955


namespace greatest_three_digit_multiple_of_17_l620_620343

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620343


namespace not_tangent_isosceles_right_eq_l620_620573

open Real

def circle (x y: ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 8

def line (m x y: ℝ) : Prop := m * x - y + 2 - m = 0

theorem not_tangent (m : ℝ) : ¬ ∃ (x y : ℝ), circle x y ∧ line m x y ∧ 
  ∀ z w, circle z w ∧ line m z w → (x, y) = (z, w) :=
sorry

theorem isosceles_right_eq (m : ℝ) (A B : ℝ × ℝ) : 
  circle A.fst A.snd → circle B.fst B.snd → line m A.fst A.snd → line m B.fst B.snd →
  ∃ C : ℝ × ℝ, (C.snd = 2) ∧ (C.fst = 1) →
  ((A.fst, A.snd) ≠ (B.fst, B.snd)) ∧ ((|A.fst - B.fst| = 4 / sqrt 2) 
  → ((|A.snd - B.snd| = 4 / sqrt 2) → m = - 3 / 4))  :=
sorry

end not_tangent_isosceles_right_eq_l620_620573


namespace relationship_among_a_b_c_l620_620447

variable (f : ℝ → ℝ)
variable [Differentiable ℝ f]

theorem relationship_among_a_b_c 
  (h1 : ∀ x ∈ set.Ioo 1 (real.top), (x - 1) * deriv f x - f x > 0) :
  let a := f 2,
      b := (1/2) * f 3,
      c := 1 / (real.sqrt 2 - 1) * f (real.sqrt 2)
  in c < a ∧ a < b := 
by
  sorry

end relationship_among_a_b_c_l620_620447


namespace max_value_xy_l620_620073

theorem max_value_xy (x y : ℝ) (h : x^2 + 4 * y^2 = 1) : xy = 1 / 4 :=
begin
  sorry,
end

end max_value_xy_l620_620073


namespace inequality_holds_for_all_x_l620_620910

theorem inequality_holds_for_all_x (x : ℝ) : 3 * x^2 + 9 * x ≥ -12 := by
  sorry

end inequality_holds_for_all_x_l620_620910


namespace ab_equals_six_l620_620045

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end ab_equals_six_l620_620045


namespace find_rate_of_interest_l620_620440

-- Conditions
def principal : ℕ := 4200
def time : ℕ := 2
def interest_12 : ℕ := principal * 12 * time / 100
def additional_interest : ℕ := 504
def total_interest_r : ℕ := interest_12 + additional_interest

-- Theorem Statement
theorem find_rate_of_interest (r : ℕ) (h : 1512 = principal * r * time / 100) : r = 18 :=
by sorry

end find_rate_of_interest_l620_620440


namespace roots_of_polynomial_l620_620539

theorem roots_of_polynomial : 
  (∀ x : ℝ, (x^3 - 6*x^2 + 11*x - 6) * (x - 2) = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3) :=
by
  intro x
  sorry

end roots_of_polynomial_l620_620539


namespace response_rate_percentage_l620_620439

theorem response_rate_percentage (responses_needed : ℕ) (num_questionnaires : ℕ) (h_resp : responses_needed = 210) (h_quest : num_questionnaires = 350) : (responses_needed * 100) / num_questionnaires = 60 :=
by
  rw [h_resp, h_quest]
  norm_num
  sorry

end response_rate_percentage_l620_620439


namespace greatest_three_digit_multiple_of_17_l620_620308

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 17 ∣ n :=
by 
  sorry

end greatest_three_digit_multiple_of_17_l620_620308


namespace largest_three_digit_multiple_of_17_l620_620277

theorem largest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n ≤ 999) ∧ (n ≥ 100) ∧ (17 ∣ n) ∧ ∀ m, ((m > n) ∧ (m ≤ 999) ∧ (m ≥ 100) ∧ (17 ∣ m)) → false :=
by {
  sorry
}

end largest_three_digit_multiple_of_17_l620_620277


namespace problem_solution_l620_620681

theorem problem_solution (f : ℝ → ℝ)
    (h : ∀ x y : ℝ, f ((x - y) ^ 2) = f x ^ 2 - 2 * x * f y + y ^ 2) :
    ∃ n s : ℕ, 
    (n = 2) ∧ 
    (s = 3) ∧
    (n * s = 6) :=
sorry

end problem_solution_l620_620681


namespace greatest_three_digit_multiple_of_17_l620_620265

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (17 ∣ n) ∧ ∀ m : ℤ, (100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m) → m ≤ n :=
begin
  use 969,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num, exact dec_trivial, },
  { intros m h_mconds,
    cases h_mconds with hm1 h_mconds,
    cases h_mconds with hm2 h_formula,
    rw h_formula, sorry, }  -- Completing this part is not necessary according to the guidelines
end

end greatest_three_digit_multiple_of_17_l620_620265


namespace period_of_cos_transformed_l620_620403

theorem period_of_cos_transformed (x : ℝ) :
  ∃ T, (∀ x, cos (3 * x / 2) = cos (3 * (x + T) / 2)) ∧ T = (4 * Real.pi) / 3 :=
sorry

end period_of_cos_transformed_l620_620403


namespace flour_masses_l620_620909

theorem flour_masses (x : ℝ) (h: 
    (x * (1 + x / 100) + (x + 10) * (1 + (x + 10) / 100) = 112.5)) :
    x = 35 ∧ (x + 10) = 45 :=
by 
  sorry

end flour_masses_l620_620909


namespace area_of_triangle_l620_620590

theorem area_of_triangle (m : ℝ) 
  (h : ∀ x y : ℝ, ((m + 3) * x + y = 3 * m - 4) → 
                  (7 * x + (5 - m) * y - 8 ≠ 0)
  ) : ((m = -2) → (1/2) * 2 * 2 = 2) := 
by {
  sorry
}

end area_of_triangle_l620_620590


namespace company_fund_initial_amount_l620_620190

theorem company_fund_initial_amount
  (n : ℕ)
  (h1 : 45 * n + 135 = 60 * n - 15)
  (h2 : ∑ i in finset.range n, 45 = 450) :
  60 * n - 15 = 585 := by
  sorry

end company_fund_initial_amount_l620_620190


namespace ab_value_l620_620026

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_value_l620_620026


namespace find_length_JK_and_angle_JKL_l620_620616

namespace SimilarTriangles

variables (G H I J K L : Type) [Fintype G] [Fintype H] [Fintype I] [Fintype J] [Fintype K] [Fintype L]
variables (GH HI JK KL : ℝ) (angle_GHI angle_JKL : ℝ)

def triangles_similar (triangle1 triangle2 : Type) : Prop := sorry -- Define similarity of triangles

noncomputable def length_GH : ℝ := 8
noncomputable def length_HI : ℝ := 16
noncomputable def length_KL : ℝ := 24
noncomputable def angle_GHI_value : ℝ := 30

axiom similarity_condition_1 : triangles_similar (triangle G H I) (triangle J K L)
axiom side_length_1 : GH = length_GH
axiom side_length_2 : HI = length_HI
axiom side_length_3 : KL = length_KL
axiom angle_condition : angle_GHI = angle_GHI_value

theorem find_length_JK_and_angle_JKL : 
  ∃ JK_value angle_JKL_value, 
    JK_value = 12 ∧ angle_JKL_value = 30 ∧
    ∀ (JK : ℝ) (angle_JKL : ℝ), 
      triangles_similar (triangle G H I) (triangle J K L) ∧
      GH = 8 ∧ HI = 16 ∧ KL = 24 ∧ angle_GHI = 30 → 
      JK = 12 ∧ angle_JKL = 30 := 
by {
  sorry
}

end SimilarTriangles

end find_length_JK_and_angle_JKL_l620_620616


namespace greatest_three_digit_multiple_of_17_l620_620316

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → m ≤ n) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620316


namespace compare_abc_l620_620631

noncomputable def a : ℝ := 3 ^ 0.3
noncomputable def b : ℝ := Real.log 3 / Real.log π  -- log base π of 3
noncomputable def c : ℝ := Real.log Real.e / Real.log 0.3 -- log base 0.3 of e

theorem compare_abc : a > b ∧ b > c := by
  have ha : a > 1 := sorry
  have hb : 0 < b ∧ b < 1 := sorry
  have hc : c < 0 := sorry
  exact ⟨ha, hb.2, hc⟩

end compare_abc_l620_620631


namespace ab_value_l620_620006

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_value_l620_620006


namespace divisibility_by_3_divisibility_by_9_divisibility_by_11_l620_620192

def digits_sum3 (a : List ℕ) : ℕ := a.foldl (· + ·) 0
def digits_sum9 (a : List ℕ) : ℕ := a.foldl (· + ·) 0
def digits_sum11 (a : List ℕ) (n : ℕ) : ℕ := a.foldr (λ b acc i => ((-1) ^ i) * b + acc) 0 (n-1)

theorem divisibility_by_3 (a : List ℕ) (n : ℕ) (A : List.range n) :
  (foldr (· + (10^(·)) * A) 0 a) % 3 = 0 ↔ (digits_sum3 a) % 3 = 0 :=
sorry

theorem divisibility_by_9 (a : List ℕ) (n : ℕ) (A : List.range n) :
  (foldr (· + (10^(·)) * A) 0 a) % 9 = 0 ↔ (digits_sum9 a) % 9 = 0 :=
sorry

theorem divisibility_by_11 (a : List ℕ) (n : ℕ) (A : List.range n) :
  (foldr (· + (10^(·)) * A) 0 a) % 11 = 0 ↔ (digits_sum11 a n) % 11 = 0 :=
sorry

end divisibility_by_3_divisibility_by_9_divisibility_by_11_l620_620192


namespace find_amplitude_period_angle_find_xs_f_of_1_l620_620950

noncomputable def sin_function (x : ℝ) (A ω φ : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem find_amplitude_period_angle :
  ∃ (A ω φ : ℝ),
    A > 0 ∧ ω > 0 ∧ |φ| < Real.pi ∧
    (∀ x, sin_function x A ω φ = 2 → (x = (Real.pi / 8) ∨ x = (5 * Real.pi / 8))) →
    sin_function x 2 2 (Real.pi / 4)
:=
sorry

theorem find_xs_f_of_1 :
  ∃ (A : ℝ) (ω : ℝ) (φ : ℝ),
    A = 2 ∧ ω = 2 ∧ φ = (Real.pi / 4) ∧
    {x : ℝ | sin_function x A ω φ = 1} = 
    {x : ℝ | ∃ (k : ℤ), x = - (Real.pi / 24) + k * Real.pi ∨ x = (7 * Real.pi / 24) + k * Real.pi}
:=
sorry

end find_amplitude_period_angle_find_xs_f_of_1_l620_620950


namespace find_first_number_l620_620762

-- Define the equation along with the constants
def equation (x : ℝ) : Prop :=
  x * 0.48 * 2.50 / (0.12 * 0.09 * 0.5) = 800.0000000000001

-- State the theorem that needs to be proved
theorem find_first_number : ∃ (x : ℝ), equation x ∧ x ≈ 3.6 :=
sorry

end find_first_number_l620_620762


namespace hot_dogs_remainder_l620_620980

theorem hot_dogs_remainder :
  let n := 16789537
  let d := 5
  n % d = 2 :=
by
  sorry

end hot_dogs_remainder_l620_620980


namespace ab_value_l620_620030

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_value_l620_620030


namespace number_of_boxes_correct_l620_620843

theorem number_of_boxes_correct : 
  ∀ (total_lids already_had per_box : ℕ), 
  total_lids = 53 → 
  already_had = 14 → 
  per_box = 13 → 
  (total_lids - already_had) / per_box = 3 :=
by 
  intros total_lids already_had per_box ht ha hp
  rw [ht, ha, hp]
  exact rfl

end number_of_boxes_correct_l620_620843


namespace intersection_l620_620104

noncomputable def f (x : ℝ) : ℝ := (x^2 - 8 * x + 12) / (2 * x - 6)

noncomputable def g (x : ℝ) (a b c d k : ℝ) : ℝ := -2 * x - 4 + k / (x - d)

theorem intersection (a b c k : ℝ) (h_d : d = 3) (h_k : k = 36) : 
  ∃ (x y : ℝ), x ≠ -3 ∧ (f x = g x 0 0 0 d k) ∧ (x, y) = (6.8, -32 / 19) :=
by
  sorry

end intersection_l620_620104


namespace number_of_plants_in_first_garden_l620_620150

/-
Phyllis has two gardens. In the first garden, she has some plants and 10% are tomato plants. 
In the second garden, she has 15 plants and 1/3 of these plants are tomato plants. 
20% of all the plants in her two gardens are tomato plants. 

Prove that the number of plants in the first garden is 20.
-/
theorem number_of_plants_in_first_garden (P : ℕ) 
  (h1 : 0.10 * P = 0.2 * (P + 15) - 5)
  (h2 : 15 * 1/3 = 5) : P = 20 :=
by
  -- Proof omitted
  sorry

end number_of_plants_in_first_garden_l620_620150


namespace distribution_correct_l620_620159

variable (total_pies : ℕ := 13)
variable (samples_per_pie : ℕ := 10)
variable (fraction_A : ℚ := 3 / 8)
variable (fraction_B_full_pies : ℕ := 2)
variable (fraction_B_partial_pie : ℚ := 6 / 10)
variable (fraction_C : ℚ := 5 / 7)

noncomputable def total_samples := total_pies * samples_per_pie

noncomputable def samples_A := (fraction_A * samples_per_pie).toNat
noncomputable def samples_B := (fraction_B_full_pies * samples_per_pie + (fraction_B_partial_pie * samples_per_pie)).toNat
noncomputable def samples_C := (fraction_C * samples_per_pie).toNat
noncomputable def samples_distributed := samples_A + samples_B + samples_C
noncomputable def samples_remaining := total_samples - samples_distributed

theorem distribution_correct :
  samples_A = 3 ∧
  samples_B = 26 ∧
  samples_C = 7 ∧
  samples_remaining = 94 ∧
  (samples_A + samples_B + samples_C + samples_remaining = total_samples) := by
  sorry

end distribution_correct_l620_620159


namespace greatest_three_digit_multiple_of_17_l620_620257

def is_three_digit (n : Int) : Prop := (n ≥ 100 ∧ n ≤ 999)
def is_multiple_of_17 (n : Int) : Prop := (∃ k : Int, n = 17 * k)

theorem greatest_three_digit_multiple_of_17 : ∀ n : Int, is_three_digit n ∧ is_multiple_of_17 n → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l620_620257


namespace greatest_three_digit_multiple_of_17_l620_620354

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (n % 17 = 0) ∧ ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ (m % 17 = 0) → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620354


namespace spilled_bag_candies_l620_620478

theorem spilled_bag_candies (c1 c2 c3 c4 c5 c6 c7 : ℕ) (avg_candies_per_bag : ℕ) (x : ℕ) 
  (h_counts : c1 = 12 ∧ c2 = 14 ∧ c3 = 18 ∧ c4 = 22 ∧ c5 = 24 ∧ c6 = 26 ∧ c7 = 29)
  (h_avg : avg_candies_per_bag = 22)
  (h_total : c1 + c2 + c3 + c4 + c5 + c6 + c7 + x = 8 * avg_candies_per_bag) : x = 31 := 
by
  sorry

end spilled_bag_candies_l620_620478


namespace tom_cars_left_l620_620223

open Nat

theorem tom_cars_left (packages cars_per_package : ℕ) (fraction_given : ℚ) :
  packages = 10 →
  cars_per_package = 5 →
  fraction_given = 1 / 5 →
  2 * (fraction_given * (packages * cars_per_package)) ≤ packages * cars_per_package →
  (packages * cars_per_package) - 2 * (fraction_given * (packages * cars_per_package)).toNat = 30 :=
by
  intros h_packages h_cars_per_package h_fraction_given h_le
  sorry

end tom_cars_left_l620_620223


namespace find_magnitude_l620_620963

noncomputable def magnitude {α : Type*} [inner_product_space ℝ α] (v : α) : ℝ :=
  real.sqrt (inner_product_space.norm_sq v)

variables (a b : EuclideanSpace ℝ (Fin 2))
variables (angle : ℝ) (magnitude_a magnitude_b : ℝ)

hypothesis h_angle : angle = 2 * real.pi / 3
hypothesis h_magnitude_a : ∥a∥ = 4
hypothesis h_magnitude_b : ∥b∥ = 2
hypothesis h_dot_product : real_inner a b = -4

theorem find_magnitude : magnitude (a - 2 • b) = 4 * real.sqrt 3 :=
sorry

end find_magnitude_l620_620963


namespace greatest_three_digit_multiple_of_17_l620_620344

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620344


namespace cube_faces_l620_620624

theorem cube_faces : ∀ (c : {s : Type | ∃ (x y z : ℝ), s = ({ (x0, y0, z0) : ℝ × ℝ × ℝ | x0 ≤ x ∧ y0 ≤ y ∧ z0 ≤ z}) }), 
  ∃ (f : ℕ), f = 6 :=
by 
  -- proof would be written here
  sorry

end cube_faces_l620_620624


namespace range_of_m_l620_620636

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x ≤ -1 -> (m^2 - m) * 2^x - (1/2)^x < 1) →
  -2 < m ∧ m < 3 :=
by
  sorry

end range_of_m_l620_620636


namespace ab_equals_six_l620_620024

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_l620_620024


namespace each_sister_received_49_50_l620_620913

def initial_money : ℝ := 400

def to_mom : ℝ := (1 / 4) * initial_money
def to_clothes : ℝ := (1 / 8) * initial_money
def to_charity : ℝ := (1 / 5) * initial_money
def to_groceries : ℝ := 0.15 * initial_money

def remaining_after_expenses : ℝ := initial_money - to_mom - to_clothes - to_charity - to_groceries

def investment : ℝ := 0.1 * remaining_after_expenses
def remaining_after_investment : ℝ := remaining_after_expenses - investment

/-- Prove that each sister received $49.50. -/
theorem each_sister_received_49_50 : (remaining_after_investment / 2) = 49.50 := by
  calc
    (remaining_after_investment / 2) = sorry

end each_sister_received_49_50_l620_620913


namespace complex_multiplication_l620_620733

variable (i : ℂ)
axiom i_square : i^2 = -1

theorem complex_multiplication : i * (1 + i) = -1 + i :=
by
  sorry

end complex_multiplication_l620_620733


namespace largest_three_digit_multiple_of_17_l620_620276

theorem largest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n ≤ 999) ∧ (n ≥ 100) ∧ (17 ∣ n) ∧ ∀ m, ((m > n) ∧ (m ≤ 999) ∧ (m ≥ 100) ∧ (17 ∣ m)) → false :=
by {
  sorry
}

end largest_three_digit_multiple_of_17_l620_620276


namespace greatest_three_digit_multiple_of_17_l620_620399

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) := 
  exists.intro 986 sorry

end greatest_three_digit_multiple_of_17_l620_620399


namespace common_measure_angle_l620_620135

theorem common_measure_angle (α β : ℝ) (m n : ℕ) (h : α = β * (m / n)) : α / m = β / n :=
by 
  sorry

end common_measure_angle_l620_620135


namespace probability_green_off_and_blue_on_l620_620715

def num_lamps := 10
def num_red := 4
def num_blue := 3
def num_green := 3
def num_on := 5

theorem probability_green_off_and_blue_on :
  let total_ways_arrange := (num_lamps.factorial / (num_red.factorial * num_blue.factorial * num_green.factorial)) in
  let total_ways_turn_on := (num_lamps.choose num_on) in
  let valid_arrangements :=
    let ways_place_green_left := 9.choose 2 * ((9.factorial / (4.factorial * 3.factorial * 2.factorial))) in
    let remaining_blue_placements := 7.choose 1 * ((8.factorial / (4.factorial * 2.factorial * 2.factorial))) in
    let ways_turn_on_4_more := 9.choose 4 in
    ways_place_green_left * remaining_blue_placements * ways_turn_on_4_more in
  let probability := valid_arrangements / (total_ways_arrange * total_ways_turn_on) in
  probability = 63 / 100 := by
    sorry

end probability_green_off_and_blue_on_l620_620715


namespace determine_parabola_l620_620961

-- Define the parabola passing through point P(1,1)
def parabola_passing_through (a b c : ℝ) :=
  (1:ℝ)^2 * a + 1 * b + c = 1

-- Define the condition that the tangent line at Q(2, -1) has a slope parallel to y = x - 3, which means slope = 1
def tangent_slope_at_Q (a b : ℝ) :=
  4 * a + b = 1

-- Define the parabola passing through point Q(2, -1)
def parabola_passing_through_Q (a b c : ℝ) :=
  (2:ℝ)^2 * a + (2:ℝ) * b + c = -1

-- The proof statement
theorem determine_parabola (a b c : ℝ):
  parabola_passing_through a b c ∧ 
  tangent_slope_at_Q a b ∧ 
  parabola_passing_through_Q a b c → 
  a = 3 ∧ b = -11 ∧ c = 9 :=
by
  sorry

end determine_parabola_l620_620961


namespace sum_of_numbers_l620_620749

theorem sum_of_numbers : ∃ (a b : ℕ), (a + b = 21) ∧ (a / b = 3 / 4) ∧ (max a b = 12) :=
by
  sorry

end sum_of_numbers_l620_620749


namespace obtain_1_50_cannot_obtain_1_100_obtain_1_n_l620_620811

-- Operations definitions
def machine_I (a b : ℕ) : (ℕ × ℕ) := (a + 1, b + 1)
def machine_II (a b : ℕ) (h : a % 2 = 0 ∧ b % 2 = 0) : (ℕ × ℕ) := (a / 2, b / 2)

-- Problem Statements

-- 1. Prove that it is possible to obtain (1, 50) from (5, 19)
theorem obtain_1_50 : ∃ steps : list (ℕ × ℕ), steps.head = (5, 19) ∧ steps.last = (1, 50) :=
  sorry

-- 2. Prove that it is impossible to obtain (1, 100) from (5, 19)
theorem cannot_obtain_1_100 : ¬ ∃ steps : list (ℕ × ℕ), steps.head = (5, 19) ∧ steps.last = (1, 100) :=
  sorry

-- 3. Provide the value of n given an initial card (a, b) with a < b
theorem obtain_1_n (a b : ℕ) (h : a < b) : ∃ k : ℕ, ∃ d : ℕ, d = b - a ∧ 1 + k * d = n :=
  sorry

end obtain_1_50_cannot_obtain_1_100_obtain_1_n_l620_620811


namespace find_constant_l620_620186

noncomputable def f (x : ℝ) : ℝ := x + 4

theorem find_constant : ∃ (c : ℝ), 
  ∀ (x : ℝ), (x = 0.4) →
  (c * (f (x - 2)) / f 0 + 4 = f (2 * x + 1)) := 
by 
  use 3
  intro x hx
  rw [hx]
  have h_f0 : f 0 = 4 := rfl
  have f_2x1 : f (2 * 0.4 + 1) = 2 * 0.4 + 5 := rfl
  
  calc
    (3 * f 0.4 - 2) / f 0 + 4
        = (3 * (0.2 + 4 - 2)) / 4 + 4 : by rw [f]
    ... = (3 * 2.4) / 4 + 4 : by ring
    ... = 1.8 + 4 : by norm_num
    ... = 5.8 : by norm_num
    ... = f (2 * 0.4 + 1) : by rw [f_2x1]
    

end find_constant_l620_620186


namespace greatest_three_digit_multiple_of_17_l620_620299

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 17 ∣ n :=
by 
  sorry

end greatest_three_digit_multiple_of_17_l620_620299


namespace cat_food_sufficiency_l620_620513

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end cat_food_sufficiency_l620_620513


namespace evaluate_expression_l620_620548

theorem evaluate_expression : 2 * (complex.I ^ 13) - 3 * (complex.I ^ 18) + 4 * (complex.I ^ 23) - 5 * (complex.I ^ 28) + 6 * (complex.I ^ 33) = 4 * complex.I - 2 := by
  have h0 : complex.I ^ 4 = 1 := by sorry
  have h1 : complex.I ^ 13 = complex.I := by sorry
  have h2 : complex.I ^ 18 = -1 := by sorry
  have h3 : complex.I ^ 23 = -complex.I := by sorry
  have h4 : complex.I ^ 28 = 1 := by sorry
  have h5 : complex.I ^ 33 = complex.I := by sorry
  sorry

end evaluate_expression_l620_620548


namespace football_goals_l620_620829

variable (A : ℚ) (G : ℚ)

theorem football_goals (A G : ℚ) 
    (h1 : G = 14 * A)
    (h2 : G + 3 = (A + 0.08) * 15) :
    G = 25.2 :=
by
  -- Proof here
  sorry

end football_goals_l620_620829


namespace greatest_three_digit_multiple_of_17_l620_620346

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620346


namespace polynomial_identity_l620_620126

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

theorem polynomial_identity (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h : ∀ g : ℝ → ℝ, ∀ x : ℝ, f (g x) = g (f x)) : 
  ∀ x : ℝ, f x = x :=
by
  sorry

end polynomial_identity_l620_620126


namespace greatest_three_digit_multiple_of_17_l620_620305

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 17 ∣ n :=
by 
  sorry

end greatest_three_digit_multiple_of_17_l620_620305


namespace evaluate_expression_l620_620891

theorem evaluate_expression (x : ℝ) : 
  (2 : ℝ)^(2 * x - 3) / ((7 : ℝ)^(-1) + (4 : ℝ)^(-1)) = (2 : ℝ)^(2 * x - 3) * (28 / 11) := 
by
  sorry

end evaluate_expression_l620_620891


namespace gcd_4536_13440_216_l620_620232

def gcd_of_three_numbers (a b c : ℕ) : ℕ :=
  Nat.gcd (Nat.gcd a b) c

theorem gcd_4536_13440_216 : gcd_of_three_numbers 4536 13440 216 = 216 :=
by
  sorry

end gcd_4536_13440_216_l620_620232


namespace angle_same_after_minutes_l620_620972

def angle_between_hands (H M : ℝ) : ℝ :=
  abs (30 * H - 5.5 * M)

theorem angle_same_after_minutes (x : ℝ) :
  x = 54 + 6 / 11 → 
  angle_between_hands (5 + (x / 60)) x = 150 :=
by
  sorry

end angle_same_after_minutes_l620_620972


namespace find_values_l620_620855

noncomputable def value_of_a (a : ℚ) : Prop :=
  4 + a = 2

noncomputable def value_of_b (b : ℚ) : Prop :=
  b^2 - 2 * b = 24 ∧ 4 * b^2 - 2 * b = 72

theorem find_values (a b : ℚ) (h1 : value_of_a a) (h2 : value_of_b b) :
  a = -2 ∧ b = -4 :=
by
  sorry

end find_values_l620_620855


namespace inequality_proof_l620_620154

variable {a b c d : ℝ}

theorem inequality_proof
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h_pos_d : 0 < d)
  (h_inequality : a / b < c / d) :
  a / b < (a + c) / (b + d) ∧ (a + c) / (b + d) < c / d := 
by
  sorry

end inequality_proof_l620_620154


namespace systematic_sampling_selection_l620_620777

theorem systematic_sampling_selection :
  ∃ (selected : List ℕ), selected = [3, 13, 23, 33, 43, 53] ∧
  (∀ (i : ℕ) (h : i < selected.length - 1), selected[i+1] = selected[i] + 10) ∧
  selected.length = 6 ∧
  selected.all (λ x, 1 ≤ x ∧ x ≤ 60) :=
by
  use [3, 13, 23, 33, 43, 53]
  split
  { refl }
  split
  { intros i hi
    cases i
    { simp }
    cases i
    { simp }
    cases i
    { simp }
    cases i
    { simp }
    cases i
    { simp }
    cases i
    { simp }
    }
  split
  { simp }
  { intros x hx
    repeat { split }
    all_goals { linarith }
  }
  sorry

end systematic_sampling_selection_l620_620777


namespace number_of_stickers_after_losing_page_l620_620766

theorem number_of_stickers_after_losing_page (stickers_per_page : ℕ) (initial_pages : ℕ) (pages_lost : ℕ) :
  stickers_per_page = 20 → initial_pages = 12 → pages_lost = 1 → (initial_pages - pages_lost) * stickers_per_page = 220 :=
by
  intros
  sorry

end number_of_stickers_after_losing_page_l620_620766


namespace find_hourly_rate_l620_620133

-- Definitions of conditions in a)
def hourly_rate : ℝ := sorry  -- This is what we will find.
def hours_worked : ℝ := 3
def tip_percentage : ℝ := 0.2
def total_paid : ℝ := 54

-- Functions based on the conditions
def cost_without_tip (rate : ℝ) : ℝ := hours_worked * rate
def tip_amount (rate : ℝ) : ℝ := tip_percentage * (cost_without_tip rate)
def total_cost (rate : ℝ) : ℝ := (cost_without_tip rate) + (tip_amount rate)

-- The goal is to prove that the rate is 15
theorem find_hourly_rate : total_cost 15 = total_paid :=
by
  sorry

end find_hourly_rate_l620_620133


namespace sum_of_all_z_for_which_f_4z_eq_8_l620_620117

noncomputable def f (x : ℝ) : ℝ := x^3 - x + 2

theorem sum_of_all_z_for_which_f_4z_eq_8 : (∑ z in {z : ℝ | f (4 * z) = 8}, z) = 0 :=
by
  sorry

end sum_of_all_z_for_which_f_4z_eq_8_l620_620117


namespace greatest_3_digit_multiple_of_17_l620_620387

theorem greatest_3_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≥ m) ∧ n = 986 := 
begin
  sorry
end

end greatest_3_digit_multiple_of_17_l620_620387


namespace man_reaches_home_at_11_pm_l620_620835

theorem man_reaches_home_at_11_pm :
  let start_time := 15 -- represents 3 pm in 24-hour format
  let level_speed := 4 -- km/hr
  let uphill_speed := 3 -- km/hr
  let downhill_speed := 6 -- km/hr
  let total_distance := 12 -- km
  let level_distance := 4 -- km
  let uphill_distance := 4 -- km
  let downhill_distance := 4 -- km
  let level_time := level_distance / level_speed -- time for 4 km on level ground
  let uphill_time := uphill_distance / uphill_speed -- time for 4 km uphill
  let downhill_time := downhill_distance / downhill_speed -- time for 4 km downhill
  let total_time_one_way := level_time + uphill_time + downhill_time + level_time
  let destination_time := start_time + total_time_one_way
  let return_time := destination_time + total_time_one_way
  return_time = 23 := -- represents 11 pm in 24-hour format
by
  sorry

end man_reaches_home_at_11_pm_l620_620835


namespace cat_food_sufficiency_l620_620510

theorem cat_food_sufficiency (L S : ℝ) (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end cat_food_sufficiency_l620_620510


namespace solve_inequality_find_a_range_l620_620952

theorem solve_inequality (a : ℝ) (x : ℝ) (h_a_pos : a > 0) :
  (f : ℝ → ℝ := λ x, x^2 - (a + 1/a) * x + 1) →
  (h_ineq : f x ≤ 0) →
  (if 0 < a ∧ a < 1 then a ≤ x ∧ x ≤ 1/a else if a > 1 then 1/a ≤ x ∧ x ≤ a else x = 1) :=
sorry

theorem find_a_range (a : ℝ) (h_a_pos : a > 0) :
  (f : ℝ → ℝ := λ x, x^2 - (a + 1/a) * x + 1) →
  (h_cond : ∀ x ∈ Ioo (1 :ℝ) (3 :ℝ), f x + 1/a * x > -3) →
  a < 4 :=
sorry

end solve_inequality_find_a_range_l620_620952


namespace greatest_three_digit_multiple_of_17_l620_620311

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → m ≤ n) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620311


namespace total_sum_of_numbers_l620_620705

theorem total_sum_of_numbers : 
  let digits := [2, 0, 1, 8]
  let all_permutations := Finset.univ.powerset.filter (λ s, s.card ≠ 0) -- Generating all non-empty subsets
  let all_numbers := all_permutations.map (finset.sum_digits) -- Convert each subset into a number
  let total_sum := all_numbers.sum
  in total_sum = 78311 := 
by
  sorry

end total_sum_of_numbers_l620_620705


namespace geometric_sequence_common_ratio_l620_620129

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_sum_ratio : (a 0 + a 1 + a 2) / a 2 = 7) :
  q = 1 / 2 :=
sorry

end geometric_sequence_common_ratio_l620_620129


namespace greatest_three_digit_multiple_of_17_l620_620236

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n, (n ∈ set.Icc 100 999) ∧ (n % 17 = 0) ∧ (∀ m, (m ∈ set.Icc 100 999 ∧ m % 17 = 0) → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620236


namespace valid_sequences_m_eq_2_sum_ge_20_m_eq_3_min_value_n_m_eq_2018_l620_620160

section problem

variables (n : ℕ) (A : ℕ → ℕ) (m : ℕ)

-- Conditions
def validSequence (n : ℕ) (A : ℕ → ℕ) :=
  n ≥ 4 ∧
  A 1 = 1 ∧
  A n = m ∧
  (∀ k, 1 ≤ k ∧ k < n → A (k + 1) - A k = 0 ∨ A (k + 1) - A k = 1) ∧
  (∀ i j, ∃ s t, A i + A j = A s + A t ∧ i ≠ j ∧ i ≠ s ∧ i ≠ t ∧ j ≠ s ∧ j ≠ t ∧ s ≠ t ∧ 
    i ≤ n ∧ j ≤ n ∧ s ≤ n ∧ t ≤ n)

def sumSequence (A : ℕ → ℕ) (n : ℕ) : ℕ :=
  finset.range n.succ.sum A

-- Proofs
theorem valid_sequences_m_eq_2 :
  validSequence n A ∧ m = 2 ↔
  (A = λ k, if k ≤ 4 then 1 else 2 ∨ A = λ k, if k ≤ 5 then 1 else 2) :=
sorry

theorem sum_ge_20_m_eq_3 :
  validSequence n A ∧ m = 3 → sumSequence A n ≥ 20 :=
sorry

theorem min_value_n_m_eq_2018 :
  validSequence n A ∧ m = 2018 → n ≥ 2026 :=
sorry

end problem

end valid_sequences_m_eq_2_sum_ge_20_m_eq_3_min_value_n_m_eq_2018_l620_620160


namespace intersection_of_A_and_B_l620_620938

open Set Int

def A : Set ℝ := { x | x ^ 2 - 6 * x + 8 ≤ 0 }
def B : Set ℤ := { x | abs (x - 3) < 2 }

theorem intersection_of_A_and_B :
  (A ∩ (coe '' B) = { x : ℝ | x = 2 ∨ x = 3 ∨ x = 4 }) :=
by
  sorry

end intersection_of_A_and_B_l620_620938


namespace greatest_three_digit_multiple_of_17_l620_620341

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620341


namespace product_of_ab_l620_620055

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end product_of_ab_l620_620055


namespace hypotenuse_length_l620_620577

theorem hypotenuse_length (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c^2 = a^2 + b^2) : c = 10 := 
by 
  rw [h1, h2] at h3
  simp at h3
  linarith

end hypotenuse_length_l620_620577


namespace cat_food_problem_l620_620496

noncomputable def L : ℝ
noncomputable def S : ℝ

theorem cat_food_problem (L S : ℝ) (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
by 
  sorry

end cat_food_problem_l620_620496


namespace tv_cost_is_230_l620_620131

-- Define original savings
def original_savings : ℝ := 920

-- Define fraction spent on furniture
def fraction_on_furniture : ℝ := 3 / 4

-- Define fraction spent on TV
def fraction_on_tv : ℝ := 1 - fraction_on_furniture

-- Define the cost of the TV
def cost_of_tv (savings : ℝ) (fraction_tv : ℝ) : ℝ := savings * fraction_tv

-- Theorem statement
theorem tv_cost_is_230 : 
  cost_of_tv original_savings fraction_on_tv = 230 := 
by
  sorry

end tv_cost_is_230_l620_620131


namespace trapezoid_area_sum_l620_620460

noncomputable theory

def sum_int_part (r1 r2 r3 : ℚ) (n1 n2 : ℕ) : ℤ :=
  ⌊r1 + r2 + r3 + n1 + n2⌋

theorem trapezoid_area_sum 
  (a b c d : ℕ)
  (h1 : set.in {4, 8, 5, 11} {a,b,c,d}) :
  ∃ r1 r2 r3 : ℚ, ∃ n1 n2 : ℕ, 
    (∑ r1 r2 r3 n1 n2 = 145) :=
begin
  sorry
end

end trapezoid_area_sum_l620_620460


namespace ab_value_l620_620027

theorem ab_value (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_value_l620_620027


namespace ab_equals_six_l620_620003

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l620_620003


namespace product_of_ab_l620_620052

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end product_of_ab_l620_620052


namespace product_evaluation_l620_620549

theorem product_evaluation :
  (∏ n in Finset.range 99 + 2, (1 - (1 / n))^2) = (1 / 100)^2 :=
by 
  sorry

end product_evaluation_l620_620549


namespace gnome_count_l620_620729

theorem gnome_count (g_R: ℕ) (g_W: ℕ) (h1: g_R = 4 * g_W) (h2: g_W = 20) : g_R - (40 * g_R / 100) = 48 := by
  sorry

end gnome_count_l620_620729


namespace greatest_three_digit_multiple_of_17_l620_620367

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 100 ≤ n ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l620_620367


namespace area_of_triangle_GTV_l620_620093

-- Define a data structure for a rectangle and its properties
structure Rectangle :=
  (E F G H : Point)
  (EF : ℝ)
  (FG : ℝ)

structure Segment :=
  (start end : Point)
  (length : ℝ)

-- Our conditions
def EFGH : Rectangle := {E := Point.mk 0 30, F := Point.mk 20 30, G := Point.mk 20 0, H := Point.mk 0 0, EF := 20, FG := 30}
def ET : Segment := {start := EFGH.E, end := Point.mk 0 20, length := 10}
def UH : Segment := {start := Point.mk 20 10, end := EFGH.G, length := 10}
def TUGH_area : ℝ := 200
def V : Point := Point.mk 10 15 -- Midpoint of TU assuming TU is horizontal

noncomputable def triangle_area (a b c : Point) : ℝ := -- some area calculation function, assumed to be available
by sorry

theorem area_of_triangle_GTV : triangle_area EFGH.G (Point.mk 20 10) V = 200 := 
by sorry

end area_of_triangle_GTV_l620_620093


namespace total_sum_of_permutations_l620_620703

theorem total_sum_of_permutations (digits : List ℕ) (h_digits : digits = [2, 0, 1, 8]) :
  let numbers := digits.permutations.map (λ l, l.foldl (λ a b, a * 10 + b) 0)
  ∑ n in numbers.to_finset, n = 78331 := 
by
  sorry

end total_sum_of_permutations_l620_620703


namespace decreasing_interval_0_pi_over_4_l620_620607

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.cos (x + φ)

theorem decreasing_interval_0_pi_over_4 (φ : ℝ) (hφ1 : 0 < |φ| ∧ |φ| < Real.pi / 2)
  (hodd : ∀ x : ℝ, f (x + Real.pi / 4) φ = -f (-x + Real.pi / 4) φ) :
  ∀ x : ℝ, 0 < x ∧ x < Real.pi / 4 → f x φ > f (x + 1e-6) φ :=
by sorry

end decreasing_interval_0_pi_over_4_l620_620607


namespace shortest_distance_parabola_line_l620_620678

theorem shortest_distance_parabola_line :
  ∀ (P Q : ℝ × ℝ),
  (P.snd = P.fst^2 - 4 * P.fst + 11) →
  (Q.snd = 2 * Q.fst - 6) →
  ∀ (p : ℝ), P = (p, p^2 - 4 * p + 11) →
  (∃ (Q : ℝ × ℝ), Q.snd = 2 * Q.fst - 6 ∧ 
  (dist P Q = (abs ((p - 3)^2 + 8)) / real.sqrt 5)) →
  (abs ((p - 3)^2 + 8)) / real.sqrt 5 = 8 / real.sqrt 5 → sorry :=
sorry

end shortest_distance_parabola_line_l620_620678


namespace modified_thirteen_prime_l620_620146

theorem modified_thirteen_prime (p q : ℕ) (hp : p = Nat.prime 3) (hq : q = 13 * p + 3) : q = 29 := by
  sorry

end modified_thirteen_prime_l620_620146


namespace least_positive_integer_with_five_distinct_factors_l620_620402

theorem least_positive_integer_with_five_distinct_factors : ∃ n : ℕ, (∀ m : ℕ, m < n → (∀ k : ℕ, (k ∣ m) → Nat.is_prime k → k = 2)) ∧ finset.card (Nat.divisors n) = 5 ∧ (∀ m : ℕ, finset.card (Nat.divisors m) = 5 → n ≤ m) ∧ n = 16 :=
by
  sorry

end least_positive_integer_with_five_distinct_factors_l620_620402


namespace largest_possible_elements_in_A_l620_620684

variable {p : ℕ} [Fact p.Prime]

def is_prime_divisors_set (A : Set ℕ) (ps : Finset ℕ) : Prop :=
  ∀ a ∈ A, (Nat.factors a).toFinset = ps

def is_not_perfect_pth_power (A : Set ℕ) : Prop :=
  ∀ (s : Finset ℕ), s.Nonempty → (∏ i in s, i) ∉ {n : ℕ | ∃ k : ℕ, n = k^p}

theorem largest_possible_elements_in_A (A : Set ℕ)
  (h1 : A ⊆ Set.univ)
  (h2 : is_prime_divisors_set A (Finset.range (p-1)))
  (h3 : is_not_perfect_pth_power A) :
  A.Finite.toFinset.card ≤ (p-1)^2 :=
sorry

end largest_possible_elements_in_A_l620_620684


namespace seating_arrangement_l620_620216

theorem seating_arrangement :
  let seats := ["A", "B", "C", "D", "E", "F"]
  let families := {1, 2}
  let num_people := 6
  -- 3 adults and 3 children, no two same-family members sit together 
  (2 * (Finset.univ : Finset (Fin 3)).card.factorial * (Finset.univ : Finset (Fin 3)).card.factorial) = 72 :=
by
  sorry

end seating_arrangement_l620_620216


namespace minimum_distance_l620_620657

theorem minimum_distance :
  ∀ (A B C D : ℝ³), 
  (DA = 5) ∧ (DB = 5) ∧ (DC = 1) ∧ pairwise_perpendicular DA DB DC → 
  min_travel_distance A B C D = 10 * real.sqrt 3 / 9 := 
sorry

end minimum_distance_l620_620657


namespace greatest_three_digit_multiple_of_17_l620_620322

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → m ≤ n) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620322


namespace stickers_after_loss_l620_620771

-- Conditions
def stickers_per_page : ℕ := 20
def initial_pages : ℕ := 12
def lost_pages : ℕ := 1

-- Problem statement
theorem stickers_after_loss : (initial_pages - lost_pages) * stickers_per_page = 220 := by
  sorry

end stickers_after_loss_l620_620771


namespace equation_solutions_l620_620142

theorem equation_solutions :
  (∑ n in (finset.range 16), if n = 0 then 4 else 4 * n) = 64 :=
  sorry

end equation_solutions_l620_620142


namespace sqrt_a_squared_b_l620_620071

variable {a b : ℝ}

theorem sqrt_a_squared_b (h: a * b < 0) : Real.sqrt (a^2 * b) = -a * Real.sqrt b := by
  sorry

end sqrt_a_squared_b_l620_620071


namespace reflection_direction_vector_l620_620191

-- Define the conditions
noncomputable def reflection_matrix : Matrix (Fin 2) (Fin 2) ℚ := 
  !![3/5, 4/5; 4/5, -3/5]

def is_int_vector (v : ℚ × ℚ) : Prop :=
  ∃ (a b : ℤ), (a : ℚ, b : ℚ) = v

def gcd_condition (v : ℚ × ℚ) : Prop :=
  ∃ (a b : ℤ), (a : ℚ, b : ℚ) = v ∧ a > 0 ∧ Int.gcd a.natAbs b.natAbs = 1

-- The statement to be proven
theorem reflection_direction_vector :
  ∃ (v : ℚ × ℚ), reflection_matrix.mul_vec v = v ∧ is_int_vector v ∧ gcd_condition v ∧ v = (4, -3) :=
sorry

end reflection_direction_vector_l620_620191


namespace greatest_three_digit_multiple_of_17_l620_620243

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n, (n ∈ set.Icc 100 999) ∧ (n % 17 = 0) ∧ (∀ m, (m ∈ set.Icc 100 999 ∧ m % 17 = 0) → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620243


namespace greatest_three_digit_multiple_of_17_l620_620359

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (n % 17 = 0) ∧ ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ (m % 17 = 0) → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620359


namespace cat_food_sufficiency_l620_620523

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S >= 11 :=
sorry

end cat_food_sufficiency_l620_620523


namespace cost_of_one_book_l620_620137

theorem cost_of_one_book (m : ℕ) (H1: 1100 < 900 + 9 * m ∧ 900 + 9 * m < 1200)
                                (H2: 1500 < 1300 + 13 * m ∧ 1300 + 13 * m < 1600) : 
                                m = 23 :=
by {
  sorry
}

end cost_of_one_book_l620_620137


namespace sum_of_squares_digits_cycle_l620_620810

theorem sum_of_squares_digits_cycle (n : ℕ) :
  (∃ k : ℕ, k ≠ 1 ∧ (∃ m : ℕ, (sum_of_squares_digits^[m] n = k) ∧ 
  (k = 145 ∨ sum_of_squares_digits^[m] k = 145))) → 
  ∃ l : ℕ, (sum_of_squares_digits^[l] k ∈ {145, 42, 20, 4, 16, 37, 58, 89}) := 
sorry

def sum_of_squares_digits (n : ℕ) : ℕ :=
  (n.digits 10).map (λ d, d^2).sum

end sum_of_squares_digits_cycle_l620_620810


namespace teams_in_league_l620_620778

def number_of_teams (n : ℕ) := n * (n - 1) / 2

theorem teams_in_league : ∃ n : ℕ, number_of_teams n = 36 ∧ n = 9 := by
  sorry

end teams_in_league_l620_620778


namespace even_integers_with_even_factors_l620_620623

theorem even_integers_with_even_factors : 
  let even_ints := {n : ℕ | 1 ≤ n ∧ n ≤ 100 ∧ Even n}
  let not_perfect_square := {n : ℕ | ∀ m : ℕ, m * m ≠ n}
  let count_even := even_ints.filter (∉ not_perfect_square)
  count_even.card = 45 :=
sorry

end even_integers_with_even_factors_l620_620623


namespace div_sum_is_four_l620_620123

def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem div_sum_is_four (K L : ℕ) (hk : K = 6 ∨ K = 4) (hl : L = 4 ∨ L = 3) :
  num_divisors (K + 2 * L) = 4 :=
by {
  cases hk with hK6 hK4;
  cases hl with hL4 hL3;
  { simp [hK6, hL4, num_divisors], sorry },
  { simp [hK6, hL3, num_divisors], sorry },
  { simp [hK4, hL4, num_divisors], sorry },
  { simp [hK4, hL3, num_divisors], sorry }
}

end div_sum_is_four_l620_620123


namespace ab_eq_6_pos_or_neg_max_ab_when_sum_neg5_ab_lt_0_sign_of_sum_l620_620426

-- Problem 1
theorem ab_eq_6_pos_or_neg (a b : ℚ) (h : a * b = 6) : a + b > 0 ∨ a + b < 0 := sorry

-- Problem 2
theorem max_ab_when_sum_neg5 (a b : ℤ) (h : a + b = -5) : a * b ≤ 6 := sorry

-- Problem 3
theorem ab_lt_0_sign_of_sum (a b : ℚ) (h : a * b < 0) : (a + b > 0 ∨ a + b = 0 ∨ a + b < 0) := sorry

end ab_eq_6_pos_or_neg_max_ab_when_sum_neg5_ab_lt_0_sign_of_sum_l620_620426


namespace ab_equals_six_l620_620001

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l620_620001


namespace greatest_three_digit_multiple_of_17_l620_620285

def is_multiple_of (n : ℕ) (k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, is_three_digit n ∧ is_multiple_of n 17 → n = 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620285


namespace trig_identity_example_l620_620866

theorem trig_identity_example :
  sin (43 * Real.pi / 180) * cos (13 * Real.pi / 180) - 
  cos (43 * Real.pi / 180) * sin (13 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end trig_identity_example_l620_620866


namespace minimum_both_likes_l620_620699

theorem minimum_both_likes (total : ℕ) (like_mozart : ℕ) (like_beethoven : ℕ) (neither : ℕ) :
  total = 120 ∧ like_mozart = 95 ∧ like_beethoven = 80 ∧ neither >= 10 →
  ∃ (both_likes : ℕ), both_likes = 25 :=
by
  intros h
  obtain ⟨ht, hm, hb, hn⟩ := h
  use 25
  sorry

end minimum_both_likes_l620_620699


namespace ab_eq_six_l620_620067

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_eq_six_l620_620067


namespace divisors_of_expression_l620_620557

-- Defining the primes and the expressions for x, y, and z
variables (p q r : ℕ)
variables (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)

def x := p^2
def y := q^2
def z := r^4

-- Main theorem statement
theorem divisors_of_expression (hx : x = p^2) (hy : y = q^2) (hz : z = r^4) :
  ∀ (x y z : ℕ) (hx : x = p^2) (hy : y = q^2) (hz : z = r^4),
    Nat.divisors_count (x^3 * y^4 * z^2) = 567 := by
  sorry

end divisors_of_expression_l620_620557


namespace greatest_three_digit_multiple_of_17_l620_620331

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

theorem greatest_three_digit_multiple_of_17 : ∀ n, n < 1000 → is_multiple_of n 17 → n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620331


namespace product_of_ab_l620_620057

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end product_of_ab_l620_620057


namespace compare_sums_of_sines_l620_620664

theorem compare_sums_of_sines {A B C : ℝ} 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (h_sum : A + B + C = π) :
  (if A < π / 2 ∧ B < π / 2 ∧ C < π / 2 then
    (1 / (Real.sin (2 * A)) + 1 / (Real.sin (2 * B)) + 1 / (Real.sin (2 * C))
      ≥ 1 / (Real.sin A) + 1 / (Real.sin B) + 1 / (Real.sin C))
  else
    (1 / (Real.sin (2 * A)) + 1 / (Real.sin (2 * B)) + 1 / (Real.sin (2 * C))
      < 1 / (Real.sin A) + 1 / (Real.sin B) + 1 / (Real.sin C))) :=
sorry

end compare_sums_of_sines_l620_620664


namespace ab_eq_six_l620_620066

theorem ab_eq_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_eq_six_l620_620066


namespace greatest_three_digit_multiple_of_17_l620_620330

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

theorem greatest_three_digit_multiple_of_17 : ∀ n, n < 1000 → is_multiple_of n 17 → n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620330


namespace selection_schemes_count_l620_620716

theorem selection_schemes_count (person : Type) (places : Type)
  (A B : person) (Xijiang : places)
  (total_people : Fin 6 → person)
  (total_places : Fin 4 → places)
  (constraints : {p : person // p ≠ A ∧ p ≠ B}) :
  (∀ i, total_people i ≠ total_people j → i ≠ j) → 
  (∀ i, total_places i ≠ total_places j → i ≠ j) →
  (∃ (f : Fin 4 → {p : person // p ≠ A ∧ p ≠ B}), f 0 ∈ set.univ) →
  (∃ (g : Fin 4 → person), g 0 = total_people 0 ∧
    ∀ i, g i ∈ {p | ∃ (j : Fin 4), p = total_people j}) →
  (4 * 10 * 6 = 240) := 
by
  sorry

end selection_schemes_count_l620_620716


namespace integral_inequality_l620_620155

theorem integral_inequality (n : ℕ) (hn : 2 ≤ n) : 
  (1 : ℝ) / (n : ℝ) < 
  (∫ x in 0 .. (real.pi / 2), 1 / (1 + real.cos x)^n) ∧
  (∫ x in 0 .. (real.pi / 2), 1 / (1 + real.cos x)^n) < 
  ((n + 5) : ℝ) / ((n : ℝ) * (n + 1)) :=
sorry

end integral_inequality_l620_620155


namespace polynomial_equation_roots_l620_620114

theorem polynomial_equation_roots (a b c t : ℝ) 
  (h1 : a^3 - 6*a^2 + 8*a - 1 = 0) 
  (h2 : b^3 - 6*b^2 + 8*b - 1 = 0) 
  (h3 : c^3 - 6*c^2 + 8*c - 1 = 0) 
  (h4 : t = sqrt a + sqrt b + sqrt c) : 
  t^4 - 20*t^2 + 4*t = -8*t^2 + 12*t - 4 :=
sorry

end polynomial_equation_roots_l620_620114


namespace common_root_l620_620564

theorem common_root (p : ℝ) :
  (∃ x : ℝ, x^2 - (p+2)*x + 2*p + 6 = 0 ∧ 2*x^2 - (p+4)*x + 2*p + 3 = 0) ↔ (p = -3 ∨ p = 9) :=
by
  sorry

end common_root_l620_620564


namespace terminal_side_of_half_angle_quadrant_l620_620979

def is_angle_in_third_quadrant (α : ℝ) (k : ℤ) : Prop :=
  k * 360 + 180 < α ∧ α < k * 360 + 270

def is_terminal_side_of_half_angle_in_quadrant (α : ℝ) : Prop :=
  (∃ n : ℤ, n * 360 + 90 < α / 2 ∧ α / 2 < n * 360 + 135) ∨ 
  (∃ n : ℤ, n * 360 + 270 < α / 2 ∧ α / 2 < n * 360 + 315)

theorem terminal_side_of_half_angle_quadrant (α : ℝ) (k : ℤ) :
  is_angle_in_third_quadrant α k → is_terminal_side_of_half_angle_in_quadrant α := 
sorry

end terminal_side_of_half_angle_quadrant_l620_620979


namespace P_at_7_is_3186_l620_620674

noncomputable def P : ℕ → ℤ := sorry  -- The polynomial P(x)

theorem P_at_7_is_3186 :
  ∀ P : ℕ → ℤ,
    (∀ k, k ∈ finset.range 7 → P k = k.factorial) ∧ 
    (∃ n, ∀ k, P k = (polynomial.eval k)) →
    P 7 = 3186 := 
by
  intros _ ⟨h1, h2⟩
  sorry

end P_at_7_is_3186_l620_620674


namespace range_of_a_l620_620587

-- Definitions for the given conditions
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*a*x + 1 > 0
def q (a : ℝ) : Prop := ∃ x : ℝ, a*x^2 + 2 ≤ 0

-- The proof problem statement in Lean 4
theorem range_of_a (a : ℝ) : ¬ (p(a) ∨ q(a)) → a ∈ set.Ici 1 := by
  sorry

end range_of_a_l620_620587


namespace equivalent_percentage_increase_is_correct_l620_620199

noncomputable def single_equivalent_percentage_increase : ℝ :=
let initial_price : ℝ := 100
let price_after_30_percent_increase := initial_price * (1 + 0.30)
let price_after_20_percent_increase := price_after_30_percent_increase * (1 + 0.20)
let price_after_10_percent_discount := price_after_20_percent_increase * (1 - 0.10)
let price_after_10_percent_reversal := price_after_10_percent_discount * (1 + 0.10)
let price_after_additional_5_percent := price_after_10_percent_reversal * (1 + 0.05)
let price_after_8_percent_tax := price_after_additional_5_percent * (1 + 0.08)
let final_price := price_after_8_percent_tax * (1 + 0.02)
let equivalent_percentage_increase := ((final_price - initial_price) / initial_price) * 100
in equivalent_percentage_increase

theorem equivalent_percentage_increase_is_correct :
  single_equivalent_percentage_increase = 78.63 := by
  sorry

end equivalent_percentage_increase_is_correct_l620_620199


namespace greatest_three_digit_multiple_of_17_l620_620329

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

theorem greatest_three_digit_multiple_of_17 : ∀ n, n < 1000 → is_multiple_of n 17 → n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l620_620329


namespace Alejandra_overall_score_l620_620845

theorem Alejandra_overall_score :
  let score1 := (60/100 : ℝ) * 20
  let score2 := (75/100 : ℝ) * 30
  let score3 := (85/100 : ℝ) * 40
  let total_score := score1 + score2 + score3
  let total_questions := 90
  let overall_percentage := (total_score / total_questions) * 100
  round overall_percentage = 77 :=
by
  sorry

end Alejandra_overall_score_l620_620845


namespace chelsea_total_time_l620_620872

def num_batches := 4
def bake_time_per_batch := 20  -- minutes
def ice_time_per_batch := 30   -- minutes
def cupcakes_per_batch := 6
def additional_time_first_batch := 10 -- per cupcake
def additional_time_second_batch := 15 -- per cupcake
def additional_time_third_batch := 12 -- per cupcake
def additional_time_fourth_batch := 20 -- per cupcake

def total_bake_ice_time := bake_time_per_batch + ice_time_per_batch
def total_bake_ice_time_all_batches := total_bake_ice_time * num_batches

def total_additional_time_first_batch := additional_time_first_batch * cupcakes_per_batch
def total_additional_time_second_batch := additional_time_second_batch * cupcakes_per_batch
def total_additional_time_third_batch := additional_time_third_batch * cupcakes_per_batch
def total_additional_time_fourth_batch := additional_time_fourth_batch * cupcakes_per_batch

def total_additional_time := 
  total_additional_time_first_batch +
  total_additional_time_second_batch +
  total_additional_time_third_batch +
  total_additional_time_fourth_batch

def total_time := total_bake_ice_time_all_batches + total_additional_time

theorem chelsea_total_time : total_time = 542 := by
  sorry

end chelsea_total_time_l620_620872


namespace probability_calculation_l620_620795

def primeFactorization_90 : (ℕ → ℕ) :=
  (λ n, if n = 2 then 1 else if n = 3 then 2 else if n = 5 then 1 else 0)

def numberOfFactors_of_90 : ℕ :=
  (primeFactorization_90 2 + 1) * (primeFactorization_90 3 + 1) * (primeFactorization_90 5 + 1)

def factors_less_than_eight_of_90 : List ℕ :=
  [1, 2, 3, 5, 6] -- since 4 and 7 do not divide 90

def count_factors_less_than_eight : ℕ := factors_less_than_eight_of_90.length

def probability_factor_less_than_eight : ℚ :=
  (count_factors_less_than_eight : ℚ) / (numberOfFactors_of_90 : ℚ)

theorem probability_calculation :
  probability_factor_less_than_eight = 5 / 12 :=
  by
  sorry

end probability_calculation_l620_620795


namespace number_of_correct_statements_l620_620948

def z : ℂ := 1 + complex.i

def statement1 : Prop := complex.abs z = real.sqrt 2
def statement2 : Prop := z = conj z
def statement3 : Prop := complex.im z = complex.i
def statement4 : Prop := 0 < complex.re z ∧ 0 < complex.im z

def count_correct_statements (s1 s2 s3 s4 : Prop) : ℕ :=
  (if s1 then 1 else 0) +
  (if s2 then 1 else 0) +
  (if s3 then 1 else 0) +
  (if s4 then 1 else 0)

theorem number_of_correct_statements :
  count_correct_statements statement1 statement2 statement3 statement4 = 3 :=
sorry

end number_of_correct_statements_l620_620948


namespace greatest_three_digit_multiple_of_17_l620_620287

def is_multiple_of (n : ℕ) (k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, is_three_digit n ∧ is_multiple_of n 17 → n = 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620287


namespace license_plate_count_l620_620625

theorem license_plate_count : 
  let consonants := 20
  let vowels := 6
  let digits := 10
  4 * consonants * vowels * consonants * digits = 24000 :=
by
  sorry

end license_plate_count_l620_620625


namespace ab_value_l620_620008

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_value_l620_620008


namespace sum_of_square_roots_le_sqrt_three_l620_620923

theorem sum_of_square_roots_le_sqrt_three (a b c : ℝ) (h1 : a + b + c = 1) (h2 : 0 ≤ a) (h3 : 0 ≤ b) (h4 : 0 ≤ c) :
  sqrt a + sqrt b + sqrt c ≤ sqrt 3 :=
sorry

end sum_of_square_roots_le_sqrt_three_l620_620923


namespace parallel_line_distance_l620_620735

theorem parallel_line_distance (x y : ℝ) (P : ℝ × ℝ) (k : ℝ) 
  (h1 : 3 * x + 4 * y - 12 = 0) 
  (h2 : P = (4, -5))
  (h3 : k = 4): 
  (∃ c, 3 * x + 4 * y + c = 0 ∧ 3 * 4 + 4 * -5 + c = 0 ∧ (| -12 - c |) / (sqrt (3^2 + 4^2)) = k) :=
begin
  sorry
end

end parallel_line_distance_l620_620735


namespace exists_m_such_that_fn_is_divisible_by_36_l620_620098

theorem exists_m_such_that_fn_is_divisible_by_36 : 
  ∃ m : ℕ, (∀ n : ℕ, 0 < n → (∃ k : ℤ, f(n) = 36 * k)) ∧ m = 9 :=
by
  let f := λ n : ℕ, 3 ^ n * (2 * n + 7) + 9
  use 9
  split
  intro n
  intros h
  sorry
  rfl

end exists_m_such_that_fn_is_divisible_by_36_l620_620098


namespace greatest_three_digit_multiple_of_17_l620_620251

def is_three_digit (n : Int) : Prop := (n ≥ 100 ∧ n ≤ 999)
def is_multiple_of_17 (n : Int) : Prop := (∃ k : Int, n = 17 * k)

theorem greatest_three_digit_multiple_of_17 : ∀ n : Int, is_three_digit n ∧ is_multiple_of_17 n → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l620_620251


namespace range_of_a_l620_620612

noncomputable def f (x : ℝ) := Real.log x + 1 / x
noncomputable def g (x a : ℝ) := x + 1 / (x - a)

theorem range_of_a (a : ℝ) :
  (∀ x1 ∈ Icc 0 2, ∃ x2 > a, f x1 ≥ g x2 a) → a ≤ -1 :=
sorry

end range_of_a_l620_620612


namespace ab_equals_six_l620_620016

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six_l620_620016


namespace equivalent_equation_l620_620072

theorem equivalent_equation (x y : ℝ) 
  (x_ne_0 : x ≠ 0) (x_ne_3 : x ≠ 3) 
  (y_ne_0 : y ≠ 0) (y_ne_5 : y ≠ 5)
  (main_equation : (3 / x) + (4 / y) = 1 / 3) : 
  x = 9 * y / (y - 12) :=
sorry

end equivalent_equation_l620_620072


namespace probability_same_penny_dime_halfdollar_l620_620725

theorem probability_same_penny_dime_halfdollar :
  let total_outcomes := 2^6
  let successful_outcomes := 2 * 2^3
  successful_outcomes / total_outcomes = 1 / 4 :=
begin
  sorry
end

end probability_same_penny_dime_halfdollar_l620_620725


namespace euler_convex_polyhedron_l620_620547

noncomputable def calculate_value (V E F T P: ℕ) : ℕ :=
  100 * P + 10 * T + V

theorem euler_convex_polyhedron (V E F T P : ℕ) (h1 : V - E + F = 2) (h2 : F = 32) (h3 : T = 2) (h4 : P = 2) (h5 : ∀ x y: ℕ, x + y = F → x = 20 → y = 12 → E = (3 * x + 5 * y) / 2) : 
  calculate_value V E F T P = 250 :=
by
  rw [h2, h3, h4, ← h5 20 12 (by refl) (by refl) (by refl)]
  sorry -- complete the proof

end euler_convex_polyhedron_l620_620547


namespace exists_point_D_l620_620697

-- Define a semicircle with diameter AB
variable {A B D C : Point}
variable {AB : Line}
variable {O : Point} -- O is the center of the semicircle

-- Assume O is the midpoint of AB, i.e., AB = 2 * AO = 2 * BO
axiom midpoint_O (A B O : Point) : distance A O = distance B O

-- Define the conditions: D lies on the semicircle and DA = DC
axiom D_on_circumference {D : Point} (A B : Point) (O : Point) : D ∈ semicircle A B O
axiom DA_eq_DC {D : Point} (A : Point) (C : Point) : distance D A = distance D C

-- Tangent at point D intersects AB at point C
axiom tangent_intersects_AB {A B D C : Point} : tangent_at D intersects diameter A B at C

-- Objective: Prove such a point D exists
theorem exists_point_D (A B C D O : Point) (midpoint_O A B O) (D_on_circumference D A B O) (DA_eq_DC D A C) (tangent_intersects_AB A B D C) : 
  ∃ D, D ∈ semicircle A B O ∧ distance D A = distance D C :=
sorry

end exists_point_D_l620_620697


namespace relationship_among_abc_l620_620571

noncomputable def a : ℝ := Real.log 3 / Real.log 2    -- a = log_2(3)
noncomputable def b : ℝ := 2^(-1 / 3)                -- b = 2^(-1/3)
noncomputable def c : ℝ := Real.log (1 / 30) / Real.log (1 / 3)  -- c = log_(1/3)(1/30)

theorem relationship_among_abc : c > a ∧ a > b :=
by
  have hab: 1 < 2 := by norm_num -- For real logarithms to work properly
  sorry

end relationship_among_abc_l620_620571


namespace solve_trig_eq_l620_620540

noncomputable def find_solutions (x : ℝ) : Prop :=
  sin x + cos x = 1

theorem solve_trig_eq : 
  ∀ x ∈ Icc 0 real.pi, find_solutions x → x = 0 ∨ x = real.pi / 2 :=
begin
  -- The proof is omitted as per instructions
  sorry
end

end solve_trig_eq_l620_620540


namespace number_of_stickers_after_losing_page_l620_620765

theorem number_of_stickers_after_losing_page (stickers_per_page : ℕ) (initial_pages : ℕ) (pages_lost : ℕ) :
  stickers_per_page = 20 → initial_pages = 12 → pages_lost = 1 → (initial_pages - pages_lost) * stickers_per_page = 220 :=
by
  intros
  sorry

end number_of_stickers_after_losing_page_l620_620765


namespace irrational_pi_l620_620848

theorem irrational_pi : 
  ∀ (x : ℝ), x = -2 ∨ x = 0 ∨ x = real.sqrt 9 ∨ x = real.pi → irrational real.pi :=
by
  intro x
  intro h
  -- application of conditions and solution specific proof will go here
  sorry

end irrational_pi_l620_620848


namespace evaluate_expression_at_x_eq_3_l620_620797

theorem evaluate_expression_at_x_eq_3 : 
  let x := 3 in x^5 - (5 * x)^2 = 18 := 
by
  let x := 3
  sorry

end evaluate_expression_at_x_eq_3_l620_620797


namespace greatest_three_digit_multiple_of_17_l620_620234

theorem greatest_three_digit_multiple_of_17 : 
  ∃ n, (n ∈ set.Icc 100 999) ∧ (n % 17 = 0) ∧ (∀ m, (m ∈ set.Icc 100 999 ∧ m % 17 = 0) → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620234


namespace cost_of_fencing_l620_620423

noncomputable def π : ℝ := Real.pi
def diameter (d : ℝ) := 14  -- Diameter of the circular field in meters
def rate_per_meter (r : ℝ) := 2.50  -- Rate of fencing per meter in Rs

theorem cost_of_fencing : 
  let C := π * (diameter 14) in
  let total_cost := C * rate_per_meter 2.50 in
  total_cost = 109.95 :=
by
  sorry

end cost_of_fencing_l620_620423


namespace power_of_two_digits_exactly_four_powers_of_two_l620_620165
noncomputable def log10_2 : ℝ := Real.log10 2

theorem power_of_two_digits (m : ℕ) :
  ∃ n_min n_max, (n_min = Int.ceil ((m - 1 : ℝ) / log10_2)) ∧ 
                (n_max = (m : ℝ) / log10_2) ∧
                (∀ n, n_min ≤ n ∧ n < n_max → 10^(m-1) ≤ 2^n ∧ 2^n < 10^m) ∧
                (∀ n, n_min ≤ n ∧ n + 3 < n_max → ∃ k, k ∈ {n, n+1, n+2, n+3, n+4} ∧ 2^k < 10^m) ∧
                (∀ n, n_min ≤ n ∧ n + 4 < n_max → False) := sorry

theorem exactly_four_powers_of_two (m : ℕ) :
  (∃ n_min n_max, (n_min = Int.ceil ((m - 1 : ℝ) / log10_2)) ∧ 
                 (n_max = (m : ℝ) / log10_2) ∧ 
                 (\(n_max - n_min ≥ 3)) 
  → 𝓝 {n_min, n_min+1, n_min+2, n_min+3}) :=
sorry

end power_of_two_digits_exactly_four_powers_of_two_l620_620165


namespace ab_value_l620_620013

theorem ab_value (a b : ℕ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end ab_value_l620_620013


namespace dot_product_intersection_l620_620834

theorem dot_product_intersection {A B C : Point} :
  let line_eq := λ x, - (Real.sqrt 3 / 3) * (x - 4)
  let circle_eq := λ x y, x^2 + y^2 - 4 * x = 0
  (4, 0) ∈ line_eq ∧ line_eq = circle_eq →  -- This checks the equation of the line passes through (4, 0) and intersects the circle
  (∃ x1 x2, x^2 - 5 * x + 4 = 0) →  -- This translates the intersection of the line with the circle 
  let |AB| := (distance (1, 0) (4, 0)) in  -- Distance or length of AB
  let radius := 2 in  -- Radius of the circle
  let ∠CAB := 30° in  -- Angle CAB is 30 degrees
  let dot := 2 * Real.sqrt 3 * 2 * Real.sqrt 3 / 2 = 6 in  -- Dot product calculation
  ∀ ⦃v w : Vector⦄, 
  v = AB → w = AC → 
  v ⬝ w = 6 := 
by 
  sorry

end dot_product_intersection_l620_620834


namespace who_plays_chess_l620_620081

def person_plays_chess (A B C : Prop) : Prop := 
  (A ∧ ¬ B ∧ ¬ C) ∨ (¬ A ∧ B ∧ ¬ C) ∨ (¬ A ∧ ¬ B ∧ C)

axiom statement_A : Prop
axiom statement_B : Prop
axiom statement_C : Prop
axiom one_statement_true : (statement_A ∧ ¬ statement_B ∧ ¬ statement_C) ∨ (¬ statement_A ∧ statement_B ∧ ¬ statement_C) ∨ (¬ statement_A ∧ ¬ statement_B ∧ statement_C)

-- Definition translating the statements made by A, B, and C
def A_plays := true
def B_not_plays := true
def A_not_plays := ¬ A_plays

-- Axiom stating that only one of A's, B's, or C's statements are true
axiom only_one_true : (statement_A ∧ ¬ statement_B ∧ ¬ statement_C) ∨ (¬ statement_A ∧ statement_B ∧ ¬ statement_C) ∨ (¬ statement_A ∧ ¬ statement_B ∧ statement_C)

-- Prove that B is the one who knows how to play Chinese chess
theorem who_plays_chess : B_plays :=
by
  -- Insert proof steps here
  sorry

end who_plays_chess_l620_620081


namespace general_term_for_a_n_T_n_formula_T_n_range_l620_620936

noncomputable def a_n : ℕ → ℕ
| n => 2 * n - 1

def b_n (n : ℕ) : ℚ :=
1 / (a_n n * a_n (n + 1))

def T_n (n : ℕ) : ℚ :=
((Finset.range n).sum fun k => b_n (k + 1))

theorem general_term_for_a_n :
  ∀ (n : ℕ), a_n n = 2 * n - 1 :=
by sorry

theorem T_n_formula (n : ℕ) :
  T_n n = n / (2 * n + 1) :=
by sorry

theorem T_n_range :
  ∀ (n : ℕ), (1 : ℚ)/4 ≤ T_n n ∧ T_n n < (1 : ℚ)/2 := 
by sorry


end general_term_for_a_n_T_n_formula_T_n_range_l620_620936


namespace cnc_completion_time_l620_620815

-- Define start time and one-fourth completion time.
def start_time : ℕ := 9 * 60  -- 9:00 AM in minutes
def one_fourth_time : ℕ := 12 * 60 + 30  -- 12:30 PM in minutes

-- Define total time to complete all tasks in minutes.
def total_time_in_minutes : ℕ := 14 * 60 -- 14 hours

-- Convert finish time from minutes back to hours and minutes (it should correspond to 11:00 PM).
def finish_time : ℕ := (start_time + total_time_in_minutes) % (24 * 60) -- Time modulo 24 hours in minutes

-- Helper function to convert time in minutes to readable hours and minutes format.
def time_in_minutes_to_hm_str (minutes : ℕ) : string :=
  let hours := minutes / 60
  let mins := minutes % 60
  s!"{hours % 24}: {if mins < 10 then "0" else ""}{mins}" ++ (if hours >= 12 then " PM" else " AM")

theorem cnc_completion_time :
  time_in_minutes_to_hm_str finish_time = "11:00 PM" := 
sorry

end cnc_completion_time_l620_620815


namespace fourth_term_is_one_l620_620183

def fourth_term_of_gp (a1 a2 a3 : Real) (r : Real) : Real :=
  a1 * r^3

theorem fourth_term_is_one :
  fourth_term_of_gp (3^(1/4)) (3^(1/6)) (3^(1/12)) (3^(-1/12)) = 1 := by
  sorry

end fourth_term_is_one_l620_620183


namespace min_value_of_b_plus_2_div_a_l620_620941

theorem min_value_of_b_plus_2_div_a (a : ℝ) (b : ℝ) (h₁ : 0 < a) 
  (h₂ : ∀ x : ℝ, 0 < x → (ax - 1) * (x^2 + bx - 4) ≥ 0) : 
  ∃ a' b', (a' > 0 ∧ b' = 4 * a' - 1 / a') ∧ b' + 2 / a' = 4 :=
by
  sorry

end min_value_of_b_plus_2_div_a_l620_620941


namespace find_b_l620_620187

def slope (a b c : ℝ) : ℝ := -a / b

def perpendicular_slopes (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem find_b (b : ℝ) :
  perpendicular_slopes (slope 2 (-3) 6) (slope b (-3) (-4)) →
  b = -9/2 :=
by
  intros h
  sorry

end find_b_l620_620187


namespace henry_oscillation_distance_l620_620970

theorem henry_oscillation_distance :
  let home := 0
  let park := 3
  let c := 3 / 4
  let d := 3 / 2
  let sequence := λ n, if even n then c else d
  abs (c - d) = 1 / 2 :=
by
  -- Definitions
  let home : ℝ := 0
  let park : ℝ := 3
  let c :=
    let c0 := park / 2 in
    let c1 := c0 / 2 in
    c1
  let d :=
    let d0 := park - c in
    let d1 := d0 / 2 + c in
    d1
  calc abs (c - d)
     = abs (3/4 - 1)                 : by
     ... = abs (-1/4)                : by
     ... = 1/4                       : by
  sorry

end henry_oscillation_distance_l620_620970


namespace greatest_three_digit_multiple_of_17_l620_620293

def is_multiple_of (n : ℕ) (k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, is_three_digit n ∧ is_multiple_of n 17 → n = 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620293


namespace car_repair_cost_l620_620819

noncomputable def total_cost (first_mechanic_rate: ℝ) (first_mechanic_hours: ℕ) 
    (first_mechanic_days: ℕ) (second_mechanic_rate: ℝ) 
    (second_mechanic_hours: ℕ) (second_mechanic_days: ℕ) 
    (discount_first: ℝ) (discount_second: ℝ) 
    (parts_cost: ℝ) (sales_tax_rate: ℝ): ℝ :=
  let first_mechanic_cost := first_mechanic_rate * first_mechanic_hours * first_mechanic_days
  let second_mechanic_cost := second_mechanic_rate * second_mechanic_hours * second_mechanic_days
  let first_mechanic_discounted := first_mechanic_cost - (discount_first * first_mechanic_cost)
  let second_mechanic_discounted := second_mechanic_cost - (discount_second * second_mechanic_cost)
  let total_before_tax := first_mechanic_discounted + second_mechanic_discounted + parts_cost
  let sales_tax := sales_tax_rate * total_before_tax
  total_before_tax + sales_tax

theorem car_repair_cost :
  total_cost 60 8 14 75 6 10 0.15 0.10 3200 0.07 = 13869.34 := by
  sorry

end car_repair_cost_l620_620819


namespace fuel_consumption_per_100_km_l620_620209

-- Defining the conditions
variable (initial_fuel : ℕ) (remaining_fuel : ℕ) (distance_traveled : ℕ)

-- Assuming the conditions provided in the problem
axiom initial_fuel_def : initial_fuel = 47
axiom remaining_fuel_def : remaining_fuel = 14
axiom distance_traveled_def : distance_traveled = 275

-- The statement to prove: fuel consumption per 100 km
theorem fuel_consumption_per_100_km (initial_fuel remaining_fuel distance_traveled : ℕ) :
  initial_fuel = 47 →
  remaining_fuel = 14 →
  distance_traveled = 275 →
  (initial_fuel - remaining_fuel) * 100 / distance_traveled = 12 :=
by
  sorry

end fuel_consumption_per_100_km_l620_620209


namespace parker_shorter_than_daisy_l620_620701

noncomputable def solve_height_difference : Nat :=
  let R := 60
  let D := R + 8
  let avg := 64
  ((3 * avg) - (D + R))

theorem parker_shorter_than_daisy :
  let P := solve_height_difference
  D - P = 4 := by
  sorry

end parker_shorter_than_daisy_l620_620701


namespace greatest_three_digit_multiple_of_17_l620_620361

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (n % 17 = 0) ∧ ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ (m % 17 = 0) → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620361


namespace greatest_three_digit_multiple_of_17_l620_620295

def is_multiple_of (n : ℕ) (k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, is_three_digit n ∧ is_multiple_of n 17 → n = 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620295


namespace opposite_of_cubic_root_of_8_l620_620745

theorem opposite_of_cubic_root_of_8 : -(Real.cbrt 8) = -2 := by
  sorry

end opposite_of_cubic_root_of_8_l620_620745


namespace angle_equality_l620_620124

variables {Point Circle : Type}
variables (K O1 O2 P1 P2 Q1 Q2 M1 M2 : Point)
variables (W1 W2 : Circle)
variables (midpoint : Point → Point → Point)
variables (is_center : Point → Circle → Prop)
variables (intersects_at : Circle → Circle → Point → Prop)
variables (common_tangent_points : Circle → Circle → (Point × Point) × (Point × Point) → Prop)
variables (intersect_circle_at : Circle → Line → Point → Point → Prop)
variables (angle : Point → Point → Point → ℝ) -- to denote the angle measure between three points

-- Conditions
axiom K_intersection : intersects_at W1 W2 K
axiom O1_center : is_center O1 W1
axiom O2_center : is_center O2 W2
axiom tangents_meet_at : common_tangent_points W1 W2 ((P1, Q1), (P2, Q2))
axiom M1_midpoint : M1 = midpoint P1 Q1
axiom M2_midpoint : M2 = midpoint P2 Q2

-- The statement to prove
theorem angle_equality : angle O1 K O2 = angle M1 K M2 := 
  sorry

end angle_equality_l620_620124


namespace cat_food_sufficiency_l620_620514

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end cat_food_sufficiency_l620_620514


namespace avg_chem_math_l620_620210

-- Given conditions
variables (P C M : ℕ)
axiom total_marks : P + C + M = P + 130

-- The proof problem
theorem avg_chem_math : (C + M) / 2 = 65 :=
by sorry

end avg_chem_math_l620_620210


namespace daughter_age_is_10_l620_620827

variable (D : ℕ)

-- Conditions
def father_current_age (D : ℕ) : ℕ := 4 * D
def father_age_in_20_years (D : ℕ) : ℕ := father_current_age D + 20
def daughter_age_in_20_years (D : ℕ) : ℕ := D + 20

-- Theorem statement
theorem daughter_age_is_10 :
  father_current_age D = 40 →
  father_age_in_20_years D = 2 * daughter_age_in_20_years D →
  D = 10 :=
by
  -- Here would be the proof steps to show that D = 10 given the conditions
  sorry

end daughter_age_is_10_l620_620827


namespace scale_model_height_l620_620643

theorem scale_model_height 
  (scale_ratio : ℚ) (actual_height : ℚ)
  (h_ratio : scale_ratio = 1/30)
  (h_actual_height : actual_height = 305) 
  : Int.ceil (actual_height * scale_ratio) = 10 := by
  -- Define variables and the necessary conditions
  let height_of_model: ℚ := actual_height * scale_ratio
  -- Skip the proof steps
  sorry

end scale_model_height_l620_620643


namespace greatest_three_digit_multiple_of_17_l620_620338

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l620_620338


namespace David_twice_Rosy_l620_620878

theorem David_twice_Rosy :
  ∃ Y : ℕ, 
    R = 12 ∧ D = R + 18 ∧ (D + Y = 2 * (R + Y)) := 
begin
  let R := 12,
  let D := R + 18,
  use 6,
  split,
  { refl, },
  split,
  { refl, },
  rw [add_comm Y, add_comm (2 * (R + Y))],
  norm_num,
end

end David_twice_Rosy_l620_620878


namespace james_second_hour_distance_l620_620667

variables (x : ℝ) 

def james_bike_ride := 
  ∃ x, 
    x + 1.20 * x + 1.50 * x = 74 ∧ 
    1.20 * x = 24

theorem james_second_hour_distance : ∃ x, james_bike_ride x :=
by {
  sorry
}

end james_second_hour_distance_l620_620667


namespace solve_max_a1_l620_620578

noncomputable def max_possible_a1 (a : ℕ → ℝ) : ℝ :=
  if (∀ n ≥ 2, a (n + 1) = a n - a (n - 1) + n) ∧ (a 2 * a 2022 = 1) then
    a 2 + (1 / a 2) - 2023
  else
    0

theorem solve_max_a1 (a : ℕ → ℝ) (h_nonneg : ∀ n, 0 ≤ a n)
  (h_rec : ∀ n ≥ 2, a (n + 1) = a n - a (n - 1) + n)
  (h_cond : a 2 * a 2022 = 1) :
  max_possible_a1 a = 4051 / 2025 :=
sorry

end solve_max_a1_l620_620578


namespace sum_b_is_8_l620_620529

theorem sum_b_is_8 (b_2 b_3 b_4 b_5 b_6 b_7 : ℤ) 
  (h1 : 0 ≤ b_2 ∧ b_2 < 2)
  (h2 : 0 ≤ b_3 ∧ b_3 < 3)
  (h3 : 0 ≤ b_4 ∧ b_4 < 4)
  (h4 : 0 ≤ b_5 ∧ b_5 < 5)
  (h5 : 0 ≤ b_6 ∧ b_6 < 6)
  (h6 : 0 ≤ b_7 ∧ b_7 < 7)
  (distinct : b_2 ≠ b_3 ∧ b_2 ≠ b_4 ∧ b_2 ≠ b_5 ∧ b_2 ≠ b_6 ∧ b_2 ≠ b_7 ∧
              b_3 ≠ b_4 ∧ b_3 ≠ b_5 ∧ b_3 ≠ b_6 ∧ b_3 ≠ b_7 ∧
              b_4 ≠ b_5 ∧ b_4 ≠ b_6 ∧ b_4 ≠ b_7 ∧
              b_5 ≠ b_6 ∧ b_5 ≠ b_7 ∧
              b_6 ≠ b_7)
  (h : (11 / 13 : ℚ) = b_2 / 2.factorial + b_3 / 3.factorial + b_4 / 4.factorial +
                      b_5 / 5.factorial + b_6 / 6.factorial + b_7 / 7.factorial) : 
  b_2 + b_3 + b_4 + b_5 + b_6 + b_7 = 8 :=
by
  sorry

end sum_b_is_8_l620_620529


namespace hyperbola_focal_length_l620_620184

theorem hyperbola_focal_length (m : ℝ) (hm : -5 < m ∧ m < 20) :
  let a² := m + 5
      b² := 20 - m
      c := Real.sqrt (a² + b²)
  in 2 * c = 10 :=
by
  let a² := m + 5
  let b² := 20 - m
  let c := Real.sqrt (a² + b²)
  rw [←Real.sqrt_add (by linarith) (by linarith)]
  suffices h : a² + b² = 25 by simpa [c]
  sorry

end hyperbola_focal_length_l620_620184


namespace greatest_three_digit_multiple_of_17_l620_620291

def is_multiple_of (n : ℕ) (k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, is_three_digit n ∧ is_multiple_of n 17 → n = 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620291


namespace ab_equals_six_l620_620038

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
by
  sorry

end ab_equals_six_l620_620038


namespace vertices_of_trapezoid_l620_620121

-- Definitions and theorems used in Lean 4 equivalent to the math problem conditions.
variables {A B C I P Q : Type*} [Incircle I A B C] [Circumcircle A I B P] [Circumcircle A I B Q]

theorem vertices_of_trapezoid (h1 : Incircle I A B C)
  (h2 : Circumcircle A I B P)
  (h3 : Circumcircle A I B Q)
  (hP : P ≠ A)
  (hQ : Q ≠ B) :
  is_trapezoid {A, B, P, Q} :=
sorry

end vertices_of_trapezoid_l620_620121


namespace geometric_progression_solution_l620_620554

theorem geometric_progression_solution (p : ℝ) :
  (3 * p + 1)^2 = (9 * p + 10) * |p - 3| ↔ p = -1 ∨ p = 29 / 18 :=
by
  sorry

end geometric_progression_solution_l620_620554


namespace smallest_k_for_perfect_cube_l620_620739

noncomputable def isPerfectCube (m : ℕ) : Prop :=
  ∃ n : ℤ, n^3 = m

theorem smallest_k_for_perfect_cube :
  ∃ k : ℕ, k > 0 ∧ (∀ m : ℕ, ((2^4) * (3^2) * (5^5) * k = m) → isPerfectCube m) ∧ k = 60 :=
sorry

end smallest_k_for_perfect_cube_l620_620739


namespace max_xy_of_perpendicular_l620_620968

open Real

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (1, x - 1)
noncomputable def vector_b (y : ℝ) : ℝ × ℝ := (y, 2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 

theorem max_xy_of_perpendicular (x y : ℝ) 
  (h_perp : dot_product (vector_a x) (vector_b y) = 0) : xy ≤ 1/2 :=
by
  sorry

end max_xy_of_perpendicular_l620_620968


namespace parabola_focus_vector_sum_l620_620943

open_locale real

noncomputable def focus_of_parabola (x : ℝ) (y : ℝ) : Prop :=
  x * x = 8 * y

noncomputable def point_a (x1 y1 : ℝ) : Prop :=
  focus_of_parabola x1 y1

noncomputable def point_b (x2 y2 : ℝ) : Prop :=
  focus_of_parabola x2 y2

noncomputable def point_c (x3 y3 : ℝ) : Prop :=
  focus_of_parabola x3 y3

noncomputable def vector_sum_zero (FA FB FC : ℝ × ℝ) : Prop :=
  FA.fst + FB.fst + FC.fst = 0 ∧ FA.snd + FB.snd + FC.snd = 0

noncomputable def vector_magnitudes_sum (FA FB FC : ℝ × ℝ) : ℝ :=
  abs FA.fst + abs FA.snd + abs FB.fst + abs FB.snd + abs FC.fst + abs FC.snd

theorem parabola_focus_vector_sum
  (x1 y1 x2 y2 x3 y3: ℝ)
  (F : ℝ × ℝ := (0, 2))
  (FA : ℝ × ℝ := (x1, y1 - 2))
  (FB : ℝ × ℝ := (x2, y2 - 2))
  (FC : ℝ × ℝ := (x3, y3 - 2))
  (hA : point_a x1 y1)
  (hB : point_b x2 y2)
  (hC : point_c x3 y3)
  (hSum : vector_sum_zero FA FB FC)
  : vector_magnitudes_sum FA FB FC = 12 := 
sorry

end parabola_focus_vector_sum_l620_620943


namespace possible_orange_cells_l620_620435

theorem possible_orange_cells :
  ∃ (n : ℕ), n = 2021 * 2020 ∨ n = 2022 * 2020 := 
sorry

end possible_orange_cells_l620_620435


namespace pirates_can_escape_l620_620836

/-- 
  There are a countably infinite number of pirates. 
  Each pirate wears a hat of one of 10 different colors. 
  The pirates are aligned such that each pirate can see the hats of all pirates in front of them but not their own. 
  Each pirate will whisper a color to the mathematician. 
  If only a finite number of pirates guess incorrectly, they are all freed. 
  Prove that the pirates can come up with a strategy to ensure their release.
-/
theorem pirates_can_escape (num_colors : ℕ) (h_num_colors : num_colors = 10) : 
  ∃ strategy : (ℕ → fin num_colors) → (ℕ → fin num_colors), 
  (∀ hat_colors : ℕ → fin num_colors, 
  (∀ n, strategy hat_colors n = hat_colors n ∨ 
    ∃ k, ∀ m ≥ k, strategy hat_colors m = hat_colors m)) :=
begin
  -- proof to be provided
  sorry
end

end pirates_can_escape_l620_620836


namespace cost_per_set_cheaper_option_l620_620781

variable (x : ℕ) {cost_B cost_A : ℕ}

/-- Given that the cost per set from Shop A is $10 more than from Shop B. --/
def cost_Shops (x : ℕ) : Prop := cost_A = x + 10 ∧ cost_B = x

/-- Given that the number of sets rented for $600 from Shop A equals the number of sets rented for $500 from Shop B. --/
def num_sets_eq (x : ℕ) : Prop := 600 / (x + 10) = 500 / x

/-- Prove that the cost per set of costumes is $60 from Shop A and $50 from Shop B --/
theorem cost_per_set (x : ℕ) (h1 : cost_Shops x) (h2 : num_sets_eq x) : cost_A = 60 ∧ cost_B = 50 :=
by 
  sorry

/-- Prove that renting 20 sets of costumes is cheaper from Shop B than from Shop A when A offers 10% discount for more than 10 sets --/
theorem cheaper_option (x : ℕ) (h1 : cost_Shops x) (h2 : num_sets_eq x) (h3 : cost_A = 60) (h4 : cost_B = 50) : 50 * 20 < 60 * 20 * 0.9 :=
by 
  sorry

end cost_per_set_cheaper_option_l620_620781


namespace probability_not_both_red_l620_620912

open Nat

/-- Considering an urn with 3 red balls and 2 white balls, 
 the probability of drawing two balls that are not both red from it -/
theorem probability_not_both_red : 
  let total_ways_draw_2 := choose 5 2
      ways_draw_2_red := choose 3 2
      p_A := (ways_draw_2_red : ℚ) / total_ways_draw_2
  in 
  1 - p_A = 7 / 10 :=
by {
  -- Definitions
  let total_ways_draw_2 := choose 5 2,
  let ways_draw_2_red := choose 3 2,
  let p_A := (ways_draw_2_red : ℚ) / total_ways_draw_2,
  
  -- Calculation
  have h1 : total_ways_draw_2 = 10 := by sorry,
  have h2 : ways_draw_2_red = 3 := by sorry,
  have h3 : p_A = 3 / 10 := by sorry,
 
  -- Proof
  have h4 : 1 - p_A = 1 - (3 / 10) := by sorry,
  have h5 : 1 - (3 / 10) = 7 / 10 := by sorry,
  exact h5,
}

end probability_not_both_red_l620_620912


namespace train_crosses_platform_in_39_seconds_l620_620813

def length_of_train : ℝ := 300
def time_to_cross_signal_pole : ℝ := 8
def length_of_platform : ℝ := 1162.5

theorem train_crosses_platform_in_39_seconds :
  let speed := length_of_train / time_to_cross_signal_pole,
      total_distance := length_of_train + length_of_platform,
      time_to_cross_platform := total_distance / speed
  in time_to_cross_platform = 39 := by
  sorry

end train_crosses_platform_in_39_seconds_l620_620813


namespace quadratic_has_solution_l620_620710

theorem quadratic_has_solution (a b : ℝ) : ∃ x : ℝ, (a^6 - b^6) * x^2 + 2 * (a^5 - b^5) * x + (a^4 - b^4) = 0 :=
  by sorry

end quadratic_has_solution_l620_620710


namespace greatest_three_digit_multiple_of_17_l620_620369

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 100 ≤ n ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → m ≤ n) :=
begin
  sorry
end

end greatest_three_digit_multiple_of_17_l620_620369


namespace vertical_asymptote_sum_l620_620737

theorem vertical_asymptote_sum : 
  let f (x : ℝ) := (4 * x ^ 2 + 1) / (6 * x ^ 2 + 7 * x + 3)
  ∃ p q : ℝ, (6 * p ^ 2 + 7 * p + 3 = 0) ∧ (6 * q ^ 2 + 7 * q + 3 = 0) ∧ p + q = -11 / 6 :=
by
  let f (x : ℝ) := (4 * x ^ 2 + 1) / (6 * x ^ 2 + 7 * x + 3)
  exact sorry

end vertical_asymptote_sum_l620_620737


namespace taxi_ride_total_cost_l620_620458

theorem taxi_ride_total_cost :
  let base_fee := 1.50
  let cost_per_mile := 0.25
  let distance1 := 5
  let distance2 := 8
  let distance3 := 3
  let cost1 := base_fee + distance1 * cost_per_mile
  let cost2 := base_fee + distance2 * cost_per_mile
  let cost3 := base_fee + distance3 * cost_per_mile
  cost1 + cost2 + cost3 = 8.50 := sorry

end taxi_ride_total_cost_l620_620458


namespace cars_left_with_tom_l620_620225

theorem cars_left_with_tom : 
  let totalCars := 10 * 5,
      carsGiven := 2 * (1 / 5 * totalCars) 
  in totalCars - carsGiven = 30 := 
by 
  -- Definitions from conditions
  let totalCars : ℕ := 10 * 5
  let carsGiven : ℕ := 2 * (1 / 5 : ℝ) * totalCars

  -- The final statement
  have h1 : totalCars - carsGiven = 50 - 20 := sorry

  -- The known correct answer
  exact h1.trans (by norm_num)

end cars_left_with_tom_l620_620225


namespace quadratic_solution_l620_620758

theorem quadratic_solution (x a b : ℂ) (hx : 5 * x^2 - 4 * x + 15 = 0)
  (h_form : x = a + b * complex.I) : a.re + (b.re ^ 2 + b.im ^ 2) = 162 / 50 :=
by
  sorry

end quadratic_solution_l620_620758


namespace correct_option_is_D_l620_620412

-- Define the expressions to be checked
def exprA (x : ℝ) := 3 * x + 2 * x = 5 * x^2
def exprB (x : ℝ) := -2 * x^2 * x^3 = 2 * x^5
def exprC (x y : ℝ) := (y + 3 * x) * (3 * x - y) = y^2 - 9 * x^2
def exprD (x y : ℝ) := (-2 * x^2 * y)^3 = -8 * x^6 * y^3

theorem correct_option_is_D (x y : ℝ) : 
  ¬ exprA x ∧ 
  ¬ exprB x ∧ 
  ¬ exprC x y ∧ 
  exprD x y := by
  -- The proof would be provided here
  sorry

end correct_option_is_D_l620_620412


namespace max_value_of_c_l620_620990

theorem max_value_of_c (a b c : ℝ) (h1 : 2^a + 2^b = 2^(a + b)) (h2 : 2^a + 2^b + 2^c = 2^(a + b + c)) : c = 2 - Real.log 3 / Real.log 2 := 
sorry

end max_value_of_c_l620_620990


namespace find_alpha_l620_620656

noncomputable section

open Real 

def curve_C1 (x y : ℝ) : Prop := x + y = 1
def curve_C2 (x y φ : ℝ) : Prop := x = 2 + 2 * cos φ ∧ y = 2 * sin φ 

def polar_coordinate_eq1 (ρ θ : ℝ) : Prop := ρ * sin (θ + π / 4) = sqrt 2 / 2
def polar_coordinate_eq2 (ρ θ : ℝ) : Prop := ρ = 4 * cos θ

def line_l (ρ θ α : ℝ)  (hα: α > 0 ∧ α < π / 2) : Prop := θ = α ∧ ρ > 0 

def OB_div_OA_eq_4 (ρA ρB α : ℝ) : Prop := ρB / ρA = 4

theorem find_alpha (α : ℝ) (hα: α > 0 ∧ α < π / 2)
  (h₁: ∀ (x y ρ θ: ℝ), curve_C1 x y → polar_coordinate_eq1 ρ θ) 
  (h₂: ∀ (x y φ ρ θ: ℝ), curve_C2 x y φ → polar_coordinate_eq2 ρ θ) 
  (h₃: ∀ (ρ θ: ℝ), line_l ρ θ α hα) 
  (h₄: ∀ (ρA ρB : ℝ), OB_div_OA_eq_4 ρA ρB α → ρA = 1 / (cos α + sin α) ∧ ρB = 4 * cos α ): 
  α = 3 * π / 8 :=
by
  sorry

end find_alpha_l620_620656


namespace find_ellipse_equation_find_eccentricity_range_l620_620949

noncomputable def ellipse_equation : Bool :=
  let a := Real.sqrt 12
  let b := Real.sqrt 3
  (∀ x y : Real, x^2 / 12 + y^2 / 3 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1)

theorem find_ellipse_equation :
  ∀ (a b : Real), (a > b ∧ b > 0) →
    let c : Real := 3
    let perimeter : Real := 4 * Real.sqrt 3 + 6
    (2 * a + 2 * c = perimeter ∧ a * a = b * b + c * c) →
    ellipse_equation =
    True := by sorry

theorem find_eccentricity_range :
  ∀ (a b : Real), (a > b ∧ b > 0) →
    let foci_l := Point.mk (-3) 0
    let foci_r := Point.mk 3 0
    let |k| > Real.sqrt 2 / 4
    (x y : Real), (x^2 / a^2 + y^2 / b^2 = 1 ∧ y = k * x) →
    let A := Point.mk x (k * x)
    let B := Point.mk (-x) (-k * x)
    let diameter : Real := 2 * x
    let r_focus := foci_r
    let e_range := Real.sqrt 2 / 2 < (Real.sqrt (a^2 - b^2) / a) ∧ (Real.sqrt (a^2 - b^2) / a) < Real.sqrt 3 / 2
    e_range := True := by sorry

end find_ellipse_equation_find_eccentricity_range_l620_620949


namespace a_seq_arithmetic_a_seq_formula_T_n_bound_l620_620932

noncomputable def a_seq : ℕ → ℝ
| 1     := 3
| (n+1) := sorry -- typically, one would define how a_{n+1} relates to a_n.

def S_n (n : ℕ) : ℝ :=
  (n + 1) * (a_seq n + 1) / 2 - 1

def T_n (n : ℕ) : ℝ :=
  (1 / 2) * ((1 / 3) - (1 / (2 * n + 3)))

theorem a_seq_arithmetic :
  ∀ n : ℕ, 2 * (a_seq (n + 1)) = a_seq (n + 2) + a_seq n :=
sorry

theorem a_seq_formula :
  a_seq = λ n, 2 * n + 1 :=
sorry

theorem T_n_bound (M : ℝ) :
  (∃ M : ℝ, ∀ n : ℕ, n > 0 → T_n n ≤ M) ∧ (∀ n : ℕ, n > 0 → T_n n ≤ 1 / 6) :=
sorry

end a_seq_arithmetic_a_seq_formula_T_n_bound_l620_620932


namespace greatest_3_digit_multiple_of_17_l620_620386

theorem greatest_3_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ (∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ 17 ∣ m → n ≥ m) ∧ n = 986 := 
begin
  sorry
end

end greatest_3_digit_multiple_of_17_l620_620386


namespace greatest_three_digit_multiple_of_17_l620_620289

def is_multiple_of (n : ℕ) (k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, is_three_digit n ∧ is_multiple_of n 17 → n = 986 := by
  sorry

end greatest_three_digit_multiple_of_17_l620_620289


namespace range_H_l620_620538

def H (x : ℝ) : ℝ := |x + 2| - |x - 4| + 3

theorem range_H : set.range H = set.Icc 5 9 :=
by sorry

end range_H_l620_620538


namespace range_function_1_l620_620959

theorem range_function_1 (y : ℝ) : 
  (∃ x : ℝ, x ≥ -1 ∧ y = (1/3) ^ x) ↔ (0 < y ∧ y ≤ 3) :=
sorry

end range_function_1_l620_620959


namespace O_lies_on_reflected_line_l620_620106

open Real EuclideanGeometry

variables {A B C D O : Point}

-- Definitions/Conditions from the problem
def is_cyclic_quadrilateral (A B C D : Point) : Prop :=
  ∃ (O : Point), ∠A + ∠C = 180 ∧ ∠B + ∠D = 180

def perpendicular (X Y : Line) : Prop :=
  ∀ (P Q R : Point), (P ∈ X ∧ Q ∈ Y ∧ R ∈ X ∩ Y) → ∠PQR = 90

def circumcenter {A B C D O : Point} (A B C D : Point) : Point := O

def reflection_across_angle_bisector (A B C D : Point) : Line :=
  let reflection_line := -- some reflection logic
  reflection_line

-- Main statement to be proved
theorem O_lies_on_reflected_line (A B C D O : Point)
  (h1 : is_cyclic_quadrilateral A B C D)
  (h2 : perpendicular (line_through_points A C) (line_through_points B D))
  (h3 : circumcenter A B C D = O)
  : let g := reflection_across_angle_bisector A B C D in O ∈ g :=
by
  sorry

end O_lies_on_reflected_line_l620_620106


namespace product_of_ab_l620_620048

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end product_of_ab_l620_620048


namespace calculate_expression_l620_620487

theorem calculate_expression : 
  (10 - 9 * 8 + 7^2 / 2 - 3 * 4 + 6 - 5 = -48.5) :=
by
  -- Proof goes here
  sorry

end calculate_expression_l620_620487


namespace cat_food_sufficiency_l620_620508

theorem cat_food_sufficiency (L S : ℝ) (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end cat_food_sufficiency_l620_620508


namespace quadratic_solution_exists_for_any_a_b_l620_620712

theorem quadratic_solution_exists_for_any_a_b (a b : ℝ) : 
  ∃ x : ℝ, (a^6 - b^6)*x^2 + 2*(a^5 - b^5)*x + (a^4 - b^4) = 0 := 
by
  -- The proof would go here
  sorry

end quadratic_solution_exists_for_any_a_b_l620_620712


namespace frac_equality_l620_620568

variables (a b : ℚ) -- Declare the variables as rational numbers

-- State the theorem with the given condition and the proof goal
theorem frac_equality (h : a / b = 2 / 3) : a / (a + b) = 2 / 5 :=
by
  sorry -- proof goes here

end frac_equality_l620_620568


namespace triathlete_average_speed_l620_620841

theorem triathlete_average_speed :
  let swimming_speed := 2
  let biking_speed := 15
  let running_speed := 12
  let harmonic_mean := 
    (3 : ℝ) / ((1 / (swimming_speed : ℝ)) + (1 / (biking_speed : ℝ)) + (1 / (running_speed : ℝ)))
  abs (harmonic_mean - 5) < 0.5 :=
by
  let swimming_speed := 2
  let biking_speed := 15
  let running_speed := 12
  let harmonic_mean := (3 : ℝ) / ((1 / (swimming_speed : ℝ)) + (1 / (biking_speed : ℝ)) + (1 / (running_speed : ℝ)))
  have h : harmonic_mean = (60 : ℝ) / 13 :=
    sorry -- calculation step
  rw h
  -- approximation step
  have approx : abs ((60 / 13) - 5) < 0.5 :=
    sorry -- actual proof required for completion
  exact approx

end triathlete_average_speed_l620_620841


namespace determine_consistency_by_variance_l620_620083

-- Assume we have two groups of results
variable (group1 group2 : List ℝ) (h₁ : group1.length = 10) (h₂ : group2.length = 10)

-- We define what it means for one group to be more consistent than the other
def more_consistent (group1_variance group2_variance : ℝ) : Prop :=
  group1_variance < group2_variance

-- The theorem stating that to determine more consistent results, we need to consider variance
theorem determine_consistency_by_variance : 
  ∀ (group1_result group2_result: List ℝ) 
  (h₁ : group1_result.length = 10) 
  (h₂ : group2_result.length = 10),
  more_consistent (variance group1_result) (variance group2_result) = 
    more_consistent (group1_variance) (group2_variance) := 
sorry

end determine_consistency_by_variance_l620_620083


namespace area_of_ABC_l620_620096

-- Define the side lengths of the triangle.
def AB : ℝ := real.sqrt 3
def AC : ℝ := real.sqrt 5
def BC : ℝ := 2

-- Define a function to calculate the area of the triangle using Heron's formula for simplicity.
noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2 in
  real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Define the theorem statement
theorem area_of_ABC : area_of_triangle AB AC BC = real.sqrt 11 / 2 :=
sorry

end area_of_ABC_l620_620096


namespace max_min_values_l620_620537

def f (x : ℝ) : ℝ := x^2 - 2*x - 1

theorem max_min_values :
  ∃ (max min : ℝ), max = 14 ∧ min = -2 ∧ 
  (∀ (x ∈ set.Icc (-3 : ℝ) 2), f x ≤ max ∧ f x ≥ min) := by
  sorry

end max_min_values_l620_620537


namespace product_of_sequence_l620_620865

theorem product_of_sequence : (∏ n in Finset.range 8, (1 - 1 / (n + 1)^2)) = (9 / 8) := 
by 
  sorry

end product_of_sequence_l620_620865


namespace parabola_focus_distance_l620_620734

theorem parabola_focus_distance (p : ℝ) (h : 2 * p = 8) : p = 4 :=
  by
  sorry

end parabola_focus_distance_l620_620734
