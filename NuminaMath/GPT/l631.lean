import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Invertible
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Ratio
import Mathlib.Analysis.SpecialFunctions
import Mathlib.Analysis.SpecialFunctions.Abs
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Integrals
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Intervals.Basic
import Mathlib.Geometry
import Mathlib.Geometry.Euclidean.Sphere.Angle
import Mathlib.MeasureTheory.Probability
import Mathlib.MeasureTheory.ProbabilityMassFunction
import Mathlib.NumberTheory.GCD
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Pigeonhole
import Mathlib.Topology.Basic
import Real

namespace mapping_f_of_neg2_and_3_l631_631078

-- Define the mapping f
def f (x y : ℝ) : ℝ × ℝ := (x + y, x * y)

-- Define the given point
def p : ℝ × ℝ := (-2, 3)

-- Define the expected corresponding point
def expected_p : ℝ × ℝ := (1, -6)

-- The theorem stating the problem to be proved
theorem mapping_f_of_neg2_and_3 :
  f p.1 p.2 = expected_p := by
  sorry

end mapping_f_of_neg2_and_3_l631_631078


namespace sound_pressure_ratio_l631_631601

theorem sound_pressure_ratio (P_ref : ℝ) (P_ref_val : P_ref = 20 * 10^(-6)) 
(SPL_day SPL_night : ℝ) (SPL_day_val : SPL_day = 50) (SPL_night_val : SPL_night = 30) :
  let P (SPL : ℝ) := 20 * 10^(SPL / 20)
  in P SPL_day / P SPL_night = 10 := by
{
  sorry
}

end sound_pressure_ratio_l631_631601


namespace max_value_fraction_l631_631079

theorem max_value_fraction (n : ℕ) (x : Fin n → ℝ) 
  (h1 : n ≥ 3) 
  (h2 : ∑ k, x k = 0)
  (h3 : ∑ k, (x k)^2 = 1)
  (hx_nonzero : ∃ k, x k ≠ 0) : 
  ( (∑ k, (x k)^3)^2 / (∑ k, (x k)^2)^3 ) 
  ≤ (n-2) / Real.sqrt (n*(n-1)) :=
sorry

end max_value_fraction_l631_631079


namespace all_roots_equal_l631_631076

theorem all_roots_equal
  (P : ℝ[X]) -- P is a polynomial with real coefficients
  (h_nonconstant : P.degree > 0) -- P is non-constant
  (h_real_roots : ∀ x, P.is_root x → x ∈ ℝ) -- all roots of P are real
  (Q : ℝ[X]) -- Q is a polynomial with real coefficients
  (h_Q_property : ∀ x : ℝ, P.eval x ^ 2 = P.eval (Q.eval x)) -- P^2(x) = P(Q(x)) for all real x
  : ∃ (r : ℝ) (A : ℝ) (d : ℕ), P = A * (X - C r) ^ d := -- all roots of P are equal
  sorry

end all_roots_equal_l631_631076


namespace sin_double_angle_tan_sub_angle_l631_631101

noncomputable def alpha : Real := sorry

def sin_alpha_cond (α : Real) : Prop := sin α = (Real.sqrt 5) / 5 ∧ α ∈ Ioo (π / 2) π

theorem sin_double_angle (α : Real) (h : sin_alpha_cond α) : sin (2 * α) = -4 / 5 :=
by sorry

theorem tan_sub_angle (α : Real) (h : sin_alpha_cond α) : tan ((π / 4) - α) = 3 / 2 :=
by sorry

end sin_double_angle_tan_sub_angle_l631_631101


namespace decimal_digit_101st_place_l631_631496

theorem decimal_digit_101st_place (n : ℕ) (h : n = 101) : 
  (∀ m, (m = 7 / 26) → (decimal_digit m n) = 3) := 
begin
  intros m hm,
  have repeating_block : ℕ → ℕ := λ k, match (k % 6) with
    | 0 := 2
    | 1 := 6
    | 2 := 9
    | 3 := 2
    | 4 := 3
    | 5 := 0
    end,
  have one_digit := (repeating_block (101 - 1) % 6),
  have digit := match one_digit with
    | 0 := 2
    | 1 := 6
    | 2 := 9
    | 3 := 2
    | 4 := 3
    | 5 := 0
    end,
  exact digit = 3,
end

end decimal_digit_101st_place_l631_631496


namespace find_initial_marbles_l631_631013

-- Definitions based on conditions
def loses_to_street (initial_marbles : ℕ) : ℕ := initial_marbles - (initial_marbles * 60 / 100)
def loses_to_sewer (marbles_after_street : ℕ) : ℕ := marbles_after_street / 2

-- The given number of marbles left
def remaining_marbles : ℕ := 20

-- Proof statement
theorem find_initial_marbles (initial_marbles : ℕ) : 
  loses_to_sewer (loses_to_street initial_marbles) = remaining_marbles -> 
  initial_marbles = 100 :=
by
  sorry

end find_initial_marbles_l631_631013


namespace rationalize_denominator_l631_631227

theorem rationalize_denominator :
  let A := -12
  let B := 7
  let C := 9
  let D := 13
  let E := 5 in
  B < D ∧
  (12/5 * Real.sqrt 7) = ((1:ℚ) * Real.sqrt B / E * A) * (-1) ∧
  (9/5 * Real.sqrt 13) = (1:ℚ * Real.sqrt D / E * C) ∧
  (A + B + C + D + E = 22) :=
by
  sorry

end rationalize_denominator_l631_631227


namespace corn_growth_first_week_l631_631787

theorem corn_growth_first_week (x : ℝ) (h1 : x + 2*x + 8*x = 22) : x = 2 :=
by
  sorry

end corn_growth_first_week_l631_631787


namespace mandy_accepted_schools_l631_631990

theorem mandy_accepted_schools (total_schools : ℕ) (fraction_applied : ℚ) (fraction_accepted : ℚ) 
    (h_total : total_schools = 42) (h_fraction_applied : fraction_applied = 1/3) 
    (h_fraction_accepted : fraction_accepted = 1/2) : 
    let applied_schools := total_schools * fraction_applied in
    let accepted_schools := applied_schools * fraction_accepted in
    accepted_schools = 7 :=
by
  sorry

end mandy_accepted_schools_l631_631990


namespace hyperbola_eccentricity_l631_631471

theorem hyperbola_eccentricity
  (m : ℝ) (h_m : m > 0)
  (focus_hyperbola : (0, 2) = (0, real.sqrt (m + 3))) :
  let a := real.sqrt m,
      c := 2 in
  let e := c / a in
  e = 2 :=
by
  sorry

end hyperbola_eccentricity_l631_631471


namespace train_crossing_platform_l631_631336

theorem train_crossing_platform (L_t t_t L_p : ℕ)
  (L_t_eq : L_t = 1200) (t_t_eq : t_t = 120) (L_p_eq : L_p = 600) :
  let v := L_t / t_t in
  let total_distance := L_t + L_p in
  let t_p := total_distance / v in
  t_p = 180 :=
by
  sorry

end train_crossing_platform_l631_631336


namespace values_of_a_l631_631414

theorem values_of_a (a : ℝ) :
  (∀ x : ℝ, 4 * x^2 - 8 * |x| + (2 * a + |x| + x)^2 = 4 -> (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  4 * x₁^2 - 8 * |x₁| + (2 * a + |x₁| + x₁)^2 = 4 ∧ 4 * x₂^2 - 8 * |x₂| + (2 * a + |x₂| + x₂)^2 = 4)) ->
  a ∈ set.Icc (-3) (-real.sqrt 2) ∪ set.Icc (-1) 1 ∪ set.Icc 1 (real.sqrt 2) :=
begin
  sorry,
end

end values_of_a_l631_631414


namespace find_ellipse_and_line_eq_l631_631830

noncomputable def ellipse_eq (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def tangent_line_eq (x y b : ℝ) : Prop :=
  x + y - b.sqrt = 0

noncomputable def circle_eq (r x y : ℝ) : Prop := 
  x^2 + y^2 = r^2

noncomputable def line_through_point (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 4)

noncomputable def final_line_eq (x y : ℝ) : Prop :=
  x - 4 * y - 4 = 0

theorem find_ellipse_and_line_eq :
  ∃ a b : ℝ,
  a > 0 ∧ b > 0 ∧ ellipse_eq a b x y ∧ eccentricity a b = 1/2 ∧
  (tangent_line_eq x y b) → (b = √3) ∧ (circle_eq b x y) →
  a = 2 ∧ ellipse_eq 2 √3 x y ∧ final_line_eq x y :=
begin
  sorry
end

def eccentricity (a b : ℝ) : ℝ :=
  (a^2 - b^2)^0.5 / a

end find_ellipse_and_line_eq_l631_631830


namespace numbers_with_7_in_1_to_800_l631_631912

theorem numbers_with_7_in_1_to_800 : 
  (card { n ∈ finset.range (800 + 1) | ∃ d ∈ n.digits 10, d = 7 }) = 152 := 
sorry

end numbers_with_7_in_1_to_800_l631_631912


namespace part1_part2_l631_631036

-- Conditions and definitions
def seq (x : ℕ → ℝ) : Prop :=
  x 0 = 1 ∧ ∀ n, 0 < x (n + 1) ∧ x (n + 1) ≤ x n

def series (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum (λ i, (x i)^2 / x (i + 1))

-- Proof for Part (1)
theorem part1 (x : ℕ → ℝ) (hx : seq x) : ∃ n, n ≥ 1 ∧ series x n ≥ 3.999 :=
sorry

-- Proof for Part (2)
theorem part2 : ∃ x : ℕ → ℝ, seq x ∧ ∀ n, n ≥ 1 → series x n < 4 :=
sorry

end part1_part2_l631_631036


namespace students_B_or_A_l631_631585

def total_students := 60
def percent_below_B := 0.4
def percent_B_Bplus := 0.3
def percent_A_Aminus := 0.2
def percent_Aplus := 0.1

theorem students_B_or_A (n : ℕ) (pbB pBBplus pAAminus pAplus : ℝ) :
  n = total_students →
  pbB = percent_below_B →
  pBBplus = percent_B_Bplus →
  pAAminus = percent_A_Aminus →
  pAplus = percent_Aplus →
  n * (pBBplus + pAAminus) = 30 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h3, h4]
  norm_num
  sorry

end students_B_or_A_l631_631585


namespace range_f_l631_631621

noncomputable def f (x : ℝ) : ℝ := 2^(-(abs x) + 1)

theorem range_f : set.range f = {y : ℝ | 0 < y ∧ y ≤ 2} :=
by
  sorry

end range_f_l631_631621


namespace evaluate_expression_when_c_eq_4_and_k_eq_2_l631_631805

theorem evaluate_expression_when_c_eq_4_and_k_eq_2 :
  ( (4^4 - 4 * (4 - 1)^4 + 2) ^ 4 ) = 18974736 :=
by
  -- Definitions
  let c := 4
  let k := 2
  -- Evaluations
  let a := c^c
  let b := c * (c - 1)^c
  let expression := (a - b + k)^c
  -- Proof
  have result : expression = 18974736 := sorry
  exact result

end evaluate_expression_when_c_eq_4_and_k_eq_2_l631_631805


namespace trader_profit_percent_l631_631367

theorem trader_profit_percent (P : ℝ) (h1 : P > 0) :
  let discount_price := 0.80 * P in
  let selling_price := discount_price + 0.60 * discount_price in
  let profit := selling_price - P in
  let profit_percent := (profit / P) * 100 in
  profit_percent = 28 :=
by
  -- definitions
  let discount_price := 0.80 * P
  let selling_price := discount_price + 0.60 * discount_price
  let profit := selling_price - P
  let profit_percent := (profit / P) * 100

  -- proof
  have eq1 : discount_price = 0.80 * P := rfl
  have eq2 : selling_price = discount_price + 0.60 * discount_price := rfl
  have eq3 : profit = selling_price - P := rfl
  have eq4 : profit_percent = (profit / P) * 100 := rfl
  have eq5 : discount_price = 0.80 * P := rfl
  rw [eq5] at eq2
  have eq6 : selling_price = 1.28 * P := by rw [eq5, eq2]; ring
  rw [eq6, eq3] at eq4
  have eq7 : profit = 0.28 * P := by rw [eq6]; linarith
  rw [eq7] at eq4
  have eq8 : profit_percent = 28 := by rw [eq7]; field_simp [h1]; norm_num
  exact eq8

end trader_profit_percent_l631_631367


namespace locus_of_Q_max_area_OPQ_l631_631197

variable {x y: ℝ}

-- Condition 1: Point P is on the ellipse
def on_ellipse (x₀ y₀: ℝ) : Prop := (x₀^2 / 4) + (y₀^2 / 3) = 1

-- Condition 2: Tangents at points M and N intersect at Q
def on_circle (x₁ y₁: ℝ) : Prop := x₁^2 + y₁^2 = 12

-- Proving the equation of the locus of Q (part 1)
theorem locus_of_Q (x₀ y₀ x₁ y₁: ℝ) (h₁: on_ellipse x₀ y₀) (h₂: on_circle x₁ y₁) :
  (x₀ = x₁ / 3) ∧ (y₀ = y₁ / 4) → (x₁^2 / 36) + (y₁^2 / 48) = 1 :=
sorry

-- Proving the maximum area of triangle OPQ (part 2)
theorem max_area_OPQ (x₀ y₀: ℝ) (h₁: on_ellipse x₀ y₀) (h₂: x₀ > 0 ∧ y₀ > 0) :
  let area := (x₀ * y₀) / 2 in area ≤ (sqrt 3) / 2 :=
sorry

end locus_of_Q_max_area_OPQ_l631_631197


namespace rationalize_denominator_l631_631238

theorem rationalize_denominator :
  let A := -12
  let B := 7
  let C := 9
  let D := 13
  let E := 5
  A + B + C + D + E = 22 :=
by
  -- Proof goes here
  sorry

end rationalize_denominator_l631_631238


namespace scheduling_methods_count_l631_631643

theorem scheduling_methods_count :
  ∃ (scheduling : Finset (Fin 5 ⊕ Fin 5 ⊕ Fin 5 → Fin 5))
  (cond : ∀ (f : Fin 5 ⊕ Fin 5 ⊕ Fin 5 → Fin 5), f ∈ scheduling → f Fin.left = f Fin.middle ∧ f Fin.left = f Fin.right ∧ f Fin.middle ≠ f Fin.right),
  scheduling.card = 20 := sorry

end scheduling_methods_count_l631_631643


namespace find_S_l631_631198

-- Definition of greatest integer function
def floor (x : ℝ) : ℤ := int.ofNat ⌊x⌋

-- Defining the sequence S
noncomputable def sequence (n : ℕ) : ℤ := floor (n / ((nat.sqrt (2 * n)) + 1))

-- The main statement
theorem find_S : ∑ i in finset.range 2016, sequence i = 454 :=
by sorry

end find_S_l631_631198


namespace count_numbers_with_digit_7_in_range_l631_631879

theorem count_numbers_with_digit_7_in_range : 
  let numbers_in_range := {n : ℕ | 1 ≤ n ∧ n ≤ 800}
      contains_digit_7 (n : ℕ) : Prop := n.digits 10.contains 7
  in (finset.filter (λ n, contains_digit_7 n) (finset.range 801)).card = 152 :=
by 
  let numbers_in_range := {n : ℕ | 1 ≤ n ∧ n ≤ 800}
  let contains_digit_7 (n : ℕ) : Prop := n.digits 10.contains 7
  have h := (finset.filter (λ n, contains_digit_7 n) (finset.range 801)).card
  sorry

end count_numbers_with_digit_7_in_range_l631_631879


namespace cost_of_asian_postcards_80s_l631_631328

def price_per_postcard (country : String) : Float :=
  if country = "Italy" then 0.07
  else if country = "Germany" then 0.07
  else if country = "Japan" then 0.05
  else if country = "India" then 0.06
  else 0.0

def postcards_80s (country : String) : Nat :=
  if country = "Italy" then 10
  else if country = "Germany" then 12
  else if country = "Japan" then 15
  else if country = "India" then 11
  else 0

def is_asian (country : String) : Bool :=
  country = "Japan" ∨ country = "India"

def total_cost_asian_80s : Float :=
  price_per_postcard "Japan" * postcards_80s "Japan" +
  price_per_postcard "India" * postcards_80s "India"

theorem cost_of_asian_postcards_80s : total_cost_asian_80s = 1.41 :=
  by
  -- To be completed
  sorry

end cost_of_asian_postcards_80s_l631_631328


namespace problem_equivalence_l631_631042

theorem problem_equivalence :
  (1 / Real.sin (Real.pi / 18) - Real.sqrt 3 / Real.sin (4 * Real.pi / 18)) = 4 := 
sorry

end problem_equivalence_l631_631042


namespace rationalize_denominator_l631_631229

theorem rationalize_denominator :
  let A := -12
  let B := 7
  let C := 9
  let D := 13
  let E := 5 in
  B < D ∧
  (12/5 * Real.sqrt 7) = ((1:ℚ) * Real.sqrt B / E * A) * (-1) ∧
  (9/5 * Real.sqrt 13) = (1:ℚ * Real.sqrt D / E * C) ∧
  (A + B + C + D + E = 22) :=
by
  sorry

end rationalize_denominator_l631_631229


namespace ratio_of_male_democrats_l631_631278

theorem ratio_of_male_democrats (F M : ℕ)
  (h1 : F + M = 660)
  (h2 : F / 2 = 110)
  (h3 : (F / 2) + dM = (1 / 3) * 660)
  (h4 : dM = 220 - 110) :
  dM / M = 1 / 4 :=
by
  -- Declare variables
  have hF : F = 220,
    from Nat.mul_right_cancel (h2 ▸ eq.refl (220 * 1)) 2,
  have hM : M = 440,
    from Nat.sub_eq_of_eq_add' h1 ▸ hF ▸ rfl,
  have hTotalDem : (F / 2) + dM = 220,
    from calc (F / 2) + dM = (1 / 3) * 660 : h3
                         ... = 220       : by norm_num,
  have hDM : dM = 110,
    from Nat.sub_eq_of_eq_add' h4,
  calc dM / M = 110 / 440 : by rw [hDM, hM]
             ... = 1 / 4  : by norm_num

end ratio_of_male_democrats_l631_631278


namespace value_of_f_neg_11_over_2_l631_631820

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom periodicity (x : ℝ) : f (x + 2) = - (f x)⁻¹
axiom interval_value (h : 2 ≤ 5 / 2 ∧ 5 / 2 ≤ 3) : f (5 / 2) = 5 / 2

theorem value_of_f_neg_11_over_2 : f (-11 / 2) = 5 / 2 :=
by
  sorry

end value_of_f_neg_11_over_2_l631_631820


namespace find_magnitude_angle_C_minimum_value_CD_l631_631861

variables {A B C a b c : ℝ}
variables m n : ℝ × ℝ
variable D : ℝ
variable CA CB CD : ℝ

-- Defining the problem conditions
def vector_m := (Real.sin A, Real.sin B)
def vector_n := (Real.cos B, Real.cos A)
def AB := (a + b = 2)
def dot_product_eq := (vector_m.1 * vector_n.1 + vector_m.2 * vector_n.2 = Real.sin (2 * C))
def midpoint_D := (D = (a + b) / 2)
def triangle_angle_sum := (A + B = Real.pi - C)
def range_C := (0 < C ∧ C < Real.pi)

-- Proof statements
theorem find_magnitude_angle_C : dot_product_eq → triangle_angle_sum → range_C → (C = Real.pi / 3) :=
sorry

theorem minimum_value_CD : AB → midpoint_D → 
  (CD = Real.sqrt_real (3 / 4)) :=
sorry

end find_magnitude_angle_C_minimum_value_CD_l631_631861


namespace greatest_int_satisfying_inequality_l631_631662

theorem greatest_int_satisfying_inequality : ∃ n : ℤ, (∀ m : ℤ, m^2 - 13 * m + 40 ≤ 0 → m ≤ n) ∧ n = 8 := 
sorry

end greatest_int_satisfying_inequality_l631_631662


namespace correct_choice_for_games_l631_631321
  
-- Define the problem context
def games_preferred (question : String) (answer : String) :=
  question = "Which of the two computer games did you prefer?" ∧
  answer = "Actually I didn’t like either of them."

-- Define the proof that the correct choice is 'either of them'
theorem correct_choice_for_games (question : String) (answer : String) :
  games_preferred question answer → answer = "either of them" :=
by
  -- Provided statement and proof assumptions
  intro h
  cases h
  exact sorry -- Proof steps will be here
  -- Here, the conclusion should be derived from given conditions

end correct_choice_for_games_l631_631321


namespace decimal_digit_101st_place_l631_631495

theorem decimal_digit_101st_place (n : ℕ) (h : n = 101) : 
  (∀ m, (m = 7 / 26) → (decimal_digit m n) = 3) := 
begin
  intros m hm,
  have repeating_block : ℕ → ℕ := λ k, match (k % 6) with
    | 0 := 2
    | 1 := 6
    | 2 := 9
    | 3 := 2
    | 4 := 3
    | 5 := 0
    end,
  have one_digit := (repeating_block (101 - 1) % 6),
  have digit := match one_digit with
    | 0 := 2
    | 1 := 6
    | 2 := 9
    | 3 := 2
    | 4 := 3
    | 5 := 0
    end,
  exact digit = 3,
end

end decimal_digit_101st_place_l631_631495


namespace vectors_parallel_x_value_l631_631522

theorem vectors_parallel_x_value :
  ∀ (x : ℝ), (∀ a b : ℝ × ℝ, a = (2, 1) → b = (4, x+1) → (a.1 / b.1 = a.2 / b.2)) → x = 1 :=
by
  intros x h
  sorry

end vectors_parallel_x_value_l631_631522


namespace cos_angle_PQR_l631_631531

variables (P Q R S : Type*) [is_tetrahedron P Q R S]
          (a b : ℝ)

-- Conditions
def angle_PRS := 90
def angle_PSQ := 90
def angle_QRS := 90
def sin_PQS := a
def sin_PRQ := b

-- Problem statement
theorem cos_angle_PQR (h1: angle_PRS = 90) (h2: angle_PSQ = 90) (h3: angle_QRS = 90)
    (h4 : sin_PQS = a) (h5 : sin_PRQ = b) :
    cos (angle P Q R) = (a^2 - b^2) / 2 :=
sorry

end cos_angle_PQR_l631_631531


namespace largest_coefficient_term_is_fifth_l631_631817

theorem largest_coefficient_term_is_fifth (n : ℕ) (h : n = 7) :
  ∀ r : ℕ, r ≠ 4 → binomial n r < binomial n 4 :=
by 
  sorry

end largest_coefficient_term_is_fifth_l631_631817


namespace remainder_of_sum_div_l631_631722

theorem remainder_of_sum_div by_14 :
  (11057 + 11059 + 11061 + 11063 + 11065 + 11067 + 11069 + 11071 + 11073 + 11075 + 11077) % 14 = 9 :=
by
  sorry

end remainder_of_sum_div_l631_631722


namespace abc_sum_71_l631_631574

theorem abc_sum_71 (a b c : ℝ) (h₁ : ∀ x, (x ≤ -3 ∨ 23 ≤ x ∧ x < 27) ↔ ( (x - a) * (x - b) / (x - c) ≥ 0)) (h₂ : a < b) : 
  a + 2 * b + 3 * c = 71 :=
sorry

end abc_sum_71_l631_631574


namespace tan_theta_l631_631003

noncomputable def triangle := mk_triangle 13 14 15

theorem tan_theta (θ : ℝ) : 
  let sides := (13, 14, 15) in
  let semi_perimeter := (13 + 14 + 15) / 2 in
  let area := Real.sqrt (semi_perimeter * (semi_perimeter - 13) * (semi_perimeter - 14) * (semi_perimeter - 15)) in
  let lines := (λ (x y : ℝ), x + y = semi_perimeter, λ (x y : ℝ), x * y * Real.sin θ = area) in
  let tan_θ := Real.tan θ in
  ∃ p q : ℝ, p + q = semi_perimeter ∧ 
    p * q * Real.sin θ = area → 
    Real.tan (θ) = this_answered_value :=
sorry

end tan_theta_l631_631003


namespace elsa_data_remaining_l631_631422

variable (initial_data : ℕ) (youtube_data : ℕ) (facebook_fraction_num : ℕ) (facebook_fraction_den : ℕ)

def remaining_data (initial_data youtube_data facebook_fraction_num facebook_fraction_den : ℕ) : ℕ :=
  let remaining_after_youtube := initial_data - youtube_data
  let facebook_data := facebook_fraction_num * remaining_after_youtube / facebook_fraction_den
  remaining_after_youtube - facebook_data

theorem elsa_data_remaining : 
  remaining_data 500 300 2 5 = 120 := 
by 
  simp [remaining_data]
  sorry

end elsa_data_remaining_l631_631422


namespace unique_prime_digit_l631_631267

def is_digit (n : ℕ) : Prop := n < 10

noncomputable def prime_digit_number (A : ℕ) : ℕ := 202100 + A

theorem unique_prime_digit :
  ∃! A, is_digit A ∧ Nat.Prime (prime_digit_number A) :=
begin
  sorry
end

end unique_prime_digit_l631_631267


namespace quadrilateral_proof_l631_631627

noncomputable def inscribed_quad_eq (A B C D K : Point) 
  (h_inscribed: inscribed_quad A B C D)
  (h_AB_AD: A.dist B = A.dist D)
  (h_angle_eq: ∠DAK = ∠ABD)
  (h_K_on_CD: on_line_segment C D K) : Prop :=
  A.dist K ^ 2 = K.dist D ^ 2 + B.dist C * K.dist D

theorem quadrilateral_proof (A B C D K : Point) 
  (h_inscribed: inscribed_quad A B C D)
  (h_AB_AD: A.dist B = A.dist D)
  (h_angle_eq: ∠DAK = ∠ABD)
  (h_K_on_CD: on_line_segment C D K) : inscribed_quad_eq A B C D K :=
sorry

end quadrilateral_proof_l631_631627


namespace ellipse_equation_unique_no_real_m_l631_631829

def is_on_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def ellipse_passing_points (a b : ℝ) : Prop :=
  is_on_ellipse a b 1 (sqrt(6) / 3) ∧ is_on_ellipse a b 0 (-1) ∧ a > b ∧ b > 0

theorem ellipse_equation_unique (a b : ℝ) 
  (h : ellipse_passing_points a b) : 
  a = sqrt(3) ∧ b = 1 ∧ ∀ x y, is_on_ellipse (sqrt(3)) 1 x y ↔ x^2 / 3 + y^2 = 1 := 
sorry 

theorem no_real_m (m : ℝ) : 
  ∀ a b, ellipse_passing_points a b → a = sqrt(3) → b = 1 → 
  ¬(∃ (x1 x2 y1 y2 : ℝ), y1 = x1 + m ∧ y2 = x2 + m ∧ 
  is_on_ellipse (sqrt(3)) 1 x1 y1 ∧ is_on_ellipse (sqrt(3)) 1 x2 y2 ∧ 
  (x1 ≠ x2 ∧ x1 ≠ 0 ∧ x2 ≠ 0) ∧ abs(x1) = abs(x2)) := 
sorry

end ellipse_equation_unique_no_real_m_l631_631829


namespace locus_is_circle_l631_631131

noncomputable def locus_of_points {O1 O2 : ℝ × ℝ} (r1 r2 : ℝ) : set (ℝ × ℝ) :=
  {P : ℝ × ℝ | let O1P := (P.1 - O1.1)^2 + (P.2 - O1.2)^2,
                     O2P := (P.1 - O2.1)^2 + (P.2 - O2.2)^2 in
                O1P - r1^2 = O2P - r2^2 ∧
                O1P < r1^2 ∧
                O2P > r2^2}

theorem locus_is_circle {O1 O2 : ℝ × ℝ} (r1 r2 : ℝ) :
  let midpoint : ℝ × ℝ := ((O1.1 + O2.1) / 2, (O1.2 + O2.2) / 2) in
  ∃ (c : ℝ × ℝ) (R : ℝ), locus_of_points r1 r2 = {P : ℝ × ℝ | (P.1 - c.1)^2 + (P.2 - c.2)^2 = R^2}
∧ c = midpoint := 
by
  sorry

end locus_is_circle_l631_631131


namespace closest_length_of_block_on_ruler_l631_631363

structure BlockOnRuler :=
  (left_edge right_edge : ℝ)
  (position_valid : 3 ≤ left_edge ∧ left_edge < 5 ∧ 5 < right_edge ∧ right_edge ≤ 6)

noncomputable def block_length (block : BlockOnRuler) : ℝ := 
  block.right_edge - block.left_edge

def closest_length (block : BlockOnRuler) : Prop :=
  let length := block_length block in
  length = 2.4

theorem closest_length_of_block_on_ruler (block : BlockOnRuler) : closest_length block :=
  sorry

end closest_length_of_block_on_ruler_l631_631363


namespace part1_part2_part3_l631_631842

-- Definitions for conditions
def C (m : ℝ) : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), (p.1 - 1)^2 + (p.2 - 2)^2 = 5 - m
def C2 : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), (p.1 - 4)^2 + (p.2 - 6)^2 = 16
def l : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), p.1 + 2 * p.2 - 4 = 0

-- Proof statements
theorem part1 (m : ℝ) : (∀ p : ℝ × ℝ, C m p) → m < 5 := sorry

theorem part2 (m : ℝ) : (∀ p : ℝ × ℝ, C m p) → (∃ p : ℝ × ℝ, C2 p) → ∀ p q, (p = (1, 2) ∧ q = (4, 6)) → (dist p q = 5) → m = 4 := sorry

theorem part3 (m : ℝ) (MN_distance : ℝ) : (∀ p : ℝ × ℝ, C m p) → 
  (∀ p : ℝ × ℝ, l p) → MN_distance = 4 * real.sqrt 5 / 5 →
  5 - m = 1 → m = 4 := sorry

end part1_part2_part3_l631_631842


namespace locus_midpoint_ellipse_l631_631360

theorem locus_midpoint_ellipse (P O : ℝ × ℝ) (a b : ℝ) (hO : a > 0 ∧ b > 0) (hP_out : ¬ (P ⊆ (x y : ℝ), ((x - O.1)^2 / a^2 + (y - O.2)^2 / b^2 <= 1))) :
  ∃ M : ℝ × ℝ, (∀ Q : ℝ × ℝ, (((Q.1 - O.1)^2 / a^2 + (Q.2 - O.2)^2 / b^2 = 1) → 
    M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2))) → 
    M = (x y : ℝ, ((x - (P.1 + O.1) / 2)^2 / (a/2)^2 + (y - (P.2 + O.2) / 2)^2 / (b/2)^2 = 1)) :=
by
  sorry

end locus_midpoint_ellipse_l631_631360


namespace number_contains_digit_7_l631_631866

noncomputable def contains_digit (d n : ℕ) : Prop :=
  ∃ k, n / 10^k % 10 = d

noncomputable def count_numbers_with_digit (d bound : ℕ) : ℕ :=
  (finset.range (bound + 1)).filter (λ n, contains_digit d n).card

theorem number_contains_digit_7 : count_numbers_with_digit 7 800 = 152 := 
sorry

end number_contains_digit_7_l631_631866


namespace length_of_train_is_250_02_l631_631314

noncomputable def train_speed_km_per_hr : ℝ := 100
noncomputable def time_to_cross_pole_sec : ℝ := 9

-- Convert speed from km/hr to m/s
noncomputable def speed_m_per_s : ℝ := train_speed_km_per_hr * (1000 / 3600)

-- Calculating the length of the train
noncomputable def length_of_train : ℝ := speed_m_per_s * time_to_cross_pole_sec

theorem length_of_train_is_250_02 :
  length_of_train = 250.02 := by
  -- Proof is omitted (replace 'sorry' with the actual proof)
  sorry

end length_of_train_is_250_02_l631_631314


namespace triangle_is_isosceles_l631_631917

variable {V : Type*} [InnerProductSpace ℝ V]
variables (A B C O : V)

def isIsosceles (A B C : V) : Prop :=
  ∥B - A∥ = ∥C - A∥ ∨ ∥A - B∥ = ∥C - B∥ ∨ ∥A - C∥ = ∥B - C∥

theorem triangle_is_isosceles
  (h : (B - O - (C - O)) ⋅ (B - O + (C - O) - 2 • (A - O)) = 0) :
  isIsosceles A B C :=
by
  sorry

end triangle_is_isosceles_l631_631917


namespace quadrilateral_proof_l631_631628

noncomputable def inscribed_quad_eq (A B C D K : Point) 
  (h_inscribed: inscribed_quad A B C D)
  (h_AB_AD: A.dist B = A.dist D)
  (h_angle_eq: ∠DAK = ∠ABD)
  (h_K_on_CD: on_line_segment C D K) : Prop :=
  A.dist K ^ 2 = K.dist D ^ 2 + B.dist C * K.dist D

theorem quadrilateral_proof (A B C D K : Point) 
  (h_inscribed: inscribed_quad A B C D)
  (h_AB_AD: A.dist B = A.dist D)
  (h_angle_eq: ∠DAK = ∠ABD)
  (h_K_on_CD: on_line_segment C D K) : inscribed_quad_eq A B C D K :=
sorry

end quadrilateral_proof_l631_631628


namespace proof_inequalities_l631_631796

variable {R : Type} [LinearOrder R] [Ring R]

def odd_function (f : R → R) : Prop :=
∀ x : R, f (-x) = -f x

def decreasing_function (f : R → R) : Prop :=
∀ x y : R, x ≤ y → f y ≤ f x

theorem proof_inequalities (f : R → R) (a b : R) 
  (h_odd : odd_function f)
  (h_decr : decreasing_function f)
  (h : a + b ≤ 0) :
  (f a * f (-a) ≤ 0) ∧ (f a + f b ≥ f (-a) + f (-b)) :=
by
  sorry

end proof_inequalities_l631_631796


namespace intersection_M_complement_R_N_l631_631126

noncomputable def M : Set ℝ := {x | x^2 - 2 * x - 3 < 0}
noncomputable def N : Set ℝ := {x | 2^x < 2}
def complement_R_N : Set ℝ := {x | x >= 1}

theorem intersection_M_complement_R_N :
  (M ∩ complement_R_N) = {x | 1 ≤ x ∧ x < 3} := by
  sorry

end intersection_M_complement_R_N_l631_631126


namespace jacob_final_score_l631_631162

theorem jacob_final_score (correct incorrect unanswered : ℕ)
  (correct_points incorrect_deduction : ℝ)
  (h_correct : correct = 20)
  (h_incorrect : incorrect = 10)
  (h_unanswered : unanswered = 5)
  (h_correct_points : correct_points = 1)
  (h_incorrect_deduction : incorrect_deduction = 0.5) :
  correct * correct_points - incorrect * incorrect_deduction = 15 :=
by 
  have h1 : correct * correct_points = 20 := by 
    change correct with 20,
    change correct_points with 1,
    linarith,
  have h2 : incorrect * incorrect_deduction = 5 := by 
    change incorrect with 10,
    change incorrect_deduction with 0.5,
    linarith,
  rw [h1, h2],
  linarith,

end jacob_final_score_l631_631162


namespace xiaolin_final_score_l631_631342

-- Define the conditions
def score_situps : ℕ := 80
def score_800m : ℕ := 90
def weight_situps : ℕ := 4
def weight_800m : ℕ := 6

-- Define the final score based on the given conditions
def final_score : ℕ :=
  (score_situps * weight_situps + score_800m * weight_800m) / (weight_situps + weight_800m)

-- Prove that the final score is 86
theorem xiaolin_final_score : final_score = 86 :=
by sorry

end xiaolin_final_score_l631_631342


namespace count_natural_numbers_divisible_by_9_in_range_l631_631865

theorem count_natural_numbers_divisible_by_9_in_range (a b : ℕ) (h₁ : a = 150) (h₂ : b = 300) : 
  (∃ m : ℕ, m = 17 ∧ 
    finset.card ((finset.filter (λ n, n % 9 = 0) (finset.Ico (a+1) b))) = m) :=
by
  revert h₁ h₂
  sorry

end count_natural_numbers_divisible_by_9_in_range_l631_631865


namespace trig_identity_l631_631446

theorem trig_identity (a : ℝ) (h : Real.tan (a + Real.pi / 4) = 1 / 2) : 
  2 * Real.sin(a) ^ 2 + Real.sin(2 * a) = -2 / 5 := 
by 
  sorry

end trig_identity_l631_631446


namespace scaling_transformation_correct_l631_631172

-- Define the original and transformed curves as sets of points.
def original_curve (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 36
def transformed_curve (x' y' : ℝ) : Prop := x'^2 + y'^2 = 1

-- Define the scaling transformation.
def scaling_transformation (x y : ℝ) : ℝ × ℝ :=
  (x / 3, y / 2)

-- The theorem to prove that the scaling transformation converts the original curve to the transformed curve.
theorem scaling_transformation_correct (x y : ℝ) (h : original_curve x y) :
  let (x', y') := scaling_transformation x y in transformed_curve x' y' :=
by
  sorry

end scaling_transformation_correct_l631_631172


namespace probability_red_then_black_l631_631770

theorem probability_red_then_black :
  let deck := 52
  let red_cards := 26
  let black_cards := 26
  let first_card_choices := red_cards
  let second_card_choices_after_red := black_cards
  let total_cards_after_first := deck - 1
  (first_card_choices * second_card_choices_after_red : ℕ) / (deck * total_cards_after_first : ℕ) = (13 / 51 : ℚ) := by
sorrow

end probability_red_then_black_l631_631770


namespace five_digit_integer_product_1000_l631_631136

theorem five_digit_integer_product_1000 : 
  (∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ (nat.digits 10 n).prod = 1000) → 
  (finset.card {n : ℕ | 10000 ≤ n ∧ n < 100000 ∧ (nat.digits 10 n).prod = 1000} = 40) :=
sorry

end five_digit_integer_product_1000_l631_631136


namespace red_lights_expected_value_l631_631786

theorem red_lights_expected_value :
  let p := 0.4 in
  let trips := 3 in
  let expected_value := p * trips in
  expected_value = 1.2 :=
by
  let p := 0.4
  let trips := 3
  let expected_value := p * trips
  have h : expected_value = 1.2 := by norm_num
  exact h

end red_lights_expected_value_l631_631786


namespace unique_positive_x_for_volume_l631_631312

variable (x : ℕ)

def prism_volume (x : ℕ) : ℕ :=
  (x + 5) * (x - 5) * (x ^ 2 + 25)

theorem unique_positive_x_for_volume {x : ℕ} (h : prism_volume x < 700) (h_pos : 0 < x) :
  ∃! x, (prism_volume x < 700) ∧ (x - 5 > 0) :=
by
  sorry

end unique_positive_x_for_volume_l631_631312


namespace triangle_equilateral_of_medians_and_angles_l631_631544

/-- Given a triangle ABC with medians AD and BE, and angles CAD and CBE equal to 30 degrees, 
prove that the triangle ABC is equilateral. -/
theorem triangle_equilateral_of_medians_and_angles 
  (ABC : Type) [nonempty (triangle ABC)]
  (A B C D E : ABC)
  (hMedians : median A D ∧ median B E)
  (hAngles : angle C A D = 30 ∧ angle C B E = 30) :
  equilateral A B C :=
begin
  sorry -- proof goes here
end

end triangle_equilateral_of_medians_and_angles_l631_631544


namespace count_numbers_with_seven_l631_631905

open Finset

def contains_digit_seven (n : ℕ) : Prop :=
  ∃ d : ℕ, d ∈ digits 10 n ∧ d = 7

theorem count_numbers_with_seven : 
  (card (filter (λ n, contains_digit_seven n) (range 801))) = 152 := 
by
  sorry

end count_numbers_with_seven_l631_631905


namespace divisors_arithmetic_mean_gt_sqrt_l631_631960

open Nat

theorem divisors_arithmetic_mean_gt_sqrt {n : ℕ} (hn : 1 < n) : 
  (∑ d in divisors n, d) / (divisors n).card > (n : ℝ) ^ (1 / 2 : ℝ) :=
by
  sorry

end divisors_arithmetic_mean_gt_sqrt_l631_631960


namespace area_of_BCD_is_15_over_2_l631_631113

/-- Coordinates of vertices -/
def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (7, -1)
def C : ℝ × ℝ := (-2, 5)

/-- Midpoint calculation -/
def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

/-- Equation of line l -/
def equation_of_line_l (A : ℝ × ℝ) (midBC : ℝ × ℝ) : ℝ × ℝ → Prop :=
λ (P : ℝ × ℝ), P.1 + P.2 - 3 = 0

/-- Coordinates of point D -/
def D : ℝ × ℝ := (2, 4)

/-- Equation of line BC -/
def equation_of_line_BC (B C : ℝ × ℝ) : ℝ × ℝ → Prop :=
λ (P : ℝ × ℝ), 2 * P.1 + 3 * P.2 - 11 = 0

/-- Distance from point to line -/
def distance_from_point_to_line (P : ℝ × ℝ) (line : ℝ × ℝ → Prop) : ℝ :=
abs (2 * P.1 + 3 * P.2 - 11) / real.sqrt (2 * 2 + 3 * 3)

/-- Length of line segment BC -/
def length_of_segment_BC (B C : ℝ × ℝ) : ℝ :=
real.sqrt ((B.1 - C.1) ^ 2 + (B.2 - C.2) ^ 2)

/-- Area of triangle calculation -/
def area_of_triangle_BCD (B C D : ℝ × ℝ) : ℝ :=
1 / 2 * length_of_segment_BC B C * distance_from_point_to_line D (equation_of_line_BC B C)

/-- Prove the area of triangle BCD is 15/2 -/
theorem area_of_BCD_is_15_over_2 : area_of_triangle_BCD B C D = 15 / 2 :=
sorry

end area_of_BCD_is_15_over_2_l631_631113


namespace complex_modulus_problem_l631_631111

open Complex

def modulus_of_z (z : ℂ) (h : (z - 2 * I) * (1 - I) = -2) : Prop :=
  abs z = Real.sqrt 2

theorem complex_modulus_problem (z : ℂ) (h : (z - 2 * I) * (1 - I) = -2) : 
  modulus_of_z z h :=
sorry

end complex_modulus_problem_l631_631111


namespace guppies_total_l631_631487

theorem guppies_total :
  let haylee := 3 * 12
  let jose := haylee / 2
  let charliz := jose / 3
  let nicolai := charliz * 4
  haylee + jose + charliz + nicolai = 84 :=
by
  sorry

end guppies_total_l631_631487


namespace slope_angle_of_tangent_line_at_point_l631_631632

theorem slope_angle_of_tangent_line_at_point :
  let f := λ x : ℝ, x^3 - 2 * x + 4 in
  let x0 := (1 : ℝ) in
  let y0 := 3 in
  let df := deriv f x0 in
  let α := real.arctan df in
  α = real.pi / 4 :=
by
  sorry

end slope_angle_of_tangent_line_at_point_l631_631632


namespace number_of_numbers_with_digit_seven_l631_631898

-- Define what it means to contain digit 7
def contains_digit_seven (n : ℕ) : Prop :=
  n.digits 10 ∈ [7]

-- Define the set of numbers from 1 to 800 containing at least one digit 7
def numbers_with_digit_seven : ℕ → Prop :=
  λ n, 1 ≤ n ∧ n ≤ 800 ∧ contains_digit_seven n

-- State the theorem
theorem number_of_numbers_with_digit_seven : (finset.filter numbers_with_digit_seven (finset.range 801)).card = 152 :=
sorry

end number_of_numbers_with_digit_seven_l631_631898


namespace total_guppies_correct_l631_631490

noncomputable def total_guppies : ℕ :=
  let haylee := 3 * 12 in
  let jose := haylee / 2 in
  let charliz := jose / 3 in
  let nicolai := charliz * 4 in
  haylee + jose + charliz + nicolai

theorem total_guppies_correct : total_guppies = 84 :=
by 
  -- skip proof
  sorry

end total_guppies_correct_l631_631490


namespace length_of_second_train_l631_631293

-- Definitions for the conditions
def speed_train1_kmph := 42
def speed_train2_kmph := 30
def length_train1_m := 200
def clear_time_s := 23.998

-- Convert speeds from kmph to m/s
def speed_train1_ms := speed_train1_kmph * 1000 / 3600
def speed_train2_ms := speed_train2_kmph * 1000 / 3600

-- Relative speed when trains move towards each other
def relative_speed_ms := speed_train1_ms + speed_train2_ms

-- Total distance covered when trains clear each other
def total_distance_m := relative_speed_ms * clear_time_s

-- Calculate the length of the second train
def length_train2_m := total_distance_m - length_train1_m

-- Lean theorem statement
theorem length_of_second_train :
  length_train2_m = 279.96 := 
sorry

end length_of_second_train_l631_631293


namespace flower_bouquet_count_l631_631345

theorem flower_bouquet_count :
  {r : ℕ // ∃ c : ℕ, 3 * r + 2 * c = 50}.card = 9 :=
sorry

end flower_bouquet_count_l631_631345


namespace time_to_pass_man_l631_631313

-- Definitions for the given problem
def length_of_train : ℝ := 120
def speed_of_train_kmh : ℝ := 68
def speed_of_man_kmh : ℝ := 8

-- Conversion factor from km/h to m/s
def kmh_to_ms (kmh : ℝ) : ℝ := kmh * (1000 / 3600)

-- Relative speed in m/s
def relative_speed_ms : ℝ := kmh_to_ms (speed_of_train_kmh - speed_of_man_kmh)

-- The time it takes for the train to pass the man in seconds
def time_to_pass : ℝ := length_of_train / relative_speed_ms

-- The theorem we want to prove
theorem time_to_pass_man : time_to_pass = 7.2 := by
  sorry

end time_to_pass_man_l631_631313


namespace max_value_sqrt_inequality_l631_631203

theorem max_value_sqrt_inequality (a b c : ℝ) 
  (h_sum : a + b + c = 3) 
  (h_a : a ≥ -1/2) 
  (h_b : b ≥ -3/2) 
  (h_c : c ≥ -2) :
  sqrt (4*a + 2) + sqrt (4*b + 6) + sqrt (4*c + 8) ≤ 2 * sqrt 21 := 
sorry

end max_value_sqrt_inequality_l631_631203


namespace leila_original_savings_l631_631188

variable (S : ℝ)

-- Define the conditions given in the problem
def spent_on_makeup := (3/5) * S
def spent_on_sweater := (1/3) * S
def cost_sweater := 40
def cost_shoes := 30
def remaining_savings := (2/5) * S

-- Translate the proof problem to Lean statement
theorem leila_original_savings : 
  spent_on_makeup S + spent_on_sweater S + cost_shoes = S → 
  spent_on_sweater S = cost_sweater → 
  remaining_savings S = cost_sweater + cost_shoes → 
  S = 175 :=
by
  -- Proof goes here
  sorry

end leila_original_savings_l631_631188


namespace circle_center_max_area_l631_631115

theorem circle_center_max_area :
  ∀ k : ℝ, let center := if k = 0 then (0, -1) else (-(k / 2), -1) in
  (∀ x y : ℝ, x^2 + y^2 + k * x + 2*y + k^2 = 0 
  → center = (0, -1)) :=
by {
  sorry
}

end circle_center_max_area_l631_631115


namespace elsa_data_remaining_l631_631421

variable (initial_data : ℕ) (youtube_data : ℕ) (facebook_fraction_num : ℕ) (facebook_fraction_den : ℕ)

def remaining_data (initial_data youtube_data facebook_fraction_num facebook_fraction_den : ℕ) : ℕ :=
  let remaining_after_youtube := initial_data - youtube_data
  let facebook_data := facebook_fraction_num * remaining_after_youtube / facebook_fraction_den
  remaining_after_youtube - facebook_data

theorem elsa_data_remaining : 
  remaining_data 500 300 2 5 = 120 := 
by 
  simp [remaining_data]
  sorry

end elsa_data_remaining_l631_631421


namespace positive_difference_eq_496_l631_631709

theorem positive_difference_eq_496 : 
  let a := 8 ^ 2 in 
  (a + a) / 8 - (a * a) / 8 = 496 :=
by
  let a := 8^2
  have h1 : (a + a) / 8 = 16 := by sorry
  have h2 : (a * a) / 8 = 512 := by sorry
  show (a + a) / 8 - (a * a) / 8 = 496 from by
    calc
      (a + a) / 8 - (a * a) / 8
            = 16 - 512 : by rw [h1, h2]
        ... = -496 : by ring
        ... = 496 : by norm_num

end positive_difference_eq_496_l631_631709


namespace tina_total_pens_l631_631288

theorem tina_total_pens : 
  let pink_pens := 12 in
  let green_pens := pink_pens - 9 in
  let blue_pens := green_pens + 3 in
  pink_pens + green_pens + blue_pens = 21 :=
by 
  let pink_pens := 12
  let green_pens := pink_pens - 9
  let blue_pens := green_pens + 3
  show pink_pens + green_pens + blue_pens = 21 from sorry

end tina_total_pens_l631_631288


namespace PC_eq_PA_add_PB_l631_631542

open EuclideanGeometry

variables {A B C D E F P Q : Point}
variables {Γ : Circle}
variables {BC AC AB : ℝ}
variables (a b c : ℝ)

-- Conditions and initial setup
-- Given that BC > AC > AB
def condition1 : BC > AC := sorry
def condition2 : AC > AB := sorry

-- Triangle ABC is within the circumcircle Γ
def condition3 : CircumscribedTriangle Γ A B C := sorry

-- Interior angle bisectors intersect sides at D, E, and F
def condition4 : AngleBisector A B D := sorry
def condition5 : AngleBisector B C E := sorry
def condition6 : AngleBisector C A F := sorry

-- A line through B parallel to EF intersects circumcircle at Q
def condition7 : ParallelThrough B EF Q := sorry
def condition8 : OnCircle Γ Q := sorry

-- Point P on Γ such that QP is parallel to AC
def condition9 : ParallelThrough Q AC P := sorry
def condition10 : OnCircle Γ P := sorry

-- The assertion we want to prove
theorem PC_eq_PA_add_PB
    (h1 : condition1)
    (h2 : condition2)
    (h3 : condition3)
    (h4 : condition4)
    (h5 : condition5)
    (h6 : condition6)
    (h7 : condition7)
    (h8 : condition8)
    (h9 : condition9)
    (h10 : condition10) :
    dist P C = dist P A + dist P B := 
sorry

end PC_eq_PA_add_PB_l631_631542


namespace sum_arithmetic_sequence_1000_to_1010_l631_631791

theorem sum_arithmetic_sequence_1000_to_1010 :
  let a1 := 3
  let d := 7
  let a n := a1 + (n - 1) * d
  (List.range 10).sum (λ k, a (1000 + k)) = 77341 :=
by
  let a1 := 3
  let d := 7
  let a n := a1 + (n - 1) * d
  sorry

end sum_arithmetic_sequence_1000_to_1010_l631_631791


namespace incenter_circumcenter_midpoints_concyclic_l631_631177

noncomputable def incenter (A B C : Point) : Point := sorry
noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def midpoint (P Q : Point) : Point := sorry
def Point := ℂ

variables (A B C : Point)
variables (h : 2 * dist A B = dist B C + dist C A)

theorem incenter_circumcenter_midpoints_concyclic 
  (I := incenter A B C)
  (O := circumcenter A B C)
  (D := midpoint B C)
  (E := midpoint A C) 
  : ∃ K : Circle, I ∈ K ∧ O ∈ K ∧ D ∈ K ∧ E ∈ K :=
sorry

end incenter_circumcenter_midpoints_concyclic_l631_631177


namespace equal_powers_equal_elements_l631_631072

theorem equal_powers_equal_elements
  (a : Fin 17 → ℕ)
  (h : ∀ i : Fin 17, a i ^ a (i + 1) % 17 = a ((i + 1) % 17) ^ a ((i + 2) % 17) % 17)
  : ∀ i j : Fin 17, a i = a j :=
by
  sorry

end equal_powers_equal_elements_l631_631072


namespace tom_paid_total_amount_l631_631290

theorem tom_paid_total_amount :
  let quantity_apples := 8
  let rate_apples := 70
  let quantity_mangoes := 9
  let rate_mangoes := 55
  let cost_apples := quantity_apples * rate_apples
  let cost_mangoes := quantity_mangoes * rate_mangoes
  let total_amount_paid := cost_apples + cost_mangoes
  total_amount_paid = 1055 := by
  -- Definitions as per conditions
  let quantity_apples := 8
  let rate_apples := 70
  let quantity_mangoes := 9
  let rate_mangoes := 55
  let cost_apples := quantity_apples * rate_apples
  let cost_mangoes := quantity_mangoes * rate_mangoes
  let total_amount_paid := cost_apples + cost_mangoes
  -- Proof statement
  show total_amount_paid = 1055, from sorry

end tom_paid_total_amount_l631_631290


namespace subsequences_with_same_sum_l631_631084

-- Define sequences and their conditions
variable (A : Fin 19 → ℕ)
variable (hA : ∀ i, 1 ≤ A i ∧ A i ≤ 88)
variable (B : Fin 88 → ℕ)
variable (hB : ∀ j, 1 ≤ B j ∧ B j ≤ 19)

-- Define partial sums for sequences A and B
def S (n : ℕ) : ℕ := ∑ i in Finset.range n, A ⟨i, sorry⟩
def T (n : ℕ) : ℕ := ∑ j in Finset.range n, B ⟨j, sorry⟩

-- The theorem we need to prove
theorem subsequences_with_same_sum :
  ∃ (i₁ i₂ : ℕ), i₁ < i₂ ∧ i₁ < 19 ∧ i₂ ≤ 19 ∧ ∃ (j₁ j₂ : ℕ), j₁ < j₂ ∧ j₁ < 88 ∧ j₂ ≤ 88 ∧ 
  (S i₂ - S i₁ = T j₂ - T j₁) :=
begin
  sorry
end

end subsequences_with_same_sum_l631_631084


namespace blue_black_pen_ratio_l631_631183

theorem blue_black_pen_ratio (B K R : ℕ) 
  (h1 : B + K + R = 31) 
  (h2 : B = 18) 
  (h3 : K = R + 5) : 
  B / Nat.gcd B K = 2 ∧ K / Nat.gcd B K = 1 := 
by 
  sorry

end blue_black_pen_ratio_l631_631183


namespace rationalize_denominator_l631_631230

theorem rationalize_denominator :
  let A := -12
  let B := 7
  let C := 9
  let D := 13
  let E := 5
  (4 * Real.sqrt 7 + 3 * Real.sqrt 13) ≠ 0 →
  B < D →
  ∀ (x : ℝ), x = (3 : ℝ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) →
    A + B + C + D + E = 22 := 
by
  intros
  -- Provide the actual theorem statement here
  sorry

end rationalize_denominator_l631_631230


namespace sandwiches_prepared_l631_631242

variable (S : ℕ)
variable (H1 : S > 0)
variable (H2 : ∃ r : ℕ, r = S / 4)
variable (H3 : ∃ b : ℕ, b = (3 * S / 4) / 6)
variable (H4 : ∃ c : ℕ, c = 2 * b)
variable (H5 : ∃ x : ℕ, 5 * x = 5)
variable (H6 : 3 * S / 8 - 5 = 4)

theorem sandwiches_prepared : S = 24 :=
by
  sorry

end sandwiches_prepared_l631_631242


namespace largest_value_of_n_l631_631596

theorem largest_value_of_n :
  ∃ (n : ℕ) (X Y Z : ℕ),
    n = 25 * X + 5 * Y + Z ∧
    n = 81 * Z + 9 * Y + X ∧
    X < 5 ∧ Y < 5 ∧ Z < 5 ∧
    n = 121 := by
  sorry

end largest_value_of_n_l631_631596


namespace lcm_is_only_function_l631_631573

noncomputable def f (x y : ℕ) : ℕ := Nat.lcm x y

theorem lcm_is_only_function 
    (f : ℕ → ℕ → ℕ)
    (h1 : ∀ x : ℕ, f x x = x) 
    (h2 : ∀ x y : ℕ, f x y = f y x) 
    (h3 : ∀ x y : ℕ, (x + y) * f x y = y * f x (x + y)) : 
  ∀ x y : ℕ, f x y = Nat.lcm x y := 
sorry

end lcm_is_only_function_l631_631573


namespace trapezium_other_parallel_side_l631_631057

theorem trapezium_other_parallel_side (a b h : ℝ) (area : ℝ) (h_area : area = (1 / 2) * (a + b) * h) (h_a : a = 18) (h_h : h = 20) (h_area_val : area = 380) :
  b = 20 :=
by 
  sorry

end trapezium_other_parallel_side_l631_631057


namespace f_increasing_on_pos_real_l631_631040

noncomputable def f (x : ℝ) : ℝ := x^2 / (x^2 + 1)

theorem f_increasing_on_pos_real : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f x1 < f x2 :=
by sorry

end f_increasing_on_pos_real_l631_631040


namespace shortest_chord_value_l631_631451

-- Define the circle C and the line l
def circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

def line (m x y : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

-- Define the fact that the chord intercepted by the line l on the circle is shortest
def shortest_chord (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle x y ∧ line m x y ∧ ∀ m', m' ≠ m → (∀ (x' y' : ℝ), circle x' y' ∧ line m' x' y' → false)

-- The value of m that makes the chord shortest
theorem shortest_chord_value (m : ℝ) : shortest_chord m ↔ m = -3 / 4 := sorry

end shortest_chord_value_l631_631451


namespace more_visitors_in_december_is_15_l631_631784

noncomputable def visitors_in_october := 100
noncomputable def visitors_in_november := visitors_in_october + 0.15 * visitors_in_october
noncomputable def total_visitors := 345
noncomputable def visitors_in_december := total_visitors - (visitors_in_october + visitors_in_november)
noncomputable def more_visitors_in_december := visitors_in_december - visitors_in_november

theorem more_visitors_in_december_is_15 :
  more_visitors_in_december = 15 :=
by
  sorry

end more_visitors_in_december_is_15_l631_631784


namespace ernie_can_make_circles_l631_631378

-- Make a statement of the problem in Lean 4
theorem ernie_can_make_circles (total_boxes : ℕ) (ali_boxes_per_circle : ℕ) (ernie_boxes_per_circle : ℕ) (ali_circles : ℕ) 
  (h1 : total_boxes = 80) (h2 : ali_boxes_per_circle = 8) (h3 : ernie_boxes_per_circle = 10) (h4 : ali_circles = 5) :
  (total_boxes - ali_boxes_per_circle * ali_circles) / ernie_boxes_per_circle = 4 := 
by 
  -- Proof of the theorem
  sorry

end ernie_can_make_circles_l631_631378


namespace Caroline_lassis_l631_631793

variable (m : ℕ) -- number of mangoes
variable (l : ℕ) -- number of lassis

-- Define the number of lassis per mango based on the provided condition
def lassis_per_mango := 24 / 3

-- Define the number of mangoes Caroline has
def mangoes_owned := 15

-- Define the expected number of lassis based on the number of mangoes and lassis per mango
def expected_result : ℕ := lassis_per_mango * mangoes_owned

theorem Caroline_lassis (h : m = 15) : l = 120 :=
by
  -- Define the number of lassis based on the condition and the number of mangoes
  let l := lassis_per_mango * m
  -- Prove that with 15 mangoes, the number of lassis is 120
  have : l = 120 := sorry
  exact this

end Caroline_lassis_l631_631793


namespace smallest_possible_sum_l631_631261

theorem smallest_possible_sum (E F G H : ℕ) (h1 : F > 0) (h2 : E + F + G = 3 * F) (h3 : F * G = 4 * F * F / 3) :
  E = 6 ∧ F = 9 ∧ G = 12 ∧ H = 16 ∧ E + F + G + H = 43 :=
by 
  sorry

end smallest_possible_sum_l631_631261


namespace count_valid_numbers_l631_631491

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.nodup

def has_at_least_one_prime_digit (n : ℕ) : Prop :=
  (n.digits 10).any is_prime_digit

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 9999 ∧ (n % 2 = 1) ∧ has_distinct_digits n ∧ has_at_least_one_prime_digit n

theorem count_valid_numbers : {n : ℕ | is_valid_number n}.card = 1874 :=
by
  sorry

end count_valid_numbers_l631_631491


namespace raisin_cost_fraction_l631_631027

theorem raisin_cost_fraction
  (R : ℝ)                -- cost of a pound of raisins
  (cost_nuts : ℝ := 2 * R)  -- cost of a pound of nuts
  (cost_raisins : ℝ := 3 * R)  -- cost of 3 pounds of raisins
  (cost_nuts_total : ℝ := 4 * cost_nuts)  -- cost of 4 pounds of nuts
  (total_cost : ℝ := cost_raisins + cost_nuts_total)  -- total cost of the mixture
  (fraction_of_raisins : ℝ := cost_raisins / total_cost)  -- fraction of cost of raisins
  : fraction_of_raisins = 3 / 11 := 
by
  sorry

end raisin_cost_fraction_l631_631027


namespace tan_435_eq_2_plus_sqrt3_l631_631400

open Real

theorem tan_435_eq_2_plus_sqrt3 : tan (435 * (π / 180)) = 2 + sqrt 3 :=
  sorry

end tan_435_eq_2_plus_sqrt3_l631_631400


namespace measure_equality_l631_631564

open MeasureTheory

variables {n : ℕ} (μ ν : MeasureTheory.Measure (fin n → ℝ))

theorem measure_equality (h : ∀ t : (fin n → ℝ), (∫ x, Complex.exp (Complex.I * t • x) ∂μ) =
  (∫ x, Complex.exp (Complex.I * t • x) ∂ν)) :
  μ = ν :=
sorry

end measure_equality_l631_631564


namespace possible_values_of_a_l631_631463

def setA := {x : ℝ | x ≥ 3}
def setB (a : ℝ) := {x : ℝ | 2 * a - x > 1}
def complementB (a : ℝ) := {x : ℝ | x ≥ (2 * a - 1)}

theorem possible_values_of_a (a : ℝ) :
  (∀ x, x ∈ setA → x ∈ complementB a) ↔ (a = -2 ∨ a = 0 ∨ a = 2) :=
by
  sorry

end possible_values_of_a_l631_631463


namespace d_value_l631_631512

theorem d_value (d : ℝ) : (∀ x : ℝ, 3 * (5 + d * x) = 15 * x + 15) ↔ (d = 5) := by 
sorry

end d_value_l631_631512


namespace find_m_range_l631_631819

theorem find_m_range 
  (A : Set ℝ := {x | ∃ y, y = Real.log_base 3 (x^2 - 2 * x - 24)})
  (B : Set ℝ := {x | x ≤ m}) 
  (m : ℝ) 
  (cond : (∀ x, x ∈ A ↔ (x < -4 ∨ x > 6)) 
  ⊓ (∀ x, x ∉ A ↔ (-4 ≤ x ∧ x ≤ 6))) 
  (cond_intersect : ∀ x, (-4 < x ∧ x < 6) → x ≤ m) :
  m ≥ 6 := 
sorry

end find_m_range_l631_631819


namespace waiter_customers_l631_631777

theorem waiter_customers (initial_customers left_customers new_customers : ℕ) :
  initial_customers = 14 → left_customers = 3 → new_customers = 39 → 
  initial_customers - left_customers + new_customers = 50 :=
by
  intros h_initial h_left h_new
  rw [h_initial, h_left, h_new]
  calc
    14 - 3 + 39 = 11 + 39 : by rfl
              ... = 50    : by rfl

end waiter_customers_l631_631777


namespace positive_difference_l631_631700

def a := 8^2
def b := a + a
def c := a * a
theorem positive_difference : ((b / 8) - (c / 8)) = 496 := by
  sorry

end positive_difference_l631_631700


namespace mark_min_correct_problems_l631_631527

noncomputable def mark_score (x : ℕ) : ℤ :=
  8 * x - 21

theorem mark_min_correct_problems (x : ℕ) :
  (4 * 2) + mark_score x ≥ 120 ↔ x ≥ 17 :=
by
  sorry

end mark_min_correct_problems_l631_631527


namespace sum_of_possible_Ns_l631_631726

theorem sum_of_possible_Ns :
  let valid_Ns := {N : ℕ | (N % 6 = 5) ∧ (N % 8 = 7) ∧ (N < 100)} in
  (∑ N in valid_Ns, N) = 236 :=
by
  sorry

end sum_of_possible_Ns_l631_631726


namespace minimum_omega_l631_631851

theorem minimum_omega (ω : ℝ) (h : ω > 0) :
  (∃ n : ℤ, \frac{2}{3} * Real.pi = n * \frac{2 * Real.pi}{ω}) → ω = 3 :=
by
  sorry

end minimum_omega_l631_631851


namespace count_numbers_with_seven_l631_631902

open Finset

def contains_digit_seven (n : ℕ) : Prop :=
  ∃ d : ℕ, d ∈ digits 10 n ∧ d = 7

theorem count_numbers_with_seven : 
  (card (filter (λ n, contains_digit_seven n) (range 801))) = 152 := 
by
  sorry

end count_numbers_with_seven_l631_631902


namespace dance_off_time_l631_631969

def combined_dancing_time (john_first_session : ℕ) (john_second_session : ℕ) (john_break : ℕ) (james_additional_fraction : ℚ) : ℕ :=
  let john_total_danced := john_first_session + john_second_session
  let john_total_including_break := john_first_session + john_second_session + john_break
  let james_total_danced := john_total_including_break + (john_total_including_break * james_additional_fraction)
  john_total_danced + james_total_danced

theorem dance_off_time :
  combined_dancing_time 3 5 1 (1 / 3) = 20 := by
  sorry

end dance_off_time_l631_631969


namespace min_value_product_expression_l631_631058

theorem min_value_product_expression (x : ℝ) : ∃ m, m = -2746.25 ∧ (∀ y : ℝ, (13 - y) * (8 - y) * (13 + y) * (8 + y) ≥ m) :=
sorry

end min_value_product_expression_l631_631058


namespace hyperbola_equation_min_distance_l631_631151

theorem hyperbola_equation (x y : ℝ) (Hx : (2,0) ∈ hyperbola M) :
  hyperbola_eq M = (x^2 / 4) - y^2 = 1 := 
sorry

theorem min_distance (x : ℝ) (H : |x| ≥ 2) (P : ℝ × ℝ) (A : ℝ × ℝ) :
  A = (3, 0) → 
  (A = (3, 0)) → 
  (∀ P ∈ hyperbola M, min_dist P A = (2 / 5) * sqrt 5) :=
sorry

end hyperbola_equation_min_distance_l631_631151


namespace a_10_l631_631484

def a : ℕ → ℚ
| 0     := 0
| 1     := 1
| 2     := 3
| (n+3) := (n:ℚ+3)⁻¹ * ((n:ℚ+2) * a (n+2) - (n:ℚ-1) * a (n+1) - (n:ℚ+2) * (n:ℚ-1) * a n)

theorem a_10 : a 10 = 46 / 10 := by
  sorry

end a_10_l631_631484


namespace sin_is_odd_function_l631_631799

theorem sin_is_odd_function : ∀ x : ℝ, sin (-x) = -sin x := 
by
  sorry

end sin_is_odd_function_l631_631799


namespace average_speed_of_train_l631_631316

-- Definitions used in conditions
def distance1 : ℝ := 250
def time1 : ℝ := 2
def distance2 : ℝ := 350
def time2 : ℝ := 4

-- Total distance and total time
def total_distance : ℝ := distance1 + distance2
def total_time : ℝ := time1 + time2

-- Average speed calculation
def average_speed : ℝ := total_distance / total_time

-- Theorem to prove the average speed is 100 km/h
theorem average_speed_of_train :
  average_speed = 100 := by
  sorry

end average_speed_of_train_l631_631316


namespace evaluate_expression_l631_631424

theorem evaluate_expression : - (16 / 4 * 7 + 25 - 2 * 7) = -39 :=
by sorry

end evaluate_expression_l631_631424


namespace simplify_expression_l631_631506

theorem simplify_expression (x y : ℝ) (h : x - 3 * y = 4) : (x - 3 * y) ^ 2 + 2 * x - 6 * y - 10 = 14 := by
  sorry

end simplify_expression_l631_631506


namespace maximum_value_of_f_over_interval_l631_631814

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2 * x + 2) / (2 * x - 2)

theorem maximum_value_of_f_over_interval :
  ∀ x : ℝ, -4 < x ∧ x < 1 → ∃ M : ℝ, (∀ y : ℝ, -4 < y ∧ y < 1 → f y ≤ M) ∧ M = -1 :=
by
  sorry

end maximum_value_of_f_over_interval_l631_631814


namespace positive_difference_l631_631682

theorem positive_difference : 496 = abs ((64 + 64) / 8 - (64 * 64) / 8) := by
  have h1 : 8^2 = 64 := rfl
  have h2 : 64 + 64 = 128 := rfl
  have h3 : (128 : ℕ) / 8 = 16 := rfl
  have h4 : 64 * 64 = 4096 := rfl
  have h5 : (4096 : ℕ) / 8 = 512 := rfl
  have h6 : 512 - 16 = 496 := rfl
  sorry

end positive_difference_l631_631682


namespace positive_difference_of_fractions_l631_631691

theorem positive_difference_of_fractions : 
  (let a := 8^2 in (a + a) / 8) = 16 ∧ (let a := 8^2 in (a * a) / 8) = 512 →
  (let a := 8^2 in ((a * a) / 8 - (a + a) / 8)) = 496 := 
by
  sorry

end positive_difference_of_fractions_l631_631691


namespace program_output_is_negative_ten_l631_631341

theorem program_output_is_negative_ten (x : ℕ) (h : x = 5) : 
  ((x^2 - x) / 2) * (-1) = -10 :=
by
  sorry

end program_output_is_negative_ten_l631_631341


namespace max_determinant_l631_631202

open Matrix

/-- Given vectors v and w as specified, and u being a unit vector orthogonal to v, 
    the largest possible determinant of the matrix formed by columns u, v, and w is 51 / √10. --/
theorem max_determinant (u v w : Vector3 ℝ) 
  (hv : v = ⟨3, 2, -2⟩) 
  (hw : w = ⟨2, -1, 4⟩) 
  (hu_unit : ‖u‖ = 1) 
  (hu_orthogonal : dot_product u v = 0) :
  let u_cross_vw := cross_product v w in
  determinant ![u, v, w] = 51 / Real.sqrt 10 :=
by
  /- The proof would involve showing the steps as per the solution. -/
  sorry

end max_determinant_l631_631202


namespace cos_of_complementary_angle_l631_631956

theorem cos_of_complementary_angle (Y Z : ℝ) (h : Y + Z = π / 2) 
  (sin_Y : Real.sin Y = 3 / 5) : Real.cos Z = 3 / 5 := 
  sorry

end cos_of_complementary_angle_l631_631956


namespace cellar_pumping_time_is_correct_l631_631747

noncomputable def cellar_pumping_time : ℕ :=
  let length := 30
  let width := 40
  let depth := 2
  let volume := length * width * depth
  let gallons := volume * 7.5
  let pump_rate := 4 * 10
  (gallons / pump_rate : ℕ)

-- The theorem to prove the determination of the pumping time.
theorem cellar_pumping_time_is_correct : cellar_pumping_time = 450 := 
  sorry

end cellar_pumping_time_is_correct_l631_631747


namespace first_99_digits_of_x_are_correct_l631_631809

noncomputable def x : ℝ := (Real.sqrt 26 + 5) ^ 29

/-
Prove: The first 99 digits after the decimal point in the expansion of x
-/
theorem first_99_digits_of_x_are_correct : 
  let digits := -- a function to extract first 99 digits after the decimal point
  true := sorry

end first_99_digits_of_x_are_correct_l631_631809


namespace pencil_of_lines_l631_631983

/-- Given two linear equations P(x, y) = A*x + B*y + C = 0 and
    P1(x, y) = A1*x + B1*y + C1 = 0,
    prove that the lines represented by the equation
    P(x, y) + k * P1(x, y) = 0 (where k is an arbitrary numerical parameter)
    form a pencil of lines passing through the intersection point of 
    the lines represented by P(x, y) = 0 and P1(x, y) = 0. -/
theorem pencil_of_lines
  (A B C A1 B1 C1 : ℝ)
  (k : ℝ)
  (x y : ℝ)
  (P := λ x y : ℝ, A * x + B * y + C)
  (P1 := λ x y : ℝ, A1 * x + B1 * y + C1) :
  ∃ x0 y0 : ℝ,
    (A * x0 + B * y0 + C = 0) ∧ (A1 * x0 + B1 * y0 + C1 = 0) ∧
    (∀ k : ℝ, (A + k * A1) * x + (B + k * B1) * y + (C + k * C1) = 0 → 
    ∃ l : ℝ, y = l * x + (-(C + k * C1) / (B + k * B1))) :=
sorry

end pencil_of_lines_l631_631983


namespace eighth_day_of_april_2000_is_saturday_l631_631618

noncomputable def april_2000_eight_day_is_saturday : Prop :=
  (∃ n : ℕ, (1 ≤ n ∧ n ≤ 7) ∧
            ((n + 0 * 7) = 2 ∨ (n + 1 * 7) = 2 ∨ (n + 2 * 7) = 2 ∨
             (n + 3 * 7) = 2 ∨ (n + 4 * 7) = 2) ∧
            ((n + 0 * 7) % 2 = 0 ∨ (n + 1 * 7) % 2 = 0 ∨
             (n + 2 * 7) % 2 = 0 ∨ (n + 3 * 7) % 2 = 0 ∨
             (n + 4 * 7) % 2 = 0) ∧
            (∃ k : ℕ, k ≤ 4 ∧ (n + k * 7 = 8))) ∧
            (8 % 7) = 1 ∧ (1 ≠ 0)

theorem eighth_day_of_april_2000_is_saturday :
  april_2000_eight_day_is_saturday := 
sorry

end eighth_day_of_april_2000_is_saturday_l631_631618


namespace jane_reads_pages_l631_631964

theorem jane_reads_pages (P : ℕ) (h1 : 7 * (P + 10) = 105) : P = 5 := by
  sorry

end jane_reads_pages_l631_631964


namespace probability_of_diff_by_3_is_1_over_9_l631_631781

noncomputable def probability_diff_3 : ℚ :=
let outcomes := [(x, y) | x ← [1, 2, 3, 4, 5, 6], y ← [1, 2, 3, 4, 5, 6]] in
let successful_outcomes := [(x, y) | (x, y) ← outcomes, |x - y| = 3] in
(successful_outcomes.length : ℚ) / (outcomes.length : ℚ)

theorem probability_of_diff_by_3_is_1_over_9 : probability_diff_3 = 1 / 9 :=
by
  have h_outcomes : outcomes.length = 36 := by sorry
  have h_successful : successful_outcomes.length = 4 := by sorry
  unfold probability_diff_3
  rw [h_outcomes, h_successful]
  norm_num
  exact rfl

end probability_of_diff_by_3_is_1_over_9_l631_631781


namespace unit_vector_example_l631_631274

-- Define the given vector
def d : ℝ × ℝ := (12, -5)

-- Define the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)

-- Define the unit vector in the same direction as a given vector
def unit_vector (v : ℝ × ℝ) : ℝ × ℝ := (v.1 / magnitude v, v.2 / magnitude v)

-- Statement to be proven
theorem unit_vector_example : unit_vector d = (12 / 13, -5 / 13) :=
by
    -- Omitted the proof steps, assume sorry
    sorry

end unit_vector_example_l631_631274


namespace inequality_of_powers_l631_631461

variable {n : ℕ}
variable {x y : Fin n → ℝ}

theorem inequality_of_powers (h1 : ∀ i, 0 < x i ∧ 0 < y i)
    (h2 : ∀ i j, i < j → x i > x j ∧ y i > y j)
    (h3 : ∀ i, (Finset.range (i+1)).sum (λ k, x k) > (Finset.range (i+1)).sum (λ k, y k)) :
    ∀ k : ℕ, (Finset.univ.sum (λ i, (x i)^k)) > (Finset.univ.sum (λ i, (y i)^k)) := sorry

end inequality_of_powers_l631_631461


namespace fractional_equation_solution_l631_631847

theorem fractional_equation_solution (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) ↔ (m ≤ 2 ∧ m ≠ -2) := 
sorry

end fractional_equation_solution_l631_631847


namespace car_miles_traveled_actual_miles_l631_631753

noncomputable def count_skipped_numbers (n : ℕ) : ℕ :=
  let count_digit7 (x : ℕ) : Bool := x = 7
  -- Function to count the number of occurrences of digit 7 in each place value
  let rec count (x num_skipped : ℕ) : ℕ :=
    if x = 0 then num_skipped else
    let digit := x % 10
    let new_count := if count_digit7 digit then num_skipped + 1 else num_skipped
    count (x / 10) new_count
  count n 0

theorem car_miles_traveled (odometer_reading : ℕ) : ℕ :=
  let num_skipped := count_skipped_numbers 3008
  odometer_reading - num_skipped

theorem actual_miles {odometer_reading : ℕ} (h : odometer_reading = 3008) : car_miles_traveled odometer_reading = 2194 :=
by sorry

end car_miles_traveled_actual_miles_l631_631753


namespace angle_SQR_l631_631562

-- Define angles
def PQR : ℝ := 40
def PQS : ℝ := 28

-- State the theorem
theorem angle_SQR : PQR - PQS = 12 := by
  sorry

end angle_SQR_l631_631562


namespace points_lie_on_hyperbola_l631_631445

theorem points_lie_on_hyperbola (s : ℝ) :
  let x := 2 * (Real.exp s + Real.exp (-s))
  let y := 4 * (Real.exp s - Real.exp (-s))
  (x^2) / 16 - (y^2) / 64 = 1 :=
by
  sorry

end points_lie_on_hyperbola_l631_631445


namespace positive_difference_l631_631701

def a := 8^2
def b := a + a
def c := a * a
theorem positive_difference : ((b / 8) - (c / 8)) = 496 := by
  sorry

end positive_difference_l631_631701


namespace parallelogram_D_coordinates_sum_l631_631529

theorem parallelogram_D_coordinates_sum :
  ∀(A B C D : ℝ × ℝ), 
    A = (-1, 2) → 
    B = (3, -4) → 
    C = (7, 3) → 
    let M_AC := ((A.1 + C.1) / 2, (A.2 + C.2) / 2) in
    let M_BD := ((B.1 + D.1) / 2, (B.2 + D.2) / 2) in
    M_AC = M_BD → 
    D.1 + D.2 = 12 :=
by
  intros A B C D hA hB hC M_AC M_BD hM
  sorry

end parallelogram_D_coordinates_sum_l631_631529


namespace oil_price_reduction_l631_631765

theorem oil_price_reduction (P P_r : ℝ) (h1 : P_r = 24.3) (h2 : 1080 / P - 1080 / P_r = 8) : 
  ((P - P_r) / P) * 100 = 18.02 := by
  sorry

end oil_price_reduction_l631_631765


namespace increase_in_radius_l631_631217

-- Define the given conditions
def initial_distance_miles : ℝ := 600
def return_distance_miles : ℝ := 585
def original_radius_inches : ℝ := 12
def inches_per_mile : ℝ := 63360
def pi_val : ℝ := Real.pi

-- Define the theorem to prove
theorem increase_in_radius : 
  let r := original_radius_inches
  let initial_circumference := 2 * pi_val * r / inches_per_mile
  let rotations := initial_distance_miles / initial_circumference
  let r' := (return_distance_miles * inches_per_mile) / (2 * pi_val * rotations)
  r' - r = 0.49 :=
by
  sorry

end increase_in_radius_l631_631217


namespace palindromic_even_greater_than_500000_length_l631_631369

def is_palindromic (n : ℕ) : Prop :=
  let digits : List ℕ := n.digits 10
  digits = digits.reverse

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def condition (n : ℕ) : Prop :=
  is_palindromic n ∧ is_even n ∧ n > 500000

noncomputable def num_palindromic_integers_of_length_n (d : ℕ) : ℕ :=
  (List.range (10^d)).filter (λ n, condition n).length

theorem palindromic_even_greater_than_500000_length :
  ∃ l, num_palindromic_integers_of_length_n l = 200 ∧
       (∀ l', num_palindromic_integers_of_length_n l' = 200 → l' = l) :=
by
  use 6
  sorry

end palindromic_even_greater_than_500000_length_l631_631369


namespace find_integer_less_than_M_div_100_l631_631099

-- The problem and proof constants
theorem find_integer_less_than_M_div_100 :
  let M := 4992 in
  let result := ⌊M / 100⌋ in
  result = 49 :=
by
  -- The conditions given and the resultant M is defined.
  have h1 : 1 / (3! * 18!) + 1 / (4! * 17!) + 1 / (5! * 16!) + 1 / (6! * 15!) + 1 / (7! * 14!) + 
            1 / (8! * 13!) + 1 / (9! * 12!) + 1 / (10! * 11!) = M / (2! * 19!) := sorry,
  -- Hence, final result.
  have h2 : M = 4992 := sorry,
  have h3 : result = ⌊4992 / 100⌋ := by simp [M, result, int.floor_eq_iff, ←div_lt_iff, int.cast_49],
  exact h3

end find_integer_less_than_M_div_100_l631_631099


namespace compare_fractions_l631_631395

theorem compare_fractions :
  +(- (5 / 6)) > -(|-( 8 / 9 )|) := 
by
  sorry

end compare_fractions_l631_631395


namespace distance_P1_P2_l631_631088

-- Define the point P
def P : ℝ × ℝ × ℝ := (1, 2, 3)

-- Define the symmetric point P1 about y-axis
def P1 : ℝ × ℝ × ℝ := (-1, 2, -3)

-- Define the symmetric point P2 about the coordinate plane xOz
def P2 : ℝ × ℝ × ℝ := (1, -2, 3)

-- Calculate the distance between P1 and P2
noncomputable def distance (a b : ℝ × ℝ × ℝ) : ℝ :=
  ((a.1 - b.1)^2 + (a.2 - b.2)^2 + (a.3 - b.3)^2).sqrt

theorem distance_P1_P2 : distance P1 P2 = 2 * Real.sqrt 14 :=
by
  rw [distance, P1, P2]
  simp
  norm_num
  sorry

end distance_P1_P2_l631_631088


namespace tan_435_eq_2_plus_sqrt3_l631_631399

open Real

theorem tan_435_eq_2_plus_sqrt3 : tan (435 * (π / 180)) = 2 + sqrt 3 :=
  sorry

end tan_435_eq_2_plus_sqrt3_l631_631399


namespace marks_lost_per_wrong_answer_l631_631939

theorem marks_lost_per_wrong_answer 
  (marks_per_correct : ℕ)
  (total_questions : ℕ)
  (total_marks : ℕ)
  (correct_answers : ℕ)
  (wrong_answers : ℕ)
  (score_from_correct : ℕ := correct_answers * marks_per_correct)
  (remaining_marks : ℕ := score_from_correct - total_marks)
  (marks_lost_per_wrong : ℕ) :
  total_questions = correct_answers + wrong_answers →
  total_marks = 130 →
  correct_answers = 38 →
  total_questions = 60 →
  marks_per_correct = 4 →
  marks_lost_per_wrong * wrong_answers = remaining_marks →
  marks_lost_per_wrong = 1 := 
sorry

end marks_lost_per_wrong_answer_l631_631939


namespace total_dots_proof_l631_631641

noncomputable def total_dots_on_figure : ℕ :=
  let single_die_dots := 1 + 2 + 3 + 4 + 5 + 6
  let total_dots_7_dice := single_die_dots * 7
  let erased_dots := 2 * (1 + 2 + 3 + 4 + 5 + 6).nat_abs
  total_dots_7_dice - erased_dots

theorem total_dots_proof :
  total_dots_on_figure = 75 :=
by
  let single_die_dots := 21
  let total_dots_7_dice := single_die_dots * 7 -- This calculates 147 dots.
  let erased_dots := 2 * 27 -- This calculates subtraction of glued pairs.
  let total_dots_figure := total_dots_7_dice - erased_dots
  show total_dots_on_figure = 75 from
    calc total_dots_figure : 147 - 54 = 93

end total_dots_proof_l631_631641


namespace anne_sequence_erasures_l631_631582

theorem anne_sequence_erasures :
  let initial_sequence : ℕ → ℕ := λ n, [1, 2, 3, 4, 5, 6, 7][n % 7]
  let erased_every_fourth : ℕ → ℕ := sorry -- function to represent the sequence after erasing every fourth digit
  let erased_every_sixth : ℕ → ℕ := sorry -- function to represent the sequence after erasing every sixth digit from the previous list
  let erased_every_seventh : ℕ → ℕ := sorry -- function to represent the sequence after erasing every seventh digit from the last list
  let final_sequence : List ℕ := List.range 14000 |>.map erased_every_seventh
  (final_sequence.nth 3029).getOrElse 0 + 
  (final_sequence.nth 3030).getOrElse 0 + 
  (final_sequence.nth 3031).getOrElse 0 = 6 := sorry

end anne_sequence_erasures_l631_631582


namespace trig_identity_l631_631396

theorem trig_identity : (1 / Real.cos (Real.degToRad 50) - 2 / Real.sin (Real.degToRad 50)) = 4 / 3 := 
by
  sorry

end trig_identity_l631_631396


namespace greatest_integer_solution_l631_631661

theorem greatest_integer_solution (n : ℤ) (h : n^2 - 13 * n + 40 ≤ 0) : n ≤ 8 :=
sorry

end greatest_integer_solution_l631_631661


namespace converse_of_propositions_is_true_l631_631782

theorem converse_of_propositions_is_true :
  (∀ x : ℝ, (x = 1 ∨ x = 2) ↔ (x^2 - 3 * x + 2 = 0)) ∧
  (∀ x y : ℝ, (x^2 + y^2 = 0) ↔ (x = 0 ∧ y = 0)) := 
by {
  sorry
}

end converse_of_propositions_is_true_l631_631782


namespace elsa_data_remaining_l631_631420

variable (data_total : ℕ) (data_youtube : ℕ)

def data_remaining_after_youtube (data_total data_youtube : ℕ) : ℕ := data_total - data_youtube

def data_fraction_spent_on_facebook (data_left : ℕ) : ℕ := (2 * data_left) / 5

theorem elsa_data_remaining
  (h_data_total : data_total = 500)
  (h_data_youtube : data_youtube = 300) :
  data_remaining_after_youtube data_total data_youtube
  - data_fraction_spent_on_facebook (data_remaining_after_youtube data_total data_youtube) 
  = 120 :=
by
  sorry

end elsa_data_remaining_l631_631420


namespace largest_non_formable_amount_l631_631942

theorem largest_non_formable_amount (n : ℕ) : 
  let denominations := [3n - 2, 6n - 1, 6n + 2, 6n + 5] in
  ∀ s, (∃ (c1 c2 c3 c4 : ℕ), s = c1 * (3n - 2) + c2 * (6n - 1) + c3 * (6n + 2) + c4 * (6n + 5)) ∨ s <= 6n^2 - 4n - 3 :=
sorry

end largest_non_formable_amount_l631_631942


namespace min_value_l631_631836

theorem min_value (a : ℝ) (h : a > 1) : a + 1 / (a - 1) ≥ 3 :=
sorry

end min_value_l631_631836


namespace number_of_numbers_with_digit_7_from_1_to_800_eq_233_l631_631888

def contains_digit (n d : ℕ) : Prop :=
  ∃ k, 10 ^ k > 0 ∧ d = (n / 10 ^ k) % 10

def numbers_without_digit (n d : ℕ) : finset ℕ :=
  (finset.range n).filter (λ x, ¬ contains_digit x d)

def count_numbers_with_digit (n d : ℕ) : ℕ :=
  n - (numbers_without_digit n d).card

theorem number_of_numbers_with_digit_7_from_1_to_800_eq_233 :
  count_numbers_with_digit 800 7 = 233 :=
  sorry

end number_of_numbers_with_digit_7_from_1_to_800_eq_233_l631_631888


namespace find_B_l631_631725

theorem find_B (A B : ℕ) (h : 5 * 100 + 10 * A + 8 - (B * 100 + 14) = 364) : B = 2 :=
sorry

end find_B_l631_631725


namespace program_output_l631_631589

theorem program_output (n : ℕ) : n = 1 → (∀ n, n < 1000 → n = 729) := 
begin
  sorry
end

end program_output_l631_631589


namespace locus_of_X_l631_631579

-- Define the elements of the problem
variables {A B C M P X Y Z : Type}
variables (α β : ℝ)

-- Define the conditions given in the problem
variables (triangle_ABC_acute : ∀ (A B C : Type), mangle A C B ≤ mangle A B C)
variables (M_midpoint_BC : midpoint M B C)
variables (P_on_MC : on_segment P M C)
variables (C1_centered_at_C : circumference C1 C)
variables (C2_centered_at_B : circumference C2 B)
variables (P_on_C1_C2 : on_circumference P C1 ∧ on_circumference P C2)
variables (X_opposite_semiplane : semiplane_opposite X B A P)
variables (Y_intersection_XB_C2 : intersection Y X B C2)
variables (Z_intersection_XC_C1 : intersection Z X C C1)
variables (angle_PAX : mangle P A X = α)
variables (angle_ABC : mangle A B C = β)

-- Specify the conditions
variables (cond_a : ∀ XY XZ XC CP XB BP, XY / XZ = (XC + CP) / (XB + BP))
variables (cond_b : cos α = AB * (sin β / AP))

-- To prove: X lies on the perpendicular bisector of BC
theorem locus_of_X (h1 : triangle_ABC_acute A B C)
                   (h2 : M_midpoint_BC M B C)
                   (h3 : P_on_MC P M C)
                   (h4 : C1_centered_at_C C1 C)
                   (h5 : C2_centered_at_B C2 B)
                   (h6 : P_on_C1_C2 P C1 C2)
                   (h7 : X_opposite_semiplane X B A P)
                   (h8 : Y_intersection_XB_C2 Y X B C2)
                   (h9 : Z_intersection_XC_C1 Z X C C1)
                   (h10 : angle_PAX α P A X)
                   (h11 : angle_ABC β A B C)
                   (h12 : cond_a XY XZ XC CP XB BP)
                   (h13 : cond_b α β A B P) : 
  is_perpendicular_bisector X B C :=
sorry

end locus_of_X_l631_631579


namespace dihedral_angle_condition_l631_631329

-- Define the necessary structures and conditions
variables {α β l : Type} [linear_order α] [linear_order β]
variable {O : α}
variable {A B : β}

-- Define the predicates and propositions
def point_on_edge (O : α) (l : Type) : Prop := sorry
def perp (X Y : Type) : Prop := sorry 
def subset (x y : Type) : Prop := sorry
def plane_angle (A O B : β) : Prop := sorry 

theorem dihedral_angle_condition
  (edge_l : l)
  (point_O_on_l : point_on_edge O l)
  (AO_perp_l : perp AO l)
  (BO_perp_l : perp BO l)
  (AO_in_alpha : subset AO α)
  (BO_in_beta : subset BO β):
  plane_angle A O B :=
sorry

end dihedral_angle_condition_l631_631329


namespace T_n_bounds_l631_631468

-- Let n be a positive integer
variable (n : ℕ) (hn : 0 < n)

-- Define the general term a_n of the sequence
def a_n (n : ℕ) : ℝ := (1 / 2) ^ n

-- Define the sum T_n of the first n terms of the sequence (n a_n)
def T_n (n : ℕ) : ℝ := ∑ k in Finset.range n, (k.succ : ℝ) * a_n k.succ

-- The theorem to prove
theorem T_n_bounds (hn : 0 < n) : 
  (1 / 2) ≤ T_n n ∧ T_n n < 2 :=
sorry

end T_n_bounds_l631_631468


namespace correct_sleep_time_is_option_l631_631311

def valid_sleep_time_options := Set (String × ℕ) := Set.of_list [
  ("9 seconds", 1),
  ("9 minutes", 2),
  ("9 hours", 3)
]

noncomputable def correct_sleep_time : (String × ℕ) :=
  ("9 hours", 3)

theorem correct_sleep_time_is_option :
  correct_sleep_time ∈ valid_sleep_time_options :=
by
  rw Set.of_list.mem
  simp
  exact or.inr (or.inr rfl)

end correct_sleep_time_is_option_l631_631311


namespace time_to_cover_escalator_l631_631008

-- Definitions for the provided conditions.
def escalator_speed : ℝ := 7
def escalator_length : ℝ := 180
def person_speed : ℝ := 2

-- Goal to prove the time taken to cover the escalator length.
theorem time_to_cover_escalator : (escalator_length / (escalator_speed + person_speed)) = 20 := by
  sorry

end time_to_cover_escalator_l631_631008


namespace tetrahedron_surface_area_l631_631005

theorem tetrahedron_surface_area (a : ℝ) (h : a = Real.sqrt 2) :
  let R := (a * Real.sqrt 6) / 4
  let S := 4 * Real.pi * R^2
  S = 3 * Real.pi := by
  /- Proof here -/
  sorry

end tetrahedron_surface_area_l631_631005


namespace simplify_expression_l631_631423

theorem simplify_expression : (2^3002 * 3^3004) / 6^3003 = 3 / 4 := by
  sorry

end simplify_expression_l631_631423


namespace ratio_parallel_segments_l631_631100

theorem ratio_parallel_segments 
  (F N F U : ℝ)
  (F R F L : ℝ)
  (F E F I : ℝ)
  (h1 : IL ∥ EU)
  (h2 : RE ∥ NI)
  (h3 : ∀ FU FL FE FI : ℝ, h1 → (FU / FL = FE / FI))
  (h4 : ∀ FN FR FI FE : ℝ, h2 → (FN / FR = FI / FE)) :
  (FN * FU) / (FR * FL) = 1 := by
  sorry

end ratio_parallel_segments_l631_631100


namespace inscribed_cube_edge_length_l631_631837

theorem inscribed_cube_edge_length
  (A B C O : Point)
  (h1 : on_surface A O)
  (h2 : on_surface B O)
  (h3 : on_surface C O)
  (h4 : dist A B = 2)
  (h5 : dist B C = 2)
  (h6 : dist C A = 2)
  (h7 : distance_to_plane O (plane ABC) = 2) :
  edge_length_of_inscribed_cube O = 8 / 3 := 
sorry

end inscribed_cube_edge_length_l631_631837


namespace find_m_plus_n_l631_631565

noncomputable def altitude_distance (AB AC BC : ℕ) : ℕ := 
  let AH := (104 * 104 + 202) / 208
  let BH := 104 - AH
  let AC := 101
  let BC := 99
  let AH_minus_BH := 0
  let RH := (AH + AC - AC) / 2
  let SH := (AC + BH - BC) / 2
  let RS := abs ((AH - BH - AC + BC) / 2)
  let rs_value := abs ((AH_minus_BH - AC + BC) / 2)
  let fraction_value := RS / rs_value 
  1

theorem find_m_plus_n : altitude_distance 104 101 99 = 2 :=
  sorry

end find_m_plus_n_l631_631565


namespace unique_A_for_prime_number_l631_631269

def is_prime (n : ℕ) : Prop := Nat.Prime n

def six_digit_number_with_A (A : ℕ) : ℕ := 202100 + A

theorem unique_A_for_prime_number :
  ∃! (A : ℕ), A ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ is_prime (six_digit_number_with_A A) :=
begin
  use 9,
  split,
  { split,
    { simp, },
    { sorry } },
  { intros B hB,
    have HB := hB.2,
    obtain ⟨H1, H2⟩ : B = 9 := by sorry,
    exact H1,
  }
end

end unique_A_for_prime_number_l631_631269


namespace sum_absolute_values_l631_631568

open Real

theorem sum_absolute_values (p q r s t : ℝ) 
  (h1 : |p - q| = 1) 
  (h2 : |q - r| = 2) 
  (h3 : |r - s| = 3) 
  (h4 : |s - t| = 5) : 
  ∑ (x : ℝ) in { |p - t| | p q r s t : ℝ ∧ 
    |p - q| = 1 ∧ |q - r| = 2 ∧ |r - s| = 3 ∧ |s - t| = 5 }.toFinset, x = 32 :=
sorry

end sum_absolute_values_l631_631568


namespace floor_sub_y_eq_zero_l631_631501

theorem floor_sub_y_eq_zero {y : ℝ} (h : ⌊y⌋ + ⌈y⌉ = 2 * y) : ⌊y⌋ - y = 0 :=
sorry

end floor_sub_y_eq_zero_l631_631501


namespace ratio_shoes_sandals_simplified_l631_631622

-- Define the given conditions
def shoes_sold : ℕ := 72
def sandals_sold : ℕ := 40

-- Define the GCD of the two given conditions
def gcd_shoes_sandals : ℕ := Nat.gcd shoes_sold sandals_sold

-- Define the simplified ratio of shoes to sandals
def simplified_shoes : ℕ := shoes_sold / gcd_shoes_sandals
def simplified_sandals : ℕ := sandals_sold / gcd_shoes_sandals

-- Prove that the ratio of shoes sold to sandals sold is 9:5
theorem ratio_shoes_sandals_simplified : simplified_shoes = 9 ∧ simplified_sandals = 5 :=
by
  -- We only state the theorem without proof
  sorry

end ratio_shoes_sandals_simplified_l631_631622


namespace calculate_expression_l631_631024

-- Define the numerator and denominator
def numerator := 11 - 10 + 9 - 8 + 7 - 6 + 5 - 4 + 3 - 2 + 1
def denominator := 2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10

-- Prove the expression equals 1
theorem calculate_expression : (numerator / denominator) = 1 := by
  sorry

end calculate_expression_l631_631024


namespace count_numbers_with_seven_l631_631903

open Finset

def contains_digit_seven (n : ℕ) : Prop :=
  ∃ d : ℕ, d ∈ digits 10 n ∧ d = 7

theorem count_numbers_with_seven : 
  (card (filter (λ n, contains_digit_seven n) (range 801))) = 152 := 
by
  sorry

end count_numbers_with_seven_l631_631903


namespace number_of_numbers_with_digit_seven_l631_631895

-- Define what it means to contain digit 7
def contains_digit_seven (n : ℕ) : Prop :=
  n.digits 10 ∈ [7]

-- Define the set of numbers from 1 to 800 containing at least one digit 7
def numbers_with_digit_seven : ℕ → Prop :=
  λ n, 1 ≤ n ∧ n ≤ 800 ∧ contains_digit_seven n

-- State the theorem
theorem number_of_numbers_with_digit_seven : (finset.filter numbers_with_digit_seven (finset.range 801)).card = 152 :=
sorry

end number_of_numbers_with_digit_seven_l631_631895


namespace max_houses_on_board_l631_631744

-- Define the concept of an 8x8 board and the shading rules
def is_house_in_shade (board : ℕ → ℕ → Prop) (r c : ℕ) : Prop :=
  board (r + 1) c ∧ board r (c + 1) ∧ board r (c - 1)

-- Define the function to count houses on the board
def count_houses (board : ℕ → ℕ → Prop) : ℕ :=
  (List.range 8).sum (λ r -> (List.range 8).count (board r))

-- Define the maximum houses without any being in the shade
def max_houses_no_shade : ℕ :=
  50

-- Define the 8x8 grid constraints and the shading rule
def valid_board (board : ℕ → ℕ → Prop) : Prop :=
  ∀ r c, r < 8 ∧ c < 8 → ¬ is_house_in_shade board r c

-- Theorem: The maximum number of houses on the board such that no house is in the shade is 50
theorem max_houses_on_board : ∃ (board : ℕ → ℕ → Prop), valid_board board ∧ count_houses board = max_houses_no_shade :=
by sorry

end max_houses_on_board_l631_631744


namespace count_numbers_with_digit_7_in_range_l631_631875

theorem count_numbers_with_digit_7_in_range : 
  let numbers_in_range := {n : ℕ | 1 ≤ n ∧ n ≤ 800}
      contains_digit_7 (n : ℕ) : Prop := n.digits 10.contains 7
  in (finset.filter (λ n, contains_digit_7 n) (finset.range 801)).card = 152 :=
by 
  let numbers_in_range := {n : ℕ | 1 ≤ n ∧ n ≤ 800}
  let contains_digit_7 (n : ℕ) : Prop := n.digits 10.contains 7
  have h := (finset.filter (λ n, contains_digit_7 n) (finset.range 801)).card
  sorry

end count_numbers_with_digit_7_in_range_l631_631875


namespace sum_of_odd_numbered_terms_l631_631826

theorem sum_of_odd_numbered_terms {a : ℕ → ℕ} (S : ℕ → ℕ) (n : ℕ) (h1 : ∀ n, S n = 2^n - 1)
  (h2 : ∀ n, a n = S (n + 1) - S n) :
  let b (n : ℕ) := a (2 * n),
  T (n : ℕ) := ∑ i in range n, b i in
  T n = (2^(2*n) - 1) / 3 := by
  sorry

end sum_of_odd_numbered_terms_l631_631826


namespace incenter_circumcenter_midpoints_concyclic_l631_631178

theorem incenter_circumcenter_midpoints_concyclic
  (A B C I O D E : Point)
  (hABC : Triangle A B C)
  (hCond : 2 * (dist A B) = (dist B C) + (dist C A))
  (hI : Incenter I A B C)
  (hO : Circumcenter O A B C)
  (hD : Midpoint D B C)
  (hE : Midpoint E A C) :
  Concyclic {I, O, D, E} :=
begin
  sorry
end

end incenter_circumcenter_midpoints_concyclic_l631_631178


namespace inscribed_angle_half_central_angle_l631_631222

theorem inscribed_angle_half_central_angle
  {O : Point} {A B C : Point}
  (hO_AO : distance O A = distance O B)
  (hO_AC : distance O A = distance O C)
  (hO_eq : ∃ O, Circle O B C A)
  (hBAC_eq : ∠ B A C = angle.inscribed O B C) 
  : ∠ B A C = (1 / 2) * ∠ B O C := 
sorry

end inscribed_angle_half_central_angle_l631_631222


namespace positive_difference_l631_631702

def a := 8^2
def b := a + a
def c := a * a
theorem positive_difference : ((b / 8) - (c / 8)) = 496 := by
  sorry

end positive_difference_l631_631702


namespace inverse_modulo_example_l631_631055

theorem inverse_modulo_example :
  ∃ x : ℤ, (2 * x ≡ 1 [ZMOD 185]) ∧ (0 ≤ x) ∧ (x < 185) ∧ (x = 93) :=
begin
  use 93,
  split,
  { -- 2 * 93 ≡ 1 [ZMOD 185]
    exact int.modeq.intro 185 93 1 186 rfl,
  },
  split,
  { -- 0 ≤ 93
    exact le_refl 93,
  },
  { -- 93 < 185
    exact nat.lt_succ_self 184,
  },
  -- and x = 93
  refl
end

end inverse_modulo_example_l631_631055


namespace distance_between_points_l631_631298

-- Define the points
def point1 := (1 : ℤ, 3 : ℤ)
def point2 := (-5 : ℤ, 7 : ℤ)

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℤ × ℤ) : ℤ := 
  Int.sqrt (((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2).toNat)

-- Define the problem statement
theorem distance_between_points : distance point1 point2 = 2 * (Int.sqrt 13) := by
  sorry

end distance_between_points_l631_631298


namespace fractional_eq_nonneg_solution_l631_631844

theorem fractional_eq_nonneg_solution 
  (m x : ℝ)
  (h1 : x ≠ 2)
  (h2 : x ≥ 0)
  (eq_fractional : m / (x - 2) + 1 = x / (2 - x)) :
  m ≤ 2 ∧ m ≠ -2 := 
  sorry

end fractional_eq_nonneg_solution_l631_631844


namespace least_multiple_of_15_greater_than_520_l631_631665

theorem least_multiple_of_15_greater_than_520 : ∃ n : ℕ, n > 520 ∧ n % 15 = 0 ∧ (∀ m : ℕ, m > 520 ∧ m % 15 = 0 → n ≤ m) ∧ n = 525 := 
by
  sorry

end least_multiple_of_15_greater_than_520_l631_631665


namespace max_queens_on_8x8_chessboard_l631_631301

-- Define the statement for the problem
theorem max_queens_on_8x8_chessboard : 
  ∃ (queen_positions : fin 8 → fin 8), 
  (∀ i j : fin 8, i ≠ j → 
    (queen_positions i ≠ queen_positions j ∧ 
     abs (i - j) ≠ abs (queen_positions i - queen_positions j))) :=
sorry

end max_queens_on_8x8_chessboard_l631_631301


namespace sum_infinite_series_l631_631037

open Real

noncomputable def series_sum (n : ℕ) : ℝ :=
  if n % 3 = 0 then (1 / 3^n)
  else if n % 3 = 1 then -(1 / 3^n)
  else -(1 / 3^n)

theorem sum_infinite_series : 
  (∑' n, series_sum n) = 5 / 26 :=
begin
  sorry
end

end sum_infinite_series_l631_631037


namespace num_digits_2_pow_15_mul_5_pow_10_l631_631021

theorem num_digits_2_pow_15_mul_5_pow_10 : 
  (nat.digits 10 (2^15 * 5^10)).length = 12 :=
by sorry

end num_digits_2_pow_15_mul_5_pow_10_l631_631021


namespace polynomial_remainder_l631_631723

theorem polynomial_remainder :
  let p : ℤ[X] := 3 * X^2 - 20 * X + 62
  let q : ℤ[X] := X - 6
  let r : ℤ := 50
  polynomial.divMod p q = (3 * X - 2, r) :=
by
  sorry

end polynomial_remainder_l631_631723


namespace numbers_with_7_in_1_to_800_l631_631911

theorem numbers_with_7_in_1_to_800 : 
  (card { n ∈ finset.range (800 + 1) | ∃ d ∈ n.digits 10, d = 7 }) = 152 := 
sorry

end numbers_with_7_in_1_to_800_l631_631911


namespace cost_price_of_book_l631_631317

theorem cost_price_of_book 
  (C : ℝ)
  (h1 : ∃ C, C > 0)
  (h2 : 1.10 * C = 1.15 * C - 120) :
  C = 2400 :=
sorry

end cost_price_of_book_l631_631317


namespace count_numbers_with_seven_l631_631906

open Finset

def contains_digit_seven (n : ℕ) : Prop :=
  ∃ d : ℕ, d ∈ digits 10 n ∧ d = 7

theorem count_numbers_with_seven : 
  (card (filter (λ n, contains_digit_seven n) (range 801))) = 152 := 
by
  sorry

end count_numbers_with_seven_l631_631906


namespace decompose_vec_d_l631_631411

def vec (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)

def vec_a : ℝ × ℝ × ℝ := vec 1 1 3
def vec_b : ℝ × ℝ × ℝ := vec 2 (-1) (-6)
def vec_c : ℝ × ℝ × ℝ := vec 5 3 (-1)
def vec_d : ℝ × ℝ × ℝ := vec (-9) 2 25

theorem decompose_vec_d :
  ∃ (x y z : ℝ), x = 2 ∧ y = -3 ∧ z = -1 ∧
  (x * vec_a.fst + y * vec_b.fst + z * vec_c.fst = vec_d.fst ∧
   x * vec_a.snd + y * vec_b.snd + z * vec_c.snd = vec_d.snd ∧
   x * vec_a.snd.snd + y * vec_b.snd.snd + z * vec_c.snd.snd = vec_d.snd.snd) :=
by
  use 2, -3, -1
  split; 
  repeat {split}; 
  sorry

end decompose_vec_d_l631_631411


namespace rationalize_denominator_l631_631233

theorem rationalize_denominator :
  let A := -12
  let B := 7
  let C := 9
  let D := 13
  let E := 5
  (4 * Real.sqrt 7 + 3 * Real.sqrt 13) ≠ 0 →
  B < D →
  ∀ (x : ℝ), x = (3 : ℝ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) →
    A + B + C + D + E = 22 := 
by
  intros
  -- Provide the actual theorem statement here
  sorry

end rationalize_denominator_l631_631233


namespace sin_405_eq_sqrt2_div2_l631_631275

theorem sin_405_eq_sqrt2_div2 (h_period : ∀ θ, sin (θ + 360) = sin θ) (h_sin_45 : sin 45 = real.sqrt 2 / 2) :
  sin 405 = real.sqrt 2 / 2 :=
by
  sorry

end sin_405_eq_sqrt2_div2_l631_631275


namespace number_contains_digit_7_l631_631867

noncomputable def contains_digit (d n : ℕ) : Prop :=
  ∃ k, n / 10^k % 10 = d

noncomputable def count_numbers_with_digit (d bound : ℕ) : ℕ :=
  (finset.range (bound + 1)).filter (λ n, contains_digit d n).card

theorem number_contains_digit_7 : count_numbers_with_digit 7 800 = 152 := 
sorry

end number_contains_digit_7_l631_631867


namespace garden_division_l631_631571

theorem garden_division (n : ℕ) : 
  let num_ways := 2^n in 
  ∃ ways : ℕ, ways = num_ways :=
by
  sorry

end garden_division_l631_631571


namespace taco_price_theorem_l631_631771

noncomputable def price_hard_shell_taco_proof
  (H : ℤ)
  (price_soft : ℤ := 2)
  (num_hard_tacos_family : ℤ := 4)
  (num_soft_tacos_family : ℤ := 3)
  (num_additional_customers : ℤ := 10)
  (total_earnings : ℤ := 66)
  : Prop :=
  4 * H + 3 * price_soft + 10 * 2 * price_soft = total_earnings → H = 5

theorem taco_price_theorem : price_hard_shell_taco_proof 5 := 
by
  sorry

end taco_price_theorem_l631_631771


namespace imaginary_part_z_l631_631112

-- Define the complex number z
def z : ℂ := (1 - complex.i) / (1 + 3 * complex.i)

-- Prove that the imaginary part of z is -2/5
theorem imaginary_part_z : z.im = -2 / 5 := 
  sorry

end imaginary_part_z_l631_631112


namespace work_completion_by_B_l631_631337

theorem work_completion_by_B :
  (A_days : ℕ) (B_days : ℕ) (work_days : ℕ)
  (complete_work_a : A_days = 30) 
  (complete_work : work_days = 20) 
  (work_together_days: ℕ) (leave : work_together_days = 5)
  (fraction_work_a : 1 / (complete_work_a : ℝ)) 
  (fraction_work_b : 1 / (B_days : ℝ))
  (fraction_work_together : work_together_days * (fraction_work_a + fraction_work_b) + (work_days - work_together_days) * fraction_work_a = 1) : B_days = 15 :=
sorry

end work_completion_by_B_l631_631337


namespace prime_solution_l631_631428

open Nat

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem prime_solution : ∀ (p q : ℕ), 
  is_prime p → is_prime q → 7 * p * q^2 + p = q^3 + 43 * p^3 + 1 → (p = 2 ∧ q = 7) :=
by
  intros p q hp hq h
  sorry

end prime_solution_l631_631428


namespace bobs_mile_time_l631_631020

theorem bobs_mile_time :
  let sisters_time := 9 * 60 + 42 in
  let improvement_percent := 9.062499999999996 / 100 in
  let bobs_current_time_sec := (1 + improvement_percent) * sisters_time in
  (bobs_current_time_sec = 634.5) :=
by
  let sisters_time := 9 * 60 + 42
  let improvement_percent := 9.062499999999996 / 100
  let bobs_current_time_sec := (1 + improvement_percent) * sisters_time
  calc
    bobs_current_time_sec = (1 + improvement_percent) * sisters_time : rfl
    ... = 1.090625 * sisters_time : by rw [improvement_percent_def]
    ... = 1.090625 * 582 : by rw [sisters_time_def]
    ... = 634.5 : by norm_num

end bobs_mile_time_l631_631020


namespace contradiction_properties_l631_631749

-- Definitions based on the conditions of the problem.
def contradictory_sides_depend (x : Type) : Prop := sorry
def contradictory_sides_interpenetrate (x : Type) : Prop := sorry
def contradictory_sides_transform (x : Type) : Prop := sorry
def contradiction_has_specificity (x : Type) : Prop := sorry

-- Conditions derived from the problem statement.
variables (scenario : Type)
axiom mother_cried_and_father_laughed : (contradictory_sides_interpenetrate scenario) ∧ (contradiction_has_specificity scenario)

-- The goal is to show that the correct options are ② and ④, implying:
theorem contradiction_properties
    (mother_cried_and_father_laughed : (contradictory_sides_interpenetrate scenario) ∧ (contradiction_has_specificity scenario))
    : (contradictory_sides_interpenetrate scenario) ∧ (contradiction_has_specificity scenario) :=
by
    apply mother_cried_and_father_laughed
sorry

end contradiction_properties_l631_631749


namespace tan_435_eq_2_add_sqrt_3_l631_631398

theorem tan_435_eq_2_add_sqrt_3 :
  Real.tan (435 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  sorry

end tan_435_eq_2_add_sqrt_3_l631_631398


namespace find_x_l631_631636

theorem find_x (x : ℝ) (α : ℝ) (h1 : sin α = 4 / 5) (h2 : P : (ℝ, ℝ) := (x, 4)) : x = 3 ∨ x = -3 :=
by
  sorry

end find_x_l631_631636


namespace population_of_Beacon_l631_631610

-- Defining the populations of Richmond, Victoria, and Beacon
variables (Richmond Victoria Beacon : ℕ)

-- Given conditions
def condition1 : Prop := Richmond = Victoria + 1000
def condition2 : Prop := Victoria = 4 * Beacon
def condition3  : Prop := Richmond = 3000

-- The theorem to prove the population of Beacon
theorem population_of_Beacon 
  (h1 : condition1) 
  (h2 : condition2) 
  (h3 : condition3) : 
  Beacon = 500 :=
sorry

end population_of_Beacon_l631_631610


namespace number_of_valid_schedules_l631_631346

-- Definitions based on given conditions
def lessons := ["Chinese", "Mathematics", "English", "Music", "PE"] -- List of lessons
def is_adjacent (a b : String) (l : List String) : Prop := 
  ∃ i, i < (l.length - 1) ∧ l[i] = a ∧ l[i+1] = b

def is_not_adjacent (a b : String) (l : List String) : Prop :=
  ∀ i, i < (l.length - 1) → ¬(l[i] = a ∧ l[i+1] = b) 

def is_valid_schedule (schedule : List String) : Prop :=
  ("Mathematics" ≠ schedule.head) ∧
  (is_adjacent "Chinese" "English" schedule ∨ is_adjacent "English" "Chinese" schedule) ∧
  (is_not_adjacent "Music" "PE" schedule ∧ is_not_adjacent "PE" "Music" schedule)

-- Problem statement
theorem number_of_valid_schedules : 
  ∃ (schedules : Finset (List String)), 
  schedules.card = 20 ∧ 
  ∀ schedule ∈ schedules, is_valid_schedule schedule :=
sorry

end number_of_valid_schedules_l631_631346


namespace employee_c_budgeted_time_l631_631937

theorem employee_c_budgeted_time : 
  ∀ (weekly_hours : ℕ) (weeks : ℕ) (budget : ℝ) 
    (rate_A : ℝ) (rate_B : ℝ) (rate_C : ℝ)
    (fraction_A : ℝ) (fraction_B : ℝ),
  weekly_hours = 40 →
  weeks = 10 →
  budget = 20000 →
  rate_A = 30 →
  rate_B = 40 →
  rate_C = 50 →
  fraction_A = 1/2 →
  fraction_B = 1/3 →
  let total_hours := (weekly_hours * weeks : ℕ) in
  let hours_A := (fraction_A * total_hours) in
  let hours_B := (fraction_B * total_hours) in
  let cost_A := (hours_A * rate_A) in
  let cost_B := (hours_B * rate_B) in
  let remaining_budget := (budget - (cost_A + cost_B)) in
  let hours_C := (remaining_budget / rate_C) in
  let fraction_C := (hours_C / total_hours) in
  fraction_C ≈ 0.43335 :=
begin
  intros weekly_hours weeks budget rate_A rate_B rate_C fraction_A fraction_B
         h1 h2 h3 h4 h5 h6 h7 h8,
  let total_hours := (weekly_hours * weeks : ℕ),
  let hours_A := (fraction_A * total_hours : ℝ),
  let hours_B := (fraction_B * total_hours : ℝ),
  let cost_A := (hours_A * rate_A : ℝ),
  let cost_B := (hours_B * rate_B : ℝ),
  let remaining_budget := (budget - (cost_A + cost_B) : ℝ),
  let hours_C := (remaining_budget / rate_C : ℝ),
  let fraction_C := (hours_C / total_hours : ℝ),
  sorry -- proof
end

end employee_c_budgeted_time_l631_631937


namespace find_greatest_integer_l631_631092

theorem find_greatest_integer :
  (M n : ℕ),
  ((∑ k in {3, 4, 5, 6, 7, 8, 9, 10}, 1 / (k! * (21 - k)!)) = (M / (2! * 19!))) →
  (⌊M / 100⌋ = 1048) :=
by
  sorry

end find_greatest_integer_l631_631092


namespace number_contains_digit_7_l631_631872

noncomputable def contains_digit (d n : ℕ) : Prop :=
  ∃ k, n / 10^k % 10 = d

noncomputable def count_numbers_with_digit (d bound : ℕ) : ℕ :=
  (finset.range (bound + 1)).filter (λ n, contains_digit d n).card

theorem number_contains_digit_7 : count_numbers_with_digit 7 800 = 152 := 
sorry

end number_contains_digit_7_l631_631872


namespace sum_of_solutions_l631_631560

theorem sum_of_solutions :
  let T := ∑ x in { x : ℝ | 0 < x ∧ x ^ (3 ^ (real.sqrt 3)) = (real.sqrt 3) ^ (3 ^ x) }, x
  in T = real.sqrt 3 := by
  sorry

end sum_of_solutions_l631_631560


namespace triangle_ratio_l631_631733

open Real

theorem triangle_ratio (AC BC : ℝ) (h1 : AC = 5) (h2 : BC = 5)
  (ABD : ∀ (A B D : ℝ), ABD ≠ 0) (h3 : ABD = 15) :
  let AB := sqrt (AC^2 + BC^2) in ∀ (DB DE : ℝ), 
  (∀ x y : ℝ, ABD x y * (AB x y) > 0) ∧ (∀ (x y : ℝ) (r : ℝ), DB x y = sqrt (r^2 - AB^2)) → 
  ∃ m n : ℕ, m.coprime n ∧ (m + n = 2) := 
by {
  sorry
}

end triangle_ratio_l631_631733


namespace polynomial_integers_for_all_x_l631_631482

noncomputable def p (a b c d x : ℤ) : ℤ := a * x^3 + b * x^2 + c * x + d

theorem polynomial_integers_for_all_x (a b c d : ℤ) (h_neg1 : p a b c d (-1) ∈ ℤ)
                                          (h_0 : p a b c d 0 ∈ ℤ)
                                          (h_1 : p a b c d 1 ∈ ℤ)
                                          (h_2 : p a b c d 2 ∈ ℤ) :
  ∀ x : ℤ, p a b c d x ∈ ℤ := by 
sorry

end polynomial_integers_for_all_x_l631_631482


namespace calculate_train_length_l631_631001

noncomputable def train_length (speed_kmph : ℕ) (time_secs : ℝ) (bridge_length_m : ℝ) : ℝ :=
  let speed_mps := (speed_kmph * 1000) / 3600
  let total_distance := speed_mps * time_secs
  total_distance - bridge_length_m

theorem calculate_train_length :
  train_length 60 14.998800095992321 140 = 110 :=
by
  sorry

end calculate_train_length_l631_631001


namespace geometric_sequence_b_sum_first_n_terms_l631_631081

def a (n : ℕ) : ℤ :=
  if n = 1 then -2
  else if n > 1 then 2 * (a (n - 1)) + 4
  else 0  -- Not defined for n < 1, added for completeness.

def b (n : ℕ) : ℤ := a n + 4

theorem geometric_sequence_b
  (n : ℕ) : 
  b (n + 1) = 2 * b n :=
by {
  sorry
}

theorem sum_first_n_terms
  (n : ℕ) : 
  let sn := (finset.range n).sum (λ i, (abs (a (i + 1)))) 
  in sn = 2^(n + 1) - 4 * n + 2 :=
by {
  sorry
}

end geometric_sequence_b_sum_first_n_terms_l631_631081


namespace difference_between_percent_and_value_is_five_l631_631282

def hogs : ℕ := 75
def ratio : ℕ := 3

def num_of_cats (hogs : ℕ) (ratio : ℕ) : ℕ := hogs / ratio

def cats : ℕ := num_of_cats hogs ratio

def percent_of_cats (cats : ℕ) : ℝ := 0.60 * cats
def value_to_subtract : ℕ := 10

def difference (percent : ℝ) (value : ℕ) : ℝ := percent - value

theorem difference_between_percent_and_value_is_five
    (hogs : ℕ)
    (ratio : ℕ)
    (cats : ℕ := num_of_cats hogs ratio)
    (percent : ℝ := percent_of_cats cats)
    (value : ℕ := value_to_subtract)
    :
    difference percent value = 5 :=
by {
    sorry
}

end difference_between_percent_and_value_is_five_l631_631282


namespace guppies_total_l631_631488

theorem guppies_total :
  let haylee := 3 * 12
  let jose := haylee / 2
  let charliz := jose / 3
  let nicolai := charliz * 4
  haylee + jose + charliz + nicolai = 84 :=
by
  sorry

end guppies_total_l631_631488


namespace lemonade_third_intermission_l631_631415

theorem lemonade_third_intermission (a b c T : ℝ) (h1 : a = 0.25) (h2 : b = 0.42) (h3 : T = 0.92) (h4 : T = a + b + c) : c = 0.25 :=
by
  sorry

end lemonade_third_intermission_l631_631415


namespace height_dependent_distances_l631_631224

variable {A B C D E F : Type} 
variable (a b c : ℝ) -- sides of triangle 
variable (h_A h_B h_C : ℝ) -- heights of triangle 
variable (α β γ : ℝ) --angles

-- Triangle setup with given sides and heights
axiom triangle_existence (A B C : Type) 
  (h_A h_B h_C : ℝ)

-- Conditions given in the problem
-- Define the points
axiom height_foot_D {AB : Type} (D : AB)
axiom perp_point_E {AC : Type} (E : AC)
axiom perp_point_F {BC : Type} (F : BC)

-- Given conditions on the angles
axiom angle_A {α : ℝ}
axiom angle_B {β : ℝ}
axiom angle_C {γ : ℝ}

-- The distances DE and DF
axiom distance_DE {h_C : ℝ} {γ : ℝ} : ℝ := h_C * real.sin γ
axiom distance_DF {h_C : ℝ} {γ : ℝ} : ℝ := h_C * real.sin γ

-- Main theorem
theorem height_dependent_distances 
  (h_C : ℝ) (γ : ℝ) : (distance_DE = distance_DF)
:= sorry

end height_dependent_distances_l631_631224


namespace sum_smallest_largest_3_digit_numbers_made_up_of_1_2_5_l631_631438

theorem sum_smallest_largest_3_digit_numbers_made_up_of_1_2_5 :
  let smallest := 125
  let largest := 521
  smallest + largest = 646 := by
  sorry

end sum_smallest_largest_3_digit_numbers_made_up_of_1_2_5_l631_631438


namespace consecutive_probability_l631_631788

def BoxA := {1, 2, 3, 4}
def BoxB := {2, 5}

def is_consecutive (x y : ℕ) : Prop :=
  (x = y + 1) ∨ (x = y - 1)

theorem consecutive_probability : 
  let events := list.product BoxA.toList BoxB.toList in
  let consecutive_events := list.filter (λ (p : ℕ × ℕ), is_consecutive p.fst p.snd) events in
  ∃ (p : ℚ), p = consecutive_events.length / events.length ∧ p = 3 / 8 :=
by
  sorry

end consecutive_probability_l631_631788


namespace inequality_holds_equality_condition_l631_631570

theorem inequality_holds (a b : ℝ) (hb : b ≠ -1) (hb' : b ≠ 0) :
  (1 + a)^2 / (1 + b) ≤ 1 + a^2 / b ↔ 
  (b ∈ Iio (-1) ∨ b ∈ Ioi 0) :=
sorry

theorem equality_condition (a b : ℝ) (hb : b ≠ -1) (hb' : b ≠ 0) :
  ((1 + a)^2 / (1 + b) = 1 + a^2 / b ↔ a = b) :=
sorry

end inequality_holds_equality_condition_l631_631570


namespace positive_difference_l631_631670

theorem positive_difference :
  let a := 8^2
  let term1 := (a + a) / 8
  let term2 := (a * a) / 8
  term2 - term1 = 496 :=
by
  let a := 8^2
  let term1 := (a + a) / 8
  let term2 := (a * a) / 8
  have h1 : a = 64 := rfl
  have h2 : term1 = 16 := by simp [a, term1]
  have h3 : term2 = 512 := by simp [a, term2]
  show 512 - 16 = 496 from sorry

end positive_difference_l631_631670


namespace solution_set_of_inequality_l631_631480

def f (x : ℝ) : ℝ :=
  if x < 0 then 2 * Real.exp x
  else Real.log (x+1) / Real.log 2 + 2

theorem solution_set_of_inequality : {x : ℝ | f x > 4} = {x : ℝ | 3 < x} :=
by
  sorry

end solution_set_of_inequality_l631_631480


namespace teagan_total_cost_l631_631997

theorem teagan_total_cost :
  let reduction_percentage := 20
  let original_price_shirt := 60
  let original_price_jacket := 90
  let reduced_price_shirt := original_price_shirt * (100 - reduction_percentage) / 100
  let reduced_price_jacket := original_price_jacket * (100 - reduction_percentage) / 100
  let cost_5_shirts := 5 * reduced_price_shirt
  let cost_10_jackets := 10 * reduced_price_jacket
  let total_cost := cost_5_shirts + cost_10_jackets
  total_cost = 960 := by
  sorry

end teagan_total_cost_l631_631997


namespace percentage_of_motorists_speeding_l631_631319

-- Definitions based on the conditions
def total_motorists : Nat := 100
def percent_motorists_receive_tickets : Real := 0.20
def percent_speeders_no_tickets : Real := 0.20

-- Define the variables for the number of speeders
variable (x : Real) -- the percentage of total motorists who speed 

-- Lean statement to formalize the problem
theorem percentage_of_motorists_speeding 
  (h1 : 20 = (0.80 * x) * (total_motorists / 100)) : 
  x = 25 :=
sorry

end percentage_of_motorists_speeding_l631_631319


namespace positive_difference_of_fractions_l631_631693

theorem positive_difference_of_fractions : 
  (let a := 8^2 in (a + a) / 8) = 16 ∧ (let a := 8^2 in (a * a) / 8) = 512 →
  (let a := 8^2 in ((a * a) / 8 - (a + a) / 8)) = 496 := 
by
  sorry

end positive_difference_of_fractions_l631_631693


namespace number_of_valid_bijections_l631_631211

/-- Let the set \( \mathbf{A} = \{1, 2, 3, 4, 5, 6\} \).
    A bijection \( f: \mathbf{A} \rightarrow \mathbf{A} \) satisfies \( f(f(f(x))) = x \) for any \( x \in \mathbf{A} \).
    The number of such bijections is \( 81 \). -/
theorem number_of_valid_bijections : 
  let A := {1, 2, 3, 4, 5, 6}
  in @Function.Injective A A f ∧ @Function.Surjective A A f ∧ (∀ x ∈ A, f (f (f x)) = x) 
→ (nat.count 
  (λ f : A → A, @Function.Bijective A A f ∧ ∀ x ∈ A, f (f (f x)) = x) eq 
  = 81) := 
begin
  sorry
end

end number_of_valid_bijections_l631_631211


namespace maximum_chord_length_at_t_minus_1_l631_631209

noncomputable def parabola (t: ℝ) (x: ℝ) (a: ℝ) (b: ℝ): ℝ := (t^2 + t + 1) * x^2 - 2 * (a + t)^2 * x + t^2 + 3 * a * t + b

theorem maximum_chord_length_at_t_minus_1 (a b : ℝ) (h₁ : a = 1) (h₂ : b = 1)
  (h₃ : ∀ t: ℝ, parabola t 1 a b = 0):
  ∃ t : ℝ, t = -1 ∧ ∀ t' : ℝ, parabola t' 1 a b = 0 → t' ≠ -1 → chord_length t' < chord_length (-1) :=
sorry

-- we ignore the definition of chord_length and implicitly assume it's correctly defined.

end maximum_chord_length_at_t_minus_1_l631_631209


namespace positive_difference_l631_631705

def a := 8^2
def b := a + a
def c := a * a
theorem positive_difference : ((b / 8) - (c / 8)) = 496 := by
  sorry

end positive_difference_l631_631705


namespace quotient_is_20_l631_631159

theorem quotient_is_20 (D d r Q : ℕ) (hD : D = 725) (hd : d = 36) (hr : r = 5) (h : D = d * Q + r) :
  Q = 20 :=
by sorry

end quotient_is_20_l631_631159


namespace total_guppies_correct_l631_631489

noncomputable def total_guppies : ℕ :=
  let haylee := 3 * 12 in
  let jose := haylee / 2 in
  let charliz := jose / 3 in
  let nicolai := charliz * 4 in
  haylee + jose + charliz + nicolai

theorem total_guppies_correct : total_guppies = 84 :=
by 
  -- skip proof
  sorry

end total_guppies_correct_l631_631489


namespace three_lines_intersect_at_one_point_l631_631416

theorem three_lines_intersect_at_one_point (lines : Fin 9 → (ℝ × ℝ) → (ℝ × ℝ))
  (square : Set (ℝ × ℝ))
  (h1 : ∀ i, (lines i) divides square into two quadrilaterals in ratio 2:3) :
  ∃ p : ℝ × ℝ, ∃ l1 l2 l3 : Fin 9, l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 ∧
    p ∈ (lines l1) ∩ (lines l2) ∩ (lines l3) := 
sorry

end three_lines_intersect_at_one_point_l631_631416


namespace number_to_multiply_l631_631150

theorem number_to_multiply (a b x : ℝ) (h1 : x * a = 4 * b) (h2 : a * b ≠ 0) (h3 : a / 4 = b / 3) : x = 3 :=
sorry

end number_to_multiply_l631_631150


namespace line_equation_l631_631256

-- Define the point P
def P : ℝ × ℝ := (-3, 1)

-- Define the given line equation 2x + 3y - 5 = 0
def given_line (x y : ℝ) : Prop := 2 * x + 3 * y - 5 = 0

-- Define the slope of the given line
def same_slope_line (c : ℝ) (x y : ℝ) : Prop := 2 * x + 3 * y + c = 0

-- Attribute to the conditions
def point_on_line (line : ℝ → ℝ → Prop) (point : ℝ × ℝ) : Prop := line point.1 point.2

-- The proof statement
theorem line_equation :
  ∃ c : ℝ, same_slope_line c P.1 P.2 ∧ (∀ x y : ℝ, same_slope_line c x y ↔ (2 * x + 3 * y + c = 0)) :=
begin
  use 3,
  split,
  { show same_slope_line 3 P.1 P.2,
    simp [P, same_slope_line],
    norm_num,
    refl, },
  { intros x y,
    show same_slope_line 3 x y ↔ (2 * x + 3 * y + 3 = 0),
    simp [same_slope_line],
    exact iff.rfl, }
end

end line_equation_l631_631256


namespace count_true_statements_l631_631794

theorem count_true_statements (a b c d : ℝ) : 
  (∃ (H1 : a ≠ b) (H2 : c ≠ d), a + c = b + d) →
  ((a ≠ b) ∧ (c ≠ d) → a + c ≠ b + d) = false ∧ 
  ((a + c ≠ b + d) → (a ≠ b) ∧ (c ≠ d)) = false ∧ 
  (∃ (H3 : a = b) (H4 : c = d), a + c ≠ b + d) = false ∧ 
  ((a + c = b + d) → (a = b) ∨ (c = d)) = false → 
  number_of_true_statements = 0 := 
by
  sorry

end count_true_statements_l631_631794


namespace parabola_equation_l631_631637

theorem parabola_equation (vertex : ℝ × ℝ)
  (axis_of_symmetry : Prop)
  (focus_line : ℝ × ℝ → Prop)
  (eq_line : focus_line = (λ p, p.1 - p.2 + 2 = 0)) :
  vertex = (0, 0) → 
  (axis_of_symmetry = (∀ x, true) ∨ axis_of_symmetry = (∀ y, true)) →
  focus_line (0, 2) →
  ∃ a : ℝ, (vertex = (0, 0) ∧ axis_of_symmetry (λ x, true) ∧ focus_line (a, 0) ∨ 
            vertex = (0, 0) ∧ axis_of_symmetry (λ y, true) ∧ focus_line (0, a)) →
            (0, 2) = (0, 2) →
            (0, 2) ∈ {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 2} ∧
            (eq_line = (λ p, p.1 - p.2 + 2 = 0)) →
            ∃ y : ℝ, x * x = 8 * y :=
by sorry

end parabola_equation_l631_631637


namespace train_pass_time_l631_631773

-- Problem Statement:
-- A train 200 meters long is running with a speed of 60 kmph.
-- In what time will it pass a man who is running at 6 kmph in the direction opposite to that in which the train is going?
theorem train_pass_time :
  let length_of_train := 200 -- meters
  let speed_of_train_kmph := 60 -- kmph
  let speed_of_man_kmph := 6 -- kmph
  let conversion_factor := 1000 / 3600 -- to convert kmph to m/s
  let relative_speed_kmph := speed_of_train_kmph + speed_of_man_kmph
  let relative_speed_mps := relative_speed_kmph * conversion_factor
  let time_to_pass := length_of_train / relative_speed_mps
  time_to_pass ≈ 10.91 -- approximately 10.91 seconds
:=
begin
  sorry
end

end train_pass_time_l631_631773


namespace trajectory_is_ellipse_l631_631978

noncomputable def fixed_point_1 : ℝ × ℝ := (-4, 0)
noncomputable def fixed_point_2 : ℝ × ℝ := (4, 0)
def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def point_satisfies_condition (M : ℝ × ℝ) : Prop :=
  distance M fixed_point_1 + distance M fixed_point_2 = 8

theorem trajectory_is_ellipse (M : ℝ × ℝ) (h : point_satisfies_condition M) : ∃ a b c : ℝ, (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ 
(separating_curve : (x : ℝ) × (y : ℝ) -> Prop := λ p, a * p.1 ^ 2 + b * p.2 ^ 2 + c = 0) ∧ separating_curve M := 
begin
  sorry
end

end trajectory_is_ellipse_l631_631978


namespace sin_sum_div_sin_eq_four_thirds_l631_631953

-- Define the variables and theorem
theorem sin_sum_div_sin_eq_four_thirds 
    {a b c A B C : ℝ}
    (h1 : b + c = 4 * ((a + b + c) / 15))
    (h2 : c + a = 5 * ((a + b + c) / 15))
    (h3 : a + b = 6 * ((a + b + c) / 15))
    (h4 : ∀ (x : ℝ), x > 0 → (a / sin A = b / sin B ∧ b / sin B = c / sin C))
    : (sin A + sin C) / sin B = 4 / 3 := 
by {
  sorry
}

end sin_sum_div_sin_eq_four_thirds_l631_631953


namespace positive_difference_l631_631680

theorem positive_difference (a k : ℕ) (h1 : a = 8^2) (h2 : k = 8) :
  abs ((a + a) / k - (a * a) / k) = 496 :=
by
  sorry

end positive_difference_l631_631680


namespace matrix_solution_exists_l631_631460

theorem matrix_solution_exists :
  ∃ X : Matrix (Fin 2) (Fin 1) ℤ, 
    let A := matrix.of ![![1, -2], ![-2, -1]],
        B := matrix.of ![![5], ![-15]]
    in A * X = B := 
sorry

end matrix_solution_exists_l631_631460


namespace exists_ai_eq_one_l631_631975

theorem exists_ai_eq_one
  (a : ℕ → ℕ)
  (h1 : ∀ i, 1 ≤ a i ∧ a i ≤ 2016)
  (h2 : ∀ n, n ≤ 2^2016 → (∏ i in Finset.range n, a i) + 1 = m * m ∧ ∃ m : ℕ) :
  ∃ i, a i = 1 :=
sorry

end exists_ai_eq_one_l631_631975


namespace product_of_consecutive_integers_is_square_l631_631225

theorem product_of_consecutive_integers_is_square (x : ℤ) : 
  x * (x + 1) * (x + 2) * (x + 3) + 1 = (x^2 + 3 * x + 1) ^ 2 :=
by
  sorry

end product_of_consecutive_integers_is_square_l631_631225


namespace Archie_started_with_100_marbles_l631_631015

theorem Archie_started_with_100_marbles
  (M : ℕ) 
  (h1 : 0.60 * M + (0.50 * 0.40 * M) + 20 = M) 
  (h2 : 0.20 * M = 20) : 
  M = 100 :=
by
  sorry

end Archie_started_with_100_marbles_l631_631015


namespace christmas_decor_l631_631994

theorem christmas_decor (total_balls : ℕ) (num_colors : ℕ) (x : ℕ) (balls_double : ℕ) 
                        (hc : num_colors = 15) (hb : total_balls = 600) 
                        (first_colors_eq : ∀ n, n < 10 → balls_per_color n = x)
                        (last_colors_eq : ∀ n, 10 ≤ n ∧ n < 15 → balls_per_color n = balls_double)
                        (double_relation : balls_double = 2 * x) :
                        (10 * x + 5 * balls_double = total_balls) → (x = 30 ∧ balls_double = 60) := 
by {
  intros h,
  sorry
}

end christmas_decor_l631_631994


namespace angle_AA1K_eq_angle_BB1M_l631_631173

-- Defining the conditions and the problem
variables (A B C A1 B1 M K : Point)
variables [Triangle ABC]

-- Assume the given conditions
variables (h_altAA1 : is_altitude A ABC A1)
variables (h_altBB1 : is_altitude B ABC B1)
variables (h_onABM : on_line_segment AB M)
variables (h_onABK : on_line_segment AB K)
variables (h_parallelB1K_BC : parallel B1K BC)
variables (h_parallelA1M_AC : parallel A1M AC)

-- The main theorem to prove
theorem angle_AA1K_eq_angle_BB1M :
  ∠ (A, A1, K) = ∠ (B, B1, M) :=
sorry

end angle_AA1K_eq_angle_BB1M_l631_631173


namespace coloring_problem_l631_631655

def colors := {1, 2, 3, 4, 5}

def adjacency : list (char × char) := [(A, B), (A, C), (B, C), (B, D), (C, D), (C, E), (D, E)]

noncomputable def count_colorings : ℕ :=
  let possible_colorings := (colors.product colors.product colors.product colors.product colors).filter (λ ⟨⟨⟨⟨a, b⟩, c⟩, d⟩, e⟩,
    (adjacency.all (λ (x : char × char), x.1 == a ∨ x.2 == b ∨ x.3 == c ∨ x.4 == d ∨ x.5 == e))
    && (a ≠ b)
    && (a ≠ c)
    && (b ≠ c)
    && (b ≠ d)
    && (c ≠ d)
    && (c ≠ e)
    && (d ≠ e)) in
  possible_colorings.card

theorem coloring_problem : count_colorings = 540 := sorry

end coloring_problem_l631_631655


namespace exists_number_in_seq_appearing_infinitely_often_l631_631325

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

noncomputable def polynomial_sum_of_digits (P : ℕ → ℕ) (n : ℕ) : ℕ :=
  sum_of_digits (P n)

theorem exists_number_in_seq_appearing_infinitely_often 
  (P : ℕ → ℕ) 
  (h : ∀ n, ∃ c, P n = c) : 
  ∃ a, ∀ N, ∃ n > N, polynomial_sum_of_digits P n = a :=
by 
  sorry

end exists_number_in_seq_appearing_infinitely_often_l631_631325


namespace axis_of_symmetry_imp_cond_l631_631257

-- Necessary definitions
variables {p q r s x y : ℝ}

-- Given conditions
def curve_eq (x y p q r s : ℝ) : Prop := y = (2 * p * x + q) / (r * x + 2 * s)
def axis_of_symmetry (x y : ℝ) : Prop := y = x

-- Main statement
theorem axis_of_symmetry_imp_cond (h1 : curve_eq x y p q r s) (h2 : axis_of_symmetry x y) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) : p = -2 * s :=
sorry

end axis_of_symmetry_imp_cond_l631_631257


namespace mandy_accepted_schools_l631_631991

theorem mandy_accepted_schools (total_schools : ℕ) (fraction_applied : ℚ) (fraction_accepted : ℚ) 
    (h_total : total_schools = 42) (h_fraction_applied : fraction_applied = 1/3) 
    (h_fraction_accepted : fraction_accepted = 1/2) : 
    let applied_schools := total_schools * fraction_applied in
    let accepted_schools := applied_schools * fraction_accepted in
    accepted_schools = 7 :=
by
  sorry

end mandy_accepted_schools_l631_631991


namespace number_of_numbers_with_digit_seven_l631_631894

-- Define what it means to contain digit 7
def contains_digit_seven (n : ℕ) : Prop :=
  n.digits 10 ∈ [7]

-- Define the set of numbers from 1 to 800 containing at least one digit 7
def numbers_with_digit_seven : ℕ → Prop :=
  λ n, 1 ≤ n ∧ n ≤ 800 ∧ contains_digit_seven n

-- State the theorem
theorem number_of_numbers_with_digit_seven : (finset.filter numbers_with_digit_seven (finset.range 801)).card = 152 :=
sorry

end number_of_numbers_with_digit_seven_l631_631894


namespace tangent_length_from_point_to_circle_l631_631123

structure Point (α : Type) := (x : α) (y : α)

def distance {α : Type} [Real] (p1 p2 : Point α) : Real :=
  Real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2)

def circle (α : Type) (center : Point α) (radius : Real) := 
  {p : Point α // distance p center = radius}

theorem tangent_length_from_point_to_circle :
  ∀ (A : Point Real) (center : Point Real) (r : Real),
  A = Point.mk (-1) 4 →
  center = Point.mk 2 3 →
  r = 1 →
  (distance A center)^2 - r^2 = 9 := 
by
  intros
  rw [distance, Real.sqrt, Real.pow, Real.pow, sub, sub, Real.sqrt, sub_eq]
  dsimp [Point.x, Point.y, Point.mk] at *
  sorry

end tangent_length_from_point_to_circle_l631_631123


namespace positive_difference_l631_631674

theorem positive_difference (a k : ℕ) (h1 : a = 8^2) (h2 : k = 8) :
  abs ((a + a) / k - (a * a) / k) = 496 :=
by
  sorry

end positive_difference_l631_631674


namespace pentagon_area_l631_631408

theorem pentagon_area 
  (edge_length : ℝ) 
  (triangle_height : ℝ) 
  (n_pentagons : ℕ) 
  (equal_convex_pentagons : ℕ) 
  (pentagon_area : ℝ) : 
  edge_length = 5 ∧ triangle_height = 2 ∧ n_pentagons = 5 ∧ equal_convex_pentagons = 5 → pentagon_area = 30 := 
by
  sorry

end pentagon_area_l631_631408


namespace arithmetic_geometric_sequences_l631_631199

theorem arithmetic_geometric_sequences (
  {a_n b_n : ℕ → ℝ}
  (h_arith : ∀ n, a_n = a_1 + (n - 1) * (a_2 - a_1))
  (h_geom : ∀ n, b_n = b_1 * (b_2 / b_1) ^ (n - 1))
  (h_pos_a : ∀ n, a_n > 0)
  (h_pos_b : ∀ n, b_n > 0)
  (h_eq1 : a_1 = b_1)
  (h_eq2015 : a_2015 = b_2015))
  : a_1008 ≥ b_1008 :=
by
  sorry

end arithmetic_geometric_sequences_l631_631199


namespace problem_part_1_problem_part_2_l631_631070

noncomputable def z (θ : ℝ) : ℂ := complex.of_real (cos θ) + complex.I * complex.of_real (sin θ)

theorem problem_part_1 (θ : ℝ) (hθ : 0 < θ ∧ θ < π) (hz : complex.abs (z θ + 1) = 1) :
  1 + z θ + (z θ) ^ 2 = 0 := sorry

noncomputable def point_A (θ : ℝ) : ℂ := z θ
noncomputable def point_B (θ : ℝ) : ℂ := -2 * complex.conj (z θ)
noncomputable def point_C : ℂ := 1 + z (θ) + (z θ) ^ 2

theorem problem_part_2 (θ : ℝ) (hθ : 0 < θ ∧ θ < π) (hz : complex.abs (z θ + 1) = 1) :
  real.triangle_area (complex.re (point_A θ), complex.im (point_A θ))
                    (complex.re (point_B θ), complex.im (point_B θ))
                    (complex.re (point_C θ), complex.im (point_C θ)) = (real.sqrt 3) / 2 := sorry

end problem_part_1_problem_part_2_l631_631070


namespace find_theta_l631_631130

-- Define the vectors a and b
def vector_a (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 * Real.sin θ)
def vector_b : ℝ × ℝ := (3, Real.sqrt 3)

-- Define collinearity condition
def are_collinear (u v : ℝ × ℝ) : Prop := ∃ λ : ℝ, u = (λ * v.1, λ * v.2)

-- The main theorem to be proved
theorem find_theta (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi) :
  are_collinear (vector_a θ) vector_b ↔ θ = Real.pi / 6 ∨ θ = 7 * Real.pi / 6 := by
  sorry

end find_theta_l631_631130


namespace smallest_degree_l631_631597

-- Define the roots
def root1 : Real := 2 - Real.sqrt 3
def root2 : Real := -2 - Real.sqrt 3
def root3 : Real := 1 + Real.sqrt 5
def root4 : Real := 1 - Real.sqrt 5

-- Statement of the theorem, using the roots and the desired polynomial degree
theorem smallest_degree (p : Polynomial ℚ) (h1 : p.is_root root1) (h2 : p.is_root root2)
  (h3 : p.is_root root3) (h4 : p.is_root root4) :
  ∃ d : ℕ, Polynomial.degree p = d ∧ d = 6 :=
begin
  sorry  -- Proof to be provided
end

end smallest_degree_l631_631597


namespace fruit_trees_l631_631760

theorem fruit_trees (total_streets : ℕ) 
  (fruit_trees_every_other : total_streets % 2 = 0) 
  (equal_fruit_trees : ∀ n : ℕ, 3 * n = total_streets / 2) : 
  ∃ n : ℕ, n = total_streets / 6 :=
by
  sorry

end fruit_trees_l631_631760


namespace min_value_of_sum_inverses_l631_631071

theorem min_value_of_sum_inverses (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : real.sqrt 3 = real.sqrt (3^a * 3^b)) :
  1/a + 1/b >= 4 :=
by
  sorry

end min_value_of_sum_inverses_l631_631071


namespace trapezoid_angles_l631_631163

theorem trapezoid_angles :
  ∀ (A B C D : Point) (AD BC : Segment),
  isosceles_trapezoid A B C D ∧ parallel AD BC ∧
  (dist A (CD C D) = length (leg A B)) ∧
  (AD.length / BC.length = 5) →
  (angles_of_trapezoid A B C D = [arccos (1 / √5), π - arccos (1 / √5)] ∨
   angles_of_trapezoid A B C D = [arccos (2 / √5), π - arccos (2 / √5)]) :=
by
  sorry

end trapezoid_angles_l631_631163


namespace positive_difference_l631_631703

def a := 8^2
def b := a + a
def c := a * a
theorem positive_difference : ((b / 8) - (c / 8)) = 496 := by
  sorry

end positive_difference_l631_631703


namespace compare_logs_l631_631447

theorem compare_logs (a b c : ℝ) 
    (h1 : a = log 2 3 + log 2 (sqrt 3))
    (h2 : b = log 2 9 - log 2 (sqrt 3))
    (h3 : c = log 3 2) : a = b ∧ b > c :=
by {
  sorry
}

end compare_logs_l631_631447


namespace probability_overlap_l631_631038

theorem probability_overlap (Ada_birth Grace_birth : ℕ) (hAda : Ada_birth ∈ {0..720}) (hGrace : Grace_birth ∈ {0..720}) :
  (∃ (Ada_life Grace_life : set ℕ), Ada_life = {Ada_birth..Ada_birth+79} ∧ Grace_life = {Grace_birth..Grace_birth+79} ∧ Ada_life ∩ Grace_life ≠ ∅) →
  (∃ (ratio : ℚ), ratio = 25/28) :=
by sorry

end probability_overlap_l631_631038


namespace number_contains_digit_7_l631_631868

noncomputable def contains_digit (d n : ℕ) : Prop :=
  ∃ k, n / 10^k % 10 = d

noncomputable def count_numbers_with_digit (d bound : ℕ) : ℕ :=
  (finset.range (bound + 1)).filter (λ n, contains_digit d n).card

theorem number_contains_digit_7 : count_numbers_with_digit 7 800 = 152 := 
sorry

end number_contains_digit_7_l631_631868


namespace fish_to_rice_l631_631933

variables (f l r e : ℚ)

theorem fish_to_rice (h1: 4 * f = 3 * l) (h2: 5 * l = 7 * r) : f = (21 / 20) * r :=
by
  -- Step through the process of solving for f in terms of r using the given conditions.
  sorry

end fish_to_rice_l631_631933


namespace number_contains_digit_7_l631_631871

noncomputable def contains_digit (d n : ℕ) : Prop :=
  ∃ k, n / 10^k % 10 = d

noncomputable def count_numbers_with_digit (d bound : ℕ) : ℕ :=
  (finset.range (bound + 1)).filter (λ n, contains_digit d n).card

theorem number_contains_digit_7 : count_numbers_with_digit 7 800 = 152 := 
sorry

end number_contains_digit_7_l631_631871


namespace black_squares_in_29th_row_l631_631403

-- Definition of the pattern for number of squares in the nth row
def num_squares (n : ℕ) : ℕ := 1 + 2 * (n - 1)

-- Definition of the number of black squares given total squares in a row
def num_black_squares (total_squares : ℕ) : ℕ := (total_squares - 1) / 2

theorem black_squares_in_29th_row : num_black_squares (num_squares 29) = 28 :=
by
  -- We calculate the total number of squares in the 29th row.
  -- num_squares 29 = 1 + 2 * (29 - 1) = 57
  have h1 : num_squares 29 = 57 := by sorry
  
  -- Calculating the number of black squares from the total squares.
  -- num_black_squares 57 = (57 - 1) / 2 = 28
  have h2 : num_black_squares 57 = 28 := by sorry
  
  -- Substituting the intermediate results to assert the final theorem.
  show num_black_squares (num_squares 29) = 28 from by
    rw [h1, h2]

end black_squares_in_29th_row_l631_631403


namespace minimum_b_value_minimum_b_value_l631_631852

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + x + b

theorem minimum_b_value (a b : ℝ) (h1 : f 1 a b = 1) (h2 : deriv (fun x => f x a b) 1 = 0) : b = 1 :=
by 
  -- Importing the necessary library

  -- Setting the function f(x)
  noncomputable def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + x + b

  -- Defining the theorem with given conditions
  theorem minimum_b_value (a b : ℝ) (h1 : f 1 a b = 1) (h2 : deriv (fun x => f x a b) 1 = 0) : b = 1 := sorry

end minimum_b_value_minimum_b_value_l631_631852


namespace four_d_minus_c_l631_631613

theorem four_d_minus_c {c d : ℕ} (h1 : c > d) (h2 : Polynomial.X ^ 2 - 20 * Polynomial.X + 96 = Polynomial.C (c) * Polynomial.X + Polynomial.C (-d) * Polynomial.X + Polynomial.C (c * d)) : 4 * d - c = 20 :=
sorry

end four_d_minus_c_l631_631613


namespace beacon_population_l631_631607

variables (Richmond Victoria Beacon : ℕ)

theorem beacon_population :
  (Richmond = Victoria + 1000) →
  (Victoria = 4 * Beacon) →
  (Richmond = 3000) →
  (Beacon = 500) :=
by
  intros h1 h2 h3
  sorry

end beacon_population_l631_631607


namespace quadrilateral_possible_perimeters_l631_631064

theorem quadrilateral_possible_perimeters (p : ℕ) [Fact (p < 300)] :
  ∃ (n : ℕ), n = 16 ∧
  ∀ (AB BC AD CD : ℕ), AB = 3 ∧ CD = 2 * AD ∧
  right_angle (B) ∧ right_angle (C) ∧
  ∃ (x y : ℕ), BC = x ∧ AD = y ∧
  x^2 + (y - 3)^2 = y^2 ∧
  p = 3 + x + y + 2 * y ∧ p < 300 :=
sorry

end quadrilateral_possible_perimeters_l631_631064


namespace quadratic_polynomial_eval_l631_631208

def q (x : ℂ) : ℂ := sorry -- Placeholder for the actual quadratic polynomial

theorem quadratic_polynomial_eval :
  (q x)^2 + 3*x = 0 → x ∈ {0, 2, 5} →
  q(8) = i * ((16 * sqrt(15) - 24 * sqrt(6) + 40 * sqrt(6) + 20 * sqrt(15)) / 30) := sorry

end quadratic_polynomial_eval_l631_631208


namespace binomial_expansion_m_value_l631_631105

theorem binomial_expansion_m_value
  (m : ℝ) (n : ℕ)
  (h1 : (mx + 1)^n.coeff 3 = 448)
  (h2 : n = 8) :
  m = 2 :=
by sorry

end binomial_expansion_m_value_l631_631105


namespace positive_difference_l631_631689

theorem positive_difference : 496 = abs ((64 + 64) / 8 - (64 * 64) / 8) := by
  have h1 : 8^2 = 64 := rfl
  have h2 : 64 + 64 = 128 := rfl
  have h3 : (128 : ℕ) / 8 = 16 := rfl
  have h4 : 64 * 64 = 4096 := rfl
  have h5 : (4096 : ℕ) / 8 = 512 := rfl
  have h6 : 512 - 16 = 496 := rfl
  sorry

end positive_difference_l631_631689


namespace correct_equation_l631_631652

variables (x : ℝ) (production_planned total_clothings : ℝ)
variables (increase_rate days_ahead : ℝ)

noncomputable def daily_production (x : ℝ) := x
noncomputable def total_production := 1000
noncomputable def production_per_day_due_to_overtime (x : ℝ) := x * (1 + 0.2 : ℝ)
noncomputable def original_completion_days (x : ℝ) := total_production / daily_production x
noncomputable def increased_production_completion_days (x : ℝ) := total_production / production_per_day_due_to_overtime x
noncomputable def days_difference := original_completion_days x - increased_production_completion_days x

theorem correct_equation : days_difference x = 2 := by
  sorry

end correct_equation_l631_631652


namespace time_to_pass_telegraph_post_l631_631318

def conversion_factor_km_per_hour_to_m_per_sec := 1000 / 3600

noncomputable def train_length := 70
noncomputable def train_speed_kmph := 36

noncomputable def train_speed_m_per_sec := train_speed_kmph * conversion_factor_km_per_hour_to_m_per_sec

theorem time_to_pass_telegraph_post : (train_length / train_speed_m_per_sec) = 7 := by
  sorry

end time_to_pass_telegraph_post_l631_631318


namespace man_older_than_son_l631_631355

variables (M S : ℕ)

theorem man_older_than_son
  (h_son_age : S = 26)
  (h_future_age : M + 2 = 2 * (S + 2)) :
  M - S = 28 :=
by sorry

end man_older_than_son_l631_631355


namespace tangent_line_to_circle_l631_631110

theorem tangent_line_to_circle :
  ∀ (m b: ℝ),
  (let center := (1, m),
       r := Real.sqrt (m^2 - 2 * m + 2) in
   r = 1 ∧ (∀ (x y : ℝ), y = x + b → (x - 1)^2 + (y - m)^2 = 1) → b = Real.sqrt 2 ∨ b = -Real.sqrt 2) :=
begin
  intros m b,
  let center := (1, m),
  let r := Real.sqrt (m^2 - 2 * m + 2),
  assume h,
  sorry -- Proof here
end

end tangent_line_to_circle_l631_631110


namespace count_numbers_with_digit_7_in_range_l631_631878

theorem count_numbers_with_digit_7_in_range : 
  let numbers_in_range := {n : ℕ | 1 ≤ n ∧ n ≤ 800}
      contains_digit_7 (n : ℕ) : Prop := n.digits 10.contains 7
  in (finset.filter (λ n, contains_digit_7 n) (finset.range 801)).card = 152 :=
by 
  let numbers_in_range := {n : ℕ | 1 ≤ n ∧ n ≤ 800}
  let contains_digit_7 (n : ℕ) : Prop := n.digits 10.contains 7
  have h := (finset.filter (λ n, contains_digit_7 n) (finset.range 801)).card
  sorry

end count_numbers_with_digit_7_in_range_l631_631878


namespace cost_of_bananas_and_cantaloupe_l631_631970

-- Let a, b, c, and d be real numbers representing the prices of apples, bananas, cantaloupe, and dates respectively.
variables (a b c d : ℝ)

-- Conditions given in the problem
axiom h1 : a + b + c + d = 40
axiom h2 : d = 3 * a
axiom h3 : c = (a + b) / 2

-- Goal is to prove that the sum of the prices of bananas and cantaloupe is 8 dollars.
theorem cost_of_bananas_and_cantaloupe : b + c = 8 :=
by
  sorry

end cost_of_bananas_and_cantaloupe_l631_631970


namespace CC₁_bisects_angle_QC₁P_l631_631598

variable {A B C P Q R C₁ : Type}
variable [Triangle ABC] [Triangle PQR]
variable (h1 : is_tangent_to_circumcircle A ABC)
variable (h2 : is_tangent_to_circumcircle B ABC)
variable (h3 : is_tangent_to_circumcircle C ABC)
variable (h4 : is_tangent_to_triangle PQR (side PQ C) (side PR B) (side QR A))
variable (h5 : is_foot_of_altitude C₁ C (side AB ABC))

theorem CC₁_bisects_angle_QC₁P : ∀ (C : Point), ∃ (CC₁ : Line), bisects_angle CC₁ (angle Q C₁ P) :=
by
  sorry

end CC₁_bisects_angle_QC₁P_l631_631598


namespace tan_sin_sum_l631_631031

theorem tan_sin_sum (θ : ℝ) (hθ : θ = 10 * π / 180) :
  tan θ + 4 * sin θ ≈ 1.355 :=
by
  sorry

end tan_sin_sum_l631_631031


namespace systematic_sampling_l631_631160

def interval (N : ℕ) (n : ℕ) : ℕ := N / n

theorem systematic_sampling
  (N n first_draw : ℕ)
  (h1 : N = 1000)
  (h2 : n = 50)
  (h3 : first_draw = 15) :
  let k := interval N n in
  first_draw + 20 * k = 415 :=
by
  sorry

end systematic_sampling_l631_631160


namespace isosceles_triangle_perimeter_l631_631164

theorem isosceles_triangle_perimeter {a b : ℕ} (h₁ : a = 4) (h₂ : b = 9) (h₃ : ∀ x y z : ℕ, 
  (x = a ∧ y = a ∧ z = b) ∨ (x = b ∧ y = b ∧ z = a) → 
  (x + y > z ∧ x + z > y ∧ y + z > x)) : 
  (a = 4 ∧ b = 9) → a + a + b = 22 :=
by sorry

end isosceles_triangle_perimeter_l631_631164


namespace distance_between_points_l631_631297

-- Define the points
def point1 := (1 : ℤ, 3 : ℤ)
def point2 := (-5 : ℤ, 7 : ℤ)

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℤ × ℤ) : ℤ := 
  Int.sqrt (((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2).toNat)

-- Define the problem statement
theorem distance_between_points : distance point1 point2 = 2 * (Int.sqrt 13) := by
  sorry

end distance_between_points_l631_631297


namespace sum_A_B_equals_1_l631_631948

-- Definitions for the digits and the properties defined in conditions
variables (A B C D : ℕ)
variable (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
variable (h_digit_bounds : A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10)
noncomputable def ABCD := 1000 * A + 100 * B + 10 * C + D
axiom h_mult : ABCD * 2 = ABCD * 10

theorem sum_A_B_equals_1 : A + B = 1 :=
by
  sorry

end sum_A_B_equals_1_l631_631948


namespace change_received_is_zero_l631_631973

noncomputable def combined_money : ℝ := 10 + 8
noncomputable def cost_chicken_wings : ℝ := 6
noncomputable def cost_chicken_salad : ℝ := 4
noncomputable def cost_cheeseburgers : ℝ := 2 * 3.50
noncomputable def cost_fries : ℝ := 2
noncomputable def cost_sodas : ℝ := 2 * 1.00
noncomputable def total_cost_before_discount : ℝ := cost_chicken_wings + cost_chicken_salad + cost_cheeseburgers + cost_fries + cost_sodas
noncomputable def discount_rate : ℝ := 0.15
noncomputable def tax_rate : ℝ := 0.08
noncomputable def discounted_total : ℝ := total_cost_before_discount * (1 - discount_rate)
noncomputable def tax_amount : ℝ := discounted_total * tax_rate
noncomputable def total_cost_after_tax : ℝ := discounted_total + tax_amount

theorem change_received_is_zero : combined_money < total_cost_after_tax → 0 = combined_money - total_cost_after_tax + combined_money := by
  intros h
  sorry

end change_received_is_zero_l631_631973


namespace constant_term_eq_fifteen_l631_631947

theorem constant_term_eq_fifteen (n : ℕ) :
  (∃ k : ℕ, (Nat.choose n k) * (-1) ^ (n - k) = 15 ∧ 3 * k = n) ↔ n = 6 :=
by
  sorry

end constant_term_eq_fifteen_l631_631947


namespace coefficient_x2_sum_l631_631790

theorem coefficient_x2_sum :
  (nat.choose 2 2) + (nat.choose 3 2) + (nat.choose 4 2) + (nat.choose 5 2) + (nat.choose 6 2)
  = 35 :=
by
  sorry

end coefficient_x2_sum_l631_631790


namespace sam_seashell_count_l631_631731

/-!
# Problem statement:
-/
def initialSeashells := 35
def seashellsGivenToJoan := 18
def seashellsFoundToday := 20
def seashellsGivenToTom := 5

/-!
# Proof goal: Prove that the current number of seashells Sam has is 32.
-/
theorem sam_seashell_count :
  initialSeashells - seashellsGivenToJoan + seashellsFoundToday - seashellsGivenToTom = 32 :=
  sorry

end sam_seashell_count_l631_631731


namespace triangle_angles_with_origin_l631_631741

-- Define the lines
def L1 (x y : ℝ) : Prop := x + 19 * y = -123
def L2 (x y : ℝ) : Prop := 14 * x - 15 * y = -36
def L3 (x y : ℝ) : Prop := 15 * x + 4 * y = 122

-- Define the intersection points, note that in practice we would solve for these points given the equations
def P1 : ℝ × ℝ := (-9, -6)
def P2 : ℝ × ℝ := (10, -7)
def P3 : ℝ × ℝ := (6, 8)

-- Define the slopes from the origin to these points
def slope (P : ℝ × ℝ) : ℝ := P.2 / P.1

def m1 : ℝ := slope P1 -- 2 / 3
def m2 : ℝ := slope P2 -- -7 / 10
def m3 : ℝ := slope P3 -- 4 / 3

-- Define the angles
def theta (m1 m2 : ℝ) : ℝ := Real.arctan ((m1 - m2) / (1 + m1 * m2))

def theta12 : ℝ := theta m1 m2
def theta13 : ℝ := theta m1 m3
def theta23 : ℝ := theta m2 m3

-- The set of the angles formed between the lines and connecting the vertices of the triangle with the origin
def angles_correct : ℝ × ℝ × ℝ :=
  (theta12, theta13, theta23)

-- Now, state the theorem
theorem triangle_angles_with_origin :
  angles_correct = (Real.arctan (41 / 32), Real.arctan (6 / 17), Real.arctan 30.5) :=
sorry

end triangle_angles_with_origin_l631_631741


namespace Megan_deleted_pictures_l631_631730

/--
Megan took 15 pictures at the zoo and 18 at the museum. She still has 2 pictures from her vacation.
Prove that Megan deleted 31 pictures.
-/
theorem Megan_deleted_pictures :
  let zoo_pictures := 15
  let museum_pictures := 18
  let remaining_pictures := 2
  let total_pictures := zoo_pictures + museum_pictures
  let deleted_pictures := total_pictures - remaining_pictures
  deleted_pictures = 31 :=
by
  sorry

end Megan_deleted_pictures_l631_631730


namespace mandy_accepted_is_7_l631_631988

def mandy_accepted_schools 
  (total_schools : ℕ) 
  (fraction_applied : ℚ) 
  (fraction_accepted : ℚ) : ℕ :=
  let schools_applied := (fraction_applied * total_schools).toNat in
  let schools_accepted := (fraction_accepted * schools_applied).toNat in
  schools_accepted

theorem mandy_accepted_is_7 : 
  mandy_accepted_schools 42 (1/3 : ℚ) (1/2 : ℚ) = 7 :=
by
  sorry

end mandy_accepted_is_7_l631_631988


namespace sequence_inequality_l631_631010

open Real

theorem sequence_inequality (x : ℕ → ℝ)
  (pos : ∀ n, 0 < x n)
  (sum_inequality : ∀ n, (∑ i in Finset.range n, x i) ≥ sqrt n) :
  ∀ n, (∑ i in Finset.range n, (x i)^2) > (1 / 4) * (∑ i in Finset.range n, 1 / (i + 1)) :=
by {
  sorry
}

end sequence_inequality_l631_631010


namespace janice_total_cost_is_correct_l631_631966

def cost_of_items (cost_juices : ℕ) (juices : ℕ) (cost_sandwiches : ℕ) (sandwiches : ℕ) (cost_pastries : ℕ) (pastries : ℕ) (cost_salad : ℕ) (discount_salad : ℕ) : ℕ :=
  let one_sandwich := cost_sandwiches / sandwiches
  let one_juice := cost_juices / juices
  let total_pastries := pastries * cost_pastries
  let discounted_salad := cost_salad - (cost_salad * discount_salad / 100)
  one_sandwich + one_juice + total_pastries + discounted_salad

-- Conditions
def cost_juices := 10
def juices := 5
def cost_sandwiches := 6
def sandwiches := 2
def cost_pastries := 4
def pastries := 2
def cost_salad := 8
def discount_salad := 20

-- Expected Total Cost
def expected_total_cost := 1940 -- in cents to avoid float numbers

theorem janice_total_cost_is_correct : 
  cost_of_items cost_juices juices cost_sandwiches sandwiches cost_pastries pastries cost_salad discount_salad = expected_total_cost :=
by
  simp [cost_of_items, cost_juices, juices, cost_sandwiches, sandwiches, cost_pastries, pastries, cost_salad, discount_salad]
  norm_num
  sorry

end janice_total_cost_is_correct_l631_631966


namespace sport_formulation_water_l631_631951

theorem sport_formulation_water
  (f c w : ℕ)  -- flavoring, corn syrup, and water respectively in standard formulation
  (f_s c_s w_s : ℕ)  -- flavoring, corn syrup, and water respectively in sport formulation
  (corn_syrup_sport : ℤ) -- amount of corn syrup in sport formulation in ounces
  (h_std_ratio : f = 1 ∧ c = 12 ∧ w = 30) -- given standard formulation ratios
  (h_sport_fc_ratio : f_s * 4 = c_s) -- sport formulation flavoring to corn syrup ratio
  (h_sport_fw_ratio : f_s * 60 = w_s) -- sport formulation flavoring to water ratio
  (h_corn_syrup_sport : c_s = corn_syrup_sport) -- amount of corn syrup in sport formulation
  : w_s = 30 := 
by 
  sorry

end sport_formulation_water_l631_631951


namespace probability_both_blue_buttons_l631_631182

theorem probability_both_blue_buttons :
  let initial_red_C := 6
  let initial_blue_C := 12
  let initial_total_C := initial_red_C + initial_blue_C
  let remaining_fraction_C := 2 / 3
  let remaining_total_C := initial_total_C * remaining_fraction_C
  let removed_buttons := initial_total_C - remaining_total_C
  let removed_red := removed_buttons / 2
  let removed_blue := removed_buttons / 2
  let remaining_blue_C := initial_blue_C - removed_blue
  let total_remaining_C := remaining_total_C
  let probability_blue_C := remaining_blue_C / total_remaining_C
  let probability_blue_D := removed_blue / removed_buttons
  probability_blue_C * probability_blue_D = 3 / 8 :=
by
  sorry

end probability_both_blue_buttons_l631_631182


namespace least_number_addition_l631_631300

def divisible_by (a b : ℕ) := ∃ c : ℕ, a = b * c

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem least_number_addition : 
  ∀ m, m = 729 →
  ∀ n, n = 1056 →
  ∀ a b, a = 35 → b = 51 → 
  let l := lcm a b in
  divisible_by (m + n) l :=
by
  intros m hm n hn a ha b hb
  have h1 : m = 729 := hm
  have h2 : n = 1056 := hn
  have h3 : a = 35 := ha
  have h4 : b = 51 := hb
  let l := lcm a b
  have h_lcm : l = lcm 35 51 := by rw [h3, h4]
  sorry

end least_number_addition_l631_631300


namespace gcd_lcm_product_75_90_l631_631022

theorem gcd_lcm_product_75_90 :
  let a := 75
  let b := 90
  let gcd_ab := Int.gcd a b
  let lcm_ab := Int.lcm a b
  gcd_ab * lcm_ab = 6750 :=
by
  let a := 75
  let b := 90
  let gcd_ab := Int.gcd a b
  let lcm_ab := Int.lcm a b
  sorry

end gcd_lcm_product_75_90_l631_631022


namespace hyperbola_focus_asymptote_distance_l631_631834

open Real

/-- Given the hyperbola defined by y^2 - m * x^2 = 3 * m where m > 0,
    prove that the distance from the focus of the hyperbola to one of its asymptotes is √3. -/
theorem hyperbola_focus_asymptote_distance (m : ℝ) (hm : 0 < m) : 
  let F := (0, sqrt (3 * m + 3))
  let asymptote := λ x, sqrt m * x
  √ 3 = abs (F.2) / sqrt (1 + m) :=
begin
  rw [F, asymptote],
  sorry
end

end hyperbola_focus_asymptote_distance_l631_631834


namespace symmetric_function_value_l631_631838

def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x * (1 - x) else sorry

theorem symmetric_function_value :
  (∀ x : ℝ, f x = f (2 - x)) ∧ (∀ x : ℝ, x ≥ 1 → f x = x * (1 - x)) →
  f (-2) = -12 :=
by
  intros h
  sorry

end symmetric_function_value_l631_631838


namespace ellipse_equation_oa_ob_range_area_quadrilateral_ABCD_l631_631457

-- Definition of the Ellipse
def ellipse_eq (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Constants for the problem
constants (a b c : ℝ)
constant ellipse_eccentricity : a > b > 0 → (c / a = sqrt 3 / 2)
constant area_rhombus : 2 * a * 2 * b = 8 -- Note: simplifying the equation of the area directly to '8' for clarity
constant basic_ellipse_eq : a^2 = b^2 + c^2

-- Conditions for the quadrilateral ABCD
constants (k_AC k_BD : ℝ)
constant k_condition : k_AC * k_BD = -1 / 4

-- Theorems to prove
theorem ellipse_equation (h1 : a > b > 0) (h2 : c / a = sqrt 3 / 2) (h3 : 2 * a * 2 * b = 8) (h4 : a^2 = b^2 + c^2) :
  ellipse_eq 1 1 2 1 := sorry

theorem oa_ob_range (h1 : a > b > 0) (h2 : c / a = sqrt 3 / 2) (h3 : 2 * a * 2 * b = 8) (h4 : a^2 = b^2 + c^2)
  (h5 : k_AC * k_BD = -1 / 4) : ∃ x : ℝ, x ∈ Set.Icc (-3/2) 0 ∪ Set.Ioc 0 (3/2) ∧ x ≠ 0 := sorry

theorem area_quadrilateral_ABCD (h1 : a > b > 0) (h2 : c / a = sqrt 3 / 2) (h3 : 2 * a * 2 * b = 8) (h4 : a^2 = b^2 + c^2)
  (h5 : k_AC * k_BD = -1 / 4) : 2 * 1 * 2 * 1 = 4 := sorry

end ellipse_equation_oa_ob_range_area_quadrilateral_ABCD_l631_631457


namespace base_8_not_divisible_by_3_l631_631306

theorem base_8_not_divisible_by_3 : 
  ∀ (b : ℕ), b = 8 → ¬ (∃ k : ℕ, ((2*b^3 + 2*b) - 2*b^2) = 3*k) :=
by
  intro b
  intro h
  rw h
  apply Exists.elim
  intro k hk
  suffices : 3 ∣ (2*8^2), from sorry,
  sorry

end base_8_not_divisible_by_3_l631_631306


namespace infinite_product_value_l631_631043

noncomputable def infinite_product := ∏ (n : ℕ) in (range ∞), (3^(n+1)/(4^(n+1)))

theorem infinite_product_value :
  infinite_product = real.rpow 81 (1/9) :=
sorry

end infinite_product_value_l631_631043


namespace length_of_LD_l631_631455

theorem length_of_LD
  (s : ℝ) -- side length of the square ABCD
  (L K B D : ℝ × ℝ)
  (hABCD_square : L = (s, 0) ∧ D = (0, 0))
  (hK_on_extension : K = (0, -6))
  (h_angle_KBL : ∠K B L = 90)
  (hKD : dist K D = 19)
  (hCL : dist (s, 0) L = 6)
  :
  dist L D = 7 := by sorry

end length_of_LD_l631_631455


namespace math_equivalent_proof_problem_l631_631116

/-
Given the following four propositions:  
P1: Tossing a coin twice, let event A be "both tosses result in heads" and event B be "both tosses result in tails", then events A and B are complementary events,  
P2: Events A and B are mutually exclusive events,  
P3: Among 10 products, there are 3 defective ones. If 3 products are randomly selected, let event A be "at most 2 of the selected products are defective" and event B be "at least 2 of the selected products are defective", then events A and B are mutually exclusive events,  
P4: If events A and B satisfy P(A) + P(B) = 1, then A and B are complementary events,  
P5: If A and B are mutually exclusive events, then ∁A ∪ ∁B is a certain event.
-/

def P1 : Prop := ∀ (A B : Prop), (A ∧ B) ∨ (¬A ∧ ¬B) → false
def P2 : Prop := ∀ (A B : Prop), (A ∧ ¬B) ∨ (¬A ∧ B)
def P3 : Prop := ∀ (A B : Prop), (¬(A ∧ B))
def P4 : Prop := ∀ (A B : Prop), P(A) + P(B) = 1 → (A ∧ B) ∨ (¬A ∧ ¬B)
def P5 : Prop := ∀ (A B : Prop), (¬A ∧ ¬B) → true

def false_props := {1, 3, 4}

theorem math_equivalent_proof_problem : ∀ (P1 P3 P4 : Prop), false_props = {1, 3, 4} := by
  sorry

end math_equivalent_proof_problem_l631_631116


namespace square_of_complex_real_iff_l631_631271

theorem square_of_complex_real_iff (a b : ℝ) :
  ((a + b * complex.i) ^ 2).im = 0 ↔ a * b = 0 :=
by
  sorry

end square_of_complex_real_iff_l631_631271


namespace distance_between_points_l631_631294

def point1 : ℝ × ℝ := (1, 3)
def point2 : ℝ × ℝ := (-5, 7)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points :
  distance point1 point2 = 2 * real.sqrt 13 :=
by
  sorry

end distance_between_points_l631_631294


namespace sum_log_floor_l631_631392

theorem sum_log_floor : (∑ N in Finset.range 2048.succ, (⌊Real.log2 N⌋ + 1)) = 33057 := by
  sorry

end sum_log_floor_l631_631392


namespace tina_total_pens_l631_631289

theorem tina_total_pens : 
  let pink_pens := 12 in
  let green_pens := pink_pens - 9 in
  let blue_pens := green_pens + 3 in
  pink_pens + green_pens + blue_pens = 21 :=
by 
  let pink_pens := 12
  let green_pens := pink_pens - 9
  let blue_pens := green_pens + 3
  show pink_pens + green_pens + blue_pens = 21 from sorry

end tina_total_pens_l631_631289


namespace hexagon_angle_sum_l631_631135

theorem hexagon_angle_sum (a b c d e : ℝ) (Q : ℝ)
  (h_sum : a + b + c + d + e + Q = 720)
  (h_a : a = 130)
  (h_b : b = 95)
  (h_c : c = 122)
  (h_d : d = 108)
  (h_e : e = 114) : Q = 151 :=
by {
  rw [h_a, h_b, h_c, h_d, h_e] at h_sum,
  linarith
}

end hexagon_angle_sum_l631_631135


namespace no_real_solution_log_eq_l631_631148

theorem no_real_solution_log_eq (x : ℝ) (h1 : 0 ≤ x + 6) (h2 : 0 ≤ x - 2) : ¬ (log (x + 6) + log (x - 2) = log (x^2 - 3 * x - 18)) :=
by
  -- Placeholder for the actual proof
  sorry

end no_real_solution_log_eq_l631_631148


namespace count_divisors_41752_l631_631864

theorem count_divisors_41752 : 
  let D := [1, 2, 3, 4, 5, 6, 7, 8, 9] in
  let num := 41752 in
  (List.count (λ d, num % d = 0) D = 3) :=
by
  let D := [1, 2, 3, 4, 5, 6, 7, 8, 9]
  let num := 41752
  have filter_d := list.count (λ d, num % d = 0) D
  show filter_d = 3
  sorry

end count_divisors_41752_l631_631864


namespace number_of_positive_factors_50400_l631_631141

theorem number_of_positive_factors_50400 : 
    ∃ n : ℕ, n = 50400 ∧ (∏ p in (unique_prime_factors 50400), (multiplicity p 50400) + 1) = 108 := 
by sorry

end number_of_positive_factors_50400_l631_631141


namespace math_proof_problem_l631_631207

noncomputable def problem_statement {n : ℕ} (hn : n ≥ 2) (x : Fin n → ℝ) 
  (hx : ∀ i j, i ≠ j → x i ≠ x j) (hxpos : ∀ i, 0 < x i) : Prop :=
  ∃ (a : Fin n → ℤ), (∀ i, a i = -1 ∨ a i = 1) ∧ ∑ i, a i * (x i) ^ 2 > (∑ i, a i * x i) ^ 2

theorem math_proof_problem (n : ℕ) (hn : n ≥ 2) (x : Fin n → ℝ) 
  (hx : ∀ i j, i ≠ j → x i ≠ x j) (hxpos : ∀ i, 0 < x i) : 
  problem_statement hn x hx hxpos :=
sorry

end math_proof_problem_l631_631207


namespace arrangement_3_people_restricted_arrangement_girls_together_arrangement_girls_not_together_arrangement_l631_631276

-- Define the general conditions
def total_people : ℕ := 7
def boys : ℕ := 4
def girls : ℕ := 3

-- Problem 1: Number of ways to select 3 people and arrange them in a row
theorem arrangement_3_people : (total_people.choose 3) * 3.factorial = 210 := by
  sorry

-- Problem 2: Number of arrangements given restrictions on boy A and girl B
theorem restricted_arrangement : (total_people.factorial) - 2 * ((total_people - 1).factorial) + (total_people - 2).factorial = 3720 := by
  sorry

-- Problem 3: Number of arrangements where girls stand together
theorem girls_together_arrangement : 5.factorial * girls.factorial = 720 := by
  sorry

-- Problem 4: Number of arrangements where girls don't stand next to each other
theorem girls_not_together_arrangement : 4.factorial * (5.choose 3) = 1440 := by
  sorry

end arrangement_3_people_restricted_arrangement_girls_together_arrangement_girls_not_together_arrangement_l631_631276


namespace relationship_t_s_l631_631449

theorem relationship_t_s (a b : ℝ) : 
  let t := a + 2 * b
  let s := a + b^2 + 1
  t <= s :=
by
  sorry

end relationship_t_s_l631_631449


namespace number_of_numbers_with_digit_7_from_1_to_800_eq_233_l631_631892

def contains_digit (n d : ℕ) : Prop :=
  ∃ k, 10 ^ k > 0 ∧ d = (n / 10 ^ k) % 10

def numbers_without_digit (n d : ℕ) : finset ℕ :=
  (finset.range n).filter (λ x, ¬ contains_digit x d)

def count_numbers_with_digit (n d : ℕ) : ℕ :=
  n - (numbers_without_digit n d).card

theorem number_of_numbers_with_digit_7_from_1_to_800_eq_233 :
  count_numbers_with_digit 800 7 = 233 :=
  sorry

end number_of_numbers_with_digit_7_from_1_to_800_eq_233_l631_631892


namespace correct_equation_l631_631748

-- Define the conditions as assumptions
variable (x : ℕ) -- number of machines originally planned to be produced per day
variable (h1 : x > 0) -- ensure x is a positive number

-- The factory produces 30 more machines per day now
def new_production_rate := x + 30

-- Proving the equation based on given conditions
theorem correct_equation : (800 / new_production_rate) = (600 / x) :=
by
  -- You can assume the conditions as hypotheses here
  have h2 : new_production_rate = x + 30 := rfl
  have h3 : 800 / (x + 30) = 600 / x ∧ x > 0 := by sorry
  exact h3.left

end correct_equation_l631_631748


namespace circle_symmetric_eq_l631_631104

theorem circle_symmetric_eq : 
  ∀ (x y : ℝ), 
  let circle := (x - 1) ^ 2 + y ^ 2 = 1 in 
  (∀ (x y : ℝ), (x, y) ∈ circle → (-y, -x) ∈ circle) →
  x^2 + (y+1)^2 = 1 :=
begin
  sorry
end

end circle_symmetric_eq_l631_631104


namespace problem_statement_l631_631940

open Set

variable {P Q R S A B C D : Point}

def areas_correct (pqrs abcd : ℝ) :=
  pqrs = 2 * abcd

def lie_on_sides (P Q R S A B C D : Point) :=
  -- Conditions that specify points A, B, C, D lie on sides of PQRS
  -- (A on PQ, B on QR, C on RS, D on SP for example)
  sorry

theorem problem_statement
  (pqrs_area abcd_area : ℝ)
  (H1 : areas_correct pqrs_area abcd_area)
  (H2 : lie_on_sides P Q R S A B C D) :
  (parallel AC QR) ∨ (parallel BD PQ) :=
sorry

end problem_statement_l631_631940


namespace integral_twice_sqrt_minus_sin_eq_pi_l631_631792

open Real

theorem integral_twice_sqrt_minus_sin_eq_pi :
  ∫ x in -1..1, 2 * sqrt (1 - x^2) - sin x = π :=
by
  have h1 : ∫ x in -1..1, sqrt (1 - x^2) = π / 2 := sorry
  have h2 : ∫ x in -1..1, sin x = 0 := interval_integral.integral_sin
  calc
    ∫ x in -1..1, 2 * sqrt (1 - x^2) - sin x
      = 2 * ∫ x in -1..1, sqrt (1 - x^2) - ∫ x in -1..1, sin x : by
        simp only [interval_integral.integral_sub]
        congr
        simp [mul_comm]
      ... = 2 * (π / 2) - 0 : by
        rw [h1, h2]
      ... = π : by
        norm_num

end integral_twice_sqrt_minus_sin_eq_pi_l631_631792


namespace axis_of_symmetry_range_of_m_l631_631464

/-- The conditions given in the original mathematical problem -/
noncomputable def f (x : ℝ) : ℝ :=
  let OA := (2 * Real.cos x, Real.sqrt 3)
  let OB := (Real.sin x + Real.sqrt 3 * Real.cos x, -1)
  (OA.1 * OB.1 + OA.2 * OB.2) + 2

/-- Question 1: The axis of symmetry for the function f(x) -/
theorem axis_of_symmetry :
  ∃ k : ℤ, ∀ x : ℝ, (2 * x + Real.pi / 3 = Real.pi / 2 + k * Real.pi) ↔ (x = k * Real.pi / 2 + Real.pi / 12) :=
sorry

/-- Question 2: The range of m such that g(x) = f(x) + m has zero points for x in (0, π/2) -/
theorem range_of_m (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  (∃ c : ℝ, (f x + c = 0)) ↔ ( -4 ≤ c ∧ c < Real.sqrt 3 - 2) :=
sorry

end axis_of_symmetry_range_of_m_l631_631464


namespace hypotenuse_length_is_correct_l631_631220

noncomputable def hypotenuse_of_30_60_90_triangle (a : ℝ) (angle_opposite : ℝ) : ℝ :=
  if a = 10 ∧ angle_opposite = 60 then 20 * Real.sqrt 3 / 3 else 0

theorem hypotenuse_length_is_correct :
  hypotenuse_of_30_60_90_triangle 10 60 = 20 * Real.sqrt 3 / 3 :=
by
  simp [hypotenuse_of_30_60_90_triangle]
  norm_num
  rw [mul_div_assoc]
  sorry

end hypotenuse_length_is_correct_l631_631220


namespace triangle_area_l631_631995

theorem triangle_area (A B C D E G : Type)
  [MetricSpace A]
  [MetricSpace B]
  [MetricSpace C]
  [MetricSpace D]
  [MetricSpace E]
  [MetricSpace G]
  (hMedAD_BE : ⟂ (median A D) (median B E)) -- Perpendicular medians
  (h_AD : AD = 18)
  (h_BE : BE = 24) :
  area ABC = 288 :=
sorry

end triangle_area_l631_631995


namespace polynomial_form_l631_631807

noncomputable def polynomial_solution (P : ℝ → ℝ) :=
  ∀ a b c : ℝ, (a * b + b * c + c * a = 0) → (P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c))

theorem polynomial_form :
  ∀ (P : ℝ → ℝ), polynomial_solution P ↔ ∃ (a b : ℝ), ∀ x : ℝ, P x = a * x^2 + b * x^4 :=
by 
  sorry

end polynomial_form_l631_631807


namespace sum_of_circle_areas_l631_631405

theorem sum_of_circle_areas 
    (r s t : ℝ)
    (h1 : r + s = 6)
    (h2 : r + t = 8)
    (h3 : s + t = 10) : 
    (π * r^2 + π * s^2 + π * t^2) = 36 * π := 
by
    sorry

end sum_of_circle_areas_l631_631405


namespace solve_xy_l631_631246

theorem solve_xy (x y : ℕ) :
  (x^2 + (x + y)^2 = (x + 9)^2) ↔ (x = 0 ∧ y = 9) ∨ (x = 8 ∧ y = 7) ∨ (x = 20 ∧ y = 1) :=
by
  sorry

end solve_xy_l631_631246


namespace gary_had_stickers_at_first_l631_631067

-- Define the conditions as constants
def stickers_given_to_Lucy : ℕ := 42
def stickers_given_to_Alex : ℕ := 26
def stickers_left : ℕ := 31

-- The main statement: Prove the total number of stickers Gary had at first
theorem gary_had_stickers_at_first :
  let stickers_given_away := stickers_given_to_Lucy + stickers_given_to_Alex in
  let total_stickers := stickers_given_away + stickers_left in
  total_stickers = 99 :=
by
  -- Use sorry to skip the proof as only the statement is required
  sorry

end gary_had_stickers_at_first_l631_631067


namespace positive_difference_is_496_l631_631716

def square (n: ℕ) : ℕ := n * n
def term1 := (square 8 + square 8) / 8
def term2 := (square 8 * square 8) / 8
def positive_difference := abs (term2 - term1)

theorem positive_difference_is_496 : positive_difference = 496 :=
by
  -- This is where the proof would go
  sorry

end positive_difference_is_496_l631_631716


namespace d_value_l631_631513

theorem d_value (d : ℝ) : (∀ x : ℝ, 3 * (5 + d * x) = 15 * x + 15) ↔ (d = 5) := by 
sorry

end d_value_l631_631513


namespace area_of_set_S_l631_631034

noncomputable def set_S : Set ℂ :=
  {z : ℂ | ∃ w : ℂ, w.abs = 6 ∧ z = w - (1 / w)}

theorem area_of_set_S :
  ∃ S : Set ℂ, S = set_S ∧ ∃ area : ℝ, area = (1295 / 36) * Real.pi :=
by
  sorry

end area_of_set_S_l631_631034


namespace find_light_bulbs_l631_631262

noncomputable def binomial_prob := sorry

theorem find_light_bulbs (n : ℕ) (p : ℝ) : n = 7 ∧ p = 0.95 → 
  (binomial_prob n p 5) ≥ 0.99 :=
by sorry

-- Without actual implementation of binomial_prob, we use sorry.

end find_light_bulbs_l631_631262


namespace positive_difference_l631_631676

theorem positive_difference (a k : ℕ) (h1 : a = 8^2) (h2 : k = 8) :
  abs ((a + a) / k - (a * a) / k) = 496 :=
by
  sorry

end positive_difference_l631_631676


namespace heather_bicycling_time_l631_631509

theorem heather_bicycling_time (distance speed : ℕ) (h1 : distance = 96) (h2 : speed = 6) : 
(distance / speed) = 16 := by
  sorry

end heather_bicycling_time_l631_631509


namespace binomial_distribution_X_n_third_l631_631980

variables (X : ℕ → ℕ) (n : ℕ) (p : ℚ)
noncomputable def binomial_prob (n k : ℕ) (p : ℚ) := (nat.choose n k) * p^k * (1 - p)^(n-k)

theorem binomial_distribution_X_n_third (h1 : X ∼ B(n, 1/3)) (h2 : EX = 2) : 
  binomial_prob 6 2 (1/3) = 80/243 :=
by sorry

end binomial_distribution_X_n_third_l631_631980


namespace cosine_angle_l631_631128

noncomputable def a : ℝ × ℝ := (2, 1)
noncomputable def b : ℝ × ℝ := (1, 2)

theorem cosine_angle (a b : ℝ × ℝ) :
  let dot_product := a.1 * b.1 + a.2 * b.2 in
  let mag_a := Real.sqrt (a.1 ^ 2 + a.2 ^ 2) in
  let mag_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2) in
  let cos_angle := dot_product / (mag_a * mag_b) in
  cos_angle = 4 / 5 :=
by sorry

end cosine_angle_l631_631128


namespace neg_p_sufficient_not_necessary_neg_q_l631_631921

variables {x : ℝ}

def p := |x| > 1
def q := x < -2

theorem neg_p_sufficient_not_necessary_neg_q : (¬ p → ¬ q) ∧ ¬ (¬ q → ¬ p) :=
by
  sorry

end neg_p_sufficient_not_necessary_neg_q_l631_631921


namespace problem_l631_631986

variable {α : Type*} [Field α]

variable (f : α → α) (f_inv : α → α) (x : α)

-- Function f has an inverse function f_inv
def has_inverse (f : α → α) (f_inv : α → α) :=
  ∀ y, f (f_inv y) = y ∧ f_inv (f y) = y

-- Condition f(x) + f(-x) = 4
def f_symm_property (f : α → α) :=
  ∀ x, f(x) + f(-x) = 4

theorem problem (invf : has_inverse f f_inv) (symm_f : f_symm_property f) :
  f_inv(x - 3) + f_inv(7 - x) = 0 :=
sorry

end problem_l631_631986


namespace positive_difference_is_496_l631_631715

def square (n: ℕ) : ℕ := n * n
def term1 := (square 8 + square 8) / 8
def term2 := (square 8 * square 8) / 8
def positive_difference := abs (term2 - term1)

theorem positive_difference_is_496 : positive_difference = 496 :=
by
  -- This is where the proof would go
  sorry

end positive_difference_is_496_l631_631715


namespace TinaTotalPens_l631_631286

variable (p g b : ℕ)
axiom H1 : p = 12
axiom H2 : g = p - 9
axiom H3 : b = g + 3

theorem TinaTotalPens : p + g + b = 21 := by
  sorry

end TinaTotalPens_l631_631286


namespace distance_between_points_l631_631295

def point1 : ℝ × ℝ := (1, 3)
def point2 : ℝ × ℝ := (-5, 7)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points :
  distance point1 point2 = 2 * real.sqrt 13 :=
by
  sorry

end distance_between_points_l631_631295


namespace spinner_probability_divisible_by_4_l631_631000

theorem spinner_probability_divisible_by_4 :
  let outcomes := {1, 2, 3, 4}
  let total_possibilities := (finset.product outcomes outcomes).card * outcomes.card
  let valid_last_two_digits := {{1, 2}, {2, 4}, {3, 2}, {4, 4}}
  let valid_combinations := 
    finset.filter (λ (tens_units : nat × nat), tens_units ∈ valid_last_two_digits)
    (finset.product outcomes outcomes)
  let valid_numbers := valid_combinations.product outcomes
  ∃ prob : rat, prob = finset.card valid_numbers / total_possibilities ∧ prob = 1 / 4 :=
by
  sorry

end spinner_probability_divisible_by_4_l631_631000


namespace number_of_numbers_with_digit_7_from_1_to_800_eq_233_l631_631889

def contains_digit (n d : ℕ) : Prop :=
  ∃ k, 10 ^ k > 0 ∧ d = (n / 10 ^ k) % 10

def numbers_without_digit (n d : ℕ) : finset ℕ :=
  (finset.range n).filter (λ x, ¬ contains_digit x d)

def count_numbers_with_digit (n d : ℕ) : ℕ :=
  n - (numbers_without_digit n d).card

theorem number_of_numbers_with_digit_7_from_1_to_800_eq_233 :
  count_numbers_with_digit 800 7 = 233 :=
  sorry

end number_of_numbers_with_digit_7_from_1_to_800_eq_233_l631_631889


namespace exponent_power_rule_l631_631308

theorem exponent_power_rule (a b : ℝ) : (a * b^3)^2 = a^2 * b^6 :=
by sorry

end exponent_power_rule_l631_631308


namespace no_butterflies_count_l631_631334

variables (x y z w : ℕ)

def student_conditions : Prop :=
  x + y + z + w = 18 ∧
  x + 2 * y + 3 * z + 0 * w = 32 ∧
  x = y + 5 ∧
  x = z + 2

theorem no_butterflies_count : ∃ w, student_conditions x y z w → w = 1 :=
begin
  sorry
end

end no_butterflies_count_l631_631334


namespace sequence_sum_100_l631_631539

def periodic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) + a (n + 2) = some_constant

theorem sequence_sum_100 (a : ℕ → ℝ) (some_constant : ℝ) 
  (h_periodic : periodic_seq a)
  (h_7 : a 7 = 2)
  (h_9 : a 9 = 3)
  (h_98 : a 98 = 4) :
  (Finset.range 100).sum a = 299 :=
sorry

end sequence_sum_100_l631_631539


namespace minor_arc_circumference_l631_631561

noncomputable def radius : ℝ := 12
noncomputable def Z_point : ℝ := 90 -- angle XZY in degrees
noncomputable def full_circle : ℝ := 2 * Real.pi * radius

theorem minor_arc_circumference 
  (r : ℝ := radius)
  (angle_XZY : ℝ:= Z_point)
  (circumference : ℝ:= full_circle):
  (angle_XZY = 90) → 
  circumference = 2 * Real.pi * r → 
  (minor_arc_circumference := (180/360) * circumference)
  → minor_arc_circumference = 12 * Real.pi := 
sorry

end minor_arc_circumference_l631_631561


namespace ball_distribution_ways_l631_631330

theorem ball_distribution_ways :
  let balls := 5
  let boxes := 3
  (balls = 5) → (boxes = 3) →
  (∃ (ways : ℕ), ways = 21) :=
by
  intros h1 h2
  use 21
  sorry

end ball_distribution_ways_l631_631330


namespace count_true_propositions_l631_631642

theorem count_true_propositions :
  let p1 := ∀ x y : ℝ, x + y = 0 → ¬(x = -y ∧ y = -x) → false
      p2 := ∀ a b : ℝ, a^2 > b^2 → ¬(a > b) → false
      p3 := ∀ x : ℝ, x > -3 → ¬(x^2 - x - 6 ≤ 0) → false
      p4 := let a := (Real.sqrt 2)^Real.sqrt 2 in
            ∀ b : ℝ, a = a → b = Real.sqrt 2 → ¬(Real.rat.pow a b ∈ ℚ) → false
  in (if p1 then 1 else 0) + (if p2 then 1 else 0) + (if p3 then 1 else 0) + (if p4 then 1 else 0) = 1 :=
by
  sorry

end count_true_propositions_l631_631642


namespace max_value_expression_l631_631132

noncomputable def unit_vectors_on_plane (a b c : ℝ^3) : Prop :=
  ∥a∥ = 1 ∧ ∥b∥ = 1 ∧ ∥c∥ = 1 ∧ (a - b).angle c = 60

theorem max_value_expression (a b c : ℝ^3) 
  (h : unit_vectors_on_plane a b c) (h_dot : a ⬝ b = 1 / 2) :
  (a - b) ⬝ (a - 2 • c) ≤ 5 / 2 :=
sorry

end max_value_expression_l631_631132


namespace task_completion_l631_631327

theorem task_completion (x y z : ℝ) 
  (h1 : 1 / x + 1 / y = 1 / 2)
  (h2 : 1 / y + 1 / z = 1 / 4)
  (h3 : 1 / z + 1 / x = 5 / 12) :
  x = 3 := 
sorry

end task_completion_l631_631327


namespace sum_cube_eq_l631_631223

theorem sum_cube_eq (a b c : ℝ) (h : a + b + c = 0) : a^3 + b^3 + c^3 = 3 * a * b * c :=
by 
  sorry

end sum_cube_eq_l631_631223


namespace tangent_angle_between_line_m_and_plane_ABCD_l631_631536

variables (A B C D A1 B1 C1 D1 E F G : Type)
variables [cube AB1CD A1BC1D1] -- Assume a cube structure
variables [is_midpoint E A A1] -- E is midpoint of AA1
variables (D1F D1A1 D1G D1C1 : ℝ)
variables (D1F_ratio : D1F / D1A1 = 1/3)
variables (D1G_ratio : D1G / D1C1 = 1/3)
-- Definitions related to intersection line m and angle θ with tangent to plane ABCD
variables (m θ : Type)
-- To be proved
theorem tangent_angle_between_line_m_and_plane_ABCD :
  tangent θ = (3 * sqrt 58) / 58 :=
sorry

end tangent_angle_between_line_m_and_plane_ABCD_l631_631536


namespace quad_ak_eq_kd_bc_kd_l631_631631

variable {A B C D K : Type}
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace K]

variable {AB AD CD BC KD AK : ℝ}
variable (a₁ : AB = AD)
variable (a₂ : ∠ D AK = ∠ AB D)

theorem quad_ak_eq_kd_bc_kd (h : AB = AD) (h2 : ∠ D AK = ∠ AB D) : AK^2 = KD^2 + BC * KD := sorry

end quad_ak_eq_kd_bc_kd_l631_631631


namespace sum_of_number_and_its_square_is_20_l631_631153

theorem sum_of_number_and_its_square_is_20 (n : ℕ) (h : n = 4) : n + n^2 = 20 :=
by
  sorry

end sum_of_number_and_its_square_is_20_l631_631153


namespace geom_seq_min_val_l631_631526

-- Definition of geometric sequence with common ratio q
def geom_seq (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Main theorem
theorem geom_seq_min_val (a : ℕ → ℝ) (q : ℝ) 
  (h_pos : ∀ n : ℕ, 0 < a n)
  (h_geom : geom_seq a q)
  (h_cond : 2 * a 3 + a 2 - 2 * a 1 - a 0 = 8) :
  2 * a 4 + a 3 = 12 * Real.sqrt 3 :=
sorry

end geom_seq_min_val_l631_631526


namespace rationalize_denominator_l631_631228

theorem rationalize_denominator :
  let A := -12
  let B := 7
  let C := 9
  let D := 13
  let E := 5 in
  B < D ∧
  (12/5 * Real.sqrt 7) = ((1:ℚ) * Real.sqrt B / E * A) * (-1) ∧
  (9/5 * Real.sqrt 13) = (1:ℚ * Real.sqrt D / E * C) ∧
  (A + B + C + D + E = 22) :=
by
  sorry

end rationalize_denominator_l631_631228


namespace evaluate_expression_at_2_l631_631304

theorem evaluate_expression_at_2 :
  (let x := 2 in (x^2 - 3*x - 10) / (x - 4)) = 6 :=
by
  sorry

end evaluate_expression_at_2_l631_631304


namespace dot_product_of_a_and_b_l631_631835

variable (i j k : ℝ³)
variable (a b : ℝ³)
variable (dot : ℝ³ → ℝ³ → ℝ)

-- Unit vectors that are mutually perpendicular
axiom unit_vectors : (dot i i = 1) ∧ (dot j j = 1) ∧ (dot k k = 1) ∧ 
                     (dot i j = 0) ∧ (dot i k = 0) ∧ (dot j k = 0)

-- Definitions of the vectors a and b
def a := i + 2 • j - k
def b := 3 • i - j + 4 • k

-- Proof goal
theorem dot_product_of_a_and_b : dot a b = -3 := sorry

end dot_product_of_a_and_b_l631_631835


namespace food_to_water_ratio_l631_631186

-- Definitions based on the conditions
def initial_water : ℝ := 20
def initial_food : ℝ := 10
def initial_gear : ℝ := 20
def water_per_hour : ℝ := 2
def total_weight_after_six_hours : ℝ := 34
def time_passed : ℝ := 6

-- Theorem to prove the ratio of food eaten per hour to water drunk per hour is 2/3
theorem food_to_water_ratio :
  (total_weight_after_six_hours =
    let water_left := initial_water - time_passed * water_per_hour in
    let food_eaten_per_hour := (time_passed : ℝ) * x in
    let food_left := initial_food - food_eaten_per_hour in
    water_left + food_left + initial_gear) →
  2 / 3 = x := 
sorry

end food_to_water_ratio_l631_631186


namespace factorize_2070_l631_631915

-- Define the conditions
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100
def is_unique_factorization (n a b : ℕ) : Prop := a * b = n ∧ is_two_digit a ∧ is_two_digit b

-- The final theorem statement
theorem factorize_2070 : 
  (∃ a b : ℕ, is_unique_factorization 2070 a b) ∧ 
  ∀ a b : ℕ, is_unique_factorization 2070 a b → (a = 30 ∧ b = 69) ∨ (a = 69 ∧ b = 30) :=
by 
  sorry

end factorize_2070_l631_631915


namespace coefficient_of_x4_in_binomial_expansion_of_x_minus_2_div_x_pow_6_l631_631611

theorem coefficient_of_x4_in_binomial_expansion_of_x_minus_2_div_x_pow_6 :
  let expansion := (x - (2 : ℝ) / x) ^ 6 in
  ∑ k in finset.range 7, (binom 6 k) * (-2) ^ k * x ^ (6 - 2 * k) = expansion →
  x^4 := -12 := sorry

end coefficient_of_x4_in_binomial_expansion_of_x_minus_2_div_x_pow_6_l631_631611


namespace triangle_XYZ_l631_631175

variable (X Y Z : Type)
variable [InnerProductSpace ℝ X]
variable [InnerProductSpace ℝ Y]
variable [InnerProductSpace ℝ Z]

theorem triangle_XYZ {XY XZ : ℝ} (hY : angle Y X Z = π / 4) (hZ : angle X Y Z = π / 2) (hXY : XY = 6) : XZ = 3 * Real.sqrt 2 :=
sorry

end triangle_XYZ_l631_631175


namespace f_31_eq_neg1_l631_631825

def f (x : ℝ) : ℝ := sorry  

axiom f_odd : ∀ x, f (-x) = -f x
axiom f_symmetry : ∀ x, f (1 + x) = f (1 - x)
axiom f_in_interval : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = Real.log2 (x + 1)

theorem f_31_eq_neg1 : f 31 = -1 := sorry

end f_31_eq_neg1_l631_631825


namespace units_digit_of_a_l631_631521

theorem units_digit_of_a (a : ℕ) (ha : (∃ b : ℕ, 1 ≤ b ∧ b ≤ 9 ∧ (a*a / 10^1) % 10 = b)) : 
  ((a % 10 = 4) ∨ (a % 10 = 6)) :=
sorry

end units_digit_of_a_l631_631521


namespace positive_difference_l631_631668

theorem positive_difference :
  let a := 8^2
  let term1 := (a + a) / 8
  let term2 := (a * a) / 8
  term2 - term1 = 496 :=
by
  let a := 8^2
  let term1 := (a + a) / 8
  let term2 := (a * a) / 8
  have h1 : a = 64 := rfl
  have h2 : term1 = 16 := by simp [a, term1]
  have h3 : term2 = 512 := by simp [a, term2]
  show 512 - 16 = 496 from sorry

end positive_difference_l631_631668


namespace cos_of_complementary_angle_l631_631957

theorem cos_of_complementary_angle (Y Z : ℝ) (h : Y + Z = π / 2) 
  (sin_Y : Real.sin Y = 3 / 5) : Real.cos Z = 3 / 5 := 
  sorry

end cos_of_complementary_angle_l631_631957


namespace positive_difference_eq_496_l631_631708

theorem positive_difference_eq_496 : 
  let a := 8 ^ 2 in 
  (a + a) / 8 - (a * a) / 8 = 496 :=
by
  let a := 8^2
  have h1 : (a + a) / 8 = 16 := by sorry
  have h2 : (a * a) / 8 = 512 := by sorry
  show (a + a) / 8 - (a * a) / 8 = 496 from by
    calc
      (a + a) / 8 - (a * a) / 8
            = 16 - 512 : by rw [h1, h2]
        ... = -496 : by ring
        ... = 496 : by norm_num

end positive_difference_eq_496_l631_631708


namespace remainder_sum_l631_631486

theorem remainder_sum (x y z : ℕ) (h1 : x % 15 = 11) (h2 : y % 15 = 13) (h3 : z % 15 = 9) :
  ((2 * (x % 15) + (y % 15) + (z % 15)) % 15) = 14 :=
by
  sorry

end remainder_sum_l631_631486


namespace clothing_store_inventory_solution_l631_631347

/-- A clothing store inventory problem -/
def clothing_store_inventory_problem : Prop :=
  let ties := 34
  let belts := 40
  let black_shirts := 63
  let white_shirts := 42
  let hats := 25
  let socks := 80
  let jeans := (2 / 3 : ℝ) * (black_shirts + white_shirts)
  let scarves := (1 / 2 : ℝ) * (ties + belts)
  let jackets := hats + (0.2 : ℝ) * hats
  let combined_scarves_jackets := scarves + jackets
  in jeans - combined_scarves_jackets = 3

-- The main theorem statement
theorem clothing_store_inventory_solution : clothing_store_inventory_problem :=
by sorry

end clothing_store_inventory_solution_l631_631347


namespace cos_x_plus_2y_eq_one_l631_631823

theorem cos_x_plus_2y_eq_one
  (x y : ℝ) (a : ℝ)
  (hx : x ∈ Icc (-π/4) (π/4))
  (hy : y ∈ Icc (-π/4) (π/4))
  (h1 : x^3 + sin x - 2 * a = 0)
  (h2 : 4 * y^3 + sin y * cos y + a = 0) :
  cos (x + 2 * y) = 1 :=
by
  sorry

end cos_x_plus_2y_eq_one_l631_631823


namespace binom_18_4_eq_3060_l631_631789

theorem binom_18_4_eq_3060 : nat.choose 18 4 = 3060 := 
by 
  sorry

end binom_18_4_eq_3060_l631_631789


namespace coefficient_x2y4_expansion_l631_631946

theorem coefficient_x2y4_expansion : 
  let f := (fun x y : ℕ => x + y)^2 * (fun x y : ℕ => x - 2 * y)^4 in
  -- Coefficient of the term x^2 y^4 in the expansion of f(x, y)
  let coeff := 16 - 64 + 24 in
  coeff = -24 :=
  sorry

end coefficient_x2y4_expansion_l631_631946


namespace student_average_vs_true_average_l631_631365

theorem student_average_vs_true_average (w x y z : ℝ) (h : w < x ∧ x < y ∧ y < z) : 
  (2 * w + 2 * x + y + z) / 6 < (w + x + y + z) / 4 :=
by
  sorry

end student_average_vs_true_average_l631_631365


namespace count_numbers_with_seven_l631_631904

open Finset

def contains_digit_seven (n : ℕ) : Prop :=
  ∃ d : ℕ, d ∈ digits 10 n ∧ d = 7

theorem count_numbers_with_seven : 
  (card (filter (λ n, contains_digit_seven n) (range 801))) = 152 := 
by
  sorry

end count_numbers_with_seven_l631_631904


namespace sum_of_roots_eq_10_div_3_l631_631061

noncomputable def sumOfRoots : ℚ :=
  let p1 := (3 : ℚ) * X^3 + 2 * X^2 - 9 * X + 5
  let p2 := (4 : ℚ) * X^3 - 16 * X^2 + 7
  let S1 := - (2 / 3 : ℚ)
  let S2 := 4
  S1 + S2

theorem sum_of_roots_eq_10_div_3 : sumOfRoots = (10 / 3 : ℚ) :=
by
  sorry

end sum_of_roots_eq_10_div_3_l631_631061


namespace adil_older_than_bav_by_732_days_l631_631371

-- Definitions based on the problem conditions
def adilBirthDate : String := "December 31, 2015"
def bavBirthDate : String := "January 1, 2018"

-- Main theorem statement 
theorem adil_older_than_bav_by_732_days :
    let daysIn2016 := 366
    let daysIn2017 := 365
    let transition := 1
    let totalDays := daysIn2016 + daysIn2017 + transition
    totalDays = 732 :=
by
    sorry

end adil_older_than_bav_by_732_days_l631_631371


namespace positive_difference_l631_631698

def a := 8^2
def b := a + a
def c := a * a
theorem positive_difference : ((b / 8) - (c / 8)) = 496 := by
  sorry

end positive_difference_l631_631698


namespace positive_difference_of_fractions_l631_631692

theorem positive_difference_of_fractions : 
  (let a := 8^2 in (a + a) / 8) = 16 ∧ (let a := 8^2 in (a * a) / 8) = 512 →
  (let a := 8^2 in ((a * a) / 8 - (a + a) / 8)) = 496 := 
by
  sorry

end positive_difference_of_fractions_l631_631692


namespace possible_values_of_b_l631_631567

theorem possible_values_of_b 
  (m : ℕ) (b : ℕ) 
  (h1 : 6 % m = b % m) 
  (h2 : (b / m : ℕ) * m = b) 
  (h3 : 1 < m) : 
  b = 2 ∨ b = 3 ∨ b = 4 :=
begin
  sorry
end

end possible_values_of_b_l631_631567


namespace cube_volume_is_10_l631_631353

def volume_of_box (length width height : ℝ) : ℝ :=
  length * width * height

def volume_of_cube (v_box : ℝ) (num_cubes : ℝ) : ℝ :=
  v_box / num_cubes

theorem cube_volume_is_10 :
  let length := 8
  let width := 15
  let height := 5
  let num_cubes := 60
  volume_of_cube (volume_of_box length width height) num_cubes = 10 := 
  by
    sorry

end cube_volume_is_10_l631_631353


namespace find_m_l631_631860

variables (a b c : ℝ × ℝ) (m : ℝ)

def a := (4, 4)
def b := (5, m)
def c := (1, 3)
def perp (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0
def vec_diff2 (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - 2 * v.1, u.2 - 2 * v.2)

theorem find_m (h : perp (vec_diff2 a c) b) : m = 5 := by
  sorry

end find_m_l631_631860


namespace Archie_started_with_100_marbles_l631_631014

theorem Archie_started_with_100_marbles
  (M : ℕ) 
  (h1 : 0.60 * M + (0.50 * 0.40 * M) + 20 = M) 
  (h2 : 0.20 * M = 20) : 
  M = 100 :=
by
  sorry

end Archie_started_with_100_marbles_l631_631014


namespace cos_alpha_in_second_quadrant_l631_631102

theorem cos_alpha_in_second_quadrant (α : ℝ) (hα : π / 2 < α ∧ α < π) (h_tan : Real.tan α = -1 / 2) :
  Real.cos α = -2 * Real.sqrt 5 / 5 :=
by
  sorry

end cos_alpha_in_second_quadrant_l631_631102


namespace collinear_points_l631_631453

-- Define the cyclic quadrilateral and necessary points
variables (A B C D P E F G H L M N K Q R : Type*)

-- Define the conditions
def cyclic_quadrilateral (A B C D : Type*) := true
def intersect_diagonals (A C B D P : Type*) := true
def projections (P A B E F G H : Type*) := true
def midpoints (A B C D L M N K : Type*) := true
def intersection1 (E G M K Q : Type*) := true
def intersection2 (F H L N R : Type*) := true

-- Question to prove: P, Q, and R are collinear
theorem collinear_points (A B C D P E F G H L M N K Q R : Type*) 
  (h1 : cyclic_quadrilateral A B C D)
  (h2 : intersect_diagonals A C B D P)
  (h3 : projections P A B E F G H)
  (h4 : midpoints A B C D L M N K)
  (h5 : intersection1 E G M K Q)
  (h6 : intersection2 F H L N R) : 
  collinear P Q R := 
sorry

end collinear_points_l631_631453


namespace quadrilateral_proof_l631_631626

noncomputable def inscribed_quad_eq (A B C D K : Point) 
  (h_inscribed: inscribed_quad A B C D)
  (h_AB_AD: A.dist B = A.dist D)
  (h_angle_eq: ∠DAK = ∠ABD)
  (h_K_on_CD: on_line_segment C D K) : Prop :=
  A.dist K ^ 2 = K.dist D ^ 2 + B.dist C * K.dist D

theorem quadrilateral_proof (A B C D K : Point) 
  (h_inscribed: inscribed_quad A B C D)
  (h_AB_AD: A.dist B = A.dist D)
  (h_angle_eq: ∠DAK = ∠ABD)
  (h_K_on_CD: on_line_segment C D K) : inscribed_quad_eq A B C D K :=
sorry

end quadrilateral_proof_l631_631626


namespace sum_of_reciprocal_arithmetic_terms_l631_631828

theorem sum_of_reciprocal_arithmetic_terms 
  (a : ℕ → ℕ)
  (h_arith_seq : ∀ n m, a (n + 1) - a n = a (m + 1) - a m)
  (h_distinct : ∀ n, a n ≠ a (n + 1))
  (h_sum_5 : ∑ i in finset.range 5, a i = 20)
  (h_geo_seq : ∃ r : ℕ, a 3 = a 1 * r ∧ a 7 = a 1 * r ^ 2) :
  ∑ i in finset.range n, (1 : ℚ) / (a i * a (i + 1)) = n / (2 * (n + 2)) := 
sorry

end sum_of_reciprocal_arithmetic_terms_l631_631828


namespace wool_usage_l631_631802

def total_balls_of_wool_used (scarves_aaron sweaters_aaron sweaters_enid : ℕ) (wool_per_scarf wool_per_sweater : ℕ) : ℕ :=
  (scarves_aaron * wool_per_scarf) + (sweaters_aaron * wool_per_sweater) + (sweaters_enid * wool_per_sweater)

theorem wool_usage :
  total_balls_of_wool_used 10 5 8 3 4 = 82 :=
by
  -- calculations done in solution steps
  -- total_balls_of_wool_used (10 scarves * 3 balls/scarf) + (5 sweaters * 4 balls/sweater) + (8 sweaters * 4 balls/sweater)
  -- total_balls_of_wool_used (30) + (20) + (32)
  -- total_balls_of_wool_used = 30 + 20 + 32 = 82
  sorry

end wool_usage_l631_631802


namespace sufficient_not_necessary_condition_for_monotonicity_l631_631253

def f (k : ℝ) (x : ℝ) : ℝ := k * x^3 + (2/3) * x + 1

theorem sufficient_not_necessary_condition_for_monotonicity (k : ℝ) :
  (∀ x, 0 ≤ x → 0 ≤ (3 * k * x^2 + (2/3))) → (k > 0) :=
sorry

end sufficient_not_necessary_condition_for_monotonicity_l631_631253


namespace cos_alpha_after_rotation_is_correct_l631_631533

noncomputable def cos_alpha_after_rotation (A : ℝ × ℝ) (θ : ℝ) : ℝ :=
  let α := atan ((1 + tan(θ)) / (1 - tan(θ))) 
  in 1 / (sqrt (1 + (tan α)^2))

theorem cos_alpha_after_rotation_is_correct :
  cos_alpha_after_rotation (2, 1) (π / 4) = (sqrt 10) / 10 :=
  sorry

end cos_alpha_after_rotation_is_correct_l631_631533


namespace equivalent_form_l631_631508

theorem equivalent_form (x y : ℝ) (h : y = x + 1/x) :
  (x^4 + x^3 - 3*x^2 + x + 2 = 0) ↔ (x^2 * (y^2 + y - 5) = 0) :=
sorry

end equivalent_form_l631_631508


namespace correctCountForDivisibilityBy15_l631_631949

namespace Divisibility

noncomputable def countWaysToMakeDivisibleBy15 : Nat := 
  let digits := [0, 2, 4, 5, 7, 9]
  let baseSum := 2 + 0 + 1 + 6 + 0 + 2
  let validLastDigit := [0, 5]
  let totalCombinations := 6^4
  let ways := 2 * totalCombinations
  let adjustment := (validLastDigit.length * digits.length * digits.length * digits.length * validLastDigit.length) / 4 -- Correcting multiplier as per reference
  adjustment

theorem correctCountForDivisibilityBy15 : countWaysToMakeDivisibleBy15 = 864 := 
  by
    sorry

end Divisibility

end correctCountForDivisibilityBy15_l631_631949


namespace election_valid_votes_l631_631738

variable (V : ℕ)
variable (invalid_pct : ℝ)
variable (exceed_pct : ℝ)
variable (total_votes : ℕ)
variable (invalid_votes : ℝ)
variable (valid_votes : ℕ)
variable (A_votes : ℕ)
variable (B_votes : ℕ)

theorem election_valid_votes :
  V = 9720 →
  invalid_pct = 0.20 →
  exceed_pct = 0.15 →
  total_votes = V →
  invalid_votes = invalid_pct * V →
  valid_votes = total_votes - invalid_votes →
  A_votes = B_votes + exceed_pct * total_votes →
  A_votes + B_votes = valid_votes →
  B_votes = 3159 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end election_valid_votes_l631_631738


namespace solve_for_n_l631_631434

-- Define the problem statement
theorem solve_for_n : ∃ n : ℕ, (3 * n^2 + n = 219) ∧ (n = 9) := 
sorry

end solve_for_n_l631_631434


namespace lateral_surface_area_and_height_l631_631174

-- Straightforward conditions
variables {A B C V : Point}
variables {AB AC BC : ℝ}
variables {dihedral_angle : ℝ}

-- Definitions and problem statement
def pyramid (A B C V : Point) (AB AC BC : ℝ) (dihedral_angle : ℝ) :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ A ≠ V ∧ B ≠ V ∧ C ≠ V ∧
  AB = 10 ∧ AC = 10 ∧ BC = 12 ∧ dihedral_angle = 45

theorem lateral_surface_area_and_height (AB AC BC : ℝ) (dihedral_angle : ℝ) :
  pyramid A B C V AB AC BC dihedral_angle →
  lateral_surface_area V A B C = 172 ∧ height V A B C = 8 :=
by
  sorry

end lateral_surface_area_and_height_l631_631174


namespace correct_proposition_l631_631821

-- Definitions of lines and planes
variable (l m : Line)
variable (α β γ : Plane)

-- Conditions as per the problem statement
variable different_lines : l ≠ m
variable different_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ

-- Definitions for perpendicularity and parallelism
axiom perp_line_plane (l : Line) (α : Plane) : Prop
axiom parallel_line_plane (l : Line) (α : Plane) : Prop
axiom line_in_plane (m : Line) (α : Plane) : Prop
axiom perp_planes (α β : Plane) : Prop
axiom parallel_planes (α β : Plane) : Prop
axiom perp_lines (l m : Line) : Prop
axiom parallel_lines (l m : Line) : Prop

-- Propositions as per the problem statement
def prop_1 := perp_line_plane l α ∧ line_in_plane m α → perp_lines l m
def prop_2 := parallel_line_plane l α ∧ line_in_plane m α → parallel_lines l m
def prop_3 := perp_planes α β ∧ perp_planes α γ → parallel_planes β γ
def prop_4 := perp_planes α β ∧ perp_line_plane l β → parallel_line_plane l α

-- The goal: verify that only Proposition 1 is true
theorem correct_proposition : prop_1 ∧ ¬prop_2 ∧ ¬prop_3 ∧ ¬prop_4 :=
by
  sorry -- Proof placeholder

end correct_proposition_l631_631821


namespace parabola_equation_line_equation_through_focus_and_parabola_l631_631077

noncomputable def parabola (p : ℝ) : ℝ := p = 2

theorem parabola_equation : 
  ∃ p, p = 2 ∧ ∀ x y, y^2 = 2 * p * x ↔ y^2 = 4 * x :=
begin
  existsi 2,
  split,
  { refl },
  { intros x y,
    split,
    { intro h,
      exact h },
    { intro h,
      exact h } }
end

noncomputable def line_through_focus (x1 x2 : ℝ) (k : ℝ) : Prop :=
  x1 + x2 = 4 ∧ k = sqrt 2 ∨ k = -sqrt 2

theorem line_equation_through_focus_and_parabola :
  ∀ x1 y1 x2 y2 k,
  (y1^2 = 4 * x1 ∧ y2^2 = 4 * x2 ∧
  (x1 + x2) / 2 = 2 ∧ line_through_focus x1 x2 k) →
    (exists (k : ℝ), (y = k * x - sqrt 2) ∨ (y = -k * x + sqrt 2)) :=
begin
  intros x1 y1 x2 y2 k h,
  rcases h with ⟨h1, h2, h3, h4⟩,
  use k,
  cases h4,
  { cases h4_right,
    { left, sorry },
    { right, sorry } },
  { cases h4_right,
    { left, sorry },
    { right, sorry } }
end

end parabola_equation_line_equation_through_focus_and_parabola_l631_631077


namespace digit_in_101st_place_l631_631498

theorem digit_in_101st_place :
  let repeating_sequence := "269230769"
  in string.nth repeating_sequence ((101 % 9) - 1) = '6' :=
by
  sorry

end digit_in_101st_place_l631_631498


namespace buoy_min_force_l631_631971

-- Define the problem in Lean
variables (M : ℝ) (ax : ℝ) (T_star : ℝ) (a : ℝ) (F_current : ℝ)
-- Conditions
variables (h_horizontal_component : T_star * Real.sin a = F_current)
          (h_zero_net_force : M * ax = 0)

theorem buoy_min_force (h_horizontal_component : T_star * Real.sin a = F_current) : 
  F_current = 400 := 
sorry

end buoy_min_force_l631_631971


namespace perimeter_isosceles_triangle_l631_631833

theorem perimeter_isosceles_triangle : 
  ∀ (x y : ℝ), |x - 4| + real.sqrt (y - 8) = 0 → 
  (∃ P : ℝ, (x = 4) ∧ (y = 8) ∧ P = 20) :=
by {
  intros x y h,
  -- introduce necessary assumptions and prove that perimeter is 20
  sorry
}

end perimeter_isosceles_triangle_l631_631833


namespace odd_function_eval_l631_631143

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then real.log (2 - x) / real.log 2 else sorry

theorem odd_function_eval
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_def : ∀ x : ℝ, x < 0 → f x = real.log (2 - x) / real.log 2) :
  f 0 + f 2 = -2 := by
sorry

end odd_function_eval_l631_631143


namespace club_truncator_wins_more_than_losses_l631_631030

theorem club_truncator_wins_more_than_losses :
  let p_win := 1/3
  let p_lose := 1/3
  let p_tie := 1/3
  let total_games := 6
  (probability_more_wins_than_losses total_games p_win p_lose p_tie) = 98/243 :=
by
  sorry

end club_truncator_wins_more_than_losses_l631_631030


namespace correct_equation_l631_631651

variables (x : ℝ) (production_planned total_clothings : ℝ)
variables (increase_rate days_ahead : ℝ)

noncomputable def daily_production (x : ℝ) := x
noncomputable def total_production := 1000
noncomputable def production_per_day_due_to_overtime (x : ℝ) := x * (1 + 0.2 : ℝ)
noncomputable def original_completion_days (x : ℝ) := total_production / daily_production x
noncomputable def increased_production_completion_days (x : ℝ) := total_production / production_per_day_due_to_overtime x
noncomputable def days_difference := original_completion_days x - increased_production_completion_days x

theorem correct_equation : days_difference x = 2 := by
  sorry

end correct_equation_l631_631651


namespace square_center_distance_l631_631171

theorem square_center_distance
  (ABCD: Type)
  (is_square : is_square ABCD)
  (E : Point)
  (is_center : is_center E ABCD)
  (P : Point)
  (on_semi_circle_AB : on_semi_circle P (diameter AB))
  (Q : Point)
  (on_semi_circle_AD : on_semi_circle Q (diameter AD))
  (collinear_QAP : collinear Q A P)
  (QA : distance Q A = 14)
  (AP : distance A P = 46)
  (AE : distance A E = x) :
  x = 34 :=
sorry

end square_center_distance_l631_631171


namespace number_of_tables_l631_631417

-- Define the conditions
def seats_per_table : ℕ := 8
def total_seating_capacity : ℕ := 32

-- Define the main statement using the conditions
theorem number_of_tables : total_seating_capacity / seats_per_table = 4 := by
  sorry

end number_of_tables_l631_631417


namespace positive_difference_is_496_l631_631714

def square (n: ℕ) : ℕ := n * n
def term1 := (square 8 + square 8) / 8
def term2 := (square 8 * square 8) / 8
def positive_difference := abs (term2 - term1)

theorem positive_difference_is_496 : positive_difference = 496 :=
by
  -- This is where the proof would go
  sorry

end positive_difference_is_496_l631_631714


namespace range_of_k_n_as_function_of_m_l631_631474

-- Definitions based on conditions
def circle (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 4
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x
def on_segment (M N Q : ℝ × ℝ) : Prop :=
  let (xM, yM) := M
  let (xN, yN) := N
  let (xQ, yQ) := Q
  (xQ - xM) * (yN - yM) = (yQ - yM) * (xN - xM) -- Collinear condition for lying on segment

-- First problem: range of k
theorem range_of_k {k : ℝ} : 
  (∃ x y : ℝ, circle x y ∧ line k x y) ↔ k ∈ (-∞ : ℝ, -sqrt 3) ∪ (sqrt 3, +∞ : ℝ) := sorry

-- Second problem: n as a function of m
theorem n_as_function_of_m {m n k : ℝ}
  (h_k : ∃ x M_y : ℝ, circle x M_y ∧ line k x M_y)
  (h_Q : ∃ M N Q : ℝ × ℝ, circle (prod.fst M) (prod.snd M) ∧ circle (prod.fst N) (prod.snd N) ∧ line k (prod.fst Q) (prod.snd Q) ∧ on_segment M N Q)
  (h_dist : (∃ Q : ℝ × ℝ, (2 / (prod.fst Q^2 + prod.snd Q^2) = (1 / (prod.fst M^2 + prod.snd M^2)) + (1 / (prod.fst N^2 + prod.snd N^2))))
  : 0 < n ∧ n = sqrt (15 * m^2 + 180) / 5 := sorry

end range_of_k_n_as_function_of_m_l631_631474


namespace find_complex_z_l631_631517

-- Definition to represent the condition
def complex_func (z : ℂ) : Prop := (z / (1 - complex.i)) = (complex.i ^ 2016 + complex.i ^ 2017)

-- The main statement to prove the value of z
theorem find_complex_z (z : ℂ) : complex_func z → z = 2 :=
begin
  sorry
end

end find_complex_z_l631_631517


namespace angle_measure_l631_631009

theorem angle_measure (A B C : ℝ) (h1 : A = B) (h2 : A + B = 110 ∨ (A = 180 - 110)) :
  A = 70 ∨ A = 55 := by
  sorry

end angle_measure_l631_631009


namespace factory_production_schedule_l631_631650

noncomputable def production_equation (x : ℝ) : Prop :=
  (1000 / x) - (1000 / (1.2 * x)) = 2

theorem factory_production_schedule (x : ℝ) (hx : x ≠ 0) : production_equation x := 
by 
  -- Assumptions based on conditions:
  -- Factory plans to produce total of 1000 sets of protective clothing.
  -- Actual production is 20% more than planned.
  -- Task completed 2 days ahead of original schedule.
  -- We need to show: (1000 / x) - (1000 / (1.2 * x)) = 2
  sorry

end factory_production_schedule_l631_631650


namespace aquarium_total_cost_is_63_l631_631017

/-- Axel bought an aquarium that was marked down 50% from an original price of $120.
    He also paid additional sales tax equal to 5% of the reduced price.
    What was the total cost of the aquarium? -/
def calculate_total_cost (original_price : ℝ) (markdown_percentage : ℝ) (sales_tax_percentage : ℝ) : ℝ :=
  let markdown := (markdown_percentage / 100) * original_price
  let reduced_price := original_price - markdown
  let sales_tax := (sales_tax_percentage / 100) * reduced_price
  reduced_price + sales_tax

theorem aquarium_total_cost_is_63 :
  calculate_total_cost 120 50 5 = 63 := 
by 
  sorry

end aquarium_total_cost_is_63_l631_631017


namespace hotel_problem_l631_631247

-- Define the conditions given in the problem
def number_of_persons (n : ℕ) := n = 9

axiom condition1 : ∀ (n : ℕ), n = 8 * 12
axiom condition2 : ∀ (A : ℝ), ∃ (n : ℕ), n = A + 8
axiom condition3 : ∀ (exp_tot : ℝ), exp_tot = 117

-- The proof statement: how many persons went to the hotel?
theorem hotel_problem : 
  (∃ (n : ℕ), number_of_persons n) → 
  (∀ (A : ℝ) (exp_tot : ℝ), (condition1 96) ∧ (condition2 (A + 8)) ∧ condition3 exp_tot → 
  number_of_persons 9) :=
by {
  sorry
}

end hotel_problem_l631_631247


namespace numbers_with_7_in_1_to_800_l631_631908

theorem numbers_with_7_in_1_to_800 : 
  (card { n ∈ finset.range (800 + 1) | ∃ d ∈ n.digits 10, d = 7 }) = 152 := 
sorry

end numbers_with_7_in_1_to_800_l631_631908


namespace dr_jones_remaining_money_l631_631046

def monthly_earning : ℕ := 6000
def house_rental : ℕ := 640
def food_expense : ℕ := 380
def electric_water_bill : ℕ := monthly_earning * (1 / 4)
def insurance_cost : ℕ := monthly_earning * (1 / 5)

theorem dr_jones_remaining_money :
  let total_bills := house_rental + food_expense + electric_water_bill + insurance_cost in
  monthly_earning - total_bills = 2280 :=
by
  sorry

end dr_jones_remaining_money_l631_631046


namespace focus_distance_l631_631255

theorem focus_distance (x: ℝ) (y: ℝ) : 
  (y^2 = x) → 
  let focus := (1/4 : ℝ, 0 : ℝ) in
  let line := (4 : ℝ, 1 : ℝ) in
  let num := |4 * (1/4) + 0 + 1| in
  let denom := real.sqrt (1^2 + 4^2) in
  let question := num / denom in
  question = (2 * real.sqrt 17) / 17 := 
sorry

end focus_distance_l631_631255


namespace number_of_numbers_with_digit_7_from_1_to_800_eq_233_l631_631891

def contains_digit (n d : ℕ) : Prop :=
  ∃ k, 10 ^ k > 0 ∧ d = (n / 10 ^ k) % 10

def numbers_without_digit (n d : ℕ) : finset ℕ :=
  (finset.range n).filter (λ x, ¬ contains_digit x d)

def count_numbers_with_digit (n d : ℕ) : ℕ :=
  n - (numbers_without_digit n d).card

theorem number_of_numbers_with_digit_7_from_1_to_800_eq_233 :
  count_numbers_with_digit 800 7 = 233 :=
  sorry

end number_of_numbers_with_digit_7_from_1_to_800_eq_233_l631_631891


namespace area_of_shape_formed_by_z_l631_631452

-- Define the condition of the complex number z
def condition (z : ℂ) := abs z ≤ 2

-- State that the area of the shape formed by the set of points satisfying the condition is 4π
theorem area_of_shape_formed_by_z (z : ℂ) (h : condition z) : 
  (set_of (condition)).volume = 4 * real.pi := 
sorry

end area_of_shape_formed_by_z_l631_631452


namespace number_of_numbers_with_digit_7_from_1_to_800_eq_233_l631_631890

def contains_digit (n d : ℕ) : Prop :=
  ∃ k, 10 ^ k > 0 ∧ d = (n / 10 ^ k) % 10

def numbers_without_digit (n d : ℕ) : finset ℕ :=
  (finset.range n).filter (λ x, ¬ contains_digit x d)

def count_numbers_with_digit (n d : ℕ) : ℕ :=
  n - (numbers_without_digit n d).card

theorem number_of_numbers_with_digit_7_from_1_to_800_eq_233 :
  count_numbers_with_digit 800 7 = 233 :=
  sorry

end number_of_numbers_with_digit_7_from_1_to_800_eq_233_l631_631890


namespace no_subset_sum_eq_100_implies_sum_ne_200_l631_631822

theorem no_subset_sum_eq_100_implies_sum_ne_200
  (a : Fin 100 → ℕ)
  (h1 : ∀ i j, i ≤ j → a i ≤ a j)
  (h2 : ∀ i, a i < 100)
  (h3 : ¬ ∃ s : Finset (Fin 100), (∑ i in s, a i) = 100) :
  (∑ i, a i) ≠ 200 :=
by
  sorry

end no_subset_sum_eq_100_implies_sum_ne_200_l631_631822


namespace spencer_total_distance_l631_631972

-- Definitions for the given conditions
def distance_house_to_library : ℝ := 0.3
def distance_library_to_post_office : ℝ := 0.1
def distance_post_office_to_home : ℝ := 0.4

-- Define the total distance based on the given conditions
def total_distance : ℝ := distance_house_to_library + distance_library_to_post_office + distance_post_office_to_home

-- Statement to prove
theorem spencer_total_distance : total_distance = 0.8 := by
  sorry

end spencer_total_distance_l631_631972


namespace positive_difference_l631_631685

theorem positive_difference : 496 = abs ((64 + 64) / 8 - (64 * 64) / 8) := by
  have h1 : 8^2 = 64 := rfl
  have h2 : 64 + 64 = 128 := rfl
  have h3 : (128 : ℕ) / 8 = 16 := rfl
  have h4 : 64 * 64 = 4096 := rfl
  have h5 : (4096 : ℕ) / 8 = 512 := rfl
  have h6 : 512 - 16 = 496 := rfl
  sorry

end positive_difference_l631_631685


namespace find_age_of_b_l631_631250

variables (A B C : ℕ)

def average_abc (A B C : ℕ) : Prop := (A + B + C) / 3 = 28
def average_ac (A C : ℕ) : Prop := (A + C) / 2 = 29

theorem find_age_of_b (h1 : average_abc A B C) (h2 : average_ac A C) : B = 26 :=
by
  sorry

end find_age_of_b_l631_631250


namespace find_CE_l631_631192

-- Definitions
variables (Γ : Type) [metric_space Γ] [normed_group Γ] [normed_space ℝ Γ]
variables (O A B C D E : Γ) 
variables (circle : metric.sphere O (dist O A))
variables (tangent_A : affine.tangent_line circle A)
variables (tangent_B : affine.tangent_line circle B)
variables (secant_CE : segment C E)

-- Given conditions
variables (AB : line_segment A B)
variables (intersects_AB_at_D_and_circle_at_E : line_intersects_segment_and_circle C AB circle D E)
variables (lies_on_CE : lies_on_segment D C E)
variables (angle_condition : ∠ B O D + ∠ E A D = 180)
variables (AE_eq_1 : dist A E = 1)
variables (BE_eq_2 : dist B E = 2)

-- The proof goal
theorem find_CE :
  dist C E = (4 * real.sqrt 2) / 3 :=
sorry

end find_CE_l631_631192


namespace perimeter_of_square_l631_631437

theorem perimeter_of_square (s : ℕ) (h : s = 13) : 4 * s = 52 :=
by {
  sorry
}

end perimeter_of_square_l631_631437


namespace find_n_l631_631987

open_locale classical

noncomputable def vector_a (n : ℝ) : ℝ × ℝ := ⟨n, 1⟩
noncomputable def vector_b : ℝ × ℝ := ⟨2, 1⟩

noncomputable def vector_length_squared (v : ℝ × ℝ) : ℝ :=
  v.1 ^ 2 + v.2 ^ 2

noncomputable def vector_dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem find_n (n : ℝ) (h : vector_length_squared (vector_a n - vector_b) = vector_length_squared (vector_a n) + vector_length_squared vector_b) :
  n = -1 / 2 :=
by
  sorry

end find_n_l631_631987


namespace original_cone_volume_l631_631349

theorem original_cone_volume
  (H R h r : ℝ)
  (Vcylinder : ℝ) (Vfrustum : ℝ)
  (cylinder_volume : Vcylinder = π * r^2 * h)
  (frustum_volume : Vfrustum = (1 / 3) * π * (R^2 + R * r + r^2) * (H - h))
  (Vcylinder_value : Vcylinder = 9)
  (Vfrustum_value : Vfrustum = 63) :
  (1 / 3) * π * R^2 * H = 64 :=
by
  sorry

end original_cone_volume_l631_631349


namespace injective_function_identity_l631_631806

theorem injective_function_identity (f : ℕ → ℕ) (h_inj : Function.Injective f)
  (h : ∀ (m n : ℕ), 0 < m → 0 < n → f (n * f m) ≤ n * m) : ∀ x : ℕ, f x = x :=
by
  sorry

end injective_function_identity_l631_631806


namespace minimize_expression_l631_631724

noncomputable def f (x y : ℝ) (a b c : ℝ) : ℝ := 
  |a * (x + y - 10 * b) * (3 * x - 6 * y - 36 * c) * (19 * x + 95 * y - 95 * a)|

theorem minimize_expression : 
  (∀ a b c : ℝ, 0 < a ∧ a ≤ 10 ∧ 0 < b ∧ b ≤ 10 ∧ 0 < c ∧ c ≤ 10 → 
    ∑ (i : ℝ × ℝ × ℝ) in set.univ.filter (λ i, 0 < i.1 ∧ i.1 ≤ 10 ∧ 0 < i.2.1 ∧ i.2.1 ≤ 10 ∧ 0 < i.2.2 ∧ i.2.2 ≤ 10)
                                           (λ (i : ℝ × ℝ × ℝ), f 55 (-4) i.1 i.2.1 i.2.2) = 2394) ∧ 
    (∃ x y : ℝ, x = 55 ∧ y = -4)
:= sorry

end minimize_expression_l631_631724


namespace temperature_representation_BelowFifteen_correct_Zero_correct_AboveTen_correct_l631_631018

theorem temperature_representation :
  (BelowFifteen = -15) ∧ (Zero = 0) ∧ (AboveTen = 10) :=
by
  sorry

-- Definitions for the conditions
def BelowFifteen : Int := -15
def Zero : Int := 0
def AboveTen : Int := 10

theorem BelowFifteen_correct : BelowFifteen = -15 := by rfl
theorem Zero_correct : Zero = 0 := by rfl
theorem AboveTen_correct : AboveTen = 10 := by rfl

end temperature_representation_BelowFifteen_correct_Zero_correct_AboveTen_correct_l631_631018


namespace number_of_seasons_l631_631761

theorem number_of_seasons 
        (episodes_per_season : ℕ) 
        (fraction_watched : ℚ) 
        (remaining_episodes : ℕ) 
        (h_episodes_per_season : episodes_per_season = 20) 
        (h_fraction_watched : fraction_watched = 1 / 3) 
        (h_remaining_episodes : remaining_episodes = 160) : 
        ∃ (seasons : ℕ), seasons = 12 :=
by
  sorry

end number_of_seasons_l631_631761


namespace total_receipts_calculation_l631_631372

theorem total_receipts_calculation :
  let adults := 280
  let adult_ticket_price := 25
  let children := 120
  let children_ticket_price := 15
  adults * adult_ticket_price + children * children_ticket_price = 8800 := by
    let adults := 280
    let adult_ticket_price := 25
    let children := 120
    let children_ticket_price := 15
    have h1 : adults * adult_ticket_price = 7000 := by
      sorry
    have h2 : children * children_ticket_price = 1800 := by
      sorry
    calc
      adults * adult_ticket_price + children * children_ticket_price
          = 7000 + 1800 : by rw [h1, h2]
      ... = 8800       : by norm_num

end total_receipts_calculation_l631_631372


namespace positive_difference_l631_631683

theorem positive_difference : 496 = abs ((64 + 64) / 8 - (64 * 64) / 8) := by
  have h1 : 8^2 = 64 := rfl
  have h2 : 64 + 64 = 128 := rfl
  have h3 : (128 : ℕ) / 8 = 16 := rfl
  have h4 : 64 * 64 = 4096 := rfl
  have h5 : (4096 : ℕ) / 8 = 512 := rfl
  have h6 : 512 - 16 = 496 := rfl
  sorry

end positive_difference_l631_631683


namespace books_bought_l631_631657

theorem books_bought (math_price : ℕ) (hist_price : ℕ) (total_cost : ℕ) (math_books : ℕ) (hist_books : ℕ) 
  (H : math_price = 4) (H1 : hist_price = 5) (H2 : total_cost = 396) (H3 : math_books = 54) 
  (H4 : math_books * math_price + hist_books * hist_price = total_cost) :
  math_books + hist_books = 90 :=
by sorry

end books_bought_l631_631657


namespace greatest_integer_solution_l631_631660

theorem greatest_integer_solution (n : ℤ) (h : n^2 - 13 * n + 40 ≤ 0) : n ≤ 8 :=
sorry

end greatest_integer_solution_l631_631660


namespace total_balls_of_wool_l631_631803

theorem total_balls_of_wool (a_scarves a_sweaters e_sweaters : ℕ)
  (wool_per_scarf wool_per_sweater : ℕ)
  (a_scarves = 10) (a_sweaters = 5) (e_sweaters = 8)
  (wool_per_scarf = 3) (wool_per_sweater = 4) :
  a_scarves * wool_per_scarf + a_sweaters * wool_per_sweater + e_sweaters * wool_per_sweater = 82 :=
by
  sorry

end total_balls_of_wool_l631_631803


namespace pure_imaginary_implies_a_is_negative_two_l631_631518

theorem pure_imaginary_implies_a_is_negative_two (a : ℝ) 
  (h1 : a^2 + a - 2 = 0)
  (h2 : a^2 - 3a + 2 ≠ 0) : 
  a = -2 :=
by sorry

end pure_imaginary_implies_a_is_negative_two_l631_631518


namespace difference_divisible_l631_631961

theorem difference_divisible (a b n : ℕ) (h : n % 2 = 0) (hab : a + b = 61) :
  (47^100 - 14^100) % 61 = 0 := by
  sorry

end difference_divisible_l631_631961


namespace intersection_M_N_l631_631859

-- Defining set M
def M : Set ℕ := {1, 2, 3, 4}

-- Defining the set N based on the condition
def N : Set ℕ := {x | ∃ n ∈ M, x = n^2}

-- Lean statement to prove the intersection
theorem intersection_M_N : M ∩ N = {1, 4} := 
by
  sorry

end intersection_M_N_l631_631859


namespace percentage_increase_is_30_l631_631155

-- Define the variables x, y, and z as real numbers
variables (x y z : ℝ)

-- Define the conditions
def condition1 : Prop := y = 0.5 * z
def condition2 : Prop := x = 0.65 * z

-- Define the goal
def percentage_increase : Prop :=
  ((x - y) / y) * 100 = 30

-- The theorem statement incorporating the conditions and the goal
theorem percentage_increase_is_30 (h1 : condition1) (h2 : condition2) : percentage_increase :=
by
  -- Ensures the theorem compiles successfully, proof is omitted
  sorry

end percentage_increase_is_30_l631_631155


namespace bisection_interval_l631_631205

def f (x : ℝ) : ℝ := 3^x + 3 * x - 8

theorem bisection_interval
  (h1 : f 1 < 0)
  (h3 : f 3 > 0)
  (h2 : f 2 > 0) :
  ∃ (a b : ℝ), (a = 1 ∧ b = 2) :=
by
  sorry

end bisection_interval_l631_631205


namespace complex_quads_area_l631_631281

theorem complex_quads_area {z : ℂ} (hz : (z * (conj z)^3 + (conj z) * z^3 = 272) ∧ (z.re ∈ ℤ) ∧ (z.im ∈ ℤ)) : let points := {w : ℂ | w = 5 + 3 * I ∨ w = 5 - 3 * I ∨ w = -5 + 3 * I ∨ w = -5 - 3 * I} in 
  (points.card = 4) → 
  let a := 10 
  let b := 6 in
  2 * (a/2 * b/2) = 60 :=
by
  sorry

end complex_quads_area_l631_631281


namespace smallest_k_for_perfect_square_l631_631302

theorem smallest_k_for_perfect_square (x : ℕ) (hx : x = 1575) : ∃ k : ℕ, k > 0 ∧ (k * x) = 3 ^ 2 * 5 ^ 2 * 7 ^ 2 ∧ ∀ k' < k, k' * 1575 ≠ 3 ^ 2 * 5 ^ 2 * 7 ^ 2 :=
by
  use 7
  split
  { exact Nat.one_lt_succ_succ Nat.succ_pos' } -- 7 > 0
  split
  { sorry } -- Proof that 7 * 1575 = 3^2 * 5^2 * 7^2
  { sorry } -- Proof that 7 is the smallest such k

end smallest_k_for_perfect_square_l631_631302


namespace chess_games_possible_l631_631194

theorem chess_games_possible (n : ℕ) (t : fin n → ℕ) 
  (h1 : 1 ≤ n) 
  (h2 : ∀ i j, i < j → t i < t j)
  (h3 : ∀ i, 1 ≤ t i) :
  ∃ g : fin (t n.pred + 1) → fin (t n.pred + 1) → Prop,
  (∀ i j, g i j → g j i ∧ i ≠ j) ∧ 
  (∀ i, ∃ j, (fin.card (fin.filter (λ k, g i k) _)) = t j) :=
by
  sorry

end chess_games_possible_l631_631194


namespace number_of_numbers_with_digit_7_from_1_to_800_eq_233_l631_631893

def contains_digit (n d : ℕ) : Prop :=
  ∃ k, 10 ^ k > 0 ∧ d = (n / 10 ^ k) % 10

def numbers_without_digit (n d : ℕ) : finset ℕ :=
  (finset.range n).filter (λ x, ¬ contains_digit x d)

def count_numbers_with_digit (n d : ℕ) : ℕ :=
  n - (numbers_without_digit n d).card

theorem number_of_numbers_with_digit_7_from_1_to_800_eq_233 :
  count_numbers_with_digit 800 7 = 233 :=
  sorry

end number_of_numbers_with_digit_7_from_1_to_800_eq_233_l631_631893


namespace problem_statement_l631_631218

-- Definitions based on the given conditions
variable (A B C D M H N K: Type)
variable [Triangle A B C] -- assuming a definition for Triangle
variable (P1 : OnSide D AC) -- point D on AC
variable (P2 : Median AM) -- AM is a median
variable (P3 : Altitude CH) -- CH is an altitude
variable (P4 : Intersection N AM CH) -- AM intersects CH at N
variable (P5 : Intersection K AM BD) -- AM intersects BD at K
variable (P6 : AK = BK) -- AK equals BK

-- Theorem statement
theorem problem_statement : 
    AN = 2 * KM := 
sorry

end problem_statement_l631_631218


namespace tangent_line_at_1_l631_631481

noncomputable def f (x : ℝ) : ℝ := Real.log x - 3 * x

noncomputable def f' (x : ℝ) : ℝ := 1 / x - 3

theorem tangent_line_at_1 :
  let y := f 1
  let k := f' 1
  y = -3 ∧ k = -2 →
  ∀ (x y : ℝ), y = k * (x - 1) + f 1 ↔ 2 * x + y + 1 = 0 :=
by
  sorry

end tangent_line_at_1_l631_631481


namespace initial_fee_calculation_l631_631549

theorem initial_fee_calculation 
  (charge_per_segment : ℝ)
  (segment_length : ℝ)
  (total_distance : ℝ)
  (total_charge : ℝ)
  (number_of_segments := total_distance / segment_length)
  (cost_for_distance := number_of_segments * charge_per_segment)
  (initial_fee := total_charge - cost_for_distance) :
  charge_per_segment = 0.35 → 
  segment_length = 2 / 5 → 
  total_distance = 3.6 → 
  total_charge = 5.5 → 
  initial_fee = 2.35 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp only [div_eq_mul_inv, mul_comm (3.6:ℝ), mul_assoc, mul_inv_cancel_left₀ (ne_of_gt (by norm_num : (2:ℝ) ≠ 0)), mul_comm 2, ←mul_assoc, mul_comm 0.35, sub_self_add]
  norm_num

end initial_fee_calculation_l631_631549


namespace triangle_problem_l631_631456

/-- 
Given a triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively, 
if b = 2 and 2*b*cos B = a*cos C + c*cos A,
prove that B = π/3 and find the maximum area of ΔABC.
-/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) (h1 : b = 2) (h2 : 2 * b * Real.cos B = a * Real.cos C + c * Real.cos A) :
  B = Real.pi / 3 ∧
  (∃ (max_area : ℝ), max_area = Real.sqrt 3 ∧ max_area = (1/2) * a * c * Real.sin B) :=
by
  sorry

end triangle_problem_l631_631456


namespace highest_power_12_divides_20_factorial_highest_power_10_divides_20_factorial_l631_631390

-- Define the conditions for the factorial
def factorial(n : Nat) : Nat :=
  if n = 0 then 1
  else n * factorial (n - 1)

-- Define a helper function to compute the highest power of a prime p in n!
def highest_power_in_factorial (n p : Nat) : Nat :=
  Nat.sum (List.range (n+1)).map (λ k => k / p^Nat.logb (Nat.log2 k + 1) p).sum

-- Lean 4 statement for the first problem
theorem highest_power_12_divides_20_factorial :
  highest_power_in_factorial 20 2 / 2 ≤ highest_power_in_factorial 20 3 → 6 :=
by
  sorry

-- Lean 4 statement for the second problem
theorem highest_power_10_divides_20_factorial :
  (highest_power_in_factorial 20 2 ≤ highest_power_in_factorial 20 5) → 4 :=
by
  sorry

end highest_power_12_divides_20_factorial_highest_power_10_divides_20_factorial_l631_631390


namespace find_length_of_DE_l631_631590

noncomputable def rectangle_side_ab : ℝ := 6
noncomputable def rectangle_side_ad : ℝ := 8
noncomputable def area_rectangle_abcd : ℝ := rectangle_side_ab * rectangle_side_ad

noncomputable def isosceles_right_triangle_dc_eq_de {DE : ℝ} : Prop :=
  area_rectangle_abcd = 2 * (1 / 2 * DE * DE)

theorem find_length_of_DE (DE : ℝ) (h1 : rectangle_side_ab = 6) (h2 : rectangle_side_ad = 8) 
(h3 : area_rectangle_abcd = 48) (h4 : isosceles_right_triangle_dc_eq_de) : 
  DE = 4 * Real.sqrt 3 := 
sorry

end find_length_of_DE_l631_631590


namespace fraction_equality_l631_631918

theorem fraction_equality (x y z : ℝ) (k : ℝ) (hx : x = 3 * k) (hy : y = 5 * k) (hz : z = 7 * k) :
  (x - y + z) / (x + y - z) = 5 := 
  sorry

end fraction_equality_l631_631918


namespace value_of_a_range_of_m_l631_631479

def f (x a : ℝ) : ℝ := abs (x - a)

-- Given the following conditions
axiom cond1 (x : ℝ) (a : ℝ) : f x a = abs (x - a)
axiom cond2 (x : ℝ) (a : ℝ) : (f x a >= 3) ↔ (x <= 1 ∨ x >= 5)

-- Prove that a = 2
theorem value_of_a (a : ℝ) : (∀ x : ℝ, (f x a >= 3) ↔ (x <= 1 ∨ x >= 5)) → a = 2 := by
  sorry

-- Additional condition for m
axiom cond3 (x : ℝ) (a : ℝ) (m : ℝ) : ∀ x : ℝ, f x a + f (x + 4) a >= m

-- Prove that m ≤ 4
theorem range_of_m (a : ℝ) (m : ℝ) : (∀ x : ℝ, f x a + f (x + 4) a >= m) → a = 2 → m ≤ 4 := by
  sorry

end value_of_a_range_of_m_l631_631479


namespace factor_exp_l631_631425

variable (x : ℤ)

theorem factor_exp : x * (x + 2) + (x + 2) = (x + 1) * (x + 2) :=
by
  sorry

end factor_exp_l631_631425


namespace incenter_circumcenter_midpoints_concyclic_l631_631176

noncomputable def incenter (A B C : Point) : Point := sorry
noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def midpoint (P Q : Point) : Point := sorry
def Point := ℂ

variables (A B C : Point)
variables (h : 2 * dist A B = dist B C + dist C A)

theorem incenter_circumcenter_midpoints_concyclic 
  (I := incenter A B C)
  (O := circumcenter A B C)
  (D := midpoint B C)
  (E := midpoint A C) 
  : ∃ K : Circle, I ∈ K ∧ O ∈ K ∧ D ∈ K ∧ E ∈ K :=
sorry

end incenter_circumcenter_midpoints_concyclic_l631_631176


namespace tangents_from_point_l631_631181

theorem tangents_from_point (P : ℝ → ℝ) (a b : ℝ) (n : ℕ) (h_P_deg : nat_degree P = n) :
  ¬ ∃ (x : set ℝ), ∀ k ∈ x, (tangent_through_point P a b k) ∧ (card x > n) :=
by {
   have h_P_der := (P.deriv),
   let Q := λ x, P x - h_P_der x * (x - a) - b,
   have h_Q_deg : nat_degree Q = n := sorry,
   have h_Q : ∀ x, Q x = 0 ↔ (P x - h_P_der x * (x - a) = b) := sorry,
   have h_Roots := fun x => set_of (λ k, Q k = 0),
   assume h_contra : ∃ x, ∀ k ∈ x, (Q k = 0) ∧ (card x > n),
   obtain ⟨ x, h_x_cond, h_card⟩ := h_contra,
   have h_num_roots := card x,
   have h_contradiction : Q can't have n+1 roots since deg Q = n := sorry,
   contradiction,
}

end tangents_from_point_l631_631181


namespace percentage_of_z_equals_39_percent_of_y_l631_631510

theorem percentage_of_z_equals_39_percent_of_y
    (x y z : ℝ)
    (h1 : y = 0.75 * x)
    (h2 : z = 0.65 * x)
    (P : ℝ)
    (h3 : (P / 100) * z = 0.39 * y) :
    P = 45 :=
by sorry

end percentage_of_z_equals_39_percent_of_y_l631_631510


namespace suitable_sampling_is_stratified_l631_631530

def AgeGroup := { x : ℕ // 15 ≤ x ∧ x ≤ 75 }

inductive SamplingMethod
| SimpleRandomSampling
| SystematicSampling
| StratifiedSampling

def DifferentFamiliarity : Prop := ∀ age_group : AgeGroup, ∃ familiarity : ℕ, True

def AvailableSamplingMethods : Set SamplingMethod := 
  { SamplingMethod.SimpleRandomSampling, SamplingMethod.SystematicSampling, SamplingMethod.StratifiedSampling }

def SuitableSamplingMethod : SamplingMethod :=
  if DifferentFamiliarity then SamplingMethod.StratifiedSampling else SamplingMethod.SimpleRandomSampling

theorem suitable_sampling_is_stratified :
  SuitableSamplingMethod = SamplingMethod.StratifiedSampling :=
sorry

end suitable_sampling_is_stratified_l631_631530


namespace sum_inequality_l631_631103

theorem sum_inequality (n : ℕ) (a : ℕ → ℝ) 
  (h_nonneg : ∀ i, 0 ≤ a i) 
  (h_min : ∀ i, a i ≥ a (n+1) ∧ a (n+1) = a 0)
  (h_a : a (n+1) = a 0) :
  (∑ i in range n, (1 + a i) / (1 + a (i+1))) ≤ 
    n + (1 / (1 + a (0))^2) * (∑ i in range n, (a (i) - a (0))^2) :=
sorry

end sum_inequality_l631_631103


namespace Hadley_walked_to_grocery_store_in_2_miles_l631_631862

-- Define the variables and conditions
def distance_to_grocery_store (x : ℕ) : Prop :=
  x + (x - 1) + 3 = 6

-- Stating the main proposition to prove
theorem Hadley_walked_to_grocery_store_in_2_miles : ∃ x : ℕ, distance_to_grocery_store x ∧ x = 2 := 
by sorry

end Hadley_walked_to_grocery_store_in_2_miles_l631_631862


namespace solve_eq1_solve_eq2_l631_631593

-- Proof for the first equation
theorem solve_eq1 (y : ℝ) : 8 * y - 4 * (3 * y + 2) = 6 ↔ y = -7 / 2 := 
by 
  sorry

-- Proof for the second equation
theorem solve_eq2 (x : ℝ) : 2 - (x + 2) / 3 = x - (x - 1) / 6 ↔ x = 1 := 
by 
  sorry

end solve_eq1_solve_eq2_l631_631593


namespace rationalize_denominator_l631_631226

theorem rationalize_denominator :
  let A := -12
  let B := 7
  let C := 9
  let D := 13
  let E := 5 in
  B < D ∧
  (12/5 * Real.sqrt 7) = ((1:ℚ) * Real.sqrt B / E * A) * (-1) ∧
  (9/5 * Real.sqrt 13) = (1:ℚ * Real.sqrt D / E * C) ∧
  (A + B + C + D + E = 22) :=
by
  sorry

end rationalize_denominator_l631_631226


namespace triangle_DEF_area_l631_631537

theorem triangle_DEF_area (area_square : ℕ)
  (side_square : ℕ)
  (side_small_squares : ℕ)
  (triangle_isosceles : ∀ (D E F : Type), DE = DF)
  (D_F_coincides_with_center : ∀ (T : Type), D = T)
  : ∃ (area_triangle : ℚ), area_square = 36 ∧ side_square = 6 ∧
    side_small_squares = 2 ∧ area_triangle = 7 := by
  use 7
  split
  sorry

end triangle_DEF_area_l631_631537


namespace multiple_for_incorrect_responses_l631_631772

-- Define the conditions
def number_of_questions : ℕ := 100
def number_of_correct_responses : ℕ := 87
def student_score : ℤ := 61
def number_of_incorrect_responses : ℕ := number_of_questions - number_of_correct_responses

-- Statement: The multiple used for the number of incorrect responses is 2
theorem multiple_for_incorrect_responses :
  ∃ m : ℤ, student_score = number_of_correct_responses - m * number_of_incorrect_responses ∧ m = 2 :=
by
  use 2
  have h1 : number_of_incorrect_responses = 13 := by
    unfold number_of_incorrect_responses
    norm_num
  rw h1
  norm_num
  split; refl
  sorry

end multiple_for_incorrect_responses_l631_631772


namespace females_employed_percentage_l631_631540

theorem females_employed_percentage
  (total_employment : ℕ → Prop)
  (employed_males : ℕ → Prop)
  (female_employment_distribution : ℕ → (ℕ → Prop) → Prop)
  (perc_total_employment : ℝ)
  (perc_employed_males : ℝ)
  (perc_males_tech : ℝ)
  (perc_males_health : ℝ)
  (perc_females_education : ℝ) :
  perc_total_employment = 0.64 →
  perc_employed_males = 0.55 →
  perc_males_tech = 0.30 →
  perc_males_health = 0.40 →
  perc_females_education = 0.60 →
  let perc_employed_females := perc_total_employment - perc_employed_males in
  let perc_females_education_employed := perc_employed_females * perc_females_education in
  let perc_females_health_employed := perc_employed_females * (1 - perc_females_education) in
  let perc_employed_females_total_pop := (perc_employed_females / perc_total_employment) * 100 in
  perc_employed_females_total_pop = 14.0625 := sorry

end females_employed_percentage_l631_631540


namespace books_more_than_figures_l631_631548

-- Definitions of initial conditions
def initial_action_figures := 2
def initial_books := 10
def added_action_figures := 4

-- Problem statement to prove
theorem books_more_than_figures :
  initial_books - (initial_action_figures + added_action_figures) = 4 :=
by
  -- Proof goes here
  sorry

end books_more_than_figures_l631_631548


namespace team_total_points_l631_631929

theorem team_total_points : 
  ∀ (Tobee Jay Sean : ℕ),
  (Tobee = 4) →
  (Jay = Tobee + 6) →
  (Sean = Tobee + Jay - 2) →
  (Tobee + Jay + Sean = 26) :=
by
  intros Tobee Jay Sean h1 h2 h3
  rw [h1, h2, h3]
  sorry

end team_total_points_l631_631929


namespace volunteer_distribution_l631_631044

theorem volunteer_distribution (volunteers : Fin 4 → Type) (communities : Fin 3 → Type) :
  (∀ v, ∃ c, volunteers v = some c) → {d : Σ (f : Fin 4 → Fin 3), ∀ c, ∃ v, f v = c}.card = 30 :=
by
  sorry

end volunteer_distribution_l631_631044


namespace distance_lines_l631_631612

def distance_between_parallel_lines 
  (A B C1 C2 : ℝ) 
  (hA : A = 1) 
  (hB : B = 2) 
  (hC1 : C1 = -5) 
  (hC2 : C2 = 0) : ℝ :=
  |C2 - C1| / Real.sqrt (A^2 + B^2)

theorem distance_lines 
  (A B C1 C2 : ℝ) 
  (hA : A = 1) 
  (hB : B = 2) 
  (hC1 : C1 = -5) 
  (hC2 : C2 = 0) : 
  distance_between_parallel_lines A B C1 C2 hA hB hC1 hC2 = Real.sqrt 5 :=
by
  sorry

end distance_lines_l631_631612


namespace complex_conjugate_product_l631_631074

theorem complex_conjugate_product (z : ℂ) (h : z = 2 + 3 * Complex.I) : z * (Complex.conj z) = 13 := by
  sorry

end complex_conjugate_product_l631_631074


namespace digits_in_s_999_l631_631412

def a : ℕ → ℝ := -- Define a(m) here
def b : ℕ → ℝ := -- Define b(p) here

-- Assuming a and b are increasing sequences
axiom a_increasing : ∀ i j, i < j → a i < a j
axiom b_increasing : ∀ i j, i < j → b i < b j

def s (n : ℕ) : list ℕ := (list.range n).map (λ i, a i ^ nat.floor (b i))

def num_digits (x : ℕ) : ℕ := nat.floor (real.log10 (x)) + 1

def total_digits (n : ℕ) : ℕ :=
(list.range n).sum (λ i, num_digits (a i ^ nat.floor (b i)))

theorem digits_in_s_999 :
  total_digits 999 = ∑ i in finset.range 999, nat.floor (b i * real.log10 (a i)) + 1 :=
sorry

end digits_in_s_999_l631_631412


namespace fixed_point_l631_631798

noncomputable def log_shifted (a x : ℝ) : ℝ := log a (x - 1) + 2

theorem fixed_point (a : ℝ) (x : ℝ) (y : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (x = 2) → (y = 2) → log_shifted a x = y :=
by
  intros
  sorry

end fixed_point_l631_631798


namespace grass_field_width_l631_631361

/--
A rectangular grass field has a length of 75 meters and a path of 2.5 meters wide around it on the outside.
The cost of constructing the path is Rs. 6750 at Rs. 10 per square meter.
Prove that the width of the grass field is 55 meters.
-/
theorem grass_field_width :
  ∃ (w : ℝ), (let length := 75
                  path_width := 2.5
                  total_cost := 6750
                  cost_per_sqm := 10
                  outer_length := length + 2 * path_width
                  outer_width := w + 2 * path_width
                  path_area := outer_length * outer_width - length * w
                  path_area = total_cost / cost_per_sqm
              in w = 55) :=
sorry

end grass_field_width_l631_631361


namespace palindromic_poly_has_root_neg_one_divide_by_x_plus_one_l631_631591

-- Define a palindromic polynomial of an odd degree
def palindromic_poly (a b : ℝ) (n : ℕ) : ℝ[X] :=
  polynomial.monomial (2 * n + 1) a + polynomial.monomial (2 * n) b + 
  polynomial.monomial 0 a + polynomial.monomial 1 b + 
  finset.sum (finset.range (n-1)) (λ i, polynomial.monomial (2 * n - 2 * (i + 1)) b + polynomial.monomial (2 * (i + 1)) b)

-- Prove that every odd-degree palindromic polynomial has a root at x = -1
theorem palindromic_poly_has_root_neg_one (a b : ℝ) (n : ℕ) :
  (palindromic_poly a b n).eval (-1) = 0 := by
  sorry

-- Prove that dividing a palindromic polynomial by (x + 1) results in another palindromic polynomial of degree one less
theorem divide_by_x_plus_one (a b : ℝ) (n : ℕ) :
  let P := palindromic_poly a b n in
  let Q := (polynomial.div_by_x_add_one P) in
  Q.mirror = Q := by
  sorry

end palindromic_poly_has_root_neg_one_divide_by_x_plus_one_l631_631591


namespace range_of_f_l631_631263

-- Given definitions
def f (x : ℝ) : ℝ := cos x ^ 2 + sin x - 1

-- Theorem statement about the range of the function 
theorem range_of_f : set.range f = set.Icc (-2 : ℝ) (1 / 4 : ℝ) := sorry

end range_of_f_l631_631263


namespace expression_value_l631_631505

theorem expression_value
  (x y : ℝ) 
  (h : x - 3 * y = 4) : 
  (x - 3 * y)^2 + 2 * x - 6 * y - 10 = 14 :=
by
  sorry

end expression_value_l631_631505


namespace positive_difference_l631_631684

theorem positive_difference : 496 = abs ((64 + 64) / 8 - (64 * 64) / 8) := by
  have h1 : 8^2 = 64 := rfl
  have h2 : 64 + 64 = 128 := rfl
  have h3 : (128 : ℕ) / 8 = 16 := rfl
  have h4 : 64 * 64 = 4096 := rfl
  have h5 : (4096 : ℕ) / 8 = 512 := rfl
  have h6 : 512 - 16 = 496 := rfl
  sorry

end positive_difference_l631_631684


namespace prime_divisibility_l631_631459

open Nat

theorem prime_divisibility (p : Nat) (hp : Nat.Prime p) :
  (p * (p^2 * (p^(p-1) - 1) / (p-1))!) % (∏ i in finset.range (p+1).succ, (p^i)!) = 0 :=
by
  sorry

end prime_divisibility_l631_631459


namespace total_balls_of_wool_l631_631804

theorem total_balls_of_wool (a_scarves a_sweaters e_sweaters : ℕ)
  (wool_per_scarf wool_per_sweater : ℕ)
  (a_scarves = 10) (a_sweaters = 5) (e_sweaters = 8)
  (wool_per_scarf = 3) (wool_per_sweater = 4) :
  a_scarves * wool_per_scarf + a_sweaters * wool_per_sweater + e_sweaters * wool_per_sweater = 82 :=
by
  sorry

end total_balls_of_wool_l631_631804


namespace probability_xi_gte_11_l631_631520

variable (ξ : ℝ → ℝ) (σ : ℝ)

noncomputable def normal_distribution_10_σ := 
  ∀ x : ℝ, ξ x = real.gaussian 10 σ x

variable (h1 : ∀ x : ℝ, real.integration (real.gaussian 10 σ) {x | 9 ≤ x ∧ x ≤ 11} = 0.4)

theorem probability_xi_gte_11 : 
  ∀ x : ℝ, real.integration (real.gaussian 10 σ) {x | 11 ≤ x} = 0.3 := 
by
  sorry

end probability_xi_gte_11_l631_631520


namespace value_of_d_l631_631514

theorem value_of_d (d : ℝ) (h : ∀ x : ℝ, 3 * (5 + d * x) = 15 * x + 15 → True) : d = 5 :=
sorry

end value_of_d_l631_631514


namespace cos_Z_l631_631954

-- Define the triangle XYZ with X being a right angle and the given sin Y.
structure TriangleXYZ where
  X Y Z : ℝ
  angle_X : X = 90
  sin_Y : sin Y = 3 / 5

-- State the theorem that proves cos Z given the properties of the triangle.
theorem cos_Z (T : TriangleXYZ) : cos T.Z = 3 / 5 := 
  sorry

end cos_Z_l631_631954


namespace find_f_2018_l631_631575

-- Define the function and conditions
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x) = f (-x)

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 3) = -1 / (f x)

def piecewise_definition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ∈ set.Icc (-3) (-2) → f x = 4 * x

noncomputable def f (x : ℝ) : ℝ := sorry

-- Main statement to prove
theorem find_f_2018 :
  (even_function f) →
  (functional_equation f) →
  (piecewise_definition f) →
  f 2018 = -8 :=
by
  sorry

end find_f_2018_l631_631575


namespace tetrahedron_median_face_area_l631_631324

variable {S : Fin 4 → ℝ} -- Areas of the faces
variable {angle : Fin 4 → Fin 4 → ℝ} -- Angles between planes (only valid if k < j)

-- Define the face area S_kj for median faces
noncomputable def S_kj (k j : Fin 4) : ℝ := 
  if h : k < j then 
    let cosΘ := angle k j in
    (1 / 2) * S k * S j * cosΘ 
  else 0

-- The main conjecture to be proved:
theorem tetrahedron_median_face_area 
  (k j : Fin 4) (hk : k < j) :
  S_kj k j ^ 2 = 
  (1 / 4) * (S k ^ 2 + S j ^ 2 + 2 * S k * S j * Real.cos (angle k j)) := 
sorry

end tetrahedron_median_face_area_l631_631324


namespace ernie_can_make_circles_l631_631381

theorem ernie_can_make_circles :
  ∀ (boxes_initial Ali_circles Ali_boxes_per_circle Ernie_boxes_per_circle : ℕ),
  Ali_circles = 5 →
  Ali_boxes_per_circle = 8 →
  Ernie_boxes_per_circle = 10 →
  boxes_initial = 80 →
  ((boxes_initial - Ali_circles * Ali_boxes_per_circle) / Ernie_boxes_per_circle) = 4 :=
by
  intros boxes_initial Ali_circles Ali_boxes_per_circle Ernie_boxes_per_circle 
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end ernie_can_make_circles_l631_631381


namespace rationalize_denominator_to_find_constants_l631_631236

-- Definitions of the given conditions
def original_fraction := 3 / (4 * Real.sqrt 7 + 3 * Real.sqrt 13)
def simplified_fraction (A B C D E : ℤ) := (A * Real.sqrt B + C * Real.sqrt D) / E

-- Statement of the proof problem
theorem rationalize_denominator_to_find_constants :
  ∃ (A B C D E : ℤ),
    original_fraction = simplified_fraction A B C D E ∧
    B < D ∧
    (∀ p : ℕ, Real.sqrt (p * p) = p) ∧ -- Ensuring that all radicals are in simplest form
    A + B + C + D + E = 22 :=
sorry

end rationalize_denominator_to_find_constants_l631_631236


namespace rationalize_denominator_l631_631232

theorem rationalize_denominator :
  let A := -12
  let B := 7
  let C := 9
  let D := 13
  let E := 5
  (4 * Real.sqrt 7 + 3 * Real.sqrt 13) ≠ 0 →
  B < D →
  ∀ (x : ℝ), x = (3 : ℝ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) →
    A + B + C + D + E = 22 := 
by
  intros
  -- Provide the actual theorem statement here
  sorry

end rationalize_denominator_l631_631232


namespace min_value_ineq_l631_631204

theorem min_value_ineq (b : Fin 8 → ℝ) (hpos : ∀ i, 0 < b i)
    (hsum : (∑ i, b i) = 2) : (∑ i, 1 / (b i)) ≥ 32 := 
sorry

end min_value_ineq_l631_631204


namespace ratio_of_areas_l631_631254

theorem ratio_of_areas (x y l : ℝ)
  (h1 : 2 * (x + 3 * y) = 2 * (l + y))
  (h2 : 2 * x + l = 3 * y) :
  (x * 3 * y) / (l * y) = 3 / 7 :=
by
  -- Proof will be provided here
  sorry

end ratio_of_areas_l631_631254


namespace solution_set_inequality_range_of_m_l631_631854

def f (x : ℝ) : ℝ := |2 * x + 1| + 2 * |x - 3|

theorem solution_set_inequality :
  ∀ x : ℝ, f x ≤ 7 * x ↔ x ≥ 1 :=
by sorry

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, f x = |m|) ↔ (m ≥ 7 ∨ m ≤ -7) :=
by sorry

end solution_set_inequality_range_of_m_l631_631854


namespace gray_opposite_black_l631_631418

/-
  Eight squares are colored, front and back:
  A = Aqua, B = Black, C = Crimson, D = Dark Blue,
  E = Emerald, F = Fuchsia, G = Gray, H = Hazel.
  They are hinged together, then folded to form a cube.
  It's given that the aqua face (A) is adjacent to the dark blue (D) and emerald (E) faces.

  The question: Which face is opposite the gray (G) face?
  The answer: The black (B) face.
-/

-- Define the colors
inductive Color
| Aqua
| Black
| Crimson
| DarkBlue
| Emerald
| Fuchsia
| Gray
| Hazel

open Color

-- Define adjacency condition as given in the problem
def adjacent (c1 c2 : Color) : Prop :=
  (c1, c2) = (Aqua, DarkBlue) ∨ (c1, c2) = (Aqua, Emerald) ∨
  (c1, c2) = (DarkBlue, Aqua) ∨ (c1, c2) = (Emerald, Aqua)

-- Define the main theorem
theorem gray_opposite_black (h_adj : adjacent Aqua DarkBlue)
                           (h_adj2 : adjacent Aqua Emerald)
                           : (∃ c : Color, c = Black ∧ is_opposite Gray c) := sorry

end gray_opposite_black_l631_631418


namespace minimum_run_distance_l631_631936

-- Definitions of the points and their positions
structure Point :=
  (x : ℝ) (y : ℝ)

def A : Point := ⟨0, 400⟩
def B : Point := ⟨150, 1000⟩ -- Since B is 600 meters above the wall and 400 + 600 = 1000

-- Function to calculate the distance between two points
def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

-- The mathematical proof to be stated
theorem minimum_run_distance : 
  distance A B ≈ 1010.623 (by library_search)

end minimum_run_distance_l631_631936


namespace positive_difference_l631_631666

theorem positive_difference :
  let a := 8^2
  let term1 := (a + a) / 8
  let term2 := (a * a) / 8
  term2 - term1 = 496 :=
by
  let a := 8^2
  let term1 := (a + a) / 8
  let term2 := (a * a) / 8
  have h1 : a = 64 := rfl
  have h2 : term1 = 16 := by simp [a, term1]
  have h3 : term2 = 512 := by simp [a, term2]
  show 512 - 16 = 496 from sorry

end positive_difference_l631_631666


namespace banana_pies_l631_631215

theorem banana_pies (total_pies : ℕ) (ratio_p : ℕ) (ratio_m : ℕ) (ratio_b : ℕ) (h_total_pies : total_pies = 30)
    (h_ratio_p : ratio_p = 2) (h_ratio_m : ratio_m = 5) (h_ratio_b : ratio_b = 3) :
    let parts := ratio_p + ratio_m + ratio_b in
    let pies_per_part := total_pies / parts in
    let banana_pies := ratio_b * pies_per_part in
    banana_pies = 9 :=
by
  sorry

end banana_pies_l631_631215


namespace distance_between_points_l631_631296

def point1 : ℝ × ℝ := (1, 3)
def point2 : ℝ × ℝ := (-5, 7)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_points :
  distance point1 point2 = 2 * real.sqrt 13 :=
by
  sorry

end distance_between_points_l631_631296


namespace angle_FYD_l631_631166

variables (AB CD EF GH : line)
variables (A X F Y D : point)
variables [parallel AB CD]
variables [parallel EF GH]

theorem angle_FYD : measure ∠FYD = 50 :=
by
  have h1 : measure ∠AXF = 130 := sorry
  have h2 : measure ∠AXF + measure ∠FYD = 180 := sorry
  simp only [h1] at *
  linarith

end angle_FYD_l631_631166


namespace edward_money_l631_631049

theorem edward_money (initial_amount spent1 spent2 : ℕ) (h_initial : initial_amount = 34) (h_spent1 : spent1 = 9) (h_spent2 : spent2 = 8) :
  initial_amount - (spent1 + spent2) = 17 :=
by
  sorry

end edward_money_l631_631049


namespace correct_operation_l631_631729

theorem correct_operation (a b : ℝ) : 
  (a+2)*(a-2) = a^2 - 4 :=
by
  sorry

end correct_operation_l631_631729


namespace three_dice_prime_probability_l631_631146

noncomputable def rolling_three_dice_prime_probability : ℚ :=
  sorry

theorem three_dice_prime_probability : rolling_three_dice_prime_probability = 1 / 24 :=
  sorry

end three_dice_prime_probability_l631_631146


namespace positive_difference_of_fractions_l631_631697

theorem positive_difference_of_fractions : 
  (let a := 8^2 in (a + a) / 8) = 16 ∧ (let a := 8^2 in (a * a) / 8) = 512 →
  (let a := 8^2 in ((a * a) / 8 - (a + a) / 8)) = 496 := 
by
  sorry

end positive_difference_of_fractions_l631_631697


namespace quadratic_completeness_l631_631992

noncomputable def quad_eqn : Prop :=
  ∃ b c : ℤ, (∀ x : ℝ, (x^2 - 10 * x + 15 = 0) ↔ ((x + b)^2 = c)) ∧ b + c = 5

theorem quadratic_completeness : quad_eqn :=
sorry

end quadratic_completeness_l631_631992


namespace ernie_can_make_circles_l631_631379

theorem ernie_can_make_circles :
  ∀ (boxes_initial Ali_circles Ali_boxes_per_circle Ernie_boxes_per_circle : ℕ),
  Ali_circles = 5 →
  Ali_boxes_per_circle = 8 →
  Ernie_boxes_per_circle = 10 →
  boxes_initial = 80 →
  ((boxes_initial - Ali_circles * Ali_boxes_per_circle) / Ernie_boxes_per_circle) = 4 :=
by
  intros boxes_initial Ali_circles Ali_boxes_per_circle Ernie_boxes_per_circle 
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end ernie_can_make_circles_l631_631379


namespace find_integer_less_than_M_div_100_l631_631097

-- The problem and proof constants
theorem find_integer_less_than_M_div_100 :
  let M := 4992 in
  let result := ⌊M / 100⌋ in
  result = 49 :=
by
  -- The conditions given and the resultant M is defined.
  have h1 : 1 / (3! * 18!) + 1 / (4! * 17!) + 1 / (5! * 16!) + 1 / (6! * 15!) + 1 / (7! * 14!) + 
            1 / (8! * 13!) + 1 / (9! * 12!) + 1 / (10! * 11!) = M / (2! * 19!) := sorry,
  -- Hence, final result.
  have h2 : M = 4992 := sorry,
  have h3 : result = ⌊4992 / 100⌋ := by simp [M, result, int.floor_eq_iff, ←div_lt_iff, int.cast_49],
  exact h3

end find_integer_less_than_M_div_100_l631_631097


namespace greatest_possible_mean_BC_l631_631743

-- Mean weights for piles A, B
def mean_weight_A : ℝ := 60
def mean_weight_B : ℝ := 70

-- Combined mean weight for piles A and B
def mean_weight_AB : ℝ := 64

-- Combined mean weight for piles A and C
def mean_weight_AC : ℝ := 66

-- Prove that the greatest possible integer value for the mean weight of
-- the rocks in the combined piles B and C
theorem greatest_possible_mean_BC : ∃ (w : ℝ), (⌊w⌋ = 75) :=
by
  -- Definitions and assumptions based on problem conditions
  have h1 : mean_weight_A = 60 := rfl
  have h2 : mean_weight_B = 70 := rfl
  have h3 : mean_weight_AB = 64 := rfl
  have h4 : mean_weight_AC = 66 := rfl
  sorry

end greatest_possible_mean_BC_l631_631743


namespace max_value_part_a_max_value_part_b_l631_631998

-- Statement for Part (a)
theorem max_value_part_a (a b : ℝ) (h₁ : a + b = 1) (h₂ : 0 < a) (h₃ : 0 < b) :
  (√(2 * a + 1) + √(2 * b + 1) ≤ 2 * √2) :=
sorry

-- Statement for Part (b)
theorem max_value_part_b (x y z : ℝ) (h₁ : x + y + z = 3) (h₂ : 0 < x) (h₃ : 0 < y) (h₄ : 0 < z) :
  (√(2 * x + 1) + √(2 * y + 1) + √(2 * z + 1) ≤ 3 * √3) :=
sorry

end max_value_part_a_max_value_part_b_l631_631998


namespace petya_lost_remaining_games_l631_631525

theorem petya_lost_remaining_games :
  ∃ (participants : Finset ℕ) (games_played : ℕ) (points : Fin ℕ → ℚ),
  (participants.card = 12) ∧
  (games_played = (Finset.card participants) * (Finset.card participants - 1) / 2) ∧
  (∀ x ∈ participants, points x ≤ 4 ∨ x = 11) ∧
  (points 11 = 9) ∧
  (∀ y ∈ participants, y ≠ 11 →
    ∃ (p1 p2 : Fin 12) (points_exp: ℚ),
    (points p1 ≤ 4 ∧ points_exp ≤ 4) ∧
    (p2 ≠ p1) ∧
    (p2 ≠ 11) ∧
    (points p2 + points_exp ≤ 4)) :=
by {
  -- Definitions of participants, games_played, and points based on provided conditions
  sorry
}

end petya_lost_remaining_games_l631_631525


namespace sequence_tenth_term_l631_631950

theorem sequence_tenth_term :
  ∃ (a : ℕ → ℚ), a 1 = 1 ∧ (∀ n : ℕ, n > 0 → a (n + 1) = a n / (1 + a n)) ∧ a 10 = 1 / 10 :=
sorry

end sequence_tenth_term_l631_631950


namespace observations_count_l631_631283

theorem observations_count (n : ℕ) (h1 : (n : ℝ) ≠ 0)
    (h2 : mean_initial : ℝ := 36)
    (h3 : sum_initial : ℝ := n * mean_initial)
    (h4 : corrected_sum : ℝ := sum_initial + 21)
    (h5 : mean_corrected : ℝ := 36.5)
    (h6 : corrected_sum = n * mean_corrected) :
    n = 42 :=
begin
  sorry
end

end observations_count_l631_631283


namespace positive_difference_of_fractions_l631_631694

theorem positive_difference_of_fractions : 
  (let a := 8^2 in (a + a) / 8) = 16 ∧ (let a := 8^2 in (a * a) / 8) = 512 →
  (let a := 8^2 in ((a * a) / 8 - (a + a) / 8)) = 496 := 
by
  sorry

end positive_difference_of_fractions_l631_631694


namespace max_product_h_k_l631_631595

theorem max_product_h_k {h k : ℝ → ℝ} (h_bound : ∀ x, -3 ≤ h x ∧ h x ≤ 5) (k_bound : ∀ x, -1 ≤ k x ∧ k x ≤ 4) :
  ∃ x y, h x * k y = 20 :=
by
  sorry

end max_product_h_k_l631_631595


namespace total_amount_invested_l631_631926

def annualIncome (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * rate

def totalInvestment (T x y : ℝ) : Prop :=
  T - x = y

def condition (T : ℝ) : Prop :=
  let income_10_percent := annualIncome (T - 800) 0.10
  let income_8_percent := annualIncome 800 0.08
  income_10_percent - income_8_percent = 56

theorem total_amount_invested :
  ∃ (T : ℝ), condition T ∧ totalInvestment T 800 800 ∧ T = 2000 :=
by
  sorry

end total_amount_invested_l631_631926


namespace percentage_of_students_wearing_blue_shirts_l631_631161

theorem percentage_of_students_wearing_blue_shirts :
  ∀ (total_students red_percentage green_percentage other_students : ℕ),
    total_students = 700 →
    red_percentage = 23 →
    green_percentage = 15 →
    other_students = 119 →
    (100 - (red_percentage + green_percentage + ((other_students * 100) / total_students))) = 45 :=
by
  intros total_students red_percentage green_percentage other_students
  assume h1 h2 h3 h4
  sorry

end percentage_of_students_wearing_blue_shirts_l631_631161


namespace find_a_plus_b_l631_631466

-- Definitions of the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := log (10^x + 1) + a * x
def g (b : ℝ) (x : ℝ) : ℝ := (4^x - b) / 2^x

-- Conditions for f being an even function and g being an odd function 
def even_function (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = h x
def odd_function (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

-- The problem statement
theorem find_a_plus_b {a b : ℝ}
  (hf : even_function (f a))
  (hg : odd_function (g b)) :
  a + b = 1 / 2 := 
  sorry

end find_a_plus_b_l631_631466


namespace positive_difference_of_fractions_l631_631696

theorem positive_difference_of_fractions : 
  (let a := 8^2 in (a + a) / 8) = 16 ∧ (let a := 8^2 in (a * a) / 8) = 512 →
  (let a := 8^2 in ((a * a) / 8 - (a + a) / 8)) = 496 := 
by
  sorry

end positive_difference_of_fractions_l631_631696


namespace inv_eq_self_l631_631189

noncomputable def g (m x : ℝ) : ℝ := (3 * x + 4) / (m * x - 3)

theorem inv_eq_self (m : ℝ) :
  (∀ x : ℝ, g m x = g m (g m x)) ↔ m ∈ Set.Iic (-9 / 4) ∪ Set.Ici (-9 / 4) :=
by
  sorry

end inv_eq_self_l631_631189


namespace inverse_f_neg_3_l631_631149

def f (x : ℝ) : ℝ := 5 - 2 * x

theorem inverse_f_neg_3 : (∃ x : ℝ, f x = -3) ∧ (f 4 = -3) :=
by
  sorry

end inverse_f_neg_3_l631_631149


namespace complex_problem_l631_631502

open Complex

theorem complex_problem
  (α θ β : ℝ)
  (h : exp (i * (α + θ)) + exp (i * (β + θ)) = 1 / 3 + (4 / 9) * i) :
  exp (-i * (α + θ)) + exp (-i * (β + θ)) = 1 / 3 - (4 / 9) * i :=
by
  sorry

end complex_problem_l631_631502


namespace number_contains_digit_7_l631_631869

noncomputable def contains_digit (d n : ℕ) : Prop :=
  ∃ k, n / 10^k % 10 = d

noncomputable def count_numbers_with_digit (d bound : ℕ) : ℕ :=
  (finset.range (bound + 1)).filter (λ n, contains_digit d n).card

theorem number_contains_digit_7 : count_numbers_with_digit 7 800 = 152 := 
sorry

end number_contains_digit_7_l631_631869


namespace circumcircle_through_midpoint_l631_631535

noncomputable theory -- due to the use of classical geometry, which might involve classical logic

open EuclideanGeometry -- this opens needed geometric constructs and definitions

/-- 
Given an acute-angled triangle ABC with altitudes intersecting opposite sides at points D, E, F.
Consider a line through D that is parallel to EF and intersects AC and AB at points Q and R, respectively.
Let P be the point where EF intersects BC. Prove that the circumcircle of triangle PQR passes through the midpoint M of BC.
-/
theorem circumcircle_through_midpoint 
  (ABC : Triangle)
  (h_acute : acute_triangle ABC)
  (D E F : Point)
  (h1 : foot D A B C)
  (h2 : foot E B C A)
  (h3 : foot F C A B)
  (Q R : Point)
  (h4 : parallel_line D EF Q R)
  (P : Point)
  (h5 : intersect_line EF BC P)
  (M : Point)
  (h6 : midpoint M B C) :
  is_on_circumcircle P Q R M := 
sorry

end circumcircle_through_midpoint_l631_631535


namespace smallest_value_l631_631503

theorem smallest_value (x : ℝ) (h : 0 < x ∧ x < 1) : 
  min x (min (x^2) (min (x^3) (min (sqrt x) (min (2*x) (1/x)))))) = x^3 := sorry

end smallest_value_l631_631503


namespace truck_travel_distance_l631_631776

/-- Theorem: The truck travels 22.5b / t yards in 6 minutes, given its speed reduces by half halfway -/
theorem truck_travel_distance (b t : ℝ) (hb : b > 0) (ht : t > 0) :
  ((\lfloor (360/2) * (b / 4 / t) \rfloor) + (\lfloor (360/2) * (b / 8 / t) \rfloor)) / 3 = 22.5 * b / t :=
by
  sorry

end truck_travel_distance_l631_631776


namespace dr_jones_remaining_money_l631_631045

def monthly_earning : ℕ := 6000
def house_rental : ℕ := 640
def food_expense : ℕ := 380
def electric_water_bill : ℕ := monthly_earning * (1 / 4)
def insurance_cost : ℕ := monthly_earning * (1 / 5)

theorem dr_jones_remaining_money :
  let total_bills := house_rental + food_expense + electric_water_bill + insurance_cost in
  monthly_earning - total_bills = 2280 :=
by
  sorry

end dr_jones_remaining_money_l631_631045


namespace range_of_a_l631_631458

variable {x a : ℝ}

def p := x^2 - 2 * x - 3 < 0
def q := x > a

theorem range_of_a (h : ∀ x, p → q) (h' : ∃ x, ¬(q → p)) : a ≤ -1 :=
sorry

end range_of_a_l631_631458


namespace geometric_mean_in_triangle_l631_631958

open Classical

theorem geometric_mean_in_triangle (A B C D : Type*) [field A] [add_comm_group B] [module A B] 
  (ABC : triangle A B) (D_on_BC : D ∈ line_segment B C) :
  (distance A D) ^ 2 = (distance B D) * (distance C D) :=
by
  sorry

end geometric_mean_in_triangle_l631_631958


namespace area_of_triangle_ABD_l631_631168

-- Definitions stating the conditions of the problem
variables (DC BD : ℝ)
hypothesis h1 : DC / BD = 4 / 3
hypothesis h2 : ∃ (ADC_area : ℝ), ADC_area = 64 

-- Definition of the area of triangle ABD
noncomputable def ABD_area := (3/4) * 64

-- The statement to prove
theorem area_of_triangle_ABD (DC BD : ℝ) (h1 : DC / BD = 4 / 3) (h2 : ∃ (ADC_area : ℝ), ADC_area = 64) : ABD_area = 48 :=
by
  sorry

end area_of_triangle_ABD_l631_631168


namespace probability_x_plus_y_lt_5_l631_631763

theorem probability_x_plus_y_lt_5 :
  let square := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 4 }
  ∫ (p : ℝ × ℝ) in square, (if p.1 + p.2 < 5 then 1 else 0) ∂(measure_theory.measure.lebesgue) = 29 / 32 := 
sorry

end probability_x_plus_y_lt_5_l631_631763


namespace bob_wins_l631_631382

open Classical

-- Definition of the finite set S given conditions
def is_chosen_set (S : set (ℤ × ℤ)) (m n : ℕ) : Prop :=
  ∀ p : ℤ × ℤ, p ∈ S ↔ m ≤ p.1 ^ 2 + p.2 ^ 2 ∧ p.1 ^ 2 + p.2 ^ 2 ≤ n

-- Definition of count of points on specific lines
def number_of_points_on_line (S : set (ℤ × ℤ)) (ℓ : ℤ × ℤ → Prop) : ℕ :=
  S.countp ℓ

-- Bob's wins condition (he can determine the set uniquely)
def bob_can_win (info : (ℤ × ℤ → Prop) → ℕ) (S : set (ℤ × ℤ)) : Prop :=
  ∃ T : set (ℤ × ℤ), (∀ ℓ : ℤ × ℤ → Prop, number_of_points_on_line T ℓ = info ℓ) → T = S

-- Main theorem to be proved
theorem bob_wins {m n : ℕ} (S : set (ℤ × ℤ)) (info : (ℤ × ℤ → Prop) → ℕ) :
  is_chosen_set S m n →
  bob_can_win info S :=
sorry

end bob_wins_l631_631382


namespace cube_split_odd_numbers_l631_631815

theorem cube_split_odd_numbers (m : ℕ) (h1 : 1 < m) (h2 : ∃ k, (31 = 2 * k + 1 ∧ (m - 1) * m / 2 = k)) : m = 6 := 
by
  sorry

end cube_split_odd_numbers_l631_631815


namespace distance_between_homes_l631_631583

theorem distance_between_homes (Maxwell_distance : ℝ) (Maxwell_speed : ℝ) (Brad_speed : ℝ) (midpoint : ℝ) 
    (h1 : Maxwell_speed = 2) 
    (h2 : Brad_speed = 4) 
    (h3 : Maxwell_distance = 12) 
    (h4 : midpoint = Maxwell_distance * 2 * (Brad_speed / Maxwell_speed) + Maxwell_distance) :
midpoint = 36 :=
by
  sorry

end distance_between_homes_l631_631583


namespace angle_cosine_third_quadrant_l631_631511

theorem angle_cosine_third_quadrant (B : ℝ) (h1 : π < B ∧ B < 3 * π / 2) (h2 : Real.sin B = 4 / 5) :
  Real.cos B = -3 / 5 :=
sorry

end angle_cosine_third_quadrant_l631_631511


namespace positive_difference_l631_631686

theorem positive_difference : 496 = abs ((64 + 64) / 8 - (64 * 64) / 8) := by
  have h1 : 8^2 = 64 := rfl
  have h2 : 64 + 64 = 128 := rfl
  have h3 : (128 : ℕ) / 8 = 16 := rfl
  have h4 : 64 * 64 = 4096 := rfl
  have h5 : (4096 : ℕ) / 8 = 512 := rfl
  have h6 : 512 - 16 = 496 := rfl
  sorry

end positive_difference_l631_631686


namespace population_of_Beacon_l631_631609

-- Defining the populations of Richmond, Victoria, and Beacon
variables (Richmond Victoria Beacon : ℕ)

-- Given conditions
def condition1 : Prop := Richmond = Victoria + 1000
def condition2 : Prop := Victoria = 4 * Beacon
def condition3  : Prop := Richmond = 3000

-- The theorem to prove the population of Beacon
theorem population_of_Beacon 
  (h1 : condition1) 
  (h2 : condition2) 
  (h3 : condition3) : 
  Beacon = 500 :=
sorry

end population_of_Beacon_l631_631609


namespace smallest_angle_between_b_and_c_l631_631201

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)
variables (h_a : ∥a∥ = 2) (h_b : ∥b∥ = 3) (h_c : ∥c∥ = 4)
variables (h_eq : a ×ᵥ (b ×ᵥ c) + 2 • b = 0)

open_locale real

theorem smallest_angle_between_b_and_c (θ : ℝ) :
  θ = 0 :=
sorry

end smallest_angle_between_b_and_c_l631_631201


namespace correct_statements_count_l631_631383

theorem correct_statements_count :
  let s1 := (∀ x : ℚ, x = ⌊x⌋ ∨ x ≠ ⌊x⌋)
  let s2 := (¬ (0 = ⌊0⌋ ∨ 0 ≠ ⌊0⌋))
  let s3 := (∀ x : ℚ, x > 0 ∨ x < 0)
  let s4 := (∀ x : ℚ, (x > 0 ∨ x < 0) → (∃ n : ℤ, x = n))
  (s1 + s2 + s3 + s4 = 2) :=
sorry

end correct_statements_count_l631_631383


namespace translate_sine_eq_simplified_l631_631291

theorem translate_sine_eq_simplified (a : ℝ) (h : 0 < a ∧ a < real.pi) :
  (∀ x : ℝ, sin (2 * (x - a) + real.pi / 3) = sin (2 * x)) ↔ a = real.pi / 6 :=
by {
  sorry
}

end translate_sine_eq_simplified_l631_631291


namespace f_x1_x2_condition_l631_631850

def f (ω x ϕ : ℝ) : ℝ := 2 * Real.sin (ω * x + ϕ)

theorem f_x1_x2_condition
  (ω ϕ : ℝ)
  (hω_pos : ω > 0)
  (hϕ : Real.abs ϕ < Real.pi / 2)
  (h_B : f ω 0 ϕ = -1)
  (h_monotonic_incr : ∀ a b : ℝ, π / 18 < a ∧ a < b ∧ b < π / 3 → f ω a ϕ < f ω b ϕ)
  (h_period : ∀ x : ℝ, f ω x ϕ = f ω (x + π) ϕ)
  (x₁ x₂ : ℝ)
  (hx_range : -17 * Real.pi / 12 < x₁ ∧ x₁ < -2 * Real.pi / 3 ∧ -17 * Real.pi / 12 < x₂ ∧ x₂ < -2 * Real.pi / 3)
  (hx_diff : x₁ ≠ x₂)
  (hf_eq : f ω x₁ ϕ = f ω x₂ ϕ) :
  f ω (x₁ + x₂) ϕ = -1 := sorry

end f_x1_x2_condition_l631_631850


namespace cost_effectiveness_l631_631004

-- Define the variables and conditions
def num_employees : ℕ := 30
def ticket_price : ℝ := 80
def group_discount_rate : ℝ := 0.8
def women_discount_rate : ℝ := 0.5

-- Define the costs for each scenario
def cost_with_group_discount : ℝ := num_employees * ticket_price * group_discount_rate

def cost_with_women_discount (x : ℕ) : ℝ :=
  ticket_price * women_discount_rate * x + ticket_price * (num_employees - x)

-- Formalize the equivalence of cost and comparison logic
theorem cost_effectiveness (x : ℕ) (h : 0 ≤ x ∧ x ≤ num_employees) :
  if x < 12 then cost_with_women_discount x > cost_with_group_discount
  else if x = 12 then cost_with_women_discount x = cost_with_group_discount
  else cost_with_women_discount x < cost_with_group_discount :=
by sorry

end cost_effectiveness_l631_631004


namespace log_base_8_of_512_l631_631053

theorem log_base_8_of_512 : ∃ (x : ℝ), log 8 512 = x ∧ x = 3 :=
by
  let a := log 8 (8^3)
  have h1 : 512 = 8^3 := by norm_num
  have h2 : log 8 (8^3) = 3 * log 8 8 := by rw log_pow
  have h3 : log 8 8 = 1 := by rw log_self
  use 3
  rw [← h2, h3, mul_one]
  exact ⟨rfl, rfl⟩

end log_base_8_of_512_l631_631053


namespace positive_difference_is_496_l631_631721

def square (n: ℕ) : ℕ := n * n
def term1 := (square 8 + square 8) / 8
def term2 := (square 8 * square 8) / 8
def positive_difference := abs (term2 - term1)

theorem positive_difference_is_496 : positive_difference = 496 :=
by
  -- This is where the proof would go
  sorry

end positive_difference_is_496_l631_631721


namespace train_length_l631_631315

theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 300)
  (h2 : time = 33) : (speed * 1000 / 3600) * time = 2750 := by
  sorry

end train_length_l631_631315


namespace caitlinAgeIsCorrect_l631_631387

-- Define Aunt Anna's age
def auntAnnAge : Nat := 48

-- Define the difference between Aunt Anna's age and 18
def ageDifference : Nat := auntAnnAge - 18

-- Define Brianna's age as twice the difference
def briannaAge : Nat := 2 * ageDifference

-- Define Caitlin's age as 6 years younger than Brianna
def caitlinAge : Nat := briannaAge - 6

-- Theorem to prove Caitlin's age
theorem caitlinAgeIsCorrect : caitlinAge = 54 := by
  sorry -- Proof to be filled in

end caitlinAgeIsCorrect_l631_631387


namespace peter_age_problem_l631_631333

theorem peter_age_problem
  (P J : ℕ) 
  (h1 : J = P + 12)
  (h2 : P - 10 = 1/3 * (J - 10)) : P = 16 :=
sorry

end peter_age_problem_l631_631333


namespace extremum_condition_range_of_a_l631_631478

noncomputable def f (x a : ℝ) : ℝ := x - 1 - a * Real.log x

def is_increasing (a : ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → x < y → f x a ≤ f y a

def has_minimum (a : ℝ) : Prop :=
  ∃ x₀ : ℝ, (0 < x₀) ∧ (∀ x : ℝ, 0 < x → f x₀ a ≤ f x a)

theorem extremum_condition (a : ℝ) :
  (a ≤ 0 → is_increasing f a ∧ ¬has_minimum a) ∧
  (0 < a → ∃ x₀, x₀ = a ∧ f x₀ a = (a - 1) - a * Real.log a) :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x → f x a ≥ 0) → a ≤ 1 :=
sorry

end extremum_condition_range_of_a_l631_631478


namespace probability_of_b_in_rabbit_l631_631534

theorem probability_of_b_in_rabbit : 
  let word := "rabbit"
  let total_letters := 6
  let num_b_letters := 2
  (num_b_letters : ℚ) / total_letters = 1 / 3 :=
by
  sorry

end probability_of_b_in_rabbit_l631_631534


namespace triangle_AC_length_l631_631952

theorem triangle_AC_length (A B C P Q M : Type) (G_eq : A ∈ G)
  (H_eq : P ∈ H) (AB : ℝ) (BC : ℝ) (AC : ℝ) 
  (BP_eq: BP = BQ) (AM_eq_three_parts : AM / 3) :
    AB = 9 → BC = 11 → 
    AC = \frac{20\sqrt{3}}{3} :=
begin
  sorry,
end

end triangle_AC_length_l631_631952


namespace count_not_diff_of_squares_up_to_1000_l631_631139

-- Define the property of a number being representable as the difference of two squares
def is_diff_of_squares (n : ℕ) : Prop :=
  ∃ a b : ℤ, n = a^2 - b^2

-- Define the property of a number being of the form 4n + 2
def is_form_4n_plus_2 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 4 * k + 2

-- Prove the main statement
theorem count_not_diff_of_squares_up_to_1000 : 
  (Finset.filter (λ n, ¬ is_diff_of_squares n ∧ 1 ≤ n ∧ n ≤ 1000) (Finset.range 1001)).card = 250 :=
by
  sorry

end count_not_diff_of_squares_up_to_1000_l631_631139


namespace sum_smallest_largest_even_integers_l631_631604

theorem sum_smallest_largest_even_integers (m b z : ℕ) (hm_even : m % 2 = 0)
  (h_mean : z = (b + (b + 2 * (m - 1))) / 2) :
  (b + (b + 2 * (m - 1))) = 2 * z :=
by
  sorry

end sum_smallest_largest_even_integers_l631_631604


namespace find_a_l631_631121

-- Define the hyperbola and its properties
def hyperbola (x y a : ℝ) := x^2 / a^2 - y^2 / 2 = 1

-- Define the eccentricity condition
def eccentricity (a : ℝ) := let c := sqrt(a^2 + 2) in c / a = 2

-- Given a > 0 and the hyperbola condition,
-- prove that a = sqrt(6) / 3
theorem find_a (a : ℝ) (h : a > 0) :
  (∀ x y, hyperbola x y a) ∧ eccentricity a → a = sqrt(6) / 3 :=
by
  sorry

end find_a_l631_631121


namespace intersection_point_of_g_and_inverse_l631_631974

def g (x : ℝ) : ℝ := x^3 + 2 * x^2 + 18 * x + 36

theorem intersection_point_of_g_and_inverse : 
  (∃ a b : ℝ, (∀ x : ℝ, g(x) = b ↔ g(b) = x) ∧ (a = -3 ∧ b = -3)) :=
by
  sorry

end intersection_point_of_g_and_inverse_l631_631974


namespace licensePlatesCount_l631_631357

def numLicensePlates : ℕ :=
  let digitChoices := 10
  let letterChoices := 26
  let positionsForBlock := 6
  positionsForBlock * (digitChoices^5) * (letterChoices^3)

theorem licensePlatesCount :
  numLicensePlates = 10545600000 :=
by
  unfold numLicensePlates
  norm_num
  sorry

end licensePlatesCount_l631_631357


namespace range_of_c_l631_631124

-- Define the propositions p and q
def p (c : ℝ) : Prop := ∀ x : ℝ, c > 0 ∧ c < 1
def q (c : ℝ) : Prop := ∀ x : ℝ, x^2 - (real.sqrt 2) * x + c > 0

-- Lean statement of the problem
theorem range_of_c (c : ℝ) : (¬ q c) ∧ (p c ∨ q c) → 0 < c ∧ c ≤ 1/2 :=
by {
  sorry -- Proof omitted
}

end range_of_c_l631_631124


namespace positive_difference_eq_496_l631_631710

theorem positive_difference_eq_496 : 
  let a := 8 ^ 2 in 
  (a + a) / 8 - (a * a) / 8 = 496 :=
by
  let a := 8^2
  have h1 : (a + a) / 8 = 16 := by sorry
  have h2 : (a * a) / 8 = 512 := by sorry
  show (a + a) / 8 - (a * a) / 8 = 496 from by
    calc
      (a + a) / 8 - (a * a) / 8
            = 16 - 512 : by rw [h1, h2]
        ... = -496 : by ring
        ... = 496 : by norm_num

end positive_difference_eq_496_l631_631710


namespace count_numbers_containing_7_l631_631885

-- Define a predicate to check if a number contains the digit 7.
def contains_digit_7 (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∈ n.digits 10 ∧ d = 7

-- Define the set of numbers from 1 to 800.
def numbers_from_1_to_800 : set ℕ := {n | 1 ≤ n ∧ n ≤ 800}

-- Define the set of numbers from 1 to 800 that contain the digit 7.
def numbers_containing_7 : set ℕ := {n | n ∈ numbers_from_1_to_800 ∧ contains_digit_7 n}

-- The theorem to prove the required count.
theorem count_numbers_containing_7 :
  (numbers_containing_7.to_finset.card = 62) :=
sorry

end count_numbers_containing_7_l631_631885


namespace train_speed_km_hr_train_speed_km_hr_l631_631002

theorem train_speed_km_hr (length : ℝ) (time : ℝ) (conversion_factor : ℝ) 
  (h_length : length = 3.3333333333333335) (h_time : time = 2) (h_conversion : conversion_factor = 3.6) :
  (length / time) * conversion_factor = 6 :=
by
  rw [h_length, h_time, h_conversion]
  calc 
    (3.3333333333333335 / 2) * 3.6 = 1.6666666666666667 * 3.6 : by rw ←div_mul_eq_mul_div
                             ...  = 6                       : by norm_num

# More clear definition without reuse:
theorem train_speed_km_hr' :
  let length : ℝ := 3.3333333333333335; let time : ℝ := 2; let conversion_factor : ℝ := 3.6
  in (length / time) * conversion_factor = 6 :=
by
  have h_length : length = 3.3333333333333335 := rfl
  have h_time : time = 2 := rfl
  have h_conversion : conversion_factor = 3.6 := rfl
  rw [h_length, h_time, h_conversion]
  calc 
    (3.3333333333333335 / 2) * 3.6 = 1.6666666666666667 * 3.6 : by rw ←div_mul_eq_mul_div
                             ...  = 6                       : by norm_num


end train_speed_km_hr_train_speed_km_hr_l631_631002


namespace unique_ellipse_endpoints_l631_631778

variable (P T : Type) [AffineSpace T P]
-- Given two perpendicular lines and the ellipse's center O.
variables (O : P) (l1 l2 : Set P)
-- Given the tangent line t and the tangency point T.
variables (t : Set P) (T : P)

-- State that the perpendicular lines intersect at O
axiom perpendicular_intersection (H_l1_l2 : ∀ x ∈ l1 ∩ l2, x = O)
-- State that t is tangent to the ellipse at T.
axiom tangent_line (H_tangent : T ∈ t)

-- We'd need to show there's a unique set of endpoints for the ellipse.
-- Here, the actual proof steps are irrelevant; we just state the resulting fact.
theorem unique_ellipse_endpoints : ∃! (A B C D : P), endpoints_of_ellipse O l1 l2 t T A B C D :=
sorry

end unique_ellipse_endpoints_l631_631778


namespace area_of_ABCD_is_correct_l631_631385

def sides_len_of_original_square : ℝ := 6
def radius_of_semicircle : ℝ := sides_len_of_original_square / 2 -- 3

-- Distance from the side of the original square to the vertex of ABCD is 3 units (radius)
-- (Original_Square_Vertices_To_ABCD_Vertices == radius_of_semicircle)
def distance_to_vertices : ℝ := radius_of_semicircle

-- The diagonal length of square ABCD leads to the side length computed from it
def original_square_diagonal : ℝ := 12

noncomputable def side_length_of_ABCD : ℝ := original_square_diagonal / Real.sqrt 2
noncomputable def area_of_square (a : ℝ) : ℝ := a * a

theorem area_of_ABCD_is_correct :
  area_of_square side_length_of_ABCD = 72 := by
  sorry

end area_of_ABCD_is_correct_l631_631385


namespace series_limit_zero_l631_631624

open Classical

variable {a : ℕ → ℝ}

theorem series_limit_zero (h_conv : Summable a) (h_bound : ∀ n i, n ≤ i ∧ i ≤ 2 * n → a i ≤ 100 * a n) :
  tendsto (fun n => n * a n) atTop (𝓝 0) :=
begin
  sorry
end

end series_limit_zero_l631_631624


namespace count_not_diff_of_squares_up_to_1000_l631_631140

-- Define the property of a number being representable as the difference of two squares
def is_diff_of_squares (n : ℕ) : Prop :=
  ∃ a b : ℤ, n = a^2 - b^2

-- Define the property of a number being of the form 4n + 2
def is_form_4n_plus_2 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 4 * k + 2

-- Prove the main statement
theorem count_not_diff_of_squares_up_to_1000 : 
  (Finset.filter (λ n, ¬ is_diff_of_squares n ∧ 1 ≤ n ∧ n ≤ 1000) (Finset.range 1001)).card = 250 :=
by
  sorry

end count_not_diff_of_squares_up_to_1000_l631_631140


namespace sequence_arithmetic_S_n_formula_l631_631848

-- Define the function f
def f (x : ℝ) : ℝ := x / (3 * x + 1)

-- Define the sequence a_n
def a : ℕ → ℝ
| 0       := 1
| (n + 1) := f (a n)

-- Define the sequence S_n which we need to find
def S : ℕ → ℝ
| 0       := 0
| (n + 1) := (a n * a (n + 1)) + S n

theorem sequence_arithmetic :
  ∀ n : ℕ, (n > 0) → (1 / a n) = (3 * n - 2) := sorry

theorem S_n_formula (n : ℕ) :
  S n = n / (3 * n + 1) := sorry

end sequence_arithmetic_S_n_formula_l631_631848


namespace problem_solutions_l631_631839

theorem problem_solutions (a b c : ℝ) (h : ∀ x, ax^2 + bx + c ≤ 0 ↔ x ≤ -4 ∨ x ≥ 3) :
  (a + b + c > 0) ∧ (∀ x, bx + c > 0 ↔ x < 12) :=
by
  -- The following proof steps are not needed as per the instructions provided
  sorry

end problem_solutions_l631_631839


namespace fourth_side_length_l631_631764

noncomputable def length_of_fourth_side (r : ℝ) (a b c : ℝ) : ℝ :=
  if r = 300 * Real.sqrt 2 ∧ a = 300 ∧ b = 300 ∧ c = 300 then 600 else 0

theorem fourth_side_length :
  ∀ (r a b c d : ℝ),
    (r = 300 * Real.sqrt 2) → (a = 300) → (b = 300) → (c = 300) → 
    (d = length_of_fourth_side r a b c) → (d = 600) :=
by
  intros r a b c d hr ha hb hc hd
  rw [length_of_fourth_side, if_pos] at hd
    (repeat {assumption}),
  assumption,
  sorry

end fourth_side_length_l631_631764


namespace projection_of_a_on_a_plus_b_l631_631465

noncomputable def unit_vector (v : ℝ × ℝ × ℝ) : Prop := ∥v∥ = 1

noncomputable def vector_projection (a b : ℝ × ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2 + a.3 * b.3) / ((b.1^2 + b.2^2 + b.3^2)^(1/2))

theorem projection_of_a_on_a_plus_b (a b : ℝ × ℝ × ℝ)
  (ha : unit_vector a) (hb : unit_vector b)
  (h : ∥(a.1 + b.1, a.2 + b.2, a.3 + b.3)∥ = real.sqrt 2 * ∥(a.1 - b.1, a.2 - b.2, a.3 - b.3)∥) :
  vector_projection a (a.1 + b.1, a.2 + b.2, a.3 + b.3) = real.sqrt 6 / 3 :=
  sorry

end projection_of_a_on_a_plus_b_l631_631465


namespace positive_difference_is_496_l631_631719

def square (n: ℕ) : ℕ := n * n
def term1 := (square 8 + square 8) / 8
def term2 := (square 8 * square 8) / 8
def positive_difference := abs (term2 - term1)

theorem positive_difference_is_496 : positive_difference = 496 :=
by
  -- This is where the proof would go
  sorry

end positive_difference_is_496_l631_631719


namespace sequence_sum_difference_l631_631827

def sequence_sum (n : ℕ) : ℤ :=
  ∑ k in finset.range n, (-1)^(k+1) * (4 * (k + 1) - 3)

theorem sequence_sum_difference :
  sequence_sum 22 - sequence_sum 11 = -65 :=
by
  sorry

end sequence_sum_difference_l631_631827


namespace largest_by_changing_first_digit_l631_631728

-- Define the original number
def original_number : ℝ := 0.7162534

-- Define the transformation that changes a specific digit to 8
def transform_to_8 (n : ℕ) (d : ℝ) : ℝ :=
  match n with
  | 1 => 0.8162534
  | 2 => 0.7862534
  | 3 => 0.7182534
  | 4 => 0.7168534
  | 5 => 0.7162834
  | 6 => 0.7162584
  | 7 => 0.7162538
  | _ => d

-- State the theorem
theorem largest_by_changing_first_digit :
  ∀ (n : ℕ), transform_to_8 1 original_number ≥ transform_to_8 n original_number :=
by
  sorry

end largest_by_changing_first_digit_l631_631728


namespace number_of_solutions_l631_631476

noncomputable def f (x : ℝ) (b c : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + b * x + c else 2

theorem number_of_solutions (b c : ℝ)
  (h1 : f (-4) b c = f 0 b c)
  (h2 : f (-2) b c = -2) :
  (setOf (fun x => f x b c = x)).finite.toFinset.card = 3 :=
by
  sorry

end number_of_solutions_l631_631476


namespace arithmetic_geometric_mean_inequality_l631_631617

theorem arithmetic_geometric_mean_inequality (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) : 
  (a + b + c) / 3 ≥ (a * b * c) ^ (1 / 3) :=
sorry

end arithmetic_geometric_mean_inequality_l631_631617


namespace number_of_rabbits_l631_631350

theorem number_of_rabbits
  (dogs : ℕ) (cats : ℕ) (total_animals : ℕ)
  (joins_each_cat : ℕ → ℕ)
  (hares_per_rabbit : ℕ)
  (h_dogs : dogs = 1)
  (h_cats : cats = 4)
  (h_total : total_animals = 37)
  (h_hares_per_rabbit : hares_per_rabbit = 3)
  (H : total_animals = dogs + cats + 4 * joins_each_cat cats + 3 * 4 * joins_each_cat cats) :
  joins_each_cat cats = 2 :=
by
  sorry

end number_of_rabbits_l631_631350


namespace solve_for_x_l631_631592

theorem solve_for_x (x : ℝ) (h : arccos(3 * x) - arccos(x) = π / 3) : x = - (3 * sqrt 21 / 28) :=
sorry

end solve_for_x_l631_631592


namespace cos_angle_value_l631_631133

noncomputable def vector_a : ℝ × ℝ × ℝ := (1, 1, 2)
noncomputable def vector_b : ℝ × ℝ × ℝ := (2, -1, 2)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

def cos_angle (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / (magnitude v1 * magnitude v2)

theorem cos_angle_value :
  cos_angle vector_a vector_b = 5 * Real.sqrt 6 / 18 :=
by
  sorry

end cos_angle_value_l631_631133


namespace positive_difference_eq_496_l631_631711

theorem positive_difference_eq_496 : 
  let a := 8 ^ 2 in 
  (a + a) / 8 - (a * a) / 8 = 496 :=
by
  let a := 8^2
  have h1 : (a + a) / 8 = 16 := by sorry
  have h2 : (a * a) / 8 = 512 := by sorry
  show (a + a) / 8 - (a * a) / 8 = 496 from by
    calc
      (a + a) / 8 - (a * a) / 8
            = 16 - 512 : by rw [h1, h2]
        ... = -496 : by ring
        ... = 496 : by norm_num

end positive_difference_eq_496_l631_631711


namespace unique_A_for_prime_number_l631_631270

def is_prime (n : ℕ) : Prop := Nat.Prime n

def six_digit_number_with_A (A : ℕ) : ℕ := 202100 + A

theorem unique_A_for_prime_number :
  ∃! (A : ℕ), A ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ is_prime (six_digit_number_with_A A) :=
begin
  use 9,
  split,
  { split,
    { simp, },
    { sorry } },
  { intros B hB,
    have HB := hB.2,
    obtain ⟨H1, H2⟩ : B = 9 := by sorry,
    exact H1,
  }
end

end unique_A_for_prime_number_l631_631270


namespace average_tree_height_is_800_l631_631545

def first_tree_height : ℕ := 1000
def other_tree_height : ℕ := first_tree_height / 2
def last_tree_height : ℕ := first_tree_height + 200
def total_height : ℕ := first_tree_height + other_tree_height + other_tree_height + last_tree_height
def average_height : ℕ := total_height / 4

theorem average_tree_height_is_800 :
  average_height = 800 := by
  sorry

end average_tree_height_is_800_l631_631545


namespace limsup_vlimsup_relationship_l631_631985

noncomputable def limsup_seq (s: ℕ → ℝ) : ℝ :=
  Inf { L : ℝ | ∃ N : ℕ, ∀ n ≥ N, s n ≤ L }

noncomputable def varlimsup_sets (A : ℕ → Set ℝ) : Set ℝ :=
  { x | ∀ N, ∃ n ≥ N, x ∈ A n }

theorem limsup_vlimsup_relationship (x_n : ℕ → ℝ) :
  let x := limsup_seq x_n
  let A_n := λ n, set.Iio (x_n n) in
  set.Iio x ⊆ varlimsup_sets A_n ∧ varlimsup_sets A_n ⊆ set.Iic x :=
by
  sorry

end limsup_vlimsup_relationship_l631_631985


namespace cost_of_building_fence_l631_631320
-- Import the necessary library

-- Define the conditions
def area_square_plot : ℕ := 289
def price_per_foot : ℕ := 55

-- Define the theorem that we want to prove 
-- (that the cost of building the fence is 3740 Rs)
theorem cost_of_building_fence : 4 * (nat.sqrt area_square_plot) * price_per_foot = 3740 := by
  -- Provide a placeholder for the proof
  sorry

end cost_of_building_fence_l631_631320


namespace coefficient_x3y5_in_expansion_l631_631659

theorem coefficient_x3y5_in_expansion :
  let a := (2 / 3 : ℚ)
  let b := -(1 / 3 : ℚ)
  let n := 8
  let term := (binomial n 5 : ℚ) * (a^3) * (b^5)
  term = - (448 / 6561 : ℚ) :=
by
  unfold a b n term
  sorry

end coefficient_x3y5_in_expansion_l631_631659


namespace sum_of_digits_of_least_N_l631_631206

-- Stating the problem in Lean 4
theorem sum_of_digits_of_least_N (N : ℕ) (hN : N > 0 ∧ N % 4 = 0) :
    let Q (N : ℕ) : ℚ := 
      let total_positions := (N + 2).choose 2
      let favorable_outcomes := 
        (λ k, if (⌈ (3/4 : ℚ) * N ⌉.nat_abs ≤ k ∧ k ≤ N) then
          -- number of ways to place k green balls in k slots
          -- total count of favorable outcomes should be the sum of such placements
          sorry 
        else 0
        )
        in ((λ k, favorable_outcomes k).sum_range (N + 1))
      in
      favorable_outcomes / total_positions 

    (Q N < 7 / 9) → natDigits 10 N.sum = 13 :=
by
  -- Using the conditions given, we need to prove that the sum of the digits is 13
  sorry

end sum_of_digits_of_least_N_l631_631206


namespace quadratic_roots_correct_l631_631816

def quadratic (b c : ℝ) (x : ℝ) : ℝ := x^2 + b * x + c

theorem quadratic_roots_correct (b c : ℝ) 
  (h₀ : quadratic b c (-2) = 5)
  (h₁ : quadratic b c (-1) = 0)
  (h₂ : quadratic b c 0 = -3)
  (h₃ : quadratic b c 1 = -4)
  (h₄ : quadratic b c 2 = -3)
  (h₅ : quadratic b c 4 = 5)
  : (quadratic b c (-1) = 0) ∧ (quadratic b c 3 = 0) :=
sorry

end quadratic_roots_correct_l631_631816


namespace minimum_value_expression_l631_631122

-- Define the conditions in the problem
variable (m n : ℝ) (h1 : m > 0) (h2 : n > 0)
variable (h3 : 2 * m + 2 * n = 2)

-- State the theorem proving the minimum value of the given expression
theorem minimum_value_expression : (1 / m + 2 / n) = 3 + 2 * Real.sqrt 2 := by
  sorry

end minimum_value_expression_l631_631122


namespace tangent_of_inclination_of_OP_l631_631106

noncomputable def point_P_x (φ : ℝ) : ℝ := 3 * Real.cos φ
noncomputable def point_P_y (φ : ℝ) : ℝ := 2 * Real.sin φ

theorem tangent_of_inclination_of_OP (φ : ℝ) (h: φ = Real.pi / 6) :
  (point_P_y φ / point_P_x φ) = 2 * Real.sqrt 3 / 9 :=
by
  have h1 : point_P_x φ = 3 * (Real.sqrt 3 / 2) := by sorry
  have h2 : point_P_y φ = 1 := by sorry
  sorry

end tangent_of_inclination_of_OP_l631_631106


namespace number_of_numbers_with_digit_seven_l631_631899

-- Define what it means to contain digit 7
def contains_digit_seven (n : ℕ) : Prop :=
  n.digits 10 ∈ [7]

-- Define the set of numbers from 1 to 800 containing at least one digit 7
def numbers_with_digit_seven : ℕ → Prop :=
  λ n, 1 ≤ n ∧ n ≤ 800 ∧ contains_digit_seven n

-- State the theorem
theorem number_of_numbers_with_digit_seven : (finset.filter numbers_with_digit_seven (finset.range 801)).card = 152 :=
sorry

end number_of_numbers_with_digit_seven_l631_631899


namespace polyline_not_always_possible_l631_631090

def LineSegment := (ℝ × ℝ) × (ℝ × ℝ)

def non_overlapping (segments : List LineSegment) : Prop :=
  ∀ s1 s2 ∈ segments, s1 ≠ s2 → ¬(intersection s1 s2)

def no_same_straight_line (segments : List LineSegment) : Prop :=
  ∀ s1 s2 ∈ segments, s1 ≠ s2 → ¬(collinear s1 s2)

def can_form_polyline (segments : List LineSegment) : Prop :=
  ∃ (additional_segments : List LineSegment),
    (non_overlapping (segments ++ additional_segments)) ∧
    (non_self_intersecting (segments ++ additional_segments))

theorem polyline_not_always_possible (segments : List LineSegment)
  (h1 : non_overlapping segments)
  (h2 : no_same_straight_line segments) :
  ¬ can_form_polyline segments := 
  sorry

end polyline_not_always_possible_l631_631090


namespace positive_difference_l631_631672

theorem positive_difference :
  let a := 8^2
  let term1 := (a + a) / 8
  let term2 := (a * a) / 8
  term2 - term1 = 496 :=
by
  let a := 8^2
  let term1 := (a + a) / 8
  let term2 := (a * a) / 8
  have h1 : a = 64 := rfl
  have h2 : term1 = 16 := by simp [a, term1]
  have h3 : term2 = 512 := by simp [a, term2]
  show 512 - 16 = 496 from sorry

end positive_difference_l631_631672


namespace toothpicks_required_l631_631757

noncomputable def total_small_triangles (n : ℕ) : ℕ :=
  n * (n + 1) / 2

noncomputable def total_initial_toothpicks (n : ℕ) : ℕ :=
  3 * total_small_triangles n

noncomputable def adjusted_toothpicks (n : ℕ) : ℕ :=
  total_initial_toothpicks n / 2

noncomputable def boundary_toothpicks (n : ℕ) : ℕ :=
  2 * n

noncomputable def total_toothpicks (n : ℕ) : ℕ :=
  adjusted_toothpicks n + boundary_toothpicks n

theorem toothpicks_required {n : ℕ} (h : n = 2500) : total_toothpicks n = 4694375 :=
by sorry

end toothpicks_required_l631_631757


namespace greatest_number_of_kits_l631_631656

-- Given conditions
def bottles_of_water := 20
def cans_of_food := 12
def flashlights := 30
def blankets := 18

def no_more_than_10_items_per_kit (kits : ℕ) := 
  (bottles_of_water / kits ≤ 10) ∧ 
  (cans_of_food / kits ≤ 10) ∧ 
  (flashlights / kits ≤ 10) ∧ 
  (blankets / kits ≤ 10)

def greater_than_or_equal_to_5_kits (kits : ℕ) := kits ≥ 5

def all_items_distributed_equally (kits : ℕ) := 
  (bottles_of_water % kits = 0) ∧ 
  (cans_of_food % kits = 0) ∧ 
  (flashlights % kits = 0) ∧ 
  (blankets % kits = 0)

-- Proof goal
theorem greatest_number_of_kits : 
  ∃ kits : ℕ, 
    no_more_than_10_items_per_kit kits ∧ 
    greater_than_or_equal_to_5_kits kits ∧ 
    all_items_distributed_equally kits ∧ 
    kits = 6 := 
sorry

end greatest_number_of_kits_l631_631656


namespace log_sum_implies_product_l631_631142

theorem log_sum_implies_product (a b : ℝ) (h : log 10 a + log 10 b = 1) : a * b = 10 :=
sorry

end log_sum_implies_product_l631_631142


namespace tangent_line_eq_k2_no_zeros_range_k_two_distinct_zeros_ln_sum_gt_2_l631_631566

section
variable {k : ℝ} (f : ℝ → ℝ) (x1 x2 : ℝ)

-- Define the function f(x) = ln x - kx
def f := λ x : ℝ, Real.log x - k * x

-- 1. Prove the equation of the tangent line when k = 2 at x = 1
theorem tangent_line_eq_k2 : (k = 2) → (∃ (y : ℝ → ℝ), ∀ x, y x = f 1 + (f' 1) * (x - 1) → x + y x + 1 = 0) :=
by
  sorry

-- 2. Prove the range of k when f(x) has no zeros
theorem no_zeros_range_k : (∀ x : ℝ, f x ≠ 0) ↔ (k > 1 / Real.exp 1) :=
by
  sorry

-- 3. Prove ln x1 + ln x2 > 2 for two distinct zeros of f(x)
theorem two_distinct_zeros_ln_sum_gt_2 : (f x1 = 0) → (f x2 = 0) → (x1 ≠ x2) → x1 > 0 → x2 > 0 → Real.log x1 + Real.log x2 > 2 :=
by
  sorry
end

end tangent_line_eq_k2_no_zeros_range_k_two_distinct_zeros_ln_sum_gt_2_l631_631566


namespace b_value_for_continuity_l631_631065

noncomputable def g (x : ℝ) (b : ℝ) : ℝ :=
if x > 2 then x + 4 else 3 * x + b

theorem b_value_for_continuity :
  ∃ b : ℝ, (∀ x : ℝ, g x b = if x > 2 then x + 4 else 3 * x + b) ∧ (3 * 2 + b = 2 + 4 → b = 0) :=
begin
  use 0,
  split,
  { intros x, unfold g, },
  { intro h,
    exact h },
end

end b_value_for_continuity_l631_631065


namespace sum_first_six_terms_l631_631470

-- Let {a_n} be a geometric sequence with a_2 = 2 and a_4 = 8
variables {a : ℕ → ℝ} (q : ℝ) {n : ℕ}
-- Assume all terms of the sequence are positive
variable (h_pos : ∀ n, 0 < a n)
-- Assume the given conditions of the sequence
variables (h2 : a 2 = 2) (h4 : a 4 = 8)

-- Define the sum of the first six terms of the sequence
def S_6 := ∑ i in range 6, a i

-- Main theorem to be proved: the sum of the first six terms of the sequence is 63
theorem sum_first_six_terms : S_6 = 63 :=
sorry

end sum_first_six_terms_l631_631470


namespace max_value_of_f_l631_631810

noncomputable def f (x : ℝ) : ℝ := 
  Real.cot (x + 3 * Real.pi / 4) - Real.cot (x + Real.pi / 4) + Real.sin (x + Real.pi / 4)

theorem max_value_of_f :
  ∀ x ∈ Set.Icc (-2 * Real.pi / 3) (-Real.pi / 4), f x ≤ Real.sqrt 2 / 2 ∧
  ∃ x ∈ Set.Icc (-2 * Real.pi / 3) (-Real.pi / 4), f x = Real.sqrt 2 / 2 :=
by
  sorry

end max_value_of_f_l631_631810


namespace concyclic_points_l631_631572

section concyclic_problem

variables (Γ₁ Γ₂ : Type*) [Circle Γ₁] [Circle Γ₂]
variables (A B D C E : Point)
variables (tangent_Γ₂_A : Tangent Γ₂ A)
variables (tangent_Γ₁_A : Tangent Γ₁ A)

-- Conditions
def circles_intersect_at : Prop := OnCircle A Γ₁ ∧ OnCircle A Γ₂ ∧ OnCircle B Γ₁ ∧ OnCircle B Γ₂
def tangent_to_Γ₁_at_A : Prop := tangent_Γ₁_A.point = A ∧ TangentPoint tangent_Γ₁_A D ∧ OnCircle D Γ₁
def tangent_to_Γ₂_at_A : Prop := tangent_Γ₂_A.point = A ∧ TangentPoint tangent_Γ₂_A C ∧ OnCircle C Γ₂
def E_is_reflection_of_A_wrt_B : Prop := Reflection A B E

-- Question: Show that A, D, E, and C are concyclic.
theorem concyclic_points 
  (h1 : circles_intersect_at Γ₁ Γ₂ A B)
  (h2 : tangent_to_Γ₁_at_A Γ₁ A D tangent_Γ₁_A)
  (h3 : tangent_to_Γ₂_at_A Γ₂ A C tangent_Γ₂_A)
  (h4 : E_is_reflection_of_A_wrt_B A B E) : Concyclic A D E C :=
sorry

end concyclic_problem

end concyclic_points_l631_631572


namespace octahedron_side_length_l631_631581

-- Define the vertices of the cube
def Q1 := (0 : ℝ, 0 : ℝ, 0 : ℝ)
def Q2 := (0 : ℝ, 2 : ℝ, 0 : ℝ)
def Q3 := (0 : ℝ, 0 : ℝ, 2 : ℝ)
def Q4 := (0 : ℝ, 2 : ℝ, 2 : ℝ)
def Q1' := (2 : ℝ, 0 : ℝ, 0 : ℝ)
def Q2' := (2 : ℝ, 2 : ℝ, 0 : ℝ)
def Q3' := (2 : ℝ, 0 : ℝ, 2 : ℝ)
def Q4' := (2 : ℝ, 2 : ℝ, 2 : ℝ)

-- Define vertices of the octahedron
def O1 := (\(2 : ℝ) / 3, 0 : ℝ, 0 : ℝ)
def O2 := (0 : ℝ, \(2 : ℝ) / 3, 0 : ℝ)
def O3 := (0 : ℝ, 0 : ℝ, \(2 : ℝ) / 3)
def O4 := (\(4 : ℝ) / 3, 2 : ℝ, 2 : ℝ)
def O5 := (2 : ℝ, \(4 : ℝ) / 3, 2 : ℝ)
def O6 := (2 : ℝ, 2 : ℝ, \(4 : ℝ) / 3)

-- Lean statement to prove the side length of the regular octahedron
theorem octahedron_side_length : 
  dist O1 O2 = 2 * real.sqrt 2 / 3 :=
sorry

end octahedron_side_length_l631_631581


namespace problem1_I_problem1_II_problem1_III_problem2_I_problem2_II_problem2_III_l631_631640

open Classical

-- Problem (1) - Arrangements
theorem problem1_I (b g : ℕ) (h_b : b = 6) (h_g : g = 4) : 
  (4! * 7!) = 604800 := sorry

theorem problem1_II (b g : ℕ) (h_b : b = 6) (h_g : g = 4) : 
  (6! * 7!/3!) = 103680 := sorry

theorem problem1_III (b g : ℕ) (h_b : b = 6) (h_g : g = 4) : 
  (10! / 3!) = 3628800 := sorry

-- Problem (2) - Distribution Methods
theorem problem2_I : 
  (choose 6 1) * (choose 5 2) * (choose 3 3) = 60 := sorry

theorem problem2_II : 
  (choose 6 2) * (choose 4 2) / (3!) = 15 := sorry

theorem problem2_III : 
  (choose 6 1) * (choose 5 1) * (choose 4 4) * 3 + 
  (choose 6 1) * (choose 5 2) * (choose 3 3) * (3!) +
  (choose 6 2) * (choose 4 2) * (choose 2 2) = 540 := sorry

end problem1_I_problem1_II_problem1_III_problem2_I_problem2_II_problem2_III_l631_631640


namespace count_numbers_with_digit_7_in_range_l631_631876

theorem count_numbers_with_digit_7_in_range : 
  let numbers_in_range := {n : ℕ | 1 ≤ n ∧ n ≤ 800}
      contains_digit_7 (n : ℕ) : Prop := n.digits 10.contains 7
  in (finset.filter (λ n, contains_digit_7 n) (finset.range 801)).card = 152 :=
by 
  let numbers_in_range := {n : ℕ | 1 ≤ n ∧ n ≤ 800}
  let contains_digit_7 (n : ℕ) : Prop := n.digits 10.contains 7
  have h := (finset.filter (λ n, contains_digit_7 n) (finset.range 801)).card
  sorry

end count_numbers_with_digit_7_in_range_l631_631876


namespace rationalize_denominator_l631_631241

theorem rationalize_denominator :
  let A := -12
  let B := 7
  let C := 9
  let D := 13
  let E := 5
  A + B + C + D + E = 22 :=
by
  -- Proof goes here
  sorry

end rationalize_denominator_l631_631241


namespace team_total_points_l631_631930

theorem team_total_points : 
  ∀ (Tobee Jay Sean : ℕ),
  (Tobee = 4) →
  (Jay = Tobee + 6) →
  (Sean = Tobee + Jay - 2) →
  (Tobee + Jay + Sean = 26) :=
by
  intros Tobee Jay Sean h1 h2 h3
  rw [h1, h2, h3]
  sorry

end team_total_points_l631_631930


namespace find_a_b_l631_631439

theorem find_a_b (a b : ℝ)
  (h1 : a < 0)
  (h2 : (-b / a) = -((1 / 2) - (1 / 3)))
  (h3 : (2 / a) = -((1 / 2) * (1 / 3))) : 
  a + b = -14 :=
sorry

end find_a_b_l631_631439


namespace numbers_with_7_in_1_to_800_l631_631909

theorem numbers_with_7_in_1_to_800 : 
  (card { n ∈ finset.range (800 + 1) | ∃ d ∈ n.digits 10, d = 7 }) = 152 := 
sorry

end numbers_with_7_in_1_to_800_l631_631909


namespace best_choice_for_square_formula_l631_631251

theorem best_choice_for_square_formula : 
  (89.8^2 = (90 - 0.2)^2) :=
by sorry

end best_choice_for_square_formula_l631_631251


namespace calc_limit_l631_631391

noncomputable
def limit_sqrt2_over_24 := 
  ∀ ε > 0, ∃ δ > 0, ∀ x, abs x < δ → abs ( ( (√(x+2) - √2) / (sin (3*x)) ) - (√2 / 24) ) < ε

theorem calc_limit : limit_sqrt2_over_24 :=
sorry

end calc_limit_l631_631391


namespace steady_numbers_count_l631_631922

def isSteadyNumber (a b c : Nat) : Prop :=
  abs (a - b) ≤ 1 ∧ abs (b - c) ≤ 1

def isThreeDigitNumber (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000

noncomputable def countSteadyNumbers : Nat :=
  (List.range 900).filter (λ n => 
    let a := n / 100
    let b := (n % 100) / 10
    let c := n % 10
    isSteadyNumber a b c
  ).length

theorem steady_numbers_count : countSteadyNumbers = 75 :=
  sorry

end steady_numbers_count_l631_631922


namespace smallest_set_symmetric_l631_631767

def point := ℝ × ℝ

def T (point_set : set point) :=
  (∀ p, p ∈ point_set → (-p.1, -p.2) ∈ point_set) ∧
  (∀ p, p ∈ point_set → ( p.1, -p.2) ∈ point_set) ∧
  (∀ p, p ∈ point_set → (-p.1,  p.2) ∈ point_set) ∧
  (∀ p, p ∈ point_set → ( p.2,  p.1) ∈ point_set)

def elem := (3, 4) : point

noncomputable def symmetric_points : set point :=
  { (3, 4), (-3, -4), (3, -4), (-3, 4), (4, 3), (-4, -3), (4, -3), (-4, 3) }

theorem smallest_set_symmetric : T symmetric_points ∧ elem ∈ symmetric_points ∧ ∀ (S : set point), (T S ∧ elem ∈ S) → (∀ p, p ∈ symmetric_points → p ∈ S) :=
by {
  split,
  -- Proving that symmetric_points set holds the symmetry conditions T
  sorry,
  split,
  -- Proving that elem is in the symmetric_points set
  sorry,
  -- Proving that any set S which holds the symmetry conditions T and contains elem must also contain symmetric_points
  sorry,
}

end smallest_set_symmetric_l631_631767


namespace find_A_find_k_range_l631_631117

noncomputable def f (x : ℝ) : ℝ := 
  real.log (x^2 - 2 * x) / real.sqrt (9 - x^2)

def A : set ℝ := {x | -3 < x ∧ x < 0} ∪ {x | 2 < x ∧ x < 3}

def B (k : ℝ) : set ℝ := { x | x^2 - 2 * x + 1 - k^2 ≥ 0 }

def valid_k (k : ℝ) : Prop := ∃ x, x ∈ A ∧ x ∈ B k

theorem find_A : A = {x | -3 < x ∧ x < 0} ∪ {x | 2 < x ∧ x < 3} :=
sorry

theorem find_k_range : ∀ k : ℝ, valid_k k ↔ k ∈ set.Icc (-4 : ℝ) (4 : ℝ) :=
sorry

end find_A_find_k_range_l631_631117


namespace tangent_parallel_to_line_l631_631056

theorem tangent_parallel_to_line (x y : ℝ) :
  (y = x^3 + x - 1) ∧ (3 * x^2 + 1 = 4) → (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -3) := by
  sorry

end tangent_parallel_to_line_l631_631056


namespace arithmetic_progression_term_l631_631634

variable (n r : ℕ)

-- Given the sum of the first n terms of an arithmetic progression is S_n = 3n + 4n^2
def S (n : ℕ) : ℕ := 3 * n + 4 * n^2

-- Prove that the r-th term of the sequence is 8r - 1
theorem arithmetic_progression_term :
  (S r) - (S (r - 1)) = 8 * r - 1 :=
by
  sorry

end arithmetic_progression_term_l631_631634


namespace range_of_a_l631_631616

theorem range_of_a (a : ℝ) (a_n : ℕ+ → ℝ) 
  (h_seq : ∀ n : ℕ+, a_n n = if n ≤ 4 then 2^n - 1 else -n^2 + (a - 1) * n) 
  (h_max : ∀ n : ℕ+, a_n 5 ≥ a_n n) : 9 ≤ a ∧ a ≤ 12 :=
by
  sorry

end range_of_a_l631_631616


namespace determine_AB_length_l631_631928

-- We are dealing with a right triangle
variables (A B C : Type) [Real] (angle : A → B → C → ℝ) (tan : B → ℝ) (A B : Point) [MetricSpace A] 

noncomputable def AB_length (A B C : A) (angle_A_eq_90 : angle A B C = π/2) (tan_B_eq_5_over_12 : tan B = 5/12) (AC_eq_39 : dist A C = 39) : Prop :=
  dist A B = 36

theorem determine_AB_length 
  (A B C : Point) 
  (angle_A_eq_90 : angle A B C = π/2)
  (tan_B_eq_5_over_12 : tan B = 5/12)
  (AC_eq_39 : dist A C = 39) :
  dist A B = 36 := 
sorry

end determine_AB_length_l631_631928


namespace fourth_power_nested_sqrt_l631_631389

noncomputable def nested_sqrt : ℝ := Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2))

theorem fourth_power_nested_sqrt : nested_sqrt ^ 4 = 16 := by
  sorry

end fourth_power_nested_sqrt_l631_631389


namespace third_grade_boys_count_l631_631280

-- Definitions based on the conditions
variables (x y b g : ℕ)
variables (total_students : ℕ) (extra_fourth_graders : ℕ)
variables (fewer_third_grade_girls : ℕ) (total_third_graders : ℕ)

-- Assign the values from the problem
def total_students := 531
def extra_fourth_graders := 31
def fewer_third_grade_girls := 22
def total_third_graders := 250

-- Conditions from the problem
axiom h1 : y = x + extra_fourth_graders
axiom h2 : x + y = total_students
axiom h3 : g = b - fewer_third_grade_girls
axiom h4 : b + g = total_third_graders

-- Statement to prove
theorem third_grade_boys_count : b = 136 :=
by sorry

end third_grade_boys_count_l631_631280


namespace factorization_l631_631426

theorem factorization (x : ℝ) : x^2 - 3 * x = x * (x - 3) :=
sorry

end factorization_l631_631426


namespace g_inv_f_five_l631_631016

-- Definition of the functions based on given conditions
variable {X : Type} [Nonempty X] (f g : X → X) 
  (hf : Function.Bijective f)
  (hg : Function.Bijective g)
  (h : ∀ x : X, Function.Injective f → Function.Injective g → f (g x) = 7 * x - 4)

-- The main goal to prove
theorem g_inv_f_five (hf : Function.Bijective f) (hg : Function.Bijective g) :
  g⁻¹ (f 5) = 9 / 7 := by
  sorry

end g_inv_f_five_l631_631016


namespace equal_tuesdays_thursdays_l631_631759

theorem equal_tuesdays_thursdays (days_in_month : ℕ) (tuesdays : ℕ) (thursdays : ℕ) : (days_in_month = 30) → (tuesdays = thursdays) → (∃ (start_days : Finset ℕ), start_days.card = 2) :=
by
  sorry

end equal_tuesdays_thursdays_l631_631759


namespace find_radius_omega1_find_angle_BDC_l631_631588

noncomputable def radius_omega1 (BK : ℝ) (DT : ℝ) : ℝ :=
  if BK = 3 * real.sqrt 3 ∧ DT = real.sqrt 3 then 3 else 0

noncomputable def angle_BDC (O1_center_circum_BOC : Prop) : ℝ :=
  if O1_center_circum_BOC then 30 else 0

theorem find_radius_omega1 (h : BK = 3 * real.sqrt 3 ∧ DT = real.sqrt 3) : radius_omega1 BK DT = 3 :=
by sorry

theorem find_angle_BDC (h : O1_center_circum_BOC) : angle_BDC O1_center_circum_BOC = 30 :=
by sorry

end find_radius_omega1_find_angle_BDC_l631_631588


namespace complex_product_l631_631558

-- Definitions for the given conditions
def P : ℂ := 3 + 4 * complex.I
def R : ℂ := 2 * complex.I
def S : ℂ := 3 - 4 * complex.I

-- Statement of the proof problem
theorem complex_product : P * R * S = 50 * complex.I := by
  -- This is to ensure the statement can be checked; no proof is required here.
  sorry

end complex_product_l631_631558


namespace positive_difference_eq_496_l631_631713

theorem positive_difference_eq_496 : 
  let a := 8 ^ 2 in 
  (a + a) / 8 - (a * a) / 8 = 496 :=
by
  let a := 8^2
  have h1 : (a + a) / 8 = 16 := by sorry
  have h2 : (a * a) / 8 = 512 := by sorry
  show (a + a) / 8 - (a * a) / 8 = 496 from by
    calc
      (a + a) / 8 - (a * a) / 8
            = 16 - 512 : by rw [h1, h2]
        ... = -496 : by ring
        ... = 496 : by norm_num

end positive_difference_eq_496_l631_631713


namespace clock_angle_solution_l631_631386

theorem clock_angle_solution (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 360) :
    (θ = 15) ∨ (θ = 165) :=
by
  sorry

end clock_angle_solution_l631_631386


namespace tangent_slopes_l631_631436

open Real

theorem tangent_slopes (a : ℝ) (curves : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), (m = sqrt (51) / 2 ∨ m = -sqrt (51) / 2 ∨ m = sqrt (285) / 5 ∨ m = -sqrt (285) / 5)):
  let curve := λ x y : ℝ => y^2 = x^3 + 39*x - 35
  let tangent_condition := λ x y : ℝ => y = m*x
  sorry

end tangent_slopes_l631_631436


namespace triangle_angle_sum_l631_631935

-- Definitions of the given angles and relationships
def angle_BAC := 95
def angle_ABC := 55
def angle_ABD := 125

-- We need to express the configuration of points and the measure of angle ACB
noncomputable def angle_ACB (angle_BAC angle_ABC angle_ABD : ℝ) : ℝ :=
  180 - angle_BAC - angle_ABC

-- The formalization of the problem statement in Lean 4
theorem triangle_angle_sum (angle_BAC angle_ABC angle_ABD : ℝ) :
  angle_BAC = 95 → angle_ABC = 55 → angle_ABD = 125 → angle_ACB angle_BAC angle_ABC angle_ABD = 30 :=
by
  intros h_BAC h_ABC h_ABD
  rw [h_BAC, h_ABC, h_ABD]
  sorry

end triangle_angle_sum_l631_631935


namespace positive_difference_l631_631704

def a := 8^2
def b := a + a
def c := a * a
theorem positive_difference : ((b / 8) - (c / 8)) = 496 := by
  sorry

end positive_difference_l631_631704


namespace find_m_l631_631450

def vec_a := (6, 4 : ℝ × ℝ)
def vec_b := (0, 2 : ℝ × ℝ)

def vec_c (m : ℝ) : ℝ × ℝ :=
  (vec_a.1 + m * vec_b.1, vec_a.2 + m * vec_b.2)

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem find_m (m : ℝ) : magnitude (vec_c m) = 10 → m = 2 ∨ m = -6 :=
by
  sorry

end find_m_l631_631450


namespace michael_height_l631_631584

theorem michael_height (flagpole_height flagpole_shadow michael_shadow : ℝ) 
                        (h1 : flagpole_height = 50) 
                        (h2 : flagpole_shadow = 25) 
                        (h3 : michael_shadow = 5) : 
                        (michael_shadow * (flagpole_height / flagpole_shadow) = 10) :=
by
  sorry

end michael_height_l631_631584


namespace leah_daily_earnings_l631_631187

def days_in_week : ℕ := 7
def total_weeks : ℕ := 4
def total_earnings : ℕ := 1680

theorem leah_daily_earnings : 
  let days_in_4_weeks := total_weeks * days_in_week in
  let earnings_per_day := total_earnings / days_in_4_weeks in
  earnings_per_day = 60 :=
by
  let days_in_4_weeks := 28
  have h : days_in_4_weeks = 28 := by rfl
  let earnings_per_day := total_earnings / days_in_4_weeks
  have h' : earnings_per_day = 60 := by
    have h1 : total_earnings = 1680 := by rfl
    have h2 : days_in_4_weeks = 28 := by rfl
    rw [h1, h2]
    norm_num
  exact h'

end leah_daily_earnings_l631_631187


namespace hours_needed_for_5_lathes_l631_631758

noncomputable def lathes_to_process_parts (lathe_count lathe_part_rate total_parts : ℕ) : ℕ :=
  let total_rate := lathe_count * lathe_part_rate
  total_parts / total_rate

theorem hours_needed_for_5_lathes :
  ∀ (lathe_count lathe_part_rate total_parts : ℕ),
    lathe_count = 3 →
    lathe_part_rate = 60 / 4 →
    total_parts = 600 →
    lathes_to_process_parts 5 lathe_part_rate total_parts = 8 :=
by 
  intros lathe_count lathe_part_rate total_parts h1 h2 h3
  rw [h1, h2, h3]
  simp only [lathes_to_process_parts]
  norm_num
  rw [mul_left_comm, nat.div_eq_of_eq_mul_left, eq_self_iff_true]
  exact
    (ne_of_lt (by norm_num : 0 < 75)).symm
    not_zero_of_gt nat.zero_lt_succ
    sorry

end hours_needed_for_5_lathes_l631_631758


namespace exists_x_in_range_implies_m_in_set_l631_631858

variable (m : ℝ)

theorem exists_x_in_range_implies_m_in_set :
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ x^2 - x = m) ↔ m ∈ Set.range (λ x : ℝ, (x^2 - x : ℝ)) :=
by
  sorry

end exists_x_in_range_implies_m_in_set_l631_631858


namespace max_area_minimal_rectangle_l631_631190

def connected_figure (f : set (ℤ × ℤ)) : Prop :=
  -- Add a definition for connected figures here
  sorry 

def minimal_rectangle_area (f : set (ℤ × ℤ)) : ℕ :=
  -- Function to compute the area of the minimal rectangle
  sorry -- Function implementation to be added

theorem max_area_minimal_rectangle {n : ℕ} (f : set (ℤ × ℤ)) 
  (hf : connected_figure f) (h_cells : f.card = n) : 
  minimal_rectangle_area f ≤ n :=
by sorry

end max_area_minimal_rectangle_l631_631190


namespace forgot_days_l631_631370

def July_days : ℕ := 31
def days_took_capsules : ℕ := 27

theorem forgot_days : July_days - days_took_capsules = 4 :=
by
  sorry

end forgot_days_l631_631370


namespace length_of_LN_l631_631594

theorem length_of_LN 
  (sinN : ℝ)
  (LM LN : ℝ)
  (h1 : sinN = 3 / 5)
  (h2 : LM = 20)
  (h3 : sinN = LM / LN) :
  LN = 100 / 3 :=
by
  sorry

end length_of_LN_l631_631594


namespace third_number_is_32_l631_631265

theorem third_number_is_32 (A B C : ℕ) 
  (hA : A = 24) (hB : B = 36) 
  (hHCF : Nat.gcd (Nat.gcd A B) C = 32) 
  (hLCM : Nat.lcm (Nat.lcm A B) C = 1248) : 
  C = 32 := 
sorry

end third_number_is_32_l631_631265


namespace positive_difference_l631_631687

theorem positive_difference : 496 = abs ((64 + 64) / 8 - (64 * 64) / 8) := by
  have h1 : 8^2 = 64 := rfl
  have h2 : 64 + 64 = 128 := rfl
  have h3 : (128 : ℕ) / 8 = 16 := rfl
  have h4 : 64 * 64 = 4096 := rfl
  have h5 : (4096 : ℕ) / 8 = 512 := rfl
  have h6 : 512 - 16 = 496 := rfl
  sorry

end positive_difference_l631_631687


namespace grid_contains_unique_integers_l631_631216

theorem grid_contains_unique_integers : 
  ∀ (grid : Fin 101 → Fin 101 → Fin 102),
  (∀ n : Fin 102, (Finset.card (Finset.univ.filter (λ i j, grid i j = n )) = 101)) →
  ∃ i : Fin 101, (Finset.card (Finset.image (λ j, grid i j) Finset.univ) ≥ 11) ∨ 
                 ∃ j : Fin 101, (Finset.card (Finset.image (λ i, grid i j) Finset.univ) ≥ 11) :=
by sorry

end grid_contains_unique_integers_l631_631216


namespace area_of_trapezoid_RSQT_l631_631006
-- Import the required library

-- Declare the geometrical setup and given areas
variables (PQ PR : ℝ)
variable (PQR_area : ℝ)
variable (small_triangle_area : ℝ)
variable (num_small_triangles : ℕ)
variable (inner_triangle_area : ℝ)
variable (trapezoid_RSQT_area : ℝ)

-- Define the conditions from part a)
def isosceles_triangle : Prop := PQ = PR
def triangle_PQR_area_given : Prop := PQR_area = 75
def small_triangle_area_given : Prop := small_triangle_area = 3
def num_small_triangles_given : Prop := num_small_triangles = 9
def inner_triangle_area_given : Prop := inner_triangle_area = 5 * small_triangle_area

-- Define the target statement (question == answer)
theorem area_of_trapezoid_RSQT :
  isosceles_triangle PQ PR ∧
  triangle_PQR_area_given PQR_area ∧
  small_triangle_area_given small_triangle_area ∧
  num_small_triangles_given num_small_triangles ∧
  inner_triangle_area_given small_triangle_area inner_triangle_area → 
  trapezoid_RSQT_area = 60 :=
sorry

end area_of_trapezoid_RSQT_l631_631006


namespace reading_to_meditation_ratio_l631_631648

-- Definitions based on conditions
def meditation_hours_per_day : ℕ := 1
def weekly_reading_hours : ℕ := 14
def days_in_week : ℕ := 7

-- Theorem to be proved
theorem reading_to_meditation_ratio : 
  let daily_reading_hours := weekly_reading_hours / days_in_week
  in (daily_reading_hours : ℕ) / (meditation_hours_per_day : ℕ) = 2 :=
by
  sorry

end reading_to_meditation_ratio_l631_631648


namespace fiona_pairs_l631_631813

theorem fiona_pairs (n : ℕ) (h : n = 12) : 66 = nat.choose n 2 :=
by
  sorry

end fiona_pairs_l631_631813


namespace isosceles_triangle_AFG_l631_631831

theorem isosceles_triangle_AFG 
  (A B C D E F G : Point) 
  (h_trapezoid : is_isosceles_trapezoid A B C D)
  (h_paral_CD_AB : Parallel CD AB)
  (h_incircle_BCD : touches_incircle_triangle B C D E)
  (h_F_angle_bis : lies_on_angle_bisector F (angle DAC))
  (h_F_perpendicular : Perpendicular EF CD)
  (h_circumcircle_AFC : lies_on_circumcircle A F C G CD) :
  is_isosceles_triangle A F G :=
sorry

end isosceles_triangle_AFG_l631_631831


namespace dog_area_l631_631769

/-- Given a square-based dog house with a side length of 1.2 meters,
and a dog tied with a 3 meter chain to a point 0.3 meters away from 
one of the corners of the dog house, find the area in which the dog can move,
proving that it equals approximately 24.9091 square meters. -/
theorem dog_area (side_length : ℝ) (chain_length : ℝ) (distance_from_corner : ℝ) :
  side_length = 1.2 →
  chain_length = 3 →
  distance_from_corner = 0.3 →
  (∃ area : ℝ, area ≈ 24.9091) :=
by
  intros h1 h2 h3
  sorry

end dog_area_l631_631769


namespace length_AM_is_correct_l631_631734

-- Definitions of the problem conditions
def length_of_square : ℝ := 9

def ratio_AP_PB : ℝ × ℝ := (7, 2)

def radius_of_quarter_circle : ℝ := 9

-- The theorem to prove
theorem length_AM_is_correct
  (AP PB PE : ℝ)
  (x : ℝ)
  (AM : ℝ) 
  (H_AP_PB  : AP = 7 ∧ PB = 2 ∧ PE = 2)
  (H_QD_QE : x = 63 / 11)
  (H_PQ : PQ = 2 + x) :
  AM = 85 / 22 :=
by
  sorry

end length_AM_is_correct_l631_631734


namespace distance_traveled_l631_631745

noncomputable def velocity (t : ℝ) := 2 * t - 3

theorem distance_traveled : 
  (∫ t in (0 : ℝ)..5, |velocity t|) = 29 / 2 := 
by
  sorry

end distance_traveled_l631_631745


namespace hypotenuse_length_l631_631528

theorem hypotenuse_length (x : ℝ) (AB : ℝ) (h₁ : 0 < x) (h₂ : x < real.pi / 2)
  (h₃ : (AB / 4) = real.tan x) (h₄ : (AB / 2) = real.cot x) :
  AB = 4 * real.sqrt 2 :=
begin
  sorry
end

end hypotenuse_length_l631_631528


namespace elsa_data_remaining_l631_631419

variable (data_total : ℕ) (data_youtube : ℕ)

def data_remaining_after_youtube (data_total data_youtube : ℕ) : ℕ := data_total - data_youtube

def data_fraction_spent_on_facebook (data_left : ℕ) : ℕ := (2 * data_left) / 5

theorem elsa_data_remaining
  (h_data_total : data_total = 500)
  (h_data_youtube : data_youtube = 300) :
  data_remaining_after_youtube data_total data_youtube
  - data_fraction_spent_on_facebook (data_remaining_after_youtube data_total data_youtube) 
  = 120 :=
by
  sorry

end elsa_data_remaining_l631_631419


namespace f_2008_value_l631_631147

noncomputable def f : ℕ → ℝ
| 1       := 1
| (n + 1) := ((n : ℝ) / (n + 2)) * f n

theorem f_2008_value : f 2008 = 1 / (1004 * 2009) :=
by
  sorry

end f_2008_value_l631_631147


namespace expected_value_of_X_l631_631431

noncomputable def pdf (x : ℝ) : ℝ := 
  if x ≥ 0 then 0.2 * Real.exp (-0.2 * x) else 0

theorem expected_value_of_X :
  ∫ x in set.Ici 0, x * pdf x = 5 :=
by
  sorry

end expected_value_of_X_l631_631431


namespace angle_OPE_eq_angle_AMB_l631_631785

/-- 
Triangle ABC is inscribed in circle O, with ∠ABC > 90°. 
Point M is the midpoint of side BC. 
Point P is inside triangle ABC such that PB ⊥ PC. 
From point P, draw a perpendicular to AP, with D and E on this perpendicular such that BD = BP and CE = CP. 
If quadrilateral ADEO is a parallelogram, prove that ∠OPE = ∠AMB. 
-/
theorem angle_OPE_eq_angle_AMB
  {A B C O M P D E : Point}
  (h1 : Circle (center := O) (radius := dist A O))
  (h2 : ∠ B A C = 90° + θ)
  (h3 : midpoint M B C)
  (h4 : inside_triangle P A B C)
  (h5 : ⟪PB, PC⟫ = 0)
  (h6 : ⟪AP, DE⟫ = 0)
  (h7 : D ≠ P ∧ E ≠ P)
  (h8 : dist B D = dist B P ∧ dist C E = dist C P)
  (h9 : parallelogram A D E O) :
  ∠ O P E = ∠ A M B := by
  sorry

end angle_OPE_eq_angle_AMB_l631_631785


namespace incenter_circumcenter_midpoints_concyclic_l631_631179

theorem incenter_circumcenter_midpoints_concyclic
  (A B C I O D E : Point)
  (hABC : Triangle A B C)
  (hCond : 2 * (dist A B) = (dist B C) + (dist C A))
  (hI : Incenter I A B C)
  (hO : Circumcenter O A B C)
  (hD : Midpoint D B C)
  (hE : Midpoint E A C) :
  Concyclic {I, O, D, E} :=
begin
  sorry
end

end incenter_circumcenter_midpoints_concyclic_l631_631179


namespace find_integer_less_than_M_div_100_l631_631098

-- The problem and proof constants
theorem find_integer_less_than_M_div_100 :
  let M := 4992 in
  let result := ⌊M / 100⌋ in
  result = 49 :=
by
  -- The conditions given and the resultant M is defined.
  have h1 : 1 / (3! * 18!) + 1 / (4! * 17!) + 1 / (5! * 16!) + 1 / (6! * 15!) + 1 / (7! * 14!) + 
            1 / (8! * 13!) + 1 / (9! * 12!) + 1 / (10! * 11!) = M / (2! * 19!) := sorry,
  -- Hence, final result.
  have h2 : M = 4992 := sorry,
  have h3 : result = ⌊4992 / 100⌋ := by simp [M, result, int.floor_eq_iff, ←div_lt_iff, int.cast_49],
  exact h3

end find_integer_less_than_M_div_100_l631_631098


namespace find_x_squared_perfect_square_l631_631555

theorem find_x_squared_perfect_square (n m : ℕ) (h1 : 0 < n) (h2 : 0 < m) (h3 : n ≠ m)
  (h4 : n > m) (h5 : n % 2 ≠ m % 2) : 
  ∃ x : ℤ, x = 0 ∧ ∀ x, (x = 0) → ∃ k : ℕ, (x ^ (2 ^ n) - 1) / (x ^ (2 ^ m) - 1) = k^2 :=
sorry

end find_x_squared_perfect_square_l631_631555


namespace dr_jones_remaining_money_l631_631047

-- Define the conditions
def monthly_earnings : ℕ := 6000
def house_rental : ℕ := 640
def food_expense : ℕ := 380
def electric_water_bill (earnings : ℕ) : ℕ := earnings / 4
def insurances (earnings : ℕ) : ℕ := earnings / 5

-- Prove the final amount left
theorem dr_jones_remaining_money :
  let earnings := monthly_earnings,
      total_expenses := house_rental + food_expense + electric_water_bill earnings + insurances earnings,
      remaining_money := earnings - total_expenses
  in
  remaining_money = 2280 :=
by
  let earnings := monthly_earnings
  let total_expenses := house_rental + food_expense + electric_water_bill earnings + insurances earnings
  let remaining_money := earnings - total_expenses
  show remaining_money = 2280 from sorry

end dr_jones_remaining_money_l631_631047


namespace solve_quadratic_l631_631633

theorem solve_quadratic (x : ℝ) (h : x^2 - 4 = 0) : x = 2 ∨ x = -2 :=
by sorry

end solve_quadratic_l631_631633


namespace cost_of_candy_bar_l631_631410

def initial_amount : ℝ := 3.0
def remaining_amount : ℝ := 2.0

theorem cost_of_candy_bar :
  initial_amount - remaining_amount = 1.0 :=
by
  sorry

end cost_of_candy_bar_l631_631410


namespace similar_triangles_perimeter_l631_631941

open Real

-- Defining the similar triangles and their associated conditions
noncomputable def triangle1 := (4, 6, 8)
noncomputable def side2 := 2

-- Define the possible perimeters of the other triangle
theorem similar_triangles_perimeter (h : True) :
  (∃ x, x = 4.5 ∨ x = 6 ∨ x = 9) :=
sorry

end similar_triangles_perimeter_l631_631941


namespace ernie_can_make_circles_l631_631377

-- Make a statement of the problem in Lean 4
theorem ernie_can_make_circles (total_boxes : ℕ) (ali_boxes_per_circle : ℕ) (ernie_boxes_per_circle : ℕ) (ali_circles : ℕ) 
  (h1 : total_boxes = 80) (h2 : ali_boxes_per_circle = 8) (h3 : ernie_boxes_per_circle = 10) (h4 : ali_circles = 5) :
  (total_boxes - ali_boxes_per_circle * ali_circles) / ernie_boxes_per_circle = 4 := 
by 
  -- Proof of the theorem
  sorry

end ernie_can_make_circles_l631_631377


namespace solution_exists_l631_631429

theorem solution_exists :
  ∃ x y z n : ℕ, x = 3 ∧ y = 1 ∧ z = 70 ∧ n = 2 ∧ 
  (n ≥ 2) ∧ (z ≤ 5 * 2^(2 * n)) ∧ (x^(2 * n + 1) - y^(2 * n + 1) = x * y * z + 2^(2 * n + 1)) :=
by {
  use [3, 1, 70, 2],
  split; norm_num,
  split; norm_num,
  split; norm_num,
  split; norm_num,
  split; norm_num,
  norm_num,
  sorry
}

end solution_exists_l631_631429


namespace train_length_correct_l631_631368

-- Define the conditions
def speed_kmh : ℕ := 45
def time_sec : ℕ := 30
def bridge_len : ℕ := 215

-- The problem is to prove the length of the train
theorem train_length_correct : 
  let speed_mps := (speed_kmh * 1000) / 3600 in
  let distance := speed_mps * time_sec in
  distance - bridge_len = 160 :=
by
  -- We skip the proof as it is not required by this task
  sorry

end train_length_correct_l631_631368


namespace smallest_integer_sum_l631_631795

noncomputable def smallest_sum_of_roots (a b c : ℝ) : ℝ :=
    let p := a * x^2 + b * x + c
    let q := c * x^2 + b * x + a
    x₁ + x₂ + (1 / x₁) + (1 / x₂)
    where
      {x₁ x₂ : ℝ // x₁ ≠ x₂ ∧ p.eval x₁ = 0 ∧ p.eval x₂ = 0 ∧ x₁ > 0 ∧ x₂ > 0}

theorem smallest_integer_sum (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b^2 - 4 * a * c > 0) (h₃ : a * c ≠ 0) :
  smallest_sum_of_roots a b c = 5 := sorry

end smallest_integer_sum_l631_631795


namespace greatest_integer_less_than_M_over_100_l631_631095

theorem greatest_integer_less_than_M_over_100
  (h : (1/(Nat.factorial 3 * Nat.factorial 18) + 1/(Nat.factorial 4 * Nat.factorial 17) + 
        1/(Nat.factorial 5 * Nat.factorial 16) + 1/(Nat.factorial 6 * Nat.factorial 15) + 
        1/(Nat.factorial 7 * Nat.factorial 14) + 1/(Nat.factorial 8 * Nat.factorial 13) + 
        1/(Nat.factorial 9 * Nat.factorial 12) + 1/(Nat.factorial 10 * Nat.factorial 11) = 
        1/(Nat.factorial 2 * Nat.factorial 19) * (M : ℚ))) :
  ⌊M / 100⌋ = 499 :=
by
  sorry

end greatest_integer_less_than_M_over_100_l631_631095


namespace prob_event_B_prob_event_A_inter_B_l631_631931

def ball_from_can_A := {1, 2, 3}
def ball_from_can_B := {1, 2, 3, 4}

def total_outcomes := (ball_from_can_A.product ball_from_can_B).card

def event_A (x : ℕ × ℕ) : Prop := x.1 + x.2 < 5
def event_B (x : ℕ × ℕ) : Prop := x.1 * x.2 % 2 = 1

def num_event_A := (ball_from_can_A.product ball_from_can_B).filter event_A
def num_event_B := (ball_from_can_A.product ball_from_can_B).filter event_B
def num_event_A_inter_B := num_event_A.filter event_B

theorem prob_event_B : (num_event_B.card / total_outcomes : ℚ) = 1 / 3 :=
by sorry

theorem prob_event_A_inter_B : (num_event_A_inter_B.card / total_outcomes : ℚ) = 1 / 4 :=
by sorry

end prob_event_B_prob_event_A_inter_B_l631_631931


namespace ellipse_triangle_ratio_l631_631007

def ellipse_ratio (a b : ℝ) (n m : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < n) (h4 : 0 < m) : ℝ :=
  (Real.sin (Real.pi / n)) / (Real.sin (Real.pi / m))

theorem ellipse_triangle_ratio
  (a b : ℝ) (n m : ℕ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < n)
  (h4 : 0 < m) :
  ellipse_ratio a b n m = (Real.sin (Real.pi / n)) / (Real.sin (Real.pi / m)) :=
sorry

end ellipse_triangle_ratio_l631_631007


namespace problem_solution_l631_631118

noncomputable def f (x : ℝ) : ℝ := x / (Real.cos x)

variables (x1 x2 x3 : ℝ)

axiom a1 : |x1| < (Real.pi / 2)
axiom a2 : |x2| < (Real.pi / 2)
axiom a3 : |x3| < (Real.pi / 2)

axiom h1 : f x1 + f x2 ≥ 0
axiom h2 : f x2 + f x3 ≥ 0
axiom h3 : f x3 + f x1 ≥ 0

theorem problem_solution : f (x1 + x2 + x3) ≥ 0 := sorry

end problem_solution_l631_631118


namespace positive_difference_l631_631671

theorem positive_difference :
  let a := 8^2
  let term1 := (a + a) / 8
  let term2 := (a * a) / 8
  term2 - term1 = 496 :=
by
  let a := 8^2
  let term1 := (a + a) / 8
  let term2 := (a * a) / 8
  have h1 : a = 64 := rfl
  have h2 : term1 = 16 := by simp [a, term1]
  have h3 : term2 = 512 := by simp [a, term2]
  show 512 - 16 = 496 from sorry

end positive_difference_l631_631671


namespace train_length_l631_631774

-- Problem Specifications
constant T : ℝ := 6 -- Time to cross
constant V_man : ℝ := 5 -- Speed of the man in kmph
constant V_train : ℝ := 40 -- Speed of the train in kmph
constant L : ℝ := 75 -- Length of the train we need to prove

-- Conversion factors and relative speed calculation
def convert_kmph_to_mps (v : ℝ) : ℝ := v * 1000 / 3600
def V_rel := V_train + V_man

theorem train_length :
  L = convert_kmph_to_mps V_rel * T := by
  -- The proof will show the conversion and relative speed
  sorry

end train_length_l631_631774


namespace count_numbers_with_seven_l631_631901

open Finset

def contains_digit_seven (n : ℕ) : Prop :=
  ∃ d : ℕ, d ∈ digits 10 n ∧ d = 7

theorem count_numbers_with_seven : 
  (card (filter (λ n, contains_digit_seven n) (range 801))) = 152 := 
by
  sorry

end count_numbers_with_seven_l631_631901


namespace solution_set_log_l631_631925

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions and the theorem
theorem solution_set_log (h : ∀ x : ℝ, f(x) ≤ 0 ↔ x ∈ Icc (-1 : ℝ) 2) (x : ℝ) :
  (0 < x ∧ x < 1/10) ∨ 100 < x ↔ f(log x) > 0 := 
sorry

end solution_set_log_l631_631925


namespace positive_difference_is_496_l631_631718

def square (n: ℕ) : ℕ := n * n
def term1 := (square 8 + square 8) / 8
def term2 := (square 8 * square 8) / 8
def positive_difference := abs (term2 - term1)

theorem positive_difference_is_496 : positive_difference = 496 :=
by
  -- This is where the proof would go
  sorry

end positive_difference_is_496_l631_631718


namespace min_max_transformation_a_min_max_transformation_b_l631_631563

theorem min_max_transformation_a {a b : ℝ} (hmin : ∀ x : ℝ, ∀ z : ℝ, (z = (x - 1) / (x^2 + 1)) → (z ≥ a))
  (hmax : ∀ x : ℝ, ∀ z : ℝ, (z = (x - 1) / (x^2 + 1)) → (z ≤ b)) :
  (∀ x : ℝ, ∀ z : ℝ, z = (x^3 - 1) / (x^6 + 1) → z ≥ a) ∧
  (∀ x : ℝ, ∀ z : ℝ, z = (x^3 - 1) / (x^6 + 1) → z ≤ b) :=
sorry

theorem min_max_transformation_b {a b : ℝ} (hmin : ∀ x : ℝ, ∀ z : ℝ, (z = (x - 1) / (x^2 + 1)) → (z ≥ a))
  (hmax : ∀ x : ℝ, ∀ z : ℝ, (z = (x - 1) / (x^2 + 1)) → (z ≤ b)) :
  (∀ x : ℝ, ∀ z : ℝ, z = (x + 1) / (x^2 + 1) → z ≥ -b) ∧
  (∀ x : ℝ, ∀ z : ℝ, z = (x + 1) / (x^2 + 1) → z ≤ -a) :=
sorry

end min_max_transformation_a_min_max_transformation_b_l631_631563


namespace triangle_area_MEQF_l631_631170

theorem triangle_area_MEQF
  (radius_P : ℝ)
  (chord_EF : ℝ)
  (par_EF_MN : Prop)
  (MQ : ℝ)
  (collinear_MQPN : Prop)
  (P MEF : ℝ × ℝ)
  (segment_P_Q : ℝ)
  (EF_length : ℝ)
  (radius_value : radius_P = 10)
  (EF_value : chord_EF = 12)
  (MQ_value : MQ = 20)
  (MN_parallel : par_EF_MN)
  (collinear : collinear_MQPN) :
  ∃ (area : ℝ), area = 48 := 
sorry

end triangle_area_MEQF_l631_631170


namespace area_KLMN_l631_631086

-- Variables for sides of triangle ABC
variable (AB BC AC BK AN BL : ℝ)
-- Conditions given
axiom h_AB : AB = 13
axiom h_BC : BC = 14
axiom h_AC : AC = 15
axiom h_BK : BK = 14 / 13
axiom h_AN : AN = 10
axiom h_BL : BL = 1

-- Final proof statement
theorem area_KLMN : ∃ (area : ℝ), area = 36503 / 1183 :=
  by {
    use 36503 / 1183, sorry
  }

end area_KLMN_l631_631086


namespace trig_identity_example_l631_631326

theorem trig_identity_example : 4 * Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 := 
by
  -- The statement "π/12" is mathematically equivalent to 15 degrees.
  sorry

end trig_identity_example_l631_631326


namespace tangent_line_eqn_at_neg1_l631_631853

def f (x : ℝ) : ℝ := x^3 + 2 * x^2

theorem tangent_line_eqn_at_neg1 :
  let p := (-1, f (-1))
  in ∃ (m b : ℝ), m = -1 ∧ b = 1 ∧ ∀ x y : ℝ, y = f x → y = m * x + b → x + y = 0 :=
by
  let p := (-1, f (-1))
  have m := -1
  have b := 1
  existsi [m, b]
  sorry

end tangent_line_eqn_at_neg1_l631_631853


namespace percentage_of_dogs_l631_631996

theorem percentage_of_dogs (total_pets : ℕ) (percent_cats : ℕ) (bunnies : ℕ) 
  (h1 : total_pets = 36) (h2 : percent_cats = 50) (h3 : bunnies = 9) : 
  ((total_pets - ((percent_cats * total_pets) / 100) - bunnies) / total_pets * 100) = 25 := by
  sorry

end percentage_of_dogs_l631_631996


namespace pudding_cost_l631_631546

theorem pudding_cost (P : ℝ) (h1 : 75 = 5 * P + 65) : P = 2 :=
sorry

end pudding_cost_l631_631546


namespace product_value_l631_631812

theorem product_value :
  (∏ k in Finset.range 2005 + 2, (k + 2) ^ 2 / ((k + 2) ^ 2 - 1)) = 4012 / 2007 :=
by
  sorry

end product_value_l631_631812


namespace trapezoid_AC_equals_2BC_l631_631085

variable {Point : Type}
variable {A B C D K : Point}
variable {BC AD : ℝ}
variable (trapezoid : is_trapezoid A B C D)
variable (AD_eq_3BC : AD = 3 * BC)
variable (K_midpoint_BD : is_midpoint K B D)
variable (AK_bisects_∠CAD : is_angle_bisector A K C D)

/- The goal is to prove that AC = 2*BC under the given conditions. -/
theorem trapezoid_AC_equals_2BC
  (h₁ : is_trapezoid A B C D)
  (h₂ : AD = 3 * BC)
  (h₃ : is_midpoint K B D)
  (h₄ : is_angle_bisector A K C D) : 
  distance A C = 2 * distance B C :=
  sorry

end trapezoid_AC_equals_2BC_l631_631085


namespace solve_problem_1_solve_problem_2_l631_631331

noncomputable def problem_1 : ℝ :=
  (0.0625)^(1/4) + ((-3)^4)^(1/4) - ((sqrt 5 - sqrt 3)^0) + real.cbrt (3 + 3/8)

noncomputable def problem_2 : ℝ :=
  logb (1/3) (sqrt 27) + real.log10 25 + real.log10 4 +
  7^(-(real.logb 7 2)) + (-0.98)^0

-- Assertion for problem 1
theorem solve_problem_1 : problem_1 = 4 :=
by
-- Proof omitted
sorry

-- Assertion for problem 2
theorem solve_problem_2 : problem_2 = 2 :=
by
-- Proof omitted
sorry

end solve_problem_1_solve_problem_2_l631_631331


namespace find_points_with_large_triangles_l631_631107

theorem find_points_with_large_triangles (A B C D : Type) [convex_quadrilateral A B C D] (h : area (quadrilateral A B C D) = 1) :
  ∃ P1 P2 P3 P4 : Type, 
  (∀ (Q R S : Type), {Q, R, S} ⊆ {P1, P2, P3, P4} → Q ≠ R → R ≠ S → S ≠ Q → area (triangle Q R S) > 1 / 4) :=
sorry

end find_points_with_large_triangles_l631_631107


namespace vector_sum_condition_iff_angle_l631_631191

noncomputable def is_orthocenter (H A B C : Point) : Prop :=
  -- definition of H being the orthocenter of triangle ABC
  sorry

noncomputable def right_isosceles_triangle (X A H : Point) : Prop :=
  -- definition for triangle XAH being right and isosceles with hypotenuse AH
  sorry

noncomputable def opposite_sides (B X A H : Point) : Prop :=
  -- definition for B and X being on opposite sides of the line AH
  sorry

noncomputable def vector_sum (XA XC XH XB : Vec) : Prop :=
  -- definition for the vector sum condition
  XA + XC + XH = XB

theorem vector_sum_condition_iff_angle :
  ∀ (A B C H X : Point), is_orthocenter H A B C ∧ right_isosceles_triangle X A H ∧ opposite_sides B X A H →
    vector_sum (vector XA) (vector XC) (vector XH) (vector XB) ↔ ∠BAC = 45 :=
by 
  sorry

end vector_sum_condition_iff_angle_l631_631191


namespace problem1_problem2_l631_631393

theorem problem1 :
  sqrt 3 + 5 * sqrt 3 - 3 * sqrt 3 = 3 * sqrt 3 :=
by sorry

theorem problem2 :
  sqrt 81 + cbrt -27 - sqrt ((-2)^2) + abs (sqrt 3 - 2) = 6 - sqrt 3 :=
by sorry

end problem1_problem2_l631_631393


namespace ernie_circles_l631_631375

theorem ernie_circles (boxes_per_circle_ali boxes_per_circle_ernie total_boxes circles_ali : ℕ) 
  (h1 : boxes_per_circle_ali = 8)
  (h2 : boxes_per_circle_ernie = 10)
  (h3 : total_boxes = 80)
  (h4 : circles_ali = 5) : 
  (total_boxes - circles_ali * boxes_per_circle_ali) / boxes_per_circle_ernie = 4 :=
by
  sorry

end ernie_circles_l631_631375


namespace distance_B_to_center_l631_631404

/-- Definitions for the geometrical scenario -/
structure NotchedCircleGeom where
  radius : ℝ
  A_pos : ℝ × ℝ
  B_pos : ℝ × ℝ
  C_pos : ℝ × ℝ
  AB_len : ℝ
  BC_len : ℝ
  angle_ABC_right : Prop
  
  -- Conditions derived from problem statement
  radius_eq_sqrt72 : radius = Real.sqrt 72
  AB_len_eq_8 : AB_len = 8
  BC_len_eq_3 : BC_len = 3
  angle_ABC_right_angle : angle_ABC_right
  
/-- Problem statement -/
theorem distance_B_to_center (geom : NotchedCircleGeom) :
  let x := geom.B_pos.1
  let y := geom.B_pos.2
  x^2 + y^2 = 50 :=
sorry

end distance_B_to_center_l631_631404


namespace CD_cube_equality_BE_over_AF_ratio_l631_631543

variables (A B C D E F : Type*)
variables [EuclideanSpace ℝ A B C D E F]

-- Given a right triangle ABC with angle ∠C being π/2
variable (A : ∀ (A B C : ℕ), rightAngle C = π / 2)

-- Defining the lengths involved in the problem
variables (AB AC BC CD DE DF : ℝ)

variable (BE : term_declaration)

-- Altitudes definitions
variable (is_altitude_CD : altitude CD AB)
variable (is_altitude_DE : altitude DE BC)
variable (is_altitude_DF : altitude DF AC)

-- Proof Statements
theorem CD_cube_equality :
  CD ^ 3 = AB * DE * DF := sorry

theorem BE_over_AF_ratio :
  BE / AF = BC ^ 3 / AC ^ 3 := sorry

end CD_cube_equality_BE_over_AF_ratio_l631_631543


namespace distance_between_points_l631_631299

-- Define the points
def point1 := (1 : ℤ, 3 : ℤ)
def point2 := (-5 : ℤ, 7 : ℤ)

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℤ × ℤ) : ℤ := 
  Int.sqrt (((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2).toNat)

-- Define the problem statement
theorem distance_between_points : distance point1 point2 = 2 * (Int.sqrt 13) := by
  sorry

end distance_between_points_l631_631299


namespace determine_y_l631_631754

theorem determine_y (y : ℚ) : 
  let square1_area := (3 * y) ^ 2,
      square2_area := (6 * y) ^ 2,
      triangle_area := (1 / 2) * (3 * y) * (6 * y),
      total_area := triangle_area + square1_area + square2_area
  in total_area = 980 → y = 70 / 9 := 
by
  intro h
  let square1_area := (3 * y) ^ 2
  let square2_area := (6 * y) ^ 2
  let triangle_area := (1 / 2) * (3 * y) * (6 * y)
  let total_area := triangle_area + square1_area + square2_area
  have h₁ : total_area = 980 := h
  have h₂ : 54 * y ^ 2 = 980 := by
    unfold total_area square1_area square2_area triangle_area at h₁
    sorry 
  have h₃ : y ^ 2 = 490 / 27 := by
    sorry
  have h₄ : y = 70 / 9 := by
    sorry
  exact h₄
 
end determine_y_l631_631754


namespace sum_reciprocal_sequence_l631_631083

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = (∑ i in Finset.range n, a (i + 1))

theorem sum_reciprocal_sequence (a : ℕ → ℕ)
  (h_seq : sequence a) :
  ∀ n, (∑ i in Finset.range n, (1 : ℝ) / (a (i + 1))) = 3 - 1 / (2 ^ (n - 2)) :=
sorry

end sum_reciprocal_sequence_l631_631083


namespace ernie_can_make_circles_l631_631380

theorem ernie_can_make_circles :
  ∀ (boxes_initial Ali_circles Ali_boxes_per_circle Ernie_boxes_per_circle : ℕ),
  Ali_circles = 5 →
  Ali_boxes_per_circle = 8 →
  Ernie_boxes_per_circle = 10 →
  boxes_initial = 80 →
  ((boxes_initial - Ali_circles * Ali_boxes_per_circle) / Ernie_boxes_per_circle) = 4 :=
by
  intros boxes_initial Ali_circles Ali_boxes_per_circle Ernie_boxes_per_circle 
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end ernie_can_make_circles_l631_631380


namespace tan_x_value_expression_value_l631_631841

variable {x : ℝ}

-- Condition: 3 * sin(x / 2) - cos(x / 2) = 0
def condition : Prop := 3 * Math.sin (x / 2) - Math.cos (x / 2) = 0

-- Proof statement for the first part
theorem tan_x_value (h : condition) : Math.tan x = 3 / 4 := sorry

-- Proof statement for the second part
theorem expression_value (h : condition) : 
  Math.cos (2 * x) / (Real.sqrt 2 * Math.cos (π / 4 + x) * Math.sin x) = 7 / 3 := sorry

end tan_x_value_expression_value_l631_631841


namespace number_not_diff_squares_1_to_1000_l631_631137

theorem number_not_diff_squares_1_to_1000 : 
  (Finset.range 1000).filter (λ x, ∀ (a b : ℤ), a^2 - b^2 ≠ x).card = 250 :=
by sorry

end number_not_diff_squares_1_to_1000_l631_631137


namespace triangle_BO_length_l631_631219

noncomputable theory

/-- Given triangle ABC, such that a circle with AC as its diameter intersects AB 
and BC at points D and E respectively. Given angle EDC is 30 degrees, the area of 
triangle AEC is sqrt(3)/2, and the area of triangle DBE is in the ratio 1:2 to 
the area of triangle ABC. Prove that the length of segment BO, 
where O is the intersection of AE and CD, is 2. -/
theorem triangle_BO_length 
  (A B C D E O : Type) 
  (AC : ℝ) 
  (angle_EDC : ℕ)
  (area_AEC : ℝ)
  (ratio_DBE_ABC : ℝ) 
  (O_on_AE_CD : O ∈ line_segment AE ∩ line_segment CD) 
  (triangle_ABC : triangle A B C) 
  (circle_AC_diameter : circle (AC) (∅ ∈ AC) D (∅ ∈ AB) E (∅ ∈ BC))
  : 
  AC = 2 → angle_EDC = 30 → area_AEC = sqrt(3) / 2 → 
  ratio_DBE_ABC = 1 / 2 → 
  length (segment BO) = 2 :=
by {
  sorry
}

end triangle_BO_length_l631_631219


namespace max_songs_played_l631_631068

theorem max_songs_played (n m t : ℕ) (h1 : n = 50) (h2 : m = 50) (h3 : t = 180) :
  3 * n + 5 * (m - ((t - 3 * n) / 5)) = 56 :=
by
  sorry

end max_songs_played_l631_631068


namespace exponential_decreasing_l631_631923

theorem exponential_decreasing (a : ℝ) (h : ∀ x y : ℝ, x < y → (a+1)^x > (a+1)^y) : -1 < a ∧ a < 0 :=
sorry

end exponential_decreasing_l631_631923


namespace rationalize_denominator_l631_631231

theorem rationalize_denominator :
  let A := -12
  let B := 7
  let C := 9
  let D := 13
  let E := 5
  (4 * Real.sqrt 7 + 3 * Real.sqrt 13) ≠ 0 →
  B < D →
  ∀ (x : ℝ), x = (3 : ℝ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) →
    A + B + C + D + E = 22 := 
by
  intros
  -- Provide the actual theorem statement here
  sorry

end rationalize_denominator_l631_631231


namespace perfect_square_division_l631_631307

-- Definitions related to the problem statement

def is_perfect_square (x : ℕ) : Prop :=
  ∃ k : ℕ, k * k = x

def f (n : ℕ) : ℕ :=
  (n! * (n + 1)!) / 2

-- Express the main theorem to be proved
theorem perfect_square_division :
  is_perfect_square (f 23) :=
sorry   -- Proof to be filled

end perfect_square_division_l631_631307


namespace b_c_value_l631_631145

theorem b_c_value (a b c d : ℕ) 
  (h₁ : a + b = 12) 
  (h₂ : c + d = 3) 
  (h₃ : a + d = 6) : 
  b + c = 9 :=
sorry

end b_c_value_l631_631145


namespace circles_and_squares_intersections_l631_631394

-- Define the lattice points, squares, and circles conditions
def circle_intersect_segment (r : ℝ) (x : ℤ) (y : ℤ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (3 * t * x = t * y ∨ 3 * t * x - x = t * y - y) -- simplified intersection condition

def square_intersect_segment (d : ℝ) (x : ℤ) (y : ℤ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (abs ((3 * t - x) * (t - y)) ≤ d / 2) -- simplified intersection condition

-- The theorem to be proven
theorem circles_and_squares_intersections (r : ℝ) (d : ℝ) :
  (r = 1/5) → (d = 1/5) →
  let num_lattice_points := 234 in 
  let m := num_lattice_points in
  let n := num_lattice_points in
  m + n = 468 :=
by
  intros hr hd num_lattice_points m n
  rw [hr, hd]
  simp [num_lattice_points, m, n]
  exact sorry

end circles_and_squares_intersections_l631_631394


namespace num_squares_l631_631443

theorem num_squares : 
  { n : ℤ | 0 ≤ n ∧ n ≤ 20 ∧ ∃ k : ℤ, (n / (20 - n) = k * k) }.card = 4 := 
sorry

end num_squares_l631_631443


namespace probability_condition_met_l631_631427

def number_of_adults : ℕ := 15

noncomputable def valid_pairings_count : ℕ :=
14.factorial + (15.factorial / (7.factorial * 8.factorial) * 2)

noncomputable def total_pairings_count : ℕ := number_of_adults.factorial

noncomputable def probability_of_valid_pairings : ℚ :=
(valid_pairings_count : ℚ) / (total_pairings_count : ℚ)

noncomputable def m_and_n_sum : ℕ :=
let m := probability_of_valid_pairings.numerator in
let n := probability_of_valid_pairings.denominator in
m + n

theorem probability_condition_met : m_and_n_sum = 17 := by
  sorry

end probability_condition_met_l631_631427


namespace logarithm_comparison_l631_631309

theorem logarithm_comparison :
  logBase 3 4 > 1 ∧ 1 > logBase (1/3) 10 :=
by
  sorry

end logarithm_comparison_l631_631309


namespace rectangle_side_ratio_l631_631800

theorem rectangle_side_ratio (s : ℝ) (A_i A_o : ℝ)
  (h_inner_octagon : A_i = 2 * (1 + Real.sqrt 2) * s^2)
  (h_area_relation : A_o = 9 * A_i) :
  let y := s,
      x := 2 * s in
  x / y = 2 :=
by
  sorry

end rectangle_side_ratio_l631_631800


namespace fractional_equation_solution_l631_631846

theorem fractional_equation_solution (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) ↔ (m ≤ 2 ∧ m ≠ -2) := 
sorry

end fractional_equation_solution_l631_631846


namespace distinct_values_expr_l631_631863

noncomputable def expr (x : ℝ) : ℤ :=
  Int.floor x + Int.floor (2 * x) + Int.floor ((5 * x) / 3) + Int.floor (3 * x) + Int.floor (4 * x)

theorem distinct_values_expr : (finset.univ.image expr).card = 734 := by
  sorry

end distinct_values_expr_l631_631863


namespace positive_difference_l631_631681

theorem positive_difference (a k : ℕ) (h1 : a = 8^2) (h2 : k = 8) :
  abs ((a + a) / k - (a * a) / k) = 496 :=
by
  sorry

end positive_difference_l631_631681


namespace circles_tangent_proof_l631_631252

theorem circles_tangent_proof
  (P : Type) [MetricSpace P]
  (A B C D : P)
  (k1 k2 : Set P) -- k1 and k2 are circles
  (hAB : A ≠ B)
  (hAC : IsTangent k1 A C)
  (hBD : IsTangent k2 B D)
  (hIntersect : IsIntersect k1 k2 A B)
  (BD BC AC AD : ℝ) -- Define segment lengths as real numbers
  (hBD_def : BD = dist B D)
  (hBC_def : BC = dist B C)
  (hAC_def : AC = dist A C)
  (hAD_def : AD = dist A D) :
  BD^2 * BC = AC^2 * AD :=
sorry

end circles_tangent_proof_l631_631252


namespace number_contains_digit_7_l631_631870

noncomputable def contains_digit (d n : ℕ) : Prop :=
  ∃ k, n / 10^k % 10 = d

noncomputable def count_numbers_with_digit (d bound : ℕ) : ℕ :=
  (finset.range (bound + 1)).filter (λ n, contains_digit d n).card

theorem number_contains_digit_7 : count_numbers_with_digit 7 800 = 152 := 
sorry

end number_contains_digit_7_l631_631870


namespace recurring_decimal_to_fraction_l631_631797

noncomputable def recurring_decimal_fraction_form (q : ℝ) (hq : 0 < q ∧ q < 1) (a1 : ℝ) : ℝ :=
  a1 / (1 - q)

theorem recurring_decimal_to_fraction : 
  let q := 0.01
  let a1 := 0.25
  recurring_decimal_fraction_form q (by interval_cases; exact ⟨by linarith, by linarith⟩) a1 = 25 / 99 → 
  5 + 25 / 99 = 520 / 99 :=
by
  sorry

end recurring_decimal_to_fraction_l631_631797


namespace binomial_coefficient_largest_l631_631154

theorem binomial_coefficient_largest (n : ℕ) (x : ℝ) 
  (h : ∑ k in finset.range (n+1), nat.choose n k = 64) : 
  n = 6 → ∀ x, max_binomial_coefficient (2*x - (1/x^(1/2)))^n = nat.choose 6 3 := 
sorry

end binomial_coefficient_largest_l631_631154


namespace puppy_cost_l631_631762

variable (P : ℕ)  -- Cost of one puppy

theorem puppy_cost (P : ℕ) (kittens : ℕ) (cost_kitten : ℕ) (total_value : ℕ) :
  kittens = 4 → cost_kitten = 15 → total_value = 100 → 
  2 * P + kittens * cost_kitten = total_value → P = 20 :=
by sorry

end puppy_cost_l631_631762


namespace find_n_eq_l631_631196

def geom_series_sum_1 (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

def geom_series_sum_2 (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - (-r)^n) / (1 + r)

theorem find_n_eq (n : ℕ) (h : n ≥ 1) :
  geom_series_sum_1 989 (1 / 3) n = geom_series_sum_2 2744 (1 / 3) n → n = 2 :=
sorry

end find_n_eq_l631_631196


namespace greatest_integer_less_than_M_over_100_l631_631094

theorem greatest_integer_less_than_M_over_100
  (h : (1/(Nat.factorial 3 * Nat.factorial 18) + 1/(Nat.factorial 4 * Nat.factorial 17) + 
        1/(Nat.factorial 5 * Nat.factorial 16) + 1/(Nat.factorial 6 * Nat.factorial 15) + 
        1/(Nat.factorial 7 * Nat.factorial 14) + 1/(Nat.factorial 8 * Nat.factorial 13) + 
        1/(Nat.factorial 9 * Nat.factorial 12) + 1/(Nat.factorial 10 * Nat.factorial 11) = 
        1/(Nat.factorial 2 * Nat.factorial 19) * (M : ℚ))) :
  ⌊M / 100⌋ = 499 :=
by
  sorry

end greatest_integer_less_than_M_over_100_l631_631094


namespace hyperbola_standard_equation_and_k_range_l631_631075

variables (e : ℝ) (a c k : ℝ)
variables (x y : ℝ)

-- Conditions of the problem
def hyperbola_conditions : Prop := 
  e = 2 * Real.sqrt 3 / 3 ∧ 
  (a^2 / c) = 3 / 2 ∧ 
  e = c / a

-- Standard equation of the hyperbola
def hyperbola_equation : Prop := 
  ∀ x y, (x^2 / 3) - y^2 = 1

-- Conditions for line intersection and vector dot product
def line_condition (k : ℝ) : Prop := 
  let y := k * x + Real.sqrt 2 in
  let discriminant := (6 * Real.sqrt 2 * k)^2 + 4 * 3 * (1 - 3 * k^2) in
  (discriminant > 0 ∧ 
   let x1 := (6 * Real.sqrt 2 * k / (1 - 3 * k^2)) in
   let x2 := (-6 * Real.sqrt 2 * k / (1 - 3 * k^2)) in
   (x1 * x2 + (kx * x1 + Real.sqrt 2) * (kx * x2 + Real.sqrt 2) > 2))

-- The range of values for k
def k_range (k : ℝ) : Prop := 
  k ∈ Set.Ioo (-1 : ℝ) (-Real.sqrt 3 / 3) ∪ 
  Set.Ioo (Real.sqrt 3 / 3) (1 : ℝ)

-- Proof statement
theorem hyperbola_standard_equation_and_k_range :
  hyperbola_conditions → 
  hyperbola_equation →
  (∀ k, line_condition k → k_range k) := 
sorry

end hyperbola_standard_equation_and_k_range_l631_631075


namespace number_of_zeros_in_interval_l631_631384

-- Define the problem's conditions and statement in Lean 4

noncomputable def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

noncomputable def monotonic_on_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (∀ x y ∈ Icc (0 : ℝ) a, x ≤ y → f x ≤ f y) ∨ (∀ x y ∈ Icc 0 a, x ≤ y → f y ≤ f x)

theorem number_of_zeros_in_interval (f : ℝ → ℝ) (a : ℝ) (h_pos : a > 0)
  (h_even : even_function f)
  (h_mono : monotonic_on_interval f a)
  (h_zero_cross : f 0 * f a < 0) :
  ∃! x ∈ Icc (0 : ℝ) a, f x = 0 ∧ ∃! y ∈ Icc (-a) (0 : ℝ), f y = 0 :=
sorry

end number_of_zeros_in_interval_l631_631384


namespace segment_length_R_R_l631_631292

theorem segment_length_R_R' :
  let R := (-4, 1)
  let R' := (-4, -1)
  let distance : ℝ := Real.sqrt ((R'.1 - R.1)^2 + (R'.2 - R.2)^2)
  distance = 2 :=
by
  sorry

end segment_length_R_R_l631_631292


namespace total_distance_walked_l631_631052

theorem total_distance_walked (s e t : ℕ) (h_susan : s = 9) (h_erin : e = s - 3) : t = s + e → t = 15 :=
by {
  intro h,
  rw [h_susan, h_erin] at h,
  simp at h,
  exact h,
  sorry
}

end total_distance_walked_l631_631052


namespace factor_count_of_x15_sub_x_l631_631406

-- Define the polynomial
def poly := λ x : ℤ, x^15 - x

-- Our main proposition to prove
theorem factor_count_of_x15_sub_x : 
  (∃ (factors : List (ℤ → ℤ)), (∀ x, (poly x) = List.prod (List.map (λ f, f x) factors)) ∧ factors.length = 5) :=
sorry

end factor_count_of_x15_sub_x_l631_631406


namespace unique_prime_digit_l631_631268

def is_digit (n : ℕ) : Prop := n < 10

noncomputable def prime_digit_number (A : ℕ) : ℕ := 202100 + A

theorem unique_prime_digit :
  ∃! A, is_digit A ∧ Nat.Prime (prime_digit_number A) :=
begin
  sorry
end

end unique_prime_digit_l631_631268


namespace length_of_goods_train_l631_631351

theorem length_of_goods_train (speed_km_hr : ℕ) (platform_length_m : ℕ) (time_sec : ℕ) 
    (h1 : speed_km_hr = 108) (h2 : platform_length_m = 150) (h3 : time_sec = 30) :
    let speed_m_s := (speed_km_hr * 1000) / 3600 in
    let distance_covered := speed_m_s * time_sec in
    let train_length := distance_covered - platform_length_m in
    train_length = 750 := 
by
  sorry

end length_of_goods_train_l631_631351


namespace count_numbers_containing_7_l631_631883

-- Define a predicate to check if a number contains the digit 7.
def contains_digit_7 (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∈ n.digits 10 ∧ d = 7

-- Define the set of numbers from 1 to 800.
def numbers_from_1_to_800 : set ℕ := {n | 1 ≤ n ∧ n ≤ 800}

-- Define the set of numbers from 1 to 800 that contain the digit 7.
def numbers_containing_7 : set ℕ := {n | n ∈ numbers_from_1_to_800 ∧ contains_digit_7 n}

-- The theorem to prove the required count.
theorem count_numbers_containing_7 :
  (numbers_containing_7.to_finset.card = 62) :=
sorry

end count_numbers_containing_7_l631_631883


namespace binary_remainder_div_8_l631_631305

theorem binary_remainder_div_8 (n : ℕ) (h : n = 0b101100110011) : n % 8 = 3 :=
by sorry

end binary_remainder_div_8_l631_631305


namespace radius_of_isosceles_tangent_circle_l631_631750

noncomputable def R : ℝ := 2 * Real.sqrt 3

variables (x : ℝ) (AB AC BD AD DC r : ℝ)

def is_isosceles (AB BC : ℝ) : Prop := AB = BC
def is_tangent (r : ℝ) (x : ℝ) : Prop := r = 2.4 * x

theorem radius_of_isosceles_tangent_circle
  (h_isosceles: is_isosceles AB BC)
  (h_area: 1/2 * AC * BD = 25)
  (h_height_ratio: BD / AC = 3 / 8)
  (h_AD_DC: AD = DC)
  (h_AC: AC = 8 * x)
  (h_BD: BD = 3 * x)
  (h_radius: is_tangent r x):
  r = R :=
sorry

end radius_of_isosceles_tangent_circle_l631_631750


namespace chairs_carried_per_trip_l631_631553

theorem chairs_carried_per_trip (x : ℕ) (friends : ℕ) (trips : ℕ) (total_chairs : ℕ) 
  (h1 : friends = 4) (h2 : trips = 10) (h3 : total_chairs = 250) 
  (h4 : 5 * (trips * x) = total_chairs) : x = 5 :=
by sorry

end chairs_carried_per_trip_l631_631553


namespace numbers_with_7_in_1_to_800_l631_631913

theorem numbers_with_7_in_1_to_800 : 
  (card { n ∈ finset.range (800 + 1) | ∃ d ∈ n.digits 10, d = 7 }) = 152 := 
sorry

end numbers_with_7_in_1_to_800_l631_631913


namespace limit_derivative_of_f_l631_631448

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem limit_derivative_of_f (a : ℝ) (h : a ≠ 0) : 
  filter.tendsto (λ x, (f x - f a) / (x - a)) (nhds a) (nhds (-1 / (a^2))) :=
by
  sorry

end limit_derivative_of_f_l631_631448


namespace first_train_travels_more_l631_631653

-- Define the conditions
def velocity_first_train := 50 -- speed of the first train in km/hr
def velocity_second_train := 40 -- speed of the second train in km/hr
def distance_between_P_and_Q := 900 -- distance between P and Q in km

-- Problem statement
theorem first_train_travels_more :
  ∃ t : ℝ, (velocity_first_train * t + velocity_second_train * t = distance_between_P_and_Q)
          → (velocity_first_train * t - velocity_second_train * t = 100) :=
by sorry

end first_train_travels_more_l631_631653


namespace solution_set_l631_631924

variables {f : ℝ → ℝ}

-- Condition 1: f is an odd function
def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)

-- Condition 2: f is decreasing on (-∞, 0]
def decreasing_on_neg (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → x ≤ 0 → y ≤ 0 → f x ≥ f y

-- Hypotheses
variable (odd_f : odd_function f)
variable (decreasing_f : decreasing_on_neg f)

-- The theorem to prove
theorem solution_set (x : ℝ) : f (Real.log x) < -f 1 ↔ x ∈ Ioi (Real.exp (-1)) :=
  sorry

end solution_set_l631_631924


namespace range_of_a_l631_631454

theorem range_of_a (f : ℝ → ℝ) (h1 : ∀ x, f (x - 3) = f (3 - (x - 3))) (h2 : ∀ x, 0 ≤ x → f x = x^2 + 2 * x) :
  {a : ℝ | f (2 - a^2) > f a} = {a | -2 < a ∧ a < 1} :=
by
  sorry

end range_of_a_l631_631454


namespace minimum_relationships_condition_l631_631984

open BigOperators

-- Let k be an integer such that 1 ≤ k and k divides 80.
variables (k : ℕ) (h1 : 1 ≤ k) (h2 : k ∣ 80)

-- Let n be the number of dinosaurs
def n := 81

-- There are exactly k different types of relationships
def relationship_types := k

-- Each dinosaur has a specific type of relationship with exactly 80/k other dinosaurs
def num_relationships_per_type := 80 / k

-- There are at least 69120 unstable triplets
def min_unstable_triplets := 69120

-- Define unstable_triplets_satisfying_conditions
def unstable_triplets_satisfying_conditions := 
  n * (n - 1)^2 * (1 - 1 / k)

-- The condition that the number of triplets is greater than or equal to 6T
theorem minimum_relationships_condition (h3 : unstable_triplets_satisfying_conditions k h1 h2 ≥ 6 * min_unstable_triplets) : 
  k ≥ 5 := 
sorry

end minimum_relationships_condition_l631_631984


namespace sqrt_diff_inequality_l631_631920

theorem sqrt_diff_inequality (c : ℝ) (hc : c > 1) : 
  sqrt c - sqrt (c - 1) > sqrt (c + 1) - sqrt c :=
sorry

end sqrt_diff_inequality_l631_631920


namespace determine_wins_l631_631339

def wins (x y : ℕ) : Prop := x + y = 12 ∧ 3 * x + y = 28

theorem determine_wins (x y : ℕ) (h : wins x y) : x = 8 :=
by
  cases h with h1 h2
  sorry

end determine_wins_l631_631339


namespace faculty_reduction_l631_631768

def original : ℝ := 224.14
def reduction_percent : ℝ := 0.13

def reduced_faculty : ℕ := 
  let reduced_amount := reduction_percent * original
  let rounded_reduced_amount := reduced_amount.round
  let final_faculty := (original - rounded_reduced_amount).round
  final_faculty.to_nat -- converting the integer round result to natural number

theorem faculty_reduction : reduced_faculty = 195 := 
by
  sorry

end faculty_reduction_l631_631768


namespace expression_value_l631_631504

theorem expression_value
  (x y : ℝ) 
  (h : x - 3 * y = 4) : 
  (x - 3 * y)^2 + 2 * x - 6 * y - 10 = 14 :=
by
  sorry

end expression_value_l631_631504


namespace series_sum_a_plus_b_l631_631441

theorem series_sum_a_plus_b (m : ℕ) (hm : 0 < m) :
  ∃ a b : ℕ, 
    (a + b = 7) ∧ 
    (∑ k in (Finset.range (m-1)).filter (λ k, k ≠ m) ∪ (Finset.range (m+1)).filter (λ k, k ≠ m), 
       1 / ((k+m) * (k-m)) = a / (b * m ^ 2)) :=
begin
  sorry
end

end series_sum_a_plus_b_l631_631441


namespace jam_fraction_eaten_l631_631547

theorem jam_fraction_eaten (x : ℚ) : 
  (1 - x) - (1 / 7) * (1 - x) = 4 / 7 ↔ x = 1 / 21 :=
by 
  split
  all_goals {
    intro h,
    sorry
  }

end jam_fraction_eaten_l631_631547


namespace inequality_solution_set_l631_631808

noncomputable def g (x : ℝ) : ℝ := (3 * x - 8) * (x - 4) / (x + 1)

theorem inequality_solution_set :
  {x : ℝ | g x ≥ 0} = Ioo (-∞) (-1) ∪ Icc (8 / 3) 4 ∪ Ioo 4 ∞ :=
by sorry

end inequality_solution_set_l631_631808


namespace factory_production_schedule_l631_631649

noncomputable def production_equation (x : ℝ) : Prop :=
  (1000 / x) - (1000 / (1.2 * x)) = 2

theorem factory_production_schedule (x : ℝ) (hx : x ≠ 0) : production_equation x := 
by 
  -- Assumptions based on conditions:
  -- Factory plans to produce total of 1000 sets of protective clothing.
  -- Actual production is 20% more than planned.
  -- Task completed 2 days ahead of original schedule.
  -- We need to show: (1000 / x) - (1000 / (1.2 * x)) = 2
  sorry

end factory_production_schedule_l631_631649


namespace factory_output_increase_l631_631619

theorem factory_output_increase (x : ℝ) (h : (1 + x / 100) ^ 4 = 4) : x = 41.4 :=
by
  -- Given (1 + x / 100) ^ 4 = 4
  sorry

end factory_output_increase_l631_631619


namespace sections_have_equal_areas_and_perimeters_l631_631344

-- Given conditions
def radius (r : ℝ) : Prop := r > 0
def circle_area (r : ℝ) : ℝ := r^2 * Real.pi
def section_area (r : ℝ) : ℝ := (r^2 * Real.pi) / 3
def circle_perimeter (r : ℝ) : ℝ := 2 * r * Real.pi
def section_perimeter (r : ℝ) : ℝ := 2 * r * Real.pi

theorem sections_have_equal_areas_and_perimeters (r : ℝ) (h : radius r) :
  ∀ i : ℕ, i ∈ {0, 1, 2} → 
  (section_area r = circle_area r / 3) ∧ 
  (section_perimeter r = circle_perimeter r) := 
sorry

end sections_have_equal_areas_and_perimeters_l631_631344


namespace nine_skiers_four_overtakes_impossible_l631_631586

theorem nine_skiers_four_overtakes_impossible :
  ∀ (skiers : Fin 9 → ℝ),  -- skiers are represented by their speeds
  (∀ i j, i < j → skiers i ≤ skiers j) →  -- skiers start sequentially and maintain constant speeds
  ¬(∀ i, (∃ a b : Fin 9, (a ≠ i ∧ b ≠ i ∧ (skiers a < skiers i ∧ skiers i < skiers b ∨ skiers b < skiers i ∧ skiers i < skiers a)))) →
    false := 
by
  sorry

end nine_skiers_four_overtakes_impossible_l631_631586


namespace segment_DC_l631_631169

noncomputable def angle_EDC (A B C D E : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] 
  (AB AC AD DE : ℝ) (BAC ADE : ℝ) (isosceles_ABC isosceles_DAE : Prop) : Prop :=
isosceles_ABC ∧
isosceles_DAE ∧
AB = AC ∧ AB = AD ∧ AD = DE ∧
BAC = 36 ∧ ADE = 36 →
angle E D C = 36

theorem segment_DC (A B C D E : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] 
  (BC AC : ℝ) (isosceles_ABC isosceles_DAE congruent_ABC_ADC : Prop) : Prop :=
isosceles_ABC ∧
isosceles_DAE ∧
congruent_ABC_ADC ∧
AB = AC ∧ AB = AD ∧ AD = DE ∧
BC = 2 →
DC = 2

noncomputable def segment_AC (A B C D E : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] 
  (BC : ℝ) (isosceles_ABC isosceles_DAE : Prop) : Prop :=
isosceles_ABC ∧
isosceles_DAE ∧
AB = AC ∧ AB = AD ∧ AD = DE ∧
BC = 2 →
AC = (sqrt 5 + 1)

#align  -- To ensure the Lean 4 alignment checking mechanism

end segment_DC_l631_631169


namespace final_share_of_bill_l631_631273

theorem final_share_of_bill : 
  ∀ (b : ℝ) (n : ℕ) (tip_percent : ℝ), 
    n = 8 → 
    b = 211.00 → 
    tip_percent = 0.15 → 
    let tip := tip_percent * b in
    let final_bill := b + tip in
    let share := final_bill / n in
    share.round(2) = 30.33 :=
by
  intros b n tip_percent h1 h2 h3 tip final_bill share
  sorry

end final_share_of_bill_l631_631273


namespace turn_over_card_3_disprove_statement_l631_631062

-- Definitions reflecting the conditions
structure Card where
  letter_side : Option Char -- None if number shown, Some l if letter l shown
  number_side : Option ℕ -- None if letter shown, Some n if number n shown

def vowel (c : Char) : Prop := c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U'

def even (n : ℕ) : Prop := n % 2 = 0

-- Jane's statement as a logical implication
def janes_statement (card : Card) : Prop :=
  (vowel c → even n) For all cards (c == letter & n == number on the same card)

-- The cards on the table
def card_p := Card.mk (Some 'P') none
def card_q := Card.mk (Some 'Q') none
def card_3 := Card.mk none (Some 3)
def card_4 := Card.mk none (Some 4)
def card_6 := Card.mk none (Some 6)

-- Express the main theorem
theorem turn_over_card_3_disprove_statement :
  ¬janes_statement card_3 → (card_3.letter_some ↔ c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U' ∧ c.any_odd n (even))) ↔ 
  ∃ (c : Char) (n : Option ℕ), (card_3.letter_side = Some c ∧ vowel c ∧ card_3.number_side = Some n ∧ ¬even n) := 
sorry

end turn_over_card_3_disprove_statement_l631_631062


namespace find_middle_number_l631_631051

theorem find_middle_number (a b c d x e f g : ℝ) 
  (h1 : (a + b + c + d + x + e + f + g) / 8 = 7)
  (h2 : (a + b + c + d + x) / 5 = 6)
  (h3 : (x + e + f + g + d) / 5 = 9) :
  x = 9.5 := 
by 
  sorry

end find_middle_number_l631_631051


namespace count_numbers_containing_7_l631_631881

-- Define a predicate to check if a number contains the digit 7.
def contains_digit_7 (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∈ n.digits 10 ∧ d = 7

-- Define the set of numbers from 1 to 800.
def numbers_from_1_to_800 : set ℕ := {n | 1 ≤ n ∧ n ≤ 800}

-- Define the set of numbers from 1 to 800 that contain the digit 7.
def numbers_containing_7 : set ℕ := {n | n ∈ numbers_from_1_to_800 ∧ contains_digit_7 n}

-- The theorem to prove the required count.
theorem count_numbers_containing_7 :
  (numbers_containing_7.to_finset.card = 62) :=
sorry

end count_numbers_containing_7_l631_631881


namespace perfect_squares_in_sequence_l631_631362

-- Definitions based on the problem's conditions
def sequence (a b : Nat) : List Nat :=
  let rec aux (l : List Nat) (n : Nat) : List Nat :=
    match n with
    | 0       => l
    | n + 1   =>
      match l with
      | []            => []
      | [_]           => []
      | (x :: y :: l₁) => aux ((x * y) :: x :: y :: l₁) n
  aux [b, a] 298

-- Statement of the theorem
theorem perfect_squares_in_sequence (a b : Nat) :
  let s := sequence a b in
  (∀ x ∈ s, x = 0 → (x = 0 ∨ x = 100 ∨ x = 300)) :=
sorry

end perfect_squares_in_sequence_l631_631362


namespace sample_size_l631_631932

theorem sample_size (k n : ℕ) (r : 2 * k + 3 * k + 5 * k = 10 * k) (h : 3 * k = 12) : n = 40 :=
by {
    -- here, we will provide a proof to demonstrate that n = 40 given the conditions
    sorry
}

end sample_size_l631_631932


namespace positive_difference_l631_631688

theorem positive_difference : 496 = abs ((64 + 64) / 8 - (64 * 64) / 8) := by
  have h1 : 8^2 = 64 := rfl
  have h2 : 64 + 64 = 128 := rfl
  have h3 : (128 : ℕ) / 8 = 16 := rfl
  have h4 : 64 * 64 = 4096 := rfl
  have h5 : (4096 : ℕ) / 8 = 512 := rfl
  have h6 : 512 - 16 = 496 := rfl
  sorry

end positive_difference_l631_631688


namespace bound_f_n_l631_631200

noncomputable def f : ℕ → ℕ := sorry -- Assuming the existence of the function f

axiom f_increasing : ∀ n m : ℕ, n < m → f(n) < f(m)
axiom f_property : ∀ n : ℕ, f (f n) = k * n

theorem bound_f_n (k : ℕ) (h_k : k > 0) : ∀ n : ℕ, 
  (2 * k : ℚ) / (k + 1) * n ≤ f n ∧ f n ≤ ((k + 1) / 2 : ℚ) * n :=
begin
  sorry
end

end bound_f_n_l631_631200


namespace number_of_triangles_for_second_star_l631_631063

theorem number_of_triangles_for_second_star (a b : ℝ) (h₁ : a + b + 90 = 180) (h₂ : 5 * (360 / 5) = 360) :
  360 / (180 - 90 - (360 / 5)) = 20 :=
by
  sorry

end number_of_triangles_for_second_star_l631_631063


namespace total_volume_is_correct_l631_631025

theorem total_volume_is_correct :
  let carl_side := 3
  let carl_count := 3
  let kate_side := 1.5
  let kate_count := 4
  let carl_volume := carl_count * carl_side ^ 3
  let kate_volume := kate_count * kate_side ^ 3
  carl_volume + kate_volume = 94.5 :=
by
  sorry

end total_volume_is_correct_l631_631025


namespace trig_arith_calculation_l631_631742

theorem trig_arith_calculation : 
  3 * Real.tan (π/4) - (1/3)^(-1:ℤ) + (Real.sin (π/6) - 2022)^0 + (Real.cos (π/6) - Real.sqrt 3 / 2).abs = 1 := 
by sorry

end trig_arith_calculation_l631_631742


namespace count_numbers_containing_7_l631_631880

-- Define a predicate to check if a number contains the digit 7.
def contains_digit_7 (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∈ n.digits 10 ∧ d = 7

-- Define the set of numbers from 1 to 800.
def numbers_from_1_to_800 : set ℕ := {n | 1 ≤ n ∧ n ≤ 800}

-- Define the set of numbers from 1 to 800 that contain the digit 7.
def numbers_containing_7 : set ℕ := {n | n ∈ numbers_from_1_to_800 ∧ contains_digit_7 n}

-- The theorem to prove the required count.
theorem count_numbers_containing_7 :
  (numbers_containing_7.to_finset.card = 62) :=
sorry

end count_numbers_containing_7_l631_631880


namespace number_of_numbers_with_digit_seven_l631_631897

-- Define what it means to contain digit 7
def contains_digit_seven (n : ℕ) : Prop :=
  n.digits 10 ∈ [7]

-- Define the set of numbers from 1 to 800 containing at least one digit 7
def numbers_with_digit_seven : ℕ → Prop :=
  λ n, 1 ≤ n ∧ n ≤ 800 ∧ contains_digit_seven n

-- State the theorem
theorem number_of_numbers_with_digit_seven : (finset.filter numbers_with_digit_seven (finset.range 801)).card = 152 :=
sorry

end number_of_numbers_with_digit_seven_l631_631897


namespace find_n_l631_631993

open Real

noncomputable def F (n : ℕ) : ℝ := 2 ^ (2 ^ n) + 1

noncomputable def a (n : ℕ) : ℝ := log 4 (F n - 1)

noncomputable def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a (i + 1)

theorem find_n (n : ℕ) (h : 32 * S n = 63 * a n) : n = 6 := by
  sorry

end find_n_l631_631993


namespace find_E2_l631_631475

variable (a b c d : ℝ)

def E1 := a - b - c + d
def E2 := a + b - c - d

theorem find_E2 (h : (b - d)^2 = 4) : E2 = a + b - c - d :=
by
  -- We are given that (b - d)^2 = 4
  -- Thus, we need to show the value of E2
  sorry

end find_E2_l631_631475


namespace positive_difference_eq_496_l631_631706

theorem positive_difference_eq_496 : 
  let a := 8 ^ 2 in 
  (a + a) / 8 - (a * a) / 8 = 496 :=
by
  let a := 8^2
  have h1 : (a + a) / 8 = 16 := by sorry
  have h2 : (a * a) / 8 = 512 := by sorry
  show (a + a) / 8 - (a * a) / 8 = 496 from by
    calc
      (a + a) / 8 - (a * a) / 8
            = 16 - 512 : by rw [h1, h2]
        ... = -496 : by ring
        ... = 496 : by norm_num

end positive_difference_eq_496_l631_631706


namespace hundred_and_first_digit_l631_631492

/-- The repeating sequence for the decimal expansion of 7/26 -/
def repeating_sequence : List ℕ := [2, 6, 9, 2, 3, 0, 7, 6, 9]

/-- The length of the repeating sequence is 9 -/
def period : ℕ := 9

/-- The function to get the n-th digit of the decimal expansion of 7/26 -/
def nth_digit (n : ℕ) : ℕ :=
  repeating_sequence.get ⟨n % period, n.mod_lt (by norm_num)⟩

/-- The 101st digit of the decimal expansion of 7/26 is 6 -/
theorem hundred_and_first_digit : nth_digit 101 = 6 := by
  sorry

end hundred_and_first_digit_l631_631492


namespace greatest_integer_less_than_M_over_100_l631_631096

theorem greatest_integer_less_than_M_over_100
  (h : (1/(Nat.factorial 3 * Nat.factorial 18) + 1/(Nat.factorial 4 * Nat.factorial 17) + 
        1/(Nat.factorial 5 * Nat.factorial 16) + 1/(Nat.factorial 6 * Nat.factorial 15) + 
        1/(Nat.factorial 7 * Nat.factorial 14) + 1/(Nat.factorial 8 * Nat.factorial 13) + 
        1/(Nat.factorial 9 * Nat.factorial 12) + 1/(Nat.factorial 10 * Nat.factorial 11) = 
        1/(Nat.factorial 2 * Nat.factorial 19) * (M : ℚ))) :
  ⌊M / 100⌋ = 499 :=
by
  sorry

end greatest_integer_less_than_M_over_100_l631_631096


namespace gear_q_revolutions_per_minute_l631_631029

-- Define the constants and conditions
def revolutions_per_minute_p : ℕ := 10
def revolutions_per_minute_q : ℕ := sorry
def time_in_minutes : ℝ := 1.5
def extra_revolutions_q : ℕ := 45

-- Calculate the number of revolutions for gear p in 90 seconds
def revolutions_p_in_90_seconds := revolutions_per_minute_p * time_in_minutes

-- Condition that gear q makes exactly 45 more revolutions than gear p in 90 seconds
def revolutions_q_in_90_seconds := revolutions_p_in_90_seconds + extra_revolutions_q

-- Correct answer
def correct_answer : ℕ := 40

-- Prove that gear q makes 40 revolutions per minute
theorem gear_q_revolutions_per_minute : 
    revolutions_per_minute_q = correct_answer :=
sorry

end gear_q_revolutions_per_minute_l631_631029


namespace sum_of_isosceles_t_values_l631_631023

theorem sum_of_isosceles_t_values :
  ∑ t in {t | is_isosceles_triangle (30 : ℝ) (90 : ℝ) t ∧ 0 ≤ t ∧ t ≤ 360}, t = 690 :=
by sorry

-- Helper definition to check if the triangle is isosceles 
def is_isosceles_triangle (a b t : ℝ) : Prop :=
  let A := (cos a, sin a)
  let B := (cos b, sin b)
  let C := (cos t, sin t)
  dist A B = dist A C ∨ dist B C = dist B A ∨ dist C A = dist C B

-- Helper function to compute Euclidean distance between two points
def dist (P Q : ℝ × ℝ): ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

end sum_of_isosceles_t_values_l631_631023


namespace train_speed_clicks_equivalence_l631_631620

theorem train_speed_clicks_equivalence (v : ℝ) :
  let feet_per_mile := 5280
  let minutes_per_hour := 60
  let rails_40 := 40
  let rails_25 := 25
  let clicks_per_min_40 := (feet_per_mile * v / minutes_per_hour) / rails_40
  let clicks_per_min_25 := (feet_per_mile * v / minutes_per_hour) / rails_25
  let avg_clicks_per_min := (clicks_per_min_40 + clicks_per_min_25) / 2
  let interval := 0.0211 / 60
  clicks_per_min_40 = feet_per_mile * v / (rails_40 * minutes_per_hour) ∧
  clicks_per_min_25 = feet_per_mile * v / (rails_25 * minutes_per_hour) ∧
  avg_clicks_per_min = (feet_per_mile * v / (rails_40 * minutes_per_hour) + feet_per_mile * v / (rails_25 * minutes_per_hour)) / 2 ∧
  avg_clicks_per_min * interval ≈ v :=
  sorry

end train_speed_clicks_equivalence_l631_631620


namespace abc_area_l631_631284

def rectangle_area (length width : ℕ) : ℕ :=
  length * width

theorem abc_area :
  let smaller_side := 7
  let longer_side := 2 * smaller_side
  let length := 3 * longer_side -- since there are 3 identical rectangles placed side by side
  let width := smaller_side
  rectangle_area length width = 294 :=
by
  sorry

end abc_area_l631_631284


namespace count_numbers_containing_7_l631_631884

-- Define a predicate to check if a number contains the digit 7.
def contains_digit_7 (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∈ n.digits 10 ∧ d = 7

-- Define the set of numbers from 1 to 800.
def numbers_from_1_to_800 : set ℕ := {n | 1 ≤ n ∧ n ≤ 800}

-- Define the set of numbers from 1 to 800 that contain the digit 7.
def numbers_containing_7 : set ℕ := {n | n ∈ numbers_from_1_to_800 ∧ contains_digit_7 n}

-- The theorem to prove the required count.
theorem count_numbers_containing_7 :
  (numbers_containing_7.to_finset.card = 62) :=
sorry

end count_numbers_containing_7_l631_631884


namespace number_of_numbers_with_digit_seven_l631_631900

-- Define what it means to contain digit 7
def contains_digit_seven (n : ℕ) : Prop :=
  n.digits 10 ∈ [7]

-- Define the set of numbers from 1 to 800 containing at least one digit 7
def numbers_with_digit_seven : ℕ → Prop :=
  λ n, 1 ≤ n ∧ n ≤ 800 ∧ contains_digit_seven n

-- State the theorem
theorem number_of_numbers_with_digit_seven : (finset.filter numbers_with_digit_seven (finset.range 801)).card = 152 :=
sorry

end number_of_numbers_with_digit_seven_l631_631900


namespace value_of_d_l631_631515

theorem value_of_d (d : ℝ) (h : ∀ x : ℝ, 3 * (5 + d * x) = 15 * x + 15 → True) : d = 5 :=
sorry

end value_of_d_l631_631515


namespace max_value_expression_l631_631388

theorem max_value_expression : ∃ (c l h e r s g m b : ℕ),
  (c ≠ l ∧ c ≠ h ∧ c ≠ e ∧ c ≠ r ∧ c ≠ s ∧ c ≠ g ∧ c ≠ m ∧ c ≠ b ∧
   l ≠ h ∧ l ≠ e ∧ l ≠ r ∧ l ≠ s ∧ l ≠ g ∧ l ≠ m ∧ l ≠ b ∧
   h ≠ e ∧ h ≠ r ∧ h ≠ s ∧ h ≠ g ∧ h ≠ m ∧ h ≠ b ∧
   e ≠ r ∧ e ≠ s ∧ e ≠ g ∧ e ≠ m ∧ e ≠ b ∧
   r ≠ s ∧ r ≠ g ∧ r ≠ m ∧ r ≠ b ∧
   s ≠ g ∧ s ≠ m ∧ s ≠ b ∧
   g ≠ m ∧ g ≠ b ∧
   m ≠ b ∧
   c ∈ {1,2,3,4,5,6,7,8,9} ∧ l ∈ {1,2,3,4,5,6,7,8,9} ∧
   h ∈ {1,2,3,4,5,6,7,8,9} ∧ e ∈ {1,2,3,4,5,6,7,8,9} ∧
   r ∈ {1,2,3,4,5,6,7,8,9} ∧ s ∈ {1,2,3,4,5,6,7,8,9} ∧
   g ∈ {1,2,3,4,5,6,7,8,9} ∧ m ∈ {1,2,3,4,5,6,7,8,9} ∧
   b ∈ {1,2,3,4,5,6,7,8,9}) ∧
  (2 * 1 + 53 * 6 + 987 * 6 = 6242) :=
by {
  use [2, 1, 5, 3, 4, 9, 8, 7, 6],
  repeat {split; try {dec_trivial}},
  exact rfl,
}

end max_value_expression_l631_631388


namespace triangle_probability_l631_631524

theorem triangle_probability :
  let points : List (ℝ × ℝ) := [(0,0), (2,0), (1,1), (0,2), (2,2)]
  let total_combinations := Nat.choose 5 3
  let collinear_combinations := 2
  let triangle_combinations := total_combinations - collinear_combinations
  let probability := (triangle_combinations : ℚ) / total_combinations
  probability = 4/5 := 
by 
  sorry

end triangle_probability_l631_631524


namespace pencils_left_in_drawer_l631_631639

theorem pencils_left_in_drawer (p c t_p r_p t_c : ℕ) 
  (hp : p = 34) (hc : c = 49) (htp : t_p = 22) (hrp : r_p = 5) (htc : t_c = 11) : 
  p - (t_p - r_p) = 17 :=
by
  rw [hp, htp, hrp]
  exact rfl

end pencils_left_in_drawer_l631_631639


namespace last_piece_length_is_correct_l631_631185

def josh_last_piece_length : ℝ :=
  let initial_length := 100
  let first_cut := initial_length / 3
  let second_cut := first_cut / 2
  let third_cut := second_cut / 4
  let fourth_cut := (third_cut + 2) / 5
  let fifth_cut := 2 * fourth_cut
  let sixth_cut := fifth_cut / 6
  let sixth_cut_inches := sixth_cut * 12
  let seventh_cut := (sixth_cut_inches - 1) / 8
  seventh_cut

theorem last_piece_length_is_correct : 
  josh_last_piece_length = 0.49162495 :=
by
  sorry

end last_piece_length_is_correct_l631_631185


namespace count_numbers_with_digit_7_in_range_l631_631877

theorem count_numbers_with_digit_7_in_range : 
  let numbers_in_range := {n : ℕ | 1 ≤ n ∧ n ≤ 800}
      contains_digit_7 (n : ℕ) : Prop := n.digits 10.contains 7
  in (finset.filter (λ n, contains_digit_7 n) (finset.range 801)).card = 152 :=
by 
  let numbers_in_range := {n : ℕ | 1 ≤ n ∧ n ≤ 800}
  let contains_digit_7 (n : ℕ) : Prop := n.digits 10.contains 7
  have h := (finset.filter (λ n, contains_digit_7 n) (finset.range 801)).card
  sorry

end count_numbers_with_digit_7_in_range_l631_631877


namespace PQ_constant_length_l631_631266

def Circle (O : Point) (R : ℝ) : set Point :=
  { M | dist M O = R }

variables {O : Point} {R : ℝ} {M : Point} {AB CD : Line}

theorem PQ_constant_length
  (h_circle : M ∈ Circle O R)
  (h_diameters_AB : is_diameter_of O AB)
  (h_diameters_CD : is_diameter_of O CD)
  (h_perpendicular_MP : is_perpendicular_to M P AB)
  (h_perpendicular_MQ : is_perpendicular_to M Q CD)
  : dist P Q = R :=
sorry

end PQ_constant_length_l631_631266


namespace ellipse_perimeter_l631_631979

theorem ellipse_perimeter (P F1 F2 : ℝ × ℝ) :
  (P.1^2 / 169 + P.2^2 / 144 = 1) →
  let c := 2 * Real.sqrt 25 in
  (Real.dist P F1 + Real.dist P F2 = 26) →
  (Real.dist F1 F2 = c) →
  (Real.dist P F1 + Real.dist P F2 + Real.dist F1 F2 = 36) :=
by
  intros hP hPF hF
  sorry

end ellipse_perimeter_l631_631979


namespace problems_per_page_l631_631214

theorem problems_per_page (total_problems finished_problems remaining_pages problems_per_page : ℕ)
  (h1 : total_problems = 40)
  (h2 : finished_problems = 26)
  (h3 : remaining_pages = 2)
  (h4 : total_problems - finished_problems = 14)
  (h5 : 14 = remaining_pages * problems_per_page) :
  problems_per_page = 7 := 
by
  sorry

end problems_per_page_l631_631214


namespace find_range_and_value_l631_631843

noncomputable def quadratic_eq (m : ℝ) : Polynomial ℝ :=
  Polynomial.C (m^2 - m) * Polynomial.X^2 + Polynomial.C (-2 * m) * Polynomial.X + 1

theorem find_range_and_value (m : ℝ) (a : ℝ) (h₁ : 0 < m) (h₂ : m ≠ 1) (h₃ : m ∈ Set.Iic 2) (h₄ : Polynomial.eval a (quadratic_eq m) = 0) :
  2 * a^2 - 3 * a - 3 = (if a = (2 + real.sqrt 2) / 2 then (-6 + real.sqrt 2) / 2 else (-6 - real.sqrt 2) / 2) :=
sorry

end find_range_and_value_l631_631843


namespace length_of_body_diagonal_l631_631840

theorem length_of_body_diagonal (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 11)
  (h2 : 4 * (a + b + c) = 24) :
  (a^2 + b^2 + c^2).sqrt = 5 :=
by {
  -- proof to be filled
  sorry
}

end length_of_body_diagonal_l631_631840


namespace number_of_numbers_with_digit_7_from_1_to_800_eq_233_l631_631887

def contains_digit (n d : ℕ) : Prop :=
  ∃ k, 10 ^ k > 0 ∧ d = (n / 10 ^ k) % 10

def numbers_without_digit (n d : ℕ) : finset ℕ :=
  (finset.range n).filter (λ x, ¬ contains_digit x d)

def count_numbers_with_digit (n d : ℕ) : ℕ :=
  n - (numbers_without_digit n d).card

theorem number_of_numbers_with_digit_7_from_1_to_800_eq_233 :
  count_numbers_with_digit 800 7 = 233 :=
  sorry

end number_of_numbers_with_digit_7_from_1_to_800_eq_233_l631_631887


namespace positive_difference_l631_631678

theorem positive_difference (a k : ℕ) (h1 : a = 8^2) (h2 : k = 8) :
  abs ((a + a) / k - (a * a) / k) = 496 :=
by
  sorry

end positive_difference_l631_631678


namespace find_a_in_triangle_l631_631156

variable (a b c B : ℝ)

theorem find_a_in_triangle (h1 : b = Real.sqrt 3) (h2 : c = 3) (h3 : B = 30) :
    a = 2 * Real.sqrt 3 := by
  sorry

end find_a_in_triangle_l631_631156


namespace number_of_subsets_A_range_of_m_for_empty_intersection_l631_631069

def A := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) := {x | (m - 1) ≤ x ∧ x ≤ (2 * m + 1)}

theorem number_of_subsets_A (x : ℕ) (hx : 0 < x ∧ x ≤ 5) :
  fintype.card (settofinset { x | ∃ hx : ℕ, x ∈ A }) = 32 :=
by {
  -- We show that the number of subsets of {1, 2, 3, 4, 5}
  sorry
}

theorem range_of_m_for_empty_intersection (A : set ℝ) (B : ℝ → set ℝ) :
  (∀ x ∈ ℝ, (x ∈ A → x ∉ B m) ∧ (x ∈ B m → x ∉ A)) →
  (m < -3/2 ∨ m > 6) :=
by {
  -- We prove the range of m when A has no intersection with B
  sorry
}

end number_of_subsets_A_range_of_m_for_empty_intersection_l631_631069


namespace parabola_line_slope_l631_631857

theorem parabola_line_slope (y1 y2 x1 x2 : ℝ) (h1 : y1 ^ 2 = 6 * x1) (h2 : y2 ^ 2 = 6 * x2) 
    (midpoint_condition : (x1 + x2) / 2 = 2 ∧ (y1 + y2) / 2 = 2) :
  (y1 - y2) / (x1 - x2) = 3 / 2 :=
by
  -- here will be the actual proof using the given hypothesis
  sorry

end parabola_line_slope_l631_631857


namespace min_value_of_f_l631_631614

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem min_value_of_f (t : ℝ) (h : t > 0) : ∃ x ∈ Set.Icc 1 (t+1), f x = 0 :=
by
  use 1
  constructor
  { exact and.intro le_rfl (le_add_of_nonneg_right h.le), }
  { simp [f], }

end min_value_of_f_l631_631614


namespace ratio_of_expenditures_l631_631264

-- Let us define the conditions and rewrite the proof problem statement.
theorem ratio_of_expenditures
  (income_P1 income_P2 expenditure_P1 expenditure_P2 : ℝ)
  (H1 : income_P1 / income_P2 = 5 / 4)
  (H2 : income_P1 = 5000)
  (H3 : income_P1 - expenditure_P1 = 2000)
  (H4 : income_P2 - expenditure_P2 = 2000) :
  expenditure_P1 / expenditure_P2 = 3 / 2 :=
sorry

end ratio_of_expenditures_l631_631264


namespace tara_spent_more_l631_631599

def Num_ice_cream := 100
def Num_yoghurt := 35
def Cost_per_ice_cream := 12
def Cost_per_yoghurt := 3
def Discount_rate_ice_cream := 0.05
def Tax_rate_yoghurt := 0.08

def Total_cost_ice_cream := Num_ice_cream * Cost_per_ice_cream
def Total_cost_yoghurt := Num_yoghurt * Cost_per_yoghurt

def Discount_ice_cream := Total_cost_ice_cream * Discount_rate_ice_cream
def Tax_yoghurt := Total_cost_yoghurt * Tax_rate_yoghurt

def Total_cost_ice_cream_after_discount := Total_cost_ice_cream - Discount_ice_cream
def Total_cost_yoghurt_after_tax := Total_cost_yoghurt + Tax_yoghurt

theorem tara_spent_more : Total_cost_ice_cream_after_discount - Total_cost_yoghurt_after_tax = 1026.60 := by
  -- calculation steps are omitted
  sorry

end tara_spent_more_l631_631599


namespace only_n_equals_1_works_l631_631322

inductive Letter : Type
| A | B | C

def reduction_rule (x y : Letter) : Letter :=
  match x, y with
  | Letter.A, Letter.B | Letter.B, Letter.A => Letter.C
  | Letter.A, Letter.C | Letter.C, Letter.A => Letter.B
  | Letter.B, Letter.C | Letter.C, Letter.B => Letter.A
  | _, _ => x

def reduce_sequence : List Letter → Letter
| [a] => a
| x :: y :: xs => reduce_sequence (reduction_rule x y :: xs)
| [] => Letter.A -- This case is not supposed to be used

theorem only_n_equals_1_works (n : ℕ) (hp : 0 < n):
  (∀ (initial_sequence : List Letter),
    (initial_sequence.length = 3 * n) →
    (List.count Letter.A initial_sequence = n) →
    (List.count Letter.B initial_sequence = n) →
    (List.count Letter.C initial_sequence = n) →
    let final_letter := reduce_sequence initial_sequence
    final_letter = Letter.A ∨ final_letter = Letter.B ∨ final_letter = Letter.C) ↔ n = 1 :=
sorry

end only_n_equals_1_works_l631_631322


namespace average_height_of_four_people_l631_631277

theorem average_height_of_four_people (
  h1 h2 h3 h4 : ℕ
) (diff12 : h2 = h1 + 2)
  (diff23 : h3 = h2 + 2)
  (diff34 : h4 = h3 + 6)
  (h4_eq : h4 = 83) :
  (h1 + h2 + h3 + h4) / 4 = 77 :=
by sorry

end average_height_of_four_people_l631_631277


namespace terrell_weight_lifting_l631_631600

theorem terrell_weight_lifting (n : ℝ) : 
  (2 * 25 * 10 = 500) → (2 * 20 * n = 500) → n = 12.5 :=
by
  intros h1 h2
  sorry

end terrell_weight_lifting_l631_631600


namespace cubic_sum_bound_l631_631462

variable {n : ℕ}
variable {x : Fin n → ℝ}

theorem cubic_sum_bound (h1 : (∑ i, x i) = 0)
                        (h2 : (∑ i, (x i)^2) = n * (n - 1)) :
    |∑ i, (x i)^3| ≤ (n - 2) * (n - 1) * n := sorry

end cubic_sum_bound_l631_631462


namespace mandy_accepted_is_7_l631_631989

def mandy_accepted_schools 
  (total_schools : ℕ) 
  (fraction_applied : ℚ) 
  (fraction_accepted : ℚ) : ℕ :=
  let schools_applied := (fraction_applied * total_schools).toNat in
  let schools_accepted := (fraction_accepted * schools_applied).toNat in
  schools_accepted

theorem mandy_accepted_is_7 : 
  mandy_accepted_schools 42 (1/3 : ℚ) (1/2 : ℚ) = 7 :=
by
  sorry

end mandy_accepted_is_7_l631_631989


namespace det_S_is_one_l631_631981

open Matrix Complex

def rotation_matrix (θ : Real) : Matrix (Fin 2) (Fin 2) Real :=
  ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]

-- Given conditions
def S : Matrix (Fin 2) (Fin 2) Real := rotation_matrix (75 * Real.pi / 180)

-- Theorem statement
theorem det_S_is_one : Matrix.det S = 1 := by
  sorry

end det_S_is_one_l631_631981


namespace closest_approximation_l631_631430

noncomputable def x := 
    ((69.28 * 0.004)^3 * Real.sin(Real.pi / 3)) / 
    ((0.03^2) * Real.log 0.58 * Real.cos(Real.pi / 4))

theorem closest_approximation : abs(x + 37.644) < 0.001 := by
  sorry

end closest_approximation_l631_631430


namespace perfect_square_problem_l631_631089

-- Define the given conditions and question
theorem perfect_square_problem 
  (a b c : ℕ) 
  (h_pos: a > 0 ∧ b > 0 ∧ c > 0)
  (h_cond: 0 < a^2 + b^2 - a * b * c ∧ a^2 + b^2 - a * b * c ≤ c + 1) : 
  ∃ k : ℕ, k^2 = a^2 + b^2 - a * b * c := 
sorry

end perfect_square_problem_l631_631089


namespace point_on_DE_l631_631541

theorem point_on_DE (ABC : Triangle)
  (AB AC BC : ℝ)
  (ha hb hc : ℝ)
  (P : Point)
  (da db dc : ℝ) :
  AB = AC ∧ AB ≠ BC →
  altitudes ABC = (ha, hb, hc) →
  is_interior_point ABC P →
  distances_from_point_to_sides P ABC = (da, db, dc) →
  da + db + dc = (ha + hb + hc) / 3 →
  ∃ (D E : Point), is_centroid ABC G ∧ 
  is_parallel (line_through G D) (side BC) ∧ 
  is_on_line_segment P D E :=
sorry

end point_on_DE_l631_631541


namespace g_neither_even_nor_odd_l631_631962

def g (x : ℝ) : ℝ := ⌊x + 0.5⌋ + 3 / 2

theorem g_neither_even_nor_odd : ¬(∀ x, g (-x) = g x) ∧ ¬(∀ x, g (-x) = -g x) := 
by 
  -- The proof is omitted
  sorry

end g_neither_even_nor_odd_l631_631962


namespace product_of_numbers_l631_631635

-- Definitions of the conditions
variables (x y : ℝ)

-- The conditions themselves
def cond1 : Prop := x + y = 20
def cond2 : Prop := x^2 + y^2 = 200

-- Statement of the proof problem
theorem product_of_numbers (h1 : cond1 x y) (h2 : cond2 x y) : x * y = 100 :=
sorry

end product_of_numbers_l631_631635


namespace find_lambda_l631_631919

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Definitions according to the problem conditions
variables (A B P : V) (λ : ℝ)

-- Conditions
def condition1 : Prop := (A - P) = (1 / 3) • (P - B)
def condition2 : Prop := (A - B) = λ • (B - P)

-- Theorem statement that proves the question == answer
theorem find_lambda (h1 : condition1 A B P) (h2 : condition2 A B P λ) : λ = - (4 / 3) :=
sorry

end find_lambda_l631_631919


namespace part1_part2_l631_631120

-- Define the conditions and claims
theorem part1 (a : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = |x - a|)
  (h_sol : ∀ x, f (2 * x) ≤ 4 ↔ 0 ≤ x ∧ x ≤ 4) : a = 4 :=
by
  sorry

theorem part2 (a : ℝ) (m : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = |x - a|)
  (h_empty : ∀ x, ¬ (f x + f (x + m) < 2)) : m ≥ 2 ∨ m ≤ -2 :=
by
  sorry

end part1_part2_l631_631120


namespace quadratic_functions_count_correct_even_functions_count_correct_l631_631066

def num_coefficients := 4
def valid_coefficients := [-1, 0, 1, 2]

def count_quadratic_functions : ℕ :=
  num_coefficients * num_coefficients * (num_coefficients - 1)

def count_even_functions : ℕ :=
  (num_coefficients - 1) * (num_coefficients - 2)

def total_quad_functions_correct : Prop := count_quadratic_functions = 18
def total_even_functions_correct : Prop := count_even_functions = 6

theorem quadratic_functions_count_correct : total_quad_functions_correct :=
by sorry

theorem even_functions_count_correct : total_even_functions_correct :=
by sorry

end quadratic_functions_count_correct_even_functions_count_correct_l631_631066


namespace range_of_a_l631_631483

-- Defining the propositions
def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) (a : ℝ) : Prop := x ≤ a

-- Main theorem statement
theorem range_of_a (a : ℝ) : (¬(∃ x, p x) → ¬(∃ x, q x a)) → a < -3 :=
by
  sorry

end range_of_a_l631_631483


namespace math_problem_1_math_problem_2_math_problem_3_l631_631832

-- Definitions of the given conditions
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 5
def point_A : ℝ × ℝ := (4, 3)
def tangent_condition (a b : ℝ) (P Q : ℝ × ℝ) : Prop := dist P Q = dist P point_A

-- Definition of the proof problems
theorem math_problem_1 (a b : ℝ) : 
  (∀ x y px py qx qy, circle_O x y ∧ tangent_condition a b (px, py) (qx, qy) → 4 * a + 3 * b = 15) := sorry

theorem math_problem_2 (a b : ℝ) : 
  (∀ x y px py qx qy, circle_O x y ∧ tangent_condition a b (px, py) (qx, qy) → dist (a, b) point_A = 4) := sorry

theorem math_problem_3 (a b : ℝ) : 
  (∃ (x y r : ℝ), circle_O x y ∧ ∃ P : ℝ × ℝ,
    tangent_condition a b P (x, y) ∧ 
    (dist (x, y) (3 - sqrt 5)) ∧
    (circle_O (x - 12/5) (y - 9/5) ∧ x * x + y * y = (3 - sqrt 5)^2)) := sorry

end math_problem_1_math_problem_2_math_problem_3_l631_631832


namespace hexagon_y_coordinate_D_l631_631134

-- Definitions of points
structure Point where
  x : ℝ
  y : ℝ

-- Vertices of the hexagon
def A := Point.mk 0 0
def B := Point.mk 0 6
def C := Point.mk 2 6
def E := Point.mk 4 6
def F := Point.mk 4 0

-- Target vertex D which we need to find its y-coordinate
def D (y_D : ℝ) := Point.mk 2 y_D

-- Condition that the total area of the hexagon is 90 square units
def total_area_condition (y_D : ℝ) : Prop :=
  let area_ABEF := 4 * 6  -- Area of rectangle ABEF
  let area_BCD_plus_DEF := 90 - area_ABEF  -- Combined area of triangles
  let base_EF := 4
  let height_of_triangles := 2 * 33 / base_EF
  y_D = C.y + height_of_triangles

-- The y-coordinate of vertex D
def y_coordinate_of_vertex_D : ℝ :=
  E.y + (33 * 2) / (E.x - C.x)

-- Proof problem
theorem hexagon_y_coordinate_D (y_D : ℝ) (h : total_area_condition y_D) : y_D = 22.5 :=
by
  -- The proof goes here.
  sorry

end hexagon_y_coordinate_D_l631_631134


namespace problem_statement_l631_631039

noncomputable def g : ℕ × ℕ → ℕ
| (x, y) :=
  if x = y then x * x
  else if x + y = 0 then 0
  else (x + y) * g (y, x)

theorem problem_statement :
  ∀ (g : ℕ × ℕ → ℕ),
    (∀ x, g (x, x) = x * x) →
    (∀ x y, g (x, y) = g (y, x)) →
    (∀ x y, (x + y) * g (x, y) = y * g (x, x + y)) →
    g (2,12) + g (5,25) = 149 :=
by
  intros g h1 h2 h3
  -- The detailed proof goes here
  sorry

end problem_statement_l631_631039


namespace lucien_balls_count_l631_631212

theorem lucien_balls_count (lucca_balls : ℕ) (lucca_percent_basketballs : ℝ) (lucien_percent_basketballs : ℝ) (total_basketballs : ℕ)
  (h1 : lucca_balls = 100)
  (h2 : lucca_percent_basketballs = 0.10)
  (h3 : lucien_percent_basketballs = 0.20)
  (h4 : total_basketballs = 50) :
  ∃ lucien_balls : ℕ, lucien_balls = 200 :=
by
  sorry

end lucien_balls_count_l631_631212


namespace sum_of_A_l631_631467

noncomputable def A (n : ℕ) : ℕ := ∑ k in finset.filter (λ x, x > 0) (finset.image (λ (ks:fin n → ℤ), (∑ i, (ks i) * 2^(i + 1))) (finset.pi (fin n) (λ _, finset.insert 1 (finset.singleton (-1))))), k

theorem sum_of_A (n : ℕ) (h : 2 ≤ n) : A n = 2^(2*n - 1) := 
sorry

end sum_of_A_l631_631467


namespace part1_part2_l631_631580

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 2 * Real.sin x)
def vec_b : ℝ × ℝ := (1, -Real.sqrt 2)
def vec_c (m : ℝ) : ℝ × ℝ := (m, -1)

-- Part 1: Prove if (vec_a - vec_b) is parallel to vec_b, then x = 3π / 4
theorem part1 (x : ℝ) (h1 : x ∈ Icc 0 Real.pi)
  (h2 : (vec_a x - vec_b).fst * vec_b.snd = (vec_a x - vec_b).snd * vec_b.fst) :
  x = 3 * Real.pi / 4 := sorry

-- Part 2: Prove the range of f(x) is [-2, √2] given vec_c is orthogonal to vec_b
theorem part2 (x : ℝ) (h1 : x ∈ Icc 0 Real.pi) (m : ℝ) (h2 : (m + Real.sqrt 2 = 0))
  (f : ℝ → ℝ := λ x, (vec_a x).fst * -Real.sqrt 2 + (vec_a x).snd * (-1)) :
  ∀ y, y = f x → y ∈ Icc (-2 : ℝ) (Real.sqrt 2) := sorry

end part1_part2_l631_631580


namespace find_initial_marbles_l631_631012

-- Definitions based on conditions
def loses_to_street (initial_marbles : ℕ) : ℕ := initial_marbles - (initial_marbles * 60 / 100)
def loses_to_sewer (marbles_after_street : ℕ) : ℕ := marbles_after_street / 2

-- The given number of marbles left
def remaining_marbles : ℕ := 20

-- Proof statement
theorem find_initial_marbles (initial_marbles : ℕ) : 
  loses_to_sewer (loses_to_street initial_marbles) = remaining_marbles -> 
  initial_marbles = 100 :=
by
  sorry

end find_initial_marbles_l631_631012


namespace solve_line_equation_l631_631625

noncomputable def line_equation : Prop :=
  ∀ (x y : ℝ), 
  let v := (x, y) in 
  let proj := (3 * x + 4 * y) / 25 * (3, 4) in
  proj = (-3/2, -2) → y = -(3 / 4) * x - 25 / 8

theorem solve_line_equation : line_equation :=
begin
  intros x y h,
  sorry
end

end solve_line_equation_l631_631625


namespace avg_salary_l631_631606

-- Conditions as definitions
def number_of_technicians : Nat := 7
def salary_per_technician : Nat := 10000
def number_of_workers : Nat := 14
def salary_per_non_technician : Nat := 6000

-- Total salary of technicians
def total_salary_technicians : Nat := number_of_technicians * salary_per_technician

-- Number of non-technicians
def number_of_non_technicians : Nat := number_of_workers - number_of_technicians

-- Total salary of non-technicians
def total_salary_non_technicians : Nat := number_of_non_technicians * salary_per_non_technician

-- Total salary
def total_salary_all_workers : Nat := total_salary_technicians + total_salary_non_technicians

-- Average salary of all workers
def avg_salary_all_workers : Nat := total_salary_all_workers / number_of_workers

-- Theorem to prove
theorem avg_salary (A : Nat) (h : A = avg_salary_all_workers) : A = 8000 := by
  sorry

end avg_salary_l631_631606


namespace positive_difference_of_fractions_l631_631690

theorem positive_difference_of_fractions : 
  (let a := 8^2 in (a + a) / 8) = 16 ∧ (let a := 8^2 in (a * a) / 8) = 512 →
  (let a := 8^2 in ((a * a) / 8 - (a + a) / 8)) = 496 := 
by
  sorry

end positive_difference_of_fractions_l631_631690


namespace M_and_N_have_no_elements_in_common_l631_631577

noncomputable def M : set (ℝ × ℝ) := {p | (p.1 + 3)^2 + (p.2 - 1)^2 = 0}
def N : set ℝ := {-3, 1}

theorem M_and_N_have_no_elements_in_common : M ∩ (N.prod N) = ∅ :=
sorry

end M_and_N_have_no_elements_in_common_l631_631577


namespace find_m_for_all_n_l631_631727

def sum_of_digits (k: ℕ) : ℕ :=
  k.digits 10 |>.sum

def A (k: ℕ) : ℕ :=
  -- Constructing the number A_k as described
  -- This is a placeholder for the actual implementation
  sorry

theorem find_m_for_all_n (n: ℕ) (hn: 0 < n) :
  ∃ m: ℕ, 0 < m ∧ n ∣ A m ∧ n ∣ m ∧ n ∣ sum_of_digits (A m) :=
sorry

end find_m_for_all_n_l631_631727


namespace number_of_real_root_equations_l631_631114

theorem number_of_real_root_equations :
  (∃ S : Finset (ℕ × ℕ), S = {(b, c) | b ∈ {1, 2, 3, 4, 5, 6} ∧ c ∈ {1, 2, 3, 4, 5, 6} ∧ b^2 - 4 * c ≥ 0} ∧ S.card = 19) := sorry

end number_of_real_root_equations_l631_631114


namespace subset_collection_bound_l631_631193

def symmetric_difference {α : Type*} (A B : set α) : set α := (A \ B) ∪ (B \ A)

theorem subset_collection_bound (n : ℕ) (h : n > 1) (U : finset ℕ := finset.range (n+1) \ finset.singleton 0)
  (𝒜 : finset (finset ℕ)) (h𝒜 : ∀ (A B ∈ 𝒜), A ≠ B → finset.card (symmetric_difference A B) ≥ 2) :
  𝒜.card ≤ 2^(n - 1) ∧ (𝒜.card = 2^(n - 1) → ∀ A ∈ 𝒜, (∀ B ∈ 𝒜, A.card % 2 = B.card % 2)) :=
sorry

end subset_collection_bound_l631_631193


namespace g_inv_equals_g_l631_631554

variable {x l : ℝ}

def g (x : ℝ) (l : ℝ) : ℝ := (3 * x + 4) / (l * x - 3)

theorem g_inv_equals_g (l : ℝ) : 
  (∀ x : ℝ, 4 * l + 9 ≠ 0 → (∃ y : ℝ, g y l = x ∧ g x l = y)) ↔ 
  l ≠ -9/4 :=
by sorry

end g_inv_equals_g_l631_631554


namespace fgh_supermarkets_in_us_more_than_canada_l631_631279

theorem fgh_supermarkets_in_us_more_than_canada
  (total_supermarkets : ℕ)
  (us_supermarkets : ℕ)
  (canada_supermarkets : ℕ)
  (h1 : total_supermarkets = 70)
  (h2 : us_supermarkets = 42)
  (h3 : us_supermarkets + canada_supermarkets = total_supermarkets):
  us_supermarkets - canada_supermarkets = 14 :=
by
  sorry

end fgh_supermarkets_in_us_more_than_canada_l631_631279


namespace five_digit_divisibility_l631_631976

-- Definitions of n and m
def n (a b c d e : ℕ) := 10000 * a + 1000 * b + 100 * c + 10 * d + e
def m (a b d e : ℕ) := 1000 * a + 100 * b + 10 * d + e

-- Condition that n is a five-digit number whose first digit is non-zero and n/m is an integer
theorem five_digit_divisibility (a b c d e : ℕ):
  1 <= a ∧ a <= 9 → 0 <= b ∧ b <= 9 → 0 <= c ∧ c <= 9 → 0 <= d ∧ d <= 9 → 0 <= e ∧ e <= 9 →
  m a b d e ∣ n a b c d e →
  ∃ x y : ℕ, a = x ∧ b = y ∧ c = 0 ∧ d = 0 ∧ e = 0 :=
by
  sorry

end five_digit_divisibility_l631_631976


namespace vector_sum_l631_631129

def a : ℝ × ℝ := (-2, 3)
def b : ℝ × ℝ := (1, -2)

theorem vector_sum:
  2 • a + b = (-3, 4) :=
by 
  sorry

end vector_sum_l631_631129


namespace max_area_l631_631167

noncomputable def PA : ℝ := 3
noncomputable def PB : ℝ := 4
noncomputable def PC : ℝ := 5
noncomputable def BC : ℝ := 6

theorem max_area (PA PB PC BC : ℝ) (hPA : PA = 3) (hPB : PB = 4) (hPC : PC = 5) (hBC : BC = 6) : 
  ∃ (A B C : Type) (area_ABC : ℝ), area_ABC = 19 := 
by 
  sorry

end max_area_l631_631167


namespace count_numbers_containing_7_l631_631886

-- Define a predicate to check if a number contains the digit 7.
def contains_digit_7 (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∈ n.digits 10 ∧ d = 7

-- Define the set of numbers from 1 to 800.
def numbers_from_1_to_800 : set ℕ := {n | 1 ≤ n ∧ n ≤ 800}

-- Define the set of numbers from 1 to 800 that contain the digit 7.
def numbers_containing_7 : set ℕ := {n | n ∈ numbers_from_1_to_800 ∧ contains_digit_7 n}

-- The theorem to prove the required count.
theorem count_numbers_containing_7 :
  (numbers_containing_7.to_finset.card = 62) :=
sorry

end count_numbers_containing_7_l631_631886


namespace moles_of_NaCl_formed_l631_631059

theorem moles_of_NaCl_formed (hcl moles : ℕ) (nahco3 moles : ℕ) (reaction : ℕ → ℕ → ℕ) :
  hcl = 3 → nahco3 = 3 → reaction 1 1 = 1 →
  reaction hcl nahco3 = 3 :=
by 
  intros h1 h2 h3
  -- Proof omitted
  sorry

end moles_of_NaCl_formed_l631_631059


namespace exists_m_divisible_by_n_with_digit_sum_l631_631469

theorem exists_m_divisible_by_n_with_digit_sum (n k : ℕ) (hn : n > 0) (hk : k ≥ n) (h3 : ¬ (n % 3 = 0)) :
  ∃ m : ℕ, m > 0 ∧ m % n = 0 ∧ (nat.digits 10 m).sum = k :=
sorry

end exists_m_divisible_by_n_with_digit_sum_l631_631469


namespace baby_frogs_on_rock_l631_631916

theorem baby_frogs_on_rock (f_l f_L f_T : ℕ) (h1 : f_l = 5) (h2 : f_L = 3) (h3 : f_T = 32) : 
  f_T - (f_l + f_L) = 24 :=
by sorry

end baby_frogs_on_rock_l631_631916


namespace initial_amount_of_A_l631_631736

theorem initial_amount_of_A (A B : ℕ) (h1 : A / B = 4 / 1)
  (h2 : (A - 24) / (B - 6 + 30) = 2 / 3) : A = 48 := by
  sorry

end initial_amount_of_A_l631_631736


namespace M_inter_N_eq_l631_631485

def M (x : ℝ) : Prop := x^2 - 3 * x + 2 > 0
def N (x : ℝ) : Prop := (1/2)^x ≥ 4

theorem M_inter_N_eq :
  ∀ x, (M x ∧ N x) ↔ x ≤ -2 :=
by
  sorry

end M_inter_N_eq_l631_631485


namespace _l631_631603

noncomputable def height_of_isosceles_trapezoid (S : ℝ) (α : ℝ) : ℝ :=
  √(S * tan (α / 2))

lemma trapezoid_height_theorem (AB CD AD BC : ℝ) (S : ℝ) (α : ℝ) 
  (h : ℝ) (isosceles : AB = CD) (parallel : AB ∥ CD) 
  (sides_equal : AD = BC) (area : S = (AB + CD) / 2 * h) 
  (angle_diagonals : α = 2 * atan (h / (AB - CD))) :
  h = height_of_isosceles_trapezoid S α :=
begin
  sorry
end

end _l631_631603


namespace number_of_valid_sets_A_l631_631210

noncomputable def num_sets_A : ℕ :=
  Nat.choose 8 0 + Nat.choose 8 1 + Nat.choose 8 2 + Nat.choose 8 3 + Nat.choose 8 5 + Nat.choose 8 6 + Nat.choose 8 7 + Nat.choose 8 8

theorem number_of_valid_sets_A :
  ∑ k in ({0, 1, 2, 3, 4, 6, 7, 8} : Finset ℕ), Nat.choose 8 k = 186 :=
by
  have h : ∑ k in ({0, 1, 2, 3, 4, 6, 7, 8} : Finset ℕ), Nat.choose 8 k = (∑ k in Finset.range 9, Nat.choose 8 k) - Nat.choose 8 4,
  by simp [Finset.sum_erase_add],
  rw [Finset.sum_range_succ, Nat.choose_succ_self, Finset.sum_range_succ] at h,
  norm_num at h,
  exact h,
  sorry

end number_of_valid_sets_A_l631_631210


namespace replacement_fraction_l631_631735

variable (Q : ℝ) (x : ℝ)

def initial_concentration : ℝ := 0.70
def new_concentration : ℝ := 0.35
def replacement_concentration : ℝ := 0.25

theorem replacement_fraction (h1 : 0.70 * Q - 0.70 * x * Q + 0.25 * x * Q = 0.35 * Q) :
  x = 7 / 9 :=
by
  sorry

end replacement_fraction_l631_631735


namespace measure_angle_BAD_l631_631811

variable (A B C D : Type) 

variables [Angle A B C] [Angle D A C]

-- Given conditions
variable (h1 : ∠ D A C = 39)
variable (h2 : A B = A C)
variable (h3 : A D = B D)

-- The statement to be proven
theorem measure_angle_BAD (A B C D : Type) [Angle A B C] [Angle D A C]
  (h1 : ∠ D A C = 39) (h2 : A B = A C) (h3 : A D = B D) : ∠ B A D = 47 := 
sorry

end measure_angle_BAD_l631_631811


namespace sum_of_squares_of_roots_eq_zero_l631_631033

theorem sum_of_squares_of_roots_eq_zero :
  let p : Polynomial ℂ := Polynomial.C 20 + Polynomial.C 2 * Polynomial.X^2 + Polynomial.C 5 * Polynomial.X^7 + Polynomial.X^10 in
  let roots := Multiset.map (λ x, x) (Polynomial.roots p) in
  (roots.sum) = 0 →
  ∑ r in roots, r^2 = 0 :=
by
  sorry

end sum_of_squares_of_roots_eq_zero_l631_631033


namespace ff_half_l631_631119

def f (x : ℝ) : ℝ :=
if x > 0 then Real.log x / Real.log 2 else 3 ^ x

theorem ff_half : f (f (1 / 2)) = 1 / 3 :=
by
  sorry

end ff_half_l631_631119


namespace kaleb_games_per_box_l631_631552

theorem kaleb_games_per_box (initial_games sold_games boxes remaining_games games_per_box : ℕ)
  (h1 : initial_games = 76)
  (h2 : sold_games = 46)
  (h3 : boxes = 6)
  (h4 : remaining_games = initial_games - sold_games)
  (h5 : games_per_box = remaining_games / boxes) :
  games_per_box = 5 :=
sorry

end kaleb_games_per_box_l631_631552


namespace rationalize_denominator_l631_631240

theorem rationalize_denominator :
  let A := -12
  let B := 7
  let C := 9
  let D := 13
  let E := 5
  A + B + C + D + E = 22 :=
by
  -- Proof goes here
  sorry

end rationalize_denominator_l631_631240


namespace triangle_area_sum_l631_631602

-- Define the isosceles triangle ABC with ∠ABC = 120°
variables (A B C P Q M N : Type) [MetricSpace B]
variables (h_iso_ab : isosceles_triangle ABC 120)
variables (h_rays_angle : angle_between_rays B 60)
variables (h_reflect_A : reflects_base_bloc AC P)
variables (h_reflect_B : reflects_base_bloc AC Q)
variables (h_hits_lat : hits_lateral_sides P M)
variables (h_hits_lat : hits_lateral_sides Q N)

theorem triangle_area_sum :
  area (triangle PTQ_X) = area (triangle AMP) + area (triangle CNQ) :=
sorry

end triangle_area_sum_l631_631602


namespace choir_members_minimum_l631_631343

theorem choir_members_minimum (n : ℕ) : (∃ n, n % 8 = 0 ∧ n % 9 = 0 ∧ n % 10 = 0 ∧ ∀ m, (m % 8 = 0 ∧ m % 9 = 0 ∧ m % 10 = 0) → n ≤ m) → n = 360 :=
by
  sorry

end choir_members_minimum_l631_631343


namespace sum_of_perimeters_l631_631272

theorem sum_of_perimeters (x y : ℝ) (h1 : x^2 + y^2 = 85) (h2 : x^2 - y^2 = 41) :
  4 * (Real.sqrt 63 + Real.sqrt 22) = 4 * (Real.sqrt x^2 + Real.sqrt y^2) :=
by
  sorry

end sum_of_perimeters_l631_631272


namespace parabola_directrix_p_value_l631_631855

-- Define the conditions and the problem statement
variables (a b p : ℝ)
variables (a_pos : a > 0) (b_pos : b > 0) (p_pos : p > 0)
variables (eccentricity_condition : (c : ℝ) (c = 2 * a) * a = 4 * a^2 - b^2)
variables (area_condition : ∃ A B : Point (xA yA : ℝ), triangle_area O A B = sqrt 3 (A = ( - p / 2, sqrt 3 * p / 2) ∧ B = ( - p / 2, - sqrt 3 * p / 2) )

-- Define the problem: proving the value of p given the conditions
theorem parabola_directrix_p_value :
  ∃ (p : ℝ), (a > 0) ∧ (b > 0) ∧ (eccentricity_condition e = 2)*a  ^2 - b^2 = 3 * a ^2 ∧ (area_condition (area_inequality : linear order solving range (p : ℝ) triangle_area O A B = sqrt 3 ) := p = 2 :=
sorry

end parabola_directrix_p_value_l631_631855


namespace locus_of_points_equidistant_l631_631332

-- Define the problem conditions
structure TrihedralAngle (α β γ : Plane) where
  -- Intersection of three planes
  intersection : α ∩ β ∩ γ = { P }
  -- Edges defined by the pairwise intersection
  edge1 : Line
  edge2 : Line
  edge3 : Line
  intersection_edge1 : α ∩ β = edge1
  intersection_edge2 : β ∩ γ = edge2
  intersection_edge3 : α ∩ γ = edge3

-- Problem statement
theorem locus_of_points_equidistant (α β γ : Plane) (T : TrihedralAngle α β γ) : ∃ l : Line, ∀ P : Point, P ∈ l ↔ dist_to_plane P α = dist_to_plane P β ∧ dist_to_plane P β = dist_to_plane P γ :=
sorry

end locus_of_points_equidistant_l631_631332


namespace smallest_positive_integer_solution_l631_631303

theorem smallest_positive_integer_solution (x : ℕ) (h : 5 * x ≡ 17 [MOD 29]) : x = 15 :=
sorry

end smallest_positive_integer_solution_l631_631303


namespace alice_sales_goal_l631_631780

def price_adidas := 45
def price_nike := 60
def price_reeboks := 35
def price_puma := 50
def price_converse := 40

def num_adidas := 10
def num_nike := 12
def num_reeboks := 15
def num_puma := 8
def num_converse := 14

def quota := 2000

def total_sales :=
  (num_adidas * price_adidas) +
  (num_nike * price_nike) +
  (num_reeboks * price_reeboks) +
  (num_puma * price_puma) +
  (num_converse * price_converse)

def exceed_amount := total_sales - quota

theorem alice_sales_goal : exceed_amount = 655 := by
  -- calculation steps would go here
  sorry

end alice_sales_goal_l631_631780


namespace count_numbers_with_seven_l631_631907

open Finset

def contains_digit_seven (n : ℕ) : Prop :=
  ∃ d : ℕ, d ∈ digits 10 n ∧ d = 7

theorem count_numbers_with_seven : 
  (card (filter (λ n, contains_digit_seven n) (range 801))) = 152 := 
by
  sorry

end count_numbers_with_seven_l631_631907


namespace jacob_ate_five_pies_l631_631213

theorem jacob_ate_five_pies (weight_hot_dog weight_burger weight_pie noah_burgers mason_hotdogs_total_weight : ℕ)
    (H1 : weight_hot_dog = 2)
    (H2 : weight_burger = 5)
    (H3 : weight_pie = 10)
    (H4 : noah_burgers = 8)
    (H5 : mason_hotdogs_total_weight = 30)
    (H6 : ∀ x, 3 * x = (mason_hotdogs_total_weight / weight_hot_dog)) :
    (∃ y, y = (mason_hotdogs_total_weight / weight_hot_dog / 3) ∧ y = 5) :=
by
  sorry

end jacob_ate_five_pies_l631_631213


namespace part_I_part_II_l631_631849

def f (x : ℝ) (m : ℕ) : ℝ := |x - m| + |x|

theorem part_I (m : ℕ) (hm : m = 1) : ∃ x : ℝ, f x m < 2 :=
by sorry

theorem part_II (α β : ℝ) (hα : 1 < α) (hβ : 1 < β) (h : f α 1 + f β 1 = 2) :
  (4 / α) + (1 / β) ≥ 9 / 2 :=
by sorry

end part_I_part_II_l631_631849


namespace four_colors_2x2_grid_l631_631054

-- Define the problem conditions and the proof statement
theorem four_colors_2x2_grid :
  ∀ (colors : Fin 100 × Fin 100 → Fin 4),
    (∀ i j, ∃! p, colors (i, p) = j) ∧
    (∀ p q, ∃! i, colors (i, q) = p) →
    ∃ (i1 i2 : Fin 100) (j1 j2 : Fin 100),
      i1 ≠ i2 ∧ j1 ≠ j2 ∧
      (colors (i1, j1) ≠ colors (i2, j1) ∧
       colors (i1, j1) ≠ colors (i1, j2) ∧
       colors (i1, j1) ≠ colors (i2, j2) ∧
       colors (i2, j1) ≠ colors (i1, j2) ∧
       colors (i2, j1) ≠ colors (i2, j2) ∧
       colors (i1, j2) ≠ colors (i2, j2)) :=
begin
  sorry
end

end four_colors_2x2_grid_l631_631054


namespace Beth_finishes_first_l631_631011

theorem Beth_finishes_first (a r : ℝ) (ha_beth : 3 * beth_area = a)
  (ha_carlos : 4 * carlos_area = a) (hr_beth : beth_rate = r / 2) 
  (hr_carlos: carlos_rate = r / 4) : 
  mowing_time(beth_area, beth_rate) < mowing_time(andy_area, r) ∧ 
  mowing_time(beth_area, beth_rate) < mowing_time(carlos_area, carlos_rate) :=
by
  -- Definitions to be added here: 
  -- mowing_time(area, rate) = area / rate
  sorry

end Beth_finishes_first_l631_631011


namespace TileD_in_AreaZ_l631_631647

namespace Tiles

structure Tile :=
  (top : ℕ)
  (right : ℕ)
  (bottom : ℕ)
  (left : ℕ)

def TileA : Tile := {top := 5, right := 3, bottom := 2, left := 4}
def TileB : Tile := {top := 2, right := 4, bottom := 5, left := 3}
def TileC : Tile := {top := 3, right := 6, bottom := 1, left := 5}
def TileD : Tile := {top := 5, right := 2, bottom := 3, left := 6}

variables (X Y Z W : Tile)
variable (tiles : List Tile := [TileA, TileB, TileC, TileD])

noncomputable def areaZContains : Tile := sorry

theorem TileD_in_AreaZ  : areaZContains = TileD := sorry

end Tiles

end TileD_in_AreaZ_l631_631647


namespace total_passengers_l631_631737

theorem total_passengers (P : ℕ)
  (h1 : P / 12 + P / 8 + P / 3 + P / 6 + 35 = P) : 
  P = 120 :=
by
  sorry

end total_passengers_l631_631737


namespace minimallyIntersectingTriples_remainder_l631_631413

noncomputable def minimallyIntersectingTriplesMod1000 : ℕ :=
  let n := 8
  let k := 3
  let elements := Finset.range n
  let choose_triples := Finset.card (Finset.powersetLen k elements)
  let remaining := n - k
  let arrangements := remaining ^ remaining
  let total := choose_triples * arrangements
  total % 1000

theorem minimallyIntersectingTriples_remainder :
  minimallyIntersectingTriplesMod1000 = 64 := by
  sorry

end minimallyIntersectingTriples_remainder_l631_631413


namespace cos_Z_l631_631955

-- Define the triangle XYZ with X being a right angle and the given sin Y.
structure TriangleXYZ where
  X Y Z : ℝ
  angle_X : X = 90
  sin_Y : sin Y = 3 / 5

-- State the theorem that proves cos Z given the properties of the triangle.
theorem cos_Z (T : TriangleXYZ) : cos T.Z = 3 / 5 := 
  sorry

end cos_Z_l631_631955


namespace initial_fee_calculation_l631_631550

theorem initial_fee_calculation 
  (charge_per_segment : ℝ)
  (segment_length : ℝ)
  (total_distance : ℝ)
  (total_charge : ℝ)
  (number_of_segments := total_distance / segment_length)
  (cost_for_distance := number_of_segments * charge_per_segment)
  (initial_fee := total_charge - cost_for_distance) :
  charge_per_segment = 0.35 → 
  segment_length = 2 / 5 → 
  total_distance = 3.6 → 
  total_charge = 5.5 → 
  initial_fee = 2.35 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp only [div_eq_mul_inv, mul_comm (3.6:ℝ), mul_assoc, mul_inv_cancel_left₀ (ne_of_gt (by norm_num : (2:ℝ) ≠ 0)), mul_comm 2, ←mul_assoc, mul_comm 0.35, sub_self_add]
  norm_num

end initial_fee_calculation_l631_631550


namespace sum_of_roots_is_14_over_3_l631_631402

noncomputable def sum_of_roots_of_polynomial : ℚ :=
    let p := (λ x : ℚ, (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7))
    if h : p = 0 then by
        -- Roots of the polynomial
        let roots := {-4/3, 6}
        -- Sum of the roots
        let sum_of_roots := (roots.sum)
        exact sum_of_roots
    else
        sorry

theorem sum_of_roots_is_14_over_3 : sum_of_roots_of_polynomial = 14 / 3 :=
    sorry

end sum_of_roots_is_14_over_3_l631_631402


namespace numbers_with_7_in_1_to_800_l631_631914

theorem numbers_with_7_in_1_to_800 : 
  (card { n ∈ finset.range (800 + 1) | ∃ d ∈ n.digits 10, d = 7 }) = 152 := 
sorry

end numbers_with_7_in_1_to_800_l631_631914


namespace tammy_investment_change_l631_631523

theorem tammy_investment_change :
  ∀ (initial_investment : ℝ) (loss_percent : ℝ) (gain_percent : ℝ),
    initial_investment = 200 → 
    loss_percent = 0.2 → 
    gain_percent = 0.25 →
    ((initial_investment * (1 - loss_percent)) * (1 + gain_percent)) = initial_investment :=
by
  intros initial_investment loss_percent gain_percent
  sorry

end tammy_investment_change_l631_631523


namespace largest_non_formable_amount_l631_631943

theorem largest_non_formable_amount (n : ℕ) : 
  let denominations := [3n - 2, 6n - 1, 6n + 2, 6n + 5] in
  ∀ s, (∃ (c1 c2 c3 c4 : ℕ), s = c1 * (3n - 2) + c2 * (6n - 1) + c3 * (6n + 2) + c4 * (6n + 5)) ∨ s <= 6n^2 - 4n - 3 :=
sorry

end largest_non_formable_amount_l631_631943


namespace find_greatest_integer_l631_631091

theorem find_greatest_integer :
  (M n : ℕ),
  ((∑ k in {3, 4, 5, 6, 7, 8, 9, 10}, 1 / (k! * (21 - k)!)) = (M / (2! * 19!))) →
  (⌊M / 100⌋ = 1048) :=
by
  sorry

end find_greatest_integer_l631_631091


namespace correct_average_l631_631605

theorem correct_average 
  (incorrect_avg : ℝ)
  (count : ℕ)
  (incorrect_num : ℝ)
  (correct_num : ℝ)
  (incorrect_total_sum : ℝ := incorrect_avg * count)
  (correct_total_sum : ℝ := incorrect_total_sum - incorrect_num + correct_num) :
  incorrect_avg = 18 →
  count = 10 →
  incorrect_num = 26 →
  correct_num = 36 →
  correct_total_sum / count = 19 :=
by
  intros h_avg h_count h_incorrect h_correct
  rw [incorrect_total_sum, h_avg, h_count]
  rw [correct_total_sum, h_incorrect, h_correct]
  sorry -- proof to be completed

end correct_average_l631_631605


namespace count_numbers_containing_7_l631_631882

-- Define a predicate to check if a number contains the digit 7.
def contains_digit_7 (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∈ n.digits 10 ∧ d = 7

-- Define the set of numbers from 1 to 800.
def numbers_from_1_to_800 : set ℕ := {n | 1 ≤ n ∧ n ≤ 800}

-- Define the set of numbers from 1 to 800 that contain the digit 7.
def numbers_containing_7 : set ℕ := {n | n ∈ numbers_from_1_to_800 ∧ contains_digit_7 n}

-- The theorem to prove the required count.
theorem count_numbers_containing_7 :
  (numbers_containing_7.to_finset.card = 62) :=
sorry

end count_numbers_containing_7_l631_631882


namespace positive_difference_l631_631673

theorem positive_difference :
  let a := 8^2
  let term1 := (a + a) / 8
  let term2 := (a * a) / 8
  term2 - term1 = 496 :=
by
  let a := 8^2
  let term1 := (a + a) / 8
  let term2 := (a * a) / 8
  have h1 : a = 64 := rfl
  have h2 : term1 = 16 := by simp [a, term1]
  have h3 : term2 = 512 := by simp [a, term2]
  show 512 - 16 = 496 from sorry

end positive_difference_l631_631673


namespace not_increasing_exp_neg_l631_631615

-- Define the function y = a^-x
def exp_neg (a : ℝ) (x : ℝ) : ℝ := a^(-x)

-- The statement to be proved
theorem not_increasing_exp_neg (a : ℝ) (x1 x2 : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  ¬ ∀ x1 x2, x1 < x2 → exp_neg a x1 < exp_neg a x2 :=
sorry

end not_increasing_exp_neg_l631_631615


namespace difference_mean_median_l631_631934

-- Definitions for the conditions
def percentage_60 := 0.15
def percentage_75 := 0.25
def percentage_85 := 0.40
def percentage_95 := 0.20

def score_60 := 60
def score_75 := 75
def score_85 := 85
def score_95 := 95

-- Define the mean score
def mean_score : ℝ := (percentage_60 * score_60) + (percentage_75 * score_75) + (percentage_85 * score_85) + (percentage_95 * score_95)

-- Define the median score
def median_score : ℝ := score_85

-- State the theorem: the difference between the mean and median score is 4
theorem difference_mean_median : abs (median_score - mean_score) = 4 :=
by
  sorry

end difference_mean_median_l631_631934


namespace ernie_can_make_circles_l631_631376

-- Make a statement of the problem in Lean 4
theorem ernie_can_make_circles (total_boxes : ℕ) (ali_boxes_per_circle : ℕ) (ernie_boxes_per_circle : ℕ) (ali_circles : ℕ) 
  (h1 : total_boxes = 80) (h2 : ali_boxes_per_circle = 8) (h3 : ernie_boxes_per_circle = 10) (h4 : ali_circles = 5) :
  (total_boxes - ali_boxes_per_circle * ali_circles) / ernie_boxes_per_circle = 4 := 
by 
  -- Proof of the theorem
  sorry

end ernie_can_make_circles_l631_631376


namespace inequality_holds_l631_631285

theorem inequality_holds (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 12 → x^2 + 25 + |x^3 - 5 * x^2| ≥ a * x) ↔ a ≤ 2.5 := 
by
  sorry

end inequality_holds_l631_631285


namespace better_representation_of_data_l631_631658

theorem better_representation_of_data (A B: Type) (C D: Type) : 
  (D = "Histogram") → 
  ∀ table_scatter_residual : A ∨ B ∨ C,
  ∀ histogram : D,
  D = "Histogram" :=
by
  intro hD 
  apply hD
  sorry  

end better_representation_of_data_l631_631658


namespace digit_in_101st_place_l631_631500

theorem digit_in_101st_place :
  let repeating_sequence := "269230769"
  in string.nth repeating_sequence ((101 % 9) - 1) = '6' :=
by
  sorry

end digit_in_101st_place_l631_631500


namespace area_of_triangle_ABC_l631_631644

theorem area_of_triangle_ABC :
  ∃ (A B C : ℝ × ℝ), 
  let dist := λ p₁ p₂ : ℝ × ℝ, (real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)) in
  let A := (-7, 3) in
  let B := (0, -4) in
  let C := (9, 5) in
  dist A B = 7 ∧ dist B C = 9 ∧
  1 / 2 * real.abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) = 63 :=
by
  sorry

end area_of_triangle_ABC_l631_631644


namespace right_triangle_condition_l631_631407

noncomputable def f : ℝ → ℝ :=
  λ x, if x < Real.exp 1 then -x^3 + x^2 else Real.log x

theorem right_triangle_condition (a : ℝ) :
  (∃ (P Q : ℝ × ℝ),
     (P = (t, f t)) ∧
     (Q = (-t, f t)) ∧
     (P ≠ O) ∧
     (Q ≠ O) ∧
     (P.1 ≠ Q.1) ∧
     (P.2 ≠ Q.2) ∧
     (⟨O.1, O.2⟩ ∈ affine_hull ℝ {P, Q}) ∧
     (P.1 * Q.1 + P.2 * Q.2 = 0)) ↔ 
  0 < a ∧ a ≤ (1 / (Real.exp 1 * Real.log (Real.exp 1) + 1)) :=
sorry

end right_triangle_condition_l631_631407


namespace CindyHomework_l631_631028

theorem CindyHomework (x : ℤ) (h : (x - 7) * 4 = 48) : (4 * x - 7) = 69 := by
  sorry

end CindyHomework_l631_631028


namespace ernie_circles_l631_631373

theorem ernie_circles (boxes_per_circle_ali boxes_per_circle_ernie total_boxes circles_ali : ℕ) 
  (h1 : boxes_per_circle_ali = 8)
  (h2 : boxes_per_circle_ernie = 10)
  (h3 : total_boxes = 80)
  (h4 : circles_ali = 5) : 
  (total_boxes - circles_ali * boxes_per_circle_ali) / boxes_per_circle_ernie = 4 :=
by
  sorry

end ernie_circles_l631_631373


namespace solve_expression_l631_631245

def num : ℝ := 0.76 * 0.76 * 0.76 - 0.008
def denom : ℝ := 0.76 * 0.76 + 0.76 * 0.2 + 0.04

theorem solve_expression : abs ((num / denom) - 0.56) < 1e-6 :=
by 
  have h_num : num = 0.430976 := by sorry
  have h_denom : denom = 0.7696 := by sorry
  rw [h_num, h_denom]
  norm_num
  sorry

end solve_expression_l631_631245


namespace augmented_matrix_solution_l631_631472

-- Define the conditions in the problem
def augmented_matrix := ![![2, 0, m], ![n, 1, 2]]
def solution := (x = 1) ∧ (y = 1)

-- Define the target statement
theorem augmented_matrix_solution (m n : ℕ) (x y : ℕ) 
(h_aug : augmented_matrix = ![![2, 0, m], ![n, 1, 2]]) 
(h_sol : solution) : 
m + n = 3 := sorry

end augmented_matrix_solution_l631_631472


namespace problem_1_problem_2_l631_631473

def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 11 = 0
def line2 (x y : ℝ) : Prop := 2 * x + 3 * y - 8 = 0
def line3 (x y : ℝ) : Prop := 3 * x + 2 * y + 5 = 0
def point_M : ℝ × ℝ := (1, 2)
def point_P : ℝ × ℝ := (3, 1)

def line_l1 (x y : ℝ) : Prop := x + 2 * y - 5 = 0
def line_l2 (x y : ℝ) : Prop := 2 * x - 3 * y + 4 = 0

theorem problem_1 :
  (line1 point_M.1 point_M.2) ∧ (line2 point_M.1 point_M.2) → 
  (line_l1 point_P.1 point_P.2) ∧ (line_l1 point_M.1 point_M.2) :=
by 
  sorry

theorem problem_2 :
  (line1 point_M.1 point_M.2) ∧ (line2 point_M.1 point_M.2) →
  (∀ (x y : ℝ), line_l2 x y ↔ line3 x y) :=
by
  sorry

end problem_1_problem_2_l631_631473


namespace fractional_eq_nonneg_solution_l631_631845

theorem fractional_eq_nonneg_solution 
  (m x : ℝ)
  (h1 : x ≠ 2)
  (h2 : x ≥ 0)
  (eq_fractional : m / (x - 2) + 1 = x / (2 - x)) :
  m ≤ 2 ∧ m ≠ -2 := 
  sorry

end fractional_eq_nonneg_solution_l631_631845


namespace distance_point_to_line_polar_l631_631538

theorem distance_point_to_line_polar :
  let point_polar := (2, Real.pi/2)
  let line_eq_polar := λ (ρ θ : Real), ρ * Real.cos θ = 1
  let point_rect := (0 : Real, 2 : Real)
  let line_eq_rect := (x : Real) = 1
  let distance := Real.abs (0 - 1)
  distance = 1 :=
by
  let point_polar := (2, Real.pi/2)
  let line_eq_polar := λ (ρ θ : Real), ρ * Real.cos θ = 1
  let point_rect := (0 : Real, 2 : Real)
  let line_eq_rect := (x : Real) = 1
  let distance := Real.abs (0 - 1)
  sorry

end distance_point_to_line_polar_l631_631538


namespace positive_difference_l631_631677

theorem positive_difference (a k : ℕ) (h1 : a = 8^2) (h2 : k = 8) :
  abs ((a + a) / k - (a * a) / k) = 496 :=
by
  sorry

end positive_difference_l631_631677


namespace positive_difference_is_496_l631_631720

def square (n: ℕ) : ℕ := n * n
def term1 := (square 8 + square 8) / 8
def term2 := (square 8 * square 8) / 8
def positive_difference := abs (term2 - term1)

theorem positive_difference_is_496 : positive_difference = 496 :=
by
  -- This is where the proof would go
  sorry

end positive_difference_is_496_l631_631720


namespace smallest_n_for_A0An_length_l631_631557

-- Definitions of points and conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A0 : Point := ⟨0, 0⟩

def on_x_axis (p : Point) : Prop := p.y = 0
def on_parabola (p : Point) : Prop := p.y = p.x^2

-- Functions to represent the conditions and sequences
def A_seq : ℕ → Point := sorry
def B_seq : ℕ → Point := sorry

-- Distinct points condition
def distinct_points {α : Type} (seq : ℕ → α) : Prop :=
  ∀ (i j : ℕ), i ≠ j → seq i ≠ seq j
  
def right_triangle_hypotenuse (A B C : Point) : Prop :=
  ((A.x - B.x)^2 + (A.y - B.y)^2 = ((A.x - C.x)^2 + (A.y + C.y)^2) + ((C.x - B.x)^2 + (C.y - B.y)^2)) ∨
  ((A.x - B.x)^2 + (A.y - B.y)^2 = ((B.x - C.x)^2 + (B.y + C.y)^2) + ((C.x - A.x)^2 + (C.y - A.y)^2))

-- The main proof statement
theorem smallest_n_for_A0An_length (n : ℕ) 
  (h1 : ∀ k, on_x_axis (A_seq k)) 
  (h2 : ∀ k, on_parabola (B_seq k)) 
  (h3 : distinct_points A_seq) 
  (h4 : distinct_points B_seq) 
  (h5 : ∀ k > 0, right_triangle_hypotenuse (A_seq (k-1)) (B_seq k) (A_seq k)) 
  : ∃ n, ∑ i in (finset.range n), (A_seq i).x^2 ≥ 100 ∧ ∀ m < n, ∑ i in (finset.range m), (A_seq i).x^2 < 100 := 
sorry

end smallest_n_for_A0An_length_l631_631557


namespace pet_store_combination_l631_631359

theorem pet_store_combination (puppies kittens hamsters : ℕ) 
  (alice_must_buy_puppy : ∃ puppy : fin (puppies + 1), true)
  (bob_charlie_variety : ∀ (bob_choice charlie_choice : Type), 
    bob_choice ≠ charlie_choice) : 
  (puppies = 20) → (kittens = 4) → (hamsters = 8) → 
  ∃ (ways_to_buy : ℕ), ways_to_buy = 64 :=
by
  sorry

end pet_store_combination_l631_631359


namespace range_of_m_min_of_sum_sq_l631_631856

variable (t m : ℝ)

-- Define the condition for problem (I)
def inequality_condition := ∀ t : ℝ, |t + 3| - |t - 2| ≤ 6 * m - m^2

-- Define the problem (I) statement
theorem range_of_m (h : inequality_condition t m) : m ≥ 1 ∧ m ≤ 5 :=
sorry

-- Define variables for problem (II)
variable (x y z : ℝ)
-- Define lambda
def lambda : ℝ := 5

-- Define the equation 3x + 4y + 5z = lambda
def equation := 3 * x + 4 * y + 5 * z = lambda

-- Define the problem (II) statement
theorem min_of_sum_sq (h₁ : equation x y z) : x^2 + y^2 + z^2 ≥ 1/2 :=
sorry

end range_of_m_min_of_sum_sq_l631_631856


namespace find_greatest_integer_l631_631093

theorem find_greatest_integer :
  (M n : ℕ),
  ((∑ k in {3, 4, 5, 6, 7, 8, 9, 10}, 1 / (k! * (21 - k)!)) = (M / (2! * 19!))) →
  (⌊M / 100⌋ = 1048) :=
by
  sorry

end find_greatest_integer_l631_631093


namespace original_profit_percentage_l631_631354

-- Our definitions based on conditions.
variables (P S : ℝ)
-- Selling at double the price results in 260% profit
axiom h : (2 * S - P) / P * 100 = 260

-- Prove that the original profit percentage is 80%
theorem original_profit_percentage : (S - P) / P * 100 = 80 := 
sorry

end original_profit_percentage_l631_631354


namespace remainder_of_expression_l631_631144

theorem remainder_of_expression (n : ℤ) (h : n % 100 = 99) : (n^2 + 2*n + 3 + n^3) % 100 = 1 :=
by
  sorry

end remainder_of_expression_l631_631144


namespace positive_difference_l631_631699

def a := 8^2
def b := a + a
def c := a * a
theorem positive_difference : ((b / 8) - (c / 8)) = 496 := by
  sorry

end positive_difference_l631_631699


namespace angle_CFD_30_l631_631977

theorem angle_CFD_30 (O A B E F C D : Type) [circle O A B] 
  (symm_EF : symmetric_about O E F) 
  (tan_BC : tangent_at B C) (tan_EC : tangent_at E C) 
  (intersect_AE_D : intersects (line_through A E) (line_through C (tangent_at F)))
  (angle_BAE_60 : angle_BAE A B E = 60) : angle_CFD C F D = 30 := 
by
  sorry

end angle_CFD_30_l631_631977


namespace number_of_numbers_with_digit_seven_l631_631896

-- Define what it means to contain digit 7
def contains_digit_seven (n : ℕ) : Prop :=
  n.digits 10 ∈ [7]

-- Define the set of numbers from 1 to 800 containing at least one digit 7
def numbers_with_digit_seven : ℕ → Prop :=
  λ n, 1 ≤ n ∧ n ≤ 800 ∧ contains_digit_seven n

-- State the theorem
theorem number_of_numbers_with_digit_seven : (finset.filter numbers_with_digit_seven (finset.range 801)).card = 152 :=
sorry

end number_of_numbers_with_digit_seven_l631_631896


namespace square_side_length_l631_631364

theorem square_side_length (P : ℝ) (s : ℝ) (h1 : P = 36) (h2 : P = 4 * s) : s = 9 := 
by sorry

end square_side_length_l631_631364


namespace pentagonal_cross_section_exists_regular_pentagonal_cross_section_impossible_l631_631959

/-
  We define the geometrical nature of the problem as follows:
  - We have a cube in 3D space.
  - A plane can intersect the cube to form different polygons.
  - We need to show the existence of a plane forming a pentagonal cross-section.
  - We need to show the impossibility of such a plane forming a regular pentagon.
-/

theorem pentagonal_cross_section_exists (cube : set (ℝ × ℝ × ℝ)) (plane : set (ℝ × ℝ × ℝ)) :
  (∃ (pent : set (ℝ × ℝ)), polygon pent ∧ sides pent = 5 ∧ (cube ∩ plane = pent)) :=
begin
  sorry
end

theorem regular_pentagonal_cross_section_impossible (cube : set (ℝ × ℝ × ℝ)) (plane : set (ℝ × ℝ × ℝ)) :
  ¬(∃ (reg_pent : set (ℝ × ℝ)), regular_polygon reg_pent ∧ sides reg_pent = 5 ∧ (cube ∩ plane = reg_pent)) :=
begin
  sorry
end

end pentagonal_cross_section_exists_regular_pentagonal_cross_section_impossible_l631_631959


namespace difference_of_extremes_l631_631732

def is_valid_number (n : ℕ) : Prop :=
  let digits := Int.to_digits 10 n in
  n > 99999 ∧ length digits = 6 ∧ count digits 1 = 2 ∧ count digits 4 = 2 ∧ count digits 0 = 2

theorem difference_of_extremes :
  ∀ (n m : ℕ), is_valid_number n ∧ is_valid_number m →
  (n ≤ m ∨ m ≤ n) →
  (∀ k, is_valid_number k → (k ≤ n ∨ k ≥ m)) →
  m - n = 340956 :=
by
  intros
  sorry

end difference_of_extremes_l631_631732


namespace arc_length_of_rho_l631_631740

def rho (φ : ℝ) : ℝ := 3 * Real.exp (3 * φ / 4)

def arc_length (φ1 φ2 : ℝ) : ℝ :=
  ∫ φ in φ1..φ2, sqrt ((rho φ)^2 + (D (fun φ, rho φ) φ)^2)

theorem arc_length_of_rho :
  arc_length 0 (π / 3) = 5 * (Real.exp (π / 4) - 1) := 
by
  sorry

end arc_length_of_rho_l631_631740


namespace find_monthly_salary_l631_631356

variables (S : ℝ) -- S represents the man's monthly salary

-- Conditions
def initial_saving_rate := 0.2
def increased_expense_rate := 0.1
def monthly_savings_after_increase := 500

-- The expenses before the increase
def initial_expenses := S - initial_saving_rate * S

-- The expenses after the increase
def increased_expenses := initial_expenses * (1 + increased_expense_rate)

-- The condition after the increase in expenses
def new_savings_condition := S - increased_expenses = monthly_savings_after_increase

theorem find_monthly_salary (h : new_savings_condition S) : S = 4166.67 :=
sorry

end find_monthly_salary_l631_631356


namespace garden_table_bench_cost_l631_631755

theorem garden_table_bench_cost (B T : ℕ) (h1 : T + B = 750) (h2 : T = 2 * B) : B = 250 :=
by
  sorry

end garden_table_bench_cost_l631_631755


namespace greatest_int_satisfying_inequality_l631_631663

theorem greatest_int_satisfying_inequality : ∃ n : ℤ, (∀ m : ℤ, m^2 - 13 * m + 40 ≤ 0 → m ≤ n) ∧ n = 8 := 
sorry

end greatest_int_satisfying_inequality_l631_631663


namespace richard_twice_as_old_as_scott_in_8_years_l631_631340

theorem richard_twice_as_old_as_scott_in_8_years :
  (richard_age - david_age = 6) ∧ (david_age - scott_age = 8) ∧ (david_age = 14) →
  (richard_age + 8 = 2 * (scott_age + 8)) :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end richard_twice_as_old_as_scott_in_8_years_l631_631340


namespace area_OMN_constant_l631_631532

noncomputable def equation_of_curve (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 2) = 1

theorem area_OMN_constant (M N : ℝ × ℝ)
  (h1 : ∃ (x y : ℝ), equation_of_curve x y)
  (h2 : M.1 * N.1 = 2)
  (h3 : (M.2 - N.2)^2 = 2) :
  ∃ (area : ℝ), area = sqrt 2 :=
sorry

end area_OMN_constant_l631_631532


namespace dance_off_time_l631_631968

def combined_dancing_time (john_first_session : ℕ) (john_second_session : ℕ) (john_break : ℕ) (james_additional_fraction : ℚ) : ℕ :=
  let john_total_danced := john_first_session + john_second_session
  let john_total_including_break := john_first_session + john_second_session + john_break
  let james_total_danced := john_total_including_break + (john_total_including_break * james_additional_fraction)
  john_total_danced + james_total_danced

theorem dance_off_time :
  combined_dancing_time 3 5 1 (1 / 3) = 20 := by
  sorry

end dance_off_time_l631_631968


namespace problem1_problem2_l631_631440

-- Problem 1: Prove that x = ±7/2 given 4x^2 - 49 = 0
theorem problem1 (x : ℝ) : 4 * x^2 - 49 = 0 → x = 7 / 2 ∨ x = -7 / 2 := 
by
  sorry

-- Problem 2: Prove that x = 2 given (x + 1)^3 - 27 = 0
theorem problem2 (x : ℝ) : (x + 1)^3 - 27 = 0 → x = 2 := 
by
  sorry

end problem1_problem2_l631_631440


namespace transformed_graph_symmetry_l631_631243

noncomputable def min_abs_phi (f : ℝ → ℝ) (φ : ℝ) : ℝ :=
  abs φ
  
theorem transformed_graph_symmetry {φ : ℝ} :
  let f := λ x : ℝ, (1 / 2) * Real.sin (2 * x + φ) in
  let f_shifted := λ x : ℝ, (1 / 2) * Real.sin (2 * (x + (Real.pi / 6)) + φ) in
  let f_stretched := λ x : ℝ, (1 / 2) * Real.sin (x + (Real.pi / 3) + φ) in
  f_stretched (Real.pi / 3) = (1 / 2) * Real.sin ((2 * Real.pi / 3) + φ)
  → ∃ k : ℤ, φ = - (Real.pi / 6) + k * Real.pi ∧ min_abs_phi f φ = Real.pi / 6 :=
sorry

end transformed_graph_symmetry_l631_631243


namespace hundred_and_first_digit_l631_631493

/-- The repeating sequence for the decimal expansion of 7/26 -/
def repeating_sequence : List ℕ := [2, 6, 9, 2, 3, 0, 7, 6, 9]

/-- The length of the repeating sequence is 9 -/
def period : ℕ := 9

/-- The function to get the n-th digit of the decimal expansion of 7/26 -/
def nth_digit (n : ℕ) : ℕ :=
  repeating_sequence.get ⟨n % period, n.mod_lt (by norm_num)⟩

/-- The 101st digit of the decimal expansion of 7/26 is 6 -/
theorem hundred_and_first_digit : nth_digit 101 = 6 := by
  sorry

end hundred_and_first_digit_l631_631493


namespace angle_ACD_eq_angle_BCM_l631_631180

variable {P : Type} [inner_product_space ℝ P]

-- Define points and angles in the quadrilateral and M
variables {A B C D M : P}
variable (ABMD_parallelogram : ∀ (A B λ M), parallelogram A B D M)

-- Define the given angle equality
variable (angle_CBM_eq_angle_CDM : ∠CBM = ∠CDM)

-- Define the proof statement
theorem angle_ACD_eq_angle_BCM (h : ABMD_parallelogram A B D M)
  (angle_CBM_eq_angle_CDM : ∠CBM = ∠CDM) : ∠ACD = ∠BCM :=
sorry

end angle_ACD_eq_angle_BCM_l631_631180


namespace count_zero_sums_l631_631559

noncomputable def S (n : ℕ) : ℝ :=
  ∑ k in finset.range (n+1), real.sin (k * real.pi / 5)

theorem count_zero_sums :
  finset.filter (λ n, S n = 0) (finset.range 2019).card = 402 :=
by
  sorry

end count_zero_sums_l631_631559


namespace exists_gcd_property_l631_631982

theorem exists_gcd_property (S : Set ℕ) (hS : S.Infinite) (a b c d : ℕ) 
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d)
  (haS : a ∈ S) (hbS : b ∈ S) (hcS : c ∈ S) (hdS : d ∈ S)
  (hgcd : Nat.gcd a b ≠ Nat.gcd c d) :
  ∃ x y z ∈ S, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ Nat.gcd x y = Nat.gcd y z ∧ Nat.gcd y z ≠ Nat.gcd z x :=
by
  sorry

end exists_gcd_property_l631_631982


namespace financial_transaction_l631_631746

def initial_value := 12000
def first_transaction_loss := 0.12
def second_transaction_gain := 0.15
def third_transaction_gain := 0.10

def first_selling_price := initial_value * (1 - first_transaction_loss)
def second_selling_price := first_selling_price * (1 + second_transaction_gain)
def final_selling_price := second_selling_price * (1 + third_transaction_gain)
def gain := final_selling_price - initial_value

-- Statement to prove
theorem financial_transaction :
  final_selling_price ≈ 13358 ∧ gain ≈ 1214 :=
by {
  sorry
}

end financial_transaction_l631_631746


namespace largest_value_of_n_l631_631432

-- Conditions
variables (A B n : ℤ)
def polynomial (A B n : ℤ) := 5 * A * B = 110 ∧ n = 5 * B + A

-- The main statement to prove
theorem largest_value_of_n : 
  ∃ A B n, polynomial A B n ∧ n = 551 :=
by
  use 1
  use 110
  use 551
  split
  { split
    { norm_num }
    { norm_num }
  }
  { refl }

end largest_value_of_n_l631_631432


namespace positive_difference_of_fractions_l631_631695

theorem positive_difference_of_fractions : 
  (let a := 8^2 in (a + a) / 8) = 16 ∧ (let a := 8^2 in (a * a) / 8) = 512 →
  (let a := 8^2 in ((a * a) / 8 - (a + a) / 8)) = 496 := 
by
  sorry

end positive_difference_of_fractions_l631_631695


namespace interest_rate_per_annum_l631_631366

def principal : ℝ := 8945
def simple_interest : ℝ := 4025.25
def time : ℕ := 5

theorem interest_rate_per_annum : (simple_interest * 100) / (principal * time) = 9 := by
  sorry

end interest_rate_per_annum_l631_631366


namespace ernie_circles_l631_631374

theorem ernie_circles (boxes_per_circle_ali boxes_per_circle_ernie total_boxes circles_ali : ℕ) 
  (h1 : boxes_per_circle_ali = 8)
  (h2 : boxes_per_circle_ernie = 10)
  (h3 : total_boxes = 80)
  (h4 : circles_ali = 5) : 
  (total_boxes - circles_ali * boxes_per_circle_ali) / boxes_per_circle_ernie = 4 :=
by
  sorry

end ernie_circles_l631_631374


namespace circle_passing_through_A_B_C_l631_631783

theorem circle_passing_through_A_B_C :
  ∃ (B C : (ℝ × ℝ)), 
    (B.1 ^ 2 + 2 * B.2 ^ 2 = 2) ∧
    (C.1 ^ 2 + 2 * C.2 ^ 2 = 2) ∧
    (B.1 + 2 * B.2 = 1) ∧
    (C.1 + 2 * C.2 = 1) ∧
    let circle_eqn := 
      λ (x y : ℝ), 6 * x ^ 2 + 6 * y ^ 2 - 8 * x - 12 * y - 3 in
    circle_eqn 2 2 = 0 ∧
    circle_eqn B.1 B.2 = 0 ∧
    circle_eqn C.1 C.2 = 0 :=
sorry

end circle_passing_through_A_B_C_l631_631783


namespace number_of_buses_l631_631756

theorem number_of_buses (vans people_per_van buses people_per_bus extra_people_in_buses : ℝ) 
  (h_vans : vans = 6.0) 
  (h_people_per_van : people_per_van = 6.0) 
  (h_people_per_bus : people_per_bus = 18.0) 
  (h_extra_people_in_buses : extra_people_in_buses = 108.0) 
  (h_eq : people_per_bus * buses = vans * people_per_van + extra_people_in_buses) : 
  buses = 8.0 :=
by
  sorry

end number_of_buses_l631_631756


namespace ratio_notes_exploring_l631_631184

theorem ratio_notes_exploring :
  ∃ (x : ℝ), (3 + x + 0.5 = 5) ∧ (x / 3 = 1 / 2) :=
by
  use 1.5
  split
  · norm_num
  · norm_num

end ratio_notes_exploring_l631_631184


namespace max_subset_size_l631_631578

theorem max_subset_size (S : Set ℕ) (hS : S ⊆ {x : ℕ | 1 ≤ x ∧ x ≤ 200})
  (h_diff : ∀ x y ∈ S, x ≠ y → |x - y| ≠ 4 ∧ |x - y| ≠ 5 ∧ |x - y| ≠ 9) :
  Finset.card (S.toFinset) = 64 :=
sorry

end max_subset_size_l631_631578


namespace find_x_l631_631041

open Real

noncomputable def satisfies_equation (x : ℝ) : Prop :=
  log (x - 1) / log 3 + log (x^2 - 1) / log (sqrt 3) + log (x - 1) / log (1 / 3) = 3

theorem find_x : ∃ x : ℝ, 1 < x ∧ satisfies_equation x ∧ x = sqrt (1 + 3 * sqrt 3) := by
  sorry

end find_x_l631_631041


namespace peter_ate_fraction_of_pizza_l631_631221

theorem peter_ate_fraction_of_pizza :
  let total_slices := 16
  let peter_alone := 2 / 16
  let shared_with_paul := 1 / 32
  let shared_with_paul_and_sarah := 1 / 48
  peter_alone + shared_with_paul + shared_with_paul_and_sarah = 17 / 96 :=
by 
  let total_slices := 16
  let peter_alone := 2 / 16
  let shared_with_paul := 1 / 32
  let shared_with_paul_and_sarah := 1 / 48
  have h1 : peter_alone = 12 / 96, by sorry
  have h2 : shared_with_paul = 3 / 96, by sorry
  have h3 : shared_with_paul_and_sarah = 2 / 96, by sorry
  show 12 / 96 + 3 / 96 + 2 / 96 = 17 / 96, by sorry

end peter_ate_fraction_of_pizza_l631_631221


namespace TinaTotalPens_l631_631287

variable (p g b : ℕ)
axiom H1 : p = 12
axiom H2 : g = p - 9
axiom H3 : b = g + 3

theorem TinaTotalPens : p + g + b = 21 := by
  sorry

end TinaTotalPens_l631_631287


namespace sum_of_diameters_exceeds_4_l631_631348

theorem sum_of_diameters_exceeds_4 (cube : ℝ) (spheres : Fin₈ → ℝ) 
  (h_cube : cube = 1) 
  (h_non_intersect : ∀ i j, i ≠ j → spheres i + spheres j ≤ 1)
  (h_diameter_sum_less : ∑ i, spheres i ≤ 4): 
  ∃ arrangement : Fin₈ → ℝ, 
    (∀ i j, i ≠ j → arrangement i + arrangement j ≤ 1) ∧
    ∑ i, arrangement i > 4 := 
sorry

end sum_of_diameters_exceeds_4_l631_631348


namespace part_one_part_two_l631_631938

open Real

variables {a b c A B C : ℝ}

noncomputable def triangle_exists_acute (a b c : ℝ) : Prop :=
  ∃ A B C, A + B + C = π ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ 
           a = sqrt(7) ∧ area = 3 * sqrt 3 / 2

theorem part_one (h : triangle_exists_acute a b c) 
  (h1 : sqrt 3 * c = 2 * a * sin C) : 
  A = π / 3 := 
begin
  sorry
end

theorem part_two (h : triangle_exists_acute a b c) 
  (h1 : sqrt 3 * c = 2 * a * sin C) 
  (h2 : a = sqrt(7)) 
  (h3 : (1/2) * b * c * (sqrt 3 / 2) = (3 * sqrt 3 / 2)) : 
  a + b + c = sqrt 7 + 5 := 
begin
  sorry
end

end part_one_part_two_l631_631938


namespace value_of_f_at_2_l631_631152

theorem value_of_f_at_2 
  (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f(x) - 2 * f(1 / x) = x + 2) :
  f(2) = -3 :=
by 
  sorry

end value_of_f_at_2_l631_631152


namespace find_common_ratio_l631_631323

-- Conditions

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), 0 < r ∧ ∀ n, a (n + 1) = a n * r

def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum (λ k, a k)

def Sn_eqn (S : ℕ → ℝ) : Prop :=
  2^10 * S 30 + S 10 = (2^10 + 1) * S 20

-- Theorem
theorem find_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) 
  (h1 : is_geometric_sequence a)
  (h2 : ∀ n, S n = sum_of_first_n_terms a n)
  (h3 : Sn_eqn S) :
  q = 1 / 2 :=
sorry

end find_common_ratio_l631_631323


namespace wool_usage_l631_631801

def total_balls_of_wool_used (scarves_aaron sweaters_aaron sweaters_enid : ℕ) (wool_per_scarf wool_per_sweater : ℕ) : ℕ :=
  (scarves_aaron * wool_per_scarf) + (sweaters_aaron * wool_per_sweater) + (sweaters_enid * wool_per_sweater)

theorem wool_usage :
  total_balls_of_wool_used 10 5 8 3 4 = 82 :=
by
  -- calculations done in solution steps
  -- total_balls_of_wool_used (10 scarves * 3 balls/scarf) + (5 sweaters * 4 balls/sweater) + (8 sweaters * 4 balls/sweater)
  -- total_balls_of_wool_used (30) + (20) + (32)
  -- total_balls_of_wool_used = 30 + 20 + 32 = 82
  sorry

end wool_usage_l631_631801


namespace probability_5_heads_in_8_flips_l631_631752

open Finset

-- Definition of the total outcomes for 8 flips
def total_outcomes := 2 ^ 8

-- Definition for at least 5 consecutive heads recursive condition
def at_least_5_consecutive_heads := sorry  -- Skipped concrete implementation.

-- Calculate the probability
def probability_like_5_heads : ℚ := 23 / 256

theorem probability_5_heads_in_8_flips : 
  (count (filter at_least_5_consecutive_heads (powerset (range 8))) : ℚ) / total_outcomes = probability_like_5_heads :=
sorry

end probability_5_heads_in_8_flips_l631_631752


namespace dots_not_visible_on_3_dice_l631_631645

theorem dots_not_visible_on_3_dice :
  let total_dots := 18 * 21 / 6
  let visible_dots := 1 + 2 + 2 + 3 + 5 + 4 + 5 + 6
  let hidden_dots := total_dots - visible_dots
  hidden_dots = 35 := 
by 
  let total_dots := 18 * 21 / 6
  let visible_dots := 1 + 2 + 2 + 3 + 5 + 4 + 5 + 6
  let hidden_dots := total_dots - visible_dots
  show total_dots - visible_dots = 35
  sorry

end dots_not_visible_on_3_dice_l631_631645


namespace holes_vert_asymp_horiz_asymp_oblique_asymp_sum_l631_631035

-- Definitions
def f (x : ℝ) : ℝ := (x^3 + 4*x^2 + 5*x + 2) / (x^4 + 2*x^3 - x^2 - 2*x)

-- Formal statement
theorem holes_vert_asymp_horiz_asymp_oblique_asymp_sum :
  let a := 1 in
  let b := 2 in
  let c := 1 in
  let d := 0 in
  a + 2 * b + 3 * c + 4 * d = 8 :=
by
  sorry

end holes_vert_asymp_horiz_asymp_oblique_asymp_sum_l631_631035


namespace num_squares_l631_631444

theorem num_squares : 
  { n : ℤ | 0 ≤ n ∧ n ≤ 20 ∧ ∃ k : ℤ, (n / (20 - n) = k * k) }.card = 4 := 
sorry

end num_squares_l631_631444


namespace ahmed_final_score_requirement_l631_631779

-- Define the given conditions
def total_assignments : ℕ := 9
def ahmed_initial_grade : ℕ := 91
def emily_initial_grade : ℕ := 92
def sarah_initial_grade : ℕ := 94
def final_assignment_weight := true -- Assuming each assignment has the same weight
def min_passing_score : ℕ := 70
def max_score : ℕ := 100
def emily_final_score : ℕ := 90

noncomputable def ahmed_min_final_score : ℕ := 98

-- The proof statement
theorem ahmed_final_score_requirement :
  let ahmed_initial_points := ahmed_initial_grade * total_assignments
  let emily_initial_points := emily_initial_grade * total_assignments
  let sarah_initial_points := sarah_initial_grade * total_assignments
  let emily_final_total := emily_initial_points + emily_final_score
  let sarah_final_total := sarah_initial_points + min_passing_score
  let ahmed_final_total_needed := sarah_final_total + 1
  let ahmed_needed_score := ahmed_final_total_needed - ahmed_initial_points
  ahmed_needed_score = ahmed_min_final_score :=
by
  sorry

end ahmed_final_score_requirement_l631_631779


namespace ratio_of_speeds_is_two_l631_631967

noncomputable def joe_speed : ℝ := 0.266666666667
noncomputable def time : ℝ := 40
noncomputable def total_distance : ℝ := 16

noncomputable def joe_distance : ℝ := joe_speed * time
noncomputable def pete_distance : ℝ := total_distance - joe_distance
noncomputable def pete_speed : ℝ := pete_distance / time

theorem ratio_of_speeds_is_two :
  joe_speed / pete_speed = 2 := by
  sorry

end ratio_of_speeds_is_two_l631_631967


namespace find_angle_B_find_sin_2C_l631_631927

-- Define the problem conditions
variables {A B C : ℝ}
variable  (A_pos : 0 < A ∧ A < π)
variable  (B_pos : 0 < B ∧ B < π)
variable  (C_pos : 0 < C ∧ C < π)
variable  (cos_eq : cos C + (cos A - √3 * sin A) * cos B = 0)
variable  (sin_eq : sin (A - π / 3) = 3 / 5)

-- Theorem 1: Prove the measure of angle B
theorem find_angle_B : B = π / 3 :=
by
  sorry

-- Theorem 2: Prove the value of sin 2C
theorem find_sin_2C : sin (2 * C) = (24 + 7 * √3) / 50 :=
by
  sorry


end find_angle_B_find_sin_2C_l631_631927


namespace number_not_diff_squares_1_to_1000_l631_631138

theorem number_not_diff_squares_1_to_1000 : 
  (Finset.range 1000).filter (λ x, ∀ (a b : ℤ), a^2 - b^2 ≠ x).card = 250 :=
by sorry

end number_not_diff_squares_1_to_1000_l631_631138


namespace numbers_with_7_in_1_to_800_l631_631910

theorem numbers_with_7_in_1_to_800 : 
  (card { n ∈ finset.range (800 + 1) | ∃ d ∈ n.digits 10, d = 7 }) = 152 := 
sorry

end numbers_with_7_in_1_to_800_l631_631910


namespace rationalize_denominator_to_find_constants_l631_631234

-- Definitions of the given conditions
def original_fraction := 3 / (4 * Real.sqrt 7 + 3 * Real.sqrt 13)
def simplified_fraction (A B C D E : ℤ) := (A * Real.sqrt B + C * Real.sqrt D) / E

-- Statement of the proof problem
theorem rationalize_denominator_to_find_constants :
  ∃ (A B C D E : ℤ),
    original_fraction = simplified_fraction A B C D E ∧
    B < D ∧
    (∀ p : ℕ, Real.sqrt (p * p) = p) ∧ -- Ensuring that all radicals are in simplest form
    A + B + C + D + E = 22 :=
sorry

end rationalize_denominator_to_find_constants_l631_631234


namespace part1_part2_l631_631127

-- Define the system of equations
def system_eq (x y k : ℝ) : Prop := 
  3 * x + y = k + 1 ∧ x + 3 * y = 3

-- Part (1): x and y are opposite in sign implies k = -4
theorem part1 (x y k : ℝ) (h_eq : system_eq x y k) (h_sign : x * y < 0) : k = -4 := by
  sorry

-- Part (2): range of values for k given extra inequalities
theorem part2 (x y k : ℝ) (h_eq : system_eq x y k) 
  (h_ineq1 : x + y < 3) (h_ineq2 : x - y > 1) : 4 < k ∧ k < 8 := by
  sorry

end part1_part2_l631_631127


namespace total_water_consumption_l631_631999

noncomputable def water_consumption (olaf: ℝ) (ten_men: ℕ → ℝ) (fourteen_men: ℕ → ℝ) (spd_normal: ℝ) (reduction: ℝ) (slower_days: ℕ) (total_dist: ℝ) :=
  let olaf_per_day := olaf
  let ten_men_per_day := 10 * ten_men 10
  let fourteen_men_per_day := 14 * fourteen_men 14
  let total_per_day := olaf_per_day + ten_men_per_day + fourteen_men_per_day

  let spd_slower := spd_normal * reduction
  let dist_slower := slower_days * spd_slower
  
  let remaining_dist := total_dist - dist_slower
  let normal_days := (remaining_dist / spd_normal).ceil -- rounding up
  
  let total_days := slower_days + normal_days

  total_per_day * total_days

theorem total_water_consumption : water_consumption (1 / 2) (λ _, 0.6) (λ _, 0.8) 200 0.85 7 4000 = 389.4 := by
  sorry

end total_water_consumption_l631_631999


namespace find_f_85_l631_631556

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x^2 + x + 3) + 2 * f(x^2 - 3 * x + 5) = 6 * x^2 - 10 * x + 17

theorem find_f_85 (f : ℝ → ℝ) (h : satisfies_condition f) : f 85 = 167 :=
by {
  sorry
}

end find_f_85_l631_631556


namespace tiffany_final_lives_l631_631646

def initial_lives : ℕ := 43
def lost_lives : ℕ := 14
def gained_lives : ℕ := 27

theorem tiffany_final_lives : (initial_lives - lost_lives + gained_lives) = 56 := by
    sorry

end tiffany_final_lives_l631_631646


namespace verify_monomial_properties_l631_631310

def monomial : ℚ := -3/5 * (1:ℚ)^1 * (2:ℚ)^2

def coefficient (m : ℚ) : ℚ := -3/5  -- The coefficient of the monomial
def degree (m : ℚ) : ℕ := 3          -- The degree of the monomial

theorem verify_monomial_properties :
  coefficient monomial = -3/5 ∧ degree monomial = 3 :=
by
  sorry

end verify_monomial_properties_l631_631310


namespace triangle_ELG_circumcircle_radius_l631_631775

open Real Geo

noncomputable def radius_circumcircle_ELG : ℝ :=
  let R := 18 in  -- Given \( R = 18 \) from the calculation of the area of triangle EGO
  let area_EGO := 81 * sqrt 3 in
  let angle_OEG := 60 in
  let EFG_obtuse := true in
  let LEF_eq_FEG := true in
  let LGF_eq_FGE := true in
  6 * sqrt 3 -- Proof (assuming) the result

theorem triangle_ELG_circumcircle_radius :
  ∀ (E F G L O : Point),
  IsInscribed E F G O →
  Angle E F G > 90 →
  Angle L E F = Angle F E G →
  Angle L G F = Angle F G E →
  Area E G O = 81 * sqrt 3 →
  Angle O E G = 60 →
  CircumcircleRadius E L G = 6 * sqrt 3 :=
begin
  intros E F G L O h1 h2 h3 h4 h5 h6,
  sorry
end

end triangle_ELG_circumcircle_radius_l631_631775


namespace count_numbers_with_digit_7_in_range_l631_631874

theorem count_numbers_with_digit_7_in_range : 
  let numbers_in_range := {n : ℕ | 1 ≤ n ∧ n ≤ 800}
      contains_digit_7 (n : ℕ) : Prop := n.digits 10.contains 7
  in (finset.filter (λ n, contains_digit_7 n) (finset.range 801)).card = 152 :=
by 
  let numbers_in_range := {n : ℕ | 1 ≤ n ∧ n ≤ 800}
  let contains_digit_7 (n : ℕ) : Prop := n.digits 10.contains 7
  have h := (finset.filter (λ n, contains_digit_7 n) (finset.range 801)).card
  sorry

end count_numbers_with_digit_7_in_range_l631_631874


namespace boat_travel_difference_l631_631338

-- Define the speeds
variables (a b : ℝ) (ha : a > b)

-- Define the travel times
def downstream_time := 3
def upstream_time := 2

-- Define the distances
def downstream_distance := downstream_time * (a + b)
def upstream_distance := upstream_time * (a - b)

-- Prove the mathematical statement
theorem boat_travel_difference : downstream_distance a b - upstream_distance a b = a + 5 * b := by
  -- sorry can be used to skip the proof
  sorry

end boat_travel_difference_l631_631338


namespace carson_harvests_seaweed_l631_631026

variable {S : ℝ} -- the total amount of seaweed harvested by Carson

-- 50% of the seaweed is only good for starting fires
def p1 (S : ℝ) : Prop := 0.50 * S

-- 25% of what's left can be eaten by humans, and the rest is fed to livestock
def p2 (S : ℝ) : Prop := (0.75 * 0.50 * S = 150)

-- the total amount of seaweed harvested by Carson
theorem carson_harvests_seaweed (S : ℝ) (h1: p1 S) (h2: p2 S) : S = 400 := 
by
  sorry

end carson_harvests_seaweed_l631_631026


namespace count_4_tuples_l631_631080

theorem count_4_tuples (p : ℕ) [hp : Fact (Nat.Prime p)] : 
  Nat.card {abcd : ℕ × ℕ × ℕ × ℕ // (0 < abcd.1 ∧ abcd.1 < p) ∧ 
                                     (0 < abcd.2.1 ∧ abcd.2.1 < p) ∧ 
                                     (0 < abcd.2.2.1 ∧ abcd.2.2.1 < p) ∧ 
                                     (0 < abcd.2.2.2 ∧ abcd.2.2.2 < p) ∧ 
                                     ((abcd.1 * abcd.2.2.2 - abcd.2.1 * abcd.2.2.1) % p = 0)} = (p - 1) * (p - 1) * (p - 1) :=
by
  sorry

end count_4_tuples_l631_631080


namespace average_even_diff_l631_631249

theorem average_even_diff :
  (let avg1 := (∑ i in (finset.Icc 16 44).filter (λ x, x % 2 = 0), i) / (finset.card (finset.Icc 16 44).filter (λ x, x % 2 = 0));
       avg2 := (∑ i in (finset.Icc 14 56).filter (λ x, x % 2 = 0), i) / (finset.card (finset.Icc 14 56).filter (λ x, x % 2 = 0))
   in avg1 - avg2 = -5) :=
sorry

end average_even_diff_l631_631249


namespace line_divides_circle_1_3_l631_631073

noncomputable def circle_equidistant_from_origin : Prop := 
  ∃ l : ℝ → ℝ, ∀ x y : ℝ, ((x-1)^2 + (y-1)^2 = 2) → 
                     (l 0 = 0 ∧ (l x = l y) ∧ 
                     ((x = 0) ∨ (y = 0)))

theorem line_divides_circle_1_3 (x y : ℝ) : 
  (x - 1)^2 + (y - 1)^2 = 2 → 
  (x = 0 ∨ y = 0) :=
by
  sorry

end line_divides_circle_1_3_l631_631073


namespace no_integer_pairs_satisfy_equation_l631_631433

theorem no_integer_pairs_satisfy_equation :
  ∀ (m n : ℤ), m^3 + 6 * m^2 + 5 * m ≠ 27 * n^3 + 27 * n^2 + 9 * n + 1 :=
by
  intros m n
  sorry

end no_integer_pairs_satisfy_equation_l631_631433


namespace solve_for_x_l631_631244

theorem solve_for_x : ∃ x : ℚ, 7 * (4 * x + 3) - 3 = -3 * (2 - 5 * x) + 5 * x / 2 ∧ x = -16 / 7 := by
  sorry

end solve_for_x_l631_631244


namespace janet_action_figures_total_l631_631965

theorem janet_action_figures_total :
  ∀ (initial : ℕ) (sold : ℕ) (bought : ℕ) (brother_gift_coeff : ℕ),
    initial = 10 →
    sold = 6 →
    bought = 4 →
    brother_gift_coeff = 2 →
    (initial - sold + bought) * brother_gift_coeff + (initial - sold + bought) = 24 := by
    intros initial sold bought brother_gift_coeff h_initial h_sold h_bought h_brother_gift_coeff
    rw [h_initial, h_sold, h_bought, h_brother_gift_coeff]
    sorry

end janet_action_figures_total_l631_631965


namespace rationalize_denominator_to_find_constants_l631_631235

-- Definitions of the given conditions
def original_fraction := 3 / (4 * Real.sqrt 7 + 3 * Real.sqrt 13)
def simplified_fraction (A B C D E : ℤ) := (A * Real.sqrt B + C * Real.sqrt D) / E

-- Statement of the proof problem
theorem rationalize_denominator_to_find_constants :
  ∃ (A B C D E : ℤ),
    original_fraction = simplified_fraction A B C D E ∧
    B < D ∧
    (∀ p : ℕ, Real.sqrt (p * p) = p) ∧ -- Ensuring that all radicals are in simplest form
    A + B + C + D + E = 22 :=
sorry

end rationalize_denominator_to_find_constants_l631_631235


namespace second_order_derivative_l631_631435

-- Define the parameterized functions x and y
noncomputable def x (t : ℝ) : ℝ := 1 / t
noncomputable def y (t : ℝ) : ℝ := 1 / (1 + t ^ 2)

-- Define the second-order derivative of y with respect to x
noncomputable def d2y_dx2 (t : ℝ) : ℝ := (2 * (t^2 - 3) * t^4) / (1 + t^2) ^ 3

-- Prove the relationship based on given conditions
theorem second_order_derivative :
  ∀ t : ℝ, (∃ x y : ℝ, x = 1 / t ∧ y = 1 / (1 + t ^ 2)) → 
    (d2y_dx2 t) = (2 * (t^2 - 3) * t^4) / (1 + t^2) ^ 3 :=
by
  intros t ht
  -- Proof omitted
  sorry

end second_order_derivative_l631_631435


namespace largest_amount_not_given_l631_631944

theorem largest_amount_not_given (n : ℕ) (hn : 0 < n) :
  ∃ m : ℕ, ∀ k ∈ Ico 0 m, ¬ ∃ a b c d : ℕ, k = a * (3 * n - 2) + b * (6 * n - 1) + 
  c * (6 * n + 2) + d * (6 * n + 5) ∧ m = 6 * n^2 - 4 * n - 3 := sorry

end largest_amount_not_given_l631_631944


namespace vertex_and_segment_condition_g_monotonically_increasing_g_minimum_value_l631_631638

def f (x : ℝ) : ℝ := -x^2 + 2 * x + 15
def g (x a : ℝ) : ℝ := (2 - 2 * a) * x - f x

theorem vertex_and_segment_condition : 
  (f 1 = 16) ∧ ∃ x1 x2 : ℝ, (f x1 = 0) ∧ (f x2 = 0) ∧ (x2 - x1 = 8) := 
sorry

theorem g_monotonically_increasing (a : ℝ) :
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 2 → g x1 a ≤ g x2 a) ↔ a ≤ 0 :=
sorry

theorem g_minimum_value (a : ℝ) :
  (0 < a ∧ g 2 a = -4 * a - 11) ∨ (a < 0 ∧ g 0 a = -15) ∨ (0 ≤ a ∧ a ≤ 2 ∧ g a a = -a^2 - 15) :=
sorry

end vertex_and_segment_condition_g_monotonically_increasing_g_minimum_value_l631_631638


namespace triangle_area_le_one_fourth_l631_631108

theorem triangle_area_le_one_fourth (A B C P₁ P₂ P₃ P₄ : Type) [linear_ordered_field A] 
  (h₁ : P₁ ∈ segment A B) (h₂ : P₂ ∈ segment B C) (h₃ : P₃ ∈ segment C A) (h₄ : P₄ ∈ segment A B) :
  ∃ (P : Type) [P ∈ {P₁, P₂, P₃, P₄}], 
  area P₁ P₂ P₃ ≤ (area A B C) / 4 ∨ 
  area P₁ P₂ P₄ ≤ (area A B C) / 4 ∨
  area P₁ P₃ P₄ ≤ (area A B C) / 4 ∨
  area P₂ P₃ P₄ ≤ (area A B C) / 4 :=
sorry

end triangle_area_le_one_fourth_l631_631108


namespace necessary_but_not_sufficient_condition_l631_631087

variable {a b : ℝ}

theorem necessary_but_not_sufficient_condition
    (h1 : a ≠ 0)
    (h2 : b ≠ 0) :
    (a^2 + b^2 ≥ 2 * a * b) → 
    (¬(a^2 + b^2 ≥ 2 * a * b) → ¬(a / b + b / a ≥ 2)) ∧ 
    ((a / b + b / a ≥ 2) → (a^2 + b^2 ≥ 2 * a * b)) :=
sorry

end necessary_but_not_sufficient_condition_l631_631087


namespace no_number_x_satisfies_conditions_l631_631664

theorem no_number_x_satisfies_conditions :
  ∀ x : ℕ, x ≥ 1 → ¬ (184 % 5 = 4 ∧ 184 % 6 = 4 ∧ 184 = 184 ∧ 184 % 12 = 4 ∧ 184 % x = 4) := 
by
  intro x hx
  simp [Nat.mod_def] at hx
  sorry

end no_number_x_satisfies_conditions_l631_631664


namespace part1_part2_l631_631125

variables {α : Type*} [linear_ordered_field α]

-- Definitions for sets A, B, and C based on the problem statement
def A (x : α) : Prop := x^2 - 2*x - 3 ≤ 0
def B (x : α) : Prop := 2*x - 4 ≥ x - 2
def C (x : α) (a : α) : Prop := x ≥ a - 1

-- Statement for the first part: finding intersection A ∩ B
theorem part1 :
  { x : α | A x } ∩ { x : α | B x } = { x : α | 2 ≤ x ∧ x ≤ 3 } := 
by {
  sorry
}

-- Statement for the second part: finding the range of a given B ∪ C = C
theorem part2 (a : α) : 
  ({ x : α | B x } ∪ { x : α | C x a }) = { x : α | C x a } → a ≤ 3 := 
by {
  sorry
}

end part1_part2_l631_631125


namespace ratio_rate_down_to_up_l631_631352

theorem ratio_rate_down_to_up 
  (rate_up : ℝ) (time_up : ℝ) (distance_down : ℝ) (time_down_eq_time_up : time_down = time_up) :
  (time_up = 2) → 
  (rate_up = 3) →
  (distance_down = 9) → 
  (time_down = time_up) →
  (distance_down / time_down / rate_up = 1.5) :=
by
  intros h1 h2 h3 h4
  sorry

end ratio_rate_down_to_up_l631_631352


namespace positive_difference_eq_496_l631_631712

theorem positive_difference_eq_496 : 
  let a := 8 ^ 2 in 
  (a + a) / 8 - (a * a) / 8 = 496 :=
by
  let a := 8^2
  have h1 : (a + a) / 8 = 16 := by sorry
  have h2 : (a * a) / 8 = 512 := by sorry
  show (a + a) / 8 - (a * a) / 8 = 496 from by
    calc
      (a + a) / 8 - (a * a) / 8
            = 16 - 512 : by rw [h1, h2]
        ... = -496 : by ring
        ... = 496 : by norm_num

end positive_difference_eq_496_l631_631712


namespace positive_difference_l631_631667

theorem positive_difference :
  let a := 8^2
  let term1 := (a + a) / 8
  let term2 := (a * a) / 8
  term2 - term1 = 496 :=
by
  let a := 8^2
  let term1 := (a + a) / 8
  let term2 := (a * a) / 8
  have h1 : a = 64 := rfl
  have h2 : term1 = 16 := by simp [a, term1]
  have h3 : term2 = 512 := by simp [a, term2]
  show 512 - 16 = 496 from sorry

end positive_difference_l631_631667


namespace positive_difference_eq_496_l631_631707

theorem positive_difference_eq_496 : 
  let a := 8 ^ 2 in 
  (a + a) / 8 - (a * a) / 8 = 496 :=
by
  let a := 8^2
  have h1 : (a + a) / 8 = 16 := by sorry
  have h2 : (a * a) / 8 = 512 := by sorry
  show (a + a) / 8 - (a * a) / 8 = 496 from by
    calc
      (a + a) / 8 - (a * a) / 8
            = 16 - 512 : by rw [h1, h2]
        ... = -496 : by ring
        ... = 496 : by norm_num

end positive_difference_eq_496_l631_631707


namespace g_symmetric_about_pi_div_12_l631_631519

noncomputable def g (x : ℝ) : ℝ := 1/2 - Real.sin (2 * x + Real.pi / 3)

theorem g_symmetric_about_pi_div_12 :
  ∀ x : ℝ, g (2 * (Real.pi / 12) - x) = g x :=
sorry

end g_symmetric_about_pi_div_12_l631_631519


namespace color_regions_inequality_l631_631260

variables (a b n : ℕ)
variables (lambda : ℕ → ℕ) 
variables (P : ℕ) 

theorem color_regions_inequality
  (divided_by_lines : ∀ i j, i ≠ j → adjacent_list_i_j xor (adjacent_list_i = color_list_j))
  (number_of_red_regions : a)
  (number_of_blue_regions : b)
  (degree_of_vertex : lambda(P)) :
  a ≤ 2 * b - 2 - ∑ P, (lambda P - 2) :=
sorry

end color_regions_inequality_l631_631260


namespace smallest_positive_period_of_f_range_of_f_in_interval_l631_631477

noncomputable def f (x : ℝ) : ℝ :=
  cos (2 * x - real.pi) + 2 * sin (x - real.pi / 2) * sin (x + real.pi / 2)

theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ (∀ T' > 0, T' < T → ∃ x, f (x + T') ≠ f x) :=
sorry

theorem range_of_f_in_interval :
  ∀ x ∈ set.Icc (-real.pi) real.pi, f x ∈ set.Icc (-1.0) 1.0 :=
sorry

end smallest_positive_period_of_f_range_of_f_in_interval_l631_631477


namespace initial_workers_l631_631248

theorem initial_workers (W : ℕ) (H1 : (8 * W) / 30 = W) (H2 : (6 * (2 * W - 45)) / 45 = 2 * W - 45) : W = 45 :=
sorry

end initial_workers_l631_631248


namespace positive_difference_l631_631679

theorem positive_difference (a k : ℕ) (h1 : a = 8^2) (h2 : k = 8) :
  abs ((a + a) / k - (a * a) / k) = 496 :=
by
  sorry

end positive_difference_l631_631679


namespace ellipse_tangent_ratio_l631_631442

-- Definitions based on conditions
def ellipse (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) : set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ (x^2) / (a^2) + (y^2) / (b^2) = 1}

def is_tangent (P : ℝ × ℝ) (A B : ℝ × ℝ) (C : set (ℝ × ℝ)) : Prop :=
  ∃ line1 line2, line1.tangentAt P A C ∧ line2.tangentAt P B C

def moving_line_intersect (P Q M N : ℝ × ℝ) (C : set (ℝ × ℝ)) (AB : set (ℝ × ℝ)) : Prop :=
  ∃ l, passesThrough l P ∧ intersectsAt_two_points l C M N ∧ intersectsAt l AB Q

-- Theorem statement
theorem ellipse_tangent_ratio
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (C := ellipse a b a_pos b_pos)
  (P A B M N Q : ℝ × ℝ)
  (h1 : ¬C.contains P)
  (h2 : is_tangent P A B C)
  (h3 : moving_line_intersect P Q M N C (line_through A B)) :
  ∣dist P M / dist P N∣ = ∣dist Q M / dist Q N∣ :=
sorry

end ellipse_tangent_ratio_l631_631442


namespace discount_in_february_l631_631358

variable (C : ℝ)
def initial_selling_price := 1.20 * C
def second_selling_price := 1.25 * initial_selling_price C
def final_selling_price := 1.32 * C
def discount_percentage := 1 - final_selling_price / second_selling_price C

theorem discount_in_february : discount_percentage C = 0.12 :=
sorry

end discount_in_february_l631_631358


namespace markers_in_jenna_desk_l631_631587

theorem markers_in_jenna_desk (ratio_pens_pencils_markers : Nat × Nat × Nat)
  (num_pens : Nat) (h_ratio : ratio_pens_pencils_markers = (2, 2, 5))
  (h_num_pens : num_pens = 10) : (num_pens / ratio_pens_pencils_markers.1) * ratio_pens_pencils_markers.2.succ.succ = 25 := 
by
  sorry

end markers_in_jenna_desk_l631_631587


namespace number_of_solutions_proof_l631_631401

noncomputable def number_of_real_solutions (x y z w : ℝ) : ℝ :=
  if (x = z + w + 2 * z * w * x) ∧ (y = w + x + 2 * w * x * y) ∧ (z = x + y + 2 * x * y * z) ∧ (w = y + z + 2 * y * z * w) then
    5
  else
    0

theorem number_of_solutions_proof :
  ∃ x y z w : ℝ, x = z + w + 2 * z * w * x ∧ y = w + x + 2 * w * x * y ∧ z = x + y + 2 * x * y * z ∧ w = y + z + 2 * y * z * w → number_of_real_solutions x y z w = 5 :=
by
  sorry

end number_of_solutions_proof_l631_631401


namespace rationalize_denominator_to_find_constants_l631_631237

-- Definitions of the given conditions
def original_fraction := 3 / (4 * Real.sqrt 7 + 3 * Real.sqrt 13)
def simplified_fraction (A B C D E : ℤ) := (A * Real.sqrt B + C * Real.sqrt D) / E

-- Statement of the proof problem
theorem rationalize_denominator_to_find_constants :
  ∃ (A B C D E : ℤ),
    original_fraction = simplified_fraction A B C D E ∧
    B < D ∧
    (∀ p : ℕ, Real.sqrt (p * p) = p) ∧ -- Ensuring that all radicals are in simplest form
    A + B + C + D + E = 22 :=
sorry

end rationalize_denominator_to_find_constants_l631_631237


namespace amount_to_pay_for_goods_l631_631739

noncomputable def original_price := 6650 : ℝ
noncomputable def rebate_percentage := 6 / 100 : ℝ
noncomputable def sales_tax_percentage := 10 / 100 : ℝ

noncomputable def rebate_amount := rebate_percentage * original_price
noncomputable def price_after_rebate := original_price - rebate_amount
noncomputable def sales_tax := sales_tax_percentage * price_after_rebate
noncomputable def total_amount_to_pay := price_after_rebate + sales_tax

theorem amount_to_pay_for_goods : total_amount_to_pay = 6876.10 := by
  sorry

end amount_to_pay_for_goods_l631_631739


namespace find_k_l631_631082

theorem find_k (a : ℕ → ℕ) (S : ℕ → ℕ) (k : ℕ) 
  (h_nz : ∀ n, S n = n ^ 2 - a n) 
  (hSk : 1 < S k ∧ S k < 9) :
  k = 2 := 
sorry

end find_k_l631_631082


namespace insects_in_lab_l631_631019

theorem insects_in_lab (total_legs number_of_legs_per_insect : ℕ) (h1 : total_legs = 36) (h2 : number_of_legs_per_insect = 6) : (total_legs / number_of_legs_per_insect) = 6 :=
by
  sorry

end insects_in_lab_l631_631019


namespace largest_amount_not_given_l631_631945

theorem largest_amount_not_given (n : ℕ) (hn : 0 < n) :
  ∃ m : ℕ, ∀ k ∈ Ico 0 m, ¬ ∃ a b c d : ℕ, k = a * (3 * n - 2) + b * (6 * n - 1) + 
  c * (6 * n + 2) + d * (6 * n + 5) ∧ m = 6 * n^2 - 4 * n - 3 := sorry

end largest_amount_not_given_l631_631945


namespace number_of_other_people_in_house_excluding_james_l631_631963

noncomputable def coffee_consumption_per_person_per_day := 2 * 0.5  -- ounces

noncomputable def coffee_cost_per_ounce := 1.25  -- dollars

noncomputable def total_spent_per_week := 35  -- dollars

theorem number_of_other_people_in_house_excluding_james : 
  let total_ounces_per_week := total_spent_per_week / coffee_cost_per_ounce,
      ounces_per_person_per_week := coffee_consumption_per_person_per_day * 7,
      total_people := total_ounces_per_week / ounces_per_person_per_week
  in total_people - 1 = 3 :=
by
  sorry

end number_of_other_people_in_house_excluding_james_l631_631963


namespace intersection_points_l631_631259

-- Definitions of the parametric equations of the line and curve
def parametric_line (t : ℝ) : ℝ × ℝ := (2 + t, -1 - t)
def parametric_curve (α : ℝ) : ℝ × ℝ := (3 * Real.cos α, 3 * Real.sin α)

-- General forms of the line and the curve
def general_line (x y : ℝ) := x + y = 1
def general_curve (x y : ℝ) := x^2 + y^2 = 9

-- The main theorem to prove
theorem intersection_points : ∃ (p1 p2 : ℝ × ℝ), 
  (general_line p1.1 p1.2) ∧ (general_curve p1.1 p1.2) ∧ 
  (general_line p2.1 p2.2) ∧ (general_curve p2.1 p2.2) ∧ 
  p1 ≠ p2 :=
by
  -- The actual proof would go here
  sorry

end intersection_points_l631_631259


namespace dr_jones_remaining_money_l631_631048

-- Define the conditions
def monthly_earnings : ℕ := 6000
def house_rental : ℕ := 640
def food_expense : ℕ := 380
def electric_water_bill (earnings : ℕ) : ℕ := earnings / 4
def insurances (earnings : ℕ) : ℕ := earnings / 5

-- Prove the final amount left
theorem dr_jones_remaining_money :
  let earnings := monthly_earnings,
      total_expenses := house_rental + food_expense + electric_water_bill earnings + insurances earnings,
      remaining_money := earnings - total_expenses
  in
  remaining_money = 2280 :=
by
  let earnings := monthly_earnings
  let total_expenses := house_rental + food_expense + electric_water_bill earnings + insurances earnings
  let remaining_money := earnings - total_expenses
  show remaining_money = 2280 from sorry

end dr_jones_remaining_money_l631_631048


namespace tan_pi_minus_alpha_l631_631818

theorem tan_pi_minus_alpha (α : ℝ) (h : 3 * Real.sin α = Real.cos α) : Real.tan (π - α) = -1 / 3 :=
by
  sorry

end tan_pi_minus_alpha_l631_631818


namespace decimal_digit_101st_place_l631_631497

theorem decimal_digit_101st_place (n : ℕ) (h : n = 101) : 
  (∀ m, (m = 7 / 26) → (decimal_digit m n) = 3) := 
begin
  intros m hm,
  have repeating_block : ℕ → ℕ := λ k, match (k % 6) with
    | 0 := 2
    | 1 := 6
    | 2 := 9
    | 3 := 2
    | 4 := 3
    | 5 := 0
    end,
  have one_digit := (repeating_block (101 - 1) % 6),
  have digit := match one_digit with
    | 0 := 2
    | 1 := 6
    | 2 := 9
    | 3 := 2
    | 4 := 3
    | 5 := 0
    end,
  exact digit = 3,
end

end decimal_digit_101st_place_l631_631497


namespace max_f_is_2_l631_631258

def f (x : ℝ) : ℝ := 
  Real.sin (x + π / 3) + Real.cos (x - π / 6)

theorem max_f_is_2 : ∃ x : ℝ, ∀ y : ℝ, f y ≤ 2 :=
by
  -- using the maximum principle of trigonometric functions
  let fx := fun x => Real.sin (x + π / 3) + Real.cos (x - π / 6)
  show ∃ x : ℝ, ∀ y : ℝ, fx y ≤ 2
  sorry

end max_f_is_2_l631_631258


namespace digit_in_101st_place_l631_631499

theorem digit_in_101st_place :
  let repeating_sequence := "269230769"
  in string.nth repeating_sequence ((101 % 9) - 1) = '6' :=
by
  sorry

end digit_in_101st_place_l631_631499


namespace probability_of_B_given_A_l631_631157

noncomputable def balls_in_box : Prop :=
  let total_balls := 12
  let yellow_balls := 5
  let blue_balls := 4
  let green_balls := 3
  let event_A := (yellow_balls * green_balls + yellow_balls * blue_balls + green_balls * blue_balls) / (total_balls * (total_balls - 1) / 2)
  let event_B := (yellow_balls * blue_balls) / (total_balls * (total_balls - 1) / 2)
  (event_B / event_A) = 20 / 47

theorem probability_of_B_given_A : balls_in_box := sorry

end probability_of_B_given_A_l631_631157


namespace positive_difference_is_496_l631_631717

def square (n: ℕ) : ℕ := n * n
def term1 := (square 8 + square 8) / 8
def term2 := (square 8 * square 8) / 8
def positive_difference := abs (term2 - term1)

theorem positive_difference_is_496 : positive_difference = 496 :=
by
  -- This is where the proof would go
  sorry

end positive_difference_is_496_l631_631717


namespace quad_ak_eq_kd_bc_kd_l631_631629

variable {A B C D K : Type}
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace K]

variable {AB AD CD BC KD AK : ℝ}
variable (a₁ : AB = AD)
variable (a₂ : ∠ D AK = ∠ AB D)

theorem quad_ak_eq_kd_bc_kd (h : AB = AD) (h2 : ∠ D AK = ∠ AB D) : AK^2 = KD^2 + BC * KD := sorry

end quad_ak_eq_kd_bc_kd_l631_631629


namespace area_of_quadrilateral_l631_631165

-- Conditions given in the problem
variables {A B C D : ℝ^2}
variable h1 : (fin.vec2 ℝ) (B - D) 2
variable h2 : (fin.vec2 ℝ) ((A - C) ⬝ (B - D)) 0
variable h3 : (fin.vec2 ℝ) ((A - B + D - C) ⬝ (B - C + A - D)) 5

-- Proof statement to show that the area of quadrilateral is 3
theorem area_of_quadrilateral (A B C D : ℝ^2) 
  (h1 : (fin.vec2 ℝ) (B - D) 2)
  (h2 : (fin.vec2 ℝ) ((A - C) ⬝ (B - D)) 0)
  (h3 : (fin.vec2 ℝ) ((A - B + D - C) ⬝ (B - C + A - D)) 5) :
  let AC := (A - C).norm,
      BD := (B - D).norm in
  1/2 * AC * BD = 3 := 
begin
  sorry
end

end area_of_quadrilateral_l631_631165


namespace positive_difference_l631_631675

theorem positive_difference (a k : ℕ) (h1 : a = 8^2) (h2 : k = 8) :
  abs ((a + a) / k - (a * a) / k) = 496 :=
by
  sorry

end positive_difference_l631_631675


namespace tan_435_eq_2_add_sqrt_3_l631_631397

theorem tan_435_eq_2_add_sqrt_3 :
  Real.tan (435 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  sorry

end tan_435_eq_2_add_sqrt_3_l631_631397


namespace part1_next_term_part2_next_term_even_part2_next_term_odd_l631_631060

/-
Part 1: Prove that given the sequence {2, 7, 12, 17, 22} with each consecutive pair having a difference of 5, the next term is 27.
-/
theorem part1_next_term : 
  ∃ n : ℕ, n = 27 ∧ 
  ∀ a b ∈ ({2, 7, 12, 17, 22} : List ℕ), b - a = 5 → n = b + 5  :=
by
  sorry

/-
Part 2: Two proofs needed:
- Prove that the next term for the pattern {2, 5, 11, 23} in even-indexed positions is 47.
- Prove that the next term for the pattern {4, 10, 22} in odd-indexed positions is 46.
-/
theorem part2_next_term_even :
  ∀ n, n ∈ ({2, 5, 11, 23} : List ℕ) → n * 2 + 1 = 47 :=
by
  sorry

theorem part2_next_term_odd :
  ∀ n, n ∈ ({4, 10, 22} : List ℕ) → n * 2 + 2 = 46 :=
by
  sorry

end part1_next_term_part2_next_term_even_part2_next_term_odd_l631_631060


namespace beacon_population_l631_631608

variables (Richmond Victoria Beacon : ℕ)

theorem beacon_population :
  (Richmond = Victoria + 1000) →
  (Victoria = 4 * Beacon) →
  (Richmond = 3000) →
  (Beacon = 500) :=
by
  intros h1 h2 h3
  sorry

end beacon_population_l631_631608


namespace rationalize_denominator_l631_631239

theorem rationalize_denominator :
  let A := -12
  let B := 7
  let C := 9
  let D := 13
  let E := 5
  A + B + C + D + E = 22 :=
by
  -- Proof goes here
  sorry

end rationalize_denominator_l631_631239


namespace initial_marbles_count_l631_631409

-- Define the conditions
def marbles_given_to_mary : ℕ := 14
def marbles_remaining : ℕ := 50

-- Prove that Dan's initial number of marbles is 64
theorem initial_marbles_count : marbles_given_to_mary + marbles_remaining = 64 := 
by {
  sorry
}

end initial_marbles_count_l631_631409


namespace arithmetic_sequence_term_2011_is_671st_l631_631335

theorem arithmetic_sequence_term_2011_is_671st:
  ∀ (a1 d n : ℕ), a1 = 1 → d = 3 → (3 * n - 2 = 2011) → n = 671 :=
by 
  intros a1 d n ha1 hd h_eq;
  sorry

end arithmetic_sequence_term_2011_is_671st_l631_631335


namespace count_numbers_with_digit_7_in_range_l631_631873

theorem count_numbers_with_digit_7_in_range : 
  let numbers_in_range := {n : ℕ | 1 ≤ n ∧ n ≤ 800}
      contains_digit_7 (n : ℕ) : Prop := n.digits 10.contains 7
  in (finset.filter (λ n, contains_digit_7 n) (finset.range 801)).card = 152 :=
by 
  let numbers_in_range := {n : ℕ | 1 ≤ n ∧ n ≤ 800}
  let contains_digit_7 (n : ℕ) : Prop := n.digits 10.contains 7
  have h := (finset.filter (λ n, contains_digit_7 n) (finset.range 801)).card
  sorry

end count_numbers_with_digit_7_in_range_l631_631873


namespace volume_ratio_of_cone_and_prism_l631_631766

theorem volume_ratio_of_cone_and_prism (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) : 
  let V_cone := (1/3) * π * r^2 * h;
      V_prism := 12 * r^2 * h
  in V_cone / V_prism = π / 36 :=
by
  -- Introduce definitions for V_cone and V_prism
  let V_cone := (1/3) * π * r^2 * h;
  let V_prism := 12 * r^2 * h;
  -- Provide the proof using the calculations from the solution
  sorry

end volume_ratio_of_cone_and_prism_l631_631766


namespace simplify_expression_l631_631507

theorem simplify_expression (x y : ℝ) (h : x - 3 * y = 4) : (x - 3 * y) ^ 2 + 2 * x - 6 * y - 10 = 14 := by
  sorry

end simplify_expression_l631_631507


namespace total_time_proof_l631_631654

/-- Define the effective speed of Tyson in a lake based on the temperature condition. -/
def lake_speed (temp_below_60 : Bool) : ℝ :=
  if temp_below_60 then 2.8 + 0.5 else 3 + 0.5

/-- Define the effective speed of Tyson in an ocean based on the wind condition. -/
def ocean_speed (strong_wind : Bool) : ℝ :=
  if strong_wind then 2.2 + 0.7 else 2.5 + 0.7

/-- Define the time taken for each race given the distance and effective speed. -/
def race_time (distance speed : ℝ) : ℝ :=
  distance / speed

/-- List of distances for lake races and their corresponding temperature conditions. -/
def lake_races : List (ℝ × Bool) := [(2, false), (3.2, true), (3.5, false), (1.8, true), (4, false)]

/-- List of distances for ocean races and their corresponding wind conditions. -/
def ocean_races : List (ℝ × Bool) := [(2.5, false), (3.5, true), (2, true), (3.7, true), (4.2, false)]

/-- Calculate total time for races given distances and conditions. -/
def total_race_time : ℝ :=
  let lake_times := lake_races.map (λ (d_c : ℝ × Bool), race_time d_c.1 (lake_speed d_c.2))
  let ocean_times := ocean_races.map (λ (d_c : ℝ × Bool), race_time d_c.1 (ocean_speed d_c.2))
  lake_times.sum + ocean_times.sum

/-- Prove that the total time Tyson spent in his races is 9.4958 hours given the conditions. -/
theorem total_time_proof : total_race_time = 9.4958 :=
  by
  sorry

end total_time_proof_l631_631654


namespace lantern_problem_l631_631623

section LanternProfit

-- Defining unit price of type A and type B lanterns according to given conditions.
variables (x : ℕ) (y : ℕ)
def cost_A_lantern : Prop := x = 26
def cost_B_lantern : Prop := y = 35

-- Defining quantity condition based on given constraints.
def lantern_quantity_condition : Prop :=
  (3120 / x = 4200 / (x + 9))

-- Defining initial conditions for profit calculation.
def initial_conditions (P_initial : ℕ) (Q_initial : ℕ) (dP : ℕ) (dQ : ℕ) : Prop :=
  P_initial = 50 ∧ Q_initial = 98 ∧ dP = 1 ∧ dQ = 2

-- Defining the profit function.
def profit_function (P_initial : ℕ) (Q_initial : ℕ) (dP : ℕ) (dQ : ℕ) (x : ℕ) : ℕ :=
  (P_initial + x - 35) * (Q_initial - dQ * x)

-- Defining the analytic expression of the profit function.
def analytic_expression (x : ℕ) : ℤ :=
  -2 * x^2 + 68 * x + 1470

-- Proving the profit maximization condition.
def maximize_profit (P_max : ℕ) : Prop :=
  P_max = 65 ∧ analytic_expression 15 = 2040

-- Theorem combining all the parts
theorem lantern_problem :
  ∃ x y, cost_A_lantern x ∧ cost_B_lantern y ∧ lantern_quantity_condition x ∧
  initial_conditions 50 98 1 2 ∧ maximize_profit 65 :=
by
  existsi 26
  existsi 35
  split
  exact rfl
  split
  exact rfl
  split
  sorry
  split
  exact ⟨rfl, rfl, rfl, rfl⟩
  exact ⟨rfl, rfl⟩
  
end LanternProfit

end lantern_problem_l631_631623


namespace base_number_unique_l631_631516

theorem base_number_unique (y : ℕ) : (3 : ℝ) ^ 16 = (9 : ℝ) ^ y → y = 8 → (9 : ℝ) = 3 ^ (16 / y) :=
by
  sorry

end base_number_unique_l631_631516


namespace license_plate_count_l631_631158

def license_plate_combinations : Nat :=
  26 * Nat.choose 25 2 * Nat.choose 4 2 * 720

theorem license_plate_count :
  license_plate_combinations = 33696000 :=
by
  unfold license_plate_combinations
  sorry

end license_plate_count_l631_631158


namespace edward_spent_money_l631_631050

-- Definitions based on the conditions
def books := 2
def cost_per_book := 3

-- Statement of the proof problem
theorem edward_spent_money : 
  (books * cost_per_book = 6) :=
by
  -- proof goes here
  sorry

end edward_spent_money_l631_631050


namespace imaginary_part_of_conjugate_l631_631824

-- Given a complex number z
def z : ℂ := -2 + 2 * Complex.I

-- The proof problem stating that imaginary part of the conjugate of z is -2
theorem imaginary_part_of_conjugate (z : ℂ) (hz : z = -2 + 2 * Complex.I) : Complex.imag (Complex.conj z) = -2 :=
by
  rw hz
  rw Complex.conj
  rw Complex.imag
  -- Additional steps
  sorry

end imaginary_part_of_conjugate_l631_631824


namespace positive_difference_l631_631669

theorem positive_difference :
  let a := 8^2
  let term1 := (a + a) / 8
  let term2 := (a * a) / 8
  term2 - term1 = 496 :=
by
  let a := 8^2
  let term1 := (a + a) / 8
  let term2 := (a * a) / 8
  have h1 : a = 64 := rfl
  have h2 : term1 = 16 := by simp [a, term1]
  have h3 : term2 = 512 := by simp [a, term2]
  show 512 - 16 = 496 from sorry

end positive_difference_l631_631669


namespace arccos_neg_one_eq_pi_l631_631032

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi := 
by
  sorry

end arccos_neg_one_eq_pi_l631_631032


namespace sequence_properties_l631_631195

-- Definitions for initial values of the sequence
def u : ℕ → ℕ
| 1 := 1
| 2 := 2
| 3 := 24
| (n + 1) := if (h : n ≥ 3) then 
               (6 * u n * u (n - 2) - 8 * u (n - 1) ^ 2) / (u (n - 1) * u (n - 2)) 
             else 
               0 -- This case technically won't happen as we always have n ≥ 1

-- Statement of the proof problem to be solved in Lean 4
theorem sequence_properties :
  (∀ n ≥ 1, (u n) ∈ Set.Nat) ∧ (∀ n ≥ 1, n ∣ (u n)) :=
by
  sorry

end sequence_properties_l631_631195


namespace sin_three_pi_over_two_plus_alpha_l631_631109

noncomputable def point_to_origin_distance (x y : ℤ) : ℝ := real.sqrt (x^2 + y^2)

theorem sin_three_pi_over_two_plus_alpha 
  (α : ℝ)
  (P : ℤ × ℤ)
  (hP : P = (-5, -12))
  (origin_dist : point_to_origin_distance (P.1) (P.2) = 13)
  : sin (3 * real.pi / 2 + α) = 5 / 13 :=
  sorry

end sin_three_pi_over_two_plus_alpha_l631_631109


namespace hundred_and_first_digit_l631_631494

/-- The repeating sequence for the decimal expansion of 7/26 -/
def repeating_sequence : List ℕ := [2, 6, 9, 2, 3, 0, 7, 6, 9]

/-- The length of the repeating sequence is 9 -/
def period : ℕ := 9

/-- The function to get the n-th digit of the decimal expansion of 7/26 -/
def nth_digit (n : ℕ) : ℕ :=
  repeating_sequence.get ⟨n % period, n.mod_lt (by norm_num)⟩

/-- The 101st digit of the decimal expansion of 7/26 is 6 -/
theorem hundred_and_first_digit : nth_digit 101 = 6 := by
  sorry

end hundred_and_first_digit_l631_631494


namespace quad_ak_eq_kd_bc_kd_l631_631630

variable {A B C D K : Type}
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace K]

variable {AB AD CD BC KD AK : ℝ}
variable (a₁ : AB = AD)
variable (a₂ : ∠ D AK = ∠ AB D)

theorem quad_ak_eq_kd_bc_kd (h : AB = AD) (h2 : ∠ D AK = ∠ AB D) : AK^2 = KD^2 + BC * KD := sorry

end quad_ak_eq_kd_bc_kd_l631_631630


namespace find_a_l631_631576

def f (x : ℝ) : ℝ := 
  if x ≤ 0 then -x 
  else x^2

theorem find_a (a : ℝ) : f a = 4 ↔ a = -4 ∨ a = 2 :=
by
  sorry

end find_a_l631_631576


namespace john_discount_percentage_l631_631551

-- Defining the given constants
def cost_to_fix_car : ℝ := 20000
def kept_percentage : ℝ := 0.90
def prize_money : ℝ := 70000
def money_made : ℝ := 47000

-- Calculating intermediate values
def kept_money : ℝ := kept_percentage * prize_money
def cost_after_discount : ℝ := kept_money - money_made
def discount_amount : ℝ := cost_to_fix_car - cost_after_discount
def discount_percentage : ℝ := (discount_amount / cost_to_fix_car) * 100

-- The proof problem
theorem john_discount_percentage : discount_percentage = 20 := by
  -- Proof is skipped using sorry
  sorry

end john_discount_percentage_l631_631551


namespace vehicle_count_l631_631569

theorem vehicle_count
    (D H K Ho F : ℕ)
    (h1 : D + H + K + Ho + F = 1000)
    (h2 : 0.35 * (D + H + K + Ho + F) = D)
    (h3 : 0.10 * (D + H + K + Ho + F) = H)
    (h4 : K = 2 * Ho + 50)
    (h5 : F = D - 200) :
    D = 350 ∧ H = 100 ∧ K = 283 ∧ Ho = 117 ∧ F = 150 :=
by
  sorry

end vehicle_count_l631_631569


namespace pentagonal_country_50_cities_has_125_routes_no_pentagonal_country_with_46_routes_l631_631751

-- Defining the concept of a pentagonal country
def is_pentagonal_country (n : ℕ) (routes : ℕ) : Prop :=
  routes = (5 * n) / 2

-- Part (b): Proving that a country with 50 cities has 125 air routes
theorem pentagonal_country_50_cities_has_125_routes :
  is_pentagonal_country 50 125 :=
by
  unfold is_pentagonal_country
  norm_num

-- Part (c): Proving that it's impossible to have a pentagonal country with 46 air routes
theorem no_pentagonal_country_with_46_routes : 
  ∀ n : ℕ, ¬ is_pentagonal_country n 46 :=
by
  intro n
  unfold is_pentagonal_country
  intro h
  have : 92 = 5 * n := by linarith
  -- As 92 isn't divisible by 5, such n doesn't exist
  norm_num at this
  linarith

end pentagonal_country_50_cities_has_125_routes_no_pentagonal_country_with_46_routes_l631_631751
