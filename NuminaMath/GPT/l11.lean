import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Basic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Trigonometry.Basic
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Gcd
import Mathlib.Data.Int.Parity
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Time
import Mathlib.NumberTheory.Prime
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Lint.Library
import Mathlib.Topology.Algebra.Order.IntermediateValue
import Real

namespace solution_set_for_inequality_l11_11860

theorem solution_set_for_inequality 
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_mono_dec : ∀ x y, 0 < x → x < y → f y ≤ f x)
  (h_f2 : f 2 = 0) :
  {x : ℝ | f x ≥ 0} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end solution_set_for_inequality_l11_11860


namespace union_M_N_l11_11951

open Set Classical

noncomputable def M : Set ℝ := {x | x^2 = x}
noncomputable def N : Set ℝ := {x | Real.log x ≤ 0}

theorem union_M_N : M ∪ N = Icc 0 1 := by
  sorry

end union_M_N_l11_11951


namespace number_of_times_each_player_plays_l11_11283

def players : ℕ := 7
def total_games : ℕ := 42

theorem number_of_times_each_player_plays (x : ℕ) 
  (H1 : 42 = (players * (players - 1) * x) / 2) : x = 2 :=
by
  sorry

end number_of_times_each_player_plays_l11_11283


namespace prime_q_with_decimal_expansion_length_l11_11891

theorem prime_q_with_decimal_expansion_length {p q : ℕ} 
  (hp : Nat.Prime p) (hq : Nat.Prime q) (h1 : 2 * p + 5 = q) (h2 : orderOf 10 q = 166) : 
  q = 167 :=
by
  sorry

end prime_q_with_decimal_expansion_length_l11_11891


namespace first_digit_after_decimal_sqrt_l11_11182

theorem first_digit_after_decimal_sqrt (n : ℕ) (h : n > 0) :
  ∃ (r : ℕ), r ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  ∃ (k : ℕ), k + (r / 10 : ℝ) ≤ (n + 1) * real.sqrt n ∧ (n + 1) * real.sqrt n < k + ((r + 1) / 10 : ℝ) :=
sorry

end first_digit_after_decimal_sqrt_l11_11182


namespace smallest_number_divisible_11_with_remainder_3_eq_5043_l11_11250

noncomputable def smallest_number_divisible_11_with_remainder_3 :=
  let lcm_2_to_8 := Nat.lcm (Nat.lcm 2 3) (Nat.lcm (Nat.lcm 4 5) (Nat.lcm (Nat.lcm 6 7) 8)) in
  let n := fun k => lcm_2_to_8 * k + 3 in
  ∃ k, (11 ∣ n k) ∧ (n k = 5043)

-- Now we restate the proposition in Lean's theorem format.
theorem smallest_number_divisible_11_with_remainder_3_eq_5043 :
  ∃ k, 11 ∣ (840 * k + 3) ∧ (840 * k + 3 = 5043) :=
by
  sorry

end smallest_number_divisible_11_with_remainder_3_eq_5043_l11_11250


namespace probability_of_one_or_two_l11_11724

/-- Represents the number of elements in the first 20 rows of Pascal's Triangle. -/
noncomputable def total_elements : ℕ := 210

/-- Represents the number of ones in the first 20 rows of Pascal's Triangle. -/
noncomputable def number_of_ones : ℕ := 39

/-- Represents the number of twos in the first 20 rows of Pascal's Triangle. -/
noncomputable def number_of_twos : ℕ :=18

/-- Prove that the probability of randomly choosing an element which is either 1 or 2
from the first 20 rows of Pascal's Triangle is 57/210. -/
theorem probability_of_one_or_two (h1 : total_elements = 210)
                                  (h2 : number_of_ones = 39)
                                  (h3 : number_of_twos = 18) :
    39 + 18 = 57 ∧ (57 : ℚ) / 210 = 57 / 210 :=
by {
    sorry
}

end probability_of_one_or_two_l11_11724


namespace sum_mod_7_l11_11247

theorem sum_mod_7 (n : ℕ) (h : n = 203) : 
  (∑ i in Finset.range (n + 1), i) % 7 = 0 := 
by 
  have h_groups : ∑ i in Finset.range (7), i % 7 = 0 :=
    by norm_num  
  have h_complete_groups : (203 / 7).floor = 29 := 
    by norm_num
  sorry

end sum_mod_7_l11_11247


namespace sufficient_but_not_necessary_l11_11276

theorem sufficient_but_not_necessary (x : ℝ) : (x^2 = 9 → x = 3) ∧ (¬(x^2 = 9 → x = 3 ∨ x = -3)) :=
by
  sorry

end sufficient_but_not_necessary_l11_11276


namespace solve_fractional_equation_l11_11222

theorem solve_fractional_equation : ∀ x : ℝ, (2 * x + 1) / 5 - x / 10 = 2 → x = 6 :=
by
  intros x h
  sorry

end solve_fractional_equation_l11_11222


namespace circle_divides_CD_in_ratio_l11_11930

variable (A B C D : Point)
variable (BC a : ℝ)
variable (AD : ℝ := (1 + Real.sqrt 15) * BC)
variable (radius : ℝ := (2 / 3) * BC)
variable (EF : ℝ := (Real.sqrt 7 / 3) * BC)
variable (is_isosceles_trapezoid : is_isosceles_trapezoid A B C D)
variable (circle_centered_at_C : circle_centered_at C radius)
variable (chord_EF : chord_intersects_base EF AD)

theorem circle_divides_CD_in_ratio (CD DK KC : ℝ) (H1 : CD = 2 * a)
  (H2 : DK + KC = CD) (H3 : KC = CD - DK) : DK / KC = 2 :=
sorry

end circle_divides_CD_in_ratio_l11_11930


namespace garden_roller_length_l11_11196

theorem garden_roller_length (d : ℝ) (A : ℝ) (pi : ℝ) (L : ℝ) (r : ℝ) 
  (h_d : d = 1.4)
  (h_A : A = 66 / 5)
  (h_pi : pi = 22 / 7)
  (h_r : r = d / 2)
  (h_area_eq : A = 2 * pi * r * L) :
  L = 2.1 :=
by {
  subst h_d,
  subst h_A,
  subst h_pi,
  subst h_r,
  sorry
}

end garden_roller_length_l11_11196


namespace hyperbola_eccentricity_sqrt2_l11_11422

noncomputable def hyperbola_eccentricity {a b : ℝ} (ha_pos : a > 0) (hb_pos : b > 0) 
  (intersect_condition : ∃ P m, P = (m, Real.sqrt m) ∧ m > 0 ∧ 
    ((Real.sqrt m) / (m + 2) = 1 / (2 * Real.sqrt m))) 
  (c_def : c = 2) 
  (hyperbola_cond : ∃ x y, (x^2 / a^2 - y^2 / b^2 = 1) ∧ (x, y) = (2, Real.sqrt 2))
  (c2_eq : c^2 = a^2 + b^2) :
  real := 
begin
  have a_eq_b : a = b,
  { sorry },
  exact c / a,
end

theorem hyperbola_eccentricity_sqrt2 (a b : ℝ) (ha_pos : a > 0) (hb_pos : b > 0) 
  (intersect_condition : ∃ P m, P = (m, Real.sqrt m) ∧ m > 0 ∧ 
    ((Real.sqrt m) / (m + 2) = 1 / (2 * Real.sqrt m))) 
  (c_def : c = 2) 
  (hyperbola_cond : ∃ x y, (x^2 / a^2 - y^2 / b^2 = 1) ∧ (x, y) = (2, Real.sqrt 2))
  (c2_eq : c^2 = a^2 + b^2) :
  hyperbola_eccentricity ha_pos hb_pos intersect_condition c_def hyperbola_cond c2_eq = Real.sqrt 2 :=
sorry

end hyperbola_eccentricity_sqrt2_l11_11422


namespace fraction_passengers_from_Asia_l11_11067

theorem fraction_passengers_from_Asia (P : ℝ) (hP : P = 120) :=
  let A := 1 / 6 in
  (1 / 12) * P + (1 / 8) * P + (1 / 3) * P + A * P + 35 = P :=
by
  have h1 : (1 / 12) * 120 = 10 := by norm_num
  have h2 : (1 / 8) * 120 = 15 := by norm_num
  have h3 : (1 / 3) * 120 = 40 := by norm_num
  have h4 : A * 120 = 20 := by norm_num
  have h5 : 10 + 15 + 40 + 20 + 35 = 120 := by norm_num
  exact calc 10 + 15 + 40 + 20 + 35 = 120 : by norm_num

end fraction_passengers_from_Asia_l11_11067


namespace equilateral_collinear_l11_11553

variables (A1 A2 A3 B1 B2 B3 C1 C2 C3 : Type*) [Groupoid A1] [Groupoid A2] [Groupoid A3] 
          [Groupoid B1] [Groupoid B2] [Groupoid B3] [Groupoid C1] [Groupoid C2] [Groupoid C3]
          (line : A1 → A2 → Set A3)

-- Defining equilateral triangle and midpoints
def equilateral_triangle (A1 A2 A3 : Type*) : Prop := 
  (dist A1 A2 = dist A2 A3) ∧ (dist A2 A3 = dist A3 A1)

def midpoints (B3 B1 B2 : Type*) (A1 A2 A3 : Type*) : Prop :=
  (midpoint B1 A1 A2) ∧ (midpoint B2 A2 A3) ∧ (midpoint B3 A3 A1)

-- Defining collinearity
def collinear (X Y Z : Type*) : Prop :=
  ∃ (l : Set Type*), X ∈ l ∧ Y ∈ l ∧ Z ∈ l

axiom midpoint (X Y Z : Type*) : Prop -- axiomatizing the concept of midpoint
axiom dist : A1 → A2 → Type* -- axiomatizing distance concept

theorem equilateral_collinear (A1 A2 A3 B1 B2 B3 C1 C2 C3 : Type*) :
  equilateral_triangle A1 A2 A3 → 
  midpoints B3 B1 B2 A1 A2 A3 → 
  collinear A1 C1 C2 ∧ collinear A2 C2 C3 ∧ collinear A3 C3 C1 :=
begin
  sorry -- proof to be filled in
end

end equilateral_collinear_l11_11553


namespace arithmetic_sequence_common_difference_l11_11223

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 1 = 3)
  (h2 : S 5 = 35)
  (h3 : ∀ n, S n = n * a 1 + n * (n - 1) / 2 * d) :
  d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l11_11223


namespace range_s_squared_minus_c_squared_l11_11558

variable {x y : ℝ}
def r : ℝ := real.sqrt (x^2 + y^2)

def s : ℝ := y / r
def c : ℝ := x / r

theorem range_s_squared_minus_c_squared : -1 ≤ s^2 - c^2 ∧ s^2 - c^2 ≤ 1 :=
by
  sorry

end range_s_squared_minus_c_squared_l11_11558


namespace pascal_triangle_probability_l11_11735

-- Define the probability problem in Lean 4
theorem pascal_triangle_probability :
  let total_elements := ((20 * (20 + 1)) / 2)
  let ones_count := (1 + 2 * 19)
  let twos_count := (2 * (19 - 2 + 1))
  (ones_count + twos_count) / total_elements = 5 / 14 :=
by
  let total_elements := ((20 * (20 + 1)) / 2)
  let ones_count := (1 + 2 * 19)
  let twos_count := (2 * (19 - 2 + 1))
  have h1 : total_elements = 210 := by sorry
  have h2 : ones_count = 39 := by sorry
  have h3 : twos_count = 36 := by sorry
  have h4 : (39 + 36) / 210 = 5 / 14 := by sorry
  exact h4

end pascal_triangle_probability_l11_11735


namespace find_b_l11_11453

theorem find_b (a b : ℝ) (B C : ℝ)
    (h1 : a * b = 60 * Real.sqrt 3)
    (h2 : Real.sin B = Real.sin C)
    (h3 : 15 * Real.sqrt 3 = 1/2 * a * b * Real.sin C) :
  b = 2 * Real.sqrt 15 :=
sorry

end find_b_l11_11453


namespace proof_triangle_properties_l11_11495

variable (A B C : ℝ)
variable (h AB : ℝ)

-- Conditions
def triangle_conditions : Prop :=
  (A + B = 3 * C) ∧ (2 * Real.sin (A - C) = Real.sin B) ∧ (AB = 5)

-- Part 1: Proving sin A
def find_sin_A (h₁ : triangle_conditions A B C h AB) : Prop :=
  Real.sin A = 3 * Real.cos A

-- Part 2: Proving the height on side AB
def find_height_on_AB (h₁ : triangle_conditions A B C h AB) : Prop :=
  h = 6

-- Combined proof statement
theorem proof_triangle_properties (h₁ : triangle_conditions A B C h AB) : 
  find_sin_A A B C h₁ ∧ find_height_on_AB A B C h AB h₁ := 
  by sorry

end proof_triangle_properties_l11_11495


namespace percentage_of_alcohol_in_vessel_Q_l11_11640

theorem percentage_of_alcohol_in_vessel_Q
  (x : ℝ)
  (h_mix : 2.5 + 0.04 * x = 6) :
  x = 87.5 :=
by
  sorry

end percentage_of_alcohol_in_vessel_Q_l11_11640


namespace proof_moles_HNO3_proof_molecular_weight_HNO3_l11_11327

variable (n_CaO : ℕ) (molar_mass_H : ℕ) (molar_mass_N : ℕ) (molar_mass_O : ℕ)

def verify_moles_HNO3 (n_CaO : ℕ) : ℕ :=
  2 * n_CaO

def verify_molecular_weight_HNO3 (molar_mass_H molar_mass_N molar_mass_O : ℕ) : ℕ :=
  molar_mass_H + molar_mass_N + 3 * molar_mass_O

theorem proof_moles_HNO3 :
  n_CaO = 7 →
  verify_moles_HNO3 n_CaO = 14 :=
sorry

theorem proof_molecular_weight_HNO3 :
  molar_mass_H = 101 / 100 ∧ molar_mass_N = 1401 / 100 ∧ molar_mass_O = 1600 / 100 →
  verify_molecular_weight_HNO3 molar_mass_H molar_mass_N molar_mass_O = 6302 / 100 :=
sorry

end proof_moles_HNO3_proof_molecular_weight_HNO3_l11_11327


namespace even_three_digit_numbers_l11_11050

theorem even_three_digit_numbers (n : ℕ) :
  (n >= 100 ∧ n < 1000) ∧
  (n % 2 = 0) ∧
  ((n % 100) / 10 + (n % 10) = 12) →
  n = 12 :=
sorry

end even_three_digit_numbers_l11_11050


namespace three_digit_even_sum_12_l11_11028

theorem three_digit_even_sum_12 : 
  ∃ (n : Finset ℕ), 
    n.card = 27 ∧ 
    ∀ x ∈ n, 
      ∃ h t u, 
        (100 * h + 10 * t + u = x) ∧ 
        (h ∈ Finset.range 9 \ {0}) ∧ 
        (u % 2 = 0) ∧ 
        (t + u = 12) := 
sorry

end three_digit_even_sum_12_l11_11028


namespace binom_19_12_l11_11852

theorem binom_19_12 :
  nat.choose 20 13 = 77520 →
  nat.choose 20 12 = 125970 →
  nat.choose 18 11 = 31824 →
  nat.choose 19 12 = 45696 :=
by
  intros h1 h2 h3
  sorry

end binom_19_12_l11_11852


namespace average_age_population_l11_11909

theorem average_age_population 
  (k : ℕ) 
  (hwomen : ℕ := 7 * k)
  (hmen : ℕ := 5 * k)
  (avg_women_age : ℕ := 40)
  (avg_men_age : ℕ := 30)
  (h_age_women : ℕ := avg_women_age * hwomen)
  (h_age_men : ℕ := avg_men_age * hmen) : 
  (h_age_women + h_age_men) / (hwomen + hmen) = 35 + 5/6 :=
by
  sorry -- proof will fill in here

end average_age_population_l11_11909


namespace average_price_per_share_l11_11309

-- Define the conditions
def Microtron_price_per_share := 36
def Dynaco_price_per_share := 44
def total_shares := 300
def Dynaco_shares_sold := 150

-- Define the theorem to be proved
theorem average_price_per_share : 
  (Dynaco_shares_sold * Dynaco_price_per_share + (total_shares - Dynaco_shares_sold) * Microtron_price_per_share) / total_shares = 40 :=
by
  -- Skip the actual proof here
  sorry

end average_price_per_share_l11_11309


namespace f_correct_l11_11363

def floor_div (a b : ℕ) : ℕ :=
  a / b

noncomputable def f (n : ℕ) : ℕ :=
  floor_div (n+1) 2 + floor_div (n+1) 3 - floor_div (n+1) 6 + 1

theorem f_correct (n : ℕ) (hn : n ≥ 4) :
  ∀ (m : ℕ), ∃ (S : set ℕ), S ⊆ (finset.range (n+1)).image (λ x, m + x) ∧ S.card = f(n) ∧
                have 3 ≤ S.card := by sorry
                (∀ {a b c : ℕ}, a ∈ S → b ∈ S → c ∈ S → a ≠ b → a ≠ c → b ≠ c → gcd a b = 1 ∧ gcd a c = 1 ∧ gcd b c = 1) :=
sorry

end f_correct_l11_11363


namespace number_of_polynomials_l11_11773

def is_prime (n : ℕ) : Prop := sorry  -- Assuming a prime checking function (as Lean library has no direct support for this)

noncomputable def polynomial_conditions (p₁ p₂ p₃ : ℕ) : Prop :=
  is_prime p₁ ∧ is_prime p₂ ∧ is_prime p₃ ∧ p₁ < 50 ∧ p₂ < 50 ∧ p₃ < 50 ∧
  ∃ (r₁ r₂ : ℚ), r₁ ≠ r₂ ∧
  polynomial.eval r₁ (polynomial.C p₁ * polynomial.X^2 + polynomial.C p₂ * polynomial.X - polynomial.C p₃) = 0 ∧
  polynomial.eval r₂ (polynomial.C p₁ * polynomial.X^2 + polynomial.C p₂ * polynomial.X - polynomial.C p₃) = 0

theorem number_of_polynomials : ∑ (p₁ p₂ p₃ : ℕ) in { p | polynomial_conditions p p p }.to_finset, 1 = 31 := sorry


end number_of_polynomials_l11_11773


namespace range_of_k_l11_11445

theorem range_of_k (k n : ℝ) (h : k ≠ 0) (h_pass : k - n^2 - 2 = k / 2) : k ≥ 4 :=
sorry

end range_of_k_l11_11445


namespace sqrt_defined_iff_nonnegative_l11_11218

theorem sqrt_defined_iff_nonnegative (x : ℝ) : (∃ y : ℝ, y = real.sqrt (x + 2)) ↔ x ≥ -2 := sorry

end sqrt_defined_iff_nonnegative_l11_11218


namespace problem1_problem2_l11_11904

variable (A B C : Real) (a b c : ℝ)

-- Define conditions in Lean 4

-- trigonometric functions requirement
noncomputable def cos : Real → Real := sorry
noncomputable def tan : Real → Real := sorry

-- Given condition for the sides of the triangle
def condition1 : Prop := 2 * c^2 - 2 * a^2 = b^2

-- Problem 1
theorem problem1 (h : condition1) : 2 * c * cos A - 2 * a * cos C = b := by
  sorry

-- Additional conditions for Problem 2
def condition2 : Prop := a = 1 ∧ tan A = 1 / 3

-- Area of the triangle using half base times height
noncomputable def area_of_triangle (a b : Real) (C : Real) : Real :=
  (1 / 2) * a * b * Math.sin(C)

-- Problem 2
theorem problem2 (h1 : condition1) (h2 : condition2) : 
  area_of_triangle 1 b (Real.pi / 4) = 1 := by
  sorry

end problem1_problem2_l11_11904


namespace triangle_similarity_length_RY_l11_11121

theorem triangle_similarity_length_RY
  (P Q R X Y Z : Type)
  [MetricSpace P] [MetricSpace Q] [MetricSpace R]
  [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
  (PQ : ℝ) (XY : ℝ) (RY_length : ℝ)
  (h1 : PQ = 10)
  (h2 : XY = 6)
  (h3 : ∀ (PR QR PX QX RZ : ℝ) (angle_PY_RZ : ℝ),
    PR + RY_length = PX ∧
    QR + RY_length = QX ∧ 
    angle_PY_RZ = 120 ∧
    PR > 0 ∧ QR > 0 ∧ RY_length > 0)
  (h4 : XY / PQ = RY_length / (PQ + RY_length)) :
  RY_length = 15 := by
  sorry

end triangle_similarity_length_RY_l11_11121


namespace cookies_in_the_fridge_l11_11641

-- Define the conditions
def total_baked : ℕ := 256
def tim_cookies : ℕ := 15
def mike_cookies : ℕ := 23
def anna_cookies : ℕ := 2 * tim_cookies

-- Define the proof problem
theorem cookies_in_the_fridge : (total_baked - (tim_cookies + mike_cookies + anna_cookies)) = 188 :=
by
  -- insert proof here
  sorry

end cookies_in_the_fridge_l11_11641


namespace circle_tangent_radius_l11_11446

theorem circle_tangent_radius
  (r : ℝ) (r_pos : r > 0)
  (circle_eq : ∀ x y, (x - 4)^2 + y^2 = r^2)
  (line_eq : ∀ x y, sqrt (3) * x - 2 * y = 0)
  (tangent : ∀ x y, circle_eq x y → line_eq x y):
  r = 4 * sqrt (21) / 7 :=
by
  sorry

end circle_tangent_radius_l11_11446


namespace three_digit_even_sum_12_l11_11032

theorem three_digit_even_sum_12 : 
  ∃ (n : Finset ℕ), 
    n.card = 27 ∧ 
    ∀ x ∈ n, 
      ∃ h t u, 
        (100 * h + 10 * t + u = x) ∧ 
        (h ∈ Finset.range 9 \ {0}) ∧ 
        (u % 2 = 0) ∧ 
        (t + u = 12) := 
sorry

end three_digit_even_sum_12_l11_11032


namespace domain_of_log_base_half_l11_11200

noncomputable def domain_log_base_half : Set ℝ := { x : ℝ | x > 5 }

theorem domain_of_log_base_half :
  (∀ x : ℝ, x > 5 ↔ x - 5 > 0) →
  (domain_log_base_half = { x : ℝ | x - 5 > 0 }) :=
by
  sorry

end domain_of_log_base_half_l11_11200


namespace subset_definition_l11_11835

variable {α : Type} {A B : Set α}

theorem subset_definition :
  A ⊆ B ↔ ∀ a ∈ A, a ∈ B :=
by sorry

end subset_definition_l11_11835


namespace three_digit_even_sum_12_l11_11029

theorem three_digit_even_sum_12 : 
  ∃ (n : Finset ℕ), 
    n.card = 27 ∧ 
    ∀ x ∈ n, 
      ∃ h t u, 
        (100 * h + 10 * t + u = x) ∧ 
        (h ∈ Finset.range 9 \ {0}) ∧ 
        (u % 2 = 0) ∧ 
        (t + u = 12) := 
sorry

end three_digit_even_sum_12_l11_11029


namespace constant_f_l11_11912

-- Definitions from conditions
variables (A B C D E F : Type) [has_scalar ℝ A] [has_scalar ℝ B] [has_scalar ℝ C] [has_scalar ℝ D] [has_scalar ℝ E] [has_scalar ℝ F]

-- Angles and properties
variables (AC BD EF : Type) [angle AC BD = 90] [parallel EF AC] [parallel EF BD]

-- Function definition as per the problem
def f (λ : ℝ) [h : 0 < λ ∧ λ < +∞] := 
  if 0 < λ then
    let α := angle EF AC 
    let β := angle EF BD 
    α + β
  else
    0

-- The Lean statement asserting the function is constant
theorem constant_f : ∀λ ∈ set.Ioo 0 (+∞), f λ = 90 :=
begin
  intro λ,
  intro h,
  rw f,
  have αλ_plus_βλ_eq_const : angle EF AC + angle EF BD = 90, from sorry,
  exact αλ_plus_βλ_eq_const,
end

end constant_f_l11_11912


namespace ordering_of_f_values_l11_11207

variables {f : ℝ → ℝ}

def a := f 0
def b := f 1
def c := f 4

theorem ordering_of_f_values
  (h_deriv : differentiable ℝ f)
  (h_sym : ∀ x, f x = f (4 - x))
  (h_ineq : ∀ x, x < 2 → (x - 2) * (deriv f x) < 0) :
  c < a ∧ a < b :=
sorry

end ordering_of_f_values_l11_11207


namespace functional_equation_satisfied_l11_11823

noncomputable def f (x : ℝ) : ℝ :=
  if x = 1 then -f 0
  else (1/2) * (x + 1 - (1/x) - (1/(1-x)))

theorem functional_equation_satisfied :
  ∀ x : ℝ, x ≠ 1 → f x + f (1 / (1 - x)) = x :=
by
  intros x hx
  rw [f, f]
  split_ifs with h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end functional_equation_satisfied_l11_11823


namespace Kuwabara_class_girls_percentage_l11_11598

variable (num_girls num_boys : ℕ)

def total_students (num_girls num_boys : ℕ) : ℕ :=
  num_girls + num_boys

def girls_percentage (num_girls num_boys : ℕ) : ℚ :=
  (num_girls : ℚ) / (total_students num_girls num_boys : ℚ) * 100

theorem Kuwabara_class_girls_percentage (num_girls num_boys : ℕ) (h1: num_girls = 10) (h2: num_boys = 15) :
  girls_percentage num_girls num_boys = 40 := 
by
  sorry

end Kuwabara_class_girls_percentage_l11_11598


namespace rays_from_three_non_collinear_points_l11_11231

theorem rays_from_three_non_collinear_points (A B C : Point) (h_non_collinear : ¬ collinear A B C) :
  number_of_rays_from_points A B C = 6 :=
sorry

end rays_from_three_non_collinear_points_l11_11231


namespace no_valid_configuration_l11_11580

-- Definitions: Assume hexagon ABCDEF with center J and assigning digits 1 through 7 uniquely
def hexagon (A B C D E F J : ℕ) := A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ J ∧ 
                                    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ J ∧ 
                                    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ J ∧ 
                                    D ≠ E ∧ D ≠ F ∧ D ≠ J ∧ 
                                    E ≠ F ∧ E ≠ J ∧ 
                                    F ≠ J ∧
                                    A + B + C + D + E + F + J = 28

-- Predicates: Define sums of lines AJE, BJF, CJG, DJH
def sums (A B C D E F J : ℕ) (x : ℕ) := A + J + E = x ∧ 
                                         B + J + F = x ∧ 
                                         C + J + G = x ∧ 
                                         D + J + H = x

-- Theorem: Prove that the number of ways is zero
theorem no_valid_configuration : ∀ (A B C D E F J : ℕ), 
  hexagon A B C D E F J →
  ¬ ∃ x, sums A B C D E F J x :=
by
  sorry

end no_valid_configuration_l11_11580


namespace part1_part2_l11_11867

noncomputable def f (a x : ℝ) : ℝ := (sqrt 5 / a) * x + (sqrt 5 * (a - 1)) / x

theorem part1 (a : ℝ) (h₁ : a ≠ 0) (h₂ : 0 < a) :
  ∀ x, (0 < x ∧ x < sqrt 6 → deriv (f a) x < 0) ∧ (sqrt 6 < x → deriv (f a) x > 0) ↔ a = 3 := sorry

theorem part2 (a : ℝ) (h₁ : a ≠ 0) : 
  (has_inverse (λ x : ℝ, f a x) (interval (- sqrt 6 / 6) 0 ∪ interval 0 (sqrt 6 / 6))) ↔ 
  (a ∈ (-∞, (3 - sqrt 15) / 6] ∪ ((3 - sqrt 3) / 6, (3 + sqrt 3) / 6) ∪ {1} ∪ [(3 + sqrt 15) / 6, ∞)) := sorry

end part1_part2_l11_11867


namespace calculate_square_difference_l11_11756

theorem calculate_square_difference : 2023^2 - 2022^2 = 4045 := by
  sorry

end calculate_square_difference_l11_11756


namespace find_point_A_coordinates_l11_11820

theorem find_point_A_coordinates :
  ∃ m, (A : ℝ × ℝ) = (m, 2) ∧ 2 = 2 * m - 4 → A = (3, 2) :=
by
  intro h
  sorry

end find_point_A_coordinates_l11_11820


namespace relationship_abc_d_l11_11838

theorem relationship_abc_d : 
  ∀ (a b c d : ℝ), 
  a < b → 
  d < c → 
  (c - a) * (c - b) < 0 → 
  (d - a) * (d - b) > 0 → 
  d < a ∧ a < c ∧ c < b :=
by
  intros a b c d a_lt_b d_lt_c h1 h2
  sorry

end relationship_abc_d_l11_11838


namespace find_ellipse_find_line_l11_11388

noncomputable def ellipse_eq (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Prop :=
  a = 2 * Real.sqrt 2 ∧ b^2 = 4 ∧ ( ∀ x y : ℝ, (x^2) / (a^2) + (y^2) / (b^2) = 1 → (x^2) / 8 + (y^2) / 4 = 1 )

theorem find_ellipse : ∃ a b : ℝ, ellipse_eq a b
by
  use 2 * Real.sqrt 2
  use 2
  unfold ellipse_eq
  sorry

noncomputable def line_eq (l : ℝ → ℝ) (area : ℝ) : Prop :=
  ∃ m : ℝ, (l = λ y, m * y + 2) ∧
           (m = Real.sqrt 2 ∨ m = -Real.sqrt 2) ∧
           (area = Real.sqrt 6 / 2)

theorem find_line : ∃ l : ℝ → ℝ, ∃ area : ℝ, (area = Real.sqrt 6 / 2) ∧ line_eq l area
by
  use (λ y, Real.sqrt 2 * y + 2)
  use Real.sqrt 6 / 2
  unfold line_eq
  sorry

end find_ellipse_find_line_l11_11388


namespace park_deer_count_l11_11633

theorem park_deer_count (D : ℕ) 
  (h1 : 0.10 * D = (D * (1 / 10))) 
  (h2 : ((1 / 4) * (0.10 * D)) = ((D * (1 / 10)) * (1 / 4))) 
  (h3 : (D * (1 / 10)) * (1 / 4) = 23) :
  D = 920 :=
by
  sorry

end park_deer_count_l11_11633


namespace even_function_properties_l11_11898

theorem even_function_properties 
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_increasing : ∀ x y : ℝ, 5 ≤ x ∧ x ≤ y ∧ y ≤ 7 → f x ≤ f y)
  (h_min_value : ∀ x : ℝ, 5 ≤ x ∧ x ≤ 7 → 6 ≤ f x) :
  (∀ x y : ℝ, -7 ≤ x ∧ x ≤ y ∧ y ≤ -5 → f y ≤ f x) ∧ (∀ x : ℝ, -7 ≤ x ∧ x ≤ -5 → 6 ≤ f x) :=
by
  sorry

end even_function_properties_l11_11898


namespace kobe_function_range_l11_11368

-- Definition of a Kobe function
def is_kobe_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < b ∧ ∀ x ∈ set.Icc a b, f x ∈ set.Icc a b

-- The function f(x) = k + sqrt(x - 1)
noncomputable def f (k x : ℝ) : ℝ := k + real.sqrt (x - 1)

-- Prove that f(x) is a Kobe function if and only if k is in the interval (3/4, 1]
theorem kobe_function_range (k : ℝ) :
  (∃ a b, is_kobe_function (f k) a b) ↔ k ∈ set.Ioo (3 / 4) 1 ∨ k = 1 :=
sorry

end kobe_function_range_l11_11368


namespace max_balloon_surface_area_l11_11284

theorem max_balloon_surface_area (a : ℝ) (h : a > 0) : 
  ∃ S : ℝ, S = 2 * Real.pi * a^2 :=
sorry

end max_balloon_surface_area_l11_11284


namespace triangle_sin_A_and_height_l11_11482

noncomputable theory

variables (A B C : ℝ) (AB : ℝ)
  (h1 : A + B = 3 * C)
  (h2 : 2 * Real.sin (A - C) = Real.sin B)
  (h3 : AB = 5)

theorem triangle_sin_A_and_height :
  Real.sin A = 3 * Real.cos A → 
  sqrt 10 / 10 * Real.sin A = 3 / sqrt (10) / 3 → 
  √10 / 10 = 3/ sqrt 10 /3 → 
  sin (A+B) =sin /sqrt10 →
  (sin (A cv)+ C) = sin( AC ) → 
  ( cos A = sinA 3) 
  ( (10 +25)+5→1= well 5 → B (PS6 S)=H1 (A3+.B9)=
 
 
   
∧   (γ = hA → ) ( (/. );



∧ side /4→ABh3 → 5=HS)  ( →AB3)=sinh1S  

then 
(
  (Real.sin A = 3 * Real.cos A) ^2 )+   
  
(Real.cos A= √ 10/10
  
  Real.sin A2 C(B)= 3√10/10
  
 ) ^(Real.sin A = 5

6)=    
    sorry

end triangle_sin_A_and_height_l11_11482


namespace encyclopedia_total_pages_l11_11974

theorem encyclopedia_total_pages
  (chapters : ℕ)
  (pages_per_chapter : ℕ)
  (h1 : chapters = 7)
  (h2 : pages_per_chapter = 566)
  : chapters * pages_per_chapter = 3962 :=
by
  rw [h1, h2]
  norm_num
  exact Eq.refl 3962

end encyclopedia_total_pages_l11_11974


namespace integer_part_inequality_l11_11542

theorem integer_part_inequality (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
 (h_cond : (x + y + z) * ((1 / x) + (1 / y) + (1 / z)) = (91 / 10)) :
  (⌊(x^3 + y^3 + z^3) * ((1 / x^3) + (1 / y^3) + (1 / z^3))⌋) = 9 :=
by
  -- proof here
  sorry

end integer_part_inequality_l11_11542


namespace red_notebooks_count_l11_11568

variable (R B : ℕ)

-- Conditions
def cost_condition : Prop := 4 * R + 4 + 3 * B = 37
def count_condition : Prop := R + 2 + B = 12
def blue_notebooks_expr : Prop := B = 10 - R

-- Prove the number of red notebooks
theorem red_notebooks_count : cost_condition R B ∧ count_condition R B ∧ blue_notebooks_expr R B → R = 3 := by
  sorry

end red_notebooks_count_l11_11568


namespace cobbler_pairs_per_week_l11_11693

def monday_wednesday_rate := 3 -- pairs per hour
def thursday_rate := 2 -- pairs per hour
def friday_rate := 4 -- pairs per hour
def break_time := 0.5 -- hours
def work_hours_per_day := 8 -- hours

theorem cobbler_pairs_per_week : 
  (monday_wednesday_rate * (work_hours_per_day - break_time) * 3) +
  (thursday_rate * (work_hours_per_day - break_time)) +
  (friday_rate * 3) = 94 :=
by 
  -- Skipping proofs
  sorry

end cobbler_pairs_per_week_l11_11693


namespace value_of_k_through_point_l11_11370

noncomputable def inverse_proportion_function (x : ℝ) (k : ℝ) : ℝ :=
  k / x

theorem value_of_k_through_point (k : ℝ) (h : k ≠ 0) : inverse_proportion_function 2 k = 3 → k = 6 :=
by
  sorry

end value_of_k_through_point_l11_11370


namespace num_five_letter_words_correct_l11_11714

noncomputable def num_five_letter_words : ℕ := 1889568

theorem num_five_letter_words_correct :
  let a := 3
  let e := 4
  let i := 2
  let o := 5
  let u := 4
  (a + e + i + o + u) ^ 5 = num_five_letter_words :=
by
  sorry

end num_five_letter_words_correct_l11_11714


namespace find_zero_interval_l11_11872

def f (x : ℝ) : ℝ := (6 / x) - Real.logb 2 x

theorem find_zero_interval : 
  ∃ a b : ℝ, (2 < a ∧ a < 4) ∧ (2 < b ∧ b < 4) ∧ f(a) > 0 ∧ f(b) < 0 ∧ (∀ c, a < c ∧ c < b → f(c) = 0) :=
by
  sorry

end find_zero_interval_l11_11872


namespace remainder_of_trailing_zeros_in_factorials_product_l11_11950

theorem remainder_of_trailing_zeros_in_factorials_product :
  let M := (trailing_zeroes (finprod (λ n : ℕ, if n ∈ (set.range (λ k, k!)) then n else 1))) in
  M % 500 = 21 := by
  sorry

def trailing_zeroes (n : ℕ) : ℕ := 
  if n = 0 then 0 else trailing_zeroes_aux n 0

@[simp] def trailing_zeroes_aux (n : ℕ) (acc : ℕ) : ℕ :=
  if n % 10 = 0 then trailing_zeroes_aux (n / 10) (acc + 1) else acc

noncomputable def finprod {α β : Type*} [comm_monoid β] (f : α → β) : β :=
  finset.univ.prod f

end remainder_of_trailing_zeros_in_factorials_product_l11_11950


namespace pascal_triangle_probability_l11_11734

-- Define the probability problem in Lean 4
theorem pascal_triangle_probability :
  let total_elements := ((20 * (20 + 1)) / 2)
  let ones_count := (1 + 2 * 19)
  let twos_count := (2 * (19 - 2 + 1))
  (ones_count + twos_count) / total_elements = 5 / 14 :=
by
  let total_elements := ((20 * (20 + 1)) / 2)
  let ones_count := (1 + 2 * 19)
  let twos_count := (2 * (19 - 2 + 1))
  have h1 : total_elements = 210 := by sorry
  have h2 : ones_count = 39 := by sorry
  have h3 : twos_count = 36 := by sorry
  have h4 : (39 + 36) / 210 = 5 / 14 := by sorry
  exact h4

end pascal_triangle_probability_l11_11734


namespace cost_to_paint_cube_l11_11191

theorem cost_to_paint_cube :
  let cost_per_kg := 50
  let coverage_per_kg := 20
  let side_length := 20
  let surface_area := 6 * (side_length * side_length)
  let amount_of_paint := surface_area / coverage_per_kg
  let total_cost := amount_of_paint * cost_per_kg
  total_cost = 6000 :=
by
  sorry

end cost_to_paint_cube_l11_11191


namespace find_cola_cost_l11_11908

variable (C : ℝ)
variable (juice_cost water_cost : ℝ)
variable (colas_sold waters_sold juices_sold : ℕ)
variable (total_earnings : ℝ)

theorem find_cola_cost
  (h1 : juice_cost = 1.5)
  (h2 : water_cost = 1)
  (h3 : colas_sold = 15)
  (h4 : waters_sold = 25)
  (h5 : juices_sold = 12)
  (h6 : total_earnings = 88) :
  15 * C + 25 * water_cost + 12 * juice_cost = total_earnings → C = 3 :=
by
  intro h
  have eqn : 15 * C + 25 * 1 + 12 * 1.5 = 88 := by rw [h1, h2]; exact h
  -- the proof steps solving for C would go here
  sorry

end find_cola_cost_l11_11908


namespace certain_number_l11_11890

theorem certain_number (a x : ℝ) (h1 : a / x * 2 = 12) (h2 : x = 0.1) : a = 0.6 := 
by
  sorry

end certain_number_l11_11890


namespace determinant_value_l11_11955

noncomputable def determinant_expression (r a b c : ℝ) : ℝ :=
  Matrix.det ![
    ![r + a, r, r],
    ![r, r + b, r],
    ![r, r, r + c]
  ]

theorem determinant_value (p q : ℝ) (a b c : ℝ) (h : a^3 - 3*p*a^2 + q*a - 2 = 0)
  (h : b^3 - 3*p*b^2 + q*b - 2 = 0)
  (h : c^3 - 3*p*c^2 + q*c - 2 = 0) :
  determinant_expression r a b c = -2 :=
begin
  sorry
end

end determinant_value_l11_11955


namespace probability_of_one_or_two_in_pascal_l11_11729

def pascal_triangle_element_probability : ℚ :=
  let total_elements := 210 -- sum of the elements in the first 20 rows
  let ones_count := 39      -- total count of 1s in the first 20 rows
  let twos_count := 36      -- total count of 2s in the first 20 rows
  let favorable_elements := ones_count + twos_count
  favorable_elements / total_elements

theorem probability_of_one_or_two_in_pascal (n : ℕ) (h : n = 20) :
  pascal_triangle_element_probability = 5 / 14 := by
  rw [h]
  dsimp [pascal_triangle_element_probability]
  sorry

end probability_of_one_or_two_in_pascal_l11_11729


namespace repeating_decimal_denominator_l11_11195

theorem repeating_decimal_denominator (S : ℚ) (h : S = 0.27) : ∃ d : ℤ, (S = d / 11) :=
by
  sorry

end repeating_decimal_denominator_l11_11195


namespace floor_sqrt_50_squared_l11_11782

theorem floor_sqrt_50_squared :
  ∃ x : ℕ, x = 7 ∧ ⌊ Real.sqrt 50 ⌋ = x ∧ x^2 = 49 := 
by {
  let x := 7,
  use x,
  have h₁ : 7 < Real.sqrt 50, from sorry,
  have h₂ : Real.sqrt 50 < 8, from sorry,
  have floor_eq : ⌊Real.sqrt 50⌋ = 7, from sorry,
  split,
  { refl },
  { split,
    { exact floor_eq },
    { exact rfl } }
}

end floor_sqrt_50_squared_l11_11782


namespace triangle_area_is_2_l11_11600

noncomputable def area_of_triangle_OAB {x₀ : ℝ} (h₀ : 0 < x₀) : ℝ :=
  let y₀ := 1 / x₀
  let slope := -1 / x₀^2
  let tangent_line (x : ℝ) := y₀ + slope * (x - x₀)
  let A : ℝ × ℝ := (2 * x₀, 0) -- Intersection with x-axis
  let B : ℝ × ℝ := (0, 2 * y₀) -- Intersection with y-axis
  1 / 2 * abs (2 * y₀ * 2 * x₀)

theorem triangle_area_is_2 (x₀ : ℝ) (h₀ : 0 < x₀) : area_of_triangle_OAB h₀ = 2 :=
by
  sorry

end triangle_area_is_2_l11_11600


namespace max_lines_through_point_P_l11_11846

theorem max_lines_through_point_P (P : Point) (A B C : Point) (P_outside_triangle : ¬(P ∈ Triangle ABC)) :
  ∃ lines : List Line, (∀ l ∈ lines, cuts_off_similar_triangle l P A B C) ∧ lines.length = 6 :=
sorry

end max_lines_through_point_P_l11_11846


namespace angle_B_equal_pi_div_3_l11_11454

-- Define the conditions and the statement to be proved
theorem angle_B_equal_pi_div_3 (A B C : ℝ) 
  (h₁ : Real.sin A / Real.sin B = 5 / 7)
  (h₂ : Real.sin B / Real.sin C = 7 / 8) : 
  B = Real.pi / 3 :=
sorry

end angle_B_equal_pi_div_3_l11_11454


namespace integral_sqrt_equivalence_l11_11258

open Real

theorem integral_sqrt_equivalence :
  2 * ∫ x in -1..1, sqrt (1 - x^2) = ∫ x in -1..1, 1 / sqrt (1 - x^2) := 
by sorry

end integral_sqrt_equivalence_l11_11258


namespace sqrt_one_div_one_hundred_l11_11596

theorem sqrt_one_div_one_hundred : (√(1 / 100) = 1 / 10) :=
by
  sorry

end sqrt_one_div_one_hundred_l11_11596


namespace distinct_students_l11_11749

theorem distinct_students 
  (students_euler : ℕ) (students_gauss : ℕ) (students_fibonacci : ℕ) (overlap_euler_gauss : ℕ)
  (h_euler : students_euler = 15) 
  (h_gauss : students_gauss = 10) 
  (h_fibonacci : students_fibonacci = 12) 
  (h_overlap : overlap_euler_gauss = 3) 
  : students_euler + students_gauss + students_fibonacci - overlap_euler_gauss = 34 :=
by
  sorry

end distinct_students_l11_11749


namespace Dr_Maths_house_number_count_l11_11776

noncomputable def two_digit_primes : List ℕ := [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

theorem Dr_Maths_house_number_count :
  let valid_combinations := 
    (two_digit_primes.product two_digit_primes).filter 
      (λ p, p.1 > p.2)
  in valid_combinations.length = 55 :=
by
  -- Defining the list of two-digit primes
  have primes := two_digit_primes
  -- Extract pairwise products where the first element is greater than the second element
  have combinations := (primes.product primes).filter (λ p, p.1 > p.2)
  -- There are 55 such pairs
  show combinations.length = 55
  sorry

end Dr_Maths_house_number_count_l11_11776


namespace count_even_three_digit_numbers_l11_11043

theorem count_even_three_digit_numbers : 
  let num_even_three_digit_numbers : ℕ := 
    have h1 : (units_digit_possible_pairs : list (ℕ × ℕ)) := 
      [(4, 8), (6, 6), (8, 4)]
    have h2 : (number_of_hundreds_digits : ℕ) := 9
    3 * number_of_hundreds_digits 
in
  num_even_three_digit_numbers = 27 := by
  -- steps skipped
  sorry

end count_even_three_digit_numbers_l11_11043


namespace train_length_l11_11261

theorem train_length (speed_kmph : ℕ) (platform_length : ℕ) (crossing_time : ℕ) :
  speed_kmph = 72 → platform_length = 280 → crossing_time = 26 → 
  let speed_mps := (speed_kmph * 1000) / 3600
  let total_distance := speed_mps * crossing_time
  let train_length := total_distance - platform_length
  train_length = 240 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  let speed_mps := (72 * 1000) / 3600
  have speed_mps_eq : speed_mps = 20 := by norm_num
  rw speed_mps_eq
  let total_distance := 20 * 26
  have total_distance_eq : total_distance = 520 := by norm_num
  rw total_distance_eq
  let train_length := 520 - 280
  have train_length_eq : train_length = 240 := by norm_num
  rw train_length_eq
  exact rfl

end train_length_l11_11261


namespace intersection_of_A_and_B_l11_11141

noncomputable def A := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
noncomputable def B := {x : ℝ | x < 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} :=
sorry

end intersection_of_A_and_B_l11_11141


namespace exponential_problem_l11_11885

-- Define the problem conditions
variables (a b : ℝ)
axiom h1 : (24:ℝ)^a = 2
axiom h2 : (24:ℝ)^b = 3

-- Define the statement to be proved
theorem exponential_problem : 8 ^ ((1 - a - b) / (2 * (1 - b))) = 2 :=
by sorry

end exponential_problem_l11_11885


namespace sin_A_eq_height_on_AB_l11_11524

-- Defining conditions
variables {A B C : ℝ}
variables (AB : ℝ)

-- Conditions based on given problem
def condition1 : Prop := A + B = 3 * C
def condition2 : Prop := 2 * sin (A - C) = sin B
def condition3 : Prop := A + B + C = Real.pi

-- Question 1: prove that sin A = (3 * sqrt 10) / 10
theorem sin_A_eq:
  condition1 → 
  condition2 → 
  condition3 → 
  sin A = (3 * Real.sqrt 10) / 10 :=
by
  sorry

-- Question 2: given AB = 5, prove the height on side AB is 6
theorem height_on_AB:
  condition1 →
  condition2 →
  condition3 →
  AB = 5 →
  -- Let's construct the height as a function of A, B, and C
  ∃ h, h = 6 :=
by
  sorry

end sin_A_eq_height_on_AB_l11_11524


namespace height_difference_center_tangency_l11_11289

theorem height_difference_center_tangency (a b r : ℝ) :
  let y_parabola := λ x : ℝ, x^2 + 1
  let y_tangent := a^2 + 1
  let center_y := a^2 + 3 / 2
  let height_difference := center_y - y_tangent
  (y_parabola a = y_tangent) 
  ∧ (y_parabola (-a) = y_tangent) 
  ∧ (center_y = b)
  ∧ (height_difference = 1 / 2)

end height_difference_center_tangency_l11_11289


namespace solve_triangle_sides_l11_11620

def triangle_sides : Prop :=
  let x := 7
  let y := 5
  let z := 3
  let θ := Real.pi / 3 * 2 -- 120 degrees in radians
  x + y = 12 ∧
  (1/2) * x * y * Real.sin θ = (17.5) * Real.sin θ ∧
  Real.cos θ = -1/2 ∧
  x^2 = y^2 + z^2 - 2 * y * z * Real.cos θ

theorem solve_triangle_sides : triangle_sides :=
by {
  let x := 7
  let y := 5
  let z := 3
  let θ := Real.pi / 3 * 2 -- 120 degrees in radians
  have h1 : x + y = 12 := rfl,
  have h2 : (1/2) * x * y * Real.sin θ = (17.5) * Real.sin θ := by sorry,
  have h3 : Real.cos θ = -1/2 := by sorry,
  have h4 : x^2 = y^2 + z^2 - 2 * y * z * Real.cos θ := by sorry,
  exact ⟨h1, h2, h3, h4⟩,
}

end solve_triangle_sides_l11_11620


namespace solution_set_ineq_l11_11220

theorem solution_set_ineq (x : ℝ) : (1 / x > 1) ↔ (0 < x ∧ x < 1) :=
by
  sorry

end solution_set_ineq_l11_11220


namespace rightmost_four_digits_of_7_pow_2045_l11_11244

theorem rightmost_four_digits_of_7_pow_2045 : (7^2045 % 10000) = 6807 :=
by
  sorry

end rightmost_four_digits_of_7_pow_2045_l11_11244


namespace circles_tangent_chord_length_l11_11323

-- Definitions of conditions given in the problem
def radius_A := 5
def radius_B := 12
def dist_centers := radius_A + radius_B
def m := 13
def n := 7
def p := 2

-- Calculation of the final results asked in the question
theorem circles_tangent_chord_length : m + n + p = 22 := by
  have radius_A := (5 : ℕ)
  have radius_B := (12 : ℕ)
  have dist_centers := radius_A + radius_B
  have m := 13
  have n := 7
  have p := 2
  show m + n + p = 22, by sorry

end circles_tangent_chord_length_l11_11323


namespace find_b_range_l11_11100

theorem find_b_range
  (b : ℝ)
  (P : ℝ × ℝ)
  (hO : ∀ x y, x^2 + y^2 = 1 ↔ (P = (x, y) → ∃ A, P ∈ tangent O A))
  (hO1 : ∀ x y, (x-4)^2 + y^2 = 4 ↔ (P = (x, y) → ∃ B, P ∈ tangent O1 B))
  (h_line : P.1 + sqrt 3 * P.2 - b = 0)
  (h_P_cond : ∀ P, P ∈ line (x + sqrt 3 * y - b = 0) → exists A B, tangent P O A ∧ tangent P O1 B ∧ dist P B = 2 * dist P A) :
  -20 / 3 < b ∧ b < 4 := 
sorry

end find_b_range_l11_11100


namespace sin_A_eq_height_on_AB_l11_11522

-- Defining conditions
variables {A B C : ℝ}
variables (AB : ℝ)

-- Conditions based on given problem
def condition1 : Prop := A + B = 3 * C
def condition2 : Prop := 2 * sin (A - C) = sin B
def condition3 : Prop := A + B + C = Real.pi

-- Question 1: prove that sin A = (3 * sqrt 10) / 10
theorem sin_A_eq:
  condition1 → 
  condition2 → 
  condition3 → 
  sin A = (3 * Real.sqrt 10) / 10 :=
by
  sorry

-- Question 2: given AB = 5, prove the height on side AB is 6
theorem height_on_AB:
  condition1 →
  condition2 →
  condition3 →
  AB = 5 →
  -- Let's construct the height as a function of A, B, and C
  ∃ h, h = 6 :=
by
  sorry

end sin_A_eq_height_on_AB_l11_11522


namespace rectangle_overlap_area_l11_11911

theorem rectangle_overlap_area (AB CD BC AD : ℝ) (O : ℝ) :
  AB = 2 ∧ CD = 2 ∧ BC = 8 ∧ AD = 8 ∧ O = 1 →
  ∃ (overlap_area : ℝ), overlap_area = 2 * (3 * Real.sqrt 2 - 2) :=
by 
  intro h,
  sorry

end rectangle_overlap_area_l11_11911


namespace min_accommodation_cost_l11_11710

theorem min_accommodation_cost :
  ∃ (x y z : ℕ), x + y + z = 20 ∧ 3 * x + 2 * y + z = 50 ∧ 100 * 3 * x + 150 * 2 * y + 200 * z = 5500 :=
by
  sorry

end min_accommodation_cost_l11_11710


namespace count_even_three_digit_numbers_sum_tens_units_eq_12_l11_11009

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def is_even (n : ℕ) : Prop := n % 2 = 0
def sum_of_tens_and_units_eq_12 (n : ℕ) : Prop :=
  (n / 10) % 10 + n % 10 = 12

theorem count_even_three_digit_numbers_sum_tens_units_eq_12 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_three_digit n ∧ is_even n ∧ sum_of_tens_and_units_eq_12 n) ∧ S.card = 24 :=
sorry

end count_even_three_digit_numbers_sum_tens_units_eq_12_l11_11009


namespace trapezoid_midsegment_l11_11914

-- Define the problem conditions and question
theorem trapezoid_midsegment (b h x : ℝ) (h_nonzero : h ≠ 0) (hx : x = b + 75)
  (equal_areas : (1 / 2) * (h / 2) * (b + (b + 75)) = (1 / 2) * (h / 2) * ((b + 75) + (b + 150))) :
  ∃ n : ℤ, n = ⌊x^2 / 120⌋ ∧ n = 3000 := 
by 
  sorry

end trapezoid_midsegment_l11_11914


namespace circle_radius_l11_11186

theorem circle_radius (A : ℝ) (r : ℝ) (h : A = 36 * Real.pi) (h2 : A = Real.pi * r ^ 2) : r = 6 :=
sorry

end circle_radius_l11_11186


namespace max_value_a7_a14_l11_11862

noncomputable def arithmetic_sequence_max_product (a_1 d : ℝ) : ℝ :=
  let a_7 := a_1 + 6 * d
  let a_14 := a_1 + 13 * d
  a_7 * a_14

theorem max_value_a7_a14 {a_1 d : ℝ} 
  (h : 10 = 2 * a_1 + 19 * d)
  (sum_first_20 : 100 = (10) * (a_1 + a_1 + 19 * d)) :
  arithmetic_sequence_max_product a_1 d = 25 :=
by
  sorry

end max_value_a7_a14_l11_11862


namespace compare_apothems_l11_11705

noncomputable def length_of_rectangle_shorter_side (x : ℝ) : Prop :=
  2 * x^2 = 6 * x

noncomputable def length_of_rectangle_longer_side (x : ℝ) : ℝ :=
  2 * x

noncomputable def apothem_of_rectangle (x : ℝ) : ℝ :=
  x / 2

noncomputable def length_of_triangle_shorter_leg (y : ℝ) : Prop :=
  (3 * y^2 / 2) = 4 * y + sqrt 10 * y

noncomputable def apothem_of_triangle (y : ℝ) : ℝ :=
  (3 * y^2 / 2) / (4 * y + sqrt 10 * y)

theorem compare_apothems {x y : ℝ} (hx : length_of_rectangle_shorter_side x) (hy : length_of_triangle_shorter_leg y) :
  if apothem_of_rectangle x = apothem_of_triangle y then "equal"
  else if apothem_of_rectangle x < apothem_of_triangle y then "less"
  else "greater" :=
sorry

end compare_apothems_l11_11705


namespace sum_of_primes_final_sum_l11_11774

theorem sum_of_primes (p : ℕ) (hp : Nat.Prime p) :
  (¬ ∃ x : ℤ, 5 * (10 * x + 2) ≡ 6 [ZMOD p]) →
  p = 2 ∨ p = 5 :=
sorry

theorem final_sum :
  (∀ p : ℕ, Nat.Prime p → (¬ ∃ x : ℤ, 5 * (10 * x + 2) ≡ 6 [ZMOD p]) → p = 2 ∨ p = 5) →
  (2 + 5 = 7) :=
sorry

end sum_of_primes_final_sum_l11_11774


namespace solve_y_l11_11586

theorem solve_y : ∃ y : ℚ, 2 * y + 3 * y = 600 - (4 * y + 5 * y + 100) ∧ y = 250 / 7 := by
  sorry

end solve_y_l11_11586


namespace a6_b6_gt_a4b2_ab4_l11_11887

theorem a6_b6_gt_a4b2_ab4 {a b : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≠ b) :
  a^6 + b^6 > a^4 * b^2 + a^2 * b^4 :=
sorry

end a6_b6_gt_a4b2_ab4_l11_11887


namespace max_consecutive_new_is_3_l11_11703

-- Define the concept of "new" number
def is_new (n : ℕ) : Prop :=
  n > 5 ∧ ∃ m : ℕ, (∀ k, k < n → m % k = 0) ∧ m % n ≠ 0

-- Define the concept of consecutive "new" numbers
def max_consecutive_new (n : ℕ) : ℕ :=
  if is_new n ∧ is_new (n + 1) ∧ is_new (n + 2) then 3 else
  if is_new n ∧ is_new (n + 1) then 2 else
  if is_new n then 1 else 0

-- Prove that the maximum number of consecutive "new" numbers is 3
theorem max_consecutive_new_is_3 : 
    ∃ n : ℕ, max_consecutive_new n = 3 :=
by {
  -- The exact proof steps are omitted as the statement should compile correctly.
  sorry
}

end max_consecutive_new_is_3_l11_11703


namespace sin_intersections_ratios_l11_11194

theorem sin_intersections_ratios :
  ∃ (p q : ℕ), p < q ∧ Nat.gcd p q = 1 ∧
  ∀ x, (sin x = sin (80 * Real.pi / 180)) → 
       (x.support_slices (λ x => sin x) = [p, q, p, q]) →
        (p, q) = (1, 17) :=
by
  sorry

end sin_intersections_ratios_l11_11194


namespace count_even_three_digit_numbers_with_sum_12_l11_11012

noncomputable def even_three_digit_numbers_with_sum_12 : Prop :=
  let valid_pairs := [(8, 4), (6, 6), (4, 8)] in
  let valid_hundreds := 9 in
  let count_pairs := valid_pairs.length in
  let total_numbers := valid_hundreds * count_pairs in
  total_numbers = 27

theorem count_even_three_digit_numbers_with_sum_12 : even_three_digit_numbers_with_sum_12 :=
by
  sorry

end count_even_three_digit_numbers_with_sum_12_l11_11012


namespace sqrt_50_floor_squared_l11_11788

theorem sqrt_50_floor_squared : (⌊Real.sqrt 50⌋ : ℝ)^2 = 49 := by
  have sqrt_50_bounds : 7 < Real.sqrt 50 ∧ Real.sqrt 50 < 8 := by
    split
    · have : Real.sqrt 49 < Real.sqrt 50 := by sorry
      linarith
    · have : Real.sqrt 50 < Real.sqrt 64 := by sorry
      linarith
  have floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := by
    sorry
  rw [floor_sqrt_50]
  norm_num

end sqrt_50_floor_squared_l11_11788


namespace triangle_angle_degrees_l11_11455

variable (P Q R S : Type)
variable [MetricSpace P]
variables [Compass.PQ PR]
variables [Compass.ParallelPQ RS]
variables [AngleMeasure P R S = 50]

theorem triangle_angle_degrees
  (h1 : PQ = PR)
  (h2 : mangle PRS = 50)
  (h3 : RS ∥ PQ) :
  mangle QRS = 50 := sorry

end triangle_angle_degrees_l11_11455


namespace simplify_fraction_l11_11585

theorem simplify_fraction (a : ℝ) (h : a ≠ 2) : (3 - a) / (a - 2) + 1 = 1 / (a - 2) :=
by
  -- proof goes here
  sorry

end simplify_fraction_l11_11585


namespace total_animals_received_l11_11214

-- Define the conditions
def cats : ℕ := 40
def additionalCats : ℕ := 20
def dogs : ℕ := cats - additionalCats

-- Prove the total number of animals received
theorem total_animals_received : (cats + dogs) = 60 := by
  -- The proof itself is not required in this task
  sorry

end total_animals_received_l11_11214


namespace coordinates_of_P_l11_11101

def P : Prod Int Int := (-1, 2)

theorem coordinates_of_P :
  P = (-1, 2) := 
  by
    -- The proof is omitted as per instructions
    sorry

end coordinates_of_P_l11_11101


namespace remainder_when_sum_divided_by_7_l11_11066

theorem remainder_when_sum_divided_by_7 (a b c : ℕ) (ha : a < 7) (hb : b < 7) (hc : c < 7)
  (h1 : a * b * c ≡ 1 [MOD 7])
  (h2 : 4 * c ≡ 3 [MOD 7])
  (h3 : 5 * b ≡ 4 + b [MOD 7]) :
  (a + b + c) % 7 = 6 := by
  sorry

end remainder_when_sum_divided_by_7_l11_11066


namespace contest_possible_scores_l11_11676

theorem contest_possible_scores : 
  let scores := {s | ∃ (x y : ℕ), x ≤ 6 ∧ y ≤ 6 - x ∧ s = 7 * x + y} in
  scores.card = 28 :=
by
  sorry

end contest_possible_scores_l11_11676


namespace sin_A_correct_height_on_AB_correct_l11_11478

noncomputable def sin_A (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) : ℝ :=
  Real.sin A

noncomputable def height_on_AB (A B C AB : ℝ) (height : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) (h4 : AB = 5) : ℝ :=
  height

theorem sin_A_correct (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) : 
  sorrry := 
begin
  -- proof omitted
  sorrry
end

theorem height_on_AB_correct (A B C AB : ℝ) (height : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) (h4 : AB = 5) :
  height = 6:= 
begin
  -- proof omitted
  sorrry
end 

end sin_A_correct_height_on_AB_correct_l11_11478


namespace count_even_three_digit_numbers_l11_11039

theorem count_even_three_digit_numbers : 
  let num_even_three_digit_numbers : ℕ := 
    have h1 : (units_digit_possible_pairs : list (ℕ × ℕ)) := 
      [(4, 8), (6, 6), (8, 4)]
    have h2 : (number_of_hundreds_digits : ℕ) := 9
    3 * number_of_hundreds_digits 
in
  num_even_three_digit_numbers = 27 := by
  -- steps skipped
  sorry

end count_even_three_digit_numbers_l11_11039


namespace triangle_ABC_circumcenter_m_n_range_l11_11905

variable {R : Type*} [LinearOrderedField R]

theorem triangle_ABC_circumcenter_m_n_range (m n : R) (C A B O : R) (hC : C = 45)
  (ho : O = circumcenter A B C) (hvec: OC = m * OA + n * OB)
  : -sqrt(2) ≤ m + n ∧ m + n < 1 :=
sorry

end triangle_ABC_circumcenter_m_n_range_l11_11905


namespace Robie_chocolates_left_l11_11991

def initial_bags : ℕ := 3
def given_away : ℕ := 2
def additional_bags : ℕ := 3

theorem Robie_chocolates_left : (initial_bags - given_away) + additional_bags = 4 :=
by
  sorry

end Robie_chocolates_left_l11_11991


namespace cos_beta_value_l11_11403

def acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < π / 2

theorem cos_beta_value (α β : ℝ)
  (hα : acute_angle α) 
  (hβ : acute_angle β) 
  (hcosα : cos α = sqrt 5 / 5) 
  (hsin_alpha_beta : sin (α + β) = 3 / 5) :
  cos β = (2 * sqrt 5) / 25 :=
  sorry

end cos_beta_value_l11_11403


namespace range_of_f_smallest_positive_period_of_f_intervals_of_monotonic_increase_of_f_l11_11826

open Real

def f (x : ℝ) := 2 * sin x * sin x + 2 * sqrt 3 * sin x * cos x + 1

theorem range_of_f : Set.Icc 0 4 = Set.range f :=
sorry

theorem smallest_positive_period_of_f : ∀ x : ℝ, f (x + π) = f x :=
sorry

theorem intervals_of_monotonic_increase_of_f (k : ℤ) :
  Set.Icc (-π / 6 + k * π) (π / 3 + k * π) ⊆ {x : ℝ | ∃ (m : ℤ), deriv f x > 0} :=
sorry

end range_of_f_smallest_positive_period_of_f_intervals_of_monotonic_increase_of_f_l11_11826


namespace bucket_size_correct_l11_11534

def leakage_rate := 1.5 -- ounces per hour
def time_duration := 12 -- hours
def leakage_over_time := leakage_rate * time_duration
def bucket_size := 2 * leakage_over_time

theorem bucket_size_correct : bucket_size = 36 := by
  sorry

end bucket_size_correct_l11_11534


namespace tickets_used_l11_11681

def total_rides (ferris_wheel_rides bumper_car_rides : ℕ) : ℕ :=
  ferris_wheel_rides + bumper_car_rides

def tickets_per_ride : ℕ := 3

def total_tickets (total_rides tickets_per_ride : ℕ) : ℕ :=
  total_rides * tickets_per_ride

theorem tickets_used :
  total_tickets (total_rides 7 3) tickets_per_ride = 30 := by
  sorry

end tickets_used_l11_11681


namespace sqrt_50_floor_square_l11_11795

theorem sqrt_50_floor_square : ⌊Real.sqrt 50⌋ ^ 2 = 49 := by
  have h : 7 < Real.sqrt 50 ∧ Real.sqrt 50 < 8 := 
    by sorry
  have floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := 
    by sorry
  show ⌊Real.sqrt 50⌋ ^ 2 = 49
  from calc
    ⌊Real.sqrt 50⌋ ^ 2 = 7 ^ 2 : by rw [floor_sqrt_50]
    ... = 49 : by norm_num

end sqrt_50_floor_square_l11_11795


namespace sin_A_calculation_height_calculation_l11_11505

variable {A B C : ℝ}

-- Given conditions
def angle_condition : Prop := A + B = 3 * C
def sine_condition : Prop := 2 * sin (A - C) = sin B

-- Part 1: Find sin A
theorem sin_A_calculation (h1 : angle_condition) (h2 : sine_condition) : sin A = 3 * real.sqrt 10 / 10 := sorry

-- Part 2: Given AB = 5, find the height
variable {AB : ℝ}
def AB_value : Prop := AB = 5

theorem height_calculation (h1 : angle_condition) (h2 : sine_condition) (h3 : AB_value) : height = 6 := sorry

end sin_A_calculation_height_calculation_l11_11505


namespace floor_sqrt_50_squared_l11_11806

theorem floor_sqrt_50_squared : ∃ x : ℕ, x = 49 ∧ (⌊real.sqrt 50⌋ : ℕ) ^ 2 = x := by
  have h1 : (7 : ℝ) < real.sqrt 50 := sorry
  have h2 : real.sqrt 50 < 8 := sorry
  have h_floor : (⌊real.sqrt 50⌋ : ℕ) = 7 := sorry
  use 49
  constructor
  · rfl
  · rw [h_floor]
    norm_num
    sorry

end floor_sqrt_50_squared_l11_11806


namespace fraction_unseated_l11_11228

theorem fraction_unseated :
  ∀ (tables seats_per_table seats_taken : ℕ),
  tables = 15 →
  seats_per_table = 10 →
  seats_taken = 135 →
  ((tables * seats_per_table - seats_taken : ℕ) / (tables * seats_per_table : ℕ) : ℚ) = 1 / 10 :=
by
  intros tables seats_per_table seats_taken h_tables h_seats_per_table h_seats_taken
  sorry

end fraction_unseated_l11_11228


namespace tangential_quadrilateral_l11_11694

variables {A B C D I J : Type*}
variables [CyclicQuadrilateral A B C D]
variables [InscribedCircleCenter I A B C]
variables [InscribedCircleCenter J A D C]
variables (Concyclic : Concyclic B I J D)

theorem tangential_quadrilateral
    (h1 : Cyclic A B C D)
    (h2 : InscribedCircleCenter I A B C)
    (h3 : InscribedCircleCenter J A D C)
    (h4 : Concyclic B I J D) :
    Tangential A B C D :=
sorry

end tangential_quadrilateral_l11_11694


namespace maximal_elements_set_l11_11537

theorem maximal_elements_set 
  (A : Set ℕ) 
  (hA : ∀ x y ∈ A, x > y → x - y ≥ (x * y) / 25) :
  Fintype.card A ≤ 24 := 
sorry

end maximal_elements_set_l11_11537


namespace simplest_square_root_l11_11661

theorem simplest_square_root :
  (λ x, let y := x in y = sqrt 3) ∧ 
  (sqrt 3) = sqrt 3 ∧
  ∀ (y : ℝ), (y = sqrt (1 / 2)  ∨ y = sqrt 0.2 ∨ y = sqrt 3  ∨ y = sqrt 8) → 
  ( sqrt 3 = y → 
    ( ∃ (r : ℝ), y = 0 ∨ y = 1 ∨ y = r)
  )
:=
begin
  sorry
end

end simplest_square_root_l11_11661


namespace darren_total_tshirts_l11_11334

def num_white_packs := 5
def num_white_tshirts_per_pack := 6
def num_blue_packs := 3
def num_blue_tshirts_per_pack := 9

def total_tshirts (wpacks : ℕ) (wtshirts_per_pack : ℕ) (bpacks : ℕ) (btshirts_per_pack : ℕ) : ℕ :=
  (wpacks * wtshirts_per_pack) + (bpacks * btshirts_per_pack)

theorem darren_total_tshirts : total_tshirts num_white_packs num_white_tshirts_per_pack num_blue_packs num_blue_tshirts_per_pack = 57 :=
by
  -- proof needed
  sorry

end darren_total_tshirts_l11_11334


namespace green_ball_probability_l11_11332

def containerA := (8, 2) -- 8 green, 2 red
def containerB := (6, 4) -- 6 green, 4 red
def containerC := (5, 5) -- 5 green, 5 red
def containerD := (8, 2) -- 8 green, 2 red

def probability_of_green : ℚ :=
  (1 / 4) * (8 / 10) + (1 / 4) * (6 / 10) + (1 / 4) * (5 / 10) + (1 / 4) * (8 / 10)
  
theorem green_ball_probability :
  probability_of_green = 43 / 160 :=
sorry

end green_ball_probability_l11_11332


namespace count_distinct_m_values_l11_11957

theorem count_distinct_m_values : 
  ∃ m_values : Finset ℤ, 
  (∀ x1 x2 : ℤ, x1 * x2 = 30 → (m_values : Set ℤ) = { x1 + x2 }) ∧ 
  m_values.card = 8 :=
by
  sorry

end count_distinct_m_values_l11_11957


namespace int_solve_ineq_l11_11433

theorem int_solve_ineq (x : ℤ) : (x + 3)^3 ≤ 8 ↔ x ≤ -1 :=
by sorry

end int_solve_ineq_l11_11433


namespace find_function_expression_l11_11869

variable (f : ℝ → ℝ)
variable (P : ℝ → ℝ → ℝ)

-- conditions
axiom a1 : f 1 = 1
axiom a2 : ∀ (x y : ℝ), f (x + y) = f x + f y + 2 * y * (x + y) + 1

-- proof statement
theorem find_function_expression (x : ℕ) (h : x ≠ 0) : f x = x^2 + 3*x - 3 := sorry

end find_function_expression_l11_11869


namespace probability_of_one_or_two_in_pascal_l11_11727

def pascal_triangle_element_probability : ℚ :=
  let total_elements := 210 -- sum of the elements in the first 20 rows
  let ones_count := 39      -- total count of 1s in the first 20 rows
  let twos_count := 36      -- total count of 2s in the first 20 rows
  let favorable_elements := ones_count + twos_count
  favorable_elements / total_elements

theorem probability_of_one_or_two_in_pascal (n : ℕ) (h : n = 20) :
  pascal_triangle_element_probability = 5 / 14 := by
  rw [h]
  dsimp [pascal_triangle_element_probability]
  sorry

end probability_of_one_or_two_in_pascal_l11_11727


namespace amount_after_3_years_l11_11562

-- Define the initial conditions
def P0 : ℝ := 500
def r : ℝ := 0.02
def additional_investment : ℝ := 500

-- Define the function to calculate the amount after n years
def amount_after_years (n : ℕ) : ℝ :=
  let rec aux (current_year : ℕ) (current_amount : ℝ) : ℝ :=
    if current_year = 0 then current_amount
    else aux (current_year - 1) (current_amount * (1 + r) + additional_investment)
  in aux n P0

-- Prove that the amount after 3 years is 2060.80
theorem amount_after_3_years : amount_after_years 3 = 2060.80 := 
    by 
      sorry

end amount_after_3_years_l11_11562


namespace sine_function_properties_l11_11750

theorem sine_function_properties
  (a b : ℝ) (φ : ℝ)
  (h1 : 0 < a) (h2 : 0 < b)
  (h3 : ∀ x : ℝ, a * sin (b * x + φ) ≤ 3)
  (h4 : ∃ x : ℝ, a * sin (b * x + φ) = 3)
  (h5 : ∀ x, a * sin (b * x + φ) = a * sin (b * (x + (π / 2)) + φ)) :
  a = 3 ∧ b = 4 :=
by
  sorry

end sine_function_properties_l11_11750


namespace sin_A_correct_height_on_AB_correct_l11_11474

noncomputable def sin_A (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) : ℝ :=
  Real.sin A

noncomputable def height_on_AB (A B C AB : ℝ) (height : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) (h4 : AB = 5) : ℝ :=
  height

theorem sin_A_correct (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) : 
  sorrry := 
begin
  -- proof omitted
  sorrry
end

theorem height_on_AB_correct (A B C AB : ℝ) (height : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) (h4 : AB = 5) :
  height = 6:= 
begin
  -- proof omitted
  sorrry
end 

end sin_A_correct_height_on_AB_correct_l11_11474


namespace inv_g_squared_l11_11345

def g (x : ℝ) : ℝ := 25 / (7 + 4 * x)

theorem inv_g_squared (h : ∃ y : ℝ, g y = 3) : (∃ y : ℝ, g y = 3) → (y^(-2)) = 9 :=
by
  sorry

end inv_g_squared_l11_11345


namespace time_difference_leak_l11_11701

/-- 
The machine usually fills one barrel in 3 minutes. 
However, with a leak, it takes 5 minutes to fill one barrel. 
Given that it takes 24 minutes longer to fill 12 barrels with the leak, prove that it will take 2n minutes longer to fill n barrels with the leak.
-/
theorem time_difference_leak (n : ℕ) : 
  (3 * 12 + 24 = 5 * 12) →
  (5 * n) - (3 * n) = 2 * n :=
by
  intros h
  sorry

end time_difference_leak_l11_11701


namespace trays_from_first_table_is_23_l11_11642

-- Definitions of conditions
def trays_per_trip : ℕ := 7
def trips_made : ℕ := 4
def trays_from_second_table : ℕ := 5

-- Total trays carried
def total_trays_carried : ℕ := trays_per_trip * trips_made

-- Number of trays picked from first table
def trays_from_first_table : ℕ :=
  total_trays_carried - trays_from_second_table

-- Theorem stating that the number of trays picked up from the first table is 23
theorem trays_from_first_table_is_23 : trays_from_first_table = 23 := by
  sorry

end trays_from_first_table_is_23_l11_11642


namespace travel_distance_in_yards_l11_11720

-- Define the constants and conditions given in the problem
variables (a r : ℝ)
constant meter_to_yard : ℝ := 1.09361
constant distance_in_meters : ℝ := a / 6
constant travel_time : ℝ := 5
constant travel_rate_in_yards_per_minute : ℝ := (a * meter_to_yard) / (6 * r)
constant expected_distance_in_yards : ℝ := (5.46805 * a) / (6 * r)

-- Lean 4 statement for the proof
theorem travel_distance_in_yards :
  (a / 6 / r * meter_to_yard * travel_time) = expected_distance_in_yards := 
sorry

end travel_distance_in_yards_l11_11720


namespace number_of_valid_three_digit_even_numbers_l11_11000

def valid_three_digit_even_numbers (n : ℕ) : Prop :=
  (100 ≤ n) ∧ (n < 1000) ∧ (n % 2 = 0) ∧ (let t := (n / 10) % 10 in
                                           let u := n % 10 in
                                           t + u = 12)

theorem number_of_valid_three_digit_even_numbers : 
  (∃ cnt : ℕ, cnt = 27 ∧ (cnt = (count (λ n, valid_three_digit_even_numbers n) (Ico 100 1000)))) :=
sorry

end number_of_valid_three_digit_even_numbers_l11_11000


namespace minimum_value_part1_minimum_value_part2_l11_11963

open Complex

noncomputable def minimization_problem_part1 (z1 z2 : ℂ) (h1 : z1.re > 0) (h2 : z2.re > 0)
  (h3 : (z1^2).re = 2) (h4 : (z2^2).re = 2) : ℝ :=
  ⨅ (z1 z2 : ℂ) (h1 : z1.re > 0) (h2 : z2.re > 0) (h3 : (z1^2).re = 2) (h4 : (z2^2).re = 2), 
     (z1 * z2).re

theorem minimum_value_part1 (z1 z2 : ℂ) (h1 : z1.re > 0) (h2 : z2.re > 0)
  (h3 : (z1^2).re = 2) (h4 : (z2^2).re = 2) : minimization_problem_part1 z1 z2 h1 h2 h3 h4 = 2 :=
sorry

noncomputable def minimization_problem_part2 (z1 z2 : ℂ) (h1 : z1.re > 0) (h2 : z2.re > 0)
  (h3 : (z1^2).re = 2) (h4 : (z2^2).re = 2) : ℝ :=
  ⨅ (z1 z2 : ℂ) (h1 : z1.re > 0) (h2 : z2.re > 0) (h3 : (z1^2).re = 2) (h4 : (z2^2).re = 2),
    abs ((z1 + 2 : ℂ)) + abs ((conj(z2) + 2 : ℂ)) - abs ((conj(z1) - z2 : ℂ))

theorem minimum_value_part2 (z1 z2 : ℂ) (h1 : z1.re > 0) (h2 : z2.re > 0)
  (h3 : (z1^2).re = 2) (h4 : (z2^2).re = 2) : minimization_problem_part2 z1 z2 h1 h2 h3 h4 = 4 * Real.sqrt 2 :=
sorry

end minimum_value_part1_minimum_value_part2_l11_11963


namespace part1_geometric_sequence_part2_sum_of_terms_l11_11219

/- Part 1 -/
theorem part1_geometric_sequence (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h₀ : a 1 = 3) 
  (h₁ : ∀ n, a (n + 1) = a n ^ 2 + 2 * a n) 
  (h₂ : ∀ n, 2 ^ b n = a n + 1) :
  ∃ r, ∀ n, b (n + 1) = r * b n ∧ r = 2 :=
by 
  use 2 
  sorry

/- Part 2 -/
theorem part2_sum_of_terms (b : ℕ → ℝ) (c : ℕ → ℝ) (T : ℕ → ℝ) 
  (h₀ : ∀ n, b n = 2 ^ n)
  (h₁ : ∀ n, c n = n / b n + 1) :
  ∀ n, T n = n + 2 - (n + 2) / 2 ^ n :=
by
  sorry

end part1_geometric_sequence_part2_sum_of_terms_l11_11219


namespace math_problem_l11_11839

theorem math_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  (ab_ineq : a * b ≤ 4) ∧
  (harmonic_ineq : ¬ (1 / a + 1 / b ≤ 1)) ∧
  (sqrt_ineq : sqrt a + sqrt b ≤ 2 * sqrt 2) ∧
  (sum_squares_ineq : a * a + b * b ≥ 8) :=
by sorry

end math_problem_l11_11839


namespace volume_ratio_cylinder_ball_l11_11273

/-- Definition of a cylinder and ball setup such that the ball is tangent to the sides, and top and bottom of the cylinder. -/
def cylinder_contains_ball (r : ℝ) (V_cylinder V_ball : ℝ) : Prop := 
  let V_cyl := 2 * π * r^3
  let V_ball := (4/3) * π * r^3
  V_cylinder = V_cyl ∧ V_ball = V_ball

/-- Theorem: The ratio of the volume of the cylinder to the volume of the ball is 3/2. -/
theorem volume_ratio_cylinder_ball {r V_cylinder V_ball : ℝ}
    (h : cylinder_contains_ball r V_cylinder V_ball) :
    V_cylinder / V_ball = 3 / 2 :=
  sorry

end volume_ratio_cylinder_ball_l11_11273


namespace exists_k_with_half_distinct_remainders_l11_11551

theorem exists_k_with_half_distinct_remainders 
  (p : ℕ) (h_prime : Nat.Prime p) (a : Fin p → ℤ) :
  ∃ k : ℤ, (Finset.univ.image (λ (i : Fin p), (a i + ↑i * k) % p)).card ≥ p / 2 := by
  sorry

end exists_k_with_half_distinct_remainders_l11_11551


namespace smallest_not_prime_l11_11128

open Nat

def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | k + 1 => 2 * seq k + 1

theorem smallest_not_prime : ∃ (m : ℕ), seq m = 95 ∧ ¬ Prime 95 := by
  sorry

end smallest_not_prime_l11_11128


namespace checkerboard_black_squares_count_l11_11759

theorem checkerboard_black_squares_count :
  let n := 32 in
  let total_squares := n * n in
  let black_squares := total_squares / 2 in
  n % 2 = 0 →
  black_squares = 512 :=
by
  intros n total_squares black_squares hn_even
  have h_total : total_squares = 1024 := by
    simp only [total_squares]
    norm_num
  rw h_total
  have h_black : black_squares = 1024 / 2 := by
    simp only [black_squares]
    norm_num
  rw h_black
  norm_num
  sorry

end checkerboard_black_squares_count_l11_11759


namespace list_price_of_shirt_l11_11300

theorem list_price_of_shirt
  (final_price : ℝ) (first_discount : ℝ) (second_discount : ℝ)
  (H1 : final_price = 105)
  (H2 : first_discount = 19.954259576901087)
  (H3 : second_discount = 12.55) :
  let P := 150 in
  final_price = (P * (1 - first_discount / 100) * (1 - second_discount / 100)) :=
by 
  sorry

end list_price_of_shirt_l11_11300


namespace max_popsicles_l11_11977

def popsicles : ℕ := 1
def box_3 : ℕ := 3
def box_5 : ℕ := 5
def box_10 : ℕ := 10
def cost_popsicle : ℕ := 1
def cost_box_3 : ℕ := 2
def cost_box_5 : ℕ := 3
def cost_box_10 : ℕ := 4
def budget : ℕ := 10

theorem max_popsicles : 
  ∀ (popsicle_count : ℕ) (b3_count : ℕ) (b5_count : ℕ) (b10_count : ℕ),
    popsicle_count * cost_popsicle + b3_count * cost_box_3 + b5_count * cost_box_5 + b10_count * cost_box_10 ≤ budget →
    popsicle_count * popsicles + b3_count * box_3 + b5_count * box_5 + b10_count * box_10 ≤ 23 →
    ∃ p b3 b5 b10, popsicle_count = p ∧ b3_count = b3 ∧ b5_count = b5 ∧ b10_count = b10 ∧
    (p * cost_popsicle + b3 * cost_box_3 + b5 * cost_box_5 + b10 * cost_box_10 ≤ budget) ∧
    (p * popsicles + b3 * box_3 + b5 * box_5 + b10 * box_10 = 23) :=
by sorry

end max_popsicles_l11_11977


namespace trihedral_angle_all_acute_l11_11356

def three_dim_space : Type := Euclidean_space -- Define a place holder for the 3D space
def planar_triangles_forming_acute (a b c : ℝ) : Prop := a^2 + b^2 > c^2

-- Trihedral angle with all planar sections forming acute-angled triangles
theorem trihedral_angle_all_acute
  (three_dim_space: Euclidean_space) 
  (S A B C : three_dim_space) 
  (ha : 0 < ∠ A S B < π / 2)
  (hb : 0 < ∠ B S C < π / 2)
  (hc : 0 < ∠ C S A < π / 2)
  (all_sections_acute : ∀ (P Q R: three_dim_space), planar_triangles_forming_acute (dist P Q) (dist Q R) (dist R P))
  : (0 < ∠ A S B < π / 2) ∧ (0 < ∠ B S C < π / 2) ∧ (0 < ∠ C S A < π / 2) :=
sorry

end trihedral_angle_all_acute_l11_11356


namespace eval_expression_l11_11656

theorem eval_expression : 7^3 + 3 * 7^2 + 3 * 7 + 1 = 512 := 
by 
  sorry

end eval_expression_l11_11656


namespace estimate_students_spending_more_than_60_l11_11599

-- Definition of the problem
def students_surveyed : ℕ := 50
def students_inclined_to_subscribe : ℕ := 8
def total_students : ℕ := 1000
def estimated_students : ℕ := 600

-- Define the proof task
theorem estimate_students_spending_more_than_60 :
  (students_inclined_to_subscribe : ℝ) / (students_surveyed : ℝ) * (total_students : ℝ) = estimated_students :=
by
  sorry

end estimate_students_spending_more_than_60_l11_11599


namespace engineering_department_men_l11_11110

theorem engineering_department_men (total_students men_percentage women_count : ℕ) (h_percentage : men_percentage = 70) (h_women : women_count = 180) (h_total : total_students = (women_count * 100) / (100 - men_percentage)) : 
  (total_students * men_percentage / 100) = 420 :=
by
  sorry

end engineering_department_men_l11_11110


namespace find_radius_squared_l11_11691

theorem find_radius_squared (r : ℝ) (AB_len CD_len BP : ℝ) (angle_APD : ℝ) (h1 : AB_len = 12)
    (h2 : CD_len = 9) (h3 : BP = 10) (h4 : angle_APD = 60) : r^2 = 111 := by
  have AB_len := h1
  have CD_len := h2
  have BP := h3
  have angle_APD := h4
  sorry

end find_radius_squared_l11_11691


namespace event_eq_conds_l11_11233

-- Definitions based on conditions
def Die := { n : ℕ // 1 ≤ n ∧ n ≤ 6 }
def sum_points (d1 d2 : Die) : ℕ := d1.val + d2.val

def event_xi_eq_4 (d1 d2 : Die) : Prop := 
  sum_points d1 d2 = 4

def condition_a (d1 d2 : Die) : Prop := 
  d1.val = 2 ∧ d2.val = 2

def condition_b (d1 d2 : Die) : Prop := 
  (d1.val = 3 ∧ d2.val = 1) ∨ (d1.val = 1 ∧ d2.val = 3)

def event_condition (d1 d2 : Die) : Prop :=
  condition_a d1 d2 ∨ condition_b d1 d2

-- The main Lean statement
theorem event_eq_conds (d1 d2 : Die) : 
  event_xi_eq_4 d1 d2 ↔ event_condition d1 d2 := 
by
  sorry

end event_eq_conds_l11_11233


namespace gcd_210_162_l11_11240

-- Define the numbers
def a := 210
def b := 162

-- The proposition we need to prove: The GCD of 210 and 162 is 6
theorem gcd_210_162 : Nat.gcd a b = 6 :=
by
  sorry

end gcd_210_162_l11_11240


namespace general_term_of_sequence_l11_11389

-- Definitions of geometric sequence and properties
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- The sum of the first n terms of a geometric sequence
def sum_geometric_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum a

theorem general_term_of_sequence (a : ℕ → ℝ) (q : ℝ)
  (h1: q < 1)
  (h2: a 3 = 1)
  (h3: sum_geometric_sequence a 4 = 5 * sum_geometric_sequence a 2) :
  (∀ n : ℕ, a n = 2 * (-1) ^ (n - 1)) ∨ (∀ n : ℕ, a n = (1 / 2) * (-2) ^ (n - 1)) :=
sorry

end general_term_of_sequence_l11_11389


namespace sin_A_correct_height_on_AB_correct_l11_11477

noncomputable def sin_A (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) : ℝ :=
  Real.sin A

noncomputable def height_on_AB (A B C AB : ℝ) (height : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) (h4 : AB = 5) : ℝ :=
  height

theorem sin_A_correct (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) : 
  sorrry := 
begin
  -- proof omitted
  sorrry
end

theorem height_on_AB_correct (A B C AB : ℝ) (height : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) (h4 : AB = 5) :
  height = 6:= 
begin
  -- proof omitted
  sorrry
end 

end sin_A_correct_height_on_AB_correct_l11_11477


namespace probability_of_one_or_two_l11_11721

/-- Represents the number of elements in the first 20 rows of Pascal's Triangle. -/
noncomputable def total_elements : ℕ := 210

/-- Represents the number of ones in the first 20 rows of Pascal's Triangle. -/
noncomputable def number_of_ones : ℕ := 39

/-- Represents the number of twos in the first 20 rows of Pascal's Triangle. -/
noncomputable def number_of_twos : ℕ :=18

/-- Prove that the probability of randomly choosing an element which is either 1 or 2
from the first 20 rows of Pascal's Triangle is 57/210. -/
theorem probability_of_one_or_two (h1 : total_elements = 210)
                                  (h2 : number_of_ones = 39)
                                  (h3 : number_of_twos = 18) :
    39 + 18 = 57 ∧ (57 : ℚ) / 210 = 57 / 210 :=
by {
    sorry
}

end probability_of_one_or_two_l11_11721


namespace circumcircles_intersect_at_steiner_point_l11_11986

variable {α : Type*} [EuclideanSpace α]

def Steiner_point (A B C : α) : α := sorry -- Assume a definition of the Steiner point based on given points A, B, C

def Euler_line (A B C : α) : Line α := sorry -- Assume a definition of the Euler line of ΔABC

def circumcircle (A B C : α) : Circle α := sorry -- Assume a definition of the circumcircle of ΔABC

def N_b (A B C N_b : α) (line_BC : Line α) : Prop :=
    Foot_perpendicular B C N_b -- Definition of point N_b projection on line BC from given point

def N_c (A B C N_c : α) (line_CA : Line α) : Prop :=
    Foot_perpendicular C A N_c -- Definition of point N_c projection on line CA from given point

def N_a (A B C N_a : α) (line_AB : Line α) : Prop :=
    Foot_perpendicular A B N_a -- Definition of point N_a projection on line AB from given point

theorem circumcircles_intersect_at_steiner_point 
  (A B C N_a N_b N_c : α)
  (hN_b : N_b (B C N_b) (line B C))
  (hN_c : N_c (C A N_c) (line C A))
  (hN_a : N_a (A B N_a) (line A B))
  (hEuler : E = Steiner_point A B C) :
  E ∈ circumcircle A N_b N_c ∧ E ∈ circumcircle B N_c N_a ∧ E ∈ circumcircle C N_a N_b :=
  sorry

end circumcircles_intersect_at_steiner_point_l11_11986


namespace real_part_of_z_l11_11684

-- Condition: (1+i)z = 2
-- Question: Prove the real part of z is 1
theorem real_part_of_z (z : ℂ) (hz : (1 + complex.I) * z = 2) : z.re = 1 :=
sorry

end real_part_of_z_l11_11684


namespace number_of_fixed_points_upto_1988_l11_11292

noncomputable def f : ℕ → ℕ
| 1       := 1
| 3       := 3
| (2*n)   := f n
| (4*n+1) := 2 * f (2*n+1) - f n
| (4*n+3) := 3 * f (2*n+1) - 2 * f n
| _       := 0 -- to handle non-positive inputs gracefully

theorem number_of_fixed_points_upto_1988 : 
  { n : ℕ | 1 ≤ n ∧ n ≤ 1988 ∧ f n = n }.toFinset.card = 92 :=
sorry

end number_of_fixed_points_upto_1988_l11_11292


namespace find_D_l11_11165

-- Definitions of points as conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨2, 5⟩
def B : Point := ⟨4, 9⟩
def C : Point := ⟨6, 5⟩
def D : Point := ⟨4, 1⟩

def reflect_x (p : Point) : Point :=
  ⟨p.x, -p.y⟩

def reflect_y_eq_neg_x_add1 (p : Point) : Point :=
  let translated := ⟨p.x, p.y - 1⟩
  let reflected := ⟨-translated.y, -translated.x⟩
  ⟨reflected.x, reflected.y + 1⟩

def D' := reflect_x D
def D'' := reflect_y_eq_neg_x_add1 D'

-- The theorem that verifies the transformation result
theorem find_D'' : D'' = ⟨2, -3⟩ :=
  sorry

end find_D_l11_11165


namespace amare_fabric_needed_l11_11162

-- Definitions for the conditions
def fabric_per_dress_yards : ℝ := 5.5
def number_of_dresses : ℕ := 4
def fabric_owned_feet : ℝ := 7
def yard_to_feet : ℝ := 3

-- Total fabric needed in yards
def total_fabric_needed_yards : ℝ := fabric_per_dress_yards * number_of_dresses

-- Total fabric needed in feet
def total_fabric_needed_feet : ℝ := total_fabric_needed_yards * yard_to_feet

-- Fabric still needed
def fabric_still_needed : ℝ := total_fabric_needed_feet - fabric_owned_feet

-- Proof
theorem amare_fabric_needed : fabric_still_needed = 59 := by
  sorry

end amare_fabric_needed_l11_11162


namespace find_k_l11_11845

-- Conditions given
variables {x y k : ℝ} (h_line : k > 0) (h_on_line : k * x + y + 4 = 0)
variables (C : ℝ × ℝ := (0,1)) (circle_eq : x^2 + y^2 - 2 * y = 0)
variables (min_area : 2)

theorem find_k :
  ∃ k > 0, ∀ x y (h_on_line : k * x + y + 4 = 0)
    (circle_cond : x^2 + y^2 - 2 * y = 0),
    (minimum_area PACB = 2 → k = 2) :=
sorry

end find_k_l11_11845


namespace angle_parallel_l11_11452

variable (a b : ℝ)

-- Definitions based on conditions
def are_corresponding_sides_parallel (a b : ℝ) : Prop :=
∃ (u v w x : ℝ), (u ≠ 0 ∧ v ≠ 0 ∧ w ≠ 0 ∧ x ≠ 0) ∧ 
                 (u / sqrt (u^2 + v^2) = w / sqrt (w^2 + x^2)) ∧ 
                 (v / sqrt (u^2 + v^2) = x / sqrt (w^2 + x^2))

-- Problem in Lean 4 statement
theorem angle_parallel (h_parallel : are_corresponding_sides_parallel a b) (h_a : a = 60) : b = 60 ∨ b = 120 :=
by
  sorry

end angle_parallel_l11_11452


namespace tens_digit_of_8_pow_2013_l11_11654

theorem tens_digit_of_8_pow_2013 : 
  let n := 8^2013 in (n % 100) / 10 = 8 := 
sorry

end tens_digit_of_8_pow_2013_l11_11654


namespace problem_1_problem_2_l11_11428

-- Definitions for the sets A and B
def A : Set ℝ := { x | x^2 - 4x - 5 ≤ 0 }
def B (m : ℝ) : Set ℝ := { x | x^2 - 2x - m < 0 }

-- Problem 1
theorem problem_1 : A ∩ {x : ℝ | x ≤ -1 ∨ x ≥ 3} = {x : ℝ | x = -1 ∨ 3 ≤ x ∧ x ≤ 5} :=
by 
  have m := 3
  sorry

-- Problem 2
theorem problem_2 (m : ℝ) : (A ∩ B m = {x : ℝ | -1 ≤ x ∧ x < 4}) → m = 8 :=
by
  sorry

end problem_1_problem_2_l11_11428


namespace exists_permutation_sum_zero_l11_11135

def is_not_power_of_2 (n : ℕ) : Prop :=
  ∀ k : ℕ, n ≠ 2^k

theorem exists_permutation_sum_zero (n : ℕ) (h1 : n > 2) (h2 : is_not_power_of_2 n) :
  ∃ (a : ℕ → ℕ), (∀ i : ℕ, i < n → a i ∈ Finset.range (n+1)) ∧ (∀ i j : ℕ, i < n → j < n → i ≠ j → a i ≠ a j) ∧ (∑ k in Finset.range n, a k * Real.cos (2 * k * Real.pi / n)) = 0 :=
sorry

end exists_permutation_sum_zero_l11_11135


namespace probability_of_one_or_two_l11_11722

/-- Represents the number of elements in the first 20 rows of Pascal's Triangle. -/
noncomputable def total_elements : ℕ := 210

/-- Represents the number of ones in the first 20 rows of Pascal's Triangle. -/
noncomputable def number_of_ones : ℕ := 39

/-- Represents the number of twos in the first 20 rows of Pascal's Triangle. -/
noncomputable def number_of_twos : ℕ :=18

/-- Prove that the probability of randomly choosing an element which is either 1 or 2
from the first 20 rows of Pascal's Triangle is 57/210. -/
theorem probability_of_one_or_two (h1 : total_elements = 210)
                                  (h2 : number_of_ones = 39)
                                  (h3 : number_of_twos = 18) :
    39 + 18 = 57 ∧ (57 : ℚ) / 210 = 57 / 210 :=
by {
    sorry
}

end probability_of_one_or_two_l11_11722


namespace tan_alpha_minus_beta_l11_11099

/-- Given angles α and β both starting from Ox and their terminal sides being symmetric about the y-axis.
If the terminal side of angle α passes through the point (3, 4), then prove that tan(α - β) = -24/7. -/
theorem tan_alpha_minus_beta (α β : ℝ)
  (h1 : α ≠ β ∨ (α = 0 ∧ β = 0 ∧ false))
  (h2 : ∃ (p : ℝ × ℝ), p = (3, 4) ∧ α = real.atan (p.2 / p.1))
  (h3 : ∃ (q : ℝ × ℝ), q = (-3, 4) ∧ β = real.atan (q.2 / q.1)) :
  real.tan (α - β) = -24 / 7 :=
sorry

end tan_alpha_minus_beta_l11_11099


namespace vector_difference_parallelogram_l11_11097

section ParallelogramVectors

variables (A B C D : Type) [AddCommGroup A] [VectorSpace ℝ A]
variables (vAB vBC vCD vDA vAC vCB vBD : A)
variables (is_parallelogram : 
  vAB + vBC = vAC ∧ 
  vAD + vDC = vAC ∧ 
  vBC = vCB ∧
  vAB = vDC ∧
  vAD = vBC)

theorem vector_difference_parallelogram : 
  vAC - vBC = vDC :=
by sorry

end ParallelogramVectors

end vector_difference_parallelogram_l11_11097


namespace exists_k_with_half_distinct_remainders_l11_11550

theorem exists_k_with_half_distinct_remainders 
  (p : ℕ) (h_prime : Nat.Prime p) (a : Fin p → ℤ) :
  ∃ k : ℤ, (Finset.univ.image (λ (i : Fin p), (a i + ↑i * k) % p)).card ≥ p / 2 := by
  sorry

end exists_k_with_half_distinct_remainders_l11_11550


namespace exradii_altitude_inequality_l11_11935

theorem exradii_altitude_inequality (h_a h_b h_c r_a r_b r_c : ℝ) (a b c : ℝ) (Δ s R r : ℝ)
  (h_aa : h_a = 2 * Δ / a)
  (h_bb : h_b = 2 * Δ / b)
  (h_cc : h_c = 2 * Δ / c)
  (r_aa : r_a = s * Real.tan (Real.angle_A / 2))
  (r_bb : r_b = s * Real.tan (Real.angle_B / 2))
  (r_cc : r_c = s * Real.tan (Real.angle_C / 2))
  (angle_A angle_B angle_C : ℝ)
  (triangle_sum : angle_A + angle_B + angle_C = π) :
  (r_a / h_a + r_b / h_b + r_c / h_c) ≥ 3 :=
by
  sorry

end exradii_altitude_inequality_l11_11935


namespace even_three_digit_numbers_l11_11049

theorem even_three_digit_numbers (n : ℕ) :
  (n >= 100 ∧ n < 1000) ∧
  (n % 2 = 0) ∧
  ((n % 100) / 10 + (n % 10) = 12) →
  n = 12 :=
sorry

end even_three_digit_numbers_l11_11049


namespace intersection_P_Q_l11_11878

def P : set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
def Q : set ℝ := {y | ∃ x : ℝ, y = x + 1}

theorem intersection_P_Q :
  P ∩ Q = {y | y ≥ 1} := 
sorry

end intersection_P_Q_l11_11878


namespace general_terms_sum_of_first_n_terms_of_c_sqrt_inequality_l11_11408

-- Definitions
def arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d c, ∀ n, a n = d * (n : ℤ) + c
def sum_of_first_n_terms (b S : ℕ → ℤ) : Prop := ∀ n, S n = 2 * (b n - 1)

variables {a b S : ℕ → ℤ}

-- Given conditions
axiom H1 : arithmetic_sequence a
axiom H2 : sum_of_first_n_terms b S
axiom H3 : a 2 = b 1 - 1
axiom H4 : a 5 = b 3 - 1

-- Proof statements
theorem general_terms :
  (∀ n, a n = 2 * n - 3) ∧ (∀ n, b n = 2 ^ n) := sorry

definition c (n : ℕ) : ℤ := a n * b n

noncomputable def T (n : ℕ) : ℤ := (2 * n - 5) * 2 ^ (n + 1) + 10

theorem sum_of_first_n_terms_of_c :
  ∀ n, T n = (2 * n - 5) * 2 ^ (n + 1) + 10 := sorry

theorem sqrt_inequality :
  ∀ n, n ≥ 2 → (∑ i in range n, real.sqrt (1 / (a i + 2))) > real.sqrt n := sorry

end general_terms_sum_of_first_n_terms_of_c_sqrt_inequality_l11_11408


namespace work_completion_l11_11288

theorem work_completion (A_days : ℕ) (B_days : ℕ) (work_completed_A : ℕ) (work_left_B : ℚ) (B_completion_days : ℕ) :
  A_days = 15 ∧ B_days = 9 ∧ work_completed_A = 5 ∧ work_left_B = 2/3 ∧ B_completion_days = 6 :=
begin
  sorry
end

end work_completion_l11_11288


namespace fire_brigade_distribution_l11_11457

theorem fire_brigade_distribution : 
  let brigades := 4
  let sites := 3
  let min_brigades_per_site := 1
  (∀ site : Fin sites, site ≥ min_brigades_per_site) -> 
  count_distribution_schemes (brigades - sites + 1) (sites) = 3 := 
by sorry

end fire_brigade_distribution_l11_11457


namespace proof_triangle_properties_l11_11490

variable (A B C : ℝ)
variable (h AB : ℝ)

-- Conditions
def triangle_conditions : Prop :=
  (A + B = 3 * C) ∧ (2 * Real.sin (A - C) = Real.sin B) ∧ (AB = 5)

-- Part 1: Proving sin A
def find_sin_A (h₁ : triangle_conditions A B C h AB) : Prop :=
  Real.sin A = 3 * Real.cos A

-- Part 2: Proving the height on side AB
def find_height_on_AB (h₁ : triangle_conditions A B C h AB) : Prop :=
  h = 6

-- Combined proof statement
theorem proof_triangle_properties (h₁ : triangle_conditions A B C h AB) : 
  find_sin_A A B C h₁ ∧ find_height_on_AB A B C h AB h₁ := 
  by sorry

end proof_triangle_properties_l11_11490


namespace sin_A_calculation_height_calculation_l11_11499

variable {A B C : ℝ}

-- Given conditions
def angle_condition : Prop := A + B = 3 * C
def sine_condition : Prop := 2 * sin (A - C) = sin B

-- Part 1: Find sin A
theorem sin_A_calculation (h1 : angle_condition) (h2 : sine_condition) : sin A = 3 * real.sqrt 10 / 10 := sorry

-- Part 2: Given AB = 5, find the height
variable {AB : ℝ}
def AB_value : Prop := AB = 5

theorem height_calculation (h1 : angle_condition) (h2 : sine_condition) (h3 : AB_value) : height = 6 := sorry

end sin_A_calculation_height_calculation_l11_11499


namespace triangle_third_side_length_l11_11574

theorem triangle_third_side_length (A B C : Type) 
  (AB : ℝ) (AC : ℝ) 
  (angle_ABC angle_ACB : ℝ) 
  (BC : ℝ) 
  (h1 : AB = 7) 
  (h2 : AC = 21) 
  (h3 : angle_ABC = 3 * angle_ACB) 
  : 
  BC = (some_correct_value ) := 
sorry

end triangle_third_side_length_l11_11574


namespace profit_increase_l11_11896

theorem profit_increase (x y : ℝ) (a : ℝ) (hx_pos : x > 0) (hy_pos : y > 0)
  (profit_eq : y - x = x * (a / 100))
  (new_profit_eq : y - 0.95 * x = 0.95 * x * (a / 100) + 0.95 * x * (15 / 100)) :
  a = 185 :=
by
  sorry

end profit_increase_l11_11896


namespace numberOfRealSolutions_l11_11340

theorem numberOfRealSolutions :
  ∀ (x : ℝ), (-4*x + 12)^2 + 1 = (x - 1)^2 → (∃ a b : ℝ, (a ≠ b) ∧ (-4*a + 12)^2 + 1 = (a - 1)^2 ∧ (-4*b + 12)^2 + 1 = (b - 1)^2) := by
  sorry

end numberOfRealSolutions_l11_11340


namespace probability_of_form_x2_minus_by2_l11_11265

theorem probability_of_form_x2_minus_by2 (x y : ℝ) : 
  let expressions := {x + y, x + 5 * y, x - y, 5 * x + y} in
  let pairs := { {e1, e2} | e1 ∈ expressions ∧ e2 ∈ expressions ∧ e1 ≠ e2 } in
  (∃ pair ∈ pairs, ∃ b : ℤ, (pair.fst * pair.snd) = x^2 - (b * y)^2) → 
  1 / 6 := 
sorry

end probability_of_form_x2_minus_by2_l11_11265


namespace even_three_digit_numbers_l11_11051

theorem even_three_digit_numbers (n : ℕ) :
  (n >= 100 ∧ n < 1000) ∧
  (n % 2 = 0) ∧
  ((n % 100) / 10 + (n % 10) = 12) →
  n = 12 :=
sorry

end even_three_digit_numbers_l11_11051


namespace a_n_minus_1_has_n_distinct_prime_divisors_l11_11242

-- Definition of the sequence a_n
def a : ℕ → ℕ
| 0     := 5
| (n+1) := (a n)^2

-- Theorem statement
theorem a_n_minus_1_has_n_distinct_prime_divisors (n : ℕ) (h : n ≥ 1) : (a n) - 1 ≥ n :=
by sorry

end a_n_minus_1_has_n_distinct_prime_divisors_l11_11242


namespace floor_sqrt_50_squared_l11_11802

theorem floor_sqrt_50_squared :
  (let x := Real.sqrt 50 in (⌊x⌋₊ : ℕ)^2 = 49) :=
by
  sorry

end floor_sqrt_50_squared_l11_11802


namespace range_of_a_for_two_roots_l11_11868

noncomputable def f (x a : ℝ) : ℝ := x^2 - a * Real.log x

theorem range_of_a_for_two_roots (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, (x₁ ≠ x₂) ∧ f x₁ a = 0 ∧ f x₂ a = 0) ↔ 2 * Real.exp(1) < a :=
sorry

end range_of_a_for_two_roots_l11_11868


namespace cistern_water_breadth_l11_11290

theorem cistern_water_breadth (length width total_area : ℝ) (h : ℝ) 
  (h_length : length = 10) 
  (h_width : width = 6) 
  (h_area : total_area = 103.2) : 
  (60 + 20*h + 12*h = total_area) → h = 1.35 :=
by
  intros
  sorry

end cistern_water_breadth_l11_11290


namespace trig_identity_l11_11683

theorem trig_identity (A B C : ℝ) (x y z : ℝ) (k : ℤ) 
  (h1 : A + B + C = k * π) 
  (h2 : x * Real.sin A + y * Real.sin B + z * Real.sin C = 0)
  (h3 : x^2 * Real.sin (2 * A) + y^2 * Real.sin (2 * B) + z^2 * Real.sin (2 * C) = 0) :
  ∀ n : ℕ+, x^n * Real.sin (n * A) + y^n * Real.sin (n * B) + z^n * Real.sin (n * C) = 0 :=
by
  intro n
  sorry

end trig_identity_l11_11683


namespace problem_solution_l11_11865

-- Definitions corresponding to the conditions
def proposition1 (a b : ℝ) (hab : a > 0) (hb : b < 0) (h : a > b) : Prop :=
  a - (1 / a) > b - (1 / b)

def proposition2 (a b : ℝ) (hb : b ≠ 0) (h : b * (b - a) ≤ 0) : Prop :=
  a / b ≥ 1

def proposition2_converse (a b : ℝ) (hb : b ≠ 0) (h : b / a ≥ 1) : Prop :=
  b * (b - a) ≤ 0

def proposition3 (a b c d : ℝ) (h : a + c > b + d) : Prop :=
  a > b ∧ c > d

-- The overall proof problem statement
theorem problem_solution :
  (∀ (a b : ℝ), a < 0 → b < 0 → a > b → proposition1 a b)
  ∧ (∀ (a b : ℝ), b ≠ 0 → b * (b - a) ≤ 0 → proposition2 a b)
  ∧ (∀ (a b : ℝ), b ≠ 0 → a / b ≥ 1 → proposition2_converse a b)
  ∧ ¬(∀ (a b c d : ℝ), a + c > b + d → proposition3 a b c d) :=
by
  repeat { sorry }

end problem_solution_l11_11865


namespace range_f_l11_11216

def f (x : ℤ) : ℤ := x + 1
def domain := {-1, 1, 2}

theorem range_f :
  set.image f domain = {0, 2, 3} :=
sorry

end range_f_l11_11216


namespace companion_vector_of_g_exists_point_P_on_h_l11_11393

open Real

def f_companion_vector (a b : ℝ) (x : ℝ) : ℝ := a * sin x + b * cos x

def g (x : ℝ) := 4 * cos (x / 2 + π / 3) * cos(x / 2) - 1

def h (x : ℝ) := 2 * cos (x / 2)

def vector_OM (a b : ℝ) : ℝ × ℝ := (a, b)

def A : ℝ × ℝ := (-2, 3)

def B : ℝ × ℝ := (2, 6)

def is_perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem companion_vector_of_g : vector_OM (-sqrt 3) 1 = (let a := -sqrt 3; let b := 1; (a, b)) :=
sorry

theorem exists_point_P_on_h : 
  ∃ P : ℝ × ℝ, 
    let x := P.1; let y := P.2 
    in y = h x ∧ is_perpendicular (x + 2, y - 3) (x - 2, y - 6) ∧ P = (0, 2) :=
sorry

end companion_vector_of_g_exists_point_P_on_h_l11_11393


namespace sin_A_calculation_height_calculation_l11_11506

variable {A B C : ℝ}

-- Given conditions
def angle_condition : Prop := A + B = 3 * C
def sine_condition : Prop := 2 * sin (A - C) = sin B

-- Part 1: Find sin A
theorem sin_A_calculation (h1 : angle_condition) (h2 : sine_condition) : sin A = 3 * real.sqrt 10 / 10 := sorry

-- Part 2: Given AB = 5, find the height
variable {AB : ℝ}
def AB_value : Prop := AB = 5

theorem height_calculation (h1 : angle_condition) (h2 : sine_condition) (h3 : AB_value) : height = 6 := sorry

end sin_A_calculation_height_calculation_l11_11506


namespace number_of_men_in_engineering_department_l11_11107

theorem number_of_men_in_engineering_department (T : ℝ) (h1 : 0.30 * T = 180) : 
  0.70 * T = 420 :=
by 
  -- The proof will be done here, but for now, we skip it.
  sorry

end number_of_men_in_engineering_department_l11_11107


namespace ab_le_neg_inv_2019_l11_11391

variable (u : Fin 2019 → ℝ)

-- Given conditions
def sum_u (u : Fin 2019 → ℝ) : Prop := ∑ i, u i = 0
def sum_squares_u (u : Fin 2019 → ℝ) : Prop := ∑ i, (u i)^2 = 1

-- Definitions of a and b
def a (u : Fin 2019 → ℝ) := Finset.min' (Finset.univ.image u) (by apply Finset.Nonempty.image; exact Finset.univ_nonempty)
def b (u : Fin 2019 → ℝ) := Finset.max' (Finset.univ.image u) (by apply Finset.Nonempty.image; exact Finset.univ_nonempty)

-- Statement to be proven
theorem ab_le_neg_inv_2019 (u : Fin 2019 → ℝ) (h_sum_u : sum_u u) (h_sum_squares_u : sum_squares_u u) :
  a u * b u ≤ - (1 / 2019) :=
sorry

end ab_le_neg_inv_2019_l11_11391


namespace volume_not_occupied_by_cones_l11_11238

/-- Two cones with given dimensions are enclosed in a cylinder, and we want to find the volume 
    in the cylinder not occupied by the cones. -/
theorem volume_not_occupied_by_cones : 
  let radius := 10
  let height_cylinder := 26
  let height_cone1 := 10
  let height_cone2 := 16
  let volume_cylinder := π * (radius ^ 2) * height_cylinder
  let volume_cone1 := (1 / 3) * π * (radius ^ 2) * height_cone1
  let volume_cone2 := (1 / 3) * π * (radius ^ 2) * height_cone2
  let total_volume_cones := volume_cone1 + volume_cone2
  volume_cylinder - total_volume_cones = (2600 / 3) * π :=
by
  let radius := 10
  let height_cylinder := 26
  let height_cone1 := 10
  let height_cone2 := 16
  let volume_cylinder := π * (radius ^ 2) * height_cylinder
  let volume_cone1 := (1 / 3) * π * (radius ^ 2) * height_cone1
  let volume_cone2 := (1 / 3) * π * (radius ^ 2) * height_cone2
  let total_volume_cones := volume_cone1 + volume_cone2
  sorry

end volume_not_occupied_by_cones_l11_11238


namespace angle_A_example_max_area_example_l11_11855

theorem angle_A_example (a b c : ℝ) (B C : ℝ) (h1 : a = 2) 
  (h2 : b + c = (4 / Real.sqrt 3) * (Real.sin B + Real.sin C)) 
  (h3 : ∀ α, α = Real.arcsin (a / 2 / ((4 / Real.sqrt 3) / 2)) → 0 < α ∧ α < π / 2) : 
  ∃ A, A = π / 3 :=
by
  use (π / 3)
  sorry

theorem max_area_example (a b c : ℝ) (B C : ℝ) (h1 : a = 2) 
  (h2 : b + c = (4 / Real.sqrt 3) * (Real.sin B + Real.sin C)) 
  (h4 : b * c ≤ 4) : 
  ∃ S, S = Real.sqrt 3 :=
by 
  use Real.sqrt 3
  sorry

end angle_A_example_max_area_example_l11_11855


namespace new_avg_weight_l11_11597

theorem new_avg_weight 
  (initial_avg_weight : ℝ)
  (initial_num_members : ℕ)
  (new_person1_weight : ℝ)
  (new_person2_weight : ℝ)
  (new_num_members : ℕ)
  (final_total_weight : ℝ)
  (final_avg_weight : ℝ) :
  initial_avg_weight = 48 →
  initial_num_members = 23 →
  new_person1_weight = 78 →
  new_person2_weight = 93 →
  new_num_members = initial_num_members + 2 →
  final_total_weight = (initial_avg_weight * initial_num_members) + new_person1_weight + new_person2_weight →
  final_avg_weight = final_total_weight / new_num_members →
  final_avg_weight = 51 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end new_avg_weight_l11_11597


namespace john_new_earnings_l11_11532

variable (original_weekly_earnings : ℝ)
variable (percentage_increase : ℝ)
variable (new_weekly_earnings : ℝ)

-- Definitions based on the conditions:
def raise_amount (original_weekly_earnings percentage_increase : ℝ) : ℝ := (percentage_increase / 100) * original_weekly_earnings
def expected_new_earnings (original_weekly_earnings raise_amount : ℝ) : ℝ := original_weekly_earnings + raise_amount

-- Given the conditions:
axiom original_earnings_condition : original_weekly_earnings = 60
axiom percentage_increase_condition : percentage_increase = 33.33

-- Prove the following statement:
theorem john_new_earnings (h1 : original_weekly_earnings = 60) (h2 : percentage_increase = 33.33) :
  new_weekly_earnings = 80 :=
by
  sorry

end john_new_earnings_l11_11532


namespace floor_sqrt_50_squared_l11_11783

theorem floor_sqrt_50_squared :
  ∃ x : ℕ, x = 7 ∧ ⌊ Real.sqrt 50 ⌋ = x ∧ x^2 = 49 := 
by {
  let x := 7,
  use x,
  have h₁ : 7 < Real.sqrt 50, from sorry,
  have h₂ : Real.sqrt 50 < 8, from sorry,
  have floor_eq : ⌊Real.sqrt 50⌋ = 7, from sorry,
  split,
  { refl },
  { split,
    { exact floor_eq },
    { exact rfl } }
}

end floor_sqrt_50_squared_l11_11783


namespace compute_a_plus_b_l11_11364

theorem compute_a_plus_b (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_terms : b - a = 999)
  (h_log_value : ∑ k in finset.range(b - a), log (k + a + 1) / log (k + a) = 3) : 
  a + b = 1010 := 
sorry

end compute_a_plus_b_l11_11364


namespace range_of_b_l11_11876

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := Real.exp x * (x*x - b*x)

theorem range_of_b (b : ℝ) : 
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, 0 < (Real.exp x * ((x*x + (2 - b) * x - b)))) →
  b < 8/3 := 
sorry

end range_of_b_l11_11876


namespace second_integer_is_ninety_point_five_l11_11224

theorem second_integer_is_ninety_point_five
  (n : ℝ)
  (first_integer fourth_integer : ℝ)
  (h1 : first_integer = n - 2)
  (h2 : fourth_integer = n + 1)
  (h_sum : first_integer + fourth_integer = 180) :
  n = 90.5 :=
by
  -- sorry to skip the proof
  sorry

end second_integer_is_ninety_point_five_l11_11224


namespace min_beacons_required_l11_11466

/-- Definition of a labyrinth for the context of this problem --/
structure Labyrinth :=
  (rooms : Type)
  (corridors : rooms → rooms → Prop)

def distance {L : Labyrinth} (r1 r2 : L.rooms) : ℕ := sorry

/-- Definition of a beacon in a specified room --/
def beacon_in_room {L : Labyrinth} (room : L.rooms) : Prop := sorry

/-- Given the map of the labyrinth with marked beacon locations --/
def robot_can_determine_location {L : Labyrinth} (b1 b2 b3 : L.rooms)
  (beacon1 : beacon_in_room b1)
  (beacon2 : beacon_in_room b2)
  (beacon3 : beacon_in_room b3) : Prop :=
∀ r1 r2 : L.rooms, (distance r1 b1, distance r1 b2, distance r1 b3) = (distance r2 b1, distance r2 b2, distance r2 b3) → r1 = r2

/-- Proof that fewer than 3 beacons are insufficient --/
def insufficient_beacons {L : Labyrinth} : Prop :=
∀ b1 b2 : L.rooms,
  (beacon_in_room b1) → (beacon_in_room b2) →
  ∃ r1 r2 : L.rooms, r1 ≠ r2 ∧ (distance r1 b1, distance r1 b2) = (distance r2 b1, distance r2 b2)

/-- Minimum number of beacons required --/
theorem min_beacons_required (L : Labyrinth) : ∃ (b1 b2 b3 : L.rooms),
  (beacon_in_room b1) ∧ (beacon_in_room b2) ∧ (beacon_in_room b3) ∧ 
  robot_can_determine_location b1 b2 b3 ∧ insufficient_beacons :=
sorry

end min_beacons_required_l11_11466


namespace max_value_is_sqrt5_l11_11397

noncomputable def max_value_of_f_add_f_prime 
  (ω : ℕ) 
  (h₀ : ω > 0) 
  (h₁ : ∀ x ∈ Ioo 0 (ω * π), (sin (ω * x + ω)) = 0 → (q x) ∈ {1, 2, 3, 4}) 
  : ℝ :=
√5

theorem max_value_is_sqrt5 
  {ω : ℕ}
  (h₀ : ω > 0)
  (h₁ : ∀ x ∈ Ioo 0 (ω * π), (sin (ω * x + ω)) = 0 → (q x) ∈ {1, 2, 3, 4})
  : 
  (∃ x ∈ Ioo 0 (ω * π), (f₁ : ℝ) (λ x, sin (ω * x + ω)) * (f₂ : ℝ) (λ x, 2 * cos (ω * x + ω)) = √5) 
:= sorry

end max_value_is_sqrt5_l11_11397


namespace sum_angles_of_mgon_sum_defects_of_convex_polyhedron_l11_11680

-- Define the setup conditions for the m-gon on the convex polyhedron
variables {Polyhedron : Type} 
variables (T : List Polyhedron) -- T is the m-gon on the polyhedron's surface
variables (m : ℕ) -- m is the number of sides of the m-gon
variables (DEF : Polyhedron → ℝ) -- DEF is a function that gives the defect at each vertex

-- Problem 1: Prove the sum of angles of T
theorem sum_angles_of_mgon (H : Convex Polyhedron) (T_vertices : Finset Polyhedron) :
  ∑ angle_in_T = 2 * π * (m - 2) + ∑ v in T_vertices, DEF v := 
sorry

-- Problem 2: Prove the sum of defects of all vertices of the convex polyhedron
theorem sum_defects_of_convex_polyhedron (H : Convex Polyhedron) (all_vertices : Finset Polyhedron) :
  ∑ v in all_vertices, DEF v = 4 * π :=
sorry

end sum_angles_of_mgon_sum_defects_of_convex_polyhedron_l11_11680


namespace floor_sqrt_50_squared_l11_11785

theorem floor_sqrt_50_squared :
  ∃ x : ℕ, x = 7 ∧ ⌊ Real.sqrt 50 ⌋ = x ∧ x^2 = 49 := 
by {
  let x := 7,
  use x,
  have h₁ : 7 < Real.sqrt 50, from sorry,
  have h₂ : Real.sqrt 50 < 8, from sorry,
  have floor_eq : ⌊Real.sqrt 50⌋ = 7, from sorry,
  split,
  { refl },
  { split,
    { exact floor_eq },
    { exact rfl } }
}

end floor_sqrt_50_squared_l11_11785


namespace part1_extreme_values_part2_min_value_in_interval_l11_11414

-- Part 1
theorem part1_extreme_values :
  let f (x : ℝ) := - (1/3) * x^3 - (1/2) * x^2 + 6 * x
  in (f 2 = 22 / 3) ∧ (f -3 = -27 / 2) :=
by
  sorry

-- Part 2
theorem part2_min_value_in_interval :
  let f (x : ℝ) := (1 / 3) * x^3 - (1 / 2) * x^2 + 2 * (-1) * x
  in (-2 < -1) → (-1 < 0) → (∀ x ∈ Icc 1 4, f x ≤ 16 / 3 → f 4 = 16 / 3)
  → f 2 = -10 / 3 :=
by
  sorry

end part1_extreme_values_part2_min_value_in_interval_l11_11414


namespace integral_equivalence_a_cubed_l11_11358

theorem integral_equivalence_a_cubed (a : ℝ) 
  (h : ∫ x in 0..(Real.pi / 4), Real.cos x = ∫ x in 0..a, x^2 ) 
  : a^3 = (3 * Real.sqrt 2) / 2 :=
sorry

end integral_equivalence_a_cubed_l11_11358


namespace kids_still_awake_l11_11281

theorem kids_still_awake (initial_count remaining_after_first remaining_after_second : ℕ) 
  (h_initial : initial_count = 20)
  (h_first_round : remaining_after_first = initial_count / 2)
  (h_second_round : remaining_after_second = remaining_after_first / 2) : 
  remaining_after_second = 5 := 
by
  sorry

end kids_still_awake_l11_11281


namespace at_least_two_greater_than_one_l11_11622

theorem at_least_two_greater_than_one
  (a b c : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_eq : a + b + c = a * b * c) : 
  1 < a ∨ 1 < b ∨ 1 < c :=
sorry

end at_least_two_greater_than_one_l11_11622


namespace find_down_payment_l11_11972

noncomputable def purchasePrice : ℝ := 118
noncomputable def monthlyPayment : ℝ := 10
noncomputable def numberOfMonths : ℝ := 12
noncomputable def interestRate : ℝ := 0.15254237288135593
noncomputable def totalPayments : ℝ := numberOfMonths * monthlyPayment -- total amount paid through installments
noncomputable def interestPaid : ℝ := purchasePrice * interestRate -- total interest paid
noncomputable def totalPaid : ℝ := purchasePrice + interestPaid -- total amount paid including interest

theorem find_down_payment : ∃ D : ℝ, D + totalPayments = totalPaid ∧ D = 16 :=
by sorry

end find_down_payment_l11_11972


namespace distinct_values_for_D_l11_11464

-- Define distinct digits
def distinct_digits (a b c d e : ℕ) :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e ∧ 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10

-- Declare the problem statement
theorem distinct_values_for_D : 
  ∃ D_values : Finset ℕ, 
    (∀ (A B C D E : ℕ), 
      distinct_digits A B C D E → 
      E + C = D ∧
      B + C = E ∧
      B + D = E) →
    D_values.card = 7 := 
by 
  sorry

end distinct_values_for_D_l11_11464


namespace price_increase_is_13_799_l11_11936

def percent_increase (P : ℝ) : ℝ :=
  let step1 := P * 1.05
  let step2 := step1 * 1.07
  let step3 := step2 * 0.97
  let finalPrice := step3 * 1.04
  ((finalPrice / P) - 1) * 100

theorem price_increase_is_13_799 (P : ℝ) : 
  percent_increase P ≈ 13.799 :=
by 
  sorry

end price_increase_is_13_799_l11_11936


namespace find_m_value_l11_11427

theorem find_m_value : ∀ (m : ℝ), 
  (let A := set.insert 3 (set.insert 1 (set.singleton (real.sqrt m)));
       B := set.insert 1 (set.singleton m)) in
  A ∪ B = A → (m = 0 ∨ m = 3) :=
sorry

end find_m_value_l11_11427


namespace pascal_triangle_probability_l11_11741

theorem pascal_triangle_probability :
  let total_elements := 20 * 21 / 2,
      ones := 1 + 19 * 2,
      twos := 18 * 2,
      elements := ones + twos in
  (total_elements = 210) →
  (ones = 39) →
  (twos = 36) →
  (elements = 75) →
  (75 / 210) = 5 / 14 :=
by
  intros,
  sorry

end pascal_triangle_probability_l11_11741


namespace hyperbola_asymptotes_l11_11468

-- Define the conditions of the hyperbola
variables (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b)
-- Define the hyperbola's eccentricity condition
def eccentricity_hyperbola (a b : ℝ) : Prop :=
  real.sqrt (1 + (b / a) ^ 2) = real.sqrt 10

-- Define the statement to prove the equations of asymptotes
theorem hyperbola_asymptotes (h₁ : eccentricity_hyperbola a b)
  (h₂ : ∃ (a b : ℝ), (0 < a) ∧ (0 < b)) : 
  ∃ k : ℝ, k = 3 ∧ ∀ x y : ℝ, 
  (y = k * x ∨ y = -k * x) :=
begin
  use 3,
  split,
  { refl },
  intros x y,
  split,
  { intro h,
    exact h },
  intro h,
  exact h
end

end hyperbola_asymptotes_l11_11468


namespace perimeter_of_DEF_l11_11854

variable {A B C D E F : Type}
variable {ABC DEF : Triangle}
variable {AB BC CA DE EF DF : ℝ}

noncomputable def perimeter_DEF (h₁ : ABC ~ DEF) 
                                (h₂ : area ABC = (3 / 2) * area DEF) 
                                (h₃ : AB = 2 ∧ BC = 2 ∧ CA = 2) 
                                : ℝ :=
2 * sqrt 6

theorem perimeter_of_DEF (h₁ : ABC ~ DEF)
                         (h₂ : area ABC = (3 / 2) * area DEF)
                         (h₃ : AB = 2 ∧ BC = 2 ∧ CA = 2) :
    perimeter DEF = 2 * sqrt 6 := 
sorry

end perimeter_of_DEF_l11_11854


namespace parabolic_arch_height_l11_11704

theorem parabolic_arch_height 
  (a : ℝ) 
  (h_eqn : ∀ x, (x = 15) → 0 = a * x^2 + 24) 
  (h : ∀ x, (x = 10) → (y : ℝ), y = a * x^2 + 24) : (y = 40 / 3) :=
by 
  sorry

end parabolic_arch_height_l11_11704


namespace count_valid_even_numbers_with_sum_12_l11_11021

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n % 2 = 0) ∧ 
  ((n / 10) % 10 + n % 10 = 12)

theorem count_valid_even_numbers_with_sum_12 :
  (finset.range 1000).filter is_valid_number).card = 27 := by
  sorry

end count_valid_even_numbers_with_sum_12_l11_11021


namespace hyperbola_eccentricity_l11_11073

theorem hyperbola_eccentricity (a : ℝ) (e : ℝ) :
  (∀ x y : ℝ, y = (1 / 8) * x^2 → x^2 = 8 * y) →
  (∀ y x : ℝ, y^2 / a - x^2 = 1 → a + 1 = 4) →
  e^2 = 4 / 3 →
  e = 2 * Real.sqrt 3 / 3 :=
by
  intros h1 h2 h3
  sorry

end hyperbola_eccentricity_l11_11073


namespace sin_beta_zero_l11_11837

theorem sin_beta_zero (α β : ℝ) (h₁ : 0 < α) (h₂ : α < β) (h₃ : β < π)
                      (h₄ : sin α = 3 / 5) (h₅ : sin (α + β) = 3 / 5) :
                      sin β = 0 :=
sorry

end sin_beta_zero_l11_11837


namespace part1_part2_part3_l11_11870

def f (a : ℝ) (x : ℝ) : ℝ := (a * x - x ^ 2) * Real.exp x

theorem part1 (a : ℝ) (hx : a = 2) :
  ∀ x, ¬((x ≤ -Real.sqrt 2 ∨ x ≥ Real.sqrt 2) → (Real.deriv (f a) x ≤ 0)) := 
sorry

theorem part2 (a : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x ≤ 1 → Real.deriv (f a) x ≥ 0) → (a ≥ 3 / 2) := 
sorry

theorem part3 (a : ℝ) : 
  ¬(∀ x : ℝ, (Real.deriv (f a) x ≤ 0) ∨ (Real.deriv (f a) x ≥ 0)) := 
sorry

end part1_part2_part3_l11_11870


namespace ellipse_equation_and_slopes_l11_11964

theorem ellipse_equation_and_slopes :
  ∃ (a b : ℝ) (k₁ k₂ : ℝ) (E : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1), 
  a > b ∧ b > 0 ∧ a = 2 * Real.sqrt 2 ∧ b = 2 ∧ 
  let e₁ := Real.sqrt (1 - b^2 / a^2) in
  let m := 1 in
  let n := 1 in
  let e₂ := Real.sqrt (1 + n / m) in
  e₁ * e₂ = 1 ∧
  k₁ + k₂ = 0 :=
by
  sorry

end ellipse_equation_and_slopes_l11_11964


namespace max_value_f_plus_f_prime_l11_11394

noncomputable def omega := 2
def f (x : ℝ) : ℝ := Real.sin (omega * x + omega)
def f_prime (x : ℝ) : ℝ := (deriv f) x

theorem max_value_f_plus_f_prime :
  ∃ x : ℝ, (0 < x) ∧ (x < omega * Real.pi) ∧ (f x + f_prime x = Real.sqrt 5) := sorry

end max_value_f_plus_f_prime_l11_11394


namespace number_of_elements_in_M_l11_11832

-- Define our custom operation "※"
def op: ℕ → ℕ → ℕ
| m, n := 
  match m % 2, n % 2 with
  | 0, 0 => m + n    -- both m and n are even
  | 1, 1 => m + n    -- both m and n are odd
  | _, _ => m * n    -- one is even and the other is odd

-- Define the set M and the proof statement
def M : Set (ℕ × ℕ) := {p | op p.1 p.2 = 12 ∧ p.1 > 0 ∧ p.2 > 0}

-- Prove that the size of M is 15
theorem number_of_elements_in_M : Fintype.card (M) = 15 :=
by
  sorry

end number_of_elements_in_M_l11_11832


namespace length_FD_is_four_l11_11966

-- First, define the necessary structures and conditions
variables {Point : Type} [MetricSpace Point]

-- Define points and their relationships
variables (A B C O D F : Point)
def distance (p q : Point) := dist p q

-- Given
def conditions 
  (A B C O D F : Point)
  (radius : ℝ) (h_radius : radius = 20)
  (h_AB : distance A B = 20)
  (h_BC : distance B C = 20)
  (h_BO : distance B O = 10)
  (h_tangent_CD : dist O D = 10)
  (h_CD_intersect : distance C D = distance B F + distance F D) :
  Prop := 
  ∃ (AB_diameter : A ≠ B ∧ distance A B = 2 * distance O A) 
    (right_triangle_ABC : ∀ (C : Point), ∠ A B C = π/2),
  true -- Add all the conditions and relationships as required
  
-- The theorem to prove |FD| = 4
theorem length_FD_is_four 
  (A B C O D F : Point)
  (radius : ℝ) (h_radius : radius = 20)
  (h_AB : distance A B = 20)
  (h_BC : distance B C = 20)
  (h_BO : distance B O = 10)
  (h_tangent_CD : dist O D = 10)
  (h_CD_intersect : distance C D = distance B F + distance F D) 
  (h : conditions A B C O D F radius h_radius h_AB h_BC h_BO h_tangent_CD h_CD_intersect) :
  distance F D = 4 :=
sorry

end length_FD_is_four_l11_11966


namespace gcd_correct_l11_11648

def gcd_87654321_12345678 : ℕ :=
  gcd 87654321 12345678

theorem gcd_correct : gcd_87654321_12345678 = 75 := by 
  sorry

end gcd_correct_l11_11648


namespace sequence_general_term_sum_inequality_l11_11426

theorem sequence_general_term (a : ℕ → ℤ) (a1 : a 1 = 1) (a2 : a 2 = 5)
  (recur : ∀ n, a (n+2) = 4 * a (n+1) - 3 * a n) :
  ∀ n, a n = 2 * 3^(n-1) - 1 :=
by sorry

theorem sum_inequality (a : ℕ → ℤ) (b : ℕ → ℤ) (T : ℕ → ℤ)
  (a1 : a 1 = 1) (a2 : a 2 = 5)
  (recur : ∀ n, a (n+2) = 4 * a (n+1) - 3 * a n)
  (b_def : ∀ n, b n = 3^n / (a n * a (n+1)))
  (T_def : ∀ n, T n = ∑ i in range n, b i) :
  ∀ n, T n < 3 / 4 :=
by sorry

end sequence_general_term_sum_inequality_l11_11426


namespace range_of_a_l11_11415

noncomputable def f (x a : ℝ) : ℝ := |x - a| + x + 5

theorem range_of_a (x a : ℝ) (h : f x a ≥ 8) : |a + 5| ≥ 3 :=
begin
  sorry
end

end range_of_a_l11_11415


namespace train_speed_is_45_km_per_hr_l11_11712

-- Definitions of conditions
def length_of_train : ℝ := 140
def time_to_cross_bridge : ℝ := 30
def length_of_bridge : ℝ := 235
def total_distance : ℝ := length_of_train + length_of_bridge
def speed_m_per_s : ℝ := total_distance / time_to_cross_bridge
def speed_km_per_hr : ℝ := speed_m_per_s * 3.6

-- Theorem statement
theorem train_speed_is_45_km_per_hr : speed_km_per_hr = 45 := by
  sorry

end train_speed_is_45_km_per_hr_l11_11712


namespace original_volume_of_cube_l11_11973

theorem original_volume_of_cube (a : ℕ) 
  (h1 : (a + 2) * (a - 2) * (a + 3) = a^3 - 7) : 
  a = 3 :=
by sorry

end original_volume_of_cube_l11_11973


namespace correct_propositions_l11_11604

-- Define the conditions for each proposition
def prop1_area := (∫ x in (0:ℝ)..2, (2 * x - x^2)) = (4 / 3)
def prop2_locus (O : Point) (A B : Point) := 
  (1 / 2 : ℝ) • (O.to_vector + A.to_vector) + (1 / 2 : ℝ) • (O.to_vector + B.to_vector) = P.to_vector)
def prop3_distribution := false -- The given distribution formula is incorrect
def prop4_geometry_constraints := false -- Geometry constraints do not imply β is ⟂ α

-- The proof problem
theorem correct_propositions : prop1_area ∧ prop2_locus ∧ ¬prop3_distribution ∧ ¬prop4_geometry_constraints :=
by sorry

end correct_propositions_l11_11604


namespace sqrt_floor_squared_l11_11816

/-- To evaluate the floor of the square root of 50 squared --/
theorem sqrt_floor_squared :
  (⌊real.sqrt 50⌋ : ℕ)^2 = 49 :=
begin
  -- We know that:
  -- 7^2 = 49 < 50 < 64 = 8^2
  have h1 : 7^2 = 49, by linarith,
  have h2 : 64 = 8^2, by linarith,
  have h3 : (7 : real) < real.sqrt 50, by {
    rw [sqrt_lt],
    exact_mod_cast h1,
  },
  have h4 : real.sqrt 50 < 8, by {
    rw [lt_sqrt],
    exact_mod_cast h2,
  },
  -- Therefore, 7 < sqrt(50) < 8.
  have h5 : (⌊real.sqrt 50⌋ : ℕ) = 7, by {
    rw [nat.floor_eq_iff],
    split,
    { exact_mod_cast h3, },
    { exact_mod_cast h4, },
  },
  -- Thus, ⌊sqrt(50)⌋^2 = 7^2 = 49.
  rw h5,
  exact h1,
end

end sqrt_floor_squared_l11_11816


namespace f_constant_91_l11_11380

def f : ℤ → ℝ
| n => if n > 100 then n - 10 else f (f (n + 11))

theorem f_constant_91 (n : ℤ) (h : n ≤ 100) : f n = 91 := 
sorry

end f_constant_91_l11_11380


namespace power_of_product_zeros_l11_11254

theorem power_of_product_zeros:
  ∀ (n : ℕ), (500 : ℕ) = 5 * 10^2 → 
  500^n = 5^n * (10^2)^n → 
  10^(2 * n) = 10^(2 * 150) → 
  500^150 = 1 * 10^300 → 
  (500^150).to_digits.count 0 = 300 := 
by 
  intros n h1 h2 h3 h4 
  sorry

end power_of_product_zeros_l11_11254


namespace count_even_three_digit_numbers_l11_11036

theorem count_even_three_digit_numbers : 
  let num_even_three_digit_numbers : ℕ := 
    have h1 : (units_digit_possible_pairs : list (ℕ × ℕ)) := 
      [(4, 8), (6, 6), (8, 4)]
    have h2 : (number_of_hundreds_digits : ℕ) := 9
    3 * number_of_hundreds_digits 
in
  num_even_three_digit_numbers = 27 := by
  -- steps skipped
  sorry

end count_even_three_digit_numbers_l11_11036


namespace number_of_students_drawn_from_B_l11_11090

/-
This statement sets up the problem by defining the relevant variables and conditions,
then asserts the result using a theorem with a given proof (which is omitted here).
-/

variables (total_students : ℕ) 
          (sample_size : ℕ)
          (num_A num_B num_C : ℕ) 
          (sample_A sample_B sample_C : ℕ)

axiom total_students_hypothesis : total_students = 1500
axiom arithmetic_sequence : num_A + num_B + num_C = total_students
axiom sample_size_hypothesis : sample_size = 120
axiom sample_students : sample_A + sample_B + sample_C = sample_size
axiom arithmetic_sequence_sample : sample_A + 2 * sample_B + sample_C = 3 * sample_B

theorem number_of_students_drawn_from_B : sample_B = 40 :=
by sorry

end number_of_students_drawn_from_B_l11_11090


namespace f_eq_91_for_n_le_100_l11_11382

noncomputable def f : ℤ → ℝ 
| n => if n > 100 then n - 10 else f(f(n + 11))

theorem f_eq_91_for_n_le_100 (n : ℤ) (h : n ≤ 100) : f n = 91 := 
sorry

end f_eq_91_for_n_le_100_l11_11382


namespace pascal_element_probability_l11_11738

open Nat

def num_elems_first_n_rows (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

def count_ones (n : ℕ) : ℕ :=
  if n = 0 then 1 else if n = 1 then 2 else 2 * (n - 1) + 1

def count_twos (n : ℕ) : ℕ :=
  if n < 2 then 0 else 2 * (n - 2)

def probability_one_or_two (n : ℕ) : ℚ :=
  let total_elems := num_elems_first_n_rows n in
  let ones := count_ones n in
  let twos := count_twos n in
  (ones + twos) / total_elems

theorem pascal_element_probability :
  probability_one_or_two 20 = 5 / 14 :=
by
  sorry

end pascal_element_probability_l11_11738


namespace probability_of_palindrome_l11_11967

-- Definitions for letters and digits
def letters := List (Fin 26) -- Assume a list of 26 letters
def digits := List (Fin 10) -- Assume a list of 10 digits

-- Definitions for palindrome conditions
def is_palindrome {α : Type} (l : List α) : Prop :=
  l = l.reverse

/-- 
  Define a license plate as a sequence of four letters followed by four digits
  and assert the presence of at least one 'X' in the letters.
-/
def license_plate := (List (Fin 26) × List (Fin 10))

-- Proving the probability that a license plate contains at least one palindrome
theorem probability_of_palindrome : 
  ∃ (
    P_digit_palindrome : ℚ := 1 / 100, 
    P_letter_palindrome : ℚ := 1 / 8784, 
    P_both_palindromes : ℚ := 1 / 878400
  ),
  (P_digit_palindrome + P_letter_palindrome - P_both_palindromes = 8883 / 878400) := 
sorry  -- Proof Placeholder

end probability_of_palindrome_l11_11967


namespace proof_slope_l11_11435

-- Given definitions of complex number operations
def complex_add (z1 z2: ℂ) : ℂ := z1 + z2
def complex_conj (z : ℂ) : ℂ := conj z

-- Condition definitions
def condition1 : Prop := complex_add ((1:ℂ) + (1:ℂ) * I)^2 (complex_conj (0 + 2 * I)) = complex_conj (ℂ.mk 2 2)

def condition2 (z : ℂ) : Prop := ∃ (a b : ℝ), z = (a : ℂ) + (b : ℂ) * I

def condition3 : Prop := complex_conj I * complex_conj I = -1

-- The question asking to prove the slope of the line bx - ay + a = 0 is -1
def slope_is_neg1 (a b : ℝ) : Prop := b ≠ 0 → -(-a / b) = -1

-- The theorem to be proven
theorem proof_slope (a b : ℝ) (z : ℂ) (h1 : condition1)
(h2 : condition2 z) (h3 : condition3) : slope_is_neg1 a b :=
sorry

end proof_slope_l11_11435


namespace curve_C_polar_and_length_PQ_l11_11780

noncomputable def polar_equation (rho theta : ℝ) : Prop :=
  rho^2 - 2 * rho * Real.cos theta - 2 = 0

noncomputable def line_l1 (rho theta : ℝ) : Prop :=
  2 * rho * Real.sin (theta + Real.pi / 3) + 3 * Real.sqrt 3 = 0

noncomputable def line_l2 (theta : ℝ) : Prop :=
  theta = Real.pi / 3

theorem curve_C_polar_and_length_PQ :
  (∀ (theta : ℝ), 
     0 ≤ theta ∧ theta ≤ Real.pi →
     ∃ (rho : ℝ), polar_equation rho theta) ∧
  let P := (2, Real.pi / 3) in
  let Q := (-3, Real.pi / 3) in
  Real.abs (P.1 - Q.1) = 5 :=
by
  sorry

end curve_C_polar_and_length_PQ_l11_11780


namespace mixed_bead_cost_per_box_l11_11579

-- Definitions based on given conditions
def red_bead_cost : ℝ := 1.30
def yellow_bead_cost : ℝ := 2.00
def total_boxes : ℕ := 10
def red_boxes_used : ℕ := 4
def yellow_boxes_used : ℕ := 4

-- Theorem statement
theorem mixed_bead_cost_per_box :
  ((red_boxes_used * red_bead_cost) + (yellow_boxes_used * yellow_bead_cost)) / total_boxes = 1.32 :=
  by sorry

end mixed_bead_cost_per_box_l11_11579


namespace percentage_of_boys_from_school_A_that_study_science_l11_11089

def boys_camp_data := 
  let total_boys : ℕ := 550
  let school_A_fraction : ℝ := 0.20
  let non_science_boys_from_A : ℕ := 77
  (total_boys, school_A_fraction, non_science_boys_from_A)

theorem percentage_of_boys_from_school_A_that_study_science (total_boys: ℕ) (school_A_fraction: ℝ) (non_science_boys_from_A: ℕ) :
  total_boys = 550 ∧ school_A_fraction = 0.20 ∧ non_science_boys_from_A = 77 → 
  let total_boys_from_A := school_A_fraction * total_boys in
  let boys_from_A_study_science := total_boys_from_A - non_science_boys_from_A in
  let percentage_study_science := (boys_from_A_study_science / total_boys_from_A) * 100 in
  percentage_study_science = 30 := by
  sorry

end percentage_of_boys_from_school_A_that_study_science_l11_11089


namespace cylinder_ball_volume_ratio_l11_11272

theorem cylinder_ball_volume_ratio (R : ℝ) : 
  let V_cylinder := 2 * Real.pi * R^3 in
  let V_ball := (4 / 3) * Real.pi * R^3 in
  V_cylinder / V_ball = 3 / 2 :=
by
  sorry

end cylinder_ball_volume_ratio_l11_11272


namespace count_valid_even_numbers_with_sum_12_l11_11023

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n % 2 = 0) ∧ 
  ((n / 10) % 10 + n % 10 = 12)

theorem count_valid_even_numbers_with_sum_12 :
  (finset.range 1000).filter is_valid_number).card = 27 := by
  sorry

end count_valid_even_numbers_with_sum_12_l11_11023


namespace count_even_three_digit_numbers_with_sum_12_l11_11019

noncomputable def even_three_digit_numbers_with_sum_12 : Prop :=
  let valid_pairs := [(8, 4), (6, 6), (4, 8)] in
  let valid_hundreds := 9 in
  let count_pairs := valid_pairs.length in
  let total_numbers := valid_hundreds * count_pairs in
  total_numbers = 27

theorem count_even_three_digit_numbers_with_sum_12 : even_three_digit_numbers_with_sum_12 :=
by
  sorry

end count_even_three_digit_numbers_with_sum_12_l11_11019


namespace interval_after_speed_limit_l11_11638

noncomputable def car_speed_before : ℝ := 80 -- speed before the sign in km/h
noncomputable def car_speed_after : ℝ := 60 -- speed after the sign in km/h
noncomputable def initial_interval : ℕ := 10 -- interval between the cars in meters

-- Convert speeds from km/h to m/s
noncomputable def v : ℝ := car_speed_before * 1000 / 3600
noncomputable def u : ℝ := car_speed_after * 1000 / 3600

-- Given the initial interval and speed before the sign, calculate the time it takes for the second car to reach the sign
noncomputable def delta_t : ℝ := initial_interval / v

-- Given u and delta_t, calculate the new interval after slowing down
noncomputable def new_interval : ℝ := u * delta_t

-- Theorem statement in Lean
theorem interval_after_speed_limit : new_interval = 7.5 :=
sorry

end interval_after_speed_limit_l11_11638


namespace amare_additional_fabric_needed_l11_11158

-- Defining the conditions
def yards_per_dress : ℝ := 5.5
def num_dresses : ℝ := 4
def initial_fabric_feet : ℝ := 7
def yard_to_feet : ℝ := 3

-- The theorem to prove
theorem amare_additional_fabric_needed : 
  (yards_per_dress * num_dresses * yard_to_feet) - initial_fabric_feet = 59 := 
by
  sorry

end amare_additional_fabric_needed_l11_11158


namespace divides_square_sum_implies_divides_l11_11068

theorem divides_square_sum_implies_divides (a b : ℤ) (h : 7 ∣ a^2 + b^2) : 7 ∣ a ∧ 7 ∣ b := 
sorry

end divides_square_sum_implies_divides_l11_11068


namespace power_of_product_zeros_l11_11252

theorem power_of_product_zeros:
  ∀ (n : ℕ), (500 : ℕ) = 5 * 10^2 → 
  500^n = 5^n * (10^2)^n → 
  10^(2 * n) = 10^(2 * 150) → 
  500^150 = 1 * 10^300 → 
  (500^150).to_digits.count 0 = 300 := 
by 
  intros n h1 h2 h3 h4 
  sorry

end power_of_product_zeros_l11_11252


namespace conditional_probabilities_l11_11581

def outcomes_all_different : ℕ := 6 * 5 * 4

def outcomes_at_least_one_three : ℕ := 6^3 - 5^3

def outcomes_different_one_three : ℕ := 3 * 5 * 4

noncomputable def P_A_given_B : ℚ :=
  outcomes_different_one_three / outcomes_at_least_one_three

noncomputable def P_B_given_A : ℚ := 1 / 2

theorem conditional_probabilities :
  P_A_given_B = 60 / 91 ∧ P_B_given_A = 1 / 2 :=
by
  sorry

end conditional_probabilities_l11_11581


namespace simplest_sqrt_l11_11664

-- Define the options
def optionA := Real.sqrt 7
def optionB := Real.sqrt (1 / 2)
def optionC := Real.sqrt 0.2
def optionD := Real.sqrt 12

-- Define the "simplest" concept for this problem
-- Here, we can consider a basic function indicating 'not simplified'
def is_not_simplified (x : ℝ) : Prop :=
  (x = Real.sqrt(1 / 2) ∧ x ≠ Real.sqrt 2 / 2) ∨
  (x = Real.sqrt 0.2 ∧ x ≠ Real.sqrt 5 / 5) ∨
  (x = Real.sqrt 12 ∧ x ≠ 2 * Real.sqrt 3)

-- Define the proof statement
theorem simplest_sqrt :
  optionA = Real.sqrt 7 ∧
  (is_not_simplified optionB) ∧
  (is_not_simplified optionC) ∧
  (is_not_simplified optionD) :=
by
  sorry

end simplest_sqrt_l11_11664


namespace find_f_neg2007_value_l11_11679

def f : ℝ → ℝ := sorry

axiom cond1 (x y w : ℝ) (h1 : x > y) (h2 : f(x) + x ≥ w) (h3 : w ≥ f(y) + y) : ∃ z ∈ Icc y x, f(z) = w - z
axiom cond2 : ∃ u : ℝ, 
  (f u = 0) ∧ 
  (∀ v : ℝ, f(v) = 0 → u ≤ v)
axiom cond3 : f(0) = 1
axiom cond4 : f(-2007) ≤ 2008
axiom cond5 (x y : ℝ) : f(x) * f(y) = f(x * f(y) + y * f(x) + x * y)

theorem find_f_neg2007_value : f(-2007) = 2008 :=
by 
  sorry

end find_f_neg2007_value_l11_11679


namespace calculate_fraction_l11_11251

theorem calculate_fraction : (2002 - 1999)^2 / 169 = 9 / 169 :=
by
  sorry

end calculate_fraction_l11_11251


namespace max_perimeter_incenter_l11_11984

-- Define point P and the distances PA, PB, and PC
variables (P A B C : Point)
variable (PA PB PC : ℝ)

-- Conditions
axiom PA_eq_3 : PA = 3
axiom PB_eq_5 : PB = 5
axiom PC_eq_7 : PC = 7
axiom P_incenter : Incenter(P, A, B, C)

-- Theorem statement
theorem max_perimeter_incenter :
  ∀ (A B C : Point), (dist P A = 3) → (dist P B = 5) → (dist P C = 7) → (incenter P A B C) → 
  ∀ (A' B' C' : Point), (dist P A' = 3) → (dist P B' = 5) → (dist P C' = 7) → (perimeter A B C ≥ perimeter A' B' C') :=
sorry

end max_perimeter_incenter_l11_11984


namespace number_of_solutions_l11_11064

theorem number_of_solutions (h : ∀ n : ℕ, (n + 8) * (n - 3) * (n - 12) < 0 → 3 < n ∧ n < 12) : ∃ k : ℕ, k = 8 :=
by
  have : ∃ n : ℕ, 3 < n ∧ n < 12 := sorry
  let sol_set := {n : ℕ | 3 < n ∧ n < 12}
  let count := sol_set.to_finset.card
  have : count = 8 := sorry
  use count
  assumption
  sorry

end number_of_solutions_l11_11064


namespace sugar_total_l11_11287

variable (sugar_for_frosting sugar_for_cake : ℝ)

theorem sugar_total (h1 : sugar_for_frosting = 0.6) (h2 : sugar_for_cake = 0.2) :
  sugar_for_frosting + sugar_for_cake = 0.8 :=
by
  sorry

end sugar_total_l11_11287


namespace complex_number_purely_imaginary_l11_11071

theorem complex_number_purely_imaginary (m : ℝ) 
  (h1 : m^2 - 5 * m + 6 = 0) 
  (h2 : m^2 - 3 * m ≠ 0) : 
  m = 2 :=
sorry

end complex_number_purely_imaginary_l11_11071


namespace pascal_triangle_probability_l11_11742

theorem pascal_triangle_probability :
  let total_elements := 20 * 21 / 2,
      ones := 1 + 19 * 2,
      twos := 18 * 2,
      elements := ones + twos in
  (total_elements = 210) →
  (ones = 39) →
  (twos = 36) →
  (elements = 75) →
  (75 / 210) = 5 / 14 :=
by
  intros,
  sorry

end pascal_triangle_probability_l11_11742


namespace part_a_part_b_l11_11616

def d (x : List ℝ) (t : ℝ) : ℝ := 
  (List.minimum (x.map (λ xi => |xi - t|)) + List.maximum (x.map (λ xi => |xi - t|))) / 2

def c (x : List ℝ) : ℝ := 
  (x.minimum + x.maximum) / 2

def m (x : List ℝ) : ℝ := 
  let sorted := List.quicksort (≤) x
  let n := sorted.length
  if n % 2 = 0 then 
    (sorted.get (n / 2 - 1) + sorted.get (n / 2)) / 2 
  else 
    sorted.get (n / 2)

theorem part_a (x : List ℝ) : 
  ¬ ∀ t, (d x t) = (x.map (λ t' => d x t')).minimum := 
sorry

theorem part_b (x : List ℝ) : 
  d x (c x) ≤ d x (m x) := 
sorry

end part_a_part_b_l11_11616


namespace num_odd_functions_l11_11719

def f1 (x : ℝ) : ℝ := x^3
def f2 (x : ℝ) : ℝ := 2^x
def f3 (x : ℝ) : ℝ := x^2 + 1
def f4 (x : ℝ) : ℝ := 2 * Real.sin x

theorem num_odd_functions :
  (∃ f : ℝ → ℝ, f = f1 ∧ (∀ x, f (-x) = -f x)) ∧
  (¬(∃ f : ℝ → ℝ, f = f2 ∧ (∀ x, f (-x) = -f x))) ∧
  (¬(∃ f : ℝ → ℝ, f = f3 ∧ (∀ x, f (-x) = -f x))) ∧
  (∃ f : ℝ → ℝ, f = f4 ∧ (∀ x, f (-x) = -f x)) →
  ∃ n : ℕ, n = 2 :=
by
  sorry

end num_odd_functions_l11_11719


namespace value_of_f_ff_f4_l11_11836

def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 3 then x ^ 2 - 5
  else log 2 x

theorem value_of_f_ff_f4 : f (f (f 4)) = 1 := 
by
  sorry

end value_of_f_ff_f4_l11_11836


namespace exists_infinitely_many_double_numbers_with_properties_l11_11297

def is_double_number (a : ℕ) : Prop :=
  let digits := a.digits 10
  a > 0 ∧
  (digits.length % 2 = 0) ∧
  (let (first_half, second_half) := List.splitAt (digits.length / 2) digits
  in first_half = second_half) ∧
  (List.head digits ≠ 0)

def is_square (n : ℕ) : Prop :=
  ∃ t : ℕ, t * t = n

def is_not_power_of_10 (n : ℕ) : Prop :=
  ¬ ∃ k : ℕ, 10 ^ k = n

theorem exists_infinitely_many_double_numbers_with_properties :
  ∃ᶠ a in at_top, is_double_number a ∧ is_square (a + 1) ∧ is_not_power_of_10 (a + 1) :=
sorry

end exists_infinitely_many_double_numbers_with_properties_l11_11297


namespace longest_segment_CD_l11_11118

theorem longest_segment_CD
  (ABD_angle : ℝ) (ADB_angle : ℝ) (BDC_angle : ℝ) (CBD_angle : ℝ)
  (angle_proof_ABD : ABD_angle = 50)
  (angle_proof_ADB : ADB_angle = 40)
  (angle_proof_BDC : BDC_angle = 35)
  (angle_proof_CBD : CBD_angle = 70) :
  true := 
by
  sorry

end longest_segment_CD_l11_11118


namespace common_tangent_at_point_l11_11074

theorem common_tangent_at_point (x₀ b : ℝ) 
  (h₁ : 6 * x₀^2 = 6 * x₀) 
  (h₂ : 1 + 2 * x₀^3 = 3 * x₀^2 - b) :
  b = 0 ∨ b = -1 :=
sorry

end common_tangent_at_point_l11_11074


namespace probability_of_one_or_two_in_pascal_l11_11730

def pascal_triangle_element_probability : ℚ :=
  let total_elements := 210 -- sum of the elements in the first 20 rows
  let ones_count := 39      -- total count of 1s in the first 20 rows
  let twos_count := 36      -- total count of 2s in the first 20 rows
  let favorable_elements := ones_count + twos_count
  favorable_elements / total_elements

theorem probability_of_one_or_two_in_pascal (n : ℕ) (h : n = 20) :
  pascal_triangle_element_probability = 5 / 14 := by
  rw [h]
  dsimp [pascal_triangle_element_probability]
  sorry

end probability_of_one_or_two_in_pascal_l11_11730


namespace A_sequence_decreasing_H_sequence_increasing_l11_11552

-- The sequences are defined as follows for given distinct positive real numbers x, y, z.

variable {x y z : ℝ} (hxyz : x ≠ y ∧ y ≠ z ∧ x ≠ z)
variable (hpos : 0 < x ∧ 0 < y ∧ 0 < z)

noncomputable def A_1 := (x + y + z) / 3
noncomputable def G_1 := (x * y * z)^(1/3)
noncomputable def H_1 := 3 / (1/x + 1/y + 1/z)
noncomputable def Q_1 := (x^2 + y^2 + z^2 / 3)^(1/2)

noncomputable def A : ℕ → ℝ
| 1     := A_1
| (n+1) := (A n + Q n) / 2

noncomputable def G : ℕ → ℝ
| 1     := G_1
| (n+1) := (G n * H n)^(1/2)

noncomputable def H : ℕ → ℝ
| 1     := H_1
| (n+1) := 2 / (1 / (H n) + 1 / (Q n))

noncomputable def Q : ℕ → ℝ
| 1     := Q_1
| (n+1) := ((A n)^2 + (G n)^2 + (H n)^2 / 3)^(1/2)

theorem A_sequence_decreasing : ∀ n ≥ 1, A (n+1) < A n :=
by
  sorry

theorem H_sequence_increasing : ∀ n ≥ 1, H (n+1) > H n :=
by
  sorry

end A_sequence_decreasing_H_sequence_increasing_l11_11552


namespace length_PQ_squared_is_correct_l11_11980

noncomputable def square_length_PQ : ℝ :=
  let P := (4/3, 47/3) in
  let Q := (-4/3, -47/3) in
  (P.1 + Q.1)^2 + (P.2 + Q.2)^2

theorem length_PQ_squared_is_correct :
  square_length_PQ = 8900 / 9 :=
sorry

end length_PQ_squared_is_correct_l11_11980


namespace geometric_locus_is_circle_l11_11325

open Real

def sphere_radius (R : ℝ) : Prop :=
  R > 0

def geographic_latitude_equals_longitude (φ : ℝ) (P : ℝ × ℝ × ℝ) (R : ℝ) : Prop :=
  P = (R * cos φ * cos φ, R * cos φ * sin φ, R * sin φ)

def projection_onto_equatorial_plane (P : ℝ × ℝ × ℝ) : ℝ × ℝ :=
  (P.1, P.2)

theorem geometric_locus_is_circle (R : ℝ) (P : ℝ × ℝ × ℝ) (φ : ℝ) :
  sphere_radius R →
  geographic_latitude_equals_longitude φ P R →
  let proj := projection_onto_equatorial_plane P
  in (proj.1 - R / 2) ^ 2 + proj.2 ^ 2 = (R / 2) ^ 2 :=
by
  intros hR hP
  sorry

end geometric_locus_is_circle_l11_11325


namespace sin_A_and_height_on_AB_l11_11511

theorem sin_A_and_height_on_AB 
  (A B C: ℝ)
  (h_triangle: ∀ A B C, A + B + C = π)
  (h_angle_sum: A + B = 3 * C)
  (h_sin_condition: 2 * Real.sin (A - C) = Real.sin B)
  (h_AB: AB = 5)
  (h_sqrt_two: Real.cos (π / 4) = Real.sin (π / 4) := by norm_num) :
  (Real.sin A = 3 * Real.sqrt 10 / 10) ∧ (height_on_AB = 6) :=
sorry

end sin_A_and_height_on_AB_l11_11511


namespace tamika_carlos_probability_l11_11185

theorem tamika_carlos_probability : 
  let tamika_set := {11, 12, 13}
  let carlos_set := {4, 6, 7}
  let tamika_sums := {x + y | x y : ℕ // x ∈ tamika_set ∧ y ∈ tamika_set ∧ x ≠ y}
  let carlos_products := {x * y | x y : ℕ // x ∈ carlos_set ∧ y ∈ carlos_set ∧ x ≠ y}
  let outcomes := (finset.product tamika_sums carlos_products).filter (λ p, p.1 > p.2)
  outcomes.card = 1 / 9 : ℚ :=
  sorry

end tamika_carlos_probability_l11_11185


namespace area_of_cosine_curve_eq_two_l11_11770

noncomputable def area_under_cosine_curve : ℝ :=
  ∫ x in -π/2..π/2, Real.cos x

theorem area_of_cosine_curve_eq_two : area_under_cosine_curve = 2 := by
  sorry

end area_of_cosine_curve_eq_two_l11_11770


namespace additional_savings_l11_11709

def final_price (initial_price : ℝ) (discounts : list ℝ) : ℝ :=
  discounts.foldl (λ price discount, price * (1 - discount)) initial_price

theorem additional_savings (order_amount : ℝ) (d1 d2 : list ℝ) :
  (final_price order_amount d1 - final_price order_amount d2) = 1379.50 :=
by
  let price1 : ℝ := final_price order_amount d1
  let price2 : ℝ := final_price order_amount d2
  have h : price1 - price2 = 1379.50 := sorry
  exact h

end additional_savings_l11_11709


namespace floor_sqrt_50_squared_l11_11784

theorem floor_sqrt_50_squared :
  ∃ x : ℕ, x = 7 ∧ ⌊ Real.sqrt 50 ⌋ = x ∧ x^2 = 49 := 
by {
  let x := 7,
  use x,
  have h₁ : 7 < Real.sqrt 50, from sorry,
  have h₂ : Real.sqrt 50 < 8, from sorry,
  have floor_eq : ⌊Real.sqrt 50⌋ = 7, from sorry,
  split,
  { refl },
  { split,
    { exact floor_eq },
    { exact rfl } }
}

end floor_sqrt_50_squared_l11_11784


namespace at_least_7_counterclockwise_triangles_l11_11298

noncomputable def regular_hexagon := sorry -- Define the structure of a regular hexagon
def triangles_in_hexagon : list (list ℝ) := sorry -- List of triangles with vertex numbers
def is_counterclockwise (t : list ℝ) : Prop := sorry -- Predicate to check counterclockwise order

theorem at_least_7_counterclockwise_triangles :
  ∃ (triangles : list (list ℝ)), 
  (count is_counterclockwise triangles) ≥ 7 :=
begin
  -- conditions: Regular hexagon with 24 triangles and 19 distinct real numbers at vertices.
  sorry
end

end at_least_7_counterclockwise_triangles_l11_11298


namespace count_even_three_digit_numbers_l11_11040

theorem count_even_three_digit_numbers : 
  let num_even_three_digit_numbers : ℕ := 
    have h1 : (units_digit_possible_pairs : list (ℕ × ℕ)) := 
      [(4, 8), (6, 6), (8, 4)]
    have h2 : (number_of_hundreds_digits : ℕ) := 9
    3 * number_of_hundreds_digits 
in
  num_even_three_digit_numbers = 27 := by
  -- steps skipped
  sorry

end count_even_three_digit_numbers_l11_11040


namespace sum_of_squares_l11_11621

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 14) (h2 : a * b + b * c + a * c = 72) : 
  a^2 + b^2 + c^2 = 52 :=
by
  sorry

end sum_of_squares_l11_11621


namespace garden_total_plants_l11_11698

theorem garden_total_plants (rows columns : ℕ) (h_rows : rows = 52) (h_columns : columns = 15) : rows * columns = 780 :=
by
  rw [h_rows, h_columns]
  numbers
  -- By the given conditions, 52 * 15 = 780
  exact sorry

end garden_total_plants_l11_11698


namespace quadratic_function_min_f0_l11_11697

-- Definition of being ever more than another function
def ever_more_than (f g : ℝ → ℝ) : Prop := ∀ x : ℝ, f x ≥ g x

-- The main theorem
theorem quadratic_function_min_f0 :
  ∃ f : ℝ → ℝ, (∀ a b c : ℝ, f = λ x, a*x^2 + b*x + c → a ≠ 0) ∧
               (f 1 = 16) ∧
               (ever_more_than f (λ x, (x + 3)^2)) ∧
               (ever_more_than f (λ x, x^2 + 9)) ∧
               f 0 = 21 / 2 :=
begin
  sorry
end

end quadratic_function_min_f0_l11_11697


namespace unit_circle_arc_length_l11_11119

theorem unit_circle_arc_length (r : ℝ) (A : ℝ) (θ : ℝ) : r = 1 ∧ A = 1 ∧ A = (1 / 2) * r^2 * θ → r * θ = 2 :=
by
  -- Given r = 1 (radius of unit circle) and area A = 1
  -- A = (1 / 2) * r^2 * θ is the formula for the area of the sector
  sorry

end unit_circle_arc_length_l11_11119


namespace amare_fabric_needed_l11_11163

-- Definitions for the conditions
def fabric_per_dress_yards : ℝ := 5.5
def number_of_dresses : ℕ := 4
def fabric_owned_feet : ℝ := 7
def yard_to_feet : ℝ := 3

-- Total fabric needed in yards
def total_fabric_needed_yards : ℝ := fabric_per_dress_yards * number_of_dresses

-- Total fabric needed in feet
def total_fabric_needed_feet : ℝ := total_fabric_needed_yards * yard_to_feet

-- Fabric still needed
def fabric_still_needed : ℝ := total_fabric_needed_feet - fabric_owned_feet

-- Proof
theorem amare_fabric_needed : fabric_still_needed = 59 := by
  sorry

end amare_fabric_needed_l11_11163


namespace fraction_simplification_l11_11319

theorem fraction_simplification : (3 : ℚ) / (2 - (3 / 4)) = 12 / 5 := by
  sorry

end fraction_simplification_l11_11319


namespace ratio_triangle_areas_l11_11429

open EuclideanGeometry

noncomputable def curve_C_eqn (x y : ℝ) : Prop := x^2 = 4 * y

def points_on_curve (x1 x2 : ℝ) : Prop := ∃ (y1 y2 : ℝ), curve_C_eqn x1 y1 ∧ curve_C_eqn x2 y2

def points_conditions {M : Type*} [plane M] (O R Q : M) : Prop :=
  O.1 = 0 ∧ O.2 = 0 ∧ 
  R.1 = -2 ∧ R.2 = 1 ∧
  Q.1 = 2 ∧ Q.2 = 1

theorem ratio_triangle_areas {M : Type*} [plane M] 
(O R Q : M)
(A B P D E : M) 
(x1 x2 k t : ℝ) 
(h_curve : ∀A B, points_on_curve A.1 A.2 ∧ points_on_curve B.1 B.2)
(h_points : points_conditions O R Q) :
let QAB := triangle Q A B in
let PDE := triangle P D E in
(area QAB) / (area PDE) = 2 :=
sorry

end ratio_triangle_areas_l11_11429


namespace maximum_parts_divided_by_three_planes_l11_11635

theorem maximum_parts_divided_by_three_planes : 
  (∀ planes : List Plane, (pairwise_parallel planes → parts planes ≤ 4) ∧ 
    (intersecting_planes planes → parts planes ≤ 8)) → 
  maximum_parts planes = 8 := 
sorry

end maximum_parts_divided_by_three_planes_l11_11635


namespace floor_sqrt_equality_l11_11982

theorem floor_sqrt_equality (n : ℕ) : 
  (floor (sqrt (4 * n + 1)) = floor (sqrt (4 * n + 2))) ∧ 
  (floor (sqrt (4 * n + 2)) = floor (sqrt (4 * n + 3))) :=
sorry

end floor_sqrt_equality_l11_11982


namespace incorrect_conclusions_l11_11462

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

noncomputable def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2

theorem incorrect_conclusions (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, arithmetic_sequence a d) →
  a 2 + a 3 + a 8 = 15 →
  a 9 = 6 →
  a 1 > 0 ∧ Sn a 3 = Sn a 8 →
  ¬ ((a 3 + a 7 = 10) ∧ (Sn a 17 = 204) ∧ (S n reaches maximum → (n = 6 ∨ n = 7)) ∧ 
  (∀ n, (Sn a n) / n is_arithmetic)) :=
sorry

end incorrect_conclusions_l11_11462


namespace correct_sampling_l11_11628

-- Let n be the total number of students
def total_students : ℕ := 60

-- Define the systematic sampling function
def systematic_sampling (n m : ℕ) (start : ℕ) : List ℕ :=
  List.map (λ k => start + k * m) (List.range n)

-- Prove that the sequence generated is equal to [3, 13, 23, 33, 43, 53]
theorem correct_sampling :
  systematic_sampling 6 10 3 = [3, 13, 23, 33, 43, 53] :=
by
  sorry

end correct_sampling_l11_11628


namespace even_three_digit_numbers_l11_11044

theorem even_three_digit_numbers (n : ℕ) :
  (n >= 100 ∧ n < 1000) ∧
  (n % 2 = 0) ∧
  ((n % 100) / 10 + (n % 10) = 12) →
  n = 12 :=
sorry

end even_three_digit_numbers_l11_11044


namespace range_of_a_monotonically_increasing_function_l11_11343

theorem range_of_a_monotonically_increasing_function (a : ℝ) : 
  0 < a ∧ a ≠ 1 ∧ (∀ x > 2, 3 * x^2 - a ≥ 0) ∧ (∀ x > 2, x^3 - ax > 0) → 1 < a ∧ a ≤ 4 :=
begin
  sorry
end

end range_of_a_monotonically_increasing_function_l11_11343


namespace solve_inequality_zero_solve_inequality_neg_solve_inequality_pos_l11_11179

variable (a x : ℝ)

def inequality (a x : ℝ) : Prop := (1 - a * x) ^ 2 < 1

theorem solve_inequality_zero : a = 0 → ¬∃ x, inequality a x := by
  sorry

theorem solve_inequality_neg (h : a < 0) : (∃ x, inequality a x) →
  ∀ x, inequality a x ↔ (a ≠ 0 ∧ (2 / a < x ∧ x < 0)) := by
  sorry

theorem solve_inequality_pos (h : a > 0) : (∃ x, inequality a x) →
  ∀ x, inequality a x ↔ (a ≠ 0 ∧ (0 < x ∧ x < 2 / a)) := by
  sorry

end solve_inequality_zero_solve_inequality_neg_solve_inequality_pos_l11_11179


namespace domain_of_f_l11_11198

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.sqrt (Real.log x / Real.log 2 - 1))

theorem domain_of_f :
  {x : ℝ | x > 2} = {x : ℝ | x > 0 ∧ Real.log x / Real.log 2 - 1 > 0} := 
by
  sorry

end domain_of_f_l11_11198


namespace digit_58_in_1_div_7_is_8_l11_11245

theorem digit_58_in_1_div_7_is_8 :
  (decimal_digit_at 58 (1 / 7) = 8) := 
sorry

end digit_58_in_1_div_7_is_8_l11_11245


namespace floor_sqrt_50_squared_l11_11810

theorem floor_sqrt_50_squared : ∃ x : ℕ, x = 49 ∧ (⌊real.sqrt 50⌋ : ℕ) ^ 2 = x := by
  have h1 : (7 : ℝ) < real.sqrt 50 := sorry
  have h2 : real.sqrt 50 < 8 := sorry
  have h_floor : (⌊real.sqrt 50⌋ : ℕ) = 7 := sorry
  use 49
  constructor
  · rfl
  · rw [h_floor]
    norm_num
    sorry

end floor_sqrt_50_squared_l11_11810


namespace candy_from_sister_l11_11830

variable (f : ℕ) (e : ℕ) (t : ℕ)

theorem candy_from_sister (h₁ : f = 47) (h₂ : e = 25) (h₃ : t = 62) :
  ∃ x : ℕ, x = t - (f - e) ∧ x = 40 :=
by sorry

end candy_from_sister_l11_11830


namespace find_second_smallest_odd_number_l11_11630

theorem find_second_smallest_odd_number (x : ℤ) (h : (x + (x + 2) + (x + 4) + (x + 6) = 112)) : (x + 2 = 27) :=
sorry

end find_second_smallest_odd_number_l11_11630


namespace distance_between_sets_is_zero_l11_11768

noncomputable def A (x : ℝ) : ℝ := 2 * x - 1
noncomputable def B (x : ℝ) : ℝ := x^2 + 1

theorem distance_between_sets_is_zero : 
  ∃ (a b : ℝ), (∃ x₀ : ℝ, a = A x₀) ∧ (∃ y₀ : ℝ, b = B y₀) ∧ abs (a - b) = 0 := 
sorry

end distance_between_sets_is_zero_l11_11768


namespace total_shaded_area_is_correct_l11_11708

variable {S T : ℝ}

-- Given conditions
def square_carpet_side : ℝ := 12
def ratio_large_shaded : ℝ := 4
def ratio_small_shaded : ℝ := 4
def S_value := square_carpet_side / ratio_large_shaded  -- S = 3
def T_value := S_value / ratio_small_shaded  -- T = 3/4

-- Calculations
def area_small_shaded := T_value^2
def total_area_small_shaded := 12 * area_small_shaded
def area_large_shaded := S_value^2

-- Define total shaded area
def total_shaded_area := total_area_small_shaded + area_large_shaded

theorem total_shaded_area_is_correct : total_shaded_area = 15.75 :=
by
  -- Proof omitted
  sorry

end total_shaded_area_is_correct_l11_11708


namespace pictures_left_l11_11257

def initial_zoo_pics : ℕ := 49
def initial_museum_pics : ℕ := 8
def deleted_pics : ℕ := 38

theorem pictures_left (total_pics : ℕ) :
  total_pics = initial_zoo_pics + initial_museum_pics →
  total_pics - deleted_pics = 19 :=
by
  intro h1
  rw [h1]
  sorry

end pictures_left_l11_11257


namespace power_inequality_l11_11133

theorem power_inequality (a b n : ℕ) (h_ab : a > b) (h_b1 : b > 1)
  (h_odd_b : b % 2 = 1) (h_n_pos : 0 < n) (h_div : b^n ∣ a^n - 1) :
  a^b > 3^n / n :=
by
  sorry

end power_inequality_l11_11133


namespace union_complement_U_B_l11_11559

def U : Set ℤ := { x | -3 < x ∧ x < 3 }
def A : Set ℤ := { 1, 2 }
def B : Set ℤ := { -2, -1, 2 }

theorem union_complement_U_B : A ∪ (U \ B) = { 0, 1, 2 } := by
  sorry

end union_complement_U_B_l11_11559


namespace polynomial_degree_l11_11246

theorem polynomial_degree :
  ∀ (p : ℝ[X]), p = 5*x^3 - 7*x + 1 → degree (p^10) = 30 :=
begin
  intros p hp,
  sorry,
end

end polynomial_degree_l11_11246


namespace first_player_wins_on_seven_cells_second_player_avoids_defeat_on_less_than_seven_cells_l11_11371

-- Define the game board, a sequence of cells
def game_board := list ℕ

-- Define the winning condition: three consecutive marks
def win_condition (board : game_board) :=
  ∃ (i : ℕ), board.nth i = some 1 ∧ board.nth (i+1) = some 1 ∧ board.nth (i+2) = some 1

-- Define the minimum number of cells required for the first player to guarantee a win
def minimum_cells_to_win := 7

-- The theorem stating the first player can always win on a board with at least 7 cells
theorem first_player_wins_on_seven_cells (board : game_board) (h : board.length = minimum_cells_to_win) :
  win_condition board :=
sorry

-- The theorem stating the second player can always avoid defeat on a board with fewer than 7 cells
theorem second_player_avoids_defeat_on_less_than_seven_cells (board : game_board) (h : board.length < minimum_cells_to_win) :
  ¬ win_condition board :=
sorry

end first_player_wins_on_seven_cells_second_player_avoids_defeat_on_less_than_seven_cells_l11_11371


namespace product_of_real_roots_l11_11587

def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem product_of_real_roots (x : ℝ) (hx : x > 0) (h : x ^ log10 x = 100) : ∃ p, p = 1 := by
  let a := log10 x
  have ha : a^2 = 2 := by sorry  -- from (\log_{10} x)^2 = 2
  let x1 := 10 ^ a
  let x2 := 10 ^ -a
  have hprod : x1 * x2 = 1 := by sorry -- from 10^{\sqrt{2}} \times 10^{-\sqrt{2}} = 1
  exact ⟨1, hprod ▸ rfl⟩

end product_of_real_roots_l11_11587


namespace division_expression_l11_11752

theorem division_expression :
  (240 : ℚ) / (12 + 12 * 2 - 3) = 240 / 33 := by
  sorry

end division_expression_l11_11752


namespace specially_monotonous_count_is_64_l11_11758

-- Define what it means to be a one-digit specially monotonous number
def one_digit_specially_monotonous (n : ℕ) : Prop :=
  n ≥ 1 ∧ n ≤ 7

-- Define what it means to be a three-digit strictly increasing specially monotonous number
def three_digit_increasing (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 7

-- Define what it means to be a three-digit strictly decreasing specially monotonous number
def three_digit_decreasing (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a > b ∧ b > c ∧ c = 0

-- Define the total count of specially monotonous numbers based on the given conditions.
def total_specially_monotonous : ℕ :=
  let num_one_digit := 7 in -- 1 through 7
  let num_increasing := (Finset.card (Finset.powersetLen 3 (Finset.range 8) \ ({0} : Finset ℕ))) in -- choosing 3 from 1 to 7
  let num_decreasing := (Finset.card (Finset.powersetLen 2 (Finset.range 8 \ {0, 8}))) in -- choosing 2 from 1 to 7 and end with a 0
  num_one_digit + num_increasing + num_decreasing

theorem specially_monotonous_count_is_64 : total_specially_monotonous = 64 := by
  sorry -- proof of the theorem

end specially_monotonous_count_is_64_l11_11758


namespace largest_n_satisfying_conditions_l11_11649

theorem largest_n_satisfying_conditions : 
  ∃ n : ℤ, 200 < n ∧ n < 250 ∧ (∃ k : ℤ, 12 * n = k^2) ∧ n = 243 :=
by
  sorry

end largest_n_satisfying_conditions_l11_11649


namespace chocolates_left_l11_11995

-- Definitions based on the conditions
def initially_bought := 3
def gave_away := 2
def additionally_bought := 3

-- Theorem statement to prove
theorem chocolates_left : initially_bought - gave_away + additionally_bought = 4 := by
  -- Proof skipped
  sorry

end chocolates_left_l11_11995


namespace alpha_beta_sum_l11_11632

theorem alpha_beta_sum (α β : ℝ) 
  (h : ∀ x : ℝ, (x - α) / (x + β) = (x^2 - 80 * x + 1551) / (x^2 + 57 * x - 2970)) :
  α + β = 137 :=
by
  sorry

end alpha_beta_sum_l11_11632


namespace finalSellingPrice_l11_11692

-- Definitions and conditions
def originalPrice : ℝ := 120
def firstDiscount : ℝ := 0.30
def secondDiscount : ℝ := 0.10
def taxRate : ℝ := 0.08

-- Calculation of the sale prices after each discount and final price including tax
def priceAfterFirstDiscount := originalPrice * (1 - firstDiscount)
def priceAfterSecondDiscount := priceAfterFirstDiscount * (1 - secondDiscount)
def totalPriceWithTax := priceAfterSecondDiscount * (1 + taxRate)

-- Proof of the final price
theorem finalSellingPrice : round totalPriceWithTax = 82 := by
  let firstPrice := priceAfterFirstDiscount
  let secondPrice := priceAfterSecondDiscount
  let finalPrice := totalPriceWithTax
  have : firstPrice = 84 := by sorry
  have : secondPrice = 75.6 := by sorry
  have : finalPrice = 81.648 := by sorry
  show round 81.648 = 82 by sorry

end finalSellingPrice_l11_11692


namespace Yihana_uphill_walking_time_l11_11666

theorem Yihana_uphill_walking_time :
  let t1 := 3
  let t2 := 2
  let t_total := t1 + t2
  t_total = 5 :=
by
  let t1 := 3
  let t2 := 2
  let t_total := t1 + t2
  show t_total = 5
  sorry

end Yihana_uphill_walking_time_l11_11666


namespace proof_triangle_properties_l11_11494

variable (A B C : ℝ)
variable (h AB : ℝ)

-- Conditions
def triangle_conditions : Prop :=
  (A + B = 3 * C) ∧ (2 * Real.sin (A - C) = Real.sin B) ∧ (AB = 5)

-- Part 1: Proving sin A
def find_sin_A (h₁ : triangle_conditions A B C h AB) : Prop :=
  Real.sin A = 3 * Real.cos A

-- Part 2: Proving the height on side AB
def find_height_on_AB (h₁ : triangle_conditions A B C h AB) : Prop :=
  h = 6

-- Combined proof statement
theorem proof_triangle_properties (h₁ : triangle_conditions A B C h AB) : 
  find_sin_A A B C h₁ ∧ find_height_on_AB A B C h AB h₁ := 
  by sorry

end proof_triangle_properties_l11_11494


namespace magnitude_b_cosine_theta_l11_11376

variables (a b : Vector ℝ) (θ : ℝ)

-- Given conditions
def conditions := 
  (‖a‖ = 1) ∧ 
  (a • b = 1 / 4) ∧ 
  ((a + b) • (a - b) = 1 / 2)

-- Part (1): Magnitude of vector b
theorem magnitude_b (h : conditions a b) : ‖b‖ = (Real.sqrt 2) / 2 :=
sorry

-- Part (2): Cosine of the angle between (a - b) and (a + b)
theorem cosine_theta (h : conditions a b) : 
  Real.cos θ = (Real.sqrt 2) / 4 :=
sorry

end magnitude_b_cosine_theta_l11_11376


namespace smallest_n_positive_odd_integer_l11_11653

theorem smallest_n_positive_odd_integer (n : ℕ) (h1 : n % 2 = 1) (h2 : 3 ^ ((n + 1)^2 / 5) > 500) : n = 6 := sorry

end smallest_n_positive_odd_integer_l11_11653


namespace square_window_side_length_l11_11094

-- Definitions based on the conditions
def total_panes := 8
def rows := 2
def cols := 4
def height_ratio := 3
def width_ratio := 1
def border_width := 3

-- The statement to prove
theorem square_window_side_length :
  let height := 3 * (1 : ℝ)
  let width := 1 * (1 : ℝ)
  let total_width := cols * width + (cols + 1) * border_width
  let total_height := rows * height + (rows + 1) * border_width
  total_width = total_height → total_width = 27 :=
by
  sorry

end square_window_side_length_l11_11094


namespace hyperbola_focus_asymptote_distance_is_correct_l11_11822

noncomputable def hyperbola_focus_asymptote_distance : ℝ :=
  let a := Real.sqrt 5
  let b := 2
  let c := 3
  let focus : ℝ × ℝ := (3, 0)
  let asymptote : ℝ × ℝ → Prop := λ p, 2 * p.1 - Real.sqrt 5 * p.2 = 0
  in dist_point_line focus asymptote

-- Prove that the distance calculated is equal to 2
theorem hyperbola_focus_asymptote_distance_is_correct :
  hyperbola_focus_asymptote_distance = 2 := sorry

end hyperbola_focus_asymptote_distance_is_correct_l11_11822


namespace sum_of_possible_x_l11_11844

def arithmetic_progression (a b c : ℝ) : Prop := b - a = c - b

theorem sum_of_possible_x :
  let lst := [18, 3, 5, 3, 7, 3, (x : ℝ), 12] in
  let mean := (18 + 3 + 5 + 3 + 7 + 3 + x + 12) / 8 in
  let mode := 3 in
  (∀ x, 
    (arithmetic_progression mode x mean ∨
     arithmetic_progression mode mean x ∨
     arithmetic_progression x mode mean) →
    x = 37) → 
  let sum := 37 in
  sum = 37 :=
by sorry

end sum_of_possible_x_l11_11844


namespace constant_sequence_possible_l11_11386

theorem constant_sequence_possible : 
  ∃ (seq : Fin 2015 → ℝ), 
  (∀ i : Fin 2015, seq i = i) → 
  ∃ (steps : ℕ), ∃ (f : (Fin 2015 → ℝ) → (Fin 2015 → ℝ)), 
  (∀ n < steps, is_move (f^[n] seq)) ∧ 
  (∀ j : Fin 2015, (f^[steps] seq) j = 1007) :=
sorry

def is_move (s : Fin 2015 → ℝ) : Prop := 
  ∃ i j : Fin 2015, i ≠ j ∧ 
  ∀ k : Fin 2015, 
  (k = i ∨ k = j → s k = (s i + s j) / 2) ∧ 
  (k ≠ i ∧ k ≠ j → s k = s k)

end constant_sequence_possible_l11_11386


namespace sin_A_correct_height_on_AB_correct_l11_11472

noncomputable def sin_A (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) : ℝ :=
  Real.sin A

noncomputable def height_on_AB (A B C AB : ℝ) (height : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) (h4 : AB = 5) : ℝ :=
  height

theorem sin_A_correct (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) : 
  sorrry := 
begin
  -- proof omitted
  sorrry
end

theorem height_on_AB_correct (A B C AB : ℝ) (height : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) (h4 : AB = 5) :
  height = 6:= 
begin
  -- proof omitted
  sorrry
end 

end sin_A_correct_height_on_AB_correct_l11_11472


namespace g_at_5_l11_11328

def g (x : ℝ) : ℝ := sorry -- Placeholder for the function definition, typically provided in further context

theorem g_at_5 : g 5 = 3 / 4 :=
by
  -- Given condition as a hypothesis
  have h : ∀ x: ℝ, g x + 3 * g (2 - x) = 4 * x^2 - 1 := sorry
  sorry  -- Full proof should go here

end g_at_5_l11_11328


namespace a_can_be_any_sign_l11_11436

theorem a_can_be_any_sign (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) (h : (a / b)^2 < (c / d)^2) (hcd : c = -d) : True :=
by
  have := h
  subst hcd
  sorry

end a_can_be_any_sign_l11_11436


namespace evaluate_f_at_5_l11_11871

def f : ℕ → ℕ
| x := if x < 4 then 2 ^ x else f (x - 1)

theorem evaluate_f_at_5 : f 5 = 8 :=
sorry

end evaluate_f_at_5_l11_11871


namespace factorial_zero_remainder_l11_11948

-- Given that M is the number of consecutive 0's at the right end of the product of factorials from 1 to 50,
-- we want to find the remainder when M is divided by 500.
def factorial_zero_count : ℤ := (List.range' 1 51).map factorial |> List.foldl (*) 1 |> nat_trailing_zeros
theorem factorial_zero_remainder : factorial_zero_count % 500 = 12 := by
  sorry

end factorial_zero_remainder_l11_11948


namespace arithmetic_mean_absolute_difference_permutations_l11_11833

open BigOperators

theorem arithmetic_mean_absolute_difference_permutations :
  let permutations := (1:ℕ).upto 10
  let sums := ∑ (perm in permutations.permutations), 
    (| perm[0] - perm[1] | + | perm[2] - perm[3] | + 
     | perm[4] - perm[5] | + | perm[6] - perm[7] | + 
     | perm[8] - perm[9] | : ℚ)
  sums / permutations.permutations.card = 55 / 3 := 
  sorry

end arithmetic_mean_absolute_difference_permutations_l11_11833


namespace smallest_palindrome_base2_base3_l11_11248

def is_palindrome {α : Type} [Inhabited α] [DecidableEq α] (s : List α) : Prop :=
  s = s.reverse

theorem smallest_palindrome_base2_base3 :
  ∃ n : ℕ, n = 17 ∧ is_palindrome (nat.to_digits 3 n) ∧ (nat.to_digits 3 n).length = 3 :=
begin
  use 17,
  split,
  { refl },
  split,
  { -- Check if 122 is a palindrome
    unfold is_palindrome,
    have h : nat.to_digits 3 17 = [1, 2, 2], by norm_num,
    rw h,
    refl },
  { -- Check the length
    have h : nat.to_digits 3 17 = [1, 2, 2], by norm_num,
    rw h,
    norm_num }
end

end smallest_palindrome_base2_base3_l11_11248


namespace hexagon_arrangements_eq_144_l11_11989

def is_valid_arrangement (arr : (Fin 7 → ℕ)) : Prop :=
  ∀ (i j k : Fin 7),
    (i.val + j.val + k.val = 18) → -- 18 being a derived constant factor (since 3x = 28 + 2G where G ∈ {1, 4, 7} and hence x = 30,34,38/3 respectively make it divisible by 3 sum is 18 always)
    arr i + arr j + arr k = arr ⟨3, sorry⟩ -- arr[3] is the position of G

noncomputable def count_valid_arrangements : ℕ :=
  sorry -- Calculation of 3*48 goes here and respective pairing and permutations.

theorem hexagon_arrangements_eq_144 :
  count_valid_arrangements = 144 :=
sorry

end hexagon_arrangements_eq_144_l11_11989


namespace solution_set_of_inequality_l11_11685

theorem solution_set_of_inequality (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_decreasing : ∀ x y, x < y → f y < f x) :
  {a : ℝ | f (a^2) + f (2*a) > 0} = set.Ioo (-2 : ℝ) 0 :=
sorry

end solution_set_of_inequality_l11_11685


namespace hunter_played_basketball_l11_11065

theorem hunter_played_basketball :
  ∀ (football_time : ℕ) (total_time_hours : ℝ),
  football_time = 60 →
  total_time_hours = 1.5 →
  let total_time_minutes := total_time_hours * 60 in
  total_time_minutes - football_time = 30 :=
by
  intros football_time total_time_hours h1 h2
  let total_time_minutes := total_time_hours * 60
  sorry

end hunter_played_basketball_l11_11065


namespace cylinder_ball_volume_ratio_l11_11271

theorem cylinder_ball_volume_ratio (R : ℝ) : 
  let V_cylinder := 2 * Real.pi * R^3 in
  let V_ball := (4 / 3) * Real.pi * R^3 in
  V_cylinder / V_ball = 3 / 2 :=
by
  sorry

end cylinder_ball_volume_ratio_l11_11271


namespace sqrt_inequality_l11_11169

theorem sqrt_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt (a ^ 2 / b) + Real.sqrt (b ^ 2 / a) ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end sqrt_inequality_l11_11169


namespace octagon_area_sum_l11_11847

theorem octagon_area_sum (s : ℝ) (h : s = 3) :
  let square_area := s * s,
      triangle_height := s * sqrt(2 + sqrt 2),
      triangle_area := (1/2) * s * triangle_height,
      total_triangle_area := 4 * triangle_area
  in square_area + total_triangle_area = 9 + 18 * sqrt(2 + sqrt 2) :=
by
  -- Proof placeholder
  sorry

end octagon_area_sum_l11_11847


namespace three_digit_even_sum_12_l11_11031

theorem three_digit_even_sum_12 : 
  ∃ (n : Finset ℕ), 
    n.card = 27 ∧ 
    ∀ x ∈ n, 
      ∃ h t u, 
        (100 * h + 10 * t + u = x) ∧ 
        (h ∈ Finset.range 9 \ {0}) ∧ 
        (u % 2 = 0) ∧ 
        (t + u = 12) := 
sorry

end three_digit_even_sum_12_l11_11031


namespace max_tan_C_minus_A_l11_11933

   variable {A B C a b c : ℝ}

   /-- In triangle ABC, the sides opposite to angles A, B, and C are a, b, and c respectively.
       Given c^2 = a^2 + (1/3) b^2,
       we need to prove that the maximum value of tan(C - A) is sqrt(2)/4. -/
   theorem max_tan_C_minus_A (h : c^2 = a^2 + (1/3) * b^2) :
     ∃ t : ℝ, t = Real.arctan (C - A) ∧ t ≤ Real.sqrt(2) / 4 :=
   sorry
   
end max_tan_C_minus_A_l11_11933


namespace mean_equality_l11_11612

theorem mean_equality (x : ℚ) : 
  (3 + 7 + 15) / 3 = (x + 10) / 2 → x = 20 / 3 := 
by 
  sorry

end mean_equality_l11_11612


namespace problem_l11_11999

variable (x y : ℝ)

theorem problem
  (h : (3 * x + 1) ^ 2 + |y - 3| = 0) :
  (x + 2 * y) * (x - 2 * y) + (x + 2 * y) ^ 2 - x * (2 * x + 3 * y) = -1 :=
sorry

end problem_l11_11999


namespace minimum_value_of_abs_sum_l11_11607

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) + 2 * cos (x) ^ 2

theorem minimum_value_of_abs_sum (x₁ x₂ : ℝ) 
  (h : f x₁ * f x₂ = -3) : 
  ∃ (k₁ k₂ : ℤ), 
    |(k₁ + k₂ : ℝ) * real.pi - real.pi / 6| = real.pi / 6 := 
sorry

end minimum_value_of_abs_sum_l11_11607


namespace inequality_proof_l11_11556

theorem inequality_proof
  (a b c d : ℝ)
  (a_nonneg : 0 ≤ a)
  (b_nonneg : 0 ≤ b)
  (c_nonneg : 0 ≤ c)
  (d_nonneg : 0 ≤ d)
  (sum_eq_one : a + b + c + d = 1) :
  abc + bcd + cda + dab ≤ (1 / 27) + (176 * abcd / 27) :=
sorry

end inequality_proof_l11_11556


namespace bob_water_percentage_l11_11322

-- Define the number of acres of each crop for each farmer
def acres_bob := (corn := 3, cotton := 9, beans := 12)
def acres_brenda := (corn := 6, cotton := 7, beans := 14)
def acres_bernie := (corn := 2, cotton := 12, beans := 0)

-- Define the water requirements per acre for each crop
def water_per_acre_corn := 20
def water_per_acre_cotton := 80
def water_per_acre_beans := 2 * water_per_acre_corn

-- Calculate total water usage for each farmer
def water_bob := acres_bob.corn * water_per_acre_corn + acres_bob.cotton * water_per_acre_cotton + acres_bob.beans * water_per_acre_beans
def water_brenda := acres_brenda.corn * water_per_acre_corn + acres_brenda.cotton * water_per_acre_cotton + acres_brenda.beans * water_per_acre_beans
def water_bernie := acres_bernie.corn * water_per_acre_corn + acres_bernie.cotton * water_per_acre_cotton

-- Calculate the total water usage for all farmers
def total_water := water_bob + water_brenda + water_bernie

-- Define the percentage of the total water used that will go to Farmer Bob's farm
def percentage_water_bob := (water_bob / total_water) * 100

-- Theorem stating that 36% of the total water used will go to Farmer Bob's farm
theorem bob_water_percentage : percentage_water_bob = 36 := 
sorry

end bob_water_percentage_l11_11322


namespace hydropolis_more_rain_than_aquaville_l11_11086

theorem hydropolis_more_rain_than_aquaville :
  ∀ (rainfall_hydro_2010 rainfall_hydro_inc_2011 rainfall_aqua_dec_2011 : ℝ)
  (months : ℕ),
  rainfall_hydro_2010 = 36.5 →
  rainfall_hydro_inc_2011 = 3.5 →
  rainfall_aqua_dec_2011 = 1.5 →
  months = 12 →
  let rainfall_hydro_2011 := rainfall_hydro_2010 + rainfall_hydro_inc_2011 in
  let total_rainfall_hydro_2011 := months * rainfall_hydro_2011 in
  let rainfall_aqua_2011 := rainfall_hydro_2011 - rainfall_aqua_dec_2011 in
  let total_rainfall_aqua_2011 := months * rainfall_aqua_2011 in
  (total_rainfall_hydro_2011 - total_rainfall_aqua_2011) = 18 :=
begin
  intros,
  dsimp,
  sorry
end

end hydropolis_more_rain_than_aquaville_l11_11086


namespace calc_expr_l11_11318

noncomputable def expr : ℝ := ((- real.sqrt 2) ^ 2) ^ - (1 / 2)

theorem calc_expr : expr = real.sqrt 2 / 2 :=
by 
  sorry

end calc_expr_l11_11318


namespace nobel_prize_laureates_l11_11282

-- Definitions for the given conditions
variables (S W N W_inter_N N' : ℕ)

-- Given conditions as hypotheses
def workshop_conditions := 
  S = 50 ∧ 
  W = 31 ∧ 
  W_inter_N = 14 ∧ 
  (S - W = N' + (N' - 3)) ∧ 
  2 * N' - 3 = 19

-- The theorem to prove
theorem nobel_prize_laureates : workshop_conditions S W N W_inter_N N' → N = 25 :=
by 
  intros h
  rw [workshop_conditions] at h
  sorry

end nobel_prize_laureates_l11_11282


namespace number_of_valid_three_digit_even_numbers_l11_11003

def valid_three_digit_even_numbers (n : ℕ) : Prop :=
  (100 ≤ n) ∧ (n < 1000) ∧ (n % 2 = 0) ∧ (let t := (n / 10) % 10 in
                                           let u := n % 10 in
                                           t + u = 12)

theorem number_of_valid_three_digit_even_numbers : 
  (∃ cnt : ℕ, cnt = 27 ∧ (cnt = (count (λ n, valid_three_digit_even_numbers n) (Ico 100 1000)))) :=
sorry

end number_of_valid_three_digit_even_numbers_l11_11003


namespace roots_of_quadratic_equation_l11_11888

theorem roots_of_quadratic_equation (a b c r s : ℝ) 
  (hr : a ≠ 0)
  (h : a * r^2 + b * r - c = 0)
  (h' : a * s^2 + b * s - c = 0)
  :
  (1 / r^2) + (1 / s^2) = (b^2 + 2 * a * c) / c^2 :=
by
  sorry

end roots_of_quadratic_equation_l11_11888


namespace james_total_payment_is_correct_l11_11124

-- Define the constants based on the conditions
def numDirtBikes : Nat := 3
def costPerDirtBike : Nat := 150
def numOffRoadVehicles : Nat := 4
def costPerOffRoadVehicle : Nat := 300
def numTotalVehicles : Nat := numDirtBikes + numOffRoadVehicles
def registrationCostPerVehicle : Nat := 25

-- Define the total calculation using the given conditions
def totalPaidByJames : Nat :=
  (numDirtBikes * costPerDirtBike) +
  (numOffRoadVehicles * costPerOffRoadVehicle) +
  (numTotalVehicles * registrationCostPerVehicle)

-- State the proof problem
theorem james_total_payment_is_correct : totalPaidByJames = 1825 := by
  sorry

end james_total_payment_is_correct_l11_11124


namespace coordinates_respect_to_origin_l11_11104

def point_coordinates : ℝ × ℝ := (-1, 2)

theorem coordinates_respect_to_origin :
  point_coordinates = (-1, 2) :=
begin
  -- We refer to the definition of the point
  -- as stated in the conditions.
  refl,
end

end coordinates_respect_to_origin_l11_11104


namespace general_term_and_sum_geometric_seq_sum_T_l11_11425

-- Definitions based on conditions
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def a (n : ℕ) : ℤ := -2 * n + 5
def S (n : ℕ) : ℤ := -n^2 + 4 * n
def c (n : ℕ) : ℤ := (5 - a n)
def b (n : ℕ) : ℤ := 2 ^ (c n)

-- Problem 1
theorem general_term_and_sum (h : arithmetic_seq a) (h2 : a 2 = 1) (h6 : a 6 = -5) :
  ∀ n : ℕ, a n = -2 * n + 5 ∧ S n = -n^2 + 4 * n := 
sorry

-- Problem 2
theorem geometric_seq (h : ∀ n : ℕ, c n = n) : ∃ r : ℤ, ∀ n : ℕ, b (n + 1) = r * b n :=
sorry

-- Problem 3
def T (n : ℕ) : ℚ := ∑ i in finset.range n, 1 / (↑(2 * (i + 1))^2 - 1)

theorem sum_T (h : ∀ n : ℕ, c n = 2 * n) :
  ∀ n : ℕ, T n = n / (2 * n + 1) :=
sorry

end general_term_and_sum_geometric_seq_sum_T_l11_11425


namespace area_of_region_l11_11450

-- The problem definition
def condition_1 (z : ℂ) : Prop := 
  0 < z.re / 20 ∧ z.re / 20 < 1 ∧
  0 < z.im / 20 ∧ z.im / 20 < 1 ∧
  0 < (20 / z).re ∧ (20 / z).re < 1 ∧
  0 < (20 / z).im ∧ (20 / z).im < 1

-- The proof statement
theorem area_of_region {z : ℂ} (h : condition_1 z) : 
  ∃ s : ℝ, s = 300 - 50 * Real.pi := sorry

end area_of_region_l11_11450


namespace midpoint_distance_ge_half_AB_l11_11578

theorem midpoint_distance_ge_half_AB (A B C P Q : Point) (circumcircle : Circle) (triangle ABC : Triangle)
  (H1 : Q = midpoint B C)
  (H2 : P = midpoint_arc circumcircle B C A) :
  dist P Q ≥ dist A B / 2 :=
sorry

end midpoint_distance_ge_half_AB_l11_11578


namespace vasya_slowest_trip_l11_11268

-- Definitions based on conditions
def vasya_speed (x : ℝ) : ℝ := x  -- Vasya's (and Petya's) speed
def kolya_speed (v : ℝ) : ℝ := v  -- Kolya's speed
def distance (S : ℝ) : ℝ := S  -- Distance from home to the lake
def time_ratio_petya_vasya := (4/3 : ℝ)  -- Time ratio of Petya to Vasya

-- Main statement to prove
theorem vasya_slowest_trip (x v S : ℝ)
    (h1 : v = 5 * x)
    (h2 : (distance S) > 0) :
    (S / x) = 3 * (2 * S / (x + v)) :=
by
  sorry

end vasya_slowest_trip_l11_11268


namespace sqrt_floor_squared_l11_11812

/-- To evaluate the floor of the square root of 50 squared --/
theorem sqrt_floor_squared :
  (⌊real.sqrt 50⌋ : ℕ)^2 = 49 :=
begin
  -- We know that:
  -- 7^2 = 49 < 50 < 64 = 8^2
  have h1 : 7^2 = 49, by linarith,
  have h2 : 64 = 8^2, by linarith,
  have h3 : (7 : real) < real.sqrt 50, by {
    rw [sqrt_lt],
    exact_mod_cast h1,
  },
  have h4 : real.sqrt 50 < 8, by {
    rw [lt_sqrt],
    exact_mod_cast h2,
  },
  -- Therefore, 7 < sqrt(50) < 8.
  have h5 : (⌊real.sqrt 50⌋ : ℕ) = 7, by {
    rw [nat.floor_eq_iff],
    split,
    { exact_mod_cast h3, },
    { exact_mod_cast h4, },
  },
  -- Thus, ⌊sqrt(50)⌋^2 = 7^2 = 49.
  rw h5,
  exact h1,
end

end sqrt_floor_squared_l11_11812


namespace legs_per_bee_l11_11689

def number_of_bees : ℕ := 8
def total_legs : ℕ := 48

theorem legs_per_bee : (total_legs / number_of_bees) = 6 := by
  sorry

end legs_per_bee_l11_11689


namespace sqrt_floor_squared_l11_11813

/-- To evaluate the floor of the square root of 50 squared --/
theorem sqrt_floor_squared :
  (⌊real.sqrt 50⌋ : ℕ)^2 = 49 :=
begin
  -- We know that:
  -- 7^2 = 49 < 50 < 64 = 8^2
  have h1 : 7^2 = 49, by linarith,
  have h2 : 64 = 8^2, by linarith,
  have h3 : (7 : real) < real.sqrt 50, by {
    rw [sqrt_lt],
    exact_mod_cast h1,
  },
  have h4 : real.sqrt 50 < 8, by {
    rw [lt_sqrt],
    exact_mod_cast h2,
  },
  -- Therefore, 7 < sqrt(50) < 8.
  have h5 : (⌊real.sqrt 50⌋ : ℕ) = 7, by {
    rw [nat.floor_eq_iff],
    split,
    { exact_mod_cast h3, },
    { exact_mod_cast h4, },
  },
  -- Thus, ⌊sqrt(50)⌋^2 = 7^2 = 49.
  rw h5,
  exact h1,
end

end sqrt_floor_squared_l11_11813


namespace area_of_triangle_ABC_l11_11571

open_locale real

noncomputable def area_ΔABC (AF AC AD AB area_ADEF : ℝ) : ℝ :=
  let ratio_AF_AC := AF / AC,
      ratio_AD_AB := AD / AB,
      area_ratio  := ratio_AF_AC * ratio_AD_AB in
  area_ADEF / area_ratio

theorem area_of_triangle_ABC :
  ∀ (AF AC AD AB area_ADEF : ℝ),
    AF = 6 → AC = 33 → AD = 7 → AB = 26 → area_ADEF = 14 →
    area_ΔABC AF AC AD AB area_ADEF = 286 :=
by intros AF AC AD AB area_ADEF hAF hAC hAD hAB hArea_ADEF
   rw [hAF, hAC, hAD, hAB, hArea_ADEF]
   sorry

end area_of_triangle_ABC_l11_11571


namespace count_even_three_digit_numbers_with_sum_12_l11_11015

noncomputable def even_three_digit_numbers_with_sum_12 : Prop :=
  let valid_pairs := [(8, 4), (6, 6), (4, 8)] in
  let valid_hundreds := 9 in
  let count_pairs := valid_pairs.length in
  let total_numbers := valid_hundreds * count_pairs in
  total_numbers = 27

theorem count_even_three_digit_numbers_with_sum_12 : even_three_digit_numbers_with_sum_12 :=
by
  sorry

end count_even_three_digit_numbers_with_sum_12_l11_11015


namespace time_to_cover_length_l11_11264

-- Definitions from conditions
def escalator_speed : Real := 15 -- ft/sec
def escalator_length : Real := 180 -- feet
def person_speed : Real := 3 -- ft/sec

-- Combined speed definition
def combined_speed : Real := escalator_speed + person_speed

-- Lean theorem statement proving the time taken
theorem time_to_cover_length : escalator_length / combined_speed = 10 := by
  sorry

end time_to_cover_length_l11_11264


namespace butterfly_distance_A5_l11_11286

noncomputable def butterfly_distance : ℝ :=
let ω := complex.exp (-real.pi * complex.I / 3) in
let z := 1 + 2 * ω + 3 * (ω ^ 2) + 4 * (ω ^ 3) + 5 * (ω ^ 4) in
complex.abs z

theorem butterfly_distance_A5 :
  butterfly_distance = (5 * real.sqrt 3) / 3 :=
by
  sorry

end butterfly_distance_A5_l11_11286


namespace markup_calculation_l11_11674

def purchase_price : ℝ := 48
def overhead_percentage : ℝ := 0.25
def net_profit : ℝ := 12

def overhead := purchase_price * overhead_percentage
def total_cost := purchase_price + overhead
def selling_price := total_cost + net_profit
def markup := selling_price - purchase_price

theorem markup_calculation : markup = 24 := by
  sorry

end markup_calculation_l11_11674


namespace solution_set_l11_11858

noncomputable def f : ℝ → ℝ := sorry
def g (x : ℝ) : ℝ := f x - 1

axiom f_domain : ∀ x : ℝ, f x ∈ ℝ

axiom f_decreasing (x1 x2 : ℝ) (h : x1 ≠ x2) : (f x1 - f x2) / (x1 - x2) < 0

axiom g_odd (x : ℝ) : g x = -g (-x)

theorem solution_set (m : ℝ) : f (m ^ 2) + f (2 * m - 3) > 2 ↔ -3 < m ∧ m < 1 :=
by
  sorry

end solution_set_l11_11858


namespace correct_answers_is_36_l11_11671

noncomputable def num_correct_answers (c w : ℕ) : Prop :=
  (c + w = 50) ∧ (4 * c - w = 130)

theorem correct_answers_is_36 (c w : ℕ) (h : num_correct_answers c w) : c = 36 :=
by
  sorry

end correct_answers_is_36_l11_11671


namespace route_b_quicker_l11_11968

def time_route_a := (2 / 25 * 60) + (5 / 35 * 60)
def time_route_b := (5 / 45 * 60) + (1 / 15 * 60)

theorem route_b_quicker : (time_route_a - time_route_b) = 2.7 := by
  sorry

end route_b_quicker_l11_11968


namespace smaller_rectangle_length_ratio_l11_11360

theorem smaller_rectangle_length_ratio 
  (s : ℝ)
  (h1 : 5 = 5)
  (h2 : ∃ r : ℝ, r = s)
  (h3 : ∀ x : ℝ, x = s)
  (h4 : ∀ y : ℝ, y / 2 = s / 2)
  (h5 : ∀ z : ℝ, z = 3 * s)
  (h6 : ∀ w : ℝ, w = s) :
  ∃ l : ℝ, l / s = 4 :=
sorry

end smaller_rectangle_length_ratio_l11_11360


namespace sin_A_calculation_height_calculation_l11_11501

variable {A B C : ℝ}

-- Given conditions
def angle_condition : Prop := A + B = 3 * C
def sine_condition : Prop := 2 * sin (A - C) = sin B

-- Part 1: Find sin A
theorem sin_A_calculation (h1 : angle_condition) (h2 : sine_condition) : sin A = 3 * real.sqrt 10 / 10 := sorry

-- Part 2: Given AB = 5, find the height
variable {AB : ℝ}
def AB_value : Prop := AB = 5

theorem height_calculation (h1 : angle_condition) (h2 : sine_condition) (h3 : AB_value) : height = 6 := sorry

end sin_A_calculation_height_calculation_l11_11501


namespace ratio_of_fractions_l11_11440

-- Given conditions
variables {x y : ℚ}
variables (h1 : 5 * x = 3 * y) (h2 : x * y ≠ 0)

-- Assertion to be proved
theorem ratio_of_fractions (h1 : 5 * x = 3 * y) (h2 : x * y ≠ 0) :
  (1 / 5 * x) / (1 / 6 * y) = 18 / 25 :=
sorry

end ratio_of_fractions_l11_11440


namespace power_of_product_zeros_l11_11253

theorem power_of_product_zeros:
  ∀ (n : ℕ), (500 : ℕ) = 5 * 10^2 → 
  500^n = 5^n * (10^2)^n → 
  10^(2 * n) = 10^(2 * 150) → 
  500^150 = 1 * 10^300 → 
  (500^150).to_digits.count 0 = 300 := 
by 
  intros n h1 h2 h3 h4 
  sorry

end power_of_product_zeros_l11_11253


namespace mean_median_difference_l11_11970

variable (total_students : ℕ)
variable (score_60_percentage score_75_percentage score_85_percentage score_90_percentage : ℕ)
variable (score_60_count score_75_count score_85_count score_90_count score_100_count : ℕ)
variable (mean_score median_score difference : ℝ)

-- Given conditions
def conditions :=
  score_60_percentage = 15 ∧
  score_75_percentage = 20 ∧
  score_85_percentage = 30 ∧
  score_90_percentage = 10 ∧
  score_60_count = total_students * score_60_percentage / 100 ∧
  score_75_count = total_students * score_75_percentage / 100 ∧
  score_85_count = total_students * score_85_percentage / 100 ∧
  score_90_count = total_students * score_90_percentage / 100 ∧
  score_100_count = total_students * (100 - (score_60_percentage + score_75_percentage + score_85_percentage + score_90_percentage)) / 100 ∧
  median_score = 85

-- Proof goal
theorem mean_median_difference :
  conditions total_students score_60_percentage score_75_percentage score_85_percentage score_90_percentage score_60_count score_75_count score_85_count score_90_count score_100_count mean_score median_score 1.5 →
  (mean_score * total_students = (60 * score_60_count + 75 * score_75_count + 85 * score_85_count + 90 * score_90_count + 100 * score_100_count)) →
  |mean_score - median_score| = difference :=
by
  sorry

end mean_median_difference_l11_11970


namespace max_edges_no_cycle_length_4_l11_11458

theorem max_edges_no_cycle_length_4 (G : SimpleGraph (Fin 8)) (h_no_cycle_4 : ∀ (C : Finset (Fin 8)), C.card = 4 → ¬ G.Cycle C) :
  G.edge_count ≤ 25 := sorry

end max_edges_no_cycle_length_4_l11_11458


namespace problem_I_problem_II_l11_11418

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x ^ 2 - a * Real.log x - 1

noncomputable def F (a : ℝ) (x : ℝ) : ℝ :=
  a - 1 - a / (1 + Real.sqrt x)

theorem problem_I (a : ℝ) :
  (∀ x ∈ set.Icc 3 5, 2 * x - a / x ≥ 0) ↔ a ≤ 18 :=
by
  sorry

theorem problem_II (x : ℝ) :
  0 < x → x ≠ 1 → (a = 2 → (if 0 < x ∧ x < 1 then (f 2 x) / (x - 1) < F 2 x else if x > 1 then (f 2 x) / (x - 1) > F 2 x else false)) :=
by
  sorry

end problem_I_problem_II_l11_11418


namespace additional_distance_if_faster_speed_l11_11892

-- Conditions
def speed_slow := 10 -- km/hr
def speed_fast := 15 -- km/hr
def actual_distance := 30 -- km

-- Question and answer
theorem additional_distance_if_faster_speed : (speed_fast * (actual_distance / speed_slow) - actual_distance) = 15 := by
  sorry

end additional_distance_if_faster_speed_l11_11892


namespace sin_A_and_height_on_AB_l11_11508

theorem sin_A_and_height_on_AB 
  (A B C: ℝ)
  (h_triangle: ∀ A B C, A + B + C = π)
  (h_angle_sum: A + B = 3 * C)
  (h_sin_condition: 2 * Real.sin (A - C) = Real.sin B)
  (h_AB: AB = 5)
  (h_sqrt_two: Real.cos (π / 4) = Real.sin (π / 4) := by norm_num) :
  (Real.sin A = 3 * Real.sqrt 10 / 10) ∧ (height_on_AB = 6) :=
sorry

end sin_A_and_height_on_AB_l11_11508


namespace arithmetic_sequence_inequality_l11_11545

noncomputable def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

noncomputable def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_inequality
  (a d : ℕ)
  (i j k l : ℕ)
  (hi : i ≤ j)
  (hj : j ≤ k)
  (hk : k ≤ l)
  (hij: i + l = j + k)
  : (arithmetic_seq a d i) * (arithmetic_seq a d l) ≤ (arithmetic_seq a d j) * (arithmetic_seq a d k) :=
sorry

end arithmetic_sequence_inequality_l11_11545


namespace min_groups_needed_l11_11699

-- Definitions based on conditions
def students := {1, 2, 3, 4, 5, 6}
def group (S : Set ℕ) : Prop := S.card = 3

-- A valid configuration contains study and service groups
structure Configuration :=
  (studyGroups : Set (Set ℕ))
  (serviceGroups : Set (Set ℕ))
  (studyGroup_cond : ∀ S ∈ studyGroups, group S)
  (serviceGroup_cond : ∀ S ∈ serviceGroups, group S)
  (pairwise_cond : ∀ x y ∈ students, 
    (∃ S1 ∈ studyGroups, x ∈ S1 ∧ y ∈ S1) = (∃ S2 ∈ serviceGroups, x ∈ S2 ∧ y ∈ S2))
  (non_empty_cond : studyGroups.Nonempty ∧ serviceGroups.Nonempty)
  (unique_triple_cond : ∀ x y z ∈ students, 
    ¬ (∃ S1 ∈ studyGroups, x ∈ S1 ∧ y ∈ S1 ∧ z ∈ S1) 
    ∧ ¬ (∃ S2 ∈ serviceGroups, x ∈ S2 ∧ y ∈ S2 ∧ z ∈ S2))

-- The theorem statement proving the minimum number of groups is 8
theorem min_groups_needed : 
  ∃ c : Configuration, c.studyGroups.card + c.serviceGroups.card = 8 :=
sorry

end min_groups_needed_l11_11699


namespace coordinates_respect_to_origin_l11_11103

def point_coordinates : ℝ × ℝ := (-1, 2)

theorem coordinates_respect_to_origin :
  point_coordinates = (-1, 2) :=
begin
  -- We refer to the definition of the point
  -- as stated in the conditions.
  refl,
end

end coordinates_respect_to_origin_l11_11103


namespace right_triangle_altitude_max_l11_11122

theorem right_triangle_altitude_max (XYZ : Triangle)
  (h1 h2 : ℝ)
  (h1_pos : h1 = 6)
  (h2_pos : h2 = 18)
  (h3_integer : ∃ (h3 : ℕ), is_altitude XYZ h3 ∧ (h3 : ℝ)) :
  ∃ (max_h : ℕ), max_h = 12 :=
by
  sorry

end right_triangle_altitude_max_l11_11122


namespace Abhay_takes_1_hour_less_than_Sameer_l11_11918

noncomputable def Sameer_speed := 42 / (6 - 2)
noncomputable def Abhay_time_doubled_speed := 42 / (2 * 7)
noncomputable def Sameer_time := 42 / Sameer_speed

theorem Abhay_takes_1_hour_less_than_Sameer
  (distance : ℝ := 42)
  (Abhay_speed : ℝ := 7)
  (Sameer_speed : ℝ := Sameer_speed)
  (time_Sameer : ℝ := distance / Sameer_speed)
  (time_Abhay_doubled_speed : ℝ := distance / (2 * Abhay_speed)) :
  time_Sameer - time_Abhay_doubled_speed = 1 :=
by
  sorry

end Abhay_takes_1_hour_less_than_Sameer_l11_11918


namespace tangent_line_eq_at_minus_one_l11_11203

theorem tangent_line_eq_at_minus_one :
  let f := λ x : ℝ, x^3
  let f' := λ x : ℝ, 3 * x^2
  ∀ x y : ℝ, x = -1 → y = -1 → 
  (∀ x_a : ℝ, y = f x → (y + 1 = f' x_a * (x - x_a) + f x_a)) → 
  ∃ m b : ℝ, y = m * x + b :=
by
  intro f f' x y hx hy hxy
  use [3, 2]
  exact sorry

end tangent_line_eq_at_minus_one_l11_11203


namespace sally_orange_balloons_l11_11582

def initial_orange_balloons : ℝ := 9.0
def found_orange_balloons : ℝ := 2.0

theorem sally_orange_balloons :
  initial_orange_balloons + found_orange_balloons = 11.0 := 
by
  sorry

end sally_orange_balloons_l11_11582


namespace starting_number_unique_l11_11467

-- Definitions based on conditions
def has_two_threes (n : ℕ) : Prop :=
  (n / 10 = 3 ∧ n % 10 = 3)

def is_starting_number (n m : ℕ) : Prop :=
  ∃ k, n + k = m ∧ k < (m - n) ∧ has_two_threes m

-- Theorem stating the proof problem
theorem starting_number_unique : ∃ n, is_starting_number n 30 ∧ n = 32 := 
sorry

end starting_number_unique_l11_11467


namespace base_8_to_base_7_add_l11_11333

theorem base_8_to_base_7_add (a : ℕ) (b : ℕ) : 
  let n1 := nat_of_digits 8 [1, 2, 3],
      n2 := nat_of_digits 7 [2, 5],
      n1_base7 := nat.digits 7 n1,
      sum_base10 := n1 + n2,
      sum_base7 := nat.digits 7 sum_base10
  in n1 = a ∧ n2 = b ∧ n1_base7 = [1, 4, 6] ∧ sum_base7 = [2, 6, 4] := 
begin
  sorry
end

end base_8_to_base_7_add_l11_11333


namespace vertex_angle_measure_l11_11310

-- Define the isosceles triangle and its properties
def is_isosceles_triangle (A B C : ℝ) (a b c : ℝ) :=
  (A = B ∨ B = C ∨ C = A) ∧ (a + b + c = 180)

-- Define the conditions based on the problem statement
def two_angles_sum_to_100 (x y : ℝ) := x + y = 100

-- The measure of the vertex angle
theorem vertex_angle_measure (A B C : ℝ) (a b c : ℝ) 
  (h1 : is_isosceles_triangle A B C a b c) (h2 : two_angles_sum_to_100 A B) :
  C = 20 ∨ C = 80 :=
sorry

end vertex_angle_measure_l11_11310


namespace cos_angle_BAD_eq_sqrt_two_fifths_l11_11934

theorem cos_angle_BAD_eq_sqrt_two_fifths
  (A B C D : Type*)
  (AB AC BC : ℝ)
  (h_AB : AB = 4)
  (h_AC : AC = 5)
  (h_BC : BC = 7)
  (angle_bisector_AD : D ∈ line_segment B C)
  (angle_bisector_property : is_angle_bisector A D) :
  cos (angle A B (line_segment A D)) = sqrt (2 / 5) :=
  sorry

end cos_angle_BAD_eq_sqrt_two_fifths_l11_11934


namespace sin_A_and_height_on_AB_l11_11514

theorem sin_A_and_height_on_AB 
  (A B C: ℝ)
  (h_triangle: ∀ A B C, A + B + C = π)
  (h_angle_sum: A + B = 3 * C)
  (h_sin_condition: 2 * Real.sin (A - C) = Real.sin B)
  (h_AB: AB = 5)
  (h_sqrt_two: Real.cos (π / 4) = Real.sin (π / 4) := by norm_num) :
  (Real.sin A = 3 * Real.sqrt 10 / 10) ∧ (height_on_AB = 6) :=
sorry

end sin_A_and_height_on_AB_l11_11514


namespace angle_between_vectors_pi_over_three_l11_11879

variable {V : Type*} [inner_product_space ℝ V]

def vectors_non_collinear (a b : V) : Prop :=
  a ≠ b ∧ a ≠ (0 : V) ∧ b ≠ (0 : V)

def vectors_equal_norm (a b : V) : Prop :=
  ∥a∥ = ∥b∥

def vectors_perpendicular (a b : V) : Prop :=
  inner a b = 0

theorem angle_between_vectors_pi_over_three
  (a b : V)
  (h_non_collinear : vectors_non_collinear a b)
  (h_equal_norm : vectors_equal_norm a b)
  (h_perpendicular : vectors_perpendicular a (a - 2 • b))
  : real.angle (inner a b / (∥a∥ * ∥b∥)) = real.angle (1 / 2) :=
  sorry

end angle_between_vectors_pi_over_three_l11_11879


namespace sqrt_exists_iff_nonneg_l11_11637

theorem sqrt_exists_iff_nonneg {x : ℝ} : (∃ y, y = sqrt (x + 2)) ↔ (x + 2 ≥ 0) := sorry

end sqrt_exists_iff_nonneg_l11_11637


namespace number_of_boys_is_60_l11_11675

-- Definitions based on conditions
def total_students : ℕ := 150

def number_of_boys (x : ℕ) : Prop :=
  ∃ g : ℕ, x + g = total_students ∧ g = (x * total_students) / 100

-- Theorem statement
theorem number_of_boys_is_60 : number_of_boys 60 := 
sorry

end number_of_boys_is_60_l11_11675


namespace f_characterization_l11_11769

noncomputable def op (a b : ℝ) := a * b

noncomputable def ot (a b : ℝ) := a + b

noncomputable def f (x : ℝ) := ot x 2 - op 2 x

-- Prove that f(x) is neither odd nor even and is a decreasing function
theorem f_characterization :
  (∀ x : ℝ, f x = -x + 2) ∧
  (∀ x : ℝ, f (-x) ≠ f x ∧ f (-x) ≠ -f x) ∧
  (∀ x y : ℝ, x < y → f x > f y) := sorry

end f_characterization_l11_11769


namespace boys_to_girls_ratio_l11_11913

theorem boys_to_girls_ratio (boys girls : ℕ) (h_boys : boys = 1500) (h_girls : girls = 1200) : 
  (boys / Nat.gcd boys girls) = 5 ∧ (girls / Nat.gcd boys girls) = 4 := 
by 
  sorry

end boys_to_girls_ratio_l11_11913


namespace max_value_is_sqrt5_l11_11396

noncomputable def max_value_of_f_add_f_prime 
  (ω : ℕ) 
  (h₀ : ω > 0) 
  (h₁ : ∀ x ∈ Ioo 0 (ω * π), (sin (ω * x + ω)) = 0 → (q x) ∈ {1, 2, 3, 4}) 
  : ℝ :=
√5

theorem max_value_is_sqrt5 
  {ω : ℕ}
  (h₀ : ω > 0)
  (h₁ : ∀ x ∈ Ioo 0 (ω * π), (sin (ω * x + ω)) = 0 → (q x) ∈ {1, 2, 3, 4})
  : 
  (∃ x ∈ Ioo 0 (ω * π), (f₁ : ℝ) (λ x, sin (ω * x + ω)) * (f₂ : ℝ) (λ x, 2 * cos (ω * x + ω)) = √5) 
:= sorry

end max_value_is_sqrt5_l11_11396


namespace multiply_binomials_l11_11570

theorem multiply_binomials (x : ℝ) : (4 * x + 3) * (2 * x - 7) = 8 * x^2 - 22 * x - 21 :=
by 
  -- Proof is to be filled here
  sorry

end multiply_binomials_l11_11570


namespace replace_integers_maintain_mean_variance_l11_11117

-- Define the initial set
def initial_set : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}

-- Define a replacement function that asserts b and c replace a in the set without changing mean and variance
theorem replace_integers_maintain_mean_variance 
(a : ℤ) (b c : ℤ) 
(h1 : a ∈ initial_set) 
(h2 : initial_set.sum = 0) 
(h3 : b + c = a) 
(h4 : a^2 + 10 = b^2 + c^2): 
b ∈ initial_set ∧ c ∈ initial_set := 
sorry

end replace_integers_maintain_mean_variance_l11_11117


namespace books_bought_at_yard_sale_l11_11567

def number_of_books_bought (before after : ℕ) : ℕ := after - before

theorem books_bought_at_yard_sale :
  ∀ (before after : ℕ), before = 35 → after = 56 → number_of_books_bought before after = 21 := 
by intros before after h_before h_after
   unfold number_of_books_bought
   rw [h_before, h_after]
   sorry

end books_bought_at_yard_sale_l11_11567


namespace sum_of_altitudes_l11_11212

open Real

theorem sum_of_altitudes : 
  let line_equation (x y : ℝ) := 10 * x + 4 * y = 40 in
  let sum_altitudes (x y : ℝ) := 
    let x_intercept : ℝ := 40 / 10 in
    let y_intercept : ℝ := 40 / 4 in
    let base1 : ℝ := x_intercept in
    let base2 : ℝ := y_intercept in
    let height1 : ℝ := x_intercept in
    let height2 : ℝ := y_intercept in
    let area : ℝ := (1 / 2) * base1 * base2 in
    let third_height : ℝ := (40 / (sqrt (10^2 + 4^2))) in
    base1 + base2 + third_height in
  sum_altitudes 4 10 = 194 / 11 := by
  sorry

end sum_of_altitudes_l11_11212


namespace new_ratio_of_dogs_to_cats_l11_11088

theorem new_ratio_of_dogs_to_cats (C0 : ℕ) (C_new : ℕ) (C_added : ℕ) (D : ℕ) (d_ratio : ℕ) (c_ratio : ℕ) :
  d_ratio = 15 → c_ratio = 7 → D = 75 → C_new = C0 + C_added → C_added = 20 → C0 = D * c_ratio / d_ratio →
  (d_ratio / Nat.gcd d_ratio (C0 + C_added)) = 15 ∧ (c_ratio / Nat.gcd d_ratio (C0 + C_added)) = 11 :=
by
  intros h1 h2 h3 h4 h5 h6
  have h7 : Nat.gcd 75 55 = 5 := sorry
  have h8 : 75 / 5 = 15 := sorry
  have h9 : 55 / 5 = 11 := sorry
  exact ⟨h8, h9⟩

sorry

end new_ratio_of_dogs_to_cats_l11_11088


namespace two_digit_number_l11_11461

theorem two_digit_number (x y : ℕ) (h1 : y = x + 4) (h2 : (10 * x + y) * (x + y) = 208) :
  10 * x + y = 26 :=
sorry

end two_digit_number_l11_11461


namespace triangle_sin_A_and_height_l11_11483

noncomputable theory

variables (A B C : ℝ) (AB : ℝ)
  (h1 : A + B = 3 * C)
  (h2 : 2 * Real.sin (A - C) = Real.sin B)
  (h3 : AB = 5)

theorem triangle_sin_A_and_height :
  Real.sin A = 3 * Real.cos A → 
  sqrt 10 / 10 * Real.sin A = 3 / sqrt (10) / 3 → 
  √10 / 10 = 3/ sqrt 10 /3 → 
  sin (A+B) =sin /sqrt10 →
  (sin (A cv)+ C) = sin( AC ) → 
  ( cos A = sinA 3) 
  ( (10 +25)+5→1= well 5 → B (PS6 S)=H1 (A3+.B9)=
 
 
   
∧   (γ = hA → ) ( (/. );



∧ side /4→ABh3 → 5=HS)  ( →AB3)=sinh1S  

then 
(
  (Real.sin A = 3 * Real.cos A) ^2 )+   
  
(Real.cos A= √ 10/10
  
  Real.sin A2 C(B)= 3√10/10
  
 ) ^(Real.sin A = 5

6)=    
    sorry

end triangle_sin_A_and_height_l11_11483


namespace number_of_coaches_l11_11746

-- Define the initial speed of the engine without coaches
def initial_speed : ℝ := 30

-- Define the direct proportionality constant (k) after evaluating given conditions
def proportionality_constant (n1 : ℕ) (v1 : ℝ) : ℝ := 
  let k := (initial_speed - v1) / (Real.sqrt n1)
  k

-- Define the speed of the train given a certain number of coaches
def train_speed (n : ℕ) (k : ℝ) : ℝ := 
  initial_speed - k * (Real.sqrt n)

-- Theorem to prove that 16 coaches result in a speed of 14 kmph
theorem number_of_coaches :
  ∀ (v : ℝ) (k : ℝ), v = 14 → k = proportionality_constant 9 18 → train_speed 16 k = v := by
  intros v k h₁ h₂
  sorry

end number_of_coaches_l11_11746


namespace leah_practice_minutes_l11_11535

theorem leah_practice_minutes :
  let mins_in_hour := 60
  let practice_day1 := 1 * mins_in_hour + 40 -- 1 hr 40 min in minutes
  let practice_day2 := 1 * mins_in_hour + 20 -- 1 hr 20 min in minutes
  ∀ (avg_minutes_per_day : ℕ) (days_total : ℕ) (days1 : ℕ) (days2 : ℕ) (day9_minutes : ℕ),
    avg_minutes_per_day = 100 ->
    days_total = 9 ->
    days1 = 6 ->
    days2 = 2 ->
    let total_minutes_needed := avg_minutes_per_day * days_total
    let total_minutes_8_days := practice_day1 * days1 + practice_day2 * days2
    total_minutes_needed - total_minutes_8_days = day9_minutes ->
    day9_minutes = 140 :=
by
  intros mins_in_hour practice_day1 practice_day2 avg_minutes_per_day days_total days1 days2 day9_minutes
  intros avg_eq days_total_eq days1_eq days2_eq h
  rw [avg_eq, days_total_eq, days1_eq, days2_eq]
  have practice_day1 : practice_day1 = 100 := rfl
  have practice_day2 : practice_day2 = 80 := rfl
  rw [practice_day1, practice_day2] at h
  have total_minutes_needed : total_minutes_needed = avg_minutes_per_day * days_total := rfl
  have total_minutes_8_days : total_minutes_8_days = practice_day1 * days1 + practice_day2 * days2 := rfl
  rw [total_minutes_needed, total_minutes_8_days, mul_comm 9, mul_comm 6, mul_comm 2] at h
  have total_minutes_needed_eq : 100 * 9 = 900 := rfl
  have total_minutes_8_days_eq : 100 * 6 + 80 * 2 = 760 := rfl
  rw [total_minutes_needed_eq, total_minutes_8_days_eq] at h
  have day9_minutes_eq : 900 - 760 = 140 := rfl
  rw day9_minutes_eq
  exact h

end leah_practice_minutes_l11_11535


namespace floor_sqrt_50_squared_l11_11808

theorem floor_sqrt_50_squared : ∃ x : ℕ, x = 49 ∧ (⌊real.sqrt 50⌋ : ℕ) ^ 2 = x := by
  have h1 : (7 : ℝ) < real.sqrt 50 := sorry
  have h2 : real.sqrt 50 < 8 := sorry
  have h_floor : (⌊real.sqrt 50⌋ : ℕ) = 7 := sorry
  use 49
  constructor
  · rfl
  · rw [h_floor]
    norm_num
    sorry

end floor_sqrt_50_squared_l11_11808


namespace floor_sqrt_50_squared_l11_11799

theorem floor_sqrt_50_squared :
  (let x := Real.sqrt 50 in (⌊x⌋₊ : ℕ)^2 = 49) :=
by
  sorry

end floor_sqrt_50_squared_l11_11799


namespace negation_example_l11_11659

universe u

variable {ℚ : Type u} [linear_ordered_field ℚ]

theorem negation_example (h : ∀ x : ℚ, 2 * x + 1 > 0) : ∃ x : ℚ, 2 * x + 1 ≤ 0 ↔ false :=
begin
  -- sorry to skip the proof
  sorry
end

end negation_example_l11_11659


namespace floor_sqrt_50_squared_l11_11809

theorem floor_sqrt_50_squared : ∃ x : ℕ, x = 49 ∧ (⌊real.sqrt 50⌋ : ℕ) ^ 2 = x := by
  have h1 : (7 : ℝ) < real.sqrt 50 := sorry
  have h2 : real.sqrt 50 < 8 := sorry
  have h_floor : (⌊real.sqrt 50⌋ : ℕ) = 7 := sorry
  use 49
  constructor
  · rfl
  · rw [h_floor]
    norm_num
    sorry

end floor_sqrt_50_squared_l11_11809


namespace exists_even_among_pythagorean_triplet_l11_11894

theorem exists_even_among_pythagorean_triplet (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  ∃ x, (x = a ∨ x = b ∨ x = c) ∧ x % 2 = 0 :=
sorry

end exists_even_among_pythagorean_triplet_l11_11894


namespace university_diploma_percentage_l11_11096

-- Define the conditions
variables (P N JD ND : ℝ)
-- P: total population assumed as 100% for simplicity
-- N: percentage of people with university diploma
-- JD: percentage of people who have the job of their choice
-- ND: percentage of people who do not have a university diploma but have the job of their choice
variables (A : ℝ) -- A: University diploma percentage of those who do not have the job of their choice
variable (total_diploma : ℝ)
axiom country_Z_conditions : 
  (P = 100) ∧ (ND = 18) ∧ (JD = 40) ∧ (A = 25)

-- Define the proof problem
theorem university_diploma_percentage :
  (N = ND + (JD - ND) + (total_diploma * (P - JD * (P / JD) / P))) →
  N = 37 :=
by
  sorry

end university_diploma_percentage_l11_11096


namespace count_even_three_digit_numbers_sum_tens_units_eq_12_l11_11005

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def is_even (n : ℕ) : Prop := n % 2 = 0
def sum_of_tens_and_units_eq_12 (n : ℕ) : Prop :=
  (n / 10) % 10 + n % 10 = 12

theorem count_even_three_digit_numbers_sum_tens_units_eq_12 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_three_digit n ∧ is_even n ∧ sum_of_tens_and_units_eq_12 n) ∧ S.card = 24 :=
sorry

end count_even_three_digit_numbers_sum_tens_units_eq_12_l11_11005


namespace positive_diff_between_two_numbers_l11_11623

theorem positive_diff_between_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 20) :  |x - y| = 2 := 
by
  sorry

end positive_diff_between_two_numbers_l11_11623


namespace area_of_circle_segment_below_line_l11_11644

open Real

theorem area_of_circle_segment_below_line :
  let circle_eq := λ x y : ℝ, (x - 5)^2 + (y - 3)^2 = 9
  let line_eq := λ x y : ℝ, y = x - 1
  ∃ (area : ℝ), abs (area - 3 * π) < 0.01 ∧
    ( ∀ x y, (circle_eq x y ∧ x < y + 1) → x ≥ 0 ∧ y ≥ 0) :=
sorry

end area_of_circle_segment_below_line_l11_11644


namespace area_of_pentagon_ABCDE_l11_11917

open EuclideanGeometry

variables {A B C D E F G M : Point}

-- Given conditions
def conditions :=
  AB = 15 ∧ BC = 16 ∧ CD = 12 ∧ DA = 25 ∧ BD = 20 ∧
  is_circumcenter M A B D ∧
  is_circumcircle γ A B D ∧
  (CB meets γ again at F) ∧
  (AF meets MC at G) ∧
  (GD meets γ again at E)

-- Theorem to prove
theorem area_of_pentagon_ABCDE
  (h : conditions) :
  area_of_pentagon A B C D E = 396 :=
sorry

end area_of_pentagon_ABCDE_l11_11917


namespace range_of_a_l11_11903

open Real

theorem range_of_a {a : ℝ} :
  (∃ x : ℝ, sqrt (3 * x + 6) + sqrt (14 - x) > a) → a < 8 :=
by
  intro h
  sorry

end range_of_a_l11_11903


namespace triangle_sin_A_and_height_l11_11485

noncomputable theory

variables (A B C : ℝ) (AB : ℝ)
  (h1 : A + B = 3 * C)
  (h2 : 2 * Real.sin (A - C) = Real.sin B)
  (h3 : AB = 5)

theorem triangle_sin_A_and_height :
  Real.sin A = 3 * Real.cos A → 
  sqrt 10 / 10 * Real.sin A = 3 / sqrt (10) / 3 → 
  √10 / 10 = 3/ sqrt 10 /3 → 
  sin (A+B) =sin /sqrt10 →
  (sin (A cv)+ C) = sin( AC ) → 
  ( cos A = sinA 3) 
  ( (10 +25)+5→1= well 5 → B (PS6 S)=H1 (A3+.B9)=
 
 
   
∧   (γ = hA → ) ( (/. );



∧ side /4→ABh3 → 5=HS)  ( →AB3)=sinh1S  

then 
(
  (Real.sin A = 3 * Real.cos A) ^2 )+   
  
(Real.cos A= √ 10/10
  
  Real.sin A2 C(B)= 3√10/10
  
 ) ^(Real.sin A = 5

6)=    
    sorry

end triangle_sin_A_and_height_l11_11485


namespace probability_of_one_or_two_in_pascal_l11_11728

def pascal_triangle_element_probability : ℚ :=
  let total_elements := 210 -- sum of the elements in the first 20 rows
  let ones_count := 39      -- total count of 1s in the first 20 rows
  let twos_count := 36      -- total count of 2s in the first 20 rows
  let favorable_elements := ones_count + twos_count
  favorable_elements / total_elements

theorem probability_of_one_or_two_in_pascal (n : ℕ) (h : n = 20) :
  pascal_triangle_element_probability = 5 / 14 := by
  rw [h]
  dsimp [pascal_triangle_element_probability]
  sorry

end probability_of_one_or_two_in_pascal_l11_11728


namespace αdotβ_value_l11_11831

noncomputable def αdotβ (α β : ℝˣ) : ℝ := α.val.dot β.val / β.val.dot β.val
notation "○" => αdotβ

theorem αdotβ_value (a b : ℝˣ) (θ : ℝ) (h1 : |a| ≥ |b|) (h2 : |b| > 0)
  (h3 : θ > 0 ∧ θ < (π / 4))
  (h4a : (a ○ b) ∈ {n / 2 | n : ℤ})
  (h4b : (b ○ a) ∈ {m / 2 | m : ℤ}) :
  a ○ b = 3 / 2 :=

sorry

end αdotβ_value_l11_11831


namespace jade_driving_hours_per_day_l11_11123

variable (Jade Krista : ℕ)
variable (days driving_hours total_hours : ℕ)

theorem jade_driving_hours_per_day :
  (days = 3) →
  (Krista = 6) →
  (total_hours = 42) →
  (total_hours = days * Jade + days * Krista) →
  Jade = 8 :=
by
  intros h_days h_krista h_total_hours h_equation
  sorry

end jade_driving_hours_per_day_l11_11123


namespace bobs_share_l11_11167

theorem bobs_share 
  (r : ℕ → ℕ → ℕ → Prop) (s : ℕ) 
  (h_ratio : r 1 2 3) 
  (bill_share : s = 300) 
  (hr : ∃ p, s = 2 * p) :
  ∃ b, b = 3 * (s / 2) ∧ b = 450 := 
by
  sorry

end bobs_share_l11_11167


namespace largest_of_A_B_C_l11_11256

noncomputable def A : ℝ := (2010 / 2009) + (2010 / 2011)
noncomputable def B : ℝ := (2010 / 2011) + (2012 / 2011)
noncomputable def C : ℝ := (2011 / 2010) + (2011 / 2012)

theorem largest_of_A_B_C : B > A ∧ B > C := by
  sorry

end largest_of_A_B_C_l11_11256


namespace problem_solution_l11_11547

noncomputable def g (n : ℕ) : ℕ :=
  (∏ d in (n.divisors.filter (λ d, 1 < d ∧ d < n)).to_finset, d)

def is_valid_n (n : ℕ) : Prop :=
  2 ≤ n ∧ n ≤ 100 ∧ ¬(n ∣ g n)

def count_valid_n : ℕ :=
  (finset.range 101).filter is_valid_n).card

theorem problem_solution : count_valid_n = 32 := sorry

end problem_solution_l11_11547


namespace sin_A_eq_height_on_AB_l11_11521

-- Defining conditions
variables {A B C : ℝ}
variables (AB : ℝ)

-- Conditions based on given problem
def condition1 : Prop := A + B = 3 * C
def condition2 : Prop := 2 * sin (A - C) = sin B
def condition3 : Prop := A + B + C = Real.pi

-- Question 1: prove that sin A = (3 * sqrt 10) / 10
theorem sin_A_eq:
  condition1 → 
  condition2 → 
  condition3 → 
  sin A = (3 * Real.sqrt 10) / 10 :=
by
  sorry

-- Question 2: given AB = 5, prove the height on side AB is 6
theorem height_on_AB:
  condition1 →
  condition2 →
  condition3 →
  AB = 5 →
  -- Let's construct the height as a function of A, B, and C
  ∃ h, h = 6 :=
by
  sorry

end sin_A_eq_height_on_AB_l11_11521


namespace find_value_of_a_l11_11601

theorem find_value_of_a 
  (a : ℝ) 
  (h1 : ∀ x ∈ set.Icc 1 2, f x = a^x) 
  (h2 : ∀ a x, 0 < a ∧ a ≠ 1 → continuous (λ x, a^x))
  (h3 : (∀ a > 1, ∃ x ∈ set.Icc 1 2, a^2 - a = (a/2)) 
      ∧ (∀ a < 1 ∧ a > 0, ∃ x ∈ set.Icc 1 2, a - a^2 = (a/2))) :
  a = 1/2 ∨ a = 3/2 :=
sorry

end find_value_of_a_l11_11601


namespace find_f_5_l11_11208

noncomputable def f : ℝ → ℝ := sorry

axiom additivity : ∀ x y : ℝ, f (x + y) = f x + f y
axiom initial_condition : f 3 = 6

theorem find_f_5 : f 5 = 10 :=
by
  have h1 : f 1 = 2 := sorry
  have h2 : f 2 = 4 := sorry
  have h3 : f 5 = f 3 + f 2 :=
    by
      sorry
  rw [h1, h2] at h3
  exact h3

end find_f_5_l11_11208


namespace fibonacci_exceeds_3000_l11_11205

def fibonacci : ℕ → ℕ 
| 0     := 0
| 1     := 1
| (n+2) := fibonacci n + fibonacci (n+1)

theorem fibonacci_exceeds_3000 :
  ∃ n, fibonacci n > 3000 ∧ ∀ m < n, fibonacci m ≤ 3000 :=
  by
  existsi 19
  split
  · -- fibonacci 19 > 3000
    exact Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (Nat.succ_lt_succ (by norm_num)))))))))))))))))
  · -- ∀ m < 19, fibonacci m ≤ 3000
    intro m hm
    interval_cases m
    · norm_num
    · exact le_refl (fibonacci 18)
    · exact le_refl (fibonacci 17)
    sorry

end fibonacci_exceeds_3000_l11_11205


namespace pascal_element_probability_l11_11739

open Nat

def num_elems_first_n_rows (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

def count_ones (n : ℕ) : ℕ :=
  if n = 0 then 1 else if n = 1 then 2 else 2 * (n - 1) + 1

def count_twos (n : ℕ) : ℕ :=
  if n < 2 then 0 else 2 * (n - 2)

def probability_one_or_two (n : ℕ) : ℚ :=
  let total_elems := num_elems_first_n_rows n in
  let ones := count_ones n in
  let twos := count_twos n in
  (ones + twos) / total_elems

theorem pascal_element_probability :
  probability_one_or_two 20 = 5 / 14 :=
by
  sorry

end pascal_element_probability_l11_11739


namespace floor_sqrt_50_squared_l11_11800

theorem floor_sqrt_50_squared :
  (let x := Real.sqrt 50 in (⌊x⌋₊ : ℕ)^2 = 49) :=
by
  sorry

end floor_sqrt_50_squared_l11_11800


namespace sqrt_floor_squared_l11_11811

/-- To evaluate the floor of the square root of 50 squared --/
theorem sqrt_floor_squared :
  (⌊real.sqrt 50⌋ : ℕ)^2 = 49 :=
begin
  -- We know that:
  -- 7^2 = 49 < 50 < 64 = 8^2
  have h1 : 7^2 = 49, by linarith,
  have h2 : 64 = 8^2, by linarith,
  have h3 : (7 : real) < real.sqrt 50, by {
    rw [sqrt_lt],
    exact_mod_cast h1,
  },
  have h4 : real.sqrt 50 < 8, by {
    rw [lt_sqrt],
    exact_mod_cast h2,
  },
  -- Therefore, 7 < sqrt(50) < 8.
  have h5 : (⌊real.sqrt 50⌋ : ℕ) = 7, by {
    rw [nat.floor_eq_iff],
    split,
    { exact_mod_cast h3, },
    { exact_mod_cast h4, },
  },
  -- Thus, ⌊sqrt(50)⌋^2 = 7^2 = 49.
  rw h5,
  exact h1,
end

end sqrt_floor_squared_l11_11811


namespace arthur_initial_amount_l11_11312

def initial_amount (X : ℝ) : Prop :=
  (1/5) * X = 40

theorem arthur_initial_amount (X : ℝ) (h : initial_amount X) : X = 200 :=
by
  sorry

end arthur_initial_amount_l11_11312


namespace isosceles_triangle_product_constant_l11_11946

noncomputable def isosceles_triangle (A B C H : Type)
  (h_orthocenter : H = orthocenter(A, B, C))
  (congr: AB = AC) : Prop :=
  S(ABC) * S(HBC) = const

theorem isosceles_triangle_product_constant 
  (A B C H : Type)
  [orthocenter : H = orthocenter(A, B, C)]
  [isos_means : AB = AC]
  [base : BC = const] :
  S(ABC) * S(HBC) = const :=
sorry

end isosceles_triangle_product_constant_l11_11946


namespace sqrt_50_floor_square_l11_11797

theorem sqrt_50_floor_square : ⌊Real.sqrt 50⌋ ^ 2 = 49 := by
  have h : 7 < Real.sqrt 50 ∧ Real.sqrt 50 < 8 := 
    by sorry
  have floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := 
    by sorry
  show ⌊Real.sqrt 50⌋ ^ 2 = 49
  from calc
    ⌊Real.sqrt 50⌋ ^ 2 = 7 ^ 2 : by rw [floor_sqrt_50]
    ... = 49 : by norm_num

end sqrt_50_floor_square_l11_11797


namespace count_even_three_digit_sum_tens_units_is_12_l11_11056

-- Define what it means to be a three-digit number
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n < 1000)

-- Define what it means to be even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the sum of the tens and units digits to be 12
def sum_tens_units_is_12 (n : ℕ) : Prop := 
  let tens := (n / 10) % 10 in
  let units := n % 10 in
  tens + units = 12

-- Count how many such numbers exist
theorem count_even_three_digit_sum_tens_units_is_12 : 
  ∃! n : ℕ, (is_three_digit n) ∧ (is_even n) ∧ (sum_tens_units_is_12 n) = 36 :=
sorry

end count_even_three_digit_sum_tens_units_is_12_l11_11056


namespace line_MN_through_fixed_point_R_l11_11572

structure Point (α : Type) := (x y : α)
structure LineSegment (α : Type) := (A B : Point α)
structure Square (α : Type) := (A B C D : Point α)
structure Circle (α : Type) := (center : Point α) (radius : α)

variables {α : Type} [LinearOrderedField α]

-- Conditions
variable (AB : LineSegment α)
variable (M : Point α)
variable (AMCD MBFE : Square α) -- Squares AMCD and MBFE
variable (P Q : Circle α)  -- Circumcircles of squares AMCD and MBFE
variable (N : Point α)
variable (R : Point α) -- Fixed point R

-- Definitions of the problem conditions
axiom AM_on_AB : M ∈ AB.A ∧ M ∈ AB.B
axiom square_AMCD_on_side : ∃ (C D : Point α), AMCD = Square.mk M AB.B D C -- same side as AB
axiom square_MBFE_on_side : ∃ (E F : Point α), MBFE = Square.mk AB.A M E F -- same side as AB
axiom circumcircle_AMCD : P = Circle.mk (point at the center of square AMCD) (radius of square AMCD)
axiom circumcircle_MBFE : Q = Circle.mk (point at the center of square MBFE) (radius of square MBFE)
axiom intersection_PQ : P ≠ Q ∧ (N = intersection point of circles P and Q other than M)

-- Mathematical proof problem
theorem line_MN_through_fixed_point_R 
    (AB : LineSegment α)
    (AMCD MBFE : Square α)
    (P Q : Circle α)
    (M N R : Point α)
    (AM_on_AB : ∃ (a b : Point α), AB = LineSegment.mk a b ∧ M ∈ AB)
    (square_AMCD_on_side : ∃ (C D : Point α), AMCD = Square.mk M (other_point_in_AB_segment AB) C D)
    (square_MBFE_on_side : ∃ (E F : Point α), MBFE = Square.mk (other_point_in_AB_segment AB) M E F)
    (circumcircle_AM : P = Circle.mk (center AMCD) (radius AMCD))
    (circumcircle_MB : Q = Circle.mk (center MBFE) (radius MBFE))
    (intersection_PQ : P ≠ Q ∧ (N = intersection_point P Q other_M))
    : (line_through MN passes_through R) := sorry

end line_MN_through_fixed_point_R_l11_11572


namespace hours_per_toy_l11_11713

-- Defining the conditions
def toys_produced (hours: ℕ) : ℕ := 40 
def hours_worked : ℕ := 80

-- Theorem: If a worker makes 40 toys in 80 hours, then it takes 2 hours to make one toy.
theorem hours_per_toy : (hours_worked / toys_produced hours_worked) = 2 :=
by
  sorry

end hours_per_toy_l11_11713


namespace sqrt_50_floor_square_l11_11793

theorem sqrt_50_floor_square : ⌊Real.sqrt 50⌋ ^ 2 = 49 := by
  have h : 7 < Real.sqrt 50 ∧ Real.sqrt 50 < 8 := 
    by sorry
  have floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := 
    by sorry
  show ⌊Real.sqrt 50⌋ ^ 2 = 49
  from calc
    ⌊Real.sqrt 50⌋ ^ 2 = 7 ^ 2 : by rw [floor_sqrt_50]
    ... = 49 : by norm_num

end sqrt_50_floor_square_l11_11793


namespace proof_triangle_properties_l11_11492

variable (A B C : ℝ)
variable (h AB : ℝ)

-- Conditions
def triangle_conditions : Prop :=
  (A + B = 3 * C) ∧ (2 * Real.sin (A - C) = Real.sin B) ∧ (AB = 5)

-- Part 1: Proving sin A
def find_sin_A (h₁ : triangle_conditions A B C h AB) : Prop :=
  Real.sin A = 3 * Real.cos A

-- Part 2: Proving the height on side AB
def find_height_on_AB (h₁ : triangle_conditions A B C h AB) : Prop :=
  h = 6

-- Combined proof statement
theorem proof_triangle_properties (h₁ : triangle_conditions A B C h AB) : 
  find_sin_A A B C h₁ ∧ find_height_on_AB A B C h AB h₁ := 
  by sorry

end proof_triangle_properties_l11_11492


namespace number_of_levels_l11_11761

-- Definitions of the conditions
def blocks_per_step : ℕ := 3
def steps_per_level : ℕ := 8
def total_blocks_climbed : ℕ := 96

-- The theorem to prove
theorem number_of_levels : (total_blocks_climbed / blocks_per_step) / steps_per_level = 4 := by
  sorry

end number_of_levels_l11_11761


namespace parallelogram_area_equality_l11_11573

variables {A B C D E F G H L M N : Point}

-- Definitions for points and line parallelisms/parallelograms areas
def is_parallelogram (p1 p2 p3 p4 : Point) : Prop :=
  parallel p1 p2 p3 p4 ∧ equal (distance p1 p2) (distance p3 p4) ∧ equal (distance p2 p3) (distance p4 p1)

def area_parallelogram (p1 p2 p3 p4 : Point) : ℝ :=
  -- The area function is assumed to be defined elsewhere
  sorry

variables (ABC : Triangle A B C)

-- Given conditions
axiom h1 : is_parallelogram A C D E
axiom h2 : is_parallelogram B C F G
axiom h3 : intersect (extend D E) (extend F G) = H
axiom h4 : is_parallelogram A B M L
axiom h5 : parallel (line A L) (line H C)
axiom h6 : equal (distance A L) (distance H C)
axiom h7 : parallel (line B M) (line H C)
axiom h8 : equal (distance B M) (distance H C)

-- To prove
theorem parallelogram_area_equality :
  area_parallelogram A B M L = area_parallelogram A C D E + area_parallelogram B C F G :=
sorry

end parallelogram_area_equality_l11_11573


namespace food_ratio_l11_11229

-- Define basic quantities and conditions
def number_of_puppies : ℕ := 4
def number_of_dogs : ℕ := 3
def dog_meals_per_day : ℕ := 3
def puppy_meals_multiple : ℕ := 3
def dog_meal_amount : ℕ := 4 -- 4 pounds per meal
def total_food_per_day : ℕ := 108

-- Define a variable for the amount of food a puppy eats in one meal
variable (P : ℕ)

-- Define the total food intake for the dogs
def dog_food_per_day : ℕ := number_of_dogs * dog_meal_amount * dog_meals_per_day

-- Define the number of meals a puppy eats in one day
def puppy_meals_per_day := dog_meals_per_day * puppy_meals_multiple

-- Define the total food intake for the puppies
def puppy_food_per_day : ℕ := number_of_puppies * P * puppy_meals_per_day

-- Prove the ratio between dog food per meal and puppy food per meal
theorem food_ratio
  (h1 : dog_food_per_day + puppy_food_per_day = total_food_per_day):
  (dog_meal_amount : P) = 2 :=
begin
  -- equation for total food intake
  have h2 : dog_food_per_day = 36 := by sorry,
  have h3 : 36 + number_of_puppies * P * puppy_meals_per_day = total_food_per_day := by sorry,
  have h4 : number_of_puppies * P * puppy_meals_per_day = 72 := by sorry,
  have h5 : P = 2 := by sorry,
  -- Ratio calculation
  sorry
end

end food_ratio_l11_11229


namespace find_f_neg_2017_l11_11874

-- Define f as given in the problem
def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 2

-- State the given problem condition
def condition (a b : ℝ) : Prop :=
  f a b 2017 = 10

-- The main problem statement proving the solution
theorem find_f_neg_2017 (a b : ℝ) (h : condition a b) : f a b (-2017) = -14 :=
by
  -- We state this theorem and provide a sorry to skip the proof
  sorry

end find_f_neg_2017_l11_11874


namespace inequality_holds_for_all_x_l11_11893

theorem inequality_holds_for_all_x (a : ℝ) :
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ (-3 / 5 < a ∧ a ≤ 1) :=
by
  sorry

end inequality_holds_for_all_x_l11_11893


namespace muffin_is_twice_as_expensive_as_banana_l11_11184

variable (m b : ℚ)
variable (h1 : 4 * m + 10 * b = 3 * m + 5 * b + 12)
variable (h2 : 3 * m + 5 * b = S)

theorem muffin_is_twice_as_expensive_as_banana (h1 : 4 * m + 10 * b = 3 * m + 5 * b + 12) : m = 2 * b :=
by
  sorry

end muffin_is_twice_as_expensive_as_banana_l11_11184


namespace area_below_line_l11_11647

noncomputable def circle := set_of (λ p : ℝ × ℝ, (p.1 - 5)^2 + (p.2 - 7.5)^2 = 56.25)
noncomputable def line := set_of (λ p : ℝ × ℝ, p.2 = p.1 - 1)

theorem area_below_line :
  ∃ (a : ℝ), a = 56.25 * π * (3 / 4) := sorry

end area_below_line_l11_11647


namespace LouShoes_Weekly_Sales_l11_11563

theorem LouShoes_Weekly_Sales
  (total_pairs : ℕ)
  (pairs_last_week : ℕ)
  (pairs_needed : ℕ)
  (pairs_this_week : ℕ)
  (h_goal : total_pairs = 80)
  (h_last_week : pairs_last_week = 27)
  (h_needed : pairs_needed = 41) :
  pairs_this_week = total_pairs - pairs_last_week - pairs_needed :=
by
  rw [h_goal, h_last_week, h_needed]
  exact eq_refl 12

end LouShoes_Weekly_Sales_l11_11563


namespace diameter_of_circumscribed_circle_l11_11910

namespace QuadrilateralWithPerpendicularDiagonals

-- Define the structure of the quadrilateral with its properties and conditions
structure Quadrilateral (A B C D O : Type) :=
(diagonal_AC : A → C)
(diagonal_BD : B → D)
(intersection_at_O_AC_BD : diagonal_AC A = O ∧ diagonal_BD B = O)
(perpendicular_diagonals : ⊤)  -- Perpendicular diagonals condition

variables {A B C D O : Type} [Quadrilateral A B C D O]

-- The main theorem or statement
theorem diameter_of_circumscribed_circle
  (m n p q a b D : ℝ)
  (h1 : m^2 + p^2 = a^2)
  (h2 : q^2 + n^2 = b^2)
  (h3 : m^2 + n^2 + p^2 + q^2 = D^2) :
  D = √(a^2 + b^2) :=
sorry

end QuadrilateralWithPerpendicularDiagonals

end diameter_of_circumscribed_circle_l11_11910


namespace ranking_l11_11432

variables (score : string → ℝ)
variables (Hannah Cassie Bridget David : string)

-- Conditions based on the problem statement
axiom Hannah_shows_her_test_to_everyone : ∀ x, x ≠ Hannah → x = Cassie ∨ x = Bridget ∨ x = David
axiom David_shows_his_test_only_to_Bridget : ∀ x, x ≠ Bridget → x ≠ David
axiom Cassie_does_not_show_her_test : ∀ x, x = Hannah ∨ x = Bridget ∨ x = David → x ≠ Cassie

-- Statements based on what Cassie and Bridget claim
axiom Cassie_statement : score Cassie > min (score Hannah) (score Bridget)
axiom Bridget_statement : score David > score Bridget

-- Final ranking to be proved
theorem ranking : score David > score Bridget ∧ score Bridget > score Cassie ∧ score Cassie > score Hannah := sorry

end ranking_l11_11432


namespace floor_sqrt_50_squared_l11_11786

theorem floor_sqrt_50_squared :
  ∃ x : ℕ, x = 7 ∧ ⌊ Real.sqrt 50 ⌋ = x ∧ x^2 = 49 := 
by {
  let x := 7,
  use x,
  have h₁ : 7 < Real.sqrt 50, from sorry,
  have h₂ : Real.sqrt 50 < 8, from sorry,
  have floor_eq : ⌊Real.sqrt 50⌋ = 7, from sorry,
  split,
  { refl },
  { split,
    { exact floor_eq },
    { exact rfl } }
}

end floor_sqrt_50_squared_l11_11786


namespace relationship_among_a_b_c_l11_11399

noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.6
noncomputable def b : ℝ := 0.3^0.6
noncomputable def c : ℝ := 0.6^0.3

theorem relationship_among_a_b_c : a > c ∧ c > b := by
  have h1 : a = Real.log 0.3 / Real.log 0.6 := rfl
  have h2 : b = 0.3^0.6 := rfl
  have h3 : c = 0.6^0.3 := rfl
  sorry

end relationship_among_a_b_c_l11_11399


namespace cyclic_quadrilateral_angles_l11_11116

theorem cyclic_quadrilateral_angles (ABCD_cyclic : True) (P_interior : True)
  (x y z t : ℝ) (h1 : x + y + z + t = 360)
  (h2 : x + t = 180) :
  x = 180 - y - z :=
by
  sorry

end cyclic_quadrilateral_angles_l11_11116


namespace pascal_triangle_probability_l11_11733

-- Define the probability problem in Lean 4
theorem pascal_triangle_probability :
  let total_elements := ((20 * (20 + 1)) / 2)
  let ones_count := (1 + 2 * 19)
  let twos_count := (2 * (19 - 2 + 1))
  (ones_count + twos_count) / total_elements = 5 / 14 :=
by
  let total_elements := ((20 * (20 + 1)) / 2)
  let ones_count := (1 + 2 * 19)
  let twos_count := (2 * (19 - 2 + 1))
  have h1 : total_elements = 210 := by sorry
  have h2 : ones_count = 39 := by sorry
  have h3 : twos_count = 36 := by sorry
  have h4 : (39 + 36) / 210 = 5 / 14 := by sorry
  exact h4

end pascal_triangle_probability_l11_11733


namespace average_annual_population_increase_l11_11931

/-- In the nation of North Southland, it is calculated that:
 - A birth occurs every 6 hours.
 - A death occurs every 2 days.
 
 We need to prove that the average annual increase in population, considering both regular and leap years, rounded to the nearest fifty, is 1300. 
 -/
theorem average_annual_population_increase :
  let births_per_day := 24 / 6,
      deaths_per_day := 1 / 2,
      net_increase_per_day := births_per_day - deaths_per_day,
      average_days_per_year := (365 + 366) / 2,
      annual_population_increase := net_increase_per_day * average_days_per_year
  in round_nearest_50 annual_population_increase = 1300 :=
by
  let births_per_day := 24 / 6
  let deaths_per_day := 1 / 2
  let net_increase_per_day := births_per_day - deaths_per_day
  let average_days_per_year := (365 + 366) / 2
  let annual_population_increase := net_increase_per_day * average_days_per_year
  have h1 : round_nearest_50 annual_population_increase = 1300 := sorry
  exact h1

/-- Function to round a number to the nearest fifty. -/
def round_nearest_50 (n : ℕ) : ℕ :=
  let rem := n % 50
  if rem < 25 then n - rem else n + (50 - rem)

end average_annual_population_increase_l11_11931


namespace count_valid_even_numbers_with_sum_12_l11_11024

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n % 2 = 0) ∧ 
  ((n / 10) % 10 + n % 10 = 12)

theorem count_valid_even_numbers_with_sum_12 :
  (finset.range 1000).filter is_valid_number).card = 27 := by
  sorry

end count_valid_even_numbers_with_sum_12_l11_11024


namespace correct_statements_count_l11_11307

def statement1_correct : Prop :=
  ∀ (θ : ℝ), 90 < θ ∧ θ < 180 → 0 < θ - 90 ∧ θ - 90 < 90

def statement2_correct : Prop :=
  ∀ (θ φ : ℝ), 90 < θ ∧ θ < 180 ∧ 0 < φ ∧ φ < 90 → ¬(90 < θ - φ ∧ θ - φ < 180)

def statement3_correct : Prop :=
  ∀ (a b c : ℝ), a + b + c = 180 ∧ 90 < a ∧ a < 180 → 0 < b ∧ b < 90 ∧ 0 < c ∧ c < 90

def statement4_correct : Prop :=
  ∀ (a b c : ℝ), a + b + c = 180 ∧ 90 ≤ a → 0 < b ∧ b < 90 ∧ 0 < c ∧ c < 90

def statement5_correct : Prop :=
  ∀ (a b c : ℝ), a + b + c = 180 → a < 90 ∧ b < 90 ∧ c < 90

def statement6_correct : Prop :=
  ∀ (a b c : ℝ), a + b + c = 180 ∧ a = 90 → ¬(90 < b ∧ b < 180) ∧ ¬(90 < c ∧ c < 180)

def statement7_correct : Prop :=
  ∀ (θ : ℝ), θ = 25 → θ ≠ 250

theorem correct_statements_count : statement1_correct ∧ statement3_correct ∧ statement4_correct ∧ statement5_correct ∧ ¬statement2_correct ∧ ¬statement6_correct ∧ ¬statement7_correct → (∑ i in {1, 2, 3, 4, 5, 6, 7} \ {2, 6, 7}, 1) = 4 :=
by
  sorry

end correct_statements_count_l11_11307


namespace angle_bisector_length_CM_l11_11105

-- Given conditions
namespace Geometry

def acute_triangle (A B C : Point) : Prop :=
  ∠A < 90° ∧ ∠B < 90° ∧ ∠C < 90°

def altitude (A : Point) (line : Line) : Line := sorry

def on_altitude (P : Point) (A line : Line) : Prop := sorry

def orthogonal (A B C : Point) : Prop :=
  ∠B A C = 90°

def distance (P Q : Point) : ℝ := sorry

def angle (A B C : Point) : ℝ := sorry

def angle_bisector_length (A B C : Point) : ℝ := sorry

-- Define the proof problem
theorem angle_bisector_length_CM :
  ∀ (A B C M N : Point),
    acute_triangle A B C →
    on_altitude M A (altitude B C) →
    on_altitude N B (altitude A C) →
    orthogonal B M C →
    orthogonal A N C →
    distance M N = 4 + 2 * sqrt 3 →
    angle M C N = 30° →
    angle_bisector_length C M N = 7 + 4 * sqrt 3 :=
by
  intros A B C M N h1 h2 h3 h4 h5 h6 h7
  sorry

end Geometry

end angle_bisector_length_CM_l11_11105


namespace sum_of_areas_of_all_squares_l11_11670

noncomputable def sum_of_infinite_squares (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

theorem sum_of_areas_of_all_squares :
  let s := 4 in
  let a := (s:ℝ)^2 in
  let r := (1:ℝ) / 2 in
  sum_of_infinite_squares a r = 32 :=
by
  let s : ℝ := 4  -- side of the first square
  let a : ℝ := s^2  -- the area of the first square
  let r : ℝ := 1 / 2  -- common ratio of the geometric series of the areas
  sorry

end sum_of_areas_of_all_squares_l11_11670


namespace rank_colleagues_l11_11335

variables (P Q R S : Prop)

-- Represent the statements as propositions
def Emma_not_highest : Prop := ¬(Emma > David ∧ Emma > Fiona ∧ Emma > George)
def David_lowest : Prop := (David < Emma ∧ David < Fiona ∧ David < George)
def George_higher_than_Fiona : Prop := (George > Fiona)
def Fiona_not_lowest : Prop := (∃ x, x ≠ Fiona ∧ x < Fiona)

-- Assume only one of these statements is true
def exactly_one_true (p q r s : Prop) := 
  (p ∧ ¬q ∧ ¬r ∧ ¬s) ∨ (¬p ∧ q ∧ ¬r ∧ ¬s) ∨ (¬p ∧ ¬q ∧ r ∧ ¬s) ∨ (¬p ∧ ¬q ∧ ¬r ∧ s)
  
theorem rank_colleagues 
  (h : exactly_one_true Emma_not_highest David_lowest George_higher_than_Fiona Fiona_not_lowest) :
  rank = ["George", "Emma", "David", "Fiona"] :=
sorry

end rank_colleagues_l11_11335


namespace inequality_with_sum_of_one_l11_11618

theorem inequality_with_sum_of_one
  (a b c d : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_sum: a + b + c + d = 1) :
  (a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) >= 1 / 2) :=
sorry

end inequality_with_sum_of_one_l11_11618


namespace range_of_a_l11_11448

theorem range_of_a :
  (¬ ∃ x : ℝ, 2 * x^2 - 3 * a * x + 9 < 0) → a ∈ set.Icc (-2 : ℝ) 2 :=
sorry

end range_of_a_l11_11448


namespace problem1_problem2_l11_11145

-- Problem 1 statement
theorem problem1 (x : ℝ) : 
  log (abs (x + 3) + abs (x - 7)) > 1 ↔ (x < -3 ∨ x > 7) := 
  sorry

-- Problem 2 statement
theorem problem2 (a : ℝ) : 
  (∀ x : ℝ, log (abs (x + 3) + abs (x - 7)) > a) ↔ a < 1 := 
  sorry

end problem1_problem2_l11_11145


namespace sum_of_prime_factors_l11_11753

-- Define the given expression
def expr := 7^7 - 7^4

-- Define the prime factors of the expression
def prime_factors := {2, 3, 7, 19}

-- Statement: The sum of the distinct prime factors of the expression is 31
theorem sum_of_prime_factors : (∑ p in prime_factors, p) = 31 := by
  sorry

end sum_of_prime_factors_l11_11753


namespace integral_result_l11_11526

noncomputable def integral_of_abs_sin_minus_cos (A B C : ℝ) (hA : sin A = 3/5) (hB : cos B = 3/5) (hABC : A + B + C = π) : ℝ :=
∫ x in 0..C, |sin x - cos x|

theorem integral_result (A B C : ℝ) (hA : sin A = 3/5) (hB : cos B = 3/5) (hABC : A + B + C = π) :
  integral_of_abs_sin_minus_cos A B C hA hB hABC = 2*sqrt 2 - 2 := 
sorry

end integral_result_l11_11526


namespace tangent_line_condition_l11_11902

theorem tangent_line_condition (a : ℝ) :
  (∃ (L : ℝ → ℝ), (L (1) = 0) ∧
    (∃ (x0 : ℝ), ((L x0 = x0 ^ 3) ∧ (L' x0 = (3 * x0 ^ 2))) ∧
    (∃ (x1 : ℝ), ((L x1 = a * x1 ^ 2 + (15 / 4) * x1 - 9) ∧
    (L' x1 = (2 * a * x1 + (15 / 4))))) ∧
  (a = -1 ∨ a = -25 / 64)) :=
begin
  sorry
end

end tangent_line_condition_l11_11902


namespace three_digit_even_sum_12_l11_11034

theorem three_digit_even_sum_12 : 
  ∃ (n : Finset ℕ), 
    n.card = 27 ∧ 
    ∀ x ∈ n, 
      ∃ h t u, 
        (100 * h + 10 * t + u = x) ∧ 
        (h ∈ Finset.range 9 \ {0}) ∧ 
        (u % 2 = 0) ∧ 
        (t + u = 12) := 
sorry

end three_digit_even_sum_12_l11_11034


namespace least_positive_integer_to_multiple_of_5_l11_11650

theorem least_positive_integer_to_multiple_of_5 : 
  ∃ n : ℕ, n > 0 ∧ (789 + n) % 5 = 0 ∧ n = 1 :=
by
  use 1
  split
  { exact Nat.zero_lt_succ 0 }
  split
  { norm_num }
  { refl }

end least_positive_integer_to_multiple_of_5_l11_11650


namespace sqrt_50_floor_squared_l11_11792

theorem sqrt_50_floor_squared : (⌊Real.sqrt 50⌋ : ℝ)^2 = 49 := by
  have sqrt_50_bounds : 7 < Real.sqrt 50 ∧ Real.sqrt 50 < 8 := by
    split
    · have : Real.sqrt 49 < Real.sqrt 50 := by sorry
      linarith
    · have : Real.sqrt 50 < Real.sqrt 64 := by sorry
      linarith
  have floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := by
    sorry
  rw [floor_sqrt_50]
  norm_num

end sqrt_50_floor_squared_l11_11792


namespace simplify_and_evaluate_l11_11998

theorem simplify_and_evaluate (a : ℝ) (h : a = 3) : ((2 * a / (a + 1) - 1) / ((a - 1)^2 / (a + 1))) = 1 / 2 := by
  sorry

end simplify_and_evaluate_l11_11998


namespace triangle_sin_A_and_height_l11_11487

noncomputable theory

variables (A B C : ℝ) (AB : ℝ)
  (h1 : A + B = 3 * C)
  (h2 : 2 * Real.sin (A - C) = Real.sin B)
  (h3 : AB = 5)

theorem triangle_sin_A_and_height :
  Real.sin A = 3 * Real.cos A → 
  sqrt 10 / 10 * Real.sin A = 3 / sqrt (10) / 3 → 
  √10 / 10 = 3/ sqrt 10 /3 → 
  sin (A+B) =sin /sqrt10 →
  (sin (A cv)+ C) = sin( AC ) → 
  ( cos A = sinA 3) 
  ( (10 +25)+5→1= well 5 → B (PS6 S)=H1 (A3+.B9)=
 
 
   
∧   (γ = hA → ) ( (/. );



∧ side /4→ABh3 → 5=HS)  ( →AB3)=sinh1S  

then 
(
  (Real.sin A = 3 * Real.cos A) ^2 )+   
  
(Real.cos A= √ 10/10
  
  Real.sin A2 C(B)= 3√10/10
  
 ) ^(Real.sin A = 5

6)=    
    sorry

end triangle_sin_A_and_height_l11_11487


namespace ab_equals_one_l11_11965

theorem ab_equals_one {a b : ℝ} (h : a ≠ b) (hf : |Real.log a| = |Real.log b|) : a * b = 1 :=
  sorry

end ab_equals_one_l11_11965


namespace rationalize_sqrt_fraction_l11_11170

theorem rationalize_sqrt_fraction {a b : ℝ} (a_pos : 0 < a) (b_pos : 0 < b) : 
  (Real.sqrt ((a : ℝ) / b)) = (Real.sqrt (a * (b / (b * b)))) → 
  (Real.sqrt (5 / 12)) = (Real.sqrt 15 / 6) :=
by
  sorry

end rationalize_sqrt_fraction_l11_11170


namespace probability_of_one_or_two_l11_11723

/-- Represents the number of elements in the first 20 rows of Pascal's Triangle. -/
noncomputable def total_elements : ℕ := 210

/-- Represents the number of ones in the first 20 rows of Pascal's Triangle. -/
noncomputable def number_of_ones : ℕ := 39

/-- Represents the number of twos in the first 20 rows of Pascal's Triangle. -/
noncomputable def number_of_twos : ℕ :=18

/-- Prove that the probability of randomly choosing an element which is either 1 or 2
from the first 20 rows of Pascal's Triangle is 57/210. -/
theorem probability_of_one_or_two (h1 : total_elements = 210)
                                  (h2 : number_of_ones = 39)
                                  (h3 : number_of_twos = 18) :
    39 + 18 = 57 ∧ (57 : ℚ) / 210 = 57 / 210 :=
by {
    sorry
}

end probability_of_one_or_two_l11_11723


namespace vector_magnitude_and_perpendicular_property_l11_11881

-- Definitions of the vectors
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (-3, 2)

-- Definition of vector addition and subtraction
def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def vec_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

-- Definition of the magnitude of a vector
def vec_mag (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Definition of the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Expressions for the problems
def vec_a_add_vec_b := vec_add vec_a vec_b
def vec_a_sub_vec_b := vec_sub vec_a vec_b

def k_vec_a_add_vec_b (k : ℝ) := vec_add (k • vec_a) vec_b
def vec_a_sub_3vec_b := vec_sub vec_a (3 • vec_b)

-- The desired Lean theorem
theorem vector_magnitude_and_perpendicular_property :
  vec_mag vec_a_add_vec_b = 2 * Real.sqrt 5 ∧
  vec_mag vec_a_sub_vec_b = 4 ∧
  ∃ k : ℝ, dot_product (k_vec_a_add_vec_b k) vec_a_sub_3vec_b = 0 ∧ k = -1/3 :=
by sorry

end vector_magnitude_and_perpendicular_property_l11_11881


namespace find_ages_l11_11316

theorem find_ages :
  ∃ (B J S : ℕ), (B = 3 * J) ∧ (B + J = 48) ∧ (J + S = 30) ∧ (B = 36) ∧ (J = 12) ∧ (S = 18) :=
by
  use 36
  use 12
  use 18
  sorry

end find_ages_l11_11316


namespace trains_pass_time_l11_11267

def length_train1 : ℕ := 200
def length_train2 : ℕ := 280

def speed_train1_kmph : ℕ := 42
def speed_train2_kmph : ℕ := 30

def kmph_to_mps (speed_kmph : ℕ) : ℚ :=
  speed_kmph * 1000 / 3600

def relative_speed_mps : ℚ :=
  kmph_to_mps (speed_train1_kmph + speed_train2_kmph)

def total_length : ℕ :=
  length_train1 + length_train2

def time_to_pass_trains : ℚ :=
  total_length / relative_speed_mps

theorem trains_pass_time :
  time_to_pass_trains = 24 := by
  sorry

end trains_pass_time_l11_11267


namespace value_independent_of_a_value_when_b_is_neg_2_l11_11410

noncomputable def algebraic_expression (a b : ℝ) : ℝ :=
  3 * a^2 + (4 * a * b - a^2) - 2 * (a^2 + 2 * a * b - b^2)

theorem value_independent_of_a (a b : ℝ) : algebraic_expression a b = 2 * b^2 :=
by
  sorry

theorem value_when_b_is_neg_2 (a : ℝ) : algebraic_expression a (-2) = 8 :=
by
  sorry

end value_independent_of_a_value_when_b_is_neg_2_l11_11410


namespace simplest_form_correct_l11_11663

variable (A : ℝ)
variable (B : ℝ)
variable (C : ℝ)
variable (D : ℝ)

def is_simplest_form (x : ℝ) : Prop :=
-- define what it means for a square root to be in simplest form
sorry

theorem simplest_form_correct :
  A = Real.sqrt (1 / 2) ∧ B = Real.sqrt 0.2 ∧ C = Real.sqrt 3 ∧ D = Real.sqrt 8 →
  ¬ is_simplest_form A ∧ ¬ is_simplest_form B ∧ is_simplest_form C ∧ ¬ is_simplest_form D :=
by
  -- prove that C is the simplest form and others are not
  sorry

end simplest_form_correct_l11_11663


namespace blue_balls_initial_count_l11_11629

theorem blue_balls_initial_count (B : ℕ)
  (h1 : 15 - 3 = 12)
  (h2 : (B - 3) / 12 = 1 / 3) :
  B = 7 :=
sorry

end blue_balls_initial_count_l11_11629


namespace gecko_bug_eating_l11_11112

theorem gecko_bug_eating (G L F T : ℝ) (hL : L = G / 2)
                                      (hF : F = 3 * L)
                                      (hT : T = 1.5 * F)
                                      (hTotal : G + L + F + T = 63) :
  G = 15 :=
by
  sorry

end gecko_bug_eating_l11_11112


namespace max_interested_graduates_l11_11349

-- Define variables and conditions
variables (graduates universities calls_per_uni half_calls max_grad)

-- There are 100 graduates from city N
def graduates := 100

-- 5 universities
def universities := 5

-- Each university calls half of the graduates
def calls_per_uni := graduates / 2

-- Total calls made by all universities
def total_calls := universities * calls_per_uni

-- Maximum graduates of interest to the military recruitment office
def max_grad : ℕ := ∀ n, n ≤ graduates ∧ 2 * n + 5 * (graduates - n) ≤ total_calls

theorem max_interested_graduates:
  max_grad ≤ 83 :=
by
  sorry

end max_interested_graduates_l11_11349


namespace dot_product_eq_l11_11437

-- Define the conditions from a)
variable (e1 e2 : EuclideanSpace) -- Vector space elements
variable (h1 : ∥e1∥ = 1) -- e1 is a unit vector
variable (h2 : ∥e2∥ = 1) -- e2 is a unit vector
variable (angle : Real.angle) (h3 : angle = π / 3) (h4 : e1 ⬝ e2 = cos (π / 3))

-- Define vectors a and b as per the conditions
def a := 2 • e1 + e2
def b := -3 • e1 + 2 • e2

-- State the proof problem
theorem dot_product_eq : a ⬝ b = -7 / 2 := by
  sorry

end dot_product_eq_l11_11437


namespace range_of_k_l11_11611

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def line (k : ℝ) (x : ℝ) : ℝ := k * x + 3 

def circle := { p : ℝ × ℝ | (p.1 - 3) ^ 2 + (p.2 - 2) ^ 2 = 4 }

theorem range_of_k (k : ℝ) :
  (∃ M N : ℝ × ℝ, 
    M ∈ circle ∧ N ∈ circle ∧ 
    (line k M.1 = M.2) ∧ (line k N.1 = N.2) ∧ 
    distance M.1 M.2 N.1 N.2 ≥ 2 * Real.sqrt 3) 
  ↔ (-3 / 4 ≤ k ∧ k ≤ 0) := 
sorry

end range_of_k_l11_11611


namespace correct_system_of_equations_l11_11923

variable (x y : Real)

-- Conditions
def condition1 : Prop := y = x + 4.5
def condition2 : Prop := 0.5 * y = x - 1

-- Main statement representing the correct system of equations
theorem correct_system_of_equations : condition1 x y ∧ condition2 x y :=
  sorry

end correct_system_of_equations_l11_11923


namespace commonTangents_proof_l11_11882

open Function

/-- Number of common tangents of two circles based on their geometric configuration. -/
inductive CircleConfig
| outside -- Both circles lie entirely outside each other
| touchExternally -- The circles touch externally at one point
| intersect -- The circles intersect each other in two points
| touchInternally -- The circles touch internally at one point
| insideWithoutTouch -- One circle lies entirely inside the other without touching
| identical -- The circles are identical
| onePointDegenerate (case : Fin 5) -- One circle degenerates to a point with subcases
| twoPointsDistinct -- Both circles degenerate to two distinct points
| twoPointsCoincide -- Both circles degenerate to the same point

def numCommonTangents : CircleConfig → ℕ∞
| CircleConfig.outside            => 4
| CircleConfig.touchExternally    => 3
| CircleConfig.intersect          => 2
| CircleConfig.touchInternally    => 1
| CircleConfig.insideWithoutTouch => 0
| CircleConfig.identical          => ∞
| CircleConfig.onePointDegenerate 0 => 2
| CircleConfig.onePointDegenerate _ => 1
| CircleConfig.twoPointsDistinct  => 1
| CircleConfig.twoPointsCoincide  => ∞

/-- Prove that for any CircleConfig, numCommonTangents returns the correct number of tangents -/
theorem commonTangents_proof (config : CircleConfig) : numCommonTangents config = 
  match config with
  | CircleConfig.outside            => 4
  | CircleConfig.touchExternally    => 3
  | CircleConfig.intersect          => 2
  | CircleConfig.touchInternally    => 1
  | CircleConfig.insideWithoutTouch => 0
  | CircleConfig.identical          => ∞
  | CircleConfig.onePointDegenerate 0 => 2
  | CircleConfig.onePointDegenerate _ => 1
  | CircleConfig.twoPointsDistinct  => 1
  | CircleConfig.twoPointsCoincide  => ∞ := 
by
  sorry

end commonTangents_proof_l11_11882


namespace percentage_of_students_wanting_oatmeal_raisin_l11_11372

theorem percentage_of_students_wanting_oatmeal_raisin
  (total_students : ℕ)
  (cookies_per_student : ℕ)
  (oatmeal_raisin_cookies : ℕ)
  (h1 : total_students = 40)
  (h2 : cookies_per_student = 2)
  (h3 : oatmeal_raisin_cookies = 8) :
  (oatmeal_raisin_cookies / cookies_per_student)%ℕ * 100 / total_students = 10 :=
by
  sorry

end percentage_of_students_wanting_oatmeal_raisin_l11_11372


namespace gcd_840_1764_gcd_440_556_l11_11277

def gcd_euclidean (a b : ℕ) : ℕ :=
  if b = 0 then a
  else gcd_euclidean b (a % b)

def gcd_successive_subtraction (a b : ℕ) : ℕ :=
  if a = b then a
  else if a > b then gcd_successive_subtraction (a - b) b
  else gcd_successive_subtraction a (b - a)

theorem gcd_840_1764 : gcd_euclidean 840 1764 = 84 := sorry

theorem gcd_440_556 : gcd_successive_subtraction 440 556 = 4 := sorry

end gcd_840_1764_gcd_440_556_l11_11277


namespace joel_age_when_dad_is_twice_l11_11940

-- Given Conditions
def joel_age_now : ℕ := 5
def dad_age_now : ℕ := 32
def age_difference : ℕ := dad_age_now - joel_age_now

-- Proof Problem Statement
theorem joel_age_when_dad_is_twice (x : ℕ) (hx : dad_age_now - joel_age_now = 27) : x = 27 :=
by
  sorry

end joel_age_when_dad_is_twice_l11_11940


namespace area_STQU_eq_five_l11_11928

-- Define the points and lengths
variables {P Q R S T U : Type}
variables (P Q R S : Point)

-- Define the rectangle properties
def rectangle (P Q R S : Point) := side PQ = 5 ∧ side QR = 3 ∧ angle PQR = 90

-- Define the division of diagonal PR into three equal segments
def equal_division (P R : Point) (T U : Point) :=
  segment_length PT = segment_length TU ∧ segment_length TU = segment_length UR

-- Define the points T and U lie on PR
def on_diagonal (T U : Point) := lies_on T PR ∧ lies_on U PR

-- Define the quadrilateral STQU's points form
def STQU (S T Q U : Point) := quadrilateral S T Q U

-- Main theorem statement
theorem area_STQU_eq_five (P Q R S T U : Point) 
  (h_rect : rectangle P Q R S) 
  (h_div : equal_division P R T U)
  (h_diag : on_diagonal T U) : 
  area (STQU S T Q U) = 5 := 
sorry

end area_STQU_eq_five_l11_11928


namespace count_even_three_digit_numbers_with_sum_12_l11_11014

noncomputable def even_three_digit_numbers_with_sum_12 : Prop :=
  let valid_pairs := [(8, 4), (6, 6), (4, 8)] in
  let valid_hundreds := 9 in
  let count_pairs := valid_pairs.length in
  let total_numbers := valid_hundreds * count_pairs in
  total_numbers = 27

theorem count_even_three_digit_numbers_with_sum_12 : even_three_digit_numbers_with_sum_12 :=
by
  sorry

end count_even_three_digit_numbers_with_sum_12_l11_11014


namespace count_even_three_digit_sum_tens_units_is_12_l11_11054

-- Define what it means to be a three-digit number
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n < 1000)

-- Define what it means to be even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the sum of the tens and units digits to be 12
def sum_tens_units_is_12 (n : ℕ) : Prop := 
  let tens := (n / 10) % 10 in
  let units := n % 10 in
  tens + units = 12

-- Count how many such numbers exist
theorem count_even_three_digit_sum_tens_units_is_12 : 
  ∃! n : ℕ, (is_three_digit n) ∧ (is_even n) ∧ (sum_tens_units_is_12 n) = 36 :=
sorry

end count_even_three_digit_sum_tens_units_is_12_l11_11054


namespace integral_inequality_l11_11541

open Real

noncomputable def is_convex (f : ℝ → ℝ) : Prop :=
∀ (x y : ℝ) (a : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 1), f (a * x + (1 - a) * y) ≤ a * f(x) + (1 - a) * f(y)

theorem integral_inequality (f : ℝ → ℝ) (α β γ : ℝ) (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ) (h_convex : is_convex f) :
  (1 / (6 * α)) * ∫ x in 0..(6 * α), f x 
  + (1 / (6 * β)) * ∫ x in 0..(6 * β), f x 
  + (1 / (6 * γ)) * ∫ x in 0..(6 * γ), f x 
  ≥ (1 / (3 * α + 2 * β + γ)) * ∫ x in 0..(3 * α + 2 * β + γ), f x 
  + (1 / (α + 3 * β + 2 * γ)) * ∫ x in 0..(α + 3 * β + 2 * γ), f x 
  + (1 / (2 * α + β + 3 * γ)) * ∫ x in 0..(2 * α + β + 3 * γ), f x := 
sorry

end integral_inequality_l11_11541


namespace eval_f_neg2_l11_11606

def f : ℝ → ℝ := 
λ x, if x ≥ 0 then 2 * x else x * (x + 1)

theorem eval_f_neg2 : f (-2) = 2 :=
by
  -- Proof placeholder
  sorry

end eval_f_neg2_l11_11606


namespace gcd_4557_1953_5115_l11_11210

theorem gcd_4557_1953_5115 : Nat.gcd (Nat.gcd 4557 1953) 5115 = 93 :=
by
  -- We use 'sorry' to skip the proof part as per the instructions.
  sorry

end gcd_4557_1953_5115_l11_11210


namespace ratio_of_elements_l11_11324

theorem ratio_of_elements
  (wt_total : ℝ)
  (wt_b : ℝ)
  (h_total : wt_total = 300)
  (h_b : wt_b = 250) :
  let wt_a := wt_total - wt_b in
  wt_a / wt_b = 1 / 5 :=
by
  intros
  sorry

end ratio_of_elements_l11_11324


namespace number_of_men_in_engineering_department_l11_11108

theorem number_of_men_in_engineering_department (T : ℝ) (h1 : 0.30 * T = 180) : 
  0.70 * T = 420 :=
by 
  -- The proof will be done here, but for now, we skip it.
  sorry

end number_of_men_in_engineering_department_l11_11108


namespace number_of_four_digit_numbers_divisible_by_17_l11_11061

theorem number_of_four_digit_numbers_divisible_by_17 :
  let k_min := Int.ceil (1000 / 17)
  let k_max := Int.floor (9999 / 17)
  k_max - k_min + 1 = 530 := by
  sorry

end number_of_four_digit_numbers_divisible_by_17_l11_11061


namespace find_a_value_l11_11350

noncomputable def curve_parametric (a : ℝ) (t : ℝ) (h : a > 0) : ℝ × ℝ :=
  (a * Real.cos t, 1 + a * Real.sin t)

noncomputable def curve_polar : ℝ → ℝ := λ θ, 4 * Real.cos θ

noncomputable def line_theta : ℝ := Real.arctan 2

theorem find_a_value (a : ℝ) (h : a > 0) :
  (∀ (t θ : ℝ), curve_parametric a t h = (4 * Real.cos θ, 2 * Real.sin θ)) → a = 1 :=
by
  intro h_common_points
  sorry

end find_a_value_l11_11350


namespace sin_A_correct_height_on_AB_correct_l11_11475

noncomputable def sin_A (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) : ℝ :=
  Real.sin A

noncomputable def height_on_AB (A B C AB : ℝ) (height : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) (h4 : AB = 5) : ℝ :=
  height

theorem sin_A_correct (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) : 
  sorrry := 
begin
  -- proof omitted
  sorrry
end

theorem height_on_AB_correct (A B C AB : ℝ) (height : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) (h4 : AB = 5) :
  height = 6:= 
begin
  -- proof omitted
  sorrry
end 

end sin_A_correct_height_on_AB_correct_l11_11475


namespace exists_small_area_triangle_l11_11463

def lattice_point (x y : ℤ) : Prop := |x| ≤ 2 ∧ |y| ≤ 2

def no_three_collinear (points : List (ℤ × ℤ)) : Prop :=
∀ (p1 p2 p3 : ℤ × ℤ), p1 ∈ points → p2 ∈ points → p3 ∈ points → 
(p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) →
¬ (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2) = 0)

noncomputable def triangle_area (p1 p2 p3 : ℤ × ℤ) : ℚ :=
(1 / 2 : ℚ) * |(p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))|

theorem exists_small_area_triangle {points : List (ℤ × ℤ)}
  (h1 : points.length = 6)
  (h2 : ∀ (p : ℤ × ℤ), p ∈ points → lattice_point p.1 p.2)
  (h3 : no_three_collinear points) :
  ∃ (p1 p2 p3 : ℤ × ℤ), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
  (p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) ∧ 
  triangle_area p1 p2 p3 ≤ 2 := 
sorry

end exists_small_area_triangle_l11_11463


namespace sequence_lower_bound_l11_11201

noncomputable section

def isSequence (s : List ℕ) : Prop :=
  s.head = 1 ∧ s.headTail = 2 ∧
  ∀ a b, a ≠ b → a+b ∉ s

theorem sequence_lower_bound (s : List ℕ) (k : ℕ) (h : isSequence s) :
  (s.filter (· < k)).length ≤ k / 3 + 2 := 
sorry

end sequence_lower_bound_l11_11201


namespace total_arrangements_of_programs_l11_11971

theorem total_arrangements_of_programs :
  let original_programs := 6
  let added_programs := 3
  let arrangements :=
    Nat.choose (original_programs + 1) 1 * (Nat.factorial added_programs) +
    Nat.perms (original_programs + added_programs) added_programs +
    (Nat.choose added_programs 1) * (Nat.choose (original_programs + 1) 1) * (Nat.choose original_programs 1) * (Nat.factorial 2)
  in arrangements = 504 := sorry

end total_arrangements_of_programs_l11_11971


namespace basketball_players_l11_11091

def total_students : ℕ := 25
def play_hockey : ℕ := 15
def play_neither : ℕ := 4
def play_both : ℕ := 10

theorem basketball_players : 
  let total_at_least_one := total_students - play_neither in
  let only_hockey := play_hockey - play_both in
  let only_basketball := total_at_least_one - only_hockey - play_both in
  only_basketball + play_both = 16 :=
by
  sorry

end basketball_players_l11_11091


namespace evaluate_expression_l11_11817

theorem evaluate_expression :
  ∀ (a b c : ℚ),
  c = b + 1 →
  b = a + 5 →
  a = 3 →
  (a + 2 ≠ 0) →
  (b - 3 ≠ 0) →
  (c + 7 ≠ 0) →
  (a + 3) * (b + 1) * (c + 9) / ((a + 2) * (b - 3) * (c + 7)) = 2.43 := 
by
  intros a b c hc hb ha h1 h2 h3
  sorry

end evaluate_expression_l11_11817


namespace coeff_x3_expansion_l11_11190

-- Define what it means to expand (1+x)^6 and (1-x)^4
def binomial_coeff (n k : ℕ) : ℤ := (nat.choose n k : ℤ)
def expansion_1_plus_x (n r : ℕ) : ℤ := binomial_coeff n r
def expansion_1_minus_x (n r : ℕ) : ℤ := (-1) ^ r * binomial_coeff n r

-- Define the main statement
theorem coeff_x3_expansion :
  let term := expansion_1_plus_x 6 ∗ expansion_1_minus_x 4,
  (term 3) = -8 :=
by
  sorry  -- Proof omitted as per instructions.

end coeff_x3_expansion_l11_11190


namespace remainder_of_trailing_zeros_in_factorials_product_l11_11949

theorem remainder_of_trailing_zeros_in_factorials_product :
  let M := (trailing_zeroes (finprod (λ n : ℕ, if n ∈ (set.range (λ k, k!)) then n else 1))) in
  M % 500 = 21 := by
  sorry

def trailing_zeroes (n : ℕ) : ℕ := 
  if n = 0 then 0 else trailing_zeroes_aux n 0

@[simp] def trailing_zeroes_aux (n : ℕ) (acc : ℕ) : ℕ :=
  if n % 10 = 0 then trailing_zeroes_aux (n / 10) (acc + 1) else acc

noncomputable def finprod {α β : Type*} [comm_monoid β] (f : α → β) : β :=
  finset.univ.prod f

end remainder_of_trailing_zeros_in_factorials_product_l11_11949


namespace square_area_diagonal_12_l11_11643

theorem square_area_diagonal_12 :
  ∃ (s : ℝ), s > 0 ∧ 2 * s^2 = 12^2 ∧ s^2 = 72 :=
begin
  sorry
end

end square_area_diagonal_12_l11_11643


namespace tangent_point_l11_11407

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x

noncomputable def g (x : ℝ) : ℝ := x + 1 - x * Real.log x

theorem tangent_point (a : ℝ) (n : ℕ) (hn : n > 0) (ha : n < a ∧ a < n + 1) :
  y = a + 1 → y = a * Real.log a → n = 3 :=
sorry

end tangent_point_l11_11407


namespace probability_diana_greater_than_apollo_l11_11347

theorem probability_diana_greater_than_apollo :
  (  let possible_outcomes := 8 * 8,
         successful_outcomes := 6 + 5 + 4 + 3 + 2 + 1
     in successful_outcomes / possible_outcomes.to_rat
  )  = 21 / 64 :=
by
  sorry

end probability_diana_greater_than_apollo_l11_11347


namespace area_below_line_l11_11646

noncomputable def circle := set_of (λ p : ℝ × ℝ, (p.1 - 5)^2 + (p.2 - 7.5)^2 = 56.25)
noncomputable def line := set_of (λ p : ℝ × ℝ, p.2 = p.1 - 1)

theorem area_below_line :
  ∃ (a : ℝ), a = 56.25 * π * (3 / 4) := sorry

end area_below_line_l11_11646


namespace triangle_sin_A_and_height_l11_11486

noncomputable theory

variables (A B C : ℝ) (AB : ℝ)
  (h1 : A + B = 3 * C)
  (h2 : 2 * Real.sin (A - C) = Real.sin B)
  (h3 : AB = 5)

theorem triangle_sin_A_and_height :
  Real.sin A = 3 * Real.cos A → 
  sqrt 10 / 10 * Real.sin A = 3 / sqrt (10) / 3 → 
  √10 / 10 = 3/ sqrt 10 /3 → 
  sin (A+B) =sin /sqrt10 →
  (sin (A cv)+ C) = sin( AC ) → 
  ( cos A = sinA 3) 
  ( (10 +25)+5→1= well 5 → B (PS6 S)=H1 (A3+.B9)=
 
 
   
∧   (γ = hA → ) ( (/. );



∧ side /4→ABh3 → 5=HS)  ( →AB3)=sinh1S  

then 
(
  (Real.sin A = 3 * Real.cos A) ^2 )+   
  
(Real.cos A= √ 10/10
  
  Real.sin A2 C(B)= 3√10/10
  
 ) ^(Real.sin A = 5

6)=    
    sorry

end triangle_sin_A_and_height_l11_11486


namespace imaginary_part_of_z_l11_11864

noncomputable def z : ℂ := (2 : ℂ) / (-1 + (1 : ℂ) * complex.I)

theorem imaginary_part_of_z : z.im = -1 := by
  sorry

end imaginary_part_of_z_l11_11864


namespace proof_triangle_properties_l11_11497

variable (A B C : ℝ)
variable (h AB : ℝ)

-- Conditions
def triangle_conditions : Prop :=
  (A + B = 3 * C) ∧ (2 * Real.sin (A - C) = Real.sin B) ∧ (AB = 5)

-- Part 1: Proving sin A
def find_sin_A (h₁ : triangle_conditions A B C h AB) : Prop :=
  Real.sin A = 3 * Real.cos A

-- Part 2: Proving the height on side AB
def find_height_on_AB (h₁ : triangle_conditions A B C h AB) : Prop :=
  h = 6

-- Combined proof statement
theorem proof_triangle_properties (h₁ : triangle_conditions A B C h AB) : 
  find_sin_A A B C h₁ ∧ find_height_on_AB A B C h AB h₁ := 
  by sorry

end proof_triangle_properties_l11_11497


namespace amare_additional_fabric_needed_l11_11159

-- Defining the conditions
def yards_per_dress : ℝ := 5.5
def num_dresses : ℝ := 4
def initial_fabric_feet : ℝ := 7
def yard_to_feet : ℝ := 3

-- The theorem to prove
theorem amare_additional_fabric_needed : 
  (yards_per_dress * num_dresses * yard_to_feet) - initial_fabric_feet = 59 := 
by
  sorry

end amare_additional_fabric_needed_l11_11159


namespace amare_fabric_needed_l11_11161

-- Definitions for the conditions
def fabric_per_dress_yards : ℝ := 5.5
def number_of_dresses : ℕ := 4
def fabric_owned_feet : ℝ := 7
def yard_to_feet : ℝ := 3

-- Total fabric needed in yards
def total_fabric_needed_yards : ℝ := fabric_per_dress_yards * number_of_dresses

-- Total fabric needed in feet
def total_fabric_needed_feet : ℝ := total_fabric_needed_yards * yard_to_feet

-- Fabric still needed
def fabric_still_needed : ℝ := total_fabric_needed_feet - fabric_owned_feet

-- Proof
theorem amare_fabric_needed : fabric_still_needed = 59 := by
  sorry

end amare_fabric_needed_l11_11161


namespace value_of_x_after_z_doubled_l11_11626

theorem value_of_x_after_z_doubled (x y z : ℕ) (hz : z = 48) (hz_d : z_d = 2 * z) (hy : y = z / 4) (hx : x = y / 3) :
  x = 8 := by
  -- Proof goes here (skipped as instructed)
  sorry

end value_of_x_after_z_doubled_l11_11626


namespace increasing_function_iff_condition_l11_11142

theorem increasing_function_iff_condition (f : ℝ → ℝ) (h : ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f(x₁) ≤ f(x₂)) :
  (∀ a b : ℝ, a + b < 0 ↔ f(a) + f(b) < f(-a) + f(-b)) :=
begin
  sorry
end

end increasing_function_iff_condition_l11_11142


namespace sin_A_correct_height_on_AB_correct_l11_11476

noncomputable def sin_A (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) : ℝ :=
  Real.sin A

noncomputable def height_on_AB (A B C AB : ℝ) (height : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) (h4 : AB = 5) : ℝ :=
  height

theorem sin_A_correct (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) : 
  sorrry := 
begin
  -- proof omitted
  sorrry
end

theorem height_on_AB_correct (A B C AB : ℝ) (height : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) (h4 : AB = 5) :
  height = 6:= 
begin
  -- proof omitted
  sorrry
end 

end sin_A_correct_height_on_AB_correct_l11_11476


namespace marbles_per_friend_l11_11147

theorem marbles_per_friend (total_marbles : ℕ) (num_friends : ℕ) (h_total : total_marbles = 30) (h_friends : num_friends = 5) :
  total_marbles / num_friends = 6 :=
by
  -- Proof skipped
  sorry

end marbles_per_friend_l11_11147


namespace kids_still_awake_l11_11278

-- Definition of the conditions
def num_kids_initial : ℕ := 20

def kids_asleep_first_5_minutes : ℕ := num_kids_initial / 2

def kids_awake_after_first_5_minutes : ℕ := num_kids_initial - kids_asleep_first_5_minutes

def kids_asleep_next_5_minutes : ℕ := kids_awake_after_first_5_minutes / 2

def kids_awake_final : ℕ := kids_awake_after_first_5_minutes - kids_asleep_next_5_minutes

-- Theorem that needs to be proved
theorem kids_still_awake : kids_awake_final = 5 := by
  sorry

end kids_still_awake_l11_11278


namespace find_position_of_sqrt_41_in_sequence_l11_11330

theorem find_position_of_sqrt_41_in_sequence :
  ∃ n : ℕ, n > 0 ∧ ∀ a : ℕ, (a = 2 + (n - 1) * 3) ∧ (sqrt a = sqrt 41) → n = 14 :=
by
  sorry

end find_position_of_sqrt_41_in_sequence_l11_11330


namespace sin_A_calculation_height_calculation_l11_11502

variable {A B C : ℝ}

-- Given conditions
def angle_condition : Prop := A + B = 3 * C
def sine_condition : Prop := 2 * sin (A - C) = sin B

-- Part 1: Find sin A
theorem sin_A_calculation (h1 : angle_condition) (h2 : sine_condition) : sin A = 3 * real.sqrt 10 / 10 := sorry

-- Part 2: Given AB = 5, find the height
variable {AB : ℝ}
def AB_value : Prop := AB = 5

theorem height_calculation (h1 : angle_condition) (h2 : sine_condition) (h3 : AB_value) : height = 6 := sorry

end sin_A_calculation_height_calculation_l11_11502


namespace ratio_range_l11_11405

theorem ratio_range (P Q M : ℝ × ℝ)
  (hP : P.1 + P.2 - 1 = 0)
  (hQ : Q.1 + Q.2 + 3 = 0)
  (hM : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2))
  (hM_cond : M.1 - M.2 + 2 < 0) :
  ∃ k : ℝ, (-1 < k ∧ k < -1 / 3) ∧ (M.2 = k * M.1) :=
begin
  sorry
end

end ratio_range_l11_11405


namespace younger_brother_age_l11_11225

variable (x y : ℕ)

-- Conditions
axiom sum_of_ages : x + y = 46
axiom younger_is_third_plus_ten : y = x / 3 + 10

theorem younger_brother_age : y = 19 := 
by
  sorry

end younger_brother_age_l11_11225


namespace determine_common_ratio_l11_11113

variable (a : ℕ → ℝ) (q : ℝ)

-- Given conditions
axiom a2 : a 2 = 1 / 2
axiom a5 : a 5 = 4
axiom geom_seq_def : ∀ n, a n = a 1 * q ^ (n - 1)

-- Prove the common ratio q == 2
theorem determine_common_ratio : q = 2 :=
by
  -- here we should unfold the proof steps given in the solution
  sorry

end determine_common_ratio_l11_11113


namespace ratio_b_to_c_l11_11424

variables (a b c d e f : ℝ)

theorem ratio_b_to_c 
  (h1 : a / b = 1 / 3)
  (h2 : c / d = 1 / 2)
  (h3 : d / e = 3)
  (h4 : e / f = 1 / 10)
  (h5 : a * b * c / (d * e * f) = 0.15) :
  b / c = 9 := 
sorry

end ratio_b_to_c_l11_11424


namespace trajectory_equation_of_P_l11_11406

theorem trajectory_equation_of_P 
  (P F1 F2 : ℝ × ℝ)
  (h1 : ∃ a : ℝ, a > sqrt 2 ∧ (dist P F1 + dist P F2 = 2 * a))
  (h2 : ∀ (| PF1 | | PF2 | : ℝ), 
        | PF1 | = dist P F1 ∧ | PF2 | = dist P F2 →
        cos_angle F1 P F2 = -1 / 3) : 
  ∀ (x y : ℝ), 
  (P = (x, y)) →
  (x ^ 2 / 3 + y ^ 2 = 1) := 
sorry

end trajectory_equation_of_P_l11_11406


namespace Diamond_result_l11_11337

-- Define the binary operation Diamond
def Diamond (a b : ℕ) : ℕ := a * b^2 - b + 1

theorem Diamond_result : Diamond (Diamond 3 4) 2 = 179 := 
by 
  sorry

end Diamond_result_l11_11337


namespace count_even_three_digit_numbers_sum_tens_units_eq_12_l11_11008

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def is_even (n : ℕ) : Prop := n % 2 = 0
def sum_of_tens_and_units_eq_12 (n : ℕ) : Prop :=
  (n / 10) % 10 + n % 10 = 12

theorem count_even_three_digit_numbers_sum_tens_units_eq_12 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_three_digit n ∧ is_even n ∧ sum_of_tens_and_units_eq_12 n) ∧ S.card = 24 :=
sorry

end count_even_three_digit_numbers_sum_tens_units_eq_12_l11_11008


namespace middle_number_l11_11226

theorem middle_number (x y z : ℤ) 
  (h1 : x + y = 21)
  (h2 : x + z = 25)
  (h3 : y + z = 28)
  (h4 : x < y)
  (h5 : y < z) : 
  y = 12 :=
sorry

end middle_number_l11_11226


namespace ratio_of_radii_l11_11120

noncomputable def cos_BAC : ℝ := 1 / 2
noncomputable def AB : ℝ := 2
noncomputable def AC : ℝ := 3
noncomputable def CD : ℝ := 3

noncomputable def circumcircle_radius (a b c : ℝ) (cosine : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  (a * b * c) / (4 * sqrt (s * (s - a) * (s - b) * (s - c)))

noncomputable def incircle_radius (a b c : ℝ) (cosine : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  sqrt (s * (s - a) * (s - b) * (s - c)) / s

theorem ratio_of_radii :
  let a := AB 
  let b := sqrt (a^2 + AC^2 - 2 * a * AC * cos_BAC)
  let c := AC
  let d := CD
  let AD := AC + CD
  let BD := sqrt (a^2 + AD^2 - 2 * a * AD * (-cos_BAC))
  let R := circumcircle_radius a b c cos_BAC
  let r := incircle_radius a BD AD (-cos_BAC)
  R / r = (7 + 4 * sqrt 7) / 9 :=
by sorry

end ratio_of_radii_l11_11120


namespace remainder_of_60_div_18_l11_11211

theorem remainder_of_60_div_18 : ∀ (G Q1 R1 : ℕ), G = 18 ∧ 60 = G * Q1 + R1 → R1 = 6 := by
  intros G Q1 R1 h
  cases h with hG hEq
  rw [hG] at hEq
  have hDiv := Nat.mod_eq_sub_mod G 60
  rw [Nat.sub_self] at hDiv
  have hMod := Nat.mod_eq_sub_mod G (G * Q1 + R1)
  sorry

end remainder_of_60_div_18_l11_11211


namespace distinct_remainders_l11_11549

theorem distinct_remainders (p : ℕ) (a : Fin p → ℤ) (hp : Nat.Prime p) :
  ∃ k : ℤ, (Finset.univ.image (fun i : Fin p => (a i + i * k) % p)).card ≥ ⌈(p / 2 : ℚ)⌉ :=
sorry

end distinct_remainders_l11_11549


namespace correct_speed_l11_11154

-- Define the conditions
def distance_travelled (t : ℝ) (v : ℝ) (adjustment : ℝ) : ℝ :=
  v * (t + adjustment)

def time_adjustment_late := (5 / 60 : ℝ)
def time_adjustment_early := (-5 / 60 : ℝ)

-- Define the speeds
def speed_late := 45
def speed_early := 65

noncomputable def ideal_time : ℝ := 
  let t := (45 * (1/12.0 : ℝ) + 65 * (1/12.0 : ℝ)) / (speed_late - speed_early)
  t

noncomputable def distance : ℝ := 
  distance_travelled ideal_time speed_late time_adjustment_late

noncomputable def required_speed : ℝ := 
  distance / ideal_time

-- Theorem we want to prove
theorem correct_speed : 
  (required_speed ≈ 53) :=
sorry

end correct_speed_l11_11154


namespace max_value_of_square_diff_max_value_of_square_diff_achieved_l11_11840

theorem max_value_of_square_diff (a b : ℝ) (h : a^2 + b^2 = 4) : (a - b)^2 ≤ 8 :=
sorry

theorem max_value_of_square_diff_achieved (a b : ℝ) (h : a^2 + b^2 = 4) : ∃ a b : ℝ, (a - b)^2 = 8 :=
sorry

end max_value_of_square_diff_max_value_of_square_diff_achieved_l11_11840


namespace bamboo_volume_l11_11922

theorem bamboo_volume :
  ∃ (a₁ d a₅ : ℚ), 
  (4 * a₁ + 6 * d = 5) ∧ 
  (3 * a₁ + 21 * d = 4) ∧ 
  (a₅ = a₁ + 4 * d) ∧ 
  (a₅ = 85 / 66) :=
sorry

end bamboo_volume_l11_11922


namespace probability_symmetric_line_l11_11460

open Set

def grid_points : Set (ℕ × ℕ) := { p | p.1 < 7 ∧ p.2 < 7 }

def center_point := (3, 3)  -- In Lean, indexing typically starts at 0

def is_symmetric_line (Q : ℕ × ℕ) : Prop :=
  (Q.1 = center_point.1 ∨ Q.2 = center_point.2 ∨ (Q.1 - center_point.1 = Q.2 - center_point.2) ∨ (Q.1 - center_point.1 = -(Q.2 - center_point.2)))

theorem probability_symmetric_line :
  ∃ (Q : ℕ × ℕ), Q ∈ grid_points ∧ Q ≠ center_point → (∑' x in grid_points, ite (is_symmetric_line x) 1 0) / (card grid_points - 1) = 1 / 2
:= sorry

end probability_symmetric_line_l11_11460


namespace selene_payment_l11_11584

/--
Selene buys two instant cameras at $110 each and three digital photo frames at $120 each.
She gets a 5% discount on all the items she purchased.
Prove that she pays $551 in all after the discount.
-/
theorem selene_payment :
  let cost_camera := 2 * 110,
      cost_frame := 3 * 120,
      total_cost_before_discount := cost_camera + cost_frame,
      discount := 0.05 * total_cost_before_discount,
      total_cost_after_discount := total_cost_before_discount - discount
  in total_cost_after_discount = 551 :=
by
  let cost_camera := 2 * 110
  let cost_frame := 3 * 120
  let total_cost_before_discount := cost_camera + cost_frame
  let discount := 0.05 * total_cost_before_discount
  let total_cost_after_discount := total_cost_before_discount - discount
  calc total_cost_after_discount : total_cost_after_discount = 551 := sorry

end selene_payment_l11_11584


namespace no_real_c_for_single_root_polynomial_l11_11557

theorem no_real_c_for_single_root_polynomial (b c : ℝ) (h : b = c + 2) :
  ∃ x, (x^2 + b * x + c = 0 ∧ ∃ y, x = y) =
  ∅ :=
by
  sorry

end no_real_c_for_single_root_polynomial_l11_11557


namespace frog_jumping_sequences_count_l11_11138

def regular_hexagon_vertices : Type := {A B C D E F : Type}

def frog_jump_sequence (start : regular_hexagon_vertices) : regular_hexagon_vertices → Prop :=
  sorry -- Definition of frog jumping rules, not needed for the statement.

theorem frog_jumping_sequences_count (start : regular_hexagon_vertices) 
  (adj1 adj2 : regular_hexagon_vertices → regular_hexagon_vertices)
  (start_at_A : start = A)
  (adjacent : ∀ (v : regular_hexagon_vertices), adj1 v ≠ adj2 v) :
  -- Conditions:
  -- The frog starts at vertex A
  -- The frog can move randomly to one of the adjacent vertices determined by adj1 and adj2
  -- Frog stops at vertex D if it reaches within 5 jumps otherwise it stops after 5 jumps
  
  ∑ (s : list regular_hexagon_vertices), (length s = 5 ∧ nth s 4 = D ∧ frog_jump_sequence start s) +
  ∑ (s : list regular_hexagon_vertices), (length s ≤ 5 ∧ nth s (length s - 1) = D ∧ frog_jump_sequence start s) = 26 :=
sorry

end frog_jumping_sequences_count_l11_11138


namespace area_of_circle_l11_11979

noncomputable def points_area_circ (C D : Point) (radius length : ℝ) : ℝ :=
  let r := 1/2 * length in
  π * r ^ 2

theorem area_of_circle (C D: Point) (h1: C = (-2, 3)) (h2: D = (4, -1)) : 
  (points_area_circ C D (sqrt (52)) / 2 = 13 * π) :=
by 
  sorry

end area_of_circle_l11_11979


namespace find_f_2017_l11_11859

def isOddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x
def hasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem find_f_2017 (f : ℝ → ℝ) (x : ℝ)
  (h1 : isOddFunction f)
  (h2 : hasPeriod f 4)
  (h3 : ∀ x, x ∈ Ioo (-3 / 2 : ℝ) 0 → f x = Real.logBase 2 (-3 * x + 1))
  (h4 : x = 2017) :
  f x = -2 :=
by
  sorry

end find_f_2017_l11_11859


namespace natural_numbers_divisible_into_pairs_l11_11818

theorem natural_numbers_divisible_into_pairs (n : ℕ) (h : n > 1) :
  ∃ (pairs : List (ℕ × ℕ)), 
    (∀ p ∈ pairs, p.1 ≠ p.2) ∧ (pairs.length = n) ∧ 
    (∀ (p1 p2 : ℕ × ℕ) (hp1 : p1 ∈ pairs) (hp2 : p2 ∈ pairs), p1 ≠ p2 → p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2) ∧ 
    (∃ (k : ℕ), (pairs.map (λ p, (p.1 + p.2))).foldl (*) 1 = k^2) := 
sorry

end natural_numbers_divisible_into_pairs_l11_11818


namespace olivia_wins_5_games_l11_11459

theorem olivia_wins_5_games
  (liam_games_won: ℕ) (liam_games_lost : ℕ)
  (noah_games_won: ℕ) (noah_games_lost: ℕ)
  (olivia_games_lost: ℕ)
  (liam_total_games : liam_games_won + liam_games_lost = 9)
  (noah_total_games : noah_games_won + noah_games_lost = 8)
  (olivia_total_games :  ∃ x:ℕ, x + olivia_games_lost = noah_total_games)
  : ∃ x: ℕ, x = 5 :=
by sorry

end olivia_wins_5_games_l11_11459


namespace solve_arcsin_cos_eq_x_over_3_l11_11588

noncomputable def arcsin (x : Real) : Real := sorry
noncomputable def cos (x : Real) : Real := sorry

theorem solve_arcsin_cos_eq_x_over_3 :
  ∀ x,
  - (3 * Real.pi / 2) ≤ x ∧ x ≤ 3 * Real.pi / 2 →
  arcsin (cos x) = x / 3 →
  x = 3 * Real.pi / 10 ∨ x = 3 * Real.pi / 8 :=
sorry

end solve_arcsin_cos_eq_x_over_3_l11_11588


namespace ratio_of_distances_l11_11945

noncomputable def regularTetrahedron := sorry -- Define properties of regular tetrahedron
noncomputable def pointInPlaneButOutsideTriangleABC := sorry -- Define properties for the point E in the plane of ABC but outside triangle ABC

-- Sum of distances from E to planes DAB, DBC, DCA
noncomputable def s : ℝ := sorry

-- Sum of distances from E to lines AB, BC, CA
noncomputable def S : ℝ := sorry

theorem ratio_of_distances (E : pointInPlaneButOutsideTriangleABC) : 
  let s := sum_of_distances_to_planes E
  let S := sum_of_distances_to_lines E
  (s / S) = Real.sqrt 2 :=
sorry

end ratio_of_distances_l11_11945


namespace one_third_of_nine_times_x_decreased_by_three_is_3x_minus_1_l11_11443

-- Definition of the conditions.
variable (x : ℝ)

-- Statement of the problem in Lean.
theorem one_third_of_nine_times_x_decreased_by_three_is_3x_minus_1 (x : ℝ) :
    (1 / 3) * (9 * x - 3) = 3 * x - 1 :=
by sorry

end one_third_of_nine_times_x_decreased_by_three_is_3x_minus_1_l11_11443


namespace horizontal_asymptote_l11_11608

-- The condition is defined
def f (x : ℝ) : ℝ := (6 * x ^ 2 - 4) / (4 * x ^ 2 + 3 * x - 1)

-- State the theorem to prove the horizontal asymptote
theorem horizontal_asymptote : 
  (∀ (x : ℝ), f x = (6 * x ^ 2 - 4) / (4 * x ^ 2 + 3 * x - 1)) → 
  (tendsto (λ x, f x) (at_top) (nhds (3 / 2))) :=
by 
  sorry

end horizontal_asymptote_l11_11608


namespace angle_bisector_AC_BAE_l11_11538

-- Define the conditions for the convex quadrilateral and the given angles
variables {A B C D E : Type*} [convex_quadrilateral A B C D] 

-- Define an angle measure type
variables {α β γ : ℝ}

-- Define the hypotheses of the theorem
variables (h1 : ∠CAD = 90) 
          (h2 : ∠BCE = 90)
          (h3 : is_bisector (angle A C D) C E intersect D B E)
          
-- Define the theorem statement
theorem angle_bisector_AC_BAE : is_bisector (angle B A E) A C :=
sorry

end angle_bisector_AC_BAE_l11_11538


namespace complex_plane_properties_l11_11927

-- Definitions for the conditions and properties in the problem.
def real_axis_points_are_real_numbers :=
  ∀ (z : ℂ), z.im = 0 ↔ (∃ (a : ℝ), z = a)

def imaginary_axis_points_are_purely_imaginary_numbers :=
  ∀ (z : ℂ), z.re = 0 ↔ (∃ (b : ℝ), z = b * complex.I)

def conjugate_complex_numbers_properties :=
  ∀ (z : ℂ), let conj_z := complex.conj z in (z.re = conj_z.re ∧ z.im = -conj_z.im)

def real_number_multiplied_by_i_is_purely_imaginary :=
  ∀ (a : ℝ), ∃ (z : ℂ), z = a * complex.I ↔ (z.re = 0 ∧ z.im ≠ 0)

-- The goal is to prove that A and C are correct under the given conditions.
theorem complex_plane_properties :
  real_axis_points_are_real_numbers ∧ conjugate_complex_numbers_properties := by
  -- Proof not necessary, using sorry to omit
  sorry

end complex_plane_properties_l11_11927


namespace engineering_department_men_l11_11109

theorem engineering_department_men (total_students men_percentage women_count : ℕ) (h_percentage : men_percentage = 70) (h_women : women_count = 180) (h_total : total_students = (women_count * 100) / (100 - men_percentage)) : 
  (total_students * men_percentage / 100) = 420 :=
by
  sorry

end engineering_department_men_l11_11109


namespace Toms_dog_age_in_6_years_l11_11235

-- Let's define the conditions
variables (B D : ℕ)
axiom h1 : B = 4 * D
axiom h2 : B + 6 = 30

-- Now we state the theorem
theorem Toms_dog_age_in_6_years :
  D + 6 = 12 :=
by
  sorry

end Toms_dog_age_in_6_years_l11_11235


namespace probability_not_overcoming_is_half_l11_11595

/-- Define the five elements. -/
inductive Element
| metal | wood | water | fire | earth

open Element

/-- Define the overcoming relation. -/
def overcomes : Element → Element → Prop
| metal, wood => true
| wood, earth => true
| earth, water => true
| water, fire => true
| fire, metal => true
| _, _ => false

/-- Define the probability calculation. -/
def probability_not_overcoming : ℚ :=
  let total_combinations := 10    -- C(5, 2)
  let overcoming_combinations := 5
  let not_overcoming_combinations := total_combinations - overcoming_combinations
  not_overcoming_combinations / total_combinations

/-- The proof problem statement. -/
theorem probability_not_overcoming_is_half : probability_not_overcoming = 1 / 2 :=
by
  sorry

end probability_not_overcoming_is_half_l11_11595


namespace sqrt_50_floor_squared_l11_11789

theorem sqrt_50_floor_squared : (⌊Real.sqrt 50⌋ : ℝ)^2 = 49 := by
  have sqrt_50_bounds : 7 < Real.sqrt 50 ∧ Real.sqrt 50 < 8 := by
    split
    · have : Real.sqrt 49 < Real.sqrt 50 := by sorry
      linarith
    · have : Real.sqrt 50 < Real.sqrt 64 := by sorry
      linarith
  have floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := by
    sorry
  rw [floor_sqrt_50]
  norm_num

end sqrt_50_floor_squared_l11_11789


namespace line_through_midpoint_bisects_ellipse_chord_l11_11411

theorem line_through_midpoint_bisects_ellipse_chord :
  (∀ A B : ℝ × ℝ,
    let P := (1/2, 1/2)
    let on_ellipse := λ p : ℝ × ℝ, (p.2 ^ 2 / 9) + (p.1 ^ 2) = 1
    on_ellipse A ∧ on_ellipse B ∧ (A.1 + B.1 = 1) ∧ (A.2 + B.2 = 1) →
    ∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ↔ 9 * x + y - 5 = 0) :=
begin
  sorry
end

end line_through_midpoint_bisects_ellipse_chord_l11_11411


namespace boat_capacity_per_trip_l11_11777

theorem boat_capacity_per_trip (trips_per_day : ℕ) (total_people : ℕ) (days : ℕ) :
  trips_per_day = 4 → total_people = 96 → days = 2 → (total_people / (trips_per_day * days)) = 12 :=
by
  intros
  sorry

end boat_capacity_per_trip_l11_11777


namespace count_even_three_digit_sum_tens_units_is_12_l11_11058

-- Define what it means to be a three-digit number
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n < 1000)

-- Define what it means to be even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the sum of the tens and units digits to be 12
def sum_tens_units_is_12 (n : ℕ) : Prop := 
  let tens := (n / 10) % 10 in
  let units := n % 10 in
  tens + units = 12

-- Count how many such numbers exist
theorem count_even_three_digit_sum_tens_units_is_12 : 
  ∃! n : ℕ, (is_three_digit n) ∧ (is_even n) ∧ (sum_tens_units_is_12 n) = 36 :=
sorry

end count_even_three_digit_sum_tens_units_is_12_l11_11058


namespace fraction_students_say_dislike_actually_like_l11_11748

variables (total_students : ℕ)
           (like_dancing_dislike_say : ℕ)
           (total_dislike_say : ℕ)

-- Definitions based on given conditions
def seventy_percent_likes := 70 / 100 * total_students
def thirty_percent_dislikes := 30 / 100 * total_students
def twenty_five_percent_likes_dislike_say := 25 / 100 * seventy_percent_likes
def eighty_five_percent_dislikes_say := 85 / 100 * thirty_percent_dislikes

-- Calculate number of students who say they dislike dancing
def total_students_dislike_say := twenty_five_percent_likes_dislike_say + eighty_five_percent_dislikes_say

-- Calculate the fraction of students who say they dislike dancing but actually like it
def fraction_like_actually_dislike_say := (twenty_five_percent_likes_dislike_say : ℚ) / total_students_dislike_say

theorem fraction_students_say_dislike_actually_like : 
    fraction_like_actually_dislike_say = 0.407 := 
sorry

end fraction_students_say_dislike_actually_like_l11_11748


namespace maci_pays_total_amount_l11_11150

theorem maci_pays_total_amount :
  let cost_blue_pen := 10 -- cents
  let cost_red_pen := 2 * cost_blue_pen -- cents
  let blue_pen_count := 10
  let red_pen_count := 15
  let total_cost_blue_pens := blue_pen_count * cost_blue_pen -- cents
  let total_cost_red_pens := red_pen_count * cost_red_pen -- cents
  let total_cost := total_cost_blue_pens + total_cost_red_pens -- cents
  total_cost / 100 = 4 -- dollars :=
by
  sorry

end maci_pays_total_amount_l11_11150


namespace discarded_plastic_bags_estimate_l11_11920

noncomputable def average (l : List ℕ) : ℕ :=
  l.foldr (· + ·) 0 / l.length

theorem discarded_plastic_bags_estimate (bags : List ℕ) (students : ℕ) 
  (h_bags : bags = [33, 25, 28, 26, 25, 31]) (h_students : students = 45)
  (h_avg : average bags = 28) : 
  average bags * students = 1260 := 
by
  have h1 : average [33, 25, 28, 26, 25, 31] = 28 := h_avg
  have h2 : 45 = students := Eq.symm h_students
  rw [h1, h2]
  exact rfl

end discarded_plastic_bags_estimate_l11_11920


namespace increasing_interval_l11_11374

noncomputable def f (x : ℝ) : ℝ := x^2 * (2 - x)

theorem increasing_interval :
  ∃ (a b : ℝ), (0 < a ∧ a < b ∧ b = 4 / 3 ∧ ∀ (x : ℝ), a < x ∧ x < b → f' x > 0) :=
sorry

end increasing_interval_l11_11374


namespace division_value_l11_11294

theorem division_value (x : ℝ) (h1 : 2976 / x - 240 = 8) : x = 12 := 
by
  sorry

end division_value_l11_11294


namespace sin_A_and_height_on_AB_l11_11512

theorem sin_A_and_height_on_AB 
  (A B C: ℝ)
  (h_triangle: ∀ A B C, A + B + C = π)
  (h_angle_sum: A + B = 3 * C)
  (h_sin_condition: 2 * Real.sin (A - C) = Real.sin B)
  (h_AB: AB = 5)
  (h_sqrt_two: Real.cos (π / 4) = Real.sin (π / 4) := by norm_num) :
  (Real.sin A = 3 * Real.sqrt 10 / 10) ∧ (height_on_AB = 6) :=
sorry

end sin_A_and_height_on_AB_l11_11512


namespace prime_sequence_bounded_l11_11763

open Nat

-- Definitions based on problem conditions
def is_prime_sequences (p : ℕ → ℕ) :=
  ∀ n : ℕ, Prime (p n)

def prime_divisor_constraint (p : ℕ → ℕ) :=
  ∀ n : ℕ, p (n + 1) = (largest_prime_divisor (p n + p (n-1) + 2008))

-- Lean statement to prove the sequence is bounded
theorem prime_sequence_bounded (p : ℕ → ℕ) (h1 : is_prime_sequences p) (h2 : prime_divisor_constraint p) : 
  ∃ B : ℕ, ∀ n : ℕ, p n ≤ B :=
sorry

end prime_sequence_bounded_l11_11763


namespace probability_of_one_or_two_l11_11725

/-- Represents the number of elements in the first 20 rows of Pascal's Triangle. -/
noncomputable def total_elements : ℕ := 210

/-- Represents the number of ones in the first 20 rows of Pascal's Triangle. -/
noncomputable def number_of_ones : ℕ := 39

/-- Represents the number of twos in the first 20 rows of Pascal's Triangle. -/
noncomputable def number_of_twos : ℕ :=18

/-- Prove that the probability of randomly choosing an element which is either 1 or 2
from the first 20 rows of Pascal's Triangle is 57/210. -/
theorem probability_of_one_or_two (h1 : total_elements = 210)
                                  (h2 : number_of_ones = 39)
                                  (h3 : number_of_twos = 18) :
    39 + 18 = 57 ∧ (57 : ℚ) / 210 = 57 / 210 :=
by {
    sorry
}

end probability_of_one_or_two_l11_11725


namespace number_of_true_propositions_l11_11962

variables {m n : Line} {α β : Plane}

def proposition_1 (α ∥ β : Prop) (m ⊂ α : Prop) : Prop :=
  m ∥ β

def proposition_2 (α ⊥ β : Prop) (m ⊂ α : Prop) : Prop :=
  m ⊥ β

def proposition_3 (m ∥ n : Prop) (n ⊥ α : Prop) : Prop :=
  m ⊥ α

def proposition_4 (m ⊥ α : Prop) (n ⊥ β : Prop) (m ∥ n : Prop) : Prop :=
  α ∥ β

theorem number_of_true_propositions (h1 : proposition_1 α ∥ β m ⊂ α)
  (h2 : ¬ proposition_2 α ⊥ β m ⊂ α)
  (h3 : proposition_3 m ∥ n n ⊥ α)
  (h4 : proposition_4 m ⊥ α n ⊥ β m ∥ n) : 
  3 = 3 :=
begin
  sorry
end

end number_of_true_propositions_l11_11962


namespace joel_age_when_dad_is_twice_l11_11941

-- Given Conditions
def joel_age_now : ℕ := 5
def dad_age_now : ℕ := 32
def age_difference : ℕ := dad_age_now - joel_age_now

-- Proof Problem Statement
theorem joel_age_when_dad_is_twice (x : ℕ) (hx : dad_age_now - joel_age_now = 27) : x = 27 :=
by
  sorry

end joel_age_when_dad_is_twice_l11_11941


namespace we_the_people_cows_l11_11243

theorem we_the_people_cows (W : ℕ) (h1 : ∃ H : ℕ, H = 3 * W + 2) (h2 : W + 3 * W + 2 = 70) : W = 17 :=
sorry

end we_the_people_cows_l11_11243


namespace three_digit_even_sum_12_l11_11033

theorem three_digit_even_sum_12 : 
  ∃ (n : Finset ℕ), 
    n.card = 27 ∧ 
    ∀ x ∈ n, 
      ∃ h t u, 
        (100 * h + 10 * t + u = x) ∧ 
        (h ∈ Finset.range 9 \ {0}) ∧ 
        (u % 2 = 0) ∧ 
        (t + u = 12) := 
sorry

end three_digit_even_sum_12_l11_11033


namespace original_contribution_amount_l11_11919

theorem original_contribution_amount (F : ℕ) (N : ℕ) (C : ℕ) (A : ℕ) 
  (hF : F = 14) (hN : N = 19) (hC : C = 4) : A = 90 :=
by 
  sorry

end original_contribution_amount_l11_11919


namespace root_symmetry_l11_11336

noncomputable def f : ℝ → ℝ := sorry

theorem root_symmetry {f : ℝ → ℝ} 
(h_symm : ∀ x, f (2 + x) = f (2 - x)) 
(h_roots : ∃ ! r1 r2 r3 : ℝ, f r1 = 0 ∧ f r2 = 0 ∧ f r3 = 0) 
(h_one_root_is_zero : ∃ r, r = 0 ∧ f r = 0) : 
(f 0 = 0 ∧ f 2 = 0 ∧ f 4 = 0) :=
sorry

end root_symmetry_l11_11336


namespace largest_value_of_function_Q_is_product_of_zeros_l11_11206

noncomputable def Q (x : ℝ) : ℝ :=
  x^3 - 2 * x^2 - 4 * x + 8

theorem largest_value_of_function_Q_is_product_of_zeros :
  let Q_product_of_zeros := 8,
      Q_at_2 := Q 2,
      sum_of_Q_coeffs := 1 - 2 - 4 + 8,
      sum_of_real_zeros := 2 - 1 + 4 in
  max Q_at_2 (max Q_product_of_zeros (max sum_of_Q_coeffs sum_of_real_zeros)) = Q_product_of_zeros :=
by
  -- This is where the proof would go, but it is skipped as per the instructions.
  sorry

end largest_value_of_function_Q_is_product_of_zeros_l11_11206


namespace sin_A_eq_height_on_AB_l11_11523

-- Defining conditions
variables {A B C : ℝ}
variables (AB : ℝ)

-- Conditions based on given problem
def condition1 : Prop := A + B = 3 * C
def condition2 : Prop := 2 * sin (A - C) = sin B
def condition3 : Prop := A + B + C = Real.pi

-- Question 1: prove that sin A = (3 * sqrt 10) / 10
theorem sin_A_eq:
  condition1 → 
  condition2 → 
  condition3 → 
  sin A = (3 * Real.sqrt 10) / 10 :=
by
  sorry

-- Question 2: given AB = 5, prove the height on side AB is 6
theorem height_on_AB:
  condition1 →
  condition2 →
  condition3 →
  AB = 5 →
  -- Let's construct the height as a function of A, B, and C
  ∃ h, h = 6 :=
by
  sorry

end sin_A_eq_height_on_AB_l11_11523


namespace proof_triangle_properties_l11_11491

variable (A B C : ℝ)
variable (h AB : ℝ)

-- Conditions
def triangle_conditions : Prop :=
  (A + B = 3 * C) ∧ (2 * Real.sin (A - C) = Real.sin B) ∧ (AB = 5)

-- Part 1: Proving sin A
def find_sin_A (h₁ : triangle_conditions A B C h AB) : Prop :=
  Real.sin A = 3 * Real.cos A

-- Part 2: Proving the height on side AB
def find_height_on_AB (h₁ : triangle_conditions A B C h AB) : Prop :=
  h = 6

-- Combined proof statement
theorem proof_triangle_properties (h₁ : triangle_conditions A B C h AB) : 
  find_sin_A A B C h₁ ∧ find_height_on_AB A B C h AB h₁ := 
  by sorry

end proof_triangle_properties_l11_11491


namespace count_even_three_digit_numbers_l11_11041

theorem count_even_three_digit_numbers : 
  let num_even_three_digit_numbers : ℕ := 
    have h1 : (units_digit_possible_pairs : list (ℕ × ℕ)) := 
      [(4, 8), (6, 6), (8, 4)]
    have h2 : (number_of_hundreds_digits : ℕ) := 9
    3 * number_of_hundreds_digits 
in
  num_even_three_digit_numbers = 27 := by
  -- steps skipped
  sorry

end count_even_three_digit_numbers_l11_11041


namespace solution_set_f_pos_l11_11402

noncomputable def f : ℝ → ℝ := sorry -- Definition of the function f(x)

-- Conditions
axiom h1 : ∀ x, f (-x) = -f x     -- f(x) is odd
axiom h2 : f 2 = 0                -- f(2) = 0
axiom h3 : ∀ x > 0, 2 * f x + x * (deriv f x) > 0 -- 2f(x) + xf'(x) > 0 for x > 0

-- Theorem to prove
theorem solution_set_f_pos : { x : ℝ | f x > 0 } = { x : ℝ | x > 2 ∨ (-2 < x ∧ x < 0) } :=
sorry

end solution_set_f_pos_l11_11402


namespace midpoint_divides_AB_externally_l11_11576

variables {A B C C1 M : Type*}
variables {k : ℝ}

/-- Point C divides segment AB internally in the ratio k -/
def point_C_internal : Prop := ∃ (l : ℝ), l = (k / (1 + k)) ∧ sorry

/-- Point C1 divides segment AB externally in the ratio k -/
def point_C1_external : Prop := ∃ (m : ℝ), m = (-k / (1 - k)) ∧ sorry

/-- Definition of the midpoint M of the segment CC1 -/
def midpoint_M : Prop := ∃ (n : ℝ), n = (1/2) * (point_C_internal + point_C1_external) ∧ sorry

/-- Prove that midpoint M divides segment AB externally in the ratio k^2 -/
theorem midpoint_divides_AB_externally : 
  point_C_internal ∧ point_C1_external → ∃ (p : ℝ), p = -k^2 := 
sorry

end midpoint_divides_AB_externally_l11_11576


namespace point_outside_circle_l11_11070

theorem point_outside_circle (a : ℝ) :
  (a > 1) → (a, a) ∉ {p : ℝ × ℝ | (p.1)^2 + (p.2)^2 - 2 * a * p.1 + a^2 - a = 0} :=
by sorry

end point_outside_circle_l11_11070


namespace inequality_solution_ge_11_l11_11766

theorem inequality_solution_ge_11
  (m n : ℝ)
  (h1 : m > 0)
  (h2 : n > 1)
  (h3 : (1/m) + (2/(n-1)) = 1) :
  m + 2 * n ≥ 11 :=
sorry

end inequality_solution_ge_11_l11_11766


namespace number_of_terms_in_arithmetic_sequence_is_39_l11_11884

theorem number_of_terms_in_arithmetic_sequence_is_39 :
  ∀ (a d l : ℤ), 
  d ≠ 0 → 
  a = 128 → 
  d = -3 → 
  l = 14 → 
  ∃ n : ℕ, (a + (↑n - 1) * d = l) ∧ (n = 39) :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_is_39_l11_11884


namespace count_even_three_digit_numbers_with_sum_12_l11_11017

noncomputable def even_three_digit_numbers_with_sum_12 : Prop :=
  let valid_pairs := [(8, 4), (6, 6), (4, 8)] in
  let valid_hundreds := 9 in
  let count_pairs := valid_pairs.length in
  let total_numbers := valid_hundreds * count_pairs in
  total_numbers = 27

theorem count_even_three_digit_numbers_with_sum_12 : even_three_digit_numbers_with_sum_12 :=
by
  sorry

end count_even_three_digit_numbers_with_sum_12_l11_11017


namespace count_even_three_digit_sum_tens_units_is_12_l11_11053

-- Define what it means to be a three-digit number
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n < 1000)

-- Define what it means to be even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the sum of the tens and units digits to be 12
def sum_tens_units_is_12 (n : ℕ) : Prop := 
  let tens := (n / 10) % 10 in
  let units := n % 10 in
  tens + units = 12

-- Count how many such numbers exist
theorem count_even_three_digit_sum_tens_units_is_12 : 
  ∃! n : ℕ, (is_three_digit n) ∧ (is_even n) ∧ (sum_tens_units_is_12 n) = 36 :=
sorry

end count_even_three_digit_sum_tens_units_is_12_l11_11053


namespace stratified_sampling_freshman_l11_11700

def total_students : ℕ := 1800 + 1500 + 1200
def sample_size : ℕ := 150
def freshman_students : ℕ := 1200

/-- if a sample of 150 students is drawn using stratified sampling, 40 students should be drawn from the freshman year -/
theorem stratified_sampling_freshman :
  (freshman_students * sample_size) / total_students = 40 :=
by
  sorry

end stratified_sampling_freshman_l11_11700


namespace range_f_period_f_monotonic_increase_intervals_l11_11827

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.sin x) ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 1 

theorem range_f : Set.Icc 0 4 = Set.range f := sorry

theorem period_f : ∀ x, f (x + Real.pi) = f x := sorry

theorem monotonic_increase_intervals (k : ℤ) :
  ∀ x, (-π / 6 + k * π : ℝ) ≤ x ∧ x ≤ (π / 3 + k * π : ℝ) → 
        ∀ y, f y ≤ f x → y ≤ x := sorry

end range_f_period_f_monotonic_increase_intervals_l11_11827


namespace sin_A_and_height_on_AB_l11_11510

theorem sin_A_and_height_on_AB 
  (A B C: ℝ)
  (h_triangle: ∀ A B C, A + B + C = π)
  (h_angle_sum: A + B = 3 * C)
  (h_sin_condition: 2 * Real.sin (A - C) = Real.sin B)
  (h_AB: AB = 5)
  (h_sqrt_two: Real.cos (π / 4) = Real.sin (π / 4) := by norm_num) :
  (Real.sin A = 3 * Real.sqrt 10 / 10) ∧ (height_on_AB = 6) :=
sorry

end sin_A_and_height_on_AB_l11_11510


namespace count_valid_even_numbers_with_sum_12_l11_11026

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n % 2 = 0) ∧ 
  ((n / 10) % 10 + n % 10 = 12)

theorem count_valid_even_numbers_with_sum_12 :
  (finset.range 1000).filter is_valid_number).card = 27 := by
  sorry

end count_valid_even_numbers_with_sum_12_l11_11026


namespace min_unattainable_score_l11_11665

theorem min_unattainable_score : ∀ (score : ℕ), (¬ ∃ (a b c : ℕ), 
  (a = 1 ∨ a = 3 ∨ a = 8 ∨ a = 12 ∨ a = 0) ∧ 
  (b = 1 ∨ b = 3 ∨ b = 8 ∨ b = 12 ∨ b = 0) ∧ 
  (c = 1 ∨ c = 3 ∨ c = 8 ∨ c = 12 ∨ c = 0) ∧ 
  score = a + b + c) ↔ score = 22 := 
by
  sorry

end min_unattainable_score_l11_11665


namespace min_area_triangle_OAB_l11_11130

noncomputable def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
(a / 4, 0)

noncomputable def area_triangle (O A B : ℝ × ℝ) : ℝ :=
1 / 2 * |(A.1 * (B.2 - O.2) + B.1 * (O.2 - A.2) + O.1 * (A.2 - B.2))|

theorem min_area_triangle_OAB : 
  let F : ℝ × ℝ := focus_of_parabola 3 in
  let A : ℝ × ℝ := (3 / 4, 3 / 2) in
  let B : ℝ × ℝ := (3 / 4, -3 / 2) in
  let O : ℝ × ℝ := (0, 0) in
  area_triangle O A B = 9 / 8 :=
by
  let F := focus_of_parabola 3
  let A := (3 / 4, 3 / 2)
  let B := (3 / 4, -3 / 2)
  let O := (0, 0)
  calc
  area_triangle O A B = 1 / 2 * |(A.1 * (B.2 - O.2) + B.1 * (O.2 - A.2) + O.1 * (A.2 - B.2))| : rfl
                   ... = 1 / 2 * |(3 / 4 * (-3 / 2 - 0) + 3 / 4 * (0 - 3 / 2) + 0 * (3 / 2 - (-3 / 2)))| : by simp
                   ... = 1 / 2 * |(-9 / 8 - 9 / 8)| : by simp
                   ... = 1 / 2 * 18 / 8 : by simp
                   ... = 9 / 8 : by norm_num

end min_area_triangle_OAB_l11_11130


namespace sin_75_eq_l11_11344

-- Define the trigonometric function values
def sin_45 := Real.sin (Real.pi / 4) = Real.sqrt 2 / 2
def cos_30 := Real.cos (Real.pi / 6) = Real.sqrt 3 / 2
def cos_45 := Real.cos (Real.pi / 4) = Real.sqrt 2 / 2
def sin_30 := Real.sin (Real.pi / 6) = 1 / 2

-- Define the sine sum formula
def sine_sum_formula (A B : ℝ) := 
  Real.sin (A + B) = Real.sin A * Real.cos B + Real.cos A * Real.sin B

-- Main theorem that needs to be proved
theorem sin_75_eq : Real.sin (5 * Real.pi / 12) = (Real.sqrt 6 + Real.sqrt 2) / 4 :=
by
  have h1 : sin_45 := sin_45
  have h2 : cos_30 := cos_30
  have h3 : cos_45 := cos_45
  have h4 : sin_30 := sin_30
  have h5 : sine_sum_formula (Real.pi / 4) (Real.pi / 6) := by rfl
  sorry

end sin_75_eq_l11_11344


namespace probability_point_between_C_and_D_l11_11166

theorem probability_point_between_C_and_D :
  ∀ (A B C D E : ℝ), A < B ∧ C < D ∧
  (B - A = 4 * (D - A)) ∧ (B - A = 4 * (B - E)) ∧
  (D - A = C - D) ∧ (C - D = E - C) ∧ (E - C = B - E) →
  (B - A ≠ 0) → 
  (C - D) / (B - A) = 1 / 4 :=
by
  intros A B C D E hAB hNonZero
  sorry

end probability_point_between_C_and_D_l11_11166


namespace friedas_probability_to_corner_l11_11834

-- Define the grid size and positions
def grid_size : Nat := 4
def start_position : ℕ × ℕ := (3, 3)
def corner_positions : List (ℕ × ℕ) := [(1, 1), (1, 4), (4, 1), (4, 4)]

-- Define the number of hops allowed
def max_hops : Nat := 4

-- Define a function to calculate the probability of reaching a corner square
-- within the given number of hops starting from the initial position.
noncomputable def prob_reach_corner (grid_size : ℕ) (start_position : ℕ × ℕ) 
                                     (corner_positions : List (ℕ × ℕ)) 
                                     (max_hops : ℕ) : ℚ :=
  -- Implementation details skipped
  sorry

-- Define the main theorem that states the desired probability
theorem friedas_probability_to_corner : 
  prob_reach_corner grid_size start_position corner_positions max_hops = 17 / 64 :=
sorry

end friedas_probability_to_corner_l11_11834


namespace length_of_bridge_l11_11266

theorem length_of_bridge (train_length : ℝ) (train_speed_kmh : ℝ) (cross_time : ℝ) : 
  train_length = 250 ∧ train_speed_kmh = 72 ∧ cross_time = 30 → 
  (let train_speed_ms := train_speed_kmh * (1000 / 3600) in
   let total_distance := train_speed_ms * cross_time in
   let bridge_length := total_distance - train_length in
   bridge_length = 350) := 
begin
  intros h,
  simp only [if_pos h.1, h.2],
  sorry
end

end length_of_bridge_l11_11266


namespace line_ups_not_next_to_each_other_l11_11237

theorem line_ups_not_next_to_each_other (brothers sisters : ℕ) (h_brothers : brothers = 2) (h_sisters : sisters = 3) : 
  ∃ n : ℕ, n = 72 ∧ (number_of_line_ups brothers sisters n) = n := 
by
  sorry

/-- Definition of the number_of_line_ups to include the constraints of not standing next to each other --/
def number_of_line_ups (brothers sisters : ℕ) : ℕ :=
  if brothers = 2 ∧ sisters = 3 then 72 else 0

end line_ups_not_next_to_each_other_l11_11237


namespace correct_proposition_l11_11390

def proposition_p : Prop :=
  ∀ (x : ℝ), log 2 (x^2 + 4) ≥ 2

def proposition_q : Prop :=
  ∀ (x : ℝ), x ≥ 0 → x^(1/2) = x^(1/2)

theorem correct_proposition : proposition_p ∨ ¬proposition_q :=
by
  sorry

end correct_proposition_l11_11390


namespace exists_f_satisfies_functional_equation_l11_11544

open Rat

noncomputable def f (x : ℚ) := sorry

theorem exists_f_satisfies_functional_equation : 
  ∃ (f : ℚ+ → ℚ+), 
    (∀ (x y : ℚ+), f (x * f y) = f x / y) ∧ 
    (∃ k : ℚ, ∀ x : ℚ+, f x = x ^ k) :=
by
  sorry

end exists_f_satisfies_functional_equation_l11_11544


namespace fractional_part_equation_no_rational_solution_l11_11841

open Real

theorem fractional_part_equation_no_rational_solution (x : ℝ) (hx : x > 0) (h_eq : frac x + frac (1/x) = 1) : ¬rat x := 
sorry

end fractional_part_equation_no_rational_solution_l11_11841


namespace sin_A_eq_height_on_AB_l11_11517

-- Defining conditions
variables {A B C : ℝ}
variables (AB : ℝ)

-- Conditions based on given problem
def condition1 : Prop := A + B = 3 * C
def condition2 : Prop := 2 * sin (A - C) = sin B
def condition3 : Prop := A + B + C = Real.pi

-- Question 1: prove that sin A = (3 * sqrt 10) / 10
theorem sin_A_eq:
  condition1 → 
  condition2 → 
  condition3 → 
  sin A = (3 * Real.sqrt 10) / 10 :=
by
  sorry

-- Question 2: given AB = 5, prove the height on side AB is 6
theorem height_on_AB:
  condition1 →
  condition2 →
  condition3 →
  AB = 5 →
  -- Let's construct the height as a function of A, B, and C
  ∃ h, h = 6 :=
by
  sorry

end sin_A_eq_height_on_AB_l11_11517


namespace find_p_plus_q_l11_11236

def s : ℝ := (15 + 39 + 36) / 2
def areaDEF : ℝ := Real.sqrt (s * (s - 15) * (s - 39) * (s - 36))
def γ (δ : ℝ) : ℝ := 39 * δ
def polynomial_area (θ δ : ℝ) : ℝ := γ(δ) * θ - δ * θ^2
def δ_value : ℝ := 60 / 169

theorem find_p_plus_q (p q : ℕ) (h_rel_prime : Nat.coprime p q) 
  (h_δ_value : (p:ℝ)/q = δ_value) : p + q = 229 := by
  sorry

end find_p_plus_q_l11_11236


namespace total_penalty_kicks_l11_11594

def players : ℕ := 25
def goalies : ℕ := 4
def penalty_kicks_per_goalie (p g : ℕ) := (p - g)

theorem total_penalty_kicks : 
  let p := players in 
  let g := goalies in 
  g * penalty_kicks_per_goalie p g = 96 := 
by
  sorry

end total_penalty_kicks_l11_11594


namespace cos_sum_condition_l11_11129

theorem cos_sum_condition {x y z : ℝ} (h1 : Real.cos x + Real.cos y + Real.cos z = 1) (h2 : Real.sin x + Real.sin y + Real.sin z = 0) : 
  Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) = 1 := 
by 
  sorry

end cos_sum_condition_l11_11129


namespace range_of_x_l11_11899

noncomputable def meaningful_expression (x : ℝ) : Prop :=
  ∃ y, y = 1 / real.sqrt (x - 2)

theorem range_of_x (x : ℝ) : meaningful_expression x ↔ x > 2 :=
by
  sorry

end range_of_x_l11_11899


namespace product_logarithms_eq_3_l11_11366

theorem product_logarithms_eq_3 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) 
  (hterms : ∃ n = 1000, 
    (finset.range (b - 1) \ finset.range a).card = n)
  (hvalue : (finset.range (b - 1) \ finset.range a).val.map 
                (λ k, real.log (k + 1) / real.log k)).prod = 3 : a + b = 1010 :=
sorry

end product_logarithms_eq_3_l11_11366


namespace count_even_three_digit_numbers_with_sum_12_l11_11013

noncomputable def even_three_digit_numbers_with_sum_12 : Prop :=
  let valid_pairs := [(8, 4), (6, 6), (4, 8)] in
  let valid_hundreds := 9 in
  let count_pairs := valid_pairs.length in
  let total_numbers := valid_hundreds * count_pairs in
  total_numbers = 27

theorem count_even_three_digit_numbers_with_sum_12 : even_three_digit_numbers_with_sum_12 :=
by
  sorry

end count_even_three_digit_numbers_with_sum_12_l11_11013


namespace fourth_rectangle_area_l11_11293

-- Define the conditions and prove the area of the fourth rectangle
theorem fourth_rectangle_area (x y z w : ℝ)
  (h_xy : x * y = 24)
  (h_xw : x * w = 35)
  (h_zw : z * w = 42)
  (h_sum : x + z = 21) :
  y * w = 33.777 := 
sorry

end fourth_rectangle_area_l11_11293


namespace sqrt_square_sub_sqrt2_l11_11757

theorem sqrt_square_sub_sqrt2 (h : 1 < Real.sqrt 2) : Real.sqrt ((1 - Real.sqrt 2) ^ 2) = Real.sqrt 2 - 1 :=
by 
  sorry

end sqrt_square_sub_sqrt2_l11_11757


namespace root_of_equation_l11_11439

theorem root_of_equation (a : ℝ) (h : a^2 * (-1)^2 + 2011 * a * (-1) - 2012 = 0) : 
  a = 2012 ∨ a = -1 :=
by sorry

end root_of_equation_l11_11439


namespace Morks_tax_rate_l11_11153

-- Definitions of initial conditions
variables {M : ℝ} {R : ℝ} 

-- Mork's income is M and his tax rate is R
-- Mindy's tax rate is 0.25 and she earns 4 times as Mork
def Mindys_income := 4 * M
def Mindys_tax_rate := 0.25

-- Combined tax rate is 28%
def combined_tax_rate := 0.28

-- Prove that Mork's tax rate is 0.4
theorem Morks_tax_rate (h1: combined_tax_rate = 0.28) (h2: Mindys_tax_rate = 0.25) (h3: Mindys_income = 4 * M): 
  R = 0.4 :=
sorry

end Morks_tax_rate_l11_11153


namespace number_of_pieces_l11_11152

def pan_length : ℕ := 24
def pan_width : ℕ := 30
def brownie_length : ℕ := 3
def brownie_width : ℕ := 4

def area (length : ℕ) (width : ℕ) : ℕ := length * width

theorem number_of_pieces :
  (area pan_length pan_width) / (area brownie_length brownie_width) = 60 := by
  sorry

end number_of_pieces_l11_11152


namespace chocolates_left_l11_11996

-- Definitions based on the conditions
def initially_bought := 3
def gave_away := 2
def additionally_bought := 3

-- Theorem statement to prove
theorem chocolates_left : initially_bought - gave_away + additionally_bought = 4 := by
  -- Proof skipped
  sorry

end chocolates_left_l11_11996


namespace sqrt_50_floor_squared_l11_11787

theorem sqrt_50_floor_squared : (⌊Real.sqrt 50⌋ : ℝ)^2 = 49 := by
  have sqrt_50_bounds : 7 < Real.sqrt 50 ∧ Real.sqrt 50 < 8 := by
    split
    · have : Real.sqrt 49 < Real.sqrt 50 := by sorry
      linarith
    · have : Real.sqrt 50 < Real.sqrt 64 := by sorry
      linarith
  have floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := by
    sorry
  rw [floor_sqrt_50]
  norm_num

end sqrt_50_floor_squared_l11_11787


namespace pascal_triangle_probability_l11_11744

theorem pascal_triangle_probability :
  let total_elements := 20 * 21 / 2,
      ones := 1 + 19 * 2,
      twos := 18 * 2,
      elements := ones + twos in
  (total_elements = 210) →
  (ones = 39) →
  (twos = 36) →
  (elements = 75) →
  (75 / 210) = 5 / 14 :=
by
  intros,
  sorry

end pascal_triangle_probability_l11_11744


namespace floor_sqrt_50_squared_l11_11781

theorem floor_sqrt_50_squared :
  ∃ x : ℕ, x = 7 ∧ ⌊ Real.sqrt 50 ⌋ = x ∧ x^2 = 49 := 
by {
  let x := 7,
  use x,
  have h₁ : 7 < Real.sqrt 50, from sorry,
  have h₂ : Real.sqrt 50 < 8, from sorry,
  have floor_eq : ⌊Real.sqrt 50⌋ = 7, from sorry,
  split,
  { refl },
  { split,
    { exact floor_eq },
    { exact rfl } }
}

end floor_sqrt_50_squared_l11_11781


namespace train_length_l11_11673

theorem train_length (speed_kmph : ℕ) (time_min : ℕ) (equal_length : Prop) (crosses_platform : Prop) :
  speed_kmph = 90 → time_min = 1 → equal_length → crosses_platform →
  let speed_mps := speed_kmph * 1000 / 3600,
      time_sec := time_min * 60,
      distance_crossed := speed_mps * time_sec / 2
  in distance_crossed = 750 :=
by
  intros h_speed h_time h_equal h_crosses
  have speed_mps_def : speed_mps = 25 := by rw [h_speed]; norm_num
  have time_sec_def : time_sec = 60 := by rw [h_time]; norm_num
  have distance_crossed_def : distance_crossed = 750 := by rw [speed_mps_def, time_sec_def]; norm_num
  exact distance_crossed_def

end train_length_l11_11673


namespace sin_A_calculation_height_calculation_l11_11503

variable {A B C : ℝ}

-- Given conditions
def angle_condition : Prop := A + B = 3 * C
def sine_condition : Prop := 2 * sin (A - C) = sin B

-- Part 1: Find sin A
theorem sin_A_calculation (h1 : angle_condition) (h2 : sine_condition) : sin A = 3 * real.sqrt 10 / 10 := sorry

-- Part 2: Given AB = 5, find the height
variable {AB : ℝ}
def AB_value : Prop := AB = 5

theorem height_calculation (h1 : angle_condition) (h2 : sine_condition) (h3 : AB_value) : height = 6 := sorry

end sin_A_calculation_height_calculation_l11_11503


namespace polynomial_binomial_representation_l11_11134

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem polynomial_binomial_representation
  (f : ℤ → ℤ) (n : ℕ)
  (hf : ∃ (p : Polynomial ℤ), Polynomial.degree p = n ∧ ∀ x ∈ Finset.range (n + 1), p.eval x = f x) :
  ∃ (d : Fin (n + 1) → ℤ), ∀ x, f x = ∑ i in Finset.range (n + 1), d i * binom x i :=
sorry

end polynomial_binomial_representation_l11_11134


namespace count_parallel_or_perpendicular_pairs_l11_11329

noncomputable def are_parallel (m1 m2 : ℝ) : Prop :=
  m1 = m2

noncomputable def are_perpendicular (m1 m2 : ℝ) : Prop :=
  m1 * m2 = 1

theorem count_parallel_or_perpendicular_pairs :
  let m1 := (2 : ℝ),
      m2 := (2 : ℝ),
      m3 := (3 : ℝ),
      m4 := (2 / 5 : ℝ),
      m5 := (1 / 3 : ℝ) in
  let parallel_pairs := if are_parallel m1 m2 then 1 else 0,
      perpendicular_pairs := if are_perpendicular m3 m5 then 1 else 0 in
  parallel_pairs + perpendicular_pairs = 2 := sorry

end count_parallel_or_perpendicular_pairs_l11_11329


namespace counterexample_to_proposition_exists_l11_11234

theorem counterexample_to_proposition_exists :
  ∃ (a b : ℤ), a^2 > b^2 ∧ a ≤ b :=
by {
  use [-3, -2],
  split,
  { norm_num },
  { norm_num }
}

end counterexample_to_proposition_exists_l11_11234


namespace sum_of_coefficients_l11_11204

theorem sum_of_coefficients :
  ∃ (A B C D E F G H J K : ℤ),
  (∀ x y : ℤ, 125 * x ^ 8 - 2401 * y ^ 8 = (A * x + B * y) * (C * x ^ 4 + D * x * y + E * y ^ 4) * (F * x + G * y) * (H * x ^ 4 + J * x * y + K * y ^ 4))
  ∧ A + B + C + D + E + F + G + H + J + K = 102 := 
sorry

end sum_of_coefficients_l11_11204


namespace part1_part2_part3_l11_11098

-- Definitions for conditions
def P := (1, -1)
def T0 (x : ℝ) := x^2
def tangent_line (P : ℝ × ℝ) (x : ℝ) := 2 * x = (x^2 + P.2) / (x - P.1)

-- Part 1: Prove values of x1 and x2
theorem part1 (x1 x2 : ℝ) (h1 : x1 < x2) (h2 : tangent_line P x1) (h3 : tangent_line P x2) : 
  x1 = 1 - Real.sqrt 2 ∧ x2 = 1 + Real.sqrt 2 :=
sorry

-- Part 2: Prove area of circle E
theorem part2 (r : ℝ) (h1 : r = 4 / Real.sqrt 5) : 
  π * r^2 = 16 * π / 5 :=
sorry

-- Part 3: Prove maximum area of quadrilateral ABCD
theorem part3 (r : ℝ) (h1 : r = 4 / Real.sqrt 5) (d1 d2 : ℝ) (h2 : d1^2 + d2^2 = 2) : 
  ∃ S : ℝ, S ≤ (22 / 5) ∧ S = 2 * (Real.sqrt (r^2 - d1^2)) * (Real.sqrt (r^2 - d2^2)) :=
sorry

end part1_part2_part3_l11_11098


namespace count_even_three_digit_sum_tens_units_is_12_l11_11059

-- Define what it means to be a three-digit number
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n < 1000)

-- Define what it means to be even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the sum of the tens and units digits to be 12
def sum_tens_units_is_12 (n : ℕ) : Prop := 
  let tens := (n / 10) % 10 in
  let units := n % 10 in
  tens + units = 12

-- Count how many such numbers exist
theorem count_even_three_digit_sum_tens_units_is_12 : 
  ∃! n : ℕ, (is_three_digit n) ∧ (is_even n) ∧ (sum_tens_units_is_12 n) = 36 :=
sorry

end count_even_three_digit_sum_tens_units_is_12_l11_11059


namespace Robie_chocolates_left_l11_11992

def initial_bags : ℕ := 3
def given_away : ℕ := 2
def additional_bags : ℕ := 3

theorem Robie_chocolates_left : (initial_bags - given_away) + additional_bags = 4 :=
by
  sorry

end Robie_chocolates_left_l11_11992


namespace solve_trig_eq_l11_11177

theorem solve_trig_eq (x : ℝ) (k : ℤ) :
  (cos (7 * x) + cos (3 * x) - real.sqrt 2 * cos (10 * x) = sin (7 * x) + sin (3 * x)) →
  (∃ k : ℤ, x = (π / 20) + (π * k / 5) ∨ x = (π / 12) + (2 * π * k / 3) ∨ x = (π / 28) + (2 * π * k / 7)) :=
by
  sorry

end solve_trig_eq_l11_11177


namespace factor_tree_value_l11_11093

theorem factor_tree_value :
  let Q := 5 * 3
  let R := 11 * 2
  let Y := 2 * Q
  let Z := 7 * R
  let X := Y * Z
  X = 4620 :=
by
  sorry

end factor_tree_value_l11_11093


namespace sqrt_50_floor_square_l11_11794

theorem sqrt_50_floor_square : ⌊Real.sqrt 50⌋ ^ 2 = 49 := by
  have h : 7 < Real.sqrt 50 ∧ Real.sqrt 50 < 8 := 
    by sorry
  have floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := 
    by sorry
  show ⌊Real.sqrt 50⌋ ^ 2 = 49
  from calc
    ⌊Real.sqrt 50⌋ ^ 2 = 7 ^ 2 : by rw [floor_sqrt_50]
    ... = 49 : by norm_num

end sqrt_50_floor_square_l11_11794


namespace vertical_asymptote_l11_11369

noncomputable def y (x : ℝ) : ℝ := (3 * x + 1) / (7 * x - 10)

theorem vertical_asymptote (x : ℝ) : (7 * x - 10 = 0) → (x = 10 / 7) :=
by
  intro h
  linarith [h]

#check vertical_asymptote

end vertical_asymptote_l11_11369


namespace find_x_l11_11441

theorem find_x (x y : ℝ) (h : 10 * 3^x = 7^(y + 6) * 2) (hy : y = -6) : x = -Real.log 5 / Real.log 3 := by
  sorry

end find_x_l11_11441


namespace smallest_k_divides_gcd_l11_11540

noncomputable def a_seq : ℕ → ℤ
| 0       := 29
| 1       := a_1 -- We assume a₁ can be any positive integer, placeholder for a proof
| (n + 2) := a_seq (n + 1) + (a_seq n) * (b_seq (n + 1)) ^ 2019

noncomputable def b_seq : ℕ → ℤ
| 0       := 1
| 1       := b_1 -- We assume b₁ can be any positive integer, placeholder for proof
| (n + 2) := b_seq (n + 1) * b_seq n

def divides(x y : ℤ) : Prop := ∃ k : ℤ, y = k * x

open Nat

theorem smallest_k_divides_gcd :
  (∀ a_1 b_1 : ℤ, 0 < a_1 ∧ 0 < b_1 ∧ ¬ divides 29 b_1 →
   ∃ k : ℕ, 0 < k ∧ (29 ∣ gcd (a_seq k) ((b_seq k) - 1)) →
   k = 28) :=
by
  intros a_1 b_1 h
  sorry

end smallest_k_divides_gcd_l11_11540


namespace count_even_three_digit_numbers_sum_tens_units_eq_12_l11_11010

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def is_even (n : ℕ) : Prop := n % 2 = 0
def sum_of_tens_and_units_eq_12 (n : ℕ) : Prop :=
  (n / 10) % 10 + n % 10 = 12

theorem count_even_three_digit_numbers_sum_tens_units_eq_12 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_three_digit n ∧ is_even n ∧ sum_of_tens_and_units_eq_12 n) ∧ S.card = 24 :=
sorry

end count_even_three_digit_numbers_sum_tens_units_eq_12_l11_11010


namespace CB_dot_CD_l11_11471

variables {A B C D : Type} [InnerProductSpace ℝ A] [inner_product_space ℝ A]
variables (hC : ∠A B C = π / 2) (hBC : dist B C = 3) (hTrisect : dist A D = 2/3 * dist A B)

theorem CB_dot_CD : 
  let CB := (B - C) in let CD := CB + (D - C) in
  CB ⬝ CD = 6 :=
by 
  let CA := C - A
  let CD := (1/3) • CA + (2/3) • (C - B)
  let CB := (C - B)
  have hBC_dot_CA_eq_zero : CB ⬝ CA = 0 := by sorry
  have hNorm_CB_sq : ∥CB∥^2 = 9 := by sorry
  let result := (1/3) * (CB ⬝ CA) + (2/3) * (CB ⬝ CB)
  have hResult : result = 6 := by sorry
  exact hResult

end CB_dot_CD_l11_11471


namespace sqrt_50_floor_square_l11_11796

theorem sqrt_50_floor_square : ⌊Real.sqrt 50⌋ ^ 2 = 49 := by
  have h : 7 < Real.sqrt 50 ∧ Real.sqrt 50 < 8 := 
    by sorry
  have floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := 
    by sorry
  show ⌊Real.sqrt 50⌋ ^ 2 = 49
  from calc
    ⌊Real.sqrt 50⌋ ^ 2 = 7 ^ 2 : by rw [floor_sqrt_50]
    ... = 49 : by norm_num

end sqrt_50_floor_square_l11_11796


namespace Sandy_total_money_eq_300_l11_11172

def Sandy_total_expense (watch_cost: ℕ) (shirt_cost: ℕ) (shirt_discount: ℕ) (shoe_cost: ℕ) (shoe_discount: ℕ) : ℕ :=
  let discounted_shirt := shirt_cost - (shirt_cost * shirt_discount / 100)
  let discounted_shoes := shoe_cost - (shoe_cost * shoe_discount / 100)
  watch_cost + discounted_shirt + discounted_shoes

def total_money_taken (remaining: ℕ) (spent_percentage: ℕ) : ℕ :=
  remaining * 100 / (100 - spent_percentage)

theorem Sandy_total_money_eq_300 : 
  (watch_cost : ℕ) (shirt_cost : ℕ) (shirt_discount : ℕ) (shoe_cost : ℕ) (shoe_discount : ℕ) (remaining : ℕ) (spent_percentage : ℕ)
  (h_watch: watch_cost = 50) 
  (h_shirt: shirt_cost = 30) 
  (h_shirt_discount: shirt_discount = 10)
  (h_shoe: shoe_cost = 70)
  (h_shoe_discount: shoe_discount = 20)
  (h_remaining: remaining = 210) 
  (h_spent: spent_percentage = 30) : 
  total_money_taken remaining spent_percentage = 300 := 
by 
  sorry

end Sandy_total_money_eq_300_l11_11172


namespace Robie_chocolates_left_l11_11993

def initial_bags : ℕ := 3
def given_away : ℕ := 2
def additional_bags : ℕ := 3

theorem Robie_chocolates_left : (initial_bags - given_away) + additional_bags = 4 :=
by
  sorry

end Robie_chocolates_left_l11_11993


namespace intersection_product_distance_l11_11404

def line_parametric (t : ℝ) := (x, y) = (-1 - 1/2 * t, 2 + √3/2 * t)
def circle_eq (x y : ℝ) := x^2 + y^2 - x + √3 * y = 0

theorem intersection_product_distance :
  ∃ M N : ℝ × ℝ, (line_parametric M.1 = M ∧ circle_eq M.1 M.2) ∧ 
  (line_parametric N.1 = N ∧ circle_eq N.1 N.2) →
  |dist (M, (-1, 2))| * |dist (N, (-1, 2))| = 6 + 2 * √3 :=
sorry

end intersection_product_distance_l11_11404


namespace count_even_three_digit_numbers_l11_11037

theorem count_even_three_digit_numbers : 
  let num_even_three_digit_numbers : ℕ := 
    have h1 : (units_digit_possible_pairs : list (ℕ × ℕ)) := 
      [(4, 8), (6, 6), (8, 4)]
    have h2 : (number_of_hundreds_digits : ℕ) := 9
    3 * number_of_hundreds_digits 
in
  num_even_three_digit_numbers = 27 := by
  -- steps skipped
  sorry

end count_even_three_digit_numbers_l11_11037


namespace minValue_l11_11375

theorem minValue (x y z : ℝ) (h : 1/x + 2/y + 3/z = 1) : x + y/2 + z/3 ≥ 9 :=
by
  sorry

end minValue_l11_11375


namespace unw_touchable_area_l11_11849

-- Define the conditions
def ball_radius : ℝ := 1
def container_edge_length : ℝ := 5

-- Define the surface area that the ball can never touch
theorem unw_touchable_area : (ball_radius = 1) ∧ (container_edge_length = 5) → 
  let total_unreachable_area := 120
  let overlapping_area := 24
  let unreachable_area := total_unreachable_area - overlapping_area
  unreachable_area = 96 :=
by
  intros
  sorry

end unw_touchable_area_l11_11849


namespace trajectory_equation_l11_11115

theorem trajectory_equation 
  (P : ℝ × ℝ)
  (h : (P.2 / (P.1 + 4)) * (P.2 / (P.1 - 4)) = -4 / 9) :
  P.1 ≠ 4 ∧ P.1 ≠ -4 → P.1^2 / 64 + P.2^2 / (64 / 9) = 1 :=
by
  sorry

end trajectory_equation_l11_11115


namespace sqrt_floor_squared_l11_11815

/-- To evaluate the floor of the square root of 50 squared --/
theorem sqrt_floor_squared :
  (⌊real.sqrt 50⌋ : ℕ)^2 = 49 :=
begin
  -- We know that:
  -- 7^2 = 49 < 50 < 64 = 8^2
  have h1 : 7^2 = 49, by linarith,
  have h2 : 64 = 8^2, by linarith,
  have h3 : (7 : real) < real.sqrt 50, by {
    rw [sqrt_lt],
    exact_mod_cast h1,
  },
  have h4 : real.sqrt 50 < 8, by {
    rw [lt_sqrt],
    exact_mod_cast h2,
  },
  -- Therefore, 7 < sqrt(50) < 8.
  have h5 : (⌊real.sqrt 50⌋ : ℕ) = 7, by {
    rw [nat.floor_eq_iff],
    split,
    { exact_mod_cast h3, },
    { exact_mod_cast h4, },
  },
  -- Thus, ⌊sqrt(50)⌋^2 = 7^2 = 49.
  rw h5,
  exact h1,
end

end sqrt_floor_squared_l11_11815


namespace center_and_radius_of_circle_l11_11202

-- Given conditions
def endpoint1 : (ℝ × ℝ) := (2, -7)
def endpoint2 : (ℝ × ℝ) := (8, 5)

-- Prove the center and radius of circle O
theorem center_and_radius_of_circle :
  let center := ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2) in
  let radius := real.sqrt ((endpoint1.1 - (center.1))^2 + (endpoint1.2 - (center.2))^2) in
  center = (5, -1) ∧ radius = 3 * real.sqrt 5 :=
by {
  sorry
}

end center_and_radius_of_circle_l11_11202


namespace pascal_element_probability_l11_11740

open Nat

def num_elems_first_n_rows (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

def count_ones (n : ℕ) : ℕ :=
  if n = 0 then 1 else if n = 1 then 2 else 2 * (n - 1) + 1

def count_twos (n : ℕ) : ℕ :=
  if n < 2 then 0 else 2 * (n - 2)

def probability_one_or_two (n : ℕ) : ℚ :=
  let total_elems := num_elems_first_n_rows n in
  let ones := count_ones n in
  let twos := count_twos n in
  (ones + twos) / total_elems

theorem pascal_element_probability :
  probability_one_or_two 20 = 5 / 14 :=
by
  sorry

end pascal_element_probability_l11_11740


namespace profit_percentage_approx_l11_11308

noncomputable def CP := 47.50
noncomputable def SP := 65.25

theorem profit_percentage_approx:
  let profit := SP - CP in
  let percentage := (profit / CP) * 100 in
  abs (percentage - 37.37) < 0.01 :=
by {
  sorry
}

end profit_percentage_approx_l11_11308


namespace mutually_exclusive_event_l11_11255

theorem mutually_exclusive_event : 
  ∀ (shoots : ℕ), 
  (shoots = 3) → 
  (∀(at_most_two_hits : ℕ), 
  (at_most_two_hits ≤ 2) → 
  (∀ (all_three_hits : ℕ), 
  (all_three_hits = 3) → 
  ¬ (at_most_two_hits = all_three_hits))).
Proof
  intros shoots h_shoots at_most_two_hits h_at_most all_three_hits h_all.
  sorry

end mutually_exclusive_event_l11_11255


namespace no_outliers_in_dataset_l11_11764

theorem no_outliers_in_dataset :
  let D := [7, 20, 34, 34, 40, 42, 42, 44, 52, 58]
  let Q1 := 34
  let Q3 := 44
  let IQR := Q3 - Q1
  let lower_threshold := Q1 - 1.5 * IQR
  let upper_threshold := Q3 + 1.5 * IQR
  (∀ x ∈ D, x ≥ lower_threshold) ∧ (∀ x ∈ D, x ≤ upper_threshold) →
  ∀ x ∈ D, ¬(x < lower_threshold ∨ x > upper_threshold) :=
by 
  sorry

end no_outliers_in_dataset_l11_11764


namespace ProofSolveProblem_l11_11877

noncomputable def solveProblem : Prop := 
  ∃ (ω φ : ℝ), 
    (∀ x : ℝ, f x = Real.sin (ω * x + φ)) ∧
    ω > 0 ∧ 
    0 ≤ φ ∧ φ ≤ π ∧
    (∀ x : ℝ, f (-x) = f x) ∧
    (∀ x : ℝ, f (3/4 * π + x) = f (3/4 * π - x)) ∧
    (∀ x y : ℝ, 0 ≤ x ∧ x ≤ y ∧ y ≤ π/2 → f x ≤ f y) ∧
    (ω = 2/3 ∨ ω = 2) ∧
    φ = π/2

theorem ProofSolveProblem : solveProblem := 
sorry

end ProofSolveProblem_l11_11877


namespace find_derivative_at_point_A_l11_11227

-- Define the parabola (we only need the point of tangency here)
def parabola_point (f : ℝ → ℝ) : Prop :=
  f 1 = 0

-- Define the tangent line's inclination and its relationship to the derivative
def tangent_line_inclination (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ θ : ℝ, θ = 45 ∧ f' x = Math.tan θ

theorem find_derivative_at_point_A (f : ℝ → ℝ) (h_parabola : parabola_point f)
  (h_tangent : tangent_line_inclination f 1) : f' 1 = 1 :=
sorry

end find_derivative_at_point_A_l11_11227


namespace candidate1_votes_l11_11230

noncomputable def W (V : ℝ) : ℝ := 0.4765573770491803 * V
def votes_candidate1 : ℝ := W 36798

theorem candidate1_votes :
  votes_candidate1 = 17546 := by
  -- Define constants
  let V := 36798
  let W_val := W V
  let other_votes := 7636 + 11628
  -- Calculate total votes
  have total_votes_eq : V = W_val + other_votes :=
    calc V = W_val + 7636 + 11628 : by sorry
  -- Substitute the value of V and simplify
  have W_eq : W_val = 0.4765573770491803 * 36798 := by sorry
  -- Compute the value and round
  have round_W : (0.4765573770491803 * 36798).round = 17546 := by sorry
  -- Result:
  exact round_W

end candidate1_votes_l11_11230


namespace monotonicity_intervals_a_eq_2_range_of_a_l11_11373

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x + a / x - Real.log x

theorem monotonicity_intervals_a_eq_2 :
  (∀ x, f(x, 2) > f(x+ε, 2) ↔ 0 < x ∧ x < 2) ∧ (∀ x, f(x, 2) < f(x+ε, 2) ↔ x > 2) :=
sorry

theorem range_of_a (a : ℝ) (h₀ : a ≤ -1/4) (h₁ : ∀ x ∈ Icc (2:ℝ) Real.exp 1, f x a ≥ -Real.log 2) :
  a ∈ Icc (-4 : ℝ) (-1/4) :=
sorry

end monotonicity_intervals_a_eq_2_range_of_a_l11_11373


namespace arithmetic_sequence_problem_l11_11924

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

theorem arithmetic_sequence_problem :
  (¬ (a 2 + a 8 = a 10) ∧ (∀ n : ℕ, S n ≠ n) ∧ 
  (∀ m n l k : ℕ, (m ≠ 0 ∧ n ≠ 0 ∧ l ≠ 0 ∧ k ≠ 0) → ((m + n = l + k) ↔ (a m + a n = a l + a k))) ∧ 
  (a 1 = 12) ∧ (S 6 = S 11) → (a 9 = 0)) :=
begin
  sorry
end

end arithmetic_sequence_problem_l11_11924


namespace domain_of_f2x_l11_11897

namespace DomainProof

variable {α β : Type} [LinearOrderedField α] {f : α → β} 

def valid_domain (f : α → β) (a b : α) : Prop :=
  ∀ x, a < x ∧ x < b → ∃ y, f x = y

theorem domain_of_f2x (a b : α) (h : valid_domain f (-3 : α) 6) :
  valid_domain (λ x, f (2 * x)) (-3/2) 3 :=
by
  sorry

end DomainProof

end domain_of_f2x_l11_11897


namespace number_of_valid_three_digit_even_numbers_l11_11001

def valid_three_digit_even_numbers (n : ℕ) : Prop :=
  (100 ≤ n) ∧ (n < 1000) ∧ (n % 2 = 0) ∧ (let t := (n / 10) % 10 in
                                           let u := n % 10 in
                                           t + u = 12)

theorem number_of_valid_three_digit_even_numbers : 
  (∃ cnt : ℕ, cnt = 27 ∧ (cnt = (count (λ n, valid_three_digit_even_numbers n) (Ico 100 1000)))) :=
sorry

end number_of_valid_three_digit_even_numbers_l11_11001


namespace sqrt_floor_squared_l11_11814

/-- To evaluate the floor of the square root of 50 squared --/
theorem sqrt_floor_squared :
  (⌊real.sqrt 50⌋ : ℕ)^2 = 49 :=
begin
  -- We know that:
  -- 7^2 = 49 < 50 < 64 = 8^2
  have h1 : 7^2 = 49, by linarith,
  have h2 : 64 = 8^2, by linarith,
  have h3 : (7 : real) < real.sqrt 50, by {
    rw [sqrt_lt],
    exact_mod_cast h1,
  },
  have h4 : real.sqrt 50 < 8, by {
    rw [lt_sqrt],
    exact_mod_cast h2,
  },
  -- Therefore, 7 < sqrt(50) < 8.
  have h5 : (⌊real.sqrt 50⌋ : ℕ) = 7, by {
    rw [nat.floor_eq_iff],
    split,
    { exact_mod_cast h3, },
    { exact_mod_cast h4, },
  },
  -- Thus, ⌊sqrt(50)⌋^2 = 7^2 = 49.
  rw h5,
  exact h1,
end

end sqrt_floor_squared_l11_11814


namespace number_of_valid_seating_arrangements_l11_11778

-- Definitions
def num_chairs : ℕ := 8

def is_valid_arrangement (orig : Fin num_chairs → Fin num_chairs) (perm : Fin num_chairs → Fin num_chairs) : Prop :=
  ∀ i : Fin num_chairs, 
    (perm i ≠ orig i) ∧ 
    ((perm i + 1) % num_chairs ≠ orig i) ∧ 
    ((perm i - 1 + num_chairs) % num_chairs ≠ orig i)

def count_valid_arrangements (orig : Fin num_chairs → Fin num_chairs) : ℕ :=
  (Fin num_chairs → Fin num_chairs).count (is_valid_arrangement orig)

-- Theorem Statement
theorem number_of_valid_seating_arrangements : count_valid_arrangements (id : Fin num_chairs → Fin num_chairs) = 38 :=
  sorry

end number_of_valid_seating_arrangements_l11_11778


namespace proof_problem_l11_11401

noncomputable def f (a b x : ℝ) : ℝ := log a (a^(-x) + 1) + b * x

theorem proof_problem (a : ℝ) (ha : 0 < a) (h_neq1 : a ≠ 1) (hb : b = 1/2) :
  f a b (a + 1/a) > f a b (1/b) :=
by {
  sorry
}

end proof_problem_l11_11401


namespace count_valid_four_digit_numbers_l11_11060

theorem count_valid_four_digit_numbers :
  {N : ℕ | ∃ a x : ℕ, N = 1000 * a + x ∧ N = 8 * x ∧ 1 ≤ a ∧ a ≤ 9 ∧ 100 ≤ x ∧ x ≤ 999}.card = 6 :=
by
  sorry

end count_valid_four_digit_numbers_l11_11060


namespace number_of_male_athletes_l11_11711

theorem number_of_male_athletes
  (total_males : ℕ) (total_females : ℕ) (sample_size : ℕ)
  (total_males = 48) (total_females = 36) (sample_size = 21)
  : (sample_size * total_males) / (total_males + total_females) = 12 := 
by
  sorry

end number_of_male_athletes_l11_11711


namespace solve_logarithmic_equation_l11_11668

theorem solve_logarithmic_equation (x : ℝ) (hx1 : 0 < x) (hx2 : x ≠ 1) (hx3 : x < 10) :
  1 + 2 * log x 2 * log 4 (10 - x) = 2 / log 4 x ↔ (x = 2 ∨ x = 8) :=
by
  sorry

end solve_logarithmic_equation_l11_11668


namespace coordinates_of_P_l11_11102

def P : Prod Int Int := (-1, 2)

theorem coordinates_of_P :
  P = (-1, 2) := 
  by
    -- The proof is omitted as per instructions
    sorry

end coordinates_of_P_l11_11102


namespace marbles_left_l11_11767

theorem marbles_left (initial_marbles : ℕ) (given_marbles : ℕ) (remaining_marbles : ℕ) :
  initial_marbles = 64 → given_marbles = 14 → remaining_marbles = (initial_marbles - given_marbles) → remaining_marbles = 50 :=
by
  intros h_initial h_given h_calculation
  rw [h_initial, h_given] at h_calculation
  exact h_calculation

end marbles_left_l11_11767


namespace necessary_but_not_sufficient_l11_11144

-- Define the sets A, B, and C
def A : Set ℝ := { x | x - 1 > 0 }
def B : Set ℝ := { x | x < 0 }
def C : Set ℝ := { x | x * (x - 2) > 0 }

-- The set A ∪ B in terms of Lean
def A_union_B : Set ℝ := A ∪ B

-- State the necessary and sufficient conditions
theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, x ∈ A_union_B → x ∈ C) ∧ ¬ (∀ x : ℝ, x ∈ C → x ∈ A_union_B) :=
sorry

end necessary_but_not_sufficient_l11_11144


namespace inequality_proof_l11_11136

theorem inequality_proof (x a : ℝ) (hx : 0 < x) (ha : 0 < a) :
  (1 / Real.sqrt (x + 1)) + (1 / Real.sqrt (a + 1)) + Real.sqrt ( (a * x) / (a * x + 8) ) ≤ 2 := 
by {
  sorry
}

end inequality_proof_l11_11136


namespace max_inscribed_rhombus_angle_l11_11916

noncomputable theory

-- Definitions related to the geometric setup
variables {A B C D E K L M N : Type}

-- Assume existence of points and relationships
variables [geometry T] (triangle_ABC : T) (point_D : T) (point_E : T)
  (rhombus_KLMN : T) (angle_BAC : ℝ) (angle_ABC : ℝ) (angle_varphi : ℝ)

-- Given definitions
def angle_bisector_feet (A B C D E : T) : Prop :=
  is_angle_bisector_feet A D C ∧ is_angle_bisector_feet B E C

def rhombus_inscribed (K L M N : T) (polygon : T) : Prop :=
  is_rhombus K L M N ∧ vertices_in_polygon K L M N polygon

def non_obtuse_angle (angle : ℝ) : Prop :=
  angle ≤ π / 2

def max_angle (angle1 angle2 : ℝ) : ℝ := max angle1 angle2

-- The theorem to be proven
theorem max_inscribed_rhombus_angle {A B C D E K L M N : T} 
  (h1 : angle_bisector_feet A B C D E)
  (h2: rhombus_inscribed K L M N (polygon AEDB))
  (h3: non_obtuse_angle angle_varphi)
  (α : ℝ) (β : ℝ) 
  (h4: angle_BAC = α)
  (h5: angle_ABC = β) :
  angle_varphi ≤ max_angle α β :=
sorry

end max_inscribed_rhombus_angle_l11_11916


namespace terry_score_l11_11592

theorem terry_score : 
  ∀ (total_problems right_answer_points wrong_answer_points wrong : ℕ), 
    total_problems = 25 → 
    right_answer_points = 4 → 
    wrong_answer_points = -1 → 
    wrong = 3 → 
    let right := total_problems - wrong in
    let score := right * right_answer_points + wrong * wrong_answer_points in
    score = 85 :=
by
  intros _ _ _ _ total_problems_eq right_points_eq wrong_points_eq wrong_eq;
  simp [total_problems_eq, right_points_eq, wrong_points_eq, wrong_eq];
  -- Calculate the number of right answers
  let right := 25 - 3;
  -- Calculate the total score
  let score := right * 4 + 3 * (-1);
  -- Check that the score is 85
  calc  
    score = right * 4 + 3 * (-1) : by rfl
    ... = (25 - 3) * 4 + 3 * (-1) : by rfl
    ... = 88 - 3 : by simp
    ... = 85 : by rfl

end terry_score_l11_11592


namespace complex_number_properties_l11_11863

open Complex

noncomputable def z : ℂ := (1 - I) / I

theorem complex_number_properties :
  z ^ 2 = 2 * I ∧ Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_number_properties_l11_11863


namespace maci_pays_total_cost_l11_11149

def cost_blue_pen : ℝ := 0.10
def num_blue_pens : ℕ := 10
def num_red_pens : ℕ := 15
def cost_red_pen : ℝ := 2 * cost_blue_pen

def total_cost : ℝ := num_blue_pens * cost_blue_pen + num_red_pens * cost_red_pen

theorem maci_pays_total_cost : total_cost = 4 := by
  -- Proof goes here
  sorry

end maci_pays_total_cost_l11_11149


namespace max_value_f_plus_f_prime_l11_11395

noncomputable def omega := 2
def f (x : ℝ) : ℝ := Real.sin (omega * x + omega)
def f_prime (x : ℝ) : ℝ := (deriv f) x

theorem max_value_f_plus_f_prime :
  ∃ x : ℝ, (0 < x) ∧ (x < omega * Real.pi) ∧ (f x + f_prime x = Real.sqrt 5) := sorry

end max_value_f_plus_f_prime_l11_11395


namespace num_factors_of_M_l11_11434

-- Define the conditions of the problem
def M : ℕ := 2^5 * 3^3 * 5^2 * 7 * 11

-- State the theorem
theorem num_factors_of_M : (finset.range 6).card * (finset.range 4).card * 
  (finset.range 3).card * (finset.range 2).card * (finset.range 2).card = 288 :=
by {
  -- The proof goes here
  sorry
}

end num_factors_of_M_l11_11434


namespace even_three_digit_numbers_l11_11047

theorem even_three_digit_numbers (n : ℕ) :
  (n >= 100 ∧ n < 1000) ∧
  (n % 2 = 0) ∧
  ((n % 100) / 10 + (n % 10) = 12) →
  n = 12 :=
sorry

end even_three_digit_numbers_l11_11047


namespace congruence_mod_l11_11140

open Nat

theorem congruence_mod (a n p : ℕ) (hp_odd : p % 2 = 1) (hp_prime : Prime p) 
  (hn_pos : 0 < n) (ha_pos : 0 < a) (h_congruence : a^p ≡ 1 [MOD p^n]) :
  a ≡ 1 [MOD p^(n-1)] :=
by
  sorry

end congruence_mod_l11_11140


namespace percentage_increase_per_inch_l11_11189

variables (initial_prob : ℝ) (final_prob : ℝ) (initial_height : ℕ) (final_height : ℕ) (baseline_height : ℕ) (baseline_prob : ℝ)
  [decidable (final_prob == baseline_prob + (final_height - baseline_height) * initial_prob)]

-- Conditions stated in the problem
def conditions :=
  initial_prob = 10 ∧
  baseline_prob = 10 ∧
  initial_height = 65 ∧
  final_height = 68 ∧
  baseline_height = 66 ∧
  final_prob = 30

-- The question we need to prove: What is the percentage increase per inch?
theorem percentage_increase_per_inch (x : ℝ) :
  conditions initial_prob final_prob initial_height final_height baseline_height baseline_prob →
  x = (final_prob - baseline_prob) / (final_height - baseline_height) →
  x = 10 :=
sorry

end percentage_increase_per_inch_l11_11189


namespace triangle_side_circumradius_l11_11527

theorem triangle_side_circumradius (a b : ℝ) (cosA : ℝ) (var_R : ℝ) (var_c : ℝ) :
  a = 2 * Real.sqrt 2 ∧
  b = Real.sqrt 6 ∧
  cosA = 1 / 3 ∧
  var_R = 3 / 2 ∧
  var_c = Real.sqrt 6 →
  (let sinA := Real.sqrt (1 - cosA ^ 2) in
   let R := a / (2 * sinA) in
   R = var_R ∧
   let sinB := b / (2 * R) in
   let cosB := Real.sqrt (1 - sinB ^ 2) in
   cosA * cosB + sinA * sinB = cosB ∧
   (let c := Real.sqrt (a ^ 2 + b ^ 2 - 2 * a * b * cosA) in
    c = var_c)) :=
begin
  intros h,
  cases h with ha hb_and_rest,
  cases hb_and_rest with hb rest,
  cases rest with hcosA hR_and_hc,
  cases hR_and_hc with hR hc,
  sorry,
end

end triangle_side_circumradius_l11_11527


namespace find_KB_l11_11095

open Real

noncomputable def right_triangle_bisector (AB BL : ℝ) (h₁ : AB > 0) (h₂ : BL > 0) : ℝ :=
  sqrt (AB * BL)

theorem find_KB :
  ∀ (AB BL : ℝ), AB = 18 → BL = 8 → right_triangle_bisector AB BL (by norm_num) (by norm_num) = 12 :=
begin
  intros AB BL h_ab h_bl,
  rw [right_triangle_bisector, h_ab, h_bl],
  norm_num,
end

end find_KB_l11_11095


namespace count_even_three_digit_sum_tens_units_is_12_l11_11057

-- Define what it means to be a three-digit number
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n < 1000)

-- Define what it means to be even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the sum of the tens and units digits to be 12
def sum_tens_units_is_12 (n : ℕ) : Prop := 
  let tens := (n / 10) % 10 in
  let units := n % 10 in
  tens + units = 12

-- Count how many such numbers exist
theorem count_even_three_digit_sum_tens_units_is_12 : 
  ∃! n : ℕ, (is_three_digit n) ∧ (is_even n) ∧ (sum_tens_units_is_12 n) = 36 :=
sorry

end count_even_three_digit_sum_tens_units_is_12_l11_11057


namespace find_ellipse_equation_l11_11392

-- Definition of the ellipse with given conditions and proof of the equation
theorem find_ellipse_equation
  (a b c : ℝ) (a_pos : a > b) (b_pos : b > 0) (c_pos : c > 0)
  (f1 f2 : ℝ × ℝ) (f1_coords : f1 = (-c, 0)) (f2_coords : f2 = (c, 0))
  (A : ℝ × ℝ) (A_coords : A = (sqrt 3, 1))
  (line_through_origin : ∀ (A : ℝ × ℝ), A ∈ line (0,0) (tan (real.pi / 6)))
  (perpendicular : ∀ (A : ℝ × ℝ) (f1 f2 : ℝ × ℝ), ∠ (A - f1) (A - f2) = real.pi / 2)
  (triangle_area : (1 / 2) * (dist f1 A) * (A.snd - f1.snd) = 2) :
  ∀ (x y : ℝ),
  (x^2 / 6) + (y^2 / 2) = 1 := by
  sorry

end find_ellipse_equation_l11_11392


namespace base_number_is_2_l11_11451

open Real

noncomputable def valid_x (x : ℝ) (n : ℕ) := sqrt (x^n) = 64

theorem base_number_is_2 (x : ℝ) (n : ℕ) (h : valid_x x n) (hn : n = 12) : x = 2 := 
by 
  sorry

end base_number_is_2_l11_11451


namespace problem_statement_l11_11952

def P : Set ℝ := {x | 2^x > 1}
def Q : Set ℝ := {x | log 2 x > 1}

theorem problem_statement : P ∪ Q = P := 
by {
  sorry
}

end problem_statement_l11_11952


namespace part_a_l11_11536

variable {A : Type*} [Ring A]
variable {D : Set A}
hypothesis (hD : ∀ a ∈ D, ¬IsUnit a)
hypothesis (h0 : ∀ a ∈ D, a * a = 0)

theorem part_a (a : A) (x : A) (ha : a ∈ D) : a * x * a = 0 :=
sorry

end part_a_l11_11536


namespace solve_trig_equation_l11_11176

open Real
open Set

noncomputable def is_solution (x : ℝ) : Prop := 
  ∀ k : ℤ, x = ± arccos (3 / 4) + k * pi

theorem solve_trig_equation (x : ℝ) (hx : cos x ≠ 0 ∧ sin x ≠ 0) :
  (3 * sin (3 * x) / sin x - 2 * cos (3 * x) / cos x = 7 * abs (cos x)) ↔ is_solution x :=
sorry

end solve_trig_equation_l11_11176


namespace pneumonia_chronic_disease_confidence_l11_11925

noncomputable def K_square :=
  let a := 40
  let b := 20
  let c := 60
  let d := 80
  let n := a + b + c + d
  (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem pneumonia_chronic_disease_confidence :
  K_square > 7.879 := by
  sorry

end pneumonia_chronic_disease_confidence_l11_11925


namespace no_integer_solutions_l11_11355

theorem no_integer_solutions :
  ∀ (m n : ℤ), (m^3 + 4 * m^2 + 3 * m ≠ 8 * n^3 + 12 * n^2 + 6 * n + 1) := by
  sorry

end no_integer_solutions_l11_11355


namespace possible_integer_roots_l11_11296

theorem possible_integer_roots (a2 a1 : ℤ) : 
  {p : ℤ | (x^3 + a2 * x^2 + a1 * x - 18).eval p = 0} ⊆ 
  {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18} :=
sorry

end possible_integer_roots_l11_11296


namespace count_perfect_cubes_divisible_by_4_l11_11062

theorem count_perfect_cubes_divisible_by_4 (a b : ℤ) 
  (h1 : a = 50)
  (h2 : b = 1500) 
  (h3 : ∀ n : ℤ, n^3 ≥ a → n^3 ≤ b → (∃ k : ℤ, n^3 = k^3 ∧ (∃ m : ℤ, n = 2 * m))) : 
  (finset.filter (λ n : ℤ, n^3 >= a ∧ n^3 <= b ∧ (n % 2 = 0)) (finset.range 20)).card = 4 :=
begin
  sorry -- Proof not required, just statement
end

end count_perfect_cubes_divisible_by_4_l11_11062


namespace triangle_sin_A_and_height_l11_11489

noncomputable theory

variables (A B C : ℝ) (AB : ℝ)
  (h1 : A + B = 3 * C)
  (h2 : 2 * Real.sin (A - C) = Real.sin B)
  (h3 : AB = 5)

theorem triangle_sin_A_and_height :
  Real.sin A = 3 * Real.cos A → 
  sqrt 10 / 10 * Real.sin A = 3 / sqrt (10) / 3 → 
  √10 / 10 = 3/ sqrt 10 /3 → 
  sin (A+B) =sin /sqrt10 →
  (sin (A cv)+ C) = sin( AC ) → 
  ( cos A = sinA 3) 
  ( (10 +25)+5→1= well 5 → B (PS6 S)=H1 (A3+.B9)=
 
 
   
∧   (γ = hA → ) ( (/. );



∧ side /4→ABh3 → 5=HS)  ( →AB3)=sinh1S  

then 
(
  (Real.sin A = 3 * Real.cos A) ^2 )+   
  
(Real.cos A= √ 10/10
  
  Real.sin A2 C(B)= 3√10/10
  
 ) ^(Real.sin A = 5

6)=    
    sorry

end triangle_sin_A_and_height_l11_11489


namespace geometric_sequence_m_value_l11_11843

theorem geometric_sequence_m_value :
  (∃ (a : ℕ → ℝ) (m : ℕ), 
    (∀ n, a n > 0) ∧ 
    (∀ n, a (n + 1) = r * a n) ∧ 
    (r > 0) ∧ 
    (a (m-1) * a (m+1) = 2 * a m) ∧ 
    (∏ k in finset.range (2 * m), a (k + 1) = 2048))
  → 
  m = 6 :=
by
  sorry

end geometric_sequence_m_value_l11_11843


namespace find_units_digit_l11_11531

def is_three_digit (n : ℕ) := 100 ≤ n ∧ n < 1000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n % 100) / 10) + (n % 10)

theorem find_units_digit (n : ℕ) :
  is_three_digit n →
  (is_perfect_square n ∨ is_even n ∨ is_divisible_by_11 n ∨ digit_sum n = 12) ∧
  (¬is_perfect_square n ∨ ¬is_even n ∨ ¬is_divisible_by_11 n ∨ ¬(digit_sum n = 12)) →
  (n % 10 = 4) :=
sorry

end find_units_digit_l11_11531


namespace cos_Z_value_l11_11470

-- The conditions given in the problem
def sin_X := 4 / 5
def cos_Y := 3 / 5

-- The theorem we want to prove
theorem cos_Z_value (sin_X : ℝ) (cos_Y : ℝ) (hX : sin_X = 4/5) (hY : cos_Y = 3/5) : 
  ∃ cos_Z : ℝ, cos_Z = 7 / 25 :=
by
  -- Attach all conditions and solve
  sorry

end cos_Z_value_l11_11470


namespace sin_A_and_height_on_AB_l11_11515

theorem sin_A_and_height_on_AB 
  (A B C: ℝ)
  (h_triangle: ∀ A B C, A + B + C = π)
  (h_angle_sum: A + B = 3 * C)
  (h_sin_condition: 2 * Real.sin (A - C) = Real.sin B)
  (h_AB: AB = 5)
  (h_sqrt_two: Real.cos (π / 4) = Real.sin (π / 4) := by norm_num) :
  (Real.sin A = 3 * Real.sqrt 10 / 10) ∧ (height_on_AB = 6) :=
sorry

end sin_A_and_height_on_AB_l11_11515


namespace acid_volume_16_liters_l11_11262

theorem acid_volume_16_liters (V A_0 B_0 A_1 B_1 : ℝ) 
  (h_initial_ratio : 4 * B_0 = A_0)
  (h_initial_volume : A_0 + B_0 = V)
  (h_remove_mixture : 10 * A_0 / V = A_1)
  (h_remove_mixture_base : 10 * B_0 / V = B_1)
  (h_new_A : A_1 = A_0 - 8)
  (h_new_B : B_1 = B_0 - 2 + 10)
  (h_new_ratio : 2 * B_1 = 3 * A_1) :
  A_0 = 16 :=
by {
  -- Here we will have the proof steps, which are omitted.
  sorry
}

end acid_volume_16_liters_l11_11262


namespace count_convex_cyclic_quadrilaterals_l11_11883

theorem count_convex_cyclic_quadrilaterals :
  let quadrilateralCondition (a b c d : ℕ) := a ≥ 5 ∧ b ≥ 5 ∧ c ≥ 5 ∧ d ≥ 5 ∧ a + b + c + d = 40
  in (finset.univ.filter (λ s : finset (ℕ × ℕ × ℕ × ℕ),
    quadrilateralCondition (s.1, s.2.1.1, s.2.1.2, s.2.2))).card = 680 :=
sorry

end count_convex_cyclic_quadrilaterals_l11_11883


namespace find_a_l11_11546

noncomputable def base25_num : ℕ := 3 * 25^7 + 1 * 25^6 + 4 * 25^5 + 2 * 25^4 + 6 * 25^3 + 5 * 25^2 + 2 * 25^1 + 3 * 25^0

theorem find_a (a : ℤ) (h0 : 0 ≤ a) (h1 : a ≤ 14) : ((base25_num - a) % 12 = 0) → a = 2 := 
sorry

end find_a_l11_11546


namespace periodic_fixed_points_relation_l11_11555

def unit_circle : Set ℂ := { z : ℂ | Complex.abs z = 1 }

def iterated_function {S : Set ℂ} (f : S → S) (k : ℕ) (z : S) : S :=
  nat.rec_on k (λ _, z) (λ n fnz, f (fnz z)) z

theorem periodic_fixed_points_relation {f : unit_circle → unit_circle} :
  ∀ n : ℕ, ∀ F_n P_n : ℕ,
    (P_n = F_n - ∑ d in finset.filter (λ d, d | n ∧ d ≠ n) (finset.range (n + 1)), P_n) :=
sorry

end periodic_fixed_points_relation_l11_11555


namespace sin_A_calculation_height_calculation_l11_11507

variable {A B C : ℝ}

-- Given conditions
def angle_condition : Prop := A + B = 3 * C
def sine_condition : Prop := 2 * sin (A - C) = sin B

-- Part 1: Find sin A
theorem sin_A_calculation (h1 : angle_condition) (h2 : sine_condition) : sin A = 3 * real.sqrt 10 / 10 := sorry

-- Part 2: Given AB = 5, find the height
variable {AB : ℝ}
def AB_value : Prop := AB = 5

theorem height_calculation (h1 : angle_condition) (h2 : sine_condition) (h3 : AB_value) : height = 6 := sorry

end sin_A_calculation_height_calculation_l11_11507


namespace length_of_side_AB_is_4_sqrt_2_l11_11915

theorem length_of_side_AB_is_4_sqrt_2 (A B C : Type) 
  [RealSpace A] [RealSpace B] [RealSpace C] 
  (angle_BAC_eq_45 : ∡ A B C = 45) 
  (BC_is_hypotenuse : (distance B C) = 8):
  (distance A B) = 4 * Real.sqrt 2 :=
by
  sorry

end length_of_side_AB_is_4_sqrt_2_l11_11915


namespace total_apples_correct_l11_11715

variable (X : ℕ)

def Sarah_apples : ℕ := X

def Jackie_apples : ℕ := 2 * Sarah_apples X

def Adam_apples : ℕ := Jackie_apples X + 5

def total_apples : ℕ := Sarah_apples X + Jackie_apples X + Adam_apples X

theorem total_apples_correct : total_apples X = 5 * X + 5 := by
  sorry

end total_apples_correct_l11_11715


namespace hyperbola_condition_l11_11275

theorem hyperbola_condition (m n : ℝ) : (m < 0 ∧ 0 < n) → (∀ x y : ℝ, nx^2 + my^2 = 1 → (n * x^2 - m * y^2 > 0)) :=
by
  sorry

end hyperbola_condition_l11_11275


namespace distance_between_A_and_B_l11_11321

-- Definitions for conditions
def travel_uniformly (carA_pos carB_pos : ℝ) := true  -- Placeholder for uniform travel

def distance_midpoint (dA dB s : ℝ) := 
  (dA = s / 2 + 12) ∧ (dB = s / 2 - 12)  -- Distance from the midpoint condition

def met_at_midpoint (late_time dB_pos : ℝ) :=
  (late_time = 10 / 60) ∧ (dB_pos = 0)  -- Meet at midpoint condition

def distance_reached_B (dA s : ℝ) :=
  dA = s ∧ dA + 20 = s  -- Condition when Car A reaches B

-- Main statement to prove the distance
theorem distance_between_A_and_B :
  ∃ s : ℝ, ∀ dA dB late_time carA_pos carB_pos : ℝ,
    travel_uniformly carA_pos carB_pos →  -- Condition 1
    distance_midpoint dA dB s →           -- Condition 2
    met_at_midpoint late_time dB →        -- Condition 3
    distance_reached_B dA s →             -- Condition 4
    s = 120 :=                            -- Conclusion
begin
  sorry
end

end distance_between_A_and_B_l11_11321


namespace rhoda_coin_toss_l11_11990

theorem rhoda_coin_toss : 
  ∃ n : ℕ, (n.choose 4) * (0.5)^4 * (0.5)^(n-4) = 0.15625 := by
  use 5
  sorry

end rhoda_coin_toss_l11_11990


namespace at_most_four_points_with_condition_l11_11669

-- Define the points and non-zero real numbers condition
def points_with_dist (n : ℕ) (A : Fin n → (ℝ × ℝ)) (k : Fin n → ℝ) : Prop :=
∀ i j : Fin n, i ≠ j → (dist (A i) (A j))^2 = k i + k j

-- Define the main theorem
theorem at_most_four_points_with_condition (n : ℕ) (A : Fin n → (ℝ × ℝ)) (k : Fin n → ℝ) 
  (h_points_dist : points_with_dist n A k) :
  n ≤ 4 ∧ (n = 4 → (1 / k 0 + 1 / k 1 + 1 / k 2 + 1 / k 3 = 0)) :=
sorry

end at_most_four_points_with_condition_l11_11669


namespace parabola_vertex_l11_11775

theorem parabola_vertex (x y : ℝ) : 
  y^2 + 10 * y + 3 * x + 9 = 0 → 
  (∃ v_x v_y, v_x = 16/3 ∧ v_y = -5 ∧ ∀ (y' : ℝ), (x, y) = (v_x, v_y) ↔ (x, y) = (-1 / 3 * ((y' + 5)^2 - 16), y')) :=
by
  sorry

end parabola_vertex_l11_11775


namespace compute_a_plus_b_l11_11365

theorem compute_a_plus_b (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_terms : b - a = 999)
  (h_log_value : ∑ k in finset.range(b - a), log (k + a + 1) / log (k + a) = 3) : 
  a + b = 1010 := 
sorry

end compute_a_plus_b_l11_11365


namespace max_y_value_l11_11181

theorem max_y_value (x y : Int) (h : x * y + 3 * x + 2 * y = -4) : y ≤ -1 :=
by sorry

end max_y_value_l11_11181


namespace find_genuine_coin_l11_11718

-- Definitions: parameterizing the setup of the problem
def num_coins : ℕ := 100
def num_fake_coins : ℕ := 4
def num_real_coins : ℕ := num_coins - num_fake_coins

-- Assume the weights of coins
parameter (weight_real : ℝ) (weight_fake : ℝ)
parameter (h1 : weight_fake < weight_real)

-- Major problem statement: proving the ability to find at least one genuine coin
theorem find_genuine_coin (coins : fin num_coins → ℝ) :
  (∃ g : fin num_coins, coins g = weight_real) →  -- There exists a genuine coin
  (∃ g : fin num_coins, coins g = weight_real) := begin
  intro h,
  sorry -- Proof to be filled in later
end

end find_genuine_coin_l11_11718


namespace value_subtracted_is_five_l11_11447

variable (N x : ℕ)

theorem value_subtracted_is_five
  (h1 : (N - x) / 7 = 7)
  (h2 : (N - 14) / 10 = 4) : x = 5 := by
  sorry

end value_subtracted_is_five_l11_11447


namespace number_of_circles_is_3_l11_11075

-- Define the radius and diameter of the circles
def radius := 4
def diameter := 2 * radius

-- Given the total horizontal length
def total_horizontal_length := 24

-- Number of circles calculated as per the given conditions
def number_of_circles := total_horizontal_length / diameter

-- The proof statement to verify
theorem number_of_circles_is_3 : number_of_circles = 3 := by
  sorry

end number_of_circles_is_3_l11_11075


namespace even_three_digit_numbers_l11_11045

theorem even_three_digit_numbers (n : ℕ) :
  (n >= 100 ∧ n < 1000) ∧
  (n % 2 = 0) ∧
  ((n % 100) / 10 + (n % 10) = 12) →
  n = 12 :=
sorry

end even_three_digit_numbers_l11_11045


namespace eval_expression_l11_11657

theorem eval_expression : 7^3 + 3 * 7^2 + 3 * 7 + 1 = 512 := 
by 
  sorry

end eval_expression_l11_11657


namespace line_intersects_circle_l11_11342

theorem line_intersects_circle (k : ℝ) :
  ∃ x y : ℝ, (kx - y - k +1 = 0) ∧ (x^2 + y^2 = 4) :=
sorry

end line_intersects_circle_l11_11342


namespace happy_valley_arrangements_l11_11593

-- Definitions based on the conditions
def num_chickens : Nat := 4
def num_dogs : Nat := 2
def num_cats : Nat := 5
def total_animals : Nat := num_chickens + num_dogs + num_cats

-- The theorem statement
theorem happy_valley_arrangements : 
  total_animals = 11 ∧ 
  (∃ (g : SymmetricGroup 3), ∃ (a_ch : Equiv.Perm (Fin num_chickens)), ∃ (a_d : Equiv.Perm (Fin num_dogs)), ∃ (a_ca : Equiv.Perm (Fin num_cats)), 
    g • (a_ch • a_d • a_ca = 34560)) := 
by
  sorry

end happy_valley_arrangements_l11_11593


namespace tangent_line_slope_l11_11900

theorem tangent_line_slope (k : ℝ) :
  (∃ m : ℝ, (m^3 - m^2 + m = k * m) ∧ (k = 3 * m^2 - 2 * m + 1)) →
  (k = 1 ∨ k = 3 / 4) :=
by
  -- Proof goes here
  sorry

end tangent_line_slope_l11_11900


namespace probability_reroll_two_dice_is_one_sixth_l11_11125

noncomputable def probability_of_rerolling_two_dice : ℚ :=
  if h : (∀ dice1 dice2 dice3 : ℕ, dice1 ∈ Finset.range 1 7 ∧ dice2 ∈ Finset.range 1 7 ∧ dice3 ∈ Finset.range 1 7 → 
          Pr (sum_9 dice1 dice2 dice3 | choose_to_reroll_two dice1 dice2 dice3) = 1 / 6) 
  then 1 / 6 
  else 0

theorem probability_reroll_two_dice_is_one_sixth :
  probability_of_rerolling_two_dice = 1 / 6 :=
  sorry

end probability_reroll_two_dice_is_one_sixth_l11_11125


namespace sum_of_products_not_zero_l11_11456

theorem sum_of_products_not_zero (f : Fin 25 → Fin 25 → Int) 
  (h : ∀ i j, f i j = 1 ∨ f i j = -1) :
  let row_products := fun i => ∏ j, f i j,
      col_products := fun j => ∏ i, f i j,
      sum_products := (Finset.univ.sum row_products) + (Finset.univ.sum col_products)
  in sum_products ≠ 0 := 
sorry

end sum_of_products_not_zero_l11_11456


namespace incorrect_foci_statement_l11_11421

-- Define the parameters of the given hyperbola to use in conditions
structure Hyperbola :=
  (a b : ℝ)
  (eqn : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)

-- Instantiate the specific given hyperbola
def myHyperbola : Hyperbola :=
  { a := sqrt 10,
    b := sqrt 2,
    eqn := by
      intro x y
      exact x^2 / 10 - y^2 / 2 = 1 }

-- Define a property indicating the foci of the hyperbola are on the y-axis
def are_foci_on_y_axis (h : Hyperbola) : Prop :=
  ∃ c : ℝ, (∃ f1 f2 : ℝ × ℝ, f1 = (0, c) ∧ f2 = (0, -c))

-- The theorem stating the incorrect statement
theorem incorrect_foci_statement : ¬ are_foci_on_y_axis myHyperbola :=
  sorry

end incorrect_foci_statement_l11_11421


namespace product_price_expression_l11_11304

def f (x : ℝ) : ℝ :=
  2000 * (Real.sin ((π / 4) * x - π / 4)) + 7000

theorem product_price_expression :
  (∀ x, 1 ≤ x ∧ x ≤ 12 ∧ 0 < (3 : ℝ) - x ∧ Real.sin ((π / 4) * x - π / 4) ≤ 1)
  ∧ (f 3 = 9000)  -- Condition for March (x = 3)
  ∧ (f 7 = 5000)  -- Condition for July (x = 7)
  ∧ (∀ x, 1 ≤ x ∧ x ≤ 12 → Real.sin ((π / 4) * x - π / 4) ≤ 1)
  ∧ ((∀ x, 1 ≤ x ∧ x ≤ 12 → A sin(ω x + varphi) + b)
    ∧ A > 0 ∧ ω > 0 ∧ abs(varphi) < π/2) → 
  f x = 2 * (real.sin ((π / 4) * x - π / 4)) + 7 :=
sorry

end product_price_expression_l11_11304


namespace relation_uvwr_l11_11589

variable (a d c e : ℝ) (u v w r : ℝ)
hypothesis h1 : a ^ u = d ^ r ∧ d ^ r = c
hypothesis h2 : d ^ v = a ^ w ∧ a ^ w = e

theorem relation_uvwr : r * w = v * u := by
  sorry

end relation_uvwr_l11_11589


namespace count_valid_even_numbers_with_sum_12_l11_11027

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n % 2 = 0) ∧ 
  ((n / 10) % 10 + n % 10 = 12)

theorem count_valid_even_numbers_with_sum_12 :
  (finset.range 1000).filter is_valid_number).card = 27 := by
  sorry

end count_valid_even_numbers_with_sum_12_l11_11027


namespace magnitude_of_sum_of_vectors_l11_11856

noncomputable def a : ℝ × ℝ := (1, real.sqrt 3)
def b : ℝ × ℝ := sorry  -- Assuming we have a vector b such that |b| = 1 and the angle between a and b is 120°

def angle_between (u v : ℝ × ℝ) : ℝ := 
  real.arccos ((u.1 * v.1 + u.2 * v.2) / (real.sqrt (u.1^2 + u.2^2) * real.sqrt (v.1^2 + v.2^2)))

def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)

def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)

theorem magnitude_of_sum_of_vectors (h_angle : angle_between a b = 2 * real.pi / 3) (h_b : magnitude b = 1) :
  magnitude (vector_add a b) = real.sqrt 3 :=
sorry

end magnitude_of_sum_of_vectors_l11_11856


namespace magnitude_2a_sub_b_l11_11431

-- Define the conditions
variables (a b : ℝ × ℝ)
variable (h1 : sqrt (a.1 ^ 2 + a.2 ^ 2) = 1)
variable (h2 : sqrt (b.1 ^ 2 + b.2 ^ 2) = 2)
variable (h3 : a - b = (sqrt 3, sqrt 2))

-- State the theorem
theorem magnitude_2a_sub_b : sqrt ((2 * a.1 - b.1) ^ 2 + (2 * a.2 - b.2) ^ 2) = 2 * sqrt 2 := 
sorry

end magnitude_2a_sub_b_l11_11431


namespace sum_of_reciprocals_of_squares_l11_11765

open Real

theorem sum_of_reciprocals_of_squares {a b c : ℝ} (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = -7) (h3 : a * b * c = -2) :
  1 / a^2 + 1 / b^2 + 1 / c^2 = 73 / 4 :=
by
  sorry

end sum_of_reciprocals_of_squares_l11_11765


namespace sin_A_calculation_height_calculation_l11_11500

variable {A B C : ℝ}

-- Given conditions
def angle_condition : Prop := A + B = 3 * C
def sine_condition : Prop := 2 * sin (A - C) = sin B

-- Part 1: Find sin A
theorem sin_A_calculation (h1 : angle_condition) (h2 : sine_condition) : sin A = 3 * real.sqrt 10 / 10 := sorry

-- Part 2: Given AB = 5, find the height
variable {AB : ℝ}
def AB_value : Prop := AB = 5

theorem height_calculation (h1 : angle_condition) (h2 : sine_condition) (h3 : AB_value) : height = 6 := sorry

end sin_A_calculation_height_calculation_l11_11500


namespace distinct_remainders_l11_11548

theorem distinct_remainders (p : ℕ) (a : Fin p → ℤ) (hp : Nat.Prime p) :
  ∃ k : ℤ, (Finset.univ.image (fun i : Fin p => (a i + i * k) % p)).card ≥ ⌈(p / 2 : ℚ)⌉ :=
sorry

end distinct_remainders_l11_11548


namespace Vasya_grades_l11_11111
open List

theorem Vasya_grades : ∃ grades : List ℕ, 
  grades.length = 5 ∧
  median grades = 4 ∧
  mean grades = 3.8 ∧
  mostFrequent grades = 5 ∧
  grades = [2, 3, 4, 5, 5] := 
by 
  sorry

noncomputable def mean (l : List ℕ) : ℝ :=
  (l.sum : ℝ) / (l.length : ℝ)

noncomputable def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (≤)
  if sorted.length % 2 = 1
  then sorted.get? (sorted.length / 2) |>.get!
  else ((sorted.get? (sorted.length / 2 - 1) |>.get! + sorted.get? (sorted.length / 2) |>.get!) / 2:ℕ)

noncomputable def mostFrequent (l : List ℕ) : ℕ :=
  l.groupBy id (≤) |>.map (λ (g, xs) => (g, xs.length)) |>.qsort (λ p q => p.snd > q.snd) |>.head! |>.fst

end Vasya_grades_l11_11111


namespace count_even_three_digit_sum_tens_units_is_12_l11_11052

-- Define what it means to be a three-digit number
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n < 1000)

-- Define what it means to be even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the sum of the tens and units digits to be 12
def sum_tens_units_is_12 (n : ℕ) : Prop := 
  let tens := (n / 10) % 10 in
  let units := n % 10 in
  tens + units = 12

-- Count how many such numbers exist
theorem count_even_three_digit_sum_tens_units_is_12 : 
  ∃! n : ℕ, (is_three_digit n) ∧ (is_even n) ∧ (sum_tens_units_is_12 n) = 36 :=
sorry

end count_even_three_digit_sum_tens_units_is_12_l11_11052


namespace sin_A_and_height_on_AB_l11_11509

theorem sin_A_and_height_on_AB 
  (A B C: ℝ)
  (h_triangle: ∀ A B C, A + B + C = π)
  (h_angle_sum: A + B = 3 * C)
  (h_sin_condition: 2 * Real.sin (A - C) = Real.sin B)
  (h_AB: AB = 5)
  (h_sqrt_two: Real.cos (π / 4) = Real.sin (π / 4) := by norm_num) :
  (Real.sin A = 3 * Real.sqrt 10 / 10) ∧ (height_on_AB = 6) :=
sorry

end sin_A_and_height_on_AB_l11_11509


namespace joel_age_when_dad_twice_l11_11942

theorem joel_age_when_dad_twice (x : ℕ) (h₁ : x = 22) : 
  let Joel_age := 5 + x 
  in Joel_age = 27 :=
by
  unfold Joel_age
  rw [h₁]
  norm_num

end joel_age_when_dad_twice_l11_11942


namespace floor_sqrt_50_squared_l11_11804

theorem floor_sqrt_50_squared :
  (let x := Real.sqrt 50 in (⌊x⌋₊ : ℕ)^2 = 49) :=
by
  sorry

end floor_sqrt_50_squared_l11_11804


namespace award_distribution_l11_11997

theorem award_distribution (n_awards n_students : ℕ)
  (h_awards : n_awards = 7) (h_students : n_students = 4) :
  ∃ ways : ℕ, ways = 3920 ∧
  (∀ dist : list ℕ, dist.length = n_students ∧
  (∀ k, k ∈ dist → 1 ≤ k) → 
  ∑ k in dist, k = n_awards → 
  -- Total number of distributions
  ways) :=
  sorry

end award_distribution_l11_11997


namespace irreducible_x_m_sub_y_n_l11_11577

open Polynomial

theorem irreducible_x_m_sub_y_n
  (m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n)
  (h_coprime : Nat.coprime m n) :
  Irreducible (Polynomial.X ^ m - Polynomial.C (Polynomial.Y ^ n)) := 
sorry

end irreducible_x_m_sub_y_n_l11_11577


namespace probability_chords_intersect_l11_11302

-- Defining the problem parameters
noncomputable def total_points := 2010

-- The four points chosen arbitrarily on the circle
variables (A B C D: ℕ) (hA: A < total_points)
(hB: B < total_points) (hC: C < total_points) (hD: D < total_points)

-- Defining "evenly placed on the circle" means uniqueness modulo total_points
variable (distinct_points : ∃ h : function.injective (λ (x : ℕ), x % total_points), true )

-- Define the concept of chords intersecting
def chords_intersect (A B C D : ℕ) : Prop :=
  (A < B ∧ C < D ∧ ((A < C ∧ C < B ∧ B < D) ∨ (C < A ∧ A < D ∧ D < B)))

-- The probability calculation
theorem probability_chords_intersect :
  (∃ (p : ℚ), p = 1 / 3) ↔ 
  ((chords_intersect A B C D ∨ chords_intersect C D A B) 
  ∧ ¬(A = B ∨ A = C ∨ A = D ∨ B = C ∨ B = D ∨ C = D)) := sorry

end probability_chords_intersect_l11_11302


namespace sum_A_B_l11_11658

theorem sum_A_B (A B : ℕ) 
  (h1 : (1 / 4 : ℚ) * (1 / 8) = 1 / (4 * A))
  (h2 : 1 / (4 * A) = 1 / B) : A + B = 40 := 
by
  sorry

end sum_A_B_l11_11658


namespace proof_problem_l11_11889

theorem proof_problem (x : ℝ) (h : x < 1) : -2 * x + 2 > 0 :=
by
  sorry

end proof_problem_l11_11889


namespace proof_BANANA_arrangements_and_probability_l11_11341

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements_BANANA : Nat := 
  factorial 6 / (factorial 1 * factorial 3 * factorial 2)

def arrangements_NN_together : Nat := 
  factorial 4 / (factorial 2 * factorial 1)

def probability_NN_together : Rational := 
  arrangements_NN_together / arrangements_BANANA

theorem proof_BANANA_arrangements_and_probability :
  arrangements_BANANA = 180 ∧ probability_NN_together = 1 / 15 := 
by
  sorry

end proof_BANANA_arrangements_and_probability_l11_11341


namespace problem_solution_l11_11305

-- Define the statements
def statement_1 := (∅ = {0})
def statement_2 := (∀ S : Set α, ¬(∅ ⊆ S))
def statement_3 := (∀ S : Set α, S.Subsets.size ≥ 2)
def statement_4 := (∀ S : Set α, ∅ ⊆ S)

-- Define the proof goal
theorem problem_solution : statement_1 = false ∧ statement_2 = false ∧ statement_3 = false ∧ statement_4 = true :=
by
  sorry

end problem_solution_l11_11305


namespace length_of_DE_l11_11188

theorem length_of_DE (base : ℝ) (area_ratio : ℝ) (height_ratio : ℝ) :
  base = 18 → area_ratio = 0.09 → height_ratio = 0.3 → DE = 2 :=
by
  sorry

end length_of_DE_l11_11188


namespace sqrt_2_3_5_not_in_arithmetic_progression_l11_11983

theorem sqrt_2_3_5_not_in_arithmetic_progression :
  ¬ ∃ d : ℝ, (sqrt 3 - sqrt 2 = d) ∧ (sqrt 5 - sqrt 3 = d) := 
sorry

end sqrt_2_3_5_not_in_arithmetic_progression_l11_11983


namespace rate_of_rainfall_on_Monday_l11_11530

theorem rate_of_rainfall_on_Monday (R : ℝ) :
  7 * R + 4 * 2 + 2 * (2 * 2) = 23 → R = 1 := 
by
  sorry

end rate_of_rainfall_on_Monday_l11_11530


namespace player2_can_prevent_player1_winning_l11_11239

def game_turns_numbers :=
  { circle : List ℤ // circle = [1, 2, 3, 4] }

def player1_move (l : List ℤ) : List ℤ := 
  -- This is a high-level description of player 1's move
  sorry -- Specific implementation is omitted

def player2_move (l : List ℤ) : List ℤ := 
  -- This is a high-level description of player 2's move
  sorry -- Specific implementation is omitted

def all_numbers_equal (l : List ℤ) : Prop := 
  ∀ a b ∈ l, a = b

noncomputable def all_numbers_alternate_parity (l : List ℤ) : Prop := 
  ∀ i, (l.nth i).even → (l.nth ((i + 1) % l.length)).odd

theorem player2_can_prevent_player1_winning (initial : game_turns_numbers) :
  ∃ (strategy : List ℤ → List ℤ), 
  ∀ l, ¬ all_numbers_equal (strategy l) :=
by
  sorry

end player2_can_prevent_player1_winning_l11_11239


namespace system_solution_l11_11180

theorem system_solution (x y z : ℝ) 
  (h1 : x - y ≥ z)
  (h2 : x^2 + 4 * y^2 + 5 = 4 * z) :
  (x = 2 ∧ y = -0.5 ∧ z = 2.5) :=
sorry

end system_solution_l11_11180


namespace sin_A_correct_height_on_AB_correct_l11_11473

noncomputable def sin_A (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) : ℝ :=
  Real.sin A

noncomputable def height_on_AB (A B C AB : ℝ) (height : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) (h4 : AB = 5) : ℝ :=
  height

theorem sin_A_correct (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) : 
  sorrry := 
begin
  -- proof omitted
  sorrry
end

theorem height_on_AB_correct (A B C AB : ℝ) (height : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) (h4 : AB = 5) :
  height = 6:= 
begin
  -- proof omitted
  sorrry
end 

end sin_A_correct_height_on_AB_correct_l11_11473


namespace distance_between_intersection_points_l11_11821

theorem distance_between_intersection_points :
  (∃ p1 p2 : ℝ × ℝ,
    (p1.fst^2 + p1.snd^2 = 25 ∧ p1.snd = p1.fst + 5) ∧
    (p2.fst^2 + p2.snd^2 = 25 ∧ p2.snd = p2.fst + 5) ∧
    p1 ≠ p2 ∧
    dist p1 p2 = 5 * Real.sqrt 2) :=
by
  -- Definitions of intersection points
  let p1 := (0 : ℝ, 5 : ℝ)
  let p2 := (-5 : ℝ, 0 : ℝ)
  
  -- Check that the points are on the curves
  have h1 : p1.fst ^ 2 + p1.snd ^ 2 = 25 := by norm_num
  have h2 : p1.snd = p1.fst + 5 := by norm_num
  have h3 : p2.fst ^ 2 + p2.snd ^ 2 = 25 := by norm_num
  have h4 : p2.snd = p2.fst + 5 := by norm_num
  have h5 : p1 ≠ p2 := by norm_num
  
  -- Calculate the distance using the Euclidean distance formula
  have dist_is_5sqrt2 : dist p1 p2 = 5 * Real.sqrt 2 := by
    simp [dist, Real.sqrt]
    norm_num
  
  -- Combine all into the existential statement
  use [p1, p2]
  exact ⟨⟨h1, h2⟩, ⟨h3, h4⟩, h5, dist_is_5sqrt2⟩

end distance_between_intersection_points_l11_11821


namespace lateral_surface_area_of_prism_l11_11975

theorem lateral_surface_area_of_prism 
  (a : ℝ) (α β V : ℝ) :
  let sin (x : ℝ) := Real.sin x 
  ∃ S : ℝ,
    S = (2 * V * sin ((α + β) / 2)) / (a * sin (α / 2) * sin (β / 2)) := 
sorry

end lateral_surface_area_of_prism_l11_11975


namespace geom_progression_sum_ratio_l11_11449

theorem geom_progression_sum_ratio (a : ℝ) (r : ℝ) (m : ℕ) :
  r = 5 →
  (a * (1 - r^6) / (1 - r)) / (a * (1 - r^m) / (1 - r)) = 126 →
  m = 3 := by
  sorry

end geom_progression_sum_ratio_l11_11449


namespace amare_additional_fabric_needed_l11_11160

-- Defining the conditions
def yards_per_dress : ℝ := 5.5
def num_dresses : ℝ := 4
def initial_fabric_feet : ℝ := 7
def yard_to_feet : ℝ := 3

-- The theorem to prove
theorem amare_additional_fabric_needed : 
  (yards_per_dress * num_dresses * yard_to_feet) - initial_fabric_feet = 59 := 
by
  sorry

end amare_additional_fabric_needed_l11_11160


namespace f_constant_91_l11_11379

def f : ℤ → ℝ
| n => if n > 100 then n - 10 else f (f (n + 11))

theorem f_constant_91 (n : ℤ) (h : n ≤ 100) : f n = 91 := 
sorry

end f_constant_91_l11_11379


namespace points_form_parallelogram_and_area_l11_11087

theorem points_form_parallelogram_and_area
  (E F G H : ℝ × ℝ × ℝ)
  (E_def : E = (2, -5, 1))
  (F_def : F = (4, -9, 4))
  (G_def : G = (3, -4, -1))
  (H_def : H = (5, -8, 2)) :
  (∃ u v : ℝ × ℝ × ℝ, 
    (u = (F.1 - E.1, F.2 - E.2, F.3 - E.3)) ∧ 
    (v = (H.1 - G.1, H.2 - G.2, H.3 - G.3)) ∧
    u = v) ∧
  (let w := (F.1 - E.1, F.2 - E.2, F.3 - E.3) in
  let z := (G.1 - E.1, G.2 - E.2, G.3 - E.3) in
  let cross_prod := ((w.2 * z.3 - w.3 * z.2), (w.3 * z.1 - w.1 * z.3), (w.1 * z.2 - w.2 * z.1)) in
  (real.sqrt (cross_prod.1^2 + cross_prod.2^2 + cross_prod.3^2)) = real.sqrt 110) :=
by
  sorry

end points_form_parallelogram_and_area_l11_11087


namespace unique_exponential_function_l11_11384

theorem unique_exponential_function (g : ℝ → ℝ) :
  (∀ x1 x2 : ℝ, g (x1 + x2) = g x1 * g x2) →
  g 1 = 3 →
  (∀ x1 x2 : ℝ, x1 < x2 → g x1 < g x2) →
  ∀ x : ℝ, g x = 3^x :=
by
  sorry

end unique_exponential_function_l11_11384


namespace profit_percentage_is_25_l11_11193

-- Definitions of the variables involved
variables (C S : ℝ)
variables (x : ℕ)

-- Condition given in the problem
def condition1 : Prop := 20 * C = x * S
def condition2 : Prop := x = 16

-- The profit percentage we're aiming to prove
def profit_percentage : ℝ := ((S - C) / C) * 100

-- The theorem to prove
theorem profit_percentage_is_25 (h1 : condition1) (h2 : condition2) :
  profit_percentage C S = 25 :=
sorry

end profit_percentage_is_25_l11_11193


namespace count_even_three_digit_numbers_sum_tens_units_eq_12_l11_11011

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def is_even (n : ℕ) : Prop := n % 2 = 0
def sum_of_tens_and_units_eq_12 (n : ℕ) : Prop :=
  (n / 10) % 10 + n % 10 = 12

theorem count_even_three_digit_numbers_sum_tens_units_eq_12 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_three_digit n ∧ is_even n ∧ sum_of_tens_and_units_eq_12 n) ∧ S.card = 24 :=
sorry

end count_even_three_digit_numbers_sum_tens_units_eq_12_l11_11011


namespace three_points_probability_l11_11953

noncomputable def length_diagonal (a b c d : ℝ) : ℝ :=
  Real.sqrt ((a - c)^2 + (b - d)^2)

noncomputable def side_length (diagonal : ℝ) : ℝ :=
  diagonal / Real.sqrt 2

noncomputable def probability_of_three_points (x y : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2020 ∧ 0 ≤ y ∧ y ≤ 2020 then 1 / 100 else 0

theorem three_points_probability:
  let S_diagonal_length := length_diagonal (1/20) (7/20) (-1/20) (-7/20)
  let S_side_length := side_length S_diagonal_length
  let x y := sorry in
  probability_of_three_points x y = 1 / 100 :=
begin
  sorry
end

end three_points_probability_l11_11953


namespace count_even_three_digit_numbers_l11_11042

theorem count_even_three_digit_numbers : 
  let num_even_three_digit_numbers : ℕ := 
    have h1 : (units_digit_possible_pairs : list (ℕ × ℕ)) := 
      [(4, 8), (6, 6), (8, 4)]
    have h2 : (number_of_hundreds_digits : ℕ) := 9
    3 * number_of_hundreds_digits 
in
  num_even_three_digit_numbers = 27 := by
  -- steps skipped
  sorry

end count_even_three_digit_numbers_l11_11042


namespace sum_of_non_solutions_l11_11139

theorem sum_of_non_solutions (A B C : ℝ) :
  (∀ x : ℝ, (x ≠ -C ∧ x ≠ -10) → (x + B) * (A * x + 40) / ((x + C) * (x + 10)) = 2) →
  (A = 2 ∧ B = 10 ∧ C = 20) →
  (-10 + -20 = -30) :=
by sorry

end sum_of_non_solutions_l11_11139


namespace f_of_3_is_22_l11_11419

-- Define the function and its properties
def f : ℕ+ → ℕ
| ⟨1, _⟩ := 8
| ⟨n + 1, hn⟩ := f ⟨n, Nat.pos_of_ne_zero hn.left.ne'⟩ + 7

-- State the theorem
theorem f_of_3_is_22 : f ⟨3, by decide⟩ = 22 := sorry

end f_of_3_is_22_l11_11419


namespace sin_A_calculation_height_calculation_l11_11504

variable {A B C : ℝ}

-- Given conditions
def angle_condition : Prop := A + B = 3 * C
def sine_condition : Prop := 2 * sin (A - C) = sin B

-- Part 1: Find sin A
theorem sin_A_calculation (h1 : angle_condition) (h2 : sine_condition) : sin A = 3 * real.sqrt 10 / 10 := sorry

-- Part 2: Given AB = 5, find the height
variable {AB : ℝ}
def AB_value : Prop := AB = 5

theorem height_calculation (h1 : angle_condition) (h2 : sine_condition) (h3 : AB_value) : height = 6 := sorry

end sin_A_calculation_height_calculation_l11_11504


namespace f_eq_91_for_n_le_100_l11_11381

noncomputable def f : ℤ → ℝ 
| n => if n > 100 then n - 10 else f(f(n + 11))

theorem f_eq_91_for_n_le_100 (n : ℤ) (h : n ≤ 100) : f n = 91 := 
sorry

end f_eq_91_for_n_le_100_l11_11381


namespace coefficient_of_x2_in_expansion_l11_11465

-- Conditions
def sqrt2_minus_x_pow_6 := (λ x: ℝ, (sqrt 2 - x)^6)

-- Statement to prove
theorem coefficient_of_x2_in_expansion : ∀ x: ℝ, (coeff 2 (sqrt2_minus_x_pow_6 x)) = 60 :=
by {
  intros,
  sorry
}

end coefficient_of_x2_in_expansion_l11_11465


namespace volume_of_liquid_in_tin_l11_11291

-- Define the conditions
def diameter_tin := 10 -- cm
def height_tin := 5 -- cm
def fill_ratio := 2 / 3
def diameter_cone := 4 -- cm
def height_cone := 2 -- cm
def radius_tin := diameter_tin / 2
def radius_cone := diameter_cone / 2
def height_liquid := fill_ratio * height_tin

-- Define the volume formulas
def V_cylinder (r h : ℝ) : ℝ := π * r^2 * h
def V_cone (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

-- Define the volumes
def V_liquid_cylinder := V_cylinder radius_tin height_liquid
def V_cavity := V_cone radius_cone height_cone

-- Define the target volume
def V_liquid := V_liquid_cylinder - V_cavity

-- Proof statement
theorem volume_of_liquid_in_tin : V_liquid = (242 / 3) * π := by
  sorry

end volume_of_liquid_in_tin_l11_11291


namespace range_of_a_l11_11416

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (x0 : ℝ) (h1 : ∀ x, f(x) = real.sqrt (real.log (x+1) + 2*x - a)) 
    (h2 : x0 ∈ set.Icc 0 1) (h3 : f(f(x0)) = x0) : 
    a ∈ set.Icc (-1) (2 + real.log 2) := by 
sorry

end range_of_a_l11_11416


namespace cannot_determine_knight_or_vampire_based_on_answer_to_reliability_cannot_determine_sanity_based_on_answer_to_reliability_l11_11146

-- Define our propositions
def Sane := Prop
def Vampire := Prop
def Reliable := Prop

-- Define constants representing sanity and reliability conditions
def H : Sane := sorry -- Transylvanian is sane (human)
def not_H : Vampire := sorry -- Transylvanian is insane (vampire)
def R : Reliable := sorry -- Transylvanian is reliable
def not_R : ¬Reliable := sorry -- Transylvanian is unreliable

-- Given conditions
axiom sane_or_vampire_imp_reliable : H ∨ not_H → R
axiom insane_or_human_imp_unreliable : not_H ∨ H → not_R
axiom reliable_tells_truth : ∀ (P : Prop), R → (R → P) → P
axiom unreliable_tells_lie : ∀ (P : Prop), not_R → (not_R → ¬P) → ¬P

-- Theorems to prove given
theorem cannot_determine_knight_or_vampire_based_on_answer_to_reliability :
  ¬ (∃ (answer : bool), answer = true ↔ ∀ (P : Prop), R → (R → P) → P ∧ answer = false ↔ ∀ (P : Prop), not_R → (not_R → ¬P) → ¬P) :=
sorry

theorem cannot_determine_sanity_based_on_answer_to_reliability :
  ¬ (∃ (answer : bool), answer = true ↔ H ∧ answer = false ↔ ¬H) :=
sorry

end cannot_determine_knight_or_vampire_based_on_answer_to_reliability_cannot_determine_sanity_based_on_answer_to_reliability_l11_11146


namespace yura_catches_up_l11_11260

theorem yura_catches_up (a : ℕ) (x : ℕ) (h1 : 2 * a * x = a * (x + 5)) : x = 5 :=
by
  sorry

end yura_catches_up_l11_11260


namespace rate_of_elephants_leaving_park_l11_11241

-- Define the premises as constants or axioms
def initial_elephants := 30000
def final_elephants := 28980
def hours_exodus := 4
def hours_entry := 7
def rate_entry := 1500

-- Define the rate of elephants leaving the park
def rate_leave := 2880

-- Prove the equivalent proof problem
theorem rate_of_elephants_leaving_park : 
  ∀ (E : ℕ),
  (E = rate_leave) →
  (final_elephants - initial_elephants) = -(hours_exodus * E) + (hours_entry * rate_entry) :=
begin
  intros E hE,
  rw hE,
  have h1 : final_elephants - initial_elephants = -1020 := by norm_num,
  have h2 : -(hours_exodus * rate_leave) + (hours_entry * rate_entry) = -1020 := by norm_num,
  exact eq.trans h1 h2,
end

end rate_of_elephants_leaving_park_l11_11241


namespace kids_still_awake_l11_11280

theorem kids_still_awake (initial_count remaining_after_first remaining_after_second : ℕ) 
  (h_initial : initial_count = 20)
  (h_first_round : remaining_after_first = initial_count / 2)
  (h_second_round : remaining_after_second = remaining_after_first / 2) : 
  remaining_after_second = 5 := 
by
  sorry

end kids_still_awake_l11_11280


namespace problem_f_sum_l11_11961

def f (x : ℝ) : ℝ :=
if x > 5 then x^3
else if x >= -5 then 2*x - 3
else 5

theorem problem_f_sum : f (-7) + f 0 + f 7 = 345 := by
  sorry

end problem_f_sum_l11_11961


namespace zoo_pandas_l11_11303

-- Defining the conditions
variable (total_couples : ℕ)
variable (pregnant_couples : ℕ)
variable (baby_pandas : ℕ)
variable (total_pandas : ℕ)

-- Given conditions
def paired_mates : Prop := ∃ c : ℕ, c = total_couples

def pregnant_condition : Prop := pregnant_couples = (total_couples * 25) / 100

def babies_condition : Prop := baby_pandas = 2

def total_condition : Prop := total_pandas = total_couples * 2 + baby_pandas

-- The theorem to be proven
theorem zoo_pandas (h1 : paired_mates total_couples)
                   (h2 : pregnant_condition total_couples pregnant_couples)
                   (h3 : babies_condition baby_pandas)
                   (h4 : pregnant_couples = 2) :
                   total_condition total_couples baby_pandas total_pandas :=
by sorry

end zoo_pandas_l11_11303


namespace not_possible_identical_nonzero_remainders_l11_11686

theorem not_possible_identical_nonzero_remainders :
  ¬ ∃ (a : ℕ → ℕ) (r : ℕ), (r > 0) ∧ (∀ i : Fin 100, a i % (a ((i + 1) % 100)) = r) :=
by
  sorry

end not_possible_identical_nonzero_remainders_l11_11686


namespace unique_solution_quadratic_eq_l11_11338

theorem unique_solution_quadratic_eq (q : ℚ) (hq : q ≠ 0) :
  (∀ x : ℚ, q * x^2 - 10 * x + 2 = 0) ↔ q = 12.5 :=
by
  sorry

end unique_solution_quadratic_eq_l11_11338


namespace proof1_proof2_l11_11755
noncomputable def problem1 : ℝ :=
  real.sqrt (25 / 9) - real.rpow (8 / 27) (1 / 3) - real.rpow (real.pi + real.exp 1) 0 + real.rpow (1 / 4) (-1 / 2)

theorem proof1 : problem1 = 2 := by
  sorry

noncomputable def problem2 : ℝ :=
  (real.log 8 + real.log 125 - real.log 2 - real.log 5) / (real.log (real.sqrt 10) * real.log 0.1)

theorem proof2 : problem2 = -4 := by
  sorry

end proof1_proof2_l11_11755


namespace area_of_region_enclosed_by_y_eq_abs_x_and_circle_l11_11771

noncomputable def area_of_smallest_region : ℝ :=
  let r := 3 in
  let theta := real.pi / 2 in
  1 / 2 * r^2 * theta

theorem area_of_region_enclosed_by_y_eq_abs_x_and_circle :
  let r := 3 in
  let theta := real.pi / 2 in
  area_of_smallest_region = 9 * real.pi / 4 :=
by
  sorry

end area_of_region_enclosed_by_y_eq_abs_x_and_circle_l11_11771


namespace sqrt_50_floor_square_l11_11798

theorem sqrt_50_floor_square : ⌊Real.sqrt 50⌋ ^ 2 = 49 := by
  have h : 7 < Real.sqrt 50 ∧ Real.sqrt 50 < 8 := 
    by sorry
  have floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := 
    by sorry
  show ⌊Real.sqrt 50⌋ ^ 2 = 49
  from calc
    ⌊Real.sqrt 50⌋ ^ 2 = 7 ^ 2 : by rw [floor_sqrt_50]
    ... = 49 : by norm_num

end sqrt_50_floor_square_l11_11798


namespace center_distance_correct_l11_11688

noncomputable def ball_diameter : ℝ := 6
noncomputable def ball_radius : ℝ := ball_diameter / 2
noncomputable def R₁ : ℝ := 150
noncomputable def R₂ : ℝ := 50
noncomputable def R₃ : ℝ := 90
noncomputable def R₄ : ℝ := 120
noncomputable def elevation : ℝ := 4

noncomputable def adjusted_R₁ : ℝ := R₁ - ball_radius
noncomputable def adjusted_R₂ : ℝ := R₂ + ball_radius + elevation
noncomputable def adjusted_R₃ : ℝ := R₃ - ball_radius
noncomputable def adjusted_R₄ : ℝ := R₄ - ball_radius

noncomputable def distance_R₁ : ℝ := 1/2 * 2 * Real.pi * adjusted_R₁
noncomputable def distance_R₂ : ℝ := 1/2 * 2 * Real.pi * adjusted_R₂
noncomputable def distance_R₃ : ℝ := 1/2 * 2 * Real.pi * adjusted_R₃
noncomputable def distance_R₄ : ℝ := 1/2 * 2 * Real.pi * adjusted_R₄

noncomputable def total_distance : ℝ := distance_R₁ + distance_R₂ + distance_R₃ + distance_R₄

theorem center_distance_correct : total_distance = 408 * Real.pi := 
  by
  sorry

end center_distance_correct_l11_11688


namespace even_three_digit_numbers_l11_11046

theorem even_three_digit_numbers (n : ℕ) :
  (n >= 100 ∧ n < 1000) ∧
  (n % 2 = 0) ∧
  ((n % 100) / 10 + (n % 10) = 12) →
  n = 12 :=
sorry

end even_three_digit_numbers_l11_11046


namespace necessary_but_not_sufficient_for_parallel_lines_l11_11617

theorem necessary_but_not_sufficient_for_parallel_lines (m : ℝ) : 
  (m = -1/2 ∨ m = 0) ↔ (∀ x y : ℝ, (x + 2*m*y - 1 = 0 ∧ (3*m + 1)*x - m*y - 1 = 0) → false) :=
sorry

end necessary_but_not_sufficient_for_parallel_lines_l11_11617


namespace count_even_three_digit_numbers_l11_11038

theorem count_even_three_digit_numbers : 
  let num_even_three_digit_numbers : ℕ := 
    have h1 : (units_digit_possible_pairs : list (ℕ × ℕ)) := 
      [(4, 8), (6, 6), (8, 4)]
    have h2 : (number_of_hundreds_digits : ℕ) := 9
    3 * number_of_hundreds_digits 
in
  num_even_three_digit_numbers = 27 := by
  -- steps skipped
  sorry

end count_even_three_digit_numbers_l11_11038


namespace cos_alpha_eq_l11_11383

variable (α : ℝ)

theorem cos_alpha_eq :
  (0 < α ∧ α < π / 2) →
  cos (α + π / 3) = -2 / 3 →
  cos α = (sqrt 15 - 2) / 6 :=
by
  intros h1 h2
  sorry

end cos_alpha_eq_l11_11383


namespace largest_possible_radius_l11_11326

-- Definition of the sides of the quadrilateral 
def AB : ℝ := 10
def BC : ℝ := 15
def CD : ℝ := 8
def DA : ℝ := 13

-- Calculate the semi-perimeter s
def s : ℝ := (AB + BC + CD + DA) / 2

-- The largest radius r of a circle that fits inside or on the boundary of the quadrilateral
def r : ℝ := sqrt ((s - AB) * (s - BC) * (s - CD) * (s - DA) / s)

-- The statement to prove
theorem largest_possible_radius : r = sqrt (15600 / 23) := sorry

end largest_possible_radius_l11_11326


namespace collinear_XAD_l11_11315

noncomputable theory
open_locale classical

variables {A B C D X : Type*}
variables [linear_ordered_field A]
variables {Γ : set A} -- Circumcircle of triangle ABC

structure Triangle (A B C : Type*) :=
(circumcircle : set A)
(intersect : set A) -- Points D, X on circumcircle
(tangent_B : set A)
(tangent_C : set A)

def square (a b : Type*) :=
true -- Representation of constructing squares; detailed definition skipped

def collinear (a b c : Type*) :=
∃ l : set A, l a ∧ l b ∧ l c

variables [Triangle triangle]

-- Given conditions
axiom circumcircle_ABC (A B C : set A) : circumcircle Γ -- Triangle ABC
axiom tangents_intersect_at_D (D B C : set A) : intersect D = tangent_B B ∩ tangent_C C
axiom squares_BAGH_ACF (B A G H : set A) (A C E F : set A) : square (BAGH) ∧ square (ACEF)
axiom intersection_EF_HG (X E F G H : set A) : intersect (EF ∩ HG) X -- Intersection of squares

-- Proof problem: prove \collinearity
theorem collinear_XAD (X A D : set A) : collinear X A D := sorry

end collinear_XAD_l11_11315


namespace triangle_sin_A_and_height_l11_11481

noncomputable theory

variables (A B C : ℝ) (AB : ℝ)
  (h1 : A + B = 3 * C)
  (h2 : 2 * Real.sin (A - C) = Real.sin B)
  (h3 : AB = 5)

theorem triangle_sin_A_and_height :
  Real.sin A = 3 * Real.cos A → 
  sqrt 10 / 10 * Real.sin A = 3 / sqrt (10) / 3 → 
  √10 / 10 = 3/ sqrt 10 /3 → 
  sin (A+B) =sin /sqrt10 →
  (sin (A cv)+ C) = sin( AC ) → 
  ( cos A = sinA 3) 
  ( (10 +25)+5→1= well 5 → B (PS6 S)=H1 (A3+.B9)=
 
 
   
∧   (γ = hA → ) ( (/. );



∧ side /4→ABh3 → 5=HS)  ( →AB3)=sinh1S  

then 
(
  (Real.sin A = 3 * Real.cos A) ^2 )+   
  
(Real.cos A= √ 10/10
  
  Real.sin A2 C(B)= 3√10/10
  
 ) ^(Real.sin A = 5

6)=    
    sorry

end triangle_sin_A_and_height_l11_11481


namespace range_of_s_is_ge_11_l11_11362

noncomputable def s (n : ℕ) : ℕ :=
  ∑ p in (n.factors.to_finset), p

theorem range_of_s_is_ge_11 :
  ∀ n : ℕ, n > 1 ∧ 11 ∣ n → ∃ m : ℕ, s n = m ∧ m ≥ 11 :=
by
  sorry

end range_of_s_is_ge_11_l11_11362


namespace range_of_f_smallest_positive_period_of_f_intervals_of_monotonic_increase_of_f_l11_11825

open Real

def f (x : ℝ) := 2 * sin x * sin x + 2 * sqrt 3 * sin x * cos x + 1

theorem range_of_f : Set.Icc 0 4 = Set.range f :=
sorry

theorem smallest_positive_period_of_f : ∀ x : ℝ, f (x + π) = f x :=
sorry

theorem intervals_of_monotonic_increase_of_f (k : ℤ) :
  Set.Icc (-π / 6 + k * π) (π / 3 + k * π) ⊆ {x : ℝ | ∃ (m : ℤ), deriv f x > 0} :=
sorry

end range_of_f_smallest_positive_period_of_f_intervals_of_monotonic_increase_of_f_l11_11825


namespace find_function_expression_find_minimum_value_l11_11605

def f (x : ℝ) (m : ℝ) (a : ℝ) := m + log a x

theorem find_function_expression (m a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) (h₃ : f 8 m a = 2) (h₄ : f 1 m a = -1) :
  f x (-1) 2 = -1 + log 2 x :=
by sorry

def g (x : ℝ) := 2 * f x (-1) 2 - f (x - 1) (-1) 2

theorem find_minimum_value (h₁ : g x = log 2 (x^2 / (x - 1)) - 1) :
  ∃ x₀ : ℝ, g x₀ = 1 ∧ 1 ≤ g x :=
by sorry

end find_function_expression_find_minimum_value_l11_11605


namespace dice_sum_perfect_square_probability_l11_11639

theorem dice_sum_perfect_square_probability :
  let outcomes := [(i, j) | i in Finset.range 1 11, j in Finset.range 1 11]
      sums := (outcomes.map (λ (x : ℕ × ℕ), x.1 + x.2)).to_finset
      perfect_squares := {4, 9, 16}
      favorable_outcomes := sums.filter (λ s, s ∈ perfect_squares)
      total_outcomes := 100
  in favorable_outcomes.card / total_outcomes = 4 / 25 := sorry

end dice_sum_perfect_square_probability_l11_11639


namespace car_travel_distance_l11_11320

-- Defining the conditions as constants
def speed_X := 35 -- miles per hour
def speed_Y := 42 -- miles per hour
def speed_Z := 50 -- miles per hour

def time_diff_X_Z := 117 / 60 -- hours Car X started before Car Z
def time_diff_Y_Z := 72 / 60 -- hours Car Y started before Car Z

-- Defining the proof problem stating that all cars traveled the same distance
theorem car_travel_distance :
    ∃ t : ℝ, 
    let distance := speed_Z * t in 
    distance = speed_X * (t + time_diff_X_Z) ∧ 
    distance = speed_Y * (t + (time_diff_Y_Z + 45 / 60)) ∧ 
    distance = 196.875 :=
begin
    -- To be filled with proof steps
    sorry
end

end car_travel_distance_l11_11320


namespace cotangent_identity_tangent_half_angle_identity_l11_11174

variable (α β γ : Real) (h : α + β + γ = 180)

theorem cotangent_identity (h : α + β + γ = 180) :
  Real.cot α * Real.cot β + Real.cot β * Real.cot γ + Real.cot γ * Real.cot α = 1 :=
sorry

theorem tangent_half_angle_identity (h : α + β + γ = 180) :
  Real.tan (α / 2) * Real.tan (β / 2) + Real.tan (β / 2) * Real.tan (γ / 2) + Real.tan (γ / 2) * Real.tan (α / 2) = 1 :=
sorry

end cotangent_identity_tangent_half_angle_identity_l11_11174


namespace cross_country_meet_winning_scores_l11_11092

theorem cross_country_meet_winning_scores :
  ∃ S : ℕ, S = 18 ∧ 
    ∀ (team1 team2 : Finset ℕ), 
    (team1.card = 6) →
    (team2.card = 6) →
    (∀ x ∈ team1, x ∈ (Finset.range 12).map Nat.succ) →
    (∀ y ∈ team2, y ∈ (Finset.range 12).map Nat.succ) →
    (∀ x y ∈ (team1 ∪ team2), x ≠ y) →
    (team1.sum id + team2.sum id = 78) →
    (∃ w : ℕ, (team1.sum id = w ∨ team2.sum id = w) ∧ w < 39 ∧ 21 ≤ w) → 
    (S = 18) := 
by
  sorry

end cross_country_meet_winning_scores_l11_11092


namespace total_number_of_elements_in_C_l11_11173

-- Definitions for the problem
variables (C D : Set ℕ)
variables (c d : ℕ)

-- Conditions
axiom h1 : c = 3 * d
axiom h2 : (C ∪ D).card = 4500
axiom h3 : (C ∩ D).card = 1200

-- Goal
theorem total_number_of_elements_in_C : c = 4275 :=
by
  -- Translating conditions
  have h : (C ∪ D).card = C.card + D.card - (C ∩ D).card := sorry
  -- Using given conditions
  have h4 : 4500 = c + d - 1200 := sorry
  have h5 : 5700 = c + d := sorry
  have h6 : c = 3 * d := sorry
  have h7 : 5700 = 4 * d := sorry
  have h8 : d = 5700 / 4 := sorry
  have h9 : c = 3 * (5700 / 4) := sorry
  exact by
    sorry -- Connect to the proof concluding c = 4275

end total_number_of_elements_in_C_l11_11173


namespace distance_between_C_and_A_l11_11082

theorem distance_between_C_and_A 
    (A B C : Type)
    (d_AB : ℝ) (d_BC : ℝ)
    (h1 : d_AB = 8)
    (h2 : d_BC = 10) :
    ∃ x : ℝ, 2 ≤ x ∧ x ≤ 18 ∧ ¬ (∃ y : ℝ, y = x) :=
sorry

end distance_between_C_and_A_l11_11082


namespace min_n_for_distance_l11_11678

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def reflect (P Q : ℝ × ℝ) : ℝ × ℝ :=
  let dx := Q.1 - P.1 in
  let dy := Q.2 - P.2 in
  (Q.1 + dx, Q.2 + dy)

def generate_K_set (K : set (ℝ × ℝ)) : set (ℝ × ℝ) :=
  K ∪ {P' | ∃ P Q ∈ K, P' = reflect P Q}

def initial_set : set (ℝ × ℝ) :=
  {((0 : ℝ), 0), (1, 0)}

def K (n : ℕ) : set (ℝ × ℝ) :=
  nat.rec_on n initial_set (λ n K_n, generate_K_set K_n)

theorem min_n_for_distance (A : ℝ × ℝ) (hundred_units : ℝ) :
  ∃ n, ∃ P ∈ K n, distance A P ≥ hundred_units := sorry

end min_n_for_distance_l11_11678


namespace triangle_sin_A_and_height_l11_11488

noncomputable theory

variables (A B C : ℝ) (AB : ℝ)
  (h1 : A + B = 3 * C)
  (h2 : 2 * Real.sin (A - C) = Real.sin B)
  (h3 : AB = 5)

theorem triangle_sin_A_and_height :
  Real.sin A = 3 * Real.cos A → 
  sqrt 10 / 10 * Real.sin A = 3 / sqrt (10) / 3 → 
  √10 / 10 = 3/ sqrt 10 /3 → 
  sin (A+B) =sin /sqrt10 →
  (sin (A cv)+ C) = sin( AC ) → 
  ( cos A = sinA 3) 
  ( (10 +25)+5→1= well 5 → B (PS6 S)=H1 (A3+.B9)=
 
 
   
∧   (γ = hA → ) ( (/. );



∧ side /4→ABh3 → 5=HS)  ( →AB3)=sinh1S  

then 
(
  (Real.sin A = 3 * Real.cos A) ^2 )+   
  
(Real.cos A= √ 10/10
  
  Real.sin A2 C(B)= 3√10/10
  
 ) ^(Real.sin A = 5

6)=    
    sorry

end triangle_sin_A_and_height_l11_11488


namespace ending_number_of_odd_integers_with_odd_factors_l11_11631

theorem ending_number_of_odd_integers_with_odd_factors
  (h : ∃ (S : Finset ℕ), S.card = 5 ∧ ∀ n ∈ S, odd n ∧ (∃ m ∈ S, n = m^2)) :
  ∃ n ∈ (Finset.range 100 \ {0}), n = 81 :=
by
  sorry

end ending_number_of_odd_integers_with_odd_factors_l11_11631


namespace simplify_complex_expression_l11_11175

variables (t : ℝ)

theorem simplify_complex_expression : (2 + t * complex.I) * (2 - t * complex.I) = 4 + t^2 := 
by sorry

end simplify_complex_expression_l11_11175


namespace find_a_b_l11_11901

noncomputable def curve (x a b : ℝ) : ℝ := x^2 + a * x + b

noncomputable def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

theorem find_a_b (a b : ℝ) :
  (∃ (y : ℝ) (x : ℝ), (y = curve x a b) ∧ tangent_line 0 b ∧ (2 * 0 + a = -1) ∧ (0 - b + 1 = 0)) ->
  a = -1 ∧ b = 1 := 
by
  sorry

end find_a_b_l11_11901


namespace solution_triples_l11_11353

noncomputable def find_triples (x y z : ℝ) : Prop :=
  x + y + z = 2008 ∧
  x^2 + y^2 + z^2 = 6024^2 ∧
  (1/x) + (1/y) + (1/z) = 1/2008

theorem solution_triples :
  ∃ (x y z : ℝ), find_triples x y z ∧ (x = 2008 ∧ y = 4016 ∧ z = -4016) :=
sorry

end solution_triples_l11_11353


namespace find_b_and_r_l11_11423

-- Define the hyperbola and its properties
def hyperbola (x y b : ℝ) : Prop :=
  x^2 - y^2 / b^2 = 1

-- Define the eccentricity of the hyperbola
def eccentricity (a c : ℝ) : ℝ :=
  c / a

-- Define the distance formula between a point (x0, y0) and a line ax + by + c = 0
def distance_point_line (x0 y0 a b c : ℝ) : ℝ :=
  abs (a*x0 + b*y0 + c) / sqrt (a^2 + b^2)

-- Problem statement
theorem find_b_and_r :
  ∀ (b r : ℝ),
    (∀ x y, hyperbola x y b) →
    eccentricity 1 (sqrt (1 + b^2)) = sqrt 5 →
    b = 2 →
    distance_point_line 2 1 2 (-1) 0 = r →
    r = (3 * sqrt 5) / 5 := by
  sorry

end find_b_and_r_l11_11423


namespace find_three_digit_number_l11_11263

theorem find_three_digit_number (A B C : ℕ) (h1 : A + B + C = 10) (h2 : B = A + C) (h3 : 100 * C + 10 * B + A = 100 * A + 10 * B + C + 99) : 100 * A + 10 * B + C = 253 :=
by {
  sorry
}

end find_three_digit_number_l11_11263


namespace arithmetic_seq_proof_l11_11850

noncomputable def arithmetic_sequence : Type := ℕ → ℝ

variables (a : ℕ → ℝ) (d : ℝ)

def is_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

variables (a₁ a₂ a₃ a₄ : ℝ)
variables (h1 : a 1 + a 2 = 10)
variables (h2 : a 4 = a 3 + 2)
variables (h3 : is_arithmetic_seq a d)

theorem arithmetic_seq_proof :
  a 3 + a 4 = 18 :=
sorry

end arithmetic_seq_proof_l11_11850


namespace p_range_l11_11956

noncomputable def p (x : ℝ) : ℝ :=
  if is_prime (floor x) then (x + 2)
  else p (greatest_prime_factor (floor x)) + (x + 2 - floor x)

def greatest_prime_factor(n : ℕ) : ℕ := 
  -- A placeholder definition; actual implementation required.
  sorry

def is_prime(n : ℕ) : Prop :=
  -- A placeholder definition; actual implementation required.
  sorry

theorem p_range : 
  (∀ x, 2 ≤ x ∧ x ≤ 15 → p(x) ∈ [4, 10) ∪ [12, 13)) :=
sorry

end p_range_l11_11956


namespace number_of_female_students_school_l11_11317

theorem number_of_female_students_school (T S G_s B_s B G : ℕ) (h1 : T = 1600)
    (h2 : S = 200) (h3 : G_s = B_s - 10) (h4 : G_s + B_s = 200) (h5 : B_s = 105) (h6 : G_s = 95) (h7 : B + G = 1600) : 
    G = 760 :=
by
  sorry

end number_of_female_students_school_l11_11317


namespace problem_solution_l11_11417

-- Definitions for conditions
def zero_at_origin (f : ℝ → ℝ) := f 0 = 0

def period_distance_survives (ω : ℝ) := (∀ x, distance_of_axes (2 * π / ω) = π)

def transformed (x : ℝ) := x - π/6

def g (x : ℝ) := 2 * sin(2 * x - π/6) - 1/2

-- The actual proof statement
theorem problem_solution :
  (∀ f : ℝ → ℝ, zero_at_origin f ∧ 
                period_distance_survives 2) →
  (f x = 2 * sin(2 * x + π/6) - 1 ∧ 
   (∃ k : ℤ, ∀ x, (x ∈  [(π/6 + k * π),(2 * π/3 + k * π)]))) ∧
  (∃ m, ∀ x, (x ∈ [- π/3, m] → g x = 3/2) → m = π/3) :=
sorry

end problem_solution_l11_11417


namespace sin_A_eq_height_on_AB_l11_11525

-- Defining conditions
variables {A B C : ℝ}
variables (AB : ℝ)

-- Conditions based on given problem
def condition1 : Prop := A + B = 3 * C
def condition2 : Prop := 2 * sin (A - C) = sin B
def condition3 : Prop := A + B + C = Real.pi

-- Question 1: prove that sin A = (3 * sqrt 10) / 10
theorem sin_A_eq:
  condition1 → 
  condition2 → 
  condition3 → 
  sin A = (3 * Real.sqrt 10) / 10 :=
by
  sorry

-- Question 2: given AB = 5, prove the height on side AB is 6
theorem height_on_AB:
  condition1 →
  condition2 →
  condition3 →
  AB = 5 →
  -- Let's construct the height as a function of A, B, and C
  ∃ h, h = 6 :=
by
  sorry

end sin_A_eq_height_on_AB_l11_11525


namespace area_increase_percent_l11_11078

theorem area_increase_percent (l w : ℝ) :
  let original_area := l * w
  let new_length := 1.15 * l
  let new_width := 1.25 * w
  let new_area := new_length * new_width
  let increase_percent := ((new_area - original_area) / original_area) * 100
  increase_percent = 43.75 :=
by
  let original_area := l * w
  let new_length := 1.15 * l
  let new_width := 1.25 * w
  let new_area := new_length * new_width
  let increase_percent := ((new_area - original_area) / original_area) * 100
  calc
    increase_percent = ((new_area - original_area) / original_area) * 100 : rfl
    ... = ((1.15 * l * 1.25 * w - l * w) / (l * w)) * 100 : by rw [← mul_assoc, ← mul_assoc, mul_comm (1.15), mul_assoc]
    ... = 43.75 : sorry

end area_increase_percent_l11_11078


namespace Q_inverse_is_zero_matrix_l11_11543

noncomputable def Q_projection_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  let a := 1 / Real.sqrt 26
  let b := 5 / Real.sqrt 26
  Matrix.vecCons (Matrix.vecCons (a * a) (Matrix.vecCons (a * b) Matrix.nil))
                 (Matrix.vecCons (Matrix.vecCons (a * b) (b * b)) Matrix.nil)
                 
theorem Q_inverse_is_zero_matrix :
  let Q := Q_projection_matrix
  det Q = 0 → ∀ Q_inv : Matrix (Fin 2) (Fin 2) ℝ, Q.inv = 0 := by
  intros Q hQ_det0
  sorry

end Q_inverse_is_zero_matrix_l11_11543


namespace road_signs_at_first_intersection_l11_11299

theorem road_signs_at_first_intersection (x : ℕ) 
    (h1 : x + (x + x / 4) + 2 * (x + x / 4) + (2 * (x + x / 4) - 20) = 270) : 
    x = 40 := 
sorry

end road_signs_at_first_intersection_l11_11299


namespace sin_A_eq_height_on_AB_l11_11518

-- Defining conditions
variables {A B C : ℝ}
variables (AB : ℝ)

-- Conditions based on given problem
def condition1 : Prop := A + B = 3 * C
def condition2 : Prop := 2 * sin (A - C) = sin B
def condition3 : Prop := A + B + C = Real.pi

-- Question 1: prove that sin A = (3 * sqrt 10) / 10
theorem sin_A_eq:
  condition1 → 
  condition2 → 
  condition3 → 
  sin A = (3 * Real.sqrt 10) / 10 :=
by
  sorry

-- Question 2: given AB = 5, prove the height on side AB is 6
theorem height_on_AB:
  condition1 →
  condition2 →
  condition3 →
  AB = 5 →
  -- Let's construct the height as a function of A, B, and C
  ∃ h, h = 6 :=
by
  sorry

end sin_A_eq_height_on_AB_l11_11518


namespace sequence_properties_l11_11848

noncomputable def x1 (β2 : ℝ) (α : ℝ) : ℝ := β2 / (1 - α)
noncomputable def x2 (β1 : ℝ) (β2 : ℝ) (α : ℝ) : ℝ := (β1 - 2 * (β2 / (1 - α))) / (1 - α)
noncomputable def x3 (β0 : ℝ) (β1 : ℝ) (β2 : ℝ) (α : ℝ) : ℝ := (β0 - (β2 / (1 - α)) - ((β1 - 2 * (β2 / (1 - α))) / (1 - α))) / (1 - α)

theorem sequence_properties 
  (α β0 β1 β2 a1 : ℝ) (n : ℕ) (hα : α ≠ 1) (hαβ2 : α * β2 ≠ 0) (h_n_pos : 0 < n) :
  let x1 := x1 β2 α
  let x2 := x2 β1 β2 α
  let x3 := x3 β0 β1 β2 α in
  let an := (a1 - (x1 + x2 + x3)) * α^((n : ℕ) - 1) + x1 * (n : ℕ)^2 + x2 * (n : ℕ) + x3 in
  let Sn := ((a1 - (x1 + x2 + x3)) * ((1 - α^n) / (1 - α)) + 
            x1 * ((n * (n + 1) * (2 * n + 1)) / 6) + 
            x2 * ((n * (n + 1)) / 2) + 
            x3 * (n : ℕ)) in
  (a_{n+1} = α a_{n} + β2 n^2 + β1 n + β0) → 
  (a_n = an ∧ S_n = Sn) := 
  sorry

end sequence_properties_l11_11848


namespace positive_diff_between_two_numbers_l11_11624

theorem positive_diff_between_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 20) :  |x - y| = 2 := 
by
  sorry

end positive_diff_between_two_numbers_l11_11624


namespace probability_of_one_or_two_in_pascal_l11_11726

def pascal_triangle_element_probability : ℚ :=
  let total_elements := 210 -- sum of the elements in the first 20 rows
  let ones_count := 39      -- total count of 1s in the first 20 rows
  let twos_count := 36      -- total count of 2s in the first 20 rows
  let favorable_elements := ones_count + twos_count
  favorable_elements / total_elements

theorem probability_of_one_or_two_in_pascal (n : ℕ) (h : n = 20) :
  pascal_triangle_element_probability = 5 / 14 := by
  rw [h]
  dsimp [pascal_triangle_element_probability]
  sorry

end probability_of_one_or_two_in_pascal_l11_11726


namespace sum_of_areas_of_6_circles_l11_11707

def radius_sequence (n : ℕ) : ℚ :=
  2 * (1 / 3)^(n - 1)

def area_of_circle (r : ℚ) : ℚ :=
  π * r^2

def area_sequence (n : ℕ) : ℚ :=
  area_of_circle (radius_sequence n)

def finite_geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_areas_of_6_circles :
  finite_geometric_series_sum (area_sequence 1) (1/9) 6 = 9 * π / 2 :=
by
  sorry

end sum_of_areas_of_6_circles_l11_11707


namespace solve_system_of_equations_l11_11178

theorem solve_system_of_equations (a b : ℝ) (h1 : a^2 ≠ 1) (h2 : b^2 ≠ 1) (h3 : a ≠ b) : 
  (∃ x y : ℝ, 
    (x - y) / (1 - x * y) = 2 * a / (1 + a^2) ∧ (x + y) / (1 + x * y) = 2 * b / (1 + b^2) ∧
    ((x = (a * b + 1) / (a + b) ∧ y = (a * b - 1) / (a - b)) ∨ 
     (x = (a + b) / (a * b + 1) ∧ y = (a - b) / (a * b - 1)))) :=
by
  sorry

end solve_system_of_equations_l11_11178


namespace harmonic_mean_trapezoid_l11_11636

structure Trapezoid (α : Type) [LinearOrder α] :=
(a b : α)  -- the lengths of bases AB and CD
(h_par : a < b) -- AB is parallel to CD

variables {α : Type} [LinearOrder α]

noncomputable def PQ {T : Trapezoid α} : α :=
  (2 * T.a * T.b) / (T.a + T.b)

theorem harmonic_mean_trapezoid (T : Trapezoid α) :
  PQ = (2 * T.a * T.b) / (T.a + T.b) :=
by sorry

end harmonic_mean_trapezoid_l11_11636


namespace correct_arrangement_l11_11747

-- Define the original polynomial
def original_polynomial : Polynomial ℕ := 1 * X ^ 3 - 5 * X * Y ^ 2 - 7 * Y ^ 3 + 8 * X ^ 2 * Y

-- Define the target polynomial arranged in ascending powers of x
def target_polynomial : Polynomial ℕ := -7 * Y ^ 3 - 5 * X * Y ^ 2 + 8 * X ^ 2 * Y + 1 * X ^ 3

-- Proof statement
theorem correct_arrangement :
  (arrange_in_ascending_powers_x original_polynomial) = target_polynomial :=
sorry -- Proof will be provided here

end correct_arrangement_l11_11747


namespace number_of_levels_l11_11760

-- Definitions of the conditions
def blocks_per_step : ℕ := 3
def steps_per_level : ℕ := 8
def total_blocks_climbed : ℕ := 96

-- The theorem to prove
theorem number_of_levels : (total_blocks_climbed / blocks_per_step) / steps_per_level = 4 := by
  sorry

end number_of_levels_l11_11760


namespace triangle_sin_A_and_height_l11_11484

noncomputable theory

variables (A B C : ℝ) (AB : ℝ)
  (h1 : A + B = 3 * C)
  (h2 : 2 * Real.sin (A - C) = Real.sin B)
  (h3 : AB = 5)

theorem triangle_sin_A_and_height :
  Real.sin A = 3 * Real.cos A → 
  sqrt 10 / 10 * Real.sin A = 3 / sqrt (10) / 3 → 
  √10 / 10 = 3/ sqrt 10 /3 → 
  sin (A+B) =sin /sqrt10 →
  (sin (A cv)+ C) = sin( AC ) → 
  ( cos A = sinA 3) 
  ( (10 +25)+5→1= well 5 → B (PS6 S)=H1 (A3+.B9)=
 
 
   
∧   (γ = hA → ) ( (/. );



∧ side /4→ABh3 → 5=HS)  ( →AB3)=sinh1S  

then 
(
  (Real.sin A = 3 * Real.cos A) ^2 )+   
  
(Real.cos A= √ 10/10
  
  Real.sin A2 C(B)= 3√10/10
  
 ) ^(Real.sin A = 5

6)=    
    sorry

end triangle_sin_A_and_height_l11_11484


namespace total_surface_area_l11_11619

-- Defining the dimensions of the box
variables (a b c : ℝ)

-- Given conditions
def sum_of_edges := 4 * a + 4 * b + 4 * c = 180
def diagonal := sqrt (a^2 + b^2 + c^2) = 25

-- Proving the total surface area
theorem total_surface_area (h1 : sum_of_edges a b c) (h2 : diagonal a b c) :
  2 * (a * b + b * c + c * a) = 1400 :=
sorry

end total_surface_area_l11_11619


namespace kite_raising_speed_ratio_l11_11939

theorem kite_raising_speed_ratio :
  (Omar_speed Jasper_speed : ℝ) 
  (omar_distance omar_time jasper_distance jasper_time : ℝ)
  (H1 : omar_distance = 240)
  (H2 : omar_time = 12)
  (H3 : jasper_distance = 600)
  (H4 : jasper_time = 10)
  (H5 : Omar_speed = omar_distance / omar_time)
  (H6 : Jasper_speed = jasper_distance / jasper_time)
  : Jasper_speed / Omar_speed = 3 :=
sorry

end kite_raising_speed_ratio_l11_11939


namespace triangle_side_lengths_l11_11682

theorem triangle_side_lengths (m : ℝ) (h1 : 3 < m) (h2 : m < 7) :
  sqrt ((m - 3)^2) + sqrt ((m - 7)^2) = 4 :=
by {
  sorry
}

end triangle_side_lengths_l11_11682


namespace composite_1991_pow_1991_add_1_composite_1991_pow_1991_sub_1_l11_11346

open Nat

theorem composite_1991_pow_1991_add_1 :
  ¬ Prime (1991 ^ 1991 + 1) :=
sorry

theorem composite_1991_pow_1991_sub_1 :
  ¬ Prime (1991 ^ 1991 - 1) :=
sorry

end composite_1991_pow_1991_add_1_composite_1991_pow_1991_sub_1_l11_11346


namespace sin_A_correct_height_on_AB_correct_l11_11479

noncomputable def sin_A (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) : ℝ :=
  Real.sin A

noncomputable def height_on_AB (A B C AB : ℝ) (height : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) (h4 : AB = 5) : ℝ :=
  height

theorem sin_A_correct (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) : 
  sorrry := 
begin
  -- proof omitted
  sorrry
end

theorem height_on_AB_correct (A B C AB : ℝ) (height : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) (h4 : AB = 5) :
  height = 6:= 
begin
  -- proof omitted
  sorrry
end 

end sin_A_correct_height_on_AB_correct_l11_11479


namespace ratio_A_B_plus_l11_11575

theorem ratio_A_B_plus (A B_plus: ℕ) (num_courses max_amount: ℕ) (reward_A_plus: ℕ) (min_A_plus: ℕ): 
  B_plus = 5 → 
  reward_A_plus = 15 → 
  min_A_plus = 2 → 
  num_courses = 10 → 
  max_amount = 190 → 
  16 * A + 2 * reward_A_plus = max_amount → 
  A = 10 → 
  (A / B_plus) = 2 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 
  rw [h1, h2, h5, h6, h7] 
  rfl 
  sorry

end ratio_A_B_plus_l11_11575


namespace maximum_lines_with_angle_l11_11969

theorem maximum_lines_with_angle (N : ℕ) : 
  (∀ (lines : Finset (Aff.line ℝ)), lines.card = N → (∀ l1 l2 ∈ lines, l1 ≠ l2 → Aff.line.intersects l1 l2) → 
  (∀ (sub_lines : Finset (Aff.line ℝ)), sub_lines.card = 15 → 
  ∃ l1 l2 ∈ sub_lines, l1.angle_with l2 = 60)) → N ≤ 42 := 
sorry

end maximum_lines_with_angle_l11_11969


namespace three_digit_multiples_25_not_75_l11_11063

theorem three_digit_multiples_25_not_75 : ∃ n, n = 24 ∧ 
  n = (finset.filter (λ x : ℕ, x % 25 = 0) (finset.Ico 100 1000)).card - 
      (finset.filter (λ x : ℕ, x % 75 = 0) (finset.Ico 100 1000)).card :=
by
  sorry

end three_digit_multiples_25_not_75_l11_11063


namespace problem1_problem2_l11_11398

-- Math definitions
def vector_a (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
def vector_b (β : ℝ) : ℝ × ℝ := (Real.cos β, Real.sin β)
def vector_c : ℝ × ℝ := (1, 0)

-- Proof Problem 1
theorem problem1 (α β : ℝ) (h : (vector_a α) • (vector_b β) = 2 / 3) :
  let θ := α - β in
  Real.sin θ ^ 2 - Real.sin (π / 2 + θ) = -1 / 9 :=
by
  sorry

-- Proof Problem 2
theorem problem2 (α β : ℝ) (hα : α ≠ k * π / 2) (hβ : β ≠ k * π) 
  (h : ∃ k : ℤ, α = 2 * k * π) (parl : vector_a α)
  (h1 : α ≠ k * π / 2 ∧ β ≠ k * π) : 
  vector_a α = ↑(k * π / 2) := by
  sorry

end problem1_problem2_l11_11398


namespace sin_A_eq_height_on_AB_l11_11520

-- Defining conditions
variables {A B C : ℝ}
variables (AB : ℝ)

-- Conditions based on given problem
def condition1 : Prop := A + B = 3 * C
def condition2 : Prop := 2 * sin (A - C) = sin B
def condition3 : Prop := A + B + C = Real.pi

-- Question 1: prove that sin A = (3 * sqrt 10) / 10
theorem sin_A_eq:
  condition1 → 
  condition2 → 
  condition3 → 
  sin A = (3 * Real.sqrt 10) / 10 :=
by
  sorry

-- Question 2: given AB = 5, prove the height on side AB is 6
theorem height_on_AB:
  condition1 →
  condition2 →
  condition3 →
  AB = 5 →
  -- Let's construct the height as a function of A, B, and C
  ∃ h, h = 6 :=
by
  sorry

end sin_A_eq_height_on_AB_l11_11520


namespace is_integer_A2n_Bn3_l11_11361

def k : ℕ := sorry

def A : ℕ → ℕ
| 0     := 0 -- not used; 0 is not considered a positive integer
| 1     := k
| 2     := k
| (n+3) := A (n+1) * A (n+2)

def B : ℕ → ℕ
| 0     := 0 -- not used; 0 is not considered a positive integer
| 1     := 1
| 2     := k
| (n+3) := (B (n+2) ^ 3 + 1) / B (n+1)

theorem is_integer_A2n_Bn3 (n : ℕ) : even (2 * n) → A (2 * n) * B (n + 3) ∈ ℤ :=
by
  intros h
  sorry

end is_integer_A2n_Bn3_l11_11361


namespace area_of_circle_segment_below_line_l11_11645

open Real

theorem area_of_circle_segment_below_line :
  let circle_eq := λ x y : ℝ, (x - 5)^2 + (y - 3)^2 = 9
  let line_eq := λ x y : ℝ, y = x - 1
  ∃ (area : ℝ), abs (area - 3 * π) < 0.01 ∧
    ( ∀ x y, (circle_eq x y ∧ x < y + 1) → x ≥ 0 ∧ y ≥ 0) :=
sorry

end area_of_circle_segment_below_line_l11_11645


namespace volume_ratio_cylinder_ball_l11_11274

/-- Definition of a cylinder and ball setup such that the ball is tangent to the sides, and top and bottom of the cylinder. -/
def cylinder_contains_ball (r : ℝ) (V_cylinder V_ball : ℝ) : Prop := 
  let V_cyl := 2 * π * r^3
  let V_ball := (4/3) * π * r^3
  V_cylinder = V_cyl ∧ V_ball = V_ball

/-- Theorem: The ratio of the volume of the cylinder to the volume of the ball is 3/2. -/
theorem volume_ratio_cylinder_ball {r V_cylinder V_ball : ℝ}
    (h : cylinder_contains_ball r V_cylinder V_ball) :
    V_cylinder / V_ball = 3 / 2 :=
  sorry

end volume_ratio_cylinder_ball_l11_11274


namespace domain_of_f_l11_11199

noncomputable def f : ℝ → ℝ := λ x, log (2 * x + 1) + 1 / (x - 2)

theorem domain_of_f :
  { x : ℝ | 2 * x + 1 > 0 ∧ x ≠ 2 } =
    (set.Ioo (-1 / 2) 2) ∪ (set.Ioi 2) :=
by
  sorry

end domain_of_f_l11_11199


namespace polynomial_root_range_l11_11269

variable (a : ℝ)

theorem polynomial_root_range (h : ∀ x : ℂ, (2 * x^4 + a * x^3 + 9 * x^2 + a * x + 2 = 0) →
  ((x.re^2 + x.im^2 ≠ 1) ∧ x.im ≠ 0)) : (-2 * Real.sqrt 10 < a ∧ a < 2 * Real.sqrt 10) :=
sorry

end polynomial_root_range_l11_11269


namespace line_eq_arith_geom_seq_l11_11851

-- Statement of the problem
theorem line_eq_arith_geom_seq (x₁ x₂ y₁ y₂ : ℝ) (d : ℝ)
  (h₁ : 1 = 1 + 0 * d)
  (h₂ : x₁ = 1 + 1 * d)
  (h₃ : x₂ = 1 + 2 * d)
  (h₄ : 7 = 1 + 3 * d)
  (r : ℝ)
  (p₁ : 1 = 1 * r)
  (p₂ : y₁ = 1 * r)
  (p₃ : y₂ = 1 * r^2)
  (p₄ : 8 = 1 * r^3) :
  line_equation (x₁, y₁) (x₂, y₂) = "x - y - 1 = 0" := 
sorry

end line_eq_arith_geom_seq_l11_11851


namespace domain_of_function_l11_11339

theorem domain_of_function :
  {x : ℝ | 2 * x + 1 ≥ 0 ∧ 3 - 4 * x > 0} = set.Icc (-1/2 : ℝ) (3/4 : ℝ) :=
by
  -- Add sorry to leave the proof incomplete
  sorry

end domain_of_function_l11_11339


namespace find_amount_with_r_l11_11672

variable (p q r : ℝ)

-- Condition 1: p, q, and r have Rs. 6000 among themselves.
def total_amount : Prop := p + q + r = 6000

-- Condition 2: r has two-thirds of the total amount with p and q.
def r_amount : Prop := r = (2 / 3) * (p + q)

theorem find_amount_with_r (h1 : total_amount p q r) (h2 : r_amount p q r) : r = 2400 := by
  sorry

end find_amount_with_r_l11_11672


namespace total_students_in_lunchroom_l11_11114

theorem total_students_in_lunchroom :
  (34 * 6) + 15 = 219 :=
by
  sorry

end total_students_in_lunchroom_l11_11114


namespace minimum_weighings_required_l11_11377

-- Definitions for the problem
def total_coins : Nat := 2023
def fake_coins : Nat := 2
def real_coins : Nat := 2021
def fake_weight : ℝ 
def real_weight : ℝ 

-- The condition that fake and real coins have different weights
axiom weight_diff : fake_weight ≠ real_weight

-- The main statement: Prove that the minimal number of weighings required is 3
theorem minimum_weighings_required : 
  ∃ n, (n = 3) ∧ (∀ c : Nat, ∀ f : Nat, ¬(n < 3) ∧ (c = total_coins) ∧ (f = fake_coins) ∧ (real_coins = total_coins - fake_coins)) :=
by { sorry }

end minimum_weighings_required_l11_11377


namespace probability_of_odd_digit_number_l11_11469

-- Defining the set of available digits
def digits : set ℕ := {1, 2, 3, 4, 5}

-- Defining the total number of ways to form a three-digit number from the given digits
def total_ways : ℕ := (set.card digits) * (set.card (digits.erase 1)) * (set.card (digits.erase 1).erase 2)

-- Defining the number of ways to form an odd three-digit number
def odd_ways : ℕ := 3 * (set.card (digits.erase 1)) * (set.card (digits.erase 1).erase 2)

-- Calculation of the probability
def probability_of_odd : ℚ := odd_ways / total_ways

-- The theorem statement
theorem probability_of_odd_digit_number : probability_of_odd = 3 / 5 :=
by {
  sorry -- This is where the proof would go
}

end probability_of_odd_digit_number_l11_11469


namespace min_area_triangle_l11_11875

open Real

noncomputable def f (a x : ℝ) : ℝ := exp (a * x)

def A (a : ℝ) : ℝ × ℝ := (a, 0)
def P (a : ℝ) : ℝ × ℝ := (a, exp (a^2))
def B (a : ℝ) : ℝ × ℝ := (a - 1 / a, 0)

def area_triangle (a : ℝ) : ℝ :=
  let base := abs ((fst (A a)) - (fst (B a)))
  let height := snd (P a)
  (base * height) / 2

theorem min_area_triangle (a : ℝ) (h : 0 < a) :
  (area_triangle a) = (sqrt (2 * exp 1)) / 2 :=
sorry

end min_area_triangle_l11_11875


namespace lines_perpendicular_to_same_plane_are_parallel_l11_11921

theorem lines_perpendicular_to_same_plane_are_parallel 
  (parallel_proj_parallel_lines : Prop)
  (planes_parallel_to_same_line : Prop)
  (planes_perpendicular_to_same_plane : Prop)
  (lines_perpendicular_to_same_plane : Prop) 
  (h1 : ¬ parallel_proj_parallel_lines)
  (h2 : ¬ planes_parallel_to_same_line)
  (h3 : ¬ planes_perpendicular_to_same_plane) :
  lines_perpendicular_to_same_plane := 
sorry

end lines_perpendicular_to_same_plane_are_parallel_l11_11921


namespace chocolates_left_l11_11994

-- Definitions based on the conditions
def initially_bought := 3
def gave_away := 2
def additionally_bought := 3

-- Theorem statement to prove
theorem chocolates_left : initially_bought - gave_away + additionally_bought = 4 := by
  -- Proof skipped
  sorry

end chocolates_left_l11_11994


namespace distance_from_A_to_line_l11_11197

-- Define the basic distance calculation components
noncomputable def distance_point_to_line (a b c x1 y1 : ℝ) : ℝ :=
  (abs (a * x1 + b * y1 + c)) / (real.sqrt (a^2 + b^2))

-- The specific distance problem
theorem distance_from_A_to_line :
  let A := (-1 : ℝ, 2 : ℝ)
  let line := (2, 1, -10)
  distance_point_to_line line.1 line.2 line.3 A.1 A.2 = 2 * real.sqrt 5 :=
by
  sorry -- The proof steps are not required per the instruction.


end distance_from_A_to_line_l11_11197


namespace conjugate_of_z_l11_11895

theorem conjugate_of_z (z : ℂ) (cond : z * (2 + complex.i) = 10 / (1 + complex.i)) : complex.conj z = 1 + 3 * complex.i :=
sorry

end conjugate_of_z_l11_11895


namespace divisibility_problem_l11_11937

theorem divisibility_problem :
  (2^62 + 1) % (2^31 + 2^16 + 1) = 0 := 
sorry

end divisibility_problem_l11_11937


namespace find_angle_and_area_l11_11084

theorem find_angle_and_area (A B C : ℝ) (a b c : ℝ) 
  (h_cos : cos B * cos C - sin B * sin C = 1/2)
  (h_a : a = 2 * sqrt 3)
  (h_bc : b + c = 4)
  (h_angle_sum : A + B + C = π) :
  A = 2 * π / 3 ∧
  (1/2 * b * c * sin A = sqrt 3) :=
by
  sorry

end find_angle_and_area_l11_11084


namespace time_after_9999_seconds_l11_11938

theorem time_after_9999_seconds :
  let initial_time := Time.mk 7 45 0 in
  let duration_sec := 9999 in
  let end_time := Time.mk 10 31 39 in
  add_seconds_to_time initial_time duration_sec = end_time :=
by
  sorry

end time_after_9999_seconds_l11_11938


namespace no_permutation_is_square_l11_11529

theorem no_permutation_is_square 
(number: ℕ) 
(h1: number.toString.length = 30) 
(h2: number.toString.count '1' = 10)
(h3: number.toString.count '2' = 10)
(h4: number.toString.count '3' = 10) : 
¬ ∃ n: ℕ, n^2 = number :=
by
  sorry

end no_permutation_is_square_l11_11529


namespace adam_has_more_apples_l11_11716

-- Define the number of apples Jackie has
def JackiesApples : Nat := 9

-- Define the number of apples Adam has
def AdamsApples : Nat := 14

-- Statement of the problem: Prove that Adam has 5 more apples than Jackie
theorem adam_has_more_apples :
  AdamsApples - JackiesApples = 5 :=
by
  sorry

end adam_has_more_apples_l11_11716


namespace range_of_a_l11_11081

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * x^2 + a * x - 1 ≤ 0) : -4 ≤ a ∧ a ≤ 0 := 
sorry

end range_of_a_l11_11081


namespace radius_of_sector_l11_11857

theorem radius_of_sector (l : ℝ) (α : ℝ) (R : ℝ) (h1 : l = 2 * π / 3) (h2 : α = π / 3) : R = 2 := by
  have : l = |α| * R := by sorry
  rw [h1, h2] at this
  sorry

end radius_of_sector_l11_11857


namespace count_19_tuples_eq_54264_l11_11354

theorem count_19_tuples_eq_54264 : 
  (∃ (s : Fin 19 → ℤ), 
    (∀ i : Fin 19, (s i)^2 = ∑ j in (Finset.univ.filter (≠ i)), s j)) ↔ 
    ∃ k, k = 54264 :=
by 
  sorry

end count_19_tuples_eq_54264_l11_11354


namespace smallest_n_with_pairwise_coprime_numbers_l11_11357

theorem smallest_n_with_pairwise_coprime_numbers :
  ∃ n : ℕ, n = 41 ∧ ∀ (S : set ℕ), S ⊆ (set.Icc 1 60) → set.card S = n →
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ pairwise_coprime {a, b, c} :=
begin
  sorry
end

end smallest_n_with_pairwise_coprime_numbers_l11_11357


namespace sum_of_ratios_equal_two_l11_11232

theorem sum_of_ratios_equal_two {ABC : Triangle} (O : Point) :
  let A1A2 := segment_parallel_to_side_through_point ABC BC O,
      B1B2 := segment_parallel_to_side_through_point ABC CA O,
      C1C2 := segment_parallel_to_side_through_point ABC AB O in
  let ratio1 := length(A1A2) / length(BC),
      ratio2 := length(B1B2) / length(CA),
      ratio3 := length(C1C2) / length(AB) in
  ratio1 + ratio2 + ratio3 = 2 := 
by
  sorry

end sum_of_ratios_equal_two_l11_11232


namespace bob_finishes_first_l11_11717

variable (a r : ℝ)

-- Define the areas of the gardens
def bob_garden_area : ℝ := a / 3
def charlie_garden_area : ℝ := a / 2

-- Define the mowing rates
def charlie_mowing_rate : ℝ := r / 4
def bob_mowing_rate : ℝ := r / 2

-- Calculate the time taken to mow each garden
def alice_mowing_time : ℝ := a / r
def bob_mowing_time : ℝ := (a / 3) / (r / 2)
def charlie_mowing_time : ℝ := (a / 2) / (r / 4)

-- Theorem stating Bob finishes mowing first
theorem bob_finishes_first {a r : ℝ} :
  bob_mowing_time a r < alice_mowing_time a r ∧ bob_mowing_time a r < charlie_mowing_time a r :=
by
  sorry

end bob_finishes_first_l11_11717


namespace number_of_valid_three_digit_even_numbers_l11_11002

def valid_three_digit_even_numbers (n : ℕ) : Prop :=
  (100 ≤ n) ∧ (n < 1000) ∧ (n % 2 = 0) ∧ (let t := (n / 10) % 10 in
                                           let u := n % 10 in
                                           t + u = 12)

theorem number_of_valid_three_digit_even_numbers : 
  (∃ cnt : ℕ, cnt = 27 ∧ (cnt = (count (λ n, valid_three_digit_even_numbers n) (Ico 100 1000)))) :=
sorry

end number_of_valid_three_digit_even_numbers_l11_11002


namespace floor_sqrt_50_squared_l11_11807

theorem floor_sqrt_50_squared : ∃ x : ℕ, x = 49 ∧ (⌊real.sqrt 50⌋ : ℕ) ^ 2 = x := by
  have h1 : (7 : ℝ) < real.sqrt 50 := sorry
  have h2 : real.sqrt 50 < 8 := sorry
  have h_floor : (⌊real.sqrt 50⌋ : ℕ) = 7 := sorry
  use 49
  constructor
  · rfl
  · rw [h_floor]
    norm_num
    sorry

end floor_sqrt_50_squared_l11_11807


namespace shortest_chord_through_P_tangent_lines_through_M_l11_11842

-- Conditions
def circle : (ℝ × ℝ) → Prop := λ p, (p.1 - 3)^2 + (p.2 - 4)^2 = 4
def point_P : (ℝ × ℝ) := (2, 5)
def point_M : (ℝ × ℝ) := (5, 0)

-- Part I: Prove the equation of the line through P with the shortest chord length is x - y + 3 = 0
theorem shortest_chord_through_P :
  ∃ l : ℝ × ℝ → Prop, (∀ p, l p ↔ p.1 - p.2 + 3 = 0) ∧ 
  ∀ q : ℝ × ℝ, l q → circle q → q = point_P :=
begin
  sorry
end

-- Part II: Prove the equations of the tangent lines through M to the circle are x = 5 and 3x + 4y - 15 = 0
theorem tangent_lines_through_M :
  ∃ l1 l2: ℝ × ℝ → Prop, 
    (∀ p, (l1 p ↔ 3 * p.1 + 4 * p.2 - 15 = 0) ∨ (l2 p ↔ p.1 = 5)) ∧ 
    ∀ q : ℝ × ℝ, (l1 q ∨ l2 q) → circle q → q = point_M :=
begin
  sorry
end

end shortest_chord_through_P_tangent_lines_through_M_l11_11842


namespace lucky_ticket_count_l11_11301

   noncomputable def count_lucky_tickets : ℕ :=
     let num_vars := {x : Fin 10 // ∀ i : Fin 6, 0 ≤ x i ∧ x i ≤ 9} 
     (finset.univ.filter (λ x : num_vars, x 0 + x 1 + x 2 = x 3 + x 4 + x 5)).card

   theorem lucky_ticket_count : count_lucky_tickets = 55252 := by
     sorry
   
end lucky_ticket_count_l11_11301


namespace woman_work_completion_woman_days_to_complete_l11_11702

theorem woman_work_completion (M W B : ℝ) (h1 : M + W + B = 1/4) (h2 : M = 1/6) (h3 : B = 1/18) : W = 1/36 :=
by
  -- Substitute h2 and h3 into h1 and solve for W
  sorry

theorem woman_days_to_complete (W : ℝ) (h : W = 1/36) : 1 / W = 36 :=
by
  -- Calculate the reciprocal of h
  sorry

end woman_work_completion_woman_days_to_complete_l11_11702


namespace construct_triangle_l11_11331

theorem construct_triangle (a AM λ : ℝ) (h_pos_a : 0 < a) (h_pos_AM : 0 < AM) (h_pos_λ : 0 < λ) : 
  ∃ (A B C : ℝ×ℝ), 
    dist B C = a ∧ 
    dist A B / dist A C = λ ∧ 
    is_angle_bisector A B C AM :=
sorry

end construct_triangle_l11_11331


namespace smaller_solution_l11_11651

theorem smaller_solution (x : ℝ) (h : x^2 + 9 * x - 22 = 0) : x = -11 :=
sorry

end smaller_solution_l11_11651


namespace log_a_lt_1_range_l11_11886

theorem log_a_lt_1_range {a : ℝ} (h : real.log a < 1) : a ∈ set.Ioo 0 1 ∪ set.Ioi 1 :=
begin
  sorry
end

end log_a_lt_1_range_l11_11886


namespace maci_pays_total_amount_l11_11151

theorem maci_pays_total_amount :
  let cost_blue_pen := 10 -- cents
  let cost_red_pen := 2 * cost_blue_pen -- cents
  let blue_pen_count := 10
  let red_pen_count := 15
  let total_cost_blue_pens := blue_pen_count * cost_blue_pen -- cents
  let total_cost_red_pens := red_pen_count * cost_red_pen -- cents
  let total_cost := total_cost_blue_pens + total_cost_red_pens -- cents
  total_cost / 100 = 4 -- dollars :=
by
  sorry

end maci_pays_total_amount_l11_11151


namespace root_in_interval_l11_11413

noncomputable def f (x : ℝ) : ℝ := (x^2 - 3*x + 2)*Real.log x + 2009*x - 2010

theorem root_in_interval : ∃ c ∈ Set.Ioo 1 2, f c = 0 :=
by
  have h₁ : f 1 < 0 := by norm_num [f]
  have h₂ : f 2 > 0 := by norm_num [f]
  exact IntermediateValueTheorem (Set.Ioo 1 2) f h₁ h₂
  sorry

end root_in_interval_l11_11413


namespace math_problem_l11_11137

variables (x y z w p q : ℕ)
variables (x_pos : 0 < x) (y_pos : 0 < y) (z_pos : 0 < z) (w_pos : 0 < w)

theorem math_problem
  (h1 : x^3 = y^2)
  (h2 : z^4 = w^3)
  (h3 : z - x = 22)
  (hx : x = p^2)
  (hy : y = p^3)
  (hz : z = q^3)
  (hw : w = q^4) : w - y = q^4 - p^3 :=
sorry

end math_problem_l11_11137


namespace evaluate_expression_l11_11215

theorem evaluate_expression :
  (sin 10 ≠ 0) → (cos 10 ≠ 0) →
  (1 / (sin 10) - sqrt 3 / (cos 10) = 2 * sqrt 3 * cot 10 - 2) :=
by
  sorry

end evaluate_expression_l11_11215


namespace range_of_m_l11_11183

noncomputable def f (x : ℝ) : ℝ := x^2 - 1

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x ∈ set.Ici 3 → f (x / m) - 4 * m^2 * f x ≤ f (x - 1) + 4 * f m) ↔
  (m ≤ -Real.sqrt 2 / 2 ∨ Real.sqrt 2 / 2 ≤ m) :=
by
  sorry

end range_of_m_l11_11183


namespace bug_returns_to_A_after_4_steps_l11_11590

-- Define the vertices as types
inductive Vertex | A | B | C | D

-- Define the edge structure of a regular tetrahedron
def edges : Vertex → list Vertex 
| Vertex.A := [Vertex.B, Vertex.C, Vertex.D]
| Vertex.B := [Vertex.A, Vertex.C, Vertex.D]
| Vertex.C := [Vertex.A, Vertex.B, Vertex.D]
| Vertex.D := [Vertex.A, Vertex.B, Vertex.C]

-- Define the probability calculation for the bug returning to vertex A
noncomputable def probability_return_A : ℚ :=
  7 / 27

-- Define the proof problem statement
theorem bug_returns_to_A_after_4_steps :
  ∀ (start : Vertex), start = Vertex.A →
  probability_return_A = 7 / 27 :=
by
  intros
  sorry

end bug_returns_to_A_after_4_steps_l11_11590


namespace count_valid_even_numbers_with_sum_12_l11_11022

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n % 2 = 0) ∧ 
  ((n / 10) % 10 + n % 10 = 12)

theorem count_valid_even_numbers_with_sum_12 :
  (finset.range 1000).filter is_valid_number).card = 27 := by
  sorry

end count_valid_even_numbers_with_sum_12_l11_11022


namespace anes_age_l11_11569

theorem anes_age (w w_d : ℤ) (n : ℤ) 
  (h1 : 1436 ≤ w ∧ w < 1445)
  (h2 : 1606 ≤ w_d ∧ w_d < 1615)
  (h3 : w_d = w + n * 40) : 
  n = 4 :=
sorry

end anes_age_l11_11569


namespace perpendicular_lines_b_eq_neg9_l11_11609

-- Definitions for the conditions.
def eq1 (x y : ℝ) : Prop := x + 3 * y + 4 = 0
def eq2 (b x y : ℝ) : Prop := b * x + 3 * y + 4 = 0

-- The problem statement
theorem perpendicular_lines_b_eq_neg9 (b : ℝ) : 
  (∀ x y, eq1 x y → eq2 b x y) ∧ (∀ x y, eq2 b x y → eq1 x y) → b = -9 :=
by
  sorry

end perpendicular_lines_b_eq_neg9_l11_11609


namespace proof_triangle_properties_l11_11498

variable (A B C : ℝ)
variable (h AB : ℝ)

-- Conditions
def triangle_conditions : Prop :=
  (A + B = 3 * C) ∧ (2 * Real.sin (A - C) = Real.sin B) ∧ (AB = 5)

-- Part 1: Proving sin A
def find_sin_A (h₁ : triangle_conditions A B C h AB) : Prop :=
  Real.sin A = 3 * Real.cos A

-- Part 2: Proving the height on side AB
def find_height_on_AB (h₁ : triangle_conditions A B C h AB) : Prop :=
  h = 6

-- Combined proof statement
theorem proof_triangle_properties (h₁ : triangle_conditions A B C h AB) : 
  find_sin_A A B C h₁ ∧ find_height_on_AB A B C h AB h₁ := 
  by sorry

end proof_triangle_properties_l11_11498


namespace floor_sqrt_50_squared_l11_11805

theorem floor_sqrt_50_squared : ∃ x : ℕ, x = 49 ∧ (⌊real.sqrt 50⌋ : ℕ) ^ 2 = x := by
  have h1 : (7 : ℝ) < real.sqrt 50 := sorry
  have h2 : real.sqrt 50 < 8 := sorry
  have h_floor : (⌊real.sqrt 50⌋ : ℕ) = 7 := sorry
  use 49
  constructor
  · rfl
  · rw [h_floor]
    norm_num
    sorry

end floor_sqrt_50_squared_l11_11805


namespace find_f_of_2_l11_11438

noncomputable def f (x : ℕ) : ℕ := x^x + 2*x + 2

theorem find_f_of_2 : f 1 + 1 = 5 := 
by 
  sorry

end find_f_of_2_l11_11438


namespace marks_bottlecap_ratio_l11_11127

theorem marks_bottlecap_ratio :
  let jenny_initial := 18
  let jenny_bounce_ratio := 1 / 3
  let mark_initial := 15
  let distance_difference := 21
  let jenny_total := jenny_initial + jenny_bounce_ratio * jenny_initial
  let mark_total := jenny_total + distance_difference
  let mark_bounce_multiple := 2 :  -- from the solution

  jenny_total = 24 ∧ mark_total = 45 ∧ 15 * mark_bounce_multiple == 30 ∧ mark_total = mark_initial + 15 * mark_bounce_multiple :=
  begin
    sorry
  end

end marks_bottlecap_ratio_l11_11127


namespace range_of_f_l11_11313

def f (x : ℝ) : ℝ := (sin x) ^ 4 + (cos x) ^ 2

theorem range_of_f : set.Icc (3/4 : ℝ) 1 = {y : ℝ | ∃ x : ℝ, f x = y} :=
by
  sorry

end range_of_f_l11_11313


namespace solution_set_ineq_l11_11221

theorem solution_set_ineq (x : ℝ) : (1 / x > 1) ↔ (0 < x ∧ x < 1) :=
by
  sorry

end solution_set_ineq_l11_11221


namespace translated_line_tangent_to_circle_l11_11080

theorem translated_line_tangent_to_circle (c : ℝ) :
  let line := λ x y : ℝ, 2 * x - y + c
  let translated_line := λ x y : ℝ, 2 * (x - 1) - (y + 1) + c
  let circle := λ x y : ℝ, x^2 + y^2 = 5
  (translated_line (0 : ℝ) (0 : ℝ) = 0) ↔ (c = 8 ∨ c = -2) :=
by
  sorry

end translated_line_tangent_to_circle_l11_11080


namespace exists_poly_with_neg_coeff_l11_11677

open Polynomial

noncomputable def g (ε : ℝ) [h : 0 < ε] : Polynomial ℝ :=
  (X^4 + X^3 + X + 1) - C ε * X^2

theorem exists_poly_with_neg_coeff {n : ℕ} (h_n : 1 < n) :
  ∃ (P : Polynomial ℝ) (ε : ℝ), (0 < ε) ∧ (P = g ε) ∧
    (∃ coeff_neg : ∃ i, coeff P i < 0) ∧
    (∀ m, 1 < m → ∀ i, coeff (P^m) i > 0) :=
by
  sorry


end exists_poly_with_neg_coeff_l11_11677


namespace profit_percentage_is_25_l11_11192

-- Definitions of the variables involved
variables (C S : ℝ)
variables (x : ℕ)

-- Condition given in the problem
def condition1 : Prop := 20 * C = x * S
def condition2 : Prop := x = 16

-- The profit percentage we're aiming to prove
def profit_percentage : ℝ := ((S - C) / C) * 100

-- The theorem to prove
theorem profit_percentage_is_25 (h1 : condition1) (h2 : condition2) :
  profit_percentage C S = 25 :=
sorry

end profit_percentage_is_25_l11_11192


namespace ten_numbers_property_l11_11352

theorem ten_numbers_property : 
  ∃ S : Finset ℕ, S.card = 10 ∧ 
  (∀ a ∈ S, (S.sum id) % a = 0) ∧ 
  S = {1, 2, 3, 6, 12, 24, 48, 96, 192, 384} := 
  sorry

end ten_numbers_property_l11_11352


namespace range_of_f_on_interval_l11_11217

noncomputable def f (x : ℝ) : ℝ := 2 / (x - 1)

theorem range_of_f_on_interval :
  Set.range (λ x, f x) (Set.Icc 2 6) = Set.Icc (2 / 5) 2 :=
sorry

end range_of_f_on_interval_l11_11217


namespace find_z_l11_11560

noncomputable def x : ℕ := 200
noncomputable def y (z : ℕ) : ℕ := 2 * z

theorem find_z (z : ℕ) (h1 : x + y z + z = 500) (h2 : x - z = 0.5 * y z) : z = 100 :=
sorry

end find_z_l11_11560


namespace count_valid_even_numbers_with_sum_12_l11_11020

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n % 2 = 0) ∧ 
  ((n / 10) % 10 + n % 10 = 12)

theorem count_valid_even_numbers_with_sum_12 :
  (finset.range 1000).filter is_valid_number).card = 27 := by
  sorry

end count_valid_even_numbers_with_sum_12_l11_11020


namespace proof_smallest_possible_fraction_l11_11539

noncomputable def smallest_possible_fraction (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) (h_eq : 18*x - 4*x^2 + 2*x^3 - 9*y - 10*x*y - x^2*y + 6*y^2 + 2*x*y^2 - y^3 = 0) : ℚ :=
  ∀ (m n : ℕ), m.gcd n = 1 → (m : ℚ)/(n : ℚ) = y/x → m + n = 7

theorem proof_smallest_possible_fraction : ∃ (m n : ℕ), m.gcd n = 1 ∧ (m : ℚ) / (n : ℚ) = 4 / 3 ∧ m + n = 7 :=
by
  let x_ne_zero : Prop := x ≠ 0
  let y_ne_zero : Prop := y ≠ 0
  let h_eq : Prop := 18*x - 4*x^2 + 2*x^3 - 9*y - 10*x*y - x^2*y + 6*y^2 + 2*x*y^2 - y^3 = 0
  exact ⟨4, 3, by rfl, rfl, rfl⟩

end proof_smallest_possible_fraction_l11_11539


namespace simplest_form_correct_l11_11662

variable (A : ℝ)
variable (B : ℝ)
variable (C : ℝ)
variable (D : ℝ)

def is_simplest_form (x : ℝ) : Prop :=
-- define what it means for a square root to be in simplest form
sorry

theorem simplest_form_correct :
  A = Real.sqrt (1 / 2) ∧ B = Real.sqrt 0.2 ∧ C = Real.sqrt 3 ∧ D = Real.sqrt 8 →
  ¬ is_simplest_form A ∧ ¬ is_simplest_form B ∧ is_simplest_form C ∧ ¬ is_simplest_form D :=
by
  -- prove that C is the simplest form and others are not
  sorry

end simplest_form_correct_l11_11662


namespace puzzle_pieces_count_l11_11164

theorem puzzle_pieces_count :
  let a := 4 in
  let b := 108 in
  let c := 4 in
  let d := 52 in
  let total_pieces := 851 in
  let n := total_pieces - (a + b + c + d) in
  n = 683 := by
{ --
    sorry 
}

end puzzle_pieces_count_l11_11164


namespace find_angle_C_l11_11932

theorem find_angle_C
  (A B C : Type)
  [MetricSpace A]
  [MetricSpace B]
  [MetricSpace C]
  (BC AB AC : ℝ)
  (h1 : BC^2 - AB^2 = AC^2)
  (h2 : B = 55)
  (triangle_ABC : Triangle A B C) :
  ∃ C, C = 35 :=
by
  -- Proof is omitted
  sorry

end find_angle_C_l11_11932


namespace sales_coincide_once_in_july_l11_11285

-- Define the sets for bookstore and shoe store sales days
def bookstore_sales_days : Finset ℕ := {5, 10, 15, 20, 25, 30}

def shoe_store_sales_days : Finset ℕ := {3 + 7 * k | k in {0, 1, 2, 3, 4}}

-- The proof problem: Show intersection of the two sets has exactly one element
theorem sales_coincide_once_in_july : (bookstore_sales_days ∩ shoe_store_sales_days).card = 1 := 
by {
  sorry
}

end sales_coincide_once_in_july_l11_11285


namespace maci_pays_total_cost_l11_11148

def cost_blue_pen : ℝ := 0.10
def num_blue_pens : ℕ := 10
def num_red_pens : ℕ := 15
def cost_red_pen : ℝ := 2 * cost_blue_pen

def total_cost : ℝ := num_blue_pens * cost_blue_pen + num_red_pens * cost_red_pen

theorem maci_pays_total_cost : total_cost = 4 := by
  -- Proof goes here
  sorry

end maci_pays_total_cost_l11_11148


namespace find_ratio_b_a_l11_11409

theorem find_ratio_b_a (a b : ℝ) 
  (h : ∀ x : ℝ, (2 * a - b) * x + (a + b) > 0 ↔ x > -3) : 
  b / a = 5 / 4 :=
sorry

end find_ratio_b_a_l11_11409


namespace IMO1991Q1_l11_11819

theorem IMO1991Q1 (x y z : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
    (h4 : 3^x + 4^y = 5^z) : x = 2 ∧ y = 2 ∧ z = 2 := by
  sorry

end IMO1991Q1_l11_11819


namespace daily_protein_intake_per_kg_eq_two_l11_11565

-- Define the conditions
def protein_content := 0.80
def matt_weight := 80
def weekly_protein_powder := 1400
def days_in_week := 7

-- Define the total protein intake from protein powder in a week
def weekly_protein_intake := protein_content * weekly_protein_powder

-- Define the desired proof statement
theorem daily_protein_intake_per_kg_eq_two :
  (weekly_protein_intake / (matt_weight * days_in_week)) = 2 := by
  sorry

end daily_protein_intake_per_kg_eq_two_l11_11565


namespace family_vacation_days_l11_11696

theorem family_vacation_days
  (rained_days : ℕ)
  (total_days : ℕ)
  (clear_mornings : ℕ)
  (H1 : rained_days = 13)
  (H2 : total_days = 18)
  (H3 : clear_mornings = 11) :
  total_days = 18 :=
by
  -- proof to be filled in here
  sorry

end family_vacation_days_l11_11696


namespace pq_true_l11_11981

-- Proposition p: a^2 + b^2 < 0 is false
def p_false (a b : ℝ) : Prop := ¬ (a^2 + b^2 < 0)

-- Proposition q: (a-2)^2 + |b-3| ≥ 0 is true
def q_true (a b : ℝ) : Prop := (a - 2)^2 + |b - 3| ≥ 0

-- Theorem stating that "p ∨ q" is true
theorem pq_true (a b : ℝ) (h1 : p_false a b) (h2 : q_true a b) : (a^2 + b^2 < 0 ∨ (a - 2)^2 + |b - 3| ≥ 0) :=
by {
  sorry
}

end pq_true_l11_11981


namespace triangle_angle_C_l11_11083

theorem triangle_angle_C (A B C : Real) (h1 : A - B = 10) (h2 : B = A / 2) :
  C = 150 :=
by
  -- Proof goes here
  sorry

end triangle_angle_C_l11_11083


namespace quadratic_transformation_l11_11861

noncomputable def transform_roots (p q r : ℚ) (u v : ℚ) 
    (h1 : u + v = -q / p) 
    (h2 : u * v = r / p) : Prop :=
  ∃ y : ℚ, y^2 - q^2 + 4 * p * r = 0

theorem quadratic_transformation (p q r u v : ℚ) 
    (h1 : u + v = -q / p) 
    (h2 : u * v = r / p) :
  ∃ y : ℚ, (y - (2 * p * u + q)) * (y - (2 * p * v + q)) = y^2 - q^2 + 4 * p * r :=
by {
  sorry
}

end quadratic_transformation_l11_11861


namespace probability_prime_numbered_ball_l11_11351

theorem probability_prime_numbered_ball :
  let numbers := Finset.range 25 \ Finset.range 10 -- {10, 11, 12, ..., 24}
  let prime_numbers := {11, 13, 17, 19, 23}.to_finset
  (∑ n in numbers.filter (λ n => n ∈ prime_numbers), 1) / (∑ n in numbers, 1) = 1 / 3 :=
by
  sorry

end probability_prime_numbered_ball_l11_11351


namespace sum_of_possible_values_l11_11976

theorem sum_of_possible_values (x y : ℝ) (h1 : x^3 = 3 * x + y) (h2 : y^3 = 3 * y + x) :
  x^2 + y^2 = 3 ∨ x^2 + y^2 = 4 ∨ x^2 + y^2 = 8 ∨ x^2 + y^2 = 0 →
  ∑ v in {v | v = (x^2 + y^2)}, v = 15 :=
by
  sorry

end sum_of_possible_values_l11_11976


namespace smallest_positive_period_and_min_val_f_of_beta_squared_eq_two_l11_11873

noncomputable def f (x : ℝ) : ℝ := sin (x + 7 * Real.pi / 4) + cos (x - 3 * Real.pi / 4)

theorem smallest_positive_period_and_min_val :
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧ (∃ x : ℝ, f x = -2) :=
by
  sorry

variables {α β : ℝ}
-- Assuming the conditions given in part (2)
theorem f_of_beta_squared_eq_two
  (h₁ : cos (β - α) = 4 / 5)
  (h₂ : cos (β + α) = -4 / 5)
  (h₃ : 0 < α ∧ α < β ∧ β ≤ π / 2) :
  (f β) ^ 2 = 2 :=
by
  sorry

end smallest_positive_period_and_min_val_f_of_beta_squared_eq_two_l11_11873


namespace convex_numbers_count_l11_11072

noncomputable def count_convex_numbers : Nat := 204

theorem convex_numbers_count : 
  ∃ (N: Nat), 
    N = count_convex_numbers ∧ 
    N = 
    (card { xyz : Finset (Nat × Nat × Nat) |
       ∃ (x y z: Nat),
         xyz = (x, y, z) ∧
         (100 ≤ 100 * x + 10 * y + z ∧ 
          999 ≥ 100 * x + 10 * y + z) ∧ 
          (y > x ∧ y > z) ∧ 
          (x ≠ y ∧ y ≠ z ∧ x ≠ z) 
     })
  :=
by 
  sorry

end convex_numbers_count_l11_11072


namespace factorial_zero_remainder_l11_11947

-- Given that M is the number of consecutive 0's at the right end of the product of factorials from 1 to 50,
-- we want to find the remainder when M is divided by 500.
def factorial_zero_count : ℤ := (List.range' 1 51).map factorial |> List.foldl (*) 1 |> nat_trailing_zeros
theorem factorial_zero_remainder : factorial_zero_count % 500 = 12 := by
  sorry

end factorial_zero_remainder_l11_11947


namespace right_triangle_set_C_l11_11306

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Given sets of line segments
def set_A := (1, 2, 3)
def set_B := (5, 11, 12)
def set_C := (5, 12, 13)
def set_D := (6, 8, 9)

theorem right_triangle_set_C :
  is_right_triangle 5 12 13 ∧
  ¬ is_right_triangle 1 2 3 ∧
  ¬ is_right_triangle 5 11 12 ∧
  ¬ is_right_triangle 6 8 9 :=
by
  split
  {
    show is_right_triangle 5 12 13, sorry
  }
  {
    split
    {
      show ¬ is_right_triangle 1 2 3, sorry
    }
    {
      split
      {
        show ¬ is_right_triangle 5 11 12, sorry
      }
      {
        show ¬ is_right_triangle 6 8 9, sorry
      }
    }
  }

end right_triangle_set_C_l11_11306


namespace length_of_c_l11_11561

theorem length_of_c (a b c : ℝ) (h1 : a = 1) (h2 : b = 3) (h_triangle : 0 < c) :
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) → c = 3 :=
by
  intros h_ineq
  sorry

end length_of_c_l11_11561


namespace angle_ADC_of_equilateral_and_square_l11_11929

theorem angle_ADC_of_equilateral_and_square 
  (A B C D E : Type) [EquilateralTriangle A B C] [Square B C D E] : 
  ∠ADC = 15 :=
sorry

end angle_ADC_of_equilateral_and_square_l11_11929


namespace positive_expression_with_b_l11_11829

-- Defining the conditions and final statement
open Real

theorem positive_expression_with_b (a : ℝ) : (a + 2) * (a + 5) * (a + 8) * (a + 11) + 82 > 0 := 
sorry

end positive_expression_with_b_l11_11829


namespace percent_decrease_l11_11533

theorem percent_decrease (p_original p_sale : ℝ) (h₁ : p_original = 100) (h₂ : p_sale = 50) :
  ((p_original - p_sale) / p_original * 100) = 50 := by
  sorry

end percent_decrease_l11_11533


namespace circles_intersect_midpoint_l11_11430

theorem circles_intersect_midpoint (m n : ℝ) :
  (∀ (x y : ℝ), x + y + n = 0 ∧ (2, 3) ∈ circles ∧ (m, 2) ∈ circles) →
  m + n = -2 :=
sorry

end circles_intersect_midpoint_l11_11430


namespace pattern_equation_l11_11155

theorem pattern_equation (n : ℕ) (hn : n > 0) : n * (n + 2) + 1 = (n + 1) ^ 2 := 
by sorry

end pattern_equation_l11_11155


namespace three_digit_even_sum_12_l11_11030

theorem three_digit_even_sum_12 : 
  ∃ (n : Finset ℕ), 
    n.card = 27 ∧ 
    ∀ x ∈ n, 
      ∃ h t u, 
        (100 * h + 10 * t + u = x) ∧ 
        (h ∈ Finset.range 9 \ {0}) ∧ 
        (u % 2 = 0) ∧ 
        (t + u = 12) := 
sorry

end three_digit_even_sum_12_l11_11030


namespace complex_transformation_l11_11311

theorem complex_transformation :
  let z := complex.mk (-3) (-8)
  let rotation := complex.mk (1) (real.sqrt 3)
  let dilation := 2
  (z * (rotation * dilation)) = complex.mk (8 * real.sqrt 3 - 3) (-(3 * real.sqrt 3 + 8)) :=
by
  -- Placeholder for the proof
  sorry

end complex_transformation_l11_11311


namespace product_fraction_l11_11348

open Int

def first_six_composites : List ℕ := [4, 6, 8, 9, 10, 12]
def first_three_primes : List ℕ := [2, 3, 5]
def next_three_composites : List ℕ := [14, 15, 16]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem product_fraction :
  (product first_six_composites : ℚ) / (product (first_three_primes ++ next_three_composites) : ℚ) = 24 / 7 :=
by 
  sorry

end product_fraction_l11_11348


namespace proof_triangle_properties_l11_11496

variable (A B C : ℝ)
variable (h AB : ℝ)

-- Conditions
def triangle_conditions : Prop :=
  (A + B = 3 * C) ∧ (2 * Real.sin (A - C) = Real.sin B) ∧ (AB = 5)

-- Part 1: Proving sin A
def find_sin_A (h₁ : triangle_conditions A B C h AB) : Prop :=
  Real.sin A = 3 * Real.cos A

-- Part 2: Proving the height on side AB
def find_height_on_AB (h₁ : triangle_conditions A B C h AB) : Prop :=
  h = 6

-- Combined proof statement
theorem proof_triangle_properties (h₁ : triangle_conditions A B C h AB) : 
  find_sin_A A B C h₁ ∧ find_height_on_AB A B C h AB h₁ := 
  by sorry

end proof_triangle_properties_l11_11496


namespace count_even_three_digit_numbers_sum_tens_units_eq_12_l11_11006

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def is_even (n : ℕ) : Prop := n % 2 = 0
def sum_of_tens_and_units_eq_12 (n : ℕ) : Prop :=
  (n / 10) % 10 + n % 10 = 12

theorem count_even_three_digit_numbers_sum_tens_units_eq_12 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_three_digit n ∧ is_even n ∧ sum_of_tens_and_units_eq_12 n) ∧ S.card = 24 :=
sorry

end count_even_three_digit_numbers_sum_tens_units_eq_12_l11_11006


namespace sin_sum_identity_l11_11259

theorem sin_sum_identity (θ : ℝ) :
  (∀ x : ℝ, sin (x + θ) = sin x * sin θ + cos x * cos θ) ↔ θ = (π / 4) :=
sorry

end sin_sum_identity_l11_11259


namespace smallest_number_of_tins_needed_l11_11978

variable (A : ℤ) (C : ℚ)

-- Conditions
def wall_area_valid : Prop := 1915 ≤ A ∧ A < 1925
def coverage_per_tin_valid : Prop := 17.5 ≤ C ∧ C < 18.5
def tins_needed_to_cover_wall (A : ℤ) (C : ℚ) : ℚ := A / C
def smallest_tins_needed : ℚ := 111

-- Proof problem statement
theorem smallest_number_of_tins_needed (A : ℤ) (C : ℚ)
    (h1 : wall_area_valid A)
    (h2 : coverage_per_tin_valid C)
    (h3 : 1915 ≤ A)
    (h4 : A < 1925)
    (h5 : 17.5 ≤ C)
    (h6 : C < 18.5) : 
  tins_needed_to_cover_wall A C + 1 ≥ smallest_tins_needed := by
    sorry

end smallest_number_of_tins_needed_l11_11978


namespace inequality_abc_l11_11985

theorem inequality_abc
    (a b c : ℝ)
    (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) :
    a + b + c ≤ (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b) ∧ 
    (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b) ≤ (a^3) / (b * c) + (b^3) / (c * a) + (c^3) / (a * b) :=
begin
    sorry,
    sorry
end

end inequality_abc_l11_11985


namespace chord_intersection_problem_l11_11926

theorem chord_intersection_problem 
  (O : Type) [metric_space O] [normed_space ℝ O]
  {A B C D E : O}
  (h1 : dist A B = dist A C)
  (h2 : dist A C = 12)
  (h3 : dist A E = 8)
  (h4 : exists O, is_circle O A ∧ is_circle O B ∧ is_circle O C ∧ is_circle O D ∧ is_circle O E)
  (h5 : ∃ E, line_segment A D ∩ line_segment B C = {E}) : 
  dist A D = 18 :=
begin
  sorry,
end

end chord_intersection_problem_l11_11926


namespace area_increase_percent_l11_11079

theorem area_increase_percent (l w : ℝ) :
  let original_area := l * w
  let new_length := 1.15 * l
  let new_width := 1.25 * w
  let new_area := new_length * new_width
  let increase_percent := ((new_area - original_area) / original_area) * 100
  increase_percent = 43.75 :=
by
  let original_area := l * w
  let new_length := 1.15 * l
  let new_width := 1.25 * w
  let new_area := new_length * new_width
  let increase_percent := ((new_area - original_area) / original_area) * 100
  calc
    increase_percent = ((new_area - original_area) / original_area) * 100 : rfl
    ... = ((1.15 * l * 1.25 * w - l * w) / (l * w)) * 100 : by rw [← mul_assoc, ← mul_assoc, mul_comm (1.15), mul_assoc]
    ... = 43.75 : sorry

end area_increase_percent_l11_11079


namespace three_digit_even_sum_12_l11_11035

theorem three_digit_even_sum_12 : 
  ∃ (n : Finset ℕ), 
    n.card = 27 ∧ 
    ∀ x ∈ n, 
      ∃ h t u, 
        (100 * h + 10 * t + u = x) ∧ 
        (h ∈ Finset.range 9 \ {0}) ∧ 
        (u % 2 = 0) ∧ 
        (t + u = 12) := 
sorry

end three_digit_even_sum_12_l11_11035


namespace calculate_num_chords_l11_11591

-- Definitions based on the problem conditions
def num_points_on_circle : ℕ := 10
def num_vertices_square : ℕ := 4

-- Main theorem stating the number of different chords
/-- Given 10 points on a circle, and 4 of these points form a square,
    prove that the number of different chords that can be drawn is 45. -/
theorem calculate_num_chords (n : ℕ) (k : ℕ) (hn : n = num_points_on_circle) (hk : k = num_vertices_square) :
  (choose n 2) = 45 :=
by
  rw [hn, hk]
  rw [Nat.choose_eq]
  sorry  -- actual proof logic skipped

end calculate_num_chords_l11_11591


namespace face_value_is_2880_l11_11187

-- Definitions of the conditions
def bank_discount : ℝ := 576
def true_discount : ℝ := 480
def relationship (FV : ℝ) : Prop :=
  bank_discount = true_discount + (true_discount * bank_discount) / FV

-- The proof statement
theorem face_value_is_2880 : ∃ (FV : ℝ), relationship FV ∧ FV = 2880 :=
by {
  use 2880,
  unfold relationship,
  have h1 : (true_discount * bank_discount) / 2880 = 96,
  { sorry },
  have h2 : true_discount + 96 = 576,
  { sorry },
  exact And.intro h2 rfl
}

end face_value_is_2880_l11_11187


namespace pascal_element_probability_l11_11736

open Nat

def num_elems_first_n_rows (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

def count_ones (n : ℕ) : ℕ :=
  if n = 0 then 1 else if n = 1 then 2 else 2 * (n - 1) + 1

def count_twos (n : ℕ) : ℕ :=
  if n < 2 then 0 else 2 * (n - 2)

def probability_one_or_two (n : ℕ) : ℚ :=
  let total_elems := num_elems_first_n_rows n in
  let ones := count_ones n in
  let twos := count_twos n in
  (ones + twos) / total_elems

theorem pascal_element_probability :
  probability_one_or_two 20 = 5 / 14 :=
by
  sorry

end pascal_element_probability_l11_11736


namespace truncated_cone_volume_l11_11625

theorem truncated_cone_volume (R r h : ℝ) (R_eq : R = 10) (r_eq : r = 5) (h_eq : h = 10) :
  let R_volume := (1/3 : ℝ) * π * (R^2) * (R + r + h)
  let r_volume := (1/3 : ℝ) * π * (r^2) * (r + R - h)
  let frustum_volume := R_volume - r_volume
  frustum_volume = (1750/3 : ℝ) * π :=
by
  intro R_eq r_eq h_eq
  unfold R_volume r_volume frustum_volume
  rw [R_eq, r_eq, h_eq]
  calc
    (1 / 3 * π * (10^2) * (10 + 5 + 10)) - (1 / 3 * π * (5^2) * (5 + 10 - 10))
      = (1 / 3 * π * (100) * (25)) - (1 / 3 * π * (25) * (5)) : by rw [sq, sq]
  ... = (1 / 3 * π * 2500) - (1 / 3 * π * 125) : by ring
  ... = (2500 / 3 * π) - (125 / 3 * π) : by field_simp
  ... = (1750 / 3) * π : by ring

end truncated_cone_volume_l11_11625


namespace floor_sqrt_50_squared_l11_11803

theorem floor_sqrt_50_squared :
  (let x := Real.sqrt 50 in (⌊x⌋₊ : ℕ)^2 = 49) :=
by
  sorry

end floor_sqrt_50_squared_l11_11803


namespace count_valid_even_numbers_with_sum_12_l11_11025

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n % 2 = 0) ∧ 
  ((n / 10) % 10 + n % 10 = 12)

theorem count_valid_even_numbers_with_sum_12 :
  (finset.range 1000).filter is_valid_number).card = 27 := by
  sorry

end count_valid_even_numbers_with_sum_12_l11_11025


namespace simplest_square_root_l11_11660

theorem simplest_square_root :
  (λ x, let y := x in y = sqrt 3) ∧ 
  (sqrt 3) = sqrt 3 ∧
  ∀ (y : ℝ), (y = sqrt (1 / 2)  ∨ y = sqrt 0.2 ∨ y = sqrt 3  ∨ y = sqrt 8) → 
  ( sqrt 3 = y → 
    ( ∃ (r : ℝ), y = 0 ∨ y = 1 ∨ y = r)
  )
:=
begin
  sorry
end

end simplest_square_root_l11_11660


namespace least_sum_of_bases_l11_11614

theorem least_sum_of_bases :
  ∃ (c d : ℕ), (5 * c + 7 = 7 * d + 5) ∧ (c > 0) ∧ (d > 0) ∧ (c + d = 14) :=
by
  sorry

end least_sum_of_bases_l11_11614


namespace arithmetic_sequence_common_difference_l11_11106

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : a 1 + a 7 = 22) 
  (h2 : a 4 + a 10 = 40) 
  (h_general_term : ∀ n : ℕ, a n = a 1 + (n - 1) * d) 
  : d = 3 :=
by 
  sorry

end arithmetic_sequence_common_difference_l11_11106


namespace infinitely_many_primes_satisfying_condition_l11_11988

theorem infinitely_many_primes_satisfying_condition :
  ∀ k : Nat, ∃ p : Nat, Nat.Prime p ∧ ∃ n : Nat, n > 0 ∧ p ∣ (2014^(2^n) + 2014) := 
sorry

end infinitely_many_primes_satisfying_condition_l11_11988


namespace base10_to_base4_addition_l11_11655

-- Define the base 10 numbers
def n1 : ℕ := 45
def n2 : ℕ := 28

-- Define the base 4 representations
def n1_base4 : ℕ := 2 * 4^2 + 3 * 4^1 + 1 * 4^0
def n2_base4 : ℕ := 1 * 4^2 + 3 * 4^1 + 0 * 4^0

-- The sum of the base 10 numbers
def sum_base10 : ℕ := n1 + n2

-- The expected sum in base 4
def sum_base4 : ℕ := 1 * 4^3 + 0 * 4^2 + 2 * 4^1 + 1 * 4^0

-- Prove the equivalence
theorem base10_to_base4_addition :
  (n1 + n2 = n1_base4  + n2_base4) →
  (sum_base10 = sum_base4) :=
by
  sorry

end base10_to_base4_addition_l11_11655


namespace problem1_problem2_l11_11762

-- Definition and conditions
def i := Complex.I

-- Problem 1
theorem problem1 : (2 + 2 * i) / (1 - i)^2 + (Real.sqrt 2 / (1 + i)) ^ 2010 = -1 := 
by
  sorry

-- Problem 2
theorem problem2 : (4 - i^5) * (6 + 2 * i^7) + (7 + i^11) * (4 - 3 * i) = 47 - 39 * i := 
by
  sorry

end problem1_problem2_l11_11762


namespace sum_of_palindromes_l11_11615

-- Define a three-digit palindrome predicate
def is_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ 0 ∧ b < 10 ∧ n = 100*a + 10*b + a

-- Define the product of the two palindromes equaling 436,995
theorem sum_of_palindromes (a b : ℕ) (h_a : is_palindrome a) (h_b : is_palindrome b) (h_prod : a * b = 436995) : 
  a + b = 1332 :=
sorry

end sum_of_palindromes_l11_11615


namespace ratio_a_to_c_l11_11171

theorem ratio_a_to_c : ∀ (a b c d : ℝ), (a / c = 4 / 5) ∧ (b / d = 4 / 5) → (a / c = 4 / 5) :=
by
  intros a b c d h
  cases h with h1 h2
  exact h1
  sorry

end ratio_a_to_c_l11_11171


namespace even_n_property_l11_11131

def S : Finset ℕ := Finset.range 1 101

def T (n : ℕ) : Finset (Fin n → ℕ) :=
  {v ∈ Finset.pi (Finset.const (Fin n) S) |
    (Finset.univ.sum (λ i, v i)) % 100 = 0}

def property_holds (n : ℕ) : Prop :=
  ∀ red_elements : Finset ℕ,
    red_elements.card = 75 → 
    (Finset.filter (λ t : Fin n → ℕ, 
      (Finset.univ.filter (λ i, t i ∈ red_elements)).card % 2 = 0) (T n)).card * 2 ≥ (T n).card

theorem even_n_property (n : ℕ) : 
  property_holds n ↔ n % 2 = 0 :=
sorry

end even_n_property_l11_11131


namespace even_three_digit_numbers_l11_11048

theorem even_three_digit_numbers (n : ℕ) :
  (n >= 100 ∧ n < 1000) ∧
  (n % 2 = 0) ∧
  ((n % 100) / 10 + (n % 10) = 12) →
  n = 12 :=
sorry

end even_three_digit_numbers_l11_11048


namespace equivalent_fraction_l11_11880

variable (α : ℝ)

def vector_a : ℝ × ℝ := (Real.cos α, 3)
def vector_b : ℝ × ℝ := (Real.sin α, -4)

theorem equivalent_fraction :
  (vector_a α).fst / (vector_b α).fst = (vector_a α).snd / (vector_b α).snd →
  (3 * Real.sin α + Real.cos α) / (2 * Real.cos α - 3 * Real.sin α) = -1 / 2 :=
by sorry

end equivalent_fraction_l11_11880


namespace kids_still_awake_l11_11279

-- Definition of the conditions
def num_kids_initial : ℕ := 20

def kids_asleep_first_5_minutes : ℕ := num_kids_initial / 2

def kids_awake_after_first_5_minutes : ℕ := num_kids_initial - kids_asleep_first_5_minutes

def kids_asleep_next_5_minutes : ℕ := kids_awake_after_first_5_minutes / 2

def kids_awake_final : ℕ := kids_awake_after_first_5_minutes - kids_asleep_next_5_minutes

-- Theorem that needs to be proved
theorem kids_still_awake : kids_awake_final = 5 := by
  sorry

end kids_still_awake_l11_11279


namespace pascal_element_probability_l11_11737

open Nat

def num_elems_first_n_rows (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

def count_ones (n : ℕ) : ℕ :=
  if n = 0 then 1 else if n = 1 then 2 else 2 * (n - 1) + 1

def count_twos (n : ℕ) : ℕ :=
  if n < 2 then 0 else 2 * (n - 2)

def probability_one_or_two (n : ℕ) : ℚ :=
  let total_elems := num_elems_first_n_rows n in
  let ones := count_ones n in
  let twos := count_twos n in
  (ones + twos) / total_elems

theorem pascal_element_probability :
  probability_one_or_two 20 = 5 / 14 :=
by
  sorry

end pascal_element_probability_l11_11737


namespace regular_tetrahedron_l11_11603

-- Let's define a structure for a tetrahedron with the given conditions.
structure Tetrahedron (V : Type) [inner_product_space ℝ V] :=
(vertex : Fin 4 → V)
(edges_equal : ∀ i j, i ≠ j → (i + 2) % 4 = j → ∥ vertex i - vertex ((i + 1) % 4) ∥ = ∥ vertex j - vertex ((j + 1) % 4) ∥)
(same_angle : ∀ i j k l, i ≠ j ∧ k ≠ l ∧ ((i + 2) % 4 = j) ∧ ((k + 2) % 4 = l) → 
  inner (vertex i - vertex ((i + 1) % 4)) (vertex k - vertex ((k + 1) % 4)) =
  inner (vertex j - vertex ((j + 1) % 4)) (vertex l - vertex ((l + 1) % 4)))

-- Define what it means for a Tetrahedron to be regular in terms of distances between vertices 
def is_regular {V : Type} [inner_product_space ℝ V] (T : Tetrahedron V) : Prop :=
∀ i j, i ≠ j → ∥ T.vertex i - T.vertex j ∥ = ∥ T.vertex (i + 1) % 4 - T.vertex (j + 1) % 4 ∥

-- Finally, we state the theorem without proof.
theorem regular_tetrahedron {V : Type} [inner_product_space ℝ V] (T : Tetrahedron V) :
  is_regular T :=
sorry

end regular_tetrahedron_l11_11603


namespace fraction_red_marbles_l11_11907

theorem fraction_red_marbles (x : ℕ) (h : x > 0) :
  let blue := (2/3 : ℚ) * x
  let red := (1/3 : ℚ) * x
  let new_red := 3 * red
  let new_total := blue + new_red
  new_red / new_total = (3/5 : ℚ) := by
  sorry

end fraction_red_marbles_l11_11907


namespace problem_1_problem_2_l11_11420

open Real

def f (x : ℝ) := exp x * (2 * x - 1)
def g (x a : ℝ) := a * x - a

theorem problem_1 (x0 : ℝ) (h_tangent : g x0 a = f x0 ∧ (differentiable_at ℝ f x0).fderiv ℝ f = a) :
  a = 1 ∨ a = 4 * exp (3 / 2) := sorry

theorem problem_2 (h_a : a < 1) (h_unique_int : ∃ x0 : ℤ, f x0 < g x0 a) :
  (3 / (2 * exp 1)) ≤ a ∧ a < 1 := sorry

end problem_1_problem_2_l11_11420


namespace carriage_cost_l11_11126

def distance : ℝ := 20
def speed : ℝ := 10
def hourly_rate : ℝ := 30
def flat_fee : ℝ := 20

theorem carriage_cost : (distance / speed) * hourly_rate + flat_fee = 80 := by
  have travel_time : ℝ := distance / speed
  have total_cost : ℝ := (travel_time * hourly_rate) + flat_fee
  rw [travel_time, show travel_time = 2 from by norm_num]
  rw [show total_cost = 80 from by norm_num]
  sorry

end carriage_cost_l11_11126


namespace area_increase_percentage_l11_11076

-- Define the original dimensions l and w as non-zero real numbers
variables (l w : ℝ) (hl : l ≠ 0) (hw : w ≠ 0)

-- Define the new dimensions after increase
def new_length := 1.15 * l
def new_width := 1.25 * w

-- Define the original and new areas
def original_area := l * w
def new_area := new_length l * new_width w

-- The statement to prove
theorem area_increase_percentage :
  ((new_area l w - original_area l w) / original_area l w) * 100 = 43.75 :=
by
  sorry

end area_increase_percentage_l11_11076


namespace triangle_arithmetic_progression_l11_11706

theorem triangle_arithmetic_progression (a d : ℝ) 
(h1 : (a-2*d)^2 + a^2 = (a+2*d)^2) 
(h2 : ∃ x : ℝ, (a = x * d) ∨ (d = x * a))
: (6 ∣ 6*d) ∧ (12 ∣ 6*d) ∧ (18 ∣ 6*d) ∧ (24 ∣ 6*d) ∧ (30 ∣ 6*d)
:= by
  sorry

end triangle_arithmetic_progression_l11_11706


namespace joel_age_when_dad_twice_l11_11943

theorem joel_age_when_dad_twice (x : ℕ) (h₁ : x = 22) : 
  let Joel_age := 5 + x 
  in Joel_age = 27 :=
by
  unfold Joel_age
  rw [h₁]
  norm_num

end joel_age_when_dad_twice_l11_11943


namespace eccentricity_of_given_ellipse_l11_11602

-- Define the given condition for the ellipse
def ellipse_equation (x y : ℝ) : Prop :=
  (x^2 / 6) + (y^2 / 8) = 1

-- Define the eccentricity based on the given equation
def eccentricity (e : ℝ) : Prop :=
  e = 1 / 2

-- The main statement of the proof
theorem eccentricity_of_given_ellipse :
  (∃ (x y : ℝ), ellipse_equation x y) → eccentricity 1/2 :=
by
  sorry

end eccentricity_of_given_ellipse_l11_11602


namespace sqrt_50_floor_squared_l11_11791

theorem sqrt_50_floor_squared : (⌊Real.sqrt 50⌋ : ℝ)^2 = 49 := by
  have sqrt_50_bounds : 7 < Real.sqrt 50 ∧ Real.sqrt 50 < 8 := by
    split
    · have : Real.sqrt 49 < Real.sqrt 50 := by sorry
      linarith
    · have : Real.sqrt 50 < Real.sqrt 64 := by sorry
      linarith
  have floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := by
    sorry
  rw [floor_sqrt_50]
  norm_num

end sqrt_50_floor_squared_l11_11791


namespace proof_triangle_properties_l11_11493

variable (A B C : ℝ)
variable (h AB : ℝ)

-- Conditions
def triangle_conditions : Prop :=
  (A + B = 3 * C) ∧ (2 * Real.sin (A - C) = Real.sin B) ∧ (AB = 5)

-- Part 1: Proving sin A
def find_sin_A (h₁ : triangle_conditions A B C h AB) : Prop :=
  Real.sin A = 3 * Real.cos A

-- Part 2: Proving the height on side AB
def find_height_on_AB (h₁ : triangle_conditions A B C h AB) : Prop :=
  h = 6

-- Combined proof statement
theorem proof_triangle_properties (h₁ : triangle_conditions A B C h AB) : 
  find_sin_A A B C h₁ ∧ find_height_on_AB A B C h AB h₁ := 
  by sorry

end proof_triangle_properties_l11_11493


namespace line_passes_through_fixed_point_l11_11610

theorem line_passes_through_fixed_point 
  (a b : ℝ) 
  (h : 2 * a + b = 1) : 
  a * 4 + b * 2 = 2 :=
sorry

end line_passes_through_fixed_point_l11_11610


namespace costly_tuple_exists_for_odd_n_exists_costly_tuple_with_odd_m_l11_11958

def is_costly_tuple (n : ℕ) (a : Fin n → ℕ) : Prop :=
  ∃ k : ℕ, (∀ i : Fin n, 0 < a i) ∧ (List.prod (List.ofFn (fun i => a i + a ((i + 1) % n))) = 2 ^ (2 * k - 1))

theorem costly_tuple_exists_for_odd_n (n : ℕ) (h : 2 ≤ n) : (∃ a : Fin n → ℕ, is_costly_tuple n a) ↔ (n % 2 = 1) :=
  sorry

theorem exists_costly_tuple_with_odd_m (m : ℕ) (hm : m % 2 = 1 ∧ 0 < m) : ∃ n ≥ 2, ∃ a : Fin n → ℕ, is_costly_tuple n a ∧ ∃ i : Fin n, a i = m :=
  sorry

end costly_tuple_exists_for_odd_n_exists_costly_tuple_with_odd_m_l11_11958


namespace sqrt_50_floor_squared_l11_11790

theorem sqrt_50_floor_squared : (⌊Real.sqrt 50⌋ : ℝ)^2 = 49 := by
  have sqrt_50_bounds : 7 < Real.sqrt 50 ∧ Real.sqrt 50 < 8 := by
    split
    · have : Real.sqrt 49 < Real.sqrt 50 := by sorry
      linarith
    · have : Real.sqrt 50 < Real.sqrt 64 := by sorry
      linarith
  have floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := by
    sorry
  rw [floor_sqrt_50]
  norm_num

end sqrt_50_floor_squared_l11_11790


namespace notebooks_count_l11_11944

-- Define the conditions
def totalCents : ℕ := 2545
def pricePerNotebook : ℕ := 235
def discountThreshold : ℕ := 5
def discountPerNotebook : ℕ := 15

-- Prove that John can buy exactly 11 notebooks
theorem notebooks_count : 
  let discountedPrice := pricePerNotebook - discountPerNotebook
  let bulkPrice := discountThreshold * discountedPrice
  let setsOfFive := totalCents / bulkPrice
  let remainingCents := totalCents - setsOfFive * bulkPrice
  let additionalNotebooks := remainingCents / pricePerNotebook in
  setsOfFive * discountThreshold + additionalNotebooks = 11 := by
  sorry

end notebooks_count_l11_11944


namespace gcd_problem_l11_11400

theorem gcd_problem (b : ℕ) (h : ∃ k : ℕ, b = 3150 * k) :
  gcd (b^2 + 9 * b + 54) (b + 4) = 2 := by
  sorry

end gcd_problem_l11_11400


namespace sin_alpha_minus_cos_alpha_eq_neg_sqrt2_div2_l11_11853

-- Define the problem statement and conditions
theorem sin_alpha_minus_cos_alpha_eq_neg_sqrt2_div2
  (α : ℝ)
  (h1 : sin α * cos α = 1 / 4)
  (h2 : 0 < α ∧ α < π / 4) :
  sin α - cos α = -real.sqrt 2 / 2 :=
sorry

end sin_alpha_minus_cos_alpha_eq_neg_sqrt2_div2_l11_11853


namespace saline_drip_drops_per_minute_l11_11295

-- Definitions of the conditions
def treatment_duration_hrs : ℝ := 2
def drops_per_5ml : ℕ := 100
def ml_received : ℕ := 120
def ml_per_5ml_drops : ℝ := 5
def minutes_per_hour : ℝ := 60

-- Derived from the given conditions
def total_minutes : ℝ := treatment_duration_hrs * minutes_per_hour
def total_drops : ℝ := (ml_received * drops_per_5ml) / ml_per_5ml_drops
def drops_per_minute : ℝ := total_drops / total_minutes

-- The theorem to prove
theorem saline_drip_drops_per_minute : drops_per_minute = 20 := by
  sorry

end saline_drip_drops_per_minute_l11_11295


namespace find_length_of_GH_l11_11906

variable {A B C F G H : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] 
          [MetricSpace F] [MetricSpace G] [MetricSpace H]

variables (AB BC Res : ℝ)
variables (ratio1 ratio2 : ℝ)
variable (similar : SimilarTriangles A B C F G H)

def length_of_GH (GH : ℝ) : Prop :=
  GH = 15

theorem find_length_of_GH (h1 : AB = 15) (h2 : BC = 25) (h3 : ratio1 = 5) (h4 : ratio2 = 3)
  (h5 : similar) : ∃ GH, length_of_GH GH :=
by
  have ratio : ratio2 / ratio1 = 3 / 5 := by assumption
  sorry

end find_length_of_GH_l11_11906


namespace question_one_question_two_question_three_l11_11412

-- Question 1
theorem question_one (a : ℝ) (h_a : a = 0) : 
  let f (x : ℝ) := (x^2 * (Real.exp (-x))) in
  deriv f 2 = 0 := sorry

-- Question 2
theorem question_two (a : ℝ) (h_a : a < 2) : 
  ∀ x : ℝ, 
  let f (x : ℝ) := (x^2 + a * x + a) * (Real.exp (-x)) in 
  (x = 0 → (∀ y, f(y) ≥ f 0)) := sorry

-- Question 3
theorem question_three (m : ℝ) :
  let g (x : ℝ) := (4 - x) * (Real.exp (x - 2)) in 
  ∀ x < 2, (3 * x - 2 * g x + m ≠ 0 ∧ (deriv g x < 1)) := sorry

end question_one_question_two_question_three_l11_11412


namespace find_functions_l11_11378

variable (X : Set ℝ) (f : ℝ → ℝ → ℝ) (x y z k : ℝ)
noncomputable theory

-- Condition 1: f(x, 1) = x and f(1, x) = x for all x ∈ X
axiom condition1 : ∀ x ∈ X, f(x, 1) = x ∧ f(1, x) = x 

-- Condition 2: f(f(x, y), z) = f(x, f(y, z)) for all x, y, z ∈ X
axiom condition2 : ∀ x y z ∈ X, f(f(x, y), z) = f(x, f(y, z))

-- Condition 3: f(xy, xz) = x^k f(y, z) for some fixed positive real k
axiom condition3 : 0 < k ∧ ∀ x y z ∈ X, f(x * y, x * z) = x^k * f(y, z)

theorem find_functions (hk1 hk2 : k = 1 ∨ k = 2) : 
  (∀ x y ∈ X, f(x, y) = x * y ∨ f(x, y) = min x y) :=
begin
  intro x,
  intro hx,
  intro y,
  intro hy,
  sorry
end

end find_functions_l11_11378


namespace triangle_area_ratio_l11_11314

section TriangleAreaRatio

variables {α : Type} [LinearOrder α] [Field α] 
variables (A B C D E F : α) -- Points, where A, B, C represent vertices of triangle ABC

-- Define segments as lengths
variables (AB BC CA : α) 
variables (BD CE AF : α)

-- Conditions given in the problem
def conditions : Prop :=
  (BD = AB / 2) ∧ (CE = BC / 2) ∧ (AF = CA / 2)

-- The statement to prove
theorem triangle_area_ratio (h : conditions AB BC CA BD CE AF) :
  area_ratio DEF ABC = 13 / 4 :=
sorry

end TriangleAreaRatio

end triangle_area_ratio_l11_11314


namespace sin_A_and_height_on_AB_l11_11516

theorem sin_A_and_height_on_AB 
  (A B C: ℝ)
  (h_triangle: ∀ A B C, A + B + C = π)
  (h_angle_sum: A + B = 3 * C)
  (h_sin_condition: 2 * Real.sin (A - C) = Real.sin B)
  (h_AB: AB = 5)
  (h_sqrt_two: Real.cos (π / 4) = Real.sin (π / 4) := by norm_num) :
  (Real.sin A = 3 * Real.sqrt 10 / 10) ∧ (height_on_AB = 6) :=
sorry

end sin_A_and_height_on_AB_l11_11516


namespace ellipse_C_equation_slopes_geometric_sequence_l11_11387

noncomputable def ellipse_equation (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c = sqrt 3) (h4 : a = 2 * c) : Prop :=
  ∃ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1

theorem ellipse_C_equation (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c = sqrt 3) (h4 : a = 2) (h5 : b^2 = a^2 - c^2) :
  ellipse_equation a b c h1 h2 h3 h4 := 
    ∃ (x y : ℝ), (x^2 / 4) + y^2 = 1 :=
sorry

theorem slopes_geometric_sequence (P Q : ℝ × ℝ) (h1 : P.1 > 0) (h2 : P.2 > 0) (h3 : Q.1 > 0) (h4 : Q.2 > 0) (h5 : ∃ m : ℝ, LineThrough(P, -1/2 * P.1 + m)) :
  ∃ (k_OP k_PQ k_OQ : ℝ), k_OP * k_OQ = k_PQ^2 :=
sorry

end ellipse_C_equation_slopes_geometric_sequence_l11_11387


namespace hexagon_side_length_proof_l11_11132

noncomputable def side_length_of_hexagon (AB AC angle_A : ℝ) (h1 : AB = 5) 
                                         (h2 : AC = 8) (h3 : angle_A = real.pi / 3) : ℝ :=
  let s := 40 / 21 in s

theorem hexagon_side_length_proof :
  side_length_of_hexagon 5 8 (real.pi / 3) = 40 / 21 :=
by
  sorry

end hexagon_side_length_proof_l11_11132


namespace distance_between_intersections_l11_11213

theorem distance_between_intersections :
  let C := 0
  let D := 0
  ∃(p q : ℕ), coprime p q ∧ p = 65 ∧ q = 2 ∧ (p - q = 63) :=
by
  sorry

end distance_between_intersections_l11_11213


namespace number_of_prime_factors_l11_11444

theorem number_of_prime_factors (n : ℕ) (h₁ : n < 200) (h₂ : ∃k : ℕ, 14 * n = 60 * k) : 
  nat.factors n.length ≥ 3 :=
by
  sorry

end number_of_prime_factors_l11_11444


namespace area_increase_percentage_l11_11077

-- Define the original dimensions l and w as non-zero real numbers
variables (l w : ℝ) (hl : l ≠ 0) (hw : w ≠ 0)

-- Define the new dimensions after increase
def new_length := 1.15 * l
def new_width := 1.25 * w

-- Define the original and new areas
def original_area := l * w
def new_area := new_length l * new_width w

-- The statement to prove
theorem area_increase_percentage :
  ((new_area l w - original_area l w) / original_area l w) * 100 = 43.75 :=
by
  sorry

end area_increase_percentage_l11_11077


namespace range_of_x_for_inequality_l11_11143

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + |x|) - 1 / (1 + x^2)

theorem range_of_x_for_inequality :
  {x : ℝ | f (2 * x) > f (x - 1)} = {x | x ∈ set.Ioo (-∞) (-1) ∪ set.Ioo (1/3) ∞} :=
by
  sorry

#check range_of_x_for_inequality

end range_of_x_for_inequality_l11_11143


namespace smallest_integer_larger_than_expr_is_248_l11_11652

noncomputable def small_int_larger_than_expr : ℕ :=
  let expr := (Real.sqrt 5 + Real.sqrt 3)^4
  248

theorem smallest_integer_larger_than_expr_is_248 :
    ∃ (n : ℕ), n > (Real.sqrt 5 + Real.sqrt 3)^4 ∧ n = small_int_larger_than_expr := 
by
  -- We introduce the target integer 248
  use (248 : ℕ)
  -- The given conditions should lead us to 248 being greater than the expression.
  sorry

end smallest_integer_larger_than_expr_is_248_l11_11652


namespace count_even_three_digit_numbers_with_sum_12_l11_11016

noncomputable def even_three_digit_numbers_with_sum_12 : Prop :=
  let valid_pairs := [(8, 4), (6, 6), (4, 8)] in
  let valid_hundreds := 9 in
  let count_pairs := valid_pairs.length in
  let total_numbers := valid_hundreds * count_pairs in
  total_numbers = 27

theorem count_even_three_digit_numbers_with_sum_12 : even_three_digit_numbers_with_sum_12 :=
by
  sorry

end count_even_three_digit_numbers_with_sum_12_l11_11016


namespace count_even_three_digit_sum_tens_units_is_12_l11_11055

-- Define what it means to be a three-digit number
def is_three_digit (n : ℕ) : Prop := (100 ≤ n) ∧ (n < 1000)

-- Define what it means to be even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the sum of the tens and units digits to be 12
def sum_tens_units_is_12 (n : ℕ) : Prop := 
  let tens := (n / 10) % 10 in
  let units := n % 10 in
  tens + units = 12

-- Count how many such numbers exist
theorem count_even_three_digit_sum_tens_units_is_12 : 
  ∃! n : ℕ, (is_three_digit n) ∧ (is_even n) ∧ (sum_tens_units_is_12 n) = 36 :=
sorry

end count_even_three_digit_sum_tens_units_is_12_l11_11055


namespace mean_of_solutions_l11_11772

theorem mean_of_solutions (x : ℝ) :
  (x^3 + 3 * x^2 - 44 * x = 0) → 
  let roots := [0, (-3 + Real.sqrt 185) / 2, (-3 - Real.sqrt 185) / 2] in
  let mean := (roots.sum) / 3 in
  mean = -1 :=
by 
  sorry

end mean_of_solutions_l11_11772


namespace pascal_triangle_probability_l11_11745

theorem pascal_triangle_probability :
  let total_elements := 20 * 21 / 2,
      ones := 1 + 19 * 2,
      twos := 18 * 2,
      elements := ones + twos in
  (total_elements = 210) →
  (ones = 39) →
  (twos = 36) →
  (elements = 75) →
  (75 / 210) = 5 / 14 :=
by
  intros,
  sorry

end pascal_triangle_probability_l11_11745


namespace min_difference_of_composite_sum_93_l11_11687

def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = n

theorem min_difference_of_composite_sum_93 :
  ∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ a + b = 93 ∧ ∀ (x y : ℕ), 
    is_composite x → is_composite y → x + y = 93 → (abs (x - y)) ≥ (abs (a - b)) := sorry

end min_difference_of_composite_sum_93_l11_11687


namespace sin_A_eq_height_on_AB_l11_11519

-- Defining conditions
variables {A B C : ℝ}
variables (AB : ℝ)

-- Conditions based on given problem
def condition1 : Prop := A + B = 3 * C
def condition2 : Prop := 2 * sin (A - C) = sin B
def condition3 : Prop := A + B + C = Real.pi

-- Question 1: prove that sin A = (3 * sqrt 10) / 10
theorem sin_A_eq:
  condition1 → 
  condition2 → 
  condition3 → 
  sin A = (3 * Real.sqrt 10) / 10 :=
by
  sorry

-- Question 2: given AB = 5, prove the height on side AB is 6
theorem height_on_AB:
  condition1 →
  condition2 →
  condition3 →
  AB = 5 →
  -- Let's construct the height as a function of A, B, and C
  ∃ h, h = 6 :=
by
  sorry

end sin_A_eq_height_on_AB_l11_11519


namespace blocks_differing_in_exactly_three_ways_l11_11690

-- Define block properties
inductive Material | plastic | wood | metal
inductive Size | small | medium | large
inductive Color | blue | green | red | yellow
inductive Shape | circle | hexagon | square | triangle
inductive Texture | smooth | rough

-- Define a block as a tuple of its properties
structure Block :=
(material : Material)
(size : Size)
(color : Color)
(shape : Shape)
(texture : Texture)

-- Specific block
def specificBlock : Block :=
  { material := Material.plastic,
    size := Size.medium,
    color := Color.red,
    shape := Shape.circle,
    texture := Texture.smooth }

-- Statement to prove
theorem blocks_differing_in_exactly_three_ways :
  (count (fun b : Block, differs_in_exactly_three_ways b specificBlock) (blocks_list) = 34) :=
sorry

-- Helper function to check if two blocks differ in exactly three ways
def differs_in_exactly_three_ways (b1 b2 : Block) : Bool :=
  (if b1.material ≠ b2.material then 1 else 0) +
  (if b1.size ≠ b2.size then 1 else 0) +
  (if b1.color ≠ b2.color then 1 else 0) +
  (if b1.shape ≠ b2.shape then 1 else 0) +
  (if b1.texture ≠ b2.texture then 1 else 0) = 3

-- Define all possible blocks (cross product of all property options)
noncomputable def blocks_list : List Block :=
  [ -- manually enumerate all combinations (omitted here for brevity)
  ]


end blocks_differing_in_exactly_three_ways_l11_11690


namespace series_sum_l11_11959

noncomputable def s : ℝ := Classical.some (exists_unique.mpr
    (by simp [real.polynomial_exists_unique_root_of_monic]))

theorem series_sum (hs : s^3 + (1/4) * s - 1 = 0) :
  (s^2 + 2*s^5 + 3*s^8 + 4*s^11 + ∑ n in finset.range 100 (λ n, n * s^(3*n + 2))) = 16 := by
begin
  sorry
end

end series_sum_l11_11959


namespace part_a_part_b_l11_11554

variables {S : Type*} [fintype S] (n : ℕ) (p_n : ℕ → ℕ)

-- Condition: S is a set of n elements
def card_S_eq_n : fintype.card S = n

-- p_n(k) denotes the number of all permutations of S that have exactly k fixed points
def pn_definition (k : ℕ) : ℕ := p_n k

theorem part_a : ∑ k in finset.range (n + 1), k * p_n k = nat.factorial n :=
by sorry

theorem part_b : ∑ k in finset.range (n + 1), (k - 1)^2 * p_n k = nat.factorial n :=
by sorry

end part_a_part_b_l11_11554


namespace parabola_hyperbola_intersection_l11_11627

theorem parabola_hyperbola_intersection :
  ∃ c a b : ℝ, 
    (∀ x y : ℝ, y^2 = 4 * c * x ↔ y^2 = 4 * x) ∧
    (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ↔ (4 * x^2 - (y^2 / (3 / 4)) = 1)) ∧
    (c = 1 ∧ a^2 = 1 / 4 ∧ b^2 = 3 / 4) := 
by 
  have c := 1
  have a := 1 / 2
  have b := sqrt 3 / 2
  use [c, a, b]
  -- proving parabola equation 
  show ∀ x y : ℝ, y^2 = 4 * c * x ↔ y^2 = 4 * x by 
    intro x y
    simp
    
  -- proving hyperbola equation 
  show ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ 4 * x^2 - y^2 / (3 / 4) = 1 by 
    intro x y
    simp [a, b]
  
  -- final conditions on constants 
  show c = 1 ∧ a^2 = 1 / 4 ∧ b^2 = 3 / 4 by 
    simp
    sorry

end parabola_hyperbola_intersection_l11_11627


namespace sum_of_solutions_l11_11359

theorem sum_of_solutions (y : ℝ) :
  (2 : ℝ)^(y^2 - 5*y - 6) = (4 : ℝ)^(y - 5) →
  (y = 4 ∨ y = 1) →
  ∃ (y1 y2 : ℝ), y1 + y2 = 5 :=
by
  intro h1 h2
  use [4, 1]
  exact (by linarith : 4 + 1 = 5)
  -- sorry

end sum_of_solutions_l11_11359


namespace equilateral_triangles_ae_div_bc_l11_11634

theorem equilateral_triangles_ae_div_bc (s : ℝ) (h : 0 < s) 
  (ABC BCD CDE : EquilateralTriangle s) : AE / BC = 3 * Real.sqrt 3 / 2 := 
by sorry

end equilateral_triangles_ae_div_bc_l11_11634


namespace volume_of_orange_concentrate_l11_11695

theorem volume_of_orange_concentrate
  (h_jug : ℝ := 8) -- height of the jug in inches
  (d_jug : ℝ := 3) -- diameter of the jug in inches
  (fraction_full : ℝ := 3 / 4) -- jug is three-quarters full
  (ratio_concentrate_to_water : ℝ := 1 / 5) -- ratio of concentrate to water
  : abs ((fraction_full * π * ((d_jug / 2)^2) * h_jug * (1 / (1 + ratio_concentrate_to_water))) - 2.25) < 0.01 :=
by
  sorry

end volume_of_orange_concentrate_l11_11695


namespace pascal_triangle_probability_l11_11743

theorem pascal_triangle_probability :
  let total_elements := 20 * 21 / 2,
      ones := 1 + 19 * 2,
      twos := 18 * 2,
      elements := ones + twos in
  (total_elements = 210) →
  (ones = 39) →
  (twos = 36) →
  (elements = 75) →
  (75 / 210) = 5 / 14 :=
by
  intros,
  sorry

end pascal_triangle_probability_l11_11743


namespace marble_arrangements_l11_11667

open Finset

-- Definitions based on problem conditions:
def marbles : Finset (List Char) := univ.image (λ l : List Char, 
  if (l.length = 5 ∧ ('S', 'T') ∉ zip l.tail l ∧ ('T', 'S') ∉ zip l l.tail ∧
      ('S', 'C') ∉ zip l.tail l ∧ ('C', 'S') ∉ zip l l.tail ∧
      'A' ∈ l ∧ 'B' ∈ l ∧ 'S' ∈ l ∧ 'T' ∈ l ∧ 'C' ∈ l) then l else ['A', 'B', 'S', 'T', 'C'])

-- Theorem statement to prove the number of valid arrangement ways:
theorem marble_arrangements : (marbles.filter (λ l, sorry)).card = 48 := 
sorry

end marble_arrangements_l11_11667


namespace f_f_2010_eq_neg1_l11_11866

def f (x : ℝ) : ℝ :=
  if x ≤ 2000 then 2 * Real.cos ( (↑π / 3) * x)
  else x - 100

theorem f_f_2010_eq_neg1 : f (f 2010) = -1 :=
by
  -- Proof omitted
  sorry

end f_f_2010_eq_neg1_l11_11866


namespace count_even_three_digit_numbers_sum_tens_units_eq_12_l11_11004

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def is_even (n : ℕ) : Prop := n % 2 = 0
def sum_of_tens_and_units_eq_12 (n : ℕ) : Prop :=
  (n / 10) % 10 + n % 10 = 12

theorem count_even_three_digit_numbers_sum_tens_units_eq_12 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_three_digit n ∧ is_even n ∧ sum_of_tens_and_units_eq_12 n) ∧ S.card = 24 :=
sorry

end count_even_three_digit_numbers_sum_tens_units_eq_12_l11_11004


namespace mike_earnings_l11_11566

theorem mike_earnings :
  let total_games := 16
  let non_working_games := 8
  let price_per_game := 7
  let working_games := total_games - non_working_games
  total_games = 16 → non_working_games = 8 → 
  price_per_game = 7 → working_games * price_per_game = 56 :=
by
  intros total_games_eq non_working_games_eq price_per_game_eq
  let working_games := total_games - non_working_games
  have working_games_eq : working_games = 8 := by
    rw [total_games_eq, non_working_games_eq]
    norm_num
  have earnings_eq : working_games * price_per_game = 56 := by
    rw [working_games_eq, price_per_game_eq]
    norm_num
  exact earnings_eq

end mike_earnings_l11_11566


namespace angle_PRT_correct_l11_11583

-- Declare the segmented lengths and midpoint properties
def length_PQ : ℝ := 12
def length_QR : ℝ := length_PQ / 2

-- Given the radius calculations
def radius_PQ : ℝ := length_PQ / 2
def radius_QR : ℝ := radius_PQ / 2

-- Semi-circle areas
def area_PQ : ℝ := real.pi * radius_PQ^2
def area_QR : ℝ := real.pi * radius_QR^2

-- Total area and half-area
def total_area : ℝ := area_PQ + area_QR
def half_area : ℝ := total_area / 2

-- The decimal number representing the correct degree measure of angle PRT
def angle_PRT_degrees : ℝ := 180 * (half_area / area_PQ)

theorem angle_PRT_correct (h_lengthPQ : length_PQ = 12)
  (h_radiusPQ : radius_PQ = length_PQ / 2)
  (h_radiusQR : radius_QR = radius_PQ / 2)
  (h_areaPQ : area_PQ = real.pi * radius_PQ^2)
  (h_areaQR : area_QR = real.pi * radius_QR^2)
  (h_total_area : total_area = area_PQ + area_QR)
  (h_half_area : half_area = total_area / 2)
  : angle_PRT_degrees = 112.5 := by
  sorry

end angle_PRT_correct_l11_11583


namespace pascal_triangle_probability_l11_11731

-- Define the probability problem in Lean 4
theorem pascal_triangle_probability :
  let total_elements := ((20 * (20 + 1)) / 2)
  let ones_count := (1 + 2 * 19)
  let twos_count := (2 * (19 - 2 + 1))
  (ones_count + twos_count) / total_elements = 5 / 14 :=
by
  let total_elements := ((20 * (20 + 1)) / 2)
  let ones_count := (1 + 2 * 19)
  let twos_count := (2 * (19 - 2 + 1))
  have h1 : total_elements = 210 := by sorry
  have h2 : ones_count = 39 := by sorry
  have h3 : twos_count = 36 := by sorry
  have h4 : (39 + 36) / 210 = 5 / 14 := by sorry
  exact h4

end pascal_triangle_probability_l11_11731


namespace volume_of_tetrahedron_eq_one_six_abc_sin_alpha_l11_11987
noncomputable theory

variables (a b c : ℝ) (α : ℝ)

-- Definition for volume of tetrahedron
def volume_tetrahedron : ℝ :=
  (1 / 6) * a * b * c * Real.sin α

-- Statement we want to prove
theorem volume_of_tetrahedron_eq_one_six_abc_sin_alpha
  (a b c : ℝ) (α : ℝ) :
  volume_tetrahedron a b c α = (1 / 6) * a * b * c * Real.sin α :=
sorry

end volume_of_tetrahedron_eq_one_six_abc_sin_alpha_l11_11987


namespace probability_of_three_blue_marbles_l11_11564

section ProbabilityProblem
variables (blue_marbles red_marbles total_picks : ℕ)

-- Given conditions
def blue_marbles := 8
def red_marbles := 7
def total_picks := 7

-- Probability space
def prob_blue : ℚ := 8 / 15
def prob_red : ℚ := 7 / 15

-- Number of ways to pick exactly 3 blue marbles
def ways_to_pick_blue := nat.choose total_picks 3

-- Total probability calculation
noncomputable def prob_exactly_three_blue :=
  (ways_to_pick_blue : ℚ) * (prob_blue ^ 3) * (prob_red ^ (total_picks - 3))

-- Stating the problem as a theorem
theorem probability_of_three_blue_marbles :
  prob_exactly_three_blue = 862 / 3417 :=
sorry
end

end probability_of_three_blue_marbles_l11_11564


namespace nested_sqrt_eq_two_l11_11442

theorem nested_sqrt_eq_two (y : ℝ) (h : y = Real.sqrt (2 + y)) : y = 2 :=
by
  sorry

end nested_sqrt_eq_two_l11_11442


namespace express_2_175_billion_in_scientific_notation_l11_11157

-- Definition of scientific notation
def scientific_notation (a : ℝ) (n : ℤ) (value : ℝ) : Prop :=
  value = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10

-- Theorem stating the problem
theorem express_2_175_billion_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), scientific_notation a n 2.175e9 ∧ a = 2.175 ∧ n = 9 :=
by
  sorry

end express_2_175_billion_in_scientific_notation_l11_11157


namespace midpoints_of_equal_perimeter_l11_11528

-- Define the initial conditions
variables {A B C C1 A1 B1 : Type} 
variables [triangle ABC: A ≠ B ∧ B ≠ C ∧ C ≠ A]
variables [C1_on_AB: C1 ∈ line_segment A B]
variables [A1_on_BC: A1 ∈ line_segment B C]
variables [B1_on_CA: B1 ∈ line_segment C A]

-- Define the equal perimeter condition
def equal_perimeters (ABC A1 B1 C1: Type) : Prop :=
  perimeter A B1 C1 = perimeter B C1 A1 ∧
  perimeter B C1 A1 = perimeter C A1 B1 ∧
  perimeter C A1 B1 = perimeter A1 B1 C1

-- The proof statement
theorem midpoints_of_equal_perimeter (h: equal_perimeters ABC A1 B1 C1) :
  is_midpoint A1 (B C) ∧ is_midpoint B1 (C A) ∧ is_midpoint C1 (A B) :=
sorry

end midpoints_of_equal_perimeter_l11_11528


namespace pascal_triangle_probability_l11_11732

-- Define the probability problem in Lean 4
theorem pascal_triangle_probability :
  let total_elements := ((20 * (20 + 1)) / 2)
  let ones_count := (1 + 2 * 19)
  let twos_count := (2 * (19 - 2 + 1))
  (ones_count + twos_count) / total_elements = 5 / 14 :=
by
  let total_elements := ((20 * (20 + 1)) / 2)
  let ones_count := (1 + 2 * 19)
  let twos_count := (2 * (19 - 2 + 1))
  have h1 : total_elements = 210 := by sorry
  have h2 : ones_count = 39 := by sorry
  have h3 : twos_count = 36 := by sorry
  have h4 : (39 + 36) / 210 = 5 / 14 := by sorry
  exact h4

end pascal_triangle_probability_l11_11732


namespace interior_points_of_segment_interior_points_except_boundary_boundary_or_interior_points_of_segment_l11_11270

-- 3a
theorem interior_points_of_segment {A B : Point} {Φ : ConvexFigure} 
  (h1 : A ∈ int(Φ)) (h2 : B ∈ int(Φ)) : ∀ D ∈ line_segment(A, B), D ∈ int(Φ) :=
sorry

-- 3b
theorem interior_points_except_boundary {A B : Point} {Φ : ConvexFigure} 
  (h1 : A ∈ int(Φ)) (h2 : B ∈ bd(Φ)) : ∀ D ∈ line_segment(A, B) \ {B}, D ∈ int(Φ) :=
sorry

-- 3c
theorem boundary_or_interior_points_of_segment {A B : Point} {Φ : ConvexFigure} 
  (h1 : A ∈ bd(Φ)) (h2 : B ∈ bd(Φ)) : 
  (∀ D ∈ line_segment(A, B), D ∈ bd(Φ)) ∨ (∀ D ∈ line_segment(A, B) \ {A, B}, D ∈ int(Φ)) :=
sorry

end interior_points_of_segment_interior_points_except_boundary_boundary_or_interior_points_of_segment_l11_11270


namespace negation_of_universal_l11_11168
-- Import the Mathlib library to provide the necessary mathematical background

-- State the theorem that we want to prove. This will state that the negation of the universal proposition is an existential proposition
theorem negation_of_universal :
  (¬ (∀ x : ℝ, x > 0)) ↔ (∃ x : ℝ, x ≤ 0) :=
sorry

end negation_of_universal_l11_11168


namespace sharing_watermelons_l11_11779

theorem sharing_watermelons (h : 8 = people_per_watermelon) : people_for_4_watermelons = 32 :=
by
  let people_per_watermelon := 8
  let watermelons := 4
  let people_for_4_watermelons := people_per_watermelon * watermelons
  sorry

end sharing_watermelons_l11_11779


namespace billy_video_suggestions_l11_11751

theorem billy_video_suggestions (total_videos : ℕ) (suggestions_per_list : ℕ) (final_video : ℕ) :
  total_videos = 65 ∧ suggestions_per_list = 15 ∧ final_video = 5 →
  let iterations_without_success := (total_videos - final_video) / suggestions_per_list in
  iterations_without_success + 1 = 5 :=
by
  intros h
  cases h with ht hrest
  cases hrest with hs hf
  rw [ht, hs, hf]
  sorry

end billy_video_suggestions_l11_11751


namespace happy_children_count_l11_11156

-- Definitions of the conditions
def total_children : ℕ := 60
def sad_children : ℕ := 10
def neither_happy_nor_sad_children : ℕ := 20
def boys : ℕ := 22
def girls : ℕ := 38
def happy_boys : ℕ := 6
def sad_girls : ℕ := 4
def boys_neither_happy_nor_sad : ℕ := 10

-- The theorem we wish to prove
theorem happy_children_count :
  total_children - sad_children - neither_happy_nor_sad_children = 30 :=
by 
  -- Placeholder for the proof
  sorry

end happy_children_count_l11_11156


namespace sum_first_100_natural_numbers_l11_11754

theorem sum_first_100_natural_numbers :
  let S : ℕ → ℕ := λ n, n * (n + 1) / 2
  S 100 = 5050 :=
by
  sorry

end sum_first_100_natural_numbers_l11_11754


namespace find_focus_ratio_l11_11954

def parabola_vertex (V : Point) (P : Point → Prop) :=
  ∃ a b, P = (λ p : Point, p.y = a * p.x^2 + b) ∧ V = (0, b)

def parabola_focus (F : Point) (P : Point → Prop) :=
  ∃ a b, P = (λ p : Point, p.y = a * p.x^2 + b) ∧ F = (0, 1/(4*a) + b)

def midpoint_of_AB_on_parabola (A B M : Point) (P : Point → Prop) :=
  (P A ∧ P B ∧ angle A V B = 90°) → 
  M = (A + B) / 2

def new_parabola (Q : Point → Prop) (M : Point → Prop) :=
  ∀ A B, midpoint_of_AB_on_parabola A B M P → Q M

theorem find_focus_ratio 
  (V1 F1 V2 F2 : Point) 
  (P Q : Point → Prop) 
  (A B M : Point) 
  (V1_is_vertex : parabola_vertex V1 P) 
  (F1_is_focus  : parabola_focus F1 P) 
  (V2_is_vertex : parabola_vertex V2 Q) 
  (F2_is_focus  : parabola_focus F2 Q)
  (locus_of_midpoints: new_parabola Q P):
  dist F1 F2 / dist V1 V2 = 7 / 8 := 
sorry

end find_focus_ratio_l11_11954


namespace find_a_l11_11209

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x^2 + 3 * x - 9

theorem find_a (a : ℝ) : (∀ x : ℝ, f x a = x^3 + a * x^2 + 3 * x - 9) 
  → (∃ x : ℝ, x = -3 ∧ has_deriv_at (f x a) 0 x) → a = 5 :=
begin
  sorry
end

end find_a_l11_11209


namespace sin_A_and_height_on_AB_l11_11513

theorem sin_A_and_height_on_AB 
  (A B C: ℝ)
  (h_triangle: ∀ A B C, A + B + C = π)
  (h_angle_sum: A + B = 3 * C)
  (h_sin_condition: 2 * Real.sin (A - C) = Real.sin B)
  (h_AB: AB = 5)
  (h_sqrt_two: Real.cos (π / 4) = Real.sin (π / 4) := by norm_num) :
  (Real.sin A = 3 * Real.sqrt 10 / 10) ∧ (height_on_AB = 6) :=
sorry

end sin_A_and_height_on_AB_l11_11513


namespace max_value_of_expression_l11_11824

theorem max_value_of_expression : 
  ∃ y : ℝ, y = 2^x → (∃ x : ℝ, ∀ y = 2^x, max (2^x - 16^x) = 3 / (4 * real.cbrt 4)) :=
sorry

end max_value_of_expression_l11_11824


namespace count_even_three_digit_numbers_sum_tens_units_eq_12_l11_11007

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def is_even (n : ℕ) : Prop := n % 2 = 0
def sum_of_tens_and_units_eq_12 (n : ℕ) : Prop :=
  (n / 10) % 10 + n % 10 = 12

theorem count_even_three_digit_numbers_sum_tens_units_eq_12 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_three_digit n ∧ is_even n ∧ sum_of_tens_and_units_eq_12 n) ∧ S.card = 24 :=
sorry

end count_even_three_digit_numbers_sum_tens_units_eq_12_l11_11007


namespace smallest_base_l11_11249

theorem smallest_base (k b : ℕ) (h_k : k = 6) : 64 ^ k > b ^ 16 ↔ b < 5 :=
by
  have h1 : 64 ^ k = 2 ^ (6 * k) := by sorry
  have h2 : 2 ^ (6 * k) > b ^ 16 := by sorry
  exact sorry

end smallest_base_l11_11249


namespace x_n_difference_bound_l11_11960

noncomputable def x_n (n : ℕ) : ℝ :=
  ∑ k in Finset.range ((n+1)^2) \ Finset.range (n^2), 1 / (k : ℝ)

theorem x_n_difference_bound (n : ℕ) (hn : 0 < n) :
  0 < x_n n - x_n (n + 1) ∧ x_n n - x_n (n + 1) < 4 / (n * (n + 2)) :=
sorry

end x_n_difference_bound_l11_11960


namespace floor_sqrt_50_squared_l11_11801

theorem floor_sqrt_50_squared :
  (let x := Real.sqrt 50 in (⌊x⌋₊ : ℕ)^2 = 49) :=
by
  sorry

end floor_sqrt_50_squared_l11_11801


namespace divide_into_L_pieces_l11_11385

-- Definitions extracted from the problem statement.
def grid (n : ℕ) : Type := (fin (3 * n + 1 + 1)) × (fin (3 * n + 1 + 1))

def removed_square (n : ℕ) : grid n := sorry -- definition of the removed square

def L_shaped_piece (n : ℕ) : Type := sorry -- definition of an L-shaped piece

theorem divide_into_L_pieces (n : ℕ) (rem : grid n) : 
  sorry -- definition of the remaining grid 
  ∃ (L_pieces : list (L_shaped_piece n)), 
  -- condition that L_pieces cover the remaining grid
  sorry :=
sorry

end divide_into_L_pieces_l11_11385


namespace count_even_three_digit_numbers_with_sum_12_l11_11018

noncomputable def even_three_digit_numbers_with_sum_12 : Prop :=
  let valid_pairs := [(8, 4), (6, 6), (4, 8)] in
  let valid_hundreds := 9 in
  let count_pairs := valid_pairs.length in
  let total_numbers := valid_hundreds * count_pairs in
  total_numbers = 27

theorem count_even_three_digit_numbers_with_sum_12 : even_three_digit_numbers_with_sum_12 :=
by
  sorry

end count_even_three_digit_numbers_with_sum_12_l11_11018


namespace sin_A_correct_height_on_AB_correct_l11_11480

noncomputable def sin_A (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) : ℝ :=
  Real.sin A

noncomputable def height_on_AB (A B C AB : ℝ) (height : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) (h4 : AB = 5) : ℝ :=
  height

theorem sin_A_correct (A B C : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) : 
  sorrry := 
begin
  -- proof omitted
  sorrry
end

theorem height_on_AB_correct (A B C AB : ℝ) (height : ℝ) (h1 : A + B = 3 * C) (h2 : 2 * Real.sin (A - C) = Real.sin B) (h3 : A + B + C = Real.pi) (h4 : AB = 5) :
  height = 6:= 
begin
  -- proof omitted
  sorrry
end 

end sin_A_correct_height_on_AB_correct_l11_11480


namespace range_f_period_f_monotonic_increase_intervals_l11_11828

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.sin x) ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 1 

theorem range_f : Set.Icc 0 4 = Set.range f := sorry

theorem period_f : ∀ x, f (x + Real.pi) = f x := sorry

theorem monotonic_increase_intervals (k : ℤ) :
  ∀ x, (-π / 6 + k * π : ℝ) ≤ x ∧ x ≤ (π / 3 + k * π : ℝ) → 
        ∀ y, f y ≤ f x → y ≤ x := sorry

end range_f_period_f_monotonic_increase_intervals_l11_11828


namespace monotonic_decreasing_interval_of_function_l11_11613

noncomputable def is_monotonically_decreasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x ≤ y → f y ≤ f x

theorem monotonic_decreasing_interval_of_function :
  is_monotonically_decreasing (λ x, 0.2 ^ abs (x - 1)) {x : ℝ | 1 < x} :=
sorry

end monotonic_decreasing_interval_of_function_l11_11613


namespace find_a_for_parallel_lines_l11_11069

theorem find_a_for_parallel_lines (a : ℝ) :
  (∀ x y : ℝ, ax + 3 * y + 1 = 0 ↔ 2 * x + (a + 1) * y + 1 = 0) → a = -3 :=
by
  sorry

end find_a_for_parallel_lines_l11_11069


namespace product_logarithms_eq_3_l11_11367

theorem product_logarithms_eq_3 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) 
  (hterms : ∃ n = 1000, 
    (finset.range (b - 1) \ finset.range a).card = n)
  (hvalue : (finset.range (b - 1) \ finset.range a).val.map 
                (λ k, real.log (k + 1) / real.log k)).prod = 3 : a + b = 1010 :=
sorry

end product_logarithms_eq_3_l11_11367


namespace triangle_circumcircle_QR_length_l11_11085

theorem triangle_circumcircle_QR_length
  (A B C Q R : Type*)
  (dAB : dist A B = 13)
  (dAC : dist A C = 12)
  (dBC : dist B C = 5)
  (circumcircle : circle)
  (tangent_to_AB : is_tangent circumcircle (line_through A B))
  (Q_on_AC : Q ≠ C ∧ on_circle Q circumcircle ∧ on_line Q (line_through A C))
  (R_on_BC : R ≠ C ∧ on_circle R circumcircle ∧ on_line R (line_through B C)) :
  dist Q R = 13 := sorry

end triangle_circumcircle_QR_length_l11_11085
