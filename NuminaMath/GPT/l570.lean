import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Ring
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.Calculus.ContDiff
import Mathlib.Analysis.Calculus.Continuity
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.SpecialFunctions.Gamma
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.VectorSpace.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Binomial
import Mathlib.Combinatorics.Catalan
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Lemmas
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Powers
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Pi.Basic
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Probability.ProbabilityTheory
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.GraphTheory.ChromaticNumber
import Mathlib.GraphTheory.FlowNumber
import Mathlib.Init.Function
import Mathlib.NumberTheory.Basic
import Mathlib.NumberTheory.Gcd
import Mathlib.Probability.Basic
import Mathlib.Probability.Independence
import Mathlib.Probability.NormalDistribution
import Mathlib.Probability.Probability
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Probability.Variance
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Basic
import Probability

namespace flower_beds_fraction_l570_570698

noncomputable def area_triangle (leg: ℝ) : ℝ := (leg * leg) / 2
noncomputable def area_rectangle (length width: ℝ) : ℝ := length * width
noncomputable def area_trapezoid (a b height: ℝ) : ℝ := ((a + b) * height) / 2

theorem flower_beds_fraction : 
  ∀ (leg len width a b height total_length: ℝ),
    a = 30 →
    b = 40 →
    height = 6 →
    total_length = 60 →
    leg = 5 →
    len = 20 →
    width = 5 →
    (area_rectangle len width + 2 * area_triangle leg) / (area_trapezoid a b height + area_rectangle len width) = 125 / 310 :=
by
  intros
  sorry

end flower_beds_fraction_l570_570698


namespace num_two_digit_factorizations_of_1995_l570_570883

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def is_valid_factorization (a b : ℕ) : Prop := a * b = 1995 ∧ is_two_digit a ∧ is_two_digit b

theorem num_two_digit_factorizations_of_1995 : 
  {ab : ℕ × ℕ // is_valid_factorization ab.fst ab.snd}.card = 2 :=
sorry

end num_two_digit_factorizations_of_1995_l570_570883


namespace chromatic_equals_flow_l570_570804

variables {G : Type*} [planar_multigraph G] -- Assuming G is a planar multigraph type
variables {G_star : Type*} [dual_graph G G_star] -- Assuming G_star is the dual graph of G

theorem chromatic_equals_flow (G : Type*) [planar_multigraph G]
  (G_star : Type*) [dual_graph G G_star] :
  chromatic_number G = flow_number G_star :=
sorry

end chromatic_equals_flow_l570_570804


namespace smallest_positive_real_is_131_div_11_l570_570771

noncomputable def smallest_positive_real_satisfying_condition :=
  ∀ (x : ℝ), (∀ y > 0, (y * y ⌊y⌋ - y ⌊y⌋ = 10) → (x ≤ y)) → 
  (⌊x*x⌋ - (x * ⌊x⌋) = 10) → 
  x = 131/11

theorem smallest_positive_real_is_131_div_11 :
  smallest_positive_real_satisfying_condition := sorry

end smallest_positive_real_is_131_div_11_l570_570771


namespace perpendicular_unique_l570_570660

theorem perpendicular_unique {P : Type} [plane : EuclideanGeometry P] (l : P → Prop) (p : P) :
  (∃! m : P → Prop, (P := plane) (m p) ∧ ∀ x, m x → l x → perpendicular (P := plane) m l) → sorry

end perpendicular_unique_l570_570660


namespace trig_identity_1_trig_identity_2_l570_570812

theorem trig_identity_1 (θ : ℝ) (h₀ : 0 < θ ∧ θ < π / 2) (h₁ : Real.tan θ = 2) :
  (Real.sin (π - θ) + Real.sin (3 * π / 2 + θ)) / 
  (3 * Real.sin (π / 2 - θ) - 2 * Real.sin (π + θ)) = 1 / 7 :=
by sorry

theorem trig_identity_2 (θ : ℝ) (h₀ : 0 < θ ∧ θ < π / 2) (h₁ : Real.tan θ = 2) :
  (1 - Real.cos (2 * θ)) / 
  (Real.sin (2 * θ) + Real.cos (2 * θ)) = 8 :=
by sorry

end trig_identity_1_trig_identity_2_l570_570812


namespace stratified_sampling_third_year_students_l570_570316

theorem stratified_sampling_third_year_students 
  (n_first : ℕ) (n_second : ℕ) (n_third : ℕ) (total_selected : ℕ)
  (h_first : n_first = 1300) (h_second : n_second = 1200) (h_third : n_third = 1500)
  (h_total_selected : total_selected = 200) :
  let total_students := n_first + n_second + n_third,
      sampling_ratio := (total_selected : ℝ) / total_students in
  n_third * sampling_ratio = 75 :=
by
  let total_students : ℕ := n_first + n_second + n_third
  have h_total_students : total_students = 4000 := by linarith
  let sampling_ratio : ℝ := (total_selected : ℝ) / total_students
  have h_sampling_ratio : sampling_ratio = 1 / 20 := by
    rw [h_total_selected]
    norm_num
    rw [h_total_students]
    norm_num
  have h_third_year_sample : n_third * (1 / 20) = 75 := by
    rw [h_third]
    norm_num
  rw [h_sampling_ratio]
  exact h_third_year_sample

end stratified_sampling_third_year_students_l570_570316


namespace selling_price_correct_l570_570205

theorem selling_price_correct (C M D: ℝ) (S: ℝ) : 
  C = 540 → M = 15 → D = 25.603864734299517 → S = ((C + (M / 100) * C) - (D / 100) * (C + (M / 100) * C)) → S = 462.036 :=
by 
  intros hC hM hD hS
  rw [hC, hM, hD] at hS
  linarith
-- sorry

end selling_price_correct_l570_570205


namespace find_a_plus_c_l570_570817

variables {a b c d : ℝ}

-- Conditions
def condition1 : Prop := a * b + b * c + c * d + d * a = 42
def condition2 : Prop := b + d = 6
def condition3 : Prop := b * d = 5

-- The theorem to prove
theorem find_a_plus_c (h1 : condition1) (h2 : condition2) (h3 : condition3) : a + c = 7 :=
by sorry

end find_a_plus_c_l570_570817


namespace determine_a_l570_570894

theorem determine_a (a : ℝ) : 
  (∃ (r : ℕ), r = 2 ∧ (coeff (x^3) (expandBinom (ax - 1)^5) = 80)) → a = 2 :=
by
  sorry

end determine_a_l570_570894


namespace range_of_m_l570_570064

theorem range_of_m (m : ℝ) : 
  (∀ x, x^2 + 2 * x - m > 0 ↔ (x = 1 → x^2 + 2 * x - m ≤ 0) ∧ (x = 2 → x^2 + 2 * x - m > 0)) ↔ (3 ≤ m ∧ m < 8) := 
sorry

end range_of_m_l570_570064


namespace count_odd_digits_base4_157_l570_570389

def base_repr (n b : ℕ) : List ℕ :=
  if b ≤ 1 then [n] else (List.unfoldr (λ x, if x = 0 then none else some (x % b, x / b)) n).reverse

def count_odd_digits (l : List ℕ) : ℕ :=
  l.countp (λ d, d % 2 = 1)

theorem count_odd_digits_base4_157 : count_odd_digits (base_repr 157 4) = 3 := by
  sorry

end count_odd_digits_base4_157_l570_570389


namespace exists_1990_gon_with_conditions_l570_570219

/-- A polygon structure with side lengths and properties to check equality of interior angles and side lengths -/
structure Polygon (n : ℕ) :=
  (sides : Fin n → ℕ)
  (angles_equal : Prop)

/-- Given conditions -/
def condition_1 (P : Polygon 1990) : Prop := P.angles_equal
def condition_2 (P : Polygon 1990) : Prop :=
  ∃ (σ : Fin 1990 → Fin 1990), ∀ i, P.sides i = (σ i + 1)^2

/-- The main theorem to be proven -/
theorem exists_1990_gon_with_conditions :
  ∃ P : Polygon 1990, condition_1 P ∧ condition_2 P :=
sorry

end exists_1990_gon_with_conditions_l570_570219


namespace smallest_N_composite_l570_570677

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

theorem smallest_N_composite : ∃ N : ℕ, (N > 0) ∧ is_composite (2 * N - 1) ∧ is_composite (2 * N + 1) ∧ (∀ M : ℕ, (M > 0) ∧ is_composite (2 * M - 1) ∧ is_composite (2 * M + 1) → N ≤ M) :=
by {
  use 13,
  split,
  {
    exact Nat.succ_pos' 12,
  },
  {
    split,
    {
      -- Proof that 2 * 13 - 1 is composite
      use [5, 5],
      split, norm_num,
      split, norm_num,
      norm_num,
    },
    {
      split,
      {
        -- Proof that 2 * 13 + 1 is composite
        use [5, 5],
        split, norm_num,
        split, norm_num,
        norm_num,
      },
      {
        intro M,
        rintro ⟨M_pos, M_composite1, M_composite2⟩,
        sorry, -- Here is where we would provide the argument that 13 is the smallest such N
      }
    }
  }
}

end smallest_N_composite_l570_570677


namespace triangle_problem_l570_570842

/-- 
  Given the perimeter of triangle abc is 10, and sin B + sin C = 4 * sin A,
  and the product of side lengths bc is 16, we aim to prove the values of
  a being 2 and cos A being 7/8.
--/
theorem triangle_problem
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : a + b + c = 10)
  (h2 : sin B + sin C = 4 * sin A)
  (h3 : b * c = 16)
  (ha : b + c = 4 * a)
  : a = 2 ∧ cos A = 7 / 8 :=
by 
  sorry

end triangle_problem_l570_570842


namespace xiao_zhang_payment_l570_570139

open Real

def purchase_amount (x : ℝ) : ℝ :=
  if x < 100 then x
  else if x ≤ 500 then 0.9 * x
  else 0.8 * (x - 500) + 0.9 * 500

def discount_total (purchases : List ℝ) : ℝ :=
  purchases.map purchase_amount |>.sum

theorem xiao_zhang_payment (pays : List ℝ) (total_price : ℝ) :
  (pays = [99, 530] ∧ (total_price = 609.2 ∨ total_price = 618)) 
  → discount_total [99, 530] = total_price :=
by
  intros
  sorry

#eval xiao_zhang_payment [99, 530] 609.2  -- to test the theorem.

end xiao_zhang_payment_l570_570139


namespace sum_of_roots_of_fx_eq_0_l570_570199

def f : ℝ → ℝ :=
| x if x < 3  => 5 * x + 10
| x          => 3 * x - 9

theorem sum_of_roots_of_fx_eq_0 : (∑ x in {x : ℝ | f x = 0}, x) = 1 := by 
  sorry

end sum_of_roots_of_fx_eq_0_l570_570199


namespace max_trailing_zeros_of_sum_l570_570208

theorem max_trailing_zeros_of_sum (numbers : List ℕ) (h₁ : numbers.length = 9)
    (h₂ : ∀ n ∈ numbers, n < 10^9)
    (h₃ : ∀ n ∈ numbers, (∑ d in (digits n), d) = 45) :
    (10 ^ 8 ∣ ∑ n in numbers, n) :=
sorry

end max_trailing_zeros_of_sum_l570_570208


namespace Djibo_sister_age_l570_570024

variable (d s : ℕ)
variable (h1 : d = 17)
variable (h2 : d - 5 + (s - 5) = 35)

theorem Djibo_sister_age : s = 28 :=
by sorry

end Djibo_sister_age_l570_570024


namespace line_parabola_unique_intersection_l570_570476

noncomputable def line := (k : ℝ) (x : ℝ) => k * x + 1
noncomputable def parabola := (y : ℝ) => y^2 / 4

theorem line_parabola_unique_intersection (k : ℝ) (h : ∃ x y : ℝ, y = line k x ∧ y = parabola x ∧
  ∀ x₁ y₁ : ℝ, y₁ = line k x₁ ∧ y₁ = parabola x₁ → x₁ = x ∧ y₁ = y) : 
  k = 0 ∨ k = 1 :=
by
  sorry

end line_parabola_unique_intersection_l570_570476


namespace fixed_point_l570_570018

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (2 * x - 1) / Real.log a + 1

theorem fixed_point (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : f 1 a = 1 := by
  unfold f
  have log_identity : Real.log 1 = 0 := by
    exact Real.log_one
  rw [mul_one two, sub_self one]
  rw [log_identity, zero_div]
  rw [add_zero]
  exact rfl

end fixed_point_l570_570018


namespace milan_total_bill_correct_l570_570805

-- Define the monthly fee, the per minute rate, and the number of minutes used last month
def monthly_fee : ℝ := 2
def per_minute_rate : ℝ := 0.12
def minutes_used : ℕ := 178

-- Define the total bill calculation
def total_bill : ℝ := minutes_used * per_minute_rate + monthly_fee

-- The proof statement
theorem milan_total_bill_correct :
  total_bill = 23.36 := 
by
  sorry

end milan_total_bill_correct_l570_570805


namespace geometric_sequence_sum_l570_570099

noncomputable def geometric_sequence (α : Type*) [field α] (a : ℕ → α) (r : α) : Prop :=
∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum 
  {α : Type*} [linear_ordered_field α]
  (a : ℕ → α) (r : α) 
  (hgeo : geometric_sequence α a r)
  (hincr : ∀ n, a n < a (n + 1))
  (ha3 : a 2 = 6)
  (hsum : a 0 + a 2 + a 4 = 21) :
  a 4 + a 6 + a 8 = 84 :=
sorry

end geometric_sequence_sum_l570_570099


namespace average_rate_of_interest_is_correct_l570_570346

-- Define conditions
variables (x : ℝ) (total_amount : ℝ := 6000) (rate3 : ℝ := 0.03) (rate5 : ℝ := 0.05)
variables (return_amount3 return_amount5 : ℝ)

-- Given statements
def amount_invested_at_3_percent := total_amount - x
def return_at_3_percent := rate3 * amount_invested_at_3_percent
def return_at_5_percent := rate5 * x

-- Equation based on equal annual returns
def equal_annual_returns := return_at_3_percent = return_at_5_percent

-- Correct average rate of interest
def average_rate_of_interest := 3.75 / 100

-- Proof Problem Statement
theorem average_rate_of_interest_is_correct 
  (h : equal_annual_returns) : 
  (rate3 * amount_invested_at_3_percent + rate5 * x) / total_amount = average_rate_of_interest := 
sorry

end average_rate_of_interest_is_correct_l570_570346


namespace question1_question2_l570_570472

namespace ProofProblem

open Real

theorem question1 (t : ℝ) (H1 : ∀ (x : ℝ), 2^(2*x) + t ≥ 2^x) : t ≥ 1/4 := sorry

theorem question2 (m n : ℝ) (a : ℝ) (H2 : (∀ x ∈ Icc m n, log a (a^(2*x) + t) ∈ Icc m n)
    (H3 : 0 < a) (H4 : a ≠ 1)) : 0 < t ∧ t < 1/4 := sorry

end ProofProblem

end question1_question2_l570_570472


namespace triangle_area_projections_l570_570549

section
variables {A B C : ℝ × ℝ × ℝ}

def triangle_area (A B C : ℝ × ℝ × ℝ) : ℝ :=
  let ⟨x1, y1, z1⟩ := B - A
  let ⟨x2, y2, z2⟩ := C - A
  0.5 * Real.sqrt ((x1*y2 - x2*y1)^2 + (y1*z2 - y2*z1)^2 + (z1*x2 - z2*x1)^2)

def projection_area (B C : ℝ × ℝ × ℝ) (f : ℝ × ℝ × ℝ → ℝ × ℝ) : ℝ :=
  let (u, w) := (f B, f C)
  0.5 * Real.abs (u.1 * w.2 - w.1 * u.2)

def S (A B C : ℝ × ℝ × ℝ) : ℝ := triangle_area A B C
def S_xy (A B C : ℝ × ℝ × ℝ) : ℝ := projection_area B C (λ ⟨x, y, z⟩ => (x, y))
def S_yz (A B C : ℝ × ℝ × ℝ) : ℝ := projection_area B C (λ ⟨x, y, z⟩ => (y, z))
def S_xz (A B C : ℝ × ℝ × ℝ) : ℝ := projection_area B C (λ ⟨x, y, z⟩ => (x, z))

theorem triangle_area_projections (A B C : ℝ × ℝ × ℝ) : 
  S A B C ^ 2 = S_xy A B C ^ 2 + S_yz A B C ^ 2 + S_xz A B C ^ 2 :=
sorry
end

end triangle_area_projections_l570_570549


namespace fourth_power_ends_in_six_l570_570130

theorem fourth_power_ends_in_six (n : ℕ) (h1 : n % 10 = 2 ∨ n % 10 = 4 ∨ n % 10 = 6 ∨ n % 10 = 8) :
  (n ^ 4) % 10 = 6 := 
begin 
  sorry 
end

end fourth_power_ends_in_six_l570_570130


namespace blue_balls_count_l570_570513

theorem blue_balls_count (Y B : ℕ) (h_ratio : 4 * B = 3 * Y) (h_total : Y + B = 35) : B = 15 :=
sorry

end blue_balls_count_l570_570513


namespace mikes_salary_l570_570802

theorem mikes_salary
  (fred_salary : ℝ)
  (mike_salary_increase_percent : ℝ)
  (mike_salary_factor : ℝ)
  (fred_salary_val : fred_salary = 1000)
  (mike_salary_factor_val : mike_salary_factor = 10)
  (mike_salary_increase_val : mike_salary_increase_percent = 40)
  : (10000 * (1 + mike_salary_increase_percent / 100)) = 14000 := 
by
  rw [fred_salary_val, mike_salary_factor_val, mike_salary_increase_val]
  norm_num
  sorry  -- Proof omitted

end mikes_salary_l570_570802


namespace cosine_identity_15_30_l570_570675

theorem cosine_identity_15_30 :
  ∀ (cos : ℝ) (thirty_sq : cos = 2 * cos^2 (15 * real.pi / 180) - 1), 
  2 * cos^2 (15 * real.pi / 180) - cos = 1 := 
by 
  intros cos thirty_sq
  sorry

end cosine_identity_15_30_l570_570675


namespace correct_answer_percentage_l570_570147

theorem correct_answer_percentage (T : ℕ) (C : ℕ) (hT : T = 84) (hC : C = 58) :
  (C.toRat / T.toRat * 100).round = 69 := by
  sorry

end correct_answer_percentage_l570_570147


namespace isosceles_right_triangle_properties_l570_570027

-- Define the basic structure and properties
theorem isosceles_right_triangle_properties (A B C B1 C1 : ℝ) 
  (is_right_triangle : ∠ BAC = 90°)
  (AB_eq_AC : AB = AC)
  (perpendicular_BB1 : ∠ ABB1 = 90°)
  (perpendicular_CC1 : ∠ ACC1 = 90°)
  : (AB1 ^ 2 + AC1 ^ 2 = BB1 ^ 2 + CC1 ^ 2) := 
sorry

end isosceles_right_triangle_properties_l570_570027


namespace reception_hall_tiling_l570_570333

theorem reception_hall_tiling :
  let rhombus_area := (1 / 2) * 10 * 8
  let tile_area := (1 / 2) * 1 * (5 / 4)
  rhombus_area / tile_area = 64 :=
by
  let rhombus_area := (1 / 2) * 10 * 8
  let tile_area := (1 / 2) * 1 * (5 / 4)
  calc
    rhombus_area / tile_area
      = 40 / (5 / 8) : by rw [show rhombus_area = 40, by norm_num, show tile_area = (5 / 8), by norm_num]
      ... = 40 * (8 / 5) : by rw div_eq_mul_inv
      ... = 40 * 1.6 : by norm_num
      ... = 64 : by norm_num

end reception_hall_tiling_l570_570333


namespace correct_assertions_l570_570072

variables {A B : Type} (f : A → B)

-- 1. Different elements in set A can have the same image in set B
def statement_1 : Prop := ∃ a1 a2 : A, a1 ≠ a2 ∧ f a1 = f a2

-- 2. A single element in set A can have different images in B
def statement_2 : Prop := ∃ a1 : A, ∃ b1 b2 : B, b1 ≠ b2 ∧ (f a1 = b1 ∧ f a1 = b2)

-- 3. There can be elements in set B that do not have a pre-image in A
def statement_3 : Prop := ∃ b : B, ∀ a : A, f a ≠ b

-- Correct answer is statements 1 and 3 are true, statement 2 is false
theorem correct_assertions : statement_1 f ∧ ¬statement_2 f ∧ statement_3 f := sorry

end correct_assertions_l570_570072


namespace cab_speed_fraction_l570_570680

theorem cab_speed_fraction (S R : ℝ) (h1 : S * 40 = R * 48) : (R / S) = (5 / 6) :=
sorry

end cab_speed_fraction_l570_570680


namespace angle_equality_l570_570160

noncomputable def triangle := Type
variables {A B C A' B' C' P Q: triangle}
variables [angle_bisectors : (AA' B' C')]

-- Defining intersection points
variables [intersection_P : (P = (A'B' ∩ CC'))]
variables [intersection_Q : (Q = (A'C_1 ∩ BB'))]

-- Main theorem
theorem angle_equality : ∠PAC = ∠QAB :=
begin
  apply sorry
end

end angle_equality_l570_570160


namespace S_2013_eq_2013_div_2014_l570_570862

-- Given function f(x) = x^2 - ax
def f (x : ℝ) (a : ℝ) := x^2 - a * x

-- Tangent line at the point A(1, f(1)) is perpendicular to the line x + 3y = 0
-- Prove that S_2013 = 2013 / 2014

theorem S_2013_eq_2013_div_2014 (a : ℝ) (S : ℕ → ℝ)
  (h1: ∀ x, f x a = x^2 - a * x)
  (h2: ∀ k, f (1 : ℝ) a = k → (k - (2 - a)) * (-1 / 3) = -1)
  (h3: ∀ n, S n = ∑ k in range (n+1), 1 / f k a)
  : S 2013 = 2013 / 2014 :=
sorry

end S_2013_eq_2013_div_2014_l570_570862


namespace minimum_rectangles_needed_l570_570921

structure Cell :=
  (row : ℕ)
  (col : ℕ)
  (label : Char)

def grid : List Cell := [
  ⟨0, 0, 'a'⟩, ⟨0, 1, 'a'⟩, ⟨0, 2, 'a'⟩, ⟨0, 3, 'd'⟩,
  ⟨1, 0, 'b'⟩, ⟨1, 1, 'c'⟩, ⟨1, 2, 'c'⟩, ⟨1, 3, 'd'⟩,
  ⟨2, 0, 'b'⟩, ⟨2, 1, 'e'⟩, ⟨2, 2, 'f'⟩, ⟨2, 3, 'g'⟩,
  ⟨3, 0, 'b'⟩, ⟨3, 1, 'e'⟩, ⟨3, 2, 'f'⟩, ⟨3, 3, 'g'⟩
]

def isValidRectangle (rect : List Cell) : Bool :=
  rect ≠ [] ∧
  let rows := rect.map Cell.row
  let cols := rect.map Cell.col
  rows.sortAndDedup = List.range (List.maximum rows - List.minimum rows + 1) ∧
  cols.sortAndDedup = List.range (List.maximum cols - List.minimum cols + 1)

def partitionIntoRectangles (cells : List Cell) (rectangles : List (List Cell)) : Prop :=
  (cells = rectangles.join) ∧
  rectangles.all isValidRectangle

def differentLabels (c1 c2 : Cell) : Prop := c1.label ≠ c2.label

theorem minimum_rectangles_needed : ∀ (rs : List (List Cell)),
  partitionIntoRectangles grid rs → rs.length ≥ 7 :=
sorry

end minimum_rectangles_needed_l570_570921


namespace birth_day_of_figure_l570_570233

theorem birth_day_of_figure :
  ∃ (birth_day : String), (birth_day = "Saturday") ∧
  let day_of_week (date_day : Int) (days_shift : Int) : String :=
    match (date_day - days_shift % 7 + 7) % 7 with
    | 0 => "Sunday"
    | 1 => "Monday"
    | 2 => "Tuesday"
    | 3 => "Wednesday"
    | 4 => "Thursday"
    | 5 => "Friday"
    | _ => "Saturday"
  in day_of_week 3 310 = "Saturday" :=
begin
  sorry
end

end birth_day_of_figure_l570_570233


namespace roberto_current_salary_l570_570983

theorem roberto_current_salary (starting_salary current_salary : ℝ) (h₀ : starting_salary = 80000)
(h₁ : current_salary = (starting_salary * 1.4) * 1.2) : 
current_salary = 134400 := by
  sorry

end roberto_current_salary_l570_570983


namespace smallest_pos_value_correct_l570_570780

noncomputable def smallest_pos_real_number : ℝ :=
  let x := 131 / 11 in
  if x > 0 ∧ (x * x).floor - x * (x.floor) = 10 then x else 0

theorem smallest_pos_value_correct (x : ℝ) (hx : 0 < x ∧ (x * x).floor - x * x.floor = 10) :
  x = 131 / 11 :=
begin
  sorry
end

end smallest_pos_value_correct_l570_570780


namespace stock_market_value_l570_570311

def face_value : ℝ := 100
def dividend_rate : ℝ := 0.05
def yield_rate : ℝ := 0.10

theorem stock_market_value :
  (dividend_rate * face_value / yield_rate = 50) :=
by
  sorry

end stock_market_value_l570_570311


namespace zero_point_interval_l570_570508

theorem zero_point_interval (k : ℤ) (x0 : ℝ) (h1 : ∀ (x : ℝ), f x = Real.log (x + 1) + x - 3)
  (h2 : x0 ∈ set.Ioo (k : ℝ) (k + 1)) (h3 : f 2 < 0) (h4 : f 3 > 0) : 
  k = 2 := sorry

end zero_point_interval_l570_570508


namespace question1_question2_l570_570451

section complex_numbers

variable {m : ℝ}
variable {z1 : ℂ := ⟨m * (m - 1), m - 1⟩}
variable {z2 : ℂ := ⟨m + 1, m^2 - 1⟩}

-- Question 1: 
theorem question1 (h1 : z1.re = 0) (h2 : z1.im ≠ 0) : m = 0 :=
  by sorry

-- Question 2: 
theorem question2 (h3 : z2.re > 0) (h4 : z2.im < 0) : -1 < m ∧ m < 1 :=
  by sorry

end complex_numbers

end question1_question2_l570_570451


namespace investment_difference_l570_570964

noncomputable theory

def mary_final_amount (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r)^t

def tom_final_amount (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n)^(n * t)

def difference (P : ℝ) (r : ℝ) (t : ℝ) (n : ℝ) : ℝ :=
  tom_final_amount P r n t - mary_final_amount P r t

theorem investment_difference :
  difference 75000 0.05 3 12 ≈ 214 :=
by
  sorry

end investment_difference_l570_570964


namespace intersection_point_on_line_l570_570980

theorem intersection_point_on_line 
  {A P R T S Q B C X : Point} 
  (h1 : angle A P T = π / 2) 
  (h2 : angle A R T = π / 2) 
  (h3 : angle A S T = π / 2) 
  (h4 : angle A Q T = π / 2) 
  (h5 : ∃ (circ : Circle), A ∈ circ ∧ P ∈ circ ∧ R ∈ circ ∧ T ∈ circ ∧ S ∈ circ ∧ Q ∈ circ ∧ diameter circ = A T)
  (h6 : ∃ (linePR : Line), P ∈ linePR ∧ R ∈ linePR ∧ X ∈ linePR)
  (h7 : ∃ (lineQS : Line), Q ∈ lineQS ∧ S ∈ lineQS ∧ X ∈ lineQS)
  (h8 : ∃ (lineBC : Line), B ∈ lineBC ∧ C ∈ lineBC) : 
  X ∈ lineBC := 
sorry

end intersection_point_on_line_l570_570980


namespace tangent_parallel_b_interval_l570_570874

theorem tangent_parallel_b_interval (b : ℝ) (h_b : 0 < b) (H : b ≠ 0)
  (point_A : (0, 1)) (curve_C : ∀ x, curve_C x = Real.exp x)
  (tangent_through_origin : ∃ x₀, Real.exp(x₀) * x₀ = Real.exp(x₀)) 
  (line_AB_parallel : ∃ k, k = Real.exp(1) ∧ ∀ x, curve_C x = k * x) :
  1 < b ∧ b < 2 :=
begin
  sorry
end

end tangent_parallel_b_interval_l570_570874


namespace num_integers_is_8_l570_570490

theorem num_integers_is_8 :
  let p (n : ℕ) := 3 ≤ n ∧ n ≤ 10 ∧ ∃ k : ℕ, (n^2 + 2*n + 1) = k^2 in
  finset.card {n ∈ finset.range 11 | p n} = 8 :=
by
  sorry

end num_integers_is_8_l570_570490


namespace number_of_5_letter_words_with_at_least_one_consonant_equals_7744_l570_570486

def is_consonant (c : Char) : Prop :=
  c = 'B' ∨ c = 'C' ∨ c = 'D' ∨ c = 'F'

def count_5_letter_words_with_at_least_one_consonant : Nat :=
  let total_words := 6 ^ 5
  let vowel_words := 2 ^ 5
  total_words - vowel_words

theorem number_of_5_letter_words_with_at_least_one_consonant_equals_7744 :
  count_5_letter_words_with_at_least_one_consonant = 7744 :=
by
  sorry

end number_of_5_letter_words_with_at_least_one_consonant_equals_7744_l570_570486


namespace inscribed_circle_radius_l570_570651

theorem inscribed_circle_radius (DE DF EF : ℝ) (hDE : DE = 26) (hDF : DF = 15) (hEF : EF = 17) : 
  let s := (DE + DF + EF) / 2,
      K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)),
      r := K / s
  in r = 6 * Real.sqrt 338 / 29 :=
by
  -- Definitions as per conditions
  have s_def : s = (DE + DF + EF) / 2 := rfl,
  have K_def : K = Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)) := rfl,
  have r_def : r = K / s := rfl,
  -- Specific values
  rw [hDE, hDF, hEF, s_def, K_def, r_def],
  sorry

end inscribed_circle_radius_l570_570651


namespace cyclist_wait_time_l570_570670

-- Define the hiker's speed, cyclist's speed, and waiting time.
def hiker_speed : ℝ := 7 -- miles per hour
def cyclist_speed : ℝ := 28 -- miles per hour
def waiting_time : ℝ := 5 / 60 -- hours (5 minutes converted to hours)

-- Minus Hours to Minutes
theorem cyclist_wait_time : ∀ (hiker_speed cyclist_speed waiting_time : ℝ), 
  hiker_speed = 7 → 
  cyclist_speed = 28 → 
  waiting_time = 5 / 60 → 
  (cyclist_speed * waiting_time / 1 - hiker_speed * waiting_time / 1) /  hiker_speed * 60 = 15 := by
  intros
  sorry

end cyclist_wait_time_l570_570670


namespace triangle_is_isosceles_l570_570909

def Triangle (α : Type) := α -> α -> α -> Prop

theorem triangle_is_isosceles {α : Type} [linear_ordered_field α]
  (a b c : α) (A B C : α) 
  (ha : A = 30) 
  (hb : a = 9) 
  (hp : a + b + c = 36)
  (h30_opp : a = 9) :
  ∃ b c, Triangle α a b c ∧ b = c :=
sorry

end triangle_is_isosceles_l570_570909


namespace car_R_speed_l570_570671

variable (v : ℝ)

-- Distance traveled by both cars
constant d : ℝ := 900

-- Speed relationship
constant speed_p : ℝ := v + 10

-- Time difference condition
constant speed_condition : (900 / v) - 2 = (900 / (v + 10))

theorem car_R_speed :
  v = 62.25 :=
by
  sorry

end car_R_speed_l570_570671


namespace min_value_expr_l570_570198

theorem min_value_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ m : ℝ, m = sqrt ((1 + real.cbrt 2 ^ 2) * (4 + real.cbrt 2 ^ 2) / (real.cbrt 2 ^ 2)) ∧
  ∀ a b : ℝ, 0 < a → 0 < b → 
    sqrt ((a^2 + b^2) * (4 * a^2 + b^2)) / (a * b) ≥ m :=
sorry

end min_value_expr_l570_570198


namespace selection_count_l570_570583

-- Define the set of people and pavilions
inductive People
| A | B | C | D | E | F

inductive Pavilion
| USA | UK | France | Saudi

open People Pavilion

def valid_combination (selection : List (People × Pavilion)) : Bool :=
  (selection.length = 4) ∧
  (selection.map Prod.snd).nodup ∧ 
  (selection.map Prod.fst).nodup ∧
  (∀ p, (p = People.A ∨ p = People.B) → ((p, France) ∉ selection))

noncomputable def count_valid_selections : Nat :=
  -- Total number of ways to permute 4 out of 6 people
  let total := (Nat.factorial 6) / (Nat.factorial 2) in

  -- Number of invalid ways where A or B are at France
  let invalid_A := (Nat.factorial 5) / (Nat.factorial 2) in
  let invalid_B := invalid_A in
   
  total - invalid_A - invalid_B

-- Statement for the proof problem
theorem selection_count : count_valid_selections = 240 := by
  sorry

end selection_count_l570_570583


namespace part_a_part_b_l570_570413

/-- Part (a) -/
theorem part_a (x : ℝ) (n : ℕ) (hx : -1 ≤ x ∧ x ≤ 1) :
  (∃ f : ℝ → ℝ, f = (λ x, (1 / (2^n)) * ((x + sqrt(1 - x^2))^n + (x - sqrt(1 - x^2))^n)) ∧ 
  polynomial.monic (polynomial.C (1 / (2^n)) * 
  (polynomial.C (((x + sqrt(1 - x^2)) : ℝ)^n) + polynomial.C ((x - sqrt(1 - x^2)) : ℝ)^n))) 
  ∧ (∀ (x : ℝ) (n : ℕ), abs ((1 / (2^(n - 1))) * cos (n * arccos x)) ≤ 1 / (2^(n - 1))) := sorry

/-- Part (b) -/
theorem part_b (p : polynomial ℝ) (n : ℕ) (hn : p.monic ∧ p.degree = n ∧ ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → p.eval x > -1 / (2^(n - 1))) :
  ∃ (x0 : ℝ), -1 ≤ x0 ∧ x0 ≤ 1 ∧ p.eval x0 ≥ 1 / (2^(n - 1)) := sorry

end part_a_part_b_l570_570413


namespace remaining_batches_l570_570425

def flour_per_batch : ℕ := 2
def batches_baked : ℕ := 3
def initial_flour : ℕ := 20

theorem remaining_batches : (initial_flour - flour_per_batch * batches_baked) / flour_per_batch = 7 := by
  sorry

end remaining_batches_l570_570425


namespace sum_of_real_solutions_eq_seven_halves_l570_570615

theorem sum_of_real_solutions_eq_seven_halves :
  ∀ x : ℝ, (2 ^ x - 4) ^ 3 + (4 ^ x - 2) ^ 3 = (4 ^ x + 2 ^ x - 6) ^ 3 → 
  ∃ S : ℝ, S = 7 / 2 := sorry

end sum_of_real_solutions_eq_seven_halves_l570_570615


namespace find_b_c_l570_570433

theorem find_b_c (a b c d : ℝ) 
  (h1 : a * b + a * c + b * d + c * d = 40) 
  (h2 : a + d = 6)
  (h3 : a * b + b * c + c * d + d * a = 28) : 
  b + c = 17 / 3 := 
by
  sorry

end find_b_c_l570_570433


namespace num_adult_tickets_is_35_l570_570684

noncomputable def num_adult_tickets_sold (A C: ℕ): Prop :=
  A + C = 85 ∧ 5 * A + 2 * C = 275

theorem num_adult_tickets_is_35: ∃ A C: ℕ, num_adult_tickets_sold A C ∧ A = 35 :=
by
  -- Definitions based on the provided conditions
  sorry

end num_adult_tickets_is_35_l570_570684


namespace problem1_f_zero_problem2_omega_value_problem3_f_geq_one_solution_set_l570_570855

noncomputable def f (ω x : ℝ) : ℝ :=
  2 * Real.cos (π / 2 - ω * x) + 2 * Real.sin (π / 3 - ω * x)

theorem problem1_f_zero (ω : ℝ) (h₀ : 0 < ω) : f ω 0 = Real.sqrt 3 :=
sorry

theorem problem2_omega_value (ω : ℝ) (h₀ : 0 < ω) 
  (h1 : f ω (π / 6) + f ω (π / 2) = 0) 
  (h2 : ∀ x ∈ Ioo (π / 6) (π / 2), f ω x ≤ f ω x) : ω = 2 :=
sorry

theorem problem3_f_geq_one_solution_set (ω x : ℝ) 
  (h₀ : ω = 2) : (f ω x ≥ 1) ↔ (∃ k : ℤ, (π / 12 + k * π ≤ x ∧ x ≤ π / 4 + k * π)) :=
sorry

end problem1_f_zero_problem2_omega_value_problem3_f_geq_one_solution_set_l570_570855


namespace piecewise_function_solution_l570_570468

theorem piecewise_function_solution (m : ℝ) :
  (m = Real.sqrt 10 ∨ m = -1) ↔
  (if m > 0 then log m = 1 / 2 else 2 ^ m = 1 / 2) :=
by
  sorry

end piecewise_function_solution_l570_570468


namespace max_value_y_l570_570193

theorem max_value_y (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) :
  let y := abs (x + 1) - 2 * abs x + abs (x - 2) in
  y ≤ 3 :=
by
  sorry

end max_value_y_l570_570193


namespace proper_subsets_count_S_l570_570247

/-- 
Define the set S consisting of natural numbers that satisfy the inequality 
-1 ≤ log_(1/4) 10 < -1/2.
-/
def S : Set ℕ := {x ∈ Set.univ | -1 ≤ Real.log 10 / Real.log (1/4) ∧ Real.log 10 / Real.log (1/4) < -1/2}

/-- 
The number of proper subsets of S is 127.
-/
theorem proper_subsets_count_S : ∃ S : Set ℕ, S ≠ ∅ ∧ (card (S.powerset) - 1) = 127 := 
sorry

end proper_subsets_count_S_l570_570247


namespace smallest_solution_l570_570779

def smallest_positive_real_x : ℝ :=
  (131 : ℝ) / 11

theorem smallest_solution (x : ℝ) (hx : 0 < x) (H : ⌊x^2⌋ - x * ⌊x⌋ = 10) : x = smallest_positive_real_x :=
  sorry

end smallest_solution_l570_570779


namespace magnitude_of_OP_eq_circle_eq_l570_570526

namespace Oblique

open Real

def e₁ : ℝ × ℝ := (1, 0)
def e₂ : ℝ × ℝ := (0, 1)
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
noncomputable def norm (v : ℝ × ℝ) : ℝ := √(dot_product v v)
def cos_angle := 1 / 2 -- cos(60°)
def OP (x y : ℝ) : ℝ × ℝ := (x, 0) + (0, y)

theorem magnitude_of_OP_eq (x y : ℝ) :
  norm (OP 3 (-2)) = √7 :=
by
  -- proof omitted
  sorry

theorem circle_eq (x y : ℝ) :
  x^2 + y^2 + x * y = 4 ↔ norm (OP x y) = 2 :=
by
  -- proof omitted
  sorry

end Oblique

end magnitude_of_OP_eq_circle_eq_l570_570526


namespace find_number_l570_570412

theorem find_number
  (n : ℕ)
  (h : 80641 * n = 806006795) :
  n = 9995 :=
by 
  sorry

end find_number_l570_570412


namespace monotonic_decreasing_interval_tan_l570_570608

theorem monotonic_decreasing_interval_tan : 
  ∀ (k : ℤ), 
    monotonic_decreasing_on 
      (Ioo (k * π - π / 6) (k * π + 5 * π / 6)) 
      (λ x, tan (π / 3 - x)) :=
by
  sorry

end monotonic_decreasing_interval_tan_l570_570608


namespace number_of_odd_digits_in_base4_rep_of_157_l570_570395

def count_odd_digits_in_base4 (n : ℕ) : ℕ :=
  (nat.digits 4 n).countp (λ d, d % 2 = 1)

theorem number_of_odd_digits_in_base4_rep_of_157 : count_odd_digits_in_base4 157 = 2 :=
by
  sorry

end number_of_odd_digits_in_base4_rep_of_157_l570_570395


namespace maximum_sum_value_l570_570551

theorem maximum_sum_value :
  (∃ (a : Fin 2000 → ℝ), (∀ i, 0 ≤ a i ∧ a i ≤ 1) ∧
    ∑ i in Finset.range 1999, ∑ j in Finset.rangeFrom (i + 1) 2000, (j - i) * |a j - a i| = 10^9) :=
sorry

end maximum_sum_value_l570_570551


namespace monotonicity_f_range_of_a_l570_570471

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1 / 2) * x^2 - a * x + Real.log x

theorem monotonicity_f (a : ℝ) : 
  (∀ x > 0, f(x, a) is increasing) ↔ (a ≤ 2) ∧ 
  (a > 2 → (∀ x ∈ (0, (a - Real.sqrt (a^2 - 4)) / 2), f(x, a) is increasing) ∧ 
               (∀ x ∈ ((a + Real.sqrt (a^2 - 4)) / 2, +∞), f(x, a) is increasing) ∧
               (∀ x ∈ ((a - Real.sqrt (a^2 - 4)) / 2, (a + Real.sqrt (a^2 - 4)) / 2), f(x, a) is decreasing)) :=
 sorry

theorem range_of_a (f_has_two_extreme_points: ∃ x1 x2 > 0, f(x1, a) is local extremum ∧ f(x2, a) is local extremum)
  (extreme_condition: |f(x1, a) - f(x2, a)| ≥ (3 / 4) - Real.log 2) : 
  a ≥ (3 * Real.sqrt 2) / 2 :=
 sorry

end monotonicity_f_range_of_a_l570_570471


namespace find_constants_and_prove_f_inequality_l570_570860

noncomputable def f (x : ℝ) : ℝ := exp x - x^2 + 1

theorem find_constants_and_prove_f_inequality :
  (∃ a b : ℝ, (∀ x : ℝ, f x = exp x - a * x^2 + 1) ∧ (∀ x : ℝ, tangent_line f 1 x = b * x + 2) ∧ a = 1 ∧ b = exp 1 - 2) ∧
  (∀ x > 0, f x ≥ (exp 1 - 2) * x + 2) :=
by
  sorry

end find_constants_and_prove_f_inequality_l570_570860


namespace leak_empties_tank_in_24_hours_l570_570669

theorem leak_empties_tank_in_24_hours (A L : ℝ) (hA : A = 1 / 8) (h_comb : A - L = 1 / 12) : 1 / L = 24 :=
by
  -- Proof will be here
  sorry

end leak_empties_tank_in_24_hours_l570_570669


namespace negation_of_one_even_is_all_odd_or_at_least_two_even_l570_570115

-- Definitions based on the problem conditions
def is_even (n : ℕ) : Prop := n % 2 = 0

def exactly_one_even (a b c : ℕ) : Prop :=
  (is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
  (¬ is_even a ∧ is_even b ∧ ¬ is_even c) ∨
  (¬ is_even a ∧ ¬ is_even b ∧ is_even c)

def all_odd (a b c : ℕ) : Prop :=
  ¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c

def at_least_two_even (a b c : ℕ) : Prop :=
  (is_even a ∧ is_even b) ∨
  (is_even a ∧ is_even c) ∨
  (is_even b ∧ is_even c)

-- The proposition to prove
theorem negation_of_one_even_is_all_odd_or_at_least_two_even (a b c : ℕ) :
  ¬ exactly_one_even a b c ↔ all_odd a b c ∨ at_least_two_even a b c :=
by sorry

end negation_of_one_even_is_all_odd_or_at_least_two_even_l570_570115


namespace complex_in_first_quadrant_l570_570097

theorem complex_in_first_quadrant (z : ℂ) (h : (1 + 2 * complex.I) / (z - 3) = - complex.I) : 
  z.re > 0 ∧ z.im > 0 := 
sorry

end complex_in_first_quadrant_l570_570097


namespace remaining_batches_l570_570426

def flour_per_batch : ℕ := 2
def batches_baked : ℕ := 3
def initial_flour : ℕ := 20

theorem remaining_batches : (initial_flour - flour_per_batch * batches_baked) / flour_per_batch = 7 := by
  sorry

end remaining_batches_l570_570426


namespace log_base_3_729_l570_570754

theorem log_base_3_729 : log 3 729 = 6 := by
  have h : 3^6 = 729 := sorry -- We acknowledge this based on the given condition.
  sorry -- Proof needed here.

end log_base_3_729_l570_570754


namespace max_MP_PQ_l570_570603

noncomputable def max_ratio_MP_PQ(A B C D A1 B1 C1 D1 M E F P Q : ℝ) : ℝ :=
  if A = (0,0,0) ∧ B = (1,0,0) ∧ C = (1,1,0) ∧ D = (0,1,0) ∧ 
     A1 = (0,0,1) ∧ B1 = (1,0,1) ∧ C1 = (1,1,1) ∧ D1 = (0,1,1) ∧
     |D - M| = 2 * sqrt(2/5) ∧
     E = (0.5, 0.5, 1) ∧
     F = (0, 1, 0.5) ∧
     P = (1/5, 1/5, 0) ∧
     Q = (0, 1/5, 0.5)

  then sqrt(2)
  else 0

theorem max_MP_PQ : ∀ (A B C D A1 B1 C1 D1 M E F P Q : ℝ), 
  (A = (0,0,0) ∧ B = (1,0,0) ∧ C = (1,1,0) ∧ D = (0,1,0) ∧ 
   A1 = (0,0,1) ∧ B1 = (1,0,1) ∧ C1 = (1,1,1) ∧ D1 = (0,1,1) ∧
   |D - M| = 2 * sqrt(2/5) ∧
   E = (0.5, 0.5, 1) ∧
   F = (0, 1, 0.5) ∧
   P = (1/5, 1/5, 0) ∧
   Q = (0, 1/5, 0.5))
  → max_ratio_MP_PQ A B C D A1 B1 C1 D1 M E F P Q = sqrt(2) :=
by sorry

end max_MP_PQ_l570_570603


namespace framed_painting_ratio_l570_570328

theorem framed_painting_ratio (w h : ℕ) (y : ℕ) 
  (hw : w = 15) (hh : h = 20) 
  (frame_wid_top_bottom : 3 * y) 
  (area_painting : w * h = 300)
  (area_frame : (w + 2 * y) * (h + 6 * y) = 2 * 300) :
  ((w + 2 * y) = 20) ∧ ((h + 6 * y) = 35) ∧ (20 / 35 = 4 / 7) :=
by
  sorry

end framed_painting_ratio_l570_570328


namespace cannot_tile_plateau1_can_tile_plateau2_cannot_tile_plateau3_l570_570363

section Tiling

-- Define the overall type to represent a board
structure Board (m n : ℕ) :=
  (occupied : Fin m → Fin n → Bool)

-- Define specific boards for each plateau based on their conditions

-- Plateau 1: 5x7 board with the condition that it's area is 35
def plateau1 : Board 5 7 :=
  ⟨λ i j => ! (false)⟩ -- Simplification for illustration

-- Plateau 2: (details from the illustration) can be generally assumed as tile-able
def plateau2 : Board _ _ :=
  -- Using underscore and assuming it's defined correctly
  ⟨λ i j => true⟩ -- Placeholder representing the existing illustration

-- Plateau 3: Checkerboard thinking, hence creating those conditions
def plateau3 : Board 8 5 :=
  ⟨λ i j => ! ((i.1 + j.1) % 2 == 0)⟩ -- Simplified checkerboard condition

-- Define the theorems to evaluable tiling conditions
theorem cannot_tile_plateau1 : ¬ (∃ (f : Fin 5 → Fin 7 → Fin 2), true) := 
  by
  sorry

theorem can_tile_plateau2 : ∃ (f : Fin _ → Fin _ → Fin 2), true :=
  by
  sorry

theorem cannot_tile_plateau3 : ¬ (∃ (f : Fin 8 → Fin 5 → Fin 2), true) :=
  by
  sorry

end Tiling

end cannot_tile_plateau1_can_tile_plateau2_cannot_tile_plateau3_l570_570363


namespace fourth_number_of_ninth_row_l570_570605

theorem fourth_number_of_ninth_row : 
  (λ n : ℕ, 7 * n - 3) 9 = 60 := 
by
  sorry

end fourth_number_of_ninth_row_l570_570605


namespace foldable_cube_configurations_l570_570822

noncomputable def can_fold_to_cube_with_one_face_missing (pos : Fin 8) : Bool := sorry

theorem foldable_cube_configurations :
  (Finset.filter can_fold_to_cube_with_one_face_missing (Finset.univ : Finset (Fin 8))).card = 4 := sorry

end foldable_cube_configurations_l570_570822


namespace simplify_expression_l570_570965

theorem simplify_expression (x : ℝ) (h : x < 1) :
  (x - 1) * (sqrt (-1 / (x - 1))) = -sqrt (1 - x) :=
by
  sorry

end simplify_expression_l570_570965


namespace triangular_prism_width_l570_570891

theorem triangular_prism_width (l h longest_edge : ℝ) (hl : l = 5) (hh : h = 13) (hle : longest_edge = 14) :
  ∃ w, sqrt (l^2 + w^2 + h^2) = longest_edge ∧ w = sqrt 2 :=
by {
  use sqrt 2,
  rw [hl, hh, hle],
  sorry
}

end triangular_prism_width_l570_570891


namespace ceil_of_fractional_square_l570_570753

theorem ceil_of_fractional_square :
  (Int.ceil ((- (7/4) + 1/4) ^ 2) = 3) :=
by
  sorry

end ceil_of_fractional_square_l570_570753


namespace number_of_5_letter_words_with_at_least_one_consonant_equals_7744_l570_570484

def is_consonant (c : Char) : Prop :=
  c = 'B' ∨ c = 'C' ∨ c = 'D' ∨ c = 'F'

def count_5_letter_words_with_at_least_one_consonant : Nat :=
  let total_words := 6 ^ 5
  let vowel_words := 2 ^ 5
  total_words - vowel_words

theorem number_of_5_letter_words_with_at_least_one_consonant_equals_7744 :
  count_5_letter_words_with_at_least_one_consonant = 7744 :=
by
  sorry

end number_of_5_letter_words_with_at_least_one_consonant_equals_7744_l570_570484


namespace product_gamma_l570_570940

noncomputable def gamma_params {α γ : ℝ} (xi : ℝ) (zeta : ℝ) :=
  xi ~ gamma_dist γ 1 ∧ zeta ~ beta_dist α (γ - α) ∧ 0 < α ∧ α < γ ∧ 0 < 1 ∧ indep xi zeta

lemma laplace_unique {α γ : ℝ} (xi zeta : ℝ) (h : gamma_params xi zeta) :
  laplace_transform (zeta * xi) = laplace_transform (gamma_dist γ 1) :=
sorry

theorem product_gamma {α γ λ: ℝ} (xi: ℝ) (zeta: ℝ) : 
  (xi ~ gamma_dist γ λ ∧ zeta ~ beta_dist α (γ - α) ∧ indep xi zeta ∧ 0 < α ∧ α < γ ∧ 0 < λ) 
  → (zeta * xi ∼ gamma_dist γ λ) :=
begin
  intros h,
  have h_gamma : gamma_params xi zeta,
  { split,
    exact h.1,
    split,
    exact h.2,
    split,
    exact h.3,
    exact h.4,
    exact h.5,
    exact h.6, },
  have h_laplace : laplace_transform (zeta * xi) = laplace_transform (gamma_dist γ 1),
  { apply laplace_unique,
    exact h_gamma, },
  admit
end

end product_gamma_l570_570940


namespace factor_expression_l570_570728

theorem factor_expression (x : ℝ) : 25 * x^2 + 10 * x = 5 * x * (5 * x + 2) :=
sorry

end factor_expression_l570_570728


namespace edges_sum_l570_570710

def edges_triangular_pyramid : ℕ := 6
def edges_triangular_prism : ℕ := 9

theorem edges_sum : edges_triangular_pyramid + edges_triangular_prism = 15 :=
by
  sorry

end edges_sum_l570_570710


namespace find_m_plus_n_l570_570955

noncomputable theory

def quadratic_has_two_distinct_real_solutions (b : ℝ) : Prop :=
  let y := (2 * b^2 - 8 * b) in
  y^2 - 4 * 16 * b^2 ≥ 0

def relevant_interval : Set ℝ := {b | b ∈ (-9 : ℝ) .. 9}

def probability_of_solution : ℝ := 
  let valid_b := {b | quadratic_has_two_distinct_real_solutions b ∧ b ∈ relevant_interval}
  (MeasureTheory.Measure.count valid_b / MeasureTheory.Measure.count relevant_interval)

theorem find_m_plus_n : ∃ (m n : ℕ), nat.gcd m n = 1 ∧ m / n = probability_of_solution ∧ m + n = 14 :=
sorry

end find_m_plus_n_l570_570955


namespace number_of_odd_digits_in_base_4_representation_l570_570400

-- Define the context and problem
def count_odd_digits_base_4 (n : ℕ) : ℕ :=
  let digits := (n.natDigits 4)
  in digits.count odd
  
theorem number_of_odd_digits_in_base_4_representation (n : ℕ) (h : n = 157) :
  count_odd_digits_base_4 n = 3 := by
  sorry

end number_of_odd_digits_in_base_4_representation_l570_570400


namespace fraction_subtraction_l570_570362

theorem fraction_subtraction : 
  (4 + 6 + 8 + 10) / (3 + 5 + 7) - (3 + 5 + 7 + 9) / (4 + 6 + 8) = 8 / 15 :=
  sorry

end fraction_subtraction_l570_570362


namespace avg_k_of_positive_integer_roots_l570_570848

theorem avg_k_of_positive_integer_roots : 
  (∑ (k : ℕ) in ({25, 14, 11, 10} : Finset ℕ), k) / 4 = 15 := 
by sorry

end avg_k_of_positive_integer_roots_l570_570848


namespace angle_between_clock_hands_8_30_l570_570234

theorem angle_between_clock_hands_8_30 : 
  let hour_position := 8 + 30 / 60,
      minute_position := 6, -- 30 minutes is equivalent to 6 on a clock (30 / 5)
      units_away := 2.5,
      angle_per_unit := 30 in
  75 = units_away * angle_per_unit :=
by 
  sorry

end angle_between_clock_hands_8_30_l570_570234


namespace circumradius_of_triangle_l570_570827

theorem circumradius_of_triangle :
  ∃ (R : ℝ), 
  (R ≈ 19.9 ∨ R ≈ 10.5) ∧
  (∀ (a b m_a : ℝ), a = 20 ∧ b = 17 ∧ m_a = 14 → 
    let t := 0.5 * a * m_a in
    let c_acute := Real.sqrt (a^2 + b^2 - 2 * a * Real.sqrt (b^2 - m_a^2)) in
    let c_obtuse := Real.sqrt (a^2 + b^2 + 2 * a * Real.sqrt (b^2 - m_a^2)) in
    let R_acute := (b * c_acute) / (2 * m_a) in
    let R_obtuse := (b * c_obtuse) / (2 * m_a) in
    (R_acute ≈ 10.5) ∨ (R_obtuse ≈ 19.9)) :=
sorry

end circumradius_of_triangle_l570_570827


namespace find_perimeter_C_l570_570154

-- Define the variables used in the problem
variables {d l h P_A P_B P_C : ℝ}

-- Given conditions for perimeters
axiom h_P_A : 4 * d + 2 * l + 4 * h = 400
axiom h_P_B : 2 * d + 2 * l + 2 * h = 240

-- Define the perimeter of building C
def P_C := 3 * d + l + 3 * h

-- The proof statement
theorem find_perimeter_C (hP: P_C = 280) : 
  4 * d + 2 * l + 4 * h = 400 → 2 * d + 2 * l + 2 * h = 240 → P_C = 280 :=
by
  sorry

end find_perimeter_C_l570_570154


namespace find_log_cos_as_a_l570_570125

variable {c : ℝ}
variable {x : ℝ}
variable {a : ℝ}

theorem find_log_cos_as_a
  (hc : 1 < c)
  (hsin : 0 < Real.sin x)
  (hcos : 0 < Real.cos x)
  (htan : Real.tan x = 1)
  (hloga : Real.log c (Real.sin x) = a) :
  Real.log c (Real.cos x) = a := by
  sorry

end find_log_cos_as_a_l570_570125


namespace common_tangent_passes_through_D_l570_570536

-- Definition of points and circles in the plane
variables (ℝ : Type) [field ℝ] [metric_space ℝ] [normed_space ℝ ℝ]

-- Let (d) and (d') be two parallel lines
variables (d d' : set (ℝ × ℝ))
axiom parallel_lines : ∀ (p1 q1 : ℝ × ℝ), p1 ∈ d ∧ q1 ∈ d' → parallel d d' 

-- Let Γ be the circle tangent to d and d' at points B and C, respectively
variables (Γ : metric.sphere ℝ)
variables (B C : (ℝ × ℝ))
axiom tangent_B : B ∈ Γ ∧ B ∈ d
axiom tangent_C : C ∈ Γ ∧ C ∈ d'

-- Let ω be a circle tangent to d and Γ at points A and Z, respectively
variables (ω : metric.sphere ℝ)
variables (A Z : (ℝ × ℝ))
axiom tangent_A : A ∈ ω ∧ A ∈ d
axiom tangent_Z : Z ∈ ω ∧ Z ∈ Γ

-- Let ω' be a circle tangent to Γ, ω, and d' at points X, Y, and D, respectively
variables (ω' : metric.sphere ℝ)
variables (X Y D : (ℝ × ℝ))
axiom tangent_X : X ∈ ω' ∧ X ∈ Γ
axiom tangent_Y : Y ∈ ω' ∧ Y ∈ ω
axiom tangent_D : D ∈ ω' ∧ D ∈ d'

-- Show that the common tangent to Γ and ω passes through D
theorem common_tangent_passes_through_D
  (tangent_Gamma_omega_at_T : ∃ T : (ℝ × ℝ), T ∈ Γ ∧ T ∈ ω)
  (common_tangent : ∀ (P : ((ℝ × ℝ))), (P ∈ Γ) ∧ (P ∈ ω) → P ∈ d)
  : ∃ D : (ℝ × ℝ), D ∈ d ∧ D ∈ d' ∧ (∃ T : (ℝ × ℝ), T ∈ Γ ∧ T ∈ ω) ∧ D ∈ ω' := 
sorry

end common_tangent_passes_through_D_l570_570536


namespace find_ordered_triple_l570_570952

theorem find_ordered_triple :
  ∃ (a b c : ℝ), a > 2 ∧ b > 2 ∧ c > 2 ∧
    (a + b + c = 30) ∧
    ( (a = 13) ∧ (b = 11) ∧ (c = 6) ) ∧
    ( ( ( (a + 3)^2 / (b + c - 3) ) + ( (b + 5)^2 / (c + a - 5) ) + ( (c + 7)^2 / (a + b - 7) ) = 45 ) ) :=
sorry

end find_ordered_triple_l570_570952


namespace find_set_B_l570_570954

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := x^2 + b * x + c

def A (f : ℝ → ℝ) : set ℝ := { x | f x = x }
def B (f : ℝ → ℝ) : set ℝ := { x | f (x - 1) = x + 1 }

theorem find_set_B (b c : ℝ) (h1 : f 2 b c = 2) (h2 : A (f b c) = { 2 }) :
  B (f b c) = { 3 + sqrt 5, 3 - sqrt 5 } :=
sorry

end find_set_B_l570_570954


namespace integer_points_in_intersection_of_spheres_l570_570010

theorem integer_points_in_intersection_of_spheres :
  {p : ℤ × ℤ × ℤ | 
     let x := p.1.1, y := p.1.2, z := p.2 in
     x^2 + y^2 + (z - 10)^2 ≤ 64 ∧ x^2 + y^2 + (z - 2)^2 ≤ 36
  }.to_finset.card = 42 :=
by
  sorry

end integer_points_in_intersection_of_spheres_l570_570010


namespace g_has_two_zeros_and_one_in_interval_l570_570063

def f (x : ℝ) : ℝ := Real.log x + x + 1

def g (x : ℝ) : ℝ := (f x)^2 - 2 * (f x) - 3

theorem g_has_two_zeros_and_one_in_interval :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ g x1 = 0 ∧ g x2 = 0) ∧ 
  (∃ x : ℝ, 1 < x ∧ x < 2 ∧ g x = 0) :=
sorry

end g_has_two_zeros_and_one_in_interval_l570_570063


namespace roots_sum_of_cubes_l570_570188

noncomputable def a := sorry
noncomputable def b := sorry
noncomputable def c := sorry

def is_root (r : ℝ) : Prop := r^3 - 2 * r^2 - r + 3 = 0

theorem roots_sum_of_cubes :
  is_root a ∧ is_root b ∧ is_root c →
  a + b + c = 2 →
  a * b + b * c + c * a = -1 →
  a * b * c = -3 →
  a^3 + b^3 + c^3 = 5 :=
by
  intros h_roots h_sum h_prod_sum h_prod
  sorry

end roots_sum_of_cubes_l570_570188


namespace solve_inequalities_l570_570763

theorem solve_inequalities :
  {x : ℝ | 4*x + 2 < (x - 1)^2 ∧ (x - 1)^2 < 9*x + 3} = 
  set.Ioo (3 + 2*real.sqrt 2) (5.5 - real.sqrt (32.25)) ∪ set.Ioo (5.5 + real.sqrt (32.25)) ∞ ∩ 
  set.Ioo (5.5 - real.sqrt 32.25) (5.5 + real.sqrt 32.25) :=
sorry

end solve_inequalities_l570_570763


namespace domain_of_function_l570_570237

theorem domain_of_function :
  ∀ x : ℝ, (x - 1 ≥ 0) ↔ (x ≥ 1) ∧ (x + 1 ≠ 0) :=
by
  sorry

end domain_of_function_l570_570237


namespace equivalent_proof_problem_l570_570417

def op (a b : ℝ) : ℝ := (a + b) ^ 2

theorem equivalent_proof_problem (x y : ℝ) : 
  op ((x + y) ^ 2) ((x - y) ^ 2) = 4 * (x ^ 2 + y ^ 2) ^ 2 := 
by 
  sorry

end equivalent_proof_problem_l570_570417


namespace test_tube_full_two_amoebas_l570_570313

def amoeba_population (n : ℕ) : ℕ := 2^n

theorem test_tube_full_two_amoebas :
  ∀ n : ℕ, (amoeba_population (n + 1) = amoeba_population 60) → n = 59 :=
begin
  sorry
end

end test_tube_full_two_amoebas_l570_570313


namespace liam_savings_per_month_l570_570557

theorem liam_savings_per_month (trip_cost bill_cost left_after_bills : ℕ) 
                               (months_in_two_years : ℕ) (total_savings_per_month : ℕ) :
  trip_cost = 7000 →
  bill_cost = 3500 →
  left_after_bills = 8500 →
  months_in_two_years = 24 →
  total_savings_per_month = 19000 →
  total_savings_per_month / months_in_two_years = 79167 / 100 :=
by
  intros
  sorry

end liam_savings_per_month_l570_570557


namespace area_of_shaded_quadrilateral_l570_570734

theorem area_of_shaded_quadrilateral : 
  let a := 3
      b := 5
      c := 7
      total_base := a + b + c
      height_ratio := c / total_base
      height1 := a * height_ratio
      height2 := (a + b) * height_ratio
      trapezoid_height := b
  in 1 / 2 * trapezoid_height * (height1 + height2) = 77 / 6 := by 
  sorry

end area_of_shaded_quadrilateral_l570_570734


namespace max_value_y_l570_570194

theorem max_value_y (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) :
  let y := abs (x + 1) - 2 * abs x + abs (x - 2) in
  y ≤ 3 :=
by
  sorry

end max_value_y_l570_570194


namespace tangent_circles_equilateral_triangle_area_l570_570726

noncomputable def areaOfTriangle (Q1 Q2 Q3 : ℝ) : ℝ := 
  let side_length := Q1.distance Q2 -- Here, we assume the distance function is predefined
  (√3 / 4) * side_length ^ 2

theorem tangent_circles_equilateral_triangle_area :
  ∀ (ω1 ω2 ω3 : circle) (Q1 Q2 Q3 : point in ω1 ω2 ω3), 
    radius ω1 = 3 → radius ω2 = 3 → radius ω3 = 3 →
    externally_tangent ω1 ω2 → externally_tangent ω1 ω3 → externally_tangent ω2 ω3 →
    Q1 ≠ Q2 → Q2 ≠ Q3 → Q3 ≠ Q1 →
    Q1Q2 = Q2Q3 ∧ Q2Q3 = Q3Q1 →
    (∀ i, tangent (Q i) (ω i)) →
    areaOfTriangle Q1 Q2 Q3 = 1944 :=
by
  sorry

end tangent_circles_equilateral_triangle_area_l570_570726


namespace x_plus_inv_x_eq_seven_l570_570093

theorem x_plus_inv_x_eq_seven (x : ℝ) (h : x^(1/2) + x^(-1/2) = 3) : x + x⁻¹ = 7 :=
by {
  sorry
}

end x_plus_inv_x_eq_seven_l570_570093


namespace min_value_l570_570086

open Real

-- Definitions
variables (a b : ℝ)
axiom a_gt_zero : a > 0
axiom b_gt_one : b > 1
axiom sum_eq : a + b = 3 / 2

-- The theorem to be proved.
theorem min_value (a : ℝ) (b : ℝ) (a_gt_zero : a > 0) (b_gt_one : b > 1) (sum_eq : a + b = 3 / 2) :
  ∃ (m : ℝ), m = 6 + 4 * sqrt 2 ∧ ∀ (x y : ℝ), (x > 0) → (y > 1) → (x + y = 3 / 2) → (∃ (z : ℝ), z = 2 / x + 1 / (y - 1) ∧ z ≥ m) :=
sorry

end min_value_l570_570086


namespace inscribed_circle_radius_l570_570650

theorem inscribed_circle_radius (DE DF EF : ℝ) (hDE : DE = 26) (hDF : DF = 15) (hEF : EF = 17) : 
  let s := (DE + DF + EF) / 2,
      K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)),
      r := K / s
  in r = 6 * Real.sqrt 338 / 29 :=
by
  -- Definitions as per conditions
  have s_def : s = (DE + DF + EF) / 2 := rfl,
  have K_def : K = Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)) := rfl,
  have r_def : r = K / s := rfl,
  -- Specific values
  rw [hDE, hDF, hEF, s_def, K_def, r_def],
  sorry

end inscribed_circle_radius_l570_570650


namespace product_of_solutions_t_squared_eq_49_l570_570767

theorem product_of_solutions_t_squared_eq_49 (t : ℝ) (h1 : t^2 = 49) : (7 * -7) = -49 :=
by
  sorry

end product_of_solutions_t_squared_eq_49_l570_570767


namespace sin_cos_eq_cos_sin_unique_solution_l570_570741

theorem sin_cos_eq_cos_sin_unique_solution :
  ∃! (x : ℝ), 0 ≤ x ∧ x ≤ π / 2 ∧ sin (cos x) = cos (sin x) :=
sorry

end sin_cos_eq_cos_sin_unique_solution_l570_570741


namespace parabola_line_intersection_angles_l570_570717

theorem parabola_line_intersection_angles :
  let f (x : ℝ) := x ^ 2 / 2
  let g (x : ℝ) := (3 * x - 2) / 2
  let x1 := 1
  let y1 := f x1
  let x2 := 2
  let y2 := f x2
  let k_AF := 1
  let k_BN := 2
  let k_AB := 3 / 2
  tan⁻¹ (abs ((k_AB - k_AF) / (1 + k_AB * k_AF))) = tan⁻¹ (1 / 5) ∧
  tan⁻¹ (abs ((k_AB - k_BN) / (1 + k_AB * k_BN))) = tan⁻¹ (1 / 8) := 
  sorry

end parabola_line_intersection_angles_l570_570717


namespace time_to_cross_l570_570278

/-- Define the lengths of trains and their speeds. -/
def TrainA_length := 250 -- in meters
def TrainA_speed := 70 * 1000 / 3600 -- converting from km/hr to m/s

def TrainB_length := 350 -- in meters
def TrainB_speed := 90 * 1000 / 3600 -- converting from km/hr to m/s

/-- Calculate the relative speed and the total length -/
def relative_speed := TrainA_speed + TrainB_speed -- m/s
def total_length := TrainA_length + TrainB_length -- meters

/-- Theorem stating the time taken for both trains to fully cross each other -/
theorem time_to_cross : total_length / relative_speed = 13.5 := by
  sorry

end time_to_cross_l570_570278


namespace no_solution_inequalities_l570_570136

theorem no_solution_inequalities (a : ℝ) : 
  (∀ x : ℝ, ¬ (x > 3 ∧ x < a)) ↔ (a ≤ 3) :=
by
  sorry

end no_solution_inequalities_l570_570136


namespace polynomial_solution_l570_570552

noncomputable def f (n : ℕ) (X Y : ℝ) : ℝ :=
  (X - 2 * Y) * (X + Y) ^ (n - 1)

theorem polynomial_solution (n : ℕ) (f : ℝ → ℝ → ℝ)
  (h1 : ∀ (t x y : ℝ), f (t * x) (t * y) = t^n * f x y)
  (h2 : ∀ (a b c : ℝ), f (a + b) c + f (b + c) a + f (c + a) b = 0)
  (h3 : f 1 0 = 1) :
  ∀ (X Y : ℝ), f X Y = (X - 2 * Y) * (X + Y) ^ (n - 1) :=
by
  sorry

end polynomial_solution_l570_570552


namespace sequence_eq_n_l570_570373

theorem sequence_eq_n {a : ℕ → ℕ} (h : ∀ n : ℕ, n > 0 → (n^2 + 1) * a (n + 1) - a n = n^3 + n^2 + 1) :
  ∀ n : ℕ, n > 0 → a n = n :=
begin
  sorry
end

end sequence_eq_n_l570_570373


namespace jenny_phone_bill_l570_570315

theorem jenny_phone_bill : 
  let base_cost := 25 in
  let text_cost := 150 * 0.08 in
  let extra_minutes := (41 - 40) * 60 in
  let extra_minutes_cost := extra_minutes * 0.12 in
  let total_cost := base_cost + text_cost + extra_minutes_cost in 
  total_cost = 44.20 := 
by
  sorry

end jenny_phone_bill_l570_570315


namespace find_smallest_x_satisfying_condition_l570_570794

theorem find_smallest_x_satisfying_condition :
  ∃ x : ℝ, 0 < x ∧ (⌊x^2⌋ - x * ⌊x⌋ = 10) ∧ x = 131 / 11 :=
by
  sorry

end find_smallest_x_satisfying_condition_l570_570794


namespace abigail_fence_building_l570_570708

theorem abigail_fence_building :
  ∀ (initial_fences : Nat) (time_per_fence : Nat) (hours_building : Nat) (minutes_per_hour : Nat),
    initial_fences = 10 →
    time_per_fence = 30 →
    hours_building = 8 →
    minutes_per_hour = 60 →
    initial_fences + (minutes_per_hour / time_per_fence) * hours_building = 26 :=
by
  intros initial_fences time_per_fence hours_building minutes_per_hour
  sorry

end abigail_fence_building_l570_570708


namespace parabola_focus_l570_570043

-- Definition of a parabola in given equation form
def parabola (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 1

-- Prove the focus of the given parabola is (1, 23/8)
theorem parabola_focus : focus parabola = (1, 23 / 8) :=
by
  sorry

end parabola_focus_l570_570043


namespace value_of_f_sin_20_l570_570818

theorem value_of_f_sin_20 (f : ℝ → ℝ) (h : ∀ x, f (Real.cos x) = Real.sin (3 * x)) :
  f (Real.sin (20 * Real.pi / 180)) = -1 / 2 :=
by sorry

end value_of_f_sin_20_l570_570818


namespace actual_speed_of_valentin_l570_570595

theorem actual_speed_of_valentin
  (claimed_speed : ℕ := 50) -- Claimed speed in m/min
  (wrong_meter : ℕ := 60)   -- Valentin thought 1 meter = 60 cm
  (wrong_minute : ℕ := 100) -- Valentin thought 1 minute = 100 seconds
  (correct_speed : ℕ := 18) -- The actual speed in m/min
  : (claimed_speed * wrong_meter / wrong_minute) * 60 / 100 = correct_speed :=
by
  sorry

end actual_speed_of_valentin_l570_570595


namespace statements_correctness_l570_570070

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - x + a

theorem statements_correctness (a : ℝ) (h0 : f 0 a < 0) (h1 : f 1 a > 0) :
  (∃ x0, 0 < x0 ∧ x0 < 1 ∧ f x0 a = 0) ∧
  (f (1/2) a > 0 → f (1/4) a < 0) ∧
  (f (1/2) a < 0 → f (1/4) a > 0 → f (3/4) a = 0) ∧
  ¬ (f (3/2) a > 0 → f (5/4) a = 0) :=
by
  sorry

end statements_correctness_l570_570070


namespace angle_Z_eq_130_degrees_l570_570961

theorem angle_Z_eq_130_degrees (p q : Line) (h_parallel : Parallel p q)
  (X Y Z : Point) (h_angle_X : mangle X = 100) (h_angle_Y : mangle Y = 130) :
  mangle Z = 130 := 
sorry

end angle_Z_eq_130_degrees_l570_570961


namespace number_equation_form_l570_570033

variable (a : ℝ)

theorem number_equation_form :
  3 * a + 5 = 4 * a := 
sorry

end number_equation_form_l570_570033


namespace probability_of_double_domino_l570_570006

-- Define the range of integers
def dominoRange : List ℕ := List.range 12

-- Define the set of all possible pairings (including doubles)
def dominoPairs : List (ℕ × ℕ) := 
  (dominoRange.product dominoRange).filter (λ ⟨x, y⟩, x ≤ y)

-- Define the set of all doubles
def doubleDominos : List (ℕ × ℕ) := 
  dominoRange.map (λ x, (x, x))

-- Prove the probability of selecting a double domino from the set of all dominos
theorem probability_of_double_domino : 
  (doubleDominos.length : ℚ) / (dominoPairs.length : ℚ) = 2 / 13 := by
  sorry

end probability_of_double_domino_l570_570006


namespace sin_2alpha_eq_2_over_3_l570_570065

theorem sin_2alpha_eq_2_over_3
  (α : ℝ)
  (h : sin α - cos α = (Real.sqrt 3) / 3) :
  sin (2 * α) = 2 / 3 :=
by
  sorry

end sin_2alpha_eq_2_over_3_l570_570065


namespace angle_between_vectors_l570_570453

-- Define the points A, B, and C
def A : ℝ × ℝ × ℝ := (2, -5, 1)
def B : ℝ × ℝ × ℝ := (2, -2, 4)
def C : ℝ × ℝ × ℝ := (1, -4, 1)

-- Define the vectors AB and AC
def vector_sub (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p1.1 - p2.1, p1.2 - p2.2, p1.3 - p2.3)

def AB := vector_sub B A
def AC := vector_sub C A

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Define the magnitude of a vector
def vector_magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

-- Add a notation for cosθ calculation
def cos_theta (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / (vector_magnitude v1 * vector_magnitude v2)

theorem angle_between_vectors :
  let θ := Real.arccos (cos_theta AB AC) in
  θ = Real.pi / 3 :=
by
  sorry

end angle_between_vectors_l570_570453


namespace max_product_distance_axis_l570_570985

open Real

-- Definitions of the parametric equations of curve C
def curve_x (θ : ℝ) : ℝ := cos θ
def curve_y (θ : ℝ) : ℝ := sin θ

-- Definitions of the scaling transformation
def scaled_x (θ : ℝ) : ℝ := 3 * cos θ
def scaled_y (θ : ℝ) : ℝ := 2 * sin θ

-- The mathematical statement for the proof
theorem max_product_distance_axis :
  ∃ θ : ℝ, |(3 * cos θ) * (2 * sin θ)| = 3 :=
sorry

end max_product_distance_axis_l570_570985


namespace min_value_F_l570_570540

def is_permutation (σ : List ℕ) (n : ℕ) : Prop :=
  σ.perm (List.range n).map (λ x, x + 1)

def F (σ : List ℕ) : ℕ :=
  (List.zipWith (*) σ (σ.rotate 1)).sum

def min_F (n : ℕ) : ℕ :=
  if n % 2 = 0 then (n^3 + 3 * n^2 + 5 * n - 6) / 6 else (n^3 + 3 * n^2 + 5 * n - 3) / 6

theorem min_value_F (σ : List ℕ) (h : is_permutation σ σ.length) :
  F σ ≥ min_F σ.length :=
sorry

end min_value_F_l570_570540


namespace vector_sum_l570_570876

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (1, 2, 3)
def b : ℝ × ℝ × ℝ := (-1, 0, 1)

-- Define the target vector c
def c : ℝ × ℝ × ℝ := (-1, 2, 5)

-- State the theorem to be proven
theorem vector_sum : a + (2:ℝ) • b = c :=
by 
  -- Not providing the proof, just adding a sorry
  sorry

end vector_sum_l570_570876


namespace line_through_points_C_D_has_undefined_slope_and_angle_90_l570_570048

theorem line_through_points_C_D_has_undefined_slope_and_angle_90 (m : ℝ) (n : ℝ) (hn : n ≠ 0) :
  ∃ θ : ℝ, (∀ (slope : ℝ), false) ∧ θ = 90 :=
by { sorry }

end line_through_points_C_D_has_undefined_slope_and_angle_90_l570_570048


namespace find_f_l570_570087

variables (a b c : ℝ)
noncomputable def f (x : ℝ) := a * x ^ 2 + b * x + c
noncomputable def g (x : ℝ) := a * x + b

theorem find_f (h : ∀ x ∈ set.Icc (-1 : ℝ) 1, g x ≤ 2) :
  f = λ x, 2 * x ^ 2 - 1 :=
sorry

end find_f_l570_570087


namespace men_in_first_group_l570_570229

theorem men_in_first_group (M : ℕ) (h1 : M * 18 * 6 = 15 * 12 * 6) : M = 10 :=
by
  sorry

end men_in_first_group_l570_570229


namespace smallest_positive_real_is_131_div_11_l570_570769

noncomputable def smallest_positive_real_satisfying_condition :=
  ∀ (x : ℝ), (∀ y > 0, (y * y ⌊y⌋ - y ⌊y⌋ = 10) → (x ≤ y)) → 
  (⌊x*x⌋ - (x * ⌊x⌋) = 10) → 
  x = 131/11

theorem smallest_positive_real_is_131_div_11 :
  smallest_positive_real_satisfying_condition := sorry

end smallest_positive_real_is_131_div_11_l570_570769


namespace line_passes_fixed_point_l570_570448

noncomputable def ellipse_equation (C : Set (ℝ × ℝ)) (a b : ℝ) (P : ℝ × ℝ) (k1 k2 : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (a > b) ∧ ((P.1 = 1) ∧ (P.2 = sqrt 2 / 2)) ∧
  (C = {p : ℝ × ℝ | (p.fst^2 / a^2) + (p.snd^2 / b^2) = 1}) ∧
  (k1 + k2 = 2)

theorem line_passes_fixed_point (C : Set (ℝ × ℝ)) (a b : ℝ) (P : ℝ × ℝ) 
 (k1 k2 : ℝ) :
  ellipse_equation C a b P k1 k2 →
  ∃ A B : ℝ × ℝ, (A ≠ B) ∧ 
  (A ∈ C) ∧ (B ∈ C) ∧ 
  (∃ k m : ℝ, (∀ x : ℝ, x ∈ line_through (1 / a) k A B → x.2 = k * x.1 + m) ∧ (k1 + k2 = 2)) → 
  ∀ x ∈ (line_through A B), x = (-1, -1) :=
sorry

end line_passes_fixed_point_l570_570448


namespace smallest_AAB_l570_570344

theorem smallest_AAB : ∃ (A B : ℕ), (1 <= A ∧ A <= 9) ∧ (1 <= B ∧ B <= 9) ∧ (AB = 10 * A + B) ∧ (AAB = 100 * A + 10 * A + B) ∧ (110 * A + B = 8 * (10 * A + B)) ∧ (AAB = 221) :=
by
  sorry

end smallest_AAB_l570_570344


namespace num_integers_P_leq_zero_l570_570372

def P (x : ℝ) := (x - 1^3) * (x - 2^3) * (x - 3^3) * (x - 4^3) * (x - 5^3) * 
                 (x - 6^3) * (x - 7^3) * (x - 8^3) * (x - 9^3) * (x - 10^3) * 
                 (x - 11^3) * (x - 12^3) * (x - 13^3) * (x - 14^3) * (x - 15^3) * 
                 (x - 16^3) * (x - 17^3) * (x - 18^3) * (x - 19^3) * (x - 20^3)

-- The theorem that states the total number of integers n such that P(n) ≤ 0 is 5150
theorem num_integers_P_leq_zero : ∃ (n : ℕ), n = 5150 ∧ ∀ (x : ℤ), P(x) ≤ 0 → x ∈ set.Icc 1^3 (20^3) :=
by
  sorry

end num_integers_P_leq_zero_l570_570372


namespace super_scholar_is_d_l570_570142

-- Definitions
def valid_rank (rank : ℕ) : Prop := rank ≤ 3

structure ranking :=
  (Chinese : ℕ)
  (Math : ℕ)
  (English : ℕ)
  (Science : ℕ)
  (SocialStudies : ℕ)
  (valid     : valid_rank Chinese ∧ valid_rank Math ∧ valid_rank English ∧ valid_rank Science ∧ valid_rank SocialStudies)

-- Rankings described by students
def StudentA := ranking.mk 1 1 2 2 4 sorry
def StudentB := ranking.mk 1 2 2 2 5 sorry
def StudentC1 := ranking.mk 1 1 1 1 6 sorry
def StudentC2 := ranking.mk 1 1 1 2 5 sorry
def StudentD := ranking.mk 1 2 2 2 3 sorry

-- Proof problem
theorem super_scholar_is_d : 
  (valid_rank 1 ∧ valid_rank 2 ∧ valid_rank 2 ∧ valid_rank 2 ∧ valid_rank 3) 
  → StudentD.valid :=
by sorry

end super_scholar_is_d_l570_570142


namespace inequalities_correct_l570_570493

theorem inequalities_correct (a b : ℝ) (h1 : 1 / a > 1 / b) (h2 : 1 / b > 0) : 
  (a ^ 3 < b ^ 3) ∧ (sqrt b - sqrt a < sqrt (b - a)) :=
by
  -- Conditions
  have ha : a > 0 := by sorry
  have hb : b > a := by sorry
  -- Inequalities proof steps
  sorry

end inequalities_correct_l570_570493


namespace largest_is_sqrt6_l570_570273

noncomputable def largest_of_three_numbers (p q r : ℝ) : ℝ :=
  if p ≥ q ∧ p ≥ r then p
  else if q ≥ p ∧ q ≥ r then q
  else r

theorem largest_is_sqrt6 (p q r : ℝ) (h1 : p + q + r = 3) (h2 : p * q + p * r + q * r = -8) (h3 : p * q * r = -18) : 
  largest_of_three_numbers p q r = real.sqrt 6 :=
sorry

end largest_is_sqrt6_l570_570273


namespace sequence_general_term_l570_570075

theorem sequence_general_term 
  (x : ℕ → ℝ)
  (h1 : x 1 = 2)
  (h2 : x 2 = 3)
  (h3 : ∀ m ≥ 1, x (2*m+1) = x (2*m) + x (2*m-1))
  (h4 : ∀ m ≥ 2, x (2*m) = x (2*m-1) + 2*x (2*m-2)) :
  ∀ m, (x (2*m-1) = ((3 - Real.sqrt 2) / 4) * (2 + Real.sqrt 2) ^ m + ((3 + Real.sqrt 2) / 4) * (2 - Real.sqrt 2) ^ m ∧ 
          x (2*m) = ((1 + 2 * Real.sqrt 2) / 4) * (2 + Real.sqrt 2) ^ m + ((1 - 2 * Real.sqrt 2) / 4) * (2 - Real.sqrt 2) ^ m) :=
sorry

end sequence_general_term_l570_570075


namespace last_two_digits_factorials_l570_570044

theorem last_two_digits_factorials :
  let s := (3! + 5! + ∑ n in (finset.filter ((≤) 10) (finset.range (101))), n!) in
  s % 100 = 26 :=
by
  let s := (3! + 5! + ∑ n in (finset.filter ((≤) 10) (finset.range (101))), n!)
  have h3 : 3! = 6 := rfl
  have h5 : 5! = 120 := rfl
  have h10 : ∀ n, 10 ≤ n → (n! % 100) = 0 := 
    by sorry -- to be proven separately
  have hsum : s = 6 + 120 := 
    by {
      rw [← finset.sum_filter, finset.sum_range_succ, finset.sum_range_succ, add_assoc],
      simp [h10, finset.sum_const_zero], sorry
    }
  rw hsum
  norm_num
  exact rfl

end last_two_digits_factorials_l570_570044


namespace problem1_problem2_problem3_l570_570440

-- Given function f(x) and its property of being an odd function
def f (a : ℝ) (x : ℝ) : ℝ := (a - 3^x) / (3^x + 1)
def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g (x)

-- Problem statements to prove
theorem problem1 (h_odd : is_odd_function (f a)) : a = 1 := sorry

theorem problem2 (h_odd : is_odd_function (f 1)) : ∀ x1 x2 : ℝ, x1 < x2 → f 1 x1 > f 1 x2 := sorry

theorem problem3 (h_odd : is_odd_function (f 1)) 
  (h_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f 1 x1 > f 1 x2) :
  ∀ k : ℝ, (∀ x ∈ Icc (-π/6) (π/3), f 1 (sin (2*x)) + f 1 (2 - k) < 0) → k < (2 - real.sqrt 3 / 2) := sorry

end problem1_problem2_problem3_l570_570440


namespace trapezoid_area_is_correct_l570_570907

noncomputable def trapezoid_area (base_short : ℝ) (angle_adj : ℝ) (angle_diag : ℝ) : ℝ :=
  let width := 2 * base_short -- calculated width from angle_adj
  let height := base_short / Real.tan (angle_adj / 2 * Real.pi / 180)
  (base_short + base_short + width) * height / 2

theorem trapezoid_area_is_correct :
  trapezoid_area 2 135 150 = 2 :=
by
  sorry

end trapezoid_area_is_correct_l570_570907


namespace MikeSalaryNow_l570_570801

-- Definitions based on conditions
def FredSalary  := 1000   -- Fred's salary five months ago
def MikeSalaryFiveMonthsAgo := 10 * FredSalary  -- Mike's salary five months ago
def SalaryIncreasePercent := 40 / 100  -- 40 percent salary increase
def SalaryIncrease := SalaryIncreasePercent * MikeSalaryFiveMonthsAgo  -- Increase in Mike's salary

-- Statement to be proved
theorem MikeSalaryNow : MikeSalaryFiveMonthsAgo + SalaryIncrease = 14000 :=
by
  -- Proof is skipped
  sorry

end MikeSalaryNow_l570_570801


namespace sum_of_bn_l570_570459

theorem sum_of_bn (n : ℕ) (a : ℕ → ℝ) (b : ℕ → ℝ) 
(han : ∀ n, a n = 3 * 2 ^ (n - 1)) 
(han_bn : ∀ n, a n + b n = n + 3) : 
∑ i in finset.range n, b i = (n * (n + 7) / 2 ) - 3 * 2^n + 3 :=
by sorry

end sum_of_bn_l570_570459


namespace k_pretty_sum_div18_l570_570739

noncomputable def is_k_pretty (n k : ℕ) : Prop :=
  (factors_count n = k) ∧ (n % k = 0) ∧ (n % 3 = 0)

def S (upper_bound : ℕ) (k : ℕ) : ℕ :=
  ∑ n in (Finset.range upper_bound).filter (λ n, is_k_pretty n k), n

theorem k_pretty_sum_div18 :
  S 1000 18 / 18 = 70 :=
by
  sorry

end k_pretty_sum_div18_l570_570739


namespace sum_of_first_9_terms_l570_570525

theorem sum_of_first_9_terms (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h1 : ∀ n m : ℕ, a (n + m) = a n * a m) -- geometric sequence a_n
  (h2 : 3 * a 5 - a 3 * a 7 = 0)
  (h3 : b 5 = a 5)
  (h4 : ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d) -- arithmetic sequence b_n
  : (∑ i in finset.range 9, b (i + 1)) = 27 := sorry

end sum_of_first_9_terms_l570_570525


namespace ana_wins_probability_l570_570169

theorem ana_wins_probability :
  let P := (1/2)^(5 : ℕ),
      q := P*(1/(1-P)) in
  q = (1 : ℚ) / 31 := by
sorry

end ana_wins_probability_l570_570169


namespace first_train_takes_4_hours_less_l570_570629

-- Definitions of conditions
def distance: ℝ := 425.80645161290323
def speed_first_train: ℝ := 75
def speed_second_train: ℝ := 44

-- Lean statement to prove the correct answer
theorem first_train_takes_4_hours_less:
  (distance / speed_second_train) - (distance / speed_first_train) = 4 := 
  by
    -- Skip the actual proof
    sorry

end first_train_takes_4_hours_less_l570_570629


namespace moles_of_water_from_reaction_l570_570766

def moles_of_water_formed (nh4cl_moles : ℕ) (naoh_moles : ℕ) : ℕ :=
  nh4cl_moles -- Because 1:1 ratio of reactants producing water

theorem moles_of_water_from_reaction :
  moles_of_water_formed 3 3 = 3 := by
  -- Use the condition of the 1:1 reaction ratio derivable from the problem's setup.
  sorry

end moles_of_water_from_reaction_l570_570766


namespace lower_bound_perimeter_of_inscribed_triangle_lower_bound_perimeter_of_inscribed_convex_polygon_l570_570227

open Real

variables {R : ℝ}

-- Definition of an inscribed triangle in a circle with radius R where the center lies within it
noncomputable def inscribed_triangle (T : Type) [is_triangle T] (C : Type) [is_circle C] (center_in_triangle : circle_center_in_triangle C T) : Prop := 
  ∃ R : ℝ, circle_radius C = R ∧ center lies_in triangle T

-- The main theorem statement for the lower bound of the perimeter of triangles inscribed in a circle
theorem lower_bound_perimeter_of_inscribed_triangle
  (T : Type) [is_triangle T] (C : Type) [is_circle C] (center_in_triangle : circle_center_in_triangle C T) 
  : inscribed_triangle T C center_in_triangle → Perimeter T >= 4 * R := 
sorry

-- Extend the theorem to convex polygons
theorem lower_bound_perimeter_of_inscribed_convex_polygon
  (P : Type) [is_convex_polygon P] (C : Type) [is_circle C] (center_in_polygon : circle_center_in_polygon C P)
  : inscribed_convex_polygon P C center_in_polygon → Perimeter P >= 4 * R :=
sorry

end lower_bound_perimeter_of_inscribed_triangle_lower_bound_perimeter_of_inscribed_convex_polygon_l570_570227


namespace registration_methods_count_l570_570624

theorem registration_methods_count :
  let students := 6 in
  let competitions := 3 in
  let choices_per_student := competitions in
  choices_per_student ^ students = 729 :=
by
  let students := 6
  let competitions := 3
  let choices_per_student := competitions
  show choices_per_student ^ students = 729
  sorry

end registration_methods_count_l570_570624


namespace distance_from_C_to_line_AB_equation_of_altitude_from_B_to_AC_l570_570520

-- Define the coordinates of points A, B, and C
def A : ℝ × ℝ := (0, 4)
def B : ℝ × ℝ := (-3, 0)
def C : ℝ × ℝ := (1, 1)

-- Define the line AB using the standard form
noncomputable def line_AB : ℝ × ℝ × ℝ := (4, -3, 12) -- 4x - 3y + 12 = 0

-- Define the altitude line from B to side AC
noncomputable def altitude_B_AC : ℝ × ℝ × ℝ := (3, 4, -7) -- 3x + 4y - 7 = 0

-- State the Lean proof problem
theorem distance_from_C_to_line_AB : distance_point_to_line C line_AB = 13 / 5 := by
  sorry

theorem equation_of_altitude_from_B_to_AC : altitude_B_AC = (3, 4, -7) := by
  sorry

end distance_from_C_to_line_AB_equation_of_altitude_from_B_to_AC_l570_570520


namespace max_gcd_expression_l570_570101

theorem max_gcd_expression (n : ℕ) (h1 : n > 0) (h2 : n % 3 = 1) : 
  Nat.gcd (15 * n + 5) (9 * n + 4) = 5 :=
by
  sorry

end max_gcd_expression_l570_570101


namespace max_squares_covered_l570_570683

theorem max_squares_covered (checkerboard : ℕ → ℕ → bool) (card : ℕ) : 
  (checkerboard = λ x y, true) →
  (card = 2) →
  ∃ n, n = 12 ∧ (∀ x y, checkerboard x y = true → (x < card ∧ y < card)) :=
by
  sorry

end max_squares_covered_l570_570683


namespace find_m_for_parallel_lines_l570_570131

theorem find_m_for_parallel_lines (m : ℝ) :
  (∀ x y, 2 * x + (m + 1) * y + 4 = 0 → mx + 3 * y - 2 = 0 → 
  -((2 : ℝ) / (m + 1)) = -(m / 3)) → (m = 2 ∨ m = -3) :=
by
  sorry

end find_m_for_parallel_lines_l570_570131


namespace at_least_2008_good_numbers_with_2007_digits_l570_570731

def is_good_number (a : ℕ → ℕ) (n : ℕ) (seq : List ℕ) : Prop :=
  ∃ r : ℕ, seq = List.ofFn (λ k, a (r + k - 1))

theorem at_least_2008_good_numbers_with_2007_digits
  (a : ℕ → ℕ)
  (h1 : ∀ i, a i ∈ {1, 2})
  (h2 : ∃ seq_list : List (List ℕ), (∀ seq, seq ∈ seq_list → seq.length = 1_000_000 ∧ is_good_number a 1_000_000 seq) ∧ seq_list.length ≥ 2008) : 
  ∃ seq_list_short : List (List ℕ), (∀ seq, seq ∈ seq_list_short → seq.length = 2007 ∧ is_good_number a 2007 seq) ∧ seq_list_short.length ≥ 2008 :=
sorry

end at_least_2008_good_numbers_with_2007_digits_l570_570731


namespace sum_of_areas_l570_570254

theorem sum_of_areas :
  (∑' n : ℕ, Real.pi * (1 / 9 ^ n)) = (9 * Real.pi) / 8 :=
by
  sorry

end sum_of_areas_l570_570254


namespace cyclic_B_D_F_G_l570_570153

section

variables {A B C P Q D E F G : Type*}
variables [geometry A B C] [geometry A P Q] [geometry Q D C]
variables [eqangles (P AB) (B CA)] [intercirc (circ APB) (side (AC) Q)]
variables [eqangles (Q DC) (C AP)] [eqdist (CD) (CE)]
variables [intercirc (circ CQE) (seg CD F)] [intersect (line QF) (side BC G)]

theorem cyclic_B_D_F_G : is_acute_angle_triangle ABC → 
  ∠BAC > ∠BCA → 
  point_on_segment BC P → 
  ∠PAB = ∠BCA →
  second_intersection (circumcircle APB) AC Q → 
  point_on_segment AP D → 
  ∠QDC = ∠CAP →
  point_on_line BD E → 
  CE = CD →
  second_intersection (circumcircle CQE) CD F  → 
  intersection_line_side QF BC G →
  cyclic B D F G := 
begin
  sorry,
end

end

end cyclic_B_D_F_G_l570_570153


namespace div_factorial_general_linear_group_l570_570177

theorem div_factorial_general_linear_group (n : ℕ) (p : ℕ) [fact (nat.prime p)] (h : 2 < p) :
  (p - 1)^n * n! ∣ (list.prod (list.map (λ k, p^n - p^k) (list.range n))) :=
sorry

end div_factorial_general_linear_group_l570_570177


namespace white_cell_never_black_l570_570529

variable {Cell : Type}

/-- Representing our cells as a generic type with conditions. 
  Assume initially_black is a finite subset of cells on the grid and white_certain is the white cell that must remain white --/
def initially_black : Set Cell := sorry
def white_cells (c : Cell) : Prop := sorry
def polygon_covering : Set Cell := sorry

theorem white_cell_never_black (c : Cell) 
    (hb : initially_black c)
    (hw : white_cells c)
    (hp : polygon_covering c) :
    ∃ c, white_cells c ∧ (∀ shifts, c ∉ polygon_covering) :=
sorry

end white_cell_never_black_l570_570529


namespace find_f_f_neg8_l570_570435

def f (x : ℝ) : ℝ := if x > -1 then x + 2 / x - 5 else -((-x)^(1/3 : ℝ))

theorem find_f_f_neg8 : f (f (-8)) = -2 := 
by sorry

end find_f_f_neg8_l570_570435


namespace find_f0_f1_l570_570466

def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g x

variables (f : ℝ → ℝ)
  (h₀ : is_odd_function (λ x, f x - 2))
  (h₁ : f (-1) = 1)

theorem find_f0_f1 : f 0 + f 1 = 5 :=
sorry

end find_f0_f1_l570_570466


namespace positive_integer_condition_l570_570421

theorem positive_integer_condition (x : ℝ) (hx : x ≠ 0) : 
  (∃ (n : ℤ), n > 0 ∧ (abs (x - abs x + 2) / x) = n) ↔ x = 2 :=
by
  sorry

end positive_integer_condition_l570_570421


namespace option_A_option_B_option_C_option_D_l570_570920

variables (A B C a b c : ℝ)
variables (sA sB sC : ℝ)
variables (triangle_ABC : A + B + C = π)

-- Definition of a triangle in Lean
-- Definition of $\triangle ABC$
def triangle (A B C a b c : ℝ) (triangle_ABC : A + B + C = π) : Prop := 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

-- Law of Sines
def law_of_sines (sA sB sC : ℝ) (a b c : ℝ) : Prop := 
  a / sA = b / sB ∧ b / sB = c / sC

-- Definition of obtuse-angled triangle
def obtuse_triangle (A B C : ℝ) : Prop := A > π / 2 ∨ B > π / 2 ∨ C > π / 2

-- A: If $sin A > sin B$, then $A > B$
theorem option_A (h1 : sin A > sin B) : A > B :=
sorry

-- B: If $\sin ^2 B + \sin ^2 C < \sin ^2 A$, then $\triangle ABC$ is obtuse-angled
theorem option_B (h2 : sin B ^ 2 + sin C ^ 2 < sin A ^ 2) : obtuse_triangle A B C :=
sorry

-- C: In acute-angled triangle $\triangle ABC$, the inequality $sin A > cos B$ always holds
theorem option_C (h3 : A < π / 2 ∧ B < π / 2 ∧ C < π / 2) : sin A > cos B :=
sorry

-- D: In $\triangle ABC$, if $a cos A = b cos B$, then $\triangle ABC$ is not necessarily isosceles
theorem option_D (h4 : a * cos A = b * cos B) : ¬(a = b ∧ A = B) :=
sorry

end option_A_option_B_option_C_option_D_l570_570920


namespace simplify_trig_expr_l570_570294

theorem simplify_trig_expr (α β : ℝ) : 
  cos α ^ 2 + cos β ^ 2 - 2 * cos α * cos β * cos (α - β) = sin (α - β) ^ 2 :=
by
  sorry

end simplify_trig_expr_l570_570294


namespace g_iter_1994_at_4_l570_570126

noncomputable def g (x : ℚ) : ℚ := (2 + x) / (2 - 4 * x)
noncomputable def g_iter : ℕ → ℚ → ℚ
| 0, x := x
| (n+1), x := g (g_iter n x)

theorem g_iter_1994_at_4 : g_iter 1994 4 = 87 / 50 := 
begin 
  sorry 
end

end g_iter_1994_at_4_l570_570126


namespace problem1_problem2_l570_570901

variable {A B C : ℝ}
variable {a b c : ℝ}
variable (h1 : cos B - 2 * cos A = (2 * a - b) * cos C / c)
variable (h2 : a = 2 * b)

theorem problem1 : a / b = 2 :=
by sorry

theorem problem2 (h3 : A > π / 2) (h4 : c = 3) : 0 < b ∧ b < 3 :=
by sorry

end problem1_problem2_l570_570901


namespace seq_geometric_general_formula_and_sum_l570_570077

-- Definition of the sequence based on initial conditions
def a : ℕ → ℕ
| 0 := 3
| (n+1) := 2 * (a n) + 1

-- Proof (1): The sequence {a_n + 1} is geometric
theorem seq_geometric :
  ∃ (a r : ℕ), a > 0 ∧ r > 0 ∧ (∀ n : ℕ, (a n) + 1 = (a * (r ^ n))) :=
sorry

-- Proof (2): Find the general formula for the sequence {a_n} and the sum S_n of the first n terms
theorem general_formula_and_sum (n: ℕ) :
  (a n = 2^(n+1) - 1) ∧ ((∑ i in range (n + 1), a i) = 2^(n + 2) - 4 - n) :=
sorry

end seq_geometric_general_formula_and_sum_l570_570077


namespace cos_angle_equiv_370_l570_570764

open Real

noncomputable def find_correct_n : ℕ :=
  sorry

theorem cos_angle_equiv_370 (n : ℕ) (h : 0 ≤ n ∧ n ≤ 180) : cos (n * π / 180) = cos (370 * π / 180) → n = 10 :=
by
  sorry

end cos_angle_equiv_370_l570_570764


namespace correct_option_c_l570_570356

variable (a b c : ℝ)

def quadratic_function (x : ℝ) : ℝ := a * x^2 + b * x + c

axiom symmetry_axis : -b / (2 * a) = 1

theorem correct_option_c (h : b = -2 * a) : c > 2 * b :=
 sorry

end correct_option_c_l570_570356


namespace Paco_initial_cookies_l570_570976

-- Define the initial number of cookies Paco had
def initial_cookies (C : ℕ) : Prop :=
  C - 7 = 0 -- C is determined to be 7 from conditions

theorem Paco_initial_cookies : ∃ (C : ℕ), initial_cookies C :=
by
  use 7
  unfold initial_cookies
  norm_num
  sorry

end Paco_initial_cookies_l570_570976


namespace minimum_value_g_l570_570475

-- Define the conditions and result in Lean

def f (x : ℝ) (φ : ℝ) := sin (2 * x + φ) + sqrt 3 * cos (2 * x + φ)

def g (x : ℝ) (φ : ℝ) := cos (x + φ)

theorem minimum_value_g :
  ∀ (φ : ℝ), 
    0 < φ ∧ φ < π →
    (∀ x : ℝ, 
      -π / 2 ≤ x ∧ x ≤ π / 6 →
      f (x - π / 4) φ = f (x) φ ↔ x + π / 2 = φ) →
    (∃ m : ℝ, 
      ∀ x : ℝ, 
        -π / 2 ≤ x ∧ x ≤ π / 6 →
        g x (π / 6) ≤ m) ∧ m = 1 / 2 :=
by
  sorry

end minimum_value_g_l570_570475


namespace angle_EFG_70_l570_570458

-- Define the angles and assumptions
variables {x : ℝ}
-- Given conditions
def parallel_AD_FG : Prop := true  -- AD parallel FG
def angle_CFG : ℝ := 2 * x
def angle_CEA : ℝ := 3 * x - 20
def angle_AEB : ℝ := x + 30

-- Lean statement to prove that angle EFG is 70
theorem angle_EFG_70 
  (h1 : parallel_AD_FG)
  (h2 : angle_CFG = 2 * x)
  (h3 : angle_CEA = 3 * x - 20)
  (h4 : angle_AEB = x + 30) :
  x = 40 ∧ (x + 30) = 70 := 
sorry

end angle_EFG_70_l570_570458


namespace integer_part_x2_arctan_sum_l570_570553

theorem integer_part_x2 (x1 x2 x3 : ℝ) (h1 : x1 + x2 + x3 = 0)
  (h2 : x1 * x2 + x2 * x3 + x3 * x1 = -17)
  (h3 : x1 * x2 * x3 = 18)
  (h4 : -4 < x1 ∧ x1 < -3)
  (h5 : 4 < x3 ∧ x3 < 5) : int.floor x2 = -2 := 
sorry

theorem arctan_sum (x1 x2 x3 : ℝ) (h1 : x1 + x2 + x3 = 0)
  (h2 : x1 * x2 + x2 * x3 + x3 * x1 = -17)
  (h3 : x1 * x2 * x3 = 18)
  (h4 : -4 < x1 ∧ x1 < -3)
  (h5 : 4 < x3 ∧ x3 < 5) : real.arctan x1 + real.arctan x2 + real.arctan x3 = -real.pi / 4 := 
sorry

end integer_part_x2_arctan_sum_l570_570553


namespace variance_of_dataSet_l570_570619

-- Define the given data set
def dataSet : List ℤ := [-2, -1, 0, 1, 2]

-- Define the function to calculate mean
def mean (data : List ℤ) : ℚ :=
  (data.sum : ℚ) / data.length

-- Define the function to calculate variance
def variance (data : List ℤ) : ℚ :=
  let μ := mean data
  (data.map (λ x => (x - μ) ^ 2)).sum / data.length

-- State the theorem: The variance of the given data set is 2
theorem variance_of_dataSet : variance dataSet = 2 := by
  sorry

end variance_of_dataSet_l570_570619


namespace triangle_PQR_QR_l570_570161

noncomputable def triangle_PQR (P Q : ℝ) : Prop :=
  cos (Q - 2 * P) + sin (P + Q) = 2 ∧
  (PQ : ℝ) = 6 ∧
  (QR : ℝ) = 2-- let's define QR just for skelton purpose, later change to match

theorem triangle_PQR_QR (P Q QR : ℝ) (PQ : ℝ) 
  (h1: cos (Q - 2 * P) + sin (P + Q) = 2)
  (h2: PQ = 6) :
  QR = 3 :=
sorry

end triangle_PQR_QR_l570_570161


namespace final_point_P_after_transformations_l570_570978

noncomputable def point := (ℝ × ℝ)

def rotate_90_clockwise (p : point) : point :=
  (-p.2, p.1)

def reflect_across_x (p : point) : point :=
  (p.1, -p.2)

def P : point := (3, -5)

def Q : point := (5, -2)

def R : point := (5, -5)

theorem final_point_P_after_transformations : reflect_across_x (rotate_90_clockwise P) = (-5, 3) :=
by 
  sorry

end final_point_P_after_transformations_l570_570978


namespace triangle_cotangent_sum_l570_570979

-- Definitions for sides, angles, and area of the triangle
variables {a b c : ℝ} -- sides of the triangle
variables {A B C : ℝ} -- angles opposite to sides a, b, c respectively
variable {S : ℝ} -- area of the triangle

-- Condition that a, b, c form a triangle and S is its area
axiom area_of_triangle : (S = 0.5 * a * b * sin C)

axiom law_of_cosines1 : (a^2 = b^2 + c^2 - 2 * b * c * cos A)
axiom law_of_cosines2 : (b^2 = a^2 + c^2 - 2 * a * c * cos B)
axiom law_of_cosines3 : (c^2 = a^2 + b^2 - 2 * a * b * cos C)

-- Final goal
theorem triangle_cotangent_sum :
  cot A + cot B + cot C = (a^2 + b^2 + c^2) / (4 * S) :=
sorry

end triangle_cotangent_sum_l570_570979


namespace sum_of_rational_roots_is_6_l570_570408

noncomputable def h : Polynomial ℚ := Polynomial.X^3 - 6 * Polynomial.X^2 + 11 * Polynomial.X - 6

theorem sum_of_rational_roots_is_6 : (h.roots.filter (λ r, r.is_rat)).sum = 6 := by
  sorry

end sum_of_rational_roots_is_6_l570_570408


namespace ae_length_l570_570736

theorem ae_length (AB CD AC AE : ℝ) (h: 2 * AE + 3 * AE = 34): 
  AE = 34 / 5 := by
  -- Proof steps will go here
  sorry

end ae_length_l570_570736


namespace average_k_of_polynomial_with_positive_integer_roots_l570_570846

-- Define the conditions and the final theorem

theorem average_k_of_polynomial_with_positive_integer_roots :
  (∑ i in {k | ∃ r1 r2 : ℕ+, r1 * r2 = 24 ∧ k = r1 + r2}.to_finset, i) / 
  ({k | ∃ r1 r2 : ℕ+, r1 * r2 = 24 ∧ k = r1 + r2}.to_finset.card : ℝ) = 15 :=
by
  sorry

end average_k_of_polynomial_with_positive_integer_roots_l570_570846


namespace measure_of_angle_B_l570_570900

-- Define the sides and angles in the triangle
variables {α β γ a b c : ℝ}

-- Assume the given condition b * cos(γ) + (2 * a + c) * cos(β) = 0
axiom given_condition (h : b * Real.cos γ + (2 * a + c) * Real.cos β = 0)

-- Prove that β = 2 * π / 3 given the condition
theorem measure_of_angle_B (h : b * Real.cos γ + (2 * a + c) * Real.cos β = 0) : β = (2 * Real.pi) / 3 :=
sorry

end measure_of_angle_B_l570_570900


namespace number_of_odd_digits_in_base_4_representation_l570_570403

-- Define the context and problem
def count_odd_digits_base_4 (n : ℕ) : ℕ :=
  let digits := (n.natDigits 4)
  in digits.count odd
  
theorem number_of_odd_digits_in_base_4_representation (n : ℕ) (h : n = 157) :
  count_odd_digits_base_4 n = 3 := by
  sorry

end number_of_odd_digits_in_base_4_representation_l570_570403


namespace larger_number_is_84_l570_570302

theorem larger_number_is_84
  (hcf lcm : ℕ)
  (hcf_condition : hcf = 84)
  (lcm_condition : lcm = 21)
  (ratio_condition : ∃ x : ℕ, (∀ a b : ℕ, a = x ∧ b = 4 * x)) :
  (λ (x : ℕ), (∀ a b : ℕ, (hcf * lcm = a * b) → (4 * x = b))) →
  (∀ a b : ℕ, (ratio_condition.some = 21) → (4 * ratio_condition.some = 84)) :=
sorry

end larger_number_is_84_l570_570302


namespace ana_wins_probability_l570_570171

theorem ana_wins_probability :
  let P := (1/2)^(5 : ℕ),
      q := P*(1/(1-P)) in
  q = (1 : ℚ) / 31 := by
sorry

end ana_wins_probability_l570_570171


namespace area_of_triangle_l570_570384

theorem area_of_triangle (a b c : ℝ) (h1: a = 10) (h2: b = 11) (h3: c = 11) :
  let s := (a + b + c) / 2 in
  let area := (s * (s - a) * (s - b) * (s - c)).sqrt in
  area = 20 * sqrt 6 :=
by
  sorry

end area_of_triangle_l570_570384


namespace ellipse_equation_center_origin_foci_xaxis_minor2_ecc_sqrt2_div_2_l570_570465

open Real

noncomputable def ellipse_equation (a b : ℝ) : ℝ × ℝ → ℝ :=
  λ (x y : ℝ), x^2 / a^2 + y^2 / b^2

theorem ellipse_equation_center_origin_foci_xaxis_minor2_ecc_sqrt2_div_2 :
  let b := 1
  let a := sqrt 2
  eccentricity := (sqrt (a^2 - b^2)) / a
  2 * b = 2 → 
  (ellipse_equation a b (x, y) = 1) :=
by
  assume b_eq : 2 * 1 = 2,
  assume ecc_eq : ((sqrt ((sqrt 2)^2 - 1^2)) / (sqrt 2)) = (sqrt 2 / 2),
  sorry

end ellipse_equation_center_origin_foci_xaxis_minor2_ecc_sqrt2_div_2_l570_570465


namespace cosine_inequality_theorem_l570_570239

noncomputable def cosine_inequality (A B C D : ℝ × ℝ × ℝ) : Prop :=
  let (xa, ya, za) := A in
  let (xb, yb, zb) := B in
  let (xc, yc, zc) := C in
  let (xd, yd, zd) := D in
  let dot_prod (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3 in
  let norm (u : ℝ × ℝ × ℝ) : ℝ := real.sqrt (u.1 * u.1 + u.2 * u.2 + u.3 * u.3) in
  let vec_ac := (xa - xc, ya - yc, za - zc) in
  let vec_bc := (xb - xc, yb - yc, zb - zc) in
  let vec_ad := (xa - xd, ya - yd, za - zd) in
  let vec_bd := (xb - xd, yb - yd, zb - zd) in
  let vec_ac_dot_vec_bc := dot_prod vec_ac vec_bc in
  let vec_ad_dot_vec_bd := dot_prod vec_ad vec_bd in
  let cosine_theta := dot_prod vec_ac vec_bd / (norm vec_ac * norm vec_bd) in
  let norm_cd := norm (xd - xc, yd - yc, zd - zc) in
  let norm_ab := norm (xa - xb, ya - yb, za - zb) in
  (vec_ac_dot_vec_bc = 0) → (vec_ad_dot_vec_bd = 0) → (cosine_theta ≤ norm_cd / norm_ab)

-- Statement without proof
theorem cosine_inequality_theorem (A B C D : ℝ × ℝ × ℝ) :
  cosine_inequality A B C D :=
sorry

end cosine_inequality_theorem_l570_570239


namespace count_odd_digits_base4_157_l570_570390

def base_repr (n b : ℕ) : List ℕ :=
  if b ≤ 1 then [n] else (List.unfoldr (λ x, if x = 0 then none else some (x % b, x / b)) n).reverse

def count_odd_digits (l : List ℕ) : ℕ :=
  l.countp (λ d, d % 2 = 1)

theorem count_odd_digits_base4_157 : count_odd_digits (base_repr 157 4) = 3 := by
  sorry

end count_odd_digits_base4_157_l570_570390


namespace max_sqrt_expr_l570_570430

theorem max_sqrt_expr (x : ℝ) (h₀ : 0 < x) (h₁ : x < 1/9) : (sqrt (x * (1 - 9 * x)) ≤ 1 / 6) :=
by 
  sorry

end max_sqrt_expr_l570_570430


namespace best_graph_for_temperature_l570_570635

-- Define the context and conditions of the problem
inductive GraphType
| LineGraph
| PieChart
| BarGraph
| FrequencyDistributionHistogram

-- Define the type of graphs available
def graph_types : List GraphType := [
  GraphType.LineGraph,
  GraphType.PieChart,
  GraphType.BarGraph,
  GraphType.FrequencyDistributionHistogram
]

-- Define the problem statement
def best_graph_to_reflect_changes_in_temperature (g : GraphType) : Prop :=
  g = GraphType.LineGraph

-- State the theorem to be proved
theorem best_graph_for_temperature : 
  ∃ g : GraphType, best_graph_to_reflect_changes_in_temperature g ∧ graph_types.contains g :=
by
  use GraphType.LineGraph
  split
  -- Here you would need to provide a proof, but as per instructions, we'll use sorry
  . exact rfl
  . sorry

end best_graph_for_temperature_l570_570635


namespace range_of_a_l570_570871

def set_A (a : ℝ) : Set ℝ := {x : ℝ | a - 1 < x ∧ x < 2 * a + 1}
def set_B : Set ℝ := {x : ℝ | 0 < x ∧ x < 1}

theorem range_of_a (a : ℝ) : (set_A a ∩ set_B = ∅) ↔ (a ≤ -2 ∨ (a > -2 ∧ a ≤ -1/2) ∨ a ≥ 2) := by
  sorry

end range_of_a_l570_570871


namespace cost_to_fill_pool_l570_570928

noncomputable def pool_cost : ℝ :=
  let base_width := 6
  let top_width := 4
  let length := 20
  let depth := 10
  let conversion_factor := 25
  let price_per_liter := 3
  let tax_rate := 0.08
  let discount_rate := 0.05
  let volume := 0.5 * depth * (base_width + top_width) * length
  let liters := volume * conversion_factor
  let initial_cost := liters * price_per_liter
  let cost_with_tax := initial_cost * (1 + tax_rate)
  let final_cost := cost_with_tax * (1 - discount_rate)
  final_cost

theorem cost_to_fill_pool : pool_cost = 76950 := by
  sorry

end cost_to_fill_pool_l570_570928


namespace graph_not_pass_first_quadrant_l570_570429

theorem graph_not_pass_first_quadrant (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : b < -1) :
  ¬ (∃ x y : ℝ, y = a^x + b ∧ x > 0 ∧ y > 0) :=
sorry

end graph_not_pass_first_quadrant_l570_570429


namespace distance_from_A_to_B_l570_570971

theorem distance_from_A_to_B (d C1A C1B C2A C2B : ℝ) (h1 : C1A + C1B = d)
  (h2 : C2A + C2B = d) (h3 : (C1A = 2 * C1B) ∨ (C1B = 2 * C1A)) 
  (h4 : (C2A = 3 * C2B) ∨ (C2B = 3 * C2A))
  (h5 : |C2A - C1A| = 10) : d = 120 ∨ d = 24 :=
sorry

end distance_from_A_to_B_l570_570971


namespace incorrect_judgment_l570_570418

-- Definitions based on the conditions
def contingency_table : Type := { sunset_rain : Nat, sunset_no_rain : Nat, nosunset_rain : Nat, nosunset_no_rain : Nat }

def data : contingency_table := { sunset_rain := 25, sunset_no_rain := 5, nosunset_rain := 25, nosunset_no_rain := 45 }

def total_days_observed : Nat := 100

def chi2_calculated : ℝ := 19.05

def chi2_critical : ℝ := 10.828

-- Statement to prove
theorem incorrect_judgment :
  let is_incorrect_judgment (chi2 : ℝ) (critical : ℝ) : Prop := chi2 > critical 
  ¬ is_incorrect_judgment chi2_calculated chi2_critical -> false := by
  sorry

end incorrect_judgment_l570_570418


namespace negation_of_prop_l570_570609

theorem negation_of_prop :
  ¬(∀ x : ℝ, x^3 - x^2 + 1 > 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 ≤ 0 :=
by
  sorry

end negation_of_prop_l570_570609


namespace range_of_a_l570_570462

noncomputable def f (a x : ℝ) : ℝ :=
if x > 4 then a * x + log 5 x
else if x > 0 then x^2 + 2^x + 3
else if x < -4 then -a * x + log 5 (-x)
else x^2 + 2^(-x) + 3

theorem range_of_a (a : ℝ) (h : f a (-5) < f a 2) : a < 2 :=
by sorry

end range_of_a_l570_570462


namespace solve_for_x_l570_570996

theorem solve_for_x : ∀ x : ℝ, 3^(2*x + 1) = 1/81 → x = -5/2 :=
by
  intro x
  intro h
  have h1 : 1/81 = 3^(-4) := by
    exact one_div_pow (3 : ℝ) 4
  rw [h1] at h
  have h_exp : 3^(2*x + 1) = 3^(-4) := h
  have eq_exp : 2*x + 1 = -4 := by
    apply eq_of_monotone_of_pow_eq
    apply pow_ne_zero
    norm_num
    exact h_exp
  linarith

end solve_for_x_l570_570996


namespace find_a_l570_570457

theorem find_a (a : ℝ) : 
  (∀ (i : ℂ), i^2 = -1 → (a * i / (2 - i) + 1 = 2 * i)) → a = 5 :=
by
  intro h
  sorry

end find_a_l570_570457


namespace max_value_quad_l570_570889

theorem max_value_quad (x : ℝ) :
  let y := -3 * x ^ 2 + 6 * x + 4 in 
  ∀ x : ℝ, y ≤ 7 :=
begin
  sorry
end

end max_value_quad_l570_570889


namespace ana_wins_probability_l570_570168

theorem ana_wins_probability :
  let p := 1 / 2
  in let prob_ana_wins_on_nth_turn (n : ℕ) := p^(4 * n)
  in ∑' (n : ℕ), prob_ana_wins_on_nth_turn n = 1 / 15 := sorry

end ana_wins_probability_l570_570168


namespace sum_of_cubes_mod_6_l570_570943

theorem sum_of_cubes_mod_6 (a : Fin 2018 → ℕ)
  (h_inc : ∀ i j, i < j → a i < a j)
  (h_sum : (Finset.univ.sum a) = 2018 ^ 2018) :
  (Finset.univ.sum (λ i, (a i) ^ 3)) % 6 = 2 :=
by
  sorry

end sum_of_cubes_mod_6_l570_570943


namespace percentage_of_girls_not_attending_college_l570_570297

theorem percentage_of_girls_not_attending_college 
  (boys girls : ℕ)
  (h_boys : boys = 300)
  (h_girls : girls = 240)
  (percent_boys_not_attend : ℝ)
  (total_percent_attended : ℝ)
  (h_percent_boys_not_attend : percent_boys_not_attend = 0.3)
  (h_total_percent_attended : total_percent_attended = 0.7)
  : (72 / 240) * 100 = 30 :=
by
  -- Constants based on given conditions
  have h_total_students : ℕ := boys + girls
  have h_num_boys_not_attend : ℕ := percent_boys_not_attend * boys
  have h_num_students_attended : ℕ := total_percent_attended * h_total_students
  have h_total_not_attend : ℕ := h_total_students - h_num_students_attended

  -- Deriving number of girls who did not attend college
  have h_num_girls_not_attend : ℕ := h_total_not_attend - h_num_boys_not_attend
  
  -- Proving the percentage of girls who did not attend college
  show (h_num_girls_not_attend / girls) * 100 = 30, from 
  sorry

end percentage_of_girls_not_attending_college_l570_570297


namespace find_common_ratio_l570_570464

noncomputable def common_ratio_of_geometric_sequence (a : ℕ → ℝ) (d : ℝ) 
  (h1 : d ≠ 0)
  (h2 : ∀ n, a (n + 1) = a n + d)
  (h3 : ∃ (r : ℝ), r ≠ 0 ∧ a 3 ^ 2 = a 1 * a 9) : ℝ :=
3

theorem find_common_ratio 
( a : ℕ → ℝ) 
( d : ℝ) 
(h1 : d ≠ 0)
(h2 : ∀ n, a (n + 1) = a n + d)
(h3 : ∃ (r : ℝ), r ≠ 0 ∧ a 3 ^ 2 = a 1 * a 9) :
common_ratio_of_geometric_sequence a d h1 h2 h3 = 3 :=
sorry

end find_common_ratio_l570_570464


namespace mark_old_bills_l570_570206

noncomputable def old_hourly_wage : ℝ := 40
noncomputable def new_hourly_wage : ℝ := 42
noncomputable def work_hours_per_week : ℝ := 8 * 5
noncomputable def personal_trainer_cost_per_week : ℝ := 100
noncomputable def leftover_after_expenses : ℝ := 980

noncomputable def new_weekly_earnings := new_hourly_wage * work_hours_per_week
noncomputable def total_weekly_spending_after_raise := leftover_after_expenses + personal_trainer_cost_per_week
noncomputable def old_bills_per_week := new_weekly_earnings - total_weekly_spending_after_raise

theorem mark_old_bills : old_bills_per_week = 600 := by
  sorry

end mark_old_bills_l570_570206


namespace l_perpendicular_to_m_l570_570083

variables (α β : Type) [plane α] [plane β]
variables (l : line α) (m : line)

-- Conditions
variable (line_l_in_plane_α : l ∈ α)
variable (line_m_not_in_plane_α : m ∉ α)
variable (planes_parallel : α ∥ β)
variable (m_perpendicular_to_β : m ⊥ β)

-- Statement to be proved
theorem l_perpendicular_to_m :
  l ⊥ m :=
sorry

end l_perpendicular_to_m_l570_570083


namespace total_volume_of_structure_l570_570049

noncomputable def volume_of_structure (diameter : ℝ) (height_cone : ℝ) (height_cylinder : ℝ) : ℝ :=
  let radius := diameter / 2
  let volume_cone := (1/3) * real.pi * radius^2 * height_cone
  let volume_cylinder := real.pi * radius^2 * height_cylinder
  volume_cone + volume_cylinder

theorem total_volume_of_structure :
  volume_of_structure 8 10 4 = (352 / 3) * real.pi :=
by
  sorry

end total_volume_of_structure_l570_570049


namespace quadrant_of_z_l570_570601

def i : ℂ := complex.I

def z : ℂ := (2 - i) * i

theorem quadrant_of_z : 1 ≤ z.re ∧ z.re ≤ 2 ∧ 1 ≤ z.im ∧ z.im ≤ 2 → z ∈ quadrant_1 :=
by
  sorry

end quadrant_of_z_l570_570601


namespace number_of_tiles_l570_570335

-- Define the conditions
def hall_condition (x y : ℝ) : Prop := 4 * |x| + 5 * |y| ≤ 20

-- Define the area of a tile
def tile_area : ℝ := (1 * (5 / 4)) / 2

-- Calculate the number of tiles required to cover the area of the hall
theorem number_of_tiles (A : ℝ) (H : ∀ (x y : ℝ), hall_condition x y → (A = 40)) : (40 / tile_area) = 64 :=
by
  sorry

end number_of_tiles_l570_570335


namespace unique_k_n_ineq_l570_570656

theorem unique_k_n_ineq (n : ℕ) (h1 : 0 < n) :
  (∃ (k : ℤ), (9 : ℝ) / 17 < n / (n + k) ∧ n / (n + k) < (8 : ℝ) / 15 ∧ ∀ k1 k2 : ℤ, k1 ≠ k2 → ¬ ((9 : ℝ) / 17 < n / (n + k1) ∧ n / (n + k1) < (8 : ℝ) / 15 ∧ (9 : ℝ) / 17 < n / (n + k2) ∧ n / (n + k2) < (8 : ℝ) / 15))) ↔ n = 3 :=
by
  sorry

end unique_k_n_ineq_l570_570656


namespace number_of_odd_digits_in_base_4_representation_l570_570402

-- Define the context and problem
def count_odd_digits_base_4 (n : ℕ) : ℕ :=
  let digits := (n.natDigits 4)
  in digits.count odd
  
theorem number_of_odd_digits_in_base_4_representation (n : ℕ) (h : n = 157) :
  count_odd_digits_base_4 n = 3 := by
  sorry

end number_of_odd_digits_in_base_4_representation_l570_570402


namespace person_age_in_1893_l570_570509

theorem person_age_in_1893 
    (x y : ℕ)
    (h1 : 0 ≤ x ∧ x < 10)
    (h2 : 0 ≤ y ∧ y < 10)
    (h3 : 1 + 8 + x + y = 93 - 10 * x - y) : 
    1893 - (1800 + 10 * x + y) = 24 :=
by
  sorry

end person_age_in_1893_l570_570509


namespace evaluate_expression_l570_570379

theorem evaluate_expression : abs (7 - 8 * (3 - 12)) - abs (5 - 11) = 73 :=
by {
  sorry
}

end evaluate_expression_l570_570379


namespace min_c_value_l570_570565

theorem min_c_value 
  (a b c : ℕ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (h3 : ∀ x y : ℕ, 
        (2 * x + y = 2025) → 
        (y = |x - a| + |x - b| + |x - c|) → 
        (∃! (x y : ℕ), (2 * x + y = 2025) ∧ (y = |x - a| + |x - b| + |x - c|))) : 
c = 1013 :=
sorry

end min_c_value_l570_570565


namespace min_value_expr_l570_570197

theorem min_value_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ m : ℝ, m = sqrt ((1 + real.cbrt 2 ^ 2) * (4 + real.cbrt 2 ^ 2) / (real.cbrt 2 ^ 2)) ∧
  ∀ a b : ℝ, 0 < a → 0 < b → 
    sqrt ((a^2 + b^2) * (4 * a^2 + b^2)) / (a * b) ≥ m :=
sorry

end min_value_expr_l570_570197


namespace sum_of_rational_roots_is_6_l570_570406

noncomputable def h : Polynomial ℚ := Polynomial.X^3 - 6 * Polynomial.X^2 + 11 * Polynomial.X - 6

theorem sum_of_rational_roots_is_6 : (h.roots.filter (λ r, r.is_rat)).sum = 6 := by
  sorry

end sum_of_rational_roots_is_6_l570_570406


namespace find_b_l570_570116

variable (a b : Prod ℝ ℝ)
variable (x y : ℝ)

theorem find_b (h1 : (Prod.fst a + Prod.fst b = 0) ∧
                    (Real.sqrt ((Prod.snd a + Prod.snd b) ^ 2) = 1))
                    (h2 : a = (2, -1)) :
                    b = (-2, 2) ∨ b = (-2, 0) :=
by sorry

end find_b_l570_570116


namespace wizard_achievable_for_odd_n_l570_570625

-- Define what it means for the wizard to achieve his goal
def wizard_goal_achievable (n : ℕ) : Prop :=
  ∃ (pairs : Finset (ℕ × ℕ)), 
    pairs.card = 2 * n ∧ 
    ∀ (sorcerer_breaks : Finset (ℕ × ℕ)), sorcerer_breaks.card = n → 
      ∃ (dwarves : Finset ℕ), dwarves.card = 2 * n ∧
      ∀ k ∈ dwarves, ((k, (k + 1) % n) ∈ pairs ∨ ((k + 1) % n, k) ∈ pairs) ∧
                     (∀ i j, (i, j) ∈ sorcerer_breaks → ¬((i, j) ∈ pairs ∨ (j, i) ∈ pairs))

theorem wizard_achievable_for_odd_n (n : ℕ) (h : Odd n) : wizard_goal_achievable n := sorry

end wizard_achievable_for_odd_n_l570_570625


namespace ratio_of_c_d_l570_570733

theorem ratio_of_c_d 
  (x y c d : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hd : d ≠ 0)
  (h1 : 8 * x - 6 * y = c)
  (h2 : 12 * y - 18 * x = d) :
  c / d = -4 / 3 :=
by 
  sorry

end ratio_of_c_d_l570_570733


namespace cylinder_height_is_six_l570_570685

-- Define Volume, Base Area, and Height
variable (Volume BaseArea : ℝ)
variable (hc_height : ℝ) -- height of the cone
variable (h_equal_volumes : Volume) 
variable (h_equal_base_areas : BaseArea)

-- Given conditions
def equal_volumes (Vcylinder Vcone : ℝ) := Vcylinder = Vcone
def equal_base_areas (Acylinder Acone : ℝ) := Acylinder = Acone
def cone_height (height : ℝ) := height = 18

-- Define height of cylinder in terms of cone's height
def cylinder_height (height_of_cone : ℝ) := height_of_cone / 3

-- Theorem: Given the conditions, we can prove that the height of the cylinder is 6 decimeters.
theorem cylinder_height_is_six 
  (h1 : equal_volumes Volume Volume) 
  (h2 : equal_base_areas BaseArea BaseArea)
  (h3 : cone_height hc_height) : 
  cylinder_height hc_height = 6 :=
by
  sorry

end cylinder_height_is_six_l570_570685


namespace new_surface_area_unchanged_l570_570969

theorem new_surface_area_unchanged
(diameter : ℝ)
(radius : ℝ)
(surface_area : ℝ)
(initial_surface_area : 4 * Real.pi * radius^2 = 100000000 * Real.pi)
(vaporize_radius : radius = diameter / 2)
(explosion_effect : vaporize_radius = radius) :
  surface_area = 100000000 * Real.pi :=
by
  -- The conditions define a spherical planet and the effect of vaporized matter
  have h1 : radius = diameter / 2, from vaporize_radius,
  have h2 : surface_area = 4 * Real.pi * radius^2, from initial_surface_area,
  rw ← h2,
  exact initial_surface_area

end new_surface_area_unchanged_l570_570969


namespace smallest_positive_x_l570_570790

theorem smallest_positive_x 
  (x : ℝ) 
  (H : 0 < x) 
  (H_eq : ⌊x^2⌋ - x * ⌊x⌋ = 10) : 
  x = 131 / 11 :=
sorry

end smallest_positive_x_l570_570790


namespace perimeter_PQRSTU_l570_570523

open Real

structure Point :=
(x : ℝ)
(y : ℝ)

def distance (p1 p2 : Point) : ℝ :=
sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

def P := Point.mk 0 10
def Q := Point.mk 5 10
def R := Point.mk 5 5
def S := Point.mk 10 0
def T := Point.mk 0 0
def U := Point.mk 5 0

def perimeter : ℝ :=
  distance P Q +
  distance Q R +
  distance R S +
  distance S U +
  distance U T +
  distance T P

theorem perimeter_PQRSTU :
  perimeter = 25 + 5 * sqrt 2 :=
by 
  sorry

end perimeter_PQRSTU_l570_570523


namespace find_k_value_l570_570050

theorem find_k_value (k : ℝ) (x y : ℝ) (h1 : -3 * x + 2 * y = k) (h2 : 0.75 * x + y = 16) (h3 : x = -6) : k = 59 :=
by 
  sorry

end find_k_value_l570_570050


namespace intercept_range_eq_l570_570611

-- Definition of the function y = e^x
def exp_fn (x : ℝ) : ℝ := Real.exp x

-- Definition of the point on the curve
def point_on_curve (x0 : ℝ) : ℝ × ℝ := (x0, exp_fn x0)

-- Definition of the tangent line's y-intercept at a point (x0, e^x0)
def tangent_line_y_intercept (x0 : ℝ) : ℝ := exp_fn x0 * (1 - x0)

-- Definition to capture the range of the intercept values
def intercept_range : Set ℝ := {y | ∃ x0 : ℝ, y = tangent_line_y_intercept x0}

-- The theorem stating the range of the y-intercept values
theorem intercept_range_eq : intercept_range = {y : ℝ | y ≤ 1} :=
by
  sorry

end intercept_range_eq_l570_570611


namespace smallest_pos_value_correct_l570_570784

noncomputable def smallest_pos_real_number : ℝ :=
  let x := 131 / 11 in
  if x > 0 ∧ (x * x).floor - x * (x.floor) = 10 then x else 0

theorem smallest_pos_value_correct (x : ℝ) (hx : 0 < x ∧ (x * x).floor - x * x.floor = 10) :
  x = 131 / 11 :=
begin
  sorry
end

end smallest_pos_value_correct_l570_570784


namespace ratio_of_number_l570_570282

theorem ratio_of_number (x : ℕ) (h : x = 300) : x / 2 = 150 :=
by
  rw [h]
  norm_num
  sorry

end ratio_of_number_l570_570282


namespace minimum_value_ineq_l570_570127

variable (m n : ℝ)

noncomputable def minimum_value := (1 / (2 * m)) + (1 / n)

theorem minimum_value_ineq (h1 : m > 0) (h2 : n > 0) (h3 : m + 2 * n = 1) : minimum_value m n = 9 / 2 := 
sorry

end minimum_value_ineq_l570_570127


namespace evaluate_expression_l570_570755

namespace ProofProblem

variables (a b : ℝ)

def expression := (a / (a + b) + b / (a - b)) / (b / (a + b) - a / (a - b))

theorem evaluate_expression (ha : a ≠ 0) (hb : b ≠ 0) (h : b ≠ a ∧ b ≠ -a) :
  expression a b = -1 := sorry

end ProofProblem

end evaluate_expression_l570_570755


namespace sum_f_1_to_2012_eq_338_l570_570014

def f (x : ℝ) : ℝ :=
  if h1 : -3 ≤ x ∧ x < -1 then -(x + 2) ^ 2
  else if h2 : -1 ≤ x ∧ x < 3 then x
  else f (x - 6)

theorem sum_f_1_to_2012_eq_338 : (∑ i in finset.range 2012, f (i + 1)) = 338 := 
  sorry

end sum_f_1_to_2012_eq_338_l570_570014


namespace particle_at_k_l570_570593

-- Define q
def q (p : ℝ) := 1 - p

-- Define the probability of the particle being at position k at time n
noncomputable def P (n k : ℤ) (p : ℝ) : ℝ :=
  if (n - k) % 2 = 0 then
    let r := (n - k) / 2 in
    (Nat.choose n r) * p^(n - r) * (q p)^r
  else
    0

-- Main theorem statement
theorem particle_at_k (n k : ℤ) (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : 
  P n k p = if (n - k) % 2 = 0 then
    let r := (n - k) / 2 in
    (Nat.choose n r : ℝ) * p^(n - r) * (1 - p)^r 
  else
    0 :=
sorry

end particle_at_k_l570_570593


namespace cost_per_person_to_pool_l570_570241

-- Definitions based on the conditions
def total_money_earned : ℕ := 30
def money_left : ℕ := 5
def number_of_people : ℕ := 10

-- Proof that the cost per person is $2.50
theorem cost_per_person_to_pool
  (h1: total_money_earned = 30)
  (h2: money_left = 5)
  (h3: number_of_people = 10) :
  (total_money_earned - money_left) / number_of_people = 2.50 :=
by sorry

end cost_per_person_to_pool_l570_570241


namespace parabola_equation_is_y_squared_4x_l570_570082

open Real

def line_l1 (p : ℝ) : Prop := ∀ (x y : ℝ), 4 * x - 3 * y + 6 = 0
def line_l2 (p : ℝ) : Prop := ∀ (x : ℝ), x = -(p / 2)
def parabola_C (p : ℝ) : Prop := ∀ x y : ℝ, y^2 = 2 * p * x
def point_on_parabola (p a : ℝ) : Prop := ∃ x y : ℝ, x = (2 * a^2) / p ∧ y = 2 * a

def distance_to_l2 (p a : ℝ) : ℝ := (2 * a^2) / p + (p / 2)
def distance_to_l1 (p a : ℝ) : ℝ := (abs ((8 * a^2) / p - 6 * a + 6)) / 5
def total_distance (p a : ℝ) : ℝ := distance_to_l2 p a + distance_to_l1 p a

theorem parabola_equation_is_y_squared_4x (p : ℝ) :
  (∃ p : ℝ, p > 0 ∧
    (∀ x y : ℝ, 4 * x - 3 * y + 6 = 0) ∧
    (∀ x : ℝ, x = -(p / 2)) ∧
    (minimum_sum_d p := ∃ a : ℝ, total_distance p (p / 3) = 2) ∧
      parabola_C p) →
  parabola_C 2 :=
sorry

end parabola_equation_is_y_squared_4x_l570_570082


namespace percentage_increase_erica_l570_570365

noncomputable def merry_go_round_question 
  (dave_time : ℝ) (chuck_mult : ℝ) (erica_time : ℝ) : ℝ :=
  let chuck_time := dave_time * chuck_mult in
  let diff_time := erica_time - chuck_time in
  (diff_time / chuck_time) * 100

theorem percentage_increase_erica (dave_time : ℝ) (chuck_mult : ℝ) (erica_time : ℝ) 
  (h_dave : dave_time = 10) (h_chuck : chuck_mult = 5) (h_erica : erica_time = 65) :
  merry_go_round_question dave_time chuck_mult erica_time = 30 :=
by
  simp [merry_go_round_question, h_dave, h_chuck, h_erica]
  sorry

end percentage_increase_erica_l570_570365


namespace gcd_of_repeated_threedigit_is_const_l570_570706

-- Define the condition: A twelve-digit number formed by repeating a three-digit integer four times
def twelve_digit_repeated (n : ℕ) : ℕ := 1000000000 * n + 1000000 * n + 1000 * n + n

-- State the theorem that we need to prove
theorem gcd_of_repeated_threedigit_is_const (n : ℕ) (hn : 100 ≤ n ∧ n < 1000) :
  ∃ d, ∀ k ∈ Set.univ, ∃ a : ℕ, twelve_digit_repeated ((k : ℕ) + a * (1000 : ℕ)) = d * a :=
begin
  sorry
end

end gcd_of_repeated_threedigit_is_const_l570_570706


namespace triangle_BD_perpendicular_AC_l570_570158

noncomputable theory

open_locale classical

variables {A B C O L D : Point}
variables {circumcircle : Triangle Point → Circle Point}
variables [Geometry]

/-- Given: In triangle ABC, ∠B = 60°, O is the circumcenter, and L is the foot of the angle bisector of angle B.
      The circumcircle of triangle BOL meets the circumcircle of triangle ABC at point D ≠ B.
    Prove: BD ⊥ AC. -/
theorem triangle_BD_perpendicular_AC (h1 : ∠ B = 60)
                                     (h2 : circumcenter A B C O)
                                     (h3 : is_angle_bisector B L)
                                     (h4 : (circumcircle ⟨B, O, L⟩).intersects (circumcircle ⟨A, B, C⟩) (D ≠ B))
                                     : is_perpendicular (B, D) (A, C) :=
sorry

end triangle_BD_perpendicular_AC_l570_570158


namespace sum_of_vectors_is_zero_l570_570977

variable {V : Type} [AddCommGroup V] [FiniteDimensional ℝ V]

def is_center_of_square (O A B C D : V) : Prop :=
  dist O A = dist O B ∧
  dist O A = dist O C ∧
  dist O A = dist O D ∧
  dist A B = dist B C ∧
  dist B C = dist C D ∧
  dist C D = dist D A 

noncomputable def vector_sum_is_zero (O A B C D : V) : Prop :=
  (A - O) + (C - A) + (D - C) + (A - D) + (B - A) + (D - B) + (O - D) = 0

theorem sum_of_vectors_is_zero (O A B C D : V) (h : is_center_of_square O A B C D) : 
  vector_sum_is_zero O A B C D :=
by
  sorry

end sum_of_vectors_is_zero_l570_570977


namespace plates_count_l570_570271

variable (x : ℕ)
variable (first_taken : ℕ)
variable (second_taken : ℕ)
variable (remaining_plates : ℕ := 9)

noncomputable def plates_initial : ℕ :=
  let first_batch := (x - 2) / 3
  let remaining_after_first := x - 2 - first_batch
  let second_batch := remaining_after_first / 2
  let remaining_after_second := remaining_after_first - second_batch
  remaining_after_second

theorem plates_count (x : ℕ) (h : plates_initial x = remaining_plates) : x = 29 := sorry

end plates_count_l570_570271


namespace angle_ACB_120_l570_570159

open Real
open EuclideanGeometry

noncomputable def ABC_Problem (A B C D E F : Point) (AB AC :ℝ) : Prop :=
  Triangle A B C ∧
  AB = 3 * AC ∧
  Collinear [B, A, D] ∧
  Collinear [B, C, E] ∧
  Angle A B E = Angle A C D ∧
  IsIntersection F A E C D ∧
  IsIsosceles C F E (rfl : CF = FE) ∧
  MeasureAngle A C B = 120

theorem angle_ACB_120 (A B C D E F : Point) (AB AC :ℝ) (hABC : ABC_Problem A B C D E F AB AC) :
  MeasureAngle A C B = 120 :=
begin
  sorry
end

end angle_ACB_120_l570_570159


namespace stones_parity_l570_570264

-- Define the weights of each stone as integer variables
def weights : ℕ → ℤ := sorry

-- Define the condition that any 12 stones can be balanced
def can_balance (subset : Finset ℕ) : Prop := 
  ∃ a b : Finset ℕ, a.card = 6 ∧ b.card = 6 ∧ a ∪ b = subset ∧ a ∩ b = ∅ ∧ 
  (∑ i in a, weights i) = (∑ i in b, weights i)

-- The main statement we want to prove
theorem stones_parity :
  (∀ subset : Finset ℕ, subset.card = 12 → can_balance subset) →
  (∀ i j : ℕ, weights i % 2 = weights j % 2) :=
begin
  sorry
end

end stones_parity_l570_570264


namespace least_distance_between_ticks_l570_570211

theorem least_distance_between_ticks :
  ∃ z : ℝ, ∀ (a b : ℤ), (a / 5 ≠ b / 7) → abs (a / 5 - b / 7) = (1 / 35) := 
sorry

end least_distance_between_ticks_l570_570211


namespace solve_T_l570_570974

theorem solve_T : ∃ T : ℝ, (1 / 3) * (1 / 6) * T = (1 / 4) * (1 / 8) * 120 ∧ T = 67.5 :=
by
  have h_eq : (1 / 3) * (1 / 6) * (67.5) = (1 / 4) * (1 / 8) * 120 :=
    by
      sorry
  use 67.5
  exact ⟨h_eq, rfl⟩

end solve_T_l570_570974


namespace age_when_Billy_born_l570_570128

-- Definitions based on conditions
def current_age_I := 4 * 4
def current_age_Billy := 4
def age_difference := current_age_I - current_age_Billy

-- Statement to prove
theorem age_when_Billy_born : age_difference = 12 :=
by
  -- Expose the calculation steps
  calc
    age_difference
    = 4 * 4 - 4 : by rw [current_age_I, current_age_Billy]
    ... = 16 - 4 : by norm_num
    ... = 12 : by norm_num

end age_when_Billy_born_l570_570128


namespace modulus_of_complex_z_l570_570501

theorem modulus_of_complex_z
  (z : ℂ)
  (h : 2 + z * complex.i = z - 2 * complex.i) :
  complex.abs z = 2 :=
sorry

end modulus_of_complex_z_l570_570501


namespace number_of_odd_digits_in_base_4_representation_l570_570401

-- Define the context and problem
def count_odd_digits_base_4 (n : ℕ) : ℕ :=
  let digits := (n.natDigits 4)
  in digits.count odd
  
theorem number_of_odd_digits_in_base_4_representation (n : ℕ) (h : n = 157) :
  count_odd_digits_base_4 n = 3 := by
  sorry

end number_of_odd_digits_in_base_4_representation_l570_570401


namespace min_elements_to_have_five_coprime_l570_570550

def S := {1, 2, ..., 280}

-- Let's ensure our definition in Lean using set notation
def subset_of_S (n : ℕ) : set ℕ := {k ∈ S | k ≤ n}

-- Define what it means for a set to have 5 pairwise coprime elements
def has_five_pairwise_coprime (s : set ℕ) : Prop :=
  ∃ a b c d e ∈ s, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
  (nat.gcd a b = 1) ∧ (nat.gcd a c = 1) ∧ (nat.gcd a d = 1) ∧ (nat.gcd a e = 1) ∧ (nat.gcd b c = 1) ∧ (nat.gcd b d = 1) ∧ (nat.gcd b e = 1) ∧ 
  (nat.gcd c d = 1) ∧ (nat.gcd c e = 1) ∧ (nat.gcd d e = 1)

-- Define the proof problem statement
theorem min_elements_to_have_five_coprime : ∀ (n : ℕ), (∀ s ⊆ S, card s = n → has_five_pairwise_coprime s) ↔ n ≥ 217 := 
sorry

end min_elements_to_have_five_coprime_l570_570550


namespace solve_for_x_l570_570587

theorem solve_for_x (x : ℝ) : sqrt (9 + sqrt (27 + 9 * x)) + sqrt (3 + sqrt (3 + x)) = 3 + 3 * sqrt 3 → x = 33 := by
  sorry

end solve_for_x_l570_570587


namespace function_words_same_meaning_and_usage_l570_570673

/-- Prove that the function word "者" in option D 
has the same meaning and usage in classical Chinese, 
given the conditions provided in the problem statement. -/
theorem function_words_same_meaning_and_usage : 
  (∀ (meaning1 meaning2 : String), 
    meaning1 = "the person" ∧ meaning2 = "the person" → 「者」inOptionD)
  → (¬ ∀ (meaning1 meaning2 : String), 
    meaning1 = "indicating a transition" ∧ meaning2 = "indicating modification" → 「而」inOptionA)
  → (¬ ∀ (meaning1 meaning2 : String), 
    meaning1 = "compared to" ∧ meaning2 = "towards" → 「于」inOptionB)
  → (¬ ∀ (meaning1 meaning2 : String), 
    meaning1 = "pronoun" ∧ meaning2 = "between subject and predicate, meaningless" → 「之」inOptionC)
  → optionD_is_correct :=
begin
  intros hD hA hB hC,
  apply hD,
  split;
  reflexivity,
end

end function_words_same_meaning_and_usage_l570_570673


namespace product_of_roots_l570_570190

theorem product_of_roots (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
    (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 7 :=
by sorry

end product_of_roots_l570_570190


namespace max_y_value_l570_570191

noncomputable def y (x : ℝ) : ℝ := |x + 1| - 2 * |x| + |x - 2|

theorem max_y_value : ∃ α, (∀ x, -1 ≤ x ∧ x ≤ 2 → y x ≤ α) ∧ α = 3 := by
  sorry

end max_y_value_l570_570191


namespace power_sum_evaluation_l570_570674

theorem power_sum_evaluation :
  (-1)^(4^3) + 2^(3^2) = 513 :=
by
  sorry

end power_sum_evaluation_l570_570674


namespace radians_turned_through_l570_570332

-- Define the given constants and conditions
def radius : ℝ := 2000
def speed : ℝ := 25 / 3
def time : ℝ := 10

-- Define the arc length covered by the train in 10 seconds
def arc_length : ℝ := speed * time

-- Define the angle in radians using arc length and radius
def angle : ℝ := arc_length / radius

-- State the theorem to be proved
theorem radians_turned_through : angle = 1 / 24 :=
by
  -- Proof will go here
  sorry

end radians_turned_through_l570_570332


namespace geometric_sequence_property_l570_570148

theorem geometric_sequence_property (b : ℕ → ℝ) (N : ℕ) (hN : N = 2011) :
  (∏ i in finset.range (N/2 + 1), b (2 * i + 1)) / (∏ i in finset.range (N/2), b (2 * (i + 1))) = b (N / 2) :=
by
  sorry

end geometric_sequence_property_l570_570148


namespace ways_to_select_fuwa_sets_l570_570232

def types : Finset String := {"贝贝", "晶晶", "欢欢", "迎迎", "妮妮"}

theorem ways_to_select_fuwa_sets (h : ∀ t ∈ types, 2) :
  ∃ (n : ℕ), n = 160 := by
  -- Proof should be filled here
  sorry

end ways_to_select_fuwa_sets_l570_570232


namespace math_proof_problem_l570_570843

noncomputable def line_param_eqs := (t : ℝ) → (x : ℝ, y : ℝ) := (2 - 1/2*t, 1 + sqrt(3)/2*t)

def rect_eq_of_line (t : ℝ) : Prop := 
  let (x, y) := line_param_eqs t
  x = 2 - 1/2*t ∧ y = 1 + sqrt(3)/2*t

def polar_to_rect_eq : Prop := 
  ∀ (ρ : ℝ), (ρ = 2) → (ρ^2 = 4) ∧ (ρ^2 = x^2 + y^2 → x^2 + y^2 = 4)

def transformed_curve_eq : Prop := 
  ∀ (x y x' y' : ℝ), (x' = x ∧ y' = 2*y) → (x^2 + y^2/4 = 4 → x^2/4 + y^2/16 = 1)

def range_of_transformed_point : Prop := 
  ∀ (θ : ℝ), let x0 := 2 * cos θ in let y0 := 4 * sin θ in 
  -4 ≤ sqrt (3) * x0 + 1/2 * y0 ∧ sqrt (3) * x0 + 1/2 * y0 ≤ 4

theorem math_proof_problem : rect_eq_of_line t ∧ polar_to_rect_eq ∧ transformed_curve_eq ∧ range_of_transformed_point :=
by 
  sorry

end math_proof_problem_l570_570843


namespace max_value_of_fraction_l570_570444

theorem max_value_of_fraction (a b c : ℝ) (h1 : 0 < a) 
  (h2 : b^2 ≤ 4 * a * c - 4 * a^2) (h3 : ∀ x : ℝ, a * x^2 + (b - 2 * a) * x + (c - b) ≥ 0) : 
  ∃ t, t = (c / a) ∧ t > 1 ∧ (b^2 / (a^2 + 2 * c^2) = sqrt 6 - 2) := 
sorry

end max_value_of_fraction_l570_570444


namespace general_term_a_n_sum_of_b_n_l570_570938

-- Define the sequence a_n
axiom a : ℕ → ℝ
axiom S : ℕ → ℝ

-- Conditions
axiom a_positive : ∀ n, a n > 0
axiom a_squared_plus_a : ∀ n, (a n)^2 + a n = 2 * (S n) + 2

-- First part: Prove the general term formula for {a_n}
theorem general_term_a_n (n : ℕ) : a n = n + 1 :=
sorry

-- Define the sequence b_n
def b (n : ℕ) : ℝ := 2 / (a n * a (n + 1))

-- Define T_n as the sum of the first n terms of the sequence {b_n}
noncomputable def T (n : ℕ) : ℝ := ∑ i in finset.range n, b i

-- Second part: Prove the sum of the first n terms of the sequence {b_n}
theorem sum_of_b_n (n : ℕ) : T n = n / (n + 2) :=
sorry

end general_term_a_n_sum_of_b_n_l570_570938


namespace projection_vector_l570_570431

theorem projection_vector : 
  let a := (-1 : ℝ, 2, 1)
  let b := (-2 : ℝ, -2, 4)
  (a.1 * b.1 + a.2 * b.2 + a.3 * b.3) / (a.1 * a.1 + a.2 * a.2 + a.3 * a.3) * a = (-1/3, 2/3, 1/3 : ℝ) :=
by
  -- Sorry to skip the proof
  sorry

end projection_vector_l570_570431


namespace percentage_more_than_l570_570975

variable (P Q : ℝ)

-- P gets 20% more than Q
def getsMoreThan (P Q : ℝ) : Prop :=
  P = 1.20 * Q

-- Q gets 20% less than P
def getsLessThan (Q P : ℝ) : Prop :=
  Q = 0.80 * P

theorem percentage_more_than :
  getsLessThan Q P → getsMoreThan P Q := 
sorry

end percentage_more_than_l570_570975


namespace compare_M_N_l570_570178

variable (a : ℝ)

def M : ℝ := 2 * a * (a - 2) + 7
def N : ℝ := (a - 2) * (a - 3)

theorem compare_M_N : M a > N a :=
by
  sorry

end compare_M_N_l570_570178


namespace sawyer_saw_octopuses_l570_570582

def number_of_legs := 40
def legs_per_octopus := 8

theorem sawyer_saw_octopuses : number_of_legs / legs_per_octopus = 5 := 
by
  sorry

end sawyer_saw_octopuses_l570_570582


namespace cone_angle_60_degrees_l570_570028

theorem cone_angle_60_degrees (r : ℝ) (h : ℝ) (θ : ℝ) 
  (arc_len : θ = 60) 
  (slant_height : h = r) : θ = 60 :=
sorry

end cone_angle_60_degrees_l570_570028


namespace sum_coefficients_zero_l570_570092

noncomputable def k : ℝ := ∫ x in 0..real.pi, (real.sin x - real.cos x)

theorem sum_coefficients_zero
  (a : ℕ → ℝ)
  (h : ∀ x : ℝ, (1 - k * x)^8 = finset.sum (finset.range 9) (λ n, a n * x^n)) :
  finset.sum (finset.range 8) (λ n, a (n + 1)) = 0 :=
by
  -- the proof should be filled here
  sorry

end sum_coefficients_zero_l570_570092


namespace expected_value_Y_variance_Y_l570_570109

theorem expected_value_Y 
  (X Y : ℤ → ℝ)
  (h1 : ∀ t, X t + Y t = 10)
  (h2 : X ∼ Binomial 10 0.8) :
  E(Y) = 2 :=
sorry

theorem variance_Y 
  (X Y : ℤ → ℝ)
  (h1 : ∀ t, X t + Y t = 10)
  (h2 : X ∼ Binomial 10 0.8) :
  D(Y) = 1.6 :=
sorry

end expected_value_Y_variance_Y_l570_570109


namespace integral_abs_value_problem_l570_570756

theorem integral_abs_value_problem :
  ∫ x in 0..2, (2 - |1 - x|) = 3 := by
  sorry

end integral_abs_value_problem_l570_570756


namespace ellipse_eq_max_area_AEBF_l570_570831

open Real

section ellipse_parabola_problem

variables {a b : ℝ} (F1 : ℝ × ℝ) (F2 : ℝ × ℝ) (x y k : ℝ) {M : ℝ × ℝ} {AO BO : ℝ} 
  (b_pos : 0 < b) (a_gt_b : b < a) (MF1_dist : abs (y - 1) = 5 / 3) (M_on_parabola : x^2 = 4 * y)
  (M_on_ellipse : (y / a)^2 + (x / b)^2 = 1) (A : ℝ × ℝ) (B : ℝ × ℝ) (D : ℝ × ℝ)
  (E F : ℝ × ℝ) (A_on_x : A.1 = b ∧ A.2 = 0) (B_on_y : B.1 = 0 ∧ B.2 = a)
  (D_intersect : D.2 = k * D.1) (E_on_ellipse : (E.2 / a)^2 + (E.1 / b)^2 = 1) 
  (F_on_ellipse : (F.2 / a)^2 + (F.1 / b)^2 = 1)
  (k_pos : 0 < k)

theorem ellipse_eq :
  a = 2 ∧ b = sqrt 3 → (y^2 / (2:ℝ)^2 + x^2 / (sqrt 3:ℝ)^2 = 1) :=
sorry

theorem max_area_AEBF :
  (a = 2 ∧ b = sqrt 3) →
  ∃ max_area : ℝ, max_area = 2 * sqrt 6 :=
sorry

end ellipse_parabola_problem

end ellipse_eq_max_area_AEBF_l570_570831


namespace min_value_sqrt_expression_l570_570196

theorem min_value_sqrt_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ c : ℝ, c = 3 ∧ ∀ x : ℝ, x = (sqrt ((a^2 + b^2) * (4 * a^2 + b^2)) / (a * b)) → x ≥ c :=
sorry

end min_value_sqrt_expression_l570_570196


namespace center_of_incircle_in_CDE_l570_570933

-- Definitions
variable (A B C D E F P : Type)
variable [triangle ABC : Type]
variable [right_angle (∠ A C B)]
variable [midpoint E B C]
variable [midpoint F A C]
variable [altitude D C A B]
variable [intersection P (internal_angle_bisector A) (line_through E F)]

-- The main theorem to prove
theorem center_of_incircle_in_CDE : is_incenter P (triangle C D E) :=
sorry

end center_of_incircle_in_CDE_l570_570933


namespace minimize_expression_l570_570185

open Real

theorem minimize_expression (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) :
  4 * p^3 + 6 * q^3 + 24 * r^3 + 8 / (3 * p * q * r) ≥ 16 :=
sorry

end minimize_expression_l570_570185


namespace number_of_vip_children_l570_570584

theorem number_of_vip_children (total_attendees children_percentage children_vip_percentage : ℕ) :
  total_attendees = 400 →
  children_percentage = 75 →
  children_vip_percentage = 20 →
  (total_attendees * children_percentage / 100) * children_vip_percentage / 100 = 60 :=
by
  intros h_total h_children_pct h_vip_pct
  sorry

end number_of_vip_children_l570_570584


namespace complex_solution_l570_570036

variables {n : ℕ}
variables {x : ℕ → ℂ}

-- Define the conditions
def condition1 (x : ℕ → ℂ) : Prop := ∑ i in finset.range n, x i = n
def condition2 (x : ℕ → ℂ) : Prop := ∑ i in finset.range n, (x i)^2 = n
def condition_general (x : ℕ → ℂ) (k : ℕ) : Prop := ∑ i in finset.range n, (x i)^k = n

-- The main theorem statement
theorem complex_solution : 
  (∀ k : ℕ, (1 ≤ k) → (k ≤ n) → condition_general x k) →
  (∃ i : ℕ, (i < n) ∧ x i = 1 ∧ (∀ j : ℕ, (j ≠ i) → j < n → x j = 0)) :=
by { sorry }

end complex_solution_l570_570036


namespace focus_dist_directrix_l570_570865

def parabola_focus_distance {x y : Type} [Field x] [Field y] := 
  ∀ {p : x}, {4 * p = 4} → p = 1 → 
  let F := (p, 0) in 
  let directrix := x.val = -p in 
  |F.val - -p| = 2

theorem focus_dist_directrix : parabola_focus_distance :=
by 
  sorry

end focus_dist_directrix_l570_570865


namespace line_passes_through_fixed_point_l570_570243

theorem line_passes_through_fixed_point (k : ℝ) : ∃ (P : ℝ × ℝ), P = (3, 6) ∧ ∀ x y, y = k * (x - 3) + 6 → y = 6 ↔ x = 3 :=
by
  use (3, 6)
  intro x y
  split
  { -- First direction
    intro h
    rw [h]
    simp },
  { -- Second direction
    intro h
    rw [h]
    simp }

end line_passes_through_fixed_point_l570_570243


namespace min_balls_to_guarantee_12_of_one_color_l570_570314

noncomputable def draw_balls : Prop :=
  let red := 30
  let green := 25
  let yellow := 20
  let blue := 15
  let white := 10
  let black := 5
  ∀ n : ℕ, (n ≥ 57) → 
    (∃ (r g y b w k : ℕ), r + g + y + b + w + k = n ∧ 
    r ≤ red ∧ g ≤ green ∧ y ≤ yellow ∧ b ≤ blue ∧ w ≤ white ∧ k ≤ black ∧ 
    (even g) ∧ (even w) ∧ 
    (r ≥ 12 ∨ g ≥ 12 ∨ y ≥ 12 ∨ b ≥ 12 ∨ w ≥ 12 ∨ k ≥ 12))

theorem min_balls_to_guarantee_12_of_one_color : draw_balls := sorry

end min_balls_to_guarantee_12_of_one_color_l570_570314


namespace bruno_score_lower_l570_570578

theorem bruno_score_lower (R B : ℕ) (hr : R = 62) (hb : B = 48) : R - B = 14 :=
by
  rw [hr, hb]
  exact rfl

end bruno_score_lower_l570_570578


namespace area_intersection_A_B_l570_570477

noncomputable def A : Set (Real × Real) := {
  p | ∃ α β : ℝ, p.1 = 2 * Real.sin α + 2 * Real.sin β ∧ p.2 = 2 * Real.cos α + 2 * Real.cos β
}

noncomputable def B : Set (Real × Real) := {
  p | Real.sin (p.1 + p.2) * Real.cos (p.1 + p.2) ≥ 0
}

theorem area_intersection_A_B :
  let intersection := Set.inter A B
  let area : ℝ := 8 * Real.pi
  ∀ (x y : ℝ), (x, y) ∈ intersection → True := sorry

end area_intersection_A_B_l570_570477


namespace proof_statement_l570_570917

-- Definitions of the problem
variables {α : Type} [linear_ordered_field α] {a b c : α} {A B C : angle α}

-- Conditions from the original problem
def conditionA (h : sin A > sin B) : Prop := A > B
def conditionB (h : sin^2 B + sin^2 C < sin^2 A) : Prop := 
  (a^2 > b^2 + c^2) ∨ (b^2 > a^2 + c^2) ∨ (c^2 > a^2 + b^2) -- At least one angle is obtuse
def conditionC (h : A + B < π/2 ∧ B + C < π/2 ∧ C + A < π/2) : Prop := sin A > cos B
def conditionD (h : a * cos A = b * cos B) : Prop := ¬(a ≠ b)

-- The corresponding Lean theorem
theorem proof_statement : 
  (∀ (A B C : angle α) (a b c : α),
    (conditionA (has order_alpha.sin A < order_alpha.sin B) →
    conditionB (has order_alpha.sin B^2 + sin C^2 < sin A^2) →
    conditionC (0 < A < π/2 ∧ 0 < B < π/2 ∧ 0 < C < π/2)
    conditionD (a * order_alpha.cos A = b * order_alpha.cos B) → 
    Prop :=
sorry


end proof_statement_l570_570917


namespace number_of_divisors_divisible_by_3_sum_of_divisors_divisible_by_3_l570_570046

def a : ℕ := (2^3) * (3^2) * (5^2)

theorem number_of_divisors_divisible_by_3 (a : ℕ) : 
  a = 2^3 * 3^2 * 5^2 → 
  ∃ n, n = 24 ∧ ∀ d, d ∣ a → 3 ∣ d → ∑ i in divisors (d), (i = n) :=
by sorry

theorem sum_of_divisors_divisible_by_3 (a : ℕ) :
  a = 2^3 * 3^2 * 5^2 → 
  ∃ s, s = 5580 ∧ ∑ d in divisors (a), (3 ∣ d) → (d = s) :=
by sorry

end number_of_divisors_divisible_by_3_sum_of_divisors_divisible_by_3_l570_570046


namespace number_of_proper_subsets_eq_127_l570_570245

def valid_set := {x : ℕ | 4 ≤ x ∧ x ≤ 10}

theorem number_of_proper_subsets_eq_127 : (2 ^ Fintype.card valid_set) - 1 = 127 := by
  have h1 : Fintype.card valid_set = 7 := by
    -- To be proved separately: the cardinality of valid_set is 7
    sorry
  simp only [h1, pow_succ, pow_zero, mul_one]
  norm_num
  sorry

end number_of_proper_subsets_eq_127_l570_570245


namespace find_dividend_l570_570642

-- Definitions based on conditions from the problem
def divisor : ℕ := 13
def quotient : ℕ := 17
def remainder : ℕ := 1

-- Statement of the proof problem
theorem find_dividend (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : ℕ :=
  (divisor * quotient) + remainder

-- Proof statement ensuring dividend is as expected
example : find_dividend divisor quotient remainder = 222 :=
by 
  sorry

end find_dividend_l570_570642


namespace problem1_proof_problem2_proof_l570_570478

-- Definitions for Proof Problem 1
def vec_a (x : ℝ) : ℝ × ℝ := (1, real.sin x)
def vec_b (x : ℝ) : ℝ × ℝ := (real.cos (2 * x + real.pi / 3), real.sin x)
def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2 - 1 / 2 * real.cos (2 * x)

-- Statement for Proof Problem 1
theorem problem1_proof (x : ℝ) :
  f(x) = real.sin (2 * x + 7 * real.pi / 6) + 1 / 2 :=
sorry

-- Definitions for Proof Problem 2
variables {A B C : ℝ}
noncomputable def side_c := real.sqrt 3
noncomputable def angle_C := real.pi / 3
noncomputable def perimeter (a b c : ℝ) := a + b + c

-- Statement for Proof Problem 2
theorem problem2_proof (a b : ℝ) :
  perimeter a b side_c ∈ Ioo (2 * real.sqrt 3) (3 * real.sqrt 3) :=
sorry

end problem1_proof_problem2_proof_l570_570478


namespace arithmetic_seq_15th_term_is_53_l570_570231

-- Define an arithmetic sequence
def arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

-- Original terms given
def a₁ : ℤ := -3
def d : ℤ := 4
def n : ℕ := 15

-- Prove that the 15th term is 53
theorem arithmetic_seq_15th_term_is_53 :
  arithmetic_seq a₁ d n = 53 :=
by
  sorry

end arithmetic_seq_15th_term_is_53_l570_570231


namespace reception_hall_tiling_l570_570334

theorem reception_hall_tiling :
  let rhombus_area := (1 / 2) * 10 * 8
  let tile_area := (1 / 2) * 1 * (5 / 4)
  rhombus_area / tile_area = 64 :=
by
  let rhombus_area := (1 / 2) * 10 * 8
  let tile_area := (1 / 2) * 1 * (5 / 4)
  calc
    rhombus_area / tile_area
      = 40 / (5 / 8) : by rw [show rhombus_area = 40, by norm_num, show tile_area = (5 / 8), by norm_num]
      ... = 40 * (8 / 5) : by rw div_eq_mul_inv
      ... = 40 * 1.6 : by norm_num
      ... = 64 : by norm_num

end reception_hall_tiling_l570_570334


namespace find_a2016_l570_570868

theorem find_a2016 (a : ℕ → ℤ) (h1 : a 1 = 4) (h2 : a 2 = 6) (h3 : ∀ n : ℕ, a (n + 2) = a (n + 1) - a n) : a 2016 = -2 := 
by sorry

end find_a2016_l570_570868


namespace proof_of_k_values_l570_570762

noncomputable def problem_statement : Prop :=
  ∀ k : ℝ,
    (∃ a b : ℝ, (6 * a^2 + 5 * a + k = 0 ∧ 6 * b^2 + 5 * b + k = 0 ∧ a ≠ b ∧
    |a - b| = 3 * (a^2 + b^2))) ↔ (k = 1 ∨ k = -20.717)

theorem proof_of_k_values : problem_statement :=
by sorry

end proof_of_k_values_l570_570762


namespace annual_income_is_36000_l570_570138

-- Definitions for conditions
def tax_rate_first_part (q : ℕ) (income : ℕ) : ℚ := (q / 100) * 30000
def tax_rate_second_part (q : ℕ) (income : ℕ) : ℚ := ((q + 3) / 100) * (income - 30000)
def total_tax (q : ℕ) (income : ℕ) : ℚ := (q / 100 + 0.005) * income

-- Main statement to prove
theorem annual_income_is_36000 (q : ℕ) (income : ℕ) :
  ((q / 100) * 30000 + ((q + 3) / 100) * (income - 30000) = (q / 100 + 0.005) * income) →
  income = 36000 := 
sorry

end annual_income_is_36000_l570_570138


namespace smallest_solution_l570_570776

def smallest_positive_real_x : ℝ :=
  (131 : ℝ) / 11

theorem smallest_solution (x : ℝ) (hx : 0 < x) (H : ⌊x^2⌋ - x * ⌊x⌋ = 10) : x = smallest_positive_real_x :=
  sorry

end smallest_solution_l570_570776


namespace relationship_among_f_values_l570_570098

variable {α : Type*} [linear_ordered_field α] 
variable {f : α → α}

-- Conditions
def increasing_on_interval (f : α → α) (a b : α) : Prop :=
  ∀ x y, a < x ∧ x < b → a < y ∧ y < b → x < y → f x < f y

def even_function (f : α → α) : Prop :=
  ∀ x, f (x + 2) = f (2 - x)

-- Problem statement
theorem relationship_among_f_values (h1 : increasing_on_interval f 0 2) (h2 : even_function f) :
  f 2.5 > f 1 ∧ f 1 > f 3.5 :=
sorry

end relationship_among_f_values_l570_570098


namespace average_k_of_polynomial_with_positive_integer_roots_l570_570844

-- Define the conditions and the final theorem

theorem average_k_of_polynomial_with_positive_integer_roots :
  (∑ i in {k | ∃ r1 r2 : ℕ+, r1 * r2 = 24 ∧ k = r1 + r2}.to_finset, i) / 
  ({k | ∃ r1 r2 : ℕ+, r1 * r2 = 24 ∧ k = r1 + r2}.to_finset.card : ℝ) = 15 :=
by
  sorry

end average_k_of_polynomial_with_positive_integer_roots_l570_570844


namespace reading_time_difference_l570_570637

theorem reading_time_difference
  (tristan_speed : ℕ := 120)
  (ella_speed : ℕ := 40)
  (book_pages : ℕ := 360) :
  let tristan_time := book_pages / tristan_speed
  let ella_time := book_pages / ella_speed
  let time_difference_hours := ella_time - tristan_time
  let time_difference_minutes := time_difference_hours * 60
  time_difference_minutes = 360 :=
by
  sorry

end reading_time_difference_l570_570637


namespace fraction_eq_l570_570013

def at_op (a b : ℝ) : ℝ := a * b - a * b^2
def hash_op (a b : ℝ) : ℝ := a^2 + b - a^2 * b

theorem fraction_eq :
  (at_op 8 3) / (hash_op 8 3) = 48 / 125 :=
by sorry

end fraction_eq_l570_570013


namespace perfect_number_of_mersenne_prime_l570_570575

theorem perfect_number_of_mersenne_prime
  (p : ℤ) 
  (h_prime : Nat.Prime (2^p - 1)) : 
  ∃ N : ℕ, N = 2^(p.to_nat - 1) * (2^p.to_nat - 1) ∧ 
           ∑ d in Nat.divisors N, d = 2 * N := 
begin
  sorry
end

end perfect_number_of_mersenne_prime_l570_570575


namespace parabola_circle_intersection_sum_distances_l570_570183

noncomputable def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_circle_intersection_sum_distances :
  let P1 := (-7, 98) in
  let P2 := (-1, 2) in
  let P3 := (6, 72) in
  let P4 := (2, 8) in
  let vertex := (0, 0) in
  distance vertex P1 + distance vertex P2 + distance vertex P3 + distance vertex P4 = 
    real.sqrt 9653 + real.sqrt 5 + real.sqrt 5220 + real.sqrt 68 :=
by
  sorry

end parabola_circle_intersection_sum_distances_l570_570183


namespace amusement_park_line_l570_570570

theorem amusement_park_line (h1 : Eunji_position = 6) (h2 : people_behind_Eunji = 7) : total_people_in_line = 13 :=
by
  sorry

end amusement_park_line_l570_570570


namespace minValueC_proof_l570_570045

noncomputable def minValueC (n : ℕ) (A : Finset ℕ) (f : ℕ → ℤ) : ℤ :=
  if n ≥ 4 ∧ (∀ x ∈ A, x ≤ n) ∧ A.card > (2 * n) / 3 then 
    let sum := A.sum (λ a, a * f a) in 
    if abs sum ≤ 1 then 2 / 3 else 0
  else 0

theorem minValueC_proof : ∀ (n : ℕ) (A : Finset ℕ) (f : ℕ → ℤ),
  n ≥ 4 → (∀ x ∈ A, x ≤ n) → A.card > (2 * n) / 3 →
  (∃ f : ℕ → ℤ, abs (A.sum (λ a, a * f a)) ≤ 1) :=
sorry

end minValueC_proof_l570_570045


namespace larry_final_channels_l570_570534

def initial_channels : Int := 150
def removed_channels : Int := 20
def replacement_channels : Int := 12
def reduced_channels : Int := 10
def sports_package_channels : Int := 8
def supreme_sports_package_channels : Int := 7

theorem larry_final_channels :
  initial_channels 
  - removed_channels 
  + replacement_channels 
  - reduced_channels 
  + sports_package_channels 
  + supreme_sports_package_channels 
  = 147 := by
  rfl  -- Reflects the direct computation as per the problem

end larry_final_channels_l570_570534


namespace find_smallest_x_satisfying_condition_l570_570792

theorem find_smallest_x_satisfying_condition :
  ∃ x : ℝ, 0 < x ∧ (⌊x^2⌋ - x * ⌊x⌋ = 10) ∧ x = 131 / 11 :=
by
  sorry

end find_smallest_x_satisfying_condition_l570_570792


namespace vector_magnitude_l570_570119

noncomputable def a : ℝ × ℝ := (1, 0)
noncomputable def b : ℝ × ℝ := (| (2:ℝ) * Real.cos (2 * Real.pi / 3) |, | (2:ℝ) * Real.sin (2 * Real.pi / 3) |)

theorem vector_magnitude :
  ‖(2:ℝ) • a + b‖ = 2 :=
by
  sorry

end vector_magnitude_l570_570119


namespace find_nth_letter_l570_570641

def repeating_sequence : List Char := 
  ['A', 'B', 'C', 'D', 'E', 'F', 'E', 'D', 'C', 'B', 'A']

def sequence_length : Nat := 11

theorem find_nth_letter (n : Nat) (h : n = 2023) : 
  repeating_sequence[(n % sequence_length) - 1] = 'B' :=
by
  sorry

end find_nth_letter_l570_570641


namespace range_of_f_l570_570255

def f (x : ℤ) : ℤ := (x - 1)^2 - 1

theorem range_of_f :
  Set.image f {-1, 0, 1, 2, 3} = {-1, 0, 3} :=
by
  sorry

end range_of_f_l570_570255


namespace marble_selection_count_l570_570884

def A : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def B : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

def count_ways : ℕ :=
  B.sum (λ b, (A.product A).count (λ p, p.1 + p.2 = b))

theorem marble_selection_count :
  count_ways = 56 :=
sorry

end marble_selection_count_l570_570884


namespace find_smallest_x_satisfying_condition_l570_570796

theorem find_smallest_x_satisfying_condition :
  ∃ x : ℝ, 0 < x ∧ (⌊x^2⌋ - x * ⌊x⌋ = 10) ∧ x = 131 / 11 :=
by
  sorry

end find_smallest_x_satisfying_condition_l570_570796


namespace minimum_value_of_a_l570_570836

variable {f : ℝ → ℝ}
variable {a : ℝ}

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

def monotonic_function (f : ℝ → ℝ) := ∀ x1 x2 : ℝ, x1 < x2 → f x1 ≤ f x2

theorem minimum_value_of_a
  (h_odd : odd_function f)
  (h_def_pos : ∀ x : ℝ, 0 < x → f x = exp x + a)
  (h_monotonic : monotonic_function f)
  (h_f0 : f 0 = 0) :
  a = -1 := 
sorry

end minimum_value_of_a_l570_570836


namespace find_k_l570_570381

theorem find_k
  (S : ℝ)    -- Distance between the village and city
  (x : ℝ)    -- Speed of the truck in km/h
  (y : ℝ)    -- Speed of the car in km/h
  (H1 : 18 = 0.75 * x - 0.75 * x ^ 2 / (x + y))  -- Condition that truck leaving earlier meets 18 km closer to the city
  (H2 : 24 = x * y / (x + y))      -- Intermediate step from solving the first condition
  : (k = 8) :=    -- We need to show that k = 8
  sorry

end find_k_l570_570381


namespace diagonal_of_prism_12_24_15_l570_570337

def is_diagonal_of_rectangular_prism (length width height d : ℝ) : Prop :=
  d = real.sqrt (length^2 + width^2 + height^2)

theorem diagonal_of_prism_12_24_15 :
  is_diagonal_of_rectangular_prism 12 24 15 (3 * real.sqrt 105) :=
by
  sorry

end diagonal_of_prism_12_24_15_l570_570337


namespace find_first_number_l570_570296

theorem find_first_number (a b : ℕ) (k : ℕ) (h1 : a = 3 * k) (h2 : b = 4 * k) (h3 : Nat.lcm a b = 84) : a = 21 := 
sorry

end find_first_number_l570_570296


namespace division_identity_l570_570640

theorem division_identity
  (x y : ℕ)
  (h1 : x = 7)
  (h2 : y = 2)
  : (x^3 + y^3) / (x^2 - x * y + y^2) = 9 :=
by
  sorry

end division_identity_l570_570640


namespace min_value_expression_l570_570885

theorem min_value_expression (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 3) :
  ∃ (M : ℝ), M = (2 : ℝ) ∧ (∀ x y : ℝ, x > 0 → y > 0 → x + y = 3 → ((y / x) + (3 / (y + 1)) ≥ M)) :=
by
  use 2
  sorry

end min_value_expression_l570_570885


namespace sum_coordinates_of_D_eq_15_l570_570456

def midpoint (A B: ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem sum_coordinates_of_D_eq_15 :
  ∀ (D : ℝ × ℝ), 
    midpoint (5, 4) D = (4, 8) → 
    D.1 + D.2 = 15 :=
by 
  sorry

end sum_coordinates_of_D_eq_15_l570_570456


namespace partition_sum_equal_l570_570946

theorem partition_sum_equal (k : ℕ) (h : k > 0) :
  ∃ (x y : Finset ℕ), 
    (x ∪ y = Finset.range (2^(k+1)))
    ∧ (x ∩ y = ∅)
    ∧ (∀ m : ℕ, m ∈ Finset.range (k+1) → (∑ i in x, i^m) = (∑ i in y, i^m)) :=
sorry

end partition_sum_equal_l570_570946


namespace min_value_of_mu_l570_570528

noncomputable def AB := Real.sqrt 2
noncomputable def AC := Real.sqrt 3
noncomputable def angleBAC : ℝ := 30
variable (P : EuclideanGeometry.Point ℝ)

def mu (P : EuclideanGeometry.Point ℝ) (A B C : EuclideanGeometry.Point ℝ) : ℝ :=
  (EuclideanGeometry.distance P A) * (EuclideanGeometry.distance P B) + 
  (EuclideanGeometry.distance P B) * (EuclideanGeometry.distance P C) + 
  (EuclideanGeometry.distance P C) * (EuclideanGeometry.distance P A)

theorem min_value_of_mu : 
  ∃ (P : EuclideanGeometry.Point ℝ) (A B C : EuclideanGeometry.Point ℝ), 
    (EuclideanGeometry.distance A B = AB) ∧ 
    (EuclideanGeometry.distance A C = AC) ∧ 
    (EuclideanGeometry.angle A B C = angleBAC) ∧ 
    mu P A B C = (Real.sqrt 2 / 2) - (5 / 3) :=
sorry

end min_value_of_mu_l570_570528


namespace parents_without_fulltime_jobs_percentage_l570_570516

theorem parents_without_fulltime_jobs_percentage :
  (let total_parents := 100
       mothers := 60 * total_parents / 100
       fathers := 40 * total_parents / 100
       mothers_with_fulltime_jobs := 5 * mothers / 6
       fathers_with_fulltime_jobs := 3 * fathers / 4
       total_fulltime_jobs := mothers_with_fulltime_jobs + fathers_with_fulltime_jobs
       total_without_fulltime_jobs := total_parents - total_fulltime_jobs
       percentage_without_fulltime_jobs := (total_without_fulltime_jobs / total_parents) * 100
   in percentage_without_fulltime_jobs = 20) :=
by 
  sorry

end parents_without_fulltime_jobs_percentage_l570_570516


namespace smallest_positive_x_l570_570786

theorem smallest_positive_x 
  (x : ℝ) 
  (H : 0 < x) 
  (H_eq : ⌊x^2⌋ - x * ⌊x⌋ = 10) : 
  x = 131 / 11 :=
sorry

end smallest_positive_x_l570_570786


namespace symmetry_center_of_tangent_function_l570_570617

theorem symmetry_center_of_tangent_function :
  ∃ (x : ℝ), 3 * tan (1/2 * x + π/3) = 0 ∧ x = -2 ∗ π / 3 :=
by
  sorry

end symmetry_center_of_tangent_function_l570_570617


namespace original_set_size_l570_570626

theorem original_set_size (n : ℕ) (h : n > 0) : 
  (∃ n, n > 0 ∧ (∀ (a : ℕ), a ∈ {x * 12 |∀ x ∈ {y | y ∈ ℕ} ∧ y ∈ (finset.range n), true} → (finset.sum (finset.image (λ x, x * 12) (finset.range n)) / n) = 84)) :=
begin
  -- Proof goes here
  sorry
end

end original_set_size_l570_570626


namespace smallest_pos_value_correct_l570_570783

noncomputable def smallest_pos_real_number : ℝ :=
  let x := 131 / 11 in
  if x > 0 ∧ (x * x).floor - x * (x.floor) = 10 then x else 0

theorem smallest_pos_value_correct (x : ℝ) (hx : 0 < x ∧ (x * x).floor - x * x.floor = 10) :
  x = 131 / 11 :=
begin
  sorry
end

end smallest_pos_value_correct_l570_570783


namespace phase_shift_of_function_l570_570746

noncomputable def function := 5 * Real.sin (x - Real.pi / 3) + 2 * Real.cos x

theorem phase_shift_of_function :
  ∀ x : ℝ, ∃ phase_shift : ℝ, function = 5 * Real.sin (x - Real.pi / 3) + 2 * Real.cos x ∧ phase_shift = Real.pi / 3 :=
by
  sorry

end phase_shift_of_function_l570_570746


namespace digit_encoding_problem_l570_570639

theorem digit_encoding_problem :
  ∃ (A B : ℕ), 0 ≤ A ∧ A < 10 ∧ 0 ≤ B ∧ B < 10 ∧ 21 * A + B = 111 * B ∧ A = 5 ∧ B = 5 :=
by
  sorry

end digit_encoding_problem_l570_570639


namespace solve_cryptarithm_l570_570999

-- Definitions for digits mapped to letters
def C : ℕ := 9
def H : ℕ := 3
def U : ℕ := 5
def K : ℕ := 4
def T : ℕ := 1
def R : ℕ := 2
def I : ℕ := 0
def G : ℕ := 6
def N : ℕ := 8
def S : ℕ := 7

-- Function to evaluate the cryptarithm sum
def cryptarithm_sum : ℕ :=
  (C*10000 + H*1000 + U*100 + C*10 + K) +
  (T*10000 + R*1000 + I*100 + G*10 + G) +
  (T*10000 + U*1000 + R*100 + N*10 + S)

-- Equation checking the result
def cryptarithm_correct : Prop :=
  cryptarithm_sum = T*100000 + R*10000 + I*1000 + C*100 + K*10 + S

-- The theorem we want to prove
theorem solve_cryptarithm : cryptarithm_correct :=
by
  -- Proof steps would be filled here
  -- but for now, we just acknowledge it is a theorem
  sorry

end solve_cryptarithm_l570_570999


namespace cross_square_field_time_l570_570325

theorem cross_square_field_time 
  (walking_speed_kmh : ℝ) 
  (area_square_m2 : ℝ) 
  (t : ℝ)
  (h_walking_speed : walking_speed_kmh = 6)
  (h_area_square : area_square_m2 = 112.5) :
  t = 9 :=
by
  -- speed in m/s
  let speed_mps := walking_speed_kmh * 1000 / 3600
  -- side length of the square
  let a := Real.sqrt area_square_m2
  -- diagonal length
  let d := a * Real.sqrt 2
  -- we need to convert speed and distance to get the time
  let time := d / speed_mps
  have h_speed_mps : speed_mps = 1.6667 := by sorry
  have h_time : time = t := by sorry
  have h_diagonal : d = 15 := by sorry
  exact h_time.2

end cross_square_field_time_l570_570325


namespace tangent_line_at_1_2_tangent_lines_through_1_1_l570_570861

noncomputable def f (x : ℝ) := x^2 + 1

theorem tangent_line_at_1_2 : ∀ x y : ℝ, (x, y) = (1, 2) → 
  let y' := (f x)' := (λ x, 2 * x)
  (y - 2 = 2 * (x - 1)) ∧ (2 * x - y = 0) :=
by
  intro x y coords_eq
  sorry

theorem tangent_lines_through_1_1 : ∀ x y : ℝ, (x, y) = (1, 1) → 
  let x0 := x 
  let y := y 
  let y' := (f x)' := (λ x, 2 * x)
  (1 - (x0^2 + 1) = 2 * x0 * (1 - x0)) →
  (x0 = 0 ∨ x0 = 2) →
  (y = 1 ∨ 4 * x - y - 3 = 0) :=
by
  intro x y coords_eq
  sorry

end tangent_line_at_1_2_tangent_lines_through_1_1_l570_570861


namespace z_is_real_z_is_pure_imag_l570_570556

noncomputable def z (m : ℝ) : ℂ :=
  complex.log (m^2 - 2 * m - 2) + complex.I * (m^2 + 3 * m + 2)

theorem z_is_real (m : ℝ) : 
  (0 < m^2 - 2 * m - 2) ∧ (m^2 + 3 * m + 2 = 0) ↔ (m = -2 ∨ m = -1) :=
by
  -- Proof omitted
  sorry

theorem z_is_pure_imag (m : ℝ) : 
  (m^2 - 2 * m - 2 = 1) ∧ (¬ (m^2 + 3 * m + 2 = 0)) ↔ (m = 3) :=
by
  -- Proof omitted
  sorry

end z_is_real_z_is_pure_imag_l570_570556


namespace trapezoid_total_area_l570_570343

/-- 
Given a trapezoid with side lengths 4, 6, 8, and 10, where sides 4 and 8 are used as parallel bases, 
prove that the total area of the trapezoid in all possible configurations is 48√2.
-/
theorem trapezoid_total_area : 
  let a := 4
  let b := 8
  let c := 6
  let d := 10
  let h := 4 * Real.sqrt 2
  let Area := (1 / 2) * (a + b) * h
  (Area + Area) = 48 * Real.sqrt 2 :=
by 
  sorry

end trapezoid_total_area_l570_570343


namespace solve_for_x_l570_570995

theorem solve_for_x : ∀ x : ℝ, 3^(2*x + 1) = 1/81 → x = -5/2 :=
by
  intro x
  intro h
  have h1 : 1/81 = 3^(-4) := by
    exact one_div_pow (3 : ℝ) 4
  rw [h1] at h
  have h_exp : 3^(2*x + 1) = 3^(-4) := h
  have eq_exp : 2*x + 1 = -4 := by
    apply eq_of_monotone_of_pow_eq
    apply pow_ne_zero
    norm_num
    exact h_exp
  linarith

end solve_for_x_l570_570995


namespace range_of_a_l570_570047

noncomputable def f (a x : ℝ) : ℝ := a * x ^ 2 + x - 1 + 3 * a

theorem range_of_a (a : ℝ) : (∃ x ∈ set.Icc (-1:ℝ) (1:ℝ), f a x = 0) ↔ (0 ≤ a ∧ a ≤ (1:ℝ) / 2) :=
by sorry

end range_of_a_l570_570047


namespace original_number_input_0_2_l570_570967

theorem original_number_input_0_2 (x : ℝ) (hx : x ≠ 0) (h : (1 / (1 / x - 1) - 1 = -0.75)) : x = 0.2 := 
sorry

end original_number_input_0_2_l570_570967


namespace intersection_x_axis_at_2_l570_570586

def grid :=
  { squares: List (ℝ × ℝ) // squares.length = 6 ∧
  (∀ i ∈ squares, 0 ≤ i.1 ∧ i.1 < 3) ∧
  (∀ i ∈ squares, 0 ≤ i.2 ∧ i.2 < 2) }

def line (p1 p2: ℝ × ℝ) := p2.2 - p1.2 = (p2.1 - p1.1) * 2 

theorem intersection_x_axis_at_2 :
  ∀ g : grid, line (2, 0) (3, 2) → ∃ x : ℝ, x = 2 :=
by
  intro g h
  use 2
  sorry

end intersection_x_axis_at_2_l570_570586


namespace djibo_sister_age_today_l570_570025

variable (djibo_age : ℕ) (sum_ages_five_years_ago : ℕ)

theorem djibo_sister_age_today (h1 : djibo_age = 17)
                               (h2 : sum_ages_five_years_ago = 35) :
  let djibo_age_five_years_ago := djibo_age - 5 in
  let sister_age_five_years_ago := sum_ages_five_years_ago - djibo_age_five_years_ago in
  sister_age_five_years_ago + 5 = 28 :=
by 
  -- Proof goes here
  sorry

end djibo_sister_age_today_l570_570025


namespace product_of_roots_l570_570404

theorem product_of_roots :
  let p1 := (3 : ℝ) * X^4 + (2 : ℝ) * X^3 + (-9 : ℝ) * X + (15 : ℝ)
  let p2 := (4 : ℝ) * X^2 + (-16 : ℝ) * X + (14 : ℝ)
  (roots (p1 * p2)).prod = (35 / 2 : ℝ) := 
by
  sorry

end product_of_roots_l570_570404


namespace compare_exponents_l570_570091

noncomputable def a : ℝ := 0.8 ^ 5.2
noncomputable def b : ℝ := 0.8 ^ 5.5
noncomputable def c : ℝ := 5.2 ^ 0.1

theorem compare_exponents (a b c : ℝ) (h1 : a = 0.8 ^ 5.2) (h2 : b = 0.8 ^ 5.5) (h3 : c = 5.2 ^ 0.1) :
  b < a ∧ a < c := sorry

end compare_exponents_l570_570091


namespace pos_int_fraction_l570_570498

theorem pos_int_fraction (p : ℕ) (hp : p > 0) :
  (∃ k : ℕ, (4 * p + 40) = k * (3 * p - 7)) ↔ p ∈ {5, 8, 18, 50} := by
sorry

end pos_int_fraction_l570_570498


namespace intersection_A_B_l570_570830

open Set

def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 1)}
def B : Set ℝ := {x : ℝ | x^2 + 2 * x - 3 ≥ 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | 1 < x} :=
by
  sorry

end intersection_A_B_l570_570830


namespace smallest_pos_value_correct_l570_570782

noncomputable def smallest_pos_real_number : ℝ :=
  let x := 131 / 11 in
  if x > 0 ∧ (x * x).floor - x * (x.floor) = 10 then x else 0

theorem smallest_pos_value_correct (x : ℝ) (hx : 0 < x ∧ (x * x).floor - x * x.floor = 10) :
  x = 131 / 11 :=
begin
  sorry
end

end smallest_pos_value_correct_l570_570782


namespace minimize_quadratic_l570_570287

theorem minimize_quadratic : ∃ x : ℝ, x = 6 ∧ ∀ y : ℝ, (y - 6)^2 ≥ (6 - 6)^2 := by
  sorry

end minimize_quadratic_l570_570287


namespace no_n_make_g_divisible_by_5_l570_570054

def g (n : ℕ) : ℕ := 2 + 5 * n + n^2 + 5 * n^3 + 2 * n^4

theorem no_n_make_g_divisible_by_5 :
  ¬ ∃ n, 2 ≤ n ∧ n ≤ 100 ∧ g(n) % 5 = 0 :=
by
  sorry

end no_n_make_g_divisible_by_5_l570_570054


namespace function_decomposition_l570_570353

open Real

noncomputable def f (x : ℝ) : ℝ := log (10^x + 1)
noncomputable def g (x : ℝ) : ℝ := x / 2
noncomputable def h (x : ℝ) : ℝ := log (10^x + 1) - x / 2

theorem function_decomposition :
  ∀ x : ℝ, f x = g x + h x ∧ (∀ x, g (-x) = -g x) ∧ (∀ x, h (-x) = h x) :=
by
  intro x
  sorry

end function_decomposition_l570_570353


namespace liquid_level_rise_ratio_l570_570277

theorem liquid_level_rise_ratio
  (r1 r2 : ℝ) (h1 h2 : ℝ) (marble_radius : ℝ)
  (initial_height_equal : r1 = 4 ∧ r2 = 8 ∧ marble_radius = 1)
  (initial_volumes_equal : (1 / 3) * real.pi * (r1^2) * h1 = (1 / 3) * real.pi * (r2^2) * h2) :
  ((h1 + marble_radius^3 * (3 / (4 * real.pi * r1^2)) - h1) / (h2 + (marble_radius^3) * (3 / (4 * real.pi * r2^2)) - h2)) = 4 := 
sorry

end liquid_level_rise_ratio_l570_570277


namespace f_prime_at_pi_over_six_l570_570897

noncomputable def f (f'_0 : ℝ) (x : ℝ) : ℝ := (1/2)*x^2 + 2*f'_0*(Real.cos x) + x

theorem f_prime_at_pi_over_six (f'_0 : ℝ) (h : f'_0 = 1) :
  (deriv (f f'_0)) (Real.pi / 6) = Real.pi / 6 := by
  sorry

end f_prime_at_pi_over_six_l570_570897


namespace range_of_a_l570_570062

def f (x : ℝ) : ℝ := 2^(|x|) + x^2

theorem range_of_a (a : ℝ) (h : f (a-1) ≤ 3) : 0 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l570_570062


namespace range_of_a_l570_570434

noncomputable def f : ℝ → ℝ
| x := if x ≤ 0 then -x^2 + 4 * x else Real.log (x + 1)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, abs (f x) ≥ a * x - 1) → -6 ≤ a ∧ a ≤ 0 :=
by
  sorry

end range_of_a_l570_570434


namespace december_19th_day_l570_570738

theorem december_19th_day (december_has_31_days : true)
  (december_1st_is_monday : true)
  (day_of_week : ℕ → ℕ) :
  day_of_week 19 = 5 :=
sorry

end december_19th_day_l570_570738


namespace number_of_5_letter_words_with_at_least_one_consonant_equals_7744_l570_570485

def is_consonant (c : Char) : Prop :=
  c = 'B' ∨ c = 'C' ∨ c = 'D' ∨ c = 'F'

def count_5_letter_words_with_at_least_one_consonant : Nat :=
  let total_words := 6 ^ 5
  let vowel_words := 2 ^ 5
  total_words - vowel_words

theorem number_of_5_letter_words_with_at_least_one_consonant_equals_7744 :
  count_5_letter_words_with_at_least_one_consonant = 7744 :=
by
  sorry

end number_of_5_letter_words_with_at_least_one_consonant_equals_7744_l570_570485


namespace percentage_big_bottles_sold_l570_570702

-- Definitions of conditions
def total_small_bottles : ℕ := 6000
def total_big_bottles : ℕ := 14000
def small_bottles_sold_percentage : ℕ := 20
def total_bottles_remaining : ℕ := 15580

-- Theorem statement
theorem percentage_big_bottles_sold : 
  let small_bottles_sold := (small_bottles_sold_percentage * total_small_bottles) / 100
  let small_bottles_remaining := total_small_bottles - small_bottles_sold
  let big_bottles_remaining := total_bottles_remaining - small_bottles_remaining
  let big_bottles_sold := total_big_bottles - big_bottles_remaining
  (100 * big_bottles_sold) / total_big_bottles = 23 := 
by
  sorry

end percentage_big_bottles_sold_l570_570702


namespace probability_no_translator_no_driver_l570_570151

theorem probability_no_translator_no_driver :
  let people := ["Zhang", "Zhao", "Li", "Luo", "Wang"]
  let roles := ["translator", "tour guide", "etiquette", "driver"]
  let total_ways := Nat.descFactorial 5 4
  let valid_ways := 72
  let probability := valid_ways.to_rat / total_ways.to_rat
  probability = 3 / 5 := 
by
  let people := ["Zhang", "Zhao", "Li", "Luo", "Wang"]
  let roles := ["translator", "tour guide", "etiquette", "driver"]
  have total_ways := Nat.descFactorial 5 4
  have valid_ways := 72
  have probability := valid_ways.to_rat / total_ways.to_rat
  show probability = 3 / 5 from sorry

end probability_no_translator_no_driver_l570_570151


namespace product_of_intersection_points_l570_570645

-- Define the two circles in the plane
def circle1 (x y : ℝ) : Prop := x^2 - 4*x + y^2 - 8*y + 16 = 0
def circle2 (x y : ℝ) : Prop := x^2 - 6*x + y^2 - 8*y + 21 = 0

-- Define the intersection points property
def are_intersection_points (x y : ℝ) : Prop := circle1 x y ∧ circle2 x y

-- The theorem to be proved
theorem product_of_intersection_points : ∃ x y : ℝ, are_intersection_points x y ∧ x * y = 12 := 
by
  sorry

end product_of_intersection_points_l570_570645


namespace Djibo_sister_age_l570_570023

variable (d s : ℕ)
variable (h1 : d = 17)
variable (h2 : d - 5 + (s - 5) = 35)

theorem Djibo_sister_age : s = 28 :=
by sorry

end Djibo_sister_age_l570_570023


namespace slope_to_y_intercept_ratio_l570_570522

theorem slope_to_y_intercept_ratio (m b : ℝ) (c : ℝ) (h1 : m = c * b) (h2 : 2 * m + b = 0) : c = -1 / 2 :=
by sorry

end slope_to_y_intercept_ratio_l570_570522


namespace age_when_Billy_born_l570_570129

-- Definitions based on conditions
def current_age_I := 4 * 4
def current_age_Billy := 4
def age_difference := current_age_I - current_age_Billy

-- Statement to prove
theorem age_when_Billy_born : age_difference = 12 :=
by
  -- Expose the calculation steps
  calc
    age_difference
    = 4 * 4 - 4 : by rw [current_age_I, current_age_Billy]
    ... = 16 - 4 : by norm_num
    ... = 12 : by norm_num

end age_when_Billy_born_l570_570129


namespace five_letter_words_with_consonant_l570_570483

theorem five_letter_words_with_consonant :
  let letters := { 'A', 'B', 'C', 'D', 'E', 'F' } in
  let consonants := { 'B', 'C', 'D', 'F' } in
  let vowels := { 'A', 'E' } in
  let total_5_letter_words := 6^5 in
  let total_vowel_only_words := 2^5 in
  total_5_letter_words - total_vowel_only_words = 7744 :=
by
  sorry

end five_letter_words_with_consonant_l570_570483


namespace marcy_total_spears_l570_570962

-- Define the conditions
def can_make_spears_from_sapling (spears_per_sapling : ℕ) (saplings : ℕ) : ℕ :=
  spears_per_sapling * saplings

def can_make_spears_from_log (spears_per_log : ℕ) (logs : ℕ) : ℕ :=
  spears_per_log * logs

-- Number of spears Marcy can make from 6 saplings and 1 log
def total_spears (spears_per_sapling : ℕ) (saplings : ℕ) (spears_per_log : ℕ) (logs : ℕ) : ℕ :=
  can_make_spears_from_sapling spears_per_sapling saplings + can_make_spears_from_log spears_per_log logs

-- Given conditions
theorem marcy_total_spears (saplings : ℕ) (logs : ℕ) : 
  total_spears 3 6 9 1 = 27 :=
by
  sorry

end marcy_total_spears_l570_570962


namespace number_of_triangles_with_positive_area_l570_570491

theorem number_of_triangles_with_positive_area :
  let vertices := finset.range 6.succ.product (finset.range 6.succ),
    total_points := 36,
    total_triangles := nat.choose total_points 3,
    collinear_triangles := 120 + 120 + 40 + 40 + 24 + 8,
    triangles_with_positive_area := total_triangles - collinear_triangles
  in triangles_with_positive_area = 6788 := by sorry

end number_of_triangles_with_positive_area_l570_570491


namespace rose_share_correct_l570_570989

-- Define the conditions
def purity_share (P : ℝ) : ℝ := P
def sheila_share (P : ℝ) : ℝ := 5 * P
def rose_share (P : ℝ) : ℝ := 3 * P
def total_rent := 5400

-- The theorem to be proven
theorem rose_share_correct (P : ℝ) (h : purity_share P + sheila_share P + rose_share P = total_rent) : 
  rose_share P = 1800 :=
  sorry

end rose_share_correct_l570_570989


namespace jame_old_job_hours_per_week_l570_570922

noncomputable def new_hourly_wage : ℤ := 20
noncomputable def new_weekly_hours : ℕ := 40
noncomputable def old_hourly_wage : ℤ := 16
noncomputable def yearly_difference : ℤ := 20800
noncomputable def weeks_per_year : ℕ := 52

theorem jame_old_job_hours_per_week :
  let new_annual_income := new_hourly_wage * new_weekly_hours * weeks_per_year,
      old_annual_income := new_annual_income - yearly_difference,
      old_weekly_income := old_annual_income / weeks_per_year in
  old_weekly_income / old_hourly_wage = 25 := sorry

end jame_old_job_hours_per_week_l570_570922


namespace smallest_positive_x_l570_570788

theorem smallest_positive_x 
  (x : ℝ) 
  (H : 0 < x) 
  (H_eq : ⌊x^2⌋ - x * ⌊x⌋ = 10) : 
  x = 131 / 11 :=
sorry

end smallest_positive_x_l570_570788


namespace double_burger_cost_l570_570723

theorem double_burger_cost (total_spent : ℝ) (total_hamburgers : ℕ) (double_burgers: ℕ) (single_burger_cost : ℝ) (H1 : total_spent = 70.50) (H2 : total_hamburgers = 50) (H3 : double_burgers = 41) (H4 : single_burger_cost = 1.00) :
  let single_burgers := total_hamburgers - double_burgers in 
  let x := (total_spent - (single_burgers * single_burger_cost)) / double_burgers in 
  x = 1.50 :=
by
  let single_burgers := total_hamburgers - double_burgers
  let x := (total_spent - (single_burgers * single_burger_cost)) / double_burgers
  sorry

end double_burger_cost_l570_570723


namespace proportional_ratios_l570_570497

variables (a b c d : ℚ)
hypothesis h1 : a / b = 3 / 4
hypothesis h2 : c / d = 3 / 4

theorem proportional_ratios (h1 : a / b = 3 / 4) (h2 : c / d = 3 / 4) : (a + c) / (b + d) = 3 / 4 :=
by
  sorry

end proportional_ratios_l570_570497


namespace tim_kittens_left_l570_570633

theorem tim_kittens_left
  (initial_kittens : ℕ)
  (kittens_to_jessica : ℕ)
  (kittens_to_sara : ℕ)
  (total_given_away : ℕ)
  (kittens_left : ℕ) :
  initial_kittens = 18 →
  kittens_to_jessica = 3 →
  kittens_to_sara = 6 →
  total_given_away = kittens_to_jessica + kittens_to_sara →
  kittens_left = initial_kittens - total_given_away →
  kittens_left = 9 :=
by
  intros h_initial h_jessica h_sara h_total h_left
  rw [h_initial, h_jessica, h_sara, h_total, h_left]
  norm_num
  sorry

end tim_kittens_left_l570_570633


namespace halfway_between_fractions_l570_570022

theorem halfway_between_fractions : 
  (2:ℚ) / 9 + (5 / 12) / 2 = 23 / 72 := 
sorry

end halfway_between_fractions_l570_570022


namespace circle_exists_with_exactly_n_integer_points_l570_570218

-- Define the context for the problem, including the center of the circle
def center : ℝ × ℝ := (Real.sqrt 2, 1 / 3)

-- Main statement of the theorem to be proved
theorem circle_exists_with_exactly_n_integer_points (n : ℤ) : ∃ R : ℝ, 
  (∃ (count : ℕ), count = n ∧ 
    ∃ (points : finset (ℤ × ℤ)),
      (∀ p ∈ points, (p.fst - center.fst)^2 + (p.snd - center.snd)^2 < R^2) 
      ∧ points.card = count) :=
sorry

end circle_exists_with_exactly_n_integer_points_l570_570218


namespace find_smallest_x_satisfying_condition_l570_570793

theorem find_smallest_x_satisfying_condition :
  ∃ x : ℝ, 0 < x ∧ (⌊x^2⌋ - x * ⌊x⌋ = 10) ∧ x = 131 / 11 :=
by
  sorry

end find_smallest_x_satisfying_condition_l570_570793


namespace find_k_value_l570_570864

-- Define the lines l1 and l2 with given conditions
def line1 (x y : ℝ) : Prop := x + 3 * y - 7 = 0
def line2 (k x y : ℝ) : Prop := k * x - y - 2 = 0

-- Define the condition for the quadrilateral to be circumscribed by a circle
def is_circumscribed (k : ℝ) : Prop :=
  ∃ (x y : ℝ), line1 x y ∧ line2 k x y ∧ 0 < x ∧ 0 < y

theorem find_k_value (k : ℝ) : is_circumscribed k → k = 3 := 
sorry

end find_k_value_l570_570864


namespace largest_even_two_digit_largest_odd_two_digit_l570_570019

-- Define conditions
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Theorem statements
theorem largest_even_two_digit : ∃ n, is_two_digit n ∧ is_even n ∧ ∀ m, is_two_digit m ∧ is_even m → m ≤ n := 
sorry

theorem largest_odd_two_digit : ∃ n, is_two_digit n ∧ is_odd n ∧ ∀ m, is_two_digit m ∧ is_odd m → m ≤ n := 
sorry

end largest_even_two_digit_largest_odd_two_digit_l570_570019


namespace inverse_110_mod_667_l570_570007

theorem inverse_110_mod_667 :
  (∃ (a b c : ℕ), a = 65 ∧ b = 156 ∧ c = 169 ∧ c^2 = a^2 + b^2) →
  (∃ n : ℕ, 110 * n % 667 = 1 ∧ 0 ≤ n ∧ n < 667 ∧ n = 608) :=
by
  sorry

end inverse_110_mod_667_l570_570007


namespace valid_triples_l570_570015

theorem valid_triples (a b c : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
  (h₄ : nat.gcd a 20 = b) (h₅ : nat.gcd b 15 = c) (h₆ : nat.gcd a c = 5) :
  ∃ k : ℕ, k > 0 ∧ (a, b, c) = (5 * k, 5, 5) ∨ (a, b, c) = (5 * k, 10, 5) ∨ (a, b, c) = (5 * k, 20, 5) :=
sorry

end valid_triples_l570_570015


namespace single_burger_cost_l570_570001

theorem single_burger_cost
  (total_cost : ℝ)
  (total_hamburgers : ℕ)
  (double_burgers : ℕ)
  (cost_double_burger : ℝ)
  (remaining_cost : ℝ)
  (single_burgers : ℕ)
  (cost_single_burger : ℝ) :
  total_cost = 64.50 ∧
  total_hamburgers = 50 ∧
  double_burgers = 29 ∧
  cost_double_burger = 1.50 ∧
  remaining_cost = total_cost - (double_burgers * cost_double_burger) ∧
  single_burgers = total_hamburgers - double_burgers ∧
  cost_single_burger = remaining_cost / single_burgers →
  cost_single_burger = 1.00 :=
by
  sorry

end single_burger_cost_l570_570001


namespace cos_A_value_triangle_area_l570_570834

theorem cos_A_value {a b c : ℝ} (A B C : ℝ) (h1 : angle_acute A) (h2 : 2 * a * Real.cos A = c * Real.cos B + b * Real.cos C) :
  Real.cos A = 1 / 2 := by
  sorry

theorem triangle_area {a b c : ℝ} (A B C : ℝ) (h1 : angle_acute A) (h2 : 2 * a * Real.cos A = c * Real.cos B + b * Real.cos C) (h3 : b^2 + c^2 = 4) :
  area_of_triangle a b c A = sqrt(3) / 4 := by
  sorry

end cos_A_value_triangle_area_l570_570834


namespace num_ways_to_sum_1995_l570_570150

def sum_of_consecutive_positive_odd_numbers_ways (n a : ℕ) : Prop :=
  n * (a + n - 1) = 1995 ∧ ∀ k, k < n → a + 2 * k > 0

theorem num_ways_to_sum_1995 :
  ∃ count : ℕ, count = 7 ∧ count = (Fintype.card {p : ℕ × ℕ // sum_of_consecutive_positive_odd_numbers_ways p.1 p.2}) :=
by sorry

end num_ways_to_sum_1995_l570_570150


namespace ratio_of_profits_l570_570668

-- Define the investments of P and Q
def p_investment : ℕ := 30000
def q_investment : ℕ := 45000

-- State that the profit share is proportional to the investment.
-- We need to prove that the ratio p_investment / q_investment is 2 : 3
theorem ratio_of_profits (hp : p_investment = 30000) (hq : q_investment = 45000) : 
  p_investment / Nat.gcd p_investment q_investment = 2 ∧ q_investment / Nat.gcd p_investment q_investment = 3 :=
by
  have gcd_val : Nat.gcd p_investment q_investment = 15000 := sorry
  rw [hp, hq, gcd_val]
  split
  { rw [Nat.div_eq_of_eq_mul_left (Nat.gcd_pos_of_pos_left 30000 45000) (by norm_num : 30000 = 2 * 15000)]
    norm_num }
  { rw [Nat.div_eq_of_eq_mul_left (Nat.gcd_pos_of_pos_left 30000 45000) (by norm_num : 45000 = 3 * 15000)]
    norm_num }
  --The gcd_val intermediate result is required, but we put it as sorry.
  sorry

end ratio_of_profits_l570_570668


namespace OP_perpendicular_from_M_l570_570873

-- Definitions based on conditions
variables {O A B P M : Point}
variable {outer_circle inner_circle : Circle}
variable [is_center : Center O outer_circle inner_circle]
variable [H_A_on_outer : OnCircle A outer_circle]
variable [H_B_on_outer : OnCircle B outer_circle]
variable [tangent_inner_A_C : TangentFrom A inner_circle]
variable [tangent_inner_B_D : TangentFrom B inner_circle]
variable [tangents_meet_at_M : MeetTangent tangent_inner_A_C tangent_inner_B_D M]
variable [non_perpendicular_M : ¬OnLineThroughPerpendicularToChord OM A B M]
variable [tangent_outer_A_P : TangentFrom A outer_circle]
variable [tangent_outer_B_P : TangentFrom B outer_circle]
variable [tangents_intersect_at_P : MeetTangent tangent_outer_A_P tangent_outer_B_P P]

-- Goal
theorem OP_perpendicular_from_M : angle O P M = π / 2 :=
sorry

end OP_perpendicular_from_M_l570_570873


namespace train_speed_l570_570295

def train_length : ℝ := 360 -- length of the train in meters
def crossing_time : ℝ := 6 -- time taken to cross the man in seconds

theorem train_speed (train_length crossing_time : ℝ) : 
  (train_length = 360) → (crossing_time = 6) → (train_length / crossing_time = 60) :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end train_speed_l570_570295


namespace penny_initial_money_l570_570569

theorem penny_initial_money
    (pairs_of_socks : ℕ)
    (cost_per_pair : ℝ)
    (number_of_pairs : ℕ)
    (cost_of_hat : ℝ)
    (money_left : ℝ)
    (initial_money : ℝ)
    (H1 : pairs_of_socks = 4)
    (H2 : cost_per_pair = 2)
    (H3 : number_of_pairs = pairs_of_socks)
    (H4 : cost_of_hat = 7)
    (H5 : money_left = 5)
    (H6 : initial_money = (number_of_pairs * cost_per_pair) + cost_of_hat + money_left) : initial_money = 20 :=
sorry

end penny_initial_money_l570_570569


namespace number_of_people_and_price_l570_570913

theorem number_of_people_and_price 
  (x y : ℤ) 
  (h1 : 8 * x - y = 3) 
  (h2 : y - 7 * x = 4) : 
  x = 7 ∧ y = 53 :=
by
  sorry

end number_of_people_and_price_l570_570913


namespace inequality_solution_set_is_empty_l570_570899

theorem inequality_solution_set_is_empty (a : ℝ) :
  (∀ x : ℝ, (a - 1) * x^2 + 2 * (a - 1) * x - 4 < 0) ↔ a ∈ Ioc (-3 : ℝ) 1 :=
sorry

end inequality_solution_set_is_empty_l570_570899


namespace factorize_quadratic_l570_570035

variable (x : ℝ)

theorem factorize_quadratic : 2 * x^2 + 4 * x - 6 = 2 * (x - 1) * (x + 3) :=
by
  sorry

end factorize_quadratic_l570_570035


namespace log_sum_eq_neg_one_l570_570051

theorem log_sum_eq_neg_one :
  log 4 (sin (π / 12)) + log 4 (cos (π / 12)) = -1 := sorry

end log_sum_eq_neg_one_l570_570051


namespace concurrency_of_midpoint_incenter_lines_l570_570576

variables (A B C D K : Point)
variables (ω : Circle)
variables (M1 M2 M3 M4 : Point)
variables (I1 I2 I3 I4 : Point)
variables [inscribed_quadrilateral A B C D ω]
variables [arc_midpoints M1 M2 M3 M4 ω A B C D]
variables [incenters I1 I2 I3 I4 A B C D K]

theorem concurrency_of_midpoint_incenter_lines :
  lines_concurrent M1 I1 M2 I2 M3 I3 M4 I4 :=
sorry

end concurrency_of_midpoint_incenter_lines_l570_570576


namespace unique_perpendicular_in_plane_l570_570663

-- Definitions and the main theorem statement

variables {α : Type*} [EuclideanGeometry α]

theorem unique_perpendicular_in_plane (l : Line α) (P : Point α) :
  ∃! m : Line α, m.perpendicular_to l ∧ P ∈ m := 
sorry

end unique_perpendicular_in_plane_l570_570663


namespace motorcycles_in_anytown_l570_570327

variable (t s m : ℕ) -- t: number of trucks, s: number of sedans, m: number of motorcycles
variable (r_trucks r_sedans r_motorcycles : ℕ) -- r_trucks : truck ratio, r_sedans : sedan ratio, r_motorcycles : motorcycle ratio
variable (n_sedans : ℕ) -- n_sedans: number of sedans

theorem motorcycles_in_anytown
  (h1 : r_trucks = 3) -- ratio of trucks
  (h2 : r_sedans = 7) -- ratio of sedans
  (h3 : r_motorcycles = 2) -- ratio of motorcycles
  (h4 : s = 9100) -- number of sedans
  (h5 : s = (r_sedans * n_sedans)) -- relationship between sedans and parts
  (h6 : t = (r_trucks * n_sedans)) -- relationship between trucks and parts
  (h7 : m = (r_motorcycles * n_sedans)) -- relationship between motorcycles and parts
  : m = 2600 := by
    sorry

end motorcycles_in_anytown_l570_570327


namespace find_solution_l570_570040

theorem find_solution (x : ℝ) (h : (5 + x / 3)^(1/3) = -4) : x = -207 :=
sorry

end find_solution_l570_570040


namespace log_sum_geometric_sequence_l570_570506

theorem log_sum_geometric_sequence (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
    (h_geom_seq : a 10 * a 11 + a 9 * a 12 = 2 * Real.exp 5) :
    Real.log (a 1) + Real.log (a 2) + ⋯ + Real.log (a 20) = 50 :=
by
  sorry

end log_sum_geometric_sequence_l570_570506


namespace aerith_seat_l570_570752

-- Let the seats be numbered 1 through 8
-- Assigned seats for Aerith, Bob, Chebyshev, Descartes, Euler, Fermat, Gauss, and Hilbert
variables (a b c d e f g h : ℕ)

-- Define the conditions described in the problem
axiom Bob_assigned : b = 1
axiom Chebyshev_assigned : c = g + 2
axiom Descartes_assigned : d = f - 1
axiom Euler_assigned : e = h - 4
axiom Fermat_assigned : f = d + 5
axiom Gauss_assigned : g = e + 1
axiom Hilbert_assigned : h = a - 3

-- Provide the proof statement to find whose seat Aerith sits
theorem aerith_seat : a = c := sorry

end aerith_seat_l570_570752


namespace arithmetic_sequence_count_l570_570424

-- Definitions based on the conditions and question
def sequence_count : ℕ := 314 -- The number of common differences for 315-term sequences
def set_size : ℕ := 2014     -- The maximum number in the set {1, 2, 3, ..., 2014}
def min_seq_length : ℕ := 315 -- The length of the arithmetic sequence

-- Lean 4 statement to verify the number of ways to form the required sequence
theorem arithmetic_sequence_count :
  ∃ (ways : ℕ), ways = 5490 ∧
  (∀ (d : ℕ), 1 ≤ d ∧ d ≤ 6 →
  (set_size - (sequence_count * d - 1)) > 0 → 
  ways = (
    if d = 1 then set_size - sequence_count + 1 else
    if d = 2 then set_size - (sequence_count * 2 - 1) + 1 else
    if d = 3 then set_size - (sequence_count * 3 - 1) + 1 else
    if d = 4 then set_size - (sequence_count * 4 - 1) + 1 else
    if d = 5 then set_size - (sequence_count * 5 - 1) + 1 else
    set_size - (sequence_count * 6 - 1) + 1) - 2
  ) :=
sorry

end arithmetic_sequence_count_l570_570424


namespace compound_ca_atoms_l570_570318

theorem compound_ca_atoms 
  (molecular_weight : ℝ := 74)
  (atomic_weight_Ca : ℝ := 40.08)
  (atomic_weight_O : ℝ := 16.00)
  (atomic_weight_H : ℝ := 1.008) 
  (num_O_atoms : ℝ := 2)
  (num_H_atoms : ℝ := 2)
  : 
  (molecular_weight - (num_O_atoms * atomic_weight_O + num_H_atoms * atomic_weight_H)) / atomic_weight_Ca ≈ 1 :=
by
  sorry

end compound_ca_atoms_l570_570318


namespace percentage_increase_l570_570666

theorem percentage_increase :
  ∀ (x : ℝ), (x = 1.325 * 143) → x = 189.475 :=
by
  intro x h
  rw h
  norm_num

end percentage_increase_l570_570666


namespace inscribed_circle_radius_DE_F_EF_l570_570647

theorem inscribed_circle_radius_DE_F_EF (DE DF EF : ℝ) (hDE : DE = 26) (hDF : DF = 15) (hEF : EF = 17) : 
    let s := (DE + DF + EF) / 2 in
    let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)) in
    let r := K / s in
    r = 3 * Real.sqrt 2 :=
by 
  sorry

end inscribed_circle_radius_DE_F_EF_l570_570647


namespace license_plate_combinations_l570_570719

-- Definitions for the conditions based on the problem statement.
def number_of_license_plate_combinations : ℕ := 
  let letters := 26
  let choose_two_from_25 := Nat.choose 25 2
  let choose_two_positions_from_4 := Nat.choose 4 2
  let two_factorial := 2.factorial
  let digit_combinations := 10 * 10
  letters * choose_two_from_25 * choose_two_positions_from_4 * two_factorial * digit_combinations

-- The theorem to prove the number of different license plate combinations
theorem license_plate_combinations : number_of_license_plate_combinations = 936000 :=
by
  -- Here is the place for the proof.
  sorry

end license_plate_combinations_l570_570719


namespace halfway_between_l570_570743

theorem halfway_between (a b : ℚ) (h₁ : a = 1/8) (h₂ : b = 1/3) : (a + b) / 2 = 11 / 48 := 
by
  sorry

end halfway_between_l570_570743


namespace optionA_incorrect_optionB_incorrect_optionC_incorrect_optionD_correct_l570_570293

theorem optionA_incorrect (a x : ℝ) : 3 * a * x^2 - 6 * a * x ≠ 3 * (a * x^2 - 2 * a * x) :=
by sorry

theorem optionB_incorrect (a x : ℝ) : (x + a) * (x - a) ≠ x^2 - a^2 :=
by sorry

theorem optionC_incorrect (a b : ℝ) : a^2 + 2 * a * b - 4 * b^2 ≠ (a + 2 * b)^2 :=
by sorry

theorem optionD_correct (a x : ℝ) : -a * x^2 + 2 * a * x - a = -a * (x - 1)^2 :=
by sorry

end optionA_incorrect_optionB_incorrect_optionC_incorrect_optionD_correct_l570_570293


namespace fractional_part_condition_l570_570039

theorem fractional_part_condition {x : ℝ} (h0 : 0 ≤ x) (h1 : x < 1) (h2 : (x + 1)^3 - x^3 ∈ {0, 1, 2, 3, 4, 5, 6}) :
  ({(x + 1) ^ 3} = x ^ 3) :=
by
  sorry

end fractional_part_condition_l570_570039


namespace triangle_angle_classification_l570_570216

theorem triangle_angle_classification (a b c : ℝ) :
  let p := (a + b + c) / 2 in
  let S := Real.sqrt (p * (p - a) * (p - b) * (p - c)) in
  let R := (a * b * c) / (4 * S) in
  let r := S / p in
  p > (a * b * c) / (2 * S) + (S / p) ∨
  p = (a * b * c) / (2 * S) + (S / p) ∨
  p < (a * b * c) / (2 * S) + (S / p) :=
sorry

end triangle_angle_classification_l570_570216


namespace time_for_q_to_complete_work_alone_l570_570667

theorem time_for_q_to_complete_work_alone (P Q : ℝ) (h1 : (1 / P) + (1 / Q) = 1 / 40) (h2 : (20 / P) + (12 / Q) = 1) : Q = 64 / 3 :=
by
  sorry

end time_for_q_to_complete_work_alone_l570_570667


namespace djibo_sister_age_today_l570_570026

variable (djibo_age : ℕ) (sum_ages_five_years_ago : ℕ)

theorem djibo_sister_age_today (h1 : djibo_age = 17)
                               (h2 : sum_ages_five_years_ago = 35) :
  let djibo_age_five_years_ago := djibo_age - 5 in
  let sister_age_five_years_ago := sum_ages_five_years_ago - djibo_age_five_years_ago in
  sister_age_five_years_ago + 5 = 28 :=
by 
  -- Proof goes here
  sorry

end djibo_sister_age_today_l570_570026


namespace crayons_at_birthday_l570_570568

theorem crayons_at_birthday
  (E : ℕ) -- number of erasers Paul got for his birthday
  (C_end : ℕ) -- number of crayons Paul had left at the end of the school year
  (no_erders_lost : E = 406) -- Paul did not lose any erasers
  (eraser_crayon_diff : E = C_end + 70) -- Paul had 70 more erasers than crayons left
  (E_equal : E = 406) --number of erasers at the end is equal to erasers Paul got for his birthday
  :
  ∃ C : ℕ, C = 406 := 
begin
  sorry 
end

end crayons_at_birthday_l570_570568


namespace equal_chord_segments_l570_570031

theorem equal_chord_segments 
  (a x y : ℝ) 
  (AM CM : ℝ → ℝ → Prop) 
  (AB CD : ℝ → Prop)
  (intersect_chords_theorem : AM x (a - x) = CM y (a - y)) :
  x = y ∨ x = a - y :=
by
  sorry

end equal_chord_segments_l570_570031


namespace stans_average_speed_l570_570592

theorem stans_average_speed:
  let distance1 := 350
  let distance2 := 400
  let time1 := 5 + 40 / 60
  let time2 := 7
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  average_speed = total_distance / total_time :=
by 
  have distance1 := 350
  have distance2 := 400
  have time1 := 5 + 40 / 60
  have time2 := 7
  have total_distance := distance1 + distance2
  have total_time := time1 + time2
  have average_speed := total_distance / total_time
  show average_speed = 59.19
  sorry

end stans_average_speed_l570_570592


namespace prove_min_sum_cubes_l570_570450

noncomputable def min_sum_cubes (n : ℕ) (hn : n ≥ 3)
  (a : Fin n → ℝ) (hmin : ∀ i j : Fin n, i < j → |a i - a j| ≥ 1) :
  Prop :=
  if odd n then
    ∑ k in Finset.range n, |a k| ^ 3 ≥ (1 / 32) * (n^2 - 1)^2
  else
    ∑ k in Finset.range n, |a k| ^ 3 ≥ (1 / 32) * n^2 * (n^2 - 2)

theorem prove_min_sum_cubes :
  ∀ (n : ℕ) (hn : n ≥ 3) (a : Fin n → ℝ),
    (∀ i j : Fin n, i < j → |a i - a j| ≥ 1) →
    min_sum_cubes n hn a sorry :=
sorry

end prove_min_sum_cubes_l570_570450


namespace expansion_largest_binomial_coefficient_expansion_largest_coefficient_l570_570096

theorem expansion_largest_binomial_coefficient {n : ℕ} (hn : n ≤ 10) 
  (arith_seq : Finset.sum (Finset.range 3) (λ k, Nat.choose n (k + 1) * 5^(n-k-1)) = 2 * (Nat.choose n 2 * 5^(n-2))) :
  ((n = 7) ∧ ((21875 = Nat.choose 7 3 * 5^4) ∧ (4375 = Nat.choose 7 4 * 5^3))) :=
begin
  sorry
end

theorem expansion_largest_coefficient {n : ℕ} (hn : n = 7) :
  109375 = Nat.choose 7 1 * 5^6 :=
begin
  sorry
end

end expansion_largest_binomial_coefficient_expansion_largest_coefficient_l570_570096


namespace winning_candidate_percentage_l570_570517

theorem winning_candidate_percentage (P: ℝ) (majority diff votes totalVotes : ℝ)
    (h1 : majority = 184)
    (h2 : totalVotes = 460)
    (h3 : diff = P * totalVotes / 100 - (100 - P) * totalVotes / 100)
    (h4 : majority = diff) : P = 70 :=
by
  sorry

end winning_candidate_percentage_l570_570517


namespace maximum_value_ab_l570_570816

noncomputable def proof_problem (a b : ℝ) : Prop :=
∀ x : ℝ, exp (x + 1) ≥ a * x + b

theorem maximum_value_ab (a b : ℝ) (h : proof_problem a b) : ab ≤ (1 / 2) * exp 3 :=
  sorry

end maximum_value_ab_l570_570816


namespace smallest_n_for_doubling_sum_l570_570590

theorem smallest_n_for_doubling_sum :
  let D (a n : ℕ) := a * (2^n - 1)
  ∃ n : ℕ, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 6 → ∃ a_i : ℕ, D a_i i = n) ∧ n = Nat.lcm (Nat.repeat (fun i : ℕ => 2^i - 1) 6) :=
sorry

end smallest_n_for_doubling_sum_l570_570590


namespace value_of_a_minus_b_l570_570124

theorem value_of_a_minus_b 
  (a b : ℤ) 
  (x y : ℤ)
  (h1 : x = -2)
  (h2 : y = 1)
  (h3 : a * x + b * y = 1)
  (h4 : b * x + a * y = 7) : 
  a - b = 2 :=
by
  sorry

end value_of_a_minus_b_l570_570124


namespace unique_intersection_point_l570_570055

theorem unique_intersection_point (c : ℝ) :
  (∀ x : ℝ, (|x - 20| + |x + 18| = x + c) → (x = 18 - 2 \/ x = 38 - x \/ x = 2 - 3 * x)) →
  c = 18 :=
by
  sorry

end unique_intersection_point_l570_570055


namespace incorrect_statement_C_l570_570104

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos (abs x)

theorem incorrect_statement_C : ¬ (∀ x ∈ Icc 0 1, ∀ y ∈ Icc 0 1, x < y → f x < f y) :=
by sorry

end incorrect_statement_C_l570_570104


namespace intersection_A_B_complement_union_A_B_l570_570872

open Set

variable (U A B : Set ℕ)

noncomputable
def U : Set ℕ := { x | x ≤ 7 }

noncomputable
def A : Set ℕ := { 2, 4, 5 }

noncomputable
def B : Set ℕ := { 1, 2, 4, 6 }

theorem intersection_A_B : A ∩ B = { 2, 4 } := by
  sorry

theorem complement_union_A_B : U \ (A ∪ B) = { 0, 3, 7 } := by
  sorry

end intersection_A_B_complement_union_A_B_l570_570872


namespace gear_teeth_count_l570_570510

theorem gear_teeth_count 
  (x y z: ℕ) 
  (h1: x + y + z = 60) 
  (h2: 4 * x - 20 = 5 * y) 
  (h3: 5 * y = 10 * z):
  x = 30 ∧ y = 20 ∧ z = 10 :=
by
  sorry

end gear_teeth_count_l570_570510


namespace orchestra_club_members_l570_570249

theorem orchestra_club_members : ∃ (n : ℕ), 150 < n ∧ n < 250 ∧ n % 8 = 1 ∧ n % 6 = 2 ∧ n % 9 = 3 ∧ n = 169 := 
by {
  sorry
}

end orchestra_club_members_l570_570249


namespace maximum_village_strength_k_is_290_l570_570310

noncomputable def max_village_strength_k : ℕ :=
  290

theorem maximum_village_strength_k_is_290 :
  ∀ (villages : list (list ℕ)), 
    villages.length = 20 →
    (∀ v, v ∈ villages → v.length = 20) →
    (∀ v1 v2, v1 ∈ villages → v2 ∈ villages → v1 ≠ v2 → disjoint (v1.to_finset) (v2.to_finset)) →
    (∀ (v1 v2 : list ℕ), 
      v1 ∈ villages → 
      v2 ∈ villages → 
      is_neighbor v1 v2 villages →
      ∃ k, k ≥ max_village_strength_k ∧ (count_wins v1 v2 ≥ k)) :=
begin
  sorry
end

-- Helper definitions required for the theorem statement
def is_neighbor (v1 v2 : list ℕ) (villages : list (list ℕ)) : Prop :=
  let idx1 := villages.index_of v1 in
  let idx2 := villages.index_of v2 in
  (idx1 + 1) % villages.length = idx2 % villages.length ∨ 
  (idx2 + 1) % villages.length = idx1 % villages.length

def count_wins (v1 v2 : list ℕ) : ℕ :=
  (v1.product v2).count (λ ⟨a, b⟩, a > b)

end maximum_village_strength_k_is_290_l570_570310


namespace final_answer_correct_l570_570053

open Nat

def log2_floor (n : ℕ) : ℕ := Nat.log2 n

def f (n : ℕ) : ℕ := 2 * n + 1 - 2 ^ (log2_floor n + 1)

def num_ones_in_binary (n : ℕ) : ℕ := (digits 2 n).count 1

def f_m (n m : ℕ) : ℕ :=
  if m = 0 then n else f (f_m n (m - 1))

def is_valid_n (n : ℕ) : Bool :=
  num_ones_in_binary n == num_ones_in_binary 2015

def count_valid_n : ℕ :=
  (Finset.range 65535).filter (λ n, is_valid_n n).card

theorem final_answer_correct : count_valid_n = 8008 :=
  sorry

end final_answer_correct_l570_570053


namespace discontinuity_at_3_l570_570992

def f (x : ℝ) : ℝ := 6 / (x - 3)^2

theorem discontinuity_at_3 : ¬ (continuous_at f 3) := 
by 
  -- Proof here
  sorry

end discontinuity_at_3_l570_570992


namespace equation_represents_ellipse_l570_570374

theorem equation_represents_ellipse : ∀ x y : ℝ, x^2 + 3 * y^2 - 6 * x - 12 * y + 9 = 0 ↔ (∃ h k a b : ℝ, h = 3 ∧ k = 2 ∧ a = sqrt 12 ∧ b = 2 ∧ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) := 
begin
  sorry
end

end equation_represents_ellipse_l570_570374


namespace probability_of_multiple_of_3_l570_570265

theorem probability_of_multiple_of_3 :
  let cards := [6, 7, 8, 9]
  let multiples_of_3 := cards.filter (fun x => x % 3 = 0)
  let total_cards := cards.length
  let num_multiples_of_3 := multiples_of_3.length
  let probability := num_multiples_of_3 / total_cards in
  probability = 1 / 2 :=
by
  sorry

end probability_of_multiple_of_3_l570_570265


namespace quadratic_root_l570_570258

-- a, b, c form an arithmetic sequence and a >= c >= b >= 0.
def is_arithmetic_sequence (a c b : ℝ) : Prop :=
  ∃ d : ℝ, c = a - d ∧ b = a - 2 * d

theorem quadratic_root (a c b : ℝ) (h1 : is_arithmetic_sequence a c b)
  (h2 : a ≥ c) (h3 : c ≥ b) (h4 : b ≥ 0) (h5 : ∃ x : ℝ, (a * x^2 + c * x + b = 0)) :
  
  let Δ := c^2 - 4 * a * b in
  (Δ = 0) → (∃ r : ℝ, ax^2 + cx + b = 0) :=
  (∃ r : ℝ, r = -2 + sqrt 3) :=
begin
  sorry
end

end quadratic_root_l570_570258


namespace classes_Mr_Gates_has_l570_570879

theorem classes_Mr_Gates_has (buns_per_package packages_bought students_per_class buns_per_student : ℕ) :
  buns_per_package = 8 → 
  packages_bought = 30 → 
  students_per_class = 30 → 
  buns_per_student = 2 → 
  (packages_bought * buns_per_package) / (students_per_class * buns_per_student) = 4 := 
by
  sorry

end classes_Mr_Gates_has_l570_570879


namespace ratio_of_a_plus_b_to_b_plus_c_l570_570494

variable (a b c : ℝ)

theorem ratio_of_a_plus_b_to_b_plus_c (h1 : b / a = 3) (h2 : c / b = 4) : (a + b) / (b + c) = 4 / 15 :=
by
  sorry

end ratio_of_a_plus_b_to_b_plus_c_l570_570494


namespace numbers_divisible_by_3_but_not_2_l570_570881

theorem numbers_divisible_by_3_but_not_2 (n : ℕ) (h₀ : n < 100) : 
  (∃ m, 1 ≤ m ∧ m < 100 ∧ m % 3 = 0 ∧ m % 2 ≠ 0) ↔ n = 17 := 
by {
  sorry
}

end numbers_divisible_by_3_but_not_2_l570_570881


namespace simplify_complex_expr_l570_570228

theorem simplify_complex_expr : 
  let i : ℂ := complex.I in 
  (4 - 6 * i) / (4 + 6 * i) + (4 + 6 * i) / (4 - 6 * i) = -10 / 13 :=
by
  sorry

end simplify_complex_expr_l570_570228


namespace compute_b_l570_570095

-- Defining the polynomial and the root conditions
def poly (x a b : ℝ) := x^3 + a * x^2 + b * x + 21

theorem compute_b (a b : ℚ) (h1 : poly (3 + Real.sqrt 5) a b = 0) (h2 : poly (3 - Real.sqrt 5) a b = 0) : 
  b = -27.5 := 
sorry

end compute_b_l570_570095


namespace ellipse_eccentricity_l570_570107

theorem ellipse_eccentricity
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : ∀ x y : ℝ, (x + 2 * y - 2 = 0) → 
                     (∃ y, (x^2 / a^2 + y^2 / b^2 = 1)) ∧ 
                     (∀ y, (x^2 / a^2 + y^2 / b^2 = 1) → (y = 0) ∨ (y = 1))) :
  let e : ℝ := 2 / Real.sqrt 5 in
  e = 2 * Real.sqrt 5 / 5 :=
by
  -- proof to be completed
  sorry

end ellipse_eccentricity_l570_570107


namespace linden_squares_l570_570213

def squares_count : ℕ := 200  -- Pedro's squares
def squares_jesus : ℕ := 60   -- Jesus's squares
def difference : ℕ := 65      -- The additional squares Pedro has over the combined squares

-- We need to show that Linden's squares (L) satisfies the equation
theorem linden_squares : ∃ (L : ℕ), squares_count = squares_jesus + L + difference ∧ L = 75 :=
by
  existsi 75
  split
  · -- Show that 200 = 60 + 75 + 65
    calc
      squares_count = squares_jesus + 75 + difference : by rfl
      ...          = 200                               : by rfl
  sorry -- Proof of the correct answer

end linden_squares_l570_570213


namespace sequence_eventually_constant_l570_570415

theorem sequence_eventually_constant (n : ℕ) (h : n ≥ 1) : 
  ∃ s, ∀ k ≥ s, (2 ^ (2 ^ k) % n) = (2 ^ (2 ^ (k + 1)) % n) :=
sorry

end sequence_eventually_constant_l570_570415


namespace prob_sum_equals_15_is_0_l570_570559

theorem prob_sum_equals_15_is_0 (coin1 coin2 : ℕ) (die_min die_max : ℕ) (age : ℕ)
  (h1 : coin1 = 5) (h2 : coin2 = 15) (h3 : die_min = 1) (h4 : die_max = 6) (h5 : age = 15) :
  ((coin1 = 5 ∨ coin2 = 15) → die_min ≤ ((if coin1 = 5 then 5 else 15) + (die_max - die_min + 1)) ∧ 
   (die_min ≤ 6) ∧ 6 ≤ die_max) → 
  0 = 0 :=
by
  sorry

end prob_sum_equals_15_is_0_l570_570559


namespace time_between_last_two_rings_l570_570345

variable (n : ℕ) (x y : ℝ)

noncomputable def timeBetweenLastTwoRings : ℝ :=
  x + (n - 3) * y

theorem time_between_last_two_rings :
  timeBetweenLastTwoRings n x y = x + (n - 3) * y :=
by
  sorry

end time_between_last_two_rings_l570_570345


namespace smallest_positive_x_l570_570789

theorem smallest_positive_x 
  (x : ℝ) 
  (H : 0 < x) 
  (H_eq : ⌊x^2⌋ - x * ⌊x⌋ = 10) : 
  x = 131 / 11 :=
sorry

end smallest_positive_x_l570_570789


namespace necessary_and_sufficient_condition_l570_570463

open Classical

noncomputable def f (x a : ℝ) := x + a / x

theorem necessary_and_sufficient_condition
  (a : ℝ) :
  (∀ x : ℝ, x > 0 → f x a ≥ 2) ↔ (a ≥ 1) :=
by
  sorry

end necessary_and_sufficient_condition_l570_570463


namespace mutually_exclusive_not_opposite_l570_570751

open Probability

-- Define the people and cards
inductive Person : Type
| A | B | C | D
inductive Card : Type
| Red | Yellow | Blue | White

-- Define a random distribution of cards to people
def random_distribution : Person → Card → Prop :=
λ p c, true

def event_A_gets_red_card := random_distribution Person.A Card.Red
def event_B_gets_blue_card := random_distribution Person.B Card.Blue

theorem mutually_exclusive_not_opposite :
  MutuallyExclusive event_A_gets_red_card event_B_gets_blue_card ∧
  ¬ Opposite event_A_gets_red_card event_B_gets_blue_card := sorry

end mutually_exclusive_not_opposite_l570_570751


namespace smallest_positive_x_l570_570791

theorem smallest_positive_x 
  (x : ℝ) 
  (H : 0 < x) 
  (H_eq : ⌊x^2⌋ - x * ⌊x⌋ = 10) : 
  x = 131 / 11 :=
sorry

end smallest_positive_x_l570_570791


namespace number_of_odd_digits_in_base4_rep_of_157_l570_570393

def count_odd_digits_in_base4 (n : ℕ) : ℕ :=
  (nat.digits 4 n).countp (λ d, d % 2 = 1)

theorem number_of_odd_digits_in_base4_rep_of_157 : count_odd_digits_in_base4 157 = 2 :=
by
  sorry

end number_of_odd_digits_in_base4_rep_of_157_l570_570393


namespace average_speed_of_trip_l570_570694

def average_speed (distance_north distance_east speed_north speed_east return_speed : ℝ) : ℝ :=
  let time_north := distance_north / speed_north
  let time_east := distance_east / speed_east
  let return_distance := Real.sqrt (distance_north ^ 2 + distance_east ^ 2)
  let time_return := return_distance / return_speed
  let total_distance := distance_north + distance_east + return_distance
  let total_time := time_north + time_east + time_return
  total_distance / total_time

theorem average_speed_of_trip : 
  average_speed 10 24 10 12 13 = 12 := 
by
  -- proof omitted
  sorry

end average_speed_of_trip_l570_570694


namespace profit_diff_is_560_l570_570664

-- Define the initial conditions
def capital_A : ℕ := 8000
def capital_B : ℕ := 10000
def capital_C : ℕ := 12000
def profit_share_B : ℕ := 1400

-- Define the ratio parts
def ratio_A : ℕ := 4
def ratio_B : ℕ := 5
def ratio_C : ℕ := 6

-- Define the value of one part based on B's profit share and ratio part
def value_per_part : ℕ := profit_share_B / ratio_B

-- Define the profit shares of A and C
def profit_share_A : ℕ := ratio_A * value_per_part
def profit_share_C : ℕ := ratio_C * value_per_part

-- Define the difference between the profit shares of A and C
def profit_difference : ℕ := profit_share_C - profit_share_A

-- The theorem to prove
theorem profit_diff_is_560 : profit_difference = 560 := 
by sorry

end profit_diff_is_560_l570_570664


namespace find_100th_term_l570_570008

/-- Define the sequence {a_n} as in the problem statement:
    a_1 = 1, a_2 = 2, and a_{n+1} is the largest prime divisor of the sum of the first n terms. -/
def a : ℕ → ℕ
| 0       := 0 -- To make it 1-based indexing
| 1       := 1
| 2       := 2
| (n + 1) := (nat.factors (finset.sum (finset.range n.succ).map (λ i, a i))).max' (begin sorry end)

theorem find_100th_term : a 100 = 53 := 
by sorry

end find_100th_term_l570_570008


namespace marcy_total_spears_l570_570963

-- Define the conditions
def can_make_spears_from_sapling (spears_per_sapling : ℕ) (saplings : ℕ) : ℕ :=
  spears_per_sapling * saplings

def can_make_spears_from_log (spears_per_log : ℕ) (logs : ℕ) : ℕ :=
  spears_per_log * logs

-- Number of spears Marcy can make from 6 saplings and 1 log
def total_spears (spears_per_sapling : ℕ) (saplings : ℕ) (spears_per_log : ℕ) (logs : ℕ) : ℕ :=
  can_make_spears_from_sapling spears_per_sapling saplings + can_make_spears_from_log spears_per_log logs

-- Given conditions
theorem marcy_total_spears (saplings : ℕ) (logs : ℕ) : 
  total_spears 3 6 9 1 = 27 :=
by
  sorry

end marcy_total_spears_l570_570963


namespace smallest_number_of_students_l570_570905

-- Definitions for the ratios
def ratio_9th_to_10th := (7, 4)
def ratio_9th_to_11th := (5, 3)

-- The main statement to prove
theorem smallest_number_of_students :
  let LCM := Nat.lcm ratio_9th_to_10th.1 ratio_9th_to_11th.1 in
  let num_9th := LCM in
  let num_10th := LCM / ratio_9th_to_10th.1 * ratio_9th_to_10th.2 in
  let num_11th := LCM / ratio_9th_to_11th.1 * ratio_9th_to_11th.2 in
  num_9th + num_10th + num_11th = 76 :=
by
  sorry

end smallest_number_of_students_l570_570905


namespace distance_traveled_on_foot_l570_570299

theorem distance_traveled_on_foot (x y : ℝ) : x + y = 61 ∧ (x / 4 + y / 9 = 9) → x = 16 :=
by {
  sorry
}

end distance_traveled_on_foot_l570_570299


namespace ceil_sqrt_sum_l570_570378

theorem ceil_sqrt_sum :
  (⌈Real.sqrt 8⌉ + ⌈Real.sqrt 48⌉ + ⌈Real.sqrt 288⌉) = 27 :=
by
  have h1 : 2 < Real.sqrt 8 ∧ Real.sqrt 8 < 3, from sorry,
  have h2 : 6 < Real.sqrt 48 ∧ Real.sqrt 48 < 7, from sorry,
  have h3 : 16 < Real.sqrt 288 ∧ Real.sqrt 288 < 17, from sorry,
  have hceil8 : ⌈Real.sqrt 8⌉ = 3, from sorry,
  have hceil48 : ⌈Real.sqrt 48⌉ = 7, from sorry,
  have hceil288 : ⌈Real.sqrt 288⌉ = 17, from sorry,
  rw [hceil8, hceil48, hceil288],
  norm_num

end ceil_sqrt_sum_l570_570378


namespace smallest_solution_l570_570774

def smallest_positive_real_x : ℝ :=
  (131 : ℝ) / 11

theorem smallest_solution (x : ℝ) (hx : 0 < x) (H : ⌊x^2⌋ - x * ⌊x⌋ = 10) : x = smallest_positive_real_x :=
  sorry

end smallest_solution_l570_570774


namespace cost_of_gas_l570_570984

-- Define the given conditions
def start_odometer_reading : ℤ := 85300
def end_odometer_reading : ℤ := 85335
def car_mileage : ℤ := 25
def price_per_gallon : ℝ := 4.20

-- Calculate the distance traveled, fuel used, and cost of gas
def distance_traveled : ℤ := end_odometer_reading - start_odometer_reading
def fuel_used : ℝ := distance_traveled.toReal / car_mileage.toReal
def cost_of_gas_used : ℝ := fuel_used * price_per_gallon

-- Theorem to prove that the cost of gas used is 5.88 dollars
theorem cost_of_gas : cost_of_gas_used = 5.88 := by
  sorry

end cost_of_gas_l570_570984


namespace pies_count_l570_570725

-- Definitions based on the conditions given in the problem
def strawberries_per_pie := 3
def christine_strawberries := 10
def rachel_strawberries := 2 * christine_strawberries

-- The theorem to prove
theorem pies_count : (christine_strawberries + rachel_strawberries) / strawberries_per_pie = 10 := by
  sorry

end pies_count_l570_570725


namespace magnitude_a_minus_b_l570_570094

variables (a b : ℝ) (λ : ℝ)
hypothesis ha : ‖a‖ = 1
hypothesis hb : ‖b‖ = 2
hypothesis hλ : a = λ • b

theorem magnitude_a_minus_b : ‖a - b‖ = 1 ∨ ‖a - b‖ = 3 :=
begin
  sorry
end

end magnitude_a_minus_b_l570_570094


namespace min_value_quadratic_l570_570285

theorem min_value_quadratic (x : ℝ) : 
  ∀ x ∈ ℝ, x = 6 ↔ x^2 - 12x + 36 = (x - 6)^2 ∨ (x - 6)^2 >= 0 := 
begin
  sorry
end

end min_value_quadratic_l570_570285


namespace probability_heads_and_die_three_l570_570122

theorem probability_heads_and_die_three :
  let total_outcomes := 24 in
  let successful_outcomes := 3 in
  (successful_outcomes : ℚ) / (total_outcomes : ℚ) = 1 / 8 :=
by
  have total_outcomes := 4 * 6  -- Total number of outcomes from flipping the coin twice and rolling the die.
  have successful_outcomes := 3  -- HH, HT, TH each paired with a 3 from the die.
  exact (successful_outcomes : ℚ) / (total_outcomes : ℚ) = 1 / 8

end probability_heads_and_die_three_l570_570122


namespace marathon_yards_l570_570326

theorem marathon_yards (miles_per_marathon : ℕ) (extra_yards_per_marathon : ℕ) (yards_per_mile : ℕ) (num_marathons : ℕ)
  (total_miles : ℕ) (total_yards : ℕ) 
  (H1 : miles_per_marathon = 26) 
  (H2 : extra_yards_per_marathon = 395) 
  (H3 : yards_per_mile = 1760) 
  (H4 : num_marathons = 15) 
  (H5 : total_miles = num_marathons * miles_per_marathon + (num_marathons * extra_yards_per_marathon) / yards_per_mile)
  (H6 : total_yards = (num_marathons * extra_yards_per_marathon) % yards_per_mile)
  (H7 : 0 ≤ total_yards ∧ total_yards < yards_per_mile) 
  : total_yards = 645 :=
sorry

end marathon_yards_l570_570326


namespace exists_boy_with_exactly_ten_good_pairs_l570_570620

theorem exists_boy_with_exactly_ten_good_pairs (n : ℕ) 
  (boys_girls_circle : list (bool)) 
  (length_eq : boys_girls_circle.length = 2 * n)
  (exactly_ten_good_pairs_for_girl : ∃ D, D ∈ boys_girls_circle ∧ count_good_pairs D 10) :
  ∃ B, B ∈ boys_girls_circle ∧ count_good_pairs B 10 :=
sorry

end exists_boy_with_exactly_ten_good_pairs_l570_570620


namespace find_m_given_sampling_conditions_l570_570144

-- Definitions for population and sampling conditions
def population_divided_into_groups : Prop :=
  ∀ n : ℕ, n < 100 → ∃ k : ℕ, k < 10 ∧ n / 10 = k

def systematic_sampling_condition (m k : ℕ) : Prop :=
  k < 10 ∧ m < 10 ∧ (m + k - 1) % 10 < 10 ∧ (m + k - 11) % 10 < 10

-- Given conditions
def given_conditions (m k : ℕ) (n : ℕ) : Prop :=
  k = 6 ∧ n = 52 ∧ systematic_sampling_condition m k

-- The statement to prove
theorem find_m_given_sampling_conditions :
  ∃ m : ℕ, given_conditions m 6 52 ∧ m = 7 :=
by
  sorry

end find_m_given_sampling_conditions_l570_570144


namespace prism_sphere_surface_area_l570_570825

-- Define the regular triangular prism
structure TriangularPrism (V : Type*) [metric_space V] :=
  (A B C A1 B1 C1 : V)
  (AB_length : dist A B = sqrt 3)
  (volume : real_volume A B C A1 B1 C1 = 3 * sqrt 3)
  (co_spherical : ∃ (O : V) (r : ℝ), ∀ p ∈ {A, B, C, A1, B1, C1}, dist O p = r)

-- Define the problem statement
noncomputable def sphere_surface_area (V : Type*) [metric_space V] (prism : TriangularPrism V) : ℝ :=
  4 * pi * (sqrt (13 / 3))^2

-- The theorem to prove
theorem prism_sphere_surface_area (V : Type*) [metric_space V] (prism : TriangularPrism V) :
  sphere_surface_area V prism = 52 * pi / 3 :=
sorry

end prism_sphere_surface_area_l570_570825


namespace problem_solution_l570_570915

noncomputable def a : ℕ → ℝ
| 1       := 1
| (n + 1) := if n = 1 then 2 else a n + (-1)^(n+1) / a n

lemma sequence_property {n : ℕ} (h : n ≥ 1) :
  a (n + 1) * a n = a n + (-1)^(n + 1) := 
sorry

theorem problem_solution : (a 3 / a 4) = 1/6 :=
sorry

end problem_solution_l570_570915


namespace reflection_through_plane_l570_570179

noncomputable def reflection_matrix (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let r := (1/3, -2/3, 2/3,
            -2/3, 1/3, 2/3,
            2/3, 2/3, 1/3)
  in (r.1 * v.1 + r.2 * v.2 + r.3 * v.3,
      r.4 * v.1 + r.5 * v.2 + r.6 * v.3,
      r.7 * v.1 + r.8 * v.2 + r.9 * v.3)

theorem reflection_through_plane (v : ℝ × ℝ × ℝ) :
  reflection_matrix v = (1/3 * v.1 - 2/3 * v.2 + 2/3 * v.3,
                         -2/3 * v.1 + 1/3 * v.2 + 2/3 * v.3,
                         2/3 * v.1 + 2/3 * v.2 + 1/3 * v.3) := 
  sorry

end reflection_through_plane_l570_570179


namespace cos_A_eq_find_a_l570_570916

variable {A B C a b c : ℝ}

-- Proposition 1: If in triangle ABC, b^2 + c^2 - (sqrt 6) / 2 * b * c = a^2, then cos A = sqrt 6 / 4
theorem cos_A_eq (h : b ^ 2 + c ^ 2 - (Real.sqrt 6) / 2 * b * c = a ^ 2) : Real.cos A = Real.sqrt 6 / 4 :=
sorry

-- Proposition 2: Given b = sqrt 6, B = 2 * A, and b^2 + c^2 - (sqrt 6) / 2 * b * c = a^2, then a = 2
theorem find_a (h1 : b ^ 2 + c ^ 2 - (Real.sqrt 6) / 2 * b * c = a ^ 2) (h2 : B = 2 * A) (h3 : b = Real.sqrt 6) : a = 2 :=
sorry

end cos_A_eq_find_a_l570_570916


namespace z_is_real_z_is_complex_z_is_purely_imaginary_l570_570419

def complex_number (m : ℝ) : ℂ :=
  (m^2 - m - 6) / (m + 3) + (m^2 - 2m - 15) * complex.I

-- Condition: m ≠ -3
variable (m : ℝ) (h1 : m ≠ -3)

-- Proof that for appropriate m values, z is classified properly
theorem z_is_real : m = 5 → (complex_number m).im = 0 :=
by sorry

theorem z_is_complex : m ≠ 5 ∧ m ≠ -3 → (complex_number m).im ≠ 0 :=
by sorry

theorem z_is_purely_imaginary : (m = 3 ∨ m = -2) → (complex_number m).re = 0 :=
by sorry

end z_is_real_z_is_complex_z_is_purely_imaginary_l570_570419


namespace existence_of_committees_l570_570309

noncomputable def committeesExist : Prop :=
∃ (C : Fin 1990 → Fin 11 → Fin 3), 
  (∀ i j, i ≠ j → C i ≠ C j) ∧
  (∀ i j, i = j + 1 ∨ (i = 0 ∧ j = 1990 - 1) → ∃ k, C i k = C j k)

theorem existence_of_committees : committeesExist :=
sorry

end existence_of_committees_l570_570309


namespace leadership_meeting_ways_l570_570317

theorem leadership_meeting_ways :
  let num_schools := 4
  let num_members_per_school := 6
  ∃ (ways_to_organize : ℕ),
  (ways_to_organize = num_schools * (num_members_per_school.choose 3) * (num_members_per_school^3) ∧
  ways_to_organize = 17280) :=
by
  let num_schools := 4
  let num_members_per_school := 6
  have ways_to_organize : ℕ := num_schools * (num_members_per_school.choose 3) * (num_members_per_school^3)
  have h : ways_to_organize = 17280
  sorry

end leadership_meeting_ways_l570_570317


namespace sum_of_positive_differences_l570_570180

open BigOperators

def S : Finset ℕ := Finset.range 11 .map (λ i, 2^i)

noncomputable def N : ℕ := ∑ x in S, ∑ y in S, (x - y).natAbs

theorem sum_of_positive_differences :
  N = 16398 :=
by sorry

end sum_of_positive_differences_l570_570180


namespace sequences_general_formulas_l570_570157

def a (n : ℕ) : ℝ := (1/4)^n
def b (n : ℕ) : ℝ := 3 * n - 2
def c (n : ℕ) : ℝ := a n * b n
def S (n : ℕ) : ℝ := (finset.range n).sum c

theorem sequences_general_formulas :
  a 1 = 1/4 ∧
  (∀ n, a (n + 1) = (1/4) * a n) ∧
  (∀ n, b n + 2 = 3 * (real.log (a n) / real.log (1/4))) ∧
  (∀ n, a n = (1/4)^n) ∧
  (∀ n, b n = 3 * n - 2) ∧
  ∀ n, S n = (2/3) - (3 * n + 2) / 3 * (1/4)^n := 
by 
  -- Initial conditions
  have a1 : a 1 = (1/4)^1 := rfl
  have a_recurrence : ∀ n, a (n + 1) = (1/4) * a n :=
    λ n, rfl
  
  -- Formulas for a_n and b_n
  have a_formula : ∀ n, a n = (1/4)^(n) :=
    λ n, rfl
  have b_relation : ∀ n, b n + 2 = 3 * (real.log (a n) / real.log (1/4)) :=
    λ n, rfl
  have b_formula : ∀ n, b n = 3 * n - 2 :=
    λ n, rfl
  
  -- Proof of S_n formula to be completed
  sorry

end sequences_general_formulas_l570_570157


namespace vertex_on_line_l570_570209

theorem vertex_on_line (m : ℝ) : 
  (vertex_x : ℝ) (vertex_y : ℝ) 
  (h₁ : vertex_x = -m)
  (h₂ : vertex_y = m - 1) :
  vertex_y = -vertex_x - 1 :=
sorry

end vertex_on_line_l570_570209


namespace apples_given_to_father_l570_570163

theorem apples_given_to_father
  (total_apples : ℤ) 
  (people_sharing : ℤ) 
  (apples_per_person : ℤ)
  (jack_and_friends : ℤ) :
  total_apples = 55 →
  people_sharing = 5 →
  apples_per_person = 9 →
  jack_and_friends = 4 →
  (total_apples - people_sharing * apples_per_person) = 10 :=
by 
  intros h1 h2 h3 h4
  sorry

end apples_given_to_father_l570_570163


namespace triangles_similar_l570_570555

variables {A B C T S B1 C1 : Type*} [euclidean_geometry A B C T S B1 C1]

-- Definitions for conditions
def is_triangle (ABC : Set Point) : Prop :=
∃ (A B C : Point), A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ (A, B, C) ∈ ABC

def is_tangent (circle : Set Point) (line : Set Point) (P : Point) : Prop :=
∃ (Q : Point), Q ≠ P ∧ Q ∈ circle ∧ P ∈ line ∧ (line ∩ circle = {Q})

def circumcircle (ABC : Set Point) : Set Point :=
{P : Point | ∃ (O : Point) (r : ℝ), r > 0 ∧ ∀ (Q ∈ ABC), dist O Q = r ∧ dist O P = r}

def perpendicular (line1 line2 : Set Point) : Prop :=
∃ (vec1 vec2 : Vector), vec1 ∈ line1 ∧ vec2 ∈ line2 ∧ vec1 ⋅ vec2 = 0

def on_line (P : Point) (line : Set Point) : Prop :=
P ∈ line

def segment_eq (P Q R : Point) : Prop :=
dist P Q = dist Q R

-- Lean statement for the problem
theorem triangles_similar
(ABC : Set Point) (A B C T S B1 C1 : Point)
(h_triangle : is_triangle ABC)
(h_tangent_B : is_tangent (circumcircle ABC) (line_through B T) B)
(h_tangent_C : is_tangent (circumcircle ABC) (line_through C T) C)
(h_S_on_BC : on_line S (line_through B C))
(h_perpendicular : perpendicular (line_through A S) (line_through A T))
(h_B1_on_ST : on_line B1 (line_through S T))
(h_C1_on_ST : on_line C1 (line_through S T))
(h_B1_between_S_C1 : between B1 S C1)
(h_segment_eq1 : segment_eq B1 T B T)
(h_segment_eq2 : segment_eq C1 T B T)
: similar_triangle ABC (triangle_mk A B1 C1) :=
sorry

end triangles_similar_l570_570555


namespace total_chips_eaten_l570_570927

theorem total_chips_eaten (dinner_chips after_dinner_chips : ℕ) (h1 : dinner_chips = 1) (h2 : after_dinner_chips = 2 * dinner_chips) : dinner_chips + after_dinner_chips = 3 := by
  sorry

end total_chips_eaten_l570_570927


namespace verify_function_values_l570_570106

def function_extreme_value_and_max_value (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
  f = λ x, a * x^3 + b * x + c ∧ 
  f 2 = c - 16 ∧ 
  ∀ x, f' (a, b, c) x = if x = 2 then c - 16 else f x ≠ c - 16 ∧ 
  ∀ x, f x = 28 → x = -2

theorem verify_function_values 
  (f : ℝ → ℝ) 
  (a b c : ℝ) 
  (h1 : function_extreme_value_and_max_value f a b c)
  (h2 : ∀ x, f x ≤ 28) :
  a = 1 ∧ b = -12 ∧ (∀ x ∈ set.Icc (-3 : ℝ) 3, f x ≤ 28) ∧ (∃ x ∈ set.Icc (-3 : ℝ) 3, f x = -4) :=
by sorry

end verify_function_values_l570_570106


namespace inequality_solution_set_in_interval_l570_570100

noncomputable def inequality_solution_set (a b : ℝ) :=
  { x : ℝ | (a * x + b) / (x - 2) ≤ 3 * a - b }

theorem inequality_solution_set_in_interval :
  ∀ (a b : ℝ), (set_of (λ x, a * x - b > 0) = set.Ioi 1) →
  a > 0 →
  b = a →
  inequality_solution_set a b = set.Iio 2 ∪ set.Ici 5 :=
by
  intros a b h1 h2 h3
  sorry

end inequality_solution_set_in_interval_l570_570100


namespace rental_cost_equation_l570_570988

theorem rental_cost_equation (x : ℕ) (h : x > 0) :
  180 / x - 180 / (x + 2) = 3 :=
sorry

end rental_cost_equation_l570_570988


namespace cannot_be_sum_of_six_consecutive_odds_l570_570659

def is_sum_of_six_consecutive_odds (n : ℕ) : Prop :=
  ∃ k : ℤ, n = (6 * k + 30)

theorem cannot_be_sum_of_six_consecutive_odds :
  ¬ is_sum_of_six_consecutive_odds 198 ∧ ¬ is_sum_of_six_consecutive_odds 390 := 
sorry

end cannot_be_sum_of_six_consecutive_odds_l570_570659


namespace midpoint_correct_l570_570521

-- Definitions of the complex numbers corresponding to the endpoints.
def z1 : Complex := Complex.mk (-8) 5
def z2 : Complex := Complex.mk 6 (-9)

-- Define the expected midpoint.
def midpoint : Complex := Complex.mk (-1) (-2)

-- Theorem stating that the midpoint of the line segment with endpoints z1 and z2 is -1 - 2i.
theorem midpoint_correct : (z1 + z2) / 2 = midpoint := 
by
  sorry

end midpoint_correct_l570_570521


namespace smallest_number_l570_570712

theorem smallest_number : 
  let a := -2
  let b := 1 / 2
  let c := 0
  let d := -real.sqrt 2
  a < b ∧ a < c ∧ a < d := 
by {
  let a := -2
  let b := 1 / 2
  let c := 0
  let d := -real.sqrt 2
  sorry
}

end smallest_number_l570_570712


namespace eq_has_infinite_solutions_l570_570420

theorem eq_has_infinite_solutions (b : ℤ) :
  (∀ x : ℤ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 :=
by 
  sorry

end eq_has_infinite_solutions_l570_570420


namespace min_n_for_even_function_l570_570740

-- Definition of the 2x2 determinant
def det2x2 (a1 a2 a3 a4 : ℝ) : ℝ := a1 * a4 - a2 * a3

-- Function as described in the problem
def f (x : ℝ) : ℝ := det2x2 (real.sqrt 3) 1 (real.sin x) (real.cos x)

-- Translated function
def translated_f (x n : ℝ) : ℝ := 2 * real.cos (x + n + real.pi / 6)

-- The proof statement
theorem min_n_for_even_function (n : ℝ) (h_positive : n > 0) : 
  (∀ x : ℝ, translated_f x n = translated_f (-x) n) → 
  n = 5 * real.pi / 6 := sorry

end min_n_for_even_function_l570_570740


namespace maximum_value_of_distance_sum_l570_570060

theorem maximum_value_of_distance_sum (x1 y1 x2 y2 : ℝ)
  (hA : x1^2 + y1^2 = 1)
  (hB : x2^2 + y2^2 = 1)
  (hAB : (x1 - x2)^2 + (y1 - y2)^2 = 1) :
  ∃ (M : ℝ), M = 2 + Real.sqrt 6 ∧ 
  (∃ (x1 y1 x2 y2 : ℝ), 
    x1^2 + y1^2 = 1 ∧ 
    x2^2 + y2^2 = 1 ∧ 
    (x1 - x2)^2 + (y1 - y2)^2 = 1 ∧ 
    M = abs (x1 + y1 - 1) + abs (x2 + y2 - 1)) :=
begin
  sorry
end

end maximum_value_of_distance_sum_l570_570060


namespace polygon_sides_l570_570665

theorem polygon_sides (n : ℕ) (h : 44 = n * (n - 3) / 2) : n = 11 :=
sorry

end polygon_sides_l570_570665


namespace continuous_at_neg_two_l570_570806

def f (x b : ℝ) : ℝ :=
  if x > -2 then
    3 * x + b
  else
    -x + 4

theorem continuous_at_neg_two : 
  ∃ b : ℝ, (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x + 2) < δ → abs (f x b - f (-2) b) < ε) → b = 12 :=
begin
  -- Assume the solution is provided here. We skip the proof.
  sorry
end

end continuous_at_neg_two_l570_570806


namespace parabola_hyperbola_intersection_l570_570441

theorem parabola_hyperbola_intersection (a b p : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_p_pos : p > 0)
  (h_eccentricity : 2 * a = b) 
  (h_area :  (sqrt 3) * p * p / 4 = sqrt 3) :
  p = 2 :=
sorry

end parabola_hyperbola_intersection_l570_570441


namespace kg_to_lbs_l570_570622

theorem kg_to_lbs (kg : ℝ) (pound_kg_factor: ℝ) (pound: ℝ):
  kg = 500 →
  pound_kg_factor = 0.9072 →
  pound = 2 →
  round (500 * (2 / 0.9072)) = 1102 :=
by
  intros kg_val factor_val pounds_val
  sorry

end kg_to_lbs_l570_570622


namespace inequality_fa_f1_fb_l570_570461

noncomputable def e := Real.exp 1

def f (x : Real) : Real := e^x + x - 2

def g (x : Real) : Real := Real.log x + x - 2

variable (a b : Real)

axiom zero_of_f : f(a) = 0

axiom zero_of_g : g(b) = 0

theorem inequality_fa_f1_fb (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) :
  f(a) < f(1) ∧ f(1) < f(b) :=
  sorry

end inequality_fa_f1_fb_l570_570461


namespace valid_n_values_count_l570_570367

theorem valid_n_values_count : 
  ∃ ℕ (a b c : ℕ), 5 * a + 55 * b + 555 * c = 5750 ∧ a + 2 * b + 3 * c ∈ { n | n < 1150 ∧ n % 9 = 0 } → 
  { n | ∃ (a b c : ℕ), 5 * a + 55 * b + 555 * c = 5750 ∧ a + 2 * b + 3 * c = n }.card = 124 :=
sorry

end valid_n_values_count_l570_570367


namespace selling_price_percentage_l570_570340

  variable (L : ℝ)  -- List price
  variable (C : ℝ)  -- Cost price after discount
  variable (M : ℝ)  -- Marked price
  variable (S : ℝ)  -- Selling price after discount

  -- Conditions
  def cost_price_condition (L : ℝ) : ℝ := 0.7 * L
  def profit_condition (C S : ℝ) : Prop := 0.75 * S = C
  def marked_price_condition (S M : ℝ) : Prop := 0.85 * M = S

  theorem selling_price_percentage (L : ℝ) (h1 : C = cost_price_condition L)
    (h2 : profit_condition C S) (h3 : marked_price_condition S M) :
    S = 0.9333 * L :=
  by
    -- This is where the proof would go
    sorry
  
end selling_price_percentage_l570_570340


namespace line_AB_eq_x_plus_3y_zero_l570_570003

variable (x y : ℝ)

def circle1 := x^2 + y^2 - 4*x + 6*y = 0
def circle2 := x^2 + y^2 - 6*x = 0

theorem line_AB_eq_x_plus_3y_zero : 
  (∃ (A B : ℝ × ℝ), circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧ circle2 A.1 A.2 ∧ circle2 B.1 B.2 ∧ (A ≠ B)) → 
  (∀ (x y : ℝ), x + 3*y = 0) := 
by
  sorry

end line_AB_eq_x_plus_3y_zero_l570_570003


namespace possible_integer_roots_count_l570_570366

theorem possible_integer_roots_count {a b c d e : ℤ} : 
  ∃ m : ℕ, m ∈ [0, 1, 2, 3, 5] ∧ ∀ r : ℤ, (r is a root with multiplicity m of f(x) = x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e) :=
sorry

end possible_integer_roots_count_l570_570366


namespace problem_solution_l570_570492

theorem problem_solution (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 3 = 21 :=
by 
  sorry

end problem_solution_l570_570492


namespace magnified_diameter_l570_570658

theorem magnified_diameter (diameter_actual : ℝ) (magnification_factor : ℕ) 
  (h_actual : diameter_actual = 0.005) (h_magnification : magnification_factor = 1000) :
  diameter_actual * magnification_factor = 5 :=
by 
  sorry

end magnified_diameter_l570_570658


namespace largest_prime_factor_l570_570765

theorem largest_prime_factor (a b c d : ℕ) (ha : a = 20) (hb : b = 15) (hc : c = 10) (hd : d = 5) :
  ∃ p, Nat.Prime p ∧ p = 103 ∧ ∀ q, Nat.Prime q ∧ q ∣ (a^3 + b^4 - c^5 + d^6) → q ≤ p :=
by
  sorry

end largest_prime_factor_l570_570765


namespace parallelogram_area_eq_3sqrt28_sum_p_q_r_s_eq_32_l570_570261

noncomputable def parallelogram_area : ℂ → ℂ → ℂ → ℂ → ℂ :=
λ z1 z2 z3 z4, (z2 - z1) * (z4 - z1)

theorem parallelogram_area_eq_3sqrt28 :
  let i := Complex.I
  let z1 := Complex.sqrt 82 + 3 * Complex.sqrt 14 * i
  let z2 := -(Complex.sqrt 82 + 3 * Complex.sqrt 14 * i)
  let z3 := Complex.sqrt 30 + Complex.sqrt 10 * i
  let z4 := -(Complex.sqrt 30 + Complex.sqrt 10 * i)
  parallelogram_area z1 z2 z3 z4 = 3 * Complex.sqrt 28 :=
sorry

theorem sum_p_q_r_s_eq_32 :
  3 + 28 + 0 + 1 = 32 :=
by norm_num

end parallelogram_area_eq_3sqrt28_sum_p_q_r_s_eq_32_l570_570261


namespace Alyosha_result_divisible_by_S_l570_570616

variable (a b S x y : ℤ)
variable (h1 : x + y = S)
variable (h2 : S ∣ a * x + b * y)

theorem Alyosha_result_divisible_by_S :
  S ∣ b * x + a * y :=
sorry

end Alyosha_result_divisible_by_S_l570_570616


namespace sufficient_not_necessary_condition_l570_570436

variable (x y : ℝ)

theorem sufficient_not_necessary_condition (h : x + y ≤ 1) : x ≤ 1/2 ∨ y ≤ 1/2 := 
  sorry

end sufficient_not_necessary_condition_l570_570436


namespace original_number_l570_570890

theorem original_number (x : ℕ) (h : x / 3 = 42) : x = 126 :=
sorry

end original_number_l570_570890


namespace number_of_words_with_at_least_one_consonant_l570_570489

def total_5_letter_words : ℕ := 6 ^ 5

def total_5_letter_vowel_words : ℕ := 2 ^ 5

def total_5_letter_words_with_consonant : ℕ := total_5_letter_words - total_5_letter_vowel_words

theorem number_of_words_with_at_least_one_consonant :
  total_5_letter_words_with_consonant = 7744 :=
  by
    -- We assert the calculation follows correctly:
    -- total_5_letter_words == 6^5 = 7776
    -- total_5_letter_vowel_words == 2^5 = 32
    -- 7776 - 32 == 7744
    sorry

end number_of_words_with_at_least_one_consonant_l570_570489


namespace no_such_function_exists_l570_570935

noncomputable def sphere_interior_points : Type := sorry
noncomputable def circle_interior_points : Type := sorry

def distance (X Y : Type) : ℝ := sorry

theorem no_such_function_exists (f : sphere_interior_points → circle_interior_points) :
  ¬ (∀ A B : sphere_interior_points, distance A B ≤ distance (f A) (f B)) := 
sorry

end no_such_function_exists_l570_570935


namespace series_sum_l570_570016

theorem series_sum :
  (∑ k : ℕ, (2^k) / (3^(3^k) + 1)) = 1 / 2 :=
by
  -- Define the series terms as a sequence
  let series_terms (n : ℕ) : ℝ := (2^n) / (3^(3^n) + 1)
  
  -- State the theorem to be proved
  have series_eq : (∑ n : ℕ, series_terms n) = 1 / 2 :=
    sorry

  exact series_eq

end series_sum_l570_570016


namespace max_value_part1_range_a_part2_l570_570473

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + x^2

theorem max_value_part1 : 
  let f := f (-4) in
  ∃ x, x ∈ set.Icc (1 : ℝ) Real.exp ∧ ∀ y, y ∈ set.Icc (1 : ℝ) Real.exp → f y ≤ f x ∧ f x = Real.exp^2 - 4 :=
by 
  let f := f (-4)
  -- sorry note
  sorry

theorem range_a_part2 : 
  (∀ x, x ∈ set.Ioo (1 : ℝ) Real.exp → (f a x ≥ 0)) ↔ a ≥ -2 * Real.exp :=
by 
  let g x := -x^2 / Real.log x
  -- sorry note
  sorry

end max_value_part1_range_a_part2_l570_570473


namespace salt_solution_mixture_l570_570121

theorem salt_solution_mixture (x : ℝ) :  
  (0.80 * x + 0.35 * 150 = 0.55 * (150 + x)) → x = 120 :=
by 
  sorry

end salt_solution_mixture_l570_570121


namespace vertices_form_parabola_l570_570732

-- Declare the constants and conditions
variables {a e : ℝ} (h_a_pos : a > 0)

-- Definitions based on the given problem
def b (d : ℝ) : ℝ := 2 * a * d + e
def c (d : ℝ) : ℝ := a * d^2 - e * d + 3

-- Definition of the vertex coordinates of the parabola
def x_d (d : ℝ) : ℝ := -d - e / (2 * a)
def y_d (d : ℝ) : ℝ := a * (-d - e / (2 * a))^2 + (2 * a * d + e) * (-d - e / (2 * a)) + a * d^2 - e * d + 3

-- Theorem to be proven
theorem vertices_form_parabola : ∃ A B C : ℝ, ∀ d : ℝ, y_d d = A + B * x_d d + C * (x_d d)^2 :=
sorry

end vertices_form_parabola_l570_570732


namespace max_value_f_x_area_triangle_ABC_l570_570103

-- Define the function f(x)
def f (x : ℝ) := 2 * sqrt 3 * sin x * cos x + 2 * cos x ^ 2

-- Define the properties of triangle ABC
def a := 2 * sqrt 3

-- Statement for the first part (maximum value)
theorem max_value_f_x : ∃ M, ∀ x, f x ≤ M ∧ ∃ x, f x = M := sorry

-- Given conditions for the second part
variables {A B C b c : ℝ}
variable (hA : f A = 2)
variable (hb_c : b + c = 6)

-- Example statement for the second part (area of the triangle)
theorem area_triangle_ABC 
  (hA : f A = 2)
  (hb_c : b + c = 6)
  (expected_A : A = π / 3) 

  (cosine_rule : a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * cos A)
  (bc_value : b * c = 8) 
  : (1 / 2 * b * c * sin A) = 2 * sqrt 3 := sorry

end max_value_f_x_area_triangle_ABC_l570_570103


namespace sqrt_meaningful_l570_570495

theorem sqrt_meaningful (x : ℝ) (h : x = 1) : (∃ y : ℝ, y = sqrt (x - 1)) :=
by 
  sorry

end sqrt_meaningful_l570_570495


namespace prove_f_solution_equivalence_l570_570089

variables (a b c x : ℝ) 

-- Given the functions f(x) and g(x)
def f (x : ℝ) : ℝ := a * x^2 + b * x + c
def g (x : ℝ) : ℝ := a * x + b

-- Given the condition that the maximum value of g(x) in the interval [-1, 1] is 2
axiom max_g_x : ∀ x, -1 ≤ x ∧ x ≤ 1 → g x ≤ 2

-- Define the specific function according to the solution
def f_solution (x : ℝ) : ℝ := 2 * x^2 - 1

-- Prove that the specific function satisfies the conditions
theorem prove_f_solution_equivalence : f_solution x = f x :=
by
  sorry

end prove_f_solution_equivalence_l570_570089


namespace find_radius_l570_570276

-- Definitions of the geometric conditions
def circle (r : ℝ) (x₀ y₀ : ℝ) : ℝ → ℝ → Prop :=
  λ x y, (x - x₀)^2 + (y - y₀)^2 = r^2

def ellipse : ℝ → ℝ → Prop :=
  λ x y, x^2 + 4 * y^2 = 5

-- The theorem statement
theorem find_radius
  (r : ℝ)
  (h₁ : ∀ x y, circle r r 0 x y → ellipse x y)
  (h₂ : ∀ x y, circle r (-r) 0 x y → ellipse x y)
  (h₃ : r ≠ 0) : r = sqrt (15) / 4 :=
begin
  sorry
end

end find_radius_l570_570276


namespace verify_correct_propositions_l570_570821

section propositions

variables {Line Plane : Type} 

-- Conditions
variables (m n : Line) (α β : Plane)

-- Propositions as functions
def prop1 : Prop := (α ⊥ β) ∧ (m ∈ α) ∧ (n ∈ β) → (m ⊥ n)
def prop2 : Prop := (m ⊥ α) ∧ (n ⊥ β) ∧ (m ⊥ n) → (α ⊥ β)
def prop3 : Prop := (m ∥ α) ∧ (n ∥ β) ∧ (m ∥ n) → (α ∥ β)
def prop4 : Prop := (m ⊥ α) ∧ (n ∥ β) ∧ (α ∥ β) → (m ⊥ n)

-- Proposition correctness
def correct_props : Prop := prop2 ∧ prop4

end propositions

theorem verify_correct_propositions (m n : Line) (α β : Plane) : correct_props m n α β :=
sorry

end verify_correct_propositions_l570_570821


namespace trapezoid_area_EFGH_l570_570291

structure Point where
  x : ℝ
  y : ℝ

def Trapezoid (E F G H : Point) : Prop :=
  E.x = F.x ∧ G.x = H.x ∧ E.y ≠ F.y ∧ G.y ≠ H.y ∧ E.x ≠ G.x

def length (p1 p2 : Point) : ℝ :=
  real.abs (p1.y - p2.y)

def height (p1 p2 : Point) : ℝ :=
  real.abs (p1.x - p2.x)

noncomputable def area (b1 b2 h : ℝ) : ℝ :=
  (1 / 2) * (b1 + b2) * h

theorem trapezoid_area_EFGH :
  ∀ (E F G H : Point),
    Trapezoid E F G H →
    E = {x := 2, y := -3} →
    F = {x := 2, y := 2} →
    G = {x := 6, y := 8} →
    H = {x := 6, y := 2} →
    area (length E F) (length G H) (height E G) = 22 :=
by
  sorry

end trapezoid_area_EFGH_l570_570291


namespace F_shaped_to_cube_l570_570005

-- Define the problem context in Lean 4
structure F_shaped_figure :=
  (squares : Finset (Fin 5) )

structure additional_squares :=
  (label : String )

def is_valid_configuration (f : F_shaped_figure) (s : additional_squares) : Prop :=
  -- This function should encapsulate the logic for checking the validity of a configuration
  sorry -- Implementation of validity check is omitted (replacing it with sorry)

-- The main theorem statement
theorem F_shaped_to_cube (f : F_shaped_figure) (squares: Finset additional_squares) : 
  ∃ valid_squares : Finset additional_squares, valid_squares.card = 3 ∧ 
    ∀ s ∈ valid_squares, is_valid_configuration f s := 
sorry

end F_shaped_to_cube_l570_570005


namespace range_of_a_l570_570898

-- Define the conditions
def line1 (a x y : ℝ) : Prop := a * x + y - 4 = 0
def line2 (x y : ℝ) : Prop := x - y - 2 = 0
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- The main theorem to state
theorem range_of_a (a : ℝ) : 
  (∃ x y : ℝ, line1 a x y ∧ line2 x y ∧ first_quadrant x y) ↔ -1 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l570_570898


namespace unique_lattice_on_circle_l570_570073

def is_lattice_point (p : ℝ × ℝ) : Prop :=
  ∃ (m n : ℤ), p = (m, n)

def circle_center := (Real.sqrt 2, Real.sqrt 3)

theorem unique_lattice_on_circle (r : ℝ) : 
  ∀ p1 p2 : ℝ × ℝ, 
  is_lattice_point p1 ∧ is_circle_center p2 ∧ (∥p1 - circle_center∥ = r) ∧ (∥p2 - circle_center∥ = r) 
  → p1 = p2 := 
sorry

end unique_lattice_on_circle_l570_570073


namespace gigi_additional_batches_l570_570427

-- Define the initial amount of flour in cups
def initialFlour : Nat := 20

-- Define the amount of flour required per batch in cups
def flourPerBatch : Nat := 2

-- Define the number of batches already baked
def batchesBaked : Nat := 3

-- Define the remaining flour
def remainingFlour : Nat := initialFlour - (batchesBaked * flourPerBatch)

-- Define the additional batches Gigi can make with the remaining flour
def additionalBatches : Nat := remainingFlour / flourPerBatch

-- Prove that with the given conditions, the additional batches Gigi can make is 7
theorem gigi_additional_batches : additionalBatches = 7 := by
  -- Calculate the remaining cups of flour after baking
  have h1 : remainingFlour = 20 - (3 * 2) := by rfl

  -- Calculate the additional batches of cookies Gigi can make
  have h2 : additionalBatches = h1 / 2 := by rfl

  -- Solve for the additional batches
  show additionalBatches = 7 from
    calc
      additionalBatches = (initialFlour - (batchesBaked * flourPerBatch)) / flourPerBatch : by rfl
      ...               = (20 - 6) / 2                               : by rw h1
      ...               = 14 / 2                                     : by rfl
      ...               = 7                                          : by rfl

end gigi_additional_batches_l570_570427


namespace parallelogram_area_calc_l570_570518

open EuclideanGeometry

-- Definitions and conditions
variables (A B C D O : Point)
variables (r₁ r₂ : ℝ) (d₁ d₂ : ℝ) (dist_circles : ℝ)

-- Given Conditions
def length_BD : ℝ := 12
def distance_circles_centers : ℝ := 16
def radius_circumcircle_AOB : ℝ := 5

-- Parallelogram condition
axiom parallelogram_ABCD : parallelogram ABCD
axiom intersection_diagonals_O : diagonal_intersection ABCD O

-- Circumcircles
axiom circumcircle_AOD (center_O4 : Point) : circumscribes_center A O D center_O4
axiom circumcircle_COD (center_O3 : Point) : circumscribes_center C O D center_O3
axiom circumcircle_AOB (center_O1 : Point) : circumscribes_center A O B center_O1
axiom circumcircle_BOC (center_O2 : Point) : circumscribes_center B O C center_O2

-- Given Measurements
axiom diagonal_BD_length : segment_length B D = length_BD
axiom circles_centers_distance : euclidean_distance center_O3 center_O4 = distance_circles_centers
axiom radius_circumcircle_AOB_value : circle_radius (circumcircle A O B center_O1) = radius_circumcircle_AOB

-- Proof to obtain
theorem parallelogram_area_calc :
  let area1 := 192 / 17 in
  let area2 := 1728 / 25 in
  parallelogram_area ABCD = area1 ∨ parallelogram_area ABCD = area2 :=
  sorry

end parallelogram_area_calc_l570_570518


namespace variance_of_set_l570_570504

theorem variance_of_set {x : ℝ} (h : mode {1, 2, x, 4} = 1) : variance {1, 2, x, 4} = 1.5 := 
by 
  sorry

end variance_of_set_l570_570504


namespace cannot_be_cylinder_l570_570676

def Solid := Type

inductive Shape
| triangle
| rectangle

inductive SolidType
| Cylinder
| Cone
| Tetrahedron
| TriangularPrism

def front_view (s : SolidType) : Shape :=
  match s with
  | SolidType.Cylinder => Shape.rectangle
  | SolidType.Cone => Shape.triangle
  | SolidType.Tetrahedron => Shape.triangle
  | SolidType.TriangularPrism => Shape.triangle
  end

theorem cannot_be_cylinder (s : SolidType) (hs : front_view s = Shape.triangle) : s ≠ SolidType.Cylinder :=
by
  sorry

end cannot_be_cylinder_l570_570676


namespace flowchart_output_proof_l570_570102

def flowchart_output (x : ℕ) : ℕ :=
  let x := x + 2
  let x := x + 2
  let x := x + 2
  x

theorem flowchart_output_proof :
  flowchart_output 10 = 16 := by
  -- Assume initial value of x is 10
  let x0 := 10
  -- First iteration
  let x1 := x0 + 2
  -- Second iteration
  let x2 := x1 + 2
  -- Third iteration
  let x3 := x2 + 2
  -- Final value of x
  have hx_final : x3 = 16 := by rfl
  -- The result should be 16
  have h_result : flowchart_output 10 = x3 := by rfl
  rw [hx_final] at h_result
  exact h_result

end flowchart_output_proof_l570_570102


namespace fraction_of_students_saying_dislike_actually_like_l570_570358

variables (total_students liking_disliking_students saying_disliking_like_students : ℚ)
          (fraction_like_dislike say_dislike : ℚ)
          (cond1 : 0.7 = liking_disliking_students / total_students) 
          (cond2 : 0.3 = (total_students - liking_disliking_students) / total_students)
          (cond3 : 0.3 * liking_disliking_students = saying_disliking_like_students)
          (cond4 : 0.8 * (total_students - liking_disliking_students) 
                    = say_dislike)

theorem fraction_of_students_saying_dislike_actually_like
    (total_students_eq: total_students = 100) : 
    fraction_like_dislike = 46.67 :=
by
  sorry

end fraction_of_students_saying_dislike_actually_like_l570_570358


namespace cos_theta_result_projection_result_l570_570877

variables (a b : ℝ × ℝ) (θ : ℝ)

def dot_product (v w : ℝ × ℝ) : ℝ :=
v.1 * w.1 + v.2 * w.2

def magnitude (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def cos_theta (a b : ℝ × ℝ) : ℝ :=
(dot_product a b) / ((magnitude a) * (magnitude b))

def projection (a b : ℝ × ℝ) : ℝ :=
magnitude b * cos_theta a b

theorem cos_theta_result : cos_theta (2, 3) (-2, 4) = 4 / Real.sqrt 65 :=
by sorry

theorem projection_result : projection (2, 3) (-2, 4) = 8 * Real.sqrt 13 / 13 :=
by sorry

end cos_theta_result_projection_result_l570_570877


namespace perpendicular_vectors_l570_570480

noncomputable def a (k : ℝ) : ℝ × ℝ := (2 * k - 4, 3)
noncomputable def b (k : ℝ) : ℝ × ℝ := (-3, k)

theorem perpendicular_vectors (k : ℝ) (h : (2 * k - 4) * (-3) + 3 * k = 0) : k = 4 :=
sorry

end perpendicular_vectors_l570_570480


namespace triangle_angle_c_and_area_l570_570813

theorem triangle_angle_c_and_area (a b c A B C : ℝ) (h1 : cos A * cos B - 1 = sin A * sin B - 2 * (sin C)^2) 
    (h2 : C = π / 3) (h3 : c = 4) (h4 : a^2 + b^2 = 32) :
    C = π / 3 ∧ (1/2 * a * b * (sin C) = 4 * sqrt 3) := by
  sorry

end triangle_angle_c_and_area_l570_570813


namespace max_real_k_l570_570742

noncomputable def max_k_condition (X : Type) (Y : Fin 31 → Set X) : Prop :=
  (∀ (x₁ x₂ : X), x₁ ≠ x₂ → ∃ (i : Fin 31), x₁ ∉ Y i ∧ x₂ ∉ Y i) ∧
  (∀ (α : Fin 31 → ℝ), (∀ i, 0 ≤ α i) → ∑ i, α i = 1 → ∃ x : X, ∑ i in (Fin 31).filter (λ i, x ∈ Y i), α i ≥ (25/31 : ℝ))

theorem max_real_k : ∃ (k : ℝ) (X : Type) (Y : Fin 31 → Set X), max_k_condition X Y ∧ k = (25/31 : ℝ) :=
sorry

end max_real_k_l570_570742


namespace count_correct_statements_l570_570057

def is_geometric_sequence (a : ℕ → ℝ) := ∃ q, ∀ n, a (n + 1) = q * a n
def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d, ∀ n, a (n + 1) = a n + d
def satisfies_general_term (a : ℕ → ℝ) := ∀ n, a n = n^2

noncomputable def is_kth_order_recursive_sequence (a : ℕ → ℝ) (k : ℕ) :=
  ∃ (λ : fin k → ℝ), ∀ n, a (n + k) = ∑ i in finset.range k, (λ i) * a (n + k - 1 - i)

theorem count_correct_statements (a : ℕ → ℝ) : 
  let S1 := is_geometric_sequence a → is_kth_order_recursive_sequence a 1,
      S2 := is_arithmetic_sequence a → is_kth_order_recursive_sequence a 2,
      S3 := satisfies_general_term a → is_kth_order_recursive_sequence a 3 in
  (S1 ∧ S2 ∧ S3) = 3 := by
  sorry

end count_correct_statements_l570_570057


namespace solve_for_r_l570_570761

theorem solve_for_r (r : ℝ) : log 49 (5 * r - 2) = -1 / 3 ↔ r = (7^(-2 / 3) + 2) / 5 :=
by
  sorry

end solve_for_r_l570_570761


namespace matches_played_by_team_B_from_city_A_l570_570140

-- Define the problem setup, conditions, and the conclusion we need to prove
structure Tournament :=
  (cities : ℕ)
  (teams_per_city : ℕ)

-- Assuming each team except Team A of city A has played a unique number of matches,
-- find the number of matches played by Team B of city A.
theorem matches_played_by_team_B_from_city_A (t : Tournament)
  (unique_match_counts_except_A : ∀ (i j : ℕ), i ≠ j → (i < t.cities → (t.teams_per_city * i ≠ t.teams_per_city * j)) ∧ (i < t.cities - 1 → (t.teams_per_city * i ≠ t.teams_per_city * (t.cities - 1)))) :
  (t.cities = 16) → (t.teams_per_city = 2) → ∃ n, n = 15 :=
by
  sorry

end matches_played_by_team_B_from_city_A_l570_570140


namespace perfect_square_trinomial_l570_570123

variable (x y : ℝ)

theorem perfect_square_trinomial (a : ℝ) :
  (∃ b c : ℝ, 4 * x^2 - (a - 1) * x * y + 9 * y^2 = (b * x + c * y) ^ 2) ↔ 
  (a = 13 ∨ a = -11) := 
by
  sorry

end perfect_square_trinomial_l570_570123


namespace third_draw_defective_probability_l570_570838

-- Defining the given conditions
def total_items : ℕ := 10
def defective_items : ℕ := 3
def draws : ℕ := 3

axiom first_is_defective : ℕ → Prop := λ n, n = 1

-- Define the problem formally as Lean statements
theorem third_draw_defective_probability :
  let non_defective_items := total_items - defective_items,
      cond1 := non_defective_items = 7,
      cond2 := set.draws = 3,
      cond3 := first_is_defective 1,
      prob_second_nondefective := 7 / (10 - 1),
      prob_third_defective := 2 / (9 - 1),
      final_prob := prob_second_nondefective * prob_third_defective
    in final_prob = 7 / 36 := by
  -- Skip proof with sorry
  sorry

end third_draw_defective_probability_l570_570838


namespace number_of_correct_statements_l570_570352

def statement_1 (x y : ℝ) : Prop :=
  sqrt (3 * x - 2) + abs (y + 1) = 0

def statement_2 (x : ℤ) : Prop :=
  x ^ 3 = x

def M (x : ℝ) : Set ℝ :=
  {y | y = x ^ 2 + 1}

def P : Set (ℝ × ℝ) :=
  {(x, y) | y = x ^ 2 + 1}

theorem number_of_correct_statements :
  let s1_correct := ∀ (x y : ℝ), statement_1 x y → (x = 2/3 ∧ y = -1)
  let s2_correct := ∀ (x : ℤ), statement_2 x → x ∈ {-1, 0, 1}
  let s3_correct := M = (λ x, {p : ℝ | p = x ^ 2 + 1})
  (ite (s1_correct ↔ false) 0 1) + 
  (ite (s2_correct = true) 1 0) + 
  (ite (s3_correct = false) 0 0) = 1 :=
by
  sorry

end number_of_correct_statements_l570_570352


namespace round_trip_time_is_three_hours_l570_570727

-- Define constants based on the conditions provided
def speed_to_work : ℝ := 80
def time_to_work_minutes : ℝ := 108
def speed_return_home : ℝ := 120

-- Convert minutes to hours
def time_to_work : ℝ := time_to_work_minutes / 60

-- Calculate distances based on speeds and times
def distance_to_work : ℝ := speed_to_work * time_to_work

-- Calculate time to return home
def time_return_home : ℝ := distance_to_work / speed_return_home

-- Define the total round trip time
def total_round_trip_time : ℝ := time_to_work + time_return_home

-- Theorem to be proved
theorem round_trip_time_is_three_hours : total_round_trip_time = 3 := by
  sorry

end round_trip_time_is_three_hours_l570_570727


namespace like_terms_mn_l570_570893

theorem like_terms_mn (m n : ℕ) (h1 : -2 * x^m * y^2 = 2 * x^3 * y^n) : m * n = 6 :=
by {
  -- Add the statements transforming the assumptions into intermediate steps
  sorry
}

end like_terms_mn_l570_570893


namespace number_of_odd_digits_in_base4_rep_of_157_l570_570394

def count_odd_digits_in_base4 (n : ℕ) : ℕ :=
  (nat.digits 4 n).countp (λ d, d % 2 = 1)

theorem number_of_odd_digits_in_base4_rep_of_157 : count_odd_digits_in_base4 157 = 2 :=
by
  sorry

end number_of_odd_digits_in_base4_rep_of_157_l570_570394


namespace sufficient_but_not_necessary_condition_l570_570307

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 0 → x^2 > 0) ∧ ¬(x^2 > 0 → x > 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l570_570307


namespace average_possible_k_l570_570851

theorem average_possible_k (k : ℕ) (r1 r2 : ℕ) (h : r1 * r2 = 24) (h_pos : r1 > 0 ∧ r2 > 0) (h_eq_k : r1 + r2 = k) : 
  (25 + 14 + 11 + 10) / 4 = 15 :=
by 
  sorry

end average_possible_k_l570_570851


namespace probability_of_both_events_l570_570538

section ProbabilityExample

variables (A B : Prop)

def Pr (p : Prop) [Decidable p] : ℝ :=
  if p then 1 else 0

variables [DecA : Decidable A] [DecB : Decidable B]

def Pr_A : ℝ := 5 / 6
def Pr_B : ℝ := 3 / 4
def independent_events : Prop := Pr (A ∧ B) = Pr A * Pr B

theorem probability_of_both_events :
  independent_events → Pr (A ∧ B) = 5 / 8 := by
  -- Proof goes here
  sorry

end ProbabilityExample

end probability_of_both_events_l570_570538


namespace boxer_initial_weight_and_fight_day_weights_l570_570958

noncomputable def initial_weight (final_weight : ℝ) (loss_rate : ℝ) (duration : ℝ) : ℝ :=
  final_weight + (loss_rate * duration)

noncomputable def weight_on_fight_day (initial_weight : ℝ) (total_loss : ℝ) : ℝ :=
  initial_weight - total_loss

theorem boxer_initial_weight_and_fight_day_weights :
  ∀ (x : ℝ), 
    (x = initial_weight 97 3 3 ∧
    weight_on_fight_day x (2 * 2 + 3 * 2) = 96 ∧
    weight_on_fight_day x 9 = 97 ∧
    weight_on_fight_day x (4 * 4) = 90) :=
by {
  intros x,
  simp [initial_weight, weight_on_fight_day],
  sorry
}

end boxer_initial_weight_and_fight_day_weights_l570_570958


namespace tutors_next_together_l570_570032

-- Define the conditions given in the problem
def Elisa_work_days := 5
def Frank_work_days := 6
def Giselle_work_days := 8
def Hector_work_days := 9

-- Theorem statement to prove the number of days until they all work together again
theorem tutors_next_together (d1 d2 d3 d4 : ℕ) 
  (h1 : d1 = Elisa_work_days) 
  (h2 : d2 = Frank_work_days) 
  (h3 : d3 = Giselle_work_days) 
  (h4 : d4 = Hector_work_days) : 
  Nat.lcm (Nat.lcm (Nat.lcm d1 d2) d3) d4 = 360 := 
by
  -- Translate the problem statement into Lean terms and structure
  sorry

end tutors_next_together_l570_570032


namespace angle_ratio_A_C_l570_570554

variable {ABC : Type} [Triangle ABC] (BC AC AB : ℝ)
variable (h : BC / (AB - BC) = (AB + BC) / AC )

theorem angle_ratio_A_C (h : BC / (AB - BC) = (AB + BC) / AC ) :
  ∃ (A C : ℝ),
    ∠A / ∠C = 1 / 2 :=
by
  /- Proof goes here -/
  sorry

end angle_ratio_A_C_l570_570554


namespace evaluate_f_at_1_l570_570722

def f : ℝ → ℝ
| x := if x < 2 then f (x + 2) else 2^(-x)

theorem evaluate_f_at_1 : f 1 = (1 / 8 : ℝ) :=
by
  sorry

end evaluate_f_at_1_l570_570722


namespace expected_value_Y_variance_Y_l570_570117

open ProbabilityTheory

def X : Type := ℝ

axiom X_normal_mean_var (X : X) : (Real.Normal 1 4) X

noncomputable def Y (X : X) : ℝ := 2 - (1 / 2) * X

theorem expected_value_Y (X : X) [X_normal_mean_var X] : E(Y X) = 3 / 2 :=
by sorry

theorem variance_Y (X : X) [X_normal_mean_var X] : Var(Y X) = 1 :=
by sorry

end expected_value_Y_variance_Y_l570_570117


namespace median_salary_is_23000_l570_570709

def positions : List (Nat × Nat) :=
  [(1, 130000), (5, 90000), (10, 75000), (6, 50000), (37, 23000)]

def total_employees_in_positions (positions: List (Nat × Nat)) : Nat :=
  positions.foldr (λ p acc => p.1 + acc) 0

def find_median_salary (positions: List (Nat × Nat)) : Nat :=
  let salaries := positions.foldr
    (λ p acc => List.append acc (List.replicate p.1 p.2))
    []
  List.nthLe (List.sort (· ≤ ·) salaries) ((List.length salaries) / 2) sorry

theorem median_salary_is_23000 :
  total_employees_in_positions positions = 59 →
  find_median_salary positions = 23000 := by 
  intro h_total
  rw total_employees_in_positions at h_total
  sorry

end median_salary_is_23000_l570_570709


namespace geometric_sequence_term_l570_570156

theorem geometric_sequence_term (a : ℕ → ℕ) (q : ℕ) (hq : q = 2) (ha2 : a 2 = 8) :
  a 6 = 128 :=
by
  sorry

end geometric_sequence_term_l570_570156


namespace maximum_compartments_l570_570303

variables (V : ℝ) (n : ℕ) (k : ℝ)

-- Conditions
def initial_speed := 96
def speed_with_compartments (n : ℕ) : ℝ := initial_speed - k * real.sqrt (n)
def speed_with_9_compartments := 24
def prop_k := k = 24

-- Statement to prove
theorem maximum_compartments
  (h1 : V = initial_speed)
  (h2 : ∀ (n : ℕ), V = initial_speed - k * real.sqrt n)
  (h3 : V = speed_with_9_compartments when n = 9)
  (h4 : prop_k) :
  ∃ n_max, (n_max = 16 ∧ speed_with_compartments n_max > 0) :=
  sorry

end maximum_compartments_l570_570303


namespace power_function_increasing_on_pos_reals_l570_570505

-- Definition of the problem conditions
def power_function (m : ℝ) (x : ℝ) : ℝ :=
  (m^2 - 4*m + 1) * x^(m^2 - 2*m - 3)

-- Proving that m = 4 makes the function increasing on (0, ∞)
theorem power_function_increasing_on_pos_reals :
  ∀ (m : ℝ), (power_function m) = (m^2 - 4*m + 1) * (x : ℝ)^ (m^2 - 2*m - 3) → 
  (0:ℝ) < x → 
  ((∀ y, 0 < y →  (power_function m y) ≤  (power_function m (y + 0.1))→ m=4): sorry

end power_function_increasing_on_pos_reals_l570_570505


namespace frog_vertical_end_probability_l570_570690

theorem frog_vertical_end_probability :
  let start := (1, 2)
  let vertex1 := (0, 0)
  let vertex2 := (0, 4)
  let vertex3 := (4, 4)
  let vertex4 := (4, 0)
  let vertices := {vertex1, vertex2, vertex3, vertex4}
  let cond := "The frog jumps randomly and stops at the boundary."
in (probability_frog_hitting_vertical_side start vertices cond) = 5 / 8 := 
sorry

end frog_vertical_end_probability_l570_570690


namespace problem_statement_l570_570187

variable {x y z : ℝ}

theorem problem_statement 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z)
  (hxyz : x * y * z = 1) :
  1 / (x ^ 3 * y) + 1 / (y ^ 3 * z) + 1 / (z ^ 3 * x) ≥ x * y + y * z + z * x :=
by sorry

end problem_statement_l570_570187


namespace find_a2_given_conditions_l570_570828

variable (a : ℕ → ℤ)

def is_arithmetic_sequence (a : ℕ → ℤ) := ∃ d a1, ∀ n, a n = a1 + (n - 1) * d

theorem find_a2_given_conditions
  {a : ℕ → ℤ}
  (h_seq : is_arithmetic_sequence a)
  (h1 : a 3 + a 5 = 24)
  (h2 : a 7 - a 3 = 24) :
  a 2 = 0 :=
by
  sorry

end find_a2_given_conditions_l570_570828


namespace tree_heights_l570_570301

theorem tree_heights (T S : ℕ) (h1 : T - S = 20) (h2 : T - 10 = 3 * (S - 10)) : T = 40 := 
by
  sorry

end tree_heights_l570_570301


namespace solve_cubic_eq_l570_570405

theorem solve_cubic_eq : 
  ∃ z : ℂ, z^3 = 8 * complex.I ∧ (z = 0 - 2 * complex.I ∨ z = complex.sqrt 3 + complex.I ∨ z = -complex.sqrt 3 + complex.I) :=
by {
  sorry  -- Proof to be filled in.
}

end solve_cubic_eq_l570_570405


namespace sequence_general_formula_l570_570110

theorem sequence_general_formula (a : ℕ → ℚ) (h₀ : a 1 = 3 / 5)
    (h₁ : ∀ n : ℕ, a (n + 1) = a n / (2 * a n + 1)) :
  ∀ n : ℕ, a n = 3 / (6 * n - 1) := 
by sorry

end sequence_general_formula_l570_570110


namespace part1_part2_l570_570820

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (a x : ℝ) : ℝ := x^3 + a * x^2 - x + 2

theorem part1 (a : ℝ) (x : ℝ) : 
  (∀ x : ℝ, -1/3 < x ∧ x < 1 → (3 * x^2 + 2 * a * x - 1) < 0) → 
  g -1 x = x^3 - x^2 - x + 2 :=
sorry

theorem part2 (a : ℝ) : 
  (∀ x : ℝ, x > 0 → 2 * f x ≤ g'' a x + 2) → 
  a ≥ -2 :=
sorry

end part1_part2_l570_570820


namespace gcd_1989_1547_l570_570279

theorem gcd_1989_1547 : Nat.gcd 1989 1547 = 221 :=
by
  sorry

end gcd_1989_1547_l570_570279


namespace div_by_3_l570_570541

theorem div_by_3 (a b : ℤ) : 
  (∃ (k : ℤ), a = 3 * k) ∨ 
  (∃ (k : ℤ), b = 3 * k) ∨ 
  (∃ (k : ℤ), a + b = 3 * k) ∨ 
  (∃ (k : ℤ), a - b = 3 * k) :=
sorry

end div_by_3_l570_570541


namespace problem1_problem2_l570_570319

noncomputable def circle_ast (a b : ℕ) : ℕ := sorry

axiom circle_ast_self (a : ℕ) : circle_ast a a = a
axiom circle_ast_zero (a : ℕ) : circle_ast a 0 = 2 * a
axiom circle_ast_add (a b c d : ℕ) : circle_ast a b + circle_ast c d = circle_ast (a + c) (b + d)

theorem problem1 : circle_ast (2 + 3) (0 + 3) = 7 := sorry

theorem problem2 : circle_ast 1024 48 = 2000 := sorry

end problem1_problem2_l570_570319


namespace remaining_solid_edges_l570_570703

-- Defining the conditions
def original_cube_side_length := 4
def small_cube_side_length := 1
def number_of_removed_cubes := 4

-- Statement of the proof problem
theorem remaining_solid_edges : 
  (original_cube_side_length = 4) →
  (small_cube_side_length = 1) →
  (number_of_removed_cubes = 4) →
  -- Conclusion: total number of edges on the remaining solid is 24
  (total_edges_on_remaining_solid original_cube_side_length small_cube_side_length number_of_removed_cubes = 24) := 
by
  sorry

end remaining_solid_edges_l570_570703


namespace range_of_m_l570_570066

def y1 (m x : ℝ) : ℝ :=
  m * (x - 2 * m) * (x + m + 2)

def y2 (x : ℝ) : ℝ :=
  x - 1

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, y1 m x < 0 ∨ y2 x < 0) ∧ (∃ x : ℝ, x < -3 ∧ y1 m x * y2 x < 0) ↔ (-4 < m ∧ m < -3/2) := 
by
  sorry

end range_of_m_l570_570066


namespace selection_methods_count_l570_570808

theorem selection_methods_count :
  ∃ (n : ℕ), (nat.choose 8 4) * (nat.choose 4 2) * (nat.factorial 2) = n ∧ n = 840 :=
begin
  use 840,
  suffices : (nat.choose 8 4) * (nat.choose 4 2) * (nat.factorial 2) = 840,
  { exact ⟨840, this, rfl⟩ },
  calc (nat.choose 8 4) * (nat.choose 4 2) * (nat.factorial 2)
      = 70 * (nat.choose 4 2) * (nat.factorial 2) : by rw nat.choose_symmetric 8 4 -- C_8^4 = 70
  ... = 70 * 6 * (nat.factorial 2)              : by rw nat.choose_symmetric 4 2 -- C_4^2 = 6
  ... = 70 * 6 * 2!                             : by rw nat.factorial -- A_2^2 = 2!
  ... = 70 * 6 * 2                              : by norm_num -- 2! = 2
  ... = 840                                     : by norm_num
end

end selection_methods_count_l570_570808


namespace negation_exists_gt_one_l570_570244

theorem negation_exists_gt_one :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) :=
sorry

end negation_exists_gt_one_l570_570244


namespace annual_interest_rate_correct_l570_570718

-- Define all the given conditions
def P : ℝ := 3000
def CI : ℝ := 630
def t : ℕ := 2
def n : ℕ := 1

-- Define the target
def expected_r : ℝ := 0.1

-- Define the compound interest formula solver
noncomputable def calculate_r (A P t n : ℝ) : ℝ :=
  (A / P)^(1 / (t * n)) - 1

-- Prove that the calculated interest rate matches the given expected rate
theorem annual_interest_rate_correct :
  let A := P + CI in
  calculate_r A P t n = expected_r :=
by
  let A := P + CI
  have h : calculate_r A P t n = (A / P)^(1 / (t * n)) - 1 := rfl
  sorry

end annual_interest_rate_correct_l570_570718


namespace container_could_be_emptied_l570_570631

theorem container_could_be_emptied (a b c : ℕ) (h : 0 ≤ a ∧ a ≤ b ∧ b ≤ c) :
  ∃ (a' b' c' : ℕ), (a' = 0 ∨ b' = 0 ∨ c' = 0) ∧
  (∀ x y z : ℕ, (a, b, c) = (x, y, z) → (a', b', c') = (y + y, z - y, x - y)) :=
sorry

end container_could_be_emptied_l570_570631


namespace ana_wins_probability_l570_570167

theorem ana_wins_probability :
  let p := 1 / 2
  in let prob_ana_wins_on_nth_turn (n : ℕ) := p^(4 * n)
  in ∑' (n : ℕ), prob_ana_wins_on_nth_turn n = 1 / 15 := sorry

end ana_wins_probability_l570_570167


namespace cube_paperclips_l570_570697

theorem cube_paperclips {V₁ V₂ : ℝ} {P₁ : ℝ} (h₁ : V₁ = 24) (h₂ : P₁ = 75) (h₃ : V₂ = 64) (h₄ : ∀ V P, (P / V) = (P₁ / V₁)) : P₂ = 200 :=
begin
  sorry
end

end cube_paperclips_l570_570697


namespace pinky_pig_apples_l570_570214

variable (P : ℕ)

theorem pinky_pig_apples (h : P + 73 = 109) : P = 36 := sorry

end pinky_pig_apples_l570_570214


namespace probability_of_factor_less_than_ten_is_half_l570_570644

-- Definitions for the factors and counts
def numFactors (n : ℕ) : ℕ :=
  let psa := 1;
  let psb := 2;
  let psc := 1;
  (psa + 1) * (psb + 1) * (psc + 1)

def factorsLessThanTen (n : ℕ) : List ℕ :=
  if n = 90 then [1, 2, 3, 5, 6, 9] else []

def probabilityLessThanTen (n : ℕ) : ℚ :=
  let totalFactors := numFactors n;
  let lessThanTenFactors := factorsLessThanTen n;
  let favorableOutcomes := lessThanTenFactors.length;
  favorableOutcomes / totalFactors

-- The proof statement
theorem probability_of_factor_less_than_ten_is_half :
  probabilityLessThanTen 90 = 1 / 2 := sorry

end probability_of_factor_less_than_ten_is_half_l570_570644


namespace part1_part2_part3_l570_570544

def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x - 2

variable {a : ℝ}
variable {k : ℤ}

-- Part 1
theorem part1 (h_slope : (deriv (λ x => f x a)) 0 = 1) : a = 0 := by
  sorry

-- Part 2
theorem part2 :
  (∀ x : ℝ, a ≤ 0 → (deriv (λ x => f x a)) x > 0) ∧
  (a > 0 →
    (∀ x : ℝ, x < Real.log a → (deriv (λ x => f x a)) x < 0) ∧
    (∀ x : ℝ, x > Real.log a → (deriv (λ x => f x a)) x > 0)) := by
  sorry

-- Part 3
theorem part3 (h_a : a = 1) (h_pos : ∀ x : ℝ, 0 < x → (x - k) * (deriv (λ x => f x 1)) x + x + 1 > 0) : 
  k ≤ 2 := by
  sorry

end part1_part2_part3_l570_570544


namespace car_distance_160_l570_570681

noncomputable theory

def car_travel_distance (d : ℝ) : Prop :=
  let t1 := d / 75 in
  let t2 := d / 80 in
  77.4193548387097 = (2 * d) / (t1 + t2) ∧ 2 * d = 320

theorem car_distance_160 : ∃ d : ℝ, car_travel_distance d ∧ d = 160 :=
by
  exists 160
  sorry

end car_distance_160_l570_570681


namespace total_books_l570_570959

-- Given conditions
def susan_books : Nat := 600
def lidia_books : Nat := 4 * susan_books

-- The theorem to prove
theorem total_books : susan_books + lidia_books = 3000 :=
by
  unfold susan_books lidia_books
  sorry

end total_books_l570_570959


namespace proper_subsets_count_l570_570882

def is_proper_subsets_of (S : Set ℕ) (n : ℕ) : Nat :=
  2^n - 1

theorem proper_subsets_count :
  let S := {x : ℕ | ∃ (H : x > 0), -1 ≤ Real.logb (1/x) 10 ∧ Real.logb (1/x) 10 < -1/2}
  card S = 90 →
  is_proper_subsets_of S 90 = (2^90 - 1) :=
by 
  intros S h₁
  sorry

end proper_subsets_count_l570_570882


namespace mittens_pair_count_l570_570269

-- Define the total number of right and left mittens for each color
def total_right_red_mittens := 10
def total_left_red_mittens := 10
def total_right_blue_mittens := 2
def total_left_blue_mittens := 2

-- Define the total number of mittens
def total_mittens := total_right_red_mittens + total_left_red_mittens +
                     total_right_blue_mittens + total_left_blue_mittens

-- Define the problem statement
theorem mittens_pair_count :
  total_right_red_mittens + total_right_blue_mittens + 1 = 13 :=
by
  rw [total_right_red_mittens, total_right_blue_mittens]
  rfl


end mittens_pair_count_l570_570269


namespace min_value_ineq_l570_570084

theorem min_value_ineq (a b c : ℝ) (hab : 0 < a) (hbb : 0 < b) (hbc : 0 < c) (h : b + c ≥ a) : 
  ∃ m : ℝ, m = sqrt 2 - 1 / 2 ∧ ∀ x y z : ℝ, (0 < x) ∧ (0 < y) ∧ (0 < z) ∧ (y + z ≥ x) → 
  ∀ v : ℝ, v = (y / z + z / (x + y)) → v ≥ m := 
sorry

end min_value_ineq_l570_570084


namespace sum_third_row_bounds_problem_statement_l570_570563

def third_row_bounds (grid_size : ℕ) (max_num : ℕ) (row_pos : ℕ) : ℕ × ℕ :=
  let left_bound := (row_pos - 1) * (grid_size - 1) + 273
  let right_bound := min max_num (left_bound + grid_size - 1)
  (left_bound, right_bound)

theorem sum_third_row_bounds :
  let third_row := third_row_bounds 17 289 3 in
  (fst third_row) + (snd third_row) = 577 :=
by
  have left_bound : ℕ := 273
  have right_bound : ℕ := 289
  have sum := left_bound + right_bound
  simp [left_bound, right_bound, sum]
  exact rfl

-- The below theorem calls the defined function and includes conditions for the grid
def grid_17x17_counterclockwise := sorry

theorem problem_statement : 
  ∀(grid : Type) (arrangement : grid_17x17_counterclockwise),
  let (least_num, greatest_num) := third_row_bounds 17 289 3 in
  least_num + greatest_num = 577
:=
by
  intros
  have proof_1 : (least_num, greatest_num) = (273, 289) := sorry
  suffices : 273 + 289 = 577, from this
  sorry

end sum_third_row_bounds_problem_statement_l570_570563


namespace evaluate_expression_base10_l570_570380

theorem evaluate_expression_base10 :
  let a := 2 * 8^2 + 5 * 8^1 + 3 * 8^0,
      b := 1 * 4^1 + 3 * 4^0,
      c := 1 * 5^2 + 4 * 5^1 + 4 * 5^0,
      d := 3 * 3^1 + 3 * 3^0
  in  (a / b : ℚ) + (c / d) = 28.511904 := 
by { 
  let a := (2 : ℚ) * 8^2 + 5 * 8^1 + 3 * 8^0,
  let b := (1 : ℚ) * 4^1 + 3 * 4^0,
  let c := (1 : ℚ) * 5^2 + 4 * 5^1 + 4 * 5^0,
  let d := (3 : ℚ) * 3^1 + 3 * 3^0,
  sorry 
}

end evaluate_expression_base10_l570_570380


namespace solve_for_x_l570_570997

theorem solve_for_x : ∀ x, (8 * x^2 + 150 * x + 2) / (3 * x + 50) = 4 * x + 2 ↔ x = -7 / 2 := by
  sorry

end solve_for_x_l570_570997


namespace expected_value_of_difference_l570_570361

noncomputable def expected_difference (num_days : ℕ) : ℝ :=
  let p_prime := 3 / 4
  let p_composite := 1 / 4
  let p_no_reroll := 2 / 3
  let expected_unsweetened_days := p_prime * p_no_reroll * num_days
  let expected_sweetened_days := p_composite * p_no_reroll * num_days
  expected_unsweetened_days - expected_sweetened_days

theorem expected_value_of_difference :
  expected_difference 365 = 121.667 := by
  sorry

end expected_value_of_difference_l570_570361


namespace matrix_power_2024_B_equals_I_l570_570932

open Matrix

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![Real.cos (π / 4), 0, -Real.sin (π / 4)],
    ![0, 1, 0],
    ![Real.sin (π / 4), 0, Real.cos (π / 4)]]

theorem matrix_power_2024_B_equals_I : B ^ 2024 = 1 :=
by
  sorry

end matrix_power_2024_B_equals_I_l570_570932


namespace platform_length_l570_570342

theorem platform_length (length_of_train : ℝ) (time_to_cross : ℝ) (speed_kmph : ℝ) :
  length_of_train = 250 →
  time_to_cross = 17.998560115190784 →
  speed_kmph = 90 →
  let speed_mps := (speed_kmph * 1000) / 3600 in
  let total_distance := speed_mps * time_to_cross in
  let length_of_platform := total_distance - length_of_train in
  length_of_platform = 199.9640028797696 :=
by
  intros h_train h_time h_speed
  have speed_mps := (speed_kmph * 1000) / 3600
  have total_distance := speed_mps * time_to_cross
  have length_of_platform := total_distance - length_of_train
  rw [h_train,h_time,h_speed]
  sorry

end platform_length_l570_570342


namespace find_abc_solution_l570_570041

theorem find_abc_solution : 
  ∃ (a b c : ℕ), 11^a + 3^b = c^2 ∧ a = 4 ∧ b = 5 ∧ c = 122 :=
by {
  use [4, 5, 122],
  split,
  { exact calc
      11^4 + 3^5 = 14641 + 243 : by norm_num
              ... = 14884       : by norm_num
              ... = 122^2       : by norm_num },
  split; reflexivity
}

end find_abc_solution_l570_570041


namespace parallel_line_b_value_l570_570442

theorem parallel_line_b_value {b : ℝ} (h₁ : ∃ (k : ℝ), ∀ x : ℝ, (y : ℝ), y = k * x + b ↔ y = 3 * x ∧ point (1, -1) ∈ {(x, y) | y = k * x + b}) : b = -4 :=
sorry

end parallel_line_b_value_l570_570442


namespace ratio_of_new_circumference_to_new_diameter_l570_570134

theorem ratio_of_new_circumference_to_new_diameter (r : ℝ) : 
  let new_radius := r + 2,
      new_diameter := 2 * new_radius,
      new_circumference := 2 * Real.pi * new_radius in 
  new_circumference / new_diameter = Real.pi := 
by 
  sorry

end ratio_of_new_circumference_to_new_diameter_l570_570134


namespace area_comparison_l570_570370

namespace Quadrilaterals

open Real

-- Define the vertices of both quadrilaterals
def quadrilateral_I_vertices : List (ℝ × ℝ) := [(0, 0), (2, 0), (2, 2), (0, 1)]
def quadrilateral_II_vertices : List (ℝ × ℝ) := [(0, 0), (3, 0), (3, 1), (0, 2)]

-- Area calculation function (example function for clarity)
def area_of_quadrilateral (vertices : List (ℝ × ℝ)) : ℝ :=
  -- This would use the actual geometry to compute the area
  2.5 -- placeholder for the area of quadrilateral I
  -- 4.5 -- placeholder for the area of quadrilateral II

theorem area_comparison :
  (area_of_quadrilateral quadrilateral_I_vertices) < (area_of_quadrilateral quadrilateral_II_vertices) :=
  sorry

end Quadrilaterals

end area_comparison_l570_570370


namespace minimal_largest_number_l570_570621

theorem minimal_largest_number
  (numbers : Finset ℕ)
  (h_len : numbers.card = 8)
  (h_2 : 6 ≤ (numbers.filter (λ n, n % 2 = 0)).card)
  (h_3 : 5 ≤ (numbers.filter (λ n, n % 3 = 0)).card)
  (h_5 : 3 ≤ (numbers.filter (λ n, n % 5 = 0)).card)
  (h_7 : 1 ≤ (numbers.filter (λ n, n % 7 = 0)).card) : 
  numbers.max' h_len = 20 :=
sorry

end minimal_largest_number_l570_570621


namespace sum_of_digits_of_unique_n_l570_570137

theorem sum_of_digits_of_unique_n :
  ∃! (n : ℕ), 0 < n ∧ log 2 (log 16 n) = log 4 (log 4 n) ∧ (2 + 5 + 6 = 13) :=
by
  sorry

end sum_of_digits_of_unique_n_l570_570137


namespace nth_value_15th_entry_l570_570052

def r_11 (n : ℕ) : ℕ := n % 11

def condition (n : ℕ) : Prop := r_11 (7 * n) ≤ 3

def sequence : List ℕ := (List.range 100).filter condition

theorem nth_value_15th_entry : sequence.nth 14 = some 41 :=
by
  sorry

end nth_value_15th_entry_l570_570052


namespace smallest_positive_real_is_131_div_11_l570_570772

noncomputable def smallest_positive_real_satisfying_condition :=
  ∀ (x : ℝ), (∀ y > 0, (y * y ⌊y⌋ - y ⌊y⌋ = 10) → (x ≤ y)) → 
  (⌊x*x⌋ - (x * ⌊x⌋) = 10) → 
  x = 131/11

theorem smallest_positive_real_is_131_div_11 :
  smallest_positive_real_satisfying_condition := sorry

end smallest_positive_real_is_131_div_11_l570_570772


namespace solve_problem_l570_570942
open Complex

noncomputable def problem (a b c d : ℝ) (ω : ℂ) : Prop :=
  (a ≠ -1) ∧ (b ≠ -1) ∧ (c ≠ -1) ∧ (d ≠ -1) ∧ (ω ^ 4 = 1) ∧ (ω ≠ 1) ∧
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 3 / ω ^ 2)
  
theorem solve_problem {a b c d : ℝ} {ω : ℂ} (h : problem a b c d ω) : 
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2) :=
sorry

end solve_problem_l570_570942


namespace min_value_expression_l570_570514

open ProbabilityTheory

-- Definitions based on given conditions
variables {σ : ℝ} {X : ℝ → ℝ}

-- Let's assume X is a normal distribution with mean 100 and variance σ²
def normal_X : MeasureTheory.Measure ℝ := MeasureTheory.Measure.dirac 100

-- Definition of probabilities a and b
def a := ProbabilityTheory.ProbabilityMeasure.probability (X > 120)
def b := ProbabilityTheory.ProbabilityMeasure.probability (80 ≤ X ∧ X ≤ 100)

-- Given relationship between a and b
axiom rel_a_b : a + b = 1 / 2

theorem min_value_expression : (4 / a + 1 / b) = 18 :=
by
  -- The proof is skipped, as only the statement is required
  sorry

end min_value_expression_l570_570514


namespace number_of_odd_digits_in_base4_rep_of_157_l570_570392

def count_odd_digits_in_base4 (n : ℕ) : ℕ :=
  (nat.digits 4 n).countp (λ d, d % 2 = 1)

theorem number_of_odd_digits_in_base4_rep_of_157 : count_odd_digits_in_base4 157 = 2 :=
by
  sorry

end number_of_odd_digits_in_base4_rep_of_157_l570_570392


namespace perpendicular_unique_l570_570661

theorem perpendicular_unique {P : Type} [plane : EuclideanGeometry P] (l : P → Prop) (p : P) :
  (∃! m : P → Prop, (P := plane) (m p) ∧ ∀ x, m x → l x → perpendicular (P := plane) m l) → sorry

end perpendicular_unique_l570_570661


namespace value_of_k_l570_570324

variables {a b : ℝ → ℝ → ℝ → ℝ} {k : ℝ}

-- Conditions that a and b are distinct vectors in ℝ³
axiom a_ne_b : a ≠ b

-- Define the line passing through a and b
def line (t : ℝ) : ℝ → ℝ := λ u, a + t * (b - a)

-- The vector we are considering must lie on the line
def point := k * a + (2 / 5) * b

-- Prove that k = 3 / 5 such that point lies on the line
theorem value_of_k : ∃ k : ℝ, point = line (2 / 5) := 
by 
  use (3 / 5),
  sorry

end value_of_k_l570_570324


namespace general_term_l570_570869

-- Define the initial sequence and its recurrence relation
def seq (n : ℕ) : ℕ :=
  Nat.recOn n 1 (λ n a_prev, a_prev + 3^(n + 1))

-- State the theorem to be proved
theorem general_term (n : ℕ) : seq n = (3^(n + 1) - 7) / 2 := 
by
  sorry

end general_term_l570_570869


namespace similar_triangles_l570_570986

theorem similar_triangles
  (P : Point)
  (l₁ l₂ : Line)
  (h_parallel : l₁ ∥ l₂)
  (A B : ℕ → Point)
  (h_A : ∀ i, A i ∈ l₁)
  (h_B : ∀ i, B i ∈ l₂)
  (h_rays_A : ∀ i, exists m, ray_through P (A i) (Rays m))
  (h_rays_B : ∀ i, exists m, ray_through P (B i) (Rays m)) :
  ∀ i, 1 ≤ i ∧ i ≤ n - 1 → similar (triangle P (A i) (A (i+1))) (triangle P (B i) (B (i+1))) :=
by
  sorry

end similar_triangles_l570_570986


namespace odd_digits_in_base4_of_157_l570_570397

theorem odd_digits_in_base4_of_157 : 
  let b4 := 2 * 4^3 + 1 * 4^2 + 3 * 4^1 + 1 * 4^0 in
  b4 = 157 → (nat.digits 4 157).countp (λ d, d % 2 = 1) = 3 := 
by
  intros
  sorry

end odd_digits_in_base4_of_157_l570_570397


namespace calculate_down_payment_l570_570222

def loan_period_years : ℕ := 5
def monthly_payment : ℝ := 250.0
def car_price : ℝ := 20000.0
def months_in_year : ℕ := 12

def total_loan_period_months : ℕ := loan_period_years * months_in_year
def total_amount_paid : ℝ := monthly_payment * total_loan_period_months
def down_payment : ℝ := car_price - total_amount_paid

theorem calculate_down_payment : down_payment = 5000 :=
by 
  simp [loan_period_years, monthly_payment, car_price, months_in_year, total_loan_period_months, total_amount_paid, down_payment]
  sorry

end calculate_down_payment_l570_570222


namespace paper_clips_in_morning_l570_570924

variable (p : ℕ) (used left : ℕ)

theorem paper_clips_in_morning (h1 : left = 26) (h2 : used = 59) (h3 : left = p - used) : p = 85 :=
by
  sorry

end paper_clips_in_morning_l570_570924


namespace sum_of_increasing_and_decreasing_sequences_l570_570375

open Nat

-- Definitions of Sets A and B
def is_strictly_increasing (n : ℕ): Prop := 
  ∀ i j, i < j → digit n i < digit n j

def is_strictly_decreasing (n : ℕ): Prop := 
  ∀ i j, i < j → digit n i > digit n j

def A : Set ℕ := {n | is_strictly_increasing n}
def B : Set ℕ := {n | is_strictly_decreasing n}

-- Sum of all natural numbers whose digits form an increasing or decreasing sequence
theorem sum_of_increasing_and_decreasing_sequences :
  (∑ n in A∪B, n) = (80 : ℚ) / 81 * 11^10 - (35 : ℚ) / 81 * 2^10 - 45 :=
sorry

end sum_of_increasing_and_decreasing_sequences_l570_570375


namespace solve_quadratic_equations_l570_570589

noncomputable def E1 := ∀ x : ℝ, x^2 - 14 * x + 21 = 0 ↔ (x = 7 + 2 * Real.sqrt 7 ∨ x = 7 - 2 * Real.sqrt 7)

noncomputable def E2 := ∀ x : ℝ, x^2 - 3 * x + 2 = 0 ↔ (x = 1 ∨ x = 2)

theorem solve_quadratic_equations :
  (E1) ∧ (E2) :=
by
  sorry

end solve_quadratic_equations_l570_570589


namespace odd_binomials_power_of_two_l570_570220

theorem odd_binomials_power_of_two (n : ℕ) : ∃ m : ℕ, (∑ k in finset.range (n + 1), if (nat.choose n k) % 2 = 1 then 1 else 0) = 2 ^ m :=
sorry

end odd_binomials_power_of_two_l570_570220


namespace find_m_l570_570438

variable (m n b : ℝ)

theorem find_m (h : log 10 m = b - log 10 n) : m = (10 ^ b) / n :=
sorry

end find_m_l570_570438


namespace student_in_16th_group_has_number_244_l570_570903

theorem student_in_16th_group_has_number_244 :
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 50 → ∃ k : ℕ, 1 ≤ k ∧ k ≤ 800 ∧ ((k - 36) % 16 = 0) ∧ (n = 3 + (k - 36) / 16)) →
  ∃ m : ℕ, 1 ≤ m ∧ m ≤ 800 ∧ ((m - 244) % 16 = 0) ∧ (16 = 3 + (m - 36) / 16) :=
by
  sorry

end student_in_16th_group_has_number_244_l570_570903


namespace jenna_tickets_total_cost_l570_570926

theorem jenna_tickets_total_cost :
  let normal_price := 50
  let website_tickets := 3 * normal_price
  let website_service_fee := 0.10 * website_tickets
  let website_total := website_tickets + website_service_fee
  let scalper_ticket_price := 2.5 * normal_price
  let scalper_tickets_cost := 4 * scalper_ticket_price
  let scalper_discount := 15
  let scalper_total_after_discount := scalper_tickets_cost - scalper_discount
  let scalper_service_fee := 0.10 * scalper_total_after_discount
  let scalper_total := scalper_total_after_discount + scalper_service_fee
  let discounted_ticket_1 := 0.60 * normal_price
  let discounted_ticket_2 := 0.75 * normal_price
  let discounted_total := discounted_ticket_1 + discounted_ticket_2
  website_total + scalper_total + discounted_total = 766 := by
    let normal_price := 50
    let website_tickets := 3 * normal_price
    let website_service_fee := 0.10 * website_tickets
    let website_total := website_tickets + website_service_fee
    let scalper_ticket_price := 2.5 * normal_price
    let scalper_tickets_cost := 4 * scalper_ticket_price
    let scalper_discount := 15
    let scalper_total_after_discount := scalper_tickets_cost - scalper_discount
    let scalper_service_fee := 0.10 * scalper_total_after_discount
    let scalper_total := scalper_total_after_discount + scalper_service_fee
    let discounted_ticket_1 := 0.60 * normal_price
    let discounted_ticket_2 := 0.75 * normal_price
    let discounted_total := discounted_ticket_1 + discounted_ticket_2
    calc
      website_total + scalper_total + discounted_total
        = (website_tickets + website_service_fee) + (scalper_total_after_discount + scalper_service_fee) + discounted_total := by sorry
        = 165 + 533.50 + 67.50 := by sorry
        = 766 := by sorry

end jenna_tickets_total_cost_l570_570926


namespace stamp_book_gcd_l570_570925

theorem stamp_book_gcd (total1 total2 total3 : ℕ) 
    (h1 : total1 = 945) (h2 : total2 = 1260) (h3 : total3 = 630) : 
    ∃ d, d = Nat.gcd (Nat.gcd total1 total2) total3 ∧ d = 315 := 
by
  sorry

end stamp_book_gcd_l570_570925


namespace maximum_N_5x5_grid_l570_570020

theorem maximum_N_5x5_grid :
  ∃ N, ∀ (grid : (Fin 5) → (Fin 5) → ℕ), 
    (∀ i j, i < 5 → j < 5 → ∃! n, (1 ≤ n ∧ n ≤ 25) → (grid i j = n) ∧ 
      (∀ x y, ((x = i ∧ y = j) ∨ (x = i ∧ y = j + 1) ∨ 
              (x = i + 1 ∧ y = j) ∨ (x = i + 1 ∧ y = j + 1)) → 
              i < 4 → j < 4 → 
              grid x y ∈ {1, 2, 3, ..., 25}) → 
              (grid i j + grid i (j + 1) + grid (i + 1) j + grid (i + 1) (j + 1) ≥ N)) :=
  sorry

end maximum_N_5x5_grid_l570_570020


namespace percentage_increase_area_l570_570135

theorem percentage_increase_area (L W : ℝ) (hL : 0 < L) (hW : 0 < W) :
  let A := L * W
  let A' := (1.35 * L) * (1.35 * W)
  let percentage_increase := ((A' - A) / A) * 100
  percentage_increase = 82.25 :=
by
  sorry

end percentage_increase_area_l570_570135


namespace average_last_4_matches_l570_570599

theorem average_last_4_matches (avg_10: ℝ) (avg_6: ℝ) (total_matches: ℕ) (first_matches: ℕ) :
  avg_10 = 38.9 → avg_6 = 42 → total_matches = 10 → first_matches = 6 → 
  (avg_10 * total_matches - avg_6 * first_matches) / (total_matches - first_matches) = 34.25 :=
by 
  intros h1 h2 h3 h4
  sorry

end average_last_4_matches_l570_570599


namespace smallest_b_value_l570_570577

def triangle_inequality (x y z : ℝ) : Prop :=
  x + y > z ∧ x + z > y ∧ y + z > x

def not_triangle (x y z : ℝ) : Prop :=
  ¬triangle_inequality x y z

theorem smallest_b_value (a b : ℝ) (h1 : 2 < a) (h2 : a < b)
    (h3 : not_triangle 2 a b) (h4 : not_triangle (1 / b) (1 / a) 1) :
    b >= 2 :=
by
  sorry

end smallest_b_value_l570_570577


namespace sequence_value_2016_l570_570826

theorem sequence_value_2016 (a : ℕ → ℕ) (h₁ : a 1 = 0) (h₂ : ∀ n, a (n + 1) = a n + 2 * n) : a 2016 = 2016 * 2015 :=
by 
  sorry

end sequence_value_2016_l570_570826


namespace shelby_today_stars_l570_570990

theorem shelby_today_stars :
  ∀ (yesterdayStars totalStars : ℕ), yesterdayStars = 4 → totalStars = 7 → (totalStars - yesterdayStars = 3)
:= by
  intros yesterdayStars totalStars h1 h2
  rw [h1, h2]  -- replace the variables with their given values
  exact rfl     -- confirm that the equation holds true
  sorry        -- skip the full proof for now

end shelby_today_stars_l570_570990


namespace tan_x_eq_k_l570_570186

theorem tan_x_eq_k (c d x k : ℝ) (h1 : tan x = c / d) (h2 : tan (2 * x) = d / (2 * c - d)) 
  (k_val : k = (1 + real.sqrt 6) / 5) : 
  tan x = k := 
begin
  sorry
end

end tan_x_eq_k_l570_570186


namespace remainder_834_l570_570339

-- Define the sequence
def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 1 ∧ ∀ n, a (n + 3) = a (n + 2) + a (n + 1) + a n

-- Given values 
def given_values (a : ℕ → ℕ) : Prop :=
  seq a ∧ a 28 = 6090307 ∧ a 29 = 11201821 ∧ a 30 = 20603361

-- Prove the remainder when sum of a_k for k from 1 to 28 is divided by 1000 is 834
theorem remainder_834 (a : ℕ → ℕ) (h : given_values a) :
    (∑ k in finset.range 29, a k) % 1000 = 834 :=
sorry

end remainder_834_l570_570339


namespace area_of_triangle_l570_570385

theorem area_of_triangle (a b c : ℝ) (h1: a = 10) (h2: b = 11) (h3: c = 11) :
  let s := (a + b + c) / 2 in
  let area := (s * (s - a) * (s - b) * (s - c)).sqrt in
  area = 20 * sqrt 6 :=
by
  sorry

end area_of_triangle_l570_570385


namespace problem_solution_l570_570610

-- Definitions for transformations
def rotate90ccw_around_point(p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  let (h, k) := center
  (h - (y - k), k + (x - h))

def reflect_about_y_equals_x(p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (y, x)

-- Theorem statement
theorem problem_solution (a b : ℝ) :
  let P := (a, b)
  rotate90ccw_around_point P (2, 3) |> reflect_about_y_equals_x = (5, -1) → b - a = 2 :=
by
  intro h
  sorry

end problem_solution_l570_570610


namespace cos_A_value_max_S_value_l570_570446

variables {a b c S : ℝ} (A : ℝ)

-- Given conditions
axiom cond1 : S = a^2 - (b - c)^2
axiom cond2 : b + c = 8

-- Statement to prove (1): cos A = 15/17
theorem cos_A_value (h1 : cond1) (h2 : cond2) : 
  ∃ A : ℝ, cos A = 15 / 17 := sorry

-- Statement to prove (2): max value of S is 64/17
theorem max_S_value (h1 : cond1) (h2 : cond2) : 
  ∃ S : ℝ, (∀ a b c, S ≤ 64 / 17) ∧ (∃ b c, S = 64 / 17) := sorry

end cos_A_value_max_S_value_l570_570446


namespace line_bisects_circle_l570_570606

theorem line_bisects_circle :
  let center := (1, 2)
  let line := λ (x y : ℝ), x - y + 1
  ∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 4 → line x y = 0 →
    (x, y) = center := by
  sorry

end line_bisects_circle_l570_570606


namespace inscribed_circle_radius_l570_570649

theorem inscribed_circle_radius (DE DF EF : ℝ) (hDE : DE = 26) (hDF : DF = 15) (hEF : EF = 17) : 
  let s := (DE + DF + EF) / 2,
      K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)),
      r := K / s
  in r = 6 * Real.sqrt 338 / 29 :=
by
  -- Definitions as per conditions
  have s_def : s = (DE + DF + EF) / 2 := rfl,
  have K_def : K = Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)) := rfl,
  have r_def : r = K / s := rfl,
  -- Specific values
  rw [hDE, hDF, hEF, s_def, K_def, r_def],
  sorry

end inscribed_circle_radius_l570_570649


namespace find_k_correct_l570_570749

noncomputable def find_k (k : ℚ) : Prop :=
  ∃ p q : Polynomial ℚ, Polynomial.div_mod_by_monic (5 * X^3 + k * X^2 + 5 * X - 22) (3 * X - 4) = (p, q) ∧ (q.degree < (3 * X - 4).degree) ∧ q = 10

theorem find_k_correct : find_k (-41/4) := 
by
  sorry

end find_k_correct_l570_570749


namespace length_of_largest_edge_l570_570906

theorem length_of_largest_edge {a b : ℝ} (h : b > a) :
  ∃ (x y z : ℝ), 3 right_dihedral_angles x y z ∧ 
  (∃ (AB CD : ℝ), AB = x ∧ CD = z ∧ dist (midpoint AB) (midpoint CD) = a) ∧
  (∃ (AC BD : ℝ), AC = x ∧ BD = z ∧ dist (midpoint AC) (midpoint BD) = b) ∧
  length_of_largest_edge x y z = sqrt (3 * a^2 + b^2) := sorry

end length_of_largest_edge_l570_570906


namespace compare_apothems_l570_570235

-- Definitions of conditions
def diagonal_eq_three_times_perimeter (s t : ℝ) : Prop := s * Real.sqrt 2 = 9 * t
def apothem_eq_area (s : ℝ) : Prop := s ^ 2 = 2 * s

-- Definition of apothems
def square_apothem (s : ℝ) : ℝ := s / 2
def triangle_apothem (t : ℝ) : ℝ := (Real.sqrt 3 / 6) * t

-- Target ratio
def apothem_ratio (s t : ℝ) : ℝ := (square_apothem s) / (triangle_apothem t)

-- Theorem to prove
theorem compare_apothems (s t : ℝ) 
  (h1 : diagonal_eq_three_times_perimeter s t) 
  (h2 : apothem_eq_area s) : 
  apothem_ratio s t = (9 * Real.sqrt 6) / 4 := 
sorry

end compare_apothems_l570_570235


namespace alex_downhill_speed_l570_570348

theorem alex_downhill_speed:
  let s1 := 20 -- speed on flat ground in mph
  let t1 := 4.5 -- time on flat ground in hours
  let s2 := 12 -- speed uphill in mph
  let t2 := 2.5 -- time uphill in hours
  let D := 164 -- total distance in miles
  let Dw := 8 -- distance walked in miles
  let td := 1.5 -- time on downhill section in hours
  let s1d := s1 * t1 -- distance on flat ground
  let s2d := s2 * t2 -- distance uphill
  let d_before_puncture := s1d + s2d + 1.5 * s3
  d_before_puncture = D - Dw -> s3 = 24 :=
begin
  sorry
end

end alex_downhill_speed_l570_570348


namespace complex_plane_problem_l570_570829

noncomputable theory

def z₁ : ℂ := -3 + 4 * complex.i
def z₂ : ℂ := 1 + 7 * complex.i
def z₃ : ℂ := 3 - 4 * complex.i
def O : ℂ := 0

def is_first_quadrant (z : ℂ) : Prop := 
  0 < z.re ∧ 0 < z.im

def symmetry_about_origin (z1 z2 : ℂ) : Prop :=
  z1 = -z2

def are_perpendicular (v1 v2 : ℂ) : Prop :=
  (v1 * complex.conj v2).im = 0

theorem complex_plane_problem :
  is_first_quadrant z₂ ∧
  symmetry_about_origin z₁ z₃ ∧
  are_perpendicular z₁ (z₂ + z₃) :=
by
  sorry

end complex_plane_problem_l570_570829


namespace repeating_decimal_difference_l570_570937

theorem repeating_decimal_difference :
  let F := (0.841 : ℚ) + 841 / 999 in
  let reduced_F := Rat.mkPnat 841 999 in
  reduced_F.num - reduced_F.den = -4 :=
by
  sorry

end repeating_decimal_difference_l570_570937


namespace problem_f_g_comp_sum_l570_570944

-- Define the functions
def f (x : ℚ) : ℚ := (4 * x^2 + 6 * x + 9) / (x^2 - 2 * x + 5)
def g (x : ℚ) : ℚ := x - 2

-- Define the statement we want to prove
theorem problem_f_g_comp_sum (x : ℚ) (h : x = 2) : f (g x) + g (f x) = 36 / 5 := by
  sorry

end problem_f_g_comp_sum_l570_570944


namespace necessary_and_sufficient_condition_l570_570452

theorem necessary_and_sufficient_condition (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + b^2 ≥ 2 * a * b) ↔ (a/b + b/a ≥ 2) :=
sorry

end necessary_and_sufficient_condition_l570_570452


namespace inscribed_circle_radius_l570_570654

theorem inscribed_circle_radius (DE DF EF : ℝ) (h1 : DE = 26) (h2 : DF = 15) (h3 : EF = 17) :
    ∃ r, r = Real.sqrt 6 ∧
    let s := (DE + DF + EF) / 2 in
    let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)) in
    K = r * s :=
by
  sorry

end inscribed_circle_radius_l570_570654


namespace veronica_reroll_probability_is_correct_l570_570638

noncomputable def veronica_reroll_probability : ℚ :=
  let P := (5 : ℚ) / 54
  P

theorem veronica_reroll_probability_is_correct :
  veronica_reroll_probability = (5 : ℚ) / 54 := sorry

end veronica_reroll_probability_is_correct_l570_570638


namespace number_of_tiles_l570_570336

-- Define the conditions
def hall_condition (x y : ℝ) : Prop := 4 * |x| + 5 * |y| ≤ 20

-- Define the area of a tile
def tile_area : ℝ := (1 * (5 / 4)) / 2

-- Calculate the number of tiles required to cover the area of the hall
theorem number_of_tiles (A : ℝ) (H : ∀ (x y : ℝ), hall_condition x y → (A = 40)) : (40 / tile_area) = 64 :=
by
  sorry

end number_of_tiles_l570_570336


namespace circle_with_min_perimeter_l570_570067

theorem circle_with_min_perimeter :
  ∀ (x y : ℝ),
  (∃ (C : ℝ × ℝ), C = (1, 1)) →
  (∃ (P : ℝ × ℝ), P.1 + P.2 = 4 ∧ (x - 1)^2 + (y - 1)^2 = 2) →
  (∃ (r : ℝ), r = sqrt (2)) →
  (x - 1)^2 + (y - 1)^2 = 2 :=
by 
  sorry

end circle_with_min_perimeter_l570_570067


namespace number_of_nonequivalent_sets_even_l570_570948

variable (A : Finset ℕ)
variable (S : Finset ℕ)

def tiles (S A : Finset ℕ) : Prop :=
  ∃ (m : ℕ) (T : Finset ℕ),
    (∀ i < m, (S + T) ⊆ A) ∧
    ((A.univ = (Finset.range m).sum (λ i, S + T))

noncomputable def number_of_nonequivalent_tiling_sets (A : Finset ℕ) : ℕ :=
  -- The hypothetical function that returns the number of nonequivalent sets that tile A
  sorry 

theorem number_of_nonequivalent_sets_even (h1 : A.card > 1) (h2 : tiles S A) : 
  Even (number_of_nonequivalent_tiling_sets A) :=
sorry

end number_of_nonequivalent_sets_even_l570_570948


namespace largest_sum_is_8_over_15_l570_570724

theorem largest_sum_is_8_over_15 :
  max ((1 / 3) + (1 / 6)) (max ((1 / 3) + (1 / 7)) (max ((1 / 3) + (1 / 5)) (max ((1 / 3) + (1 / 9)) ((1 / 3) + (1 / 8))))) = 8 / 15 :=
sorry

end largest_sum_is_8_over_15_l570_570724


namespace magnitude_of_F3_l570_570330

-- Define the problem conditions
def point_mass_in_equilibrium {V : Type*} [InnerProductSpace ℝ V] (F1 F2 F3 : V) : Prop :=
  F1 + F2 + F3 = 0

def angle_between_forces (F1 F2 : ℝ) : Prop :=
  real.angle F1 F2 = real.pi / 3 -- 120 degrees in radians is pi/3

def magnitudes (F1 F2 : V) : Prop :=
  ∥F1∥ = 6 ∧ ∥F2∥ = 6

-- State the theorem to be proved
theorem magnitude_of_F3 {V : Type*} [InnerProductSpace ℝ V]
  (F1 F2 F3 : V)
  (h_eq : point_mass_in_equilibrium F1 F2 F3)
  (h_angle : angle_between_forces F1 F2)
  (h_magnitudes : magnitudes F1 F2) :
  ∥F3∥ = 6 :=
sorry

end magnitude_of_F3_l570_570330


namespace ana_wins_probability_l570_570174

noncomputable def prob_ana_wins (n : ℕ) : ℚ :=
  (1 / 2) ^ (4 * n + 1)

theorem ana_wins_probability : 
  (Σ' n, prob_ana_wins n) = 1 / 30 := 
by sorry

end ana_wins_probability_l570_570174


namespace market_survey_l570_570210

theorem market_survey (X Y Z : ℕ) (h1 : X / Y = 3)
  (h2 : X / Z = 2 / 3) (h3 : X = 60) : X + Y + Z = 170 :=
by
  sorry

end market_survey_l570_570210


namespace chocolate_bars_partial_boxes_l570_570499

-- Define the total number of bars for each type
def totalA : ℕ := 853845
def totalB : ℕ := 537896
def totalC : ℕ := 729763

-- Define the box capacities for each type
def capacityA : ℕ := 9
def capacityB : ℕ := 11
def capacityC : ℕ := 15

-- State the theorem we want to prove
theorem chocolate_bars_partial_boxes :
  totalA % capacityA = 4 ∧
  totalB % capacityB = 3 ∧
  totalC % capacityC = 8 :=
by
  -- Proof omitted for this task
  sorry

end chocolate_bars_partial_boxes_l570_570499


namespace geom_seq_common_ratio_l570_570600

noncomputable def log_custom_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem geom_seq_common_ratio (a : ℝ) :
  let u₁ := a + log_custom_base 2 3
  let u₂ := a + log_custom_base 4 3
  let u₃ := a + log_custom_base 8 3
  u₂ / u₁ = u₃ / u₂ →
  u₂ / u₁ = 1 / 3 :=
by
  intro h
  sorry

end geom_seq_common_ratio_l570_570600


namespace isosceles_triangle_count_l570_570355

-- Define a type for our 2D grid point
structure Point2D where
  x : ℕ
  y : ℕ

-- Check if a set of three points forms an isosceles triangle
def is_isosceles (p1 p2 p3 : Point2D) : Bool :=
  let d1 := (p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2
  let d2 := (p2.x - p3.x) ^ 2 + (p2.y - p3.y) ^ 2
  let d3 := (p3.x - p1.x) ^ 2 + (p3.y - p1.y) ^ 2
  d1 = d2 ∨ d2 = d3 ∨ d3 = d1

-- Define the 9 grid points in a 3x3 grid
def grid_points : List Point2D :=
  [ ⟨0, 0⟩, ⟨0, 1⟩, ⟨0, 2⟩,
    ⟨1, 0⟩, ⟨1, 1⟩, ⟨1, 2⟩,
    ⟨2, 0⟩, ⟨2, 1⟩, ⟨2, 2⟩ ]

-- Calculate the number of isosceles triangles
def count_isosceles : Nat :=
  List.countp (λ t : (Point2D × Point2D × Point2D), is_isosceles t.1 t.2 t.3)
    (List.triples grid_points)

theorem isosceles_triangle_count : count_isosceles = 36 := by
  -- Proof will be added here
  sorry

end isosceles_triangle_count_l570_570355


namespace inscribed_circle_radius_DE_F_EF_l570_570648

theorem inscribed_circle_radius_DE_F_EF (DE DF EF : ℝ) (hDE : DE = 26) (hDF : DF = 15) (hEF : EF = 17) : 
    let s := (DE + DF + EF) / 2 in
    let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)) in
    let r := K / s in
    r = 3 * Real.sqrt 2 :=
by 
  sorry

end inscribed_circle_radius_DE_F_EF_l570_570648


namespace proj_z_2v_l570_570539

variables (v w : Vector ℝ 2) (z : Vector ℝ 2)
noncomputable def proj (u v : Vector ℝ 2) : Vector ℝ 2 :=
  ((u.inner v) / (v.inner v)) • v

-- Conditions
axiom proj_vw : proj v w = ⟨4, 1⟩
def z := 3 • w - 4 • v

-- Proof Problem
theorem proj_z_2v : proj (2 • v) z = ((-112) / (z.normSq)) • z :=
sorry

end proj_z_2v_l570_570539


namespace grade_assignments_count_l570_570331

theorem grade_assignments_count (n : ℕ) (g : ℕ) (h : n = 15) (k : g = 4) : g^n = 1073741824 :=
by
  sorry

end grade_assignments_count_l570_570331


namespace time_to_catch_up_l570_570347

variables (v a y t : ℝ)
hypothesis ha : a > 0
-- Ace's and Flash's distance conditions
def ace_distance (t : ℝ) : ℝ := v * t
def flash_distance (t : ℝ) : ℝ := (1 / 2) * a * t^2 + y

theorem time_to_catch_up (v a y : ℝ) (ha : a > 0) : t = (v + real.sqrt (v^2 + 2 * a * y)) / a :=
  sorry

end time_to_catch_up_l570_570347


namespace average_age_of_4_students_l570_570597

theorem average_age_of_4_students :
  let total_age_15 := 15 * 15
  let age_15th := 25
  let total_age_9 := 16 * 9
  (total_age_15 - total_age_9 - age_15th) / 4 = 14 :=
by
  sorry

end average_age_of_4_students_l570_570597


namespace find_AD_l570_570972

theorem find_AD (A B C D : Type) [Real ABCD] {a b c AD : ℝ} 
  (hAB : AB = a) 
  (hCD : CD = b) 
  (hBC : BC = c) 
  (h_similar1: ∠DBC = ∠BAD) 
  (h_similar2: ∠ABD = ∠BCD) :
  AD = b * (a / c)^2 := by 
  sorry

end find_AD_l570_570972


namespace negation_of_universal_proposition_l570_570108

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 > Real.log x)) ↔ (∃ x : ℝ, x^2 ≤ Real.log x) :=
by
  sorry

end negation_of_universal_proposition_l570_570108


namespace order_of_values_l570_570840

-- Definitions of the conditions given in the problem
variable (f : ℝ → ℝ)

-- Condition 1: f is increasing on [4, 8]
def increasing_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2, 4 ≤ x1 → x1 < x2 → x2 ≤ 8 → (f x1 - f x2) / (x1 - x2) > 0

-- Condition 2: f(x + 4) = -f(x)
def odd_shift_symmetry (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 4) = -f x

-- Condition 3: y = f(x + 4) is even 
def even_about_neg_four (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 4) = f (-x - 4)

-- Statement using these conditions
theorem order_of_values :
  increasing_on_interval f →
  odd_shift_symmetry f →
  even_about_neg_four f →
  let a := f 6
  let b := f 11
  let c := f 2017
  b < a ∧ a < c :=
by {
  intro h_incr h_symm h_even,
  let a := f 6,
  let b := f 11,
  let c := f 2017,
  sorry
}

end order_of_values_l570_570840


namespace unique_solution_of_log_eq_exp_l570_570503

theorem unique_solution_of_log_eq_exp (a : ℝ) (h : 1 < a) :
  (∃ x : ℝ, a^x = log a x) ∧ (∀ x y : ℝ, a^x = log a x → a^y = log a y → x = y) → 
  (a = real.exp (1/real.exp 1)) :=
by admit

end unique_solution_of_log_eq_exp_l570_570503


namespace max_red_socks_l570_570688

-- Define r (red socks), b (blue socks), t (total socks), with the given constraints
def socks_problem (r b t : ℕ) : Prop :=
  t = r + b ∧
  t ≤ 2023 ∧
  (2 * r * (r - 1) + 2 * b * (b - 1)) = 2 * 5 * t * (t - 1)

-- State the theorem that the maximum number of red socks is 990
theorem max_red_socks : ∃ r b t, socks_problem r b t ∧ r = 990 :=
sorry

end max_red_socks_l570_570688


namespace area_isosceles_triangle_l570_570383

-- Define the sides of the triangle
def side_a : ℝ := 10
def side_b : ℝ := 11
def side_c : ℝ := 11

-- Prove that the area of this triangle is 20 * real.sqrt 6
theorem area_isosceles_triangle (a b c : ℝ) (h₀ : a = side_a) (h₁ : b = side_b) (h₂ : c = side_c) :
  ∃ area : ℝ, area = 20 * real.sqrt 6 :=
by
  use 20 * real.sqrt 6
  sorry

end area_isosceles_triangle_l570_570383


namespace apply_composite_functions_l570_570543

def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := x / 4
def f_inv (x : ℝ) : ℝ := (x - 3) / 2
def g_inv (x : ℝ) : ℝ := 4 * x

theorem apply_composite_functions :
  f (g_inv (f_inv (f (g (f_inv (f (g (f 11)))))))) = 55 :=
by
  sorry

end apply_composite_functions_l570_570543


namespace trigonometric_identity_l570_570132

theorem trigonometric_identity (θ : ℝ) (h : 2 * (Real.cos θ) + (Real.sin θ) = 0) :
  Real.cos (2 * θ) + 1/2 * Real.sin (2 * θ) = -1 := 
sorry

end trigonometric_identity_l570_570132


namespace ratio_part_to_third_fraction_l570_570567

variable (P N : ℕ)

-- Definitions based on conditions
def one_fourth_one_third_P_eq_14 : Prop := (1/4 : ℚ) * (1/3 : ℚ) * (P : ℚ) = 14

def forty_percent_N_eq_168 : Prop := (40/100 : ℚ) * (N : ℚ) = 168

-- Theorem stating the required ratio
theorem ratio_part_to_third_fraction (h1 : one_fourth_one_third_P_eq_14 P) (h2 : forty_percent_N_eq_168 N) : 
  (P : ℚ) / ((1/3 : ℚ) * (N : ℚ)) = 6 / 5 := by
  sorry

end ratio_part_to_third_fraction_l570_570567


namespace megan_popsicles_l570_570968

variable (t_rate : ℕ) (t_hours : ℕ)

def popsicles_eaten (rate: ℕ) (hours: ℕ) : ℕ :=
  60 * hours / rate

theorem megan_popsicles : popsicles_eaten 20 5 = 15 := by
  sorry

end megan_popsicles_l570_570968


namespace johnson_oldest_child_age_l570_570598

/-- The average age of the three Johnson children is 10 years. 
    The two younger children are 6 years old and 8 years old. 
    Prove that the age of the oldest child is 16 years. -/
theorem johnson_oldest_child_age :
  ∃ x : ℕ, (6 + 8 + x) / 3 = 10 ∧ x = 16 :=
by
  sorry

end johnson_oldest_child_age_l570_570598


namespace find_JK_l570_570735

noncomputable def FG : ℝ := 4.5
noncomputable def GH : ℝ := 6
noncomputable def FJ : ℝ := 3
noncomputable def angle_FGH : ℝ
noncomputable def angle_FJK : ℝ
axiom angle_eq : angle_FGH = angle_FJK
axiom ratio_eq : GH / FJ = FG / JK

theorem find_JK :
  (GH / FJ = FG / JK) → JK = 2.25 := 
by
  sorry

end find_JK_l570_570735


namespace question1_question2_l570_570910

-- Definitions for the triangle ABC
variables {A B C a b c : ℝ}
-- Ensure the triangle is acute and the angles and sides hold
axiom acute_triangle (A B C : ℝ) (a b c : ℝ) : A + B + C = π ∧ 0 < A < π / 2 ∧ 0 < B < π / 2 ∧ 0 < C < π / 2
axiom side_conditions (a b c : ℝ) : a > 0 ∧ b > 0 ∧ c > 0
-- Given equation
axiom given_eq (A B C : ℝ) : √((1 - cos(2 * C)) / 2) + sin(B - A) = 2 * sin(2 * A)

-- Prove \frac{a}{b} = \frac{1}{2}
theorem question1 : ∀ {A B C a b c : ℝ}, 
  acute_triangle A B C a b c → side_conditions a b c → given_eq A B C → a / b = 1 / 2 :=
by sorry

-- Given that AB is the longest side
axiom longest_side {a b c : ℝ} (h : a < b ≤ c)

-- Prove 0 < cos(C) ≤ 1/4
theorem question2 : ∀ {A B C a b c : ℝ}, 
  acute_triangle A B C a b c → side_conditions a b c → given_eq A B C → longest_side a b c → 0 < cos C ∧ cos C ≤ 1/4 :=
by sorry

end question1_question2_l570_570910


namespace distance_from_Q_to_BC_l570_570591

-- Definitions for the problem
structure Square :=
(A B C D : ℝ × ℝ)
(side_length : ℝ)

def P : (ℝ × ℝ) := (3, 6)
def circle1 (x y : ℝ) : Prop := (x - 3)^2 + (y - 6)^2 = 9
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 25
def side_BC (x y : ℝ) : Prop := x = 6

-- Lean proof statement
theorem distance_from_Q_to_BC (Q : ℝ × ℝ) (hQ1 : circle1 Q.1 Q.2) (hQ2 : circle2 Q.1 Q.2) :
  Exists (fun d : ℝ => Q.1 = 6 ∧ Q.2 = d) := sorry

end distance_from_Q_to_BC_l570_570591


namespace smallest_positive_x_l570_570787

theorem smallest_positive_x 
  (x : ℝ) 
  (H : 0 < x) 
  (H_eq : ⌊x^2⌋ - x * ⌊x⌋ = 10) : 
  x = 131 / 11 :=
sorry

end smallest_positive_x_l570_570787


namespace equal_chessboard_numbers_l570_570566

theorem equal_chessboard_numbers (n : ℕ) (board : ℕ → ℕ → ℕ) 
  (mean_property : ∀ (x y : ℕ), board x y = (board (x-1) y + board (x+1) y + board x (y-1) + board x (y+1)) / 4) : 
  ∀ (x y : ℕ), board x y = board 0 0 :=
by
  -- Proof not required
  sorry

end equal_chessboard_numbers_l570_570566


namespace find_bc_div_a_l570_570469

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x + 2 * Real.cos x + 1

variable (a b c : ℝ)

def satisfied (x : ℝ) : Prop := a * f x + b * f (x - c) = 1

theorem find_bc_div_a (ha : ∀ x, satisfied a b c x) : (b * Real.cos c / a) = -1 := 
by sorry

end find_bc_div_a_l570_570469


namespace time_to_office_l570_570280

theorem time_to_office (S T : ℝ) (h1 : T > 0) (h2 : S > 0) 
    (h : S * (T + 15) = (4/5) * S * T) :
    T = 75 := by
  sorry

end time_to_office_l570_570280


namespace smallest_solution_l570_570775

def smallest_positive_real_x : ℝ :=
  (131 : ℝ) / 11

theorem smallest_solution (x : ℝ) (hx : 0 < x) (H : ⌊x^2⌋ - x * ⌊x⌋ = 10) : x = smallest_positive_real_x :=
  sorry

end smallest_solution_l570_570775


namespace intersection_of_A_and_B_l570_570500

theorem intersection_of_A_and_B :
  let A := {x : ℝ | x > 0}
  let B := {x : ℝ | x^2 - 2*x - 3 < 0}
  (A ∩ B) = {x : ℝ | 0 < x ∧ x < 3} := by
  sorry

end intersection_of_A_and_B_l570_570500


namespace find_pairs_l570_570973

noncomputable def possibleValues (α β : ℝ) : Prop :=
  (∃ (n l : ℤ), α = 2*n*Real.pi ∧ β = -(Real.pi/3) + 2*l*Real.pi) ∨
  (∃ (n l : ℤ), α = 2*n*Real.pi ∧ β = (Real.pi/3) + 2*l*Real.pi)

theorem find_pairs (α β : ℝ) (h1 : Real.sin (α - β) = Real.sin α - Real.sin β)
  (h2 : Real.cos (α - β) = Real.cos α - Real.cos β) :
  possibleValues α β :=
sorry

end find_pairs_l570_570973


namespace smallest_t_for_circle_sin_l570_570604

theorem smallest_t_for_circle_sin (t : ℝ) : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → ∃ r : ℝ, r = sin θ) ∧ 
  (∀ q : ℝ, 0 ≤ q ∧ q < t → (sin q ≠ sin (q + π))) ∧ 
  (∀ q : ℝ, 0 ≤ q < t → (cos q ≠ cos (q + π))) → 
  t = π :=
by
  sorry

end smallest_t_for_circle_sin_l570_570604


namespace area_A1B1C1D1_l570_570970

variables {A B C D A1 B1 C1 D1 : Type}
variables [convex_quadrilateral : convex_quadrilateral ABCD] (S : ℝ)
variables (H1 : (vectorDA1 = 2 * vectorDA))
variables (H2 : (vectorAB1 = 2 * vectorAB))
variables (H3 : (vectorBC1 = 2 * vectorBC))
variables (H4 : (vectorCD1 = 2 * vectorCD))
variables (area_S : quadrilateralArea ABCD = S)

theorem area_A1B1C1D1 : quadrilateralArea A1B1C1D1 = 5 * S :=
sorry

end area_A1B1C1D1_l570_570970


namespace smallest_base_max_l570_570632

noncomputable theory

-- Define the scalene triangle and its properties
variable {D E F : Type*} [triangle : scalene_triangle D E F]

-- Define the altitudes and the corresponding smallest base height condition
variable (hD : altitude D E F = 9)
variable (hE : altitude E F D = 3)
variable (hF : altitude F D E = 6)
variable (h_min_base : height_smallest_base D E F = 6)

-- Declare the theorem matching the mathematical proof problem
theorem smallest_base_max (D E F : Type*) [triangle : scalene_triangle D E F] (hD : altitude D E F = 9) (hE : altitude E F D = 3) (hF : altitude F D E = 6) (h_min_base : height_smallest_base D E F = 6) : 
  smallest_base_max_possible D E F = 3 := 
sorry

end smallest_base_max_l570_570632


namespace problem_l570_570111

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem problem (A_def : A = {-1, 0, 1}) : B = {0, 1} :=
by sorry

end problem_l570_570111


namespace probability_different_ranks_l570_570059

theorem probability_different_ranks :
  let cards := {card | card = 'A' ∨ card = 'K' ∨ card = 'K' ∨ card = 'Q' ∨ card = 'Q'}
  let different_rank_draws := {draw | ∃ c1 c2, c1 ≠ c2 ∧ c1 ∈ cards ∧ c2 ∈ cards}
  let total_draws := {draw | ∃ c1 c2, c1 ∈ cards ∧ c2 ∈ cards}
  (finset.card different_rank_draws / finset.card total_draws) = 4 / 5 := by
  sorry

end probability_different_ranks_l570_570059


namespace arithmetic_expression_evaluation_l570_570721

noncomputable theory

theorem arithmetic_expression_evaluation : 25.3 - 0.432 + 1.25 = 26.118 :=
by
-- skipping the proof
sorry

end arithmetic_expression_evaluation_l570_570721


namespace niraek_donut_holes_1078_l570_570359

theorem niraek_donut_holes_1078
  (r_niraek : ℕ)
  (r_theo : ℕ)
  (r_akshaj : ℕ)
  (H1 : r_niraek = 5)
  (H2 : r_theo = 7)
  (H3 : r_akshaj = 11) :
  let A_niraek := 4 * Real.pi * r_niraek^2,
      A_theo := 4 * Real.pi * r_theo^2,
      A_akshaj := 4 * Real.pi * r_akshaj^2 in
  ∃ (L : ℕ), L = Nat.lcm (Nat.lcm A_niraek.nat_abs A_theo.nat_abs) A_akshaj.nat_abs ∧ 
  (L / A_niraek.nat_abs) = 1078 :=
by
  sorry

end niraek_donut_holes_1078_l570_570359


namespace count_nonnegative_integers_l570_570936

-- Define the function f recursively as per the problem conditions
def f : ℕ → ℤ
| 0       := 1
| (n + 1) := let m := n + 1 in
              512 ^ (m / 10) * f (m / 10)

-- Prove the theorem where the required condition holds
theorem count_nonnegative_integers (n : ℕ) :
  (finset.range 1112).card = 1112 ∧
  (∀ m, m ≤ 1111 → 
    let f_m := f m in
    (⌈real.logb 16 (f_m.to_real)⌉.nat_abs ≤ 2500)) :=
sorry

end count_nonnegative_integers_l570_570936


namespace sum_of_values_l570_570443

theorem sum_of_values (N : ℝ) (R : ℝ) (h : N ≠ 0) (h_eq : N + 5 / N = R) : N = R := 
sorry

end sum_of_values_l570_570443


namespace math_problem_l570_570357

theorem math_problem
  (p q r s : ℕ)
  (hpq : p^3 = q^2)
  (hrs : r^4 = s^3)
  (hrp : r - p = 25) :
  s - q = 73 := by
  sorry

end math_problem_l570_570357


namespace sum_binom_P_eq_l570_570306

-- Definition of P_k(x) as given in the problem statement.
def P (k : ℕ) (x : ℝ) : ℝ := (finset.range k).sum (λ i, x^i)

theorem sum_binom_P_eq (x : ℝ) (n : ℕ) (hn : 0 < n) :
  (finset.range n).sum (λ k, nat.choose n (k + 1) * P (k + 1) x) = 
  2^(n - 1) * P n ((1 + x) / 2) :=
sorry

end sum_binom_P_eq_l570_570306


namespace part1_part2_l570_570859

noncomputable def f (a b c x : ℝ) : ℝ := a + b * cos x + c * sin x

theorem part1 (a b c : ℝ) (b_pos : b > 0) (h1 : f a b c 0 = 1) (h2 : f a b c (π / 2) = 1) :
  ∀ k : ℤ, ∀ x : ℝ, 2 * k * π + π / 4 ≤ x ∧ x ≤ 2 * k * π + 5 * π / 4 → 
  (f a b c x) is decreasing x :=
sorry

theorem part2 (a b c : ℝ) (h1 : f a b c 0 = 1) (h2 : f a b c (π / 2) = 1) 
  (h3 : ∀ x : ℝ, (0 < x ∧ x < π / 2) → |f a b c x| ≤ 2) :
  -sqrt 2 ≤ a ∧ a ≤ 4 + 3 * sqrt 2 :=
sorry

end part1_part2_l570_570859


namespace wooden_statue_cost_l570_570263

def cost_of_wooden_statue (W : ℝ) : Prop :=
  let income_stone := 200 in
  let income_wooden := 20 * W in
  let total_income := income_stone + income_wooden in
  let after_tax_income := 0.9 * total_income in
  after_tax_income = 270

theorem wooden_statue_cost : ∃ W, cost_of_wooden_statue W ∧ W = 5 :=
by
  use 5
  unfold cost_of_wooden_statue
  simp
  linarith

end wooden_statue_cost_l570_570263


namespace total_students_is_correct_l570_570564

-- Define the number of students in each class based on the conditions
def number_of_students_finley := 24
def number_of_students_johnson := (number_of_students_finley / 2) + 10
def number_of_students_garcia := 2 * number_of_students_johnson
def number_of_students_smith := number_of_students_finley / 3
def number_of_students_patel := (3 / 4) * (number_of_students_finley + number_of_students_johnson + number_of_students_garcia)

-- Define the total number of students in all five classes combined
def total_number_of_students := 
  number_of_students_finley + 
  number_of_students_johnson + 
  number_of_students_garcia +
  number_of_students_smith + 
  number_of_students_patel

-- The theorem statement to prove
theorem total_students_is_correct : total_number_of_students = 166 := by
  sorry

end total_students_is_correct_l570_570564


namespace train_takes_approximately_10_3228_seconds_to_pass_tree_l570_570341

noncomputable def train_time_to_pass_tree (L : ℝ) (u : ℝ) (a : ℝ) : ℝ :=
  let u_m_per_s := u * (1000 / 3600)
  let discriminant := u_m_per_s ^ 2 + 2 * a * L
  let t1 := (-u_m_per_s + Real.sqrt discriminant) / a
  let t2 := (-u_m_per_s - Real.sqrt discriminant) / a
  max t1 t2

theorem train_takes_approximately_10_3228_seconds_to_pass_tree :
  train_time_to_pass_tree 250 50 2 ≈ 10.3228 :=
by
  sorry

end train_takes_approximately_10_3228_seconds_to_pass_tree_l570_570341


namespace log_equation_solution_l570_570588

theorem log_equation_solution :
  ∃ x : ℝ, log (3 * x) - 4 * log 9 = 3 ∧ x = 2187000 :=
by
  use 2187000
  sorry

end log_equation_solution_l570_570588


namespace arithmetic_mean_l570_570042

theorem arithmetic_mean (x b : ℝ) (h : x ≠ 0) : 
  (1 / 2) * ((2 + (b / x)) + (2 - (b / x))) = 2 :=
by sorry

end arithmetic_mean_l570_570042


namespace sum_of_powers_of_3_4_7_l570_570217

theorem sum_of_powers_of_3_4_7 (n : ℕ) :
  ∃ (k_3 k_4 k_7 : ℕ → ℕ) (h3 : ∀ i j, i ≠ j → k_3 i ≠ k_3 j) (h4 : ∀ i j, i ≠ j → k_4 i ≠ k_4 j) (h7 : ∀ i j, i ≠ j → k_7 i ≠ k_7 j),
    n = (Finset.range n).sum (λ i, 3^(k_3 i)) +
        (Finset.range n).sum (λ j, 4^(k_4 j)) +
        (Finset.range n).sum (λ l, 7^(k_7 l)) :=
by
  sorry

end sum_of_powers_of_3_4_7_l570_570217


namespace monthly_income_of_P_l570_570300

theorem monthly_income_of_P (P Q R : ℕ) (h1 : P + Q = 10100) (h2 : Q + R = 12500) (h3 : P + R = 10400) : 
  P = 4000 := 
by 
  sorry

end monthly_income_of_P_l570_570300


namespace average_possible_k_l570_570852

theorem average_possible_k (k : ℕ) (r1 r2 : ℕ) (h : r1 * r2 = 24) (h_pos : r1 > 0 ∧ r2 > 0) (h_eq_k : r1 + r2 = k) : 
  (25 + 14 + 11 + 10) / 4 = 15 :=
by 
  sorry

end average_possible_k_l570_570852


namespace dog_catches_fox_at_distance_l570_570686

def initial_distance : ℝ := 30
def dog_leap_distance : ℝ := 2
def fox_leap_distance : ℝ := 1
def dog_leaps_per_time_unit : ℝ := 2
def fox_leaps_per_time_unit : ℝ := 3

noncomputable def dog_speed : ℝ := dog_leaps_per_time_unit * dog_leap_distance
noncomputable def fox_speed : ℝ := fox_leaps_per_time_unit * fox_leap_distance
noncomputable def relative_speed : ℝ := dog_speed - fox_speed
noncomputable def time_to_catch := initial_distance / relative_speed
noncomputable def distance_dog_runs := time_to_catch * dog_speed

theorem dog_catches_fox_at_distance :
  distance_dog_runs = 120 :=
  by sorry

end dog_catches_fox_at_distance_l570_570686


namespace find_f_neg_5pi_over_6_l570_570069

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined_on_R : ∀ x : ℝ, ∃ y : ℝ, f y = f x
axiom f_periodic : ∀ x : ℝ, f (x + (3 * Real.pi / 2)) = f x
axiom f_on_interval : ∀ x : ℝ, 0 ≤ x → x ≤ Real.pi → f x = Real.cos x

theorem find_f_neg_5pi_over_6 : f (-5 * Real.pi / 6) = -1 / 2 := 
by 
  -- use the axioms to prove the result 
  sorry

end find_f_neg_5pi_over_6_l570_570069


namespace expression_for_a_six_minus_inv_a_six_l570_570017

theorem expression_for_a_six_minus_inv_a_six (a : ℂ) :
  a^6 - a^(-6) = (a^2 - 1 / a^2) * (a^4 + 1 + 1 / a^4) :=
by
  sorry

end expression_for_a_six_minus_inv_a_six_l570_570017


namespace minimum_value_expression_l570_570548

theorem minimum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  4 * x^4 + 16 * y^4 + 36 * z^4 + 9 / (x * y * z) ≥ 24 :=
by
  sorry

end minimum_value_expression_l570_570548


namespace five_letter_words_with_consonant_l570_570482

theorem five_letter_words_with_consonant :
  let letters := { 'A', 'B', 'C', 'D', 'E', 'F' } in
  let consonants := { 'B', 'C', 'D', 'F' } in
  let vowels := { 'A', 'E' } in
  let total_5_letter_words := 6^5 in
  let total_vowel_only_words := 2^5 in
  total_5_letter_words - total_vowel_only_words = 7744 :=
by
  sorry

end five_letter_words_with_consonant_l570_570482


namespace range_of_m_eq_l570_570612

theorem range_of_m_eq (m: ℝ) (x: ℝ) :
  (m+1 = 0 ∧ 4 > 0) ∨ 
  ((m + 1 > 0) ∧ ((m^2 - 2 * m - 3)^2 - 4 * (m + 1) * (-m + 3) < 0)) ↔ 
  (m ∈ Set.Icc (-1 : ℝ) 1 ∪ Set.Ico (1 : ℝ) 3) := 
sorry

end range_of_m_eq_l570_570612


namespace unique_players_in_chemistry_l570_570715

variable {U : Type} [Fintype U]
variable (B C P : Finset U)

-- Conditions
def condition1 : B.card = 15 := sorry
def condition2 : C.card = 10 := sorry
def condition3 : P.card = 5 := sorry
def condition4 : (B ∩ C ∩ P).card = 2 := sorry

-- Proof problem
theorem unique_players_in_chemistry :
  C.card = 10 :=
by sorry

end unique_players_in_chemistry_l570_570715


namespace prove_f_solution_equivalence_l570_570090

variables (a b c x : ℝ) 

-- Given the functions f(x) and g(x)
def f (x : ℝ) : ℝ := a * x^2 + b * x + c
def g (x : ℝ) : ℝ := a * x + b

-- Given the condition that the maximum value of g(x) in the interval [-1, 1] is 2
axiom max_g_x : ∀ x, -1 ≤ x ∧ x ≤ 1 → g x ≤ 2

-- Define the specific function according to the solution
def f_solution (x : ℝ) : ℝ := 2 * x^2 - 1

-- Prove that the specific function satisfies the conditions
theorem prove_f_solution_equivalence : f_solution x = f x :=
by
  sorry

end prove_f_solution_equivalence_l570_570090


namespace circle_cd_touches_ab_iff_bc_parallel_ad_l570_570934

def is_tangent_to_line (diameter : ℝ) (center : Point) (line : Line) : Prop :=
  -- define the condition where a circle with the given diameter and center is tangent to the given line

variable (A B C D : Point)
variable (circle_diameter_ab circle_diameter_cd : ℝ)
variable (circle_center_ab circle_center_cd : Point)
variable (line_ab line_cd : Line)

-- Condition: Let ABCD be a convex quadrilateral
-- Note: In Lean, we assume this property generally holds or is asserted elsewhere

-- Condition: The circle with diameter AB touches the line CD
axiom convexe_quad (quad : Quadrilateral A B C D) : Prop
axiom circle_ab_touches_cd : is_tangent_to_line circle_diameter_ab circle_center_ab line_cd

-- Question: Prove the equivalence
theorem circle_cd_touches_ab_iff_bc_parallel_ad
  (quad : Quadrilateral A B C D)
  (touch_cd : is_tangent_to_line circle_diameter_ab circle_center_ab line_cd ) :
  (is_tangent_to_line(circle_diameter_cd circle_center_cd line_ab) ↔ Parallel (Segment B C) (Segment A D)) :=
sorry

end circle_cd_touches_ab_iff_bc_parallel_ad_l570_570934


namespace constant_term_is_19_l570_570056

theorem constant_term_is_19 (x y C : ℝ) 
  (h1 : 7 * x + y = C) 
  (h2 : x + 3 * y = 1) 
  (h3 : 2 * x + y = 5) : 
  C = 19 :=
sorry

end constant_term_is_19_l570_570056


namespace count_odd_digits_base4_157_l570_570388

def base_repr (n b : ℕ) : List ℕ :=
  if b ≤ 1 then [n] else (List.unfoldr (λ x, if x = 0 then none else some (x % b, x / b)) n).reverse

def count_odd_digits (l : List ℕ) : ℕ :=
  l.countp (λ d, d % 2 = 1)

theorem count_odd_digits_base4_157 : count_odd_digits (base_repr 157 4) = 3 := by
  sorry

end count_odd_digits_base4_157_l570_570388


namespace converse_angles_complements_l570_570602

theorem converse_angles_complements (α β : ℝ) (h : ∀γ : ℝ, α + γ = 90 ∧ β + γ = 90 → α = β) : 
  ∀ δ, α + δ = 90 ∧ β + δ = 90 → α = β :=
by 
  sorry

end converse_angles_complements_l570_570602


namespace petya_vacation_days_l570_570571

-- Defining the conditions
def total_days : ℕ := 90

def swims (d : ℕ) : Prop := d % 2 = 0
def shops (d : ℕ) : Prop := d % 3 = 0
def solves_math (d : ℕ) : Prop := d % 5 = 0

def does_all (d : ℕ) : Prop := swims d ∧ shops d ∧ solves_math d

def does_any_task (d : ℕ) : Prop := swims d ∨ shops d ∨ solves_math d

-- "Pleasant" days definition: swims, not shops, not solves math
def is_pleasant_day (d : ℕ) : Prop := swims d ∧ ¬shops d ∧ ¬solves_math d
-- "Boring" days definition: does nothing
def is_boring_day (d : ℕ) : Prop := ¬does_any_task d

-- Theorem stating the number of pleasant and boring days
theorem petya_vacation_days :
  (∃ pleasant_days : Finset ℕ, pleasant_days.card = 24 ∧ ∀ d ∈ pleasant_days, is_pleasant_day d)
  ∧ (∃ boring_days : Finset ℕ, boring_days.card = 24 ∧ ∀ d ∈ boring_days, is_boring_day d) :=
by
  sorry

end petya_vacation_days_l570_570571


namespace number_of_grade2_students_l570_570613

theorem number_of_grade2_students (ratio1 ratio2 ratio3 : ℕ) (total_students : ℕ) (ratio_sum : ratio1 + ratio2 + ratio3 = 12)
  (total_sample_size : total_students = 240) : 
  total_students * ratio2 / (ratio1 + ratio2 + ratio3) = 80 :=
by
  have ratio1_val : ratio1 = 5 := sorry
  have ratio2_val : ratio2 = 4 := sorry
  have ratio3_val : ratio3 = 3 := sorry
  rw [ratio1_val, ratio2_val, ratio3_val] at ratio_sum
  rw [ratio1_val, ratio2_val, ratio3_val]
  exact sorry

end number_of_grade2_students_l570_570613


namespace parabola_equation_l570_570854

theorem parabola_equation (a : ℝ) : 
(∀ x y : ℝ, y = x → y = a * x^2)
∧ (∃ P : ℝ × ℝ, P = (2, 2) ∧ P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) 
  → A = (x₁, y₁) ∧ B = (x₂, y₂) ∧ y₁ = x₁ ∧ y₂ = x₂ ∧ x₂ = x₁ → 
  ∃ f : ℝ × ℝ, f.fst ≠ 0 ∧ f.snd = 0) →
  a = (1 : ℝ) / 7 := 
sorry

end parabola_equation_l570_570854


namespace popsicle_sticks_left_l570_570371

-- Conditions definitions
def initial_budget : ℕ := 10
def max_popsicles : ℕ := 30
def molds_cost : ℕ := 3
def sticks_cost : ℕ := 1
def total_sticks : ℕ := 100
def orange_juice_cost : ℕ := 2
def orange_juice_popsicles : ℕ := 20
def apple_juice_cost : ℕ := 3
def apple_juice_popsicles : ℕ := 30
def grape_juice_cost : ℕ := 4
def grape_juice_popsicles : ℕ := 40

-- Question and Answer as a theorem to be proven in Lean
theorem popsicle_sticks_left : (initial_budget - (molds_cost + sticks_cost)) = 6 → 
                              max_popsicles = 30 → 
                              grape_juice_popsicles = 40 →
                              total_sticks - max_popsicles = 70 :=
begin
  sorry
end

end popsicle_sticks_left_l570_570371


namespace count_colorings_l570_570524

-- Define the number of disks
def num_disks : ℕ := 6

-- Define colorings with constraints: 2 black, 2 white, 2 blue considering rotations and reflections as equivalent
def valid_colorings : ℕ :=
  18  -- This is the result obtained using Burnside's Lemma as shown in the solution

theorem count_colorings : valid_colorings = 18 := by
  sorry

end count_colorings_l570_570524


namespace vector_magnitude_and_trig_identity_l570_570479

theorem vector_magnitude_and_trig_identity (a b : ℝ × ℝ) (α : ℝ)
  (h_cond1 : a = (4, 5 * Real.cos α))
  (h_cond2 : b = (3, -4 * Real.tan α))
  (h_cond3 : 0 < α ∧ α < Real.pi / 2)
  (h_orthogonal : a.1 * b.1 + a.2 * b.2 = 0) :
  (∥((4 : ℝ), 5 * Real.cos α) - (3, -4 * Real.tan α)∥ = 5 * Real.sqrt 2) ∧
  (Real.cos (3 * Real.pi / 2 + α) - Real.sin (α - Real.pi) = 6 / 5) := by
  sorry

end vector_magnitude_and_trig_identity_l570_570479


namespace distinct_integers_count_l570_570744

-- Define the sequence based on the problem statement
def sequence : List ℕ := List.map (λ n : ℕ => (n ^ 2) / 500) (List.filter (λ n : ℕ => n % 2 = 1) (List.range 1000))

-- Define the statement to prove that the length of the distinct integers in the sequence is 469
theorem distinct_integers_count : (sequence.toFinset.card = 469) :=
by
  sorry

end distinct_integers_count_l570_570744


namespace number_of_students_with_D_l570_570141

theorem number_of_students_with_D 
  (num_students : ℕ)
  (A_ratio : ℚ) (B_ratio : ℚ) (C_ratio : ℚ)
  (total_students : num_students = 100)
  (A_fraction : A_ratio = 1/5)
  (B_fraction : B_ratio = 1/4)
  (C_fraction : C_ratio = 1/2)
  :
  let A_students := A_ratio * num_students,
      B_students := B_ratio * num_students,
      C_students := C_ratio * num_students,
      D_students := num_students - (A_students + B_students + C_students) in
  D_students = 5 :=
by
  -- intermediate calculations and steps can be detailed here
  sorry

end number_of_students_with_D_l570_570141


namespace number_of_proper_subsets_eq_127_l570_570246

def valid_set := {x : ℕ | 4 ≤ x ∧ x ≤ 10}

theorem number_of_proper_subsets_eq_127 : (2 ^ Fintype.card valid_set) - 1 = 127 := by
  have h1 : Fintype.card valid_set = 7 := by
    -- To be proved separately: the cardinality of valid_set is 7
    sorry
  simp only [h1, pow_succ, pow_zero, mul_one]
  norm_num
  sorry

end number_of_proper_subsets_eq_127_l570_570246


namespace sale_in_fifth_month_l570_570322

-- Define the sale amounts and average sale required.
def sale_first_month : ℕ := 7435
def sale_second_month : ℕ := 7920
def sale_third_month : ℕ := 7855
def sale_fourth_month : ℕ := 8230
def sale_sixth_month : ℕ := 6000
def average_sale_required : ℕ := 7500

-- State the theorem to determine the sale in the fifth month.
theorem sale_in_fifth_month
  (s1 s2 s3 s4 s6 avg : ℕ)
  (h1 : s1 = sale_first_month)
  (h2 : s2 = sale_second_month)
  (h3 : s3 = sale_third_month)
  (h4 : s4 = sale_fourth_month)
  (h6 : s6 = sale_sixth_month)
  (havg : avg = average_sale_required) :
  s1 + s2 + s3 + s4 + s6 + x = 6 * avg →
  x = 7560 :=
by
  sorry

end sale_in_fifth_month_l570_570322


namespace radius_of_large_circle_correct_l570_570630

noncomputable def radius_of_large_circle : ℝ :=
  1 + 1 / ( 1 / sqrt 3 )

noncomputable def expected_radius : ℝ :=
  (3 + 2 * sqrt 3) / 3

theorem radius_of_large_circle_correct :
  radius_of_large_circle = expected_radius :=
by
  sorry

end radius_of_large_circle_correct_l570_570630


namespace new_device_significant_improvement_l570_570682

def old_device_samples : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
def new_device_samples : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

def mean (l : List ℝ) : ℝ := l.sum / l.length
def variance (l : List ℝ) : ℝ := (l.map (λ x, (x - mean l) ^ 2)).sum / l.length

def x̄ : ℝ := mean old_device_samples
def ȳ : ℝ := mean new_device_samples
def s₁_sq : ℝ := variance old_device_samples
def s₂_sq : ℝ := variance new_device_samples
def improvement_threshold : ℝ := 2 * Math.sqrt((s₁_sq + s₂_sq) / 10)

def significant_improvement : Prop := (ȳ - x̄) ≥ improvement_threshold

theorem new_device_significant_improvement : significant_improvement :=
by
  sorry

end new_device_significant_improvement_l570_570682


namespace first_step_eliminate_four_l570_570707

-- Define the conditions
def total_staff : ℕ := 624
def survey_percentage : ℕ := 10
def fraction_threshold : ℕ := 4

-- Theorem statement that embodies the problem
theorem first_step_eliminate_four :
  ∃ (n : ℕ), n = 4 ∧ systematic_sampling (total_staff - n) (survey_percentage * total_staff / 100) → systematic_sampling total_staff (survey_percentage * total_staff / 100) :=
begin
  -- Define the conditions as per problem statement
  let total := total_staff,
  let percentage := survey_percentage,
  let threshold := fraction_threshold,
  -- Existence of a value that makes the sampling valid
  use threshold,
  split,
  -- Prove value is 4
  exact rfl,
  -- Rest of the proof (Skipped)
  sorry
end

end first_step_eliminate_four_l570_570707


namespace fixed_point_l570_570071

-- Definitions for the conditions and the problem statement
def linear_function (k b x : ℝ) : ℝ := k * x + b

def condition (k b : ℝ) : Prop := 3 * k - b = 2

theorem fixed_point (k b : ℝ) (h : condition k b) : linear_function k b (-3) = -2 :=
by 
  rw [linear_function, condition] at h 
  subst h
  sorry

end fixed_point_l570_570071


namespace projection_is_circumcenter_l570_570507

theorem projection_is_circumcenter
  (P A B C O : Point)
  (h1 : Angle P A O = Angle P B O)
  (h2 : Angle P B O = Angle P C O)
  (h3 : PO = PO)
  (h4 : POA = 90°)
  (h5 : POB = 90°)
  (h6 : POC = 90°) : 
  is_circumcenter O A B C :=
begin
  sorry
end

end projection_is_circumcenter_l570_570507


namespace four_is_descendant_of_every_natural_l570_570203

def successor (x : ℕ) : ℕ :=
  if x % 10 == 0 then x / 10
  else if x % 10 == 4 then x / 10
  else 2 * x

def is_descendant (x y : ℕ) : Prop :=
  ∃ n : ℕ, (successor^[n] x = y)

theorem four_is_descendant_of_every_natural :
  ∀ x : ℕ, is_descendant x 4 :=
begin
  sorry
end

end four_is_descendant_of_every_natural_l570_570203


namespace intersection_point_of_lines_l570_570747

theorem intersection_point_of_lines :
  let x := -12 / 7
  let y := 22 / 7
  (3 * y = -2 * x + 6) ∧ (-2 * y = 6 * x + 4) :=
by
  let x := -12 / 7
  let y := 22 / 7
  have h1 : (3 : ℝ) * y = -2 * (x : ℝ) + 6 := by
    linarith
  have h2 : -(2 : ℝ) * y = 6 * (x : ℝ) + 4 := by
    linarith
  exact ⟨h1, h2⟩

end intersection_point_of_lines_l570_570747


namespace integer_solutions_system_l570_570037

theorem integer_solutions_system (x y z t : ℤ) :
  (xz - 2 * yt = 3 ∧ xt + yz = 1) ↔
  (x, y, z, t) ∈ { (1, 0, 3, 1), (-1, 0, 3, 1), (1, 0, -3, -1), (-1, 0, -3, -1),
                   (1, 0, 3, -1), (-1, 0, 3, -1), (1, 0, -3, 1), (-1, 0, -3, 1),
                   (3, 1, 1, 0), (-3, 1, 1, 0), (3, -1, 1, 0), (-3, -1, 1, 0),
                   (3, 1, -1, 0), (-3, 1, -1, 0), (3, -1, -1, 0), (-3, -1, -1, 0) } :=
by sorry

end integer_solutions_system_l570_570037


namespace inscribed_circle_radius_l570_570653

theorem inscribed_circle_radius (DE DF EF : ℝ) (h1 : DE = 26) (h2 : DF = 15) (h3 : EF = 17) :
    ∃ r, r = Real.sqrt 6 ∧
    let s := (DE + DF + EF) / 2 in
    let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)) in
    K = r * s :=
by
  sorry

end inscribed_circle_radius_l570_570653


namespace plan_cost_comparison_l570_570149

def base_fare : ℝ := 10
def distance_fare : ℝ := 1.5
def return_fare : ℝ := 2.5

def f (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 3 then base_fare else
if h : 3 < x ∧ x ≤ 5 then base_fare + distance_fare * (x - 3) else
base_fare + distance_fare * 2 + return_fare * (x - 5)

def g (x : ℝ) : ℝ :=
let k := (⌊x / 5⌋ : ℕ) in
if h : 5 * k < x ∧ x ≤ 5 * k + 3 then 13 * k + base_fare else
13 * k + base_fare + distance_fare * (x - 5 * k - 3)

theorem plan_cost_comparison (x : ℝ) (k : ℕ) (h : 5 * k < x ∧ x ≤ 5 * k + 5) :
  (0 < x ∧ x ≤ 5 → f(x) = g(x)) ∧
  (x > 5 → f(x) < g(x)) :=
by {
  sorry
}

end plan_cost_comparison_l570_570149


namespace neg_universal_proposition_l570_570867

theorem neg_universal_proposition :
  (¬ ∀ x : ℝ, 2^x = 5) ↔ ∃ x : ℝ, 2^x ≠ 5 :=
by sorry

end neg_universal_proposition_l570_570867


namespace find_prob_B_l570_570759

variable {Ω : Type} {P : Ω → Prop}

-- Defining events as predicates on the type Ω
variable (A B C : Ω → Prop)

-- Assumptions given in the problem
axiom mutual_independence : IsIndependent P [A, B, C]
axiom P_A_and_B : P (λ ω, A ω ∧ B ω) = 1/6
axiom P_not_B_and_C : P (λ ω, ¬B ω ∧ C ω) = 1/8
axiom P_A_and_B_and_not_C : P (λ ω, A ω ∧ B ω ∧ ¬C ω) = 1/8

-- Final statement to prove
theorem find_prob_B : P (λ ω, B ω) = 1/2 := sorry

end find_prob_B_l570_570759


namespace problem_statement_l570_570496

theorem problem_statement (a b c x y z : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < x) (h5 : 0 < y) (h6 : 0 < z)
  (h7 : a^2 + b^2 + c^2 = 16) (h8 : x^2 + y^2 + z^2 = 49) (h9 : a * x + b * y + c * z = 28) : 
  (a + b + c) / (x + y + z) = 4 / 7 := 
by
  sorry

end problem_statement_l570_570496


namespace sum_of_nu_with_lcm_120_l570_570657

theorem sum_of_nu_with_lcm_120 :
  (∑ ν in {ν : ℕ | Nat.lcm ν 24 = 120}.toFinset, ν) = 300 := by
  sorry

end sum_of_nu_with_lcm_120_l570_570657


namespace tom_total_time_l570_570275

def total_time_reading_examining (chapters : ℕ) (pages_per_chapter : ℕ) (illustrations_per_chapter : ℕ) 
  (time_per_page : ℕ) (time_per_illustration : ℕ) : ℕ :=
  let total_pages := chapters * pages_per_chapter
  let total_reading_time := total_pages * time_per_page
  let total_illustrations := chapters * illustrations_per_chapter
  let total_examination_time := total_illustrations * time_per_illustration
  total_reading_time + total_examination_time

theorem tom_total_time :
  total_time_reading_examining 25 12 3 10 5 = 3375 :=
by
  calc
  total_time_reading_examining 25 12 3 10 5
      = (25 * 12 * 10) + (25 * 3 * 5) : by rw [total_time_reading_examining]
  ... = 3000 + 375         : by norm_num
  ... = 3375               : by norm_num

end tom_total_time_l570_570275


namespace inscribed_tetrahedron_volume_l570_570445

theorem inscribed_tetrahedron_volume :
  ∀ (R : ℝ), R = 6 → ∃ (V : ℝ), V = 32 * Real.sqrt 3 :=
by
  intro R hr
  use 32 * Real.sqrt 3
  exact Eq.symm hr

end inscribed_tetrahedron_volume_l570_570445


namespace find_a12_l570_570824

variable (a : ℕ → ℝ) (q : ℝ)
variable (h1 : ∀ n, a (n + 1) = a n * q)
variable (h2 : abs q > 1)
variable (h3 : a 1 + a 6 = 2)
variable (h4 : a 3 * a 4 = -15)

theorem find_a12 : a 11 = -25 / 3 :=
by sorry

end find_a12_l570_570824


namespace odd_digits_in_base4_of_157_l570_570399

theorem odd_digits_in_base4_of_157 : 
  let b4 := 2 * 4^3 + 1 * 4^2 + 3 * 4^1 + 1 * 4^0 in
  b4 = 157 → (nat.digits 4 157).countp (λ d, d % 2 = 1) = 3 := 
by
  intros
  sorry

end odd_digits_in_base4_of_157_l570_570399


namespace laura_rental_cost_l570_570930

def rental_cost_per_day : ℝ := 30
def driving_cost_per_mile : ℝ := 0.25
def days_rented : ℝ := 3
def miles_driven : ℝ := 300

theorem laura_rental_cost : rental_cost_per_day * days_rented + driving_cost_per_mile * miles_driven = 165 := by
  sorry

end laura_rental_cost_l570_570930


namespace eccentricity_hyperbola_eq_sqrt2_l570_570455

-- Definitions and conditions
def hyperbola (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  ∃ x y : ℝ, P = (x, y) ∧ (x^2 / a^2) - (y^2 / b^2) = 1 ∧ a > 0 ∧ b > 0

def is_perpendicular_to_x_axis (P F2 : ℝ × ℝ) : Prop :=
  F2.2 = P.2

def sin_angle_PF1F2 (F1 F2 P : ℝ × ℝ) : Prop :=
  let θ := real.angle F1 P - real.angle F2 P in
  real.sin θ = 1 / 3

-- Goal
theorem eccentricity_hyperbola_eq_sqrt2
  (a b : ℝ) (P F1 F2 : ℝ × ℝ)
  (h_hyperbola : hyperbola a b P)
  (h_perpendicular : is_perpendicular_to_x_axis P F2)
  (h_sin_angle : sin_angle_PF1F2 F1 F2 P)
  : let e := (real.sqrt (a^2 + b^2)) / a in e = real.sqrt 2 := sorry

end eccentricity_hyperbola_eq_sqrt2_l570_570455


namespace triangle_circle_fill_l570_570912

theorem triangle_circle_fill (A B C D : ℕ) : 
  (A ≠ B) → (A ≠ C) → (A ≠ D) → (B ≠ C) → (B ≠ D) → (C ≠ D) →
  (A = 6 ∨ A = 7 ∨ A = 8 ∨ A = 9) →
  (B = 6 ∨ B = 7 ∨ B = 8 ∨ B = 9) →
  (C = 6 ∨ C = 7 ∨ C = 8 ∨ C = 9) →
  (D = 6 ∨ D = 7 ∨ D = 8 ∨ D = 9) →
  (A + B + 1 + 8 =  A + 4 + 3 + 7) →  (D + 4 + 2 + 5 = 5 + 1 + 8 + B) →
  (5 + 1 + 8 + 6 = 5 + C + 7 + 4 ) →
  (A = 6) ∧ (B = 8) ∧ (C = 7) ∧ (D = 9) := by
  sorry

end triangle_circle_fill_l570_570912


namespace farmer_goats_sheep_unique_solution_l570_570320

theorem farmer_goats_sheep_unique_solution:
  ∃ g h : ℕ, 0 < g ∧ 0 < h ∧ 28 * g + 30 * h = 1200 ∧ h > g :=
by
  sorry

end farmer_goats_sheep_unique_solution_l570_570320


namespace jovana_shells_l570_570531

theorem jovana_shells (initial_shells : ℕ) (added_shells : ℕ) (total_shells : ℕ) 
  (h_initial : initial_shells = 5) (h_added : added_shells = 12) :
  total_shells = 17 :=
by
  sorry

end jovana_shells_l570_570531


namespace TriangleIsRight_iff_max_area_l570_570078

namespace Geometry

variables {a b c : ℝ}
variables {A B C : ℝ} -- Angles in radians

-- Definitions and main theorems for Proof 1:
def sin_law_sinA (a c : ℝ) : ℝ := a / c
def sin_law_sinB (b c : ℝ) : ℝ := b / c
def sin_law_sinC : ℝ := 1

def cos_law_cosA (a b c : ℝ) : ℝ := (b^2 + c^2 - a^2) / (2 * b * c)
def cos_law_cosB (a b c : ℝ) : ℝ := (a^2 + c^2 - b^2) / (2 * a * c)

theorem TriangleIsRight_iff (h : sin_law_sinA a c + sin_law_sinB b c = (cos_law_cosA a b c + cos_law_cosB a b c) * sin_law_sinC):
  a^2 + b^2 = c^2 := sorry

-- Definitions and main theorems for Proof 2:
def perimeter (a b c : ℝ) : ℝ := a + b + c

def right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

def area (a b : ℝ) : ℝ := 0.5 * a * b

theorem max_area (h1 : right_triangle a b c) (h2 : perimeter a b c = 1 + sqrt 2) : 
  area a b ≤ 0.25 := sorry

end Geometry

end TriangleIsRight_iff_max_area_l570_570078


namespace total_tickets_sold_l570_570230

theorem total_tickets_sold :
  ∃(S : ℕ), 4 * S + 6 * 388 = 2876 ∧ S + 388 = 525 :=
by
  sorry

end total_tickets_sold_l570_570230


namespace curved_quadrilateral_circumscribed_l570_570272

open Point
open Circle
open Arc
open Line
open Segment
open Ray
open CurvedQuadrilateral

-- Define points
def A : Point := {x := 0, y := 0}
def B : Point := {x := 4, y := 0}
def C : Point := {x := 10, y := 0}

-- Define circular arcs
def γ1 : Arc := { center := {x := 5, y := -7}, radius := 7}
def γ2 : Arc := { center := {x := 5, y := 7}, radius := 7}
def γ3 : Arc := { center := {x := 5, y := 14}, radius := 11}

-- Define rays
def h1 : Ray := { origin := B, direction := {x := 1.07, y := 6 }}
def h2 : Ray := { origin := B, direction := {x := 5, y := 7 }}
def h3 : Ray := { origin := B, direction := {x := 9, y := 6 }}

-- Define intersection points
def V (i j : ℕ) : Point :=
  if j = 1 then if i = 1 then γ1.to_curve.inter (h1.to_line) else if i = 2 then γ1.to_curve.inter (h2.to_line) else γ1.to_curve.inter (h3.to_line)
  else if j = 2 then if i = 1 then γ2.to_curve.inter (h1.to_line) else if i = 2 then γ2.to_curve.inter (h2.to_line) else γ2.to_curve.inter (h3.to_line)
  else if j = 3 then if i = 1 then γ3.to_curve.inter (h1.to_line) else if i = 2 then γ3.to_curve.inter (h2.to_line) else γ3.to_curve.inter (h3.to_line)
  else {x := 0, y := 0}  -- Placeholder

def curved_quad_circumscribed (quad : CurvedQuadrilateral) : Prop :=
  ∃ (circle : Circle), circle.tangent_to (quad.seg_1) ∧ circle.tangent_to (quad.arc_1) ∧ 
                        circle.tangent_to (quad.seg_2) ∧ circle.tangent_to (quad.arc_2)

def quad1 := CurvedQuadrilateral.mk (V 1 1) (V 2 1) (V 2 2) (V 1 2) 
def quad2 := CurvedQuadrilateral.mk (V 1 2) (V 2 2) (V 2 3) (V 1 3) 
def quad3 := CurvedQuadrilateral.mk (V 2 1) (V 3 1) (V 3 2) (V 2 2)
def quad4 := CurvedQuadrilateral.mk (V 2 2) (V 3 2) (V 3 3) (V 2 3)

theorem curved_quadrilateral_circumscribed :
   (curved_quad_circumscribed quad1) → 
   (curved_quad_circumscribed quad2) → 
   (curved_quad_circumscribed quad3) → 
   (curved_quad_circumscribed quad4) :=
by sorry

end curved_quadrilateral_circumscribed_l570_570272


namespace point_P_characterization_l570_570537

open EuclideanGeometry

noncomputable def point (P: Type*) := P
noncomputable def vec3 (x y z : ℝ) : point ℝ := (x, y, z)

def A : point ℝ := vec3 10 0 0
def B : point ℝ := vec3 0 (-6) 0
def C : point ℝ := vec3 0 0 8
def D : point ℝ := vec3 0 0 0

def P : point ℝ := vec3 5 (-3) 4

theorem point_P_characterization :
  dist A P = dist B P ∧
  dist B P = dist C P ∧
  dist C P = 2 * dist D P :=
by
  sorry

end point_P_characterization_l570_570537


namespace find_intersection_l570_570113

def A : Set ℝ := {x | abs (x + 1) = x + 1}

def B : Set ℝ := {x | x^2 + x < 0}

def intersection (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∈ B}

theorem find_intersection : intersection A B = {x | -1 < x ∧ x < 0} :=
by
  sorry

end find_intersection_l570_570113


namespace f_at_2_f_a_eq_1_l570_570105

def f (x : ℝ) : ℝ :=
  if x >= 0 then 2^x - 1 else x^2 - x - 1

-- Proof requirement 1: f(2) = 3
theorem f_at_2 : f 2 = 3 :=
sorry

-- Proof requirement 2: If f(a) = 1, then a = 1 or a = -1
theorem f_a_eq_1 (a : ℝ) (h : f a = 1) : a = 1 ∨ a = -1 :=
sorry

end f_at_2_f_a_eq_1_l570_570105


namespace unique_square_side_l570_570627

noncomputable def curve (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem unique_square_side (a b c : ℝ) (h: ∃! s : ℝ × ℝ, curve a b c s.fst = s.snd ∧ curve a b c (-s.fst) = -s.snd ∧ s ∈ set_of_squares_with_side_length) : 
  ∃ d : ℝ, (d ^ 4 = 72) :=
sorry

end unique_square_side_l570_570627


namespace ana_wins_probability_l570_570173

noncomputable def prob_ana_wins (n : ℕ) : ℚ :=
  (1 / 2) ^ (4 * n + 1)

theorem ana_wins_probability : 
  (Σ' n, prob_ana_wins n) = 1 / 30 := 
by sorry

end ana_wins_probability_l570_570173


namespace area_isosceles_triangle_l570_570382

-- Define the sides of the triangle
def side_a : ℝ := 10
def side_b : ℝ := 11
def side_c : ℝ := 11

-- Prove that the area of this triangle is 20 * real.sqrt 6
theorem area_isosceles_triangle (a b c : ℝ) (h₀ : a = side_a) (h₁ : b = side_b) (h₂ : c = side_c) :
  ∃ area : ℝ, area = 20 * real.sqrt 6 :=
by
  use 20 * real.sqrt 6
  sorry

end area_isosceles_triangle_l570_570382


namespace ana_wins_probability_l570_570172

noncomputable def prob_ana_wins (n : ℕ) : ℚ :=
  (1 / 2) ^ (4 * n + 1)

theorem ana_wins_probability : 
  (Σ' n, prob_ana_wins n) = 1 / 30 := 
by sorry

end ana_wins_probability_l570_570172


namespace hyperbola_eccentricity_l570_570863

open Real

variables (a b : ℝ) (c : ℝ)
variable (hyperbola : set (ℝ × ℝ)) -- The set representing hyperbola 
variable (circle : set (ℝ × ℝ)) -- The set representing circle
variable (focus : ℝ × ℝ) -- Left focus of the hyperbola
variable (tangent_point : ℝ × ℝ) -- Point E
variable (hyperbola_point : ℝ × ℝ) -- Point P
variable (origin : ℝ × ℝ := (0, 0)) -- Origin point

noncomputable def hyperbola_eq (p : ℝ × ℝ) : Prop :=
  (p.1 / a) ^ 2 - (p.2 / b) ^ 2 = 1

noncomputable def circle_eq (p : ℝ × ℝ) : Prop :=
  p.1 ^ 2 + p.2 ^ 2 = a ^ 2

noncomputable def condition1 : Prop := a > 0 ∧ b > 0

noncomputable def condition2 : Prop := focus = (-c, 0)

noncomputable def condition3 : Prop := 
  circle_eq tangent_point ∧ hyperbola_eq tangent_point

noncomputable def condition4 : Prop := 
  ∃ p: ℝ × ℝ, hyperbola_eq p ∧ ∃ α : ℝ, p = α • tangent_point

noncomputable def condition5 (OE OF OP : ℝ × ℝ) : Prop :=
  OE = (OP + OF) / 2

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  sqrt (1 + (b / a) ^ 2)

theorem hyperbola_eccentricity : 
  condition1 a b ∧ condition2 c ∧ condition3 a b ∧ condition4  ∧ condition5 tangent_point focus hyperbola_point → 
  eccentricity a b = sqrt 5 :=
sorry

end hyperbola_eccentricity_l570_570863


namespace inscribed_circle_radius_DE_F_EF_l570_570646

theorem inscribed_circle_radius_DE_F_EF (DE DF EF : ℝ) (hDE : DE = 26) (hDF : DF = 15) (hEF : EF = 17) : 
    let s := (DE + DF + EF) / 2 in
    let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)) in
    let r := K / s in
    r = 3 * Real.sqrt 2 :=
by 
  sorry

end inscribed_circle_radius_DE_F_EF_l570_570646


namespace sally_balance_fraction_l570_570225

def gold_card_limit (G : ℝ) : ℝ := G
def platinum_card_limit (G : ℝ) : ℝ := 2 * G
def gold_card_balance (G x : ℝ) : ℝ := x * G
def platinum_card_balance (G : ℝ) : ℝ := 1/6 * (2 * G)
def transfer_balance (G x : ℝ) : ℝ := (1/6 * (2 * G)) + x * G
def platinum_unspent_limit : ℝ := 2/3

theorem sally_balance_fraction (G x : ℝ) (h : transfer_balance G x = (1 - platinum_unspent_limit) * (2 * G)) : x = 1/3 :=
by
  sorry

end sally_balance_fraction_l570_570225


namespace geom_seq_min_value_l570_570074

noncomputable def minimum_sum (m n : ℕ) (a : ℕ → ℝ) : ℝ :=
  if (a 7 = a 6 + 2 * a 5) ∧ (a m * a n = 16 * (a 1) ^ 2) ∧ (m > 0) ∧ (n > 0) then
    (1 / m) + (4 / n)
  else
    0

theorem geom_seq_min_value (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →
  a 7 = a 6 + 2 * a 5 →
  (∃ m n, a m * a n = 16 * (a 1) ^ 2 ∧ m > 0 ∧ n > 0) →
  (minimum_sum m n a = 3 / 2) := sorry

end geom_seq_min_value_l570_570074


namespace trig_identity_l570_570832

noncomputable def sin_cos_identity (α β : ℝ) : Prop :=
  (sin α = -3/5) → (cos β = 1) → sin (α + β) = -3/5

theorem trig_identity : ∀ α β : ℝ, sin_cos_identity α β :=
by
  intros α β
  intro h1
  intro h2
  have h3 : sin β = 0 :=
    by sorry  -- Proof of sin β from cos β = 1
  have h4 : cos α is determined by further conditions that can be derived, but assume it holds
  sorry  -- To be proven by using trigonometric identities and the given conditions

-- This will pass the compilers as long as syntax is correct and required libraries are imported

end trig_identity_l570_570832


namespace prob_intermediate_find_c_value_l570_570252

-- Definitions
def P (c : ℝ) (k : ℕ) : ℝ := c / (k * (k + 1))

def sum_prob (c : ℝ) : ℝ :=
  P c 1 + P c 2 + P c 3 + P c 4

-- Theorem to prove
theorem prob_intermediate 
  (c : ℝ) (H : sum_prob c = 1) :
  P c 1 + P c 2 = 5 / 6 :=
sorry

-- Additional statement for c value derived from H
theorem find_c_value 
  (c : ℝ) (H : sum_prob c = 1) :
  c = 5 / 4 :=
sorry

end prob_intermediate_find_c_value_l570_570252


namespace min_value_expr_l570_570542

theorem min_value_expr (a b : ℝ) (h : a * b > 0) : (a^4 + 4 * b^4 + 1) / (a * b) ≥ 4 := 
sorry

end min_value_expr_l570_570542


namespace find_f_of_neg2_l570_570502

theorem find_f_of_neg2 (a b c : ℝ) 
  (h : (4 * b^2 - 4 * a * c) - (4 * b^2 - 4 * a * c + 16 * b - 16 * a - 4 * c) = 24) : 
  a * 4 - b * 4 + c = 6 := by
suffices h1 : -16 * b + 16 * a + 4 * c = 24, by 
  have h2 : -16 * b + 16 * a + 4 * c = 4 * (4 * a - 4 * b + c), by ring 
  exact (mul_left_cancel₀ (norm_num.nonneg_of_nat 4 zero_lt_four) h2).mp h1
have : 4 * (4 * a - 4 * b + c) = -16 * b + 16 * a + 4 * c, by ring
linarith


end find_f_of_neg2_l570_570502


namespace total_cost_correct_without_food_tax_total_cost_correct_with_food_tax_l570_570207

def item_cost (cost: ℝ) (tax: ℝ) : ℝ := cost + cost * tax

def puppy_cost : ℝ := 150
def daily_food_consumption : ℝ := 1 / 3
def total_days : ℝ := 42
def total_food_consumption : ℝ := daily_food_consumption * total_days
def bag_food_content : ℝ := 3.5
def bag_food_cost : ℝ := 2
def total_bags_needed : ℝ := total_food_consumption / bag_food_content
def total_food_cost : ℝ := total_bags_needed * bag_food_cost
def total_food_cost_taxed : ℝ := total_food_cost + (total_food_cost * 0.06)

def leash_cost : ℝ := 15
def collar_cost : ℝ := 12
def collar_discount : ℝ := 0.10
def collar_cost_after_discount : ℝ := collar_cost - collar_cost * collar_discount
def dog_bed_cost : ℝ := 25
def tax_rate : ℝ := 0.06

def total_cost_without_food_tax : ℝ :=
  puppy_cost +
  total_food_cost +
  item_cost leash_cost tax_rate +
  item_cost collar_cost_after_discount tax_rate +
  item_cost dog_bed_cost tax_rate

def total_cost_with_food_tax : ℝ :=
  puppy_cost +
  total_food_cost_taxed +
  item_cost leash_cost tax_rate +
  item_cost collar_cost_after_discount tax_rate +
  item_cost dog_bed_cost tax_rate

theorem total_cost_correct_without_food_tax : total_cost_without_food_tax = 211.85 := by
  sorry

theorem total_cost_correct_with_food_tax : total_cost_with_food_tax = 212.33 := by
  sorry

end total_cost_correct_without_food_tax_total_cost_correct_with_food_tax_l570_570207


namespace five_letter_words_with_consonant_l570_570481

theorem five_letter_words_with_consonant :
  let letters := { 'A', 'B', 'C', 'D', 'E', 'F' } in
  let consonants := { 'B', 'C', 'D', 'F' } in
  let vowels := { 'A', 'E' } in
  let total_5_letter_words := 6^5 in
  let total_vowel_only_words := 2^5 in
  total_5_letter_words - total_vowel_only_words = 7744 :=
by
  sorry

end five_letter_words_with_consonant_l570_570481


namespace balls_initial_count_90_l570_570257

theorem balls_initial_count_90 (n : ℕ) (total_initial_balls : ℕ)
  (initial_green_balls : ℕ := 3 * n)
  (initial_yellow_balls : ℕ := 7 * n)
  (remaining_green_balls : ℕ := initial_green_balls - 9)
  (remaining_yellow_balls : ℕ := initial_yellow_balls - 9)
  (h_ratio_1 : initial_green_balls = 3 * n)
  (h_ratio_2 : initial_yellow_balls = 7 * n)
  (h_ratio_3 : remaining_green_balls * 3 = remaining_yellow_balls * 1)
  (h_total : total_initial_balls = initial_green_balls + initial_yellow_balls)
  : total_initial_balls = 90 := 
by
  sorry

end balls_initial_count_90_l570_570257


namespace part1_part2_l570_570858

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * (2 * (cos (x / 2))^2 + sin x) + b

theorem part1 (a : ℝ) (b : ℝ) : 
  a = -1 → ∀ k : ℤ, ∀ x : ℝ, 
  (2 * k * Real.pi + Real.pi / 4) ≤ x ∧ x ≤ (2 * k * Real.pi + 5 * Real.pi / 4) →
  Monotone (f a b) :=
sorry

theorem part2 (a b : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi → 5 ≤ f a b x ∧ f a b x ≤ 8) →
  (a = 3 * Real.sqrt 2 - 3 ∧ b = 5) ∨ (a = 3 - 3 * Real.sqrt 2 ∧ b = 8) :=
sorry

end part1_part2_l570_570858


namespace Lizette_average_above_94_l570_570558

noncomputable def Lizette_new_weighted_average
  (score3: ℝ) (avg3: ℝ) (weight3: ℝ) (score_new1 score_new2: ℝ) (weight_new: ℝ) :=
  let total_points3 := avg3 * 3
  let total_weight3 := 3 * weight3
  let total_points := total_points3 + score_new1 + score_new2
  let total_weight := total_weight3 + 2 * weight_new
  total_points / total_weight

theorem Lizette_average_above_94:
  ∀ (score3 avg3 weight3 score_new1 score_new2 weight_new: ℝ),
  score3 = 92 →
  avg3 = 94 →
  weight3 = 0.15 →
  score_new1 > 94 →
  score_new2 > 94 →
  weight_new = 0.20 →
  Lizette_new_weighted_average score3 avg3 weight3 score_new1 score_new2 weight_new > 94 :=
by
  intros score3 avg3 weight3 score_new1 score_new2 weight_new h1 h2 h3 h4 h5 h6
  sorry

end Lizette_average_above_94_l570_570558


namespace dylans_mom_hotdogs_l570_570878

theorem dylans_mom_hotdogs (hotdogs_total : ℕ) (helens_mom_hotdogs : ℕ) (dylans_mom_hotdogs : ℕ) 
  (h1 : hotdogs_total = 480) (h2 : helens_mom_hotdogs = 101) (h3 : hotdogs_total = helens_mom_hotdogs + dylans_mom_hotdogs) :
dylans_mom_hotdogs = 379 :=
by
  sorry

end dylans_mom_hotdogs_l570_570878


namespace ratio_simplified_l570_570369

variable (a b c : ℕ)
variable (n m p : ℕ) (h1 : n > 0) (h2 : m > 0) (h3 : p > 0)

theorem ratio_simplified (h_ratio : a^n = 3 * c^p ∧ b^m = 4 * c^p ∧ c^p = 7 * c^p) :
  (a^n + b^m + c^p) / c^p = 2 := sorry

end ratio_simplified_l570_570369


namespace find_real_x_l570_570038

theorem find_real_x (x : ℝ) : 
  (2 ≤ 3 * x / (3 * x - 7)) ∧ (3 * x / (3 * x - 7) < 6) ↔ (7 / 3 < x ∧ x < 42 / 15) :=
by
  sorry

end find_real_x_l570_570038


namespace smallest_m_for_triples_l570_570748

theorem smallest_m_for_triples (m : ℕ) 
  (h1 : ∃ e f g : ℕ, Nat.gcd (Nat.gcd e f) g = 12 
    ∧ m = Nat.lcm (Nat.lcm e f) g)
  (h2 : ∀ a b c : ℕ, (Nat.gcd (Nat.gcd a b) c = 12 
    ∧ Nat.lcm (Nat.lcm a b) c = m) ↔ ∃k E F G, E = a / 12 
    ∧ F = b / 12 ∧ G = c / 12 ∧ Nat.gcd (Nat.gcd E F) G = 1 
    ∧ Nat.lcm (Nat.lcm E F) G = m / 12)
  (h3 : ∑' (E : ℕ) (F : ℕ) (G : ℕ), (Nat.gcd (Nat.gcd E F) G = 1 
    ∧ Nat.lcm (Nat.lcm E F) G = m / 12) = 20000) 
  : m = 2 ^ 100 * 12 :=
sorry

end smallest_m_for_triples_l570_570748


namespace smallest_value_wawbwcwd_l570_570545

noncomputable def g (x : ℝ) : ℝ := x^4 + 10 * x^3 + 35 * x^2 + 50 * x + 24

theorem smallest_value_wawbwcwd (w1 w2 w3 w4 : ℝ) : 
  (∀ x : ℝ, g x = 0 ↔ x = w1 ∨ x = w2 ∨ x = w3 ∨ x = w4) →
  |w1 * w2 + w3 * w4| = 12 ∨ |w1 * w3 + w2 * w4| = 12 ∨ |w1 * w4 + w2 * w3| = 12 :=
by 
  sorry

end smallest_value_wawbwcwd_l570_570545


namespace total_flowers_is_288_l570_570354

-- Definitions from the Conditions in a)
def arwen_tulips : ℕ := 20
def arwen_roses : ℕ := 18
def elrond_tulips : ℕ := 2 * arwen_tulips
def elrond_roses : ℕ := 3 * arwen_roses
def galadriel_tulips : ℕ := 3 * elrond_tulips
def galadriel_roses : ℕ := 2 * arwen_roses

-- Total number of tulips
def total_tulips : ℕ := arwen_tulips + elrond_tulips + galadriel_tulips

-- Total number of roses
def total_roses : ℕ := arwen_roses + elrond_roses + galadriel_roses

-- Total number of flowers
def total_flowers : ℕ := total_tulips + total_roses

theorem total_flowers_is_288 : total_flowers = 288 :=
by
  -- Placeholder for proof
  sorry

end total_flowers_is_288_l570_570354


namespace bisect_segment_l570_570215

open EuclideanGeometry

variables {α : Type*} [MetricSpace α] [NormedAddCommGroup α] [NormedSpace ℝ α]

theorem bisect_segment
  {A B C D E P : α} {ω : Circle α}
  (hA : A ∈ ω)
  (hB : B ∈ ω)
  (hC : C ∈ ω)
  (hD : D ∈ ω)
  (hE : E ∈ ω)
  (hP : P ∉ ω)
  (tangentPB : IsTangent (Line.mk P B) ω)
  (tangentPD : IsTangent (Line.mk P D) ω)
  (collinearPAC : Collinear ({P, A, C} : Set α))
  (parallelDEAC : ∥Segment.mk D E∥ = ∥Segment.mk A C∥) :
  Midpoint (A + C) ((B + E) : α) :=
sorry -- proof to be filled

end bisect_segment_l570_570215


namespace problem_1_problem_2_l570_570819

-- Condition definitions
def f (x : ℝ) : ℝ := Real.log x
def g (x : ℝ) : ℝ := (1 / 3) * x^3 + (1 / 2) * x^2 + m * x + n

-- Proof goals
theorem problem_1 (m n : ℝ) : 
  (∀ (l : ℝ → ℝ), (∀ x, l x = x - 1) → (l 1 = 0) → (g 1 = 0) → ((deriv g) 1 = 1) → (m = -1) ∧ (n = 1 / 6)) :=
sorry

theorem problem_2 : 
  (∀ (h : ℝ → ℝ), (∀ x, h x = Real.log x - (deriv g x)) → (∀ x, 0 < x → h x ≤ (1 / 4) - Real.log 2)) :=
sorry

end problem_1_problem_2_l570_570819


namespace jessica_rate_increase_is_22_l570_570530

def jessica_success_rate_increase (initial_success_attempts : ℕ) (initial_attempts : ℕ) (next_attempts : ℕ) (success_rate_fraction : ℚ) : ℕ :=
  let new_success_attempts := (success_rate_fraction * next_attempts) in
  let total_success_attempts := initial_success_attempts + new_success_attempts in
  let total_attempts := initial_attempts + next_attempts in
  let new_success_rate := (total_success_attempts / total_attempts : ℚ) * 100 in
  let initial_success_rate := (initial_success_attempts / initial_attempts : ℚ) * 100 in
  Int.toNat (new_success_rate - initial_success_rate).round

theorem jessica_rate_increase_is_22 :
  jessica_success_rate_increase 4 10 16 (3 / 4) = 22 := by
    sorry

end jessica_rate_increase_is_22_l570_570530


namespace prob_x_gt_6y_plus_5_of_random_point_in_rectangle_l570_570574

theorem prob_x_gt_6y_plus_5_of_random_point_in_rectangle :
  let rect_area := (2010 : ℝ) * 2011
  let tri_area := (1/2) * 2005 * (2005 / 6)
  let prob := tri_area / rect_area
  prob = 287 / 1727 := 
by
  let rect_area := (2010 : ℝ) * 2011
  let tri_area := (1/2) * 2005 * (2005 / 6)
  let prob := tri_area / rect_area
  have h1 : tri_area = 2005^2 / 12 := by sorry
  have h2 : rect_area = 2010 * 2011 := by sorry
  have h3 : prob = (2005^2 / 12) / (2010 * 2011) := by sorry
  have h4 : (2005^2 / 12) / (2010 * 2011) = 287 / 1727 := by sorry
  exact h4

end prob_x_gt_6y_plus_5_of_random_point_in_rectangle_l570_570574


namespace mikes_salary_l570_570803

theorem mikes_salary
  (fred_salary : ℝ)
  (mike_salary_increase_percent : ℝ)
  (mike_salary_factor : ℝ)
  (fred_salary_val : fred_salary = 1000)
  (mike_salary_factor_val : mike_salary_factor = 10)
  (mike_salary_increase_val : mike_salary_increase_percent = 40)
  : (10000 * (1 + mike_salary_increase_percent / 100)) = 14000 := 
by
  rw [fred_salary_val, mike_salary_factor_val, mike_salary_increase_val]
  norm_num
  sorry  -- Proof omitted

end mikes_salary_l570_570803


namespace odd_number_diff_squares_unique_l570_570991

theorem odd_number_diff_squares_unique (n : ℕ) (h : 0 < n) : 
  ∃! (x y : ℤ), (2 * n + 1) = x^2 - y^2 :=
by {
  sorry
}

end odd_number_diff_squares_unique_l570_570991


namespace curve_is_ellipse_A_coords_distance_from_origin_to_line_AB_l570_570152

noncomputable def point := ℝ × ℝ

def origin : point := (0, 0)
def C : point := (-1, 0)
def D : point := (1, 0)

def on_curve_E (P : point) : Prop := 
  abs (P.1 + 1) + abs (P.1 - 1) = 2 * real.sqrt 3

def ellipse_E (x y : ℝ) : Prop := 
  x^2 / 3 + y^2 / 2 = 1

def ortho (A B : point) : Prop := 
  A.1 * B.1 + A.2 * B.2 = 0

def OA_ratio_OB (A B : point) : Prop := 
  real.norm A = real.sqrt 3 / 2 * real.norm B

def A_coordinates := (real.sqrt 30 / 10, 3 * real.sqrt 5 / 5)

theorem curve_is_ellipse : 
  ∀ (P : point), on_curve_E P ↔ ellipse_E P.1 P.2 := 
sorry

theorem A_coords :
  ∃ A B : point, ortho A B ∧ OA_ratio_OB A B ∧ A = A_coordinates := 
sorry

theorem distance_from_origin_to_line_AB :
  ∀ (A B : point), ortho A B ∧ ellipse_E A.1 A.2 ∧ ellipse_E B.1 B.2 → 
  ∃ d : ℝ, d = real.sqrt 30 / 5 :=
sorry

end curve_is_ellipse_A_coords_distance_from_origin_to_line_AB_l570_570152


namespace total_trip_time_l570_570532

-- Given definitions based on conditions
def distance_interstate := 120  -- miles
def distance_mountain := 40     -- miles
def distance_dirt := 5          -- miles

def speed_interstate := 2 * v   -- Julia's speed on interstate
def speed_mountain := v         -- Julia's speed on mountain road
def speed_dirt := v / 2         -- Julia's speed on dirt track

def time_interstate := 60       -- minutes
def v := 1                      -- speed on the mountain road (computed from the condition)

-- Function to calculate time taken on each road based on speed and distance
def time_on_road (distance : ℕ) (speed : ℕ) : ℕ :=
  distance / speed

-- We need to prove that total trip time is 110 minutes
theorem total_trip_time :
  let v := 1 in
  let time_interstate := distance_interstate / (2 * v) in
  let time_mountain := distance_mountain / v in
  let time_dirt := distance_dirt / (v / 2) in
  time_interstate + time_mountain + time_dirt = 110 :=
by
  -- Proof skipped for problem statement requirement
  sorry

end total_trip_time_l570_570532


namespace factorial_mod_prime_l570_570547

theorem factorial_mod_prime (p : ℕ) (hp : Nat.Prime p) :
  let half_sum := 
    List.sum (List.map (λ a, (Nat.factorial (p - 1)) / (a * (p - a))) (List.range (p / 2)))
  half_sum % p = 0 := by
  sorry

end factorial_mod_prime_l570_570547


namespace proper_subsets_count_S_l570_570248

/-- 
Define the set S consisting of natural numbers that satisfy the inequality 
-1 ≤ log_(1/4) 10 < -1/2.
-/
def S : Set ℕ := {x ∈ Set.univ | -1 ≤ Real.log 10 / Real.log (1/4) ∧ Real.log 10 / Real.log (1/4) < -1/2}

/-- 
The number of proper subsets of S is 127.
-/
theorem proper_subsets_count_S : ∃ S : Set ℕ, S ≠ ∅ ∧ (card (S.powerset) - 1) = 127 := 
sorry

end proper_subsets_count_S_l570_570248


namespace probability_is_1_over_4_l570_570312

-- Define the setup situation
def digits : List ℕ := [1, 2, 3, 4, 5]

-- Define the set of all possible four-digit numbers
def four_digit_numbers : List (List ℕ) := digits.permutations.filter (λ l, l.length = 4)

-- Define the condition for a number to be a multiple of 4 (last two digits form a number divisible by 4)
def is_multiple_of_4 (num : List ℕ) : Bool :=
  if (10 * num[2] + num[3]) % 4 = 0 then true else false

-- Define the probability calculation
def probability_multiple_of_4 : ℚ :=
  let total := four_digit_numbers.length
  let favorable := (four_digit_numbers.filter is_multiple_of_4).length
  favorable / total

-- The theorem to prove
theorem probability_is_1_over_4 : probability_multiple_of_4 = 1 / 4 :=
by
  sorry

end probability_is_1_over_4_l570_570312


namespace ana_wins_probability_l570_570170

theorem ana_wins_probability :
  let P := (1/2)^(5 : ℕ),
      q := P*(1/(1-P)) in
  q = (1 : ℚ) / 31 := by
sorry

end ana_wins_probability_l570_570170


namespace find_m_l570_570957

variable (m : ℝ)
def A := {m + 1, -3}
def B := {2 * m + 1, m - 3}

theorem find_m : A m ∩ B m = {-3} → m = -2 :=
by
  sorry

end find_m_l570_570957


namespace option_A_option_B_option_C_option_D_l570_570919

variables (A B C a b c : ℝ)
variables (sA sB sC : ℝ)
variables (triangle_ABC : A + B + C = π)

-- Definition of a triangle in Lean
-- Definition of $\triangle ABC$
def triangle (A B C a b c : ℝ) (triangle_ABC : A + B + C = π) : Prop := 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

-- Law of Sines
def law_of_sines (sA sB sC : ℝ) (a b c : ℝ) : Prop := 
  a / sA = b / sB ∧ b / sB = c / sC

-- Definition of obtuse-angled triangle
def obtuse_triangle (A B C : ℝ) : Prop := A > π / 2 ∨ B > π / 2 ∨ C > π / 2

-- A: If $sin A > sin B$, then $A > B$
theorem option_A (h1 : sin A > sin B) : A > B :=
sorry

-- B: If $\sin ^2 B + \sin ^2 C < \sin ^2 A$, then $\triangle ABC$ is obtuse-angled
theorem option_B (h2 : sin B ^ 2 + sin C ^ 2 < sin A ^ 2) : obtuse_triangle A B C :=
sorry

-- C: In acute-angled triangle $\triangle ABC$, the inequality $sin A > cos B$ always holds
theorem option_C (h3 : A < π / 2 ∧ B < π / 2 ∧ C < π / 2) : sin A > cos B :=
sorry

-- D: In $\triangle ABC$, if $a cos A = b cos B$, then $\triangle ABC$ is not necessarily isosceles
theorem option_D (h4 : a * cos A = b * cos B) : ¬(a = b ∧ A = B) :=
sorry

end option_A_option_B_option_C_option_D_l570_570919


namespace smallest_pos_value_correct_l570_570785

noncomputable def smallest_pos_real_number : ℝ :=
  let x := 131 / 11 in
  if x > 0 ∧ (x * x).floor - x * (x.floor) = 10 then x else 0

theorem smallest_pos_value_correct (x : ℝ) (hx : 0 < x ∧ (x * x).floor - x * x.floor = 10) :
  x = 131 / 11 :=
begin
  sorry
end

end smallest_pos_value_correct_l570_570785


namespace geometric_sequence_constant_l570_570512

theorem geometric_sequence_constant (a : ℕ → ℝ) (q : ℝ)
    (h1 : ∀ n, a (n+1) = q * a n)
    (h2 : ∀ n, a n > 0)
    (h3 : (a 1 + a 3) * (a 5 + a 7) = 4 * (a 4) ^ 2) :
    ∀ n, a n = a 0 :=
by
  sorry

end geometric_sequence_constant_l570_570512


namespace roll_not_six_probability_l570_570165

theorem roll_not_six_probability : 
  (P (not (roll 6)) = 5 / 6) :=
by
  -- Definitions
  let fair_die := ℚ → ℚ
  assume independent_rolls (die : fair_die),
  assume probability_each_face,
  -- Implementation details
  sorry

end roll_not_six_probability_l570_570165


namespace sen_donut_holes_l570_570716

/-- 
At the MP Donut Hole Factory, Sen, Jamie, and Mel are coating spherical donut holes in chocolate. 
Sen's donut holes have a radius of 5mm, Jamie's have a radius of 7mm, Mel's have a radius of 9mm.
They all coat their donut holes at the same rate and begin simultaneously. 
Assuming that the chocolate layer has a negligible thickness and spreads evenly over each donut hole, 
proves that Sen will have coated 441 donut holes when all three workers finish coating a donut hole at the same time for the first time.
-/
theorem sen_donut_holes (r_sen r_jamie r_mel : ℕ) (H_sen : r_sen = 5) (H_jamie : r_jamie = 7) (H_mel : r_mel = 9) : 
  let s_sen := 4 * π * r_sen^2,
      s_jamie := 4 * π * r_jamie^2,
      s_mel := 4 * π * r_mel^2 in
  let lcm_s := Nat.lcm (Nat.lcm (s_sen.to_nat) (s_jamie.to_nat)) (s_mel.to_nat) in
  s_sen.to_nat * 441 = lcm_s :=
by
  sorry

end sen_donut_holes_l570_570716


namespace relationship_f_l570_570841

-- Define the function f which is defined on the reals and even
variable (f : ℝ → ℝ)
-- Condition: f is an even function
axiom even_f : ∀ x, f (-x) = f x
-- Condition: (x₁ - x₂)[f(x₁) - f(x₂)] > 0 for all x₁, x₂ ∈ [0, +∞)
axiom increasing_cond : ∀ (x₁ x₂ : ℝ), 0 ≤ x₁ → 0 ≤ x₂ → x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) > 0

theorem relationship_f : f (1/2) < f 1 ∧ f 1 < f (-2) := by
  sorry

end relationship_f_l570_570841


namespace find_scalar_k_l570_570182

noncomputable theory
open_locale classical

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]

theorem find_scalar_k
  {a b c d : V}
  (h : a + b + c + d = 0) :
  ∃ k : ℝ, k * (c × b) + (b × d) + (a × c) + (d × a) = 0 :=
begin
  use 1,
  have h_d : d = -a - b - c, by linarith,
  rw [h_d],
  simp only [add_comm, neg_add_cancel_left, add_left_comm, add_assoc, add_eq_zero_iff_eq_neg],
  sorry,
end

end find_scalar_k_l570_570182


namespace total_revenue_proof_l570_570628

-- Define constants for the problem
def original_price_per_case : ℝ := 25
def first_group_customers : ℕ := 8
def first_group_cases_per_customer : ℕ := 3
def first_group_discount_percentage : ℝ := 0.15
def second_group_customers : ℕ := 4
def second_group_cases_per_customer : ℕ := 2
def second_group_discount_percentage : ℝ := 0.10
def third_group_customers : ℕ := 8
def third_group_cases_per_customer : ℕ := 1

-- Calculate the prices after discount
def discounted_price_first_group : ℝ := original_price_per_case * (1 - first_group_discount_percentage)
def discounted_price_second_group : ℝ := original_price_per_case * (1 - second_group_discount_percentage)
def regular_price : ℝ := original_price_per_case

-- Calculate the total revenue
def total_revenue_first_group : ℝ := first_group_customers * first_group_cases_per_customer * discounted_price_first_group
def total_revenue_second_group : ℝ := second_group_customers * second_group_cases_per_customer * discounted_price_second_group
def total_revenue_third_group : ℝ := third_group_customers * third_group_cases_per_customer * regular_price

def total_revenue : ℝ := total_revenue_first_group + total_revenue_second_group + total_revenue_third_group

-- Prove that the total revenue is $890
theorem total_revenue_proof : total_revenue = 890 := by
  sorry

end total_revenue_proof_l570_570628


namespace gift_exchange_equation_l570_570253

theorem gift_exchange_equation (x : ℕ) (h : x * (x - 1) = 40) : 
  x * (x - 1) = 40 :=
by
  exact h

end gift_exchange_equation_l570_570253


namespace max_principals_in_7_years_l570_570376

theorem max_principals_in_7_years (term_length : ℕ) (period_length : ℕ) :
  term_length = 4 → period_length = 7 → (max_principals : ℕ) := 
by
  sorry

end max_principals_in_7_years_l570_570376


namespace remaining_pie_is_15_percent_l570_570364

-- Carlos's share:
def carlos_share : ℝ := 0.80

-- Initial portion of the pie:
def whole_pie : ℝ := 1

-- Remaining pie after Carlos's share:
def remaining_pie_after_carlos : ℝ := whole_pie - carlos_share

-- Maria's share:
def maria_share : ℝ := remaining_pie_after_carlos / 4

-- Final remaining pie after Maria's share:
def final_remaining_pie : ℝ := remaining_pie_after_carlos - maria_share

-- The statement we need to prove:
theorem remaining_pie_is_15_percent :
  final_remaining_pie = 0.15 :=
by
  sorry

end remaining_pie_is_15_percent_l570_570364


namespace ana_wins_probability_l570_570166

theorem ana_wins_probability :
  let p := 1 / 2
  in let prob_ana_wins_on_nth_turn (n : ℕ) := p^(4 * n)
  in ∑' (n : ℕ), prob_ana_wins_on_nth_turn n = 1 / 15 := sorry

end ana_wins_probability_l570_570166


namespace find_k_l570_570704

-- Definitions based on conditions
def original_regression (x : ℝ) : ℝ := 2 * x + 5

def center_of_sample_points := (2 : ℝ, 2 * 2 + 5)

def incorrect_data_points := [(7, 3), (4, -6)]

def correct_data_points := [(3, 7), (4, 6)]

def corrected_center := (3/2 : ℝ, 11)

def corrected_regression (x : ℝ) : ℝ := (13/3) * x + k

-- Proof statement
theorem find_k (k : ℝ) : 
  center_of_sample_points = (2, 9) →
  corrected_center = (3/2, 11) →
  corrected_regression (3/2) = 11 →
  k = 9 / 2 :=
by
  sorry

end find_k_l570_570704


namespace min_value_expression_l570_570951

noncomputable def w (a b : ℝ) : ℂ := a + b * Complex.I

theorem min_value_expression :
  ∃ w : ℂ, (Complex.abs (w - (3 - 2 * Complex.I)) = 4) ∧
  (∀ w', Complex.abs (w' - (3 - 2 * Complex.I)) = 4 → 
    |w' + (1 + 2 * Complex.I)|^2 + |w' - (7 + 2 * Complex.I)|^2 ≥ 48) :=
by
  sorry

end min_value_expression_l570_570951


namespace inf_many_solutions_system_l570_570982

theorem inf_many_solutions_system :
  ∃ (x : Fin 1985 → ℕ) (y z : ℕ), 
    (∑ i, x i ^ 2 = y ^ 3) ∧ (∑ i, x i ^ 3 = z ^ 2) ∧ 
    (∀ i j, i ≠ j → x i ≠ x j) ∧ 
    ∃ f : ℤ → ℕ → Prop, ∀ n, ∃ p, f n p := 
begin
  sorry
end

end inf_many_solutions_system_l570_570982


namespace arithmetic_geometric_sequence_l570_570447

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- Define the first term, common difference and positions of terms in geometric sequence
def a1 : ℤ := -8
def d : ℤ := 2
def a3 := arithmetic_sequence a1 d 2
def a4 := arithmetic_sequence a1 d 3

-- Conditions for the terms forming a geometric sequence
def geometric_condition (a b c : ℤ) : Prop :=
  b^2 = a * c

-- Statement to prove
theorem arithmetic_geometric_sequence :
  geometric_condition a1 a3 a4 → a1 = -8 :=
by
  intro h
  -- Proof can be filled in here
  sorry

end arithmetic_geometric_sequence_l570_570447


namespace quadratic_has_exactly_one_real_solution_iff_l570_570799

theorem quadratic_has_exactly_one_real_solution_iff (m : ℝ) :
  (∃ x, 3 * x^2 + m * x + 16 = 0) ∧ ∃ x₁ x₂, ∀ x, (3 * x^2 + m * x + 16 = 0 ↔ x = x₁) :=
  m = 8 * Real.sqrt 3 ∨ m = -8 * Real.sqrt 3 :=
sorry

end quadratic_has_exactly_one_real_solution_iff_l570_570799


namespace find_n_from_binomial_expansion_l570_570467

theorem find_n_from_binomial_expansion (n : ℕ) (x : ℝ) :
  (let p := (3 * x ^ (1 / 3) + 1 / x) ^ n;
       q := (binom n k) in
       p + q = 272) →
  n = 4 := 
sorry

end find_n_from_binomial_expansion_l570_570467


namespace op_15_5_eq_33_l570_570133

def op (x y : ℕ) : ℕ :=
  2 * x + x / y

theorem op_15_5_eq_33 : op 15 5 = 33 := by
  sorry

end op_15_5_eq_33_l570_570133


namespace price_of_battery_l570_570422

def cost_of_tire : ℕ := 42
def cost_of_tires (num_tires : ℕ) : ℕ := num_tires * cost_of_tire
def total_cost : ℕ := 224
def num_tires : ℕ := 4
def cost_of_battery : ℕ := total_cost - cost_of_tires num_tires

theorem price_of_battery : cost_of_battery = 56 := by
  sorry

end price_of_battery_l570_570422


namespace ratio_lavinia_son_to_katie_daughter_l570_570931

theorem ratio_lavinia_son_to_katie_daughter :
  ∀ (K Ld Ls : ℕ),
  K = 12 →
  Ld = K - 10 →
  Ls = Ld + 22 →
  (Ls / K) = 2 :=
by
  intros K Ld Ls hK hLd hLs
  have hK12 : K = 12 := hK
  have hLd2 : Ld = K - 10 := hLd
  have hLs24 : Ls = Ld + 22 := hLs
  rw [← hK12, ← hLd2, ← hLs24]
  sorry

end ratio_lavinia_son_to_katie_daughter_l570_570931


namespace geometric_sequence_S9_l570_570181

theorem geometric_sequence_S9 (S : ℕ → ℝ) (S3_eq : S 3 = 2) (S6_eq : S 6 = 6) : S 9 = 14 :=
by
  sorry

end geometric_sequence_S9_l570_570181


namespace circle_graph_to_bar_graph_correct_l570_570823

theorem circle_graph_to_bar_graph_correct :
  ∀ (white black gray blue : ℚ) (w_proportion b_proportion g_proportion blu_proportion : ℚ),
    white = 1/2 →
    black = 1/4 →
    gray = 1/8 →
    blue = 1/8 →
    w_proportion = 1/2 →
    b_proportion = 1/4 →
    g_proportion = 1/8 →
    blu_proportion = 1/8 →
    white = w_proportion ∧ black = b_proportion ∧ gray = g_proportion ∧ blue = blu_proportion :=
by
sorry

end circle_graph_to_bar_graph_correct_l570_570823


namespace exists_n_consecutive_composites_l570_570226

theorem exists_n_consecutive_composites (n : ℕ) (h : n ≥ 1) (a r : ℕ) :
  ∃ K : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ¬(Nat.Prime (a + (K + i) * r)) := 
sorry

end exists_n_consecutive_composites_l570_570226


namespace no_integer_roots_polynomial_l570_570950

-- Definitions used from conditions
variables {n : ℕ} (coeffs : Fin (2 * n + 1) → ℤ)

-- Conditions
def non_zero_cards : Prop := ∀ i, coeffs i ≠ 0
def sum_non_zero : Prop := (Finset.univ.sum coeffs) ≠ 0

-- The statement to prove
theorem no_integer_roots_polynomial
  (hnz : non_zero_cards coeffs)
  (hsnz : sum_non_zero coeffs) :
  ∃ P : Polynomial ℤ, (P.coeffs) = (coeffs : List ℤ) ∧ ∀ x : ℤ, P.eval x ≠ 0 :=
by
  sorry

end no_integer_roots_polynomial_l570_570950


namespace probability_point_below_x_axis_l570_570212

structure Point2D :=
  (x : ℝ)
  (y : ℝ)

structure Parallelogram :=
  (P Q R S : Point2D)

def vertices_of_PQRS : Parallelogram :=
  ⟨⟨4, 4⟩, ⟨-2, -2⟩, ⟨-8, -2⟩, ⟨-2, 4⟩⟩

def point_lies_below_x_axis_probability (parallelogram : Parallelogram) : ℝ :=
  sorry

theorem probability_point_below_x_axis :
  point_lies_below_x_axis_probability vertices_of_PQRS = 1 / 2 :=
sorry

end probability_point_below_x_axis_l570_570212


namespace length_NM_constant_l570_570175

-- Define the general conditions and variables
variables (O : Type) [metric_space O] [inner_product_space ℝ O] (circle_O : set O)
variables (A B W C : O)
variables (circumference : ∀ P ∈ circle_O, ∃ t : ℝ, P = cos t • A + sin t • B)
variables (midpoint_of_minor_arc : W = (circumference.some • A + (1 - circumference.some) • B) / 2)
variables (C_on_major_arc : ∃ t : ℝ, t ∈ (circumference.some, circumference.some + 2 * π) ∧ C = cos t • A + sin t • B)

-- Define the tangent lines
variables (tangent_C : ∀ x y, inner_product (x - C) (y - C) = 0)
variables (tangent_A : ∀ x y, inner_product (x - A) (y - A) = 0)
variables (tangent_B : ∀ x y, inner_product (x - B) (y - B) = 0)

-- Define the intersection points X and Y
variables (X Y : O)
variables (intersection_X : X = tangent_A ∩ tangent_C)
variables (intersection_Y : Y = tangent_B ∩ tangent_C)

-- Define the intersection points N and M
variables (N M : O)
variables (intersection_N : N = line W X ∩ line A B)
variables (intersection_M : M = line W Y ∩ line A B)

-- Prove that the length NM does not depend on C
theorem length_NM_constant {O : Type} [metric_space O] [inner_product_space ℝ O]
  [circle_O : metric_space O]
  [A B W C : O]
  [midpoint_of_minor_arc : W = some • A + (1 - some) • B]
  [C_on_major_arc : ∃ t : ℝ, t ∈ (some, some + 2 * π) ∧ C = cos t • A + sin t • B]
  [X Y : O]
  [intersection_X : X = tangent_A ∩ tangent_C]
  [intersection_Y : Y = tangent_B ∩ tangent_C]
  [N M : O]
  [intersection_N : N = line W X ∩ line A B]
  [intersection_M : M = line W Y ∩ line A B] :
  dist N M = dist A B / 2 :=
sorry

end length_NM_constant_l570_570175


namespace inscribed_circle_radius_l570_570652

theorem inscribed_circle_radius (DE DF EF : ℝ) (h1 : DE = 26) (h2 : DF = 15) (h3 : EF = 17) :
    ∃ r, r = Real.sqrt 6 ∧
    let s := (DE + DF + EF) / 2 in
    let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)) in
    K = r * s :=
by
  sorry

end inscribed_circle_radius_l570_570652


namespace dorms_and_students_l570_570987

theorem dorms_and_students (x : ℕ) :
  (4 * x + 19) % 6 ≠ 0 → ∃ s : ℕ, (x = 10 ∧ s = 59) ∨ (x = 11 ∧ s = 63) ∨ (x = 12 ∧ s = 67) :=
by
  sorry

end dorms_and_students_l570_570987


namespace minimum_possible_value_of_P_l570_570416

def nearest_int (m n : ℤ) : ℤ := (m + n / 2) / n -- definition of nearest integer to m/n

def satisfies_condition (n k : ℤ) : Prop :=
  nearest_int n k + nearest_int (150 - n) k = nearest_int 150 k

def P (k : ℤ) : ℚ :=
  (Finset.filter (λ n : ℤ, satisfies_condition n k) (Finset.range (149 + 1))).card / 149

noncomputable def min_P_value : ℚ :=
  Finset.inf' (Finset.filter (λ k, k % 2 = 1 ∧ 101 ≤ k ∧ k ≤ 199) Finset.range' (101 - 1) (199 - 100 + 1))
    (by decide) P

theorem minimum_possible_value_of_P :
  min_P_value = 75 / 151 :=
by
  sorry

end minimum_possible_value_of_P_l570_570416


namespace distance_between_projections_l570_570519

open Real

def A : ℝ × ℝ × ℝ := (-1, 2, -3)

def projection_onto_yOz_plane (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (0, p.2, p.3)

def projection_onto_x_axis (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, 0, 0)

def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

theorem distance_between_projections : 
  distance (projection_onto_yOz_plane A) (projection_onto_x_axis A) = sqrt 14 := by
  sorry

end distance_between_projections_l570_570519


namespace rose_initial_rice_l570_570579

theorem rose_initial_rice : 
  ∀ (R : ℝ), (R - 9 / 10 * R - 1 / 4 * (R - 9 / 10 * R) = 0.75) → (R = 10) :=
by
  intro R h
  sorry

end rose_initial_rice_l570_570579


namespace range_of_a_l570_570085

variable (a : ℝ)

def p := ∀ x, (2 * a - 6) < 1 ∧ (2 * a - 6) > 0
def q := ∀ x, x^2 - 3 * a * x + 2 * a^2 + 1 = 0 → x > 3

theorem range_of_a (h : (p a ∨ q a) ∧ ¬(p a ∧ q a)) : a > 7 / 2 :=
by sorry

end range_of_a_l570_570085


namespace determinant_cosine_matrix_l570_570729

open matrix

noncomputable def A : matrix (fin 3) (fin 3) ℝ :=
  ![![cos 0, cos 1, cos 2],
    ![cos 3, cos 4, cos 5],
    ![cos 6, cos 7, cos 8]]

theorem determinant_cosine_matrix : det A = 0 := by
  sorry

end determinant_cosine_matrix_l570_570729


namespace sin_angle_DAC_zero_l570_570162

variables {A B C D : Type} [PlaneGeometry A] [Point B] [Point C] [Point D] [RightTriangle A B C]
variables (h1 : sin (angle A B C) = 3 / 5)
variables (h2 : cos (angle A B D) = 4 / 5)

theorem sin_angle_DAC_zero :
  sin (angle D A C) = 0 :=
sorry

end sin_angle_DAC_zero_l570_570162


namespace problem1_l570_570308

theorem problem1 (a : ℝ) (h : Real.sqrt a + Real.sqrt (1/a) = 3) : 
  (a^2 + a^(-2) + 1) / (a + a^(-1) - 1) = 8 := 
  sorry

end problem1_l570_570308


namespace intersection_PQ_eq_23_l570_570114

def P : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}
def Q : Set ℝ := {x : ℝ | 2 < x}

theorem intersection_PQ_eq_23 : P ∩ Q = {x : ℝ | 2 < x ∧ x < 3} := 
by {
  sorry
}

end intersection_PQ_eq_23_l570_570114


namespace modulus_of_complex_l570_570895

theorem modulus_of_complex (a : ℝ) (h : a < 0) : complex.abs (3 * a - 4 * a * complex.I) = -5 * a :=
by
  sorry

end modulus_of_complex_l570_570895


namespace first_place_clay_l570_570349

def Clay := "Clay"
def Allen := "Allen"
def Bart := "Bart"
def Dick := "Dick"

-- Statements made by the participants
def Allen_statements := ["I finished right before Bart", "I am not the first"]
def Bart_statements := ["I finished right before Clay", "I am not the second"]
def Clay_statements := ["I finished right before Dick", "I am not the third"]
def Dick_statements := ["I finished right before Allen", "I am not the last"]

-- Conditions
def only_two_true_statements : Prop := sorry -- This represents the condition that only two of these statements are true.
def first_place_told_truth : Prop := sorry -- This represents the condition that the person who got first place told at least one truth.

def person_first_place := Clay

theorem first_place_clay : person_first_place = Clay ∧ only_two_true_statements ∧ first_place_told_truth := 
sorry

end first_place_clay_l570_570349


namespace unique_perpendicular_in_plane_l570_570662

-- Definitions and the main theorem statement

variables {α : Type*} [EuclideanGeometry α]

theorem unique_perpendicular_in_plane (l : Line α) (P : Point α) :
  ∃! m : Line α, m.perpendicular_to l ∧ P ∈ m := 
sorry

end unique_perpendicular_in_plane_l570_570662


namespace bob_paid_correctly_l570_570711

-- Define the variables involved
def alice_acorns : ℕ := 3600
def price_per_acorn : ℕ := 15
def multiplier : ℕ := 9
def total_amount_alice_paid : ℕ := alice_acorns * price_per_acorn

-- Define Bob's payment amount
def bob_payment : ℕ := total_amount_alice_paid / multiplier

-- The main theorem
theorem bob_paid_correctly : bob_payment = 6000 := by
  sorry

end bob_paid_correctly_l570_570711


namespace compute_star_l570_570750

-- Define the star operation
def star (a b : ℝ) : ℝ := (a + b) / (a - b)

-- The theorem to prove
theorem compute_star : star (star 3 5) 8 = -1 / 3 := by
  sorry

end compute_star_l570_570750


namespace correct_calculation_l570_570350

theorem correct_calculation (x y : ℝ) : (x^2 * y)^3 = x^6 * y^3 :=
  sorry

end correct_calculation_l570_570350


namespace intersection_A_B_l570_570454

def setA : Set (ℝ × ℝ) := {p | ∃ (x: ℝ), p = (x, x^2)}
def setB : Set (ℝ × ℝ) := {p | ∃ (x: ℝ), p = (x, Real.sqrt x)}

theorem intersection_A_B :
  (setA ∩ setB) = {(0, 0), (1, 1)} := by
  sorry

end intersection_A_B_l570_570454


namespace equilateral_triangle_perimeter_l570_570250

theorem equilateral_triangle_perimeter (s : ℕ) (b : ℕ) (h1 : 40 = 2 * s + b) (h2 : b = 10) : 3 * s = 45 :=
by {
  sorry
}

end equilateral_triangle_perimeter_l570_570250


namespace probability_at_least_one_woman_l570_570695

-- Given definitions
def total_employees : ℕ := 10
def total_men : ℕ := 6
def total_women : ℕ := 4
def unavailable_man : ℕ := 1
def unavailable_woman : ℕ := 1
def available_men : ℕ := total_men - unavailable_man
def available_women : ℕ := total_women - unavailable_woman
def available_employees : ℕ := available_men + available_women
def selections : ℕ := 3

-- Binomial coefficient calculation
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Probability calculation
def prob_at_least_one_woman : ℚ := 
  1 - (binom available_men selections : ℚ) / binom available_employees selections

-- Statement to prove
theorem probability_at_least_one_woman :
  prob_at_least_one_woman = 23 / 28 :=
sorry

end probability_at_least_one_woman_l570_570695


namespace inscribed_circle_fraction_l570_570242

theorem inscribed_circle_fraction (s : ℝ) (h : s = 4) (ratio : 1 / (1 + 3) = 1 / 4) :
  let height := sqrt (s^2 - 1^2) / 2
  let r := height / 2
  let A_c := π * r^2
  let A_r := s * height * 2 in
  A_c / A_r = π * sqrt 15 / 16 :=
by
  sorry

end inscribed_circle_fraction_l570_570242


namespace average_k_of_polynomial_with_positive_integer_roots_l570_570845

-- Define the conditions and the final theorem

theorem average_k_of_polynomial_with_positive_integer_roots :
  (∑ i in {k | ∃ r1 r2 : ℕ+, r1 * r2 = 24 ∧ k = r1 + r2}.to_finset, i) / 
  ({k | ∃ r1 r2 : ℕ+, r1 * r2 = 24 ∧ k = r1 + r2}.to_finset.card : ℝ) = 15 :=
by
  sorry

end average_k_of_polynomial_with_positive_integer_roots_l570_570845


namespace total_investment_amount_l570_570338

-- Define the conditions
def total_interest_in_one_year : ℝ := 1023
def invested_at_6_percent : ℝ := 8200
def interest_rate_6_percent : ℝ := 0.06
def interest_rate_7_5_percent : ℝ := 0.075

-- Define the equation based on the conditions
def interest_from_6_percent_investment : ℝ := invested_at_6_percent * interest_rate_6_percent

def total_investment_is_correct (T : ℝ) : Prop :=
  let interest_from_7_5_percent_investment := (T - invested_at_6_percent) * interest_rate_7_5_percent
  interest_from_6_percent_investment + interest_from_7_5_percent_investment = total_interest_in_one_year

-- Statement to prove
theorem total_investment_amount : total_investment_is_correct 15280 :=
by
  unfold total_investment_is_correct
  unfold interest_from_6_percent_investment
  simp
  sorry

end total_investment_amount_l570_570338


namespace consumer_installment_credit_value_l570_570298

variable (consumer_installment_credit : ℝ) 

noncomputable def automobile_installment_credit := 0.36 * consumer_installment_credit

noncomputable def finance_company_credit := 35

theorem consumer_installment_credit_value :
  (∃ C : ℝ, automobile_installment_credit C = 0.36 * C ∧ finance_company_credit = (1 / 3) * automobile_installment_credit C) →
  consumer_installment_credit = 291.67 :=
by
  sorry

end consumer_installment_credit_value_l570_570298


namespace functional_eq_zero_func_l570_570672

theorem functional_eq_zero_func (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(f(x) + x + y) = f(x + y) + y * f(y)) → (∀ x : ℝ, f(x) = 0) :=
by
  sorry

end functional_eq_zero_func_l570_570672


namespace mark_percentage_increase_l570_570904

-- Given a game with the following conditions:
-- Condition 1: Samanta has 8 more points than Mark
-- Condition 2: Eric has 6 points
-- Condition 3: The total points of Samanta, Mark, and Eric is 32

theorem mark_percentage_increase (S M : ℕ) (h1 : S = M + 8) (h2 : 6 + S + M = 32) : 
  (M - 6) / 6 * 100 = 50 :=
sorry

end mark_percentage_increase_l570_570904


namespace sum_of_first_six_terms_l570_570079

theorem sum_of_first_six_terms 
  {S : ℕ → ℝ} 
  (h_arith_seq : ∀ n, S n = n * (-2) + (n * (n - 1) * 3 ))
  (S_2_eq_2 : S 2 = 2)
  (S_4_eq_10 : S 4 = 10) : S 6 = 18 := 
  sorry

end sum_of_first_six_terms_l570_570079


namespace sum_f_always_negative_l570_570439

noncomputable def f (x : ℝ) : ℝ := -x - x^3

theorem sum_f_always_negative
  (α β γ : ℝ)
  (h1 : α + β > 0)
  (h2 : β + γ > 0)
  (h3 : γ + α > 0) :
  f α + f β + f γ < 0 :=
by
  unfold f
  sorry

end sum_f_always_negative_l570_570439


namespace ellipse_properties_l570_570080

theorem ellipse_properties :
  ∃ (a b: ℝ) (C : ℝ → ℝ → Prop),
  (C = λ x y, x^2/6 + y^2/2 = 1) ∧
  ∃ (l : ℝ → ℝ) (k : ℝ),
  (l = λ x, k * (x - 2)) ∧
  (|k| = 1) ∧
  (k = 1 ∨ k = -1) :=
begin
  sorry
end

end ellipse_properties_l570_570080


namespace arc_length_correct_l570_570000

noncomputable def x (t : ℝ) : ℝ := Real.exp t * (Real.cos t + Real.sin t)
noncomputable def y (t : ℝ) : ℝ := Real.exp t * (Real.cos t - Real.sin t)

noncomputable def x' (t : ℝ) : ℝ := Real.exp t * (2 * Real.cos t)
noncomputable def y' (t : ℝ) : ℝ := Real.exp t * (-2 * Real.sin t)

noncomputable def integrand (t : ℝ) : ℝ := Real.sqrt ((x' t)^2 + (y' t)^2)

noncomputable def arc_length : ℝ :=
  intervalIntegral (integrand) 0 π

theorem arc_length_correct :
  arc_length = 2 * (Real.exp π - 1) :=
by
  sorry

end arc_length_correct_l570_570000


namespace proof_statement_l570_570918

-- Definitions of the problem
variables {α : Type} [linear_ordered_field α] {a b c : α} {A B C : angle α}

-- Conditions from the original problem
def conditionA (h : sin A > sin B) : Prop := A > B
def conditionB (h : sin^2 B + sin^2 C < sin^2 A) : Prop := 
  (a^2 > b^2 + c^2) ∨ (b^2 > a^2 + c^2) ∨ (c^2 > a^2 + b^2) -- At least one angle is obtuse
def conditionC (h : A + B < π/2 ∧ B + C < π/2 ∧ C + A < π/2) : Prop := sin A > cos B
def conditionD (h : a * cos A = b * cos B) : Prop := ¬(a ≠ b)

-- The corresponding Lean theorem
theorem proof_statement : 
  (∀ (A B C : angle α) (a b c : α),
    (conditionA (has order_alpha.sin A < order_alpha.sin B) →
    conditionB (has order_alpha.sin B^2 + sin C^2 < sin A^2) →
    conditionC (0 < A < π/2 ∧ 0 < B < π/2 ∧ 0 < C < π/2)
    conditionD (a * order_alpha.cos A = b * order_alpha.cos B) → 
    Prop :=
sorry


end proof_statement_l570_570918


namespace no_integer_solution_l570_570998

theorem no_integer_solution : ¬ ∃ (x y : ℤ), x^2 - 7 * y = 10 :=
by
  sorry

end no_integer_solution_l570_570998


namespace smallest_pos_value_correct_l570_570781

noncomputable def smallest_pos_real_number : ℝ :=
  let x := 131 / 11 in
  if x > 0 ∧ (x * x).floor - x * (x.floor) = 10 then x else 0

theorem smallest_pos_value_correct (x : ℝ) (hx : 0 < x ∧ (x * x).floor - x * x.floor = 10) :
  x = 131 / 11 :=
begin
  sorry
end

end smallest_pos_value_correct_l570_570781


namespace charlie_delta_purchases_l570_570714

-- Defining the setup for the problem
def num_oreo_flavors := 6
def num_milk_flavors := 4
def total_num_flavors := num_oreo_flavors + num_milk_flavors

-- Defining the conditions for Charlie and Delta's purchase
def charlie_purchases_at_most (n : ℕ) := n <= total_num_flavors
def delta_purchases_at_least_one_oreo (o : ℕ) := 1 <= o ∧ o <= num_oreo_flavors
def total_purchases (c : ℕ) (o : ℕ) := c + o = 4

-- Theorem statement
theorem charlie_delta_purchases : 
  (∑ c in finset.range (total_num_flavors + 1), 
   ∑ o in finset.range (num_oreo_flavors + 1), 
   if (charlie_purchases_at_most c ∧ delta_purchases_at_least_one_oreo o ∧ total_purchases c o)
   then finset.card (finset.powerset_len c (finset.range total_num_flavors)) * 
        finset.card (finset.range o) 
   else 0) = 2225 := 
sorry -- Proof omitted

end charlie_delta_purchases_l570_570714


namespace compute_expression_l570_570814

-- Definitions used in the conditions
def a : Real := log 25
def b : Real := log 49

-- Statement of the theorem
theorem compute_expression : 
  5^(a/b) + 2 * 7^(b/a) = Real.root 7 (5^5) + 2 * Real.root 5 (7^7) := 
by
  sorry

end compute_expression_l570_570814


namespace smallest_positive_real_is_131_div_11_l570_570770

noncomputable def smallest_positive_real_satisfying_condition :=
  ∀ (x : ℝ), (∀ y > 0, (y * y ⌊y⌋ - y ⌊y⌋ = 10) → (x ≤ y)) → 
  (⌊x*x⌋ - (x * ⌊x⌋) = 10) → 
  x = 131/11

theorem smallest_positive_real_is_131_div_11 :
  smallest_positive_real_satisfying_condition := sorry

end smallest_positive_real_is_131_div_11_l570_570770


namespace triangle_properties_l570_570857

theorem triangle_properties
  (f : ℝ → ℝ)
  (a b c A B C : ℝ)
  (R : ℝ)
  (h_f : ∀ x : ℝ, f x = 2 * sin (x / 2) * cos (x / 2) + 2 * sqrt 3 * (cos (x / 2))^2 - sqrt 3)
  (h_fA : f A = sqrt 3)
  (h_R : R = sqrt 3)
  (h_a : a = 2 * R * sin A) :
  (A = π / 3) ∧ ((1 / 2) * a * b * (sqrt 3 / 2) = 9 * sqrt 3 / 4) := 
by
  sorry

end triangle_properties_l570_570857


namespace number_of_words_with_at_least_one_consonant_l570_570487

def total_5_letter_words : ℕ := 6 ^ 5

def total_5_letter_vowel_words : ℕ := 2 ^ 5

def total_5_letter_words_with_consonant : ℕ := total_5_letter_words - total_5_letter_vowel_words

theorem number_of_words_with_at_least_one_consonant :
  total_5_letter_words_with_consonant = 7744 :=
  by
    -- We assert the calculation follows correctly:
    -- total_5_letter_words == 6^5 = 7776
    -- total_5_letter_vowel_words == 2^5 = 32
    -- 7776 - 32 == 7744
    sorry

end number_of_words_with_at_least_one_consonant_l570_570487


namespace general_formula_of_an_sum_of_sequence_n_an_l570_570076

open Nat

def sequence_a : ℕ → ℕ
| 0 := 0  -- Indexing starts with 1 based on the given problem (so a_1 = 1)
| (n+1) := 2 * sequence_a n + 1

def a (n : ℕ) : ℕ := sequence_a n.succ

noncomputable def S (n : ℕ) : ℕ := 
  (n-1) * 2^(n+1) + 2 - n^2 / 2 - n / 2

theorem general_formula_of_an : ∀ n, a n = 2^n - 1 :=
by
  sorry

theorem sum_of_sequence_n_an : ∀ n, (∑ i in range n, i * a i) = S n :=
by
  sorry

end general_formula_of_an_sum_of_sequence_n_an_l570_570076


namespace polynomial_value_at_l570_570618

noncomputable def p (n : ℕ) (q : Polynomial ℤ) : ℤ := 
  if is_odd n then 1 else (n / (n + 2))

theorem polynomial_value_at (n : ℕ) (p : ℤ → ℤ) :
  (∀ k, 0 ≤ k ∧ k ≤ n → p k = k / (k + 1)) →
  p (n + 1) = if n % 2 = 1 then 1 else n / (n + 2) :=
begin
  sorry
end

end polynomial_value_at_l570_570618


namespace b_arithmetic_sequence_max_S_n_l570_570460

variables {a : ℕ → ℝ} {b : ℕ → ℝ} {S : ℕ → ℝ}

-- Conditions
noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m, a m ≠ 0 → a n = a (n + 1) * a (m-1) / (a m)

axiom a_pos_terms : ∀ n, 0 < a n
axiom a11_eight : a 11 = 8
axiom b_log : ∀ n, b n = Real.log (a n) / Real.log 2
axiom b4_seventeen : b 4 = 17

-- Question I: Prove b_n is an arithmetic sequence with common difference -2
theorem b_arithmetic_sequence (d : ℝ) (h_d : d = (-2)) :
  ∃ d, ∀ n, b (n + 1) - b n = d :=
sorry

-- Question II: Find the maximum value of S_n
theorem max_S_n : ∃ n, S n = 144 :=
sorry

end b_arithmetic_sequence_max_S_n_l570_570460


namespace train_speed_length_l570_570798

theorem train_speed_length (t1 t2 s : ℕ) (p : ℕ)
  (h1 : t1 = 7) 
  (h2 : t2 = 25) 
  (h3 : p = 378)
  (h4 : t2 - t1 = 18)
  (h5 : p / (t2 - t1) = 21) 
  (h6 : (p / (t2 - t1)) * t1 = 147) :
  (21, 147) = (21, 147) :=
by {
  sorry
}

end train_speed_length_l570_570798


namespace div_by_7_l570_570981

theorem div_by_7 (n : ℕ) (h : n ≥ 1) : 7 ∣ (8^n + 6) :=
sorry

end div_by_7_l570_570981


namespace solve_for_x_l570_570994

theorem solve_for_x : ∀ x : ℝ, 3^(2*x + 1) = 1/81 → x = -5/2 :=
by
  intro x
  intro h
  have h1 : 1/81 = 3^(-4) := by
    exact one_div_pow (3 : ℝ) 4
  rw [h1] at h
  have h_exp : 3^(2*x + 1) = 3^(-4) := h
  have eq_exp : 2*x + 1 = -4 := by
    apply eq_of_monotone_of_pow_eq
    apply pow_ne_zero
    norm_num
    exact h_exp
  linarith

end solve_for_x_l570_570994


namespace sum_of_rational_roots_of_h_l570_570411

theorem sum_of_rational_roots_of_h :
  let h(x : ℚ) := x^3 - 6*x^2 + 11*x - 6
  h(1) = 0 ∧ h(2) = 0 ∧ h(3) = 0 →
  (1 + 2 + 3 = 6) := sorry

end sum_of_rational_roots_of_h_l570_570411


namespace smallest_positive_real_is_131_div_11_l570_570773

noncomputable def smallest_positive_real_satisfying_condition :=
  ∀ (x : ℝ), (∀ y > 0, (y * y ⌊y⌋ - y ⌊y⌋ = 10) → (x ≤ y)) → 
  (⌊x*x⌋ - (x * ⌊x⌋) = 10) → 
  x = 131/11

theorem smallest_positive_real_is_131_div_11 :
  smallest_positive_real_satisfying_condition := sorry

end smallest_positive_real_is_131_div_11_l570_570773


namespace positive_divisors_of_40_and_72_l570_570880

-- Define the main problem
def problem_statement : Prop :=
  let common_divisors := {d : ℕ | d > 0 ∧ 40 % d = 0 ∧ 72 % d = 0} in
  common_divisors.to_finset.card = 4

-- The statement of the proof problem
theorem positive_divisors_of_40_and_72 : problem_statement :=
by
  sorry

end positive_divisors_of_40_and_72_l570_570880


namespace diameter_of_circle_A_l570_570002

def radius_of_circle_B : ℝ := 10

def area_of_circle (r : ℝ) : ℝ := π * r ^ 2

def area_ratio_circle_A_shaded (area_A shaded : ℝ) : Prop :=
  area_A / shaded = 1 / 7

theorem diameter_of_circle_A
  (h1 : radius_of_circle_B = 10)
  (h2 : ∀ r_A : ℝ, area_ratio_circle_A_shaded (area_of_circle r_A) (area_of_circle radius_of_circle_B - area_of_circle r_A)) :
  ∃ d_A : ℝ, d_A = 7.08 :=
begin
  sorry
end

end diameter_of_circle_A_l570_570002


namespace positively_correlated_pairs_l570_570268

-- Definitions of "correlated" (we assume positive correlation means an increasing linear relationship)
def positively_correlated (x y : ℝ) : Prop := ∃ c > 0, ∀ a b : ℝ, (a > b) → (c * a + y > c * b + y)
def negatively_correlated (x y : ℝ) : Prop := ∃ c < 0, ∀ a b : ℝ, (a > b) → (c * a + y < c * b + y)
def functional_relationship (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, (x ≠ y) → (f x ≠ f y)

-- Given conditions as definitions (translated):
def cond1 (weight distance_per_liter : ℝ) := negatively_correlated weight distance_per_liter
def cond2 (study_time academic_performance : ℝ) := positively_correlated study_time academic_performance
def cond3 (smoking_health : ℝ) := negatively_correlated (fst smoking_health) (snd smoking_health)
def cond4 (side_length : ℝ) := ∀ a : ℝ, functional_relationship (λ x, x * x)
def cond5 (weight fuel_consumption : ℝ) := positively_correlated weight fuel_consumption

theorem positively_correlated_pairs :
  cond2 study_time academic_performance ∧ cond5 weight fuel_consumption →
  "The pairs in ② and ⑤ are positively correlated." := by sorry

end positively_correlated_pairs_l570_570268


namespace angle_EAC_l570_570572

variables {A B C E : Type}
variables {angle : A → B → C → ℝ}

-- Definitions from the conditions
def point_on_side (E : B) (A B C : A) : Prop := -- A definition denoting point E is on the side BC of triangle ABC
sorry

def angle_BAE (A B E : A) : ℝ := 20
def angle_ABE (A B E : A) : ℝ := 40

-- Theorem statement
theorem angle_EAC (h : point_on_side E A B C) : angle A E C = 40 :=
sorry

end angle_EAC_l570_570572


namespace largest_possible_a_l570_570953

theorem largest_possible_a :
  ∀ (a b c d : ℕ), a < 3 * b ∧ b < 4 * c ∧ c < 5 * d ∧ d < 80 ∧ 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d → a ≤ 4724 := by
  sorry

end largest_possible_a_l570_570953


namespace remainder_is_zero_l570_570511

theorem remainder_is_zero (D R r : ℕ) (h1 : D = 12 * 42 + R)
                           (h2 : D = 21 * 24 + r)
                           (h3 : r < 21) :
                           r = 0 :=
by 
  sorry

end remainder_is_zero_l570_570511


namespace water_wheel_effective_horsepower_l570_570515

noncomputable def effective_horsepower 
  (velocity : ℝ) (width : ℝ) (thickness : ℝ) (density : ℝ) 
  (diameter : ℝ) (efficiency : ℝ) (g : ℝ) (hp_conversion : ℝ) : ℝ :=
  let mass_flow_rate := velocity * width * thickness * density
  let kinetic_energy_per_second := 0.5 * mass_flow_rate * velocity^2
  let potential_energy_per_second := mass_flow_rate * diameter * g
  let indicated_power := kinetic_energy_per_second + potential_energy_per_second
  let horsepower := indicated_power / hp_conversion
  efficiency * horsepower

theorem water_wheel_effective_horsepower :
  effective_horsepower 1.4 0.5 0.13 1000 3 0.78 9.81 745.7 = 2.9 :=
by
  sorry

end water_wheel_effective_horsepower_l570_570515


namespace find_x_l570_570947

namespace MathProof

variables {a b x : ℝ}
variables (h1 : a > 0) (h2 : b > 0)

theorem find_x (h3 : (a^2)^(2 * b) = a^b * x^b) : x = a^3 :=
by sorry

end MathProof

end find_x_l570_570947


namespace erin_paths_tetrahedron_l570_570377

theorem erin_paths_tetrahedron : 
  let V := {A, B, C, D} -- The vertices of the tetrahedron
  in ∃ paths : Set (List V), 
    (∀ p ∈ paths, List.Nodup p ∧ List.length p = 4 ∧ 
    (∀ (u v : V), (u, v)∈ List.zip p (List.tail p) → (Set.singleton (u, v)).subset edges)) ∧ 
    paths.card = 6 :=
sorry

end erin_paths_tetrahedron_l570_570377


namespace sum_of_a_unique_solution_l570_570058

def f (x : ℝ) : ℝ := abs ((2 * x^3 - 5 * x^2 - 2 * x + 5) / ((1.5 * x - 3)^2 - (0.5 * x - 2)^2))

def p (x : ℝ) (a : ℝ) : ℝ := abs (2 * x + 5) + a

theorem sum_of_a_unique_solution :
    (finset.univ.filter (λ a, (∃! x : ℝ, x ≠ 1 ∧ x ≠ 2.5 ∧ f x = p x a))).sum id = -10 :=
by
  sorry

end sum_of_a_unique_solution_l570_570058


namespace blue_paint_amount_l570_570809

theorem blue_paint_amount
  (blue_white_ratio : ℚ := 4 / 5)
  (white_paint : ℚ := 15)
  (blue_paint : ℚ) :
  blue_paint = 12 :=
by
  sorry

end blue_paint_amount_l570_570809


namespace probability_A_shot_twice_stops_l570_570202

theorem probability_A_shot_twice_stops 
  (pA : ℚ) (pB : ℚ) (independent_shots : Prop) (turns : list (str : String)) :
  (pA = 3 / 4) →
  (pB = 4 / 5) →
  (independent_shots) →
  (turns = ["A", "B", "A", "B"]) →
  let P1 := (1 - pA) * (1 - pB) * pA
  let P2 := (1 - pA) * (1 - pB) * (1 - pA) * pB
  P1 + P2 = 19 / 400 :=
by
  intros _ _ _ _
  sorry

end probability_A_shot_twice_stops_l570_570202


namespace smallest_circle_radius_l570_570281

open Real

/-- There exists a circle which can enclose any planar closed polygonal 
line with perimeter 1, such that the radius of the circle is at most 1/4. -/
theorem smallest_circle_radius (p : ℝ → ℝ × ℝ) (h : ∀ t, (p 0 = p (1 / 1)) ∧ (∫ x in 0..1, ∥ (deriv p x) ∥ = 1)) :
  ∃ (c : ℝ × ℝ) (r : ℝ), (r = 1 / 4) ∧ (∀ t, dist (p t) c ≤ r) :=
sorry

end smallest_circle_radius_l570_570281


namespace smallest_solution_l570_570777

def smallest_positive_real_x : ℝ :=
  (131 : ℝ) / 11

theorem smallest_solution (x : ℝ) (hx : 0 < x) (H : ⌊x^2⌋ - x * ⌊x⌋ = 10) : x = smallest_positive_real_x :=
  sorry

end smallest_solution_l570_570777


namespace find_m_l570_570112

open Set

def A : Set ℕ := {1, 3, 5}
def B (m : ℕ) : Set ℕ := {1, m}
def C (m : ℕ) : Set ℕ := {1, m}

theorem find_m (m : ℕ) (h : A ∩ B m = C m) : m = 3 ∨ m = 5 :=
sorry

end find_m_l570_570112


namespace sum_of_intersections_l570_570176

-- Definitions and conditions
variable (n : ℕ)
def S : Finset ℕ := Finset.range (n + 1)

structure TripleSubset (S : Finset ℕ) :=
(A1 A2 A3 : Finset ℕ)
(h_union : A1 ∪ A2 ∪ A3 = S)

def T (S : Finset ℕ) : Finset (TripleSubset S) :=
  Finset.univ.filter (λ t, t.h_union)

-- Main statement
theorem sum_of_intersections :
  ∑ t in T (S n), (t.A1 ∩ t.A2 ∩ t.A3).card = n * 7^(n-1) :=
sorry

end sum_of_intersections_l570_570176


namespace find_smallest_x_satisfying_condition_l570_570797

theorem find_smallest_x_satisfying_condition :
  ∃ x : ℝ, 0 < x ∧ (⌊x^2⌋ - x * ⌊x⌋ = 10) ∧ x = 131 / 11 :=
by
  sorry

end find_smallest_x_satisfying_condition_l570_570797


namespace smallest_k_l570_570259

def a : ℕ → ℝ 
| 0     := 1
| 1     := real.root 19 2
| (n+2) := a (n+1) * (a n)^2

def accumulate (k : ℕ) (f : ℕ → ℝ) : ℝ :=
  (finset.range k).sum f

theorem smallest_k : ∃ k : ℕ, k = 17 ∧ (accumulate (k + 1) a).isInteger :=
by
  sorry

end smallest_k_l570_570259


namespace min_value_sqrt_expression_l570_570195

theorem min_value_sqrt_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ c : ℝ, c = 3 ∧ ∀ x : ℝ, x = (sqrt ((a^2 + b^2) * (4 * a^2 + b^2)) / (a * b)) → x ≥ c :=
sorry

end min_value_sqrt_expression_l570_570195


namespace volume_of_mixture_removed_replaced_l570_570678

noncomputable def volume_removed (initial_mixture: ℝ) (initial_milk: ℝ) (final_concentration: ℝ): ℝ :=
  (1 - final_concentration / initial_milk) * initial_mixture

theorem volume_of_mixture_removed_replaced (initial_mixture: ℝ) (initial_milk: ℝ) (final_concentration: ℝ) (V: ℝ):
  initial_mixture = 100 →
  initial_milk = 36 →
  final_concentration = 9 →
  V = 50 →
  volume_removed initial_mixture initial_milk final_concentration = V :=
by
  intros h1 h2 h3 h4
  have h5 : initial_mixture = 100 := h1
  have h6 : initial_milk = 36 := h2
  have h7 : final_concentration = 9 := h3
  rw [h5, h6, h7]
  sorry

end volume_of_mixture_removed_replaced_l570_570678


namespace melinda_textbooks_probability_l570_570560

theorem melinda_textbooks_probability :
  let total_books := 16
  let science_books := 4
  let box_capacities := [2, 4, 5, 5]
  let total_combinations := (nat.choose 16 5) * (nat.choose 11 5) * (nat.choose 6 2)
  let favorable_combinations := 2 * 5 * 12 * (nat.choose 11 5) * (nat.choose 6 2)
  let probability := (favorable_combinations : ℚ) / (total_combinations : ℚ)
  let reduced_fraction := (5 / 182 : ℚ)
  total_books = 16 ∧
  science_books = 4 ∧
  box_capacities = [2, 4, 5, 5] →
  probability = reduced_fraction :=
by
  sorry

end melinda_textbooks_probability_l570_570560


namespace smallest_positive_period_monotonically_decreasing_range_of_f_on_interval_l570_570470

noncomputable def f (x : ℝ) : ℝ := 2 * real.sqrt 3 * real.sin x * real.cos x - 2 * real.cos x ^ 2

theorem smallest_positive_period_monotonically_decreasing :
  (∀ x : ℝ, f (x + π) = f x) ∧ (∀ k : ℤ, ∀ x ∈ set.Icc (k * π + π / 3) (k * π + 5 * π / 6), isDecreasing (f x)) :=
sorry

theorem range_of_f_on_interval :
  set.image f (set.Icc 0 (π / 2)) = set.Icc (-2 : ℝ) 1 :=
sorry

end smallest_positive_period_monotonically_decreasing_range_of_f_on_interval_l570_570470


namespace find_element_with_mass_percentage_in_AlI3_l570_570386

-- Define atomic masses
def atomic_mass_Al : ℝ := 26.98
def atomic_mass_I : ℝ := 126.90

-- Define the formula for aluminum iodide
def aluminum_iodide_mass : ℝ := atomic_mass_Al + 3 * atomic_mass_I

-- Define mass percentages
def Al_mass_percentage : ℝ := (atomic_mass_Al / aluminum_iodide_mass) * 100
def I_mass_percentage : ℝ := (3 * atomic_mass_I / aluminum_iodide_mass) * 100

-- Define the problem statement
theorem find_element_with_mass_percentage_in_AlI3 : I_mass_percentage = 93.38 := by 
sorry

end find_element_with_mass_percentage_in_AlI3_l570_570386


namespace meaningful_expression_iff_l570_570896

theorem meaningful_expression_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (Real.sqrt (x - 3))) ↔ x > 3 := by
  sorry

end meaningful_expression_iff_l570_570896


namespace feasibility_orderings_l570_570368

theorem feasibility_orderings (a : ℝ) :
  (a ≠ 0) →
  (∀ a > 0, a < 2 * a ∧ 2 * a < 3 * a + 1) ∧
  ¬∃ a, a < 3 * a + 1 ∧ 3 * a + 1 < 2 * a ∧ 2 * a < 3 * a + 1 ∧ a ≠ 0 ∧ a > 0 ∧ a < -1 / 2 ∧ a < 0 ∧ a < -1 ∧ a < -1 / 2 ∧ a < -1 / 2 ∧ a < 0 :=
sorry

end feasibility_orderings_l570_570368


namespace Emily_money_made_l570_570030

def price_per_bar : ℕ := 4
def total_bars : ℕ := 8
def bars_sold : ℕ := total_bars - 3
def money_made : ℕ := bars_sold * price_per_bar

theorem Emily_money_made : money_made = 20 :=
by
  sorry

end Emily_money_made_l570_570030


namespace find_prob_B_l570_570758

/-- Define mutual independence of events A, B, and C. --/
def mutually_independent (A B C : Prop → Prop) : Prop :=
  (∀ (x : Prop), A x ∧ B x → C x) ∧
  (∀ (y : Prop), B y ∧ A y → C y) ∧
  (∀ (z : Prop), A z ∧ C z → B z)

/-- Given probabilities: --/
variables (P : (Prop → Prop) → ℝ)
variables (A B C : Prop → Prop)

axiom prob_A_div_B : P (λ x, A x ∧ B x) = 1/6
axiom prob_compl_B_div_C : P (λ x, (¬ B x) ∧ C x) = 1/8
axiom prob_A_div_B_compl_C : P (λ x, A x ∧ B x ∧ (¬ C x)) = 1/8
axiom independence : mutually_independent A B C

/-- Theorem to prove that P(B) = 1/2 --/
theorem find_prob_B : P B = 1/2 :=
sorry

end find_prob_B_l570_570758


namespace exponential_function_example_l570_570351

def is_exponential_function (f : ℝ → ℝ) : Prop :=
  ∃ a > 0, a ≠ 1 ∧ ∀ x, f x = a ^ x

theorem exponential_function_example : is_exponential_function (fun x => 3 ^ x) :=
by
  sorry

end exponential_function_example_l570_570351


namespace number_of_valid_orders_eq_Catalan_l570_570274

-- Define the conditions for the sequence to be valid.
def valid_order (seq : List ℤ) : Prop :=
  seq.length = 2 * n ∧ 
  seq.count (+1) = n ∧ 
  seq.count (-1) = n ∧ 
  ∀ k, k < seq.length → 0 ≤ seq.take (k + 1).sum

-- Prove the number of valid sequences is the n-th Catalan number.
theorem number_of_valid_orders_eq_Catalan (n : ℕ) : 
  ∃ seqs : List (List ℤ), 
    (∀ seq ∈ seqs, valid_order seq) ∧ 
    seqs.length = Catalan n :=
sorry

end number_of_valid_orders_eq_Catalan_l570_570274


namespace count_odd_digits_base4_157_l570_570391

def base_repr (n b : ℕ) : List ℕ :=
  if b ≤ 1 then [n] else (List.unfoldr (λ x, if x = 0 then none else some (x % b, x / b)) n).reverse

def count_odd_digits (l : List ℕ) : ℕ :=
  l.countp (λ d, d % 2 = 1)

theorem count_odd_digits_base4_157 : count_odd_digits (base_repr 157 4) = 3 := by
  sorry

end count_odd_digits_base4_157_l570_570391


namespace general_term_defines_sequence_l570_570870

/-- Sequence definition -/
def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = (2 * a n + 6) / (a n + 1)

/-- General term formula -/
def general_term (n : ℕ) : ℚ :=
  (3 * 4 ^ n + 2 * (-1) ^ n) / (4 ^ n - (-1) ^ n)

/-- Theorem stating that the general term formula defines the sequence -/
theorem general_term_defines_sequence : ∀ (a : ℕ → ℚ), seq a → ∀ n, a n = general_term n :=
by
  intros a h_seq n
  sorry

end general_term_defines_sequence_l570_570870


namespace area_of_region_l570_570305

theorem area_of_region :
  (∃ R : ℝ, R = 50 * real.pi ∧
  ∀ (x y : ℝ), (y^100 + 1/y^100 ≤ x^100 + 1/x^100) ∧ (x^2 + y^2 ≤ 100) →
  x^2 + y^2 ≤ 100 → sorry) :=
begin
  sorry
end

end area_of_region_l570_570305


namespace definite_integral_correct_l570_570835

noncomputable def integral_value (a : ℝ) (h₀ : a > 0) (h₁ : is_const_term_15 : (∃ r : ℕ, r = 2 ∧ (nat.choose 6 r) * a^(6-r) * (-1)^r = 15)) : ℝ :=
  ∫ x in -a..a, x^2 + x + real.sqrt (1 - x^2)

theorem definite_integral_correct {a : ℝ} (h₀ : a > 0) 
  (h₁ : (∃ r : ℕ, r = 2 ∧ (nat.choose 6 r) * a^(6-r) * (-1)^r = 15)) :
  integral_value a h₀ h₁ = 2/3 + real.pi / 2 :=
begin
  sorry
end

end definite_integral_correct_l570_570835


namespace find_prob_B_l570_570760

variable {Ω : Type} {P : Ω → Prop}

-- Defining events as predicates on the type Ω
variable (A B C : Ω → Prop)

-- Assumptions given in the problem
axiom mutual_independence : IsIndependent P [A, B, C]
axiom P_A_and_B : P (λ ω, A ω ∧ B ω) = 1/6
axiom P_not_B_and_C : P (λ ω, ¬B ω ∧ C ω) = 1/8
axiom P_A_and_B_and_not_C : P (λ ω, A ω ∧ B ω ∧ ¬C ω) = 1/8

-- Final statement to prove
theorem find_prob_B : P (λ ω, B ω) = 1/2 := sorry

end find_prob_B_l570_570760


namespace h_inequality_l570_570081

-- Assumptions
variables {α : Type*} [linear_ordered_field α]
variables {f g : α → α}

-- Key conditions
axiom f_symm (x : α) : f (2 + x) = f (2 - x)
axiom g_periodic (x : α) : g (x + 1) = g (x - 1)
axiom f_decreasing (a b : α) : 2 ≤ a → a ≤ b → f b ≤ f a

-- Definition of h
def h (x : α) : α := f x * |g x|

-- Main statement
theorem h_inequality : h (-2) ≤ h 4 :=
sorry

end h_inequality_l570_570081


namespace geom_seq_product_l570_570155

noncomputable def geom_sequence (a r : ℝ) (n : ℕ) : ℝ :=
  a * r^n

variable (a r : ℝ)

def a_5 := geom_sequence a r 5
def a_14 := geom_sequence a r 14
def a_8 := geom_sequence a r 8
def a_9 := geom_sequence a r 9
def a_10 := geom_sequence a r 10
def a_11 := geom_sequence a r 11

theorem geom_seq_product :
  a_5 * a_14 = 5 →
  a_8 * a_9 * a_10 * a_11 = 25 :=
by
  sorry

end geom_seq_product_l570_570155


namespace odd_digits_in_base4_of_157_l570_570396

theorem odd_digits_in_base4_of_157 : 
  let b4 := 2 * 4^3 + 1 * 4^2 + 3 * 4^1 + 1 * 4^0 in
  b4 = 157 → (nat.digits 4 157).countp (λ d, d % 2 = 1) = 3 := 
by
  intros
  sorry

end odd_digits_in_base4_of_157_l570_570396


namespace tan_angleQDE_eq_l570_570573

noncomputable def tan_angleQDE (Q D E F : Point) (d_eq : dist D E = 10) (e_fq_eq : dist E F = 11) (f_d_eq : dist F D = 12) 
  (angle_eq : ∠(Q, D, E) = ∠(Q, E, F) ∧ ∠(Q, E, F) = ∠(Q, F, D)) : ℝ :=
  if dist Q D = 0 then 0 else abs ((108.16) / (182.5))

-- The goal is to prove the statement below
theorem tan_angleQDE_eq (Q D E F : Point) (d_eq : dist D E = 10) (e_fq_eq : dist E F = 11) (f_d_eq : dist F D = 12) 
  (angle_eq : ∠(Q, D, E) = ∠(Q, E, F) ∧ ∠(Q, E, F) = ∠(Q, F, D)) : tan (∠(Q, D, E)) = 108.16 / 182.5 := sorry

end tan_angleQDE_eq_l570_570573


namespace max_x_satisfying_inequality_is_2_l570_570643

noncomputable def max_x_satisfying_inequality : ℤ :=
  let s := {k : ℤ | (k^2 + 5*k < 30)}
  Nat.find_greatest (λk, k ∈ s) 2

theorem max_x_satisfying_inequality_is_2 : max_x_satisfying_inequality = 2 := 
by
  sorry

end max_x_satisfying_inequality_is_2_l570_570643


namespace max_tangent_lines_values_exist_l570_570580

-- Defining the problem conditions
def first_circle_radius : ℝ := 5
def second_circle_radius : ℝ := 7
def variable_distance : ℝ → ℝ → Prop := λ d r, r = d -- just a placeholder as the distance is variable

-- State the theorem
theorem max_tangent_lines_values_exist :
  ∃ k : ℕ, k = 5 := sorry

end max_tangent_lines_values_exist_l570_570580


namespace total_prep_time_is_8_l570_570223

-- Defining the conditions
def prep_vocab_sentence_eq := 3
def prep_analytical_writing := 2
def prep_quantitative_reasoning := 3

-- Stating the total preparation time
def total_prep_time := prep_vocab_sentence_eq + prep_analytical_writing + prep_quantitative_reasoning

-- The Lean statement of the mathematical proof problem
theorem total_prep_time_is_8 : total_prep_time = 8 := by
  sorry

end total_prep_time_is_8_l570_570223


namespace ellipse_and_circle_l570_570449

noncomputable theory

-- Define the initial conditions and ellipse properties
def ellipse_eq (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (∃ e : ℝ, e = sqrt 3 / 2 ∧ c = a * e 
    ∧ a^2 - b^2 = c^2 ∧ 2 * b^2 / a = 1)

-- Prove the equation of the ellipse
def ellipse_equation : Prop :=
  ∀ a b : ℝ, ellipse_eq a b → (∃ k: ℝ, k = 1 ∧ ∀ x y : ℝ, (x^2 / 4 + y^2 = k))

-- Prove the existence and equation of the circle O
def fixed_circle (a b : ℝ) : Prop :=
  ∀ P Q : ℝ × ℝ, ellipse_eq a b → 
  P.1 > 0 ∧ P.2 > 0 ∧ Q.1 < 0 ∧ Q.2 > 0 ∧
  P.1 * Q.1 + P.2 * Q.2 = 0 →
  (∃ r : ℝ, r^2 = 4/5 ∧ ∀ x y : ℝ, (x^2 + y^2 = r^2 
    ∧ ∀ PQ : ℝ, line_tangent_to_circle PQ (x^2 + y^2 = r^2))

-- Main theorem combining both parts
theorem ellipse_and_circle : Prop :=
  ∀ a b : ℝ, ellipse_eq a b →
    (ellipse_equation a b ∧ fixed_circle a b)
    
#print axioms ellipse_and_circle

end ellipse_and_circle_l570_570449


namespace total_dividend_is_825_l570_570693

def face_value : ℝ := 100
def premium_rate_A : ℝ := 20 / 100
def discount_rate_B : ℝ := 10 / 100
def dividend_rate_A : ℝ := 7 / 100
def dividend_rate_B : ℝ := 5 / 100
def total_investment : ℝ := 14400
def investment_ratio : ℝ := 2

noncomputable def total_dividend_received : ℝ :=
  let investment_B := total_investment / (investment_ratio + 1)
  let investment_A := investment_ratio * investment_B
  let price_per_share_A := face_value * (1 + premium_rate_A)
  let price_per_share_B := face_value * (1 - discount_rate_B)
  let shares_A := investment_A / price_per_share_A
  let shares_B := (investment_B / price_per_share_B).toInt
  let dividend_A := shares_A * face_value * dividend_rate_A
  let dividend_B := shares_B * face_value * dividend_rate_B
  dividend_A + dividend_B

theorem total_dividend_is_825 : total_dividend_received = 825 := by
  sorry

end total_dividend_is_825_l570_570693


namespace minimum_value_of_f_l570_570021

noncomputable def f (x : ℝ) : ℝ := sin x * cos x - cos x ^ 2 + 1 / 2

theorem minimum_value_of_f : 
  ∃ x ∈ set.Icc 0 (π / 2), ∀ y ∈ set.Icc 0 (π / 2), f y ≥ f x ∧ f x = -1 / 2 :=
sorry

end minimum_value_of_f_l570_570021


namespace second_number_is_22_l570_570236

theorem second_number_is_22 
    (A B : ℤ)
    (h1 : A - B = 88) 
    (h2 : A = 110) :
    B = 22 :=
by
  sorry

end second_number_is_22_l570_570236


namespace math_problem_proof_l570_570807

noncomputable def question_to_equivalent_proof_problem : Prop :=
  ∃ (p q r : ℤ), 
    (p + q + r = 0) ∧ 
    (p * q + q * r + r * p = -2023) ∧ 
    (|p| + |q| + |r| = 84)

theorem math_problem_proof : question_to_equivalent_proof_problem := 
  by 
    -- proof goes here
    sorry

end math_problem_proof_l570_570807


namespace tan_sum_in_nat_l570_570221

theorem tan_sum_in_nat (n : ℕ) : 
  let α := (Real.pi / 7) in 
  let f := (fun n => Real.tan^(2 * n) α + Real.tan^(2 * n) (2 * α) + Real.tan^(2 * n) (3 * α)) in 
  f n ∈ ℕ := sorry

end tan_sum_in_nat_l570_570221


namespace necessary_not_sufficient_l570_570892

theorem necessary_not_sufficient (a b : ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 1 → ax + b > 0) :
  a + 2b > 0 ∧ ¬(∀ a b, a + 2b > 0 → ∀ x, 0 ≤ x ∧ x ≤ 1 → ax + b > 0) :=
by
  refine ⟨_, _⟩;
  sorry

end necessary_not_sufficient_l570_570892


namespace find_pq_at_x1_l570_570535

theorem find_pq_at_x1 (p q : ℕ[X])
  (hp : p.monic) (hq : q.monic)
  (hnp : p.degree > 0) (hnq : q.degree > 0)
  (hint : ∀ x : ℕ, x ∈ p.coeffs → x ∈ ℤ)
  (hint : ∀ x : ℕ, x ∈ q.coeffs → x ∈ ℤ)
  (hx : X ^ 6 - 50 * X ^ 3 + 4 = p * q) : 
  p.eval 1 + q.eval 1 = 6 :=
  sorry

end find_pq_at_x1_l570_570535


namespace song_book_cost_correct_l570_570561

noncomputable def cost_of_trumpet : ℝ := 145.16
noncomputable def total_spent : ℝ := 151.00
noncomputable def cost_of_song_book : ℝ := total_spent - cost_of_trumpet

theorem song_book_cost_correct : cost_of_song_book = 5.84 :=
  by
    sorry

end song_book_cost_correct_l570_570561


namespace cos_value_l570_570833

variable (α : ℝ)

theorem cos_value (h : Real.sin (π / 4 + α) = 2 / 3) : Real.cos (π / 4 - α) = 2 / 3 := 
by 
  sorry 

end cos_value_l570_570833


namespace num_ordered_triples_l570_570701

/-
Let Q be a right rectangular prism with integral side lengths a, b, and c such that a ≤ b ≤ c, and b = 2023.
A plane parallel to one of the faces of Q cuts Q into two prisms, one of which is similar to Q, and both have nonzero volume.
Prove that the number of ordered triples (a, b, c) such that b = 2023 is 7.
-/

theorem num_ordered_triples (a c : ℕ) (h : a ≤ 2023 ∧ 2023 ≤ c) (ac_eq_2023_squared : a * c = 2023^2) :
  ∃ count, count = 7 :=
by {
  sorry
}

end num_ordered_triples_l570_570701


namespace projection_problem_l570_570939

noncomputable def norm (v : ℝ) : ℝ := sorry -- Definition of norm, not elaborated here

noncomputable def proj (a b : ℝ) : ℝ := sorry -- Definition of projection, not elaborated here

theorem projection_problem 
  (v w p q r : ℝ)
  (h_proj_pw : p = proj v w)
  (h_proj_qv : q = proj p v)
  (h_proj_rw : r = proj q w)
  (h_norm_ratio : norm p / norm v = 3 / 8) : 
  norm r / norm w = 81 / 4096 :=
by sorry

end projection_problem_l570_570939


namespace vector_sum_parallel_l570_570120

variable (x : ℝ)
def vector_a : ℝ × ℝ := (2, 1)
def vector_b : ℝ × ℝ := (x, 2)

theorem vector_sum_parallel (h_parallel : vector_a ∥ vector_b)
: vector_a + vector_b = (6, 3) :=
sorry

end vector_sum_parallel_l570_570120


namespace probability_closer_to_seven_than_zero_l570_570329

theorem probability_closer_to_seven_than_zero : 
  let segment_length : ℝ := 9
  let midpoint : ℝ := (0 + 7) / 2
  let favorable_region_start : ℝ := midpoint
  let favorable_region_end : ℝ := 9
  let favorable_region_length : ℝ := favorable_region_end - favorable_region_start
  let probability : ℝ := favorable_region_length / segment_length
  in probability = 5.5 / 9 := 
begin
  sorry
end

end probability_closer_to_seven_than_zero_l570_570329


namespace cylinder_volume_l570_570387

/--
  Given:
  Let l be the side length of a square, l = 14 centimeters
  A cylinder is formed by rotating the square about its vertical line of symmetry.

  Prove:
  The volume of the cylinder is 686 * π cubic centimeters.
-/
theorem cylinder_volume (l : ℝ) (h : l = 14) : (cylinder_volume l 7) = 686 * Real.pi :=
by
  sorry

noncomputable def cylinder_volume (l r : ℝ) : ℝ :=
  Real.pi * (r^2) * l

end cylinder_volume_l570_570387


namespace min_value_at_six_l570_570289

def f (x : ℝ) : ℝ := x^2 - 12 * x + 36

theorem min_value_at_six : ∃ x : ℝ, x = 6 ∧ (∀ y : ℝ, f y ≥ f x) :=
by
  use 6
  split
  case h.left =>
    rfl
  case h.right =>
    intro y
    have h : (y - 6)^2 ≥ 0 := sq_nonneg (y - 6)
    linarith

end min_value_at_six_l570_570289


namespace hyperbola_eccentricity_correct_l570_570839

def hyperbola_center_origin : Prop :=
  center_at_origin

def hyperbola_focus_y_axis : Prop :=
  focus_on_y_axis

def asymptote_parallel : Prop :=
  asymptote_parallel_to_sqrt2x_minus_y_minus_1

def eccentricity_of_hyperbola : ℝ :=
  sqrt(3) / 2

theorem hyperbola_eccentricity_correct :
  hyperbola_center_origin → 
  hyperbola_focus_y_axis →
  asymptote_parallel → 
  eccentricity_of_hyperbola = sqrt(3) / 2 :=
by
  intros
  -- Definitions and logic required for proof
  sorry

end hyperbola_eccentricity_correct_l570_570839


namespace sum_of_rational_roots_is_6_l570_570407

noncomputable def h : Polynomial ℚ := Polynomial.X^3 - 6 * Polynomial.X^2 + 11 * Polynomial.X - 6

theorem sum_of_rational_roots_is_6 : (h.roots.filter (λ r, r.is_rat)).sum = 6 := by
  sorry

end sum_of_rational_roots_is_6_l570_570407


namespace min_distance_to_line_l570_570866

-- Define the parametric equation of the line l
def parametric_line (t : ℝ) : ℝ × ℝ :=
  ⟨1 + 1/2 * t, (√3 / 2) * t⟩

-- Define the polar equation of the curve C1
def polar_curve (θ : ℝ) : ℝ :=
  1

-- Cartesian equation of curve C1
def cartesian_curve (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

-- Parametric equation of curve C1
def parametric_curve_C1 (θ : ℝ) : ℝ × ℝ :=
  ⟨Real.cos θ, Real.sin θ⟩

-- Parametric equation of curve C2 after compression
def parametric_curve_C2 (θ : ℝ) : ℝ × ℝ :=
  ⟨1/2 * Real.cos θ, (√3 / 2) * Real.sin θ⟩

-- Minimum distance from a point on C2 to the line l
theorem min_distance_to_line : 
  (∀ (θ : ℝ), ∃ (dmin : ℝ), dmin = 
  (√3 / 2) * |1/2 * Real.cos θ - (√3 / 2) * Real.sin θ - 1| → dmin = √3) :=
sorry

end min_distance_to_line_l570_570866


namespace price_of_battery_l570_570423

def cost_of_tire : ℕ := 42
def cost_of_tires (num_tires : ℕ) : ℕ := num_tires * cost_of_tire
def total_cost : ℕ := 224
def num_tires : ℕ := 4
def cost_of_battery : ℕ := total_cost - cost_of_tires num_tires

theorem price_of_battery : cost_of_battery = 56 := by
  sorry

end price_of_battery_l570_570423


namespace find_lambda_l570_570118

open Real EuclideanGeometry

def vector_a : EuclideanSpace ℝ (fin 3) := ![2, 1, 3]
def vector_b : EuclideanSpace ℝ (fin 3) := ![-1, 2, 1]

theorem find_lambda (λ : ℝ) (h : dot_product vector_a (vector_a - λ • vector_b) = 0) : λ = 14 / 3 :=
sorry

end find_lambda_l570_570118


namespace solve_problem_l570_570745

noncomputable def parametrization (s l t : ℝ) : ℝ × ℝ :=
  (-4 + t * l, s + t * (-3))

def line_equation (x : ℝ) : ℝ :=
  2 * x + 3

theorem solve_problem : 
  ∃ s l : ℝ, s = -5 ∧ l = -3/2 ∧ (∀ t : ℝ, parametrization s l t = (x t, y t) → y t = line_equation (x t)) := 
sorry

end solve_problem_l570_570745


namespace islet_cell_transplant_meds_l570_570687

-- Define the conditions
def indicators_post_transplant : Type := 
| urine_sugar_and_insulin
| insulin_and_antiallergics
| blood_sugar_and_insulin
| blood_sugar_and_immunosuppressants

open indicators_post_transplant

-- Define the problem
theorem islet_cell_transplant_meds : 
  ∃ (cure_indicator medication : indicators_post_transplant), 
    (cure_indicator = blood_sugar_and_immunosuppressants) →
    (medication = blood_sugar_and_immunosuppressants) :=
by
  -- Skipping the proof
  sorry

end islet_cell_transplant_meds_l570_570687


namespace max_C_trees_l570_570634

-- Define the conditions
def price_ratio_A_B_C := (2, 2, 3)
def price_A := 200
def total_budget := 220120
def total_trees := 1000

-- Prove the maximum number of C-type scenic trees that can be purchased
theorem max_C_trees :
  ∃ x : ℕ, (∀ y : ℕ, (200 * (total_trees - y) + 300 * y ≤ total_budget) → y ≤ 201) ∧ (200 * (total_trees - x) + 300 * x ≤ total_budget) ∧ x = 201 :=
begin
  sorry
end

end max_C_trees_l570_570634


namespace construct_triangle_num_of_solutions_l570_570012

theorem construct_triangle_num_of_solutions
  (r : ℝ) -- Circumradius
  (beta_gamma_diff : ℝ) -- Angle difference \beta - \gamma
  (KA1 : ℝ) -- Segment K A_1
  (KA1_lt_r : KA1 < r) -- Segment K A1 should be less than the circumradius
  (delta : ℝ := beta_gamma_diff) : 1 ≤ num_solutions ∧ num_solutions ≤ 2 :=
sorry

end construct_triangle_num_of_solutions_l570_570012


namespace factor_expression_l570_570034

theorem factor_expression (x : ℤ) : 63 * x + 28 = 7 * (9 * x + 4) :=
by sorry

end factor_expression_l570_570034


namespace rectangle_perimeter_126_l570_570260

/-- Define the sides of the rectangle in terms of a common multiplier -/
def sides (x : ℝ) : ℝ × ℝ := (4 * x, 3 * x)

/-- Define the area of the rectangle given the common multiplier -/
def area (x : ℝ) : ℝ := (4 * x) * (3 * x)

example : ∃ (x : ℝ), area x = 972 :=
by
  sorry

/-- Calculate the perimeter of the rectangle given the common multiplier -/
def perimeter (x : ℝ) : ℝ := 2 * ((4 * x) + (3 * x))

/-- The final proof statement, stating that the perimeter of the rectangle is 126 meters,
    given the ratio of its sides and its area. -/
theorem rectangle_perimeter_126 (x : ℝ) (h: area x = 972) : perimeter x = 126 :=
by
  sorry

end rectangle_perimeter_126_l570_570260


namespace vector_b_length_l570_570875

-- Define the vectors and the known conditions
def a : ℝ × ℝ := (1, real.sqrt 3)

-- Angle between the vectors
def theta : ℝ := real.pi / 3

-- Given condition
def condition (b : ℝ × ℝ) : Prop :=
  real.sqrt ( ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2)) = 2 * real.sqrt 3

-- Length of vector b to be proven
def length_b (b : ℝ × ℝ) : ℝ := real.sqrt (b.1^2 + b.2^2)

-- Prove that the length of b is 2
theorem vector_b_length : ∃ b : ℝ × ℝ, condition b ∧ length_b b = 2 := by
  sorry

end vector_b_length_l570_570875


namespace sum_of_rational_roots_of_h_l570_570410

theorem sum_of_rational_roots_of_h :
  let h(x : ℚ) := x^3 - 6*x^2 + 11*x - 6
  h(1) = 0 ∧ h(2) = 0 ∧ h(3) = 0 →
  (1 + 2 + 3 = 6) := sorry

end sum_of_rational_roots_of_h_l570_570410


namespace find_f_l570_570088

variables (a b c : ℝ)
noncomputable def f (x : ℝ) := a * x ^ 2 + b * x + c
noncomputable def g (x : ℝ) := a * x + b

theorem find_f (h : ∀ x ∈ set.Icc (-1 : ℝ) 1, g x ≤ 2) :
  f = λ x, 2 * x ^ 2 - 1 :=
sorry

end find_f_l570_570088


namespace sinusoidal_function_properties_l570_570856

theorem sinusoidal_function_properties 
  (A w φ : ℝ) (A_pos : 0 < A) (w_pos : 0 < w) (φ_bound : 0 < φ ∧ φ < π / 2)
  (min_period_positive : ∀ x, f (x + π) = f x)
  (lowest_point_M : f (2 * π / 3) = -2)
  : f = (λ x, 2 * sin (2 * x + π / 6)) ∧ 
    ∀ x, x ∈ Icc 0 (π / 2) → f x ∈ Icc (-1 : ℝ) 2 :=
sorry

end sinusoidal_function_properties_l570_570856


namespace min_value_quadratic_l570_570286

theorem min_value_quadratic (x : ℝ) : 
  ∀ x ∈ ℝ, x = 6 ↔ x^2 - 12x + 36 = (x - 6)^2 ∨ (x - 6)^2 >= 0 := 
begin
  sorry
end

end min_value_quadratic_l570_570286


namespace ellipse_equation_l570_570713

theorem ellipse_equation
  (centered_at_origin : True)
  (focus_on_axis : True)
  (eccentricity : ℝ)
  (eccentricity_eq_sqrt3_div_2 : eccentricity = Real.sqrt 3 / 2)
  (point_on_ellipse: Prop := (2,0) ∈ ellipse_points)
  (ellipse_equation: Prop) : 
  ellipse_equation = (x^2 / 4 + y^2 = 1) ∨ (x^2 / 4 + y^2 / 16 = 1) :=
sorry

end ellipse_equation_l570_570713


namespace p_iff_q_l570_570886

variable (a b : ℝ)

def p := a > 2 ∧ b > 3

def q := a + b > 5 ∧ (a - 2) * (b - 3) > 0

theorem p_iff_q : p a b ↔ q a b := by
  sorry

end p_iff_q_l570_570886


namespace problem_statement_l570_570655

noncomputable def seq_sub_triples: ℚ :=
  let a := (5 / 6 : ℚ)
  let b := (1 / 6 : ℚ)
  let c := (1 / 4 : ℚ)
  a - b - c

theorem problem_statement : seq_sub_triples = 5 / 12 := by
  sorry

end problem_statement_l570_570655


namespace kendra_baked_batches_l570_570533

-- Definitions based on conditions
def cookiesPerBatch : ℕ := 12
def familyMembers : ℕ := 4
def chipsPerCookie : ℕ := 2
def chipsEatenPerMember : ℕ := 18

-- Theorem stating the goal
theorem kendra_baked_batches : ∃ n : ℕ, n = 3 ∧ cookiesPerBatch * n = familyMembers * (chipsEatenPerMember / chipsPerCookie) :=
begin
  sorry
end

end kendra_baked_batches_l570_570533


namespace B_wins_probability_l570_570143

def a_values : Finset ℕ := {2, 3}
def b_values : Finset ℕ := {2, 3}

def win_condition (a b : ℕ) : Prop :=
|a - b| ≤ 1

def total_events : ℕ := a_values.card * b_values.card

def winning_events : Finset (ℕ × ℕ) :=
(a_values.product b_values).filter (λ ab, win_condition ab.1 ab.2)

def probability_of_winning : ℚ :=
(winning_events.card : ℚ) / (total_events : ℚ)

theorem B_wins_probability :
  probability_of_winning = 2 / 3 :=
by sorry

end B_wins_probability_l570_570143


namespace min_heaviest_weight_integer_min_heaviest_weight_non_integer_l570_570270

variable {W : Set ℝ} (W_finite : Finite W) (W_card : (W.card ≤ 20)) (W_meas : ∀ w, w ∈ W → ∃ n, n ∈ ℤ ∧ n ≥ 1 ∧ n ≤ 1997)

theorem min_heaviest_weight_integer :
  (∀ w ∈ W, ∃ n ∈ ℤ, n = w) → ∃ h ∈ W, ∀ w ∈ W, h ≥ w ∧ h = 146 :=
sorry -- proof required

theorem min_heaviest_weight_non_integer :
  ∃ h ∈ W, ∀ w ∈ W, h ≥ w ∧ h = 145.25 :=
sorry -- proof required

end min_heaviest_weight_integer_min_heaviest_weight_non_integer_l570_570270


namespace min_abs_sum_l570_570321

noncomputable def f (α γ z : ℂ) : ℂ := (3 + 2*complex.I) * z^3 + (3 + 2*complex.I) * z^2 + α * z + γ

theorem min_abs_sum (α γ : ℂ)
  (h1 : ∀ α γ, f α γ 1 ∈ ℝ)
  (h2 : ∀ α γ, f α γ complex.I ∈ ℝ) :
  ∃ α γ, complex.abs α + complex.abs γ = real.sqrt 2 := sorry

end min_abs_sum_l570_570321


namespace max_y_value_l570_570192

noncomputable def y (x : ℝ) : ℝ := |x + 1| - 2 * |x| + |x - 2|

theorem max_y_value : ∃ α, (∀ x, -1 ≤ x ∧ x ≤ 2 → y x ≤ α) ∧ α = 3 := by
  sorry

end max_y_value_l570_570192


namespace fraction_of_yard_occupied_by_flower_beds_l570_570699

-- Conditions
variables {a b : ℝ} (h1 : a = 18) (h2 : b = 30)

-- Dimensions of the triangles and area of the yard
def leg_length_of_triangle : ℝ := (b - a) / 2
def area_of_triangle : ℝ := (1 / 2) * (leg_length_of_triangle h1 h2) ^ 2
def total_area_flower_beds : ℝ := 2 * area_of_triangle h1 h2
def area_of_yard : ℝ := b * leg_length_of_triangle h1 h2

-- Statement to prove
theorem fraction_of_yard_occupied_by_flower_beds :
  total_area_flower_beds h1 h2 / area_of_yard h1 h2 = 1 / 5 :=
  sorry

end fraction_of_yard_occupied_by_flower_beds_l570_570699


namespace find_a_of_cool_frog_meeting_l570_570929

-- Question and conditions
def frogs : ℕ := 16
def friend_probability : ℚ := 1 / 2
def cool_condition (f: ℕ → ℕ) : Prop := ∀ i, f i % 4 = 0

-- Example theorem where we need to find 'a'
theorem find_a_of_cool_frog_meeting :
  let a := 1167
  let b := 2 ^ 41
  ∀ (f: ℕ → ℕ), ∀ (p: ℚ) (h: p = friend_probability),
    (cool_condition f) →
    (∃ a b, a / b = p ∧ a % gcd a b = 0 ∧ gcd a b = 1) ∧ a = 1167 :=
by
  sorry

end find_a_of_cool_frog_meeting_l570_570929


namespace MikeSalaryNow_l570_570800

-- Definitions based on conditions
def FredSalary  := 1000   -- Fred's salary five months ago
def MikeSalaryFiveMonthsAgo := 10 * FredSalary  -- Mike's salary five months ago
def SalaryIncreasePercent := 40 / 100  -- 40 percent salary increase
def SalaryIncrease := SalaryIncreasePercent * MikeSalaryFiveMonthsAgo  -- Increase in Mike's salary

-- Statement to be proved
theorem MikeSalaryNow : MikeSalaryFiveMonthsAgo + SalaryIncrease = 14000 :=
by
  -- Proof is skipped
  sorry

end MikeSalaryNow_l570_570800


namespace find_p_l570_570011

noncomputable def vector_a : ℝ × ℝ × ℝ := (2, -2, 4)
noncomputable def vector_b : ℝ × ℝ × ℝ := (3, 1, 1)

theorem find_p (v p : ℝ × ℝ × ℝ) 
  (h1 : ∀ t : ℝ, p = (2 + t, -2 + 3 * t, 4 - 3 * t)) 
  (h2 : p • (1, 3, -3) = 0) 
  (h3 : ∃ k1 k2 : ℝ, vector_a = k1 • p ∧ vector_b = k2 • p) :  
 p = (18, 46, -44) := 
 sorry

end find_p_l570_570011


namespace surface_area_and_volume_l570_570204

-- Define the surface area and volume for a given triangle ABC rotated around a tangent at point A
noncomputable def area_and_volume (a b c h S : ℝ) : ℝ × ℝ :=
  let f := π * 2 * S * (b^2 + c^2) / (b * c)
  let t := (4 * S^2 * π * (b^2 + c^2)) / (3 * a * b * c)
  (f, t)

-- Define the theorem to prove that the calculated surface area and volume are correct
theorem surface_area_and_volume (a b c h S : ℝ) :
  area_and_volume a b c h S =
    (π * 2 * S * (b^2 + c^2) / (b * c),
     (4 * S^2 * π * (b^2 + c^2)) / (3 * a * b * c)) := by
sorry

end surface_area_and_volume_l570_570204


namespace cos_theta_eq_neg_2_div_sqrt_13_l570_570061

theorem cos_theta_eq_neg_2_div_sqrt_13 
  (θ : ℝ) 
  (h1 : 0 < θ) 
  (h2 : θ < π) 
  (h3 : Real.tan θ = -3/2) : 
  Real.cos θ = -2 / Real.sqrt 13 :=
sorry

end cos_theta_eq_neg_2_div_sqrt_13_l570_570061


namespace parity_of_expression_l570_570941

theorem parity_of_expression (a b c : ℕ) (ha : odd a) (hb : even b) : 
  if (odd c) then even (3^a + (b+1)^2 * c) else odd (3^a + (b+1)^2 * c) := 
sorry

end parity_of_expression_l570_570941


namespace compare_neg_nine_and_neg_sqrt_eighty_l570_570004

theorem compare_neg_nine_and_neg_sqrt_eighty : -9 < -Real.sqrt 80 :=
by {
  have h1 : 9 * 9 = 81 := by norm_num,
  have h2 : (Real.sqrt 80) * (Real.sqrt 80) = 80 := by simp [Real.sqrt_mul_self, ne_of_lt, zero_lt_iff_ne_zero],
  have h3 : 81 > 80 := by norm_num,
  have h4 : ∀ x y : ℝ, x > y → -x < -y := λ x y h => by linarith,
  exact h4 9 (Real.sqrt 80) (by linarith [h1, h2, h3])
}

end compare_neg_nine_and_neg_sqrt_eighty_l570_570004


namespace log_evaluation_exp_radical_evaluation_l570_570730

-- Logarithmic properties and evaluation
theorem log_evaluation : log 6 9 + 2 * log 6 2 = 2 := sorry

-- Exponential and radical simplifications
theorem exp_radical_evaluation : 
  exp 0 + sqrt ((1 - sqrt 2)^2) - 8^(1/6) = 1 + sqrt 5 - sqrt 2 - 2^(1/3) := sorry

end log_evaluation_exp_radical_evaluation_l570_570730


namespace distance_to_base_is_42_l570_570029

theorem distance_to_base_is_42 (x : ℕ) (hx : 4 * x + 3 * (x + 3) = x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6)) :
  4 * x = 36 ∨ 4 * x + 6 = 42 := 
by
  sorry

end distance_to_base_is_42_l570_570029


namespace smallest_solution_l570_570778

def smallest_positive_real_x : ℝ :=
  (131 : ℝ) / 11

theorem smallest_solution (x : ℝ) (hx : 0 < x) (H : ⌊x^2⌋ - x * ⌊x⌋ = 10) : x = smallest_positive_real_x :=
  sorry

end smallest_solution_l570_570778


namespace track_length_l570_570224

theorem track_length {y : ℝ} (d0 : y / 2 ≠ 0) (d1 : 90 / (y / 2 - 90) = (y / 2 - 110) / (y / 2 + 110)) :
  y = 360 :=
begin
  sorry
end

end track_length_l570_570224


namespace train_length_l570_570705

theorem train_length
  (train_speed_kmph : ℝ)
  (time_seconds : ℝ)
  (platform_length_m : ℝ) :
  train_speed_kmph = 102 →
  time_seconds = 50 →
  platform_length_m = 396.78 →
  let speed_mps := train_speed_kmph * (1000 / 3600)
  let distance_m := speed_mps * time_seconds 
  let train_length_m := distance_m - platform_length_m 
  train_length_m = 1019.89 
:= by
  intros h1 h2 h3
  rw [h1, h2, h3]
  let speed_mps := 102 * (1000 / 3600)
  let distance_m := speed_mps * 50
  let train_length_m := distance_m - 396.78
  sorry

end train_length_l570_570705


namespace winning_ticket_probability_l570_570146

theorem winning_ticket_probability 
(select_numbers : Finset ℕ)
(h1 : ∀ x ∈ select_numbers, 1 ≤ x ∧ x ≤ 60)
(h2 : select_numbers.card = 6)
(h3 : ∃ k : ℕ, (select_numbers.prod id) = 10^k)
(h4 : select_numbers.sum id % 3 = 0) :
  (Real.infty_ennreal 1 / 4) ∈ { p : Real | 0 ≤ p ∧ p ≤ 1 } :=
sorry

end winning_ticket_probability_l570_570146


namespace sum_areas_of_selected_quadrilaterals_eq_one_over_k_area_l570_570068

theorem sum_areas_of_selected_quadrilaterals_eq_one_over_k_area 
  (ABCD : Type) [quadrilateral ABCD] (k : ℕ) (h1 : k > 0)
  (S_ABCD : ℝ) (h_convex : convex_quadrilateral ABCD)
  (has_equal_segments : all_sides_divided_into_equal_segments ABCD k)
  (first_series_connected : division_points_connected AB CD k)
  (second_series_connected : division_points_connected BC DA k)
  (num_quadrilaterals_formed : total_quadrilaterals_formed ABCD k = k^2)
  (selected_quadrilaterals_separated : selected_quadrilaterals_separated_by_lines ABCD k)
  : ∑ selected_areas = (1 / k : ℝ) * S_ABCD :=
sorry

end sum_areas_of_selected_quadrilaterals_eq_one_over_k_area_l570_570068


namespace weight_of_four_cakes_l570_570262

variable (C B : ℕ)  -- We declare C and B as natural numbers representing the weights in grams.

def cake_bread_weight_conditions (C B : ℕ) : Prop :=
  (3 * C + 5 * B = 1100) ∧ (C = B + 100)

theorem weight_of_four_cakes (C B : ℕ) 
  (h : cake_bread_weight_conditions C B) : 
  4 * C = 800 := 
by 
  {sorry}

end weight_of_four_cakes_l570_570262


namespace domain_of_sqrt_log_l570_570238

noncomputable def domain_of_function : Set ℝ := { x : ℝ | (2 * x - 1 > 0) ∧ (log (1 / 3) (2 * x - 1) >= 0) }

theorem domain_of_sqrt_log :
  domain_of_function = Set.Ioo (1 / 2) 1 ∪ {1} :=
by
  sorry

end domain_of_sqrt_log_l570_570238


namespace bob_net_income_change_l570_570720

theorem bob_net_income_change
  (raise_per_hour : ℚ := 0.50)
  (hours_per_week : ℕ := 40)
  (housing_benefit_reduction_per_month : ℚ := 60)
  (federal_tax_rate : ℚ := 0.15)
  (state_tax_rate : ℚ := 0.05)
  (social_security_rate : ℚ := 0.062)
  (medicare_rate : ℚ := 0.0145)
  (k401_contribution_rate : ℚ := 0.03)
  : 
  let additional_income_per_week := raise_per_hour * hours_per_week,
      total_deductions_percentage := federal_tax_rate + state_tax_rate + social_security_rate + medicare_rate + k401_contribution_rate,
      total_deductions := additional_income_per_week * total_deductions_percentage,
      net_increase_from_raise := additional_income_per_week - total_deductions,
      housing_benefit_reduction_per_week := housing_benefit_reduction_per_month / 4,
      net_increase_in_weekly_income := net_increase_from_raise - housing_benefit_reduction_per_week
  in net_increase_in_weekly_income = -1.13 := 
by
  -- Calculation steps here (omitted for brevity)
  sorry

end bob_net_income_change_l570_570720


namespace simplify_and_evaluate_expression_l570_570585

theorem simplify_and_evaluate_expression (a : ℤ) (ha : a = -2) : 
  (1 + 1 / (a - 1)) / ((2 * a) / (a ^ 2 - 1)) = -1 / 2 := by
  sorry

end simplify_and_evaluate_expression_l570_570585


namespace integer_root_is_zero_l570_570251

-- Define the conditions
variables {a b : ℚ} -- Rational numbers a and b

-- Polynomial equation
def polynomial (x : ℝ) := x^3 + a * x + b

-- Given root
axiom root1 : polynomial (3 - real.sqrt 5) = 0

-- Prove that the integer root is 0
theorem integer_root_is_zero (r : ℤ) (h : polynomial r = 0) : r = 0 :=
sorry

end integer_root_is_zero_l570_570251


namespace required_empty_boxes_2006_l570_570267

noncomputable def boxes_needed (n: ℕ): ℕ :=
  let rec k (m: ℕ) (i: ℕ): ℕ := if m < 2^i then i else k m (i+1)
  in k n 0

theorem required_empty_boxes_2006: boxes_needed 2006 = 11 := by
  unfold boxes_needed
  have h: 2^10 ≤ 2006 := by norm_num
  have h2: 2006 < 2^11 := by norm_num
  have k_eq_11: ∀ (i: ℕ), (2006 < 2^i) → (2006 ≥ 2^(i-1)) → i = 11 := by
    intros i hi1 hi2
    match i with
    | 11 => norm_num
    | _ => exact false.elim (Nat.not_lt_zero 10 h hi2)
  exact k_eq_11 _ h2 h

end required_empty_boxes_2006_l570_570267


namespace fountain_length_l570_570888

theorem fountain_length :
  let work_rate_20_men := 56 / 7 in
  let work_rate_per_man := work_rate_20_men / 20 in
  let work_rate_35_men := work_rate_per_man * 35 in
  let new_wall_time := 3 * 0.5 in
  let fountain_days := 9 - new_wall_time in
  let total_fountain_length := work_rate_35_men * fountain_days in
  total_fountain_length = 105 := by
  sorry

end fountain_length_l570_570888


namespace division_exact_iff_n_eq_two_l570_570290

theorem division_exact_iff_n_eq_two (n : ℕ) (hn : 0 < n) :
  (2n + 1) ∣ (n^4 + n^2) ↔ n = 2 := by
sorry

end division_exact_iff_n_eq_two_l570_570290


namespace solve_for_x_l570_570993

theorem solve_for_x (x : ℝ) : 2^(x-4) = 8^(x+2) → x = -5 := by
  sorry

end solve_for_x_l570_570993


namespace log_expansion_l570_570432

theorem log_expansion (a : ℝ) (h : a = Real.log 4 / Real.log 5) : Real.log 64 / Real.log 5 - 2 * (Real.log 20 / Real.log 5) = a - 2 :=
by
  sorry

end log_expansion_l570_570432


namespace volume_tetrahedron_proof_l570_570145

noncomputable def volume_of_tetrahedron (A1 D1 M N : ℝ×ℝ×ℝ) : ℝ :=
  -- Assuming coordinates are given or calculated according to the problem setup
  -- The actual volume calculation function would be implemented here
  sorry

theorem volume_tetrahedron_proof (A1 D1 M N : ℝ×ℝ×ℝ) (h1: A1 = (0, 0, 2)) (h2: D1 = (√3, 0, 0))
 (h3: M = (√3 / 2, 1, 0)) (h4: N = (1, 1, 1)) : volume_of_tetrahedron A1 D1 M N = (√3 / 2) :=
begin
  -- Coordinates are roughly assumed based on the prism's properties for Lean simplicity.
  -- Further detailed geometric arguments would go here.
  sorry
end

end volume_tetrahedron_proof_l570_570145


namespace common_ratio_of_geometric_sequence_l570_570914

variable (a₁ q : ℝ)

def geometric_sequence (n : ℕ) := a₁ * q^n

theorem common_ratio_of_geometric_sequence
  (h_sum : geometric_sequence a₁ q 0 + geometric_sequence a₁ q 1 + geometric_sequence a₁ q 2 = 3 * a₁) :
  q = 1 ∨ q = -2 :=
sorry

end common_ratio_of_geometric_sequence_l570_570914


namespace sum_angles_eq_990_degrees_l570_570614

noncomputable def alpha_sum : ℝ :=
  (54 + 126 + 198 + 270 + 342 : ℝ)

theorem sum_angles_eq_990_degrees :
  ∑ k in (finset.range 5), real.angle.to_degrees (real.angle.of_real ((270 + 360 * k) / 5)) = 990 :=
by {
  sorry
}

end sum_angles_eq_990_degrees_l570_570614


namespace number_of_subsets_with_sum_18_including_6_l570_570623

open Finset

def setA : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

theorem number_of_subsets_with_sum_18_including_6 :
  (setA.subsets (λ s, s.card = 3 ∧ 6 ∈ s ∧ s.sum = 18)).card = 4 := by
  sorry

end number_of_subsets_with_sum_18_including_6_l570_570623


namespace find_f_l570_570184

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := 2 * (λ x : ℝ, x) + 1

theorem find_f :
  (∀ x y : ℝ, f(f x + y) = g x + f (x + y) + x * y - x ^ 2 + 1) →
  (g = λ x, 2 * x + 1) →
  (f 0 = 1) →
  (f = (λ x, x + 2)) :=
by
  sorry

end find_f_l570_570184


namespace toothpicks_for_10_squares_l570_570636

theorem toothpicks_for_10_squares : (4 + 3 * (10 - 1)) = 31 :=
by 
  sorry

end toothpicks_for_10_squares_l570_570636


namespace g100_eq_0_has_one_solution_l570_570546

noncomputable def g0 (x : ℝ) : ℝ :=
x + abs (x - 150) - abs (x + 50)

noncomputable def g : ℕ → ℝ → ℝ
| 0, x := g0 x
| (n + 1), x := abs (g n x) - 2

theorem g100_eq_0_has_one_solution : ∃! x : ℝ, g 100 x = 0 :=
by
  sorry

end g100_eq_0_has_one_solution_l570_570546


namespace find_n_and_p_l570_570810

theorem find_n_and_p (n p : ℝ) (X : Type) [Distribution Binomial X] 
  (h1 : E X = 8) (h2 : D X = 1.6) : n = 10 ∧ p = 0.8 := by
  sorry

end find_n_and_p_l570_570810


namespace eccentricity_of_ellipse_l570_570414

variable (a b c d1 d2 : ℝ)
variable (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
variable (h4 : 2 * c = (d1 + d2) / 2)
variable (h5 : d1 + d2 = 2 * a)

theorem eccentricity_of_ellipse : (c / a) = 1 / 2 :=
by
  sorry

end eccentricity_of_ellipse_l570_570414


namespace parallel_mn_o1o2_l570_570949

open EuclideanGeometry

variables {A B C D P S T M N O1 O2 : Point}
variables (ω1 ω2 : Circle)

/-- Given a quadrilateral ABCD with AC = BD, diagonals AC and BD meet at P.
    Circle ω1 with center O1 is the circumcircle of triangle ABP.
    Circle ω2 with center O2 is the circumcircle of triangle CDP.
    Segment BC meets ω1 and ω2 again at points S and T respectively (other than B and C).
    M and N are the midpoints of minor arcs SP (not including B) and TP (not including C).
    Proof that MN is parallel to O1O2. -/
theorem parallel_mn_o1o2
  (hEq : distance A C = distance B D)
  (hP_AC : lies_on P (line_through A C))
  (hP_BD : lies_on P (line_through B D))
  (hCirc1 : circumcircle_of_triangle A B P = ω1)
  (hO1 : center_of ω1 = O1)
  (hCirc2 : circumcircle_of_triangle C D P = ω2)
  (hO2 : center_of ω2 = O2)
  (hS : lies_on S ω1 ∧ lies_on S (line_through B C) ∧ S ≠ B)
  (hT : lies_on T ω2 ∧ lies_on T (line_through B C) ∧ T ≠ C)
  (hM : midpoint_arc_sp M S P ω1)
  (hN : midpoint_arc_tp N T P ω2) :
  parallel (line_through M N) (line_through O1 O2) :=
sorry

end parallel_mn_o1o2_l570_570949


namespace a_value_l570_570189

def d (n : ℕ) : ℕ := (finset.range n.succ).filter (λ d, n % d = 0).card

def F (m : ℕ) (x : ℝ) : ℝ :=
  ∑ n in finset.range (105^m).succ, (d n : ℝ) / (n : ℝ)^x

def a (m n : ℕ) : ℝ := 
  ∑ k in (finset.range n.succ), (d k : ℝ) * (d (n / k) : ℝ)

theorem a_value (m : ℕ) (hpos : m > 0) :
  a m (105^m) = (nat.choose (m+3) 3)^3 :=
sorry

end a_value_l570_570189


namespace maximum_chess_players_l570_570266

theorem maximum_chess_players :
  ∀ (total : ℕ) (T C : ℕ),
    (total = 55) →
    (total = T + C) →
    (∀ c1 c2 c3 c4 k : ℕ,
      c1 ≠ c2 → c1 ≠ c3 → c1 ≠ c4 → c2 ≠ c3 → c2 ≠ c4 → c3 ≠ c4 →
      (c1, c2, c3, c4 : {x // x ≤ T}) → (∀ c : {x // x ≤ T}, k = c.1)) →
    (C ≤ 42) :=
by
  intros total T C htotal hsum hno4same
  sorry

end maximum_chess_players_l570_570266


namespace Sandy_tokens_difference_l570_570581

theorem Sandy_tokens_difference :
  let total_tokens : ℕ := 1000000
  let siblings : ℕ := 4
  let Sandy_tokens : ℕ := total_tokens / 2
  let sibling_tokens : ℕ := Sandy_tokens / siblings
  Sandy_tokens - sibling_tokens = 375000 :=
by
  sorry

end Sandy_tokens_difference_l570_570581


namespace min_value_of_expression_l570_570811

/-- 
Given α and β are the two real roots of the quadratic equation x^2 - 2a * x + a + 6 = 0,
prove that the minimum value of (α - 1)^2 + (β - 1)^2 is 8.
-/
theorem min_value_of_expression (a α β : ℝ) (h1 : α ^ 2 - 2 * a * α + a + 6 = 0) (h2 : β ^ 2 - 2 * a * β + a + 6 = 0) :
  (α - 1)^2 + (β - 1)^2 ≥ 8 := 
sorry

end min_value_of_expression_l570_570811


namespace farmer_seeds_per_ear_l570_570689

/-- Given conditions:
  - Each ear of corn sells for $0.1.
  - It costs $0.5 for a bag with 100 seeds.
  - The farmer makes $40 in profit.
  - The farmer sold 500 ears of corn.
  Prove the number of seeds needed to plant to get one ear of corn (S) is 4.
-/
def seeds_per_ear 
  (price_per_ear : ℚ)
  (bag_cost : ℚ)
  (seeds_per_bag : ℕ)
  (profit : ℚ)
  (ears_sold : ℕ)
  (S : ℕ)
  : Prop :=
  let revenue := ears_sold * price_per_ear in
  let cost_of_seeds := revenue - profit in
  let bags_bought := cost_of_seeds / bag_cost in
  let total_seeds := bags_bought * seeds_per_bag in
  S = total_seeds / ears_sold

theorem farmer_seeds_per_ear : seeds_per_ear 0.1 0.5 100 40 500 4 := by
  sorry

end farmer_seeds_per_ear_l570_570689


namespace cherries_on_June_5_l570_570360

theorem cherries_on_June_5 : 
  ∃ c : ℕ, (c + (c + 8) + (c + 16) + (c + 24) + (c + 32) = 130) ∧ (c + 32 = 42) :=
by
  sorry

end cherries_on_June_5_l570_570360


namespace number_of_words_with_at_least_one_consonant_l570_570488

def total_5_letter_words : ℕ := 6 ^ 5

def total_5_letter_vowel_words : ℕ := 2 ^ 5

def total_5_letter_words_with_consonant : ℕ := total_5_letter_words - total_5_letter_vowel_words

theorem number_of_words_with_at_least_one_consonant :
  total_5_letter_words_with_consonant = 7744 :=
  by
    -- We assert the calculation follows correctly:
    -- total_5_letter_words == 6^5 = 7776
    -- total_5_letter_vowel_words == 2^5 = 32
    -- 7776 - 32 == 7744
    sorry

end number_of_words_with_at_least_one_consonant_l570_570488


namespace man_out_of_well_l570_570692

theorem man_out_of_well (depth climb slip : ℕ) (h_depth : depth = 30) (h_climb : climb = 4) (h_slip : slip = 3) : 
  let effective_climb := climb - slip in
  effective_climb > 0 → 
  (depth - 1) / effective_climb + 1 = 30 := 
by
  sorry

end man_out_of_well_l570_570692


namespace sin_value_of_2alpha_add_pi_div_12_l570_570837

theorem sin_value_of_2alpha_add_pi_div_12 (α : ℝ) (h0 : 0 < α) (h1 : α < π / 2) (h2 : cos (α + π / 6) = 4 / 5) :
  sin (2 * α + π / 12) = 17 * Real.sqrt 2 / 50 :=
sorry

end sin_value_of_2alpha_add_pi_div_12_l570_570837


namespace range_of_f_l570_570256

def f (x : ℤ) : ℤ := (x - 1)^2 - 1

theorem range_of_f :
  Set.image f {-1, 0, 1, 2, 3} = {-1, 0, 3} :=
by
  sorry

end range_of_f_l570_570256


namespace avg_k_of_positive_integer_roots_l570_570849

theorem avg_k_of_positive_integer_roots : 
  (∑ (k : ℕ) in ({25, 14, 11, 10} : Finset ℕ), k) / 4 = 15 := 
by sorry

end avg_k_of_positive_integer_roots_l570_570849


namespace find_lambda_l570_570437

variables {V : Type*} [inner_product_space ℝ V] 
variables (a b : V)
variables (λ : ℝ)

-- Given conditions
def magnitude_a (a : V) : Prop := ‖a‖ = 3
def magnitude_b (b : V) : Prop := ‖b‖ = 5
def perpendicular_condition (a b : V) (λ : ℝ) : Prop := ⟪a + λ • b, a - λ • b⟫ = 0

-- Theorem statement
theorem find_lambda (a b : V) (λ : ℝ)
  (ha : magnitude_a a)
  (hb : magnitude_b b)
  (perp_cond : perpendicular_condition a b λ) :
  λ = 3 / 5 ∨ λ = -3 / 5 :=
sorry

end find_lambda_l570_570437


namespace find_stream_speed_l570_570304

variable (D : ℝ) (v : ℝ)

theorem find_stream_speed 
  (h1 : ∀D v, D / (63 - v) = 2 * (D / (63 + v)))
  (h2 : v = 21) :
  true := 
  by
  sorry

end find_stream_speed_l570_570304


namespace abs_sum_div_diff_of_nonzero_reals_l570_570887

-- We define the problem in Lean 4.
theorem abs_sum_div_diff_of_nonzero_reals (x y : ℝ) (hx0 : x ≠ 0) (hy0 : y ≠ 0) 
(h : x^2 + y^2 = 18 * x * y) : abs ((x + y) / (x - y)) = sqrt 5 / 2 :=
by
  sorry -- Proof is not required according to the instructions.

end abs_sum_div_diff_of_nonzero_reals_l570_570887


namespace average_possible_k_l570_570850

theorem average_possible_k (k : ℕ) (r1 r2 : ℕ) (h : r1 * r2 = 24) (h_pos : r1 > 0 ∧ r2 > 0) (h_eq_k : r1 + r2 = k) : 
  (25 + 14 + 11 + 10) / 4 = 15 :=
by 
  sorry

end average_possible_k_l570_570850


namespace regular_heptagon_diagonals_l570_570700

theorem regular_heptagon_diagonals (a b c : ℝ) 
  (h_regular : ∃ (p q : ℕ), p = 7 ∧ q = 2) 
  (h_heptagon : True) : 
  1 / a = 1 / b + 1 / c :=
begin
  sorry
end

end regular_heptagon_diagonals_l570_570700


namespace part_a_part_b_l570_570966

-- Define the cost variables for chocolates, popsicles, and lollipops
variables (C P L : ℕ)

-- Given conditions
axiom cost_relation1 : 3 * C = 2 * P
axiom cost_relation2 : 2 * L = 5 * C

-- Part (a): Prove that Mário can buy 5 popsicles with the money for 3 lollipops
theorem part_a : 
  (3 * L) / P = 5 :=
by sorry

-- Part (b): Prove that Mário can buy 11 chocolates with the money for 3 chocolates, 2 popsicles, and 2 lollipops combined
theorem part_b : 
  (3 * C + 2 * P + 2 * L) / C = 11 :=
by sorry

end part_a_part_b_l570_570966


namespace bulls_win_nba_finals_l570_570596

open ProbabilityTheory

def bulls_win_probability : ℝ := 
  let p_bulls := (2:ℝ) / 3 in
  let p_lakers := (1:ℝ) / 3 in
  ∑ k in Finset.range 6, 
    (Nat.choose (5 + k) k * (p_bulls^6) * (p_lakers^k))

theorem bulls_win_nba_finals : (bulls_win_probability * 100).round = 86 := 
  by
  sorry

end bulls_win_nba_finals_l570_570596


namespace find_a_and_b_and_max_value_on_interval_l570_570474

variables {a b : ℝ} (f : ℝ → ℝ)

def f (x : ℝ) : ℝ := a * log x - b * x^2

theorem find_a_and_b_and_max_value_on_interval :
  (∀ x > 0, deriv f x = (a / x) - 2 * b * x) →
  (f 1 = -(1/2) ∧ deriv f 1 = 0) →
  a = 1 ∧ b = 1/2 ∧ (∀ x ∈ Icc (1 / exp 1) (exp 1), f x ≤ f 1) ∧ f 1 = -(1/2) :=
by sorry

end find_a_and_b_and_max_value_on_interval_l570_570474


namespace find_slope_of_l3_l570_570960

def line_l1 (x y : ℝ) : Prop := 4 * x - 3 * y = 2
def line_l2 (y : ℝ) : Prop := y = 2
def point_A : (ℝ × ℝ) := (0, -2/3)
def point_B : (ℝ × ℝ) := (2, 2)
def area_triangle_ABC (BC : ℝ) : Prop := (1 / 2) * BC * (2 - (-2/3)) = 6

theorem find_slope_of_l3 (x y BC : ℝ) (hC : (x, y) ∈ ({C : ℝ × ℝ | line_l2 C.2} ∩ {C : ℝ × ℝ | C.1 = x ∧ C.2 = y}))
  (h_area : area_triangle_ABC BC) : (y - (-2/3)) / (x - 0) = 16/39 :=
sorry

end find_slope_of_l3_l570_570960


namespace count_lattice_points_on_segment_l570_570323

def gcd(a : ℕ, b : ℕ) : ℕ := Nat.gcd a b

theorem count_lattice_points_on_segment :
  ∀ (x1 y1 x2 y2 : ℤ), 
    x1 = 7 ∧ y1 = 23 ∧ x2 = 67 ∧ y2 = 391 →
    (gcd (y2 - y1).natAbs (x2 - x1).natAbs + 1 = 5) :=
by
  intros x1 y1 x2 y2 h
  sorry

end count_lattice_points_on_segment_l570_570323


namespace fixed_rate_calculation_l570_570691

theorem fixed_rate_calculation (f n : ℕ) (h1 : f + 4 * n = 220) (h2 : f + 7 * n = 370) : f = 20 :=
by
  sorry

end fixed_rate_calculation_l570_570691


namespace percentage_of_allowance_spent_l570_570562

noncomputable def amount_spent : ℝ := 14
noncomputable def amount_left : ℝ := 26
noncomputable def total_allowance : ℝ := amount_spent + amount_left

theorem percentage_of_allowance_spent :
  ((amount_spent / total_allowance) * 100) = 35 := 
by 
  sorry

end percentage_of_allowance_spent_l570_570562


namespace find_prob_B_l570_570757

/-- Define mutual independence of events A, B, and C. --/
def mutually_independent (A B C : Prop → Prop) : Prop :=
  (∀ (x : Prop), A x ∧ B x → C x) ∧
  (∀ (y : Prop), B y ∧ A y → C y) ∧
  (∀ (z : Prop), A z ∧ C z → B z)

/-- Given probabilities: --/
variables (P : (Prop → Prop) → ℝ)
variables (A B C : Prop → Prop)

axiom prob_A_div_B : P (λ x, A x ∧ B x) = 1/6
axiom prob_compl_B_div_C : P (λ x, (¬ B x) ∧ C x) = 1/8
axiom prob_A_div_B_compl_C : P (λ x, A x ∧ B x ∧ (¬ C x)) = 1/8
axiom independence : mutually_independent A B C

/-- Theorem to prove that P(B) = 1/2 --/
theorem find_prob_B : P B = 1/2 :=
sorry

end find_prob_B_l570_570757


namespace triangle_inequality_simplification_l570_570815

theorem triangle_inequality_simplification
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : a + c > b)
  (h3 : b + c > a) :
  (a + b - c > 0) ∧ (a - b + c > 0) ∧ (a - b - c < 0) ∧ (|a + b - c| - |a - b + c| + |a - b - c| = -a + 3b - c) :=
by {
  sorry
}

end triangle_inequality_simplification_l570_570815


namespace avg_k_of_positive_integer_roots_l570_570847

theorem avg_k_of_positive_integer_roots : 
  (∑ (k : ℕ) in ({25, 14, 11, 10} : Finset ℕ), k) / 4 = 15 := 
by sorry

end avg_k_of_positive_integer_roots_l570_570847


namespace union_of_A_and_B_l570_570956

-- Define the sets A and B as given in the problem
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 4}

-- State the theorem to prove that A ∪ B = {0, 1, 2, 4}
theorem union_of_A_and_B : A ∪ B = {0, 1, 2, 4} := by
  sorry

end union_of_A_and_B_l570_570956


namespace length_of_platform_l570_570679

theorem length_of_platform
  (length_of_train : ℝ)
  (time_to_cross_tree : ℝ)
  (time_to_cross_platform : ℝ)
  (h_train_length : length_of_train = 1200)
  (h_tree_time : time_to_cross_tree = 80)
  (h_platform_time : time_to_cross_platform = 146.67) :
  let speed_of_train := length_of_train / time_to_cross_tree in
  let total_distance := speed_of_train * time_to_cross_platform in
  let length_of_platform := total_distance - length_of_train in
  length_of_platform = 1000.05 :=
by
  sorry

end length_of_platform_l570_570679


namespace element_in_set_l570_570200

theorem element_in_set (A : Set ℕ) (h : A = {1, 2}) : 1 ∈ A := 
by 
  rw[h]
  simp

end element_in_set_l570_570200


namespace find_smallest_x_satisfying_condition_l570_570795

theorem find_smallest_x_satisfying_condition :
  ∃ x : ℝ, 0 < x ∧ (⌊x^2⌋ - x * ⌊x⌋ = 10) ∧ x = 131 / 11 :=
by
  sorry

end find_smallest_x_satisfying_condition_l570_570795


namespace isosceles_triangle_surface_area_isosceles_triangle_volume_l570_570911

variables (b α : ℝ)

theorem isosceles_triangle_surface_area
  (hb : 0 < b)
  (hα : 0 < α ∧ α < π) :
  let F := 12 * π * b^2 * sin (α / 2) * (1 + sin (α / 2)) in
  F = 12 * π * b^2 * sin (α / 2) * (1 + sin (α / 2)) :=
by sorry

theorem isosceles_triangle_volume
  (hb : 0 < b)
  (hα : 0 < α ∧ α < π) :
  let K := 6 * π * b^3 * sin (α / 2)^2 * cos (α / 2) in
  K = 6 * π * b^3 * sin (α / 2)^2 * cos (α / 2) :=
by sorry

end isosceles_triangle_surface_area_isosceles_triangle_volume_l570_570911


namespace range_of_b_l570_570853

theorem range_of_b (a b : ℝ) : 
  (∀ x : ℝ, -3 < x ∧ x < 1 → (1 - a) * x^2 - 4 * x + 6 > 0) ∧
  (∀ x : ℝ, 3 * x^2 + b * x + 3 ≥ 0) →
  (-6 ≤ b ∧ b ≤ 6) :=
by
  sorry

end range_of_b_l570_570853


namespace smallest_positive_real_is_131_div_11_l570_570768

noncomputable def smallest_positive_real_satisfying_condition :=
  ∀ (x : ℝ), (∀ y > 0, (y * y ⌊y⌋ - y ⌊y⌋ = 10) → (x ≤ y)) → 
  (⌊x*x⌋ - (x * ⌊x⌋) = 10) → 
  x = 131/11

theorem smallest_positive_real_is_131_div_11 :
  smallest_positive_real_satisfying_condition := sorry

end smallest_positive_real_is_131_div_11_l570_570768


namespace solve_for_x_l570_570945

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 3 * x

-- State the theorem
theorem solve_for_x (x : ℝ) : f(f(x)) = f(x) ↔ x = 0 ∨ x = 3 ∨ x = -1 ∨ x = 4 := by
  sorry

end solve_for_x_l570_570945


namespace overall_loss_percentage_is_correct_l570_570009
open scoped Real

def radio := (cost_price : Real) (selling_price : Real) (units_sold : Nat)

def radio_A := radio 1500 1290 5
def radio_B := radio 2200 1900 8
def radio_C := radio 3000 2800 10

noncomputable def total_cost_price (radios : List radio) : Real :=
  radios.foldl (λ acc r => acc + (r.cost_price * r.units_sold)) 0

noncomputable def total_selling_price (radios : List radio) : Real :=
  radios.foldl (λ acc r => acc + (r.selling_price * r.units_sold)) 0

noncomputable def loss_percentage (total_cost : Real) (total_selling : Real) : Real :=
  ((total_cost - total_selling) / total_cost) * 100

theorem overall_loss_percentage_is_correct :
  let radios := [radio_A, radio_B, radio_C]
  let total_cost := total_cost_price radios
  let total_selling := total_selling_price radios
  let loss_percent := loss_percentage total_cost total_selling
  loss_percent ≈ 9.89 :=
by
  sorry

end overall_loss_percentage_is_correct_l570_570009


namespace sum_of_rational_roots_of_h_l570_570409

theorem sum_of_rational_roots_of_h :
  let h(x : ℚ) := x^3 - 6*x^2 + 11*x - 6
  h(1) = 0 ∧ h(2) = 0 ∧ h(3) = 0 →
  (1 + 2 + 3 = 6) := sorry

end sum_of_rational_roots_of_h_l570_570409


namespace hyperbola_eccentricity_l570_570240

theorem hyperbola_eccentricity
  (M F1 F2 : euclidean_plane.point)
  (angle_F1MF2 : ∠ F1 M F2 = 2 * π / 3) 
  -- Assume the hyperbola has its center at the origin
  (O : euclidean_plane.point) 
  (OM : euclidean_plane.distance O M = b)
  (OF2 : euclidean_plane.distance O F2 = c)
  : let e := c / (sqrt (c^2 - b^2)) in 
    e = sqrt 6 / 2 :=
by
  sorry

end hyperbola_eccentricity_l570_570240


namespace total_three_digit_convex_numbers_eq_240_l570_570284

noncomputable def is_convex (a b c : Nat) : Prop :=
  a < b ∧ b > c

noncomputable def is_three_digit (a b c : Nat) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9

noncomputable def is_three_digit_convex_number (abc : Nat × Nat × Nat) : Prop :=
  let (a, b, c) := abc
  is_three_digit a b c ∧ is_convex a b c

noncomputable def count_three_digit_convex_numbers : Nat :=
  (Finset.range (10 * 10 * 10)).filter (fun n =>
    let abc := (n / 100, (n / 10) % 10, n % 10)
    is_three_digit_convex_number abc
  ).card

theorem total_three_digit_convex_numbers_eq_240 :
  count_three_digit_convex_numbers = 240 :=
sorry

end total_three_digit_convex_numbers_eq_240_l570_570284


namespace trapezoid_area_l570_570908

theorem trapezoid_area (PR PQ QR : ℝ) (h_iso : PR = PQ) (h_area_PQR : 1/2 * PR * QR = 100)
  (h_small_triangles : ∀ t, t ∈ (finset.range 20).image (λ i, 1) → t = 1) 
  (h_PQS_sim_PQR : ∀ (s1 : ℝ × ℝ) (s2 : ℝ × ℝ), s1 ≠ s2 → s1.1 = s2.1 ∧ s1.2 = s2.2 → 
                     ∃ (r : ℝ), 0 < r ∧ r ≠ 1 ∧ s1.1 = r * s2.1 ∧ s1.2 = r * s2.2) :
  100 - 6 = 94 :=
by
  sorry

end trapezoid_area_l570_570908


namespace part1_part2_l570_570201

section
variables (x a m n : ℝ)
-- Define the function f
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x - 3)

-- a) Prove the solution of the inequality f(x) >= 4 + |x-3| - |x-1| given a=3.
theorem part1 (h_a : a = 3) :
  {x | f x a ≥ 4 + abs (x - 3) - abs (x - 1)} = {x | x ≤ 0} ∪ {x | x ≥ 4} :=
sorry

-- b) Prove that m + 2n >= 2 given f(x) <= 1 + |x-3| with solution set [1, 3] and 1/m + 1/(2n) = a
theorem part2 (h_sol : ∀ x, 1 ≤ x ∧ x ≤ 3 → f x a ≤ 1 + abs (x - 3)) 
  (h_a : 1 / m + 1 / (2 * n) = 2) (h_m_pos : m > 0) (h_n_pos : n > 0) :
  m + 2 * n ≥ 2 :=
sorry
end

end part1_part2_l570_570201


namespace angle_F_measure_l570_570527

-- Given conditions
def D := 74
def sum_of_angles (x E D : ℝ) := x + E + D = 180
def E_formula (x : ℝ) := 2 * x - 10

-- Proof problem statement in Lean 4
theorem angle_F_measure :
  ∃ x : ℝ, x = (116 / 3) ∧
    sum_of_angles x (E_formula x) D :=
sorry

end angle_F_measure_l570_570527


namespace ladder_rung_length_l570_570923

noncomputable def ladder_problem : Prop :=
  let total_height_ft := 50
  let spacing_in := 6
  let wood_ft := 150
  let feet_to_inches(ft : ℕ) : ℕ := ft * 12
  let total_height_in := feet_to_inches total_height_ft
  let wood_in := feet_to_inches wood_ft
  let number_of_rungs := total_height_in / spacing_in
  let length_of_each_rung := wood_in / number_of_rungs
  length_of_each_rung = 18

theorem ladder_rung_length : ladder_problem := sorry

end ladder_rung_length_l570_570923


namespace trip_time_l570_570737

theorem trip_time (distance half_dist speed1 speed2 : ℝ) 
  (h_distance : distance = 360) 
  (h_half_distance : half_dist = distance / 2) 
  (h_speed1 : speed1 = 50) 
  (h_speed2 : speed2 = 45) : 
  (half_dist / speed1 + half_dist / speed2) = 7.6 := 
by
  -- Simplify the expressions based on provided conditions
  sorry

end trip_time_l570_570737


namespace odd_digits_in_base4_of_157_l570_570398

theorem odd_digits_in_base4_of_157 : 
  let b4 := 2 * 4^3 + 1 * 4^2 + 3 * 4^1 + 1 * 4^0 in
  b4 = 157 → (nat.digits 4 157).countp (λ d, d % 2 = 1) = 3 := 
by
  intros
  sorry

end odd_digits_in_base4_of_157_l570_570398


namespace only_f_C_is_both_decreasing_and_odd_l570_570292

def f_A (x : ℝ) : ℝ := (1 / 2) ^ x - 1
def f_B (x : ℝ) : ℝ := 1 / x
def f_C (x : ℝ) : ℝ := -x
def f_D (x : ℝ) : ℝ := -Real.exp (abs x)

theorem only_f_C_is_both_decreasing_and_odd :
  (∃ f : ℝ → ℝ, f = f_C ∧ ∀ x : ℝ, f (-x) = -f x ∧ ∀ x y : ℝ, x < y → f y < f x)
  ∧ (∀ f : ℝ → ℝ, (f = f_A ∨ f = f_B ∨ f = f_D) → ¬ (∀ x : ℝ, f (-x) = -f x ∧ ∀ x y : ℝ, x < y → f y < f x))
:= by
  sorry

end only_f_C_is_both_decreasing_and_odd_l570_570292


namespace kittens_price_l570_570696

theorem kittens_price (x : ℕ) 
  (h1 : 2 * x + 5 = 17) : x = 6 := by
  sorry

end kittens_price_l570_570696


namespace valid_paths_count_l570_570164

def num_paths_avoiding_dangerous_points : ℕ :=
  30

theorem valid_paths_count : 
  ∀ (startX startY endX endY : ℕ),
  (startX, startY) = (0, 0) → 
  (endX, endY) = (4, 3) → 
  num_paths_avoiding_dangerous_points = 30 := 
by 
  intros startX startY endX endY hs he;
  cases hs;
  cases he;
  refl

end valid_paths_count_l570_570164


namespace average_speed_equal_time_l570_570594

theorem average_speed_equal_time (swim_speed : ℝ) (run_speed : ℝ) (t : ℝ)
  (h1 : swim_speed = 1) (h2 : run_speed = 8) (h3 : t > 0) :
  (swim_speed + run_speed) / 2 = 4.5 :=
by
  simp [h1, h2]
  norm_num
  sorry

end average_speed_equal_time_l570_570594


namespace maximum_value_of_f_on_interval_l570_570607

noncomputable def f (x : ℝ) : ℝ := - 1 / 2 * x^2 + Real.log x + 1

theorem maximum_value_of_f_on_interval : 
  ∃ x ∈ Icc (1 / Real.exp 1) Real.exp 1, 
  ∀ y ∈ Icc (1 / Real.exp 1) Real.exp 1, f y ≤ f x ∧ f x = 1 / 2 :=
by
  sorry

end maximum_value_of_f_on_interval_l570_570607


namespace volume_ratio_cones_l570_570283

theorem volume_ratio_cones :
  let rC := 16.5
  let hC := 33
  let rD := 33
  let hD := 16.5
  let VC := (1 / 3) * Real.pi * rC^2 * hC
  let VD := (1 / 3) * Real.pi * rD^2 * hD
  (VC / VD) = (1 / 2) :=
by
  sorry

end volume_ratio_cones_l570_570283


namespace gigi_additional_batches_l570_570428

-- Define the initial amount of flour in cups
def initialFlour : Nat := 20

-- Define the amount of flour required per batch in cups
def flourPerBatch : Nat := 2

-- Define the number of batches already baked
def batchesBaked : Nat := 3

-- Define the remaining flour
def remainingFlour : Nat := initialFlour - (batchesBaked * flourPerBatch)

-- Define the additional batches Gigi can make with the remaining flour
def additionalBatches : Nat := remainingFlour / flourPerBatch

-- Prove that with the given conditions, the additional batches Gigi can make is 7
theorem gigi_additional_batches : additionalBatches = 7 := by
  -- Calculate the remaining cups of flour after baking
  have h1 : remainingFlour = 20 - (3 * 2) := by rfl

  -- Calculate the additional batches of cookies Gigi can make
  have h2 : additionalBatches = h1 / 2 := by rfl

  -- Solve for the additional batches
  show additionalBatches = 7 from
    calc
      additionalBatches = (initialFlour - (batchesBaked * flourPerBatch)) / flourPerBatch : by rfl
      ...               = (20 - 6) / 2                               : by rw h1
      ...               = 14 / 2                                     : by rfl
      ...               = 7                                          : by rfl

end gigi_additional_batches_l570_570428


namespace sin_A_is_4_over_5_B_is_pi_over_4_projection_of_BA_in_direction_BC_l570_570902

-- Given conditions
def m (A B : ℝ) := (Real.cos (A - B), Real.sin (A - B))
def n (B : ℝ) := (Real.cos B, -Real.sin B)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

noncomputable def A := sorry -- Placeholder for A
noncomputable def B := sorry -- Placeholder for B
noncomputable def a := 4 * Real.sqrt 2
noncomputable def b := 5

theorem sin_A_is_4_over_5 (A B : ℝ) (h1 : dot_product (m A B) (n B) = -3 / 5) : Real.sin A = 4 / 5 := sorry

theorem B_is_pi_over_4 (A B : ℝ) (h1 : Real.sin A = 4 / 5) (h2 : a = 4 * Real.sqrt 2) (h3 : b = 5)
: B = Real.pi / 4 := sorry

theorem projection_of_BA_in_direction_BC
  (A B c : ℝ)
  (h1 : a = 4 * Real.sqrt 2)
  (h2 : b = 5)
  (h3 : B = Real.pi / 4)
  : let projection := (1 : ℝ) * (Real.sqrt 2 / 2) in
    projection = Real.sqrt 2 / 2 := sorry

end sin_A_is_4_over_5_B_is_pi_over_4_projection_of_BA_in_direction_BC_l570_570902


namespace minimize_quadratic_l570_570288

theorem minimize_quadratic : ∃ x : ℝ, x = 6 ∧ ∀ y : ℝ, (y - 6)^2 ≥ (6 - 6)^2 := by
  sorry

end minimize_quadratic_l570_570288
