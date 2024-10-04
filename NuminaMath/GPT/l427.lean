import LinearAlgebra
import Mathlib
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Pointwise
import Mathlib.Algebra.Order.Floor
import Mathlib.Algebra.Pigeonhole
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Quadratics
import Mathlib.Algebra.Star.Basic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Integral
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.GraphTheory
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Fin.VecNotation
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Perm
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.LinearAlgebra.Matrix
import Mathlib.Logic.Basic
import Mathlib.NumberTheory.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Linarith

namespace proves_distance_eq_10_implies_z_zero_l427_427041

/-- Let A and B be points in 3D space. Given the coordinates of A and B, the following 
    theorem proves that z must be equal to 0 if the distance between A and B is 10. --/
theorem distance_eq_10_implies_z_zero (z : ℝ) :
    (sqrt ((-3 - 3)^2 + (4 - (-4))^2 + (z - 0)^2) = 10) → z = 0 :=
by
  intro h
  have h_eq : (-3 - 3 : ℝ)^2 + (4 - (-4) : ℝ)^2 + (z - 0 : ℝ)^2 = 10^2 := by
    rw [←sq_sqrt _ h]
    ring
    linarith
  linarith

end proves_distance_eq_10_implies_z_zero_l427_427041


namespace quadratic_equation_solutions_l427_427159

theorem quadratic_equation_solutions : ∀ x : ℝ, x^2 - 2 * x = 0 ↔ (x = 0 ∨ x = 2) := 
by sorry

end quadratic_equation_solutions_l427_427159


namespace range_dot_product_l427_427868

-- Definition of the problem parameters and conditions
def A := (1 : ℝ, 0 : ℝ)
def B := (0 : ℝ, 0 : ℝ)
def P (α : ℝ) := (real.cos α, real.sin α)

-- Definition of vectors BA and BP
def BA := (A.1 - B.1, A.2 - B.2)
def BP (α : ℝ) := ((P α).1 - B.1, (P α).2 - B.2)

-- Dot product of vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The main theorem to be proven
theorem range_dot_product (α : ℝ) (hα : 0 ≤ α ∧ α ≤ π/2) :
  1 ≤ dot_product (BP α) BA ∧ dot_product (BP α) BA ≤ 1 + real.sqrt 2 :=
sorry

end range_dot_product_l427_427868


namespace transformed_data_properties_l427_427396

-- Definitions based on problem conditions
variables {n : ℕ} (x y : Fin n → ℝ) (a b c d : ℝ)

-- Given conditions
def average (x : Fin n → ℝ) (a : ℝ) : Prop :=
  (∑ i, x i) / n = a

def variance (x : Fin n → ℝ) (b : ℝ) : Prop :=
  (∑ i, (x i - (∑ j, x j) / n) ^ 2) / n = b

def median (x : Fin n → ℝ) (c : ℝ) : Prop :=
  ∃ perm : Fin n → Fin n, 
  Permutation (Finset.univ.val.map x) (perm.val.map x) ∧ 
  ((n.even ∧ (x (perm ⟨n/2 - 1, sorry⟩) + x (perm ⟨n/2, sorry⟩)) / 2 = c) ∨ 
  (n.odd ∧ x (perm ⟨n / 2, sorry⟩) = c))

def range (x : Fin n → ℝ) (d : ℝ) : Prop :=
  (Finset.sup Finset.univ (λ i, x i)) - (Finset.inf Finset.univ (λ i, x i)) = d

-- New dataset transformation
def new_data (x : Fin n → ℝ) (i : Fin n) : ℝ :=
  2 * x i + 1

-- Proof objectives
theorem transformed_data_properties :
  average x a →
  variance x b →
  median x c →
  range x d →
  average (new_data x) (2 * a + 1) ∧
  variance (new_data x) (4 * b) ∧
  median (new_data x) (2 * c + 1) ∧
  range (new_data x) (2 * d) :=
by 
  intros h_avg h_var h_med h_range
  sorry

end transformed_data_properties_l427_427396


namespace total_mass_of_fruit_l427_427701

theorem total_mass_of_fruit :
  let num_apple_trees := 30 in
  let mass_per_apple_tree := 150 in
  let num_peach_trees := 45 in
  let mass_per_peach_tree := 65 in
  let total_mass_apples := num_apple_trees * mass_per_apple_tree in
  let total_mass_peaches := num_peach_trees * mass_per_peach_tree in
  total_mass_apples + total_mass_peaches = 7425 :=
by
  sorry

end total_mass_of_fruit_l427_427701


namespace determine_m_range_l427_427028

variable {R : Type} [OrderedCommGroup R]

-- Define the odd function f: ℝ → ℝ
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Define the increasing function f: ℝ → ℝ
def increasing_function (f : ℝ → ℝ) := ∀ x y, x < y → f x < f y

-- Define the main theorem
theorem determine_m_range (f : ℝ → ℝ) (odd_f : odd_function f) (inc_f : increasing_function f) :
    (∀ θ : ℝ, f (Real.cos (2 * θ) - 5) + f (2 * m + 4 * Real.sin θ) > 0) → m > 5 :=
by
  sorry

end determine_m_range_l427_427028


namespace sum_of_abc_is_33_l427_427167

theorem sum_of_abc_is_33 (a b c N : ℕ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c)
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hN1 : N = 5 * a + 3 * b + 5 * c)
    (hN2 : N = 4 * a + 5 * b + 4 * c) (hN_range : 131 < N ∧ N < 150) :
  a + b + c = 33 := 
sorry

end sum_of_abc_is_33_l427_427167


namespace largest_value_of_n_l427_427352

theorem largest_value_of_n (A B n : ℤ) (h1 : A * B = 72) (h2 : n = 6 * B + A) : n = 433 :=
sorry

end largest_value_of_n_l427_427352


namespace triangle_length_l427_427119

theorem triangle_length (LM LN : ℝ) (h1 : sin (atan (3/4)) = 3/5) (h2 : LM = 18) : LN = 30 :=
sorry

end triangle_length_l427_427119


namespace kenneth_money_left_l427_427895

theorem kenneth_money_left (I : ℕ) (C_b : ℕ) (N_b : ℕ) (C_w : ℕ) (N_w : ℕ) (L : ℕ) :
  I = 50 → C_b = 2 → N_b = 2 → C_w = 1 → N_w = 2 → L = I - (N_b * C_b + N_w * C_w) → L = 44 :=
by
  intros h₀ h₁ h₂ h₃ h₄ h₅
  sorry

end kenneth_money_left_l427_427895


namespace valid_bases_for_625_l427_427371

theorem valid_bases_for_625 (b : ℕ) : (b^3 ≤ 625 ∧ 625 < b^4) → ((625 % b) % 2 = 1) ↔ (b = 6 ∨ b = 7 ∨ b = 8) :=
by
  sorry

end valid_bases_for_625_l427_427371


namespace union_complements_eq_l427_427011

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

-- Define the universal set U
def universal_set : Set ℕ := {x | 0 ≤ x ∧ x < 6}

-- Define set A
def set_A : Set ℕ := {1, 3, 5}

-- Define set B where B is determined by solving the equation x^2 + 4 = 5x
def set_B : Set ℕ := {x | x^2 + 4 = 5 * x}

-- Define the complement of A in U
def complement_U_A : Set ℕ := universal_set \ set_A

-- Define the complement of B in U
def complement_U_B : Set ℕ := universal_set \ set_B

-- The theorem we need to prove
theorem union_complements_eq : 
  (complement_U_A universal_set set_A) ∪ (complement_U_B universal_set set_B) = {0, 2, 3, 4, 5} :=
by {
    sorry
}

end union_complements_eq_l427_427011


namespace product_of_differences_l427_427417

theorem product_of_differences (p q p' q' α β α' β' : ℝ)
  (h1 : α + β = -p) (h2 : α * β = q)
  (h3 : α' + β' = -p') (h4 : α' * β' = q') :
  ((α - α') * (α - β') * (β - α') * (β - β') = (q - q')^2 + (p - p') * (q' * p - p' * q)) :=
sorry

end product_of_differences_l427_427417


namespace fraction_of_amount_l427_427704

noncomputable def calculate_fraction (r : ℚ) (t : ℚ) : ℚ :=
  let P := 1 in  -- Arbitrary principal, won't affect the fraction result
  let SI := (P * r * t) / 100 in
  SI / (P + SI)

theorem fraction_of_amount (r t : ℚ) (hr : r = 2) (ht : t = 10) :
  calculate_fraction r t = 1/6 :=
by
  rw [calculate_fraction, hr, ht]
  -- calculation steps can be filled in proof
  sorry

end fraction_of_amount_l427_427704


namespace baker_usual_pastries_l427_427653

variable (P : ℕ)

theorem baker_usual_pastries
  (h1 : 2 * 14 + 4 * 25 - (2 * P + 4 * 10) = 48) : P = 20 :=
by
  sorry

end baker_usual_pastries_l427_427653


namespace compute_value_of_fractions_l427_427565

theorem compute_value_of_fractions (a b c : ℝ) 
  (h1 : (ac / (a + b)) + (ba / (b + c)) + (cb / (c + a)) = 0)
  (h2 : (bc / (a + b)) + (ca / (b + c)) + (ab / (c + a)) = 1) :
  (b / (a + b)) + (c / (b + c)) + (a / (c + a)) = 5 / 2 :=
sorry

end compute_value_of_fractions_l427_427565


namespace maximize_distance_difference_on_l_l427_427388

noncomputable def point := (ℝ × ℝ)

def A : point := (4, 1)

def B : point := (0, 4)

def l (x y : ℝ) := 3 * x - y - 1 = 0

def PA (P : point) := real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)

def PB (P : point) := real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)

def line_contains (l: ℝ × ℝ → Prop) (P: point) := l P.1 P.2

theorem maximize_distance_difference_on_l:
  ∃ P : point, line_contains l P ∧
  ∀ Q : point, line_contains l Q → (PA P - PB P) ≥ (PA Q - PB Q) → P = (2, 5) :=
begin
  sorry
end

end maximize_distance_difference_on_l_l427_427388


namespace sum_distinct_vars_eq_1716_l427_427507

open Real

theorem sum_distinct_vars_eq_1716 (p q r s : ℝ) (hpqrs_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : r + s = 12 * p)
  (h2 : r * s = -13 * q)
  (h3 : p + q = 12 * r)
  (h4 : p * q = -13 * s) :
  p + q + r + s = 1716 :=
sorry

end sum_distinct_vars_eq_1716_l427_427507


namespace sufficient_but_not_necessary_l427_427232

theorem sufficient_but_not_necessary (x : ℝ) (h : 1 < x ∧ x < 2) : x < 2 ∧ ∀ y, (y < 2 → y ≤ 1 ∨ y ≥ 2) :=
by
  sorry

end sufficient_but_not_necessary_l427_427232


namespace math_problem_l427_427380

-- Define 2D point
structure Point := 
  (x : ℝ) 
  (y : ℝ)

-- Define the main conditions 
def F : Point := ⟨0, 1⟩
def l (x : ℝ) : ℝ := -1
def line_60 (x : ℝ) : ℝ := sqrt 3 * x + 1

-- Definitions for the trajectory equation and the length of chord
def trajectory_eqn (M : Point) : Prop :=
  Real.sqrt (M.x^2 + (M.y - F.y)^2) = M.y + 1 -> M.x^2 = 4 * M.y

def chord_length (A B : Point) : ℝ :=
  sqrt ((B.x - A.x)^2 + (B.y - A.y)^2)

-- The theorem to prove 
theorem math_problem :
  (∀ M : Point, Real.sqrt (M.x^2 + (M.y - F.y)^2) = M.y + 1 → M.x^2 = 4 * M.y) ∧
  (∃ A B : Point, 
    A.y = line_60 A.x ∧
    B.y = line_60 B.x ∧
    A.x^2 = 4 * A.y ∧
    B.x^2 = 4 * B.y ∧
    chord_length A B = 16) :=
begin
  sorry
end

end math_problem_l427_427380


namespace probability_slope_condition_l427_427491

noncomputable def fixed_point : ℝ × ℝ := (3 / 4, 1 / 4)

def unit_square (P : ℝ × ℝ) : Prop := 
  0 ≤ P.1 ∧ P.1 ≤ 1 ∧ 0 ≤ P.2 ∧ P.2 ≤ 1

def slope_condition (P : ℝ × ℝ) : Prop :=
  let fp := fixed_point
  let slope : ℝ := (P.2 - fp.2) / (P.1 - fp.1)
  slope < 1 / 3

theorem probability_slope_condition : 
  let P_chosen := λ P, unit_square P ∧ slope_condition P
  ∃ m n : ℕ, Nat.coprime m n ∧ (∃ A : ℝ, A = ⟦∫ (P_chosen)⟧) ∧ A = 1 / 9 ∧ (m + n) = 10 := 
sorry

end probability_slope_condition_l427_427491


namespace other_five_say_equal_numbers_l427_427095

noncomputable def knights_and_liars_problem : Prop :=
  ∃ (K L : ℕ), K + L = 10 ∧
  ∀ (x : ℕ), (x < 5 → "There are more liars" = true) ∨ (x >= 5 → "There are equal numbers of knights and liars" = true)

theorem other_five_say_equal_numbers :
  knights_and_liars_problem :=
sorry

end other_five_say_equal_numbers_l427_427095


namespace part_one_part_two_l427_427288

-- Definitions for the conditions given in the problem for Part (1)
def O (A B C : Point) : Point := sorry  -- Definition of circumcenter O of ∆ABC
def I (A B C : Point) : Point := sorry  -- Definition of incenter I of ∆ABC
def R (A B C : Point) (O : Point) : ℝ := sorry  -- Definition of circumradius R

-- Define basic properties about the angles and points involved
axiom angle_B_eq_60 (A B C : Point) (O I : Point) (h: ∠B ≡ 60) : true
axiom angle_A_lt_angle_C (A B C : Point) (h: ∠A < ∠C) : true
axiom external_angle_bisector_intersects_O_at_E (A B C E : Point) (h: E ∈ circle O) : true

-- Statement for Part (1)
theorem part_one (A B C E O I : Point) (R : ℝ)
  (hB : ∠B = 60)
  (hA : ∠A < ∠C)
  (hext : external_angle_bisector_intersects_O_at_E A B C E)
  : dist I O = dist A E := sorry

-- Definitions for the conditions given in the problem for Part (2)
def IAO (A I O : Point) : ℝ := sorry  -- Definitions leading to distances IA, IO, IC
def IAO_distance (A I O : Point) : ℝ := sorry
def IIO_distance (A I O : Point) : ℝ := sorry
def IIC_distance (A I C : Point) : ℝ := sorry

-- Statement for Part (2)
theorem part_two (A B C E O I : Point) (R : ℝ)
  (hB : ∠B = 60)
  (hA : ∠A < ∠C)
  (hext : external_angle_bisector_intersects_O_at_E A B C E)
  (hIA : dist I A)
  (hIC : dist I C)
  (hIO : dist I O)
  : 2 * R < hIO + hIA + hIC ∧ hIO + hIA + hIC < (1 + sqrt 3) * R := sorry

end part_one_part_two_l427_427288


namespace sum_distinct_vars_eq_1716_l427_427509

open Real

theorem sum_distinct_vars_eq_1716 (p q r s : ℝ) (hpqrs_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : r + s = 12 * p)
  (h2 : r * s = -13 * q)
  (h3 : p + q = 12 * r)
  (h4 : p * q = -13 * s) :
  p + q + r + s = 1716 :=
sorry

end sum_distinct_vars_eq_1716_l427_427509


namespace composite_divisor_bound_l427_427936

theorem composite_divisor_bound (n : ℕ) (hn : ¬Prime n ∧ 1 < n) : 
  ∃ a : ℕ, 1 < a ∧ a ≤ Int.sqrt (n : ℤ) ∧ a ∣ n :=
sorry

end composite_divisor_bound_l427_427936


namespace x_inv_cubed_equals_one_ninth_l427_427437

theorem x_inv_cubed_equals_one_ninth
  (x : ℝ)
  (h : log 9 (log 5 (log 3 x)) = 0) : 
  x ^ (-1 / 3) = 1 / 9 :=
  by 
    sorry

end x_inv_cubed_equals_one_ninth_l427_427437


namespace evaluate_f_at_3_l427_427831

def f (x : ℤ) : ℤ := 5 * x^3 + 3 * x^2 + 7 * x - 2

theorem evaluate_f_at_3 : f 3 = 181 := by
  sorry

end evaluate_f_at_3_l427_427831


namespace opposite_of_neg_two_l427_427996

theorem opposite_of_neg_two : -(-2) = 2 := 
by 
  sorry

end opposite_of_neg_two_l427_427996


namespace closest_integer_to_x_minus_y_l427_427443

theorem closest_integer_to_x_minus_y (x y : ℝ) (hx : |x| ≠ 0) (hy : y ≠ 0)
  (h1 : |x| + y = 3) (h2 : |x| * y + x^3 = 0) : 
  Int.nearest (x - y) = -3 :=
sorry

end closest_integer_to_x_minus_y_l427_427443


namespace degree_measure_of_supplement_of_complement_of_35_degree_angle_l427_427197

def complement (α : ℝ) : ℝ := 90 - α
def supplement (β : ℝ) : ℝ := 180 - β

theorem degree_measure_of_supplement_of_complement_of_35_degree_angle : 
  supplement (complement 35) = 125 :=
by
  sorry

end degree_measure_of_supplement_of_complement_of_35_degree_angle_l427_427197


namespace opposite_of_neg_two_l427_427992

theorem opposite_of_neg_two : -(-2) = 2 := 
by 
  sorry

end opposite_of_neg_two_l427_427992


namespace ratio_of_squares_l427_427157

theorem ratio_of_squares (a b c : ℕ) (h : (a = 2) ∧ (b = 1) ∧ (c = 1)) :
  (∑ i in {a, b, c}, i) = 4 :=
by {
  cases h with h1 h2,
  cases h2 with h2 h3,
  rw [h1, h2, h3],
  simp,
}

end ratio_of_squares_l427_427157


namespace color_films_count_l427_427259

variables (x y C : ℕ)
variables (h1 : 0.9615384615384615 = (C : ℝ) / ((2 * (y : ℝ) / 5) + (C : ℝ)))

theorem color_films_count (x y : ℕ) (C : ℕ) (h1 : 0.9615384615384615 = (C : ℝ) / ((2 * (y : ℝ) / 5) + (C : ℝ))) :
  C = 10 * y :=
sorry

end color_films_count_l427_427259


namespace eliot_votes_l427_427472

theorem eliot_votes (randy_votes shaun_votes eliot_votes : ℕ)
                    (h1 : randy_votes = 16)
                    (h2 : shaun_votes = 5 * randy_votes)
                    (h3 : eliot_votes = 2 * shaun_votes) :
                    eliot_votes = 160 :=
by {
  -- Proof will be conducted here
  sorry
}

end eliot_votes_l427_427472


namespace base_six_representation_l427_427372

theorem base_six_representation (b : ℕ) (h₁ : b = 6) :
  625₁₀.toDigits b = [2, 5, 2, 1] ∧ (625₁₀.toDigits b).length = 4 ∧ (625₁₀.toDigits b).head % 2 = 1 :=
by
  sorry

end base_six_representation_l427_427372


namespace coefficient_of_x3y7_in_expansion_l427_427185

-- Definitions based on the conditions in the problem
def a : ℚ := (2 / 3)
def b : ℚ := - (3 / 4)
def n : ℕ := 10
def k1 : ℕ := 3
def k2 : ℕ := 7

-- Statement of the math proof problem
theorem coefficient_of_x3y7_in_expansion :
  (a * x ^ k1 + b * y ^ k2) ^ n = x3y7_coeff * x ^ k1 * y ^ k2  :=
sorry

end coefficient_of_x3y7_in_expansion_l427_427185


namespace area_union_square_circle_l427_427686

-- Define the side length of the square and the radius of the circle
def side_length_square : ℕ := 12
def radius_circle : ℕ := 12

-- Define the area of the square
def area_square : ℝ := (side_length_square : ℝ) * (side_length_square : ℝ)

-- Define the area of the circle
def area_circle : ℝ := Real.pi * (radius_circle : ℝ) * (radius_circle : ℝ)

-- Define the area of one quarter of the circle
def area_quarter_circle : ℝ := area_circle / 4

-- Define the area of the union of the square and the circle
def area_union : ℝ := area_square + (area_circle - area_quarter_circle)

-- Theorem: The area of the union of the regions enclosed by the square and the circle
theorem area_union_square_circle : area_union = 144 + 108 * Real.pi :=
by
  sorry

end area_union_square_circle_l427_427686


namespace perfect_square_n_l427_427770

def is_perfect_square (x : ℕ) : Prop :=
  ∃ y : ℕ, y * y = x

theorem perfect_square_n (n : ℕ) : 
  is_perfect_square (nat.factorial 1 * nat.factorial 2 * nat.factorial 3 * 
    ((finset.range (2 * n + 1).succ).filter nat.even).prod (!.) * 
    nat.factorial (2 * n)) ∧ (n = 4 * k * (k + 1) ∨ n = 2 * k ^ 2 - 1) :=
by
  sorry

end perfect_square_n_l427_427770


namespace total_clowns_l427_427964

theorem total_clowns (mobiles : ℕ) (clowns_per_mobile : ℕ) (num_clowns : ℕ) 
    (h1 : mobiles = 5) (h2 : clowns_per_mobile = 28) : 
    (clowns_per_mobile * mobiles = num_clowns) → num_clowns = 140 :=
by
  intros h3
  rw [h1, h2] at h3
  exact h3

end total_clowns_l427_427964


namespace number_of_digits_in_base_8_l427_427589

-- Define the given expression
def expr : ℕ := 4^20 * 5^18

-- Base to which we want to convert (base 8)
def base : ℕ := 8

-- Hypothesis: number of digits when expr is written in base 8 should be 34
theorem number_of_digits_in_base_8 :
  let digits := Nat.logBase base expr in
  digits + 1 = 34 := 
by
  -- Skipping the proof steps, as per instruction
  sorry

end number_of_digits_in_base_8_l427_427589


namespace find_a_range_l427_427407

noncomputable def f (x : ℝ) := (x - 1) / Real.exp x

noncomputable def condition_holds (a : ℝ) : Prop :=
∀ t ∈ (Set.Icc (1/2 : ℝ) 2), f t > t

theorem find_a_range (a : ℝ) (h : condition_holds a) : a > Real.exp 2 + 1/2 := sorry

end find_a_range_l427_427407


namespace bob_salary_is_14400_l427_427923

variables (mario_salary_current : ℝ) (mario_salary_last_year : ℝ) (bob_salary_last_year : ℝ) (bob_salary_current : ℝ)

-- Given Conditions
axiom mario_salary_increase : mario_salary_current = 4000
axiom mario_salary_equation : 1.40 * mario_salary_last_year = mario_salary_current
axiom bob_salary_last_year_equation : bob_salary_last_year = 3 * mario_salary_current
axiom bob_salary_increase : bob_salary_current = bob_salary_last_year + 0.20 * bob_salary_last_year

-- Theorem to prove
theorem bob_salary_is_14400 
    (mario_salary_last_year_eq : mario_salary_last_year = 4000 / 1.40)
    (bob_salary_last_year_eq : bob_salary_last_year = 3 * 4000)
    (bob_salary_current_eq : bob_salary_current = 12000 + 0.20 * 12000) :
    bob_salary_current = 14400 := 
by
  sorry

end bob_salary_is_14400_l427_427923


namespace sum_of_distinct_real_numbers_l427_427504

theorem sum_of_distinct_real_numbers (p q r s : ℝ) (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : ∀ x : ℝ, x^2 - 12 * p * x - 13 * q = 0 -> (x = r ∨ x = s)) 
  (h2 : ∀ x : ℝ, x^2 - 12 * r * x - 13 * s = 0 -> (x = p ∨ x = q)) :
  p + q + r + s = 2028 :=
begin
  sorry
end

end sum_of_distinct_real_numbers_l427_427504


namespace ten_women_seating_l427_427567

def seating_arrangements : ℕ → ℕ
| 1 := 1
| 2 := 2
| n := seating_arrangements (n - 1) + seating_arrangements (n - 2)

theorem ten_women_seating : seating_arrangements 10 = 89 := 
by
  unfold seating_arrangements
  simp
  sorry

end ten_women_seating_l427_427567


namespace polynomial_roots_l427_427748

open Real

theorem polynomial_roots : 
    ∀ x : ℝ, x^4 - 4 * x^3 + 3 * x^2 + 4 * x - 4 = 0 ↔ x = 1 ∨ x = 2 ∨ x = -1 :=
by
  intro x
  split
  repeat sorry

end polynomial_roots_l427_427748


namespace distance_AB_eq_sqrt3_max_distance_C2_to_l_eq_sqrt10_over_4_add_1_over_2_l427_427814

noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (1 + (1/2) * t, (sqrt 3) / 6 * t)

noncomputable def curve_C1 (θ : ℝ) : ℝ × ℝ :=
  (cos θ, sin θ)

noncomputable def curve_C2 (θ : ℝ) : ℝ × ℝ :=
  (1/2 * cos θ, (sqrt 3) / 2 * sin θ)

theorem distance_AB_eq_sqrt3 :
  let A := (1, 0)
  let B := (-1/2, - sqrt 3 / 2)
  real.dist A B = sqrt 3 :=
sorry

theorem max_distance_C2_to_l_eq_sqrt10_over_4_add_1_over_2 :
  ∃ θ : ℝ, let P := curve_C2 θ in
  let d := abs ((1/2 * cos θ - (3/2) * sin θ - 1)/2) in
  d = (sqrt (10) / 4) + (1/2) :=
sorry

end distance_AB_eq_sqrt3_max_distance_C2_to_l_eq_sqrt10_over_4_add_1_over_2_l427_427814


namespace prove_problem_1_prove_problem_2_l427_427715

noncomputable def problem_1 : Prop :=
  (\sqrt 24 - \sqrt 6) / \sqrt 3 - (\sqrt 3 + \sqrt 2) * (\sqrt 3 - \sqrt 2) = \sqrt 2 - 1

noncomputable def problem_2 : Prop :=
  ∃ x : ℝ, 2 * x^3 - 16 = 0 ∧ x = 2

theorem prove_problem_1 : problem_1 :=
  sorry

theorem prove_problem_2 : problem_2 :=
  sorry

end prove_problem_1_prove_problem_2_l427_427715


namespace num_satisfying_n_l427_427084

theorem num_satisfying_n : 
  {n : ℕ // n > 0 ∧ n ≤ 2016 ∧ 
  (frac (n / 2) + frac (n / 4) + frac (n / 6) + frac (n / 12) = 3)}.card = 168 := 
sorry

end num_satisfying_n_l427_427084


namespace symmetric_point_exists_l427_427955

noncomputable def symmetric_point (A B : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  ∃ x y, B = (x, y) ∧ A = (-1, 2) ∧ l (x, y) ∧ l ((x - 1) / 2, (y + 2) / 2)

theorem symmetric_point_exists :
  symmetric_point (-1, 2) (1, 4) (λ P, (P.1 + P.2 - 3 = 0)) :=
sorry

end symmetric_point_exists_l427_427955


namespace sum_of_coefficients_is_25_l427_427750

def polynomial_sum_of_coefficients : ℤ := 
  let p₁ := -2 * (C - C * X^7 + C * X^4 - C * 3 * X^2 - C * 5)
  let p₂ := 4 * (C + 2 * X)
  let p₃ := -3 * (C - X^5 - C * (-4))
  let polynomial := p₁ + p₂ + p₃
  polynomial.sum

theorem sum_of_coefficients_is_25 : polynomial_sum_of_coefficients = 25 := by
  sorry

end sum_of_coefficients_is_25_l427_427750


namespace volume_of_pyramid_l427_427270

-- Define the geometry of the problem
structure Tetrahedron :=
  (A B C D : Point) -- Vertices of the tetrahedron
  (unit_length : ∀ {X Y}, (X = A ∨ X = B ∨ X = C ∨ X = D) → (Y = A ∨ Y = B ∨ Y = C ∨ Y = D) → X ≠ Y → distance X Y = 1)
  (angle_60 : ∀ {X Y Z}, (Y = A ∨ Z = A) → angle X Y Z = 60)
  (angle_90 : ∀ {X Y Z}, (Y = A ∨ Z = A) → angle X Y Z = 90)
  (angle_120 : ∀ {X Y Z}, (Y = A ∨ Z = A) → angle X Y Z = 120)

-- The main theorem to prove
theorem volume_of_pyramid (T : Tetrahedron) : volume T = sqrt 2 / 12 := 
sorry

end volume_of_pyramid_l427_427270


namespace total_number_of_cars_l427_427450

theorem total_number_of_cars (T A R : ℕ)
  (h1 : T - A = 37)
  (h2 : R ≥ 41)
  (h3 : ∀ x, x ≤ 59 → A = x + 37) :
  T = 133 :=
by
  sorry

end total_number_of_cars_l427_427450


namespace average_weight_of_children_l427_427419

variable (ages : List ℕ)
variable (regression : ℕ → ℕ)

def average (lst : List ℕ) : ℕ := (lst.sum) / lst.length

theorem average_weight_of_children : 
  let avg_age := average ages 
  ages = [2, 3, 3, 5, 2, 6, 7, 3, 4, 5] → 
  regression = λ x, 2 * x + 7 → 
  let avg_weight := regression avg_age 
  avg_weight = 15 :=
by
  intros
  sorry

end average_weight_of_children_l427_427419


namespace sqrt_x_minus_2_meaningful_l427_427610

theorem sqrt_x_minus_2_meaningful (x : ℝ) (hx : x = 0 ∨ x = -1 ∨ x = -2 ∨ x = 2) : (x = 2) ↔ (x - 2 ≥ 0) :=
by
  sorry

end sqrt_x_minus_2_meaningful_l427_427610


namespace exact_one_correct_proposition_l427_427406

open_locale classical
noncomputable theory

def is_unit_vector (v : ℝ × ℝ) : Prop := ∥v∥ = 1
def collinear (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, v = k • w
def parallel (v w : ℝ × ℝ) : Prop := collinear v w
def magnitude (v : ℝ × ℝ) : ℝ := ∥v∥

def proposition_1 (a b : ℝ × ℝ) : Prop :=
  is_unit_vector a ∧ is_unit_vector b ∧ collinear a b → a = b

def proposition_2 (AB BA : ℝ × ℝ) (hAB_nonzero : AB ≠ 0 ∧ BA ≠ 0) : Prop :=
  parallel AB BA ∧ magnitude AB = magnitude BA

def proposition_3 (a b : ℝ × ℝ) : Prop :=
  collinear a b → magnitude (a + b) = magnitude a + magnitude b

def proposition_4 (a b c : ℝ × ℝ) : Prop :=
  parallel a b ∧ parallel b c → parallel a c

def number_of_correct_propositions : ℕ :=
  (if ¬ (proposition_1 a b) then 0 else 1) +
  (if proposition_2 AB BA hAB_nonzero then 1 else 0) +
  (if ¬ (proposition_3 a b) then 0 else 1) +
  (if ¬ (proposition_4 a b c) then 0 else 1)

theorem exact_one_correct_proposition (a b c AB BA : ℝ × ℝ)
    (hAB_nonzero : AB ≠ 0 ∧ BA ≠ 0) :
  number_of_correct_propositions = 1 :=
by sorry

end exact_one_correct_proposition_l427_427406


namespace remaining_macaroons_weight_is_103_l427_427477

-- Definitions based on the conditions
def coconutMacaroonsInitialCount := 12
def coconutMacaroonWeight := 5
def coconutMacaroonsBags := 4

def almondMacaroonsInitialCount := 8
def almondMacaroonWeight := 8
def almondMacaroonsBags := 2

def whiteChocolateMacaroonsInitialCount := 2
def whiteChocolateMacaroonWeight := 10

def steveAteCoconutMacaroons := coconutMacaroonsInitialCount / coconutMacaroonsBags
def steveAteAlmondMacaroons := (almondMacaroonsInitialCount / almondMacaroonsBags) / 2
def steveAteWhiteChocolateMacaroons := 1

-- Calculation of remaining macaroons weights
def remainingCoconutMacaroonsCount := coconutMacaroonsInitialCount - steveAteCoconutMacaroons
def remainingAlmondMacaroonsCount := almondMacaroonsInitialCount - steveAteAlmondMacaroons
def remainingWhiteChocolateMacaroonsCount := whiteChocolateMacaroonsInitialCount - steveAteWhiteChocolateMacaroons

-- Calculation of total remaining weight
def remainingCoconutMacaroonsWeight := remainingCoconutMacaroonsCount * coconutMacaroonWeight
def remainingAlmondMacaroonsWeight := remainingAlmondMacaroonsCount * almondMacaroonWeight
def remainingWhiteChocolateMacaroonsWeight := remainingWhiteChocolateMacaroonsCount * whiteChocolateMacaroonWeight

def totalRemainingWeight := remainingCoconutMacaroonsWeight + remainingAlmondMacaroonsWeight + remainingWhiteChocolateMacaroonsWeight

-- Statement to be proved
theorem remaining_macaroons_weight_is_103 :
  totalRemainingWeight = 103 := by
  sorry

end remaining_macaroons_weight_is_103_l427_427477


namespace count_integers_between_fractions_l427_427829

theorem count_integers_between_fractions : 
  ∃ (T_count : ℕ), T_count = 33 ∧ 
  ∀ (T : ℕ), 0 < T →
    let a := 2010 / T in
    let b := (2010 + T) / (2 * T) in
    (4 < a - b ∧ a - b ≤ 6) → 
    T ≥ 168 ∧ T ≤ 182 :=
by sorry

end count_integers_between_fractions_l427_427829


namespace work_completion_time_l427_427233

noncomputable def A_rate : ℚ := 1 / 6
noncomputable def B_rate : ℚ := 1 / 12
noncomputable def combined_rate : ℚ := A_rate + B_rate

theorem work_completion_time :
  combined_rate = 1 / 4 →
  1 / combined_rate = 4 :=
by
  intros h
  rw h
  norm_num
  sorry

end work_completion_time_l427_427233


namespace distance_between_parallel_lines_l427_427014

theorem distance_between_parallel_lines :
  let a := 4
  let b := -3
  let c1 := 2
  let c2 := -1
  let d := (abs (c1 - c2)) / (Real.sqrt (a^2 + b^2))
  d = 3 / 5 :=
by
  sorry

end distance_between_parallel_lines_l427_427014


namespace min_value_an_div_n_l427_427420

noncomputable def a : ℕ → ℕ
| 1 := 33
| (n + 1) := a n + 2 * n

theorem min_value_an_div_n : (∃ n, ∀ m, 1 ≤ m → (a n / n) ≤ (a m / m)) ∧ (a 6 / 6) = 21 / 2 := by
  sorry

end min_value_an_div_n_l427_427420


namespace all_cells_black_l427_427901

noncomputable def smallest_k (n : ℕ) : ℕ := n^2 + n + 1

theorem all_cells_black (n : ℕ) (h_pos : n > 0) :
    ∀ (grid : ℕ → ℕ → bool), 
    (∃ k, k = smallest_k n ∧ count_black_cells grid = k) →
    (∀ (i j : ℕ), i < 2 * n → j < 2 * n → grid i j == tt) :=
begin
    sorry,
end

def count_black_cells(grid : ℕ → ℕ → bool) : ℕ := 
    fin (2 * 2).sum (fin (2 * 2)).sum (λ i j, if grid i j then 1 else 0)


end all_cells_black_l427_427901


namespace part1_c_eqn_cartesian_part1_l_eqn_rectangular_part2_m_range_l427_427457

-- Definitions for parametric equations of curve C
def C_parametric_eqns (α : ℝ) : ℝ × ℝ :=
  (1 + 3 * Real.cos α, 3 + 3 * Real.sin α)

-- Definition for polar coordinate equation of line l
def l_polar_eqn (ρ θ m : ℝ) : Prop :=
  ρ * Real.cos θ + ρ * Real.sin θ = m

-- Cartesian equation of curve C
def C_cartesian_eqn (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 9

-- Rectangular coordinate equation of line l
def l_rectangular_eqn (x y m : ℝ) : Prop :=
  x + y - m = 0

-- Range of m for curve C to intersect line l at two points
def m_range (m : ℝ) : Prop :=
  4 - 3 * Real.sqrt 2 < m ∧ m < 4 + 3 * Real.sqrt 2

-- Theorem statements
theorem part1_c_eqn_cartesian (α : ℝ) : ∀ x y, (x, y) = C_parametric_eqns α → C_cartesian_eqn x y := by
  sorry

theorem part1_l_eqn_rectangular (ρ θ m x y : ℝ) (h : l_polar_eqn ρ θ m) : (x, y) = (ρ * Real.cos θ, ρ * Real.sin θ) → l_rectangular_eqn x y m := by
  sorry

theorem part2_m_range (m : ℝ) : ∀ ρ θ, ∃ x y, (x, y) = C_parametric_eqns (Real.atan 2) → l_polar_eqn ρ θ m → m_range m := by
  sorry

end part1_c_eqn_cartesian_part1_l_eqn_rectangular_part2_m_range_l427_427457


namespace true_propositions_l427_427803

noncomputable def x_neg := ∀ x : ℝ, x^2 + x + 1 < 0 → false
noncomputable def sin_contrapositive := ∀ x y : ℝ, (sin x = sin y) → (x = y)

theorem true_propositions :
    x_neg ∧ sin_contrapositive :=
by
  sorry

end true_propositions_l427_427803


namespace ratio_side_lengths_sum_l427_427146

open Real

def ratio_of_areas (a b : ℝ) : ℝ := a / b

noncomputable def sum_of_abc (area_ratio : ℝ) : ℤ :=
  let side_length_ratio := sqrt area_ratio
  let a := 2
  let b := 1
  let c := 1
  a + b + c

theorem ratio_side_lengths_sum :
  sum_of_abc (ratio_of_areas 300 75) = 4 :=
by
  sorry

end ratio_side_lengths_sum_l427_427146


namespace golden_state_team_points_l427_427044

theorem golden_state_team_points :
  let Draymond := 12
  let Curry := 2 * Draymond
  let Kelly := 9
  let Durant := 2 * Kelly
  let Klay := Draymond / 2
  Draymond + Curry + Kelly + Durant + Klay = 69 :=
by
  let Draymond := 12
  let Curry := 2 * Draymond
  let Kelly := 9
  let Durant := 2 * Kelly
  let Klay := Draymond / 2
  rw [Draymond, Curry, Kelly, Durant, Klay]
  sorry

end golden_state_team_points_l427_427044


namespace bill_tossed_21_objects_l427_427707

-- Definitions based on the conditions from step a)
def ted_sticks := 10
def ted_rocks := 10
def bill_sticks := ted_sticks + 6
def bill_rocks := ted_rocks / 2

-- The condition of total objects tossed by Bill
def bill_total_objects := bill_sticks + bill_rocks

-- The theorem we want to prove
theorem bill_tossed_21_objects :
  bill_total_objects = 21 :=
  by
  sorry

end bill_tossed_21_objects_l427_427707


namespace minimum_dot_product_l427_427998

noncomputable def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
noncomputable def circle (r a x y : ℝ) : Prop := (x - a)^2 + y^2 = r^2
noncomputable def vectors_dot_product (px py mx my fx fy : ℝ) : ℝ := 
(px - mx) * (px - fx) + (py - my) * (py - fy)

theorem minimum_dot_product (p : ℝ) (x y mx my fx fy : ℝ) 
  (h₀ : p = 2)
  (h₁ : parabola p x y)
  (h₂ : circle 2 2 mx my)
  (h₃ : fx = p / 2)  -- focus of the parabola when p = 2
  (h₄ : my = 0)       -- y-coordinate of circle's center is 0
  : vectors_dot_product p y mx my fx fy ≥ 2 :=
by
  sorry

end minimum_dot_product_l427_427998


namespace no_positive_pairs_l427_427310

theorem no_positive_pairs (x y : ℕ) (h1 : x^2 + y^2 = x^4) (h2 : x > y) :
  false :=
begin
  sorry
end

end no_positive_pairs_l427_427310


namespace y_relationship_l427_427389

theorem y_relationship (x1 x2 x3 y1 y2 y3 : ℝ) 
  (h1: y1 = -4 / x1) (h2: y2 = -4 / x2) (h3: y3 = -4 / x3)
  (h4: x1 < 0) (h5: 0 < x2) (h6: x2 < x3) :
  y1 > y3 ∧ y3 > y2 :=
by
  sorry

end y_relationship_l427_427389


namespace path_exists_every_cell_and_ends_at_marked_l427_427127

-- Define the checkerboard pattern and the movement constraints.
structure Checkerboard (n : ℕ) :=
(board : list (list bool))
(dim : ℕ := 2015)
(is_checkerboard : ∀ i j, board i j = (i + j) % 2 = 1)
(corner_black : board 0 0 = tt ∧ board 0 (dim-1) = tt ∧ board (dim-1) 0 = tt ∧ board (dim-1) (dim-1) = tt)

-- Define possible moves on the board.
inductive Move : Type
| left
| right
| up
| down

-- Define adjacency relation based on possible moves.
def is_adjacent (i1 j1 i2 j2 : ℕ) : Prop :=
(i1 = i2 ∧ (j1 = j2 + 1 ∨ j1 = j2 - 1)) ∨ (j1 = j2 ∧ (i1 = i2 + 1 ∨ i1 = i2 - 1))

-- The main problem statement in Lean.
theorem path_exists_every_cell_and_ends_at_marked : ∀ (start : ℕ × ℕ) (end : ℕ × ℕ),
  start.1 < 2015 ∧ start.2 < 2015 ∧ end.1 < 2015 ∧ end.2 < 2015 ∧
  (start.1 + start.2) % 2 = 0 ∧ (end.1 + end.2) % 2 = 0 →
  ∃ (path : list (ℕ × ℕ)),
  (∀ (i j : ℕ), (i < 2015 ∧ j < 2015) → ∃ (k : ℕ), (i, j) = path.nth k) ∧ (path.head = start) ∧ (path.last = end) :=
begin
  intros start end h,
  -- Proof goes here
  sorry
end

end path_exists_every_cell_and_ends_at_marked_l427_427127


namespace variance_scaled_variance_formula_variance_transformed_data_l427_427496

-- Define the average of a list
def avg (l : List ℝ) : ℝ := l.sum / l.length

-- Definitions related to variance
def variance (l : List ℝ) : ℝ := (l.map (λ x => (x - avg l) ^ 2)).sum / l.length

-- Given conditions
variables (x : List ℝ)
variables (m n : ℝ)
variables (a : ℝ) (h_var : variance [2, 3, 5, m, n] = 2)

-- Statement 1: Prove that scaling the data by 'a' scales the variance by 'a^2'
theorem variance_scaled (a : ℝ) (x : List ℝ) : variance (x.map (λ xi => a * xi)) = a ^ 2 * (variance x) :=
sorry

-- Statement 2: Prove the given formula for variance
theorem variance_formula (x : List ℝ) : variance x = (x.map (λ xi => xi ^ 2)).sum / x.length - (avg x) ^ 2 :=
sorry

-- Statement 3: Prove the variance for transformed data set is still 2
theorem variance_transformed_data (m n : ℝ) (h_var : variance [2, 3, 5, m, n] = 2) :
  variance [4, 5, 7, m+2, n+2] = 2 :=
sorry

end variance_scaled_variance_formula_variance_transformed_data_l427_427496


namespace supplement_of_complement_is_125_l427_427206

-- Definition of the initial angle
def initial_angle : ℝ := 35

-- Definition of the complement of the angle
def complement_angle (θ : ℝ) : ℝ := 90 - θ

-- Definition of the supplement of an angle
def supplement_angle (θ : ℝ) : ℝ := 180 - θ

-- Main theorem statement
theorem supplement_of_complement_is_125 : 
  supplement_angle (complement_angle initial_angle) = 125 := 
by
  sorry

end supplement_of_complement_is_125_l427_427206


namespace max_dot_product_condition_l427_427797

noncomputable def vector_a : ℝ × ℝ := (1, 0)
noncomputable def vector_b : ℝ × ℝ := (-1/2, sqrt 3 / 2)

def minimum_value_condition (t : ℝ) : ℝ :=
  sqrt ((t - (-1/2))^2 + (sqrt 3 / 2)^2)

theorem max_dot_product_condition :
  (∀ t : ℝ, minimum_value_condition t ≥ sqrt(3)/2) →
  (∀ c : ℝ × ℝ, (c - vector_a) • (c - vector_b) = 0 → 
    max (c • (vector_a + vector_b)) = (sqrt 3 + 1) / 2) :=
by
  sorry

end max_dot_product_condition_l427_427797


namespace last_installment_value_l427_427143

-- Define the initial conditions
def initial_price : ℝ := 10000
def num_installments : ℕ := 20
def installment_amount : ℝ := 1000
def annual_interest_rate : ℝ := 0.06
def first_installment_paid_at_purchase : Prop := true

-- Define the total amount paid
def total_installments_paid : ℝ := installment_amount * num_installments

-- Define the function to calculate the remaining balance after each installment
def remaining_balance (n : ℕ) : ℝ :=
  initial_price - (n * installment_amount)

-- Define interest calculation based on the remaining balance
def monthly_interest (balance : ℝ) : ℝ :=
  balance * (annual_interest_rate / 12)

-- Define that interest is paid only on the remaining balance each month
def interest_paid (n : ℕ) : ℝ :=
  monthly_interest (remaining_balance n)

-- Define the statement to be proved
theorem last_installment_value :
  first_installment_paid_at_purchase →
  (remaining_balance (num_installments - 1) = 0) →
  total_installments_paid = 20000 →
  installment_amount = 1000 :=
by
  intros h_first_inst h_rem_balance h_total_paid
  sorry

end last_installment_value_l427_427143


namespace power_function_through_point_l427_427397

-- Define the conditions
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x, f(x) = x^α

def passes_through_point (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  f p.1 = p.2

-- State the main theorem to be proven
theorem power_function_through_point {f : ℝ → ℝ} (h1 : is_power_function f) (h2 : passes_through_point f (2, 4)) : 
  f = (λ x, x^2) :=
sorry

end power_function_through_point_l427_427397


namespace ratio_of_squares_l427_427155

theorem ratio_of_squares (a b c : ℕ) (h : (a = 2) ∧ (b = 1) ∧ (c = 1)) :
  (∑ i in {a, b, c}, i) = 4 :=
by {
  cases h with h1 h2,
  cases h2 with h2 h3,
  rw [h1, h2, h3],
  simp,
}

end ratio_of_squares_l427_427155


namespace shaded_area_correct_l427_427873

noncomputable def total_shaded_area : ℝ :=
  let r1 := 3 : ℝ
  let r2 := 5 : ℝ
  let A_rectangles := (2 * r1 * r1) + (2 * r2 * r2)
  let A_small_circle := π * (r1 ^ 2)
  let A_large_circle := π * (r2 ^ 2)
  let A_overlap := (π * (r1 ^ 2)) / 2
  A_rectangles - A_small_circle - A_large_circle + A_overlap

theorem shaded_area_correct : total_shaded_area = 68 - 29.5 * π :=
by
  unfold total_shaded_area
  sorry

end shaded_area_correct_l427_427873


namespace fraction_reducible_to_17_l427_427611

theorem fraction_reducible_to_17 (m n : ℕ) (h_coprime : Nat.gcd m n = 1)
  (h_reducible : ∃ d : ℕ, d ∣ (3 * m - n) ∧ d ∣ (5 * n + 2 * m)) :
  ∃ k : ℕ, (3 * m - n) / k = 17 ∧ (5 * n + 2 * m) / k = 17 :=
by
  have key : Nat.gcd (3 * m - n) (5 * n + 2 * m) = 17 := sorry
  -- using the result we need to construct our desired k
  use 17 / (Nat.gcd (3 * m - n) (5 * n + 2 * m))
  -- rest of intimate proof here
  sorry

end fraction_reducible_to_17_l427_427611


namespace math_problem_l427_427592

def otimes (a b : ℚ) : ℚ := (a^3) / (b^2)

theorem math_problem : ((otimes (otimes 2 4) 6) - (otimes 2 (otimes 4 6))) = -23327 / 288 := by sorry

end math_problem_l427_427592


namespace number_of_children_per_seat_l427_427650

variable (children : ℕ) (seats : ℕ)

theorem number_of_children_per_seat (h1 : children = 58) (h2 : seats = 29) :
  children / seats = 2 := by
  sorry

end number_of_children_per_seat_l427_427650


namespace complement_of_intersection_l427_427085

def I := {x : ℕ | |x - 2| ≤ 2 ∧ x > 0}
def P := {1, 2, 3}
def Q := {2, 3, 4}

theorem complement_of_intersection (h : ∀ x, x ∈ I → x ∈ P ∪ Q) : 
  {x ∈ I | x ∉ (P ∩ Q)} = {1, 4} := by
  sorry

end complement_of_intersection_l427_427085


namespace number_of_5_dollar_bills_l427_427538

def total_money : ℤ := 45
def value_of_each_bill : ℤ := 5

theorem number_of_5_dollar_bills : total_money / value_of_each_bill = 9 := by
  sorry

end number_of_5_dollar_bills_l427_427538


namespace cube_sum_equal_one_l427_427118

theorem cube_sum_equal_one (x y z : ℝ) (h1 : x + y + z = 3) (h2 : xy + xz + yz = 1) (h3 : xyz = 1) :
  x^3 + y^3 + z^3 = 1 := 
sorry

end cube_sum_equal_one_l427_427118


namespace system_solution_l427_427947

theorem system_solution (x y : ℝ) :
  (x^2 * y + x * y^2 - 2 * x - 2 * y + 10 = 0) ∧ 
  (x^3 * y - x * y^3 - 2 * x^2 + 2 * y^2 - 30 = 0) ↔ 
  (x = -4) ∧ (y = -1) :=
by
  sorry

end system_solution_l427_427947


namespace column_independent_of_row_l427_427912

theorem column_independent_of_row (n : ℕ) (a b : Fin n → ℝ) 
  (ha : Function.Injective a) (hb : Function.Injective b)
  (h : ∀ i j1 j2, (∏ j, a i + b j1 = ∏ j, a i + b j2)) :
  ∀ i1 i2 j, (∏ i, a i1 + b j = ∏ i, a i2 + b j) :=
by
  sorry

end column_independent_of_row_l427_427912


namespace degree_meas_supp_compl_35_l427_427204

noncomputable def degree_meas_supplement_complement (θ : ℝ) : ℝ :=
  180 - (90 - θ)

theorem degree_meas_supp_compl_35 : degree_meas_supplement_complement 35 = 125 :=
by
  unfold degree_meas_supplement_complement
  norm_num
  sorry

end degree_meas_supp_compl_35_l427_427204


namespace find_m_l427_427843

theorem find_m (m : ℝ) :
  (∀ x y : ℝ, (3 * x + (m + 1) * y - (m - 7) = 0) → 
              (m * x + 2 * y + 3 * m = 0)) →
  (m + 1 ≠ 0) →
  m = -3 :=
by
  sorry

end find_m_l427_427843


namespace smallest_possible_combined_munificence_l427_427013

noncomputable def polynomial_munificence (f : ℝ → ℝ) (interval : Set ℝ) : ℝ :=
  Sup (AbsImage f interval)

def AbsImage (f : ℝ → ℝ) (interval : Set ℝ) : Set ℝ :=
  {y | ∃ x ∈ interval, y = |f x|}

theorem smallest_possible_combined_munificence :
  ∀ (b c d e : ℝ), b ≠ d →
  let f := λ x : ℝ, x^2 + b * x + c
  let g := λ x : ℝ, x^2 + d * x + e
  let interval := Set.Icc (-1 : ℝ) (1 : ℝ)
  let Mf := polynomial_munificence f interval
  let Mg := polynomial_munificence g interval
  Mf = 1 / 2 ∧ Mg = 3 / 2 → max Mf Mg = 3 / 2 :=
by
  sorry

end smallest_possible_combined_munificence_l427_427013


namespace find_smallest_M_l427_427363

/-- 
Proof of the smallest real number M such that 
for all real numbers a, b, and c, the following inequality holds:
    |a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)|
    ≤ (9 * Real.sqrt 2 / 32) * (a^2 + b^2 + c^2)^2. 
-/
theorem find_smallest_M (a b c : ℝ) : 
    |a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)| 
    ≤ (9 * Real.sqrt 2 / 32) * (a^2 + b^2 + c^2)^2 :=
by
  sorry

end find_smallest_M_l427_427363


namespace degree_meas_supp_compl_35_l427_427199

noncomputable def degree_meas_supplement_complement (θ : ℝ) : ℝ :=
  180 - (90 - θ)

theorem degree_meas_supp_compl_35 : degree_meas_supplement_complement 35 = 125 :=
by
  unfold degree_meas_supplement_complement
  norm_num
  sorry

end degree_meas_supp_compl_35_l427_427199


namespace max_n_factorable_l427_427741

theorem max_n_factorable :
  ∃ n : ℤ, (∀ A B : ℤ, 3 * A * B = 24 → 3 * B + A = n) ∧ (n = 73) :=
sorry

end max_n_factorable_l427_427741


namespace is_it_possible_to_fence_l427_427678

-- Define a structure for holding the problem's conditions
structure FieldFencingProblem where
  field_length : ℝ
  field_width : ℝ
  field_area : ℝ
  obstacle_side : ℝ
  available_fencing : ℝ

-- Define the field fencing specific problem with the given conditions
def specificFieldFencingProblem : FieldFencingProblem :=
  { field_length := 40
  , field_width := 25
  , field_area := 1000
  , obstacle_side := 15
  , available_fencing := 100
  }

-- Define the concept of calculating the total fencing required
def totalFencingRequired (ffp : FieldFencingProblem) : ℝ :=
  (ffp.field_length + 2 * ffp.field_width) + 4 * ffp.obstacle_side

-- The main theorem stating the answer to the converted problem
theorem is_it_possible_to_fence (ffp : FieldFencingProblem) : totalFencingRequired ffp > ffp.available_fencing :=
  by
    unfold specificFieldFencingProblem
    unfold totalFencingRequired
    sorry

end is_it_possible_to_fence_l427_427678


namespace perfect_square_condition_l427_427766

-- Define the product of factorials from 1 to 2n
def factorial_product (n : ℕ) : ℕ :=
  (List.prod (List.map factorial (List.range (2 * n + 1))))

-- The main theorem statement
theorem perfect_square_condition (n : ℕ) : 
  (∃ k : ℕ, n = 4 * k * (k + 1) ∨ n = 2 * k^2 - 1) ↔
  (∃ m : ℕ, (factorial_product n) / ((n + 1)!) = m^2) := sorry

end perfect_square_condition_l427_427766


namespace like_terms_sum_l427_427141

theorem like_terms_sum (m n : ℕ) (h1 : 6 * x ^ 5 * y ^ (2 * n) = 6 * x ^ m * y ^ 4) : m + n = 7 := by
  sorry

end like_terms_sum_l427_427141


namespace find_xy_l427_427402

theorem find_xy (x y : ℝ) (h1 : x^2 + y^2 = 15) (h2 : (x - y)^2 = 9) : x * y = 3 :=
sorry

end find_xy_l427_427402


namespace sum_of_distinct_roots_l427_427519

theorem sum_of_distinct_roots 
  (p q r s : ℝ)
  (h1 : p ≠ q)
  (h2 : p ≠ r)
  (h3 : p ≠ s)
  (h4 : q ≠ r)
  (h5 : q ≠ s)
  (h6 : r ≠ s)
  (h_roots1 : (x : ℝ) -> x^2 - 12*p*x - 13*q = 0 -> x = r ∨ x = s)
  (h_roots2 : (x : ℝ) -> x^2 - 12*r*x - 13*s = 0 -> x = p ∨ x = q) : 
  p + q + r + s = 1716 := 
by 
  sorry

end sum_of_distinct_roots_l427_427519


namespace coefficient_of_term_in_expansion_l427_427183

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  nat.choose n k

noncomputable def power_rat (r : ℚ) (n : ℕ) : ℚ :=
  r ^ n
  
theorem coefficient_of_term_in_expansion :
  ∃ (c : ℚ), 
  (∃ (x y : ℚ), (x = 3 ∧ y = 7) → 
  x^3 * y^7 = c * ((binomial_coefficient 10 3) * 
  (power_rat (2/3) 3) * (power_rat (-3/4) 7))) ∧
  c = -3645/7656 :=
begin
  sorry
end

end coefficient_of_term_in_expansion_l427_427183


namespace arithmetic_progression_sum_l427_427486

variables {n : ℕ} (a : ℕ → ℝ)
  (h_pos : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i) -- Positive real numbers
  (h_arith_prog : ∀ i, 1 ≤ i ∧ i ≤ n-1 → a (i+1) - a i = a 2 - a 1) -- Arithmetic progression

theorem arithmetic_progression_sum :
  ∑ i in finset.range (n-1), 1 / (real.sqrt (a (i+1)) + real.sqrt (a (i+2))) = 
    (n - 1) / (real.sqrt (a 1) + real.sqrt (a n)) :=
by
  sorry

end arithmetic_progression_sum_l427_427486


namespace gender_related_evaluation_expected_value_X_correct_l427_427656

-- Definitions
def total_viewers : ℕ := 100
def ratio_male : ℕ := 9
def ratio_female : ℕ := 11
def total_likes : ℕ := 50
def alpha : ℝ := 0.005
def critical_value : ℝ := 7.879
def male_viewers : ℕ := (ratio_male * total_viewers) / (ratio_male + ratio_female)
def female_viewers : ℕ := (ratio_female * total_viewers) / (ratio_male + ratio_female)

noncomputable def chi_squared_val (a b c d n : ℕ) : ℝ :=
  let term := n * (a * d - b * c)^2
  term / (n * (a + b) * (c + d) * (a + c) * (b + d))

-- Part 1: Proving gender is related to evaluation results
theorem gender_related_evaluation :
  chi_squared_val 30 20 35 15 total_viewers > critical_value :=
by sorry

-- Additional definitions for Part 2
def dislike_adoption_prob : ℝ := 1 / 4
def like_adoption_prob : ℝ := 3 / 4
def adopted_reward : ℕ := 100
def non_adopted_reward : ℕ := 50

noncomputable def expected_value_X : ℝ :=
  (150 * (9 / 64)) +
  (200 * (33 / 64)) +
  (250 * (19 / 64)) +
  (300 * (3 / 64))

-- Part 2: Proving expected value of X
theorem expected_value_X_correct : expected_value_X = 212.5 :=
by sorry

end gender_related_evaluation_expected_value_X_correct_l427_427656


namespace draw_red_before_green_probability_l427_427692

def probability_red_before_green (num_red num_green num_blue : ℕ) (prob : ℚ) : Prop :=
  num_red = 4 ∧ num_green = 3 ∧ num_blue = 1 →
  prob = 3/5

theorem draw_red_before_green_probability :
  probability_red_before_green 4 3 1 (3 / 5) :=
begin
  sorry
end

end draw_red_before_green_probability_l427_427692


namespace isosceles_triangle_base_angle_l427_427037

theorem isosceles_triangle_base_angle (x : ℝ) 
  (h1 : ∀ (a b : ℝ), a + b + (20 + 2 * b) = 180)
  (h2 : 20 + 2 * x = 180 - 2 * x - x) : x = 40 :=
by sorry

end isosceles_triangle_base_angle_l427_427037


namespace ratio_side_lengths_sum_l427_427149

open Real

def ratio_of_areas (a b : ℝ) : ℝ := a / b

noncomputable def sum_of_abc (area_ratio : ℝ) : ℤ :=
  let side_length_ratio := sqrt area_ratio
  let a := 2
  let b := 1
  let c := 1
  a + b + c

theorem ratio_side_lengths_sum :
  sum_of_abc (ratio_of_areas 300 75) = 4 :=
by
  sorry

end ratio_side_lengths_sum_l427_427149


namespace number_of_solutions_l427_427970

-- Define the given equation as a function
def equation (x : ℝ) : ℝ := x^2 - 2 * (Real.floor x : ℝ) - 3

-- State the main theorem to be proved
theorem number_of_solutions : ∃ (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, equation x = 0 :=
by
  sorry

end number_of_solutions_l427_427970


namespace problem_1_solution_set_problem_2_solution_set_problem_2_solution_set_special_l427_427364

-- Define the problems conditions
variable {x : ℝ}
variable {a : ℝ}

-- Problem 1
theorem problem_1_solution_set :
  {x : ℝ | (|x + 1| / |x + 2| >= 1)} = {x | x ≤ - 3 / 2 ∧ x ≠ - 2} :=
sorry

-- Problem 2
theorem problem_2_solution_set (h : a ≠ 1) :
  if (a > 1) then {x : ℝ | x > 2 ∨ x < (a - 2) / (a - 1)}
  else if (a = 1) then {x : ℝ | x > 2}
  else if (0 < a ∧ a < 1) then {x : ℝ | 2 < x ∧ x < (a - 2) / (a - 1)}
  else if (a = 0) then ∅
  else {x : ℝ | (a - 2) / (a - 1) < x ∧ x < 2} :=
sorry

-- Special case when a = 1
theorem problem_2_solution_set_special :
  {x : ℝ | (a = 1) → (a * (x - 1)) / (x - 2) > 1} = {x | x > 2} :=
sorry

end problem_1_solution_set_problem_2_solution_set_problem_2_solution_set_special_l427_427364


namespace B_N_Q_collinear_l427_427398

/-- Define point positions -/
structure Point where
  x : ℝ
  y : ℝ

def M : Point := ⟨-1, 0⟩
def N : Point := ⟨1, 0⟩

/-- Define the curve C -/
def on_curve_C (P : Point) : Prop :=
  P.x^2 + P.y^2 - 6 * P.x + 1 = 0

/-- Define reflection of point A across the x-axis -/
def reflection_across_x (A : Point) : Point :=
  ⟨A.x, -A.y⟩

/-- Define the condition that line l passes through M and intersects curve C at two distinct points A and B -/
def line_l_condition (A B: Point) (k : ℝ) (hk : k ≠ 0) : Prop :=
  A.y = k * (A.x + 1) ∧ B.y = k * (B.x + 1) ∧ on_curve_C A ∧ on_curve_C B

/-- Main theorem to prove collinearity of B, N, Q -/
theorem B_N_Q_collinear (A B : Point) (k : ℝ) (hk : k ≠ 0)
  (hA : on_curve_C A) (hB : on_curve_C B)
  (h_l : line_l_condition A B k hk) :
  let Q := reflection_across_x A
  (B.x - N.x) * (Q.y - N.y) = (B.y - N.y) * (Q.x - N.x) :=
sorry

end B_N_Q_collinear_l427_427398


namespace largest_increase_1998_l427_427579

/-- Sales figures from 1995 to 2005 in millions of dollars -/
def sales : ℕ → ℝ
| 0 => 3.0  -- 1995
| 1 => 4.5  -- 1996
| 2 => 5.1  -- 1997
| 3 => 7.0  -- 1998
| 4 => 8.5  -- 1999
| 5 => 9.7  -- 2000
| 6 => 10.7 -- 2001
| 7 => 12.0 -- 2002
| 8 => 13.2 -- 2003
| 9 => 13.7 -- 2004
| 10 => 7.5 -- 2005
| _ => 0.0  -- Default case (not used in this specific range)

/-- The year with the largest increase in sales compared to the previous year is 1998 -/
theorem largest_increase_1998 :
  ∀ k : ℕ, k ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} →
  (sales (k + 1) - sales k ≤ sales 3 - sales 2) :=
begin
  sorry
end

end largest_increase_1998_l427_427579


namespace exists_special_number_divisible_by_1991_l427_427114

theorem exists_special_number_divisible_by_1991 :
  ∃ (N : ℤ) (n : ℕ), n > 2 ∧ (N % 1991 = 0) ∧ 
  (∃ a b x : ℕ, N = 10 ^ (n + 1) * a + 10 ^ n * x + 9 * 10 ^ (n - 1) + b) :=
sorry

end exists_special_number_divisible_by_1991_l427_427114


namespace pqrs_sum_l427_427512

theorem pqrs_sum (p q r s : ℝ)
  (h1 : (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 → x = r ∨ x = s))
  (h2 : (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 → x = p ∨ x = q))
  (h3 : p ≠ q) (h4 : p ≠ r) (h5 : p ≠ s) (h6 : q ≠ r) (h7 : q ≠ s) (h8 : r ≠ s) :
  p + q + r + s = 2028 :=
sorry

end pqrs_sum_l427_427512


namespace range_of_a_l427_427780

open Real

theorem range_of_a (a : ℝ) :
  ((a = 0 ∨ (a > 0 ∧ a^2 - 4 * a < 0)) ∨ (a^2 - 2 * a - 3 < 0)) ∧
  ¬((a = 0 ∨ (a > 0 ∧ a^2 - 4 * a < 0)) ∧ (a^2 - 2 * a - 3 < 0)) ↔
  (-1 < a ∧ a < 0) ∨ (3 ≤ a ∧ a < 4) := 
sorry

end range_of_a_l427_427780


namespace a_share_is_approx_560_l427_427642

noncomputable def investment_share (a_invest b_invest c_invest total_months b_share : ℕ) : ℝ :=
  let total_invest := a_invest + b_invest + c_invest
  let total_profit := (b_share * total_invest) / b_invest
  let a_share_ratio := a_invest / total_invest
  (a_share_ratio * total_profit)

theorem a_share_is_approx_560 
  (a_invest : ℕ := 7000) 
  (b_invest : ℕ := 11000) 
  (c_invest : ℕ := 18000) 
  (total_months : ℕ := 8) 
  (b_share : ℕ := 880) : 
  ∃ (a_share : ℝ), abs (a_share - 560) < 1 :=
by
  let a_share := investment_share a_invest b_invest c_invest total_months b_share
  existsi a_share
  sorry

end a_share_is_approx_560_l427_427642


namespace area_of_triangle_l427_427885

variable {α : Type*} [DecidableEq α]

-- Define the triangle sides and their properties
variable (a b c : ℝ)
variable (A B C : ℝ)  -- Angles opposite to the sides a, b, c
variable (area : ℝ)

-- Conditions given in the problem
def is_arithmetic_sequence (a b c : ℝ) : Prop := b - a = a - c
def given_equation (a b c : ℝ) (B : ℝ) : Prop := c + a = 2 * a * (Real.cos (B / 2) ^ 2) + (1 / 2) * b
def given_value_of_a : Prop := a = 2

-- Final statement to be proved
theorem area_of_triangle : 
  is_arithmetic_sequence a b c → 
  given_equation a b c B → 
  given_value_of_a → 
  area = Real.sqrt 3 := 
by 
  sorry

end area_of_triangle_l427_427885


namespace employee_payment_l427_427241

theorem employee_payment (X Y : ℝ) (h1 : X + Y = 528) (h2 : X = 1.2 * Y) : Y = 240 :=
by
  sorry

end employee_payment_l427_427241


namespace minimum_and_maximum_attendees_more_than_one_reunion_l427_427608

noncomputable def minimum_attendees_more_than_one_reunion (total_guests oates_attendees hall_attendees brown_attendees : ℕ) : ℕ :=
  let total_unique_attendees := oates_attendees + hall_attendees + brown_attendees
  total_unique_attendees - total_guests

noncomputable def maximum_attendees_more_than_one_reunion (total_guests oates_attendees hall_attendees brown_attendees : ℕ) : ℕ :=
  oates_attendees

theorem minimum_and_maximum_attendees_more_than_one_reunion
  (total_guests oates_attendees hall_attendees brown_attendees : ℕ)
  (H1 : total_guests = 200)
  (H2 : oates_attendees = 60)
  (H3 : hall_attendees = 90)
  (H4 : brown_attendees = 80) :
  minimum_attendees_more_than_one_reunion total_guests oates_attendees hall_attendees brown_attendees = 30 ∧
  maximum_attendees_more_than_one_reunion total_guests oates_attendees hall_attendees brown_attendees = 60 :=
by
  sorry

end minimum_and_maximum_attendees_more_than_one_reunion_l427_427608


namespace part_one_l427_427007

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 3 ∧ a 2 = 6 ∧ ∀ n : ℕ, n ≥ 1 → a (n+2) = (a (n+1))^2 + 9 / a n

theorem part_one (a : ℕ → ℤ) (h_seq : sequence a) : ∀ n, a n > 0 := by
sorry

end part_one_l427_427007


namespace equation_solution_exists_l427_427573

noncomputable def S (n : ℕ) : set ℕ := {x : ℕ | x ≤ n}

def more_than_one_fourth_colored (S : set ℕ) (color_count : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ c, color_count c > n / 4

def different_colors (color : ℕ → ℕ) (x y z : ℕ) : Prop :=
  color x ≠ color y ∧ color y ≠ color z ∧ color z ≠ color x

def colors (color : ℕ → ℕ) (x y z : ℕ) : Prop :=
  ∃ a b c, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ color x = a ∧ color y = b ∧ color z = c

theorem equation_solution_exists 
  (n : ℕ) 
  (color_count : ℕ → ℕ) 
  (color : ℕ → ℕ)
  (h1 : more_than_one_fourth_colored (S n) color_count n) :
  (∃ x y z ∈ S n, x = y + z ∧ different_colors color x y z) :=
begin
  sorry,
end


end equation_solution_exists_l427_427573


namespace derivative_at_1_l427_427383

-- Definition of the function f
def f (x : ℝ) : ℝ := x^3 * sin x

-- Statement to prove that f'(1) = 3 * sin 1 + cos 1
theorem derivative_at_1 :
  deriv f 1 = 3 * sin 1 + cos 1 :=
sorry

end derivative_at_1_l427_427383


namespace pqrs_sum_l427_427516

theorem pqrs_sum (p q r s : ℝ)
  (h1 : (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 → x = r ∨ x = s))
  (h2 : (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 → x = p ∨ x = q))
  (h3 : p ≠ q) (h4 : p ≠ r) (h5 : p ≠ s) (h6 : q ≠ r) (h7 : q ≠ s) (h8 : r ≠ s) :
  p + q + r + s = 2028 :=
sorry

end pqrs_sum_l427_427516


namespace stream_current_l427_427268

noncomputable def solve_stream_current : Prop :=
  ∃ (r w : ℝ), (24 / (r + w) + 6 = 24 / (r - w)) ∧ (24 / (3 * r + w) + 2 = 24 / (3 * r - w)) ∧ (w = 2)

theorem stream_current : solve_stream_current :=
  sorry

end stream_current_l427_427268


namespace clock_hand_angle_7_to_8_l427_427603

noncomputable def initial_hour_angle : ℝ := 210
noncomputable def initial_minute_angle : ℝ := 0
noncomputable def relative_speed : ℝ := 330

theorem clock_hand_angle_7_to_8 : 
  (exists (t1 t2 : ℕ), t1 = 16 ∧ t2 = 60 ∧ 
    angle_at_time (initial_hour_angle, initial_minute_angle, relative_speed) t1 = 120 
    ∧ angle_at_time (initial_hour_angle, initial_minute_angle, relative_speed) t2 = 120) :=
  sorry

end clock_hand_angle_7_to_8_l427_427603


namespace museum_visit_orders_l427_427550

theorem museum_visit_orders : ∃ (orders : ℕ), orders = 5! := by
  existsi 120
  sorry

end museum_visit_orders_l427_427550


namespace sum_distinct_vars_eq_1716_l427_427506

open Real

theorem sum_distinct_vars_eq_1716 (p q r s : ℝ) (hpqrs_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : r + s = 12 * p)
  (h2 : r * s = -13 * q)
  (h3 : p + q = 12 * r)
  (h4 : p * q = -13 * s) :
  p + q + r + s = 1716 :=
sorry

end sum_distinct_vars_eq_1716_l427_427506


namespace hyperbola_focal_length_l427_427966

theorem hyperbola_focal_length : 
  let a_sq := 4
  let b_sq := 1
  let equation := ∀ x y : ℝ, ((y^2) / a_sq) - (x^2) = 1 in
  2 * Real.sqrt (a_sq + b_sq) = 2 * Real.sqrt 5 := by
  sorry

end hyperbola_focal_length_l427_427966


namespace james_tin_collection_l427_427890

theorem james_tin_collection :
  let total_tins := 500
  let first_day := 50
  let second_day := 3 * first_day
  let remaining_days := 4 * first_day
  ∃ T : ℕ, first_day + second_day + T + remaining_days = total_tins ∧ second_day - T = 50 :=
by
  let total_tins := 500
  let first_day := 50
  let second_day := 3 * first_day
  let remaining_days := 4 * first_day
  use 100
  sorry

end james_tin_collection_l427_427890


namespace median_of_list_l427_427218

def list1 := List.range 4080  -- Generating list [1, 2, 3, ... , 4080]
def list2 := list1.map (λ x => x ^ 2)  -- Generating list of squares [1^2, 2^2, ... , 4080^2]
def combined_list := list1 ++ list2  -- Concatenating both lists
def sorted_combined_list := combined_list.sort

theorem median_of_list : 
  let mid1 := sorted_combined_list.get (4080 - 1) in -- Get 4080-th term after sorting
  let mid2 := sorted_combined_list.get 4080 in -- Get 4081-st term after sorting
  (mid1 + mid2) / 2 = 4088 := 
by
  sorry

end median_of_list_l427_427218


namespace xiao_ming_math_score_l427_427636

noncomputable def math_score (C M E : ℕ) : ℕ :=
  let A := 94
  let N := 3
  let total_score := A * N
  let T_CE := (A - 1) * 2
  total_score - T_CE

theorem xiao_ming_math_score (C M E : ℕ)
    (h1 : (C + M + E) / 3 = 94)
    (h2 : (C + E) / 2 = 93) :
  math_score C M E = 96 := by
  sorry

end xiao_ming_math_score_l427_427636


namespace rental_difference_l427_427242

variable (C K : ℕ)

theorem rental_difference
  (hc : 15 * C + 18 * K = 405)
  (hr : 3 * K = 2 * C) :
  C - K = 5 :=
sorry

end rental_difference_l427_427242


namespace no_isosceles_triangle_of_distinct_elements_l427_427850

-- Definitions based on the conditions provided in a)

def distinct_elements (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Statement of the problem
theorem no_isosceles_triangle_of_distinct_elements (a b c : ℕ) 
  (h : distinct_elements a b c) : ¬ is_isosceles a b c := sorry

end no_isosceles_triangle_of_distinct_elements_l427_427850


namespace most_likely_outcomes_l427_427751

/-
Five children were born at City Hospital yesterday. Assume each child is equally likely to be a boy or a girl.
Which of the outcomes is most likely?
- (C) 3 are girls and 2 are boys
- (D) 4 are of one gender and 1 is of the other gender
-/

noncomputable def probability_five_children (outcome : List Nat) : ℚ :=
  let n := 5
  let p := 1 / 2
  let q := 1 / 2
  let k := outcome.count (fun x => x = 1) -- assuming 1 represents girl and 0 represents boy
  Nat.choose n k * p ^ k * q ^ (n - k)

theorem most_likely_outcomes :
  let outcomes := [
    ([1, 1, 1, 0, 0], (5 / 16)), -- 3 girls, 2 boys
    ([1, 1, 1, 1, 0], (5 / 16)), -- 4 girls, 1 boy
    ([0, 1, 1, 1, 1], (5 / 16))  -- 1 boy, 4 girls
  ]
  let prob := (3 girls, 2 boys)
  (∀ (prob ∈ outcomes), prob = 5 / 16) ∧
  (prob / outcomes) := 5 / 16 :=
sorry

end most_likely_outcomes_l427_427751


namespace simplify_expression_l427_427553

-- We define the given expressions and state the theorem.
variable (x : ℝ)

theorem simplify_expression : (3 * x - 4) * (x + 8) - (x + 6) * (3 * x - 2) = 4 * x - 20 := by
  -- Proof goes here
  sorry

end simplify_expression_l427_427553


namespace rectangle_in_bottom_rightmost_is_D_l427_427637

-- Define rectangles and their sides
structure Rectangle :=
  (w : Int)
  (x : Int)
  (y : Int)
  (z : Int)

def RectA : Rectangle := { w := 4, x := 2, y := 9, z := 5 }
def RectB : Rectangle := { w := 2, x := 1, y := 7, z := 4 }
def RectC : Rectangle := { w := 6, x := 8, y := 3, z := 9 }
def RectD : Rectangle := { w := 8, x := 6, y := 1, z := 3 }
def RectE : Rectangle := { w := 5, x := 3, y := 4, z := 7 }
def RectF : Rectangle := { w := 7, x := 0, y := 6, z := 2 }

-- The 2x3 grid alignment condition
def lower_w_top_left (rects : List Rectangle) : Prop :=
  ∀ i j (h_i : i < j) (_ : i < length rects) (_ : j < length rects),
    (rects[i].w < rects[j].w) ∨ (i/3 < j/3)

-- Define the grid
def grid := [RectB, RectC, RectA, RectE, RectF, RectD]

-- The bottom rightmost position in a 2x3 grid (index 5)
def bottom_rightmost (rects : List Rectangle) : Rectangle := rects[5]

-- Statement to prove
theorem rectangle_in_bottom_rightmost_is_D (rects : List Rectangle)
    (h_align : lower_w_top_left rects) : bottom_rightmost rects = RectD := by
  sorry

end rectangle_in_bottom_rightmost_is_D_l427_427637


namespace unique_solution_k_values_l427_427959

theorem unique_solution_k_values (k : ℝ) :
  (∃! x : ℝ, k * x ^ 2 - 3 * x + 2 = 0) ↔ (k = 0 ∨ k = 9 / 8) :=
by
  sorry

end unique_solution_k_values_l427_427959


namespace ab_product_l427_427029

theorem ab_product (a b : ℝ) (h_sol : ∀ x, -1 < x ∧ x < 4 → x^2 + a * x + b < 0) 
  (h_roots : ∀ x, x^2 + a * x + b = 0 ↔ x = -1 ∨ x = 4) : 
  a * b = 12 :=
sorry

end ab_product_l427_427029


namespace area_parabola_line_l427_427569

-- Define the parabola y^2 = 4x
def parabola (x : ℝ) : ℝ := real.sqrt (4 * x)

-- Define the line y = x - 3
def line (x : ℝ) : ℝ := x - 3

-- Define the integral function to compute the area
noncomputable def area_under_curve (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ y in a..b, f y

-- The statement to prove
theorem area_parabola_line :
  area_under_curve
    (λ y, y + 3 - (y^2 / 4))
    (-2) 6 = 64 / 3 := by
suspend
  sorry

end area_parabola_line_l427_427569


namespace intersection_of_A_and_B_l427_427488

def A : set ℝ := { x : ℝ | -1 < x ∧ x ≤ 4 }
def B : set ℕ := { x : ℕ | 2 ≤ x }

theorem intersection_of_A_and_B : A ∩ B = {2, 3, 4} :=
by
  sorry

end intersection_of_A_and_B_l427_427488


namespace degree_meas_supp_compl_35_l427_427201

noncomputable def degree_meas_supplement_complement (θ : ℝ) : ℝ :=
  180 - (90 - θ)

theorem degree_meas_supp_compl_35 : degree_meas_supplement_complement 35 = 125 :=
by
  unfold degree_meas_supplement_complement
  norm_num
  sorry

end degree_meas_supp_compl_35_l427_427201


namespace sum_abc_equals_33_l427_427168

theorem sum_abc_equals_33 (a b c N : ℕ) (h_neq : ∀ x y, x ≠ y → x ≠ y → x ≠ y → x ≠ y ) 
(hN1 : N = 5 * a + 3 * b + 5 * c) (hN2 : N = 4 * a + 5 * b + 4 * c)
(h_range : 131 < N ∧ N < 150) : a + b + c = 33 :=
sorry

end sum_abc_equals_33_l427_427168


namespace max_n_for_factored_polynomial_l427_427333

theorem max_n_for_factored_polynomial :
  ∃ (n : ℤ), (∀ (A B : ℤ), 6 * 72 + n = 6 * B + A ∧ A * B = 72 → n ≤ 433) ∧
             (∃ (A B : ℤ), 6 * B + A = 433 ∧ A * B = 72) :=
by sorry

end max_n_for_factored_polynomial_l427_427333


namespace SumOfFirstElevenTerms_l427_427871

def sum_of_first_eleven_terms (a_3 a_9 : ℝ) (c : ℝ) : Prop :=
  a_3 + a_9 = 16 ∧ c < 64 ∧ ∃ (a_1 d : ℝ), 
  (∀ n : ℕ, 0 < n → a_n = a_1 + (n - 1) * d) ∧
  (∑ i in finset.range 11, a_1 + i • d) = 88asible.

theorem SumOfFirstElevenTerms (a_3 a_9 c : ℝ) (h1 : a_3 + a_9 = 16) (h2 : c < 64) :
  ∃ (a_1 d : ℝ), 
    (∀ n : ℕ, 0 < n → ∃ m : ℕ, a_n = a_1 + m • d) ∧
    (∑ i in finset.range 11, a_1 + i • d) = 88 :=
by
  sorry

end SumOfFirstElevenTerms_l427_427871


namespace probability_three_non_red_purple_balls_l427_427251

def total_balls : ℕ := 150
def prob_white : ℝ := 0.15
def prob_green : ℝ := 0.20
def prob_yellow : ℝ := 0.30
def prob_red : ℝ := 0.30
def prob_purple : ℝ := 0.05
def prob_not_red_purple : ℝ := 1 - (prob_red + prob_purple)

theorem probability_three_non_red_purple_balls :
  (prob_not_red_purple * prob_not_red_purple * prob_not_red_purple) = 0.274625 :=
by
  sorry

end probability_three_non_red_purple_balls_l427_427251


namespace d_2_is_zero_l427_427070

noncomputable def d_4 : ℤ := 0
noncomputable def d_3 : ℤ := 1
noncomputable def d_2 : ℤ := 0
noncomputable def d_1 : ℤ := 0
noncomputable def d_0 : ℤ := 0

def p (x : ℤ) : ℤ := d_4 * x^4 + d_3 * x^3 + d_2 * x^2 + d_1 * x + d_0

theorem d_2_is_zero :
  ∀ m : ℤ, m ≥ 3 → E(m) = p(m) → d_2 = 0 :=
by
  intros m hm heq
  -- reasonable proof steps would follow here
  simp at heq
  sorry

end d_2_is_zero_l427_427070


namespace hyperbola_parabola_focus_directrix_l427_427813

theorem hyperbola_parabola_focus_directrix (p : ℝ) :
  let hyperbola_focus := (- real.sqrt (3 + p^2 / 16))
  let parabola_directrix := - p / 2
  (hyperbola_focus = parabola_directrix) → p = 4 :=
by
  intros h
  sorry

end hyperbola_parabola_focus_directrix_l427_427813


namespace Kenneth_money_left_l427_427897

def initial_amount : ℕ := 50
def number_of_baguettes : ℕ := 2
def cost_per_baguette : ℕ := 2
def number_of_bottles : ℕ := 2
def cost_per_bottle : ℕ := 1

-- This theorem states that Kenneth has $44 left after his purchases.
theorem Kenneth_money_left : initial_amount - (number_of_baguettes * cost_per_baguette + number_of_bottles * cost_per_bottle) = 44 := by
  sorry

end Kenneth_money_left_l427_427897


namespace question1_question2_l427_427967

-- Definitions of the given conditions
def cond1 (f : ℝ × ℝ → ℝ × ℝ) (a : ℝ × ℝ) (x : ℝ × ℝ) : Prop :=
  f x = (x.1 - 2 * (x.1 * a.1 + x.2 * a.2) * a.1, x.2 - 2 * (x.1 * a.1 + x.2 * a.2) * a.2)

def cond2 (f : ℝ × ℝ → ℝ × ℝ) : Prop :=
  ∀ (x : ℝ × ℝ), ∃ a : ℝ × ℝ, a ≠ (0, 0) ∧ f (f x) = f x

def cond3 : ℝ × ℝ := (1, -2)

def cond4 (P B A : ℝ × ℝ) : Prop :=
  P = (A.1 - (B.1 - A.1) / 3, A.2 - (B.2 - A.2) / 3)

-- Main theorem statements based on the given conditions
theorem question1 (f : ℝ × ℝ → ℝ × ℝ) (a : ℝ × ℝ) :
  cond1 f a ∧ cond2 f ∧ a ≠ (0, 0) → |a.1 ^ 2 + a.2 ^ 2 | = 1 / 2 := sorry

theorem question2 (f : ℝ × ℝ → ℝ × ℝ) (A B P : ℝ × ℝ) :
  cond1 f (B.1 - A.1, B.2 - A.2) ∧ cond3 = A ∧ cond4 P B A →
  (P.1 - 1) ^ 2 + (P.2 + 2) ^ 2 = 1 / 8 := sorry

end question1_question2_l427_427967


namespace range_of_a_l427_427919

-- Definitions of the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, ax^2 - x + (1 / 4) * a > 0

def q (a : ℝ) : Prop := ∀ x : ℝ, 0 < x → 3^x - 9^x < a

-- The theorem statement that needs to be proved
theorem range_of_a : ∃ a : set ℝ, p a → q a → ((p a ∧ ¬ q a) ∨ (¬ p a ∧ q a)) → a = [0, 1] := sorry

end range_of_a_l427_427919


namespace simplify_expression_l427_427558

theorem simplify_expression (x : ℝ) : 
  (3 * x - 4) * (x + 8) - (x + 6) * (3 * x - 2) = 4 * x - 20 := 
by
  sorry

end simplify_expression_l427_427558


namespace intersection_of_two_spheres_has_15_points_l427_427720

theorem intersection_of_two_spheres_has_15_points :
  let first_sphere := { p : ℤ × ℤ × ℤ | p.1^2 + p.2^2 + (p.3 - 5)^2 ≤ 16 }
  let second_sphere := { p : ℤ × ℤ × ℤ | p.1^2 + p.2^2 + p.3^2 ≤ 9 }
  (first_sphere ∩ second_sphere).finite ∧ (first_sphere ∩ second_sphere).to_finset.card = 15 := 
by
  sorry

end intersection_of_two_spheres_has_15_points_l427_427720


namespace y_coordinate_M_l427_427404

def ellipse (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

def focus : ℝ × ℝ := (Real.sqrt 3, 0)

def point_on_y_axis (m : ℝ) : ℝ × ℝ := (0, m)

def midpoint (P F M : ℝ × ℝ) : Prop := 
  M.1 = (F.1 + P.1) / 2 ∧ M.2 = (F.2 + P.2) / 2

theorem y_coordinate_M (m : ℝ) (x y : ℝ) 
  (hP : ellipse x y) 
  (hM : point_on_y_axis m) 
  (hMid : midpoint (x, y) focus (0, m)) : 
  m = 1 / 4 :=
sorry

end y_coordinate_M_l427_427404


namespace sum_periodic_terms_l427_427881

noncomputable def a : ℕ → ℝ
| 0       := real.sqrt 3
| (n + 1) := (1 + a n) / (1 - a n)

theorem sum_periodic_terms :
  (finset.range 503).sum (λ k, a (4 * k + 1)) = 503 * real.sqrt 3 :=
by
  sorry

end sum_periodic_terms_l427_427881


namespace BF_perpendicular_TC_l427_427492

variables (A B C D O T U F : Point) (AB CD : LineSegment)
variables [Rhombus ABCD] (O : Center ABCD) (T : Line)

-- Assume T is the foot of the perpendicular from the center O to side AB
axiom foot_center_to_AB (h1 : foot_perpendicular O AB T)

-- Assume U is the foot of the perpendicular from vertex D to side AB
axiom foot_D_to_AB (h2 : foot_perpendicular D AB U)

-- Assume F is the midpoint of segment DU
axiom midpoint_DU (h3 : midpoint D U F)

-- We need to prove that BF is perpendicular to TC
theorem BF_perpendicular_TC : Perpendicular (LineSegment.mk B F) (LineSegment.mk T C) :=
sorry

end BF_perpendicular_TC_l427_427492


namespace magnitude_of_complex_raised_to_fifth_power_l427_427711

def complex_number := Complex.mk 2 (-3 * Real.sqrt 2)

theorem magnitude_of_complex_raised_to_fifth_power :
  Complex.abs (complex_number ^ 5) = 22^(5/2) := by
  sorry

end magnitude_of_complex_raised_to_fifth_power_l427_427711


namespace supplement_of_complement_of_35_degree_angle_l427_427191

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_of_35_degree_angle : 
  supplement (complement 35) = 125 := 
by sorry

end supplement_of_complement_of_35_degree_angle_l427_427191


namespace total_clowns_in_mobiles_l427_427962

theorem total_clowns_in_mobiles (mobiles : ℕ) (clowns_per_mobile : ℕ) (h_mobiles : mobiles = 5) (h_clowns_per_mobile : clowns_per_mobile = 28) :
  mobiles * clowns_per_mobile = 140 :=
by
  rw [h_mobiles, h_clowns_per_mobile]
  norm_num

end total_clowns_in_mobiles_l427_427962


namespace centrally_symmetric_pattern_l427_427695

-- Definitions for the four patterns
def pattern_A := "A"
def pattern_B := "B"
def pattern_C := "C"
def pattern_D := "D"

-- Conditions
def not_centrally_symmetric (pattern : String) : Prop := 
  pattern = pattern_A ∨ 
  pattern = pattern_C ∨ 
  pattern = pattern_D

def is_centrally_symmetric (pattern : String) : Prop := 
  pattern = pattern_B

-- Theorem
theorem centrally_symmetric_pattern : 
  ∀ pattern, is_centrally_symmetric pattern ↔ pattern = pattern_B :=
by 
  intro pattern
  unfold is_centrally_symmetric
  split
  all_goals { intro h }
  { exact h }
  { exact h }

end centrally_symmetric_pattern_l427_427695


namespace range_of_x_inequality_l427_427082

theorem range_of_x_inequality (x : ℝ) (h : |2 * x - 1| + x + 3 ≤ 5) : -1 ≤ x ∧ x ≤ 1 :=
by
  sorry

end range_of_x_inequality_l427_427082


namespace g_g_x_eq_6_has_two_solutions_l427_427080

def g (x : ℝ) : ℝ :=
  if x ≤ 1 then -x + 2 else 2 * x - 4

theorem g_g_x_eq_6_has_two_solutions : ∃ x₁ x₂, (g (g x₁) = 6 ∧ g (g x₂) = 6) ∧ (x₁ ≠ x₂) :=
by
  sorry

end g_g_x_eq_6_has_two_solutions_l427_427080


namespace degree_measure_of_supplement_of_complement_of_35_degree_angle_l427_427198

def complement (α : ℝ) : ℝ := 90 - α
def supplement (β : ℝ) : ℝ := 180 - β

theorem degree_measure_of_supplement_of_complement_of_35_degree_angle : 
  supplement (complement 35) = 125 :=
by
  sorry

end degree_measure_of_supplement_of_complement_of_35_degree_angle_l427_427198


namespace opposite_of_neg_two_l427_427973

theorem opposite_of_neg_two : -(-2) = 2 :=
by
  sorry

end opposite_of_neg_two_l427_427973


namespace find_C_l427_427468

theorem find_C 
  (m n : ℝ)
  (C : ℝ)
  (h1 : m = 6 * n + C)
  (h2 : m + 2 = 6 * (n + 0.3333333333333333) + C) 
  : C = 0 := by
  sorry

end find_C_l427_427468


namespace max_red_socks_l427_427666

theorem max_red_socks (r b g t : ℕ) (h1 : t ≤ 2500) (h2 : r + b + g = t) 
  (h3 : (r * (r - 1) + b * (b - 1) + g * (g - 1)) = (2 / 3) * t * (t - 1)) : 
  r ≤ 1625 :=
by 
  sorry

end max_red_socks_l427_427666


namespace problem_statement_l427_427792

def rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b
def p : Prop := rational Real.pi
def q : Prop := {x : ℝ | x^2 - 3 * x + 2 < 0} = set.Ioo 1 2

theorem problem_statement : p = False ∧ q = True →
  (¬p ∧ q) ∧
  ¬(p ∧ ¬q) ∧
  ((¬p) ∨ q) ∧
  ¬((¬p) ∨ (¬q)) :=
by
  intros h
  obtain ⟨hp, hq⟩ := h
  rw [hp, hq]
  exact ⟨⟨(not_false), rfl⟩, not.intro (λ ⟨hp, hnq⟩, hp), (or.inr rfl), not.intro (λ ⟨hnp, hnq⟩, hnq)⟩

end problem_statement_l427_427792


namespace find_angle_between_vectors_l427_427826

variables (a b : EuclideanSpace ℝ (Fin 2))

-- Definitions based on given conditions
def vector_length (v : EuclideanSpace ℝ (Fin 2)) := reals.sqrt (v ⬝ v)


-- Conditions
axiom a_length : vector_length a = 1
axiom b_length : vector_length b = 6
axiom a_dot_b_minus_a : (a ⬝ (b - a)) = 2

-- Angle calculation
noncomputable def angle_between (u v : EuclideanSpace ℝ (Fin 2)) : ℝ :=
real.acos ((u ⬝ v) / (vector_length u * vector_length v))

-- Theorem
theorem find_angle_between_vectors :
  angle_between a b = π / 3 :=
sorry

end find_angle_between_vectors_l427_427826


namespace lines_distance_is_two_l427_427012

-- Definitions for lines l1 and l2
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def line2 (m x y : ℝ) : Prop := m * x + 2 * y + 1 + 2 * m = 0

-- Condition that the lines are parallel
def are_parallel (m : ℝ) : Prop := 3 / 4 = m / 2

-- Distance computation between two parallel lines
def distance_between_parallel_lines (m : ℝ) : ℝ :=
(abs (3 * 0 + 4 * 0 + 8)) / (real.sqrt (3 ^ 2 + 4 ^ 2))

theorem lines_distance_is_two (m : ℝ) (h : are_parallel m) : distance_between_parallel_lines m = 2 := by
  sorry

end lines_distance_is_two_l427_427012


namespace compute_fraction_at_five_l427_427301

theorem compute_fraction_at_five :
  let x := 5
  in (x^4 - 8 * x^2 + 16) / (x^2 - 4) = 21 :=
by
  -- Placeholder for the proof
  sorry

end compute_fraction_at_five_l427_427301


namespace youtube_dislikes_proof_l427_427651

/--
A YouTube video got 5000 likes and 150 more than triple the number of likes as dislikes.
If the video receives 2000 more likes and 400 more dislikes, and the new total likes is doubled before calculating the difference in dislikes,
prove that the video has 2017 dislikes after the increase.
-/
theorem youtube_dislikes_proof (D : ℕ) (h1 : 5000 - 3 * D = 150)
  (likes_increase : 2000) (dislikes_increase : 400) 
  (dbl : ℕ → ℕ := λ x, 2 * x) :
  D + dislikes_increase = 2017 :=
by
  change 5000 - 150 with 4850 in h1
  suffices : 3 * D = 4850
  exact_mod_cast nat.div_lt_self (lt_add_one_of_ne (by linarith)) (by simp)
  let new_likes := 5000 + likes_increase
  let new_dislikes := D + dislikes_increase
  have doubled_likes := dbl new_likes
  change doubled_likes - new_dislikes = 11983
  suffices D = 1617 from sorry
  have round_D := nat.ceil (4850 / 3 : ℝ) 
  exact_mod_cast suffices : round_D = 1617 from round_D

end youtube_dislikes_proof_l427_427651


namespace calculate_expression_l427_427295

theorem calculate_expression : 5 * 401 + 4 * 401 + 3 * 401 + 400 = 5212 := by
  sorry

end calculate_expression_l427_427295


namespace largest_n_factorable_l427_427337

theorem largest_n_factorable :
  ∃ n, (∀ A B : ℤ, 6x^2 + n • x + 72 = (6 • x + A) * (x + B)) ∧ 
    (∀ x', 6x' + A = 0 ∨ x' + B = 0) ∧ 
    n = 433 := 
sorry

end largest_n_factorable_l427_427337


namespace six_digit_numbers_count_l427_427830

-- Define the predicate for a six-digit number
def is_six_digit_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

-- Define the predicate for having the penultimate digit being 1
def penultimate_digit_is_1 (n : ℕ) : Prop :=
  (n / 10 % 10) = 1

-- Define the predicate for divisibility by 4
def divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

-- Define the count of such six-digit numbers
def count_six_digit_numbers_with_conditions : ℕ :=
  { n : ℕ // is_six_digit_number n ∧ penultimate_digit_is_1 n ∧ divisible_by_4 n }.tag

theorem six_digit_numbers_count :
  count_six_digit_numbers_with_conditions = 18000 :=
sorry

end six_digit_numbers_count_l427_427830


namespace line_equation_correct_l427_427961

-- Definitions given in the problem
def slope_angle : ℝ := 60
def x_intercept : ℝ := real.sqrt 3

-- Lean proof problem statement
theorem line_equation_correct : 
  ∃ (m : ℝ) (b : ℝ), m = real.tan (slope_angle * real.pi / 180) ∧
  b = -m * x_intercept ∧
  (∀ x y, y = m * x + b ↔ √3 * x - y - 3 = 0) :=
sorry

end line_equation_correct_l427_427961


namespace simple_interest_rate_l427_427289

-- Define all conditions as def and the target statement
def principal : ℕ := 12000
def amount : ℕ := 17500
def time : ℕ := 13

-- Define the rate calculation function
def rate (si p t : ℝ) : ℝ := (si * 100) / (p * t)

-- Calculate the simple interest
def simple_interest (a p : ℝ) : ℝ := a - p

theorem simple_interest_rate :
  let si := simple_interest (amount) (principal),
      r := rate si principal time in
  abs (r - 3.53) < 0.01 :=
by 
  rw [amount, principal, time]
  sorry

end simple_interest_rate_l427_427289


namespace minimize_square_sum_l427_427523

theorem minimize_square_sum (x1 x2 x3 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) 
  (h4 : x1 + 3 * x2 + 5 * x3 = 100) : 
  x1^2 + x2^2 + x3^2 ≥ 2000 / 7 :=
sorry

end minimize_square_sum_l427_427523


namespace sum_of_solutions_l427_427625

theorem sum_of_solutions (x : ℝ) (h : (3 * x / 15 = 6 / x)) : (∑ x in { x | 3 * x / 15 = 6 / x }, x) = 0 :=
begin
  sorry
end

end sum_of_solutions_l427_427625


namespace find_k_l427_427386

variable {a : ℕ → ℤ} -- Defining a_n as a function from natural numbers to integers
variable S : ℕ → ℤ   -- Defining S_n as a function from natural numbers to integers

-- Conditions
axiom S2018_positive : S 2018 > 0
axiom S2019_negative : S 2019 < 0
axiom abs_a_ge_abs_k : ∀ n : ℕ, |a n| ≥ |a 1010|

-- Objective
theorem find_k : 1010 = 1010 :=
by
  sorry -- Proof to be filled in

end find_k_l427_427386


namespace nonagon_coloring_l427_427604

-- Define the conditions: a regular nonagon where adjacent vertices and vertices forming an equilateral triangle are colored differently
variables (nonagon : Fin 9 → Prop) [is_regular_nonagon : is_regular_nonagon nonagon]
variables (adj_diff_colors : ∀ (i j : Fin 9), (i = j + 1) → nonagon i ≠ nonagon j)
variables (equi_tri_diff_colors : ∀ (i j k : Fin 9), (i = j + 3 ∧ j = k + 3) → nonagon i ≠ nonagon j ∧ nonagon j ≠ nonagon k ∧ nonagon i ≠ nonagon k)

-- Prove the product mn is 162
theorem nonagon_coloring : ∃ m n, (m = 3) → (n = 54) → (m * n = 162) :=
by
  existsi (3 : ℕ)
  existsi (54 : ℕ)
  intro h1 h2
  rw [h1, h2]
  exact rfl

end nonagon_coloring_l427_427604


namespace quadratic_decreasing_condition_l427_427847

-- Define the quadratic function
def quadratic_function (x m : ℝ) : ℝ := (x - m)^2 - 1

-- Conditions and the proof problem wrapped as a theorem statement
theorem quadratic_decreasing_condition (m : ℝ) :
  (∀ x : ℝ, x ≤ 3 → quadratically_decreasing x m) → m ≥ 3 :=
sorry

-- Helper function defining the decreasing condition
def quadratically_decreasing (x m : ℝ) : Prop :=
∀ y : ℝ, y < x → quadratic_function y m > quadratic_function x m

end quadratic_decreasing_condition_l427_427847


namespace vector_parallel_l427_427016

theorem vector_parallel (m : ℝ) : 
  let a := (m, 4) in
  let b := (1, -2) in
  a.1 * b.2 = a.2 * b.1 → m = -2 :=
by
  intro hz
  sorry

end vector_parallel_l427_427016


namespace similar_triangles_legs_sum_l427_427615

theorem similar_triangles_legs_sum (a b : ℕ) (h1 : a * b = 18) (h2 : a^2 + b^2 = 25) (bigger_area : ℕ) (smaller_area : ℕ) (hypotenuse : ℕ) 
  (h_similar : bigger_area = 225) 
  (h_smaller_area : smaller_area = 9) 
  (h_hypotenuse : hypotenuse = 5) 
  (h_non_3_4_5 : ¬ (a = 3 ∧ b = 4 ∨ a = 4 ∧ b = 3)) : 
  5 * (a + b) = 45 := 
by sorry

end similar_triangles_legs_sum_l427_427615


namespace concyclic_quadrilateral_l427_427072

open EuclideanGeometry

variable {A B C D E S : Point}

-- Definitions of conditions
variables (triangle_ABC : AcuteTriangle A B C)
variables (AB_eq_AC : dist A B = dist A C)
variables (angle_condition : ∠ A C D = ∠ C B D)
variables (midpoint_E : Midpoint E B D)
variables (circumcenter_S : Circumcenter S B C D)

-- To prove that A, E, S, and C are concyclic
theorem concyclic_quadrilateral
  (triangle_ABC : AcuteTriangle A B C)
  (AB_eq_AC : dist A B = dist A C)
  (angle_condition : ∠ A C D = ∠ C B D)
  (midpoint_E : Midpoint E B D)
  (circumcenter_S : Circumcenter S B C D):
  Concyclic A E S C :=
sorry

end concyclic_quadrilateral_l427_427072


namespace degree_measure_of_supplement_of_complement_of_35_degree_angle_l427_427193

def complement (α : ℝ) : ℝ := 90 - α
def supplement (β : ℝ) : ℝ := 180 - β

theorem degree_measure_of_supplement_of_complement_of_35_degree_angle : 
  supplement (complement 35) = 125 :=
by
  sorry

end degree_measure_of_supplement_of_complement_of_35_degree_angle_l427_427193


namespace supplement_of_complement_is_125_l427_427209

-- Definition of the initial angle
def initial_angle : ℝ := 35

-- Definition of the complement of the angle
def complement_angle (θ : ℝ) : ℝ := 90 - θ

-- Definition of the supplement of an angle
def supplement_angle (θ : ℝ) : ℝ := 180 - θ

-- Main theorem statement
theorem supplement_of_complement_is_125 : 
  supplement_angle (complement_angle initial_angle) = 125 := 
by
  sorry

end supplement_of_complement_is_125_l427_427209


namespace perfect_square_condition_l427_427765

-- Define the product of factorials from 1 to 2n
def factorial_product (n : ℕ) : ℕ :=
  (List.prod (List.map factorial (List.range (2 * n + 1))))

-- The main theorem statement
theorem perfect_square_condition (n : ℕ) : 
  (∃ k : ℕ, n = 4 * k * (k + 1) ∨ n = 2 * k^2 - 1) ↔
  (∃ m : ℕ, (factorial_product n) / ((n + 1)!) = m^2) := sorry

end perfect_square_condition_l427_427765


namespace count_r_lt_r_and_le_pow_l427_427820

def a : ℕ → ℕ
| 0     := 1
| (n+1) := if a n ≤ n then a n + n else a n - n

theorem count_r_lt_r_and_le_pow (h : ∀ n, a n = if a (n - 1) ≤ n - 1 then a (n - 1) + (n - 1) else a (n - 1) - (n - 1)) :
  let count := λ bound f, (∑ i in range (bound + 1), if f i then 1 else 0)
  in count (3^2017) (λ r, a r < r) = (3^2017 - 2019) / 2 :=
by
  sorry

end count_r_lt_r_and_le_pow_l427_427820


namespace ball_height_l427_427252

theorem ball_height (a : ℝ) (r : ℝ) (initial_height : ℝ) (target_height : ℝ) :
  a = 2000 → r = 1/2 → initial_height = 2000 → target_height = 0.5 →
  ∃ k : ℕ, 2000 * (1/2)^k < 0.5 ∧ k = 12 := by
  intros h_a h_r h_initial h_target
  use 12
  rw [← h_a, ← h_initial]
  rw [← h_target, ← h_r]
  sorry

end ball_height_l427_427252


namespace sixth_root_of_1642064901_l427_427294

theorem sixth_root_of_1642064901 :
  (1642064901 : ℕ) = (51 : ℕ) ^ 6 → real.sqrt (1642064901 : ℝ) ^ (1 / 6) = 51 :=
by
  intro h
  have h' : (1642064901 : ℝ) = (51 : ℝ) ^ 6 := by norm_cast; assumption
  rw [←real.rpow_nat_cast, h'] at *
  apply_real.sqrt rpow_nat_cast (by norm_num) -- ensures all conditions match
  exact rfl

end sixth_root_of_1642064901_l427_427294


namespace opposite_neg_two_l427_427979

theorem opposite_neg_two : -(-2) = 2 := by
  sorry

end opposite_neg_two_l427_427979


namespace brother_d_payment_l427_427460

theorem brother_d_payment :
  ∃ d : ℕ, 
    let a₁ := 300 in
    let a₂ := a₁ + d in
    let a₃ := a₁ + 2 * d in
    let a₄ := a₁ + 3 * d in
    let a₅ := a₁ + 4 * d in
    (a₁ + a₂ + a₃ + a₄ + a₅ = 1000) ∧ (a₄ = 450) :=
by {
  sorry
}

end brother_d_payment_l427_427460


namespace largest_n_factorable_l427_427335

theorem largest_n_factorable :
  ∃ n, (∀ A B : ℤ, 6x^2 + n • x + 72 = (6 • x + A) * (x + B)) ∧ 
    (∀ x', 6x' + A = 0 ∨ x' + B = 0) ∧ 
    n = 433 := 
sorry

end largest_n_factorable_l427_427335


namespace quadratic_solution_range_l427_427597

theorem quadratic_solution_range {x : ℝ} 
  (h : x^2 - 6 * x + 8 < 0) : 
  25 < x^2 + 6 * x + 9 ∧ x^2 + 6 * x + 9 < 49 :=
sorry

end quadratic_solution_range_l427_427597


namespace min_triangles_bound_four_color_edge_coloring_exists_l427_427786

noncomputable theory

def P : Set Point := {P_1, P_2, ..., P_1994}

structure PointsGroups :=
  (groups : List (List Point))
  (group_sizes : ∀ g ∈ groups, 3 ≤ g.length)
  (partition_sum : groups.foldr (λ g acc, g.length + acc) 0 = 1994)
  (partition_unique : (∀ p1 p2 ∈ P, (∃ g, g ∈ groups ∧ p1 ∈ g ∧ p2 ∈ g) ⊕ (∀ g, g ∈ groups → p1 ∈ g → p2 ∉ g)))

def graph (pg : PointsGroups) : Graph Point :=
{ V := P,
  E := { (p1, p2) | ∃ g ∈ pg.groups, p1 ∈ g ∧ p2 ∈ g ∧ p1 ≠ p2 } }

def min_triangles (G : Graph Point) : ℕ :=
∑ g in G.V.groups, if g.length ≥ 3 then (g.length.choose 3) else 0

theorem min_triangles_bound (P : Set Point) (pg : PointsGroups) :
  ∃ G : Graph Point, min_triangles (graph pg) = 168544 := sorry

theorem four_color_edge_coloring_exists (P : Set Point) (pg : PointsGroups) (G : Graph Point)
  (hG : min_triangles (graph pg) = 168544) :
  ∃ (col : G.E → Fin 4), ∀ (p1 p2 p3 : Point), 
    (G.adj p1 p2 ∧ G.adj p2 p3 ∧ G.adj p1 p3) → (col (p1, p2) ≠ col (p2, p3) ∨ col (p2, p3) ≠ col (p1, p3) ∨ col (p1, p2) ≠ col (p1, p3)) := sorry

end min_triangles_bound_four_color_edge_coloring_exists_l427_427786


namespace angle_B_degree_l427_427040

variable (A B C D : Type)
variable [IsParallelogram A B C D]

variables (angle_A angle_B angle_C angle_D : ℝ)
variables (h1 : angle_A + angle_C = 100)
variables (h2 : angle_A = angle_C)
variables (h3 : angle_A + angle_D = 180)
variables (h4 : angle_B + angle_A = 180)

theorem angle_B_degree :
  angle_B = 130 :=
by
  sorry

end angle_B_degree_l427_427040


namespace sum_of_digits_of_5_pow_23_l427_427629

theorem sum_of_digits_of_5_pow_23 : 
  let n := 23 
  let base := 2 + 3
  let tens_digit : ℕ := 2
  let ones_digit : ℕ := 5
  tens_digit + ones_digit = 7 := 
by 
  let n := 23 
  let base := 2 + 3
  let tens_digit : ℕ := 2
  let ones_digit : ℕ := 5
  exact 2 + 5 = 7

end sum_of_digits_of_5_pow_23_l427_427629


namespace triangle_ABC_area_l427_427866
open Classical

-- Definitions of the problem conditions
variables (A B C D E F : Point)
variables (AB AD : Line)
variables [parallelogram ABCD : Parallelogram A B C D] -- Parallelogram ABCD
variables [point_on_line E AB]  -- Point E on side AB
variables (h1 : length(A, E) / length(E, B) = 1 / 2)  -- Ratio AE:EB = 1:2
variables [inter h2 : intersects DE AC F] -- Line DE intersects AC at F
variables (h3 : area (triangle A E F) = 6) -- Area of ΔAEF is 6 cm²

-- The statement we need to prove
theorem triangle_ABC_area : area (triangle A B C) = 36 := 
sorry

end triangle_ABC_area_l427_427866


namespace opposite_of_neg_two_l427_427976

theorem opposite_of_neg_two : -(-2) = 2 :=
by
  sorry

end opposite_of_neg_two_l427_427976


namespace find_contaminated_constant_l427_427630

theorem find_contaminated_constant (contaminated_constant : ℝ) (x : ℝ) 
  (h1 : 2 * (x - 3) - contaminated_constant = x + 1) 
  (h2 : x = 9) : contaminated_constant = 2 :=
  sorry

end find_contaminated_constant_l427_427630


namespace tangent_line_parallel_x_axis_l427_427326

noncomputable def curve : ℝ → ℝ := λ x, 2 * x^3 - 6 * x

noncomputable def derivative (x : ℝ) : ℝ :=
  (deriv curve) x

theorem tangent_line_parallel_x_axis (x y : ℝ) (h : curve x = y) : 
  (derivative x = 0) ↔ (x = -1 ∧ y = 4) ∨ (x = 1 ∧ y = -4) :=
by
  -- Proof will go here
  sorry

end tangent_line_parallel_x_axis_l427_427326


namespace num_ordered_triplets_abc_eq_2008_l427_427359

theorem num_ordered_triplets_abc_eq_2008 : 
  ∃ (n : ℕ), n = 30 ∧ ∀ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 2008 → 
  ∃ (a_2 b_2 c_2 a_251 b_251 c_251 : ℕ),
  a = 2^a_2 * 251^a_251 ∧ b = 2^b_2 * 251^b_251 ∧ c = 2^c_2 * 251^c_251 ∧
  a_2 + b_2 + c_2 = 3 ∧ a_251 + b_251 + c_251 = 1 ∧
  ∏ x in {5, 2}, choose 5 2 = 10 ∧ ∏ x in {3, 2}, choose 3 2 = 3 ∧ 
  n = 10 * 3 :=
begin
  sorry
end

end num_ordered_triplets_abc_eq_2008_l427_427359


namespace contradiction_assumption_l427_427688

variable (f : ℝ → ℝ)
variable (x1 x2 : ℝ)
variable (h0 : ∀ x ∈ set.Icc 0 1, f(x) = f(0) → f(1))
variable (h1 : ∀ {x1 x2 : ℝ}, x1 ≠ x2 → (x1 ∈ set.Icc 0 1) → (x2 ∈ set.Icc 0 1) → |f x1 - f x2| < |x1 - x2|)

theorem contradiction_assumption :
  ∃ (x1 x2 : ℝ), (x1 ∈ set.Icc 0 1) ∧ (x2 ∈ set.Icc 0 1) ∧ (h : |f x1 - f x2| < |x1 - x2|), |f x1 - f x2| ≥ |x1 - x2| := 
sorry

end contradiction_assumption_l427_427688


namespace find_number_l427_427269

theorem find_number (x : ℝ) : 
  ( ((x - 1.9) * 1.5 + 32) / 2.5 = 20 ) → x = 13.9 :=
by
  sorry

end find_number_l427_427269


namespace dartboard_problem_l427_427705

theorem dartboard_problem
  (darts : ℕ) 
  (dartboards : ℕ)
  (distr_counts : list (list ℕ))
  (all_distributions : darts = 5 ∧ dartboards = 4) :
  ∃ lists : list (list ℕ),
    (∀ l ∈ lists, list.length l = 4 ∧ list.sum l = 5 ∧ l.sorted (>=) = l) ∧
    lists.length = 6 :=
by
  sorry

end dartboard_problem_l427_427705


namespace sixth_power_sum_l427_427802

/-- Given:
     (1) a + b = 1
     (2) a^2 + b^2 = 3
     (3) a^3 + b^3 = 4
     (4) a^4 + b^4 = 7
     (5) a^5 + b^5 = 11
    Prove:
     a^6 + b^6 = 18 -/
theorem sixth_power_sum (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^6 + b^6 = 18 :=
sorry

end sixth_power_sum_l427_427802


namespace min_tiles_needed_l427_427272

theorem min_tiles_needed : 
  ∀ (tile_length tile_width region_length region_width: ℕ),
  tile_length = 5 → 
  tile_width = 6 → 
  region_length = 3 * 12 → 
  region_width = 4 * 12 → 
  (region_length * region_width) / (tile_length * tile_width) ≤ 58 :=
by
  intros tile_length tile_width region_length region_width h_tile_length h_tile_width h_region_length h_region_width
  sorry

end min_tiles_needed_l427_427272


namespace locus_of_points_is_circle_l427_427940

variables {A B C M : Type*}

/-- Conditions to define an Isosceles triangle --/
axiom isosceles_triangle (A B C : Type*) (AB AC : ℝ): AB = AC

/-- Condition for the point M such that product of distances equals square of distance --/
axiom point_M_condition (A B C M : Type*) (d_AB d_AC d_BC : ℝ):
  d_AB * d_AC = d_BC^2

/--
  Prove that the locus of points M satisfying the given condition
  in an isosceles triangle is a circle.
--/
theorem locus_of_points_is_circle (A B C M : Type*) (AB AC d_AB d_AC d_BC : ℝ) :
  isosceles_triangle A B C AB AC →
  point_M_condition A B C M d_AB d_AC d_BC →
  ∃ (O : Type*) (r : ℝ), 
  ∀ (M : Type*), point_M_condition A B C M d_AB d_AC d_BC → 
  dist O M = r :=
by
  sorry

end locus_of_points_is_circle_l427_427940


namespace problem_statement_l427_427385

theorem problem_statement (x : ℝ) (h : x^2 - 3 * x + 1 = 0) : x^2 + 1 / x^2 = 7 :=
sorry

end problem_statement_l427_427385


namespace find_extra_lives_first_level_l427_427033

-- Conditions as definitions
def initial_lives : ℕ := 2
def extra_lives_second_level : ℕ := 11
def total_lives_after_second_level : ℕ := 19

-- Definition representing the extra lives in the first level
def extra_lives_first_level (x : ℕ) : Prop :=
  initial_lives + x + extra_lives_second_level = total_lives_after_second_level

-- The theorem we need to prove
theorem find_extra_lives_first_level : ∃ x : ℕ, extra_lives_first_level x ∧ x = 6 :=
by
  sorry  -- Placeholder for the proof

end find_extra_lives_first_level_l427_427033


namespace sum_of_integers_l427_427130

theorem sum_of_integers (x y : ℤ) (h_pos : 0 < y) (h_gt : x > y) (h_diff : x - y = 14) (h_prod : x * y = 48) : x + y = 20 :=
sorry

end sum_of_integers_l427_427130


namespace determine_k_l427_427907

-- Define the conditions given in the problem
def x₀ := {x : ℝ | 8 - x = log x}

theorem determine_k : ∀ k : ℤ, (∃ x : ℝ, x ∈ x₀ ∧ x ∈ (k : ℝ), (k + 1)) → k = 7 := by
  sorry

end determine_k_l427_427907


namespace students_registered_for_three_classes_l427_427449

theorem students_registered_for_three_classes(
  (total_members : fin 100) (history_students : fin 100) 
  (math_students : fin 100) (english_students : fin 100) 
  (students_exactly_two_classes : fin 50) :
  total_members = 68 →
  history_students = 20 - 1 →
  math_students = 20 - 6 →
  english_students = 30 - 4 →
  students_exactly_two_classes = 7 →
  ∃ (students_three_classes : ℕ), students_three_classes =  8 :=
begin
  intros h_total_members h_history_students h_math_students h_english_students h_two_classes,
  unfold fin.val at *,
  use 8,
  sorry,
end

end students_registered_for_three_classes_l427_427449


namespace ratio_Smax_Smin_l427_427123

-- Define the area of a cube's diagonal cross-section through BD1
def cross_section_area (a : ℝ) : ℝ := sorry

theorem ratio_Smax_Smin (a : ℝ) (S S_min S_max : ℝ) :
  cross_section_area a = S →
  S_min = (a^2 * Real.sqrt 6) / 2 →
  S_max = a^2 * Real.sqrt 6 →
  S_max / S_min = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end ratio_Smax_Smin_l427_427123


namespace sum_equality_l427_427064

noncomputable def greatest_integer_function (x : ℝ) : ℤ := floor x

noncomputable def fractional_part (x : ℝ) : ℝ := x - floor x

-- Define the set of x such that given conditions hold
def satisfies_condition (x : ℝ) : Prop :=
  sqrt (greatest_integer_function x * greatest_integer_function (x ^ 3)) +
  sqrt (fractional_part x * fractional_part (x ^ 3)) = x ^ 2

-- Define the sequence x_i
noncomputable def sequence_x (i : ℕ) : ℝ :=
  if h : ∃ y, y ≥ 1 ∧ satisfies_condition y then Classical.choose h else 0

-- Define the sum to calculate
noncomputable def required_sum : ℝ :=
  ∑ k in Finset.range 50, 1 / (sequence_x (2 * k + 2) ^ 2 - sequence_x (2 * k + 1) ^ 2)

theorem sum_equality : required_sum = 1275 := sorry

end sum_equality_l427_427064


namespace pqrs_sum_l427_427511

theorem pqrs_sum (p q r s : ℝ)
  (h1 : (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 → x = r ∨ x = s))
  (h2 : (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 → x = p ∨ x = q))
  (h3 : p ≠ q) (h4 : p ≠ r) (h5 : p ≠ s) (h6 : q ≠ r) (h7 : q ≠ s) (h8 : r ≠ s) :
  p + q + r + s = 2028 :=
sorry

end pqrs_sum_l427_427511


namespace solve_system_of_equations_l427_427948

theorem solve_system_of_equations (x y z : ℝ) : 
  (y * z = 3 * y + 2 * z - 8) ∧
  (z * x = 4 * z + 3 * x - 8) ∧
  (x * y = 2 * x + y - 1) ↔ 
  ((x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 5 / 2 ∧ z = -1)) :=
by
  sorry

end solve_system_of_equations_l427_427948


namespace max_handshakes_25_people_l427_427249

-- Define the number of people attending the conference.
def num_people : ℕ := 25

-- Define the combinatorial formula to calculate the maximum number of handshakes.
def max_handshakes (n : ℕ) : ℕ := n.choose 2

-- State the theorem that we need to prove.
theorem max_handshakes_25_people : max_handshakes num_people = 300 :=
by
  -- Proof will be filled in later
  sorry

end max_handshakes_25_people_l427_427249


namespace simplify_expression_l427_427945

def a : ℚ := (3 / 4) * 60
def b : ℚ := (8 / 5) * 60
def c : ℚ := 63

theorem simplify_expression : a - b + c = 12 := by
  sorry

end simplify_expression_l427_427945


namespace part_a_part_b_part_c_l427_427304

noncomputable def a : ℕ → ℚ
| 0 := 3 / 2
| (n + 1) := (3 * (a n)^2 + 4 * a n - 3) / (4 * (a n)^2)

theorem part_a (n : ℕ) : 1 < a n ∧ a (n + 1) < a n :=
sorry

theorem part_b : ∃ l : ℚ, (∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - l| < ε) ∧ l = 1 :=
sorry

theorem part_c : ∃ l : ℚ, (∀ ε > 0, ∃ N, ∀ n ≥ N, ∏ i in finset.range n, a i - l < ε) ∧ l = 1 :=
sorry

end part_a_part_b_part_c_l427_427304


namespace supplement_of_complement_of_35_degree_angle_l427_427189

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_of_35_degree_angle : 
  supplement (complement 35) = 125 := 
by sorry

end supplement_of_complement_of_35_degree_angle_l427_427189


namespace correct_statements_truth_of_statements_l427_427311

-- Define basic properties related to factor and divisor
def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k
def is_divisor (d n : ℕ) : Prop := ∃ k : ℕ, n = d * k

-- Given conditions as definitions
def condition_A : Prop := is_factor 4 100
def condition_B1 : Prop := is_divisor 19 133
def condition_B2 : Prop := ¬ is_divisor 19 51
def condition_C1 : Prop := is_divisor 30 90
def condition_C2 : Prop := ¬ is_divisor 30 53
def condition_D1 : Prop := is_divisor 7 21
def condition_D2 : Prop := ¬ is_divisor 7 49
def condition_E : Prop := is_factor 10 200

-- Statement that needs to be proved
theorem correct_statements : 
  (condition_A ∧ 
  (condition_B1 ∧ condition_B2) ∧ 
  condition_E) :=
by sorry -- proof to be inserted

-- Equivalent Lean 4 statement with all conditions encapsulated
theorem truth_of_statements :
  (is_factor 4 100) ∧ 
  (is_divisor 19 133 ∧ ¬ is_divisor 19 51) ∧ 
  is_factor 10 200 :=
by sorry -- proof to be inserted

end correct_statements_truth_of_statements_l427_427311


namespace least_subtract_divisible_by_8_l427_427355

def least_subtracted_to_divisible_by (n : ℕ) (d : ℕ) : ℕ :=
  n % d

theorem least_subtract_divisible_by_8 (n : ℕ) (d : ℕ) (h : n = 964807) (h_d : d = 8) :
  least_subtracted_to_divisible_by n d = 7 :=
by
  sorry

end least_subtract_divisible_by_8_l427_427355


namespace find_d_l427_427917

noncomputable theory

def b : ℕ → ℝ 
| 0 := 7 / 25
| (n + 1) := 2 * (b n)^2 - 1

def abs_prod_b (n : ℕ) : ℝ := 
(list.prod (list.map b (list.range n))).abs

theorem find_d (d : ℝ) : 
(∀ n : ℕ, abs_prod_b n ≤ d / (3^n)) → 100 * d = 108 :=
begin
  sorry
end

end find_d_l427_427917


namespace negative_number_is_A_l427_427281

theorem negative_number_is_A :
  ∃ (x : ℤ), (x = -6 ∧ x < 0) ∧ (∀ (y : ℤ), (y = 0 ∨ y = 2 / 10 ∨ y = 3) → y ≥ 0) :=
begin
  sorry
end

end negative_number_is_A_l427_427281


namespace sin_angle_DAO_l427_427703

noncomputable def point := ℝ × ℝ
def origin : point := (0, 0)
def P : point := (3, 4)
def circle_eq (p : point) : Prop := p.1^2 + p.2^2 = 25
def y_axis (p : point) : Prop := p.1 = 0
def isosceles_triangle (P E F : point) : Prop := E.2 = -F.2
def intersection_with_circle (P E : point) : point := sorry -- Defined through intersection calculation
def line_through (p1 p2 : point) (x : ℝ) : ℝ := sorry -- y-value of the line through points p1 and p2 at x
def intersection_with_y_axis (C D : point) : point := sorry -- Calculation where line CD intersects y-axis

theorem sin_angle_DAO :
  circle_eq P ∧ ∃ E F, y_axis E ∧ y_axis F ∧ isosceles_triangle P E F ∧
  let D := intersection_with_circle P E in
  let C := intersection_with_circle P F in
  let A := intersection_with_y_axis C D in
  ∃ O, O = origin ∧ sin (angle D A O) = 4 / 5 := sorry

end sin_angle_DAO_l427_427703


namespace radius_of_cone_base_l427_427683

theorem radius_of_cone_base (θ : ℝ) (R : ℝ) (r : ℝ) 
  (hθ : θ = 90) 
  (hR : R = 20) : 
  (2 * Real.pi * r = θ / 360 * 2 * Real.pi * R) → r = 5 :=
by {
  intro h,
  sorry,
}

end radius_of_cone_base_l427_427683


namespace sine_cosine_relationship_sine_cosine_not_neccesary_sine_cosine_suff_not_necc_l427_427223

theorem sine_cosine_relationship (α : ℝ) :
  (sin α + cos α = 1) → (sin (2 * α) = 0) := 
by 
  -- This is where the detailed proof would go
  sorry

theorem sine_cosine_not_neccesary (α : ℝ) :
  (sin (2 * α) = 0) → (sin α + cos α = 1 ∨ sin α + cos α = -1) := 
by 
  -- This is where the detailed proof would go
  sorry

theorem sine_cosine_suff_not_necc (α : ℝ) :
  (sin α + cos α = 1) → (sin (2 * α) = 0) ∧ ¬(sin (2 * α) = 0 → sin α + cos α = 1) := 
by 
  -- Combine the two theorems above and add the logic to show it is sufficient but not necessary
  sorry

end sine_cosine_relationship_sine_cosine_not_neccesary_sine_cosine_suff_not_necc_l427_427223


namespace valid_range_for_a_l427_427411

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then (1 / 2) ^ x else (1 / e) ^ x

theorem valid_range_for_a (a : ℝ) :
  (∀ x : ℝ, 1 - 2 * a ≤ x ∧ x ≤ 1 + 2 * a → f (2 * x + a) ≥ (f x) ^ 3) →
  0 < a ∧ a ≤ 1 / 3 :=
by
  sorry

end valid_range_for_a_l427_427411


namespace eric_pencils_l427_427319

theorem eric_pencils (num_boxes : ℕ) (pencils_per_box : ℕ) (h1 : num_boxes = 3) (h2 : pencils_per_box = 9) : num_boxes * pencils_per_box = 27 := by
  rw [h1, h2]
  norm_num

end eric_pencils_l427_427319


namespace sequence_recurrence_l427_427532

theorem sequence_recurrence (a : ℕ → ℝ) (h₁ : a 1 = 1) (h₂ : a 2 = 2) (h₃ : ∀ n, n ≥ 1 → a (n + 2) / a n = (a (n + 1) ^ 2 + 1) / (a n ^ 2 + 1)):
  (∀ n, a (n + 1) = a n + 1 / a n) ∧ 63 < a 2008 ∧ a 2008 < 78 :=
by
  sorry

end sequence_recurrence_l427_427532


namespace opposite_of_neg_two_l427_427974

theorem opposite_of_neg_two : -(-2) = 2 :=
by
  sorry

end opposite_of_neg_two_l427_427974


namespace sum_of_distinct_roots_l427_427518

theorem sum_of_distinct_roots 
  (p q r s : ℝ)
  (h1 : p ≠ q)
  (h2 : p ≠ r)
  (h3 : p ≠ s)
  (h4 : q ≠ r)
  (h5 : q ≠ s)
  (h6 : r ≠ s)
  (h_roots1 : (x : ℝ) -> x^2 - 12*p*x - 13*q = 0 -> x = r ∨ x = s)
  (h_roots2 : (x : ℝ) -> x^2 - 12*r*x - 13*s = 0 -> x = p ∨ x = q) : 
  p + q + r + s = 1716 := 
by 
  sorry

end sum_of_distinct_roots_l427_427518


namespace time_per_accessory_l427_427480

theorem time_per_accessory (dolls : ℕ) (shoes_per_doll bags_per_doll cosmetics_per_doll hats_per_doll : ℕ) 
  (time_per_doll total_combined_time : ℕ) 
  (h_dolls : dolls = 12000)
  (h_shoes : shoes_per_doll = 2)
  (h_bags : bags_per_doll = 3)
  (h_cosmetics : cosmetics_per_doll = 1)
  (h_hats : hats_per_doll = 5)
  (h_time_per_doll : time_per_doll = 45)
  (h_total_combined_time : total_combined_time = 1860000) :
  (time_per_accessory = 10) := sorry

end time_per_accessory_l427_427480


namespace greatest_3_digit_base8_divisible_by_7_l427_427217

theorem greatest_3_digit_base8_divisible_by_7 :
  ∃ n : ℕ, (n < 8^3) ∧ (n ≥ 8^2) ∧ (n % 7 = 0) ∧ int_to_base8(n) = "777" := 
sorry

noncomputable def int_to_base8 (n : ℕ) : string :=
-- assumes n is a valid base 8 integer and returns its base 8 string representation
sorry

end greatest_3_digit_base8_divisible_by_7_l427_427217


namespace decimal_equivalent_one_half_pow_five_l427_427619

theorem decimal_equivalent_one_half_pow_five :
  (1 / 2) ^ 5 = 0.03125 :=
by sorry

end decimal_equivalent_one_half_pow_five_l427_427619


namespace sum_distinct_vars_eq_1716_l427_427508

open Real

theorem sum_distinct_vars_eq_1716 (p q r s : ℝ) (hpqrs_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : r + s = 12 * p)
  (h2 : r * s = -13 * q)
  (h3 : p + q = 12 * r)
  (h4 : p * q = -13 * s) :
  p + q + r + s = 1716 :=
sorry

end sum_distinct_vars_eq_1716_l427_427508


namespace find_f_2_l427_427413

def f (a b x : ℝ) := a * x^3 - b * x + 1

theorem find_f_2 (a b : ℝ) (h : f a b (-2) = -1) : f a b 2 = 3 :=
by
  sorry

end find_f_2_l427_427413


namespace f_at_3_f_expr_in_2_to_4_l427_427083

noncomputable def f (x : ℝ) : ℝ := 
  if 0 ≤ x ∧ x ≤ 2 then (x - 1) ^ 2
  else f (x - 2)

theorem f_at_3 : f 3 = 0 := sorry

theorem f_expr_in_2_to_4 {x : ℝ} (hx : 2 ≤ x ∧ x ≤ 4) : f x = (x - 3) ^ 2 := sorry

end f_at_3_f_expr_in_2_to_4_l427_427083


namespace collinear_points_xy_l427_427791

theorem collinear_points_xy (x y : ℝ) 
  (A : ℝ × ℝ × ℝ := (1, -2, 11))
  (B : ℝ × ℝ × ℝ := (4, 2, 3))
  (C : ℝ × ℝ × ℝ := (x, y, 15))
  (h_collinear : C.1 - A.1 = (B.1 - A.1) * (C.2 - A.2) / (B.2 - A.2) 
    ∧ C.1 - A.1 = (B.1 - A.1) * (C.3 - A.3) / (B.3 - A.3))
  : x * y = 2 := 
by
  sorry

end collinear_points_xy_l427_427791


namespace count_valid_n_count_valid_n_le_500_l427_427756

theorem count_valid_n (n : ℕ) : 
  (∃ k:ℕ, n = 4 * k + 1 ∧ n ≤ 500) ↔ (0 ≤ n - 1) ∧ (n - 1) % 4 = 0 ∧ n ≤ 500 :=
sorry

theorem count_valid_n_le_500 : ∃ nums : finset ℕ, 
  (∀ n ∈ nums, ∃ k:ℕ, n = 4 * k + 1 ∧ n ≤ 500) ∧ 
  nums.card = 125 :=
sorry

end count_valid_n_count_valid_n_le_500_l427_427756


namespace maximum_value_y_eq_x_plus_cosx_in_0_pi_over_2_l427_427357

open Real

theorem maximum_value_y_eq_x_plus_cosx_in_0_pi_over_2 :
  ∃ (x : ℝ), x ∈ Icc 0 (π / 2) ∧ (∀ y ∈ Icc 0 (π / 2), y + cos y ≤ x + cos x) ∧ x = π / 2 := 
sorry

end maximum_value_y_eq_x_plus_cosx_in_0_pi_over_2_l427_427357


namespace tan_double_angle_l427_427043

variable (tan α : ℝ)

def is_symmetric_point (O A : ℝ × ℝ) (l : ℝ → ℝ → ℝ) : Prop :=
  ∃ (α : ℝ), l O.1 O.2 = 0 ∧ l A.1 A.2 = 0 ∧
  (A.1 = (O.1 + A.1) / 2) ∧ (A.2 = (O.2 + A.2) / 2)

def line_eq (α : ℝ) := λ x y, 2 * x * (Real.tan α) + y - 1

noncomputable def α_val := Real.arctan (1/2)

theorem tan_double_angle (h : is_symmetric_point (0, 0) (1, 1) (line_eq α)) :
  Real.tan (2 * α_val) = 4 / 3 :=
by sorry

end tan_double_angle_l427_427043


namespace remainder_13_plus_y_l427_427526

theorem remainder_13_plus_y :
  (∃ y : ℕ, (0 < y) ∧ (7 * y ≡ 1 [MOD 31])) → (∃ y : ℕ, (13 + y ≡ 22 [MOD 31])) :=
by 
  sorry

end remainder_13_plus_y_l427_427526


namespace f_value_l427_427439

noncomputable def f (y : ℝ) : ℝ := 
  if h : ∃ x : ℝ, (0 < x ∧ x < π) ∧ sin 2 * x = y then
    let x := classical.some h in 5 * sin x - 5 * cos x - 6
  else 
    0  -- f is only defined for values that y = sin 2x for x in (0, π)

theorem f_value : f (-24 / 25) = 1 := 
  by
    sorry

end f_value_l427_427439


namespace certain_number_minus_star_value_l427_427366

-- Definition of the greatest positive even integer less than or equal to a number
def even_part (z : ℝ) : ℝ := 
  if z < 0 then 0 else 2 * int_floor (z / 2)
where 
  int_floor (x : ℝ) : ℕ := 
    if x % 1 = 0 then x.toNat else x.toNat - 1

theorem certain_number_minus_star_value (y : ℝ) :
  y - even_part y = 0.4500000000000002 :=
sorry

end certain_number_minus_star_value_l427_427366


namespace number_of_small_triangles_is_odd_l427_427105

theorem number_of_small_triangles_is_odd
  (V : Type) [Fintype V] [DecidableEq V]
  (T : SimpleGraph V) (triangle : V → V → V → Prop)
  (vertices : set V)
  (finite_points : set V)
  (conditions : ∀ (v₁ v₂ v₃ : V), v₁ ∈ vertices → v₂ ∈ vertices → v₃ ∈ vertices →
    triangle v₁ v₂ v₃ →
    ∀ (p₁ p₂ : V), p₁ ∈ finite_points → p₂ ∈ finite_points →
    ¬ (∃ (x : V), T.adj p₁ x ∧ T.adj p₂ x)) :
    ∃ n : ℕ, (∃ (e : ℕ), e = (3 * n - 3) / 2) ∧ Odd n :=
sorry

end number_of_small_triangles_is_odd_l427_427105


namespace probability_none_given_not_D_l427_427452

variable {Ω : Type} [Fintype Ω] [DecidableEq Ω]

-- Define the probabilities for various combinations of risk factors
def P_single (P_single_val : ℝ) : Prop := P_single_val = 0.08
def P_double (P_double_val : ℝ) : Prop := P_double_val = 0.2
def P_all_cond (P_all_val P_two_val : ℝ) : Prop := P_all_val = (1/4) * (P_all_val + P_two_val)
def P_none (P_none_val : ℝ) : Prop := P_none_val = 0.05

-- Define the conditional probability we need to prove
def P_none_given_not_D (P : MeasureTheory.ProbabilityMeasure Ω) := 
  P ({ω : Ω | ω ∉ {D}} ∩ {ω | ¬D ω ∧ ¬E ω ∧ ¬F ω}) = 1/5

theorem probability_none_given_not_D
  (P_single_val P_double_val P_none_val P_all_val P_two_val : ℝ)
  (D E F : Ω → Prop) [MeasurableSet (set_of D)]
  (P : MeasureTheory.ProbabilityMeasure Ω)
  (h1 : P_single P_single_val)
  (h2 : P_double P_double_val)
  (h3 : P_all_cond P_all_val P_two_val)
  (h4 : P_none P_none_val) :
  P_none_given_not_D P := 
by
  sorry

end probability_none_given_not_D_l427_427452


namespace area_ratios_l427_427462

-- Definitions of the points and segments
variables {A B C D E F : Type}
variables [Add A] [Add B] [Add C] [Add D] [Add E] [Add F]

-- Conditions given in the problem
def AB : ℝ := 150
def AC : ℝ := 150
def AD : ℝ := 50
def CF : ℝ := 100

-- Definition that we need to prove
theorem area_ratios :
  let BD := AB - AD in
  let AF := AC + CF in
  let area_ratio_CEF_DBE := 5 in
  (BD = 100) ∧ (AF = 250) ∧ (area_ratio_CEF_DBE = 5) := 
sorry

end area_ratios_l427_427462


namespace bisects_segment_l427_427425

-- Definitions of the given conditions
variables {O_1 O_2 A_1 A_2 B_1 B_2 C : Type*}

-- Assume O_1 and O_2 are the centers of two non-intersecting circles
axiom centers_non_intersecting_circles (O_1 O_2 : Type*) : ¬(O_1 = O_2)

-- Assume A_1 and A_2 are the points where the common external tangent touches the circles
axiom common_external_tangent (A_1 A_2 : Type*) (O_1 O_2 : Type*) : A_1 ≠ A_2 ∧ A_1 ∈ circle₁ O_1 ∧ A_2 ∈ circle₂ O_2

-- Assume B_1 and B_2 are the intersection points of the segment O_1O_2 with the circles
axiom intersection_points (B_1 B_2 : Type*) (O_1 O_2 : Type*) (A_1 A_2 : Type*) : B_1 ∈ segment O_1O_2 ∧ B_2 ∈ segment O_1O_2 ∧ B_1 ∈ circle₁ O_1 ∧ B_2 ∈ circle₂ O_2

-- Assume C is the intersection point of the lines A_1B_1 and A_2B_2
axiom intersection_point_C (C : Type*) (A_1 B_1 A_2 B_2 : Type*) : C = intersect_lines A_1 B_1 A_2 B_2

-- Define the main theorem to be proved
theorem bisects_segment (A_1 A_2 B_1 B_2 C : Type*) :
  let D := perp_line C (line B_1 B_2) in
  bisects D (segment A_1 A_2) :=
sorry

end bisects_segment_l427_427425


namespace two_digit_numbers_satisfying_l427_427073

def P (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  a * b

def S (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  a + b

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_numbers_satisfying (n : ℕ) : 
  is_two_digit n → n = P n + S n ↔ (n % 10 = 9) :=
by
  sorry

end two_digit_numbers_satisfying_l427_427073


namespace cartesian_eq_of_parametric_eq_cartesian_line_eq_of_polar_eq_range_of_m_l427_427459

-- Definitions from conditions
def parametric_eq (α : ℝ) : ℝ × ℝ :=
  (1 + 3 * Real.cos α, 3 + 3 * Real.sin α)

def polar_eq (ρ θ m : ℝ) : Prop :=
  ρ * Real.cos θ + ρ * Real.sin θ = m

-- Theorem to be proven
theorem cartesian_eq_of_parametric_eq (α : ℝ) :
  let (x, y) := parametric_eq α in (x - 1) ^ 2 + (y - 3) ^ 2 = 9 :=
sorry

theorem cartesian_line_eq_of_polar_eq (ρ θ m : ℝ) (h : polar_eq ρ θ m) :
  let x := ρ * Real.cos θ
  let y := ρ * Real.sin θ in
  x + y = m :=
sorry

theorem range_of_m (m : ℝ) :
  ∀ x y : ℝ, (x - 1) ^ 2 + (y - 3) ^ 2 = 9 ∧ (x + y = m) →
  4 - 3 * Real.sqrt 2 < m ∧ m < 4 + 3 * Real.sqrt 2 :=
sorry

end cartesian_eq_of_parametric_eq_cartesian_line_eq_of_polar_eq_range_of_m_l427_427459


namespace opposite_of_neg_two_is_two_l427_427989

theorem opposite_of_neg_two_is_two : -(-2) = 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l427_427989


namespace solution_A_to_B_ratio_l427_427563

def ratio_solution_A_to_B (V_A V_B : ℝ) : Prop :=
  (21 / 25) * V_A + (2 / 5) * V_B = (3 / 5) * (V_A + V_B) → V_A / V_B = 5 / 6

theorem solution_A_to_B_ratio (V_A V_B : ℝ) (h : (21 / 25) * V_A + (2 / 5) * V_B = (3 / 5) * (V_A + V_B)) :
  V_A / V_B = 5 / 6 :=
sorry

end solution_A_to_B_ratio_l427_427563


namespace selection_of_people_l427_427110

theorem selection_of_people (boys girls : ℕ) (h_boys : boys = 4) (h_girls : girls = 3) : 
  (∃ (total_selections : ℕ), total_selections = nat.choose (boys + girls) 4 - 1 ∧ total_selections = 34) :=
by
  sorry

end selection_of_people_l427_427110


namespace mark_team_free_throws_l427_427924

theorem mark_team_free_throws (F : ℕ) : 
  let mark_2_pointers := 25
  let mark_3_pointers := 8
  let opp_2_pointers := 2 * mark_2_pointers
  let opp_3_pointers := 1 / 2 * mark_3_pointers
  let total_points := 201
  2 * mark_2_pointers + 3 * mark_3_pointers + F + 2 * mark_2_pointers + 3 / 2 * mark_3_pointers + F / 2 = total_points →
  F = 10 := by
  sorry

end mark_team_free_throws_l427_427924


namespace probability_of_conditions_l427_427590

/-- Set of numbers 2-12 used in the grid -/
def numbers_set : Finset ℕ := Finset.range 13 \ {0, 1}

/-- The grid is a 3x3 matrix -/
def grid (α : Type*) := Matrix (Fin 3) (Fin 3) α

/-- Prime numbers in the set of numbers 2-12 -/
def primes : Finset ℕ := {2, 3, 5, 7, 11}

/-- Define the center square position -/
def center : Fin 3 × Fin 3 := (1, 1)

/-- Predicate for sum of each row being odd -/
def row_sum_odd (m : grid ℕ) : Prop :=
  ∀ i : Fin 3, (∑ j, m i j) % 2 = 1

/-- Predicate for center being prime -/
def center_is_prime (m : grid ℕ) : Prop :=
  m center.1 center.2 ∈ primes

/-- Statement of the problem -/
theorem probability_of_conditions :
  (∃ (m : grid ℕ), 
    (∀ i j, m i j ∈ numbers_set ∧ (∀ i j i' j', (i, j) ≠ (i', j') → m i j ≠ m i' j')) ∧
    row_sum_odd m ∧
    center_is_prime m) →
  1 / 6153 :=
sorry

end probability_of_conditions_l427_427590


namespace exist_n_div_k_l427_427498

open Function

theorem exist_n_div_k (k : ℕ) (h1 : k ≥ 1) (h2 : Nat.gcd k 6 = 1) :
  ∃ n : ℕ, n ≥ 0 ∧ k ∣ (2^n + 3^n + 6^n - 1) := 
sorry

end exist_n_div_k_l427_427498


namespace ab_bc_ca_leq_zero_l427_427381

theorem ab_bc_ca_leq_zero (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ca ≤ 0 :=
sorry

end ab_bc_ca_leq_zero_l427_427381


namespace intersection_A_B_l427_427421

def A := { x : ℝ | x / (x - 1) ≥ 0 }
def B := { y : ℝ | ∃ x : ℝ, y = 3 * x^2 + 1 }

theorem intersection_A_B : A ∩ B = { y : ℝ | y > 1 } :=
by sorry

end intersection_A_B_l427_427421


namespace sum_abc_equals_33_l427_427169

theorem sum_abc_equals_33 (a b c N : ℕ) (h_neq : ∀ x y, x ≠ y → x ≠ y → x ≠ y → x ≠ y ) 
(hN1 : N = 5 * a + 3 * b + 5 * c) (hN2 : N = 4 * a + 5 * b + 4 * c)
(h_range : 131 < N ∧ N < 150) : a + b + c = 33 :=
sorry

end sum_abc_equals_33_l427_427169


namespace circle_problem_l427_427781

theorem circle_problem (a : ℝ) (h_a : a ≠ 0) :
  ∃ C : ℝ × ℝ → Prop,
    (C = λ p, (p.1 - 3) ^ 2 + (p.2 - 1) ^ 2 = 2 ∧
    ((∃ A B : ℝ × ℝ, 
      A ≠ B ∧ 
      (l : ℝ × ℝ → Prop) ∃ l = λ p, p.1 - p.2 + a = 0) ∧ 
      |A.1 - A.2 - (B.1 - B.2)| = 2)) ∧
    (a = √2 - 2 ∨ a = -√2 - 2) :=
by
  sorry

end circle_problem_l427_427781


namespace product_of_22nd_and_23rd_multiples_of_3_l427_427537

theorem product_of_22nd_and_23rd_multiples_of_3 :
  (∃ n1 n2 : ℕ, 22 ≤ n1 ∧ n1 < 23 ∧ 66 = 3 * 22 ∧ 69 = 3 * 23 ∧ 66 * 69 = 4554) :=
begin
  sorry
end

end product_of_22nd_and_23rd_multiples_of_3_l427_427537


namespace rodney_has_35_more_than_ian_l427_427109

variable (Jessica Rodney Ian : ℝ) 

def jessica_has_100 (h1 : Jessica = 100) : Prop := 
Jessica = 100

def jessica_has_15_more_than_rodney (h2 : Jessica = Rodney + 15) : Prop :=
Jessica = Rodney + 15

def ian_has_half_as_much_money_as_jessica (h3 : Ian = Jessica / 2) : Prop :=
Ian = Jessica / 2

theorem rodney_has_35_more_than_ian  
  (h1 : jessica_has_100 Jessica) 
  (h2 : jessica_has_15_more_than_rodney Jessica Rodney)
  (h3 : ian_has_half_as_much_money_as_jessica Jessica Ian) : 
  Rodney - Ian = 35 :=
sorry

end rodney_has_35_more_than_ian_l427_427109


namespace perfect_square_iff_form_l427_427759

-- Defining the function to check if a natural number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

-- Defining the function for the given expression
def factorial_product (n : ℕ) : ℕ :=
  (List.range (2 * n + 1)).map Nat.factorial |> List.product

-- Defining the main expression
def given_expression (n : ℕ) : ℕ :=
  factorial_product n / Nat.factorial (n + 1)

-- The proof problem
theorem perfect_square_iff_form (n : ℕ) :
  (is_perfect_square (given_expression n)) ↔
    (∃ k : ℕ, n = 4 * k * (k + 1)) ∨ (∃ k : ℕ, n = 2 * k^2 - 1) := by
  sorry

end perfect_square_iff_form_l427_427759


namespace solve_part_one_solve_part_two_l427_427384

-- Define function f
def f (a x : ℝ) : ℝ := |a * x - 2| - |x + 2|

-- Prove for part (1)
theorem solve_part_one : 
  {x : ℝ | -1 / 3 ≤ x ∧ x ≤ 5} = {x : ℝ | f 2 x ≤ 1} :=
by
  -- Replace the proof with sorry
  sorry

-- Prove for part (2)
theorem solve_part_two :
  {a : ℝ | a = 1 ∨ a = -1} = {a : ℝ | ∀ x : ℝ, -4 ≤ f a x ∧ f a x ≤ 4} :=
by
  -- Replace the proof with sorry
  sorry

end solve_part_one_solve_part_two_l427_427384


namespace card_probability_correct_l427_427839

noncomputable def calc_card_prob (total_cards : ℕ) (draws : ℕ) : ℚ :=
  (1 / (rat.of_nat total_cards)) ^ draws

def card_prob_scenario : Prop :=
  let total_cards := 52
  let draws := 5
  let suit_cards := 13
  let prob_first_card := 1
  let prob_other_suits := (suit_cards * 3 / total_cards)   -- cards from other suits for the second draw
  let prob_third_fourth_fifth :=
    (suit_cards * 2 / total_cards) *  -- drawing remaining suit cards
    (suit_cards / total_cards) *       -- as constrained
    (suit_cards / total_cards)         -- last needed conditions
  
  prob_first_card * prob_other_suits * prob_third_fourth_fifth = 15 / 512

theorem card_probability_correct : card_prob_scenario :=
by {
  sorry
}

end card_probability_correct_l427_427839


namespace projection_check_l427_427596

open Matrix

-- Define the two vectors for projections
def v1 := ![2, 5] : Fin 2 → ℝ
def v2 := ![3, 2] : Fin 2 → ℝ
def u_proj := ![1, 5/2] : Fin 2 → ℝ

-- Define the scalar multiple k such that u is a multiple of u_proj
def k := Real.sqrt 7.25
def u := k • u_proj

-- Definition for the projection calculation
def proj (v u : Fin 2 → ℝ) : Fin 2 → ℝ :=
  ((v.dot_product u) / (u.dot_product u)) • u

-- Given condition: projection of v1 onto u is u_proj
axiom condition : proj v1 u = u_proj

-- Statement to check projection of v2 onto u
theorem projection_check : proj v2 u = ![(800/725), (2000/725)] :=
by
  sorry

end projection_check_l427_427596


namespace solve_for_x_l427_427927

theorem solve_for_x :
  ∃ x : ℝ, 0.035 * x = 42 ∧ x = 1200 :=
by
  use 1200
  split
  · simp
  · sorry

end solve_for_x_l427_427927


namespace chord_length_l427_427125

/-- Given two concentric circles with radii R and r, where the area of the annulus between them is 16π,
    a chord of the larger circle that is tangent to the smaller circle has a length of 8. -/
theorem chord_length {R r c : ℝ} 
  (h1 : R^2 - r^2 = 16)
  (h2 : (c / 2)^2 + r^2 = R^2) :
  c = 8 :=
by
  sorry

end chord_length_l427_427125


namespace max_tetrahedron_volume_l427_427685

noncomputable def tetrahedron_volume (a b c d : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := a in
  let (x2, y2, z2) := b in
  let (x3, y3, z3) := c in
  let (x4, y4, z4) := d in
  1/6 * |det![
    [x2 - x1, y2 - y1, z2 - z1],
    [x3 - x1, y3 - y1, z3 - z1],
    [x4 - x1, y4 - y1, z4 - z1]]|

def point_in_sphere (p : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := p in
  x^2 + y^2 + z^2 ≤ 85

theorem max_tetrahedron_volume : 
  tetrahedron_volume (0, 0, 9) (9, 0, 0) (0, 9, 0) (0, 0, -9) = 243 := 
sorry

end max_tetrahedron_volume_l427_427685


namespace math_problem_l427_427853

-- Definitions of the geometric setup
structure Triangle (α : Type _) :=
  (A B C : α)
  (angle_A : ℕ := 60)
  (AB_gt_AC : Prop)

structure Point (α : Type _) (triangle : Triangle α) :=
  (O : α) -- circumcenter
  (H : α) -- orthocenter
  (M : α) -- point on BH
  (N : α) -- point on HF

-- The properties given in the problem as conditions
def ProblemConditions (α : Type _) [LinearOrderedField α] (triangle : Triangle α) :=
  ∀ (points : Point α triangle),
  ∃ (BE CF : α) (intersect_B_to_E intersect_C_to_F : α),
    BE = CF ∧
    intersect_B_to_E = points.H ∧ intersect_C_to_F = points.H ∧
    points.M ∈ BE ∧ points.N ∈ CF ∧
    distance points.M points.B = distance points.N points.C 

-- The actual theorem statement we aim to prove
theorem math_problem
  {α : Type _} [LinearOrderedField α]
  (triangle : Triangle α) (points : Point α triangle)
  (h_conditions : ProblemConditions α triangle) :
  ∃ (OH MH NH : α),
  distance points.MH points.H + distance points.NH points.H = distance points.OH points.H * √3 :=
begin
  sorry
end

end math_problem_l427_427853


namespace coefficient_of_term_in_expansion_l427_427182

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  nat.choose n k

noncomputable def power_rat (r : ℚ) (n : ℕ) : ℚ :=
  r ^ n
  
theorem coefficient_of_term_in_expansion :
  ∃ (c : ℚ), 
  (∃ (x y : ℚ), (x = 3 ∧ y = 7) → 
  x^3 * y^7 = c * ((binomial_coefficient 10 3) * 
  (power_rat (2/3) 3) * (power_rat (-3/4) 7))) ∧
  c = -3645/7656 :=
begin
  sorry
end

end coefficient_of_term_in_expansion_l427_427182


namespace digit_after_point_l427_427180

theorem digit_after_point (n : ℕ) (a b : ℕ) (h : n = 222) (hab : a = 55) (hbb : b = 777) :
  (a / b) = 0.070817070817070817... ->
  (n % 6) = 0 ->
  (repeating_block : string) = "070817" ->
  repeating_block[(n - 1) % 6] = '7' :=
begin
  -- sorry placeholder for proof
  sorry,
end

end digit_after_point_l427_427180


namespace part1_part2_l427_427409

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

-- Statement for part (1)
theorem part1 (m : ℝ) : (m > -2) → (∀ x : ℝ, m + f x > 0) :=
sorry

-- Statement for part (2)
theorem part2 (m : ℝ) : (m > 2) ↔ (∀ x : ℝ, m - f x > 0) :=
sorry

end part1_part2_l427_427409


namespace value_of_x_in_set_l427_427022

theorem value_of_x_in_set :
  (∃ x : ℝ, 2 ∈ ({1, x^2 + x} : Set ℝ) ∧ (x = 1 ∨ x = -2)) :=
begin
  sorry
end

end value_of_x_in_set_l427_427022


namespace ellipse_equation_and_fixed_point_l427_427081

theorem ellipse_equation_and_fixed_point (a b : ℝ) (h : a > b) (h1 : 0 < b) (h_perimeter : ∀ (x y : ℝ), \sqrt{x^2 + y^2}) :
  (on_ellipse : ∀ (p : ℝ × ℝ), ellipse p) 
  (h_focus_distance: \frac{x^2}{4} + \frac{y^2}{3} = 1 ) :
  (standard_form :  \)
(sorry:  proof_here)

end ellipse_equation_and_fixed_point_l427_427081


namespace sum_possible_n_values_l427_427687

def is_possible_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

noncomputable def sum_of_possible_values (n_values : List ℕ) : ℕ :=
  n_values.sum

theorem sum_possible_n_values :
  let n_values := [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] in
  sum_of_possible_values n_values = 121 :=
by
  let n_values := [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
  show sum_of_possible_values n_values = 121
  sorry

end sum_possible_n_values_l427_427687


namespace total_income_per_minute_l427_427566

theorem total_income_per_minute :
  let black_shirt_price := 30
  let black_shirt_quantity := 250
  let white_shirt_price := 25
  let white_shirt_quantity := 200
  let red_shirt_price := 28
  let red_shirt_quantity := 100
  let blue_shirt_price := 25
  let blue_shirt_quantity := 50

  let black_discount := 0.05
  let white_discount := 0.08
  let red_discount := 0.10

  let total_black_income_before_discount := black_shirt_quantity * black_shirt_price
  let total_white_income_before_discount := white_shirt_quantity * white_shirt_price
  let total_red_income_before_discount := red_shirt_quantity * red_shirt_price
  let total_blue_income_before_discount := blue_shirt_quantity * blue_shirt_price

  let total_income_before_discount :=
    total_black_income_before_discount + total_white_income_before_discount + total_red_income_before_discount + total_blue_income_before_discount

  let total_black_discount := black_discount * total_black_income_before_discount
  let total_white_discount := white_discount * total_white_income_before_discount
  let total_red_discount := red_discount * total_red_income_before_discount

  let total_discount :=
    total_black_discount + total_white_discount + total_red_discount

  let total_income_after_discount :=
    total_income_before_discount - total_discount

  let total_minutes := 40
  let total_income_per_minute := total_income_after_discount / total_minutes

  total_income_per_minute = 387.38 := by
  sorry

end total_income_per_minute_l427_427566


namespace find_t_l427_427849

noncomputable def f (t x : ℝ) : ℝ := x^3 + (2 * t - 1) * x + 3

theorem find_t (t : ℝ) (h : f t (-1) = 3) : 2 * t + 2 = 0 → t = -1 :=
begin
  intro h_tangent,
  linarith,
end

end find_t_l427_427849


namespace initial_markup_percentage_l427_427681

variable (W : ℝ) (initial_price : ℝ) (final_price : ℝ)

theorem initial_markup_percentage :
  (initial_price - W) / W * 100 = 70 :=
by
  assume (h1 : initial_price = 34) (h2 : final_price = 40) 
  (h3 : final_price = initial_price + 6) (h4 : final_price = 2 * W)
  sorry

end initial_markup_percentage_l427_427681


namespace max_minute_hands_l427_427229

theorem max_minute_hands (m n : ℕ) (h : m * n = 27) : m + n ≤ 28 :=
  sorry

end max_minute_hands_l427_427229


namespace no_lambda_for_even_d_l427_427374

theorem no_lambda_for_even_d (d : ℕ) (h_even : d % 2 = 0) :
  ¬ (∃ λ > 0, ∀ f : ℝ[X], f.degree = d ∧ f.coeff_range ⊆ Int ∧ ¬ ∃ x : ℝ, f.eval x = 0 → ∀ x : ℝ, f.eval x > λ) := sorry

end no_lambda_for_even_d_l427_427374


namespace smallest_distance_l427_427561

open Real

-- Definition for the condition: six points in or on a square of side length 2
def points_in_square (p : Fin 6 → ℝ × ℝ) : Prop :=
  ∀ i, abs (p i).1 ≤ 1 ∧ abs (p i).2 ≤ 1

-- The theorem statement to be proved
theorem smallest_distance (p : Fin 6 → ℝ × ℝ) (h : points_in_square p): ∃ (b : ℝ), 
  b = sqrt 2 ∧ ∃ (i j : Fin 6), i ≠ j ∧ dist (p i) (p j) ≤ b :=
by
  sorry

end smallest_distance_l427_427561


namespace equilateral_triangle_dot_product_l427_427464

theorem equilateral_triangle_dot_product (A B C : Type) [InnerProductSpace ℝ A] 
  (AB BC : A) (h_eq_triangle : IsEquilateralTriangle A B C) (h_length : ∥AB∥ = 1) : 
  ⟪AB, BC⟫ = - (1 / 2) :=
by 
  sorry

end equilateral_triangle_dot_product_l427_427464


namespace bad_carrots_eq_13_l427_427645

-- Define the number of carrots picked by Haley
def haley_picked : ℕ := 39

-- Define the number of carrots picked by her mom
def mom_picked : ℕ := 38

-- Define the number of good carrots
def good_carrots : ℕ := 64

-- Define the total number of carrots picked
def total_carrots : ℕ := haley_picked + mom_picked

-- State the theorem to prove the number of bad carrots
theorem bad_carrots_eq_13 : total_carrots - good_carrots = 13 := by
  sorry

end bad_carrots_eq_13_l427_427645


namespace minimum_value_of_function_l427_427390

theorem minimum_value_of_function (x : ℝ) (hx : 0 < x ∧ x < 1) : 
  ∃ y : ℝ, (∀ z : ℝ, z = (1 / x) + (4 / (1 - x)) → y ≤ z) ∧ y = 9 :=
by
  sorry

end minimum_value_of_function_l427_427390


namespace six_digit_number_theorem_l427_427684

-- Define the problem conditions
def six_digit_number_condition (N : ℕ) (x : ℕ) : Prop :=
  N = 200000 + x ∧ N < 1000000 ∧ (10 * x + 2 = 3 * N)

-- Define the value of x
def value_of_x : ℕ := 85714

-- Main theorem to prove
theorem six_digit_number_theorem (N : ℕ) (x : ℕ) (h1 : x = value_of_x) :
  six_digit_number_condition N x → N = 285714 :=
by
  intros h
  sorry

end six_digit_number_theorem_l427_427684


namespace minimum_quadratic_value_l427_427418

theorem minimum_quadratic_value (h : ℝ) (x : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 3 → (x - h)^2 + 1 ≥ 10) ∧ (∃ x, 1 ≤ x ∧ x ≤ 3 ∧ (x - h)^2 + 1 = 10) 
  ↔ h = -2 ∨ h = 6 :=
by
  sorry

end minimum_quadratic_value_l427_427418


namespace greatest_least_50th_term_diff_l427_427285

noncomputable def arithmetic_sequence_50th_term_difference : Prop :=
  ∃ (a : ℕ → ℝ) (d : ℝ),
    (∀ n, 0 ≤ n → n < 300 → 10 ≤ a n ∧ a n ≤ 100) ∧
    (∑ n in Finset.range 300, a n = 15000) ∧
    (∃ (L G : ℝ), 
      L = 50 - 251 * (40 / 299) ∧
      G = 50 + 251 * (40 / 299) ∧
      G - L = 2 * (40 / 299)) ∧
    (2 * (40 / 299) = 20080 / 299)

theorem greatest_least_50th_term_diff : arithmetic_sequence_50th_term_difference :=
  sorry

end greatest_least_50th_term_diff_l427_427285


namespace no_high_quality_triangle_exist_high_quality_quadrilateral_l427_427475

-- Define the necessary predicate for a number being a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define the property of being a high-quality triangle
def high_quality_triangle (a b c : ℕ) : Prop :=
  is_perfect_square (a + b) ∧ is_perfect_square (b + c) ∧ is_perfect_square (c + a)

-- Define the property of non-existence of a high-quality triangle
theorem no_high_quality_triangle (a b c : ℕ) (ha : Prime a) (hb : Prime b) (hc : Prime c) : 
  ¬high_quality_triangle a b c := by sorry

-- Define the property of being a high-quality quadrilateral
def high_quality_quadrilateral (a b c d : ℕ) : Prop :=
  is_perfect_square (a + b) ∧ is_perfect_square (b + c) ∧ is_perfect_square (c + d) ∧ is_perfect_square (d + a)

-- Define the property of existence of a high-quality quadrilateral
theorem exist_high_quality_quadrilateral (a b c d : ℕ) (ha : Prime a) (hb : Prime b) (hc : Prime c) (hd : Prime d) : 
  high_quality_quadrilateral a b c d := by sorry

end no_high_quality_triangle_exist_high_quality_quadrilateral_l427_427475


namespace determinant_real_root_unique_l427_427075

theorem determinant_real_root_unique {a b c : ℝ} (ha : 0 < a ∧ a ≠ 1) (hb : 0 < b ∧ b ≠ 1) (hc : 0 < c ∧ c ≠ 1) :
  ∃! x : ℝ, (Matrix.det ![
    ![x - 1, c - 1, -(b - 1)],
    ![-(c - 1), x - 1, a - 1],
    ![b - 1, -(a - 1), x - 1]
  ]) = 0 :=
by
  sorry

end determinant_real_root_unique_l427_427075


namespace line_equation_of_parabola_focus_slope_l427_427815

theorem line_equation_of_parabola_focus_slope
  (F : ℝ × ℝ)
  (hF : F = (0, 1))
  (k : ℝ) (hk : k > 0)
  (A B : ℝ × ℝ)
  (hA : A.1 ^ 2 = 4 * A.2) (hB : B.1 ^ 2 = 4 * B.2)
  (P : ℝ × ℝ)
  (hP : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (M : ℝ × ℝ)
  (hM : M = (2 * sqrt 3, 3))
  (hMF : dist M F = 4) :
  ∃ l : ℝ → ℝ, l x = sqrt 3 * x + 1 :=
begin
  sorry
end

end line_equation_of_parabola_focus_slope_l427_427815


namespace fixed_point_functions_l427_427038

def is_fixed_point_function (f : ℝ → ℝ) :=
  ∃ x : ℝ, f x = x

def fA : ℝ → ℝ := λ x, x^2 - x - 3
def fB : ℝ → ℝ := λ x, 2^x + x
def fC : ℝ → ℝ := λ x, sqrt x + 1
def fD : ℝ → ℝ := λ x, 2^x - 2

theorem fixed_point_functions :
  is_fixed_point_function fA ∧
  ¬ is_fixed_point_function fB ∧
  is_fixed_point_function fC ∧
  is_fixed_point_function fD :=
by
  sorry

end fixed_point_functions_l427_427038


namespace books_solution_l427_427658

def books_problem (x y m : ℕ) : Prop :=
  x = (y - 14) / m ∧ x = (y + 3) / 9

theorem books_solution : ∃ (x y m : ℕ), books_problem x y m ∧ x = 17 ∧ y = 150 :=
by
  use 17, 150, 8
  unfold books_problem
  split
  {
    -- Proof of (150 − 14) / 8 = 17
    calc
      (150 - 14) / 8 = 136 / 8 : by simp
      _ = 17 : by norm_num
  }
  {
    -- Proof of (150 + 3) / 9 = 17
    calc
      (150 + 3) / 9 = 153 / 9 : by simp
      _ = 17 : by norm_num
  }

end books_solution_l427_427658


namespace ratio_side_lengths_sum_l427_427148

open Real

def ratio_of_areas (a b : ℝ) : ℝ := a / b

noncomputable def sum_of_abc (area_ratio : ℝ) : ℤ :=
  let side_length_ratio := sqrt area_ratio
  let a := 2
  let b := 1
  let c := 1
  a + b + c

theorem ratio_side_lengths_sum :
  sum_of_abc (ratio_of_areas 300 75) = 4 :=
by
  sorry

end ratio_side_lengths_sum_l427_427148


namespace count_valid_n_l427_427367

def is_perfect_square (x : ℤ) : Prop :=
  ∃ k : ℤ, k * k = x

def valid_n (n : ℤ) : Prop :=
  is_perfect_square (n / (24 - n))

theorem count_valid_n : finset.card (finset.filter valid_n (finset.range 25)) = 2 := sorry

end count_valid_n_l427_427367


namespace opposite_of_neg_two_l427_427984

theorem opposite_of_neg_two : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_two_l427_427984


namespace min_distance_origin_to_line_l427_427999

-- Definitions of the conditions
def line_eq (x y : ℝ) : Prop := 2 * x - y + 5 = 0
def origin := (0 : ℝ, 0 : ℝ)

-- Distance formula from a point to a line
def distance_from_origin_to_line : ℝ := abs 5 / Real.sqrt (2^2 + (-1)^2)

-- Lean Theorem statement
theorem min_distance_origin_to_line : line_eq x y → distance_from_origin_to_line = Real.sqrt 5 := 
by 
  sorry

end min_distance_origin_to_line_l427_427999


namespace ratio_of_blue_to_red_l427_427534

variable (B : ℕ) -- Number of blue lights

def total_white := 59
def total_colored := total_white - 5
def red_lights := 12
def green_lights := 6

def total_bought := red_lights + green_lights + B

theorem ratio_of_blue_to_red (h : total_bought = total_colored) :
  B / red_lights = 3 :=
by
  sorry

end ratio_of_blue_to_red_l427_427534


namespace centroids_quadrilateral_l427_427545

theorem centroids_quadrilateral {A B C D : Point}
  (h_ABCD : quadrilateral A B C D) :
  let G_ABC := centroid_of_triangle A B C,
      G_BCD := centroid_of_triangle B C D,
      G_CDA := centroid_of_triangle C D A,
      G_DAB := centroid_of_triangle D A B
  in (quadrilateral G_ABC G_BCD G_CDA G_DAB) ∧
     (parallel (side G_ABC G_BCD) (side A D)) ∧
     (parallel (side G_BCD G_CDA) (side B C)) ∧
     (parallel (side G_CDA G_DAB) (side C A)) ∧
     (parallel (side G_DAB G_ABC) (side D B)) ∧
     (parallel (diagonal G_ABC G_CDA) (diagonal A C)) ∧
     (parallel (diagonal G_BCD G_DAB) (diagonal B D)) :=
sorry

end centroids_quadrilateral_l427_427545


namespace sum_of_abc_is_33_l427_427166

theorem sum_of_abc_is_33 (a b c N : ℕ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c)
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hN1 : N = 5 * a + 3 * b + 5 * c)
    (hN2 : N = 4 * a + 5 * b + 4 * c) (hN_range : 131 < N ∧ N < 150) :
  a + b + c = 33 := 
sorry

end sum_of_abc_is_33_l427_427166


namespace marble_problem_l427_427225

-- Define the initial number of marbles
def initial_marbles : Prop :=
  ∃ (x y : ℕ), (y - 4 = 2 * (x + 4)) ∧ (y + 2 = 11 * (x - 2)) ∧ (y = 20) ∧ (x = 4)

-- The main theorem to prove the initial number of marbles
theorem marble_problem (x y : ℕ) (cond1 : y - 4 = 2 * (x + 4)) (cond2 : y + 2 = 11 * (x - 2)) :
  y = 20 ∧ x = 4 :=
sorry

end marble_problem_l427_427225


namespace perpendicular_vectors_have_zero_dot_product_l427_427427

variables (m : ℝ)

def a : ℝ × ℝ := (2, m)
def b : ℝ × ℝ := (5, 1)
def a_minus_b : ℝ × ℝ := (2 - 5, m - 1)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem perpendicular_vectors_have_zero_dot_product : 
  (dot_product a a_minus_b = 0) ↔ (m = -2 ∨ m = 3) := 
sorry

end perpendicular_vectors_have_zero_dot_product_l427_427427


namespace number_of_terms_in_sequence_l427_427020

def arithmetic_sequence_terms (a d l : ℕ) : ℕ :=
  (l - a) / d + 1

theorem number_of_terms_in_sequence : arithmetic_sequence_terms 1 4 57 = 15 :=
by {
  sorry
}

end number_of_terms_in_sequence_l427_427020


namespace remaining_people_statement_l427_427097

-- Definitions of conditions
def number_of_people : Nat := 10
def number_of_knights (K : Nat) : Prop := K ≤ number_of_people
def number_of_liars (L : Nat) : Prop := L ≤ number_of_people
def statement (s : String) : Prop := s = "There are more liars" ∨ s = "There are equal numbers"

-- Main theorem
theorem remaining_people_statement (K L : Nat) (h_total : K + L = number_of_people) 
  (h_knights_behavior : ∀ k, k < K → statement "There are equal numbers") 
  (h_liars_behavior : ∀ l, l < L → statement "There are more liars") :
  K = 5 → L = 5 → ∀ i, i < number_of_people → (i < 5 → statement "There are more liars") ∧ (i >= 5 → statement "There are equal numbers") := 
by
  sorry

end remaining_people_statement_l427_427097


namespace sum_of_distinct_real_numbers_l427_427502

theorem sum_of_distinct_real_numbers (p q r s : ℝ) (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : ∀ x : ℝ, x^2 - 12 * p * x - 13 * q = 0 -> (x = r ∨ x = s)) 
  (h2 : ∀ x : ℝ, x^2 - 12 * r * x - 13 * s = 0 -> (x = p ∨ x = q)) :
  p + q + r + s = 2028 :=
begin
  sorry
end

end sum_of_distinct_real_numbers_l427_427502


namespace area_triangle_l427_427392

noncomputable def hyperbola (x y : ℝ) : Prop := (y^2 / 4) - x^2 = 1

def asymptote1 (x y : ℝ) : Prop := 2 * x + y = 0

def asymptote2 (x y : ℝ) : Prop := 2 * x - y = 0

def perpendicular_distance (x y : ℝ) (a : ℝ → Prop) : ℝ :=
  if a = asymptote1 then abs (2 * x - y) / sqrt 5 else abs (2 * x + y) / sqrt 5

theorem area_triangle (x y : ℝ) (h : x ≠ 0)
  (h_hyperbola : hyperbola x y) : 
  let PA := perpendicular_distance x y asymptote1,
      PB := perpendicular_distance x y asymptote2,
      sin_angle_APB := 4 / 5 in
  (1/2) * PA * PB * sin_angle_APB = 8 / 25 :=
by
  sorry

end area_triangle_l427_427392


namespace max_value_xy_expression_l427_427915

theorem max_value_xy_expression : ∀ (x y : ℝ), x > 0 → y > 0 → 5 * x + 6 * y < 96 → xy(96 - 5x - 6y) ≤ 1092.267 := 
    sorry

end max_value_xy_expression_l427_427915


namespace problem_solution_l427_427696

theorem problem_solution (h1 : 0.8^(-0.1) < 0.8^(-0.2)) (h2 : log 2 3.4 < log 2 real.pi → false) (h3 : log 7 6 > log 8 6) (h4 : 1.7^1.01 < 1.6^1.01 → false) :
  (0.8^(-0.1) < 0.8^(-0.2)) ∧ (log 7 6 > log 8 6) :=
by
  sorry -- proof steps are not required

end problem_solution_l427_427696


namespace opposite_of_neg_two_is_two_l427_427990

theorem opposite_of_neg_two_is_two : -(-2) = 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l427_427990


namespace compare_smallest_positive_roots_l427_427299

noncomputable def smallest_positive_root (p : polynomial ℝ) : ℝ :=
  classical.some (polynomial.exists_root p)

theorem compare_smallest_positive_roots :
  let p1 := (X^2011 + C 2011 * X - 1 : polynomial ℝ)
  let p2 := (X^2011 - C 2011 * X + 1 : polynomial ℝ)
  let x1 := smallest_positive_root p1
  let x2 := smallest_positive_root p2
  0 < x1 → 0 < x2 → is_root p1 x1 ∧ is_root p2 x2 → x1 < x2 :=
by
  intros p1 p2 x1 x2 h1pos h2pos hroots
  sorry

end compare_smallest_positive_roots_l427_427299


namespace cards_traded_between_Padma_and_Robert_l427_427540

def total_cards_traded (padma_first_trade padma_second_trade robert_first_trade robert_second_trade : ℕ) : ℕ :=
  padma_first_trade + padma_second_trade + robert_first_trade + robert_second_trade

theorem cards_traded_between_Padma_and_Robert (h1 : padma_first_trade = 2) 
                                            (h2 : robert_first_trade = 10)
                                            (h3 : padma_second_trade = 15)
                                            (h4 : robert_second_trade = 8) :
                                            total_cards_traded 2 15 10 8 = 35 := 
by 
  sorry

end cards_traded_between_Padma_and_Robert_l427_427540


namespace ratio_of_squares_l427_427156

theorem ratio_of_squares (a b c : ℕ) (h : (a = 2) ∧ (b = 1) ∧ (c = 1)) :
  (∑ i in {a, b, c}, i) = 4 :=
by {
  cases h with h1 h2,
  cases h2 with h2 h3,
  rw [h1, h2, h3],
  simp,
}

end ratio_of_squares_l427_427156


namespace sqrt_simplification_l427_427631

theorem sqrt_simplification (x : ℝ) (hx : x ≠ 0) :
  sqrt (1 + ( (x^6 - 1) / (2 * x^3) )^2) = (x^3 / 2) + (1 / (2 * x^3)) :=
sorry

end sqrt_simplification_l427_427631


namespace polar_circle_equation_l427_427042

-- Definitions of the circle's center (1,1) and its radius √2
def center : ℝ × ℝ := (1, 1)
def radius : ℝ := Real.sqrt 2

-- The Cartesian to polar coordinate conversions
def x (ρ θ : ℝ) : ℝ := ρ * cos θ
def y (ρ θ : ℝ) : ℝ := ρ * sin θ

theorem polar_circle_equation (ρ θ : ℝ) :
  ((x ρ θ - center.1) ^ 2 + (y ρ θ - center.2) ^ 2 = radius ^ 2) →
  (ρ = 2 * Real.sqrt 2 * cos (θ - Real.pi / 4)) :=
sorry

end polar_circle_equation_l427_427042


namespace statistics_remain_unchanged_l427_427708

def scores_before : List ℕ := [30, 40, 50, 50, 60, 60, 60, 70, 80, 90]
def score_added : ℕ := 50

noncomputable def range (l : List ℕ) : ℕ := l.maximumD 0 - l.minimumD 0

noncomputable def median (l : List ℕ) : ℕ := 
  let sorted := l.sorted
  if sorted.length % 2 = 0 then
    (sorted.get! (sorted.length / 2 - 1) + sorted.get! (sorted.length / 2)) / 2
  else
    sorted.get! (sorted.length / 2)

noncomputable def mode (l : List ℕ) : List ℕ :=
  let freq_map := l.groupBy id
  let max_freq := freq_map.map (λ x => x.length).maximumD 0
  freq_map.filter (λ x => x.length = max_freq).map (λ x => x.headI)

noncomputable def mid_range (l : List ℕ) : ℕ := (l.maximumD 0 + l.minimumD 0) / 2

theorem statistics_remain_unchanged :
  ∀ (l : List ℕ) (a : ℕ), 
    l = scores_before → 
    a = score_added → 
    range l = range (a :: l) ∧ median l = median (a :: l) ∧ mode l = mode (a :: l) ∧ mid_range l = mid_range (a :: l) :=
by
  sorry

end statistics_remain_unchanged_l427_427708


namespace exists_nat_numbers_permutation_and_sum_to_nines_l427_427474

open Nat List

-- Define a function to check if one natural number is a permutation of the digits of another
def isPermutation (m n : ℕ) : Prop :=
  (Nat.digits 10 m).perm (Nat.digits 10 n)

-- Define a function to check if a number is composed entirely of 9s
def isAllNines (x : ℕ) : Prop :=
  ∀ d ∈ Nat.digits 10 x, d = 9

theorem exists_nat_numbers_permutation_and_sum_to_nines :
  ∃ m n : ℕ, isPermutation m n ∧ isAllNines (m + n) := sorry

end exists_nat_numbers_permutation_and_sum_to_nines_l427_427474


namespace prove_bride_age_l427_427605

variables (B G : ℕ)

noncomputable def bride_age_proof : Prop :=
  (B = G + 19) ∧ (B + G = 185) → (B = 102)

theorem prove_bride_age : bride_age_proof B G :=
by {
  intros h,
  cases h with h1 h2,
  sorry
}

end prove_bride_age_l427_427605


namespace product_of_points_l427_427278

def f (n : ℕ) : ℕ :=
  if n % 6 = 0 then 7
  else if n % 2 = 0 then 3
  else if isPrime n then 5
  else 0

def alliePoints : List ℕ := [2, 3, 4, 5, 6]
def bettyPoints : List ℕ := [6, 3, 4, 2, 1]

def totalPoints (rolls : List ℕ) : ℕ :=
  rolls.foldl (fun acc n => acc + f n) 0

theorem product_of_points :
  totalPoints alliePoints * totalPoints bettyPoints = 500 := by
  sorry

end product_of_points_l427_427278


namespace perpendicular_vectors_magnitude_l427_427428

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem perpendicular_vectors_magnitude:
  ∀ (λ : ℝ), (1 * λ + 2 * (-1) = 0) → vector_magnitude (1 + λ, 2 - 1) = Real.sqrt 10 :=
by
  intros λ h
  -- The proof goes here
  sorry

end perpendicular_vectors_magnitude_l427_427428


namespace gate_probability_sum_eq_l427_427951

noncomputable def num_gates : Nat := 10
noncomputable def distance_between_gates : Nat := 80
noncomputable def max_distance_to_walk : Nat := 320

theorem gate_probability_sum_eq :
  let total_choices := num_gates * (num_gates - 1)
  let valid_pairs := 58
  let prob := valid_pairs / total_choices
  let simplified_prob := (29 : ℚ) / 45
  m + n = 74 :=
by
  have h1 : total_choices = 90 := by norm_num
  have h2 : valid_pairs = 58 := by norm_num
  have h3 : simplified_prob = (29 : ℚ) / 45 := by norm_num
  have h4 : (SimplifiedProb.num, SimplifiedProb.denom) = (29, 45) := by norm_num
  exact h4
  sorry

end gate_probability_sum_eq_l427_427951


namespace opposite_neg_two_l427_427980

theorem opposite_neg_two : -(-2) = 2 := by
  sorry

end opposite_neg_two_l427_427980


namespace ralphStartsWith_l427_427546

def ralphEndsWith : ℕ := 15
def ralphLoses : ℕ := 59

theorem ralphStartsWith : (ralphEndsWith + ralphLoses = 74) :=
by
  sorry

end ralphStartsWith_l427_427546


namespace incorrect_statements_l427_427134

-- Conditions
def statement_1 := ({0} = ∅)
def statement_2 := ({2} ⊆ {2, 4, 6})
def statement_3 := (2 ∈ {x | x^2 - 3*x + 2 = 0})
def statement_4 := (0 ∈ {0})

-- Theorem to prove which statements are incorrect
theorem incorrect_statements : 
  ¬ statement_1 ∧ statement_2 ∧ ¬ statement_3 ∧ statement_4 :=
by
  sorry

end incorrect_statements_l427_427134


namespace find_largest_average_l427_427165

noncomputable def multiples_average (a d n : ℕ) : ℚ :=
  (a + d * n) / 2

theorem find_largest_average :
  let avg_11 := multiples_average 11 11 (100810 / 11)
  let avg_13 := multiples_average 13 13 (100810 / 13)
  let avg_17 := multiples_average 17 17 (100810 / 17)
  let avg_19 := multiples_average 19 19 (100810 / 19) in
  max avg_11 (max avg_13 (max avg_17 avg_19)) = 50413.5 :=
begin
  let avg_11 := multiples_average 11 11 (100810 / 11),
  let avg_13 := multiples_average 13 13 (100810 / 13),
  let avg_17 := multiples_average 17 17 (100810 / 17),
  let avg_19 := multiples_average 19 19 (100810 / 19),
  have : max avg_11 (max avg_13 (max avg_17 avg_19)) = 50413.5,
  { sorry },
  exact this,
end

end find_largest_average_l427_427165


namespace sum_f_roots_eq_a0_l427_427077

open Complex

noncomputable def f (a : List ℂ) (x : ℂ) : ℂ :=
  a.foldr (λ (c : ℂ) (acc : ℂ) => acc * x + c) 0

theorem sum_f_roots_eq_a0 (n : ℕ) (h_n : 0 < n) 
    (a : List ℂ) (m : ℕ) (h_m : 0 < m ∧ m < n) 
    (z : Fin n → ℂ) (hz : ∀ k : Fin n, z k ^ n = 1) :
    1 / (n : ℂ) * (∑ k : Fin n, f a (z k)) = a.head' :=
by
  sorry

end sum_f_roots_eq_a0_l427_427077


namespace distance_between_poles_correctness_l427_427271

noncomputable def distance_between_poles (L : ℕ) (W : ℕ) (n : ℕ) : ℝ :=
  let P := 2 * (L + W) -- Perimeter of the rectangle
  P / (n - 1)

theorem distance_between_poles_correctness :
  distance_between_poles 90 50 56 ≈ 5.09 :=
by
  sorry

end distance_between_poles_correctness_l427_427271


namespace final_amount_left_l427_427674

noncomputable def total_income : ℝ := 1000000

noncomputable def child_percentage : ℝ := 0.20
noncomputable def num_children : ℕ := 3
noncomputable def wife_percentage : ℝ := 0.30
noncomputable def donation_percentage : ℝ := 0.05

noncomputable def distributed_to_children := child_percentage * num_children
noncomputable def distributed_to_wife := wife_percentage

theorem final_amount_left :
  let total_distributed := distributed_to_children + distributed_to_wife in
  let remaining_income := 1 - total_distributed in
  let donation := donation_percentage * remaining_income in
  (remaining_income * total_income - donation * total_income) = 95000 :=
by
  sorry

end final_amount_left_l427_427674


namespace coprime_among_five_consecutive_l427_427544

theorem coprime_among_five_consecutive (n : ℕ) : 
  ∃ k ∈ {n, n+1, n+2, n+3, n+4}, ∀ m ∈ {n, n+1, n+2, n+3, n+4}, k ≠ m → Nat.Coprime k m := 
by
  sorry

end coprime_among_five_consecutive_l427_427544


namespace range_of_theta_l427_427067

variable {α : Type*} [inner_product_space ℝ α]

theorem range_of_theta (a b : α)
  (unit_a : ∥a∥ = 1) (unit_b : ∥b∥ = 1) (θ : ℝ) (angle_ab : real.angle a b = θ)
  (norm_add_gt : ∥a + b∥ > 1) (norm_sub_gt : ∥a - b∥ > 1) :
  real.arcsin(-1 / 2) < θ ∧ θ < real.arcsin(1 / 2) :=
by
  sorry

end range_of_theta_l427_427067


namespace extreme_value_of_f_max_m_condition_l427_427003

noncomputable def f : ℝ → ℝ := λ x, Real.exp (-x) + 6 * x

theorem extreme_value_of_f :
  ∃ x, f(x) = 6 - 6 * Real.log 6 ∧ 
  (∀ y, y ≠ x → f(y) > f(x)) := sorry

theorem max_m_condition :
  ∃ m : ℕ, (∀ x ∈ Set.Icc 1 m, f(x) > x^2 + 3) ∧
  (∀ n : ℕ, n > 5 → ∃ x ∈ Set.Icc 1 n, f(x) ≤ x^2 + 3) := sorry

end extreme_value_of_f_max_m_condition_l427_427003


namespace computer_literate_female_employees_l427_427864

-- Given conditions
def total_employees := 1500
def percentage_female := 0.6
def percentage_male_computer_literate := 0.5
def percentage_computer_literate := 0.62

-- Intermediate calculations from conditions
def number_female_employees := percentage_female * total_employees
def number_male_employees := total_employees - number_female_employees
def number_computer_literate_male := percentage_male_computer_literate * number_male_employees
def total_computer_literate_employees := percentage_computer_literate * total_employees

-- Statement to prove
theorem computer_literate_female_employees :
  (total_computer_literate_employees - number_computer_literate_male) = 630 := by
  -- Placeholder for the proof
  sorry

end computer_literate_female_employees_l427_427864


namespace problem1_problem2_l427_427244

variable {n : ℕ}
variable {a b : ℝ}

-- Part 1
theorem problem1 (h : ∀ ε > 0, ∃ N, ∀ n ≥ N, |(2 * n^2 / (n + 2) - n * a) - b| < ε) :
  a = 2 ∧ b = 4 := sorry

-- Part 2
theorem problem2 (h : ∀ ε > 0, ∃ N, ∀ n ≥ N, |(3^n / (3^(n + 1) + (a + 1)^n) - 1/3)| < ε) :
  -4 < a ∧ a < 2 := sorry

end problem1_problem2_l427_427244


namespace larger_group_men_count_l427_427837

open Real

theorem larger_group_men_count :
  (∃ men1 hectares days daily_rate,
    men1 = 10 ∧ 
    hectares = 80 ∧ 
    days = 24 ∧ 
    daily_rate = hectares / days) ∧ 
  (∃ hectares' days' men' daily_rate',
    hectares' = 360 ∧ 
    days' = 30 ∧ 
    (daily_rate' = hectares' / days' ∧
     (hectares' = (daily_rate' * days' := 10 * (hectares / days)) = 30 * x )) ∧ 
    men' = men1 * 3.6 ∧
    men' = 36) :=
sorry

end larger_group_men_count_l427_427837


namespace minimum_socks_to_pair_l427_427862

theorem minimum_socks_to_pair
    (red_socks : ℕ) (blue_socks : ℕ) (total_socks : ℕ) :
    red_socks = 10 ∧ blue_socks = 10 ∧ total_socks = red_socks + blue_socks →
    ∃ n, n = 3 ∧ (∀ taken, taken ⊆ {sock // sock < total_socks} →
    (∃ same_color, ∃ m, m ≥ 2 ∧ m ≤ n ∧ same_color = red ∨ same_color = blue ∧
    ∀ i j, i ≠ j → i ∈ taken → j ∈ taken → (sock_color i = same_color ∧ sock_color j = same_color))) :=
begin
    sorry
end

end minimum_socks_to_pair_l427_427862


namespace sum_of_fibonacci_factorial_last_two_digits_sum_of_digits_of_fifty_l427_427713

def factorial_last_two_digits (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | 4 => 24 % 100
  | 5 => 120 % 100
  | 6 => 720 % 100
  | 7 => 5040 % 100
  | 8 => 40320 % 100
  | 9 => 362880 % 100
  | 10 => 10 * factorial_last_two_digits 9 % 100
  | _ => 0

def sum_last_two_digits := (factorial_last_two_digits 1 +
                            factorial_last_two_digits 1 +
                            factorial_last_two_digits 2 +
                            factorial_last_two_digits 3 +
                            factorial_last_two_digits 5 +
                            factorial_last_two_digits 8 +
                            factorial_last_two_digits 13 +
                            factorial_last_two_digits 21 +
                            factorial_last_two_digits 34 +
                            factorial_last_two_digits 55 +
                            factorial_last_two_digits 55) % 100

theorem sum_of_fibonacci_factorial_last_two_digits : sum_last_two_digits = 50 :=
by sorry

theorem sum_of_digits_of_fifty : Nat.digits 10 50.sum = 5 :=
by sorry

end sum_of_fibonacci_factorial_last_two_digits_sum_of_digits_of_fifty_l427_427713


namespace obtuse_angle_of_parallel_vectors_l427_427379

noncomputable def is_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem obtuse_angle_of_parallel_vectors (θ : ℝ) :
  let a := (2, 1 - Real.cos θ)
  let b := (1 + Real.cos θ, 1 / 4)
  is_parallel a b → 90 < θ ∧ θ < 180 → θ = 135 :=
by
  intro ha hb
  sorry

end obtuse_angle_of_parallel_vectors_l427_427379


namespace base_six_representation_l427_427373

theorem base_six_representation (b : ℕ) (h₁ : b = 6) :
  625₁₀.toDigits b = [2, 5, 2, 1] ∧ (625₁₀.toDigits b).length = 4 ∧ (625₁₀.toDigits b).head % 2 = 1 :=
by
  sorry

end base_six_representation_l427_427373


namespace geom_seq_question_l427_427875

noncomputable def geom_seq (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

theorem geom_seq_question
  (a₁ q : ℝ)
  (h1 : 0 < a₁)
  (h2 : a₃ = 2 ∨ a₃ = 4 ∧ a₁ * q^2 > 0)
  (h3 : a₃ = geom_seq a₁ q 3)
  (h4 : a₁ * (q ^ 2)^6 = 2 ∨ a₁ * (q ^ 2)^6 = 4) :
  (a_3 := geom_seq a₁ q 3)
  (a₉ := geom_seq a₁ q 9)
  (a₁₇ := geom_seq a₁ q 17) :
  (a₁ * a₁₇ / a₉) = 2 * Real.sqrt 2 :=
sorry

end geom_seq_question_l427_427875


namespace combined_molecular_weight_l427_427186

theorem combined_molecular_weight {m1 m2 : ℕ} 
  (MW_C : ℝ) (MW_H : ℝ) (MW_O : ℝ)
  (Butanoic_acid : ℕ × ℕ × ℕ)
  (Propanoic_acid : ℕ × ℕ × ℕ)
  (MW_Butanoic_acid : ℝ)
  (MW_Propanoic_acid : ℝ)
  (weight_Butanoic_acid : ℝ)
  (weight_Propanoic_acid : ℝ)
  (total_weight : ℝ) :
MW_C = 12.01 → MW_H = 1.008 → MW_O = 16.00 →
Butanoic_acid = (4, 8, 2) → MW_Butanoic_acid = (4 * MW_C) + (8 * MW_H) + (2 * MW_O) →
Propanoic_acid = (3, 6, 2) → MW_Propanoic_acid = (3 * MW_C) + (6 * MW_H) + (2 * MW_O) →
m1 = 9 → weight_Butanoic_acid = m1 * MW_Butanoic_acid →
m2 = 5 → weight_Propanoic_acid = m2 * MW_Propanoic_acid →
total_weight = weight_Butanoic_acid + weight_Propanoic_acid →
total_weight = 1163.326 :=
by {
  intros;
  sorry
}

end combined_molecular_weight_l427_427186


namespace hexagon_area_l427_427902

noncomputable theory

def trapezoid_ABCD (AB : ℝ) (BC : ℝ) (CD : ℝ) (DA : ℝ) : Prop :=
  AB = 13 ∧ BC = 7 ∧ CD = 23 ∧ DA = 9

def is_parallel (AB CD : ℝ) : Prop :=
  true  -- Representing the fact that AB is parallel to CD

def angle_bisectors (angle_A angle_D : ℝ) : Prop :=
  angle_A = 30 ∧ angle_D = 60

def area_of_hexagon (AB : ℝ) (BC : ℝ) (CD : ℝ) (DA : ℝ) (angle_A angle_D : ℝ) : ℝ :=
  if (trapezoid_ABCD AB BC CD DA ∧ is_parallel AB CD ∧ angle_bisectors angle_A angle_D)
  then 14 * real.sqrt 37.44
  else 0

theorem hexagon_area :
  area_of_hexagon 13 7 23 9 30 60 = 14 * real.sqrt 37.44 :=
by
  unfold area_of_hexagon
  rw [if_pos]
  {simp}
  {repeat {split}; norm_num}

end hexagon_area_l427_427902


namespace possible_values_of_b_l427_427968

theorem possible_values_of_b (b : ℝ) (h : ∃ x y : ℝ, y = 2 * x + b ∧ y > 0 ∧ x = 0) : b > 0 :=
sorry

end possible_values_of_b_l427_427968


namespace original_apples_l427_427264

-- Define the conditions
def sells_50_percent (initial remaining : ℕ) : Prop :=
  (initial / 2) = remaining

-- Define the goal
theorem original_apples (remaining : ℕ) (initial : ℕ) (h : sells_50_percent initial remaining) : initial = 10000 :=
by
  sorry

end original_apples_l427_427264


namespace complex_modulus_eq_one_l427_427775

open Complex

theorem complex_modulus_eq_one (a b : ℝ) (h : (1 + 2 * Complex.I) / (a + b * Complex.I) = 2 - Complex.I) :
  abs (a - b * Complex.I) = 1 := by
  sorry

end complex_modulus_eq_one_l427_427775


namespace expression_eq_16x_l427_427026

variable (x y z w : ℝ)

theorem expression_eq_16x
  (h1 : y = 2 * x)
  (h2 : z = 3 * y)
  (h3 : w = z + x) :
  x + y + z + w = 16 * x :=
sorry

end expression_eq_16x_l427_427026


namespace lines_concurrent_l427_427287

section Geometry

variables {A B C P Q M N D E U V Z : Type}
variable [Point A B C P Q M N D E U V Z]

-- Given conditions
axiom triangle_abc : Triangle ABC
axiom angle_bisectors_intersect : ∀ (A B C P Q M N : Point) (hTriangle : Triangle ABC), 
  BissectAngle ∠A P ∧ BissectAngle ∠B Q ∧ 
  (IntersectsCircle P A ∧ IntersectsCircle Q B) ∧ 
  Parallel CA BQ ∧ Parallel CB AP
axiom circumcenters : ∀ (BME AND CPQ : Triangle) (U V Z : Point),
  Circumcenter BME U ∧ Circumcenter AND V ∧ Circumcenter CPQ Z

-- Goal
theorem lines_concurrent : Concurrent B U A V C Z :=
sorry

end Geometry

end lines_concurrent_l427_427287


namespace supplement_of_complement_of_35_degree_angle_l427_427188

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_of_35_degree_angle : 
  supplement (complement 35) = 125 := 
by sorry

end supplement_of_complement_of_35_degree_angle_l427_427188


namespace power_plus_sqrt5_diff_divisible_by_p_l427_427106

noncomputable def power_plus_sqrt5_diff (p : ℕ) : ℤ :=
  real.to_real(((2 : ℤ) + real.sqrt 5) ^ p) - 2 ^ (p + 1)

theorem power_plus_sqrt5_diff_divisible_by_p (p : ℕ) [fact p.prime] (h : p > 2) : p ∣ power_plus_sqrt5_diff p :=
sorry

end power_plus_sqrt5_diff_divisible_by_p_l427_427106


namespace max_n_for_factored_polynomial_l427_427331

theorem max_n_for_factored_polynomial :
  ∃ (n : ℤ), (∀ (A B : ℤ), 6 * 72 + n = 6 * B + A ∧ A * B = 72 → n ≤ 433) ∧
             (∃ (A B : ℤ), 6 * B + A = 433 ∧ A * B = 72) :=
by sorry

end max_n_for_factored_polynomial_l427_427331


namespace quadratic_modulus_condition_l427_427818

theorem quadratic_modulus_condition (a : ℝ) (h1 : a * complex.i ^ 2 + complex.i + 1 = 0) 
(h2 : ∀ x : ℂ, x ∈ (complex.roots (λ x, a * x ^ 2 + x + 1)) → |x| < 1) : 1 < a :=
by sorry

end quadratic_modulus_condition_l427_427818


namespace a1000_value_l427_427484

theorem a1000_value :
  (∑ i in (Finset.Ico 1 (1001)), x^i / (1 - x^i) = ∑ i in Finset.range 1001, a i * x^i) → 
  a 1000 = 16 :=
by sorry

end a1000_value_l427_427484


namespace material_mix_ratio_l427_427659

variable (x y : ℤ)

theorem material_mix_ratio 
  (h1 : 50 * x + 40 * y = 50 * (1 + 0.1) * x + 40 * (1 - 0.15) * y) :
  x / y = 6 / 5 :=
by sorry

end material_mix_ratio_l427_427659


namespace salt_cups_l427_427925

theorem salt_cups (S : ℕ) (h1 : 8 = S + 1) : S = 7 := by
  -- Problem conditions
  -- 1. The recipe calls for 8 cups of sugar.
  -- 2. Mary needs to add 1 more cup of sugar than cups of salt.
  -- This corresponds to h1.

  -- Prove S = 7
  sorry

end salt_cups_l427_427925


namespace opposite_of_neg_two_l427_427986

theorem opposite_of_neg_two : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_two_l427_427986


namespace octagon_area_l427_427273

-- Define that we are dealing with a regular octagon
def regular_octagon (r : ℝ) (n : ℕ) := n = 8 ∧ r = 2

-- State the problem: prove the area of this regular octagon is 8√2
theorem octagon_area (r : ℝ) (n : ℕ) : regular_octagon r n → n = 8 → r = 2 → 8√2 = 16 - 8*(2) := sorry

end octagon_area_l427_427273


namespace total_distance_travelled_l427_427100

theorem total_distance_travelled (D : ℝ) (h1 : (D / 2) / 30 + (D / 2) / 25 = 11) : D = 150 :=
sorry

end total_distance_travelled_l427_427100


namespace proof_mod_55_l427_427490

theorem proof_mod_55 (M : ℕ) (h1 : M % 5 = 3) (h2 : M % 11 = 9) : M % 55 = 53 := 
  sorry

end proof_mod_55_l427_427490


namespace find_all_functions_l427_427900

theorem find_all_functions (f : ℕ → ℕ) : 
  (∀ a b : ℕ, 0 < a → 0 < b → f (a^2 + b^2) = f a * f b) →
  (∀ a : ℕ, 0 < a → f (a^2) = f a ^ 2) →
  (∀ n : ℕ, 0 < n → f n = 1) :=
by
  intros h1 h2 a ha
  sorry

end find_all_functions_l427_427900


namespace equal_angles_implies_inequality_l427_427107

theorem equal_angles_implies_inequality (n : ℕ) (P : fin n → euclidean_space ℝ (fin 2)) 
  (convex : convex_hull ℝ (set.range P)) 
  (equal_angles : ∀ i, angle (P i) (P (i+1) % n) = 2 * real.pi / n) : 
  ∃ i, dist (P i) (P ((i+1) % n)) ≤ dist (P ((i+1) % n)) (P ((i+2) % n)) :=
sorry

end equal_angles_implies_inequality_l427_427107


namespace solution_l427_427283

noncomputable def problem : Prop :=
  let A := -6
  let B := 0
  let C := 0.2
  let D := 3
  A < 0 ∧ ¬ (B < 0) ∧ ¬ (C < 0) ∧ ¬ (D < 0)

theorem solution : problem := 
by
  let A := -6
  let B := 0
  let C := 0.2
  let D := 3
  exact ⟨ by norm_num, by norm_num, by norm_num, by norm_num ⟩

end solution_l427_427283


namespace angle_of_inclination_135_l427_427960

theorem angle_of_inclination_135 (a b c : ℝ) : 
  ∃ θ : ℝ, θ = 135 ∧ (∃ x y : ℝ, equation : ax - by + c = 0) :=
sorry

end angle_of_inclination_135_l427_427960


namespace circular_divisors_classification_l427_427737
noncomputable theory

def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ n = a * b

def has_divisors_in_circle (n : ℕ) : Prop :=
  ∃ (D : list ℕ), (∀ d ∈ D, d ∣ n ∧ d > 1) ∧ 
  (∀ i, i < D.length → nat.coprime (D.nth_le i sorry) (D.nth_le ((i + 1) % D.length) sorry) → false)

theorem circular_divisors_classification (n : ℕ) :
  (is_composite n ∧ has_divisors_in_circle n) →
  (∃ p : ℕ, ∃ α : ℕ, nat.prime p ∧ α ≥ 2 ∧ n = p^α) ∨ 
  (∃ k : ℕ, k > 1 ∧ (∃ L : list ℕ, L.length = k ∧ ∀ p ∈ L, nat.prime p ∧ n = L.prod)) :=
sorry

end circular_divisors_classification_l427_427737


namespace number_of_digits_sum_l427_427248

noncomputable def log10_2 : ℝ := 0.3010
noncomputable def log10_5 : ℝ := 0.6990
def num_digits (x : ℝ) := (Real.log10 x).floor + 1

theorem number_of_digits_sum :
  let a := num_digits (2 ^ 1998)
  let b := num_digits (5 ^ 1998) in
  a + b = 1999 :=
by
  let a := (1998 * log10_2).floor + 1
  let b := (1998 * log10_5).floor + 1
  have ha : a = 602 := by
    have : 1998 * 0.3010 = 601.398 := by norm_num
    have : (601.398 : ℝ).floor + 1 = 602 := by norm_num
    simp [a]
  have hb : b = 1397 := by
    have : 1998 * 0.6990 = 1396.602 := by norm_num
    have : (1396.602 : ℝ).floor + 1 = 1397 := by norm_num
    simp [b]
  rw [ha, hb]
  norm_num

  sorry

end number_of_digits_sum_l427_427248


namespace real_values_satisfy_eq_l427_427158

theorem real_values_satisfy_eq (x y : ℝ) (h : x - 3 * complex.I = (8 * x - y) * complex.I) : x = 0 ∧ y = 3 :=
by
  sorry

end real_values_satisfy_eq_l427_427158


namespace find_inverse_condition_l427_427832

variable {α β : Type} [inv : InverseFunction α β]

def condition (f g : α → β) (f_inv : β → α) : Prop :=
  ∀ x, f_inv (g x) = x^4 - 1

theorem find_inverse_condition (f g : α → β) (f_inv : β → α) 
  (h₁ : condition f g f_inv) (h₂ : has_inverse g) : g⁻¹ (f 15) = 2 := 
sorry

end find_inverse_condition_l427_427832


namespace solve_for_z_l427_427946

theorem solve_for_z (z : ℂ) (h : 3 - 2 * complex.I * z = -3 + 2 * complex.I * z) : 
  z = -3 * complex.I / 2 :=
by
  sorry

end solve_for_z_l427_427946


namespace gnome_weight_with_shoes_l427_427669

noncomputable theory

variable (W_g W_s : ℕ) -- weights of the gnome without shoes and the shoes respectively

-- Condition 1: A gnome with shoes weighs 2 kg more than a gnome without shoes
axiom condition1 : W_s = 2

-- Condition 2: The total weight of five gnomes in shoes and five gnomes without shoes is 330 kg
axiom condition2 : 5 * (W_g + W_s) + 5 * W_g = 330

-- Let's prove the weight of a gnome with shoes
theorem gnome_weight_with_shoes : W_g + W_s = 34 :=
by
  -- The filled proof is skipped
  sorry

end gnome_weight_with_shoes_l427_427669


namespace christine_stickers_l427_427298

theorem christine_stickers (stickers_has stickers_needs : ℕ) (h_has : stickers_has = 11) (h_needs : stickers_needs = 19) : 
  stickers_has + stickers_needs = 30 :=
by 
  sorry

end christine_stickers_l427_427298


namespace sum_of_extreme_numbers_is_846_l427_427638

theorem sum_of_extreme_numbers_is_846 :
  let digits := [0, 2, 4, 6]
  let is_valid_hundreds_digit (d : Nat) := d ≠ 0
  let create_three_digit_number (h t u : Nat) := h * 100 + t * 10 + u
  let max_num := create_three_digit_number 6 4 2
  let min_num := create_three_digit_number 2 0 4
  max_num + min_num = 846 := by
  sorry

end sum_of_extreme_numbers_is_846_l427_427638


namespace Kenneth_money_left_l427_427898

def initial_amount : ℕ := 50
def number_of_baguettes : ℕ := 2
def cost_per_baguette : ℕ := 2
def number_of_bottles : ℕ := 2
def cost_per_bottle : ℕ := 1

-- This theorem states that Kenneth has $44 left after his purchases.
theorem Kenneth_money_left : initial_amount - (number_of_baguettes * cost_per_baguette + number_of_bottles * cost_per_bottle) = 44 := by
  sorry

end Kenneth_money_left_l427_427898


namespace pqrs_sum_l427_427513

theorem pqrs_sum (p q r s : ℝ)
  (h1 : (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 → x = r ∨ x = s))
  (h2 : (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 → x = p ∨ x = q))
  (h3 : p ≠ q) (h4 : p ≠ r) (h5 : p ≠ s) (h6 : q ≠ r) (h7 : q ≠ s) (h8 : r ≠ s) :
  p + q + r + s = 2028 :=
sorry

end pqrs_sum_l427_427513


namespace peter_faster_than_kristin_l427_427541

theorem peter_faster_than_kristin :
  (let peter_time_per_book := 18
          kristin_books_read := 10
          kristin_total_time := 540 in
   let kristin_time_per_book := kristin_total_time / kristin_books_read in
   (kristin_time_per_book / peter_time_per_book) = 3) :=
by
  sorry

end peter_faster_than_kristin_l427_427541


namespace _l427_427377

variables {A B C D A0 B0 C0 A1 B1 C1 A2 B2 C2 A3 C3 : Type}
variables [P : A ≠ B] [Q : A ≠ C] [R : B ≠ C] 

def midpoint (X Y Z : Type) := sorry
def intersection (X Y Z W : Type) := sorry
def parallel_lines (X Y Z W : Type) := sorry

-- Conditions
noncomputable def conditions 
  (ΔABC : A ≠ B ∧ B ≠ C ∧ C ≠ A)
  (D_inside : true)
  (A0 : intersection A D B C)
  (B0 : intersection B D A C)
  (C0 : intersection C D A B)
  (A1 : midpoint B C A)
  (B1 : midpoint A C B)
  (C1 : midpoint A B C)
  (A2 : midpoint A D B)
  (B2 : midpoint B D A)
  (C2 : midpoint C D A)
  (A3 : intersection (parallel_lines A1 A2 B0 C2) B1 B2)
  (C3 : intersection (parallel_lines C1 C2 B0 A2) B1 B2)
  : Prop :=
-- The theorem statement
  ∃ (A3 C3 : Type), 
    (A3 ≠ B1 ∧ A3 ≠ B2 ∧ C3 ≠ B1 ∧ C3 ≠ B2 ∧
    (by ∃ (B1 B2 : Type), (A3 ≠ B1) ∧ (C3 ≠ B2) ∧
      (by ∃ (A1 A2 C1 C2 : Type), (parallel_lines A1 C2 B0 C2) ∧
          parallel_lines C1 C2 B0 A1) ∧
      ((∃ (A3 B1 A3 B2 : Type), (A3 ≠ B1) ∧ (A3 ≠ B2)) ∧
       (∃ (C3 B1 C3 B2 : Type), (C3 ≠ B1) ∧ (C3 ≠ B2) ∧
        (∃ (A1 C1 A2 C2 : Type), 
          parallel_lines (midpoint B C A) (midpoint A D B) B0 C2) ∧
          ∃ (A3 B1 A3 B2 : Type), 
          ∃ (C3 B1 C3 B2 : Type), 
          ∃ (cross_ratio : A3 ≠ B1) ∧ 
           (cross_ratio (A3 ≠ B2)) ∧ 
           (cross_ratio (C3 ≠ B1)) ∧
           (cross_ratio (C3 ≠ B2)))))) sorry


end _l427_427377


namespace max_min_diff_c_l427_427910

theorem max_min_diff_c (a b c : ℝ) 
  (h1 : a + b + c = 3) 
  (h2 : a^2 + b^2 + c^2 = 18) : 
  ∃ c_max c_min, 
  (∀ c', (a + b + c' = 3 ∧ a^2 + b^2 + c'^2 = 18) → c_min ≤ c' ∧ c' ≤ c_max) 
  ∧ (c_max - c_min = 6) :=
  sorry

end max_min_diff_c_l427_427910


namespace cos_double_angle_neg_three_fifths_l427_427776

noncomputable def cos_double_angle (θ : ℝ) : ℝ :=
1 - (tan θ) ^ 2 / (1 + (tan θ) ^ 2)

theorem cos_double_angle_neg_three_fifths (θ : ℝ) (h1 : tan θ = 2) (h2 : 0 < θ ∧ θ < π / 2) : 
  cos_double_angle θ = -3 / 5 :=
by
  sorry

end cos_double_angle_neg_three_fifths_l427_427776


namespace time_to_cover_remaining_distance_l427_427633

def movement_rate (distance_feet time_minutes : ℕ) : ℕ := 
    distance_feet / time_minutes

def remaining_distance_yards (distance_yards : ℕ) : ℕ := 
    distance_yards * 3

theorem time_to_cover_remaining_distance :
  movement_rate 40 20 = 2 ∧ remaining_distance_yards 90 = 270 →
  270 / 2 = 135 := 
by 
  intros h
  cases h with h1 h2
  rw [h1, h2]
  norm_num
  sorry

end time_to_cover_remaining_distance_l427_427633


namespace log_fixed_point_l427_427135

theorem log_fixed_point (a : ℝ) (h_pos : 0 < a) (h_neq : a ≠ 1) : ∃ x y : ℝ, (x = 1/2) ∧ (y = 0) ∧ (y = log a (4 * x - 1)) :=
by
  sorry

end log_fixed_point_l427_427135


namespace johns_squat_before_training_l427_427056

theorem johns_squat_before_training 
  (initial_increase : ℕ) 
  (bracer_multiplier : ℕ) 
  (final_squat : ℕ) 
  (h_init_inc : initial_increase = 265) 
  (h_bracer_mul : bracer_multiplier = 7) 
  (h_final_squat : final_squat = 2800) :
  ∃ W : ℕ, W + initial_increase = 400 ∧ W = 135 :=
begin
  let W := 135,
  have hW : W + initial_increase = 400,
  { rw initial_increase, exact nat.add_sub_cancel' 400 265, },
  exact ⟨W, ⟨hW, rfl⟩⟩,
end

end johns_squat_before_training_l427_427056


namespace find_AE_l427_427867

theorem find_AE (A B C D E : Type) [LinearOrderSpace A] 
  (h1 : RightTriangle A B C)
  (h2 : IsPerpendicular C D B)
  (h3 : IsPerpendicular A E C)
  (hAB : length A B = 25)
  (hBC : length B C = 20)
  (hCD : length C D = 12) :
  length A E = 9.6 :=
sorry

end find_AE_l427_427867


namespace opposite_of_neg_three_l427_427997

theorem opposite_of_neg_three : -(-3) = 3 :=
by 
  sorry

end opposite_of_neg_three_l427_427997


namespace odd_prime_form_l427_427944

theorem odd_prime_form (p x y v : ℤ) (h_prime : Nat.Prime p)
  (h_odd : p % 2 = 1)
  (h_expr : p = x^5 - y^5) :
  (∃ v, v % 2 = 1 ∧ (Int.sqrt ((4 * p + 1) / 5)) = (v^2 + 1) / 2) :=
sorry

end odd_prime_form_l427_427944


namespace a_n_strictly_monotonic_increasing_l427_427725

noncomputable def a_n (n : ℕ) : ℝ := 
  2 * ((1 + 1 / (n : ℝ)) ^ (2 * n + 1)) / (((1 + 1 / (n : ℝ)) ^ n) + ((1 + 1 / (n : ℝ)) ^ (n + 1)))

theorem a_n_strictly_monotonic_increasing : ∀ n : ℕ, a_n (n + 1) > a_n n :=
sorry

end a_n_strictly_monotonic_increasing_l427_427725


namespace restaurant_table_difference_l427_427640

theorem restaurant_table_difference :
  ∃ (N O : ℕ), N + O = 40 ∧ 6 * N + 4 * O = 212 ∧ (N - O) = 12 :=
by
  sorry

end restaurant_table_difference_l427_427640


namespace advertising_customers_l427_427891

theorem advertising_customers 
  (advertising_cost : ℝ) (purchase_percentage : ℝ) (item_cost : ℝ) (profit : ℝ)
  (h1 : advertising_cost = 1000)
  (h2 : purchase_percentage = 0.80)
  (h3 : item_cost = 25)
  (h4 : profit = 1000) :
  let C := 2000 / (purchase_percentage * item_cost) in
  C = 100 := by
{
  sorry
}

end advertising_customers_l427_427891


namespace inscribed_circle_radius_l427_427622

theorem inscribed_circle_radius (DE DF EF : ℝ) (hDE : DE = 8) (hDF : DF = 10) (hEF : EF = 12) :
  let s := (DE + DF + EF) / 2 in
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF)) in
  let r := K / s in
  r = Real.sqrt 7 :=
by {
  sorry
}

end inscribed_circle_radius_l427_427622


namespace soda_per_can_is_12_l427_427893

def cans_weight (empty_weight_per_can total_cans : ℕ) := empty_weight_per_can * total_cans
def total_weight (bridge_weight empty_cans_weight : ℕ) := bridge_weight - empty_cans_weight
def soda_per_can (soda_weight filled_cans : ℕ) := soda_weight / filled_cans

theorem soda_per_can_is_12
  (empty_weight_per_can : ℕ)
  (filled_cans additional_empty_cans bridge_weight : ℕ)
  (h_empty_weight : empty_weight_per_can = 2)
  (h_filled_cans : filled_cans = 6)
  (h_additional_empty_cans : additional_empty_cans = 2)
  (h_bridge_weight : bridge_weight = 88)
  :
  soda_per_can (total_weight bridge_weight (cans_weight empty_weight_per_can (filled_cans + additional_empty_cans))) filled_cans = 12 :=
by
  let empty_cans_weight := cans_weight empty_weight_per_can (filled_cans + additional_empty_cans)
  have h_empty_cans_weight : empty_cans_weight = 16 := by
    simp [cans_weight, h_empty_weight, h_filled_cans, h_additional_empty_cans]
  let remaining_weight := total_weight bridge_weight empty_cans_weight
  have h_remaining_weight : remaining_weight = 72 := by
    simp [total_weight, h_bridge_weight, h_empty_cans_weight]
  have h_soda_per_can := soda_per_can remaining_weight filled_cans
  have h_soda_per_can_eq := by
    simp [soda_per_can, h_remaining_weight, h_filled_cans]
  exact h_soda_per_can_eq.trans (by norm_num)

end soda_per_can_is_12_l427_427893


namespace find_y_l427_427045

noncomputable def angle_ABC := 75
noncomputable def angle_BAC := 70
noncomputable def angle_CDE := 90
noncomputable def angle_BCA : ℝ := 180 - (angle_ABC + angle_BAC)
noncomputable def y : ℝ := 90 - angle_BCA

theorem find_y : y = 55 :=
by
  have h1: angle_BCA = 180 - (75 + 70) := rfl
  have h2: y = 90 - angle_BCA := rfl
  rw [h1] at h2
  exact h2.trans (by norm_num)

end find_y_l427_427045


namespace contradictory_goldbach_l427_427133

theorem contradictory_goldbach : ¬ (∀ n : ℕ, 2 < n ∧ Even n → ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q) :=
sorry

end contradictory_goldbach_l427_427133


namespace part1_part2_l427_427811

noncomputable def problem := 
  let a : ℝ := 1
  let b : ℝ := Real.sqrt 2
  let e : ℝ := Real.sqrt 3
  let x0 y0 : ℝ 
  let S : ℝ := Real.sqrt 2
  ∃ (P : ℝ × ℝ), 
    (P = (x0, y0)) ∧ 
    ((P.1^2) / a^2 - (P.2^2) / b^2 = 1) → 
    let l := λ (x y : ℝ), (x0 * x) / a^2 - (y0 * y) / b^2 = 1
    ∃ (M N : ℝ × ℝ), 
      (l M.1 M.2) ∧ 
      (l N.1 N.2) ∧ 
      (M ≠ P ∧ N ≠ P) ∧ 
      (let S := (Real.sqrt 2)
        (∀ (λ : ℝ), vector P M = λ * vector P N → 
          let c := 2 * S
          (2 * Real.sqrt 2 = c)))

theorem part1 (hprob : problem) : 
  let S := Real.sqrt 2,
  ∃ (x y : ℝ), 
    ∀ (M N : ℝ × ℝ), 
      (S = Real.sqrt 2) :=
sorry

theorem part2 (hprob : problem) : 
  let λ : ℝ, 
  let S := Real.sqrt 2,
  ∃ (x1 x2 : ℝ), 
    ∃ (y1 y2 : ℝ), 
      (λ * S = λ * Real.sqrt 2) :=
sorry

end part1_part2_l427_427811


namespace part1_part2_l427_427836

-- Lean statement for part 1:
theorem part1 (x : ℝ) (h1 : x + x⁻¹ = 3) (h2 : 0 < x ∧ x < 1) :
  x^2 + x⁻² = 7 :=
sorry

-- Lean statement for part 2:
theorem part2 (x : ℝ) (h1 : x + x⁻¹ = 3) (h2 : 0 < x ∧ x < 1) :
  x^(3/2) - x^(-3/2) = -4 :=
sorry

end part1_part2_l427_427836


namespace largest_value_of_n_l427_427351

theorem largest_value_of_n (A B n : ℤ) (h1 : A * B = 72) (h2 : n = 6 * B + A) : n = 433 :=
sorry

end largest_value_of_n_l427_427351


namespace perfect_square_iff_form_l427_427757

-- Defining the function to check if a natural number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

-- Defining the function for the given expression
def factorial_product (n : ℕ) : ℕ :=
  (List.range (2 * n + 1)).map Nat.factorial |> List.product

-- Defining the main expression
def given_expression (n : ℕ) : ℕ :=
  factorial_product n / Nat.factorial (n + 1)

-- The proof problem
theorem perfect_square_iff_form (n : ℕ) :
  (is_perfect_square (given_expression n)) ↔
    (∃ k : ℕ, n = 4 * k * (k + 1)) ∨ (∃ k : ℕ, n = 2 * k^2 - 1) := by
  sorry

end perfect_square_iff_form_l427_427757


namespace matrix_A_cubed_identity_has_27_solutions_l427_427494

theorem matrix_A_cubed_identity_has_27_solutions :
  let ω := Complex.exp (2 * Real.pi * Complex.I / 3)
  ∃ (A : Matrix (Fin 3) (Fin 3) ℂ), A^3 = Matrix.id ∧ ∃! (n : ℕ), n = 27 :=
by
  sorry

end matrix_A_cubed_identity_has_27_solutions_l427_427494


namespace sum_distinct_vars_eq_1716_l427_427505

open Real

theorem sum_distinct_vars_eq_1716 (p q r s : ℝ) (hpqrs_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : r + s = 12 * p)
  (h2 : r * s = -13 * q)
  (h3 : p + q = 12 * r)
  (h4 : p * q = -13 * s) :
  p + q + r + s = 1716 :=
sorry

end sum_distinct_vars_eq_1716_l427_427505


namespace lambda_n_irrational_for_all_non_negative_integers_l427_427732

theorem lambda_n_irrational_for_all_non_negative_integers (n : ℕ) :
  irrational (Real.sqrt (3 * n^2 + 2 * n + 2)) :=
sorry

end lambda_n_irrational_for_all_non_negative_integers_l427_427732


namespace total_mass_of_fruit_l427_427700

theorem total_mass_of_fruit :
  let num_apple_trees := 30 in
  let mass_per_apple_tree := 150 in
  let num_peach_trees := 45 in
  let mass_per_peach_tree := 65 in
  let total_mass_apples := num_apple_trees * mass_per_apple_tree in
  let total_mass_peaches := num_peach_trees * mass_per_peach_tree in
  total_mass_apples + total_mass_peaches = 7425 :=
by
  sorry

end total_mass_of_fruit_l427_427700


namespace factorial_sequence_perfect_square_l427_427761

-- Definitions based on the conditions
def is_perf_square (x : ℕ) : Prop := ∃ (k : ℕ), x = k * k

def factorial (n : ℕ) : ℕ := Nat.recOn n 1 (λ n fac_n, (n + 1) * fac_n)

def factorial_seq_prod (n : ℕ) : ℕ :=
  Nat.recOn n 1 (λ n prodN, factorial (2 * n) * prodN)

-- Main statement
theorem factorial_sequence_perfect_square (n : ℕ) :
  is_perf_square (factorial_seq_prod n / factorial (n + 1)) ↔
  ∃ k : ℕ, n = 4 * k * (k + 1) ∨ n = 2 * k ^ 2 - 1 := by
  sorry

end factorial_sequence_perfect_square_l427_427761


namespace power_mod_eq_nine_l427_427949

theorem power_mod_eq_nine :
  ∃ n : ℕ, 13^6 ≡ n [MOD 11] ∧ 0 ≤ n ∧ n < 11 ∧ n = 9 :=
by
  sorry

end power_mod_eq_nine_l427_427949


namespace inequality_proof_l427_427913

theorem inequality_proof
  (n : ℕ) 
  (h1 : n ≥ 3)
  (a : ℕ → ℝ)
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → 2 ≤ a i ∧ a i ≤ 3) 
  (S : ℝ := ∑ i in finset.range n, a i):
  (∑ i in finset.range n, (a i)^2 + (a (i + 1 % n))^2 - (a (i + 2 % n))^2 / (a i + a (i + 1 % n) - a (i + 2 % n))) 
  ≤ 2 * S - 2 * n :=
by sorry

end inequality_proof_l427_427913


namespace max_value_of_n_l427_427341

-- Define the main variables and conditions
noncomputable def max_n : ℕ := 
  let n_pairs := [(1, 72), (2, 36), (3, 24), (4, 18), (6, 12), (8, 9)] 
  max (n_pairs.map (λ p, 6 * p.2 + p.1))

-- Lean theorem statement for the equivalence
theorem max_value_of_n :
  max_n = 433 := by
  sorry

end max_value_of_n_l427_427341


namespace perfect_square_iff_form_l427_427758

-- Defining the function to check if a natural number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

-- Defining the function for the given expression
def factorial_product (n : ℕ) : ℕ :=
  (List.range (2 * n + 1)).map Nat.factorial |> List.product

-- Defining the main expression
def given_expression (n : ℕ) : ℕ :=
  factorial_product n / Nat.factorial (n + 1)

-- The proof problem
theorem perfect_square_iff_form (n : ℕ) :
  (is_perfect_square (given_expression n)) ↔
    (∃ k : ℕ, n = 4 * k * (k + 1)) ∨ (∃ k : ℕ, n = 2 * k^2 - 1) := by
  sorry

end perfect_square_iff_form_l427_427758


namespace opposite_of_neg_two_l427_427993

theorem opposite_of_neg_two : -(-2) = 2 := 
by 
  sorry

end opposite_of_neg_two_l427_427993


namespace opposite_of_neg_two_l427_427995

theorem opposite_of_neg_two : -(-2) = 2 := 
by 
  sorry

end opposite_of_neg_two_l427_427995


namespace median_length_AD_l427_427827

open Real

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def midpoint (p1 p2 : Point3D) : Point3D :=
  { x := (p1.x + p2.x) / 2
  , y := (p1.y + p2.y) / 2
  , z := (p1.z + p2.z) / 2 }

def distance (p1 p2 : Point3D) : ℝ :=
  sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

def A : Point3D := { x := 2, y := -1, z := 4 }
def B : Point3D := { x := 3, y := 2, z := -6 }
def C : Point3D := { x := -5, y := 0, z := 2 }
def D : Point3D := midpoint B C

theorem median_length_AD : distance A D = 7 := by
  sorry

end median_length_AD_l427_427827


namespace other_five_say_equal_numbers_l427_427096

noncomputable def knights_and_liars_problem : Prop :=
  ∃ (K L : ℕ), K + L = 10 ∧
  ∀ (x : ℕ), (x < 5 → "There are more liars" = true) ∨ (x >= 5 → "There are equal numbers of knights and liars" = true)

theorem other_five_say_equal_numbers :
  knights_and_liars_problem :=
sorry

end other_five_say_equal_numbers_l427_427096


namespace smaller_tv_diagonal_l427_427598

theorem smaller_tv_diagonal
  (area_diff : ℝ) (diag_large_tv : ℝ) (diag_small_tv : ℝ) (side_large_tv : ℝ) (side_small_tv : ℝ)
  (h_area_diff : area_diff = 143.5)
  (h_diag_large_tv : diag_large_tv = 24)
  (h_side_large_tv : side_large_tv = (24 / (2^(1/2))))
  (h_area_eq : side_large_tv^2 = side_small_tv^2 + area_diff)
  (h_side_small_tv : side_small_tv = (diag_small_tv / (2^(1/2))))
  (h_diag_small_tv : diag_small_tv = 17) :
  diag_small_tv = 17 := 
by 
  rw [h_diag_small_tv]
  sorry

end smaller_tv_diagonal_l427_427598


namespace min_f_value_l427_427745

open Real

noncomputable def f (x : ℝ) : ℝ :=
  (∫ θ in 0..x, (1 / cos θ)) + (∫ θ in x..(π / 2), (1 / sin θ))

theorem min_f_value : ∃ x : ℝ, 0 < x ∧ x < π / 2 ∧ is_minimum f x ∧ f x = ln (3 + 2 * sqrt 2) :=
sorry

end min_f_value_l427_427745


namespace supplement_of_complement_of_35_degree_angle_l427_427216

theorem supplement_of_complement_of_35_degree_angle : 
  ∀ α β : ℝ,  α = 35 ∧ β = (90 - α) → (180 - β) = 125 :=
by
  intros α β h
  rcases h with ⟨h1, h2⟩
  rw h1 at h2
  rw h2
  linarith

end supplement_of_complement_of_35_degree_angle_l427_427216


namespace f_of_3_eq_9_l427_427410

def f (x : ℝ) : ℝ :=
if x ≥ 0 then x ^ 2 else x

theorem f_of_3_eq_9 : f 3 = 9 :=
by
  have h : 3 ≥ 0 := by linarith
  rw [f]
  rw [if_pos h]
  norm_num
  sorry

end f_of_3_eq_9_l427_427410


namespace pyramid_height_proof_l427_427682

noncomputable def pyramid_height (side_length diag_length height : ℝ) : ℝ :=
  sqrt (height^2 - (diag_length / 2)^2)

theorem pyramid_height_proof (s : ℝ) (perimeter: ℝ) (slant_height : ℝ) :
  perimeter = 24 →
  slant_height = 9 →
  let side_length := perimeter / 4 in
  let diag_length := side_length * sqrt 2 in
  pyramid_height side_length diag_length slant_height = 3 * sqrt 7 :=
by
  intros h_perimeter h_slant_height
  simp [perimeter, slant_height, h_perimeter, h_slant_height]
  let side_length := 24 / 4
  let diag_length := side_length * sqrt 2
  have h_side_length : side_length = 6 := rfl
  have h_diag_length : diag_length = 6 * sqrt 2 := by simp [side_length, h_side_length]
  simp [pyramid_height, side_length, diag_length, sqrt]
  sorry

end pyramid_height_proof_l427_427682


namespace kevin_total_distance_l427_427059

def v1 : ℝ := 10
def t1 : ℝ := 0.5
def v2 : ℝ := 20
def t2 : ℝ := 0.5
def v3 : ℝ := 8
def t3 : ℝ := 0.25

theorem kevin_total_distance : v1 * t1 + v2 * t2 + v3 * t3 = 17 := by
  sorry

end kevin_total_distance_l427_427059


namespace g_neither_even_nor_odd_l427_427303

def g (x : Real) : Real := 1 / (3^x - 1) + 1 / 3

theorem g_neither_even_nor_odd : ¬ (∀ x, g (-x) = g x) ∧ ¬ (∀ x, g (-x) = -g x) :=
by
  sorry

end g_neither_even_nor_odd_l427_427303


namespace yogurt_net_content_acceptable_l427_427256

theorem yogurt_net_content_acceptable (x : ℕ) (h : x = 245) : 245 ≤ 250 + 5 ∧ 250 - 5 ≤ x :=
by {
  have h₁ : 250 + 5 = 255, from sorry,
  have h₂ : 250 - 5 = 245, from sorry,
  rw h at *,
  exact ⟨le_of_eq h₂, le_of_lt (lt_of_le_of_lt h₂ (nat.lt_add_right 250 5 1 (by dec_trivial)))⟩
}

end yogurt_net_content_acceptable_l427_427256


namespace exists_excursion_l427_427855

noncomputable def student := Fin 20
variable (S : Finset student) (E : Finset (Finset student))

-- Hypotheses
hypothesis hS : S.card = 20
hypothesis hE : ∀ e ∈ E, (e.card ≥ 4)

-- Theorem statement
theorem exists_excursion (hS : S.card = 20) (hE : ∀ e ∈ E, e.card ≥ 4) :
  ∃ e ∈ E, ∀ s ∈ e, (∃ f : Finset student, f ∈ E ∧ s ∈ f) → 
               (Finset.filter (λ f, s ∈ f) E).card ≥ E.card / 17 :=
sorry

end exists_excursion_l427_427855


namespace S_11_eq_21_l427_427008

-- Define the sequence a_n
def a (n : ℕ) : ℤ := (-1)^(n-1) * (4 * n - 3)

-- Define the sum S_n of the first n terms of the sequence
def S (n : ℕ) : ℤ := ∑ i in finset.range n, a (i + 1)

-- State the problem to prove S_{11} = 21
theorem S_11_eq_21 : S 11 = 21 :=
by
  sorry

end S_11_eq_21_l427_427008


namespace groups_of_three_in_class_of_fifteen_l427_427856

theorem groups_of_three_in_class_of_fifteen : 
  ∀ (n k : ℕ), n = 15 → k = 3 → nat.choose n k = 455 :=
by
  intros n k hn hk
  rw [hn, hk, nat.choose] -- Using the combination formula for computations
  sorry

end groups_of_three_in_class_of_fifteen_l427_427856


namespace polynomial_unique_l427_427746

theorem polynomial_unique (p : ℝ → ℝ) (h₁ : p 3 = 10) (h₂ : ∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 5) : 
  p = (λ x : ℝ, x^2 + 1) :=
sorry

end polynomial_unique_l427_427746


namespace douglas_votes_in_county_y_l427_427238

variable (V : ℝ) -- Number of voters in County Y
variable (A B : ℝ) -- Votes won by Douglas in County X and County Y respectively

-- Conditions
axiom h1 : A = 0.74 * 2 * V
axiom h2 : A + B = 0.66 * 3 * V
axiom ratio : (2 * V) / V = 2

-- Proof Statement
theorem douglas_votes_in_county_y :
  (B / V) * 100 = 50 := by
sorry

end douglas_votes_in_county_y_l427_427238


namespace max_n_for_factored_polynomial_l427_427330

theorem max_n_for_factored_polynomial :
  ∃ (n : ℤ), (∀ (A B : ℤ), 6 * 72 + n = 6 * B + A ∧ A * B = 72 → n ≤ 433) ∧
             (∃ (A B : ℤ), 6 * B + A = 433 ∧ A * B = 72) :=
by sorry

end max_n_for_factored_polynomial_l427_427330


namespace no_polynomials_exist_l427_427733

theorem no_polynomials_exist :
  ¬ ∃ (a b : ℚ[X]) (c d : ℚ[Y]), 1 + x * y + x^2 * y^2 = a * c + b * d :=
sorry

end no_polynomials_exist_l427_427733


namespace ibrahim_lacks_euros_l427_427434

theorem ibrahim_lacks_euros :
  let mp3_price := 120
  let cd_price := 19
  let savings := 55
  let father_contribution := 20
  let total_cost := mp3_price + cd_price
  let total_money := savings + father_contribution
  let amount_lacking := total_cost - total_money
  in amount_lacking = 64 :=
by 
  let mp3_price := 120
  let cd_price := 19
  let savings := 55
  let father_contribution := 20
  let total_cost := mp3_price + cd_price
  let total_money := savings + father_contribution
  let amount_lacking := total_cost - total_money
  show amount_lacking = 64
  sorry

end ibrahim_lacks_euros_l427_427434


namespace sin_120_eq_sqrt3_div_2_l427_427365

theorem sin_120_eq_sqrt3_div_2 : sin (120 * Real.pi / 180) = sqrt 3 / 2 :=
by
  have h1 : sin (120 * Real.pi / 180) = sin (Real.pi - (60 * Real.pi / 180)),
  { rw ← Real.sin_sub_pi_div_two,
    ring },
  rw h1,
  have h2 : sin (Real.pi - (60 * Real.pi / 180)) = sin (60 * Real.pi / 180),
  { simp [Real.sin_pi_sub] },
  rw h2,
  norm_num

end sin_120_eq_sqrt3_div_2_l427_427365


namespace projection_is_correct_l427_427747

noncomputable def v : ℝ × ℝ × ℝ := (2, 3, 1)
noncomputable def n : ℝ × ℝ × ℝ := (2, -3, 1)
noncomputable def q : ℝ × ℝ × ℝ := (19 / 7, 6 / 7, 19 / 14)
noncomputable def projection (v n : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
    let dot_product := λ u v : ℝ × ℝ × ℝ, u.1 * v.1 + u.2 * v.2 + u.3 * v.3
    let scalar_mult := λ (k : ℝ) (u : ℝ × ℝ × ℝ), (k * u.1, k * u.2, k * u.3)
    let diff := λ u v : ℝ × ℝ × ℝ, (u.1 - v.1, u.2 - v.2, u.3 - v.3)
    diff v (scalar_mult (dot_product v n / dot_product n n) n)

theorem projection_is_correct : projection v n = q := by
  sorry

end projection_is_correct_l427_427747


namespace range_of_x_l427_427006

theorem range_of_x (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : x + y + z = 1) (h4 : x^2 + y^2 + z^2 = 3) : 1 ≤ x ∧ x ≤ 5 / 3 :=
by
  sorry

end range_of_x_l427_427006


namespace inscribed_circle_radius_l427_427220

theorem inscribed_circle_radius (d1 d2 : ℝ) (h1 : d1 = 14) (h2 : d2 = 30) : 
  ∃ r : ℝ, r = (105 * Real.sqrt 274) / 274 := 
by 
  sorry

end inscribed_circle_radius_l427_427220


namespace first_place_team_ties_l427_427034

noncomputable def teamPoints (wins ties: ℕ) : ℕ := 2 * wins + ties

theorem first_place_team_ties {T : ℕ} : 
  teamPoints 13 1 + teamPoints 8 10 + teamPoints 12 T = 81 → T = 4 :=
by
  sorry

end first_place_team_ties_l427_427034


namespace find_valid_a_range_l427_427971

variables (a : ℝ) (N M : ℕ)

-- Given conditions
def distinct_floors (f : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i j, 1 ≤ i → i < j → j ≤ n → floor (f i) ≠ floor (f j)

def valid_a (a : ℝ) (N M : ℕ) : Prop :=
  (N > 1 → M > 1) → (N > 0) → (M > 0) → (a > 0) →
    (distinct_floors (λ k, a * k) N) ∧ (distinct_floors (λ k, (k:ℝ) / a) M)

-- Main theorem statement
theorem find_valid_a_range (N M : ℕ) : 
  (1 < N) ∧ (1 < M) →
  ∃ a : ℝ, (real.abs a) ∈ set.Icc ((N - 1) / N : ℝ) (M / (M - 1)) ∧ valid_a a N M :=
begin
  sorry,
end

end find_valid_a_range_l427_427971


namespace sin_theta_plus_phi_l427_427438

theorem sin_theta_plus_phi (θ φ : ℝ) 
  (h1 : complex.exp (complex.I * θ) = complex.of_real 4 / 5 + complex.I * (complex.of_real 3 / 5))
  (h2 : complex.exp (complex.I * φ) = -(complex.of_real 5 / 13) + complex.I * (complex.of_real 12 / 13)) :
  real.sin (θ + φ) = 33 / 65 := 
sorry

end sin_theta_plus_phi_l427_427438


namespace find_a1_and_q_bn_expression_when_m_1_range_of_m_for_Sn_l427_427399

noncomputable section 
open_locale big_operators

variables (n : ℕ+) (a b : ℕ → ℝ)
variable (m : ℝ)

-- Given conditions
def a_is_geometric_seq : Prop := ∃ (a₁ q : ℝ), ∀ n, a n = a₁ * q^n
def b_seq_satisfies_cond : Prop := ∀ n, b n = ∑ i in finset.range (n+1), (n - i) * (a i)
def b1_val_is_m : Prop := b 1 = m
def b2_val_is_3m_by_2 : Prop := b 2 = (3 * m) / 2
def m_non_zero : Prop := m ≠ 0 

-- Proofs requested
-- Proof 1: First term a₁ and common ratio q giving a₁ = m and q = -1/2.
theorem find_a1_and_q
  (h1 : a_is_geometric_seq)
  (h2 : b_seq_satisfies_cond)
  (h3 : b1_val_is_m)
  (h4 : b2_val_is_3m_by_2)
  (h5 : m_non_zero) :
∃ (a₁ q : ℝ), a₁ = m ∧ q = -1 / 2 := 
sorry

-- Proof 2: When m = 1, provide expression for b_n.
theorem bn_expression_when_m_1
  (h1 : a_is_geometric_seq)
  (h2 : b_seq_satisfies_cond)
  (h3 : b1_val_is_m)
  (h4 : b2_val_is_3m_by_2)
  (h5 : m_non_zero)
  (hm : m = 1) :
∀ n, b n = (6 * n + 2 + (-2)^(1 - n)) / 9 := 
sorry

-- Proof 3: Find the range of m given Sn ∈ [1, 3] for any n.
theorem range_of_m_for_Sn
  (h1 : a_is_geometric_seq)
  (h2 : b_seq_satisfies_cond)
  (h3 : b1_val_is_m)
  (h4 : b2_val_is_3m_by_2)
  (h5 : m_non_zero)
  (hSn : ∀ n, let Sn := a 1 * (1 - (-1/2)^n) / (1 - (-1/2)) in Sn ∈ set.Icc (1 : ℝ) 3 ):
2 ≤ m ∧ m ≤ 3 := 
sorry

end find_a1_and_q_bn_expression_when_m_1_range_of_m_for_Sn_l427_427399


namespace max_n_for_factored_polynomial_l427_427329

theorem max_n_for_factored_polynomial :
  ∃ (n : ℤ), (∀ (A B : ℤ), 6 * 72 + n = 6 * B + A ∧ A * B = 72 → n ≤ 433) ∧
             (∃ (A B : ℤ), 6 * B + A = 433 ∧ A * B = 72) :=
by sorry

end max_n_for_factored_polynomial_l427_427329


namespace total_days_1998_to_2001_l427_427433

theorem total_days_1998_to_2001 : 
  let is_leap_year := λ y, (y % 4 = 0) ∧ (y % 100 ≠ 0 ∨ y % 400 = 0) in
  let non_leap_year_days := 3 * 365 in
  let leap_year_days := 366 in
  let total_days := non_leap_year_days + leap_year_days in
  total_days = 1461 :=
by
  let is_leap_year := λ y, (y % 4 = 0) ∧ (y % 100 ≠ 0 ∨ y % 400 = 0)
  let non_leap_year_days := 3 * 365
  let leap_year_days := 366
  let total_days := non_leap_year_days + leap_year_days
  show total_days = 1461, from sorry

end total_days_1998_to_2001_l427_427433


namespace qingqiu_country_connected_l427_427163

theorem qingqiu_country_connected (n : ℕ)
  (h₁ : ∀ i j : ℕ, i ≠ j → (∃ d : ℕ, d = distance i j))
  (h₂ : ∀ i j : ℕ, i ≠ j → distance i j ≠ distance j i)
  (h₃ : ∀ i : ℕ, 1 ≤ i → i < n → ∃ j : ℕ, j ≠ i ∨ i = n ∧ distance i j = min (dist i)) :
  ∃ city_numbering : { p // p.perm = (list.range n).perm },
    (∀ i j : ℕ, i ≠ j → ∃ k, distance i j = min (dist k)) ∧
    ∀ i j : ℕ, connected_by_airline i j :=
begin
  sorry
end

end qingqiu_country_connected_l427_427163


namespace max_n_for_factored_polynomial_l427_427332

theorem max_n_for_factored_polynomial :
  ∃ (n : ℤ), (∀ (A B : ℤ), 6 * 72 + n = 6 * B + A ∧ A * B = 72 → n ≤ 433) ∧
             (∃ (A B : ℤ), 6 * B + A = 433 ∧ A * B = 72) :=
by sorry

end max_n_for_factored_polynomial_l427_427332


namespace david_reading_time_l427_427723

theorem david_reading_time (total_time : ℕ) (math_time : ℕ) (spelling_time : ℕ) 
  (reading_time : ℕ) (h1 : total_time = 60) (h2 : math_time = 15) 
  (h3 : spelling_time = 18) (h4 : reading_time = total_time - (math_time + spelling_time)) : 
  reading_time = 27 := by
  sorry

end david_reading_time_l427_427723


namespace degree_meas_supp_compl_35_l427_427202

noncomputable def degree_meas_supplement_complement (θ : ℝ) : ℝ :=
  180 - (90 - θ)

theorem degree_meas_supp_compl_35 : degree_meas_supplement_complement 35 = 125 :=
by
  unfold degree_meas_supplement_complement
  norm_num
  sorry

end degree_meas_supp_compl_35_l427_427202


namespace translate_polygon_forms_prism_l427_427160

-- Define a planar polygon (this can be descriptive as needed for the context)
structure Polygon where
  vertices : List (ℝ × ℝ)   -- A list of vertices in 2D

-- Define what a translation is
def translate (p : Polygon) (v : ℝ × ℝ × ℝ) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ p.vertices → ∃ z : ℝ, (x + v.1, y + v.2, z + v.3)  -- Translation in 3D

-- Define a Prism in terms of translating a polygon
structure Prism where
  base : Polygon
  translation_vec : ℝ × ℝ × ℝ
  is_translated : translate base translation_vec
  
-- The statement we need to prove
theorem translate_polygon_forms_prism (p : Polygon) (v : ℝ × ℝ × ℝ) : ∃ (prism : Prism), prism.base = p ∧ prism.translation_vec = v :=
  sorry

end translate_polygon_forms_prism_l427_427160


namespace min_area_tangent_circle_l427_427328

theorem min_area_tangent_circle :
  ∃ (a r : ℝ),
    a > 0 ∧
    r = | 2 * a + 2 / a + 1 | / √5 ∧
    (x y : ℝ), (x - a)^2 + (y - 2 / a)^2 = r^2 ∧
    (∀ x, x > 0 → (2 * √ (2 * x / 2 / a) + 1) = 5 → a = x) ∧ 
    (r_min = √5 → (x - 1)^2 + (y - 2)^2 = 5)
:= sorry

end min_area_tangent_circle_l427_427328


namespace petya_vasya_equal_again_l427_427102

theorem petya_vasya_equal_again (n : ℤ) (hn : n ≠ 0) :
  ∃ (k m : ℕ), (∃ P V : ℤ, P = n + 10 * k ∧ V = n - 10 * k ∧ 2014 * P * V = n) :=
sorry

end petya_vasya_equal_again_l427_427102


namespace minimum_value_l427_427415

theorem minimum_value :
  ∀ (m n : ℝ), m > 0 → n > 0 → (3 * m + n = 1) → (3 / m + 1 / n) ≥ 16 :=
by
  intros m n hm hn hline
  sorry

end minimum_value_l427_427415


namespace lakshmi_investment_time_l427_427547

theorem lakshmi_investment_time (x y : ℝ) (initial_investment : ℝ)
  (annual_gain lakshmi_share raman_investment muthu_investment : ℝ)
  (total_gain : annual_gain = 36000)
  (lakshmi_contrib : lakshmi_share = 12000)
  (ratio : lakshmi_share / annual_gain = 1 / 3)
  (raman_investment : raman_investment = x * 12)
  (lakshmi_investment : lakshmi_investment = 2 * x * (12 - y))
  (muthu_investment : muthu_investment = 3 * x * 4)
  (total_investment : initial_investment = raman_investment + lakshmi_investment + muthu_investment) :
  y = 6 :=
begin
  sorry
end

end lakshmi_investment_time_l427_427547


namespace min_stamps_l427_427892

theorem min_stamps : ∃ (x y : ℕ), 5 * x + 7 * y = 35 ∧ x + y = 5 :=
by
  have : ∀ (x y : ℕ), 5 * x + 7 * y = 35 → x + y = 5 → True := sorry
  sorry

end min_stamps_l427_427892


namespace gcd_98_63_l427_427582

def gcd (a : ℕ) (b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_98_63 : gcd 98 63 = 7 := 
by sorry

end gcd_98_63_l427_427582


namespace domain_f_l427_427739

noncomputable def f (x : ℝ) : ℝ := real.cbrt (x - 2) + real.cbrt (7 - x)

theorem domain_f : ∀ x : ℝ, -∞ < x ∧ x < ∞ :=
by
  intro x
  sorry

end domain_f_l427_427739


namespace probability_all_yellow_l427_427054

-- Definitions and conditions
def total_apples : ℕ := 8
def red_apples : ℕ := 5
def yellow_apples : ℕ := 3
def chosen_apples : ℕ := 3

-- Theorem to prove
theorem probability_all_yellow :
  (yellow_apples.choose chosen_apples : ℚ) / (total_apples.choose chosen_apples) = 1 / 56 := sorry

end probability_all_yellow_l427_427054


namespace total_clowns_l427_427965

theorem total_clowns (mobiles : ℕ) (clowns_per_mobile : ℕ) (num_clowns : ℕ) 
    (h1 : mobiles = 5) (h2 : clowns_per_mobile = 28) : 
    (clowns_per_mobile * mobiles = num_clowns) → num_clowns = 140 :=
by
  intros h3
  rw [h1, h2] at h3
  exact h3

end total_clowns_l427_427965


namespace total_interest_proof_l427_427240

open Real

def initial_investment : ℝ := 10000
def interest_6_months : ℝ := 0.02 * initial_investment
def reinvested_amount_6_months : ℝ := initial_investment + interest_6_months
def interest_10_months : ℝ := 0.03 * reinvested_amount_6_months
def reinvested_amount_10_months : ℝ := reinvested_amount_6_months + interest_10_months
def interest_18_months : ℝ := 0.04 * reinvested_amount_10_months

def total_interest : ℝ := interest_6_months + interest_10_months + interest_18_months

theorem total_interest_proof : total_interest = 926.24 := by
    sorry

end total_interest_proof_l427_427240


namespace pqrs_sum_l427_427515

theorem pqrs_sum (p q r s : ℝ)
  (h1 : (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 → x = r ∨ x = s))
  (h2 : (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 → x = p ∨ x = q))
  (h3 : p ≠ q) (h4 : p ≠ r) (h5 : p ≠ s) (h6 : q ≠ r) (h7 : q ≠ s) (h8 : r ≠ s) :
  p + q + r + s = 2028 :=
sorry

end pqrs_sum_l427_427515


namespace convex_quadrilateral_is_parallelogram_l427_427087

variable {A B C D : Point}

-- Define the angles
def angle_A_B_D := angle B A D
def angle_B_A_D := angle A B D
def angle_A_D_B := angle D A B
def angle_C_B_D := angle B C D
def angle_B_C_D := angle C B D
def angle_C_D_B := angle D C B
def angle_D_A_B := angle A D B
def angle_D_B_C := angle B D C

-- Conditions: convex quadrilateral and sum of sines of opposite angles equal
def convex_quadrilateral (A B C D : Point) : Prop :=
  convex A B C D

def sine_sum_condition (A B C D : Point) : Prop :=
  sin (angle A B D) + sin (angle C D B) = sin (angle B A D) + sin (angle D C B)

-- The theorem to prove
theorem convex_quadrilateral_is_parallelogram 
  (A B C D : Point)
  (h_convex : convex_quadrilateral A B C D)
  (h_sine_sum : sine_sum_condition A B C D) :
  parallelogram A B C D := 
sorry

end convex_quadrilateral_is_parallelogram_l427_427087


namespace collinear_HMN_l427_427543

variables {A B C A1 B1 M H N : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
[MetricSpace A1] [MetricSpace B1] [MetricSpace M] [MetricSpace H] [MetricSpace N]
(A B C A1 B1 M H N : Point MetricSpace)

-- Conditions
def midpoint (P Q : Point MetricSpace) := ∃ M, dist P M = dist M Q

def tangent_circle (center : Point MetricSpace) (M : Point MetricSpace) (P Q : Point MetricSpace) := 
  dist center P = dist center Q ∧ 
  ∃ radius, Metric.Ball M radius = Circle(center, radius)

variables (hA1 : midpoint B C A1)
(hB1 : midpoint A C B1)
(hM : midpoint A1 B1 M)
(hH : foot_of_altitude C A B H)
(hCircleA1 : tangent_circle M A1 B C)
(hCircleB1 : tangent_circle M B1 A C)
(hN : second_intersection_of_circles M M A1 B1 N)

-- Proof Statement
theorem collinear_HMN : collinear H M N :=
sorry

end collinear_HMN_l427_427543


namespace perfect_square_n_l427_427769

def is_perfect_square (x : ℕ) : Prop :=
  ∃ y : ℕ, y * y = x

theorem perfect_square_n (n : ℕ) : 
  is_perfect_square (nat.factorial 1 * nat.factorial 2 * nat.factorial 3 * 
    ((finset.range (2 * n + 1).succ).filter nat.even).prod (!.) * 
    nat.factorial (2 * n)) ∧ (n = 4 * k * (k + 1) ∨ n = 2 * k ^ 2 - 1) :=
by
  sorry

end perfect_square_n_l427_427769


namespace max_value_of_n_l427_427343

-- Define the main variables and conditions
noncomputable def max_n : ℕ := 
  let n_pairs := [(1, 72), (2, 36), (3, 24), (4, 18), (6, 12), (8, 9)] 
  max (n_pairs.map (λ p, 6 * p.2 + p.1))

-- Lean theorem statement for the equivalence
theorem max_value_of_n :
  max_n = 433 := by
  sorry

end max_value_of_n_l427_427343


namespace find_k_find_min_cost_l427_427660

noncomputable def satisfy_conditions (W1 : ℝ) (k : ℝ) (v : ℝ) : Prop :=
  W1 = k * v^2 ∧ W1 = 96 ∧ v = 10

-- Theorem to prove part (1)
theorem find_k : ∃ k : ℝ, satisfy_conditions 96 k 10 ∧ k = 0.96 :=
by
  use 0.96
  unfold satisfy_conditions
  split
  { exact rfl }
  split
  { rfl }
  { linarith }

-- Theorem to prove part (2)
theorem find_min_cost (v : ℝ) (h_max_speed : v ≤ 15) (h_pos_speed : 0 < v) : ∃ W : ℝ, W = (96 * v + 15000 / v) ∧ W = 2400 :=
by
  use 2400
  have h := min_of_convex_on (λ v, ↑(96 * v) + ↑(15000 / v)) (0, 15)
  sorry

end find_k_find_min_cost_l427_427660


namespace liters_in_cubic_foot_l427_427478

def pool_length : Float := 20
def pool_width : Float := 6
def pool_depth : Float := 10
def total_cost : Float := 90000
def liters_per_cubic_foot : Float := 28.3168

theorem liters_in_cubic_foot :
  (pool_length * pool_width * pool_depth * liters_per_cubic_foot = 1200 * liters_per_cubic_foot) ∧
  (total_cost / (1200 * liters_per_cubic_foot) ≈ 2.65) ->
  1 = 28.3168 :=
by
  sorry

end liters_in_cubic_foot_l427_427478


namespace number_A_is_read_without_zero_l427_427466

def number_A : ℕ := 4006530
def number_B : ℕ := 4650003
def number_C : ℕ := 4650300
def number_D : ℕ := 4006053

-- Define a function to check if a number is read without saying "zero"
def read_without_zero (n : ℕ) : Prop :=
  let s := n.to_string in
  not (s.contains '0')

theorem number_A_is_read_without_zero : read_without_zero number_A :=
by
  sorry

end number_A_is_read_without_zero_l427_427466


namespace ellipse_foci_xaxis_focal_length_2_l427_427842

theorem ellipse_foci_xaxis_focal_length_2 (a : ℝ) (h : a > 0) :
  (∃ c b : ℝ, c = 1 ∧ b^2 = 2 ∧ a^2 = c^2 + b^2) → a = Real.sqrt 3 :=
by
  intros h_exists
  obtain ⟨c, b, hc, hb, ha⟩ := h_exists
  have hc1 : c^2 = 1^2, by rw [hc, pow_two]
  have hb2 : b^2 = 2, by exact hb
  rw ha at hc1 hb2
  sorry

end ellipse_foci_xaxis_focal_length_2_l427_427842


namespace geom_mean_does_not_exist_l427_427578

theorem geom_mean_does_not_exist (a b : Real) (h1 : a = 2) (h2 : b = -2) : ¬ ∃ g : Real, g^2 = a * b := 
by
  sorry

end geom_mean_does_not_exist_l427_427578


namespace leah_chocolates_l427_427018

theorem leah_chocolates :
  ∃ y : ℝ, (y - 8 = y / 3) ∧ y = 12 :=
by
  use 12
  split
  { sorry }
  { refl }

end leah_chocolates_l427_427018


namespace sum_of_square_areas_l427_427047

theorem sum_of_square_areas (a b : ℝ)
  (h1 : a + b = 14)
  (h2 : a - b = 2) :
  a^2 + b^2 = 100 := by
  sorry

end sum_of_square_areas_l427_427047


namespace max_omega_value_l427_427408

noncomputable def f (ω φ x : ℝ) := Real.sin (ω * x + φ)

def center_of_symmetry (ω φ : ℝ) := 
  ∃ n : ℤ, ω * (-Real.pi / 4) + φ = n * Real.pi

def extremum_point (ω φ : ℝ) :=
  ∃ n' : ℤ, ω * (Real.pi / 4) + φ = n' * Real.pi + Real.pi / 2

def monotonic_in_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < b → a < y ∧ y < b → x ≤ y → f x ≤ f y

theorem max_omega_value (ω : ℝ) (φ : ℝ) : 
  (ω > 0) →
  (|φ| ≤ Real.pi / 2) →
  center_of_symmetry ω φ →
  extremum_point ω φ →
  monotonic_in_interval (f ω φ) (5 * Real.pi / 18) (2 * Real.pi / 5) →
  ω = 5 :=
by
  sorry

end max_omega_value_l427_427408


namespace shaded_area_l427_427465

-- Definitions based on the conditions
variable (ABCD : Rectangle)
variable (area_ABCD : ℝ)
variable (AOD BOC ABE BCF CDG ADH : EquilateralTriangle)
variable (M N P Q : Point)
variable (centers : M = center_of ABE ∧ N = center_of BCF ∧ P = center_of CDG ∧ Q = center_of ADH)

-- The area of rectangle ABCD is 2013 square centimeters
axiom h1 : area ABCD = 2013

-- Equilateral triangles are defined within the rectangle
axiom h2 : is_equilateral AOD ∧ is_equilateral BOC ∧ is_equilateral ABE ∧ 
           is_equilateral BCF ∧ is_equilateral CDG ∧ is_equilateral ADH

-- Points M, N, P, Q are the centers of corresponding equilateral triangles
axiom h3 : centers

-- Theorem to prove the area of the shaded region
theorem shaded_area : area_shaded = 2684 :=
by
  sorry

end shaded_area_l427_427465


namespace pow_succ_ge_mul_add_one_l427_427394

theorem pow_succ_ge_mul_add_one 
  (m : ℕ) (x : ℝ) (h : x > -1) : (1 + x)^(m + 1) ≥ 1 + (m + 1) * x :=
begin
  induction m with k hk,
  { simp [h], },
  { calc (1 + x)^(k + 1 + 1) = (1 + x) * (1 + x)^(k + 1) : by rw [pow_succ]
    ... ≥ (1 + x) * (1 + k * x) : mul_le_mul_of_nonneg_left hk (le_of_lt h)
    ... = 1 + (k + 1) * x + k * x^2 : by ring
    ... ≥ 1 + (k + 1) * x : add_le_add_left (mul_self_nonneg x) (1 + (k + 1) * x) },
end

end pow_succ_ge_mul_add_one_l427_427394


namespace sum_of_solutions_eq_zero_l427_427627

theorem sum_of_solutions_eq_zero : 
  (∑ x in {x : ℝ | 3 * x / 15 = 6 / x}.to_finset, x) = 0 :=
by sorry

end sum_of_solutions_eq_zero_l427_427627


namespace infinite_m_satisfying_m_minus_f_eq_1000_l427_427528

def f (m : ℕ) : ℕ := 
  (List.range (m + 1)).sum (λ i, m / 2^i)

theorem infinite_m_satisfying_m_minus_f_eq_1000 :
  ∃^∞ m : ℕ, m > 0 ∧ m - f m = 1000 :=
sorry

end infinite_m_satisfying_m_minus_f_eq_1000_l427_427528


namespace supplement_of_complement_of_35_degree_angle_l427_427212

theorem supplement_of_complement_of_35_degree_angle : 
  ∀ α β : ℝ,  α = 35 ∧ β = (90 - α) → (180 - β) = 125 :=
by
  intros α β h
  rcases h with ⟨h1, h2⟩
  rw h1 at h2
  rw h2
  linarith

end supplement_of_complement_of_35_degree_angle_l427_427212


namespace parallelogram_side_square_diff_lt_diag_prod_l427_427108

/-- Define the basic setup for the parallelogram ABCD and its properties --/
variables {A B C D : Type*}
variable [normed_add_comm_group A]
variable [inner_product_space ℝ A]
variables (AB BC CA DA : A)

/-- Define conditions for ABCD being a parallelogram --/
def is_parallelogram (A B C D : A) : Prop :=
  (B - A = C - D) ∧ (D - A = C - B)

noncomputable def diag_AC := AB + BC
noncomputable def diag_BD := BC - AB

/-- The final statement of the theorem --/
theorem parallelogram_side_square_diff_lt_diag_prod
  (h : is_parallelogram AB BC CA DA) :
  ∥AB∥^2 - ∥BC∥^2 < ∥diag_AC AB BC∥ * ∥diag_BD AB BC∥ :=
sorry

end parallelogram_side_square_diff_lt_diag_prod_l427_427108


namespace solution_problem_l427_427617

def is_good (t : ℕ) : Prop :=
  ∃ (a : ℕ → ℕ), a 0 = 15 ∧ a 1 = t ∧ ∀ n, (a n)*(a (n+2)) = ((a (n+1)) - 1) * ((a (n+1)) + 1)

def sum_good_numbers : ℕ :=
  ∑ t in { t : ℕ | is_good t }, t

theorem solution_problem : sum_good_numbers = 296 :=
sorry

end solution_problem_l427_427617


namespace projection_of_a_onto_b_is_minus_three_fifths_l427_427790

def vec_b : ℝ × ℝ := (3, 4)
def dot_product_a_b : ℝ := -3

theorem projection_of_a_onto_b_is_minus_three_fifths :
  (dot_product_a_b / (Real.sqrt(vec_b.1 ^ 2 + vec_b.2 ^ 2))) = -3 / 5 :=
by
  sorry

end projection_of_a_onto_b_is_minus_three_fifths_l427_427790


namespace one_third_way_l427_427577

theorem one_third_way (a b : ℚ) (h₁ : a = 1/4) (h₂ : b = 1/6) : ((2 * a + b) / 3) = 2/9 :=
by 
  have ha : a = 3 /12 := by rw [h₁, show (1:ℚ)/4 = 3/12, by norm_num]
  have hb : b = 2 /12 := by rw [h₂, show (1:ℚ)/6 = 2/12, by norm_num]
  calc
  ((2 * a + b) / 3) 
      = (2 * (3 / 12) + (2 / 12)) / 3 : by rw [ha, hb]
  ... = (6 / 12 + 2 / 12) / 3 : by ring
  ... = (8 / 12) / 3 : by norm_num
  ... = 8 / 36 : by field_simp
  ... = 2 / 9  : by norm_num

end one_third_way_l427_427577


namespace largest_n_for_factored_quad_l427_427346

theorem largest_n_for_factored_quad (n : ℤ) (b d : ℤ) 
  (h1 : 6 * d + b = n) (h2 : b * d = 72) 
  (factorable : ∃ x : ℤ, (6 * x + b) * (x + d) = 6 * x ^ 2 + n * x + 72) : 
  n ≤ 433 :=
sorry

end largest_n_for_factored_quad_l427_427346


namespace match_context_l427_427929

-- Given conditions
def cond1 : Prop := "Cultural diversity is a concentrated display of national culture"
def cond2 : Prop := "Culture is national, but also global"
def cond3 : Prop := "Commercial trade is an important way to spread culture"
def cond4 : Prop := "Acrobatic performers have become messengers of Chinese culture"

-- Correct answer based on the provided analysis
def correct_answer : Prop := (cond2 ∧ cond4) ∧ ¬(cond1 ∨ cond3)

theorem match_context : correct_answer := 
by
  -- Assumptions from analysis
  have h2 : cond2 := sorry
  have h4 : cond4 := sorry
  have not_h1 : ¬cond1 := sorry
  have not_h3 : ¬cond3 := sorry
  -- Combine into correct answer
  exact ⟨⟨h2, h4⟩, ⟨not_h1, not_h3⟩⟩

end match_context_l427_427929


namespace cyclic_BCOKL_l427_427899

-- Definitions necessary to set up the problem
variables {A B C O H K L E : Point}
variable (circumcircle : Triangle → Circle) -- circumcircle
variable (orthocenter : Triangle → Point) -- orthocenter
variable (line_parallel : Line → Line → Prop) -- line parallel property
variable (meets : Line → Circle → Point → Prop) -- line circle meet property

-- Assuming necessary conditions
axiom scalene_triangle (ΔABC : Triangle) : scalene ΔABC
axiom circumcircle_condition (ΔABC : Triangle) : circumcircle ΔABC = O

axiom orthocenter_condition (ΔABC : Triangle) : orthocenter ΔABC = H

axiom line_A_parallel_OH (line : Line) : line_parallel line (line_through_points A H)
axiom line_K_parallel_AH (line : Line) : line_parallel line (line_through_points K H)
axiom line_L_parallel_OA (line : Line) : line_parallel line (line_through_points L A)

axiom line_through_points_meets_O (line : Line) (circle : Circle) (point : Point) :
  meets line circle point

-- Main statement to prove
theorem cyclic_BCOKL (ΔABC : Triangle) (O H K L E : Point) :
  scalene ΔABC →
  circumcircle ΔABC = O →
  orthocenter ΔABC = H →
  (∃ line_A, line_parallel line_A (line_through_points A H) ∧
             meets line_A (circumcircle ΔABC) K) →
  (∃ line_K, line_parallel line_K (line_through_points K H) ∧
             meets line_K (circumcircle ΔABC) L) →
  (∃ line_L, line_parallel line_L (line_through_points L A) ∧
             meets line_L (line_through_points O H) E) →
  cyclic {B, C, O, E} :=
sorry

end cyclic_BCOKL_l427_427899


namespace WR_eq_35_l427_427872

theorem WR_eq_35 (PQ ZY SX : ℝ) (hPQ : PQ = 30) (hZY : ZY = 15) (hSX : SX = 10) :
    let WS := ZY - SX
    let SR := PQ
    let WR := WS + SR
    WR = 35 := by
  sorry

end WR_eq_35_l427_427872


namespace simplify_expression_l427_427559

theorem simplify_expression (x : ℝ) : 
  (3 * x - 4) * (x + 8) - (x + 6) * (3 * x - 2) = 4 * x - 20 := 
by
  sorry

end simplify_expression_l427_427559


namespace distance_sum_eq_circumradius_l427_427542

variable {R : Type*} [NormedField R] [NormedSpace ℝ R] [CompleteSpace R]

theorem distance_sum_eq_circumradius (A B C P : R) (h_eq_triangle : dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B)
    (h_circumcircle : (P = arc AB) ∧ circumscribed (set A B C)) :
    dist P C = dist P A + dist P B :=
  sorry

end distance_sum_eq_circumradius_l427_427542


namespace range_of_f_l427_427145

noncomputable def f (x : ℝ) : ℝ := Real.arcsin (Real.cos x) + Real.arccos (Real.sin x)

theorem range_of_f : Set.range f = Set.Icc 0 Real.pi :=
sorry

end range_of_f_l427_427145


namespace sum_of_distinct_roots_l427_427522

theorem sum_of_distinct_roots 
  (p q r s : ℝ)
  (h1 : p ≠ q)
  (h2 : p ≠ r)
  (h3 : p ≠ s)
  (h4 : q ≠ r)
  (h5 : q ≠ s)
  (h6 : r ≠ s)
  (h_roots1 : (x : ℝ) -> x^2 - 12*p*x - 13*q = 0 -> x = r ∨ x = s)
  (h_roots2 : (x : ℝ) -> x^2 - 12*r*x - 13*s = 0 -> x = p ∨ x = q) : 
  p + q + r + s = 1716 := 
by 
  sorry

end sum_of_distinct_roots_l427_427522


namespace sum_of_values_of_M_l427_427144

theorem sum_of_values_of_M (M : ℝ) (h : M * (M - 8) = 12) :
  (∃ M1 M2 : ℝ, M^2 - 8 * M - 12 = 0 ∧ M1 + M2 = 8) :=
sorry

end sum_of_values_of_M_l427_427144


namespace fraction_not_equal_valid_l427_427314

def fraction_not_equal := ∀ x, x = (1 + 1 / 5) → (7 / 5 ≠ x) 

theorem fraction_not_equal_valid: fraction_not_equal :=
by
  sorry

-- Now defining that the problematic options follow the fraction_not_equal definition
def C := 1 + (7 / 35)
def D := 1 + (4 / 20)
def E := 1 + (3 / 15)

example : fraction_not_equal C :=
by
  sorry

example : fraction_not_equal D :=
by
  sorry

example : fraction_not_equal E :=
by
  sorry

end fraction_not_equal_valid_l427_427314


namespace cos_double_angle_l427_427825

theorem cos_double_angle (α : ℝ) (h : ‖(Real.cos α, Real.sqrt 2 / 2)‖ = Real.sqrt 3 / 2) : Real.cos (2 * α) = -1 / 2 :=
sorry

end cos_double_angle_l427_427825


namespace degree_measure_of_supplement_of_complement_of_35_degree_angle_l427_427194

def complement (α : ℝ) : ℝ := 90 - α
def supplement (β : ℝ) : ℝ := 180 - β

theorem degree_measure_of_supplement_of_complement_of_35_degree_angle : 
  supplement (complement 35) = 125 :=
by
  sorry

end degree_measure_of_supplement_of_complement_of_35_degree_angle_l427_427194


namespace farmer_extra_days_l427_427263

theorem farmer_extra_days (total_area planned_rate actual_rate remaining_area : ℝ) 
  (H1 : total_area = 312) 
  (H2 : planned_rate = 260) 
  (H3 : actual_rate = 85) 
  (H4 : remaining_area = 40) : 
  let planned_days := (total_area / planned_rate).ceil,
      worked_days := ((total_area - remaining_area) / actual_rate).ceil in
  worked_days - planned_days = 2 :=
by
  let planned_days := (total_area / planned_rate).ceil;
  let worked_days := ((total_area - remaining_area) / actual_rate).ceil;
  sorry

end farmer_extra_days_l427_427263


namespace minimum_cost_l427_427255

-- Define the conditions of the problem
def distance : ℝ := 130
def fuel_price : ℝ := 2
def consumption_rate (x : ℝ) : ℝ := 2 + (x^2 / 360)
def wage_rate : ℝ := 14
def speed_range := { x : ℝ | 50 ≤ x ∧ x ≤ 100 }

-- Define the total cost function
noncomputable def total_cost (x : ℝ) : ℝ := 
  (distance / x) * consumption_rate(x) * fuel_price + wage_rate * (distance / x)

-- Prove the expression for total cost and the minimum cost conditions
theorem minimum_cost : 
  (∀ x ∈ speed_range, total_cost(x) = (2340 / x) + (13 * x / 18)) ∧
  (∀ x ∈ speed_range, (2340 / x) + (13 * x / 18) ≥ 26 * sqrt(10)) ∧
  (total_cost(18 * sqrt(10)) = 26 * sqrt(10)) :=
sorry

end minimum_cost_l427_427255


namespace part1_part2_part3_l427_427807

section
variables {a m b : ℝ} (f : ℝ → ℝ)
variable [log_unique : log a (1 / 3) > 0]

-- Condition: f(x) = log_a (1 - mx)/(x + 1) is odd and a > 0, a ≠ 1, m ≠ -1
variable h₁ : ∀ x ∈ (-1, 1), f (−x) + f x = 0
variables h₂ : a > 0
variables h₃ : a ≠ 1
variable h₄ : m ≠ -1

-- (1) Prove that m = 1
theorem part1 : m = 1 := sorry

-- (2) Prove the monotonicity of f(x) on (-1, 1) for m = 1
theorem part2 : ∀ x₁ x₂ ∈ (-1:ℝ, 1:ℝ),
  (x₁ < x₂ → f x₁ > f x₂) ↔ a > 1 ∧ (x₁ < x₂ → f x₁ < f x₂) ↔ 0 < a < 1 := sorry

-- (3) Prove the range of the real number b
theorem part3 : ∃ (b : ℝ), (4/3 < b ∧ b < 3/2) ∧
  (f (b - 2) + f (2 * b - 2)) > 0 := sorry

end

end part1_part2_part3_l427_427807


namespace percentage_increase_l427_427239

def original_amount : ℝ := 65
def new_amount : ℝ := 72

theorem percentage_increase :
  let percentage_increase := ((new_amount - original_amount) / original_amount) * 100 in
  percentage_increase = 10.77 :=
by
  sorry

end percentage_increase_l427_427239


namespace james_monthly_income_l427_427476

def tier1_subscribers : Nat := 130
def tier2_subscribers : Nat := 75
def tier3_subscribers : Nat := 45

def tier1_cost : Float := 4.99
def tier2_cost : Float := 9.99
def tier3_cost : Float := 24.99

def tier1_percentage : Float := 0.70
def tier2_percentage : Float := 0.80
def tier3_percentage : Float := 0.90

def tier1_revenue : Float := tier1_subscribers * tier1_cost * tier1_percentage
def tier2_revenue : Float := tier2_subscribers * tier2_cost * tier2_percentage
def tier3_revenue : Float := tier3_subscribers * tier3_cost * tier3_percentage

def total_revenue : Float := tier1_revenue + tier2_revenue + tier3_revenue

theorem james_monthly_income : total_revenue = 2065.41 := by
  sorry

end james_monthly_income_l427_427476


namespace triangle_inequality_example_l427_427400

theorem triangle_inequality_example {x : ℝ} (h1: 3 + 4 > x) (h2: abs (3 - 4) < x) : 1 < x ∧ x < 7 :=
  sorry

end triangle_inequality_example_l427_427400


namespace largest_n_factorable_l427_427338

theorem largest_n_factorable :
  ∃ n, (∀ A B : ℤ, 6x^2 + n • x + 72 = (6 • x + A) * (x + B)) ∧ 
    (∀ x', 6x' + A = 0 ∨ x' + B = 0) ∧ 
    n = 433 := 
sorry

end largest_n_factorable_l427_427338


namespace solution_l427_427282

noncomputable def problem : Prop :=
  let A := -6
  let B := 0
  let C := 0.2
  let D := 3
  A < 0 ∧ ¬ (B < 0) ∧ ¬ (C < 0) ∧ ¬ (D < 0)

theorem solution : problem := 
by
  let A := -6
  let B := 0
  let C := 0.2
  let D := 3
  exact ⟨ by norm_num, by norm_num, by norm_num, by norm_num ⟩

end solution_l427_427282


namespace no_calls_days_l427_427089

-- Definitions based on the given conditions
def call_frequency_1 := 6
def call_frequency_2 := 8
def call_frequency_3 := 9
def total_days := 366

-- Theorem statement
theorem no_calls_days : 
  ∀ (call_frequency_1 call_frequency_2 call_frequency_3 total_days : ℕ),
  call_frequency_1 = 6 →
  call_frequency_2 = 8 →
  call_frequency_3 = 9 →
  total_days = 366 →
  let calls := (total_days / call_frequency_1) + 
               (total_days / call_frequency_2) + 
               (total_days / call_frequency_3) - 
               (total_days / Nat.lcm call_frequency_1 call_frequency_2) - 
               (total_days / Nat.lcm call_frequency_1 call_frequency_3) - 
               (total_days / Nat.lcm call_frequency_2 call_frequency_3) + 
               (total_days / Nat.lcm (Nat.lcm call_frequency_1 call_frequency_2) call_frequency_3)
  in total_days - calls = 255 :=
by 
  intros _ _ _ _ h1 h2 h3 h4
  rsuffices (calls : ℕ) 
  sorry

end no_calls_days_l427_427089


namespace supplement_of_complement_of_35_degree_angle_l427_427187

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_of_35_degree_angle : 
  supplement (complement 35) = 125 := 
by sorry

end supplement_of_complement_of_35_degree_angle_l427_427187


namespace registration_methods_l427_427435

-- Define the number of students and groups
def num_students : ℕ := 4
def num_groups : ℕ := 3

-- Theorem stating the total number of different registration methods
theorem registration_methods : (num_groups ^ num_students) = 81 := 
by sorry

end registration_methods_l427_427435


namespace square_of_radius_of_inscribed_circle_l427_427662

theorem square_of_radius_of_inscribed_circle :
  ∃ (r : ℝ), 
    (∃ (a b c d : ℝ), a = 17 ∧ b = 19 ∧ c = 41 ∧ d = 31) →
    r^2 = 1040 :=
begin
  sorry
end

end square_of_radius_of_inscribed_circle_l427_427662


namespace ratio_side_lengths_sum_l427_427147

open Real

def ratio_of_areas (a b : ℝ) : ℝ := a / b

noncomputable def sum_of_abc (area_ratio : ℝ) : ℤ :=
  let side_length_ratio := sqrt area_ratio
  let a := 2
  let b := 1
  let c := 1
  a + b + c

theorem ratio_side_lengths_sum :
  sum_of_abc (ratio_of_areas 300 75) = 4 :=
by
  sorry

end ratio_side_lengths_sum_l427_427147


namespace remainder_13_plus_y_l427_427527

theorem remainder_13_plus_y :
  (∃ y : ℕ, (0 < y) ∧ (7 * y ≡ 1 [MOD 31])) → (∃ y : ℕ, (13 + y ≡ 22 [MOD 31])) :=
by 
  sorry

end remainder_13_plus_y_l427_427527


namespace coefficient_of_x3y7_in_expansion_l427_427184

-- Definitions based on the conditions in the problem
def a : ℚ := (2 / 3)
def b : ℚ := - (3 / 4)
def n : ℕ := 10
def k1 : ℕ := 3
def k2 : ℕ := 7

-- Statement of the math proof problem
theorem coefficient_of_x3y7_in_expansion :
  (a * x ^ k1 + b * y ^ k2) ^ n = x3y7_coeff * x ^ k1 * y ^ k2  :=
sorry

end coefficient_of_x3y7_in_expansion_l427_427184


namespace cartesian_eq_of_polar_max_min_values_l427_427816

-- Definition of the parametric equation of the line
def line_param (t : ℝ) : ℝ × ℝ :=
  (1 - (sqrt 3 / 2) * t, sqrt 3 + (1 / 2) * t)

-- Definition of the polar equation of the circle
def circle_polar_eq (ρ θ : ℝ) : Prop :=
  ρ = 4 * cos (θ - π / 3)

-- Definition of the Cartesian equation of the circle
def circle_cartesian_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*sqrt 3*y = 0

-- The main theorem statements for Lean 4
theorem cartesian_eq_of_polar (ρ θ : ℝ) : 
  circle_polar_eq ρ θ →
  ∃ x y : ℝ, circle_cartesian_eq x y := sorry

theorem max_min_values (t : ℝ) : 
  -2 ≤ t ∧ t ≤ 2 →
  ∃ x y : ℝ, line_param t = (x, y) ∧ 
  (sqrt 3 * x + y = 2 * sqrt 3 - t) → 
  -- Boundaries for maximum and minimum value
  (2 * sqrt 3 - 2) ≤ (2 * sqrt 3 - t) ∧ (2 * sqrt 3 - t) ≤ (2 * sqrt 3 + 2) := sorry

end cartesian_eq_of_polar_max_min_values_l427_427816


namespace product_area_perimeter_l427_427094

def square_points (E F G H : ℝ × ℝ) : Prop :=
  E = (4, 5) ∧ F = (6, 2) ∧ G = (2, 0) ∧ H = (0, 3)

theorem product_area_perimeter (E F G H : ℝ × ℝ) (h : square_points E F G H) : 
  let EH := real.sqrt ((E.1 - H.1) ^ 2 + (E.2 - H.2) ^ 2),
      area := EH ^ 2,
      perimeter := 4 * EH
  in area * perimeter = 160 * real.sqrt 5 := 
sorry

end product_area_perimeter_l427_427094


namespace sqrt_approx_difference_l427_427222

theorem sqrt_approx_difference :
  (10 : ℝ) - (real.sqrt 96) ≈ 0.20 :=
begin
  sorry
end

end sqrt_approx_difference_l427_427222


namespace triangle_DEF_area_l427_427675

-- Definitions
variables (Δ_DEF : Type) [fintype Δ_DEF] -- Δ_DEF representing the triangle DEF
variable (Q : Δ_DEF) -- Point Q inside the triangle DEF

-- Areas of smaller triangles
variables (area_s1 : ℕ) (area_s2 : ℕ) (area_s3 : ℕ)
variables (area_DEF : ℕ)

-- Conditions
axiom h1 : area_s1 = 16
axiom h2 : area_s2 = 25
axiom h3 : area_s3 = 36

-- Theorem
theorem triangle_DEF_area : area_DEF = 1140 :=
by sorry

end triangle_DEF_area_l427_427675


namespace slope_angle_at_1_eq_2pi_over_3_l427_427749

-- Define the function f(x)
def f (x : ℝ) : ℝ := - (Real.sqrt 3 / 3) * x^3 + 2

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := - Real.sqrt 3 * x^2

-- Define the angle of the slope of the tangent line
def slope_angle (x : ℝ) : ℝ := Real.arctan (f' x)

-- Define the problem statement
theorem slope_angle_at_1_eq_2pi_over_3 : slope_angle 1 = 2 * Real.pi / 3 :=
by
  sorry

end slope_angle_at_1_eq_2pi_over_3_l427_427749


namespace find_f500_l427_427497

variable (f : ℕ → ℕ)
variable (h : ∀ x y : ℕ, f (x * y) = f x + f y)
variable (h₁ : f 10 = 16)
variable (h₂ : f 40 = 24)

theorem find_f500 : f 500 = 44 :=
sorry

end find_f500_l427_427497


namespace work_time_l427_427641

-- Define the work rates of the man, his father, and his son
def man_work_rate := 1 / 10 : ℚ
def father_work_rate := 1 / 20 : ℚ
def son_work_rate := 1 / 25 : ℚ

-- Define the combined work rate
def combined_work_rate := man_work_rate + father_work_rate + son_work_rate

-- Prove that the total time taken is the reciprocal of the combined work rate
theorem work_time :
  1 / combined_work_rate ≈ 5.26 :=
by
  -- This proof is not provided
  sorry

end work_time_l427_427641


namespace original_proposition_converse_proposition_inverse_proposition_contrapositive_proposition_l427_427635

variable {α : Type}
variable (a b c : α)

theorem original_proposition (h : a * b * c = 0) : a = 0 ∨ b = 0 ∨ c = 0 := sorry

theorem converse_proposition (h : a = 0 ∨ b = 0 ∨ c = 0) : a * b * c = 0 := sorry

theorem inverse_proposition (h : a * b * c ≠ 0) : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 := sorry

theorem contrapositive_proposition (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) : a * b * c ≠ 0 := sorry

end original_proposition_converse_proposition_inverse_proposition_contrapositive_proposition_l427_427635


namespace largest_n_for_factored_quad_l427_427348

theorem largest_n_for_factored_quad (n : ℤ) (b d : ℤ) 
  (h1 : 6 * d + b = n) (h2 : b * d = 72) 
  (factorable : ∃ x : ℤ, (6 * x + b) * (x + d) = 6 * x ^ 2 + n * x + 72) : 
  n ≤ 433 :=
sorry

end largest_n_for_factored_quad_l427_427348


namespace kevin_total_distance_l427_427060

theorem kevin_total_distance :
  let d1 := 10 * 0.5,
      d2 := 20 * 0.5,
      d3 := 8 * 0.25 in
  d1 + d2 + d3 = 17 := by
  sorry

end kevin_total_distance_l427_427060


namespace largest_value_of_n_l427_427349

theorem largest_value_of_n (A B n : ℤ) (h1 : A * B = 72) (h2 : n = 6 * B + A) : n = 433 :=
sorry

end largest_value_of_n_l427_427349


namespace appointment_ways_l427_427859

theorem appointment_ways :
  let members := ["Alice", "Bob", "Carol", "Dave"]
  let roles := ["president", "secretary", "treasurer"]
  let total_ways := (members.combinations 3).sum (λ group, if "Alice" ∈ group then 2 * (group.length - 1)! else group.length!)
  in  total_ways = 18 :=
by {
  -- Define the members and roles
  let members := ["Alice", "Bob", "Carol", "Dave"],
  let roles := ["president", "secretary", "treasurer"],
  -- Calculate total ways
  let total_ways := (members.combinations 3).sum (λ group, 
    if "Alice" ∈ group then 2 * (group.erase "Alice").length! else group.length!),
  show total_ways = 18,
  sorry
}

end appointment_ways_l427_427859


namespace sum_of_common_ratios_of_sequences_l427_427904

def arithmetico_geometric_sequence (a b c : ℕ → ℝ) (r : ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = r * a n + d ∧ b (n + 1) = r * b n + d

theorem sum_of_common_ratios_of_sequences {m n : ℝ}
    {a1 a2 a3 b1 b2 b3 : ℝ}
    (p q : ℝ)
    (h_a1 : a1 = m)
    (h_a2 : a2 = m * p + 5)
    (h_a3 : a3 = m * p^2 + 5 * p + 5)
    (h_b1 : b1 = n)
    (h_b2 : b2 = n * q + 5)
    (h_b3 : b3 = n * q^2 + 5 * q + 5)
    (h_cond : a3 - b3 = 3 * (a2 - b2)) :
    p + q = 4 :=
by
  sorry

end sum_of_common_ratios_of_sequences_l427_427904


namespace part1_sales_volume_part2_price_reduction_l427_427921

noncomputable def daily_sales_volume (x : ℝ) : ℝ :=
  100 + 200 * x

noncomputable def profit_eq (x : ℝ) : Prop :=
  (4 - 2 - x) * (100 + 200 * x) = 300

theorem part1_sales_volume (x : ℝ) : daily_sales_volume x = 100 + 200 * x :=
sorry

theorem part2_price_reduction (hx : profit_eq (1 / 2)) : 1 / 2 = 1 / 2 :=
sorry

end part1_sales_volume_part2_price_reduction_l427_427921


namespace common_ratio_of_geom_seq_general_term_of_arith_seq_l427_427416

-- Definitions and conditions
def geom_seq (a : ℕ → ℝ) : Prop :=
  ∃ q > 0, a 2 = 4 ∧ a 4 = 16 ∧ ∀ n, a (n + 1) = a n * q

noncomputable def arith_seq (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  a 3 = b 3 ∧ a 5 = b 5 ∧ ∃ d, ∀ n, b (n + 1) = b n + d

-- Part 1: Prove the common ratio q
theorem common_ratio_of_geom_seq {a : ℕ → ℝ} (h : geom_seq a) : ∃ q, q = 2 :=
  sorry

-- Part 2: Prove the general term formula of the arithmetic sequence b_n
theorem general_term_of_arith_seq {a b : ℕ → ℝ} (h_geom : geom_seq a) (h_arith : arith_seq a b) :
  ∃ t : ℕ → ℝ, t = λ n, 12 * n - 28 :=
  sorry

end common_ratio_of_geom_seq_general_term_of_arith_seq_l427_427416


namespace problem_1_problem_2_l427_427245

-- Problem (1)
variables (a b : ℝ) (θ : ℝ)
hypothesis h1 : ∥a∥ = 3
hypothesis h2 : ∥b∥ = 4
hypothesis h3 : θ = real.pi / 3
def dot_product := a * b * real.cos θ
def norm_difference := real.sqrt (∥a∥^2 + ∥b∥^2 - 2 * (a * b * real.cos θ))

theorem problem_1 : dot_product a b θ = 6 ∧ norm_difference a b θ = real.sqrt 13 := sorry

-- Problem (2)
variables (a : ℝ × ℝ) (b : ℝ × ℝ)
hypothesis h4 : real.sqrt (a.1^2 + a.2^2) = 3
hypothesis h5 : b = (1, 2)
hypothesis h6 : ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem problem_2 : 
  (a = (-3 * real.sqrt 5 / 5, -6 * real.sqrt 5 / 5) ∨ 
   a = (3 * real.sqrt 5 / 5, 6 * real.sqrt 5 / 5)) := sorry

end problem_1_problem_2_l427_427245


namespace transformed_point_is_correct_l427_427593

def rotate_90_x (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match p with
  | (x, y, z) => (x, -z, y)

def reflect_xy_plane (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match p with
  | (x, y, z) => (x, y, -z)

def reflect_yz_plane (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match p with
  | (x, y, z) => (-x, y, z)

def rotate_90_y (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match p with
  | (x, y, z) => (z, y, -x)

def reflect_xz_plane (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match p with
  | (x, y, z) => (x, -y, z)

theorem transformed_point_is_correct :
  let p := (2, 2, 2)
  let p1 := rotate_90_x p
  let p2 := reflect_xy_plane p1
  let p3 := reflect_yz_plane p2
  let p4 := rotate_90_y p3
  let p5 := reflect_xz_plane p4
  p5 = (2, 2, -2) :=
by
  let p := (2, 2, 2)
  let p1 := rotate_90_x p
  let p2 := reflect_xy_plane p1
  let p3 := reflect_yz_plane p2
  let p4 := rotate_90_y p3
  let p5 := reflect_xz_plane p4
  show p5 = (2, 2, -2)
  sorry

end transformed_point_is_correct_l427_427593


namespace sum_of_c_and_d_l427_427840

theorem sum_of_c_and_d (c d : ℝ) 
  (h1 : ∀ x, x ≠ 2 ∧ x ≠ -1 → x^2 + c * x + d ≠ 0)
  (h_asymp_2 : 2^2 + c * 2 + d = 0)
  (h_asymp_neg1 : (-1)^2 + c * (-1) + d = 0) :
  c + d = -3 :=
by 
  -- Proof placeholder
  sorry

end sum_of_c_and_d_l427_427840


namespace height_proof_l427_427886

noncomputable def max_height_table (DE EF FD : ℝ) (h : ℝ) : ℝ :=
  let h_d := (2 * 30 * Real.sqrt 129) / EF
  let h_e := (2 * 30 * Real.sqrt 129) / FD
  let h_f := (2 * 30 * Real.sqrt 129) / DE
  /-
    Using similar triangles and properties of parallel lines:
    h ≤ min (h_d * h_e) / (h_d + h_e)), (h_d * h_f) / (h_d + h_f)), (h_e * h_f) / (h_e + h_f))
  -/
  Real.min ((h_d * h_e) / (h_d + h_e)) (Real.min ((h_d * h_f) / (h_d + h_f)) ((h_e * h_f) / (h_e + h_f)))

theorem height_proof :
  let DE := 25
  let EF := 28
  let FD := 33
  let h : ℝ := (60 * Real.sqrt 129) / 61
  h = max_height_table DE EF FD h → 60 + 129 + 61 = 250 :=
begin
  intros,
  simp *,
  sorry
end

end height_proof_l427_427886


namespace min_points_to_guarantee_highest_score_l427_427861

-- Define the points awarded for each position.
def points_awarded (position : ℕ) : ℕ :=
  if position = 1 then 7
  else if position = 2 then 4
  else if position = 3 then 2
  else if position = 4 then 1
  else 0

-- Define the condition for no ties and limit to three races.
def valid_positions (positions : list ℕ) : Prop :=
  positions.length = 3 ∧ positions.all (λ p, p ∈ [1, 2, 3, 4])

-- Define a function to calculate the total points for a list of positions.
def total_points (positions : list ℕ) : ℕ :=
  positions.map points_awarded |>.sum

-- Define the main problem statement
theorem min_points_to_guarantee_highest_score :
  ∀ (positions1 positions2 : list ℕ), valid_positions positions1 → valid_positions positions2 →
  total_points positions1 ≥ total_points positions2 →
  total_points positions1 = 18 :=
by sorry

end min_points_to_guarantee_highest_score_l427_427861


namespace prime_count_between_80_and_100_l427_427021

theorem prime_count_between_80_and_100 : 
  (finset.filter nat.prime (finset.Ico 80 101)).card = 3 :=
by
  sorry

end prime_count_between_80_and_100_l427_427021


namespace parabola_axis_of_symmetry_parabola_vertex_intersections_with_x_axis_l427_427405

noncomputable def parabola (x : ℝ) : ℝ :=
  x^2 - 8 * x + 12

namespace parabola_properties

def axis_of_symmetry : ℝ := 4

def vertex : ℝ × ℝ := (4, -4)

def intersection_points : List (ℝ × ℝ) := [(2, 0), (6, 0)]

theorem parabola_axis_of_symmetry (x : ℝ) :
  (parabola x = (x - axis_of_symmetry)^2 - 4) :=
begin
  sorry
end

theorem parabola_vertex : 
  ∃ h k, vertex = (h, k) ∧ parabola_properties.parabola h = k :=
begin
  use 4, use -4,
  sorry
end

theorem intersections_with_x_axis :
  ∃ x1 x2, intersection_points = [(x1, 0), (x2, 0)] ∧
    parabola x1 = 0 ∧ parabola x2 = 0 :=
begin
  use 2, use 6,
  sorry
end

end parabola_properties

end parabola_axis_of_symmetry_parabola_vertex_intersections_with_x_axis_l427_427405


namespace inequality_solution_l427_427136

-- We define the problem
def interval_of_inequality : Set ℝ := { x : ℝ | (x + 1) * (2 - x) > 0 }

-- We define the expected solution set
def expected_solution_set : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }

-- The theorem to be proved
theorem inequality_solution :
  interval_of_inequality = expected_solution_set := by 
  sorry

end inequality_solution_l427_427136


namespace house_selling_price_l427_427957

theorem house_selling_price (C S : ℝ)
  (h1 : .5 * S = 100000)
  (h2 : 1.5 * S - (C + 100000) = 60000) :
  S = 120000 :=
  sorry

end house_selling_price_l427_427957


namespace fraction_solution_l427_427576

theorem fraction_solution (a : ℤ) (h : 0 < a ∧ (a : ℚ) / (a + 36) = 775 / 1000) : a = 124 := 
by
  sorry

end fraction_solution_l427_427576


namespace abs_x_lt_2_sufficient_not_necessary_for_x_sq_minus_x_minus_6_lt_0_l427_427646

theorem abs_x_lt_2_sufficient_not_necessary_for_x_sq_minus_x_minus_6_lt_0 :
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧ (¬ ∀ x : ℝ, x^2 - x - 6 < 0 → |x| < 2) :=
by
  sorry

end abs_x_lt_2_sufficient_not_necessary_for_x_sq_minus_x_minus_6_lt_0_l427_427646


namespace answer_to_problem_l427_427019

noncomputable def count_n_satisfying_conditions : Nat :=
  let condition (n : Nat) : Prop :=
    n ≥ 2 ∧ 
    ∀ (z : Fin n → ℂ), 
    (∀ i, |z i| = 1) ∧
    (∑ i in Finset.univ, (z i) ^ 2 = 0) → 
    ∀ i j, 2 * π * (i - j) / n % (2 * π) ∈ {0, π}
  ∃ count : Nat, count = (Finset.filter condition (Finset.Icc 2 10)).card

theorem answer_to_problem : count_n_satisfying_conditions = 2 := by
  sorry

end answer_to_problem_l427_427019


namespace find_number_l427_427250

theorem find_number : 
  ∃ (x : ℝ), 0.30 * x = 30 + (0.60 * 50) → x = 200 :=
by
  intro x
  rw [mul_comm 0.60 50]
  have h1: 0.60 * 50 = 30 := by norm_num
  rw h1
  intro h
  exact eq_of_mul_eq_mul_left (by norm_num : 0.30 ≠ 0) h

end find_number_l427_427250


namespace supplement_of_complement_is_125_l427_427208

-- Definition of the initial angle
def initial_angle : ℝ := 35

-- Definition of the complement of the angle
def complement_angle (θ : ℝ) : ℝ := 90 - θ

-- Definition of the supplement of an angle
def supplement_angle (θ : ℝ) : ℝ := 180 - θ

-- Main theorem statement
theorem supplement_of_complement_is_125 : 
  supplement_angle (complement_angle initial_angle) = 125 := 
by
  sorry

end supplement_of_complement_is_125_l427_427208


namespace first_pipe_fills_in_8_minutes_l427_427099

noncomputable def time_to_fill_first_pipe (T : ℝ) : Prop :=
  let rate_first_pipe := 1 / T
  let rate_second_pipe := 1 / 12
  let combined_rate := 1 / 4.8

  rate_first_pipe + rate_second_pipe = combined_rate

theorem first_pipe_fills_in_8_minutes : ∃ (T : ℝ), time_to_fill_first_pipe T ∧ T = 8 :=
by
  use 8
  rw time_to_fill_first_pipe
  sorry

end first_pipe_fills_in_8_minutes_l427_427099


namespace simplify_expression_l427_427555

-- We define the given expressions and state the theorem.
variable (x : ℝ)

theorem simplify_expression : (3 * x - 4) * (x + 8) - (x + 6) * (3 * x - 2) = 4 * x - 20 := by
  -- Proof goes here
  sorry

end simplify_expression_l427_427555


namespace collinear_A_B_D_l427_427429

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b : V)
def vector_AB : V := a + 2 • b
def vector_BC : V := -5 • a + 6 • b
def vector_CD : V := 7 • a - 2 • b

theorem collinear_A_B_D :
  ∃ k : ℝ, vector_CD = k • vector_AB + vector_BC :=
sorry

end collinear_A_B_D_l427_427429


namespace moles_HBr_formed_l427_427358

theorem moles_HBr_formed 
    (moles_CH4 : ℝ) (moles_Br2 : ℝ) (reaction : ℝ) : 
    moles_CH4 = 1 ∧ moles_Br2 = 1 → reaction = 1 :=
by
  intros h
  cases h
  sorry

end moles_HBr_formed_l427_427358


namespace votes_for_eliot_l427_427470

theorem votes_for_eliot (randy_votes : ℕ) (shaun_votes : ℕ) (eliot_votes : ℕ)
  (h_randy : randy_votes = 16)
  (h_shaun : shaun_votes = 5 * randy_votes)
  (h_eliot : eliot_votes = 2 * shaun_votes) :
  eliot_votes = 160 :=
by
  sorry

end votes_for_eliot_l427_427470


namespace triangle_angle_C_30_degrees_l427_427887

theorem triangle_angle_C_30_degrees 
  (A B C : ℝ) 
  (h1 : 3 * Real.sin A + 4 * Real.cos B = 6) 
  (h2 : 4 * Real.sin B + 3 * Real.cos A = 1) 
  (h3 : A + B + C = 180) 
  : C = 30 :=
  sorry

end triangle_angle_C_30_degrees_l427_427887


namespace factorial_sequence_perfect_square_l427_427763

-- Definitions based on the conditions
def is_perf_square (x : ℕ) : Prop := ∃ (k : ℕ), x = k * k

def factorial (n : ℕ) : ℕ := Nat.recOn n 1 (λ n fac_n, (n + 1) * fac_n)

def factorial_seq_prod (n : ℕ) : ℕ :=
  Nat.recOn n 1 (λ n prodN, factorial (2 * n) * prodN)

-- Main statement
theorem factorial_sequence_perfect_square (n : ℕ) :
  is_perf_square (factorial_seq_prod n / factorial (n + 1)) ↔
  ∃ k : ℕ, n = 4 * k * (k + 1) ∨ n = 2 * k ^ 2 - 1 := by
  sorry

end factorial_sequence_perfect_square_l427_427763


namespace total_collisions_100_balls_l427_427649

def num_of_collisions (n: ℕ) : ℕ :=
  n * (n - 1) / 2

theorem total_collisions_100_balls :
  num_of_collisions 100 = 4950 :=
by
  sorry

end total_collisions_100_balls_l427_427649


namespace part1_part2_l427_427422

open Set

variable {α : Type*} [LinearOrder α]

def A (m : α) : Set α := { x | 0 < x - m ∧ x - m < 3 }
def B : Set α := { x | x ≤ 0 ∨ x ≥ 3 }

theorem part1 {α : Type*} [LinearOrder α] (m : α) (h : m = 1) :
  (A m ∩ B) = { x | 3 ≤ x ∧ x < 4 } :=
by
  rw [h]
  sorry

theorem part2 {α : Type*} [LinearOrder α] :
  (A ∩ B = B) → ∀ m, m ≥ 3 ∨ m ≤ -3 :=
by
  intro h
  sorry

end part1_part2_l427_427422


namespace orthocenter_min_AD2_plus_BE2_plus_CF2_l427_427050

noncomputable def is_triangle (A B C: ℝ) (u v w: ℝ) : Prop :=
  u + v > w ∧ u + w > v ∧ v + w > u

noncomputable def is_orthocenter_of_triangle 
  (A B C P D E F: ℝ) (u v w: ℝ) (h: is_triangle A B C u v w) : Prop :=
  ∃ z x y, 
    z = P - D ∧ x = P - E ∧ y = P - F ∧
    u^2 + v^2 = w^2 ∧ 
    A^2 + B^2 + C^2 = 77

theorem orthocenter_min_AD2_plus_BE2_plus_CF2 :
  ∀ (A B C P D E F: ℝ) (u v w: ℝ),
    is_triangle A B C u v w →
    u = 8 ∧ v = 12 ∧ w = 10 →
    ∃ (z x y : ℝ), 
      (z = P - D) ∧ (x = P - E) ∧ (y = P - F) ∧ 
      (u^2 + v^2 = w^2) ∧
      (A^2 + B^2 + C^2 = 77) →
      is_orthocenter_of_triangle A B C P D E F u v w sorry

end orthocenter_min_AD2_plus_BE2_plus_CF2_l427_427050


namespace no_n_satisfies_mod_5_l427_427729

theorem no_n_satisfies_mod_5 (n : ℤ) : (n^3 + 2*n - 1) % 5 ≠ 0 :=
by
  sorry

end no_n_satisfies_mod_5_l427_427729


namespace parabola_smallest_a_l427_427950

noncomputable def parabola_vertex (a b c : ℝ) : Prop :=
  (a > 0) ∧ (2 * a + b + 3 * c).denominator = 1 ∧
  (2 * a + b + 3 * c).numerator.mod 1 = 0

theorem parabola_smallest_a : ∃ a : ℝ, ∃ b c : ℝ, 
  (∃ vertex_y vertex_x : ℝ, vertex_x = -1/3 ∧ vertex_y = -1/9 ∧
  y = a * (x + vertex_x)^2 + vertex_y) ∧ 
  parabola_vertex a b c ∧ 
  a = 1/3 :=
by
  sorry

end parabola_smallest_a_l427_427950


namespace find_initial_girls_l427_427774

variable (b g : ℕ)

theorem find_initial_girls 
  (h1 : 3 * (g - 18) = b)
  (h2 : 4 * (b - 36) = g - 18) :
  g = 31 := 
by
  sorry

end find_initial_girls_l427_427774


namespace second_student_speed_l427_427616

theorem second_student_speed
  (d_initial : ℝ)
  (speed1 : ℝ)
  (time_meet : ℝ)
  (distance1 : ℝ)
  (distance2 : ℝ)
  (v : ℝ) :
  d_initial = 350 ∧ speed1 = 1.6 ∧ time_meet = 100 ∧ distance1 = speed1 * time_meet ∧ distance2 = v * time_meet ∧ distance1 + distance2 = d_initial → v = 1.9 :=
begin
  sorry
end

end second_student_speed_l427_427616


namespace F_arithmetic_mean_l427_427122

def F (n r : ℕ) : ℚ :=
  let subsets := {s : finset ℕ | s.card = r ∧ ∀ x ∈ s, x ≤ n}
  (∑ s in subsets, s.min' (finset.nonempty_of_ne_empty (by simp [ne_empty_of_card_pos _]))) / subsets.card

theorem F_arithmetic_mean : ∀ (n r : ℕ), 1 ≤ r → r ≤ n → F n r = (n + 1 : ℚ) / (r + 1) :=
  by
    sorry -- The proof will be inserted here

end F_arithmetic_mean_l427_427122


namespace range_of_x_l427_427002

noncomputable def f (x : ℝ) : ℝ := logBase 2 (sqrt (x ^ 2 + 1) - x) - x ^ 3

theorem range_of_x (x y : ℝ) (h : x + y = 2018) : f(x) + f(2018) > f(y) → x < 0 := sorry

end range_of_x_l427_427002


namespace supplement_of_complement_is_125_l427_427205

-- Definition of the initial angle
def initial_angle : ℝ := 35

-- Definition of the complement of the angle
def complement_angle (θ : ℝ) : ℝ := 90 - θ

-- Definition of the supplement of an angle
def supplement_angle (θ : ℝ) : ℝ := 180 - θ

-- Main theorem statement
theorem supplement_of_complement_is_125 : 
  supplement_angle (complement_angle initial_angle) = 125 := 
by
  sorry

end supplement_of_complement_is_125_l427_427205


namespace line_angle_construction_l427_427789

noncomputable theory

-- Given an angle (AOB) and a line (l)
variable {Point : Type}
variable (A O B : Point)
variable (l : set Point)

-- Constructing lines l1 such that the angle between l and l1 is equal to the angle AOB
variable [EuclideanGeometry Point]

theorem line_angle_construction (A O B : Point) (l : set Point) (hAOB : angle A O B)
  : ∃ l1 : set Point, angle_between l l1 = hAOB := 
sorry

end line_angle_construction_l427_427789


namespace right_triangle_to_square_l427_427315

theorem right_triangle_to_square :
  ∀ (A B C : Type) (AC BC : ℝ) (angleA : ℝ),
  AC = 1 ∧ angleA = 30 ∧ BC = sqrt 3 
  → (∃ parts : List (Set ℝ), 
        (∀ part ∈ parts, Is_subtriangle part (triangle A B C)) ∧ 
        can_reassemble_to_square parts (sqrt (sqrt 3 / 2))) :=
by
  sorry

end right_triangle_to_square_l427_427315


namespace tangent_line_slope_l427_427030

theorem tangent_line_slope (x₀ y₀ k : ℝ)
    (h_tangent_point : y₀ = x₀ + Real.exp (-x₀))
    (h_tangent_line : y₀ = k * x₀) :
    k = 1 - Real.exp 1 := 
sorry

end tangent_line_slope_l427_427030


namespace Petya_lives_in_sixth_entrance_l427_427101

-- Define the entrances and their numbering
def Entrance : Type := ℕ

-- Define Petya and Vasya's houses and their neighboring condition
def neighboring_houses (P V : Entrance) : Prop := |P - V| = 1

-- Assume Vasya lives in the fourth entrance
def Vasya_lives_in_fourth_entrance (V : Entrance) : Prop := V = 4

-- Express the condition on the shortest path equivalence for Petya reaching Vasya
def shortest_path_equivalence (P V : Entrance) : Prop :=
  ∀ A B D : ℕ, (A = P ∧ B = P ∧ D = V) → (dist A D = dist B D)

-- State the theorem that Petya lives in the sixth entrance
theorem Petya_lives_in_sixth_entrance (P V : Entrance) :
  neighboring_houses P V →
  Vasya_lives_in_fourth_entrance V →
  shortest_path_equivalence P V →
  P = 6 :=
by
  intros h1 h2 h3
  sorry

end Petya_lives_in_sixth_entrance_l427_427101


namespace smallest_w_l427_427838

theorem smallest_w (w : ℕ) (h1 : 1916 = 2^2 * 479) (h2 : w > 0) : w = 74145392000 ↔ 
  (∀ p e, (p = 2 ∨ p = 3 ∨ p = 5 ∨ p = 7 ∨ p = 11) → (∃ k, (1916 * w = p^e * k ∧ e ≥ if p = 2 then 6 else 3))) :=
sorry

end smallest_w_l427_427838


namespace smallest_number_ending_in_6_moved_front_gives_4_times_l427_427361

theorem smallest_number_ending_in_6_moved_front_gives_4_times (x m n : ℕ) 
  (h1 : n = 10 * x + 6)
  (h2 : 6 * 10^m + x = 4 * n) :
  n = 1538466 :=
by
  sorry

end smallest_number_ending_in_6_moved_front_gives_4_times_l427_427361


namespace count_of_n_with_terminating_decimal_and_non_zero_tenths_digit_l427_427755

-- Define the conditions and the proof problem
def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2^a * 5^b

def non_zero_tenths_digit (n : ℕ) : Prop :=
  let d := ((10 / n : ℚ) - ((10 / n : ℚ).to_int)) in
  d ≠ 0

theorem count_of_n_with_terminating_decimal_and_non_zero_tenths_digit :
  ( {n : ℕ | is_terminating_decimal n ∧ non_zero_tenths_digit n}.count ≤ 10) = 5 :=
by
  sorry

end count_of_n_with_terminating_decimal_and_non_zero_tenths_digit_l427_427755


namespace product_of_distances_l427_427796

noncomputable def distance_to_asymptote_1 (x y : ℝ) : ℝ := (abs (x + sqrt 3 * y)) / 2
noncomputable def distance_to_asymptote_2 (x y : ℝ) : ℝ := (abs (x - sqrt 3 * y)) / 2

def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

theorem product_of_distances (x y : ℝ) (h : hyperbola x y) :
  distance_to_asymptote_1 x y * distance_to_asymptote_2 x y = 3 / 4 := by
  sorry

end product_of_distances_l427_427796


namespace national_news_minutes_l427_427583

theorem national_news_minutes (total_duration : ℕ) (international_news : ℕ) (sports : ℕ) 
(weather_forecasts : ℕ) (advertising : ℕ) : 
  total_duration = 30 → international_news = 5 → sports = 5 → weather_forecasts = 2 → advertising = 6 → 
  ∃ national_news : ℕ, national_news = 12 :=
by
  intros h_total h_international h_sports h_weather h_advertising
  use total_duration - (international_news + sports + weather_forecasts + advertising)
  rw [h_total, h_international, h_sports, h_weather, h_advertising]
  norm_num
  sorry

end national_news_minutes_l427_427583


namespace quadratic_decreasing_condition_l427_427846

-- Define the quadratic function
def quadratic_function (x m : ℝ) : ℝ := (x - m)^2 - 1

-- Conditions and the proof problem wrapped as a theorem statement
theorem quadratic_decreasing_condition (m : ℝ) :
  (∀ x : ℝ, x ≤ 3 → quadratically_decreasing x m) → m ≥ 3 :=
sorry

-- Helper function defining the decreasing condition
def quadratically_decreasing (x m : ℝ) : Prop :=
∀ y : ℝ, y < x → quadratic_function y m > quadratic_function x m

end quadratic_decreasing_condition_l427_427846


namespace characters_in_book_is_700_l427_427535

noncomputable def total_characters_in_book : ℕ :=
  let x := 28 in
  25 * x

theorem characters_in_book_is_700
  (days_first : ℕ)
  (days_second : ℕ)
  (write_first : ℕ)
  (write_second : ℕ)
  (h1 : write_first = 25)
  (h2 : write_second = write_first + 3)
  (h3 : days_second = days_first - 3)
  (h_eq : write_first * days_first = write_second * days_second) :
  total_characters_in_book = 700 :=
by
  sorry

end characters_in_book_is_700_l427_427535


namespace count_monomials_l427_427046

def is_monomial (expr : String) : Bool :=
  match expr with
  | "ab/3" => true
  | "-2/3abc" => true
  | "0" => true
  | "-5" => true
  | _ => false

theorem count_monomials : 
  let expressions := ["ab/3", "-2/3abc", "0", "-5", "x-y", "2/x"]
  ∑ e in expressions, if is_monomial e then 1 else 0  = 4 :=
by
  sorry

end count_monomials_l427_427046


namespace abe_and_bob_colors_match_l427_427276

noncomputable def probability_matching_colors : ℚ :=
  let abe_colors := {green := 2, red := 2}
  let bob_colors := {green := 2, yellow := 2, red := 3}
  let prob_green : ℚ := (abe_colors.green / 4) * (bob_colors.green / 7)
  let prob_red : ℚ := (abe_colors.red / 4) * (bob_colors.red / 7)
  prob_green + prob_red

theorem abe_and_bob_colors_match : probability_matching_colors = 5 / 14 :=
  sorry

end abe_and_bob_colors_match_l427_427276


namespace trapezium_angle_equality_l427_427647

structure Trapezium (A B C D : Type) [Point A] [Point B] [Point C] [Point D] :=
  (AD_parallel_BC : ∥(AD)∥ = ∥(BC)∥)

variables {A B C D P Q : Type} [Point A] [Point B] [Point C] [Point D] [Point P] [Point Q]

noncomputable def maximal_angle_CPD (CPD_max : angle C P D) : Prop := 
  ∀ Q, angle C P D ≥ angle C Q D

noncomputable def maximal_angle_BQA (BQA_max : angle B Q A) : Prop := 
  ∀ P, angle B Q A ≥ angle B P A

theorem trapezium_angle_equality (t : Trapezium A B C D) (hP : P ∈ line A B) 
  (hQ : Q ∈ line C D) (hP_max : maximal_angle_CPD (angle C P D))
  (hQ_max : maximal_angle_BQA (angle B Q A)) : angle C P D = angle B Q A :=
by {
  sorry
}

end trapezium_angle_equality_l427_427647


namespace books_bought_l427_427709

theorem books_bought (cost_crayons cost_calculators total_money cost_per_bag bags_bought cost_per_book remaining_money books_bought : ℕ) 
  (h1: cost_crayons = 5 * 5)
  (h2: cost_calculators = 3 * 5)
  (h3: total_money = 200)
  (h4: cost_per_bag = 10)
  (h5: bags_bought = 11)
  (h6: remaining_money = total_money - (cost_crayons + cost_calculators) - (bags_bought * cost_per_bag)) :
  books_bought = remaining_money / cost_per_book → books_bought = 10 :=
by
  sorry

end books_bought_l427_427709


namespace cosine_identity_in_pentagon_l427_427880

noncomputable def angles_in_pentagon (A B C D E : ℝ) := 
  A + B + C + D + E = 540 

theorem cosine_identity_in_pentagon (A B C D E : ℝ) 
  (h1 : A = 95) (h2 : B = 105) (h3 : angles_in_pentagon A B C D E) :
  cos A = cos C + cos D - cos (C + D) - 1 / 2 := 
sorry

end cosine_identity_in_pentagon_l427_427880


namespace butterfly_back_at_origin_l427_427655

-- The butterfly's movement is represented in the complex plane
noncomputable def theta : ℂ := Complex.exp (Real.pi * Complex.I / 4)

-- The butterfly's position after k steps
noncomputable def Q (k : ℕ) : ℂ :=
  ∑ i in Finset.range k, 2 * theta^i

-- The final position after 1024 steps
noncomputable def Q_final : ℂ := Q 1024

-- Prove the butterfly is back at the origin after 1024 steps
theorem butterfly_back_at_origin : Complex.abs Q_final = 0 := sorry

end butterfly_back_at_origin_l427_427655


namespace solve_real_range_a_l427_427663

noncomputable def condition1 (z : ℂ) : Prop :=
  (z + complex.I * 2).im = 0

noncomputable def condition2 (z : ℂ) : Prop :=
  ((z * (2 + complex.I)) / 5).im = 0

noncomputable def condition3 (z a : ℂ) : Prop :=
  let w := (z + complex.I * a) * (z + complex.I * a)
  in w.re > 0 ∧ w.im > 0

noncomputable def real_range_a (z : ℂ) : Set ℝ :=
  { a : ℝ | 2 < a ∧ a < 4 }

theorem solve_real_range_a (z : ℂ) :
  condition1 z → condition2 z → 
  ∀ a : ℝ, condition3 z a → a ∈ real_range_a z :=
sorry

end solve_real_range_a_l427_427663


namespace supplement_of_complement_of_35_degree_angle_l427_427190

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_of_35_degree_angle : 
  supplement (complement 35) = 125 := 
by sorry

end supplement_of_complement_of_35_degree_angle_l427_427190


namespace license_plate_count_l427_427432

theorem license_plate_count : (26^3 * 5 * 5 * 4) = 1757600 := 
by 
  sorry

end license_plate_count_l427_427432


namespace equal_points_in_tournament_l427_427858

-- Definitions specific to the conditions
def round_robin_tournament (teams : ℕ) := ∀ i j : ℕ, i ≠ j → game_result

structure game_result :=
  (team1 : ℕ)
  (team2 : ℕ)
  (result : Result)

inductive Result
| win (team : ℕ) : Result
| draw : Result
| loss (team : ℕ) : Result

-- The main theorem
theorem equal_points_in_tournament :
  ∀ (teams : ℕ),
  teams = 28 →
  (∃ G : round_robin_tournament 28,
    ∃ N : ℕ,
      N > (3 / 4) * (teams * (teams - 1) / 2) →
      (∃ (points : ℕ → ℕ) (i j : ℕ), i ≠ j ∧ points i = points j)) := 
sorry

end equal_points_in_tournament_l427_427858


namespace conclusion1_conclusion2_conclusion4_l427_427673

-- Define the new operation
def at (a b: ℝ) : ℝ := (a + b)^2 - (a - b)^2

-- Define the proof statements that need to be proven
theorem conclusion1 : at 1 (-2) = -8 := sorry

theorem conclusion2 (a b: ℝ) : at a b = at b a := sorry

theorem conclusion4 (a b: ℝ) (h: a + b = 0) : at a a + at b b = 8 * a^2 := sorry

end conclusion1_conclusion2_conclusion4_l427_427673


namespace reduction_percentage_increase_percentage_final_price_l427_427594

open Real

-- Define the original price and reductions
def original_price : ℝ := 500
def reduction_amount : ℝ := 300
def first_discount : ℝ := 0.10
def second_discount : ℝ := 0.15

-- 1. Prove the reduction percentage
theorem reduction_percentage : 
  (reduction_amount / original_price) * 100 = 60 := 
by
  sorry

-- 2. Prove the increase percentage to return to original price
theorem increase_percentage :
  ((original_price - reduction_amount) / reduction_amount) * 100 = 150 :=
by
  sorry

-- 3. Prove the final price after all discounts
theorem final_price : 
  let reduced_price := original_price - reduction_amount in
  let price_after_first_discount := reduced_price * (1 - first_discount) in
  price_after_first_discount * (1 - second_discount) = 153 :=
by
  sorry

end reduction_percentage_increase_percentage_final_price_l427_427594


namespace sum_of_all_possible_values_of_N_l427_427595

noncomputable def sum_of_possible_N : ℕ :=
  let pairs := [(1, 16), (2, 8), (4, 4)] in
  let N := pairs.map (λ p, let (a, b) := p in a * b * (a + b)) in
  N.sum

theorem sum_of_all_possible_values_of_N : sum_of_possible_N = 560 :=
by {
  sorry
}

end sum_of_all_possible_values_of_N_l427_427595


namespace oxygen_mass_percentage_is_58_3_l427_427356

noncomputable def C_molar_mass := 12.01
noncomputable def H_molar_mass := 1.01
noncomputable def O_molar_mass := 16.0

noncomputable def molar_mass_C6H8O7 :=
  6 * C_molar_mass + 8 * H_molar_mass + 7 * O_molar_mass

noncomputable def O_mass := 7 * O_molar_mass

noncomputable def oxygen_mass_percentage_C6H8O7 :=
  (O_mass / molar_mass_C6H8O7) * 100

theorem oxygen_mass_percentage_is_58_3 :
  oxygen_mass_percentage_C6H8O7 = 58.3 := by
  sorry

end oxygen_mass_percentage_is_58_3_l427_427356


namespace find_values_of_A_l427_427602

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem find_values_of_A (A B C : ℕ) :
  sum_of_digits A = B ∧
  sum_of_digits B = C ∧
  A + B + C = 60 →
  (A = 44 ∨ A = 50 ∨ A = 47) :=
by
  sorry

end find_values_of_A_l427_427602


namespace both_shots_hit_target_exactly_one_shot_hits_target_l427_427035

variable (p q : Prop)

theorem both_shots_hit_target : (p ∧ q) := sorry

theorem exactly_one_shot_hits_target : ((p ∧ ¬ q) ∨ (¬ p ∧ q)) := sorry

end both_shots_hit_target_exactly_one_shot_hits_target_l427_427035


namespace smallest_n_exists_l427_427731

theorem smallest_n_exists (n : ℕ) (hn : n ≥ 9) (s : Finset ℤ) (hs : s.card = n) :
  ∃ a b c d ∈ s, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  20 ∣ (a + b - c - d) := by
  sorry

end smallest_n_exists_l427_427731


namespace equilateral_triangle_properties_l427_427587

theorem equilateral_triangle_properties (ABC A1B1C1 : Triangle)
  (H1 : ABC.is_equilateral) (H2 : A1B1C1.is_equilateral)
  (M : Point) (H3 : M.is_midpoint_of ABC.BC)
  (H4 : M.is_midpoint_of A1B1C1.B1C1) :
  (angle ABC.A A1B1C1.A = 90) ∧ (length ABC.A A1B1C1.A / length ABC.B A1B1C1.B = Math.sqrt 3) := sorry

end equilateral_triangle_properties_l427_427587


namespace degree_meas_supp_compl_35_l427_427203

noncomputable def degree_meas_supplement_complement (θ : ℝ) : ℝ :=
  180 - (90 - θ)

theorem degree_meas_supp_compl_35 : degree_meas_supplement_complement 35 = 125 :=
by
  unfold degree_meas_supplement_complement
  norm_num
  sorry

end degree_meas_supp_compl_35_l427_427203


namespace total_number_of_members_l427_427454

-- Define the conditions
variables (B T B_inter_T Neither N : ℕ)

-- The conditions given in the problem
def condition1 : Prop := B = 20
def condition2 : Prop := T = 18
def condition3 : Prop := B_inter_T = 3
def condition4 : Prop := Neither = 5
def total_members : Prop := N = B + T - B_inter_T + Neither

-- The statement to prove
theorem total_number_of_members (B T B_inter_T Neither N : ℕ) 
  (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) (h5 : total_members) : N = 40 := 
by sorry

end total_number_of_members_l427_427454


namespace probability_of_license_expected_value_attempts_l427_427680

-- Lean statement for the probability of obtaining a driver's license
theorem probability_of_license (p1 p2 p3 : ℝ) (hp1 : p1 = 0.9) (hp2 : p2 = 0.7) (hp3 : p3 = 0.6) :
  p1 * p2 * p3 = 0.378 :=
by {
  rw [hp1, hp2, hp3], -- Substitute given probabilities
  norm_num, -- Evaluate the multiplication
}

-- Lean statement for the expected value of the number of attempts
theorem expected_value_attempts (p1 p2 p3 : ℝ) (hp1 : p1 = 0.1) (hp2 : p2 = 0.27) (hp3 : p3 = 0.63) :
  1 * p1 + 2 * p2 + 3 * p3 = 2.53 :=
by {
  rw [hp1, hp2, hp3], -- Substitute given probabilities
  norm_num, -- Evaluate the expression
}

end probability_of_license_expected_value_attempts_l427_427680


namespace tan_fraction_simplification_l427_427313

theorem tan_fraction_simplification :
  (tan (105 * Real.pi / 180) - tan (45 * Real.pi / 180)) / (1 + tan (105 * Real.pi / 180) * tan (45 * Real.pi / 180)) = Real.sqrt 3 := by

  -- We'll finish the proof later
  sorry

end tan_fraction_simplification_l427_427313


namespace molecular_weight_of_one_mole_l427_427219

-- Definitions as Conditions
def total_molecular_weight := 960
def number_of_moles := 5

-- The theorem statement
theorem molecular_weight_of_one_mole :
  total_molecular_weight / number_of_moles = 192 :=
by
  sorry

end molecular_weight_of_one_mole_l427_427219


namespace distance_from_vertex_to_plane_correct_l427_427260

theorem distance_from_vertex_to_plane_correct :
  ∃ (p q u : ℕ), (p = 9) ∧ (q = 186) ∧ (u = 1) ∧ p + q + u = 196 ∧
                 ∀ (a b c d : ℝ), a^2 + b^2 + c^2 = 1 →
                   (8 * d - (8 - d)^2 - (9 - d)^2 - (10 - d)^2 = 0) →
                   d = (54 - sqrt 744) / 6 :=
begin
  sorry
end

end distance_from_vertex_to_plane_correct_l427_427260


namespace opposite_neg_two_l427_427977

theorem opposite_neg_two : -(-2) = 2 := by
  sorry

end opposite_neg_two_l427_427977


namespace ratio_of_walkway_to_fountain_l427_427451

theorem ratio_of_walkway_to_fountain (n s d : ℝ) (h₀ : n = 10) (h₁ : n^2 * s^2 = 0.40 * (n*s + 2*n*d)^2) : 
  d / s = 1 / 3.44 := 
sorry

end ratio_of_walkway_to_fountain_l427_427451


namespace votes_for_eliot_l427_427469

theorem votes_for_eliot (randy_votes : ℕ) (shaun_votes : ℕ) (eliot_votes : ℕ)
  (h_randy : randy_votes = 16)
  (h_shaun : shaun_votes = 5 * randy_votes)
  (h_eliot : eliot_votes = 2 * shaun_votes) :
  eliot_votes = 160 :=
by
  sorry

end votes_for_eliot_l427_427469


namespace length_of_MR_l427_427784

-- Definition of the problem conditions
structure Pyramid :=
(apex : Point)
(A B C M R : Point)
(SA : dist apex A = 2 * sqrt 3)
(BC : dist B C = 3)
(BM_median : is_median B M apex A B C)
(AR_height : is_height apex A B apex A R)

-- The theorem statement to prove
theorem length_of_MR (p : Pyramid)
  (h_pyramid : regular_triangular_pyramid p.apex p.A p.B p.C)
  (MR_length : dist p.M p.R = sqrt 14 / 4) : 
  dist p.M p.R = sqrt 14 / 4 := 
sorry

end length_of_MR_l427_427784


namespace find_px_l427_427676

theorem find_px (p : ℕ → ℚ) (h1 : p 1 = 1) (h2 : p 2 = 1 / 4) (h3 : p 3 = 1 / 9) 
  (h4 : p 4 = 1 / 16) (h5 : p 5 = 1 / 25) : p 6 = 1 / 18 :=
sorry

end find_px_l427_427676


namespace staff_discount_l427_427262

theorem staff_discount (d : ℝ) (S : ℝ) (h1 : d > 0)
    (h2 : 0.455 * d = (1 - S / 100) * (0.65 * d)) : S = 30 := by
    sorry

end staff_discount_l427_427262


namespace S_contains_finite_but_not_infinite_arith_progressions_l427_427941

noncomputable def S : Set ℤ := {n | ∃ k : ℕ, n = Int.floor (k * Real.pi)}

theorem S_contains_finite_but_not_infinite_arith_progressions :
  (∀ (k : ℕ), ∃ (a d : ℤ), ∀ (i : ℕ) (h : i < k), (a + i * d) ∈ S) ∧
  ¬(∃ (a d : ℤ), ∀ (n : ℕ), (a + n * d) ∈ S) :=
by
  sorry

end S_contains_finite_but_not_infinite_arith_progressions_l427_427941


namespace monotonic_decrease_g_l427_427808

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (2 - x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := log a (1 - x^2)

theorem monotonic_decrease_g (a : ℝ) (ha : 1 < a) :
  ∀ x : ℝ, 0 ≤ x ∧ x < 1 → (f a x) = (g a x) → [0, 1) :=
by
  sorry

end monotonic_decrease_g_l427_427808


namespace no_magic_square_l427_427724

theorem no_magic_square :
  ¬ (∃ (C : Fin 10 → Fin 10 → ℕ), 
      (∀ i, ∑ j, C i j = 505) ∧ 
      (∀ j, ∑ i, C i j = 505) ∧ 
      (∀ k, ∑ (i : Fin 10), C i ((i + k) % 10) = 505) ∧
      (∑ (i j : Fin 10), C i j = 5050)) :=
sorry

end no_magic_square_l427_427724


namespace most_likely_outcome_is_draw_l427_427652

variable (P_A_wins : ℝ) (P_A_not_loses : ℝ)

def P_draw (P_A_wins P_A_not_loses : ℝ) : ℝ := 
  P_A_not_loses - P_A_wins

def P_B_wins (P_A_not_loses P_A_wins : ℝ) : ℝ :=
  1 - P_A_not_loses

theorem most_likely_outcome_is_draw 
  (h₁: P_A_wins = 0.3) 
  (h₂: P_A_not_loses = 0.7)
  (h₃: 0 ≤ P_A_wins) 
  (h₄: P_A_wins ≤ 1) 
  (h₅: 0 ≤ P_A_not_loses) 
  (h₆: P_A_not_loses ≤ 1) : 
  max (P_A_wins) (max (P_B_wins P_A_not_loses P_A_wins) (P_draw P_A_wins P_A_not_loses)) = P_draw P_A_wins P_A_not_loses :=
by
  sorry

end most_likely_outcome_is_draw_l427_427652


namespace attention_focused_l427_427549

/-
  Definitions for the variation of students' attention levels over time t (in minutes) 
  f(t) being larger indicates more concentrated attention
-/
def f1 (t : ℝ) : ℝ := -t^2 + 26 * t + 80 -- for 0 < t ≤ 10
def f2 : ℝ := 240 -- for 10 ≤ t ≤ 20
def f3 (k : ℝ) (t : ℝ) : ℝ := k * t + 400 -- for 20 ≤ t ≤ 40

theorem attention_focused (k : ℝ) (t1 t2 : ℝ) :
  (∀ t, 0 < t ∧ t ≤ 10 → f1 t = -t^2 + 26 * t + 80) →
  (∀ t, 10 ≤ t ∧ t ≤ 20 → f1 t = 240) →
  (∀ t, 20 ≤ t ∧ t ≤ 40 → f3 k t = k * t + 400) →
  f1 10 = 240 →
  ∃ k = -8, 
  ∃ t1 = 10, ∃ t2 = 20, 
  ∀ t, 0 < t ∧ t ≤ 40 → 
    (if 0 < t ∧ t ≤ 10 then f1 t else if 10 ≤ t ∧ t ≤ 20 then f1 t else f3 k t) ∃ (interval_sufficient : 26.875 - 5 < 24),
    (∀ t, 5 ≤ t ∧ t <= 26.875 → 
      (if 5 < t ∧ t ≤ 10 then f1 t else if 10 ≤ t ∧ t ≤ 20 then f1 t else f3 k t ≥ 185)).
sorry

end attention_focused_l427_427549


namespace angle_C_measure_l427_427086

theorem angle_C_measure (p q : Line) (h_parallel : p || q)
  (A B C : Angle) (h_A : A = B / 4) (h_A_C : C = A)
  (h_line : B + C = 180) :
  C = 36 := 
by 
  sorry

end angle_C_measure_l427_427086


namespace smallest_repeating_block_length_of_11_over_13_l427_427431

-- Define the fraction 11/13
def frac : ℚ := 11 / 13

-- Define the smallest repeating block length
def repeating_block_length : ℕ := 6

-- State the theorem
theorem smallest_repeating_block_length_of_11_over_13 :
  ∀ s : String, s.length = repeating_block_length → (frac.to_decimal_string = "0." ++ s ++ s) :=
sorry

end smallest_repeating_block_length_of_11_over_13_l427_427431


namespace remainder_77_pow_77_minus_15_mod_19_l427_427360

theorem remainder_77_pow_77_minus_15_mod_19 : (77^77 - 15) % 19 = 5 := by
  sorry

end remainder_77_pow_77_minus_15_mod_19_l427_427360


namespace matrix_P_property_l427_427728

def mat_P : Matrix (Fin 3) (Fin 3) ℝ := ![![3, 0, 0], ![0, 0, 1], ![0, 1, 0]]

theorem matrix_P_property
  (Q : Matrix (Fin 3) (Fin 3) ℝ) :
  mat_P ⬝ Q = ![
    ![3 * Q 0 0, 3 * Q 0 1, 3 * Q 0 2], 
    ![Q 2 0, Q 2 1, Q 2 2], 
    ![Q 1 0, Q 1 1, Q 1 2]] :=
by
  sorry

end matrix_P_property_l427_427728


namespace inclination_angle_of_line_cartesian_eq_of_curve_sum_of_distances_l427_427005

variables (t : ℝ) (x y : ℝ) (θ ρ : ℝ)
def line_l := x = t ∧ y = (sqrt 2 / 2) + sqrt 3 * t
def curve_C := ρ = 2 * cos (θ - π / 4)

theorem inclination_angle_of_line :
  line_l t x y → θ = π / 3 :=
sorry

theorem cartesian_eq_of_curve :
  curve_C ρ θ → (x - sqrt 2 / 2) ^ 2 + (y - sqrt 2 / 2) ^ 2 = 1 :=
sorry

theorem sum_of_distances (PA PB : ℝ) :
  line_l t x y ∧ curve_C ρ θ → PA + PB = sqrt 10 / 2 :=
sorry

end inclination_angle_of_line_cartesian_eq_of_curve_sum_of_distances_l427_427005


namespace find_pairs_l427_427323

theorem find_pairs (a b : ℕ) (h1 : a + b = 60) (h2 : Nat.lcm a b = 72) : (a = 24 ∧ b = 36) ∨ (a = 36 ∧ b = 24) := 
sorry

end find_pairs_l427_427323


namespace simplify_expression_l427_427557

theorem simplify_expression (x : ℝ) : 
  (3 * x - 4) * (x + 8) - (x + 6) * (3 * x - 2) = 4 * x - 20 := 
by
  sorry

end simplify_expression_l427_427557


namespace construction_of_triangle_l427_427305

-- Define the basic points A, M, and S
variables (A M S : Point)

-- Assume the existence of the required points to form the triangle
def exists_triangle_with_given_conditions (A M S : Point) : Prop :=
  ∃ (B C : Point), 
    is_vertex A ∧ 
    is_orthocenter A M S B C ∧
    is_centroid A M S B C 

-- The following statement is the Lean equivalent of the proof problem
theorem construction_of_triangle (A M S : Point) : 
  exists_triangle_with_given_conditions A M S := 
sorry

end construction_of_triangle_l427_427305


namespace seating_arrangements_l427_427317

open Finset

theorem seating_arrangements (n : ℕ) (h_n : n = 8) : 
  let ways_to_choose := choose n 7,
      factorial := ∏ i in range' 1 7, i
  in ways_to_choose * factorial = 5760 :=
by
  sorry

end seating_arrangements_l427_427317


namespace opposite_of_neg_two_l427_427972

theorem opposite_of_neg_two : -(-2) = 2 :=
by
  sorry

end opposite_of_neg_two_l427_427972


namespace range_of_a_l427_427817

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) → (0 < a ∧ a < 1) :=
by
  sorry

end range_of_a_l427_427817


namespace g_neg_five_l427_427918

def g (x : ℝ) : ℝ :=
if x < 0 then 3 * x + 4 else 7 - 3 * x

theorem g_neg_five : g (-5) = -11 := by
  sorry

end g_neg_five_l427_427918


namespace trigonometric_order_l427_427591

theorem trigonometric_order (a b c : ℝ) (h₁ : a ∈ set.Ioo 0 (Real.pi / 2))
  (h₂ : b ∈ set.Ioo 0 (Real.pi / 2)) (h₃ : c ∈ set.Ioo 0 (Real.pi / 2))
  (h₄ : Real.cos a = a) (h₅ : Real.sin (Real.cos b) = b) (h₆ : Real.cos (Real.sin c) = c) : 
  b < a ∧ a < c :=
sorry

end trigonometric_order_l427_427591


namespace sum_of_distinct_real_numbers_l427_427503

theorem sum_of_distinct_real_numbers (p q r s : ℝ) (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : ∀ x : ℝ, x^2 - 12 * p * x - 13 * q = 0 -> (x = r ∨ x = s)) 
  (h2 : ∀ x : ℝ, x^2 - 12 * r * x - 13 * s = 0 -> (x = p ∨ x = q)) :
  p + q + r + s = 2028 :=
begin
  sorry
end

end sum_of_distinct_real_numbers_l427_427503


namespace simplify_trig_expr_l427_427560

theorem simplify_trig_expr (x y : ℝ) :
  sin (x + y) * cos x - cos (x + y) * sin x = sin y :=
by
  sorry

end simplify_trig_expr_l427_427560


namespace remainder_1582_times_2031_mod_600_l427_427221

theorem remainder_1582_times_2031_mod_600 :
  (1582 * 2031) % 600 = 42 :=
by
  have h1 : 1582 % 600 = -18 := sorry
  have h2 : 2031 % 600 = 231 := sorry
  have h3 : (-18 * 231) % 600 = -4158 % 600 := sorry
  have h4 : -4158 % 600 = 42 := sorry
  exact_mod_cast h4

end remainder_1582_times_2031_mod_600_l427_427221


namespace problem_1_problem_2_l427_427246

noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) / (x - 2)

theorem problem_1 (a b : ℝ) (h1 : f a b 3 - 3 + 12 = 0) (h2 : f a b 4 - 4 + 12 = 0) :
  f a b x = (2 - x) / (x - 2) := sorry

theorem problem_2 (k : ℝ) (h : k > 1) :
  ∀ x, f (-1) 2 x < k ↔ (if 1 < k ∧ k < 2 then (1 < x ∧ x < k) ∨ (2 < x) 
                         else if k = 2 then 1 < x ∧ x ≠ 2 
                         else (1 < x ∧ x < 2) ∨ (k < x)) := sorry

-- Function definition for clarity
noncomputable def f_spec (x : ℝ) : ℝ := (2 - x) / (x - 2)

end problem_1_problem_2_l427_427246


namespace triangle_concurrencies_proof_l427_427113

noncomputable def triangle_concurrencies (A B C : Point) : Prop :=
  ∃ (P : Point), is_circumcenter A B C P ∧
    ∃ (I : Point), is_incenter A B C I ∧
    ∃ (G : Point), is_centroid A B C G ∧
    ∃ (H : Point), is_orthocenter A B C H

theorem triangle_concurrencies_proof (A B C : Point) (h : non_degenerate (triangle A B C)) :
  triangle_concurrencies A B C :=
sorry

end triangle_concurrencies_proof_l427_427113


namespace vandermonde_identity_l427_427112

open Nat

theorem vandermonde_identity (n m k : ℕ) (h : k ≤ min n m) :
  ∑ i in finset.range (k + 1), (nat.choose n i) * (nat.choose m (k - i)) = nat.choose (m + n) k := by
  sorry

end vandermonde_identity_l427_427112


namespace larry_stickers_l427_427479

theorem larry_stickers (start : ℕ) (lost : ℕ) : start = 93 → lost = 6 → start - lost = 87 :=
by {
  intros,
  subst_vars,
  sorry
}

end larry_stickers_l427_427479


namespace sum_of_first_11_terms_l427_427870

noncomputable def a (n : ℕ) : ℝ := sorry  -- Because a_n is part of the arithmetic sequence

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem sum_of_first_11_terms 
  (h1 : is_arithmetic_sequence a)
  (h2 : a 5 * a 7 = -6)
  (h3 : a 5 + a 7 = 2) :
  ∑ i in range 11, a i = 11 :=
by
  sorry

end sum_of_first_11_terms_l427_427870


namespace winner_vote_percentage_l427_427878

theorem winner_vote_percentage (N V W L : ℕ) (hN : N = 2000) (hV : V = 0.25 * N) 
  (hW : W = L + 50) (h_votes : W + L = V) : (W / V : ℚ) * 100 = 55 := 
by {
  sorry
}

end winner_vote_percentage_l427_427878


namespace cos_A_of_triangle_l427_427533

/-- Let the sides opposite to the angles A, B, C of triangle ABC be a, b, c respectively. 
Given that tan A / tan B = (3c - b) / b, then we prove cos A = 1 / 3. -/
theorem cos_A_of_triangle (a b c A B C : ℝ) 
  (h1: a = b * tan A / tan B)
  (h2: tan A / tan B = (3 * c - b) / b) :
  cos A = 1 / 3 := 
sorry

end cos_A_of_triangle_l427_427533


namespace triangle_crease_length_l427_427718

theorem triangle_crease_length (a b c : ℝ) (h₁ : a = 7) (h₂ : b = 24) (h₃ : c = 25) (h₀ : a^2 + b^2 = c^2) : 
  let m := c / 2 in 
  sqrt (b^2 - m^2) = 20.5 := 
by 
  sorry

end triangle_crease_length_l427_427718


namespace complementary_event_at_most_one_defective_l427_427247

theorem complementary_event_at_most_one_defective (A : ℕ → Prop) :
  (∀ n: ℕ, 2 ≤ n → A n) → (∀ n: ℕ, ¬ A n ↔ n ≤ 1) :=
by
  intros hA n
  split
  · intro hAn
    by_contradiction
    have hn := not_le_of_gt h
    exact hAn (hA n hn)
  · intro hn
    intro hAn
    exact nat.not_le_of_gt (hA n hn) hn

end complementary_event_at_most_one_defective_l427_427247


namespace find_abc_sum_l427_427120

theorem find_abc_sum (a b c : ℕ) (h : (a + b + c)^3 - a^3 - b^3 - c^3 = 294) : a + b + c = 8 :=
sorry

end find_abc_sum_l427_427120


namespace mean_proportional_l427_427375

variables {C O A B N : Point}
variables {AB CA CB : Line}
variables {ND NE NF : Line}

-- Definitions of tangents and perpendiculars according to conditions
def is_tangent (l : Line) (α : Circle) : Prop := ∃ P : Point, P ∈ l ∧ P ∈ α
def is_perpendicular (l1 l2 : Line) : Prop := ∃ P : Point, P ∈ l1 ∧ P ∈ l2 ∧ angle l1 l2 = 90

-- Assuming tangents are defined properly
axiom tangent_CA : is_tangent CA (circle O)
axiom tangent_CB : is_tangent CB (circle O)

-- Assuming perpendiculars
axiom perp_ND_AB : is_perpendicular ND AB
axiom perp_NE_CA : is_perpendicular NE CA
axiom perp_NF_CB : is_perpendicular NF CB

-- Problem Statement: Prove ND^2 = NE * NF
theorem mean_proportional (C O A B N : Point) (AB CA CB ND NE NF : Line) :
    is_tangent CA (circle O) ∧
    is_tangent CB (circle O) ∧
    is_perpendicular ND AB ∧
    is_perpendicular NE CA ∧
    is_perpendicular NF CB →
    (length ND) ^ 2 = (length NE) * (length NF) :=
by 
    intro h
    sorry

end mean_proportional_l427_427375


namespace symmetry_center_of_g_l427_427612

noncomputable def f (x : ℝ) : ℝ := sin (2*x) + sqrt 3 * cos (2*x)

def g (x : ℝ) : ℝ := f (x - π/6)

def is_symmetry_center (h : ℝ → ℝ) (c : ℝ × ℝ) := ∀ x : ℝ, h (2 * c.1 - x) + c.2 = h x

theorem symmetry_center_of_g :
  is_symmetry_center g ⟨π / 2, 0⟩ :=
sorry

end symmetry_center_of_g_l427_427612


namespace geometric_sequence_formula_sequence_sum_formula_l427_427874

section GeometricSequence

variable {a : ℕ → ℕ} {b : ℕ → ℚ} {S : ℕ → ℚ}

/-- Given a geometric sequence 'a' with a_1 = 1, 
  and a_2 is the arithmetic mean of a_1 and a_3 - 1, 
  we need to show that the general formula is a_n = 2^(n-1). -/
theorem geometric_sequence_formula (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 2 = (a 1 + a 3 - 1) / 2) :
  ∀ n : ℕ, a n = 2 ^ (n-1) :=
by
  sorry

/-- Given b_n = (1 + n(n+1)a_n) / (n(n+1)) and 
  sum of the first n terms S_n = 2^n - 1/(n+1),
  we need to show that this is indeed the sum. -/
theorem sequence_sum_formula (b : ℕ → ℚ) (S : ℕ → ℚ)
  (h_b : ∀ n : ℕ, b n = (1 + n * (n + 1) * 2^(n-1)) / (n * (n + 1)))
  (h_S : ∀ n : ℕ, S n = ∑ i in List.range n, b i) :
  ∀ n : ℕ, S n = 2^n - 1/(n + 1) :=
by
  sorry

end GeometricSequence

end geometric_sequence_formula_sequence_sum_formula_l427_427874


namespace focus_of_parabola_l427_427956

-- Definition of the condition: the parabola y = -2x^2
def parabola : ℝ → ℝ := λ x, -2 * x^2

-- Statement of the theorem: proving that the focus of this parabola is at (0, -1/8)
theorem focus_of_parabola : ∃ f : ℝ × ℝ, f = (0, -1 / 8) ∧ ∀ x : ℝ, parabola x = -2 * x^2 :=
by
  unfold parabola
  have h_focus : (0, -1 / 8) = (0, -1 / 8) := by rfl
  use (0, -1/8)
  split
  · exact h_focus
  · intros x
    simp only [parabola]
    rfl
  sorry

end focus_of_parabola_l427_427956


namespace find_children_ages_l427_427607

-- Define the conditions as Lean definitions or assumptions
variable {n : ℕ} -- number of children excluding the 10-year-old child
variable (a : ℕ) -- age of the oldest child
variable (d : ℕ) -- common difference in the arithmetic sequence
variable (ages : List ℕ) -- list of ages of all children

-- Conditions
axiom total_age_of_children : ∑ k in ages, k = 50
axiom oldest_child : a = 13
axiom has_ten_year_old : 10 ∈ ages
axiom remaining_form_arith_seq : ∃ (n : ℕ), (List.filter (λ x => x ≠ 10) ages = List.range (n + 1) (λ i => a - i * d))

-- Theorem statement with expected result
theorem find_children_ages
  (total_agcs : ∑ k in ages, k = 50)
  (oldest : a = 13)
  (has10 : 10 ∈ ages)
  (arithmetic_seq : ∃ (n : ℕ), (List.filter (λ x => x ≠ 10) ages = List.range (n + 1) (λ i => a - i * d))) 
: ages = [13, 11, 10, 9, 7] := by
  sorry

end find_children_ages_l427_427607


namespace problem_statement_l427_427382

noncomputable def a : ℝ := 1.27^0.2
noncomputable def b : ℝ := Real.log₀ (0.3) (Real.tan (Real.pi * 46 / 180))
noncomputable def c : ℝ := 2 * Real.sin (Real.pi * 29 / 180)

theorem problem_statement : a > c ∧ c > b := by
  sorry

end problem_statement_l427_427382


namespace determine_teeth_l427_427176

theorem determine_teeth (x V : ℝ) (h1 : V = 63 * x / (x + 10)) (h2 : V = 28 * (x + 10)) :
  x = 20 ∧ (x + 10) = 30 :=
by
  sorry

end determine_teeth_l427_427176


namespace volume_of_solid_is_108_l427_427599

-- Define the conditions
variables (s : ℝ) (sqrt_2 : ℝ) (sqrt_3 : ℝ)
variables (s := 6 * sqrt_2)
variables (sqrt_2 := Real.sqrt 2) (sqrt_3 := Real.sqrt 3)

-- Define the height of the equilateral triangle base
noncomputable def height_triangle_base := s * (sqrt_3 / 2)

-- Define the area of the equilateral triangle base
noncomputable def area_triangle_base := (s * height_triangle_base) / 2

-- Define the height of the solid (given as the length of the connecting edges)
noncomputable def height_solid := s

-- Define the volume of the solid
noncomputable def volume_solid := (1 / 3) * area_triangle_base * height_solid

-- Main theorem to prove
theorem volume_of_solid_is_108 : volume_solid s sqrt_2 sqrt_3 = 108 :=
by
  sorry

end volume_of_solid_is_108_l427_427599


namespace arith_seq_a1_a7_sum_l427_427461

variable (a : ℕ → ℝ) (d : ℝ)

-- Conditions
def arithmetic_sequence : Prop :=
  ∀ n, a (n + 1) = a n + d

def condition_sum : Prop :=
  a 3 + a 4 + a 5 = 12

-- Equivalent proof problem statement
theorem arith_seq_a1_a7_sum :
  arithmetic_sequence a d →
  condition_sum a →
  a 1 + a 7 = 8 :=
by
  sorry

end arith_seq_a1_a7_sum_l427_427461


namespace cylinder_ratio_max_volume_l427_427261

theorem cylinder_ratio_max_volume :
  ∀ (r h : ℝ), 2 * Real.pi * r^2 + 2 * Real.pi * r * h = 6 * Real.pi → 
  (π * r^2 * h) → 
    h / r = Real.sqrt (3 / 2) := 
by
  intros r h
  assume surface_area_eqvol
  sorry

end cylinder_ratio_max_volume_l427_427261


namespace total_area_to_paint_l427_427307

-- Define the conditions
def length : ℕ := 15
def width : ℕ := 12
def height : ℕ := 9
def rooms : ℕ := 4
def window_and_door_area : ℕ := 80

-- Define the area of one bedroom's walls
def wall_area_one_bedroom : ℕ :=
  2 * (length * height) + 2 * (width * height)

-- Define the area to be painted in one bedroom
def paintable_area_one_bedroom : ℕ :=
  wall_area_one_bedroom - window_and_door_area

-- Define the total area to be painted
def total_paintable_area : ℕ :=
  rooms * paintable_area_one_bedroom

-- The theorem stating the total area to be painted
theorem total_area_to_paint : total_paintable_area = 1624 :=
  by
    sorry

end total_area_to_paint_l427_427307


namespace find_children_tickets_l427_427609

variable (A C S : ℝ)

theorem find_children_tickets 
  (h1 : A + C + S = 600)
  (h2 : 6 * A + 4.5 * C + 5 * S = 3250) :
  C = (350 - S) / 1.5 := 
sorry

end find_children_tickets_l427_427609


namespace ball_heights_equal_at_t_l427_427584

-- Define the quadratic height function
def height (a : ℝ) (h : ℝ) (t : ℝ) : ℝ := a * (t - 1.2) ^ 2 + h

-- Prove that the heights of the two balls are the same at t = 2.2 seconds after the first ball is thrown
theorem ball_heights_equal_at_t : ∀ (a h : ℝ), a ≠ 0 → 
  ∃ t : ℝ, (height a h t = height a h (t - 2)) ∧ t = 2.2 :=
by
  intros a h hka
  use 2.2
  split
  · unfold height
    have eq1 : a * (2.2 - 1.2) ^ 2 + h = a * (2.2 - 3.2) ^ 2 + h
      by sorry
    exact eq1
  · rfl

end ball_heights_equal_at_t_l427_427584


namespace number_of_ways_to_place_lids_l427_427121

theorem number_of_ways_to_place_lids :
  let cups := {1, 2, 3, 4, 5, 6}
  let lids := {1, 2, 3, 4, 5, 6}
  ∃! (arrangement : list ℕ), (arrangement.length = 6)
    ∧ (list.countp (λ (i : ℕ), arrangement[i-1] = i) arrangement = 2)
    ∧ (∀ i ∈ cups, arrangement[i-1] ∈ lids)
    → (list.countp
        (λ (i : ℕ), let lid_location := finRange 6
                    ∃! (j : ℕ) (h2 : j ∈ lid_location), arrangement[j] = i
                    ∧ j ≠ i)
        lids
      = 4)
    → ∃! (ways : ℕ), ways = 135
:= sorry

end number_of_ways_to_place_lids_l427_427121


namespace supplement_of_complement_of_35_degree_angle_l427_427192

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_of_35_degree_angle : 
  supplement (complement 35) = 125 := 
by sorry

end supplement_of_complement_of_35_degree_angle_l427_427192


namespace g_12_30_equals_20_l427_427726

theorem g_12_30_equals_20 (g : ℕ → ℕ → ℕ) 
  (h1 : ∀ x, g(x, x) = x)
  (h2 : ∀ x y, g(x, y) = g(y, x))
  (h3 : ∀ x y, (x + y) * g(x, y) = y^2 * g(x, x + y)) :
  g 12 30 = 20 := 
sorry

end g_12_30_equals_20_l427_427726


namespace time_difference_is_16_point_5_l427_427722

noncomputable def time_difference : ℝ :=
  let danny_to_steve : ℝ := 33
  let steve_to_danny := 2 * danny_to_steve -- Steve takes twice the time as Danny
  let emma_to_houses : ℝ := 40
  let danny_halfway := danny_to_steve / 2 -- Halfway point for Danny
  let steve_halfway := steve_to_danny / 2 -- Halfway point for Steve
  let emma_halfway := emma_to_houses / 2 -- Halfway point for Emma
  -- Additional times to the halfway point
  let steve_additional := steve_halfway - danny_halfway
  let emma_additional := emma_halfway - danny_halfway
  -- The final result is the maximum of these times
  max steve_additional emma_additional

theorem time_difference_is_16_point_5 : time_difference = 16.5 :=
  by
  sorry

end time_difference_is_16_point_5_l427_427722


namespace area_of_second_inscribed_square_l427_427036

-- Define the conditions
noncomputable def side_length_of_first_square : ℝ := real.sqrt 484
noncomputable def area_of_first_square : ℝ := 484
noncomputable def hypotenuse_length : ℝ := 2 * side_length_of_first_square

-- Given conditions
def isosceles_right_triangle_with_inscribed_square (area_of_first_square = 484) : Prop :=
  ∃ (side_length_of_first_square hypotenuse_length : ℝ),
  side_length_of_first_square = real.sqrt 484 ∧ hypotenuse_length = 2 * side_length_of_first_square

-- The statement we need to prove
theorem area_of_second_inscribed_square :
  isosceles_right_triangle_with_inscribed_square 484 →
  ∃ (side_length_of_second_square : ℝ),
  side_length_of_second_square = 44 / 3 ∧
  (side_length_of_second_square ^ 2 = 1936 / 9) :=
sorry

end area_of_second_inscribed_square_l427_427036


namespace simplify_expression_l427_427554

-- We define the given expressions and state the theorem.
variable (x : ℝ)

theorem simplify_expression : (3 * x - 4) * (x + 8) - (x + 6) * (3 * x - 2) = 4 * x - 20 := by
  -- Proof goes here
  sorry

end simplify_expression_l427_427554


namespace minimum_distinct_sums_seven_l427_427487

theorem minimum_distinct_sums_seven (a_1 a_2 a_3 a_4 a_5 : ℝ) (h_distinct: a_1 ≠ a_2 ∧ a_1 ≠ a_3 ∧ a_1 ≠ a_4 ∧ a_1 ≠ a_5 ∧ a_2 ≠ a_3 ∧ a_2 ≠ a_4 ∧ a_2 ≠ a_5 ∧ a_3 ≠ a_4 ∧ a_3 ≠ a_5 ∧ a_4 ≠ a_5) :
  7 ≤ (finset.card (({a_1, a_2, a_3, a_4, a_5}.product {a_1, a_2, a_3, a_4, a_5}).filter (λ x, x.1 ≠ x.2).image (λ x, x.1 + x.2))) :=
by sorry

end minimum_distinct_sums_seven_l427_427487


namespace quadrilateral_AFHD_area_is_24_dm2_l427_427613

def area (x : Type) [geometry x] (rx : segment x) : ℝ := sorry

variables (A B C D E F H I : Type) [geometry A] [geometry B] [geometry C] [geometry D] [geometry E] [geometry F] [geometry H] [geometry I]
variables [parallel AD AB] [parallel DE AB]
variables (area_CDH : area C D H = 8)
variables (area_CHI : area C H I = 8)
variables (area_CIE : area C I E = 8)
variables (area_FIH : area F I H = 8)

theorem quadrilateral_AFHD_area_is_24_dm2 : area A F H D = 24 := by sorry

end quadrilateral_AFHD_area_is_24_dm2_l427_427613


namespace magnitude_of_vector_addition_l427_427828

def vector_a : ℝ × ℝ := (2, -1)
def vector_b : ℝ × ℝ := (0, 1)

def vector_addition : ℝ × ℝ := 
(vector_a.1 + 2 * vector_b.1, vector_a.2 + 2 * vector_b.2)

def magnitude (v : ℝ × ℝ) : ℝ := 
Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem magnitude_of_vector_addition : magnitude vector_addition = Real.sqrt 5 := by
  sorry

end magnitude_of_vector_addition_l427_427828


namespace union_M_N_l427_427824

open Set

variable (x y : ℝ)

def M : Set ℝ := { x | -1 < x ∧ x < 2 }
def N : Set ℝ := { x | ∃ y, y = real.sqrt (x - 1) }

theorem union_M_N : (M ∪ N) = { x | x > -1 } :=
sorry

end union_M_N_l427_427824


namespace intersection_is__l427_427823

def M := {-1, 0, 1, 2, 3}
def N := {x : ℝ | x^2 - 2 * x > 0}

theorem intersection_is_{-1_3} : (M ∩ N : Set ℝ) = {-1, 3} := 
by {
  sorry
}

end intersection_is__l427_427823


namespace opposite_face_of_3_l427_427091

-- Define the faces of the cube and their properties
def cube_faces := {1, 2, 3, 4, 5, 6}

-- Define a condition for the sum of lateral faces in two rolls
def sum_lateral_faces (a b c d : ℕ) : Prop :=
  a + b + c + d

-- Sum of all numbers on a standard six-sided die
def total_sum : ℕ := 1 + 2 + 3 + 4 + 5 + 6

-- Problem statement
theorem opposite_face_of_3 
  (a1 a2 a3 a4 a5 a6 : ℕ)
  (h1_neq : ∀ {x y}, x ≠ y → (a1, a2, a3, a4, a5, a6).Nodup)
  (h2 : {a1, a2, a3, a4, a5, a6} = cube_faces)
  (h3 : sum_lateral_faces a1 a2 a3 a4 = 12)
  (h4 : sum_lateral_faces a1 a2 a3 a4 = 15)
  (h5 : ∃ i j, i + j = total_sum - sum_lateral_faces a1 a2 a3 a4)
  (h6 : i ≠ j ∧ i ∈ cube_faces ∧ j ∈ cube_faces) :
    (∑ᶠ i in cube_faces, 7 - i) = 6 := 
sorry

end opposite_face_of_3_l427_427091


namespace sum_of_abc_l427_427152

theorem sum_of_abc (a b c : ℕ) (h1 : (300 / 75 : ℝ) = 4) (h2 : (sqrt 4 : ℝ) = 2) 
    (h3 : 2 = a * sqrt b / c) (h4 : b = 1) (h5 : c = 1) : a + b + c = 4 :=
by
  have a_val : a = 2 := by
    sorry
  rw [a_val, h4, h5]
  exact rfl

end sum_of_abc_l427_427152


namespace probability_of_rolling_at_least_9_at_least_seven_times_l427_427667

theorem probability_of_rolling_at_least_9_at_least_seven_times :
  let p := (2:ℚ) / 10 in
  let q := (8 * p^7 * (4:ℚ)/5 + p^8) in
  q = 33 / 78125 := 
by
  sorry

end probability_of_rolling_at_least_9_at_least_seven_times_l427_427667


namespace discount_on_soap_l427_427671

theorem discount_on_soap :
  (let chlorine_price := 10
   let chlorine_discount := 0.20 * chlorine_price
   let discounted_chlorine_price := chlorine_price - chlorine_discount

   let soap_price := 16

   let total_savings := 26

   let chlorine_savings := 3 * chlorine_price - 3 * discounted_chlorine_price
   let soap_savings := total_savings - chlorine_savings

   let discount_per_soap := soap_savings / 5
   let discount_percentage_per_soap := (discount_per_soap / soap_price) * 100
   discount_percentage_per_soap = 25) := sorry

end discount_on_soap_l427_427671


namespace smallest_b_for_factorable_polynomial_l427_427362

theorem smallest_b_for_factorable_polynomial :
  ∃ (b : ℕ), b > 0 ∧ (∃ (p q : ℤ), x^2 + b * x + 1176 = (x + p) * (x + q) ∧ p * q = 1176 ∧ p + q = b) ∧ 
  (∀ (b' : ℕ), b' > 0 → (∃ (p' q' : ℤ), x^2 + b' * x + 1176 = (x + p') * (x + q') ∧ p' * q' = 1176 ∧ p' + q' = b') → b ≤ b') :=
sorry

end smallest_b_for_factorable_polynomial_l427_427362


namespace complex_in_third_quadrant_l427_427403

def complex_plane_quadrant : ℂ → ℕ
| z => if z.re > 0 ∧ z.im > 0 then 1
       else if z.re < 0 ∧ z.im > 0 then 2
       else if z.re < 0 ∧ z.im < 0 then 3
       else if z.re > 0 ∧ z.im < 0 then 4
       else 0 -- edge case not considered in this problem

theorem complex_in_third_quadrant :
  let z : ℂ := (2 + 3 * complex.I) * (1 - 2 * complex.I) / complex.I
  in complex_plane_quadrant z = 3 :=
by
  let z : ℂ := (2 + 3 * complex.I) * (1 - 2 * complex.I) / complex.I
  sorry

end complex_in_third_quadrant_l427_427403


namespace remainder_when_13_plus_y_divided_by_31_l427_427525

theorem remainder_when_13_plus_y_divided_by_31
  (y : ℕ)
  (hy : 7 * y % 31 = 1) :
  (13 + y) % 31 = 22 :=
sorry

end remainder_when_13_plus_y_divided_by_31_l427_427525


namespace max_regions_divided_l427_427015

theorem max_regions_divided (m n : ℕ) (hm : m ≠ 0) (hn : n ≠ 0) : 
  ∃ R, R = m * n + 2 * m + 2 * n - 1 :=
by
  use m * n + 2 * m + 2 * n - 1
  sorry

end max_regions_divided_l427_427015


namespace lines_parallel_a_eq_neg_one_l427_427851

theorem lines_parallel_a_eq_neg_one (a : ℝ) 
  (h : ∀ x y : ℝ, (a * x + 2 * y + 6 = 0 ∧ x + (a - 1) * y + 3 = 0) → a = -1) :
  ∃ a, a = -1 :=
by
  use a
  assumption

end lines_parallel_a_eq_neg_one_l427_427851


namespace perpendicular_medians_cotangents_inequality_l427_427939

noncomputable def cotangent (x : ℝ) := 1 / Real.tan x

theorem perpendicular_medians_cotangents_inequality 
  {A B C : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (med_AD : MetricSpace A) (med_BE : MetricSpace B) (med_CF : MetricSpace C)
  (perpendicular : BE ∩ CF = ⊥) :
  cotangent (angle B A C) + cotangent (angle C A B) ≥ 2 / 3 :=
by
  sorry

end perpendicular_medians_cotangents_inequality_l427_427939


namespace perfect_square_n_l427_427772

def is_perfect_square (x : ℕ) : Prop :=
  ∃ y : ℕ, y * y = x

theorem perfect_square_n (n : ℕ) : 
  is_perfect_square (nat.factorial 1 * nat.factorial 2 * nat.factorial 3 * 
    ((finset.range (2 * n + 1).succ).filter nat.even).prod (!.) * 
    nat.factorial (2 * n)) ∧ (n = 4 * k * (k + 1) ∨ n = 2 * k ^ 2 - 1) :=
by
  sorry

end perfect_square_n_l427_427772


namespace rectangle_sides_l427_427677

theorem rectangle_sides (n : ℕ) (hpos : n > 0)
  (h1 : (∃ (a : ℕ), (a^2 * n = n)))
  (h2 : (∃ (b : ℕ), (b^2 * (n + 98) = n))) :
  (∃ (l w : ℕ), l * w = n ∧ 
  ((n = 126 ∧ (l = 3 ∧ w = 42 ∨ l = 6 ∧ w = 21)) ∨
  (n = 1152 ∧ l = 24 ∧ w = 48))) :=
sorry

end rectangle_sides_l427_427677


namespace max_minute_hands_l427_427231

theorem max_minute_hands (m n : ℕ) (h : m * n = 27) : m + n ≤ 28 :=
sorry

end max_minute_hands_l427_427231


namespace part1_solution_set_part2_min_value_l427_427001

-- Part 1
noncomputable def f (x : ℝ) : ℝ := 2 * |x + 1| + |3 * x|

theorem part1_solution_set :
  {x : ℝ | f x ≥ 3 * |x| + 1} = {x : ℝ | x ≥ -1/2} ∪ {x : ℝ | x ≤ -3/2} :=
by
  sorry

-- Part 2
noncomputable def f_min (x a b : ℝ) : ℝ := 2 * |x + a| + |3 * x - b|

theorem part2_min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : ∃ x, f_min x a b = 2) :
  3 * a + b = 3 :=
by
  sorry

end part1_solution_set_part2_min_value_l427_427001


namespace select_numbers_l427_427111

theorem select_numbers (a1 a2 a3 : ℕ) (h1 : a1 ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
                       (h2 : a2 ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
                       (h3 : a3 ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
                       (h4 : a1 ≠ a2) (h5 : a1 ≠ a3) (h6 : a2 ≠ a3)
                       (h7 : a2 - a1 ≥ 2) (h8 : a3 - a2 ≥ 3) :
  ∃ n : ℕ, n = 35 :=
sorry

end select_numbers_l427_427111


namespace value_of_r_l427_427906

theorem value_of_r (n : ℕ) (h : n = 3) : 
  let s := 2^n - 1
  let r := 4^s - s
  r = 16377 := by
  let s := 2^3 - 1
  let r := 4^s - s
  sorry

end value_of_r_l427_427906


namespace Flynn_tv_minutes_weekday_l427_427752

theorem Flynn_tv_minutes_weekday :
  ∀ (tv_hours_per_weekend : ℕ)
    (tv_hours_per_year : ℕ)
    (weeks_per_year : ℕ) 
    (weekdays_per_week : ℕ),
  tv_hours_per_weekend = 2 →
  tv_hours_per_year = 234 →
  weeks_per_year = 52 →
  weekdays_per_week = 5 →
  (tv_hours_per_year - (tv_hours_per_weekend * weeks_per_year)) / (weekdays_per_week * weeks_per_year) * 60
  = 30 :=
by
  intros tv_hours_per_weekend tv_hours_per_year weeks_per_year weekdays_per_week
        h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end Flynn_tv_minutes_weekday_l427_427752


namespace part1_monotonic_intervals_part2_minimum_value_part3_inequality_l427_427903

-- Definition of the function
def f (m : ℝ) (x : ℝ) : ℝ := Real.log x - m * x

-- Part (1): Monotonic intervals
theorem part1_monotonic_intervals (m : ℝ) :
    (m ≤ 0 → (∀ x > 0, f m x < f m (x + 1))) ∧ 
    (m > 0 → (∀ x > 0, x < 1 / m → f m x < f m (x + 1)) ∧ (∀ x > 1 / m, f m x > f m (x + 1))) :=
sorry

-- Part (2): Minimum value of a + 2b when m = e
theorem part2_minimum_value :
  ∃ (x0 : ℝ), (a = 1/x0 - Real.e) ∧ (b = Real.log x0 - Real.e) ∧ (a + 2 * b = -Real.e - 2 * Real.log 2) :=
sorry

-- Part (3): Inequality involving roots
theorem part3_inequality (m n x1 x2 : ℝ) (h1 : x1 < x2) (h2 : f m x1 = (2 - m) * x1 + n) (h3 : f m x2 = (2 - m) * x2 + n) :
    2 * x1 + x2 > Real.e / 2 :=
sorry

end part1_monotonic_intervals_part2_minimum_value_part3_inequality_l427_427903


namespace largest_n_factorable_l427_427336

theorem largest_n_factorable :
  ∃ n, (∀ A B : ℤ, 6x^2 + n • x + 72 = (6 • x + A) * (x + B)) ∧ 
    (∀ x', 6x' + A = 0 ∨ x' + B = 0) ∧ 
    n = 433 := 
sorry

end largest_n_factorable_l427_427336


namespace sum_factorials_mod_20_l427_427031

theorem sum_factorials_mod_20 :
  (1! + 2! + 3! + 4! + 5! + 6!) % 20 = 13 := 
sorry

end sum_factorials_mod_20_l427_427031


namespace largest_n_factorable_l427_427334

theorem largest_n_factorable :
  ∃ n, (∀ A B : ℤ, 6x^2 + n • x + 72 = (6 • x + A) * (x + B)) ∧ 
    (∀ x', 6x' + A = 0 ∨ x' + B = 0) ∧ 
    n = 433 := 
sorry

end largest_n_factorable_l427_427334


namespace problem1_problem2_l427_427116

-- Define the first problem as a proof statement in Lean
theorem problem1 (x : ℝ) : (x - 2) ^ 2 = 25 → (x = 7 ∨ x = -3) := sorry

-- Define the second problem as a proof statement in Lean
theorem problem2 (x : ℝ) : (x - 5) ^ 2 = 2 * (5 - x) → (x = 5 ∨ x = 3) := sorry

end problem1_problem2_l427_427116


namespace increase_segment_by_n_l427_427235

variable (A B : Point)
variable (n : ℕ)

theorem increase_segment_by_n (hAB : distance A B > 0) :
  ∃ M : Point, distance A M = n * distance A B := sorry

end increase_segment_by_n_l427_427235


namespace greatest_num_consecutive_integers_sum_eq_36_l427_427621

theorem greatest_num_consecutive_integers_sum_eq_36 :
    ∃ a : ℤ, ∃ N : ℕ, N > 0 ∧ (N = 9) ∧ (N * (2 * a + N - 1) = 72) :=
sorry

end greatest_num_consecutive_integers_sum_eq_36_l427_427621


namespace valid_bases_for_625_l427_427370

theorem valid_bases_for_625 (b : ℕ) : (b^3 ≤ 625 ∧ 625 < b^4) → ((625 % b) % 2 = 1) ↔ (b = 6 ∨ b = 7 ∨ b = 8) :=
by
  sorry

end valid_bases_for_625_l427_427370


namespace fair_game_x_value_l427_427448

theorem fair_game_x_value (x : ℕ) (h : x + 2 * x + 2 * x = 15) : x = 3 := 
by sorry

end fair_game_x_value_l427_427448


namespace domain_f_l427_427327

def f (x : ℝ) : ℝ := sqrt (x - 1) + sqrt (8 - x)

theorem domain_f :
  {x : ℝ | 1 ≤ x ∧ x ≤ 8} = {x : ℝ | ∃ y : ℝ, f y = x } :=
sorry

end domain_f_l427_427327


namespace range_of_m_l427_427446

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (2 * x + m) / (x - 2) + (x - 1) / (2 - x) = 3) ↔ (m > -7 ∧ m ≠ -3) :=
by
  sorry

end range_of_m_l427_427446


namespace geometric_seq_root_l427_427048

theorem geometric_seq_root (a : ℕ → ℝ) (h_geometric : ∀ n, a (n + 1) / a n = a 1 / a 0) 
  (h_roots : ∀ x, 3 * x^2 - 11 * x + 9 = 0 → (x = a 3 ∨ x = a 9)) :
  a 6 = ± sqrt 3 :=
by
  sorry

end geometric_seq_root_l427_427048


namespace find_primes_pqr_eq_5_sum_l427_427738

theorem find_primes_pqr_eq_5_sum (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) :
  p * q * r = 5 * (p + q + r) → (p = 2 ∧ q = 5 ∧ r = 7) ∨ (p = 2 ∧ q = 7 ∧ r = 5) ∨
                                         (p = 5 ∧ q = 2 ∧ r = 7) ∨ (p = 5 ∧ q = 7 ∧ r = 2) ∨
                                         (p = 7 ∧ q = 2 ∧ r = 5) ∨ (p = 7 ∧ q = 5 ∧ r = 2) :=
by
  sorry

end find_primes_pqr_eq_5_sum_l427_427738


namespace roots_are_complex_conjugates_l427_427905

theorem roots_are_complex_conjugates (p q : ℝ)
    (h : ∀ z : ℂ, z^2 + (6 : ℝ) + (p : ℂ)*complex.I) z + (13 : ℝ) + (q : ℂ)*complex.I = complex.conj z^2 + (6 : ℝ) + (p : ℂ)*complex.I) z + (13 : ℝ) + (q : ℂ)*complex.I):
    p = 0 ∧ q = 0 :=
by
  sorry

end roots_are_complex_conjugates_l427_427905


namespace degree_meas_supp_compl_35_l427_427200

noncomputable def degree_meas_supplement_complement (θ : ℝ) : ℝ :=
  180 - (90 - θ)

theorem degree_meas_supp_compl_35 : degree_meas_supplement_complement 35 = 125 :=
by
  unfold degree_meas_supplement_complement
  norm_num
  sorry

end degree_meas_supp_compl_35_l427_427200


namespace complex_cube_root_identity_l427_427079

theorem complex_cube_root_identity (a b c : ℂ) (ω : ℂ)
  (h1 : ω^3 = 1)
  (h2 : 1 + ω + ω^2 = 0) :
  (a + b * ω + c * ω^2) * (a + b * ω^2 + c * ω) = a^2 + b^2 + c^2 - ab - ac - bc :=
by
  sorry

end complex_cube_root_identity_l427_427079


namespace largest_two_digit_number_after_sequence_l427_427657

-- Definitions for the conditions
def button_A (x : ℕ) : ℕ := 2 * x + 1
def button_B (x : ℕ) : ℕ := 3 * x - 1

-- The initial value on the display
def initial_value : ℕ := 5

-- Statement of the theorem
theorem largest_two_digit_number_after_sequence : 
  ∃ s : list (ℕ → ℕ), s.foldl (λ x f, f x) initial_value = 95 :=
sorry

end largest_two_digit_number_after_sequence_l427_427657


namespace exists_three_sum_geq_five_l427_427529

theorem exists_three_sum_geq_five 
  (x : Fin 9 → ℝ) 
  (hx_nonneg : ∀ i, 0 ≤ x i) 
  (hx_sum_sq : (∑ i, (x i) ^ 2) ≥ 25) : 
  ∃ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (x a + x b + x c) ≥ 5 :=
by
  sorry

end exists_three_sum_geq_five_l427_427529


namespace cos_square_sum_eq_one_l427_427938

theorem cos_square_sum_eq_one (α β γ : Real) (h : α + β + γ = π) :
  (cos α)^2 + (cos β)^2 + (cos γ)^2 + 2 * (cos α) * (cos β) * (cos γ) = 1 :=
  sorry

end cos_square_sum_eq_one_l427_427938


namespace largest_value_of_n_l427_427353

theorem largest_value_of_n (A B n : ℤ) (h1 : A * B = 72) (h2 : n = 6 * B + A) : n = 433 :=
sorry

end largest_value_of_n_l427_427353


namespace sum_i2_binom_eq_l427_427296

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the sum
def sum_i2_binom (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), i^2 * binom n i

-- Theorem statement
theorem sum_i2_binom_eq (n : ℕ) : sum_i2_binom n = 2^(n-1) * (n+2) := 
  by
    sorry

end sum_i2_binom_eq_l427_427296


namespace largest_n_for_factored_quad_l427_427345

theorem largest_n_for_factored_quad (n : ℤ) (b d : ℤ) 
  (h1 : 6 * d + b = n) (h2 : b * d = 72) 
  (factorable : ∃ x : ℤ, (6 * x + b) * (x + d) = 6 * x ^ 2 + n * x + 72) : 
  n ≤ 433 :=
sorry

end largest_n_for_factored_quad_l427_427345


namespace mean_temperature_is_zero_l427_427952

def temperatures : list ℤ := [-3, -1, -6, 0, 4, 6]

theorem mean_temperature_is_zero (temps : list ℤ) (h : temps = temperatures) : 
  (temps.sum : ℚ) / temps.length = 0 := 
by {
  rw h,
  norm_num,
}

end mean_temperature_is_zero_l427_427952


namespace min_dot_product_l427_427909

theorem min_dot_product (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (ha_curve : a * (1 / a) = 1) (hb_curve : b * (1 / b) = 1) :
  let OA := Real.sqrt (a^2 + (1 / a)^2),
      OB := b * (1 / b),
      m := (1, OA),
      B := (b, 1 / b),
      dot_product := (1 * b) + OA * (1 / b) in  
  (∀ a b, dot_product ≥ 2 * Real.root 4 2) :=
by
  sorry

end min_dot_product_l427_427909


namespace order_of_a_b_c_l427_427778

noncomputable def a : ℝ := (Real.log 5) / 5
noncomputable def b : ℝ := 1 / Real.exp 1
noncomputable def c : ℝ := (Real.log 4) / 4

theorem order_of_a_b_c : a < c ∧ c < b := by
  sorry

end order_of_a_b_c_l427_427778


namespace degree_measure_of_supplement_of_complement_of_35_degree_angle_l427_427196

def complement (α : ℝ) : ℝ := 90 - α
def supplement (β : ℝ) : ℝ := 180 - β

theorem degree_measure_of_supplement_of_complement_of_35_degree_angle : 
  supplement (complement 35) = 125 :=
by
  sorry

end degree_measure_of_supplement_of_complement_of_35_degree_angle_l427_427196


namespace arithmetic_sequence_sum_and_mean_l427_427712

theorem arithmetic_sequence_sum_and_mean :
  let a1 := 1
  let d := 2
  let an := 21
  let n := 11
  let S := (n / 2) * (a1 + an)
  S = 121 ∧ (S / n) = 11 :=
by
  let a1 := 1
  let d := 2
  let an := 21
  let n := 11
  let S := (n / 2) * (a1 + an)
  have h1 : S = 121 := sorry
  have h2 : (S / n) = 11 := by
    rw [h1]
    exact sorry
  exact ⟨h1, h2⟩

end arithmetic_sequence_sum_and_mean_l427_427712


namespace largest_fraction_l427_427004
-- Import the necessary library

-- State the problem in Lean 4
theorem largest_fraction 
  (a b c d e : ℝ) 
  (h1 : 0 < a) 
  (h2 : a < b) 
  (h3 : b < c) 
  (h4 : c < d) 
  (h5 : d < e) :
  max (max (max (max (a + c) / (b + d) (b + e) / (c + d)) (c + d) / (a + e)) (d + e) / (a + b)) (e + a) / (b + c) = (d + e) / (a + b) := 
sorry -- Proof is omitted

end largest_fraction_l427_427004


namespace validate_expression_l427_427933

-- Define the expression components
def a := 100
def b := 6
def c := 7
def d := 52
def e := 8
def f := 9

-- Define the expression using the given numbers and operations
def expression := (a - b) * c - d + e + f

-- The theorem statement asserting that the expression evaluates to 623
theorem validate_expression : expression = 623 := 
by
  -- Proof would go here
  sorry

end validate_expression_l427_427933


namespace age_problem_l427_427639

theorem age_problem (A B C D E : ℕ)
  (h1 : A = B + 2)
  (h2 : B = 2 * C)
  (h3 : D = C / 2)
  (h4 : E = D - 3)
  (h5 : A + B + C + D + E = 52) : B = 16 :=
by
  sorry

end age_problem_l427_427639


namespace mmobile_additional_line_cost_l427_427536

noncomputable def cost_tmobile (n : ℕ) : ℕ :=
  if n ≤ 2 then 50 else 50 + (n - 2) * 16

noncomputable def cost_mmobile (x : ℕ) (n : ℕ) : ℕ :=
  if n ≤ 2 then 45 else 45 + (n - 2) * x

theorem mmobile_additional_line_cost
  (x : ℕ)
  (ht : cost_tmobile 5 = 98)
  (hm : cost_tmobile 5 - cost_mmobile x 5 = 11) :
  x = 14 :=
by
  sorry

end mmobile_additional_line_cost_l427_427536


namespace simplify_expr1_correct_l427_427297

def simplify_expr1 (x : ℝ) : ℝ := x * x^5 + (-2 * x^3)^2 - 3 * x^8 / x^2

theorem simplify_expr1_correct (x : ℝ) : simplify_expr1(x) = 2 * x^6 :=
by 
  sorry

end simplify_expr1_correct_l427_427297


namespace max_slope_OM_l427_427908

noncomputable def parabola_focus (p : ℝ) (hp : p > 0) : ℝ × ℝ := (p / 2, 0)

noncomputable def parabola_point (p y0 : ℝ) (hp : p > 0) (hy0 : y0 > 0) : ℝ × ℝ :=
  (y0^2 / (2 * p), y0)

noncomputable def M_point (p y0 : ℝ) (hp : p > 0) (hy0 : y0 > 0) : ℝ × ℝ :=
  (y0^2 / (6 * p) + p / 3, y0^2 / 3)

noncomputable def slope_OM (p y0 : ℝ) (hp : p > 0) (hy0 : y0 > 0) : ℝ :=
  let M := M_point p y0 hp hy0
  in M.2 / M.1

theorem max_slope_OM (p : ℝ) (hp : p > 0) :
  ∃ (y0 : ℝ) (hy0 : y0 > 0), slope_OM p y0 hp hy0 = (Real.sqrt 2 / 2) :=
sorry

end max_slope_OM_l427_427908


namespace find_sum_of_squares_l427_427495

variable {R : Type*} [Field R]

-- Define the matrix B
def B (e f g h i j : R) : Matrix (Fin 3) (Fin 3) R :=
  ![![e, f, g],
    ![f, h, i],
    ![g, i, j]]

-- B is symmetric
lemma B_symmetric (e f g h i j : R) : (B e f g h i j)ᵀ = B e f g h i j :=
  by simp [B, Matrix.transpose]

-- B is orthogonal
lemma B_orthogonal (e f g h i j : R) (h₁ : (B e f g h i j) ⬝ (B e f g h i j) = 1) : 
  (Matrix.mul (B e f g h i j) (B e f g h i j)) = 1 := 
  by assumption

-- The main theorem
theorem find_sum_of_squares (e f g h i j : ℝ) 
  (h₁ : (B e f g h i j) ⬝ (B e f g h i j) = 1) :
  e^2 + f^2 + g^2 + h^2 + i^2 + j^2 = 3 :=
  by 
    -- Expand Matrix multiplication and use the given conditions
    sorry

end find_sum_of_squares_l427_427495


namespace sum_of_distinct_roots_l427_427517

theorem sum_of_distinct_roots 
  (p q r s : ℝ)
  (h1 : p ≠ q)
  (h2 : p ≠ r)
  (h3 : p ≠ s)
  (h4 : q ≠ r)
  (h5 : q ≠ s)
  (h6 : r ≠ s)
  (h_roots1 : (x : ℝ) -> x^2 - 12*p*x - 13*q = 0 -> x = r ∨ x = s)
  (h_roots2 : (x : ℝ) -> x^2 - 12*r*x - 13*s = 0 -> x = p ∨ x = q) : 
  p + q + r + s = 1716 := 
by 
  sorry

end sum_of_distinct_roots_l427_427517


namespace circles_in_square_l427_427051

theorem circles_in_square (side_length : ℝ) (r : ℝ) (seg_len : ℝ) (n : ℕ) 
  (square : set (ℝ × ℝ)) (circles : finset (set (ℝ × ℝ))) :
  side_length = 100 →
  r = 1 →
  seg_len = 10 →
  square = {p : ℝ × ℝ | abs p.1 ≤ side_length / 2 ∧ abs p.2 ≤ side_length / 2} →
  (∀ l, (abs l = seg_len) → ∃ c ∈ circles, (c ∩ {p : ℝ × ℝ | sqrt ((p.1 - fst l)^2 + (p.2 - snd l)^2) = abs seg_len}) ≠ ∅) →
  n = circles.card →
  n ≥ 400 := sorry

end circles_in_square_l427_427051


namespace blocks_from_gallery_to_work_l427_427290

theorem blocks_from_gallery_to_work (b_store b_gallery b_already_walked b_more_to_work total_blocks blocks_to_work_from_gallery : ℕ) 
  (h1 : b_store = 11)
  (h2 : b_gallery = 6)
  (h3 : b_already_walked = 5)
  (h4 : b_more_to_work = 20)
  (h5 : total_blocks = b_store + b_gallery + b_more_to_work)
  (h6 : blocks_to_work_from_gallery = total_blocks - b_already_walked - b_store - b_gallery) :
  blocks_to_work_from_gallery = 15 :=
by
  sorry

end blocks_from_gallery_to_work_l427_427290


namespace sea_level_information_acquisition_l427_427568

-- Given options as definitions
def remote_sensing : Type := sorry
def gps : Type := sorry
def gis : Type := sorry
def digital_earth : Type := sorry

-- Problem statement in Lean 4
theorem sea_level_information_acquisition :
  remote_sensing = "A" ∧ gps = "B" ∧ gis = "C" ∧ digital_earth = "D" →
  remote_sensing = "A" := sorry

end sea_level_information_acquisition_l427_427568


namespace range_of_m_l427_427845

-- Define the quadratic function
def quadratic_function (x m : ℝ) : ℝ := (x - m) ^ 2 - 1

-- State the main theorem
theorem range_of_m (m : ℝ) :
  (∀ x ≤ 3, quadratic_function x m ≥ quadratic_function (x + 1) m) ↔ m ≥ 3 :=
by
  sorry

end range_of_m_l427_427845


namespace intersection_point_on_line_bc_l427_427916

theorem intersection_point_on_line_bc (A B C D E F X : Point) 
  (h₁ : is_triangle A B C)
  (h₂ : is_foot_of_altitude D A B C)
  (h₃ : is_foot_of_altitude E B C A)
  (h₄ : is_foot_of_altitude F C A B)
  (Γ : Circle)
  (Γ1 : Circle)
  (Γ2 : Circle)
  (h₅ : circumcircle Γ A E F)
  (h₆ : circletangent_to Γ1 Γ E ∧ passes_through Γ1 D)
  (h₇ : circletangent_to Γ2 Γ F ∧ passes_through Γ2 D)
  (h₈ : second_intersection Γ1 Γ2 X) :
  lies_on X (line_from_points B C) :=
sorry

end intersection_point_on_line_bc_l427_427916


namespace kevin_total_distance_l427_427058

def v1 : ℝ := 10
def t1 : ℝ := 0.5
def v2 : ℝ := 20
def t2 : ℝ := 0.5
def v3 : ℝ := 8
def t3 : ℝ := 0.25

theorem kevin_total_distance : v1 * t1 + v2 * t2 + v3 * t3 = 17 := by
  sorry

end kevin_total_distance_l427_427058


namespace symmetric_point_xOy_plane_l427_427049

-- Definitions based on conditions
def pointP := (1, 3, -5)

def symmetric_point_with_respect_to_xOy_plane (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, p.2, -p.3)

-- Theorem statement, proof omitted
theorem symmetric_point_xOy_plane :
  symmetric_point_with_respect_to_xOy_plane pointP = (1, 3, 5) :=
by
  sorry

end symmetric_point_xOy_plane_l427_427049


namespace probability_of_negative_product_l427_427053

-- Definitions based on the problem's conditions
def die_faces := {-3, -2, -1, 0, 1, 2}
def total_outcomes := Finset.card (die_faces ×ˢ die_faces) -- total outcomes when rolling two dice

-- A function to check if the product of two numbers is negative
def is_negative_product (a b : ℤ) : Prop := a * b < 0

-- Counting how many outcomes yield a negative product
def negative_product_outcome_count :=
  (die_faces ×ˢ die_faces).filter (λ (p : ℤ × ℤ), is_negative_product p.1 p.2).card

-- Calculate the probability
def probability_negative_product :=
  (negative_product_outcome_count : ℝ) / (total_outcomes : ℝ)

-- The theorem to prove
theorem probability_of_negative_product :
  probability_negative_product = (1 : ℝ) / 3 :=
by 
  sorry

end probability_of_negative_product_l427_427053


namespace freight_cost_minimization_l427_427164

-- Define the main parameters: tonnage and costs for the trucks.
def freight_cost (num_seven_ton_trucks : ℕ) (num_five_ton_trucks : ℕ) : ℕ :=
  65 * num_seven_ton_trucks + 50 * num_five_ton_trucks

-- Define the total transported capacity by the two types of trucks.
def total_capacity (num_seven_ton_trucks : ℕ) (num_five_ton_trucks : ℕ) : ℕ :=
  7 * num_seven_ton_trucks + 5 * num_five_ton_trucks

-- Define the minimum freight cost given the conditions.
def minimum_freight_cost := 685

-- The theorem we want to prove.
theorem freight_cost_minimization : ∃ x y : ℕ, total_capacity x y ≥ 73 ∧
  (freight_cost x y = minimum_freight_cost) :=
by
  sorry

end freight_cost_minimization_l427_427164


namespace complex_number_condition_l427_427848

theorem complex_number_condition (b : ℝ) :
  (2 + b) / 5 = (2 * b - 1) / 5 → b = 3 :=
by
  sorry

end complex_number_condition_l427_427848


namespace angle_BDC_l427_427877

theorem angle_BDC {A B C D : Point} (h_isosceles : ∠ACB = 40 ∧ ∠ABC = 40)
  (h_AD_eq_BC : dist AD BC = 0) : ∠BDC = 30 :=
by
  sorry

end angle_BDC_l427_427877


namespace angle_AEB_eq_90_l427_427702

-- Define the basic setup
noncomputable def circlePoint (α β γ δ ε : Type) := sorry

-- Define the points on a circle
variables {A B C D E : Type}

-- Define our conditions
axiom points_on_circle : circlePoint A B C D E
axiom cd_parallel_ab : parallel (line C D) (line A B)
axiom ad_cd_bc : (distance A D) = (distance C D) ∧ (distance C D) = (distance B C)
axiom angle_ADC : angle A D C = 120

-- Define what we need to prove
theorem angle_AEB_eq_90 :
  ∀ {A B C D E : Type},
    circlePoint A B C D E →
    parallel (line C D) (line A B) →
    (distance A D) = (distance C D) ∧ (distance C D) = (distance B C) →
    angle A D C = 120 →
    (angle A E B = 90) := sorry

end angle_AEB_eq_90_l427_427702


namespace ratio_8_to_15_rounded_to_nearest_tenth_l427_427857

/-- In a fictional federation, eight out of the original fifteen regions had to approve a new charter for it to become effective. What is this ratio, eight to fifteen, rounded to the nearest tenth? --/
theorem ratio_8_to_15_rounded_to_nearest_tenth : (Real.ratio 8 15).roundToNearestTenth = 0.5 := by
  sorry

end ratio_8_to_15_rounded_to_nearest_tenth_l427_427857


namespace composite_divisor_bound_l427_427937

theorem composite_divisor_bound (n : ℕ) (hn : ¬Prime n ∧ 1 < n) : 
  ∃ a : ℕ, 1 < a ∧ a ≤ Int.sqrt (n : ℤ) ∧ a ∣ n :=
sorry

end composite_divisor_bound_l427_427937


namespace opposite_of_neg_two_is_two_l427_427991

theorem opposite_of_neg_two_is_two : -(-2) = 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l427_427991


namespace solve_for_c_l427_427473

noncomputable def triangle_side_c_proof (a c : ℝ) (C : ℝ) (S : ℝ) : Prop :=
  let b := 5 in -- derived from S and given conditions
  C = π * (2 / 3) ∧ -- 120 degrees in radians
  a = 3 ∧
  S = (15 * real.sqrt 3) / 4 ∧
  (c * c = a * a + b * b - 2 * a * b * real.cos C)

theorem solve_for_c (a : ℝ) (C : ℝ) (S : ℝ) :
  triangle_side_c_proof a 7 C S :=
by
  sorry -- Proof omitted

end solve_for_c_l427_427473


namespace first_term_exceeds_10k_l427_427575

def sequence : ℕ → ℕ
| 0     := 3
| (n+1) := 2 * (Finset.range (n+1)).sum sequence

theorem first_term_exceeds_10k : sequence 8 = 4374 ∧ sequence 9 = 13122 ∧ 10000 < sequence 9 :=
by
  -- We specify here since we're not providing the proof
  sorry

end first_term_exceeds_10k_l427_427575


namespace sum_of_solutions_l427_427626

theorem sum_of_solutions (x : ℝ) (h : (3 * x / 15 = 6 / x)) : (∑ x in { x | 3 * x / 15 = 6 / x }, x) = 0 :=
begin
  sorry
end

end sum_of_solutions_l427_427626


namespace power_function_solution_l427_427581

theorem power_function_solution :
  (∃ α : ℝ, (-2 : ℝ) ^ α = -1 / 8) →
  (∀ α : ℝ, (¬(-2 ^ α = -1 / 8) ∨ x ^ α = 27 → x = 1 / 3)) :=
by
  intros h α
  cases h with α' hα'
  apply or.intro_left 
  intro hf
  sorry

end power_function_solution_l427_427581


namespace glucose_in_mixed_solution_l427_427614

def concentration1 := 20 / 100  -- concentration of first solution in grams per cubic centimeter
def concentration2 := 30 / 100  -- concentration of second solution in grams per cubic centimeter
def volume1 := 80               -- volume of first solution in cubic centimeters
def volume2 := 50               -- volume of second solution in cubic centimeters

theorem glucose_in_mixed_solution :
  (concentration1 * volume1) + (concentration2 * volume2) = 31 := by
  sorry

end glucose_in_mixed_solution_l427_427614


namespace max_min_distance_difference_l427_427799

noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (x - y - 1) / real.sqrt 2

def on_curve (x y : ℝ) : Prop :=
  x^2 + (y - 1)^2 = 1 ∧ x ≤ 0

theorem max_min_distance_difference : 
  ∀ (a b : ℝ) (ha : ∃ (x y : ℝ), on_curve x y ∧ distance_to_line x y = a)
  (hb : ∃ (x y : ℝ), on_curve x y ∧ distance_to_line x y = b), 
  a - b = (real.sqrt 2 / 2) + 1 :=
sorry

end max_min_distance_difference_l427_427799


namespace f_sin_alpha_lt_f_cos_beta_l427_427308

variables {α β : ℝ}

def f : ℝ → ℝ :=
  sorry

-- Conditions
axiom f_even : ∀ x, f (-x) = f x
axiom f_periodic : ∀ x, f (x + 1) = 2 / f x ∧ f x ≠ 0
axiom f_monotone : ∀ x y, (2013 < x ∧ x < y ∧ y < 2014) → f x < f y
axiom α_acute : 0 < α ∧ α < π / 2
axiom β_acute : 0 < β ∧ β < π / 2

-- Proof goal
theorem f_sin_alpha_lt_f_cos_beta : f (sin α) < f (cos β) :=
  sorry

end f_sin_alpha_lt_f_cos_beta_l427_427308


namespace remaining_people_statement_l427_427098

-- Definitions of conditions
def number_of_people : Nat := 10
def number_of_knights (K : Nat) : Prop := K ≤ number_of_people
def number_of_liars (L : Nat) : Prop := L ≤ number_of_people
def statement (s : String) : Prop := s = "There are more liars" ∨ s = "There are equal numbers"

-- Main theorem
theorem remaining_people_statement (K L : Nat) (h_total : K + L = number_of_people) 
  (h_knights_behavior : ∀ k, k < K → statement "There are equal numbers") 
  (h_liars_behavior : ∀ l, l < L → statement "There are more liars") :
  K = 5 → L = 5 → ∀ i, i < number_of_people → (i < 5 → statement "There are more liars") ∧ (i >= 5 → statement "There are equal numbers") := 
by
  sorry

end remaining_people_statement_l427_427098


namespace nurse_missy_serving_time_l427_427928

noncomputable def standard_care_time (total_patients : ℕ) (special_fraction : ℚ) (percentage_increase : ℚ) (total_time : ℚ) : ℚ :=
  let standard_patients := total_patients * (1 - special_fraction)
  let special_patients := total_patients * special_fraction
  let standard_time := (total_time / (standard_patients + special_patients * (1 + percentage_increase)))
  standard_time

theorem nurse_missy_serving_time : 
  let total_patients := 12
  let special_fraction := 1 / 3 : ℚ
  let percentage_increase := 0.20 : ℚ
  let total_time := 64 : ℚ
  standard_care_time total_patients special_fraction percentage_increase total_time = 5 :=
by
  sorry

end nurse_missy_serving_time_l427_427928


namespace computation_result_l427_427300

theorem computation_result :
  (3 + 6 - 12 + 24 + 48 - 96 + 192 - 384) / (6 + 12 - 24 + 48 + 96 - 192 + 384 - 768) = 1 / 2 :=
by
  sorry

end computation_result_l427_427300


namespace total_tickets_sold_l427_427227

-- Define the conditions as constants or hypotheses
def total_collected : ℕ := 104
def price_adult : ℕ := 6
def price_child : ℕ := 4
def num_children_tickets : ℕ := 11

-- Define the statement to prove
theorem total_tickets_sold :
  (∃ (num_adult_tickets : ℕ), price_adult * num_adult_tickets + price_child * num_children_tickets = total_collected) →
  (∃ (num_adult_tickets : ℕ), (num_adult_tickets + num_children_tickets = 21)) :=
begin
  intro h,
  cases h with num_adult_tickets h_eq,
  use num_adult_tickets,
  rw [total_collected, price_adult, price_child, num_children_tickets] at h_eq,
  have h_simplified : 6 * num_adult_tickets + 44 = 104 := h_eq,
  have h_adult_tickets : 6 * num_adult_tickets = 60,
  { linarith, },
  have h_num_adult_tickets : num_adult_tickets = 10,
  { linarith, },
  rw h_num_adult_tickets,
  linarith,
end

end total_tickets_sold_l427_427227


namespace sum_of_distinct_real_numbers_l427_427499

theorem sum_of_distinct_real_numbers (p q r s : ℝ) (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : ∀ x : ℝ, x^2 - 12 * p * x - 13 * q = 0 -> (x = r ∨ x = s)) 
  (h2 : ∀ x : ℝ, x^2 - 12 * r * x - 13 * s = 0 -> (x = p ∨ x = q)) :
  p + q + r + s = 2028 :=
begin
  sorry
end

end sum_of_distinct_real_numbers_l427_427499


namespace remainder_of_500th_T_l427_427493

def sequence_T : ℕ → ℕ
| n := sorry -- Definition of the sequence T where each T(n) is the nth number in the order with exactly 9 ones in binary

theorem remainder_of_500th_T :
  let T := sequence_T in
  let M := T 500 in
  M % 500 = 281 :=
sorry

end remainder_of_500th_T_l427_427493


namespace supplement_of_complement_is_125_l427_427207

-- Definition of the initial angle
def initial_angle : ℝ := 35

-- Definition of the complement of the angle
def complement_angle (θ : ℝ) : ℝ := 90 - θ

-- Definition of the supplement of an angle
def supplement_angle (θ : ℝ) : ℝ := 180 - θ

-- Main theorem statement
theorem supplement_of_complement_is_125 : 
  supplement_angle (complement_angle initial_angle) = 125 := 
by
  sorry

end supplement_of_complement_is_125_l427_427207


namespace symmetric_line_equivalency_l427_427777

variable (A B C : Point ℝ)
variable (x y : ℝ)

-- Define the points and line equations
def pointA : Point ℝ := (5, 1)
def medianCM (p : Point ℝ) : Prop := 2 * p.1 - p.2 - 5 = 0
def altitudeBH (p : Point ℝ) : Prop := p.1 - 2 * p.2 - 5 = 0

-- Define what needs to be proved: Equations of BC and the symmetric line
theorem symmetric_line_equivalency (A_eq : A = (5, 1))
  (median_eq : ∀ p : Point ℝ, medianCM p)
  (altitude_eq : ∀ p : Point ℝ, altitudeBH p) :
  (∃ BC : Line ℝ, BC = 6 * x - 5 * y - 9 = 0) ∧
  (∃ sym_line : Line ℝ, sym_line = 38 * x - 9 * y - 125 = 0) :=
by
  sorry

end symmetric_line_equivalency_l427_427777


namespace sandwich_total_l427_427292

theorem sandwich_total :
  let billy_sandwiches := 49
  let katelyn_sandwiches := billy_sandwiches + Int.ofNat (Nat.round (0.3 * billy_sandwiches))
  let chloe_sandwiches := Int.ofNat (Nat.round ((3/5:Rat) * katelyn_sandwiches))
  let emma_sandwiches := 25
  let stella_sandwiches := 2 * emma_sandwiches
  billy_sandwiches + katelyn_sandwiches + chloe_sandwiches + emma_sandwiches + stella_sandwiches = 226 := 
by {
  -- Define the number of sandwiches each person made
  let billy_sandwiches := 49
  let katelyn_sandwiches := billy_sandwiches + Int.ofNat (Nat.round (0.3 * billy_sandwiches))
  let chloe_sandwiches := Int.ofNat (Nat.round ((3/5:Rat) * katelyn_sandwiches))
  let emma_sandwiches := 25
  let stella_sandwiches := 2 * emma_sandwiches

  -- Calculate the total number of sandwiches
  let total_sandwiches := billy_sandwiches + katelyn_sandwiches + chloe_sandwiches + emma_sandwiches + stella_sandwiches

  -- Prove the total number of sandwiches is 226
  show billy_sandwiches + katelyn_sandwiches + chloe_sandwiches + emma_sandwiches + stella_sandwiches = 226 from sorry
}

end sandwich_total_l427_427292


namespace cos_sum_diff_to_product_l427_427322

theorem cos_sum_diff_to_product (a b : ℝ) : 
  cos (a + b) - cos (a - b) = -2 * sin a * sin b := 
sorry

end cos_sum_diff_to_product_l427_427322


namespace smallest_n_for_partition_condition_l427_427914

theorem smallest_n_for_partition_condition :
  ∃ n : ℕ, n = 4 ∧ ∀ T, (T = {i : ℕ | 2 ≤ i ∧ i ≤ n}) →
  (∀ A B, (T = A ∪ B ∧ A ∩ B = ∅) →
   (∃ a b c, (a ∈ A ∨ a ∈ B) ∧ (b ∈ A ∨ b ∈ B) ∧ (a + b = c))) := sorry

end smallest_n_for_partition_condition_l427_427914


namespace sum_of_abc_l427_427150

theorem sum_of_abc (a b c : ℕ) (h1 : (300 / 75 : ℝ) = 4) (h2 : (sqrt 4 : ℝ) = 2) 
    (h3 : 2 = a * sqrt b / c) (h4 : b = 1) (h5 : c = 1) : a + b + c = 4 :=
by
  have a_val : a = 2 := by
    sorry
  rw [a_val, h4, h5]
  exact rfl

end sum_of_abc_l427_427150


namespace cardinality_inequality_l427_427482

open Set

variable {n m : ℕ}
variable {A : ℕ → Set ℕ}

-- Assume A_i are subsets of {1, 2, ..., n}
def valid_subsets : Prop :=
  ∀ i, 1 ≤ i ∧ i ≤ m → A i ⊆ (Finset.range (n+1)).toSet

-- Assume cardinal of each A_i is not divisible by 30
def cardinal_not_divisible_by_30 : Prop :=
  ∀ i, 1 ≤ i ∧ i ≤ m → Finset.card (A i).toFinSet % 30 ≠ 0

-- Assume cardinal of each A_i ∩ A_j (i ≠ j) is divisible by 30
def intersection_divisible_by_30 : Prop :=
  ∀ i j, 1 ≤ i ∧ i ≤ m → 1 ≤ j ∧ j ≤ m → i ≠ j → Finset.card (A i ∩ A j).toFinSet % 30 = 0

-- Prove the inequality
theorem cardinality_inequality (h1 : valid_subsets) (h2 : cardinal_not_divisible_by_30) (h3 : intersection_divisible_by_30) :
  2 * m - m / 30 ≤ 3 * n := 
by
  sorry

end cardinality_inequality_l427_427482


namespace complex_division_l427_427436

-- We define the variables a, b as real numbers
variables (a b : ℝ) (i : ℂ) [algebra ℝ ℂ]

-- i is the imaginary unit
axiom i_square : i * i = -1

-- Given that 3 + b * i over 1 - i equals a + b * i
theorem complex_division (h : (3 + b * i) / (1 - i) = a + b * i) : a + b = 3 :=
by
  -- Temporary placeholder for proof
  sorry

end complex_division_l427_427436


namespace increasing_interval_l427_427138

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem increasing_interval : {x : ℝ | 2 < x} = { x : ℝ | (x - 3) * Real.exp x > 0 } :=
by
  sorry

end increasing_interval_l427_427138


namespace maxValidSubsets_l427_427074

-- Definition of the set S
def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Definition of the condition for subsets Ai
def isValidSubset (S : Set ℕ) (A : Set ℕ) : Prop := A ⊆ S ∧ A.card = 5

-- Definition of the condition for intersections between Ai's
def validIntersection (A B : Set ℕ) : Prop := (A ∩ B).card ≤ 2

-- Definition of the problem statement
theorem maxValidSubsets : ∃ k, ∀ (A : Fin k → Set ℕ), (∀ i, isValidSubset S (A i)) ∧ (∀ i j, i < j → validIntersection (A i) (A j)) → k ≤ 6 := sorry

end maxValidSubsets_l427_427074


namespace cyclic_inequality_l427_427025

theorem cyclic_inequality
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / (a^3 + b^3 + a * b * c) + 1 / (b^3 + c^3 + a * b * c) + 1 / (c^3 + a^3 + a * b * c)) ≤ 1 / (a * b * c) :=
by
  sorry

end cyclic_inequality_l427_427025


namespace permutation_count_x_y_z_l427_427177

theorem permutation_count_x_y_z :
  let n := 6
  let count_x : ℕ → ℕ := λ n, (choose n 2) + (choose n 3)
  let count_y : ℕ → ℕ := λ n, (choose n 1) + (choose n 3) + (choose n 5)
  let count_z := (choose 6 2)
  (Σ x in {2, 3}, Σ y in {1, 3, 5}, (x + y + 2 = n → (count_x x) * (count_y y) * count_z)) = 60 :=
by
  sorry

end permutation_count_x_y_z_l427_427177


namespace max_sqrt_expr_l427_427743

open Real

theorem max_sqrt_expr : ∃ a ∈ Icc (-6 : ℝ) 3, √((3 - a) * (a + 6)) = 9 / 2 := by
  sorry

end max_sqrt_expr_l427_427743


namespace composite_number_iff_ge_2_l427_427773

theorem composite_number_iff_ge_2 (n : ℕ) : 
  ¬(Prime (3^(2*n+1) - 2^(2*n+1) - 6^n)) ↔ n ≥ 2 := by
  sorry

end composite_number_iff_ge_2_l427_427773


namespace numblian_words_l427_427931

theorem numblian_words (alphabet_size max_letters : ℕ)
  (h1 : alphabet_size = 6)
  (h2 : max_letters = 3)
  (word_size : ℕ → Prop)
  (h3 : ∀ n, n > 0 → word_size n)
  (h4 : ∀ s, word_size s → (s = 1 ∨ s = 2 ∨ (s = 3 → ∃ l, ∃ n, l ≠ n ∧ multiset.card (multiset.replicate_lift l 3) = 3 ∨ multiset.card (multiset.replicate_lift l 2 + multiset.singleton n) = 3)))
  : finset.card (finset.filter word_size (finset.univ : finset (finset (finset.range (alphabet_size)))) = 138 :=
by
  sorry

end numblian_words_l427_427931


namespace probability_circle_in_square_l427_427426

theorem probability_circle_in_square (x y : ℝ) (h₁ : -1 ≤ x ∧ x ≤ 1) (h₂ : -1 ≤ y ∧ y ≤ 1) :
  (set.probability (λ p : ℝ × ℝ, p.1^2 + p.2^2 < 1 / 4) ([-1, 1] ×ˢ [-1, 1])) = π / 16 :=
sorry

end probability_circle_in_square_l427_427426


namespace probability_property_l427_427065

def Q (x : ℝ) : ℝ := x^2 - 4*x - 4
def a : ℕ := 1
def b : ℕ := 0
def c : ℕ := 0
def d : ℕ := 0
def e : ℕ := 9

theorem probability_property (x : ℝ) (hx : 3 ≤ x ∧ x ≤ 12) :
  (floor (real.sqrt (Q x)) = real.sqrt (Q (ceil x))) →
  a + b + c + d + e = 10 :=
by
  sorry

end probability_property_l427_427065


namespace complex_number_in_first_quadrant_l427_427531

-- Definition of the imaginary unit
def i : ℂ := Complex.I

-- Definition of the complex number z
def z : ℂ := i * (1 - i)

-- Coordinates of the complex number z
def z_coords : ℝ × ℝ := (z.re, z.im)

-- Statement asserting that the point corresponding to z lies in the first quadrant
theorem complex_number_in_first_quadrant : z_coords.fst > 0 ∧ z_coords.snd > 0 := 
by
  sorry

end complex_number_in_first_quadrant_l427_427531


namespace annual_growth_rate_for_GDP_doubling_l427_427714

theorem annual_growth_rate_for_GDP_doubling :
  ∃ x : ℝ, (1 + x)^10 = 2 ∧ x ≈ 0.071773462 :=
by
sorry

end annual_growth_rate_for_GDP_doubling_l427_427714


namespace unique_solution_exists_l427_427719

theorem unique_solution_exists : ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ (real.sqrt (a^2 + b^2) = a^2 + b^2) :=
sorry

end unique_solution_exists_l427_427719


namespace daily_sales_when_priced_at_3_5_price_to_achieve_profit_l427_427257

theorem daily_sales_when_priced_at_3_5
  (base_price : ℕ) (items_sold_base : ℕ) (price_increase : ℚ) (items_decrease_per_0_1 : ℕ) :
  base_price = 3 → items_sold_base = 500 → 
  price_increase = 0.5 → items_decrease_per_0_1 = 10 → 
  let decrease_in_items := (price_increase / 0.1) * items_decrease_per_0_1,
      new_price := base_price + price_increase,
      new_sales := items_sold_base - decrease_in_items
  in new_price = 3.5 → new_sales = 450 :=
by
  intros
  sorry

theorem price_to_achieve_profit
  (wholesale_price : ℕ) (regulation_factor : ℚ)
  (base_price : ℕ) (items_sold_base : ℕ)
  (price_increase_per_0_1 : ℚ) (decrease_per_0_1 : ℕ)
  (target_profit : ℕ) :
  wholesale_price = 2 → regulation_factor = 2.5 →
  base_price = 3 → items_sold_base = 500 → 
  price_increase_per_0_1 = 0.1 → decrease_per_0_1 = 10 → 
  target_profit = 800 → 
  let max_price := wholesale_price * regulation_factor,
      increase_price := max_price - base_price,
      decreased_items := decrease_per_0_1 * (increase_price/price_increase_per_0_1).to_nat,
      new_sales_base := items_sold_base - decreased_items,
      profit_per_item := base_price + increase_price - wholesale_price,
      total_profit := profit_per_item * new_sales_base
  in increase_price ≤ max_price - base_price ∧ total_profit = target_profit → 
     base_price + increase_price = 4 :=
by
  intros
  sorry

end daily_sales_when_priced_at_3_5_price_to_achieve_profit_l427_427257


namespace probability_inside_seven_spheres_l427_427679

-- Definitions of the radii
def tetrahedron_edge_length (s : ℝ) : Prop := s > 0
def circumscribed_sphere_radius (s : ℝ) : ℝ := (s * real.sqrt 6) / 4
def inscribed_sphere_radius (s : ℝ) : ℝ := (s * real.sqrt 6) / 12
def external_sphere_radius (s : ℝ) : ℝ := ((circumscribed_sphere_radius s) - (inscribed_sphere_radius s)) / 2
def small_sphere_radius (s : ℝ) : ℝ := (inscribed_sphere_radius s) / 2

-- Definition of the volumes
def circumscribed_sphere_volume (s : ℝ) : ℝ := (4 / 3) * real.pi * (circumscribed_sphere_radius s)^3
def total_spheres_volume (s : ℝ) : ℝ :=
  (4 * (4 / 3) * real.pi * (external_sphere_radius s)^3) +
  (2 * (4 / 3) * real.pi * (small_sphere_radius s)^3)

-- Probability calculation
def probability_inside_any_sphere (s : ℝ) : ℝ :=
  (total_spheres_volume s) / (circumscribed_sphere_volume s)

-- The theorem to prove
theorem probability_inside_seven_spheres (s : ℝ) (hs : tetrahedron_edge_length s) :
  probability_inside_any_sphere s = 0.25 :=
by
  sorry

end probability_inside_seven_spheres_l427_427679


namespace construct_foci_l427_427424

noncomputable def hyperbola_foci (a b x₁ y₁ : ℝ) : ℝ × ℝ := 
  let c := real.sqrt (a^2 + b^2)
  (c, -c)

theorem construct_foci (a b x₁ y₁ : ℝ) (h_on_hyperbola : (x₁ * x₁ / a^2) - (y₁ * y₁ / b^2) = 1) :
  hyperbola_foci a b x₁ y₁ = (real.sqrt (a^2 + b^2), -real.sqrt (a^2 + b^2)) :=
sorry

end construct_foci_l427_427424


namespace total_clowns_in_mobiles_l427_427963

theorem total_clowns_in_mobiles (mobiles : ℕ) (clowns_per_mobile : ℕ) (h_mobiles : mobiles = 5) (h_clowns_per_mobile : clowns_per_mobile = 28) :
  mobiles * clowns_per_mobile = 140 :=
by
  rw [h_mobiles, h_clowns_per_mobile]
  norm_num

end total_clowns_in_mobiles_l427_427963


namespace isosceles_trapezoid_ratio_l427_427455

theorem isosceles_trapezoid_ratio (a b h : ℝ) 
  (h1: h = b / 2)
  (h2: a = 1 - ((1 - b) / 2))
  (h3 : 1 = ((a + 1) / 2)^2 + (b / 2)^2) :
  b / a = (-1 + Real.sqrt 7) / 2 := 
sorry

end isosceles_trapezoid_ratio_l427_427455


namespace systematic_sampling_example_l427_427661

theorem systematic_sampling_example (rows seats : ℕ) (all_seats_filled : Prop) (chosen_seat : ℕ):
  rows = 50 ∧ seats = 60 ∧ all_seats_filled ∧ chosen_seat = 18 → sampling_method = "systematic_sampling" :=
by
  sorry

end systematic_sampling_example_l427_427661


namespace inv_func_eval_l427_427444

theorem inv_func_eval (a : ℝ) (h : 8^(1/3) = a) : (fun y => (Real.log y / Real.log 8)) (a + 2) = 2/3 :=
by
  sorry

end inv_func_eval_l427_427444


namespace smallest_n_for_distinct_set_l427_427911

theorem smallest_n_for_distinct_set (k : ℕ) (hk : 2 ≤ k) : 
  ∃ (n : ℕ), (n ≥ k + 1) ∧ (∀ A : Finset ℝ, A.card = n → (∀ a ∈ A, ∃ B ⊆ A, B.card = k ∧ a = B.sum)) ↔ n = k + 4 := 
by
  sorry

end smallest_n_for_distinct_set_l427_427911


namespace inequality_no_real_solutions_l427_427889

theorem inequality_no_real_solutions (a b : ℝ) 
  (h : ∀ x : ℝ, a * Real.cos x + b * Real.cos (3 * x) ≤ 1) : 
  |b| ≤ 1 :=
sorry

end inequality_no_real_solutions_l427_427889


namespace hyperbola_eccentricity_eq_sqrt_two_l427_427570

theorem hyperbola_eccentricity_eq_sqrt_two
  (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
  (h_parallel : b / a = 1) :
  (a^2 + b^2) / a^2 = 2 :=
by
  have h_eq : b = a,
  { -- proving that b = a based on h_parallel
    sorry },
  have h_eccentricity : (a^2 + a^2) / a^2 = 2,
  { -- proving the eccentricity formula for equilateral hyperbola
    sorry },
  exact h_eccentricity

end hyperbola_eccentricity_eq_sqrt_two_l427_427570


namespace orchard_total_mass_l427_427698

-- Define the number of apple trees, apple yield, number of peach trees, and peach yield
def num_apple_trees : ℕ := 30
def apple_yield_per_tree : ℕ := 150
def num_peach_trees : ℕ := 45
def peach_yield_per_tree : ℕ := 65

-- Calculate the total yield of apples and peaches
def total_apple_yield : ℕ := num_apple_trees * apple_yield_per_tree
def total_peach_yield : ℕ := num_peach_trees * peach_yield_per_tree

-- Define the total mass of fruit harvested
def total_fruit_yield : ℕ := total_apple_yield + total_peach_yield

-- Theorem stating that the total mass of fruit harvested is 7425 kg 
theorem orchard_total_mass : total_fruit_yield = 7425 := by
  simp [total_apple_yield, total_peach_yield, num_apple_trees, apple_yield_per_tree, num_peach_trees, peach_yield_per_tree]
  sorry

end orchard_total_mass_l427_427698


namespace arithmetic_seq_general_formula_sum_of_first_n_terms_of_combined_sequences_l427_427787

theorem arithmetic_seq_general_formula (a : ℕ → ℚ) (h₁ : a 1 = 1) (h₂ : a 2 + a 3 = 5) :
  ∀ n, a n = 1 + (2 / 3) * (n - 1) := sorry

theorem sum_of_first_n_terms_of_combined_sequences (a b : ℕ → ℚ) (h₁ : a 1 = 1) (h₂ : a 2 + a 3 = 5)
  (h₃ : b 1 = 2) (h₄ : b 3 = 2 * b 2) :
  ∀ n, 
    (∑ k in range 1 (n+1), (a k + b k)) = (2*n^2 + n)/3 + 2^(n+1) - 2 := sorry

end arithmetic_seq_general_formula_sum_of_first_n_terms_of_combined_sequences_l427_427787


namespace parabola_find_a_l427_427580

theorem parabola_find_a (a b c : ℤ) :
  (∀ x y : ℤ, (x, y) ∈ [(1, 4), (-2, 3)] → y = a * x ^ 2 + b * x + c) →
  (∃ x y : ℤ, y = a * (x + 1) ^ 2 + 2 ∧ (x, y) = (-1, 2)) →
  a = 1 := 
by 
  sorry

end parabola_find_a_l427_427580


namespace hyperbola_center_l427_427325

theorem hyperbola_center :
  ∃ (c : ℝ × ℝ), c = (3, 5) ∧
  (9 * (x - c.1)^2 - 36 * (y - c.2)^2 - (1244 - 243 - 1001) = 0) :=
sorry

end hyperbola_center_l427_427325


namespace geometric_sequence_a12_l427_427876

noncomputable def a_n (a1 r : ℝ) (n : ℕ) : ℝ :=
  a1 * r ^ (n - 1)

theorem geometric_sequence_a12 (a1 r : ℝ) 
  (h1 : a_n a1 r 7 * a_n a1 r 9 = 4)
  (h2 : a_n a1 r 4 = 1) :
  a_n a1 r 12 = 16 := sorry

end geometric_sequence_a12_l427_427876


namespace supplement_of_complement_of_35_degree_angle_l427_427211

theorem supplement_of_complement_of_35_degree_angle : 
  ∀ α β : ℝ,  α = 35 ∧ β = (90 - α) → (180 - β) = 125 :=
by
  intros α β h
  rcases h with ⟨h1, h2⟩
  rw h1 at h2
  rw h2
  linarith

end supplement_of_complement_of_35_degree_angle_l427_427211


namespace overtime_hours_correct_l427_427234

def regular_pay_rate : ℕ := 3
def max_regular_hours : ℕ := 40
def total_pay_received : ℕ := 192
def overtime_pay_rate : ℕ := 2 * regular_pay_rate
def regular_earnings : ℕ := regular_pay_rate * max_regular_hours
def additional_earnings : ℕ := total_pay_received - regular_earnings
def calculated_overtime_hours : ℕ := additional_earnings / overtime_pay_rate

theorem overtime_hours_correct :
  calculated_overtime_hours = 12 :=
by
  sorry

end overtime_hours_correct_l427_427234


namespace question1_question2_l427_427822

def setA (a : ℝ) : set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 3}
def setB : set ℝ := {x | x < -1 ∨ x > 1}

theorem question1 (a : ℝ) : (setA a ∩ setB = ∅) ↔ (a > 3) := by
  -- proof placeholder
  sorry

theorem question2 (a : ℝ) : (setA a ∪ setB = set.univ) ↔ (-2 ≤ a ∧ a ≤ -1/2) := by
  -- proof placeholder
  sorry

end question1_question2_l427_427822


namespace determine_lambda_l427_427368

open Complex

noncomputable def proof_problem (ω : ℂ) (λ : ℝ) : Prop :=
  abs ω = 3 ∧ λ > 1 ∧
  equilateral_triangle ω ω^3 (λ * ω) → 
  λ = 1 + Real.sqrt(32 / 3)

theorem determine_lambda (ω : ℂ) (λ : ℝ) : proof_problem ω λ := sorry

end determine_lambda_l427_427368


namespace solve_poly_l427_427564

open Real

-- Define the condition as a hypothesis
def prob_condition (x : ℝ) : Prop :=
  arctan (1 / x) + arctan (1 / (x^5)) = π / 6

-- Define the statement to be proven that x satisfies the polynomial equation
theorem solve_poly (x : ℝ) (h : prob_condition x) :
  x^6 - sqrt 3 * x^5 - sqrt 3 * x - 1 = 0 :=
sorry

end solve_poly_l427_427564


namespace average_weight_increase_l427_427126

theorem average_weight_increase 
  (A : ℝ) (X : ℝ)
  (h1 : 8 * (A + X) = 8 * A + 36) :
  X = 4.5 := 
sorry

end average_weight_increase_l427_427126


namespace polar_circle_equation_l427_427467

theorem polar_circle_equation {r : ℝ} {phi : ℝ} {rho theta : ℝ} :
  (r = 2) → (phi = π / 3) → (rho = 4 * Real.cos (theta - π / 3)) :=
by
  intros hr hphi
  sorry

end polar_circle_equation_l427_427467


namespace decreasing_fun_iff_l427_427806

noncomputable def f (a x : ℝ) : ℝ := (log a + log x) / x

theorem decreasing_fun_iff (a : ℝ) : (∀ x : ℝ, 1 ≤ x → f a x ≤ f a 1) ↔ a ≥ real.exp 1 := by
  sorry

end decreasing_fun_iff_l427_427806


namespace max_cells_colored_without_tetromino_l427_427742

theorem max_cells_colored_without_tetromino (n : ℕ) (n = 3000) :
  ∃ max_c : ℕ, max_c = 7000 ∧
  ∀ (board : fin 4 × fin n → bool), 
    (∀ (i j k l : fin 4) (a b c d : fin n), 
      board (i, a) = tt ∧ board (j, b) = tt ∧ board (k, c) = tt ∧ board (l, d) = tt 
      → ¬(is_tetromino (i, a) (j, b) (k, c) (l, d)))
    → ∑ x in finset.univ, ∑ y in finset.univ, (if board (x, y) then 1 else 0) ≤ max_c :=
begin
  sorry
end


end max_cells_colored_without_tetromino_l427_427742


namespace set_cardinality_congruence_l427_427378

open Nat

def S (m n : ℕ) : Finset (ℕ × ℕ) :=
  {ab ∈ (Finset.product (Finset.range (m + 1)) (Finset.range (n + 1))) | ab.1 > 0 ∧ ab.2 > 0 ∧ gcd ab.1 ab.2 = 1}

theorem set_cardinality_congruence (d r : ℕ) (hd : 0 < d) (hr : 0 < r) :
  ∃ (m n : ℕ), d ≤ m ∧ d ≤ n ∧ (S m n).card % d = r % d :=
by
  sorry

end set_cardinality_congruence_l427_427378


namespace base_n_multiple_of_5_l427_427754

-- Define the polynomial f(n)
def f (n : ℕ) : ℕ := 4 + n + 3 * n^2 + 5 * n^3 + n^4 + 4 * n^5

-- The main theorem to be proven
theorem base_n_multiple_of_5 (n : ℕ) (h1 : 2 ≤ n) (h2 : n ≤ 100) : 
  f n % 5 ≠ 0 :=
by sorry

end base_n_multiple_of_5_l427_427754


namespace sum_fractions_geq_six_l427_427369

variable (x y z : ℝ)
variable (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)

theorem sum_fractions_geq_six : 
  (x / y + y / z + z / x + x / z + z / y + y / x) ≥ 6 := 
by
  sorry

end sum_fractions_geq_six_l427_427369


namespace PJ_approx_10_81_l427_427884

noncomputable def PJ_length (P Q R J : Type) (PQ PR QR : ℝ) : ℝ :=
  if PQ = 30 ∧ PR = 29 ∧ QR = 27 then 10.81 else 0

theorem PJ_approx_10_81 (P Q R J : Type) (PQ PR QR : ℝ):
  PQ = 30 ∧ PR = 29 ∧ QR = 27 → PJ_length P Q R J PQ PR QR = 10.81 :=
by sorry

end PJ_approx_10_81_l427_427884


namespace find_b_for_extreme_value_l427_427393

noncomputable def f (x b : ℝ) : ℝ :=
  (1 / 3) * x^3 + (1 / 2) * (b - 1) * x^2 + b^2 * x

theorem find_b_for_extreme_value :
  ∃ b : ℝ, ∀ x : ℝ, f x b = (1 / 3) * x^3 + (1 / 2) * (b - 1) * x^2 + b^2 * x
      ∧ ∃ (b = 0), (f' (1) == 0) :=
sorry

end find_b_for_extreme_value_l427_427393


namespace perfect_square_iff_form_l427_427760

-- Defining the function to check if a natural number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

-- Defining the function for the given expression
def factorial_product (n : ℕ) : ℕ :=
  (List.range (2 * n + 1)).map Nat.factorial |> List.product

-- Defining the main expression
def given_expression (n : ℕ) : ℕ :=
  factorial_product n / Nat.factorial (n + 1)

-- The proof problem
theorem perfect_square_iff_form (n : ℕ) :
  (is_perfect_square (given_expression n)) ↔
    (∃ k : ℕ, n = 4 * k * (k + 1)) ∨ (∃ k : ℕ, n = 2 * k^2 - 1) := by
  sorry

end perfect_square_iff_form_l427_427760


namespace longest_badminton_match_duration_l427_427586

theorem longest_badminton_match_duration :
  let hours := 12
  let minutes := 25
  (hours * 60 + minutes = 745) :=
by
  sorry

end longest_badminton_match_duration_l427_427586


namespace sufficient_but_not_necessary_for_agtb_l427_427279

theorem sufficient_but_not_necessary_for_agtb (a b : ℝ) : 
  (a > b + 1) → (a > b) ∧ ¬((a > b) → a > b + 1) :=
by
  intro h
  split
  { exact h.trans (lt_add_of_pos_right b zero_lt_one) }
  { intro h1
    have h2 : ¬ (a > b → a > b + 1) := 
      by
        push_neg
        use b + 1
        linarith
    exact h2 h1 }
  sorry

end sufficient_but_not_necessary_for_agtb_l427_427279


namespace sum_of_distinct_roots_l427_427521

theorem sum_of_distinct_roots 
  (p q r s : ℝ)
  (h1 : p ≠ q)
  (h2 : p ≠ r)
  (h3 : p ≠ s)
  (h4 : q ≠ r)
  (h5 : q ≠ s)
  (h6 : r ≠ s)
  (h_roots1 : (x : ℝ) -> x^2 - 12*p*x - 13*q = 0 -> x = r ∨ x = s)
  (h_roots2 : (x : ℝ) -> x^2 - 12*r*x - 13*s = 0 -> x = p ∨ x = q) : 
  p + q + r + s = 1716 := 
by 
  sorry

end sum_of_distinct_roots_l427_427521


namespace sum_of_radii_of_tangent_circles_l427_427258

theorem sum_of_radii_of_tangent_circles : 
  ∃ r1 r2 : ℝ, 
    r1 > 0 ∧
    r2 > 0 ∧
    ((r1 - 4)^2 + r1^2 = (r1 + 2)^2) ∧ 
    ((r2 - 4)^2 + r2^2 = (r2 + 2)^2) ∧
    r1 + r2 = 12 :=
by
  sorry

end sum_of_radii_of_tangent_circles_l427_427258


namespace negative_number_is_A_l427_427280

theorem negative_number_is_A :
  ∃ (x : ℤ), (x = -6 ∧ x < 0) ∧ (∀ (y : ℤ), (y = 0 ∨ y = 2 / 10 ∨ y = 3) → y ≥ 0) :=
begin
  sorry
end

end negative_number_is_A_l427_427280


namespace max_minute_hands_l427_427230

theorem max_minute_hands (m n : ℕ) (h : m * n = 27) : m + n ≤ 28 :=
sorry

end max_minute_hands_l427_427230


namespace find_g_inv_f_15_l427_427834

variable (f g : ℝ → ℝ)
variable (h : f⁻¹ ∘ g = λ x, x^4 - 1)
variable (g_inv_exists : ∃ g_inv : ℝ → ℝ, g ∘ g_inv = id ∧ g_inv ∘ g = id)

theorem find_g_inv_f_15 : g⁻¹ (f 15) = 2 :=
by
  sorry

end find_g_inv_f_15_l427_427834


namespace problem_statement_l427_427062

open Classical

noncomputable def P (x : ℝ) (b : ℕ) : ℝ :=
  (x + Real.sqrt (x^2 - 4))^b + (x - Real.sqrt (x^2 - 4))^b / 2^b

def sequence_S (b : ℕ) (n : ℕ) : ℕ → ℝ
| 0     := P 6 b
| (i+1) := P (sequence_S b n i) b

noncomputable def M (b : ℕ) (n : ℕ) : ℕ :=
  (b^(2^n) + 1) / 2

theorem problem_statement (b : ℕ) (n : ℕ) (h1 : Nat.Odd b) (h2 : 2 ≤ n) :
  M b n ∣ (sequence_S b n (2^n - 1) - 6) :=
sorry

end problem_statement_l427_427062


namespace gcd_1729_1309_eq_7_l427_427620

theorem gcd_1729_1309_eq_7 : Nat.gcd 1729 1309 = 7 := by
  sorry

end gcd_1729_1309_eq_7_l427_427620


namespace inequality_holds_l427_427414

noncomputable def f (a x : ℝ) := a * exp (x / 2) - x
noncomputable def g (x : ℝ) := x * log x - (1 / 2) * x ^ 2

theorem inequality_holds (a : ℝ) (h : a < -exp (-2)) : ∀ x : ℝ, 0 < x → f a x < g x :=
by sorry

end inequality_holds_l427_427414


namespace find_standard_equation_slopes_are_constant_l427_427788

-- Define the ellipse and given conditions
def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Given conditions
variables (a b : ℝ)
def eccentricity : ℝ := 1 / 2
def F_2 := (1 : ℝ, 0)
def T := (4 : ℝ, 0)
def perimeter (A B F_1 : ℝ × ℝ) : ℝ := dist A B + dist B F_1 + dist F_1 A

-- Problem Ⅰ
theorem find_standard_equation (h1 : a > 0) (h2 : b > 0) (h_ecc : eccentricity = 1 / 2)
  (h_perim : ∃ (A B F_1 : ℝ × ℝ), line_through B F_2 ∧ 
  (A, B ∈ ellipse a b) ∧ perimeter A B F_1 = 8) : 
  ellipse 4 3 := sorry

-- Problem Ⅱ
theorem slopes_are_constant (l : ℝ × ℝ → Prop)
  (h_line : ∀ x, l (x, (x - 1))) :
  ∀ (A B : ℝ × ℝ), A ∈ ellipse a b → B ∈ ellipse a b → 
  (T.2 - A.1) / (T.1 - A.1) + (T.2 - B.1) / (T.1 - B.1) = 0 := sorry

end find_standard_equation_slopes_are_constant_l427_427788


namespace cost_of_supplies_l427_427171

theorem cost_of_supplies (x y z : ℝ) 
  (h1 : 3 * x + 7 * y + z = 3.15) 
  (h2 : 4 * x + 10 * y + z = 4.2) :
  (x + y + z = 1.05) :=
by 
  sorry

end cost_of_supplies_l427_427171


namespace inequality_solution_l427_427601

theorem inequality_solution (x : ℝ) :
  x + 1 ≥ -3 ∧ -2 * (x + 3) > 0 ↔ -4 ≤ x ∧ x < -3 :=
by sorry

end inequality_solution_l427_427601


namespace composite_divisor_le_sqrt_l427_427934

noncomputable def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d, d > 1 ∧ d < n ∧ n % d = 0

theorem composite_divisor_le_sqrt (n : ℕ) (h : is_composite n) :
  ∃ d, 1 < d ∧ d ≤ nat.sqrt n ∧ n % d = 0 :=
sorry

end composite_divisor_le_sqrt_l427_427934


namespace interval_length_l427_427879

theorem interval_length (a b : ℝ) 
  (h_freq : real := 0.3)
  (h_height : real := 0.06)
  (h_interval : b - a = h_freq / h_height) :
  |a - b| = 5 := by
    sorry

end interval_length_l427_427879


namespace unique_pos_int_log_condition_l427_427312

theorem unique_pos_int_log_condition 
  (m : ℕ) 
  (hm : m = 3^18) 
  (log_cond : Real.log 3 (Real.log 27 m) = Real.log 9 (Real.log 9 m)) : 
  m = 3 ^ 18 ∧ (m.digits.sum = 45) :=
by
  sorry

end unique_pos_int_log_condition_l427_427312


namespace largest_n_for_factored_quad_l427_427347

theorem largest_n_for_factored_quad (n : ℤ) (b d : ℤ) 
  (h1 : 6 * d + b = n) (h2 : b * d = 72) 
  (factorable : ∃ x : ℤ, (6 * x + b) * (x + d) = 6 * x ^ 2 + n * x + 72) : 
  n ≤ 433 :=
sorry

end largest_n_for_factored_quad_l427_427347


namespace typing_together_time_typing_in_turns_time_possible_reordering_l427_427275

noncomputable def A : ℝ := 1 / 24
noncomputable def B : ℝ := 1 / 20
noncomputable def C : ℝ := 1 / 16
noncomputable def D : ℝ := 1 / 12

theorem typing_together_time :
  (A + B + C + D) * (80 / 19) = 1 :=
by sorry

theorem typing_in_turns_time :
  let total_time := 16 + 2 / 3 in
  total_time + 5 * A ≥ 17 + 1 / 6 :=
by sorry

theorem possible_reordering :
  ∃ orders : list (fin 4),
  (∀ (order : list (fin 4)),
  (order ≠ [0, 1, 2, 3] ∧ order ≠ [1, 2, 3, 0] ∧ order ≠ [2, 3, 0, 1] ∧ order ≠ [3, 0, 1, 2]) ∧
  ((D + A + B + C) * (16 + 2 / 3) = 1)) :=
by sorry

end typing_together_time_typing_in_turns_time_possible_reordering_l427_427275


namespace sum_of_distinct_roots_l427_427520

theorem sum_of_distinct_roots 
  (p q r s : ℝ)
  (h1 : p ≠ q)
  (h2 : p ≠ r)
  (h3 : p ≠ s)
  (h4 : q ≠ r)
  (h5 : q ≠ s)
  (h6 : r ≠ s)
  (h_roots1 : (x : ℝ) -> x^2 - 12*p*x - 13*q = 0 -> x = r ∨ x = s)
  (h_roots2 : (x : ℝ) -> x^2 - 12*r*x - 13*s = 0 -> x = p ∨ x = q) : 
  p + q + r + s = 1716 := 
by 
  sorry

end sum_of_distinct_roots_l427_427520


namespace polar_conversion_equiv_l427_427039

noncomputable def polar_convert (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
if r < 0 then (-r, θ + Real.pi) else (r, θ)

theorem polar_conversion_equiv : polar_convert (-3) (Real.pi / 4) = (3, 5 * Real.pi / 4) :=
by
  sorry

end polar_conversion_equiv_l427_427039


namespace opposite_of_neg_two_is_two_l427_427988

theorem opposite_of_neg_two_is_two : -(-2) = 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l427_427988


namespace opposite_of_neg_two_l427_427982

theorem opposite_of_neg_two : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_two_l427_427982


namespace overall_percentage_badminton_l427_427129

theorem overall_percentage_badminton (N S : ℕ) (pN pS : ℝ) :
  N = 1500 → S = 1800 → pN = 0.30 → pS = 0.35 → 
  ( (N * pN + S * pS) / (N + S) ) * 100 = 33 := 
by
  intros hN hS hpN hpS
  sorry

end overall_percentage_badminton_l427_427129


namespace sum_f_k_1_to_22_l427_427800

noncomputable def f : ℝ → ℝ :=
sorry

theorem sum_f_k_1_to_22 :
  (∀ x y : ℝ, f (x + y) + f (x - y) = f x * f y) →
  f 1 = 1 →
  (∑ k in finset.range 22, f (k + 1)) = -3 :=
begin
  intros h1 h2,
  sorry
end

end sum_f_k_1_to_22_l427_427800


namespace no_negative_exponents_l427_427441

theorem no_negative_exponents (a b c d : ℤ) (h : 2^a + 2^b = 3^c + 3^d) : 
  (a < 0 → False) ∧ (b < 0 → False) ∧ (c < 0 → False) ∧ (d < 0 → False) :=
sorry

end no_negative_exponents_l427_427441


namespace num_nat_solutions_l427_427883

theorem num_nat_solutions : 
  {n : ℕ | ∃ (x y : ℕ), 2 * x + y = 7}.card = 4 := by
sorry

end num_nat_solutions_l427_427883


namespace coefficient_x3_term_expansion_l427_427953

theorem coefficient_x3_term_expansion :
  (∑ k in Finset.range (5 + 1), (Nat.choose 5 k) * (1 : ℝ)^(5 - k) * (2 : ℝ)^(k) * (x^(k : ℕ))).coeff 3 = 80 := by sorry

end coefficient_x3_term_expansion_l427_427953


namespace find_a1_a2_general_formula_max_sum_bn_l427_427785

-- Definition of the sequence and conditions
def sequence_sum (n : ℕ) (a : ℕ → ℕ) : ℕ := ∑ i in Finset.range n, a i

axiom seq_condition (a : ℕ → ℕ) (n : ℕ) : 
  sequence_sum n a = (1 / 4) * ((a n + 1) ^ 2) ∧ a n > 0

-- Statement: Find a_1 and a_2
theorem find_a1_a2 (a : ℕ → ℕ) (h : ∀ n, seq_condition a n) : 
  a 1 = 1 ∧ a 2 = 3 :=
sorry

-- Statement: General formula for {a_n}
theorem general_formula (a : ℕ → ℕ) (h : ∀ n, seq_condition a n) :
  ∀ n, a n = 2 * n - 1 :=
sorry

-- Statement: Maximum sum of the first several terms of {b_n}
theorem max_sum_bn (a : ℕ → ℕ) (b : ℕ → ℕ) (h : ∀ n, seq_condition a n) 
  (h_b_def : ∀ n, b n = 20 - a n) : 
  ∑ i in Finset.range 10, b i = (190 : ℕ) :=
sorry

end find_a1_a2_general_formula_max_sum_bn_l427_427785


namespace triangleSimilarity_l427_427175

open EuclideanGeometry

variables {A B C I T S B' C' : Point}
variable [Incenter I]
variable [Incircle I]

noncomputable def isTriangleABC (A B C : Point) : Prop :=
  ∃ (T : Point) (S : Point) (B' : Point) (C' : Point),
    IncircleTouchesBC I T ∧ 
    (lineThroughParallel T I A S) ∧ 
    (tangentMeetsAt S AB C' AC B') ∧
    TriangleABC A B C

theorem triangleSimilarity (A B C I T S B' C' : Point) :
  isTriangleABC A B C →
  similar (△ A B' C') (△ A B C) :=
by
  sorry

end triangleSimilarity_l427_427175


namespace simplify_expression_l427_427556

theorem simplify_expression (x : ℝ) : 
  (3 * x - 4) * (x + 8) - (x + 6) * (3 * x - 2) = 4 * x - 20 := 
by
  sorry

end simplify_expression_l427_427556


namespace range_of_x_l427_427137

noncomputable def quadratic_inequality (a x : ℝ) : Prop :=
  x^2 - a * x + 1 ≥ 1

theorem range_of_x :
  (∀ a : ℝ, a ∈ set.Icc (-3 : ℝ) (3 : ℝ) → quadratic_inequality a x)
  ↔ (x ≥ (3 + real.sqrt 5) / 2 ∨ x ≤ (-3 - real.sqrt 5) / 2) := by
  sorry

end range_of_x_l427_427137


namespace max_value_of_n_l427_427342

-- Define the main variables and conditions
noncomputable def max_n : ℕ := 
  let n_pairs := [(1, 72), (2, 36), (3, 24), (4, 18), (6, 12), (8, 9)] 
  max (n_pairs.map (λ p, 6 * p.2 + p.1))

-- Lean theorem statement for the equivalence
theorem max_value_of_n :
  max_n = 433 := by
  sorry

end max_value_of_n_l427_427342


namespace degree_measure_of_supplement_of_complement_of_35_degree_angle_l427_427195

def complement (α : ℝ) : ℝ := 90 - α
def supplement (β : ℝ) : ℝ := 180 - β

theorem degree_measure_of_supplement_of_complement_of_35_degree_angle : 
  supplement (complement 35) = 125 :=
by
  sorry

end degree_measure_of_supplement_of_complement_of_35_degree_angle_l427_427195


namespace possible_values_of_a_l427_427969

theorem possible_values_of_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ax^2 + 2*x + 1 = 0) ↔ (a ∈ set.Ioo (-∞) 0 ∪ set.Ioo 0 1) := 
sorry

end possible_values_of_a_l427_427969


namespace pencil_and_eraser_cost_l427_427088

theorem pencil_and_eraser_cost (p e : ℕ) :
  2 * p + e = 40 →
  p > e →
  e ≥ 3 →
  p + e = 22 :=
by
  sorry

end pencil_and_eraser_cost_l427_427088


namespace perfect_square_condition_l427_427768

-- Define the product of factorials from 1 to 2n
def factorial_product (n : ℕ) : ℕ :=
  (List.prod (List.map factorial (List.range (2 * n + 1))))

-- The main theorem statement
theorem perfect_square_condition (n : ℕ) : 
  (∃ k : ℕ, n = 4 * k * (k + 1) ∨ n = 2 * k^2 - 1) ↔
  (∃ m : ℕ, (factorial_product n) / ((n + 1)!) = m^2) := sorry

end perfect_square_condition_l427_427768


namespace cos_half_sum_of_acute_angles_l427_427841

theorem cos_half_sum_of_acute_angles (α β : ℝ) (hα: 0 < α ∧ α < π/2) (hβ: 0 < β ∧ β < π/2) (h: cos α ^ 2 + cos β ^ 2 = 1) : cos ((α + β) / 2) = (real.sqrt 2) / 2 :=
sorry

end cos_half_sum_of_acute_angles_l427_427841


namespace probability_of_drawing_at_least_three_white_balls_l427_427253

open_locale big_operators

noncomputable def factorial (n : ℕ) : ℕ := if h : n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ := if k > n then 0 else factorial n / (factorial k * factorial (n - k))

def total_ways_of_drawing : ℕ := binomial 15 6

def ways_to_draw_at_least_three_white_balls : ℕ :=
  binomial 8 3 * binomial 7 3 +
  binomial 8 4 * binomial 7 2 +
  binomial 8 5 * binomial 7 1 +
  binomial 8 6 * binomial 7 0

def probability : ℚ :=
  ways_to_draw_at_least_three_white_balls / total_ways_of_drawing

theorem probability_of_drawing_at_least_three_white_balls :
  probability = 550 / 715 := by
  sorry

end probability_of_drawing_at_least_three_white_balls_l427_427253


namespace max_norm_add_eq_l427_427798

variables (a b : ℝ)

-- Condition: The angle between vectors a and b is 30 degrees
def angle_30_degrees (a b : ℝ) : Prop := real.angle_between a b = π / 6

-- Condition: |a - b| = 2
def norm_sub_eq_2 (a b : ℝ) : Prop := abs (a - b) = 2

-- Statement: Prove the maximum value of |a + b| is 4 + 2√3
theorem max_norm_add_eq (a b : ℝ) (h1 : angle_30_degrees a b) (h2 : norm_sub_eq_2 a b) :
  abs (a + b) ≤ 4 + 2 * real.sqrt 3 :=
sorry

end max_norm_add_eq_l427_427798


namespace track_length_l427_427293

theorem track_length
  (x : ℕ)
  (run1_Brenda : x / 2 + 80 = a)
  (run2_Sally : x / 2 + 100 = b)
  (run1_ratio : 80 / (x / 2 - 80) = c)
  (run2_ratio : (x / 2 - 100) / (x / 2 + 100) = c)
  : x = 520 :=
by sorry

end track_length_l427_427293


namespace collinear_R_S_T_l427_427286

theorem collinear_R_S_T
    (circle : Type)
    (P : circle)
    (A B C D : circle)
    (E F : Type → Type)
    (angle : ∀ (x y z : circle), ℝ)   -- Placeholder for angles
    (quadrilateral_inscribed_in_circle : ∀ (A B C D : circle), Prop)   -- Placeholder for the condition of the quadrilateral
    (extensions_intersect : ∀ (A B C D : circle) (E F : Type → Type), Prop)   -- Placeholder for extensions intersections
    (diagonals_intersect_at : ∀ (A C B D T : circle), Prop)   -- Placeholder for diagonals intersections
    (P_on_circle : ∀ (P : circle), Prop)        -- Point P is on the circle
    (PE_PF_intersect_again : ∀ (P R S : circle) (E F : Type → Type), Prop)   -- PE and PF intersect the circle again at R and S
    (R S T : circle) :
    quadrilateral_inscribed_in_circle A B C D →
    extensions_intersect A B C D E F →
    P_on_circle P →
    PE_PF_intersect_again P R S E F →
    diagonals_intersect_at A C B D T →
    ∃ collinearity : ∀ (R S T : circle), Prop,
    collinearity R S T := 
by
  intro h1 h2 h3 h4 h5
  sorry

end collinear_R_S_T_l427_427286


namespace sum_log_equals_l427_427302

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

noncomputable def sum_term (k : ℝ) : ℝ :=
  (log_base 3 (1 + 1/k)) * (log_base k 3) * (log_base (k + 1) 3)

def sum_log_terms : ℝ :=
  (∑ k in Finset.range (100 - 2 + 1).map (λ x, x + 2), sum_term k)

theorem sum_log_equals :
  sum_log_terms = 1.0861 :=
  by sorry

end sum_log_equals_l427_427302


namespace OC_dot_OM_l427_427391

variables (A B C : EuclideanSpace (Fin 2) ℝ)
def circle_o := {p : EuclideanSpace (Fin 2) ℝ | ∥p∥ = 2}

def midpoint (a b : EuclideanSpace (Fin 2) ℝ) : EuclideanSpace (Fin 2) ℝ := (1 / 2 : ℝ) • (a + b)

-- Given conditions
axiom A_on_circle : A ∈ circle_o
axiom B_on_circle : B ∈ circle_o
axiom OC_eq : C = (5 / 2 : ℝ) • A - (sqrt 2 / 2 : ℝ) • B
axiom AB_length : dist A B = 2 * sqrt 2
def M : EuclideanSpace (Fin 2) ℝ := midpoint A B

-- Proof goal
theorem OC_dot_OM : inner C M = 5 - sqrt 2 := sorry

end OC_dot_OM_l427_427391


namespace grains_of_rice_calculation_l427_427170

def grains_of_rice_in_one_cup : ℕ :=
  sorry

theorem grains_of_rice_calculation :
  -- Conditions
  (half_cup_tbs: ℕ := 8) →
  (tbs_to_tsp: ℕ := 3) →
  (grains_per_tsp: ℕ := 10) →
  -- Calculation result
  grains_of_rice_in_one_cup = 480 :=
sorry

end grains_of_rice_calculation_l427_427170


namespace find_a_l427_427010

def setA (a : ℝ) : Set ℝ := { x | a * x - 1 = 0 }
def setB : Set ℝ := { x | x^2 - 3 * x + 2 = 0 }

theorem find_a (a : ℝ) : setA a ⊆ setB ↔ a = 0 ∨ a = 1 ∨ a = 1 / 2 :=
by
  sorry

end find_a_l427_427010


namespace sum_telescope_l427_427179

theorem sum_telescope :
  ∑ k in finset.range 50, (-1)^(k+1) * (k^3 + k^2 + k + 1) / k.factorial = 2601 / 50.factorial - 1 → 
  2652 = 2601 + 50 + 1 :=
by
  sorry

end sum_telescope_l427_427179


namespace number_of_real_values_of_p_l427_427730

theorem number_of_real_values_of_p :
  ∃ p_values : Finset ℝ, (∀ p ∈ p_values, ∀ x, x^2 - 2 * p * x + 3 * p = 0 → (x = p)) ∧ Finset.card p_values = 2 :=
by
  sorry

end number_of_real_values_of_p_l427_427730


namespace line_direction_vector_correct_l427_427665

theorem line_direction_vector_correct :
  ∃ (A B C : ℝ), (A = 2 ∧ B = -3 ∧ C = 1) ∧ 
  ∃ (v w : ℝ), (v = A ∧ w = B) :=
by
  sorry

end line_direction_vector_correct_l427_427665


namespace supplement_of_complement_of_35_degree_angle_l427_427213

theorem supplement_of_complement_of_35_degree_angle : 
  ∀ α β : ℝ,  α = 35 ∧ β = (90 - α) → (180 - β) = 125 :=
by
  intros α β h
  rcases h with ⟨h1, h2⟩
  rw h1 at h2
  rw h2
  linarith

end supplement_of_complement_of_35_degree_angle_l427_427213


namespace circum_sphere_surface_area_equilateral_tetrahedron_l427_427691

-- Define the side length of equilateral triangle base
def side_length_base : ℝ := 4

-- Define the length from S to A and B
def length_SA_SB : ℝ := sqrt 19

-- Define the length from S to C
def length_SC : ℝ := 3

-- The surface area of the circumscribed sphere of tetrahedron S-ABC
def surface_area_circumsphere (s_base : ℝ) (len_SA_SB : ℝ) (len_SC : ℝ) : ℝ :=
  if s_base = 4 ∧ len_SA_SB = sqrt 19 ∧ len_SC = 3 then
    (4 * Real.pi * (sqrt (61 / 11))^2)
  else
    0  -- default case, invalid input

-- Theorem to prove the surface area
theorem circum_sphere_surface_area_equilateral_tetrahedron :
  surface_area_circumsphere side_length_base length_SA_SB length_SC = (244 * Real.pi) / 11 :=
by
  -- Proof to be done
  sorry

end circum_sphere_surface_area_equilateral_tetrahedron_l427_427691


namespace alice_met_tweedledee_l427_427932

noncomputable def brother_statement (day : ℕ) : Prop :=
  sorry -- Define the exact logical structure of the statement "I am lying today, and my name is Tweedledum" here

theorem alice_met_tweedledee (day : ℕ) : brother_statement day → (∃ (b : String), b = "Tweedledee") :=
by
  sorry -- provide the proof here

end alice_met_tweedledee_l427_427932


namespace fill_tank_in_2_minutes_l427_427103

theorem fill_tank_in_2_minutes :
  (let R_A := 1 / 6 in
   let R_B := 2 * R_A in
   let R_combined := R_A + R_B in
   1 / R_combined = 2) :=
by
  sorry

end fill_tank_in_2_minutes_l427_427103


namespace triangle_equilateral_l427_427854

noncomputable def triangle_shape (a b c : ℝ) (A B C : ℝ) : Prop :=
  2 * a = b + c ∧ sin A ^ 2 = sin B * sin C → a = b ∧ b = c

theorem triangle_equilateral (a b c A B C : ℝ) :
  2 * a = b + c ∧ sin A ^ 2 = sin B * sin C → a = b ∧ b = c :=
by sorry

end triangle_equilateral_l427_427854


namespace problem1_simplified_problem2_solution_l427_427243

-- Definition for Problem 1
def problem1_expr : ℝ := 
  (Real.sqrt 6 - Real.sqrt (8 / 3)) * Real.sqrt 3 - (2 + Real.sqrt 3) * (2 - Real.sqrt 3)

-- Proof for Problem 1
theorem problem1_simplified : problem1_expr = Real.sqrt 2 - 1 := by
  sorry

-- Definitions for Problem 2
def eq1 (x y : ℝ) := 2 * x - 5 * y = 7
def eq2 (x y : ℝ) := 3 * x + 2 * y = 1

-- Proof for Problem 2
theorem problem2_solution : ∃ x y : ℝ, eq1 x y ∧ eq2 x y ∧ x = 1 ∧ y = -1 := by
  use 1, -1
  split; 
  { simp [eq1, eq2], norm_num }
  split; refl


end problem1_simplified_problem2_solution_l427_427243


namespace sum_of_abc_l427_427153

theorem sum_of_abc (a b c : ℕ) (h1 : (300 / 75 : ℝ) = 4) (h2 : (sqrt 4 : ℝ) = 2) 
    (h3 : 2 = a * sqrt b / c) (h4 : b = 1) (h5 : c = 1) : a + b + c = 4 :=
by
  have a_val : a = 2 := by
    sorry
  rw [a_val, h4, h5]
  exact rfl

end sum_of_abc_l427_427153


namespace solve_abs_linear_eq_l427_427024

theorem solve_abs_linear_eq (x : ℝ) : (|x - 1| + x - 1 = 0) ↔ (x ≤ 1) :=
sorry

end solve_abs_linear_eq_l427_427024


namespace flower_bed_area_l427_427055

theorem flower_bed_area (total_posts : ℕ) (corner_posts : ℕ) (spacing : ℕ) (long_side_multiplier : ℕ)
  (h1 : total_posts = 24)
  (h2 : corner_posts = 4)
  (h3 : spacing = 3)
  (h4 : long_side_multiplier = 3) :
  ∃ (area : ℕ), area = 144 := 
sorry

end flower_bed_area_l427_427055


namespace triangle_existence_alt_bis_med_l427_427178

-- Define the elements of the triangle
variables {A B C H D M O E: Type}
variables (H_alt: Line A H) (D_bisector: Line A D) (M_median: Line A M)

-- Define the triangle existence with the given conditions
theorem triangle_existence_alt_bis_med : 
  ∃ (A B C : Point), 
  (is_altitude A H B C) ∧ 
  (is_angle_bisector A D B C) ∧ 
  (is_median A M B C) :=
begin
  sorry
end

end triangle_existence_alt_bis_med_l427_427178


namespace largest_n_for_factored_quad_l427_427344

theorem largest_n_for_factored_quad (n : ℤ) (b d : ℤ) 
  (h1 : 6 * d + b = n) (h2 : b * d = 72) 
  (factorable : ∃ x : ℤ, (6 * x + b) * (x + d) = 6 * x ^ 2 + n * x + 72) : 
  n ≤ 433 :=
sorry

end largest_n_for_factored_quad_l427_427344


namespace part1_part2_l427_427412

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x)

theorem part1 (h : ∀ x, f x = 2 * Real.sin (2 * x - π / 6)) :
  (Real.T_visible_period f = π) ∧ 
  (∀ k : ℤ, is_increasing f (k * π - π / 6, k * π + π / 3)) :=
by sorry

variables (A B C a b c : ℝ)
variables (triangle_ABC : A + B + C = π)
variables (opp_sides : a * Real.sin A = b * Real.sin B ∧ b * Real.sin B = c * Real.sin C)
variables (h1 : f (A / 2) = 2)
variables (h2 : b = 1)
variables (h3 : c = 2)

theorem part2 (h : ∀ a b c, triangle_ABC ∧ opp_sides ∧ 
    A = 2 * atan (c/b) ∧ h1 ∧ h2 ∧ h3 → a = sqrt 7) : 
  a = sqrt 7 := 
by sorry

end part1_part2_l427_427412


namespace slower_speed_for_on_time_arrival_l427_427920

variable (distance : ℝ) (actual_speed : ℝ) (time_early : ℝ)

theorem slower_speed_for_on_time_arrival 
(h1 : distance = 20)
(h2 : actual_speed = 40)
(h3 : time_early = 1 / 15) :
  actual_speed - (600 / 17) = 4.71 :=
by 
  sorry

end slower_speed_for_on_time_arrival_l427_427920


namespace opposite_of_neg_two_l427_427975

theorem opposite_of_neg_two : -(-2) = 2 :=
by
  sorry

end opposite_of_neg_two_l427_427975


namespace geometric_chord_ratios_l427_427634

theorem geometric_chord_ratios
  (x y : ℝ)
  (circle_eq : x^2 + y^2 - 5*x = 0)
  (point_in_circle : ∃ (px py : ℝ), (px = 5/2) ∧ (py = 3/2) ∧ (px, py).in_circle circle_eq)
  (chords_form_geom_seq : ∃ (a_1 a_2 a_3 : ℝ), (a_1 < a_2 ∧ a_2 < a_3) ∧ (a_2^2 = a_1 * a_3) ∧ (∀ pi : ℝ, pi ∈ {a_1, a_2, a_3} → chord_passes_through_point pi (5/2, 3/2)))
  : ∃ q : ℝ, (q ∈ Icc (2/real.sqrt 5) (real.sqrt 5 / 2)) :=
sorry

end geometric_chord_ratios_l427_427634


namespace triangle_not_congruent_thm_l427_427852

noncomputable def triangle_not_congruent_given_conditions : Prop :=
  ∀ (A B C D E F : Type)
    (AB DE BC EF : ℝ)
    (A_angle D_angle : ℝ)
    (triangle_ABC : (A ≠ B ∧ B ≠ C ∧ C ≠ A))
    (triangle_DEF : (D ≠ E ∧ E ≠ F ∧ F ≠ D)),
  AB = DE →
  BC = EF →
  A_angle = D_angle →
  ¬(∃ (angle_B : ℝ) (triangle_ABC_is_congruent : true),
     triangle_ABC_is_congruent = (AB = DE) ∧
                                   (BC = EF) ∧
                                   (∃ (E_angle : ℝ), A_angle = D_angle ∧ angle_B = E_angle) )

-- Proof placeholder
theorem triangle_not_congruent_thm {A B C D E F : Type}
    {AB DE BC EF : ℝ}
    {A_angle D_angle : ℝ}
    (triangle_ABC_neq : (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A))
    (triangle_DEF_neq : (D ≠ E) ∧ (E ≠ F) ∧ (F ≠ D))
    (h1 : AB = DE)
    (h2 : BC = EF)
    (h3 : A_angle = D_angle) :
  ¬(∃ (B_angle : ℝ),
     true ∧ 
     (AB = DE) ∧
     (BC = EF) ∧
     (A_angle = D_angle ∧ A_angle = D_angle)) :=
by
  sorry

end triangle_not_congruent_thm_l427_427852


namespace perpendicular_k_parallel_k_l427_427017

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- Define the scalar multiple operations and vector operations
def smul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def add (v₁ v₂ : ℝ × ℝ) : ℝ × ℝ := (v₁.1 + v₂.1, v₂.2 + v₂.2)
def sub (v₁ v₂ : ℝ × ℝ) : ℝ × ℝ := (v₁.1 - v₂.1, v₂.2 - v₂.2)
def dot (v₁ v₂ : ℝ × ℝ) : ℝ := (v₁.1 * v₂.1 + v₁.2 * v₂.2)

-- Problem 1: If k*a + b is perpendicular to a - 3*b, then k = 19
theorem perpendicular_k (k : ℝ) :
  let vak := add (smul k a) b
  let amb := sub a (smul 3 b)
  dot vak amb = 0 → k = 19 := sorry

-- Problem 2: If k*a + b is parallel to a - 3*b, then k = -1/3 and they are in opposite directions
theorem parallel_k (k : ℝ) :
  let vak := add (smul k a) b
  let amb := sub a (smul 3 b)
  ∃ m : ℝ, vak = smul m amb ∧ m < 0 → k = -1/3 := sorry

end perpendicular_k_parallel_k_l427_427017


namespace eliot_votes_l427_427471

theorem eliot_votes (randy_votes shaun_votes eliot_votes : ℕ)
                    (h1 : randy_votes = 16)
                    (h2 : shaun_votes = 5 * randy_votes)
                    (h3 : eliot_votes = 2 * shaun_votes) :
                    eliot_votes = 160 :=
by {
  -- Proof will be conducted here
  sorry
}

end eliot_votes_l427_427471


namespace measure_of_minor_arc_LB_l427_427463

theorem measure_of_minor_arc_LB (C : Circle) (L B M : Point) (hLBM : inscribed_angle C L B M ∧ ∠LBM = 58) : 
  measure_minor_arc C L B = 244 :=
by
  sorry

end measure_of_minor_arc_LB_l427_427463


namespace initial_amount_spent_l427_427226

theorem initial_amount_spent (X : ℝ) 
    (h_bread : X - 3 ≥ 0) 
    (h_candy : X - 3 - 2 ≥ 0) 
    (h_turkey : X - 3 - 2 - (1/3) * (X - 3 - 2) ≥ 0) 
    (h_remaining : X - 3 - 2 - (1/3) * (X - 3 - 2) = 18) : X = 32 := 
sorry

end initial_amount_spent_l427_427226


namespace period_and_monotonicity_max_and_min_values_l427_427000

noncomputable def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * cos x - 2 * sin x ^ 2

theorem period_and_monotonicity :
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ k : ℤ, ∀ (x : ℝ), (k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6) → (∀ y, k * π - π / 3 ≤ y ∧ y ≤ x → f y ≤ f x)) :=
by
  sorry

theorem max_and_min_values :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ π / 4 ∧ f x = 0) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ π / 4 ∧ f x = 1) :=
by
  sorry

end period_and_monotonicity_max_and_min_values_l427_427000


namespace kevin_total_distance_l427_427061

theorem kevin_total_distance :
  let d1 := 10 * 0.5,
      d2 := 20 * 0.5,
      d3 := 8 * 0.25 in
  d1 + d2 + d3 = 17 := by
  sorry

end kevin_total_distance_l427_427061


namespace find_g_inv_f_15_l427_427835

variable (f g : ℝ → ℝ)
variable (h : f⁻¹ ∘ g = λ x, x^4 - 1)
variable (g_inv_exists : ∃ g_inv : ℝ → ℝ, g ∘ g_inv = id ∧ g_inv ∘ g = id)

theorem find_g_inv_f_15 : g⁻¹ (f 15) = 2 :=
by
  sorry

end find_g_inv_f_15_l427_427835


namespace tangent_line_circle_l427_427445

theorem tangent_line_circle (m : ℝ) :
  (∀ x y : ℝ, x - 2*y + m = 0 ↔ (x^2 + y^2 - 4*x + 6*y + 8 = 0)) →
  m = -3 ∨ m = -13 :=
sorry

end tangent_line_circle_l427_427445


namespace train_pass_man_in_time_l427_427693

-- Definitions for the given conditions
def length_of_train : ℝ := 200 -- in meters
def speed_of_train : ℝ := 80 * 1000 / 3600 -- converting 80 km/hr to m/s
def speed_of_man : ℝ := 10 * 1000 / 3600 -- converting 10 km/hr to m/s

-- Statement to be proved: The time for the train to pass the man
theorem train_pass_man_in_time : 
    let relative_speed := (speed_of_train + speed_of_man) in
    let time := length_of_train / relative_speed in
    time = 8 := 
by 
  -- leave the proof for the user
  sorry

end train_pass_man_in_time_l427_427693


namespace range_of_n_l427_427574

noncomputable def hyperbola_foci_range (m n : ℝ) : Prop :=
  (m^2 = 1) ∧
  (4 = (m^2 + n) + (3 * m^2 - n)) ∧
  ((m^2 + n) * (3 * m^2 - n) > 0) ∧
  (-1 < n ∧ n < 3)

theorem range_of_n (m n : ℝ) : hyperbola_foci_range m n → (-1 < n ∧ n < 3) :=
by 
  intro h
  cases' h with h1 h2
  cases' h2 with h2 h3
  exact h3

end range_of_n_l427_427574


namespace part1_c_eqn_cartesian_part1_l_eqn_rectangular_part2_m_range_l427_427456

-- Definitions for parametric equations of curve C
def C_parametric_eqns (α : ℝ) : ℝ × ℝ :=
  (1 + 3 * Real.cos α, 3 + 3 * Real.sin α)

-- Definition for polar coordinate equation of line l
def l_polar_eqn (ρ θ m : ℝ) : Prop :=
  ρ * Real.cos θ + ρ * Real.sin θ = m

-- Cartesian equation of curve C
def C_cartesian_eqn (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 9

-- Rectangular coordinate equation of line l
def l_rectangular_eqn (x y m : ℝ) : Prop :=
  x + y - m = 0

-- Range of m for curve C to intersect line l at two points
def m_range (m : ℝ) : Prop :=
  4 - 3 * Real.sqrt 2 < m ∧ m < 4 + 3 * Real.sqrt 2

-- Theorem statements
theorem part1_c_eqn_cartesian (α : ℝ) : ∀ x y, (x, y) = C_parametric_eqns α → C_cartesian_eqn x y := by
  sorry

theorem part1_l_eqn_rectangular (ρ θ m x y : ℝ) (h : l_polar_eqn ρ θ m) : (x, y) = (ρ * Real.cos θ, ρ * Real.sin θ) → l_rectangular_eqn x y m := by
  sorry

theorem part2_m_range (m : ℝ) : ∀ ρ θ, ∃ x y, (x, y) = C_parametric_eqns (Real.atan 2) → l_polar_eqn ρ θ m → m_range m := by
  sorry

end part1_c_eqn_cartesian_part1_l_eqn_rectangular_part2_m_range_l427_427456


namespace remainder_when_13_plus_y_divided_by_31_l427_427524

theorem remainder_when_13_plus_y_divided_by_31
  (y : ℕ)
  (hy : 7 * y % 31 = 1) :
  (13 + y) % 31 = 22 :=
sorry

end remainder_when_13_plus_y_divided_by_31_l427_427524


namespace shortest_distance_to_circle_l427_427585

variable (A O T : Type)
variable (r d : ℝ)
variable [MetricSpace A]
variable [MetricSpace O]
variable [MetricSpace T]

open Real

theorem shortest_distance_to_circle (h : d = (4 / 3) * r) : 
  OA = (5 / 3) * r → shortest_dist = (2 / 3) * r :=
by
  sorry

end shortest_distance_to_circle_l427_427585


namespace infinite_nested_radical_solution_l427_427869

theorem infinite_nested_radical_solution (x : ℝ) (h : x = Real.sqrt (4 + 3 * x)) : x = 4 := 
by 
  sorry

end infinite_nested_radical_solution_l427_427869


namespace sum_of_all_possible_perimeters_of_triangle_ACD_l427_427104

-- Conditions as Lean 4 definitions
structure point (α : Type*) := (coord : α × α)

def A : point ℝ := ⟨(0, 0)⟩
def B : point ℝ := ⟨(12, 0)⟩
def C : point ℝ := ⟨(36, 0)⟩

-- Define distances as given in conditions
def distance (p1 p2 : point ℝ) : ℝ :=
  real.sqrt ((p1.coord.1 - p2.coord.1)^2 + (p1.coord.2 - p2.coord.2)^2)

lemma distance_AB : distance A B = 12 := by simp [distance, A, B]; norm_num
lemma distance_BC : distance B C = 24 := by simp [distance, B, C]; norm_num

-- Prove given conditions and question
theorem sum_of_all_possible_perimeters_of_triangle_ACD : 
  (∑ (x y : ℕ), (distance A B + distance B C + 2 * x) = 150) :=
sorry

end sum_of_all_possible_perimeters_of_triangle_ACD_l427_427104


namespace mike_ride_distance_l427_427643

-- Definitions from conditions
def mike_cost (m : ℕ) : ℝ := 2.50 + 0.25 * m
def annie_cost : ℝ := 2.50 + 5.00 + 0.25 * 16

-- Theorem to prove
theorem mike_ride_distance (m : ℕ) (h : mike_cost m = annie_cost) : m = 36 := by
  sorry

end mike_ride_distance_l427_427643


namespace part_a_l427_427076

variable (f : ℝ → ℝ)

-- Given:
-- 1. f is continuous on [1, ∞)
-- 2. ∀ a > 0, ∃ x ∈ [1, ∞) such that f(x) = a * x

noncomputable def continuous_function := 
  continuous_on f (set.Ici 1) ∧
  (∀ a > 0, ∃ x ≥ 1, f x = a * x)

-- Problem (a): Prove that ∀ a > 0, f(x) = a * x has infinitely many solutions
theorem part_a (h : continuous_function f) (a : ℝ) (ha : 0 < a) :
  ∃∞ x, f x = a * x := 
sorry

-- Problem (b): Provide an example of such a function
example : 
  ∃ f : ℝ → ℝ, 
    continuous_on f (set.Ici 1) ∧
    strict_mono_on f (set.Ici 1) ∧
    (∀ a > 0, ∃ x ≥ 1, f x = a * x) := 
sorry

end part_a_l427_427076


namespace no_nonnegative_integers_for_absolute_difference_l427_427052

theorem no_nonnegative_integers_for_absolute_difference :
  ∀ (a b : ℕ), |3^a - 2^b| = 41 → false :=
by
  sorry

end no_nonnegative_integers_for_absolute_difference_l427_427052


namespace selection_ways_l427_427265

-- Define the binomial coefficient
def binom : ℕ → ℕ → ℕ
| n, 0 := 1
| 0, k := 0
| n+1, k+1 := binom n k + binom (n+1) k

-- Define the number of boys and girls
def num_boys : ℕ := 10
def num_girls : ℕ := 12
def select_boys : ℕ := 4
def select_girls : ℕ := 4

-- Define the proof problem statement
theorem selection_ways : binom num_boys select_boys * binom num_girls select_girls = 103950 := by
  sorry

end selection_ways_l427_427265


namespace supplement_of_complement_of_35_degree_angle_l427_427215

theorem supplement_of_complement_of_35_degree_angle : 
  ∀ α β : ℝ,  α = 35 ∧ β = (90 - α) → (180 - β) = 125 :=
by
  intros α β h
  rcases h with ⟨h1, h2⟩
  rw h1 at h2
  rw h2
  linarith

end supplement_of_complement_of_35_degree_angle_l427_427215


namespace simplify_expression_l427_427552

-- We define the given expressions and state the theorem.
variable (x : ℝ)

theorem simplify_expression : (3 * x - 4) * (x + 8) - (x + 6) * (3 * x - 2) = 4 * x - 20 := by
  -- Proof goes here
  sorry

end simplify_expression_l427_427552


namespace sum_of_abc_l427_427151

theorem sum_of_abc (a b c : ℕ) (h1 : (300 / 75 : ℝ) = 4) (h2 : (sqrt 4 : ℝ) = 2) 
    (h3 : 2 = a * sqrt b / c) (h4 : b = 1) (h5 : c = 1) : a + b + c = 4 :=
by
  have a_val : a = 2 := by
    sorry
  rw [a_val, h4, h5]
  exact rfl

end sum_of_abc_l427_427151


namespace sum_of_distinct_real_numbers_l427_427500

theorem sum_of_distinct_real_numbers (p q r s : ℝ) (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : ∀ x : ℝ, x^2 - 12 * p * x - 13 * q = 0 -> (x = r ∨ x = s)) 
  (h2 : ∀ x : ℝ, x^2 - 12 * r * x - 13 * s = 0 -> (x = p ∨ x = q)) :
  p + q + r + s = 2028 :=
begin
  sorry
end

end sum_of_distinct_real_numbers_l427_427500


namespace exists_xy_in_S_l427_427483

open Set

theorem exists_xy_in_S
  (S : Set ℝ) (hS : S.card = 10) (h_distinct : ∀ ⦃x ⦄, x ∈ S → ∀ ⦃y ⦄, y ∈ S → x = y → false) :
  ∃ (x y : ℝ), x ∈ S ∧ y ∈ S ∧ 0 < x - y ∧ x ≠ y ∧ x - y < (1 + x) * (1 + y) / 9 :=  
begin
  sorry
end

end exists_xy_in_S_l427_427483


namespace max_subsets_of_A_inter_B_l427_427821

def setA : Set (ℝ × ℝ) := {p | ∃ x ∈ Ioo 0 (2 * Real.pi), p = (x, Real.sin x)}
def setB (a : ℝ) : Set (ℝ × ℝ) := {p | ∃ x : ℝ, p = (x, a)}

theorem max_subsets_of_A_inter_B (a : ℝ) : 
  ∃ (n : ℕ), (∀ S : Set (ℝ × ℝ), S ⊆ (setA ∩ setB a) → n = S.powerset.card) ∧ n = 4 :=
sorry

end max_subsets_of_A_inter_B_l427_427821


namespace john_writing_time_l427_427057

def pages_per_day : ℕ := 20
def pages_per_book : ℕ := 400
def number_of_books : ℕ := 3

theorem john_writing_time : (pages_per_book / pages_per_day) * number_of_books = 60 :=
by
  -- The proof should be placed here.
  sorry

end john_writing_time_l427_427057


namespace round_2_65_to_nearest_tenth_l427_427942

def nearest_tenth(x: ℝ) : ℝ :=
  ((x * 10).round) / 10

theorem round_2_65_to_nearest_tenth : nearest_tenth 2.65 = 2.7 :=
by
  sorry

end round_2_65_to_nearest_tenth_l427_427942


namespace divisibility_by_n_l427_427551

theorem divisibility_by_n (n : ℕ) (h : n > 0) : ∃ (a b : ℤ), n ∣ (4 * a^2 + 9 * b^2 - 1) :=
  sorry

end divisibility_by_n_l427_427551


namespace solve_for_A_l427_427068

variable (A B : ℝ) (f g : ℝ → ℝ)

def f_def := f = λ x => A * x^2 - 3 * B^3
def g_def := g = λ x => 2 * B * x
def B_nonzero := B ≠ 0
def fg2_zero := f (g 2) = 0

theorem solve_for_A : f_def A B f → g_def B g → B_nonzero B → fg2_zero A B f g → A = (3 * B) / 16 := sorry

end solve_for_A_l427_427068


namespace ratio_of_squares_l427_427154

theorem ratio_of_squares (a b c : ℕ) (h : (a = 2) ∧ (b = 1) ∧ (c = 1)) :
  (∑ i in {a, b, c}, i) = 4 :=
by {
  cases h with h1 h2,
  cases h2 with h2 h3,
  rw [h1, h2, h3],
  simp,
}

end ratio_of_squares_l427_427154


namespace distance_of_given_parallel_lines_l427_427131

open Real

def distance_between_parallel_lines (a b c1 c2 : ℝ) := abs (c1 - c2) / sqrt (a^2 + b^2)

theorem distance_of_given_parallel_lines:
  distance_between_parallel_lines 3 4 (-6) 3 = 1.5 :=
by
  sorry

end distance_of_given_parallel_lines_l427_427131


namespace opposite_of_neg_two_is_two_l427_427987

theorem opposite_of_neg_two_is_two : -(-2) = 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l427_427987


namespace vibrations_proof_l427_427926

-- Define the conditions
def vibrations_lowest : ℕ := 1600
def increase_percentage : ℕ := 60
def use_time_minutes : ℕ := 5

-- Convert percentage to a multiplier
def percentage_to_multiplier (p : ℕ) : ℤ := (p : ℤ) / 100

-- Calculate the vibrations per second at the highest setting
def vibrations_highest := vibrations_lowest + (vibrations_lowest * percentage_to_multiplier increase_percentage).toNat

-- Convert time from minutes to seconds
def use_time_seconds := use_time_minutes * 60

-- Calculate the total vibrations Matt experiences
noncomputable def total_vibrations : ℕ := vibrations_highest * use_time_seconds

-- State the theorem
theorem vibrations_proof : total_vibrations = 768000 := 
by
  sorry

end vibrations_proof_l427_427926


namespace smallest_possible_n_l427_427632

theorem smallest_possible_n (n : ℕ) (h1 : 0 < n) (h2 : 0 < 60) 
  (h3 : (Nat.lcm 60 n) / (Nat.gcd 60 n) = 24) : n = 20 :=
by sorry

end smallest_possible_n_l427_427632


namespace factorial_sequence_perfect_square_l427_427764

-- Definitions based on the conditions
def is_perf_square (x : ℕ) : Prop := ∃ (k : ℕ), x = k * k

def factorial (n : ℕ) : ℕ := Nat.recOn n 1 (λ n fac_n, (n + 1) * fac_n)

def factorial_seq_prod (n : ℕ) : ℕ :=
  Nat.recOn n 1 (λ n prodN, factorial (2 * n) * prodN)

-- Main statement
theorem factorial_sequence_perfect_square (n : ℕ) :
  is_perf_square (factorial_seq_prod n / factorial (n + 1)) ↔
  ∃ k : ℕ, n = 4 * k * (k + 1) ∨ n = 2 * k ^ 2 - 1 := by
  sorry

end factorial_sequence_perfect_square_l427_427764


namespace ratio_of_works_l427_427224

noncomputable def temp_proportional_square_volume 
  (α : ℝ) (V : ℝ) : ℝ :=
  α * V^2

noncomputable def ideal_gas_law
  (p : ℝ) (V : ℝ) (n : ℝ) (R : ℝ) (T : ℝ) : Prop :=
  p * V = n * R * T

noncomputable def work
  (V1 V2 : ℝ) (p : ℝ) : ℝ :=
  ∫ x in V1..V2, p * x

theorem ratio_of_works
  (p10 p20 V10 V20 V11 V21 n R α : ℝ)
  (h1 : p20 = 2 * p10)
  (h2 : V20 = V10)
  (T1 : ℝ := temp_proportional_square_volume α V10)
  (T2 : ℝ := temp_proportional_square_volume α V10)
  (h3 : ideal_gas_law p10 V10 n R T1)
  (h4 : ideal_gas_law p20 V20 n R T2) :
  let A1 := work V10 V11 (p10 / V10) in
  let A2 := work V20 V21 (p20 / V20) in
  A2 / A1 = 2 :=
 by 
    sorry

end ratio_of_works_l427_427224


namespace a1_lt_a2_iff_seq_inc_l427_427882

noncomputable def a_n (n : ℕ) (λ : ℝ) : ℝ := n^2 + λ * n

theorem a1_lt_a2_iff_seq_inc {λ : ℝ} :
  (a_n 1 λ < a_n 2 λ) ↔ (∀ n : ℕ, 0 < n → a_n (n + 1) λ > a_n n λ) :=
by
  sorry

end a1_lt_a2_iff_seq_inc_l427_427882


namespace book_cost_price_l427_427654

theorem book_cost_price
  (C : ℝ) (P : ℝ) (SP : ℝ)
  (h1 : SP = 1.25 * C)
  (h2 : 0.95 * P = SP)
  (h3 : SP = 62.5) : 
  C = 50 := 
by
  sorry

end book_cost_price_l427_427654


namespace opposite_of_neg_two_l427_427994

theorem opposite_of_neg_two : -(-2) = 2 := 
by 
  sorry

end opposite_of_neg_two_l427_427994


namespace units_digit_R_6789_l427_427009

noncomputable theory
open Locale BigOperators

def a : ℝ := 4 + Real.sqrt 15
def b : ℝ := 4 - Real.sqrt 15

def R (n : ℕ) : ℝ := (1 / 2) * (a^n + b^n)

theorem units_digit_R_6789 : 
  (R 6789) % 10 = 4 :=
sorry

end units_digit_R_6789_l427_427009


namespace calc_Ca_concentration_l427_427562

/-- Given Ksp(CaCO3) = 4.96×10^{-9}. When 0.40 mol·L^{-1} Na2CO3 solution and 
0.20 mol·L^{-1} CaCl2 solution are mixed in equal volumes, 
the concentration of Ca2+ in the solution after mixing -/
theorem calc_Ca_concentration :
  let Ksp_CaCO3 := 4.96 * 10^(-9) in 
  let conc_Na2CO3 := 0.40 in
  let conc_CaCl2 := 0.20 in
  let mixed_conc_CO3 := conc_Na2CO3 / 2 in
  let mixed_conc_Ca := conc_CaCl2 / 2 in
  let resulting_conc_Ca := Ksp_CaCO3 / mixed_conc_CO3 
  in resulting_conc_Ca = 4.96 * 10^(-8) := 
sorry

end calc_Ca_concentration_l427_427562


namespace taxi_total_cost_l427_427174

theorem taxi_total_cost (U L T : ℝ)
  (h1 : U = L + 3)
  (h2 : L = T + 4)
  (h3 : U = 22) 
  (detour_rate tip_rate : ℝ)
  (h4 : detour_rate = 0.15)
  (h5 : tip_rate = 0.20) :
  let detour_cost := T * detour_rate,
      cost_after_detour := T + detour_cost,
      tip := T * tip_rate,
      total_cost := cost_after_detour + tip in
  total_cost = 20.25 :=
by
  -- The proof goes here. We'll just use 'sorry' for now as only the statement is required.
  sorry

end taxi_total_cost_l427_427174


namespace part1_part2_l427_427071

-- Definitions and conditions
def prop_p (a : ℝ) : Prop := 
  let Δ := -4 * a^2 + 4 * a + 24 
  Δ ≥ 0

def neg_prop_p (a : ℝ) : Prop := ¬ prop_p a

def prop_q (m a : ℝ) : Prop := 
  (m - 1 ≤ a ∧ a ≤ m + 3)

-- Part 1 theorem statement
theorem part1 (a : ℝ) : neg_prop_p a → (a < -2 ∨ a > 3) :=
by sorry

-- Part 2 theorem statement
theorem part2 (m : ℝ) : 
  (∀ a : ℝ, prop_q m a → prop_p a) ∧ (∃ a : ℝ, prop_p a ∧ ¬ prop_q m a) → (-1 ≤ m ∧ m < 0) :=
by sorry

end part1_part2_l427_427071


namespace altitudes_segments_equal_imply_equilateral_l427_427618

noncomputable def segments_decompose_on_incircle {A B C : Type*} [∀ S, MetricSpace S] 
  (triangle : Triangle A B C) : Prop :=
  let incircle := triangle.incircle in
  let D := incircle.touch B C in
  let E := incircle.touch C A in
  let F := incircle.touch A B in
  let A_h := triangle.altitude A in
  let B_h := triangle.altitude B in
  let C_h := triangle.altitude C in
  A_h.Dist D = B_h.Dist E ∧ B_h.Dist E = C_h.Dist F

noncomputable def is_equilateral {A B C : Type*} [∀ S, MetricSpace S] 
  (triangle : Triangle A B C) : Prop :=
  triangle.a = triangle.b ∧ triangle.b = triangle.c

theorem altitudes_segments_equal_imply_equilateral 
  {A B C : Type*} [∀ S, MetricSpace S] (triangle : Triangle A B C) :
  segments_decompose_on_incircle triangle → is_equilateral triangle :=
by
  -- Proof omitted
  sorry

end altitudes_segments_equal_imply_equilateral_l427_427618


namespace distance_from_P_to_l_l427_427958

variables (A P m : EuclideanSpace ℝ (Fin 3))
variables (l : AffineSubspace ℝ (EuclideanSpace ℝ (Fin 3)))

-- Define the given entities
def A := ![1, 1, 1]
def P := ![1, -1, -1]
def m := ![1, 0, -1]
def l := AffineSubspace.mk' A (Span ℝ {m})

-- Define the distance function
def distance_point_to_line (P : EuclideanSpace ℝ (Fin 3)) (l : AffineSubspace ℝ (EuclideanSpace ℝ (Fin 3))) : ℝ :=
  infi (λ (Q : EuclideanSpace ℝ (Fin 3)) (hq : Q ∈ l), dist P Q)

-- The theorem statement
theorem distance_from_P_to_l : distance_point_to_line P l = Real.sqrt 6 :=
by
  sorry

end distance_from_P_to_l_l427_427958


namespace total_simple_interest_l427_427689

noncomputable def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100

theorem total_simple_interest : simple_interest 2500 10 4 = 1000 := 
by
  sorry

end total_simple_interest_l427_427689


namespace ratio_proof_l427_427023

variable (a b c d : ℚ)

theorem ratio_proof 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 7) :
  d / a = 2 / 35 := by
  sorry

end ratio_proof_l427_427023


namespace sum_of_solutions_eq_zero_l427_427628

theorem sum_of_solutions_eq_zero : 
  (∑ x in {x : ℝ | 3 * x / 15 = 6 / x}.to_finset, x) = 0 :=
by sorry

end sum_of_solutions_eq_zero_l427_427628


namespace composite_divisor_le_sqrt_l427_427935

noncomputable def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d, d > 1 ∧ d < n ∧ n % d = 0

theorem composite_divisor_le_sqrt (n : ℕ) (h : is_composite n) :
  ∃ d, 1 < d ∧ d ≤ nat.sqrt n ∧ n % d = 0 :=
sorry

end composite_divisor_le_sqrt_l427_427935


namespace intersection_points_property_l427_427318

noncomputable def parametric_line (t : ℝ) := (1 + 1/2 * t, (sqrt 3)/2 * t)

def polar_equation (ρ θ : ℝ) : Prop := ρ^2 * (1 + 2 * (sin θ)^2) = 3

def cartesian_curve_C2 (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

def general_eq_C1 (x y : ℝ) : Prop := sqrt 3 * x - y - sqrt 3 = 0

theorem intersection_points_property : 
  (∀ t1 t2 : ℝ, 
    parametric_line t1 = parametric_line t2 → t1 = t2) ∧
  (∀ x y, general_eq_C1 x y ↔ (∃ t : ℝ, x = 1 + 1/2 * t ∧ y = (sqrt 3)/2 * t)) ∧
  (∀ x y, cartesian_curve_C2 x y ↔ (x*x / 3 + y*y = 1)) →
  |distance (1, 0) (parametric_line t1) - distance (1, 0) (parametric_line t2)| = 2 / 5 := 
sorry

end intersection_points_property_l427_427318


namespace toothpick_staircase_steps_270_l427_427943

theorem toothpick_staircase_steps_270 (n : ℕ) :
  (3 * n * (n + 1)) / 2 = 270 → n = 12 :=
by
  intro h
  have h_eq : 3 * n * (n + 1) = 540 := by
    linarith
  have quad_eq : n * (n + 1) = 180 := by
    linarith
  have eq1 : n * (n + 1) = 180 := by
    exact quad_eq
  have eq2 : n^2 + n - 180 = 0 := by
    ring_nf
    rw [eq1]
  sorry  -- Completing the proof for the quadratic equation

end toothpick_staircase_steps_270_l427_427943


namespace supplement_of_complement_is_125_l427_427210

-- Definition of the initial angle
def initial_angle : ℝ := 35

-- Definition of the complement of the angle
def complement_angle (θ : ℝ) : ℝ := 90 - θ

-- Definition of the supplement of an angle
def supplement_angle (θ : ℝ) : ℝ := 180 - θ

-- Main theorem statement
theorem supplement_of_complement_is_125 : 
  supplement_angle (complement_angle initial_angle) = 125 := 
by
  sorry

end supplement_of_complement_is_125_l427_427210


namespace calc_expr_value_x_inv_sum_l427_427648

-- Definition for part (1) of the question
def calc_expr := (- (27 / 8))^(- (2 / 3)) 
                + (Real.logBase 8 27) / (Real.logBase 2 3) 
                + (Real.sqrt 2 - Real.sqrt 3)^0 
                - (Real.logBase 3 1) 
                + 2 * Real.log 5 
                + Real.log 4 
                - 5^(Real.logBase 5 2)

theorem calc_expr_value : calc_expr = 22 / 9 := 
by sorry

-- Definition for part (2) of the question
variable (x : ℝ)
hypothesis : x^(1/2) + x^(-1/2) = 3

theorem x_inv_sum : x + x^(-1) = 7 := 
by sorry

end calc_expr_value_x_inv_sum_l427_427648


namespace supplement_of_complement_of_35_degree_angle_l427_427214

theorem supplement_of_complement_of_35_degree_angle : 
  ∀ α β : ℝ,  α = 35 ∧ β = (90 - α) → (180 - β) = 125 :=
by
  intros α β h
  rcases h with ⟨h1, h2⟩
  rw h1 at h2
  rw h2
  linarith

end supplement_of_complement_of_35_degree_angle_l427_427214


namespace find_pairs_l427_427736

def is_integer_solution (p q : ℝ) : Prop :=
  ∃ (a b : ℝ), (a * b = q ∧ a + b = p ∧ a * b ∈ ℤ ∧ a + b ∈ ℤ)

theorem find_pairs (p q : ℝ) (h : p + q = 1998) :
  ((p = 1998 ∧ q = 0) ∨ (p = -2002 ∧ q = 4000)) ∧ is_integer_solution p q :=
by sorry

end find_pairs_l427_427736


namespace complex_division_l427_427954

theorem complex_division :
  (1 + 2 * Complex.i) / (2 - Complex.i) = Complex.i :=
by
  sorry

end complex_division_l427_427954


namespace interest_rate_difference_correct_l427_427690

noncomputable def interest_rate_difference (P r R T : ℝ) :=
  let I := P * r * T
  let I' := P * R * T
  (I' - I) = 140

theorem interest_rate_difference_correct:
  ∀ (P r R T : ℝ),
  P = 1000 ∧ T = 7 ∧ interest_rate_difference P r R T →
  (R - r) = 0.02 :=
by
  intros P r R T h
  sorry

end interest_rate_difference_correct_l427_427690


namespace smaller_cube_side_length_l427_427664

theorem smaller_cube_side_length : 
  ∀ (R : ℝ) (r : ℝ),
    (∀ (a : ℝ), a = 2 → R = a * Real.sqrt 3 / 2) ∧
    (∀ (d : ℝ), d = 2 * Real.sqrt 2 ∧ r * Real.sqrt 2 = R) ∧
    (∀ (s : ℝ), s = 2/3) :=
begin
  sorry
end

end smaller_cube_side_length_l427_427664


namespace max_contribution_l427_427236

theorem max_contribution (total_contribution : ℝ) (num_people : ℕ) (min_contribution_each : ℝ) (h1 : total_contribution = 45.00) (h2 : num_people = 25) (h3 : min_contribution_each = 1.00) : 
  ∃ max_cont : ℝ, max_cont = 21.00 :=
by
  sorry

end max_contribution_l427_427236


namespace range_of_a_l427_427697

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a <= x ∧ x < y ∧ y <= b → f y <= f x

theorem range_of_a (f : ℝ → ℝ) :
  odd_function f →
  decreasing_on_interval f (-1) 1 →
  (∀ a : ℝ, 0 < a ∧ a < 1 → f (1 - a) + f (2 * a - 1) < 0) →
  (∀ a : ℝ, 0 < a ∧ a < 1) :=
sorry

end range_of_a_l427_427697


namespace smallest_number_conditions_l427_427624

theorem smallest_number_conditions :
  ∃ n : ℕ, n > 0 ∧ (50 ∣ n) ∧ (75 ∣ n) ∧ ¬(18 ∣ n) ∧ (7 ∣ n) ∧ 
  ∀ m : ℕ, m > 0 ∧ (50 ∣ m) ∧ (75 ∣ m) ∧ ¬(18 ∣ m) ∧ (7 ∣ m) → m ≥ n :=
begin
  use 1050,
  split,
  { exact nat.pos_of_ne_zero (nat.ne_of_lt (lt_add_one 1050)) },
  split,
  { exact dvd.trans (dvd_mul_right 75 7) (dvd_mul_right 2 1050) },
  split,
  { exact dvd_mul_right 75 14 },
  split,
  { intro h, cases h with k hk,
    replace hk : 1050 = 18 * k := hk,
    have : 150 = 3 * (50 * 1),
    rw [hk, mul_comm 3 50, mul_comm 7 k] at this,
    exact nat.succ_ne_zero (50 * k) (nat.eq_zero_of_mul_eq_zero_left this) },
  split,
  { exact dvd_mul_right 1 1050 },
  intro m,
  intros conditions,
  sorry -- Proof skipped as instructed,
end

end smallest_number_conditions_l427_427624


namespace overall_gain_in_whole_transaction_l427_427267

def purchase_price_1 : ℝ := 675958
def sale_price_1 : ℝ := 725000
def purchase_price_2 : ℝ := 848592
def sale_price_2 : ℝ := 921500
def purchase_price_3 : ℝ := 940600
def sale_price_3 : ℝ := 982000
def purchase_tax_rate : ℝ := 0.02
def sale_tax_rate : ℝ := 0.01

theorem overall_gain_in_whole_transaction :
  let purchase_tax amount := amount * purchase_tax_rate in
  let sale_tax amount := amount * sale_tax_rate in
  let total_cost amount := amount + purchase_tax amount in
  let total_revenue amount := amount - sale_tax amount in
  let gain purchase sale := total_revenue sale - total_cost purchase in
  let gain_1 := gain purchase_price_1 sale_price_1 in
  let gain_2 := gain purchase_price_2 sale_price_2 in
  let gain_3 := gain purchase_price_3 sale_price_3 in
  (gain_1 + gain_2 + gain_3) = 87762 :=
by
  sorry

end overall_gain_in_whole_transaction_l427_427267


namespace satisfactory_orders_l427_427430

-- Definitions and conditions
variables (C1 C2 C3 D1 D2 D3 : Type)

-- Predicate to represent the order constraint: D_i must be solved before C_i
def correct_order (D C : Type) : Prop := true -- this is abstract; specifics needed for actual order checking

-- Statement of the theorem
theorem satisfactory_orders (C1 C2 C3 D1 D2 D3 : Type)
  (h1 : correct_order D1 C1) (h2 : correct_order D2 C2) (h3 : correct_order D3 C3) : 
  -- Number of satisfactory orders
  nat.factorial 6 / 2^3 = 90 :=
by
  sorry

end satisfactory_orders_l427_427430


namespace real_number_m_l427_427489

theorem real_number_m (m : ℤ) (M := {4, 5, -3*m + (m-3)*complex.I}) (N := {-9, 3}) (h : ∃ x : ℂ, x ∈ M ∧ x ∈ N) : m = 3 :=
begin
  sorry
end

end real_number_m_l427_427489


namespace product_of_ages_l427_427172

-- Definitions based on the conditions
variables {R T K J : ℕ}

-- Conditions from the problem
def condition1 : Prop := T = R - 6
def condition2 : Prop := T = K + 4
def condition3 : Prop := R = J + 8
def condition4 : Prop := R = K + 4
def condition5 : Prop := R + 2 = 3 * (J + 2)
def condition6 : Prop := T + 2 = 2 * (K + 2)

-- The main proof problem
theorem product_of_ages (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) (h5 : condition5) (h6 : condition6) :
  (R + 2) * (K + 2) * (T + 2) = 576 :=
sorry

end product_of_ages_l427_427172


namespace sum_of_distinct_real_numbers_l427_427501

theorem sum_of_distinct_real_numbers (p q r s : ℝ) (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : ∀ x : ℝ, x^2 - 12 * p * x - 13 * q = 0 -> (x = r ∨ x = s)) 
  (h2 : ∀ x : ℝ, x^2 - 12 * r * x - 13 * s = 0 -> (x = p ∨ x = q)) :
  p + q + r + s = 2028 :=
begin
  sorry
end

end sum_of_distinct_real_numbers_l427_427501


namespace pants_needed_l427_427922

def skirt_cost : ℕ := 20
def blouse_cost : ℕ := 15
def pants_cost : ℕ := 30
def pants_discounted_cost : ℕ := 15
def skirts_needed : ℕ := 3
def blouses_needed : ℕ := 5
def budget : ℕ := 180

theorem pants_needed : 
  (skirts_needed * skirt_cost) + (blouses_needed * blouse_cost) + 2 * pants_cost - pants_discounted_cost = budget →
  2 ∗ pants_cost - pants_discounted_cost = 45 →
  2 ∗ pants_cost - pants_discounted_cost = ⟨ pants_needed ⟩

by
  sorry

end pants_needed_l427_427922


namespace find_clique_of_size_6_l427_427860

-- Defining the conditions of the graph G
variable (G : SimpleGraph (Fin 12))

-- Condition: For any subset of 9 vertices, there exists a subset of 5 vertices that form a complete subgraph K_5.
def condition (s : Finset (Fin 12)) : Prop :=
  s.card = 9 → ∃ t : Finset (Fin 12), t ⊆ s ∧ t.card = 5 ∧ (∀ u v : Fin 12, u ∈ t → v ∈ t → u ≠ v → G.Adj u v)

-- The theorem to prove given the conditions
theorem find_clique_of_size_6 (h : ∀ s : Finset (Fin 12), condition G s) : 
  ∃ t : Finset (Fin 12), t.card = 6 ∧ (∀ u v : Fin 12, u ∈ t → v ∈ t → u ≠ v → G.Adj u v) :=
sorry

end find_clique_of_size_6_l427_427860


namespace range_of_m_l427_427844

-- Define the quadratic function
def quadratic_function (x m : ℝ) : ℝ := (x - m) ^ 2 - 1

-- State the main theorem
theorem range_of_m (m : ℝ) :
  (∀ x ≤ 3, quadratic_function x m ≥ quadratic_function (x + 1) m) ↔ m ≥ 3 :=
by
  sorry

end range_of_m_l427_427844


namespace houses_in_block_l427_427672

theorem houses_in_block (junk_mail_per_house : ℕ) (total_junk_mail : ℕ) (h1 : junk_mail_per_house = 2) (h2 : total_junk_mail = 14) :
  total_junk_mail / junk_mail_per_house = 7 := by
  sorry

end houses_in_block_l427_427672


namespace value_of_f_2018_l427_427069

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodicity (x : ℝ) : f (x + 3) * f x = -1
axiom initial_condition : f (-1) = 2

theorem value_of_f_2018 : f 2018 = -1 / 2 :=
by
  sorry

end value_of_f_2018_l427_427069


namespace range_of_a_l427_427423

open Set

def real_intervals (a : ℝ) : Prop :=
  let S := {x : ℝ | (x - 2)^2 > 9}
  let T := Ioo a (a + 8)
  S ∪ T = univ → -3 < a ∧ a < -1

theorem range_of_a (a : ℝ) : real_intervals a :=
sorry

end range_of_a_l427_427423


namespace coeff_x2_product_l427_427181

open Polynomial

noncomputable def poly1 : Polynomial ℤ := -5 * X^3 - 5 * X^2 - 7 * X + 1
noncomputable def poly2 : Polynomial ℤ := -X^2 - 6 * X + 1

theorem coeff_x2_product : (poly1 * poly2).coeff 2 = 36 := by
  sorry

end coeff_x2_product_l427_427181


namespace right_triangle_expression_l427_427453

theorem right_triangle_expression (a c b : ℝ) (h1 : c = a + 2) (h2 : a^2 + b^2 = c^2) : 
  b^2 = 4 * (a + 1) :=
by
  sorry

end right_triangle_expression_l427_427453


namespace log_base_5_of_390625_l427_427321

theorem log_base_5_of_390625 : log 5 390625 = 8 :=
by {
  have h1 : 390625 = 5 ^ 8 := by norm_num,
  sorry
}

end log_base_5_of_390625_l427_427321


namespace arrangements_possible_l427_427606

theorem arrangements_possible :
  let english_speakers := 4
  let french_speakers := 2
  let bilingual_speakers := 2
  let total_volunteers := 8
  
  -- Number of ways to choose the delegates
  (finset.card (finset.filter (λ x : fin 8, x < english_speakers + french_speakers + bilingual_speakers)
    (finset.univ : finset (fin total_volunteers)))) = 28 :=
by
  -- Definitions without detailed steps
  let english_speakers := 4
  let french_speakers := 2
  let bilingual_speakers := 2
  sorry

end arrangements_possible_l427_427606


namespace reflections_on_circumcircle_l427_427078

variable {Point : Type}
variable {triangle : Point → Point → Point → Prop}
variable {circumcircle : Point → Point → Point → (Point → Prop)}
variable {orthocenter : Point → Point → Point → Point}
variable {reflection_over : Point → Point → Point}

theorem reflections_on_circumcircle
  (A B C H A' B' C' : Point)
  (ΔABC : triangle A B C)
  (Γ : circumcircle A B C)
  (H_is_orthocenter : H = orthocenter A B C)
  (A'_is_reflection : A' = reflection_over H (λ x, x ∈ segment B C))
  (B'_is_reflection : B' = reflection_over H (λ x, x ∈ segment C A))
  (C'_is_reflection : C' = reflection_over H (λ x, x ∈ segment A B)) :
  Γ A' ∧ Γ B' ∧ Γ C' :=
sorry

end reflections_on_circumcircle_l427_427078


namespace distance_apart_after_2_hours_l427_427277

-- Defining the rates and times
def alex_rate := 1 / 15 -- miles per minute
def sam_rate := 1.5 / 45 -- miles per minute
def time := 120 -- total time in minutes, which is 2 hours

-- Definitions of distances traveled by Alex and Sam
def alex_distance := alex_rate * time
def sam_distance := sam_rate * time

-- Proof statement that needs to be verified
theorem distance_apart_after_2_hours : alex_distance + sam_distance = 12 := by
  sorry

end distance_apart_after_2_hours_l427_427277


namespace find_angles_and_area_l427_427863

noncomputable def triangle_ABC : Type := sorry
noncomputable def isosceles (ABC : triangle_ABC) : Prop := sorry
noncomputable def median_AM (ABC : triangle_ABC) : Type := sorry
noncomputable def median_CN (ABC : triangle_ABC) : Type := sorry
noncomputable def medians_intersect_at_right_angle (AM CN : Type) (D : Type) : Prop := sorry

theorem find_angles_and_area (ABC : triangle_ABC) 
  (h1 : isosceles ABC)
  (h2 : medians_intersect_at_right_angle (median_AM ABC) (median_CN ABC) sorry) 
  (h3 : AC_length ABC = 1) :
  (angle_BAC ABC = Real.arctan 3) ∧
  (angle_BCA ABC = Real.arctan 3) ∧
  (angle_ABC ABC = Real.pi - 2 * Real.arctan 3) ∧
  (area_NBMD ABC = 1 / 4) := 
sorry

end find_angles_and_area_l427_427863


namespace find_H_coordinates_l427_427173

theorem find_H_coordinates (E F G : ℝ × ℝ × ℝ) (H : ℝ × ℝ × ℝ) 
  (hE : E = (2, 3, 1)) (hF : F = (4, -1, -3)) (hG : G = (0, -4, 1)) :
  H = (-2, -1, 5) ↔
  let midEG := ((E.1 + G.1) / 2, (E.2 + G.2) / 2, (E.3 + G.3) / 2) in
  let midFH := ((F.1 + H.1) / 2, (F.2 + H.2) / 2, (F.3 + H.3) / 2) in
  midEG = midFH := by
  sorry

end find_H_coordinates_l427_427173


namespace opposite_of_neg_two_l427_427985

theorem opposite_of_neg_two : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_two_l427_427985


namespace perfect_square_condition_l427_427767

-- Define the product of factorials from 1 to 2n
def factorial_product (n : ℕ) : ℕ :=
  (List.prod (List.map factorial (List.range (2 * n + 1))))

-- The main theorem statement
theorem perfect_square_condition (n : ℕ) : 
  (∃ k : ℕ, n = 4 * k * (k + 1) ∨ n = 2 * k^2 - 1) ↔
  (∃ m : ℕ, (factorial_product n) / ((n + 1)!) = m^2) := sorry

end perfect_square_condition_l427_427767


namespace Ferris_wheel_cost_l427_427316

theorem Ferris_wheel_cost
  (rides_ferris_wheel : ℕ) 
  (rides_roller_coaster : ℕ) 
  (rides_log_ride : ℕ) 
  (cost_roller_coaster : ℕ) 
  (cost_log_ride : ℕ) 
  (initial_tickets : ℕ) 
  (additional_tickets : ℕ) 
  (total_tickets_needed : ℕ) 
  (total_tickets : ℕ) 
  (remaining_tickets : ℕ) 
  (cost_per_ferris_wheel : ℕ)
  (rides_ferris_wheel = 2)
  (rides_roller_coaster = 3)
  (rides_log_ride = 7)
  (cost_roller_coaster = 5)
  (cost_log_ride = 1)
  (initial_tickets = 20)
  (additional_tickets = 6)
  (total_tickets = initial_tickets + additional_tickets)
  (total_tickets_needed = rides_roller_coaster * cost_roller_coaster + rides_log_ride * cost_log_ride)
  (remaining_tickets = total_tickets - total_tickets_needed)
  (cost_per_ferris_wheel = remaining_tickets / rides_ferris_wheel) :
  cost_per_ferris_wheel = 2 := sorry

end Ferris_wheel_cost_l427_427316


namespace sum_a_n_l427_427793

def unit_digit (n : ℕ) : ℕ := n % 10

def a_n (n : ℕ) : ℕ := unit_digit (n^2) - unit_digit (n)

theorem sum_a_n : (∑ n in Finset.range 2008, a_n (n+1)) = 6421 := by
  sorry

end sum_a_n_l427_427793


namespace find_inverse_condition_l427_427833

variable {α β : Type} [inv : InverseFunction α β]

def condition (f g : α → β) (f_inv : β → α) : Prop :=
  ∀ x, f_inv (g x) = x^4 - 1

theorem find_inverse_condition (f g : α → β) (f_inv : β → α) 
  (h₁ : condition f g f_inv) (h₂ : has_inverse g) : g⁻¹ (f 15) = 2 := 
sorry

end find_inverse_condition_l427_427833


namespace length_of_chord_equation_of_line_l427_427783

-- Definitions for part (I)
def point_M : ℝ × ℝ := (-1, 2)
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 8
def inclination_angle : ℝ := 3 * Real.pi / 4

-- Problem I: Prove length of chord AB
theorem length_of_chord
  (alpha : ℝ)
  (M : ℝ × ℝ) 
  (inside_circle : circle_eq M.1 M.2) 
  (h_alpha : alpha = inclination_angle) : 
  ∃ AB : ℝ, AB = Real.sqrt 30 :=
  sorry

-- Definitions for part (II)
def slope_perpendicular (k : ℝ) : ℝ := -1 / k

-- Problem II: Prove equation of line AB
theorem equation_of_line
  (M : ℝ × ℝ) 
  (inside_circle : circle_eq M.1 M.2)
  (bisected_by_M : True) -- Here, we assume a placeholder for the bisection condition
  : ∃ (a b c : ℝ), a * M.1 + b * M.2 + c = 0 ∧ a = 1 ∧ b = -2 ∧ c = 5 :=
  sorry

end length_of_chord_equation_of_line_l427_427783


namespace vertex_of_quadratic_l427_427548

theorem vertex_of_quadratic (x : ℝ) : 
  (y : ℝ) = -2 * (x + 1) ^ 2 + 3 →
  (∃ vertex_x vertex_y : ℝ, vertex_x = -1 ∧ vertex_y = 3 ∧ y = -2 * (vertex_x + 1) ^ 2 + vertex_y) :=
by
  intro h
  exists -1, 3
  simp [h]
  sorry

end vertex_of_quadratic_l427_427548


namespace find_natural_numbers_l427_427727

theorem find_natural_numbers (n : ℕ) : 
  (∃ d : ℕ, d ≤ 9 ∧ 10 * n + d = 13 * n) ↔ n = 1 ∨ n = 2 ∨ n = 3 :=
by {
  sorry
}

end find_natural_numbers_l427_427727


namespace machine_value_after_two_years_l427_427140

noncomputable def machine_market_value (initial_value : ℝ) (years : ℕ) (decrease_rate : ℝ) : ℝ :=
  initial_value * (1 - decrease_rate) ^ years

theorem machine_value_after_two_years :
  machine_market_value 8000 2 0.2 = 5120 := by
  sorry

end machine_value_after_two_years_l427_427140


namespace initial_bananas_proof_l427_427093

noncomputable def initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) (extra_bananas : ℕ) : ℕ :=
  (extra_bananas * (total_children - absent_children)) / (total_children - extra_bananas)

theorem initial_bananas_proof
  (total_children : ℕ)
  (absent_children : ℕ)
  (extra_bananas : ℕ)
  (h_total : total_children = 640)
  (h_absent : absent_children = 320)
  (h_extra : extra_bananas = 2) : initial_bananas_per_child total_children absent_children extra_bananas = 2 :=
by
  sorry

end initial_bananas_proof_l427_427093


namespace solution_to_system_l427_427115

theorem solution_to_system :
  ∀ (x y z : ℝ), 
  x * (3 * y^2 + 1) = y * (y^2 + 3) →
  y * (3 * z^2 + 1) = z * (z^2 + 3) →
  z * (3 * x^2 + 1) = x * (x^2 + 3) →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ 
  (x = -1 ∧ y = -1 ∧ z = -1) :=
by
  sorry

end solution_to_system_l427_427115


namespace solution_set_inequality_l427_427600

theorem solution_set_inequality (x : ℝ) : (x^2-2*x-3)*(x^2+1) < 0 ↔ -1 < x ∧ x < 3 :=
by
  sorry

end solution_set_inequality_l427_427600


namespace polynomial_inequality_l427_427063

noncomputable def P (X : ℝ) : ℝ := sorry -- Assume a polynomial with positive coefficients

theorem polynomial_inequality (n : ℕ) (h_n : n ≥ 2) 
  (x : fin n → ℝ) (h_x : ∀ i, 0 < x i) :
  (finset.univ.sum (λ i, (P (x i / x ((i : fin n).succ % n))) ^ 2)) ≥ n * (P 1) ^ 2 :=
sorry

end polynomial_inequality_l427_427063


namespace compute_expression_l427_427716

theorem compute_expression : (3 + 6 + 9)^3 + (3^3 + 6^3 + 9^3) = 6804 := by
  sorry

end compute_expression_l427_427716


namespace minimum_points_for_parallel_lines_l427_427668

noncomputable def minimum_number_of_points : Nat := 10

def valid_set_of_points (M : Finset (EuclideanSpace ℝ (Fin 3))) : Prop :=
  ∀ (A B : EuclideanSpace ℝ (Fin 3)), A ∈ M → B ∈ M → 
  A ≠ B → ∃ (C D : EuclideanSpace ℝ (Fin 3)), C ∈ M ∧ D ∈ M ∧ 
  (A - B).cross (C - D) = 0 ∧ (A - B) ≠ ±(C - D)

theorem minimum_points_for_parallel_lines :
  ∃ (M : Finset (EuclideanSpace ℝ (Fin 3))),
  M.card = minimum_number_of_points ∧ valid_set_of_points M :=
begin
  sorry
end

end minimum_points_for_parallel_lines_l427_427668


namespace family_work_together_l427_427266

theorem family_work_together :
  let man_days := 10
  let father_days := 20
  let son_days := 25
  let wife_days := 15
  let daughter_days := 30
  let total_inverse_work_rate := (1 / man_days) + (1 / father_days) + (1 / son_days) + (1 / wife_days) + (1 / daughter_days)
  let total_days := 1 / total_inverse_work_rate
  total_days ≈ 3.45 :=
begin
  let man_days := 10,
  let father_days := 20,
  let son_days := 25,
  let wife_days := 15,
  let daughter_days := 30,
  let total_inverse_work_rate := (1 / man_days) + (1 / father_days) + (1 / son_days) + (1 / wife_days) + (1 / daughter_days),
  let total_days := 1 / total_inverse_work_rate,
  show total_days ≈ 3.45,
  sorry
end

end family_work_together_l427_427266


namespace sum_of_arithmetic_seq_l427_427440

variable {n : ℕ}

noncomputable def arithmeticSeqSum (n : ℕ) : ℚ :=
  let x : ℕ → ℚ
  | 0       => 2
  | k + 1   => x k + (1 / 3)
  in (Finset.range n).sum x

theorem sum_of_arithmetic_seq (n : ℕ) (h : n > 0) :
  arithmeticSeqSum n = n * (n + 11) / 6 :=
sorry

end sum_of_arithmetic_seq_l427_427440


namespace evaluate_poly_at_fraction_l427_427735

theorem evaluate_poly_at_fraction :
  let a := (4 : ℚ) / 3 in 
  (7 * a^2 - 15 * a + 2) * (3 * a - 4) = 0 :=
by
  let a := (4 : ℚ) / 3
  sorry

end evaluate_poly_at_fraction_l427_427735


namespace george_correct_answer_l427_427376

variable (y : ℝ)

theorem george_correct_answer (h : y / 7 = 30) : 70 + y = 280 :=
sorry

end george_correct_answer_l427_427376


namespace domain_of_f_smallest_positive_period_of_f_max_min_values_of_f_decreasing_intervals_of_f_l427_427804

noncomputable def f (x : ℝ) : ℝ := ((Real.cos x - Real.sin x) * Real.sin (2 * x)) / Real.cos x

theorem domain_of_f : ∀ x, (∃ k : ℤ, x = k * Real.pi + Real.pi / 2) → ¬ ∃ y, f y = 0 → y = x :=
begin
  sorry
end

theorem smallest_positive_period_of_f : ∃ T, T > 0 ∧ ∀ x, f x = f (x + T) :=
begin
  exact ⟨Real.pi, Real.pi_pos, sorry⟩,
end

theorem max_min_values_of_f (x : ℝ) : x ∈ Ioc (-Real.pi / 2) 0 →
  (∀ y, f y = 0 → y = 0) ∧ (∀ y, f y = -Real.sqrt 2 - 1 → y = -3 * Real.pi / 8) :=
begin
  sorry
end

theorem decreasing_intervals_of_f (k : ℤ) : 
  ∀ x, x ≠ k * Real.pi + Real.pi / 2 →
  (x ∈ Icc (k * Real.pi + Real.pi / 8) (k * Real.pi + Real.pi / 2) ∨ x ∈ Icc (k * Real.pi + Real.pi / 2) (k * Real.pi + 5 * Real.pi / 8)) → 
  ∃ I : Set ℝ, I = Icc (k * Real.pi + Real.pi / 8) (k * Real.pi + Real.pi / 2) ∪ Icc (k * Real.pi + Real.pi / 2) (k * Real.pi + 5 * Real.pi / 8) ∧ 
  ∀ a b : ℝ, a, b ∈ I → a < b → f a ≥ f b :=
begin
  sorry
end

end domain_of_f_smallest_positive_period_of_f_max_min_values_of_f_decreasing_intervals_of_f_l427_427804


namespace sum_of_squares_AP_l427_427237

theorem sum_of_squares_AP 
  (a d : ℤ)
  (h: 2*a^2 + 28*a*d + 130*d^2 = 364)
  : (∑ i in Finset.range 15, (a + i * d)^2) = 
    15*a^2 + 2*d^2 * (14 * 15 * 29) / 6 + 2*a*d * (14 * 15) / 2 := 
by
  sorry

end sum_of_squares_AP_l427_427237


namespace combined_work_days_l427_427644

-- Definitions for the conditions given in the problem.
def work_rate_x (W : ℝ) : ℝ := W / 20
def work_rate_y (W : ℝ) : ℝ := W / 40

-- The theorem to prove the number of days for x and y to complete the work together.
theorem combined_work_days (W : ℝ) (hW : W ≠ 0) : 
  ∃ d : ℝ, d = 40 / 3 :=
by 
  have wrx := work_rate_x W,
  have wry := work_rate_y W,
  let combined_rate := wrx + wry,
  have d : ℝ := W / combined_rate,
  use d,
  sorry

end combined_work_days_l427_427644


namespace ball_hits_ground_l427_427132

theorem ball_hits_ground :
  ∃ t : ℝ, t = 4.00 ∧ (∀ ε > 0, abs((-4.5 * t^2 - 12 * t + 72)) < ε) := sorry

end ball_hits_ground_l427_427132


namespace orchard_total_mass_l427_427699

-- Define the number of apple trees, apple yield, number of peach trees, and peach yield
def num_apple_trees : ℕ := 30
def apple_yield_per_tree : ℕ := 150
def num_peach_trees : ℕ := 45
def peach_yield_per_tree : ℕ := 65

-- Calculate the total yield of apples and peaches
def total_apple_yield : ℕ := num_apple_trees * apple_yield_per_tree
def total_peach_yield : ℕ := num_peach_trees * peach_yield_per_tree

-- Define the total mass of fruit harvested
def total_fruit_yield : ℕ := total_apple_yield + total_peach_yield

-- Theorem stating that the total mass of fruit harvested is 7425 kg 
theorem orchard_total_mass : total_fruit_yield = 7425 := by
  simp [total_apple_yield, total_peach_yield, num_apple_trees, apple_yield_per_tree, num_peach_trees, peach_yield_per_tree]
  sorry

end orchard_total_mass_l427_427699


namespace relationship_among_abc_l427_427782

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def decreasing_f_x_f' (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x < 0 → f x + x * (deriv f x) < 0

variables (f : ℝ → ℝ)
variable (h_even : is_even f)
variable (h_decreasing : decreasing_f_x_f' f)
noncomputable def a := (0.7 ^ 6) * f (0.7 ^ 6)
noncomputable def b := Real.logb (10/7) 6 * f (Real.logb (10/7) 6)
noncomputable def c := (6 ^ 0.6) * f (6 ^ 0.6)

theorem relationship_among_abc : a f > c f > b f :=
  sorry

end relationship_among_abc_l427_427782


namespace opposite_neg_two_l427_427981

theorem opposite_neg_two : -(-2) = 2 := by
  sorry

end opposite_neg_two_l427_427981


namespace max_value_of_n_l427_427339

-- Define the main variables and conditions
noncomputable def max_n : ℕ := 
  let n_pairs := [(1, 72), (2, 36), (3, 24), (4, 18), (6, 12), (8, 9)] 
  max (n_pairs.map (λ p, 6 * p.2 + p.1))

-- Lean theorem statement for the equivalence
theorem max_value_of_n :
  max_n = 433 := by
  sorry

end max_value_of_n_l427_427339


namespace record_loss_as_negative_l427_427027

def record_financial_transaction : ℤ → ℤ
| (profit : ℤ) := profit

theorem record_loss_as_negative 
  (profit : ℤ) (h : record_financial_transaction profit = 370) : 
  record_financial_transaction (-60) = -60 :=
by
  -- The proof is omitted here as instructed.
  sorry

end record_loss_as_negative_l427_427027


namespace factorial_sequence_perfect_square_l427_427762

-- Definitions based on the conditions
def is_perf_square (x : ℕ) : Prop := ∃ (k : ℕ), x = k * k

def factorial (n : ℕ) : ℕ := Nat.recOn n 1 (λ n fac_n, (n + 1) * fac_n)

def factorial_seq_prod (n : ℕ) : ℕ :=
  Nat.recOn n 1 (λ n prodN, factorial (2 * n) * prodN)

-- Main statement
theorem factorial_sequence_perfect_square (n : ℕ) :
  is_perf_square (factorial_seq_prod n / factorial (n + 1)) ↔
  ∃ k : ℕ, n = 4 * k * (k + 1) ∨ n = 2 * k ^ 2 - 1 := by
  sorry

end factorial_sequence_perfect_square_l427_427762


namespace shortest_path_between_points_l427_427865

theorem shortest_path_between_points {P : Type*} [metric_space P] (a b : P) :
  (∀ path, path.is_winding → (∃! path', path'.is_straight_and_connects a b ∧ path'.distance = (dist a b))) →
  ∃! s : set P, is_straight_and_connects s a b ∧ ∀ q ∈ s, dist a q + dist q b = dist a b :=
by
  sorry

end shortest_path_between_points_l427_427865


namespace sum_x_coordinates_common_points_l427_427092

-- Definition of the equivalence relation modulo 9
def equiv_mod (a b n : ℤ) : Prop := ∃ k : ℤ, a = b + n * k

-- Definitions of the given conditions
def graph1 (x y : ℤ) : Prop := equiv_mod y (3 * x + 6) 9
def graph2 (x y : ℤ) : Prop := equiv_mod y (7 * x + 3) 9

-- Definition of when two graphs intersect
def points_in_common (x y : ℤ) : Prop := graph1 x y ∧ graph2 x y

-- Proof that the sum of the x-coordinates of the points in common is 3
theorem sum_x_coordinates_common_points : 
  ∃ x y, points_in_common x y ∧ (x = 3) := 
sorry

end sum_x_coordinates_common_points_l427_427092


namespace team_selection_l427_427090

open Finset

/-
My school's math club currently includes 10 boys and 12 girls.
I need to form a team of 8 people to send to a national math competition.
How many ways can I select this team if the team must include at least 3 boys?
-/

def number_of_ways_to_select_team (boys girls : ℕ) (team_size min_boys : ℕ) : ℕ :=
  (range (min_boys, team_size + 1)).sum (λ b => (choose boys b) * (choose girls (team_size - b)))

theorem team_selection :
  number_of_ways_to_select_team 10 12 8 3 = 221775 :=
by
  sorry

end team_selection_l427_427090


namespace right_triangle_third_side_l427_427401

theorem right_triangle_third_side :
  (∃ (a b : ℝ), a ≠ b ∧ (a^2 - 9 * a + 20 = 0) ∧ (b^2 - 9 * b + 20 = 0) ∧
  (∃ (c : ℝ), (c = real.sqrt (a^2 + b^2)) ∨ (c = real.sqrt (b^2 - a^2)))
  → (c = 3 ∨ c = real.sqrt 41) :=
by
  sorry

end right_triangle_third_side_l427_427401


namespace max_value_of_n_l427_427340

-- Define the main variables and conditions
noncomputable def max_n : ℕ := 
  let n_pairs := [(1, 72), (2, 36), (3, 24), (4, 18), (6, 12), (8, 9)] 
  max (n_pairs.map (λ p, 6 * p.2 + p.1))

-- Lean theorem statement for the equivalence
theorem max_value_of_n :
  max_n = 433 := by
  sorry

end max_value_of_n_l427_427340


namespace proof_circumcircle_perpendicular_l427_427530

-- Definitions for the given problem
variable {A B C D K : Type} [MetricSpace A]

-- Condition 1: ABC is a right triangle with ∠C = 90°
variable (ABC : Triangle A B C) (hC : angle ABC A C = 90⁰)

-- Condition 2: CD is the altitude from C to AB
variable (CD : Line C D) (hCD : isAltitude C D)

-- Condition 3: K is a point on the plane such that |AK| = |AC|
variable (K : Point) (hAK : dist A K = dist A C)

-- Question and Answer: Prove that the length of the diameter of the circumcircle passing through vertex A is perpendicular to the line DK
theorem proof_circumcircle_perpendicular (hABC : isRightTriangle ABC C) (hAltitude : isAltitude C D) (hAK_eq_AC : dist A K = dist A C)
  (circumcircle : Circle (triangleCircumcenter K A B) (dist (triangleCircumcenter K A B) A))
  (diameter : Line segment (triangleCircumcenter K A B) A) :
  isPerpendicular diameter (Line D K) :=
sorry

end proof_circumcircle_perpendicular_l427_427530


namespace poly_remainder_div_l427_427066

noncomputable def Q (x : ℝ) : ℝ := sorry

theorem poly_remainder_div (Q : ℝ → ℝ) (h1 : Q 5 = 15) (h2 : Q 15 = 5) :
  ∃ (a b : ℝ), (Q x = (x - 5) * (x - 15) * (some_polynomial x) + a * x + b) ∧ a = -1 ∧ b = 20 := 
sorry

end poly_remainder_div_l427_427066


namespace sin_alpha_value_l427_427795

theorem sin_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : cos (α + π / 6) = 2 / 3) :
  sin α = (sqrt 15 - 2) / 6 :=
by
  sorry

end sin_alpha_value_l427_427795


namespace initial_walnuts_l427_427254

theorem initial_walnuts (W : ℕ) (boy_effective : ℕ) (girl_effective : ℕ) (total_walnuts : ℕ) :
  boy_effective = 5 → girl_effective = 3 → total_walnuts = 20 → W + boy_effective + girl_effective = total_walnuts → W = 12 :=
by
  intros h_boy h_girl h_total h_eq
  rw [h_boy, h_girl, h_total] at h_eq
  linarith

end initial_walnuts_l427_427254


namespace integral_abs_cos_eq_4_l427_427717

theorem integral_abs_cos_eq_4 :
  ∫ x in 0..(2 * Real.pi), |Real.cos x| = 4 :=
by
  sorry

end integral_abs_cos_eq_4_l427_427717


namespace alice_wins_if_PC_valid_ratio_l427_427694

open_locale classical

noncomputable theory

variables {A B C D P Q : Type}
variables [euclidean_geometry B C A]

-- Given: Equilateral triangle ABC
def equilateral_triangle (A B C : Type) : Prop := 
  ∀ (Δ : triangle A B C), Δ.is_equilateral

-- Point D on segment BC such that BD = 3 and CD = 5
def point_D_on_BC (B C D : Type) : Prop := 
  ∃ (BD CD : ℝ), BD = 3 ∧ CD = 5 ∧ B = C + BD + CD

-- P is a point on line AD
def point_P_on_AD (A D P : Type) : Prop := 
  ∃ (P : Type), P ∈ line A D

-- Main statement: Condition for Alice to win (no Q ≠ P exists on AD s.t. BQ/QC = BP/PC)
theorem alice_wins_if_PC_valid_ratio :
  (equilateral_triangle A B C) → 
  (point_D_on_BC B C D) → 
  (point_P_on_AD A D P) → 
  (Q ≠ P → Q ∈ line A D → (BQ / QC = BP / PC) → false) → 
  ∃ r : ℝ, r ∈ {1, sqrt(3) / 3, (3 * sqrt(3)) / 5} :=
begin
  sorry
end

end alice_wins_if_PC_valid_ratio_l427_427694


namespace worth_of_presents_is_33536_36_l427_427894

noncomputable def total_worth_of_presents : ℝ :=
  let ring := 4000
  let car := 2000
  let bracelet := 2 * ring
  let gown := bracelet / 2
  let jewelry := 1.2 * ring
  let painting := 3000 * 1.2
  let honeymoon := 180000 / 110
  let watch := 5500
  ring + car + bracelet + gown + jewelry + painting + honeymoon + watch

theorem worth_of_presents_is_33536_36 : total_worth_of_presents = 33536.36 := by
  sorry

end worth_of_presents_is_33536_36_l427_427894


namespace each_persons_final_share_l427_427161

theorem each_persons_final_share
  (total_dining_bill : ℝ)
  (number_of_people : ℕ)
  (tip_percentage : ℝ) :
  total_dining_bill = 211.00 →
  tip_percentage = 0.15 →
  number_of_people = 5 →
  ((total_dining_bill + total_dining_bill * tip_percentage) / number_of_people) = 48.53 :=
by
  intros
  sorry

end each_persons_final_share_l427_427161


namespace smallest_integer_divisibility_conditions_l427_427623

theorem smallest_integer_divisibility_conditions :
  ∃ n : ℕ, n > 0 ∧ (24 ∣ n^2) ∧ (900 ∣ n^3) ∧ (1024 ∣ n^4) ∧ n = 120 :=
by
  sorry

end smallest_integer_divisibility_conditions_l427_427623


namespace range_of_a_for_monotonicity_l427_427805

noncomputable def f (x a : ℝ) : ℝ := -x^2 + x - a * Real.log x

theorem range_of_a_for_monotonicity (a : ℝ) :
  (∀ x ∈ Ioi (0:ℝ), (f x a)' x ≥ 0) ∨ (∀ x ∈ Ioi (0:ℝ), (f x a)' x ≤ 0) ↔ a ∈ Icc (1/8 : ℝ) (⊤) :=
sorry

end range_of_a_for_monotonicity_l427_427805


namespace square_side_length_l427_427117

theorem square_side_length (a b c : ℕ) (habc_pos : 0 < a ∧ 0 < b ∧ 0 < c) (hprime_b : Prime b) :
  let s := (a : ℝ) - Real.sqrt (b : ℝ) / (c : ℝ) in
  a + b + c = 5 :=
begin
  sorry
end

end square_side_length_l427_427117


namespace amount_lent_by_A_to_B_l427_427670

variable (P : ℝ)
def interest_from_C := P * (18.5 / 100) * 3
def interest_to_A := P * (15 / 100) * 3
def gain_B := interest_from_C P - interest_to_A P

theorem amount_lent_by_A_to_B (h : gain_B P = 294) : P = 2800 :=
by
  rw [gain_B, interest_from_C, interest_to_A] at h
  sorry

end amount_lent_by_A_to_B_l427_427670


namespace chord_length_is_sqrt_14_l427_427801

-- Define the line as a function
def line (x y : ℝ) : Prop := x - y + 2 = 0

-- Define the circle as a function
def circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

-- Definition of the length of chord where the line intersects the circle
def chord_length (A B : ℝ × ℝ) : ℝ := 
  let (x1, y1) := A
  let (x2, y2) := B in
  Real.dist ⟨x1, y1⟩ ⟨x2, y2⟩

-- Main theorem
theorem chord_length_is_sqrt_14 : ∃ A B : ℝ × ℝ, line A.1 A.2 ∧ line B.1 B.2 ∧ circle A.1 A.2 ∧ circle B.1 B.2 ∧ chord_length A B = Real.sqrt 14 := 
sorry

end chord_length_is_sqrt_14_l427_427801


namespace total_spots_l427_427706

variable (P : ℕ)
variable (Bill_spots : ℕ := 2 * P - 1)

-- Given conditions
variable (h1 : Bill_spots = 39)

-- Theorem we need to prove
theorem total_spots (P : ℕ) (Bill_spots : ℕ := 2 * P - 1) (h1 : Bill_spots = 39) : 
  Bill_spots + P = 59 := 
by
  sorry

end total_spots_l427_427706


namespace parabola_equation_l427_427812

variables (a b c p : ℝ)
variables (h1 : a > 0) (h2 : b > 0) (h3 : p > 0)
variables (h_eccentricity : c / a = 2)
variables (h_b : b = Real.sqrt (3) * a)
variables (h_c : c = Real.sqrt (a^2 + b^2))
variables (d : ℝ) (h_distance : d = 2) (h_d_formula : d = (a * p) / (2 * c))

theorem parabola_equation (h : (a > 0) ∧ (b > 0) ∧ (p > 0) ∧ (c / a = 2) ∧ (b = (Real.sqrt 3) * a) ∧ (c = Real.sqrt (a^2 + b^2)) ∧ (d = 2) ∧ (d = (a * p) / (2 * c))) : x^2 = 16 * y :=
by {
  -- Lean does not require an actual proof here, so we use sorry.
  sorry
}

end parabola_equation_l427_427812


namespace value_of_a_l427_427442

theorem value_of_a (a : ℝ) : 
  (-1, a) lies_on_terminal_side_of_angle 600 → a = -sqrt 3 := 
by 
  sorry

end value_of_a_l427_427442


namespace ab_plus_cd_is_composite_l427_427481

theorem ab_plus_cd_is_composite 
  (a b c d : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_order : a > b ∧ b > c ∧ c > d)
  (h_eq : a^2 + a * c - c^2 = b^2 + b * d - d^2) : 
  ∃ p q : ℕ, p > 1 ∧ q > 1 ∧ ab + cd = p * q :=
by
  sorry

end ab_plus_cd_is_composite_l427_427481


namespace geom_prog_a_n_product_ineq_l427_427779

noncomputable def a_seq : ℕ → ℚ
| 1 := 5 / 4
| (n+1) := if n = 0 then 5 / 4 else (5 * (n + 1) * a_seq n) / (4 * a_seq n + n)

theorem geom_prog_a_n (n : ℕ) (h : n ≥ 2) :
  a_seq n = n / (1 - (1 / 5)^n) :=
sorry

theorem product_ineq (n : ℕ) (h : n ≥ 2) :
  (∏ i in range n, a_seq (i + 1)) < n! / (1 - (1 / 5) - (1 / 5^2) - ... - (1 / 5^n)) :=
sorry

end geom_prog_a_n_product_ineq_l427_427779


namespace floor_sqrt_23_squared_l427_427320

theorem floor_sqrt_23_squared : (Int.floor (Real.sqrt 23))^2 = 16 := 
by
  -- conditions
  have h1 : 4^2 = 16 := by norm_num
  have h2 : 5^2 = 25 := by norm_num
  have h3 : 16 < 23 := by norm_num
  have h4 : 23 < 25 := by norm_num
  -- statement (goal)
  sorry

end floor_sqrt_23_squared_l427_427320


namespace part1_l427_427810

def f (x : ℝ) (a : ℝ) : ℝ := (2 * x ^ 2 - 4 * x + 4) * Real.exp x - a * x ^ 2 - Real.exp 1

theorem part1 (a : ℝ) (h : ∀ (x : ℝ), f 1 a + f' (1) (a) * (x - 1) = 1 - Real.exp 1 → x = 0) : a = 1 :=
by
  sorry

end part1_l427_427810


namespace cards_traded_between_Padma_and_Robert_l427_427539

def total_cards_traded (padma_first_trade padma_second_trade robert_first_trade robert_second_trade : ℕ) : ℕ :=
  padma_first_trade + padma_second_trade + robert_first_trade + robert_second_trade

theorem cards_traded_between_Padma_and_Robert (h1 : padma_first_trade = 2) 
                                            (h2 : robert_first_trade = 10)
                                            (h3 : padma_second_trade = 15)
                                            (h4 : robert_second_trade = 8) :
                                            total_cards_traded 2 15 10 8 = 35 := 
by 
  sorry

end cards_traded_between_Padma_and_Robert_l427_427539


namespace marvel_value_l427_427139

def alphabet_value : ℕ → ℤ
| 1 := -3
| 2 := -2
| 3 := -1
| 4 := 0
| 5 := 1
| 6 := 2
| 7 := 3
| 8 := 2
| (n + 1) := alphabet_value ((n % 8) + 1)

def value_of_word (word : List ℕ) : ℤ :=
  word.sum (alphabet_value)

theorem marvel_value :
  value_of_word [13, 1, 18, 22, 5, 12] = -5 :=
by {
  -- The proof will go here.
  sorry
}

end marvel_value_l427_427139


namespace domain_ln_sqrt_l427_427572

theorem domain_ln_sqrt :
  (∀ x : ℝ, (x + 1 > 0) ∧ (-x^2 - 3x + 4 > 0)) →
  (∀ x : ℝ, -1 < x ∧ x < 1) :=
by
  intro h
  sorry

end domain_ln_sqrt_l427_427572


namespace opposite_face_A_is_F_l427_427142

def faces : List (Char) := ['A', 'B', 'C', 'D', 'E', 'F']

theorem opposite_face_A_is_F (F_is_bottom : faces.get! 5 = 'F') : 
  ∃ (face : Char), face = 'A' ∧ faces.get! 5 = 'F' ∧ (∃ (opposite : Char), opposite = 'F') :=
by
  -- here we set face 'A' as the top face
  let top_face := faces.get! 0 -- Assuming 'A' is face 0
  exists top_face
  split
  . rfl
  . split
  . exact F_is_bottom
  . exists 'F'
  . rfl
  sorry

end opposite_face_A_is_F_l427_427142


namespace last_two_digits_l427_427354

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Compute the sum of factorials of multiples of 5 up to 100, modulo 100
def sum_factorials_mod_100 : ℕ :=
  (List.range 21).sum (λ k => factorial (5 * k) % 100)

-- The theorem statement we want to prove
theorem last_two_digits : sum_factorials_mod_100 = 21 := 
by 
  -- The start of proof, putting sorry as a placeholder
  sorry

end last_two_digits_l427_427354


namespace regression_correct_expectation_correct_variance_correct_l427_427753

-- Conditions
def mean_x := 4.36
def mean_y := 220.4
def mean_x2 := 19.0
def mean_xy := 961
def sum_xy := 14612.3
def sum_x2 := 286.84
def n := 15 -- Number of data points

-- Given regression formulas
def b_hat := (sum_xy - n * mean_x * mean_y) / (sum_x2 - n * mean_x ^ 2)
def a := mean_y - b_hat * mean_x
def y_hat (x : ℝ) := b_hat * x + a

-- Expected results
def expected_y_regression (x : ℝ) : Prop :=
  y_hat(x) ≈ 107.2 * x - 247.0

-- For Part 2
def p := 0.9
def q := 1 - p
def X_expectation := 3 * p
def X_variance := 3 * p * q

-- Expected results for Part 2
def expected_expectation : Prop := X_expectation = 2.7
def expected_variance : Prop := X_variance = 0.27

-- Theorems to be proved
theorem regression_correct (x : ℝ) : expected_y_regression(x) :=
  by sorry

theorem expectation_correct : expected_expectation :=
  by sorry

theorem variance_correct : expected_variance :=
  by sorry

end regression_correct_expectation_correct_variance_correct_l427_427753


namespace geometric_seq_gen_term_sum_inequality_l427_427819

-- Definitions based on given conditions
def seq (a : ℕ → ℝ) := ∀ n, a (n + 1) = 4 * a n + 1
def init_cond (a : ℕ → ℝ) := a 1 = 1

-- Part (I)
theorem geometric_seq (a : ℕ → ℝ) (h_seq : seq a) (h_init : init_cond a) :
  ∃ b : ℕ → ℝ, (∀ n, b (n + 1) = 4 * b n) ∧ b 1 = 4 / 3 ∧ (∀ n, a n + 1 / 3 = b n) 
:= sorry

-- General term formula
theorem gen_term (a : ℕ → ℝ) (h_seq : seq a) (h_init : init_cond a) :
  ∀ n, a n = (4 ^ n - 1) / 3 
:= sorry

-- Part (II)
theorem sum_inequality (a : ℕ → ℝ) (h_seq : seq a) (h_init : init_cond a) (h_gen_term : ∀ n, a n = (4 ^ n - 1) / 3) :
  ∀ n, (∑ k in Finset.range n, 1 / a (k + 1)) < 4 / 3 
:= sorry

end geometric_seq_gen_term_sum_inequality_l427_427819


namespace mike_took_23_green_marbles_l427_427721

-- Definition of the conditions
def original_green_marbles : ℕ := 32
def remaining_green_marbles : ℕ := 9

-- Definition of the statement we want to prove
theorem mike_took_23_green_marbles : original_green_marbles - remaining_green_marbles = 23 := by
  sorry

end mike_took_23_green_marbles_l427_427721


namespace largest_value_of_n_l427_427350

theorem largest_value_of_n (A B n : ℤ) (h1 : A * B = 72) (h2 : n = 6 * B + A) : n = 433 :=
sorry

end largest_value_of_n_l427_427350


namespace opposite_of_neg_two_l427_427983

theorem opposite_of_neg_two : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_two_l427_427983


namespace prime_number_form_101_of_alternating_digits_l427_427324

theorem prime_number_form_101_of_alternating_digits :
  ∀ (A : ℕ) (n : ℕ) (q : ℕ), (q = 10) ∧ (A = ∑ k in Finset.range(n+1), q^(2 * k)) →
  Prime 101 →
  A = 101 :=
begin
  sorry
end

end prime_number_form_101_of_alternating_digits_l427_427324


namespace percentage_of_passengers_in_first_class_l427_427930

theorem percentage_of_passengers_in_first_class (total_passengers : ℕ) (percentage_female : ℝ) (females_coach : ℕ) 
  (males_perc_first_class : ℝ) (Perc_first_class : ℝ) : 
  total_passengers = 120 → percentage_female = 0.45 → females_coach = 46 → males_perc_first_class = (1/3) → 
  Perc_first_class = 10 := by
  sorry

end percentage_of_passengers_in_first_class_l427_427930


namespace max_minute_hands_l427_427228

theorem max_minute_hands (m n : ℕ) (h : m * n = 27) : m + n ≤ 28 :=
  sorry

end max_minute_hands_l427_427228


namespace area_PQR_is_216_l427_427447

variable (P Q R M N S : Type)
variable [Inhabited P] [Inhabited Q] [Inhabited R] [Inhabited M] [Inhabited N] [Inhabited S]
variables (area : ∀ {A B C : Type}, Type)

-- Conditions
axiom midpoint_M : M = midpoint Q R
axiom ratio_PN_NR : ∃ (PR : Line P R), N ∈ PR ∧ PN / NR = 1 / 3
axiom ratio_PS_SM : ∃ (PM : Line P M), S ∈ PM ∧ PS / SM = 2 / 1
axiom area_MNS : area M N S = 12

-- Statement to prove
theorem area_PQR_is_216 : area P Q R = 216 :=
sorry

end area_PQR_is_216_l427_427447


namespace min_distance_point_circle_to_line_l427_427744

/-- The minimum distance from a point on the circle x^2 + y^2 = 4 to the line 3x + 4y - 25 = 0 is 3. -/
theorem min_distance_point_circle_to_line : 
  let circle := (x y : ℝ) → x^2 + y^2 = 4
  let line := (x y : ℝ) → 3 * x + 4 * y - 25 = 0
  ∃ (p : ℝ × ℝ), circle p.1 p.2 ∧ ∃ (q : ℝ × ℝ), line q.1 q.2 ∧ 
  ∀ (p : ℝ × ℝ), circle p.1 p.2 → (dist p (0, 0) - dist (0, 0) (3, 4) = 3) :=
begin
  sorry
end

end min_distance_point_circle_to_line_l427_427744


namespace solution_set_l427_427395

noncomputable def satisfies_inequality (f : ℝ → ℝ) :=
  even_function (f : ℝ → ℝ) ∧ (∀ x > 0, (deriv f x) < 0) ∧ continuous f

theorem solution_set (f : ℝ → ℝ) (h : satisfies_inequality f) :
  {x : ℝ | f (real.log x) > f 1} = set.Ioo (1 / 10) 10 :=
sorry

end solution_set_l427_427395


namespace cell_value_last_digit_l427_427571

theorem cell_value_last_digit :
  let table := Array.init 2021 (λ i => Array.init 2021 (λ j => 0)) in
  let cell := λ (x y : Nat), 2 ^ (x + y - 2) - 1 in
  (cell 2021 2021) % 10 = 5 := 
by
  -- The detailed proof is omitted
  sorry

end cell_value_last_digit_l427_427571


namespace pqrs_sum_l427_427514

theorem pqrs_sum (p q r s : ℝ)
  (h1 : (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 → x = r ∨ x = s))
  (h2 : (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 → x = p ∨ x = q))
  (h3 : p ≠ q) (h4 : p ≠ r) (h5 : p ≠ s) (h6 : q ≠ r) (h7 : q ≠ s) (h8 : r ≠ s) :
  p + q + r + s = 2028 :=
sorry

end pqrs_sum_l427_427514


namespace max_tan_theta_l427_427485

-- Given integers with distinct absolute values
variables {a1 a2 a3 a4 : ℤ}
-- Given points on the coordinate plane
def A1 : ℤ × ℤ := (a1, a1 ^ 2)
def A2 : ℤ × ℤ := (a2, a2 ^ 2)
def A3 : ℤ × ℤ := (a3, a3 ^ 2)
def A4 : ℤ × ℤ := (a4, a4 ^ 2)

noncomputable def slope (p1 p2 : ℤ × ℤ) : ℚ :=
  (p2.2 - p1.2 : ℚ) / (p2.1 - p1.1)

-- Define the slopes
def m1 := slope A1 A2
def m2 := slope A3 A4

-- Define the tangent of the angle of intersection
def tan_theta : ℚ :=
  |(m1 - m2) / (1 + m1 * m2)|

-- Main theorem to prove
theorem max_tan_theta :
  Int.gcd 5 3 = 1 → -- Relatively prime condition
  ∀ {a1 a2 a3 a4 : ℤ},
    (a1 ≠ a2) → (a1 ≠ a3) → (a1 ≠ a4) → (a2 ≠ a3) → (a2 ≠ a4) → (a3 ≠ a4) →
    (|a1| ≠ |a2|) → (|a1| ≠ |a3|) → (|a1| ≠ |a4|) → (|a2| ≠ |a3|) → (|a2| ≠ |a4|) → (|a3| ≠ |a4|) →
    tan_theta = 5 / 3 :=
sorry

end max_tan_theta_l427_427485


namespace tiles_in_row_l427_427124

def room_area : ℝ := 288 -- Area in square feet
def ratio_length_width : ℝ := 2 -- Length is twice the width

-- Convert feet to inches
def feet_to_inches (x : ℝ) : ℝ := x * 12

def number_of_tiles (area : ℝ) (ratio : ℝ) : ℝ :=
  let w := real.sqrt (area / ratio)
  let width_in_inches := feet_to_inches w
  width_in_inches / 4 -- Tile width in inches

theorem tiles_in_row :
  number_of_tiles room_area ratio_length_width = 36 :=
by
  sorry

end tiles_in_row_l427_427124


namespace problem_statement_l427_427162

open Set

def U := ℕ
def A := {x | ∃ n : ℕ, x = 2 * n}
def B := {x | ∃ n : ℕ, x = 4 * n}

theorem problem_statement : A ∪ (U \ B) = U := by
  sorry

end problem_statement_l427_427162


namespace perfect_square_n_l427_427771

def is_perfect_square (x : ℕ) : Prop :=
  ∃ y : ℕ, y * y = x

theorem perfect_square_n (n : ℕ) : 
  is_perfect_square (nat.factorial 1 * nat.factorial 2 * nat.factorial 3 * 
    ((finset.range (2 * n + 1).succ).filter nat.even).prod (!.) * 
    nat.factorial (2 * n)) ∧ (n = 4 * k * (k + 1) ∨ n = 2 * k ^ 2 - 1) :=
by
  sorry

end perfect_square_n_l427_427771


namespace opposite_neg_two_l427_427978

theorem opposite_neg_two : -(-2) = 2 := by
  sorry

end opposite_neg_two_l427_427978


namespace part_I_part_II_part_III_l427_427809

-- Define the function f(x)
def f (x a : ℝ) : ℝ := x - 1 + a / exp x

-- Problem Part I: Prove a = e for the tangent line to be parallel to the x-axis at (1, f(1))
theorem part_I (a : ℝ) : (1 - a / exp 1 = 0) → (a = Real.exp 1) := by
  intro h
  have : 1 - a / Real.exp 1 = 0 := h
  sorry

-- Problem Part II: Prove the extremum results of the function f based on the value of a
theorem part_II (a : ℝ) : 
  (a ≤ 0 → (∀ x, f x a ≠ (ℝ.minor f (f(1 a))))  ∧ ∀ x, f x a ≠ (ℝ.major f (f(1 a))))  ∧
  (a > 0 → 
    ((∃ (x : ℝ), x = Real.log a ∧ f x a = Real.log a) ∧ 
     ∀ (x1 : ℝ), (x1 < Real.log a → f x1 a > f (Real.log a) a) ∧ (x1 > Real.log a → f x1 a > f (Real.log a) a))) := by
  sorry

-- Problem Part III: Prove the maximum value of k is 1 when a = 1, and the line y = kx - 1 does not intersect the curve
theorem part_III (k : ℝ) : 
  (∀ (x : ℝ), k ≠ 1 → ((f x 1 ≠ k * x - 1)) → (k ≤ 1)) := by
  intro x
  intro h1
  intro h2
  have : k = 1
  sorry

end part_I_part_II_part_III_l427_427809


namespace sin_sub_cos_l427_427387

theorem sin_sub_cos (A : ℝ) (hA : 0 < A ∧ A < π) (h1 : sin A + cos A = 3/5) : 
  sin A - cos A = sqrt 41 / 5 := 
sorry

end sin_sub_cos_l427_427387


namespace kenneth_money_left_l427_427896

theorem kenneth_money_left (I : ℕ) (C_b : ℕ) (N_b : ℕ) (C_w : ℕ) (N_w : ℕ) (L : ℕ) :
  I = 50 → C_b = 2 → N_b = 2 → C_w = 1 → N_w = 2 → L = I - (N_b * C_b + N_w * C_w) → L = 44 :=
by
  intros h₀ h₁ h₂ h₃ h₄ h₅
  sorry

end kenneth_money_left_l427_427896


namespace determine_constants_l427_427309

theorem determine_constants :
  ∃ P Q R,
    (∀ x, x ≠ 4 ∧ x ≠ 2 → 
       5 * x ^ 2 / ((x - 4) * (x - 2) ^ 3) = 
       P / (x - 4) + Q / (x - 2) + R / (x - 2) ^ 3)
    ∧ P = 10 ∧ Q = -10 ∧ R = -10 :=
by
  use 10, -10, -10
  intros x hx
  sorry

end determine_constants_l427_427309


namespace sum_distinct_vars_eq_1716_l427_427510

open Real

theorem sum_distinct_vars_eq_1716 (p q r s : ℝ) (hpqrs_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : r + s = 12 * p)
  (h2 : r * s = -13 * q)
  (h3 : p + q = 12 * r)
  (h4 : p * q = -13 * s) :
  p + q + r + s = 1716 :=
sorry

end sum_distinct_vars_eq_1716_l427_427510


namespace cartesian_eq_of_parametric_eq_cartesian_line_eq_of_polar_eq_range_of_m_l427_427458

-- Definitions from conditions
def parametric_eq (α : ℝ) : ℝ × ℝ :=
  (1 + 3 * Real.cos α, 3 + 3 * Real.sin α)

def polar_eq (ρ θ m : ℝ) : Prop :=
  ρ * Real.cos θ + ρ * Real.sin θ = m

-- Theorem to be proven
theorem cartesian_eq_of_parametric_eq (α : ℝ) :
  let (x, y) := parametric_eq α in (x - 1) ^ 2 + (y - 3) ^ 2 = 9 :=
sorry

theorem cartesian_line_eq_of_polar_eq (ρ θ m : ℝ) (h : polar_eq ρ θ m) :
  let x := ρ * Real.cos θ
  let y := ρ * Real.sin θ in
  x + y = m :=
sorry

theorem range_of_m (m : ℝ) :
  ∀ x y : ℝ, (x - 1) ^ 2 + (y - 3) ^ 2 = 9 ∧ (x + y = m) →
  4 - 3 * Real.sqrt 2 < m ∧ m < 4 + 3 * Real.sqrt 2 :=
sorry

end cartesian_eq_of_parametric_eq_cartesian_line_eq_of_polar_eq_range_of_m_l427_427458


namespace min_workers_to_complete_project_on_schedule_l427_427306

theorem min_workers_to_complete_project_on_schedule
  (total_days : ℕ)
  (initial_workers : ℕ)
  (portion_completed : ℚ)
  (time_passed : ℕ)
  (project_total : ℚ)
  (worker_rate : ℚ)
  (remaining_days : ℕ)
  (required_rate : ℚ)
  (needed_workers : ℕ) :

  total_days = 40 →
  initial_workers = 10 →
  portion_completed = 2/5 →
  time_passed = 10 →
  project_total = 1 →
  worker_rate = portion_completed / (initial_workers * time_passed) →
  remaining_days = total_days - time_passed →
  required_rate = (project_total - portion_completed) / remaining_days →
  needed_workers = required_rate / worker_rate →
  needed_workers = 5 :=

by
  intros
  unfold1 worker_rate
  unfold1 required_rate
  sorry

end min_workers_to_complete_project_on_schedule_l427_427306


namespace men_women_in_group_l427_427274

theorem men_women_in_group (group_size elderly_men possible_ways : ℕ) (h : group_size = 18) (e : elderly_men = 2) (p : possible_ways = 64) :
  let x := 10 in
  x = 10 ∧ (group_size - x) = 8 := 
by
  simp [h, e, p]
  sorry

end men_women_in_group_l427_427274


namespace log_product_is_five_sixths_l427_427710

open Real

theorem log_product_is_five_sixths :
  log 3 / log 8 * log 32 / log 9 = 5 / 6 := 
begin
  suffices : (log 3 / (3 * log 2) * (5 * log 2) / (2 * log 3)) = 5 / 6,
  {
    calc
      log 3 / log 8 * log 32 / log 9 = log 3 / (log 2 ^ 3) * log 32 / log 9   : by rw [log_base_change, log_base_change]
                                  ...  = log 3 / (3 * log 2) * log (2 ^ 5) / log (3 ^ 2) : by rw [log_pow, log (3 ^ 2)]
                                  ...  = log 3 / (3 * log 2) * 5 * log 2 / 2 * log 3    : by rw [log_pow (2^5), log (3 ^ 2)]
                                  ...  = (log 3 / (3 * log 2)) * (5 * log 2) / (2 * log 3)  : by ring 
                                  ...  = 5 / 6 : by exact this 
  },
  
  exact calc
    (log 3 / (3 * log 2) * (5 * log 2) / (2 * log 3)) = (log 3 * 5 * log 2) / (3 * log 2 * 2 * log 3)   : by { field_simp }
    ...                                               = 5 / 6                                        : by { field_simp }
end

end log_product_is_five_sixths_l427_427710


namespace triangle_sin_C_l427_427032

/-- Given a triangle ABC where angle A is π/4 and cos B is √10/10,
    prove that sin C equals 2√5/5. -/
theorem triangle_sin_C (A B C : ℝ) (hA : A = π / 4) (hCosB : cos B = sqrt 10 / 10) :
  sin C = 2 * sqrt 5 / 5 :=
by
  sorry

end triangle_sin_C_l427_427032


namespace gcd_12012_21021_l427_427740

-- Definitions
def factors_12012 : List ℕ := [2, 2, 3, 7, 11, 13] -- Factors of 12,012
def factors_21021 : List ℕ := [3, 7, 7, 11, 13] -- Factors of 21,021

def common_factors := [3, 7, 11, 13] -- Common factors between 12,012 and 21,021

def gcd (ls : List ℕ) : ℕ :=
ls.foldr Nat.gcd 0 -- Function to calculate gcd of list of numbers

-- Main statement
theorem gcd_12012_21021 : gcd common_factors = 1001 := by
  -- Proof is not required, so we use sorry to skip the proof.
  sorry

end gcd_12012_21021_l427_427740


namespace emma_list_count_l427_427734

theorem emma_list_count : 
  let m1 := 900
  let m2 := 27000
  let d := 30
  (m1 / d <= m2 / d) → (m2 / d - m1 / d + 1 = 871) :=
by
  intros m1 m2 d h
  have h1 : m1 / d ≤ m2 / d := h
  have h2 : m2 / d - m1 / d + 1 = 871 := by sorry
  exact h2

end emma_list_count_l427_427734


namespace probability_female_likes_running_probability_male_likes_running_relationship_with_gender_l427_427291

def total_students := 200
def total_boys := 120
def boys_not_like_running := 50
def girls_like_running := 30

def total_girls := total_students - total_boys
def boys_like_running := total_boys - boys_not_like_running

def prob_female_likes_running := girls_like_running / total_girls.toRat
def prob_male_likes_running := boys_like_running / total_boys.toRat

theorem probability_female_likes_running :
  prob_female_likes_running = (3/8 : ℚ) := sorry

theorem probability_male_likes_running :
  prob_male_likes_running = (7/12 : ℚ) := sorry

def K_squared : ℚ := (total_students * (boys_like_running * boys_not_like_running - girls_like_running * boys_not_like_running) ^ 2) / 
                   ((total_boys * (total_students - total_boys) * (boys_like_running + girls_like_running) * (boys_not_like_running + girls_like_running)).toRat)

def critical_value_99 : ℚ := 6.635

theorem relationship_with_gender :
  K_squared > critical_value_99 := sorry

end probability_female_likes_running_probability_male_likes_running_relationship_with_gender_l427_427291


namespace problem_statement_l427_427794

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry
noncomputable def f'' : ℝ → ℝ := sorry

def a := f 1 - 1
def b := - (1 / 2) * f (-2) - 4
def c := f 0 - 1

theorem problem_statement :
  is_odd f →
  (∀ x ≠ 0, f'' x > 2 * x^2 + f x / x) →
  a < b :=
by
  intros h1 h2
  sorry

end problem_statement_l427_427794


namespace instantaneous_velocity_at_2_l427_427588

-- Define the motion equation
def s (t : ℝ) : ℝ := 3 + t^2

-- State the problem: Prove the instantaneous velocity at t = 2 is 4
theorem instantaneous_velocity_at_2 : (deriv s) 2 = 4 := by
  sorry

end instantaneous_velocity_at_2_l427_427588


namespace frustum_radius_l427_427128

theorem frustum_radius (r : ℝ) (h1 : ∃ r1 r2, r1 = r 
                                  ∧ r2 = 3 * r 
                                  ∧ r1 * 2 * π * 3 = r2 * 2 * π
                                  ∧ (lateral_area = 84 * π)) (h2 : slant_height = 3) : 
  r = 7 :=
sorry

end frustum_radius_l427_427128


namespace impossible_sign_replacement_l427_427888

theorem impossible_sign_replacement :
  ¬ (∃ a b c d e f g : ℤ,
    (| (420 * a + 280 * b + 210 * c + 168 * d + 140 * e + 120 * f + 105 * g) / 840 | < 1 / 500)) := 
sorry

end impossible_sign_replacement_l427_427888


namespace difference_between_faruk_and_ranjiths_shares_is_2000_l427_427284

def ratio_parts : ℕ × ℕ × ℕ := (3, 3, 7)
def vasim_share : ℕ := 1500

theorem difference_between_faruk_and_ranjiths_shares_is_2000 
  (ratios : ℕ × ℕ × ℕ) (v_share : ℕ) 
  (h_ratio : ratios = ratio_parts) 
  (h_vasim : v_share = vasim_share) : 
  let value_of_one_part := (v_share / ratios.2) in
  let faruk_share := ratios.1 * value_of_one_part in
  let ranjith_share := ratios.3 * value_of_one_part in
  (ranjith_share - faruk_share) = 2000 := 
by
  sorry

end difference_between_faruk_and_ranjiths_shares_is_2000_l427_427284
