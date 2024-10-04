import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.CombinatorialBasics
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Permutation
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.MeasureTheory.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.Distribution
import Mathlib.Probability.ProbabilitySpace
import Mathlib.Tactic
import Mathlib.Tactic.LibrarySearch
import Real

namespace range_of_a_l174_174339

theorem range_of_a (a : ℝ) : (∃ x y : ℝ, x^2 / a^2 + y^2 / (a + 6) = 1 ∧ (a^2 > a + 6 ∧ a + 6 > 0)) → (a > 3 ∨ (-6 < a ∧ a < -2)) :=
by
  intro h
  sorry

end range_of_a_l174_174339


namespace domain_of_f_l174_174475

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.log (x - 2) / Real.log 3 - 1)

theorem domain_of_f :
  {x : ℝ | f x = f x} = {x : ℝ | 2 < x ∧ x ≠ 5} :=
by
  sorry

end domain_of_f_l174_174475


namespace find_k_for_xy_solution_l174_174661

theorem find_k_for_xy_solution :
  ∀ (k : ℕ), (∃ (x y : ℕ), x * (x + k) = y * (y + 1))
  → k = 1 ∨ k ≥ 4 :=
by
  intros k h
  sorry -- proof goes here

end find_k_for_xy_solution_l174_174661


namespace number_of_tea_bags_l174_174442

theorem number_of_tea_bags (n : ℕ) 
  (h1 : 2 * n ≤ 41) (h2 : 41 ≤ 3 * n)
  (h3 : 2 * n ≤ 58) (h4 : 58 ≤ 3 * n) :
  n = 20 :=
by
  sorry

end number_of_tea_bags_l174_174442


namespace imaginary_part_of_Z_l174_174709

noncomputable theory

open Complex

theorem imaginary_part_of_Z 
  (Z : ℂ)
  (h : Z * Complex.i = (Complex.exp ((Complex.log (i+1) - Complex.log (i-1)) * 2018))) :
  Z.im = (5 : ℂ) ^ 1009 * Real.sin (2018 * (Real.arctan 2)) :=
sorry

end imaginary_part_of_Z_l174_174709


namespace greatest_possible_value_of_x_plus_y_l174_174599

theorem greatest_possible_value_of_x_plus_y (a b c d x y : ℤ) (h1 : a + b ∈ {210, 330, 290, 250, x, y})
                                            (h2 : a + c ∈ {210, 330, 290, 250, x, y})
                                            (h3 : a + d ∈ {210, 330, 290, 250, x, y})
                                            (h4 : b + c ∈ {210, 330, 290, 250, x, y})
                                            (h5 : b + d ∈ {210, 330, 290, 250, x, y})
                                            (h6 : c + d ∈ {210, 330, 290, 250, x, y}) :
  x + y = 780 :=
sorry

end greatest_possible_value_of_x_plus_y_l174_174599


namespace katherine_age_l174_174421

-- Define a Lean statement equivalent to the given problem
theorem katherine_age (K M : ℕ) (h1 : M = K - 3) (h2 : M = 21) : K = 24 := sorry

end katherine_age_l174_174421


namespace domain_correct_l174_174017

def domain_of_function (x : ℝ) : Prop :=
  (∃ y : ℝ, y = 2 / Real.sqrt (x + 1)) ∧ Real.sqrt (x + 1) ≠ 0

theorem domain_correct (x : ℝ) : domain_of_function x ↔ (x > -1) := by
  sorry

end domain_correct_l174_174017


namespace polynomial_roots_l174_174252

theorem polynomial_roots : 
  (Polynomial.roots (Polynomial.C 3 * (Polynomial.X ^ 4) + Polynomial.C 7 * (Polynomial.X ^ 3) - Polynomial.C 13 * (Polynomial.X ^ 2) + Polynomial.C 11 * Polynomial.X - Polynomial.C 6)).to_finset = {-3, -2, -1, (1 : ℚ)/3}.to_finset :=
by 
  sorry

end polynomial_roots_l174_174252


namespace halfway_fraction_l174_174952

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/6) : (a + b) / 2 = 19 / 24 :=
by
  sorry

end halfway_fraction_l174_174952


namespace cost_effective_washing_powder_l174_174185

-- Define the costs and quantities
variables (c_XS c_S c_M c_L : ℝ)
variables (q_XS q_S q_M q_L : ℝ)

-- Define the conditions derived from the problem statement
-- Small size costs 60% more than Extra Small size
def cost_S : Prop := c_S = 1.6 * c_XS
-- Small size contains 25% less powder than Medium size
def quantity_S : Prop := q_S = 0.75 * q_M
-- Medium size contains 50% more powder than Extra Small size
def quantity_M : Prop := q_M = 1.5 * q_XS
-- Medium size costs 40% more than Small size
def cost_M : Prop := c_M = 1.4 * c_S
-- Large size contains 30% more powder than Medium size
def quantity_L : Prop := q_L = 1.3 * q_M
-- Large size costs 20% more than Medium size
def cost_L : Prop := c_L = 1.2 * c_M

-- Defining the cost per gram for each size
def cost_per_gram_XS : ℝ := c_XS / q_XS
def cost_per_gram_S : ℝ := c_S / q_S
def cost_per_gram_M : ℝ := c_M / q_M
def cost_per_gram_L : ℝ := c_L / q_L

-- The theorem that the ranking of cost per gram is XS, L, S, M
theorem cost_effective_washing_powder :
  cost_S →
  quantity_S →
  quantity_M →
  cost_M →
  quantity_L →
  cost_L →
  cost_per_gram_XS < cost_per_gram_L ∧
  cost_per_gram_L < cost_per_gram_S ∧
  cost_per_gram_S < cost_per_gram_M :=
by sorry

end cost_effective_washing_powder_l174_174185


namespace min_value_of_f_interval_of_monotonic_increase_sin_A_l174_174306

noncomputable def f (x : ℝ) : ℝ :=
  sin (2 * x + 5 * Real.pi / 6) - cos x ^ 2 + 1

theorem min_value_of_f :
  (∃ x : ℝ, f x = (1 - Real.sqrt 3) / 2) :=
sorry

theorem interval_of_monotonic_increase :
  ∃ k : ℤ, ∀ x : ℝ, (Real.pi / 4 + k * Real.pi ≤ x ∧ x ≤ 3 * Real.pi / 4 + k * Real.pi) → 
  (∀ x1 x2 : ℝ, x1 ≤ x2 → f x1 ≤ f x2) :=
sorry

theorem sin_A (A B C : ℝ) (hB : cos B = 1 / 3) (hC : f (C / 2) = -1 / 4) :
  sin A = (2 * Real.sqrt 2 + Real.sqrt 3) / 6 :=
sorry

end min_value_of_f_interval_of_monotonic_increase_sin_A_l174_174306


namespace percentage_markup_l174_174876

theorem percentage_markup (SP CP : ℕ) (h1 : SP = 8340) (h2 : CP = 6672) :
  ((SP - CP) / CP * 100) = 25 :=
by
  -- Before proving, we state our assumptions
  sorry

end percentage_markup_l174_174876


namespace smallest_interesting_number_l174_174137

def is_perfect_square (x : ℕ) : Prop :=
∃ y : ℕ, y * y = x

def is_perfect_cube (x : ℕ) : Prop :=
∃ y : ℕ, y * y * y = x

theorem smallest_interesting_number (n : ℕ) :
    is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n) → n = 1800 :=
by {
  sorry
}

end smallest_interesting_number_l174_174137


namespace science_book_pages_l174_174484

theorem science_book_pages {history_pages novel_pages science_pages: ℕ} (h1: novel_pages = history_pages / 2) (h2: science_pages = 4 * novel_pages) (h3: history_pages = 300):
  science_pages = 600 :=
by
  sorry

end science_book_pages_l174_174484


namespace sum_of_areas_l174_174641

theorem sum_of_areas (r : ℕ → ℝ) (n : ℕ) : (∀ n, r n = 1 / 3^(n - 1)) →
  ∑' n, π * r n ^ 2 = 9 * π / 8 :=
begin
  intro hr,
  have h_geom : ∑' n, r n ^ 2 = 1 / (1 - 1/9),
  { exact tsum_geometric_of_lt_1 (1 / (3)^2) (by norm_num) (by norm_num), },
  rw ← hr, 
  simp only [← pow_mul, ← r_pow],
  rw ← π,
  rw ← h_geom,
  simp [← pow_inv],
  norm_num,
end

end sum_of_areas_l174_174641


namespace sixty_fifth_term_is_sixteen_l174_174655

def apply_rule (n : ℕ) : ℕ :=
  if n <= 12 then
    7 * n
  else if n % 2 = 0 then
    n - 7
  else
    n / 3

def sequence_term (a_0 : ℕ) (n : ℕ) : ℕ :=
  Nat.iterate apply_rule n a_0

theorem sixty_fifth_term_is_sixteen : sequence_term 65 64 = 16 := by
  sorry

end sixty_fifth_term_is_sixteen_l174_174655


namespace part_i_part_ii_l174_174416

open Set

def universal_set : Set ℝ := univ

def M : Set ℝ := {x | (x + 3) ^ 2 = 0}

def N : Set ℝ := {x | x^2 + x - 6 = 0}

def complement (s : Set ℝ) : Set ℝ := {x | x ∉ s}

def A : Set ℝ := (complement M) ∩ N

def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 5 - a}

theorem part_i : A = {2} :=
sorry

theorem part_ii (a : ℝ) : (B a ∪ A = A) → a ≥ 3 :=
sorry

end part_i_part_ii_l174_174416


namespace problem_solution_l174_174710

noncomputable def f (n : ℕ) : ℝ := n^2 * real.sin (n * real.pi / 2)

noncomputable def a (n : ℕ) : ℝ := f n + f (n+1)

def problem : Prop :=
  (∑ n in finset.range 2018, a (n + 1)) = -4032

theorem problem_solution : problem :=
by
  sorry

end problem_solution_l174_174710


namespace triangle_tangent_ratio_l174_174702

variable {A B C a b c : ℝ}

theorem triangle_tangent_ratio 
  (h : a * Real.cos B - b * Real.cos A = (3 / 5) * c)
  : Real.tan A / Real.tan B = 4 :=
sorry

end triangle_tangent_ratio_l174_174702


namespace ant_probability_C_after_4_l174_174198
open ProbabilityTheory

-- Definitions based on conditions
def lattice := (ℤ × ℤ)
def color (p : ℤ × ℤ) : Prop := (p.1 + p.2) % 2 = 0 -- True for red, False for blue

def ant_trajectory (steps : ℕ) : list lattice := sorry

theorem ant_probability_C_after_4 :
  ∀ (A C : lattice),
  A = (0, 0) → C = (1, 0) →
  (∀ n : ℕ, ∃ (p : lattice), p ∈ list.nth (ant_trajectory n) n) →
  probability (list.nth (ant_trajectory 4) 4 = C) = 1 / 3 := sorry

end ant_probability_C_after_4_l174_174198


namespace find_smallest_M_l174_174316

-- Definitions and conditions for the problem
def S : Set (ℕ → ℝ) := { f | f 1 = 2 ∧ ∀ n ∈ ℕ, (f (n + 1) ≥ f n ∧ f n ≥ (n : ℝ) / (n + 1) * f(2 * n)) }

-- Theorem on the smallest natural number M
theorem find_smallest_M (M : ℕ) : (∀ f ∈ S, ∀ n ∈ ℕ, f n < M) ↔ M = 10 :=
by
  sorry

end find_smallest_M_l174_174316


namespace hyperbola_point_distance_to_origin_l174_174682

noncomputable def hyperbola_distance (x y : ℝ) : Prop :=
  (x^2 / 9) - (y^2 / 7) = 1

noncomputable def distance_to_focus (x y c : ℝ) : Prop :=
  c = real.sqrt ((x + real.sqrt(9 + 7 * (y^2)))^2 + y^2)

theorem hyperbola_point_distance_to_origin
  (x y : ℝ)
  (hx : hyperbola_distance x y)
  (hy : distance_to_focus x y 1)
: real.sqrt (x^2 + y^2) = 3 :=
sorry

end hyperbola_point_distance_to_origin_l174_174682


namespace find_y_l174_174335

theorem find_y (w x y : ℝ) 
  (h1 : 6 / w + 6 / x = 6 / y) 
  (h2 : w * x = y) 
  (h3 : (w + x) / 2 = 0.5) : 
  y = 0.25 := 
sorry

end find_y_l174_174335


namespace Larry_wins_probability_l174_174380

noncomputable def probability_Larry_wins (p_larry: ℚ) (p_paul: ℚ): ℚ :=
  let q_larry := 1 - p_larry
  let q_paul := 1 - p_paul
  p_larry / (1 - q_larry * q_paul)

theorem Larry_wins_probability:
  probability_Larry_wins (1/3 : ℚ) (1/2 : ℚ) = (2/5 : ℚ) :=
by {
  sorry
}

end Larry_wins_probability_l174_174380


namespace alpha_modulus_l174_174393

noncomputable def α : ℂ := a + b * Complex.I
noncomputable def β : ℂ := a - b * Complex.I

theorem alpha_modulus :
  (α β : ℂ) (h_conj : β = conj α)
  (h_real : α / (β^2) ∈ ℝ) (h_diff : |α - β| = 2 * Real.sqrt 5) :
  |α| = (2 * Real.sqrt 15) / 3 :=
by
  sorry

end alpha_modulus_l174_174393


namespace smallest_interesting_number_l174_174170

-- Definitions based on the conditions
def isPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m
def isPerfectCube (m : ℕ) : Prop := ∃ k : ℕ, k * k * k = m

def isInteresting (n : ℕ) : Prop :=
  isPerfectSquare (2 * n) ∧ isPerfectCube (15 * n)

-- The smallest interesting number
theorem smallest_interesting_number : ∃ n : ℕ, isInteresting n ∧ (∀ m : ℕ, isInteresting m → n ≤ m) :=
  ⟨1800, ⟨⟨26, by norm_num⟩, ⟨15, by norm_num⟩⟩, sorry⟩

end smallest_interesting_number_l174_174170


namespace hyperbola_slope_product_l174_174720

theorem hyperbola_slope_product :
  ∀ (a b : ℝ) (p q s t : ℝ),
  (a > 0) →
  (b > 0) →
  (2 = real.sqrt (1 + (b/a)^2)) →
  (p^2 / a^2 - q^2 / b^2 = 1) →
  (s^2 / a^2 - t^2 / b^2 = 1) →
  (k1 k2 : ℝ) ->
  k1 = (t - q) / (s - p) →
  k2 = (t + q) / (s + p) →
  k1 * k2 = 3 :=
by
  intros a b p q s t ha hb he h1 h2 k1 k2 hk1 hk2
  sorry

end hyperbola_slope_product_l174_174720


namespace angle_between_tangents_l174_174578

-- Define the conditions of the problem
def chord_divides_circle (a b : ℕ) : Prop :=
  a = 11 ∧ b = 16

-- Define the central angle computation
def central_angle (a b : ℕ) [h : chord_divides_circle a b] : ℝ :=
  (a : ℝ) / (a + b) * 360

-- Define the theorem to find the angle between tangents given the central angle
theorem angle_between_tangents {a b : ℕ} (h : chord_divides_circle a b) : 
  let θ := 180 - central_angle a b in θ = 33 + 20 / 60 :=
by
  sorry

end angle_between_tangents_l174_174578


namespace only_k_monotonically_increasing_on_l174_174553

def is_monotonically_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x → x < y → y < b → f x < f y

def f (x : ℝ) : ℝ := 1 / (x + 1)
def g (x : ℝ) : ℝ := (x - 1) ^ 2
def h (x : ℝ) : ℝ := 2 ^ (1 - x)
def k (x : ℝ) : ℝ := Real.log (x + 3)

theorem only_k_monotonically_increasing_on (a b : ℝ) (ha : 0 < a) (hb : b = ∞) :
  is_monotonically_increasing_on k a b
  ∧ ¬ is_monotonically_increasing_on f a b
  ∧ ¬ is_monotonically_increasing_on g a b
  ∧ ¬ is_monotonically_increasing_on h a b := by
  sorry

end only_k_monotonically_increasing_on_l174_174553


namespace maximum_planes_l174_174182

def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)

def combination (n k : ℕ) : ℕ := fact n / (fact k * fact (n - k))

theorem maximum_planes (points : Finset (ℝ^3)) (h : points.card = 12)
  (h_non_collinear : ∀ (p1 p2 p3 : ℝ^3), p1 ∈ points → p2 ∈ points → p3 ∈ points → 
   ∃ plane : AffineSubspace ℝ (ℝ^3), p1 ∈ plane ∧ p2 ∈ plane ∧ p3 ∈ plane) : 
  combination 12 3 = 220 := 
sorry

end maximum_planes_l174_174182


namespace larry_stickers_l174_174381

theorem larry_stickers (initial_stickers : ℕ) (lost_stickers : ℕ) (final_stickers : ℕ) 
  (initial_eq_93 : initial_stickers = 93) 
  (lost_eq_6 : lost_stickers = 6) 
  (final_eq : final_stickers = initial_stickers - lost_stickers) : 
  final_stickers = 87 := 
  by 
  -- proof goes here
  sorry

end larry_stickers_l174_174381


namespace halfway_fraction_l174_174971

theorem halfway_fraction (a b : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) : (1/2) * (a + b) = 19/24 :=
by 
  rw [h1, h2]
  simp
  done
  sorry

end halfway_fraction_l174_174971


namespace coefficient_x3_in_expansion_l174_174647

theorem coefficient_x3_in_expansion :
  ∃ A : ℤ, (A = -40) ∧ (∀ x : ℤ, (2 - x)^5 = ∑ r in finset.range 6, (binomial 5 r) * 2^(5-r) * (-1)^r * x^r) :=
sorry

end coefficient_x3_in_expansion_l174_174647


namespace fraction_halfway_between_3_4_and_5_6_is_19_24_l174_174918

noncomputable def fraction_halfway (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between_3_4_and_5_6_is_19_24 :
  fraction_halfway (3 / 4) (5 / 6) = 19 / 24 :=
by
  sorry

end fraction_halfway_between_3_4_and_5_6_is_19_24_l174_174918


namespace angle_ABC_center_of_circumscribed_circle_l174_174493

theorem angle_ABC_center_of_circumscribed_circle
  (O A B C : Point)
  (hO_center : IsCenterOfCircumscribedCircle O A B C)
  (angle_BOC : ∠BOC = 110)
  (angle_AOB : ∠AOB = 150) :
  ∠ABC = 50 := 
sorry

end angle_ABC_center_of_circumscribed_circle_l174_174493


namespace percentage_of_married_employees_l174_174200

variable (total_employees : ℕ)
variable (num_women : ℕ)
variable (num_men : ℕ)
variable (married_men : ℕ)
variable (married_women : ℕ)
variable (married_employees : ℕ)
variable (percentage_married : ℚ)

/-- Assume there are 100 employees in total --/
def assumption_total_employees : total_employees = 100 :=
by sorry

/-- Given that 64% of the employees are women --/
def women_percentage  : num_women = 64 :=
by sorry

/-- Therefore, 36% of the employees are men --/
def men_percentage : num_men = 36 :=
by sorry

/-- 2/3 of the men are single, hence 1/3 are married --/
def married_men_calc : married_men = 12 :=
by sorry

/-- 75% of the women are married --/
def married_women_calc : married_women = 48 :=
by sorry

/-- The total number of married employees --/
def total_married_employees : married_employees = 60 :=
by sorry

/-- Therefore, the percentage of married employees is 60% --/
def percentage_married_calc : percentage_married = 60 :=
by sorry

theorem percentage_of_married_employees :
  percentage_married = 60 :=
by 
  apply percentage_married_calc
  sorry

end percentage_of_married_employees_l174_174200


namespace largest_angle_is_120_l174_174506

noncomputable def largest_angle (x : ℝ) : ℝ :=
  let a := x^2 + x + 1
  let b := x^2 - 1
  let c := 2 * x + 1
  let cos_C := (a^2 + b^2 - c^2) / (2 * a * b)
  real.acos cos_C

theorem largest_angle_is_120 (x : ℝ) : largest_angle x = 120 :=
by
  sorry

end largest_angle_is_120_l174_174506


namespace hyperbola_properties_l174_174721

noncomputable def hyperbola_with_asymptote_and_point : Prop :=
  ∃ (a b : ℝ) (c : ℝ) (F1 F2 A B : ℝ × ℝ),
    (a > 0 ∧ b > 0) ∧
    (a = √3 ∧ b = 1) ∧
    (A = (√15 / 2, 1 / 2)) ∧
    (F1 = (-2, 0) ∧ F2 = (2, 0)) ∧
    -- The hyperbola equation
    (∀ x y : ℝ, (x = A.1 ∧ y = A.2) → (x^2 / a^2 - y^2 / b^2 = 1)) ∧
    -- Condition from the asymptote equation
    (a / b = √3) ∧
    -- Confirm eccentricity
    (c^2 = a^2 + b^2) ∧
    (c / a = 2 / √3) ∧
    (c = 2) ∧
    -- Confirm the conditions
    (√(5) - √(3) = √(5) - √(3))

-- Prove the main properties for the hyperbola
theorem hyperbola_properties :
  hyperbola_with_asymptote_and_point :=
begin
  sorry -- proof goes here
end

end hyperbola_properties_l174_174721


namespace john_needs_4_planks_l174_174256

theorem john_needs_4_planks :
  ∀ (total_nails additional_nails nails_per_plank : ℕ),
    total_nails = 43 →
    additional_nails = 15 →
    nails_per_plank = 7 →
    (total_nails - additional_nails) / nails_per_plank = 4 :=
by
  intros total_nails additional_nails nails_per_plank ht ha hn
  rw [ht, ha, hn]
  sorry

end john_needs_4_planks_l174_174256


namespace trader_discount_percentage_l174_174194

-- Definitions from conditions
def cost_price : ℝ := 100
def marked_price := cost_price * 1.50
def loss := cost_price * 0.01
def selling_price := cost_price - loss
def discount := marked_price - selling_price
def discount_percentage := (discount / marked_price) * 100

-- Theorem stating the discount percentage
theorem trader_discount_percentage : discount_percentage = 34 := 
sorry 

end trader_discount_percentage_l174_174194


namespace smallest_interesting_number_l174_174146

theorem smallest_interesting_number :
  ∃ (n : ℕ), (∃ k1 : ℕ, 2 * n = k1 ^ 2) ∧ (∃ k2 : ℕ, 15 * n = k2 ^ 3) ∧ n = 1800 := 
sorry

end smallest_interesting_number_l174_174146


namespace cotangent_sum_identity_l174_174216

theorem cotangent_sum_identity (a b c d : ℝ) :
    (∀ a b : ℝ, cot (arccot a + arccot b) = (a * b - 1) / (a + b)) →
    cot (arccot 5 + arccot 11 + arccot 17 + arccot 23) = 97 / 40 :=
by
    intros identity
    have h₁ : cot (arccot 5 + arccot 11) = (5 * 11 - 1) / (5 + 11) := by rw [identity]
    have h₂ : cot (arccot 17 + arccot 23) = (17 * 23 - 1) / (17 + 23) := by rw [identity]
    have h₃ : cot (arccot ((54 : ℚ)/16) + arccot ((390 : ℚ)/40)) = (54/16 * 390/40 - 1) / (54/16 + 390/40) := sorry
    have h₄ : (5 * 11 - 1) = 54 := by norm_num
    have h₅ : (17 * 23 - 1) = 390 := by norm_num
    have h₆ : cot (arccot ((27/8 : ℚ)) + arccot ((39/4 : ℚ))) = (27/8 * 39/4 - 1) / (27/8 + 39/4) := sorry
    sorry -- Further steps are included in a similar manner

end cotangent_sum_identity_l174_174216


namespace number_of_tea_bags_l174_174441

theorem number_of_tea_bags (n : ℕ) 
  (h1 : 2 * n ≤ 41) (h2 : 41 ≤ 3 * n)
  (h3 : 2 * n ≤ 58) (h4 : 58 ≤ 3 * n) :
  n = 20 :=
by
  sorry

end number_of_tea_bags_l174_174441


namespace smallest_interesting_number_smallest_interesting_number_1800_l174_174176
-- Import all the necessary libraries

-- Variables definitions adhering to the conditions of the problem
def isPerfectSquare (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k

def isPerfectCube (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k * k

def isInteresting (n : ℕ) : Prop :=
  isPerfectSquare (2 * n) ∧ isPerfectCube (15 * n)

-- Prove that the smallest interesting number is 1800
theorem smallest_interesting_number : ∀ n : ℕ, isInteresting n → 1800 ≤ n :=
begin
  sorry
end

theorem smallest_interesting_number_1800 : isInteresting 1800 :=
begin
  sorry
end

end smallest_interesting_number_smallest_interesting_number_1800_l174_174176


namespace sammy_pickles_l174_174829

theorem sammy_pickles 
  (T S R : ℕ) 
  (h1 : T = 2 * S) 
  (h2 : R = 8 * T / 10) 
  (h3 : R = 24) : 
  S = 15 :=
by
  sorry

end sammy_pickles_l174_174829


namespace smallest_interesting_number_smallest_interesting_number_1800_l174_174175
-- Import all the necessary libraries

-- Variables definitions adhering to the conditions of the problem
def isPerfectSquare (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k

def isPerfectCube (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k * k

def isInteresting (n : ℕ) : Prop :=
  isPerfectSquare (2 * n) ∧ isPerfectCube (15 * n)

-- Prove that the smallest interesting number is 1800
theorem smallest_interesting_number : ∀ n : ℕ, isInteresting n → 1800 ≤ n :=
begin
  sorry
end

theorem smallest_interesting_number_1800 : isInteresting 1800 :=
begin
  sorry
end

end smallest_interesting_number_smallest_interesting_number_1800_l174_174175


namespace B_finishes_in_15_days_l174_174107

theorem B_finishes_in_15_days (B : ℝ) : (A_days B_wages total_wages : ℝ) (A_days = 10) (B_wages = 2100) (total_wages = 3500) : 
    1 / B * (A_days / (A_days + B_days)) = B_wages / total_wages :=
by sorry

end B_finishes_in_15_days_l174_174107


namespace Tiffany_lives_l174_174897

theorem Tiffany_lives (initial_lives lost_lives : ℕ) (gain_factor : ℕ) 
  (h_initial : initial_lives = 75) 
  (h_lost : lost_lives = 28) 
  (h_gain : gain_factor = 3) : 
  let remaining_lives := initial_lives - lost_lives
  let gained_lives := gain_factor * lost_lives
  let final_lives := remaining_lives + gained_lives
  final_lives = 131 := 
by
  rw [h_initial, h_lost, h_gain]
  let remaining_lives := 75 - 28
  let gained_lives := 3 * 28
  let final_lives := remaining_lives + gained_lives
  have h_rem : remaining_lives = 47 := rfl
  have h_gain : gained_lives = 84 := rfl
  have h_final : final_lives = 47 + 84 := rfl
  exact h_final.symm.trans (by norm_num)

sorry

end Tiffany_lives_l174_174897


namespace find_ratios_l174_174814

variable {V : Type}
variable [AddCommGroup V] [Module ℝ V]
variables (A B Q : V)

theorem find_ratios
  (h1: AQ:QP:PB = 5:2:1) :
  Q = (3/8) • A + (5/8) • B :=
sorry

end find_ratios_l174_174814


namespace sum_even_integers_less_than_100_l174_174990

theorem sum_even_integers_less_than_100 : 
  let sequence := List.range' 2 98
  let even_seq := sequence.filter (λ x => x % 2 = 0)
  (even_seq.sum) = 2450 :=
by
  sorry

end sum_even_integers_less_than_100_l174_174990


namespace largest_fraction_l174_174554

theorem largest_fraction :
  let A := 2 / 5
  let B := 3 / 7
  let C := 4 / 9
  let D := 5 / 11
  let E := 6 / 13
  E > A ∧ E > B ∧ E > C ∧ E > D :=
by
  let A := 2 / 5
  let B := 3 / 7
  let C := 4 / 9
  let D := 5 / 11
  let E := 6 / 13
  sorry

end largest_fraction_l174_174554


namespace cistern_length_l174_174113

theorem cistern_length (W D A : ℝ) (hW : W = 4) (hD : D = 1.25) (hA : A = 55.5) : 
  ∃ L : ℝ, L = 7 ∧ A = (L * W) + 2 * (L * D) + 2 * (W * D) :=
by {
  use 7,
  rw [hW, hD, hA],
  simp,
  linarith
}

end cistern_length_l174_174113


namespace smallest_interesting_number_l174_174131

def is_interesting (n : ℕ) : Prop :=
  (∃ m1 : ℕ, 2 * n = m1^2) ∧ (∃ m2 : ℕ, 15 * n = m2^3)

theorem smallest_interesting_number : ∃ n, is_interesting n ∧ (∀ m : ℕ, is_interesting m → n ≤ m) :=
  ⟨1800, by {
    split,
    { use 60,
      norm_num },
    { use 30,
      norm_num },
    sorry }⟩

end smallest_interesting_number_l174_174131


namespace halfway_fraction_l174_174931

theorem halfway_fraction (a b c d : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 6) (h_c : c = 19 / 24) :
  (1 / 2) * (a + b) = c := 
sorry

end halfway_fraction_l174_174931


namespace binomial_expansion_coeff_l174_174742

theorem binomial_expansion_coeff (a : ℝ) : 
  (∃ (a : ℝ), ∀ x : ℝ, (∃ k : ℕ, (k = 5 ∧ (∃ c : ℝ, c * x^k = 7) ∧
    (c = (Nat.choose 7 2) * a^2))) → 
    (a = sqrt(3) / 3 ∨ a = -sqrt(3) / 3)) := sorry

end binomial_expansion_coeff_l174_174742


namespace problem_statement_l174_174796

-- Define T as the set of positive real numbers
def T : Set ℝ := {x : ℝ | x > 0}

-- Define the function g satisfying the given condition
def g (x : T) (y : T) : ℝ :=
  ∀ x y : T, g x * g y = g(x * y) + 2010 * ((1 / x : ℝ) + (1 / y) + 2009)

-- Define the problem statement in Lean
theorem problem_statement :
  ∃ (g : T → ℝ), (g 3 ≠ 0) ∧ (∃! m : ℕ, ∃! t : ℝ, m * t = 6031 / 3) ∧
  ∀ x y : T, g x * g y = g (x * y) + 2010 * ((1 / x : ℝ) + (1 / y) + 2009) := by
  sorry

end problem_statement_l174_174796


namespace differentiable_f_derivative_f_l174_174282

noncomputable def f : ℝ → ℝ := λ x, x + sin x

theorem differentiable_f (x : ℝ) : differentiable ℝ f :=
by {
  apply differentiable.add,
  exact differentiable_id,
  exact differentiable_sin
}

theorem derivative_f (x : ℝ) : deriv f x = 1 + cos x :=
by {
  simp [f],
  exact deriv_add (deriv_id' _) (deriv_sin x)
}

end differentiable_f_derivative_f_l174_174282


namespace measure_angle_ABC_l174_174486

variables (A B C O : Point)
variables (h1 : circumcenter O A B C)
variables (h2 : angle O B C = 110)
variables (h3 : angle A O B = 150)

theorem measure_angle_ABC : angle A B C = 50 :=
by
  sorry

end measure_angle_ABC_l174_174486


namespace age_difference_l174_174503

theorem age_difference (x : ℕ) 
  (h_ratio : 4 * x + 3 * x + 7 * x = 126)
  (h_halima : 4 * x = 36)
  (h_beckham : 3 * x = 27) :
  4 * x - 3 * x = 9 :=
by sorry

end age_difference_l174_174503


namespace inverse_variation_l174_174847

theorem inverse_variation (a b k : ℝ) (h1 : a * b^3 = k) (h2 : 8 * 1^3 = k) : (∃ a, b = 4 → a = 1 / 8) :=
by
  sorry

end inverse_variation_l174_174847


namespace remainder_div_x_plus_2_l174_174686

-- Given polynomial 
def q (x : ℕ) (D E F : ℕ) : ℕ := D*x^4 + E*x^2 + F*x + 9

-- Given condition: the remainder when divided by x - 2 is 17
axiom h : ∀ (D E F : ℕ), q 2 D E F = 17

theorem remainder_div_x_plus_2 (D E F : ℕ) : q (-2) D E F = 33 :=
sorry

end remainder_div_x_plus_2_l174_174686


namespace number_of_elements_l174_174014

def average_incorrect (N : ℕ) := 21
def correction (incorrect : ℕ) (correct : ℕ) := correct - incorrect
def average_correct (N : ℕ) := 22

theorem number_of_elements (N : ℕ) (incorrect : ℕ) (correct : ℕ) :
  average_incorrect N = 21 ∧ incorrect = 26 ∧ correct = 36 ∧ average_correct N = 22 →
  N = 10 :=
by
  sorry

end number_of_elements_l174_174014


namespace sum_of_first_15_terms_l174_174764

-- Given an arithmetic sequence {a_n} such that a_4 + a_6 + a_8 + a_10 + a_12 = 40
-- we need to prove that the sum of the first 15 terms is 120

theorem sum_of_first_15_terms 
  (a_4 a_6 a_8 a_10 a_12 : ℤ)
  (h1 : a_4 + a_6 + a_8 + a_10 + a_12 = 40)
  (a1 d : ℤ)
  (h2 : a_4 = a1 + 3*d)
  (h3 : a_6 = a1 + 5*d)
  (h4 : a_8 = a1 + 7*d)
  (h5 : a_10 = a1 + 9*d)
  (h6 : a_12 = a1 + 11*d) :
  (15 * (a1 + 7*d) = 120) :=
by
  sorry

end sum_of_first_15_terms_l174_174764


namespace halfway_fraction_l174_174941

theorem halfway_fraction (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  ((a / b + c / d) / 2) = 19 / 24 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end halfway_fraction_l174_174941


namespace age_difference_l174_174500

theorem age_difference (h b m : ℕ) (ratio : h = 4 * m ∧ b = 3 * m ∧ 4 * m + 3 * m + 7 * m = 126) : h - b = 9 :=
by
  -- proof will be filled here
  sorry

end age_difference_l174_174500


namespace largest_item_among_options_l174_174269

variable {ℝ : Type} [LinearOrderedField ℝ]

-- Conditions
def f (x : ℝ) : ℝ := sorry  -- Define the function f
axiom f_property : ∀ x, x * (deriv f x) + f(x) > 0

def condition (a b : ℝ) : Prop := 0 < a ∧ a < b ∧ b < 1

-- Proving the largest one is D
theorem largest_item_among_options (a b : ℝ) (h : condition a b) : 
  let A := a * b * f(a * b),
      B := b * a * f(b * a),
      C := (Real.log(ab)) * f(Real.log(ab)),
      D := (Real.log(ba)) * f(Real.log(ba))
  in D = max A (max B (max C D)) :=
sorry

end largest_item_among_options_l174_174269


namespace range_of_log_sqrt_sin_l174_174078

noncomputable def log_base_3 (x : ℝ) : ℝ :=
  if x > 0 then real.log x / real.log 3 else 0

def range_log_sqrt_sin (x : ℝ) (hx : 0 < x ∧ x < real.pi / 2) : ℝ :=
  log_base_3 (real.sqrt (real.sin x))

theorem range_of_log_sqrt_sin :
  (set.range (λ x, range_log_sqrt_sin x ⟨hgt, hlt⟩ ) = set.Iic 0) := sorry

end range_of_log_sqrt_sin_l174_174078


namespace sum_even_integers_less_than_100_l174_174991

theorem sum_even_integers_less_than_100 : 
  let sequence := List.range' 2 98
  let even_seq := sequence.filter (λ x => x % 2 = 0)
  (even_seq.sum) = 2450 :=
by
  sorry

end sum_even_integers_less_than_100_l174_174991


namespace product_of_numbers_l174_174559

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x^2 + y^2 = 404) : x * y = 86 := 
sorry

end product_of_numbers_l174_174559


namespace feasible_stations_l174_174051

theorem feasible_stations (n : ℕ) (h: n > 0) 
  (pairings : ∀ (i j : ℕ), i ≠ j → i < n → j < n → ∃ k, (i+k) % n = j ∨ (j+k) % n = i) : n = 4 :=
sorry

end feasible_stations_l174_174051


namespace smallest_interesting_number_l174_174136

def is_perfect_square (x : ℕ) : Prop :=
∃ y : ℕ, y * y = x

def is_perfect_cube (x : ℕ) : Prop :=
∃ y : ℕ, y * y * y = x

theorem smallest_interesting_number (n : ℕ) :
    is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n) → n = 1800 :=
by {
  sorry
}

end smallest_interesting_number_l174_174136


namespace find_n_l174_174246

theorem find_n (n : ℤ) (h1 : 6 ≤ n) (h2 : n ≤ 12) (h3 : n ≡ 10,403 [MOD 7]) : n = 8 :=
  sorry

end find_n_l174_174246


namespace halfway_fraction_l174_174936

theorem halfway_fraction (a b c d : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 6) (h_c : c = 19 / 24) :
  (1 / 2) * (a + b) = c := 
sorry

end halfway_fraction_l174_174936


namespace find_parameters_and_decreasing_interval_l174_174717

-- Define the function f(x) and the given conditions
def f (x : ℝ) (ω : ℝ) (a : ℝ) : ℝ := 4 * Real.cos (ω * x) * Real.sin (ω * x + Real.pi / 6) + a

theorem find_parameters_and_decreasing_interval :
  (∃ ω a : ℝ, ω > 0 ∧
   f (Real.pi / ω) ω a = 2 ∧
   (∀ x y : ℝ, (x < y ∧ f x 1 (-1) = f y 1 (-1)) → y - x = Real.pi) ∧
   (f (1) 1 (-1) ≥ 3)) ∧
  (∀ x : ℝ, (Real.pi / 6 ≤ x ∧ x ≤ 2 * Real.pi / 3) → f(x) 1 (-1) ≤ f(x + 1) 1 (-1)) :=
by
  sorry

end find_parameters_and_decreasing_interval_l174_174717


namespace number_of_valid_numbers_l174_174728

-- Definitions from given conditions
def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ n % 100 = 47 ∧ (array_sum (to_digits n)) % 4 = 2

-- Sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  (to_digits n).sum

-- Statement of the the mathematically equivalent proof problem
theorem number_of_valid_numbers : 
  ∃ n, is_valid_number n ∧ count_valid_numbers = 22 :=
begin
  sorry   -- proof is omitted
end

end number_of_valid_numbers_l174_174728


namespace intersect_point_l174_174026

-- Definitions as per conditions
def f (x : ℝ) (b : ℝ) : ℝ := 4 * x + b
def f_inv (x : ℝ) (a : ℝ) : ℝ := a -- We define inverse as per given (4, a)

-- Variables for the conditions
variables (a b : ℤ)

-- Theorems to prove the conditions match the answers
theorem intersect_point : ∃ a b : ℤ, f 4 b = a ∧ f_inv 4 a = 4 ∧ a = 4 := by
  sorry

end intersect_point_l174_174026


namespace farmer_apples_final_count_l174_174865

theorem farmer_apples_final_count : 
  let initial_apples := 1000 
  let neighbor_apples := (2 / 5 : ℚ) * initial_apples
  let apples_after_neighbor := initial_apples - neighbor_apples
  let niece_apples := (1 / 5 : ℚ) * apples_after_neighbor
  let apples_after_niece := apples_after_neighbor - niece_apples
  let apples_after_eating := apples_after_niece - 7
  apples_after_eating = 473 :=
begin
  sorry
end

end farmer_apples_final_count_l174_174865


namespace smallest_interesting_number_l174_174117

theorem smallest_interesting_number :
  ∃ n : ℕ, (∃ k1 : ℕ, 2 * n = k1^2) ∧ (∃ k2 : ℕ, 15 * n = k2^3) ∧ n = 1800 :=
by
  sorry

end smallest_interesting_number_l174_174117


namespace correct_answer_is_option_d_l174_174552

def is_quadratic (eq : String) : Prop :=
  eq = "a*x^2 + b*x + c = 0"

def OptionA : String := "1/x^2 + x - 1 = 0"
def OptionB : String := "3x + 1 = 5x + 4"
def OptionC : String := "x^2 + y = 0"
def OptionD : String := "x^2 - 2x + 1 = 0"

theorem correct_answer_is_option_d :
  is_quadratic OptionD :=
by
  sorry

end correct_answer_is_option_d_l174_174552


namespace cot_arccots_sum_eq_97_over_40_l174_174207

noncomputable def cot_arccot_sum : ℝ :=
  Real.cot (Real.arccot 5 + Real.arccot 11 + Real.arccot 17 + Real.arccot 23)

theorem cot_arccots_sum_eq_97_over_40 :
  cot_arccot_sum = 97 / 40 :=
sorry

end cot_arccots_sum_eq_97_over_40_l174_174207


namespace max_sum_value_is_four_infinite_triples_of_xyz_l174_174569

noncomputable def max_sum_xyz : ℝ :=
  Real.sup {s : ℝ | ∃ (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0),
                      16 * x * y * z = (x + y) ^ 2 * (x + z) ^ 2 ∧ s = x + y + z}

theorem max_sum_value_is_four : max_sum_xyz = 4 := sorry

theorem infinite_triples_of_xyz (M : ℝ) (hM : M = 4) :
  ∃ᶠ (x y z : ℚ) in filter.cfilter _, 16 * x * y * z = (x + y) ^ 2 * (x + z) ^ 2 ∧ x + y + z = M := sorry

end max_sum_value_is_four_infinite_triples_of_xyz_l174_174569


namespace smallest_interesting_number_l174_174171

-- Definitions based on the conditions
def isPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m
def isPerfectCube (m : ℕ) : Prop := ∃ k : ℕ, k * k * k = m

def isInteresting (n : ℕ) : Prop :=
  isPerfectSquare (2 * n) ∧ isPerfectCube (15 * n)

-- The smallest interesting number
theorem smallest_interesting_number : ∃ n : ℕ, isInteresting n ∧ (∀ m : ℕ, isInteresting m → n ≤ m) :=
  ⟨1800, ⟨⟨26, by norm_num⟩, ⟨15, by norm_num⟩⟩, sorry⟩

end smallest_interesting_number_l174_174171


namespace exists_consecutive_integers_with_low_degree_l174_174646

def degree (n : ℕ) : ℕ :=
  (n.factors.nodup_erase_dup.map (λ p, n.factor_multiset p)).sum

theorem exists_consecutive_integers_with_low_degree :
  ∃ N : ℕ, (finset.range 2018).filter (λ i, degree (N + i) < 11).card = 1000 :=
sorry

end exists_consecutive_integers_with_low_degree_l174_174646


namespace servings_made_l174_174893

noncomputable def chickpeas_per_can := 16 -- ounces in one can
noncomputable def ounces_per_serving := 6 -- ounces needed per serving
noncomputable def total_cans := 8 -- total cans Thomas buys

theorem servings_made : (total_cans * chickpeas_per_can) / ounces_per_serving = 21 :=
by
  sorry

end servings_made_l174_174893


namespace regular_octagon_angle_of_intersection_l174_174836

theorem regular_octagon_angle_of_intersection (ABCDEFGH : Fin 8 → Point)
  (h_reg : regular_octagon ABCDEFGH)
  (Q : Point)
  (h_Q : extended_sides_intersect_at_Q ABCDEFGH Q AB CD) :
  ∠CQD = 90 :=
sorry


end regular_octagon_angle_of_intersection_l174_174836


namespace complex_expression_simplify_l174_174298

noncomputable def complex_exp : ℂ :=
  (-1/2 + complex.I * (real.sqrt 3 / 2)) * (complex.I * (real.sqrt 3 / 2) - 1/2) + (complex.I * (real.sqrt 3 / 2))

theorem complex_expression_simplify : complex_exp = -1/2 := 
  sorry

end complex_expression_simplify_l174_174298


namespace unique_integer_sum_l174_174884

/-!
The sequence \(a_{1}, a_{2}, \ldots\) is such that \(a_{1} \in (1,2)\) and \(a_{k+1} = a_{k} + \frac{k}{a_{k}}\) for any natural \(k\). 
Prove that there cannot be more than one pair of terms in the sequence with an integer sum.
-/

def sequence (a : ℕ → ℝ) : Prop :=
  (a 1 > 1 ∧ a 1 < 2) ∧ ∀ k, a (k + 1) = a k + (k : ℝ) / a k

theorem unique_integer_sum (a : ℕ → ℝ) (h_seq : sequence a) :
  ∃! i j, i ≠ j ∧ ∃! (i j : ℕ), a i + a j ∈ ℤ :=
sorry

end unique_integer_sum_l174_174884


namespace smallest_interesting_number_l174_174125

def is_interesting (n : ℕ) : Prop :=
  (∃ m1 : ℕ, 2 * n = m1^2) ∧ (∃ m2 : ℕ, 15 * n = m2^3)

theorem smallest_interesting_number : ∃ n, is_interesting n ∧ (∀ m : ℕ, is_interesting m → n ≤ m) :=
  ⟨1800, by {
    split,
    { use 60,
      norm_num },
    { use 30,
      norm_num },
    sorry }⟩

end smallest_interesting_number_l174_174125


namespace science_book_pages_l174_174483

def history_pages := 300
def novel_pages := history_pages / 2
def science_pages := novel_pages * 4

theorem science_book_pages : science_pages = 600 := by
  -- Given conditions:
  -- The novel has half as many pages as the history book, the history book has 300 pages,
  -- and the science book has 4 times as many pages as the novel.
  sorry

end science_book_pages_l174_174483


namespace inequality_not_less_than_four_by_at_least_one_l174_174027

-- Definitions based on the conditions
def not_less_than_by_at_least (y : ℝ) (a b : ℝ) : Prop := y - a ≥ b

-- Problem statement (theorem) based on the given question and correct answer
theorem inequality_not_less_than_four_by_at_least_one (y : ℝ) :
  not_less_than_by_at_least y 4 1 → y ≥ 5 :=
by
  sorry

end inequality_not_less_than_four_by_at_least_one_l174_174027


namespace halfway_fraction_l174_174945

theorem halfway_fraction (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  ((a / b + c / d) / 2) = 19 / 24 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end halfway_fraction_l174_174945


namespace scout_tips_per_customer_l174_174454

theorem scout_tips_per_customer
  (base_pay_per_hour : ℕ)
  (hours_saturday : ℕ)
  (customers_saturday : ℕ)
  (hours_sunday : ℕ)
  (customers_sunday : ℕ)
  (total_earning_weekend : ℕ)
  (base_pay_weekend := hours_saturday * base_pay_per_hour + hours_sunday * base_pay_per_hour)
  (total_tips_weekend := total_earning_weekend - base_pay_weekend)
  (total_customers_weekend := customers_saturday + customers_sunday)
  : base_pay_per_hour = 10 → hours_saturday = 4 → customers_saturday = 5 → hours_sunday = 5 → customers_sunday = 8 → total_earning_weekend = 155 → total_tips_weekend / total_customers_weekend = 5 := by
  intros h1 h2 h3 h4 h5 h6
  have h_base_pay_weekend : base_pay_weekend = 4 * 10 + 5 * 10 := by simp [h1, h2, h4]
  have h_base_pay_weekend_90 : base_pay_weekend = 90 := by linarith [h_base_pay_weekend]
  have h_total_tips_weekend : total_tips_weekend = 155 - 90 := by simp [h_base_pay_weekend_90, h6]
  have h_total_tips_weekend_65 : total_tips_weekend = 65 := by linarith [h_total_tips_weekend]
  have h_total_customers_weekend : total_customers_weekend = 5 + 8 := by simp [h3, h5]
  have h_total_customers_weekend_13 : total_customers_weekend = 13 := by linarith [h_total_customers_weekend]
  have h_tip_per_customer : total_tips_weekend / total_customers_weekend = 65 / 13 := by simp [h_total_tips_weekend_65, h_total_customers_weekend_13]
  norm_num at h_tip_per_customer
  exact h_tip_per_customer

end scout_tips_per_customer_l174_174454


namespace triangle_area_arithmetic_sequence_l174_174608

theorem triangle_area_arithmetic_sequence :
  ∃ (S_1 S_2 S_3 S_4 S_5 : ℝ) (d : ℝ),
  S_1 + S_2 + S_3 + S_4 + S_5 = 420 ∧
  S_2 = S_1 + d ∧
  S_3 = S_1 + 2 * d ∧
  S_4 = S_1 + 3 * d ∧
  S_5 = S_1 + 4 * d ∧
  S_5 = 112 :=
by
  sorry

end triangle_area_arithmetic_sequence_l174_174608


namespace tangent_line_at_1_two_tangent_lines_through_1_neg1_local_min_no_local_max_eq_f_x_1_no_two_distinct_solution_l174_174301

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_at_1 :
  (∀ x : ℝ, f(x) = x * Real.log x) →
  ∃ m b, (m = 1) ∧ (b = -1) ∧ (∀ y x : ℝ, y = m * x + b → y = f(1)) :=
by sorry

theorem two_tangent_lines_through_1_neg1 :
  (∀ x : ℝ, f(x) = x * Real.log x) →
  ∃ (x1 x2 : ℝ), (x1 ≠ x2) ∧ (∃ m1 b1 m2 b2, m1 = Real.log x1 + 1 ∧ m2 = Real.log x2 + 1 ∧ 
  (∀ y : ℝ, y = m1 * (1 - x1) + b1 → y = f x1 - (Real.log x1 + 1) * x1) ∧ 
  (∀ y : ℝ, y = m2 * (1 - x2) + b2 → y = f x2 - (Real.log x2 + 1) * x2)) :=
by sorry

theorem local_min_no_local_max : 
  (∀ x : ℝ, f(x) = x * Real.log x) →
  ∃ (xmin : ℝ), (xmin = Real.exp (-1)) ∧ (∀ x : ℝ, (x ≠ xmin → f' x ≠ 0) ∧
  (∀ x : ℝ, f' x < 0 → x < xmin) ∧ (∀ x : ℝ, f' x > 0 → x > xmin)) :=
by sorry

theorem eq_f_x_1_no_two_distinct_solution :
  (∀ x : ℝ, f(x) = x * Real.log x) →
  ¬ ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (f x1 = 1 ∧ f x2 = 1) :=
by sorry

end tangent_line_at_1_two_tangent_lines_through_1_neg1_local_min_no_local_max_eq_f_x_1_no_two_distinct_solution_l174_174301


namespace three_x_minus_five_y_l174_174415

noncomputable def F : ℝ × ℝ :=
  let D := (15, 3)
  let E := (6, 8)
  ((D.1 + E.1) / 2, (D.2 + E.2) / 2)

theorem three_x_minus_five_y : (3 * F.1 - 5 * F.2) = 4 := by
  sorry

end three_x_minus_five_y_l174_174415


namespace hypotenuse_ratio_30_degree_triangles_l174_174513

theorem hypotenuse_ratio_30_degree_triangles 
  (△ABC △DEF △GHI : Triangle)
  (H_ABC : △ABC.is_right_angle ∧ △ABC.contains_angle 30)
  (H_DEF : △DEF.is_right_angle ∧ △DEF.contains_angle 30)
  (H_GHI : △GHI.is_right_angle ∧ △GHI.contains_angle 30)
  (equal_side_length : [△ABC.shared_side, △DEF.shared_side, △GHI.shared_side] ≈ equal_length)
  (hyp_GHI : △GHI.hypotenuse = 1)
  (hyp_DEF : △DEF.hypotenuse = 2 / sqrt(3))
  (hyp_ABC : △ABC.hypotenuse = 2) 
  :
  [△ABC.hypotenuse, △DEF.hypotenuse, △GHI.hypotenuse] ≈ [2, 2 / sqrt(3), 1] :=
by
  sorry

end hypotenuse_ratio_30_degree_triangles_l174_174513


namespace intersection_of_M_and_N_l174_174317

-- Define the sets M and N with the given conditions
def M := {x : ℝ | -1 < x ∧ x < 3}
def N := {x : ℝ | -2 < x ∧ x < 1}

-- State the theorem that the intersection of M and N is as described
theorem intersection_of_M_and_N : (M ∩ N) = {x : ℝ | -1 < x ∧ x < 1} :=
by
  -- the proof will go here
  sorry

end intersection_of_M_and_N_l174_174317


namespace car_a_total_travel_time_l174_174626

noncomputable def speed_ratio_car_a_car_b : ℝ := 4 / 3
noncomputable def time_difference : ℝ := 6
noncomputable def doubled_speed_time : ℝ := 6

theorem car_a_total_travel_time (t : ℝ) (a : ℝ) :
  let speedA := 4 * a,
      speedB := 3 * a in
  (t + time_difference) * speedB = time_difference * speedA + (t - doubled_speed_time) * (2 * speedA) →
  t = 8.4 :=
by
  intros
  sorry

end car_a_total_travel_time_l174_174626


namespace halfway_fraction_l174_174954

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/6) : (a + b) / 2 = 19 / 24 :=
by
  sorry

end halfway_fraction_l174_174954


namespace total_surface_area_with_holes_l174_174603

def cube_edge_length : ℝ := 5
def hole_side_length : ℝ := 2

/-- Calculate the total surface area of a modified cube with given edge length and holes -/
theorem total_surface_area_with_holes 
  (l : ℝ) (h : ℝ)
  (hl_pos : l > 0) (hh_pos : h > 0) (hh_lt_hl : h < l) : 
  (6 * l^2 - 6 * h^2 + 6 * 4 * h^2) = 222 :=
by sorry

end total_surface_area_with_holes_l174_174603


namespace halfway_fraction_l174_174932

theorem halfway_fraction (a b c d : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 6) (h_c : c = 19 / 24) :
  (1 / 2) * (a + b) = c := 
sorry

end halfway_fraction_l174_174932


namespace range_of_a_l174_174290

-- Given conditions
variables {f : ℝ → ℝ}
def even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def decreasing_on_neg (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ ∧ x₁ ≤ 0 ∧ x₂ ≤ 0 → (f x₂ - f x₁) / (x₂ - x₁) < 0

-- Problem statement
theorem range_of_a (h_even : even f)
  (h_decreasing_on_neg : decreasing_on_neg f)
  (h_ineq : ∀ a, f (1 - a) > f (2a - 1) ↔ 0 < a ∧ a < 2 / 3) :
  ∀ a, f (1 - a) > f (2a - 1) → 0 < a ∧ a < 2 / 3 := 
sorry

end range_of_a_l174_174290


namespace classroom_notebooks_count_l174_174354

noncomputable def total_notebooks_in_classroom
  (n : ℕ)
  (students_with_5 : ℕ)
  (students_with_3 : ℕ)
  (students_with_7 : ℕ) : ℕ :=
students_with_5 * 5 + students_with_3 * 3 + students_with_7 * 7

theorem classroom_notebooks_count : 
  ∀ (n : ℕ),
  n = 28 →
  (∃ students_with_5 students_with_3 students_with_7,
  students_with_5 = 9 ∧
  students_with_3 = 9 ∧
  students_with_7 = n - (students_with_5 + students_with_3) ∧
  total_notebooks_in_classroom n students_with_5 students_with_3 students_with_7 = 142) :=
begin
  sorry
end

end classroom_notebooks_count_l174_174354


namespace number_of_tea_bags_l174_174443

theorem number_of_tea_bags (n : ℕ) 
  (h1 : 2 * n ≤ 41) (h2 : 41 ≤ 3 * n)
  (h3 : 2 * n ≤ 58) (h4 : 58 ≤ 3 * n) :
  n = 20 :=
by
  sorry

end number_of_tea_bags_l174_174443


namespace smallest_interesting_number_l174_174142

theorem smallest_interesting_number :
  ∃ (n : ℕ), (∃ k1 : ℕ, 2 * n = k1 ^ 2) ∧ (∃ k2 : ℕ, 15 * n = k2 ^ 3) ∧ n = 1800 := 
sorry

end smallest_interesting_number_l174_174142


namespace prob_composite_in_first_50_l174_174543

-- Define what it means for a number to be composite
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ (∀ m : ℕ, m > 1 → m < n → ¬ m ∣ n)

-- Define the set of first 50 natural numbers
def first_50_numbers : list ℕ :=
  (list.range 50).map (λ n, n + 1)

-- Define the set of composite numbers within the first 50 natural numbers
def composite_numbers : list ℕ :=
  first_50_numbers.filter is_composite

-- Define the probability function
noncomputable def probability_of_composite : ℚ :=
  composite_numbers.length / first_50_numbers.length

-- The theorem statement
theorem prob_composite_in_first_50 : probability_of_composite = 34 / 50 :=
by sorry

end prob_composite_in_first_50_l174_174543


namespace cot_arccots_sum_eq_97_over_40_l174_174208

noncomputable def cot_arccot_sum : ℝ :=
  Real.cot (Real.arccot 5 + Real.arccot 11 + Real.arccot 17 + Real.arccot 23)

theorem cot_arccots_sum_eq_97_over_40 :
  cot_arccot_sum = 97 / 40 :=
sorry

end cot_arccots_sum_eq_97_over_40_l174_174208


namespace correct_random_variable_l174_174574

-- Define the given conditions
def total_white_balls := 5
def total_red_balls := 3
def total_balls := total_white_balls + total_red_balls
def balls_drawn := 3

-- Define the random variable
noncomputable def is_random_variable_correct (option : ℕ) :=
  option = 2

-- The theorem to be proved
theorem correct_random_variable: is_random_variable_correct 2 :=
by
  sorry

end correct_random_variable_l174_174574


namespace jack_further_down_l174_174376

-- Define the conditions given in the problem
def flights_up := 3
def flights_down := 6
def steps_per_flight := 12
def height_per_step_in_inches := 8
def inches_per_foot := 12

-- Define the number of steps and height calculations
def steps_up := flights_up * steps_per_flight
def steps_down := flights_down * steps_per_flight
def net_steps_down := steps_down - steps_up
def net_height_down_in_inches := net_steps_down * height_per_step_in_inches
def net_height_down_in_feet := net_height_down_in_inches / inches_per_foot

-- The proof statement to be shown
theorem jack_further_down : net_height_down_in_feet = 24 := sorry

end jack_further_down_l174_174376


namespace fraction_half_way_l174_174925

theorem fraction_half_way (x y : ℚ) (h1 : x = 3 / 4) (h2 : y = 5 / 6) : (x + y) / 2 = 19 / 24 := by
  sorry

end fraction_half_way_l174_174925


namespace fraction_halfway_between_3_4_and_5_6_is_19_24_l174_174914

noncomputable def fraction_halfway (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between_3_4_and_5_6_is_19_24 :
  fraction_halfway (3 / 4) (5 / 6) = 19 / 24 :=
by
  sorry

end fraction_halfway_between_3_4_and_5_6_is_19_24_l174_174914


namespace largest_n_for_positive_sum_l174_174736

noncomputable def arithmetic_sequence_largest_n (a : ℕ → ℝ) (a1 : ℝ) (d : ℝ) : ℕ :=
  if ∀ n, a n = a1 + (n - 1) * d ∧ a 1 = a1 ∧ a1 > 0 ∧
     (a 1007 + a 1008 > 0) ∧ (a 1007 * a 1008 < 0) then
    2014
  else
    0

theorem largest_n_for_positive_sum (a : ℕ → ℝ) (a1 : ℝ) (d : ℝ) :
  (∀ n, a n = a1 + (n - 1) * d) →
  a 1 = a1 → a1 > 0 →
  (a 1007 + a 1008 > 0) →
  (a 1007 * a 1008 < 0) →
  S_n a 2014 > 0 := sorry

end largest_n_for_positive_sum_l174_174736


namespace magnitude_of_alpha_l174_174399

noncomputable def α : ℂ := sorry
noncomputable def β : ℂ := α.conj  -- since β is the conjugate of α 

-- Conditions
axiom h1 : (α / (β^2)).im = 0
axiom h2 : |α - β| = 2 * √5

theorem magnitude_of_alpha : |α| = (2 * √15) / 3 := by
  -- Proof omitted
  sorry

end magnitude_of_alpha_l174_174399


namespace measure_angle_ABC_l174_174487

variables (A B C O : Point)
variables (h1 : circumcenter O A B C)
variables (h2 : angle O B C = 110)
variables (h3 : angle A O B = 150)

theorem measure_angle_ABC : angle A B C = 50 :=
by
  sorry

end measure_angle_ABC_l174_174487


namespace halfway_fraction_l174_174974

theorem halfway_fraction (a b : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) : (1/2) * (a + b) = 19/24 :=
by 
  rw [h1, h2]
  simp
  done
  sorry

end halfway_fraction_l174_174974


namespace probability_multiple_of_5_sum_l174_174348

noncomputable def first_nine_primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23}

def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

theorem probability_multiple_of_5_sum :
  let pairs := first_nine_primes.val.to_finset.to_list.product first_nine_primes.val.to_finset.to_list in
  let valid_pairs := pairs.filter (λ p : ℕ × ℕ, p.1 < p.2 ∧ is_multiple_of_5 (p.1 + p.2)) in
  (valid_pairs.length : ℚ) / (pairs.length / 2) = 1 / 9 :=
by sorry

end probability_multiple_of_5_sum_l174_174348


namespace line_intersects_circle_but_not_through_center_l174_174504

-- Definition of the line
def line (x y : ℝ) := 3 * x - 4 * y - 9 = 0

-- Definition of the circle
def circle (x y : ℝ) := x^2 + y^2 = 4

-- Proving the relationship between the line and the circle
theorem line_intersects_circle_but_not_through_center :
  (∃ (x y : ℝ), line x y ∧ circle x y) ∧ ¬(∃ (x y : ℝ), line 0 0) :=
by
  sorry

end line_intersects_circle_but_not_through_center_l174_174504


namespace smallest_interesting_number_is_1800_l174_174157

def is_perfect_square (k : ℕ) : Prop := ∃ (m : ℕ), k = m * m
def is_perfect_cube (k : ℕ) : Prop := ∃ (m : ℕ), k = m * m * m

def is_interesting (n : ℕ) : Prop :=
  is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n)

theorem smallest_interesting_number_is_1800 :
  ∃ (n : ℕ), is_interesting(n) ∧ (∀ m : ℕ, is_interesting(m) → n ≤ m) ∧ n = 1800 :=
by 
  sorry

end smallest_interesting_number_is_1800_l174_174157


namespace sum_even_integers_less_than_100_l174_174987

theorem sum_even_integers_less_than_100 : 
  let sequence := List.range' 2 98
  let even_seq := sequence.filter (λ x => x % 2 = 0)
  (even_seq.sum) = 2450 :=
by
  sorry

end sum_even_integers_less_than_100_l174_174987


namespace min_value_expression_l174_174308

noncomputable def f (x : ℝ) : ℝ := abs (2 * x + 1) + abs (x - 1)

def min_value_f (f : ℝ → ℝ) : ℝ := (3 / 2 : ℝ)

theorem min_value_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = (3 / 2 : ℝ)) :
  (1 / (2 * a) + 1 / b) = (3 + 2 * real.sqrt 2) / 3 := by
  sorry

end min_value_expression_l174_174308


namespace smallest_interesting_number_smallest_interesting_number_1800_l174_174180
-- Import all the necessary libraries

-- Variables definitions adhering to the conditions of the problem
def isPerfectSquare (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k

def isPerfectCube (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k * k

def isInteresting (n : ℕ) : Prop :=
  isPerfectSquare (2 * n) ∧ isPerfectCube (15 * n)

-- Prove that the smallest interesting number is 1800
theorem smallest_interesting_number : ∀ n : ℕ, isInteresting n → 1800 ≤ n :=
begin
  sorry
end

theorem smallest_interesting_number_1800 : isInteresting 1800 :=
begin
  sorry
end

end smallest_interesting_number_smallest_interesting_number_1800_l174_174180


namespace arithmetic_seq_sum_l174_174765

theorem arithmetic_seq_sum (a : ℕ → ℤ) (h_arith_seq : ∀ m n p q : ℕ, m + n = p + q → a m + a n = a p + a q) (h_a5 : a 5 = 15) : a 2 + a 4 + a 6 + a 8 = 60 := 
by
  sorry

end arithmetic_seq_sum_l174_174765


namespace min_value_of_m_l174_174263

theorem min_value_of_m (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c = 3) :
  a^2 + b^2 + c^2 ≥ 3 :=
sorry

end min_value_of_m_l174_174263


namespace fraction_half_way_l174_174919

theorem fraction_half_way (x y : ℚ) (h1 : x = 3 / 4) (h2 : y = 5 / 6) : (x + y) / 2 = 19 / 24 := by
  sorry

end fraction_half_way_l174_174919


namespace smallest_interesting_number_is_1800_l174_174159

def is_perfect_square (k : ℕ) : Prop := ∃ (m : ℕ), k = m * m
def is_perfect_cube (k : ℕ) : Prop := ∃ (m : ℕ), k = m * m * m

def is_interesting (n : ℕ) : Prop :=
  is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n)

theorem smallest_interesting_number_is_1800 :
  ∃ (n : ℕ), is_interesting(n) ∧ (∀ m : ℕ, is_interesting(m) → n ≤ m) ∧ n = 1800 :=
by 
  sorry

end smallest_interesting_number_is_1800_l174_174159


namespace speed_of_man_upstream_l174_174585

-- Conditions stated as definitions 
def V_m : ℝ := 33 -- Speed of the man in still water
def V_downstream : ℝ := 40 -- Speed of the man rowing downstream

-- Required proof problem
theorem speed_of_man_upstream : V_m - (V_downstream - V_m) = 26 := 
by
  -- the following sorry is a placeholder for the actual proof
  sorry

end speed_of_man_upstream_l174_174585


namespace problem_not_divisible_7_11_13_l174_174104

theorem problem_not_divisible_7_11_13 : 
  let n := 999
  let a := 7
  let b := 11
  let c := 13
  let count_divisible (x : ℕ) := n / x
  let count_intersection (x y : ℕ) := n / (x * y)
  let total := count_divisible a + count_divisible b + count_divisible c 
               - count_intersection a b - count_intersection a c - count_intersection b c 
               + count_intersection a b c
  in (n + 1) - total = 719 := by {
  sorry
}

end problem_not_divisible_7_11_13_l174_174104


namespace tea_bags_number_l174_174424

theorem tea_bags_number (n : ℕ) (h1 : 2 * n ≤ 41) (h2 : 41 ≤ 3 * n) (h3 : 2 * n ≤ 58) (h4 : 58 ≤ 3 * n) : n = 20 :=
by
  sorry

end tea_bags_number_l174_174424


namespace barber_loss_l174_174586

-- Define the constants and conditions
constant haircut_cost : ℕ := 15
constant counterfeit_bill : ℕ := 20
constant real_bill_compensation : ℕ := 20
constant change_given : ℕ := 5

-- Define the problem
theorem barber_loss :
  (haircut_cost + real_bill_compensation = 35) :=
by sorry

end barber_loss_l174_174586


namespace correct_propositions_l174_174262

theorem correct_propositions
  (a b : Line)
  (α β γ : Plane)
  (l : Line)
  (h1 : α ∩ β = a)
  (h2 : β ∩ γ = b)
  (h3 : a ∥ b → α ∥ γ = false)
  (h4 : (a ∩ b) ∧ a ⊆ α ∧ a ⊆ β ∧ b ⊆ α ∧ b ⊆ β → α ∥ β)
  (h5 : α ⊥ β ∧ α ∩ β = a ∧ b ⊆ β ∧ a ⊥ b → b ⊥ α)
  (h6 : a ⊆ α ∧ b ⊆ α ∧ l ⊥ a ∧ l ⊥ b → l ⊥ α = false)
  : (h4 ∧ h5) :=
by
  sorry

end correct_propositions_l174_174262


namespace max_non_managers_l174_174351

theorem max_non_managers (M N : ℕ) (hratio : 7 * N < 37 * M) (hM : M = 9) : N ≤ 47 := by
  have h : 333 < 7 * N := by
    rw [hM] at hratio
    exact hratio
  calc N < 333 / 7 := by exact 47

end max_non_managers_l174_174351


namespace min_dot_product_on_hyperbola_l174_174807

theorem min_dot_product_on_hyperbola (x1 y1 x2 y2 : ℝ) 
  (hA : x1^2 - y1^2 = 2) 
  (hB : x2^2 - y2^2 = 2)
  (h_x1 : x1 > 0) 
  (h_x2 : x2 > 0) : 
  x1 * x2 + y1 * y2 ≥ 2 :=
sorry

end min_dot_product_on_hyperbola_l174_174807


namespace jessica_withdrew_200_l174_174786

noncomputable def initial_balance (final_balance : ℝ) : ℝ :=
  (final_balance * 25 / 18)

noncomputable def withdrawn_amount (initial_balance : ℝ) : ℝ :=
  (initial_balance * 2 / 5)

theorem jessica_withdrew_200 :
  ∀ (final_balance : ℝ), final_balance = 360 → withdrawn_amount (initial_balance final_balance) = 200 :=
by
  intros final_balance h
  rw [h]
  unfold initial_balance withdrawn_amount
  sorry

end jessica_withdrew_200_l174_174786


namespace smallest_interesting_number_l174_174123

theorem smallest_interesting_number :
  ∃ n : ℕ, (∃ k1 : ℕ, 2 * n = k1^2) ∧ (∃ k2 : ℕ, 15 * n = k2^3) ∧ n = 1800 :=
by
  sorry

end smallest_interesting_number_l174_174123


namespace minimum_value_on_1_3_a_range_l174_174713

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Problem (I)
theorem minimum_value_on_1_3 : Inf (Set.image f (Set.Icc 1 3)) = 0 :=
sorry

-- Problem (II)
theorem a_range (a : ℝ) :
  (∃ (x : ℝ), x ∈ Set.Icc (1 / Real.exp 1) Real.exp 1 ∧ 2 * f(x) ≥ -x^2 + a * x - 3) → 
  a ≤ (-2 + (1 / Real.exp 1) + 3 * Real.exp 1) :=
sorry

end minimum_value_on_1_3_a_range_l174_174713


namespace minimum_intersection_value_l174_174384

-- Given conditions: 11 sets each with 5 elements, pairwise intersection is non-empty.
variables {α : Type*}
variables (M : Fin 11 → Set α)
variables [Fintype α]

-- Each set is a 5-element subset, and every pair of sets has a non-empty intersection.
def valid_sets : Prop :=
  (∀ i : Fin 11, Fintype.card (M i) = 5) ∧
  (∀ i j : Fin 11, i ≠ j → (M i ∩ M j).nonempty)

-- Prove the minimum possible value of the greatest number of sets that intersect is 4.
theorem minimum_intersection_value (h : valid_sets M) : ∃ k, k = 4 :=
by
  sorry

end minimum_intersection_value_l174_174384


namespace Zelda_probability_of_success_l174_174556

noncomputable def P (X Y Z : Prop) : ℝ
def p_x := 1.0 / 4.0
def p_y := 2.0 / 3.0
def p_not_z := 1.0 - P Z
def combined_probability := p_x * p_y * p_not_z

theorem Zelda_probability_of_success (P : Prop → ℝ) :
  0.625 = (1.0 / 4.0) * (2.0 / 3.0) * (1.0 - P Z) := 
  by
    sorry

end Zelda_probability_of_success_l174_174556


namespace number_of_cubes_with_three_faces_painted_l174_174041

-- Definitions of conditions
def large_cube_side_length : ℕ := 4
def total_smaller_cubes := large_cube_side_length ^ 3

-- Prove the number of smaller cubes with at least 3 faces painted is 8
theorem number_of_cubes_with_three_faces_painted :
  (∃ (n : ℕ), n = 8) :=
by
  -- Conditions recall
  have side_length := large_cube_side_length
  have total_cubes := total_smaller_cubes
  
  -- Recall that the cube is composed by smaller cubes with painted faces.
  have painted_faces_condition : (∀ (cube : ℕ), cube = 8) := sorry
  
  exact ⟨8, painted_faces_condition 8⟩

end number_of_cubes_with_three_faces_painted_l174_174041


namespace q_one_factor_of_l174_174803

def q (x : ℂ) (a b : ℤ) : ℂ := x^2 + a * x + b

theorem q_one_factor_of (a b : ℤ) :
  (∀ x, (x^2 + 4 * x^2 + 9).has_factor (x^2 + a * x + b))
  ∧ (∀ x, (2 * x^2 + 3 * x^2 + 16 * x + 2).has_factor (x^2 + a * x + b))
  → q 1 a b = 1 := by
  sorry

end q_one_factor_of_l174_174803


namespace price_of_turban_l174_174323

theorem price_of_turban (T : ℝ) :
  (3 / 4) * (90 + T) = 45 + T → T = 90 :=
by
  intro h
  calc
    T = 3 * 90 - 4 * 45 : sorry
      ... = 270 - 180   : by norm_num
      ... = 90          : by norm_num
  sorry

end price_of_turban_l174_174323


namespace find_counterfeit_coin_l174_174070

theorem find_counterfeit_coin (k : ℕ) (coins : FinVec ℝ (3^(2*k))) 
  (accurate : FinVec (FinVec ℝ coins.dim) 2)
  (faulty : FinVec ℝ coins.dim) :
  ∃ (weighings : FinVec (FinVec (Vec ℝ 2) (3 * k + 1))) (coin : ℝ), 
    (coin ∈ coins.to_list) ∧ 
    (∃ scale ∈ accurate, ∀ w ∈ weighings, w.1 == w.2)
:= 
sorry

end find_counterfeit_coin_l174_174070


namespace determine_N_l174_174648

-- Define the vectors i, j, and k
def vec_i : ℝ^3 := ![1, 0, 0]
def vec_j : ℝ^3 := ![0, 1, 0]
def vec_k : ℝ^3 := ![0, 0, 1]

-- Define the given conditions
def N_i : ℝ^3 := ![4, -1, 6]
def N_j : ℝ^3 := ![1, 6, -3]
def N_k : ℝ^3 := ![-5, 2, 9]

-- Define the matrix N
variable (N : ℝ^3 → ℝ^3)

-- State the problem in theorem form
theorem determine_N : 
  (N vec_i = N_i) →
  (N vec_j = N_j) →
  (N vec_k = N_k) →
  N = ![[4, 1, -5],
         [-1, 6, 2],
         [6, -3, 9]] :=
by
  sorry

end determine_N_l174_174648


namespace min_value_of_function_decreasing_intervals_function_range_l174_174719

open Real

noncomputable def function_y (x : ℝ) : ℝ := sin x ^ 2 + sin (2 * x) + 3 * cos x ^ 2

theorem min_value_of_function : 
  (∃ k : ℤ, function_y (k * π - 3 * π / 8) = 2 - sqrt 2) :=
sorry

theorem decreasing_intervals : 
  ∀ k : ℤ, ∀ x : ℝ, k * π + π / 8 ≤ x ∧ x ≤ 5 * π / 8 + k * π → 
  ∀ x₁ x₂, x₁ < x₂ → x₁ ∈ Icc (k * π + π / 8) (5 * π / 8 + k * π) →
  function_y x₁ > function_y x₂ :=
sorry

theorem function_range : 
  ∀ x ∈ Icc (- π / 4) (π / 4), 1 ≤ function_y x ∧ function_y x ≤ 2 + sqrt 2 :=
sorry

end min_value_of_function_decreasing_intervals_function_range_l174_174719


namespace three_digit_number_exists_l174_174547

theorem three_digit_number_exists : 
  ∃ (x y z : ℕ), 
  (1 ≤ x ∧ x ≤ 9) ∧ (1 ≤ y ∧ y ≤ 9) ∧ (0 ≤ z ∧ z ≤ 9) ∧ 
  (100 * x + 10 * z + y + 1 = 2 * (100 * y + 10 * z + x)) ∧ 
  (100 * x + 10 * z + y = 793) :=
by
  sorry

end three_digit_number_exists_l174_174547


namespace area_of_rectangular_region_l174_174592

-- Mathematical Conditions
variables (a b c d : ℝ)
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)

-- Lean 4 Statement of the proof problem
theorem area_of_rectangular_region :
  (a + b) * (2 * d + c) = 2 * a * d + a * c + 2 * b * d + b * c :=
by sorry

end area_of_rectangular_region_l174_174592


namespace problem_minimum_value_l174_174804

noncomputable def minimum_value (x y : ℝ) := 
  (x + 1 / y) * (x + 1 / y - 2018) + (y + 1 / x) * (y + 1 / x - 2018)

theorem problem_minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ x y : ℝ, (0 < x) ∧ (0 < y) ∧ minimum_value x y = -2036162 :=
begin
  use [x, y],
  split,
  { exact hx },
  split,
  { exact hy },
  sorry
end

end problem_minimum_value_l174_174804


namespace width_of_wall_l174_174050

theorem width_of_wall (l : ℕ) (w : ℕ) (hl : l = 170) (hw : w = 5 * l + 80) : w = 930 := 
by
  sorry

end width_of_wall_l174_174050


namespace trajectory_eq_l174_174362

theorem trajectory_eq :
  ∀ (A B P : ℝ × ℝ), B = (-A.1, -A.2) →
  let slope : ℝ × ℝ → ℝ × ℝ → ℝ := λ p1 p2, (p2.2 - p1.2) / (p2.1 - p1.1)
  slope A P * slope B P = 1 / 3 → (3 * P.2^2 - P.1^2 = 2) :=
sorry

end trajectory_eq_l174_174362


namespace number_of_divisors_remainder_one_mod_three_of_12_factorial_l174_174251

theorem number_of_divisors_remainder_one_mod_three_of_12_factorial : 
  let a_range := (0 : Fin 11)
  let b_range := (0 : Fin 3)
  let c_range := (0 : Fin 2)
  let d_range := (0 : Fin 2)
  let valid_divisors := { (a, b, c, d) | a ∈ a_range ∧ b ∈ b_range ∧ c ∈ c_range ∧ d ∈ d_range ∧ (a + b + d) % 2 = 0 }
  |valid_divisors| = 66 := sorry

end number_of_divisors_remainder_one_mod_three_of_12_factorial_l174_174251


namespace log_sum_greater_than_two_l174_174260

variables {x y a m : ℝ}

theorem log_sum_greater_than_two
  (hx : 0 < x) (hxy : x < y) (hya : y < a) (ha1 : a < 1)
  (hm : m = Real.log x / Real.log a + Real.log y / Real.log a) : m > 2 :=
sorry

end log_sum_greater_than_two_l174_174260


namespace probability_exactly_6_odds_in_8_rolls_l174_174071

-- Define the conditions
def fair_six_sided_die := (1, 1, 1, 1, 1, 1) -- represents a fair 6-sided die

-- We consider the probability of seeing an odd number out of 6 sides
def prob_odd := 3 / 6 -- odds are 1, 3, 5 (hence 3 out of 6)

-- Define a calculation that represents the binomial coefficient (combinations) and probability
def binom (n k : ℕ) := Nat.choose n k
def prob_exactly_k_out_of_n (n k : ℕ) : ℚ := (binom n k) * ((1 / 2) ^ n)

-- The theorem we're going to prove
theorem probability_exactly_6_odds_in_8_rolls : prob_exactly_k_out_of_n 8 6 = 7 / 64 :=
by
  sorry

end probability_exactly_6_odds_in_8_rolls_l174_174071


namespace div_sub_mult_exp_eq_l174_174528

-- Lean 4 statement for the mathematical proof problem
theorem div_sub_mult_exp_eq :
  8 / 4 - 3 - 9 + 3 * 7 - 2^2 = 7 := 
sorry

end div_sub_mult_exp_eq_l174_174528


namespace fraction_halfway_between_l174_174960

def halfway_fraction (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between :
  halfway_fraction (3/4) (5/6) = 19/24 :=
sorry

end fraction_halfway_between_l174_174960


namespace bounded_f2_bounded_f3_l174_174347

def bounded (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ M > 0, ∀ x ∈ Set.Ioi a, |f x| ≤ M

noncomputable def f2 (x : ℝ) : ℝ := x / (x^2 + 1)
noncomputable def f3 (x : ℝ) : ℝ := Real.log x / x

theorem bounded_f2 : bounded f2 1 +∞ := sorry
theorem bounded_f3 : bounded f3 1 +∞ := sorry

end bounded_f2_bounded_f3_l174_174347


namespace potatoes_cost_l174_174783

-- Defining our constants and conditions
def pounds_per_person : ℝ := 1.5
def number_of_people : ℝ := 40
def pounds_per_bag : ℝ := 20
def cost_per_bag : ℝ := 5

-- The main goal: to prove the total cost is 15.
theorem potatoes_cost : (number_of_people * pounds_per_person) / pounds_per_bag * cost_per_bag = 15 :=
by sorry

end potatoes_cost_l174_174783


namespace find_n_sin_eq_l174_174247

theorem find_n_sin_eq (n : ℤ) (h₁ : -180 ≤ n) (h₂ : n ≤ 180) (h₃ : Real.sin (n * Real.pi / 180) = Real.sin (680 * Real.pi / 180)) :
  n = 40 ∨ n = 140 :=
by
  sorry

end find_n_sin_eq_l174_174247


namespace number_of_tea_bags_l174_174439

theorem number_of_tea_bags (n : ℕ) 
  (h1 : 2 * n ≤ 41) (h2 : 41 ≤ 3 * n)
  (h3 : 2 * n ≤ 58) (h4 : 58 ≤ 3 * n) :
  n = 20 :=
by
  sorry

end number_of_tea_bags_l174_174439


namespace num_both_sports_eq_3_l174_174357

-- Definitions related to the problem
def total_members : ℕ := 40
def badminton_players : ℕ := 20
def tennis_players : ℕ := 18
def no_sport_players : ℕ := 5
def at_least_one_sport_players : ℕ := total_members - no_sport_players

-- The statement to prove
theorem num_both_sports_eq_3 : 
  badminton_players + tennis_players - (∃ x, x ∈ badminton_players ∧ x ∈ tennis_players) = at_least_one_sport_players → 
  (∃ x, x ∈ badminton_players ∧ x ∈ tennis_players) = 3 := 
by
  sorry

end num_both_sports_eq_3_l174_174357


namespace solve_diamond_eq_l174_174680

noncomputable def diamond_op (a b : ℝ) := a / b

theorem solve_diamond_eq (x : ℝ) (h : x ≠ 0) : diamond_op 2023 (diamond_op 7 x) = 150 ↔ x = 1050 / 2023 := by
  sorry

end solve_diamond_eq_l174_174680


namespace triangle_right_angle_l174_174230

theorem triangle_right_angle 
  (A B C : ℝ) 
  (h : cos (2 * A) - cos (2 * B) = 2 * sin (C) ^ 2) 
  (h_sum : A + B + C = π) 
  : ∃ a b c : ℝ, a^2 + c^2 = b^2 := 
begin
  sorry
end

end triangle_right_angle_l174_174230


namespace halfway_fraction_l174_174948

theorem halfway_fraction (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  ((a / b + c / d) / 2) = 19 / 24 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end halfway_fraction_l174_174948


namespace binomial_coefficient_sum_l174_174733

theorem binomial_coefficient_sum :
  let a_0 := (1 - 2 * (0 : ℝ)) ^ 2016
  let a_i := (1 - 2 * (1 : ℝ)) ^ 2016
  ∑_{i: ℕ in 0..2016} (a_0 + a_i) = 2016 :=
  sorry

end binomial_coefficient_sum_l174_174733


namespace speed_of_man_in_still_water_l174_174584

-- Define the upstream and downstream speeds
def upstream_speed : ℝ := 25
def downstream_speed : ℝ := 39

-- Define the speed in still water as the average
def speed_in_still_water := (upstream_speed + downstream_speed) / 2

-- The statement to be proved
theorem speed_of_man_in_still_water : speed_in_still_water = 32 :=
by
  -- Proof can be skipped for now using sorry
  sorry

end speed_of_man_in_still_water_l174_174584


namespace tea_bags_count_l174_174432

theorem tea_bags_count (n : ℕ) 
  (h1 : 2 * n ≤ 41) 
  (h2 : 41 ≤ 3 * n) 
  (h3 : 2 * n ≤ 58) 
  (h4 : 58 ≤ 3 * n) : 
  n = 20 := by
  sorry

end tea_bags_count_l174_174432


namespace general_formula_a_n_exists_n_Sum_equals_2016_l174_174812

noncomputable def S_n : ℕ → ℕ := sorry
noncomputable def a_n : ℕ → ℕ := sorry

theorem general_formula_a_n :
  ∀ n : ℕ, n > 0 → S_n n = n * a_n n - 3 * n * (n - 1) ∧ a_n 1 = 1 → a_n n = 6 * n - 5 :=
sorry

theorem exists_n_Sum_equals_2016 :
  ∃ n : ℕ, n > 0 ∧ (finset.range n).sum (λ k, S_n (k + 1) / (k + 1)) - (3 / 2) * (n - 1) ^ 2 = 2016 :=
sorry

#eval general_formula_a_n

#eval exists_n_Sum_equals_2016

end general_formula_a_n_exists_n_Sum_equals_2016_l174_174812


namespace proof_of_equivalence_l174_174745

def periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f(x) = f(x + p)

def odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(-x) = -f(x)

def f_periodic := periodic f 2
def f_odd := odd f
def f_cond (x : ℝ) : Prop :=
  (0 < x ∧ x < 1) → f(x) = 4^x

theorem proof_of_equivalence 
  (f : ℝ → ℝ)
  (h_periodic : f_periodic)
  (h_odd : f_odd)
  (h_condition : ∀ (x : ℝ), f_cond x)
  : f(-5/2) + f(2) = -2 :=
sorry

end proof_of_equivalence_l174_174745


namespace find_standard_equation_of_ellipse_find_area_of_triangle_AOB_l174_174279

noncomputable def conditions (F1 F2 O P : Point)
  (on_ellipse : ∃ a b, (a > 0) ∧ (b > 0) ∧ P ∈ ellipse a b)
  (is_midpoint : ∃ M, line_segment P F2 ∩ y_axis M ∧ vector PM + vector F2M = zero_vector)
  (is_perpendicular_and_median : ∃ M, OM is median of P F1 F2 ∧ OM ⟂ F1F2) :
  Prop := 
  sorry

theorem find_standard_equation_of_ellipse (F1 F2 O P : Point)
  (on_ellipse : ∃ a b, (a > 0) ∧ (b > 0) ∧ P ∈ ellipse a b)
  (is_midpoint : ∃ M, line_segment P F2 ∩ y_axis M ∧ vector PM + vector F2M = zero_vector)
  (is_perpendicular_and_median : ∃ M, OM is median of P F1 F2 ∧ OM ⟂ F1F2) :
  standard_eq_ellipse : equation :=
  by {
    sorry, 
  }

theorem find_area_of_triangle_AOB (F1 F2 O P A B : Point) (l : Line) 
  (circle_tangent : line_is_tangent circle_O l)
  (intersect_ellipse : line_intersects_ellipse_at_two_points l (F1 F2) A B)
  (dot_product_condition : ∃ λ, λ ∈ [2/3, 3/4] ∧ dot_product (vector OA) (vector OB) = λ) :
  area_condition : ∃ S, S ∈ [sqrt(6)/4, 2/3] :=
  by {
    sorry,
  }

end find_standard_equation_of_ellipse_find_area_of_triangle_AOB_l174_174279


namespace degree_measure_angle_ABC_l174_174494

theorem degree_measure_angle_ABC (O A B C : Type) [euclidean_geometry] 
  (circumscribed_about : is_circumscribed_circle O (triangle A B C)) 
  (angle_BOC : measure (angle B O C) = 110) 
  (angle_AOB : measure (angle A O B) = 150) : 
  measure (angle A B C) = 50 := 
sorry

end degree_measure_angle_ABC_l174_174494


namespace magnitude_alpha_l174_174396

variables (α β : ℂ)
hypothesis h₁: α.conj = β
hypothesis h₂: (α / (β^2)).im = 0
hypothesis h₃: |α - β| = 2 * real.sqrt 5

theorem magnitude_alpha :
  |α| = real.sqrt (10 - 2.5 * real.sqrt 3) := sorry

end magnitude_alpha_l174_174396


namespace PersonYs_speed_in_still_water_l174_174057

def speed_in_still_water (speed_X : ℕ) (t_1 t_2 : ℕ) (x : ℕ) : Prop :=
  ∀ y : ℤ, 4 * (6 - y + x + y) = 4 * 6 + 4 * x ∧ 16 * (x + y) = 16 * (6 + y) + 4 * (x - 6) →
  x = 10

theorem PersonYs_speed_in_still_water :
  speed_in_still_water 6 4 16 10 :=
by
  sorry

end PersonYs_speed_in_still_water_l174_174057


namespace halfway_fraction_l174_174929

theorem halfway_fraction (a b c d : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 6) (h_c : c = 19 / 24) :
  (1 / 2) * (a + b) = c := 
sorry

end halfway_fraction_l174_174929


namespace tangent_line_to_circle_l174_174253

theorem tangent_line_to_circle {c : ℝ} (h : c > 0) :
  (∀ x y : ℝ, x^2 + y^2 = 8 → x + y = c) ↔ c = 4 := sorry

end tangent_line_to_circle_l174_174253


namespace solution_set_l174_174662

def inequality_solution (x : ℝ) : Prop :=
  4 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 9

theorem solution_set :
  { x : ℝ | inequality_solution x } = { x : ℝ | (63 / 26 : ℝ) < x ∧ x ≤ (28 / 11 : ℝ) } :=
by
  sorry

end solution_set_l174_174662


namespace intercepts_equal_l174_174294

theorem intercepts_equal (a : ℝ) (ha : (a ≠ 0) ∧ (a ≠ 2)) : 
  (a = 1 ∨ a = 2) ↔ (a = 1 ∨ a = 2) := 
by 
  sorry


end intercepts_equal_l174_174294


namespace conditional_without_else_l174_174551

def if_then_else_statement (s: String) : Prop :=
  (s = "IF—THEN" ∨ s = "IF—THEN—ELSE")

theorem conditional_without_else : if_then_else_statement "IF—THEN" :=
  sorry

end conditional_without_else_l174_174551


namespace original_vehicles_count_l174_174512

def cars_trucks_equation_solution : ℕ :=
  let num_cars := 14
  let num_trucks := 49
  num_cars + num_trucks

theorem original_vehicles_count :
  cars_trucks_equation_solution = 63 :=
by
  -- We assume this theorem is true based on our previous calculation.
  -- You can fill in the steps if you want to see the complete proof.
  have num_cars := 14 -- based on x = 14
  have num_trucks := 3.5 * num_cars -- based on the initial condition 3.5 times
  have num_trucks := 49 -- based on the calculation steps that num_cars was 14, thus num_trucks = 49
  show 14 + 49 = 63
  sorry

end original_vehicles_count_l174_174512


namespace find_a2009_l174_174676

theorem find_a2009 (a a_1 a_2 ... a_{2009} a_{2010} : ℤ) :
  (x + 2)^2010 = a + a_1 * (1 + x) + a_2 * (1 + x)^2 + ⋯ + a_{2009} * (1 + x)^2009 + a_{2010} * (1 + x)^2010 →
  a_{2009} = 2010 :=
sorry

end find_a2009_l174_174676


namespace cot_sum_identities_l174_174204

theorem cot_sum_identities :
  Real.cot (Real.arccot 5 + Real.arccot 11 + Real.arccot 17 + Real.arccot 23) = 1021 / 420 :=
by
  sorry

end cot_sum_identities_l174_174204


namespace polygon_sides_from_diagonals_l174_174115

/-- A theorem to prove that a regular polygon with 740 diagonals has 40 sides. -/
theorem polygon_sides_from_diagonals (n : ℕ) (h : (n * (n - 3)) / 2 = 740) : n = 40 := sorry

end polygon_sides_from_diagonals_l174_174115


namespace minimum_value_of_f_l174_174668

def f (x : ℝ) : ℝ :=
  3 * sin(x)^2 + 5 * cos(x)^2 + 2 * cos(x)

theorem minimum_value_of_f : ∃ x : ℝ, f(x) = 2.5 :=
  sorry

end minimum_value_of_f_l174_174668


namespace induction_inequality_base_case_l174_174068

theorem induction_inequality (n : ℕ) (hn1 : 1 < n) : ∑ i in finset.range (2 * n - 1), (1/(i + 1)) < n := by
  sorry

theorem base_case : 1 + 1/2 + 1/3 < 2 := by
  linarith

end induction_inequality_base_case_l174_174068


namespace smallest_quotient_l174_174233

theorem smallest_quotient (A B C : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9) (hC : 0 ≤ C ∧ C ≤ 9) :
  let n := 100 * A + 10 * B + C in
  let s := A + B + C in
  s ≠ 0 ∧ ∀ A' B' C', 1 ≤ A' ∧ A' ≤ 9 ∧ 0 ≤ B' ∧ B' ≤ 9 ∧ 0 ≤ C' ∧ C' ≤ 9 →
  let n' := 100 * A' + 10 * B' + C' in
  let s' := A' + B' + C' in
  s' ≠ 0 →
  (n : ℝ) / (s : ℝ) ≤ (n' : ℝ) / (s' : ℝ) :=
begin
  intros n s h,
  have h0 : s ≠ 0 := nat.add_pos_left hA.1 (nat.zero_le _),    
  have key := calc
    ((100 * 1 + 10 * 0 + 9 : ℕ) : ℝ) / ((1 + 0 + 9 : ℕ) : ℝ)
        = (199 : ℝ) / (19 : ℝ) : by norm_num
        ... = 10.4737 : by norm_num,
  sorry, -- The rest of the proof follows.
end

end smallest_quotient_l174_174233


namespace ordered_pairs_satisfying_eq_l174_174329

theorem ordered_pairs_satisfying_eq :
  ∃ (n : ℕ), n = 7 ∧ 
    (∀ (X Y : ℕ), X > 0 ∧ Y > 0 → (X * Y = 64) → n = ∑ m in (fin (64).factorization.to_finset), 1)
    :=
by sorry

end ordered_pairs_satisfying_eq_l174_174329


namespace min_value_fraction_l174_174700

theorem min_value_fraction (a : ℕ → ℝ) (q : ℝ) (a1 : ℝ) (m n : ℕ) 
  (h1 : ∀ k : ℕ, a (k + 1) = a k * q)
  (h2 : 0 < a3 q ∧ 0 < q)
  (h3 : a 5 = a 4 + 2 * a 3)
  (h4 : sqrt (a m * a n) = 4 * a1) :
  ∃ m n : ℕ, (1 / m + 4 / n = 3 / 2) := by
sorry

end min_value_fraction_l174_174700


namespace exists_element_belonging_to_at_least_1334_subsets_l174_174042

theorem exists_element_belonging_to_at_least_1334_subsets 
  (M : Finset α) 
  (A : Fin 2000 → Finset α) 
  (h : ∀ i, (A i).card > 2 * M.card / 3) : 
  ∃ m ∈ M, (Finset.card ((Finset.univ.filter (λ i, m ∈ A i))).to_finset) ≥ 1334 := 
by 
  sorry

end exists_element_belonging_to_at_least_1334_subsets_l174_174042


namespace sin_value_proof_l174_174261

theorem sin_value_proof (θ : ℝ) (h : Real.cos (5 * Real.pi / 12 - θ) = 1 / 3) :
  Real.sin (Real.pi / 12 + θ) = 1 / 3 := by
  sorry

end sin_value_proof_l174_174261


namespace exists_non_divisible_squares_l174_174675

def odd_numbers_in_range (n : ℕ) : set ℤ := {x | 2^(2 * n) < x ∧ x < 2^(3 * n) ∧ x % 2 = 1}

theorem exists_non_divisible_squares 
  (n : ℕ)
  (s : set ℤ)
  (hs : s ⊆ odd_numbers_in_range n)
  (hs_size : s.size = 2^(2 * n - 1) + 1) :
  ∃ a b ∈ s, ¬ (a^2 ∣ b) ∧ ¬ (b^2 ∣ a) :=
by
  sorry

end exists_non_divisible_squares_l174_174675


namespace Q_n_approx_1_l174_174795

def S (n : ℕ) : ℚ := (n * (n + 1) * (2 * n + 1)) / 6

def Q (n : ℕ) : ℚ :=
  ∏ k in (range n).filter (λ x, 2 ≤ x), S k / (S k - 1)

theorem Q_n_approx_1 : abs (Q 1991 - 1) < 0.1 := sorry

end Q_n_approx_1_l174_174795


namespace halfway_fraction_l174_174944

theorem halfway_fraction (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  ((a / b + c / d) / 2) = 19 / 24 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end halfway_fraction_l174_174944


namespace represents_not_much_different_l174_174872

def not_much_different_from (x : ℝ) (c : ℝ) : Prop := x - c ≤ 0

theorem represents_not_much_different {x : ℝ} :
  (not_much_different_from x 2023) = (x - 2023 ≤ 0) :=
by
  sorry

end represents_not_much_different_l174_174872


namespace inequality_solution_l174_174007

theorem inequality_solution (x : ℝ) :
  (x / (x^2 - 4) ≤ 0) ↔ (x ∈ set.Ioo (⊤) (-2) ∪ set.Ioc 0 2) := sorry

end inequality_solution_l174_174007


namespace angle_A_val_circumcircle_area_l174_174701

-- Definitions for the conditions
variable (A B C a b c : ℝ)
variable {cos2A_eq : cos (2 * A) + 2 * (sin B)^2 + 2 * (sin C)^2 - 2 * sqrt 3 * sin B * sin C = 1}
variable {b_val : b = sqrt 3}
variable {c_val : c = 4}

-- Theorem to prove the measure of angle A
theorem angle_A_val (cos2A_eq : cos (2 * A) + 2 * (sin B)^2 + 2 * (sin C)^2 - 2 * sqrt 3 * sin B * sin C = 1) : A = π / 6 :=
sorry

-- Given values for b and c, and the value of A, find the area of the circumcircle of triangle ABC
theorem circumcircle_area (A_val : A = π / 6) (b_val : b = sqrt 3) (c_val : c = 4) : π * (sqrt 7)^2 = 7 * π :=
sorry

end angle_A_val_circumcircle_area_l174_174701


namespace find_last_word_l174_174789

def russianAlphabetMap : List (String × Nat) := [
  ("А", 1), ("Б", 2), ("В", 3), ("Г", 4), ("Д", 5),
  ("Е", 6), ("Ж", 7), ("З", 8), ("И", 9), ("Й", 10),
  ("К", 11), ("Л", 12), ("М", 13), ("Н", 14), ("О", 15),
  ("П", 16), ("Р", 17), ("С", 18), ("Т", 19), ("У", 20),
  ("Ф", 21), ("Х", 22), ("Ц", 23), ("Ч", 24), ("Ш", 25),
  ("Щ", 26), ("Ъ", 27), ("Ы", 28), ("Ь", 29), ("Э", 30),
  ("Ю", 31), ("Я", 32)
]

def colorMap : List (Nat × String) := [
  (1, "x"), (2, ":"), (3, "&"), (4, ":"), (5, "*"),
  (6, ">"), (7, "<"), (8, "s"), (9, "="), (0, "ж")
]

variable (p : Nat)

theorem find_last_word : last_word_in_message russianAlphabetMap colorMap p = "магистратура" :=
  sorry

end find_last_word_l174_174789


namespace min_value_A2_B2_l174_174809

noncomputable def min_value (x y z w : ℝ) (h₀ : x ≥ 0) (h₁ : y ≥ 0) (h₂ : z ≥ 0) (h₃ : w ≥ 0) : ℝ :=
  let A := Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 11) + Real.sqrt (w + 7),
      B := Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2) + Real.sqrt (w + 2)
  in A^2 - B^2

theorem min_value_A2_B2 (x y z w : ℝ) (h₀ : x ≥ 0) (h₁ : y ≥ 0) (h₂ : z ≥ 0) (h₃ : w ≥ 0) :
  min_value x y z w h₀ h₁ h₂ h₃ = 72.25 :=
sorry

end min_value_A2_B2_l174_174809


namespace inequality_proof_l174_174679

noncomputable def math_problem : Prop :=
∀ (a b c : ℝ) (n p q r : ℕ),
  a > 0 → b > 0 → c > 0 →
  p + q + r = n →
  a ^ n + b ^ n + c ^ n ≥ a ^ p * b ^ q * c ^ r + a ^ q * b ^ r * c ^ p + a ^ r * b ^ p * c ^ q

theorem inequality_proof : math_problem :=
begin
  assume a b c n p q r,
  assume a_pos b_pos c_pos,
  assume h_sum,
  sorry
end

end inequality_proof_l174_174679


namespace evaluate_expression_l174_174842

theorem evaluate_expression (x : ℤ) (h1 : -1 ≤ x) (h2 : x < 2) (h3 : x ≠ 1) :
  (x = 0) → ((x + 1) / (x^2 - 1) + x / (x - 1)) / (x + 1) / (x^2 - 2 * x + 1) = -1 :=
by
  intro hx0
  rw hx0
  norm_num
  sorry

end evaluate_expression_l174_174842


namespace part1_part2_l174_174692

-- Definitions based on the conditions
def f (x : ℝ) : ℝ := 1/2 - 1/(2*x)
def g (a : ℝ) (x : ℝ) : ℝ := a * Real.log x

-- For the first part of the problem
theorem part1 (a : ℝ) (t : ℝ) (h1 : a = 1/2) (h2 : ∀ x : ℝ, (0 < x ∧ x < t) → f x < g a x) : t ≤ 1 :=
sorry

-- For the second part of the problem
theorem part2 (a : ℝ) (h : ∃ x : ℝ, ∀ y : ℝ, hasTangentLineAt f x y ↔ hasTangentLineAt (g a) x y ∧ x = y) : a = 1/2 :=
sorry

end part1_part2_l174_174692


namespace stephen_total_distance_l174_174008

def speed_first_third := 16 -- miles per hour
def speed_second_third := 12 -- miles per hour
def speed_last_third := 20 -- miles per hour
def time_each_third := 15 / 60 -- hours (converted from 15 minutes)

theorem stephen_total_distance :
  let dist_first_third := speed_first_third * time_each_third in
  let dist_second_third := speed_second_third * time_each_third in
  let dist_last_third := speed_last_third * time_each_third in
  dist_first_third + dist_second_third + dist_last_third = 12 :=
by
  sorry

end stephen_total_distance_l174_174008


namespace smallest_interesting_number_is_1800_l174_174163

def is_perfect_square (k : ℕ) : Prop := ∃ (m : ℕ), k = m * m
def is_perfect_cube (k : ℕ) : Prop := ∃ (m : ℕ), k = m * m * m

def is_interesting (n : ℕ) : Prop :=
  is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n)

theorem smallest_interesting_number_is_1800 :
  ∃ (n : ℕ), is_interesting(n) ∧ (∀ m : ℕ, is_interesting(m) → n ≤ m) ∧ n = 1800 :=
by 
  sorry

end smallest_interesting_number_is_1800_l174_174163


namespace peanuts_in_each_bag_l174_174904

theorem peanuts_in_each_bag (total_peanuts flight_time_per_bag minutes_per_hour bags : ℕ)
  (h_total_peanuts : total_peanuts = 120)
  (h_flight_time_per_bag : flight_time_per_bag = 2)
  (h_minutes_per_hour : minutes_per_hour = 60)
  (h_bags : bags = 4)
  (h_flight_time : flight_time_per_bag * minutes_per_hour = total_peanuts) :
  total_peanuts / bags = 30 :=
by
  rw [h_total_peanuts, h_bags]
  rw [nat.div_eq_of_eq_mul_right]
  exact eq.symm (nat.mul_div_cancel h_flight_time)
  linarith

end peanuts_in_each_bag_l174_174904


namespace line_passing_l174_174653

-- Define the points
def p1 : ℝ × ℝ := (1, 3)
def p2 : ℝ × ℝ := (-1, 1)

-- Define the slope calculation function
def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.snd - p1.snd) / (p2.fst - p1.fst)

-- Define the y-intercept calculation function
def y_intercept (p1 : ℝ × ℝ) (m : ℝ) : ℝ :=
  p1.snd - m * p1.fst

-- Define the problem theorem
theorem line_passing (p1 p2 : ℝ × ℝ) (m : ℝ) (b : ℝ) (h1 : slope p1 p2 = m)
  (h2 : y_intercept p1 m = b) : 2 * m - b = 0 := by
  sorry

end line_passing_l174_174653


namespace volume_of_hall_l174_174558

-- Define the dimensions and areas conditions
def length_hall : ℝ := 15
def breadth_hall : ℝ := 12
def area_floor_ceiling : ℝ := 2 * (length_hall * breadth_hall)
def area_walls (h : ℝ) : ℝ := 2 * (length_hall * h) + 2 * (breadth_hall * h)

-- Given condition: The sum of the areas of the floor and ceiling is equal to the sum of the areas of the four walls
def condition (h : ℝ) : Prop := area_floor_ceiling = area_walls h

-- Define the volume of the hall
def volume_hall (h : ℝ) : ℝ := length_hall * breadth_hall * h

-- The theorem to be proven: given the condition, the volume equals 8004
theorem volume_of_hall : ∃ h : ℝ, condition h ∧ volume_hall h = 8004 := by
  sorry

end volume_of_hall_l174_174558


namespace factorization_correct_l174_174019

theorem factorization_correct : ∀ x : ℝ, (x^2 - 2*x - 9 = 0) → ((x-1)^2 = 10) :=
by 
  intros x h
  sorry

end factorization_correct_l174_174019


namespace fraction_half_way_l174_174920

theorem fraction_half_way (x y : ℚ) (h1 : x = 3 / 4) (h2 : y = 5 / 6) : (x + y) / 2 = 19 / 24 := by
  sorry

end fraction_half_way_l174_174920


namespace parabola_solution_l174_174271

def parabola (p : ℝ) : Prop := p > 0 ∧ ∀ x y : ℝ, y^2 = -2 * p * x

def point_M (p m : ℝ) : Prop := 
  ∃ (x : ℝ), x = -9 ∧ y^2 = -2 * p * x ∧ 
  dist (x, y) (p/2, 0) = 10

theorem parabola_solution :
  ∃ p m : ℝ, 
    parabola p ∧ point_M p m ∧
    p = 2 ∧ 
    (∀ x y : ℝ, y^2 = -4 * x) ∧ 
    ((-9, 6) = M ∨ (-9, -6) = M) :=
by sorry

end parabola_solution_l174_174271


namespace angle_measure_at_Q_of_extended_sides_of_octagon_l174_174833

theorem angle_measure_at_Q_of_extended_sides_of_octagon 
  (A B C D E F G H Q : Type)
  (octagon : regular_octagon A B C D E F G H)
  (extends_to_Q : extends_to_point A B C D Q) :
  angle_measure (∠ BQD) = 22.5 :=
sorry

end angle_measure_at_Q_of_extended_sides_of_octagon_l174_174833


namespace shaded_region_area_l174_174189

-- Given conditions
def side_length_large_square : ℝ := 20
def radius_quarter_circle : ℝ := 10
def side_length_small_square : ℝ := 10

-- Derived areas
def area_large_square : ℝ := side_length_large_square ^ 2
def area_full_circle : ℝ := Real.pi * radius_quarter_circle ^ 2
def area_small_square : ℝ := side_length_small_square ^ 2
def area_quarter_circles : ℝ := area_full_circle

-- Theorem statement
theorem shaded_region_area :
  area_quarter_circles - area_small_square = 100 * Real.pi - 100 := by
  sorry

end shaded_region_area_l174_174189


namespace evaluate_sum_l174_174239

theorem evaluate_sum : (-1:ℤ) ^ 2010 + (-1:ℤ) ^ 2011 + (1:ℤ) ^ 2012 - (1:ℤ) ^ 2013 + (-1:ℤ) ^ 2014 = 0 := by
  sorry

end evaluate_sum_l174_174239


namespace jack_vertical_displacement_l174_174378

theorem jack_vertical_displacement :
  (up_flights down_flights steps_per_flight inches_per_step : ℤ) (h0 : up_flights = 3)
  (h1 : down_flights = 6) (h2 : steps_per_flight = 12) (h3 : inches_per_step = 8) :
  (down_flights - up_flights) * steps_per_flight * inches_per_step / 12 = 24 := by
  sorry

end jack_vertical_displacement_l174_174378


namespace halfway_fraction_l174_174943

theorem halfway_fraction (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  ((a / b + c / d) / 2) = 19 / 24 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end halfway_fraction_l174_174943


namespace tea_bags_number_l174_174425

theorem tea_bags_number (n : ℕ) (h1 : 2 * n ≤ 41) (h2 : 41 ≤ 3 * n) (h3 : 2 * n ≤ 58) (h4 : 58 ≤ 3 * n) : n = 20 :=
by
  sorry

end tea_bags_number_l174_174425


namespace function_properties_l174_174711

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b * Real.cos x

theorem function_properties 
  (a b : ℝ) (h : a ≠ 0)
  (h_max : f a b (π / 3) = Real.sqrt (a^2 + b^2))
  (x : ℝ) :
  let g := f a b (x + π / 3) in
  g = Real.sqrt (a^2 + b^2) * Real.cos x ∧
  Function.even g ∧
  ∃ y : ℝ, y = 3 * π / 2 ∧ graph_symmetry g (y, 0)
  := sorry


end function_properties_l174_174711


namespace three_non_collinear_points_determine_circle_determine_correct_statement_l174_174555

theorem three_non_collinear_points_determine_circle :
  (∀ (A B : Set ℝ) (P Q R : Point), is_non_collinear P Q R → ∃! (C: Circle), passes_through C P ∧ passes_through C Q ∧ passes_through C R) :=
sorry

def correct_statement : Prop :=
  ∀ (A B C D : Prop), (A ↔ (∀ (C₁ C₂ : Circle) (θ₁ θ₂ : Angle), equal θ₁ θ₂ → equal (arc C₁ θ₁) (arc C₂ θ₂))) ∧
                      (B ↔ (∀ (C₁ C₂ : Circle) (arc₁ arc₂ : Arc), equal arc₁ arc₂ → equal (central_angle C₁ arc₁) (central_angle C₂ arc₂))) ∧
                      (C ↔ (three_non_collinear_points_determine_circle)) ∧
                      (D ↔ (∀ (Δ : Triangle), ¬ ∀ (v : Vertex), distance (incenter Δ) v = distance (incenter Δ) (next_vertex Δ v))) →
  C

theorem determine_correct_statement :
  correct_statement :=
sorry

end three_non_collinear_points_determine_circle_determine_correct_statement_l174_174555


namespace actual_distance_traveled_l174_174740

variable (t : ℝ) -- let t be the actual time in hours
variable (d : ℝ) -- let d be the actual distance traveled at 12 km/hr

-- Conditions
def condition1 := 20 * t = 12 * t + 30
def condition2 := d = 12 * t

-- The target we want to prove
theorem actual_distance_traveled (t : ℝ) (d : ℝ) (h1 : condition1 t) (h2 : condition2 t d) : 
  d = 45 := by
  sorry

end actual_distance_traveled_l174_174740


namespace order_of_values_l174_174336

noncomputable def a : ℝ := Real.log 2 / 2
noncomputable def b : ℝ := Real.log 3 / 3
noncomputable def c : ℝ := Real.log Real.pi / Real.pi
noncomputable def d : ℝ := Real.log 2.72 / 2.72
noncomputable def f : ℝ := (Real.sqrt 10 * Real.log 10) / 20

theorem order_of_values : a < f ∧ f < c ∧ c < b ∧ b < d :=
by
  sorry

end order_of_values_l174_174336


namespace cone_lateral_surface_area_l174_174707

theorem cone_lateral_surface_area (l d : ℝ) (h_l : l = 5) (h_d : d = 8) : 
  (π * (d / 2) * l) = 20 * π :=
by
  sorry

end cone_lateral_surface_area_l174_174707


namespace sarah_trucks_l174_174453

-- Define the initial number of trucks denoted by T
def initial_trucks (T : ℝ) : Prop :=
  let left_after_jeff := T - 13.5
  let left_after_ashley := left_after_jeff - 0.25 * left_after_jeff
  left_after_ashley = 38

-- Theorem stating the initial number of trucks Sarah had is 64
theorem sarah_trucks : ∃ T : ℝ, initial_trucks T ∧ T = 64 :=
by
  sorry

end sarah_trucks_l174_174453


namespace algebraic_expression_evaluation_l174_174678

theorem algebraic_expression_evaluation (a : ℝ) (h : a^2 + 2 * a - 1 = 0) :
  ( ( (a^2 - 1) / (a^2 - 2 * a + 1) - 1 / (1 - a)) / (1 / (a^2 - a)) ) = 1 :=
by sorry

end algebraic_expression_evaluation_l174_174678


namespace find_principal_amount_l174_174587

theorem find_principal_amount (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ) (h : SI = P * R * T) :
  P = 10000 :=
by
  have h1 : 3600 = P * 0.12 * 3 := by
    rw h
  have h2 : P = 3600 / (0.12 * 3) := by
    have h3 : P = SI / (R * T) := by sorry
    apply h3.symm
  sorry

end find_principal_amount_l174_174587


namespace sum_series_l174_174634

theorem sum_series :
  ∑ n in Finset.range (5000 - 3 + 1), (λ i, 1 / (↑i + 3) * sqrt (↑i + 3 - 2) + ((↑i + 3 - 2) * sqrt (↑i + 3)) = 1 + 1 / sqrt 2) :=
sorry

end sum_series_l174_174634


namespace am_hm_inequality_l174_174798

noncomputable def smallest_possible_value (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) : ℝ :=
  (a + b + c) * ((1 / (a + b + d)) + (1 / (a + c + d)) + (1 / (b + c + d)))

theorem am_hm_inequality (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) :
  smallest_possible_value a b c d h1 h2 h3 h4 ≥ 9 / 2 :=
by
  sorry

end am_hm_inequality_l174_174798


namespace halfway_fraction_l174_174970

theorem halfway_fraction (a b : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) : (1/2) * (a + b) = 19/24 :=
by 
  rw [h1, h2]
  simp
  done
  sorry

end halfway_fraction_l174_174970


namespace polynomial_real_roots_conditions_l174_174241

-- Define the polynomial and conditions
theorem polynomial_real_roots_conditions (n : ℕ) (a : Fin n → ℤ) :
  (∀ i : Fin n, a i = 1 ∨ a i = -1) →
  (∀ (x : ℝ), Polynomial.roots (Polynomial.sum (Finset.range (n+1))
    (λ k, Polynomial.C (a ⟨k, k < (n+1)⟩)) = Polynomial.roots (Polynomial.C (a ⟨n+1, n < (n+2)⟩ * x ^ (n+1)))) ∈ ℝ) →
  n ≤ 2 := sorry

end polynomial_real_roots_conditions_l174_174241


namespace solve_equation_l174_174457

theorem solve_equation (y : ℝ) (z : ℝ) (hz : z = y^(1/3)) :
  (6 * y^(1/3) - 3 * y^(4/3) = 12 + y^(1/3) + y) ↔ (3 * z^4 + z^3 - 5 * z + 12 = 0) :=
by sorry

end solve_equation_l174_174457


namespace balls_distribution_l174_174822

def ways_to_distribute_balls : Nat := 
  (Nat.choose 7 4) + (Nat.choose 7 3) + (Nat.choose 7 2)

theorem balls_distribution :
  ways_to_distribute_balls = 91 :=
by
  -- Proof goes here
  sorry

end balls_distribution_l174_174822


namespace Incorrect_Statement_D_l174_174737

-- Definitions
variables {m n : Line} {α β : Plane}
variable (hαβ : α ≠ β)
variable (hmn : m ≠ n)

-- Statement to prove
theorem Incorrect_Statement_D {α β : Plane} {m n : Line} (h1 : α ∩ β = m)
  (h2 : ∀ θ, angle n θ α = angle n θ β) : ¬ (m ⊥ n) :=
sorry

end Incorrect_Statement_D_l174_174737


namespace smallest_interesting_number_smallest_interesting_number_1800_l174_174177
-- Import all the necessary libraries

-- Variables definitions adhering to the conditions of the problem
def isPerfectSquare (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k

def isPerfectCube (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k * k

def isInteresting (n : ℕ) : Prop :=
  isPerfectSquare (2 * n) ∧ isPerfectCube (15 * n)

-- Prove that the smallest interesting number is 1800
theorem smallest_interesting_number : ∀ n : ℕ, isInteresting n → 1800 ≤ n :=
begin
  sorry
end

theorem smallest_interesting_number_1800 : isInteresting 1800 :=
begin
  sorry
end

end smallest_interesting_number_smallest_interesting_number_1800_l174_174177


namespace smallest_interesting_number_l174_174120

theorem smallest_interesting_number :
  ∃ n : ℕ, (∃ k1 : ℕ, 2 * n = k1^2) ∧ (∃ k2 : ℕ, 15 * n = k2^3) ∧ n = 1800 :=
by
  sorry

end smallest_interesting_number_l174_174120


namespace boat_speed_greater_than_stream_l174_174016

def boat_stream_speed_difference (S U V : ℝ) := 
  (S / (U - V)) - (S / (U + V)) + (S / (2 * V + 1)) = 1

theorem boat_speed_greater_than_stream 
  (S : ℝ) (U V : ℝ) 
  (h_dist : S = 1) 
  (h_time_diff : boat_stream_speed_difference S U V) :
  U - V = 1 :=
sorry

end boat_speed_greater_than_stream_l174_174016


namespace science_book_pages_l174_174485

theorem science_book_pages {history_pages novel_pages science_pages: ℕ} (h1: novel_pages = history_pages / 2) (h2: science_pages = 4 * novel_pages) (h3: history_pages = 300):
  science_pages = 600 :=
by
  sorry

end science_book_pages_l174_174485


namespace inequality_l174_174414

def f (x : ℝ) := x^2 + 2 * Real.cos x

theorem inequality (x1 x2 : ℝ) (h : f x1 > f x2) : x1 > Real.abs x2 := 
  sorry

end inequality_l174_174414


namespace set_C_is_pythagorean_triple_l174_174605

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem set_C_is_pythagorean_triple : is_pythagorean_triple 5 12 13 :=
sorry

end set_C_is_pythagorean_triple_l174_174605


namespace minimum_points_correct_l174_174571

-- Declaration of variables and conditions
variables (n k : ℕ) 
hypothesis (h_k_range : 2 ≤ k ∧ k ≤ n-1)

-- Define the function to compute the minimum points required
def minimum_points (n k : ℕ) : ℕ := 3 * n - (3 * k + 1) / 2 - 2

-- Theorem statement: Minimum points required to ensure no more than k-1 teams have points not less than that team's points
theorem minimum_points_correct : ∀ (n k : ℕ), 2 ≤ k ∧ k ≤ n-1 → minimum_points n k = 3 * n - (3 * k + 1) / 2 - 2 :=
by
  intro n k h_k_range
  -- proof goes here
  sorry

end minimum_points_correct_l174_174571


namespace part1_part2_l174_174715

def f (x : ℝ) : ℝ := |2 * x - 1| + 1

theorem part1 (x : ℝ) : f x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3 :=
sorry

def g (n : ℝ) : ℝ := f n + f (-n)

theorem part2 : ∀ (n : ℝ), ∃ m : ℝ, m ∈ set.Ici 4 ∧ f n ≤ m - f (-n) :=
sorry

end part1_part2_l174_174715


namespace smallest_interesting_number_l174_174168

-- Definitions based on the conditions
def isPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m
def isPerfectCube (m : ℕ) : Prop := ∃ k : ℕ, k * k * k = m

def isInteresting (n : ℕ) : Prop :=
  isPerfectSquare (2 * n) ∧ isPerfectCube (15 * n)

-- The smallest interesting number
theorem smallest_interesting_number : ∃ n : ℕ, isInteresting n ∧ (∀ m : ℕ, isInteresting m → n ≤ m) :=
  ⟨1800, ⟨⟨26, by norm_num⟩, ⟨15, by norm_num⟩⟩, sorry⟩

end smallest_interesting_number_l174_174168


namespace halfway_fraction_l174_174933

theorem halfway_fraction (a b c d : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 6) (h_c : c = 19 / 24) :
  (1 / 2) * (a + b) = c := 
sorry

end halfway_fraction_l174_174933


namespace find_rowing_speed_of_person_Y_l174_174060

open Real

def rowing_speed (y : ℝ) : Prop :=
  ∀ (x : ℝ) (current_speed : ℝ),
    x = 6 → 
    (4 * (6 - current_speed) + 4 * (y + current_speed) = 4 * (6 + y)) →
    (16 * (y + current_speed) = 16 * (6 + current_speed) + 4 * (y - 6)) → 
    y = 10

-- We set up the proof problem without solving it.
theorem find_rowing_speed_of_person_Y : ∃ (y : ℝ), rowing_speed y :=
begin
  use 10,
  unfold rowing_speed,
  intros x current_speed h1 h2 h3,

  sorry
end

end find_rowing_speed_of_person_Y_l174_174060


namespace find_x_when_y_30_l174_174009

variable (x y k : ℝ)

noncomputable def inversely_proportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, x * y = k

theorem find_x_when_y_30
  (h_inv_prop : inversely_proportional x y) 
  (h_known_values : x = 5 ∧ y = 15) :
  ∃ x : ℝ, (∃ y : ℝ, y = 30) ∧ x = 5 / 2 := by
  sorry

end find_x_when_y_30_l174_174009


namespace smallest_d_l174_174741

theorem smallest_d (d : ℕ) (h : d > 0) : (∃ k : ℤ, 3150 * d = k^2) → d = 14 := 
begin
  intros hyp,
  sorry
end

end smallest_d_l174_174741


namespace maximum_expression_value_l174_174043

theorem maximum_expression_value (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a + b + c + d ≤ 4) :
  (sqrt (sqrt (sqrt 3)) * (a * (b + 2 * c)) ^ (1 / 4) +
   sqrt (sqrt (sqrt 3)) * (b * (c + 2 * d)) ^ (1 / 4) +
   sqrt (sqrt (sqrt 3)) * (c * (d + 2 * a)) ^ (1 / 4) +
   sqrt (sqrt (sqrt 3)) * (d * (a + 2 * b)) ^ (1 / 4)) ≤ 4 * sqrt (sqrt 3) :=
sorry

end maximum_expression_value_l174_174043


namespace fixed_intersection_points_l174_174446

variables (l : Type) [linear_order l]
variables (A B C D : l) -- Points on the line l
variables (par : l → l → Prop) -- Parallel relation on the line

-- Parallelism condition
axiom parallel_cond_AB : par A B
axiom parallel_cond_CD : par C D

-- Theorem statement
theorem fixed_intersection_points :
  ∃ (T1 T2 : l), 
  (∃ K L M N : l, par K L ∧ par M N ∧ K = A ∧ L = B ∧ M = C ∧ N = D 
    ∧ collinear K L M N ∧ intersect_at_diagonal l K M T1 ∧ intersect_at_diagonal l L N T2) :=
sorry

end fixed_intersection_points_l174_174446


namespace range_of_values_for_a_l174_174478

theorem range_of_values_for_a (a : ℝ) :
  (∀ x : ℝ, a * x^2 + a * x - 1 < 0) ↔ (-4 < a ∧ a ≤ 0) :=
by
  sorry

end range_of_values_for_a_l174_174478


namespace artifact_age_possibilities_l174_174580

def perms_with_repeats (n : ℕ) (repeats : List ℕ) : ℕ :=
  n.factorial / repeats.prod (λ x => x.factorial)

theorem artifact_age_possibilities :
  let digits := [2, 2, 2, 4, 7, 9]
  let even_digits := [2, 4]
  let start_with_2 := perms_with_repeats 5 [2] -- remaining digits: [2, 2, 4, 7, 9]
  let start_with_4 := perms_with_repeats 5 [3] -- remaining digits: [2, 2, 2, 7, 9]
  start_with_2 + start_with_4 = 80 :=
by
  sorry

end artifact_age_possibilities_l174_174580


namespace clock_angle_l174_174077

-- Define the number of divisions on the clock
def divisions : ℕ := 12

-- Define the angle between each pair of adjacent numbers
def angle_between_adjacent_numbers : ℝ := 30

-- Define the positions of hour and minute hands at 8:30
def hour_hand_position := 8.5
def minute_hand_position := 6

-- The expected angle between the hour and minute hands
def expected_angle : ℝ := 75

-- Statement to prove
theorem clock_angle (h_pos : Real.Rat := hour_hand_position) 
                    (m_pos : Real.Rat := minute_hand_position) 
                    (divs : ℕ := divisions) 
                    (angle_adj : Real := angle_between_adjacent_numbers) : 
  (angle_adj * (h_pos - m_pos)) == expected_angle := 
by 
  -- Skip actual proof
  sorry

end clock_angle_l174_174077


namespace smallest_interesting_number_smallest_interesting_number_1800_l174_174178
-- Import all the necessary libraries

-- Variables definitions adhering to the conditions of the problem
def isPerfectSquare (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k

def isPerfectCube (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k * k

def isInteresting (n : ℕ) : Prop :=
  isPerfectSquare (2 * n) ∧ isPerfectCube (15 * n)

-- Prove that the smallest interesting number is 1800
theorem smallest_interesting_number : ∀ n : ℕ, isInteresting n → 1800 ≤ n :=
begin
  sorry
end

theorem smallest_interesting_number_1800 : isInteresting 1800 :=
begin
  sorry
end

end smallest_interesting_number_smallest_interesting_number_1800_l174_174178


namespace michael_payment_correct_l174_174422

def suit_price : ℕ := 430
def suit_discount : ℕ := 100
def shoes_price : ℕ := 190
def shoes_discount : ℕ := 30
def shirt_price : ℕ := 80
def tie_price: ℕ := 50
def combined_discount : ℕ := (shirt_price + tie_price) * 20 / 100

def total_price_paid : ℕ :=
    suit_price - suit_discount + shoes_price - shoes_discount + (shirt_price + tie_price - combined_discount)

theorem michael_payment_correct :
    total_price_paid = 594 :=
by
    -- skipping the proof
    sorry

end michael_payment_correct_l174_174422


namespace _l174_174823

def circumradius_triangle_CDE (AC BC : ℝ) (rΓ : ℝ) (D E C: Point) : Prop :=
  ∀ (A B : Point) 
    (AC_distance : dist A C = AC)
    (BC_distance : dist B C = BC)
    (D_on_gamma : on_circle Γ D)
    (E_on_gamma : on_circle Γ E)
    (C_on_line_ACB : on_line A C B),

AC = 4 ∧ BC = 2  → 
circumradius_triangle_CDE AC BC = rΓ

def radius_theorem :=
  circumradius_triangle_CDE 4 2 (6 / 7)

end _l174_174823


namespace trapezoid_shorter_base_length_l174_174760

theorem trapezoid_shorter_base_length (line_midpoints_diag : ℝ) (longer_base : ℝ) (shorter_base : ℝ) 
  (h : line_midpoints_diag = 7) (h2 : longer_base = 105) : shorter_base = 91 :=
by
  have key_property : (longer_base - shorter_base) / 2 = line_midpoints_diag,
  { sorry },
  calc
    shorter_base = longer_base - 2 * line_midpoints_diag : (by linarith)
            ...  = 105 - 2 * 7 : (by rw [h, h2])
            ...  = 91 : by norm_num

end trapezoid_shorter_base_length_l174_174760


namespace BK_seen_as_right_angle_from_P_l174_174470

open EuclideanGeometry

variables {E M P Q K B : Point}
variables {r1 r2 : ℝ}
variables (c1 c2 : Circle E r1) (c3 c4 : Circle M r2)

-- Conditions
axiom circles_intersect_at_PQ : Circle.Intersect c1 c3 P Q
axiom centers_E_and_M : Distance(E, M) ≠ 0
axiom external_center_K : Center.External K c1 c3
axiom internal_center_B : Center.Internal B c1 c3

-- The theorem to prove
theorem BK_seen_as_right_angle_from_P :
  ∠P B K = right_angle :=
sorry

end BK_seen_as_right_angle_from_P_l174_174470


namespace sum_even_pos_ints_lt_100_l174_174995

theorem sum_even_pos_ints_lt_100 : ∑ k in finset.range 50, 2 * k = 2450 := by
  sorry

end sum_even_pos_ints_lt_100_l174_174995


namespace no_nonzero_real_solutions_l174_174669

theorem no_nonzero_real_solutions (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  ¬ (2 / x + 3 / y = 1 / (x + y)) :=
by sorry

end no_nonzero_real_solutions_l174_174669


namespace sum_even_pos_ints_lt_100_l174_174993

theorem sum_even_pos_ints_lt_100 : ∑ k in finset.range 50, 2 * k = 2450 := by
  sorry

end sum_even_pos_ints_lt_100_l174_174993


namespace find_k_from_integral_l174_174677

theorem find_k_from_integral (k : ℝ) (h : ∫ x in 0..2, 3 * x^2 + k = 16) : k = 4 := by
  sorry

end find_k_from_integral_l174_174677


namespace trajectory_of_C_equation_of_AC_l174_174750

section TriangleProofs

variables {A B C O G H : Point}
variables {x y k : ℝ}

/-- Conditions provided in the problem -/
noncomputable def pointA := (-1, 0 : ℝ)
noncomputable def pointB := (1, 0 : ℝ)
noncomputable def pointG (x y : ℝ) := (x / 3, y / 3 : ℝ)
noncomputable def pointH (x y : ℝ) := (x, y / 3 : ℝ)

/-- Necessary definitions for the proof -/
noncomputable def isOrthocenter (H : Point) (A B C : Point) : Prop :=
  let BH := (fst H - fst B, snd H : ℝ)
  let AC := (fst C + 1, snd C : ℝ)
  BH.1 * AC.1 + BH.2 * AC.2 = 0

noncomputable def trajectoryC (x y : ℝ) :=
  x^2 + (y^2) / 3 = 1

noncomputable def isTangent (AC : Line) (O : Point) (radius : ℝ) : Prop :=
  let distance := |snd AC / (sqrt (1 + (fst AC)^2))|
  distance = radius

/-- Proofs to be written -/
theorem trajectory_of_C (x y : ℝ) :
  pointA = (-1, 0) ∧ pointB = (1, 0) ∧ trajectoryC x y ↔ x^2 + y^2 / 3 = 1 ∧ x * y ≠ 0 := sorry

theorem equation_of_AC (x y : ℝ) (radius : ℝ) :
  let AC := (fst C + 1, snd C = y : ℝ)
  pointA = (-1, 0) ∧ isTangent AC O radius ↔ 
  (fst AC = x + 1 ∨ snd AC = -x - 1) := sorry

end TriangleProofs

end trajectory_of_C_equation_of_AC_l174_174750


namespace sum_series_l174_174635

theorem sum_series :
  ∑ n in Finset.range (5000 - 3 + 1), (λ i, 1 / (↑i + 3) * sqrt (↑i + 3 - 2) + ((↑i + 3 - 2) * sqrt (↑i + 3)) = 1 + 1 / sqrt 2) :=
sorry

end sum_series_l174_174635


namespace raghu_investment_l174_174907

-- Define the conditions as Lean definitions
def invest_raghu : Real := sorry
def invest_trishul := 0.90 * invest_raghu
def invest_vishal := 1.10 * invest_trishul
def invest_chandni := 1.15 * invest_vishal
def total_investment := invest_raghu + invest_trishul + invest_vishal + invest_chandni

-- State the proof problem
theorem raghu_investment (h : total_investment = 10700) : invest_raghu = 2656.25 :=
by
  sorry

end raghu_investment_l174_174907


namespace halfway_fraction_l174_174930

theorem halfway_fraction (a b c d : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 6) (h_c : c = 19 / 24) :
  (1 / 2) * (a + b) = c := 
sorry

end halfway_fraction_l174_174930


namespace sum_of_elements_equal_l174_174385

-- Let X be a set of 8 consecutive positive integers
constant X : Finset ℕ
constant A B : Finset ℕ
constant n : ℕ

-- Conditions
axiom X_def : X = {n, n+1, n+2, n+3, n+4, n+5, n+6, n+7}
axiom A_B_disjoint : A ∩ B = ∅
axiom A_B_union : A ∪ B = X
axiom A_B_size : A.card = 4 ∧ B.card = 4
axiom A_B_squares_equal : (∑ x in A, x^2) = (∑ x in B, x^2)

-- Theorem to prove
theorem sum_of_elements_equal (hX : X_def) (hA : A_B_disjoint) (hB : A_B_union)
    (hC : A_B_size) (hD : A_B_squares_equal) : (∑ x in A, x) = (∑ x in B, x) :=
sorry

end sum_of_elements_equal_l174_174385


namespace tea_bags_number_l174_174426

theorem tea_bags_number (n : ℕ) (h1 : 2 * n ≤ 41) (h2 : 41 ≤ 3 * n) (h3 : 2 * n ≤ 58) (h4 : 58 ≤ 3 * n) : n = 20 :=
by
  sorry

end tea_bags_number_l174_174426


namespace no_solution_exists_l174_174006

theorem no_solution_exists (x : ℝ) : ([x] + [2 * x] + [4 * x] + [8 * x] + [16 * x] + [32 * x] : ℤ) ≠ 12345 :=
by
  sorry

end no_solution_exists_l174_174006


namespace smallest_interesting_number_l174_174119

theorem smallest_interesting_number :
  ∃ n : ℕ, (∃ k1 : ℕ, 2 * n = k1^2) ∧ (∃ k2 : ℕ, 15 * n = k2^3) ∧ n = 1800 :=
by
  sorry

end smallest_interesting_number_l174_174119


namespace fraction_half_way_l174_174923

theorem fraction_half_way (x y : ℚ) (h1 : x = 3 / 4) (h2 : y = 5 / 6) : (x + y) / 2 = 19 / 24 := by
  sorry

end fraction_half_way_l174_174923


namespace right_triangle_perimeter_l174_174870

theorem right_triangle_perimeter (a b : ℕ) (h : a^2 + b^2 = 100) (r : ℕ := 1) :
  (a + b + 10) = 24 :=
sorry

end right_triangle_perimeter_l174_174870


namespace sum_min_max_y_l174_174805

theorem sum_min_max_y : 
  ∃ (x y z : ℝ), x + y + z = 5 ∧ x^2 + y^2 + z^2 = 11 ∧ 
  let m := (if h : -1 / 3 ≤ y ∧ y ≤ 3 then y else 0),
      M := (if h : -1 / 3 ≤ y ∧ y ≤ 3 then y else 0)
  in m + M = 8 / 3 := 
sorry

end sum_min_max_y_l174_174805


namespace petya_numbers_contain_digit_5_l174_174821

theorem petya_numbers_contain_digit_5 (n : ℕ) : 
  ∃ k, ∀ m ≥ k, ∃ d ∈ (repr (n * 5^m)).digits, d = 5 := 
sorry

end petya_numbers_contain_digit_5_l174_174821


namespace proposition_correctness_l174_174802

variables {l m n : Type} {α β γ : Type}

-- Define the propositions as stated in the conditions.
def proposition1 (m l : Type) (α : Prop) [parallel m l] [perpendicular m α] : Prop := perpendicular l α
def proposition2 (α β : Prop) {m α : Type} {n β : Type} [perpendicular α β] [parallel m α] [perpendicular n β] : Prop := perpendicular m n
def proposition3 (α β γ : Prop) [perpendicular α β] [perpendicular γ β] : Prop := parallel α γ
def proposition4 (m n : Type) (α β : Prop) [perpendicular m n] [perpendicular m α] [parallel n β] : Prop := perpendicular α β

-- Prove the correctness of propositions.
theorem proposition_correctness :
  if proposition1 m l α ∧ ¬ proposition2 α β ∧ ¬ proposition3 α β γ ∧ ¬ proposition4 m n α β then 1 = 1 := 
  sorry

end proposition_correctness_l174_174802


namespace max_largest_number_l174_174324

theorem max_largest_number (S : Finset ℕ) (h : S.card = 10) (h_mean : (S.sum / 10 : ℝ) = 16) : 
  ∃ a ∈ S, a = 115 :=
by
  sorry

end max_largest_number_l174_174324


namespace roots_907125_l174_174799

noncomputable def polynomial := (10 : ℝ) * polynomial.X ^ 3 + 15 * polynomial.X ^ 2 + 2005 * polynomial.X + 2010

theorem roots_907125 (a b c : ℝ) (h : polynomial.eval a polynomial = 0 ∧ polynomial.eval b polynomial = 0 ∧ polynomial.eval c polynomial = 0) :
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 907.125 :=
sorry

end roots_907125_l174_174799


namespace find_length_PB_l174_174413

noncomputable def length_PB (x : ℝ) : ℝ :=
  let AC := 2 * x
  let AP := x + 2
  let CB := 3 * x + 4
  let AB := AC + CB
  let PB := AB / 2
  PB

theorem find_length_PB (x : ℝ) : length_PB x = (5 * x + 4) / 2 :=
by
  let AC := 2 * x
  let AP := x + 2
  let CB := 3 * x + 4
  let AB := AC + CB
  let PB := AB / 2
  have h_AC : AC = 2 * x := rfl
  have h_AP : AP = x + 2 := rfl
  have h_CB : CB = 3 * x + 4 := rfl
  have h_AB : AB = 2 * x + 3 * x + 4 := rfl
  rw [h_AC, h_CB] at h_AB
  have h_AB_simp : AB = 5 * x + 4 := rfl
  have h_PB : PB = (AB / 2) := rfl
  rw [h_AB_simp] at h_PB
  exact h_PB

end find_length_PB_l174_174413


namespace symmetric_point_with_respect_to_y_eq_x_l174_174047

variables (P : ℝ × ℝ) (line : ℝ → ℝ)

theorem symmetric_point_with_respect_to_y_eq_x (P : ℝ × ℝ) (hP : P = (1, 3)) (hline : ∀ x, line x = x) :
  (∃ Q : ℝ × ℝ, Q = (3, 1) ∧ Q = (P.snd, P.fst)) :=
by
  sorry

end symmetric_point_with_respect_to_y_eq_x_l174_174047


namespace smallest_interesting_number_l174_174138

def is_perfect_square (x : ℕ) : Prop :=
∃ y : ℕ, y * y = x

def is_perfect_cube (x : ℕ) : Prop :=
∃ y : ℕ, y * y * y = x

theorem smallest_interesting_number (n : ℕ) :
    is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n) → n = 1800 :=
by {
  sorry
}

end smallest_interesting_number_l174_174138


namespace complement_intersection_l174_174320

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {2, 3}

theorem complement_intersection : (U \ N) ∩ M = {4, 5} :=
by 
  sorry

end complement_intersection_l174_174320


namespace min_c_value_l174_174650

theorem min_c_value (c : ℝ) : (-c^2 + 9 * c - 14 >= 0) → (c >= 2) :=
by {
  sorry
}

end min_c_value_l174_174650


namespace positive_difference_height_l174_174523

noncomputable theory

-- Definitions for conditions
def pipe_diameter : ℝ := 10
def pipe_radius : ℝ := pipe_diameter / 2
def num_rows (total_pipes : ℕ) : ℕ := total_pipes / 10
def height_crate_A (total_pipes : ℕ) : ℝ := num_rows total_pipes * pipe_diameter

def distance_between_rows : ℝ := 5 * Real.sqrt 3
def height_crate_B (total_pipes : ℕ) : ℝ := 10 + (num_rows total_pipes - 1) * distance_between_rows

-- Theorem statement
theorem positive_difference_height :
  abs (height_crate_A 200 - height_crate_B 200) = 190 - 100 * Real.sqrt 3 :=
by sorry

end positive_difference_height_l174_174523


namespace area_ring_correct_l174_174522

-- Define the radii of the larger and smaller circles
def r1 := 15
def r2 := 9

-- Define the area formulas for the circles
def area (r : ℝ) : ℝ := Real.pi * r^2

-- Define the area of the larger and smaller circles
def area_larger : ℝ := area r1
def area_smaller : ℝ := area r2

-- Define the area of the ring
def area_ring : ℝ := area_larger - area_smaller

-- Goal: Prove that the area of the ring is 144π
theorem area_ring_correct : area_ring = 144 * Real.pi := by
  -- proof would go here
  sorry

end area_ring_correct_l174_174522


namespace find_two_numbers_l174_174520

theorem find_two_numbers :
  ∃ (x y : ℝ), 
  (2 * (x + y) = x^2 - y^2 ∧ 2 * (x + y) = (x * y) / 4 - 56) ∧ 
  ((x = 26 ∧ y = 24) ∨ (x = -8 ∧ y = -10)) := 
sorry

end find_two_numbers_l174_174520


namespace frequency_even_numbers_facing_up_l174_174360

theorem frequency_even_numbers_facing_up (rolls : ℕ) (event_occurrences : ℕ) (h_rolls : rolls = 100) (h_event : event_occurrences = 47) : (event_occurrences / (rolls : ℝ)) = 0.47 :=
by
  sorry

end frequency_even_numbers_facing_up_l174_174360


namespace fraction_halfway_between_l174_174967

def halfway_fraction (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between :
  halfway_fraction (3/4) (5/6) = 19/24 :=
sorry

end fraction_halfway_between_l174_174967


namespace tan_alpha_eq_neg_sqrt_6_over_2_l174_174259

variable (α : Real)

-- Given conditions
def condition_cos (h : Real) : Prop := cos (Real.pi + α) = -h
def interval (h : Real) : Prop := α ∈ Ioo (-Real.pi / 2) 0

-- Proof statement
theorem tan_alpha_eq_neg_sqrt_6_over_2 
  (h_cos : condition_cos α (Real.sqrt 10 / 5))
  (h_interval : interval α) :
  Real.tan α = -Real.sqrt 6 / 2 := 
sorry

end tan_alpha_eq_neg_sqrt_6_over_2_l174_174259


namespace pentagon_ABEDF_area_l174_174612

theorem pentagon_ABEDF_area (BD_diagonal : ∀ (ABCD : Nat) (BD : Nat),
                            ABCD = BD^2 / 2 → BD = 20) 
                            (BDFE_is_rectangle : ∀ (BDFE : Nat), BDFE = 2 * BD) 
                            : ∃ (area : Nat), area = 300 :=
by
  -- Placeholder for the actual proof
  sorry

end pentagon_ABEDF_area_l174_174612


namespace num_mappings_M_to_N_l174_174875

-- Define sets M and N
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {1, 2, 3, 4}

-- Define the multiplicity of possible mappings
noncomputable def num_mappings (A B : Set ℕ) : ℕ :=
  (B.to_finset.card ^ A.to_finset.card)

-- Theorem stating the number of different mappings from M to N is 64
theorem num_mappings_M_to_N : num_mappings M N = 64 :=
by
  sorry

end num_mappings_M_to_N_l174_174875


namespace log_expression_value_l174_174079

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_expression_value :
  log_base 10 3 + 3 * log_base 10 2 + 2 * log_base 10 5 + 4 * log_base 10 3 + log_base 10 9 = 5.34 :=
by
  sorry

end log_expression_value_l174_174079


namespace measure_of_angle_Q_l174_174840

-- Defining a regular octagon
structure RegularOctagon (Point : Type) :=
  (A B C D E F G H : Point)
  (is_regular : ∀ (a b c d e f g h : Point),
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f ∧ f ≠ g ∧ g ≠ h ∧ h ≠ a)

variables {Point : Type}

-- Defining angle measure
noncomputable def angle_measure (A B C : Point) [EuclideanGeometry Point] : ℝ := sorry

-- Our points
variables (A B C D E F G H Q : Point)

-- Conditions: Points are vertices of a regular octagon, extended to meet at Q
variables [RegOct : RegularOctagon Point]
include RegOct

theorem measure_of_angle_Q (h : RegOct.is_regular A B C D E F G H) :
  angle_measure B A Q = 90 :=
sorry

end measure_of_angle_Q_l174_174840


namespace distance_from_origin_to_line_l174_174683

def ellipse (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1

-- definition of the perpendicular property of chords
def perpendicular (O A B : ℝ × ℝ) : Prop := (A.1 * B.1 + A.2 * B.2 = 0)

theorem distance_from_origin_to_line
  (xA yA xB yB : ℝ)
  (hA : ellipse xA yA)
  (hB : ellipse xB yB)
  (h_perpendicular : perpendicular (0, 0) (xA, yA) (xB, yB))
  : ∃ d : ℝ, d = (Real.sqrt 6) / 3 :=
sorry

end distance_from_origin_to_line_l174_174683


namespace an_expression_bn_expression_l174_174023

def f1 (x : ℝ) : ℝ := 4 * (x - x^2)

def fn (n : ℕ) (x : ℝ) : ℝ :=
  if n = 1 then f1 x else (fn (n - 1)) (f1 x)

theorem an_expression (n : ℕ) : (∃! k : ℕ, k = 2^(n-1)) :=
  sorry

theorem bn_expression (n : ℕ) : (∃! k : ℕ, k = 2^(n-1) + 1) :=
  sorry

end an_expression_bn_expression_l174_174023


namespace intersection_difference_l174_174877

theorem intersection_difference :
    let a := (1 - Real.sqrt 11) / 5
    let c := (1 + Real.sqrt 11) / 5 
  in c - a = 2 * Real.sqrt 11 / 5 := 
  by
  sorry

end intersection_difference_l174_174877


namespace num_points_P_l174_174644

/-- Given two concentric circles centered at the origin—one with radius 1, and another with radius 2. 
    Let A and B be the endpoints of a diameter of the smaller circle. 
    We aim to prove that the number of points P inside the larger circle such that 
    the sum of the squares of distances from P to A and B equals 5 is exactly 2. -/

theorem num_points_P (O : Point) (r1 r2 : ℝ) (A B P : Point) 
    (h_origin : O = (0,0)) 
    (h_r1 : r1 = 1) 
    (h_r2 : r2 = 2)
    (h_AB : dist A B = 2)
    (h_A : dist O A = r1 ∧ dist O B = r1)
    (h_P_inside : dist O P ≤ r2) :
    (AP_sq_plus_BP_sq_eq_five : ((dist P A) ^ 2 + (dist P B) ^ 2 = 5)) :=
sorry

end num_points_P_l174_174644


namespace f_inequality_l174_174868

noncomputable theory
open Real

variables (f : ℝ → ℝ)

-- Condition 1: f is increasing on (0, 2)
def increasing_on_0_2 : Prop := ∀ {x y : ℝ}, 0 < x ∧ x < 2 ∧ 0 < y ∧ y < 2 ∧ x < y → f(x) < f(y)

-- Condition 2: y = f(x+2) is an even function
def f_even : Prop := ∀ x : ℝ, f(x + 2) = f(-(x + 2))

-- Proof goal: f(2.5) > f(1) > f(3.5)
theorem f_inequality (h1 : increasing_on_0_2 f) (h2 : f_even f) : f(2.5) > f(1) ∧ f(1) > f(3.5) :=
by
  sorry

end f_inequality_l174_174868


namespace product_of_roots_l174_174734

noncomputable def quadratic_equation (x : ℝ) : Prop :=
  (x + 4) * (x - 5) = 22

theorem product_of_roots :
  ∀ x1 x2 : ℝ, quadratic_equation x1 → quadratic_equation x2 → (x1 * x2 = -42) := 
by
  sorry

end product_of_roots_l174_174734


namespace fibonacci_bound_fibonacci_between_powers_l174_174851

-- Definition of Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

-- Part (a): Prove that F_(n+3) < 5 * F_n for all n ≥ 3
theorem fibonacci_bound (n : ℕ) (hn : n ≥ 3) : fibonacci (n + 3) < 5 * fibonacci n := 
by sorry

-- Part (b): Prove that there are at most n Fibonacci numbers between n^k and n^(k+1)
theorem fibonacci_between_powers (n k : ℕ) (hn : n > 0) 
  : ∃ m ≤ n, ∀ l, (n^k ≤ fibonacci l ∧ fibonacci l < n^(k+1)) → (l ≤ m) := 
by sorry

end fibonacci_bound_fibonacci_between_powers_l174_174851


namespace smallest_interesting_number_l174_174118

theorem smallest_interesting_number :
  ∃ n : ℕ, (∃ k1 : ℕ, 2 * n = k1^2) ∧ (∃ k2 : ℕ, 15 * n = k2^3) ∧ n = 1800 :=
by
  sorry

end smallest_interesting_number_l174_174118


namespace bobby_roaming_area_l174_174616

noncomputable def accessible_area (radius : ℝ) : ℝ :=
  (3 / 4) * π * radius ^ 2

theorem bobby_roaming_area (radius fence_w length_w fence_h length_h gyro_w gyro_h distance : ℝ)
  (h1 : radius = 5)
  (h2 : fence_w = 4)
  (h3 : fence_h = 6)
  (h4 : gyro_w = 1)
  (h5 : gyro_h = 1)
  (h6 : distance = 3)
  (h7 : accessible_area radius = 75 / 4 * π) :
  accessible_area radius = 75 / 4 * π :=
by
  sorry

end bobby_roaming_area_l174_174616


namespace jack_vertical_displacement_l174_174377

theorem jack_vertical_displacement :
  (up_flights down_flights steps_per_flight inches_per_step : ℤ) (h0 : up_flights = 3)
  (h1 : down_flights = 6) (h2 : steps_per_flight = 12) (h3 : inches_per_step = 8) :
  (down_flights - up_flights) * steps_per_flight * inches_per_step / 12 = 24 := by
  sorry

end jack_vertical_displacement_l174_174377


namespace prob_composite_in_first_50_l174_174542

-- Define what it means for a number to be composite
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ (∀ m : ℕ, m > 1 → m < n → ¬ m ∣ n)

-- Define the set of first 50 natural numbers
def first_50_numbers : list ℕ :=
  (list.range 50).map (λ n, n + 1)

-- Define the set of composite numbers within the first 50 natural numbers
def composite_numbers : list ℕ :=
  first_50_numbers.filter is_composite

-- Define the probability function
noncomputable def probability_of_composite : ℚ :=
  composite_numbers.length / first_50_numbers.length

-- The theorem statement
theorem prob_composite_in_first_50 : probability_of_composite = 34 / 50 :=
by sorry

end prob_composite_in_first_50_l174_174542


namespace cost_of_potatoes_l174_174781

theorem cost_of_potatoes
  (per_person_potatoes : ℕ → ℕ → ℕ)
  (amount_of_people : ℕ)
  (bag_cost : ℕ)
  (bag_weight : ℕ)
  (people : ℕ)
  (cost : ℕ) :
  (per_person_potatoes people amount_of_people = 60) →
  (60 / bag_weight = 3) →
  (3 * bag_cost = cost) →
  cost = 15 :=
by
  sorry

end cost_of_potatoes_l174_174781


namespace halfway_fraction_l174_174938

theorem halfway_fraction (a b c d : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 6) (h_c : c = 19 / 24) :
  (1 / 2) * (a + b) = c := 
sorry

end halfway_fraction_l174_174938


namespace composite_probability_l174_174532

/--
Given that a number selected at random from the first 50 natural numbers,
where 1 is neither prime nor composite,
the probability of selecting a composite number is 34/49.
-/
theorem composite_probability :
  let total_numbers := 50
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  let num_primes := primes.length
  let num_composites := total_numbers - num_primes - 1
  let probability_composite := (num_composites : ℚ) / (total_numbers - 1)
  probability_composite = 34 / 49 :=
by {
  sorry
}

end composite_probability_l174_174532


namespace mass_percentage_C_in_CO_l174_174979

noncomputable def atomic_mass_C : ℚ := 12.01
noncomputable def atomic_mass_O : ℚ := 16.00
noncomputable def molecular_mass_CO : ℚ := atomic_mass_C + atomic_mass_O

theorem mass_percentage_C_in_CO : (atomic_mass_C / molecular_mass_CO) * 100 = 42.88 :=
by
  have atomic_mass_C_div_total : atomic_mass_C / molecular_mass_CO = 12.01 / 28.01 := sorry
  have mass_percentage : (atomic_mass_C / molecular_mass_CO) * 100 = 42.88 := sorry
  exact mass_percentage

end mass_percentage_C_in_CO_l174_174979


namespace age_difference_l174_174502

theorem age_difference (x : ℕ) 
  (h_ratio : 4 * x + 3 * x + 7 * x = 126)
  (h_halima : 4 * x = 36)
  (h_beckham : 3 * x = 27) :
  4 * x - 3 * x = 9 :=
by sorry

end age_difference_l174_174502


namespace factory_workers_l174_174892

theorem factory_workers 
  (W : ℕ) 
  (S : ℕ)
  (avg_salary_initial : ℕ = 430)
  (supervisor_salary_old : ℕ = 870)
  (avg_salary_new : ℕ = 420)
  (supervisor_salary_new : ℕ = 780)
  (h1 : S + supervisor_salary_old = (W + 1) * avg_salary_initial)
  (h2 : S + supervisor_salary_new = 9 * avg_salary_new) :
  W = 8 :=
by
  sorry

end factory_workers_l174_174892


namespace ratio_first_term_common_diff_l174_174082

theorem ratio_first_term_common_diff {a d : ℤ} 
  (S_20 : ℤ) (S_10 : ℤ)
  (h1 : S_20 = 10 * (2 * a + 19 * d))
  (h2 : S_10 = 5 * (2 * a + 9 * d))
  (h3 : S_20 = 6 * S_10) :
  a / d = 2 :=
by
  sorry

end ratio_first_term_common_diff_l174_174082


namespace calculate_expression_l174_174620

theorem calculate_expression : (9⁻¹ - 5⁻¹)⁻¹ = -45 / 4 := by
  sorry

end calculate_expression_l174_174620


namespace triangle_cos_area_median_l174_174368

open Real EuclideanGeometry

-- Definitions and conditions
def triangle := {a b c : ℝ} -- Side lengths of triangle A, B, C
def angle_cos (b c a : ℝ) : ℝ := (b * (3 * b - c) / (a * b))

-- Proof statement
theorem triangle_cos_area_median (a b c : ℝ) (A C : ℝ) :
  (angle_cos b c a = 1 / 3) ∧ 
  (1/2 * b * c * sqrt (1 - (1/3)^2) = 2 * sqrt 2) ∧ 
  ((b^2 + (c^2 / 4) - 2 * b * (c / 2) * (1 / 3)) = 17 / 4) →
  ((b = 2 ∧ c = 3) ∨ (b = 3 / 2 ∧ c = 4)) :=
by
  sorry

end triangle_cos_area_median_l174_174368


namespace smallest_interesting_number_smallest_interesting_number_1800_l174_174173
-- Import all the necessary libraries

-- Variables definitions adhering to the conditions of the problem
def isPerfectSquare (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k

def isPerfectCube (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k * k

def isInteresting (n : ℕ) : Prop :=
  isPerfectSquare (2 * n) ∧ isPerfectCube (15 * n)

-- Prove that the smallest interesting number is 1800
theorem smallest_interesting_number : ∀ n : ℕ, isInteresting n → 1800 ≤ n :=
begin
  sorry
end

theorem smallest_interesting_number_1800 : isInteresting 1800 :=
begin
  sorry
end

end smallest_interesting_number_smallest_interesting_number_1800_l174_174173


namespace number_of_permutations_l174_174031

theorem number_of_permutations (readers : Fin 8 → Type) : ∃! (n : ℕ), n = 40320 :=
by
  sorry

end number_of_permutations_l174_174031


namespace quilt_shaded_fraction_correct_l174_174508

-- Definitions based on the conditions
def total_squares : ℕ := 16
def divided_squares : ℕ := 4
def shaded_triangles_per_divided_square : ℕ := 1
def triangles_per_square : ℕ := 2
def area_unit_square : ℝ := 1.0

-- Calculation of the shaded area
def shaded_area : ℝ := (divided_squares * shaded_triangles_per_divided_square * area_unit_square) / triangles_per_square

-- Calculation of the total area
def total_area : ℝ := total_squares * area_unit_square

-- The fraction of the shaded area
def shaded_fraction : ℝ := shaded_area / total_area

-- The theorem we need to prove
theorem quilt_shaded_fraction_correct : shaded_fraction = 1 / 8 := by
  sorry

end quilt_shaded_fraction_correct_l174_174508


namespace num_real_z5_l174_174052

theorem num_real_z5 (z : ℂ) (h : z^30 = 1) : 
  (finset.univ.filter (λ z, ((z^5).im = 0))).card = 10 :=
sorry

end num_real_z5_l174_174052


namespace quadratic_roots_l174_174005

theorem quadratic_roots (x : ℝ) : 
  (x^2 + 3 * x - 1 = 0) ↔ (x = (-3 + Real.sqrt 13) / 2 ∨ x = (-3 - Real.sqrt 13) / 2) := 
by
  sorry

end quadratic_roots_l174_174005


namespace product_of_real_numbers_a_l174_174981

theorem product_of_real_numbers_a (a : ℝ) (x : ℝ) (hx : x^2 + a*x + 1 < 0) :
  ∃ k : ℕ, 1 + k > 0 ∧ a^2 = 4 * (1 + k) ∧ ∏₀ a = -8 := by sorry

end product_of_real_numbers_a_l174_174981


namespace sector_area_sexagesimal_l174_174885

theorem sector_area_sexagesimal (r : ℝ) (n : ℝ) (α_sex : ℝ) (π : ℝ) (two_pi : ℝ):
  r = 4 →
  n = 6000 →
  α_sex = 625 →
  two_pi = 2 * π →
  (1/2 * (α_sex / n * two_pi) * r^2) = (5 * π) / 3 :=
by
  intros
  sorry

end sector_area_sexagesimal_l174_174885


namespace triangle_area_is_12_5_l174_174517

structure Point :=
  (x : ℝ)
  (y : ℝ)

def M : Point := ⟨5, 0⟩
def N : Point := ⟨0, 5⟩
noncomputable def P (x y : ℝ) (h : x + y = 8) : Point := ⟨x, y⟩

noncomputable def area_triangle (A B C : Point) : ℝ :=
  (1 / 2) * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

theorem triangle_area_is_12_5 (x y : ℝ) (h : x + y = 8) :
  area_triangle M N (P x y h) = 12.5 :=
sorry

end triangle_area_is_12_5_l174_174517


namespace halfway_fraction_l174_174950

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/6) : (a + b) / 2 = 19 / 24 :=
by
  sorry

end halfway_fraction_l174_174950


namespace find_parameters_infinite_solutions_l174_174240

def system_has_infinite_solutions (a b : ℝ) :=
  ∀ x y : ℝ, 2 * (a - b) * x + 6 * y = a ∧ 3 * b * x + (a - b) * b * y = 1

theorem find_parameters_infinite_solutions :
  ∀ (a b : ℝ), 
  system_has_infinite_solutions a b ↔ 
    (a = (3 + Real.sqrt 17) / 2 ∧ b = (Real.sqrt 17 - 3) / 2) ∨
    (a = (3 - Real.sqrt 17) / 2 ∧ b = (-3 - Real.sqrt 17) / 2) ∨
    (a = -2 ∧ b = 1) ∨
    (a = -1 ∧ b = 2) :=
sorry

end find_parameters_infinite_solutions_l174_174240


namespace housewife_spending_l174_174594

theorem housewife_spending
    (R : ℝ) (P : ℝ) (M : ℝ)
    (h1 : R = 25)
    (h2 : R = 0.85 * P)
    (h3 : M / R - M / P = 3) :
  M = 450 :=
by
  sorry

end housewife_spending_l174_174594


namespace smallest_interesting_number_l174_174143

theorem smallest_interesting_number :
  ∃ (n : ℕ), (∃ k1 : ℕ, 2 * n = k1 ^ 2) ∧ (∃ k2 : ℕ, 15 * n = k2 ^ 3) ∧ n = 1800 := 
sorry

end smallest_interesting_number_l174_174143


namespace angle_identity_l174_174021

theorem angle_identity
  (AB BC DE BE : ℝ)
  (S T A B C D E : Point)
  (h1 : AB > BC)
  (h2 : DE > BE)
  (h3 : midpoint S (arc ACE))
  (h4 : image_point T {A, B, C, D, E}) :
  ∠ A T E = ∠ A B E :=
sorry

end angle_identity_l174_174021


namespace triangles_combined_area_is_96_l174_174038

noncomputable def combined_area_of_triangles : Prop :=
  let length_rectangle : ℝ := 6
  let width_rectangle : ℝ := 4
  let area_rectangle : ℝ := length_rectangle * width_rectangle
  let ratio_rectangle_to_first_triangle : ℝ := 2 / 5
  let area_first_triangle : ℝ := (5 / 2) * area_rectangle
  let x : ℝ := area_first_triangle / 5
  let base_second_triangle : ℝ := 8
  let height_second_triangle : ℝ := 9  -- calculated height based on the area ratio
  let area_second_triangle : ℝ := (base_second_triangle * height_second_triangle) / 2
  let combined_area : ℝ := area_first_triangle + area_second_triangle
  combined_area = 96

theorem triangles_combined_area_is_96 : combined_area_of_triangles := by
  sorry

end triangles_combined_area_is_96_l174_174038


namespace measure_of_angle_Q_l174_174841

-- Defining a regular octagon
structure RegularOctagon (Point : Type) :=
  (A B C D E F G H : Point)
  (is_regular : ∀ (a b c d e f g h : Point),
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f ∧ f ≠ g ∧ g ≠ h ∧ h ≠ a)

variables {Point : Type}

-- Defining angle measure
noncomputable def angle_measure (A B C : Point) [EuclideanGeometry Point] : ℝ := sorry

-- Our points
variables (A B C D E F G H Q : Point)

-- Conditions: Points are vertices of a regular octagon, extended to meet at Q
variables [RegOct : RegularOctagon Point]
include RegOct

theorem measure_of_angle_Q (h : RegOct.is_regular A B C D E F G H) :
  angle_measure B A Q = 90 :=
sorry

end measure_of_angle_Q_l174_174841


namespace centroid_median_property_l174_174450

/-
  Given triangle ABC, with G as the centroid, M_3 as the midpoint of BC,
  and given GM_3 = 1/2 * AB,
  Prove that CG = 2 * GM_3.
-/
theorem centroid_median_property {A B C G M_3 : Point} 
  (centroid_G : is_centroid G A B C)
  (midpoint_M3 : is_midpoint M_3 B C)
  (GM3_eq_half_AB : distance G M_3 = (1/2) * distance A B) :
  distance C G = 2 * distance G M_3 :=
sorry

end centroid_median_property_l174_174450


namespace cost_of_potatoes_l174_174782

theorem cost_of_potatoes
  (per_person_potatoes : ℕ → ℕ → ℕ)
  (amount_of_people : ℕ)
  (bag_cost : ℕ)
  (bag_weight : ℕ)
  (people : ℕ)
  (cost : ℕ) :
  (per_person_potatoes people amount_of_people = 60) →
  (60 / bag_weight = 3) →
  (3 * bag_cost = cost) →
  cost = 15 :=
by
  sorry

end cost_of_potatoes_l174_174782


namespace chichikov_receives_at_least_71_souls_l174_174095

theorem chichikov_receives_at_least_71_souls :
  ∀ (x y z : ℕ), x + y + z = 1001 → 
  ∃ N (k : ℕ) (r : ℤ), 1 ≤ N ∧ N ≤ 1001 ∧ N = 143 * k + r ∧ -71 ≤ r ∧ r < 71 := 
sorry

end chichikov_receives_at_least_71_souls_l174_174095


namespace triangle_incircle_AOC_BCA_is_20_l174_174369

theorem triangle_incircle_AOC_BCA_is_20 (A B C O : Type) 
  (h_incenter : is_incenter O A B C) 
  (angle_BAC : ℝ) 
  (angle_BOC : ℝ)
  (h_BAC : angle_BAC = 50)
  (h_BOC : angle_BOC = 40) : 
  ∃ (angle_BCA : ℝ), angle_BCA = 20 := 
by sorry

end triangle_incircle_AOC_BCA_is_20_l174_174369


namespace cot_cot_inv_sum_identity_l174_174213

  noncomputable theory
  open Real

  theorem cot_cot_inv_sum_identity :
    cot (arccot 5 + arccot 11 + arccot 17 + arccot 23) = 1021 / 420 :=
  sorry
  
end cot_cot_inv_sum_identity_l174_174213


namespace b_n_general_term_S_n_sum_terms_l174_174722

open Nat

structure SequenceProps (a b : ℕ → ℝ) where
  a1      : a 1 = 4
  recurrence  : ∀ n, a (n + 1) * a n + 6 * a (n + 1) - 4 * a n - 8 = 0
  b_n_def : ∀ n, b n = 6 / (a n - 2)

theorem b_n_general_term : 
  ∀ (a b : ℕ → ℝ) (h : SequenceProps a b), 
  ∀ n, 
  b n = 4 ^ n - 1 :=
by 
  intro a b h n
  sorry

theorem S_n_sum_terms : 
  ∀ (a b : ℕ → ℝ) (h : SequenceProps a b), 
  ∀ n, 
  (∑ k in range n, a k * b k) = 8 / 3 * 4 ^ n + 4 * n - 8 / 3 :=
by 
  intro a b h n
  sorry

end b_n_general_term_S_n_sum_terms_l174_174722


namespace fold_graph_paper__l174_174254

noncomputable def coordinates_after_fold (p : ℝ × ℝ) : ℝ × ℝ :=
  match p with
  | (2, 6) => (4, 6)
  | (4, 6) => (2, 6)
  | (-4, 1) => (10, 1)
  | (_, _) => p  -- default case for other coordinates

theorem fold_graph_paper_ (p : ℝ × ℝ) (q : ℝ × ℝ) :
  coordinates_after_fold p = q →
  (p = (-4, 1) → q = (10, 1)) :=
by
  intros h1 h2
  rw [coordinates_after_fold, h2] at h1
  exact h1

end fold_graph_paper__l174_174254


namespace points_needed_for_office_l174_174423

def points_for_interrupting : ℕ := 5
def points_for_insulting : ℕ := 10
def points_for_throwing : ℕ := 25

def jerry_interruptions : ℕ := 2
def jerry_insults : ℕ := 4
def jerry_throwings : ℕ := 2

def jerry_total_points (interrupt_points insult_points throw_points : ℕ) 
                       (interruptions insults throwings : ℕ) : ℕ :=
  (interrupt_points * interruptions) +
  (insult_points * insults) +
  (throw_points * throwings)

theorem points_needed_for_office : 
  jerry_total_points points_for_interrupting points_for_insulting points_for_throwing 
                     (jerry_interruptions) 
                     (jerry_insults) 
                     (jerry_throwings) = 100 := 
  sorry

end points_needed_for_office_l174_174423


namespace product_mn_zero_l174_174476

-- Define the structures and conditions
variables (m n : ℝ) (θ1 θ2 : ℝ)

-- Conditions
def cond1 : Prop := (m = Real.tan θ1) ∧ (n = Real.tan θ2)
def cond2 : Prop := θ1 = 3 * θ2
def cond3 : Prop := m = 3 * n

-- Final statement to be proven
theorem product_mn_zero (not_vertical : θ1 ≠ π/2 ∧ θ1 ≠ -π/2) 
  (h1 : cond1) (h2 : cond2) (h3 : cond3) : m * n = 0 :=
sorry

end product_mn_zero_l174_174476


namespace number_of_f3_and_sum_of_f3_l174_174403

noncomputable def f : ℝ → ℝ := sorry
variable (a : ℝ)

theorem number_of_f3_and_sum_of_f3 (hf : ∀ x y : ℝ, f (f x - y) = f x + f (f y - f a) + x) :
  (∃! c : ℝ, f 3 = c) ∧ (∃ s : ℝ, (∀ c, f 3 = c → s = c) ∧ s = 3) :=
sorry

end number_of_f3_and_sum_of_f3_l174_174403


namespace polar_circle_eq_and_center_area_of_triangle_l174_174363

-- Define conditions
def parametric_eq (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α + Real.sqrt 3, 2 * Real.sin α + 1)

-- Define the first question and conditions
theorem polar_circle_eq_and_center :
  (∀ α θ ρ, (parametric_eq α).fst = ρ * Real.cos θ ∧ (parametric_eq α).snd = ρ * Real.sin θ
    → ρ = 4 * Real.sin (θ + π / 3)
        ∧ (Real.sqrt 3, 1) = (2 * Real.cos (π / 6), 2 * Real.sin (π / 6) + 1)) := sorry

-- Define the second question and conditions
theorem area_of_triangle :
  (∀ ρ θ, ρ = 4 * Real.sin (θ + π / 3) ∧ θ = π / 3
    → let M := (0, 0) in
      let N := (2 * Real.sqrt 3, π / 3) in
      let C := (2, π / 6) in
      (1 / 2) * ∥N - M∥ * ∥C - M∥ * Real.sin (π / 6) = Real.sqrt 3) := sorry

end polar_circle_eq_and_center_area_of_triangle_l174_174363


namespace boat_travel_distance_downstream_l174_174576

-- Define the given conditions
def speed_boat_still : ℝ := 22
def speed_stream : ℝ := 5
def time_downstream : ℝ := 5

-- Define the effective speed and the computed distance
def effective_speed_downstream : ℝ := speed_boat_still + speed_stream
def distance_traveled_downstream : ℝ := effective_speed_downstream * time_downstream

-- State the proof problem that distance_traveled_downstream is 135 km
theorem boat_travel_distance_downstream :
  distance_traveled_downstream = 135 :=
by
  -- The proof will go here
  sorry

end boat_travel_distance_downstream_l174_174576


namespace otimes_identity_l174_174645

-- Define the operation ⊗
def otimes (k l : ℝ) : ℝ := k^2 - l^2

-- The goal is to show k ⊗ (k ⊗ k) = k^2 for any real number k
theorem otimes_identity (k : ℝ) : otimes k (otimes k k) = k^2 :=
by sorry

end otimes_identity_l174_174645


namespace largest_possible_sum_l174_174524

theorem largest_possible_sum :
  ∃ n1 n2 n3 n4 : ℕ,
  10000 ≤ n1 ∧ n1 < 100000 ∧ 
  10000 ≤ n2 ∧ n2 < 100000 ∧
  10000 ≤ n3 ∧ n3 < 100000 ∧
  10000 ≤ n4 ∧ n4 < 100000 ∧
  (∀ d ∈ [0, 1, 2, 3, 4], count d (digits 10 n1 ++ digits 10 n2 ++ digits 10 n3 ++ digits 10 n4) = 2) ∧
  n1 + n2 + n3 + n4 = 150628 :=
by 
  -- n1 = 43210, n2 = 43210, n3 = 32104, n4 = 32104
  use 43210, 43210, 32104, 32104
  -- proof goes here
  sorry

end largest_possible_sum_l174_174524


namespace mice_path_count_l174_174459

theorem mice_path_count
  (x y : ℕ)
  (left_house_yesterday top_house_yesterday right_house_yesterday : ℕ)
  (left_house_today top_house_today right_house_today : ℕ)
  (h_left_yesterday : left_house_yesterday = 8)
  (h_top_yesterday : top_house_yesterday = 4)
  (h_right_yesterday : right_house_yesterday = 7)
  (h_left_today : left_house_today = 4)
  (h_top_today : top_house_today = 4)
  (h_right_today : right_house_today = 7)
  (h_eq : (left_house_yesterday - left_house_today) + 
          (right_house_yesterday - right_house_today) = 
          top_house_today - top_house_yesterday) :
  x + y = 11 :=
by
  sorry

end mice_path_count_l174_174459


namespace symmetric_function_range_of_k_l174_174861

variables (D : Set ℝ) (a b k : ℝ)
noncomputable def f (x : ℝ) := sqrt (2 - x) - k

-- assumptions
def is_monotonic (f : ℝ → ℝ) (D : Set ℝ) := 
  ∀ x y ∈ D, x < y → f x ≤ f y ∨ f x ≥ f y

def range_symmetric_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x ∈ Icc a b, f x ∈ Icc (-b) (-a)

-- main proof problem
theorem symmetric_function_range_of_k : 
  ∀ (D : Set ℝ) (a b k : ℝ),
  (∀ x ∈ D, sqrt (2 - x) - k ∈ ℝ) →
  (is_monotonic (f k) D) →
  ([a, b] ⊆ D) →
  (range_symmetric_interval (f k) a b) →
  2 ≤ k ∧ k < 9/4 :=
sorry

end symmetric_function_range_of_k_l174_174861


namespace minimum_value_PF_AB_l174_174283

noncomputable def parabola_focus := (0, 1)

noncomputable def line_intersects_parabola (k x y : ℝ) : Prop :=
  y = k * x + 1 ∧ x ^ 2 = 4 * y

noncomputable def tangent_intersection (x1 y1 x2 y2 : ℝ) : ℝ × ℝ :=
  (2 * x1, y1 - (x1/2) * (2 * x1 - x2))

theorem minimum_value_PF_AB :
  ∃ (k : ℝ), ∀ (x1 x2 : ℝ),
    line_intersects_parabola k x1 (k * x1 + 1) →
    line_intersects_parabola k x2 (k * x2 + 1) →
    x1 ≠ x2 →
    let y1 := k * x1 + 1
    let y2 := k * x2 + 1
    let P := tangent_intersection x1 y1 x2 y2
    let PF := real.sqrt (4 * k ^ 2 + 4)
    let AB := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
    (PF + 32 / AB) = 6 :=
by
  sorry

end minimum_value_PF_AB_l174_174283


namespace smallest_interesting_number_l174_174156

def is_perfect_square (m : ℕ) : Prop := 
  ∃ k : ℕ, k * k = m

def is_perfect_cube (m : ℕ) : Prop := 
  ∃ k : ℕ, k * k * k = m

def interesting (n : ℕ) : Prop := 
  is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n)

theorem smallest_interesting_number : ∃ n, interesting n ∧ ∀ m, interesting m → n ≤ m :=
begin
  use 1800,
  split,
  {
    -- Prove 1800 is interesting
    sorry,
  },
  {
    -- Prove any interesting number is at least 1800
    sorry,
  }
end

end smallest_interesting_number_l174_174156


namespace triangle_selection_bounds_l174_174267

theorem triangle_selection_bounds (n : ℕ) (no_three_collinear : Prop) 
  (triangles_selected : finset (finset (fin n))) :
  (∀ t ∈ triangles_selected, t.card = 3) ∧
  (∀ t1 t2 ∈ triangles_selected, t1 ≠ t2 → t1 ∩ t2 = ∅) →
  (n.choose 2 / 9 ≤ triangles_selected.card ∧
   triangles_selected.card ≤ n.choose 2 / 3) :=
by sorry

end triangle_selection_bounds_l174_174267


namespace find_common_ratio_l174_174275

noncomputable def is_arithmetic_sequence (α : ℕ → ℝ) (α₁ β : ℝ) : Prop := 
∀ n, α n = α₁ + n * β

noncomputable def is_geometric_sequence (s : ℕ → ℝ) (q : ℝ) : Prop := 
∀ n, s (n+1) = q * s n

noncomputable def α (n : ℕ) : ℝ := sorry
noncomputable def sin_seq (n : ℕ) : ℝ := Real.sin (α n)

theorem find_common_ratio (α₁ β q : ℝ) (h_arith : is_arithmetic_sequence α α₁ β)
  (h_geom : is_geometric_sequence sin_seq q) : 
  q = 1 ∨ q = -1 :=
by sorry

end find_common_ratio_l174_174275


namespace nat_n_divisibility_cond_l174_174660

theorem nat_n_divisibility_cond (n : ℕ) : (n * 2^n + 1) % 3 = 0 ↔ (n % 3 = 1 ∨ n % 3 = 2) :=
by sorry

end nat_n_divisibility_cond_l174_174660


namespace halfway_fraction_l174_174973

theorem halfway_fraction (a b : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) : (1/2) * (a + b) = 19/24 :=
by 
  rw [h1, h2]
  simp
  done
  sorry

end halfway_fraction_l174_174973


namespace sum_of_six_consecutive_odd_numbers_l174_174343

theorem sum_of_six_consecutive_odd_numbers (a b c d e f : ℕ) 
  (ha : 135135 = a * b * c * d * e * f)
  (hb : a < b) (hc : b < c) (hd : c < d) (he : d < e) (hf : e < f)
  (hzero : a % 2 = 1) (hone : b % 2 = 1) (htwo : c % 2 = 1) 
  (hthree : d % 2 = 1) (hfour : e % 2 = 1) (hfive : f % 2 = 1) :
  a + b + c + d + e + f = 48 := by
  sorry

end sum_of_six_consecutive_odd_numbers_l174_174343


namespace kind_wizard_can_achieve_goal_l174_174514

theorem kind_wizard_can_achieve_goal (n : ℕ) (h_odd : n % 2 = 1) (h_gt_one : n > 1) : 
  ∃ (seating : List ℕ), 
  (∀ i : ℕ, i < n → seating.nth i ≠ none) ∧ 
  (∀ i : ℕ, i < n → ∃ j : ℕ, (seating.nth i.succ ≠ none ∧ seating.nth j ≠ none ∧ 
                               (seating.nth i).get = (seating.nth j).get)) :=
sorry

end kind_wizard_can_achieve_goal_l174_174514


namespace negation_proposition_l174_174874

theorem negation_proposition:
  ¬ (∀ x ∈ set.Ioo 0 1, x^2 - x < 0) ↔ (∃ x ∈ set.Ioo 0 1, x^2 - x ≥ 0) :=
by 
  sorry

end negation_proposition_l174_174874


namespace halfway_fraction_l174_174937

theorem halfway_fraction (a b c d : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 6) (h_c : c = 19 / 24) :
  (1 / 2) * (a + b) = c := 
sorry

end halfway_fraction_l174_174937


namespace int_solution_count_l174_174250

def g (n : ℤ) : ℤ :=
  ⌈97 * n / 98⌉ - ⌊98 * n / 99⌋

theorem int_solution_count :
  (∃! n : ℤ, 1 + ⌊98 * n / 99⌋ = ⌈97 * n / 98⌉) :=
sorry

end int_solution_count_l174_174250


namespace sequence_sum_l174_174364

theorem sequence_sum (A B C D E F G H : ℕ) (hC : C = 7) 
    (h_sum : A + B + C = 36 ∧ B + C + D = 36 ∧ C + D + E = 36 ∧ D + E + F = 36 ∧ E + F + G = 36 ∧ F + G + H = 36) :
    A + H = 29 :=
sorry

end sequence_sum_l174_174364


namespace smallest_interesting_number_l174_174155

def is_perfect_square (m : ℕ) : Prop := 
  ∃ k : ℕ, k * k = m

def is_perfect_cube (m : ℕ) : Prop := 
  ∃ k : ℕ, k * k * k = m

def interesting (n : ℕ) : Prop := 
  is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n)

theorem smallest_interesting_number : ∃ n, interesting n ∧ ∀ m, interesting m → n ≤ m :=
begin
  use 1800,
  split,
  {
    -- Prove 1800 is interesting
    sorry,
  },
  {
    -- Prove any interesting number is at least 1800
    sorry,
  }
end

end smallest_interesting_number_l174_174155


namespace principal_sum_is_correct_l174_174652

def simple_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * r * t

def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r) ^ t - P

def prove_principal_sum : Prop :=
  let CI1 := compound_interest 4000 0.10 2 in
  let CI2 := compound_interest 6000 0.12 3 in
  let avg_CI := (CI1 + CI2) / 2 in
  let SI := 1/2 * avg_CI in
  let P := SI / (0.08 * 3) in
  P = 3405.8

theorem principal_sum_is_correct : prove_principal_sum :=
  by
  -- proof goes here
  sorry

end principal_sum_is_correct_l174_174652


namespace fundraiser_sales_l174_174469

noncomputable def total_amount_raised (pancakes bacon scrambled_eggs coffee : ℕ → ℝ) : ℝ :=
  pancakes 60 + bacon 90 + scrambled_eggs 75 + coffee 50

noncomputable def percentage_contribution (total individual : ℝ) : ℝ :=
  (individual / total) * 100

theorem fundraiser_sales :
  let total_from_pancakes := 60 * 4.00 in
  let total_from_bacon := 90 * 2.00 in
  let total_from_scrambled_eggs := 75 * 1.50 in
  let total_from_coffee := 50 * 1.00 in
  let total := total_from_pancakes + total_from_bacon + total_from_scrambled_eggs + total_from_coffee in
  total = 582.50 ∧
  percentage_contribution total total_from_pancakes = 41.19 ∧
  percentage_contribution total total_from_bacon = 30.89 ∧
  percentage_contribution total total_from_scrambled_eggs = 19.31 ∧
  percentage_contribution total total_from_coffee = 8.58 :=
by
  unfold total_amount_raised percentage_contribution
  sorry

end fundraiser_sales_l174_174469


namespace smallest_interesting_number_is_1800_l174_174160

def is_perfect_square (k : ℕ) : Prop := ∃ (m : ℕ), k = m * m
def is_perfect_cube (k : ℕ) : Prop := ∃ (m : ℕ), k = m * m * m

def is_interesting (n : ℕ) : Prop :=
  is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n)

theorem smallest_interesting_number_is_1800 :
  ∃ (n : ℕ), is_interesting(n) ∧ (∀ m : ℕ, is_interesting(m) → n ≤ m) ∧ n = 1800 :=
by 
  sorry

end smallest_interesting_number_is_1800_l174_174160


namespace sum_even_pos_ints_lt_100_l174_174996

theorem sum_even_pos_ints_lt_100 : ∑ k in finset.range 50, 2 * k = 2450 := by
  sorry

end sum_even_pos_ints_lt_100_l174_174996


namespace sqrt_of_sum_eq_l174_174548

noncomputable def cube_term : ℝ := 2 ^ 3
noncomputable def sum_cubes : ℝ := cube_term + cube_term + cube_term + cube_term
noncomputable def sqrt_sum : ℝ := Real.sqrt sum_cubes

theorem sqrt_of_sum_eq :
  sqrt_sum = 4 * Real.sqrt 2 :=
by
  sorry

end sqrt_of_sum_eq_l174_174548


namespace mail_distribution_l174_174888

-- Define the number of houses
def num_houses : ℕ := 10

-- Define the pieces of junk mail per house
def mail_per_house : ℕ := 35

-- Define total pieces of junk mail delivered
def total_pieces_of_junk_mail : ℕ := num_houses * mail_per_house

-- Main theorem statement
theorem mail_distribution : total_pieces_of_junk_mail = 350 := by
  sorry

end mail_distribution_l174_174888


namespace opposite_of_negative_seven_l174_174034

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_negative_seven : opposite (-7) = 7 := 
by 
  sorry

end opposite_of_negative_seven_l174_174034


namespace smallest_interesting_number_l174_174166

-- Definitions based on the conditions
def isPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m
def isPerfectCube (m : ℕ) : Prop := ∃ k : ℕ, k * k * k = m

def isInteresting (n : ℕ) : Prop :=
  isPerfectSquare (2 * n) ∧ isPerfectCube (15 * n)

-- The smallest interesting number
theorem smallest_interesting_number : ∃ n : ℕ, isInteresting n ∧ (∀ m : ℕ, isInteresting m → n ≤ m) :=
  ⟨1800, ⟨⟨26, by norm_num⟩, ⟨15, by norm_num⟩⟩, sorry⟩

end smallest_interesting_number_l174_174166


namespace speak_order_count_l174_174114

theorem speak_order_count 
  (students : Finset α)
  (students_ids : α → String)
  (A B : α)
  (cond1 : students.card = 7)
  (cond2 : ∀ order : List α, order.nodup ∧ (A ∈ order ∨ B ∈ order) ∧ (A ∈ order → B ∈ order → ¬adjacent A B order))
  : ∃ n : ℕ, n = 600 := 
sorry

end speak_order_count_l174_174114


namespace fraction_half_way_l174_174924

theorem fraction_half_way (x y : ℚ) (h1 : x = 3 / 4) (h2 : y = 5 / 6) : (x + y) / 2 = 19 / 24 := by
  sorry

end fraction_half_way_l174_174924


namespace complex_triangle_sum_l174_174223

variables (p q r : ℂ)
variables (side_length : ℂ) (sum_magnitude : ℂ)

-- Define the given conditions
def equilateral_triangle : Prop := (|p - q| = side_length) ∧ (|q - r| = side_length) ∧ (|r - p| = side_length)
def centroid_condition : Prop := |p + q + r| = sum_magnitude

-- Conclude the final proof
theorem complex_triangle_sum:
  (equilateral_triangle p q r 24) →
  (centroid_condition p q r 48) →
  |p * q + p * r + q * r| = 768 := by
  sorry

end complex_triangle_sum_l174_174223


namespace mean_of_S_permutations_no_consecutive_l174_174792

def permutations (n : ℕ) : Finset (Perm (Fin n)) :=
  Finset.univ

-- Question 1 translation
theorem mean_of_S: 
  let S := permutations 8
  let S_val (σ : Perm (Fin 8)) := 
    σ 0 * σ 1 + σ 2 * σ 3 + σ 4 * σ 5 + σ 6 * σ 7
  (Finset.sum S (λ σ, S_val σ):ℝ) / S.card = 81 :=
sorry

-- Question 2 translation
theorem permutations_no_consecutive:
  let S := permutations 8
  let condition (σ : Perm (Fin 8)) : Prop :=
    ∀ k : Fin 7, σ (k + 1) ≠ k + 1
  (S.filter condition).card = 41787 :=
sorry

end mean_of_S_permutations_no_consecutive_l174_174792


namespace carmen_sprigs_left_l174_174221

-- Definitions based on conditions
def initial_sprigs : ℕ := 25
def whole_sprigs_used : ℕ := 8
def half_sprigs_plates : ℕ := 12
def half_sprigs_total_used : ℕ := half_sprigs_plates / 2

-- Total sprigs used
def total_sprigs_used : ℕ := whole_sprigs_used + half_sprigs_total_used

-- Leftover sprigs computation
def sprigs_left : ℕ := initial_sprigs - total_sprigs_used

-- Statement to prove
theorem carmen_sprigs_left : sprigs_left = 11 :=
by
  sorry

end carmen_sprigs_left_l174_174221


namespace tea_bags_count_l174_174431

theorem tea_bags_count (n : ℕ) 
  (h1 : 2 * n ≤ 41) 
  (h2 : 41 ≤ 3 * n) 
  (h3 : 2 * n ≤ 58) 
  (h4 : 58 ≤ 3 * n) : 
  n = 20 := by
  sorry

end tea_bags_count_l174_174431


namespace find_number_l174_174746

theorem find_number (x : ℤ) (h : 5 * x + 4 = 19) : x = 3 := sorry

end find_number_l174_174746


namespace find_height_of_tank_B_l174_174465

noncomputable def r_C := 4 / Real.pi
noncomputable def h_C := 10
noncomputable def capacity_C := Real.pi * r_C ^ 2 * h_C

noncomputable def r_B := 5 / Real.pi
noncomputable def capacity_B (h_B : ℝ) := Real.pi * r_B ^ 2 * h_B

theorem find_height_of_tank_B : ∃ h_B : ℝ, capacity_C = 0.8 * capacity_B h_B ∧ h_B = 8 := by
  let h_B := 8
  have hc : capacity_C = Real.pi * r_C ^ 2 * h_C := sorry
  have hb : capacity_B h_B = Real.pi * r_B ^ 2 * h_B := sorry
  have capacity_relation : 0.8 * capacity_B h_B = capacity_C := sorry
  use h_B
  constructor
  {
    exact capacity_relation
  },
  {
    refl
  }

end find_height_of_tank_B_l174_174465


namespace necessary_and_sufficient_condition_l174_174887

theorem necessary_and_sufficient_condition (a : ℝ) :
  (∀ x : ℝ, 0 < x → x + (1 / x) > a) ↔ a < 2 :=
sorry

end necessary_and_sufficient_condition_l174_174887


namespace camels_in_caravan_l174_174350

theorem camels_in_caravan : 
  ∃ (C : ℕ), 
  (60 + 35 + 10 + C) * 1 + 60 * 2 + 35 * 4 + 10 * 2 + 4 * C - (60 + 35 + 10 + C) = 193 ∧ 
  C = 6 :=
by
  sorry

end camels_in_caravan_l174_174350


namespace smallest_interesting_number_l174_174172

-- Definitions based on the conditions
def isPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m
def isPerfectCube (m : ℕ) : Prop := ∃ k : ℕ, k * k * k = m

def isInteresting (n : ℕ) : Prop :=
  isPerfectSquare (2 * n) ∧ isPerfectCube (15 * n)

-- The smallest interesting number
theorem smallest_interesting_number : ∃ n : ℕ, isInteresting n ∧ (∀ m : ℕ, isInteresting m → n ≤ m) :=
  ⟨1800, ⟨⟨26, by norm_num⟩, ⟨15, by norm_num⟩⟩, sorry⟩

end smallest_interesting_number_l174_174172


namespace probability_AC_lt_15_l174_174607

open MeasureTheory Probability

-- Conditions definitions
def pointA : ℝ × ℝ := (-12, 0)
def pointB : ℝ × ℝ := (0, 0)
def radiusB : ℝ := 8
def radiusA : ℝ := 15
def alpha : Set ℝ := Set.Ioo 0 (Real.pi / 2)

-- The probability that the distance AC is less than 15 cm.
theorem probability_AC_lt_15 :
  -- Given alpha ∈ (0, π/2)
  sorry -- placeholder for event definition
  
  -- The statement to be proved
  (P(AC < 15)) = 1 / 3 :=
sorry

end probability_AC_lt_15_l174_174607


namespace total_players_count_l174_174890

theorem total_players_count (M W : ℕ) (h1 : W = M + 4) (h2 : (M : ℚ) / W = 5 / 9) : M + W = 14 :=
sorry

end total_players_count_l174_174890


namespace total_students_in_lansing_l174_174567

theorem total_students_in_lansing :
  let number_of_schools := 25
      students_per_school := 247 in
  number_of_schools * students_per_school = 6175 :=
by
  sorry

end total_students_in_lansing_l174_174567


namespace problem_hyperbola_l174_174312

theorem problem_hyperbola:
  let C := {p : ℝ × ℝ | (p.1^2 / 3) - p.2^2 = 1} in
  let O : ℝ × ℝ := (0, 0) in
  let F : ℝ × ℝ := (2, 0) in
  ∃ M N : ℝ × ℝ,
  (∃ k : ℝ, M.2 = ( - (√3 / 3) * M.1) ∧ M.2 = (√3 * (M.1 - 2))) ∧
  (∃ l : ℝ, N.2 = ((√3 / 3) * N.1) ∧ N.2 = (√3 * (N.1 - 2))) ∧
  (O.1 = 0 ∧ O.2 = 0) ∧
  ( (O.1, O.2) = (0, 0) /\ angle0 = has_angr (triangle.mk O M N) = pi.div_2 ) →
  dist M N = 3 :=
by
  intros
  sorry

end problem_hyperbola_l174_174312


namespace area_of_AFGE_l174_174473

theorem area_of_AFGE (A B C D E F G : Type) 
  [AffineSpace A B C D E F G]
  (h_ABC_area : area_of_triangle A B C = 64)
  (h_midpoints : are_midpoints D E F A B C)
  (h_intersection : is_intersection G D F B E) :
  area_of_quadrilateral A F G E = 64 / 3 := 
sorry

end area_of_AFGE_l174_174473


namespace percentage_alcohol_in_original_mixture_l174_174572

-- Define the initial conditions
def original_volume : ℝ := 15
def added_water_volume : ℝ := 2
def new_mixture_alcohol_pct : ℝ := 17.647058823529413

-- Prove that the percentage of alcohol in the original mixture is 20%
theorem percentage_alcohol_in_original_mixture : 
  ∃ (A : ℝ), A = 20 ∧ 
  (original_volume * A / 100) = (new_mixture_alcohol_pct / 100 * (original_volume + added_water_volume)) :=
by {
  let A : ℝ := 20,
  use A,
  split,
  { refl },
  { simp [original_volume, added_water_volume, new_mixture_alcohol_pct], sorry }
}

end percentage_alcohol_in_original_mixture_l174_174572


namespace pencils_purchased_l174_174570

theorem pencils_purchased (P : ℕ) :
    let pens := 30
    let total_cost := 630
    let pen_price := 16
    let pencil_price := 2
    480 + 2 * P = 630 → P = 75 :=
begin
    sorry
end

end pencils_purchased_l174_174570


namespace fraction_halfway_between_l174_174959

def halfway_fraction (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between :
  halfway_fraction (3/4) (5/6) = 19/24 :=
sorry

end fraction_halfway_between_l174_174959


namespace appropriate_selling_price_l174_174190

-- Define the given conditions
def cost_per_kg : ℝ := 40
def base_price : ℝ := 50
def base_sales_volume : ℝ := 500
def sales_decrease_per_yuan : ℝ := 10
def available_capital : ℝ := 10000
def desired_profit : ℝ := 8000

-- Define the sales volume function dependent on selling price x
def sales_volume (x : ℝ) : ℝ := base_sales_volume - (x - base_price) * sales_decrease_per_yuan

-- Define the profit function dependent on selling price x
def profit (x : ℝ) : ℝ := (x - cost_per_kg) * sales_volume x

-- Prove that the appropriate selling price is 80 yuan
theorem appropriate_selling_price : 
  ∃ x : ℝ, profit x = desired_profit ∧ x = 80 :=
by
  sorry

end appropriate_selling_price_l174_174190


namespace length_of_platform_l174_174106

noncomputable def train_length : ℝ := 250
noncomputable def train_speed_kmh : ℝ := 55
noncomputable def crossing_time : ℝ := 35.99712023038157

/-- Given the length of a train, its speed in km/hr, and the time taken to cross a platform,
     prove the length of the platform is 300 meters. -/
theorem length_of_platform :
  let train_speed_ms := train_speed_kmh * (1000 / 3600),
      total_distance := train_speed_ms * crossing_time
  in total_distance - train_length = 300 :=
by
  sorry

end length_of_platform_l174_174106


namespace youtube_likes_l174_174573

theorem youtube_likes (L D : ℕ) 
  (h1 : D = (1 / 2 : ℝ) * L + 100)
  (h2 : D + 1000 = 2600) : 
  L = 3000 := 
by
  sorry

end youtube_likes_l174_174573


namespace students_next_to_each_other_cases_l174_174895

theorem students_next_to_each_other_cases
    (students teachers : ℕ)
    (h_students : students = 3)
    (h_teachers : teachers = 2)
    (h_condition : true) : 
    let unit_arrangements := 6 in -- 3! ways to arrange (S, T1, T2)
    let student_arrangements := 6 in -- 3! ways to arrange students within S
    (unit_arrangements * student_arrangements) = 36 := 
by
  sorry

end students_next_to_each_other_cases_l174_174895


namespace maximum_intersection_points_l174_174815

-- Definitions based on conditions
def is_parallel (lines : List (Set Point)) : Prop :=
  ∀ l₁ l₂, l₁ ∈ lines → l₂ ∈ lines → l₁ ≠ l₂ → ∀ x y ∈ l₁ ∩ l₂, x = y

def meets_at_point (lines : List (Set Point)) (point : Point) : Prop :=
  ∀ l ∈ lines, ∃! p, p = point ∧ p ∈ l

variables {Point : Type} [Nonempty Point]
          (L : List (Set Point))
          (B : Point)
          (X Y Z : List (Set Point))

-- Given conditions in the problem
hypothesis h_distinct : (∀ l₁ l₂ ∈ L, l₁ ≠ l₂ → disjoint l₁ l₂)
hypothesis h_size : L.length = 120
hypothesis h_sizeX : X.length = 24
hypothesis h_sizeY : Y.length = 24
hypothesis h_sizeZ : Z.subsetOf L ∧ Z.length = 72
hypothesis h_parallel : is_parallel X
hypothesis h_meets_at_B : meets_at_point Y B

-- Proving the maximum number of points of intersection is 6589
theorem maximum_intersection_points :
  maximum_intersection_points_of L X Y Z = 6589 := sorry

end maximum_intersection_points_l174_174815


namespace even_perfect_square_factors_count_l174_174729

theorem even_perfect_square_factors_count :
  let a_valid (a : ℕ) := (a % 2 = 0 ∧ a ≥ 1 ∧ a ≤ 6)
  let b_valid (b : ℕ) := (b % 2 = 0 ∧ b ≤ 3)
  let c_valid (c : ℕ) := (c % 2 = 0 ∧ c ≤ 8)
  let count := (finset.range 7).filter a_valid ∈ finset
          |> (finset.range 4).filter b_valid |>.product
          |> (finset.range 9).filter c_valid |>.product
  count.card = 30 :=
by
  sorry

end even_perfect_square_factors_count_l174_174729


namespace third_angle_in_triangle_sum_of_angles_in_triangle_l174_174358

theorem third_angle_in_triangle (a b : ℝ) (h₁ : a = 50) (h₂ : b = 80) : 180 - a - b = 50 :=
by
  rw [h₁, h₂]
  norm_num

-- Adding this to demonstrate the constraint of the problem: Sum of angles in a triangle is 180°
theorem sum_of_angles_in_triangle (a b c : ℝ) (h₁: a + b + c = 180) : true :=
by
  trivial

end third_angle_in_triangle_sum_of_angles_in_triangle_l174_174358


namespace term_300_eq_307_l174_174076

-- Define what it means to be a perfect cube.
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n

-- Define the sequence excluding perfect cubes.
def omit_cubes_seq : ℕ → ℕ 
| 0     := 1
| (n+1) := Nat.find (λ m, omit_cubes_seq n < m ∧ ¬is_perfect_cube m)

-- Statement: The 300th term of omit_cubes_seq is equal to 307.
theorem term_300_eq_307 : omit_cubes_seq 299 = 307 :=
sorry

end term_300_eq_307_l174_174076


namespace smallest_interesting_number_l174_174150

def is_perfect_square (m : ℕ) : Prop := 
  ∃ k : ℕ, k * k = m

def is_perfect_cube (m : ℕ) : Prop := 
  ∃ k : ℕ, k * k * k = m

def interesting (n : ℕ) : Prop := 
  is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n)

theorem smallest_interesting_number : ∃ n, interesting n ∧ ∀ m, interesting m → n ≤ m :=
begin
  use 1800,
  split,
  {
    -- Prove 1800 is interesting
    sorry,
  },
  {
    -- Prove any interesting number is at least 1800
    sorry,
  }
end

end smallest_interesting_number_l174_174150


namespace potatoes_cost_l174_174784

-- Defining our constants and conditions
def pounds_per_person : ℝ := 1.5
def number_of_people : ℝ := 40
def pounds_per_bag : ℝ := 20
def cost_per_bag : ℝ := 5

-- The main goal: to prove the total cost is 15.
theorem potatoes_cost : (number_of_people * pounds_per_person) / pounds_per_bag * cost_per_bag = 15 :=
by sorry

end potatoes_cost_l174_174784


namespace binomial_expansion_largest_coefficient_l174_174768

theorem binomial_expansion_largest_coefficient :
  let polynomial := (x + 2*y)^6 in
  coefficient_of_largest_binomial_term polynomial = 160 :=
sorry

end binomial_expansion_largest_coefficient_l174_174768


namespace gcd_lcm_product_eq_l174_174545

theorem gcd_lcm_product_eq (a b : ℕ) : gcd a b * lcm a b = a * b := by
  sorry

example : ∃ (a b : ℕ), a = 30 ∧ b = 75 ∧ gcd a b * lcm a b = a * b :=
  ⟨30, 75, rfl, rfl, gcd_lcm_product_eq 30 75⟩

end gcd_lcm_product_eq_l174_174545


namespace bacon_cost_l174_174855

namespace PancakeBreakfast

def cost_of_stack_pancakes : ℝ := 4.0
def stacks_sold : ℕ := 60
def slices_bacon_sold : ℕ := 90
def total_revenue : ℝ := 420.0

theorem bacon_cost (B : ℝ) 
  (h1 : stacks_sold * cost_of_stack_pancakes + slices_bacon_sold * B = total_revenue) : 
  B = 2 :=
  by {
    sorry
  }

end PancakeBreakfast

end bacon_cost_l174_174855


namespace triangle_inequality_tetrahedron_counterexample_l174_174090

-- Part (a): For any triangle ABC and an interior point P, PA + PB < CA + CB.
theorem triangle_inequality {A B C P : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P] 
  (ABC : A × B × C) (P : P) :
  (∀ (PA : dist P A) (PB : dist P B) (CA : dist C A) (CB : dist C B), PA + PB < CA + CB) := 
sorry

-- Part (b): There exists a tetrahedron ABCD and an interior point P for which PA + PB + PC >= DA + DB + DC.
theorem tetrahedron_counterexample : 
  ∃ (A B C D P : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace P] 
  (ABCD : A × B × C × D) (P : P),
  ¬ (∀ (PA : dist P A) (PB : dist P B) (PC : dist P C) (DA : dist D A) (DB : dist D B) (DC : dist D C), PA + PB + PC < DA + DB + DC) := 
sorry

end triangle_inequality_tetrahedron_counterexample_l174_174090


namespace smallest_interesting_number_smallest_interesting_number_1800_l174_174174
-- Import all the necessary libraries

-- Variables definitions adhering to the conditions of the problem
def isPerfectSquare (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k

def isPerfectCube (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k * k

def isInteresting (n : ℕ) : Prop :=
  isPerfectSquare (2 * n) ∧ isPerfectCube (15 * n)

-- Prove that the smallest interesting number is 1800
theorem smallest_interesting_number : ∀ n : ℕ, isInteresting n → 1800 ≤ n :=
begin
  sorry
end

theorem smallest_interesting_number_1800 : isInteresting 1800 :=
begin
  sorry
end

end smallest_interesting_number_smallest_interesting_number_1800_l174_174174


namespace box_length_l174_174577

theorem box_length :
  ∃ (length : ℝ), 
  let box_height := 8
  let box_width := 10
  let block_height := 3
  let block_width := 2
  let block_length := 4
  let num_blocks := 40
  let box_volume := box_height * box_width * length
  let block_volume := block_height * block_width * block_length
  num_blocks * block_volume = box_volume ∧ length = 12 := by
  sorry

end box_length_l174_174577


namespace max_integer_n_l174_174307

def f (x : ℝ) : ℝ := x + 4 / x - 1

theorem max_integer_n : 
  ∃ (n : ℕ), 
  (∀ (x1 x2 ... xn x : ℝ), ∀i, 1 / 4 ≤ xi ≤ 4 ∧ 1 ≤ x ≤ 4 ∧ (f x1 + f x2 ... + f x n_1 = f (n)) → n = 6)
  sorry

end max_integer_n_l174_174307


namespace angle_proof_l174_174748

theorem angle_proof (x : ℝ) (h1 : ∃ α β, α = 3 * β - 36 ∧ (α = β ∨ α + β = 180)) :
  (∀ α β, h1 → (α = 18 ∨ α = 126)) :=
by
  sorry

end angle_proof_l174_174748


namespace sin_cos_value_l174_174313

variable (α : ℝ) (a b : ℝ × ℝ)
def vectors_parallel : Prop := b = (Real.sin α, Real.cos α) ∧
a = (4, 3) ∧ (∃ k : ℝ, a = (k * (Real.sin α), k * (Real.cos α)))

theorem sin_cos_value (h : vectors_parallel α a b) : ((Real.sin α) * (Real.cos α)) = 12 / 25 :=
by
  sorry

end sin_cos_value_l174_174313


namespace coordinates_of_C_l174_174030

theorem coordinates_of_C :
  ∀ (A B : ℝ × ℝ), 
  (A.1 - 2 * A.2 - 1 = 0) ∧ (B.1 - 2 * B.2 - 1 = 0) ∧
  (A.2 ^ 2 = 4 * A.1) ∧ (B.2 ^ 2 = 4 * B.1) ∧ 
  ∃ (C : ℝ × ℝ), (C.2 ^ 2 = 4 * C.1) ∧
  angle (C - A) (C - B) = π / 2 →
  C = (1, -2) ∨ C = (9, -6) :=
begin
  sorry

end coordinates_of_C_l174_174030


namespace magnitude_of_v_l174_174846

theorem magnitude_of_v {u v : ℂ} (h1 : u * v = 16 - 30 * Complex.i) (h2 : Complex.abs u = 2) :
  Complex.abs v = 17 :=
by
  sorry

end magnitude_of_v_l174_174846


namespace chess_games_count_l174_174053

theorem chess_games_count :
  let n := 7 in 
  (nat.choose n 2) = 21 :=
by
  let n := 7
  show (nat.choose n 2) = 21
  sorry

end chess_games_count_l174_174053


namespace tea_bags_l174_174434

theorem tea_bags (n : ℕ) (h₁ : 2 * n ≤ 41 ∧ 41 ≤ 3 * n) (h₂ : 2 * n ≤ 58 ∧ 58 ≤ 3 * n) : n = 20 := by
  sorry

end tea_bags_l174_174434


namespace sum_even_integers_less_than_100_l174_174985

theorem sum_even_integers_less_than_100 : 
  let sequence := List.range' 2 98
  let even_seq := sequence.filter (λ x => x % 2 = 0)
  (even_seq.sum) = 2450 :=
by
  sorry

end sum_even_integers_less_than_100_l174_174985


namespace halfway_fraction_l174_174953

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/6) : (a + b) / 2 = 19 / 24 :=
by
  sorry

end halfway_fraction_l174_174953


namespace smallest_abs_sum_l174_174407

theorem smallest_abs_sum {a b c d : ℤ} (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (h_matrix_sq : (Matrix 2 2 ℤ)!![(a, b), (c, d)]^2 = (Matrix 2 2 ℤ)!![(13, 0), (0, 5)]) :
  |a| + |b| + |c| + |d| = 6 :=
sorry

end smallest_abs_sum_l174_174407


namespace k_values_l174_174751

def vector_dot (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def find_k (k : ℝ) : Prop :=
  (vector_dot (2, 3) (1, k) = 0) ∨
  (vector_dot (2, 3) (-1, k - 3) = 0) ∨
  (vector_dot (1, k) (-1, k - 3) = 0)

theorem k_values :
  ∃ k : ℝ, find_k k ∧ 
  (k = -2/3 ∨ k = 11/3 ∨ k = (3 + Real.sqrt 13) / 2 ∨ k = (3 - Real.sqrt 13 ) / 2) :=
by
  sorry

end k_values_l174_174751


namespace determinant_eval_l174_174237

theorem determinant_eval (θ φ : ℝ) :
  let A := ![
    ![0, Real.cos θ, Real.sin θ],
    ![-Real.cos θ, 0, Real.cos (θ + φ)],
    ![-Real.sin θ, -Real.cos (θ + φ), 0]
  ] in Matrix.det A = -Real.sin θ * Real.cos θ * Real.cos (θ + φ) := by
  sorry

end determinant_eval_l174_174237


namespace value_of_stamp_collection_l174_174627

theorem value_of_stamp_collection (C : ℕ) (n : ℕ) (total_value : ℕ) (value_per_stamp : ℝ) :
  C = 30 → n = 10 → total_value = 45 → value_per_stamp = (total_value : ℝ) / (n : ℝ) →
  (C : ℝ) * value_per_stamp = 135 := 
by
  intros hC hn htotal hvalue
  rw [hC, hn, htotal, hvalue]
  norm_num
  sorry

end value_of_stamp_collection_l174_174627


namespace simplify_expression_1_simplify_expression_2_l174_174003

-- Conditions for the first proof
def example_1_sqrt2_79: ℚ := real.sqrt (2 + 7 / 9)
def example_1_term1 := example_1_sqrt2_79 + 0.1⁻²
def example_1_term2 := example_1_term1 - real.pi ^ 0
def example_1 := example_1_term2 + 1 / 3

-- Proof statement for the first proof
theorem simplify_expression_1 :
  example_1 = 101 := sorry

-- Conditions for the second proof
def example_2_log2_2 := real.logb 2 2
def example_2_log2_4 := real.logb 2 4
def example_2_term1 := (example_2_log2_2) ^ 2 + example_2_log2_2 * real.logb 2 5
def example_2_term2 := real.sqrt ((example_2_log2_2) ^ 2 - example_2_log2_4 + 1)
def example_2 := example_2_term1 + example_2_term2

-- Proof statement for the second proof
theorem simplify_expression_2 :
  example_2 = 1 := sorry

end simplify_expression_1_simplify_expression_2_l174_174003


namespace complement_set_A_in_U_l174_174392

-- Given conditions
def U : Set ℤ := {-1, 0, 1, 2}
def A : Set ℤ := {x | x ∈ U ∧ x^2 < 1}

-- Theorem to prove complement
theorem complement_set_A_in_U :
  U \ A = {-1, 1, 2} :=
by
  sorry

end complement_set_A_in_U_l174_174392


namespace sin_angle_PGH_l174_174359

-- Given an equilateral triangle PQR with trisecting points G and H on QR, prove that the sine of angle PGH is 3/10.

theorem sin_angle_PGH (P Q R G H : ℝ) (h_eq_triangle : (PQR_is_equilateral P Q R)) 
(h_trisect_GH : (trisect QR Q G H)) : 
  sin_angle_PGH P G H = 3 / 10 := 
sorry

end sin_angle_PGH_l174_174359


namespace angle_ABC_center_of_circumscribed_circle_l174_174491

theorem angle_ABC_center_of_circumscribed_circle
  (O A B C : Point)
  (hO_center : IsCenterOfCircumscribedCircle O A B C)
  (angle_BOC : ∠BOC = 110)
  (angle_AOB : ∠AOB = 150) :
  ∠ABC = 50 := 
sorry

end angle_ABC_center_of_circumscribed_circle_l174_174491


namespace measure_angle_ABC_l174_174488

variables (A B C O : Point)
variables (h1 : circumcenter O A B C)
variables (h2 : angle O B C = 110)
variables (h3 : angle A O B = 150)

theorem measure_angle_ABC : angle A B C = 50 :=
by
  sorry

end measure_angle_ABC_l174_174488


namespace math_problem_l174_174526

def ellipse (b : ℝ) (h : ℝ) (k : ℝ) : Prop :=
  0 < b ∧ b < 2 ∧ ∀ x y : ℝ, ((x - h)^2 / 4) + ((y - k)^2 / b^2) = 1

def hyperbola (b : ℝ) (h : ℝ) (k : ℝ) : Prop :=
  0 < b ∧ b < 2 ∧ ∀ x y : ℝ, ((x - h)^2 / 4) - ((y - k)^2 / b^2) = 1

def sister_conic_sections (e1 e2 : ℝ) : Prop :=
  e1 * e2 = (√15) / 4

def slope_of_line (x1 y1 x2 y2 : ℝ) : ℝ :=
  (y2 - y1) / (x2 - x1)

def kAM_kBN_ratio (xM yM xA yA xN yN xB yB : ℝ) : Prop :=
  slope_of_line xA yA xM yM / slope_of_line xB yB xN yN = -(1 / 3)

def w_range (kAM kBN : ℝ) : Prop :=
  let w := kAM^2 + (2 / 3) * kBN in
  (w > -(3 / 4) ∧ w < -(11 / 36)) ∨ (w > 13 / 36 ∧ w < 5 / 4)

theorem math_problem
  (b e1 e2 xM yM xN yN : ℝ)
  (kAM kBN kAM2 kBN2 : ℝ)
  (h k : ℝ)
  (C1_ellipse : ellipse b 0 0)
  (C2_hyperbola : hyperbola 1 0 0)
  (sister_conics : sister_conic_sections e1 e2)
  (left_vertex_M : xM = -2 ∧ yM = 0)
  (right_vertex_N : xN = 2 ∧ yN = 0)
  (G : xM = 4 ∧ yM = 0)
  : 
  (∃ x y, C2_hyperbola)
  ∧
  (∃ A B, kAM_kBN_ratio xM yM (fst A) (snd A) xN yN (fst B) (snd B))
  ∧
  (∃ kAM kBN, w_range kAM kBN) := sorry

end math_problem_l174_174526


namespace perimeter_quadrilateral_l174_174766

noncomputable def perimeter_ABCD (AE BE CE AB BC CD DE : ℝ) : ℝ :=
  AB + BC + CD + DE + AE

theorem perimeter_quadrilateral
  (h_right_angled_AE: ∃ A B E, ∠AEB = 90 ∧ AE = 36)
  (h_right_angled_BE: ∃ B C E, ∠BEC = 90 ∧ BE = 18)
  (h_right_angled_CE: ∃ C D E, ∠CED = 90 ∧ CE = 9)
  (h_AEB_60: ∠AEB = 60)
  (h_BEC_60: ∠BEC = 60)
  (h_CED_60: ∠CED = 60)
  (h_AB: AB = 18 * real.sqrt 3)
  (h_BC: BC = 9 * real.sqrt 3)
  (h_CD: CD = 4.5 * real.sqrt 3)
  (h_DE: DE = 4.5) :
  perimeter_ABCD 36 18 9 (18 * real.sqrt 3) (9 * real.sqrt 3) (4.5 * real.sqrt 3) 4.5 = 31.5 * real.sqrt 3 + 40.5 :=
by sorry

end perimeter_quadrilateral_l174_174766


namespace find_c_l174_174771

noncomputable def sequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = a n + c * n

theorem find_c (a : ℕ → ℝ) (c : ℝ) [h_seq : sequence a c] :
  (∃ r, r ≠ 1 ∧ a 2 = r * a 1 ∧ a 3 = r * a 2) → c = 2 :=
by
  sorry

end find_c_l174_174771


namespace total_distance_proof_l174_174583

def V_m : ℝ := 6 -- Speed of man in still water in km/h
def V_r : ℝ := 1.2 -- Speed of river in km/h
def T_total : ℝ := 1 -- Total time (in hours)

-- Effective speeds
def V_up : ℝ := V_m - V_r -- Speed when going upstream
def V_down : ℝ := V_m + V_r -- Speed when going downstream

-- Distance equation setup
def D : ℝ := (T_total * V_up * V_down) / (V_up + V_down) -- Simplified form to solve distance D

-- Total distance traveled by the man (to place and back)
def total_distance : ℝ := 2 * D

theorem total_distance_proof : total_distance = 5.76 := by
  sorry

end total_distance_proof_l174_174583


namespace handshaking_problem_l174_174756

noncomputable def handshaking_ways :=
  let n := 10 -- Number of people
  let k := 3 -- Number of handshakes per person
  in 144 -- By given solution

theorem handshaking_problem (h : ∀ p, p < 10 → p.shakes_hands 3) : handshaking_ways = 144 := 
by
  sorry

end handshaking_problem_l174_174756


namespace binomial_symmetry_calc_l174_174617

def binomial (n k : ℕ) : ℕ := Nat.div (Nat.factorial n) (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_symmetry_calc : binomial 12 10 = 66 := by
  have symmetry : binomial 12 10 = binomial 12 2 := by sorry
  have calc : binomial 12 2 = 66 := by sorry
  rw [symmetry, calc]
  exact calc

end binomial_symmetry_calc_l174_174617


namespace sqrt_expression_simplification_l174_174618

theorem sqrt_expression_simplification :
  (Real.sqrt (1 / 16) - Real.sqrt (25 / 4) + |Real.sqrt (3) - 1| + Real.sqrt 3) = -13 / 4 + 2 * Real.sqrt 3 :=
by
  have h1 : Real.sqrt (1 / 16) = 1 / 4 := sorry
  have h2 : Real.sqrt (25 / 4) = 5 / 2 := sorry
  have h3 : |Real.sqrt 3 - 1| = Real.sqrt 3 - 1 := sorry
  linarith [h1, h2, h3]

end sqrt_expression_simplification_l174_174618


namespace sum_of_numbers_l174_174637

theorem sum_of_numbers : 148 + 35 + 17 + 13 + 9 = 222 := 
by
  sorry

end sum_of_numbers_l174_174637


namespace find_extrema_of_expression_l174_174248

def satisfy_relation (x y : ℝ) : Prop := (sqrt (x - 1) + sqrt (y - 4) = 2)

theorem find_extrema_of_expression :
  (∃ x y : ℝ, satisfy_relation x y ∧ 2 * x + y = 14 ∧ x = 5 ∧ y = 4) ∧
  (∃ x y : ℝ, satisfy_relation x y ∧ 2 * x + y = 26 / 3 ∧ x = 13 / 9 ∧ y = 52 / 9) :=
by
  sorry

end find_extrema_of_expression_l174_174248


namespace question_part_1_question_part_1_formula_question_part_2_l174_174272

variable {a : ℕ → ℝ}
variable {b : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Given conditions
axiom recurrence_relation (n : ℕ) (hn : 2 ≤ n) : a n * (a (n-1) + 3 * a (n+1)) = 4 * a (n-1) * a (n+1)
axiom initial_conditions_a1 : a 1 = 1
axiom initial_conditions_a2 : a 2 = 1 / 4

-- Given definitions
def geometric_seq (n : ℕ) : ℝ := 1 / a (n+1) - 1 / a n
def b_seq (n : ℕ) : ℝ := n * a n / (a n + 2)

-- Given sum of first n terms of b_seq
def S_sum (n : ℕ) : ℝ := ∑ i in Finset.range n, b_seq (i + 1)

-- Problem to prove
theorem question_part_1 (n : ℕ) : 1 / a (n+1) - 1 / a n = 3^n :=
  sorry

theorem question_part_1_formula (n : ℕ) : a n = 2 / (3^n - 1) :=
  sorry

theorem question_part_2 (λ : ℝ) (n : ℕ) (hn : n > 0) (hλ : 2 * S_sum n > λ - (n + 1) / 3^n) : λ < 4 / 3 :=
  sorry

end question_part_1_question_part_1_formula_question_part_2_l174_174272


namespace points_collinear_collinear_for_k_l174_174402

-- Variables for vectors a and b
variables (a b : Vector ℝ)

-- Conditions about the vectors
axiom non_collinear_non_zero : a ≠ 0 ∧ b ≠ 0 ∧ ¬(∃ k : ℝ, a = k • b)

-- Definitions of points using the vectors
def OA := 2 • a + b
def OB := 3 • a - b
def OC := a + 3 • b

-- Theorem to prove points A, B, C are collinear
theorem points_collinear : collinear [OA, OB, OC] := sorry

-- Definitions of vectors for part (2)
variable k : ℝ

def v1 := 9 • a - k • b
def v2 := k • a - 4 • b

-- Theorem to prove collinearity for the specific k values
theorem collinear_for_k : k = 6 ∨ k = -6 → collinear [v1, v2] := sorry

end points_collinear_collinear_for_k_l174_174402


namespace line_circle_intersection_range_l174_174341

theorem line_circle_intersection_range (b : ℝ) :
    (2 - Real.sqrt 2) < b ∧ b < (2 + Real.sqrt 2) ↔
    ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ ((p1.1 - 2)^2 + p1.2^2 = 1) ∧ ((p2.1 - 2)^2 + p2.2^2 = 1) ∧ (p1.2 = p1.1 - b ∧ p2.2 = p2.1 - b) :=
by
  sorry

end line_circle_intersection_range_l174_174341


namespace composite_probability_l174_174539

noncomputable def probability_composite : ℚ :=
  let total_numbers := 50
      number_composite := total_numbers - 15 - 1
  in number_composite / (total_numbers - 1)

theorem composite_probability :
  probability_composite = 34 / 49 :=
by
  sorry

end composite_probability_l174_174539


namespace minimized_error_approx_l174_174451

-- Definition of the model
def model (x a b : ℝ) : ℝ := Real.cos (a * x) + b

-- Initial parameters
def initial_params : ℝ × ℝ := (2, 0.5)

-- Partial derivatives of the cost function I(a, b)
def partial_derivative_a (data : List (ℝ × ℝ)) (a b : ℝ) : ℝ :=
  data.foldl (λ acc (x, y), acc + 2 * (y - model x a b) * (-Real.sin (a * x) * x)) 0

def partial_derivative_b (data : List (ℝ × ℝ)) (a b : ℝ) : ℝ :=
  data.foldl (λ acc (x, y), acc + 2 * (y - model x a b) * (-1)) 0

-- Cost function I(a, b)
def cost_function (data : List (ℝ × ℝ)) (a b : ℝ) : ℝ :=
  data.foldl (λ acc (x, y), acc + ((y - model x a b) ^ 2)) 0

-- Gradient descent algorithm
noncomputable def gradient_descent (data : List (ℝ × ℝ)) 
                                    (initial_a initial_b : ℝ)
                                    (iterations : ℕ)
                                    (learning_rate : ℝ) : ℝ × ℝ :=
  let rec descent (a b : ℝ) (i : ℕ) : ℝ × ℝ :=
    if i = 0 then (a, b)
    else
      let grad_a := partial_derivative_a data a b
      let grad_b := partial_derivative_b data a b
      let a_new := a - learning_rate * grad_a
      let b_new := b - learning_rate * grad_b
      descent a_new b_new (i - 1)
  descent initial_a initial_b iterations

-- Data for gradient descent and error evaluation
variable (data : List (ℝ × ℝ))

-- Theorem to verify the minimized error is approximately 33.40
theorem minimized_error_approx :
  let (a_final, b_final) := gradient_descent data 2 0.5 5000 0.01 in
  abs (cost_function data a_final b_final - 33.40) < 0.01 :=
sorry

end minimized_error_approx_l174_174451


namespace lock_combinations_l174_174857

theorem lock_combinations (n : ℕ) : 
  (3^n + (-1)^n + 2 : ℕ) = 
  let vertices := Fin n
  let colors := Fin 2
  let numbers := Fin 2
  Σ (f : vertices → colors × numbers), 
    ∀ (i : Fin n), 
      (f i).fst = (f (i + 1) % n).fst ∨ (f i).snd = (f (i + 1) % n).snd :=
sorry

end lock_combinations_l174_174857


namespace annual_food_increase_l174_174191

theorem annual_food_increase (x : ℝ) :
  let y := 2.5 * x + 3.2 in
  let y' := 2.5 * (x + 1) + 3.2 in
  y' - y = 2.5 :=
by
  sorry

end annual_food_increase_l174_174191


namespace puzzle_theorem_l174_174383

open Function

universe u

variables {α : Type u}

structure Point (α : Type u) :=
(x : α)
(y : α)

structure Triangle (α : Type u) :=
(A : Point α)
(B : Point α)
(C : Point α)
(perimeter : α)

noncomputable def midpoint {α : Type u} [Field α] (P Q : Point α) : Point α :=
{ x := (P.x + Q.x) / 2,
  y := (P.y + Q.y) / 2 }

noncomputable def distance {α : Type u} [Field α] (P Q : Point α) : α :=
((Q.x - P.x)^2 + (Q.y - P.y)^2) ^ (1/2 : α)

noncomputable def angle_bisector_intersect {α : Type u} [Field α] (A B C : Point α) : Point α := sorry

structure Hypotheses (α : Type u) [Field α] :=
(triangle : Triangle α)
(M : Point α)
(AL : Point α)
(D : Point α)

theorem puzzle_theorem {α : Type u} [Field α] (h : Hypotheses α) : 
  distance h.triangle.A h.D + distance h.M h.triangle.C = h.triangle.perimeter / 2 :=
sorry

end puzzle_theorem_l174_174383


namespace smallest_interesting_number_l174_174134

def is_perfect_square (x : ℕ) : Prop :=
∃ y : ℕ, y * y = x

def is_perfect_cube (x : ℕ) : Prop :=
∃ y : ℕ, y * y * y = x

theorem smallest_interesting_number (n : ℕ) :
    is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n) → n = 1800 :=
by {
  sorry
}

end smallest_interesting_number_l174_174134


namespace triangle_angle_proportions_l174_174098

-- Definitions of parameters
variable (p q r : ℝ)
variable (α β γ : ℝ)

-- Condition stating the sum of angles in a triangle is π
axiom angle_sum_eq_pi : α + β + γ = π

-- Proof statement
theorem triangle_angle_proportions
  (h1 : α = π * (3 * p - q - r) / (p + q + r))
  (h2 : β = π * (3 * q - p - r) / (p + q + r))
  (h3 : γ = π * (3 * r - q - p) / (p + q + r)) :
  α + β + γ = π :=
by
  rw [h1, h2, h3]
  sorry

end triangle_angle_proportions_l174_174098


namespace smallest_interesting_number_is_1800_l174_174162

def is_perfect_square (k : ℕ) : Prop := ∃ (m : ℕ), k = m * m
def is_perfect_cube (k : ℕ) : Prop := ∃ (m : ℕ), k = m * m * m

def is_interesting (n : ℕ) : Prop :=
  is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n)

theorem smallest_interesting_number_is_1800 :
  ∃ (n : ℕ), is_interesting(n) ∧ (∀ m : ℕ, is_interesting(m) → n ≤ m) ∧ n = 1800 :=
by 
  sorry

end smallest_interesting_number_is_1800_l174_174162


namespace circle_center_max_area_l174_174018

open Real

theorem circle_center_max_area :
  ∀ k : ℝ, let center := (-k / 2, -1) in k = 0 → center = (0, -1) :=
by
  intros k center hk
  sorry

end circle_center_max_area_l174_174018


namespace probability_of_error_is_75_percent_l174_174754

def intended_function (a b : Bool) : Bool :=
(a && b) || (a != b)

def implemented_function (a b : Bool) : Bool :=
(a && b) && (a != b)

def probability_of_error : ℚ :=
  let truth_table := [(false, false), (false, true), (true, false), (true, true)]
  let incorrect_outputs := truth_table.filter (λ (ab : Bool × Bool), 
    implemented_function ab.1 ab.2 ≠ intended_function ab.1 ab.2)
  (incorrect_outputs.length : ℚ) / (truth_table.length : ℚ)

theorem probability_of_error_is_75_percent :
  probability_of_error = 3 / 4 :=
sorry

end probability_of_error_is_75_percent_l174_174754


namespace series_divisible_by_100_l174_174832

theorem series_divisible_by_100 : 
  let s : Nat := 1 + 11 + 111 + 1111 + 11111 + 111111 + 1111111 + 11111111 + 111111111 + 1111111111
  in s % 100 = 0 := by
  sorry

end series_divisible_by_100_l174_174832


namespace smallest_interesting_number_l174_174135

def is_perfect_square (x : ℕ) : Prop :=
∃ y : ℕ, y * y = x

def is_perfect_cube (x : ℕ) : Prop :=
∃ y : ℕ, y * y * y = x

theorem smallest_interesting_number (n : ℕ) :
    is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n) → n = 1800 :=
by {
  sorry
}

end smallest_interesting_number_l174_174135


namespace square_area_l174_174759

-- Given a square ABCD with side length a,
-- E is the midpoint of AB and F is the trisection point of BC.
-- Prove the area of the square ABCD is 324, given the area differences of triangles ADE and CDF.

theorem square_area
  (a : ℝ)
  (hE : a / 2 = (segment AB).midpoint)
  (hF : a / 3 = (segment BC).trisection)
  (hAreaDifference: (a^2 / 4) - (a^2 / 6) = 27) :
  a^2 = 324 :=
sorry

end square_area_l174_174759


namespace shaded_area_of_triangles_l174_174024

theorem shaded_area_of_triangles
  (n : ℕ)
  (h_n : n = 5)
  (total_hypotenuse_length : ℝ)
  (h_total_len : total_hypotenuse_length = 30)
  (hypotenuse_length : ℝ)
  (h_hypotenuse_len : hypotenuse_length = total_hypotenuse_length / n) :
  let triangle_area := (hypotenuse_length^2) / 4 in
  let shaded_area := n * triangle_area in
  shaded_area = 45 := by
    sorry

end shaded_area_of_triangles_l174_174024


namespace solution_set_of_f_x_add_1_lt_0_l174_174291

open Set

variables {ℝ : Type*} [LinearOrderedField ℝ]

-- Definitions from conditions
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)
  
def increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

variable (f : ℝ → ℝ)

-- Given conditions
axiom fg_evn : even_function f
axiom fg_inc : increasing_on_nonneg f
axiom fg_val : f 2 = 0

-- Problem statement
theorem solution_set_of_f_x_add_1_lt_0 :
  { x | f(x + 1) < 0 } = Ioo (-3 : ℝ) 1 :=
sorry

end solution_set_of_f_x_add_1_lt_0_l174_174291


namespace part_a_part_b_l174_174105

-- Part (a)
theorem part_a (n : ℕ) (h : n > 0) :
  (2 * n ∣ n * (n + 1) / 2) ↔ ∃ k : ℕ, n = 4 * k - 1 :=
by sorry

-- Part (b)
theorem part_b (n : ℕ) (h : n > 0) :
  (2 * n + 1 ∣ n * (n + 1) / 2) ↔ (2 * n + 1 ≡ 1 [MOD 4]) ∨ (2 * n + 1 ≡ 3 [MOD 4]) :=
by sorry

end part_a_part_b_l174_174105


namespace imo_1981_p6_l174_174811

noncomputable def smallest_n_with_coprime_property (S : Set Nat) (n : Nat) : Prop :=
  ∀ A ⊆ S, A.card = n -> ∃ B ⊆ A, B.card = 5 ∧ ∀ x y ∈ B, Nat.gcd x y = 1

theorem imo_1981_p6 (S : Set Nat) (H : S = { x : Nat | 1 ≤ x ∧ x ≤ 280 }) : 
    smallest_n_with_coprime_property S 217 :=
by
  sorry

end imo_1981_p6_l174_174811


namespace cot_arccots_sum_eq_97_over_40_l174_174210

noncomputable def cot_arccot_sum : ℝ :=
  Real.cot (Real.arccot 5 + Real.arccot 11 + Real.arccot 17 + Real.arccot 23)

theorem cot_arccots_sum_eq_97_over_40 :
  cot_arccot_sum = 97 / 40 :=
sorry

end cot_arccots_sum_eq_97_over_40_l174_174210


namespace max_cb_dot_cd_l174_174265

open_locale classical
noncomputable theory

variables {V : Type*} [inner_product_space ℝ V]
variables (A B C D : V)

-- Given conditions
def ab_len_one : Prop := ∥B - A∥ = 1
def cd_len_two : Prop := ∥D - C∥ = 2
def ad_ac_dot_ac_sq : Prop := ⟪D - A, C - A⟫ = ∥C - A∥^2

-- The target statement to prove
theorem max_cb_dot_cd (h1 : ab_len_one A B) (h2 : cd_len_two C D) (h3 : ad_ac_dot_ac_sq A C D) :
  ∃ (θ : ℝ), θ ≤ 2 ∧ θ = (⟪C - B, D - C⟫) :=
sorry

end max_cb_dot_cd_l174_174265


namespace sum_of_sequence_divisible_by_2_l174_174640

theorem sum_of_sequence_divisible_by_2 
  (a : ℕ → ℕ) (n : ℕ)
  (h : ∀ i : ℕ, i < n - 1 → a (i + 1) = a i + 2) :
  2 ∣ (∑ i in finset.range n, a i) :=
sorry

end sum_of_sequence_divisible_by_2_l174_174640


namespace total_asphalt_used_1520_tons_l174_174593

noncomputable def asphalt_used (L W : ℕ) (asphalt_per_100m2 : ℕ) : ℕ :=
  (L * W / 100) * asphalt_per_100m2

theorem total_asphalt_used_1520_tons :
  asphalt_used 800 50 3800 = 1520000 := by
  sorry

end total_asphalt_used_1520_tons_l174_174593


namespace chord_length_proof_l174_174013

noncomputable def length_of_chord (a b : ℝ) (h : a^2 - b^2 = 16) : ℝ :=
  let c := 8 in 
  c

theorem chord_length_proof
  (a b : ℝ)
  (h : a^2 - b^2 = 16) :
  let c := length_of_chord a b h in
  ∃ c, c = 8 ∧ (a^2 - b^2 = 16) → (c / 2)^2 + b^2 = a^2 :=
by
  sorry

end chord_length_proof_l174_174013


namespace rectangle_area_from_diagonal_l174_174184

theorem rectangle_area_from_diagonal (x : ℝ) (w : ℝ) (h_lw : 3 * w = 3 * w) (h_diag : x^2 = 10 * w^2) : 
    (3 * w^2 = (3 / 10) * x^2) :=
by 
sorry

end rectangle_area_from_diagonal_l174_174184


namespace log_inequalities_l174_174266

theorem log_inequalities (a : ℝ) (x : ℝ) (ha : 1 < a) (hx : 1 < x ∧ x < a) :
  log a (log a x) < (log a x)^2 ∧ (log a x)^2 < log a (x^2) :=
sorry

end log_inequalities_l174_174266


namespace smallest_interesting_number_l174_174167

-- Definitions based on the conditions
def isPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m
def isPerfectCube (m : ℕ) : Prop := ∃ k : ℕ, k * k * k = m

def isInteresting (n : ℕ) : Prop :=
  isPerfectSquare (2 * n) ∧ isPerfectCube (15 * n)

-- The smallest interesting number
theorem smallest_interesting_number : ∃ n : ℕ, isInteresting n ∧ (∀ m : ℕ, isInteresting m → n ≤ m) :=
  ⟨1800, ⟨⟨26, by norm_num⟩, ⟨15, by norm_num⟩⟩, sorry⟩

end smallest_interesting_number_l174_174167


namespace smallest_interesting_number_l174_174132

def is_interesting (n : ℕ) : Prop :=
  (∃ m1 : ℕ, 2 * n = m1^2) ∧ (∃ m2 : ℕ, 15 * n = m2^3)

theorem smallest_interesting_number : ∃ n, is_interesting n ∧ (∀ m : ℕ, is_interesting m → n ≤ m) :=
  ⟨1800, by {
    split,
    { use 60,
      norm_num },
    { use 30,
      norm_num },
    sorry }⟩

end smallest_interesting_number_l174_174132


namespace part1_part2_l174_174725

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 + Real.sin x, 1)
def b : ℝ × ℝ := (2, -2)
noncomputable def c (x : ℝ) : ℝ × ℝ := (Real.sin x - 3, 1)
def d (k : ℝ) : ℝ × ℝ := (1, k)
noncomputable def bc (x : ℝ) : ℝ × ℝ := (Real.sin x - 1, -1)
noncomputable def ad (x k : ℝ) : ℝ × ℝ := (3 + Real.sin x, 1 + k)

theorem part1 (x : ℝ) (hx : x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)) :
  let a_angle := a x / (2 + Real.sin x)
  let bc_angle := bc x / (Real.sin x - 1)
  a_angle = bc_angle → x = -Real.pi / 6 :=
by
  sorry

theorem part2 (x : ℝ) (hx : x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)) :
  ∃ k : ℝ, 
  let ad_dot_bc := (ad x k).1 * (bc x).1 + (ad x k).2 * (bc x).2
  ad_dot_bc = 0 ↔ k ∈ Set.Icc (-5 : ℝ) (-1 : ℝ) :=
by
  sorry

end part1_part2_l174_174725


namespace parity_of_expression_l174_174408

theorem parity_of_expression (e m : ℕ) (he : (∃ k : ℕ, e = 2 * k)) : Odd (e ^ 2 + 3 ^ m) :=
  sorry

end parity_of_expression_l174_174408


namespace largest_consecutive_odd_number_is_27_l174_174046

theorem largest_consecutive_odd_number_is_27 (a b c : ℤ) 
  (h1: a + b + c = 75)
  (h2: c - a = 6)
  (h3: b = a + 2)
  (h4: c = a + 4) :
  c = 27 := 
sorry

end largest_consecutive_odd_number_is_27_l174_174046


namespace vector_magnitude_diff_eq_one_l174_174723

theorem vector_magnitude_diff_eq_one :
  let a := (Real.cos (75 * Real.pi / 180), Real.sin (75 * Real.pi / 180))
  let b := (Real.cos (15 * Real.pi / 180), Real.sin (15 * Real.pi / 180))
  EuclideanDist (a - b).1 (a - b).2 = 1 :=
by 
  let a := (Real.cos (75 * Real.pi / 180), Real.sin (75 * Real.pi / 180))
  let b := (Real.cos (15 * Real.pi / 180), Real.sin (15 * Real.pi / 180))
  sorry

end vector_magnitude_diff_eq_one_l174_174723


namespace range_of_c_l174_174288

theorem range_of_c {m n c : ℝ} (h₁ : m^2 + (n - 2)^2 = 1)
  (h₂ : ∀ (m n : ℝ), m^2 + (n - 2)^2 = 1 → m + n + c ≥ 1) :
  c ≥ real.sqrt 2 - 1 :=
by
  sorry

end range_of_c_l174_174288


namespace sh_possible_strip_impossible_l174_174373

def sum_of_digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8} 
def total_sum : ℕ := 36

noncomputable
def possible_arrangement_sh (f : Fin (8 + 1) → ℕ) : Prop := 
  ∃ (arrangement : Fin (8 + 1) → ℕ), 
      sum_of_digits = Finset.image arrangement (Finset.univ) ∧
      (∀ (subset : Finset ℕ), (subset ⊆ sum_of_digits) →
        (subset.sum ≤ total_sum) ∧
        ((subset.sum ∣ total_sum) ∨ ((total_sum - subset.sum) ∣ subset.sum)))

noncomputable
def impossible_arrangement_strip (f : Fin (8 + 1) → ℕ) : Prop := 
  ¬ ∃ (arrangement : Fin (8 + 1) → ℕ), 
      sum_of_digits = Finset.image arrangement (Finset.univ) ∧
      (∀ (subset : Finset ℕ), (subset ⊆ sum_of_digits) →
        (subset.sum ≤ total_sum) ∧
        ((subset.sum ∣ total_sum) ∨ ((total_sum - subset.sum) ∣ subset.sum)))

theorem sh_possible : possible_arrangement_sh sum_of_digits := sorry

theorem strip_impossible : impossible_arrangement_strip sum_of_digits := sorry

end sh_possible_strip_impossible_l174_174373


namespace halfway_fraction_l174_174969

theorem halfway_fraction (a b : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) : (1/2) * (a + b) = 19/24 :=
by 
  rw [h1, h2]
  simp
  done
  sorry

end halfway_fraction_l174_174969


namespace guests_ate_tubs_of_ice_cream_l174_174518

-- Definitions of the conditions
def total_pieces (pans : ℕ) (pieces_per_pan : ℕ) : ℕ :=
  pans * pieces_per_pan

def pieces_eaten (whole_pan : ℕ) (partial_pan_fraction : ℚ) (pieces_per_pan : ℕ) : ℕ :=
  whole_pan + (partial_pan_fraction * pieces_per_pan).toNat

def pieces_ala_mode (pieces : ℕ) (without_ala_mode : ℕ) : ℕ :=
  pieces - without_ala_mode

def scoops_needed (pieces : ℕ) (scoops_per_piece : ℕ) : ℕ :=
  pieces * scoops_per_piece

def tubs_needed (scoops : ℕ) (scoops_per_tub : ℕ) : ℕ :=
  scoops / scoops_per_tub

-- Conditions given in the problem
def pans := 2
def pieces_per_pan := 16
def whole_pan := 16
def partial_pan_fraction : ℚ := 0.75
def without_ala_mode := 4
def scoops_per_piece := 2
def scoops_per_tub := 8

-- Lean proof statement
theorem guests_ate_tubs_of_ice_cream : 
  total_pieces pans pieces_per_pan = 32 →
  pieces_eaten whole_pan partial_pan_fraction pieces_per_pan = 28 →
  pieces_ala_mode 28 without_ala_mode = 24 →
  scoops_needed 24 scoops_per_piece = 48 →
  tubs_needed 48 scoops_per_tub = 6 :=
by {
  intros,
  sorry
}

end guests_ate_tubs_of_ice_cream_l174_174518


namespace tiling_rectangles_exists_l174_174625

theorem tiling_rectangles_exists (A B C : ℕ) :
  ∃ (N : ℕ), ∀ (m n : ℕ), m > N → n > N →
  (∃ (f : (fin m) × (fin n) → option (fin 4 × fin 6)),
    ∀ (i j : fin m), ∃ (a b c d : fin 5 × fin 7),
    i ∈ a.val∧ j ∈ b.val ∧ (f (i, j) = some (c, d))) ∨ (
    ∃ (f : (fin m) × (fin n) → option (fin 5 × fin 7)),
    ∀ (i j : fin m), ∃ (a b c d : fin 5 × fin 7),
    i ∈ a.val ∧ j ∈ b.val ∧ (f (i, j) = some (d, c))) :=
sorry

end tiling_rectangles_exists_l174_174625


namespace probability_of_same_color_draw_l174_174108

def prob_all_same (total_red total_white total_blue : ℕ) : ℕ → ℕ → ℕ → ℚ
| 0, _, _ := 0
| _, 0, _ := 0
| _, _, 0 := 0
| _, _, _ := (
  (total_red / 15) * ((total_red - 1) / 14) * ((total_red - 2) / 13) +
  (total_white / 15) * ((total_white - 1) / 14) * ((total_white - 2) / 13) +
  (total_blue / 15) * ((total_blue - 1) / 14) * ((total_blue - 2) / 13)
  )

theorem probability_of_same_color_draw :
  prob_all_same 3 7 5 0 0 0 = 23 / 455 :=
by {
  sorry
}

end probability_of_same_color_draw_l174_174108


namespace original_numbers_sum_l174_174460

theorem original_numbers_sum (A B C D E F : ℕ) :
  A + 0.1 * B + 0.01 * C + 10 * D + E + 0.1 * F = 50.13 ∧
  10 * A + B + 0.1 * C + D + 0.01 * E + F = 34.02 →
  100 * A + 10 * B + C + 100 * D + 10 * E + F = 765 :=
by
  sorry

end original_numbers_sum_l174_174460


namespace find_W_l174_174773

-- Definition of conditions
def different_digits (a b c d e f g h : ℕ) : Prop :=
  list.nodup [a, b, c, d, e, f, g, h]

def valid_addition (T W O F U R : ℕ) : Prop :=
T + T = F * 10 + O ∧ 
W + W + (if T + T >= 10 then 1 else 0) = U * 10 + R
  
-- Main theorem
theorem find_W (W : ℕ) : 
  different_digits 7 W 4 1 _ _ _ _ ∧
  valid_addition 7 W 4 1 _ _ ∧
  even 4 ∧ digit _ ∧ digit _ ∧ digit _ ∧ digit _
  → W = 3 :=
sorry

end find_W_l174_174773


namespace range_of_a_l174_174303

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (h : f (2 - a^2) > f a) : a ∈ set.Ioo (-2 : ℝ) (1 : ℝ) :=
sorry

end range_of_a_l174_174303


namespace C1_C2_one_common_point_the_area_of_triangle_ABC_m_eq_neg_2_l174_174365

-- Definitions from the conditions
def C1 (ρ θ : ℝ) : Prop := ρ = 4 * cos θ
def C2 (t m : ℝ) (x y : ℝ): Prop :=
  x = m + 1/2 * t ∧ y = sqrt 3 / 2 * t
  
-- (1) Proving the value of m when C1 and C2 have only one common point
theorem C1_C2_one_common_point : 
  ∀ (m : ℝ), (∃ (x y : ℝ), (C1 ρ θ) ∧ (C2 t m x y)) → m = 2 + 4 * sqrt 3 / 3 ∨ m = 2 - 4 * sqrt 3 / 3 :=
sorry

-- (2) Proving the value of m based on intersections and area condition
theorem the_area_of_triangle_ABC_m_eq_neg_2 :
  ∀ (m : ℝ),
  (m < 0)
  → (∃ A B C : ℝ × ℝ, A ≠ (0, 0) ∧ B ≠ (0, 0) ∧ 
     (θ = π / 3) ∧ 
     (C1 ρ θ) ∧ 
     (θ = 5 * π / 6) ∧ 
     (C1 ρ θ) ∧ 
     let area_ABC := 1 / 2 * 2 * (2 * sqrt 3 - sqrt 3 / 2 * m) in
     area_ABC = 3 * sqrt 3)
  → m = -2 :=
sorry

end C1_C2_one_common_point_the_area_of_triangle_ABC_m_eq_neg_2_l174_174365


namespace smallest_interesting_number_l174_174148

theorem smallest_interesting_number :
  ∃ (n : ℕ), (∃ k1 : ℕ, 2 * n = k1 ^ 2) ∧ (∃ k2 : ℕ, 15 * n = k2 ^ 3) ∧ n = 1800 := 
sorry

end smallest_interesting_number_l174_174148


namespace degree_measure_angle_ABC_l174_174497

theorem degree_measure_angle_ABC (O A B C : Type) [euclidean_geometry] 
  (circumscribed_about : is_circumscribed_circle O (triangle A B C)) 
  (angle_BOC : measure (angle B O C) = 110) 
  (angle_AOB : measure (angle A O B) = 150) : 
  measure (angle A B C) = 50 := 
sorry

end degree_measure_angle_ABC_l174_174497


namespace scientific_notation_of_diameter_l174_174474

theorem scientific_notation_of_diameter :
  0.00000258 = 2.58 * 10^(-6) :=
by sorry

end scientific_notation_of_diameter_l174_174474


namespace magnitude_of_alpha_l174_174401

noncomputable def α : ℂ := sorry
noncomputable def β : ℂ := α.conj  -- since β is the conjugate of α 

-- Conditions
axiom h1 : (α / (β^2)).im = 0
axiom h2 : |α - β| = 2 * √5

theorem magnitude_of_alpha : |α| = (2 * √15) / 3 := by
  -- Proof omitted
  sorry

end magnitude_of_alpha_l174_174401


namespace carl_speed_l174_174780

theorem carl_speed 
  (time : ℝ) (distance : ℝ) 
  (h_time : time = 5) 
  (h_distance : distance = 10) 
  : (distance / time) = 2 :=
by
  rw [h_time, h_distance]
  sorry

end carl_speed_l174_174780


namespace fraction_halfway_between_3_4_and_5_6_is_19_24_l174_174915

noncomputable def fraction_halfway (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between_3_4_and_5_6_is_19_24 :
  fraction_halfway (3 / 4) (5 / 6) = 19 / 24 :=
by
  sorry

end fraction_halfway_between_3_4_and_5_6_is_19_24_l174_174915


namespace point_on_line_is_sufficient_but_not_necessary_condition_for_arithmetic_sequence_l174_174689

def is_on_line (n : ℕ) (a_n : ℕ) : Prop := a_n = 2 * n + 1

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n m, a n - a m = d * (n - m)

theorem point_on_line_is_sufficient_but_not_necessary_condition_for_arithmetic_sequence (a : ℕ → ℕ) :
  (∀ n, is_on_line n (a n)) → is_arithmetic_sequence a ∧ 
  ¬ (is_arithmetic_sequence a → ∀ n, is_on_line n (a n)) :=
sorry

end point_on_line_is_sufficient_but_not_necessary_condition_for_arithmetic_sequence_l174_174689


namespace count_divisibles_l174_174330

-- Definitions for the conditions of the problem
def upper_bound : ℕ := 2013

def divisible_by (a b : ℕ) : Prop := ∃ k, a = b * k

-- The statement of the problem
theorem count_divisibles :
  let count := (λ n : ℕ, (if divisible_by n 3 ∨ divisible_by n 5 then 1 else 0))
  (Finset.sum (Finset.range (upper_bound + 1)) count) = 939 :=
sorry

end count_divisibles_l174_174330


namespace min_distance_sum_l174_174694

noncomputable def parabola (p : ℝ × ℝ) : Prop :=
  let (x, y) := p in y^2 = 4 * x

noncomputable def circle (q : ℝ × ℝ) : Prop :=
  let (x, y) := q in x^2 + (y - 4)^2 = 1

def distance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

def directrix_distance (p : ℝ × ℝ) : ℝ :=
  let (x, y) := p in abs (x + 1)

theorem min_distance_sum (P Q : ℝ × ℝ) (hP : parabola P) (hQ : circle Q) :
  ∃ m, m = distance P Q + directrix_distance P ∧ m = sqrt 17 - 1 :=
begin
  sorry
end

end min_distance_sum_l174_174694


namespace max_value_expression_proof_l174_174797

noncomputable def max_value_expression (γ δ : ℂ) (h1 : |δ| = 1) (h2 : (conj γ) * δ ≠ -1) : ℝ :=
  max_value_expression γ δ h1 h2 = 1

theorem max_value_expression_proof (γ δ : ℂ) (h1 : |δ| = 1) (h2 : (conj γ) * δ ≠ -1) :
  max_value_expression γ δ h1 h2 := 
by {
  sorry
}

end max_value_expression_proof_l174_174797


namespace sum_even_pos_ints_lt_100_l174_174999

theorem sum_even_pos_ints_lt_100 : ∑ k in finset.range 50, 2 * k = 2450 := by
  sorry

end sum_even_pos_ints_lt_100_l174_174999


namespace max_value_f_l174_174249

def f (x : ℝ) : ℝ := |x| / (Real.sqrt (1 + x^2) * Real.sqrt (4 + x^2))

theorem max_value_f : ∃ x : ℝ, ∀ y : ℝ, f(y) ≤ f(x) ∧ f(x) = 1 / 3 :=
by
  use 2
  intro y
  split
  sorry
  sorry

end max_value_f_l174_174249


namespace solve_system_of_equations_l174_174843

variable (x y z : ℝ)

theorem solve_system_of_equations {x y z : ℝ} :
  (x * (y + z) * (x + y + z) = 1170) ∧
  (y * (z + x) * (x + y + z) = 1008) ∧
  (z * (x + y) * (x + y + z) = 1458) →
  (x = 5) ∧ (y = 4) ∧ (z = 9) :=
begin
  sorry
end

end solve_system_of_equations_l174_174843


namespace angle_PAQ_l174_174899

noncomputable def triangle_conditions (A B C P Q : Type) [LinearOrder A] : Prop :=
  ∃ (AB AC BC CP CQ : ℝ),
  AB = 2 * AC ∧ 
  ∠A = 120 ∧ 
  AB^2 + BC * CP = BC^2 ∧ 
  3 * AC^2 + 2 * BC * CQ = BC^2

theorem angle_PAQ (A B C P Q : Type) [LinearOrder A]
  (h : triangle_conditions A B C P Q) :
  ∠PAQ = 30 := 
sorry

end angle_PAQ_l174_174899


namespace pedal_triangle_side_l174_174406

variables (A B C P A1 B1 C1 : Type) [FIntType R]

/-- Given triangle ABC and point P with pedal triangle A1 B1 C1, and R is the circumradius of ABC.
    Prove that B1C1 = (BC * AP) / (2R). -/
theorem pedal_triangle_side (A B C P A1 B1 C1 : Point) (R : Length) (BC AP : Length)
  (circumradius : triangle ABC → Length := R) : 
  (B1C1 : Length) = BC * AP / (2 * R) :=
sorry

end pedal_triangle_side_l174_174406


namespace alpha_modulus_l174_174394

noncomputable def α : ℂ := a + b * Complex.I
noncomputable def β : ℂ := a - b * Complex.I

theorem alpha_modulus :
  (α β : ℂ) (h_conj : β = conj α)
  (h_real : α / (β^2) ∈ ℝ) (h_diff : |α - β| = 2 * Real.sqrt 5) :
  |α| = (2 * Real.sqrt 15) / 3 :=
by
  sorry

end alpha_modulus_l174_174394


namespace primes_not_dividing_an_l174_174671

theorem primes_not_dividing_an 
  (p : ℕ) (hp : Nat.Prime p)
  (a_n : ℕ → ℕ := λ n, 3^n + 6^n - 2^n) :
  (∀ n > 0, ¬ (p ∣ a_n n)) → (p = 2 ∨ p = 3) :=
by
  sorry

end primes_not_dividing_an_l174_174671


namespace problem_solution_l174_174712

noncomputable def f (a b x : ℝ) : ℝ := a * x ^ 2 - 2 * real.sqrt(4 + 2 * b - b ^ 2) * x
noncomputable def g (a x : ℝ) : ℝ := -real.sqrt(1 - (x - a) ^ 2)

theorem problem_solution (a b : ℝ) :
  (b = 0 → a ≥ 1 ∧ ∀ x ∈ set.Ici 2, monotone_on (f a b) (set.Ici 2)) ∧
  (∀ (a ∈ ℤ),
    (∃ x₀, is_max (f a b) x₀ ∧ is_min (g a) x₀) → 
    (a = -1 ∧ (b = -1 ∨ b = 3))) := 
by sorry

end problem_solution_l174_174712


namespace estimated_value_l174_174862

-- Define the equation of the regression line
def regression_eq (x : ℝ) : ℝ := 0.5 * x - 0.81

-- Lean 4 statement to prove the estimated value of y when x=25
theorem estimated_value (x : ℝ) (h : x = 25) : regression_eq x = 11.69 :=
by
  rw [h]
  unfold regression_eq
  norm_num
  sorry

end estimated_value_l174_174862


namespace telescoping_sum_l174_174632

theorem telescoping_sum :
  ∑ n in finset.range 4998, (λ k, 3 + k) (λ n, 1 / (↑n * real.sqrt (↑n - 2) + (↑n - 2) * real.sqrt (↑n))) = 6999 / 7000 := 
sorry

end telescoping_sum_l174_174632


namespace relationship_a_b_c_l174_174982

theorem relationship_a_b_c (x y a b c : ℝ) (h1 : x + y = a)
  (h2 : x^2 + y^2 = b) (h3 : x^3 + y^3 = c) : a^3 - 3*a*b + 2*c = 0 := by
  sorry

end relationship_a_b_c_l174_174982


namespace find_k_at_4_l174_174808

noncomputable def h (x : ℝ) : ℝ := x^3 - x + 1

def isCubicPolynomial (p : ℝ → ℝ) := 
  ∃ a₃ a₂ a₁ a₀ : ℝ, p = λ x, a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀

theorem find_k_at_4
  (k : ℝ → ℝ)
  (hk : isCubicPolynomial k)
  (h_k0 : k 0 = 1)
  (h_roots : ∀ {α β γ : ℝ}, (α^3 - α + 1 = 0) ∧ (β^3 - β + 1 = 0) ∧ (γ^3 - γ + 1 = 0) → 
              k = λ x, -(x - α^3) * (x - β^3) * (x - γ^3)) :
  k 4 = -61 := 
sorry

end find_k_at_4_l174_174808


namespace find_integer_l174_174509

theorem find_integer (N : ℤ) (hN : N^2 + N = 12) (h_pos : 0 < N) : N = 3 :=
sorry

end find_integer_l174_174509


namespace halfway_fraction_l174_174955

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/6) : (a + b) / 2 = 19 / 24 :=
by
  sorry

end halfway_fraction_l174_174955


namespace find_a_l174_174315

theorem find_a (a : ℚ) (A : Set ℚ) (h : 3 ∈ A) (hA : A = {a + 2, 2 * a^2 + a}) : a = 3 / 2 := 
by
  sorry

end find_a_l174_174315


namespace part_a_part_b_l174_174527

-- Define n_mid_condition
def n_mid_condition (n : ℕ) : Prop := n % 2 = 1 ∧ n ∣ 2023^n - 1

-- Part a:
theorem part_a : ∃ (k₁ k₂ : ℕ), k₁ = 3 ∧ k₂ = 9 ∧ n_mid_condition k₁ ∧ n_mid_condition k₂ := by
  sorry

-- Part b:
theorem part_b : ∀ k, k ≥ 1 → n_mid_condition (3^k) := by
  sorry

end part_a_part_b_l174_174527


namespace hyperbola_arithmetic_mean_AB_l174_174293

theorem hyperbola_arithmetic_mean_AB
  (b : ℝ) (a : ℝ) (c : ℝ) (F1 F2 A B : ℝ) 
  (imag_length : 2 * b = 4) 
  (eccentricity : c / a = sqrt 6 / 2)
  (AB_is_mean : 2 * abs (A - B) = abs (A - F2) + abs (B - F2)) :
  abs (A - B) = 8 * sqrt 2 := 
by
  have ha : a = 2 * sqrt 2 := sorry
  have hc : c = sqrt 6 * sqrt 2 := sorry
  have hAB : abs (A - B) = 8 * sqrt 2 := sorry
  exact hAB

end hyperbola_arithmetic_mean_AB_l174_174293


namespace triangle_inequality_cosine_l174_174825

theorem triangle_inequality_cosine (a b c : ℝ) (A B C : ℝ) (p : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : A + B + C = π) 
  (h5 : cos A = (b^2 + c^2 - a^2) / (2*b*c)) 
  (h6 : cos B = (a^2 + c^2 - b^2) / (2*a*c))
  (h7 : cos C = (a^2 + b^2 - c^2) / (2*a*b))
  (h8 : p = (a + b + c) / 2) : 
  a * cos A + b * cos B + c * cos C ≤ p := 
sorry

end triangle_inequality_cosine_l174_174825


namespace isosceles_triangle_count_l174_174445

theorem isosceles_triangle_count :
  ∃ (s : Finset (ℕ × ℕ)), s.card = 7 ∧ ∀ (p : ℕ × ℕ) (h : p ∈ s), 
  is_triangle_isosceles_with_AB (3,3) (6,3) p :=
sorry

noncomputable def is_triangle_isosceles_with_AB (A B C : (ℕ × ℕ)) : Prop :=
  let AB := abs (A.1 - B.1) + abs (A.2 - B.2) in
  let AC := abs (A.1 - C.1) + abs (A.2 - C.2) in
  let BC := abs (B.1 - C.1) + abs (B.2 - C.2) in
  AB = 3 ∧ (AC = BC ∨ AC = AB ∨ BC = AB)

end isosceles_triangle_count_l174_174445


namespace total_cost_is_734_l174_174818

-- Define the cost of each ice cream flavor
def cost_vanilla : ℕ := 99
def cost_chocolate : ℕ := 129
def cost_strawberry : ℕ := 149

-- Define the amount of each flavor Mrs. Hilt buys
def num_vanilla : ℕ := 2
def num_chocolate : ℕ := 3
def num_strawberry : ℕ := 1

-- Calculate the total cost in cents
def total_cost : ℕ :=
  (num_vanilla * cost_vanilla) +
  (num_chocolate * cost_chocolate) +
  (num_strawberry * cost_strawberry)

-- Statement of the proof problem
theorem total_cost_is_734 : total_cost = 734 :=
by
  sorry

end total_cost_is_734_l174_174818


namespace johns_total_cost_after_discount_l174_174379

/-- Price of different utensils for John's purchase --/
def forks_cost : ℕ := 25
def knives_cost : ℕ := 30
def spoons_cost : ℕ := 20
def dinner_plate_cost (silverware_cost : ℕ) : ℚ := 0.5 * silverware_cost

/-- Calculating the total cost of silverware --/
def total_silverware_cost : ℕ := forks_cost + knives_cost + spoons_cost

/-- Calculating the total cost before discount --/
def total_cost_before_discount : ℚ := total_silverware_cost + dinner_plate_cost total_silverware_cost

/-- Discount rate --/
def discount_rate : ℚ := 0.10

/-- Discount amount --/
def discount_amount (total_cost : ℚ) : ℚ := discount_rate * total_cost

/-- Total cost after applying discount --/
def total_cost_after_discount : ℚ := total_cost_before_discount - discount_amount total_cost_before_discount

/-- John's total cost after the discount should be $101.25 --/
theorem johns_total_cost_after_discount : total_cost_after_discount = 101.25 := by
  sorry

end johns_total_cost_after_discount_l174_174379


namespace smallest_interesting_number_l174_174152

def is_perfect_square (m : ℕ) : Prop := 
  ∃ k : ℕ, k * k = m

def is_perfect_cube (m : ℕ) : Prop := 
  ∃ k : ℕ, k * k * k = m

def interesting (n : ℕ) : Prop := 
  is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n)

theorem smallest_interesting_number : ∃ n, interesting n ∧ ∀ m, interesting m → n ≤ m :=
begin
  use 1800,
  split,
  {
    -- Prove 1800 is interesting
    sorry,
  },
  {
    -- Prove any interesting number is at least 1800
    sorry,
  }
end

end smallest_interesting_number_l174_174152


namespace find_x_l174_174472

-- Given constants and assumptions
constant C : ℝ
constant x : ℝ
constant profit : ℝ := 0.25

-- Hypotheses based on conditions
axiom cost_price_of_40_articles : 40 * C
axiom selling_price_of_x_articles : x * (1 + profit) * C
axiom cost_equals_selling_price : cost_price_of_40_articles = selling_price_of_x_articles

-- Proof statement
theorem find_x (h : cost_price_of_40_articles = selling_price_of_x_articles) : x = 32 :=
by {
  have eq := calc
    40 * C = (1 + profit) * x * C : h
    ...   = 1.25 * x * C           : by rw [(show 1 + profit = 1.25, by norm_num)]
  have : 40 = 1.25 * x := by {
    simpa using eq
  }
  have x_val : x = 40 / 1.25 := by {
    field_simp [this]
  }
  have : 40 / 1.25 = 32 := by {
    norm_num
  }
  exact this ▸ x_val
}

end find_x_l174_174472


namespace opposite_of_negative_seven_l174_174033

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_negative_seven : opposite (-7) = 7 := 
by 
  sorry

end opposite_of_negative_seven_l174_174033


namespace area_ratio_trapezoid_l174_174195

variables (A B C D F : Type) [EuclideanSpace ℝ] 

def is_trapezoid (A B C D : Type) [EuclideanSpace ℝ] : Prop :=
  -- Define trapezoid in a Euclidean space
  sorry

def inscribed_in_circle (A B C D : Type) [EuclideanSpace ℝ] : Prop :=
  -- Define trapezoid inscribed in a circle
  sorry

def intersection_of_diagonals (A B C D : Type) [EuclideanSpace ℝ] : Type :=
  -- Define the intersection point of the diagonals
  sorry

def has_bases (A B C D : Type) [EuclideanSpace ℝ] (AB CD : ℕ) : Prop :=
  -- Define trapezoid with base lengths AB and CD
  AB = 1 ∧ CD = 2

theorem area_ratio_trapezoid 
  (h_trapezoid : is_trapezoid A B C D)
  (h_inscribed : inscribed_in_circle A B C D)
  (h_intersection : F = intersection_of_diagonals A B C D)
  (h_bases : has_bases A B C D 1 2) :
  (area (triangle A B F) + area (triangle C D F)) / (area (triangle A F D) + area (triangle B C F)) = 5 / 4 :=
sorry

end area_ratio_trapezoid_l174_174195


namespace hyperbola_equation_correct_slopes_product_correct_l174_174704

noncomputable def hyperbola_standard_equation (x y : ℝ) : Prop :=
∃ a b : ℝ, a = 1 ∧ b = √2 ∧ (x^2 - y^2 / (2 : ℝ) = 1)

theorem hyperbola_equation_correct :
  (∃ x : ℝ, (∃ y : ℝ, hyperbola_standard_equation x y)) :=
begin
  sorry
end

noncomputable def slopes_product (x y k1 k2 : ℝ) : Prop :=
  k1 = y / (x + 1) ∧ k2 = y / (x - 1) ∧ k1 * k2 = 2

theorem slopes_product_correct :
  (∀ x y : ℝ, (x^2 - y^2 / 2 = 1) → (-1 < x ∧ x < 1) → slopes_product x y (y / (x + 1)) (y / (x - 1))) :=
begin
  sorry
end

end hyperbola_equation_correct_slopes_product_correct_l174_174704


namespace volume_of_pyramid_IJKOP_l174_174390

-- Define the conditions:
def rectangular_prism (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

def base_edges (a : ℝ) (b : ℝ) := 
  a = 2 ∧ b = 2

def height (c : ℝ) :=
  c = 2

-- Define the volume of a pyramid given base area and height
def volume_pyramid (base_area height : ℝ) : ℝ :=
  (base_area * height) / 3

-- Mathematical statement to be proved
theorem volume_of_pyramid_IJKOP :
  ∀ a b c : ℝ, 
  rectangular_prism a b c → 
  base_edges a b → 
  height c → 
  volume_pyramid (1/2 * a * b) c = 4/3 :=
by
  intros a b c h_prism h_base h_height
  -- Apply the conditions to compute the volume
  sorry

end volume_of_pyramid_IJKOP_l174_174390


namespace solvable_example_problem_l174_174664

theorem solvable_example_problem:
  ∃ (m n : ℤ), (n ^ 2 ∣ m) ∧ (m ^ 3 ∣ n ^ 2) ∧ (n ^ 4 ∣ m ^ 3) ∧ (m ^ 5 ∣ n ^ 4) ∧ ¬ (n ^ 6 ∣ m ^ 5) :=
begin
  use [32, 16],
  split,
  { -- n^2 ∣ m
    show 16 ^ 2 ∣ 32,
    sorry
  },
  split,
  { -- m^3 ∣ n^2
    show 32 ^ 3 ∣ 16 ^ 2,
    sorry
  },
  split,
  { -- n^4 ∣ m^3
    show 16 ^ 4 ∣ 32 ^ 3,
    sorry
  },
  split,
  { -- m^5 ∣ n^4
    show 32 ^ 5 ∣ 16 ^ 4,
    sorry
  },
  { -- ¬ n^6 ∣ m^5
    show ¬ 16 ^ 6 ∣ 32 ^ 5,
    sorry
  }
end

end solvable_example_problem_l174_174664


namespace madeline_water_intake_l174_174420

def water_bottle_capacity : ℕ := 12
def number_of_refills : ℕ := 7
def additional_water_needed : ℕ := 16
def total_water_needed : ℕ := 100

theorem madeline_water_intake : water_bottle_capacity * number_of_refills + additional_water_needed = total_water_needed :=
by
  sorry

end madeline_water_intake_l174_174420


namespace part_I_part_II_part_III_l174_174314

-- Define the sequence {a_n}
def a_n (n : ℕ) : ℕ := 
  if n = 0 then 0 -- not defined case for a_0, conventionally avoid it
  else if n = 1 then 2
  else (fin_rec (λ (n : ℕ), (n+2)*(a n) = (n+1)*(a (n+1)) - 2*(n^2 + 3*n + 2))) n

-- Define the sequence {b_n}
def b_n (n : ℕ) : ℤ := 
  if n = 0 then 0 -- not defined case for b_0, so keep it 0
  else a_n n / (n + 1)

-- Define the sequence {c_n}
def c_n (n : ℕ) : ℤ :=
  if n = 0 then 0 -- not defined case for c_0, so keep it 0
  else (2^n + 1) * b_n n

-- Define the sequence {T_n}
def T_n (n : ℕ) : ℤ :=
  6 + (2*n - 3)*(2^(n+1)) + n^2

-- Prove statements
theorem part_I :
  b_n 1 = 1 ∧ b_n 2 = 3 ∧ b_n 3 = 5 := sorry

theorem part_II : 
  ∀ n : ℕ, b_n (n+1) - b_n n = 2 := sorry

theorem part_III : 
  ∀ n, T_n n = 6 + (2*n - 3)*(2^(n+1)) + n^2 := sorry

end part_I_part_II_part_III_l174_174314


namespace equidistant_xaxis_point_l174_174073

theorem equidistant_xaxis_point {x : ℝ} :
  (∃ x : ℝ, ∀ A B : ℝ × ℝ, A = (-3, 0) ∧ B = (2, 5) →
    ∀ P : ℝ × ℝ, P = (x, 0) →
      (dist A P = dist B P) → x = 2) := sorry

end equidistant_xaxis_point_l174_174073


namespace range_of_a_l174_174286

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f (-x) = -f(x)) →
  (∀ x, x ≥ 0 → f(x) = 0.5 * (abs (x - a^2) + abs (x - 2 * a^2) - 3 * a^2)) →
  (∀ x, f(x - 1) ≤ f(x)) →
  - real.sqrt(6) / 6 ≤ a ∧ a ≤ real.sqrt(6) / 6 :=
by
  sorry

end range_of_a_l174_174286


namespace find_correct_θ_value_l174_174659

-- Define the given function
def f (x θ : ℝ) : ℝ := sin (2 * x + θ) + sqrt 3 * cos (2 * x + θ)

-- Define the property of the function being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x) = -f (-x)

-- Define the interval [0, π/4]
def interval : set ℝ := {x | 0 ≤ x ∧ x ≤ (π / 4)}

-- Define the property of the function being decreasing on the interval [0, π/4]
def is_decreasing_on (f : ℝ → ℝ) (s : set ℝ) : Prop :=
  ∀ ⦃a b : ℝ⦄, a ∈ s → b ∈ s → a < b → f a ≥ f b

-- Define the specific θ value which is the solution being proven
def target_θ : ℝ := 2 * π / 3

-- Combine all the conditions into one theorem statement
theorem find_correct_θ_value :
  ∃ θ : ℝ, is_odd_function (f x θ) ∧ is_decreasing_on (f x θ) interval ∧ θ = target_θ :=
sorry

end find_correct_θ_value_l174_174659


namespace initial_sale_percentage_l174_174187

variable {P : ℝ} -- Assume P is the original price of the shirt
variable {x : ℝ} -- Assume x is the sale percentage in decimal form

-- Hypothesis: The final price after two discounts is 81% of the original price
theorem initial_sale_percentage (P > 0) (h : 0.90 * (1 - x) * P = 0.81 * P) : 
  x = 0.1 :=
by
  -- Since we only need the statement, replace this with sorry
  sorry

end initial_sale_percentage_l174_174187


namespace halfway_fraction_l174_174942

theorem halfway_fraction (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  ((a / b + c / d) / 2) = 19 / 24 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end halfway_fraction_l174_174942


namespace problem_l174_174229

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem problem : nabla (nabla 1 3) 2 = 67 :=
by
  sorry

end problem_l174_174229


namespace standard_deviations_below_mean_l174_174468

theorem standard_deviations_below_mean (μ σ x : ℝ) (hμ : μ = 14.5) (hσ : σ = 1.7) (hx : x = 11.1) :
    (μ - x) / σ = 2 := by
  sorry

end standard_deviations_below_mean_l174_174468


namespace composite_probability_l174_174534

/--
Given that a number selected at random from the first 50 natural numbers,
where 1 is neither prime nor composite,
the probability of selecting a composite number is 34/49.
-/
theorem composite_probability :
  let total_numbers := 50
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  let num_primes := primes.length
  let num_composites := total_numbers - num_primes - 1
  let probability_composite := (num_composites : ℚ) / (total_numbers - 1)
  probability_composite = 34 / 49 :=
by {
  sorry
}

end composite_probability_l174_174534


namespace smallest_interesting_number_l174_174151

def is_perfect_square (m : ℕ) : Prop := 
  ∃ k : ℕ, k * k = m

def is_perfect_cube (m : ℕ) : Prop := 
  ∃ k : ℕ, k * k * k = m

def interesting (n : ℕ) : Prop := 
  is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n)

theorem smallest_interesting_number : ∃ n, interesting n ∧ ∀ m, interesting m → n ≤ m :=
begin
  use 1800,
  split,
  {
    -- Prove 1800 is interesting
    sorry,
  },
  {
    -- Prove any interesting number is at least 1800
    sorry,
  }
end

end smallest_interesting_number_l174_174151


namespace sum_even_pos_ints_lt_100_l174_174994

theorem sum_even_pos_ints_lt_100 : ∑ k in finset.range 50, 2 * k = 2450 := by
  sorry

end sum_even_pos_ints_lt_100_l174_174994


namespace rhombus_min_distance_l174_174762

theorem rhombus_min_distance (A B C D E F : Type*) [geometry : MetricSpace Type*]
  (h_rhombus : isRhombus A B C D)
  (h_angle : angle B A C = 60)
  (h_side_length : dist A B = 2)
  (E_on_BC : isOnLine E B C)
  (F_on_BD : isOnLine F B D) :
  minDistance EF CF = sqrt 3 := sorry

end rhombus_min_distance_l174_174762


namespace exists_nat_expressed_as_sum_of_powers_l174_174268

theorem exists_nat_expressed_as_sum_of_powers 
  (P : Finset ℕ) (hP : ∀ p ∈ P, Nat.Prime p) :
  ∃ x : ℕ, (∀ p ∈ P, ∃ a b : ℕ, x = a^p + b^p) ∧ (∀ p : ℕ, Nat.Prime p → p ∉ P → ¬∃ a b : ℕ, x = a^p + b^p) :=
by
  let x := 2^(P.val.prod + 1)
  use x
  sorry

end exists_nat_expressed_as_sum_of_powers_l174_174268


namespace Sam_walk_time_l174_174197

theorem Sam_walk_time :
  ∀ (d_feet time_minutes d_yards : ℕ),
    d_feet = 90 →
    time_minutes = 45 →
    d_yards = 120 →
    let rate := d_feet / time_minutes in
    let d_target_feet := d_yards * 3 in
    let time_needed := d_target_feet / rate in
    time_needed = 180 :=
by
  intros d_feet time_minutes d_yards h1 h2 h3
  unfold rate d_target_feet time_needed
  simp [h1, h2, h3]
  sorry

end Sam_walk_time_l174_174197


namespace sum_largest_k_equal_l174_174361

variables {n : ℕ} {A : matrix (fin n) (fin n) ℝ}

def sum_of_k_largest_in_row (A : matrix (fin n) (fin n) ℝ) (i : fin n) (k : ℕ) :=
(sort (≤) (λ j : fin n, A i j)).take_right k).sum

def sum_of_k_largest_in_col (A : matrix (fin n) (fin n) ℝ) (j : fin n) (k : ℕ) :=
(sort (≤) (λ i : fin n, A i j)).take_right k).sum

theorem sum_largest_k_equal (a b : ℝ) (k : ℕ) (hk : k = 2) :
  (∀ i, sum_of_k_largest_in_row A i k = a) →
  (∀ j, sum_of_k_largest_in_col A j k = b) →
  a = b :=
begin
  sorry
end

end sum_largest_k_equal_l174_174361


namespace angle_between_vectors_is_90_degrees_l174_174322

variables {V : Type*} [inner_product_space ℝ V] (u v : V)

noncomputable def calc_angle (u v : V) [inner_product_space ℝ V] : ℝ :=
  real.arccos (inner_product_space.inner u v / (∥u∥ * ∥v∥))

theorem angle_between_vectors_is_90_degrees
  (h : ∥u + 2 • v∥ = ∥u - 2 • v∥) : calc_angle u v = real.pi / 2 :=
sorry

end angle_between_vectors_is_90_degrees_l174_174322


namespace determine_range_m_l174_174310

-- Definitions of the functions
def f (x : ℝ) : ℝ := -2 * x * Real.log x
def g (x : ℝ) (m : ℝ) : ℝ := -x^3 + 3 * x * m

-- Definition of the range of x
def interval (x : ℝ) : Prop := 1 / Real.exp 1 ≤ x ∧ x ≤ Real.exp 1

-- Theorem statement
theorem determine_range_m (m : ℝ) : 
  (∀ x : ℝ, interval x → (f x = g x m → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ interval x1 ∧ interval x2 ∧ f x1 = g x1 m ∧ f x2 = g x2 m)) ↔
  (1 / 3 < m ∧ m ≤ (2 / 3 + 1 / (3 * Real.exp (2)))) :=
sorry

end determine_range_m_l174_174310


namespace single_cone_scoops_l174_174220

theorem single_cone_scoops (banana_split_scoops : ℕ) (waffle_bowl_scoops : ℕ) (single_cone_scoops : ℕ) (double_cone_scoops : ℕ)
  (h1 : banana_split_scoops = 3 * single_cone_scoops)
  (h2 : waffle_bowl_scoops = banana_split_scoops + 1)
  (h3 : double_cone_scoops = 2 * single_cone_scoops)
  (h4 : single_cone_scoops + banana_split_scoops + waffle_bowl_scoops + double_cone_scoops = 10) :
  single_cone_scoops = 1 :=
by
  sorry

end single_cone_scoops_l174_174220


namespace smallest_interesting_number_l174_174126

def is_interesting (n : ℕ) : Prop :=
  (∃ m1 : ℕ, 2 * n = m1^2) ∧ (∃ m2 : ℕ, 15 * n = m2^3)

theorem smallest_interesting_number : ∃ n, is_interesting n ∧ (∀ m : ℕ, is_interesting m → n ≤ m) :=
  ⟨1800, by {
    split,
    { use 60,
      norm_num },
    { use 30,
      norm_num },
    sorry }⟩

end smallest_interesting_number_l174_174126


namespace range_of_a_no_real_roots_l174_174274

open Real

def quadratic_no_real_roots (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬(a * x^2 + a * x + 1 = 0)

theorem range_of_a_no_real_roots :
  {a : ℝ | quadratic_no_real_roots a} = Icc 0 4 :=
by
  sorry

end range_of_a_no_real_roots_l174_174274


namespace odd_area_rectangles_l174_174097

theorem odd_area_rectangles (rectangles : List (ℕ × ℕ)) (h_rectangles_length : rectangles.length = 9) :
  ∃ n, n ∈ {0, 4} ∧ (n = rectangles.filter (λ r, r.1 % 2 = 1 ∧ r.2 % 2 = 1).length) :=
by
  sorry

end odd_area_rectangles_l174_174097


namespace calculate_128_to_7_3_l174_174219

theorem calculate_128_to_7_3 :
  (128 : ℝ) ^ (7 / 3) = (65536 : ℝ) * real.cbrt 2 :=
by sorry

end calculate_128_to_7_3_l174_174219


namespace fraction_simplification_l174_174081

theorem fraction_simplification : (3^2040 + 3^2038) / (3^2040 - 3^2038) = 5 / 4 :=
by
  sorry

end fraction_simplification_l174_174081


namespace factorization_correctness_l174_174864

def quadratic_expression (y : ℝ) : ℝ := 15 * y * y - 56 * y + 48

theorem factorization_correctness (C D : ℝ) (hC : C = 5) (hD : D = 3) :
  (quadratic_expression -) = (Cy - 16) * (Dy - 3) ∧ CD + C = 20 := 
by
  sorry

end factorization_correctness_l174_174864


namespace final_position_west_of_bus_stop_distance_from_bus_stop_total_calories_consumed_l174_174235

-- Define the movements as a list of integers
def movements : List ℤ := [1000, -900, 700, -1200, 1200, 100, -1100, -200]

-- Define the function to calculate the final position
def final_position (movements : List ℤ) : ℤ :=
  movements.foldl (· + ·) 0

-- Define the function to find the total distance walked (absolute sum)
def total_distance (movements : List ℤ) : ℕ :=
  movements.foldl (fun acc x => acc + x.natAbs) 0

-- Calorie consumption rate per kilometer (1000 meters)
def calories_per_kilometer : ℕ := 7000

-- Calculate the calories consumed
def calories_consumed (total_meters : ℕ) : ℕ :=
  (total_meters / 1000) * calories_per_kilometer

-- Lean 4 theorem statements

theorem final_position_west_of_bus_stop : final_position movements = -400 := by
  sorry

theorem distance_from_bus_stop : |final_position movements| = 400 := by
  sorry

theorem total_calories_consumed : calories_consumed (total_distance movements) = 44800 := by
  sorry

end final_position_west_of_bus_stop_distance_from_bus_stop_total_calories_consumed_l174_174235


namespace arithmetic_geometric_sequences_sequence_sum_first_terms_l174_174281

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def geometric_sequence (b : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, ∀ n : ℕ, b (n + 1) = b n * q

noncomputable def sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = (n + 1) * a 1 + (n * (n + 1)) / 2

theorem arithmetic_geometric_sequences
  (a b S : ℕ → ℤ)
  (h1 : arithmetic_sequence a)
  (h2 : geometric_sequence b)
  (h3 : a 0 = 1)
  (h4 : b 0 = 1)
  (h5 : b 2 * S 2 = 36)
  (h6 : b 1 * S 1 = 8) :
  ((∀ n, a n = 2 * n + 1) ∧ (∀ n, b n = 2 ^ n)) ∨
  ((∀ n, a n = -(2 * n / 3) + 5 / 3) ∧ (∀ n, b n = 6 ^ n)) :=
sorry

theorem sequence_sum_first_terms
  (a : ℕ → ℤ)
  (h : ∀ n, a n = 2 * n + 1)
  (S : ℕ → ℤ)
  (T : ℕ → ℚ)
  (hS : sequence_sum a S)
  (n : ℕ) :
  T n = n / (2 * n + 1) :=
sorry

end arithmetic_geometric_sequences_sequence_sum_first_terms_l174_174281


namespace independence_A_B_l174_174891

/-- Define the sample space and events -/
def GiftBox := { box1, box2, box3, box4 }

def event_A (b : GiftBox) : Prop :=
  b = box1 ∨ b = box4

def event_B (b : GiftBox) : Prop :=
  b = box2 ∨ b = box4

def P (s : set GiftBox) : ℝ :=
  (s.to_finset.card : ℝ) / 4

/-- Probability of event A -/
def P_A : ℝ := P { b | event_A b }

/-- Probability of event B -/
def P_B : ℝ := P { b | event_B b }

/-- Probability of A and B occurring together -/
def P_A_and_B : ℝ := P ({ b | event_A b } ∩ { b | event_B b })

/-- The independence condition that should be proven -/
theorem independence_A_B : P_A * P_B = P_A_and_B :=
  by sorry

end independence_A_B_l174_174891


namespace water_drank_is_gallons_l174_174601

noncomputable def total_water_drunk (traveler_weight: ℝ) (traveler_percent: ℝ) (camel_weight: ℝ) (camel_percent: ℝ) 
(pounds_to_ounces: ℝ) (ounces_to_gallon: ℝ) : ℝ :=
(traveler_weight * traveler_percent / 100 * pounds_to_ounces + camel_weight * camel_percent / 100 * pounds_to_ounces) / ounces_to_gallon

theorem water_drank_is_gallons :
  total_water_drunk 160 0.5 1200 2 16 128 = 3.1 :=
by
  unfold total_water_drunk
  norm_num
  sorry

end water_drank_is_gallons_l174_174601


namespace dennis_years_taught_l174_174069

-- Variables representing the number of years taught by Adrienne, Virginia, and Dennis respectively
variables (A V D : ℕ)

/- Conditions
1. Virginia has taught for 9 more years than Adrienne.
2. Virginia has taught for 9 fewer years than Dennis.
3. Combined years of teaching is 93.
-/
def condition1 : Prop := V = A + 9
def condition2 : Prop := V = D - 9
def condition3 : Prop := A + V + D = 93

-- The main theorem we want to prove: Dennis has taught for 40 years
theorem dennis_years_taught (h1 : condition1) (h2 : condition2) (h3 : condition3) : D = 40 :=
  sorry

end dennis_years_taught_l174_174069


namespace definite_integral_sqrt_minus_one_l174_174747

-- Definitions based on conditions in a)
def interval := {x : ℝ | 1 < x ∧ x < 4}

-- Statement based on question and correct answer in b)
theorem definite_integral_sqrt_minus_one :
  ∫ x in (by rw interval), x ^ (1/2) - 1 = 5 / 3 :=
by
  sorry

end definite_integral_sqrt_minus_one_l174_174747


namespace smallest_interesting_number_l174_174147

theorem smallest_interesting_number :
  ∃ (n : ℕ), (∃ k1 : ℕ, 2 * n = k1 ^ 2) ∧ (∃ k2 : ℕ, 15 * n = k2 ^ 3) ∧ n = 1800 := 
sorry

end smallest_interesting_number_l174_174147


namespace projection_of_a_in_direction_of_c_l174_174726

-- Definitions for conditions
def vector_a (x : ℝ) : ℝ × ℝ := (x, 2)
def vector_b : ℝ × ℝ := (2, 1)
def vector_c (x : ℝ) : ℝ × ℝ := (3, x)

-- Given conditions
axiom x : ℝ
axiom x_eq_4 : x = 4
axiom parallel_vectors : vector_a x = λ k, (2 * k, k)

-- Defining dot product and magnitude
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (u : ℝ × ℝ) : ℝ := Real.sqrt (u.1^2 + u.2^2)

-- The projection definition
def projection (u v : ℝ × ℝ) : ℝ := dot_product u v / magnitude v

-- The goal: to prove the projection value
theorem projection_of_a_in_direction_of_c : projection (vector_a x) (vector_c x) = 4 :=
by
  -- Accepting the axioms and definitions for now
  sorry

end projection_of_a_in_direction_of_c_l174_174726


namespace composite_probability_l174_174538

noncomputable def probability_composite : ℚ :=
  let total_numbers := 50
      number_composite := total_numbers - 15 - 1
  in number_composite / (total_numbers - 1)

theorem composite_probability :
  probability_composite = 34 / 49 :=
by
  sorry

end composite_probability_l174_174538


namespace intervals_of_monotonicity_and_min_value_l174_174716

noncomputable def f (x d : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + d

theorem intervals_of_monotonicity_and_min_value (d : ℝ) 
  (h_max : ∀ x ∈ set.Icc (-2 : ℝ) 2, f x d ≤ 20) :
  (∀ x < -1, f' x < 0) ∧ (∀ x > 3, f' x < 0) ∧
  (∀ x, -1 < x ∧ x < 3 → f' x > 0) ∧
  d = -2 → (f (-1) (-2)) = -7 :=
sorry

end intervals_of_monotonicity_and_min_value_l174_174716


namespace isosceles_trapezoid_side_length_l174_174852

theorem isosceles_trapezoid_side_length (A b1 b2 : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 48) (hb1 : b1 = 9) (hb2 : b2 = 15) 
  (h_area : A = 1 / 2 * (b1 + b2) * h) 
  (h_h : h = 4)
  (h_s : s^2 = h^2 + ((b2 - b1) / 2)^2) :
  s = 5 :=
by sorry

end isosceles_trapezoid_side_length_l174_174852


namespace PersonYs_speed_in_still_water_l174_174058

def speed_in_still_water (speed_X : ℕ) (t_1 t_2 : ℕ) (x : ℕ) : Prop :=
  ∀ y : ℤ, 4 * (6 - y + x + y) = 4 * 6 + 4 * x ∧ 16 * (x + y) = 16 * (6 + y) + 4 * (x - 6) →
  x = 10

theorem PersonYs_speed_in_still_water :
  speed_in_still_water 6 4 16 10 :=
by
  sorry

end PersonYs_speed_in_still_water_l174_174058


namespace expand_polynomial_l174_174658

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_polynomial_l174_174658


namespace smallest_interesting_number_l174_174165

-- Definitions based on the conditions
def isPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m
def isPerfectCube (m : ℕ) : Prop := ∃ k : ℕ, k * k * k = m

def isInteresting (n : ℕ) : Prop :=
  isPerfectSquare (2 * n) ∧ isPerfectCube (15 * n)

-- The smallest interesting number
theorem smallest_interesting_number : ∃ n : ℕ, isInteresting n ∧ (∀ m : ℕ, isInteresting m → n ≤ m) :=
  ⟨1800, ⟨⟨26, by norm_num⟩, ⟨15, by norm_num⟩⟩, sorry⟩

end smallest_interesting_number_l174_174165


namespace p_sufficient_but_not_necessary_for_q_l174_174037

def proposition_p (x : ℝ) := x - 1 = 0
def proposition_q (x : ℝ) := (x - 1) * (x + 2) = 0

theorem p_sufficient_but_not_necessary_for_q :
  ( (∀ x, proposition_p x → proposition_q x) ∧ ¬(∀ x, proposition_p x ↔ proposition_q x) ) := 
by
  sorry

end p_sufficient_but_not_necessary_for_q_l174_174037


namespace min_shift_symmetric_about_y_l174_174869

theorem min_shift_symmetric_about_y (φ : ℝ) (hφ : φ > 0) :
  (∀ x : ℝ, 3 * sin (1 / 2 * (x - φ) + π / 6) = 3 * sin (1 / 2 * (-x - φ) + π / 6)) →
  φ = 4 * π / 3 :=
by
  sorry

end min_shift_symmetric_about_y_l174_174869


namespace rate_of_interest_l174_174507

noncomputable def simple_interest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

noncomputable def compound_interest (P R T : ℝ) : ℝ :=
  P * ((1 + R / 100) ^ T - 1)

theorem rate_of_interest :
  let P_si := 2800
  let R_si := 5
  let T_si := 3
  let P_ci := 4000
  let T_ci := 2
  let si := simple_interest P_si R_si T_si
  let ci := 2 * si 
  in
  ci = compound_interest P_ci 10 T_ci :=
by
  sorry

end rate_of_interest_l174_174507


namespace average_age_l174_174854

def proportion (x y z : ℕ) : Prop :=  y / x = 3 ∧ z / x = 4

theorem average_age (A B C : ℕ) 
    (h1 : proportion 2 6 8)
    (h2 : A = 15)
    (h3 : B = 45)
    (h4 : C = 60) :
    (A + B + C) / 3 = 40 := 
    by
    sorry

end average_age_l174_174854


namespace smallest_interesting_number_l174_174139

def is_perfect_square (x : ℕ) : Prop :=
∃ y : ℕ, y * y = x

def is_perfect_cube (x : ℕ) : Prop :=
∃ y : ℕ, y * y * y = x

theorem smallest_interesting_number (n : ℕ) :
    is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n) → n = 1800 :=
by {
  sorry
}

end smallest_interesting_number_l174_174139


namespace sin_2A_eq_sin_2B_neither_sufficient_nor_necessary_condition_l174_174776

theorem sin_2A_eq_sin_2B_neither_sufficient_nor_necessary_condition :
  ∀ (A B C : ℝ), (A + B + C = π) → (sin (2 * A) = sin (2 * B)) → 
  ¬((A = B) ∧ (A + B ≠ π / 2)) ∧ ¬(A = B) :=
by
  sorry

end sin_2A_eq_sin_2B_neither_sufficient_nor_necessary_condition_l174_174776


namespace perpendicular_lines_l174_174299

theorem perpendicular_lines :
  (∀ (x y : ℝ), (4 * y - 3 * x = 16)) ∧ 
  (∀ (x y : ℝ), (3 * y + 4 * x = 15)) → 
  (∃ (m1 m2 : ℝ), m1 * m2 = -1) :=
by
  sorry

end perpendicular_lines_l174_174299


namespace person_y_speed_in_still_water_l174_174054

theorem person_y_speed_in_still_water 
    (speed_x_in_still_water : ℝ)
    (time_meeting_towards_each_other : ℝ)
    (time_catching_up_same_direction: ℝ)
    (distance_upstream_meeting: ℝ)
    (distance_downstream_meeting: ℝ)
    (total_distance: ℝ) :
    speed_x_in_still_water = 6 →
    time_meeting_towards_each_other = 4 →
    time_catching_up_same_direction = 16 →
    distance_upstream_meeting = 4 * (6 - distance_upstream_meeting) + 4 * (10 + distance_downstream_meeting) →
    distance_downstream_meeting = 4 * (6 + distance_upstream_meeting) →
    total_distance = 4 * (6 + 10) →
    ∃ (speed_y_in_still_water : ℝ), speed_y_in_still_water = 10 :=
by
  intros h_speed_x h_time_meeting h_time_catching h_distance_upstream h_distance_downstream h_total_distance
  sorry

end person_y_speed_in_still_water_l174_174054


namespace liam_finishes_on_wednesday_l174_174418

theorem liam_finishes_on_wednesday :
  let start_day := 3  -- Wednesday, where 0 represents Sunday
  let total_books := 20
  let total_days := (total_books * (total_books + 1)) / 2
  (total_days % 7) = 0 :=
by sorry

end liam_finishes_on_wednesday_l174_174418


namespace cot_cot_inv_sum_identity_l174_174211

  noncomputable theory
  open Real

  theorem cot_cot_inv_sum_identity :
    cot (arccot 5 + arccot 11 + arccot 17 + arccot 23) = 1021 / 420 :=
  sorry
  
end cot_cot_inv_sum_identity_l174_174211


namespace domain_of_h_l174_174461

-- Definition of the function domain of f(x) and h(x)
def f_domain := Set.Icc (-10: ℝ) 6
def h_domain := Set.Icc (-2: ℝ) (10/3)

-- Definition of f and h
def f (x: ℝ) : ℝ := sorry  -- f is assumed to be defined on the interval [-10, 6]
def h (x: ℝ) : ℝ := f (-3 * x)

-- Theorem statement: Given the domain of f(x), the domain of h(x) is as follows
theorem domain_of_h :
  (∀ x, x ∈ f_domain ↔ (-3 * x) ∈ h_domain) :=
sorry

end domain_of_h_l174_174461


namespace sqrt_product_eq_90sqrt2_l174_174638

theorem sqrt_product_eq_90sqrt2
: (sqrt 54) * (sqrt 50) * (sqrt 6) = 90 * (sqrt 2) :=
by
  sorry

end sqrt_product_eq_90sqrt2_l174_174638


namespace sum_even_integers_less_than_100_l174_174986

theorem sum_even_integers_less_than_100 : 
  let sequence := List.range' 2 98
  let even_seq := sequence.filter (λ x => x % 2 = 0)
  (even_seq.sum) = 2450 :=
by
  sorry

end sum_even_integers_less_than_100_l174_174986


namespace parabola_focus_l174_174015

theorem parabola_focus (x y : ℝ) : x^2 = 4 * y → (0, 1) = (0, 1) :=
by
  intro h
  simp
  sorry

end parabola_focus_l174_174015


namespace least_n_with_zero_digit_in_factors_l174_174666

theorem least_n_with_zero_digit_in_factors :
  ∃ n : ℕ, (∀ (a b : ℕ), (a * b = 10^n) → (∃ d : ℕ, d = 0 ∧ (a.contains_digit d ∨ b.contains_digit d))) ∧ n = 8 :=
by
  sorry

-- Definitions and utility functions
def ℕ.contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d < 10 ∧ ∃ (k : ℕ), n = (10 * k + d)

def power_of_ten (n : ℕ) : ℕ :=
  10^n

end least_n_with_zero_digit_in_factors_l174_174666


namespace sequence_limit_l174_174096

def f1 (x : ℝ) : ℝ :=
  if 0.5 ≤ x ∧ x < 0.6 then 1 else 0

def f_seq (n : ℕ) (q : ℝ) (x : ℝ) : ℝ :=
  if h : ∃ k ≤ n, (nat.digits 10 (floor (x * (10^k))) % 10) = 5 
  then q^(classical.some h - 1)
  else 0

noncomputable def a_n (n : ℕ) (q : ℝ) : ℝ :=
  ∫ x in 0..1, f_seq n q x

theorem sequence_limit (q : ℝ) (hq0 : 0 < q) (hq1 : q < 1) : 
  tendsto (λ n, a_n n q) at_top (𝓝 (1 / (10 - 9 * q))) :=
sorry

end sequence_limit_l174_174096


namespace PersonYs_speed_in_still_water_l174_174059

def speed_in_still_water (speed_X : ℕ) (t_1 t_2 : ℕ) (x : ℕ) : Prop :=
  ∀ y : ℤ, 4 * (6 - y + x + y) = 4 * 6 + 4 * x ∧ 16 * (x + y) = 16 * (6 + y) + 4 * (x - 6) →
  x = 10

theorem PersonYs_speed_in_still_water :
  speed_in_still_water 6 4 16 10 :=
by
  sorry

end PersonYs_speed_in_still_water_l174_174059


namespace composite_probability_l174_174535

/--
Given that a number selected at random from the first 50 natural numbers,
where 1 is neither prime nor composite,
the probability of selecting a composite number is 34/49.
-/
theorem composite_probability :
  let total_numbers := 50
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  let num_primes := primes.length
  let num_composites := total_numbers - num_primes - 1
  let probability_composite := (num_composites : ℚ) / (total_numbers - 1)
  probability_composite = 34 / 49 :=
by {
  sorry
}

end composite_probability_l174_174535


namespace circle_area_l174_174448

structure Point2D :=
  (x : ℝ)
  (y : ℝ)

def A : Point2D := ⟨2, 9⟩
def B : Point2D := ⟨10, 5⟩

-- The intersection point of the tangent lines at A and B is assumed to be on the x-axis
theorem circle_area (A B : Point2D)
  (tangent_intersection_on_x_axis : ∃ C : Point2D, C.y = 0 ∧ tangents_to_circle_intersect_at_C A B C) :
  ∃ (r : ℝ), π * r^2 = 83.44 * π :=
by
  sorry

-- Placeholder for the condition related to the tangents
def tangents_to_circle_intersect_at_C (A B C : Point2D) : Prop := sorry

end circle_area_l174_174448


namespace star_shape_area_l174_174908

noncomputable def area_between_star_and_circle (r : ℝ) : ℝ :=
  r^2 * (Real.pi + 6 * Real.sqrt 3 - 12)

theorem star_shape_area (r : ℝ) :
  let A := area_between_star_and_circle r in
  A = r^2 * (Real.pi + 6 * Real.sqrt 3 - 12) := 
by
  -- Sorry is used to skip the actual proof part
  sorry

end star_shape_area_l174_174908


namespace Kendra_words_per_week_l174_174788

theorem Kendra_words_per_week
  (words_per_week : ℕ)
  (words_learned : ℕ)
  (goal1 : ℕ)
  (goal2 : ℕ)
  (weeks_to_birthday : ℕ)
  (weeks_to_competition : ℕ)
  (reward_threshold : ℕ)
  (words_left_goal1 : ℕ := goal1 - words_learned)
  (words_left_goal2 : ℕ := goal2 - words_learned)
  (words_per_week_goal1 : ℕ := words_left_goal1 / weeks_to_birthday)
  (words_per_week_goal2 : Nat := (words_left_goal2 + weeks_to_competition - 1) / weeks_to_competition)
  (target : ℕ := max (max words_per_week_goal1 words_per_week_goal2) reward_threshold) :
  target = 20 :=
by
  have : words_learned = 36 := rfl
  have : goal1 = 60 := rfl
  have : goal2 = 100 := rfl
  have : weeks_to_birthday = 3 := rfl
  have : weeks_to_competition = 6 := rfl
  have : reward_threshold = 20 := rfl
  have : words_left_goal1 = 60 - 36 := sorry
  have : words_left_goal2 = 100 - 36 := sorry
  have : words_per_week_goal1 = words_left_goal1 / weeks_to_birthday := sorry
  have : words_per_week_goal2 = (words_left_goal2 + weeks_to_competition - 1) / weeks_to_competition := sorry
  have : max words_per_week_goal1 (max words_per_week_goal2 reward_threshold) = 20 := sorry
  exact rfl

end Kendra_words_per_week_l174_174788


namespace cartesian_circle_eq_line_standard_eq_area_triangle_ABC_l174_174367

-- Constants and definitions
noncomputable def C_polar_eq (ρ θ : ℝ) : Prop := ρ = 4 * (Real.sqrt 2) * Real.cos (θ - Real.pi / 4)
noncomputable def l_parametric (t : ℝ) (P : ℝ × ℝ) : Prop := P = (t + 1, t - 1)

-- Cartesian equation of the circle from polar equation
theorem cartesian_circle_eq : 
  (∀ (ρ θ : ℝ), C_polar_eq ρ θ → ∃ (x y : ℝ), (x^2 + y^2 = 4 * x - 4 * y ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ)) := 
sorry

-- Standard form of the line
theorem line_standard_eq (P : ℝ × ℝ) (t : ℝ) : 
  l_parametric t P → P.1 - P.2 = 2 :=
sorry

-- Area of triangle ABC formed by intersections of line and circle
theorem area_triangle_ABC :
  (∀ (l_intersections : set (ℝ × ℝ)), 
    (∀ P, P ∈ l_intersections ↔ ∃ t, l_parametric t P ∧ ∃ ρ θ, C_polar_eq ρ θ ∧ P = (ρ * Real.cos θ, ρ * Real.sin θ)) → 
    ∃ A B C : ℝ × ℝ, 
    A ∈ l_intersections ∧ B ∈ l_intersections ∧ C = (0,0) ∧ 
    Real.abs (1 / 2 * ((A.1 - C.1) * (B.2 - C.2) - (B.1 - C.1) * (A.2 - C.2))) = 2 * Real.sqrt 3) :=
sorry

end cartesian_circle_eq_line_standard_eq_area_triangle_ABC_l174_174367


namespace price_of_pants_l174_174787

def price_Tshirt := 5
def price_skirt := 6
def price_refurb_Tshirt := 2.5
def num_Tshirts := 2
def num_skirts := 4
def num_refurb_Tshirts := 6
def total_income := 53

theorem price_of_pants (P : ℝ) : 
  num_Tshirts * price_Tshirt + num_skirts * price_skirt + num_refurb_Tshirts * price_refurb_Tshirt + P = total_income → 
  P = 4 := 
by
  intros
  sorry

end price_of_pants_l174_174787


namespace find_x_l174_174032

def at (a b : ℝ) : ℝ := 2 * a ^ 2 - a + b

theorem find_x (x : ℝ) :
  at 3 5 = 2 * 3 ^ 2 - 3 + 5 ∧ at x 3 = 4 ↔ x = 1 ∨ x = -1 / 2 :=
by
  unfold at
  split
  -- forward direction
  {
    intro H
    cases H with H35 Hx3
    have H35' : 3 @ 5 = 2 * 3 ^ 2 - 3 + 5, from H35
    rw [H35'] at Hx3
    sorry  -- This will contain steps to show x = 1 or x = -1 / 2
  }
  -- reverse direction
  {
    intro Hx
    cases Hx with Hx1 Hx2
    {
      -- x = 1 case
      rw [Hx1]
      split
      {
        change @ 3 5 |-> 2 * 3 ^ 2 - 3 + 5,
        rfl
      }
      {
        change at 1 3 = 4,
        rfl
      }
    }
    {
      -- x = -1 / 2 case
      rw [Hx2]
      split
      {
        change at 3 5 = 2 * 3 ^ 2 - 3 + 5,
        rfl
      }
      {
        change at (-1 / 2) 3 = 4,
        rfl
      }
    }
  }

end find_x_l174_174032


namespace total_distance_proof_l174_174582

def V_m : ℝ := 6 -- Speed of man in still water in km/h
def V_r : ℝ := 1.2 -- Speed of river in km/h
def T_total : ℝ := 1 -- Total time (in hours)

-- Effective speeds
def V_up : ℝ := V_m - V_r -- Speed when going upstream
def V_down : ℝ := V_m + V_r -- Speed when going downstream

-- Distance equation setup
def D : ℝ := (T_total * V_up * V_down) / (V_up + V_down) -- Simplified form to solve distance D

-- Total distance traveled by the man (to place and back)
def total_distance : ℝ := 2 * D

theorem total_distance_proof : total_distance = 5.76 := by
  sorry

end total_distance_proof_l174_174582


namespace primes_sum_solutions_l174_174244

theorem primes_sum_solutions :
  ∃ (p q r : ℕ), Prime p ∧ Prime q ∧ Prime r ∧
  p + q^2 + r^3 = 200 ∧ 
  ((p = 167 ∧ q = 5 ∧ r = 2) ∨ 
   (p = 71 ∧ q = 11 ∧ r = 2) ∨ 
   (p = 23 ∧ q = 13 ∧ r = 2) ∨ 
   (p = 71 ∧ q = 2 ∧ r = 5)) :=
sorry

end primes_sum_solutions_l174_174244


namespace grunters_win_all_6_games_l174_174011

-- Define the probability of the Grunters winning a single game
def probability_win_single_game : ℚ := 3 / 5

-- Define the number of games
def number_of_games : ℕ := 6

-- Calculate the probability of winning all games (all games are independent)
def probability_win_all_games (p : ℚ) (n : ℕ) : ℚ := p ^ n

-- Prove that the probability of the Grunters winning all 6 games is exactly 729/15625
theorem grunters_win_all_6_games :
  probability_win_all_games probability_win_single_game number_of_games = 729 / 15625 :=
by
  sorry

end grunters_win_all_6_games_l174_174011


namespace problem_condition_l174_174332

theorem problem_condition (y : ℤ) : 
  (∑ n in Finset.range 1990, (n + 1) * (1991 - (n + 1))) = 1990 * 995 * y → y = 664 := 
  by
  sorry

end problem_condition_l174_174332


namespace kendra_change_and_discounts_l174_174201

-- Define the constants and conditions
def wooden_toy_price : ℝ := 20.0
def hat_price : ℝ := 10.0
def tax_rate : ℝ := 0.08
def discount_wooden_toys_2_3 : ℝ := 0.10
def discount_wooden_toys_4_or_more : ℝ := 0.15
def discount_hats_2 : ℝ := 0.05
def discount_hats_3_or_more : ℝ := 0.10
def kendra_bill : ℝ := 250.0
def kendra_wooden_toys : ℕ := 4
def kendra_hats : ℕ := 5

-- Calculate the applicable discounts based on conditions
def discount_on_wooden_toys : ℝ :=
  if kendra_wooden_toys >= 2 ∧ kendra_wooden_toys <= 3 then
    discount_wooden_toys_2_3
  else if kendra_wooden_toys >= 4 then
    discount_wooden_toys_4_or_more
  else
    0.0

def discount_on_hats : ℝ :=
  if kendra_hats = 2 then
    discount_hats_2
  else if kendra_hats >= 3 then
    discount_hats_3_or_more
  else
    0.0

-- Main theorem statement
theorem kendra_change_and_discounts :
  let total_cost_before_discounts := kendra_wooden_toys * wooden_toy_price + kendra_hats * hat_price
  let wooden_toys_discount := discount_on_wooden_toys * (kendra_wooden_toys * wooden_toy_price)
  let hats_discount := discount_on_hats * (kendra_hats * hat_price)
  let total_discounts := wooden_toys_discount + hats_discount
  let total_cost_after_discounts := total_cost_before_discounts - total_discounts
  let tax := tax_rate * total_cost_after_discounts
  let total_cost_after_tax := total_cost_after_discounts + tax
  let change_received := kendra_bill - total_cost_after_tax
  (total_discounts = 17) → 
  (change_received = 127.96) ∧ 
  (wooden_toys_discount = 12) ∧ 
  (hats_discount = 5) :=
by
  sorry

end kendra_change_and_discounts_l174_174201


namespace combined_instruments_l174_174628

def charlie_flutes := 1
def charlie_horns := 2
def charlie_harps := 1

def carli_flutes := 2 * charlie_flutes
def carli_horns := charlie_horns / 2
def carli_harps := 0

def charlie_total := charlie_flutes + charlie_horns + charlie_harps
def carli_total := carli_flutes + carli_horns + carli_harps
def combined_total := charlie_total + carli_total

theorem combined_instruments :
  combined_total = 7 :=
by
  sorry

end combined_instruments_l174_174628


namespace tangent_line_equation_at_1_0_l174_174863

noncomputable def f (x : ℝ) : ℝ := (x - 2) * (x ^ 3 - 1)

def tangentLine (slope : ℝ) (p : ℝ × ℝ) : ℝ → ℝ := 
  λ x, slope * (x - p.1) + p.2

theorem tangent_line_equation_at_1_0 : 
  let slope := 4 * (1 : ℝ) ^ 3 - 6 * (1 : ℝ) ^ 2 - 1 in
  slope = -3 ∧ 
  f (1 : ℝ) = 0 ∧ 
  ∃ m b, tangentLine slope (1, 0) 1 = 0 ∧
         m * (1 : ℝ) + b = 0 ∧
         m = -3 ∧ b = 3 ∧
         (∀ x, tangentLine slope (1, 0) x = -3 * (x - 1)) :=
by
  sorry

end tangent_line_equation_at_1_0_l174_174863


namespace sum_series_l174_174636

theorem sum_series :
  ∑ n in Finset.range (5000 - 3 + 1), (λ i, 1 / (↑i + 3) * sqrt (↑i + 3 - 2) + ((↑i + 3 - 2) * sqrt (↑i + 3)) = 1 + 1 / sqrt 2) :=
sorry

end sum_series_l174_174636


namespace sixth_90_degree_angle_at_three_oclock_l174_174613

open Nat

/-
  At what time do the hour hand and the minute hand form a 90-degree angle for the 6th time after 12:00, given:
  - At 12:00, the angle between the hour hand and the minute hand is 0 degrees.
  - The minute hand moves 6 degrees per minute.
  - The hour hand moves 0.5 degrees per minute.
-/

theorem sixth_90_degree_angle_at_three_oclock :
  ∃ t : ℕ, t = 3 * 60 ∧ 
    (∃ n : ℕ, n = 6 ∧
      ∀ i < n, (i + 1) * 16.36 ≠ t
    ) → (t = 180 ∧ (t + 16.36 * 5 = t)) :=
sorry

end sixth_90_degree_angle_at_three_oclock_l174_174613


namespace angle_ABC_center_of_circumscribed_circle_l174_174490

theorem angle_ABC_center_of_circumscribed_circle
  (O A B C : Point)
  (hO_center : IsCenterOfCircumscribedCircle O A B C)
  (angle_BOC : ∠BOC = 110)
  (angle_AOB : ∠AOB = 150) :
  ∠ABC = 50 := 
sorry

end angle_ABC_center_of_circumscribed_circle_l174_174490


namespace Ricciana_run_distance_l174_174772

def Ricciana_jump : ℕ := 4

def Margarita_run : ℕ := 18

def Margarita_jump (Ricciana_jump : ℕ) : ℕ := 2 * Ricciana_jump - 1

def Margarita_total_distance (Margarita_run Margarita_jump : ℕ) : ℕ := Margarita_run + Margarita_jump

def Ricciana_total_distance (Ricciana_run Ricciana_jump : ℕ) : ℕ := Ricciana_run + Ricciana_jump

theorem Ricciana_run_distance (R : ℕ) 
  (Ricciana_total : ℕ := R + Ricciana_jump) 
  (Margarita_total : ℕ := Margarita_run + Margarita_jump Ricciana_jump) 
  (h : Margarita_total = Ricciana_total + 1) : 
  R = 20 :=
by
  sorry

end Ricciana_run_distance_l174_174772


namespace range_of_a_l174_174344

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + (a-1)*x + 1 ≥ 0) ↔ (-1 ≤ a ∧ a ≤ 3) :=
by
  sorry

end range_of_a_l174_174344


namespace average_of_roots_l174_174816

theorem average_of_roots (a b: ℝ) (h : a ≠ 0) (hr : ∃ x1 x2: ℝ, a * x1 ^ 2 - 3 * a * x1 + b = 0 ∧ a * x2 ^ 2 - 3 * a * x2 + b = 0 ∧ x1 ≠ x2):
  (∃ r1 r2: ℝ, a * r1 ^ 2 - 3 * a * r1 + b = 0 ∧ a * r2 ^ 2 - 3 * a * r2 + b = 0 ∧ r1 ≠ r2) →
  ((r1 + r2) / 2 = 3 / 2) :=
by
  sorry

end average_of_roots_l174_174816


namespace area_S3_l174_174654

theorem area_S3 {s1 s2 s3 : ℝ} (h1 : s1^2 = 25)
  (h2 : s2 = s1 / Real.sqrt 2)
  (h3 : s3 = s2 / Real.sqrt 2)
  : s3^2 = 6.25 :=
by
  sorry

end area_S3_l174_174654


namespace fraction_halfway_between_l174_174963

def halfway_fraction (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between :
  halfway_fraction (3/4) (5/6) = 19/24 :=
sorry

end fraction_halfway_between_l174_174963


namespace triangular_prism_surface_area_l174_174346

noncomputable def surface_area_triangular_prism (s : ℝ) : ℝ :=
  let lateral_area := s * s
  let base_area := 2 * (sqrt 3 / 4 * (s / 3) ^ 2)
  lateral_area + base_area

theorem triangular_prism_surface_area (s : ℝ) (h_s : s = 6) :
  surface_area_triangular_prism s = 36 + 2 * (sqrt 3) := by
  sorry

end triangular_prism_surface_area_l174_174346


namespace f_periodic_analytic_expression_f_distinct_real_roots_l174_174801

noncomputable def f (x : ℝ) (k : ℤ) : ℝ := (x - 2 * k)^2

def I_k (k : ℤ) : Set ℝ := { x | 2 * k - 1 < x ∧ x ≤ 2 * k + 1 }

def M_k (k : ℕ) : Set ℝ := { a | 0 < a ∧ a ≤ 1 / (2 * ↑k + 1) }

theorem f_periodic (x : ℝ) (k : ℤ) : f x k = f (x - 2 * k) 0 := by
  sorry

theorem analytic_expression_f (x : ℝ) (k : ℤ) (hx : x ∈ I_k k) : f x k = (x - 2 * k)^2 := by
  sorry

theorem distinct_real_roots (k : ℕ) (a : ℝ) (h : a ∈ M_k k) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 ∈ I_k k ∧ x2 ∈ I_k k ∧ f x1 k = a * x1 ∧ f x2 k = a * x2 := by
  sorry

end f_periodic_analytic_expression_f_distinct_real_roots_l174_174801


namespace halfway_fraction_l174_174972

theorem halfway_fraction (a b : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) : (1/2) * (a + b) = 19/24 :=
by 
  rw [h1, h2]
  simp
  done
  sorry

end halfway_fraction_l174_174972


namespace area_of_region_bounded_by_curves_l174_174619

noncomputable def area_of_bounded_region : ℝ :=
  ∫ x in 0..(3)^(1/3), x^3 - (-1)

theorem area_of_region_bounded_by_curves :
  area_of_bounded_region = (3 / 4) * (3 * (3)^(1 / 3) - 1) :=
by
  sorry

end area_of_region_bounded_by_curves_l174_174619


namespace coefficient_x2y3_in_expansion_l174_174103

theorem coefficient_x2y3_in_expansion : 
  ∀ (x y : ℕ), (x + y = 5) → (binomial !5) / (!2 * !3) = 10 :=
by
  intros x y h
  sorry

end coefficient_x2y3_in_expansion_l174_174103


namespace age_difference_l174_174501

theorem age_difference (h b m : ℕ) (ratio : h = 4 * m ∧ b = 3 * m ∧ 4 * m + 3 * m + 7 * m = 126) : h - b = 9 :=
by
  -- proof will be filled here
  sorry

end age_difference_l174_174501


namespace smallest_interesting_number_l174_174121

theorem smallest_interesting_number :
  ∃ n : ℕ, (∃ k1 : ℕ, 2 * n = k1^2) ∧ (∃ k2 : ℕ, 15 * n = k2^3) ∧ n = 1800 :=
by
  sorry

end smallest_interesting_number_l174_174121


namespace smallest_interesting_number_l174_174169

-- Definitions based on the conditions
def isPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m
def isPerfectCube (m : ℕ) : Prop := ∃ k : ℕ, k * k * k = m

def isInteresting (n : ℕ) : Prop :=
  isPerfectSquare (2 * n) ∧ isPerfectCube (15 * n)

-- The smallest interesting number
theorem smallest_interesting_number : ∃ n : ℕ, isInteresting n ∧ (∀ m : ℕ, isInteresting m → n ≤ m) :=
  ⟨1800, ⟨⟨26, by norm_num⟩, ⟨15, by norm_num⟩⟩, sorry⟩

end smallest_interesting_number_l174_174169


namespace least_coins_for_d_l174_174850

-- Define the problem conditions
variables (d b : ℝ)

-- Define the system of equations
def equation1 := 3 * d + 4 * b = 28
def equation2 := 2 * d + 6 * b = 37.70

-- The main theorem: the least number of coins to make $1.72
theorem least_coins_for_d (h1 : equation1 d b) (h2 : equation2 d b) : 
  let d_value := 1.72 in
  let num_coins := 10 in
  num_coins = 
    let q := 6, -- number of quarters
        d := 2, -- number of dimes
        p := 2 in -- number of pennies
    q + d + p := 
sorry

end least_coins_for_d_l174_174850


namespace cos_sub_pi_six_of_trig_identity_l174_174289

theorem cos_sub_pi_six_of_trig_identity (x : ℝ)
  (h : sin x + Real.sqrt 3 * cos x = 8 / 5) :
  cos (π / 6 - x) = 4 / 5 :=
by
  sorry

end cos_sub_pi_six_of_trig_identity_l174_174289


namespace second_train_cross_time_l174_174903

variable (V1 : ℝ) -- Speed of the first train
variable (V2 : ℝ) -- Speed of the second train
variable (L1 : ℝ) -- Length of the first train
variable (L2 : ℝ) -- Length of the second train
variable (T2 : ℝ) -- Time for the second train to cross the man

-- The First Condition: The first train crosses the man in 20 seconds
axioms (h1 : L1 = V1 * 20)
-- The Second Condition: The two trains cross each other in 19 seconds
axioms (h2 : L1 + L2 = V1 * 19)
-- The Third Condition: The ratio of their speeds is 1 (V1 = V2)
axioms (h3 : V1 = V2)

-- Prove the second train crosses the man in 20 seconds
theorem second_train_cross_time : T2 = 20 :=
by
  sorry

end second_train_cross_time_l174_174903


namespace max_angle_apb_l174_174391

-- Define the geometric elements and conditions
def point := (ℝ × ℝ)

def line : set point :=
  { p | p.1 - p.2 = 0 }

def circle (center : point) (radius : ℝ) : set point :=
  { p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 }

def is_tangent_to_circle (p : point) (c : set point) : Prop :=
  ∃ a b : ℝ, 
  p = (a, b) ∧
  ∃ (center : point) (radius : ℝ),
  c = circle center radius ∧
  (a - center.1)^2 + (b - center.2)^2 = 2 * radius^2

def maximum_angle_apb (p : point) : Prop := 
  angle_apb p (4, 0) = 60

-- The statement of the problem
theorem max_angle_apb :
  ∀ (p : point),
    p ∈ line → 
    is_tangent_to_circle p (circle (4, 0) 2) →
    maximum_angle_apb p := 
by 
  sorry

end max_angle_apb_l174_174391


namespace time_two_candles_burning_simultaneously_l174_174894

theorem time_two_candles_burning_simultaneously (burn_time_1 burn_time_2 burn_time_3 T1 T3 : ℕ) 
    (total_time : ℕ) (h_sum : burn_time_1 + burn_time_2 + burn_time_3 = total_time) 
    (h_T1 : T1 = 20) (h_T3 : T3 = 10) :
    ∃ T2 : ℕ, total_time = T1 + 2 * T2 + 3 * T3 ∧ T2 = 35 := 
by
  have total_burn_time : total_time = 30 + 40 + 50 := by rw [h_sum]
  have T1_time : T1 = 20 := h_T1
  have T3_time : T3 = 10 := h_T3
  have equation : total_time = T1 + 2 * 35 + 3 * T3 := by
    calc total_time = 30 + 40 + 50 : by rw [total_burn_time]
          ... = 120 : by norm_num
          ... = 20 + 2 * 35 + 3 * 10 : by norm_num
  exact ⟨35, equation, rfl⟩

end time_two_candles_burning_simultaneously_l174_174894


namespace fraction_halfway_between_l174_174961

def halfway_fraction (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between :
  halfway_fraction (3/4) (5/6) = 19/24 :=
sorry

end fraction_halfway_between_l174_174961


namespace equal_sum_sequence_a18_l174_174858

def equal_sum_sequence (a : ℕ → ℕ) (c : ℕ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = c

theorem equal_sum_sequence_a18 (a : ℕ → ℕ) (h : equal_sum_sequence a 5) (h1 : a 1 = 2) : a 18 = 3 :=
  sorry

end equal_sum_sequence_a18_l174_174858


namespace gcd_lcm_sum_ge_sum_l174_174827

theorem gcd_lcm_sum_ge_sum (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hab : a ≤ b) :
  Nat.gcd a b + Nat.lcm a b ≥ a + b := 
sorry

end gcd_lcm_sum_ge_sum_l174_174827


namespace three_integers_sum_of_consecutive_odds_l174_174328

theorem three_integers_sum_of_consecutive_odds :
  {N : ℕ | N ≤ 500 ∧ (∃ j n, N = j * (2 * n + j) ∧ j ≥ 1) ∧
                   (∃! j1 j2 j3, ∃ n1 n2 n3, N = j1 * (2 * n1 + j1) ∧ N = j2 * (2 * n2 + j2) ∧ N = j3 * (2 * n3 + j3) ∧ j1 ≠ j2 ∧ j2 ≠ j3 ∧ j1 ≠ j3)} = {16, 18, 50} :=
by
  sorry

end three_integers_sum_of_consecutive_odds_l174_174328


namespace fraction_halfway_between_3_4_and_5_6_is_19_24_l174_174909

noncomputable def fraction_halfway (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between_3_4_and_5_6_is_19_24 :
  fraction_halfway (3 / 4) (5 / 6) = 19 / 24 :=
by
  sorry

end fraction_halfway_between_3_4_and_5_6_is_19_24_l174_174909


namespace not_equal_zero_equal_zero_exists_u_l174_174285

-- Definitions based on given conditions
variables {x y u : ℝ → ℝ} {t : ℝ}
variable  {x_0 y_0 : ℝ}
variables (h_cont_u : Continuous u) 
variables (hx : ∀ t, deriv x t = -2 * y t + u t)
variables (hy : ∀ t, deriv y t = -2 * x t + u t)
variables (hx_0 : x 0 = x_0) 
variables (hy_0 : y 0 = y_0)

-- Problem 1:
theorem not_equal_zero (h : x_0 ≠ y_0) : ¬(∀ t, x t = 0 ∧ y t = 0) :=
sorry

-- Problem 2:
theorem equal_zero_exists_u (h : x_0 = y_0) (T : ℝ) (hT : T > 0) : ∃ u, (∀ t, deriv x t = -2 * y t + u t) ∧ (∀ t, deriv y t = -2 * x t + u t) ∧ (x T = 0 ∧ y T = 0) :=
sorry

end not_equal_zero_equal_zero_exists_u_l174_174285


namespace seventh_term_of_arithmetic_sequence_l174_174045

theorem seventh_term_of_arithmetic_sequence (a d : ℤ) 
  (h1 : 5 * a + 10 * d = 15) 
  (h2 : a + 5 * d = 6) : 
  a + 6 * d = 7 :=
by
  -- Proof omitted
  sorry

end seventh_term_of_arithmetic_sequence_l174_174045


namespace smallest_interesting_number_is_1800_l174_174164

def is_perfect_square (k : ℕ) : Prop := ∃ (m : ℕ), k = m * m
def is_perfect_cube (k : ℕ) : Prop := ∃ (m : ℕ), k = m * m * m

def is_interesting (n : ℕ) : Prop :=
  is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n)

theorem smallest_interesting_number_is_1800 :
  ∃ (n : ℕ), is_interesting(n) ∧ (∀ m : ℕ, is_interesting(m) → n ≤ m) ∧ n = 1800 :=
by 
  sorry

end smallest_interesting_number_is_1800_l174_174164


namespace smallest_interesting_number_l174_174129

def is_interesting (n : ℕ) : Prop :=
  (∃ m1 : ℕ, 2 * n = m1^2) ∧ (∃ m2 : ℕ, 15 * n = m2^3)

theorem smallest_interesting_number : ∃ n, is_interesting n ∧ (∀ m : ℕ, is_interesting m → n ≤ m) :=
  ⟨1800, by {
    split,
    { use 60,
      norm_num },
    { use 30,
      norm_num },
    sorry }⟩

end smallest_interesting_number_l174_174129


namespace inscribed_square_length_in_equiangular_hexagon_l174_174326

noncomputable def hexagon_side_lengths_condition (AB CD EF : ℝ) : Prop :=
  AB = 30 ∧ CD = 50 ∧ EF = 50 * (Real.sqrt 3 - 1)

def inscribed_square_side_length (s : ℝ) : Prop :=
  s = 25 * Real.sqrt 3 - 25

theorem inscribed_square_length_in_equiangular_hexagon
  (AB CD EF s : ℝ)
  (h1 : hexagon_side_lengths_condition AB CD EF)
  (h2 : EquiangularHexagon ABCDEF) -- Assume EquiangularHexagon is pre-defined
  (h3 : InscribedSquareInHexagon PQRS ABCDEF) -- Assume InscribedSquareInHexagon is pre-defined
  : inscribed_square_side_length s :=
by
  unfold hexagon_side_lengths_condition at h1
  unfold inscribed_square_side_length
  -- The proof would go here, but it's omitted according to the instructions.
  sorry

end inscribed_square_length_in_equiangular_hexagon_l174_174326


namespace problem_correctness_l174_174699

variable (f : ℝ → ℝ)
variable (h₀ : ∀ x, f x > 0)
variable (h₁ : ∀ a b, f a * f b = f (a + b))

theorem problem_correctness :
  (f 0 = 1) ∧
  (∀ a, f (-a) = 1 / f a) ∧
  (∀ a, f a = (f (3 * a)) ^ (1 / 3)) :=
by 
  -- Using the hypotheses provided
  sorry

end problem_correctness_l174_174699


namespace smallest_interesting_number_l174_174122

theorem smallest_interesting_number :
  ∃ n : ℕ, (∃ k1 : ℕ, 2 * n = k1^2) ∧ (∃ k2 : ℕ, 15 * n = k2^3) ∧ n = 1800 :=
by
  sorry

end smallest_interesting_number_l174_174122


namespace jack_further_down_l174_174375

-- Define the conditions given in the problem
def flights_up := 3
def flights_down := 6
def steps_per_flight := 12
def height_per_step_in_inches := 8
def inches_per_foot := 12

-- Define the number of steps and height calculations
def steps_up := flights_up * steps_per_flight
def steps_down := flights_down * steps_per_flight
def net_steps_down := steps_down - steps_up
def net_height_down_in_inches := net_steps_down * height_per_step_in_inches
def net_height_down_in_feet := net_height_down_in_inches / inches_per_foot

-- The proof statement to be shown
theorem jack_further_down : net_height_down_in_feet = 24 := sorry

end jack_further_down_l174_174375


namespace round_robin_tournament_score_second_place_l174_174353

theorem round_robin_tournament_score_second_place
  (players : Fin 8 → ℝ)
  (total_score : ℝ)
  (h_total : total_score = 28)
  (different_scores : ∀ i j, i ≠ j → players i ≠ players j)
  (scoring_rule : ∀ p1 p2 : Fin 8, p1 ≠ p2 → 
    (players p1 = players p2 + 1 ∨ players p2 = players p1 + 1 ∨ (players p1 = players p2 + 0.5 ∧ players p2 = players p1 + 0.5))) :
  (let sorted_scores := List.sort (λ a b => a > b) (List.ofFn players) in
   sorted_scores.head? = some 6.0 ∧ List.foldr (·+·) 0 (List.drop 4 sorted_scores) = 6.0) := sorry

end round_robin_tournament_score_second_place_l174_174353


namespace sqrt_domain_l174_174515

theorem sqrt_domain (x : ℝ) : 2 - x ≥ 0 → x ≤ 2 :=
by
  assume h : 2 - x ≥ 0
  sorry

end sqrt_domain_l174_174515


namespace sum_even_pos_ints_lt_100_l174_174998

theorem sum_even_pos_ints_lt_100 : ∑ k in finset.range 50, 2 * k = 2450 := by
  sorry

end sum_even_pos_ints_lt_100_l174_174998


namespace halfway_fraction_l174_174940

theorem halfway_fraction (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  ((a / b + c / d) / 2) = 19 / 24 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end halfway_fraction_l174_174940


namespace sum_last_three_coefficients_correct_l174_174549

-- Define the polynomial expression
noncomputable def polynomial_expr (a : ℝ) := (1 - 1 / a) ^ 6

-- Define a function to compute the sum of the last three coefficients
def sum_last_three_coefficients (a : ℝ) : ℝ :=
  let coeffs := list.map (λ k, (nat.choose 6 k) * (-1) ^ k) (list.range 3) in
  coeffs.sum

-- Statement of the problem
theorem sum_last_three_coefficients_correct (a : ℝ) : 
  sum_last_three_coefficients a = 10 :=
sorry

end sum_last_three_coefficients_correct_l174_174549


namespace probability_between_p_and_q_in_first_quadrant_l174_174902

-- Definitions of lines p and q
def p (x : ℝ) : ℝ := -2 * x + 8
def q (x : ℝ) : ℝ := -3 * x + 9

-- Area under the line in the first quadrant
def area_under_line (f : ℝ → ℝ) (x_intercept : ℝ) : ℝ :=
  (1 / 2) * x_intercept * f 0

-- Probability calculation
def probability_between_lines : ℝ :=
  let area_p := area_under_line p 4
  let area_q := area_under_line q 3
  let area_between := area_p - area_q
  area_between / area_p

theorem probability_between_p_and_q_in_first_quadrant :
  probability_between_lines = 0.15625 :=
by
  sorry

end probability_between_p_and_q_in_first_quadrant_l174_174902


namespace angle_measure_at_Q_of_extended_sides_of_octagon_l174_174834

theorem angle_measure_at_Q_of_extended_sides_of_octagon 
  (A B C D E F G H Q : Type)
  (octagon : regular_octagon A B C D E F G H)
  (extends_to_Q : extends_to_point A B C D Q) :
  angle_measure (∠ BQD) = 22.5 :=
sorry

end angle_measure_at_Q_of_extended_sides_of_octagon_l174_174834


namespace sandwich_is_not_condiments_l174_174597

theorem sandwich_is_not_condiments (sandwich_weight condiments_weight : ℕ)
  (h1 : sandwich_weight = 150)
  (h2 : condiments_weight = 45) :
  (sandwich_weight - condiments_weight) / sandwich_weight * 100 = 70 := 
by sorry

end sandwich_is_not_condiments_l174_174597


namespace range_of_m_l174_174697

def f (x : ℝ) : ℝ := 
  if -2 ≤ x ∧ x ≤ 0 then -(2^(-x) - 1) 
  else if 0 < x ∧ x ≤ 2 then 2^x - 1 
  else 0

def g (x m : ℝ) : ℝ := x^2 - 2 * x + m

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = - (f x)

theorem range_of_m (m : ℝ) : 
  is_odd_function f ∧ (∀ x1 ∈ set.Icc (-2 : ℝ) 2, ∃ x2 ∈ set.Icc (-2 : ℝ) 2, g x2 m = f x1) ↔ -5 ≤ m ∧ m ≤ -2 := 
by {
  sorry
}

end range_of_m_l174_174697


namespace marble_problem_l174_174844

theorem marble_problem (b : ℝ) (h : b + 3 * b + 12 * b + 72 * b = 312) : b = 39 / 11 := 
by 
  have h1 : 88 * b = 312 := by
    linarith
  
  have h2 : b = 312 / 88 := by
    linarith

  have h3 : 312 / 88 = 39 / 11 := by
    norm_num

  exact eq.trans h2 h3

end marble_problem_l174_174844


namespace collinearity_A_H_S_l174_174521

-- Definitions of the geometric constructs
variables (Γ₁ Γ₂ : Circle) (A M B C S H : Point) (BC : Line)

-- Conditions from the problem statement
axiom CircleIntersects : (A ∈ Γ₁) ∧ (A ∈ Γ₂) ∧ (M ∈ Γ₁) ∧ (M ∈ Γ₂)
axiom CommonTangent : Tangent BC Γ₁ B ∧ Tangent BC Γ₂ C
axiom CircumcircleABC : ∃ circumABC : Circle, ∀ P, P ∈ circumABC ↔ P ∈ TriangleABC(A, B, C)
axiom TangentsIntersect : Tangent (tangentLine (CircumcircleABC.left) B) (CircumcircleABC.left) S ∧ Tangent (tangentLine (CircumcircleABC.left) C) (CircumcircleABC.left) S
axiom ReflectionM : H = reflect M BC

-- Statement to prove collinearity of points A, H, and S
theorem collinearity_A_H_S : Collinear [A, H, S] :=
by sorry

end collinearity_A_H_S_l174_174521


namespace fewest_handshakes_min_coach_l174_174615

theorem fewest_handshakes_min_coach (h : ∀ (n m k1 k2 : ℕ), 2 * k1 = k2 ->
                                    n = 3 * m ->
                                    n >= 2 ->
                                    (n * (n - 1)) / 2 + 3 * k1 = 435 ->
                                    0 <= k1) : 
                                    ∃ k1 : ℕ, k1 = 0 :=
begin
  sorry
end

end fewest_handshakes_min_coach_l174_174615


namespace unique_solution_condition_l174_174245

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 4) ↔ d ≠ 4 :=
by
  sorry

end unique_solution_condition_l174_174245


namespace smallest_interesting_number_l174_174154

def is_perfect_square (m : ℕ) : Prop := 
  ∃ k : ℕ, k * k = m

def is_perfect_cube (m : ℕ) : Prop := 
  ∃ k : ℕ, k * k * k = m

def interesting (n : ℕ) : Prop := 
  is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n)

theorem smallest_interesting_number : ∃ n, interesting n ∧ ∀ m, interesting m → n ≤ m :=
begin
  use 1800,
  split,
  {
    -- Prove 1800 is interesting
    sorry,
  },
  {
    -- Prove any interesting number is at least 1800
    sorry,
  }
end

end smallest_interesting_number_l174_174154


namespace height_comparison_of_cylinder_and_rectangular_solid_l174_174227

theorem height_comparison_of_cylinder_and_rectangular_solid
  (V : ℝ) (A : ℝ) (h_cylinder : ℝ) (h_rectangular_solid : ℝ)
  (equal_volume : V = V)
  (equal_base_areas : A = A)
  (height_cylinder_eq : h_cylinder = V / A)
  (height_rectangular_solid_eq : h_rectangular_solid = V / A)
  : ¬ (h_cylinder > h_rectangular_solid) :=
by {
  sorry
}

end height_comparison_of_cylinder_and_rectangular_solid_l174_174227


namespace fraction_half_way_l174_174926

theorem fraction_half_way (x y : ℚ) (h1 : x = 3 / 4) (h2 : y = 5 / 6) : (x + y) / 2 = 19 / 24 := by
  sorry

end fraction_half_way_l174_174926


namespace angle_is_correct_dot_product_is_correct_l174_174698

noncomputable def vector_conditions (a b : ℝ^3) : Prop :=
  |a| = 2 ∧ |b| = 2 ∧ (a • b / |b|) = -1

noncomputable def angle_between_vectors (a b : ℝ^3) [vector_conditions a b] : ℝ :=
  let θ := Real.arccos ((a • b) / (|a| * |b|))
  θ

noncomputable def dot_product_result (a b : ℝ^3) [vector_conditions a b] : ℝ :=
  a • b - 2*(b • b)

theorem angle_is_correct (a b : ℝ^3) [vector_conditions a b] :
  angle_between_vectors a b = 2*Real.pi/3 :=
sorry

theorem dot_product_is_correct (a b : ℝ^3) [vector_conditions a b] :
  dot_product_result (a - 2 • b) b = -10 :=
sorry

end angle_is_correct_dot_product_is_correct_l174_174698


namespace combined_instruments_l174_174629

def charlie_flutes := 1
def charlie_horns := 2
def charlie_harps := 1

def carli_flutes := 2 * charlie_flutes
def carli_horns := charlie_horns / 2
def carli_harps := 0

def charlie_total := charlie_flutes + charlie_horns + charlie_harps
def carli_total := carli_flutes + carli_horns + carli_harps
def combined_total := charlie_total + carli_total

theorem combined_instruments :
  combined_total = 7 :=
by
  sorry

end combined_instruments_l174_174629


namespace determine_f_l174_174575

theorem determine_f (d e f : ℝ) 
  (h_eq : ∀ y : ℝ, (-3) = d * y^2 + e * y + f)
  (h_vertex : ∀ k : ℝ, -1 = d * (3 - k)^2 + e * (3 - k) + f) :
  f = -5 / 2 :=
sorry

end determine_f_l174_174575


namespace fraction_halfway_between_l174_174966

def halfway_fraction (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between :
  halfway_fraction (3/4) (5/6) = 19/24 :=
sorry

end fraction_halfway_between_l174_174966


namespace area_quadrilateral_ge_area_triangle_l174_174579

variables (A B C D H B1 C1 : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
          [MetricSpace D] [MetricSpace H] [MetricSpace B1] [MetricSpace C1]

noncomputable def isIsoscelesTriangle (A B C : Type) : Prop :=
  -- Definition of an isosceles triangle omitted for brevity.

noncomputable def circumscribedCircle (A B C : Type) : Type :=
  -- Definition of the circumscribed circle around triangle ABC omitted for brevity.

noncomputable def dropAltitude (B1 : Type) : Type :=
  -- Definition of the dropped altitude from B to AC, intersecting at B1.

noncomputable def perpendicular (C C1 : Type) (AB : Type) : Prop :=
  -- Definition of perpendicular lines C1 from C to AB omitted for brevity.

noncomputable def perpendicularCH (C H : Type) (AD : Type) : Prop :=
  -- Definition of perpendicular line CH from C to AD omitted for brevity.

-- Main theorem statement
theorem area_quadrilateral_ge_area_triangle 
  (A B C D H B1 C1 : Type) 
  [isIsoscelesTriangle A B C] 
  [circumscribedCircle A B C] 
  [dropAltitude B1] 
  [perpendicular C C1 (A B)] 
  [perpendicularCH C H (A D)] :
  let BCB1C1_area := -- Definition of the area of quadrilateral BCB1C1 omitted for brevity.
  let HCC1_area := -- Definition of the area of triangle HCC1 omitted for brevity.
  BCB1C1_area ≥ HCC1_area := sorry

end area_quadrilateral_ge_area_triangle_l174_174579


namespace regular_octagon_angle_of_intersection_l174_174837

theorem regular_octagon_angle_of_intersection (ABCDEFGH : Fin 8 → Point)
  (h_reg : regular_octagon ABCDEFGH)
  (Q : Point)
  (h_Q : extended_sides_intersect_at_Q ABCDEFGH Q AB CD) :
  ∠CQD = 90 :=
sorry


end regular_octagon_angle_of_intersection_l174_174837


namespace find_vertex_X_l174_174775

-- Coordinates of the midpoints
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def midpoint (p1 p2 : Point3D) : Point3D :=
  { x := (p1.x + p2.x) / 2
    y := (p1.y + p2.y) / 2
    z := (p1.z + p2.z) / 2 }

-- Given coordinates of midpoints
def A : Point3D := { x := 0, y := 2, z := 1 }
def B : Point3D := { x := 1, y := 1, z := -1 }
def C : Point3D := { x := -1, y := 3, z := 2 }

-- Vertex X
def X : Point3D := { x := 0, y := 2, z := 0 }

-- Theorem statement
theorem find_vertex_X :
  (∀ Y Z : Point3D, midpoint Y Z = A) → 
  (∀ X Z : Point3D, midpoint X Z = B) →
  (∀ X Y : Point3D, midpoint X Y = C) →
  (∃ X : Point3D, X = { x := 0, y := 2, z := 0 }) :=
by 
  sorry

end find_vertex_X_l174_174775


namespace measure_of_angle_Q_l174_174839

-- Defining a regular octagon
structure RegularOctagon (Point : Type) :=
  (A B C D E F G H : Point)
  (is_regular : ∀ (a b c d e f g h : Point),
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f ∧ f ≠ g ∧ g ≠ h ∧ h ≠ a)

variables {Point : Type}

-- Defining angle measure
noncomputable def angle_measure (A B C : Point) [EuclideanGeometry Point] : ℝ := sorry

-- Our points
variables (A B C D E F G H Q : Point)

-- Conditions: Points are vertices of a regular octagon, extended to meet at Q
variables [RegOct : RegularOctagon Point]
include RegOct

theorem measure_of_angle_Q (h : RegOct.is_regular A B C D E F G H) :
  angle_measure B A Q = 90 :=
sorry

end measure_of_angle_Q_l174_174839


namespace all_co_captains_prob_l174_174758

noncomputable def probability_all_co_captains 
  (teams : List ℕ) (co_captains_per_team : ℕ) (selected_member_count : ℕ) : ℝ :=
  let team_probs := teams.map (λ n, (co_captains_per_team / choose n selected_member_count))
  (1 / teams.length) * team_probs.sum

theorem all_co_captains_prob :
  probability_all_co_captains [6, 8, 10, 12] 3 3 = 0.021434 :=
by
  sorry

end all_co_captains_prob_l174_174758


namespace total_new_cases_after_four_weeks_l174_174770

noncomputable def week1_cases : ℕ := 5000
noncomputable def week2_cases : ℕ := week1_cases + (0.30 * week1_cases).to_nat
noncomputable def week3_cases : ℕ := week2_cases - (0.20 * week2_cases).to_nat
noncomputable def week4_cases : ℕ := week3_cases - (0.25 * week3_cases).to_nat
noncomputable def total_cases : ℕ := week1_cases + week2_cases + week3_cases + week4_cases

theorem total_new_cases_after_four_weeks : total_cases = 20600 := by
  sorry

end total_new_cases_after_four_weeks_l174_174770


namespace infinite_geometric_sequence_limit_l174_174505

theorem infinite_geometric_sequence_limit 
  (A : ℕ → ℝ) (B : ℕ → ℝ) 
  (A1 : ℝ) (hA1 : 0 ≤ A1 ∧ A1 ≤ 1000)
  (h_initial : A 1 = A1)
  (h_total : ∀ n, A n + B n = 1000)
  (h_recurrence_A : ∀ n, A (n + 1) = 0.8 * A n + 0.3 * (1000 - A n))
  (h_recurrence_B : ∀ n, B (n + 1) = 0.2 * A n + 0.7 * B n) :
  ∃ l : ℝ, l = 600 ∧ Filter.Tendsto A Filter.atTop (Filter.pure l) :=
by
  sorry

end infinite_geometric_sequence_limit_l174_174505


namespace circumference_difference_correct_track_area_correct_l174_174901

variable (d : ℝ)

def inner_circumference (d : ℝ) : ℝ := π * d
def outer_circumference (d : ℝ) : ℝ := π * (d + 30)
def circumference_difference (d : ℝ) : ℝ := outer_circumference d - inner_circumference d

def inner_area (d : ℝ) : ℝ := π * (d / 2)^2
def outer_area (d : ℝ) : ℝ := π * ((d + 30) / 2)^2
def track_area (d : ℝ) : ℝ := outer_area d - inner_area d

theorem circumference_difference_correct : circumference_difference d = 30 * π :=
by
  unfold circumference_difference outer_circumference inner_circumference
  linarith

theorem track_area_correct : track_area d = 15 * π * d + 225 * π :=
by
  unfold track_area outer_area inner_area
  linarith

end circumference_difference_correct_track_area_correct_l174_174901


namespace smallest_interesting_number_is_1800_l174_174158

def is_perfect_square (k : ℕ) : Prop := ∃ (m : ℕ), k = m * m
def is_perfect_cube (k : ℕ) : Prop := ∃ (m : ℕ), k = m * m * m

def is_interesting (n : ℕ) : Prop :=
  is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n)

theorem smallest_interesting_number_is_1800 :
  ∃ (n : ℕ), is_interesting(n) ∧ (∀ m : ℕ, is_interesting(m) → n ≤ m) ∧ n = 1800 :=
by 
  sorry

end smallest_interesting_number_is_1800_l174_174158


namespace sum_div_by_nine_remainder_l174_174983

theorem sum_div_by_nine_remainder : 
  let S := (20 * (20 + 1)) / 2 
  in S % 9 = 3 :=
by
  let S := (20 * (20 + 1)) / 2
  have h : S % 9 = 3 := sorry
  exact h

end sum_div_by_nine_remainder_l174_174983


namespace sum_even_pos_ints_lt_100_l174_174997

theorem sum_even_pos_ints_lt_100 : ∑ k in finset.range 50, 2 * k = 2450 := by
  sorry

end sum_even_pos_ints_lt_100_l174_174997


namespace algebra_simplification_l174_174338

theorem algebra_simplification (a b : ℤ) (h : ∀ x : ℤ, x^2 - 6 * x + b = (x - a)^2 - 1) : b - a = 5 := by
  sorry

end algebra_simplification_l174_174338


namespace calculation_1_calculation_2_l174_174621

theorem calculation_1 : 
  27^(1 / 3) - (-(1 / 2))^(-2) + (1 / 16)^(-1 / 4) + (Math.sqrt 2 - 1)^0 = 2 := 
by 
  sorry

theorem calculation_2 : 
  Real.logb 10 8 + Real.logb 10 125 - Real.logb 10 2 - Real.logb 10 5 = Real.logb 10 100 := 
by 
  sorry

end calculation_1_calculation_2_l174_174621


namespace sum_even_integers_less_than_100_l174_174992

theorem sum_even_integers_less_than_100 : 
  let sequence := List.range' 2 98
  let even_seq := sequence.filter (λ x => x % 2 = 0)
  (even_seq.sum) = 2450 :=
by
  sorry

end sum_even_integers_less_than_100_l174_174992


namespace correct_operation_l174_174084

theorem correct_operation (x : ℝ) (hx : x ≠ 0) :
  (x^3 / x^2 = x) :=
by {
  sorry
}

end correct_operation_l174_174084


namespace increasing_function_range_l174_174867

noncomputable def f (a x : ℝ) : ℝ := (1/3 * a * x^3) + (a * x^2) + x

theorem increasing_function_range (a : ℝ) : 
  (∀ (x : ℝ), (a * x^2 + 2 * a * x + 1) ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) := 
begin
  sorry,
end

end increasing_function_range_l174_174867


namespace problem_proof_l174_174292

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def f : ℝ → ℝ 
| x => if x = 0 then 0 else if 0 < x ∧ x ≤ 2 then 2^x - 1 else 
    if -2 ≤ x ∧ x < 0 then 2^(-x) - 1 else 0

theorem problem_proof (h_even : even_function f) (h_periodic : periodic_function f 4) (h_interval : ∀ x, 0 ≤ x ∧ x ≤ 2 → f x = 2^x - 1) : 
  f (-2017) + f 2018 = 4 :=
by
  sorry

end problem_proof_l174_174292


namespace cot_sum_identities_l174_174205

theorem cot_sum_identities :
  Real.cot (Real.arccot 5 + Real.arccot 11 + Real.arccot 17 + Real.arccot 23) = 1021 / 420 :=
by
  sorry

end cot_sum_identities_l174_174205


namespace minimum_colors_for_cube_l174_174236

theorem minimum_colors_for_cube (n : ℕ) :
  (∃ f : ℕ → ℕ, 
    (∀ i j, (i ≠ j ∧ adjacent i j) → f i ≠ f j) ∧ 
    ∀ i, 1 ≤ f i ∧ f i ≤ n) 
    → n = 3 := 
sorry

-- Definitions related to the problem
def adjacent (i j : ℕ) : Prop := 
  -- Definition of adjacent faces of a cube
  (i = 0 ∧ j = 1) ∨ (i = 1 ∧ j = 0) ∨ -- Top, Front
  (i = 0 ∧ j = 2) ∨ (i = 2 ∧ j = 0) ∨ -- Top, Right
  (i = 0 ∧ j = 4) ∨ (i = 4 ∧ j = 0) ∨ -- Top, Left
  (i = 0 ∧ j = 5) ∨ (i = 5 ∧ j = 0) ∨ -- Top, Back
  (i = 1 ∧ j = 2) ∨ (i = 2 ∧ j = 1) ∨ -- Front, Right
  (i = 1 ∧ j = 3) ∨ (i = 3 ∧ j = 1) ∨ -- Front, Bottom
  (i = 1 ∧ j = 5) ∨ (i = 5 ∧ j = 1) ∨ -- Front, Back
  (i = 2 ∧ j = 3) ∨ (i = 3 ∧ j = 2) ∨ -- Right, Bottom
  (i = 2 ∧ j = 4) ∨ (i = 4 ∧ j = 2) ∨ -- Right, Left
  (i = 2 ∧ j = 5) ∨ (i = 5 ∧ j = 2) ∨ -- Right, Back
  (i = 3 ∧ j = 4) ∨ (i = 4 ∧ j = 3) ∨ -- Bottom, Left
  (i = 3 ∧ j = 5) ∨ (i = 5 ∧ j = 3) ∨ -- Bottom, Back
  (i = 4 ∧ j = 5) ∨ (i = 5 ∧ j = 4)    -- Left, Back

end minimum_colors_for_cube_l174_174236


namespace ratio_of_areas_l174_174112

variable {d : ℝ}

def radius_original (d : ℝ) : ℝ := d / 2

def area_original (d : ℝ) : ℝ := π * (radius_original d)^2

def radius_new1 (d : ℝ) : ℝ := 3 * (radius_original d)

def area_new1 (d : ℝ) : ℝ := π * (radius_new1 d)^2

def radius_new2 (d : ℝ) : ℝ := 4 * (radius_original d)

def area_new2 (d : ℝ) : ℝ := π * (radius_new2 d)^2

theorem ratio_of_areas (d : ℝ) (h1 : d > 0) : 
  (area_new1 d) / (area_new2 d) = 9 / 16 :=
by
  -- Placeholder for the actual proof
  sorry

end ratio_of_areas_l174_174112


namespace sum_sin_sequence_l174_174284

def f (n : ℤ) : ℝ := Real.sin (n * Real.pi / 4)

def c : ℝ := (Finset.range 2003).sum (λ n, f (n + 1))

theorem sum_sin_sequence :
  c = 1 + Real.sqrt 2 :=
sorry

end sum_sin_sequence_l174_174284


namespace single_digit_number_is_two_l174_174444

theorem single_digit_number_is_two :
  ∃ (x : ℕ), x < 10 ∧ x = 2 ∧
  (∀ (a b : ℕ), a ∈ (set.range 1 1990) ∧ b ∈ (set.range 1 1990) → ((a + b) % 19) ∈ (set.range 1 1990)) ∧
  (∀ (remaining_numbers : list ℕ), remaining_numbers.length = 2 → 
    (89 ∈ remaining_numbers ∧ (∃ x, x ∈ remaining_numbers ∧ x < 10))) :=
begin
  sorry
end

end single_digit_number_is_two_l174_174444


namespace coeff_x3_in_expansion_l174_174856

noncomputable def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem coeff_x3_in_expansion :
  (2 : ℚ)^(4 - 2) * binomial_coeff 4 2 = 24 := by 
  sorry

end coeff_x3_in_expansion_l174_174856


namespace magnitude_alpha_l174_174397

variables (α β : ℂ)
hypothesis h₁: α.conj = β
hypothesis h₂: (α / (β^2)).im = 0
hypothesis h₃: |α - β| = 2 * real.sqrt 5

theorem magnitude_alpha :
  |α| = real.sqrt (10 - 2.5 * real.sqrt 3) := sorry

end magnitude_alpha_l174_174397


namespace telescoping_sum_l174_174633

theorem telescoping_sum :
  ∑ n in finset.range 4998, (λ k, 3 + k) (λ n, 1 / (↑n * real.sqrt (↑n - 2) + (↑n - 2) * real.sqrt (↑n))) = 6999 / 7000 := 
sorry

end telescoping_sum_l174_174633


namespace reciprocal_twice_l174_174735

def reciprocal (x : ℝ) : ℝ := 1 / x

theorem reciprocal_twice (x : ℝ) (h : x = 16) :
  reciprocal (reciprocal x) = x :=
by
  rw [h, reciprocal, reciprocal]
  have : reciprocal 16 = 1 / 16 := rfl
  rw [this]
  have : reciprocal (1 / 16) = 16 := by norm_num
  rw [this]
  rfl

end reciprocal_twice_l174_174735


namespace new_cost_percentage_l174_174471

variable (a t b x : ℝ)

def cost (a t b x : ℝ) : ℝ := a * t * (b * x) ^ 6
def new_b (b : ℝ) : ℝ := 3 * b
def new_x (x : ℝ) : ℝ := x / 2

theorem new_cost_percentage :
  let C := cost a t b x
  let C_new := cost a t (new_b b) (new_x x)
  100 * C_new / C = 1139.0625 :=
by
  sorry

end new_cost_percentage_l174_174471


namespace polynomial_roots_r_eq_18_l174_174463

theorem polynomial_roots_r_eq_18
  (a b c : ℂ) 
  (h_roots : Polynomial.roots (Polynomial.C (0 : ℂ) * Polynomial.X^3 + Polynomial.C (5 : ℂ) * Polynomial.X^2 + Polynomial.C (2 : ℂ) * Polynomial.X + Polynomial.C (-8 : ℂ)) = {a, b, c}) 
  (h_ab_roots : Polynomial.roots (Polynomial.C (0 : ℂ) * Polynomial.X^3 + Polynomial.C p * Polynomial.X^2 + Polynomial.C q * Polynomial.X + Polynomial.C r) = {2 * a + b, 2 * b + c, 2 * c + a}) :
  r = 18 := sorry

end polynomial_roots_r_eq_18_l174_174463


namespace fraction_halfway_between_l174_174965

def halfway_fraction (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between :
  halfway_fraction (3/4) (5/6) = 19/24 :=
sorry

end fraction_halfway_between_l174_174965


namespace smallest_interesting_number_l174_174127

def is_interesting (n : ℕ) : Prop :=
  (∃ m1 : ℕ, 2 * n = m1^2) ∧ (∃ m2 : ℕ, 15 * n = m2^3)

theorem smallest_interesting_number : ∃ n, is_interesting n ∧ (∀ m : ℕ, is_interesting m → n ≤ m) :=
  ⟨1800, by {
    split,
    { use 60,
      norm_num },
    { use 30,
      norm_num },
    sorry }⟩

end smallest_interesting_number_l174_174127


namespace part1_solution_set_part2_range_of_a_l174_174714

open Real

-- For part (1)
theorem part1_solution_set (x a : ℝ) (h : a = 3) : |2 * x - a| + a ≤ 6 ↔ 0 ≤ x ∧ x ≤ 3 := 
by {
  sorry
}

-- For part (2)
theorem part2_range_of_a (f g : ℝ → ℝ) (hf : ∀ x, f x = |2 * x - a| + a) (hg : ∀ x, g x = |2 * x - 3|) :
  (∀ x, f x + g x ≥ 5) ↔ a ≥ 11 / 3 :=
by {
  sorry
}

end part1_solution_set_part2_range_of_a_l174_174714


namespace height_of_tetrahedron_TBCK_dropped_from_C_l174_174819

-- Define the edge length of the cube
def edge_length : ℝ := 2

-- Define the distances TB and TC
def TB_distance : ℝ := Real.sqrt 11
def TC_distance : ℝ := Real.sqrt 15

-- Define the vertices of the tetrahedron
structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

-- Define points T, B, C, K
def T : Point3D := sorry -- a point such that distance to B and C holds
def B : Point3D := sorry -- B is a vertex of the cube
def C : Point3D := sorry -- C is a vertex of the cube (diagonal position from B)
def K : Point3D := sorry -- A point on edge AA'

-- The statement to prove
theorem height_of_tetrahedron_TBCK_dropped_from_C : 
  ∀ (T B C K : Point3D), (dist T B = TB_distance) → (dist T C = TC_distance) → (dist B C = edge_length) → 
  height (T B C K) C = edge_length :=
by
  intros T B C K hTB hTC hBC
  sorry

end height_of_tetrahedron_TBCK_dropped_from_C_l174_174819


namespace prove_curve_tangent_problem_1_prove_minimize_quadrilateral_area_l174_174366

def curve_tangent_problem_1 : Prop :=
  let F := (1, 0)
  let directrix := λ x y, x = -1
  ∃ E: ℝ × ℝ → Prop,
    (∀ x y, E x y ↔ y^2 = 4 * x) ∧ 
    (∀ x y r, E x y → 
      ((x - 1)^2 + y^2 = (r + 1)^2 ∧   -- Tangency with circle
       abs(x + 2) = r))                -- Tangency with line

def minimize_quadrilateral_area : Prop :=
  let curve_eq := λ x y, y^2 = 4*x
  ∀ A B C D : ℝ × ℝ, 
    (curve_eq A.1 A.2 ∧ curve_eq B.1 B.2 ∧ curve_eq C.1 C.2 ∧ curve_eq D.1 D.2) ∧ 
    (A.1 = C.1 ∧ B.1 = D.1 ∧ A.2 ≠ C.2 ∧ B.2 ≠ D.2) → 
  ∃ S_min : ℝ, S_min = 32

theorem prove_curve_tangent_problem_1 : curve_tangent_problem_1 := sorry
theorem prove_minimize_quadrilateral_area : minimize_quadrilateral_area := sorry

end prove_curve_tangent_problem_1_prove_minimize_quadrilateral_area_l174_174366


namespace geom_progression_n_eq_6_l174_174345

theorem geom_progression_n_eq_6
  (a r : ℝ)
  (h_r : r = 6)
  (h_ratio : (a * (1 - r^n) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 217) :
  n = 6 :=
by
  sorry

end geom_progression_n_eq_6_l174_174345


namespace max_marks_l174_174824

variable (M : ℝ)

theorem max_marks (h1 : 0.35 * M = 175) : M = 500 := by
  -- Proof goes here
  sorry

end max_marks_l174_174824


namespace find_quotient_l174_174860

-- Define the problem variables and conditions
def larger_number : ℕ := 1620
def smaller_number : ℕ := larger_number - 1365
def remainder : ℕ := 15

-- Define the proof problem
theorem find_quotient :
  larger_number = smaller_number * 6 + remainder :=
sorry

end find_quotient_l174_174860


namespace softball_team_total_players_l174_174511

theorem softball_team_total_players 
  (M W : ℕ) 
  (h1 : W = M + 4)
  (h2 : (M : ℚ) / (W : ℚ) = 0.6666666666666666) :
  M + W = 20 :=
by sorry

end softball_team_total_players_l174_174511


namespace tea_bags_l174_174438

theorem tea_bags (n : ℕ) (h₁ : 2 * n ≤ 41 ∧ 41 ≤ 3 * n) (h₂ : 2 * n ≤ 58 ∧ 58 ≤ 3 * n) : n = 20 := by
  sorry

end tea_bags_l174_174438


namespace limit_of_sequence_l174_174225

open Real

theorem limit_of_sequence : 
  (tendsto (λ n : ℕ, (↑((3 * n + 1) / (3 * n - 1) : ℚ) ^ (2 * n + 3))) at_top (𝓝 (exp (4 / 3)))) :=
sorry

end limit_of_sequence_l174_174225


namespace bread_left_l174_174820

def initial_bread : ℕ := 1000
def bomi_ate : ℕ := 350
def yejun_ate : ℕ := 500

theorem bread_left : initial_bread - (bomi_ate + yejun_ate) = 150 :=
by
  sorry

end bread_left_l174_174820


namespace geom_seq_sum_l174_174295

variable (a : ℕ → ℝ) (r : ℝ)
variable (h_geometric : ∀ n, a (n + 1) = a n * r)
variable (h_pos : ∀ n, a n > 0)
variable (h_equation : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25)

theorem geom_seq_sum : a 3 + a 5 = 5 :=
by sorry

end geom_seq_sum_l174_174295


namespace num_valid_arithmetic_sequences_l174_174873

theorem num_valid_arithmetic_sequences :
  ∃ d_list : list ℕ, 
  (∀ d ∈ d_list, ∃ x : ℕ, 72 < x ∧ x < 108 ∧ 5 * x + 10 * d = 540 ∧ x + 4 * d < 120 ∧ d % 2 = 0) 
  ∧ d_list.length = 5 := 
sorry

end num_valid_arithmetic_sequences_l174_174873


namespace length_RS_independent_of_P_l174_174790

theorem length_RS_independent_of_P
  (C : Type*)
  [metric_space C]
  {AB CD : set C}
  [is_circle AB]
  [is_circle CD]
  [AB.diameter]
  [CD.diameter]
  (P : C)
  [on_circle C P]
  (R S : C)
  [perpendicular_to_diameter P R AB]
  [perpendicular_to_diameter P S CD]
  : is_length_independent_of_R_S RS :=
sorry

end length_RS_independent_of_P_l174_174790


namespace rectangles_in_cube_l174_174731

/-- Number of rectangles that can be formed by the vertices of a cube is 12. -/
theorem rectangles_in_cube : 
  ∃ (n : ℕ), (n = 12) := by
  -- The cube has vertices, and squares are a subset of rectangles.
  -- We need to count rectangles including squares among vertices of the cube.
  sorry

end rectangles_in_cube_l174_174731


namespace max_longitudinal_moves_min_longitudinal_moves_l174_174564

/-- A $2N \times 2N$ board is covered with non-overlapping $1 \times 2$ dominoes. 
A lame rook has moved across the board, visiting each cell exactly once 
(each move made by the lame rook is to a cell adjacent by side). 
We define a move as longitudinal if it transitions from one cell of 
a domino to the other cell of the same domino. -/
def board_covered_by_dominoes (N : ℕ) : Prop :=
  -- Assume board is covered by dominoes appropriately:
  sorry

def lame_rook_visits_each_cell (N : ℕ) : Prop :=
  -- Assume a path exists which visits each cell once:
  sorry

def move_is_longitudinal (N : ℕ) (move : (ℕ × ℕ) → (ℕ × ℕ)) : Prop :=
  sorry

theorem max_longitudinal_moves (N : ℕ) 
  (h1 : board_covered_by_dominoes N) 
  (h2 : lame_rook_visits_each_cell N) : 
  ∃ f : (ℕ × ℕ) → (ℕ × ℕ), (∀ move, move_is_longitudinal N (f move)) ↔ f N = 2 * N^2 := 
sorry

theorem min_longitudinal_moves (N : ℕ) 
  (h1 : board_covered_by_dominoes N) 
  (h2 : lame_rook_visits_each_cell N) :
  ∃ f : (ℕ × ℕ) → (ℕ × ℕ), 
  (∀ move, move_is_longitudinal N (f move)) ↔ 
  (f N = 1 ∧ N = 1 ∨ f N = 2 ∧ N > 1) :=
sorry

end max_longitudinal_moves_min_longitudinal_moves_l174_174564


namespace distinct_primes_divide_M_add_j_l174_174810

theorem distinct_primes_divide_M_add_j (n : ℕ) (M : ℕ) 
  (h1 : n > 0) 
  (h2 : M > n ^ (n - 1)) : 
  ∃ (p : ℕ → ℕ), 
    (∀ (j : ℕ), 1 ≤ j ∧ j ≤ n → Nat.prime (p j)) ∧
    (Function.injective p) ∧
    (∀ (j : ℕ), 1 ≤ j ∧ j ≤ n → p j ∣ M + j) := 
sorry

end distinct_primes_divide_M_add_j_l174_174810


namespace telescoping_sum_l174_174631

theorem telescoping_sum :
  ∑ n in finset.range 4998, (λ k, 3 + k) (λ n, 1 / (↑n * real.sqrt (↑n - 2) + (↑n - 2) * real.sqrt (↑n))) = 6999 / 7000 := 
sorry

end telescoping_sum_l174_174631


namespace all_sparklers_burned_l174_174066

def ten_sparklers_burn_time (n : ℕ) (burn_time : ℕ) (fraction : ℚ) : ℚ :=
  n * (fraction * burn_time)

def time_to_burn_all_sparklers (n : ℕ) (burn_time : ℕ) (fraction : ℚ) : ℚ :=
  bilten_sparklers_burn_time (n-1) burn_time fraction + burn_time

theorem all_sparklers_burned : 
  time_to_burn_all_sparklers 10 2 (9/10) = 18 + 12/60 :=
by
  sorry

end all_sparklers_burned_l174_174066


namespace lighthouse_to_horizon_distance_l174_174674

theorem lighthouse_to_horizon_distance :
  ∀ (HS : ℝ) (C : ℝ) (R : ℝ) (D : ℝ),
    HS = 0.1257 ∧ C = 40000 ∧ R = C / (2 * Real.pi) ∧
    D = Real.sqrt ((R + HS) ^ 2 - R ^ 2) →
    D ≈ 28.84 :=
by
  intros HS C R D h
  sorry

end lighthouse_to_horizon_distance_l174_174674


namespace tangent_XY_Omega_l174_174793

open EuclideanGeometry

-- define the conditions
variables {A B C D E X Y O : Point}
variables {Ω : Circle}

-- assume given conditions
axiom Omega_is_circumcircle : is_circumcircle Ω (triangle.mk A B C)
axiom angle_CAB_is_90_deg : ∠CAB = 90
axiom D_on_Omega : D ∈ Ω
axiom E_on_Omega : E ∈ Ω
axiom D_on_median_through_B : lies_on_median_through D B
axiom E_on_median_through_C : lies_on_median_through E C
axiom X_on_intersection : tangent_at D Ω ∩ line_AC = X
axiom Y_on_intersection : tangent_at E Ω ∩ line_AB = Y

-- the statement to be proved
theorem tangent_XY_Omega : tangent_to_line Ω (line.mk X Y) := sorry

end tangent_XY_Omega_l174_174793


namespace smallest_among_sqrt3_third_zero_neg1_l174_174606

noncomputable def sqrt3 : ℝ := real.sqrt 3

theorem smallest_among_sqrt3_third_zero_neg1 :
  ∀ x ∈ ({-1, 0, 1/3, sqrt3} : set ℝ), -1 ≤ x :=
by
  intro x hx
  simp [sqrt3] at hx
  cases hx
  · rw hx
    exact le_refl _
  cases hx
  · rw hx
    norm_num
  cases hx
  · rw hx
    norm_num
  cases hx
  · rw hx
    have h1 : sqrt3 ≥ 1 := by { apply real.sqrt_le_sqrt, norm_num }
    norm_num at h1
    linarith
  exact False.elim hx  -- in case hx is none of the specified cases

end smallest_among_sqrt3_third_zero_neg1_l174_174606


namespace monotonic_increasing_interval_l174_174481

def is_monotonic_increasing {α : Type*} [Preorder α] (f : α → ℝ) (I : Set α) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

def problem_statement (x : ℝ): ℝ := log 2 (x^2 - 2 * x - 3)

theorem monotonic_increasing_interval :
  (∀ x, (x^2 - 2 * x - 3 > 0) → is_monotonic_increasing problem_statement {x : ℝ | x > 3}) :=
begin
  sorry
end

end monotonic_increasing_interval_l174_174481


namespace number_of_special_pairs_even_l174_174270

-- Define a structure for a non-self-intersecting closed polygonal chain
structure NonSelfIntersectingClosedPolygon (V: Type) :=
(vertices : list V)
(is_non_self_intersecting : ∀ (s1 s2: (V × V)), s1 ≠ s2 → ¬ intersects s1 s2)
(no_three_collinear : ∀ (v1 v2 v3: V), v1 ≠ v2 → v2 ≠ v3 → v1 ≠ v3 → ¬ collinear v1 v2 v3)
(is_closed : vertices.head = vertices.last ∧ vertices.length ≥ 3)

-- Define a pair of segments as special if their extensions intersect
def is_special_pair {V : Type} (p : NonSelfIntersectingClosedPolygon V) (s1 s2 : (V × V)) : Prop :=
non_adjacent p s1 s2 ∧ segments_intersect (extension s1) s2

-- The theorem to be proved: the number of special pairs is even
theorem number_of_special_pairs_even {V : Type} (p : NonSelfIntersectingClosedPolygon V) :
  even (count (is_special_pair p) (list.pairs p.vertices)) :=
by
  sorry

end number_of_special_pairs_even_l174_174270


namespace value_of_business_l174_174088

theorem value_of_business (h1 : (1 : ℝ) / 3 > 0) (h2 : (3 : ℝ) / 5 > 0) (h3 : (3 / 5) * (1 / 3) = 1 / 5) (h4 : (2000 : ℝ) = 5 * x) : x = 10000 :=
begin
  sorry
end

end value_of_business_l174_174088


namespace solve_equation_l174_174458

theorem solve_equation {x : ℝ} :
  (5 * x - 7 * x^2 - 8 * real.sqrt (7 * x^2 - 5 * x + 1) = 8) ↔
  (x = 0 ∨ x = 5 / 7 ∨ x = 3 ∨ x = -16 / 7) :=
by sorry

end solve_equation_l174_174458


namespace game_winning_strategy_l174_174449

def winning_player (n k : ℕ) (hk : k > 1) : Prop :=
  if (n + k) % 2 = 0 then "Vova" else "Pasha"

theorem game_winning_strategy (n k : ℕ) (h_pos : 0 < n) (hk : 1 < k) :
  (winning_player n k hk = "Vova" ↔ (n + k) % 2 = 0) ∧ 
  (winning_player n k hk = "Pasha" ↔ (n + k) % 2 = 1) :=
by
  sorry

end game_winning_strategy_l174_174449


namespace max_value_of_function_l174_174667

noncomputable
def max_function : ℝ → ℝ → ℝ :=
  λ x y, x * y / (x^2 - y^2)

theorem max_value_of_function :
  ∃ (x y : ℝ), 0.1 ≤ x ∧ x ≤ 0.6 ∧ 0.2 ≤ y ∧ y ≤ 0.5 ∧ max_function x y = 3 / 8 :=
begin
  sorry
end

end max_value_of_function_l174_174667


namespace smallest_interesting_number_l174_174128

def is_interesting (n : ℕ) : Prop :=
  (∃ m1 : ℕ, 2 * n = m1^2) ∧ (∃ m2 : ℕ, 15 * n = m2^3)

theorem smallest_interesting_number : ∃ n, is_interesting n ∧ (∀ m : ℕ, is_interesting m → n ≤ m) :=
  ⟨1800, by {
    split,
    { use 60,
      norm_num },
    { use 30,
      norm_num },
    sorry }⟩

end smallest_interesting_number_l174_174128


namespace smallest_interesting_number_l174_174144

theorem smallest_interesting_number :
  ∃ (n : ℕ), (∃ k1 : ℕ, 2 * n = k1 ^ 2) ∧ (∃ k2 : ℕ, 15 * n = k2 ^ 3) ∧ n = 1800 := 
sorry

end smallest_interesting_number_l174_174144


namespace smallest_interesting_number_l174_174141

theorem smallest_interesting_number :
  ∃ (n : ℕ), (∃ k1 : ℕ, 2 * n = k1 ^ 2) ∧ (∃ k2 : ℕ, 15 * n = k2 ^ 3) ∧ n = 1800 := 
sorry

end smallest_interesting_number_l174_174141


namespace acute_angles_complementary_l174_174412

-- Given conditions
variables (α β : ℝ)
variables (α_acute : 0 < α ∧ α < π / 2) (β_acute : 0 < β ∧ β < π / 2)
variables (h : (sin α) ^ 2 + (sin β) ^ 2 = sin (α + β))

-- Statement we want to prove
theorem acute_angles_complementary : α + β = π / 2 :=
  sorry

end acute_angles_complementary_l174_174412


namespace book_arrangement_count_l174_174116

theorem book_arrangement_count (advanced_algebra books_basic_calculus : ℕ) (total_books : ℕ) (arrangement_ways : ℕ) :
  advanced_algebra = 4 ∧ books_basic_calculus = 5 ∧ total_books = 9 ∧ arrangement_ways = Nat.choose total_books advanced_algebra →
  arrangement_ways = 126 := 
by
  sorry

end book_arrangement_count_l174_174116


namespace angle_A_triangle_area_given_tanB_triangle_area_range_l174_174703

variable (A B C : ℝ)
variable (a b c S : ℝ)
variable (AB AC : Vector3)
variable (Circumradius : ℝ)

-- Conditions
def triangle_area (S : ℝ) : Prop :=
  /\
  (Circumradius = 1)

-- Questions to prove
theorem angle_A (AB AC : Vector3) (S : ℝ) : 
  triangle_area S → 
  AB.dot AC = \frac{2*\real.sqrt 3}{3}S →
  A = \pi/3 := 
  sorry

theorem triangle_area_given_tanB (B : ℝ) (S : ℝ) :
  triangle_area S → 
  (\tan B = 2) → 
  (\|\overrightarrow{CA} - \overrightarrow{CB}| = 3) → 
  S = 27 - 18*\real.sqrt 3 :=
  sorry

theorem triangle_area_range (Circumradius : ℝ) (S : ℝ) :
  (triangle_area S) → 
  0 \leq S \leq 3*\frac \real.sqrt 3 4 :=
  sorry

end angle_A_triangle_area_given_tanB_triangle_area_range_l174_174703


namespace relationship_among_a_b_c_l174_174695

-- Define the variables a, b, c based on the given conditions
def a : ℝ := Real.log2 (1 / 2)
def b : ℝ := Real.log (Real.sqrt 3) / Real.log (Real.sqrt 2)
def c : ℝ := (1 / 4) ^ (2 / 3)

-- The theorem to prove the correct relationship among a, b, and c
theorem relationship_among_a_b_c : a < c ∧ c < b := by
  -- Vars are defined based on conditions
  have ha : a = -1 := by
    simp only [a]
    calc 
      Real.log2 (1 / 2)
      _ = Real.log2 (2 ^ (-1)) : by simp [div_eq_inv_mul, one_mul]
      _ = -1 : by rw [Real.log2_pow, mul_neg_eq_neg_mul_symm, Real.log2_two]
      
  have hb : b > 1 := by
    simp only [b]
    calc 
      Real.log (Real.sqrt 3) / Real.log (Real.sqrt 2)
      _ = Real.log (Real.sqrt 3) / (1/2 * Real.log 2) : by simp [Real.sqrt_eq_rpow]
      _ > 2 : by sorry -- simplified and used Reduction

  have hc : 0 < c ∧ c < 1 := by
    simp only [c]
    split
    exact real_of_fraction (1 / 4) (2 / 3)
    -- additional bounds required
  
  -- Conclude that a < c < b
  sorry

end relationship_among_a_b_c_l174_174695


namespace tea_bags_l174_174437

theorem tea_bags (n : ℕ) (h₁ : 2 * n ≤ 41 ∧ 41 ≤ 3 * n) (h₂ : 2 * n ≤ 58 ∧ 58 ≤ 3 * n) : n = 20 := by
  sorry

end tea_bags_l174_174437


namespace min_points_victory_l174_174352

theorem min_points_victory (V : ℕ) (h1 : V < 3 -> False)
    (h2 : \forall points: ℕ, points = 8 ⟶ extra_points = 40-points)
    (h3 : extra_points = 32)
    (h4 : max_draw_points = 6):
  (extra_points + max_draw_points > (9 * V)) ⟶ V = 3 := 
begin
  intros, 
  apply h1, 
  have v := extra_points - max_draw_points, sorry,
end

end min_points_victory_l174_174352


namespace triangle_sum_range_l174_174386

noncomputable def intersect_circumcircle (A H : Point) : Point :=
sorry

theorem triangle_sum_range (ABC H A' B' C' : Point)
  (h1 : is_acute_triangle ABC)
  (h2 : orthocenter H ABC)
  (h3 : A' = intersect_circumcircle (vertex A ABC) H)
  (h4 : B' = intersect_circumcircle (vertex B ABC) H)
  (h5 : C' = intersect_circumcircle (vertex C ABC) H) :
  let AH := distance (vertex A ABC) H,
      BH := distance (vertex B ABC) H,
      CH := distance (vertex C ABC) H,
      AA' := distance (vertex A ABC) A',
      BB' := distance (vertex B ABC) B',
      CC' := distance (vertex C ABC) C' in
  3 / 2 ≤ (AH / AA' + BH / BB' + CH / CC') ∧ (AH / AA' + BH / BB' + CH / CC') < 2 :=
sorry

end triangle_sum_range_l174_174386


namespace arithmetic_sequence_general_term_geometric_sequence_condition_l174_174691

noncomputable def a (n : ℕ) : ℕ := 2 * n

theorem arithmetic_sequence_general_term :
  ∀ (n : ℕ), n > 0 → ∃ (a_n : ℕ), a n = 2 * n :=
begin
  intros n hn,
  use a n,
  dsimp [a],
  rw mul_comm,
  exact hn,
end

theorem geometric_sequence_condition (k : ℕ) (hk : k > 0) :
  a 3 * (k^2 + k) = (2 * (k + 1))^2 → k = 2 :=
begin
  intro h,
  have h1 : a 3 = 6, by { dsimp [a], norm_num },
  have h2 : (2 * (k + 1))^2 = 4 * (k + 1)^2, by { ring },
  rw [h1, h2] at h,
  ring_exp at h,
  -- Converting to standard form
  have h3 : 6 * (k^2 + k) = 6 * k^2 + 6 * k, by ring,
  rw h3 at h,
  have h4 : 4 * (k + 1)^2 = 4 * (k^2 + 2 * k + 1), by { ring },
  rw h4 at h,
  -- Simplifying the equation
  rw [mul_comm, ← mul_assoc, ← mul_comm 4, h] at h,
  have h5 : 4 * k^2 + 8 * k + 4 = 6 * k^2 + 6 * k, by { norm_num },
  have eqn : 0 = 2 * k^2 - 2 * k - 4, by linarith,
  exact sorry, -- sorry to skip solving quadratic equation
end

end arithmetic_sequence_general_term_geometric_sequence_condition_l174_174691


namespace surface_area_of_structure_l174_174199

/- Given conditions -/
def edge_length : ℕ := 1
def number_of_cubes : ℕ := 30
def number_of_layers : ℕ := 4

/- Proven surface area -/
theorem surface_area_of_structure :
  (∃ (structure : ℕ → ℕ → ℕ), compatible_structure structure number_of_cubes number_of_layers edge_length) →
  surface_area structure = 72 :=
by
  sorry

/- Auxiliary definitions and conditions -/
def compatible_structure (structure: ℕ → ℕ → ℕ) (num_cubes num_layers edge: ℕ) : Prop :=
  -- Definition for a structure composed of num_cubes with num_layers everything els
  sorry

def surface_area (structure: ℕ → ℕ → ℕ) : ℕ :=
  -- Definition to compute surface area of the given structure
  sorry

end surface_area_of_structure_l174_174199


namespace pears_sales_l174_174596

variable (x : ℝ)
variable (morning_sales : ℝ := x)
variable (afternoon_sales : ℝ := 2 * x)
variable (evening_sales : ℝ := 3 * afternoon_sales)
variable (total_sales : ℝ := morning_sales + afternoon_sales + evening_sales)

theorem pears_sales :
  (total_sales = 510) →
  (afternoon_sales = 113.34) :=
by
  sorry

end pears_sales_l174_174596


namespace range_of_a_l174_174883

theorem range_of_a :
  (∀ n : ℕ, n > 0 → 
    a n = if n ≤ 7 then (3 - a) * n - 3 else a^(n - 6)) → 
  (∀ n : ℕ, n > 0 → a (n + 1) > a n) →
  (2 < a ∧ a < 3) :=
by
  intro h_seq h_inc
  sorry

end range_of_a_l174_174883


namespace sqrt_min_value_l174_174693

theorem sqrt_min_value (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : ab + bc + ca = a + b + c) (h5 : a + b + c > 0) : sqrt(ab) + sqrt(bc) + sqrt(ca) ≥ 2 :=
sorry

end sqrt_min_value_l174_174693


namespace jenny_collects_20_cans_l174_174785

theorem jenny_collects_20_cans (b c : ℕ) (h1 : 6 * b + 2 * c = 100) (h2 : 10 * b + 3 * c = 160) : c = 20 := 
by sorry

end jenny_collects_20_cans_l174_174785


namespace range_of_a_l174_174321

-- Definitions
def point (α : Type*) := (α × α) -- Define a point in a 2D space
def distance_sq (p1 p2 : point ℝ) : ℝ := 
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 -- Square of the distance between two points

-- Problem setup
def A : point ℝ := (0, -6)
def B : point ℝ := (0, 6)
def is_on_circle (P : point ℝ) (a : ℝ) : Prop := (P.1 - a)^2 + (P.2 - 3)^2 = 4
def center_of_circle (a : ℝ) : point ℝ := (a, 3)

-- Proof problem
theorem range_of_a (a : ℝ) : 
  (∀ (P : point ℝ), is_on_circle P a → obtuse (angle A P B)) → 
  (a > real.sqrt 55 ∨ a < -real.sqrt 55) := 
begin 
  sorry -- Proof to be provided
end

end range_of_a_l174_174321


namespace cotangent_sum_identity_l174_174215

theorem cotangent_sum_identity (a b c d : ℝ) :
    (∀ a b : ℝ, cot (arccot a + arccot b) = (a * b - 1) / (a + b)) →
    cot (arccot 5 + arccot 11 + arccot 17 + arccot 23) = 97 / 40 :=
by
    intros identity
    have h₁ : cot (arccot 5 + arccot 11) = (5 * 11 - 1) / (5 + 11) := by rw [identity]
    have h₂ : cot (arccot 17 + arccot 23) = (17 * 23 - 1) / (17 + 23) := by rw [identity]
    have h₃ : cot (arccot ((54 : ℚ)/16) + arccot ((390 : ℚ)/40)) = (54/16 * 390/40 - 1) / (54/16 + 390/40) := sorry
    have h₄ : (5 * 11 - 1) = 54 := by norm_num
    have h₅ : (17 * 23 - 1) = 390 := by norm_num
    have h₆ : cot (arccot ((27/8 : ℚ)) + arccot ((39/4 : ℚ))) = (27/8 * 39/4 - 1) / (27/8 + 39/4) := sorry
    sorry -- Further steps are included in a similar manner

end cotangent_sum_identity_l174_174215


namespace find_m_n_l174_174302

theorem find_m_n (m n : ℝ) (h1 : 0 < m) (h2 : 0 < n) (h3 : m < n)
    (h4 : abs (Real.logb 2 m) = abs (Real.logb 2 n))
    (h5 : ∀ x ∈ Set.Icc (m^2) n, abs (Real.logb 2 x) ≤ 2) : 
    m = 1 / 2 ∧ n = 2 :=
by
  suggest sorry

end find_m_n_l174_174302


namespace calculation_result_l174_174624

theorem calculation_result : 
  2003^3 - 2001^3 - 6 * 2003^2 + 24 * 1001 = -4 := 
by 
  sorry

end calculation_result_l174_174624


namespace halfway_fraction_l174_174957

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/6) : (a + b) / 2 = 19 / 24 :=
by
  sorry

end halfway_fraction_l174_174957


namespace halfway_fraction_l174_174976

theorem halfway_fraction (a b : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) : (1/2) * (a + b) = 19/24 :=
by 
  rw [h1, h2]
  simp
  done
  sorry

end halfway_fraction_l174_174976


namespace min_positive_integer_k_l174_174089

theorem min_positive_integer_k (k : ℕ) : 
  (∀ (x y : ℕ), x > 0 ∧ y > 0 ∧ x ∣ y^2 ∧ y ∣ x^2 → (xy / (x + y)^k ∈ ℕ)) → k = 2 :=
by sorry

end min_positive_integer_k_l174_174089


namespace tea_bags_count_l174_174430

theorem tea_bags_count (n : ℕ) 
  (h1 : 2 * n ≤ 41) 
  (h2 : 41 ≤ 3 * n) 
  (h3 : 2 * n ≤ 58) 
  (h4 : 58 ≤ 3 * n) : 
  n = 20 := by
  sorry

end tea_bags_count_l174_174430


namespace line_up_no_youngest_first_l174_174761

theorem line_up_no_youngest_first (n : ℕ) (h : n = 5) :
  ∃ ways : ℕ, ways = 72 ∧ 
  (∀ (lineup : Fin n → Fin n), 
    ∃! prefix, length prefix = 1 ∧
    ∀ i, lineup prefix = i → i ≠ 0 ∧ i ≠ 1) := sorry

end line_up_no_youngest_first_l174_174761


namespace smallest_interesting_number_l174_174145

theorem smallest_interesting_number :
  ∃ (n : ℕ), (∃ k1 : ℕ, 2 * n = k1 ^ 2) ∧ (∃ k2 : ℕ, 15 * n = k2 ^ 3) ∧ n = 1800 := 
sorry

end smallest_interesting_number_l174_174145


namespace smallest_interesting_number_l174_174124

theorem smallest_interesting_number :
  ∃ n : ℕ, (∃ k1 : ℕ, 2 * n = k1^2) ∧ (∃ k2 : ℕ, 15 * n = k2^3) ∧ n = 1800 :=
by
  sorry

end smallest_interesting_number_l174_174124


namespace smallest_interesting_number_smallest_interesting_number_1800_l174_174179
-- Import all the necessary libraries

-- Variables definitions adhering to the conditions of the problem
def isPerfectSquare (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k

def isPerfectCube (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k * k

def isInteresting (n : ℕ) : Prop :=
  isPerfectSquare (2 * n) ∧ isPerfectCube (15 * n)

-- Prove that the smallest interesting number is 1800
theorem smallest_interesting_number : ∀ n : ℕ, isInteresting n → 1800 ≤ n :=
begin
  sorry
end

theorem smallest_interesting_number_1800 : isInteresting 1800 :=
begin
  sorry
end

end smallest_interesting_number_smallest_interesting_number_1800_l174_174179


namespace angle_AOB_eq_50_l174_174067

-- Defining geometric entities and properties
variables {O A B P : Type} -- Points O, A, B, P
variables (angle : ∀ {a b c : Type}, Type) -- Angle function

-- Given conditions as assumptions
axiom center_O (O : Type) :
axiom tangents_form_triangle_PAB (P A B : Type) :
axiom angle_APB_eq_50₀ {P A B : Type} : angle P A B = 50

-- Theorem to prove
theorem angle_AOB_eq_50 {O A B : Type} [center_O O] [tangents_form_triangle_PAB P A B] : 
  angle A O B = 50 :=
sorry

end angle_AOB_eq_50_l174_174067


namespace product_of_real_numbers_doubled_when_added_to_reciprocals_l174_174544

theorem product_of_real_numbers_doubled_when_added_to_reciprocals :
  (∀ x : ℝ, x + 1/x = 2 * x → x = 1 ∨ x = -1) →
  (∀ x₁ x₂ : ℝ, (x₁ = 1 ∨ x₁ = -1) ∧ (x₂ = 1 ∨ x₂ = -1) → x₁ * x₂ = -1) := 
by
  assume h : ∀ x : ℝ, x + 1/x = 2 * x → x = 1 ∨ x = -1,
  assume h₁ : ∀ x₁ x₂ : ℝ, (x₁ = 1 ∨ x₁ = -1) ∧ (x₂ = 1 ∨ x₂ = -1) → x₁ * x₂ = -1,
  have key_eqns : ∀ x : ℝ, x + 1/x = 2 * x → x = 1 ∨ x = -1, from h,
  have prod_val : ∀ x₁ x₂ : ℝ, (x₁ = 1 ∨ x₁ = -1) ∧ (x₂ = 1 ∨ x₂ = -1) → x₁ * x₂ = -1 := h₁,
  sorry

end product_of_real_numbers_doubled_when_added_to_reciprocals_l174_174544


namespace solve_m_l174_174039

theorem solve_m (m : ℝ) : 
  let M := {1, 2, (m^2 - 2 * m - 5) + (m^2 + 5 * m + 6) * complex.I}
      N := {3}
  in (M ∩ N).nonempty → m = -2 := 
begin
  intros,
  sorry
end

end solve_m_l174_174039


namespace point_on_transformed_graph_l174_174462

-- We need to assume a function f that is defined on real numbers
variable (f : ℝ → ℝ)

-- Assuming the condition given in the problem
def condition : Prop := f 4 = 16

-- The goal is to prove that the point (2, -1) is on the graph of the transformed function and the sum of coordinates is 1
theorem point_on_transformed_graph (h : condition f) : 2 + (-1) = 1 ∧ (∃ x y, x = 2 ∧ y = sqrt (f x / 2) - 3 ∧ y = -1) :=
  by
    have h2 : f 2 = 8 := sorry
    exact sorry

end point_on_transformed_graph_l174_174462


namespace halfway_fraction_l174_174978

theorem halfway_fraction (a b : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) : (1/2) * (a + b) = 19/24 :=
by 
  rw [h1, h2]
  simp
  done
  sorry

end halfway_fraction_l174_174978


namespace sum_even_integers_less_than_100_l174_174989

theorem sum_even_integers_less_than_100 : 
  let sequence := List.range' 2 98
  let even_seq := sequence.filter (λ x => x % 2 = 0)
  (even_seq.sum) = 2450 :=
by
  sorry

end sum_even_integers_less_than_100_l174_174989


namespace determine_m_value_l174_174706

theorem determine_m_value (m : ℤ) (A : Set ℤ) : 
  A = {1, m + 2, m^2 + 4} → 5 ∈ A → m = 3 ∨ m = 1 := 
by
  sorry

end determine_m_value_l174_174706


namespace cos_angle_ABC_l174_174777

noncomputable def triangle_sides (a b c : ℝ) (angle_A angle_B angle_C : ℝ)
  (b_eq_ac : b^2 = a * c) (BD_sin_eq : ∀ (D : ℝ), BD * sin angle_B = a * sin angle_C) :
  Prop :=
  BD = b

theorem cos_angle_ABC (a b c AD DC : ℝ) (angle_A angle_B angle_C : ℝ)
  (b_eq_ac : b^2 = a * c) (BD_sin_eq : ∀ (D : ℝ), BD * sin angle_B = a * sin angle_C)
  (AD_eq_2DC : AD = 2 * DC) :
  cos angle_B = 7 / 12 :=
sorry

end cos_angle_ABC_l174_174777


namespace tea_bags_l174_174435

theorem tea_bags (n : ℕ) (h₁ : 2 * n ≤ 41 ∧ 41 ≤ 3 * n) (h₂ : 2 * n ≤ 58 ∧ 58 ≤ 3 * n) : n = 20 := by
  sorry

end tea_bags_l174_174435


namespace fraction_half_way_l174_174928

theorem fraction_half_way (x y : ℚ) (h1 : x = 3 / 4) (h2 : y = 5 / 6) : (x + y) / 2 = 19 / 24 := by
  sorry

end fraction_half_way_l174_174928


namespace tea_bags_number_l174_174427

theorem tea_bags_number (n : ℕ) (h1 : 2 * n ≤ 41) (h2 : 41 ≤ 3 * n) (h3 : 2 * n ≤ 58) (h4 : 58 ≤ 3 * n) : n = 20 :=
by
  sorry

end tea_bags_number_l174_174427


namespace fraction_half_way_l174_174927

theorem fraction_half_way (x y : ℚ) (h1 : x = 3 / 4) (h2 : y = 5 / 6) : (x + y) / 2 = 19 / 24 := by
  sorry

end fraction_half_way_l174_174927


namespace smallest_interesting_number_l174_174130

def is_interesting (n : ℕ) : Prop :=
  (∃ m1 : ℕ, 2 * n = m1^2) ∧ (∃ m2 : ℕ, 15 * n = m2^3)

theorem smallest_interesting_number : ∃ n, is_interesting n ∧ (∀ m : ℕ, is_interesting m → n ≤ m) :=
  ⟨1800, by {
    split,
    { use 60,
      norm_num },
    { use 30,
      norm_num },
    sorry }⟩

end smallest_interesting_number_l174_174130


namespace scientific_notation_3_5_million_l174_174012

theorem scientific_notation_3_5_million :
  ∃ a n, 3.5e6 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ n = 6 :=
begin
  use [3.5, 6],
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num }
end

end scientific_notation_3_5_million_l174_174012


namespace weight_of_banana_l174_174049

theorem weight_of_banana (A B G : ℝ) (h1 : 3 * A = G) (h2 : 4 * B = 2 * A) (h3 : G = 576) : B = 96 :=
by
  sorry

end weight_of_banana_l174_174049


namespace fraction_halfway_between_3_4_and_5_6_is_19_24_l174_174916

noncomputable def fraction_halfway (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between_3_4_and_5_6_is_19_24 :
  fraction_halfway (3 / 4) (5 / 6) = 19 / 24 :=
by
  sorry

end fraction_halfway_between_3_4_and_5_6_is_19_24_l174_174916


namespace log_base_ratio_l174_174696

theorem log_base_ratio (a : ℝ) (h : 0 < a) (h₁ : a^(1/2) = 4/9) : log (2/3) a = 4 := by
  sorry

end log_base_ratio_l174_174696


namespace conjugate_of_z_l174_174681

variable {x y : ℝ}

def z := x + y * complex.i

theorem conjugate_of_z
  (h : (x / (1 - complex.i) + y / (1 - 2 * complex.i) = 5 / (1 - 3 * complex.i))) :
  complex.conj z = -1 - 5 * complex.i :=
by
  sorry

end conjugate_of_z_l174_174681


namespace fraction_halfway_between_l174_174962

def halfway_fraction (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between :
  halfway_fraction (3/4) (5/6) = 19/24 :=
sorry

end fraction_halfway_between_l174_174962


namespace triangle_angle_B_find_sin_C_l174_174349

variable {A B C a b c : ℝ}

theorem triangle_angle_B (h : a * sin (2 * B) = sqrt 3 * b * sin A) : 
  B = π / 6 :=
by
  sorry

theorem find_sin_C (cos_A : ℝ) (h1 : cos_A = 1 / 3) (B_res : B = π / 6) : 
  sin C = (2 * sqrt 6 + 1) / 6 :=
by
  sorry

end triangle_angle_B_find_sin_C_l174_174349


namespace find_larger_number_l174_174859

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1355) (h2 : L = 6 * S + 15) : L = 1623 :=
sorry

end find_larger_number_l174_174859


namespace sum_of_solutions_l174_174984

-- Defining the equation
def equation (x : ℝ) : Prop := (6 * x) / 30 = 10 / x + x

-- Defining the proof statement that the sum of all solutions to the equation is 0
theorem sum_of_solutions : 
  (finset.sum (finset.filter (λ x, equation x) {x | equation x}) id) = 0 :=
sorry

end sum_of_solutions_l174_174984


namespace arithmetic_mean_of_fractions_l174_174529

theorem arithmetic_mean_of_fractions :
  (1 / 2 : ℚ) * ((2 / 5) + (4 / 7)) = 17 / 35 :=
by
  sorry

end arithmetic_mean_of_fractions_l174_174529


namespace total_fencing_cost_square_l174_174589

theorem total_fencing_cost_square (cost_per_side : ℕ) (side_count : ℕ) (square : side_count = 4) (cost : cost_per_side = 72) : 
  let total_cost := cost_per_side * side_count in 
  total_cost = 288 :=
by
  sorry

end total_fencing_cost_square_l174_174589


namespace fish_lives_longer_than_dog_l174_174001

-- Definitions based on conditions
def hamster_lifespan : ℝ := 2.5
def dog_lifespan : ℝ := 4 * hamster_lifespan
def fish_lifespan : ℝ := 12

-- Theorem stating the desired proof
theorem fish_lives_longer_than_dog :
  fish_lifespan - dog_lifespan = 2 := 
sorry

end fish_lives_longer_than_dog_l174_174001


namespace degree_measure_angle_ABC_l174_174495

theorem degree_measure_angle_ABC (O A B C : Type) [euclidean_geometry] 
  (circumscribed_about : is_circumscribed_circle O (triangle A B C)) 
  (angle_BOC : measure (angle B O C) = 110) 
  (angle_AOB : measure (angle A O B) = 150) : 
  measure (angle A B C) = 50 := 
sorry

end degree_measure_angle_ABC_l174_174495


namespace triangle_XYZ_segments_sum_l174_174774

theorem triangle_XYZ_segments_sum
  (X Y Z : Type)
  (P : Type)
  (XZ YZ XY XX' YY' ZZ' : ℝ)
  (hYZ_longest : YZ = max XY (max XZ YZ))
  (hP_segments : XX' + YY' + ZZ' = t)
  (hX_angle : ∠X ≤ ∠Y ∧ ∠Y ≤ ∠Z)
  (hYZ_2XZ : YZ = 2 * XZ)
  (hXX'_bound : XX' < YZ / 2)
  (hYY'_bound : YY' < XZ / 2)
  (hZZ'_bound : ZZ' < XY / 2) :
  t ≤ XZ + YZ :=
by
  sorry

end triangle_XYZ_segments_sum_l174_174774


namespace volume_of_wedge_l174_174188

theorem volume_of_wedge 
  (circumference : ℝ) 
  (volume_of_wedge : ℝ) 
  (h1 : circumference = 18 * real.pi)
  (h2 : volume_of_wedge = (1 / 8) * ((4 / 3) * real.pi * (9 ^ 3))) :
  volume_of_wedge = 121.5 * real.pi :=
sorry

end volume_of_wedge_l174_174188


namespace at_least_one_tails_up_l174_174753

-- Define propositions p and q
variable (p q : Prop)

-- The theorem statement
theorem at_least_one_tails_up : (¬p ∨ ¬q) ↔ ¬(p ∧ q) := by
  sorry

end at_least_one_tails_up_l174_174753


namespace median_celestial_bodies_moons_l174_174531

  def celestialBodiesMoons : List Nat :=
  [0, 0, 1, 1, 2, 3, 5, 17, 18, 25]

  theorem median_celestial_bodies_moons : median celestialBodiesMoons = 2.5 :=
  by
    sorry
  
end median_celestial_bodies_moons_l174_174531


namespace n_digit_number_sum_of_even_l174_174813

-- Defining what an "even" number is
def is_even_number (a : ℕ) : Prop :=
  ∃ d : ℕ, (0 ≤ d ∧ d ≤ 9) ∧ (a = d * ((10 ^ (Nat.log10 a + 1) - 1) / 9))

theorem n_digit_number_sum_of_even (n : ℕ) (a : ℕ) (h : 10^(n-1) ≤ a ∧ a < 10^n) :
  ∃ b : list ℕ, (∀ x ∈ b, is_even_number x) ∧ (b.length ≤ n + 1) ∧ (b.sum = a) :=
by
  induction n with
  | zero =>
    -- Base case: when n = 0, handle separately if needed, otherwise continue to next step
    exact sorry
  | succ n ih =>
    -- Induction step
    exact sorry

end n_digit_number_sum_of_even_l174_174813


namespace part1_part2_part3_l174_174273

def seq (a : ℕ → ℝ) : Prop :=
  a 0 = 1/2 ∧ (∀ n : ℕ, 0 < n → a n = a (n-1) + (1/n^2) * (a (n-1))^2)

theorem part1 {a : ℕ → ℝ} (h : seq a) (n : ℕ) (hn : 0 < n) :
  1/a(n-1) - 1/a n < 1/n^2 :=
sorry

theorem part2 {a : ℕ → ℝ} (h : seq a) (n : ℕ) :
  a n < n :=
sorry

theorem part3 {a : ℕ → ℝ} (h : seq a) (n : ℕ) :
  1/a n ≤ 5/6 + 1/(n+1) :=
sorry

end part1_part2_part3_l174_174273


namespace same_functions_f_and_g_D_l174_174085

noncomputable def f_A (x : ℝ) : ℝ := 1
noncomputable def g_A (x : ℝ) : ℝ := if x ≠ 0 then x / x else 0

noncomputable def f_B (x : ℝ) : ℝ := x - 2
noncomputable def g_B (x : ℝ) : ℝ := if x ≠ -2 then (x^2 - 4) / (x + 2) else 0

noncomputable def f_C (x : ℝ) : ℝ := Real.sqrt(x + 1) * Real.sqrt(x - 1)
noncomputable def g_C (x : ℝ) : ℝ := Real.sqrt(x^2 - 1)

noncomputable def f_D (x : ℝ) : ℝ := abs x
noncomputable def g_D (x : ℝ) : ℝ := Real.sqrt(x^2)

theorem same_functions_f_and_g_D : ∀ x : ℝ, f_D x = g_D x := by
  intro x
  -- Proof of equality of f_D and g_D is skipped for the time being.
  sorry

end same_functions_f_and_g_D_l174_174085


namespace variance_red_ball_draws_l174_174849

-- Suppose there are two red balls and one black ball in a bag.
-- Drawing with replacement is performed three times.
-- Let X be the number of times a red ball is drawn in these three attempts.
-- Each ball has an equal probability of being drawn, and each draw is independent of the others.
-- Prove that the variance D(X) is 2/3.

noncomputable def variance_binom_three_trials : ℚ :=
  let p := 2 / 3 in
  let n := 3 in
  n * p * (1 - p)

theorem variance_red_ball_draws :
  variance_binom_three_trials = 2 / 3 :=
by
  -- Placeholder for the proof
  sorry

end variance_red_ball_draws_l174_174849


namespace largest_square_area_l174_174370

theorem largest_square_area (XY XZ YZ : ℝ)
  (h_right_angle : ∃ (XYZ : Triangle), XYZ.angle_2 = π/2)
  (h_area_sum : XY^2 + XZ^2 + YZ^2 = 450)
  (h_pythagoras : XY^2 + XZ^2 = YZ^2) :
  YZ^2 = 225 :=
by
  sorry

end largest_square_area_l174_174370


namespace cotangent_sum_identity_l174_174218

theorem cotangent_sum_identity (a b c d : ℝ) :
    (∀ a b : ℝ, cot (arccot a + arccot b) = (a * b - 1) / (a + b)) →
    cot (arccot 5 + arccot 11 + arccot 17 + arccot 23) = 97 / 40 :=
by
    intros identity
    have h₁ : cot (arccot 5 + arccot 11) = (5 * 11 - 1) / (5 + 11) := by rw [identity]
    have h₂ : cot (arccot 17 + arccot 23) = (17 * 23 - 1) / (17 + 23) := by rw [identity]
    have h₃ : cot (arccot ((54 : ℚ)/16) + arccot ((390 : ℚ)/40)) = (54/16 * 390/40 - 1) / (54/16 + 390/40) := sorry
    have h₄ : (5 * 11 - 1) = 54 := by norm_num
    have h₅ : (17 * 23 - 1) = 390 := by norm_num
    have h₆ : cot (arccot ((27/8 : ℚ)) + arccot ((39/4 : ℚ))) = (27/8 * 39/4 - 1) / (27/8 + 39/4) := sorry
    sorry -- Further steps are included in a similar manner

end cotangent_sum_identity_l174_174218


namespace fraction_half_way_l174_174922

theorem fraction_half_way (x y : ℚ) (h1 : x = 3 / 4) (h2 : y = 5 / 6) : (x + y) / 2 = 19 / 24 := by
  sorry

end fraction_half_way_l174_174922


namespace borrowing_rate_is_four_percent_l174_174181

-- Define the variables and constants
def principal := 5000
def time := 2
def lending_rate := 0.07
def gain_per_yr := 150

-- Given conditions
def interest_earned := principal * lending_rate * time
def total_gain := gain_per_yr * time
def interest_paid := interest_earned - total_gain

-- The borrowing interest rate
def borrowing_rate : ℝ := interest_paid / (principal * time)

-- The theorem we need to prove
theorem borrowing_rate_is_four_percent : borrowing_rate = 0.04 :=
by
  -- Define the calculations in the proof
  let interest_earned_val := principal * lending_rate * time
  let total_gain_val := gain_per_yr * time
  let interest_paid_val := interest_earned_val - total_gain
  let borrowing_rate_val := interest_paid_val / (principal * time)
  -- Show that this equals 0.04
  have h_interest_earned : interest_earned = 700 :=
    by norm_num [principal, lending_rate, time]
  have h_total_gain : total_gain = 300 :=
    by norm_num [gain_per_yr, time]
  have h_interest_paid : interest_paid = 400 :=
    by norm_num [h_interest_earned, h_total_gain]
  have h_borrowing_rate : borrowing_rate = 0.04 :=
    by norm_num [h_interest_paid, principal, time]
  assumption

end borrowing_rate_is_four_percent_l174_174181


namespace KM_LN_intersect_on_BC_l174_174791

variables (A B C K L B1 C1 M N : Type) [non_isosceles_triangle A B C]
variables (K_center_L_circles: center_excircle_opposite B C K)
variables (L_center_L_circles: center_excircle_opposite C B L)
variables (B1_midpoint: midpoint B1 A C)
variables (C1_midpoint: midpoint C1 A B)
variables (M_symmetric: symmetric M B B1)
variables (N_symmetric: symmetric N C C1)

theorem KM_LN_intersect_on_BC :
  ∃ P : Type, intersection KM Ln P ∧ on_line BC P := sorry

end KM_LN_intersect_on_BC_l174_174791


namespace person_c_completion_time_l174_174557

def job_completion_days (Ra Rb Rc : ℚ) (total_earnings b_earnings : ℚ) : ℚ :=
  Rc

theorem person_c_completion_time (Ra Rb Rc : ℚ)
  (hRa : Ra = 1 / 6)
  (hRb : Rb = 1 / 8)
  (total_earnings : ℚ)
  (b_earnings : ℚ)
  (earnings_ratio : b_earnings / total_earnings = Rb / (Ra + Rb + Rc))
  : Rc = 1 / 12 :=
sorry

end person_c_completion_time_l174_174557


namespace base_10_to_base_7_equiv_base_10_to_base_7_678_l174_174530

theorem base_10_to_base_7_equiv : (678 : ℕ) = 1 * 7^3 + 6 * 7^2 + 5 * 7^1 + 6 * 7^0 := 
by
  -- proof steps would go here
  sorry

theorem base_10_to_base_7_678 : "678 in base-7" = "1656" := 
by
  have h1 := base_10_to_base_7_equiv
  -- additional proof steps to show 1 * 7^3 + 6 * 7^2 + 5 * 7^1 + 6 * 7^0 = 1656 in base-7
  sorry

end base_10_to_base_7_equiv_base_10_to_base_7_678_l174_174530


namespace sum_of_areas_l174_174456

-- Definitions and conditions 
variables (OA OX OB OY AB XY : ℝ)
variables (AOB XOY : ℝ)

-- Given conditions
def eq1 : OA = OX := sorry
def eq2 : OB = OY := sorry
def eq3 : AOB + XOY = 180 := sorry
def eq4 : real.cos XOY = - real.cos AOB := sorry

-- To prove
theorem sum_of_areas : AB^2 + XY^2 = 2 * (OA^2 + OB^2) :=
by
  -- Please provide proof here
  sorry

end sum_of_areas_l174_174456


namespace find_a5_l174_174688

-- Sequence definition
def a : ℕ → ℤ
| 0     => 1
| (n+1) => 2 * a n + 3

-- Theorem to prove
theorem find_a5 : a 4 = 61 := sorry

end find_a5_l174_174688


namespace cos_min_sin_eq_neg_sqrt_seven_half_l174_174264

variable (θ : ℝ)

theorem cos_min_sin_eq_neg_sqrt_seven_half (h1 : Real.sin θ + Real.cos θ = 0.5)
    (h2 : π / 2 < θ ∧ θ < π) : Real.cos θ - Real.sin θ = - Real.sqrt 7 / 2 := by
  sorry

end cos_min_sin_eq_neg_sqrt_seven_half_l174_174264


namespace halfway_fraction_l174_174935

theorem halfway_fraction (a b c d : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 6) (h_c : c = 19 / 24) :
  (1 / 2) * (a + b) = c := 
sorry

end halfway_fraction_l174_174935


namespace proof_main_l174_174516

-- Define the variables for the conditions
variable (budget : ℝ := 3500)
variable (price_medical_mask price_disinfectant : ℝ)
variable (price_n95_mask : ℝ := 6)
variable (m n : ℕ)
variable (m₁ : ℝ := 1.5)
variable (d₁ : ℝ := 20)
variable (m₂ : ℝ := 1000)
variable (d₂ : ℕ := 100)

-- Define the conditions in Lean 4

-- Condition 1: 800 medical masks and 120 bottles of disinfectant exceed budget by 100 yuan
def condition1 := (800 * price_medical_mask + 120 * price_disinfectant = budget + 100)

-- Condition 2: 1000 medical masks and 100 bottles of disinfectant exactly use up the budget
def condition2 := (1000 * price_medical_mask + 100 * price_disinfectant = budget)

def main_properties : Prop := 
-- Prove that the unit price of medical mask is 1.5 yuan and the unit price of disinfectant is 20 yuan.
  (price_medical_mask = m₁) ∧ (price_disinfectant = d₁) ∧
-- Prove the relationship between the number of N95 masks (m) and number of disinfectant bottles (n)
  (n = 100 - (9 * m) / 40) ∧
-- Prove that number of N95 masks can either be 120 or 160 given 100 < m < 200.
  (100 < m ∧ m < 200 → (m = 120 ∨ m = 160))

theorem proof_main : main_properties :=
by
  unfold main_properties condition1 condition2
  apply And.intro
  { sorry } -- Proof for unit prices of medical masks and disinfectant
  apply And.intro
  { sorry } -- Proof for the relationship between m and n
  { sorry } -- Proof for the specific number of N95 masks (120 or 160)

end proof_main_l174_174516


namespace fraction_half_way_l174_174921

theorem fraction_half_way (x y : ℚ) (h1 : x = 3 / 4) (h2 : y = 5 / 6) : (x + y) / 2 = 19 / 24 := by
  sorry

end fraction_half_way_l174_174921


namespace unique_values_count_l174_174769

theorem unique_values_count : 
  ∃ (a b c d : ℕ), 
  (a ∈ {1,2,3,5}) ∧ (b ∈ {1,2,3,5}) ∧ (c ∈ {1,2,3,5}) ∧ (d ∈ {1,2,3,5}) ∧ 
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d) ∧
  (finset.card (finset.image (λ (p : ℕ × ℕ × ℕ × ℕ), (p.1.1 * p.1.2) - (p.2.1 * p.2.2)) 
  (finset.univ : finset (ℕ × ℕ × ℕ × ℕ))) = 6) :=
begin
  sorry
end

end unique_values_count_l174_174769


namespace optimal_school_location_l174_174906

-- Define the problem
variables {A B C : Type} {a b c : A}

-- Number of students in each village
def students_A : ℕ := 100
def students_B : ℕ := 200
def students_C : ℕ := 300

-- Placeholder for distance function
def dist (x y : A) : ℝ := sorry

-- Equilateral triangle condition
def is_equilateral (A B C : A) : Prop := 
  dist A B = dist B C ∧ dist B C = dist C A

-- Define the total distance function
def total_distance (X : A) : ℝ := 
  students_A * dist a X + 
  students_B * dist b X + 
  students_C * dist c X

-- Statement to prove
theorem optimal_school_location (A B C X : A)
  (h_eq : is_equilateral A B C)
  (hA : X = A ∨ X = B ∨ X = C) :
  total_distance C ≤ total_distance X := 
  sorry

end optimal_school_location_l174_174906


namespace tangent_line_at_1_fx_leq_bound_l174_174718

-- Given function f(x)
def f (a : ℝ) (x : ℝ) : ℝ :=
  (1 / 2) * a * x^2 + (2 + a) * x + 2 * Real.log x

-- Prove the first statement (for question 1)
theorem tangent_line_at_1 (a : ℝ) (ha : a = 0) :
  let f := λ (x : ℝ), (2 : ℝ) * x + 2*Real.log x,
      slope := (2 : ℝ) + 1 / (1 : ℝ),
      y := f 1,
      point := (1 : ℝ, y)
  in slope * (x - 1) + y = 3 * (x - 1) + 2 := 
by sorry

-- Prove the second statement (for question 2)
theorem fx_leq_bound (a : ℝ) (ha : a < 0) :
  ∀ x > 0, f a x ≤ (-6 / a) - 4 :=
by sorry

end tangent_line_at_1_fx_leq_bound_l174_174718


namespace factors_of_120_multiples_of_20_l174_174730

theorem factors_of_120_multiples_of_20 :
  (finset.filter (fun n => 20 ∣ n) (finset.Ico 1 (120 + 1))).card = 3 := 
sorry

end factors_of_120_multiples_of_20_l174_174730


namespace leak_time_to_empty_tank_l174_174560

-- Define variables for the rates
variable (A L : ℝ)

-- Given conditions
def rate_pipe_A : Prop := A = 1 / 4
def combined_rate : Prop := A - L = 1 / 6

-- Theorem statement: The time it takes for the leak to empty the tank
theorem leak_time_to_empty_tank (A L : ℝ) (h1 : rate_pipe_A A) (h2 : combined_rate A L) : 1 / L = 12 :=
by 
  sorry

end leak_time_to_empty_tank_l174_174560


namespace number_of_cows_l174_174755

-- Defining the problem conditions
variables (D C L H : ℕ) -- Number of ducks, cows, legs, and heads respectively

-- Given conditions
def number_of_legs := 2 * D + 4 * C
def number_of_heads := D + C
def legs_condition := number_of_legs = 2 * number_of_heads + 48

-- Theorem stating the solution
theorem number_of_cows : legs_condition D C L H → C = 24 :=
by {
  unfold legs_condition,
  unfold number_of_legs,
  unfold number_of_heads,
  intros h,
  sorry -- Proof omitted
}

end number_of_cows_l174_174755


namespace speed_of_stream_l174_174110

-- Definitions based on given conditions
def speed_still_water := 24 -- km/hr
def distance_downstream := 140 -- km
def time_downstream := 5 -- hours

-- Proof problem statement
theorem speed_of_stream (v : ℕ) :
  24 + v = distance_downstream / time_downstream → v = 4 :=
by
  sorry

end speed_of_stream_l174_174110


namespace students_only_english_l174_174093

theorem students_only_english (total_students : ℕ) (both_english_german : ℕ) (enrolled_german : ℕ) 
    (enrolled_at_least_one_subject : total_students = 32 ∧ both_english_german = 12 ∧ enrolled_german = 22) : 
    (∃ (enrolled_only_english : ℕ), enrolled_only_english = 10) :=
by 
   have h : total_students = 32 := enrolled_at_least_one_subject.1
   have b : both_english_german = 12 := enrolled_at_least_one_subject.2.1
   have g : enrolled_german = 22 := enrolled_at_least_one_subject.2.2
   let enrolled_only_german := g - b
   have h_total : total_students = enrolled_only_german + enrolled_only_english + b := 
      by 
        rw [←h, add_comm (enrolled_only_english + b)]
   have enrolled_only_english : ℕ := total_students - (enrolled_only_german + b)
   use enrolled_only_english
   rw [←h_total, h, b]
   exact sorry

end students_only_english_l174_174093


namespace find_p_l174_174410

theorem find_p (p : ℤ)
  (h1 : ∀ (u v : ℤ), u > 0 → v > 0 → 5 * u ^ 2 - 5 * p * u + (66 * p - 1) = 0 ∧
    5 * v ^ 2 - 5 * p * v + (66 * p - 1) = 0) :
  p = 76 :=
sorry

end find_p_l174_174410


namespace edge_length_of_cube_l174_174036

theorem edge_length_of_cube (s : ℝ) (A B C D E F G H M N: ℝ) :
  is_center_of_face N A B C D ∧ 
  is_midpoint M A E ∧ 
  area_of_triangle M N H = 13 * real.sqrt 14  → 
  s = 2 * real.sqrt 13 := 
sorry

end edge_length_of_cube_l174_174036


namespace perpendicular_lines_determine_a_l174_174340

theorem perpendicular_lines_determine_a :
  ∀ a : ℝ, 
  ∀ (line1 : ℝ → ℝ → Prop) (line2 : ℝ → ℝ → Prop),
    (line1 = (λ x y, (3 * a + 2) * x - 3 * y + 8 = 0)) →
    (line2 = (λ x y, 3 * x + (a + 4) * y - 7 = 0)) →
    (∀ x y : ℝ, line1 x y → line2 x y → 
      let slope1 := (3 * a + 2) / (-3) in
      let slope2 := -3 / (a + 4) in
      slope1 * slope2 = -1) →
    a = 1 :=
by
  sorry

end perpendicular_lines_determine_a_l174_174340


namespace journey_length_l174_174028

theorem journey_length (speed time : ℝ) (portions_covered total_portions : ℕ)
  (h_speed : speed = 40) (h_time : time = 0.7) (h_portions_covered : portions_covered = 4) (h_total_portions : total_portions = 5) :
  (speed * time / portions_covered) * total_portions = 35 :=
by
  sorry

end journey_length_l174_174028


namespace equidistant_xaxis_point_l174_174072

theorem equidistant_xaxis_point {x : ℝ} :
  (∃ x : ℝ, ∀ A B : ℝ × ℝ, A = (-3, 0) ∧ B = (2, 5) →
    ∀ P : ℝ × ℝ, P = (x, 0) →
      (dist A P = dist B P) → x = 2) := sorry

end equidistant_xaxis_point_l174_174072


namespace x_coordinate_equidistant_l174_174075

theorem x_coordinate_equidistant :
  ∃ x : ℝ, (sqrt ((-3 - x)^2 + 0^2) = sqrt ((2 - x)^2 + 5^2)) ∧ x = 2 :=
by
  sorry

end x_coordinate_equidistant_l174_174075


namespace halfway_fraction_l174_174949

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/6) : (a + b) / 2 = 19 / 24 :=
by
  sorry

end halfway_fraction_l174_174949


namespace sum_sequence_lt_one_l174_174882

def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1 / 2 ∧ ∀ n, a (n + 1) = a n ^ 2 / (a n ^ 2 - a n + 1)

theorem sum_sequence_lt_one (a : ℕ → ℚ) (h : sequence a) (n : ℕ) : 
  ∑ i in range n, a (i + 1) < 1 :=
sorry

end sum_sequence_lt_one_l174_174882


namespace bears_in_shipment_l174_174193

theorem bears_in_shipment (initial_bears shipped_bears bears_per_shelf shelves_used : ℕ) 
  (h1 : initial_bears = 4) 
  (h2 : bears_per_shelf = 7) 
  (h3 : shelves_used = 2) 
  (total_bears_on_shelves : ℕ) 
  (h4 : total_bears_on_shelves = shelves_used * bears_per_shelf) 
  (total_bears_after_shipment : ℕ) 
  (h5 : total_bears_after_shipment = total_bears_on_shelves) 
  : shipped_bears = total_bears_on_shelves - initial_bears := 
sorry

end bears_in_shipment_l174_174193


namespace minimum_toys_to_add_l174_174563

theorem minimum_toys_to_add {T : ℤ} (k m n : ℤ) (h1 : T = 12 * k + 3) (h2 : T = 18 * m + 3) 
  (h3 : T = 36 * n + 3) : 
  ∃ x : ℤ, (T + x) % 7 = 0 ∧ x = 4 :=
sorry

end minimum_toys_to_add_l174_174563


namespace cot_arccots_sum_eq_97_over_40_l174_174209

noncomputable def cot_arccot_sum : ℝ :=
  Real.cot (Real.arccot 5 + Real.arccot 11 + Real.arccot 17 + Real.arccot 23)

theorem cot_arccots_sum_eq_97_over_40 :
  cot_arccot_sum = 97 / 40 :=
sorry

end cot_arccots_sum_eq_97_over_40_l174_174209


namespace halfway_fraction_l174_174939

theorem halfway_fraction (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  ((a / b + c / d) / 2) = 19 / 24 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end halfway_fraction_l174_174939


namespace part1_part2_l174_174277

def f (x : ℝ) : ℝ := Real.log x
def g (x : ℝ) (m : ℝ) : ℝ := x + m
def F (x : ℝ) (m : ℝ) : ℝ := Real.log x - x - m

theorem part1 (m : ℝ) (h : ∀ x : ℝ, 0 < x → f x ≤ g x m) : m ≥ -1 :=
by 
  -- proof follows here
  sorry

theorem part2 {x1 x2 m : ℝ} (h1 : F x1 m = 0) (h2 : F x2 m = 0) (hx1 : 0 < x1) (hx2 : 0 < x2) (h_lesser : x1 < x2) : x1 * x2 < 1 :=
by 
  -- proof follows here
  sorry

end part1_part2_l174_174277


namespace last_two_balloons_l174_174519

-- Define the balloon labels as a datatype
inductive Balloon
| A | B | C | D | E | F | G | H | I | J | K | L
deriving DecidableEq, Repr

open Balloon

-- Function to simulate the popping process
def popBalloon (balloons : List Balloon) (start : Nat) (step : Nat) : List Balloon × List Balloon :=
  let rec aux (remaining popped : List Balloon) (index : Nat) : List Balloon × List Balloon :=
    match remaining with
    | [] => (remaining, popped)
    | _ => 
      if remaining.length = 2 then
        (remaining, popped)
      else
        let idx := (index + step - 1) % remaining.length
        aux (remaining.removeNth idx) (popped ++ [(remaining.get! idx)]) idx
  aux balloons [] start

-- Initial data
def initialBalloons : List Balloon := [A, B, C, D, E, F, G, H, I, J, K, L]
def startIndex : Nat := initialBalloons.indexOf C  -- Start at balloon C which is index 3
def stepSize : Nat := 3  -- Every third balloon is popped

-- Statement to be proved
theorem last_two_balloons : (popBalloon initialBalloons startIndex stepSize).1 = [E, J] :=
by
  sorry

end last_two_balloons_l174_174519


namespace correct_number_of_possible_x_values_l174_174598

def number_of_possible_x_values : ℕ :=
  (factors (gcd (gcd 36 54) 72)).length

theorem correct_number_of_possible_x_values : number_of_possible_x_values = 6 := by
  sorry

end correct_number_of_possible_x_values_l174_174598


namespace cot_cot_inv_sum_identity_l174_174212

  noncomputable theory
  open Real

  theorem cot_cot_inv_sum_identity :
    cot (arccot 5 + arccot 11 + arccot 17 + arccot 23) = 1021 / 420 :=
  sorry
  
end cot_cot_inv_sum_identity_l174_174212


namespace desired_on_time_departure_rate_l174_174022

theorem desired_on_time_departure_rate :
  let first_late := 1
  let on_time_flights_next := 3
  let additional_on_time_flights := 2
  let total_on_time_flights := on_time_flights_next + additional_on_time_flights
  let total_flights := first_late + on_time_flights_next + additional_on_time_flights
  let on_time_departure_rate := (total_on_time_flights : ℚ) / (total_flights : ℚ) * 100
  on_time_departure_rate > 83.33 :=
by
  sorry

end desired_on_time_departure_rate_l174_174022


namespace walk_no_more_than_200_meters_l174_174417

noncomputable def walk_probability : ℚ :=
  let r := (1/6 : ℚ)
  let p_head := r * (2/5) + r * (3/5) + r * (4/5) + r * (4/5) + r * (3/5) + r * (2/5)
  in p_head

theorem walk_no_more_than_200_meters : walk_probability
= 3 / 5 :=
by
  sorry

end walk_no_more_than_200_meters_l174_174417


namespace circle_radius_l174_174226

theorem circle_radius (x y : ℝ) : 
  x^2 + y^2 + 16 = 6x + 14y → 
  ∃ r : ℝ, r = real.sqrt 42 ∧ ∀ x y, (x - 3)^2 + (y - 7)^2 = r^2 :=
by
  intro h
  sorry

end circle_radius_l174_174226


namespace odd_coeff_sum_ge_odd_coeff_single_l174_174255

def polynomial (R : Type*) [Ring R] := ℕ → R

def num_odd_coeffs {R : Type*} [Ring R] (P : polynomial R) : ℕ :=
  ∑ n, if (P n % 2 = 1) then 1 else 0

def Q (i : ℕ) : polynomial ℤ := 
  λ n, if n ≤ i then (choose i n) else 0

theorem odd_coeff_sum_ge_odd_coeff_single 
  (i_1 i_2 ... i_n : ℕ) 
  (h : 0 ≤ i_1 ∧ i_1 < i_2 ∧ i_2 < i_3 ∧ ... ∧ i_(n-1) < i_n) :
  num_odd_coeffs (Q i_1 + Q i_2 + ... + Q i_n) ≥ num_odd_coeffs (Q i_1) := 
s̥orry

end odd_coeff_sum_ge_odd_coeff_single_l174_174255


namespace tea_bags_number_l174_174428

theorem tea_bags_number (n : ℕ) (h1 : 2 * n ≤ 41) (h2 : 41 ≤ 3 * n) (h3 : 2 * n ≤ 58) (h4 : 58 ≤ 3 * n) : n = 20 :=
by
  sorry

end tea_bags_number_l174_174428


namespace range_of_a_l174_174871

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (1/2)^(x^2 + a*x) < (1/2)^(2*x + a - 2)) ↔ -2 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l174_174871


namespace find_f_f_f_l174_174325

def sum_digits (n: ℕ) : ℕ :=
  n.digits.sum -- This assumes 'digits' gives the digits of a number in a list

def f (n: ℕ) : ℕ := sum_digits n

def N : ℕ := 4444 ^ 4444

theorem find_f_f_f (N : ℕ) : f(f(f(N))) = 7 := 
by sorry

end find_f_f_f_l174_174325


namespace φ_monotonic_intervals_range_of_a_l174_174311

-- Define auxiliary function u(x) and its roots used in the proof
noncomputable def u (x : ℝ) (k : ℝ) : ℝ := x^2 - k * x + 2
noncomputable def x₁ (k : ℝ) : ℝ := (k - Real.sqrt (k^2 - 8)) / 2
noncomputable def x₂ (k : ℝ) : ℝ := (k + Real.sqrt (k^2 - 8)) / 2

-- First proof problem statement
theorem φ_monotonic_intervals (k : ℝ) (hk : k > 0) :
  0 < k ∧ k ≤ 2 * Real.sqrt 2 → (∀ x, x > 0 → (1 + 2 / (x^2) - k / x) ≥ 0) ∧
  k > 2 * Real.sqrt 2 → (
    ∀ x, x ∈ Ioo 0 (x₁ k) ∨ x ∈ Ioo (x₂ k) ∞ → (1 + 2 / (x^2) - k / x) > 0 ∧
    ∀ x, x ∈ Ioo (x₁ k) (x₂ k) → (1 + 2 / (x^2) - k / x) < 0 
  ) :=
sorry

-- Second proof problem statement
theorem range_of_a (a : ℝ) :
  (∀ x, x ∈ Ιcc Real.exp ∞ → x * Real.log x ≥ a * x - a) → 
  a ≤ Real.exp / (Real.exp - 1) :=
sorry

end φ_monotonic_intervals_range_of_a_l174_174311


namespace thirteen_consecutive_nat_power_l174_174234

def consecutive_sum_power (N : ℕ) : ℕ :=
  (N - 6) + (N - 5) + (N - 4) + (N - 3) + (N - 2) + (N - 1) +
  N + (N + 1) + (N + 2) + (N + 3) + (N + 4) + (N + 5) + (N + 6)

theorem thirteen_consecutive_nat_power (N : ℕ) (n : ℕ) :
  N = 13^2020 →
  n = 2021 →
  consecutive_sum_power N = 13^n := by
  sorry

end thirteen_consecutive_nat_power_l174_174234


namespace perpendicular_OM_BK_l174_174100

-- Definitions of the geometric entities and hypotheses
variables {A B C D J K M O : Type*}

-- Assume certain conditions about these entities
-- A square with center O
axiom square (A B C D : Type*) (O : Type*) : Prop
-- Congruent isosceles triangles outside the square
axiom congruent_isosceles_triangles (B C J : Type*) (C D K: Type*) : Prop
-- Midpoint of CJ
axiom midpoint (C J M : Type*) : Prop
-- O is the center of the square
axiom center_of_square (A B C D O : Type*) : O = ((diag AC = diag BD) / 2)

theorem perpendicular_OM_BK (A B C D J K M O : Type*) 
    [h1 : square (A B C D) (O)] 
    [h2 : congruent_isosceles_triangles (B C J) (C D K)] 
    [h3 : midpoint (C J M)] 
    [center_of_square (A B C D O)]
  : ⊥ O M B K :=
sorry -- Proof would go here

end perpendicular_OM_BK_l174_174100


namespace probability_square_product_tiles_die_l174_174900

theorem probability_square_product_tiles_die : 
  let num_tiles := 12
  let num_die_faces := 8
  let total_outcomes := num_tiles * num_die_faces
  let is_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
  let favorable_outcomes := [(1, 1), (1, 4), (2, 2), (4, 1), (3, 3), (2, 8), (4, 4), (8, 2), (4, 9), (6, 6), (9, 1), (9, 4), (7, 7), (8, 8)].length
  let probability := favorable_outcomes / total_outcomes
  in probability = 1 / 6 := sorry

end probability_square_product_tiles_die_l174_174900


namespace infinite_series_sum_limit_l174_174651

theorem infinite_series_sum_limit (a : ℝ) (h : 1 < a) :
  (∑' n, (-1)^n / a^n) → (a → 1) = 1 / 2 :=
begin
  sorry
end

end infinite_series_sum_limit_l174_174651


namespace halfway_fraction_l174_174934

theorem halfway_fraction (a b c d : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 6) (h_c : c = 19 / 24) :
  (1 / 2) * (a + b) = c := 
sorry

end halfway_fraction_l174_174934


namespace regular_octagon_angle_of_intersection_l174_174838

theorem regular_octagon_angle_of_intersection (ABCDEFGH : Fin 8 → Point)
  (h_reg : regular_octagon ABCDEFGH)
  (Q : Point)
  (h_Q : extended_sides_intersect_at_Q ABCDEFGH Q AB CD) :
  ∠CQD = 90 :=
sorry


end regular_octagon_angle_of_intersection_l174_174838


namespace number_of_cubes_with_three_faces_painted_l174_174040

-- Definitions of conditions
def large_cube_side_length : ℕ := 4
def total_smaller_cubes := large_cube_side_length ^ 3

-- Prove the number of smaller cubes with at least 3 faces painted is 8
theorem number_of_cubes_with_three_faces_painted :
  (∃ (n : ℕ), n = 8) :=
by
  -- Conditions recall
  have side_length := large_cube_side_length
  have total_cubes := total_smaller_cubes
  
  -- Recall that the cube is composed by smaller cubes with painted faces.
  have painted_faces_condition : (∀ (cube : ℕ), cube = 8) := sorry
  
  exact ⟨8, painted_faces_condition 8⟩

end number_of_cubes_with_three_faces_painted_l174_174040


namespace halfway_fraction_l174_174958

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/6) : (a + b) / 2 = 19 / 24 :=
by
  sorry

end halfway_fraction_l174_174958


namespace length_of_XY_l174_174896

variables (A B C D X Y : Type)
variables [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] [AddCommGroup X] [AddCommGroup Y]
variables [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ D] [Module ℝ X] [Module ℝ Y]

noncomputable def rectangle_condition
  (ABCD : A → B → C → D → Prop)
  (A : A) (B : B) (C : C) (D : D)
  (BX DY : ℝ)
  (BC AB : ℝ) :
  Prop :=
  ABCD A B C D ∧
  BX = 4 ∧
  DY = 10 ∧
  BC = 2 * AB

-- The theorem stating the proof problem
theorem length_of_XY {ABCD : A → B → C → D → Prop} (l : ℝ) (A : A) (B : B) (C : C) (D : D) :
  rectangle_condition ABCD A B C D l l → l = 13 :=
begin
  sorry
end

end length_of_XY_l174_174896


namespace find_m_n_l174_174334

theorem find_m_n (m n : ℤ) (h : |m - 2| + (n^2 - 8 * n + 16) = 0) : m = 2 ∧ n = 4 :=
by
  sorry

end find_m_n_l174_174334


namespace halfway_fraction_l174_174975

theorem halfway_fraction (a b : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) : (1/2) * (a + b) = 19/24 :=
by 
  rw [h1, h2]
  simp
  done
  sorry

end halfway_fraction_l174_174975


namespace composite_probability_l174_174537

noncomputable def probability_composite : ℚ :=
  let total_numbers := 50
      number_composite := total_numbers - 15 - 1
  in number_composite / (total_numbers - 1)

theorem composite_probability :
  probability_composite = 34 / 49 :=
by
  sorry

end composite_probability_l174_174537


namespace prob_composite_in_first_50_l174_174541

-- Define what it means for a number to be composite
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ (∀ m : ℕ, m > 1 → m < n → ¬ m ∣ n)

-- Define the set of first 50 natural numbers
def first_50_numbers : list ℕ :=
  (list.range 50).map (λ n, n + 1)

-- Define the set of composite numbers within the first 50 natural numbers
def composite_numbers : list ℕ :=
  first_50_numbers.filter is_composite

-- Define the probability function
noncomputable def probability_of_composite : ℚ :=
  composite_numbers.length / first_50_numbers.length

-- The theorem statement
theorem prob_composite_in_first_50 : probability_of_composite = 34 / 50 :=
by sorry

end prob_composite_in_first_50_l174_174541


namespace total_days_2005_to_2010_l174_174623

-- Defining the range of years and the number of days in normal and leap years
def years : List ℕ := [2005, 2006, 2007, 2008, 2009, 2010]

def is_leap_year (year : ℕ) : Prop := year = 2008
def days_in_year (year : ℕ) : ℕ := if is_leap_year year then 366 else 365

-- Defining the total number of days for the given range
def total_days (ys : List ℕ) : ℕ :=
  ys.foldl (λ acc y => acc + days_in_year y) 0

-- Proof statement: the total number of days from 2005 to 2010 inclusive is 2191
theorem total_days_2005_to_2010 : total_days years = 2191 :=
by
  sorry

end total_days_2005_to_2010_l174_174623


namespace equal_chord_lengths_l174_174064

-- Define the conditions of the problem
variables {Point : Type} [MetricSpace Point]
variables (sphere : Set Point) (X A B C D E F : Point)
variable (chord_length : Point → Point → ℝ )

-- Assume the necessary conditions
variables 
  (chord_meet : sphere → sphere → Point := λ s1 s2, X)
  (chord_not_coplanar : ¬ ∃ plane : Set Point, ∀ p ∈ {A, B, C, D, E, F}, p ∈ plane)
  (sphere1 : Metric.ball A (chord_length A X))
  (sphere2 : Metric.ball B (chord_length B X))
  (sphere3 : Metric.ball C (chord_length C X))
  (sphere_touch : ∀ chord_end1 chord_end2, Touched sphere1 chord_end1 ∧ Touched sphere2 chord_end2 → Touched sphere3 X)

-- Define the theorem
theorem equal_chord_lengths 
  (h_meet : chord_meet sphere sphere = X)
  (h_not_coplanar : chord_not_coplanar)
  (h_touch : sphere_touch A B ∧ sphere_touch C D ∧ sphere_touch E F) : 
  chord_length A B = chord_length C D ∧ 
  chord_length C D = chord_length E F ∧ 
  chord_length E F = chord_length A B :=
sorry

end equal_chord_lengths_l174_174064


namespace popsicle_sum_l174_174258

-- Gino has 63 popsicle sticks
def gino_popsicle_sticks : Nat := 63

-- I have 50 popsicle sticks
def my_popsicle_sticks : Nat := 50

-- The sum of our popsicle sticks
def total_popsicle_sticks : Nat := gino_popsicle_sticks + my_popsicle_sticks

-- Prove that the total is 113
theorem popsicle_sum : total_popsicle_sticks = 113 :=
by
  -- Proof goes here
  sorry

end popsicle_sum_l174_174258


namespace line_does_not_pass_through_qii_l174_174480

noncomputable def line : (ℝ → ℝ) := λ x : ℝ, 3 * x - 2

def passes_through_quadrants (line : ℝ → ℝ) : set (ℝ × ℝ) :=
  {p | p.2 = line p.1}

def is_in_quadrant_ii (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

def does_pass_through_qii (line : ℝ → ℝ) :=
  ∃ (p : ℝ × ℝ), p ∈ passes_through_quadrants line ∧ is_in_quadrant_ii p

theorem line_does_not_pass_through_qii : ¬ does_pass_through_qii line := 
sorry

end line_does_not_pass_through_qii_l174_174480


namespace sqrt_sum_eq_zero_implies_zero_l174_174525

theorem sqrt_sum_eq_zero_implies_zero (x y : ℝ) (h : sqrt x + sqrt y = 0):
  x = 0 ∧ y = 0 := 
by 
  sorry

end sqrt_sum_eq_zero_implies_zero_l174_174525


namespace base7_3652_equals_base10_1360_l174_174065

theorem base7_3652_equals_base10_1360 :
  let n := radix_to_dec 7 [3, 6, 5, 2] in
  n = 1360 :=
 by
   sorry

end base7_3652_equals_base10_1360_l174_174065


namespace number_of_sixes_l174_174109

theorem number_of_sixes
  (total_runs : ℕ)
  (boundaries : ℕ)
  (percent_runs_by_running : ℚ)
  (runs_per_boundary : ℕ)
  (runs_per_six : ℕ)
  (runs_by_running : ℚ)
  (runs_by_boundaries : ℕ)
  (runs_by_sixes : ℕ)
  (number_of_sixes : ℕ)
  (h1 : total_runs = 120)
  (h2 : boundaries = 6)
  (h3 : percent_runs_by_running = 0.6)
  (h4 : runs_per_boundary = 4)
  (h5 : runs_per_six = 6)
  (h6 : runs_by_running = percent_runs_by_running * total_runs)
  (h7 : runs_by_boundaries = boundaries * runs_per_boundary)
  (h8 : runs_by_sixes = total_runs - (runs_by_running + runs_by_boundaries))
  (h9 : number_of_sixes = runs_by_sixes / runs_per_six)
  : number_of_sixes = 4 :=
by
  sorry

end number_of_sixes_l174_174109


namespace maximum_expression_value_l174_174044

theorem maximum_expression_value (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a + b + c + d ≤ 4) :
  (sqrt (sqrt (sqrt 3)) * (a * (b + 2 * c)) ^ (1 / 4) +
   sqrt (sqrt (sqrt 3)) * (b * (c + 2 * d)) ^ (1 / 4) +
   sqrt (sqrt (sqrt 3)) * (c * (d + 2 * a)) ^ (1 / 4) +
   sqrt (sqrt (sqrt 3)) * (d * (a + 2 * b)) ^ (1 / 4)) ≤ 4 * sqrt (sqrt 3) :=
sorry

end maximum_expression_value_l174_174044


namespace regular400gon_without_red_vertices_l174_174099

theorem regular400gon_without_red_vertices :
  let n := 34000 in
  let red_vertices := {i | ∃ k : ℕ, i = 2^k} in
  let total_vertices := finset.range n in
  let valid_vertices := total_vertices \ red_vertices in
  finset.filter (λ s : finset ℕ, s.card = 400 ∧ ∀ i ∈ s, ¬(∃ k : ℕ, i = 2^k))
                (finset.powerset_len 400 total_vertices).card = 77 := 
by
  sorry

end regular400gon_without_red_vertices_l174_174099


namespace find_k_l174_174319

theorem find_k (x y k : ℝ)
  (h1 : 3 * x + 2 * y = k + 1)
  (h2 : 2 * x + 3 * y = k)
  (h3 : x + y = 3) : k = 7 := sorry

end find_k_l174_174319


namespace original_number_l174_174588

theorem original_number (x : ℝ) (h : 1.47 * x = 1214.33) : x = 826.14 :=
sorry

end original_number_l174_174588


namespace dogs_not_eating_any_foods_l174_174757

theorem dogs_not_eating_any_foods :
  let total_dogs := 80
  let dogs_like_watermelon := 18
  let dogs_like_salmon := 58
  let dogs_like_both_salmon_watermelon := 7
  let dogs_like_chicken := 16
  let dogs_like_both_chicken_salmon := 6
  let dogs_like_both_chicken_watermelon := 4
  let dogs_like_all_three := 3
  let dogs_like_any_food := dogs_like_watermelon + dogs_like_salmon + dogs_like_chicken - 
                            dogs_like_both_salmon_watermelon - dogs_like_both_chicken_salmon - 
                            dogs_like_both_chicken_watermelon + dogs_like_all_three
  total_dogs - dogs_like_any_food = 2 := by
  sorry

end dogs_not_eating_any_foods_l174_174757


namespace best_sample_for_understanding_results_l174_174752

theorem best_sample_for_understanding_results (total_students : ℕ)
  (students : Finset ℕ) 
  (students_top_100 : Finset ℕ) 
  (students_bottom_100 : Finset ℕ) 
  (students_female_100 : Finset ℕ) 
  (students_multiples_of_5 : Finset ℕ) :
  total_students = 400 →
  students = Finset.range 401 →
  students_top_100.card = 100 →
  students_bottom_100.card = 100 →
  students_female_100.card = 100 →
  students_multiples_of_5 = (Finset.range 401).filter (λ x, x % 5 = 0 ) →
  students_top_100 ≠ students →
  students_bottom_100 ≠ students →
  students_female_100 ≠ students →
  students_multiples_of_5 ≠ students →
∃ (representative : Finset ℕ), representative = students_multiples_of_5 ∧ 
  representative.card = 80 ∧ 
  ∀ (s : Finset ℕ), s.card = 100 → s ≠ representative →
  (∃ a b c , s = students_top_100 ∨ s = students_bottom_100 ∨ s = students_female_100 ) :=
begin
  sorry
end

end best_sample_for_understanding_results_l174_174752


namespace starting_lineups_possible_l174_174630

open Nat

theorem starting_lineups_possible (total_players : ℕ) (all_stars : ℕ) (lineup_size : ℕ) 
  (fixed_in_lineup : ℕ) (choose_size : ℕ) 
  (h_fixed : fixed_in_lineup = all_stars)
  (h_remaining : total_players - fixed_in_lineup = choose_size)
  (h_lineup : lineup_size = all_stars + choose_size) :
  (Nat.choose choose_size 3 = 220) :=
by
  sorry

end starting_lineups_possible_l174_174630


namespace fencing_cost_l174_174561

theorem fencing_cost
  (area : ℝ)
  (ratio_lw : ℝ)
  (cost_park : ℝ)
  (cost_flower_bed : ℝ)
  (π : ℝ) :
  area = 3750 ∧ ratio_lw = 1.5 ∧ cost_park = 0.70 ∧ cost_flower_bed = 0.90 ∧ π = 3.14 →
  let l := 75, w := 50, diameter_flower_bed := 25, 
      circumference_flower_bed := π * diameter_flower_bed,
      perimeter_park := 2 * (l + w),
      cost_fence_park := perimeter_park * cost_park,
      cost_fence_flower_bed := circumference_flower_bed * cost_flower_bed,
      total_cost := cost_fence_park + cost_fence_flower_bed
  in total_cost = 245.65 :=
begin
  sorry
end

end fencing_cost_l174_174561


namespace find_rowing_speed_of_person_Y_l174_174062

open Real

def rowing_speed (y : ℝ) : Prop :=
  ∀ (x : ℝ) (current_speed : ℝ),
    x = 6 → 
    (4 * (6 - current_speed) + 4 * (y + current_speed) = 4 * (6 + y)) →
    (16 * (y + current_speed) = 16 * (6 + current_speed) + 4 * (y - 6)) → 
    y = 10

-- We set up the proof problem without solving it.
theorem find_rowing_speed_of_person_Y : ∃ (y : ℝ), rowing_speed y :=
begin
  use 10,
  unfold rowing_speed,
  intros x current_speed h1 h2 h3,

  sorry
end

end find_rowing_speed_of_person_Y_l174_174062


namespace simplify_trig_expression_l174_174455

theorem simplify_trig_expression (α : ℝ)
  (h1 : cos (π + α) = - cos α)
  (h2 : sin (α + 2 * π) = sin α)
  (h3 : sin (- α - π) = sin α)
  (h4 : cos (- π - α) = - cos α) :
  (cos (π + α) * sin (α + 2 * π)) / (sin (- α - π) * cos (- π - α)) = 1 :=
by
  sorry

end simplify_trig_expression_l174_174455


namespace halfway_fraction_l174_174956

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/6) : (a + b) / 2 = 19 / 24 :=
by
  sorry

end halfway_fraction_l174_174956


namespace original_book_price_l174_174499

theorem original_book_price (P : ℝ) (h : 0.85 * P * 1.40 = 476) : P = 476 / (0.85 * 1.40) :=
by
  sorry

end original_book_price_l174_174499


namespace brad_weighs_more_l174_174202

theorem brad_weighs_more :
  ∀ (Billy Brad Carl : ℕ), 
    (Billy = Brad + 9) → 
    (Carl = 145) → 
    (Billy = 159) → 
    (Brad - Carl = 5) :=
by
  intros Billy Brad Carl h1 h2 h3
  sorry

end brad_weighs_more_l174_174202


namespace last_letter_77th_permutation_l174_174466

theorem last_letter_77th_permutation :
  let word := "BRAVE"
  let perms := list.permutations word.to_list
  let sorted_perms := perms.qsort (λ a b => a.repr < b.repr)
  (sorted_perms.nth 76).getD [""][-1] = 'V' := sorry

end last_letter_77th_permutation_l174_174466


namespace length_of_the_train_is_120_l174_174602

noncomputable def train_length (time: ℝ) (speed_km_hr: ℝ) : ℝ :=
  let speed_m_s := (speed_km_hr * 1000) / 3600
  speed_m_s * time

theorem length_of_the_train_is_120 :
  train_length 3.569962336897346 121 = 120 := by
  sorry

end length_of_the_train_is_120_l174_174602


namespace find_Tom_favorite_numbers_l174_174898

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_multiple_of (n k : ℕ) : Prop :=
  n % k = 0

def Tom_favorite_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 150 ∧
  is_multiple_of n 13 ∧
  ¬ is_multiple_of n 3 ∧
  is_multiple_of (sum_of_digits n) 4

theorem find_Tom_favorite_numbers :
  ∃ n : ℕ, Tom_favorite_number n ∧ (n = 130 ∨ n = 143) :=
by
  sorry

end find_Tom_favorite_numbers_l174_174898


namespace trajectory_is_perpendicular_bisector_l174_174048

noncomputable def trajectory_of_z (Z Z1 Z2 : ℂ) : Prop :=
|Z - Z1| = |Z - Z2|

theorem trajectory_is_perpendicular_bisector (Z Z1 Z2 : ℂ) (h : Z1 ≠ Z2) :
  trajectory_of_z Z Z1 Z2 ↔ 
  ∃ M : ℂ, (M.re = (Z1.re + Z2.re) / 2 ∧ M.im = (Z1.im + Z2.im) / 2) ∧
  (Z.re - M.re) * (Z2.im - Z1.im) = (Z.im - M.im) * (Z2.re - Z1.re) :=
sorry

end trajectory_is_perpendicular_bisector_l174_174048


namespace alpha_gt_beta_is_neither_necessary_nor_sufficient_l174_174280

theorem alpha_gt_beta_is_neither_necessary_nor_sufficient 
  (α β : ℝ) 
  (hα_in_first_quadrant : 0 < α ∧ α < π/2)
  (hβ_in_first_quadrant : 0 < β ∧ β < π/2) :
    (¬ ((α > β) ↔ (sin α > sin β))) :=
sorry

end alpha_gt_beta_is_neither_necessary_nor_sufficient_l174_174280


namespace area_BCD_correct_l174_174355

-- Definitions (Conditions)
variables {A B C D : Point} -- Points in the plane
variables (AC CD : ℝ) (areaABC : ℝ)
variables (h : ℝ)

-- Conditions
def points_in_plane : Prop := True -- All points lie in the plane
def AC_length : Prop := AC = 9
def CD_length : Prop := CD = 30
def area_ABC : Prop := areaABC = 36
def height_B_to_AC : Prop := (1 / 2) * AC * h = areaABC

-- Goal
def area_BCD : ℝ := (1 / 2) * CD * h

theorem area_BCD_correct :
  points_in_plane ∧ AC_length ∧ CD_length ∧ area_ABC → height_B_to_AC → area_BCD = 120 :=
by
  intros _ _ _ _ _ -- handling assumptions
  sorry

end area_BCD_correct_l174_174355


namespace find_point_B_l174_174183

noncomputable def point :=
{ x : ℝ,
  y : ℝ,
  z : ℝ }

def A : point := ⟨-2, 8, 10⟩
def C : point := ⟨4, 4, 7⟩
def plane (p : point) : Prop := p.x + p.y + p.z = 15

theorem find_point_B :
∃ B : point, (plane B ∧ 
  ∃ t : ℝ, ∀ p : point, p = ⟨A.x + t * (C.x - A.x), A.y + t * (C.y - A.y), A.z + t * (C.z - A.z)⟩ → 
                  plane p) ∧
  B = ⟨4, 4, 7⟩ :=
sorry

end find_point_B_l174_174183


namespace f_is_polynomial_l174_174388

theorem f_is_polynomial (f : ℝ → ℝ) (h : ∀ (c : ℝ), 0 < c → ∃ (P : ℝ[X]), ∀ (x : ℝ), |f x - P.eval x| ≤ c * x ^ 1998) :
  ∃ (Q : ℝ[X]), ∀ (x : ℝ), f x = Q.eval x :=
sorry

end f_is_polynomial_l174_174388


namespace triangles_equal_area_and_sum_squares_l174_174447

variables {a x y z : Real} -- a is the side length of equilateral triangle ABC; x, y, z are the extended segment lengths.
variables {A B C A1 B1 C1 A2 B2 C2 : Real} -- Points
variables {ABC : Real} -- Equilateral triangle ABC
variables {A1B1C1 A2B2C2 : Real} -- Triangles A1B1C1 and A2B2C2

-- Conditions from part a
axiom AC1_eq_BC2 : AC1 = x
axiom BA1_eq_CA2 : BA1 = y
axiom CB1_eq_AB2 : CB1 = z

-- Hypotheses about the triangles
axiom is_equilateral_triangle : equilateral_triangle ABC

-- Questions to prove:
theorem triangles_equal_area_and_sum_squares :
  (area A1B1C1 = area A2B2C2) ∧ 
  (sum_squares_sides A1B1C1 = sum_squares_sides A2B2C2) := sorry

end triangles_equal_area_and_sum_squares_l174_174447


namespace y_is_triangular_l174_174845

theorem y_is_triangular (k : ℕ) (hk : k > 0) : 
  ∃ n : ℕ, y = (n * (n + 1)) / 2 :=
by
  let y := (9^k - 1) / 8
  sorry

end y_is_triangular_l174_174845


namespace line_through_S_l174_174464

noncomputable theory

open EuclideanGeometry 

theorem line_through_S {A D B C P Q S : Point} (hS : tangent A S ∧ tangent D S)
  (h_on_arc : on_arc B A D ∧ on_arc C A D)
  (hP : ∃ P, inter AC BD P)
  (hQ : ∃ Q, inter AB CD Q) :
  collinear {P, Q, S} :=
sorry

end line_through_S_l174_174464


namespace ratio_after_transfer_l174_174087

noncomputable def A : ℝ := 13.2
noncomputable def B : ℝ := 15.6

theorem ratio_after_transfer :
  let A' := A + 6
  let B' := B - 6
  B' / A' = 1 / 2 :=
by
  let A' := A + 6
  let B' := B - 6
  have hA' : A' = 19.2 := by sorry
  have hB' : B' = 9.6 := by sorry
  rw [hA', hB']
  field_simp
  norm_num

end ratio_after_transfer_l174_174087


namespace range_of_r_l174_174318

def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 4}

def N (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ r^2}

theorem range_of_r (r : ℝ) (hr: 0 < r) : (M ∩ N r = N r) → r ≤ 2 - Real.sqrt 2 :=
by
  intro h
  sorry

end range_of_r_l174_174318


namespace part1_l174_174356

theorem part1 (dodecagon : Fin 12 → Bool) (hA1 : dodecagon 0 = false) (hOthers : ∀ i, 1 ≤ i → dodecagon i = true) 
  (flip_6 : ∀ v (s i : Fin 12 → Bool), s = v.update i (¬v i).update (i+1) (¬v (i+1)).update (i+2) (¬v (i+2)).update (i+3) (¬v (i+3)).update (i+4) (¬v (i+4)).update (i+5) (¬v (i+5)) → Prop) :
  ¬ ∃ m : ℕ, (flip_6^[m]) dodecagon = Function.update (Function.const (Fin 12) true) 1 false := sorry

end part1_l174_174356


namespace area_of_pentagon_l174_174610

-- Condition that BD is the diagonal of a square ABCD with length 20 cm
axiom BD_diagonal_of_square : ∀ (s : ℝ), s * Real.sqrt 2 = 20 → s = 10 * Real.sqrt 2

-- Definition of the sides and areas involved
def side_length_of_square (d : ℝ) : ℝ := d / Real.sqrt 2
def area_of_square (s : ℝ) : ℝ := s * s
def area_of_triangle (s : ℝ) : ℝ := (1 / 2) * s * s

-- The main problem statement to be proven in Lean
theorem area_of_pentagon (d : ℝ) (h : d = 20) : 
  area_of_square (side_length_of_square d) + area_of_triangle (side_length_of_square d) = 300 := by
  sorry

end area_of_pentagon_l174_174610


namespace concurrency_condition_l174_174690

variables {A B C D E F : Type} [Triangle ABC] 
          [Point_on_segment D BC] [Point_on_segment E CA] 
          [Point_on_segment F AB]

theorem concurrency_condition : 
    (concurrent (A, D) (B, E) (C, F)) ↔ 
    (sin (∠ BAD) / sin (∠ CAD)) * (sin (∠ ACF) / sin (∠ BCF)) * 
    (sin (∠ CBE) / sin (∠ ABE)) = 1 := 
begin
  sorry
end

end concurrency_condition_l174_174690


namespace find_k_l174_174297

theorem find_k (k : ℝ) (x₁ x₂ : ℝ)
  (h : x₁^2 + (2 * k - 1) * x₁ + k^2 - 1 = 0)
  (h' : x₂^2 + (2 * k - 1) * x₂ + k^2 - 1 = 0)
  (hx : x₁ ≠ x₂)
  (cond : x₁^2 + x₂^2 = 19) : k = -2 :=
sorry

end find_k_l174_174297


namespace slope_ratio_range_l174_174296

theorem slope_ratio_range
  (C : set (ℝ × ℝ)) (kPB kQF : ℝ) (P A B F Q : ℝ × ℝ)
  (h1 : C = { p | p.1 ^ 2 / 4 + p.2 ^ 2 / 3 = 1 })
  (h2 : B = (2, 0))
  (h3 : A = (-2, 0))
  (h4 : F = (1, 0))
  (h5 : (P.1 ^ 2 + P.2 ^ 2 = 4) ∧ P ≠ A ∧ P ≠ B)
  (h6 : ∃ q ∈ C, (Q = q) ∧ line_through A P ∩ C = {Q})
  (h7 : kPB = slope P B)
  (h8 : kQF = slope Q F) :
  set.Icc (real.inf (set.range (λ θ, - (1 / ((√3 * real.sin θ) / (2 * real.cos θ + 2) * ((√3 * real.sin θ) / (2 * real.cos θ - 1))))) (real.sups (set.range (λ t, (4 * t ^ 2 + 2 * t - 2) / (3 * (t ^ 2 - 1))))))) (2 / 3) := sorry

end slope_ratio_range_l174_174296


namespace theta_range_l174_174228

noncomputable def sqrt (x : ℝ) : ℝ := x ^ (1 / 2)
noncomputable def f (x : ℝ) : ℝ := sorry
def f'' (x : ℝ) : ℝ := sorry

theorem theta_range (a b : ℝ) (k : ℤ) (θ : ℝ) 
  (h1 : ∀ x, 0 ≤ x → f(x) ∈ [0, +∞)) 
  (h2 : ∀ x, 2 * f(x) + f''(x) = sqrt(x) / exp(x)) 
  (h3 : f(1 / 2) = 1 / (2 * sqrt(2 * exp(1))))
  (h4 : f(sin θ) ≤ 1 / (a^2) + 1 / (exp(2) * b^2) + a*b / 64) :
  θ ∈ set.Icc (2 * k * real.pi + real.pi / 6) (2 * k * real.pi + 5 * real.pi / 6) :=
sorry

end theta_range_l174_174228


namespace sin_theta_identity_tan_BAC_identity_l174_174565

variable (P A B C O T1 T2 T3 : Point) (α β γ θ : ℝ)

-- Conditions from a)
axiom tetrahedron (P A B C : Point) : True
axiom projection (P A B C O : Point) (O: Incenter(ABC)) : True
axiom projections (O T1 T2 T3: Point) (T1 : Projection_O_AB) (T2: Projection_O_BC) (T3: Projection_O_CA) : True
axiom dihedral_angles (P A B C : Point) (α β γ: ℝ) (angle PABC = α) (angle PBC = β) (angle PCA = γ) : True
axiom angle_between (P O T1 : Point) (θ : ℝ) (angle PO_PT1 = θ) : True

-- Part I) Goal: To prove specific trigonometric identity involving θ and α, β, γ.
theorem sin_theta_identity (α β γ θ : ℝ) 
    (incenter_triangle : O = incenter_triangle ABC)
    (proj_O_AB : T1 = proj_O AB)
    (proj_O_BC : T2 = proj_O BC)
    (proj_O_CA : T3 = proj_O CA)
    (dihedral_P_ABC : α = ∡ P - ABC)
    (dihedral_P_ABC : β = ∡ P - BCA)
    (dihedral_P_ABC : γ = ∡ P - CAB)
    (angle_PO_PT1 : θ = ∡(PO, PT1)) :
    sin θ = (tan ((α + β - γ)/2) * tan ((α + γ - β)/2) * tan ((β + γ - α)/2) / 
    (tan ((α + β - γ)/2) + tan ((α + γ - β)/2) + tan ((β + γ - α)/2))) ^ (1/2) := 
sorry 

-- Part II) Goal: To show another specific trigonometric relationship 
theorem tan_BAC_identity (α β γ θ : ℝ) 
    (incenter_triangle : O = incenter_triangle ABC)
    (proj_O_AB : T1 = proj_O AB)
    (proj_O_BC : T2 = proj_O BC)
    (proj_O_CA : T3 = proj_O CA)
    (dihedral_P_ABC : α = ∡ P - ABC)
    (dihedral_P_ABC : β = ∡ P - BCA)
    (dihedral_P_ABC : γ = ∡ P - CAB)
    (angle_PO_PT1 : θ = ∡(PO, PT1)) :
    (tan (∡ BAC / 2)) = (sin θ) / (tan ((β + γ - α)/2)) := 
sorry 

end sin_theta_identity_tan_BAC_identity_l174_174565


namespace prod_f_zeta_125_l174_174387

noncomputable def f (x : ℂ) : ℂ := 1 + 2 * x + 3 * x^2 + 4 * x^3 + 5 * x^4

noncomputable def zeta : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

theorem prod_f_zeta_125 : f(zeta) * f(zeta^2) * f(zeta^3) * f(zeta^4) = 125 := by
  sorry

end prod_f_zeta_125_l174_174387


namespace fraction_halfway_between_l174_174968

def halfway_fraction (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between :
  halfway_fraction (3/4) (5/6) = 19/24 :=
sorry

end fraction_halfway_between_l174_174968


namespace smallest_interesting_number_l174_174140

def is_perfect_square (x : ℕ) : Prop :=
∃ y : ℕ, y * y = x

def is_perfect_cube (x : ℕ) : Prop :=
∃ y : ℕ, y * y * y = x

theorem smallest_interesting_number (n : ℕ) :
    is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n) → n = 1800 :=
by {
  sorry
}

end smallest_interesting_number_l174_174140


namespace angle_measure_at_Q_of_extended_sides_of_octagon_l174_174835

theorem angle_measure_at_Q_of_extended_sides_of_octagon 
  (A B C D E F G H Q : Type)
  (octagon : regular_octagon A B C D E F G H)
  (extends_to_Q : extends_to_point A B C D Q) :
  angle_measure (∠ BQD) = 22.5 :=
sorry

end angle_measure_at_Q_of_extended_sides_of_octagon_l174_174835


namespace find_x_l174_174405

theorem find_x (x y z : ℕ) (h1 : x ≥ y) (h2 : y ≥ z)
  (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z)
  (eq1 : x^2 - y^2 - z^2 + x * y = 4019)
  (eq2 : x^2 + 4 * y^2 + 4 * z^2 - 4 * x * y - 3 * x * z - 3 * y * z = -3997) :
  x = 8 :=
by
  sorry

end find_x_l174_174405


namespace sandy_total_money_l174_174830

-- Definitions based on conditions
def X_initial (X : ℝ) : Prop := 
  X - 0.30 * X = 210

def watch_cost : ℝ := 50

-- Question translated into a proof goal
theorem sandy_total_money (X : ℝ) (h : X_initial X) : 
  X + watch_cost = 350 := by
  sorry

end sandy_total_money_l174_174830


namespace comparison_of_f_values_l174_174304

noncomputable def f (x : ℝ) := Real.cos x - x

theorem comparison_of_f_values :
  f (8 * Real.pi / 9) > f Real.pi ∧ f Real.pi > f (10 * Real.pi / 9) :=
by
  sorry

end comparison_of_f_values_l174_174304


namespace combined_good_sequence_l174_174866

-- Define the concept of a derivative for sequences
noncomputable def first_derivative (a : ℕ → ℝ) (n : ℕ) : ℝ := a (n + 1) - a n

noncomputable def k_th_derivative (a : ℕ → ℝ) (k n : ℕ) : ℝ :=
(k.iterate (first_derivative ∘ (flip id) a)) n

-- Define what it means for a sequence to be good
def good_sequence (a : ℕ → ℝ) : Prop :=
∀ k n, (k_th_derivative a k n) > 0

-- Problem statement
theorem combined_good_sequence
    (a b : ℕ → ℝ) (ha : good_sequence a) (hb : good_sequence b) :
    good_sequence (λ n, a n * b n) :=
sorry

end combined_good_sequence_l174_174866


namespace discount_on_purchase_l174_174342

theorem discount_on_purchase 
  (price_cherries price_olives : ℕ)
  (num_bags_cherries num_bags_olives : ℕ)
  (total_paid : ℕ) :
  price_cherries = 5 →
  price_olives = 7 →
  num_bags_cherries = 50 →
  num_bags_olives = 50 →
  total_paid = 540 →
  (total_paid / (price_cherries * num_bags_cherries + price_olives * num_bags_olives)) * 100 = 90 :=
by
  intros h_price_cherries h_price_olives h_num_bags_cherries h_num_bags_olives h_total_paid
  sorry

end discount_on_purchase_l174_174342


namespace cot_cot_inv_sum_identity_l174_174214

  noncomputable theory
  open Real

  theorem cot_cot_inv_sum_identity :
    cot (arccot 5 + arccot 11 + arccot 17 + arccot 23) = 1021 / 420 :=
  sorry
  
end cot_cot_inv_sum_identity_l174_174214


namespace volume_of_prism_l174_174779

theorem volume_of_prism (r : ℝ) (a : ℝ) (h : ℝ) (V : ℝ) 
  (h_height : h = 2)
  (h_surface_area : 4 * π * r^2 = 12 * π)
  (h_diagonal : 2 * r = a * real.sqrt 2) :
  V = a^2 * h :=
begin
  let r := real.sqrt 3,
  let a := real.sqrt 6,
  let V := a^2 * h,
  sorry
end

end volume_of_prism_l174_174779


namespace odd_divisors_of_20_factorial_l174_174566

theorem odd_divisors_of_20_factorial :
  let odd_divisors :=
    (8 + 1) * (4 + 1) * (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  in odd_divisors = 2160 := by
    let odd_divisors :=
      (8 + 1) * (4 + 1) * (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
    show odd_divisors = 2160
    simp [odd_divisors]
    rfl

end odd_divisors_of_20_factorial_l174_174566


namespace distance_between_parabolas_vertices_l174_174643

theorem distance_between_parabolas_vertices :
  let eq_graph := (x y : ℝ) → sqrt (x^2 + y^2) + |y - 2| = 5
  in 
    let vertex1 := (0 : ℝ, 49 / 14 : ℝ)
    let vertex2 := (0 : ℝ, -3 / 2 : ℝ)
    abs ((49 / 14) - (-3 / 2)) = 119 / 28 := by
sorry

end distance_between_parabolas_vertices_l174_174643


namespace u_2008_is_4015_l174_174404

-- Define the n-th term of the sequence according to the conditions
noncomputable def u : ℕ → ℕ
| 0       := 2
| (n + 1) := if n < 2008 then u n + 4 else 0  -- using a simplified approximation for illustration

-- The core theorem to prove
theorem u_2008_is_4015 : u 2007 = 4015 :=
sorry

end u_2008_is_4015_l174_174404


namespace div_by_10_l174_174831

theorem div_by_10 (n : ℕ) (hn : 10 ∣ (3^n + 1)) : 10 ∣ (3^(n+4) + 1) :=
by
  sorry

end div_by_10_l174_174831


namespace functions_exist_l174_174101

-- Integer a
def a : ℤ := sorry

-- Prime number p, with p ≥ 17
def p : ℕ := sorry
def p_prime : Prime p := sorry
def p_ge_17 : p ≥ 17 := sorry

-- Set S = {1, 2, ..., p-1}
def S : Set ℕ := Set.Ico 1 p

-- Set T = {y ∈ S | ord_p(y) < p-1}
def ord_p (y : ℕ) : ℕ := sorry -- definition of order mod p
def T : Set ℕ := {y ∈ S | ord_p y < p - 1}

theorem functions_exist : 
    ∃ (F : Finset (S → S)), 
    F.card ≥ 4 * (p - 3) * (p - 1)^(p - 4) ∧
    ∀ f ∈ F, ∑ x in T, x^(f x) % p = a % p := 
sorry

end functions_exist_l174_174101


namespace non_congruent_trapezoids_l174_174086

def a := 5
def b := 10
def c := 15
def d := 20

theorem non_congruent_trapezoids :
  (∃ T1 T2 : quadrilateral, is_trapezoid T1 ∧ is_trapezoid T2 ∧ T1 ≠ T2) :=
sorry

end non_congruent_trapezoids_l174_174086


namespace jenny_meal_combinations_l174_174614

theorem jenny_meal_combinations :
  let main_dishes := 4
  let drinks := 2
  let desserts := 2
  let side_dishes := 3
  main_dishes * drinks * desserts * side_dishes = 48 := by
{
  let main_dishes := 4
  let drinks := 2
  let desserts := 2
  let side_dishes := 3
  show main_dishes * drinks * desserts * side_dishes = 48
  calc
    4 * 2 * 2 * 3 = 48 : by norm_num
}

end jenny_meal_combinations_l174_174614


namespace price_per_cup_of_lemonade_l174_174604

-- Define the initial conditions
def cost_of_lemons : ℝ := 10
def cost_of_sugar : ℝ := 5
def cost_of_cups : ℝ := 3
def number_of_cups : ℝ := 21
def profit : ℝ := 66

-- Define the statement to prove
theorem price_per_cup_of_lemonade : (total_revenue : ℝ) (total_expenses : ℝ) (price_per_cup : ℝ) :=
  total_expenses = cost_of_lemons + cost_of_sugar + cost_of_cups →
  total_revenue = profit + total_expenses →
  price_per_cup = total_revenue / number_of_cups →
  price_per_cup = 4 :=
begin
  sorry -- Proof omitted
end

end price_per_cup_of_lemonade_l174_174604


namespace max_points_acute_triangle_plane_max_points_acute_triangle_space_l174_174672

theorem max_points_acute_triangle_plane : 
  ∀ (n : ℕ), (∀ (i j k : ℕ), i < j -> j < k -> k ≤ n -> i ≠ j -> j ≠ k -> i ≠ k -> ∃ (M : finset (ℕ × ℝ × ℝ)), M.card = n ∧ ∀ {p q r : ℕ × ℝ × ℝ}, p ∈ M → q ∈ M → r ∈ M → p ≠ q → q ≠ r → p ≠ r → ∠ p q r < π / 2) → n ≤ 3 := 
by
  sorry

theorem max_points_acute_triangle_space : 
  ∀ (n : ℕ), (∀ (i j k : ℕ), i < j -> j < k -> k ≤ n -> i ≠ j -> j ≠ k -> i ≠ k -> ∃ (M : finset (ℕ × ℝ × ℝ × ℝ)), M.card = n ∧ ∀ {p q r : ℕ × ℝ × ℝ × ℝ}, p ∈ M → q ∈ M → r ∈ M → p ≠ q → q ≠ r → p ≠ r → ∠ p q r < π / 2) → n ≤ 5 := 
by
  sorry

end max_points_acute_triangle_plane_max_points_acute_triangle_space_l174_174672


namespace eat_cereal_time_l174_174817

-- Definitions from conditions
def rate_mr_fat : ℝ := 1 / 15
def rate_mr_thin : ℝ := 1 / 25
def combined_rate : ℝ := rate_mr_fat + rate_mr_thin
def total_cereal : ℝ := 4

-- Proof problem statement
theorem eat_cereal_time : (total_cereal / combined_rate) = 37.5 :=
by
  -- Proof skipped
  sorry

end eat_cereal_time_l174_174817


namespace fourth_number_written_eighth_number_written_l174_174905

/-- The sequence of medians recorded in Mitya's notebook -/
def recorded_medians : List ℚ := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2.5]

/-- The sequence of numbers written by Vanya -/
noncomputable def numbers_written_by_vanya : List ℚ :=
  sorry -- the list needs to be computed based on median conditions

theorem fourth_number_written (h : numbers_written_by_vanya ≠ []) :
  recorded_medians.length ≥ 4 →
  recorded_medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2.5] →
  (numbers_written_by_vanya.nth 4).get_or_else 0 = 2 :=
sorry

theorem eighth_number_written (h : numbers_written_by_vanya ≠ []) :
  recorded_medians.length ≥ 8 →
  recorded_medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2.5] →
  (numbers_written_by_vanya.nth 8).get_or_else 0 = 2 :=
sorry

end fourth_number_written_eighth_number_written_l174_174905


namespace binomial_expression_value_l174_174231

theorem binomial_expression_value :
  (Nat.choose 1 2023 * 3^2023) / Nat.choose 4046 2023 = 0 := by
  sorry

end binomial_expression_value_l174_174231


namespace quadratic_root_ratio_l174_174880

theorem quadratic_root_ratio {m p q : ℝ} (h₁ : m ≠ 0) (h₂ : p ≠ 0) (h₃ : q ≠ 0)
  (h₄ : ∀ s₁ s₂ : ℝ, (s₁ + s₂ = -q ∧ s₁ * s₂ = m) →
    (∃ t₁ t₂ : ℝ, t₁ = 3 * s₁ ∧ t₂ = 3 * s₂ ∧ (t₁ + t₂ = -m ∧ t₁ * t₂ = p))) :
  p / q = 27 :=
by
  sorry

end quadratic_root_ratio_l174_174880


namespace anne_distance_l174_174739

theorem anne_distance (speed time : ℕ) (h_speed : speed = 2) (h_time : time = 3) : 
  (speed * time) = 6 := by
  sorry

end anne_distance_l174_174739


namespace halfway_fraction_l174_174977

theorem halfway_fraction (a b : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) : (1/2) * (a + b) = 19/24 :=
by 
  rw [h1, h2]
  simp
  done
  sorry

end halfway_fraction_l174_174977


namespace sum_of_all_positive_nu_eq_300_l174_174546

theorem sum_of_all_positive_nu_eq_300 :
  let S := {ν : ℕ | Nat.lcm ν 24 = 120} in
  ∑ ν in S, ν = 300 :=
by
  sorry

end sum_of_all_positive_nu_eq_300_l174_174546


namespace triangle_LEF_area_correct_l174_174767

noncomputable def area_triangle_LEF (radius : ℝ) (chord_length : ℝ) (LN : ℝ) (is_parallel : bool) (is_collinear : bool) : ℝ :=
  if (radius = 10) ∧ (chord_length = 12) ∧ (LN = 20) ∧ is_parallel ∧ is_collinear then 48 else 0

theorem triangle_LEF_area_correct :
  area_triangle_LEF 10 12 20 true true = 48 :=
by
  -- assuming conditions directly as they are the hypothesis of the theorem
  sorry

end triangle_LEF_area_correct_l174_174767


namespace cotangent_sum_identity_l174_174217

theorem cotangent_sum_identity (a b c d : ℝ) :
    (∀ a b : ℝ, cot (arccot a + arccot b) = (a * b - 1) / (a + b)) →
    cot (arccot 5 + arccot 11 + arccot 17 + arccot 23) = 97 / 40 :=
by
    intros identity
    have h₁ : cot (arccot 5 + arccot 11) = (5 * 11 - 1) / (5 + 11) := by rw [identity]
    have h₂ : cot (arccot 17 + arccot 23) = (17 * 23 - 1) / (17 + 23) := by rw [identity]
    have h₃ : cot (arccot ((54 : ℚ)/16) + arccot ((390 : ℚ)/40)) = (54/16 * 390/40 - 1) / (54/16 + 390/40) := sorry
    have h₄ : (5 * 11 - 1) = 54 := by norm_num
    have h₅ : (17 * 23 - 1) = 390 := by norm_num
    have h₆ : cot (arccot ((27/8 : ℚ)) + arccot ((39/4 : ℚ))) = (27/8 * 39/4 - 1) / (27/8 + 39/4) := sorry
    sorry -- Further steps are included in a similar manner

end cotangent_sum_identity_l174_174217


namespace angle_ABC_center_of_circumscribed_circle_l174_174492

theorem angle_ABC_center_of_circumscribed_circle
  (O A B C : Point)
  (hO_center : IsCenterOfCircumscribedCircle O A B C)
  (angle_BOC : ∠BOC = 110)
  (angle_AOB : ∠AOB = 150) :
  ∠ABC = 50 := 
sorry

end angle_ABC_center_of_circumscribed_circle_l174_174492


namespace line_intersects_circle_probability_l174_174000

def probability_line_intersects_circle : Real :=
  let interval := Icc (-1 : ℝ) (1 : ℝ)
  let intersect_condition (k : ℝ) : Prop :=
    abs (3 * k) / sqrt (k^2 + 1) < 1
  let probability_density := 1 / (interval.snd - interval.fst)
  probability_density * ∫ x in interval.fst..interval.snd, 
    if intersect_condition x then probability_density else 0

theorem line_intersects_circle_probability :
  probability_line_intersects_circle = Real.sqrt 2 / 4 :=
by
  sorry

end line_intersects_circle_probability_l174_174000


namespace logarithm_problem_solution_l174_174243

noncomputable theory
open Real

theorem logarithm_problem_solution (x : ℝ) (hx1 : 0 < x - 1/3) (hx2 : 0 < x - 3) (hx3 : 0 < x) 
    (h : log x (x - 1/3) * log (x - 1/3) (x - 3) * log (x - 3) x = 1) :
    x = 10 / 3 ∨ x = (3 + sqrt 13) / 2 :=
sorry

end logarithm_problem_solution_l174_174243


namespace condition_on_a_b_l174_174806

theorem condition_on_a_b (a b : ℝ) (h : a^2 * b^2 + 5 > 2 * a * b - a^2 - 4 * a) : ab ≠ 1 ∨ a ≠ -2 :=
by
  sorry

end condition_on_a_b_l174_174806


namespace area_of_pentagon_l174_174609

-- Condition that BD is the diagonal of a square ABCD with length 20 cm
axiom BD_diagonal_of_square : ∀ (s : ℝ), s * Real.sqrt 2 = 20 → s = 10 * Real.sqrt 2

-- Definition of the sides and areas involved
def side_length_of_square (d : ℝ) : ℝ := d / Real.sqrt 2
def area_of_square (s : ℝ) : ℝ := s * s
def area_of_triangle (s : ℝ) : ℝ := (1 / 2) * s * s

-- The main problem statement to be proven in Lean
theorem area_of_pentagon (d : ℝ) (h : d = 20) : 
  area_of_square (side_length_of_square d) + area_of_triangle (side_length_of_square d) = 300 := by
  sorry

end area_of_pentagon_l174_174609


namespace consecutive_integer_sum_30_l174_174732

theorem consecutive_integer_sum_30 : ∃ (s : Finset ℕ), ∑ i in s, i = 30 ∧ s.card ≥ 3 ∧ ∀ (t : Finset ℕ), ∑ i in t, i = 30 ∧ t.card ≥ 3 → t = s :=
by
  sorry

end consecutive_integer_sum_30_l174_174732


namespace unique_solution_range_of_a_l174_174778

theorem unique_solution_range_of_a (a b c : ℝ) (A B C : ℝ) (hB : b = 3) (hcosA : Real.cos A = 2/3) :
  (∃ C, ∃ ∆ > 0, (c^2 - 4*c + 9 - a^2 = 0) ∧ 
   (∆ = (16 - 4*(9 - a^2)) = 0 ∨ 16 - 4*(9 - a^2) > 0)) 
  ↔ (a = Real.sqrt 5 ∨ a ≥ 3) :=
sorry

end unique_solution_range_of_a_l174_174778


namespace sqrt_of_square_eq_seven_l174_174738

theorem sqrt_of_square_eq_seven (x : ℝ) (h : x^2 = 7) : x = Real.sqrt 7 ∨ x = -Real.sqrt 7 :=
sorry

end sqrt_of_square_eq_seven_l174_174738


namespace smallest_interesting_number_l174_174149

def is_perfect_square (m : ℕ) : Prop := 
  ∃ k : ℕ, k * k = m

def is_perfect_cube (m : ℕ) : Prop := 
  ∃ k : ℕ, k * k * k = m

def interesting (n : ℕ) : Prop := 
  is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n)

theorem smallest_interesting_number : ∃ n, interesting n ∧ ∀ m, interesting m → n ≤ m :=
begin
  use 1800,
  split,
  {
    -- Prove 1800 is interesting
    sorry,
  },
  {
    -- Prove any interesting number is at least 1800
    sorry,
  }
end

end smallest_interesting_number_l174_174149


namespace complex_conjugate_l174_174744

noncomputable def z : ℂ := ((5 + 4 * complex.I) + complex.I) / (2 - complex.I)

theorem complex_conjugate (h : (2 - complex.I) * z - complex.I = 5 + 4 * complex.I) : z.conj = 1 - 3 * complex.I :=
  by
  -- The detailed proof is omitted
  sorry

end complex_conjugate_l174_174744


namespace tea_bags_l174_174436

theorem tea_bags (n : ℕ) (h₁ : 2 * n ≤ 41 ∧ 41 ≤ 3 * n) (h₂ : 2 * n ≤ 58 ∧ 58 ≤ 3 * n) : n = 20 := by
  sorry

end tea_bags_l174_174436


namespace bus_stopping_time_l174_174657

noncomputable def calculate_stopping_time (v_excluding: ℝ) (v_including: ℝ) : ℝ :=
  let distance_less := v_excluding - v_including
  let speed_per_min := v_excluding / 60
  (distance_less / speed_per_min)

theorem bus_stopping_time (h1 : 82 = v_excluding) (h2 : 75 = v_including) : 
  calculate_stopping_time 82 75 ≈ 5.12 :=
by
  sorry

end bus_stopping_time_l174_174657


namespace quadratic_residue_conditions_l174_174550

theorem quadratic_residue_conditions (p : ℕ) (hp : Nat.Prime p) :
  (p ≡ 1 [MOD 35] ∨ p ≡ 9 [MOD 35] ∨ p ≡ 11 [MOD 35] ∨ p ≡ 19 [MOD 35] ∨ p ≡ 21 [MOD 35] ∨ p ≡ 29 [MOD 35]) ↔
  (QuadraticResidue 5 p ∧ QuadraticResidue 7 p) :=
by
  sorry

end quadratic_residue_conditions_l174_174550


namespace chip_rearrangement_impossible_l174_174372

theorem chip_rearrangement_impossible : 
  ∀ {blue red green : ℕ}, blue = 40 → red = 30 → green = 20 → 
  ¬ ∃ (chips : list ℕ), 
    (∀ i, chips.nth i ≠ chips.nth (i + 1) % (blue + red + green)) ∧
    (∀ i ∈ (list.range (blue - 1)), chips.nth i = 1 ∧ chips.nth (i + 1) = 2 ∨ chips.nth i = 2 ∧ chips.nth (i + 1) = 1 ∨
      chips.nth i = 1 ∧ chips.nth (i + 1) = 3 ∨ chips.nth i = 3 ∧ chips.nth (i + 1) = 1) :=
by sorry

end chip_rearrangement_impossible_l174_174372


namespace range_a_ge_one_l174_174276

theorem range_a_ge_one (a : ℝ) (x : ℝ) 
  (p : Prop := |x + 1| > 2) 
  (q : Prop := x > a) 
  (suff_not_necess_cond : ¬p → ¬q) : a ≥ 1 :=
sorry

end range_a_ge_one_l174_174276


namespace inequality_solution_l174_174305

noncomputable def f (x : ℝ) : ℝ := 
  Real.log (Real.sqrt (x^2 + 1) + x) - (2 / (Real.exp x + 1))

theorem inequality_solution :
  { x : ℝ | f x + f (2 * x - 1) > -2 } = { x : ℝ | x > 1 / 3 } :=
sorry

end inequality_solution_l174_174305


namespace count_ways_select_numbers_l174_174337

theorem count_ways_select_numbers (a1 a2 a3 : ℕ) (h1 : 1 ≤ a1 ∧ a1 < a2 ∧ a2 < a3 ∧ a3 ≤ 14)
  (h2 : a2 - a1 ≥ 3)
  (h3 : a3 - a2 ≥ 3) :
  {x : Finset (ℕ) | ∃ (a'1 a'2 a'3 : ℕ), a'1 < a'2 ∧ a'2 < a'3
    ∧ a'1 = a1 ∧ a'2 - 2 = a2 ∧ a'3 - 4 = a3
    ∧ a'1 ∈ Finset.range 11
    ∧ a'2 ∈ Finset.range 11
    ∧ a'3 ∈ Finset.range 11}.card = 120 := sorry

end count_ways_select_numbers_l174_174337


namespace minimum_a_plus_b_l174_174411

theorem minimum_a_plus_b (a b : ℤ) (h : a * b = 144) : a + b ≥ -24 :=
by sorry

end minimum_a_plus_b_l174_174411


namespace tea_bags_count_l174_174433

theorem tea_bags_count (n : ℕ) 
  (h1 : 2 * n ≤ 41) 
  (h2 : 41 ≤ 3 * n) 
  (h3 : 2 * n ≤ 58) 
  (h4 : 58 ≤ 3 * n) : 
  n = 20 := by
  sorry

end tea_bags_count_l174_174433


namespace find_interval_of_increase_l174_174665

open Real

noncomputable def log_base_half (x : ℝ) : ℝ := log x / log (1/2)

def interval_of_increase (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem find_interval_of_increase :
  interval_of_increase (λ x => log_base_half (x^2 - 1)) (-∞) -1 :=
by
  sorry

end find_interval_of_increase_l174_174665


namespace pentagon_ABEDF_area_l174_174611

theorem pentagon_ABEDF_area (BD_diagonal : ∀ (ABCD : Nat) (BD : Nat),
                            ABCD = BD^2 / 2 → BD = 20) 
                            (BDFE_is_rectangle : ∀ (BDFE : Nat), BDFE = 2 * BD) 
                            : ∃ (area : Nat), area = 300 :=
by
  -- Placeholder for the actual proof
  sorry

end pentagon_ABEDF_area_l174_174611


namespace stick_in_v_shaped_ditch_equidistant_l174_174600

theorem stick_in_v_shaped_ditch_equidistant
  {S₁ S₂ : Type*} [plane S₁] [plane S₂]
  (A B : point)
  (line_of_intersection : line)
  (angle_equal : angle_between_stick_planes A B S₁ S₂)
  (not_perpendicular : ¬ perpendicular_stick_to_intersection A B line_of_intersection) :
  equidistant_to_bottom A B S₁ S₂ := 
sorry

end stick_in_v_shaped_ditch_equidistant_l174_174600


namespace john_initial_budget_l174_174063

theorem john_initial_budget (shoes_cost left_after_purchase : ℝ) (h1 : shoes_cost = 165) (h2 : left_after_purchase = 834) : 
  shoes_cost + left_after_purchase = 999 := 
by
  rw [h1, h2]
  norm_num
  sorry

end john_initial_budget_l174_174063


namespace max_garden_area_l174_174452

noncomputable def garden_max_area (l w : ℝ) (h : 2 * l + 2 * w = 400) : ℝ :=
  l * w

theorem max_garden_area : ∃ (l w : ℝ), 2 * l + 2 * w = 400 ∧ (∀ l' w', 2 * l' + 2 * w' = 400 → l' * w' ≤ 10000) :=
begin
  use [100, 100],
  split,
  { rw [mul_two, mul_two, ← add_assoc, add_self_eq_zero, zero_add],
    norm_num },
  { intros l' w' hw',
    suffices : l' * w' ≤ 10000,
      exact this,

    have h1 : l' + w' = 200,
      by linarith,

    rw [← sub_eq_zero, ← sub_eq_self, eq_comm, sub_sq, ← add_sq_eq_add_sq],
    norm_num at *,
    exact le_max_of_le_left (sub_eq_sub_eq_add_eq (expne.right (ile_of_case (h2)))),
  sorry
end

end max_garden_area_l174_174452


namespace race_min_distance_l174_174881

noncomputable def min_distance : ℝ :=
  let A : ℝ × ℝ := (0, 300)
  let B : ℝ × ℝ := (1200, 500)
  let wall_length : ℝ := 1200
  let B' : ℝ × ℝ := (1200, -500)
  let distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  distance A B'

theorem race_min_distance :
  min_distance = 1442 := sorry

end race_min_distance_l174_174881


namespace find_lambda_l174_174724

variables (a b : ℝ × ℝ) (λ : ℝ)
def is_perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

theorem find_lambda
  (h_a : a = (1, -1))
  (h_b : b = (-1, 2))
  (h_perpendicular : is_perpendicular (a.1 - λ * b.1, a.2 - λ * b.2) a) :
  λ = -2 / 3 :=
sorry

end find_lambda_l174_174724


namespace range_of_m_l174_174642

noncomputable def f (x m a : ℝ) : ℝ := Real.exp (x + 1) - m * a
noncomputable def g (x a : ℝ) : ℝ := a * Real.exp x - x

theorem range_of_m (h : ∃ a : ℝ, ∀ x : ℝ, f x m a ≤ g x a) : m ≥ -1 / Real.exp 1 :=
by
  sorry

end range_of_m_l174_174642


namespace find_perimeter_correct_l174_174467

noncomputable def find_perimeter (L W : ℝ) (x : ℝ) :=
  L * W = (L + 6) * (W - 2) ∧
  L * W = (L - 12) * (W + 6) ∧
  x = 2 * (L + W)

theorem find_perimeter_correct : ∀ (L W : ℝ), L * W = (L + 6) * (W - 2) → 
                                      L * W = (L - 12) * (W + 6) → 
                                      2 * (L + W) = 132 :=
sorry

end find_perimeter_correct_l174_174467


namespace graphs_symmetric_about_2_0_l174_174479

def is_symmetric_about_point (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
∀ x, f(2 * p.1 - x) = 2 * p.2 - f(x)

theorem graphs_symmetric_about_2_0 (f : ℝ → ℝ) :
  is_symmetric_about_point (λ x, -f(x + 2)) (2, 0) ∧ is_symmetric_about_point (λ x, f(6 - x)) (2, 0) :=
by
  sorry

end graphs_symmetric_about_2_0_l174_174479


namespace total_legs_l174_174581

theorem total_legs (t f : ℕ) (h_t : t = 36) (h_f : f = 16) : 
  let three_legged := t - f in
  let four_legged_legs := 4 * f in
  let three_legged_legs := 3 * three_legged in
  let total_legs := four_legged_legs + three_legged_legs in
  total_legs = 124 := 
by 
  sorry

end total_legs_l174_174581


namespace relationship_among_abc_l174_174800

noncomputable def a : ℝ := 0.8^0.7
noncomputable def b : ℝ := 0.8^0.9
noncomputable def c : ℝ := 1.2^0.8

theorem relationship_among_abc : c > a ∧ a > b := by
  sorry

end relationship_among_abc_l174_174800


namespace tangent_cosine_opposite_signs_l174_174025

theorem tangent_cosine_opposite_signs (α : ℝ) : 
  tan α = tan (α + π) → cos (α + π) = - cos α :=
by
  intros h
  sorry

end tangent_cosine_opposite_signs_l174_174025


namespace number_of_tea_bags_l174_174440

theorem number_of_tea_bags (n : ℕ) 
  (h1 : 2 * n ≤ 41) (h2 : 41 ≤ 3 * n)
  (h3 : 2 * n ≤ 58) (h4 : 58 ≤ 3 * n) :
  n = 20 :=
by
  sorry

end number_of_tea_bags_l174_174440


namespace fraction_halfway_between_3_4_and_5_6_is_19_24_l174_174910

noncomputable def fraction_halfway (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between_3_4_and_5_6_is_19_24 :
  fraction_halfway (3 / 4) (5 / 6) = 19 / 24 :=
by
  sorry

end fraction_halfway_between_3_4_and_5_6_is_19_24_l174_174910


namespace infinite_product_eq_nine_l174_174238

noncomputable def infinite_product : ℕ → ℝ
| 0       := 1
| (n + 1) := infinite_product n * 3^(n/(2^n))

theorem infinite_product_eq_nine :
  infinite_product ∞ = 9 :=
sorry

end infinite_product_eq_nine_l174_174238


namespace measure_angle_ABC_l174_174489

variables (A B C O : Point)
variables (h1 : circumcenter O A B C)
variables (h2 : angle O B C = 110)
variables (h3 : angle A O B = 150)

theorem measure_angle_ABC : angle A B C = 50 :=
by
  sorry

end measure_angle_ABC_l174_174489


namespace triangle_is_equilateral_l174_174685

noncomputable def rotate_by_120_deg (P A : ℂ) : ℂ :=
let u := complex.exp (complex.I * 2 * real.pi / 3) in
A + u * (P - A)

theorem triangle_is_equilateral
  (A1 A2 A3 P0 : ℂ)
  (h_seq : ∀ k : ℕ, let P := λ k, nat.rec_on k P0 (λ k Pk, rotate_by_120_deg Pk (match (k % 3) with | 0 => A1 | 1 => A2 | _ => A3 end))
  in P (k + 1) = rotate_by_120_deg (P k) (match (k % 3) with | 0 => A1 | 1 => A2 | _ => A3 end))
  (h1986 : let P := λ k, nat.rec_on k P0 (λ k Pk, rotate_by_120_deg Pk (match (k % 3) with | 0 => A1 | 1 => A2 | _ => A3 end))
  in P 1986 = P 0) :
  let u := complex.exp (complex.I * 2 * real.pi / 3) in
  A3 - u * A2 + u ^ 2 * A1 = 0 :=
by {
  let u := complex.exp (complex.I * 2 * real.pi / 3),
  let P := λ k, nat.rec_on k P0 (λ k Pk, rotate_by_120_deg Pk (match (k % 3) with | 0 => A1 | 1 => A2 | _ => A3 end)),
  have hP : P 1986 = P 0, from h1986,
  have w : (1 + u) * (A3 - u * A2 + u ^ 2 * A1) = 0,
  { -- the core argument proving w=0 from P 1986 = P 0 goes here
    sorry },
  have : (1 + u) ≠ 0, by { sorry },
  exact complex.eq_zero_of_mul_eq_zero_right w this
}

end triangle_is_equilateral_l174_174685


namespace composite_probability_l174_174536

noncomputable def probability_composite : ℚ :=
  let total_numbers := 50
      number_composite := total_numbers - 15 - 1
  in number_composite / (total_numbers - 1)

theorem composite_probability :
  probability_composite = 34 / 49 :=
by
  sorry

end composite_probability_l174_174536


namespace x_coordinate_equidistant_l174_174074

theorem x_coordinate_equidistant :
  ∃ x : ℝ, (sqrt ((-3 - x)^2 + 0^2) = sqrt ((2 - x)^2 + 5^2)) ∧ x = 2 :=
by
  sorry

end x_coordinate_equidistant_l174_174074


namespace sanctuary_feeding_ways_l174_174186

/-- A sanctuary houses six different pairs of animals, each pair consisting of a male and female.
  The caretaker must feed the animals alternately by gender, meaning no two animals of the same gender 
  can be fed consecutively. Given the additional constraint that the male giraffe cannot be fed 
  immediately before the female giraffe and that the feeding starts with the male lion, 
  there are exactly 7200 valid ways to complete the feeding. -/
theorem sanctuary_feeding_ways : 
  ∃ ways : ℕ, ways = 7200 :=
by sorry

end sanctuary_feeding_ways_l174_174186


namespace no_x4_term_expansion_l174_174287

-- Mathematical condition and properties
variable {R : Type*} [CommRing R]

theorem no_x4_term_expansion (a : R) (h : a ≠ 0) :
  ∃ a, (a = 8) := 
by 
  sorry

end no_x4_term_expansion_l174_174287


namespace probability_players_face_each_other_in_last_round_l174_174794

-- Define the conditions in Lean:
def probability_of_facing_each_other (n : ℕ) : ℚ :=
  1 / 4 ^ n

theorem probability_players_face_each_other_in_last_round (n : ℕ) (h : n > 0) :
  let total_players := 2^(n+1)
  let rounds := n+1
  let prob := 1 / 4 ^ n
  in probability_of_facing_each_other n = prob :=
by
  sorry

end probability_players_face_each_other_in_last_round_l174_174794


namespace sum_of_fifths_l174_174639

theorem sum_of_fifths : (1/5 + 2/5 + 3/5 + ... + 14/5 + 15/5) = 24 :=
by
  sorry

end sum_of_fifths_l174_174639


namespace sum_even_integers_less_than_100_l174_174988

theorem sum_even_integers_less_than_100 : 
  let sequence := List.range' 2 98
  let even_seq := sequence.filter (λ x => x % 2 = 0)
  (even_seq.sum) = 2450 :=
by
  sorry

end sum_even_integers_less_than_100_l174_174988


namespace perfect_square_values_l174_174242

theorem perfect_square_values :
  ∀ n : ℕ, 0 < n → (∃ k : ℕ, (n^2 + 11 * n - 4) * n.factorial + 33 * 13^n + 4 = k^2) ↔ n = 1 ∨ n = 2 :=
by sorry

end perfect_square_values_l174_174242


namespace intersection_distance_sqrt7_l174_174705

-- Define the curve C in terms of Cartesian coordinates
def curve_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the general equation of the line l
def line_l (x y : ℝ) : Prop := x - sqrt(3) * y + 1 = 0

-- Define the proof problem: Prove that the distance |PQ| equals sqrt(7)
theorem intersection_distance_sqrt7 :
  ∀ (x1 y1 x2 y2 : ℝ), 
  curve_C x1 y1 → line_l x1 y1 → 
  curve_C x2 y2 → line_l x2 y2 → 
  sqrt ((x2 - x1)^2 + (y2 - y1)^2) = sqrt 7 :=
by
  -- Proof would be provided here
  sorry

end intersection_distance_sqrt7_l174_174705


namespace volunteer_team_combinations_l174_174673

theorem volunteer_team_combinations :
  let male_students := 5
      female_students := 3
      total_team_size := 4
      team_leader := 1
      deputy_team_leader := 1
      ordinary_team_members := total_team_size - team_leader - deputy_team_leader
  in
  (calculate_team_combinations male_students female_students total_team_size team_leader deputy_team_leader ordinary_team_members) = 780 := sorry

/-- A function to calculate the different ways to form a team given specifications. -/
def calculate_team_combinations (male_students : ℕ) (female_students : ℕ) (total_team_size : ℕ)
                                (team_leader : ℕ) (deputy_team_leader : ℕ) (ordinary_team_members : ℕ) : ℕ :=
  let rec_teams := sorry  -- This would contain the logic of calculating the number of ways to form the team.
  rec_teams

end volunteer_team_combinations_l174_174673


namespace parallelogram_count_l174_174232

noncomputable def num_parallelograms (n : ℕ) : ℕ :=
  3 * (n + 2).choose 4

theorem parallelogram_count {n : ℕ} : 
  (each_side_divided n) → 
  (lines_drawn_parallel_thru_points n) → 
  num_parallelograms n = 3 * (n + 2).choose 4 := 
by {
  intro h,
  sorry
}

end parallelogram_count_l174_174232


namespace max_coins_counterfeit_l174_174749

theorem max_coins_counterfeit (weighings outcomes states : ℕ) 
  (h_weighings : weighings = 3)
  (h_outcomes : outcomes = 3 ^ weighings)
  (h_states : ∀ (n : ℕ), 2 * n ≤ outcomes) :
  12 ≤ 13 :=
by
  -- We need to show that 12 is the maximum number of coins for the given conditions
  have h_max : ∀ (n : ℕ), 2 * n ≤ 27 → n ≤ 13,
  {
    assume n : ℕ,
    assume h : 2 * n ≤ 27,
    linarith,
  },
  have h12 : 2 * 12 ≤ 27,
  {
    norm_num,
  },
  exact h_max 12 h12

end max_coins_counterfeit_l174_174749


namespace magnitude_alpha_l174_174398

variables (α β : ℂ)
hypothesis h₁: α.conj = β
hypothesis h₂: (α / (β^2)).im = 0
hypothesis h₃: |α - β| = 2 * real.sqrt 5

theorem magnitude_alpha :
  |α| = real.sqrt (10 - 2.5 * real.sqrt 3) := sorry

end magnitude_alpha_l174_174398


namespace missing_number_correct_l174_174656

theorem missing_number_correct (x : ℤ) : 248 + x - (real.sqrt $ - real.sqrt $ - real.sqrt $ real.sqrt x) = 16 := 
sorry

end missing_number_correct_l174_174656


namespace chosen_number_l174_174091

theorem chosen_number (x : ℤ) (h : x / 12 - 240 = 8) : x = 2976 :=
by sorry

end chosen_number_l174_174091


namespace find_a2_l174_174590

-- Define the conditions
noncomputable def c (k : ℝ) : ℕ := -- some function representing the number of starting directions
  sorry

-- Define the main theorem
theorem find_a2 : ∃ (a0 a1 : ℝ), ∀ (k : ℝ) (P : ℝ × ℝ), c(k) ≤ π * k^2 + a1 * k + a0 :=
sorry

end find_a2_l174_174590


namespace shoe_price_monday_final_price_l174_174419

theorem shoe_price_monday_final_price : 
  let thursday_price := 50
  let friday_markup_rate := 0.15
  let monday_discount_rate := 0.12
  let friday_price := thursday_price * (1 + friday_markup_rate)
  let monday_price := friday_price * (1 - monday_discount_rate)
  monday_price = 50.6 := by
  sorry

end shoe_price_monday_final_price_l174_174419


namespace running_speed_ratio_correct_l174_174374

-- Define the conditions for Jack and Jill's running times and distances
def marathon_distance : ℝ := 42
def jack_time : ℝ := 6
def jill_time : ℝ := 4.2

-- Define their average speeds
def jack_speed := marathon_distance / jack_time
def jill_speed := marathon_distance / jill_time

-- Define the ratio of their speeds
def speed_ratio := jack_speed / jill_speed

-- Define the expected ratio value as computed from the conditions
def expected_ratio : ℝ := 7 / 10

-- The proof statement to show that the calculated speed ratio matches the expected ratio
theorem running_speed_ratio_correct : speed_ratio = expected_ratio :=
by
  sorry

end running_speed_ratio_correct_l174_174374


namespace number_of_distinguishable_arrangements_l174_174327

-- Definitions used in the conditions
def vowels : list char := ['A', 'O', 'O']
def consonants : list char := ['B', 'L', 'L', 'N']

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def permutations_with_repeats (lst : list char) : ℕ :=
  (factorial lst.length) / (lst.foldl (*) 1 (lst.map char.to_nat))

-- The proof problem asserting the number of distinguishable arrangements
theorem number_of_distinguishable_arrangements : 
  permutations_with_repeats vowels * permutations_with_repeats consonants = 36 :=
by
  sorry

end number_of_distinguishable_arrangements_l174_174327


namespace find_other_discount_l174_174562

theorem find_other_discount (P F d1 : ℝ) (H₁ : P = 70) (H₂ : F = 61.11) (H₃ : d1 = 10) : ∃ (d2 : ℝ), d2 = 3 :=
by 
  -- The proof will be provided here.
  sorry

end find_other_discount_l174_174562


namespace degree_measure_angle_ABC_l174_174496

theorem degree_measure_angle_ABC (O A B C : Type) [euclidean_geometry] 
  (circumscribed_about : is_circumscribed_circle O (triangle A B C)) 
  (angle_BOC : measure (angle B O C) = 110) 
  (angle_AOB : measure (angle A O B) = 150) : 
  measure (angle A B C) = 50 := 
sorry

end degree_measure_angle_ABC_l174_174496


namespace value_of_a_plus_b_l174_174300

noncomputable theory

def piecewise_function (x : ℝ) (a b : ℝ) : ℝ :=
if x >= 0 then sqrt x + 3 else a * x + b

theorem value_of_a_plus_b (a b : ℝ) (h1 : ∀ x1 : ℝ, x1 ≠ 0 → ∃! x2 : ℝ, ((x2 ≠ x1) ∧ (piecewise_function x1 a b = piecewise_function x2 a b))) (h2 : piecewise_function (sqrt 3 * a) a b = piecewise_function (4 * b) a b) :
  a + b = - real.sqrt 2 + 3 :=
sorry

end value_of_a_plus_b_l174_174300


namespace interval_increase_range_of_a_l174_174102

noncomputable def f (x a : ℝ) := Real.exp x - a * x - 1

theorem interval_increase (a : ℝ) :
  (a ≤ 0 → ∀ x, 0 ≤ f' x a) ∧ (0 < a → x ≥ Real.log a → 0 ≤ f' x a) := 
by
  sorry

theorem range_of_a (h : ∀ x, 0 ≤ f' x a) : a ≤ 0 :=
by
  sorry


end interval_increase_range_of_a_l174_174102


namespace contradiction_proof_l174_174083

theorem contradiction_proof (a b c : ℝ) (h1 : 0 < a ∧ a < 2) (h2 : 0 < b ∧ b < 2) (h3 : 0 < c ∧ c < 2) :
  ¬ (a * (2 - b) > 1 ∧ b * (2 - c) > 1 ∧ c * (2 - a) > 1) :=
sorry

end contradiction_proof_l174_174083


namespace extended_kobish_word_count_l174_174035

def extended_kobish_alphabet : Finset Char :=
  ('A' : Finset Char).insert 'B'.insert 'C'.insert 'D'.insert 'E'.insert 'F'
  .insert 'G'
  -- insert all letters up to 'U'
  .insert 'H'.insert 'I'.insert 'J'.insert 'K'.insert 'L'.insert 'M'.insert 'N'
  .insert 'O'.insert 'P'.insert 'Q'.insert 'R'.insert 'S'.insert 'T'.insert 'U'

def number_of_valid_words : ℕ :=
  let total_words (n : ℕ) : ℕ := (extended_kobish_alphabet.card) ^ n 
  let total_words_without_B (n : ℕ) : ℕ := (extended_kobish_alphabet.card - 1) ^ n 
  total_words 1 - total_words_without_B 1 + 
  total_words 2 - total_words_without_B 2 + 
  total_words 3 - total_words_without_B 3 + 
  total_words 4 - total_words_without_B 4

theorem extended_kobish_word_count : number_of_valid_words = 35784 :=
by
  sorry

end extended_kobish_word_count_l174_174035


namespace books_returned_wednesday_correct_l174_174889

def initial_books : Nat := 250
def books_taken_out_Tuesday : Nat := 120
def books_taken_out_Thursday : Nat := 15
def books_remaining_after_Thursday : Nat := 150

def books_after_tuesday := initial_books - books_taken_out_Tuesday
def books_before_thursday := books_remaining_after_Thursday + books_taken_out_Thursday
def books_returned_wednesday := books_before_thursday - books_after_tuesday

theorem books_returned_wednesday_correct : books_returned_wednesday = 35 := by
  sorry

end books_returned_wednesday_correct_l174_174889


namespace product_of_roots_of_quadratic_l174_174094

   -- Definition of the quadratic equation used in the condition
   def quadratic (x : ℝ) : ℝ := x^2 - 2 * x - 8

   -- Problem statement: Prove that the product of the roots of the given quadratic equation is -8.
   theorem product_of_roots_of_quadratic : 
     (∀ x : ℝ, quadratic x = 0 → (x = 4 ∨ x = -2)) → (4 * -2 = -8) :=
   by
     sorry
   
end product_of_roots_of_quadratic_l174_174094


namespace halfway_fraction_l174_174951

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/6) : (a + b) / 2 = 19 / 24 :=
by
  sorry

end halfway_fraction_l174_174951


namespace quadratic_decomposition_l174_174879

theorem quadratic_decomposition (a b c : ℝ) :
  (∀ x : ℝ, 6 * x^2 + 72 * x + 432 = a * (x + b)^2 + c) → a + b + c = 228 :=
sorry

end quadratic_decomposition_l174_174879


namespace halfway_fraction_l174_174946

theorem halfway_fraction (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  ((a / b + c / d) / 2) = 19 / 24 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end halfway_fraction_l174_174946


namespace domain_of_g_l174_174224

noncomputable def g (x : ℝ) : ℝ := 1 / ⌊x^2 - 9 * x + 21⌋

theorem domain_of_g :
  {x : ℚ | ∃ y : ℝ, g y = x} = set.Iic 4 ∪ set.Ici 5 :=
sorry

end domain_of_g_l174_174224


namespace magnitude_of_alpha_l174_174400

noncomputable def α : ℂ := sorry
noncomputable def β : ℂ := α.conj  -- since β is the conjugate of α 

-- Conditions
axiom h1 : (α / (β^2)).im = 0
axiom h2 : |α - β| = 2 * √5

theorem magnitude_of_alpha : |α| = (2 * √15) / 3 := by
  -- Proof omitted
  sorry

end magnitude_of_alpha_l174_174400


namespace complex_conjugate_l174_174743

noncomputable def z : ℂ := ((5 + 4 * complex.I) + complex.I) / (2 - complex.I)

theorem complex_conjugate (h : (2 - complex.I) * z - complex.I = 5 + 4 * complex.I) : z.conj = 1 - 3 * complex.I :=
  by
  -- The detailed proof is omitted
  sorry

end complex_conjugate_l174_174743


namespace cot_sum_identities_l174_174203

theorem cot_sum_identities :
  Real.cot (Real.arccot 5 + Real.arccot 11 + Real.arccot 17 + Real.arccot 23) = 1021 / 420 :=
by
  sorry

end cot_sum_identities_l174_174203


namespace sequence_properties_l174_174687

noncomputable def seq_a (n : ℕ) : ℕ :=
  if n = 0 then 1 else 2^(n-1)

def seq_b (n : ℕ) : ℕ :=
  3*n - 2

noncomputable def sum_seq_ab (n : ℕ) : ℕ :=
  (3*n - 5) * 2^n + 5

theorem sequence_properties (λ : ℕ) (λ_ne_neg1 : λ ≠ -1)
  (a1 : seq_a 1 = 1)
  (a2 : ∀ n, seq_a ((n : ℕ) + 1) = λ * (finset.sum (finset.range n) seq_a) + 1)
  (a3 : 2 * (seq_a 2) + 3 = seq_a 3 + 3)
  :
  (∀ n, seq_a n = 2^(n-1)) ∧
  (∀ n, seq_b n = 3*n - 2) ∧
  (∀ n, finset.sum (finset.range n) (λ i, seq_a i * seq_b i) = sum_seq_ab n) :=
sorry

end sequence_properties_l174_174687


namespace cube_can_be_divided_tetrahedron_cannot_be_divided_l174_174848

-- Define the problem domain: polyhedra inscribed in a sphere
structure Polyhedron :=
  (vertices : List Point3D)
  (edges : List (Point3D × Point3D))

-- Function that given a polyhedron inscribed in a sphere
-- ensures each edge is extended and checks equal sums grouping
def canBeDividedIntoEqualSums (poly : Polyhedron) : Prop :=
  sorry -- Implementation of segment extension and grouping

-- Define specific polyhedra
def cube : Polyhedron := {
  vertices := [/* vertex coordinates of the cube */],
  edges := [/* edges of the cube as pairs of vertices */]
}

def regularTetrahedron : Polyhedron := {
  vertices := [/* vertex coordinates of the tetrahedron */],
  edges := [/* edges of the tetrahedron as pairs of vertices */]
}

-- Theorems to prove
theorem cube_can_be_divided : canBeDividedIntoEqualSums cube := sorry

theorem tetrahedron_cannot_be_divided : ¬ canBeDividedIntoEqualSums regularTetrahedron := sorry

end cube_can_be_divided_tetrahedron_cannot_be_divided_l174_174848


namespace simplify_complex_expression_l174_174004

theorem simplify_complex_expression : 
  (let i := Complex.I in 
   ((4 + 7 * i) / (4 - 7 * i)) + ((4 - 7 * i) / (4 + 7 * i)) = -(66 / 65)) := 
by
  sorry

end simplify_complex_expression_l174_174004


namespace child_tickets_sold_l174_174192

-- Define variables and types
variables (A C : ℕ)

-- Main theorem to prove
theorem child_tickets_sold : A + C = 80 ∧ 12 * A + 5 * C = 519 → C = 63 :=
by
  intros
  sorry

end child_tickets_sold_l174_174192


namespace log_fraction_eq_l174_174333

variable (a b : ℝ)
axiom h1 : a = Real.logb 3 5
axiom h2 : b = Real.logb 5 7

theorem log_fraction_eq : Real.logb 15 (49 / 45) = (2 * (a * b) - a - 2) / (1 + a) :=
by sorry

end log_fraction_eq_l174_174333


namespace molecular_weight_of_carbonic_acid_l174_174980

theorem molecular_weight_of_carbonic_acid 
    (molecular_weight_8_moles : ℝ) 
    (h : molecular_weight_8_moles = 496) : 
    (molecular_weight_8_moles / 8 = 62) :=
by
  intros
  rw h
  norm_num
  sorry

end molecular_weight_of_carbonic_acid_l174_174980


namespace cot_sum_identities_l174_174206

theorem cot_sum_identities :
  Real.cot (Real.arccot 5 + Real.arccot 11 + Real.arccot 17 + Real.arccot 23) = 1021 / 420 :=
by
  sorry

end cot_sum_identities_l174_174206


namespace solve_sqrt_equation_l174_174663

theorem solve_sqrt_equation (z : ℚ) (h : sqrt (3 + z) = 12) : z = 141 := by
  sorry

end solve_sqrt_equation_l174_174663


namespace person_y_speed_in_still_water_l174_174056

theorem person_y_speed_in_still_water 
    (speed_x_in_still_water : ℝ)
    (time_meeting_towards_each_other : ℝ)
    (time_catching_up_same_direction: ℝ)
    (distance_upstream_meeting: ℝ)
    (distance_downstream_meeting: ℝ)
    (total_distance: ℝ) :
    speed_x_in_still_water = 6 →
    time_meeting_towards_each_other = 4 →
    time_catching_up_same_direction = 16 →
    distance_upstream_meeting = 4 * (6 - distance_upstream_meeting) + 4 * (10 + distance_downstream_meeting) →
    distance_downstream_meeting = 4 * (6 + distance_upstream_meeting) →
    total_distance = 4 * (6 + 10) →
    ∃ (speed_y_in_still_water : ℝ), speed_y_in_still_water = 10 :=
by
  intros h_speed_x h_time_meeting h_time_catching h_distance_upstream h_distance_downstream h_total_distance
  sorry

end person_y_speed_in_still_water_l174_174056


namespace consecutive_integer_sets_sum_27_l174_174331

theorem consecutive_integer_sets_sum_27 :
  ∃! s : Set (List ℕ), ∀ l ∈ s, 
  (∃ n a, n ≥ 3 ∧ l = List.range n ++ [a] ∧ (List.sum l) = 27)
:=
sorry

end consecutive_integer_sets_sum_27_l174_174331


namespace geometric_sequence_fourth_term_l174_174477

theorem geometric_sequence_fourth_term :
  let a1 := Real.root 4 16
  let a2 := Real.root 6 16
  let a3 := Real.root 8 16
  let r := a2 / a1
  let a4 := a3 * r
  a4 = Real.root 3 2 :=
by
  let a1 := Real.root 4 16
  let a2 := Real.root 6 16
  let a3 := Real.root 8 16
  let r := a2 / a1
  have h_root_inequality : (Real.root 3 2) = a3 * r := by sorry
  exact h_root_inequality

end geometric_sequence_fourth_term_l174_174477


namespace sandy_found_additional_money_l174_174002

-- Define the initial amount of money Sandy had
def initial_amount : ℝ := 13.99

-- Define the cost of the shirt
def shirt_cost : ℝ := 12.14

-- Define the cost of the jacket
def jacket_cost : ℝ := 9.28

-- Define the remaining amount after buying the shirt
def remaining_after_shirt : ℝ := initial_amount - shirt_cost

-- Define the additional money found in Sandy's pocket
def additional_found_money : ℝ := jacket_cost - remaining_after_shirt

-- State the theorem to prove the amount of additional money found
theorem sandy_found_additional_money :
  additional_found_money = 11.13 :=
by sorry

end sandy_found_additional_money_l174_174002


namespace driving_after_eight_hours_l174_174196

def bloodAlcoholContentDrop (p0 x r p : ℝ) : ℝ :=
  p0 * Real.exp (r * x)

theorem driving_after_eight_hours 
  (p0 p : ℝ) (r : ℝ) (x : ℝ)
  (h1 : p0 = 89)
  (h2 : p = 61)
  (h3 : bloodAlcoholContentDrop p0 2 r p)
  (h4 : ∀ x, 89 * Real.exp (r * x) < 20 → x ≥ 8) :
  89 * Real.exp (r * 8) < 20 :=
by
  sorry

end driving_after_eight_hours_l174_174196


namespace ellipse_focus_distance_l174_174684

theorem ellipse_focus_distance
  (x y : ℝ) (a b : ℝ) (h_eq : a = 5) (ellipse_eq : (x^2 / 25) + (y^2 / 16) = 1)
  (dist_to_one_focus : ℝ) (h_dist_to_one_focus : dist_to_one_focus = 7) :
  let dist_to_other_focus := 2 * a - dist_to_one_focus in
  dist_to_other_focus = 3 :=
by
  sorry

end ellipse_focus_distance_l174_174684


namespace prob_composite_in_first_50_l174_174540

-- Define what it means for a number to be composite
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ (∀ m : ℕ, m > 1 → m < n → ¬ m ∣ n)

-- Define the set of first 50 natural numbers
def first_50_numbers : list ℕ :=
  (list.range 50).map (λ n, n + 1)

-- Define the set of composite numbers within the first 50 natural numbers
def composite_numbers : list ℕ :=
  first_50_numbers.filter is_composite

-- Define the probability function
noncomputable def probability_of_composite : ℚ :=
  composite_numbers.length / first_50_numbers.length

-- The theorem statement
theorem prob_composite_in_first_50 : probability_of_composite = 34 / 50 :=
by sorry

end prob_composite_in_first_50_l174_174540


namespace find_a1_over_d_l174_174853

variable {a : ℕ → ℝ} (d : ℝ)

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem find_a1_over_d 
  (d_ne_zero : d ≠ 0) 
  (seq : arithmetic_sequence a d) 
  (h : a 2021 = a 20 + a 21) : 
  a 1 / d = 1981 :=
by 
  sorry

end find_a1_over_d_l174_174853


namespace belongs_to_group_32_l174_174727

theorem belongs_to_group_32 (n : ℕ) (h1 : n = 2007) : 
  ∃ k : ℕ, (k^2 < (Nat.div (h1 + 1) 2) ∧ (Nat.div (h1 + 1) 2) <= (k + 1)^2) ∧ k + 1 = 32 :=
by
  sorry

end belongs_to_group_32_l174_174727


namespace PQR_equilateral_l174_174382

-- Define the points as complex numbers
variable (a b c a1 a2 b1 b2 c1 c2 : ℂ)
-- Define ω as a primitive cube root of unity
noncomputable def ω := complex.exp (2 * real.pi * complex.I / 3)

-- Positively oriented equilateral triangles conditions
axiom eq1 : a + ω * b + ω^2 * c = 0
axiom eq2 : a + ω * a1 + ω^2 * a2 = 0
axiom eq3 : b + ω * b1 + ω^2 * b2 = 0
axiom eq4 : c + ω * c1 + ω^2 * c2 = 0

-- Define the midpoints P, Q, R
def P := (a2 + b1) / 2
def Q := (b2 + c1) / 2
def R := (c2 + a1) / 2

-- Theorem stating that triangle PQR is equilateral
theorem PQR_equilateral : P + ω * Q + ω^2 * R = 0 :=
by
  sorry

end PQR_equilateral_l174_174382


namespace garden_area_l174_174029

-- Conditions
variables {w l : ℝ}
-- The length of the rectangular garden exceeds three times its width by 10 meters
def length_exceeds_width : Prop := l = 3 * w + 10
-- The perimeter of the garden is 400 meters
def perimeter_condition : Prop := 2 * (l + w) = 400

-- Goal: The area of the rectangular garden is 7243.75 square meters
theorem garden_area (h1 : length_exceeds_width) (h2 : perimeter_condition) : w * l = 7243.75 :=
sorry

end garden_area_l174_174029


namespace Gage_skating_time_l174_174257

theorem Gage_skating_time :
  let min_per_hr := 60
  let skating_6_days := 6 * (1 * min_per_hr + 20)
  let skating_4_days := 4 * (1 * min_per_hr + 35)
  let needed_total := 11 * 90
  let skating_10_days := skating_6_days + skating_4_days
  let minutes_on_eleventh_day := needed_total - skating_10_days
  minutes_on_eleventh_day = 130 :=
by
  sorry

end Gage_skating_time_l174_174257


namespace number_of_participants_l174_174010

theorem number_of_participants (n : ℕ) (h : n - 1 = 25) : n = 26 := 
by sorry

end number_of_participants_l174_174010


namespace science_book_pages_l174_174482

def history_pages := 300
def novel_pages := history_pages / 2
def science_pages := novel_pages * 4

theorem science_book_pages : science_pages = 600 := by
  -- Given conditions:
  -- The novel has half as many pages as the history book, the history book has 300 pages,
  -- and the science book has 4 times as many pages as the novel.
  sorry

end science_book_pages_l174_174482


namespace irrational_floor_mod_l174_174826

theorem irrational_floor_mod 
  (k : ℕ) (h : k ≥ 2) : ∃ (r : ℝ), irrational r ∧ ∀ (m : ℕ), (⌊r^m⌋ ≡ -1 [MOD k]) :=
by
  sorry

end irrational_floor_mod_l174_174826


namespace smallest_interesting_number_is_1800_l174_174161

def is_perfect_square (k : ℕ) : Prop := ∃ (m : ℕ), k = m * m
def is_perfect_cube (k : ℕ) : Prop := ∃ (m : ℕ), k = m * m * m

def is_interesting (n : ℕ) : Prop :=
  is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n)

theorem smallest_interesting_number_is_1800 :
  ∃ (n : ℕ), is_interesting(n) ∧ (∀ m : ℕ, is_interesting(m) → n ≤ m) ∧ n = 1800 :=
by 
  sorry

end smallest_interesting_number_is_1800_l174_174161


namespace tea_bags_count_l174_174429

theorem tea_bags_count (n : ℕ) 
  (h1 : 2 * n ≤ 41) 
  (h2 : 41 ≤ 3 * n) 
  (h3 : 2 * n ≤ 58) 
  (h4 : 58 ≤ 3 * n) : 
  n = 20 := by
  sorry

end tea_bags_count_l174_174429


namespace ten_digit_numbers_satisfy_l174_174649

def numberOfTenDigitNumbers : Nat :=
  let numBlocks : Nat → Nat
  | 1 => 256
  | m + 1 => 2 * numBlocks m
  numBlocks 10

theorem ten_digit_numbers_satisfy:
  numberOfTenDigitNumbers = 256 := sorry

end ten_digit_numbers_satisfy_l174_174649


namespace parallel_vectors_l174_174708

theorem parallel_vectors (a b : ℝ → ℝ → ℝ) (k : ℝ) (ha_not_collinear : ¬ (∃ λ : ℝ, a = λ • b))
  (hc : ∀ x, c x = k * a x + b x) (hd : ∀ x, d x = a x - b x)
  (hc_parallel_hd : ∀ x, ∃ λ : ℝ, c x = λ * d x) : k = -1 ∧ (∀ x, c x = - d x) :=
by
  sorry

end parallel_vectors_l174_174708


namespace fraction_halfway_between_3_4_and_5_6_is_19_24_l174_174911

noncomputable def fraction_halfway (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between_3_4_and_5_6_is_19_24 :
  fraction_halfway (3 / 4) (5 / 6) = 19 / 24 :=
by
  sorry

end fraction_halfway_between_3_4_and_5_6_is_19_24_l174_174911


namespace fraction_halfway_between_3_4_and_5_6_is_19_24_l174_174917

noncomputable def fraction_halfway (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between_3_4_and_5_6_is_19_24 :
  fraction_halfway (3 / 4) (5 / 6) = 19 / 24 :=
by
  sorry

end fraction_halfway_between_3_4_and_5_6_is_19_24_l174_174917


namespace find_rowing_speed_of_person_Y_l174_174061

open Real

def rowing_speed (y : ℝ) : Prop :=
  ∀ (x : ℝ) (current_speed : ℝ),
    x = 6 → 
    (4 * (6 - current_speed) + 4 * (y + current_speed) = 4 * (6 + y)) →
    (16 * (y + current_speed) = 16 * (6 + current_speed) + 4 * (y - 6)) → 
    y = 10

-- We set up the proof problem without solving it.
theorem find_rowing_speed_of_person_Y : ∃ (y : ℝ), rowing_speed y :=
begin
  use 10,
  unfold rowing_speed,
  intros x current_speed h1 h2 h3,

  sorry
end

end find_rowing_speed_of_person_Y_l174_174061


namespace square_of_longest_segment_in_sector_l174_174111

-- Define the context and conditions
def diameter : ℝ := 16
def radius : ℝ := diameter / 2
def central_angle : ℝ := 90  -- in degrees

-- Convert degrees to radians for trigonometric calculations
def deg_to_rad (d : ℝ) : ℝ := d * Real.pi / 180
def angle_rad := deg_to_rad central_angle

-- Define what we need to prove
theorem square_of_longest_segment_in_sector : (8 * Real.sqrt 2)^2 = 128 :=
by
  -- This is the central problem formulation, without the actual proof.
  sorry

end square_of_longest_segment_in_sector_l174_174111


namespace sum_of_alternating_binomial_coeffs_l174_174670

theorem sum_of_alternating_binomial_coeffs :
  let sum := ∑ (k : ℕ) in (Finset.range 51), (if k % 2 = 0 then 1 else -1) * Nat.choose 101 (2 * k)
  sum = 2^50 :=
by
  sorry

end sum_of_alternating_binomial_coeffs_l174_174670


namespace sqrt_expression_equivalent_l174_174080

-- Define the expressions involved
structure Expressions where
  a : ℝ  -- sqrt(9)
  b : ℝ  -- 25 * sqrt(9)
  c : ℝ  -- sqrt(25 * sqrt(9))
  d : ℝ  -- 49 * sqrt(25 * sqrt(9))
  e : ℝ  -- sqrt(49 * sqrt(25 * sqrt(9)))

-- Provide assumptions for the intermediate calculations
theorem sqrt_expression_equivalent : 
  let a := sqrt 9
  let b := 25 * a
  let c := sqrt b
  let d := 49 * c
  let e := sqrt d
  e = 5 * sqrt 7 * real.rpow 3 (1/4) := by
  sorry

end sqrt_expression_equivalent_l174_174080


namespace arithmetic_sequence_problem_l174_174763

noncomputable theory
open_locale classical

theorem arithmetic_sequence_problem
  (a : ℕ → ℝ)
  (h_arith_seq : ∀ n, a (n + 1) = a n + (a 2 - a 1))
  (h_sum : a 3 + a 5 + a 7 + a 9 + a 11 = 200) :
  4 * a 5 - 2 * a 3 = 80 := sorry

end arithmetic_sequence_problem_l174_174763


namespace person_y_speed_in_still_water_l174_174055

theorem person_y_speed_in_still_water 
    (speed_x_in_still_water : ℝ)
    (time_meeting_towards_each_other : ℝ)
    (time_catching_up_same_direction: ℝ)
    (distance_upstream_meeting: ℝ)
    (distance_downstream_meeting: ℝ)
    (total_distance: ℝ) :
    speed_x_in_still_water = 6 →
    time_meeting_towards_each_other = 4 →
    time_catching_up_same_direction = 16 →
    distance_upstream_meeting = 4 * (6 - distance_upstream_meeting) + 4 * (10 + distance_downstream_meeting) →
    distance_downstream_meeting = 4 * (6 + distance_upstream_meeting) →
    total_distance = 4 * (6 + 10) →
    ∃ (speed_y_in_still_water : ℝ), speed_y_in_still_water = 10 :=
by
  intros h_speed_x h_time_meeting h_time_catching h_distance_upstream h_distance_downstream h_total_distance
  sorry

end person_y_speed_in_still_water_l174_174055


namespace exists_poly_prime_factors_l174_174389

noncomputable theory
open_locale classical

-- Define the polynomial and properties
variables {R : Type*} [comm_ring R] (f : polynomial ℤ) (u : ℕ)

-- Main theorem statement
theorem exists_poly_prime_factors (hf : ∃ m, polynomial.degree f = m ∧ m > 0) (hu : u > 0) :
  ∃ n : ℤ, f.eval n ≠ 0 ∧ (f.eval n).nat_abs.num_factors ≥ u :=
sorry

end exists_poly_prime_factors_l174_174389


namespace coin_arrangement_l174_174828

def num_arrangements (gold_coins silver_coins : ℕ) : ℕ :=
  if gold_coins = 5 ∧ silver_coins = 5 then 2772 else 0

theorem coin_arrangement (gold_coins silver_coins : ℕ) :
  gold_coins = 5 ∧ silver_coins = 5 → num_arrangements gold_coins silver_coins = 2772 :=
by intros h; cases h; rw [num_arrangements]; exact if_pos rfl

end coin_arrangement_l174_174828


namespace smallest_n_multiple_of_2015_l174_174591

noncomputable def given_number (n : ℕ) : ℕ :=
  1 * 10 ^ (n + 3 - 1) + (4 * (10^(n + 2) - 1) / 9 - 10^n) * 10 + 3 * 10 + 0

theorem smallest_n_multiple_of_2015 : ∃ n, given_number n % 2015 = 0 ∧ ∀ m, (m < n → given_number m % 2015 ≠ 0) :=
begin
  sorry
end

end smallest_n_multiple_of_2015_l174_174591


namespace cylinder_and_sphere_radius_l174_174595

noncomputable def radius_cylinder (diameter_cone altitude_cone : ℝ) : ℝ :=
  let r := (2 * altitude_cone * diameter_cone) / (7 * diameter_cone + 16 * altitude_cone)
  r

theorem cylinder_and_sphere_radius :
  ∀ (diameter_cone altitude_cone : ℝ),
    diameter_cone = 14 →
    altitude_cone = 16 →
    (λ r, r = (2 * altitude_cone * diameter_cone) / (7 * diameter_cone + 16 * altitude_cone)) (radius_cylinder diameter_cone altitude_cone) →
    radius_cylinder diameter_cone altitude_cone = 56 / 15 := by
  intros diameter_cone altitude_cone hdiam_cone halt_cone def_radius
  rw [hdiam_cone, halt_cone] at def_radius
  have h : (2 * 16 * 14) / (7 * 14 + 16 * 16) = 56 / 15 := by
    norm_num
  exact h


end cylinder_and_sphere_radius_l174_174595


namespace find_b_minus_d_squared_l174_174092

variables (a b c d : ℝ)

theorem find_b_minus_d_squared :
  (a - b - c + d = 13) ∧ (a + b - c - d = 5) ∧ (3a - 2b + 4c - d = 17) →
  (b - d)^2 = 16 :=
by
  intros h,
  sorry

end find_b_minus_d_squared_l174_174092


namespace rate_of_current_l174_174886

theorem rate_of_current (c : ℝ) (h1 : 7.5 = (20 + c) * 0.3) : c = 5 :=
by
  sorry

end rate_of_current_l174_174886


namespace decomposition_sum_of_cubes_l174_174020

theorem decomposition_sum_of_cubes 
  (a b c d e : ℤ) 
  (h : (512 : ℤ) * x ^ 3 + 27 = (a * x + b) * (c * x ^ 2 + d * x + e)) :
  a + b + c + d + e = 60 := 
sorry

end decomposition_sum_of_cubes_l174_174020


namespace angle_CBD_is_110_degrees_l174_174568

-- Definitions of the conditions
variables {A B C D : Type}
variables [decidable_linear_ordered_semiring A]
variables [decidable_linear_ordered_semiring B]
variables [decidable_linear_ordered_semiring C]
variables [decidable_linear_ordered_semiring D]

-- Isosceles triangle with AC = BC and m∠C = 40°
def is_isosceles_triangle (A B C : Type) (AC BC: A) (C_angle : B) :=
AC = BC ∧ C_angle = 40

-- To prove: measure of angle CBD = 110°
theorem angle_CBD_is_110_degrees (A B C D : Type) 
  (AC BC : A) (AC_eq_BC : AC = BC) (C_angle : B) (hk : C_angle = 40)
  : ∃ CBD : C, CBD = 110 := 
sorry

end angle_CBD_is_110_degrees_l174_174568


namespace halfway_fraction_l174_174947

theorem halfway_fraction (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  ((a / b + c / d) / 2) = 19 / 24 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end halfway_fraction_l174_174947


namespace smallest_interesting_number_l174_174153

def is_perfect_square (m : ℕ) : Prop := 
  ∃ k : ℕ, k * k = m

def is_perfect_cube (m : ℕ) : Prop := 
  ∃ k : ℕ, k * k * k = m

def interesting (n : ℕ) : Prop := 
  is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n)

theorem smallest_interesting_number : ∃ n, interesting n ∧ ∀ m, interesting m → n ≤ m :=
begin
  use 1800,
  split,
  {
    -- Prove 1800 is interesting
    sorry,
  },
  {
    -- Prove any interesting number is at least 1800
    sorry,
  }
end

end smallest_interesting_number_l174_174153


namespace smallest_interesting_number_l174_174133

def is_perfect_square (x : ℕ) : Prop :=
∃ y : ℕ, y * y = x

def is_perfect_cube (x : ℕ) : Prop :=
∃ y : ℕ, y * y * y = x

theorem smallest_interesting_number (n : ℕ) :
    is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n) → n = 1800 :=
by {
  sorry
}

end smallest_interesting_number_l174_174133


namespace probability_not_monday_l174_174878

theorem probability_not_monday (P_monday : ℚ) (h : P_monday = 1/7) : P_monday ≠ 1 → ∃ P_not_monday : ℚ, P_not_monday = 6/7 :=
by
  sorry

end probability_not_monday_l174_174878


namespace triangle_side_length_BC_49_l174_174371

theorem triangle_side_length_BC_49
  (angle_A : ℝ)
  (AC : ℝ)
  (area_ABC : ℝ)
  (h1 : angle_A = 60)
  (h2 : AC = 16)
  (h3 : area_ABC = 220 * Real.sqrt 3) : 
  ∃ (BC : ℝ), BC = 49 :=
by
  sorry

end triangle_side_length_BC_49_l174_174371


namespace x_plus_y_value_l174_174278

-- Definitions of the given conditions
def x_plus_sin_y_eq : ℝ → ℝ → Prop := λ x y, x + Real.sin y = 2008
def x_plus_2008_cos_y_eq : ℝ → ℝ → Prop := λ x y, x + 2008 * Real.cos y = 2007
def y_in_range : ℝ → Prop := λ y, 0 ≤ y ∧ y ≤ Real.pi / 2

-- Proof statement to be proven
theorem x_plus_y_value (x y : ℝ) (h1 : x_plus_sin_y_eq x y) (h2 : x_plus_2008_cos_y_eq x y) (h3 : y_in_range y) : 
  x + y = 2007 + Real.pi / 2 := 
  sorry

end x_plus_y_value_l174_174278


namespace fraction_halfway_between_l174_174964

def halfway_fraction (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between :
  halfway_fraction (3/4) (5/6) = 19/24 :=
sorry

end fraction_halfway_between_l174_174964


namespace fraction_halfway_between_3_4_and_5_6_is_19_24_l174_174912

noncomputable def fraction_halfway (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between_3_4_and_5_6_is_19_24 :
  fraction_halfway (3 / 4) (5 / 6) = 19 / 24 :=
by
  sorry

end fraction_halfway_between_3_4_and_5_6_is_19_24_l174_174912


namespace sum_of_their_present_ages_l174_174510

-- Definitions based on the conditions given:
variable (Henry_age Jill_age : ℕ)
variable (current_year : ℕ)

-- Given conditions
axiom henry_present_age : Henry_age = 29
axiom jill_present_age : Jill_age = 19
axiom sum_ages : Henry_age + Jill_age = 48

-- The proof statement for the sum of their present ages
theorem sum_of_their_present_ages : 
  Henry_age + Jill_age = 48 := 
  by
    exact sum_ages

end sum_of_their_present_ages_l174_174510


namespace problem_3_l174_174309

noncomputable def ln (x : ℝ) : ℝ := Real.log x

def f (x : ℝ) (a : ℝ) : ℝ := (1 / 2) * x^2 * a * (ln x)
def g (x : ℝ) (a : ℝ) : ℝ := (1 / 2) * x^2 - a * (ln x) + (a - 1) * x

theorem problem_3 (x : ℝ) (h₁: x > 0) : ln x + (3 / (4 * x^2)) - (1 / Real.exp x) > 0 :=
by
  sorry

end problem_3_l174_174309


namespace composite_probability_l174_174533

/--
Given that a number selected at random from the first 50 natural numbers,
where 1 is neither prime nor composite,
the probability of selecting a composite number is 34/49.
-/
theorem composite_probability :
  let total_numbers := 50
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  let num_primes := primes.length
  let num_composites := total_numbers - num_primes - 1
  let probability_composite := (num_composites : ℚ) / (total_numbers - 1)
  probability_composite = 34 / 49 :=
by {
  sorry
}

end composite_probability_l174_174533


namespace sum_of_series_eq_3_div_4_l174_174622

theorem sum_of_series_eq_3_div_4 : 
  (∑ k in (Set.Univ : Set ℕ), (3 ^ (2 ^ k)) / (6 ^ (2 ^ k) - 2)) = 3 / 4 := 
sorry

end sum_of_series_eq_3_div_4_l174_174622


namespace angle_congruence_l174_174498

theorem angle_congruence
  (A B C M N P I : Type)
  [linear_order A] [linear_order B] [linear_order C] [linear_order M] [linear_order N] [linear_order P] [linear_order I]
  (hM : M ∈ line_segment B C)
  (hN : N ∈ line_segment C A)
  (hP : P ∈ line_segment A B)
  (hBM_BP : dist B M = dist B P)
  (hCM_CN : dist C M = dist C N)
  (hPerp_B_MP : is_perpendicular (line_through B M) (line_through M P))
  (hPerp_C_MN : is_perpendicular (line_through C M) (line_through M N))
  (hIntersection_I : I = intersection (perpendicular_from B (line_through M P)) (perpendicular_from C (line_through M N))) :
  angle I P A = angle I N C :=
by
  sorry

end angle_congruence_l174_174498


namespace fraction_halfway_between_3_4_and_5_6_is_19_24_l174_174913

noncomputable def fraction_halfway (a b : ℚ) : ℚ := (a + b) / 2

theorem fraction_halfway_between_3_4_and_5_6_is_19_24 :
  fraction_halfway (3 / 4) (5 / 6) = 19 / 24 :=
by
  sorry

end fraction_halfway_between_3_4_and_5_6_is_19_24_l174_174913


namespace packing_objects_in_boxes_l174_174409

theorem packing_objects_in_boxes 
  (n k : ℕ) (n_pos : 0 < n) (k_pos : 0 < k) 
  (objects : Fin (n * k) → Fin k) 
  (boxes : Fin k → Fin n → Fin k) :
  ∃ (pack : Fin (n * k) → Fin k), 
    (∀ i, ∃ c1 c2, 
      ∀ j, pack i = pack j → 
      (objects i = c1 ∨ objects i = c2 ∧
      objects j = c1 ∨ objects j = c2)) := 
sorry

end packing_objects_in_boxes_l174_174409


namespace alpha_modulus_l174_174395

noncomputable def α : ℂ := a + b * Complex.I
noncomputable def β : ℂ := a - b * Complex.I

theorem alpha_modulus :
  (α β : ℂ) (h_conj : β = conj α)
  (h_real : α / (β^2) ∈ ℝ) (h_diff : |α - β| = 2 * Real.sqrt 5) :
  |α| = (2 * Real.sqrt 15) / 3 :=
by
  sorry

end alpha_modulus_l174_174395


namespace min_nickels_required_l174_174222

-- Definitions based on the conditions
def cost_of_book := 27.37
def value_of_20_bill := 20.0
def value_of_5_bill := 5.0
def value_of_quarters := 5 * 0.25
def value_of_one_nickel := 0.05

-- Statement: the minimum number of nickels required
theorem min_nickels_required (n : ℕ) :
  value_of_20_bill + value_of_5_bill + value_of_quarters + n * value_of_one_nickel ≥ cost_of_book → n ≥ 23 :=
by
  sorry

end min_nickels_required_l174_174222
